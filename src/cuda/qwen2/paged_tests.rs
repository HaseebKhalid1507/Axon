//! Paged KV at the kernel level (specs/r11-paged-kv.md §3.1.1): a kernel
//! reading and writing the cache through a page table of 64-position pages
//! (decode.cu pg_row) gives the bits it gives on contiguous rows. Every case
//! runs on the plain layout (null table), the identity table, and random
//! tables into a pool with spare pages, the data physically scattered
//! accordingly and every unused row 0xff (NaN in f16 and f32): outputs
//! must be bit-equal, and the cache gathered back through the table must
//! equal the plain run's (the kernels that store rows).
use super::*;
use crate::cuda::Kernels;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * scale).collect()
    }
}

const PAGE: usize = 64;

/// One cache array: `rb` bytes per position, positions in order.
struct Arr {
    data: Vec<u8>,
    rb: usize,
}

fn bytes<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    // SAFETY: plain numbers, read as the bytes they are.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) }.to_vec()
}

/// The f16 cache of `rows` positions (np of them data, the rest 0xff):
/// K, V.
fn f16_cache(rng: &mut Rng, rows: usize, np: usize, kvd: usize) -> Vec<Arr> {
    let mut one = |scale: f32| {
        let mut h: Vec<u16> = rng
            .vec(np * kvd, scale)
            .into_iter()
            .map(quant::f32_to_f16)
            .collect();
        h.resize(rows * kvd, 0xffff);
        Arr {
            data: bytes(&h),
            rb: kvd * 2,
        }
    };
    vec![one(4.0), one(1.0)]
}

/// The q8 cache (group kg): K, V int8, their scales (NaN past np).
fn q8_cache(rng: &mut Rng, rows: usize, np: usize, kvd: usize, kg: usize) -> Vec<Arr> {
    let mut out = Vec::new();
    for _ in 0..2 {
        let v: Vec<u8> = (0..rows * kvd).map(|_| (rng.next() % 255) as u8).collect();
        out.push(Arr { data: v, rb: kvd });
    }
    for scale in [4.0f32, 1.0] {
        let mut d: Vec<f32> = (0..np * kvd / kg)
            .map(|_| (rng.f() + 1.5) * scale / 127.0)
            .collect();
        d.resize(rows * kvd / kg, f32::NAN);
        out.push(Arr {
            data: bytes(&d),
            rb: kvd / kg * 4,
        });
    }
    out
}

/// The arrays in a pool of `phys` pages, page b at tab[b] (None: in place).
fn place<'g>(g: &'g Gpu, arrs: &[Arr], tab: Option<&[u32]>, phys: usize) -> Vec<DevBuf<'g>> {
    arrs.iter()
        .map(|a| {
            let pb = PAGE * a.rb;
            let mut out = vec![0xffu8; phys * pb];
            for (b, src) in a.data.chunks(pb).enumerate() {
                let p = tab.map_or(b, |t| t[b] as usize);
                out[p * pb..p * pb + src.len()].copy_from_slice(src);
            }
            g.upload(&out).unwrap()
        })
        .collect()
}

/// The first `pages` pages of an array, back in position order.
fn gather(phys: &[u8], rb: usize, tab: Option<&[u32]>, pages: usize) -> Vec<u8> {
    let pb = PAGE * rb;
    (0..pages)
        .flat_map(|b| {
            let p = tab.map_or(b, |t| t[b] as usize);
            phys[p * pb..(p + 1) * pb].iter().copied()
        })
        .collect()
}

/// What a case's kernels produced, by name (raw bytes).
type Outs = Vec<(String, Vec<u8>)>;
/// A case's kernels on the cache buffers through a table (None: plain).
type Run<'a> = dyn FnMut(&[DevBuf], Option<&[u32]>) -> Outs + 'a;

/// Run `run` (the case's kernels, given the cache buffers and the table,
/// None = plain) on the plain layout, the identity table and `nperm` random
/// tables; every output and the gathered cache must equal the plain run's.
fn check_paged(g: &Gpu, what: &str, arrs: &[Arr], nperm: usize, rng: &mut Rng, run: &mut Run) {
    let pages = arrs[0].data.len() / arrs[0].rb / PAGE;
    for a in arrs {
        assert_eq!(a.data.len(), pages * PAGE * a.rb, "{what}: whole pages");
    }
    let mut layouts: Vec<(String, Option<Vec<u32>>, usize)> = vec![
        ("plain".into(), None, pages),
        ("identity".into(), Some((0..pages as u32).collect()), pages),
    ];
    for k in 0..nperm {
        // A random injection of the pages into a pool with spares.
        let phys = pages + pages / 2 + 3;
        let mut pool: Vec<u32> = (0..phys as u32).collect();
        for i in (1..phys).rev() {
            pool.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        pool.truncate(pages);
        layouts.push((format!("shuffled {k}"), Some(pool), phys));
    }
    let mut want: Option<(Outs, Vec<Vec<u8>>)> = None;
    for (name, tab, phys) in &layouts {
        let tab = tab.as_deref();
        let bufs = place(g, arrs, tab, *phys);
        let outs = run(&bufs, tab);
        let caches: Vec<Vec<u8>> = bufs
            .iter()
            .zip(arrs)
            .map(|(b, a)| gather(&b.read::<u8>().unwrap(), a.rb, tab, pages))
            .collect();
        match &want {
            None => want = Some((outs, caches)),
            Some((wo, wc)) => {
                assert_eq!(wo.len(), outs.len());
                for ((wn, w), (n, o)) in wo.iter().zip(&outs) {
                    assert_eq!(wn, n);
                    if w != o {
                        let i = w.iter().zip(o).position(|(a, b)| a != b);
                        panic!("{what}: {n} differs on the {name} table (first byte {i:?})");
                    }
                }
                for (i, (w, c)) in wc.iter().zip(&caches).enumerate() {
                    if w != c {
                        let at = w.iter().zip(c).position(|(a, b)| a != b).unwrap();
                        let row = at / arrs[i].rb;
                        panic!("{what}: cache array {i} differs on the {name} table at row {row}");
                    }
                }
            }
        }
    }
}

fn tab_arg<'g>(g: &'g Gpu, tab: Option<&[u32]>) -> (Option<DevBuf<'g>>, Arg) {
    tab_at(g, tab, 0)
}

/// A pool's table with this sequence's at entry `off` (entries before it:
/// page 0, which no row of the sequence is on unless tab says so).
fn tab_at<'g>(g: &'g Gpu, tab: Option<&[u32]>, off: usize) -> (Option<DevBuf<'g>>, Arg) {
    let b = tab.map(|t| g.upload(&[vec![0u32; off], t.to_vec()].concat()).unwrap());
    let a = b.as_ref().map_or(Arg::Ptr(0), |b| b.arg());
    (b, a)
}

/// `Place::Rows` for nb tokens of one sequence at positions start..: rp's
/// (position, row) pairs, or with a table (position, the sequence's first
/// entry `off`).
fn rows_rp<'g>(g: &'g Gpu, start: usize, nb: usize, paged: bool, off: usize) -> DevBuf<'g> {
    let rp: Vec<u32> = (0..nb)
        .flat_map(|z| {
            let pos = (start + z) as u32;
            [pos, if paged { off as u32 } else { pos }]
        })
        .collect();
    g.upload(&rp).unwrap()
}

/// (sin, cos) pairs for `n` rows of hd / 2 frequencies (any values do).
fn rope_rows(rng: &mut Rng, n: usize, hd: usize) -> Vec<f32> {
    (0..n * hd / 2)
        .flat_map(|_| {
            let a = rng.f() * 3.0;
            [a.sin(), a.cos()]
        })
        .collect()
}

/// Kernel shapes (n_heads, n_kv, hd): 7B, a 64-wide head (group 2), Qwen3-8B.
const SHAPES: [(usize, usize, usize); 3] = [(28, 4, 128), (16, 8, 64), (32, 8, 128)];
const NPS: [usize; 7] = [1, 63, 64, 65, 1000, 4097, 32768];

/// Random tables per case: fewer at 32k (each layout uploads the cache).
fn nperm(np: usize) -> usize {
    if np > 8192 { 3 } else { 20 }
}

/// The decode attention (attn_blk, f16 and q8, with and without qwen3's
/// q/k norms; q8: this token's store at np - 1 too) and the prefill
/// attention (attn_fa_b, attn_fa_bs, attn_fa_k on f16; the q8 cache's
/// attn_fa_b / attn_fa_bs) through a page table.
#[test]
fn gpu_paged_attention_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    if !km.mma() {
        return;
    }
    let mut rng = Rng(6464);
    for (nh, nkv, hd) in SHAPES {
        let (dim, kvd, group) = (nh * hd, nkv * hd, nh / nkv);
        for kg in [0, hd, 32] {
            let q8n = match kg {
                0 => String::new(),
                _ if kg == hd => "q8_".into(),
                _ => "q8g32_".into(),
            };
            let fd = km.function(&format!("attn_blk_{q8n}{hd}")).unwrap();
            let mut ffs: Vec<(batch::Fa, Func)> = Vec::new();
            for (v, n) in [
                (batch::Fa::Full, "attn_fa_b"),
                (batch::Fa::Small, "attn_fa_bs"),
                (batch::Fa::Kv, "attn_fa_k"),
            ] {
                if kg > 0 && v == batch::Fa::Kv {
                    continue;
                }
                let f = km.function(&format!("{n}_{q8n}{hd}")).unwrap();
                f.set_max_shared(batch::fa_shape(hd, v, kg).2).unwrap();
                ffs.push((v, f));
            }
            let pre = Arg::F32(att_pre(hd));
            for np in NPS {
                let rows = np.next_multiple_of(PAGE);
                let pages = rows / PAGE;
                let arrs = match kg {
                    0 => f16_cache(&mut rng, rows, np, kvd),
                    _ => q8_cache(&mut rng, rows, np, kvd, kg),
                };
                let nb = if np > 8192 { 512 } else { np.min(77) };
                let start = np - nb;
                let q = rng.vec(nb * dim, 1.0);
                let kf = rng.vec(2 * kvd, 2.0);
                let kraw = rng.vec(kvd, 2.0);
                let nw = rng.vec(2 * hd, 1.0);
                let rope: Vec<f32> = (0..hd / 2)
                    .flat_map(|i| {
                        let a = (np as f32) / 10000f32.powf(2.0 * i as f32 / hd as f32);
                        [a.sin(), a.cos()]
                    })
                    .collect();
                let gy = pages.min(23);
                let what = format!("{nh}/{nkv}x{hd} kg {kg} np {np}");
                check_paged(&g, &what, &arrs, nperm(np), &mut rng, &mut |bufs, tab| {
                    let (_tb, ta) = tab_arg(&g, tab);
                    let kva: Vec<Arg> = bufs.iter().map(|b| b.arg()).collect();
                    let mut outs = Outs::new();
                    let params = g.upload(&[0u32, (np - 1) as u32, 0, 0]).unwrap();
                    let qd = g.upload(&q[(nb - 1) * dim..]).unwrap();
                    let kfd = g.upload(&kf).unwrap();
                    let krd = g.upload(&kraw).unwrap();
                    let nwd = g.upload(&nw).unwrap();
                    let rpd = g.upload(&rope).unwrap();
                    for norms in [false, true] {
                        let po = g.alloc(nh * pages * hd * 4).unwrap();
                        let pm = g.alloc(nh * pages * 2 * 4).unwrap();
                        po.zero().unwrap();
                        pm.zero().unwrap();
                        let (kr, qn, kn) = match norms {
                            false => (Arg::Ptr(0), Arg::Ptr(0), Arg::Ptr(0)),
                            true => (krd.arg(), nwd.arg(), nwd.arg_at(hd * 4)),
                        };
                        let mid = [
                            params.arg(),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            pre,
                            Arg::I32(pages as i32),
                            po.arg(),
                            pm.arg(),
                        ];
                        let args: Vec<Arg> = match kg {
                            // kraw, qn, kn, eps, rope, kc_out (= K).
                            0 => [
                                &[qd.arg()][..],
                                &kva,
                                &mid,
                                &[kr, qn, kn, Arg::F32(1e-6), rpd.arg(), kva[0], ta],
                            ]
                            .concat(),
                            // kf (stored at np - 1), ksh, qn, kn, eps, rope.
                            _ => [
                                &[qd.arg()][..],
                                &kva,
                                &[kfd.arg()],
                                &mid,
                                &[Arg::Ptr(0), qn, kn, Arg::F32(1e-6), rpd.arg(), ta],
                            ]
                            .concat(),
                        };
                        // SAFETY: decode.cu's signature; part_* of `pages`
                        // blocks, the cache `pages` pages through tab.
                        unsafe {
                            g.launch(
                                &fd,
                                (nkv as u32, gy as u32, 1),
                                (32 * att_nw(hd, kg > 0), 1, 1),
                                0,
                                &args,
                            )
                            .unwrap();
                        }
                        outs.push((format!("attn_blk norms {norms} part_o"), po.read().unwrap()));
                        outs.push((
                            format!("attn_blk norms {norms} part_ml"),
                            pm.read().unwrap(),
                        ));
                    }
                    for (v, ff) in &ffs {
                        let (fa_rows, fa_thr, fa_smem) = batch::fa_shape(hd, *v, kg);
                        let qb = g.upload(&q).unwrap();
                        let y = g.alloc(nb * dim * 4).unwrap();
                        y.zero().unwrap();
                        let args = [
                            &[qb.arg()][..],
                            &kva,
                            &[
                                Arg::I32(start as i32),
                                Arg::I32(nb as i32),
                                Arg::I32(nh as i32),
                                Arg::I32(nkv as i32),
                                pre,
                                y.arg(),
                                ta,
                            ],
                        ]
                        .concat();
                        // SAFETY: prefill.cu's signature; q, y nb rows.
                        unsafe {
                            g.launch(
                                ff,
                                (nkv as u32, (nb * group).div_ceil(fa_rows) as u32, 1),
                                (fa_thr as u32, 1, 1),
                                fa_smem as u32,
                                &args,
                            )
                            .unwrap();
                        }
                        outs.push((format!("attn_fa {v:?} y"), y.read().unwrap()));
                    }
                    outs
                });
            }
        }
    }
}

/// The kernel names of a cache format: "" (f16), "q8_" (group hd), "q8g32_".
fn q8_name(kg: usize, hd: usize) -> &'static str {
    match kg {
        0 => "",
        _ if kg == hd => "q8_",
        _ => "q8g32_",
    }
}

/// The split attention (pre-sm_80 GPUs; tensor cores or not, run here
/// directly): the decode kernels attn_split (f16, with and without qwen3's
/// norms: their K store at np - 1) and attn_split_q8 (this token's store),
/// and per query the batched attn_split_b and attn_blk_b (sm_80+), one
/// sequence (rp null, its table) and `Place::Rows` (rp: the sequence's
/// first entry into a pool's table), through a page table.
#[test]
fn gpu_paged_split_and_batched_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mma = km.mma();
    let mut rng = Rng(6465);
    for (nh, nkv, hd) in SHAPES {
        let (dim, kvd) = (nh * hd, nkv * hd);
        for kg in [0, hd, 32] {
            let q8n = q8_name(kg, hd);
            let fs = km.function(&format!("attn_split_{q8n}{hd}")).unwrap();
            let fsb = km.function(&format!("attn_split_b_{q8n}{hd}")).unwrap();
            let fbb = km.function(&format!("attn_blk_b_{q8n}{hd}")).unwrap();
            let scale = Arg::F32(1.0 / (hd as f32).sqrt());
            let pre = Arg::F32(att_pre(hd));
            for np in NPS {
                let rows = np.next_multiple_of(PAGE);
                let pages = rows / PAGE;
                let arrs = match kg {
                    0 => f16_cache(&mut rng, rows, np, kvd),
                    _ => q8_cache(&mut rng, rows, np, kvd, kg),
                };
                // Queries of the batched kernels: the last nz positions.
                let nz = np.min(9);
                let start = np - nz;
                let ms = if np > 4097 { 184 } else { 23 };
                let q = rng.vec(nz * dim, 1.0);
                let kf = rng.vec(2 * kvd, 2.0);
                let kraw = rng.vec(kvd, 2.0);
                let nw = rng.vec(2 * hd, 1.0);
                let rope = rope_rows(&mut rng, 1, hd);
                let what = format!("{nh}/{nkv}x{hd} kg {kg} np {np}");
                // Batched kernels' table offset in Rows mode.
                let off = 1 + np % 5;
                check_paged(&g, &what, &arrs, nperm(np), &mut rng, &mut |bufs, tab| {
                    let (_tb, ta) = tab_arg(&g, tab);
                    let kva: Vec<Arg> = bufs.iter().map(|b| b.arg()).collect();
                    let mut outs = Outs::new();
                    let params = g.upload(&[0u32, (np - 1) as u32, 0, 0]).unwrap();
                    let qd = g.upload(&q[(nz - 1) * dim..]).unwrap();
                    let kfd = g.upload(&kf).unwrap();
                    let krd = g.upload(&kraw).unwrap();
                    let nwd = g.upload(&nw).unwrap();
                    let rpd = g.upload(&rope).unwrap();
                    // Batched first: the decode kernels store row np - 1.
                    let qb = g.upload(&q).unwrap();
                    for rows_mode in [false, true] {
                        let (_tr, tra) = match rows_mode {
                            false => tab_arg(&g, tab),
                            true => tab_at(&g, tab, off),
                        };
                        let rp = rows_rp(&g, start, nz, tab.is_some(), off);
                        let (st, rpa) = match rows_mode {
                            false => (start, Arg::Ptr(0)),
                            true => (0, rp.arg()),
                        };
                        let kernels: &[(&str, &Func, usize, u32, Arg)] = &[
                            ("attn_split_b", &fsb, ms, ATT_TILE as u32, scale),
                            ("attn_blk_b", &fbb, pages, 32, pre),
                        ];
                        for &(name, f, stride, thr, sc) in kernels {
                            if name == "attn_blk_b" && !mma {
                                continue;
                            }
                            let po = g.alloc(nz * nh * stride * hd * 4).unwrap();
                            let pm = g.alloc(nz * nh * stride * 2 * 4).unwrap();
                            po.zero().unwrap();
                            pm.zero().unwrap();
                            let args = [
                                &[qb.arg()][..],
                                &kva,
                                &[
                                    Arg::I32(st as i32),
                                    Arg::I32(0),
                                    Arg::I32(nh as i32),
                                    Arg::I32(nkv as i32),
                                    sc,
                                    Arg::I32(stride as i32),
                                    po.arg(),
                                    pm.arg(),
                                    rpa,
                                    tra,
                                ],
                            ]
                            .concat();
                            let gy = if thr == 32 { pages.min(23) } else { ms };
                            // SAFETY: prefill.cu's signatures; part_* of nz
                            // queries, rp nz pairs, the table as described.
                            unsafe {
                                g.launch(
                                    f,
                                    (nkv as u32, gy as u32, nz as u32),
                                    (thr, 1, 1),
                                    0,
                                    &args,
                                )
                                .unwrap();
                            }
                            let tag = format!("{name} rows {rows_mode}");
                            outs.push((format!("{tag} part_o"), po.read().unwrap()));
                            outs.push((format!("{tag} part_ml"), pm.read().unwrap()));
                        }
                    }
                    for norms in [false, true] {
                        let po = g.alloc(nh * ms * hd * 4).unwrap();
                        let pm = g.alloc(nh * ms * 2 * 4).unwrap();
                        po.zero().unwrap();
                        pm.zero().unwrap();
                        let (kr, qn, kn) = match norms {
                            false => (Arg::Ptr(0), Arg::Ptr(0), Arg::Ptr(0)),
                            true => (krd.arg(), nwd.arg(), nwd.arg_at(hd * 4)),
                        };
                        let mid = [
                            params.arg(),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            scale,
                            Arg::I32(ms as i32),
                            po.arg(),
                            pm.arg(),
                        ];
                        let args: Vec<Arg> = match kg {
                            0 => [
                                &[qd.arg()][..],
                                &kva,
                                &mid,
                                &[kr, qn, kn, Arg::F32(1e-6), rpd.arg(), kva[0], ta],
                            ]
                            .concat(),
                            _ => [
                                &[qd.arg()][..],
                                &kva,
                                &[kfd.arg()],
                                &mid,
                                &[Arg::Ptr(0), qn, kn, Arg::F32(1e-6), rpd.arg(), ta],
                            ]
                            .concat(),
                        };
                        // SAFETY: decode.cu's signature; part_* of ms splits.
                        unsafe {
                            g.launch(
                                &fs,
                                (nkv as u32, ms as u32, 1),
                                (ATT_TILE as u32, 1, 1),
                                0,
                                &args,
                            )
                            .unwrap();
                        }
                        outs.push((
                            format!("attn_split norms {norms} part_o"),
                            po.read().unwrap(),
                        ));
                        outs.push((
                            format!("attn_split norms {norms} part_ml"),
                            pm.read().unwrap(),
                        ));
                    }
                    outs
                });
            }
        }
    }
}

/// The batched cache stores (prefill and the multi-sequence step): rope_kv_b
/// (f16 cache, or queries only), rope_kv_norm_b (qwen3), kv_q8_b (both
/// groups, with qwen3's key norm), one sequence and `Place::Rows`, tokens
/// straddling page boundaries: q and the cache gathered back through the
/// table are the plain run's.
#[test]
fn gpu_paged_store_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let (frk, frn) = (
        km.function("rope_kv_b").unwrap(),
        km.function("rope_kv_norm_b").unwrap(),
    );
    let mut rng = Rng(6466);
    for (nh, nkv, hd) in SHAPES {
        let (nq, nk) = (nh * hd, nkv * hd);
        for kg in [0, hd, 32] {
            let fq8 = (kg > 0).then(|| {
                let n = if kg == hd { "kv_q8_b" } else { "kv_q8g32_b" };
                km.function(&format!("{n}_{hd}")).unwrap()
            });
            for (start, nb) in [(0usize, 1usize), (60, 9), (64, 64), (1000, 130), (4090, 7)] {
                let rows = (start + nb).next_multiple_of(PAGE) + PAGE;
                let arrs = match kg {
                    0 => f16_cache(&mut rng, rows, start, nk),
                    _ => q8_cache(&mut rng, rows, start, nk, kg),
                };
                let qkv = rng.vec(nb * (nq + 2 * nk), 3.0);
                let rope = rope_rows(&mut rng, nb, hd);
                let nw = rng.vec(2 * hd, 1.0);
                let off = 3;
                let what = format!("{nh}/{nkv}x{hd} kg {kg} start {start} nb {nb}");
                check_paged(&g, &what, &arrs, 6, &mut rng, &mut |bufs, tab| {
                    let kva: Vec<Arg> = bufs.iter().map(|b| b.arg()).collect();
                    let mut outs = Outs::new();
                    let qkvb = g.upload(&qkv).unwrap();
                    let ropeb = g.upload(&rope).unwrap();
                    let nwd = g.upload(&nw).unwrap();
                    for rows_mode in [false, true] {
                        let (_tr, tra) = match rows_mode {
                            false => tab_arg(&g, tab),
                            true => tab_at(&g, tab, off),
                        };
                        let rp = rows_rp(&g, start, nb, tab.is_some(), off);
                        let (st, rpa) = match rows_mode {
                            false => (start, Arg::Ptr(0)),
                            true => (0, rp.arg()),
                        };
                        for norms in [false, true] {
                            let q = g.alloc(nb * nq * 4).unwrap();
                            q.zero().unwrap();
                            let (kc, vc) = match kg {
                                0 => (kva[0], kva[1]),
                                _ => (Arg::Ptr(0), Arg::Ptr(0)),
                            };
                            let (f, grid, mut args) = match norms {
                                false => {
                                    let items = if kg > 0 { nq / 2 } else { nq / 2 + nk / 2 + nk };
                                    (
                                        &frk,
                                        (items.div_ceil(128) as u32, nb as u32, 1),
                                        vec![
                                            qkvb.arg(),
                                            Arg::I32(nq as i32),
                                            Arg::I32(nk as i32),
                                            Arg::I32(hd as i32),
                                            Arg::I32(st as i32),
                                            ropeb.arg(),
                                            q.arg(),
                                        ],
                                    )
                                }
                                true => (
                                    &frn,
                                    (((nq + 2 * nk) / hd).div_ceil(4) as u32, nb as u32, 1),
                                    vec![
                                        qkvb.arg(),
                                        Arg::I32(nq as i32),
                                        Arg::I32(nk as i32),
                                        Arg::I32(hd as i32),
                                        Arg::I32(st as i32),
                                        ropeb.arg(),
                                        nwd.arg(),
                                        nwd.arg_at(hd * 4),
                                        Arg::F32(1e-6),
                                        q.arg(),
                                    ],
                                ),
                            };
                            args.extend_from_slice(&[kc, vc, rpa, tra]);
                            // SAFETY: prefill.cu's signatures; qkv nb rows,
                            // rope nb x hd / 2 pairs, the cache through tab.
                            unsafe {
                                g.launch(f, grid, (128, 1, 1), 0, &args).unwrap();
                                if let Some(f8) = &fq8 {
                                    let kn = if norms {
                                        nwd.arg_at(hd * 4)
                                    } else {
                                        Arg::Ptr(0)
                                    };
                                    let a8 = [
                                        &[
                                            qkvb.arg(),
                                            Arg::I32(nq as i32),
                                            Arg::I32(nk as i32),
                                            Arg::I32(st as i32),
                                            ropeb.arg(),
                                        ][..],
                                        &kva,
                                        &[Arg::Ptr(0), kn, Arg::F32(1e-6), rpa, tra],
                                    ]
                                    .concat();
                                    g.launch(f8, (nkv as u32, nb as u32, 1), (128, 1, 1), 0, &a8)
                                        .unwrap();
                                }
                            }
                            let tag = format!("rows {rows_mode} norms {norms}");
                            outs.push((format!("{tag} q"), q.read().unwrap()));
                        }
                    }
                    outs
                });
            }
        }
    }
}

/// gemma3's head-256 path on every GPU: the decode attn_split_w_256 (with
/// its q/k norms and K store) and the batched attn_split_bw_256 (one
/// sequence and `Place::Rows`), sliding windows whose first position
/// (win_lo, now an integer offset before the table) falls inside, on and
/// across page boundaries, window lengths that are no multiple of 64 or
/// 128; and rope_kv_norm_b_256's stores.
#[test]
fn gpu_paged_gemma_window_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let (fw, fbw, frn) = (
        km.function("attn_split_w_256").unwrap(),
        km.function("attn_split_bw_256").unwrap(),
        km.function("rope_kv_norm_b_256").unwrap(),
    );
    let hd = 256;
    let mut rng = Rng(6467);
    // gemma3 1B (4 heads, 1 kv head) and 4B (8, 4).
    for (nh, nkv) in [(4usize, 1usize), (8, 4)] {
        let (dim, kvd) = (nh * hd, nkv * hd);
        let scale = Arg::F32(1.0 / (hd as f32).sqrt());
        for (np, win) in [
            (1usize, 512usize),
            (500, 512),
            (513, 512),
            (577, 512),
            (600, 512),
            (1020, 512),
            (1100, 512),
            (1100, 0),
            (1000, 100),
            (4097, 1024),
            (4097, 333),
        ] {
            let rows = np.next_multiple_of(PAGE);
            let arrs = f16_cache(&mut rng, rows, np, kvd);
            let nz = np.min(9);
            let start = np - nz;
            let ms = 23;
            let q = rng.vec(nz * dim, 1.0);
            let kraw = rng.vec(kvd, 2.0);
            let nw = rng.vec(2 * hd, 1.0);
            let rope = rope_rows(&mut rng, nz, hd);
            let qkv = rng.vec(nz * (dim + 2 * kvd), 3.0);
            let off = 2;
            let what = format!("gemma {nh}/{nkv}x256 np {np} win {win}");
            check_paged(&g, &what, &arrs, nperm(np), &mut rng, &mut |bufs, tab| {
                let (_tb, ta) = tab_arg(&g, tab);
                let (kc, vc) = (bufs[0].arg(), bufs[1].arg());
                let mut outs = Outs::new();
                let qb = g.upload(&q).unwrap();
                let krd = g.upload(&kraw).unwrap();
                let nwd = g.upload(&nw).unwrap();
                let rpd = g.upload(&rope).unwrap();
                for rows_mode in [false, true] {
                    let (_tr, tra) = match rows_mode {
                        false => tab_arg(&g, tab),
                        true => tab_at(&g, tab, off),
                    };
                    let rp = rows_rp(&g, start, nz, tab.is_some(), off);
                    let (st, rpa) = match rows_mode {
                        false => (start, Arg::Ptr(0)),
                        true => (0, rp.arg()),
                    };
                    let po = g.alloc(nz * nh * ms * hd * 4).unwrap();
                    let pm = g.alloc(nz * nh * ms * 2 * 4).unwrap();
                    po.zero().unwrap();
                    pm.zero().unwrap();
                    let args = [
                        qb.arg(),
                        kc,
                        vc,
                        Arg::I32(st as i32),
                        Arg::I32(0),
                        Arg::I32(nh as i32),
                        Arg::I32(nkv as i32),
                        scale,
                        Arg::I32(ms as i32),
                        po.arg(),
                        pm.arg(),
                        rpa,
                        Arg::I32(win as i32),
                        tra,
                    ];
                    // SAFETY: gemma.cu's signature; part_* of nz queries.
                    unsafe {
                        g.launch(
                            &fbw,
                            (nkv as u32, ms as u32, nz as u32),
                            (ATT_TILE as u32, 1, 1),
                            0,
                            &args,
                        )
                        .unwrap();
                    }
                    outs.push((
                        format!("attn_split_bw rows {rows_mode} part_o"),
                        po.read().unwrap(),
                    ));
                    outs.push((
                        format!("attn_split_bw rows {rows_mode} part_ml"),
                        pm.read().unwrap(),
                    ));
                }
                // The decode kernel, with norms (stores K at np - 1) and without.
                let params = g.upload(&[0u32, (np - 1) as u32, 0, 0]).unwrap();
                let qd = g.upload(&q[(nz - 1) * dim..]).unwrap();
                for norms in [false, true] {
                    let po = g.alloc(nh * ms * hd * 4).unwrap();
                    let pm = g.alloc(nh * ms * 2 * 4).unwrap();
                    po.zero().unwrap();
                    pm.zero().unwrap();
                    let (kr, qn, kn, ko) = match norms {
                        false => (Arg::Ptr(0), Arg::Ptr(0), Arg::Ptr(0), Arg::Ptr(0)),
                        true => (krd.arg(), nwd.arg(), nwd.arg_at(hd * 4), kc),
                    };
                    let args = [
                        qd.arg(),
                        kc,
                        vc,
                        params.arg(),
                        Arg::I32(nh as i32),
                        Arg::I32(nkv as i32),
                        scale,
                        Arg::I32(ms as i32),
                        po.arg(),
                        pm.arg(),
                        kr,
                        qn,
                        kn,
                        Arg::F32(1e-6),
                        rpd.arg(),
                        ko,
                        Arg::I32(win as i32),
                        ta,
                    ];
                    // SAFETY: gemma.cu's signature; part_* of ms splits.
                    unsafe {
                        g.launch(
                            &fw,
                            (nkv as u32, ms as u32, 1),
                            (ATT_TILE as u32, 1, 1),
                            0,
                            &args,
                        )
                        .unwrap();
                    }
                    outs.push((
                        format!("attn_split_w norms {norms} part_o"),
                        po.read().unwrap(),
                    ));
                    outs.push((
                        format!("attn_split_w norms {norms} part_ml"),
                        pm.read().unwrap(),
                    ));
                }
                // rope_kv_norm_b_256 storing the last nz positions again
                // (different values), one sequence then Rows.
                let qkvb = g.upload(&qkv).unwrap();
                for rows_mode in [false, true] {
                    let (_tr, tra) = match rows_mode {
                        false => tab_arg(&g, tab),
                        true => tab_at(&g, tab, off),
                    };
                    let rp = rows_rp(&g, start, nz, tab.is_some(), off);
                    let (st, rpa) = match rows_mode {
                        false => (start, Arg::Ptr(0)),
                        true => (0, rp.arg()),
                    };
                    let qo = g.alloc(nz * dim * 4).unwrap();
                    let args = [
                        qkvb.arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(st as i32),
                        rpd.arg(),
                        nwd.arg(),
                        nwd.arg_at(hd * 4),
                        Arg::F32(1e-6),
                        qo.arg(),
                        kc,
                        vc,
                        rpa,
                        tra,
                    ];
                    let heads = (dim + 2 * kvd) / hd;
                    // SAFETY: gemma.cu's signature; qkv nz rows, rope nz rows.
                    unsafe {
                        g.launch(
                            &frn,
                            (heads.div_ceil(4) as u32, nz as u32, 1),
                            (128, 1, 1),
                            0,
                            &args,
                        )
                        .unwrap();
                    }
                    outs.push((
                        format!("rope_kv_norm_b_256 rows {rows_mode} q"),
                        qo.read().unwrap(),
                    ));
                }
                outs
            });
        }
    }
}

/// Each layer's cache arrays (bytes per position) in `layer_kv`'s order.
fn layer_arrays(net: &CudaQwen2, c: &CudaCache) -> Vec<usize> {
    let kvd = net.kv_dim();
    match c.mode {
        CacheKv::F16 => vec![2 * kvd, 2 * kvd],
        CacheKv::Q8 { group } => vec![kvd, kvd, kvd / group * 4, kvd / group * 4],
    }
}

/// Positions 0..n of every layer's arrays, through `tab` (None: plain).
fn kv_bytes(net: &CudaQwen2, c: &CudaCache, tab: Option<&[u32]>, n: usize) -> Vec<Vec<u8>> {
    let rbs = layer_arrays(net, c);
    pipeline::layer_kv(net, c)
        .iter()
        .flat_map(|l| {
            let mut at = 0;
            rbs.iter()
                .map(|&rb| {
                    let a = &l[at..at + c.max_ctx * rb];
                    at += c.max_ctx * rb;
                    let mut v = gather(a, rb, tab, n.div_ceil(PAGE));
                    v.truncate(n * rb);
                    v
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// gpu_paged_cache_matches_plain's script on one cache: its picks.
fn paged_script<'g>(net: &CudaQwen2<'g>, c: &mut CudaCache<'g>, prompt: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut at = 0;
    for n in [600, 1, 37, 64, 300, 97] {
        net.feed_batch(&prompt[at..at + n], at, c);
        at += n;
    }
    let mut tok = prompt[at];
    for _ in 0..40 {
        tok = net.forward_argmax(tok, at, c);
        out.push(tok);
        at += 1;
    }
    // A verify pass with guesses, then truncate and re-feed.
    let guess: Vec<u32> = (0..5).map(|i| out[i]).collect();
    out.extend(net.verify_argmax(&guess, at, c));
    let back = at - 70;
    net.feed_batch(&prompt[back - 20..back], back - 20, c);
    out.push(net.forward_argmax(prompt[back], back, c));
    out
}

/// The host side (stage 1): a cache read and written through a shuffled
/// page table (`CudaCache::set_tab`, rows physically scattered over a
/// larger cache) runs the same script as a plain cache - prefill in chunks
/// of several sizes (the flash, per-query and split attention), graphed
/// decode steps, a verify pass, a re-feed after a truncation - with the same
/// picks, and the same KV bytes per position gathered back through the
/// table. f16 and q8, tensor cores on and off, Qwen2.5 0.5B, Qwen3 0.6B
/// (q/k norms), Llama 3.2 1B, Gemma 3 1B (head 256, the 512 window across
/// pages).
/// One test per model, so a small host can run the 0.5B alone.
#[test]
fn gpu_paged_cache_matches_plain_qwen25() {
    paged_cache_matches_plain("qwen2.5", crate::testutil::model());
}

#[test]
fn gpu_paged_cache_matches_plain_qwen3() {
    paged_cache_matches_plain("qwen3", crate::testutil::qwen3_model());
}

#[test]
fn gpu_paged_cache_matches_plain_llama() {
    paged_cache_matches_plain("llama", crate::testutil::llama_model());
}

#[test]
fn gpu_paged_cache_matches_plain_gemma3() {
    paged_cache_matches_plain("gemma3", crate::testutil::gemma3_model());
}

fn paged_cache_matches_plain(name: &str, m: Option<Model>) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = m else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(6468);
    let ctx = 1280;
    let pages = ctx / PAGE;
    {
        let m = &m;
        for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
            let Ok(mut net) = CudaQwen2::with_kv(&g, &km, m, ctx, kv, false) else {
                continue; // gemma3: no q8 cache
            };
            let tcs: &[bool] = match (km.mma(), net.head_dim) {
                (true, 64 | 128) => &[true, false],
                _ => &[false],
            };
            for &tc in tcs {
                net.set_tc_attn(tc);
                let v = net.vocab as u64;
                let prompt: Vec<u32> = (0..1100).map(|_| (rng.next() % v) as u32).collect();
                let phys = pages + pages / 2 + 3;
                let mut pool: Vec<u32> = (0..phys as u32).collect();
                for i in (1..phys).rev() {
                    pool.swap(i, (rng.next() % (i as u64 + 1)) as usize);
                }
                pool.truncate(pages);
                let what = format!("{name} {kv:?} tc {tc}");
                let mut c0 = net.try_cache_kv(ctx, kv).unwrap();
                let mut c1 = net.try_cache_kv(phys * PAGE, kv).unwrap();
                c1.set_tab(Some(g.upload(&pool).unwrap()));
                let p0 = paged_script(&net, &mut c0, &prompt);
                let p1 = paged_script(&net, &mut c1, &prompt);
                assert_eq!(p0, p1, "{what}: picks");
                let n = 1100 + 40 + 5;
                let (k0, k1) = (
                    kv_bytes(&net, &c0, None, n),
                    kv_bytes(&net, &c1, Some(&pool), n),
                );
                for (i, (a, b)) in k0.iter().zip(&k1).enumerate() {
                    if a != b {
                        let at = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
                        panic!("{what}: layer array {i} differs at byte {at}");
                    }
                }
            }
        }
    }
}
