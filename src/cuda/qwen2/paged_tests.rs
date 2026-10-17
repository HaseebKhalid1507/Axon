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

/// Suffix of the entry points with a table (stage 0: test wrappers).
const PG: &str = "_pg";

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
    let b = tab.map(|t| g.upload(t).unwrap());
    let a = b.as_ref().map_or(Arg::Ptr(0), |b| b.arg());
    (b, a)
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
            let fd = km.function(&format!("attn_blk_{q8n}{hd}{PG}")).unwrap();
            let mut ffs: Vec<(batch::Fa, Func)> = Vec::new();
            for (v, n) in [
                (batch::Fa::Full, "attn_fa_b"),
                (batch::Fa::Small, "attn_fa_bs"),
                (batch::Fa::Kv, "attn_fa_k"),
            ] {
                if kg > 0 && v == batch::Fa::Kv {
                    continue;
                }
                let f = km.function(&format!("{n}_{q8n}{hd}{PG}")).unwrap();
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
