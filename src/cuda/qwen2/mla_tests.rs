//! mla.cu's kernels on their own: the attention against an f64 reference
//! and per-token bits (one token vs a batch), the per-head products
//! against the CPU's dequantization, the router rule against the CPU's.

use super::*;
use crate::cuda::Kernels;

struct Rng(u64);
impl Rng {
    /// Uniform in [-1, 1).
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// mla_attn + mla_attn_cmb of nb tokens at positions start.. over cache
/// `kv` (f16 bits).
#[allow(clippy::too_many_arguments)]
fn attn_gpu(
    g: &Gpu,
    ml: &Module,
    qa: &[f32],
    kv: &DevBuf,
    krow: usize,
    lat: usize,
    n_h: usize,
    scale: f32,
    start: usize,
) -> Vec<f32> {
    let (fa, fc) = (
        ml.function("mla_attn").unwrap(),
        ml.function("mla_attn_cmb").unwrap(),
    );
    let nb = qa.len() / (n_h * krow);
    let ns = (start + nb).div_ceil(MLA_SPLIT);
    let q = g.upload(qa).unwrap();
    let part = g.alloc(nb * ns * n_h * (lat + 2) * 4).unwrap();
    let out = g.alloc(nb * n_h * lat * 4).unwrap();
    let hpb = n_h.min(8);
    let busy = g.info().unwrap().sms as i32;
    let pos = [
        Arg::Ptr(0),
        Arg::I32(start as i32),
        Arg::I32(MLA_SPLIT as i32),
        Arg::I32(ns as i32),
    ];
    // SAFETY: the kernels' signatures; the cache holds start + nb rows,
    // part nb x ns splits.
    unsafe {
        let mut a = vec![
            q.arg(),
            kv.arg(),
            Arg::I32(krow as i32),
            Arg::I32(lat as i32),
            Arg::I32(n_h as i32),
            Arg::I32(hpb as i32),
            Arg::F32(scale),
        ];
        a.extend(pos);
        a.push(part.arg());
        a.push(Arg::I32(busy));
        a.push(Arg::I32(0));
        g.launch(
            &fa,
            (
                attn_blocks(n_h, hpb, ns, nb, busy as usize) as u32,
                nb as u32,
                1,
            ),
            (256, 1, 1),
            (MLA_TILE * krow * 2) as u32,
            &a,
        )
        .unwrap();
        let mut a = vec![part.arg(), Arg::I32(lat as i32), Arg::I32(n_h as i32)];
        a.extend(pos);
        a.push(out.arg());
        g.launch(&fc, (n_h as u32, nb as u32, 1), (256, 1, 1), 0, &a)
            .unwrap();
    }
    out.read().unwrap()
}

/// The attention (its splits of MLA_SPLIT rows and their combine) on
/// Moonlight's shapes (16 heads, row 576, latent 512), K2's (64 heads)
/// and the fixture's (2, 80, 64), up to 3 splits: within 1e-5 of an f64
/// softmax over the same f16 rows (|K| < 2, |q| < 1; an f32 numpy
/// emulation of the same online softmax is off by 3.3e-7 at Moonlight's
/// shape), and each token's output the same bits alone (decode) as in a
/// batch.
#[test]
fn mla_attn_matches_reference_and_batch() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let ml = Kernels::mla_module(&g).unwrap();
    for (n_h, krow, lat, np) in [(16, 576, 512, 300), (2, 80, 64, 70), (64, 576, 512, 140)] {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15 ^ krow as u64);
        let k16: Vec<u16> = (0..np * krow)
            .map(|_| crate::quant::f32_to_f16(r.f() * 2.0))
            .collect();
        let kf: Vec<f32> = k16.iter().map(|&h| crate::quant::f16_to_f32(h)).collect();
        let kv = g.upload(&k16).unwrap();
        let qa: Vec<f32> = (0..np * n_h * krow).map(|_| r.f()).collect();
        let scale = 1.0 / (192f32).sqrt();
        let all = attn_gpu(&g, &ml, &qa, &kv, krow, lat, n_h, scale, 0);
        let w = n_h * krow;
        for p in [0, 1, 31, 32, 33, 127, 128, 129, np / 2, np - 1]
            .into_iter()
            .filter(|&p| p < np)
        {
            let one = attn_gpu(
                &g,
                &ml,
                &qa[p * w..(p + 1) * w],
                &kv,
                krow,
                lat,
                n_h,
                scale,
                p,
            );
            let o = n_h * lat;
            assert_eq!(
                bits(&one),
                bits(&all[p * o..(p + 1) * o]),
                "({n_h}, {krow}) position {p}: alone vs batch"
            );
            for h in 0..n_h {
                let q = &qa[p * w + h * krow..p * w + (h + 1) * krow];
                let s: Vec<f64> = (0..=p)
                    .map(|t| {
                        let kr = &kf[t * krow..(t + 1) * krow];
                        q.iter()
                            .zip(kr)
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for c in 0..lat {
                    let want: f64 =
                        (0..=p).map(|t| e[t] * kf[t * krow + c] as f64).sum::<f64>() / z;
                    let got = one[h * lat + c] as f64;
                    assert!(
                        (got - want).abs() <= 1e-5,
                        "({n_h}, {krow}) pos {p} head {h} col {c}: {got} vs {want}"
                    );
                }
            }
        }
    }
}

/// `n` values of type qt as random bytes, the blocks' f16 scales set to
/// small finite values (any other byte pattern is a valid block).
fn random_blocks(r: &mut Rng, qt: u32, n: usize) -> Vec<u8> {
    use crate::quant::{Q4_K, Q5_0, Q8_0};
    let (bs, bb, scales): (usize, usize, &[usize]) = match qt {
        Q8_0 => (32, 34, &[0]),
        Q5_0 => (32, 22, &[0]),
        Q4_K => (256, 144, &[0, 2]),
        crate::quant::Q6_K => (256, 210, &[208]),
        crate::quant::Q4_0 => (32, 18, &[0]),
        crate::quant::F16 => {
            let v: Vec<u8> = (0..n)
                .flat_map(|_| crate::quant::f32_to_f16(r.f()).to_le_bytes())
                .collect();
            return v;
        }
        crate::quant::F32 => return (0..n).flat_map(|_| r.f().to_le_bytes()).collect(),
        _ => unimplemented!("random_blocks: type {qt}"),
    };
    let mut b: Vec<u8> = (0..n / bs * bb)
        .map(|_| (r.f() * 128.0 + 128.0) as u8)
        .collect();
    for blk in b.chunks_exact_mut(bb) {
        for &o in scales {
            let h = crate::quant::f32_to_f16(0.01 + 0.02 * r.f().abs());
            blk[o..o + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
    b
}

/// mla_heads_mv against the CPU's dequantization (f64 sums), for Q8_0
/// (K2's k_b), Q5_0 (Moonlight's k_b) and Q4_K (Moonlight's v_b), and one
/// token's bits alone == in a batch of 11.
#[test]
fn mla_heads_mv_matches_dequant() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let ml = Kernels::mla_module(&g).unwrap();
    let f = ml.function("mla_heads_mv").unwrap();
    use crate::quant::{Q4_K, Q5_0, Q8_0};
    for (qt, rph, cols, n_h) in [(Q8_0, 64, 128, 4), (Q5_0, 64, 128, 3), (Q4_K, 32, 512, 2)] {
        let mut r = Rng(17 + qt as u64);
        let rows = rph * n_h;
        let bytes = random_blocks(&mut r, qt, rows * cols);
        let mut deq = vec![0f32; rows * cols];
        crate::quant::dequant(qt, &bytes, &mut deq);
        let rb = crate::quant::row_bytes(qt, cols);
        let w = g.upload(&bytes).unwrap();
        let nb = 11;
        // Inputs x[z][h * xs + c], xs > cols (strided like q's nope parts).
        let xs = cols + 32;
        let x: Vec<f32> = (0..nb * n_h * xs).map(|_| r.f()).collect();
        let xd = g.upload(&x).unwrap();
        let run = |z0: usize, n: usize| -> Vec<f32> {
            let out = g.alloc(n * rows * 4).unwrap();
            let (rpb, smem) = mv_plan((qt, rb), rph, cols, n_h, n);
            // SAFETY: mla_heads_mv's signature; buffers sized for n tokens.
            unsafe {
                g.launch(
                    &f,
                    (rows.div_ceil(rpb) as u32, n.div_ceil(MLA_MV_T) as u32, 1),
                    (256, 1, 1),
                    smem as u32,
                    &[
                        Arg::I32(qt as i32),
                        w.arg(),
                        Arg::U64(rb as u64),
                        Arg::I32(rph as i32),
                        Arg::I32(cols as i32),
                        Arg::I32(n_h as i32),
                        xd.arg_at(z0 * n_h * xs * 4),
                        Arg::I32(xs as i32),
                        Arg::U64((n_h * xs) as u64),
                        out.arg(),
                        Arg::I32(rph as i32),
                        Arg::U64(rows as u64),
                        Arg::I32(n as i32),
                        Arg::I32(rpb as i32),
                    ],
                )
                .unwrap();
            }
            out.read().unwrap()
        };
        let all = run(0, nb);
        for z in [0, 5, 10] {
            assert_eq!(
                bits(&run(z, 1)),
                bits(&all[z * rows..(z + 1) * rows]),
                "type {qt} token {z}: alone vs batch"
            );
            for gr in 0..rows {
                let h = gr / rph;
                let want: f64 = (0..cols)
                    .map(|c| deq[gr * cols + c] as f64 * x[z * n_h * xs + h * xs + c] as f64)
                    .sum();
                let got = all[z * rows + gr] as f64;
                assert!(
                    (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "type {qt} token {z} row {gr}: {got} vs {want}"
                );
            }
        }
    }
}

/// One launch of mla_heads_mv (`rf`: main's mla_heads_mv_ref, else the
/// rewrite with rpb rows per block) over n tokens of x from token z0.
#[allow(clippy::too_many_arguments)]
fn heads_mv_run(
    g: &Gpu,
    f: &Func,
    rf: bool,
    rpb: usize,
    (qt, rph, cols, n_h): (u32, usize, usize, usize),
    w: Arg,
    xd: &DevBuf,
    xs: usize,
    (z0, n): (usize, usize),
) -> Vec<f32> {
    let rows = rph * n_h;
    let rb = crate::quant::row_bytes(qt, cols);
    let out = g.alloc(n * rows * 4).unwrap();
    let mut a = vec![
        Arg::I32(qt as i32),
        w,
        Arg::U64(rb as u64),
        Arg::I32(rph as i32),
        Arg::I32(cols as i32),
        Arg::I32(n_h as i32),
        xd.arg_at(z0 * n_h * xs * 4),
        Arg::I32(xs as i32),
        Arg::U64((n_h * xs) as u64),
        out.arg(),
        Arg::I32(rph as i32),
        Arg::U64(rows as u64),
        Arg::I32(n as i32),
    ];
    let (gx, smem) = match rf {
        true => (rows.div_ceil(8), 0),
        false => {
            a.push(Arg::I32(rpb as i32));
            (
                rows.div_ceil(rpb),
                mv_smem(rpb, (qt, rb), rph, cols, n_h, n),
            )
        }
    };
    // SAFETY: the kernels' signatures; buffers sized for n tokens, the
    // weights for `rows` rows from w.
    unsafe {
        g.launch(
            f,
            (gx as u32, n.div_ceil(MLA_MV_T) as u32, 1),
            (256, 1, 1),
            smem as u32,
            &a,
        )
        .unwrap();
    }
    out.read().unwrap()
}

/// S2.2's contract: the rewritten mla_heads_mv gives main be13f1f's bits
/// (mla_heads_mv_ref) for every type it specializes and one it doesn't
/// (Q4_0: deq at run time), Moonlight's shapes (k_b Q5_0 8192 x 128, v_b
/// Q4_K 2048 x 512) and odd ones, 1 to 17 tokens (the decode step and the
/// MLA_MV_T tiles of a pass, a ragged last one), rows per block 1, 3, 8,
/// 16 and the host's, and weights at 16-, 4- and 2-byte alignment (each
/// copy path).
#[test]
fn mla_heads_mv_equals_main() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let ml = Kernels::mla_ref_module(&g).unwrap();
    let (fnew, fref) = (
        ml.function("mla_heads_mv").unwrap(),
        ml.function("mla_heads_mv_ref").unwrap(),
    );
    use crate::quant::{F16, F32, Q4_0, Q4_K, Q5_0, Q6_K, Q8_0};
    let mut checked = 0;
    for (qt, rph, cols, n_h) in [
        (Q5_0, 512, 128, 16),
        (Q4_K, 128, 512, 16),
        (Q8_0, 64, 128, 5),
        (Q6_K, 24, 512, 3),
        (F16, 33, 96, 2),
        (F32, 9, 64, 3),
        (Q4_0, 40, 160, 2),
        (Q5_0, 7, 32, 3),
    ] {
        let mut r = Rng(0xabcd ^ (qt as u64) << 8 ^ cols as u64);
        let rows = rph * n_h;
        let rb = crate::quant::row_bytes(qt, cols);
        let mut bytes = vec![0u8; 16];
        bytes.extend(random_blocks(&mut r, qt, rows * cols));
        let wd = g.upload(&bytes).unwrap();
        let xs = cols + 8;
        let nb = 17;
        let x: Vec<f32> = (0..nb * n_h * xs).map(|_| r.f()).collect();
        let xd = g.upload(&x).unwrap();
        let shape = (qt, rph, cols, n_h);
        for off in [16usize, 4, 2] {
            let w = wd.arg_at(off);
            for (z0, n) in [(0, 1), (5, 1), (0, 3), (0, 8), (0, 11), (0, 17), (9, 8)] {
                let want = heads_mv_run(&g, &fref, true, 0, shape, w, &xd, xs, (z0, n));
                // Every rows-per-block that fits (the host halves MLA_MV_ROWS until it does).
                for rpb in [mv_plan((qt, rb), rph, cols, n_h, n).0, 1, 3, 8, 16, 32]
                    .into_iter()
                    .filter(|&p| mv_smem(p, (qt, rb), rph, cols, n_h, n) <= MLA_MV_SMEM)
                {
                    let got = heads_mv_run(&g, &fnew, false, rpb, shape, w, &xd, xs, (z0, n));
                    assert_eq!(
                        bits(&got),
                        bits(&want),
                        "type {qt} ({rph} x {cols}, {n_h} heads) offset {off} tokens {z0}+{n} rpb {rpb}"
                    );
                    checked += got.len();
                }
            }
        }
    }
    eprintln!("mla_heads_mv == main: {checked} outputs");
}

/// One launch of mla_heads_mm over n tokens of x from token z0 (as
/// `heads_mv_run`).
#[allow(clippy::too_many_arguments)]
fn heads_mm_run(
    g: &Gpu,
    f: &Func,
    (qt, rph, cols, n_h): (u32, usize, usize, usize),
    w: Arg,
    xd: &DevBuf,
    xs: usize,
    (z0, n): (usize, usize),
) -> Vec<f32> {
    let rows = rph * n_h;
    let rb = crate::quant::row_bytes(qt, cols);
    let out = g.alloc(n * rows * 4).unwrap();
    // SAFETY: mla_heads_mm's signature; buffers sized for n tokens.
    unsafe {
        g.launch(
            f,
            ((rows / MLA_MM_R) as u32, n.div_ceil(MLA_MM_T) as u32, 1),
            (256, 1, 1),
            0,
            &[
                Arg::I32(qt as i32),
                w,
                Arg::U64(rb as u64),
                Arg::I32(rph as i32),
                Arg::I32(cols as i32),
                Arg::I32(n_h as i32),
                xd.arg_at(z0 * n_h * xs * 4),
                Arg::I32(xs as i32),
                Arg::U64((n_h * xs) as u64),
                out.arg(),
                Arg::I32(rph as i32),
                Arg::U64(rows as u64),
                Arg::I32(n as i32),
            ],
        )
        .unwrap();
    }
    out.read().unwrap()
}

/// S2.5's mla_heads_mm (the batched per-head products as a tiled GEMM: the
/// lane chains in bit-reversed order folded into the xor tree's shape)
/// gives main's mla_heads_mv bits (`*_ref`: be13f1f's kernels; main
/// 680a2c5's give the same bits, #145): every type, Moonlight's shapes
/// (k_b Q5_0 8192 x 128, v_b Q4_K 2048 x 512) and others (chains of 3, 4,
/// 8, 16 and 32 columns; stages that split the chains unevenly; rows staged
/// in shared memory or read from global memory (Q4_K x 1024, F32)), batches of
/// 2 to 2100 tokens (ragged last tiles), from token 0 and from token 7.
#[test]
fn mla_heads_mm_equals_main() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let ml = Kernels::mla_ref_module(&g).unwrap();
    let (fmm, fref) = (
        ml.function("mla_heads_mm").unwrap(),
        ml.function("mla_heads_mv_ref").unwrap(),
    );
    use crate::quant::{F16, F32, Q4_0, Q4_K, Q5_0, Q6_K, Q8_0};
    let mut checked = 0;
    for (qt, rph, cols, n_h, nbs) in [
        (Q5_0, 512, 128, 16, &[2usize, 33, 512, 2100][..]),
        (Q4_K, 128, 512, 16, &[2, 33, 512, 2100][..]),
        (Q8_0, 64, 256, 3, &[5, 32, 100][..]),
        (Q6_K, 128, 256, 2, &[3, 64, 65][..]),
        (F16, 64, 96, 2, &[2, 31, 70][..]),
        (F32, 192, 1024, 1, &[4, 40][..]),
        (Q4_K, 64, 1024, 2, &[3, 70][..]),
        (Q4_0, 64, 160, 2, &[9, 48][..]),
    ] {
        let mut r = Rng(0x77 ^ (qt as u64) << 8 ^ cols as u64);
        let rows = rph * n_h;
        let wd = g.upload(&random_blocks(&mut r, qt, rows * cols)).unwrap();
        let xs = cols + 8;
        let nmax = nbs.iter().max().unwrap() + 7;
        let x: Vec<f32> = (0..nmax * n_h * xs).map(|_| r.f()).collect();
        let xd = g.upload(&x).unwrap();
        let shape = (qt, rph, cols, n_h);
        for &n in nbs {
            for z0 in [0, 7] {
                let want = heads_mv_run(&g, &fref, true, 0, shape, wd.arg(), &xd, xs, (z0, n));
                let got = heads_mm_run(&g, &fmm, shape, wd.arg(), &xd, xs, (z0, n));
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "type {qt} ({rph} x {cols}, {n_h} heads) tokens {z0}+{n}"
                );
                checked += got.len();
            }
        }
    }
    eprintln!("mla_heads_mm == main: {checked} outputs");
}

/// One mla_attn + mla_attn_cmb over nb tokens at positions start.. (or,
/// with `dev_pos`, one token whose position the kernels read from P, as a
/// decode step's graph does, with `ns` splits launched): the combined
/// output and the split partials. `rf`: main's mla_attn_ref (hpb_max
/// heads per block), else the rewrite with `busy` and `dbuf`.
#[allow(clippy::too_many_arguments)]
fn attn_run(
    g: &Gpu,
    ml: &Module,
    rf: bool,
    (busy, dbuf): (i32, bool),
    qa: &DevBuf,
    kv: &DevBuf,
    (krow, lat, n_h): (usize, usize, usize),
    (start, nb, ns, dev_pos): (usize, usize, usize, bool),
) -> (Vec<f32>, Vec<f32>) {
    let fa = ml
        .function(if rf { "mla_attn_ref" } else { "mla_attn" })
        .unwrap();
    let fc = ml.function("mla_attn_cmb").unwrap();
    let part = g.alloc(nb * ns * n_h * (lat + 2) * 4).unwrap();
    part.zero().unwrap();
    let out = g.alloc(nb * n_h * lat * 4).unwrap();
    let pd = g.upload(&[0u32, start as u32]).unwrap();
    let hpb = n_h.min(8);
    let scale = 1.0 / (192f32).sqrt();
    let pos = [
        if dev_pos { pd.arg() } else { Arg::Ptr(0) },
        Arg::I32(start as i32),
        Arg::I32(MLA_SPLIT as i32),
        Arg::I32(ns as i32),
    ];
    let mut a = vec![
        qa.arg(),
        kv.arg(),
        Arg::I32(krow as i32),
        Arg::I32(lat as i32),
        Arg::I32(n_h as i32),
        Arg::I32(hpb as i32),
        Arg::F32(scale),
    ];
    a.extend(pos);
    a.push(part.arg());
    let (gx, gz, smem) = match rf {
        true => (n_h / hpb, ns, attn_smem(krow, false)),
        false => {
            a.push(Arg::I32(busy));
            a.push(Arg::I32(dbuf as i32));
            (
                attn_blocks(n_h, hpb, ns, nb, busy.max(0) as usize),
                1,
                attn_smem(krow, dbuf),
            )
        }
    };
    // SAFETY: the kernels' signatures; the cache holds start + nb rows,
    // part nb x ns splits.
    unsafe {
        g.launch(
            &fa,
            (gx as u32, nb as u32, gz as u32),
            (256, 1, 1),
            smem as u32,
            &a,
        )
        .unwrap();
        let mut a = vec![part.arg(), Arg::I32(lat as i32), Arg::I32(n_h as i32)];
        a.extend(pos);
        a.push(out.arg());
        g.launch(&fc, (n_h as u32, nb as u32, 1), (256, 1, 1), 0, &a)
            .unwrap();
    }
    (out.read().unwrap(), part.read().unwrap())
}

/// mla_attn_b + mla_attn_cmb over nb tokens at positions start.. (qa: their
/// queries): the combined output and the partials (as `attn_run`).
fn attn_b_run(
    g: &Gpu,
    ml: &Module,
    qa: &DevBuf,
    kv: &DevBuf,
    (krow, lat, n_h): (usize, usize, usize),
    (start, nb, ns): (usize, usize, usize),
) -> (Vec<f32>, Vec<f32>) {
    let fa = ml.function("mla_attn_b").unwrap();
    let fc = ml.function("mla_attn_cmb").unwrap();
    let part = g.alloc(nb * ns * n_h * (lat + 2) * 4).unwrap();
    part.zero().unwrap();
    let out = g.alloc(nb * n_h * lat * 4).unwrap();
    let hb = n_h.min(MLA_AB_Q);
    let scale = 1.0 / (192f32).sqrt();
    let nsp = (start + nb).div_ceil(MLA_SPLIT);
    // SAFETY: the kernels' signatures; the cache holds start + nb rows,
    // part nb x ns splits.
    unsafe {
        g.launch(
            &fa,
            (
                (nsp * (n_h / hb)) as u32,
                nb.div_ceil(MLA_AB_Q / hb) as u32,
                1,
            ),
            (MLA_AB_THREADS as u32, 1, 1),
            attn_b_smem(krow) as u32,
            &[
                qa.arg(),
                kv.arg(),
                Arg::I32(krow as i32),
                Arg::I32(lat as i32),
                Arg::I32(n_h as i32),
                Arg::F32(scale),
                Arg::I32(start as i32),
                Arg::I32(MLA_SPLIT as i32),
                Arg::I32(ns as i32),
                Arg::I32(nb as i32),
                part.arg(),
            ],
        )
        .unwrap();
        let a = [
            part.arg(),
            Arg::I32(lat as i32),
            Arg::I32(n_h as i32),
            Arg::Ptr(0),
            Arg::I32(start as i32),
            Arg::I32(MLA_SPLIT as i32),
            Arg::I32(ns as i32),
            out.arg(),
        ];
        g.launch(&fc, (n_h as u32, nb as u32, 1), (256, 1, 1), 0, &a)
            .unwrap();
    }
    (out.read().unwrap(), part.read().unwrap())
}

/// S2.5's contract: mla_attn_b (a batch: K tiles shared by 32 (token,
/// head) pairs) gives main's bits (mla_attn_ref: be13f1f's kernel; main
/// 680a2c5's gives the same bits, #145), partials
/// included: Moonlight's shape from position 0 (2,100 tokens: 17 splits;
/// and 700) and from 300 (a pass after a cache prefix), K2's (64 heads), an
/// 8-head one and the fixture's (2 heads, row 80), and batches of 2 to 33
/// tokens across a split edge. sm_80+ (its shared memory); skipped below.
#[test]
fn mla_attn_b_equals_main() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let ml = Kernels::mla_ref_module(&g).unwrap();
    if !ml.mma() {
        eprintln!("SKIPPED: mla_attn_b is sm_80+");
        return;
    }
    ml.function("mla_attn_b")
        .unwrap()
        .set_max_shared(attn_b_smem(MLA_MAX_KROW))
        .unwrap();
    let mut checked = 0;
    for (n_h, krow, lat, np, cases) in [
        (
            16usize,
            576usize,
            512usize,
            2100usize,
            &[
                (0usize, 2100usize),
                (0, 700),
                (300, 400),
                (120, 2),
                (126, 5),
                (100, 33),
            ][..],
        ),
        (64, 576, 512, 260, &[(0, 260), (100, 17)][..]),
        (8, 576, 512, 140, &[(0, 140), (31, 3)][..]),
        (2, 80, 64, 300, &[(0, 300), (250, 50)][..]),
    ] {
        let mut r = Rng(0xa77 ^ (n_h * krow) as u64);
        let k16: Vec<u16> = (0..np * krow)
            .map(|_| crate::quant::f32_to_f16(r.f() * 2.0))
            .collect();
        let kv = g.upload(&k16).unwrap();
        let w = n_h * krow;
        let qa: Vec<f32> = (0..np * w).map(|_| r.f()).collect();
        for &(start, nb) in cases {
            let qd = g.upload(&qa[start * w..(start + nb) * w]).unwrap();
            let ns = (start + nb).div_ceil(MLA_SPLIT);
            let shape = (krow, lat, n_h);
            let want = attn_run(
                &g,
                &ml,
                true,
                (0, false),
                &qd,
                &kv,
                shape,
                (start, nb, ns, false),
            );
            let got = attn_b_run(&g, &ml, &qd, &kv, shape, (start, nb, ns));
            assert_eq!(
                bits(&got.0),
                bits(&want.0),
                "({n_h}, {krow}) tokens {start}+{nb}"
            );
            assert_eq!(
                bits(&got.1),
                bits(&want.1),
                "({n_h}, {krow}) tokens {start}+{nb} partials"
            );
            checked += got.0.len();
        }
    }
    eprintln!("mla_attn_b == main: {checked} outputs");
}

/// S2.3's contract: the rewritten mla_attn (heads per block by the work,
/// cooperative row scores, double-buffered tiles) gives main be13f1f's
/// bits (mla_attn_ref), partials included, on Moonlight's shape, K2's (64
/// heads), the fixture's (2 heads, row 80) and an 8-head one: a batch of
/// tokens from 0 (6 splits at Moonlight's), and lone tokens at positions
/// read from P (the decode graph's way, 8 splits launched) across tile and
/// split edges; at every hpb the rule can pick (busy 0: hpb_max, the SM
/// count, 100000: 1) and with and without double buffering.
#[test]
fn mla_attn_equals_main() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let ml = Kernels::mla_ref_module(&g).unwrap();
    let can_dbuf = ml.mma();
    if can_dbuf {
        ml.function("mla_attn")
            .unwrap()
            .set_max_shared(attn_smem(MLA_MAX_KROW, true))
            .unwrap();
    }
    let sms = g.info().unwrap().sms as i32;
    let mut checked = 0;
    for (n_h, krow, lat, np) in [
        (16usize, 576usize, 512usize, 700usize),
        (2, 80, 64, 300),
        (64, 576, 512, 260),
        (8, 576, 512, 140),
    ] {
        let mut r = Rng(0x5eed ^ (n_h * krow) as u64);
        let k16: Vec<u16> = (0..np * krow)
            .map(|_| crate::quant::f32_to_f16(r.f() * 2.0))
            .collect();
        let kv = g.upload(&k16).unwrap();
        let w = n_h * krow;
        let qa: Vec<f32> = (0..np * w).map(|_| r.f()).collect();
        let qd = g.upload(&qa).unwrap();
        let shape = (krow, lat, n_h);
        let ns_all = np.div_ceil(MLA_SPLIT);
        let modes: Vec<(i32, bool)> = [0, sms, 100_000]
            .into_iter()
            .flat_map(|b| {
                [false, true]
                    .into_iter()
                    .filter(|&d| !d || (can_dbuf && krow % 8 == 0))
                    .map(move |d| (b, d))
            })
            .collect();
        let want = attn_run(
            &g,
            &ml,
            true,
            (0, false),
            &qd,
            &kv,
            shape,
            (0, np, ns_all, false),
        );
        for &m in &modes {
            let got = attn_run(&g, &ml, false, m, &qd, &kv, shape, (0, np, ns_all, false));
            assert_eq!(
                bits(&got.0),
                bits(&want.0),
                "({n_h}, {krow}) batch of {np}, mode {m:?}"
            );
            assert_eq!(
                bits(&got.1),
                bits(&want.1),
                "({n_h}, {krow}) batch partials, mode {m:?}"
            );
            checked += got.0.len();
        }
        for p in [
            0,
            1,
            7,
            31,
            32,
            33,
            63,
            127,
            128,
            129,
            136,
            255,
            256,
            np - 1,
        ] {
            if p >= np {
                continue;
            }
            let q1 = g.upload(&qa[p * w..(p + 1) * w]).unwrap();
            let ns = 8.max(ns_all);
            let want = attn_run(&g, &ml, true, (0, false), &q1, &kv, shape, (p, 1, ns, true));
            for &m in &modes {
                let got = attn_run(&g, &ml, false, m, &q1, &kv, shape, (p, 1, ns, true));
                assert_eq!(
                    bits(&got.0),
                    bits(&want.0),
                    "({n_h}, {krow}) position {p}, mode {m:?}"
                );
                assert_eq!(
                    bits(&got.1),
                    bits(&want.1),
                    "({n_h}, {krow}) position {p} partials, mode {m:?}"
                );
                checked += got.0.len();
            }
        }
    }
    eprintln!("mla_attn == main: {checked} outputs");
}

/// Kernel times, main's kernels vs the rewrite, at Moonlight's decode
/// shapes (run on its own: --ignored --nocapture): mla_heads_mv k_b (Q5_0
/// 8192 x 128) and v_b (Q4_K 2048 x 512) for one token, rows per block 8,
/// 16, 32; mla_attn for one token at positions 8 .. 4000 with a 4096-row
/// cache's 32 splits launched (the decode graph's), each hpb mode.
#[test]
#[ignore]
fn mla_decode_kernel_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    // AXON_MLA_SRC=file: a whole MLA module's source (kernels.cu, moe.cu,
    // a variant of mla.cu, mla_ref.cu) instead of the built-in one, so
    // kernel variants can be timed without a rebuild (tuning only).
    let ml = match std::env::var("AXON_MLA_SRC") {
        Ok(f) => {
            let src = std::fs::read_to_string(&f).unwrap();
            eprintln!("bench: kernels from {f}");
            crate::cuda::Module::load(&g, &src, &crate::cuda::KERNEL_OPTS_EXT)
                .unwrap()
                .0
        }
        Err(_) => Kernels::mla_ref_module(&g).unwrap(),
    };
    let time = |f: &mut dyn FnMut()| -> f64 {
        for _ in 0..20 {
            f();
        }
        g.sync().unwrap();
        let t = std::time::Instant::now();
        let n = 400;
        for _ in 0..n {
            f();
        }
        g.sync().unwrap();
        t.elapsed().as_secs_f64() * 1e6 / n as f64
    };
    let (fnew, fref) = (
        ml.function("mla_heads_mv").unwrap(),
        ml.function("mla_heads_mv_ref").unwrap(),
    );
    use crate::quant::{Q4_K, Q5_0};
    for (name, qt, rph, cols) in [("k_b", Q5_0, 512, 128), ("v_b", Q4_K, 128, 512)] {
        let n_h = 16;
        let mut r = Rng(3);
        let rows = rph * n_h;
        let rb = crate::quant::row_bytes(qt, cols);
        let wd = g.upload(&random_blocks(&mut r, qt, rows * cols)).unwrap();
        let x: Vec<f32> = (0..n_h * cols).map(|_| r.f()).collect();
        let xd = g.upload(&x).unwrap();
        let out = g.alloc(rows * 4).unwrap();
        let args = |extra: Option<usize>| {
            let mut a = vec![
                Arg::I32(qt as i32),
                wd.arg(),
                Arg::U64(rb as u64),
                Arg::I32(rph as i32),
                Arg::I32(cols as i32),
                Arg::I32(n_h as i32),
                xd.arg(),
                Arg::I32(cols as i32),
                Arg::U64((n_h * cols) as u64),
                out.arg(),
                Arg::I32(rph as i32),
                Arg::U64(rows as u64),
                Arg::I32(1),
            ];
            if let Some(rpb) = extra {
                a.push(Arg::I32(rpb as i32));
            }
            a
        };
        let a = args(None);
        // SAFETY (each launch): the kernels' signatures, buffers sized above.
        let t_ref = time(&mut || unsafe {
            g.launch(&fref, (rows.div_ceil(8) as u32, 1, 1), (256, 1, 1), 0, &a)
                .unwrap()
        });
        let mut line = format!("heads_mv {name}: main {t_ref:.1} us");
        for rpb in [8, 16, 32, 64] {
            let a = args(Some(rpb));
            let smem = mv_smem(rpb, (qt, rb), rph, cols, n_h, 1);
            let t = time(&mut || unsafe {
                g.launch(
                    &fnew,
                    (rows.div_ceil(rpb) as u32, 1, 1),
                    (256, 1, 1),
                    smem as u32,
                    &a,
                )
                .unwrap()
            });
            line += &format!(" | rpb {rpb} {t:.1} us");
        }
        println!("{line}");
        // Batches: main's kernel, the rewrite's mla_heads_mv, mla_heads_mm.
        let fmm = ml.function("mla_heads_mm").unwrap();
        for nb in [8usize, 16, 32, 512, 2100] {
            let xb: Vec<f32> = (0..nb * n_h * cols).map(|_| r.f()).collect();
            let xbd = g.upload(&xb).unwrap();
            let ob = g.alloc(nb * rows * 4).unwrap();
            let mk = |extra: Option<usize>| {
                let mut a = vec![
                    Arg::I32(qt as i32),
                    wd.arg(),
                    Arg::U64(rb as u64),
                    Arg::I32(rph as i32),
                    Arg::I32(cols as i32),
                    Arg::I32(n_h as i32),
                    xbd.arg(),
                    Arg::I32(cols as i32),
                    Arg::U64((n_h * cols) as u64),
                    ob.arg(),
                    Arg::I32(rph as i32),
                    Arg::U64(rows as u64),
                    Arg::I32(nb as i32),
                ];
                if let Some(rpb) = extra {
                    a.push(Arg::I32(rpb as i32));
                }
                a
            };
            let gy = nb.div_ceil(MLA_MV_T) as u32;
            let a0 = mk(None);
            let t_ref = time(&mut || unsafe {
                g.launch(&fref, (rows.div_ceil(8) as u32, gy, 1), (256, 1, 1), 0, &a0)
                    .unwrap()
            });
            let (rpb, smem) = mv_plan((qt, rb), rph, cols, n_h, nb);
            let a1 = mk(Some(rpb));
            let t_mv = time(&mut || unsafe {
                g.launch(
                    &fnew,
                    (rows.div_ceil(rpb) as u32, gy, 1),
                    (256, 1, 1),
                    smem as u32,
                    &a1,
                )
                .unwrap()
            });
            let t_mm = time(&mut || unsafe {
                g.launch(
                    &fmm,
                    ((rows / MLA_MM_R) as u32, nb.div_ceil(MLA_MM_T) as u32, 1),
                    (256, 1, 1),
                    0,
                    &a0,
                )
                .unwrap()
            });
            println!(
                "heads {name} nb {nb}: main {t_ref:.1} us | mv {t_mv:.1} us | mm {t_mm:.1} us"
            );
        }
    }
    let fa_ref = ml.function("mla_attn_ref").unwrap();
    let fa = ml.function("mla_attn").unwrap();
    let can_dbuf = ml.mma();
    if can_dbuf {
        fa.set_max_shared(attn_smem(MLA_MAX_KROW, true)).unwrap();
    }
    let sms = g.info().unwrap().sms as i32;
    let (n_h, krow, lat, ctx) = (16usize, 576usize, 512usize, 4096usize);
    let ns = ctx / MLA_SPLIT;
    let mut r = Rng(9);
    let k16: Vec<u16> = (0..ctx * krow)
        .map(|_| crate::quant::f32_to_f16(r.f() * 2.0))
        .collect();
    let kv = g.upload(&k16).unwrap();
    let qa: Vec<f32> = (0..n_h * krow).map(|_| r.f()).collect();
    let qd = g.upload(&qa).unwrap();
    let part = g.alloc(ns * n_h * (lat + 2) * 4).unwrap();
    for p in [8usize, 64, 136, 300, 1000, 4000] {
        let pd = g.upload(&[0u32, p as u32]).unwrap();
        let base = |hpb: usize| {
            vec![
                qd.arg(),
                kv.arg(),
                Arg::I32(krow as i32),
                Arg::I32(lat as i32),
                Arg::I32(n_h as i32),
                Arg::I32(hpb as i32),
                Arg::F32(0.07),
                pd.arg(),
                Arg::I32(0),
                Arg::I32(MLA_SPLIT as i32),
                Arg::I32(ns as i32),
                part.arg(),
            ]
        };
        let a = base(8);
        let t_ref = time(&mut || unsafe {
            g.launch(
                &fa_ref,
                (2, 1, ns as u32),
                (256, 1, 1),
                attn_smem(krow, false) as u32,
                &a,
            )
            .unwrap()
        });
        let mut line = format!("attn pos {p}: main {t_ref:.1} us");
        for (busy, label) in [(0, "hpb 8"), (sms, "rule"), (100_000, "hpb 1")] {
            for dbuf in [false, true].into_iter().filter(|&d| !d || can_dbuf) {
                let mut a = base(8);
                a.push(Arg::I32(busy));
                a.push(Arg::I32(dbuf as i32));
                let t = time(&mut || unsafe {
                    g.launch(
                        &fa,
                        (
                            attn_blocks(n_h, 8, ns, 1, busy.max(0) as usize) as u32,
                            1,
                            1,
                        ),
                        (256, 1, 1),
                        attn_smem(krow, dbuf) as u32,
                        &a,
                    )
                    .unwrap()
                });
                line += &format!(" | {label}{} {t:.1}", if dbuf { " dbuf" } else { "" });
            }
        }
        println!("{line} us");
    }
    // Batches from position 0 (a prompt's prefill): main's kernel vs mla_attn_b.
    if can_dbuf {
        let fb = ml.function("mla_attn_b").unwrap();
        fb.set_max_shared(attn_b_smem(MLA_MAX_KROW)).unwrap();
        for nb in [64usize, 512, 2100] {
            let nsp = nb.div_ceil(MLA_SPLIT);
            let qa: Vec<f32> = (0..nb * n_h * krow).map(|_| r.f()).collect();
            let qd = g.upload(&qa).unwrap();
            let pt = g.alloc(nb * nsp * n_h * (lat + 2) * 4).unwrap();
            let a_ref = vec![
                qd.arg(),
                kv.arg(),
                Arg::I32(krow as i32),
                Arg::I32(lat as i32),
                Arg::I32(n_h as i32),
                Arg::I32(8),
                Arg::F32(0.07),
                Arg::Ptr(0),
                Arg::I32(0),
                Arg::I32(MLA_SPLIT as i32),
                Arg::I32(nsp as i32),
                pt.arg(),
            ];
            let t_ref = time(&mut || unsafe {
                g.launch(
                    &fa_ref,
                    (2, nb as u32, nsp as u32),
                    (256, 1, 1),
                    attn_smem(krow, false) as u32,
                    &a_ref,
                )
                .unwrap()
            });
            let mut a_new = a_ref.clone();
            a_new.push(Arg::I32(sms));
            a_new.push(Arg::I32(0));
            let t_new = time(&mut || unsafe {
                g.launch(
                    &fa,
                    (
                        attn_blocks(n_h, 8, nsp, nb, sms as usize) as u32,
                        nb as u32,
                        1,
                    ),
                    (256, 1, 1),
                    attn_smem(krow, false) as u32,
                    &a_new,
                )
                .unwrap()
            });
            let a_b = [
                qd.arg(),
                kv.arg(),
                Arg::I32(krow as i32),
                Arg::I32(lat as i32),
                Arg::I32(n_h as i32),
                Arg::F32(0.07),
                Arg::I32(0),
                Arg::I32(MLA_SPLIT as i32),
                Arg::I32(nsp as i32),
                Arg::I32(nb as i32),
                pt.arg(),
            ];
            let t_b = time(&mut || unsafe {
                g.launch(
                    &fb,
                    (nsp as u32, nb.div_ceil(MLA_AB_Q / n_h) as u32, 1),
                    (MLA_AB_THREADS as u32, 1, 1),
                    attn_b_smem(krow) as u32,
                    &a_b,
                )
                .unwrap()
            });
            println!(
                "attn batch {nb} from 0: main {t_ref:.0} us | decode kernel {t_new:.0} us | attn_b {t_b:.0} us"
            );
        }
    }
}

/// The dispatch rule (`mm_fits`) at the measured points: k_b from 32 tokens
/// on, v_b at 64 and from 116, not at 65 .. 115.
#[test]
fn heads_mm_dispatch() {
    let k_b = |nb| mm_fits(nb, 512, 128);
    let v_b = |nb| mm_fits(nb, 128, 512);
    assert!(!k_b(31) && k_b(32) && k_b(64) && k_b(96) && k_b(2100));
    assert!(!v_b(63) && v_b(64) && !v_b(65) && !v_b(96) && !v_b(115));
    assert!(v_b(116) && v_b(128) && !v_b(130) && v_b(2100));
}

/// `attn_blocks` covers every token's blocks under mla_attn's hpb rule
/// (emulated: the largest power of two <= hpb_max with n_h / hpb * nsp *
/// nz >= busy, else 1), for any token's split count nsp <= ns.
#[test]
fn attn_blocks_cover_the_hpb_rule() {
    for n_h in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        let hpb_max = n_h.min(8);
        for ns in [1usize, 2, 3, 7, 32, 64, 2048] {
            for nz in [1usize, 2, 5, 64, 512] {
                for busy in [0usize, 1, 20, 46, 132, 100_000] {
                    let x = attn_blocks(n_h, hpb_max, ns, nz, busy);
                    for nsp in (1..=ns).filter(|&v| v <= 70 || v == ns) {
                        let mut hpb = hpb_max;
                        while hpb > 1 && (n_h / hpb) * nsp * nz < busy {
                            hpb >>= 1;
                        }
                        let need = (n_h / hpb) * nsp;
                        assert!(
                            need <= x,
                            "n_h {n_h} ns {ns} nz {nz} busy {busy} nsp {nsp}: {need} > {x}"
                        );
                    }
                }
            }
        }
    }
}

/// S2.5's A/B at the kernel level: main 680a2c5's batched kernels (its
/// `mla_heads_mv` with `mv_plan`, its `mla_attn` with the hpb rule) vs this
/// branch's `mla_heads_mm` and `mla_attn_b`, on Moonlight's shapes. Each
/// point: 3 rounds, arms alternating (main, cand, main, cand, ..), each a
/// mean over back-to-back launches; printed as the median [min-max] µs.
/// heads: nb 8 .. 2100 (the dispatch crossover); attn: batches of 64, 512
/// and 2,100 tokens from position 0. Run on its own: --ignored --nocapture.
/// AXON_MLA_SRC=file: another module source (variants); AXON_MLA_AB_THREADS:
/// mla_attn_b's block size for such a variant.
#[test]
#[ignore]
fn mla_prefill_kernel_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let ml = match std::env::var("AXON_MLA_SRC") {
        Ok(f) => {
            let src = std::fs::read_to_string(&f).unwrap();
            eprintln!("bench: kernels from {f}");
            crate::cuda::Module::load(&g, &src, &crate::cuda::KERNEL_OPTS_EXT)
                .unwrap()
                .0
        }
        Err(_) => Kernels::mla_ref_module(&g).unwrap(),
    };
    let ab_threads: u32 = std::env::var("AXON_MLA_AB_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MLA_AB_THREADS as u32);
    let info = g.info().unwrap();
    println!(
        "bench on {} (sm_{}{}, {} SMs)",
        info.name, info.cc.0, info.cc.1, info.sms
    );
    // Mean µs per launch over n launches (after 3 warm-up launches).
    let time = |n: usize, f: &mut dyn FnMut()| -> f64 {
        for _ in 0..3 {
            f();
        }
        g.sync().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..n {
            f();
        }
        g.sync().unwrap();
        t.elapsed().as_secs_f64() * 1e6 / n as f64
    };
    let med = |v: &mut Vec<f64>| -> String {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        format!("{:.1} [{:.1}-{:.1}]", v[v.len() / 2], v[0], v[v.len() - 1])
    };
    let (fmv, fmm) = (
        ml.function("mla_heads_mv").unwrap(),
        ml.function("mla_heads_mm").unwrap(),
    );
    use crate::quant::{Q4_K, Q5_0};
    for (name, qt, rph, cols) in [("k_b", Q5_0, 512usize, 128usize), ("v_b", Q4_K, 128, 512)] {
        let n_h = 16;
        let mut r = Rng(3);
        let rows = rph * n_h;
        let rb = crate::quant::row_bytes(qt, cols);
        let wd = g.upload(&random_blocks(&mut r, qt, rows * cols)).unwrap();
        for nb in [
            8usize, 16, 24, 32, 48, 64, 80, 96, 112, 120, 128, 192, 512, 2100,
        ] {
            let x: Vec<f32> = (0..nb * n_h * cols).map(|_| r.f()).collect();
            let xd = g.upload(&x).unwrap();
            let ob = g.alloc(nb * rows * 4).unwrap();
            let mut a = vec![
                Arg::I32(qt as i32),
                wd.arg(),
                Arg::U64(rb as u64),
                Arg::I32(rph as i32),
                Arg::I32(cols as i32),
                Arg::I32(n_h as i32),
                xd.arg(),
                Arg::I32(cols as i32),
                Arg::U64((n_h * cols) as u64),
                ob.arg(),
                Arg::I32(rph as i32),
                Arg::U64(rows as u64),
                Arg::I32(nb as i32),
            ];
            let a_mm = a.clone();
            let (rpb, smem) = mv_plan((qt, rb), rph, cols, n_h, nb);
            a.push(Arg::I32(rpb as i32));
            let n = (25_600 / nb).clamp(20, 400);
            let (mut tm, mut tc) = (Vec::new(), Vec::new());
            for _ in 0..3 {
                // SAFETY (each launch): the kernels' signatures, buffers sized above.
                tm.push(time(n, &mut || unsafe {
                    g.launch(
                        &fmv,
                        (rows.div_ceil(rpb) as u32, nb.div_ceil(MLA_MV_T) as u32, 1),
                        (256, 1, 1),
                        smem as u32,
                        &a,
                    )
                    .unwrap()
                }));
                tc.push(time(n, &mut || unsafe {
                    g.launch(
                        &fmm,
                        ((rows / MLA_MM_R) as u32, nb.div_ceil(MLA_MM_T) as u32, 1),
                        (256, 1, 1),
                        0,
                        &a_mm,
                    )
                    .unwrap()
                }));
            }
            let pick = if mm_fits(nb, rph, cols) { "mm" } else { "mv" };
            println!(
                "heads {name} nb {nb}: main {} us | cand mm {} us | dispatch: {pick}",
                med(&mut tm),
                med(&mut tc)
            );
        }
    }
    if !ml.mma() {
        println!("attn: mla_attn_b is sm_80+, skipped");
        return;
    }
    let (fa, fb) = (
        ml.function("mla_attn").unwrap(),
        ml.function("mla_attn_b").unwrap(),
    );
    fb.set_max_shared(attn_b_smem(MLA_MAX_KROW)).unwrap();
    let sms = info.sms as usize;
    let (n_h, krow, lat) = (16usize, 576usize, 512usize);
    let mut r = Rng(9);
    for nb in [64usize, 512, 2100] {
        let nsp = nb.div_ceil(MLA_SPLIT);
        let k16: Vec<u16> = (0..nb * krow)
            .map(|_| crate::quant::f32_to_f16(r.f() * 2.0))
            .collect();
        let kv = g.upload(&k16).unwrap();
        let qa: Vec<f32> = (0..nb * n_h * krow).map(|_| r.f()).collect();
        let qd = g.upload(&qa).unwrap();
        let pt = g.alloc(nb * nsp * n_h * (lat + 2) * 4).unwrap();
        let a_main = vec![
            qd.arg(),
            kv.arg(),
            Arg::I32(krow as i32),
            Arg::I32(lat as i32),
            Arg::I32(n_h as i32),
            Arg::I32(8),
            Arg::F32(0.07),
            Arg::Ptr(0),
            Arg::I32(0),
            Arg::I32(MLA_SPLIT as i32),
            Arg::I32(nsp as i32),
            pt.arg(),
            Arg::I32(sms as i32),
            Arg::I32(0),
        ];
        let a_b = [
            qd.arg(),
            kv.arg(),
            Arg::I32(krow as i32),
            Arg::I32(lat as i32),
            Arg::I32(n_h as i32),
            Arg::F32(0.07),
            Arg::I32(0),
            Arg::I32(MLA_SPLIT as i32),
            Arg::I32(nsp as i32),
            Arg::I32(nb as i32),
            pt.arg(),
        ];
        let n = (12_800 / nb).clamp(5, 200);
        let (mut tm, mut tc) = (Vec::new(), Vec::new());
        for _ in 0..3 {
            // SAFETY (each launch): the kernels' signatures, buffers sized above.
            tm.push(time(n, &mut || unsafe {
                g.launch(
                    &fa,
                    (attn_blocks(n_h, 8, nsp, nb, sms) as u32, nb as u32, 1),
                    (256, 1, 1),
                    attn_smem(krow, false) as u32,
                    &a_main,
                )
                .unwrap()
            }));
            tc.push(time(n, &mut || unsafe {
                g.launch(
                    &fb,
                    (nsp as u32, nb.div_ceil(MLA_AB_Q / n_h) as u32, 1),
                    (ab_threads, 1, 1),
                    attn_b_smem(krow) as u32,
                    &a_b,
                )
                .unwrap()
            }));
        }
        println!(
            "attn batch {nb} from 0: main {} us | cand attn_b ({ab_threads} threads) {} us",
            med(&mut tm),
            med(&mut tc)
        );
    }
}

/// `n_h` = 128 (DeepSeek V3's): 4 head groups per split in mla_attn_b.
#[test]
fn mla_attn_b_equals_main_128_heads() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let ml = Kernels::mla_ref_module(&g).unwrap();
    if !ml.mma() {
        eprintln!("SKIPPED: mla_attn_b is sm_80+");
        return;
    }
    ml.function("mla_attn_b")
        .unwrap()
        .set_max_shared(attn_b_smem(MLA_MAX_KROW))
        .unwrap();
    let (n_h, krow, lat, np) = (128usize, 576usize, 512usize, 200usize);
    let mut r = Rng(0x128);
    let k16: Vec<u16> = (0..np * krow)
        .map(|_| crate::quant::f32_to_f16(r.f() * 2.0))
        .collect();
    let kv = g.upload(&k16).unwrap();
    let w = n_h * krow;
    let qa: Vec<f32> = (0..np * w).map(|_| r.f()).collect();
    for (start, nb) in [(0usize, 200usize), (120, 9)] {
        let qd = g.upload(&qa[start * w..(start + nb) * w]).unwrap();
        let ns = (start + nb).div_ceil(MLA_SPLIT);
        let shape = (krow, lat, n_h);
        let want = attn_run(
            &g,
            &ml,
            true,
            (0, false),
            &qd,
            &kv,
            shape,
            (start, nb, ns, false),
        );
        let got = attn_b_run(&g, &ml, &qd, &kv, shape, (start, nb, ns));
        assert_eq!(
            bits(&got.0),
            bits(&want.0),
            "128 heads, tokens {start}+{nb}"
        );
        assert_eq!(
            bits(&got.1),
            bits(&want.1),
            "128 heads, tokens {start}+{nb} partials"
        );
    }
}

/// moe_router (sigmoid / softmax gates) against the CPU router's rule (moe.rs `route` with a `Gate`):
/// sigmoid + bias selection, unbiased weights renormalized then scaled;
/// softmax without bias. Selection on f32 values as the kernel's (ties,
/// which a saturated sigmoid makes, to the lower index, as `route`); the
/// same ids, weights within 1e-6 of f64.
#[test]
fn mla_router_matches_cpu_rule() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let ml = Kernels::mla_module(&g).unwrap();
    let f = ml.function("moe_router").unwrap();
    let (n_e, k, nb) = (64, 6, 9);
    let mut r = Rng(5);
    let lg: Vec<f32> = (0..nb * n_e).map(|_| 4.0 * r.f()).collect();
    let bias: Vec<f32> = (0..n_e).map(|_| 0.3 * r.f()).collect();
    for (sigm, use_bias, norm, scale) in [
        (1, true, 1, 2.446f32),
        (0, false, 1, 1.0),
        (1, false, 0, 2.5),
    ] {
        let lgd = g.upload(&lg).unwrap();
        let bd = g.upload(&bias).unwrap();
        let ids = g.alloc(nb * k * 4).unwrap();
        let w = g.alloc(nb * k * 4).unwrap();
        // SAFETY: buffers sized for nb tokens, the bias n_e floats.
        unsafe {
            super::super::moe::launch_router(
                &g,
                &f,
                lgd.arg(),
                (n_e, k, nb),
                if sigm == 1 {
                    GATE_SIGMOID
                } else {
                    GATE_SOFTMAX
                },
                if use_bias { bd.arg() } else { Arg::Ptr(0) },
                norm == 1,
                scale,
                (ids.arg(), w.arg()),
            )
            .unwrap();
        }
        let gi: Vec<i32> = ids.read().unwrap();
        let gw: Vec<f32> = w.read().unwrap();
        for z in 0..nb {
            let l = &lg[z * n_e..(z + 1) * n_e];
            let p: Vec<f64> = if sigm == 1 {
                l.iter().map(|&x| 1.0 / (1.0 + (-x as f64).exp())).collect()
            } else {
                let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let e: Vec<f64> = l.iter().map(|&x| (x as f64 - mx).exp()).collect();
                let s: f64 = e.iter().sum();
                e.iter().map(|v| v / s).collect()
            };
            // The kernel's f32 selection values (sigmoid: its ops).
            let sel: Vec<f32> = (0..n_e)
                .map(|e| {
                    let pf = if sigm == 1 {
                        1.0 / (1.0 + (-l[e]).exp())
                    } else {
                        p[e] as f32
                    };
                    pf + if use_bias { bias[e] } else { 0.0 }
                })
                .collect();
            let mut order: Vec<usize> = (0..n_e).collect();
            order.sort_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap().then(a.cmp(&b)));
            let top = &order[..k];
            let s: f64 = top.iter().map(|&e| p[e]).sum();
            for (i, &e) in top.iter().enumerate() {
                assert_eq!(gi[z * k + i] as usize, e, "token {z} rank {i}");
                let want = p[e] / if norm == 1 { s } else { 1.0 } * scale as f64;
                let got = gw[z * k + i] as f64;
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "{got} vs {want}"
                );
            }
        }
    }
}

/// The MLA module compiles (NVRTC, no GPU needed) for sm_70 (no tensor
/// cores; the MLA kernels use none), sm_80 and sm_90, with every weight
/// type (its deq), and the constants mla.rs mirrors match its #defines.
/// Compiling is skipped without NVRTC.
#[test]
fn mla_kernels_compile_for_archs() {
    use crate::cuda::qwen2::moe::{GATE_OAI, MOE_THREADS};
    use crate::cuda::{KERNEL_OPTS_EXT, MLA_KERNELS, Target, compile, ffi};
    // The constants mla.rs mirrors are the source's (a drift would launch
    // with the wrong shapes silently).
    for (name, v) in [
        ("MLA_TILE", MLA_TILE),
        ("MLA_MV_T", MLA_MV_T),
        ("MLA_AB_Q", MLA_AB_Q),
        ("MLA_AB_THREADS", MLA_AB_THREADS),
        ("MLA_AB_CJ", MLA_AB_CJ),
        ("MLA_MM_R", MLA_MM_R),
        ("MLA_MM_T", MLA_MM_T),
        ("MLA_MM_SC", MLA_MM_SC),
        ("MLA_MAX_SPLITS", MLA_MAX_SPLITS),
        ("MLA_MAXI", MLA_MAX_KROW / 64),
        ("MLA_THREADS", 256),
        ("MOE_THREADS", MOE_THREADS as usize),
        ("MOE_LG_WARPS", MOE_LG_WARPS),
        ("MOE_GATE_SOFTMAX", GATE_SOFTMAX as usize),
        ("MOE_GATE_SIGMOID", GATE_SIGMOID as usize),
        ("MOE_GATE_OAI", GATE_OAI as usize),
    ] {
        let d = format!("#define {name} {v}");
        assert!(
            MLA_KERNELS.lines().any(|l| l.starts_with(&d)
                && l[d.len()..]
                    .trim_start()
                    .chars()
                    .next()
                    .is_none_or(|c| c == '/')),
            "{d} not in the MLA module's source"
        );
    }
    assert!(MLA_SPLIT.is_multiple_of(MLA_TILE));
    let n = match ffi::nvrtc() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let archs = ffi::nvrtc_archs(n).unwrap();
    for a in [70, 80, 90].into_iter().filter(|a| archs.contains(a)) {
        let t = Target::Cubin(a);
        let mut opts = vec![format!("--gpu-architecture={t}")];
        opts.extend(KERNEL_OPTS_EXT.iter().map(|s| s.to_string()));
        let image = compile(n, MLA_KERNELS, &opts, t).unwrap_or_else(|e| panic!("sm_{a}: {e}"));
        eprintln!("mla sm_{a}: {} bytes of cubin", image.len());
        compile(n, crate::cuda::MLA_REF_KERNELS, &opts, t)
            .unwrap_or_else(|e| panic!("sm_{a} (with the reference): {e}"));
    }
}
