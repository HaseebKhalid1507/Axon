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
        g.launch(
            &fa,
            ((n_h / hpb) as u32, nb as u32, ns as u32),
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
            // SAFETY: mla_heads_mv's signature; buffers sized for n tokens.
            unsafe {
                g.launch(
                    &f,
                    (rows.div_ceil(8) as u32, n.div_ceil(MLA_MV_T) as u32, 1),
                    (256, 1, 1),
                    0,
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
        // SAFETY: moe_router's signature (gate 1: sigmoid, 0: softmax); buffers sized for nb tokens.
        unsafe {
            g.launch(
                &f,
                (nb as u32, 1, 1),
                (128, 1, 1),
                (2 * n_e * 4) as u32,
                &[
                    lgd.arg(),
                    Arg::I32(n_e as i32),
                    Arg::I32(k as i32),
                    Arg::I32(sigm),
                    if use_bias { bd.arg() } else { Arg::Ptr(0) },
                    Arg::I32(norm),
                    Arg::F32(scale),
                    ids.arg(),
                    w.arg(),
                ],
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
    }
}
