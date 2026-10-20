//! The GPU twins of the CPU's expert numerics (moe_twin.cu, the TW products):
//! bit for bit the CPU's activation quantizer, `vmath::exp_f` /
//! `swiglu1_f::<F>`, and the r8 / r8q kernels' row sums.

use super::*;
use crate::quant::{QuantizedAct, r8, r8q};
use crate::vmath;

/// Bits of a float, every NaN as one value (the GPU's NaNs are canonical,
/// the CPU's keep the input's payload).
fn key(v: f32) -> u32 {
    if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() }
}

/// Indices where a and b differ in `key`, and the first few of them.
fn diffs(a: &[f32], b: &[f32]) -> (usize, Vec<(usize, f32, f32)>) {
    assert_eq!(a.len(), b.len());
    let d: Vec<usize> = (0..a.len()).filter(|&i| key(a[i]) != key(b[i])).collect();
    let first = d.iter().take(4).map(|&i| (i, a[i], b[i])).collect();
    (d.len(), first)
}

/// The GPU's quantized view of `x` (decode `quantize`, or batched
/// `quantize_b` of one row) as (Q8_0 scales, Q8_0 values, Q8_K scales, Q8_K
/// values, Q8_K sums of 32), the valid parts only.
type GpuViews = (Vec<f32>, Vec<i8>, Vec<f32>, Vec<i8>, Vec<i32>);

fn gpu_views(g: &Gpu, km: &Module, x: &[f32], batched: bool) -> GpuViews {
    let n = x.len();
    let np = n.next_multiple_of(256);
    let want_k = n.is_multiple_of(256);
    let st = act_stride_t(n);
    let act = g.alloc(st).unwrap();
    act.zero().unwrap();
    let xb = g.upload(x).unwrap();
    let mut args = vec![
        xb.arg(),
        Arg::I32(n as i32),
        Arg::I32(want_k as i32),
        act.arg(),
    ];
    if batched {
        args.push(Arg::U64(st as u64));
    }
    let f = km
        .function(if batched { "quantize_b" } else { "quantize" })
        .unwrap();
    // SAFETY: decode.cu quantize / prefill.cu quantize_b's signatures; act
    // holds act_bytes(n), x n floats.
    unsafe {
        g.launch(&f, (n.div_ceil(256) as u32, 1, 1), (32, 1, 1), 0, &args)
            .unwrap();
    }
    g.sync().unwrap();
    let b: Vec<u8> = act.read().unwrap();
    let f32s = |o: usize, c: usize| -> Vec<f32> {
        (0..c)
            .map(|i| f32::from_le_bytes(b[o + 4 * i..o + 4 * i + 4].try_into().unwrap()))
            .collect()
    };
    let i8s = |o: usize, c: usize| -> Vec<i8> { b[o..o + c].iter().map(|&v| v as i8).collect() };
    let (nk, n32) = (if want_k { n / 256 } else { 0 }, n / 32);
    let b32: Vec<i32> = (0..if want_k { n32 } else { 0 })
        .map(|i| {
            let o = 2 * np + np / 64 + 4 * i;
            i32::from_le_bytes(b[o..o + 4].try_into().unwrap())
        })
        .collect();
    (
        f32s(2 * np + np / 64 + np / 8, n32),
        i8s(np, n),
        f32s(2 * np, nk),
        i8s(0, if want_k { n } else { 0 }),
        b32,
    )
}

/// Vectors that exercise the quantizer's rules: random of several scales,
/// an all-zero block, a block of round-half ties (amax 127: x * 127/amax
/// lands on k + 0.5), the largest magnitude twice with both signs (the
/// first wins), a negative largest value, tiny and huge magnitudes.
fn quant_inputs(rng: &mut Rng, n: usize) -> Vec<Vec<f32>> {
    let mut v = vec![
        rng.vec(n, 1.0),
        rng.vec(n, 37.5),
        rng.vec(n, 1e-30),
        rng.vec(n, 3e37),
    ];
    let mut e = rng.vec(n, 2.0);
    e[..32].fill(0.0);
    for (i, x) in e[32..64].iter_mut().enumerate() {
        *x = i as f32 - 15.5;
    }
    e[40] = 127.0;
    if n >= 512 {
        // 256-block 1: ties for Q8_K's -127 / max scale (max = -127).
        for (i, x) in e[256..288].iter_mut().enumerate() {
            *x = 0.5 + i as f32 - 16.0;
        }
        e[300] = -127.0;
    }
    let mut t = rng.vec(n, 1.0);
    t[5] = 4.0;
    t[9] = -4.0;
    t[70] = -6.0;
    t[90] = 6.0;
    v.extend([e, t]);
    v
}

/// The GPU's activation quantizer (decode `quantize` and batched
/// `quantize_b`, both `quant_warp`) equals the CPU's
/// `QuantizedAct::for_types` bit for bit: Q8_0 scales and values, Q8_K
/// scales, values and sums (the GPU's sums of 32 = two CPU sums of 16).
#[test]
fn gpu_twin_quant_matches_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(0x7a1e_0000_0000_0001);
    let mut checked = 0;
    for n in [256, 1408, 2048, 4864, 224] {
        for x in quant_inputs(&mut rng, n) {
            let cpu = QuantizedAct::for_types(&x, &[Q8_0, Q4_K]);
            let (c0, ck) = cpu.views();
            let (cd0, cq0) = c0.expect("n % 32 == 0");
            for batched in [false, true] {
                let (d0, q0, dk, qk, b32) = gpu_views(&g, &km, &x, batched);
                let what = format!("n {n} batched {batched} scale {}", x[100].abs());
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&d0), bits(&cd0), "Q8_0 scales, {what}");
                assert_eq!(q0, cq0, "Q8_0 values, {what}");
                if let Some((cdk, cqk, cbs)) = &ck {
                    assert_eq!(bits(&dk), bits(cdk), "Q8_K scales, {what}");
                    assert_eq!(&qk, cqk, "Q8_K values, {what}");
                    let s32: Vec<i32> = cbs.chunks(2).map(|p| p[0] as i32 + p[1] as i32).collect();
                    assert_eq!(b32, s32, "Q8_K sums, {what}");
                } else {
                    assert!(qk.is_empty() && !n.is_multiple_of(256));
                }
                checked += 1;
            }
        }
    }
    eprintln!("twin quant: {checked} vectors, every view equal");
}

/// The twin inputs: every 4099th f32 bit pattern (both signs, all
/// exponents: subnormals, the clamps' neighbourhoods, overflow), the
/// clamp edges and specials, and a dense band of SiLU's working range.
fn twin_inputs(rng: &mut Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..u32::MAX / 4099)
        .map(|i| f32::from_bits(i * 4099))
        .collect();
    for b in [-87.3f32, 88.8, -86.99, 87.0, 88.72, -88.8, 87.3] {
        let u = b.to_bits();
        v.extend((0..64).map(|i| f32::from_bits(u - 32 + i)));
    }
    v.extend([
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1e-45,
    ]);
    v.extend((0..200_000).map(|_| rng.f() * 30.0));
    v
}

/// One-float-per-thread twin kernel `k` over xs (and us for swiglu).
fn gpu_twin(
    g: &Gpu,
    km: &Module,
    k: &str,
    xs: &[f32],
    us: Option<&[f32]>,
    fused: bool,
) -> Vec<f32> {
    let n = xs.len();
    let xb = g.upload(xs).unwrap();
    let ub = us.map(|u| g.upload(u).unwrap());
    let out = g.alloc(n * 4).unwrap();
    let f = km.function(k).unwrap();
    let mut args = vec![xb.arg()];
    if let Some(u) = &ub {
        args.push(u.arg());
    }
    args.extend([Arg::I32(n as i32), Arg::I32(fused as i32), out.arg()]);
    // SAFETY: moe_twin.cu's signatures; n floats in and out.
    unsafe {
        g.launch(&f, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &args)
            .unwrap();
    }
    g.sync().unwrap();
    out.read().unwrap()
}

/// exp_twin / swiglu_twin equal `vmath::exp_f::<F>` / `swiglu1_f::<F>` for
/// F true and false (the host passes its CPU's `fused()`), and with
/// F = fused() the slice form the CPU's experts call (`vmath::swiglu`: AVX2
/// `exp8` where the CPU has it), on every input class (NaN as NaN).
#[test]
fn gpu_twin_exp_swiglu_match_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(0x7a1e_0000_0000_0002);
    let xs = twin_inputs(&mut rng);
    let us: Vec<f32> = (0..xs.len())
        .map(|i| if i % 7 == 0 { 1.0 } else { rng.f() * 4.0 })
        .collect();
    for fused in [true, false] {
        let (ce, cs): (Vec<f32>, Vec<f32>) = if fused {
            (
                xs.iter().map(|&x| vmath::exp_f::<true>(x)).collect(),
                xs.iter()
                    .zip(&us)
                    .map(|(&x, &u)| vmath::swiglu1_f::<true>(x, u))
                    .collect(),
            )
        } else {
            (
                xs.iter().map(|&x| vmath::exp_f::<false>(x)).collect(),
                xs.iter()
                    .zip(&us)
                    .map(|(&x, &u)| vmath::swiglu1_f::<false>(x, u))
                    .collect(),
            )
        };
        let ge = gpu_twin(&g, &km, "exp_twin_k", &xs, None, fused);
        let gs = gpu_twin(&g, &km, "swiglu_twin_k", &xs, Some(&us), fused);
        let (n, first) = diffs(&ge, &ce);
        assert_eq!(
            n,
            0,
            "exp_twin F={fused}: {n} of {} differ, e.g. {first:?}",
            xs.len()
        );
        let (n, first) = diffs(&gs, &cs);
        assert_eq!(
            n,
            0,
            "swiglu_twin F={fused}: {n} of {} differ, e.g. {first:?}",
            xs.len()
        );
        if fused == vmath::fused() {
            let mut sl = xs.clone();
            vmath::swiglu(&mut sl, &us);
            let (n, first) = diffs(&gs, &sl);
            assert_eq!(
                n, 0,
                "swiglu_twin vs vmath::swiglu: {n} differ, e.g. {first:?}"
            );
        }
    }
    eprintln!(
        "twin exp/swiglu: {} inputs, F true and false, equal (host fused {})",
        xs.len(),
        vmath::fused()
    );
}

/// The CPU's expert kernels: the r8 (Q4_K, Q6_K) or r8q (Q5_0, Q8_0) repack of w
/// and out[r * nb + t] = row r . xs[t] (`gemm`, what `rows_batch` runs).
fn cpu_rows(qt: u32, w: &[u8], rows: usize, cols: usize, xs: &[QuantizedAct]) -> Vec<f32> {
    let refs: Vec<&QuantizedAct> = xs.iter().collect();
    let mut out = vec![0f32; rows * xs.len()];
    if matches!(qt, Q4_K | Q6_K) {
        let p = r8::repack(qt, w, rows, cols);
        r8::gemm(qt, &p, &refs, &mut out, xs.len());
    } else {
        let p = r8q::repack(qt, w, rows, cols);
        r8q::gemm(qt, &p, &refs, &mut out, xs.len());
    }
    out
}

/// `cpu_rows` for one token by `gemv` (the decode path).
fn cpu_row1(qt: u32, w: &[u8], rows: usize, cols: usize, x: &QuantizedAct) -> Vec<f32> {
    let mut out = vec![0f32; rows];
    if matches!(qt, Q4_K | Q6_K) {
        r8::gemv(qt, &r8::repack(qt, w, rows, cols), x, &mut out, 0);
    } else {
        r8q::gemv(qt, &r8q::repack(qt, w, rows, cols), x, &mut out, 0);
    }
    out
}

/// The expert shapes of every MoE family the resident path serves
/// (deepseek2 / Moonlight: gate/up Q4_K 1408 x 2048, down Q5_0 or Q8_0
/// 2048 x 1408; qwen3moe / Qwen3-30B-A3B Q4_K_M: gate/up Q4_K 768 x 2048,
/// down Q6_K 2048 x 768) and small ragged ones (rows not a multiple of the
/// GEMM's 64-row tile, widths of few blocks).
const TWIN_SHAPES: [(u32, usize, usize); 10] = [
    (Q4_K, 1408, 2048),
    (Q4_K, 768, 2048),
    (Q6_K, 2048, 768),
    (Q6_K, 200, 512),
    (Q4_K, 200, 768),
    (Q4_K, 8, 256),
    (Q5_0, 2048, 1408),
    (Q5_0, 72, 224),
    (Q8_0, 2048, 1408),
    (Q8_0, 136, 96),
];

/// gemv_tw equals the CPU's r8 / r8q `gemv` bit for bit (the decode
/// twin); the plain gemv1 (fused chain) doesn't, so the test has teeth.
#[test]
fn gpu_twin_gemv_matches_cpu_r8() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(0x7a1e_0000_0000_0003);
    let mut plain_diff = 0;
    for (qt, rows, cols) in TWIN_SHAPES {
        let w = synth_matrix(&mut rng, qt, rows, cols);
        for scale in [1.0, 50.0] {
            let x = rng.vec(cols, scale);
            let cpu = cpu_row1(qt, &w, rows, cols, &QuantizedAct::for_types(&x, &[qt]));
            let tw = gpu_gemv(&g, &km, "gemv_tw", qt, &w, cols, &x);
            let (n, first) = diffs(&tw, &cpu);
            assert_eq!(
                n, 0,
                "gemv_tw {qt} {rows}x{cols}: {n} rows differ, e.g. {first:?}"
            );
            plain_diff += diffs(&gpu_gemv(&g, &km, "gemv1", qt, &w, cols, &x), &cpu).0;
        }
    }
    assert!(
        plain_diff > 0,
        "the plain gemv equals the CPU everywhere: no teeth"
    );
    eprintln!("twin gemv: equal on every shape; plain gemv1 differs on {plain_diff} rows");
}

/// gemm2s_*_tw (tensor cores, sm_80+) equals the CPU's r8 / r8q `gemm`
/// bit for bit for every token of batches of 1, 7, 48, 129 and 1024, and
/// the same tokens' `gemv` (decode == batch on the twin); the plain
/// gemm2s doesn't.
#[test]
fn gpu_twin_gemm2s_matches_cpu_r8() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    if g.info().unwrap().cc.0 < 8 {
        eprintln!("skipping: gemm2s needs sm_80+ (twin GEMMs are refused below)");
        return;
    }
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(0x7a1e_0000_0000_0004);
    let mut plain_diff = 0;
    for (qt, rows, cols) in TWIN_SHAPES {
        let w = synth_matrix(&mut rng, qt, rows, cols);
        let wb = g.upload(&w).unwrap();
        // Q6_K: the twin on the raw blocks and on the prefill copy (Q6K8).
        let w8 = (qt == Q6_K).then(|| q6k8_copy(&g, &km, &wb, rows, cols));
        let kernel = format!("gemm2s_{}_tw", type_name(qt));
        for nb in [1, 7, 48, 129, 1024] {
            if nb == 1024 && rows < 1000 {
                continue;
            }
            let xs: Vec<Vec<f32>> = (0..nb)
                .map(|t| rng.vec(cols, if t % 3 == 0 { 40.0 } else { 1.0 }))
                .collect();
            let qa: Vec<QuantizedAct> = xs
                .iter()
                .map(|x| QuantizedAct::for_types(x, &[qt]))
                .collect();
            let cpu = cpu_rows(qt, &w, rows, cols, &qa);
            let act = quantize_batch(&g, &km, &xs, cols);
            let out = gpu_batched(
                &g,
                &km,
                &kernel,
                qt,
                &wb,
                rows,
                cols,
                &act,
                nb,
                &Epi::default(),
            );
            // out is token-major (out[t * rows + r]); cpu row-major.
            let gt: Vec<f32> = (0..rows * nb)
                .map(|i| out[(i % nb) * rows + i / nb])
                .collect();
            let (n, first) = diffs(&gt, &cpu);
            if let Some(w8) = &w8 {
                let k8 = "gemm2s_q6k8_tw";
                let o = gpu_batched(&g, &km, k8, qt, w8, rows, cols, &act, nb, &Epi::default());
                let t8: Vec<f32> = (0..rows * nb)
                    .map(|i| o[(i % nb) * rows + i / nb])
                    .collect();
                let (n8, f8) = diffs(&t8, &cpu);
                assert_eq!(
                    n8, 0,
                    "{k8} {rows}x{cols} nb {nb}: {n8} differ, e.g. {f8:?}"
                );
            }
            assert_eq!(
                n, 0,
                "{kernel} {rows}x{cols} nb {nb}: {n} differ, e.g. {first:?}"
            );
            if nb == 48 {
                let plain = format!("gemm2s_{}", type_name(qt));
                let o = gpu_batched(
                    &g,
                    &km,
                    &plain,
                    qt,
                    &wb,
                    rows,
                    cols,
                    &act,
                    nb,
                    &Epi::default(),
                );
                let pt: Vec<f32> = (0..rows * nb)
                    .map(|i| o[(i % nb) * rows + i / nb])
                    .collect();
                plain_diff += diffs(&pt, &cpu).0;
            }
            if nb <= 7 {
                let t = nb - 1;
                let one = cpu_row1(qt, &w, rows, cols, &qa[t]);
                let col: Vec<f32> = (0..rows).map(|r| cpu[r * nb + t]).collect();
                assert_eq!(diffs(&one, &col).0, 0, "CPU gemm != gemv");
            }
        }
    }
    assert!(
        plain_diff > 0,
        "the plain gemm2s equals the CPU everywhere: no teeth"
    );
    eprintln!("twin gemm2s: equal for every shape and batch; plain gemm2s differs on {plain_diff}");
}
