//! FP8 (fp8.cu, docs/design/fp8-weights.md §3, §6 B-D): each kernel against
//! its bits reference and the independent f64 anchor, and the FP8 0.5B
//! checkpoint end to end: exact mode's Q8_0 form keeps the contract's bits,
//! fast mode's native kernels are graded against the CPU's fp8-ref.

use super::numerics::{Arm, Floor, Sizes, assert_graded_vs_exact_with, grade_sized};
use super::*;
use crate::cuda::Kernels;
use crate::kv::KvType;
use crate::lm::{Backend, BackendOptions, Device};
use crate::numerics::Numerics;
use crate::quant::fp8::{
    self, F8S_TOK, check_anchor, e4m3_to_f32, f8_gemm_f64, f32_to_e4m3, gemv_f8_ref,
    quantize_act_f8, test_codes,
};

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 11
}

/// Random floats over many binades, both signs.
fn floats(n: usize, s: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let r = lcg(s);
            let m = (r & 0xffff) as f32 / 65536.0 + 0.5;
            let e = ((r >> 16) % 40) as i32 - 20;
            let v = m * 2f32.powi(e);
            if r >> 60 & 1 == 1 { -v } else { v }
        })
        .collect()
}

/// A product kernel's outputs ([nb][rows]) for weights `w` ([rows][k] codes,
/// scales `ws`) and tokens `qa` ([nb][k] codes, scales `das`).
#[allow(clippy::too_many_arguments)]
fn product(
    g: &Gpu,
    km: &Module,
    kernel: &str,
    w: &[u8],
    ws: &[f32],
    k: usize,
    qa: &[u8],
    das: &[f32],
) -> Vec<f32> {
    let (rows, nb) = (ws.len(), das.len());
    let f = km.function(kernel).unwrap();
    let (grid, block, smem) = match kernel {
        "gemv_f8" => ((rows.div_ceil(8), 1), 256, nb * 512 * 4),
        "gemm_f8" => ((nb.div_ceil(64), rows.div_ceil(64)), 256, 2 * 2 * 64 * 144),
        "gemm_ref_f8" => ((nb.div_ceil(8), rows.div_ceil(16)), 32, 0),
        _ => unreachable!(),
    };
    let (wd, wsd, qd, dd) = (
        g.upload(w).unwrap(),
        g.upload(ws).unwrap(),
        g.upload(qa).unwrap(),
        g.upload(das).unwrap(),
    );
    let out = g.alloc(rows * nb * 4).unwrap();
    out.zero().unwrap();
    // SAFETY: the kernels' signature (fp8.cu); buffers of the sizes above.
    unsafe {
        g.launch(
            &f,
            (grid.0 as u32, grid.1 as u32, 1),
            (block, 1, 1),
            smem as u32,
            &[
                wd.arg(),
                Arg::I32(rows as i32),
                Arg::I32(k as i32),
                qd.arg(),
                dd.arg(),
                Arg::I32(nb as i32),
                wsd.arg(),
                Arg::Ptr(0),
                Arg::I32(0),
                out.arg(),
                Arg::U64(rows as u64),
            ],
        )
        .unwrap();
    }
    g.sync().unwrap();
    out.read::<f32>().unwrap()
}

/// Weights and tokens for a product: random finite codes, scales of a
/// checkpoint's size.
fn operands(
    rows: usize,
    k: usize,
    nb: usize,
    s: &mut u64,
) -> (Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>) {
    let w = test_codes(rows * k, s);
    let ws = (0..rows)
        .map(|r| 1e-3 * (1.0 + (r % 13) as f32 / 5.0))
        .collect();
    let qa = test_codes(nb * k, s);
    let das = (0..nb)
        .map(|t| 2e-3 * (1.0 + (t % 7) as f32 / 3.0))
        .collect();
    (w, ws, qa, das)
}

/// `f8_gemm_f64` for a sample of outputs (all when few): (index, E, tol).
fn anchor_sample(
    w: &[u8],
    ws: &[f32],
    k: usize,
    qa: &[u8],
    das: &[f32],
    s: &mut u64,
) -> Vec<(usize, (f64, f64))> {
    let (rows, nb) = (ws.len(), das.len());
    let n = rows * nb;
    let picks: Vec<usize> = if n <= 4096 {
        (0..n).collect()
    } else {
        (0..2048).map(|_| lcg(s) as usize % n).collect()
    };
    picks
        .into_iter()
        .map(|i| {
            let (t, r) = (i / rows, i % rows);
            let e = f8_gemm_f64(
                &w[r * k..(r + 1) * k],
                &ws[r..=r],
                k,
                &qa[t * k..(t + 1) * k],
                &das[t..=t],
            );
            (i, e[0])
        })
        .collect()
}

fn check_sample(what: &str, got: &[f32], sample: &[(usize, (f64, f64))], slack: f64) {
    let g: Vec<f32> = sample.iter().map(|&(i, _)| got[i]).collect();
    let w: Vec<(f64, f64)> = sample.iter().map(|&(_, e)| e).collect();
    if let Err(e) = check_anchor(&g, &w, slack) {
        panic!("{what}: {e}");
    }
}

/// The quantizer's GPU half (fp8.cu f8x2 and quant_f8_tok) against the Rust
/// twin, bit for bit: every e4m3 code's value, the midpoints between codes
/// and their neighbours, 10^6 random floats; then whole rows (K 896, half a
/// 256-block at the end; 4864; a zero row; a row of subnormal codes).
/// The test GPU if it has e4m3 tensor cores (sm_89 and later).
fn f8_gpu() -> Option<crate::cuda::Gpu> {
    let g = crate::testutil::gpu()?;
    if !crate::cuda::f8_native_cc(&g).unwrap() {
        eprintln!("skipping: no e4m3 tensor cores (sm_89 and later) on this GPU");
        return None;
    }
    Some(g)
}

#[test]
fn gpu_fp8_quantizer_matches_twin() {
    let Some(g) = f8_gpu() else {
        return;
    };
    let km = Kernels::module_f8(&g).unwrap();
    let mut s = 1u64;
    let mut x = floats(1_000_000, &mut s);
    for c in 0..0x7eu8 {
        let (lo, hi) = (e4m3_to_f32(c), e4m3_to_f32(c + 1));
        let mid = (lo + hi) / 2.0;
        for v in [
            lo,
            mid,
            f32::from_bits(mid.to_bits() + 1),
            f32::from_bits(mid.to_bits() - 1),
        ] {
            x.extend([v, -v]);
        }
    }
    x.extend([
        448.0,
        464.0,
        1e30,
        f32::INFINITY,
        -f32::INFINITY,
        0.0,
        -0.0,
        f32::from_bits(1),
    ]);
    if x.len() % 2 == 1 {
        x.push(0.0);
    }
    let f = km.function("f8_cvt").unwrap();
    let (xd, out) = (g.upload(&x).unwrap(), g.alloc(x.len()).unwrap());
    // SAFETY: f8_cvt's signature; x.len() even, out x.len() bytes.
    unsafe {
        g.launch(
            &f,
            ((x.len() / 2).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &[xd.arg(), Arg::I32(x.len() as i32), out.arg()],
        )
        .unwrap();
    }
    g.sync().unwrap();
    let got = out.read::<u8>().unwrap();
    for (i, (&v, &c)) in x.iter().zip(&got).enumerate() {
        assert_eq!(c, f32_to_e4m3(v), "cvt of {v:e} (#{i})");
    }
    // Rows through quant_f8_tok: f32 values at act_f8_off in act rows.
    let fq = km.function("quant_f8_tok").unwrap();
    for n in [896usize, 4864, 128] {
        let mut rows = vec![floats(n, &mut s), vec![0.0; n], floats(n, &mut s)];
        rows[2].iter_mut().for_each(|v| *v *= 1e-30);
        rows.push(
            (0..n)
                .map(|i| {
                    if i == 5 {
                        1000.0
                    } else {
                        (i % 7) as f32 * 1e-3
                    }
                })
                .collect(),
        );
        let st = super::super::act_bytes_f8(n).next_multiple_of(256);
        let mut host = vec![0u8; st * rows.len()];
        for (z, r) in rows.iter().enumerate() {
            let o = z * st + super::super::act_f8_off(n);
            for (i, v) in r.iter().enumerate() {
                host[o + 4 * i..o + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let act = g.upload(&host).unwrap();
        let qa = g.alloc(rows.len() * n).unwrap();
        let das = g.alloc(rows.len() * 4).unwrap();
        // SAFETY: quant_f8_tok's signature; act rows of st bytes, qa nb x n, das nb.
        unsafe {
            g.launch(
                &fq,
                (rows.len() as u32, 1, 1),
                (256, 1, 1),
                0,
                &[
                    act.arg(),
                    Arg::U64(st as u64),
                    Arg::I32(n as i32),
                    qa.arg(),
                    das.arg(),
                ],
            )
            .unwrap();
        }
        g.sync().unwrap();
        let (gq, gd) = (qa.read::<u8>().unwrap(), das.read::<f32>().unwrap());
        for (z, r) in rows.iter().enumerate() {
            let (q, sc) = quantize_act_f8(r);
            assert_eq!(gd[z].to_bits(), sc.to_bits(), "n {n} row {z}: scale");
            assert_eq!(&gq[z * n..(z + 1) * n], &q[..], "n {n} row {z}: codes");
        }
    }
}

/// gemv_f8 == quant::fp8::gemv_f8_ref bit for bit, and within the anchor,
/// at the design's shapes (§3).
#[test]
fn gpu_fp8_gemv_bits_and_anchor() {
    let Some(g) = f8_gpu() else {
        return;
    };
    let km = Kernels::module_f8(&g).unwrap();
    let mut s = 2u64;
    for k in [128usize, 896, 4864] {
        for rows in [64usize, 128, 576, 896, 4864] {
            for nb in [1usize, 2, 3, 8, 16] {
                let (w, ws, qa, das) = operands(rows, k, nb, &mut s);
                let got = product(&g, &km, "gemv_f8", &w, &ws, k, &qa, &das);
                let want = gemv_f8_ref(&w, &ws, k, &qa, &das);
                let bad = got
                    .iter()
                    .zip(&want)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert!(
                    bad.is_none(),
                    "gemv_f8 k {k} rows {rows} nb {nb}: output {bad:?} {:?}",
                    bad.map(|i| (got[i], want[i]))
                );
                let sample = anchor_sample(&w, &ws, k, &qa, &das, &mut s);
                check_sample(
                    &format!("gemv_f8 k {k} rows {rows} nb {nb}"),
                    &got,
                    &sample,
                    1.0,
                );
            }
        }
    }
}

/// gemm_f8 == gemm_ref_f8 bit for bit (tiling, staging, ragged rows and
/// tokens), both within the anchor (the mma's own semantics); a token's
/// outputs the same bits in every n position among other tokens as alone;
/// gemv_f8 at nb 16 and gemm_f8 at nb 17 (those 16 + 1) within 2x the
/// tolerance of each other.
#[test]
fn gpu_fp8_gemm_bits_and_anchor() {
    let Some(g) = f8_gpu() else {
        return;
    };
    let km = Kernels::module_f8(&g).unwrap();
    let mut s = 3u64;
    for k in [128usize, 896, 4864] {
        for rows in [64usize, 128, 576, 1024] {
            for nb in [17usize, 24, 64, 65, 128, 160, 520, 1024] {
                let (w, ws, qa, das) = operands(rows, k, nb, &mut s);
                let got = product(&g, &km, "gemm_f8", &w, &ws, k, &qa, &das);
                let rf = product(&g, &km, "gemm_ref_f8", &w, &ws, k, &qa, &das);
                let bad = got
                    .iter()
                    .zip(&rf)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert!(
                    bad.is_none(),
                    "gemm_f8 k {k} rows {rows} nb {nb}: output {bad:?} {:?}",
                    bad.map(|i| (got[i], rf[i]))
                );
                let sample = anchor_sample(&w, &ws, k, &qa, &das, &mut s);
                check_sample(
                    &format!("gemm_ref_f8 k {k} rows {rows} nb {nb}"),
                    &rf,
                    &sample,
                    1.0,
                );
            }
        }
    }
    // Neighbours: token x alone (gemm_ref_f8, nb 1: 7 zero columns) vs at
    // each position of a 65-token gemm_f8 among random tokens.
    let (k, rows) = (896, 128);
    let (w, ws, x, dx) = operands(rows, k, 1, &mut s);
    let alone = product(&g, &km, "gemm_ref_f8", &w, &ws, k, &x, &dx);
    for p in [0usize, 1, 7, 8, 15, 31, 63, 64] {
        let (_, _, mut qa, mut das) = operands(rows, k, 65, &mut s);
        qa[p * k..(p + 1) * k].copy_from_slice(&x);
        das[p] = dx[0];
        let got = product(&g, &km, "gemm_f8", &w, &ws, k, &qa, &das);
        for r in 0..rows {
            assert_eq!(
                got[p * rows + r].to_bits(),
                alone[r].to_bits(),
                "position {p} row {r}"
            );
        }
    }
    // gemv at 16 vs gemm at 17.
    for k in [896usize, 4864] {
        let (w, ws, qa, das) = operands(576, k, 17, &mut s);
        let mm = product(&g, &km, "gemm_f8", &w, &ws, k, &qa, &das);
        let mv = product(&g, &km, "gemv_f8", &w, &ws, k, &qa[..16 * k], &das[..16]);
        let want: Vec<(f64, f64)> = f8_gemm_f64(&w, &ws, k, &qa[..16 * k], &das[..16]);
        let diff: Vec<(f64, f64)> = mm[..16 * 576]
            .iter()
            .zip(&want)
            .map(|(&a, &(_, tol))| (a as f64, 2.0 * tol))
            .collect();
        check_anchor(&mv, &diff, 1.0).unwrap_or_else(|e| panic!("gemv 16 vs gemm 17, k {k}: {e}"));
    }
}

/// The FP8 0.5B checkpoint, loaded (None: skipped).
fn fp8_model() -> Option<Model> {
    let p = crate::testutil::fp8_model_path()?;
    Some(crate::gguf::load_with(p.to_str().unwrap(), Some(false)).unwrap())
}

/// Exact mode runs the checkpoint's Q8_0 form on today's kernels (D1): the
/// contract's batched-prefill bits hold, and no fast kernel runs.
#[test]
fn gpu_fp8_exact_is_q8_and_bit_identical() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(m) = fp8_model() else {
        return;
    };
    prefill_bits(&m, "0.5B FP8 (exact: Q8_0)", CacheKv::F16);
    let o = BackendOptions::default()
        .device(Device::Cuda)
        .max_ctx(512)
        .numerics(Numerics::Exact);
    let b = Backend::new(&m, o).unwrap();
    assert!(!b.fast_kernels());
}

/// The #173 gate in its FP8 form (§6 B, ruling 2): the GPU's native FP8
/// (fast) against the CPU's fp8-ref (exact dequant, f32 activations), the
/// floor the CPU's fp8-ref with the fast path's activation rounding
/// (e4m3 per token): the same precision change in the CPU's order. Every
/// arm graded, and a fast kernel seen in each (decode on a fresh net: its
/// steps are batched passes of one row; the batched arm's prefixes all
/// past the GEMV's 16 rows). At >= 1000 positions per arm in a release
/// build (`Sizes::FP8`, the binding run of record for the statistical
/// bars); a debug-assertions build runs every arm and route on fewer
/// (`Sizes::FP8_DEBUG`: the lock timeout) for the checks only it makes,
/// with the sanity backstops and no statistical grading
/// (`assert_sane_vs_exact`: lead ruling, 10-31, after 1031-115732-b712).
#[test]
fn gpu_fp8_graded_vs_fp8_ref() {
    if f8_gpu().is_none() {
        return;
    }
    let Some(p) = crate::testutil::fp8_model_path() else {
        return;
    };
    let load = || crate::gguf::load_with(p.to_str().unwrap(), Some(false)).unwrap();
    let (mr, mf, mo) = (load(), load(), load());
    let net = |m, d, n| {
        let o = BackendOptions::default()
            .device(d)
            .max_ctx(super::numerics::CTX)
            .kv(KvType::F16)
            .numerics(n);
        Backend::new(m, o).unwrap()
    };
    let r = net(&mr, Device::Cpu, Numerics::Exact);
    let f = crate::model::with_fp8_cpu_qt(F8S_TOK, || crate::model::Qwen2::new_with(&mf, true));
    let o = net(&mo, Device::Cuda, Numerics::Fast);
    assert!(o.fast_kernels(), "fast mode on this GPU runs FP8 natively");
    for &n in super::numerics::prefixes(Device::Cuda) {
        assert_eq!(
            super::batch::f8_route(n, false),
            super::batch::F8Route::MmaSync,
            "prefix {n}"
        );
    }
    let (sz, min) = match cfg!(debug_assertions) {
        false => (Sizes::FP8, 1000),
        true => (Sizes::FP8_DEBUG, 98),
    };
    assert!(super::batch::ATT_NB_MIN <= sz.load);
    let g = grade_sized(&r, &f, &o, &mr, Floor::Fp8Emu, Device::Cuda, &sz);
    let arms: Vec<_> = g.arms.iter().map(|a| a.arm).collect();
    assert_eq!(arms, [Arm::Decode, Arm::Batched, Arm::Cached, Arm::Load]);
    for a in &g.arms {
        assert!(a.launches > 0, "[{}]: no FP8 kernel ran", a.arm.name());
    }
    for a in &g.arms {
        assert!(
            a.mode.positions >= min,
            "[{}]: {} positions",
            a.arm.name(),
            a.mode.positions
        );
    }
    let what = "0.5B FP8 cuda fast vs fp8-ref";
    match cfg!(debug_assertions) {
        false => assert_graded_vs_exact_with(what, &g, false, Top1::Binomial),
        true => super::numerics::assert_sane_vs_exact(what, &g),
    }
    let _ = fp8::F8S;
}

/// The load arm's max |dlogit|, diagnosed (`numerics::load_diag`): run by
/// hand (`--ignored`), on bella.
#[test]
#[ignore]
fn gpu_fp8_load_diag() {
    if f8_gpu().is_none() {
        return;
    }
    let Some(p) = crate::testutil::fp8_model_path() else {
        return;
    };
    let load = || crate::gguf::load_with(p.to_str().unwrap(), Some(false)).unwrap();
    let (mr, mf, mo) = (load(), load(), load());
    let net = |m, d, n| {
        let o = BackendOptions::default()
            .device(d)
            .max_ctx(super::numerics::CTX)
            .kv(KvType::F16)
            .numerics(n);
        Backend::new(m, o).unwrap()
    };
    let r = net(&mr, Device::Cpu, Numerics::Exact);
    let f = crate::model::with_fp8_cpu_qt(F8S_TOK, || crate::model::Qwen2::new_with(&mf, true));
    let o = net(&mo, Device::Cuda, Numerics::Fast);
    super::numerics::load_diag(&r, &f, &o, &mr, &Sizes::FP8);
}
