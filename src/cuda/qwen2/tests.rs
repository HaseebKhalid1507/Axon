use super::*;
use crate::cuda::Kernels;
use crate::model::Qwen2;
use std::path::PathBuf;

fn model_at(rel: &str) -> Option<Model> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    if !p.exists() {
        eprintln!("skipping: {} not present", p.display());
        return None;
    }
    Some(crate::gguf::load_with(p.to_str().unwrap(), Some(false)).unwrap())
}

/// log-softmax in f64.
fn log_softmax(l: &[f32]) -> Vec<f64> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = l.iter().map(|&x| (x as f64 - m).exp()).sum();
    l.iter().map(|&x| x as f64 - m - s.ln()).collect()
}

/// KL(p || q) of two logit vectors (p the reference).
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
}

#[derive(Default, Debug)]
struct Agree {
    positions: usize,
    top1: usize,
    max_kl: f64,
    sum_kl: f64,
    max_abs: f32,
    /// Reference logit margin of each top-1 disagreement.
    flips: Vec<f32>,
}

/// Reference `r` vs `o` logits after the prompt `ids` and after each of
/// `steps` greedy tokens of `r`, fed to both.
fn compare<R: Lm, O: Lm>(r: &R, o: &O, ids: &[u32], steps: usize, a: &mut Agree) {
    let mut rc = r.new_cache(ids.len() + steps + 1);
    let mut oc = o.new_cache(ids.len() + steps + 1);
    let (&last, rest) = ids.split_last().unwrap();
    r.feed_batch(rest, 0, &mut rc);
    o.feed_batch(rest, 0, &mut oc);
    let mut tok = last;
    for s in 0..=steps {
        let pos = rest.len() + s;
        let lr = r.forward(tok, pos, &mut rc);
        let lo = o.forward(tok, pos, &mut oc);
        let (ar, ao) = (crate::argmax(&lr), crate::argmax(&lo));
        a.positions += 1;
        a.top1 += (ar == ao) as usize;
        let k = kl(&lr, &lo);
        a.max_kl = a.max_kl.max(k);
        a.sum_kl += k;
        for (x, y) in lr.iter().zip(&lo) {
            a.max_abs = a.max_abs.max((x - y).abs());
        }
        if ar != ao {
            // How close a call it was, in the reference's logits.
            a.flips.push(lr[ar as usize] - lr[ao as usize]);
        }
        tok = ar;
    }
}

const PROMPTS: [&str; 6] = [
    "The capital of France is",
    "fn add(a: i32, b: i32) -> i32 { a + b }\nfn sub(a: i32, b: i32) -> i32 {",
    "Once upon a time, in a small village by the sea, there lived",
    "Q: What is 17 times 23?\nA:",
    "The three laws of thermodynamics are",
    "<|im_start|>user\nWrite a haiku about autumn.<|im_end|>\n<|im_start|>assistant\n",
];

/// (GPU vs CPU, CPU per-row kernels vs CPU default kernels) over PROMPTS
/// with `steps` greedy steps each.
fn agreement(m: &Model, steps: usize) -> (Agree, Agree) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    // GPU and per-row first: the default CPU model releases the bytes it
    // repacks.
    let gpu = CudaQwen2::new(&g, &km, m, 256).unwrap();
    let rows = Qwen2::new_with(m, false);
    let cpu = Qwen2::new(m);
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    for p in PROMPTS {
        let ids = tok.encode(p);
        compare(&cpu, &gpu, &ids, steps, &mut ag);
        compare(&cpu, &rows, &ids, steps, &mut ac);
    }
    (ag, ac)
}

fn report(name: &str, a: &Agree) {
    eprintln!(
        "{name}: top-1 {}/{} ({:.1}%), KL mean {:.2e} max {:.2e}, max |dlogit| {:.3}, \
         flip margins {:?}",
        a.top1,
        a.positions,
        100.0 * a.top1 as f64 / a.positions as f64,
        a.sum_kl / a.positions as f64,
        a.max_kl,
        a.max_abs,
        a.flips
    );
}

/// GPU logits vs the CPU's after each prompt and 32 greedy steps, graded
/// against the CPU's own spread between its two kernel sets (repacked r8 vs
/// per-row: the same quantized math summed in other orders, like the GPU).
///
/// Why relative: both backends quantize every activation to 8 bits, and
/// Qwen2's few huge residual dimensions make those blocks' steps coarse, so
/// a last-bit difference before a quantization occasionally flips one
/// rounding, which the next layers carry on. Measured (6 prompts x 33
/// positions), GPU vs CPU / CPU per-row vs CPU: 0.5B top-1 97.5% / 97.0%,
/// mean KL 2.9e-3 / 2.8e-3, max KL 0.083 / 0.030, max |dlogit| 2.8 / 1.5;
/// 1.5B 95.5% / 97.0%, 6.6e-3 / 7.8e-3, 0.16 / 0.34, 3.3 / 3.5. Every
/// disagreement is a near-tie (reference margin <= 0.41 logits, most
/// < 0.2), often the same positions in both comparisons. A 99% top-1 bar
/// would fail the CPU against itself.
///
/// Asserted: top-1 within 2 points of the CPU's self-agreement (and >= 94%);
/// only near-ties (< 0.5) flip; mean KL within 2x the CPU's own. The
/// extremes are single positions, as heavy-tailed CPU vs CPU as GPU vs CPU,
/// so they get absolute gross-bug bounds (a wrong kernel gives KL ~1-10 and
/// logit errors ~10): max KL < 0.5, max |dlogit| < 6.
fn logits_match(m: &Model, name: &str) {
    let (ag, ac) = agreement(m, 32);
    report(&format!("{name} gpu vs cpu"), &ag);
    report(&format!("{name} cpu rows vs cpu"), &ac);
    let n = ag.positions;
    // Top-1: at most 2% of positions behind the CPU's self-agreement.
    assert!(
        ag.top1 * 100 + 2 * n >= ac.top1 * 100 && ag.top1 * 100 >= 94 * n,
        "top-1 agreement {}/{n} (cpu vs itself {})",
        ag.top1,
        ac.top1
    );
    // Only near-ties flip.
    assert!(
        ag.flips.iter().all(|&d| d < 0.5),
        "a top-1 flip with margin >= 0.5: {:?}",
        ag.flips
    );
    assert!(ag.sum_kl <= 2.0 * ac.sum_kl, "mean KL");
    assert!(ag.max_kl < 0.5, "max KL");
    assert!(ag.max_abs < 6.0, "max logit difference");
}

#[test]
fn gpu_logits_match_cpu_0_5b() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    // Mapped: the per-row CPU model reads the file after the default one
    // releases its copy of the bytes it repacks.
    let Some(m) = crate::testutil::model_with(Some(false)) else {
        return;
    };
    logits_match(&m, "0.5B");
}

#[test]
fn gpu_logits_match_cpu_1_5b() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    logits_match(&m, "1.5B");
}

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

/// `quantize` then `gemv` (mode 0) on the GPU: W x for a matrix of type qt.
fn gpu_gemv(
    g: &Gpu,
    km: &Module,
    kernel: &str,
    qt: u32,
    w: &[u8],
    cols: usize,
    x: &[f32],
) -> Vec<f32> {
    gpu_gemv_epi(g, km, kernel, qt, w, cols, x, &Epi::default())
}

/// The products' epilogue: bias (or none), and with `init` the residual
/// add (mode 1: out = init + (W x + bias)).
#[derive(Default)]
struct Epi {
    bias: Option<Vec<f32>>,
    init: Option<Vec<f32>>,
}

/// `gpu_gemv` with an epilogue.
#[allow(clippy::too_many_arguments)]
fn gpu_gemv_epi(
    g: &Gpu,
    km: &Module,
    kernel: &str,
    qt: u32,
    w: &[u8],
    cols: usize,
    x: &[f32],
    epi: &Epi,
) -> Vec<f32> {
    let per = if kernel == "gemv" { 2 } else { 1 };
    let rows = w.len() / quant::row_bytes(qt, cols);
    let wb = g.upload(w).unwrap();
    let xb = g.upload(x).unwrap();
    let act = g.alloc(act_bytes(cols)).unwrap();
    let out = match &epi.init {
        Some(v) => g.upload(v).unwrap(),
        None => g.alloc(rows * 4).unwrap(),
    };
    let bias = epi.bias.as_ref().map(|b| g.upload(b).unwrap());
    let bias_arg = bias.as_ref().map_or(Arg::Ptr(0), |b| b.arg());
    let (fq, fg) = (
        km.function("quantize").unwrap(),
        km.function(kernel).unwrap(),
    );
    // SAFETY: decode.cu's signatures; buffers sized for cols inputs, rows outputs.
    unsafe {
        g.launch(
            &fq,
            (cols.div_ceil(256) as u32, 1, 1),
            (32, 1, 1),
            0,
            &[
                xb.arg(),
                Arg::I32(cols as i32),
                Arg::I32(cols.is_multiple_of(256) as i32),
                act.arg(),
            ],
        )
        .unwrap();
        g.launch(
            &fg,
            (rows.div_ceil(per) as u32, 1, 1),
            (GEMV_THREADS, 1, 1),
            0,
            &[
                Arg::I32(qt as i32),
                wb.arg(),
                Arg::U64(quant::row_bytes(qt, cols) as u64),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
                act.arg(),
                bias_arg,
                Arg::I32(epi.init.is_some() as i32),
                out.arg(),
            ],
        )
        .unwrap();
    }
    out.read().unwrap()
}

/// GEMV of every weight type on real matrices (both models' shapes) and
/// random inputs: vs the CPU's dot_row_q8 on the same quantized
/// activation (equal integer block sums: only the order blocks are added in
/// differs), and vs an f64 dot of the dequantized row with x itself (the
/// activation quantization error, bounded loosely).
#[test]
fn gpu_gemv_matches_cpu_every_type() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut seen = Vec::new();
    for name in [
        "blk.0.attn_q.weight", // Q5_0, 896 wide
        "blk.0.attn_v.weight", // Q8_0 or Q5_0
        "blk.1.attn_v.weight",
        "blk.0.ffn_down.weight", // Q6_K or Q4_K, 4864 wide
        "blk.1.ffn_down.weight",
        "blk.2.ffn_down.weight",
        "output.weight", // Q8_0, 151936 rows
    ] {
        let t = tensor(&m, name).unwrap();
        let (qt, cols) = (t.qtype.id, t.dims[0] as usize);
        let rb = quant::row_bytes(qt, cols);
        // At most 4096 rows (the head: its first ones).
        let rows = (t.n_elements as usize / cols).min(4095); // odd: a lone last row
        let w = &m.tensor_data(t)[..rows * rb];
        for scale in [1.0f32, 1e-3, 40.0] {
            let x = rng.vec(cols, scale);
            let got = gpu_gemv(&g, &km, "gemv", qt, w, cols, &x);
            // One row per block: the same sums.
            let got1 = gpu_gemv(&g, &km, "gemv1", qt, w, cols, &x);
            assert_eq!(bits(&got), bits(&got1), "{name}: gemv1 vs gemv");
            let xq = quant::QuantizedAct::for_types(&x, &[qt]);
            let (mut worst, mut worst_f) = (0f64, 0f64);
            let mut row = vec![0f32; cols];
            for r in 0..rows {
                let wr = &w[r * rb..(r + 1) * rb];
                let cpu = quant::dot_row_q8(qt, wr, &xq) as f64;
                quant::dequant(qt, wr, &mut row);
                let exact: f64 = row.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum();
                let mag: f64 = row
                    .iter()
                    .zip(&x)
                    .map(|(&a, &b)| (a as f64 * b as f64).abs())
                    .sum();
                let d = got[r] as f64;
                worst = worst.max((d - cpu).abs() / mag.max(1e-30));
                worst_f = worst_f.max((d - exact).abs() / mag.max(1e-30));
            }
            eprintln!(
                "{name} {} x{scale}: vs cpu q8 {worst:.2e}, vs f64 exact {worst_f:.2e} (of sum |w x|)",
                crate::gguf::qtype(qt).name
            );
            // Same block values summed in another order: f32 rounding only.
            assert!(worst < 1e-6, "{name}: {worst:e} vs the CPU");
            // 8-bit activations: ~1/254 relative per element at worst.
            assert!(worst_f < 1e-2, "{name}: {worst_f:e} vs exact");
        }
        seen.push(qt);
    }
    for qt in [Q4_K, Q6_K, Q5_0, Q8_0] {
        assert!(seen.contains(&qt), "type {qt} not covered");
    }
}

/// Bits of a vector, for exact comparisons.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// rmsnorm_quant's f32 output vs the CPU's RMSNorm and an f64 one, for the
/// models' widths and odd ones, normal and tiny/huge magnitudes.
#[test]
fn gpu_rmsnorm_matches_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let f = km.function("rmsnorm_quant").unwrap();
    let mut rng = Rng(77);
    for n in [32, 96, 896, 1536, 2048, 3584, 8960, 18944] {
        for scale in [1.0f32, 1e-4, 300.0] {
            let x = rng.vec(n, scale);
            let w = rng.vec(n, 2.0);
            let eps = 1e-6f32;
            let (xb, wb) = (g.upload(&x).unwrap(), g.upload(&w).unwrap());
            let act = g.alloc(act_bytes(n)).unwrap();
            let y = g.alloc(n * 4).unwrap();
            // SAFETY: rmsnorm_quant's signature; buffers of n floats / act_bytes(n).
            unsafe {
                g.launch(
                    &f,
                    (1, 1, 1),
                    (1024, 1, 1),
                    0,
                    &[
                        xb.arg(),
                        wb.arg(),
                        Arg::F32(eps),
                        Arg::I32(n as i32),
                        Arg::I32(n.is_multiple_of(256) as i32),
                        act.arg(),
                        y.arg(),
                    ],
                )
                .unwrap();
            }
            let got: Vec<f32> = y.read().unwrap();
            let cpu = crate::model::tests::rmsnorm_cpu(&x, &w, eps);
            let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
            let inv = 1.0 / (ss / n as f64 + eps as f64).sqrt();
            let (mut dc, mut de) = (0f64, 0f64);
            for i in 0..n {
                let e = x[i] as f64 * inv * w[i] as f64;
                let tol = (w[i] as f64 * x[i] as f64 * inv).abs().max(1e-30);
                dc = dc.max((got[i] as f64 - cpu[i] as f64).abs() / tol);
                de = de.max((got[i] as f64 - e).abs() / tol);
            }
            // The sums of squares differ in order only: a few ulps.
            assert!(
                dc < 1e-5 && de < 1e-5,
                "n {n} x{scale}: vs cpu {dc:e}, vs f64 {de:e}"
            );
        }
    }
}

/// Decode attention (attn_split + attn_combine) on random q and f16 K/V vs
/// the CPU's `attend` and an f64 reference: every model's head layout,
/// short, odd and long contexts, and split counts from 1 to more than there
/// are tiles.
#[test]
fn gpu_attention_matches_cpu() {
    use crate::kv::{Half, KvElem};
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let fc = km.function("attn_combine").unwrap();
    let mut rng = Rng(4242);
    // (n_heads, n_kv, hd): 0.5B, 1.5B, 3B, 7B.
    for (nh, nkv, hd) in [(14, 2, 64), (12, 2, 128), (16, 2, 128), (28, 4, 128)] {
        let (dim, kvd) = (nh * hd, nkv * hd);
        let fs = km.function(&format!("attn_split_{hd}")).unwrap();
        for (np, max_split) in [
            (1, 23),
            (2, 1),
            (7, 46),
            (127, 4),
            (128, 23),
            (129, 23),
            (1000, 1),
            (1000, 23),
            (5857, 184),
            (2000, 184),
            (4099, 23),
            (9001, 46),
        ] {
            let q = rng.vec(dim, 1.0);
            // Keys scaled up so softmax is peaky (exercises the max shift).
            let k: Vec<Half> = rng
                .vec(np * kvd, 4.0)
                .into_iter()
                .map(Half::from_f32)
                .collect();
            let v: Vec<Half> = rng
                .vec(np * kvd, 1.0)
                .into_iter()
                .map(Half::from_f32)
                .collect();
            let kh: Vec<u16> = k.iter().map(|h| h.0).collect();
            let vh: Vec<u16> = v.iter().map(|h| h.0).collect();
            let (qb, kb, vb) = (
                g.upload(&q).unwrap(),
                g.upload(&kh).unwrap(),
                g.upload(&vh).unwrap(),
            );
            let params = g.upload(&[0u32, (np - 1) as u32, 0, 0]).unwrap();
            let po = g.alloc(nh * max_split * hd * 4).unwrap();
            let pml = g.alloc(nh * max_split * 2 * 4).unwrap();
            let act = g.alloc(act_bytes(dim)).unwrap();
            let out = g.alloc(dim * 4).unwrap();
            // SAFETY: decode.cu's signatures; buffers sized as `try_cache`
            // sizes them, K/V with np rows.
            unsafe {
                g.launch(
                    &fs,
                    (nkv as u32, max_split as u32, 1),
                    (ATT_TILE as u32, 1, 1),
                    0,
                    &[
                        qb.arg(),
                        kb.arg(),
                        vb.arg(),
                        params.arg(),
                        Arg::I32(nh as i32),
                        Arg::I32(nkv as i32),
                        Arg::F32(1.0 / (hd as f32).sqrt()),
                        Arg::I32(max_split as i32),
                        po.arg(),
                        pml.arg(),
                    ],
                )
                .unwrap();
                g.launch(
                    &fc,
                    (dim.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &[
                        po.arg(),
                        pml.arg(),
                        params.arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(max_split as i32),
                        Arg::I32(dim.is_multiple_of(256) as i32),
                        act.arg(),
                        out.arg(),
                    ],
                )
                .unwrap();
            }
            let got: Vec<f32> = out.read().unwrap();
            let tile = (np >= 512).then_some(64);
            let cpu = crate::model::tests::attend_cpu(&q, &k, &v, kvd, hd, np, tile);
            // f64 reference on the same f16 values.
            let group = nh / nkv;
            let (mut dc, mut de) = (0f64, 0f64);
            for h in 0..nh {
                let kvh = h / group;
                let s: Vec<f64> = (0..np)
                    .map(|t| {
                        (0..hd)
                            .map(|d| {
                                q[h * hd + d] as f64 * k[t * kvd + kvh * hd + d].to_f32() as f64
                            })
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let l: f64 = p.iter().sum();
                for d in 0..hd {
                    let e: f64 = (0..np)
                        .map(|t| p[t] * v[t * kvd + kvh * hd + d].to_f32() as f64)
                        .sum::<f64>()
                        / l;
                    let i = h * hd + d;
                    dc = dc.max((got[i] as f64 - cpu[i] as f64).abs());
                    de = de.max((got[i] as f64 - e).abs());
                }
            }
            // Outputs are O(1) (|v| < 1): absolute error of a few ulps of 1
            // per step of the score and value sums.
            assert!(
                dc < 2e-5 && de < 2e-5,
                "heads {nh}/{nkv} hd {hd} np {np} splits {max_split}: vs cpu {dc:e}, vs f64 {de:e}"
            );
        }
    }
}

/// argmax: the last of equal maxima, NaN never picked unless all are (then
/// the last index), -0.0 ties +0.0, -inf picked if everything else is NaN:
/// the CPU's `argmax` on every case.
#[test]
fn gpu_argmax_matches_cpu_rule() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let f = km.function("argmax").unwrap();
    let mut rng = Rng(9);
    let n = 151_936;
    let mut cases: Vec<Vec<f32>> = Vec::new();
    cases.push(rng.vec(n, 10.0));
    let mut ties = rng.vec(n, 1.0);
    for i in [5, 77_777, 151_000] {
        ties[i] = 3.0;
    }
    cases.push(ties);
    let mut nan = rng.vec(n, 1.0);
    nan[100] = f32::NAN;
    nan[n - 1] = f32::NAN;
    cases.push(nan);
    cases.push(vec![f32::NAN; n]);
    let mut neg = vec![f32::NAN; n];
    neg[1234] = f32::NEG_INFINITY;
    cases.push(neg);
    let mut zeros = vec![-1.0f32; n];
    zeros[10] = 0.0;
    zeros[20] = -0.0;
    cases.push(zeros);
    cases.push(vec![0.5; n]);
    cases.push(vec![1.0; 3]);
    for (c, x) in cases.iter().enumerate() {
        let xb = g.upload(x).unwrap();
        let out = g.alloc(4).unwrap();
        // SAFETY: argmax's signature; x.len() floats, one u32 out.
        unsafe {
            g.launch(
                &f,
                (1, 1, 1),
                (1024, 1, 1),
                0,
                &[xb.arg(), Arg::I32(x.len() as i32), out.arg()],
            )
            .unwrap();
        }
        let got: Vec<u32> = out.read().unwrap();
        assert_eq!(got[0], crate::argmax(x), "case {c}");
    }
}

/// qkv_rope on real layers: queries (rotated), keys (rotated, f16) and
/// values (f16) at several positions vs the CPU's row dots + bias + the
/// NEOX rotation with rope_table's f64 angles, f32 ops in rope_head's order.
#[test]
fn gpu_qkv_rope_matches_cpu() {
    use crate::kv::{Half, KvElem};
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let f = km.function("qkv_rope").unwrap();
    let fq = km.function("quantize").unwrap();
    let meta = |k: &str| meta(&m, k).unwrap();
    let (dim, nh, nkv) = (
        meta("embedding_length") as usize,
        meta("attention.head_count") as usize,
        meta("attention.head_count_kv") as usize,
    );
    let hd = dim / nh;
    let kvd = nkv * hd;
    let base = meta("rope.freq_base");
    let freq: Vec<f64> = (0..hd / 2)
        .map(|i| base.powf(-2.0 * i as f64 / hd as f64))
        .collect();
    let fb: Vec<u64> = freq.iter().map(|f| f.to_bits()).collect();
    let freq_b = g.upload(&fb).unwrap();
    let fr = km.function("rope_table").unwrap();
    let mut rng = Rng(31337);
    for layer in [0, 1] {
        let t = |s: &str| tensor(&m, &format!("blk.{layer}.{s}")).unwrap();
        let (tq, tk, tv) = (t("attn_q.weight"), t("attn_k.weight"), t("attn_v.weight"));
        let bias: Vec<Vec<f32>> = ["attn_q.bias", "attn_k.bias", "attn_v.bias"]
            .iter()
            .map(|s| vec_f32(&m, t(s)))
            .collect();
        let wb: Vec<DevBuf> = [tq, tk, tv]
            .iter()
            .map(|t| g.upload(m.tensor_data(t)).unwrap())
            .collect();
        let bb: Vec<DevBuf> = bias.iter().map(|b| g.upload(b).unwrap()).collect();
        for pos in [0usize, 3, 4097, 32767] {
            let x = rng.vec(dim, 1.0);
            let xb = g.upload(&x).unwrap();
            let act = g.alloc(act_bytes(dim)).unwrap();
            let params = g.upload(&[0u32, pos as u32, 0, 0]).unwrap();
            let q = g.alloc(dim * 4).unwrap();
            let rope = g.alloc(hd / 2 * 8).unwrap();
            let kc = g.alloc((pos + 1) * kvd * 2).unwrap();
            let vc = g.alloc((pos + 1) * kvd * 2).unwrap();
            let rb = |t: &Tensor| Arg::U64(quant::row_bytes(t.qtype.id, dim) as u64);
            let items = dim / 2 + kvd;
            // SAFETY: decode.cu's signatures; q holds dim floats, the
            // caches pos + 1 rows; matrices dim wide.
            unsafe {
                g.launch(
                    &fq,
                    (dim.div_ceil(256) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    &[
                        xb.arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(dim.is_multiple_of(256) as i32),
                        act.arg(),
                    ],
                )
                .unwrap();
                g.launch(
                    &fr,
                    (1, 1, 1),
                    ((hd / 2) as u32, 1, 1),
                    0,
                    &[
                        freq_b.arg(),
                        Arg::I32((hd / 2) as i32),
                        params.arg(),
                        rope.arg(),
                    ],
                )
                .unwrap();
                g.launch(
                    &f,
                    (items as u32, 1, 1),
                    (GEMV_THREADS, 1, 1),
                    0,
                    &[
                        Arg::I32(tq.qtype.id as i32),
                        wb[0].arg(),
                        rb(tq),
                        Arg::I32(tk.qtype.id as i32),
                        wb[1].arg(),
                        rb(tk),
                        Arg::I32(tv.qtype.id as i32),
                        wb[2].arg(),
                        rb(tv),
                        bb[0].arg(),
                        bb[1].arg(),
                        bb[2].arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(dim as i32),
                        act.arg(),
                        rope.arg(),
                        params.arg(),
                        q.arg(),
                        kc.arg(),
                        vc.arg(),
                    ],
                )
                .unwrap();
            }
            let gq: Vec<f32> = q.read().unwrap();
            let gk: Vec<u16> = kc.read().unwrap();
            let gv: Vec<u16> = vc.read().unwrap();
            // CPU reference.
            let proj = |t: &Tensor, b: &[f32]| -> Vec<f32> {
                let xq = quant::QuantizedAct::for_types(&x, &[t.qtype.id]);
                let rb = quant::row_bytes(t.qtype.id, dim);
                let d = m.tensor_data(t);
                (0..b.len())
                    .map(|r| quant::dot_row_q8(t.qtype.id, &d[r * rb..(r + 1) * rb], &xq) + b[r])
                    .collect()
            };
            let rope = |v: &mut [f32]| {
                for head in v.chunks_mut(hd) {
                    for i in 0..hd / 2 {
                        let a = pos as f64 * freq[i];
                        let (sn, cs) = (a.sin() as f32, a.cos() as f32);
                        let (x0, x1) = (head[i], head[i + hd / 2]);
                        head[i] = x0 * cs - x1 * sn;
                        head[i + hd / 2] = x0 * sn + x1 * cs;
                    }
                }
            };
            let (mut cq, mut ck, cv) = (proj(tq, &bias[0]), proj(tk, &bias[1]), proj(tv, &bias[2]));
            rope(&mut cq);
            rope(&mut ck);
            let scale = cq.iter().fold(0f32, |a, v| a.max(v.abs()));
            for i in 0..dim {
                assert!(
                    (gq[i] - cq[i]).abs() <= 1e-5 * scale,
                    "layer {layer} pos {pos} q[{i}]: {} vs {}",
                    gq[i],
                    cq[i]
                );
            }
            let row = pos * kvd;
            for (what, got, want) in [("k", &gk, &ck), ("v", &gv, &cv)] {
                for i in 0..kvd {
                    let (a, b) = (
                        Half(got[row + i]).to_f32(),
                        Half::from_f32(want[i]).to_f32(),
                    );
                    // One f16 rounding of values that differ in the last f32
                    // bits: equal, or adjacent halves.
                    let ulp = (b.abs() * 2f32.powi(-10)).max(2f32.powi(-24));
                    assert!(
                        (a - b).abs() <= ulp,
                        "layer {layer} pos {pos} {what}[{i}]: {a} vs {b}"
                    );
                }
            }
        }
    }
}

/// Greedy tokens after `ids`: `n` picks via forward_argmax, plus the logits
/// of every step if `keep`.
fn greedy<M: Lm>(net: &M, ids: &[u32], n: usize, keep: bool) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut c = net.new_cache(ids.len() + n + 1);
    let (&last, rest) = ids.split_last().unwrap();
    net.feed_batch(rest, 0, &mut c);
    let (mut tok, mut out, mut ls) = (last, Vec::new(), Vec::new());
    for i in 0..n {
        let pos = rest.len() + i;
        tok = if keep {
            let l = net.forward(tok, pos, &mut c);
            let t = crate::argmax(&l);
            ls.push(l);
            t
        } else {
            net.forward_argmax(tok, pos, &mut c)
        };
        out.push(tok);
    }
    (out, ls)
}

/// Same tokens and the same logit bits on every run, with and without
/// graphs, through `forward` and `forward_argmax` alike.
#[test]
fn gpu_deterministic_graph_or_not() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let mut net = CudaQwen2::new(&g, &km, &m, 512).unwrap();
    for p in [PROMPTS[0], PROMPTS[2]] {
        let ids = tok.encode(p);
        net.graphs = true;
        let (t1, l1) = greedy(&net, &ids, 40, true);
        let (t2, l2) = greedy(&net, &ids, 40, true);
        let (t3, _) = greedy(&net, &ids, 40, false);
        net.graphs = false;
        let (t4, l4) = greedy(&net, &ids, 40, true);
        let (t5, _) = greedy(&net, &ids, 40, false);
        assert_eq!(t1, t2);
        assert_eq!(t1, t3);
        assert_eq!(t1, t4);
        assert_eq!(t1, t5);
        for i in 0..l1.len() {
            assert_eq!(bits(&l1[i]), bits(&l2[i]), "run to run, step {i}");
            assert_eq!(bits(&l1[i]), bits(&l4[i]), "graph vs not, step {i}");
        }
    }
}

/// How `o`'s 32-token greedy continuations of PROMPTS compare with the
/// reference `r`'s: (prompts identical, tokens matched before the first
/// divergence, summed), asserting that every divergence is a near-tie in
/// r's own logits.
fn continuations<R: Lm, O: Lm>(
    r: &R,
    o: &O,
    tok: &crate::tokenizer::Tokenizer,
    name: &str,
) -> (usize, usize) {
    let (mut same, mut matched) = (0, 0);
    for p in PROMPTS {
        let ids = tok.encode(p);
        let (to, _) = greedy(o, &ids, 32, false);
        let (tr, lr) = greedy(r, &ids, 32, true);
        match (0..32).find(|&i| to[i] != tr[i]) {
            None => {
                same += 1;
                matched += 32;
            }
            Some(i) => {
                matched += i;
                let l = &lr[i];
                let margin = l[tr[i] as usize] - l[to[i] as usize];
                eprintln!(
                    "{name}: {p:?} diverges at token {i}: {:?} vs {:?}, reference margin {margin:.3}",
                    tok.decode(&[tr[i]]),
                    tok.decode(&[to[i]])
                );
                assert!(margin < 0.5, "not a near-tie: margin {margin}");
            }
        }
    }
    eprintln!(
        "{name}: {same}/{} continuations identical, {matched}/{} tokens before divergence",
        PROMPTS.len(),
        32 * PROMPTS.len()
    );
    (same, matched)
}

/// Greedy continuations on the GPU vs the CPU, next to the CPU's per-row
/// kernels vs its default ones. At the ~3-4% per-position near-tie flip
/// rate both comparisons show (`logits_match`), a 32-token continuation
/// survives with probability ~0.97^32 ~ 1/3, so "most match" can't hold even
/// CPU vs CPU; what's asserted is that every divergence is a near-tie and
/// the GPU stays about as close to the CPU as the CPU is to itself.
fn continuations_match(m: &Model, name: &str) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let gpu = CudaQwen2::new(&g, &km, m, 256).unwrap();
    let rows = Qwen2::new_with(m, false);
    let cpu = Qwen2::new(m);
    let (gs, gm) = continuations(&cpu, &gpu, &tok, &format!("{name} gpu"));
    let (cs, cm) = continuations(&cpu, &rows, &tok, &format!("{name} cpu rows"));
    assert!(
        gs + 2 >= cs,
        "identical continuations: gpu {gs}, cpu rows {cs}"
    );
    assert!(2 * gm + 32 >= cm, "matched tokens: gpu {gm}, cpu rows {cm}");
}

#[test]
fn gpu_continuations_match_cpu_0_5b() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(m) = crate::testutil::model_with(Some(false)) else {
        return;
    };
    continuations_match(&m, "0.5B");
}

#[test]
fn gpu_continuations_match_cpu_1_5b() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    continuations_match(&m, "1.5B");
}

/// Per-kernel GPU time of one decode step at a few positions, and the
/// matrices' effective bandwidth. A benchmark, not a check: run with
/// `--ignored`, the model from AXON_TEST_MODEL.
#[test]
#[ignore]
fn cuda_profile() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let path = crate::testutil::model_path().unwrap();
    let m = crate::gguf::load_with(path.to_str().unwrap(), Some(false)).unwrap();
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::new(&g, &km, &m, 4200).unwrap();
    let mut c = net.new_cache(4200);
    for pos in [8usize, 1000, 4100] {
        // Warm up (and fill the cache up to pos, cheaply: positions past
        // what was written read zeros, which is fine for timing).
        let _ = net.profile(&mut c, 100, pos).unwrap();
        // Per kernel, the best of 10 steps: the GPU is shared with a
        // desktop, whose work lands in random kernels' windows.
        let runs: Vec<_> = (0..10)
            .map(|_| net.profile(&mut c, 100, pos).unwrap())
            .collect();
        let mut k = runs[0].0.clone();
        for (i, row) in k.iter_mut().enumerate() {
            row.ms = runs.iter().map(|r| r.0[i].ms).fold(f64::INFINITY, f64::min);
        }
        let total: f64 = k.iter().map(|r| r.ms).sum();
        // The same step replayed as a graph, GPU time only (best, median of 20).
        let mut gt: Vec<f64> = (0..20)
            .map(|_| {
                let e0 = g.record().unwrap();
                net.step(&mut c, 100, pos, true).unwrap();
                let e1 = g.record().unwrap();
                e0.ms_to(&e1).unwrap() as f64
            })
            .collect();
        gt.sort_by(f64::total_cmp);
        // And through the Lm entry point, wall clock: what generation sees.
        let t = std::time::Instant::now();
        for i in 0..20 {
            net.forward_argmax(100, pos + i, &mut c);
        }
        let wall = t.elapsed().as_secs_f64() * 1e3 / 20.0;
        let (mut tl, mut ts) = (0f64, 0f64);
        for i in 0..20 {
            let t0 = std::time::Instant::now();
            net.step(&mut c, 100, pos + i, true).unwrap();
            let t1 = std::time::Instant::now();
            g.sync().unwrap();
            tl += (t1 - t0).as_secs_f64();
            ts += t1.elapsed().as_secs_f64();
        }
        eprintln!(
            "pos {pos}: step() call {:.3} ms, sync {:.3} ms",
            tl * 50.0,
            ts * 50.0
        );
        let bpt = net.bytes_per_token(true) as f64;
        eprintln!(
            "pos {pos}: graph step {:.3} ms best ({:.0} GB/s of weights), {:.3} median; \
             forward_argmax wall {wall:.3} ms",
            gt[0],
            bpt / (gt[0] * 1e-3) / 1e9,
            gt[10]
        );
        let bytes: usize = k.iter().map(|r| r.bytes).sum();
        eprintln!(
            "pos {pos}: kernels {total:.3} ms (best of 10 per kernel), {:.2} GB of matrices, \
             {:.0} GB/s overall",
            bytes as f64 / 1e9,
            bytes as f64 / (total * 1e-3) / 1e9
        );
        for KernelTime {
            name,
            ms,
            launches: n,
            bytes,
        } in &k
        {
            let gbs = if *bytes > 0 {
                *bytes as f64 / (ms * 1e-3) / 1e9
            } else {
                0.0
            };
            eprintln!(
                "  {name:<10} {n:>4} x {:>8.1} us = {ms:>7.3} ms  {:>6.1}%  {}",
                ms * 1e3 / *n as f64,
                100.0 * ms / total,
                if *bytes > 0 {
                    format!("{gbs:.0} GB/s")
                } else {
                    String::new()
                }
            );
        }
    }
}

#[test]
#[ignore]
fn cuda_gemv_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let fq = km.function("quantize").unwrap();
    // The ceiling: a plain streaming read of 512 MB.
    {
        let f = km.function("stream_read").unwrap();
        let n = 512usize << 20;
        let buf = g.alloc(n).unwrap();
        buf.zero().unwrap();
        let out = g.alloc(4).unwrap();
        let sms = g.info().unwrap().sms;
        for blocks in [sms * 4, sms * 8, sms * 16] {
            let mut best = f64::INFINITY;
            for _ in 0..10 {
                let e0 = g.record().unwrap();
                // SAFETY: stream_read reads n / 16 uint4s of buf, writes out.
                unsafe {
                    g.launch(
                        &f,
                        (blocks, 1, 1),
                        (256, 1, 1),
                        0,
                        &[buf.arg(), Arg::U64((n / 16) as u64), out.arg()],
                    )
                    .unwrap();
                }
                let e1 = g.record().unwrap();
                best = best.min(e0.ms_to(&e1).unwrap() as f64);
            }
            eprintln!(
                "stream read {blocks} blocks: {:.0} GB/s",
                n as f64 / (best * 1e-3) / 1e9
            );
        }
    }
    let mut rng = Rng(5);
    for (what, qt, cols, rows) in [
        ("q/wo  Q4_K", Q4_K, 3584, 3584),
        ("gate  Q4_K", Q4_K, 3584, 18944),
        ("down  Q4_K", Q4_K, 18944, 3584),
        ("down  Q6_K", Q6_K, 18944, 3584),
        ("head  Q6_K", Q6_K, 3584, 152064),
        ("0.5B head Q8_0", Q8_0, 896, 151936),
        ("0.5B ffn Q5_0", Q5_0, 896, 4864),
    ] {
        let rb = quant::row_bytes(qt, cols);
        let mut w = vec![0u8; rb * rows];
        for x in w.iter_mut() {
            *x = rng.next() as u8;
        }
        // Enough copies to cycle through more than the L2 cache (tens of
        // MB) holds, as a token's weights do.
        let copies = (300 << 20) / w.len() + 1;
        let wbs: Vec<DevBuf> = (0..copies).map(|_| g.upload(&w).unwrap()).collect();
        drop(w);
        let xb = g.upload(&rng.vec(cols, 1.0)).unwrap();
        let act = g.alloc(act_bytes(cols)).unwrap();
        let out = g.alloc(rows * 4).unwrap();
        // SAFETY: decode.cu's signatures; sizes as in gpu_gemv.
        unsafe {
            g.launch(
                &fq,
                (cols.div_ceil(256) as u32, 1, 1),
                (32, 1, 1),
                0,
                &[
                    xb.arg(),
                    Arg::I32(cols as i32),
                    Arg::I32(cols.is_multiple_of(256) as i32),
                    act.arg(),
                ],
            )
            .unwrap();
        }
        for (kname, per) in [("gemv", 2), ("gemv1", 1)] {
            let fg = km.function(kname).unwrap();
            let mut best = f64::INFINITY;
            for i in 0..20.max(2 * copies) {
                let wb = &wbs[i % copies];
                let e0 = g.record().unwrap();
                // SAFETY: as above.
                unsafe {
                    g.launch(
                        &fg,
                        (rows.div_ceil(per) as u32, 1, 1),
                        (GEMV_THREADS, 1, 1),
                        0,
                        &[
                            Arg::I32(qt as i32),
                            wb.arg(),
                            Arg::U64(rb as u64),
                            Arg::I32(rows as i32),
                            Arg::I32(cols as i32),
                            act.arg(),
                            Arg::Ptr(0),
                            Arg::I32(0),
                            out.arg(),
                        ],
                    )
                    .unwrap();
                }
                let e1 = g.record().unwrap();
                best = best.min(e0.ms_to(&e1).unwrap() as f64);
            }
            eprintln!(
                "{what:<16} {kname:<6} {:>7.1} MB {:>8.1} us {:>6.0} GB/s",
                (rb * rows) as f64 / 1e6,
                best * 1e3,
                (rb * rows) as f64 / (best * 1e-3) / 1e9
            );
        }
    }
}

/// A model and cache that won't fit is an Err saying so, before anything
/// is allocated; one that fits leaves the memory as it found it on drop.
#[test]
fn gpu_too_big_is_an_error() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let free0 = g.info().unwrap().free_bytes;
    let e = CudaQwen2::new(&g, &km, &m, 50_000_000).err().unwrap();
    assert!(e.contains("GiB is free") && e.contains("--ctx"), "{e}");
    {
        let net = CudaQwen2::new(&g, &km, &m, 1024).unwrap();
        let _c = net.new_cache(1024);
        assert!(g.info().unwrap().free_bytes < free0);
        // Every cache is checked against what is free when it is made (new
        // budgets one): one too big is an Err before allocating, and a
        // second normal one still fits.
        let e = net.try_cache(50_000_000).err().unwrap();
        assert!(e.contains("GiB is free") && e.contains("--ctx"), "{e}");
        let _c2 = net.try_cache(1024).unwrap();
    }
    // Others (the desktop) allocate too: allow some slack.
    let free1 = g.info().unwrap().free_bytes;
    assert!(
        free1 + (64 << 20) >= free0,
        "{free0} free before, {free1} after"
    );
}

/// Logits deep into a long context, through the replayed graph: ~1100
/// prompt tokens (a prompt repeated) put attention over 9 tiles and many
/// splits. GPU vs CPU graded as in `logits_match`, against the CPU's own
/// two kernel sets.
#[test]
fn gpu_logits_match_cpu_long_context() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model_with(Some(false)) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let gpu = CudaQwen2::new(&g, &km, &m, 1300).unwrap();
    assert!(gpu.graphs || std::env::var_os("AXON_CUDA_GRAPHS").is_some());
    let rows = Qwen2::new_with(&m, false);
    let cpu = Qwen2::new(&m);
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    for p in [PROMPTS[2], PROMPTS[1]] {
        let one = tok.encode(&format!("{p} "));
        let ids: Vec<u32> = one
            .iter()
            .copied()
            .cycle()
            .take(1100 + one.len() / 2)
            .collect();
        compare(&cpu, &gpu, &ids, 8, &mut ag);
        compare(&cpu, &rows, &ids, 8, &mut ac);
    }
    report("0.5B long gpu vs cpu", &ag);
    report("0.5B long cpu rows vs cpu", &ac);
    let n = ag.positions;
    assert!(
        ag.top1 * 100 + 2 * n >= ac.top1 * 100,
        "top-1 {}/{n}",
        ag.top1
    );
    assert!(ag.flips.iter().all(|&d| d < 0.5), "{:?}", ag.flips);
    assert!(ag.sum_kl <= 2.0 * ac.sum_kl + 1e-4, "mean KL");
    assert!(ag.max_kl < 0.5 && ag.max_abs < 6.0, "extremes");
}

/// `gpu_logits_match_cpu_long_context` at ~2000 tokens, the prompt fed by
/// the batched prefill in the GPU's own batch size (four passes, the last
/// ragged), as `axon gen` does it: graded the same way.
#[test]
fn gpu_logits_match_cpu_2000_tokens() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model_with(Some(false)) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let gpu = CudaQwen2::new(&g, &km, &m, 2100).unwrap();
    let rows = Qwen2::new_with(&m, false);
    let cpu = Qwen2::new(&m);
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    let one = tok.encode(&format!("{} ", PROMPTS[4]));
    let ids: Vec<u32> = one.iter().copied().cycle().take(2000).collect();
    compare(&cpu, &gpu, &ids, 6, &mut ag);
    compare(&cpu, &rows, &ids, 6, &mut ac);
    report("0.5B 2000 gpu vs cpu", &ag);
    report("0.5B 2000 cpu rows vs cpu", &ac);
    let n = ag.positions;
    assert!(
        ag.top1 * 100 + 2 * n >= ac.top1 * 100,
        "top-1 {}/{n}",
        ag.top1
    );
    assert!(ag.flips.iter().all(|&d| d < 0.5), "{:?}", ag.flips);
    assert!(ag.sum_kl <= 2.0 * ac.sum_kl + 1e-4, "mean KL");
    assert!(ag.max_kl < 0.5 && ag.max_abs < 6.0, "extremes");
}

/// Programmatic dependent launch changes when kernels start, never what
/// they compute: tokens and logit bits equal with it on and off.
#[test]
fn gpu_pdl_on_off_identical() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km_on = Kernels::module(&g).unwrap();
    let mut km_off = Kernels::module(&g).unwrap();
    km_off.disable_pdl();
    if !km_on.pdl() {
        eprintln!("PDL unavailable here: comparing off with off");
    }
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let on = CudaQwen2::new(&g, &km_on, &m, 256).unwrap();
    let off = CudaQwen2::new(&g, &km_off, &m, 256).unwrap();
    for p in [PROMPTS[0], PROMPTS[3]] {
        let ids = tok.encode(p);
        let (ta, la) = greedy(&on, &ids, 24, true);
        let (tb, lb) = greedy(&off, &ids, 24, true);
        assert_eq!(ta, tb);
        for i in 0..la.len() {
            assert_eq!(bits(&la[i]), bits(&lb[i]), "step {i}");
        }
    }
}

/// gate_up with gate and up of different types (its two-pass branch) and
/// of the same type (one pass), vs the CPU's row dots and SwiGLU in f64.
#[test]
fn gpu_gate_up_mixed_types() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let (fq, fgu) = (
        km.function("quantize").unwrap(),
        km.function("gate_up").unwrap(),
    );
    // The 0.5B's ffn_down matrices are Q6_K or Q4_K, 896 rows of 4864.
    let mats: Vec<&Tensor> = (0..24)
        .map(|i| tensor(&m, &format!("blk.{i}.ffn_down.weight")).unwrap())
        .collect();
    let q4 = *mats.iter().find(|t| t.qtype.id == Q4_K).unwrap();
    let q6 = *mats.iter().find(|t| t.qtype.id == Q6_K).unwrap();
    let n = q4.dims[0] as usize;
    let rows = q4.dims[1] as usize;
    let mut rng = Rng(808);
    let x = rng.vec(n, 1.0);
    let xb = g.upload(&x).unwrap();
    let act = g.alloc(act_bytes(n)).unwrap();
    let xq = quant::QuantizedAct::for_types(&x, &[Q4_K]);
    for (gt, ut) in [(q4, q6), (q6, q4), (q4, q4)] {
        let (wg, wu) = (
            g.upload(m.tensor_data(gt)).unwrap(),
            g.upload(m.tensor_data(ut)).unwrap(),
        );
        let out = g.alloc(rows * 4).unwrap();
        let rb = |t: &Tensor| quant::row_bytes(t.qtype.id, n);
        // SAFETY: decode.cu's signatures; act for n inputs, rows outputs.
        unsafe {
            g.launch(
                &fq,
                (n.div_ceil(256) as u32, 1, 1),
                (32, 1, 1),
                0,
                &[xb.arg(), Arg::I32(n as i32), Arg::I32(1), act.arg()],
            )
            .unwrap();
            g.launch(
                &fgu,
                (rows as u32, 1, 1),
                (GEMV_THREADS, 1, 1),
                0,
                &[
                    Arg::I32(gt.qtype.id as i32),
                    wg.arg(),
                    Arg::U64(rb(gt) as u64),
                    Arg::I32(ut.qtype.id as i32),
                    wu.arg(),
                    Arg::U64(rb(ut) as u64),
                    Arg::I32(rows as i32),
                    Arg::I32(n as i32),
                    act.arg(),
                    out.arg(),
                ],
            )
            .unwrap();
        }
        let got: Vec<f32> = out.read().unwrap();
        let dot = |t: &Tensor, r: usize| {
            let d = m.tensor_data(t);
            quant::dot_row_q8(t.qtype.id, &d[r * rb(t)..(r + 1) * rb(t)], &xq) as f64
        };
        for (r, &gr) in got.iter().enumerate() {
            let (gv, uv) = (dot(gt, r), dot(ut, r));
            let want = gv / (1.0 + (-gv).exp()) * uv;
            let tol = 1e-5 * (gv.abs().max(1.0) * uv.abs().max(1.0));
            assert!(
                (gr as f64 - want).abs() <= tol,
                "{} x {} row {r}: {gr} vs {want}",
                gt.qtype.name,
                ut.qtype.name
            );
        }
    }
}

/// embed: the token's row, bit for bit `quant::dequant` of it, for every
/// type (real matrices of each standing in for the embedding).
#[test]
fn gpu_embed_matches_dequant() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let f = km.function("embed").unwrap();
    let mut types = Vec::new();
    for name in [
        "token_embd.weight",
        "output.weight",
        "blk.0.ffn_down.weight",
        "blk.2.ffn_down.weight",
        "blk.0.attn_norm.weight",
    ] {
        let t = tensor(&m, name).unwrap();
        let (qt, n) = (t.qtype.id, t.dims[0] as usize);
        let nrows = t.n_elements as usize / n;
        let rb = quant::row_bytes(qt, n);
        let w = g.upload(m.tensor_data(t)).unwrap();
        let x = g.alloc(n * 4).unwrap();
        for tok in [0, 1, nrows / 2 + 7, nrows - 1]
            .into_iter()
            .filter(|&r| r < nrows)
        {
            let params = g.upload(&[tok as u32, 0, 0, 0]).unwrap();
            // SAFETY: embed's signature; row tok < nrows of w, x has n floats.
            unsafe {
                g.launch(
                    &f,
                    (n.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &[
                        Arg::I32(qt as i32),
                        w.arg(),
                        Arg::U64(rb as u64),
                        Arg::I32(n as i32),
                        params.arg(),
                        x.arg(),
                    ],
                )
                .unwrap();
            }
            let got: Vec<f32> = x.read().unwrap();
            let mut want = vec![0f32; n];
            quant::dequant(qt, &m.tensor_data(t)[tok * rb..(tok + 1) * rb], &mut want);
            assert_eq!(bits(&got), bits(&want), "{name} row {tok}");
        }
        types.push(t.qtype.name);
    }
    eprintln!("embed types: {types:?}");
}

/// The batch contract, as the CPU's: an empty batch and one past the cache
/// panic before any work; a cache only runs with the model that made it.
#[test]
fn gpu_batch_and_cache_contract() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let a = CudaQwen2::new(&g, &km, &m, 64).unwrap();
    let b = CudaQwen2::new(&g, &km, &m, 64).unwrap();
    let panics = |f: &dyn Fn()| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
            .err()
            .map(|e| match e.downcast_ref::<String>() {
                Some(s) => s.clone(),
                None => e.downcast_ref::<&str>().map_or("", |s| s).to_string(),
            })
            .unwrap_or_default()
    };
    let mut c = a.new_cache(8);
    let cell = std::cell::RefCell::new(&mut c);
    let e = panics(&|| a.feed_batch(&[], 0, &mut cell.borrow_mut()));
    assert!(e.contains("empty batch"), "{e}");
    let e = panics(&|| {
        a.verify_argmax(&[], 0, &mut cell.borrow_mut());
    });
    assert!(e.contains("empty batch"), "{e}");
    let e = panics(&|| a.feed_batch(&[1; 5], 4, &mut cell.borrow_mut()));
    assert!(e.contains("exceed cache size 8"), "{e}");
    // A token past the vocabulary, anywhere in a batch, before any work
    // (embed_b would read past the embedding).
    let v = a.vocab as u32;
    let e = panics(&|| a.feed_batch(&[1, v, 2], 0, &mut cell.borrow_mut()));
    assert!(e.contains(&format!("token {v} out of range")), "{e}");
    let e = panics(&|| {
        a.verify_argmax(&[3, 4, u32::MAX], 0, &mut cell.borrow_mut());
    });
    assert!(e.contains("out of range"), "{e}");
    let e = panics(&|| {
        b.forward_argmax(1, 0, &mut cell.borrow_mut());
    });
    assert!(e.contains("another CudaQwen2"), "{e}");
    // Still usable after all that (nothing ran).
    a.feed_batch(&[1, 2], 0, &mut cell.borrow_mut());
    let _ = a.forward_argmax(3, 2, &mut cell.borrow_mut());
}

/// Bytes per token of a batched activation buffer (qwen2.rs act_stride).
fn act_stride_t(n: usize) -> usize {
    act_bytes(n).next_multiple_of(256)
}

/// nb random vectors quantized (decode `quantize`, one launch each) into a
/// batched activation buffer.
fn quantize_batch<'g>(g: &'g Gpu, km: &Module, xs: &[Vec<f32>], cols: usize) -> DevBuf<'g> {
    let fq = km.function("quantize").unwrap();
    let st = act_stride_t(cols);
    let act = g.alloc(st * xs.len()).unwrap();
    for (z, x) in xs.iter().enumerate() {
        let xb = g.upload(x).unwrap();
        // SAFETY: quantize's signature; act + z * st holds act_bytes(cols).
        unsafe {
            g.launch(
                &fq,
                (cols.div_ceil(256) as u32, 1, 1),
                (32, 1, 1),
                0,
                &[
                    xb.arg(),
                    Arg::I32(cols as i32),
                    Arg::I32(cols.is_multiple_of(256) as i32),
                    act.arg_at(z * st),
                ],
            )
            .unwrap();
        }
        g.sync().unwrap();
    }
    act
}

/// A batched product on the GPU: `kernel` (a gemm_* or gemvn_*) over the
/// batch; rows x nb outputs, row-major per token.
#[allow(clippy::too_many_arguments)]
fn gpu_batched(
    g: &Gpu,
    km: &Module,
    kernel: &str,
    qt: u32,
    w: &DevBuf,
    rows: usize,
    cols: usize,
    act: &DevBuf,
    nb: usize,
    epi: &Epi,
) -> Vec<f32> {
    let out = match &epi.init {
        // The same residual rows for every token.
        Some(v) => g.upload(&v.repeat(nb)).unwrap(),
        None => {
            let o = g.alloc(rows * nb * 4).unwrap();
            o.zero().unwrap();
            o
        }
    };
    let bias = epi.bias.as_ref().map(|b| g.upload(b).unwrap());
    let f = km.function(kernel).unwrap();
    let st = act_stride_t(cols) as u64;
    let rb = quant::row_bytes(qt, cols) as u64;
    let common = [
        w.arg(),
        Arg::U64(rb),
        Arg::I32(rows as i32),
        Arg::I32(cols as i32),
        act.arg(),
        Arg::U64(st),
        Arg::I32(nb as i32),
        bias.as_ref().map_or(Arg::Ptr(0), |b| b.arg()),
        Arg::I32(epi.init.is_some() as i32),
        out.arg(),
        Arg::U64(rows as u64),
    ];
    // SAFETY: prefill.cu's signatures; act holds nb tokens of cols inputs,
    // out rows * nb floats.
    unsafe {
        if kernel.starts_with("gemvm") {
            g.launch(
                &f,
                (rows.div_ceil(16) as u32, 1, 1),
                (128, 1, 1),
                0,
                &common,
            )
            .unwrap();
        } else if kernel.starts_with("gemm") {
            let (bn, sm) = if kernel.starts_with("gemm1") || kernel.starts_with("gemms1") {
                (GM1_BN, GEMM1_SMEM)
            } else {
                (GM_BN, GEMM_SMEM)
            };
            f.set_max_shared(sm).unwrap();
            g.launch(
                &f,
                (nb.div_ceil(bn) as u32, rows.div_ceil(GM_BM) as u32, 1),
                (256, 1, 1),
                sm as u32,
                &common,
            )
            .unwrap();
        } else {
            let (r, t): (usize, usize) = if kernel == "gemvn_2x8" {
                (2, 8)
            } else {
                (2, 16)
            };
            let mut args = vec![Arg::I32(qt as i32)];
            args.extend_from_slice(&common);
            g.launch(
                &f,
                (rows.div_ceil(r) as u32, nb.div_ceil(t) as u32, 1),
                (GEMV_THREADS, 1, 1),
                0,
                &args,
            )
            .unwrap();
        }
    }
    g.sync().unwrap();
    out.read().unwrap()
}

/// prefill.cu's name for a type.
fn type_name(qt: u32) -> &'static str {
    match qt {
        Q4_K => "q4_k",
        Q6_K => "q6_k",
        Q8_0 => "q8_0",
        _ => "q5_0",
    }
}

/// The GEMMs batch.rs uses for a type: (wide rows, K-quant rows of <= 16 blocks).
fn gemm_names(qt: u32) -> (String, Option<String>) {
    let tn = type_name(qt);
    match qt {
        Q4_K => ("gemm_q4_k".into(), Some("gemms1_q4_k".into())),
        Q6_K => ("gemm1_q6_k".into(), Some("gemms1_q6_k".into())),
        _ => (format!("gemm1_{tn}"), None),
    }
}

/// A random matrix of type qt (rows x cols) with sane f16 scales (1/1024)
/// in every block, so products stay finite.
fn synth_matrix(rng: &mut Rng, qt: u32, rows: usize, cols: usize) -> Vec<u8> {
    let rb = quant::row_bytes(qt, cols);
    let mut w: Vec<u8> = (0..rb * rows).map(|_| rng.next() as u8).collect();
    let (bb, vb) = match qt {
        Q4_K => (144, 256),
        Q6_K => (210, 256),
        Q8_0 => (34, 32),
        _ => (22, 32),
    };
    for b in 0..rows * cols / vb {
        let o = b * bb + if qt == Q6_K { 208 } else { 0 };
        w[o..o + 2].copy_from_slice(&0x1400u16.to_le_bytes());
        if qt == Q4_K {
            w[o + 2..o + 4].copy_from_slice(&0x1400u16.to_le_bytes());
        }
    }
    w
}

/// The batched products (tensor-core GEMMs and the dp4a gemvn) give, bit for
/// bit, the decode GEMV's row sums for every token: every weight type, odd
/// row counts, batches that aren't multiples of the tiles.
#[test]
fn gpu_batched_products_match_gemv_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let mut seen = Vec::new();
    for name in [
        "blk.0.attn_q.weight", // Q5_0, 896 wide
        "blk.0.attn_v.weight",
        "blk.1.attn_v.weight",
        "blk.0.ffn_down.weight", // Q6_K or Q4_K, 4864 wide
        "blk.1.ffn_down.weight",
        "blk.2.ffn_down.weight",
        "blk.0.ffn_gate.weight",
        "output.weight", // Q8_0
        // Synthetic: K-quant rows of fewer blocks than the GEMV's 16 lane
        // groups (empty groups), and other widths.
        "synth Q4_K 768",
        "synth Q4_K 1536",
        "synth Q6_K 1536",
        "synth Q6_K 2304",
        "synth Q8_0 3584",
        "synth Q5_0 1536",
        // The 7B's shapes: 14 blocks (no running sums), 74 (groups of 4-5
        // blocks: running sums carried over several stages).
        "synth Q4_K 3584",
        "synth Q4_K 18944",
        "synth Q6_K 3584",
        "synth Q6_K 18944",
    ] {
        let synth: Vec<u8>;
        let (qt, cols, rows, wd): (u32, usize, usize, &[u8]) = if name.starts_with("synth") {
            let f: Vec<&str> = name.split(' ').collect();
            let qt = match f[1] {
                "Q4_K" => Q4_K,
                "Q6_K" => Q6_K,
                "Q8_0" => Q8_0,
                _ => Q5_0,
            };
            let cols: usize = f[2].parse().unwrap();
            synth = synth_matrix(&mut rng, qt, 333, cols);
            (qt, cols, 333, &synth[..])
        } else {
            let t = tensor(&m, name).unwrap();
            let (qt, cols) = (t.qtype.id, t.dims[0] as usize);
            let rb = quant::row_bytes(qt, cols);
            let rows = (t.n_elements as usize / cols).min(1000) - 1; // odd, not a tile multiple
            (qt, cols, rows, &m.tensor_data(t)[..rows * rb])
        };
        let w = g.upload(wd).unwrap();
        for nb in [1usize, 6, 7, 8, 13, 64, 66, 77] {
            let xs: Vec<Vec<f32>> = (0..nb)
                .map(|i| rng.vec(cols, [1.0f32, 1e-3, 40.0][i % 3]))
                .collect();
            // Epilogues by batch: plain, bias, residual add, both (the
            // qkv biases and the wo / down residual of a pass).
            let epi = match nb % 4 {
                0 => Epi::default(),
                1 => Epi {
                    bias: Some(rng.vec(rows, 3.0)),
                    init: None,
                },
                2 => Epi {
                    bias: None,
                    init: Some(rng.vec(rows, 5.0)),
                },
                _ => Epi {
                    bias: Some(rng.vec(rows, 3.0)),
                    init: Some(rng.vec(rows, 5.0)),
                },
            };
            let want: Vec<Vec<f32>> = xs
                .iter()
                .map(|x| gpu_gemv_epi(&g, &km, "gemv", qt, wd, cols, x, &epi))
                .collect();
            let act = quantize_batch(&g, &km, &xs, cols);
            let mut kernels: Vec<String> = vec!["gemvn_2x8".into(), "gemvn_2x16".into()];
            if km.mma() {
                let (wide, short) = gemm_names(qt);
                kernels.push(wide);
                if let Some(sh) = short.filter(|_| cols <= 16 * 256) {
                    kernels.push(sh);
                }
                let tn = type_name(qt);
                if nb <= 8 {
                    kernels.push(format!("gemvm8_{tn}"));
                }
                if nb <= 16 {
                    kernels.push(format!("gemvm16_{tn}"));
                }
            }
            for k in &kernels {
                let got = gpu_batched(&g, &km, k, qt, &w, rows, cols, &act, nb, &epi);
                for z in 0..nb {
                    let gz = &got[z * rows..(z + 1) * rows];
                    if bits(gz) != bits(&want[z]) {
                        let r = (0..rows).find(|&r| gz[r].to_bits() != want[z][r].to_bits());
                        let r = r.unwrap();
                        panic!(
                            "{name} {} {k} nb {nb} token {z} row {r}: {} vs gemv {}",
                            crate::gguf::qtype(qt).name,
                            gz[r],
                            want[z][r]
                        );
                    }
                }
            }
        }
        eprintln!("{name} {}: bit-identical", crate::gguf::qtype(qt).name);
        seen.push(qt);
    }
    for qt in [Q4_K, Q6_K, Q5_0, Q8_0] {
        assert!(seen.contains(&qt), "type {qt} not covered");
    }
}

/// Batched products' speed (best of N, events): tensor-core GEMM for
/// prefill batches, gemvn for verify-sized ones, vs the GEMV's per-token
/// time. `AXON_SKIP_GPU_TESTS` aside, run with --ignored.
#[test]
#[ignore]
fn cuda_gemm_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    for k in [
        "gemm_q4_k",
        "gemms1_q4_k",
        "gemm1_q6_k",
        "gemms1_q6_k",
        "gemm1_q8_0",
        "gemm1_q5_0",
        "gemvn_2x8",
        "gemvm8_q4_k",
        "gemvm8_q6_k",
        "gemvm16_q4_k",
        "gemvm16_q6_k",
    ] {
        eprintln!(
            "{k}: regs, local bytes {:?}",
            km.function(k).unwrap().regs_local().unwrap()
        );
    }
    let mut rng = Rng(7);
    let only = std::env::var("AXON_BENCH_ONLY").ok();
    for (what, qt, cols, rows) in [
        ("q/wo  Q4_K", Q4_K, 3584, 3584),
        ("gate  Q4_K", Q4_K, 3584, 18944),
        ("down  Q4_K", Q4_K, 18944, 3584),
        ("down  Q6_K", Q6_K, 18944, 3584),
        ("0.5B ffn Q5_0", Q5_0, 896, 4864),
        ("0.5B q Q8_0", Q8_0, 896, 4864),
    ] {
        if only.as_deref().is_some_and(|o| !what.contains(o)) {
            continue;
        }
        let rb = quant::row_bytes(qt, cols);
        let mut w = vec![0u8; rb * rows];
        for x in w.iter_mut() {
            *x = rng.next() as u8;
        }
        // Sane scales: f16 1/1024 in every block's d (and dmin).
        let (bb, vb) = match qt {
            Q4_K => (144, 256),
            Q6_K => (210, 256),
            Q8_0 => (34, 32),
            _ => (22, 32),
        };
        for b in 0..rows * cols / vb {
            let o = b * bb + if qt == Q6_K { 208 } else { 0 };
            w[o..o + 2].copy_from_slice(&0x1400u16.to_le_bytes());
            if qt == Q4_K {
                w[o + 2..o + 4].copy_from_slice(&0x1400u16.to_le_bytes());
            }
        }
        let wb = g.upload(&w).unwrap();
        for nb in [8usize, 16, 128, 512] {
            let xs: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 1.0)).collect();
            let act = quantize_batch(&g, &km, &xs, cols);
            let out = g.alloc(rows * nb * 4).unwrap();
            let tn = type_name(qt);
            let mut kernels: Vec<String> = Vec::new();
            if nb <= 16 {
                kernels.push(if nb <= 8 { "gemvn_2x8" } else { "gemvn_2x16" }.into());
                kernels.push(format!("gemvm{}_{tn}", if nb <= 8 { 8 } else { 16 }));
            }
            let (wide, short) = gemm_names(qt);
            kernels.push(wide);
            if let Some(sh) = short.filter(|_| cols <= 16 * 256) {
                kernels.push(sh);
            }
            if qt == Q4_K {
                kernels.push("gemmf_q4_k".into()); // the non-exact fast form, for comparison
            }
            // Rounds over all kernels (interleaved: clocks drift as the GPU
            // heats up), best time of each.
            let mut best = vec![f64::INFINITY; kernels.len()];
            for _round in 0..8 {
                for (ki, k) in kernels.iter().enumerate() {
                    let k: &str = k;
                    let f = km.function(k).unwrap();
                    let st = act_stride_t(cols) as u64;
                    let args = [
                        wb.arg(),
                        Arg::U64(rb as u64),
                        Arg::I32(rows as i32),
                        Arg::I32(cols as i32),
                        act.arg(),
                        Arg::U64(st),
                        Arg::I32(nb as i32),
                        Arg::Ptr(0),
                        Arg::I32(0),
                        out.arg(),
                        Arg::U64(rows as u64),
                    ];
                    {
                        let e0 = g.record().unwrap();
                        // SAFETY: as in gpu_batched.
                        unsafe {
                            if k.starts_with("gemvm") {
                                g.launch(
                                    &f,
                                    (rows.div_ceil(16) as u32, 1, 1),
                                    (128, 1, 1),
                                    0,
                                    &args,
                                )
                                .unwrap();
                            } else if k.starts_with("gemm") {
                                let (bn, sm) = if k.starts_with("gemm1")
                                    || k.starts_with("gemms1")
                                    || k.starts_with("gemmf")
                                {
                                    (GM1_BN, GEMM1_SMEM)
                                } else {
                                    (GM_BN, GEMM_SMEM)
                                };
                                f.set_max_shared(sm).unwrap();
                                g.launch(
                                    &f,
                                    (nb.div_ceil(bn) as u32, rows.div_ceil(GM_BM) as u32, 1),
                                    (256, 1, 1),
                                    sm as u32,
                                    &args,
                                )
                                .unwrap();
                            } else {
                                let (t, r) = (if k.ends_with("x16") { 16 } else { 8 }, 2);
                                let mut a = vec![Arg::I32(qt as i32)];
                                a.extend_from_slice(&args);
                                g.launch(
                                    &f,
                                    (rows.div_ceil(r) as u32, nb.div_ceil(t) as u32, 1),
                                    (GEMV_THREADS, 1, 1),
                                    0,
                                    &a,
                                )
                                .unwrap();
                            }
                        }
                        let e1 = g.record().unwrap();
                        best[ki] = best[ki].min(e0.ms_to(&e1).unwrap() as f64);
                    }
                }
            }
            for (ki, k) in kernels.iter().enumerate() {
                let best = best[ki];
                let macs = (rows * cols * nb) as f64;
                eprintln!(
                    "{what:<14} nb {nb:>3} {k:<10} {:>8.1} us {:>6.1} TMAC/s {:>5.0} GB/s weights",
                    best * 1e3,
                    macs / (best * 1e-3) / 1e12,
                    (rb * rows) as f64 / (best * 1e-3) / 1e9
                );
            }
        }
    }
}

/// ~n tokens of mixed text (PROMPTS cycled), for prefill tests.
fn long_ids(m: &Model, n: usize) -> Vec<u32> {
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let mut ids = Vec::new();
    let mut i = 0;
    while ids.len() < n {
        ids.extend(tok.encode(&format!("{} ", PROMPTS[i % PROMPTS.len()])));
        i += 1;
    }
    ids.truncate(n);
    ids
}

/// KV cache bytes, positions 0..n of every layer.
fn kv_bytes(net: &CudaQwen2, c: &CudaCache, n: usize) -> Vec<u16> {
    let all: Vec<u16> = c.kv.read().unwrap();
    let row = net.kv_dim();
    let mut out = Vec::new();
    for l in 0..2 * net.n_layers {
        let base = l * c.max_ctx * row;
        out.extend_from_slice(&all[base..base + n * row]);
    }
    out
}

/// Prefill through feed_batch in `sizes` (cycled) chunks, then `forward`
/// of the last token: (KV bytes, logits).
fn prefill_with(net: &CudaQwen2, ids: &[u32], sizes: &[usize]) -> (Vec<u16>, Vec<f32>) {
    let mut c = net.new_cache(ids.len() + 1);
    let (&last, rest) = ids.split_last().unwrap();
    let mut pos = 0;
    for &s in sizes.iter().cycle() {
        if pos == rest.len() {
            break;
        }
        let e = (pos + s).min(rest.len());
        if s == 0 {
            // Per token through the decode step (graph), the reference.
            net.feed(&mut c, rest[pos], pos).unwrap();
            pos += 1;
            continue;
        }
        net.feed_batch(&rest[pos..e], pos, &mut c);
        pos = e;
    }
    let l = net.forward(last, rest.len(), &mut c);
    (kv_bytes(net, &c, ids.len()), l)
}

/// The batched prefill's contract: whatever the batching (1, 7, 64, the
/// max, ragged, per-token decode steps), the KV cache bytes and the next
/// token's logit bits are identical.
fn prefill_bits(m: &Model, name: &str) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module(&g).unwrap();
    let ids = long_ids(m, 700);
    let mut net = CudaQwen2::new(&g, &km, m, ids.len() + 1).unwrap();
    let max = net.prefill_batch();
    assert!(ids.len() > max, "covers chunking past the max batch");
    // The GPU's own attention splits (every split <= 32 positions here:
    // attn_warp_b), then few splits, so that queries past 32 * 8 positions
    // take attn_split_b, and batches straddle the two.
    for max_split in [net.max_split, 8] {
        net.max_split = max_split;
        let (kv0, l0) = prefill_with(&net, &ids, &[0]);
        for sizes in [
            vec![1],
            vec![7],
            vec![64],
            vec![max],
            vec![5, 100, 3, 17, 260, 1, 33],
            vec![16, 0, 9, 0, 0, 200],
        ] {
            let (kv, l) = prefill_with(&net, &ids, &sizes);
            let bad = kv.iter().zip(&kv0).position(|(a, b)| a != b);
            assert!(
                bad.is_none(),
                "{name} {sizes:?} split {max_split}: KV differs at {bad:?}"
            );
            assert_eq!(
                bits(&l),
                bits(&l0),
                "{name} {sizes:?} split {max_split}: logits"
            );
        }
    }
    eprintln!(
        "{name}: prefill bit-identical over 7 batchings of {} tokens, 2 split counts",
        ids.len()
    );
}

#[test]
fn gpu_prefill_batchings_bit_identical_0_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    prefill_bits(&m, "0.5B");
}

#[test]
fn gpu_prefill_batchings_bit_identical_1_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    prefill_bits(&m, "1.5B");
}

/// verify_argmax's picks equal forward_argmax per token for drafts that
/// are right, wrong and random, small (gemvn) and large (GEMM, chunked past
/// the max batch) passes.
fn verify_bits(m: &Model, name: &str) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module(&g).unwrap();
    let prompt = long_ids(m, 90);
    let net = CudaQwen2::new(&g, &km, m, 1400).unwrap();
    let (right, _) = greedy(&net, &prompt, 20, false);
    let mut rng = Rng(77);
    let mut cases: Vec<Vec<u32>> = Vec::new();
    for n in 1..=16 {
        let mut d = vec![*prompt.last().unwrap()];
        d.extend_from_slice(&right[..n - 1]);
        cases.push(d.clone()); // all right
        if n > 2 {
            d[n / 2] = (rng.next() % net.vocab as u64) as u32; // wrong from the middle
            cases.push(d);
        }
        cases.push(
            (0..n)
                .map(|_| (rng.next() % net.vocab as u64) as u32)
                .collect(),
        );
    }
    cases.push((0..40).map(|_| (rng.next() % 5000) as u32).collect());
    cases.push((0..600).map(|_| (rng.next() % 5000) as u32).collect());
    let (_, rest) = prompt.split_last().unwrap();
    for toks in &cases {
        let start = rest.len();
        let mut a = net.new_cache(start + toks.len() + 1);
        let mut b = net.new_cache(start + toks.len() + 1);
        net.feed_batch(rest, 0, &mut a);
        net.feed_batch(rest, 0, &mut b);
        // Stale slots: a different pass over the same positions first.
        let junk: Vec<u32> = toks.iter().map(|t| (t + 1) % 1000).collect();
        net.verify_argmax(&junk, start, &mut a);
        // Passes of up to 64 tokens (one pass: the cache's max batch is
        // larger) keep every row's logits, compared bit for bit below.
        let rows = toks.len() <= 64;
        if rows {
            let bb = a.b.as_mut().unwrap();
            bb.logits = g.alloc(64 * net.vocab * 4).unwrap();
            bb.logits_rows = true;
        }
        let got = net.verify_argmax(toks, start, &mut a);
        if rows {
            let all: Vec<f32> = a.b.as_ref().unwrap().logits.read().unwrap();
            for (i, &t) in toks.iter().enumerate() {
                let want = net.forward(t, start + i, &mut b);
                let gr = &all[i * net.vocab..(i + 1) * net.vocab];
                assert_eq!(
                    bits(gr),
                    bits(&want),
                    "{name}: logits of verify row {i}/{}",
                    toks.len()
                );
                assert_eq!(got[i], crate::argmax(&want), "{name}: pick {i}");
            }
        } else {
            let want: Vec<u32> = toks
                .iter()
                .enumerate()
                .map(|(i, &t)| net.forward_argmax(t, start + i, &mut b))
                .collect();
            assert_eq!(got, want, "{name}: verify of {} tokens", toks.len());
        }
        assert_eq!(
            kv_bytes(&net, &a, start + toks.len()),
            kv_bytes(&net, &b, start + toks.len()),
            "{name}: KV after verify of {}",
            toks.len()
        );
    }
    eprintln!(
        "{name}: {} verify passes == per-token picks (logit bits up to 64)",
        cases.len()
    );
}

#[test]
fn gpu_verify_matches_forward_argmax_0_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    verify_bits(&m, "0.5B");
}

#[test]
fn gpu_verify_matches_forward_argmax_1_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    verify_bits(&m, "1.5B");
}

/// Where a batched prefill pass spends its time, per kernel kind (events
/// between launches; no overlap). AXON_PROF_MODEL picks the model file
/// (default the 0.5B), AXON_PROF_NB the batch, AXON_PROF_START its position.
#[test]
#[ignore]
fn cuda_prefill_profile() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let rel = std::env::var("AXON_PROF_MODEL").unwrap_or("models/qwen2.5-0.5b-q4_k_m.gguf".into());
    let Some(m) = model_at(&rel) else {
        return;
    };
    let nb: usize = std::env::var("AXON_PROF_NB").map_or(512, |v| v.parse().unwrap());
    let start: usize = std::env::var("AXON_PROF_START").map_or(0, |v| v.parse().unwrap());
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::new(&g, &km, &m, start + nb + 1).unwrap();
    let mut c = net.new_cache(start + nb + 1);
    let ids = long_ids(&m, start + nb);
    if start > 0 {
        net.feed_batch(&ids[..start], 0, &mut c);
    }
    for round in 0..3 {
        let mut evs = vec![("start", g.record().unwrap())];
        net.pass(&mut c, &ids[start..], start, nb <= 16, &mut |name| {
            evs.push((name, g.record()?));
            Ok(())
        })
        .unwrap();
        g.sync().unwrap();
        if round < 2 {
            continue;
        }
        let mut agg: Vec<(&str, f64, usize)> = Vec::new();
        for i in 1..evs.len() {
            let ms = evs[i - 1].1.ms_to(&evs[i].1).unwrap() as f64;
            match agg.iter_mut().find(|a| a.0 == evs[i].0) {
                Some(a) => {
                    a.1 += ms;
                    a.2 += 1
                }
                None => agg.push((evs[i].0, ms, 1)),
            }
        }
        let total: f64 = agg.iter().map(|a| a.1).sum();
        eprintln!(
            "{rel} nb {nb} at {start}: {total:.2} ms = {:.0} tok/s",
            nb as f64 / total * 1e3
        );
        for (n, ms, k) in agg {
            eprintln!(
                "  {n:<8} {ms:>8.3} ms {:>5.1}% ({k} launches)",
                100.0 * ms / total
            );
        }
    }
}

/// The GEMMs' hardware f16 -> f32 conversion of block scales is exact for
/// how they use it (`h2f_mul_check`): no half, NaNs included, multiplies
/// to different bits than with quant::f16_to_f32's exact conversion.
#[test]
fn gpu_hw_f16_scales_exact_in_products() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let f = km.function("h2f_mul_check").unwrap();
    let out = g.upload(&[0u32]).unwrap();
    // SAFETY: h2f_mul_check's signature; out holds one u32.
    unsafe {
        g.launch(&f, (256, 1, 1), (256, 1, 1), 0, &[out.arg()])
            .unwrap()
    };
    let n: Vec<u32> = out.read().unwrap();
    assert_eq!(n[0], 0, "products differing by conversion");
}

/// Batched attention speed, 7B shape (28 heads, 4 kv heads, 128 wide), 64
/// queries at `start`: the warp-per-split kernel vs the decode block per
/// split, and the combine. Interleaved rounds, best of each.
#[test]
#[ignore]
fn cuda_attn_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let (nh, nkv, hd, nz) = (28usize, 4usize, 128usize, 64usize);
    let start: usize = std::env::var("AXON_PROF_START").map_or(1536, |v| v.parse().unwrap());
    let ctx = start + nz;
    let max_split = (8 * g.info().unwrap().sms as usize / nkv).clamp(1, MAX_SPLIT);
    let mut rng = Rng(3);
    let kvh: Vec<u16> = (0..ctx * nkv * hd)
        .map(|_| quant::f32_to_f16(rng.f() * 0.5))
        .collect();
    let kc = g.upload(&kvh).unwrap();
    let vc = g.upload(&kvh).unwrap();
    let q = g.upload(&rng.vec(nz * nh * hd, 1.0)).unwrap();
    let po = g.alloc(nz * nh * max_split * hd * 4).unwrap();
    let pm = g.alloc(nz * nh * max_split * 2 * 4).unwrap();
    let act = g.alloc(nz * act_stride_t(nh * hd)).unwrap();
    let names = ["attn_warp_b_128", "attn_split_b_128", "attn_combine_b"];
    let fs: Vec<Func> = names.iter().map(|n| km.function(n).unwrap()).collect();
    for (n, f) in names.iter().zip(&fs) {
        eprintln!("{n}: regs, local {:?}", f.regs_local().unwrap());
    }
    let mut best = [f64::INFINITY; 3];
    for _ in 0..6 {
        for (k, f) in fs.iter().enumerate() {
            let e0 = g.record().unwrap();
            // SAFETY: prefill.cu's signatures; buffers sized for nz queries
            // over ctx positions.
            unsafe {
                if k < 2 {
                    let (gy, bx) = if k == 0 {
                        (max_split.div_ceil(4), 128)
                    } else {
                        (max_split, ATT_TILE)
                    };
                    g.launch(
                        f,
                        (nkv as u32, gy as u32, nz as u32),
                        (bx as u32, 1, 1),
                        0,
                        &[
                            q.arg(),
                            kc.arg(),
                            vc.arg(),
                            Arg::I32(start as i32),
                            Arg::I32(0),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            Arg::F32(1.0 / (hd as f32).sqrt()),
                            Arg::I32(max_split as i32),
                            po.arg(),
                            pm.arg(),
                        ],
                    )
                    .unwrap();
                } else {
                    g.launch(
                        f,
                        ((nh * hd / 256) as u32, nz as u32, 1),
                        (256, 1, 1),
                        0,
                        &[
                            po.arg(),
                            pm.arg(),
                            Arg::I32(start as i32),
                            Arg::I32(0),
                            Arg::I32((nh * hd) as i32),
                            Arg::I32(hd as i32),
                            Arg::I32(max_split as i32),
                            Arg::I32(0),
                            act.arg(),
                            Arg::U64(act_stride_t(nh * hd) as u64),
                        ],
                    )
                    .unwrap();
                }
            }
            let e1 = g.record().unwrap();
            best[k] = best[k].min(e0.ms_to(&e1).unwrap() as f64);
        }
    }
    for (n, b) in names.iter().zip(best) {
        eprintln!("{n:<18} {nz} queries at {start}: {:>8.1} us", b * 1e3);
    }
}

/// Batched attention (attn_split_b, attn_warp_b, attn_combine_b) gives each
/// query the decode kernels' bits: split partials (m, l, sums) and the
/// quantized output, for every layout here (GQA groups 6, 7, 8; heads of 64
/// and 128) and queries whose splits are short (warp path), long (one and
/// several tiles), and on both sides of the warp path's limit.
#[test]
fn gpu_batched_attention_matches_decode_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let (fc, fcb) = (
        km.function("attn_combine").unwrap(),
        km.function("attn_combine_b").unwrap(),
    );
    let mut rng = Rng(9191);
    // (n_heads, n_kv, hd): 0.5B, 1.5B, 3B (group 8), 7B.
    for (nh, nkv, hd) in [(14, 2, 64), (12, 2, 128), (16, 2, 128), (28, 4, 128)] {
        let (dim, kvd) = (nh * hd, nkv * hd);
        let fs = km.function(&format!("attn_split_{hd}")).unwrap();
        let fsb = km.function(&format!("attn_split_b_{hd}")).unwrap();
        let fw = km.function(&format!("attn_warp_b_{hd}")).unwrap();
        // (start, queries, max_split)
        for (start, nz, ms) in [(0, 40, 23), (200, 70, 8), (1000, 12, 46), (3000, 6, 23)] {
            let ctx = start + nz;
            let to_h = |v: Vec<f32>| -> Vec<u16> { v.into_iter().map(quant::f32_to_f16).collect() };
            let kb = g.upload(&to_h(rng.vec(ctx * kvd, 4.0))).unwrap();
            let vb = g.upload(&to_h(rng.vec(ctx * kvd, 1.0))).unwrap();
            let q = rng.vec(nz * dim, 1.0);
            let qb = g.upload(&q).unwrap();
            let (po_n, pm_n) = (nh * ms * hd, nh * ms * 2);
            let ast = act_stride_t(dim);
            let scale = Arg::F32(1.0 / (hd as f32).sqrt());
            // Batched: the split kernel for all queries, the warp kernel for
            // those within its limit (np <= 32 ms), each into its own buffers.
            let nsplit = |np: usize| ms.min(np.div_ceil(32));
            let warp_n = (0..nz).filter(|&z| start + z < 32 * ms).count();
            let mut batched = Vec::new();
            for (f, n) in [(&fsb, nz), (&fw, warp_n)] {
                if n == 0 {
                    continue;
                }
                let po = g.alloc(n * po_n * 4).unwrap();
                let pm = g.alloc(n * pm_n * 4).unwrap();
                let act = g.alloc(n * ast).unwrap();
                act.zero().unwrap(); // parts no kernel writes (n % 256 != 0) compare equal
                let warp = std::ptr::eq(f, &fw);
                // SAFETY: prefill.cu's signatures; buffers for n queries over
                // ctx positions, K/V of ctx rows.
                unsafe {
                    g.launch(
                        f,
                        (
                            nkv as u32,
                            if warp { ms.div_ceil(4) } else { ms } as u32,
                            n as u32,
                        ),
                        (if warp { 128 } else { ATT_TILE as u32 }, 1, 1),
                        0,
                        &[
                            qb.arg(),
                            kb.arg(),
                            vb.arg(),
                            Arg::I32(start as i32),
                            Arg::I32(0),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            scale,
                            Arg::I32(ms as i32),
                            po.arg(),
                            pm.arg(),
                        ],
                    )
                    .unwrap();
                    g.launch(
                        &fcb,
                        ((dim / 256) as u32, n as u32, 1),
                        (256, 1, 1),
                        0,
                        &[
                            po.arg(),
                            pm.arg(),
                            Arg::I32(start as i32),
                            Arg::I32(0),
                            Arg::I32(dim as i32),
                            Arg::I32(hd as i32),
                            Arg::I32(ms as i32),
                            Arg::I32(dim.is_multiple_of(256) as i32),
                            act.arg(),
                            Arg::U64(ast as u64),
                        ],
                    )
                    .unwrap();
                }
                let (po, pm, act): (Vec<f32>, Vec<f32>, Vec<u8>) =
                    (po.read().unwrap(), pm.read().unwrap(), act.read().unwrap());
                batched.push((warp, n, po, pm, act));
            }
            for z in 0..nz {
                let np = start + z + 1;
                let qz = g.upload(&q[z * dim..(z + 1) * dim]).unwrap();
                let params = g.upload(&[0u32, (np - 1) as u32, 0, 0]).unwrap();
                let po = g.alloc(po_n * 4).unwrap();
                let pm = g.alloc(pm_n * 4).unwrap();
                let act = g.alloc(act_bytes(dim)).unwrap();
                act.zero().unwrap();
                // SAFETY: decode.cu's signatures, as gpu_attention_matches_cpu.
                unsafe {
                    g.launch(
                        &fs,
                        (nkv as u32, ms as u32, 1),
                        (ATT_TILE as u32, 1, 1),
                        0,
                        &[
                            qz.arg(),
                            kb.arg(),
                            vb.arg(),
                            params.arg(),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            scale,
                            Arg::I32(ms as i32),
                            po.arg(),
                            pm.arg(),
                        ],
                    )
                    .unwrap();
                    g.launch(
                        &fc,
                        ((dim / 256) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        &[
                            po.arg(),
                            pm.arg(),
                            params.arg(),
                            Arg::I32(dim as i32),
                            Arg::I32(hd as i32),
                            Arg::I32(ms as i32),
                            Arg::I32(dim.is_multiple_of(256) as i32),
                            act.arg(),
                            Arg::Ptr(0),
                        ],
                    )
                    .unwrap();
                }
                let (wpo, wpm, wact): (Vec<f32>, Vec<f32>, Vec<u8>) =
                    (po.read().unwrap(), pm.read().unwrap(), act.read().unwrap());
                for (warp, n, bpo, bpm, bact) in &batched {
                    if z >= *n {
                        continue;
                    }
                    let ctxs = format!("{nh}/{nkv}x{hd} np {np} splits {ms} warp {warp}");
                    for h in 0..nh {
                        for s in 0..nsplit(np) {
                            let o = (h * ms + s) * hd;
                            assert_eq!(
                                bits(&bpo[z * po_n + o..z * po_n + o + hd]),
                                bits(&wpo[o..o + hd]),
                                "{ctxs}: sums head {h} split {s}"
                            );
                            let m = (h * ms + s) * 2;
                            assert_eq!(
                                bits(&bpm[z * pm_n + m..z * pm_n + m + 2]),
                                bits(&wpm[m..m + 2]),
                                "{ctxs}: m, l head {h} split {s}"
                            );
                        }
                    }
                    let ab = act_bytes(dim);
                    assert!(bact[z * ast..z * ast + ab] == wact[..], "{ctxs}: output");
                }
            }
        }
    }
}
