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
pub(crate) struct Agree {
    pub(crate) positions: usize,
    pub(crate) top1: usize,
    pub(crate) max_kl: f64,
    /// Positions whose KL exceeds 0.5 (`assert_graded`): (position index,
    /// KL, the reference's top-2 logit margin there, the previous
    /// position's KL).
    pub(crate) over_half: Vec<(usize, f64, f32, f64)>,
    /// Indexes of the positions whose top-1 disagrees.
    pub(crate) flip_at: Vec<usize>,
    /// The previous position's KL (within one sequence).
    last_kl: f64,
    pub(crate) sum_kl: f64,
    max_abs: f32,
    /// Reference logit margin of each top-1 disagreement.
    flips: Vec<f32>,
}

/// Reference `r` vs `o` logits after the prompt `ids` and after each of
/// `steps` greedy tokens of `r`, fed to both.
pub(crate) fn compare<R: Lm, O: Lm>(r: &R, o: &O, ids: &[u32], steps: usize, a: &mut Agree) {
    let mut rc = r.new_cache(ids.len() + steps + 1);
    let mut oc = o.new_cache(ids.len() + steps + 1);
    let (&last, rest) = ids.split_last().unwrap();
    r.feed_batch(rest, 0, &mut rc);
    o.feed_batch(rest, 0, &mut oc);
    compare_from(r, &mut rc, o, &mut oc, last, rest.len(), steps, a);
}

/// `compare` from caches already holding positions 0..pos: `last` at pos,
/// then `steps` greedy tokens of `r`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_from<R: Lm, O: Lm>(
    r: &R,
    rc: &mut R::Cache,
    o: &O,
    oc: &mut O::Cache,
    last: u32,
    pos: usize,
    steps: usize,
    a: &mut Agree,
) {
    let mut tok = last;
    a.last_kl = 0.0;
    for s in 0..=steps {
        let pos = pos + s;
        let lr = r.forward(tok, pos, rc);
        let lo = o.forward(tok, pos, oc);
        let (ar, ao) = (crate::argmax(&lr), crate::argmax(&lo));
        a.positions += 1;
        a.top1 += (ar == ao) as usize;
        let k = kl(&lr, &lo);
        a.max_kl = a.max_kl.max(k);
        if k > 0.5 {
            a.over_half
                .push((a.positions, k, top2_margin(&lr), a.last_kl));
        }
        a.last_kl = k;
        a.sum_kl += k;
        for (x, y) in lr.iter().zip(&lo) {
            a.max_abs = a.max_abs.max((x - y).abs());
        }
        if ar != ao {
            // How close a call it was, in the reference's logits.
            a.flips.push(lr[ar as usize] - lr[ao as usize]);
            a.flip_at.push(a.positions);
        }
        tok = ar;
    }
}

/// The gap between the two largest logits.
fn top2_margin(l: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in l {
        if x > a {
            (a, b) = (x, a);
        } else if x > b {
            b = x;
        }
    }
    a - b
}

pub(crate) const PROMPTS: [&str; 6] = [
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
    let km = Kernels::module_for(&g, m).unwrap();
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

pub(crate) fn report(name: &str, a: &Agree) {
    eprintln!(
        "{name}: top-1 {}/{} ({:.1}%), KL mean {:.2e} max {:.2e} (> 0.5: {:?}), max |dlogit| \
         {:.3}, flip margins {:?}",
        a.top1,
        a.positions,
        100.0 * a.top1 as f64 / a.positions as f64,
        a.sum_kl / a.positions as f64,
        a.max_kl,
        a.over_half,
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
/// so they get absolute gross-bug bounds (a wrong kernel gives KL ~1-10 at
/// many positions and logit errors ~10): KL above 0.5 at one position at
/// most, and only where the CPU's two kernel sets flip too; above 2 at none
/// (`assert_graded` has the measurements); max |dlogit| below 6.
fn logits_match(m: &Model, name: &str) {
    let (ag, ac) = agreement(m, 32);
    report(&format!("{name} gpu vs cpu"), &ag);
    report(&format!("{name} cpu rows vs cpu"), &ac);
    assert_graded(&ag, &ac);
}

/// `logits_match`'s bounds: `ag` (a backend vs the CPU) against `ac` (the
/// CPU's per-row kernels vs its default ones), over the same positions
/// (index for index: the single-position exemption below looks `ac` up).
pub(crate) fn assert_graded(ag: &Agree, ac: &Agree) {
    assert_graded_floor(ag, ac, 94);
}

/// `assert_graded` with top-1 at least `floor` % (natural text, prose and
/// code, has more near-ties: the CPU agrees with itself on only 93.8 % of
/// natural_long's 1.5B positions).
pub(crate) fn assert_graded_floor(ag: &Agree, ac: &Agree, floor: usize) {
    let n = ag.positions;
    // Top-1: at most 2% of positions behind the CPU's self-agreement.
    assert!(
        ag.top1 * 100 + 2 * n >= ac.top1 * 100 && ag.top1 * 100 >= floor * n,
        "top-1 agreement {}/{n} (cpu vs itself {})",
        ag.top1,
        ac.top1
    );
    // Only near-ties flip. Flipping a margin m takes the top two logits
    // moving m apart, which the reference's own spread (`ac.max_abs`, the
    // CPU's two kernel sets: 1.26 on 1.5B natural text, 1.9-2.3 on chat
    // histories) does too: the CPU flipped a 0.86 margin against itself on
    // a chat history (sm_120), and sm_89's tensor-core attention (its mma
    // sums f32 in another order than sm_120's) flips 0.73 (1.5B natural)
    // and 0.92 / 0.82 (0.5B streamed chat) with |dlogit| statistics no
    // worse than sm_120's. So: every flip below max(0.5, ac.max_abs), and
    // at most max(2, 1% of positions) of them at 0.5 or more.
    let tie = ac.max_abs.max(0.5);
    let wide = ag.flips.iter().filter(|&&d| d >= 0.5).count();
    assert!(
        ag.flips.iter().all(|&d| d < tie) && wide <= (n / 100).max(2),
        "a top-1 flip with margin >= {tie} or {wide} >= 0.5: {:?}",
        ag.flips
    );
    assert!(ag.sum_kl <= 2.0 * ac.sum_kl, "mean KL");
    // Extremes, per position. A wrong kernel moves many positions: a mask
    // off by one position (deliberately injected into the tensor-core
    // attention) put 48 (1.5B GPU) and 14 (hybrid k=1) of 198 positions
    // above 0.5, max 3.3-8.3 (0.5B: 36 / 35, max 4.8-5.1). Correct kernels
    // stay below 0.25 almost everywhere: base and r3 attention, GPU and
    // hybrid k=1, over three prompt sets (PROMPTS, six others, six windows
    // of natural text) on 0.5B and 1.5B, never above 0.17. The one
    // exception seen: the 1.5B hybrid k=1 here, KL 1.22 at a single
    // position (the one before 3e-4), which disappears (0.06) when only
    // the queries' f16 rounding changes - and at which the CPU's own
    // per-row kernels flip top-1 too (margin 0.32): a near-tie the
    // reference itself doesn't resolve stably. So: at most one position
    // above 0.5, none above 2, and that one only where `ac` (the CPU vs
    // its other kernel set) also flips.
    assert!(
        ag.over_half.len() <= 1 && ag.max_kl < 2.0,
        "max KL {} (above 0.5 (index, KL, reference margin, previous KL): {:?})",
        ag.max_kl,
        ag.over_half
    );
    for &(i, k, mg, _) in &ag.over_half {
        assert!(
            ac.flip_at.contains(&i),
            "KL {k} at position {i} (reference margin {mg}), where the CPU's kernel sets agree"
        );
    }
    assert!(ag.max_abs < 6.0, "max logit difference");
}

/// The GPU's head sizes (no GPU or model needed): 64 and 128, gemma3 256
/// only (its window-taking kernels); a gemma3 model of head size 128 (27B)
/// or 64 is refused at load, not launched short of an argument.
#[test]
fn gpu_head_sizes() {
    for hd in [64, 128] {
        assert!(gpu_head_ok(false, hd, 7).is_ok(), "head {hd}");
        let e = gpu_head_ok(true, hd, 2).unwrap_err();
        assert!(
            e.contains(&format!("head size {hd}")) && e.contains("--device cpu"),
            "{e}"
        );
    }
    assert!(gpu_head_ok(false, 256, 1).is_err(), "head 256 off gemma3");
    assert!(gpu_head_ok(true, 256, 2).is_ok(), "gemma3 head 256");
    assert!(gpu_head_ok(true, 256, MAX_GROUP / 2).is_ok());
    assert!(gpu_head_ok(true, 256, MAX_GROUP / 2 + 1).is_err());
    assert!(gpu_head_ok(false, 96, 1).is_err());
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

/// `logits_match` on the Q5_K_M / Q4_0 GGUFs (bartowski's; skipped when
/// absent): Q5_K, Q5_1 (0.5B Q5_K_M), Q4_0 and Q4_1 (Q4_0 files' few
/// ffn_down) end to end, GPU vs CPU.
#[test]
fn gpu_logits_match_cpu_q5_k_m_q4_0() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    for (f, name) in [
        ("Qwen2.5-0.5B-Instruct-Q5_K_M.gguf", "0.5B Q5_K_M"),
        ("Qwen2.5-0.5B-Instruct-Q4_0.gguf", "0.5B Q4_0"),
        ("Qwen2.5-1.5B-Instruct-Q5_K_M.gguf", "1.5B Q5_K_M"),
        ("Qwen2.5-1.5B-Instruct-Q4_0.gguf", "1.5B Q4_0"),
        ("Qwen2.5-1.5B-Instruct-Q3_K_M.gguf", "1.5B Q3_K_M"),
        ("Llama-3.2-1B-Instruct-Q3_K_L.gguf", "Llama 1B Q3_K_L"),
        ("Qwen2.5-0.5B-Instruct-Q2_K.gguf", "0.5B Q2_K"),
        ("Qwen2.5-0.5B-Instruct-Q3_K_M.gguf", "0.5B Q3_K_M"),
    ] {
        let Some(m) = model_at(&format!("models/{f}")) else {
            continue;
        };
        let (ag, ac) = agreement(&m, 32);
        report(&format!("{name} gpu vs cpu"), &ag);
        report(&format!("{name} cpu rows vs cpu"), &ac);
        let n = ag.positions as f64;
        if ac.sum_kl / n > 1e-4 {
            assert_graded(&ag, &ac);
        } else {
            // The CPU's two kernel sets nearly agree here (Q4_0 / Q4_1 have
            // no repack: only the head differs), so no self-noise to grade
            // against: absolute bounds, about 2x what these files measure
            // (GPU vs CPU, 0.5B / 1.5B Q4_0: mean KL 1.5e-3 / 3.0e-3, max
            // 0.017 / 0.040, max |dlogit| 1.2 / 1.7).
            let n = ag.positions;
            assert!(ag.top1 * 100 >= 94 * n, "{name}: top-1 {}/{n}", ag.top1);
            assert!(ag.flips.iter().all(|&d| d < 0.5), "{name}: {:?}", ag.flips);
            assert!(ag.sum_kl <= 5e-3 * n as f64, "{name}: mean KL");
            assert!(ag.max_kl <= 0.2 && ag.max_abs <= 3.0, "{name}: extremes");
        }
    }
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

/// `gpu_gemv_matches_cpu_every_type` for the types the test model doesn't
/// have (Q5_K, Q4_0), on synthetic matrices of the models' widths: the
/// GEMV vs the CPU's dot_row_q8 on the same quantized activation, and the
/// GPU's element dequant (dequant_rows) vs quant::dequant, bit for bit.
#[test]
fn gpu_gemv_matches_cpu_new_types() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module_ext(&g).unwrap();
    let mut rng = Rng(0x0051_ce0f_4b17);
    for (qt, cols) in [
        (Q4_0, 896),
        (Q4_0, 4864),
        (Q5_K, 1536),
        (Q5_K, 8960),
        (Q4_1, 896),
        (Q5_1, 4864),
        (Q2_K, 1536),
        (Q2_K, 18944),
        (Q3_K, 3584),
        (Q3_K, 8960),
        (IQ4_NL, 896),
        (IQ4_NL, 4864),
    ] {
        let rows = 333;
        let w = synth_matrix(&mut rng, qt, rows, cols);
        let rb = quant::row_bytes(qt, cols);
        let wb = g.upload(&w).unwrap();
        let idx: Vec<u32> = (0..rows as u32).collect();
        let deq = Kernels::new(&km)
            .unwrap()
            .dequant_rows(qt, &wb, cols, &idx)
            .unwrap();
        let mut row = vec![0f32; cols];
        for r in 0..rows {
            quant::dequant(qt, &w[r * rb..(r + 1) * rb], &mut row);
            assert_eq!(
                bits(&row),
                bits(&deq[r * cols..(r + 1) * cols]),
                "dequant row {r}"
            );
        }
        for scale in [1.0f32, 1e-3, 40.0] {
            let x = rng.vec(cols, scale);
            let got = gpu_gemv(&g, &km, "gemv", qt, &w, cols, &x);
            let got1 = gpu_gemv(&g, &km, "gemv1", qt, &w, cols, &x);
            assert_eq!(bits(&got), bits(&got1), "gemv1 vs gemv");
            let xq = quant::QuantizedAct::for_types(&x, &[qt]);
            let mut worst = 0f64;
            for r in 0..rows {
                let wr = &w[r * rb..(r + 1) * rb];
                let cpu = quant::dot_row_q8(qt, wr, &xq) as f64;
                quant::dequant(qt, wr, &mut row);
                let mag: f64 = row
                    .iter()
                    .zip(&x)
                    .map(|(&a, &b)| (a as f64 * b as f64).abs())
                    .sum();
                worst = worst.max((got[r] as f64 - cpu).abs() / mag.max(1e-30));
            }
            let name = crate::gguf::qtype(qt).name;
            eprintln!("{name} {cols} x{scale}: vs cpu q8 {worst:.2e}");
            assert!(worst < 1e-6, "{name} {cols}: {worst:e} vs the CPU");
        }
    }
}

/// The two kernel builds (`Kernels::module`, main's four types, and
/// `module_ext`, every type) give the same decode GEMV bits for the four:
/// the extra switch cases change register allocation, not the arithmetic.
#[test]
fn gpu_ext_module_same_bits_old_types() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let kx = Kernels::module_ext(&g).unwrap();
    assert!(!km.ext() && kx.ext());
    let mut rng = Rng(0x0e47_0b17);
    for (qt, cols) in [(Q4_K, 4864), (Q6_K, 1536), (Q5_0, 896), (Q8_0, 896)] {
        let w = synth_matrix(&mut rng, qt, 333, cols);
        for scale in [1.0f32, 40.0] {
            let x = rng.vec(cols, scale);
            for k in ["gemv", "gemv1"] {
                let a = gpu_gemv(&g, &km, k, qt, &w, cols, &x);
                let b = gpu_gemv(&g, &kx, k, qt, &w, cols, &x);
                assert_eq!(bits(&a), bits(&b), "{k} type {qt}");
            }
        }
    }
}

/// The decode step's combine (decode.cu attn_cmb): bufs = (part_o,
/// part_ml, params), dims = (outputs, head size, the partials' stride);
/// quantized into act (want_k when the outputs are whole 256s), f32 into y.
fn launch_cmb(g: &Gpu, f: &Func, bufs: [&DevBuf; 3], dims: [usize; 3], act: &DevBuf, y: &DevBuf) {
    let [dim, hd, stride] = dims;
    let cnt = g.alloc(dim.div_ceil(256) * 4).unwrap();
    cnt.zero().unwrap();
    // SAFETY: decode.cu attn_cmb's signature; the partials hold dim / hd
    // heads of `stride` blocks, act and y dim outputs.
    unsafe {
        g.launch(
            f,
            ((dim / 64) as u32, 1, 1),
            (64, 1, 1),
            0,
            &[
                bufs[0].arg(),
                bufs[1].arg(),
                bufs[2].arg(),
                Arg::I32(dim as i32),
                Arg::I32(hd as i32),
                Arg::I32(stride as i32),
                Arg::I32(dim.is_multiple_of(256) as i32),
                act.arg(),
                y.arg(),
                cnt.arg(),
            ],
        )
        .unwrap();
    }
    g.sync().unwrap();
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
    // (n_heads, n_kv, hd): 0.5B, 1.5B, 3B, 7B; Llama 1B, Qwen3 0.6B.
    for (nh, nkv, hd) in [
        (14, 2, 64),
        (12, 2, 128),
        (16, 2, 128),
        (28, 4, 128),
        (32, 8, 64),
        (16, 8, 128),
    ] {
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
                        // No q/k norms (QkNorm: kraw, qn, kn, eps, rope, kc_out).
                        Arg::Ptr(0),
                        Arg::Ptr(0),
                        Arg::Ptr(0),
                        Arg::F32(0.0),
                        Arg::Ptr(0),
                        Arg::Ptr(0),
                        Arg::Ptr(0), // no page table
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

/// The tensor-core decode attention (attn_blk + attn_cmb: f16 queries and
/// probabilities, f32 accumulation) against an f64 reference on the same
/// f16 keys and values: within the f16 roundings' reach (|v| < 1, peaky
/// softmax), at positions on both sides of 64-position blocks, grids
/// smaller than the block count, and contexts past 8k.
#[test]
fn gpu_tc_attention_close_to_f64() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    if !km.mma() {
        return;
    }
    let fc = km.function("attn_cmb").unwrap();
    let fc256 = km.function("attn_cmb256").unwrap();
    let mut rng = Rng(4243);
    let mut worst = 0f64;
    for (nh, nkv, hd) in [(14, 2, 64), (12, 2, 128), (16, 2, 128), (28, 4, 128)] {
        let (dim, kvd, group) = (nh * hd, nkv * hd, nh / nkv);
        let fs = km.function(&format!("attn_blk_{hd}")).unwrap();
        for (np, gy) in [
            (1, 23),
            (2, 1),
            (63, 46),
            (64, 4),
            (65, 23),
            (1000, 1),
            (2000, 184),
            (4099, 23),
            (9001, 46),
        ] {
            let q = rng.vec(dim, 1.0);
            let kh: Vec<u16> = rng
                .vec(np * kvd, 4.0)
                .into_iter()
                .map(quant::f32_to_f16)
                .collect();
            let vh: Vec<u16> = rng
                .vec(np * kvd, 1.0)
                .into_iter()
                .map(quant::f32_to_f16)
                .collect();
            let (qb, kb, vb) = (
                g.upload(&q).unwrap(),
                g.upload(&kh).unwrap(),
                g.upload(&vh).unwrap(),
            );
            let params = g.upload(&[0u32, (np - 1) as u32, 0, 0]).unwrap();
            let nblk_max = np.div_ceil(64);
            let po = g.alloc(nh * nblk_max * hd * 4).unwrap();
            let pml = g.alloc(nh * nblk_max * 2 * 4).unwrap();
            let act = g.alloc(act_bytes(dim)).unwrap();
            act.zero().unwrap();
            let out = g.alloc(dim * 4).unwrap();
            // SAFETY: decode.cu's signatures; buffers sized as `try_cache`
            // sizes them, K/V with np rows.
            unsafe {
                g.launch(
                    &fs,
                    (nkv as u32, gy.min(nblk_max) as u32, 1),
                    (32 * att_nw(hd, false), 1, 1),
                    0,
                    &[
                        qb.arg(),
                        kb.arg(),
                        vb.arg(),
                        params.arg(),
                        Arg::I32(nh as i32),
                        Arg::I32(nkv as i32),
                        Arg::F32(att_pre(hd)),
                        Arg::I32(nblk_max as i32),
                        po.arg(),
                        pml.arg(),
                        Arg::Ptr(0),
                        Arg::Ptr(0),
                        Arg::Ptr(0),
                        Arg::F32(0.0),
                        Arg::Ptr(0),
                        Arg::Ptr(0),
                        Arg::Ptr(0), // no page table
                    ],
                )
                .unwrap();
                launch_cmb(
                    &g,
                    &fc,
                    [&po, &pml, &params],
                    [dim, hd, nblk_max],
                    &act,
                    &out,
                );
            }
            // attn_cmb256 (the decode step's combine below 8192 positions)
            // on the same partials: the same bits, f32 and quantized.
            let act256 = g.alloc(act.len()).unwrap();
            act256.zero().unwrap();
            let out256 = g.alloc(out.len()).unwrap();
            // SAFETY: decode.cu attn_cmb256's signature, buffers as above.
            unsafe {
                g.launch(
                    &fc256,
                    (dim.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &[
                        po.arg(),
                        pml.arg(),
                        params.arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(nblk_max as i32),
                        Arg::I32(dim.is_multiple_of(256) as i32),
                        act256.arg(),
                        out256.arg(),
                    ],
                )
                .unwrap();
            }
            let got: Vec<f32> = out.read().unwrap();
            assert_eq!(
                bits(&got),
                bits(&out256.read::<f32>().unwrap()),
                "attn_cmb vs attn_cmb256, hd {hd} np {np}"
            );
            assert!(
                act.read::<u8>().unwrap() == act256.read::<u8>().unwrap(),
                "attn_cmb vs attn_cmb256 quantized, hd {hd} np {np}"
            );
            let f = |h: u16| quant::f16_to_f32(h) as f64;
            let mut de = 0f64;
            for h in 0..nh {
                let kvh = h / group;
                let s: Vec<f64> = (0..np)
                    .map(|t| {
                        (0..hd)
                            .map(|d| q[h * hd + d] as f64 * f(kh[t * kvd + kvh * hd + d]))
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let l: f64 = p.iter().sum();
                for d in 0..hd {
                    let e: f64 = (0..np)
                        .map(|t| p[t] * f(vh[t * kvd + kvh * hd + d]))
                        .sum::<f64>()
                        / l;
                    de = de.max((got[h * hd + d] as f64 - e).abs());
                }
            }
            worst = worst.max(de);
            assert!(
                de < 1e-2,
                "heads {nh}/{nkv} hd {hd} np {np} grid {gy}: vs f64 {de:e}"
            );
        }
    }
    eprintln!("tc attention: worst |out - f64| {worst:e}");
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
/// On qwen3 (q/k norms) the kernel leaves q and k raw, in q and kraw (the
/// attention normalizes and rotates them; logits_match, prefill_bits and
/// the oracle runs cover that): checked against the CPU's plain row dots.
#[test]
fn gpu_qkv_rope_matches_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    if let Some(m) = crate::testutil::model() {
        qkv_rope_check(&g, &m);
    }
    if let Some(m) = crate::testutil::qwen3_model() {
        qkv_rope_check(&g, &m);
    }
}

fn qkv_rope_check(g: &Gpu, m: &Model) {
    use crate::kv::{Half, KvElem};
    let km = Kernels::module(g).unwrap();
    let f = km.function("qkv_rope").unwrap();
    let fq = km.function("quantize").unwrap();
    let meta_or = |k: &str| meta(m, k);
    let meta = |k: &str| meta(m, k).unwrap();
    let (dim, nh, nkv) = (
        meta("embedding_length") as usize,
        meta("attention.head_count") as usize,
        meta("attention.head_count_kv") as usize,
    );
    let hd = meta_or("attention.key_length").map_or(dim / nh, |v| v as usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let kraw = g.alloc(kvd * 4).unwrap();
    let base = meta("rope.freq_base");
    let freq: Vec<f64> = (0..hd / 2)
        .map(|i| base.powf(-2.0 * i as f64 / hd as f64))
        .collect();
    let fb: Vec<u64> = freq.iter().map(|f| f.to_bits()).collect();
    let freq_b = g.upload(&fb).unwrap();
    let fr = km.function("rope_table").unwrap();
    let mut rng = Rng(31337);
    for layer in [0, 1] {
        let t = |s: &str| tensor(m, &format!("blk.{layer}.{s}")).unwrap();
        let t_opt = |s: &str| tensor_opt(m, &format!("blk.{layer}.{s}"));
        let (tq, tk, tv) = (t("attn_q.weight"), t("attn_k.weight"), t("attn_v.weight"));
        // Absent biases: null on the GPU, zeros (an exact no-op) on the CPU.
        let bias: Vec<Option<Vec<f32>>> = ["attn_q.bias", "attn_k.bias", "attn_v.bias"]
            .iter()
            .map(|s| t_opt(s).map(|t| vec_f32(m, t)))
            .collect();
        let norm = t_opt("attn_q_norm.weight").is_some();
        let wb: Vec<DevBuf> = [tq, tk, tv]
            .iter()
            .map(|t| g.upload(m.tensor_data(t)).unwrap())
            .collect();
        let up = |v: &Option<Vec<f32>>| v.as_ref().map(|v| g.upload(v).unwrap());
        let bb: Vec<Option<DevBuf>> = bias.iter().map(up).collect();
        let arg = |b: &Option<DevBuf>| b.as_ref().map_or(Arg::Ptr(0), |b| b.arg());
        for pos in [0usize, 3, 4097, 32767] {
            let x = rng.vec(dim, 1.0);
            let xb = g.upload(&x).unwrap();
            let act = g.alloc(act_bytes(dim)).unwrap();
            let params = g.upload(&[0u32, pos as u32, 0, 0]).unwrap();
            let q = g.alloc(qd * 4).unwrap();
            let rope = g.alloc(hd / 2 * 8).unwrap();
            let kc = g.alloc((pos + 1) * kvd * 2).unwrap();
            let vc = g.alloc((pos + 1) * kvd * 2).unwrap();
            let rb = |t: &Tensor| Arg::U64(quant::row_bytes(t.qtype.id, dim) as u64);
            let items = qd / 2 + kvd;
            // SAFETY: decode.cu's signatures; q holds qd floats, the
            // caches pos + 1 rows, kraw kvd floats;
            // matrices dim wide.
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
                        Arg::F32(1.0),
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
                        arg(&bb[0]),
                        arg(&bb[1]),
                        arg(&bb[2]),
                        Arg::I32(qd as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(dim as i32),
                        act.arg(),
                        rope.arg(),
                        params.arg(),
                        q.arg(),
                        kc.arg(),
                        vc.arg(),
                        if norm { kraw.arg() } else { Arg::Ptr(0) },
                        Arg::Ptr(0), // no page table
                    ],
                )
                .unwrap();
            }
            let gq: Vec<f32> = q.read().unwrap();
            let gk: Vec<u16> = kc.read().unwrap();
            let gv: Vec<u16> = vc.read().unwrap();
            // CPU reference.
            let proj = |t: &Tensor, b: &Option<Vec<f32>>| -> Vec<f32> {
                let xq = quant::QuantizedAct::for_types(&x, &[t.qtype.id]);
                let rb = quant::row_bytes(t.qtype.id, dim);
                let d = m.tensor_data(t);
                (0..t.dims[1] as usize)
                    .map(|r| {
                        quant::dot_row_q8(t.qtype.id, &d[r * rb..(r + 1) * rb], &xq)
                            + b.as_ref().map_or(0.0, |b| b[r])
                    })
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
            if !norm {
                rope(&mut cq);
                rope(&mut ck);
            }
            let scale = cq.iter().fold(0f32, |a, v| a.max(v.abs()));
            for i in 0..qd {
                assert!(
                    (gq[i] - cq[i]).abs() <= 1e-5 * scale,
                    "layer {layer} pos {pos} q[{i}]: {} vs {}",
                    gq[i],
                    cq[i]
                );
            }
            let row = pos * kvd;
            if norm {
                // Raw keys, f32, as the queries.
                let gr: Vec<f32> = kraw.read().unwrap();
                let ks = ck.iter().fold(0f32, |a, v| a.max(v.abs()));
                for i in 0..kvd {
                    assert!(
                        (gr[i] - ck[i]).abs() <= 1e-5 * ks,
                        "layer {layer} pos {pos} raw k[{i}]: {} vs {}",
                        gr[i],
                        ck[i]
                    );
                }
            }
            let kv: &[(&str, &Vec<u16>, &Vec<f32>)] = if norm {
                &[("v", &gv, &cv)]
            } else {
                &[("k", &gk, &ck), ("v", &gv, &cv)]
            };
            for &(what, got, want) in kv {
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
    deterministic(CacheKv::F16);
}

/// `gpu_deterministic_graph_or_not` with the q8 cache (both scale groups).
#[test]
fn gpu_deterministic_graph_or_not_q8() {
    deterministic(CacheKv::Q8 { group: 0 });
    deterministic(CacheKv::Q8 { group: 32 });
}

fn deterministic(kv: CacheKv) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    // qwen3 too: graphs replay its norm prologue (the attention writes the
    // new K row itself, so a replay must leave the same bits).
    for m in [crate::testutil::model(), crate::testutil::qwen3_model()]
        .into_iter()
        .flatten()
    {
        deterministic_on(&g, &m, kv);
    }
}

fn deterministic_on(g: &Gpu, m: &Model, kv: CacheKv) {
    let km = Kernels::module(g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let mut net = CudaQwen2::new(g, &km, m, 512).unwrap();
    net.kv = kv;
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
/// divergence, summed), asserting that every divergence is a near-tie: a
/// margin below `tie` in r's logits or in o's.
fn continuations_with<R: Lm, O: Lm>(
    r: &R,
    o: &O,
    tok: &crate::tokenizer::Tokenizer,
    name: &str,
    tie: f32,
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
                let (a, b) = (tr[i] as usize, to[i] as usize);
                let margin = lr[i][a] - lr[i][b];
                eprintln!(
                    "{name}: {p:?} diverges at token {i}: {:?} vs {:?}, reference margin {margin:.3}",
                    tok.decode(&[tr[i]]),
                    tok.decode(&[to[i]])
                );
                if margin >= tie {
                    // Both sides' logits are rounded: the pair is a near-tie
                    // if either calls it one (the CPU alone can be off by
                    // more than the bound: llama 1B, "17 times 23" step 30,
                    // CPU 0.938, an f64 forward 0.259, the split attention
                    // 0.122 the other way).
                    let (t2, lo) = greedy(o, &ids, i + 1, true);
                    assert_eq!(t2[..], to[..=i], "{name}: forward vs forward_argmax");
                    let own = lo[i][b] - lo[i][a];
                    eprintln!("{name}: its own margin {own:.3}");
                    assert!(
                        own < tie,
                        "not a near-tie: margin {margin} (reference), {own} (its own)"
                    );
                }
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
/// On a GPU with tensor cores, through both attention paths: its own, then
/// the split attention every GPU without them (sm < 80: Pascal, Volta,
/// Turing) runs.
fn continuations_match(m: &Model, name: &str) {
    let tie = 0.5;
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let mut gpu = CudaQwen2::new(&g, &km, m, 256).unwrap();
    let rows = Qwen2::new_with(m, false);
    let cpu = Qwen2::new(m);
    let (cs, cm) = continuations_with(&cpu, &rows, &tok, &format!("{name} cpu rows"), tie);
    let paths: &[bool] = if km.mma() { &[true, false] } else { &[false] };
    for &tc in paths {
        gpu.set_tc_attn(tc); // greedy makes its caches after this
        let (gs, gm) = continuations_with(&cpu, &gpu, &tok, &format!("{name} gpu tc {tc}"), tie);
        assert!(
            gs + 2 >= cs,
            "tc {tc}: identical continuations: gpu {gs}, cpu rows {cs}"
        );
        assert!(
            2 * gm + 32 >= cm,
            "tc {tc}: matched tokens: gpu {gm}, cpu rows {cm}"
        );
    }
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

/// Qwen3-0.6B, mapped (as the 0.5B: the per-row CPU model reads the file
/// after the default one releases what it repacks).
fn qwen3_mapped() -> Option<Model> {
    model_at(crate::testutil::QWEN3_MODEL)
}

#[test]
fn gpu_logits_match_cpu_qwen3_0_6b() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(m) = qwen3_mapped() else {
        return;
    };
    logits_match(&m, "qwen3 0.6B");
}

#[test]
fn gpu_continuations_match_cpu_qwen3_0_6b() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(m) = qwen3_mapped() else {
        return;
    };
    continuations_match(&m, "qwen3 0.6B");
}

#[test]
fn gpu_prefill_batchings_bit_identical_qwen3() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = qwen3_mapped() else {
        return;
    };
    prefill_bits(&m, "qwen3 0.6B", CacheKv::F16);
}

#[test]
fn gpu_verify_matches_forward_argmax_qwen3() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = qwen3_mapped() else {
        return;
    };
    verify_bits(&m, "qwen3 0.6B", CacheKv::F16);
}

/// The GPU's qwen3 (decode with and without graphs, batched prefill; f16
/// and q8 caches) against model.rs's independent `qwen3_ref` on the Q8_0
/// fixture, within QWEN3_Q8_TOL (8-bit activations, as the CPU there;
/// measured ~0.2). model.rs's qwen3_matches_reference_forward checks that
/// every NormBug moves the reference by more than 5x that bound, so a norm
/// bug in the decode prologue or rope_kv_norm_b / kv_q8_b fails here even
/// if the CPU shared it.
#[test]
fn gpu_qwen3_matches_reference_forward() {
    use crate::model::tests::{NormBug, QWEN3_Q8_TOL, max_diff, qwen3_fixture, qwen3_ref};
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let ids = [3u32, 17, 42, 9];
    let m = qwen3_fixture(Q8_0);
    let want = qwen3_ref(&m, &ids, NormBug::None);
    for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
        let mut net = CudaQwen2::with_kv(&g, &km, &m, 16, kv, false).unwrap();
        for graphs in [true, false] {
            net.graphs = graphs;
            let mut c = net.new_cache(ids.len());
            let dec: Vec<Vec<f32>> = (0..ids.len())
                .map(|p| net.forward(ids[p], p, &mut c))
                .collect();
            let mut c = net.new_cache(ids.len());
            net.feed_batch(&ids[..3], 0, &mut c);
            let pre = net.forward(ids[3], 3, &mut c);
            let (d, dp) = (max_diff(&dec, &want), max_diff(&[pre], &want[3..]));
            eprintln!(
                "gpu {kv:?} graphs {graphs} vs reference: max |dlogit| {d:.4}, after prefill {dp:.4}"
            );
            assert!(
                d <= QWEN3_Q8_TOL && dp <= QWEN3_Q8_TOL,
                "gpu {kv:?}: {d} / {dp}"
            );
        }
    }
}

/// `--rope-scaling yarn --rope-scale F --yarn-orig-ctx N` on `m`.
fn with_yarn(m: &mut Model, factor: f64, orig: usize) {
    let args = crate::rope::RopeArgs {
        scaling: Some("yarn".into()),
        factor: Some(factor),
        orig_ctx: Some(orig),
        ..Default::default()
    };
    args.apply(m).unwrap();
    crate::model::check_supported(m).unwrap();
}

/// YaRN on the GPU: the qwen3 fixture (orig 16, so the ramp is crossed in
/// a few positions) against the reference's own YaRN on both caches, decode
/// bit-identical to batched prefill; then qwen2.5-0.5B with the 7B recipe
/// (factor 4, orig 32768): batchings, verify and GPU vs CPU as without.
#[test]
fn gpu_yarn_matches_reference_and_keeps_bits() {
    use crate::model::tests::{NormBug, QWEN3_Q8_TOL, max_diff, qwen3_fixture, qwen3_ref};
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let ids = [3u32, 17, 42, 9, 5, 11, 60, 2, 33, 21, 8, 40];
    let n = ids.len();
    let mut m = qwen3_fixture(Q8_0);
    with_yarn(&mut m, 4.0, 16);
    let want = qwen3_ref(&m, &ids, NormBug::None);
    // Far from the plain reference and from YaRN without the mscale, so a
    // kernel ignoring the ramp or `m` fails below.
    let plain = qwen3_ref(&qwen3_fixture(Q8_0), &ids, NormBug::None);
    let mut m1 = qwen3_fixture(Q8_0);
    let no_m = crate::rope::RopeArgs {
        scaling: Some("yarn".into()),
        factor: Some(4.0),
        orig_ctx: Some(16),
        mscale: Some(1.0),
        ..Default::default()
    };
    no_m.apply(&mut m1).unwrap();
    let no_m = qwen3_ref(&m1, &ids, NormBug::None);
    for (what, r) in [("plain", &plain), ("no mscale", &no_m)] {
        let d = max_diff(r, &want);
        eprintln!("gpu yarn test: {what} reference vs yarn's: {d:.3}");
        assert!(d > 4.0 * QWEN3_Q8_TOL, "{what}: only {d}");
    }
    for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
        let mut net = CudaQwen2::with_kv(&g, &km, &m, 16, kv, false).unwrap();
        assert_eq!(net.rope_m, crate::rope::mscale(&m));
        for graphs in [true, false] {
            net.graphs = graphs;
            let mut c = net.new_cache(n);
            let dec: Vec<Vec<f32>> = (0..n).map(|p| net.forward(ids[p], p, &mut c)).collect();
            let mut c = net.new_cache(n);
            net.feed_batch(&ids[..n - 1], 0, &mut c);
            let pre = net.forward(ids[n - 1], n - 1, &mut c);
            let d = max_diff(&dec, &want);
            eprintln!("gpu yarn {kv:?} graphs {graphs} vs reference: max |dlogit| {d:.4}");
            assert!(d <= QWEN3_Q8_TOL, "gpu yarn {kv:?}: {d}");
            assert_eq!(
                bits(&pre),
                bits(&dec[n - 1]),
                "gpu yarn {kv:?}: decode != prefill"
            );
        }
    }
    let Some(mut m) = model_at(crate::testutil::MODEL) else {
        return;
    };
    with_yarn(&mut m, 4.0, 32768);
    prefill_bits(&m, "qwen2.5 0.5B yarn", CacheKv::F16);
    prefill_bits(&m, "qwen2.5 0.5B yarn", CacheKv::Q8 { group: 0 });
    verify_bits(&m, "qwen2.5 0.5B yarn", CacheKv::F16);
    logits_match(&m, "qwen2.5 0.5B yarn");
}

/// Llama-3.2-1B-Instruct, mapped (as `qwen3_mapped`): head_dim 64, GQA 4,
/// tied Q6_K head, Llama 3 RoPE factors, q/k rows reordered at load.
fn llama_mapped() -> Option<Model> {
    crate::testutil::llama_model()?;
    model_at(crate::testutil::LLAMA_MODEL)
}

/// Llama-3.2-3B-Instruct Q4_K_M (head_dim 128, GQA 3: 24 query heads on 8),
/// and Llama-3.1-8B-Instruct (untied Q6_K head, GQA 4), if present.
const LLAMA_3B: &str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";
const LLAMA_8B: &str = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

#[test]
fn gpu_logits_match_cpu_llama_1b() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(m) = llama_mapped() else {
        return;
    };
    logits_match(&m, "llama 1B");
}

#[test]
fn gpu_continuations_match_cpu_llama_1b() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(m) = llama_mapped() else {
        return;
    };
    continuations_match(&m, "llama 1B");
}

#[test]
fn gpu_prefill_batchings_bit_identical_llama() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = llama_mapped() else {
        return;
    };
    prefill_bits(&m, "llama 1B", CacheKv::F16);
    verify_bits(&m, "llama 1B", CacheKv::F16);
}

/// llama on the q8 cache: batchings and verify keep the bits (both scale
/// groups), accuracy within qwen2's bounds.
#[test]
fn gpu_q8_llama() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = llama_mapped() else {
        return;
    };
    prefill_bits(&m, "llama 1B", CacheKv::Q8 { group: 64 });
    prefill_bits(&m, "llama 1B", CacheKv::Q8 { group: 32 });
    verify_bits(&m, "llama 1B", CacheKv::Q8 { group: 64 });
    q8_accuracy(&m, "llama 1B");
}

#[test]
fn gpu_bits_llama_3b_8b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    for (path, name) in [(LLAMA_3B, "llama 3B"), (LLAMA_8B, "llama 8B")] {
        let Some(m) = model_at(path) else {
            continue;
        };
        prefill_bits(&m, name, CacheKv::F16);
        verify_bits(&m, name, CacheKv::F16);
        prefill_bits(&m, name, CacheKv::Q8 { group: 128 });
    }
}

/// The GPU's llama (decode with and without graphs, batched prefill; f16
/// and q8 caches) against model.rs's independent `llama_ref` on the Q8_0
/// fixture's raw tensors, within LLAMA_Q8_TOL (model.rs's
/// llama_matches_reference_forward shows each RopeBug moves the reference
/// by > 10x that).
#[test]
fn gpu_llama_matches_reference_forward() {
    use crate::model::tests::{LLAMA_Q8_TOL, RopeBug, llama_fixture, llama_ref, max_diff};
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let ids = [3u32, 17, 42, 9, 5, 61, 33, 2, 28, 50, 11, 7];
    let n = ids.len();
    let (m, raw) = llama_fixture(Q8_0);
    let want = llama_ref(&m, &raw, &ids, RopeBug::None);
    for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
        let mut net = CudaQwen2::with_kv(&g, &km, &m, 16, kv, false).unwrap();
        for graphs in [true, false] {
            net.graphs = graphs;
            let mut c = net.new_cache(n);
            let dec: Vec<Vec<f32>> = (0..n).map(|p| net.forward(ids[p], p, &mut c)).collect();
            let mut c = net.new_cache(n);
            net.feed_batch(&ids[..n - 1], 0, &mut c);
            let pre = net.forward(ids[n - 1], n - 1, &mut c);
            let (d, dp) = (max_diff(&dec, &want), max_diff(&[pre], &want[n - 1..]));
            eprintln!(
                "llama gpu {kv:?} graphs {graphs} vs reference: {d:.4}, after prefill {dp:.4}"
            );
            assert!(
                d <= LLAMA_Q8_TOL && dp <= LLAMA_Q8_TOL,
                "gpu {kv:?}: {d} / {dp}"
            );
        }
    }
}

/// Qwen3-4B Q4_K_M (Q4_K/Q6_K, q_dim 4096 != dim 2560, GQA 4), if present.
const QWEN3_4B: &str = "models/Qwen3-4B-Q4_K_M.gguf";

#[test]
fn gpu_prefill_batchings_bit_identical_qwen3_4b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at(QWEN3_4B) else {
        return;
    };
    prefill_bits(&m, "qwen3 4B", CacheKv::F16);
}

#[test]
fn gpu_verify_matches_forward_argmax_qwen3_4b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at(QWEN3_4B) else {
        return;
    };
    verify_bits(&m, "qwen3 4B", CacheKv::F16);
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
    // AXON_PROF_CTX: the cache size (default 4200); AXON_PROF_POS: the
    // positions, comma-separated (default 8,1000,4100).
    let ctx: usize = std::env::var("AXON_PROF_CTX").map_or(4200, |v| v.parse().unwrap());
    let poss: Vec<usize> = std::env::var("AXON_PROF_POS").map_or(vec![8, 1000, 4100], |v| {
        v.split(',').map(|x| x.trim().parse().unwrap()).collect()
    });
    let mut net = CudaQwen2::new(&g, &km, &m, ctx).unwrap();
    // AXON_PROF_KV=q8: the q8 cache.
    if std::env::var("AXON_PROF_KV").is_ok_and(|v| v == "q8") {
        net.kv = CacheKv::Q8 { group: 0 };
    }
    let mut c = net.new_cache(ctx);
    for pos in poss {
        assert!(
            pos + 40 <= ctx,
            "AXON_PROF_POS {pos} past AXON_PROF_CTX {ctx}"
        );
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
                net.step(&mut c, Input::Token(100), pos, true).unwrap();
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
            net.step(&mut c, Input::Token(100), pos + i, true).unwrap();
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
///
/// `free_bytes` is the whole card's: other processes (the desktop, another
/// suite on a shared test pod) allocate and free between our reads, so one
/// before/after pair can't tell a leak of ours from their churn. A leak
/// shows on every attempt and churn doesn't, so the memory checks get a few
/// attempts and pass on the first clean one.
#[test]
fn gpu_too_big_is_an_error() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let free = || g.info().unwrap().free_bytes;
    let mut seen = Vec::new();
    for _ in 0..8 {
        let free0 = free();
        let e = CudaQwen2::new(&g, &km, &m, 50_000_000).err().unwrap();
        assert!(e.contains("GiB is free") && e.contains("--ctx"), "{e}");
        let held = {
            let net = CudaQwen2::new(&g, &km, &m, 1024).unwrap();
            let _c = net.new_cache(1024);
            let held = free();
            // Every cache is checked against what is free when it is made (new
            // budgets one): one too big is an Err before allocating, and a
            // second normal one still fits.
            let e = net.try_cache(50_000_000).err().unwrap();
            assert!(e.contains("GiB is free") && e.contains("--ctx"), "{e}");
            let _c2 = net.try_cache(1024).unwrap();
            held
        };
        // Allow some slack for small allocations by others.
        let free1 = free();
        if held < free0 && free1 + (64 << 20) >= free0 {
            return;
        }
        seen.push((free0, held, free1));
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    panic!("free bytes (before, while held, after) on every attempt: {seen:?}");
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
/// the batched prefill in the GPU's own batch size (two passes, the last
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

/// GPU vs CPU logits on 1600 tokens of real, varied text (natural_ids:
/// prose and code, nothing repeated): prefilled in 512-token passes (the
/// last one starting at 1024), then positions 1100..1600 one by one,
/// teacher-forced, graded like `logits_match` against the CPU's per-row
/// kernels. Errors that only show past 1024 positions land here.
fn natural_long(m: &Model, name: &str) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module(&g).unwrap();
    let gpu = CudaQwen2::new(&g, &km, m, 1600).unwrap();
    let rows = Qwen2::new_with(m, false);
    let cpu = Qwen2::new(m);
    let ids = natural_ids(m, 1600);
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    compare_forced(&cpu, &gpu, &ids, 1100, &mut ag);
    compare_forced(&cpu, &rows, &ids, 1100, &mut ac);
    report(&format!("{name} natural 1100..1600 gpu vs cpu"), &ag);
    report(&format!("{name} natural 1100..1600 cpu rows vs cpu"), &ac);
    // Measured, GPU / CPU per-row vs CPU (top-1, mean KL, max KL): 0.5B
    // 95.2 % / 96.2 %, 2.6e-3 / 2.5e-3, 0.021 / 0.012 (f32 attention
    // before: 96.4 %, 2.8e-3, 0.059); 1.5B 93.6 % / 93.8 %, 7.8e-3 /
    // 7.3e-3, 0.076 / 0.030 (before: 94.2 %, 7.4e-3, 0.034).
    assert_graded_floor(&ag, &ac, 90);
}

#[test]
fn gpu_logits_match_cpu_natural_long_0_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model_with(Some(false)) else {
        return;
    };
    natural_long(&m, "0.5B");
}

#[test]
fn gpu_logits_match_cpu_natural_long_1_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    natural_long(&m, "1.5B");
}

/// Programmatic dependent launch changes when kernels start, never what
/// they compute: tokens and logit bits equal with it on and off.
#[test]
fn gpu_pdl_on_off_identical() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    for m in [crate::testutil::model(), crate::testutil::qwen3_model()]
        .into_iter()
        .flatten()
    {
        pdl_on_off(&g, &m);
    }
}

fn pdl_on_off(g: &Gpu, m: &Model) {
    let km_on = Kernels::module(g).unwrap();
    let mut km_off = Kernels::module(g).unwrap();
    km_off.disable_pdl();
    if !km_on.pdl() {
        eprintln!("PDL unavailable here: comparing off with off");
    }
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let on = CudaQwen2::new(g, &km_on, m, 256).unwrap();
    let off = CudaQwen2::new(g, &km_off, m, 256).unwrap();
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

/// The decode step's rmsnorm_quant_v2 and gate_up4 against rmsnorm_quant
/// and gate_up: the same tokens and logit bits, each on and off, graphs
/// and not. gate_up4 where the rule takes it (the 0.5B's Q5_0 rows, the
/// 0.5B Q5_K_M's Q5_1 rows through the ext module, the 0.5B Q4_0's, the
/// 1.5B's Q4_K) and forced on the rest (the 1.5B Q5_K_M's Q5_K, qwen3's
/// Q8_0, Llama's and the 4B's wider Q4_K rows).
#[test]
fn gpu_decode_norm2_gate_up4_identical() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let models = [
        crate::testutil::model(),
        model_at("models/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf"),
        model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        model_at("models/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf"),
        model_at("models/Qwen2.5-0.5B-Instruct-Q4_0.gguf"),
        crate::testutil::qwen3_model(),
        crate::testutil::llama_model(),
        model_at("models/Qwen3-4B-Q4_K_M.gguf"),
    ];
    for m in models.into_iter().flatten() {
        let km = Kernels::module_for(&g, &m).unwrap();
        let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
        let mut net = CudaQwen2::new(&g, &km, &m, 256).unwrap();
        let lay = &net.layers[0];
        assert!(lay.gate.qt == lay.up.qt && net.ffn.is_multiple_of(2));
        let rule = lay.gate.rb < GATE_UP4_ROW;
        assert!(net.dim <= NORM2_MAX);
        let ids = tok.encode(PROMPTS[3]);
        net.norm2 = false;
        net.gate_up4_row = 0;
        let (t0, l0) = greedy(&net, &ids, 16, true);
        for (norm2, gu4, graphs) in [
            (true, false, true),
            (false, true, true),
            (true, true, true),
            (true, true, false),
        ] {
            net.norm2 = norm2;
            net.gate_up4_row = if gu4 { usize::MAX } else { 0 };
            net.graphs = graphs;
            let (t, l) = greedy(&net, &ids, 16, true);
            let (ta, _) = greedy(&net, &ids, 16, false);
            assert_eq!(t, t0, "norm2 {norm2} gate_up4 {gu4} graphs {graphs}");
            assert_eq!(ta, t0, "forward_argmax: norm2 {norm2} gate_up4 {gu4}");
            for i in 0..l.len() {
                assert_eq!(
                    bits(&l[i]),
                    bits(&l0[i]),
                    "norm2 {norm2} gate_up4 {gu4} step {i}"
                );
            }
        }
        // And as the rule picks (the default), with graphs.
        net.gate_up4_row = GATE_UP4_ROW;
        net.graphs = true;
        let (t, l) = greedy(&net, &ids, 16, true);
        assert_eq!(t, t0);
        assert!(l.iter().zip(&l0).all(|(a, b)| bits(a) == bits(b)));
        eprintln!(
            "dim {} ffn {} gate {} B/row (gate_up4 by rule: {rule}): rmsnorm_quant_v2 and \
             gate_up4 bit-identical over 16 steps",
            net.dim, net.ffn, net.layers[0].gate.rb
        );
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
    // The Q6_K prefill copy's kernels take the copy (`q6k8_copy`).
    let rb = match kernel.ends_with("q6k8") {
        true => (cols / 256 * Q6K8_BYTES) as u64,
        false => quant::row_bytes(qt, cols) as u64,
    };
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
    unsafe { launch_product(g, &f, kernel, qt, rows, nb, &common) };
    g.sync().unwrap();
    out.read().unwrap()
}

/// Q6_K matrix `w` (rows x cols) converted to its prefill copy (q6k_to_i8).
fn q6k8_copy<'g>(g: &'g Gpu, km: &Module, w: &DevBuf, rows: usize, cols: usize) -> DevBuf<'g> {
    let nblk = cols / 256;
    let out = g.alloc(rows * nblk * Q6K8_BYTES).unwrap();
    let f = km.function("q6k_to_i8").unwrap();
    // SAFETY: q6k_to_i8's signature; w holds rows Q6_K rows, out their copy.
    unsafe {
        g.launch(
            &f,
            (nblk as u32, rows as u32, 1),
            (64, 1, 1),
            0,
            &[
                w.arg(),
                Arg::U64(quant::row_bytes(Q6_K, cols) as u64),
                Arg::I32(nblk as i32),
                out.arg(),
            ],
        )
        .unwrap()
    };
    g.sync().unwrap();
    out
}

/// Launch product kernel `kernel` (gemv_ref, gemvn_*, gemvm*, gemm*) with
/// its grid and shared memory; `common` its arguments after the type.
///
/// # Safety
/// `common` matches prefill.cu's signature for the kernel and the buffers
/// it points to hold what the kernel reads and writes.
unsafe fn launch_product(
    g: &Gpu,
    f: &Func,
    kernel: &str,
    qt: u32,
    rows: usize,
    nb: usize,
    common: &[Arg],
) {
    let with_qt = || {
        let mut args = vec![Arg::I32(qt as i32)];
        args.extend_from_slice(common);
        args
    };
    // SAFETY: the caller's.
    unsafe {
        if kernel == "gemv_ref" {
            let args = with_qt();
            g.launch(
                f,
                (rows.div_ceil(128) as u32, nb as u32, 1),
                (128, 1, 1),
                0,
                &args,
            )
            .unwrap();
        } else if kernel.starts_with("gemvm") {
            g.launch(f, (rows.div_ceil(16) as u32, 1, 1), (128, 1, 1), 0, common)
                .unwrap();
        } else if kernel.starts_with("gemm2") {
            let (bn, st) = if kernel.starts_with("gemm2s") {
                (64, 3)
            } else {
                (128, 2)
            };
            let sm = gemm2_smem(qt, bn, st);
            f.set_max_shared(sm).unwrap();
            g.launch(
                f,
                (nb.div_ceil(bn) as u32, rows.div_ceil(G2_BM) as u32, 1),
                (256, 1, 1),
                sm as u32,
                common,
            )
            .unwrap();
        } else {
            let (r, t): (usize, usize) = if kernel == "gemvn_2x8" {
                (2, 8)
            } else {
                (2, 16)
            };
            let args = with_qt();
            g.launch(
                f,
                (rows.div_ceil(r) as u32, nb.div_ceil(t) as u32, 1),
                (GEMV_THREADS, 1, 1),
                0,
                &args,
            )
            .unwrap();
        }
    }
}

/// prefill.cu's name for a type.
fn type_name(qt: u32) -> &'static str {
    match qt {
        Q4_K => "q4_k",
        Q6_K => "q6_k",
        Q8_0 => "q8_0",
        Q5_K => "q5_k",
        Q4_0 => "q4_0",
        Q4_1 => "q4_1",
        Q5_1 => "q5_1",
        Q2_K => "q2_k",
        Q3_K => "q3_k",
        IQ4_NL => "iq4_nl",
        _ => "q5_0",
    }
}

/// A random matrix of type qt (rows x cols) with sane f16 scales in every
/// block: d (and Q4_K's dmin, drawn independently) of either sign and
/// magnitude 2^-11..2^-10, different per block, so a scale applied to the
/// wrong row, block or term changes the bits; products stay finite.
fn synth_matrix(rng: &mut Rng, qt: u32, rows: usize, cols: usize) -> Vec<u8> {
    let rb = quant::row_bytes(qt, cols);
    let mut w: Vec<u8> = (0..rb * rows).map(|_| rng.next() as u8).collect();
    let (bb, vb) = match qt {
        Q4_K => (144, 256),
        Q5_K => (176, 256),
        Q6_K => (210, 256),
        Q8_0 => (34, 32),
        Q4_0 => (18, 32),
        Q4_1 => (20, 32),
        Q5_1 => (24, 32),
        Q2_K => (84, 256),
        Q3_K => (110, 256),
        IQ4_NL => (18, 32),
        _ => (22, 32),
    };
    for b in 0..rows * cols / vb {
        let o = b * bb
            + match qt {
                Q6_K => 208,
                Q2_K => 80,
                Q3_K => 108,
                _ => 0,
            };
        let mut scale = || (0x1000u16 + (rng.next() & 0x7ff) as u16) | (rng.next() as u16 & 0x8000);
        let d = scale();
        w[o..o + 2].copy_from_slice(&d.to_le_bytes());
        if matches!(qt, Q4_K | Q5_K | Q4_1 | Q5_1 | Q2_K) {
            let dmin = scale();
            w[o + 2..o + 4].copy_from_slice(&dmin.to_le_bytes());
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
    let km = Kernels::module_ext(&g).unwrap();
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
        // groups, and other widths.
        "synth Q4_K 768",
        "synth Q4_K 1536",
        "synth Q6_K 1536",
        "synth Q6_K 2304",
        "synth Q8_0 3584",
        "synth Q5_0 1536",
        // The 7B's shapes: 14 blocks, 74.
        "synth Q4_K 3584",
        "synth Q4_K 18944",
        "synth Q6_K 3584",
        "synth Q6_K 18944",
        // The new types: the 0.5B's 896 (Q4_0) and K-quant widths.
        "synth Q4_0 896",
        "synth Q4_0 4864",
        "synth Q4_0 18944",
        "synth Q5_K 768",
        "synth Q5_K 1536",
        "synth Q5_K 3584",
        "synth Q5_K 18944",
        "synth Q4_1 896",
        "synth Q4_1 4864",
        "synth Q5_1 896",
        "synth Q5_1 4864",
        "synth Q2_K 1536",
        "synth Q2_K 3584",
        "synth Q2_K 18944",
        "synth Q3_K 768",
        "synth Q3_K 1536",
        "synth Q3_K 18944",
        "synth IQ4_NL 896",
        "synth IQ4_NL 4864",
    ] {
        let synth: Vec<u8>;
        let (qt, cols, rows, wd): (u32, usize, usize, &[u8]) = if name.starts_with("synth") {
            let f: Vec<&str> = name.split(' ').collect();
            let qt = match f[1] {
                "Q4_K" => Q4_K,
                "Q6_K" => Q6_K,
                "Q8_0" => Q8_0,
                "Q5_K" => Q5_K,
                "Q4_0" => Q4_0,
                "Q4_1" => Q4_1,
                "Q5_1" => Q5_1,
                "Q2_K" => Q2_K,
                "Q3_K" => Q3_K,
                "IQ4_NL" => IQ4_NL,
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
        for nb in [1usize, 6, 7, 8, 13, 64, 66, 77, 129] {
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
                kernels.extend(gemm2_names(qt));
                let tn = type_name(qt);
                // (Q4_1 / Q5_1 have no gemvm: small batches take gemvn.)
                if nb <= 8 && has_gemvm(qt) {
                    kernels.push(format!("gemvm8_{tn}"));
                }
                if nb <= 16 && has_gemvm(qt) {
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
    for qt in [Q4_K, Q6_K, Q5_0, Q8_0, Q5_K, Q4_0, Q4_1, Q5_1, Q2_K, Q3_K, IQ4_NL] {
        assert!(seen.contains(&qt), "type {qt} not covered");
    }
}

/// Every product kernel against gemv_ref, the canonical order written out
/// one thread per output: the decode GEMV (per token), gemvn, gemvm and
/// the GEMMs, bit for bit, every weight type, random widths up to the 7B's
/// 18944, odd row counts, batches around the tiles' sizes, all epilogues.
#[test]
fn gpu_products_match_reference_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module_ext(&g).unwrap();
    let mut rng = Rng(0x0dd5_eed5_2026_1014);
    let mut cases: Vec<(u32, usize, usize)> = Vec::new();
    for qt in [Q4_K, Q6_K, Q8_0, Q5_0, Q5_K, Q4_0, Q4_1, Q5_1, Q2_K, Q3_K, IQ4_NL] {
        let vb = if matches!(qt, Q4_K | Q5_K | Q6_K | Q2_K | Q3_K) {
            256
        } else {
            32
        };
        let mut widths = vec![256, 18944];
        if vb == 32 {
            widths.push(896);
        }
        for _ in 0..2 {
            widths.push(vb * (1 + (rng.next() % (18944 / vb) as u64) as usize));
        }
        for (i, &cols) in widths.iter().enumerate() {
            let rows = [333, 77, 129, 1001, 65][i % 5];
            cases.push((qt, cols, rows));
        }
    }
    for &(qt, cols, rows) in &cases {
        let wd = synth_matrix(&mut rng, qt, rows, cols);
        let w = g.upload(&wd).unwrap();
        let w8 = (qt == Q6_K && km.mma()).then(|| q6k8_copy(&g, &km, &w, rows, cols));
        let tn = type_name(qt);
        for nb in [1usize, 7, 16, 17, 100, 128, 129, 512] {
            let xs: Vec<Vec<f32>> = (0..nb)
                .map(|i| rng.vec(cols, [1.0f32, 1e-3, 40.0][i % 3]))
                .collect();
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
            let act = quantize_batch(&g, &km, &xs, cols);
            let want = gpu_batched(&g, &km, "gemv_ref", qt, &w, rows, cols, &act, nb, &epi);
            let check = |k: &str, z: usize, got: &[f32]| {
                if let Some(r) =
                    (0..rows).find(|&r| got[r].to_bits() != want[z * rows + r].to_bits())
                {
                    panic!(
                        "{tn} {cols}x{rows} {k} nb {nb} token {z} row {r}: {} vs ref {}",
                        got[r],
                        want[z * rows + r]
                    );
                }
            };
            // The decode GEMV, a few tokens.
            for z in [0, nb / 2, nb - 1] {
                for k in ["gemv", "gemv1"] {
                    check(k, z, &gpu_gemv_epi(&g, &km, k, qt, &wd, cols, &xs[z], &epi));
                }
            }
            let mut kernels: Vec<String> = vec!["gemvn_2x8".into(), "gemvn_2x16".into()];
            if km.mma() {
                // (Q4_1 / Q5_1 have no gemvm: small batches take gemvn.)
                if nb <= 8 && has_gemvm(qt) {
                    kernels.push(format!("gemvm8_{tn}"));
                }
                if nb <= 16 && has_gemvm(qt) {
                    kernels.push(format!("gemvm16_{tn}"));
                }
                kernels.extend(gemm2_names(qt));
            }
            for k in &kernels {
                let got = gpu_batched(&g, &km, k, qt, &w, rows, cols, &act, nb, &epi);
                for z in 0..nb {
                    check(k, z, &got[z * rows..(z + 1) * rows]);
                }
            }
            if let (Some(w8), true) = (&w8, nb > 16) {
                for k in ["gemm2_q6k8", "gemm2s_q6k8"] {
                    let got = gpu_batched(&g, &km, k, qt, w8, rows, cols, &act, nb, &epi);
                    for z in 0..nb {
                        check(k, z, &got[z * rows..(z + 1) * rows]);
                    }
                }
            }
        }
        eprintln!("{tn} {cols} x {rows}: every kernel == gemv_ref");
    }
}

/// The up product's swiglu epilogue (mode 2: out = silu(gate) * sum) in
/// every product kernel, and gemm2s_q4_k_gu (gate and up in one launch),
/// against gemv_ref: gate by gemv_ref (mode 0), then up by gemv_ref in
/// mode 2 on it.
#[test]
fn gpu_swiglu_epilogue_and_gate_up_match_reference() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(0x5eed_9a7e_0042);
    for (cols, rows) in [(256usize, 333usize), (1536, 129), (3584, 1000)] {
        let (qt, tn) = (Q4_K, "q4_k");
        let gd = synth_matrix(&mut rng, qt, rows, cols);
        let ud = synth_matrix(&mut rng, qt, rows, cols);
        let (wg, wu) = (g.upload(&gd).unwrap(), g.upload(&ud).unwrap());
        for nb in [1usize, 9, 16, 17, 64, 100, 129] {
            let xs: Vec<Vec<f32>> = (0..nb)
                .map(|i| rng.vec(cols, [1.0f32, 1e-2, 4.0][i % 3]))
                .collect();
            let act = quantize_batch(&g, &km, &xs, cols);
            let st = act_stride_t(cols) as u64;
            let rb = quant::row_bytes(qt, cols) as u64;
            let run = |k: &str, w: &DevBuf, bias: Arg, mode: i32| -> DevBuf {
                let out = g.alloc(rows * nb * 4).unwrap();
                out.zero().unwrap();
                let f = km.function(k).unwrap();
                let common = [
                    w.arg(),
                    Arg::U64(rb),
                    Arg::I32(rows as i32),
                    Arg::I32(cols as i32),
                    act.arg(),
                    Arg::U64(st),
                    Arg::I32(nb as i32),
                    bias,
                    Arg::I32(mode),
                    out.arg(),
                    Arg::U64(rows as u64),
                ];
                // SAFETY: prefill.cu's signatures; act holds nb tokens,
                // out (and the mode-2 gate) rows * nb floats.
                unsafe { launch_product(&g, &f, k, qt, rows, nb, &common) };
                g.sync().unwrap();
                out
            };
            let gate = run("gemv_ref", &wg, Arg::Ptr(0), 0);
            let want: Vec<f32> = run("gemv_ref", &wu, gate.arg(), 2).read().unwrap();
            let mut kernels: Vec<String> = vec!["gemvn_2x8".into(), "gemvn_2x16".into()];
            if km.mma() {
                kernels.push(format!("gemvm{}_{tn}", if nb <= 8 { 8 } else { 16 }));
                kernels.extend(gemm2_names(qt));
            }
            for k in &kernels {
                if k.starts_with("gemvm") && nb > 16 {
                    continue;
                }
                let got: Vec<f32> = run(k, &wu, gate.arg(), 2).read().unwrap();
                assert_eq!(bits(&got), bits(&want), "mode 2 {k} {cols}x{rows} nb {nb}");
            }
            if km.mma() {
                let out = g.alloc(rows * nb * 4).unwrap();
                let f = km.function("gemm2s_q4_k_gu").unwrap();
                f.set_max_shared(gemm2_smem(Q4_K, 64, 3)).unwrap();
                // SAFETY: gemm2s_q4_k_gu's signature; as above.
                unsafe {
                    g.launch(
                        &f,
                        (nb.div_ceil(64) as u32, rows.div_ceil(32) as u32, 1),
                        (256, 1, 1),
                        gemm2_smem(Q4_K, 64, 3) as u32,
                        &[
                            wg.arg(),
                            wu.arg(),
                            Arg::U64(rb),
                            Arg::I32(rows as i32),
                            Arg::I32(cols as i32),
                            act.arg(),
                            Arg::U64(st),
                            Arg::I32(nb as i32),
                            out.arg(),
                            Arg::U64(rows as u64),
                        ],
                    )
                    .unwrap()
                };
                g.sync().unwrap();
                let got: Vec<f32> = out.read().unwrap();
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "gemm2s_q4_k_gu {cols}x{rows} nb {nb}"
                );
            }
        }
        eprintln!("Q4_K {cols} x {rows}: mode 2 and the fused gate+up == gemv_ref");
    }
}

/// The tensor-core GEMMs of a type (prefill.cu gemm2_*).
/// Whether type qt has gemvm kernels.
fn has_gemvm(qt: u32) -> bool {
    !matches!(qt, Q4_1 | Q5_1 | Q2_K | Q3_K | IQ4_NL)
}

fn gemm2_names(qt: u32) -> Vec<String> {
    let tn = type_name(qt);
    vec![format!("gemm2_{tn}"), format!("gemm2s_{tn}")]
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
    let km = Kernels::module_ext(&g).unwrap();
    let info = g.info().unwrap();
    eprintln!(
        "{} sm_{}{} {} SMs",
        info.name, info.cc.0, info.cc.1, info.sms
    );
    let kernels_all = [
        "gemm2_q4_k",
        "gemm2s_q4_k",
        "gemm2_q6_k",
        "gemm2s_q6_k",
        "gemm2_q8_0",
        "gemm2_q5_0",
        "gemm2s_q8_0",
        "gemm2s_q5_0",
        "gemm2_q6k8",
        "gemm2s_q6k8",
        "gemm2s_q4_k_gu",
        "gemvn_2x8",
        "gemvm8_q4_k",
        "gemvm8_q6_k",
        "gemvm16_q4_k",
        "gemvm16_q6_k",
    ];
    for k in kernels_all {
        if let Ok(f) = km.function(k) {
            eprintln!("{k}: regs, local bytes {:?}", f.regs_local().unwrap());
        }
    }
    // The int8 mma.sync ceiling: independent m16n8k32s, 4 blocks of 8 warps per SM.
    if km.mma() {
        let f = km.function("mma_peak").unwrap();
        let o = g.alloc(4 * 4096).unwrap();
        let (blocks, iters) = (4 * info.sms as usize, 4096usize);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let e0 = g.record().unwrap();
            // SAFETY: mma_peak's signature; o has a slot per block.
            unsafe {
                g.launch(
                    &f,
                    (blocks as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &[Arg::I32(iters as i32), o.arg()],
                )
                .unwrap();
            }
            let e1 = g.record().unwrap();
            best = best.min(e0.ms_to(&e1).unwrap() as f64);
        }
        let macs = (blocks * 8 * iters * 8 * 16 * 8 * 32) as f64;
        eprintln!(
            "mma_peak: {:.1} TMAC/s int8 (m16n8k32, {:.1} ms)",
            macs / (best * 1e-3) / 1e12,
            best
        );
    }
    // The f16 x f16 -> f32 ceiling (the tensor-core attention's mma).
    if km.mma() {
        let f = km.function("mma_peak_h").unwrap();
        let o = g.alloc(4 * 4096).unwrap();
        let (blocks, iters) = (4 * info.sms as usize, 4096usize);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let e0 = g.record().unwrap();
            // SAFETY: mma_peak_h's signature; o has a slot per block.
            unsafe {
                g.launch(
                    &f,
                    (blocks as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &[Arg::I32(iters as i32), o.arg()],
                )
                .unwrap();
            }
            let e1 = g.record().unwrap();
            best = best.min(e0.ms_to(&e1).unwrap() as f64);
        }
        let macs = (blocks * 8 * iters * 8 * 16 * 8 * 16) as f64;
        eprintln!(
            "mma_peak_h: {:.1} TMAC/s f16->f32 (m16n8k16, {:.1} ms)",
            macs / (best * 1e-3) / 1e12,
            best
        );
    }
    let mut rng = Rng(7);
    let only = std::env::var("AXON_BENCH_ONLY").ok();
    for (what, qt, cols, rows) in [
        ("q/wo  Q4_K", Q4_K, 3584, 3584),
        ("kv    Q4_K", Q4_K, 3584, 512),
        ("gate  Q4_K", Q4_K, 3584, 18944),
        ("down  Q4_K", Q4_K, 18944, 3584),
        ("down  Q6_K", Q6_K, 18944, 3584),
        ("0.5B ffn Q5_0", Q5_0, 896, 4864),
        ("0.5B q Q8_0", Q8_0, 896, 4864),
        ("0.5B down Q4_K", Q4_K, 4864, 896),
        ("1.5B gate Q4_K", Q4_K, 1536, 8960),
        ("tbo Q4_K 4096x14336", Q4_K, 14336, 4096),
        ("tbo Q6_K 4096x14336", Q6_K, 14336, 4096),
        ("gate  Q5_K", Q5_K, 3584, 18944),
        ("down  Q5_K", Q5_K, 18944, 3584),
        ("1.5B gate Q5_K", Q5_K, 1536, 8960),
        ("1.5B gate Q4_0", Q4_0, 1536, 8960),
        ("gate  Q4_0", Q4_0, 3584, 18944),
        ("0.5B ffn Q5_1", Q5_1, 896, 4864),
        ("0.5B ffn Q4_0", Q4_0, 896, 4864),
    ] {
        if only.as_deref().is_some_and(|o| !what.contains(o)) {
            continue;
        }
        let rb = quant::row_bytes(qt, cols);
        let w = synth_matrix(&mut rng, qt, rows, cols);
        let wb = g.upload(&w).unwrap();
        let nbs: Vec<usize> = std::env::var("AXON_BENCH_NB")
            .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or(vec![8, 16, 128, 512]);
        for nb in nbs {
            let xs: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 1.0)).collect();
            let act = quantize_batch(&g, &km, &xs, cols);
            let out = g.alloc(rows * nb * 4).unwrap();
            let tn = type_name(qt);
            let mut kernels: Vec<String> = Vec::new();
            if nb <= 16 {
                kernels.push(if nb <= 8 { "gemvn_2x8" } else { "gemvn_2x16" }.into());
                if has_gemvm(qt) {
                    kernels.push(format!("gemvm{}_{tn}", if nb <= 8 { 8 } else { 16 }));
                }
            }
            kernels.extend(gemm2_names(qt));
            // Rounds over all kernels (interleaved: clocks drift as the GPU
            // heats up), best time of each.
            let mut best = vec![f64::INFINITY; kernels.len()];
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
                Arg::I32(std::env::var("AXON_BENCH_MODE").map_or(0, |v| v.parse().unwrap())),
                out.arg(),
                Arg::U64(rows as u64),
            ];
            for _round in 0..8 {
                for (ki, k) in kernels.iter().enumerate() {
                    let f = km.function(k).unwrap();
                    let e0 = g.record().unwrap();
                    // SAFETY: as in gpu_batched.
                    unsafe { launch_product(&g, &f, k, qt, rows, nb, &args) };
                    let e1 = g.record().unwrap();
                    best[ki] = best[ki].min(e0.ms_to(&e1).unwrap() as f64);
                }
            }
            for (ki, k) in kernels.iter().enumerate() {
                let best = best[ki];
                let macs = (rows * cols * nb) as f64;
                eprintln!(
                    "{what:<14} nb {nb:>3} {k:<12} {:>8.1} us {:>6.1} TMAC/s {:>5.0} GB/s weights",
                    best * 1e3,
                    macs / (best * 1e-3) / 1e12,
                    (rb * rows) as f64 / (best * 1e-3) / 1e9
                );
            }
            // AXON_BENCH_LOOP=n: each kernel n times back to back (sustained
            // rate under the power cap), average.
            if let Ok(n) = std::env::var("AXON_BENCH_LOOP") {
                let n: usize = n.parse().unwrap();
                for k in kernels.iter() {
                    let f = km.function(k).unwrap();
                    let e0 = g.record().unwrap();
                    for _ in 0..n {
                        // SAFETY: as above.
                        unsafe { launch_product(&g, &f, k, qt, rows, nb, &args) };
                    }
                    let e1 = g.record().unwrap();
                    let ms = e0.ms_to(&e1).unwrap() as f64 / n as f64;
                    let macs = (rows * cols * nb) as f64;
                    eprintln!(
                        "{what:<14} nb {nb:>3} {k:<12} {:>8.1} us {:>6.1} TMAC/s SUSTAINED x{n}",
                        ms * 1e3,
                        macs / (ms * 1e-3) / 1e12
                    );
                }
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

/// KV cache bytes, positions 0..n of every layer (q8: the whole cache,
/// int8 rows and scales; positions past n are zero in both when compared).
fn kv_bytes(net: &CudaQwen2, c: &CudaCache, n: usize) -> Vec<u16> {
    let all: Vec<u16> = c.kv.read().unwrap();
    if c.mode != CacheKv::F16 {
        return all;
    }
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
fn prefill_bits(m: &Model, name: &str, kv: CacheKv) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module_for(&g, m).unwrap();
    let ids = long_ids(m, 1200);
    let mut net = CudaQwen2::new(&g, &km, m, ids.len() + 1).unwrap();
    net.kv = kv;
    let max = net.prefill_batch();
    assert!(ids.len() > max, "covers chunking past the max batch");
    // Both attention paths: the tensor cores' (attn_fa_b / attn_blk_b,
    // decode attn_blk), then the split kernels of GPUs without them
    // (attn_split_b per query, decode attn_split), forced here. Each with
    // the GPU's own split count, then 8: attn_blk grids of 8 warps per kv
    // head, each taking several blocks; or splits of more than 32
    // positions past 256.
    // (Head size 256, gemma3: the split kernels only.)
    let paths: &[bool] = if km.mma() && net.head_dim != 256 {
        &[true, false]
    } else {
        &[false]
    };
    let own = net.max_split;
    for (&tc, max_split) in paths.iter().flat_map(|t| [(t, own), (t, 8)]) {
        net.set_tc_attn(tc);
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
                "{name} {sizes:?} tc {tc} split {max_split}: KV differs at {bad:?}"
            );
            assert_eq!(
                bits(&l),
                bits(&l0),
                "{name} {sizes:?} tc {tc} split {max_split}: logits"
            );
        }
        // prefill_logits (the last token in the final pass, Head::Last),
        // from 0 and after a fed prefix; and swiglu_quant_b in place of the
        // up product's swiglu epilogue.
        // Last: without the Q6_K prefill copy (made for these small models
        // by the VRAM rule, so the passes above used it if they have Q6_K).
        let copy = net.q6k8_vram() > 0;
        // (first position, swiglu in the up product's epilogue, gate and up
        // fused in one launch where both are Q4_K).
        for (from, epi, fused) in [
            (0, true, true),
            (300, true, true),
            (1000, true, false),
            (0, false, false),
            (5, true, true),
            (7, true, true),
        ] {
            net.swiglu_epi = epi;
            net.gate_up_fused = fused;
            if from == 5 {
                net.drop_q6k8().unwrap();
            }
            // 7: fused asked for but no gemm2s_q4_k_gu (as without int8
            // tensor cores): the separate gate and up products must run.
            let gu = if from == 7 {
                net.bf.gemm_gu.take()
            } else {
                None
            };
            let mut c = net.new_cache(ids.len() + 1);
            if from > 0 {
                net.feed_batch(&ids[..from], 0, &mut c);
            }
            let l = net.prefill_logits(&ids[from..], from, &mut c);
            let kv = kv_bytes(&net, &c, ids.len());
            assert!(
                kv == kv0,
                "{name} prefill_logits from {from} epi {epi} tc {tc}: KV"
            );
            assert_eq!(
                bits(&l),
                bits(&l0),
                "{name} prefill_logits from {from} epi {epi} fused {fused} tc {tc} split \
                 {max_split}: logits"
            );
            if gu.is_some() {
                net.bf.gemm_gu = gu;
            }
        }
        net.gate_up_fused = true;
        net.swiglu_epi = false;
        if copy {
            eprintln!("{name}: with and without the Q6_K prefill copy");
            net.make_q6k8(net.cache_bytes(ids.len() + 1)).unwrap();
        }
    }
    eprintln!(
        "{name} {kv:?}: prefill bit-identical over 7 batchings + 4 prefill_logits of {} \
         tokens, {} attention paths x 2 split counts",
        ids.len(),
        paths.len()
    );
}

/// The Q6_K prefill copy only from VRAM nothing else was planned for: the
/// cache, HEADROOM and Q6K8_MARGIN stay free (forced: the cache).
#[test]
fn q6k8_never_costs_context() {
    let (gib, mib) = (1u64 << 30, 1u64 << 20);
    let (cache, copy) = (2 * gib, gib);
    let need = cache + HEADROOM + Q6K8_MARGIN + copy;
    assert!(q6k8_fits(need, cache, copy, false));
    assert!(!q6k8_fits(need - 1, cache, copy, false));
    assert!(!q6k8_fits(cache + copy + 300 * mib, cache, copy, false));
    assert!(q6k8_fits(cache + copy, cache, copy, true));
    assert!(!q6k8_fits(cache + copy - 1, cache, copy, true));
    assert!(!q6k8_fits(100 * gib, cache, 0, false));
}

/// A cuda_fa_bench median (us; lcpre NOTES): (cc, SMs, n_heads, n_kv, hd,
/// queries, start, [Full, Small, Kv]).
type FaRow = (u32, usize, usize, usize, usize, usize, usize, [f64; 3]);
#[rustfmt::skip]
const FA_MEASURED: &[FaRow] = &[
    // RTX 5070 Ti Laptop (sm_120, 46 SMs; bella), c5-bella.log.
    (120, 46, 28, 4, 128, 1121, 16384, [7156.2, 6652.4, 6118.7]),
    (120, 46, 28, 4, 128, 1121, 32768, [14094.0, 13082.4, 12168.3]),
    (120, 46, 28, 4, 128, 1042, 32768, [11705.7, 11812.5, 11088.1]),
    (120, 46, 28, 4, 128, 1016, 32768, [11906.5, 11919.9, 11044.9]),
    (120, 46, 32, 8, 128, 1121, 16384, [8422.7, 7703.7, 7214.0]),
    (120, 46, 32, 8, 128, 1121, 32768, [16845.2, 15394.6, 14380.3]),
    (120, 46, 32, 8, 128, 1042, 32768, [14577.2, 14623.8, 13475.8]),
    (120, 46, 32, 8, 128, 1016, 32768, [14175.7, 14423.7, 13246.4]),
    (120, 46, 14, 2, 64, 1121, 16384, [1473.4, 2041.5, 1671.1]),
    (120, 46, 14, 2, 64, 1121, 32768, [2928.2, 3967.4, 3289.9]),
    (120, 46, 14, 2, 64, 1016, 32768, [3094.5, 3898.1, 2900.3]),
    // RTX 5090 (sm_120, 170 SMs), c5-pod3-5090.log.
    (120, 170, 28, 4, 128, 1121, 16384, [1836.3, 1469.4, 1374.1]),
    (120, 170, 28, 4, 128, 1121, 32768, [3652.1, 2910.3, 2709.2]),
    (120, 170, 28, 4, 128, 1016, 32768, [3650.0, 2904.4, 2711.3]),
    (120, 170, 32, 8, 128, 1121, 16384, [1860.9, 1895.1, 1775.6]),
    (120, 170, 32, 8, 128, 1121, 32768, [3693.4, 3904.5, 3512.7]),
    (120, 170, 32, 8, 128, 1016, 32768, [3660.5, 3805.4, 3479.5]),
    (120, 170, 14, 2, 64, 1121, 16384, [572.2, 597.6, 520.7]),
    (120, 170, 14, 2, 64, 1121, 32768, [1112.9, 1159.3, 1012.9]),
    (120, 170, 14, 2, 64, 1016, 32768, [1110.9, 1155.4, 1033.3]),
    // RTX 4090 (sm_89, 128 SMs), c5-pod2-4090.log.
    (89, 128, 28, 4, 128, 1121, 16384, [2134.4, 2285.9, 2146.2]),
    (89, 128, 28, 4, 128, 1121, 32768, [3803.4, 4085.8, 3832.0]),
    (89, 128, 28, 4, 128, 1016, 32768, [3794.6, 4053.7, 3814.4]),
    (89, 128, 32, 8, 128, 1121, 16384, [2887.1, 2552.2, 2418.3]),
    (89, 128, 32, 8, 128, 1121, 32768, [5680.5, 5024.1, 4759.6]),
    (89, 128, 32, 8, 128, 1016, 32768, [3813.4, 4090.2, 3867.9]),
    (89, 128, 14, 2, 64, 1121, 16384, [472.9, 807.5, 457.4]),
    (89, 128, 14, 2, 64, 1121, 32768, [917.1, 1583.8, 881.7]),
    (89, 128, 14, 2, 64, 1016, 32768, [912.7, 1570.1, 884.4]),
    // RTX 4050 Laptop (sm_89, 20 SMs; avante), c5-avante.log.
    (89, 20, 28, 4, 128, 1121, 16384, [15306.4, 16479.6, 13934.3]),
    (89, 20, 28, 4, 128, 1121, 32768, [29937.3, 32213.3, 28750.2]),
    (89, 20, 28, 4, 128, 1016, 32768, [28110.7, 29784.3, 26120.3]),
    (89, 20, 32, 8, 128, 1121, 16384, [18688.7, 20330.9, 17777.7]),
    (89, 20, 32, 8, 128, 1121, 32768, [35845.5, 39916.9, 36830.2]),
    (89, 20, 32, 8, 128, 1016, 32768, [31105.3, 33705.6, 32274.5]),
    (89, 20, 14, 2, 64, 1121, 16384, [3786.6, 4706.8, 3533.8]),
    (89, 20, 14, 2, 64, 1121, 32768, [7521.5, 9423.8, 6975.1]),
    (89, 20, 14, 2, 64, 1016, 32768, [6896.2, 8654.1, 6585.9]),
];

/// A regression guard on the rows `fa_eff` was fitted to (FA_MEASURED is
/// the same data, not held out): `fa_pick` never takes a variant measured
/// slower than Full (beyond 1 % noise), and is within 5 % of the fastest
/// where that beats Full by more than 10 %. A new GPU needs its own rows.
#[test]
fn fa_pick_never_slower_than_full_measured() {
    use super::batch::{Fa, FaGeom, fa_pick};
    for &(cc, sms, nh, n_kv, hd, nb, start, t) in FA_MEASURED {
        let g = FaGeom {
            cc,
            sms,
            hd,
            kg: 0,
            n_kv,
            group: nh / n_kv,
        };
        let v = fa_pick(g, nb);
        let what = format!("sm_{cc} {sms} SMs {nh}/{n_kv}x{hd} nb {nb} at {start}: {v:?} {t:?}");
        assert!(t[v as usize] <= 1.01 * t[Fa::Full as usize], "{what}");
        let best = t.iter().cloned().fold(f64::INFINITY, f64::min);
        if best < 0.9 * t[Fa::Full as usize] {
            assert!(t[v as usize] <= 1.05 * best, "{what}");
        }
    }
}

/// `fa_pick`: Small for short batches as before; the q8 cache and the
/// GPUs not measured keep Full.
#[test]
fn fa_pick_rules() {
    use super::batch::{Fa, FaGeom, fa_pick};
    let g = |cc, sms, kg| FaGeom {
        cc,
        sms,
        hd: 128,
        kg,
        n_kv: 4,
        group: 7,
    };
    assert_eq!(fa_pick(g(120, 46, 0), 100), Fa::Small);
    assert_eq!(fa_pick(g(120, 170, 32), 1121), Fa::Full);
    assert_eq!(fa_pick(g(86, 84, 0), 1121), Fa::Full);
    assert_eq!(fa_pick(g(90, 132, 0), 1121), Fa::Full);
    assert_eq!(fa_pick(g(120, 170, 0), 1121), Fa::Kv);
}

/// `fa_passes`: whole prompts in at most a few more passes than `passes`,
/// none longer than max, no more GEMM token tiles; short prompts and
/// unmeasured GPUs as `passes`; fewer attention waves where it pays.
#[test]
fn fa_passes_shapes() {
    use super::batch::{FaGeom, fa_passes, fa_time, passes};
    let geoms = [
        (120, 46, 28, 4, 128),
        (120, 46, 32, 8, 128),
        (120, 46, 14, 2, 64),
        (120, 170, 28, 4, 128),
        (89, 128, 32, 8, 128),
        (89, 20, 28, 4, 128),
        (86, 84, 28, 4, 128),
    ];
    for (cc, sms, nh, n_kv, hd) in geoms {
        let g = FaGeom {
            cc,
            sms,
            hd,
            kg: 0,
            n_kv,
            group: nh / n_kv,
        };
        for (n, start) in [
            (2100, 0),
            (8192, 0),
            (16384, 0),
            (32512, 0),
            (5000, 20000),
            (8000, 100),
            (30, 9000),
        ] {
            let (v, base) = (fa_passes(n, 1152, start, g), passes(n, 1152));
            let tiles = |p: &[std::ops::Range<usize>]| -> usize {
                p.iter().map(|r| r.len().div_ceil(128)).sum()
            };
            assert_eq!(v.first().unwrap().start, 0);
            assert_eq!(v.last().unwrap().end, n);
            assert!(v.windows(2).all(|w| w[0].end == w[1].start));
            assert!(v.iter().all(|r| !r.is_empty() && r.len() <= 1152));
            assert!(v.len() >= base.len() && v.len() <= base.len() + (base.len() / 8).max(1));
            assert!(tiles(&v) <= tiles(&base));
            if start + n < 8192 || cc == 86 || hd == 64 {
                assert_eq!(v, base, "sm_{cc} n {n} at {start}");
            }
        }
    }
    // A regression guard on fitted data (the 4090 rows fa_eff came from), not
    // a validation: RTX 4090, Qwen3-8B at 32512: 29 passes of 1121 (Kv: 576
    // blocks, 2.25 waves) -> 32 of 1016 (512 blocks, 2 waves): MEASURED 4760
    // us per 1121 vs 3868 per 1016 at 32k, -12 % per token.
    let g = FaGeom {
        cc: 89,
        sms: 128,
        hd: 128,
        kg: 0,
        n_kv: 8,
        group: 4,
    };
    let v = fa_passes(32512, 1152, 0, g);
    assert_eq!((v.len(), v[0].len()), (32, 1016));
    let w = |nb: usize| fa_time(g, nb) / nb as f64;
    assert!(w(1016) < 0.97 * w(1122));
}

#[test]
fn batch_passes_equal_split() {
    use super::batch::passes;
    assert_eq!(passes(2100, 1152), vec![0..1050, 1050..2100]);
    assert_eq!(passes(559, 1152), vec![0..559]);
    assert_eq!(passes(1152, 1152), vec![0..1152]);
    assert_eq!(passes(1153, 1152), vec![0..577, 577..1153]);
    assert_eq!(passes(7, 3), vec![0..3, 3..5, 5..7]);
    for n in 1..300 {
        for max in [1, 2, 7, 16, 64, 100] {
            let p = passes(n, max);
            assert_eq!(p.len(), n.div_ceil(max));
            assert_eq!(p[0].start, 0);
            assert_eq!(p.last().unwrap().end, n);
            assert!(p.windows(2).all(|w| w[0].end == w[1].start));
            let (lo, hi) = (
                p.iter().map(|r| r.len()).min().unwrap(),
                p.iter().map(|r| r.len()).max().unwrap(),
            );
            assert!(hi <= max && hi - lo <= 1, "{n} {max}: {p:?}");
        }
    }
}

/// `prefill_logits` (the last prompt token inside the final batched pass)
/// == `feed_batch` + `forward`, bit for bit, on the big models present
/// (AXON_BIG_MODELS=1: qwen2.5 7B and qwen3 8B, prompts of 2, 17, 559 and
/// 2100 tokens; else 3B if present).
#[test]
fn gpu_prefill_logits_match_forward_big() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let big = std::env::var("AXON_BIG_MODELS").is_ok_and(|v| v == "1");
    let files: &[&str] = if big {
        &[
            "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "models/Qwen3-8B-Q4_K_M.gguf",
        ]
    } else {
        &["models/qwen2.5-3b-instruct-q4_k_m.gguf"]
    };
    let km = Kernels::module(&g).unwrap();
    for f in files {
        let Some(m) = model_at(f) else {
            continue;
        };
        // 1152: one Head::Last pass of exactly the batch max; 1153: two.
        let lens: &[usize] = if big {
            &[2, 17, 559, 1152, 1153, 2100]
        } else {
            &[2, 17, 600, 1152, 1153]
        };
        let all = long_ids(&m, *lens.iter().max().unwrap());
        let net = CudaQwen2::new(&g, &km, &m, all.len() + 1).unwrap();
        for &n in lens {
            let ids = &all[..n];
            let mut a = net.new_cache(n + 1);
            let pb = net.prefill_batch();
            for (i, ch) in ids[..n - 1].chunks(pb).enumerate() {
                net.feed_batch(ch, i * pb, &mut a);
            }
            let want = net.forward(ids[n - 1], n - 1, &mut a);
            let mut b = net.new_cache(n + 1);
            let got = net.prefill_logits(ids, 0, &mut b);
            assert_eq!(bits(&got), bits(&want), "{f} {n} tokens: logits");
            assert!(
                kv_bytes(&net, &a, n) == kv_bytes(&net, &b, n),
                "{f} {n} tokens: KV"
            );
            if n == 559 || n == 600 {
                // The decode reference: every token through the one-token
                // step (no batched GEMM, fused gate+up or Q6_K copy).
                let mut d = net.new_cache(n + 1);
                for (p, &t) in ids[..n - 1].iter().enumerate() {
                    net.feed(&mut d, t, p).unwrap();
                }
                let dec = net.forward(ids[n - 1], n - 1, &mut d);
                assert_eq!(bits(&got), bits(&dec), "{f} {n} tokens: logits vs decode");
                assert!(
                    kv_bytes(&net, &d, n) == kv_bytes(&net, &b, n),
                    "{f} {n} tokens: KV vs decode"
                );
                eprintln!(
                    "{f}: == per-token decode at {n} tokens (Q6_K copy {} B, fused gate+up {})",
                    net.q6k8_vram(),
                    net.gate_up_fused
                );
            }
            eprintln!("{f}: prefill_logits == feed_batch + forward at {n} tokens");
        }
    }
}

#[test]
fn gpu_prefill_batchings_bit_identical_0_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    prefill_bits(&m, "0.5B", CacheKv::F16);
}

#[test]
fn gpu_prefill_batchings_bit_identical_1_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    prefill_bits(&m, "1.5B", CacheKv::F16);
}

/// verify_argmax's picks equal forward_argmax per token for drafts that
/// are right, wrong and random, small (gemvn) and large (GEMM, chunked past
/// the max batch) passes.
fn verify_bits(m: &Model, name: &str, kv: CacheKv) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module_for(&g, m).unwrap();
    let prompt = long_ids(m, 90);
    let mut net = CudaQwen2::new(&g, &km, m, 1400).unwrap();
    net.kv = kv;
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
        "{name} {kv:?}: {} verify passes == per-token picks (logit bits up to 64)",
        cases.len()
    );
}

/// Decode attention at long positions (many ATT_BLK blocks per kv head
/// and many partials per head in attn_cmb): a 32768 cache filled by
/// feed_batch with random tokens, then at each position p of `at` the
/// decode step's logits (`forward`, graph and not) against the batched
/// pass's (128 tokens ending at p: attn_fa_b, the oracle), bit for bit.
/// On a GPU without room for 32768 positions, 16384 and the positions
/// below it; without room for that either, skipped.
fn decode_attention_long(m: &Model, name: &str, kv: CacheKv, at: &[usize]) {
    const W: usize = 128;
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module(&g).unwrap();
    let made = [32768, 16384].into_iter().find_map(|ctx| {
        let mut net = CudaQwen2::new(&g, &km, m, ctx).ok()?;
        net.kv = kv;
        let c = net.try_cache(ctx).ok()?;
        Some((net, c, ctx))
    });
    let Some((mut net, mut c, ctx)) = made else {
        eprintln!("skipping {name} {kv:?}: no room for a 16384-position cache");
        return;
    };
    let at: Vec<usize> = at.iter().copied().filter(|&p| p < ctx).collect();
    if ctx < 32768 {
        eprintln!("{name} {kv:?}: a {ctx}-position cache (no room for 32768): pos {at:?}");
    }
    let last = *at.iter().max().unwrap();
    let mut rng = Rng(0x5eed_1a7e);
    let ids: Vec<u32> = (0..=last).map(|_| (rng.next() % 20000) as u32).collect();
    let pb = net.prefill_batch();
    for (i, ch) in ids[..last].chunks(pb).enumerate() {
        net.feed_batch(ch, i * pb, &mut c);
    }
    for &p in &at {
        let lo = p + 1 - W;
        let want = net.prefill_logits(&ids[lo..=p], lo, &mut c);
        for graphs in [true, false] {
            net.graphs = graphs;
            let got = net.forward(ids[p], p, &mut c);
            assert_eq!(
                bits(&got),
                bits(&want),
                "{name} {kv:?}: decode logits at pos {p} (graphs {graphs})"
            );
        }
        // attn_blk's grid: any number of warps per kv head, the same bits.
        net.graphs = false;
        for gy in [1, 7, 10_000] {
            net.att_gy = gy;
            let got = net.forward(ids[p], p, &mut c);
            assert_eq!(
                bits(&got),
                bits(&want),
                "{name} {kv:?}: pos {p}, grid y {gy}"
            );
        }
        net.att_gy = 0;
        net.graphs = true;
        eprintln!("{name} {kv:?}: decode == batched (attn_fa_b) logits at pos {p}");
    }
}

/// The decode step's combine depends on the position (attn_cmb past 8192,
/// attn_cmb256 before), its graphs on the cache: a 32768 cache at short
/// positions, and caches of 8192 and 8193 positions at 8191, give the logit
/// bits of a small cache; 8193's step at 8192 (attn_cmb) matches the batched
/// pass.
#[test]
fn gpu_decode_cache_size_same_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    // The big cache: 32768 positions, or 16384 on a small GPU.
    let Some((net, big)) = [32768, 16384].into_iter().find_map(|ctx| {
        let net = CudaQwen2::new(&g, &km, &m, ctx).ok()?;
        net.try_cache(ctx).ok()?;
        Some((net, ctx))
    }) else {
        eprintln!("skipping: no room for a 16384-position cache");
        return;
    };
    let mut rng = Rng(0x0c7c_5123);
    let ids: Vec<u32> = (0..8193).map(|_| (rng.next() % 20000) as u32).collect();
    // Steps at `at` in a cache of each size (feed_batch up to each).
    let run = |ctx: usize, at: &[usize]| -> Vec<Vec<u32>> {
        let mut c = net.new_cache(ctx);
        let mut pos = 0;
        let mut out = Vec::new();
        for &p in at {
            if p > pos {
                net.feed_batch(&ids[pos..p], pos, &mut c);
            }
            out.push(bits(&net.forward(ids[p], p, &mut c)));
            pos = p + 1;
        }
        out
    };
    let short = [8, 100, 4000];
    assert_eq!(
        run(big, &short),
        run(4001, &short),
        "{big} vs 4001 cache, pos {short:?}"
    );
    let a = run(8192, &[8191]);
    let b = run(8193, &[8191, 8192]);
    assert_eq!(a[0], b[0], "8192 vs 8193 cache at pos 8191");
    let mut c = net.new_cache(8193);
    net.feed_batch(&ids[..8192 - 127], 0, &mut c);
    let want = net.prefill_logits(&ids[8192 - 127..=8192], 8192 - 127, &mut c);
    assert_eq!(b[1], bits(&want), "8193 cache at pos 8192 vs batched");
    eprintln!("decode bits independent of the cache size ({big}/4001/8192/8193)");
}

#[test]
fn gpu_decode_attention_long() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    if let Some(m) = crate::testutil::model() {
        for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
            // 8191: the last step by attn_cmb256, 8192 the first by attn_cmb.
            decode_attention_long(&m, "0.5B", kv, &[8191, 8192, 16383, 32511]);
        }
        decode_attention_long(&m, "0.5B", CacheKv::Q8 { group: 32 }, &[32511]);
    }
    // qwen3: the q/k norms in the decode attention (kpos, 128-wide heads).
    if let Some(m) = qwen3_mapped() {
        for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
            decode_attention_long(&m, "qwen3 0.6B", kv, &[8191]);
        }
    }
}

#[test]
fn gpu_verify_matches_forward_argmax_0_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    verify_bits(&m, "0.5B", CacheKv::F16);
}

#[test]
fn gpu_verify_matches_forward_argmax_1_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    verify_bits(&m, "1.5B", CacheKv::F16);
}

/// Q2_K / Q3_K files (bartowski's, whichever are present): decode ==
/// every prefill batching == prefill from a fed prefix (cache reuse) ==
/// verify, bit for bit, on both caches; the 7B Q2_K's Q2_K rows (and its
/// Q3_K ffn_down) at its shapes.
#[test]
fn gpu_prefill_batchings_bit_identical_q2k_q3k() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    for (f, name, q8) in [
        ("Qwen2.5-0.5B-Instruct-Q2_K.gguf", "0.5B Q2_K", true),
        ("Qwen2.5-0.5B-Instruct-Q3_K_M.gguf", "0.5B Q3_K_M", false),
        ("Qwen2.5-1.5B-Instruct-Q3_K_M.gguf", "1.5B Q3_K_M", true),
        ("Llama-3.2-1B-Instruct-Q3_K_L.gguf", "llama 1B Q3_K_L", false),
        ("Qwen2.5-7B-Instruct-Q2_K.gguf", "7B Q2_K", false),
    ] {
        let Some(m) = model_at(&format!("models/{f}")) else {
            continue;
        };
        prefill_bits(&m, name, CacheKv::F16);
        verify_bits(&m, name, CacheKv::F16);
        if q8 {
            prefill_bits(&m, name, CacheKv::Q8 { group: 64 });
            verify_bits(&m, name, CacheKv::Q8 { group: 64 });
        }
    }
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
    let net = CudaQwen2::new(&g, &km, &m, start + nb + 8).unwrap();
    let mut c = net.new_cache(start + nb + 8);
    let ids = long_ids(&m, start + nb);
    if start > 0 {
        net.feed_batch(&ids[..start], 0, &mut c);
    }
    // AXON_PROF_LOOP=n: n back-to-back passes (sustained power/clocks: sample
    // nvidia-smi meanwhile), per-pass wall ms printed.
    if let Ok(n) = std::env::var("AXON_PROF_LOOP") {
        let n: usize = n.parse().unwrap();
        let mut ws = Vec::new();
        for _ in 0..n {
            let t = std::time::Instant::now();
            net.pass(
                &mut c,
                batch::BatchIn::Tokens(&ids[start..]),
                start,
                false,
                &mut |_| Ok(()),
            )
            .unwrap();
            g.sync().unwrap();
            ws.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut s = ws.clone();
        s.sort_by(f64::total_cmp);
        eprintln!(
            "loop {n} passes nb {nb} at {start}: median {:.2} ms = {:.0} tok/s; first {:.2} last {:.2}",
            s[n / 2],
            nb as f64 / s[n / 2] * 1e3,
            ws[0],
            ws[n - 1]
        );
        return;
    }
    for round in 0..3 {
        let mut evs = vec![("start", g.record().unwrap())];
        net.pass(
            &mut c,
            batch::BatchIn::Tokens(&ids[start..]),
            start,
            nb <= 16,
            &mut |name| {
                evs.push((name, g.record()?));
                Ok(())
            },
        )
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
            "{rel} nb {nb} at {start}: {total:.2} ms = {:.0} tok/s ({} segments)",
            nb as f64 / total * 1e3,
            evs.len() - 1
        );
        // MACs per token of each GEMM class (names after the split: q, k,
        // qkv = v, wo, gate, gate_up = up, down).
        let l0 = &net.layers[0];
        let nl = net.layers.len() as f64;
        let macs = |mt: &Mat| (mt.rows * mt.cols) as f64 * nb as f64 * nl;
        for (n, ms, k) in agg {
            let m = match n {
                "q" => macs(&l0.q),
                "k" => macs(&l0.k),
                "v4" | "v6" => macs(&l0.v) * k as f64 / nl,
                "wo" => macs(&l0.o),
                "gate" => macs(&l0.gate),
                "gate_up" => macs(&l0.up),
                "down4" | "down6" => macs(&l0.down) * k as f64 / nl,
                _ => 0.0,
            };
            let shape = match n {
                "q" => format!("{}x{} {}", l0.q.rows, l0.q.cols, type_name(l0.q.qt)),
                "k" => format!("{}x{} {}", l0.k.rows, l0.k.cols, type_name(l0.k.qt)),
                "v4" => format!("v {}x{} q4_k", l0.v.rows, l0.v.cols),
                "wo" => format!("{}x{} {}", l0.o.rows, l0.o.cols, type_name(l0.o.qt)),
                "gate" => format!(
                    "{}x{} {}",
                    l0.gate.rows,
                    l0.gate.cols,
                    type_name(l0.gate.qt)
                ),
                "gate_up" => format!("up {}x{} {}", l0.up.rows, l0.up.cols, type_name(l0.up.qt)),
                "down4" => format!("{}x{} q4_k", l0.down.rows, l0.down.cols),
                "down6" => format!("{}x{} q6_k", l0.down.rows, l0.down.cols),
                "v6" => format!("v {}x{} q6_k", l0.v.rows, l0.v.cols),
                _ => String::new(),
            };
            eprintln!(
                "  {n:<8} {ms:>8.3} ms {:>5.1}% ({k} launches) {:>6.1} TMAC/s {shape}",
                100.0 * ms / total,
                if m > 0.0 { m / (ms * 1e-3) / 1e12 } else { 0.0 }
            );
        }
        // Wall clock of the same pass without the event callbacks (the
        // launch gaps the events hide), and of the one-token forward the
        // `gen` prompt timer also counts.
        let mut walls = Vec::new();
        let mut gpus = Vec::new();
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let e0 = g.record().unwrap();
            net.pass(
                &mut c,
                batch::BatchIn::Tokens(&ids[start..]),
                start,
                nb <= 16,
                &mut |_| Ok(()),
            )
            .unwrap();
            let e1 = g.record().unwrap();
            g.sync().unwrap();
            walls.push(t.elapsed().as_secs_f64() * 1e3);
            gpus.push(e0.ms_to(&e1).unwrap() as f64);
        }
        eprintln!(
            "  pass wall {:?} ms, gpu events {:?} ms",
            walls
                .iter()
                .map(|x| (x * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
            gpus.iter()
                .map(|x| (x * 100.0).round() / 100.0)
                .collect::<Vec<_>>()
        );
        let mut fw = Vec::new();
        for i in 0..3 {
            let t = std::time::Instant::now();
            let _ = net.forward_argmax(ids[0], start + nb - 1 + i, &mut c);
            fw.push(t.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!("  one-token forward (decode step incl. lm head) {fw:?} ms");
    }
}

/// The products' hardware f16 -> f32 conversion of block scales is exact
/// for how they use it (`h2f_mul_check`): no half, NaNs included, multiplies
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

/// The load-bearing assumption of the tensor-core attention (decode.cu
/// "Tensor-core attention"): an f16 mma.sync row result depends only on its
/// own A row, the B column and its accumulator - not on its m16 tile-mates,
/// the warp or block size, or how the fragments were loaded (global,
/// ldmatrix, ldmatrix.trans). Plus: products with A = 0 contribute nothing
/// whatever B holds there (prefill's V rows past a row's np are real, decode's
/// zero), up to the sign of an exactly zero result.
#[test]
fn gpu_mma_f16_is_deterministic() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    if !km.mma() {
        return;
    }
    let f = km.function("mma_f16_check").unwrap();
    let mut rng = Rng(4242);
    for &(k, n) in &[(128usize, 64usize), (64, 64), (32, 48)] {
        // Wide dynamic range (Qwen2's big key-bias channels), exact zeros.
        let val = |r: &mut Rng| {
            let x = r.f();
            let e = (r.next() % 12) as i32 - 6;
            if r.next().is_multiple_of(9) {
                0.0
            } else {
                x * 2f32.powi(e)
            }
        };
        let a: Vec<f32> = (0..64 * k).map(|_| val(&mut rng)).collect();
        let b: Vec<f32> = (0..n * k).map(|_| val(&mut rng)).collect();
        let h = |v: &[f32]| {
            v.iter()
                .map(|&x| quant::f32_to_f16(x))
                .collect::<Vec<u16>>()
        };
        let run = |a: &[f32], b: &[f32], perm: &[i32], am: i32, bm: i32, thr: u32| -> Vec<u32> {
            let bt: Vec<f32> = (0..k * n).map(|i| b[(i % n) * k + i / n]).collect();
            let (da, db, dbt) = (
                g.upload(&h(a)).unwrap(),
                g.upload(&h(b)).unwrap(),
                g.upload(&h(&bt)).unwrap(),
            );
            let dp = g.upload(perm).unwrap();
            let d = g.alloc(64 * n * 4).unwrap();
            // SAFETY: mma_f16_check's signature; buffers of the sizes it reads.
            unsafe {
                g.launch(
                    &f,
                    (1, 1, 1),
                    (thr, 1, 1),
                    0,
                    &[
                        da.arg(),
                        db.arg(),
                        dbt.arg(),
                        Arg::I32(k as i32),
                        Arg::I32(n as i32),
                        dp.arg(),
                        Arg::I32(am),
                        Arg::I32(bm),
                        d.arg(),
                    ],
                )
                .unwrap();
            }
            d.read::<f32>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        let id: Vec<i32> = (0..64).collect();
        let rev: Vec<i32> = (0..64).rev().collect();
        let mut shuf = id.clone();
        for i in (1..64).rev() {
            shuf.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let want = run(&a, &b, &id, 0, 0, 32);
        // Sanity: close to the f64 product of the f16 values.
        for r in 0..64 {
            for c in 0..n {
                let t = |i: usize| {
                    quant::f16_to_f32(quant::f32_to_f16(a[r * k + i])) as f64
                        * quant::f16_to_f32(quant::f32_to_f16(b[c * k + i])) as f64
                };
                let e: f64 = (0..k).map(t).sum();
                let mag: f64 = (0..k).map(|i| t(i).abs()).sum();
                let got = f32::from_bits(want[r * n + c]) as f64;
                assert!((got - e).abs() <= 1e-5 * (1.0 + mag), "{got} vs {e}");
            }
        }
        for (perm, am, bm, thr) in [
            (&id, 1, 1, 128),
            (&rev, 0, 2, 64),
            (&shuf, 1, 0, 96),
            (&shuf, 0, 1, 32),
            (&rev, 1, 2, 128),
        ] {
            assert!(
                run(&a, &b, perm, am, bm, thr) == want,
                "k {k} n {n}: perm, amode {am}, bmode {bm}, {thr} threads differ"
            );
        }
        // Zeros in A from column kz on: B's columns there don't matter.
        for kz in [1, 17, 37, k - 1] {
            let az: Vec<f32> = (0..64 * k)
                .map(|i| if i % k >= kz { 0.0 } else { a[i] })
                .collect();
            let bz: Vec<f32> = (0..n * k)
                .map(|i| if i % k >= kz { 0.0 } else { b[i] })
                .collect();
            let x = run(&az, &b, &shuf, 1, 1, 128);
            let y = run(&az, &bz, &id, 0, 0, 32);
            for (i, (p, q)) in x.iter().zip(&y).enumerate() {
                assert!(
                    p == q || (p | q) == 0x8000_0000,
                    "k {k} n {n} kz {kz} [{i}]: {p:#x} vs {q:#x}"
                );
            }
        }
    }
}

/// Batched attention speed, 7B shape (28 heads, 4 kv heads, 128 wide) by
/// default, AXON_ATT_NZ (64) queries at AXON_PROF_START (1536): the flash
/// kernel (attn_fa_b) and its quantize_b, the per-query decode kernels
/// (attn_blk_b, attn_cmb_b), and one decode step's attn_blk + attn_cmb at
/// the last query's position. Interleaved rounds, best of each.
#[test]
#[ignore]
fn cuda_attn_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    // AXON_ATT_SHAPE: n_heads,n_kv,hd (default the 7B's).
    let shape: Vec<usize> = std::env::var("AXON_ATT_SHAPE")
        .unwrap_or("28,4,128".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let (nh, nkv, hd) = (shape[0], shape[1], shape[2]);
    let nz: usize = std::env::var("AXON_ATT_NZ").map_or(64, |v| v.parse().unwrap());
    let start: usize = std::env::var("AXON_PROF_START").map_or(1536, |v| v.parse().unwrap());
    let ctx = start + nz;
    let nblk_max = ctx.div_ceil(64);
    let max_split = (8 * g.info().unwrap().sms as usize / nkv).clamp(1, MAX_SPLIT);
    let gy = max_split.min(nblk_max) as u32;
    let mut rng = Rng(3);
    let kvh: Vec<u16> = (0..ctx * nkv * hd)
        .map(|_| quant::f32_to_f16(rng.f() * 0.5))
        .collect();
    let kc = g.upload(&kvh).unwrap();
    let vc = g.upload(&kvh).unwrap();
    let dim = nh * hd;
    let q = g.upload(&rng.vec(nz * dim, 1.0)).unwrap();
    let po = g.alloc(nz.min(64) * nh * nblk_max * hd * 4).unwrap();
    let pm = g.alloc(nz.min(64) * nh * nblk_max * 2 * 4).unwrap();
    let ast = act_stride_t(dim);
    let act = g.alloc(nz * ast).unwrap();
    let params = g.upload(&[0u32, (ctx - 1) as u32, 0, 0]).unwrap();
    let ycmb = g.alloc(dim * 4).unwrap();
    let cnt = g.alloc(dim.div_ceil(256) * 4).unwrap();
    cnt.zero().unwrap();
    let names = [
        format!("attn_fa_b_{hd}"),
        "quantize_b".to_string(),
        format!("attn_blk_b_{hd}"),
        "attn_cmb_b".to_string(),
        format!("attn_blk_{hd}"),
        "attn_cmb".to_string(),
        format!("attn_fa_bs_{hd}"),
    ];
    let fs: Vec<Func> = names.iter().map(|n| km.function(n).unwrap()).collect();
    fs[0]
        .set_max_shared(batch::fa_shape(hd, batch::Fa::Full, 0).2)
        .unwrap();
    fs[6]
        .set_max_shared(batch::fa_shape(hd, batch::Fa::Small, 0).2)
        .unwrap();
    for (n, f) in names.iter().zip(&fs) {
        eprintln!("{n}: regs, local {:?}", f.regs_local().unwrap());
    }
    for q in ["q8", "q8g32"] {
        for v in ["attn_fa_b", "attn_fa_bs"] {
            let n = format!("{v}_{q}_{hd}");
            let f = km.function(&n).unwrap();
            eprintln!("{n}: regs, local {:?}", f.regs_local().unwrap());
        }
    }
    // The per-query kernels on at most 64 queries (SMALL sweeps).
    let nq = nz.min(64);
    let pre = Arg::F32(att_pre(hd));
    let mut best = [f64::INFINITY; 7];
    for _ in 0..6 {
        for (k, f) in fs.iter().enumerate() {
            let e0 = g.record().unwrap();
            // SAFETY: decode.cu / prefill.cu's signatures; buffers sized for
            // nz queries over ctx positions (part_* for nq).
            unsafe {
                let fv = if k == 6 {
                    batch::Fa::Small
                } else {
                    batch::Fa::Full
                };
                let (fa_rows, fa_thr, fa_smem) = batch::fa_shape(hd, fv, 0);
                match k {
                    0 | 6 => g.launch(
                        f,
                        (nkv as u32, (nz * nh / nkv).div_ceil(fa_rows) as u32, 1),
                        (fa_thr as u32, 1, 1),
                        fa_smem as u32,
                        &[
                            q.arg(),
                            kc.arg(),
                            vc.arg(),
                            Arg::I32(start as i32),
                            Arg::I32(nz as i32),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            pre,
                            q.arg(),
                            Arg::Ptr(0), // no page table
                        ],
                    ),
                    1 => g.launch(
                        f,
                        (dim.div_ceil(256) as u32, nz as u32, 1),
                        (32, 1, 1),
                        0,
                        &[
                            q.arg(),
                            Arg::I32(dim as i32),
                            Arg::I32(0),
                            act.arg(),
                            Arg::U64(ast as u64),
                        ],
                    ),
                    2 => g.launch(
                        f,
                        (nkv as u32, gy, nq as u32),
                        (32, 1, 1),
                        0,
                        &[
                            q.arg(),
                            kc.arg(),
                            vc.arg(),
                            Arg::I32((ctx - nq) as i32),
                            Arg::I32(0),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            pre,
                            Arg::I32(nblk_max as i32),
                            po.arg(),
                            pm.arg(),
                            Arg::Ptr(0),
                            Arg::Ptr(0), // no page table
                        ],
                    ),
                    3 => g.launch(
                        f,
                        (dim.div_ceil(256) as u32, nq as u32, 1),
                        (256, 1, 1),
                        0,
                        &[
                            po.arg(),
                            pm.arg(),
                            Arg::I32((ctx - nq) as i32),
                            Arg::I32(0),
                            Arg::I32(dim as i32),
                            Arg::I32(hd as i32),
                            Arg::I32(nblk_max as i32),
                            Arg::I32(0),
                            act.arg(),
                            Arg::U64(ast as u64),
                            Arg::Ptr(0),
                        ],
                    ),
                    4 => g.launch(
                        f,
                        (nkv as u32, gy, 1),
                        (32 * att_nw(hd, false), 1, 1),
                        0,
                        &[
                            q.arg(),
                            kc.arg(),
                            vc.arg(),
                            params.arg(),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            pre,
                            Arg::I32(nblk_max as i32),
                            po.arg(),
                            pm.arg(),
                            Arg::Ptr(0),
                            Arg::Ptr(0),
                            Arg::Ptr(0),
                            Arg::F32(0.0),
                            Arg::Ptr(0),
                            Arg::Ptr(0),
                            Arg::Ptr(0), // no page table
                        ],
                    ),
                    // attn_cmb: 64 outputs per block, y and the counters.
                    _ => g.launch(
                        f,
                        ((dim / 64) as u32, 1, 1),
                        (64, 1, 1),
                        0,
                        &[
                            po.arg(),
                            pm.arg(),
                            params.arg(),
                            Arg::I32(dim as i32),
                            Arg::I32(hd as i32),
                            Arg::I32(nblk_max as i32),
                            Arg::I32(0),
                            act.arg(),
                            ycmb.arg(),
                            cnt.arg(),
                        ],
                    ),
                }
                .unwrap();
            }
            let e1 = g.record().unwrap();
            best[k] = best[k].min(e0.ms_to(&e1).unwrap() as f64);
        }
    }
    let flops = |n: usize| -> f64 {
        (0..n).map(|z| (ctx - n + z + 1) as f64).sum::<f64>() * (nh * hd * 4) as f64
    };
    for (k, (n, b)) in names.iter().zip(best).enumerate() {
        let (qn, tf) = match k {
            0 | 6 => (nz, flops(nz) / (b * 1e-3) / 1e12),
            2 => (nq, flops(nq) / (b * 1e-3) / 1e12),
            4 => (1, flops(1) / (b * 1e-3) / 1e12),
            _ => (
                if k == 1 {
                    nz
                } else if k == 3 {
                    nq
                } else {
                    1
                },
                0.0,
            ),
        };
        eprintln!(
            "{nh}/{nkv}x{hd} {n:<16} {qn:>4} queries up to {ctx}: {:>8.1} us {tf:>6.2} TFLOP/s",
            b * 1e3
        );
    }
}

/// The flash attention kernels side by side, sustained (AXON_FA_LOOP
/// launches between two events), interleaved over AXON_FA_ROUNDS rounds:
/// AXON_FA_KERNELS (attn_fa_b,attn_fa_bs,attn_fa_k: batch::Fa; `_{hd}`
/// appended) for
/// every shape in AXON_FA_SHAPES (n_heads,n_kv,hd;... default 7B, 0.5B,
/// Qwen3-8B), AXON_FA_NZ (1121: pp32512's passes; a list) queries at each of AXON_FA_STARTS
/// (16384,32768). y goes to its own buffer and must match the first
/// kernel's bit for bit.
#[test]
#[ignore]
fn cuda_fa_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or(d.into());
    let nums = |s: &str| -> Vec<usize> { s.split(',').map(|v| v.parse().unwrap()).collect() };
    let shapes = env("AXON_FA_SHAPES", "28,4,128;14,2,64;32,8,128");
    let kernels = env("AXON_FA_KERNELS", "attn_fa_b,attn_fa_bs,attn_fa_k");
    let nzs = nums(&env("AXON_FA_NZ", "1121"));
    let starts = nums(&env("AXON_FA_STARTS", "16384,32768"));
    let rounds: usize = env("AXON_FA_ROUNDS", "5").parse().unwrap();
    let lp: usize = env("AXON_FA_LOOP", "3").parse().unwrap();
    for shape in shapes.split(';') {
        let s = nums(shape);
        let (nh, nkv, hd) = (s[0], s[1], s[2]);
        let names: Vec<String> = kernels.split(',').map(|k| format!("{k}_{hd}")).collect();
        let shp = |n: &str| match n.trim_end_matches(&format!("_{hd}")) {
            "attn_fa_bs" => batch::fa_shape(hd, batch::Fa::Small, 0),
            "attn_fa_k" => batch::fa_shape(hd, batch::Fa::Kv, 0),
            _ => batch::fa_shape(hd, batch::Fa::Full, 0),
        };
        let fs: Vec<Func> = names.iter().map(|n| km.function(n).unwrap()).collect();
        for (n, f) in names.iter().zip(&fs) {
            f.set_max_shared(shp(n).2).unwrap();
            eprintln!("{n}: regs, local {:?}", f.regs_local().unwrap());
        }
        for (&start, &nz) in starts.iter().flat_map(|s| nzs.iter().map(move |n| (s, n))) {
            let ctx = start + nz;
            let mut rng = Rng(5);
            let kh: Vec<u16> = (0..ctx * nkv * hd)
                .map(|_| quant::f32_to_f16(rng.f() * 2.0))
                .collect();
            let vh: Vec<u16> = (0..ctx * nkv * hd)
                .map(|_| quant::f32_to_f16(rng.f()))
                .collect();
            let (kc, vc) = (g.upload(&kh).unwrap(), g.upload(&vh).unwrap());
            let dim = nh * hd;
            let q = g.upload(&rng.vec(nz * dim, 1.0)).unwrap();
            let y = g.alloc(nz * dim * 4).unwrap();
            let pre = Arg::F32(att_pre(hd));
            let flops: f64 =
                (0..nz).map(|z| (start + z + 1) as f64).sum::<f64>() * (nh * hd * 4) as f64;
            let mut ms = vec![Vec::new(); fs.len()];
            let mut y0: Option<Vec<u32>> = None;
            for r in 0..rounds {
                for (k, f) in fs.iter().enumerate() {
                    let (rows, thr, smem) = shp(&names[k]);
                    let e0 = g.record().unwrap();
                    for _ in 0..lp {
                        // SAFETY: prefill.cu's attn_fa signature; q / y nz
                        // rows of dim floats, the cache ctx rows.
                        unsafe {
                            g.launch(
                                f,
                                (nkv as u32, (nz * nh / nkv).div_ceil(rows) as u32, 1),
                                (thr as u32, 1, 1),
                                smem as u32,
                                &[
                                    q.arg(),
                                    kc.arg(),
                                    vc.arg(),
                                    Arg::I32(start as i32),
                                    Arg::I32(nz as i32),
                                    Arg::I32(nh as i32),
                                    Arg::I32(nkv as i32),
                                    pre,
                                    y.arg(),
                                    Arg::Ptr(0), // no page table
                                ],
                            )
                            .unwrap();
                        }
                    }
                    let e1 = g.record().unwrap();
                    ms[k].push(e0.ms_to(&e1).unwrap() as f64 / lp as f64);
                    if r == 0 {
                        let v: Vec<u32> = y.read().unwrap();
                        match &y0 {
                            None => y0 = Some(v),
                            Some(y0) => {
                                let bad = v.iter().zip(y0).filter(|(a, b)| a != b).count();
                                assert_eq!(bad, 0, "{}: {bad} y values differ", names[k]);
                            }
                        }
                    }
                }
            }
            for (n, mut m) in names.iter().zip(ms) {
                m.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = m[m.len() / 2];
                eprintln!(
                    "FA {nh}/{nkv}x{hd} {n:<16} nz {nz} start {start}: med {:>9.1} us (min {:.1} max {:.1}) {:>6.2} TFLOP/s",
                    med * 1e3,
                    m[0] * 1e3,
                    m[m.len() - 1] * 1e3,
                    flops / (med * 1e-3) / 1e12
                );
            }
        }
    }
}

/// The batched tensor-core attention gives each query the decode kernels'
/// bits (decode.cu "Tensor-core attention"): attn_fa_b + quantize_b (y up
/// to the sign of a zero, act exactly) and attn_blk_b + attn_cmb_b (block
/// partials and act exactly) against attn_blk + attn_cmb run per query, for
/// every layout here (GQA groups 6, 7, 8; heads of 64 and 128), batches of
/// 1, 16, 17, 1024 queries at positions from 0 to past 1024 and 2048, np = 0,
/// 1, 63 mod 64 and blocks of rows straddling block boundaries, attn_blk_b
/// at query offsets z0 > 0, decode grids smaller than the block count (a
/// warp taking several blocks). The cache past the last query's position
/// is NaN (f16 0xffff): the kernels must never read it.
#[test]
fn gpu_batched_attention_matches_decode_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    if !km.mma() {
        return;
    }
    let (fc, fcb, fq) = (
        km.function("attn_cmb").unwrap(),
        km.function("attn_cmb_b").unwrap(),
        km.function("quantize_b").unwrap(),
    );
    let mut rng = Rng(9191);
    // (n_heads, n_kv, hd): 0.5B, 1.5B, 3B (group 8), 7B; the f16 cache,
    // the q8 cache with a scale per head row (kg = hd) or per 32 values.
    for (nh, nkv, hd, kg) in [
        (14, 2, 64, 0),
        (12, 2, 128, 0),
        (16, 2, 128, 0),
        (28, 4, 128, 0),
        (14, 2, 64, 64),
        (28, 4, 128, 32),
        (16, 2, 128, 128),
    ] {
        let (dim, kvd, group) = (nh * hd, nkv * hd, nh / nkv);
        let q8n = match kg {
            0 => String::new(),
            _ if kg == hd => "q8_".into(),
            _ => "q8g32_".into(),
        };
        let fd = km.function(&format!("attn_blk_{q8n}{hd}")).unwrap();
        let fb = km.function(&format!("attn_blk_b_{q8n}{hd}")).unwrap();
        // Every attn_fa_b variant of the cache format (batch::Fa).
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
        // (start, nb, z0 of the attn_blk_b launch, decode grid y).
        let cases: &[(usize, usize, usize, usize)] = &[
            (0, 1, 0, 1),
            (0, 17, 3, 46),
            (63, 16, 5, 2),
            (64, 17, 0, 1),
            (126, 40, 7, 3),
            (200, 70, 16, 46),
            (1000, 24, 8, 5),
            (1023, 1, 0, 46),
            (1100, 64, 33, 7),
            (1983, 130, 64, 23),
            (3000, 17, 1, 46),
            (8190, 37, 2, 46),
            (0, 1024, 1000, 46),
        ];
        for &(start, nb, z0, gy) in cases {
            if nb == 1024 && (hd == 128 && nh == 12 || kg > 0) {
                continue; // one 1024-batch per head size is enough
            }
            let ctx = start + nb;
            // Room past the batch (and NaN there).
            let max_ctx = ctx + 77;
            let nblk_max = max_ctx.div_ceil(64);
            let to_h = |v: Vec<f32>| -> Vec<u16> { v.into_iter().map(quant::f32_to_f16).collect() };
            // The cache's buffers (f16: K, V; q8: K, V, their scales).
            let mut bufs = Vec::new();
            if kg == 0 {
                let mut kh = to_h(rng.vec(ctx * kvd, 4.0));
                let mut vh = to_h(rng.vec(ctx * kvd, 1.0));
                kh.resize(max_ctx * kvd, 0xffff);
                vh.resize(max_ctx * kvd, 0xffff);
                bufs.push(g.upload(&kh).unwrap());
                bufs.push(g.upload(&vh).unwrap());
            } else {
                // Int8 values (garbage past ctx), scales (NaN past ctx).
                let mut i8s = || -> Vec<u8> {
                    (0..max_ctx * kvd)
                        .map(|_| (rng.next() % 255) as u8)
                        .collect()
                };
                let (k8, v8) = (i8s(), i8s());
                let mut sc = |scale: f32| -> Vec<f32> {
                    let mut d: Vec<f32> = (0..ctx * kvd / kg)
                        .map(|_| (rng.f() + 1.5) * scale / 127.0)
                        .collect();
                    d.resize(max_ctx * kvd / kg, f32::NAN);
                    d
                };
                let (ks, vs) = (sc(4.0), sc(1.0));
                bufs.push(g.upload(&k8).unwrap());
                bufs.push(g.upload(&v8).unwrap());
                bufs.push(g.upload(&ks).unwrap());
                bufs.push(g.upload(&vs).unwrap());
            }
            let kva: Vec<Arg> = bufs.iter().map(|b| b.arg()).collect();
            let q = rng.vec(nb * dim, 1.0);
            let ast = act_stride_t(dim);
            let want_k = Arg::I32(dim.is_multiple_of(256) as i32);
            // Flash, every variant: all nb queries, y in place of q, then act.
            let ctxs = |what: &str, z: usize| {
                format!(
                    "{nh}/{nkv}x{hd} kg {kg} start {start} nb {nb} z0 {z0}: {what} query {z} (np {})",
                    start + z + 1
                )
            };
            let mut flash: Vec<(batch::Fa, Vec<f32>, Vec<u8>)> = Vec::new();
            for (v, ff) in &ffs {
                let (fa_rows, fa_thr, fa_smem) = batch::fa_shape(hd, *v, kg);
                let qy = g.upload(&q).unwrap();
                let fact = g.alloc(nb * ast).unwrap();
                fact.zero().unwrap();
                // SAFETY: prefill.cu's signatures; q/y nb rows of dim, the
                // cache max_ctx rows, act nb strides.
                unsafe {
                    g.launch(
                        ff,
                        (nkv as u32, (nb * group).div_ceil(fa_rows) as u32, 1),
                        (fa_thr as u32, 1, 1),
                        fa_smem as u32,
                        &[
                            &[qy.arg()][..],
                            &kva[..],
                            &[
                                Arg::I32(start as i32),
                                Arg::I32(nb as i32),
                                Arg::I32(nh as i32),
                                Arg::I32(nkv as i32),
                                pre,
                                qy.arg(),
                                Arg::Ptr(0), // no page table
                            ],
                        ]
                        .concat(),
                    )
                    .unwrap();
                    g.launch(
                        &fq,
                        (dim.div_ceil(256) as u32, nb as u32, 1),
                        (32, 1, 1),
                        0,
                        &[
                            qy.arg(),
                            Arg::I32(dim as i32),
                            want_k,
                            fact.arg(),
                            Arg::U64(ast as u64),
                        ],
                    )
                    .unwrap();
                }
                flash.push((*v, qy.read().unwrap(), fact.read().unwrap()));
            }
            // Per query: queries z0.. of the batch.
            let nz = nb - z0;
            let qb = g.upload(&q).unwrap();
            let (po_n, pm_n) = (nh * nblk_max * hd, nh * nblk_max * 2);
            let bpo = g.alloc(nz * po_n * 4).unwrap();
            let bpm = g.alloc(nz * pm_n * 4).unwrap();
            let bact = g.alloc(nb * ast).unwrap();
            bact.zero().unwrap();
            // SAFETY: as above; part_* for nz queries of nblk_max blocks.
            unsafe {
                g.launch(
                    &fb,
                    (nkv as u32, gy.min(nblk_max) as u32, nz as u32),
                    (32, 1, 1),
                    0,
                    &[
                        &[qb.arg()][..],
                        &kva[..],
                        &[
                            Arg::I32(start as i32),
                            Arg::I32(z0 as i32),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            pre,
                            Arg::I32(nblk_max as i32),
                            bpo.arg(),
                            bpm.arg(),
                            Arg::Ptr(0),
                            Arg::Ptr(0), // no page table
                        ],
                    ]
                    .concat(),
                )
                .unwrap();
                g.launch(
                    &fcb,
                    (dim.div_ceil(256) as u32, nz as u32, 1),
                    (256, 1, 1),
                    0,
                    &[
                        bpo.arg(),
                        bpm.arg(),
                        Arg::I32(start as i32),
                        Arg::I32(z0 as i32),
                        Arg::I32(dim as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(nblk_max as i32),
                        want_k,
                        bact.arg(),
                        Arg::U64(ast as u64),
                        Arg::Ptr(0),
                    ],
                )
                .unwrap();
            }
            let (bpo, bpm, bact): (Vec<f32>, Vec<f32>, Vec<u8>) = (
                bpo.read().unwrap(),
                bpm.read().unwrap(),
                bact.read().unwrap(),
            );
            // Decode, query by query (all nb, for the flash side).
            let po = g.alloc(po_n * 4).unwrap();
            let pm = g.alloc(pm_n * 4).unwrap();
            let act = g.alloc(act_bytes(dim)).unwrap();
            let y = g.alloc(dim * 4).unwrap();
            let ab = act_bytes(dim);
            for z in 0..nb {
                let np = start + z + 1;
                let qz = g.upload(&q[z * dim..(z + 1) * dim]).unwrap();
                let params = g.upload(&[0u32, (np - 1) as u32, 0, 0]).unwrap();
                act.zero().unwrap();
                // SAFETY: decode.cu's signatures; part_* nblk_max blocks.
                unsafe {
                    g.launch(
                        &fd,
                        (nkv as u32, gy.min(nblk_max) as u32, 1),
                        (32 * att_nw(hd, kg > 0), 1, 1),
                        0,
                        &[
                            &[qz.arg()][..],
                            &kva[..],
                            // q8: no new row to store (kf null).
                            if kg > 0 { &[Arg::Ptr(0)][..] } else { &[] },
                            &[
                                params.arg(),
                                Arg::I32(nh as i32),
                                Arg::I32(nkv as i32),
                                pre,
                                Arg::I32(nblk_max as i32),
                                po.arg(),
                                pm.arg(),
                            ],
                            // No q/k norms (f16: kraw, qn, kn, eps, rope,
                            // kc_out; q8: ksh, qn, kn, eps, rope).
                            &[
                                Arg::Ptr(0),
                                Arg::Ptr(0),
                                Arg::Ptr(0),
                                Arg::F32(0.0),
                                Arg::Ptr(0),
                            ],
                            if kg > 0 { &[] } else { &[Arg::Ptr(0)][..] },
                            &[Arg::Ptr(0)], // no page table
                        ]
                        .concat(),
                    )
                    .unwrap();
                    launch_cmb(&g, &fc, [&po, &pm, &params], [dim, hd, nblk_max], &act, &y);
                }
                let (wy, wact): (Vec<f32>, Vec<u8>) = (y.read().unwrap(), act.read().unwrap());
                assert!(
                    wy.iter().all(|x| x.is_finite()),
                    "{}",
                    ctxs("decode y finite", z)
                );
                for (v, fy, fact) in &flash {
                    for (d, (a, b)) in fy[z * dim..(z + 1) * dim].iter().zip(&wy).enumerate() {
                        assert!(
                            a.to_bits() == b.to_bits() || (*a == 0.0 && *b == 0.0),
                            "{}: y[{d}] {a} vs {b}",
                            ctxs(&format!("flash {v:?}"), z)
                        );
                    }
                    assert!(
                        fact[z * ast..z * ast + ab] == wact[..],
                        "{}",
                        ctxs(&format!("flash {v:?} act"), z)
                    );
                }
                if z < z0 {
                    continue;
                }
                let zl = z - z0;
                let (wpo, wpm): (Vec<f32>, Vec<f32>) = (po.read().unwrap(), pm.read().unwrap());
                for h in 0..nh {
                    for b in 0..np.div_ceil(64) {
                        let o = (h * nblk_max + b) * hd;
                        assert_eq!(
                            bits(&bpo[zl * po_n + o..zl * po_n + o + hd]),
                            bits(&wpo[o..o + hd]),
                            "{}: head {h} block {b} sums",
                            ctxs("attn_blk_b", z)
                        );
                        let m = (h * nblk_max + b) * 2;
                        assert_eq!(
                            bits(&bpm[zl * pm_n + m..zl * pm_n + m + 2]),
                            bits(&wpm[m..m + 2]),
                            "{}: head {h} block {b} m, l",
                            ctxs("attn_blk_b", z)
                        );
                    }
                }
                assert!(
                    bact[z * ast..z * ast + ab] == wact[..],
                    "{}",
                    ctxs("attn_blk_b act", z)
                );
            }
        }
    }
}

/// The split attention of GPUs without tensor cores (forced here on any
/// GPU): attn_split_b (+ attn_combine_b) per query gives the decode
/// kernels' (attn_split + attn_combine) bits, split partials and output,
/// on the f16 cache and the q8 one (scale per head row or per 32 values),
/// queries at z0 > 0, past 1024 positions, one and several 128-position
/// tiles per split. The cache past the batch is NaN / garbage.
#[test]
fn gpu_split_attention_matches_decode_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let (fc, fcb) = (
        km.function("attn_combine").unwrap(),
        km.function("attn_combine_b").unwrap(),
    );
    let mut rng = Rng(9192);
    for (nh, nkv, hd, kg) in [
        (14, 2, 64, 0),
        (28, 4, 128, 0),
        (12, 2, 128, 0),
        (14, 2, 64, 32),
        (28, 4, 128, 128),
    ] {
        let (dim, kvd) = (nh * hd, nkv * hd);
        let q8n = match kg {
            0 => String::new(),
            _ if kg == hd => "q8_".into(),
            _ => "q8g32_".into(),
        };
        let fd = km.function(&format!("attn_split_{q8n}{hd}")).unwrap();
        let fb = km.function(&format!("attn_split_b_{q8n}{hd}")).unwrap();
        // (start, queries, z0, max_split)
        for (start, nz, z0, ms) in [
            (0, 40, 0, 23),
            (200, 30, 9, 8),
            (1000, 12, 3, 46),
            (1300, 9, 0, 4),
        ] {
            let ctx = start + z0 + nz;
            let max_ctx = ctx + 50;
            let mut bufs = Vec::new();
            if kg == 0 {
                for sc in [4.0, 1.0] {
                    let mut h: Vec<u16> = rng
                        .vec(ctx * kvd, sc)
                        .into_iter()
                        .map(quant::f32_to_f16)
                        .collect();
                    h.resize(max_ctx * kvd, 0xffff);
                    bufs.push(g.upload(&h).unwrap());
                }
            } else {
                for _ in 0..2 {
                    let v: Vec<u8> = (0..max_ctx * kvd)
                        .map(|_| (rng.next() % 255) as u8)
                        .collect();
                    bufs.push(g.upload(&v).unwrap());
                }
                // Scales: one per kg values (kg > 0 here).
                let groups = |pos: usize| pos * (kvd / kg.max(1));
                for sc in [4.0f32, 1.0] {
                    let mut d: Vec<f32> = (0..groups(ctx))
                        .map(|_| (rng.f() + 1.5) * sc / 127.0)
                        .collect();
                    d.resize(groups(max_ctx), f32::NAN);
                    bufs.push(g.upload(&d).unwrap());
                }
            }
            let kva: Vec<Arg> = bufs.iter().map(|b| b.arg()).collect();
            let q = rng.vec((z0 + nz) * dim, 1.0);
            let qb = g.upload(&q).unwrap();
            let (po_n, pm_n) = (nh * ms * hd, nh * ms * 2);
            let ast = act_stride_t(dim);
            let scale = Arg::F32(1.0 / (hd as f32).sqrt());
            let want_k = Arg::I32(dim.is_multiple_of(256) as i32);
            let bpo = g.alloc(nz * po_n * 4).unwrap();
            let bpm = g.alloc(nz * pm_n * 4).unwrap();
            let bact = g.alloc((z0 + nz) * ast).unwrap();
            bact.zero().unwrap();
            // SAFETY: prefill.cu's signatures; part_* for nz queries of ms
            // splits, the cache max_ctx rows.
            unsafe {
                let a = [
                    &[qb.arg()][..],
                    &kva[..],
                    &[
                        Arg::I32(start as i32),
                        Arg::I32(z0 as i32),
                        Arg::I32(nh as i32),
                        Arg::I32(nkv as i32),
                        scale,
                        Arg::I32(ms as i32),
                        bpo.arg(),
                        bpm.arg(),
                        Arg::Ptr(0),
                        Arg::Ptr(0), // no page table
                    ],
                ]
                .concat();
                g.launch(
                    &fb,
                    (nkv as u32, ms as u32, nz as u32),
                    (ATT_TILE as u32, 1, 1),
                    0,
                    &a,
                )
                .unwrap();
                g.launch(
                    &fcb,
                    (dim.div_ceil(256) as u32, nz as u32, 1),
                    (256, 1, 1),
                    0,
                    &[
                        bpo.arg(),
                        bpm.arg(),
                        Arg::I32(start as i32),
                        Arg::I32(z0 as i32),
                        Arg::I32(dim as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(ms as i32),
                        want_k,
                        bact.arg(),
                        Arg::U64(ast as u64),
                        Arg::Ptr(0),
                    ],
                )
                .unwrap();
            }
            let (bpo, bpm, bact): (Vec<f32>, Vec<f32>, Vec<u8>) = (
                bpo.read().unwrap(),
                bpm.read().unwrap(),
                bact.read().unwrap(),
            );
            let (po, pm) = (g.alloc(po_n * 4).unwrap(), g.alloc(pm_n * 4).unwrap());
            let act = g.alloc(act_bytes(dim)).unwrap();
            for zl in 0..nz {
                let np = start + z0 + zl + 1;
                let qz = g.upload(&q[(z0 + zl) * dim..(z0 + zl + 1) * dim]).unwrap();
                let params = g.upload(&[0u32, (np - 1) as u32, 0, 0]).unwrap();
                act.zero().unwrap();
                // SAFETY: decode.cu's signatures; part_* of ms splits.
                unsafe {
                    let a = [
                        &[qz.arg()][..],
                        &kva[..],
                        // q8: no new row to store (kf null).
                        if kg > 0 { &[Arg::Ptr(0)][..] } else { &[] },
                        &[
                            params.arg(),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            scale,
                            Arg::I32(ms as i32),
                            po.arg(),
                            pm.arg(),
                        ],
                        // No q/k norms (f16: kraw, qn, kn, eps, rope,
                        // kc_out; q8: ksh, qn, kn, eps, rope).
                        &[
                            Arg::Ptr(0),
                            Arg::Ptr(0),
                            Arg::Ptr(0),
                            Arg::F32(0.0),
                            Arg::Ptr(0),
                        ],
                        if kg > 0 { &[] } else { &[Arg::Ptr(0)][..] },
                        &[Arg::Ptr(0)], // no page table
                    ]
                    .concat();
                    g.launch(
                        &fd,
                        (nkv as u32, ms as u32, 1),
                        (ATT_TILE as u32, 1, 1),
                        0,
                        &a,
                    )
                    .unwrap();
                    g.launch(
                        &fc,
                        (dim.div_ceil(256) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        &[
                            po.arg(),
                            pm.arg(),
                            params.arg(),
                            Arg::I32(dim as i32),
                            Arg::I32(hd as i32),
                            Arg::I32(ms as i32),
                            want_k,
                            act.arg(),
                            Arg::Ptr(0),
                        ],
                    )
                    .unwrap();
                }
                let (wpo, wpm, wact): (Vec<f32>, Vec<f32>, Vec<u8>) =
                    (po.read().unwrap(), pm.read().unwrap(), act.read().unwrap());
                let at = format!("{nh}/{nkv}x{hd} kg {kg} np {np} z0 {z0} splits {ms}");
                let nsplit = ms.min(np.div_ceil(32));
                for h in 0..nh {
                    for sp in 0..nsplit {
                        let o = (h * ms + sp) * hd;
                        assert_eq!(
                            bits(&bpo[zl * po_n + o..zl * po_n + o + hd]),
                            bits(&wpo[o..o + hd]),
                            "{at}: sums head {h} split {sp}"
                        );
                        let mm = (h * ms + sp) * 2;
                        assert_eq!(
                            bits(&bpm[zl * pm_n + mm..zl * pm_n + mm + 2]),
                            bits(&wpm[mm..mm + 2]),
                            "{at}: m, l head {h} split {sp}"
                        );
                    }
                }
                let za = (z0 + zl) * ast;
                assert!(bact[za..za + act_bytes(dim)] == wact[..], "{at}: output");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The q8 KV cache (CacheKv::Q8).

#[test]
fn gpu_prefill_batchings_bit_identical_q8_0_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    prefill_bits(&m, "0.5B", CacheKv::Q8 { group: 64 });
    prefill_bits(&m, "0.5B", CacheKv::Q8 { group: 32 });
}

#[test]
fn gpu_prefill_batchings_bit_identical_q8_1_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    prefill_bits(&m, "1.5B", CacheKv::Q8 { group: 128 });
    prefill_bits(&m, "1.5B", CacheKv::Q8 { group: 32 });
}

#[test]
fn gpu_verify_matches_forward_argmax_q8_0_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    verify_bits(&m, "0.5B", CacheKv::Q8 { group: 64 });
}

#[test]
fn gpu_verify_matches_forward_argmax_q8_1_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    verify_bits(&m, "1.5B", CacheKv::Q8 { group: 128 });
}

/// decode.cu's q8 rule on one group of values: (int8 values, scale).
fn q8_ref(x: &[f32]) -> (Vec<i8>, f32) {
    // Saturate as the f16 cache (f32::clamp passes NaN through).
    let sat = |v: f32| v.clamp(-65504.0, 65504.0);
    let amax = x
        .iter()
        .map(|&v| sat(v).abs())
        .fold(0f32, |m, a| if a > m { a } else { m });
    let d = if x.iter().any(|v| v.is_nan()) {
        f32::NAN
    } else {
        amax / 127.0
    };
    let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
    let q = x
        .iter()
        .map(|&v| {
            let y = sat(v) * id;
            if y.is_nan() {
                0
            } else {
                (y.round_ties_even() as i32).clamp(-127, 127) as i8
            }
        })
        .collect();
    (q, d)
}

/// The batched q8 store (kv_q8*_b, the code the decode step's attn_split_q8
/// runs too: the batchings tests show the two agree) == `q8_ref` bit for
/// bit, for both head sizes and scale groups, without and with a key shift
/// (keys - shift, then the rule): random rows, exact ties (amax 127, so x *
/// 127 / amax = x: k + 0.5 rounds to even), values past f16's range
/// (saturated), an all-zero row, a NaN (its group's scale NaN, the value
/// 0). RoPE at angle 0 (sin 0, cos 1) leaves the keys as given.
#[test]
fn gpu_q8_store_matches_rule() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(4242);
    for (hd, shifted) in [(64usize, false), (128, false), (64, true), (128, true)] {
        for group in [hd, 32] {
            let name = if group == hd {
                format!("kv_q8_b_{hd}")
            } else {
                format!("kv_q8g32_b_{hd}")
            };
            let f = km.function(&name).unwrap();
            let (nkv, nb, start) = (2usize, 4usize, 3usize);
            let nk = nkv * hd;
            // Token rows [nq = 0 | keys | values].
            let mut rows: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(2 * nk, 3.0)).collect();
            for (i, v) in rows[1].iter_mut().enumerate() {
                *v = [127.0, -127.0, 2.5, -2.5, 3.5, 0.5, -0.5, 1.5, 126.5, -126.5][i % 10];
            }
            rows[2][5] = 1e6;
            rows[2][nk + 7] = -7e4;
            rows[2][nk + hd + 1] = 65504.5;
            for v in &mut rows[3][hd..nk] {
                *v = 0.0;
            }
            rows[3][nk + 3] = f32::NAN;
            // A shift of zeros but for a few big entries (as the slow RoPE
            // pairs' key bias), the tie row's left alone.
            let shift: Vec<f32> = (0..nk)
                .map(|i| {
                    if shifted && i % 13 == 5 {
                        100.0 + i as f32
                    } else {
                        0.0
                    }
                })
                .collect();
            for (i, v) in rows[1][..nk].iter_mut().enumerate() {
                *v += shift[i];
            }
            let qkv: Vec<f32> = rows.concat();
            let shift_b = g.upload(&shift).unwrap();
            let rope: Vec<f32> = (0..nb * hd / 2).flat_map(|_| [0.0f32, 1.0]).collect();
            let qkv_b = g.upload(&qkv).unwrap();
            let rope_b = g.upload(&rope).unwrap();
            let n = (start + nb) * nk;
            let (kq, vq) = (g.alloc(n).unwrap(), g.alloc(n).unwrap());
            let (ks, vs) = (
                g.alloc(n / group * 4).unwrap(),
                g.alloc(n / group * 4).unwrap(),
            );
            // SAFETY: prefill.cu kv_q8_b's signature; qkv holds nb rows of
            // 2 nk floats, rope nb x hd / 2 pairs, the caches start + nb rows.
            unsafe {
                g.launch(
                    &f,
                    (nkv as u32, nb as u32, 1),
                    (128, 1, 1),
                    0,
                    &[
                        qkv_b.arg(),
                        Arg::I32(0),
                        Arg::I32(nk as i32),
                        Arg::I32(start as i32),
                        rope_b.arg(),
                        kq.arg(),
                        vq.arg(),
                        ks.arg(),
                        vs.arg(),
                        if shifted { shift_b.arg() } else { Arg::Ptr(0) },
                        Arg::Ptr(0), // no qwen3 k norm
                        Arg::F32(0.0),
                        Arg::Ptr(0), // one sequence
                        Arg::Ptr(0), // no page table
                    ],
                )
                .unwrap();
            }
            g.sync().unwrap();
            let i8s = |b: &DevBuf| -> Vec<i8> {
                b.read::<u8>().unwrap().iter().map(|&x| x as i8).collect()
            };
            let (kq, vq) = (i8s(&kq), i8s(&vq));
            let (ks, vs): (Vec<f32>, Vec<f32>) = (ks.read().unwrap(), vs.read().unwrap());
            for (z, row) in rows.iter().enumerate() {
                let pos = start + z;
                for (half, (q, s)) in [(&kq, &ks), (&vq, &vs)].into_iter().enumerate() {
                    let x: Vec<f32> = row[half * nk..(half + 1) * nk]
                        .iter()
                        .zip(&shift)
                        .map(|(&x, &c)| if half == 0 { x - c } else { x })
                        .collect();
                    for g0 in (0..nk).step_by(group) {
                        let (wq, wd) = q8_ref(&x[g0..g0 + group]);
                        let e = pos * nk + g0;
                        let at = format!("{name} shift {shifted} token {z} kv {half} @{g0}");
                        assert_eq!(&q[e..e + group], &wq[..], "{at}");
                        let d = s[e / group];
                        assert!(
                            d.to_bits() == wd.to_bits() || d.is_nan() && wd.is_nan(),
                            "{at}: {d} vs {wd}"
                        );
                    }
                }
            }
            // The rule's edge cases came out as meant.
            assert_eq!(kq[(start + 1) * nk + 2], 2); // 2.5 -> 2
            assert_eq!(kq[(start + 1) * nk + 4], 4); // 3.5 -> 4
            assert!(vs[((start + 3) * nk + 3) / group].is_nan());
            if !shifted {
                assert_eq!(ks[((start + 3) * nk + hd) / group], 0.0); // all-zero keys
            }
        }
    }
}

/// A `CudaQwen2` whose caches are of type `kv` (comparisons of cache types
/// over one upload of the weights).
struct KvNet<'a, 'g>(&'a CudaQwen2<'g>, CacheKv);

impl<'g> Lm for KvNet<'_, 'g> {
    type Cache = CudaCache<'g>;
    fn vocab(&self) -> usize {
        self.0.vocab
    }
    fn new_cache(&self, max_ctx: usize) -> CudaCache<'g> {
        self.0.try_cache_kv(max_ctx, self.1).unwrap()
    }
    fn forward(&self, token: u32, pos: usize, c: &mut CudaCache<'g>) -> Vec<f32> {
        Lm::forward(self.0, token, pos, c)
    }
    fn forward_argmax(&self, token: u32, pos: usize, c: &mut CudaCache<'g>) -> u32 {
        self.0.forward_argmax(token, pos, c)
    }
    fn feed_batch(&self, tokens: &[u32], start: usize, c: &mut CudaCache<'g>) {
        Lm::feed_batch(self.0, tokens, start, c)
    }
    fn verify_argmax(&self, tokens: &[u32], start: usize, c: &mut CudaCache<'g>) -> Vec<u32> {
        Lm::verify_argmax(self.0, tokens, start, c)
    }
    fn prefill_batch(&self) -> usize {
        self.0.prefill_batch()
    }
}

/// `r` vs `o` logits at positions from..ids.len(), both fed ids (teacher
/// forcing: every position sees the same text), ids[..from] by feed_batch.
fn compare_forced<R: Lm, O: Lm>(r: &R, o: &O, ids: &[u32], from: usize, a: &mut Agree) {
    let mut rc = r.new_cache(ids.len());
    let mut oc = o.new_cache(ids.len());
    for (i, ch) in ids[..from].chunks(512).enumerate().filter(|_| from > 0) {
        r.feed_batch(ch, i * 512, &mut rc);
        o.feed_batch(ch, i * 512, &mut oc);
    }
    a.last_kl = 0.0;
    for (pos, &t) in ids.iter().enumerate().skip(from) {
        let lr = r.forward(t, pos, &mut rc);
        let lo = o.forward(t, pos, &mut oc);
        let (ar, ao) = (crate::argmax(&lr), crate::argmax(&lo));
        a.positions += 1;
        a.top1 += (ar == ao) as usize;
        let k = kl(&lr, &lo);
        a.max_kl = a.max_kl.max(k);
        if k > 0.5 {
            a.over_half
                .push((a.positions, k, top2_margin(&lr), a.last_kl));
        }
        a.last_kl = k;
        a.sum_kl += k;
        for (x, y) in lr.iter().zip(&lo) {
            a.max_abs = a.max_abs.max((x - y).abs());
        }
        if ar != ao {
            a.flips.push(lr[ar as usize] - lr[ao as usize]);
            a.flip_at.push(a.positions);
        }
    }
}

/// ~n tokens of natural text, prose and code: tests/fixtures/natural_text.txt,
/// a frozen copy of README.md, src/kv.rs and src/chat.rs as of 5f96039
/// (joined by newlines). It used to read the live files, so every docs edit
/// moved the near-ties these accuracy bounds were measured on: after #28's
/// README rewrite, natural_long_1_5b (464/500 vs a floor of 466) and
/// gpu_q8_qwen3 (text top-1 76.3 % vs 80 %) failed on every sm_89 host,
/// though both pass on this text there (and on sm_80).
fn natural_ids(m: &Model, n: usize) -> Vec<u32> {
    let text = include_str!("../../../tests/fixtures/natural_text.txt");
    let mut ids = crate::tokenizer::Tokenizer::from_gguf(m).encode(text);
    assert!(ids.len() >= n, "{} tokens of text", ids.len());
    ids.truncate(n);
    ids
}

/// Reference `r` vs `o`: PROMPTS with 32 greedy steps each (r's picks fed
/// to both); positions 0..near and long..long + far of natural text,
/// teacher-forced.
pub(crate) fn kv_compare<R: Lm, O: Lm>(
    r: &R,
    o: &O,
    m: &Model,
    near: usize,
    long: usize,
    far: usize,
) -> [Agree; 3] {
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let mut a: [Agree; 3] = Default::default();
    for p in PROMPTS {
        compare(r, o, &tok.encode(p), 32, &mut a[0]);
    }
    let ids = natural_ids(m, long + far);
    compare_forced(r, o, &ids[..near], 0, &mut a[1]);
    compare_forced(r, o, &ids, long, &mut a[2]);
    a
}

/// q8 (group 0 = head_dim, or 32) vs f16 caches on one GPU model
/// (`kv_compare`).
fn q8_vs_f16(
    net: &CudaQwen2,
    m: &Model,
    group: usize,
    near: usize,
    long: usize,
    far: usize,
) -> [Agree; 3] {
    let f16 = KvNet(net, CacheKv::F16);
    let q8 = KvNet(net, CacheKv::Q8 { group });
    kv_compare(&f16, &q8, m, near, long, far)
}

/// The q8 cache (per-row scales, key shift) against the f16 one, both on
/// the GPU: PROMPTS (32 greedy steps each) and natural text teacher-forced
/// at positions 0..300 and 2048..2304. Measured (NOTES), mean KL prompts /
/// text near / text far: 0.5B 1.4e-2 / 1.0e-2 / 6.4e-3, 1.5B 1.3e-2 / 9.8e-3
/// / 7.2e-3; the CPU's own two kernel sets differ by 2.8e-3 / 3.0e-3 /
/// 2.5e-3 (0.5B) and 7.8e-3 / 8.5e-3 / 6.1e-3, top-1 down to 93.7% (1.5B)
/// there. Without the key shift q8 gives 2.8e-2 to 7.6e-2, a broken kernel
/// ~1. Asserted: text windows mean KL <= 1.5e-2 and top-1 >= 90%; prompts
/// mean KL <= 3e-2; and as a gross-bug check only, every position's KL < 4
/// (the worst measured: 1.24, one 1.5B prompt position).
/// 0.5B is the sensitive one (one fast-rotating K bias pair of |b| ~ 47 no
/// shift can remove); 1.5B and 7B sit at the CPU-vs-CPU level.
fn q8_accuracy(m: &Model, name: &str) {
    q8_accuracy_within(m, name, 3e-2, 1.5e-2, 90);
}

/// `q8_accuracy` with its bounds: prompts' and text windows' mean KL, and
/// the text windows' top-1 in percent.
fn q8_accuracy_within(m: &Model, name: &str, prompts_kl: f64, text_kl: f64, top1_pct: usize) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::new(&g, &km, m, 2400).unwrap();
    let a = q8_vs_f16(&net, m, 0, 300, 2048, 256);
    for (w, what) in a.iter().zip(["prompts", "text 0..300", "text 2048+256"]) {
        report(&format!("{name} q8 vs f16 {what}"), w);
        assert!(w.max_kl < 4.0, "{name} {what}: max KL {}", w.max_kl);
    }
    assert!(
        a[0].sum_kl <= prompts_kl * a[0].positions as f64,
        "{name} prompts: mean KL"
    );
    for w in &a[1..] {
        assert!(
            w.sum_kl <= text_kl * w.positions as f64,
            "{name} text: mean KL"
        );
        assert!(
            w.top1 * 100 >= top1_pct * w.positions,
            "{name} text: top-1 {}",
            w.top1
        );
    }
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    continuations_with(
        &KvNet(&net, CacheKv::F16),
        &KvNet(&net, CacheKv::Q8 { group: 0 }),
        &tok,
        &format!("{name} q8 vs f16"),
        f32::INFINITY,
    );
}

#[test]
fn gpu_q8_kv_close_to_f16_0_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    q8_accuracy(&m, "0.5B");
}

#[test]
fn gpu_q8_kv_close_to_f16_1_5b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at("models/qwen2.5-1.5b-instruct-q4_k_m.gguf") else {
        return;
    };
    q8_accuracy(&m, "1.5B");
}

/// qwen3 on the q8 cache: the decode q8 attention normalizes and rotates
/// the raw keys before quantizing them (attn_split_t), batched passes in
/// kv_q8_b, the same warp_norm_rope: batchings and verify keep the bits,
/// both scale groups.
///
/// Accuracy: 0.6B's k_norm gains have large outlier channels (max / median
/// 6.5x in the median layer, up to 42x; 4B 2.7x / 23x), which set each
/// row's q8 scale, so per-row q8 is coarse there. Measured q8 vs f16 mean
/// KL prompts / text near / text far: 3.8e-2 / 5.0e-2 / 3.3e-2, top-1 down
/// to 86.7% (q8/32: 1.4e-2 / 2.9e-2 / 1.4e-2); the model's own floor (CPU
/// kernel sets, GPU f16 vs CPU) 1.0e-3 to 1.9e-3. A wrong norm (gain, order)
/// gives KL ~1, so the bounds here, 2x the measured q8, still catch one; 4B
/// meets qwen2's bounds.
#[test]
fn gpu_q8_qwen3() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = qwen3_mapped() else {
        return;
    };
    prefill_bits(&m, "qwen3 0.6B", CacheKv::Q8 { group: 128 });
    prefill_bits(&m, "qwen3 0.6B", CacheKv::Q8 { group: 32 });
    verify_bits(&m, "qwen3 0.6B", CacheKv::Q8 { group: 128 });
    q8_accuracy_within(&m, "qwen3 0.6B", 8e-2, 1e-1, 80);
}

#[test]
fn gpu_q8_qwen3_4b() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = model_at(QWEN3_4B) else {
        return;
    };
    prefill_bits(&m, "qwen3 4B", CacheKv::Q8 { group: 128 });
    verify_bits(&m, "qwen3 4B", CacheKv::Q8 { group: 128 });
    q8_accuracy(&m, "qwen3 4B");
}

/// Measurement (not a check; `--ignored`, the model from AXON_TEST_MODEL):
/// q8 and q8/32 vs f16 (GPU), next to f16 GPU vs CPU and the CPU's per-row
/// kernels vs its default ones (its own noise; AXON_Q8_CPU=0 skips the CPU),
/// short prompts and at AXON_Q8_LONG (default 2048) tokens of text.
#[test]
#[ignore]
fn cuda_kv_q8_accuracy() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model_with(Some(false)) else {
        return;
    };
    let long: usize = std::env::var("AXON_Q8_LONG").map_or(2048, |v| v.parse().unwrap());
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::new(&g, &km, &m, long + 300).unwrap();
    let name = crate::testutil::model_path().unwrap();
    let name = name.file_name().unwrap().to_string_lossy();
    let rep3 = |what: &str, a: &[Agree; 3]| {
        report(&format!("{name} {what} prompts"), &a[0]);
        report(&format!("{name} {what} text 0..300"), &a[1]);
        report(&format!("{name} {what} text {long}+256"), &a[2]);
    };
    for group in [0, 32] {
        rep3(
            &format!("q8/{group} vs f16"),
            &q8_vs_f16(&net, &m, group, 300, long, 256),
        );
    }
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    for group in [0, 32] {
        continuations_with(
            &KvNet(&net, CacheKv::F16),
            &KvNet(&net, CacheKv::Q8 { group }),
            &tok,
            &format!("{name} q8/{group} vs f16"),
            f32::INFINITY,
        );
    }
    if std::env::var("AXON_Q8_CPU").is_ok_and(|v| v == "0") {
        return;
    }
    let rows = Qwen2::new_with(&m, false);
    let cpu = Qwen2::new(&m);
    continuations_with(
        &cpu,
        &rows,
        &tok,
        &format!("{name} cpu rows vs cpu"),
        f32::INFINITY,
    );
    rep3(
        "cpu rows vs cpu",
        &kv_compare(&cpu, &rows, &m, 300, long, 256),
    );
    if std::env::var("AXON_Q8_CPU").is_ok_and(|v| v == "rows") {
        return;
    }
    rep3(
        "gpu f16 vs cpu",
        &kv_compare(&cpu, &KvNet(&net, CacheKv::F16), &m, 300, long, 256),
    );
}

/// The q8 decode attention == an f64 reference on the dequantized cache
/// (d q per value), the new token's row stored by the kernel itself from
/// f32 (kf) by `q8_ref`'s rule: both head sizes and scale groups, single and
/// many tiles and splits; the split kernels (attn_split_q8*, + attn_combine;
/// GPUs without tensor cores) to f32 rounding, the tensor-core ones
/// (attn_blk_q8*, + attn_cmb: f16 queries, values and probabilities) to
/// f16's.
#[test]
fn gpu_q8_attention_matches_reference() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(777);
    let mut worst = 0f64;
    for tc in [false, true] {
        if tc && !km.mma() {
            continue;
        }
        let fc = km
            .function(if tc { "attn_cmb" } else { "attn_combine" })
            .unwrap();
        for (nh, nkv, hd) in [(14, 2, 64), (28, 4, 128), (40, 8, 128)] {
            let (dim, kvd) = (nh * hd, nkv * hd);
            for group in [hd, 32] {
                let q8n = if group == hd { "q8" } else { "q8g32" };
                let kn = if tc { "attn_blk" } else { "attn_split" };
                let fs = km.function(&format!("{kn}_{q8n}_{hd}")).unwrap();
                for (np, max_split) in
                    [(1usize, 23usize), (7, 46), (129, 23), (1000, 4), (4099, 46)]
                {
                    // The split count, or the tensor-core kernels' grid and
                    // blocks.
                    let nblk = np.div_ceil(64);
                    let (grid, block, parts) = match tc {
                        true => (max_split.min(nblk), 32 * att_nw(hd, true), nblk),
                        false => (max_split, ATT_TILE as u32, max_split),
                    };
                    let q = rng.vec(dim, 1.0);
                    // Keys with a few large channels (as Qwen's), values O(1).
                    let mut k = rng.vec(np * kvd, 2.0);
                    for (i, x) in k.iter_mut().enumerate() {
                        if i % hd == 3 || i % hd == hd / 2 + 3 {
                            *x *= 20.0;
                        }
                    }
                    let v = rng.vec(np * kvd, 1.0);
                    // Rows 0..np - 1 quantized here; row np - 1 by the kernel.
                    let quant = |x: &[f32]| {
                        let (mut qs, mut ds) = (Vec::new(), Vec::new());
                        for c in x.chunks(group) {
                            let (qq, d) = q8_ref(c);
                            qs.extend(qq.iter().map(|&b| b as u8));
                            ds.push(d);
                        }
                        (qs, ds)
                    };
                    let (kq, ks) = quant(&k);
                    let (vq, vs) = quant(&v);
                    let deq =
                        |qs: &[u8], ds: &[f32], i: usize| ds[i / group] as f64 * qs[i] as i8 as f64;
                    let up8 = |b: &[u8], new: bool| {
                        let mut b = b.to_vec();
                        if new {
                            b[(np - 1) * kvd..].fill(0x55); // the kernel writes these
                        }
                        g.upload(&b).unwrap()
                    };
                    let (kb, vb) = (up8(&kq, true), up8(&vq, true));
                    // Row np - 1's scales poisoned too: the kernel must store them.
                    let upf = |d: &[f32]| {
                        let mut d = d.to_vec();
                        d[(np - 1) * kvd / group..].fill(f32::NAN);
                        g.upload(&d).unwrap()
                    };
                    let (ksb, vsb) = (upf(&ks), upf(&vs));
                    let mut kf = k[(np - 1) * kvd..].to_vec();
                    kf.extend_from_slice(&v[(np - 1) * kvd..]);
                    let (qb, kfb) = (g.upload(&q).unwrap(), g.upload(&kf).unwrap());
                    let params = g.upload(&[0u32, (np - 1) as u32, 0, 0]).unwrap();
                    let po = g.alloc(nh * parts * hd * 4).unwrap();
                    let pml = g.alloc(nh * parts * 2 * 4).unwrap();
                    let act = g.alloc(act_bytes(dim)).unwrap();
                    let out = g.alloc(dim * 4).unwrap();
                    // SAFETY: decode.cu's signatures; the cache holds np rows
                    // (int8) and their scales, kf 2 kv_dim floats.
                    unsafe {
                        g.launch(
                            &fs,
                            (nkv as u32, grid as u32, 1),
                            (block, 1, 1),
                            0,
                            &[
                                qb.arg(),
                                kb.arg(),
                                vb.arg(),
                                ksb.arg(),
                                vsb.arg(),
                                kfb.arg(),
                                params.arg(),
                                Arg::I32(nh as i32),
                                Arg::I32(nkv as i32),
                                Arg::F32(if tc {
                                    att_pre(hd)
                                } else {
                                    1.0 / (hd as f32).sqrt()
                                }),
                                Arg::I32(parts as i32),
                                po.arg(),
                                pml.arg(),
                                Arg::Ptr(0),
                                // No qwen3 norms (qn, kn, eps, rope).
                                Arg::Ptr(0),
                                Arg::Ptr(0),
                                Arg::F32(0.0),
                                Arg::Ptr(0),
                                Arg::Ptr(0), // no page table
                            ],
                        )
                        .unwrap();
                        if tc {
                            launch_cmb(&g, &fc, [&po, &pml, &params], [dim, hd, parts], &act, &out);
                        } else {
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
                                    Arg::I32(parts as i32),
                                    Arg::I32(dim.is_multiple_of(256) as i32),
                                    act.arg(),
                                    out.arg(),
                                ],
                            )
                            .unwrap();
                        }
                    }
                    let got: Vec<f32> = out.read().unwrap();
                    // The kernel's store of row np - 1 is the rule's.
                    let at = format!("{kn} {q8n} hd {hd} np {np}");
                    assert_eq!(kb.read::<u8>().unwrap(), kq, "{at}: stored keys");
                    assert_eq!(vb.read::<u8>().unwrap(), vq, "{at}: stored values");
                    let ksgot: Vec<f32> = ksb.read().unwrap();
                    let vsgot: Vec<f32> = vsb.read().unwrap();
                    assert_eq!(bits(&ksgot), bits(&ks), "{at}: stored key scales");
                    assert_eq!(bits(&vsgot), bits(&vs), "{at}: stored value scales");
                    let gq = nh / nkv;
                    let mut de = 0f64;
                    for h in 0..nh {
                        let kvh = h / gq;
                        let s: Vec<f64> = (0..np)
                            .map(|t| {
                                (0..hd)
                                    .map(|d| {
                                        q[h * hd + d] as f64 * deq(&kq, &ks, t * kvd + kvh * hd + d)
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
                                .map(|t| p[t] * deq(&vq, &vs, t * kvd + kvh * hd + d))
                                .sum::<f64>()
                                / l;
                            de = de.max((got[h * hd + d] as f64 - e).abs());
                        }
                    }
                    // As gpu_attention_matches_cpu: outputs O(1), f32 sums;
                    // tc: f16 roundings (gpu_tc_attention_close_to_f64).
                    let bound = if tc { 1e-2 } else { 2e-5 };
                    if tc {
                        worst = worst.max(de);
                    }
                    assert!(
                        de < bound,
                        "{kn} {q8n} heads {nh}/{nkv} hd {hd} np {np}: vs f64 {de:e}"
                    );
                }
            }
        }
    }
    eprintln!("q8 tc attention: worst |out - f64| {worst:e}");
}

/// Sustained, interleaved A/B of gemm2 (64 x 128) vs gemm2s (64 x 64) per
/// shape: AXON_SUS_SHAPES "Q4_K:rowsxcols,Q6_K:rowsxcols,..",
/// AXON_SUS_NB "559,1050", AXON_SUS_N launches per block (default 50),
/// AXON_SUS_ROUNDS (default 4): blocks alternate kernel A, B, A, B.. after a
/// ~1 s warm-up (the GPU at its power cap, as in a prefill); per kernel the
/// median block's us per launch. The host's choice (bgemm) is printed too.
#[test]
#[ignore]
fn cuda_gemm_sustained() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let shapes = std::env::var("AXON_SUS_SHAPES")
        .unwrap_or("Q4_K:18944x3584,Q4_K:3584x18944,Q6_K:3584x18944,Q4_K:3584x3584".into());
    let nbs: Vec<usize> = std::env::var("AXON_SUS_NB")
        .unwrap_or("559,1050".into())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let n: usize = std::env::var("AXON_SUS_N").map_or(50, |v| v.parse().unwrap());
    let rounds: usize = std::env::var("AXON_SUS_ROUNDS").map_or(4, |v| v.parse().unwrap());
    let mut rng = Rng(7);
    let mut warm = false;
    for sh in shapes.split(',') {
        let (tq, rc) = sh.split_once(':').unwrap();
        let qt = match tq {
            "Q4_K" => Q4_K,
            "Q6_K" => Q6_K,
            "Q8_0" => Q8_0,
            _ => Q5_0,
        };
        let (rows, cols) = rc.split_once('x').unwrap();
        let (rows, cols): (usize, usize) = (rows.parse().unwrap(), cols.parse().unwrap());
        let rb = quant::row_bytes(qt, cols);
        let w = synth_matrix(&mut rng, qt, rows, cols);
        let wb = g.upload(&w).unwrap();
        for &nb in &nbs {
            let xs: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 1.0)).collect();
            let act = quantize_batch(&g, &km, &xs, cols);
            let out = g.alloc(rows * nb * 4).unwrap();
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
            let mut kernels = gemm2_names(qt);
            if std::env::var("AXON_SUS_ONLY2").is_ok() {
                kernels.truncate(2);
            }
            // Q6_K: its prefill copy's kernels too (the copy's arguments).
            let w8 = (qt == Q6_K).then(|| q6k8_copy(&g, &km, &wb, rows, cols));
            let mut args8 = args;
            if let Some(w8) = &w8 {
                kernels.extend(["gemm2_q6k8".to_string(), "gemm2s_q6k8".to_string()]);
                args8[0] = w8.arg();
                args8[1] = Arg::U64((cols / 256 * Q6K8_BYTES) as u64);
            }
            let fs: Vec<_> = kernels.iter().map(|k| km.function(k).unwrap()).collect();
            let block = |ki: usize, reps: usize| {
                let a = if kernels[ki].ends_with("q6k8") {
                    &args8
                } else {
                    &args
                };
                let e0 = g.record().unwrap();
                for _ in 0..reps {
                    // SAFETY: as in gpu_batched.
                    unsafe { launch_product(&g, &fs[ki], &kernels[ki], qt, rows, nb, a) };
                }
                let e1 = g.record().unwrap();
                e0.ms_to(&e1).unwrap() as f64 * 1e3 / reps as f64
            };
            if !warm {
                let t = std::time::Instant::now();
                while t.elapsed().as_secs_f64() < 1.5 {
                    block(0, 10);
                }
                warm = true;
            }
            let mut us = vec![Vec::new(); kernels.len()];
            for r in 0..rounds {
                for i in 0..kernels.len() {
                    // A B / B A alternately (neither always goes first).
                    let ki = if r % 2 == 0 { i } else { kernels.len() - 1 - i };
                    us[ki].push(block(ki, n));
                }
            }
            let macs = (rows * cols * nb) as f64;
            let med: Vec<f64> = us
                .iter_mut()
                .map(|v| {
                    v.sort_by(f64::total_cmp);
                    v[v.len() / 2]
                })
                .collect();
            let line: Vec<String> = kernels
                .iter()
                .zip(&med)
                .map(|(k, u)| format!("{k} {u:8.1} us {:5.1} TMAC/s", macs / (u * 1e-6) / 1e12))
                .collect();
            let wi = (0..med.len())
                .min_by(|&a, &b| med[a].total_cmp(&med[b]))
                .unwrap();
            eprintln!(
                "SUS {tq} {rows}x{cols} nb {nb:>5}: {}  | best {} (gemm2s/gemm2 {:+.1}%)",
                line.join("  "),
                kernels[wi],
                100.0 * (med[1] / med[0] - 1.0)
            );
        }
    }
}

/// Teacher-forced dump for the llama.cpp oracle (scratch tooling, run by
/// hand): AXON_TF_MODEL (a GGUF), AXON_TF_IN (JSON: {"gens": [{"prompt_ids",
/// "tokens"}]}), AXON_TF_DEV (cpu or cuda); writes AXON_TF_OUT: per
/// sequence, at the prompt's end and after each of its tokens, the top 5
/// (id, logprob). Prompt prefilled in batches, the rest one token a pass.
/// AXON_TF_CTX: the GPU context (default 1024); AXON_TF_YARN=F: YaRN
/// factor F over the file's context (`with_yarn`).
#[test]
#[ignore]
fn teacher_forced_dump() {
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
    let mut m = crate::gguf::load_with(&var("AXON_TF_MODEL"), Some(false)).unwrap();
    // AXON_TF_YARN=F: --rope-scaling yarn --rope-scale F.
    if let Ok(f) = std::env::var("AXON_TF_YARN") {
        let orig = m.ctx_len() as usize;
        with_yarn(&mut m, f.parse().unwrap(), orig);
    }
    let max_ctx = std::env::var("AXON_TF_CTX").map_or(1024, |v| v.parse().unwrap());
    let inp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(var("AXON_TF_IN")).unwrap()).unwrap();
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        (v.as_array().unwrap().iter())
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let top = |l: &[f32]| -> Vec<(usize, f64)> {
        let lp = log_softmax(l);
        let mut o: Vec<usize> = (0..lp.len()).collect();
        o.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
        o[..5].iter().map(|&i| (i, lp[i])).collect()
    };
    let gpu = var("AXON_TF_DEV") == "cuda";
    let g = gpu.then(|| crate::testutil::gpu().unwrap());
    let km = g.as_ref().map(|g| Kernels::module(g).unwrap());
    let cnet = (!gpu).then(|| Qwen2::new(&m));
    let gnet = g.as_ref().map(|g| {
        CudaQwen2::with_kv(g, km.as_ref().unwrap(), &m, max_ctx, CacheKv::F16, false).unwrap()
    });
    let mut out = Vec::new();
    for gn in inp["gens"].as_array().unwrap() {
        let (p, t) = (ids(&gn["prompt_ids"]), ids(&gn["tokens"]));
        let seq: Vec<u32> = p.iter().chain(&t).copied().collect();
        let n = p.len();
        let mut rows = Vec::new();
        if let Some(net) = &cnet {
            let mut c = net.new_cache(seq.len() + 1);
            net.feed_batch(&seq[..n - 1], 0, &mut c);
            for (i, &id) in seq.iter().enumerate().skip(n - 1) {
                rows.push(top(&net.forward(id, i, &mut c)));
            }
        } else {
            let net = gnet.as_ref().unwrap();
            let mut c = net.new_cache(seq.len() + 1);
            net.feed_batch(&seq[..n - 1], 0, &mut c);
            for (i, &id) in seq.iter().enumerate().skip(n - 1) {
                rows.push(top(&net.forward(id, i, &mut c)));
            }
        }
        out.push(rows);
    }
    std::fs::write(var("AXON_TF_OUT"), serde_json::to_string(&out).unwrap()).unwrap();
}

/// hopper.cu gemmh's dynamic shared memory for `wgs` warpgroups
/// (GEMMH_SMEM): two stages of unpacked weights (32 KiB a warpgroup),
/// activation values (16 KiB), raw Q4_K rows, dk / b32 pieces, per-row
/// headers; the barriers; 1 KiB of alignment slack.
fn gemmh_smem(wgs: usize) -> usize {
    2 * (wgs * 32768 + 16384 + wgs * 64 * 144 + 256 + 1024 + wgs * 64 * 16) + 16 + 1024
}

/// hopper.cu gemmh2's dynamic shared memory (GEMMH2_SMEM): 4 stages of
/// activation values (16 KiB) and both warpgroups' raw Q4_K rows, the
/// warpgroups' dk / b32 pieces, the barriers, 1 KiB of alignment slack.
const GEMMH2_SMEM: usize = 4 * 16384 + 4 * 128 * 144 + 4 * 2 * 1280 + 2 * 4 * 8 + 1024;

/// A device buffer's address.
fn dev_addr(b: &DevBuf) -> u64 {
    match b.arg() {
        Arg::Ptr(p) => p,
        _ => unreachable!(),
    }
}

/// Launch hopper.cu's `kernel` (gemmh_q4_k: 2 warpgroups, gemmh1_q4_k: 1,
/// gemmh2_q4_k: 2, register-unpacked) on Q4_K matrix `w` (rows x cols) and `nb` tokens of `act` (stride
/// `st`): TMA maps for both, grid (ceil(nb / 64), ceil(rows / 64 wgs)).
///
/// # Safety
/// `w` holds rows Q4_K rows of cols, `act` nb activations of stride st,
/// `out` (stride `rows`) rows * nb floats, `bias` rows floats or null.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemmh(
    g: &Gpu,
    km: &Module,
    kernel: &str,
    w: &DevBuf,
    rows: usize,
    cols: usize,
    act: &DevBuf,
    st: usize,
    nb: usize,
    bias: Arg,
    mode: i32,
    out: &DevBuf,
) {
    let wgs = if kernel.starts_with("gemmh1") { 1 } else { 2 };
    let rb = (cols / 256 * 144) as u64;
    let mw = g
        .tensor_map_2d(
            dev_addr(w),
            (rb, rows as u64),
            rb,
            (144, 64 * wgs as u32),
            false,
        )
        .unwrap();
    let mx = g
        .tensor_map_2d(
            dev_addr(act),
            (cols as u64, nb as u64),
            st as u64,
            (128, 64),
            true,
        )
        .unwrap();
    let f = km.function(kernel).unwrap();
    let sm = if kernel.starts_with("gemmh2") {
        GEMMH2_SMEM
    } else {
        gemmh_smem(wgs)
    };
    f.set_max_shared(sm).unwrap();
    // SAFETY: gemmh's signature; the caller's buffers.
    unsafe {
        g.launch(
            &f,
            (nb.div_ceil(64) as u32, rows.div_ceil(64 * wgs) as u32, 1),
            (128 * wgs as u32, 1, 1),
            sm as u32,
            &[
                Arg::Blob128(mw),
                Arg::Blob128(mx),
                act.arg(),
                Arg::U64(st as u64),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
                Arg::I32(nb as i32),
                bias,
                Arg::I32(mode),
                out.arg(),
                Arg::U64(rows as u64),
            ],
        )
        .unwrap();
    }
}

/// hopper.cu's wgmma Q4_K GEMMs (gemmh_q4_k, gemmh1_q4_k, gemmh2_q4_k) against
/// gemv_ref bit for bit: row counts around the 64-row tiles and the 7B's,
/// 1 to 74 blocks (odd counts: the two-stage ring's parity), token counts
/// around the 64-token tiles, every epilogue. Hopper (sm_90a) only.
#[test]
fn gpu_gemmh_matches_reference_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module_ext(&g).unwrap();
    if !km.hopper() {
        eprintln!("skipping: needs an sm_90a module (compute capability 9.0)");
        return;
    }
    let mut rng = Rng(0x0090_a0b1_7e57);
    let cases: [(usize, usize, &[usize]); 9] = [
        (64, 256, &[1, 17, 64, 65]),
        (65, 768, &[17, 64, 65, 128]),
        (333, 1536, &[17, 65, 128, 1050]),
        (512, 3584, &[64, 1152]),
        (1001, 4864, &[77, 129]),
        (3584, 3584, &[1050]),
        (18944, 3584, &[129]),
        (3584, 18944, &[65]),
        (127, 18944, &[200]),
    ];
    for (rows, cols, nbs) in cases {
        let wd = synth_matrix(&mut rng, Q4_K, rows, cols);
        let w = g.upload(&wd).unwrap();
        for &nb in nbs {
            let xs: Vec<Vec<f32>> = (0..nb)
                .map(|i| rng.vec(cols, [1.0f32, 1e-3, 40.0][i % 3]))
                .collect();
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
            let act = quantize_batch(&g, &km, &xs, cols);
            let want = gpu_batched(&g, &km, "gemv_ref", Q4_K, &w, rows, cols, &act, nb, &epi);
            for k in ["gemmh_q4_k", "gemmh1_q4_k", "gemmh2_q4_k"] {
                let out = match &epi.init {
                    Some(v) => g.upload(&v.repeat(nb)).unwrap(),
                    None => {
                        let o = g.alloc(rows * nb * 4).unwrap();
                        o.zero().unwrap();
                        o
                    }
                };
                let bias = epi.bias.as_ref().map(|b| g.upload(b).unwrap());
                // SAFETY: w rows x cols Q4_K, act nb tokens, out rows * nb.
                unsafe {
                    launch_gemmh(
                        &g,
                        &km,
                        k,
                        &w,
                        rows,
                        cols,
                        &act,
                        act_stride_t(cols),
                        nb,
                        bias.as_ref().map_or(Arg::Ptr(0), |b| b.arg()),
                        epi.init.is_some() as i32,
                        &out,
                    )
                };
                g.sync().unwrap();
                let got: Vec<f32> = out.read().unwrap();
                if let Some(i) = (0..rows * nb).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                    panic!(
                        "{k} {cols}x{rows} nb {nb} token {} row {}: {} vs ref {}",
                        i / rows,
                        i % rows,
                        got[i],
                        want[i]
                    );
                }
            }
            eprintln!("Q4_K {cols}x{rows} nb {nb}: gemmh bit-identical");
        }
    }
}

/// gemmh vs gemm2 on the 7B's Q4_K shapes: interleaved rounds (clocks
/// drift), each kernel's median / min / max TMAC/s of real MACs (rows x
/// cols x nb). AXON_BENCH_NB (default 1050), AXON_BENCH_ROUNDS (default 7).
/// Hopper only for gemmh; run with --ignored.
#[test]
#[ignore]
fn cuda_gemmh_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module_ext(&g).unwrap();
    let info = g.info().unwrap();
    eprintln!(
        "{} sm_{}{} {} SMs, hopper module {}",
        info.name,
        info.cc.0,
        info.cc.1,
        info.sms,
        km.hopper()
    );
    let nbs: Vec<usize> = std::env::var("AXON_BENCH_NB")
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![1050]);
    let rounds: usize = std::env::var("AXON_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let mut rng = Rng(7);
    for (what, cols, rows) in [
        ("gate  Q4_K", 3584usize, 18944usize),
        ("q/wo  Q4_K", 3584, 3584),
        ("down  Q4_K", 18944, 3584),
        ("kv    Q4_K", 3584, 512),
    ] {
        let w = g.upload(&synth_matrix(&mut rng, Q4_K, rows, cols)).unwrap();
        for &nb in &nbs {
            let xs: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 1.0)).collect();
            let act = quantize_batch(&g, &km, &xs, cols);
            let st = act_stride_t(cols);
            let out = g.alloc(rows * nb * 4).unwrap();
            let mut kernels = vec!["gemm2s_q4_k", "gemm2_q4_k"];
            if km.hopper() {
                kernels.extend(["gemmh_q4_k", "gemmh1_q4_k", "gemmh2_q4_k"]);
            }
            let mut ms: Vec<Vec<f64>> = vec![Vec::new(); kernels.len()];
            for _ in 0..rounds {
                for (ki, k) in kernels.iter().enumerate() {
                    let run = || {
                        if k.starts_with("gemmh") {
                            // SAFETY: as in gpu_gemmh_matches_reference_bits.
                            unsafe {
                                launch_gemmh(
                                    &g,
                                    &km,
                                    k,
                                    &w,
                                    rows,
                                    cols,
                                    &act,
                                    st,
                                    nb,
                                    Arg::Ptr(0),
                                    0,
                                    &out,
                                )
                            };
                        } else {
                            let f = km.function(k).unwrap();
                            let common = [
                                w.arg(),
                                Arg::U64(quant::row_bytes(Q4_K, cols) as u64),
                                Arg::I32(rows as i32),
                                Arg::I32(cols as i32),
                                act.arg(),
                                Arg::U64(st as u64),
                                Arg::I32(nb as i32),
                                Arg::Ptr(0),
                                Arg::I32(0),
                                out.arg(),
                                Arg::U64(rows as u64),
                            ];
                            // SAFETY: prefill.cu's gemm2 signature; buffers as above.
                            unsafe { launch_product(&g, &f, k, Q4_K, rows, nb, &common) };
                        }
                    };
                    run(); // warm (and the smem attribute)
                    let e0 = g.record().unwrap();
                    for _ in 0..3 {
                        run();
                    }
                    let e1 = g.record().unwrap();
                    ms[ki].push(e0.ms_to(&e1).unwrap() as f64 / 3.0);
                }
            }
            let macs = (rows * cols * nb) as f64;
            for (ki, k) in kernels.iter().enumerate() {
                let v = &mut ms[ki];
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let t = |x: f64| macs / (x * 1e-3) / 1e12;
                eprintln!(
                    "{what} {cols}x{rows} nb {nb} {k}: median {:.1} TMAC/s (min {:.1} max {:.1}; {:.3} ms)",
                    t(v[v.len() / 2]),
                    t(v[v.len() - 1]),
                    t(v[0]),
                    v[v.len() / 2]
                );
            }
        }
    }
    // The fused gate + up (7B: 2 x 18944 x 3584): gemm2s_q4_k_gu vs gemmh2_q4_k_gu.
    let (cols, rows) = (3584usize, 18944usize);
    let wg = g.upload(&synth_matrix(&mut rng, Q4_K, rows, cols)).unwrap();
    let wu = g.upload(&synth_matrix(&mut rng, Q4_K, rows, cols)).unwrap();
    for &nb in &nbs {
        let xs: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 1.0)).collect();
        let act = quantize_batch(&g, &km, &xs, cols);
        let st = act_stride_t(cols);
        let out = g.alloc(rows * nb * 4).unwrap();
        let mut kernels = vec!["gemm2s_q4_k_gu"];
        if km.hopper() {
            kernels.push("gemmh2_q4_k_gu");
        }
        let f2 = km.function("gemm2s_q4_k_gu").unwrap();
        f2.set_max_shared(gemm2_smem(Q4_K, 64, 3)).unwrap();
        let mut ms: Vec<Vec<f64>> = vec![Vec::new(); kernels.len()];
        for _ in 0..rounds {
            for (ki, k) in kernels.iter().enumerate() {
                let run = || {
                    if k.starts_with("gemmh") {
                        // SAFETY: as in gpu_gemmh_gate_up_match_reference.
                        unsafe {
                            launch_gemmh_gu(&g, &km, &wg, &wu, rows, cols, &act, st, nb, &out)
                        };
                    } else {
                        // SAFETY: gemm2s_q4_k_gu's signature; buffers as above.
                        unsafe {
                            g.launch(
                                &f2,
                                (nb.div_ceil(64) as u32, rows.div_ceil(32) as u32, 1),
                                (256, 1, 1),
                                gemm2_smem(Q4_K, 64, 3) as u32,
                                &[
                                    wg.arg(),
                                    wu.arg(),
                                    Arg::U64(quant::row_bytes(Q4_K, cols) as u64),
                                    Arg::I32(rows as i32),
                                    Arg::I32(cols as i32),
                                    act.arg(),
                                    Arg::U64(st as u64),
                                    Arg::I32(nb as i32),
                                    out.arg(),
                                    Arg::U64(rows as u64),
                                ],
                            )
                            .unwrap()
                        };
                    }
                };
                run();
                let e0 = g.record().unwrap();
                for _ in 0..3 {
                    run();
                }
                let e1 = g.record().unwrap();
                ms[ki].push(e0.ms_to(&e1).unwrap() as f64 / 3.0);
            }
        }
        let macs = (2 * rows * cols * nb) as f64;
        for (ki, k) in kernels.iter().enumerate() {
            let v = &mut ms[ki];
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let t = |x: f64| macs / (x * 1e-3) / 1e12;
            eprintln!(
                "gate+up {cols}x2x{rows} nb {nb} {k}: median {:.1} TMAC/s (min {:.1} max {:.1}; {:.3} ms)",
                t(v[v.len() / 2]),
                t(v[v.len() - 1]),
                t(v[0]),
                v[v.len() / 2]
            );
        }
    }
}

/// hopper.cu's gemmh kernels in mode 2 (out = silu(gate) * sum) and
/// gemmh2_q4_k_gu (gate and up in one launch) against gemv_ref: gate by
/// gemv_ref (mode 0), then up by gemv_ref in mode 2 on it. Hopper only.
#[test]
fn gpu_gemmh_gate_up_match_reference() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    if !km.hopper() {
        eprintln!("skipping: needs an sm_90a module (compute capability 9.0)");
        return;
    }
    let mut rng = Rng(0x5eed_9a7e_0090);
    for (cols, rows) in [
        (256usize, 333usize),
        (1536, 129),
        (3584, 1000),
        (3584, 18944),
    ] {
        let gd = synth_matrix(&mut rng, Q4_K, rows, cols);
        let ud = synth_matrix(&mut rng, Q4_K, rows, cols);
        let (wg, wu) = (g.upload(&gd).unwrap(), g.upload(&ud).unwrap());
        let nbs: &[usize] = if rows > 10000 {
            &[1050]
        } else {
            &[17, 64, 100, 129]
        };
        for &nb in nbs {
            let xs: Vec<Vec<f32>> = (0..nb)
                .map(|i| rng.vec(cols, [1.0f32, 1e-2, 4.0][i % 3]))
                .collect();
            let act = quantize_batch(&g, &km, &xs, cols);
            let st = act_stride_t(cols);
            let reference = |w: &DevBuf, bias: Arg, mode: i32| -> DevBuf {
                let out = g.alloc(rows * nb * 4).unwrap();
                out.zero().unwrap();
                let f = km.function("gemv_ref").unwrap();
                let common = [
                    w.arg(),
                    Arg::U64(quant::row_bytes(Q4_K, cols) as u64),
                    Arg::I32(rows as i32),
                    Arg::I32(cols as i32),
                    act.arg(),
                    Arg::U64(st as u64),
                    Arg::I32(nb as i32),
                    bias,
                    Arg::I32(mode),
                    out.arg(),
                    Arg::U64(rows as u64),
                ];
                // SAFETY: gemv_ref's signature; act nb tokens, out rows * nb.
                unsafe { launch_product(&g, &f, "gemv_ref", Q4_K, rows, nb, &common) };
                g.sync().unwrap();
                out
            };
            let gate = reference(&wg, Arg::Ptr(0), 0);
            let want: Vec<f32> = reference(&wu, gate.arg(), 2).read().unwrap();
            for k in ["gemmh_q4_k", "gemmh1_q4_k", "gemmh2_q4_k"] {
                let out = g.alloc(rows * nb * 4).unwrap();
                // SAFETY: wu rows x cols Q4_K, act nb tokens, out and the
                // gate rows * nb floats.
                unsafe {
                    launch_gemmh(
                        &g,
                        &km,
                        k,
                        &wu,
                        rows,
                        cols,
                        &act,
                        st,
                        nb,
                        gate.arg(),
                        2,
                        &out,
                    )
                };
                g.sync().unwrap();
                let got: Vec<f32> = out.read().unwrap();
                assert_eq!(bits(&got), bits(&want), "mode 2 {k} {cols}x{rows} nb {nb}");
            }
            let out = g.alloc(rows * nb * 4).unwrap();
            // SAFETY: gate / up rows x cols Q4_K, act nb tokens, out rows * nb.
            unsafe { launch_gemmh_gu(&g, &km, &wg, &wu, rows, cols, &act, st, nb, &out) };
            g.sync().unwrap();
            let got: Vec<f32> = out.read().unwrap();
            assert_eq!(
                bits(&got),
                bits(&want),
                "gemmh2_q4_k_gu {cols}x{rows} nb {nb}"
            );
        }
        eprintln!("Q4_K {cols} x {rows}: gemmh mode 2 and gemmh2_q4_k_gu == gemv_ref");
    }
}

/// Launch gemmh2_q4_k_gu: out = silu(wg x) * (wu x), rows x nb.
///
/// # Safety
/// As `launch_gemmh`, for both matrices.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemmh_gu(
    g: &Gpu,
    km: &Module,
    wg: &DevBuf,
    wu: &DevBuf,
    rows: usize,
    cols: usize,
    act: &DevBuf,
    st: usize,
    nb: usize,
    out: &DevBuf,
) {
    let rb = (cols / 256 * 144) as u64;
    let map = |w: &DevBuf| {
        g.tensor_map_2d(dev_addr(w), (rb, rows as u64), rb, (144, 64), false)
            .unwrap()
    };
    let (mg, mu) = (map(wg), map(wu));
    let mx = g
        .tensor_map_2d(
            dev_addr(act),
            (cols as u64, nb as u64),
            st as u64,
            (128, 64),
            true,
        )
        .unwrap();
    let f = km.function("gemmh2_q4_k_gu").unwrap();
    f.set_max_shared(GEMMH2_SMEM).unwrap();
    // SAFETY: gemmh2_q4_k_gu's signature; the caller's buffers.
    unsafe {
        g.launch(
            &f,
            (nb.div_ceil(64) as u32, rows.div_ceil(64) as u32, 1),
            (256, 1, 1),
            GEMMH2_SMEM as u32,
            &[
                Arg::Blob128(mg),
                Arg::Blob128(mu),
                Arg::Blob128(mx),
                act.arg(),
                Arg::U64(st as u64),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
                Arg::I32(nb as i32),
                out.arg(),
                Arg::U64(rows as u64),
            ],
        )
        .unwrap();
    }
}

/// The default Hopper routing of plain Q4_K products (batch::gemmh_plain):
/// the 7B's q / wo (3584 rows) and gate / down shapes at prefill batch
/// sizes, never the small-grid shapes (512-row k / v at nb 512 measured
/// slower on an H100 PCIe, 114 SMs).
#[test]
fn gemmh_plain_routing() {
    use batch::gemmh_plain;
    for sms in [114, 132] {
        assert!(gemmh_plain(3584, 1050, sms));
        assert!(gemmh_plain(3584, 512, sms)); // 8 x 28 = 224 blocks
        assert!(gemmh_plain(18944, 1050, sms));
        assert!(gemmh_plain(18944, 64, sms)); // 148
        assert!(!gemmh_plain(3584, 256, sms)); // 112
        assert!(!gemmh_plain(3584, 64, sms));
        assert!(!gemmh_plain(512, 512, sms));
        assert!(!gemmh_plain(512, 1050, sms));
        assert!(!gemmh_plain(896, 1050, sms));
    }
}

/// Gemma 3 1B Q8_0 (head size 256, 4 query heads on 1, sliding window 512,
/// tied Q8_0 head of 262144 rows), mapped; and the 4B Q4_K_M if present
/// (8 on 4, window 1024, linear RoPE scaling 8 on the global layers).
fn gemma3_mapped() -> Option<Model> {
    crate::testutil::gemma3_model_path()?;
    model_at(crate::testutil::GEMMA3_MODEL)
}
const GEMMA3_4B: &str = "models/google_gemma-3-4b-it-Q4_K_M.gguf";
const GEMMA3_1B_Q4: &str = "models/google_gemma-3-1b-it-Q4_K_M.gguf";

/// The batching contract on gemma3: 1200 tokens cross the 1B's 512 window
/// (and the 4B's 1024), so batched passes and decode steps both run the
/// windowed attention, the post-norms, GeGLU and the two RoPE tables.
#[test]
fn gpu_gemma3_prefill_and_verify_bits() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = gemma3_mapped() else {
        return;
    };
    prefill_bits(&m, "gemma3 1B", CacheKv::F16);
    verify_bits(&m, "gemma3 1B", CacheKv::F16);
    if let Some(mq) = model_at(GEMMA3_1B_Q4) {
        prefill_bits(&mq, "gemma3 1B Q4_K_M", CacheKv::F16);
        verify_bits(&mq, "gemma3 1B Q4_K_M", CacheKv::F16);
    }
    if let Some(m4) = model_at(GEMMA3_4B) {
        prefill_bits(&m4, "gemma3 4B", CacheKv::F16);
        verify_bits(&m4, "gemma3 4B", CacheKv::F16);
    }
}

/// `assert_graded` for gemma3, whose CPU disagrees with itself more than
/// Qwen's does (1B Q8_0, sm_120 and sm_89: CPU per-row vs repacked max
/// |dlogit| 7.3, a top-1 flip at margin 1.6): the same top-1, flip-margin
/// (all below max(0.5, the CPU's own spread), at most max(2, 1%) at 0.5 or
/// more), mean-KL, max-KL and |dlogit| bounds. A position with KL above 0.5
/// passes if the reference's top-2 margin there is below that tie bound (a
/// near-tie either side may resolve), or else if `ref_ok(index)` holds: the
/// independent f64 `gemma3_ref` on that prefix is at least as close to the
/// GPU as (twice) to the CPU (`gemma3_ref_closer`).
fn assert_graded_gemma(ag: &Agree, ac: &Agree, ref_ok: &dyn Fn(usize) -> bool) {
    let n = ag.positions;
    assert!(
        ag.top1 * 100 + 2 * n >= ac.top1 * 100 && ag.top1 * 100 >= 94 * n,
        "top-1 agreement {}/{n} (cpu vs itself {})",
        ag.top1,
        ac.top1
    );
    let tie = ac.max_abs.max(0.5);
    let wide = ag.flips.iter().filter(|&&d| d >= 0.5).count();
    assert!(
        ag.flips.iter().all(|&d| d < tie) && wide <= (n / 100).max(2),
        "a top-1 flip with margin >= {tie} or {wide} >= 0.5: {:?}",
        ag.flips
    );
    assert!(ag.sum_kl <= 2.0 * ac.sum_kl, "mean KL");
    assert!(ag.max_kl < 2.0, "max KL {}", ag.max_kl);
    for &(i, k, mg, _) in &ag.over_half {
        assert!(
            mg < tie || ref_ok(i),
            "KL {k} at position {i} (reference margin {mg} >= {tie}), and the f64 reference \
             isn't as close to the GPU as to the CPU there"
        );
    }
    assert!(ag.max_abs < 6.0, "max logit difference");
}

/// KL(p || q) of f64 logits p and f32 logits q.
fn kl64(p: &[f64], q: &[f32]) -> f64 {
    let lse = |v: &mut dyn Iterator<Item = f64>| -> Vec<f64> {
        let v: Vec<f64> = v.collect();
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z = v.iter().map(|x| (x - mx).exp()).sum::<f64>().ln() + mx;
        v.iter().map(|x| x - z).collect()
    };
    let lp = lse(&mut p.iter().copied());
    let lq = lse(&mut q.iter().map(|&x| x as f64));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
}

/// At `prefix` (the last token included), whether the f64 `gemma3_ref`
/// next-token distribution is as close to the GPU's as to the CPU's:
/// KL(ref || gpu) <= max(0.5, 2 KL(ref || cpu)). Fresh caches (both
/// backends' batched prefill equals their decode bit for bit).
fn gemma3_ref_closer(m: &Model, cpu: &Qwen2, gpu: &CudaQwen2, prefix: &[u32]) -> bool {
    let n = prefix.len();
    let r = &crate::model::gemma3_tests::gemma3_ref(m, prefix, &[n - 1])[0];
    let mut cc = cpu.new_cache(n);
    cpu.feed_batch(&prefix[..n - 1], 0, &mut cc);
    let lc = cpu.forward(prefix[n - 1], n - 1, &mut cc);
    let mut gc = gpu.new_cache(n);
    gpu.feed_batch(&prefix[..n - 1], 0, &mut gc);
    let lg = gpu.forward(prefix[n - 1], n - 1, &mut gc);
    let (kg, kc) = (kl64(r, &lg), kl64(r, &lc));
    eprintln!("gemma3 ref at {n} tokens: KL(ref||gpu) {kg:.4}, KL(ref||cpu) {kc:.4}");
    kg <= (2.0 * kc).max(0.5)
}

/// `continuations_with`'s walk, returning every divergence's reference
/// margin: (identical, matched tokens, margins).
fn continuation_margins<R: Lm, O: Lm>(
    r: &R,
    o: &O,
    tok: &crate::tokenizer::Tokenizer,
) -> (usize, usize, Vec<f32>) {
    let (mut same, mut matched, mut margins) = (0, 0, Vec::new());
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
                margins.push(lr[i][tr[i] as usize] - lr[i][to[i] as usize]);
            }
        }
    }
    (same, matched, margins)
}

/// One gemma3 file: GPU vs CPU logits graded (`assert_graded_gemma`, with
/// the f64 reference judging positions of KL above 0.5 that aren't
/// near-ties), and greedy continuations whose divergences all have a
/// reference margin below max(0.5, the largest the CPU's own kernel sets
/// diverge at).
fn gemma3_vs_cpu(m: &Model, name: &str) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module_for(&g, m).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let gpu = CudaQwen2::new(&g, &km, m, 256).unwrap();
    let rows = Qwen2::new_with(m, false);
    let cpu = Qwen2::new(m);
    const STEPS: usize = 32;
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    let mut seqs = Vec::new();
    for p in PROMPTS {
        let ids = tok.encode(p);
        compare(&cpu, &gpu, &ids, STEPS, &mut ag);
        compare(&cpu, &rows, &ids, STEPS, &mut ac);
        // compare's position s of this prompt follows ids + the CPU's
        // first s greedy picks.
        let (picks, _) = greedy(&cpu, &ids, STEPS, false);
        seqs.push((ids, picks));
    }
    report(&format!("{name} gpu vs cpu"), &ag);
    report(&format!("{name} cpu rows vs cpu"), &ac);
    let ref_ok = |i: usize| {
        let (k, s) = ((i - 1) / (STEPS + 1), (i - 1) % (STEPS + 1));
        let (ids, picks) = &seqs[k];
        let prefix: Vec<u32> = ids.iter().chain(&picks[..s]).copied().collect();
        gemma3_ref_closer(m, &cpu, &gpu, &prefix)
    };
    assert_graded_gemma(&ag, &ac, &ref_ok);
    let (cs, cm, cmar) = continuation_margins(&cpu, &rows, &tok);
    let (gs, gm, gmar) = continuation_margins(&cpu, &gpu, &tok);
    let tie = cmar.iter().cloned().fold(0.5f32, f32::max);
    eprintln!(
        "{name} continuations: gpu {gs}/{} identical, {gm} tokens, margins {gmar:?}; cpu rows \
         {cs}, {cm}, margins {cmar:?} (tie {tie})",
        PROMPTS.len()
    );
    assert!(
        gmar.iter().all(|&d| d < tie),
        "not a near-tie: {gmar:?} (tie {tie})"
    );
    assert!(
        gs + 2 >= cs,
        "identical continuations: gpu {gs}, cpu rows {cs}"
    );
    assert!(2 * gm + 32 >= cm, "matched tokens: gpu {gm}, cpu rows {cm}");
}

/// Diagnostic (ignored): the 1B Q4_K_M, GPU vs CPU over PROMPTS
/// as `gemma3_vs_cpu`; at every top-1 flip, the f64 `gemma3_ref` there:
/// its top two, KL(ref || gpu), KL(ref || cpu), KL(ref || cpu rows).
#[test]
#[ignore]
fn gemma3_flip_probe() {
    use crate::model::gemma3_tests::gemma3_ref;
    let g = crate::testutil::gpu().unwrap();
    let m = model_at(GEMMA3_1B_Q4).unwrap();
    let km = Kernels::module_for(&g, &m).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let gpu = CudaQwen2::new(&g, &km, &m, 256).unwrap();
    let rows = Qwen2::new_with(&m, false);
    let cpu = Qwen2::new(&m);
    const STEPS: usize = 32;
    let mut ag = Agree::default();
    let mut seqs = Vec::new();
    for p in PROMPTS {
        let ids = tok.encode(p);
        compare(&cpu, &gpu, &ids, STEPS, &mut ag);
        let (picks, _) = greedy(&cpu, &ids, STEPS, false);
        seqs.push((ids, picks));
    }
    report("probe gpu vs cpu", &ag);
    let fresh = |net: &dyn Fn(&[u32]) -> Vec<f32>, pre: &[u32]| net(pre);
    for &i in &ag.flip_at {
        let (k, st) = ((i - 1) / (STEPS + 1), (i - 1) % (STEPS + 1));
        let (ids, picks) = &seqs[k];
        let pre: Vec<u32> = ids.iter().chain(&picks[..st]).copied().collect();
        let n = pre.len();
        let r = &gemma3_ref(&m, &pre, &[n - 1])[0];
        let run_cpu = |net: &Qwen2| {
            let mut c = net.new_cache(n);
            net.feed_batch(&pre[..n - 1], 0, &mut c);
            net.forward(pre[n - 1], n - 1, &mut c)
        };
        let lc = run_cpu(&cpu);
        let lr = run_cpu(&rows);
        let lg = fresh(
            &|p: &[u32]| {
                let mut c = gpu.new_cache(n);
                gpu.feed_batch(&p[..n - 1], 0, &mut c);
                gpu.forward(p[n - 1], n - 1, &mut c)
            },
            &pre,
        );
        let r32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        let (rt, gt, ct) = (crate::argmax(&r32), crate::argmax(&lg), crate::argmax(&lc));
        eprintln!(
            "flip at {i} (prompt {k} step {st}): ref top {rt} margin(ref top - gpu top) {:.3}, \
             (ref top - cpu top) {:.3}; KL(ref||gpu) {:.4} KL(ref||cpu) {:.4} KL(ref||rows) {:.4}; \
             cpu margin {:.3} gpu margin {:.3}",
            r[rt as usize] - r[gt as usize],
            r[rt as usize] - r[ct as usize],
            kl64(r, &lg),
            kl64(r, &lc),
            kl64(r, &lr),
            lc[ct as usize] - lc[gt as usize],
            lg[gt as usize] - lg[ct as usize],
        );
    }
}

/// Every position of `gemma3_vs_cpu`'s walk (PROMPTS, the CPU's greedy
/// picks fed to both) against the f64 `gemma3_ref` on the same tokens:
/// per backend the mean and max KL(ref || backend) and how often its top-1
/// is the reference's. (gpu, cpu, cpu rows) in that order.
fn gemma3_ref_walk(
    m: &Model,
    gpu: &CudaQwen2,
    cpu: &Qwen2,
    rows: &Qwen2,
    steps: usize,
) -> [(f64, f64, usize, usize); 3] {
    use crate::model::gemma3_tests::{gemma3_ref, top1};
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let mut acc = [(0f64, 0f64, 0usize, 0usize); 3];
    for p in PROMPTS {
        let ids = tok.encode(p);
        let (picks, _) = greedy(cpu, &ids, steps, false);
        let seq: Vec<u32> = ids.iter().chain(&picks).copied().collect();
        let first = ids.len() - 1;
        let at: Vec<usize> = (first..seq.len()).collect();
        let r = gemma3_ref(m, &seq, &at);
        let walk = |lm: &dyn Fn(u32, usize) -> Vec<f32>| -> Vec<Vec<f32>> {
            at.iter().map(|&p| lm(seq[p], p)).collect()
        };
        let mut gc = gpu.new_cache(seq.len());
        gpu.feed_batch(&seq[..first], 0, &mut gc);
        let gl = {
            let gc = std::cell::RefCell::new(gc);
            walk(&|t, p| gpu.forward(t, p, &mut gc.borrow_mut()))
        };
        let run = |net: &Qwen2| {
            let c = std::cell::RefCell::new(net.new_cache(seq.len()));
            net.feed_batch(&seq[..first], 0, &mut c.borrow_mut());
            walk(&|t, p| net.forward(t, p, &mut c.borrow_mut()))
        };
        let (cl, rl) = (run(cpu), run(rows));
        for (b, ls) in [gl, cl, rl].iter().enumerate() {
            for (l, rr) in ls.iter().zip(&r) {
                let k = kl64(rr, l);
                let l64: Vec<f64> = l.iter().map(|&v| v as f64).collect();
                acc[b].0 += k;
                acc[b].1 = acc[b].1.max(k);
                acc[b].2 += (top1(&l64) == top1(rr)) as usize;
                acc[b].3 += 1;
            }
        }
    }
    for a in &mut acc {
        a.0 /= a.3 as f64;
    }
    acc
}

/// Diagnostic (ignored): `gemma3_ref_walk` on the 1B Q8_0 and Q4_K_M.
#[test]
#[ignore]
fn gemma3_ref_walk_probe() {
    let g = crate::testutil::gpu().unwrap();
    for path in [crate::testutil::GEMMA3_MODEL, GEMMA3_1B_Q4] {
        let Some(m) = model_at(path) else { continue };
        let km = Kernels::module_for(&g, &m).unwrap();
        let gpu = CudaQwen2::new(&g, &km, &m, 256).unwrap();
        let rows = Qwen2::new_with(&m, false);
        let cpu = Qwen2::new(&m);
        let a = gemma3_ref_walk(&m, &gpu, &cpu, &rows, 32);
        for (name, (mean, mx, top, n)) in ["gpu", "cpu", "cpu rows"].iter().zip(a) {
            eprintln!("{path} {name} vs f64 ref: mean KL {mean:.4} max {mx:.4}, top-1 {top}/{n}");
        }
    }
}

/// Diagnostic (ignored, CPU only): the 1B Q4_K_M at AXON_PROBE's
/// "prompt:step,..." positions of `gemma3_flip_probe`'s walk: KL(ref ||
/// cpu) and KL(ref || cpu rows) against the f64 `gemma3_ref`.
#[test]
#[ignore]
fn gemma3_cpu_probe() {
    use crate::model::gemma3_tests::gemma3_ref;
    let m = model_at(GEMMA3_1B_Q4).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let rows = Qwen2::new_with(&m, false);
    let cpu = Qwen2::new(&m);
    for spec in std::env::var("AXON_PROBE").unwrap().split(',') {
        let (k, st) = spec.split_once(':').unwrap();
        let (k, st): (usize, usize) = (k.parse().unwrap(), st.parse().unwrap());
        let ids = tok.encode(PROMPTS[k]);
        let (picks, _) = greedy(&cpu, &ids, 32, false);
        let pre: Vec<u32> = ids.iter().chain(&picks[..st]).copied().collect();
        let n = pre.len();
        let r = &gemma3_ref(&m, &pre, &[n - 1])[0];
        let run = |net: &Qwen2| {
            let mut c = net.new_cache(n);
            net.feed_batch(&pre[..n - 1], 0, &mut c);
            net.forward(pre[n - 1], n - 1, &mut c)
        };
        let (lc, lr) = (run(&cpu), run(&rows));
        eprintln!(
            "probe {k}:{st} ({n} tokens): KL(ref||cpu) {:.4} KL(ref||rows) {:.4}",
            kl64(r, &lc),
            kl64(r, &lr)
        );
    }
}

/// A gemma3 file graded against the independent f64 `gemma3_ref` instead
/// of the CPU (`gemma3_ref_walk`): the GPU's top-1 agrees with the
/// reference within 2% of positions of the CPU's own agreement, its mean
/// KL(ref || gpu) is at most twice the CPU's, and no position is above 2.
/// For the 1B Q4_K_M, whose Q4_K / Q6_K products take Q8_K activations (one
/// scale per 256 values; Gemma's residual stream has large outliers), the
/// CPU itself is far from the reference at some positions (KL up to 0.53;
/// per-row vs repacked CPU kernels 0.47), so GPU vs CPU flips there say
/// little. Measured (sm_61 jade, same numbers as sm_89): top-1 vs ref gpu
/// 191/198, cpu 191, cpu rows 192; mean KL 0.0201 / 0.0151 / 0.0124; max
/// 0.86 / 0.53 / 0.47. (The 1B Q8_0, same walk: 194 / 195 / 193, mean
/// 0.0093 / 0.0103 / 0.0138.) A decode-only SiLU on these narrow rows
/// (#30's gate_up4) is caught by gpu_gemma3_prefill_and_verify_bits on
/// this file: the batched FFN has no gate_up4.
fn gemma3_vs_ref(m: &Model, name: &str) {
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module_for(&g, m).unwrap();
    let gpu = CudaQwen2::new(&g, &km, m, 256).unwrap();
    let rows = Qwen2::new_with(m, false);
    let cpu = Qwen2::new(m);
    let [ag, ac, ar] = gemma3_ref_walk(m, &gpu, &cpu, &rows, 32);
    eprintln!(
        "{name} vs f64 ref: gpu mean KL {:.4} max {:.4} top-1 {}/{}; cpu {:.4} {:.4} {}; cpu rows \
         {:.4} {:.4} {}",
        ag.0, ag.1, ag.2, ag.3, ac.0, ac.1, ac.2, ar.0, ar.1, ar.2
    );
    let n = ag.3;
    assert!(
        ag.2 * 100 + 2 * n >= ac.2 * 100,
        "top-1 vs reference: gpu {} cpu {}",
        ag.2,
        ac.2
    );
    assert!(
        ag.0 <= 2.0 * ac.0,
        "mean KL vs reference: gpu {} cpu {}",
        ag.0,
        ac.0
    );
    assert!(ag.1 < 2.0, "max KL vs reference {}", ag.1);
}

/// gemma3 GPU vs CPU (`gemma3_vs_cpu`) on the 1B Q8_0, the 1B Q4_K_M
/// against the f64 reference (`gemma3_vs_ref`; its narrow gate/up rows go
/// through the gate/up selection #30's gate_up4 sits in), then past the
/// 1B's 512 window (an 1100-token prompt, as
/// gpu_logits_match_cpu_long_context).
#[test]
fn gpu_gemma3_logits_match_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = gemma3_mapped() else {
        return;
    };
    gemma3_vs_cpu(&m, "gemma3 1B Q8_0");
    if let Some(mq) = model_at(GEMMA3_1B_Q4) {
        gemma3_vs_ref(&mq, "gemma3 1B Q4_K_M");
    }
    let km = Kernels::module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let rows = Qwen2::new_with(&m, false);
    let cpu = Qwen2::new(&m);
    let gpu = CudaQwen2::new(&g, &km, &m, 1300).unwrap();
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
    report("gemma3 1B long gpu vs cpu", &ag);
    report("gemma3 1B long cpu rows vs cpu", &ac);
    // (A reference pass over 1100 tokens is too slow: positions above 0.5
    // here must be near-ties.)
    assert_graded_gemma(&ag, &ac, &|_| false);
}

/// The GPU's gemma3 (decode with and without graphs, batched prefill)
/// against model.rs's independent f64 `gemma3_ref`, with the file's window
/// and with a window of 6 (shorter than the prompt): max |dlogit| within
/// the CPU's bound (1.0; the CPU measures ~0.55 there) and the same top
/// token.
#[test]
fn gpu_gemma3_matches_reference_forward() {
    use crate::model::gemma3_tests::{gemma3_ref, top1};
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(path) = crate::testutil::gemma3_model_path() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let ids: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563, 9079, 236761, 669, 5279, 529];
    let n = ids.len();
    for window in [None, Some(6u32)] {
        let mut m = crate::gguf::load_with(path.to_str().unwrap(), Some(false)).unwrap();
        if let Some(w) = window {
            for (k, v) in m.metadata.iter_mut() {
                if k == "gemma3.attention.sliding_window" {
                    *v = crate::gguf::Val::U32(w);
                }
            }
        }
        let want = gemma3_ref(&m, &ids, &[n - 1]);
        let mut net = CudaQwen2::new(&g, &km, &m, 16).unwrap();
        for graphs in [true, false] {
            net.graphs = graphs;
            let mut c = net.new_cache(n);
            let mut dec = Vec::new();
            for (p, &id) in ids.iter().enumerate() {
                dec = net.forward(id, p, &mut c);
            }
            let mut c = net.new_cache(n);
            net.feed_batch(&ids[..n - 1], 0, &mut c);
            let pre = net.forward(ids[n - 1], n - 1, &mut c);
            assert_eq!(
                bits(&pre),
                bits(&dec),
                "window {window:?}: prefill != decode"
            );
            let d = dec
                .iter()
                .zip(&want[0])
                .fold(0f64, |mx, (&a, &b)| mx.max((a as f64 - b).abs()));
            let d64: Vec<f64> = dec.iter().map(|&v| v as f64).collect();
            eprintln!(
                "gemma3 gpu window {window:?} graphs {graphs} vs reference: max |dlogit| {d:.4}"
            );
            assert!(d < 1.0, "window {window:?}: {d}");
            assert_eq!(top1(&d64), top1(&want[0]), "window {window:?}: top token");
        }
    }
}

/// Decode tok/s of greedy steps (forward_argmax, as `gen`) at short or long
/// positions in a cache of any size, which `gen` (cache = prompt + n) can't
/// make. A benchmark: `--ignored`, the model from AXON_TEST_MODEL;
/// AXON_TG_CTX (32768), AXON_TG_POS (8,500: the first step's position),
/// AXON_TG_N (128 steps), AXON_TG_KV=q8.
#[test]
#[ignore]
fn cuda_decode_tg() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let ctx: usize = env("AXON_TG_CTX", "32768").parse().unwrap();
    let n: usize = env("AXON_TG_N", "128").parse().unwrap();
    let poss: Vec<usize> = env("AXON_TG_POS", "8,500")
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect();
    let path = crate::testutil::model_path().unwrap();
    let m = crate::gguf::load_with(path.to_str().unwrap(), Some(false)).unwrap();
    let km = Kernels::module(&g).unwrap();
    let mut net = CudaQwen2::new(&g, &km, &m, ctx).unwrap();
    if env("AXON_TG_KV", "f16") == "q8" {
        net.kv = CacheKv::Q8 { group: 0 };
    }
    let mut c = net.new_cache(ctx);
    let mut rng = Rng(77);
    let mut pos = 0;
    for p in poss {
        assert!(p >= pos && p + n + 1 < ctx, "AXON_TG_POS {p}");
        if p > pos {
            let ids: Vec<u32> = (pos..p).map(|_| (rng.next() % 20000) as u32).collect();
            net.feed_batch(&ids, pos, &mut c);
        }
        // One step outside the timing (graph capture), then n.
        let mut t = net.forward_argmax(100, p, &mut c);
        let t0 = std::time::Instant::now();
        for i in 1..=n {
            t = net.forward_argmax(t, p + i, &mut c);
        }
        let s = t0.elapsed().as_secs_f64();
        eprintln!(
            "TG ctx {ctx} pos {p} kv {:?}: {n} steps in {:.4} s ({:.2} tok/s)",
            net.kv,
            s,
            n as f64 / s
        );
        pos = p + n + 1;
    }
}
