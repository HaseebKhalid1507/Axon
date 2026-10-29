use super::*;
use crate::cuda::Kernels;
use crate::model::Qwen2;
use std::path::PathBuf;

mod twin;

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

/// The reference's half of `compare`, run once: its logits after the
/// prompt and after each of `steps` greedy tokens of its own (exactly the
/// calls `compare` makes on it), so several models can be graded against
/// one reference pass (`compare_trace`) instead of recomputing it per model.
pub(crate) struct Trace {
    ids: Vec<u32>,
    steps: usize,
    /// (token fed at position rest.len() + s, the reference's logits there).
    at: Vec<(u32, Vec<f32>)>,
}

impl Trace {
    /// The reference's greedy picks: the tokens fed after the prompt's last
    /// (`steps` of them).
    pub(crate) fn picks(&self) -> Vec<u32> {
        self.at[1..].iter().map(|(t, _)| *t).collect()
    }
}

pub(crate) fn trace<R: Lm>(r: &R, ids: &[u32], steps: usize) -> Trace {
    let mut rc = r.new_cache(ids.len() + steps + 1);
    let (&last, rest) = ids.split_last().unwrap();
    r.feed_batch(rest, 0, &mut rc);
    let mut tok = last;
    let mut at = Vec::with_capacity(steps + 1);
    for s in 0..=steps {
        let lr = r.forward(tok, rest.len() + s, &mut rc);
        let next = crate::argmax(&lr);
        at.push((tok, lr));
        tok = next;
    }
    Trace {
        ids: ids.to_vec(),
        steps,
        at,
    }
}

/// `compare(r, o, ids, steps, a)` with `t = trace(r, ids, steps)`: the same
/// calls on `o`, the same tally.
pub(crate) fn compare_trace<O: Lm>(t: &Trace, o: &O, a: &mut Agree) {
    let mut oc = o.new_cache(t.ids.len() + t.steps + 1);
    let rest = &t.ids[..t.ids.len() - 1];
    o.feed_batch(rest, 0, &mut oc);
    a.last_kl = 0.0;
    for (s, (tok, lr)) in t.at.iter().enumerate() {
        let lo = o.forward(*tok, rest.len() + s, &mut oc);
        a.tally(lr, &lo);
    }
}

/// `compare` against several models at once: `trace` once, then
/// `compare_trace` for each (model, tally) pair.
macro_rules! compare_each {
    ($r:expr, $ids:expr, $steps:expr, $(($o:expr, $a:expr)),+ $(,)?) => {{
        let t = $crate::cuda::qwen2::tests::trace($r, $ids, $steps);
        $($crate::cuda::qwen2::tests::compare_trace(&t, $o, $a);)+
    }};
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
        a.tally(&lr, &lo);
        tok = crate::argmax(&lr);
    }
}

impl Agree {
    /// One position: reference logits `lr`, the other's `lo`.
    fn tally(&mut self, lr: &[f32], lo: &[f32]) {
        let (ar, ao) = (crate::argmax(lr), crate::argmax(lo));
        self.positions += 1;
        self.top1 += (ar == ao) as usize;
        let k = kl(lr, lo);
        self.max_kl = self.max_kl.max(k);
        if k > 0.5 {
            self.over_half
                .push((self.positions, k, top2_margin(lr), self.last_kl));
        }
        self.last_kl = k;
        self.sum_kl += k;
        for (x, y) in lr.iter().zip(lo) {
            self.max_abs = self.max_abs.max((x - y).abs());
        }
        if ar != ao {
            // How close a call it was, in the reference's logits.
            self.flips.push(lr[ar as usize] - lr[ao as usize]);
            self.flip_at.push(self.positions);
        }
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
    let cpu = Qwen2::new_with(m, true);
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    for p in PROMPTS {
        let ids = tok.encode(p);
        compare_each!(&cpu, &ids, steps, (&gpu, &mut ag), (&rows, &mut ac));
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

/// The grades are relative to `ac`, the CPU's per-row kernels against its
/// default (repacked) ones: a reference with no spread (identical logits
/// everywhere) means both were the same model, and every relative bound
/// would be 0. Fail on that by name, not on the first bound it breaks.
pub(crate) fn assert_ref_spread(ac: &Agree) {
    assert!(
        ac.positions == 0 || ac.max_abs > 0.0,
        "reference spread is 0: cpu and rows are the same model (AXON_REPACK=0? build the reference \
         with Qwen2::new_with(m, true))"
    );
}

/// A zero reference spread fails by name (AXON_REPACK=0 once made cpu ==
/// rows and failed `mean KL` instead); an empty reference and a nonzero one
/// pass.
#[test]
fn ref_spread_zero_fails_by_name() {
    assert_ref_spread(&Agree::default());
    assert_ref_spread(&Agree {
        positions: 3,
        max_abs: 0.1,
        ..Agree::default()
    });
    let r = std::panic::catch_unwind(|| {
        assert_ref_spread(&Agree {
            positions: 3,
            ..Agree::default()
        })
    });
    let p = r.expect_err("a zero spread must fail");
    let msg = p
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| p.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(msg.contains("reference spread is 0"), "{msg}");
}

/// `assert_graded` with top-1 at least `floor` % (natural text, prose and
/// code, has more near-ties: the CPU agrees with itself on only 93.8 % of
/// natural_long's 1.5B positions).
pub(crate) fn assert_graded_floor(ag: &Agree, ac: &Agree, floor: usize) {
    assert_ref_spread(ac);
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
    continuations_by(&mut |ids| greedy(r, ids, 32, true), o, tok, name, tie)
}

/// `greedy(.., keep: true)`'s picks and logits.
type Continuation = (Vec<u32>, Vec<Vec<f32>>);

/// `continuations_with` with the reference's `greedy(r, ids, 32, true)`
/// from `reference` (called once per prompt, in PROMPTS order, after `o`'s
/// continuation): `continuations_match` computes it once for several models.
fn continuations_by<O: Lm>(
    reference: &mut dyn FnMut(&[u32]) -> Continuation,
    o: &O,
    tok: &crate::tokenizer::Tokenizer,
    name: &str,
    tie: f32,
) -> (usize, usize) {
    let (mut same, mut matched) = (0, 0);
    for p in PROMPTS {
        let ids = tok.encode(p);
        let (to, _) = greedy(o, &ids, 32, false);
        let (tr, lr) = reference(&ids);
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
    let cpu = Qwen2::new_with(m, true);
    // The CPU's continuations (with logits), once for all the comparisons.
    let refs: Vec<_> = PROMPTS
        .iter()
        .map(|p| {
            let ids = tok.encode(p);
            let c = greedy(&cpu, &ids, 32, true);
            (ids, c)
        })
        .collect();
    let refs = &refs;
    let by_ref = || {
        let mut i = 0;
        move |ids: &[u32]| {
            i += 1;
            let (want, c) = &refs[i - 1];
            debug_assert_eq!(ids, &want[..], "continuations_by asks in PROMPTS order");
            c.clone()
        }
    };
    let (cs, cm) = continuations_by(&mut by_ref(), &rows, &tok, &format!("{name} cpu rows"), tie);
    let paths: &[bool] = if km.mma() { &[true, false] } else { &[false] };
    for &tc in paths {
        gpu.set_tc_attn(tc); // greedy makes its caches after this
        let name = format!("{name} gpu tc {tc}");
        let (gs, gm) = continuations_by(&mut by_ref(), &gpu, &tok, &name, tie);
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
    // llama_model's path (AXON_TEST_LLAMA, skip or panic as it does),
    // mapped; not llama_model itself, which loads a whole copy.
    let p = crate::testutil::llama_model_path()?;
    Some(crate::gguf::load_with(p.to_str().unwrap(), Some(false)).unwrap())
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
    let cpu = Qwen2::new_with(&m, true);
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    for p in [PROMPTS[2], PROMPTS[1]] {
        let one = tok.encode(&format!("{p} "));
        let ids: Vec<u32> = one
            .iter()
            .copied()
            .cycle()
            .take(1100 + one.len() / 2)
            .collect();
        compare_each!(&cpu, &ids, 8, (&gpu, &mut ag), (&rows, &mut ac));
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
    let cpu = Qwen2::new_with(&m, true);
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    let one = tok.encode(&format!("{} ", PROMPTS[4]));
    let ids: Vec<u32> = one.iter().copied().cycle().take(2000).collect();
    compare_each!(&cpu, &ids, 6, (&gpu, &mut ag), (&rows, &mut ac));
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
    let cpu = Qwen2::new_with(m, true);
    let ids = natural_ids(m, 1600);
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    compare_forced2(&cpu, &gpu, &rows, &ids, 1100, &mut ag, &mut ac);
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
        assert!(lay.dense().0.qt == lay.dense().1.qt && net.ffn.is_multiple_of(2));
        let rule = lay.dense().0.rb < GATE_UP4_ROW;
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
            net.dim,
            net.ffn,
            net.layers[0].dense().0.rb
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
    let rb = match kernel.contains("q6k8") {
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
        } else if kernel.starts_with("gemm2t") {
            let sm = gemm2t_smem(qt);
            f.set_max_shared(sm).unwrap();
            g.launch(
                f,
                (nb.div_ceil(64) as u32, rows.div_ceil(G2T_BM) as u32, 1),
                (256, 1, 1),
                sm as u32,
                common,
            )
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
        for nb in [1usize, 6, 7, 8, 13, 31, 33, 64, 66, 77, 129] {
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
                for vm in [16, 32] {
                    if nb <= vm && has_gemvm(qt) {
                        kernels.push(format!("gemvm{vm}_{tn}"));
                    }
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
    for qt in [
        Q4_K, Q6_K, Q5_0, Q8_0, Q5_K, Q4_0, Q4_1, Q5_1, Q2_K, Q3_K, IQ4_NL,
    ] {
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
    for qt in [
        Q4_K, Q6_K, Q8_0, Q5_0, Q5_K, Q4_0, Q4_1, Q5_1, Q2_K, Q3_K, IQ4_NL,
    ] {
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
        for nb in [1usize, 7, 16, 17, 32, 45, 64, 100, 128, 129, 512] {
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
                for vm in [16, 32] {
                    if nb <= vm && has_gemvm(qt) {
                        kernels.push(format!("gemvm{vm}_{tn}"));
                    }
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
        for nb in [1usize, 9, 16, 17, 33, 64, 100, 129] {
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
                for vm in [8, 16, 32] {
                    if nb <= vm {
                        kernels.push(format!("gemvm{vm}_{tn}"));
                    }
                }
                kernels.extend(gemm2_names(qt));
            }
            for k in &kernels {
                let got: Vec<f32> = run(k, &wu, gate.arg(), 2).read().unwrap();
                assert_eq!(bits(&got), bits(&want), "mode 2 {k} {cols}x{rows} nb {nb}");
            }
            for (k, br, sm) in [
                ("gemm2s_q4_k_gu", 32, gemm2_smem(Q4_K, 64, 3)),
                ("gemm2t_q4_k_gu", 64, gemm2t_smem(Q4_K)),
            ] {
                if !km.mma() {
                    break;
                }
                let out = g.alloc(rows * nb * 4).unwrap();
                let f = km.function(k).unwrap();
                f.set_max_shared(sm).unwrap();
                // SAFETY: gemm2s_q4_k_gu's signature (gemm2t's is the same); as above.
                unsafe {
                    g.launch(
                        &f,
                        (nb.div_ceil(64) as u32, rows.div_ceil(br) as u32, 1),
                        (256, 1, 1),
                        sm as u32,
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
                assert_eq!(bits(&got), bits(&want), "{k} {cols}x{rows} nb {nb}");
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
    let mut v = vec![format!("gemm2_{tn}"), format!("gemm2s_{tn}")];
    if matches!(qt, Q4_K | Q5_K) {
        v.push(format!("gemm2t_{tn}"));
    }
    v
}

/// gemm2s (64 x 64) vs gemm2t (128 x 64) Q4_K tiles, per MAC, the r17 way:
/// per sample a ~AXON_BENCH_WIN_MS (200) window of back-to-back launches
/// after half a window of warm-up (power-capped laptops boost on the
/// previous kernel's credit), the cases in a shuffled order every round, and
/// the SM clock sampled (nvidia-smi, 50 ms) over each window. 7B shapes at
/// AXON_BENCH_NB (1024,1088), AXON_BENCH_ROUNDS (7). Results not checked
/// (the bit tests do that).
#[test]
#[ignore]
fn cuda_gemm_tile_bench() {
    use std::io::BufRead;
    use std::time::Instant;
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    if !km.mma() {
        return;
    }
    let info = g.info().unwrap();
    eprintln!(
        "{} sm_{}{} {} SMs",
        info.name, info.cc.0, info.cc.1, info.sms
    );
    // The device's own (NVRTC) build: registers and local (spill) bytes.
    for k in [
        "gemm2s_q4_k",
        "gemm2s_q4_k_gu",
        "gemm2t_q4_k",
        "gemm2t_q5_k",
        "gemm2t_q4_k_gu",
    ] {
        let rl = km.function(k).unwrap().regs_local().unwrap();
        eprintln!("REGS {k}: {} regs, {} local bytes", rl.0, rl.1);
    }
    let nbs: Vec<usize> = std::env::var("AXON_BENCH_NB")
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![1024, 1088]);
    let rounds: usize = std::env::var("AXON_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let win_ms: f64 = std::env::var("AXON_BENCH_WIN_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200.0);
    let samples: std::sync::Arc<std::sync::Mutex<Vec<(Instant, f64)>>> = Default::default();
    let child = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks.sm",
            "--format=csv,noheader,nounits",
            "-lms",
            "50",
        ])
        .stdout(std::process::Stdio::piped())
        .spawn();
    // Killed and reaped on drop, panic or not.
    struct Smi(Option<std::process::Child>);
    impl Drop for Smi {
        fn drop(&mut self) {
            if let Some(c) = self.0.as_mut() {
                let _ = c.kill();
                let _ = c.wait();
            }
        }
    }
    let mut child = Smi(child.ok());
    if let Some(out) = child.0.as_mut().and_then(|c| c.stdout.take()) {
        let sm = samples.clone();
        std::thread::spawn(move || {
            for l in std::io::BufReader::new(out).lines() {
                let Ok(l) = l else { break };
                if let Ok(v) = l.trim().parse::<f64>() {
                    sm.lock().unwrap().push((Instant::now(), v));
                }
            }
        });
    }
    let clock = |t0: Instant, t1: Instant| {
        let v = samples.lock().unwrap();
        let w: Vec<f64> = v
            .iter()
            .filter(|x| x.0 >= t0 && x.0 <= t1)
            .map(|x| x.1)
            .collect();
        if w.is_empty() {
            f64::NAN
        } else {
            w.iter().sum::<f64>() / w.len() as f64
        }
    };
    let mut rng = Rng(7);
    // (name, cols, rows); gu = the fused gate + up of the gate shape.
    let shapes = [
        ("gate", 3584usize, 18944usize),
        ("down", 18944, 3584),
        ("q/wo", 3584, 3584),
        ("kv", 3584, 512),
    ];
    let ws: Vec<DevBuf> = shapes
        .iter()
        .map(|&(_, c, r)| g.upload(&synth_matrix(&mut rng, Q4_K, r, c)).unwrap())
        .collect();
    let wu = g
        .upload(&synth_matrix(&mut rng, Q4_K, 18944, 3584))
        .unwrap();
    let nbmax = *nbs.iter().max().unwrap();
    let out = g.alloc(18944 * nbmax * 4).unwrap();
    let acts: Vec<(usize, DevBuf, DevBuf)> = nbs
        .iter()
        .map(|&nb| {
            let x1: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(3584, 1.0)).collect();
            let x2: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(18944, 1.0)).collect();
            (
                nb,
                quantize_batch(&g, &km, &x1, 3584),
                quantize_batch(&g, &km, &x2, 18944),
            )
        })
        .collect();
    // Cases: (nb index, shape index or 4 = gu, kernel).
    let mut cases: Vec<(usize, usize, &str)> = Vec::new();
    for ni in 0..nbs.len() {
        for si in 0..5 {
            let ks: [&str; 2] = if si == 4 {
                ["gemm2s_q4_k_gu", "gemm2t_q4_k_gu"]
            } else {
                ["gemm2s_q4_k", "gemm2t_q4_k"]
            };
            for k in ks {
                cases.push((ni, si, k));
            }
        }
    }
    let run = |&(ni, si, k): &(usize, usize, &str)| {
        let (nb, a1, a2) = &acts[ni];
        let nb = *nb;
        let f = km.function(k).unwrap();
        if si == 4 {
            let (br, sm) = if k.starts_with("gemm2t") {
                (64, gemm2t_smem(Q4_K))
            } else {
                (32, gemm2_smem(Q4_K, 64, 3))
            };
            f.set_max_shared(sm).unwrap();
            // SAFETY: gemm2s_q4_k_gu's signature; buffers as above.
            unsafe {
                g.launch(
                    &f,
                    (nb.div_ceil(64) as u32, 18944usize.div_ceil(br) as u32, 1),
                    (256, 1, 1),
                    sm as u32,
                    &[
                        ws[0].arg(),
                        wu.arg(),
                        Arg::U64(quant::row_bytes(Q4_K, 3584) as u64),
                        Arg::I32(18944),
                        Arg::I32(3584),
                        a1.arg(),
                        Arg::U64(act_stride_t(3584) as u64),
                        Arg::I32(nb as i32),
                        out.arg(),
                        Arg::U64(18944),
                    ],
                )
                .unwrap()
            };
        } else {
            let (_, cols, rows) = shapes[si];
            let act = if cols == 3584 { a1 } else { a2 };
            let common = [
                ws[si].arg(),
                Arg::U64(quant::row_bytes(Q4_K, cols) as u64),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
                act.arg(),
                Arg::U64(act_stride_t(cols) as u64),
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
    let nc = cases.len();
    let mut reps = vec![0usize; nc];
    let mut ms = vec![Vec::<f64>::new(); nc];
    let mut mhz = vec![Vec::<f64>::new(); nc];
    for r in 0..rounds {
        let mut order: Vec<usize> = (0..nc).collect();
        let mut st = 0x9e37_79b9_7f4a_7c15u64 ^ (r as u64).wrapping_mul(0x1_0000_01b3);
        for i in (1..nc).rev() {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            order.swap(i, (st % (i as u64 + 1)) as usize);
        }
        for &ci in &order {
            if reps[ci] == 0 {
                run(&cases[ci]);
                let e0 = g.record().unwrap();
                for _ in 0..3 {
                    run(&cases[ci]);
                }
                let e1 = g.record().unwrap();
                let t = e0.ms_to(&e1).unwrap() as f64 / 3.0;
                reps[ci] = ((win_ms / t).ceil() as usize).max(3);
            }
            let k = reps[ci];
            for _ in 0..k / 2 {
                run(&cases[ci]);
            }
            g.sync().unwrap();
            let t0 = Instant::now();
            let e0 = g.record().unwrap();
            for _ in 0..k {
                run(&cases[ci]);
            }
            let e1 = g.record().unwrap();
            let el = e0.ms_to(&e1).unwrap() as f64;
            g.sync().unwrap();
            ms[ci].push(el / k as f64);
            mhz[ci].push(clock(t0, Instant::now()));
        }
    }
    drop(child);
    let med = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    for (ci, &(ni, si, k)) in cases.iter().enumerate() {
        let nb = acts[ni].0;
        let (name, cols, rows) = if si == 4 {
            ("gu", 3584, 2 * 18944)
        } else {
            shapes[si]
        };
        let tmac = (rows * cols * nb) as f64 / (med(&ms[ci]) * 1e-3) / 1e12;
        let clk = med(&mhz[ci]);
        let per: Vec<f64> = ms[ci]
            .iter()
            .map(|x| (rows * cols * nb) as f64 / (x * 1e-3) / 1e12)
            .collect();
        let mut p = per.clone();
        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "TILE {name:<4} nb {nb:<5} {k:<15} median {tmac:6.2} TMAC/s (min {:.2} max {:.2}) clk {clk:.0} MHz -> {:.0} MAC/clk/SM",
            p[0],
            p[p.len() - 1],
            tmac * 1e12 / (info.sms as f64 * clk * 1e6)
        );
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
        "gemvm32_q4_k",
        "gemvm32_q6_k",
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
            }
            if let Some(vm) = [8, 16, 32].into_iter().find(|&v| nb <= v)
                && has_gemvm(qt)
            {
                kernels.push(format!("gemvm{vm}_{tn}"));
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

/// `prefill_with` without the KV read-back: the logits only.
fn prefill_with_nokv(net: &CudaQwen2, ids: &[u32], sizes: &[usize]) -> Vec<f32> {
    let mut c = net.new_cache(ids.len() + 1);
    let (&last, rest) = ids.split_last().unwrap();
    let mut pos = 0;
    for &s in sizes.iter().cycle() {
        if pos == rest.len() {
            break;
        }
        let e = (pos + s).min(rest.len());
        if s == 0 {
            net.feed(&mut c, rest[pos], pos).unwrap();
            pos += 1;
            continue;
        }
        net.feed_batch(&rest[pos..e], pos, &mut c);
        pos = e;
    }
    net.forward(last, rest.len(), &mut c)
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
    // Random chunk sizes of 1..48 tokens: one sequence's passes either side
    // of att_per_query and MQ_MAX (attn_blk_b, attn_blk_mq, attn_fa_b).
    let mut rng = Rng(4242);
    let rand: Vec<usize> = (0..60).map(|_| 1 + (rng.next() % 48) as usize).collect();
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
            vec![24, 32, 31, 2, 47],
            rand.clone(),
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

/// A Q4_K_M model's batched products as `wave_gemms` lists them (fused
/// gate + up): dim, q rows, k / v rows, ffn, layers, and how many layers
/// have v and down in Q6_K (llama.cpp's Q4_K_M: half).
fn wave_model(dim: usize, q: usize, kv: usize, ffn: usize, layers: usize) -> Vec<batch::Gemm> {
    use batch::Gemm;
    let one = |qt, rows, cols| Gemm::One { qt, rows, cols };
    (0..layers)
        .flat_map(|l| {
            let t6 = if l % 2 == 0 { Q6_K } else { Q4_K };
            [
                one(Q4_K, q, dim),
                one(Q4_K, kv, dim),
                one(t6, kv, dim),
                one(Q4_K, dim, q),
                Gemm::GateUp { ffn, dim },
                one(t6, dim, ffn),
            ]
        })
        .collect()
}

/// `wave_passes`: as many passes as `passes`, none longer than max, all
/// but the last whole 64-token tiles, never more modelled GEMM time; the
/// equal split for one pass and where nothing beats it; r15's 4090 7B
/// case (1050 + 1050 -> 1088 + 1012). Prints the wave table per GPU (the
/// PR's): 64 x 64 x dim block-waves per layer, summed over the passes.
/// gemm2t where its waves cost less: the big 7B products everywhere, not
/// the 512-row k / v where halving gemm2s's blocks leaves SMs idle.
#[test]
fn tile_t_picks() {
    use batch::tile_t;
    for sms in [20, 46, 128, 170] {
        for rows in [3584, 18944] {
            assert!(tile_t(rows, 64, 1024, sms), "{rows} rows, {sms} SMs");
            assert!(
                tile_t(rows, 64, 2100, sms),
                "{rows} rows nb 2100, {sms} SMs"
            );
        }
        // gate + up: 32 rows of each per gemm2s block.
        assert!(tile_t(18944, 32, 1024, sms), "gu {sms} SMs");
    }
    // 7B k / v at nb 1024 on a 4090: 128 gemm2s blocks, 64 gemm2t.
    assert!(!tile_t(512, 64, 1024, 128));
    // ... but on 20 SMs both grids are several waves.
    assert!(tile_t(512, 64, 1024, 20));
    // A small batch's single partial wave: gemm2s.
    assert!(!tile_t(3584, 64, 64, 128));
    // The floor: one token tile or less always keeps gemm2s, even where
    // the waves alone would pick gemm2t (the 7B FFN on any GPU, the 3584-row
    // products on 20 SMs), and the first nb past it may take gemm2t.
    for sms in [20, 46, 70, 128, 170] {
        for nb in 1..=batch::TILE_T_MIN_NB {
            for (rows, br) in [(18944, 64), (3584, 64), (18944, 32), (12288, 32)] {
                assert!(!tile_t(rows, br, nb, sms), "{rows}/{br} nb {nb} {sms} SMs");
            }
        }
    }
    assert!(tile_t(18944, 64, batch::TILE_T_MIN_NB + 1, 20));
}

#[test]
fn wave_passes_shapes() {
    use batch::{WaveGeom, gemm_cost, passes, wave_floor, wave_passes};
    let models = [
        (
            "qwen2.5-7B",
            wave_model(3584, 3584, 512, 18944, 28),
            3584,
            28,
        ),
        (
            "qwen3-8B",
            wave_model(4096, 4096, 1024, 12288, 36),
            4096,
            36,
        ),
        ("qwen2.5-0.5B", wave_model(896, 896, 128, 4864, 24), 896, 24),
        (
            "qwen2.5-1.5B",
            wave_model(1536, 1536, 256, 8960, 28),
            1536,
            28,
        ),
    ];
    // (GPU, SMs, sm_90a gemmh2)
    let gpus = [
        ("avante 4050L", 20, false),
        ("bella 5070TiL", 46, false),
        ("H100 PCIe", 114, true),
        ("RTX 4090", 128, false),
        ("H100 SXM", 132, true),
        ("RTX 5090", 170, false),
    ];
    for (name, ps, dim, layers) in &models {
        for (gpu, sms, gemmh, tile_t) in gpus
            .iter()
            .flat_map(|&(n, s, h)| [(n, s, h, false), (n, s, h, true)])
        {
            let g = WaveGeom {
                sms,
                gemmh,
                gemmh_all: false,
                tile_t,
            };
            let cost = |nb: usize| gemm_cost(ps, nb, g);
            // 64 x 64 x dim blocks' full waves per layer.
            let unit = (100 * sms * 64 * 64 * dim * layers) as f64;
            let total =
                |p: &[std::ops::Range<usize>]| -> u64 { p.iter().map(|r| cost(r.len())).sum() };
            for max in [1152, 1000, 700, 256] {
                for n in (1..4000).step_by(37).chain([1153, 1777, 2100, 2304, 3000]) {
                    let base = passes(n, max);
                    let Some(v) = wave_passes(n, max, cost) else {
                        continue;
                    };
                    assert_eq!(v.len(), base.len(), "{n} {max}");
                    assert_eq!((v[0].start, v.last().unwrap().end), (0, n));
                    assert!(v.windows(2).all(|w| w[0].end == w[1].start));
                    assert!(v.iter().all(|r| r.len() <= max && r.len() > batch::VM_MAX));
                    assert!(
                        v.iter().all(|r| r.len() >= wave_floor(max)),
                        "{name} {gpu} {n} {max}: a pass under the floor: {v:?}"
                    );
                    assert!(v[..v.len() - 1].iter().all(|r| r.len() % 64 == 0));
                    assert!(
                        (total(&v) as f64) < 0.97 * total(&base) as f64,
                        "{name} {gpu} {n} {max}"
                    );
                }
                for n in [1, 2, 33, 500, max].into_iter().filter(|&n| n <= max) {
                    assert_eq!(wave_passes(n, max, cost), None, "{n} {max}: one pass");
                }
            }
            for n in [1153, 1777, 2100, 2304, 3000, 4000, 6000] {
                let base = passes(n, 1152);
                let v = wave_passes(n, 1152, cost).unwrap_or_else(|| base.clone());
                let lens = |p: &[std::ops::Range<usize>]| -> Vec<usize> {
                    p.iter().map(|r| r.len()).collect()
                };
                let (a, b) = (total(&base) as f64 / unit, total(&v) as f64 / unit);
                eprintln!(
                    "WAVES {name:<13} {gpu:<14} {sms:>3} SMs t {tile_t:<5} pp{n:<5} {:?} {a:7.1} -> {:?} {b:7.1} ({:+.1} %)",
                    lens(&base),
                    lens(&v),
                    100.0 * (b / a - 1.0)
                );
            }
        }
    }
    // Degenerate max: one tile per pass (64: every split the same cost)
    // or none (63).
    let ps = &models[0].1;
    let g = WaveGeom {
        sms: 128,
        gemmh: false,
        gemmh_all: false,
        tile_t: true,
    };
    for n in 65..400 {
        assert_eq!(
            wave_passes(n, 64, |nb| gemm_cost(ps, nb, g)),
            None,
            "{n} / 64"
        );
        assert_eq!(
            wave_passes(n, 63, |nb| gemm_cost(ps, nb, g)),
            None,
            "{n} / 63"
        );
    }
    // Under the margin: no wave effect (cost by the token alone), or a
    // split that saves 2 % (< 3 %) of a cost made up for it.
    assert_eq!(wave_passes(2100, 1152, |nb| nb as u64), None);
    let toy = |nb: usize| if nb == 1088 || nb == 1012 { 98 } else { 100 };
    assert_eq!(wave_passes(2100, 1152, toy), None);
    let toy = |nb: usize| if nb == 1088 || nb == 1012 { 96 } else { 100 };
    assert_eq!(
        wave_passes(2100, 1152, toy),
        Some(vec![0..1088, 1088..2100])
    );
    // The RTX 4090 cells measured (#92), with gemm2s (AXON_GEMM_TILE=64):
    // r15 pick 1's pp2100 split for 7B and Qwen3-8B (+3.0 % / +3.9 % e2e)
    // stays; 7B pp1153 (704 + 449, -0.4 %), pp1777 (1024 + 753, +1.8 %) and
    // pp3000 (1152 + 1152 + 696, +1.1 %) are under the margin now.
    let g64 = WaveGeom { tile_t: false, ..g };
    let at =
        |m: usize, n: usize, g: WaveGeom| wave_passes(n, 1152, |nb| gemm_cost(&models[m].1, nb, g));
    for g in [g64, g] {
        assert_eq!(at(0, 1153, g), None);
        assert_eq!(at(0, 1777, g), None);
        assert_eq!(at(0, 3000, g), None);
    }
    assert_eq!(at(0, 2100, g64), Some(vec![0..1088, 1088..2100]));
    assert_eq!(at(1, 2100, g64), Some(vec![0..1088, 1088..2100]));
    // With gemm2t (the default), pp2100 still splits two ways under the
    // margin, but which split is a coin flip of the model: for 7B,
    // 1152 + 948 and 1088 + 1012 (both 33 token tiles) are 0.005 % apart
    // (gemm_cost per pass, 1e15 units: 1088 61.56, 1012 57.75, 1152 64.13,
    // 948 55.18; equal 2 x 61.56), far inside the error of PARTIAL_WAVE
    // and TILE_T_TIME. So: assert a split, within 0.1 % of the cheapest
    // of those two candidates, not the exact one.
    for m in [0, 1] {
        let ps = &models[m].1;
        let c = |v: &[std::ops::Range<usize>]| -> f64 {
            v.iter().map(|r| gemm_cost(ps, r.len(), g)).sum::<u64>() as f64
        };
        let v = at(m, 2100, g).expect("pp2100 splits with gemm2t");
        assert_eq!(v.len(), 2);
        let best = c(&[0..1088, 1088..2100]).min(c(&[0..1152, 1152..2100]));
        assert!(c(&v) <= best * 1.001, "{m}: {v:?}");
        assert!(c(&v) < 0.97 * c(&passes(2100, 1152)), "{m}: {v:?}");
    }
}

/// Batched prefill with AXON_WAVE_SPLIT's pass sizes == with the equal
/// passes (main's), KV bytes and logit bits, at fixed prompt lengths plus
/// the first two up to 4600 tokens whose split differs on this GPU
/// (printed); one of those again from position 37 (after a fed prefix);
/// and == per-token decode at 1777 tokens on 0.5B. It flips the
/// `CudaQwen2::wave_split` field, not the environment: AXON_WAVE_SPLIT's
/// parsing (a `map_or` at construction) isn't covered here. 0.5B and 1.5B;
/// AXON_BIG_MODELS=1 adds qwen2.5 7B and qwen3 8B.
///
/// It fails as vacuous if no length up to 4600 splits differently on the
/// GPU it runs on (a GPU whose SM count makes every equal split as good:
/// widen the search before calling that a bug).
#[test]
fn gpu_prefill_wave_split_bits() {
    use batch::passes;
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let mut files = vec![
        "models/qwen2.5-0.5b-q4_k_m.gguf",
        "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ];
    if std::env::var("AXON_BIG_MODELS").is_ok_and(|v| v == "1") {
        files.extend([
            "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "models/Qwen3-8B-Q4_K_M.gguf",
        ]);
    }
    const FROM: usize = 37;
    const LONGEST: usize = 4600;
    for f in files {
        let Some(m) = model_at(f) else {
            continue;
        };
        let km = Kernels::module_for(&g, &m).unwrap();
        let all = long_ids(&m, LONGEST + FROM);
        let mut net = CudaQwen2::new(&g, &km, &m, all.len() + 1).unwrap();
        let max = net.prefill_batch();
        net.wave_split = true;
        let mut lens = vec![1153, 1300, 1777, 2100, 2304, 2900];
        let found: Vec<usize> = (max + 1..=LONGEST)
            .filter(|n| !lens.contains(n) && net.wave_split_passes(*n, max).is_some())
            .take(2)
            .collect();
        lens.extend(&found);
        let mut differ = 0;
        for &n in &lens {
            let ids = &all[..n];
            net.wave_split = true;
            let split = net.wave_split_passes(n, max);
            differ += split.is_some() as usize;
            let mut a = net.new_cache(n + 1);
            let la = net.prefill_logits(ids, 0, &mut a);
            net.wave_split = false;
            let mut b = net.new_cache(n + 1);
            let lb = net.prefill_logits(ids, 0, &mut b);
            assert_eq!(bits(&la), bits(&lb), "{f} {n}: logits, wave split vs equal");
            assert!(
                kv_bytes(&net, &a, n) == kv_bytes(&net, &b, n),
                "{f} {n}: KV, wave split vs equal"
            );
            let lens = |p: &[std::ops::Range<usize>]| -> Vec<usize> {
                p.iter().map(|r| r.len()).collect()
            };
            eprintln!(
                "{f} pp{n} on {} SMs: equal {:?}, wave {:?}: identical",
                net.bf.sms,
                lens(&passes(n, max)),
                split.as_deref().map(lens)
            );
            if n == 1777 && f.contains("0.5b") {
                let mut d = net.new_cache(n + 1);
                for (p, &t) in ids[..n - 1].iter().enumerate() {
                    net.feed(&mut d, t, p).unwrap();
                }
                let dec = net.forward(ids[n - 1], n - 1, &mut d);
                assert_eq!(
                    bits(&la),
                    bits(&dec),
                    "{f} {n}: wave split vs per-token decode"
                );
                assert!(
                    kv_bytes(&net, &a, n) == kv_bytes(&net, &d, n),
                    "{f} {n}: KV vs decode"
                );
                eprintln!("{f} pp{n}: wave split == per-token decode");
            }
        }
        // A multi-pass prompt after a fed prefix (start > 0).
        if let Some(&n) = found.first() {
            let ids = &all[..FROM + n];
            let mut out = Vec::new();
            for wave in [true, false] {
                net.wave_split = wave;
                let mut c = net.new_cache(FROM + n + 1);
                net.feed_batch(&ids[..FROM], 0, &mut c);
                let l = net.prefill_logits(&ids[FROM..], FROM, &mut c);
                out.push((bits(&l), kv_bytes(&net, &c, FROM + n)));
            }
            assert!(out[0] == out[1], "{f} {n} at {FROM}: wave split vs equal");
            eprintln!("{f} pp{n} at {FROM}: identical");
        }
        net.wave_split = true;
        if net.wave_gemms().is_some() {
            assert!(
                differ > 0,
                "{f}: no prompt length up to {LONGEST} splits differently here (test vacuous)"
            );
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

/// The files the Q2_K / Q3_K tests run on, and the new types each holds.
/// bartowski's 0.5B "Q2_K" and "Q3_K_M" have NO Q2_K and NO Q3_K tensors
/// respectively at 896 columns: llama.cpp stores those rows as IQ4_NL (896
/// isn't a multiple of 256); the 0.5B Q2_K's 4864-wide ffn_down is Q3_K.
/// Real Q2_K tensors exist only in the 7B Q2_K here.
const Q23_FILES: [(&str, &str, &[u32]); 5] = [
    (
        "Qwen2.5-0.5B-Instruct-Q2_K.gguf",
        "0.5B Q2_K",
        &[IQ4_NL, Q3_K],
    ),
    (
        "Qwen2.5-0.5B-Instruct-Q3_K_M.gguf",
        "0.5B Q3_K_M",
        &[IQ4_NL],
    ),
    ("Qwen2.5-1.5B-Instruct-Q3_K_M.gguf", "1.5B Q3_K_M", &[Q3_K]),
    (
        "Llama-3.2-1B-Instruct-Q3_K_L.gguf",
        "llama 1B Q3_K_L",
        &[Q3_K],
    ),
    ("Qwen2.5-7B-Instruct-Q2_K.gguf", "7B Q2_K", &[Q2_K, Q3_K]),
];

/// `f` (one of `Q23_FILES`) loaded, after checking it holds the types it's
/// listed with; None (said loudly) when the file isn't there.
fn q23_model(f: &str, want: &[u32]) -> Option<Model> {
    let m = model_at(&format!("models/{f}"));
    match &m {
        None => eprintln!("SKIPPING {f}: not in models/ (its Q2_K/Q3_K/IQ4_NL coverage is lost)"),
        Some(m) => {
            for &qt in want {
                assert!(
                    m.tensors.iter().any(|t| t.qtype.id == qt),
                    "{f}: no {} tensor",
                    crate::gguf::qtype(qt).name
                );
            }
        }
    }
    m
}

/// Asserts the runs covered all three new types, unless a file was missing.
fn q23_assert_covered(seen: &[u32], all_present: bool, what: &str) {
    if !all_present {
        eprintln!("SKIPPING {what}'s coverage check: model files missing (see above)");
        return;
    }
    for qt in [Q2_K, Q3_K, IQ4_NL] {
        assert!(
            seen.contains(&qt),
            "{what}: {} not covered",
            crate::gguf::qtype(qt).name
        );
    }
}

/// The new types' products on the real files' weights (every Q2_K / Q3_K /
/// IQ4_NL matrix of the first two layers and the last): the decode GEMV within
/// f32 rounding of the CPU's dot_row_q8 (relative to sum |w x|) on the same
/// quantized activation, and every batched kernel (gemvn, the tensor-core
/// gemm2 / gemm2s) == the GEMV bit for bit, at batches below, inside and past
/// the tiles. A wrong sub-block scale moves a row by ~1e-3 of sum |w x|.
#[test]
fn gpu_products_real_weights_q2k_q3k() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module_ext(&g).unwrap();
    let mut rng = Rng(0x0a3b_2c1d_7e6f_5a4b);
    let (mut seen, mut all_present) = (Vec::new(), true);
    for (f, _, want) in Q23_FILES {
        let Some(m) = q23_model(f, want) else {
            all_present = false;
            continue;
        };
        let nl = m
            .tensors
            .iter()
            .filter(|t| t.name.ends_with("attn_norm.weight"))
            .count();
        let mut file_seen = Vec::new();
        for t in &m.tensors {
            let qt = t.qtype.id;
            let layer_ok = [0, 1, nl - 1]
                .iter()
                .any(|l| t.name.starts_with(&format!("blk.{l}.")));
            if !matches!(qt, Q2_K | Q3_K | IQ4_NL) || !layer_ok || t.dims.len() != 2 {
                continue;
            }
            let cols = t.dims[0] as usize;
            let rb = quant::row_bytes(qt, cols);
            let rows = (t.n_elements as usize / cols).min(1001);
            let wd = &m.tensor_data(t)[..rows * rb];
            let w = g.upload(wd).unwrap();
            let x = rng.vec(cols, 2.0);
            let want = gpu_gemv(&g, &km, "gemv", qt, wd, cols, &x);
            let xq = quant::QuantizedAct::for_types(&x, &[qt]);
            let mut row = vec![0f32; cols];
            let mut worst = 0f64;
            for r in 0..rows {
                let wr = &wd[r * rb..(r + 1) * rb];
                let cpu = quant::dot_row_q8(qt, wr, &xq) as f64;
                quant::dequant(qt, wr, &mut row);
                let mag: f64 = row
                    .iter()
                    .zip(&x)
                    .map(|(&a, &b)| (a as f64 * b as f64).abs())
                    .sum();
                worst = worst.max((want[r] as f64 - cpu).abs() / mag.max(1e-30));
            }
            assert!(worst < 1e-6, "{f} {}: gemv vs cpu {worst:e}", t.name);
            for nb in [7usize, 77, 129] {
                let xs: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 2.0)).collect();
                let wants: Vec<Vec<f32>> = xs
                    .iter()
                    .map(|x| gpu_gemv(&g, &km, "gemv", qt, wd, cols, x))
                    .collect();
                let act = quantize_batch(&g, &km, &xs, cols);
                let mut kernels: Vec<String> = vec!["gemvn_2x8".into(), "gemvn_2x16".into()];
                if km.mma() {
                    kernels.extend(gemm2_names(qt));
                }
                for k in &kernels {
                    let got =
                        gpu_batched(&g, &km, k, qt, &w, rows, cols, &act, nb, &Epi::default());
                    for z in 0..nb {
                        assert_eq!(
                            bits(&got[z * rows..(z + 1) * rows]),
                            bits(&wants[z]),
                            "{f} {} {k} nb {nb} token {z}",
                            t.name
                        );
                    }
                }
            }
            eprintln!(
                "{f} {} ({}): gemv vs cpu {worst:.2e}; batched == gemv",
                t.name,
                crate::gguf::qtype(qt).name
            );
            file_seen.push(qt);
        }
        for &qt in want {
            assert!(
                file_seen.contains(&qt),
                "{f}: no {} matrix checked",
                crate::gguf::qtype(qt).name
            );
        }
        seen.extend(file_seen);
    }
    q23_assert_covered(&seen, all_present, "gpu_products_real_weights_q2k_q3k");
}

/// The Q2_K / Q3_K / IQ4_NL files, GPU vs CPU: `assert_graded`'s rules, with
/// the near-tie flip rules scaled by the CPU's own spread on the same file
/// (`ac`, its per-row vs its default kernels; on these files only their
/// Q4_K / Q6_K / Q5_0 / Q8_0 rows differ between the two sets, so `ac`
/// measures how much the model amplifies last-bit noise, not Q2_K/Q3_K).
/// Measured on sm_120 (1017-173844-9d79), 1.5B Q3_K_M: GPU vs CPU 188/198
/// top-1, mean KL 1.13e-2, flips 0.55 / 0.78 / 0.81 at >= 0.5; the CPU vs
/// itself 187/198, 1.23e-2 (1.6x the 1.5B Q4_K_M's 7.8e-3), largest flip
/// 0.42. So:
/// - top-1 within 2 points of the CPU's self-agreement and >= 94%;
/// - every flip below max(0.5, 2x the CPU's largest self-flip margin, half
///   the CPU's own max |dlogit| on the file): flipping a margin m moves two
///   logits m apart, and the CPU's two kernel sets move single logits by up
///   to `ac.max_abs` against each other on the same positions. Measured on
///   the 0.5B Q2_K (sm_120, 1018-014632-e746 and 1017-195144-d5d0, the same
///   numbers twice): one flip of 0.95 at the haiku prompt's step 22 (EOT vs
///   "\n"), CPU max |dlogit| 2.29, so a 1.14 bound. It is attention-order
///   noise, not a product deviation: on the same positions
///   (dump_logits_positions, 1018-023350-ffd4) the GPU with the split
///   attention (another in-contract float order, the same weight kernels)
///   keeps the CPU's token there (margin 0.88), and flips other near-ties
///   instead (0.53, 0.59); llama.cpp's CPU and CUDA both keep the CPU's token
///   (1018-022009-945f), though its own CPU vs CUDA spread on this file is
///   smaller (max flip 0.15; 3.48 on the 0.5B Q3_K_M: 1018-013634-86b9), and
///   the products are exact (gpu_products_real_weights_q2k_q3k);
/// - flips at >= 0.5: at most max(2, CPU's own + 2, 2 x the CPU's self-KL
///   over the 1.5B Q4_K_M's 7.8e-3, rounded up): a file whose CPU kernel sets
///   disagree k times as much gets k times assert_graded's 2;
/// - mean KL within 2x the CPU's own (absolute 5e-3 where the CPU's two
///   sets nearly agree), KL above 0.5 at one position at most and only where
///   the CPU flips too, none above 2, max |dlogit| below 6.
///
/// The products themselves are checked exactly (gpu_products_real_weights_q2k_q3k).
#[test]
fn gpu_logits_match_cpu_q2k_q3k() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let (mut seen, mut all_present) = (Vec::new(), true);
    for (f, name, want) in Q23_FILES {
        let Some(m) = q23_model(f, want) else {
            all_present = false;
            continue;
        };
        let (ag, ac) = agreement(&m, 32);
        report(&format!("{name} gpu vs cpu"), &ag);
        report(&format!("{name} cpu rows vs cpu"), &ac);
        let n = ag.positions;
        assert!(
            ag.top1 * 100 + 2 * n >= ac.top1 * 100 && ag.top1 * 100 >= 94 * n,
            "{name}: top-1 {}/{n} (cpu vs itself {})",
            ag.top1,
            ac.top1
        );
        let ac_flip = ac.flips.iter().fold(0f32, |a, &d| a.max(d));
        let tie = (2.0 * ac_flip).max(0.5).max(ac.max_abs / 2.0);
        let self_kl = ac.sum_kl / n as f64;
        let ac_wide = ac.flips.iter().filter(|&&d| d >= 0.5).count();
        let scaled = (2.0 * self_kl / 7.8e-3).ceil() as usize;
        let max_wide = 2.max(ac_wide + 2).max(scaled);
        let wide = ag.flips.iter().filter(|&&d| d >= 0.5).count();
        assert!(
            ag.flips.iter().all(|&d| d < tie) && wide <= max_wide,
            "{name}: a flip >= {tie} or {wide} > {max_wide} at >= 0.5: {:?}",
            ag.flips
        );
        let kl_cap = if self_kl > 1e-4 {
            2.0 * ac.sum_kl
        } else {
            5e-3 * n as f64
        };
        assert!(ag.sum_kl <= kl_cap, "{name}: mean KL");
        assert!(
            ag.over_half.len() <= 1 && ag.max_kl < 2.0,
            "{name}: max KL {} (above 0.5: {:?})",
            ag.max_kl,
            ag.over_half
        );
        for &(i, k, mg, _) in &ag.over_half {
            assert!(
                ac.flip_at.contains(&i),
                "{name}: KL {k} at position {i} (margin {mg}), where the CPU's kernel sets agree"
            );
        }
        assert!(
            ag.max_abs < 6.0,
            "{name}: max logit difference {}",
            ag.max_abs
        );
        seen.extend_from_slice(want);
    }
    q23_assert_covered(&seen, all_present, "gpu_logits_match_cpu_q2k_q3k");
}

/// Dumps `gpu_logits_match_cpu_q2k_q3k`'s positions for an outside
/// comparison (llama.cpp CPU vs CUDA on the same token prefixes): for the
/// model at $AXON_DUMP_MODEL, PROMPTS with 32 greedy CPU steps each; per
/// position the prefix ids and the top-20 log-probabilities of the CPU (to
/// $AXON_DUMP_OUT.axon-cpu.json) and of the GPU (.axon-cuda.json), in
/// scripts/oracle/llama_logits.py's format.
#[test]
#[ignore]
fn dump_logits_positions() {
    let (Ok(path), Ok(out)) = (
        std::env::var("AXON_DUMP_MODEL"),
        std::env::var("AXON_DUMP_OUT"),
    ) else {
        return;
    };
    let g = crate::testutil::gpu().unwrap();
    let m = crate::gguf::load_with(&path, Some(false)).unwrap();
    let km = Kernels::module_for(&g, &m).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let mut gpu = CudaQwen2::new(&g, &km, &m, 256).unwrap();
    // AXON_DUMP_SPLIT=1: the split attention (no tensor cores), another
    // in-contract GPU float order, to see how far a position moves.
    if std::env::var("AXON_DUMP_SPLIT").as_deref() == Ok("1") {
        gpu.set_tc_attn(false);
    }
    let cpu = Qwen2::new(&m);
    let top = |l: &[f32]| -> String {
        let ls = log_softmax(l);
        let mut idx: Vec<usize> = (0..l.len()).collect();
        idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]).then(a.cmp(&b)));
        let v: Vec<String> = idx[..20]
            .iter()
            .map(|&i| format!("[{i}, {}]", ls[i]))
            .collect();
        format!("[{}]", v.join(", "))
    };
    let (mut jc, mut jg) = (Vec::new(), Vec::new());
    for (pi, p) in PROMPTS.iter().enumerate() {
        let mut ids = tok.encode(p);
        let mut rc = cpu.new_cache(ids.len() + 34);
        let mut oc = gpu.new_cache(ids.len() + 34);
        let (&last, rest) = ids.split_last().unwrap();
        let start = rest.len();
        cpu.feed_batch(rest, 0, &mut rc);
        gpu.feed_batch(rest, 0, &mut oc);
        let mut t = last;
        for s in 0..=32 {
            let pos = start + s;
            let lr = cpu.forward(t, pos, &mut rc);
            let lo = gpu.forward(t, pos, &mut oc);
            let idl: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
            let head = format!(
                "{{\"prompt\": \"p{pi}s{s}\", \"ids\": [{}], \"top\": ",
                idl.join(", ")
            );
            jc.push(format!("{head}{}}}", top(&lr)));
            jg.push(format!("{head}{}}}", top(&lo)));
            t = crate::argmax(&lr);
            ids.push(t);
        }
    }
    std::fs::write(
        format!("{out}.axon-cpu.json"),
        format!("[{}]", jc.join(",\n")),
    )
    .unwrap();
    std::fs::write(
        format!("{out}.axon-cuda.json"),
        format!("[{}]", jg.join(",\n")),
    )
    .unwrap();
    eprintln!("dumped {} positions", jc.len());
}

/// The Q2_K / Q3_K files: decode == every prefill batching == prefill from a
/// fed prefix (cache reuse) == verify, bit for bit, on the f16 cache, and on
/// the q8 one for the 0.5B Q2_K and the 1.5B Q3_K_M.
#[test]
fn gpu_prefill_batchings_bit_identical_q2k_q3k() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let (mut seen, mut all_present) = (Vec::new(), true);
    for (f, name, want) in Q23_FILES {
        let Some(m) = q23_model(f, want) else {
            all_present = false;
            continue;
        };
        prefill_bits(&m, name, CacheKv::F16);
        verify_bits(&m, name, CacheKv::F16);
        if matches!(name, "0.5B Q2_K" | "1.5B Q3_K_M") {
            // Group 32: valid at every head size (64 only where it's head_dim).
            prefill_bits(&m, name, CacheKv::Q8 { group: 32 });
            verify_bits(&m, name, CacheKv::Q8 { group: 32 });
        }
        seen.extend_from_slice(want);
    }
    q23_assert_covered(
        &seen,
        all_present,
        "gpu_prefill_batchings_bit_identical_q2k_q3k",
    );
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
    let mut net = CudaQwen2::new(&g, &km, &m, start + nb + 8).unwrap();
    let mut c = net.new_cache(start + nb + 8);
    let ids = long_ids(&m, start + nb);
    if start > 0 {
        net.feed_batch(&ids[..start], 0, &mut c);
    }
    // AXON_PROF_TILE_AB=n: n rounds, each a pass with gemm2t (the default)
    // and one with AXON_GEMM_TILE=64's gemm2s, in alternating order; per arm
    // the median of 3 timed passes after a warm one, then the medians.
    if let Ok(n) = std::env::var("AXON_PROF_TILE_AB") {
        let n: usize = n.parse().unwrap();
        let mut spare = net.bf.gemm_t.take();
        assert!(spare.is_some(), "no gemm2t in this module");
        let mut ms = [Vec::new(), Vec::new()];
        for r in 0..n {
            for arm in [r % 2, 1 - r % 2] {
                if arm == 1 {
                    if net.bf.gemm_t.is_none() {
                        net.bf.gemm_t = spare.take();
                    }
                } else if net.bf.gemm_t.is_some() {
                    spare = net.bf.gemm_t.take();
                }
                let mut one = || {
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
                    t.elapsed().as_secs_f64() * 1e3
                };
                one();
                let mut v = [one(), one(), one()];
                v.sort_by(f64::total_cmp);
                ms[arm].push(v[1]);
            }
        }
        for v in &mut ms {
            v.sort_by(f64::total_cmp);
        }
        let (a, b) = (ms[0][n / 2], ms[1][n / 2]);
        eprintln!(
            "TILE_AB {rel} nb {nb} at {start}: gemm2s {a:.2} ms ({:.0} tok/s), gemm2t {b:.2} ms ({:.0} tok/s): {:+.1} % tok/s ({n} rounds)",
            nb as f64 / a * 1e3,
            nb as f64 / b * 1e3,
            100.0 * (a / b - 1.0)
        );
        return;
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
                "gate" => macs(l0.dense().0),
                "gate_up" => macs(l0.dense().1),
                "down4" | "down6" => macs(l0.dense().2) * k as f64 / nl,
                _ => 0.0,
            };
            let shape = match n {
                "q" => format!("{}x{} {}", l0.q.rows, l0.q.cols, type_name(l0.q.qt)),
                "k" => format!("{}x{} {}", l0.k.rows, l0.k.cols, type_name(l0.k.qt)),
                "v4" => format!("v {}x{} q4_k", l0.v.rows, l0.v.cols),
                "wo" => format!("{}x{} {}", l0.o.rows, l0.o.cols, type_name(l0.o.qt)),
                "gate" => format!(
                    "{}x{} {}",
                    l0.dense().0.rows,
                    l0.dense().0.cols,
                    type_name(l0.dense().0.qt)
                ),
                "gate_up" => format!(
                    "up {}x{} {}",
                    l0.dense().1.rows,
                    l0.dense().1.cols,
                    type_name(l0.dense().1.qt)
                ),
                "down4" => format!("{}x{} q4_k", l0.dense().2.rows, l0.dense().2.cols),
                "down6" => format!("{}x{} q6_k", l0.dense().2.rows, l0.dense().2.cols),
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
/// (attn_blk_b, attn_cmb_b), one decode step's attn_blk + attn_cmb at the
/// last query's position, attn_blk_mq, and the batched pass's per-query
/// path over all the queries (attn_blk_mq or attn_blk_b, then attn_cmb_b,
/// att_per_query at a time). Interleaved rounds, best of each.
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
    let nz: usize = std::env::var("AXON_ATT_NZ").map_or(64, |v| v.parse().unwrap());
    let start: usize = std::env::var("AXON_PROF_START").map_or(1536, |v| v.parse().unwrap());
    attn_bench(&g, &km, [shape[0], shape[1], shape[2]], nz, start);
}

/// cuda_attn_bench at AXON_MQ_SHAPE (n_heads,n_kv,hd; default the 7B's
/// 28,4,128) over a grid, three interleaved rounds:
/// AXON_MQ_NZ queries (default 4,8,16,24,32,48,64,96) at AXON_MQ_DEPTH
/// positions (default 4096,8192,16384,24576). Per point a `SWEEP` line:
/// depth, queries, then us of the per-query path over all of them with
/// attn_blk_mq and with attn_blk_b (+ attn_cmb_b, att_per_query at a
/// time), attn_fa_b + quantize_b, attn_fa_bs + quantize_b.
#[test]
#[ignore]
fn cuda_attn_mq_sweep() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let list = |k: &str, d: &str| -> Vec<usize> {
        std::env::var(k)
            .unwrap_or(d.into())
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect()
    };
    let nzs = list("AXON_MQ_NZ", "4,8,16,24,32,48,64,96");
    let depths = list("AXON_MQ_DEPTH", "4096,8192,16384,24576");
    // AXON_MQ_SHAPE: n_heads,n_kv,hd (default the 7B's).
    let sh = list("AXON_MQ_SHAPE", "28,4,128");
    for r in 0..3 {
        for &start in &depths {
            for &nz in &nzs {
                let b = attn_bench(&g, &km, [sh[0], sh[1], sh[2]], nz, start);
                eprintln!(
                    "SWEEP round {r} depth {start} nz {nz} mq {:.1} blk_b {:.1} fa_b {:.1} fa_bs {:.1}",
                    b[8] * 1e3,
                    b[9] * 1e3,
                    (b[0] + b[1]) * 1e3,
                    (b[6] + b[1]) * 1e3
                );
            }
        }
    }
}

/// One cuda_attn_bench point; the best ms of each entry.
fn attn_bench(
    g: &Gpu,
    km: &crate::cuda::Module,
    shape: [usize; 3],
    nz: usize,
    start: usize,
) -> [f64; 10] {
    let [nh, nkv, hd] = shape;
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
        format!("attn_blk_mq_{hd}"),
        "mq+cmb_b (all)".to_string(),
        "blk_b+cmb_b (all)".to_string(),
    ];
    // 8, 9: the batched pass's per-query path over all nz queries,
    // att_per_query at a time (batch.rs): attn_blk_mq or attn_blk_b, then
    // attn_cmb_b.
    let fs: Vec<Func> = names
        .iter()
        .enumerate()
        .map(|(k, n)| match k {
            8 => km.function(&format!("attn_blk_mq_{hd}")).unwrap(),
            9 => km.function(&format!("attn_blk_b_{hd}")).unwrap(),
            _ => km.function(n).unwrap(),
        })
        .collect();
    let fcmb = km.function("attn_cmb_b").unwrap();
    let group = nh / nkv;
    let per = batch::att_per_query(hd);
    for f in [&fs[7], &fs[8]] {
        f.set_max_shared(batch::mq_smem(hd, 0)).unwrap();
    }
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
    let mut best = [f64::INFINITY; 10];
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
                    7 => g.launch(
                        f,
                        (nkv as u32, gy, nq.div_ceil(batch::MQ_Q) as u32),
                        (32 * group as u32, 1, 1),
                        batch::mq_smem(hd, 0) as u32,
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
                            Arg::Ptr(0), // no page table
                            Arg::I32(nq as i32),
                        ],
                    ),
                    8 | 9 => {
                        for z0 in (0..nz).step_by(per) {
                            let n = (nz - z0).min(per);
                            let mut a = vec![
                                q.arg(),
                                kc.arg(),
                                vc.arg(),
                                Arg::I32(start as i32),
                                Arg::I32(z0 as i32),
                                Arg::I32(nh as i32),
                                Arg::I32(nkv as i32),
                                pre,
                                Arg::I32(nblk_max as i32),
                                po.arg(),
                                pm.arg(),
                            ];
                            if k == 8 {
                                a.extend([Arg::Ptr(0), Arg::I32(n as i32)]);
                                g.launch(
                                    f,
                                    (nkv as u32, gy, n.div_ceil(batch::MQ_Q) as u32),
                                    (32 * group as u32, 1, 1),
                                    batch::mq_smem(hd, 0) as u32,
                                    &a,
                                )
                                .unwrap();
                            } else {
                                a.extend([Arg::Ptr(0), Arg::Ptr(0)]);
                                g.launch(f, (nkv as u32, gy, n as u32), (32, 1, 1), 0, &a)
                                    .unwrap();
                            }
                            g.launch(
                                &fcmb,
                                (dim.div_ceil(256) as u32, n as u32, 1),
                                (256, 1, 1),
                                0,
                                &[
                                    po.arg(),
                                    pm.arg(),
                                    Arg::I32(start as i32),
                                    Arg::I32(z0 as i32),
                                    Arg::I32(dim as i32),
                                    Arg::I32(hd as i32),
                                    Arg::I32(nblk_max as i32),
                                    Arg::I32(0),
                                    act.arg(),
                                    Arg::U64(ast as u64),
                                    Arg::Ptr(0),
                                ],
                            )
                            .unwrap();
                        }
                        Ok(())
                    }
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
            0 | 6 | 8 | 9 => (nz, flops(nz) / (b * 1e-3) / 1e12),
            2 | 7 => (nq, flops(nq) / (b * 1e-3) / 1e12),
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
            "{nh}/{nkv}x{hd} {n:<18} {qn:>4} queries up to {ctx}: {:>8.1} us {tf:>6.2} TFLOP/s",
            b * 1e3
        );
    }
    best
}

/// The serve pool's decode attention (batch.rs `rows`, tensor cores): nb
/// rows, each its own sequence at position `depth` in its own slot (rp),
/// the path the pool step takes today: att_per_query (16 at 128-wide heads)
/// queries per launch, attn_blk_b (or attn_blk_b_q8) then attn_cmb_b.
/// nblk_max as pool_step_cost's pool: ceil((depth + 64) / 64) rounded up.
/// AXON_POOL_NB (32,64,128), AXON_POOL_DEPTH (600,4096), AXON_POOL_KV
/// (f16,q8), AXON_ATT_SHAPE (28,4,128). Per point a `POOL` line: the
/// median and best of 7 timed steps (each one pass over all nb rows) of
/// attn alone, combine alone, and both interleaved as batch.rs runs them,
/// and the KV bytes read / (attn + cmb) in GB/s; then batch.rs's
/// attn_blk_nb path (nb_split slices, AXON_ATT_NB_S forces it) on the same
/// rows and its speedup over "both". "cmb" alone rereads one launch's
/// partials, L2-hot; in "both" the KV stream has evicted them. MEASURE,
/// not a check.
#[test]
#[ignore]
fn cuda_attn_pool_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let list = |k: &str, d: &str| -> Vec<String> {
        std::env::var(k)
            .unwrap_or(d.into())
            .split(',')
            .map(|v| v.trim().to_string())
            .collect()
    };
    let num = |v: &[String]| -> Vec<usize> { v.iter().map(|s| s.parse().unwrap()).collect() };
    let nbs = num(&list("AXON_POOL_NB", "32,64,128"));
    let depths = num(&list("AXON_POOL_DEPTH", "600,4096"));
    let kvs = list("AXON_POOL_KV", "f16,q8");
    let sh = num(&list("AXON_ATT_SHAPE", "28,4,128"));
    let (nh, nkv, hd) = (sh[0], sh[1], sh[2]);
    let gi = g.info().unwrap();
    eprintln!("POOL bench {} ({} SMs), {nh}/{nkv}x{hd}", gi.name, gi.sms);
    let nb_max = *nbs.iter().max().unwrap();
    let dim = nh * hd;
    let kvd = nkv * hd;
    let per = batch::att_per_query(hd);
    let max_split = (8 * gi.sms as usize / nkv).clamp(1, MAX_SPLIT);
    let pre = Arg::F32(att_pre(hd));
    let ast = act_stride_t(dim);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let q = g
        .upload(
            &(0..nb_max * dim)
                .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
                .collect::<Vec<f32>>(),
        )
        .unwrap();
    let act = g.alloc(nb_max * ast).unwrap();
    for &depth in &depths {
        let ctx = (depth + 64).next_multiple_of(64);
        let nblk_max = ctx.div_ceil(64);
        let gy = max_split.min(nblk_max) as u32;
        let rows = nb_max * ctx;
        let po = g.alloc(per * nh * nblk_max * hd * 4).unwrap();
        let pm = g.alloc(per * nh * nblk_max * 2 * 4).unwrap();
        for kv in &kvs {
            let q8 = kv == "q8";
            // f16: |x| in [2^-6, 2^-1) with random sign and mantissa; q8:
            // random int8 with scales ~ 0.5 / 127 (KG = head_dim).
            let mut f16buf = || {
                g.upload(
                    &(0..rows * kvd)
                        .map(|_| {
                            let r = next();
                            ((r & 0x83ff) | ((9 + (r >> 16) % 5) << 10)) as u16
                        })
                        .collect::<Vec<u16>>(),
                )
                .unwrap()
            };
            let bufs: Vec<_> = if q8 {
                let mut i8buf = || {
                    g.upload(&(0..rows * kvd).map(|_| next() as u8).collect::<Vec<u8>>())
                        .unwrap()
                };
                let (a, b) = (i8buf(), i8buf());
                let sc = g.upload(&vec![0.5f32 / 127.0; rows * nkv]).unwrap();
                let sc2 = g.upload(&vec![0.4f32 / 127.0; rows * nkv]).unwrap();
                vec![a, b, sc, sc2]
            } else {
                vec![f16buf(), f16buf()]
            };
            let fa = km
                .function(&if q8 {
                    format!("attn_blk_b_q8_{hd}")
                } else {
                    format!("attn_blk_b_{hd}")
                })
                .unwrap();
            let fc = km.function("attn_cmb_b").unwrap();
            let kg = if q8 { hd } else { 0 };
            let fnb = km
                .function(&if q8 {
                    format!("attn_blk_nb_q8_{hd}")
                } else {
                    format!("attn_blk_nb_{hd}")
                })
                .unwrap();
            let nsm = batch::nb_smem(hd, kg);
            fnb.set_max_shared(nsm).unwrap();
            let fq = km.function("quantize_b").unwrap();
            let yq = g.alloc(nb_max * dim * 4).unwrap();
            for &nb in &nbs {
                let rp: Vec<u32> = (0..nb)
                    .flat_map(|z| [depth as u32, (z * ctx + depth) as u32])
                    .collect();
                let rp = g.upload(&rp).unwrap();
                // which: 0 attn only, 1 combine only, 2 both (batch.rs order).
                let step = |which: u32| {
                    for z0 in (0..nb).step_by(per) {
                        let nz = (nb - z0).min(per) as u32;
                        if which != 1 {
                            let mut a = vec![q.arg()];
                            a.extend(bufs.iter().map(|b| b.arg()));
                            a.extend([
                                Arg::I32(0),
                                Arg::I32(z0 as i32),
                                Arg::I32(nh as i32),
                                Arg::I32(nkv as i32),
                                pre,
                                Arg::I32(nblk_max as i32),
                                po.arg(),
                                pm.arg(),
                                rp.arg(),
                                Arg::Ptr(0), // no page table
                            ]);
                            // SAFETY: prefill.cu attn_blk_b(_q8)'s signature;
                            // the cache holds nb_max slots of ctx rows, part_*
                            // `per` queries of nblk_max blocks.
                            unsafe { g.launch(&fa, (nkv as u32, gy, nz), (32, 1, 1), 0, &a) }
                                .unwrap();
                        }
                        if which != 0 {
                            // SAFETY: prefill.cu attn_cmb_b's signature.
                            unsafe {
                                g.launch(
                                    &fc,
                                    (dim.div_ceil(256) as u32, nz, 1),
                                    (256, 1, 1),
                                    0,
                                    &[
                                        po.arg(),
                                        pm.arg(),
                                        Arg::I32(0),
                                        Arg::I32(z0 as i32),
                                        Arg::I32(dim as i32),
                                        Arg::I32(hd as i32),
                                        Arg::I32(nblk_max as i32),
                                        Arg::I32(0),
                                        act.arg(),
                                        Arg::U64(ast as u64),
                                        rp.arg(),
                                    ],
                                )
                            }
                            .unwrap();
                        }
                    }
                };
                // batch.rs's attn_blk_nb path (nb_split), y into its own
                // buffer (q stays the input).
                // None (batch.rs keeps attn_blk_b): time S = 1 anyway, the
                // kernel's own number, and say so.
                let picked = batch::nb_split(nb, nkv, gi.sms as usize, batch::nb_force_env());
                let sl = picked.unwrap_or(1);
                let step_nb = || {
                    let launch = |z0: usize, nz: usize, s: usize, y: Arg| {
                        let mut a = vec![q.arg()];
                        a.extend(bufs.iter().map(|b| b.arg()));
                        a.extend([
                            Arg::I32(0),
                            Arg::I32(z0 as i32),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            pre,
                            Arg::I32(nblk_max as i32),
                            po.arg(),
                            pm.arg(),
                            rp.arg(),
                            Arg::Ptr(0),
                            y,
                        ]);
                        // SAFETY: prefill.cu attn_blk_nb's signature; as above.
                        unsafe {
                            g.launch(
                                &fnb,
                                (nkv as u32, s as u32, nz as u32),
                                (128, 1, 1),
                                nsm as u32,
                                &a,
                            )
                        }
                        .unwrap();
                    };
                    if sl == 1 {
                        launch(0, nb, 1, yq.arg());
                        // SAFETY: prefill.cu quantize_b's signature.
                        unsafe {
                            g.launch(
                                &fq,
                                (dim.div_ceil(256) as u32, nb as u32, 1),
                                (32, 1, 1),
                                0,
                                &[
                                    yq.arg(),
                                    Arg::I32(dim as i32),
                                    Arg::I32(0),
                                    act.arg(),
                                    Arg::U64(ast as u64),
                                ],
                            )
                        }
                        .unwrap();
                    } else {
                        for z0 in (0..nb).step_by(per) {
                            let nz = (nb - z0).min(per);
                            launch(z0, nz, sl, Arg::Ptr(0));
                            // SAFETY: prefill.cu attn_cmb_b's signature.
                            unsafe {
                                g.launch(
                                    &fc,
                                    (dim.div_ceil(256) as u32, nz as u32, 1),
                                    (256, 1, 1),
                                    0,
                                    &[
                                        po.arg(),
                                        pm.arg(),
                                        Arg::I32(0),
                                        Arg::I32(z0 as i32),
                                        Arg::I32(dim as i32),
                                        Arg::I32(hd as i32),
                                        Arg::I32(nblk_max as i32),
                                        Arg::I32(0),
                                        act.arg(),
                                        Arg::U64(ast as u64),
                                        rp.arg(),
                                    ],
                                )
                            }
                            .unwrap();
                        }
                    }
                };
                let mut t = [vec![], vec![], vec![], vec![]];
                step(2);
                step_nb();
                for _ in 0..7 {
                    for w in 0..4u32 {
                        let e0 = g.record().unwrap();
                        if w == 3 {
                            step_nb();
                        } else {
                            step(w);
                        }
                        let e1 = g.record().unwrap();
                        t[w as usize].push(e0.ms_to(&e1).unwrap() as f64);
                    }
                }
                let md = |v: &mut Vec<f64>| {
                    v.sort_by(f64::total_cmp);
                    (v[v.len() / 2] * 1e3, v[0] * 1e3)
                };
                let (a, ab) = md(&mut t[0]);
                let (c, cb) = md(&mut t[1]);
                let (s, sb) = md(&mut t[2]);
                let (n, nbest) = md(&mut t[3]);
                // KV bytes the step must read: nb rows x (depth + 1)
                // positions x K and V.
                let per_pos = if q8 { 2 * (kvd + nkv * 4) } else { 2 * kvd * 2 };
                let gbs = (nb * (depth + 1) * per_pos) as f64 / (s * 1e-6) / 1e9;
                let gbn = (nb * (depth + 1) * per_pos) as f64 / (n * 1e-6) / 1e9;
                eprintln!(
                    "POOL kv {kv} depth {depth} nb {nb} launches {} attn {a:.1} ({ab:.1}) \
                     cmb {c:.1} ({cb:.1}) both {s:.1} ({sb:.1}) us/layer KV {gbs:.0} GB/s | \
                     nb S {sl}{} {n:.1} ({nbest:.1}) us KV {gbn:.0} GB/s speedup {:.2}x",
                    2 * nb.div_ceil(per),
                    if picked.is_none() {
                        " (batch.rs keeps attn_blk_b here)"
                    } else {
                        ""
                    },
                    s / n
                );
            }
        }
    }
}

/// attn_blk_nb (prefill.cu) against attn_blk_b + attn_cmb_b, the pool path
/// it replaces: act bit for bit, the fold in the block (S = 1, then
/// quantize_b) and slices into partials (S = 2, 3, 7, then attn_cmb_b),
/// f16 / q8 / q8g32, 128- and 64-wide heads, plain rows (rp) and paged
/// (a table per row, pages shuffled), ragged depths (1, 63, 64, 65, ..),
/// want_k off and on.
#[test]
fn cuda_attn_nb_exact() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    if !km.mma() {
        eprintln!("skipping: no tensor cores");
        return;
    }
    let sms = g.info().unwrap().sms as usize;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let depths = [0usize, 1, 63, 64, 65, 128, 129, 600, 1000, 1023, 1500];
    // GQA groups 7 (7B, 0.5B), 8 (every row of the m16 tile's top half
    // live: Qwen2.5-3B, Qwen3-30B) and 6 (1.5B).
    for (nh, nkv, hd) in [
        (28usize, 4usize, 128usize),
        (14, 2, 64),
        (16, 2, 128),
        (12, 2, 128),
    ] {
        let nb = 40;
        let dim = nh * hd;
        let kvd = nkv * hd;
        let per = batch::att_per_query(hd);
        let pos: Vec<usize> = (0..nb)
            .map(|z| {
                if z < depths.len() {
                    depths[z]
                } else {
                    (next() % 1600) as usize
                }
            })
            .collect();
        let ctx = (pos.iter().max().unwrap() + 1).next_multiple_of(64);
        let nblk_max = ctx / 64;
        let rows = nb * ctx;
        let qh: Vec<f32> = (0..nb * dim)
            .map(|_| (next() >> 40) as f32 / (1u64 << 22) as f32 - 2.0)
            .collect();
        let q = g.upload(&qh).unwrap();
        let po = g.alloc(per * nh * nblk_max * hd * 4).unwrap();
        let pm = g.alloc(per * nh * nblk_max * 2 * 4).unwrap();
        let ast = act_stride_t(dim);
        let act_ref = g.alloc(nb * ast).unwrap();
        let act = g.alloc(nb * ast).unwrap();
        let f16 = g
            .upload(
                &(0..2 * rows * kvd)
                    .map(|_| {
                        let r = next();
                        ((r & 0x83ff) | ((11 + (r >> 16) % 5) << 10)) as u16
                    })
                    .collect::<Vec<u16>>(),
            )
            .unwrap();
        let i8s = g
            .upload(
                &(0..2 * rows * kvd)
                    .map(|_| next() as u8)
                    .collect::<Vec<u8>>(),
            )
            .unwrap();
        let scs = g
            .upload(
                &(0..2 * rows * kvd / 32)
                    .map(|_| (1 + next() % 64) as f32 / 2048.0)
                    .collect::<Vec<f32>>(),
            )
            .unwrap();
        // Paged: row z's table at z * nblk_max, its pages a shuffle of the
        // slot's (any page of the cache would do).
        let mut tabv: Vec<u32> = (0..(nb * nblk_max) as u32).collect();
        for i in (1..tabv.len()).rev() {
            tabv.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        let tab = g.upload(&tabv).unwrap();
        for paged in [false, true] {
            let rpv: Vec<u32> = (0..nb)
                .flat_map(|z| {
                    let second = if paged {
                        z * nblk_max
                    } else {
                        z * ctx + pos[z]
                    };
                    [pos[z] as u32, second as u32]
                })
                .collect();
            let rp = g.upload(&rpv).unwrap();
            let tabarg = if paged { tab.arg() } else { Arg::Ptr(0) };
            for (mode, kg) in [("f16", 0usize), ("q8", hd), ("q8g32", 32)] {
                let kvargs: Vec<Arg> = if kg == 0 {
                    vec![f16.arg(), f16.arg_at(rows * kvd * 2)]
                } else {
                    vec![
                        i8s.arg(),
                        i8s.arg_at(rows * kvd),
                        scs.arg(),
                        scs.arg_at(rows * kvd / kg.max(1) * 4),
                    ]
                };
                let sfx = if kg == 0 {
                    format!("{hd}")
                } else {
                    format!("{mode}_{hd}")
                };
                let fb = km.function(&format!("attn_blk_b_{sfx}")).unwrap();
                let fnb = km.function(&format!("attn_blk_nb_{sfx}")).unwrap();
                let smem = batch::nb_smem(hd, kg);
                fnb.set_max_shared(smem).unwrap();
                let fc = km.function("attn_cmb_b").unwrap();
                let fq = km.function("quantize_b").unwrap();
                let pre = Arg::F32(att_pre(hd));
                let gy = ((8 * sms / nkv).clamp(1, MAX_SPLIT)).min(nblk_max) as u32;
                for want_k in [0, 1] {
                    let cmb = |z0: usize, nz: usize, out: &DevBuf| {
                        // SAFETY: prefill.cu attn_cmb_b's signature.
                        unsafe {
                            g.launch(
                                &fc,
                                (dim.div_ceil(256) as u32, nz as u32, 1),
                                (256, 1, 1),
                                0,
                                &[
                                    po.arg(),
                                    pm.arg(),
                                    Arg::I32(0),
                                    Arg::I32(z0 as i32),
                                    Arg::I32(dim as i32),
                                    Arg::I32(hd as i32),
                                    Arg::I32(nblk_max as i32),
                                    Arg::I32(want_k),
                                    out.arg(),
                                    Arg::U64(ast as u64),
                                    rp.arg(),
                                ],
                            )
                        }
                        .unwrap();
                    };
                    let head = |z0: usize, qa: Arg| {
                        let mut a = vec![qa];
                        a.extend_from_slice(&kvargs);
                        a.extend([
                            Arg::I32(0),
                            Arg::I32(z0 as i32),
                            Arg::I32(nh as i32),
                            Arg::I32(nkv as i32),
                            pre,
                            Arg::I32(nblk_max as i32),
                            po.arg(),
                            pm.arg(),
                            rp.arg(),
                            tabarg,
                        ]);
                        a
                    };
                    act_ref.zero().unwrap();
                    for z0 in (0..nb).step_by(per) {
                        let nz = (nb - z0).min(per);
                        // SAFETY: attn_blk_b(_q8)'s signature; buffers as sized above.
                        unsafe {
                            g.launch(
                                &fb,
                                (nkv as u32, gy, nz as u32),
                                (32, 1, 1),
                                0,
                                &head(z0, q.arg()),
                            )
                        }
                        .unwrap();
                        cmb(z0, nz, &act_ref);
                    }
                    let want = act_ref.read::<u8>().unwrap();
                    for s in [1usize, 2, 3, 7] {
                        act.zero().unwrap();
                        if s == 1 {
                            // y in place of q: on a copy, q stays the input.
                            let qcopy = g.upload(&qh).unwrap();
                            let mut a = head(0, qcopy.arg());
                            a.push(qcopy.arg());
                            // SAFETY: attn_blk_nb's signature; smem nb_smem.
                            unsafe {
                                g.launch(
                                    &fnb,
                                    (nkv as u32, 1, nb as u32),
                                    (128, 1, 1),
                                    smem as u32,
                                    &a,
                                )
                            }
                            .unwrap();
                            // SAFETY: prefill.cu quantize_b's signature.
                            unsafe {
                                g.launch(
                                    &fq,
                                    (dim.div_ceil(256) as u32, nb as u32, 1),
                                    (32, 1, 1),
                                    0,
                                    &[
                                        qcopy.arg(),
                                        Arg::I32(dim as i32),
                                        Arg::I32(want_k),
                                        act.arg(),
                                        Arg::U64(ast as u64),
                                    ],
                                )
                            }
                            .unwrap();
                        } else {
                            for z0 in (0..nb).step_by(per) {
                                let nz = (nb - z0).min(per);
                                let mut a = head(z0, q.arg());
                                a.push(Arg::Ptr(0));
                                // SAFETY: as above.
                                unsafe {
                                    g.launch(
                                        &fnb,
                                        (nkv as u32, s as u32, nz as u32),
                                        (128, 1, 1),
                                        smem as u32,
                                        &a,
                                    )
                                }
                                .unwrap();
                                cmb(z0, nz, &act);
                            }
                        }
                        let got = act.read::<u8>().unwrap();
                        let bad = (0..nb)
                            .find(|&z| got[z * ast..z * ast + ast] != want[z * ast..z * ast + ast]);
                        assert!(
                            bad.is_none(),
                            "{nh}/{nkv}x{hd} {mode} paged {paged} want_k {want_k} S {s}: row {:?} \
                             (position {:?}) differs from attn_blk_b + attn_cmb_b",
                            bad,
                            bad.map(|z| pos[z])
                        );
                    }
                }
            }
        }
        eprintln!(
            "attn_blk_nb {nh}/{nkv}x{hd}: f16 / q8 / q8g32, plain and paged, S 1/2/3/7 = attn_blk_b"
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
        let fm = km.function(&format!("attn_blk_mq_{q8n}{hd}")).unwrap();
        fm.set_max_shared(batch::mq_smem(hd, kg)).unwrap();
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
            // attn_blk_mq: the same queries, MQ_Q per block (a partial last
            // one), the same partials for attn_cmb_b.
            let mpo = g.alloc(nz * po_n * 4).unwrap();
            let mpm = g.alloc(nz * pm_n * 4).unwrap();
            let mact = g.alloc(nb * ast).unwrap();
            mact.zero().unwrap();
            // SAFETY: decode.cu attn_blk_mq's signature; buffers as above.
            unsafe {
                g.launch(
                    &fm,
                    (
                        nkv as u32,
                        gy.min(nblk_max) as u32,
                        nz.div_ceil(batch::MQ_Q) as u32,
                    ),
                    (32 * group as u32, 1, 1),
                    batch::mq_smem(hd, kg) as u32,
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
                            mpo.arg(),
                            mpm.arg(),
                            Arg::Ptr(0), // no page table
                            Arg::I32(nz as i32),
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
                        mpo.arg(),
                        mpm.arg(),
                        Arg::I32(start as i32),
                        Arg::I32(z0 as i32),
                        Arg::I32(dim as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(nblk_max as i32),
                        want_k,
                        mact.arg(),
                        Arg::U64(ast as u64),
                        Arg::Ptr(0),
                    ],
                )
                .unwrap();
            }
            let (mpo, mpm, mact): (Vec<f32>, Vec<f32>, Vec<u8>) = (
                mpo.read().unwrap(),
                mpm.read().unwrap(),
                mact.read().unwrap(),
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
                for (kname, po, pm, pact) in [
                    ("attn_blk_b", &bpo, &bpm, &bact),
                    ("attn_blk_mq", &mpo, &mpm, &mact),
                ] {
                    for h in 0..nh {
                        for b in 0..np.div_ceil(64) {
                            let o = (h * nblk_max + b) * hd;
                            assert_eq!(
                                bits(&po[zl * po_n + o..zl * po_n + o + hd]),
                                bits(&wpo[o..o + hd]),
                                "{}: head {h} block {b} sums",
                                ctxs(kname, z)
                            );
                            let m = (h * nblk_max + b) * 2;
                            assert_eq!(
                                bits(&pm[zl * pm_n + m..zl * pm_n + m + 2]),
                                bits(&wpm[m..m + 2]),
                                "{}: head {h} block {b} m, l",
                                ctxs(kname, z)
                            );
                        }
                    }
                    assert!(
                        pact[z * ast..z * ast + ab] == wact[..],
                        "{}",
                        ctxs(&format!("{kname} act"), z)
                    );
                }
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

impl<'g> crate::lm::Sealed for KvNet<'_, 'g> {}
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
        a.tally(&lr, &lo);
    }
}

/// `compare_forced(r, o1, ..)` and `compare_forced(r, o2, ..)` in one
/// pass: the same calls on each model, the reference's run once (the
/// teacher-forced inputs don't depend on any model's output).
#[allow(clippy::too_many_arguments)]
fn compare_forced2<R: Lm, O1: Lm, O2: Lm>(
    r: &R,
    o1: &O1,
    o2: &O2,
    ids: &[u32],
    from: usize,
    a1: &mut Agree,
    a2: &mut Agree,
) {
    let mut rc = r.new_cache(ids.len());
    let mut c1 = o1.new_cache(ids.len());
    let mut c2 = o2.new_cache(ids.len());
    for (i, ch) in ids[..from].chunks(512).enumerate().filter(|_| from > 0) {
        r.feed_batch(ch, i * 512, &mut rc);
        o1.feed_batch(ch, i * 512, &mut c1);
        o2.feed_batch(ch, i * 512, &mut c2);
    }
    a1.last_kl = 0.0;
    a2.last_kl = 0.0;
    for (pos, &t) in ids.iter().enumerate().skip(from) {
        let lr = r.forward(t, pos, &mut rc);
        a1.tally(&lr, &o1.forward(t, pos, &mut c1));
        a2.tally(&lr, &o2.forward(t, pos, &mut c2));
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
    let cpu = Qwen2::new_with(&m, true);
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
    // AXON_TF_DEV=moe: the MoE backend (crate::moe: attention on the GPU,
    // experts on the CPU).
    if var("AXON_TF_DEV") == "moe" {
        type Top = Vec<(usize, f64)>;
        fn rows<L: Lm>(net: &L, seq: &[u32], n: usize, top: &dyn Fn(&[f32]) -> Top) -> Vec<Top> {
            let mut c = net.new_cache(seq.len() + 1);
            net.feed_batch(&seq[..n - 1], 0, &mut c);
            let mut r = Vec::new();
            for (i, &id) in seq.iter().enumerate().skip(n - 1) {
                r.push(top(&net.forward(id, i, &mut c)));
            }
            r
        }
        let g = crate::testutil::gpu().unwrap();
        let km = Kernels::module_for(&g, &m).unwrap();
        let mm = Kernels::moe_module(&g).unwrap();
        let net = crate::moe::tests::moe_net(&g, &km, &mm, &m, max_ctx);
        let out: Vec<_> = (inp["gens"].as_array().unwrap().iter())
            .map(|gn| {
                let (p, t) = (ids(&gn["prompt_ids"]), ids(&gn["tokens"]));
                let seq: Vec<u32> = p.iter().chain(&t).copied().collect();
                rows(&net, &seq, p.len(), &top)
            })
            .collect();
        std::fs::write(var("AXON_TF_OUT"), serde_json::to_string(&out).unwrap()).unwrap();
        return;
    }
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

/// The test-only gemmh2 variants (`gemmh2_variant_src`); never built by a
/// product path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum G2Var {
    /// DELIBERATELY INEXACT: no per-256 drain (the contract's upper bound).
    NoDrain,
    /// Exact by construction (to be proven by the bits test on sm_90a): S
    /// converted with the magic-number form instead of I2F.
    Magic,
}

impl G2Var {
    const ALL: [G2Var; 2] = [G2Var::NoDrain, G2Var::Magic];
    /// Kernel name infix: gemmh2<tag>_q4_k, gemmh2<tag>_q4_k_gu.
    fn tag(self) -> &'static str {
        match self {
            G2Var::NoDrain => "nd",
            G2Var::Magic => "mg",
        }
    }
    fn name(self) -> &'static str {
        match self {
            G2Var::NoDrain => "nodrain",
            G2Var::Magic => "magic",
        }
    }
    /// The variant a kernel name belongs to (None: the real kernels).
    fn of(kernel: &str) -> Option<G2Var> {
        G2Var::ALL
            .into_iter()
            .find(|v| kernel.starts_with(&format!("gemmh2{}_", v.tag())))
    }
}

/// TEST-ONLY kernel sources: KERNELS with hopper.cu's `gemmh2_t` patched,
/// its two kernels renamed gemmh2<tag>_q4_k / gemmh2<tag>_q4_k_gu (so a
/// profiler can tell them apart). They live only here (`#[cfg(test)]`): no
/// product path builds them, and the kernel cache is keyed by the source,
/// so their cubins can never be loaded for the real kernels. Panics if
/// hopper.cu no longer has the patched lines exactly once
/// (`gemmh2_variant_patches_apply` catches that on any machine).
///
/// - `NoDrain`, DELIBERATELY INEXACT: the S (dlo / dhi) wgmma chains
///   accumulate over the whole K in int32 (scale-d 0 only on block 0's first
///   k32), M by dp4a likewise, and the float epilogue (d S - dmin M, the fma
///   into acc) runs once, on the last block's scales. Its results are wrong
///   by design; its speed is the upper bound of what the per-256 drain (the
///   bit-identity contract's only Hopper-specific cost, outside review 10-19
///   §2.3) costs. It still waits for each block's wgmmas (the A fragments'
///   registers are refilled for the next block), so it bounds the drain's
///   arithmetic, not the wait.
/// - `Magic` (research hopper-exact-w4a8 §3.3, E0' (a)): the sub-block
///   scale split 3 + 3 instead of 4 + 2 (sc = lo3 + 8 hi3: pre-scaled quants
///   q lo3, q hi3 <= 105, so |Dlo|, |Dhi| <= 8 * 7 * 32 * 15 * 127 =
///   3,413,760 < 2^22), S = 8 Dhi + Dlo the same integer as before, and
///   rn(S) as fadd(fsub(bits(0x4CC00000 + Dhi), 1.5 * 2^26), fsub(bits(
///   0x4B400000 + Dlo), 1.5 * 2^23)): both terms are exact floats (8 Dhi and
///   Dlo), so the one rounding fadd is rn(S) = cvt.rn.f32.s32(S), with no
///   I2F (the conversion pipe's 16 results per clock per SM). (The research
///   note's 0x4D400000 is 1.5 * 2^27, ulp 16: wrong; `magic_i2f_is_cvt_rn`.)
///   Its bits must equal the real kernel's: `gpu_gemmh2_magic_matches_reference_bits`.
fn gemmh2_variant_src(v: G2Var) -> String {
    let src = crate::cuda::KERNELS;
    let a = src
        .find("__device__ __forceinline__ void gemmh2_t(")
        .expect("hopper.cu: gemmh2_t");
    let b = a + src[a..]
        .find("gemmh2_q4_k(const __grid_constant__")
        .expect("hopper.cu: gemmh2_q4_k");
    let mut body = src[a..b].to_string();
    let edits: &[(&str, &str)] = match v {
        G2Var::NoDrain => &[
            (
                "wg_mma_rs_n64_u8(dlo, alo[j], wg_desc(b0 + (j >> 2) * 8192) + 2 * (j & 3), j);",
                "wg_mma_rs_n64_u8(dlo, alo[j], wg_desc(b0 + (j >> 2) * 8192) + 2 * (j & 3), (b | j) != 0);",
            ),
            (
                "wg_mma_rs_n64_u8(dhi, ahi[j], wg_desc(b0 + (j >> 2) * 8192) + 2 * (j & 3), j);",
                "wg_mma_rs_n64_u8(dhi, ahi[j], wg_desc(b0 + (j >> 2) * 8192) + 2 * (j & 3), (b | j) != 0);",
            ),
            (
                "acc[i] = 0.0f, dlo[i] = 0, dhi[i] = 0;",
                "acc[i] = 0.0f, dlo[i] = 0, dhi[i] = 0, M[i] = 0;",
            ),
            (
                "M[4 * i + e] = ml + 64 * mh;",
                "M[4 * i + e] += ml + 64 * mh;",
            ),
            (
                "int h = e >> 1;\n                int S = 16 * dhi",
                "if (b != nblk - 1) continue; // NODRAIN: one epilogue\n                int h = e >> 1;\n                int S = 16 * dhi",
            ),
        ],
        G2Var::Magic => &[
            (
                "alo[j][h] = v0 * (scj & 15);",
                "alo[j][h] = v0 * (scj & 7);",
            ),
            (
                "alo[j][2 + h] = v1 * (scj & 15);",
                "alo[j][2 + h] = v1 * (scj & 7);",
            ),
            (
                "ahi[j][h] = v0 * (scj >> 4);",
                "ahi[j][h] = v0 * (scj >> 3);",
            ),
            (
                "ahi[j][2 + h] = v1 * (scj >> 4);",
                "ahi[j][2 + h] = v1 * (scj >> 3);",
            ),
            (
                "int S = 16 * dhi[4 * i + e] + dlo[4 * i + e];\n                float v = __fsub_rn(__fmul_rn(dr[h], i2f_rn(S)),",
                "float fS = __fadd_rn(\n                    __fsub_rn(__int_as_float(0x4CC00000 + dhi[4 * i + e]), 100663296.0f),\n                    __fsub_rn(__int_as_float(0x4B400000 + dlo[4 * i + e]), 12582912.0f));\n                float v = __fsub_rn(__fmul_rn(dr[h], fS),",
            ),
        ],
    };
    for (from, to) in edits {
        assert_eq!(body.matches(from).count(), 1, "hopper.cu gemmh2_t: {from}");
        body = body.replacen(from, to, 1);
    }
    let mut tail = src[b..].to_string();
    for k in [
        "gemmh2_q4_k(const __grid_constant__",
        "gemmh2_q4_k_gu(const __grid_constant__",
    ] {
        assert_eq!(tail.matches(k).count(), 1, "hopper.cu: {k}");
        tail = tail.replacen(
            k,
            &k.replacen("gemmh2_", &format!("gemmh2{}_", v.tag()), 1),
            1,
        );
    }
    format!(
        "{}// AXON TEST-ONLY: gemmh2_t variant {} (tests.rs gemmh2_variant_src)\n{body}{tail}",
        &src[..a],
        v.name()
    )
}

/// `gemmh2_variant_src(v)` compiled as module_ext is (sm_90a only).
fn gemmh2_variant_module(g: &Gpu, v: G2Var) -> Module<'_> {
    let src = gemmh2_variant_src(v);
    let m = Module::load(g, &src, &crate::cuda::KERNEL_OPTS_EXT)
        .unwrap()
        .0;
    assert!(m.hopper(), "the {} module isn't sm_90a", v.name());
    m
}

/// The variant patches still apply to hopper.cu (no GPU needed) and change
/// only gemmh2_t and the two kernels' names. AXON_VARIANT_SRC_DIR=DIR
/// writes them there as gemmh2-<name>.cu (the H100 kit compiles them
/// offline into the pod's kernel cache).
#[test]
fn gemmh2_variant_patches_apply() {
    let a = crate::cuda::KERNELS
        .find("__device__ __forceinline__ void gemmh2_t(")
        .unwrap();
    for v in G2Var::ALL {
        let p = gemmh2_variant_src(v);
        if let Some(d) = std::env::var_os("AXON_VARIANT_SRC_DIR") {
            let f = std::path::Path::new(&d).join(format!("gemmh2-{}.cu", v.name()));
            std::fs::write(f, &p).unwrap();
        }
        assert_eq!(&p[..a], &crate::cuda::KERNELS[..a]);
        let k = format!("gemmh2{}_q4_k(const", v.tag());
        assert_eq!(p.matches(&k).count(), 1, "{v:?}");
        assert!(!p.contains("\ngemmh2_q4_k(const"), "{v:?}");
        match v {
            G2Var::NoDrain => assert_eq!(p.matches("(b | j) != 0").count(), 2),
            G2Var::Magic => assert_eq!(p.matches("0x4CC00000 + dhi").count(), 1),
        }
    }
}

/// `G2Var::Magic`'s conversion, emulated on the host (IEEE f32, round to
/// nearest even, no contraction): rn(8 hi + lo) for |hi|, |lo| < 2^22.
fn magic_s(hi: i32, lo: i32) -> f32 {
    let fh = f32::from_bits((0x4CC0_0000i32 + hi) as u32) - 100_663_296.0;
    let fl = f32::from_bits((0x4B40_0000i32 + lo) as u32) - 12_582_912.0;
    fh + fl
}

/// The magic-number conversion equals `as f32` (= cvt.rn.f32.s32) for
/// every Dhi with Dlo at its edges and a few others, and every Dlo with Dhi
/// likewise, over the whole |x| < 2^22 range (Q4_K's 3 + 3 split needs
/// 3,413,760), including |S| >= 2^24 where the conversion rounds. No GPU.
#[test]
fn magic_i2f_is_cvt_rn() {
    const R: i32 = (1 << 22) - 1;
    let edge = [
        -R,
        -R + 1,
        -3_413_760,
        -12_345,
        -1,
        0,
        1,
        7,
        3_413_760,
        R - 1,
        R,
    ];
    let mut rng = Rng(0x3a61c);
    let mut picks: Vec<i32> = edge.to_vec();
    picks.extend((0..5).map(|_| (rng.next() % (2 * R as u64 + 1)) as i32 - R));
    let mut n = 0u64;
    for x in -R..=R {
        for &y in &picks {
            for (hi, lo) in [(x, y), (y, x)] {
                let want = (8 * hi + lo) as f32;
                assert_eq!(magic_s(hi, lo).to_bits(), want.to_bits(), "hi {hi} lo {lo}");
                n += 1;
            }
        }
    }
    // The research note's constant (0x4D400000 = 1.5 * 2^27, ulp 16) is not it.
    let doc = f32::from_bits((0x4D40_0000i32 + 1) as u32) - 100_663_296.0;
    assert_ne!(doc, 8.0);
    eprintln!("magic I2F == cvt.rn on {n} pairs");
}

/// The magic conversion and I2F on the device, and an I2F / magic
/// throughput probe (`cuda_i2f_bench`).
const MAGIC_SRC: &str = r#"
__device__ __forceinline__ float magic_s(int hi, int lo) {
    return __fadd_rn(__fsub_rn(__int_as_float(0x4CC00000 + hi), 100663296.0f),
                     __fsub_rn(__int_as_float(0x4B400000 + lo), 12582912.0f));
}
// Thread t: hi from a stride walk over [-R, R], lo over 16 values (edges and
// a walk); mismatches counted, the first one kept.
extern "C" __global__ void magic_check(int n, unsigned* bad, int* first) {
    const int R = (1 << 22) - 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int hi = (int)(((long long)t * 2 * R) / (n - 1)) - R;
    const int edge[8] = {-R, -3413760, -1, 0, 1, 7, 3413760, R};
    for (int k = 0; k < 16; k++) {
        int lo = k < 8 ? edge[k] : (int)(((unsigned)t * 2654435761u + k * 40503u) % (2u * R + 1)) - R;
        for (int s = 0; s < 2; s++) {
            int a = s ? lo : hi, b = s ? hi : lo;
            float w = __int2float_rn(8 * a + b), m = magic_s(a, b);
            if (__float_as_uint(w) != __float_as_uint(m) && atomicAdd(bad, 1u) == 0) {
                first[0] = a;
                first[1] = b;
            }
        }
    }
}
// One block per SM: iters x 8 conversions per thread (8 independent
// chains), I2F (mode 0) or the magic form (1); the block's clock64 span.
extern "C" __global__ void i2f_tp(int iters, int mode, float* out, long long* clk) {
    int x[8];
    float acc[8];
    for (int k = 0; k < 8; k++) x[k] = threadIdx.x * 8 + k, acc[k] = 0.0f;
    __syncthreads();
    long long t0 = clock64();
    if (mode == 0) {
        for (int i = 0; i < iters; i++)
#pragma unroll
            for (int k = 0; k < 8; k++) acc[k] = __fadd_rn(acc[k], __int2float_rn(x[k] += 3));
    } else {
        for (int i = 0; i < iters; i++)
#pragma unroll
            for (int k = 0; k < 8; k++)
                acc[k] = __fadd_rn(acc[k], __fsub_rn(__int_as_float(0x4B400000 + (x[k] += 3)), 12582912.0f));
    }
    __syncthreads();
    long long t1 = clock64();
    float s = 0.0f;
    for (int k = 0; k < 8; k++) s += acc[k];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (threadIdx.x == 0) clk[blockIdx.x] = t1 - t0;
}
"#;

/// `magic_i2f_is_cvt_rn` on the device (any GPU: the ops are plain IEEE
/// f32): 2^22 values of Dhi x 16 of Dlo, both ways round, bit for bit
/// against __int2float_rn.
#[test]
fn gpu_magic_i2f_is_cvt_rn() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let m = Module::load(&g, MAGIC_SRC, &crate::cuda::KERNEL_OPTS)
        .unwrap()
        .0;
    let f = m.function("magic_check").unwrap();
    let n = 1usize << 22;
    let bad = g.alloc(4).unwrap();
    bad.zero().unwrap();
    let first = g.alloc(8).unwrap();
    first.zero().unwrap();
    // SAFETY: magic_check's signature; bad 1 u32, first 2 i32.
    unsafe {
        g.launch(
            &f,
            (n.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &[Arg::I32(n as i32), bad.arg(), first.arg()],
        )
        .unwrap();
    }
    g.sync().unwrap();
    let b: Vec<u32> = bad.read().unwrap();
    let fi: Vec<i32> = first.read().unwrap();
    assert_eq!(
        b[0], 0,
        "magic != cvt.rn, first at hi {} lo {}",
        fi[0], fi[1]
    );
    eprintln!("magic I2F == __int2float_rn on {} pairs (device)", n * 32);
}

/// I2F vs the magic form, results per clock per SM (one block of 1024
/// threads per SM, 8 chains each, clock64 per block; median of 5). The
/// research note cites 16 I2F per clock per SM on cc 8.x / 9.0; FADD-class
/// is 128. Run with --ignored.
#[test]
#[ignore]
fn cuda_i2f_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let m = Module::load(&g, MAGIC_SRC, &crate::cuda::KERNEL_OPTS)
        .unwrap()
        .0;
    let f = m.function("i2f_tp").unwrap();
    let info = g.info().unwrap();
    let (blocks, threads, iters) = (info.sms as usize, 1024usize, 4096usize);
    let out = g.alloc(blocks * threads * 4).unwrap();
    let clk = g.alloc(blocks * 8).unwrap();
    for (mode, what) in [(0, "I2F"), (1, "magic")] {
        let mut v: Vec<f64> = (0..5)
            .map(|_| {
                // SAFETY: i2f_tp's signature; out blocks x threads floats, clk blocks i64.
                unsafe {
                    g.launch(
                        &f,
                        (blocks as u32, 1, 1),
                        (threads as u32, 1, 1),
                        0,
                        &[Arg::I32(iters as i32), Arg::I32(mode), out.arg(), clk.arg()],
                    )
                    .unwrap();
                }
                g.sync().unwrap();
                let c: Vec<u64> = clk.read().unwrap();
                let mean = c.iter().sum::<u64>() as f64 / c.len() as f64;
                (threads * iters * 8) as f64 / mean
            })
            .collect();
        v.sort_by(f64::total_cmp);
        eprintln!(
            "I2FBENCH {} sm_{}{} {what}: {:.1} results/clk/SM (median of 5; + 1 FADD and 1 IADD each)",
            info.name, info.cc.0, info.cc.1, v[2]
        );
    }
}

/// gemmh2_q4_k and gemmh2_q4_k_gu in `G2Var::Magic` against gemv_ref bit
/// for bit (the real kernels' test cases, trimmed), plus an adversarial
/// fixture at the bounds the proof uses: every scale and min 63, every
/// quant 15, activations +-127 (|S| up to 30.7M > 2^24, where the
/// conversion rounds). Hopper (sm_90a) only.
#[test]
fn gpu_gemmh2_magic_matches_reference_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module_ext(&g).unwrap();
    if !km.hopper() {
        eprintln!("skipping: needs an sm_90a module (compute capability 9.0)");
        return;
    }
    let kv = gemmh2_variant_module(&g, G2Var::Magic);
    let mut rng = Rng(0x3a61_c0de_0090);
    // Q4_K, every byte of scales and quants 0xff (sc = m = 63, q = 15), d =
    // dmin = 1.0 (f16 0x3c00).
    let adversarial = |rows: usize, cols: usize| -> Vec<u8> {
        let blk: Vec<u8> = [0x00u8, 0x3c, 0x00, 0x3c]
            .into_iter()
            .chain([0xffu8; 140])
            .collect();
        blk.repeat(rows * cols / 256)
    };
    let cases: [(usize, usize, &[usize], bool); 7] = [
        (64, 256, &[1, 17, 65], false),
        (333, 1536, &[17, 129], false),
        (3584, 3584, &[64], false),
        (18944, 3584, &[129], false),
        (3584, 18944, &[65], false),
        (192, 3584, &[66], true),
        (128, 18944, &[64], true),
    ];
    for (rows, cols, nbs, adv) in cases {
        let wd = match adv {
            true => adversarial(rows, cols),
            false => synth_matrix(&mut rng, Q4_K, rows, cols),
        };
        let w = g.upload(&wd).unwrap();
        for &nb in nbs {
            let xs: Vec<Vec<f32>> = (0..nb)
                .map(|i| match adv {
                    // all +1 / all -1 (q = +-127), then alternating signs
                    true => match i % 3 {
                        0 => vec![1.0; cols],
                        1 => vec![-1.0; cols],
                        _ => (0..cols).map(|c| [1.0f32, -1.0][c % 2]).collect(),
                    },
                    false => rng.vec(cols, [1.0f32, 1e-3, 40.0][i % 3]),
                })
                .collect();
            let epi = match nb % 2 {
                0 => Epi::default(),
                _ => Epi {
                    bias: Some(rng.vec(rows, 3.0)),
                    init: None,
                },
            };
            let act = quantize_batch(&g, &km, &xs, cols);
            let want = gpu_batched(&g, &km, "gemv_ref", Q4_K, &w, rows, cols, &act, nb, &epi);
            let out = g.alloc(rows * nb * 4).unwrap();
            out.zero().unwrap();
            let bias = epi.bias.as_ref().map(|b| g.upload(b).unwrap());
            // SAFETY: w rows x cols Q4_K, act nb tokens, out rows * nb.
            unsafe {
                launch_gemmh(
                    &g,
                    &kv,
                    "gemmh2mg_q4_k",
                    &w,
                    rows,
                    cols,
                    &act,
                    act_stride_t(cols),
                    nb,
                    bias.as_ref().map_or(Arg::Ptr(0), |b| b.arg()),
                    0,
                    &out,
                )
            };
            g.sync().unwrap();
            let got: Vec<f32> = out.read().unwrap();
            if let Some(i) = (0..rows * nb).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                panic!(
                    "gemmh2mg_q4_k {cols}x{rows} nb {nb} adv {adv} token {} row {}: {} vs ref {}",
                    i / rows,
                    i % rows,
                    got[i],
                    want[i]
                );
            }
            eprintln!("MAGICBITS Q4_K {cols}x{rows} nb {nb} adversarial {adv}: bit-identical");
        }
    }
    // The fused gate + up: silu(gate) * up against gemv_ref's mode 2.
    for (cols, rows, adv) in [(3584usize, 1000usize, false), (3584, 256, true)] {
        let mk = |rng: &mut Rng| match adv {
            true => adversarial(rows, cols),
            false => synth_matrix(rng, Q4_K, rows, cols),
        };
        let (gd, ud) = (mk(&mut rng), mk(&mut rng));
        let (wg, wu) = (g.upload(&gd).unwrap(), g.upload(&ud).unwrap());
        let nb = 100;
        let xs: Vec<Vec<f32>> = (0..nb)
            .map(|i| match adv {
                true => vec![[1.0f32, -1.0][i % 2]; cols],
                false => rng.vec(cols, [1.0f32, 1e-2, 4.0][i % 3]),
            })
            .collect();
        let act = quantize_batch(&g, &km, &xs, cols);
        let gate = gpu_batched(
            &g,
            &km,
            "gemv_ref",
            Q4_K,
            &wg,
            rows,
            cols,
            &act,
            nb,
            &Epi::default(),
        );
        let gate_b = g.upload(&gate).unwrap();
        let want = {
            let out = g.alloc(rows * nb * 4).unwrap();
            out.zero().unwrap();
            let f = km.function("gemv_ref").unwrap();
            let common = [
                wu.arg(),
                Arg::U64(quant::row_bytes(Q4_K, cols) as u64),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
                act.arg(),
                Arg::U64(act_stride_t(cols) as u64),
                Arg::I32(nb as i32),
                gate_b.arg(),
                Arg::I32(2),
                out.arg(),
                Arg::U64(rows as u64),
            ];
            // SAFETY: gemv_ref's signature; act nb tokens, out and gate rows * nb.
            unsafe { launch_product(&g, &f, "gemv_ref", Q4_K, rows, nb, &common) };
            g.sync().unwrap();
            out.read::<f32>().unwrap()
        };
        let out = g.alloc(rows * nb * 4).unwrap();
        // SAFETY: wg, wu rows x cols Q4_K, act nb tokens, out rows * nb.
        unsafe {
            launch_gemmh_gu(
                &g,
                &kv,
                "gemmh2mg_q4_k_gu",
                &wg,
                &wu,
                rows,
                cols,
                &act,
                act_stride_t(cols),
                nb,
                &out,
            )
        };
        g.sync().unwrap();
        let got: Vec<f32> = out.read().unwrap();
        if let Some(i) = (0..rows * nb).find(|&i| got[i].to_bits() != want[i].to_bits()) {
            panic!(
                "gemmh2mg_q4_k_gu {cols}x{rows} adv {adv} token {} row {}: {} vs ref {}",
                i / rows,
                i % rows,
                got[i],
                want[i]
            );
        }
        eprintln!("MAGICBITS gate+up {cols}x{rows} nb {nb} adversarial {adv}: bit-identical");
    }
}

/// gemmh vs gemm2 on the 7B's Q4_K shapes: interleaved rounds (clocks
/// drift), each kernel's median / min / max TMAC/s of real MACs (rows x
/// cols x nb). AXON_BENCH_NB (default 1050), AXON_BENCH_ROUNDS (default 7).
/// gemm2s, gemm2, gemm2t everywhere; on Hopper (sm_90a) the gemmh kernels
/// too and the `G2Var` variants: gemmh2nd_q4_k[_gu] (no drain, the INEXACT
/// upper bound) and gemmh2mg_q4_k[_gu] (the magic-number S conversion).
/// AXON_BENCH_VARIANTS (default "nodrain,magic"; "" none),
/// AXON_BENCH_SHAPE (a substring of the shapes' names, e.g. "gate"; the
/// fused gate + up is "gate+up"), AXON_BENCH_KERNELS (kernel name prefix,
/// e.g. "gemmh2": what ncu profiles). Run with --ignored.
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
    // The variants' modules (Hopper only): the same options as module_ext.
    let want_v = std::env::var("AXON_BENCH_VARIANTS").unwrap_or("nodrain,magic".into());
    let vars: Vec<(G2Var, Module)> = (G2Var::ALL.into_iter())
        .filter(|v| km.hopper() && want_v.split(',').any(|w| w.trim() == v.name()))
        .map(|v| {
            eprintln!(
                "variant module {}: gemmh2{}_q4_k[_gu]{}",
                v.name(),
                v.tag(),
                match v {
                    G2Var::NoDrain => " without the per-256 drain (INEXACT, bench only)",
                    G2Var::Magic =>
                        " with the magic-number S conversion (exact if MAGICBITS passed)",
                }
            );
            (v, gemmh2_variant_module(&g, v))
        })
        .collect();
    let module_of = |k: &str| match G2Var::of(k) {
        Some(v) => &vars.iter().find(|x| x.0 == v).unwrap().1,
        None => &km,
    };
    let shape_ok = |s: &str| std::env::var("AXON_BENCH_SHAPE").map_or(true, |f| s.contains(&f));
    let kpre = std::env::var("AXON_BENCH_KERNELS").unwrap_or_default();
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
        if !shape_ok(what) || what.contains("gate+up") {
            continue;
        }
        let w = g.upload(&synth_matrix(&mut rng, Q4_K, rows, cols)).unwrap();
        for &nb in &nbs {
            let xs: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 1.0)).collect();
            let act = quantize_batch(&g, &km, &xs, cols);
            let st = act_stride_t(cols);
            let out = g.alloc(rows * nb * 4).unwrap();
            let mut kernels: Vec<String> = ["gemm2s_q4_k", "gemm2_q4_k", "gemm2t_q4_k"]
                .map(String::from)
                .to_vec();
            if km.hopper() {
                kernels.extend(["gemmh_q4_k", "gemmh1_q4_k", "gemmh2_q4_k"].map(String::from));
            }
            kernels.extend(vars.iter().map(|v| format!("gemmh2{}_q4_k", v.0.tag())));
            kernels.retain(|k| k.starts_with(&kpre));
            let mut ms: Vec<Vec<f64>> = vec![Vec::new(); kernels.len()];
            for _ in 0..rounds {
                for (ki, k) in kernels.iter().enumerate() {
                    let run = || {
                        if k.starts_with("gemmh") {
                            // SAFETY: as in gpu_gemmh_matches_reference_bits.
                            unsafe {
                                launch_gemmh(
                                    &g,
                                    module_of(k),
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
    if !shape_ok("gate+up") {
        return;
    }
    let (cols, rows) = (3584usize, 18944usize);
    let wg = g.upload(&synth_matrix(&mut rng, Q4_K, rows, cols)).unwrap();
    let wu = g.upload(&synth_matrix(&mut rng, Q4_K, rows, cols)).unwrap();
    for &nb in &nbs {
        let xs: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 1.0)).collect();
        let act = quantize_batch(&g, &km, &xs, cols);
        let st = act_stride_t(cols);
        let out = g.alloc(rows * nb * 4).unwrap();
        let mut kernels: Vec<String> = vec!["gemm2s_q4_k_gu".into()];
        if km.hopper() {
            kernels.push("gemmh2_q4_k_gu".into());
        }
        kernels.extend(vars.iter().map(|v| format!("gemmh2{}_q4_k_gu", v.0.tag())));
        kernels.retain(|k| k.starts_with(&kpre));
        let f2 = km.function("gemm2s_q4_k_gu").unwrap();
        f2.set_max_shared(gemm2_smem(Q4_K, 64, 3)).unwrap();
        let mut ms: Vec<Vec<f64>> = vec![Vec::new(); kernels.len()];
        for _ in 0..rounds {
            for (ki, k) in kernels.iter().enumerate() {
                let run = || {
                    if k.starts_with("gemmh") {
                        // SAFETY: as in gpu_gemmh_gate_up_match_reference.
                        unsafe {
                            launch_gemmh_gu(
                                &g,
                                module_of(k),
                                k,
                                &wg,
                                &wu,
                                rows,
                                cols,
                                &act,
                                st,
                                nb,
                                &out,
                            )
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
            unsafe {
                launch_gemmh_gu(
                    &g,
                    &km,
                    "gemmh2_q4_k_gu",
                    &wg,
                    &wu,
                    rows,
                    cols,
                    &act,
                    st,
                    nb,
                    &out,
                )
            };
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
    kernel: &str,
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
    let f = km.function(kernel).unwrap();
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
    assert_ref_spread(ac);
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
    let cpu = Qwen2::new_with(m, true);
    const STEPS: usize = 32;
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    let mut seqs = Vec::new();
    for p in PROMPTS {
        let ids = tok.encode(p);
        let t = trace(&cpu, &ids, STEPS);
        compare_trace(&t, &gpu, &mut ag);
        compare_trace(&t, &rows, &mut ac);
        // compare's position s of this prompt follows ids + the CPU's
        // first s greedy picks (the tokens the trace fed).
        seqs.push((ids, t.picks()));
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
    let cpu = Qwen2::new_with(&m, true);
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
        let cpu = Qwen2::new_with(&m, true);
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
    let cpu = Qwen2::new_with(&m, true);
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
    let cpu = Qwen2::new_with(m, true);
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
    let cpu = Qwen2::new_with(&m, true);
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
        compare_each!(&cpu, &ids, 8, (&gpu, &mut ag), (&rows, &mut ac));
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

/// `crate::lm::chunk_ranges`: 0..n in order, pieces of pb, the last shorter.
#[test]
fn lm_chunk_ranges() {
    use crate::lm::chunk_ranges;
    assert!(chunk_ranges(0, 4).is_empty());
    assert_eq!(chunk_ranges(2100, 1152), vec![0..1152, 1152..2100]);
    assert_eq!(chunk_ranges(5, 2), vec![0..2, 2..4, 4..5]);
}

/// `Lm::prompt_chunks` on the GPU (what serve's single stream and chat feed
/// by): `prompt_passes`, contiguous, none over the batch max; and serve's
/// way of feeding (all but the last token by those chunks, then `forward`)
/// and chat's (the last chunk by `prefill_logits`), from 0 and after a
/// prefix, give the bits of the prompt's per-token decode. Whether the
/// planner's cut differs from the old fixed prefill_batch cut depends on
/// the GPU's SMs at 2099/2100 and 1799/1800 tokens (`planner_cut_differs`:
/// a NOTE when it doesn't); at 1200/1201 (from 899) it differs on every
/// GPU, asserted: max + a last pass under `wave_floor` (288 at max 1152) is
/// no planner choice (`passes`: equal; `wave_passes`: last >= wave_floor;
/// `fa_passes`: `passes`, or from FA_LONG on equal passes too).
#[test]
fn gpu_prompt_chunks_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module_for(&g, &m).unwrap();
    let n = 2100;
    let ids = long_ids(&m, n);
    let net = CudaQwen2::new(&g, &km, &m, n + 1).unwrap();
    let max = net.prefill_batch();
    let mut d = net.new_cache(n + 1);
    for (p, &t) in ids[..n - 1].iter().enumerate() {
        net.feed(&mut d, t, p).unwrap();
    }
    let dec = net.forward(ids[n - 1], n - 1, &mut d);
    let sms = g.info().unwrap().sms;
    for from in [0, 300, n - 1 - 1200] {
        let mut c = net.new_cache(n + 1);
        if from > 0 {
            net.feed_batch(&ids[..from], 0, &mut c);
        }
        let rs = net.prompt_chunks(n - 1 - from, from, &c);
        // Multi-pass, and not the old fixed prefill_batch cut where no
        // planner choice can be (n - 1 - from == 1200; elsewhere a NOTE).
        assert!(rs.len() >= 2);
        let k = n - 1 - from;
        let old = crate::lm::chunk_ranges(k, max);
        let what = format!("{k} tokens from {from}");
        let differs = crate::testutil::planner_cut_differs(&rs, &old, sms, &what);
        assert!(
            differs || k - max >= max / 4,
            "{what}: the old prefill_batch cut {rs:?}"
        );
        assert_eq!((rs[0].start, rs.last().unwrap().end), (0, n - 1 - from));
        assert!(rs.windows(2).all(|w| w[0].end == w[1].start));
        assert!(rs.iter().all(|r| !r.is_empty() && r.len() <= max));
        for r in &rs {
            net.feed_batch(&ids[from + r.start..from + r.end], from + r.start, &mut c);
        }
        let l = net.forward(ids[n - 1], n - 1, &mut c);
        assert_eq!(bits(&l), bits(&dec), "serve's way from {from}: logits");
        assert!(
            kv_bytes(&net, &c, n) == kv_bytes(&net, &d, n),
            "serve's way from {from}: KV"
        );
        let mut c = net.new_cache(n + 1);
        if from > 0 {
            net.feed_batch(&ids[..from], 0, &mut c);
        }
        let rs = net.prompt_chunks(n - from, from, &c);
        let k = n - from;
        let what = format!("{k} tokens from {from}");
        let old = crate::lm::chunk_ranges(k, max);
        let differs = crate::testutil::planner_cut_differs(&rs, &old, sms, &what);
        assert!(
            differs || k - max >= max / 4,
            "{what}: the old prefill_batch cut {rs:?}"
        );
        let mut l = Vec::new();
        for (i, r) in rs.iter().enumerate() {
            let b = &ids[from + r.start..from + r.end];
            if i + 1 < rs.len() {
                net.feed_batch(b, from + r.start, &mut c);
            } else {
                l = net.prefill_logits(b, from + r.start, &mut c);
            }
        }
        assert_eq!(bits(&l), bits(&dec), "chat's way from {from}: logits");
        assert!(
            kv_bytes(&net, &c, n) == kv_bytes(&net, &d, n),
            "chat's way from {from}: KV"
        );
        eprintln!(
            "prompt_chunks from {from}: {:?} == per-token decode",
            rs.iter().map(|r| r.len()).collect::<Vec<_>>()
        );
    }
    // Exactly one pass (n == the batch max) and just under: one range,
    // and its KV rows are the decode's.
    for k in [max, max - 1] {
        let mut c = net.new_cache(n + 1);
        let rs = net.prompt_chunks(k, 0, &c);
        assert_eq!(rs, vec![0..k], "{k} tokens: one pass");
        net.feed_batch(&ids[..k], 0, &mut c);
        assert!(
            kv_bytes(&net, &c, k) == kv_bytes(&net, &d, k),
            "{k} tokens: KV"
        );
    }
}

/// Hunting aid: one child process of `gpu_concurrency_matrix_hunt`.
/// Decode-heavy victims (per-token decode-step graphs, 1200 tokens, file-
/// mapped models; each pass checked bit for bit against its own first)
/// next to one aggressor thread doing AXON_HUNT_AGG for AXON_STRESS_SECS.
#[test]
#[ignore = "hunting aid: run by gpu_concurrency_matrix_hunt"]
fn gpu_concurrency_hunt_child() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let agg = std::env::var("AXON_HUNT_AGG").unwrap_or_default();
    let secs: u64 = std::env::var("AXON_STRESS_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let victims = std::env::var("AXON_HUNT_VICTIMS").unwrap_or_else(|_| {
        format!(
            "models/qwen2.5-1.5b-instruct-q4_k_m.gguf,{},{}",
            crate::testutil::LLAMA_MODEL,
            crate::testutil::MODEL
        )
    });
    let end = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    let stop = AtomicBool::new(false);
    let live = || !stop.load(Ordering::Relaxed) && std::time::Instant::now() < end;
    std::thread::scope(|sc| {
        let mut hs = Vec::new();
        for (w, rel) in victims.split(',').enumerate() {
            let rel = rel.to_string();
            let live = &live;
            hs.push(sc.spawn(move || {
                let m = model_at(&rel).unwrap();
                let g = Gpu::open().unwrap();
                let km = Kernels::module(&g).unwrap();
                let ids = long_ids(&m, 1200);
                let net = CudaQwen2::new(&g, &km, &m, ids.len() + 1).unwrap();
                // AXON_HUNT_NOKV=1: no KV read-back (no pageable copy), the
                // logits only (through the pinned host_out).
                let nokv = std::env::var("AXON_HUNT_NOKV").is_ok_and(|v| v == "1");
                let pw = |sizes: &[usize]| {
                    if nokv {
                        (Vec::new(), prefill_with_nokv(&net, &ids, sizes))
                    } else {
                        prefill_with(&net, &ids, sizes)
                    }
                };
                let (kv0, l0) = pw(&[0]);
                let mut n = 0;
                while live() {
                    let sizes: &[usize] = if n % 2 == 0 {
                        &[0]
                    } else {
                        &[16, 0, 9, 0, 0, 200]
                    };
                    let (kv, l) = pw(sizes);
                    assert!(kv == kv0, "victim {w} {rel}: KV differs");
                    assert_eq!(bits(&l), bits(&l0), "victim {w} {rel}: logits");
                    n += 1;
                }
                eprintln!("victim {w} {rel}: {n} passes ok");
            }));
        }
        let live = &live;
        let agg2 = agg.clone();
        hs.push(sc.spawn(move || {
            let small = crate::testutil::model_path().unwrap();
            let small = small.to_str().unwrap().to_string();
            let mut n = 0u64;
            // Let the victims load and start decoding first.
            std::thread::sleep(std::time::Duration::from_secs(8));
            while live() {
                match agg2.as_str() {
                    "none" | "" => std::thread::sleep(std::time::Duration::from_millis(100)),
                    // What a test's start and end do: a whole model made and dropped.
                    "churn" => {
                        let m = crate::gguf::load_with(&small, Some(false)).unwrap();
                        let g = Gpu::open().unwrap();
                        let km = Kernels::module(&g).unwrap();
                        let net = CudaQwen2::new(&g, &km, &m, 64).unwrap();
                        let mut c = net.new_cache(64);
                        net.feed(&mut c, 1, 0).unwrap();
                        let _ = net.forward(2, 1, &mut c);
                    }
                    "pin" => {
                        let g = Gpu::open().unwrap();
                        let map = memmap2::MmapMut::map_anon(64 << 20).unwrap();
                        // SAFETY: map outlives the registration (dropped first).
                        drop(unsafe { g.register(map.as_ptr(), map.len(), false) }.unwrap());
                    }
                    "alloc" => {
                        let g = Gpu::open().unwrap();
                        for _ in 0..8 {
                            let b = g.alloc(256 << 20).unwrap();
                            b.zero().unwrap();
                        }
                    }
                    "module" => {
                        let g = Gpu::open().unwrap();
                        for _ in 0..4 {
                            drop(Kernels::module(&g).unwrap());
                        }
                    }
                    "ctx" => drop(Gpu::open().unwrap()),
                    // Another process on the GPU (a packed pod's co-tenant):
                    // short decode runs of the 0.5B, back to back.
                    "proc" => {
                        let out = std::process::Command::new(std::env::current_exe().unwrap())
                            .args(["--exact", "--ignored", "--nocapture"])
                            .arg("cuda::qwen2::tests::gpu_concurrency_hunt_child")
                            .env("AXON_HUNT_AGG", "none")
                            .env("AXON_HUNT_VICTIMS", crate::testutil::MODEL)
                            .env("AXON_STRESS_SECS", "12")
                            .output()
                            .unwrap();
                        assert!(out.status.success(), "co-tenant process failed");
                    }
                    // `churn` on a fresh thread that then exits, as libtest runs
                    // each test on its own thread.
                    "tchurn" => std::thread::scope(|t| {
                        t.spawn(|| {
                            let m = crate::gguf::load_with(&small, Some(false)).unwrap();
                            let g = Gpu::open().unwrap();
                            let km = Kernels::module(&g).unwrap();
                            let net = CudaQwen2::new(&g, &km, &m, 64).unwrap();
                            let mut c = net.new_cache(64);
                            net.feed(&mut c, 1, 0).unwrap();
                            let _ = net.forward(2, 1, &mut c);
                        });
                    }),
                    // A thread that touches the context and exits.
                    "texit" => std::thread::scope(|t| {
                        t.spawn(|| {
                            let g = Gpu::open().unwrap();
                            let b = g.alloc(1 << 20).unwrap();
                            b.zero().unwrap();
                        });
                    }),
                    // Processes that never touch CUDA (fork/exec alone).
                    "spawn" => {
                        let out = std::process::Command::new("sh")
                            .args(["-c", "sleep 0.2"])
                            .output()
                            .unwrap();
                        assert!(out.status.success());
                    }
                    // Another process that only opens the GPU and loads the
                    // kernels (a context made and torn down), no kernels run.
                    "proc-ctx" => {
                        let out = std::process::Command::new(std::env::current_exe().unwrap())
                            .args(["--exact", "--nocapture", "testutil::gpu_lock_is_reentrant"])
                            .output()
                            .unwrap();
                        assert!(out.status.success(), "co-tenant process failed");
                    }
                    // The real neighbour: a whole prefill_bits (0.5B) in this process.
                    "bits" => {
                        let m = crate::gguf::load_with(&small, Some(false)).unwrap();
                        prefill_bits(&m, "0.5B", CacheKv::F16);
                    }
                    "hostbuf" => {
                        let g = Gpu::open().unwrap();
                        let mut h = g.host_alloc(64 << 20).unwrap();
                        h.bytes_mut().unwrap()[0] = 1;
                    }
                    a => panic!("unknown AXON_HUNT_AGG {a}"),
                }
                n += 1;
            }
            eprintln!("aggressor {agg2}: {n} rounds");
        }));
        let mut panicked = None;
        for h in hs {
            if let Err(e) = h.join() {
                stop.store(true, Ordering::Relaxed);
                panicked.get_or_insert(e);
            }
        }
        if let Some(e) = panicked {
            std::panic::resume_unwind(e);
        }
    });
}

/// Hunting aid: `gpu_concurrency_hunt_child` in fresh processes, AXON_HUNT_ROUNDS
/// (default 2) per aggressor of AXON_HUNT_AGGS (default all of them, the
/// ones that faulted first), and a table of which ones poisoned the
/// context.
#[test]
#[ignore = "hunting aid: --ignored --nocapture, AXON_HUNT_AGGS / AXON_HUNT_ROUNDS / AXON_STRESS_SECS"]
fn gpu_concurrency_matrix_hunt() {
    let aggs = std::env::var("AXON_HUNT_AGGS")
        // spawn / proc / proc-ctx first: the only ones that ever faulted
        // (docs/log/2026-10-18-cuda-concurrency.md); the rest are the
        // in-process controls, all clean there.
        .unwrap_or_else(|_| "spawn,proc,proc-ctx,none,churn,pin,alloc,module,ctx,hostbuf".into());
    let rounds: usize = std::env::var("AXON_HUNT_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let exe = std::env::current_exe().unwrap();
    let mut table = Vec::new();
    for r in 0..rounds {
        for agg in aggs.split(',') {
            let t = std::time::Instant::now();
            let out = std::process::Command::new(&exe)
                .args(["--exact", "--ignored", "--nocapture", "--test-threads", "1"])
                .arg("cuda::qwen2::tests::gpu_concurrency_hunt_child")
                .env("AXON_HUNT_AGG", agg)
                .output()
                .unwrap();
            let s = String::from_utf8_lossy(&out.stdout).to_string()
                + &String::from_utf8_lossy(&out.stderr);
            let illegal = s.contains("ILLEGAL_ADDRESS");
            let first = s
                .lines()
                .find(|l| l.contains("panicked at"))
                .unwrap_or("")
                .to_string();
            let line = format!(
                "round {r} agg {agg}: ok={} illegal={illegal} {:.0}s {first}",
                out.status.success(),
                t.elapsed().as_secs_f64()
            );
            eprintln!("HUNT {line}");
            for l in s
                .lines()
                .filter(|l| l.starts_with("victim") || l.starts_with("aggressor"))
            {
                eprintln!("    {l}");
            }
            if !out.status.success() {
                for l in s
                    .lines()
                    .filter(|l| l.contains("panicked") || l.contains("CUDA"))
                    .take(6)
                {
                    eprintln!("    | {l}");
                }
            }
            table.push(line);
        }
    }
    eprintln!("HUNT TABLE\n{}", table.join("\n"));
}

// ---------------------------------------------------------------------------
// WP-A1 (2026-10-29): the H100 probe of two `--numerics fast` GEMM candidates
// on the T3 skeleton (probe_fast.cu; the plan's WP-A1, pre-registered rules in
// the GitHub issue "A1: fast GEMM probe, pre-registered rules"). TEST-ONLY:
// probe_fast.cu is never part of KERNELS; it is built here from hopper.cu's
// helper prefix + probe_fast.cu as a module of its own (its own cache key), so
// no product path can load it.
// - (a) probe_fp8_128x64 / _128x128 (CTA tile rows x tokens): e4m3 x e4m3 SS
//   wgmma.m64n64k32, f32 over the whole K, one epilogue (per-row weight scale
//   x per-token activation scale);
// - (b) probe_w4a16_128x64 / _256x64: Q4_K nibbles -> fp16 in registers, RS
//   wgmma.m64n64k16 f16;
// - (c) each with split-K (ksplit S) or stream-K (ksplit 0), the partials
//   added with atomics after a zeroing launch (inside the timed region).
// Two tiles per candidate, the same instructions: T3's 128 x 64, and one that
// cuts the L2 -> SM bytes per MAC (FP8 128 x 64 would need 8.2 TB/s at 350
// TMAC/s; 128 x 128 5.5. W4A16 128 x 64 5.8 TB/s at 230; 256 x 64 4.0).
// Fast mode: the results are checked against a CPU f64 reference within a
// tolerance (A1_TOL_*), not bit for bit.

/// probe_fast.cu's shared-memory carve-up (A1* macros), byte offsets from the
/// 1024-aligned base: the stages, then the full / empty mbarriers.
/// FP8, T tokens per tile: activations 2 x {128 B, T tokens} SW128 (token
/// sub-tile h at + h 8 KiB), then weights 2 x {128 B, 128 rows} SW128
/// (consumer warpgroup cw's rows at + cw 8 KiB); 4 stages at T 64, 3 at 128.
const fn a1f_st(t: usize) -> usize {
    if t == 64 { 4 } else { 3 }
}
const fn a1f_x(t: usize) -> usize {
    t * 256
}
const A1F_W: usize = 128 * 256;
const fn a1f_stage(t: usize) -> usize {
    a1f_x(t) + A1F_W
}
const fn a1f_smem(t: usize) -> usize {
    a1f_st(t) * a1f_stage(t) + 2 * a1f_st(t) * 8 + 1024
}
/// W4A16, R rows per tile: activations 4 x {128 B = 64 halves, 64 tokens}
/// SW128, the tile's repacked nibbles (R lines of 128 B, no swizzle), its
/// scales (R rows x 8 half2 (s, -z), no swizzle); 4 stages at R 128, 3 at 256.
const fn a1h_st(r: usize) -> usize {
    if r == 128 { 4 } else { 3 }
}
const A1H_X: usize = 64 * 512;
const fn a1h_n(r: usize) -> usize {
    r * 128
}
const fn a1h_s(r: usize) -> usize {
    r * 32
}
const fn a1h_stage(r: usize) -> usize {
    A1H_X + a1h_n(r) + a1h_s(r)
}
const fn a1h_smem(r: usize) -> usize {
    a1h_st(r) * a1h_stage(r) + 2 * a1h_st(r) * 8 + 1024
}

/// A candidate kernel, probe_<name>: its family, CTA tile (rows x tokens) and
/// the N of its wgmma instruction.
#[derive(Clone, Copy, PartialEq, Debug)]
struct A1Var {
    name: &'static str,
    fp8: bool,
    rows: usize,
    toks: usize,
    ni: usize,
}

/// The five candidate kernels: FP8 on m64n64k32 at two tiles and on
/// m64n128k32 at the bigger one (a rule-1 KILL at n64 alone could measure
/// the instruction, not FP8), W4A16 on m64n64k16 at two tiles.
const A1_VARS: [A1Var; 5] = [
    A1Var {
        name: "fp8_128x64",
        fp8: true,
        rows: 128,
        toks: 64,
        ni: 64,
    },
    A1Var {
        name: "fp8_128x128",
        fp8: true,
        rows: 128,
        toks: 128,
        ni: 64,
    },
    A1Var {
        name: "fp8n128_128x128",
        fp8: true,
        rows: 128,
        toks: 128,
        ni: 128,
    },
    A1Var {
        name: "w4a16_128x64",
        fp8: false,
        rows: 128,
        toks: 64,
        ni: 64,
    },
    A1Var {
        name: "w4a16_256x64",
        fp8: false,
        rows: 256,
        toks: 64,
        ni: 64,
    },
];

impl A1Var {
    fn kernel(&self) -> String {
        format!("probe_{}", self.name)
    }
    fn smem(&self) -> usize {
        if self.fp8 {
            a1f_smem(self.toks)
        } else {
            a1h_smem(self.rows)
        }
    }
    fn tiles(&self, rows: usize, nb: usize) -> usize {
        rows.div_ceil(self.rows) * nb.div_ceil(self.toks)
    }
}

/// The probe module's kernels.
const A1_KERNELS: [&str; 9] = [
    "probe_fp8_128x64",
    "probe_fp8_128x128",
    "probe_fp8n128_128x128",
    "probe_w4a16_128x64",
    "probe_w4a16_256x64",
    "a1_zero_f32",
    "a1_peak_fp8_ss_n64",
    "a1_peak_fp8_ss_n128",
    "a1_peak_f16_rs_n64",
];

/// (rel-rms, rel-max) bounds on |GPU - CPU f64 reference|, relative to the
/// rms of the reference over the checked outputs. Structural, not a quality
/// bar: a wrong descriptor, K order, tile or scale gives errors of order 1.
/// - FP8: the products are exact (e4m3 x e4m3), but Hopper's FP8 tensor-core
///   accumulation keeps only ~14 bits (DeepSeek-V3 report, section 3.3.2), so
///   over K 18944 (592 k32 steps) the error can reach a few percent: loose.
/// - W4A16: s = d sc and z = dmin m rounded to fp16, one rounding for q s - z
///   (2^-11 relative per weight), f32 accumulation: tight.
const A1_TOL_FP8: (f64, f64) = (0.1, 0.5);
const A1_TOL_W4A16: (f64, f64) = (0.005, 0.03);

/// Every alignment and size the probe kernels rely on, checked on the host
/// before each plan is built (the T3 lesson: its first H100 launch died on a
/// misaligned load), so a bad configuration fails here with a message:
/// - global (cuTensorMapEncodeTiled): base addresses and row strides
///   multiples of 16 B; K a multiple of 256 (whole stages);
/// - smem: SW128 TMA destinations and wgmma descriptor bases 1024-aligned
///   (each stage, the activation boxes, the FP8 weight boxes and their
///   64-row warpgroup halves); the no-swizzle TMA destinations (nibbles,
///   scales) 128-aligned; mbarriers 8-aligned; the totals (+ 1024 B of base
///   alignment slack) within the 227 KiB opt-in limit;
/// - expect-tx: each stage's size is exactly the bytes its TMA boxes bring
///   (FP8 2 x T x 128 B + 2 x 16 KiB; W4A16 4 x 8 KiB + R x 128 B + R x 32 B).
fn a1_layout(bases: &[(&str, u64)], strides: &[(&str, usize)], cols: usize) -> Result<(), String> {
    let mut checks: Vec<(bool, String)> = vec![(
        cols.is_multiple_of(256) && cols > 0,
        format!("K {cols} not a multiple of 256"),
    )];
    for &(n, b) in bases {
        checks.push((
            b.is_multiple_of(16),
            format!("{n} at {b:#x}: TMA needs 16 B"),
        ));
    }
    for &(n, s) in strides {
        checks.push((
            s.is_multiple_of(16) && s > 0,
            format!("{n} stride {s}: TMA needs 16 B"),
        ));
    }
    for t in [64, 128] {
        checks.extend([
            (
                a1f_stage(t).is_multiple_of(1024)
                    && (a1f_x(t) / 2).is_multiple_of(1024)
                    && (A1F_W / 2).is_multiple_of(1024)
                    && A1F_W / 2 == 2 * 64 * 128,
                format!("fp8 T {t}: SW128 boxes or warpgroup halves % 1024"),
            ),
            (
                a1f_stage(t) == 2 * t * 128 + 2 * 128 * 128,
                format!("fp8 T {t}: expect-tx != the stage's TMA boxes"),
            ),
            (
                (a1f_st(t) * a1f_stage(t)).is_multiple_of(8) && a1f_smem(t) <= 232448,
                format!("fp8 T {t}: smem {} > 227 KiB or barriers % 8", a1f_smem(t)),
            ),
        ]);
    }
    for r in [128, 256] {
        checks.extend([
            (
                a1h_stage(r).is_multiple_of(1024) && (A1H_X / 4).is_multiple_of(1024),
                format!("w4a16 R {r}: SW128 boxes % 1024"),
            ),
            (
                A1H_X.is_multiple_of(128) && (A1H_X + a1h_n(r)).is_multiple_of(128) && r <= 256,
                format!("w4a16 R {r}: nibble / scale destinations % 128 or box rows > 256"),
            ),
            (
                a1h_stage(r) == 4 * 64 * 128 + r * 128 + r * 32,
                format!("w4a16 R {r}: expect-tx != the stage's TMA boxes"),
            ),
            (
                (a1h_st(r) * a1h_stage(r)).is_multiple_of(8) && a1h_smem(r) <= 232448,
                format!(
                    "w4a16 R {r}: smem {} > 227 KiB or barriers % 8",
                    a1h_smem(r)
                ),
            ),
        ]);
    }
    match checks.iter().find(|c| !c.0) {
        Some((_, m)) => Err(format!("probe_fast layout: {m}")),
        None => Ok(()),
    }
}

/// The layout invariants hold, catch what they claim to, and probe_fast.cu
/// agrees with the constants above (no GPU).
#[test]
fn a1_layout_pinned() {
    for cols in [256usize, 768, 1280, 3584, 4352, 18944] {
        a1_layout(
            &[("w", 0x7f00_0000_0000), ("x", 0x7f10_0000_0100)],
            &[("w", cols), ("x", 2 * cols), ("nib", 128), ("scl", 32)],
            cols,
        )
        .unwrap();
    }
    assert!(a1_layout(&[("w", 8)], &[], 256).is_err());
    assert!(a1_layout(&[], &[("x", 264)], 256).is_err());
    assert!(a1_layout(&[], &[], 300).is_err());
    let src = include_str!("../probe_fast.cu");
    for (k, v) in [
        (
            "#define A1F_ST(T) ((T) == 64 ? 4 : 3)",
            a1f_st(64) == 4 && a1f_st(128) == 3,
        ),
        ("#define A1F_X(T) ((T) * 256)", a1f_x(64) == 64 * 256),
        ("#define A1F_W (128 * 256)", A1F_W == 128 * 256),
        (
            "#define A1F_STAGE(T) (A1F_X(T) + A1F_W)",
            a1f_stage(128) == a1f_x(128) + A1F_W,
        ),
        (
            "#define A1F_SMEM(T) (A1F_ST(T) * A1F_STAGE(T) + 2 * A1F_ST(T) * 8 + 1024)",
            a1f_smem(64) == 4 * a1f_stage(64) + 64 + 1024,
        ),
        (
            "#define A1H_ST(R) ((R) == 128 ? 4 : 3)",
            a1h_st(128) == 4 && a1h_st(256) == 3,
        ),
        ("#define A1H_X (64 * 512)", A1H_X == 64 * 512),
        ("#define A1H_N(R) ((R) * 128)", a1h_n(256) == 256 * 128),
        ("#define A1H_S(R) ((R) * 32)", a1h_s(256) == 256 * 32),
        (
            "#define A1H_STAGE(R) (A1H_X + A1H_N(R) + A1H_S(R))",
            a1h_stage(128) == A1H_X + a1h_n(128) + a1h_s(128),
        ),
        (
            "#define A1H_SMEM(R) (A1H_ST(R) * A1H_STAGE(R) + 2 * A1H_ST(R) * 8 + 1024)",
            a1h_smem(256) == 3 * a1h_stage(256) + 48 + 1024,
        ),
        ("mbar_expect(&full[s], A1F_STAGE(T));", true),
        ("mbar_expect(&full[s], A1H_STAGE(R));", true),
        ("probe_fp8_t<64, 64>(", true),
        ("probe_fp8_t<128, 64>(", true),
        ("probe_fp8_t<128, 128>(", true),
        ("probe_w4a16_t<128>(", true),
        ("probe_w4a16_t<256>(", true),
    ] {
        assert!(
            src.contains(k) && v,
            "probe_fast.cu and tests.rs disagree: {k}"
        );
    }
    for k in A1_KERNELS {
        assert!(src.contains(&format!("{k}(")), "probe_fast.cu has no {k}");
    }
    for v in A1_VARS {
        let body = &src[src.find(&format!("{}(", v.kernel())).unwrap()..];
        let call = if v.fp8 {
            format!("probe_fp8_t<{}, {}>(", v.toks, v.ni)
        } else {
            format!("probe_w4a16_t<{}>(", v.rows)
        };
        assert!(
            body[..body.find('}').unwrap()].contains(&call) && (v.fp8 || v.ni == 64),
            "{}: A1_VARS and probe_fast.cu disagree ({call})",
            v.name
        );
    }
    // No printf call (a call inside the wgmma pipeline serializes every wgmma).
    assert!(!src.contains("printf("));
}

/// The probe module's source: kernels.cu's typedefs, hopper.cu's helpers
/// (TMap .. the RS wgmma wrappers, up to HOPPER_CHECK_SMEM), probe_fast.cu.
/// Panics if a marker moved.
fn probe_fast_src() -> String {
    let k = crate::cuda::KERNELS;
    let (from, to) = (
        "// Hopper (sm_90a: H100/H200/H20/GH200) only",
        "// Test (gpu_hopper_tma_wgmma)",
    );
    assert_eq!(k.matches(from).count(), 1, "KERNELS: {from}");
    let a = k.find(from).unwrap();
    let b = a + k[a..].find(to).unwrap_or_else(|| panic!("KERNELS: {to}"));
    format!(
        "// AXON TEST-ONLY: the WP-A1 fast-GEMM probe module (tests.rs probe_fast_src)\n\
         typedef unsigned char u8;\ntypedef unsigned int u32;\ntypedef unsigned long long u64;\n\
         {}\n#endif // the helpers' __CUDA_ARCH_FEAT_SM90_ALL\n{}",
        &k[a..b],
        include_str!("../probe_fast.cu")
    )
}

/// The probe source compiles for sm_90a through NVRTC (no GPU needed): every
/// kernel is in the cubin, the two GEMMs with no spills and no stack frame,
/// the log has no "Potential Performance Loss" note (C7510: wgmma
/// serialized, C7507: setmaxnreg ignored); for plain sm_90 it compiles to
/// nothing. Prints `A1 COMPILE` lines (ptxas -v). AXON_A1_SRC_DIR=DIR writes
/// the source there as probe-fast.cu (the kit's kernel cache is prewarmed
/// from it).
#[test]
fn nvrtc_probe_fast_compiles_sm90a() {
    let src = probe_fast_src();
    if let Some(d) = std::env::var_os("AXON_A1_SRC_DIR") {
        std::fs::write(std::path::Path::new(&d).join("probe-fast.cu"), &src).unwrap();
    }
    let n = match crate::cuda::ffi::nvrtc() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let ver = crate::cuda::ffi::nvrtc_version(n).unwrap();
    if !crate::cuda::ffi::nvrtc_archs(n).unwrap().contains(&90) || ver.0 < 12 {
        eprintln!("skipping: NVRTC has no sm_90a");
        return;
    }
    for t in [
        crate::cuda::Target::Cubin90a,
        crate::cuda::Target::Cubin(90),
    ] {
        let mut opts = vec![format!("--gpu-architecture={t}")];
        opts.extend(crate::cuda::KERNEL_OPTS_EXT.iter().map(|s| s.to_string()));
        opts.push("--ptxas-options=-v".into());
        // NVRTC 13.x has its own compile cache, keyed on the module's PTX: a hit
        // returns the cubin with an EMPTY log, which would pass the C75xx check
        // vacuously (a -D the source never reads doesn't change the PTX). A
        // uniquely named empty kernel makes every check compile a miss.
        let t_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let src_check = format!(
            "{src}\nextern \"C\" __global__ void a1_nocache_{t_ns}(int* p) {{ if (p) p[0] = {}; }}\n",
            t_ns % 1000
        );
        let (image, log) = crate::cuda::compile_logged(n, &src_check, &opts, t)
            .unwrap_or_else(|e| panic!("probe_fast {t}: {e}"));
        eprintln!(
            "A1 COMPILE {t}: {} bytes of cubin (NVRTC {}.{})",
            image.len(),
            ver.0,
            ver.1
        );
        assert!(
            !log.contains("Performance Loss")
                && !log.contains("serialized")
                && !log.contains("C75"),
            "probe_fast {t}: performance notes in the NVRTC log:\n{log}"
        );
        for k in A1_KERNELS {
            let has = image.windows(k.len()).any(|w| w == k.as_bytes());
            assert_eq!(
                has,
                t == crate::cuda::Target::Cubin90a,
                "probe_fast {t}: {k}"
            );
        }
        if t != crate::cuda::Target::Cubin90a {
            continue;
        }
        assert!(
            log.contains("Compiling entry function"),
            "probe_fast {t}: no ptxas -v output (a cache hit?):\n{log}"
        );
        // ptxas -v: "Function properties for K" / "N bytes stack frame, S bytes
        // spill stores, ..." / "Used R registers, ...".
        let lines: Vec<&str> = log.lines().collect();
        for v in A1_VARS {
            let k = format!("probe_{}", v.name);
            let i = lines
                .iter()
                .position(|l| l.ends_with(&format!("Function properties for {k}")))
                .unwrap_or_else(|| panic!("no ptxas -v block for {k}:\n{log}"));
            let num = |l: &str, unit: &str| -> i64 {
                let p = l
                    .find(unit)
                    .unwrap_or_else(|| panic!("{k}: no '{unit}' in {l}"));
                l[..p].split_whitespace().last().unwrap().parse().unwrap()
            };
            let (stack, spill) = (
                num(lines[i + 1], " bytes stack frame"),
                num(lines[i + 1], " bytes spill stores"),
            );
            let regs = num(lines[i + 2], " registers");
            eprintln!(
                "A1 COMPILE {k}: {regs} registers, {spill} bytes spill stores, {stack} bytes stack frame"
            );
            assert!(spill == 0 && stack == 0, "{k}: spills / stack");
        }
    }
}

/// The probe's module (sm_90a devices only: None elsewhere).
fn probe_fast_module(g: &Gpu) -> Option<Module<'_>> {
    let m = Module::load(g, &probe_fast_src(), &crate::cuda::KERNEL_OPTS_EXT)
        .unwrap()
        .0;
    m.hopper().then_some(m)
}

/// OCP FP8 E4M3 (e4m3fn) -> f32: bias 7, 3 mantissa bits, subnormals
/// m 2^-9, no infinities, S.1111.111 NaN.
fn e4m3_f32(c: u8) -> f32 {
    let s = if c & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (((c >> 3) & 15) as i32, (c & 7) as f32);
    if e == 15 && m == 7.0 {
        f32::NAN
    } else if e == 0 {
        s * m * (2f32).powi(-9)
    } else {
        s * (1.0 + m / 8.0) * (2f32).powi(e - 7)
    }
}

/// n random e4m3 codes, never NaN: any sign and mantissa, exponent field
/// 3..=12 (|v| from 2^-4 to 1.875 x 2^5).
fn a1_e4m3_codes(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let r = rng.next();
        for i in 0..4 {
            if v.len() < n {
                let x = (r >> (16 * i)) as u16;
                v.push((x & 0x80) as u8 | ((3 + (x >> 8) % 10) as u8) << 3 | (x & 7) as u8);
            }
        }
    }
    v
}

/// Known e4m3 values; the generator never makes a NaN (no GPU).
#[test]
fn a1_e4m3_decode() {
    for (c, v) in [
        (0x7eu8, 448.0f32),
        (0x38, 1.0),
        (0xb8, -1.0),
        (0x08, 1.0 / 64.0),
        (0x01, 1.0 / 512.0),
        (0x00, 0.0),
        (0x3c, 1.5),
        (0x62, 40.0),
    ] {
        assert_eq!(e4m3_f32(c), v, "{c:#x}");
    }
    assert!(e4m3_f32(0x7f).is_nan() && e4m3_f32(0xff).is_nan());
    let mut rng = Rng(3);
    let v = a1_e4m3_codes(&mut rng, 100_003);
    assert_eq!(v.len(), 100_003);
    assert!(v.iter().all(|&c| {
        let x = e4m3_f32(c).abs();
        (1.0 / 16.0..=60.0).contains(&x)
    }));
}

/// Per-row (or per-token) scales in [0.5, 1.5) x 2^-6.
fn a1_scales(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| (1.0 + 0.5 * rng.f()) / 64.0).collect()
}

/// Q4_K sub-block j's 6-bit (scale, min) (llama.cpp get_scale_min_k4).
fn q4k_scale_min(j: usize, sc: &[u8]) -> (u8, u8) {
    if j < 4 {
        (sc[j] & 63, sc[j + 4] & 63)
    } else {
        (
            (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4),
            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
        )
    }
}

/// K offsets (from 32 j + 2 q) of the 8 nibbles of a W4A16 u32 (n0 = bits
/// 0..3 .. n7 = bits 28..31): a1_dq4 makes half2 (n0, n4), (n1, n5), (n2, n6),
/// (n3, n7) = the A fragment's (K 2q, 2q+1), (2q+8, 2q+9) of k16 step 2 j,
/// then the same of step 2 j + 1 (+16).
const A1_NIB_K: [usize; 8] = [0, 8, 16, 24, 1, 9, 17, 25];

/// Q4_K weights repacked for probe_w4a16_* (rows padded to rows_p, a multiple
/// of 256 (both tile heights), with zeros):
/// - nib: per K-block kb and row r, 128-byte lines at kb rows_p + r; the
///   16-row group of r (2 KiB) holds, for lane t = 4 g + q (g = r % 8) and
///   h = 2 rr + u (rr = (r % 16) / 8, u = sub-blocks 4 u ..), the uint4 at
///   (32 h + t) 16 whose u32 jj is sub-block j = 4 u + jj's nibbles at the
///   K offsets A1_NIB_K from 32 j + 2 q;
/// - scl: per (kb, r), 8 u32 = half2 (s, -z) per sub-block, s = fp16(d sc),
///   z = fp16(dmin m) (each rounded once from the exact f32 product).
struct A1Q4f {
    nib: Vec<u8>,
    scl: Vec<u32>,
    rows_p: usize,
}

fn a1_q4f_pack(w: &[u8], rows: usize, cols: usize) -> A1Q4f {
    let (nblk, rb) = (cols / 256, cols / 256 * 144);
    let rows_p = rows.next_multiple_of(256);
    let mut nib = vec![0u8; nblk * rows_p * 128];
    let mut scl = vec![0u32; nblk * rows_p * 8];
    for r in 0..rows {
        let (g, rr) = (r % 8, (r % 16) / 8);
        for b in 0..nblk {
            let blk = &w[r * rb + b * 144..][..144];
            let d = quant::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let dmin = quant::f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
            let (sc, qs) = (&blk[4..16], &blk[16..]);
            let q = |k: usize| -> u32 {
                let j = k / 32;
                ((qs[32 * (j / 2) + k % 32] >> (4 * (j % 2))) & 15) as u32
            };
            for j in 0..8 {
                let (s, m) = q4k_scale_min(j, sc);
                let hs = quant::f32_to_f16(d * s as f32) as u32;
                let hz = quant::f32_to_f16(-(dmin * m as f32)) as u32;
                scl[(b * rows_p + r) * 8 + j] = hs | hz << 16;
                for qq in 0..4 {
                    let mut v = 0u32;
                    for (i, &o) in A1_NIB_K.iter().enumerate() {
                        v |= q(32 * j + 2 * qq + o) << (4 * i);
                    }
                    let (u, jj, t) = (j / 4, j % 4, 4 * g + qq);
                    let off =
                        (b * rows_p + 16 * (r / 16)) * 128 + ((2 * rr + u) * 32 + t) * 16 + 4 * jj;
                    nib[off..off + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    A1Q4f { nib, scl, rows_p }
}

/// half -> f32 -> the nearest half (one rounding: every operation emulated
/// here is exact in f64 before it).
fn a1_h(x: f64) -> u16 {
    quant::f32_to_f16(x as f32)
}

/// probe_w4a16_*'s register path on the CPU (no GPU), both tile heights: for
/// every consumer thread of every tile and K-block, read its uint4s and scales
/// where the kernel reads them (A1H_* offsets, nib[32 h + lane]), run a1_dq4's ops
/// (LOP3 magic, HSUB2 / HFMA2, the final HFMA2) in emulated fp16, and place
/// each half where the wgmma m64nNk16 A fragment says it goes (rows g, g + 8
/// of the warp's 16; K 2q, 2q+1, 2q+8, 2q+9 of each k16 step). Every weight
/// must be produced exactly once, within fp16 rounding of quant::dequant.
#[test]
fn a1_w4a16_fragments_match_dequant() {
    let mut rng = Rng(11);
    let (rows, cols) = (200usize, 768usize);
    let w = synth_matrix(&mut rng, Q4_K, rows, cols);
    let p = a1_q4f_pack(&w, rows, cols);
    let rb = cols / 256 * 144;
    let mut want = vec![0f32; rows * cols];
    for r in 0..rows {
        quant::dequant(Q4_K, &w[r * rb..][..rb], &mut want[r * cols..][..cols]);
    }
    let mut seen = vec![0u8; rows * cols];
    let hf = |h: u16| quant::f16_to_f32(h) as f64;
    let dq = |v: u32, sz: u32| -> [u32; 4] {
        // a1_dq4: per half lane (lo, hi).
        let (s, nz) = (hf(sz as u16), hf((sz >> 16) as u16));
        let lop = |a: u32, mask: u32| (a & mask) | 0x6400_6400;
        let lanes = |x: u32| [x as u16, (x >> 16) as u16];
        let mut out = [0u32; 4];
        for (i, (x, hi)) in [
            (lop(v, 0x000f_000f), false),
            (lop(v, 0x00f0_00f0), true),
            (lop(v >> 8, 0x000f_000f), false),
            (lop(v >> 8, 0x00f0_00f0), true),
        ]
        .into_iter()
        .enumerate()
        {
            let mut r = 0u32;
            for (k, &h) in lanes(x).iter().enumerate() {
                // HSUB2(x, 1024) or HFMA2(x, 1/16, -64): the exact nibble.
                let q = if hi {
                    hf(a1_h(hf(h) * hf(0x2c00) + hf(0xd400)))
                } else {
                    hf(a1_h(hf(h) - 1024.0))
                };
                assert!((0.0..16.0).contains(&q) && q.fract() == 0.0, "magic");
                r |= (a1_h(q * s + nz) as u32) << (16 * k);
            }
            out[i] = r;
        }
        out
    };
    for rt in [128usize, 256] {
        seen.iter_mut().for_each(|c| *c = 0);
        for kb in 0..cols / 256 {
            for r0 in (0..p.rows_p).step_by(rt) {
                for cw in 0..2 {
                    for (m, wp) in (0..rt / 128).flat_map(|m| (0..4).map(move |w| (m, w))) {
                        // probe_w4a16_t: grp = (R / 32) cw + 4 m + w.
                        let grp = rt / 32 * cw + 4 * m + wp;
                        for l in 0..32 {
                            let (g, q) = (l / 4, l % 4);
                            for j in 0..8 {
                                for rr in 0..2 {
                                    let h = 2 * rr + j / 4;
                                    let off = (kb * p.rows_p + r0) * 128
                                        + grp * 2048
                                        + (32 * h + l) * 16
                                        + 4 * (j % 4);
                                    let v =
                                        u32::from_le_bytes(p.nib[off..off + 4].try_into().unwrap());
                                    // The scales at (16 grp + g) * 8 + 64 rr + j; the
                                    // epilogue's row r0 + R / 2 cw + 64 m + 16 w + g + 8 rr.
                                    let row_t = 16 * grp + g + 8 * rr;
                                    assert_eq!(row_t, rt / 2 * cw + 64 * m + 16 * wp + g + 8 * rr);
                                    let sz = p.scl[(kb * p.rows_p + r0 + row_t) * 8 + j];
                                    let hv = dq(v, sz);
                                    let r = r0 + row_t;
                                    // hv[0]: step 2j K 2q, 2q+1; hv[1]: 2q+8, +9;
                                    // hv[2], hv[3]: the same of step 2j+1 (+16).
                                    for (i, x) in hv.iter().enumerate() {
                                        for lane in 0..2 {
                                            let k = 256 * kb
                                                + 32 * j
                                                + 16 * (i / 2)
                                                + 2 * q
                                                + 8 * (i % 2)
                                                + lane;
                                            let got = hf((x >> (16 * lane)) as u16);
                                            if r >= rows {
                                                assert_eq!(got, 0.0, "pad row {r}");
                                                continue;
                                            }
                                            let e = want[r * cols + k] as f64;
                                            seen[r * cols + k] += 1;
                                            let tol = 3.0 * (e.abs() + 0.5) / 2048.0 + 1e-6;
                                            assert!(
                                                (got - e).abs() <= tol,
                                                "r {r} k {k}: got {got} want {e}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "R {rt}: a weight not produced exactly once"
        );
    }
}

/// a1_next on the CPU: for tiles (1), split-K and stream-K (0), every
/// (tile, K-block) is covered exactly once over the grid, never by an empty
/// range (no GPU).
#[test]
fn a1_sched_covers() {
    // a1_sched + a1_next of probe_fast.cu, for CTA c of grid G.
    fn ranges(
        ksplit: usize,
        nblk: usize,
        ntiles: usize,
        c: usize,
        gn: usize,
    ) -> Vec<(usize, usize, usize)> {
        let mut v = Vec::new();
        if ksplit == 0 {
            let tot = ntiles * nblk;
            let (mut pos, end) = (tot * c / gn, tot * (c + 1) / gn);
            while pos < end {
                let (t, kb0) = (pos / nblk, pos % nblk);
                let kb1 = if end - pos < nblk - kb0 {
                    kb0 + (end - pos)
                } else {
                    nblk
                };
                v.push((t, kb0, kb1));
                pos += kb1 - kb0;
            }
        } else {
            let mut u = c;
            while u < ntiles * ksplit {
                let (t, p) = (u / ksplit, u % ksplit);
                let (kb0, kb1) = (p * nblk / ksplit, (p + 1) * nblk / ksplit);
                if kb1 > kb0 {
                    v.push((t, kb0, kb1));
                }
                u += gn;
            }
        }
        v
    }
    for (nblk, ntiles) in [
        (1usize, 1usize),
        (14, 8),
        (74, 56),
        (14, 296),
        (3, 5),
        (17, 2),
    ] {
        for ksplit in [1usize, 2, 3, 7, 14, 0] {
            for gn in [1usize, 3, 7, 132] {
                let mut seen = vec![0u32; ntiles * nblk];
                for c in 0..gn {
                    for (t, kb0, kb1) in ranges(ksplit, nblk, ntiles, c, gn) {
                        assert!(kb1 > kb0 && kb1 <= nblk && t < ntiles);
                        for kb in kb0..kb1 {
                            seen[t * nblk + kb] += 1;
                        }
                    }
                }
                assert!(
                    seen.iter().all(|&s| s == 1),
                    "nblk {nblk} ntiles {ntiles} ksplit {ksplit} grid {gn}"
                );
            }
        }
    }
    let src = include_str!("../probe_fast.cu");
    for k in [
        "s.pos = tot * blockIdx.x / gridDim.x;",
        "s.end = tot * (blockIdx.x + 1) / gridDim.x;",
        "kb1 = left < (long long)(s.nblk - kb0) ? kb0 + (int)left : s.nblk;",
        "kb0 = p * s.nblk / s.ksplit;",
        "kb1 = (p + 1) * s.nblk / s.ksplit;",
        "if (kb1 > kb0) return true;",
    ] {
        assert!(src.contains(k), "probe_fast.cu a1_next changed: {k}");
    }
}

/// One prepared kernel launch: function, geometry and arguments, all built
/// before any timing.
struct A1Launch<'m> {
    f: Func<'m>,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    args: Vec<Arg>,
}

/// A timed unit: its launches in order (the zeroing launch, then the GEMM).
struct A1Plan<'m> {
    name: String,
    ls: Vec<A1Launch<'m>>,
}

impl A1Plan<'_> {
    /// Per launch, `Gpu::launch` and nothing else: cuCtxSetCurrent (a
    /// thread-local context swap, no sync), the kernel-parameter pointer Vec
    /// and cuLaunchKernelEx, the path the baselines and production take.
    /// Every tensor map, function lookup and smem attribute was done when the
    /// plan was built (P1 was voided by a driver query, `g.info()`, inside
    /// its timed launcher). No sync, no allocation on the device, no query.
    ///
    /// # Safety
    /// The plan's arguments match its kernels and its buffers are live.
    unsafe fn fire(&self, g: &Gpu) {
        for l in &self.ls {
            // SAFETY: the caller's.
            unsafe { g.launch(&l.f, l.grid, l.block, l.smem, &l.args).unwrap() };
        }
    }
}

/// The plan name of a variant's schedule: "fp8_128x64" (tiles),
/// "fp8_128x64_splitk2", "fp8_128x64_streamk".
fn a1_name(cand: &str, ksplit: usize) -> String {
    match ksplit {
        1 => cand.to_string(),
        0 => format!("{cand}_streamk"),
        s => format!("{cand}_splitk{s}"),
    }
}

/// The zeroing launch before a split-K / stream-K GEMM (none for tiles).
fn a1_zero<'m>(pm: &'m Module, out: &DevBuf, n: usize, ksplit: usize) -> Vec<A1Launch<'m>> {
    if ksplit == 1 {
        return Vec::new();
    }
    vec![A1Launch {
        f: pm.function("a1_zero_f32").unwrap(),
        grid: ((n / 4).div_ceil(256).clamp(1, 1024) as u32, 1, 1),
        block: (256, 1, 1),
        smem: 0,
        args: vec![out.arg(), Arg::U64(n as u64)],
    }]
}

/// probe_fp8_128x<T> (v) on e4m3 weights `w` (rows x cols, scales ws) and nb
/// tokens of e4m3 activations `x` (stride cols, scales xs), ctas persistent
/// CTAs, into out (rows * nb f32, stride rows).
#[allow(clippy::too_many_arguments)]
fn a1_plan_fp8<'m>(
    g: &Gpu,
    pm: &'m Module,
    v: A1Var,
    w: &DevBuf,
    ws: &DevBuf,
    x: &DevBuf,
    xs: &DevBuf,
    (rows, cols, nb): (usize, usize, usize),
    ksplit: usize,
    ctas: usize,
    out: &DevBuf,
) -> A1Plan<'m> {
    a1_layout(
        &[
            ("fp8 weights", dev_addr(w)),
            ("fp8 activations", dev_addr(x)),
        ],
        &[("fp8 weights", cols), ("fp8 activations", cols)],
        cols,
    )
    .unwrap();
    let mw = g
        .tensor_map_2d(
            dev_addr(w),
            (cols as u64, rows as u64),
            cols as u64,
            (128, 128),
            true,
        )
        .unwrap();
    let mx = g
        .tensor_map_2d(
            dev_addr(x),
            (cols as u64, nb as u64),
            cols as u64,
            (128, v.toks as u32),
            true,
        )
        .unwrap();
    let f = pm.function(&v.kernel()).unwrap();
    f.set_max_shared(v.smem()).unwrap();
    let mut ls = a1_zero(pm, out, rows * nb, ksplit);
    ls.push(A1Launch {
        f,
        grid: (ctas as u32, 1, 1),
        block: (384, 1, 1),
        smem: v.smem() as u32,
        args: vec![
            Arg::Blob128(mw),
            Arg::Blob128(mx),
            ws.arg(),
            xs.arg(),
            Arg::I32(rows as i32),
            Arg::I32(cols as i32),
            Arg::I32(nb as i32),
            Arg::I32(ksplit as i32),
            out.arg(),
        ],
    });
    A1Plan {
        name: a1_name(v.name, ksplit),
        ls,
    }
}

/// probe_w4a16_<R>x64 (v) on repacked weights (nib, scl: a1_q4f_pack of rows
/// x cols, padded to rows_p) and nb tokens of fp16 activations `x` (stride 2
/// cols).
#[allow(clippy::too_many_arguments)]
fn a1_plan_w4a16<'m>(
    g: &Gpu,
    pm: &'m Module,
    v: A1Var,
    nib: &DevBuf,
    scl: &DevBuf,
    x: &DevBuf,
    (rows, rows_p, cols, nb): (usize, usize, usize, usize),
    ksplit: usize,
    ctas: usize,
    out: &DevBuf,
) -> A1Plan<'m> {
    a1_layout(
        &[
            ("w4a16 nibbles", dev_addr(nib)),
            ("w4a16 scales", dev_addr(scl)),
            ("fp16 activations", dev_addr(x)),
        ],
        &[
            ("nibbles", 128),
            ("scales", 32),
            ("fp16 activations", 2 * cols),
        ],
        cols,
    )
    .unwrap();
    assert!(rows_p.is_multiple_of(v.rows) && rows_p >= rows);
    let (lines, rt) = ((cols / 256 * rows_p) as u64, v.rows as u32);
    let mn = g
        .tensor_map_2d(dev_addr(nib), (128, lines), 128, (128, rt), false)
        .unwrap();
    let msc = g
        .tensor_map_2d(dev_addr(scl), (32, lines), 32, (32, rt), false)
        .unwrap();
    let mx = g
        .tensor_map_2d(
            dev_addr(x),
            (2 * cols as u64, nb as u64),
            2 * cols as u64,
            (128, 64),
            true,
        )
        .unwrap();
    let f = pm.function(&v.kernel()).unwrap();
    f.set_max_shared(v.smem()).unwrap();
    let mut ls = a1_zero(pm, out, rows * nb, ksplit);
    ls.push(A1Launch {
        f,
        grid: (ctas as u32, 1, 1),
        block: (384, 1, 1),
        smem: v.smem() as u32,
        args: vec![
            Arg::Blob128(mn),
            Arg::Blob128(msc),
            Arg::Blob128(mx),
            Arg::I32(rows as i32),
            Arg::I32(rows_p as i32),
            Arg::I32(cols as i32),
            Arg::I32(nb as i32),
            Arg::I32(ksplit as i32),
            out.arg(),
        ],
    });
    A1Plan {
        name: a1_name(v.name, ksplit),
        ls,
    }
}

/// Baseline gemm2s_q4_k (main's mma.sync Q4_K x Q8_K product), as
/// launch_product launches it.
#[allow(clippy::too_many_arguments)]
fn a1_plan_gemm2s<'m>(
    km: &'m Module,
    w: &DevBuf,
    (rows, cols, nb): (usize, usize, usize),
    act: &DevBuf,
    st: usize,
    out: &DevBuf,
) -> A1Plan<'m> {
    let f = km.function("gemm2s_q4_k").unwrap();
    let sm = gemm2_smem(Q4_K, 64, 3);
    f.set_max_shared(sm).unwrap();
    A1Plan {
        name: "gemm2s_q4_k".into(),
        ls: vec![A1Launch {
            f,
            grid: (nb.div_ceil(64) as u32, rows.div_ceil(G2_BM) as u32, 1),
            block: (256, 1, 1),
            smem: sm as u32,
            args: vec![
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
            ],
        }],
    }
}

/// Baseline gemmh2_q4_k (main's exact wgmma kernel), as launch_gemmh
/// launches it.
fn a1_plan_gemmh2<'m>(
    g: &Gpu,
    km: &'m Module,
    w: &DevBuf,
    (rows, cols, nb): (usize, usize, usize),
    act: &DevBuf,
    st: usize,
    out: &DevBuf,
) -> A1Plan<'m> {
    let rb = (cols / 256 * 144) as u64;
    let mw = g
        .tensor_map_2d(dev_addr(w), (rb, rows as u64), rb, (144, 128), false)
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
    let f = km.function("gemmh2_q4_k").unwrap();
    f.set_max_shared(GEMMH2_SMEM).unwrap();
    A1Plan {
        name: "gemmh2_q4_k".into(),
        ls: vec![A1Launch {
            f,
            grid: (nb.div_ceil(64) as u32, rows.div_ceil(128) as u32, 1),
            block: (256, 1, 1),
            smem: GEMMH2_SMEM as u32,
            args: vec![
                Arg::Blob128(mw),
                Arg::Blob128(mx),
                act.arg(),
                Arg::U64(st as u64),
                Arg::I32(rows as i32),
                Arg::I32(cols as i32),
                Arg::I32(nb as i32),
                Arg::Ptr(0),
                Arg::I32(0),
                out.arg(),
                Arg::U64(rows as u64),
            ],
        }],
    }
}

/// A candidate's inputs on host and device for one (shape, nb).
enum A1Data<'g> {
    /// e4m3 weights / activations and their f32 scales.
    Fp8 {
        w: Vec<u8>,
        ws: Vec<f32>,
        x: Vec<u8>,
        xs: Vec<f32>,
        dev: [DevBuf<'g>; 4],
    },
    /// Q4_K weights (host, for the reference), their repacked device copy
    /// (nibbles, scales), fp16 activations.
    W4a16 {
        w: Vec<u8>,
        x: Vec<u16>,
        rows_p: usize,
        dev: [DevBuf<'g>; 3],
    },
}

/// FP8 inputs: random codes (a1_e4m3_codes), scales in [0.5, 1.5) 2^-6.
fn a1_fp8_data<'g>(g: &'g Gpu, rng: &mut Rng, rows: usize, cols: usize, nb: usize) -> A1Data<'g> {
    let (w, ws) = (a1_e4m3_codes(rng, rows * cols), a1_scales(rng, rows));
    let (x, xs) = (a1_e4m3_codes(rng, nb * cols), a1_scales(rng, nb));
    let dev = [
        g.upload(&w).unwrap(),
        g.upload(&ws).unwrap(),
        g.upload(&x).unwrap(),
        g.upload(&xs).unwrap(),
    ];
    A1Data::Fp8 { w, ws, x, xs, dev }
}

/// W4A16 inputs: Q4_K `w` repacked, activations `xf` (nb x cols) to fp16.
fn a1_w4a16_data<'g>(
    g: &'g Gpu,
    w: &[u8],
    rows: usize,
    cols: usize,
    xf: &[Vec<f32>],
) -> A1Data<'g> {
    let p = a1_q4f_pack(w, rows, cols);
    let x: Vec<u16> = xf.iter().flatten().map(|&v| quant::f32_to_f16(v)).collect();
    let dev = [
        g.upload(&p.nib).unwrap(),
        g.upload(&p.scl).unwrap(),
        g.upload(&x).unwrap(),
    ];
    A1Data::W4a16 {
        w: w.to_vec(),
        x,
        rows_p: p.rows_p,
        dev,
    }
}

impl A1Data<'_> {
    fn tol(&self) -> (f64, f64) {
        match self {
            A1Data::Fp8 { .. } => A1_TOL_FP8,
            A1Data::W4a16 { .. } => A1_TOL_W4A16,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn plan<'m>(
        &self,
        g: &Gpu,
        pm: &'m Module,
        v: A1Var,
        (rows, cols, nb): (usize, usize, usize),
        ksplit: usize,
        ctas: usize,
        out: &DevBuf,
    ) -> A1Plan<'m> {
        match self {
            A1Data::Fp8 { dev, .. } => {
                assert!(v.fp8);
                a1_plan_fp8(
                    g,
                    pm,
                    v,
                    &dev[0],
                    &dev[1],
                    &dev[2],
                    &dev[3],
                    (rows, cols, nb),
                    ksplit,
                    ctas,
                    out,
                )
            }
            A1Data::W4a16 { dev, rows_p, .. } => {
                assert!(!v.fp8);
                a1_plan_w4a16(
                    g,
                    pm,
                    v,
                    &dev[0],
                    &dev[1],
                    &dev[2],
                    (rows, *rows_p, cols, nb),
                    ksplit,
                    ctas,
                    out,
                )
            }
        }
    }

    /// The CPU f64 reference at the (token z, row r) pairs, as (z, r, value).
    fn reference(&self, cols: usize, pairs: &[(usize, usize)]) -> Vec<(usize, usize, f64)> {
        let mut ps = pairs.to_vec();
        ps.sort_by_key(|&(z, r)| (r, z));
        let mut out = Vec::with_capacity(ps.len());
        match self {
            A1Data::Fp8 { w, ws, x, xs, .. } => {
                let lut: Vec<f64> = (0..=255u8).map(|c| e4m3_f32(c) as f64).collect();
                for (z, r) in ps {
                    let s: f64 = w[r * cols..][..cols]
                        .iter()
                        .zip(&x[z * cols..][..cols])
                        .map(|(&a, &b)| lut[a as usize] * lut[b as usize])
                        .sum();
                    out.push((z, r, s * ws[r] as f64 * xs[z] as f64));
                }
            }
            A1Data::W4a16 { w, x, .. } => {
                let rb = cols / 256 * 144;
                let mut row = vec![0f32; cols];
                let mut cur = usize::MAX;
                for (z, r) in ps {
                    if r != cur {
                        quant::dequant(Q4_K, &w[r * rb..][..rb], &mut row);
                        cur = r;
                    }
                    let s: f64 = row
                        .iter()
                        .zip(&x[z * cols..][..cols])
                        .map(|(&a, &b)| a as f64 * quant::f16_to_f32(b) as f64)
                        .sum();
                    out.push((z, r, s));
                }
            }
        }
        out
    }
}

/// (rel-rms, rel-max) of |got - want| over the reference points, relative to
/// the reference's rms; a NaN counts as infinite.
fn a1_err(got: &[f32], rows: usize, want: &[(usize, usize, f64)]) -> (f64, f64) {
    let (mut se, mut sw, mut mx) = (0f64, 0f64, 0f64);
    for &(z, r, v) in want {
        let e = (got[z * rows + r] as f64 - v).abs();
        let e = if e.is_nan() { f64::INFINITY } else { e };
        se += e * e;
        sw += v * v;
        mx = mx.max(e);
    }
    let rms = (sw / want.len() as f64).sqrt().max(f64::MIN_POSITIVE);
    ((se / want.len() as f64).sqrt() / rms, mx / rms)
}

/// The tolerance verdict text: "tol PASS rel-rms .. rel-max .." (or FAIL).
fn a1_tol_text((rr, rm): (f64, f64), (tr, tm): (f64, f64)) -> (bool, String) {
    let ok = rr <= tr && rm <= tm;
    (
        ok,
        format!(
            "tol {} rel-rms {rr:.2e} rel-max {rm:.2e} (bounds {tr} / {tm})",
            if ok { "PASS" } else { "FAIL" }
        ),
    )
}

/// Run each (rows, cols, nb, ctas, ksplit) case of variant `v` once,
/// every output against the CPU reference; prints `A1 RAGGED` lines (and
/// `A1 TOL FAIL` on a miss). Returns the number of failed cases.
fn a1_check(
    g: &Gpu,
    pm: &Module,
    v: A1Var,
    cases: &[(usize, usize, usize, usize, usize)],
) -> usize {
    let mut rng = Rng(0xa1c);
    let mut bad = 0;
    for &(rows, cols, nb, ctas, ksplit) in cases {
        let data = if v.fp8 {
            a1_fp8_data(g, &mut rng, rows, cols, nb)
        } else {
            let w = synth_matrix(&mut rng, Q4_K, rows, cols);
            let xf: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 1.0)).collect();
            a1_w4a16_data(g, &w, rows, cols, &xf)
        };
        // NaN first: a missed output can't read as a stale right answer.
        let out = g.upload(&vec![f32::NAN; rows * nb]).unwrap();
        let plan = data.plan(g, pm, v, (rows, cols, nb), ksplit, ctas, &out);
        // SAFETY: the plan's buffers (data, out) live until the sync below.
        unsafe { plan.fire(g) };
        g.sync().unwrap();
        let got = out.read::<f32>().unwrap();
        let pairs: Vec<(usize, usize)> = (0..nb)
            .flat_map(|z| (0..rows).map(move |r| (z, r)))
            .collect();
        let want = data.reference(cols, &pairs);
        let (ok, txt) = a1_tol_text(a1_err(&got, rows, &want), data.tol());
        let what = format!(
            "{} {rows}x{cols} nb {nb} ctas {ctas} ksplit {ksplit}",
            plan.name
        );
        eprintln!("A1 RAGGED {what}: {} outputs, {txt}", pairs.len());
        if !ok {
            eprintln!("A1 TOL FAIL {what}");
            bad += 1;
        }
    }
    bad
}

/// The tiny cases (what the session runs under compute-sanitizer memcheck
/// first): one tile, one CTA; split-K over 2 CTAs; stream-K over 3 CTAs on
/// 2 tiles; stream-K with more CTAs than K-blocks.
const A1_TINY: [(usize, usize, usize, usize, usize); 4] = [
    (128, 256, 64, 1, 1),
    (128, 512, 64, 2, 2),
    (256, 768, 64, 3, 0),
    (128, 256, 64, 4, 0),
];

/// The ragged cases, every output checked: rows / tokens off the 128 x 64
/// tile, 1 to 74 K-blocks, few CTAs (several tiles each: the persistent loop,
/// the ring wrapping across tiles and ranges), split-K 2 / 3 / 14, stream-K
/// with ranges cut inside tiles.
const A1_RAGGED: [(usize, usize, usize, usize, usize); 10] = [
    (128, 256, 64, 1, 1),
    (200, 768, 100, 3, 1),
    (384, 1280, 129, 2, 1),
    (128, 4352, 64, 1, 1),
    (512, 3584, 200, 7, 1),
    (200, 3584, 100, 5, 2),
    (384, 4352, 129, 7, 3),
    (136, 3584, 72, 11, 14),
    (512, 3584, 200, 11, 0),
    (256, 18944, 64, 13, 0),
];

/// Variant `name`'s test over `cases`, skipping off sm_90a.
fn a1_gpu_check(name: &str, cases: &[(usize, usize, usize, usize, usize)]) {
    let v = *A1_VARS.iter().find(|v| v.name == name).unwrap();
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(pm) = probe_fast_module(&g) else {
        eprintln!("skipping: not sm_90a");
        return;
    };
    let bad = a1_check(&g, &pm, v, cases);
    assert_eq!(bad, 0, "a1 {name}: tolerance failures");
}

/// probe_fp8_128x64, tiny shapes (sm_90a only; skips elsewhere).
#[test]
fn gpu_a1_fp8_128x64_tiny() {
    a1_gpu_check("fp8_128x64", &A1_TINY);
}

/// probe_fp8_128x128, tiny shapes.
#[test]
fn gpu_a1_fp8_128x128_tiny() {
    a1_gpu_check("fp8_128x128", &A1_TINY);
}

/// probe_fp8n128_128x128 (m64n128k32), tiny shapes.
#[test]
fn gpu_a1_fp8n128_128x128_tiny() {
    a1_gpu_check("fp8n128_128x128", &A1_TINY);
}

/// probe_w4a16_128x64, tiny shapes.
#[test]
fn gpu_a1_w4a16_128x64_tiny() {
    a1_gpu_check("w4a16_128x64", &A1_TINY);
}

/// probe_w4a16_256x64, tiny shapes.
#[test]
fn gpu_a1_w4a16_256x64_tiny() {
    a1_gpu_check("w4a16_256x64", &A1_TINY);
}

/// probe_fp8_128x64 against the CPU at every output of the ragged shapes.
#[test]
fn gpu_a1_fp8_128x64_ragged() {
    a1_gpu_check("fp8_128x64", &A1_RAGGED);
}

/// probe_fp8_128x128, the ragged shapes.
#[test]
fn gpu_a1_fp8_128x128_ragged() {
    a1_gpu_check("fp8_128x128", &A1_RAGGED);
}

/// probe_fp8n128_128x128, the ragged shapes.
#[test]
fn gpu_a1_fp8n128_128x128_ragged() {
    a1_gpu_check("fp8n128_128x128", &A1_RAGGED);
}

/// probe_w4a16_128x64, the ragged shapes.
#[test]
fn gpu_a1_w4a16_128x64_ragged() {
    a1_gpu_check("w4a16_128x64", &A1_RAGGED);
}

/// probe_w4a16_256x64, the ragged shapes.
#[test]
fn gpu_a1_w4a16_256x64_ragged() {
    a1_gpu_check("w4a16_256x64", &A1_RAGGED);
}

/// The bench's result line for one cell (a1_verdict.py's CELL pattern; the
/// format is pinned by a1_cell_line_pinned and scripts/a1/a1_battery.py):
/// medians to 2 decimals (a 349.995 must not print as 350.0), the median
/// time per fire in ms, then the tolerance text for a candidate.
#[allow(clippy::too_many_arguments)]
fn a1_cell_line(
    what: &str,
    (cols, rows, nb): (usize, usize, usize),
    name: &str,
    (med, lo, hi): (f64, f64, f64),
    rounds: usize,
    tmed_ms: f64,
    tol: &str,
) -> String {
    format!(
        "A1 CELL {what} {cols}x{rows} nb {nb} {name}: median {med:.2} TMAC/s (min {lo:.2} max {hi:.2}; {rounds} rounds; {tmed_ms:.4} ms){tol}"
    )
}

/// a1_cell_line's format, as scripts/a1/a1_battery.py's FORMAT case has it
/// verbatim (the battery checks a1_verdict.py parses it): no GPU.
#[test]
fn a1_cell_line_pinned() {
    let (ok, tol) = a1_tol_text((1.234e-2, 5.0e-2), A1_TOL_FP8);
    assert!(ok);
    assert_eq!(
        a1_cell_line(
            "gate",
            (3584, 18944, 1024),
            "fp8n128_128x128",
            (380.004, 377.5, 383.25),
            9,
            0.18291,
            &format!(" {tol}")
        ),
        "A1 CELL gate 3584x18944 nb 1024 fp8n128_128x128: median 380.00 TMAC/s (min 377.50 max 383.25; 9 rounds; 0.1829 ms) tol PASS rel-rms 1.23e-2 rel-max 5.00e-2 (bounds 0.1 / 0.5)"
    );
    let battery = include_str!("../../../scripts/a1/a1_battery.py");
    assert!(
        battery.contains(
            "A1 CELL gate 3584x18944 nb 1024 fp8n128_128x128: median 380.00 TMAC/s (min 377.50 max 383.25; 9 rounds; 0.1829 ms) tol PASS rel-rms 1.23e-2 rel-max 5.00e-2 (bounds 0.1 / 0.5)"
        ),
        "scripts/a1/a1_battery.py's FORMAT line is not a1_cell_line's"
    );
}

/// (median, min, max) of v (sorted in place).
fn a1_stats(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

/// One child of cuda_a1_session: the ceilings of the probe's three wgmma
/// forms at 2 warpgroups per SM (the probe's consumer count), raw MACs:
/// a1_peak_fp8_ss_n64 (SS m64n64k32 e4m3), a1_peak_fp8_ss_n128 (SS
/// m64n128k32 e4m3) and a1_peak_f16_rs_n64 (RS m64n64k16 f16). Recorded, not
/// a rule input. sm_90a only.
#[test]
#[ignore]
fn cuda_a1_peak() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let info = g.info().unwrap();
    let Some(pm) = probe_fast_module(&g) else {
        eprintln!("A1 SKIP peaks (no sm_90a device)");
        eprintln!("A1 PEAK END ok");
        return;
    };
    let rounds: usize = std::env::var("AXON_A1_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);
    let (iters, wgs, blocks) = (4096usize, 2usize, info.sms as usize);
    let o = g.alloc(4 * 4096).unwrap();
    for (k, kmac) in [
        ("a1_peak_fp8_ss_n64", 64 * 64 * 32usize),
        ("a1_peak_fp8_ss_n128", 64 * 128 * 32),
        ("a1_peak_f16_rs_n64", 64 * 64 * 16),
    ] {
        let f = pm.function(k).unwrap();
        let mut v = Vec::new();
        for _ in 0..rounds + 1 {
            let e0 = g.record().unwrap();
            // SAFETY: (int iters, int* out); o has a slot per block.
            unsafe {
                g.launch(
                    &f,
                    (blocks as u32, 1, 1),
                    (128 * wgs as u32, 1, 1),
                    0,
                    &[Arg::I32(iters as i32), o.arg()],
                )
                .unwrap();
            }
            let e1 = g.record().unwrap();
            let ms = e0.ms_to(&e1).unwrap() as f64;
            v.push((blocks * wgs * iters * 4 * kmac) as f64 / (ms * 1e-3) / 1e12);
        }
        v.remove(0); // the first launch warms up
        let (med, lo, hi) = a1_stats(&mut v);
        eprintln!(
            "A1 PEAK {k} {wgs} WG/SM: median {med:.1} TMAC/s raw (min {lo:.1} max {hi:.1}; {rounds} rounds)"
        );
    }
    eprintln!("A1 PEAK END ok");
}

/// The bench cells: (name, K, rows, nbs). nb 128 also runs the (c)
/// variants: split-K and stream-K on down, q/wo and kv; stream-K on gate
/// (information).
const A1_CELLS: [(&str, usize, usize, &[usize]); 4] = [
    ("gate", 3584, 18944, &[128, 512, 1024]),
    ("down", 18944, 3584, &[128, 512, 1024]),
    ("q/wo", 3584, 3584, &[128, 1024]),
    ("kv", 3584, 512, &[128, 1024]),
];

/// The split-K factor variant v's (c) cells use: enough units to fill one
/// wave of CTAs, at least 2, at most one K-block each.
fn a1_split(v: A1Var, rows: usize, cols: usize, nb: usize, sms: usize) -> usize {
    (sms / v.tiles(rows, nb)).clamp(2, cols / 256)
}

/// One child of cuda_a1_session: the candidates against the baselines, the
/// same rounds interleaved, every cell of A1_CELLS. Per round each plan runs
/// once to warm, then 3 times between two events; median of
/// AXON_A1_ROUNDS (9) rounds, TMAC/s of real MACs (rows K nb / time). The
/// candidate variants (AXON_A1_KERNELS, default all of A1_VARS; "none" =
/// baselines only) are checked against the CPU at 4096 sampled outputs + corners
/// after the last round (`tol PASS` / `FAIL`; a FAIL cell doesn't count for
/// the rules). Baselines: gemm2s_q4_k (any GPU) and gemmh2_q4_k (sm_90a),
/// on the same Q4_K weights and the same activations (Q8_K) as W4A16's.
/// AXON_A1_SHAPE: only cells whose name contains it. Off sm_90a the
/// candidates are skipped (`A1 SKIP`): the rehearsal. Exits 124 after
/// AXON_A1_TIMEOUT_S (900) whatever runs.
#[test]
#[ignore]
fn cuda_a1_bench() {
    let lim: u64 = std::env::var("AXON_A1_TIMEOUT_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(900);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(lim));
        eprintln!("A1 TIMEOUT after {lim} s: exiting");
        std::process::exit(124);
    });
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let info = g.info().unwrap();
    let km = Kernels::module_ext(&g).unwrap();
    let pm = probe_fast_module(&g);
    let all: Vec<&str> = A1_VARS.iter().map(|v| v.name).collect();
    let kset = std::env::var("AXON_A1_KERNELS").unwrap_or(all.join(","));
    let on = |c: &str| pm.is_some() && kset.split(',').any(|k| k.trim() == c);
    let vars: Vec<A1Var> = A1_VARS.iter().copied().filter(|v| on(v.name)).collect();
    let rounds: usize = std::env::var("AXON_A1_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);
    let shape_ok = |s: &str| std::env::var("AXON_A1_SHAPE").map_or(true, |f| s.contains(&f));
    let sms = info.sms as usize;
    eprintln!(
        "A1 BENCH device {} sm_{}{} {} SMs; hopper module {}, probe module {}; candidates [{}]; {rounds} rounds",
        info.name,
        info.cc.0,
        info.cc.1,
        sms,
        km.hopper(),
        pm.is_some(),
        vars.iter().map(|v| v.name).collect::<Vec<_>>().join(",")
    );
    for c in all {
        if !on(c) {
            eprintln!(
                "A1 SKIP {c} cells ({})",
                if pm.is_none() {
                    "no sm_90a device"
                } else {
                    "not in AXON_A1_KERNELS"
                }
            );
        }
    }
    if !km.hopper() {
        eprintln!("A1 SKIP gemmh2_q4_k cells (no sm_90a module)");
    }
    let mut rng = Rng(0xa1);
    let mut tol_bad = 0;
    for &(what, cols, rows, nbs) in &A1_CELLS {
        if !shape_ok(what) {
            continue;
        }
        let wq = synth_matrix(&mut rng, Q4_K, rows, cols);
        let w = g.upload(&wq).unwrap();
        for &nb in nbs {
            let shape = (rows, cols, nb);
            let xf: Vec<Vec<f32>> = (0..nb).map(|_| rng.vec(cols, 1.0)).collect();
            let act = quantize_batch(&g, &km, &xf, cols);
            let st = act_stride_t(cols);
            let base_out = g.alloc(rows * nb * 4).unwrap();
            // One data set per family (both tiles of a family share it).
            let mut data = Vec::new();
            let mut fam = [None, None]; // data index of fp8, w4a16
            if vars.iter().any(|v| v.fp8) {
                fam[0] = Some(data.len());
                data.push(a1_fp8_data(&g, &mut rng, rows, cols, nb));
            }
            if vars.iter().any(|v| !v.fp8) {
                fam[1] = Some(data.len());
                data.push(a1_w4a16_data(&g, &wq, rows, cols, &xf));
            }
            // Each variant: tiles everywhere; at nb 128 split-K and stream-K
            // (down, q/wo, kv) or stream-K only (gate).
            let mut outs = Vec::new(); // (variant, data index, ksplit, out buffer)
            for &v in &vars {
                let di = fam[usize::from(!v.fp8)].unwrap();
                let mut ks = vec![1usize];
                if nb == 128 {
                    if what != "gate" {
                        ks.push(a1_split(v, rows, cols, nb, sms));
                    }
                    ks.push(0);
                }
                for k in ks {
                    // NaN first: an output no launch writes reads as NaN
                    // (rel-rms inf: tol FAIL), never as a recycled allocation's
                    // stale right answer.
                    outs.push((v, di, k, g.upload(&vec![f32::NAN; rows * nb]).unwrap()));
                }
            }
            let mut plans = vec![a1_plan_gemm2s(&km, &w, shape, &act, st, &base_out)];
            if km.hopper() {
                plans.push(a1_plan_gemmh2(&g, &km, &w, shape, &act, st, &base_out));
            }
            let nbase = plans.len();
            if let Some(pm) = pm.as_ref() {
                for (v, di, k, out) in &outs {
                    plans.push(data[*di].plan(&g, pm, *v, shape, *k, sms, out));
                }
            }
            let mut ms: Vec<Vec<f64>> = vec![Vec::new(); plans.len()];
            for _ in 0..rounds {
                for (pi, p) in plans.iter().enumerate() {
                    // SAFETY: every plan's buffers (w, act, data, outs) live to the end of the cell.
                    unsafe { p.fire(&g) };
                    let e0 = g.record().unwrap();
                    for _ in 0..3 {
                        // SAFETY: as above.
                        unsafe { p.fire(&g) };
                    }
                    let e1 = g.record().unwrap();
                    ms[pi].push(e0.ms_to(&e1).unwrap() as f64 / 3.0);
                }
            }
            g.sync().unwrap();
            // The tolerance checks: 4096 sampled outputs + the tile corners,
            // one reference per candidate for all its variants.
            let mut srng = Rng(0x5eed ^ (rows * 31 + nb) as u64);
            let mut pairs: Vec<(usize, usize)> = (0..4096)
                .map(|_| ((srng.next() as usize) % nb, (srng.next() as usize) % rows))
                .collect();
            for z in [0, 63.min(nb - 1), nb - 1] {
                for r in [0, 127.min(rows - 1), rows - 1] {
                    pairs.push((z, r));
                }
            }
            let refs: Vec<Vec<(usize, usize, f64)>> =
                data.iter().map(|d| d.reference(cols, &pairs)).collect();
            let macs = (rows * cols * nb) as f64;
            for (pi, p) in plans.iter().enumerate() {
                let mut v: Vec<f64> = ms[pi].iter().map(|x| macs / (x * 1e-3) / 1e12).collect();
                let (med, lo, hi) = a1_stats(&mut v);
                let mut mt = ms[pi].clone();
                let (tmed, _, _) = a1_stats(&mut mt);
                let mut tag = String::new();
                if pi >= nbase {
                    let (_, di, _, out) = &outs[pi - nbase];
                    let got = out.read::<f32>().unwrap();
                    let (ok, txt) = a1_tol_text(a1_err(&got, rows, &refs[*di]), data[*di].tol());
                    if !ok {
                        tol_bad += 1;
                    }
                    tag = format!(" {txt}");
                }
                eprintln!(
                    "{}",
                    a1_cell_line(
                        what,
                        (cols, rows, nb),
                        &p.name,
                        (med, lo, hi),
                        rounds,
                        tmed,
                        &tag
                    )
                );
            }
        }
    }
    eprintln!(
        "A1 BENCH END {}",
        if tol_bad == 0 {
            "ok".to_string()
        } else {
            format!("ok, {tol_bad} cells out of tolerance (they don't count)")
        }
    );
}

/// Run this test binary's `test` in a child process (its own CUDA context),
/// optionally under `wrap` (compute-sanitizer), output teed into this log and
/// kept; killed after `limit_s`. Returns (rc, seconds, output); rc -1 = killed
/// on timeout. (T3's probe_child.)
fn a1_child(
    wrap: &[String],
    test: &str,
    ignored: bool,
    env: &[(&str, &str)],
    limit_s: u64,
) -> (i32, f64, String) {
    let exe = std::env::current_exe().unwrap();
    let mut args: Vec<String> = wrap.to_vec();
    args.push(exe.display().to_string());
    args.extend([
        format!("cuda::qwen2::tests::{test}"),
        "--exact".into(),
        "--nocapture".into(),
        "--test-threads".into(),
        "1".into(),
    ]);
    if ignored {
        args.push("--ignored".into());
    }
    let unix = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64())
    };
    eprintln!(
        "A1 STAGE {test}: start at unix {:.1}, limit {limit_s} s ({})",
        unix(),
        args.join(" ")
    );
    let t = std::time::Instant::now();
    let mut c = std::process::Command::new(&args[0]);
    c.args(&args[1..])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    for (k, v) in env {
        c.env(k, v);
    }
    let mut child = match c.spawn() {
        Ok(ch) => ch,
        Err(e) => {
            eprintln!("A1 STAGE {test}: spawn failed: {e}");
            return (-2, 0.0, String::new());
        }
    };
    let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let tee = |r: Box<dyn std::io::Read + Send>| {
        let seen = seen.clone();
        std::thread::spawn(move || {
            use std::io::BufRead;
            for l in std::io::BufReader::new(r).lines().map_while(Result::ok) {
                eprintln!("{l}");
                let mut s = seen.lock().unwrap();
                s.push_str(&l);
                s.push('\n');
            }
        })
    };
    let t1 = tee(Box::new(child.stdout.take().unwrap()));
    let t2 = tee(Box::new(child.stderr.take().unwrap()));
    let rc = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break st.code().unwrap_or(-3);
        }
        if t.elapsed().as_secs() >= limit_s {
            let _ = child.kill();
            let _ = child.wait();
            break -1;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    let _ = (t1.join(), t2.join());
    let s = t.elapsed().as_secs_f64();
    eprintln!(
        "A1 STAGE {test}: rc {rc} after {s:.1} s at unix {:.1}{}",
        unix(),
        if rc == -1 { " (TIMEOUT, killed)" } else { "" }
    );
    let out = seen.lock().unwrap().clone();
    (rc, s, out)
}

/// How a candidate's stage ended: "ok", "skipped (no sm_90a)", "tolerance"
/// (ran, no fault, out of tolerance: its bench still runs, its cells carry
/// their own tol) or "FAULT" (CUDA error, trap, sanitizer error, timeout,
/// anything else: the candidate is left out of the bench).
fn a1_outcome(rc: i32, out: &str) -> &'static str {
    if out.contains("skipping: not sm_90a") && rc == 0 {
        "skipped (no sm_90a)"
    } else if rc == 0 {
        "ok"
    } else if rc == 101
        && out.contains("A1 TOL FAIL")
        && !out.contains("CUDA_ERROR")
        && !out.contains("CUresult ")
        && (!out.contains("ERROR SUMMARY:") || out.contains("ERROR SUMMARY: 0 errors"))
    {
        "tolerance"
    } else {
        "FAULT"
    }
}

/// WP-A1's ONE H100 job, stages in separate processes so a faulting kernel
/// can't take the rest down (each child has its own CUDA context), cheapest
/// and most likely to break first:
/// 0. the probe's sm_90a NVRTC compile check (no GPU; a failure is recorded
///    and the session goes on: the kernels come from the kit's cubin);
/// 1. with AXON_A1_SANITIZER=<compute-sanitizer>: a self-check (memcheck
///    over gpu_rmsnorm_matches_cpu: unusable -> stage dropped, not failed),
///    then each candidate variant's tiny shapes under memcheck;
/// 2. each variant's tiny shapes, then its ragged shapes (every output);
/// 3. cuda_a1_peak (the two wgmma forms' ceilings);
/// 4. cuda_a1_bench with the variants that didn't fault; if it fails, once
///    per remaining variant, then baselines only (AXON_A1_REHEARSE_FALLBACK
///    =1 takes that path anyway: the rehearsal).
///
/// A fault leaves its variant out of the bench; a tolerance miss doesn't
/// (its cells report their own tolerance). Every stage has a time limit and
/// the whole session AXON_A1_BUDGET_S (2400 s = 40 min): a stage that would
/// overrun it is cut to what's left (or skipped under 30 s). Off sm_90a
/// every candidate stage skips. Fails (after everything has run) if a stage
/// failed. Run with --ignored --nocapture.
#[test]
#[ignore]
fn cuda_a1_session() {
    let t0 = std::time::Instant::now();
    let budget: u64 = std::env::var("AXON_A1_BUDGET_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2400);
    let left = |want: u64| -> Option<u64> {
        let rest = budget.saturating_sub(t0.elapsed().as_secs());
        (rest >= 30).then(|| want.min(rest))
    };
    let mut fails: Vec<String> = Vec::new();
    let mut fault = std::collections::BTreeSet::new();
    let run = |wrap: &[String],
               test: &str,
               ignored: bool,
               env: &[(&str, &str)],
               want: u64|
     -> (i32, String) {
        match left(want) {
            Some(l) => {
                let (rc, _, out) = a1_child(wrap, test, ignored, env, l);
                (rc, out)
            }
            None => {
                eprintln!("A1 STAGE {test}: SKIPPED (session budget {budget} s spent)");
                (-4, String::new())
            }
        }
    };
    let (rc, _) = run(&[], "nvrtc_probe_fast_compiles_sm90a", false, &[], 180);
    eprintln!(
        "A1 STAGE compile: {}",
        if rc == 0 {
            "ok"
        } else {
            "FAILED (recorded; the kit's cubin is what runs)"
        }
    );
    if rc != 0 {
        fails.push(format!("compile check rc {rc}"));
    }
    if let Some(san) = std::env::var_os("AXON_A1_SANITIZER") {
        let san = san.to_string_lossy().to_string();
        let wrap: Vec<String> = [
            san.as_str(),
            "--tool",
            "memcheck",
            "--error-exitcode",
            "9",
            "--print-limit",
            "20",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let usable = std::path::Path::new(&san).is_file() && {
            let (rc, out) = run(&wrap, "gpu_rmsnorm_matches_cpu", false, &[], 120);
            rc == 0
                && out.contains("ERROR SUMMARY: 0 errors")
                && !out.contains("terminated before first instrumented")
                && out.contains("1 passed")
        };
        if !usable {
            eprintln!("A1 STAGE sanitizer-selfcheck: UNUSABLE ({san}): sanitizer stages dropped");
        } else {
            eprintln!(
                "A1 STAGE sanitizer-selfcheck: ok (memcheck instrumented gpu_rmsnorm_matches_cpu, 0 errors)"
            );
            for c in A1_VARS.map(|v| v.name) {
                let (rc, out) = run(&wrap, &format!("gpu_a1_{c}_tiny"), false, &[], 150);
                let o = a1_outcome(rc, &out);
                let o = if o == "ok" && !out.contains("ERROR SUMMARY: 0 errors") {
                    "FAULT"
                } else {
                    o
                };
                eprintln!("A1 STAGE sanitizer {c} tiny: {o}");
                if o == "FAULT" {
                    fault.insert(c);
                }
                if o == "FAULT" || o == "tolerance" {
                    fails.push(format!("sanitizer {c} tiny {o} (rc {rc})"));
                }
            }
        }
    } else {
        eprintln!("A1 STAGE sanitizer: AXON_A1_SANITIZER unset, skipped");
    }
    for c in A1_VARS.map(|v| v.name) {
        for (part, want) in [("tiny", 90u64), ("ragged", 240)] {
            if fault.contains(c) {
                eprintln!("A1 STAGE {c} {part}: not run ({c} faulted before)");
                continue;
            }
            let (rc, out) = run(&[], &format!("gpu_a1_{c}_{part}"), false, &[], want);
            let o = a1_outcome(rc, &out);
            eprintln!("A1 STAGE {c} {part}: {o}");
            if o == "FAULT" {
                fault.insert(c);
            }
            if o == "FAULT" || o == "tolerance" {
                fails.push(format!("{c} {part} {o} (rc {rc})"));
            }
        }
    }
    let (rc, _) = run(&[], "cuda_a1_peak", true, &[], 120);
    if rc != 0 {
        fails.push(format!("peak rc {rc}"));
    }
    let healthy: Vec<&str> = A1_VARS
        .map(|v| v.name)
        .into_iter()
        .filter(|c| !fault.contains(c))
        .collect();
    let kset = if healthy.is_empty() {
        "none".to_string()
    } else {
        healthy.join(",")
    };
    eprintln!("A1 STAGE bench: candidates {kset}");
    let (rc, _) = run(
        &[],
        "cuda_a1_bench",
        true,
        &[("AXON_A1_KERNELS", &kset)],
        720,
    );
    let rehearse = std::env::var("AXON_A1_REHEARSE_FALLBACK").is_ok_and(|v| v == "1");
    if rc != 0 || rehearse {
        if rc != 0 {
            fails.push(format!("bench ({kset}) rc {rc}"));
        }
        eprintln!(
            "A1 STAGE fallback: the bench {}: once per candidate, then baselines only",
            if rc != 0 {
                "failed"
            } else {
                "passed (forced: AXON_A1_REHEARSE_FALLBACK=1)"
            }
        );
        for c in &healthy {
            let (rc, _) = run(&[], "cuda_a1_bench", true, &[("AXON_A1_KERNELS", *c)], 420);
            if rc != 0 {
                fails.push(format!("fallback bench {c} rc {rc}"));
            }
        }
        let (rc, _) = run(
            &[],
            "cuda_a1_bench",
            true,
            &[("AXON_A1_KERNELS", "none")],
            300,
        );
        if rc != 0 {
            fails.push(format!("fallback baselines rc {rc}"));
        }
    }
    eprintln!(
        "A1 SESSION {:.0} s of budget {budget} s",
        t0.elapsed().as_secs_f64()
    );
    if fails.is_empty() {
        eprintln!("A1 END ok");
    } else {
        eprintln!("A1 END FAILED: {}", fails.join(", "));
        panic!("a1 stages failed: {}", fails.join(", "));
    }
}
