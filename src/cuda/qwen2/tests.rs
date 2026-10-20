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
    // Only near-ties flip.
    assert!(
        ag.flips.iter().all(|&d| d < 0.5),
        "a top-1 flip with margin >= 0.5: {:?}",
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
    // "gemv@37": at most 37 blocks, looping over the rest (decode.cu
    // "Persistent row loop").
    let (kernel, cap) = match kernel.split_once('@') {
        Some((k, c)) => (k, c.parse::<usize>().unwrap()),
        None => (kernel, usize::MAX),
    };
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
            (rows.div_ceil(per).min(cap) as u32, 1, 1),
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
                Arg::Ptr(0),
                Arg::F32(0.0),
                Arg::I32(0),
                Arg::Ptr(0),
                Arg::Ptr(0),
                Arg::I32(0),
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
            // Blocks looping over several rows: the same sums.
            for k in ["gemv@37", "gemv1@46", "gemv@1"] {
                let got_p = gpu_gemv(&g, &km, k, qt, w, cols, &x);
                assert_eq!(bits(&got), bits(&got_p), "{name}: {k} vs gemv");
            }
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
                        // No q/k norms (QkNorm: kraw, qn, kn, eps, rope, kc_out).
                        Arg::Ptr(0),
                        Arg::Ptr(0),
                        Arg::Ptr(0),
                        Arg::F32(0.0),
                        Arg::Ptr(0),
                        Arg::Ptr(0),
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
            let out = g.alloc(dim * 4).unwrap();
            // SAFETY: decode.cu's signatures; buffers sized as `try_cache`
            // sizes them, K/V with np rows.
            unsafe {
                g.launch(
                    &fs,
                    (nkv as u32, gy.min(nblk_max) as u32, 1),
                    (32, 1, 1),
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
                        Arg::I32(nblk_max as i32),
                        Arg::I32(dim.is_multiple_of(256) as i32),
                        act.arg(),
                        out.arg(),
                    ],
                )
                .unwrap();
            }
            let got: Vec<f32> = out.read().unwrap();
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
                        Arg::I32(0),
                    ],
                )
                .unwrap();
            }
            let gq: Vec<f32> = q.read().unwrap();
            let gk: Vec<u16> = kc.read().unwrap();
            let gv: Vec<u16> = vc.read().unwrap();
            // A looping grid (decode.cu "Persistent row loop"): the same bits.
            {
                let gk_raw: Vec<f32> = kraw.read().unwrap();
                let q2 = g.alloc(qd * 4).unwrap();
                let kc2 = g.alloc((pos + 1) * kvd * 2).unwrap();
                let vc2 = g.alloc((pos + 1) * kvd * 2).unwrap();
                let kraw2 = g.alloc(kvd * 4).unwrap();
                for b in [&q2, &kc2, &vc2, &kraw2] {
                    b.zero().unwrap();
                }
                // SAFETY: as the launch above, into same-sized buffers.
                unsafe {
                    g.launch(
                        &f,
                        (items.min(37) as u32, 1, 1),
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
                            q2.arg(),
                            kc2.arg(),
                            vc2.arg(),
                            if norm { kraw2.arg() } else { Arg::Ptr(0) },
                            Arg::I32(0),
                        ],
                    )
                    .unwrap();
                }
                let row = pos * kvd..(pos + 1) * kvd;
                assert_eq!(bits(&gq), bits(&q2.read::<f32>().unwrap()), "q, 37 blocks");
                // (qwen3: keys go to kraw, the cache row is left as it was.)
                if !norm {
                    assert_eq!(
                        gk[row.clone()],
                        kc2.read::<u16>().unwrap()[row.clone()],
                        "k"
                    );
                }
                assert_eq!(gv[row.clone()], vc2.read::<u16>().unwrap()[row], "v");
                if norm {
                    assert_eq!(bits(&gk_raw), bits(&kraw2.read::<f32>().unwrap()), "kraw");
                }
            }
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
/// divergence, summed), asserting that every divergence is a near-tie in
/// r's own logits.
fn continuations<R: Lm, O: Lm>(
    r: &R,
    o: &O,
    tok: &crate::tokenizer::Tokenizer,
    name: &str,
) -> (usize, usize) {
    continuations_with(r, o, tok, name, 0.5)
}

/// `continuations`, divergences asserted to have a margin below `tie`.
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
                let l = &lr[i];
                let margin = l[tr[i] as usize] - l[to[i] as usize];
                eprintln!(
                    "{name}: {p:?} diverges at token {i}: {:?} vs {:?}, reference margin {margin:.3}",
                    tok.decode(&[tr[i]]),
                    tok.decode(&[to[i]])
                );
                assert!(margin < tie, "not a near-tie: margin {margin}");
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
    let net = CudaQwen2::new(&g, &km, &m, 4200).unwrap();
    let mut c = net.new_cache(4200);
    let poss: Vec<usize> = std::env::var("AXON_PROF_POS")
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![8usize, 1000, 4100]);
    for pos in poss {
        // Warm up (and fill the cache up to pos, cheaply: positions past
        // what was written read zeros, which is fine for timing).
        let _ = net.profile(&mut c, 100, pos).unwrap();
        // Per kernel, the best of 10 steps: the GPU is shared with a
        // desktop, whose work lands in random kernels' windows.
        let runs: Vec<_> = (0..10)
            .map(|_| net.profile(&mut c, 100, pos).unwrap())
            .collect();
        let mut k = runs[0].0.clone();
        let mut kmed = runs[0].0.clone();
        for (i, row) in k.iter_mut().enumerate() {
            row.ms = runs.iter().map(|r| r.0[i].ms).fold(f64::INFINITY, f64::min);
            let mut v: Vec<f64> = runs.iter().map(|r| r.0[i].ms).collect();
            v.sort_by(f64::total_cmp);
            kmed[i].ms = v[v.len() / 2];
        }
        let total: f64 = k.iter().map(|r| r.ms).sum();
        let total_med: f64 = kmed.iter().map(|r| r.ms).sum();
        let serial: Vec<f64> = runs.iter().map(|r| r.1).collect();
        eprintln!(
            "pos {pos}: serialized step (events between kernels) per run: {:?}; sum of per-kernel medians {total_med:.3} ms",
            serial.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>()
        );
        for (a, b) in k.iter().zip(kmed.iter()) {
            eprintln!(
                "  median {:<10} {:>4} x {:>8.1} us = {:>7.3} ms",
                a.name,
                a.launches,
                b.ms * 1e3 / a.launches as f64,
                b.ms
            );
        }
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
        eprintln!(
            "pos {pos}: graph step p10 {:.3} p25 {:.3} p50 {:.3} p75 {:.3} p90 {:.3} max {:.3} ms",
            gt[2], gt[5], gt[10], gt[15], gt[18], gt[19]
        );
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
        ("3B q/wo Q4_K", Q4_K, 2048, 2048),
        ("3B gate Q4_K", Q4_K, 2048, 11008),
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
        let sms = g.info().unwrap().sms as usize;
        // One block per item, then a resident grid looping (decode.cu
        // "Persistent row loop").
        for (kname, per, cap) in [
            ("gemv", 2, usize::MAX),
            ("gemv1", 1, usize::MAX),
            ("gemv", 2, sms * 8),
            ("gemv1", 1, sms * 8),
        ] {
            let fg = km.function(kname).unwrap();
            let grid = rows.div_ceil(per).min(cap);
            let mut best = f64::INFINITY;
            for i in 0..20.max(2 * copies) {
                let wb = &wbs[i % copies];
                let e0 = g.record().unwrap();
                // SAFETY: as above.
                unsafe {
                    g.launch(
                        &fg,
                        (grid as u32, 1, 1),
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
                            Arg::Ptr(0),
                            Arg::F32(0.0),
                            Arg::I32(0),
                            Arg::Ptr(0),
                            Arg::Ptr(0),
                            Arg::I32(0),
                        ],
                    )
                    .unwrap();
                }
                let e1 = g.record().unwrap();
                best = best.min(e0.ms_to(&e1).unwrap() as f64);
            }
            eprintln!(
                "{what:<16} {kname:<6} {grid:>6} blk {:>7.1} MB {:>8.1} us {:>6.0} GB/s",
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

/// The GEMVs' fused RMSNorm epilogue (decode.cu rmsnorm_quant_1024 run by
/// the last block): the same residual stream and the same quantized bytes
/// as the GEMV followed by the 1024-thread rmsnorm_quant kernel, for
/// vectors shorter and longer than 1024, with and without the Q8_K view,
/// on a looping grid and one block per item, the ticket left at zero.
#[test]
fn gpu_gemv_norm_epilogue() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let (fq, fnorm) = (
        km.function("quantize").unwrap(),
        km.function("rmsnorm_quant").unwrap(),
    );
    let mut rng = Rng(4242);
    for n in [896usize, 1024, 2560, 3584, 5120] {
        // Square Q8_0 matrix, sane scales (f16 1/256), random int8 values.
        let rb = quant::row_bytes(Q8_0, n);
        let mut w = vec![0u8; rb * n];
        for blk in w.chunks_mut(34) {
            blk[0] = 0x00;
            blk[1] = 0x1c;
            for b in &mut blk[2..] {
                *b = rng.next() as u8;
            }
        }
        let wb = g.upload(&w).unwrap();
        let xb = g.upload(&rng.vec(n, 1.0)).unwrap();
        let init = rng.vec(n, 3.0);
        let gam = g.upload(&rng.vec(n, 1.5)).unwrap();
        let act = g.alloc(act_bytes(n)).unwrap();
        let wk = Arg::I32(n.is_multiple_of(256) as i32);
        let eps = 1e-6f32;
        // SAFETY: decode.cu's signatures; buffers for n values.
        unsafe {
            g.launch(
                &fq,
                (n.div_ceil(256) as u32, 1, 1),
                (32, 1, 1),
                0,
                &[xb.arg(), Arg::I32(n as i32), wk, act.arg()],
            )
            .unwrap();
        }
        let tick = g.alloc(4).unwrap();
        tick.zero().unwrap();
        let mut want: Option<(Vec<u32>, Vec<u8>)> = None;
        for (kernel, per, fused, cap) in [
            ("gemv", 2, false, usize::MAX),
            ("gemv", 2, true, 37),
            ("gemv", 2, true, usize::MAX),
            ("gemv1", 1, true, 46),
            ("gemv1", 1, true, 1),
        ] {
            let fg = km.function(kernel).unwrap();
            let out = g.upload(&init).unwrap();
            let nact = g.alloc(act_bytes(n)).unwrap();
            nact.zero().unwrap();
            let p = |b: &DevBuf, on: bool| if on { b.arg() } else { Arg::Ptr(0) };
            // SAFETY: as above; nact for n values, tick one u32.
            unsafe {
                g.launch(
                    &fg,
                    (n.div_ceil(per).min(cap) as u32, 1, 1),
                    (GEMV_THREADS, 1, 1),
                    0,
                    &[
                        Arg::I32(Q8_0 as i32),
                        wb.arg(),
                        Arg::U64(rb as u64),
                        Arg::I32(n as i32),
                        Arg::I32(n as i32),
                        act.arg(),
                        Arg::Ptr(0),
                        Arg::I32(1),
                        out.arg(),
                        p(&gam, fused),
                        Arg::F32(eps),
                        wk,
                        p(&nact, fused),
                        p(&tick, fused),
                        Arg::I32(0),
                    ],
                )
                .unwrap();
                if !fused {
                    g.launch(
                        &fnorm,
                        (1, 1, 1),
                        (1024, 1, 1),
                        0,
                        &[
                            out.arg(),
                            gam.arg(),
                            Arg::F32(eps),
                            Arg::I32(n as i32),
                            wk,
                            nact.arg(),
                            Arg::Ptr(0),
                        ],
                    )
                    .unwrap();
                }
            }
            let got = (
                bits(&out.read::<f32>().unwrap()),
                nact.read::<u8>().unwrap(),
            );
            match &want {
                None => want = Some(got),
                Some(w) => {
                    assert_eq!(w.0, got.0, "n {n} {kernel} grid cap {cap}: residual");
                    assert!(
                        w.1 == got.1,
                        "n {n} {kernel} grid cap {cap}: quantized norm"
                    );
                }
            }
            assert_eq!(tick.read::<u32>().unwrap(), vec![0], "ticket reset");
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
    let all = q4.dims[1] as usize;
    let mut rng = Rng(808);
    let x = rng.vec(n, 1.0);
    let xb = g.upload(&x).unwrap();
    let act = g.alloc(act_bytes(n)).unwrap();
    let xq = quant::QuantizedAct::for_types(&x, &[Q4_K]);
    // 300 rows: the first rows only (a 44-row last group for the fused
    // quantize).
    for (gt, ut, rows) in [(q4, q6, all), (q6, q4, all), (q4, q4, all), (q4, q6, 300)] {
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
                    Arg::I32(0),
                    Arg::Ptr(0),
                    Arg::Ptr(0),
                    Arg::I32(0),
                ],
            )
            .unwrap();
        }
        let got: Vec<f32> = out.read().unwrap();
        // The fused quantize (tickets per 256 rows; 896 and 300 rows: a
        // partial last group, no Q8_K view), a looping grid: the same outputs, and
        // the quantize kernel's bytes of them; and the loop without it.
        {
            let qa = g.alloc(act_bytes(rows)).unwrap();
            let qb = g.alloc(act_bytes(rows)).unwrap();
            qa.zero().unwrap();
            qb.zero().unwrap();
            let tick = g.alloc(4 * rows.div_ceil(256)).unwrap();
            tick.zero().unwrap();
            let out2 = g.alloc(rows * 4).unwrap();
            let wk = Arg::I32(rows.is_multiple_of(256) as i32);
            for (grid, fused) in [(37u32, true), (rows as u32, true), (37, false)] {
                // SAFETY: as above; qa for rows inputs, a ticket per 256 rows.
                unsafe {
                    g.launch(
                        &fgu,
                        (grid, 1, 1),
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
                            out2.arg(),
                            wk,
                            qa.arg(),
                            if fused { tick.arg() } else { Arg::Ptr(0) },
                            Arg::I32(0),
                        ],
                    )
                    .unwrap();
                    g.launch(
                        &fq,
                        (rows.div_ceil(256) as u32, 1, 1),
                        (32, 1, 1),
                        0,
                        &[out2.arg(), Arg::I32(rows as i32), wk, qb.arg()],
                    )
                    .unwrap();
                }
                let got2: Vec<f32> = out2.read().unwrap();
                assert_eq!(bits(&got), bits(&got2), "gate_up grid {grid} fused {fused}");
                if !fused {
                    continue;
                }
                let (a, b): (Vec<u8>, Vec<u8>) = (qa.read().unwrap(), qb.read().unwrap());
                assert!(a == b, "fused quantize grid {grid}");
                let t: Vec<u32> = tick.read().unwrap();
                assert!(t.iter().all(|&v| v == 0), "tickets reset");
            }
        }
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
    unsafe { launch_product(g, &f, kernel, qt, rows, nb, &common) };
    g.sync().unwrap();
    out.read().unwrap()
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
        Q6_K => (210, 256),
        Q8_0 => (34, 32),
        _ => (22, 32),
    };
    for b in 0..rows * cols / vb {
        let o = b * bb + if qt == Q6_K { 208 } else { 0 };
        let mut scale = || (0x1000u16 + (rng.next() & 0x7ff) as u16) | (rng.next() as u16 & 0x8000);
        let d = scale();
        w[o..o + 2].copy_from_slice(&d.to_le_bytes());
        if qt == Q4_K {
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

/// Every product kernel against gemv_ref, the canonical order written out
/// one thread per output: the decode GEMV (per token), gemvn, gemvm and
/// the GEMMs, bit for bit, every weight type, random widths up to the 7B's
/// 18944, odd row counts, batches around the tiles' sizes, all epilogues.
#[test]
fn gpu_products_match_reference_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let mut rng = Rng(0x0dd5_eed5_2026_1014);
    let mut cases: Vec<(u32, usize, usize)> = Vec::new();
    for qt in [Q4_K, Q6_K, Q8_0, Q5_0] {
        let vb = if qt == Q4_K || qt == Q6_K { 256 } else { 32 };
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
                if nb <= 8 {
                    kernels.push(format!("gemvm8_{tn}"));
                }
                if nb <= 16 {
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
        }
        eprintln!("{tn} {cols} x {rows}: every kernel == gemv_ref");
    }
}

/// The tensor-core GEMMs of a type (prefill.cu gemm2_*).
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
    let km = Kernels::module(&g).unwrap();
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
    ] {
        if only.as_deref().is_some_and(|o| !what.contains(o)) {
            continue;
        }
        let rb = quant::row_bytes(qt, cols);
        let w = synth_matrix(&mut rng, qt, rows, cols);
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
                Arg::I32(0),
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
    let km = Kernels::module(&g).unwrap();
    let ids = long_ids(m, 1100);
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
    let paths: &[bool] = if km.mma() { &[true, false] } else { &[false] };
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
    }
    eprintln!(
        "{name} {kv:?}: prefill bit-identical over 7 batchings of {} tokens, {} attention \
         paths x 2 split counts",
        ids.len(),
        paths.len()
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
    let km = Kernels::module(&g).unwrap();
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
        .set_max_shared(batch::fa_shape(hd, false, 0).2)
        .unwrap();
    fs[6]
        .set_max_shared(batch::fa_shape(hd, true, 0).2)
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
                let (fa_rows, fa_thr, fa_smem) = batch::fa_shape(hd, k == 6, 0);
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
                        ],
                    ),
                    4 => g.launch(
                        f,
                        (nkv as u32, gy, 1),
                        (32, 1, 1),
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
                        ],
                    ),
                    _ => g.launch(
                        f,
                        (dim.div_ceil(256) as u32, 1, 1),
                        (256, 1, 1),
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
                            Arg::Ptr(0),
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
        let ffs: Vec<Func> = ["attn_fa_b", "attn_fa_bs"]
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let f = km.function(&format!("{n}_{q8n}{hd}")).unwrap();
                f.set_max_shared(batch::fa_shape(hd, i == 1, kg).2).unwrap();
                f
            })
            .collect();
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
            // Flash (both block sizes, alternately): all nb queries, y in
            // place of q, then act.
            let small = (start + nb) % 2 == 1;
            let ctxs = |what: &str, z: usize| {
                format!(
                    "{nh}/{nkv}x{hd} kg {kg} start {start} nb {nb} z0 {z0} small {small}: {what} query {z} (np {})",
                    start + z + 1
                )
            };
            let (fa_rows, fa_thr, fa_smem) = batch::fa_shape(hd, small, kg);
            let qy = g.upload(&q).unwrap();
            let fact = g.alloc(nb * ast).unwrap();
            fact.zero().unwrap();
            // SAFETY: prefill.cu's signatures; q/y nb rows of dim, the cache
            // max_ctx rows, act nb strides.
            unsafe {
                g.launch(
                    &ffs[small as usize],
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
            let (fy, fact): (Vec<f32>, Vec<u8>) = (qy.read().unwrap(), fact.read().unwrap());
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
                        (32, 1, 1),
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
                        ]
                        .concat(),
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
                            Arg::I32(nblk_max as i32),
                            want_k,
                            act.arg(),
                            y.arg(),
                        ],
                    )
                    .unwrap();
                }
                let (wy, wact): (Vec<f32>, Vec<u8>) = (y.read().unwrap(), act.read().unwrap());
                assert!(
                    wy.iter().all(|x| x.is_finite()),
                    "{}",
                    ctxs("decode y finite", z)
                );
                for (d, (a, b)) in fy[z * dim..(z + 1) * dim].iter().zip(&wy).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits() || (*a == 0.0 && *b == 0.0),
                        "{}: y[{d}] {a} vs {b}",
                        ctxs("flash", z)
                    );
                }
                assert!(
                    fact[z * ast..z * ast + ab] == wact[..],
                    "{}",
                    ctxs("flash act", z)
                );
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

/// ~n tokens of natural text, prose and code (README.md, then kv.rs).
fn natural_ids(m: &Model, n: usize) -> Vec<u32> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let text = ["README.md", "src/kv.rs", "src/chat.rs"]
        .iter()
        .map(|f| std::fs::read_to_string(dir.join(f)).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    let mut ids = crate::tokenizer::Tokenizer::from_gguf(m).encode(&text);
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
                        true => (max_split.min(nblk), 32, nblk),
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
                                Arg::I32(parts as i32),
                                Arg::I32(dim.is_multiple_of(256) as i32),
                                act.arg(),
                                out.arg(),
                            ],
                        )
                        .unwrap();
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
