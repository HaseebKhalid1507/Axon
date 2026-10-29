//! Fast mode against exact mode, graded: the scaffolding of
//! docs/design/determinism-modes.md §4.2 (ADR step M3), built before any
//! fast kernel exists so the first one's PR has a gate waiting for it.
//!
//! The gate is `assert_graded`, the GPU-vs-CPU gate since G3, unchanged,
//! pointed at a new reference: the net under test in its numerics mode
//! against the same device in exact mode, graded relative to an
//! *exact-vs-exact floor*, two placements the project already accepts as
//! "the same model" (`Floor`):
//! - a second GPU of another architecture in this process (sm_89 vs sm_120:
//!   their tensor-core attention sums f32 in other orders), if the host
//!   has one;
//! - else this GPU's f16 KV cache against its q8 one, both exact;
//! - on the CPU, the r8 repacked kernels against the per-row ones (the
//!   floor the GPU-vs-CPU gate grades against).
//!
//! **Arms** (`Arm`). Each is graded on its own, so a fast kernel can't
//! hide in the positions of an arm it never runs in:
//! - `Decode`: `PROMPTS` with `STEPS` greedy steps of the reference each
//!   (its picks fed to every net), then `DECODE_WINDOWS` windows of natural
//!   text teacher-forced from position 0, a token per step: the one-token
//!   kernels (the gemv, the decode attention).
//! - `Batched`: per window of natural text, each prefix length of
//!   `prefixes` prefilled by one `prefill_logits` call: 24 tokens (a gemvm
//!   pass, past the 16-token gemvn cut), 160 (the tensor-core GEMM, at
//!   least 128 rows) and on the GPU 520 (at least 512: the long passes'
//!   tiles). Graded: the pass's last row (its output head), then `AFTER`
//!   positions decoded on that cache (reading the keys and values the
//!   batched pass wrote); and `verify_argmax` of the same prefix, every
//!   row's pick against the reference's (`Graded::rows`), since a kernel
//!   wrong on one tile or one range of rows flips picks there and nowhere
//!   else.
//! - `Cached`: `CACHED.0` tokens prefilled, then `CACHED.1` more in a
//!   second batched pass on the same cache (a prefix from the cache and the
//!   rest of the prompt: a chat follow-up, serve's prefix cache, chunked
//!   prefill); its last row and `AFTER` decoded positions graded.
//! - `Load` (the GPU): `LOAD` sequences in one pool (`BatchLm`, serve's
//!   engine), each fed its own prefix, then `LOAD_STEPS` steps of all of
//!   them together (steps of `LOAD` rows: past the gemvn and gemvm cuts),
//!   every row's logits graded against the reference's alone.
//!
//! In exact mode every arm is the decode arm's bits (the contract,
//! docs/determinism.md); in fast mode each arm is graded against exact on
//! its own, which is all fast mode promises: tolerance on every path, bits
//! on none.
//!
//! **A fast kernel has to be seen running.** The net under test counts its
//! fast-set kernel launches (`Backend::fast_launches`); each arm records
//! its own. With fast kernels (`Backend::fast_kernels`) the count must
//! move, so a kernel routed where no arm reaches can't pass by never
//! running; without them it must stay 0, or the bit check below is wrong.
//!
//! ΔNLL (mode - exact, nats per position) is taken at every teacher-forced
//! position, with an SE clustered by window of text (methodology.md
//! "Numerics-changing work"). With a dozen windows it is a smoke check,
//! not a measurement: a PR's table uses `scripts/numerics_dnll.py` (any
//! model, any corpus, `--prefix` for the batched passes) with at least 30
//! windows.
//!
//! Today it runs green in exact vs exact (the sanity check: a mode graded
//! against itself has KL 0, ΔNLL 0, every top-1 the same, in every arm)
//! and fast vs exact (no fast kernel yet: fast runs exact's kernels, so the
//! same), and another quantization of the same weights fails it in every
//! arm (`graded_catches_another_model`): the gate has teeth.
//!
//! What the floor doesn't settle (the ADR's §7, an open item for WP-A1):
//! it is a reordering floor, two accepted float orders of one model. The
//! first fast kernels planned change precision, not only order (FP8
//! activations, W4A16), and their reference bound is still to be decided.

use super::{
    Agree, PROMPTS, Top1, assert_graded_with, compare_trace, log_softmax, natural_ids, report,
    trace,
};
use crate::cuda::Gpu;
use crate::gguf::Model;
use crate::kv::KvType;
use crate::lm::{Backend, BackendOptions, BatchLm, Device, Lm, RowHead, RowOut, SeqRow};
use crate::numerics::Numerics;
use crate::sample::CandSpec;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

/// Greedy steps after each prompt.
const STEPS: usize = 16;
/// The decode arm's windows of natural text, of `WINDOW` tokens each.
const DECODE_WINDOWS: usize = 4;
const WINDOW: usize = 48;
/// The batched and cached arms' windows (other text than the decode arm's).
const BATCH_WINDOWS: usize = 4;
/// Positions decoded after each batched pass.
const AFTER: usize = 16;
/// The cached arm: a prefix of `.0` tokens in the cache, then `.1` more.
const CACHED: (usize, usize) = (64, 96);
/// The load arm: sequences stepped together, and steps.
const LOAD: usize = 40;
const LOAD_STEPS: usize = 4;
/// The load arm's longest prefix (`load_prefix`).
const LOAD_PREFIX_MAX: usize = 63;
/// The nets' context.
pub(crate) const CTX: usize = 640;

/// How many positions each arm grades (`grade_sized`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sizes {
    /// The decode arm's windows of `WINDOW` tokens.
    pub(crate) decode_windows: usize,
    /// The batched arm's windows; `verify_argmax` on the first
    /// `verify_windows` of them.
    pub(crate) batch_windows: usize,
    pub(crate) verify_windows: usize,
    /// The cached arm's windows: None, the batched arm's (the same text);
    /// Some(k), k of its own.
    pub(crate) cached_windows: Option<usize>,
    /// Positions decoded after each batched pass.
    pub(crate) after: usize,
    /// The load arm's sequences, and steps.
    pub(crate) load: usize,
    pub(crate) load_steps: usize,
}

impl Sizes {
    /// The constants above: every gate's before the FP8 one.
    pub(crate) const BASE: Sizes = Sizes {
        decode_windows: DECODE_WINDOWS,
        batch_windows: BATCH_WINDOWS,
        verify_windows: BATCH_WINDOWS,
        cached_windows: None,
        after: AFTER,
        load: LOAD,
        load_steps: LOAD_STEPS,
    };
    /// At least 1000 positions in every arm on the GPU (the FP8 gate,
    /// docs/design/fp8-weights.md §6 B): decode 6 x 17 + 19 x 48 = 1014,
    /// batched 7 x 3 x 49 = 1029, cached 21 x 49 = 1029, load 40 x 25.
    pub(crate) const FP8: Sizes = Sizes {
        decode_windows: 19,
        batch_windows: 7,
        verify_windows: BATCH_WINDOWS,
        cached_windows: Some(21),
        after: 48,
        load: LOAD,
        load_steps: 25,
    };
}

/// The batched arm's prefix lengths on `device` (module doc). The CPU
/// prefills in 128-token batches (`PREFILL_BATCH`): 520 is no new shape
/// there, only time.
pub(crate) fn prefixes(device: Device) -> &'static [usize] {
    match device {
        Device::Cpu => &[24, 160],
        _ => &[24, 160, 520],
    }
}

/// The load arm's sequence `s`'s prefix: 8..=63 tokens, every size class
/// of a pass.
fn load_prefix(s: usize) -> usize {
    8 + (s * 13) % (LOAD_PREFIX_MAX - 7)
}

/// What one arm runs (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arm {
    Decode,
    Batched,
    Cached,
    Load,
}

impl Arm {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Arm::Decode => "decode",
            Arm::Batched => "batched prefill",
            Arm::Cached => "cached prefix + batched rest",
            Arm::Load => "under load (pool steps)",
        }
    }
}

/// One arm's grading: the net under test and the floor, each against the
/// exact reference over the same positions, index for index
/// (`assert_graded` looks the floor's flips up by index).
pub(crate) struct ArmGrade {
    pub(crate) arm: Arm,
    pub(crate) mode: Agree,
    pub(crate) floor: Agree,
    /// Per teacher-forced position, (window, -ln p(next) - the
    /// reference's): the net under test's and the floor's.
    dm: Vec<(usize, f64)>,
    df: Vec<(usize, f64)>,
    /// Fast-set kernel launches on the net under test during the arm.
    pub(crate) launches: u64,
}

impl ArmGrade {
    fn new(arm: Arm) -> ArmGrade {
        ArmGrade {
            arm,
            mode: Agree::default(),
            floor: Agree::default(),
            dm: Vec::new(),
            df: Vec::new(),
            launches: 0,
        }
    }

    /// A new sequence (`Agree`'s previous-position KL starts again).
    fn start(&mut self) {
        (self.mode.last_kl, self.floor.last_kl) = (0.0, 0.0);
    }

    /// One position: the reference's logits, the floor's and the net's;
    /// `next`: (window, the teacher-forced token after it).
    fn tally(&mut self, lr: &[f32], lf: &[f32], lo: &[f32], next: Option<(usize, u32)>) {
        self.floor.tally(lr, lf);
        self.mode.tally(lr, lo);
        if let Some((w, t)) = next {
            let n = nll(lr, t);
            self.df.push((w, nll(lf, t) - n));
            self.dm.push((w, nll(lo, t) - n));
        }
    }

    /// ΔNLL (mean, clustered SE): the net under test's and the floor's.
    pub(crate) fn dnll(&self) -> ((f64, f64), (f64, f64)) {
        (clustered_mean(&self.dm), clustered_mean(&self.df))
    }
}

/// A grading (`grade`): every arm, and the batched passes' rows.
pub(crate) struct Graded {
    pub(crate) arms: Vec<ArmGrade>,
    /// `verify_argmax` over the batched arm's prefixes, row by row: (rows,
    /// the net under test's picks equal to the reference's, the floor's).
    pub(crate) rows: (usize, usize, usize),
    /// Which floor (`Floor::name`).
    pub(crate) floor_name: &'static str,
    /// The net under test runs a fast-set kernel (`Backend::fast_kernels`).
    pub(crate) fast_kernels: bool,
}

impl Graded {
    /// ΔNLL over every arm's teacher-forced positions: (the net under
    /// test's, the floor's), each (mean, SE clustered by window).
    pub(crate) fn dnll(&self) -> ((f64, f64), (f64, f64)) {
        let all = |f: fn(&ArmGrade) -> &Vec<(usize, f64)>| {
            let v: Vec<_> = self
                .arms
                .iter()
                .flat_map(|a| f(a).iter().copied())
                .collect();
            clustered_mean(&v)
        };
        (all(|a| &a.dm), all(|a| &a.df))
    }

    /// Fast-set kernel launches on the net under test, every arm.
    pub(crate) fn launches(&self) -> u64 {
        self.arms.iter().map(|a| a.launches).sum()
    }
}

/// The exact-vs-exact floor (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Floor {
    /// Exact on GPU 1, of another architecture than GPU 0's.
    CrossArch,
    /// Exact with the q8 KV cache, on the reference's GPU.
    KvQ8,
    /// Exact on the CPU's per-row kernels (AXON_REPACK=0's).
    CpuRows,
    /// FP8 (tests/fp8.rs): the CPU's fp8-ref with the fast path's e4m3
    /// activations, against fp8-ref (docs/design/fp8-weights.md §6 B).
    Fp8Emu,
}

impl Floor {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Floor::CrossArch => "exact on another GPU architecture",
            Floor::KvQ8 => "exact with the q8 KV cache",
            Floor::CpuRows => "exact on the CPU's per-row kernels",
            Floor::Fp8Emu => "the CPU's fp8-ref with e4m3 per-token activations",
        }
    }
}

/// Mean of `d`'s values and its standard error clustered by `d`'s keys
/// (Liang-Zeger with the CR1 factor C/(C-1)); SE NaN for one cluster.
pub(crate) fn clustered_mean(d: &[(usize, f64)]) -> (f64, f64) {
    if d.is_empty() {
        return (0.0, 0.0);
    }
    let n = d.len() as f64;
    let mean = d.iter().map(|x| x.1).sum::<f64>() / n;
    let mut sums = std::collections::BTreeMap::<usize, f64>::new();
    for &(c, x) in d {
        *sums.entry(c).or_default() += x - mean;
    }
    let k = sums.len() as f64;
    let v = sums.values().map(|s| s * s).sum::<f64>() / (n * n) * k / (k - 1.0);
    (mean, v.sqrt())
}

/// `-ln p(next)` under logits `l`.
fn nll(l: &[f32], next: u32) -> f64 {
    -log_softmax(l)[next as usize]
}

/// `after` positions from `p` on, teacher-forced, a token per step, on
/// caches that hold positions 0..p: graded into `a` (window `w`).
#[allow(clippy::too_many_arguments)]
fn decode_after<R: Lm, F: Lm, O: Lm>(
    a: &mut ArmGrade,
    (r, rc): (&R, &mut R::Cache),
    (f, fc): (&F, &mut F::Cache),
    (o, oc): (&O, &mut O::Cache),
    seq: &[u32],
    p: usize,
    w: usize,
    after: usize,
) {
    for pos in p..p + after {
        let t = seq[pos];
        let lr = r.forward(t, pos, rc);
        let lf = f.forward(t, pos, fc);
        let lo = o.forward(t, pos, oc);
        a.tally(&lr, &lf, &lo, Some((w, seq[pos + 1])));
    }
}

/// Grade `o` against the exact reference `r`, with the floor `f` (exact
/// too: another accepted placement), on `m`'s tokenizer, every arm that
/// runs on `device` (module doc).
pub(crate) fn grade<F: Lm>(
    r: &Backend,
    f: &F,
    o: &Backend,
    m: &Model,
    floor: Floor,
    device: Device,
) -> Graded {
    grade_sized(r, f, o, m, floor, device, &Sizes::BASE)
}

/// `grade` with the arms' sizes `s`.
pub(crate) fn grade_sized<F: Lm>(
    r: &Backend,
    f: &F,
    o: &Backend,
    m: &Model,
    floor: Floor,
    device: Device,
    sz: &Sizes,
) -> Graded {
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let pre = prefixes(device);
    let (c0, c1) = CACHED;
    let longest = *pre.iter().max().unwrap();
    let (span, cspan) = match sz.cached_windows {
        None => ((longest.max(c0 + c1)) + sz.after + 1, 0),
        Some(_) => (longest + sz.after + 1, c0 + c1 + sz.after + 1),
    };
    let cwins = sz.cached_windows.unwrap_or(0);
    let load_len = LOAD_PREFIX_MAX + sz.load_steps + 1;
    let ids = natural_ids(
        m,
        sz.decode_windows * WINDOW + sz.batch_windows * span + cwins * cspan + sz.load * load_len,
    );
    let (dec, rest) = ids.split_at(sz.decode_windows * WINDOW);
    let (bat, rest) = rest.split_at(sz.batch_windows * span);
    let (cac, load) = rest.split_at(cwins * cspan);
    let caches = || (r.new_cache(CTX), f.new_cache(CTX), o.new_cache(CTX));
    let mut arms = Vec::new();

    // Decode: the prompts' greedy traces, then text from position 0.
    let mut a = ArmGrade::new(Arm::Decode);
    let l0 = o.fast_launches();
    for p in PROMPTS {
        let t = trace(r, &tok.encode(p), STEPS);
        compare_trace(&t, f, &mut a.floor);
        compare_trace(&t, o, &mut a.mode);
    }
    for (w, seq) in dec.chunks(WINDOW).enumerate() {
        let (mut rc, mut fc, mut oc) = caches();
        a.start();
        for (pos, &t) in seq.iter().enumerate() {
            let lr = r.forward(t, pos, &mut rc);
            let lf = f.forward(t, pos, &mut fc);
            let lo = o.forward(t, pos, &mut oc);
            a.tally(&lr, &lf, &lo, seq.get(pos + 1).map(|&n| (w, n)));
        }
    }
    a.launches = o.fast_launches() - l0;
    arms.push(a);

    // Batched: each prefix in one prefill_logits call, the decode after
    // it; verify_argmax's rows over the same prefix.
    let mut a = ArmGrade::new(Arm::Batched);
    let mut rows = (0, 0, 0);
    let l0 = o.fast_launches();
    for (i, seq) in bat.chunks(span).enumerate() {
        let w = sz.decode_windows + i;
        for &p in pre {
            let (mut rc, mut fc, mut oc) = caches();
            a.start();
            let lr = r.prefill_logits(&seq[..p], 0, &mut rc);
            let lf = f.prefill_logits(&seq[..p], 0, &mut fc);
            let lo = o.prefill_logits(&seq[..p], 0, &mut oc);
            a.tally(&lr, &lf, &lo, Some((w, seq[p])));
            let (rr, ff, oo) = ((r, &mut rc), (f, &mut fc), (o, &mut oc));
            decode_after(&mut a, rr, ff, oo, seq, p, w, sz.after);
            if i >= sz.verify_windows {
                continue;
            }
            let (mut rc, mut fc, mut oc) = caches();
            let pr = r.verify_argmax(&seq[..p], 0, &mut rc);
            let pf = f.verify_argmax(&seq[..p], 0, &mut fc);
            let po = o.verify_argmax(&seq[..p], 0, &mut oc);
            let same = |x: &[u32]| pr.iter().zip(x).filter(|(a, b)| a == b).count();
            rows = (rows.0 + p, rows.1 + same(&po), rows.2 + same(&pf));
        }
    }
    a.launches = o.fast_launches() - l0;
    arms.push(a);

    // Cached: a prefix in the cache, the rest in a second batched pass.
    let mut a = ArmGrade::new(Arm::Cached);
    let l0 = o.fast_launches();
    let (cwin, w0) = match sz.cached_windows {
        None => (bat.chunks(span), sz.decode_windows),
        Some(_) => (cac.chunks(cspan), sz.decode_windows + sz.batch_windows),
    };
    for (w, seq) in cwin.enumerate() {
        let w = w0 + w;
        let (mut rc, mut fc, mut oc) = caches();
        a.start();
        r.feed_batch(&seq[..c0], 0, &mut rc);
        f.feed_batch(&seq[..c0], 0, &mut fc);
        o.feed_batch(&seq[..c0], 0, &mut oc);
        let lr = r.prefill_logits(&seq[c0..c0 + c1], c0, &mut rc);
        let lf = f.prefill_logits(&seq[c0..c0 + c1], c0, &mut fc);
        let lo = o.prefill_logits(&seq[c0..c0 + c1], c0, &mut oc);
        a.tally(&lr, &lf, &lo, Some((w, seq[c0 + c1])));
        decode_after(
            &mut a,
            (r, &mut rc),
            (f, &mut fc),
            (o, &mut oc),
            seq,
            c0 + c1,
            w,
            sz.after,
        );
    }
    a.launches = o.fast_launches() - l0;
    arms.push(a);

    // Load (the GPU: a pool is the GPU model's): every row of LOAD
    // sequences' steps, taken together, against each sequence alone.
    if device != Device::Cpu {
        let mut a = ArmGrade::new(Arm::Load);
        let l0 = o.fast_launches();
        let seqs: Vec<&[u32]> = load.chunks(load_len).collect();
        let lw = sz.decode_windows + sz.batch_windows + cwins;
        let mut pool = o.new_pool(sz.load, CTX).unwrap();
        let mut alone = Vec::with_capacity(sz.load);
        for (s, seq) in seqs.iter().enumerate() {
            let p = load_prefix(s);
            let (mut rc, mut fc) = (r.new_cache(CTX), f.new_cache(CTX));
            r.feed_batch(&seq[..p], 0, &mut rc);
            f.feed_batch(&seq[..p], 0, &mut fc);
            let at: Vec<_> = (p..p + sz.load_steps)
                .map(|pos| {
                    let lr = r.forward(seq[pos], pos, &mut rc);
                    (lr, f.forward(seq[pos], pos, &mut fc))
                })
                .collect();
            alone.push(at.into_iter());
            o.feed_slot(&mut pool, s, &seq[..p], 0).unwrap();
        }
        // Every logit back (k past CAND_MAX: the host selects from the
        // row's logits, no penalty).
        let spec = CandSpec {
            k: o.vocab(),
            penalty: 1.0,
            last_n: 0,
        };
        for k in 0..sz.load_steps {
            let step: Vec<SeqRow> = (seqs.iter().enumerate())
                .map(|(s, seq)| {
                    let pos = load_prefix(s) + k;
                    SeqRow {
                        token: seq[pos],
                        pos,
                        slot: s,
                        head: RowHead::Sample {
                            spec,
                            recent: &seq[..=pos],
                        },
                    }
                })
                .collect();
            let out = o.step(&mut pool, &step).unwrap();
            for (s, row) in out.into_iter().enumerate() {
                let RowOut::Cands(cs) = row else {
                    panic!("a sampled row's candidates")
                };
                assert_eq!(cs.len(), o.vocab(), "every logit back");
                let mut lo = vec![f32::NAN; o.vocab()];
                for c in cs {
                    lo[c.id as usize] = c.logit;
                }
                let (lr, lf) = alone[s].next().unwrap();
                let next = seqs[s][step[s].pos + 1];
                a.start();
                a.tally(&lr, &lf, &lo, Some((lw + s, next)));
            }
        }
        a.launches = o.fast_launches() - l0;
        arms.push(a);
    }
    Graded {
        arms,
        rows,
        floor_name: floor.name(),
        fast_kernels: o.fast_kernels(),
    }
}

/// A panic's message.
fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// `assert_graded` on one arm, its failure naming the arm.
fn assert_arm(what: &str, a: &ArmGrade, top1: Top1) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| {
        assert_graded_with(&a.mode, &a.floor, top1)
    }))
    .map_err(|p| format!("{what} [{}]: {}", a.arm.name(), panic_text(&*p)))
}

/// Print `g`, every arm.
pub(crate) fn report_graded(what: &str, g: &Graded) {
    for a in &g.arms {
        let n = a.arm.name();
        report(&format!("{what} [{n}] vs exact"), &a.mode);
        report(
            &format!("{what} [{n}]: floor, {} vs exact", g.floor_name),
            &a.floor,
        );
        let ((m, sm), (f, sf)) = a.dnll();
        eprintln!(
            "{what} [{n}]: ΔNLL vs exact {m:+.2e} ± {sm:.1e} (floor {f:+.2e} ± {sf:.1e}), {} \
             teacher-forced positions; fast-set launches {}",
            a.dm.len(),
            a.launches
        );
    }
    let ((m, sm), (f, sf)) = g.dnll();
    let (n, same, fsame) = g.rows;
    eprintln!(
        "{what}: ΔNLL vs exact, every arm {m:+.2e} ± {sm:.1e} (floor {f:+.2e} ± {sf:.1e}; a smoke \
         check, scripts/numerics_dnll.py for a table); verify rows {same}/{n} the reference's \
         picks (floor {fsame}/{n}); fast kernels {}, launches {}",
        g.fast_kernels,
        g.launches()
    );
}

/// The gate (module doc): every arm's `mode` graded against its `floor`
/// (`assert_graded`, the GPU-vs-CPU bounds); the verify rows at most 2
/// points behind the floor's (and at least 90 %); a fast kernel seen
/// running if there is one, none if not. With `bits` (the net under test
/// runs no kernel of its own: exact, or fast with no fast kernel) also
/// nothing at all to grade: every logit equal in every arm, ΔNLL 0, every
/// row's pick the reference's.
pub(crate) fn assert_graded_vs_exact(what: &str, g: &Graded, bits: bool) {
    assert_graded_vs_exact_with(what, g, bits, Top1::Points(94));
}

/// `assert_graded_vs_exact` with the top-1 rule `top1` (`assert_graded_with`).
pub(crate) fn assert_graded_vs_exact_with(what: &str, g: &Graded, bits: bool, top1: Top1) {
    report_graded(what, g);
    let launched = g.launches();
    if g.fast_kernels {
        assert!(
            launched > 0,
            "{what}: fast kernels, but none ran in any arm: the gate graded exact's kernels"
        );
    } else {
        assert_eq!(
            launched, 0,
            "{what}: fast-set launches where `fast_kernels` says none: the bit check is wrong"
        );
    }
    let (n, same, fsame) = g.rows;
    if bits {
        assert!(!g.fast_kernels, "{what}: bits with fast kernels");
        for a in &g.arms {
            let m = &a.mode;
            assert!(
                m.top1 == m.positions && m.max_abs == 0.0 && m.sum_kl == 0.0,
                "{what} [{}]: no kernel of its own, so exact's bits: {m:?}",
                a.arm.name()
            );
            assert!(a.dm.iter().all(|d| d.1 == 0.0), "{what}: ΔNLL");
        }
        assert_eq!(same, n, "{what}: verify rows");
    }
    for a in &g.arms {
        assert!(
            a.mode.positions > 0,
            "{what} [{}]: nothing graded",
            a.arm.name()
        );
        if let Err(e) = assert_arm(what, a, top1) {
            panic!("{e}");
        }
    }
    assert!(
        same * 100 + 2 * n >= fsame * 100 && same * 100 >= 90 * n,
        "{what}: verify rows {same}/{n} the reference's picks (floor {fsame}/{n})"
    );
}

/// A GPU of another architecture than GPU 0's, as device 1 (the cross-arch
/// floor), if this host has one.
fn other_arch_gpu() -> Option<&'static Gpu> {
    let a0 = Gpu::open_device(0).ok()?.info().ok()?.arch();
    let g1 = Gpu::open_device(1).ok()?;
    if g1.info().ok()?.arch() == a0 {
        return None;
    }
    Some(Box::leak(Box::new(g1)))
}

/// The model `m` placed in `mode` on `device` (`BackendOptions`).
fn net(m: &Model, device: Device, kv: KvType, mode: Numerics) -> Backend<'_> {
    let o = BackendOptions::default()
        .device(device)
        .max_ctx(CTX)
        .kv(kv)
        .numerics(mode);
    Backend::new(m, o).unwrap()
}

fn load(path: &Path) -> Model {
    crate::gguf::load_with(path.to_str().unwrap(), Some(false)).unwrap()
}

/// The model at `path` (`under`: the net under test's file, the same by
/// default) graded in numerics mode `mode` against exact on `device`, with
/// this host's floor; and whether the net under test has no kernel of its
/// own (`assert_graded_vs_exact`'s `bits`).
pub(crate) fn graded_vs_exact(
    path: &Path,
    under: Option<&Path>,
    mode: Numerics,
    device: Device,
) -> (Graded, bool) {
    // A model each: a CPU model releases the bytes it repacks.
    let (mr, mo, mf) = (load(path), load(under.unwrap_or(path)), load(path));
    let r = net(&mr, device, KvType::F16, Numerics::Exact);
    let o = net(&mo, device, KvType::F16, mode);
    let bits = under.is_none() && (mode == Numerics::Exact || !o.fast_kernels());
    let g = match device {
        Device::Cpu => {
            let f = crate::model::Qwen2::new_with(&mf, false);
            grade(&r, &f, &o, &mr, Floor::CpuRows, device)
        }
        _ => match other_arch_gpu() {
            Some(g1) => {
                let km = Box::leak(Box::new(crate::cuda::Kernels::module_for(g1, &mf).unwrap()));
                let f = crate::cuda::qwen2::CudaQwen2::new(g1, km, &mf, CTX).unwrap();
                grade(&r, &f, &o, &mr, Floor::CrossArch, device)
            }
            None => {
                let f = net(&mf, device, KvType::Q8, Numerics::Exact);
                grade(&r, &f, &o, &mr, Floor::KvQ8, device)
            }
        },
    };
    (g, bits)
}

/// `graded_vs_exact` for `mode` on the GPU, asserted.
fn gpu_mode(mode: Numerics) {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let (g, bits) = graded_vs_exact(&p, None, mode, Device::Cuda);
    let arms: Vec<_> = g.arms.iter().map(|a| a.arm).collect();
    assert_eq!(arms, [Arm::Decode, Arm::Batched, Arm::Cached, Arm::Load]);
    assert_graded_vs_exact(&format!("0.5B cuda {mode}"), &g, bits);
}

/// `graded_vs_exact` for `mode` on the CPU, asserted.
fn cpu_mode(mode: Numerics) {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let (g, bits) = graded_vs_exact(&p, None, mode, Device::Cpu);
    assert_eq!(g.floor_name, Floor::CpuRows.name());
    let arms: Vec<_> = g.arms.iter().map(|a| a.arm).collect();
    assert_eq!(arms, [Arm::Decode, Arm::Batched, Arm::Cached]);
    assert_graded_vs_exact(&format!("0.5B cpu {mode}"), &g, bits);
}

/// The sanity check (ADR M3): exact graded against exact is green, with
/// nothing to grade in any arm (KL 0, ΔNLL 0), against a floor that isn't
/// zero.
#[test]
fn graded_exact_vs_exact() {
    gpu_mode(Numerics::Exact);
}

/// Fast graded against exact: green; while no fast kernel exists, exact's
/// bits in every arm (the fast module is the exact source under
/// `cuda::FAST_DEFINE`), and no fast-set launch.
#[test]
fn graded_fast_vs_exact() {
    gpu_mode(Numerics::Fast);
}

#[test]
fn graded_exact_vs_exact_cpu() {
    cpu_mode(Numerics::Exact);
}

#[test]
fn graded_fast_vs_exact_cpu() {
    cpu_mode(Numerics::Fast);
}

/// The gate has teeth: another quantization of the same weights (Q4_0 for
/// the Q4_K_M reference: a model change, which a reordering must never be)
/// fails it on the GPU in every arm, the batched ones included, on a bound
/// (not on the floor check: the floor is the same as in the passing
/// tests).
#[test]
fn graded_catches_another_model() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let other =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/Qwen2.5-0.5B-Instruct-Q4_0.gguf");
    if !other.exists() {
        eprintln!("skipping: {} not present", other.display());
        return;
    }
    let (g, bits) = graded_vs_exact(&p, Some(&other), Numerics::Exact, Device::Cuda);
    assert!(!bits);
    report_graded("0.5B Q4_0 vs Q4_K_M exact", &g);
    assert_eq!(g.arms.len(), 4);
    for a in &g.arms {
        let e = assert_arm("Q4_0", a, Top1::Points(94))
            .expect_err("another model passed an arm of the gate");
        eprintln!("as it must: {e}");
        assert!(!e.contains("reference spread"), "{e}");
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        assert_graded_vs_exact("Q4_0", &g, false)
    }));
    assert!(r.is_err(), "another model passed the fast-vs-exact gate");
}

/// The clustered SE: a hand-worked case, and one cluster has none.
#[test]
fn clustered_mean_by_hand() {
    // Clusters {0: 1, 3}, {1: 2, 2}: mean 2; cluster sums of deviations
    // (-1 + 1) = 0 and (0 + 0) = 0 -> SE 0.
    assert_eq!(
        clustered_mean(&[(0, 1.0), (0, 3.0), (1, 2.0), (1, 2.0)]),
        (2.0, 0.0)
    );
    // {0: 0, 0}, {1: 2, 2}: mean 1, sums -2 and 2: v = 8 / 16 * 2 = 1.
    assert_eq!(
        clustered_mean(&[(0, 0.0), (0, 0.0), (1, 2.0), (1, 2.0)]),
        (1.0, 1.0)
    );
    assert!(clustered_mean(&[(0, 1.0), (0, 2.0)]).1.is_nan());
    assert_eq!(clustered_mean(&[]), (0.0, 0.0));
}

/// The launch count the gate reads: a fast GPU backend's moves when a
/// fast-set kernel counts its launch (`CudaQwen2::fast_launched`), and the
/// gate then needs it to (`assert_graded_vs_exact`): a backend claiming
/// fast kernels that ran none fails, one with launches but no claim fails.
#[test]
fn fast_launch_count_is_read() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let m = load(&p);
    let o = net(&m, Device::Cuda, KvType::F16, Numerics::Fast);
    assert_eq!(o.fast_launches(), 0);
    o.gpu_nets()[0].fast_launched();
    assert_eq!(o.fast_launches(), 1);
    // A grading with nothing in it but its claims.
    let g = |fast_kernels, launches| Graded {
        arms: vec![ArmGrade {
            launches,
            ..ArmGrade::new(Arm::Batched)
        }],
        rows: (0, 0, 0),
        floor_name: "none",
        fast_kernels,
    };
    for (fk, l, why) in [
        (true, 0, "none ran in any arm"),
        (false, 1, "the bit check is wrong"),
    ] {
        let r = catch_unwind(AssertUnwindSafe(|| {
            assert_graded_vs_exact("t", &g(fk, l), false)
        }));
        let e = panic_text(&*r.expect_err(why));
        assert!(e.contains(why), "{e}");
    }
}
