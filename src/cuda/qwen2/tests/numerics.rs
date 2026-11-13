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
use crate::lm::{
    Backend, BackendCache, BackendOptions, BatchLm, Device, Lm, RowHead, RowOut, SeqRow,
};
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
    /// PROVISIONAL (#232; the size is set from measurements): the release
    /// suite's FP8 gate, the same bars as `FP8` on about half its
    /// positions: decode 6 x 17 + 9 x 48 = 534, batched 4 x 3 x 49 = 588,
    /// cached 11 x 49 = 539, load 40 x 13 = 520.
    pub(crate) const FP8_FAST: Sizes = Sizes {
        decode_windows: 9,
        batch_windows: 4,
        verify_windows: BATCH_WINDOWS,
        cached_windows: Some(11),
        after: 48,
        load: LOAD,
        load_steps: 13,
    };
    /// `FP8` in a debug-assertions build: every arm, prefix length and
    /// route of the release gate (the load arm's 32 rows reach the pool's
    /// tensor-core attention, `ATT_NB_MIN`), on about a quarter of the CPU
    /// reference's tokens: decode 6 x 17 + 48 = 150, batched 3 x 49 = 147,
    /// cached 2 x 49 = 98, load 32 x 4 = 128 positions. The two CPU nets
    /// are the test's cost: at `FP8`'s size it held the GPU test lock for
    /// over 2700 s in a debug shard with 4 test threads (1031-080835-d64f).
    /// This size keeps what only a debug build checks on those paths:
    /// every launch's argument count (`cuda::Kernel`'s cuFuncGetParamInfo
    /// check), the pool's page table, the fast counter's assertion and
    /// overflow checks. The statistical verdict at >= 1000 positions per
    /// arm is the release build's (`Sizes::FP8`): the debug build adds
    /// checks, not other arithmetic, so its larger run would grade the
    /// same numbers again.
    pub(crate) const FP8_DEBUG: Sizes = Sizes {
        decode_windows: 1,
        batch_windows: 1,
        verify_windows: 1,
        cached_windows: Some(2),
        after: 48,
        load: 32,
        load_steps: 4,
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
#[derive(Debug)]
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

    /// One position: the reference's logits, the floor's and the net's
    /// (each None in the pass that doesn't run it: `Reference`);
    /// `next`: (window, the teacher-forced token after it).
    fn tally(
        &mut self,
        lr: &[f32],
        lf: Option<&[f32]>,
        lo: Option<&[f32]>,
        next: Option<(usize, u32)>,
    ) {
        let n = next.map(|(w, t)| (w, t, nll(lr, t)));
        if let Some(lf) = lf {
            self.floor.tally(lr, lf);
            if let Some((w, t, n)) = n {
                self.df.push((w, nll(lf, t) - n));
            }
        }
        if let Some(lo) = lo {
            self.mode.tally(lr, lo);
            if let Some((w, t, n)) = n {
                self.dm.push((w, nll(lo, t) - n));
            }
        }
    }

    /// ΔNLL (mean, clustered SE): the net under test's and the floor's.
    pub(crate) fn dnll(&self) -> ((f64, f64), (f64, f64)) {
        (clustered_mean(&self.dm), clustered_mean(&self.df))
    }
}

/// A grading (`grade`): every arm, and the batched passes' rows.
#[derive(Debug)]
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

/// `grade` with the arms' sizes `s`: the three nets stepped together,
/// position by position (`reference` + `Reference::grade` run the same
/// calls in two passes).
pub(crate) fn grade_sized<F: Lm>(
    r: &Backend,
    f: &F,
    o: &Backend,
    m: &Model,
    floor: Floor,
    device: Device,
    sz: &Sizes,
) -> Graded {
    let inp = Inputs::new(m, device, sz);
    let nets = Nets {
        r: Some(r),
        f: Some(f),
        o: Some(o),
    };
    walk(&nets, &inp, floor, &mut Kept::default(), None)
}

/// The reference's half of a grading, run before the net under test's
/// (#232): every arm's floor graded against the reference, and what the
/// net under test will be graded against (the reference's logits at every
/// graded position, its decode traces and verify picks, the inputs).
/// `Reference::grade` runs the net under test on the same inputs and
/// finishes the grading: the same `Graded` as `grade_sized` with these
/// nets, which makes the same calls on each net in the same order, only
/// not interleaved. For a GPU net graded against CPU references: the CPU
/// work is done before the test takes the GPU test lock (the FP8 gate's
/// CPU nets held it for over 2700 s on a packed pod). The cost is memory:
/// a vocabulary of logits per graded position (0.6 MB at Qwen2.5's
/// 151,936).
pub(crate) struct Reference {
    inp: Inputs,
    floor: Floor,
    kept: Kept,
    graded: Graded,
}

/// `Reference`: run the reference `r` and the floor `f` over `m`'s inputs
/// for `device` at sizes `sz`.
pub(crate) fn reference<F: Lm>(
    r: &Backend,
    f: &F,
    m: &Model,
    floor: Floor,
    device: Device,
    sz: &Sizes,
) -> Reference {
    let inp = Inputs::new(m, device, sz);
    let nets: Nets<'_, '_, F> = Nets {
        r: Some(r),
        f: Some(f),
        o: None,
    };
    let mut kept = Kept::default();
    let graded = walk(&nets, &inp, floor, &mut kept, None);
    Reference {
        inp,
        floor,
        kept,
        graded,
    }
}

impl Reference {
    /// The net under test's half: `o` on the same inputs, graded against
    /// the kept reference (consumed as it goes).
    pub(crate) fn grade(mut self, o: &Backend) -> Graded {
        let nets: Nets<'_, '_, Backend<'_>> = Nets {
            r: None,
            f: None,
            o: Some(o),
        };
        let g = walk(
            &nets,
            &self.inp,
            self.floor,
            &mut self.kept,
            Some(self.graded),
        );
        assert!(
            self.kept.next == self.kept.logits.len() && self.kept.traces.is_empty(),
            "the net under test's pass read {} of {} kept positions",
            self.kept.next,
            self.kept.logits.len()
        );
        g
    }
}

/// A grading's inputs (module doc): the prompts' tokens and the natural
/// text's, split between the arms.
struct Inputs {
    device: Device,
    sz: Sizes,
    prompts: Vec<Vec<u32>>,
    ids: Vec<u32>,
    /// A batched window's tokens, and a cached one's (`Sizes::cached_windows`).
    span: usize,
    cspan: usize,
    load_len: usize,
}

impl Inputs {
    fn new(m: &Model, device: Device, sz: &Sizes) -> Inputs {
        let tok = crate::tokenizer::Tokenizer::from_gguf(m);
        let (c0, c1) = CACHED;
        let longest = *prefixes(device).iter().max().unwrap();
        let (span, cspan) = match sz.cached_windows {
            None => ((longest.max(c0 + c1)) + sz.after + 1, 0),
            Some(_) => (longest + sz.after + 1, c0 + c1 + sz.after + 1),
        };
        let load_len = LOAD_PREFIX_MAX + sz.load_steps + 1;
        let n = sz.decode_windows * WINDOW
            + sz.batch_windows * span
            + sz.cached_windows.unwrap_or(0) * cspan
            + sz.load * load_len;
        Inputs {
            device,
            sz: *sz,
            prompts: PROMPTS.iter().map(|p| tok.encode(p)).collect(),
            ids: natural_ids(m, n),
            span,
            cspan,
            load_len,
        }
    }

    /// The natural text's split: (decode, batched, cached, load) tokens.
    fn split(&self) -> (&[u32], &[u32], &[u32], &[u32]) {
        let sz = &self.sz;
        let (dec, rest) = self.ids.split_at(sz.decode_windows * WINDOW);
        let (bat, rest) = rest.split_at(sz.batch_windows * self.span);
        let (cac, load) = rest.split_at(sz.cached_windows.unwrap_or(0) * self.cspan);
        (dec, bat, cac, load)
    }
}

/// The nets a pass of `walk` runs. All three: `grade_sized`. The reference
/// and the floor, `o` None: `reference` (the reference's results kept).
/// `o` alone: `Reference::grade` (the kept results read back, in order).
struct Nets<'a, 'm, F: Lm> {
    r: Option<&'a Backend<'m>>,
    f: Option<&'a F>,
    o: Option<&'a Backend<'m>>,
}

/// The reference's results a `reference` pass keeps for the pass of the
/// net under test, in the order `walk` makes them.
#[derive(Default)]
struct Kept {
    /// Its logits at every graded position (read back from `next` on;
    /// each is taken out as it is read).
    logits: Vec<Vec<f32>>,
    next: usize,
    /// The decode arm's traces of the prompts (taken from the front).
    traces: std::collections::VecDeque<super::Trace>,
    /// Its `verify_argmax` picks per batched prefix.
    picks: std::collections::VecDeque<Vec<u32>>,
}

impl Kept {
    /// The next kept logits, taken out.
    fn read(&mut self) -> Vec<f32> {
        let l = std::mem::take(&mut self.logits[self.next]);
        self.next += 1;
        l
    }
}

/// One position's logits from each net the pass runs: (the reference's,
/// the floor's, the net under test's).
type At = (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>);

/// A cache per net the pass runs (`Nets::caches`).
struct Caches<RC, FC, OC> {
    r: Option<RC>,
    f: Option<FC>,
    o: Option<OC>,
}

type NetCaches<F> = Caches<BackendCache, <F as Lm>::Cache, BackendCache>;

impl<F: Lm> Nets<'_, '_, F> {
    /// The reference's logits: `r`'s by `run` (kept when the pass has no
    /// net under test), or a `reference` pass's, read back.
    fn reference(&self, kept: &mut Kept, run: impl FnOnce(&Backend) -> Vec<f32>) -> Vec<f32> {
        match self.r {
            Some(r) => {
                let l = run(r);
                if self.o.is_none() {
                    kept.logits.push(l.clone());
                }
                l
            }
            None => kept.read(),
        }
    }

    /// Fresh caches of `CTX` positions (`o`'s too with `with_o`).
    fn caches(&self, with_o: bool) -> NetCaches<F> {
        Caches {
            r: self.r.map(|r| r.new_cache(CTX)),
            f: self.f.map(|f| f.new_cache(CTX)),
            o: self.o.filter(|_| with_o).map(|o| o.new_cache(CTX)),
        }
    }

    fn forward(&self, kept: &mut Kept, c: &mut NetCaches<F>, t: u32, pos: usize) -> At {
        let lr = self.reference(kept, |r| r.forward(t, pos, c.r.as_mut().unwrap()));
        let lf = self.f.map(|f| f.forward(t, pos, c.f.as_mut().unwrap()));
        let lo = self.o.map(|o| o.forward(t, pos, c.o.as_mut().unwrap()));
        (lr, lf, lo)
    }

    fn prefill(&self, kept: &mut Kept, c: &mut NetCaches<F>, ids: &[u32], pos: usize) -> At {
        let lr = self.reference(kept, |r| r.prefill_logits(ids, pos, c.r.as_mut().unwrap()));
        let lf = self
            .f
            .map(|f| f.prefill_logits(ids, pos, c.f.as_mut().unwrap()));
        let lo = self
            .o
            .map(|o| o.prefill_logits(ids, pos, c.o.as_mut().unwrap()));
        (lr, lf, lo)
    }

    fn feed(&self, c: &mut NetCaches<F>, ids: &[u32], pos: usize) {
        if let (Some(r), Some(rc)) = (self.r, c.r.as_mut()) {
            r.feed_batch(ids, pos, rc);
        }
        if let (Some(f), Some(fc)) = (self.f, c.f.as_mut()) {
            f.feed_batch(ids, pos, fc);
        }
        if let (Some(o), Some(oc)) = (self.o, c.o.as_mut()) {
            o.feed_batch(ids, pos, oc);
        }
    }
}

/// `after` positions from `p` on, teacher-forced, a token per step, on
/// caches that hold positions 0..p: graded into `a` (window `w`).
#[allow(clippy::too_many_arguments)]
fn decode_after<F: Lm>(
    a: &mut ArmGrade,
    n: &Nets<'_, '_, F>,
    kept: &mut Kept,
    c: &mut NetCaches<F>,
    seq: &[u32],
    p: usize,
    w: usize,
    after: usize,
) {
    for pos in p..p + after {
        let (lr, lf, lo) = n.forward(kept, c, seq[pos], pos);
        a.tally(&lr, lf.as_deref(), lo.as_deref(), Some((w, seq[pos + 1])));
    }
}

/// Every arm over `inp`, for the nets in `n` (`Nets`): graded into
/// `prev`'s arms when a `reference` pass made them (the net under test's
/// half), else new ones.
fn walk<F: Lm>(
    n: &Nets<'_, '_, F>,
    inp: &Inputs,
    floor: Floor,
    kept: &mut Kept,
    prev: Option<Graded>,
) -> Graded {
    let (sz, pre, (c0, c1)) = (&inp.sz, prefixes(inp.device), CACHED);
    let (dec, bat, cac, load) = inp.split();
    let cwins = sz.cached_windows.unwrap_or(0);
    let (mut rows, mut prev_arms) = match prev {
        Some(g) => (g.rows, Some(g.arms.into_iter())),
        None => ((0, 0, 0), None),
    };
    let mut arm = |which: Arm| match prev_arms.as_mut() {
        Some(it) => {
            let a = it.next().unwrap();
            assert_eq!(a.arm, which, "the reference pass's arms");
            a
        }
        None => ArmGrade::new(which),
    };
    let launches = || n.o.map_or(0, |o| o.fast_launches());
    let mut arms = Vec::new();

    // Decode: the prompts' greedy traces, then text from position 0.
    let mut a = arm(Arm::Decode);
    let l0 = launches();
    for ids in &inp.prompts {
        let t = match n.r {
            Some(r) => trace(r, ids, STEPS),
            None => kept.traces.pop_front().unwrap(),
        };
        if let Some(f) = n.f {
            compare_trace(&t, f, &mut a.floor);
        }
        if let Some(o) = n.o {
            compare_trace(&t, o, &mut a.mode);
        }
        if n.o.is_none() {
            kept.traces.push_back(t);
        }
    }
    for (w, seq) in dec.chunks(WINDOW).enumerate() {
        let mut c = n.caches(true);
        a.start();
        for (pos, &t) in seq.iter().enumerate() {
            let (lr, lf, lo) = n.forward(kept, &mut c, t, pos);
            let next = seq.get(pos + 1).map(|&t| (w, t));
            a.tally(&lr, lf.as_deref(), lo.as_deref(), next);
        }
    }
    a.launches += launches() - l0;
    arms.push(a);

    // Batched: each prefix in one prefill_logits call, the decode after
    // it; verify_argmax's rows over the same prefix.
    let mut a = arm(Arm::Batched);
    let l0 = launches();
    for (i, seq) in bat.chunks(inp.span).enumerate() {
        let w = sz.decode_windows + i;
        for &p in pre {
            let mut c = n.caches(true);
            a.start();
            let (lr, lf, lo) = n.prefill(kept, &mut c, &seq[..p], 0);
            a.tally(&lr, lf.as_deref(), lo.as_deref(), Some((w, seq[p])));
            decode_after(&mut a, n, kept, &mut c, seq, p, w, sz.after);
            if i >= sz.verify_windows {
                continue;
            }
            let ids = &seq[..p];
            let pr = match n.r {
                Some(r) => r.verify_argmax(ids, 0, &mut r.new_cache(CTX)),
                None => kept.picks.pop_front().unwrap(),
            };
            let same = |x: &[u32]| pr.iter().zip(x).filter(|(a, b)| a == b).count();
            if let Some(f) = n.f {
                rows.0 += p;
                rows.2 += same(&f.verify_argmax(ids, 0, &mut f.new_cache(CTX)));
            }
            if let Some(o) = n.o {
                rows.1 += same(&o.verify_argmax(ids, 0, &mut o.new_cache(CTX)));
            } else {
                kept.picks.push_back(pr);
            }
        }
    }
    a.launches += launches() - l0;
    arms.push(a);

    // Cached: a prefix in the cache, the rest in a second batched pass.
    let mut a = arm(Arm::Cached);
    let l0 = launches();
    let (cwin, w0) = match sz.cached_windows {
        None => (bat.chunks(inp.span), sz.decode_windows),
        Some(_) => (cac.chunks(inp.cspan), sz.decode_windows + sz.batch_windows),
    };
    for (w, seq) in cwin.enumerate() {
        let w = w0 + w;
        let mut c = n.caches(true);
        a.start();
        n.feed(&mut c, &seq[..c0], 0);
        let (lr, lf, lo) = n.prefill(kept, &mut c, &seq[c0..c0 + c1], c0);
        a.tally(&lr, lf.as_deref(), lo.as_deref(), Some((w, seq[c0 + c1])));
        decode_after(&mut a, n, kept, &mut c, seq, c0 + c1, w, sz.after);
    }
    a.launches += launches() - l0;
    arms.push(a);

    // Load (the GPU: a pool is the GPU model's): every row of LOAD
    // sequences' steps, taken together, against each sequence alone.
    if inp.device != Device::Cpu {
        let mut a = arm(Arm::Load);
        let l0 = launches();
        let seqs: Vec<&[u32]> = load.chunks(inp.load_len).collect();
        let lw = sz.decode_windows + sz.batch_windows + cwins;
        let mut pool = n.o.map(|o| o.new_pool(sz.load, CTX).unwrap());
        let mut alone = Vec::with_capacity(sz.load);
        for (s, seq) in seqs.iter().enumerate() {
            let p = load_prefix(s);
            let mut c = n.caches(false);
            n.feed(&mut c, &seq[..p], 0);
            // The reference's logits are kept (or read back) at the tally
            // below, in its order (step by step), not here.
            let at: Vec<_> = (p..p + sz.load_steps)
                .map(|pos| {
                    let t = seq[pos];
                    let lr = n.r.map(|r| r.forward(t, pos, c.r.as_mut().unwrap()));
                    let lf = n.f.map(|f| f.forward(t, pos, c.f.as_mut().unwrap()));
                    (lr, lf)
                })
                .collect();
            alone.push(at.into_iter());
            if let (Some(o), Some(pool)) = (n.o, pool.as_mut()) {
                o.feed_slot(pool, s, &seq[..p], 0).unwrap();
            }
        }
        for k in 0..sz.load_steps {
            // Every logit back (k past CAND_MAX: the host selects from the
            // row's logits, no penalty).
            let mut out = match (n.o, pool.as_mut()) {
                (Some(o), Some(pool)) => {
                    let spec = CandSpec {
                        k: o.vocab(),
                        penalty: 1.0,
                        last_n: 0,
                    };
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
                    Some((o.vocab(), o.step(pool, &step).unwrap().into_iter()))
                }
                _ => None,
            };
            for (s, seq) in seqs.iter().enumerate() {
                let lo = out.as_mut().map(|(vocab, rows)| {
                    let RowOut::Cands(cs) = rows.next().unwrap() else {
                        panic!("a sampled row's candidates")
                    };
                    assert_eq!(cs.len(), *vocab, "every logit back");
                    let mut lo = vec![f32::NAN; *vocab];
                    for c in cs {
                        lo[c.id as usize] = c.logit;
                    }
                    lo
                });
                let (lr, lf) = alone[s].next().unwrap();
                let lr = lr.unwrap_or_else(|| kept.read());
                let next = seq[load_prefix(s) + k + 1];
                a.start();
                a.tally(&lr, lf.as_deref(), lo.as_deref(), Some((lw + s, next)));
                if n.o.is_none() {
                    kept.logits.push(lr);
                }
            }
        }
        a.launches += launches() - l0;
        arms.push(a);
    }
    Graded {
        arms,
        rows,
        floor_name: floor.name(),
        fast_kernels: n.o.is_some_and(|o| o.fast_kernels()),
    }
}

/// The load arm's rows, diagnosed (the FP8 gate's max |dlogit|, an
/// `#[ignore]`d test): the same ids as `grade_sized`'s load arm, stepped
/// in a pool of all `sz.load` (as the gate), the same pool in reversed
/// slot and row order, a pool of the first half only, and each sequence
/// alone through `o`'s decode path; against `r` (the reference) and `f`
/// (the floor). Prints the worst rows: the logit with the max |pool - r|,
/// its rank in r and both probabilities, its value in every run; and per
/// row the max |a - b| between the runs.
pub(crate) fn load_diag<F: Lm>(r: &Backend, f: &F, o: &Backend, m: &Model, sz: &Sizes) {
    let pre = prefixes(Device::Cuda);
    let (c0, c1) = CACHED;
    let longest = *pre.iter().max().unwrap();
    let (span, cspan) = match sz.cached_windows {
        None => ((longest.max(c0 + c1)) + sz.after + 1, 0),
        Some(_) => (longest + sz.after + 1, c0 + c1 + sz.after + 1),
    };
    let cwins = sz.cached_windows.unwrap_or(0);
    let load_len = LOAD_PREFIX_MAX + sz.load_steps + 1;
    let total = sz.decode_windows * WINDOW + sz.batch_windows * span + cwins * cspan;
    let ids = natural_ids(m, total + sz.load * load_len);
    let seqs: Vec<&[u32]> = ids[total..].chunks(load_len).collect();
    let n = seqs.len();
    let spec = CandSpec {
        k: o.vocab(),
        penalty: 1.0,
        last_n: 0,
    };
    // A pool run: `order` the sequences (slot i holds seqs[order[i]]),
    // rows in that order; logits [seq][step].
    let pooled = |order: &[usize]| -> Vec<Vec<Vec<f32>>> {
        let mut pool = o.new_pool(order.len(), CTX).unwrap();
        for (slot, &s) in order.iter().enumerate() {
            o.feed_slot(&mut pool, slot, &seqs[s][..load_prefix(s)], 0)
                .unwrap();
        }
        let mut out = vec![Vec::new(); n];
        for k in 0..sz.load_steps {
            let step: Vec<SeqRow> = (order.iter().enumerate())
                .map(|(slot, &s)| {
                    let pos = load_prefix(s) + k;
                    SeqRow {
                        token: seqs[s][pos],
                        pos,
                        slot,
                        head: RowHead::Sample {
                            spec,
                            recent: &seqs[s][..=pos],
                        },
                    }
                })
                .collect();
            for (i, row) in o.step(&mut pool, &step).unwrap().into_iter().enumerate() {
                let RowOut::Cands(cs) = row else {
                    panic!("candidates")
                };
                let mut lo = vec![f32::NAN; o.vocab()];
                for c in cs {
                    lo[c.id as usize] = c.logit;
                }
                out[order[i]].push(lo);
            }
        }
        out
    };
    let all: Vec<usize> = (0..n).collect();
    let rev: Vec<usize> = (0..n).rev().collect();
    let half: Vec<usize> = (0..n / 2).collect();
    let (pa, pb, ph) = (pooled(&all), pooled(&rev), pooled(&half));
    let mut rows = Vec::new();
    let mx = |a: &[f32], b: &[f32]| {
        (a.iter().zip(b))
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let (mut s_ab, mut s_ah, mut s_as, mut s_sr, mut s_fr, mut s_ar) =
        (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
    let (mut t20_a, mut t20_s, mut t20_f) = (0f32, 0f32, 0f32);
    for (s, seq) in seqs.iter().enumerate() {
        let p = load_prefix(s);
        let (mut rc, mut fc, mut oc) = (r.new_cache(CTX), f.new_cache(CTX), o.new_cache(CTX));
        r.feed_batch(&seq[..p], 0, &mut rc);
        f.feed_batch(&seq[..p], 0, &mut fc);
        o.feed_batch(&seq[..p], 0, &mut oc);
        for k in 0..sz.load_steps {
            let pos = p + k;
            let lr = r.forward(seq[pos], pos, &mut rc);
            let lf = f.forward(seq[pos], pos, &mut fc);
            let ls = o.forward(seq[pos], pos, &mut oc);
            let la = &pa[s][k];
            let (lpr, lpa) = (log_softmax(&lr), log_softmax(la));
            let mut top: Vec<usize> = (0..lr.len()).collect();
            top.sort_by(|&x, &y| lr[y].total_cmp(&lr[x]));
            let t20 = |l: &[f32]| {
                top[..20]
                    .iter()
                    .map(|&i| (l[i] - lr[i]).abs())
                    .fold(0f32, f32::max)
            };
            t20_a = t20_a.max(t20(la));
            t20_s = t20_s.max(t20(&ls));
            t20_f = t20_f.max(t20(&lf));
            let (i, d) = (la.iter().zip(&lr).map(|(x, y)| (x - y).abs()).enumerate())
                .fold((0, 0f32), |b, (i, d)| if d > b.1 { (i, d) } else { b });
            let rank = top.iter().position(|&j| j == i).unwrap();
            let h = (s < n / 2).then(|| ph[s][k][i]);
            let ab = mx(la, &pb[s][k]);
            let ah = if s < n / 2 { mx(la, &ph[s][k]) } else { 0.0 };
            let (as_, sr, fr) = (mx(la, &ls), mx(&ls, &lr), mx(&lf, &lr));
            s_ab = s_ab.max(ab);
            s_ah = s_ah.max(ah);
            s_as = s_as.max(as_);
            s_sr = s_sr.max(sr);
            s_fr = s_fr.max(fr);
            s_ar = s_ar.max(d);
            rows.push((d, format!(
                "seq {s:2} step {k:2} pos {pos:3}: max|pool-ref| {d:.3} at token {i} rank {rank} \
                 (p_ref {:.2e}, p_pool {:.2e}; ref top-1 {:.2}): ref {:.3} floor {:.3} pool {:.3} \
                 rev {:.3} half {} solo {:.3} | row max |pool-rev| {ab:.3} |pool-half| {ah:.3} \
                 |pool-solo| {as_:.3} |solo-ref| {sr:.3} |floor-ref| {fr:.3}",
                lpr[i].exp(), lpa[i].exp(), lr[top[0]], lr[i], lf[i], la[i], pb[s][k][i],
                h.map_or("-".into(), |v| format!("{v:.3}")), ls[i]
            )));
        }
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, l) in rows.iter().take(15) {
        eprintln!("{l}");
    }
    let over = |t: f32| rows.iter().filter(|r| r.0 >= t).count();
    eprintln!(
        "load diag over {} rows: max |pool-ref| {s_ar:.3} (rows >= 6: {}, >= 4: {}), |solo-ref| {s_sr:.3}, \
         |floor-ref| {s_fr:.3}; |pool-rev| {s_ab:.3}, |pool-half| {s_ah:.3}, |pool-solo| {s_as:.3}; \
         over r's top-20 only: pool {t20_a:.3} solo {t20_s:.3} floor {t20_f:.3}",
        rows.len(),
        over(6.0),
        over(4.0)
    );
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
    // Every arm graded before failing, so one run names every bound that
    // trips.
    let mut errs = Vec::new();
    for a in &g.arms {
        assert!(
            a.mode.positions > 0,
            "{what} [{}]: nothing graded",
            a.arm.name()
        );
        if let Err(e) = assert_arm(what, a, top1) {
            errs.push(e);
        }
    }
    assert!(errs.is_empty(), "{}", errs.join("\n"));
    assert!(
        same * 100 + 2 * n >= fsame * 100 && same * 100 >= 90 * n,
        "{what}: verify rows {same}/{n} the reference's picks (floor {fsame}/{n})"
    );
}

/// The FP8 gate in a debug-assertions build (lead ruling, 10-31): no
/// statistics. The graded bars (the binomial top-1, the KL mean, the
/// outlier count, the verify rows) are a test of >= 1000 positions per arm,
/// whose run of record is the release build (the debug build does the same
/// arithmetic); `Sizes::FP8_DEBUG`'s 98-150 positions can't carry them (job
/// 1031-115732-b712: the batched arm's 137/147 against a floor of 142,
/// 2 SE 4.4, where the release build's 1000-position arms pass). Kept:
/// every arm graded with a fast kernel seen running (the per-arm count is
/// the caller's), and the sanity backstops on the net under test
/// (`assert_sane`: finite logits, max |dlogit| < 12 over the whole
/// vocabulary). The numbers are printed, labelled info only.
pub(crate) fn assert_sane_vs_exact(what: &str, g: &Graded) {
    report_graded(&format!("{what} (debug build: info only, not graded)"), g);
    assert!(
        !g.fast_kernels || g.launches() > 0,
        "{what}: fast kernels, but none ran in any arm"
    );
    let errs: Vec<String> = (g.arms.iter())
        .filter_map(|a| {
            catch_unwind(AssertUnwindSafe(|| super::assert_sane(&a.mode)))
                .err()
                .map(|p| format!("{what} [{}]: {}", a.arm.name(), panic_text(&*p)))
        })
        .collect();
    assert!(errs.is_empty(), "{}", errs.join("\n"));
    eprintln!(
        "{what}: debug build: sanity checks only (every arm ran a fast kernel, finite logits, max \
         |dlogit| < 12); the statistical bars are the release build's run of record"
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

/// `reference` + `Reference::grade` make `grade_sized`'s `Graded`, field
/// for field (#232): every arm's tallies, ΔNLL, launches and the verify
/// rows, the f64 sums bit for bit (`Debug` prints the shortest exact
/// form). Three nets that all differ (the reference, a q8-KV floor, another
/// quantization under test), so no tally is trivially 0; every arm and
/// branch of the walk at a few positions each (two batched windows, one of
/// them past `verify_windows`; the load arm's pool), all on the GPU.
#[test]
fn graded_in_two_passes_is_graded_at_once() {
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
    let (mr, mf, mo) = (load(&p), load(&p), load(&other));
    let r = net(&mr, Device::Cuda, KvType::F16, Numerics::Exact);
    let f = net(&mf, Device::Cuda, KvType::Q8, Numerics::Exact);
    let o = net(&mo, Device::Cuda, KvType::F16, Numerics::Exact);
    let sz = Sizes {
        decode_windows: 1,
        batch_windows: 2,
        verify_windows: 1,
        cached_windows: Some(1),
        after: 3,
        load: 3,
        load_steps: 2,
    };
    let at_once = grade_sized(&r, &f, &o, &mr, Floor::KvQ8, Device::Cuda, &sz);
    let two = reference(&r, &f, &mr, Floor::KvQ8, Device::Cuda, &sz).grade(&o);
    assert_eq!(format!("{at_once:?}"), format!("{two:?}"));
    assert_eq!(two.arms.len(), 4);
    for a in &two.arms {
        assert!(
            a.mode.positions > 0 && a.mode.sum_kl > 0.0 && a.floor.sum_kl > 0.0,
            "{a:?}"
        );
    }
    assert!(two.rows.0 > 0, "{:?}", two.rows);
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
