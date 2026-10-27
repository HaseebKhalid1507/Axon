//! `axon serve`'s continuous batching (stage S2-lite, specs/r6-pick.md §4):
//! many requests decode together, one token each per step, over a pool of
//! KV slots on the GPU (`BatchLm`, cuda/qwen2/slots.rs).
//!
//! The engine thread runs a loop: take new requests, give a waiting one a
//! free slot, feed the next chunk of the prompts being prefilled into their
//! slots (`Batch::prefill`), then one step with a row per running request
//! (its last token at its
//! position, the greedy pick or its sampling candidates back), each request's
//! next token picked on the host (`Sampler::pick`, its own seeded rng), its
//! text sent, and finished requests' slots freed. A request admitted while
//! nothing else runs goes the S1 way (`Engine::job`: `generate` with CUDA
//! graphs and drafting, on a cache that is a view of a slot's rows (the
//! slot `lend_solo` picks, as `admit` would),
//! `BatchLm::slot_cache`: no cache beside the pool); if another request
//! arrives meanwhile, its rows are already in that slot (`BatchLm::adopt`
//! copies nothing) and it carries on in the steps.
//!
//! So while requests decode, a long prompt is fed a chunk per step, in
//! between their steps: they keep producing tokens while it prefills (a
//! whole 20k-token prompt as one pass would stop them for seconds). A chunk
//! is sized to a time budget (`--step-ms`: a step's chunk and decode
//! together take about that long, at the measured cost: `StepCost`), at least `--prefill-chunk` tokens; with
//! nothing decoding the prompt is fed whole. A chunk is a feed into its own
//! slot at its position (`feed_slot`), as a cache reuse feeds the rest of a
//! prompt, so its rows are the bits one feed of the whole prompt writes,
//! whatever the chunk sizes.
//!
//! Every row's results are the bits the one-sequence calls give
//! (`BatchLm`'s contract), and the host does per request what `generate`
//! does per token (a seeded pick from the same candidates, the same stop
//! rules), so a request's tokens are the same whatever else runs with it.

use super::api::Usage;
use super::tier::PoolTier;
use super::{Event, Job, TextOut};
use crate::kvsave::PAGE;
use crate::lm::{Backend, BatchLm, PoolPages, RowHead, RowOut, SeqRow, StepError};
use crate::sample::Sampler;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// A request decoding in a slot.
pub(super) struct Seq<'t> {
    pub job: Job,
    pub slot: usize,
    /// Prompt then generated tokens; the last one is the next step's row
    /// (its position hist.len() - 1).
    pub hist: Vec<u32>,
    pub ids: Vec<u32>,
    pub sampler: Option<Sampler>,
    pub out: TextOut<'t>,
    /// Prompt tokens whose rows were reused.
    pub cached: usize,
    pub prompt_ms: f64,
    /// When decoding started.
    pub t1: Instant,
}

/// A request whose prompt is being fed into its slot, a chunk per step.
struct Prefill<'t> {
    /// Its state once decoding (`hist` = the prompt).
    seq: Seq<'t>,
    /// Prompt tokens whose rows the slot holds.
    fed: usize,
    /// Its misses (`starve`): steps in a row passed over while requests
    /// decode; on idle steps, younger prompts that finished ahead of it.
    starved: usize,
    t0: Instant,
}

/// Misses (`starve`) after which a prompt goes first. While requests
/// decode a miss is a step passed over (in a row: under a steady trickle
/// of short prompts a long one still gets about one prefill step in
/// STARVE + 1); on idle steps it is a younger prompt finishing ahead of it
/// (at most STARVE of them).
const STARVE: usize = 4;

/// Prompt tokens a step feeds at least while requests decode (default,
/// `--prefill-chunk`; 0: no chunks, every prompt fed whole).
pub const PREFILL_CHUNK: usize = 256;
/// A step's target wall time while requests decode, in ms (default,
/// `--step-ms`, alias `--prefill-ms`; 0: no time budget, exactly
/// `--prefill-chunk` tokens): the time between a running request's tokens
/// the server aims at. 60, main's `--prefill-ms`: at 50 the bella 7B C=16
/// mix step ran at 60.0 ms p50 (+20 %, outside the spec's ±15 %: the 256
/// floor binds) and the avante 1.5B ITL p99 rose 17 % over main (jobs
/// 1025-174838-bb0d, 1025-174837-1554); 50 is `--step-ms 50`.
pub const STEP_MS: usize = 60;
/// The first chunk with a time budget, before a chunk's cost is known.
const SEED_CHUNK: usize = 512;
/// The prefill cost model: a token at depth d costs cost * (d + DEPTH_K)
/// (the weights' share as DEPTH_K positions of attention), so a chunk's
/// time is predicted at the depth it will run, not the last one's.
const DEPTH_K: f64 = 8192.0;
/// A step's prompt tokens when nothing decodes, before a chunk's cost is
/// known (as hybrid.rs STREAM_BATCH), and its time budget once it is: long
/// pieces at full speed, but an arrival still starts within about that (up
/// to twice it once rounded up to whole passes).
const IDLE_CHUNK: usize = 4096;
const IDLE_MS: f64 = 250.0;
/// Feeds of at least this many tokens teach the cost (smaller ones are
/// mostly a pass's fixed part).
const LEARN_MIN: usize = 128;
/// Feeds of fewer tokens than this teach a pass's fixed part (`StepCost`).
const PASS_FIXED_MAX: usize = 64;
/// A short feed's fixed-part sample is at most this many times the
/// estimate (`StepCost::learn_feed`): a stalled feed (a graph capture, a
/// page allocation, a foreign GPU) is one sample, not a new level.
const PASS_JUMP: f64 = 4.0;
/// Decode-only step times are kept per row count bucket: 1, 2, 3-4, ...,
/// 129-256 rows (`StepCost::dec`).
const DEC_BUCKETS: usize = 9;

/// What a step may feed: tokens, or µs at the measured cost.
#[derive(Clone, Copy)]
enum Budget {
    Tokens(usize),
    Micros(f64, f64),
}

/// How much prompt a step feeds while requests decode.
#[derive(Clone, Copy)]
pub struct ChunkPolicy {
    /// Tokens at least (0: no chunking).
    pub floor: usize,
    /// The step's target wall time in ms (`STEP_MS`; 0: `floor` tokens).
    pub step_ms: usize,
    /// A step's prompt tokens, pinned (0: from the cost model and
    /// `step_ms`): `--step-tokens`.
    pub step_tokens: usize,
    /// AXON_STEP_COST=legacy: the budget before `StepCost` (`step_ms`
    /// less the last decode step, at a cost per token with the pass's
    /// fixed part in it). With the default fifo admission (2 lanes at any
    /// slot count) main's scheduling exactly: the rollback, AXON_STEP_COST=
    /// legacy alone (without AXON_ADMIT=sjf).
    pub legacy: bool,
}

/// The admission order of waiting requests (`--admit-order`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmitOrder {
    /// Shortest uncached prompt first, aged requests (`admit_age_ms`)
    /// before all of them, several admissions per engine iteration.
    Sjf,
    /// The first waiting request that fits a slot, one per iteration, at
    /// most `lanes` prompts prefilling (the order before; the rollback).
    Fifo,
}

/// The scheduler's knobs (`axon serve` flags; defaults from the
/// environment for A/B jobs that can't change a command line).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sched {
    pub order: AdmitOrder,
    /// A waiting request this old goes before shorter ones (0: never).
    pub age_ms: u64,
    /// Prompts longer than a step's chunk prefilling at once at most (0:
    /// sjf `slots / 16` within 2..=8, fifo 2; `Batch::lanes`).
    pub lanes: usize,
    /// `ChunkPolicy::step_tokens`.
    pub step_tokens: usize,
    /// `ChunkPolicy::legacy`.
    pub legacy: bool,
}

/// `Sched::age_ms` by default: the showdown's TTFT SLO for a 7.6k prompt
/// is 2000 + 250 x 7.6 ms; 3 s of age plus about a second of chunked
/// prefill lands at its edge.
pub const ADMIT_AGE_MS: u64 = 3000;

impl Default for Sched {
    /// FIFO unless AXON_ADMIT=sjf (shortest-first missed its kill-rule
    /// clause on the bella cells: short TTFT p50 0.56x fifo's at 1.5B C=32,
    /// 0.79x on the showdown client, against <= 0.5x; it stays opt-in until
    /// the H100 cells decide it); the cost model unless AXON_STEP_COST=legacy.
    fn default() -> Sched {
        let admit = std::env::var("AXON_ADMIT").ok();
        let cost = std::env::var("AXON_STEP_COST").ok();
        Sched::from_env(admit.as_deref(), cost.as_deref())
    }
}

impl Sched {
    /// The defaults for AXON_ADMIT = `admit` and AXON_STEP_COST = `cost`
    /// (`Default` reads them; tests pass them). Note: "AXON_STEP_COST=legacy
    /// alone is main's scheduling" holds only without AXON_ADMIT=sjf.
    pub fn from_env(admit: Option<&str>, cost: Option<&str>) -> Sched {
        Sched {
            order: match admit == Some("sjf") {
                true => AdmitOrder::Sjf,
                false => AdmitOrder::Fifo,
            },
            age_ms: ADMIT_AGE_MS,
            lanes: 0,
            step_tokens: 0,
            legacy: cost == Some("legacy"),
        }
    }
}

/// Prompts prefilling at once at most for `sched` with `slots` slots
/// (`Batch::lanes`): `sched.lanes`, else sjf `slots / 16` within 2..=8 (2
/// at up to 32 slots, 8 at 128), fifo 2 (main's `MAX_PREFILLING`: the
/// lane scaling is part of shortest-first, opt-in with it).
pub(super) fn lanes_for(sched: &Sched, slots: usize) -> usize {
    match (sched.lanes, sched.order) {
        (0, AdmitOrder::Sjf) => (slots / 16).clamp(2, 8),
        (0, AdmitOrder::Fifo) => 2,
        (n, _) => n,
    }
}

/// What a step costs, learned from the steps that ran (`--step-ms` turns
/// it into tokens). A prompt token at depth d costs tok * (d + DEPTH_K)
/// µs on top of a pass's fixed part (launches and one read of the
/// weights), so eight short prompts in their own passes cost eight fixed
/// parts; a decode step's time is known per row count, for the step about
/// to run. Exponential moving averages; None until measured.
#[derive(Clone, Debug, Default)]
pub(super) struct StepCost {
    /// µs per prompt token per (depth + DEPTH_K), the fixed part out:
    /// two-pass feeds of at least LEARN_MIN tokens.
    tok: Option<f64>,
    /// A pass's fixed µs: feeds of fewer than PASS_FIXED_MAX tokens, each
    /// sample at most PASS_JUMP x the estimate and `pass_cap_us`; read
    /// through `pass` (bounded by the fastest decode step).
    pass_us: Option<f64>,
    /// The most a fixed-part sample may be (half a step: `Batch::new`
    /// sets it from `--step-ms`; 0: no cap).
    pass_cap_us: f64,
    /// Decode-only step ms per row count bucket (`bucket`).
    dec_ms: [Option<f64>; DEC_BUCKETS],
    /// The last decode-only step's ms.
    last_dec_ms: f64,
    /// The cost before StepCost (fixed part in, AXON_STEP_COST=legacy and
    /// idle steps): t / (n (d + n/2 + DEPTH_K)) over feeds of at least
    /// LEARN_MIN tokens.
    legacy_tok: Option<f64>,
}

/// The row count bucket of `rows` (`StepCost::dec_ms`): ceil(log2 rows).
fn bucket(rows: usize) -> usize {
    match rows {
        0 | 1 => 0,
        r => ((usize::BITS - (r - 1).leading_zeros()) as usize).min(DEC_BUCKETS - 1),
    }
}

/// Token units of `n` prompt tokens fed from depth `d`.
fn units(d: usize, n: usize) -> f64 {
    n as f64 * (d as f64 + n as f64 / 2.0 + DEPTH_K)
}

impl StepCost {
    /// The fastest decode-only step seen, µs: one pass over the weights
    /// with next to no token work, so about a pass's fixed part.
    fn pass_proxy(&self) -> Option<f64> {
        let min = self.dec_ms.iter().flatten().copied().reduce(f64::min);
        min.map(|ms| ms * 1e3)
    }

    /// A pass's fixed µs: the learned one, at most twice the fastest
    /// decode step (so an outlier the short feeds haven't corrected yet
    /// can't pin the budget: decode-only steps run every step); before a short feed taught it, the
    /// fastest decode step (a cold wave of short prompts is charged a pass
    /// each, not fed all in one step); 0 before either.
    fn pass(&self) -> f64 {
        match (self.pass_us, self.pass_proxy()) {
            (Some(p), Some(d)) => p.min(2.0 * d),
            (Some(p), None) => p,
            (None, Some(d)) => d,
            (None, None) => 0.0,
        }
    }

    /// A feed of `n` tokens from depth `d` took `us`.
    fn learn_feed(&mut self, d: usize, n: usize, us: f64) {
        let u = units(d, n);
        let ema = |a: Option<f64>, x: f64| Some(a.map_or(x, |a| 0.5 * a + 0.5 * x));
        if n >= LEARN_MIN {
            self.legacy_tok = ema(self.legacy_tok, us / u);
            self.tok = ema(self.tok, (us - self.pass()).max(0.25 * us) / u);
        } else if n < PASS_FIXED_MAX {
            let mut fixed = (us - self.tok.unwrap_or(0.0) * u).max(0.0);
            if let Some(p) = self.pass_us {
                fixed = fixed.min(PASS_JUMP * p);
            }
            if self.pass_cap_us > 0.0 {
                fixed = fixed.min(self.pass_cap_us);
            }
            self.pass_us = ema(self.pass_us, fixed);
        }
    }

    /// A decode-only step of `rows` rows took `ms`.
    fn learn_decode(&mut self, rows: usize, ms: f64) {
        let b = &mut self.dec_ms[bucket(rows)];
        *b = Some(b.map_or(ms, |a| 0.75 * a + 0.25 * ms));
        self.last_dec_ms = ms;
    }

    /// A decode step's ms at `rows` rows: its bucket's, else the nearest
    /// seen bucket's scaled by the square root of the row ratio, else the
    /// last decode step's. Past 256 rows: the last bucket's, unscaled
    /// (under-budgets a `--parallel` over 256; none is served today).
    fn dec(&self, rows: usize) -> f64 {
        let b = bucket(rows);
        if let Some(ms) = self.dec_ms[b] {
            return ms;
        }
        let near = (0..DEC_BUCKETS)
            .filter_map(|k| self.dec_ms[k].map(|ms| (k, ms)))
            .min_by_key(|&(k, _)| (k.abs_diff(b), k));
        match near {
            Some((k, ms)) => ms * (rows.max(1) as f64 / (1usize << k) as f64).sqrt(),
            None => self.last_dec_ms,
        }
    }
}

/// A two-pass step's budget (`Batch::budget`) for policy `p` with `rows`
/// requests decoding: everything without chunking; with nothing decoding
/// (nobody to stall) IDLE_MS worth (IDLE_CHUNK before a cost is known),
/// so requests arriving meanwhile still start soon; else `step_tokens`
/// if pinned, `floor` tokens without a time budget, or what fits in
/// `step_ms` beside the modelled decode step (SEED_CHUNK before a cost
/// is known), a pass's fixed part charged per prompt fed (the µs is
/// returned with it). Legacy: the last decode step and the old cost.
fn two_pass_budget(p: &ChunkPolicy, cost: &StepCost, rows: usize) -> (Budget, f64) {
    let (floor, ms) = (p.floor, p.step_ms);
    if floor == 0 {
        return (Budget::Tokens(usize::MAX), 0.0);
    }
    let idle = rows == 0;
    let old = cost.legacy_tok.filter(|&c| c > 0.0);
    if idle || p.legacy {
        let b = match (old, idle, ms) {
            (_, true, 0) | (None, true, _) => Budget::Tokens(IDLE_CHUNK.max(floor)),
            (Some(c), true, _) => Budget::Micros(IDLE_MS * 1e3, c),
            (_, false, 0) => Budget::Tokens(floor),
            (None, false, _) => Budget::Tokens(SEED_CHUNK.max(floor)),
            (Some(c), false, _) => Budget::Micros((ms as f64 - cost.last_dec_ms).max(0.0) * 1e3, c),
        };
        return (b, 0.0);
    }
    if p.step_tokens > 0 {
        return (Budget::Tokens(p.step_tokens), 0.0);
    }
    match (cost.tok.filter(|&c| c > 0.0), ms) {
        (_, 0) => (Budget::Tokens(floor), 0.0),
        (None, _) => (Budget::Tokens(SEED_CHUNK.max(floor)), 0.0),
        (Some(c), _) => {
            let left = (ms as f64 - cost.dec(rows)).max(0.0) * 1e3;
            (Budget::Micros(left, c), cost.pass())
        }
    }
}

/// A two-pass step's prompt tokens for one prompt at depth 0
/// (`two_pass_budget` less a pass's fixed part): `Batch::capacity`.
fn two_pass_capacity(p: &ChunkPolicy, cost: &StepCost, rows: usize) -> usize {
    match two_pass_budget(p, cost, rows) {
        (Budget::Tokens(t), _) => t,
        (Budget::Micros(us, c), pass) => {
            let us = (us - pass).max(0.0);
            ((DEPTH_K * DEPTH_K + 2.0 * us / c).sqrt() - DEPTH_K) as usize
        }
    }
}

/// `Batch::short_passes` for two-pass steps: the passes of a 64-token
/// prompt the µs budget fits (at least 1); a token budget: no bound.
fn two_pass_short_passes(p: &ChunkPolicy, cost: &StepCost, rows: usize) -> usize {
    match two_pass_budget(p, cost, rows) {
        (Budget::Micros(us, c), pass) => {
            let one = pass + c * units(0, PASS_FIXED_MAX);
            ((us / one) as usize).max(1)
        }
        (Budget::Tokens(_), _) => usize::MAX / 4,
    }
}

/// Prompts a step feeds at most (each is its own pass): as many as
/// `step_ms` fits beside the modelled decode step at a pass's fixed part
/// plus a 64-token chunk each (at least 1); nothing measured yet (the
/// first step of a fresh server): 1, which teaches a pass's cost. Without
/// it an idle or cold step (a token budget, no pass charge) fed a whole
/// wave of short prompts in one step and every one of them waited for all
/// their passes (bella 1025-153024-9998: wave-1 short TTFT 57-62 ms each
/// at C=16, 144-160 at C=32; 7B ~275 ms). No bound with legacy (main's
/// plan), `--step-ms 0`, `--step-tokens` or no chunking.
fn step_prompts(p: &ChunkPolicy, cost: &StepCost, rows: usize) -> usize {
    if p.legacy || p.floor == 0 || p.step_ms == 0 || p.step_tokens > 0 {
        return usize::MAX;
    }
    let pass = cost.pass();
    if pass <= 0.0 {
        return 1;
    }
    let one = pass + cost.tok.unwrap_or(0.0) * units(0, PASS_FIXED_MAX);
    let dec = if rows > 0 { cost.dec(rows) * 1e3 } else { 0.0 };
    (((p.step_ms as f64 * 1e3 - dec) / one) as usize).max(1)
}

/// Why a waiting request can't start now (`Batch::admit_check`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Admit {
    Yes,
    /// No free slot.
    Slot,
    /// Its class is full (`admit_gate`): another request may still start.
    Lane,
    /// Not enough KV pages.
    Pages,
}

/// A paged pool's admission by prompt class, before the pages test:
/// `want` the job's prompt tokens to feed, `pre_rem` the prompts being
/// prefilled's remainders, `short` a step's chunk tokens (its capacity:
/// `Batch::short_tokens`), `passes` the short prompts a step feeds at most
/// (one pass each, `Batch::short_passes`). Fifo: at most `lanes` prompts prefilling (the rule before,
/// `lanes` 2). Sjf: a prompt longer than `short` takes one of `lanes`
/// lanes; a short one (fed whole in one step) never waits for a lane,
/// only until the short prompts prefilling already fill about two steps,
/// by tokens and by passes (more only lengthens the youngest's wait:
/// `plan` feeds shortest remainder first).
fn admit_gate(
    order: AdmitOrder,
    lanes: usize,
    short: usize,
    passes: usize,
    pre_rem: &[usize],
    want: usize,
) -> bool {
    match order {
        AdmitOrder::Fifo => pre_rem.len() < lanes,
        AdmitOrder::Sjf => {
            let longs = pre_rem.iter().filter(|&&r| r > short).count();
            let shorts = pre_rem.iter().filter(|&&r| r <= short);
            let (n, toks) = shorts.fold((0, 0), |(n, t), &r| (n + 1, t + r));
            match want <= short {
                true => toks + want <= 2 * short && n < 2 * passes.max(1),
                false => longs < lanes,
            }
        }
    }
}

/// A waiting request's admission key (smallest first): aged ones (waited
/// at least `bound_ms`, if not 0) first, oldest first; then the shortest
/// `uncached` prompt (what admission would feed); then arrival (`id`).
pub(super) fn admit_key(
    uncached: usize,
    age_ms: u64,
    bound_ms: u64,
    id: u64,
) -> (bool, usize, u64) {
    let aged = bound_ms > 0 && age_ms >= bound_ms;
    (!aged, if aged { 0 } else { uncached }, id)
}

/// How a request ended.
enum End {
    Stop,
    Length,
    /// The client went away (cancelled, or a send failed).
    Gone,
}

/// The pool and the requests in it.
pub(super) struct Batch<'a, 't> {
    net: &'a Backend<'a>,
    pool: <Backend<'a> as BatchLm>::Pool,
    pub slots: usize,
    /// Positions per slot.
    pub ctx: usize,
    /// The tokens whose rows each slot holds (prefix reuse), from 0.
    held: Vec<Vec<u32>>,
    busy: Vec<bool>,
    /// When each slot was last taken (`uses` then; 0: never).
    used: Vec<u64>,
    uses: u64,
    /// The slot the S1 path runs in (`lend_solo`).
    solo: usize,
    pub seqs: Vec<Seq<'t>>,
    /// Requests being prefilled, oldest first.
    pre: Vec<Prefill<'t>>,
    policy: ChunkPolicy,
    /// Admission order and lanes (`Sched`; the serve loop reads the rest).
    pub sched: Sched,
    /// What steps cost (`StepCost`), learned as they run.
    cost: StepCost,
    /// Prompt tokens the step running fed (two-pass `feed`s; `step`).
    fed_now: usize,
    /// Most tokens of one of `feed_slot`'s passes (`BatchLm::pass_max`):
    /// an idle piece is a whole number of them (no extra partial passes).
    pass: usize,
    log: bool,
    generated: Arc<AtomicU64>,
    /// Requests preempted for pages so far (`preempt`).
    pub preempted: u64,
    /// Prompts being prefilled that gave their rows up for pages so far
    /// (`unfeed_youngest`; fed again from 0).
    pub unfed: u64,
    /// Generated tokens admission reserves pages for (`GROW_TOKENS`; tests
    /// lower it to force preemption).
    grow_tokens: usize,
    /// Where the slots' state is shown (/slots, /metrics; `publish`).
    pub metrics: Option<Arc<super::ops::Metrics>>,
    /// AXON_SERVE_TRACE's per-step log (`StepTrace`; None: off, the
    /// default, and then nothing below is measured).
    trace: Option<StepTrace>,
    /// The KV RAM tier (`--cache-ram`, tier.rs; None: off, the default,
    /// and then nothing of it runs).
    tier: Option<PoolTier>,
}

/// AXON_SERVE_TRACE=FILE: one JSON line per engine step appended to FILE
/// (a measurement aid, default off; it changes no output bits): `t_ms`
/// since the trace opened, the step's wall `ms`, its prompt feeds (`feeds`
/// passes, `fed` tokens, `feed_ms`), its decode pass (`rows`, mean
/// `depth`, `dec_ms`; null without one), then the requests `prefilling`
/// and `running`. Wall clock on the host: `feed_slot` and `step` return
/// once the GPU is done, so a pass's time is its kernels plus launches;
/// `ms` less both is the step's host work.
struct StepTrace {
    out: std::io::LineWriter<std::fs::File>,
    t0: Instant,
    /// This step's feeds so far: passes, tokens, ms.
    feeds: usize,
    fed: usize,
    feed_ms: f64,
    /// This step's decode pass: rows, the sum of their positions, ms.
    dec: Option<(usize, usize, f64)>,
}

impl StepTrace {
    /// The trace AXON_SERVE_TRACE names, if any (a file that won't open:
    /// said on stderr, no trace).
    fn from_env() -> Option<StepTrace> {
        let path = std::env::var_os("AXON_SERVE_TRACE").filter(|p| !p.is_empty())?;
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path);
        match f {
            Ok(f) => {
                eprintln!("axon serve: step trace to {}", path.to_string_lossy());
                Some(StepTrace {
                    out: std::io::LineWriter::new(f),
                    t0: Instant::now(),
                    feeds: 0,
                    fed: 0,
                    feed_ms: 0.0,
                    dec: None,
                })
            }
            Err(e) => {
                eprintln!(
                    "axon serve: AXON_SERVE_TRACE {}: {e}",
                    path.to_string_lossy()
                );
                None
            }
        }
    }

    /// The JSON line of a step that started `t_ms` after the trace opened
    /// and took `ms`, from what the step recorded.
    fn line(&self, t_ms: f64, ms: f64, prefilling: usize, running: usize) -> String {
        let dec = match self.dec {
            Some((rows, pos, ms)) => format!(
                "\"rows\":{rows},\"depth\":{:.1},\"dec_ms\":{ms:.3}",
                pos as f64 / rows.max(1) as f64
            ),
            None => "\"rows\":0,\"depth\":null,\"dec_ms\":null".into(),
        };
        format!(
            "{{\"t_ms\":{t_ms:.3},\"ms\":{ms:.3},\"feeds\":{},\"fed\":{},\"feed_ms\":{:.3},{dec},\"prefilling\":{prefilling},\"running\":{running}}}",
            self.feeds, self.fed, self.feed_ms
        )
    }
}

/// What gives up pages for running request `growing`'s next page
/// (`Batch::grow`): the youngest prompt being prefilled that holds rows and
/// is younger than it (an older one keeps its rows: unfeeding it for a
/// younger request would refeed it over and over, a priority inversion),
/// else the youngest running request (maybe `growing` itself), never the
/// oldest request of all (running or prefilling: it always progresses).
#[derive(Debug, PartialEq, Eq)]
enum Victim {
    /// Index into the prefills.
    Unfeed(usize),
    /// Index into the running requests.
    Preempt(usize),
    /// Nothing may give way (the oldest request alone is short: never, as a
    /// slot's cap is at most the pool's pages).
    Stuck,
}

/// `Victim` from the running requests' ids and the prefills' (id, holds rows).
fn victim(growing: u64, running: &[u64], prefills: &[(u64, bool)]) -> Victim {
    let unfeed = (prefills.iter().enumerate())
        .filter(|(_, (id, holds))| *holds && *id > growing)
        .max_by_key(|(_, (id, _))| *id);
    if let Some((j, _)) = unfeed {
        return Victim::Unfeed(j);
    }
    let oldest = running
        .iter()
        .chain(prefills.iter().map(|(id, _)| id))
        .min();
    match (0..running.len()).max_by_key(|&k| running[k]) {
        Some(y) if Some(&running[y]) != oldest => Victim::Preempt(y),
        _ => Victim::Stuck,
    }
}

/// Generated tokens a paged admission reserves pages for, per request (at
/// most its n_gen): without it a request admitted on its prompt's pages
/// alone gets preempted by the others' growth (a 15k prompt fed again:
/// MEASURED in 1018-044243-ceb5, 1 preemption a round at 7 x 15k + 600).
const GROW_TOKENS: usize = 1024;

// Prompts a paged pool prefills at once: `Batch::lanes` (was a constant 2:
// more of them share each step's chunk and finish together, late; 7 x 15k:
// mean TTFT 37 s vs 33 s fed in turn, ceb5). A plain pool's slots bound it.

/// Pages of positions 0..n.
fn pages_for(n: usize, size: usize) -> usize {
    n.div_ceil(size)
}

impl<'a, 't> Batch<'a, 't> {
    pub fn new(
        net: &'a Backend<'a>,
        pool: <Backend<'a> as BatchLm>::Pool,
        slots: usize,
        ctx: usize,
        policy: ChunkPolicy,
        log: bool,
        generated: Arc<AtomicU64>,
    ) -> Batch<'a, 't> {
        let pass = net.pass_max(&pool).max(1);
        Batch {
            net,
            pass,
            pool,
            slots,
            ctx,
            held: vec![Vec::new(); slots],
            busy: vec![false; slots],
            used: vec![0; slots],
            uses: 0,
            solo: 0,
            seqs: Vec::new(),
            pre: Vec::new(),
            policy,
            // From the environment (AXON_ADMIT, AXON_STEP_COST: `Sched`'s
            // Default); the server sets it from its flags after `new`.
            sched: Sched::default(),
            cost: StepCost {
                pass_cap_us: policy.step_ms as f64 * 0.5e3,
                ..StepCost::default()
            },
            fed_now: 0,
            log,
            generated,
            preempted: 0,
            unfed: 0,
            grow_tokens: GROW_TOKENS,
            metrics: None,
            trace: StepTrace::from_env(),
            tier: None,
        }
    }

    /// Turn the KV RAM tier on with `cap` bytes (`--cache-ram`), its
    /// page-locked memory taken as `how` says: its size and how it was
    /// taken, or None if it stays off (no page image on this backend, or
    /// less than a page); Err: the memory couldn't be had.
    pub fn set_cache_ram(
        &mut self,
        cap: u64,
        how: super::tier::Alloc,
    ) -> Result<Option<(usize, String)>, String> {
        let t = PoolTier::new(
            self.net,
            &self.pool,
            self.slots,
            cap,
            how,
            self.log,
            self.metrics.clone(),
        )?;
        self.tier = t;
        Ok(self.tier.as_ref().map(|t| (t.t.store.cap(), t.how())))
    }

    /// Slot `s` is about to give up its rows past `keep`: the tier stores
    /// its whole pages first (all of them: those below `keep` stay, the
    /// rest are its conversation's, and go to RAM instead of away), and
    /// forgets the stored pages past `keep`. A tier error turns the tier
    /// off (`tier_failed`), never the server.
    fn spill(&mut self, s: usize, keep: usize) {
        let Some(t) = self.tier.as_mut() else {
            return;
        };
        match t.write(self.net, &self.pool, s, &self.held[s], usize::MAX) {
            Ok(_) => t.t.truncate(s, keep / PAGE),
            Err(e) => self.tier_failed(&e),
        }
    }

    /// The tier failed (`e`): it is off from here, as the no-pool path's;
    /// the slots' rows are untouched (what it copied in is only trusted
    /// once a restore succeeded), so serving goes on as with
    /// `--cache-ram 0`.
    fn tier_failed(&mut self, e: &str) {
        eprintln!("axon serve: --cache-ram: {e}; the KV RAM tier is off from here");
        self.tier = None;
    }

    /// The tier's write-behind: `bytes` of copies of the slots' unstored
    /// pages (usize::MAX: all of them).
    pub fn write_behind(&mut self, bytes: usize) {
        let Some(t) = self.tier.as_mut() else {
            return;
        };
        if let Err(e) = t.write_some(self.net, &self.pool, &self.held, bytes) {
            self.tier_failed(&e);
        }
    }

    /// The engine loop's tick: the tier adopts the slabs its allocator
    /// thread has pinned (and shows them in /metrics).
    pub fn poll_tier(&mut self) {
        if let Some(t) = self.tier.as_mut() {
            t.poll(self.net);
        }
    }

    /// Positions of `prompt` (but its last token) that admission counts as
    /// cached: the free slot's `common`, or the tier's stored prefix if
    /// longer.
    fn cached_len(&self, prompt: &[u32], common: usize) -> usize {
        let last = prompt.len() - 1;
        let stored = match &self.tier {
            Some(t) if last > common => t.t.peek(&prompt[..last]),
            _ => 0,
        };
        common.max(stored).min(last)
    }

    /// Slot `slot` holds the first `have` positions of `prompt` (below its
    /// last token): copy in the rest of the longest prefix of it the tier
    /// stores (as far as the slot gets pages for it; idle prefixes give
    /// theirs up, as for a feed). The positions it holds after; `held`
    /// says so.
    fn restore(&mut self, slot: usize, prompt: &[u32], have: usize) -> Result<usize, String> {
        let Some(t) = self.tier.as_mut() else {
            return Ok(have);
        };
        let last = prompt.len() - 1;
        let pages = t.lookup(&prompt[..last]);
        let mut to = pages.len();
        if to * PAGE > have {
            // Pages for them (the looked-up pages are held: a slot spilled
            // to make room can't evict them).
            let short = self.make_room(slot, to * PAGE)?;
            if short > 0 {
                to = self.pages().map_or(0, |p| p.held[slot]).min(to);
            }
        }
        let t = self.tier.as_mut().expect("on");
        if let Err(e) = t.restore(self.net, &self.pool, slot, &pages, have, to) {
            // The slot keeps `have` (its rows past it are fed again).
            self.tier_failed(&e);
            return Ok(have);
        }
        if to * PAGE <= have {
            return Ok(have);
        }
        self.held[slot] = prompt[..to * PAGE].to_vec();
        Ok(to * PAGE)
    }

    /// Show the slots' state in `metrics`: every engine step, and before a
    /// request's last event, so that a client that has its whole response
    /// never sees its slot busy (#112).
    pub fn publish(&self) {
        let Some(m) = &self.metrics else {
            return;
        };
        let pages = self.pages().map(|p| super::ops::PagesView {
            free: p.free,
            total: p.total,
            size: p.size,
            preempted: self.preempted,
            unfed: self.unfed,
        });
        m.publish_pool(self.slot_states(), pages);
        let short = match self.sched.order {
            AdmitOrder::Fifo => 0,
            AdmitOrder::Sjf => self.short_tokens(),
        };
        let c = &self.cost;
        m.publish_sched(super::ops::SchedView {
            waiting: Vec::new(),
            lanes_busy: (self.pre.iter())
                .filter(|f| f.seq.hist.len() - 1 - f.fed > short)
                .count(),
            lanes: self.lanes(),
            tok_us: c.tok,
            pass_us: c.pass_us,
            dec_ms: c.dec_ms.to_vec(),
            legacy: self.policy.legacy,
        });
    }

    /// Tests: reserve pages for `n` generated tokens a request at admission
    /// (GROW_TOKENS by default; 0 forces the preemption path under pressure).
    pub fn set_grow_tokens(&mut self, n: usize) {
        self.grow_tokens = n;
    }

    pub fn idle(&self) -> bool {
        self.seqs.is_empty() && self.pre.is_empty()
    }

    /// Whether `job` can start now (`admit_check`).
    pub fn can_admit(&self, job: &Job) -> bool {
        self.admit_check(job) == Admit::Yes
    }

    /// Whether `job` can start now: a free slot, and (a paged pool) a
    /// place in its class (`admit_gate`: lanes for long prompts, about two
    /// steps' worth of short ones; fifo: `lanes` prompts prefilling at
    /// most), and pages for its prompt past the prefix the slot keeps and
    /// its first `GROW_TOKENS` generated tokens, beside what the prompts
    /// being prefilled and the running requests still need on the same
    /// terms; idle slots' prefixes count, they give theirs up (`reclaim`).
    /// Otherwise it waits (never a failure for pages).
    pub(super) fn admit_check(&self, job: &Job) -> Admit {
        let prompt = &job.prompt;
        let Some((slot, common)) = self.free_slot(prompt) else {
            return Admit::Slot;
        };
        let Some(pg) = self.pages() else {
            return Admit::Yes;
        };
        let last = prompt.len() - 1;
        let rem: Vec<usize> = (self.pre.iter())
            .map(|f| f.seq.hist.len() - 1 - f.fed)
            .collect();
        let (short, passes) = match self.sched.order {
            AdmitOrder::Fifo => (0, 0),
            AdmitOrder::Sjf => (self.short_tokens(), self.short_passes()),
        };
        if !admit_gate(
            self.sched.order,
            self.lanes(),
            short,
            passes,
            &rem,
            last - self.cached_len(prompt, common),
        ) {
            return Admit::Lane;
        }
        // Pages a sequence of `at` positions needs to grow `left` more.
        let grow = self.grow_tokens;
        let need = |at: usize, left: usize| pages_for((at + left.min(grow)).min(self.ctx), pg.size);
        let keep = pages_for(common.min(last), pg.size);
        let want = need(last, job.n_gen).saturating_sub(keep);
        let others: usize = (self.pre.iter().map(|f| &f.seq))
            .chain(&self.seqs)
            .map(|q| {
                let left = q.job.n_gen.saturating_sub(q.ids.len());
                need(q.hist.len(), left).saturating_sub(pg.held[q.slot])
            })
            .sum();
        let have = pg.free + self.idle_pages(&pg, slot) + pg.held[slot].saturating_sub(keep);
        match have >= want + others {
            true => Admit::Yes,
            false => Admit::Pages,
        }
    }

    /// Prompts prefilling at once at most (`lanes_for`).
    pub fn lanes(&self) -> usize {
        lanes_for(&self.sched, self.slots)
    }

    /// A short prompt's most tokens: what one step can feed (`capacity`),
    /// at least the floor.
    fn short_tokens(&self) -> usize {
        self.capacity().max(self.policy.floor)
    }

    /// Whether `job`'s prompt (past the free slot's prefix) is short.
    pub(super) fn is_short(&self, job: &Job) -> bool {
        let n = self.uncached(&job.prompt);
        n.is_some_and(|n| n <= self.short_tokens())
    }

    /// The prompt tokens a step can feed (its capacity), from the rows and
    /// the cost alone, never from the prompts already prefilling (so a
    /// prompt's class can't change as an admission round fills `pre`): the
    /// budget for one prompt at depth 0 less a pass's fixed part
    /// (everything when chunking is off).
    fn capacity(&self) -> usize {
        if self.policy.floor == 0 {
            return usize::MAX / 4;
        }
        two_pass_capacity(&self.policy, &self.cost, self.seqs.len())
    }

    /// Short prompts a step feeds at most: each in its own pass, as many
    /// as `step_ms` fits at a pass's fixed part plus a 64-token chunk each
    /// (at least 1); a token budget: no bound but the tokens.
    fn short_passes(&self) -> usize {
        let rows = self.seqs.len();
        let n = two_pass_short_passes(&self.policy, &self.cost, rows);
        n.min(step_prompts(&self.policy, &self.cost, rows))
    }

    /// The prompt tokens `admit` would feed for `prompt` (past the prefix
    /// the free slot it would get holds), None without a free slot.
    pub fn uncached(&self, prompt: &[u32]) -> Option<usize> {
        let last = prompt.len().checked_sub(1)?;
        self.free_slot(prompt)
            .map(|(_, c)| last - self.cached_len(prompt, c))
    }

    /// The pool's pages (None: plain).
    pub fn pages(&self) -> Option<PoolPages> {
        self.net.pages(&self.pool)
    }

    /// Pages idle slots other than `keep` hold (`reclaim`'s).
    fn idle_pages(&self, pg: &PoolPages, keep: usize) -> usize {
        (0..self.slots)
            .filter(|&s| s != keep && !self.busy[s])
            .map(|s| pg.held[s])
            .sum()
    }

    /// Free slots.
    pub fn free(&self) -> usize {
        self.busy.iter().filter(|b| !**b).count()
    }

    /// A free slot for `prompt` (`best_slot`).
    fn free_slot(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        best_slot(&self.held, &self.busy, &self.used, prompt)
    }

    /// Slot `s` is taken now (`used`).
    fn touch(&mut self, s: usize) {
        self.uses += 1;
        self.used[s] = self.uses;
    }

    /// Give `job` a free slot: its prompt (all but its last token, which is
    /// its first row) is fed a chunk per step from the next step on, past
    /// the rows the slot already holds; it decodes from the step that feeds
    /// its last chunk.
    pub fn admit(&mut self, job: Job, out: TextOut<'t>) -> Result<(), String> {
        let p = &job.prompt;
        let last = p.len() - 1;
        let (slot, common) = self.free_slot(p).expect("admit: a free slot");
        let reuse = common.min(last);
        self.busy[slot] = true;
        self.touch(slot);
        // The tier keeps the slot's pages past the prefix (its last
        // conversation's).
        self.spill(slot, reuse);
        self.held[slot].truncate(reuse);
        // The rest of the slot's pages are free (a paged pool).
        self.net.release(&mut self.pool, slot, reuse)?;
        let reuse = self.restore(slot, p, reuse)?;
        if super::debug() {
            eprintln!(
                "axon serve: #{} prefill: chunked, slot {slot}, {} tok ({reuse} cached), {} decoding",
                job.id,
                p.len(),
                self.seqs.len()
            );
        }
        let sampler = job.sample.map(Sampler::new);
        let now = Instant::now();
        self.pre.push(Prefill {
            seq: Seq {
                hist: p.clone(),
                ids: Vec::new(),
                sampler,
                out,
                cached: reuse,
                prompt_ms: 0.0,
                // Set when it starts decoding.
                t1: now,
                slot,
                job,
            },
            fed: reuse,
            starved: 0,
            t0: now,
        });
        Ok(())
    }

    /// What this step may feed (`two_pass_budget`), and the µs a pass's
    /// fixed part costs (charged per prompt fed).
    fn budget(&self) -> (Budget, f64) {
        two_pass_budget(&self.policy, &self.cost, self.seqs.len())
    }

    /// Feed this step's prompt (`budget`, split by `plan`); prompts then
    /// in their slots join the running requests.
    fn prefill(&mut self) -> Result<(), String> {
        if self.pre.is_empty() {
            return Ok(());
        }
        let (budget, pass_us) = self.budget();
        let rem: Vec<usize> = self
            .pre
            .iter()
            .map(|f| f.seq.hist.len() - 1 - f.fed)
            .collect();
        let fed: Vec<usize> = self.pre.iter().map(|f| f.fed).collect();
        let starved: Vec<usize> = self.pre.iter().map(|f| f.starved).collect();
        let idle = self.seqs.is_empty();
        let most = step_prompts(&self.policy, &self.cost, self.seqs.len());
        let plan = plan_charged(
            budget,
            pass_us,
            most,
            self.policy.floor,
            self.pass,
            &fed,
            &rem,
            &starved,
            idle,
        );
        for (f, s) in self.pre.iter_mut().zip(starve(&plan, &rem, &starved, idle)) {
            f.starved = s;
        }
        let mut moved = false;
        for (i, &n) in plan.iter().enumerate() {
            if n > 0 {
                moved |= self.feed(i, n)? > 0;
            }
        }
        // Nothing decodes and no prompt has the pages to go on: the
        // youngest ones give theirs up, the oldest goes first.
        if !moved && idle && plan.iter().any(|&n| n > 0) {
            let oldest = (0..self.pre.len()).min_by_key(|&i| self.pre[i].seq.job.id);
            if let Some(o) = oldest {
                self.pre[o].starved = STARVE;
                self.unfeed_youngest(Some(self.pre[o].seq.job.id))?;
            }
        }
        let mut i = 0;
        while i < self.pre.len() {
            let f = &self.pre[i];
            if f.fed == f.seq.hist.len() - 1 {
                let mut f = self.pre.remove(i);
                // A preempted request (`preempt`) keeps its first timings.
                if f.seq.ids.is_empty() {
                    f.seq.prompt_ms = f.t0.elapsed().as_secs_f64() * 1e3;
                    f.seq.t1 = Instant::now();
                }
                self.seqs.push(f.seq);
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Feed the next `n` tokens of prefill `i`'s prompt into its slot (a
    /// paged pool: as many as there are pages for, beside a page for each
    /// running request's growth; none: it waits, decode first).
    fn feed(&mut self, i: usize, n: usize) -> Result<usize, String> {
        let n = self.feedable(i, n)?;
        if n == 0 {
            return Ok(0);
        }
        let f = &mut self.pre[i];
        let s = f.seq.slot;
        let toks = &f.seq.hist[f.fed..f.fed + n];
        let t = Instant::now();
        self.net.feed_slot(&mut self.pool, s, toks, f.fed)?;
        let us = t.elapsed().as_secs_f64() * 1e6;
        // Long feeds teach the cost per token, short ones a pass's fixed part.
        self.cost.learn_feed(f.fed, n, us);
        if let Some(tr) = self.trace.as_mut() {
            tr.feeds += 1;
            tr.fed += n;
            tr.feed_ms += us / 1e3;
        }
        self.held[s].extend_from_slice(toks);
        f.fed += n;
        self.fed_now += n;
        Ok(n)
    }

    /// How many of the next `n` tokens of prefill `i` have pages (taken
    /// now): all of them, else up to the end of the pages that are free
    /// once idle prefixes are reclaimed, less a page per running request.
    fn feedable(&mut self, i: usize, n: usize) -> Result<usize, String> {
        let (slot, fed) = (self.pre[i].seq.slot, self.pre[i].fed);
        if self.pages().is_none() || self.make_room(slot, fed + n)? == 0 {
            return Ok(n);
        }
        let pg = self.pages().expect("paged");
        let spare = pg.free.saturating_sub(self.seqs.len());
        let fit = ((pg.held[slot] + spare) * pg.size)
            .saturating_sub(fed)
            .min(n);
        if fit > 0 {
            let short = self.net.reserve(&mut self.pool, slot, fed + fit)?;
            debug_assert_eq!(short, 0);
        }
        Ok(fit)
    }

    /// A slot for the S1 path to run `prompt` in, chosen as `admit`
    /// chooses (`best_slot`: the longest prefix of it, else the least
    /// recently used), `cache` (a `slot_cache` of the pool) pointed at it,
    /// and the tokens the slot holds (the engine reuses as many as the
    /// prompt shares); the slot holds nothing meanwhile. The batch is idle.
    /// A paged pool: the slot holds pages for every position the request
    /// may write (`n_gen` past its prompt, at most a slot's ctx), since the
    /// `Lm` calls on the view take none; idle slots' prefixes give theirs
    /// up for them, least recently used first.
    pub fn lend_solo(
        &mut self,
        prompt: &[u32],
        n_gen: usize,
        cache: &mut <Backend<'a> as crate::lm::Lm>::Cache,
    ) -> Result<Vec<u32>, String> {
        assert!(self.idle(), "lend_solo: the batch is running");
        let (slot, common) = self.free_slot(prompt).expect("lend_solo: a free slot");
        self.net.move_slot_cache(&self.pool, cache, slot)?;
        self.solo = slot;
        self.touch(slot);
        // The request writes over the slot's rows past the prefix it
        // shares: the tier keeps them first.
        let have = common.min(prompt.len() - 1);
        if self.tier.is_some() {
            self.spill(slot, have);
            self.held[slot].truncate(have);
        }
        let want = prompt.len().saturating_add(n_gen).min(self.ctx);
        let short = self.make_room(slot, want)?;
        if short > 0 {
            return Err(format!("{short} KV pages short for a request alone"));
        }
        // Its pages are there: the stored prefix of its prompt is copied
        // in, the view prefills the rest.
        self.restore(slot, prompt, have)?;
        // Its batched passes run on the pool's scratch (a paged pool's view
        // has none of its own), back at `end_solo` / `adopt`.
        self.net.lend_scratch(&mut self.pool, cache)?;
        Ok(std::mem::take(&mut self.held[slot]))
    }

    /// Free pages for slot `slot`'s positions 0..hi: idle prefixes give
    /// theirs up as needed (`reclaim`). The pages still short (0: held).
    fn make_room(&mut self, slot: usize, hi: usize) -> Result<usize, String> {
        loop {
            let short = self.net.reserve(&mut self.pool, slot, hi)?;
            if short == 0 || !self.reclaim(slot)? {
                return Ok(short);
            }
        }
    }

    /// The least recently used idle slot (not `keep`) holding pages gives
    /// them all up (its prefix forgotten): whether there was one.
    fn reclaim(&mut self, keep: usize) -> Result<bool, String> {
        let Some(pages) = self.net.pages(&self.pool) else {
            return Ok(false);
        };
        let victim = (0..self.slots)
            .filter(|&s| s != keep && !self.busy[s] && pages.held[s] > 0)
            .min_by_key(|&s| self.used[s]);
        let Some(s) = victim else {
            return Ok(false);
        };
        // Its prefix goes to RAM (the tier on), not away.
        self.spill(s, 0);
        self.held[s].clear();
        self.net.release(&mut self.pool, s, 0)?;
        Ok(true)
    }

    /// The slot the S1 path runs in (`lend_solo`).
    pub fn solo_slot(&self) -> usize {
        self.solo
    }

    /// The S1 path ended (without a hand-over): its slot holds the rows of
    /// `hist`.
    pub fn end_solo(
        &mut self,
        hist: Vec<u32>,
        cache: &mut <Backend<'a> as crate::lm::Lm>::Cache,
    ) -> Result<(), String> {
        self.net.return_scratch(&mut self.pool, cache)?;
        // Pages past its rows (reserved for what it didn't generate) free.
        self.net.release(&mut self.pool, self.solo, hist.len())?;
        self.held[self.solo] = hist;
        // The batch is idle: the tier takes the request's pages now.
        self.spill(self.solo, usize::MAX);
        Ok(())
    }

    /// Take over a request `generate` was running on `cache` (the S1 path,
    /// in the slot `lend_solo` gave it, so its rows 0..hist.len() - 1 are
    /// already there and `BatchLm::adopt` copies nothing).
    pub fn adopt(
        &mut self,
        mut seq: Seq<'t>,
        cache: &mut <Backend<'a> as crate::lm::Lm>::Cache,
    ) -> Result<(), String> {
        self.net.return_scratch(&mut self.pool, cache)?;
        let n = seq.hist.len() - 1;
        let slot = self.solo;
        assert!(!self.busy[slot], "adopt: the S1 slot is busy");
        self.net.adopt(&mut self.pool, slot, cache, n)?;
        // Its pages past its rows free: the steps take each next one.
        self.net.release(&mut self.pool, slot, n)?;
        self.busy[slot] = true;
        self.held[slot] = seq.hist[..n].to_vec();
        seq.slot = slot;
        self.seqs.push(seq);
        Ok(())
    }

    /// Drop requests whose client went away (their slots freed).
    pub fn reap(&mut self) {
        let mut i = 0;
        while i < self.pre.len() {
            if self.pre[i].seq.job.cancel.load(Ordering::Relaxed) {
                let f = self.pre.remove(i);
                self.finish(f.seq, End::Gone);
            } else {
                i += 1;
            }
        }
        i = 0;
        while i < self.seqs.len() {
            if self.seqs[i].job.cancel.load(Ordering::Relaxed) {
                let s = self.seqs.swap_remove(i);
                self.finish(s, End::Gone);
            } else {
                i += 1;
            }
        }
    }

    /// One step: a chunk of the prompts being prefilled, then a row per
    /// running request and each one's next token. With a `StepTrace`, a
    /// line about it once it did any work.
    pub fn step(&mut self) -> Result<(), String> {
        self.step_traced()?;
        // The tier's write-behind: a budget while requests run, the rest
        // once none do.
        let bytes = match self.idle() {
            true => usize::MAX,
            false => super::tier::STEP_BYTES,
        };
        self.write_behind(bytes);
        Ok(())
    }

    fn step_traced(&mut self) -> Result<(), String> {
        if self.trace.is_none() {
            return self.step_inner();
        }
        let t = Instant::now();
        let res = self.step_inner();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let (prefilling, running) = (self.pre.len(), self.seqs.len());
        let tr = self.trace.as_mut().expect("tracing");
        if tr.feeds > 0 || tr.dec.is_some() {
            let t_ms = (t - tr.t0).as_secs_f64() * 1e3;
            let line = tr.line(t_ms, ms, prefilling, running);
            use std::io::Write;
            if let Err(e) = writeln!(tr.out, "{line}") {
                eprintln!("axon serve: step trace: {e}; tracing stops");
                self.trace = None;
            }
        }
        if let Some(tr) = self.trace.as_mut() {
            (tr.feeds, tr.fed, tr.feed_ms, tr.dec) = (0, 0, 0.0, None);
        }
        res
    }

    /// `step`'s work.
    fn step_inner(&mut self) -> Result<(), String> {
        let t_step = Instant::now();
        self.fed_now = 0;
        self.reap();
        self.prefill()?;
        self.grow()?;
        if self.seqs.is_empty() {
            return Ok(());
        }
        let specs: Vec<_> = self
            .seqs
            .iter()
            .map(|s| s.sampler.as_ref().map(|x| x.cfg.spec()))
            .collect();
        let rows: Vec<SeqRow> = self
            .seqs
            .iter()
            .zip(&specs)
            .map(|(s, spec)| {
                let h = &s.hist;
                SeqRow {
                    token: h[h.len() - 1],
                    pos: h.len() - 1,
                    slot: s.slot,
                    head: match spec {
                        None => RowHead::Argmax,
                        // The penalty window: the tail of the history (as
                        // generate's forward_cands takes it).
                        Some(spec) => RowHead::Sample {
                            spec: *spec,
                            recent: &h[h.len() - spec.last_n.clamp(1, h.len())..],
                        },
                    },
                }
            })
            .collect();
        let t = Instant::now();
        let res = self.net.step(&mut self.pool, &rows);
        let dec_ms = t.elapsed().as_secs_f64() * 1e3;
        self.cost.learn_decode(rows.len(), dec_ms);
        if let Some(tr) = self.trace.as_mut() {
            let pos = rows.iter().map(|r| r.pos).sum();
            tr.dec = Some((rows.len(), pos, dec_ms));
        }
        // A step that fed prompt beside running rows: what `--step-ms`
        // aims at (/metrics `axon_step_seconds`).
        if let Some(m) = (self.metrics.as_ref()).filter(|_| self.fed_now > 0) {
            m.observe_step(t_step.elapsed(), rows.len(), self.fed_now);
        }
        drop(rows);
        let outs = match res {
            Ok(o) => o,
            // Refused before any work: those requests fail, the rest of the
            // server goes on.
            Err(StepError::Input(e)) => {
                eprintln!("axon serve: a step was refused: {e}");
                self.fail_all(&e);
                return Ok(());
            }
            Err(StepError::Device(e)) => return Err(e),
        };
        let mut done = Vec::new();
        for (i, o) in outs.into_iter().enumerate() {
            let s = &mut self.seqs[i];
            let fed = s.hist.len();
            self.held[s.slot].push(s.hist[fed - 1]);
            let next = match (o, s.sampler.as_mut()) {
                (RowOut::Pick(t), None) => t,
                (RowOut::Cands(c), Some(sm)) => sm.pick(&c),
                _ => unreachable!("a row's result is its head's"),
            };
            if s.job.stop_ids.contains(&next) {
                done.push((i, End::Stop));
                continue;
            }
            s.ids.push(next);
            s.hist.push(next);
            self.generated.fetch_add(1, Ordering::Relaxed);
            if let Some(end) = emit(s, next) {
                done.push((i, end));
            } else if s.ids.len() >= s.job.n_gen {
                done.push((i, End::Length));
            }
        }
        // Highest index first: swap_remove keeps the lower ones in place.
        done.sort_by_key(|d| std::cmp::Reverse(d.0));
        for (i, end) in done {
            let s = self.seqs.swap_remove(i);
            self.finish(s, end);
        }
        Ok(())
    }

    /// A paged pool: every running request's next row has its page, oldest
    /// request first (so `step_rows` never refuses a step for pages). Short
    /// of pages: idle prefixes go (`reclaim`), then `victim`: a younger
    /// prompt being prefilled gives its rows up (fed again later: the same
    /// rows), else the youngest running request is preempted (`preempt`:
    /// its prompt and tokens fed again, then it goes on, the same bits).
    /// The oldest request of all never gives way, so the server always
    /// makes progress (a slot's cap is at most the pool's pages).
    fn grow(&mut self) -> Result<(), String> {
        if self.pages().is_none() {
            return Ok(());
        }
        let mut ids: Vec<u64> = self.seqs.iter().map(|s| s.job.id).collect();
        ids.sort_unstable();
        for id in ids {
            while let Some(i) = self.seqs.iter().position(|s| s.job.id == id) {
                let (slot, hi) = (self.seqs[i].slot, self.seqs[i].hist.len());
                if self.make_room(slot, hi)? == 0 {
                    break;
                }
                let running: Vec<u64> = self.seqs.iter().map(|s| s.job.id).collect();
                let prefills: Vec<(u64, bool)> =
                    self.pre.iter().map(|f| (f.seq.job.id, f.fed > 0)).collect();
                match victim(id, &running, &prefills) {
                    Victim::Unfeed(j) => self.unfeed(j)?,
                    Victim::Preempt(y) => self.preempt(y)?,
                    Victim::Stuck => return Err("out of KV pages for the oldest request".into()),
                }
            }
        }
        Ok(())
    }

    /// The youngest prompt being prefilled that holds rows (not `keep`, a
    /// request id) gives them up: it is fed from 0 again later (the same
    /// rows). Whether there was one.
    fn unfeed_youngest(&mut self, keep: Option<u64>) -> Result<bool, String> {
        let young = (self.pre.iter().enumerate())
            .filter(|(_, f)| f.fed > 0 && Some(f.seq.job.id) != keep)
            .max_by_key(|(_, f)| f.seq.job.id)
            .map(|(j, _)| j);
        let Some(j) = young else {
            return Ok(false);
        };
        self.unfeed(j)?;
        Ok(true)
    }

    /// Prefill `j` gives its rows up: fed from 0 again later.
    fn unfeed(&mut self, j: usize) -> Result<(), String> {
        let s = self.pre[j].seq.slot;
        self.spill(s, 0);
        let f = &mut self.pre[j];
        f.fed = 0;
        f.seq.cached = 0;
        self.held[s].clear();
        self.net.release(&mut self.pool, s, 0)?;
        self.unfed += 1;
        Ok(())
    }

    /// Running request `i` gives up its pages and goes back to the
    /// prompts being prefilled: its prompt and tokens so far are fed again
    /// (cache reuse == fresh prefill), then it decodes on from where it
    /// was, its sampler as it was.
    fn preempt(&mut self, i: usize) -> Result<(), String> {
        let mut seq = self.seqs.swap_remove(i);
        // Its rows are fed again from 0: none of them is a cached prefix.
        seq.cached = 0;
        let s = seq.slot;
        if self.log {
            eprintln!(
                "axon serve: #{} preempted after {} tokens (out of KV pages): fed again later",
                seq.job.id,
                seq.ids.len()
            );
        }
        self.spill(s, 0);
        self.held[s].clear();
        self.net.release(&mut self.pool, s, 0)?;
        let at = self.pre.partition_point(|f| f.seq.job.id < seq.job.id);
        self.pre.insert(
            at,
            Prefill {
                seq,
                fed: 0,
                starved: STARVE,
                t0: Instant::now(),
            },
        );
        self.preempted += 1;
        Ok(())
    }

    /// Each slot's state, for GET /slots and /metrics (read only).
    pub fn slot_states(&self) -> Vec<super::ops::SlotState> {
        let pg = self.pages();
        let mut v: Vec<super::ops::SlotState> = (0..self.slots)
            .map(|s| super::ops::SlotState {
                held: self.held[s].len(),
                busy: self.busy[s],
                request: None,
                pages: pg.as_ref().map(|p| p.held[s]),
            })
            .collect();
        for s in self.pre.iter().map(|f| &f.seq).chain(&self.seqs) {
            v[s.slot].request = Some((s.job.id, s.job.arrived));
        }
        v
    }

    /// Every running request gets error `e` and leaves (its slot freed).
    pub fn fail_all(&mut self, e: &str) {
        let pre = self.pre.drain(..).map(|f| f.seq);
        let failed: Vec<_> = pre.chain(self.seqs.drain(..)).collect();
        for s in &failed {
            self.busy[s.slot] = false;
        }
        self.publish();
        for s in failed {
            let _ = s.job.tx.send(Event::Error(e.to_string()));
        }
    }

    fn finish(&mut self, mut s: Seq<'t>, end: End) {
        self.busy[s.slot] = false;
        let decode_ms = s.t1.elapsed().as_secs_f64() * 1e3;
        let finish = match end {
            End::Gone => {
                if self.log {
                    eprintln!(
                        "axon serve: #{} cancelled after {} tokens",
                        s.job.id,
                        s.ids.len()
                    );
                }
                return;
            }
            End::Stop => "stop",
            End::Length => "length",
        };
        let (content, reasoning) = s.out.finish();
        if !content.is_empty() || !reasoning.is_empty() {
            let _ = s.job.tx.send(Event::Text { content, reasoning });
        }
        let p = s.job.prompt.len();
        let usage = Usage {
            prompt_tokens: p,
            cached_tokens: s.cached,
            completion_tokens: s.ids.len(),
            prompt_ms: s.prompt_ms,
            decode_ms,
        };
        if self.log {
            eprintln!(
                "axon serve: #{} prompt {} tok ({} cached) in {:.1} ms, {} tok in {:.1} ms \
                 ({:.1} tok/s), {finish}, slot {} of {} running",
                s.job.id,
                p,
                s.cached,
                s.prompt_ms,
                s.ids.len(),
                decode_ms,
                if decode_ms > 0.0 {
                    s.ids.len() as f64 * 1e3 / decode_ms
                } else {
                    0.0
                },
                s.slot,
                self.seqs.len() + 1
            );
        }
        self.publish();
        let _ = s.job.tx.send(Event::Done { usage, finish });
    }
}

/// One step's prefill plan: the tokens to feed each prompt being
/// prefilled (`fed` rows in its slot, `rem` still to feed, passed over
/// `starved` steps in a row; all in arrival order). Prompts passed over
/// STARVE steps in a row go first, oldest first; then the others, shortest
/// remainder first. Each gets what is left of `budget`: tokens, or what the
/// time left buys at its own depth (n tokens from depth d cost
/// c n (d + n/2 + DEPTH_K)), the first at least `floor`. A remainder at
/// most half a floor past its share is fed whole (no last sliver of a pass
/// of its own); a later prompt's share under `floor` is skipped unless it
/// finishes the prompt (no sliver of a long one beside another's chunk).
/// On an `idle` step (nothing decodes) the first's time share rounds up to
/// whole passes of `pass` tokens and takes the step: prompts go one after
/// another. (The engine calls `plan_charged`; this is it with no pass
/// charge, which the plan tests pin.)
#[cfg(test)]
fn plan(
    budget: Budget,
    floor: usize,
    pass: usize,
    fed: &[usize],
    rem: &[usize],
    starved: &[usize],
    idle: bool,
) -> Vec<usize> {
    plan_charged(
        budget,
        0.0,
        usize::MAX,
        floor,
        pass,
        fed,
        rem,
        starved,
        idle,
    )
}

/// `plan`, at most `most` prompts fed (`step_prompts`), each prompt fed on a µs budget also charged `pass_us`, a
/// pass's fixed part (two-pass steps feed each prompt in its own pass):
/// its share is what the time left after that buys, and a later prompt
/// is fed only if all of it fits (no half-floor slack), so a step of
/// short prompts feeds as many as their passes fit, not all of them.
#[allow(clippy::too_many_arguments)]
fn plan_charged(
    budget: Budget,
    pass_us: f64,
    most: usize,
    floor: usize,
    pass: usize,
    fed: &[usize],
    rem: &[usize],
    starved: &[usize],
    idle: bool,
) -> Vec<usize> {
    let mut plan = vec![0; rem.len()];
    let mut order: Vec<usize> = (0..rem.len()).collect();
    order.sort_by_key(|&i| {
        if starved[i] >= STARVE {
            (false, 0, i)
        } else {
            (true, rem[i], i)
        }
    });
    let mut budget = budget;
    let mut first = true;
    let mut n_fed = 0;
    for i in order {
        if n_fed >= most {
            break;
        }
        let share = match budget {
            Budget::Tokens(t) => t,
            Budget::Micros(us, c) => {
                let dk = fed[i] as f64 + DEPTH_K;
                let us = (us - pass_us).max(0.0);
                let n = ((dk * dk + 2.0 * us / c).sqrt() - dk) as usize;
                match (first, idle) {
                    (true, true) => n.max(floor).div_ceil(pass).saturating_mul(pass),
                    (true, false) => n.max(floor),
                    _ => n,
                }
            }
        };
        let want = rem[i];
        let slack = match pass_us > 0.0 && !first {
            true => 0,
            false => floor / 2,
        };
        let n = if want <= share.saturating_add(slack) {
            want
        } else if first || share >= floor {
            share
        } else {
            0
        };
        if n == 0 || share == 0 {
            continue;
        }
        first = false;
        n_fed += 1;
        plan[i] = n;
        budget = match budget {
            Budget::Tokens(t) => Budget::Tokens(t.saturating_sub(n)),
            Budget::Micros(us, c) => {
                let spent = pass_us + c * n as f64 * (fed[i] as f64 + n as f64 / 2.0 + DEPTH_K);
                Budget::Micros((us - spent).max(0.0), c)
            }
        };
    }
    plan
}

/// Each prompt's misses after `plan` (`rem`: what each still had to feed
/// before it). While requests decode: reset when fed, else one more. On an
/// idle step a prompt counts a miss only when a younger one finished in
/// that step, and a feed doesn't reset the count. So a younger prompt
/// that took the head (shortest first) runs to its end, two long prompts
/// go one after another with no misses, a stream of short prompts that
/// each finish within their step holds a long one back STARVE of them at
/// most, and a prompt that reached STARVE keeps the head to its end.
fn starve(plan: &[usize], rem: &[usize], starved: &[usize], idle: bool) -> Vec<usize> {
    let youngest_done = (0..plan.len())
        .rev()
        .find(|&j| plan[j] > 0 && plan[j] == rem[j]);
    plan.iter()
        .zip(starved)
        .enumerate()
        .map(|(i, (&n, &s))| match (n > 0, idle) {
            (true, false) => 0,
            (false, false) => s + 1,
            (true, true) => s,
            (false, true) if youngest_done.is_some_and(|j| j > i) => s + 1,
            (false, true) => s,
        })
        .collect()
}

/// Send `id`'s text; Some(end) if that ended the request (a stop string,
/// or the client is gone).
fn emit(s: &mut Seq, id: u32) -> Option<End> {
    if s.job.cancel.load(Ordering::Relaxed) {
        return Some(End::Gone);
    }
    let (content, reasoning, hit) = s.out.push(id);
    if (!content.is_empty() || !reasoning.is_empty())
        && s.job.tx.send(Event::Text { content, reasoning }).is_err()
    {
        return Some(End::Gone);
    }
    hit.then_some(End::Stop)
}

/// The free slot (not `busy`) for `prompt`, and how many of its `held`
/// tokens are a prefix of `prompt`. The slot with the longest such prefix,
/// if that prefix counts (`counts`); else, or on a tie, the least recently
/// `used` slot (then the lowest): a request, batched or alone, evicts the
/// conversation idle the longest, as one cache per conversation would.
fn best_slot(
    held: &[Vec<u32>],
    busy: &[bool],
    used: &[u64],
    prompt: &[u32],
) -> Option<(usize, usize)> {
    (0..held.len())
        .filter(|&s| !busy[s])
        .map(|s| {
            let n = held[s]
                .iter()
                .zip(prompt)
                .take_while(|(a, b)| a == b)
                .count();
            (s, n)
        })
        .max_by_key(|&(s, n)| {
            let n = if counts(n, held[s].len(), prompt.len()) {
                n
            } else {
                0
            };
            (n, std::cmp::Reverse((used[s], s)))
        })
}

/// Whether a shared prefix of `n` tokens makes a slot holding `held` tokens
/// the place for a prompt of `prompt` tokens: it is at least half of what
/// the slot holds (the request continues that conversation), or more than
/// half of the prompt (llama-server's `--slot-prompt-similarity` 0.5). A
/// <bos> or a system prompt every chat starts with is neither, and doesn't
/// evict a conversation when another slot is free.
fn counts(n: usize, held: usize, prompt: usize) -> bool {
    n > 0 && (2 * n >= held || 2 * n > prompt)
}

#[cfg(test)]
mod slot_tests {
    use super::best_slot;

    /// The longest shared prefix wins; on a tie (no match at all, the
    /// usual case for a new conversation) the least recently used free
    /// slot, then the lowest; busy slots never.
    #[test]
    fn best_slot_prefers_prefix_then_least_recent() {
        let held = vec![vec![1, 2, 3], vec![9], vec![], vec![1, 2]];
        let free = [false; 4];
        let used = [4, 1, 3, 2];
        assert_eq!(best_slot(&held, &free, &used, &[1, 2, 3, 4]), Some((0, 3)));
        // Slots 0 and 3 share 2 tokens: 3 was used longer ago.
        assert_eq!(best_slot(&held, &free, &used, &[1, 2, 7]), Some((3, 2)));
        assert_eq!(best_slot(&held, &free, &used, &[5, 6]), Some((1, 0)));
        assert_eq!(best_slot(&held, &free, &used, &[]), Some((1, 0)));
        assert_eq!(best_slot(&held, &free, &[0; 4], &[5]), Some((0, 0)));
        let busy = [false, true, false, true];
        assert_eq!(best_slot(&held, &busy, &used, &[5]), Some((2, 0)));
        assert_eq!(best_slot(&held, &[true; 4], &used, &[1]), None);
    }

    /// A short shared prefix (a <bos>, a common system prompt) doesn't pull
    /// a request into a conversation's slot: the least recently used one
    /// takes it, with what it shares (its reuse) reported.
    #[test]
    fn best_slot_ignores_a_trivial_prefix() {
        let conv: Vec<u32> = (1..=100).collect();
        let held = vec![conv.clone(), vec![]];
        let free = [false; 2];
        // <bos> + "Hello": 1 of 100 held, 1 of 2 prompt tokens.
        assert_eq!(best_slot(&held, &free, &[2, 0], &[1, 500]), Some((1, 0)));
        // A system prompt of 20 shared, a new 60-token chat after it.
        let mut chat = conv[..20].to_vec();
        chat.extend(1000..1040);
        assert_eq!(best_slot(&held, &free, &[2, 0], &chat), Some((1, 0)));
        // The conversation continued: its slot.
        let mut more = conv.clone();
        more.extend([7, 8, 9]);
        assert_eq!(best_slot(&held, &free, &[2, 0], &more), Some((0, 100)));
        // Most of a short prompt is the slot's prefix (60 of 100 held, 60 of
        // 70 prompt tokens): its slot.
        let mut part = conv[..60].to_vec();
        part.extend(2000..2010);
        assert_eq!(best_slot(&held, &free, &[2, 0], &part), Some((0, 60)));
        // Every slot busy but the conversation's: it takes what it shares.
        assert_eq!(
            best_slot(&held, &[false, true], &[2, 0], &[1, 500]),
            Some((0, 1))
        );
    }
}

#[cfg(test)]
mod plan_tests {
    use super::{Budget, DEPTH_K, STARVE, plan, starve};

    /// The GPU's prefill pass (cuda PREFILL_BATCH).
    const PASS: usize = 1152;

    /// µs that `n` tokens from depth `d` cost at `c` (the model `plan` uses).
    fn cost(c: f64, d: usize, n: usize) -> f64 {
        c * n as f64 * (d as f64 + n as f64 / 2.0 + DEPTH_K)
    }

    #[test]
    fn tokens_shortest_first_and_no_sliver() {
        // A short prompt goes first; the 212 tokens left would be a sliver
        // of the long one (under the floor): skipped this step.
        assert_eq!(
            plan(
                Budget::Tokens(512),
                256,
                PASS,
                &[0, 0],
                &[10_000, 300],
                &[0, 0],
                false
            ),
            [0, 300]
        );
        // Left with at least a floor, the long one gets it.
        assert_eq!(
            plan(
                Budget::Tokens(1024),
                256,
                PASS,
                &[0, 0],
                &[10_000, 300],
                &[0, 0],
                false
            ),
            [724, 300]
        );
        // A remainder within half a floor past the share is fed whole.
        assert_eq!(
            plan(Budget::Tokens(512), 256, PASS, &[0], &[600], &[0], false),
            [600]
        );
        assert_eq!(
            plan(Budget::Tokens(512), 256, PASS, &[0], &[700], &[0], false),
            [512]
        );
    }

    #[test]
    fn micros_shares_follow_each_prompts_depth() {
        let c = 0.005;
        let us = 50_000.0;
        let shallow = plan(
            Budget::Micros(us, c),
            16,
            PASS,
            &[0],
            &[100_000],
            &[0],
            false,
        )[0];
        let deep = plan(
            Budget::Micros(us, c),
            16,
            PASS,
            &[30_000],
            &[100_000],
            &[0],
            false,
        )[0];
        assert!(shallow > deep && deep > 16, "{shallow} {deep}");
        // Each share costs about the budget at its own depth.
        for (d, n) in [(0, shallow), (30_000, deep)] {
            assert!((cost(c, d, n) - us).abs() < cost(c, d, 1) + 1.0, "{d} {n}");
        }
        // Two prompts: the shorter (and here deeper) one is sized at its
        // own depth, the other gets the time left at its depth.
        let p = plan(
            Budget::Micros(us, c),
            16,
            PASS,
            &[0, 30_000],
            &[100_000, 50_000],
            &[0, 0],
            false,
        );
        assert_eq!(p[1], deep);
        assert_eq!(p[0], 0, "{p:?}: nothing (or a sliver) left");
    }

    #[test]
    fn micros_first_gets_the_floor() {
        assert_eq!(
            plan(
                Budget::Micros(0.0, 0.005),
                256,
                PASS,
                &[0],
                &[10_000],
                &[0],
                false
            ),
            [256]
        );
        assert_eq!(
            plan(
                Budget::Micros(0.0, 0.005),
                256,
                PASS,
                &[0, 0],
                &[10_000, 9_000],
                &[0, 0],
                false
            ),
            [0, 256]
        );
    }

    #[test]
    fn starved_prompts_go_oldest_first() {
        // All passed over STARVE times: arrival order, not the remainder,
        // decides (by remainder the two short ones would take the step).
        let s = [STARVE; 3];
        assert_eq!(
            plan(
                Budget::Tokens(256),
                256,
                PASS,
                &[0; 3],
                &[5_000, 100, 200],
                &s,
                false
            ),
            [256, 0, 0]
        );
        // A starved long prompt goes before fresher short ones.
        assert_eq!(
            plan(
                Budget::Tokens(256),
                256,
                PASS,
                &[0; 3],
                &[5_000, 100, 200],
                &[STARVE, 0, 0],
                false
            ),
            [256, 0, 0]
        );
        // Not starved yet: shortest first (the 200 fed whole: within half a
        // floor past the 156 left).
        assert_eq!(
            plan(
                Budget::Tokens(256),
                256,
                PASS,
                &[0; 3],
                &[5_000, 100, 200],
                &[STARVE - 1, 0, 0],
                false
            ),
            [0, 100, 200]
        );
    }

    #[test]
    fn idle_feeds_the_head_in_whole_passes() {
        // Nothing decodes: the head (shortest first) takes the whole step,
        // its share rounded up to whole passes; the other gets nothing.
        let c = 0.005;
        let us = 250_000.0;
        let p = plan(
            Budget::Micros(us, c),
            256,
            PASS,
            &[0, 0],
            &[14_000, 14_400],
            &[0, 0],
            true,
        );
        assert_eq!(p[1], 0, "{p:?}");
        assert!(p[0] > 0 && p[0].is_multiple_of(PASS), "{p:?}");
        // Before a cost is known: IDLE_CHUNK tokens, all to the head.
        assert_eq!(
            plan(
                Budget::Tokens(4096),
                256,
                PASS,
                &[0, 0],
                &[14_000, 14_400],
                &[0, 0],
                true
            ),
            [4096, 0]
        );
        // An already starved prompt still goes first.
        assert_eq!(
            plan(
                Budget::Tokens(4096),
                256,
                PASS,
                &[0, 0],
                &[14_000, 14_400],
                &[0, STARVE],
                true
            ),
            [0, 4096]
        );
    }

    #[test]
    fn idle_misses_count_only_behind_a_younger_prompt_that_finished() {
        // The older one fed: the younger waits without a miss.
        assert_eq!(starve(&[4096, 0], &[9000, 9000], &[0, 2], true), [0, 2]);
        // A younger one fed, not finished: no miss for the older one.
        assert_eq!(starve(&[0, 2304], &[14_400, 10_000], &[1, 0], true), [1, 0]);
        // A younger one finished in this step: the older one counts a miss.
        assert_eq!(starve(&[0, 1000], &[14_400, 1000], &[1, 0], true), [2, 0]);
        // Idle feeds don't reset: a starved head stays starved.
        assert_eq!(
            starve(&[4096, 0], &[9000, 9000], &[STARVE, 0], true),
            [STARVE, 0]
        );
        // While requests decode: reset when fed, else one more.
        assert_eq!(starve(&[4096, 0], &[9000, 9000], &[0, 2], false), [0, 3]);
        assert_eq!(starve(&[0, 300], &[9000, 300], &[1, 3], false), [2, 0]);
    }

    #[test]
    fn idle_two_long_prompts_go_one_after_another() {
        // Arrival order, no misses: the older finishes first, untouched.
        let (mut fed, mut rem, mut st) = (vec![0, 0], vec![14_400, 14_400], vec![0, 0]);
        let mut steps = 0;
        while rem[0] > 0 {
            let p = plan(Budget::Tokens(4096), 256, PASS, &fed, &rem, &st, true);
            // The older one every step (the younger only gets what its
            // last piece leaves).
            assert!(
                p[0] > 0 && (p[1] == 0 || p[0] == rem[0]),
                "step {steps}: {p:?}"
            );
            st = starve(&p, &rem, &st, true);
            for i in 0..2 {
                fed[i] += p[i];
                rem[i] -= p[i];
            }
            steps += 1;
        }
        assert_eq!(st, [0, 0]);
        assert!(steps <= 4, "{steps}");
    }

    #[test]
    fn idle_short_prompts_finishing_in_their_step_cant_starve_a_long_one() {
        // Each step one more short prompt (1000 tokens, shorter than the
        // long one's remainder) arrives, takes the head and finishes in its
        // step (as a request with max_tokens 1 leaves): the long one is
        // passed over at most STARVE steps, then keeps the head to its end.
        let (c, floor) = (0.005, 256);
        let us = cost(c, 0, 1000) + 1.0; // a short prompt takes the step
        let (mut fed, mut rem, mut st) = (vec![0], vec![60_000], vec![0]);
        let mut missed = 0;
        let mut steps = 0;
        while rem[0] > 0 {
            fed.push(0);
            rem.push(1000);
            st.push(0);
            let p = plan(Budget::Micros(us, c), floor, PASS, &fed, &rem, &st, true);
            if p[0] == 0 {
                missed += 1;
            } else {
                assert!(missed >= STARVE, "fed before starving? {p:?}");
            }
            st = starve(&p, &rem, &st, true);
            for i in 0..p.len() {
                fed[i] += p[i];
                rem[i] -= p[i];
            }
            // Finished short prompts leave (they decode, then end).
            let keep: Vec<bool> = (0..rem.len()).map(|i| i == 0 || rem[i] > 0).collect();
            for v in [&mut fed, &mut rem, &mut st] {
                let mut k = keep.iter();
                v.retain(|_| *k.next().unwrap());
            }
            steps += 1;
            assert!(steps < 1000, "the long prompt starved");
        }
        assert_eq!(missed, STARVE);
    }

    #[test]
    fn idle_round_up_is_capped_and_floor_applies_first() {
        // A remainder within half a floor of the rounded share: fed whole,
        // not rounded up past it.
        assert_eq!(
            plan(
                Budget::Micros(1.0, 0.005),
                256,
                PASS,
                &[0],
                &[1000],
                &[0],
                true
            ),
            [1000]
        );
        assert_eq!(
            plan(
                Budget::Micros(1.0, 0.005),
                256,
                PASS,
                &[0],
                &[1280],
                &[0],
                true
            ),
            [1280]
        );
        // A larger one: whole passes.
        assert_eq!(
            plan(
                Budget::Micros(1.0, 0.005),
                256,
                PASS,
                &[0],
                &[5000],
                &[0],
                true
            ),
            [1152]
        );
        // The floor first (2000 > the time share), then whole passes.
        assert_eq!(
            plan(
                Budget::Micros(0.0, 0.005),
                2000,
                PASS,
                &[0],
                &[100_000],
                &[0],
                true
            ),
            [2304]
        );
        // The pass size is the pool's (PLAN_BATCH 1024 passes).
        assert_eq!(
            plan(
                Budget::Micros(0.0, 0.005),
                2000,
                1024,
                &[0],
                &[100_000],
                &[0],
                true
            ),
            [2048]
        );
    }

    /// Idle steps until both prompts are in (two prompts sent together):
    /// the plan each step.
    fn idle_two(rem0: usize, rem1: usize) -> Vec<Vec<usize>> {
        let (mut fed, mut rem, mut st) = (vec![0, 0], vec![rem0, rem1], vec![0, 0]);
        let mut steps = Vec::new();
        while rem.iter().any(|&r| r > 0) {
            let p = plan(Budget::Tokens(2304), 256, PASS, &fed, &rem, &st, true);
            st = starve(&p, &rem, &st, true);
            for i in 0..2 {
                fed[i] += p[i];
                rem[i] -= p[i];
                // (A finished prompt leaves the batch: its count is moot.)
                assert!(
                    rem[i] > 0 || p[i] == 0 || st[i] == 0,
                    "a miss for {i}: {st:?}"
                );
            }
            assert!(rem[0] == 0 || rem[1] == 0 || st == [0, 0], "a miss: {st:?}");
            steps.push(p);
            assert!(steps.len() < 100);
        }
        steps
    }

    #[test]
    fn idle_younger_shorter_long_prompt_runs_to_its_end() {
        // The younger, shorter one (shortest first) keeps the head until it
        // finishes; the older one (no misses: nothing finished ahead of it
        // until then) follows.
        let mut rem1 = 10_000;
        for p in idle_two(14_400, 10_000) {
            rem1 -= p[1];
            assert!(
                rem1 == 0 || p[0] == 0,
                "the older one took the head: {p:?}, {rem1} left"
            );
        }
    }

    #[test]
    fn idle_unequal_long_prompts_go_one_after_another() {
        // The older and shorter one first, untouched by the younger.
        let mut rem0 = 10_000;
        for p in idle_two(10_000, 14_400) {
            rem0 -= p[0];
            assert!(
                rem0 == 0 || p[1] == 0,
                "the younger one cut in: {p:?}, {rem0} left"
            );
        }
    }
}

#[cfg(test)]
mod sched_tests {
    use super::{
        AdmitOrder, Budget, ChunkPolicy, DEPTH_K, STARVE, Sched, StepCost, admit_gate, admit_key,
        bucket, lanes_for, plan, plan_charged, starve, step_prompts, two_pass_budget,
        two_pass_capacity, two_pass_short_passes, units,
    };

    /// The rollback's admission side (shady S2 on #131): AXON_STEP_COST=
    /// legacy alone (no AXON_ADMIT, or AXON_ADMIT=fifo from older configs)
    /// is fifo with 2 lanes at any slot count, main's `MAX_PREFILLING`;
    /// AXON_ADMIT=sjf breaks the "alone" (sjf's lanes scale with slots).
    #[test]
    fn legacy_alone_is_main_admission() {
        for admit in [None, Some("fifo"), Some("anything")] {
            let s = Sched::from_env(admit, Some("legacy"));
            assert!(s.legacy && s.order == AdmitOrder::Fifo, "{admit:?}");
            for slots in [1, 8, 16, 32, 64, 128, 256] {
                assert_eq!(lanes_for(&s, slots), 2, "{admit:?} {slots}");
            }
        }
        let sjf = Sched::from_env(Some("sjf"), Some("legacy"));
        assert_eq!(sjf.order, AdmitOrder::Sjf);
        let l: Vec<usize> = [16, 32, 64, 128, 256].map(|n| lanes_for(&sjf, n)).to_vec();
        assert_eq!(l, [2, 2, 4, 8, 8]);
        // No env at all: the model's budgets, fifo, 2 lanes.
        let d = Sched::from_env(None, None);
        assert!(!d.legacy && d.order == AdmitOrder::Fifo && lanes_for(&d, 128) == 2);
    }

    const PASS: usize = 1152;

    fn policy(floor: usize, step_ms: usize) -> ChunkPolicy {
        ChunkPolicy {
            floor,
            step_ms,
            step_tokens: 0,
            legacy: false,
        }
    }

    /// Shortest uncached prompt first, ties by arrival; a request aged
    /// past the bound goes before all of them (aged ones oldest first);
    /// a bound of 0 never ages.
    #[test]
    fn admit_key_sjf_then_age() {
        // (uncached, age ms, id): 7000 / 60 / 2000 tokens, arrived in order.
        let jobs = [(7000, 900, 1u64), (60, 500, 2), (2000, 100, 3)];
        let order = |bound: u64, jobs: &[(usize, u64, u64)]| {
            let mut v: Vec<_> = jobs
                .iter()
                .map(|&(u, a, id)| (admit_key(u, a, bound, id), id))
                .collect();
            v.sort();
            v.into_iter().map(|x| x.1).collect::<Vec<_>>()
        };
        assert_eq!(order(0, &jobs), [2, 3, 1]);
        assert_eq!(order(3000, &jobs), [2, 3, 1]);
        // The 7000 aged past the bound: first.
        let aged = [(7000, 3100, 1u64), (60, 500, 2), (2000, 100, 3)];
        assert_eq!(order(3000, &aged), [1, 2, 3]);
        // Bound 0: never aged.
        assert_eq!(order(0, &aged), [2, 3, 1]);
        // Two aged: oldest (lowest id) first, whatever their length.
        let two = [(100, 4000, 5u64), (9000, 5000, 4), (50, 10, 6)];
        assert_eq!(order(3000, &two), [4, 5, 6]);
        // Equal lengths: arrival order.
        let eq = [(300, 10, 9u64), (300, 20, 8), (300, 30, 7)];
        assert_eq!(order(3000, &eq), [7, 8, 9]);
    }

    /// Lanes bound long prompts, never short ones; short ones stop at
    /// about two steps' worth; fifo is the old rule (`lanes` prompts
    /// prefilling at most, whatever their length).
    #[test]
    fn can_admit_lanes_and_short_budget() {
        // A token budget: no bound on short prompts' passes but their tokens.
        const ALL: usize = usize::MAX / 4;
        let b = 256;
        // Two long prefills, lanes 2: a 60-token job starts, a 3000 doesn't.
        let longs = [5000, 7000];
        assert!(admit_gate(AdmitOrder::Sjf, 2, b, ALL, &longs, 60));
        assert!(!admit_gate(AdmitOrder::Sjf, 2, b, ALL, &longs, 3000));
        assert!(admit_gate(AdmitOrder::Sjf, 3, b, ALL, &longs, 3000));
        // Short remainders summing to 2b: no more short ones; a long one
        // still gets a free lane.
        let shorts = [200, 156, 156];
        assert_eq!(shorts.iter().sum::<usize>(), 2 * b);
        assert!(!admit_gate(AdmitOrder::Sjf, 2, b, ALL, &shorts, 1));
        assert!(admit_gate(AdmitOrder::Sjf, 2, b, ALL, &shorts, 3000));
        assert!(admit_gate(AdmitOrder::Sjf, 2, b, ALL, &shorts[1..], 200));
        // Fifo: two prompts prefilling of any length block everything.
        assert!(!admit_gate(AdmitOrder::Fifo, 2, 0, 0, &[60, 70], 10));
        assert!(!admit_gate(AdmitOrder::Fifo, 2, 0, 0, &longs, 60));
        assert!(admit_gate(AdmitOrder::Fifo, 2, 0, 0, &[5000], 60));
        assert!(admit_gate(AdmitOrder::Fifo, 3, 0, 0, &longs, 60));
        // Two-pass: at most two steps' passes of short prompts (here 3 a
        // step): five queued, a sixth goes; six queued, no seventh.
        assert!(admit_gate(AdmitOrder::Sjf, 2, b, 3, &[60; 5], 60));
        assert!(!admit_gate(AdmitOrder::Sjf, 2, b, 3, &[60; 6], 60));
        // ... and a long one still gets its lane.
        assert!(admit_gate(AdmitOrder::Sjf, 2, b, 3, &[60; 6], 3000));
        // (A plain pool returns before the gate: `admit_check`.)
    }

    /// The pass cost a two-pass step charges: 40 ms left beside the decode
    /// step, a pass's fixed part 8 ms, 512 tokens 30 ms: 3 short prompts
    /// a step (`fed`), and `two_pass_short_passes` agrees.
    fn shorts_fed(cost: &StepCost) -> usize {
        let p = policy(256, 50);
        let (b, pass) = two_pass_budget(&p, cost, 8);
        let (n, rem) = ([0; 10], [60; 10]);
        let plan = plan_charged(b, pass, usize::MAX, 256, PASS, &n, &rem, &n, false);
        plan.iter().filter(|&&x| x > 0).count()
    }

    /// A warm model: decode steps of 10 ms, a 512-token feed, then short
    /// feeds whose fixed part is 8 ms; and that short feed's time.
    fn warm_cost() -> (StepCost, f64) {
        let mut c = StepCost {
            pass_cap_us: 25_000.0,
            ..Default::default()
        };
        c.learn_decode(8, 10.0);
        c.learn_feed(0, 512, 30_000.0);
        let short = 8000.0 + c.tok.unwrap() * units(0, 59);
        for _ in 0..3 {
            c.learn_feed(0, 59, short);
        }
        (c, short)
    }

    /// Normal short feeds (`us` each) until `shorts_fed` is what 50 of
    /// them give: how many it took.
    fn recover(c: &mut StepCost, us: f64) -> usize {
        let mut t = c.clone();
        for _ in 0..50 {
            t.learn_feed(0, 59, us);
        }
        let target = shorts_fed(&t);
        let mut steps = 0;
        while shorts_fed(c) != target {
            c.learn_feed(0, 59, us);
            steps += 1;
            assert!(steps < 50, "never back: pass {}", c.pass());
        }
        steps
    }

    /// MUST 1 (review of #126): one stalled short feed (10x) can't pin
    /// two-pass prefill: its sample is capped (4x the estimate, half the
    /// step), the estimate is bounded by twice the fastest decode step,
    /// and normal feeds bring the budget back within 3 steps. A first
    /// sample that is an outlier (nothing to cap it against) is bounded
    /// the same way and recovers within 5 (EMA 1/2 from 20 ms to 8.5).
    #[test]
    fn pass_us_outlier_recovers() {
        let (warm, short) = warm_cost();
        assert_eq!((warm.pass(), shorts_fed(&warm)), (8000.0, 3));
        let p = policy(256, 50);
        assert_eq!(two_pass_short_passes(&p, &warm, 8), 3);
        let mut c = warm.clone();
        c.learn_feed(0, 59, 10.0 * short);
        assert!(c.pass() <= 25_000.0, "{}", c.pass());
        assert!(shorts_fed(&c) < 3, "the outlier is felt");
        assert!(recover(&mut c, short) <= 3);
        assert_eq!(shorts_fed(&c), 3);
        // A first sample of 80 ms: at most half the step, and twice the
        // fastest decode step.
        let mut c = StepCost {
            pass_cap_us: 25_000.0,
            ..Default::default()
        };
        c.learn_decode(8, 10.0);
        c.learn_feed(0, 59, 80_000.0);
        assert_eq!(c.pass(), 20_000.0);
        c.learn_feed(0, 512, 30_000.0);
        let us = 8000.0 + c.tok.unwrap() * units(0, 59);
        assert!(recover(&mut c, us) <= 5);
        // No short feeds at all: the decode-step bound alone.
        let mut h = warm.clone();
        h.learn_feed(0, 59, 10.0 * short);
        h.learn_decode(1, 4.0);
        assert!(h.pass() <= 8000.0, "{}", h.pass());
    }

    /// The first wave on a fresh, idle server (bella 1025-153024-9998):
    /// a token budget, nothing measured. Before, all of a wave's short
    /// prompts were fed in one step, each in its own pass, and all their
    /// first tokens waited for the whole step; now the first step feeds 1
    /// (teaching a pass's cost), then as many as `step_ms` fits: with 8 ms
    /// passes (~10.4 ms with a 64-token chunk) and 10 ms decode steps, 3 a
    /// step at 50 ms (idle: 4). Shortest
    /// first and starved first still hold; legacy, a pinned step and
    /// `--step-ms 0` keep no bound.
    #[test]
    fn idle_wave_fed_a_step_at_a_time() {
        let p = policy(256, 50);
        let cold = StepCost::default();
        let (b, pass) = two_pass_budget(&p, &cold, 0);
        assert!(matches!(b, Budget::Tokens(_)));
        let rem = [40, 60, 50, 9000, 45, 55, 70, 65];
        let z = [0; 8];
        // Before (no bound): every short and the long one, one step.
        let all = plan_charged(b, pass, usize::MAX, 256, PASS, &z, &rem, &z, true);
        assert_eq!(all.iter().filter(|&&n| n > 0).count(), 8, "{all:?}");
        // Now: the shortest alone.
        assert_eq!(step_prompts(&p, &cold, 0), 1);
        let one = plan_charged(b, pass, 1, 256, PASS, &z, &rem, &z, true);
        assert_eq!(one, [40, 0, 0, 0, 0, 0, 0, 0]);
        // Warm: 8 ms passes, 10 ms decode steps.
        let (warm, _) = warm_cost();
        assert_eq!(step_prompts(&p, &warm, 8), 3);
        assert_eq!(step_prompts(&p, &warm, 0), 4);
        // A starved long prompt still goes first (idle: it takes the step).
        let st = [0, 0, 0, super::STARVE, 0, 0, 0, 0];
        let k = plan_charged(b, pass, 2, 256, PASS, &z, &rem, &st, true);
        assert_eq!(k, [0, 0, 0, 4096, 0, 0, 0, 0]);
        // The admission gate follows (twice the step's prompts).
        assert!(step_prompts(&p, &warm, 8) <= two_pass_short_passes(&p, &warm, 8));
        // No bound: legacy, a pinned step, --step-ms 0.
        for q in [
            ChunkPolicy { legacy: true, ..p },
            ChunkPolicy {
                step_tokens: 256,
                ..p
            },
            policy(256, 0),
        ] {
            assert_eq!(step_prompts(&q, &warm, 8), usize::MAX);
        }
    }

    /// SHOULD (c): before any short feed, a pass is charged the fastest
    /// decode step (measured, no fixed seed): a cold wave of short prompts
    /// is fed a few a step, not all in one.
    #[test]
    fn cold_pass_cost_is_the_decode_step() {
        let mut c = StepCost::default();
        c.learn_feed(0, 512, 30_000.0);
        assert_eq!(c.pass(), 0.0);
        assert_eq!(shorts_fed(&c), 10, "nothing measured: no charge");
        c.learn_decode(8, 10.0);
        assert_eq!(c.pass(), 10_000.0);
        assert!(shorts_fed(&c) <= 4, "{}", shorts_fed(&c));
    }

    /// MUST 2 (review of #126): a prompt's class doesn't depend on what is
    /// prefilling: the short bound is the step's capacity (one prompt at
    /// depth 0, from the rows and the cost: `two_pass_capacity` takes no
    /// prefill state), the same for every candidate of a round however
    /// `pre` fills; only the rows (the next iteration) move it.
    #[test]
    fn short_class_stable_across_a_round() {
        let (mut cost, _) = warm_cost();
        cost.learn_decode(32, 20.0);
        let p = policy(256, 50);
        let cap = two_pass_capacity(&p, &cost, 8).max(256);
        assert!(cap > 300 && cap < 4000, "{cap}");
        let mut pre: Vec<usize> = Vec::new();
        let mut classes = Vec::new();
        for want in [300, 60, 7600, 300, 6000, 60, cap, cap + 1] {
            let short = two_pass_capacity(&p, &cost, 8).max(256);
            assert_eq!(short, cap);
            classes.push((want, want <= short));
            let passes = two_pass_short_passes(&p, &cost, 8);
            if admit_gate(AdmitOrder::Sjf, 2, short, passes, &pre, want) {
                pre.push(want);
            }
        }
        assert!(classes.iter().all(|&(w, s)| s == (w <= cap)), "{classes:?}");
        assert!(pre.contains(&7600) && pre.contains(&6000), "{pre:?}");
        // More rows decoding: a slower decode step, less room.
        assert!(two_pass_capacity(&p, &cost, 32) < two_pass_capacity(&p, &cost, 8));
    }

    /// Two-pass steps charge a pass's fixed part per prompt fed: with 40
    /// ms left beside the decode step and 8 ms per pass, 60-token prompts
    /// costing ~1 ms each are fed four at a time (40 / 9), not all of them.
    #[test]
    fn plan_two_pass_charges_pass_fixed() {
        let (pass_us, floor) = (8000.0, 256);
        let c = 1000.0 / (60.0 * (30.0 + DEPTH_K));
        let rem = [60; 10];
        for (dec_ms, want) in [(10.0, 4), (6.0, 4), (14.0, 4), (30.0, 2), (49.0, 1)] {
            let left = (50.0 - dec_ms) * 1e3;
            let p = plan_charged(
                Budget::Micros(left, c),
                pass_us,
                usize::MAX,
                floor,
                PASS,
                &[0; 10],
                &rem,
                &[0; 10],
                false,
            );
            let fed = p.iter().filter(|&&n| n > 0).count();
            assert!(
                fed <= ((50.0 - dec_ms) / 9.0).floor().max(1.0) as usize,
                "{dec_ms}: {p:?}"
            );
            assert_eq!(fed, want, "{dec_ms}: {p:?}");
            assert!(p.iter().all(|&n| n == 0 || n == 60), "{p:?}");
        }
        // Without the charge (the old rule): every short prompt fits.
        let p = plan(
            Budget::Micros(40e3, c),
            floor,
            PASS,
            &[0; 10],
            &rem,
            &[0; 10],
            false,
        );
        assert_eq!(p, [60; 10]);
        // The budget: step_ms less the modelled decode step, the pass's
        // fixed part handed to plan.
        let mut cost = StepCost::default();
        cost.learn_decode(8, 10.0);
        // A short feed first: all of it is the fixed part (no cost per
        // token yet); the long feed's per-token cost is learned past it.
        cost.learn_feed(0, 59, 9000.0);
        cost.learn_feed(0, 512, 30_000.0);
        let (b, pass) = two_pass_budget(&policy(256, 50), &cost, 8);
        assert_eq!(pass, 9000.0);
        let tok = cost.tok.unwrap();
        assert!(
            (tok * 512.0 * (256.0 + DEPTH_K) - 21_000.0).abs() < 1e-6,
            "{tok}"
        );
        match b {
            Budget::Micros(us, _) => assert_eq!(us, 40_000.0),
            Budget::Tokens(t) => panic!("{t}"),
        }
    }

    /// dec(rows): a seen bucket's own value; an unseen bucket from the
    /// nearest seen one scaled by sqrt(rows ratio); nothing seen: the last
    /// decode step.
    #[test]
    fn step_cost_buckets() {
        assert_eq!(
            [1, 2, 3, 4, 5, 8, 9, 16, 128, 129, 256, 1000].map(bucket),
            [0, 1, 2, 2, 3, 3, 4, 4, 7, 8, 8, 8]
        );
        let mut c = StepCost::default();
        assert_eq!(c.dec(16), 0.0);
        c.last_dec_ms = 12.0;
        assert_eq!(c.dec(16), 12.0);
        c.learn_decode(16, 20.0);
        assert_eq!(c.dec(16), 20.0);
        assert_eq!(c.dec(9), 20.0);
        // 64 rows: bucket 6 unseen, nearest seen bucket 4 (16 rows): x2.
        assert!((c.dec(64) - 40.0).abs() < 1e-9, "{}", c.dec(64));
        // Learning: an EMA of 1/4.
        c.learn_decode(16, 24.0);
        assert_eq!(c.dec(16), 21.0);
        // The last step's time follows every decode step.
        assert_eq!(c.last_dec_ms, 24.0);
        // A pass's fixed part: from short feeds, the per-token part out.
        let mut f = StepCost::default();
        f.learn_feed(0, 512, 51_200.0);
        f.learn_feed(0, 40, 9_000.0);
        let tok = f.tok.unwrap();
        let fixed = 9000.0 - tok * 40.0 * (20.0 + DEPTH_K);
        assert!((f.pass_us.unwrap() - fixed).abs() < 1e-6);
        // Feeds 64..128: teach nothing.
        let g = f.clone();
        f.learn_feed(0, 100, 1e9);
        assert_eq!(
            (f.tok, f.pass_us, f.legacy_tok),
            (g.tok, g.pass_us, g.legacy_tok)
        );
    }

    /// main's `plan` (engine.rs at d6b7051, unchanged since ab5568c), verbatim.
    fn plan_main(
        budget: Budget,
        floor: usize,
        pass: usize,
        fed: &[usize],
        rem: &[usize],
        starved: &[usize],
        idle: bool,
    ) -> Vec<usize> {
        let mut plan = vec![0; rem.len()];
        let mut order: Vec<usize> = (0..rem.len()).collect();
        order.sort_by_key(|&i| {
            if starved[i] >= STARVE {
                (false, 0, i)
            } else {
                (true, rem[i], i)
            }
        });
        let mut budget = budget;
        let mut first = true;
        for i in order {
            let share = match budget {
                Budget::Tokens(t) => t,
                Budget::Micros(us, c) => {
                    let dk = fed[i] as f64 + DEPTH_K;
                    let n = ((dk * dk + 2.0 * us / c).sqrt() - dk) as usize;
                    match (first, idle) {
                        (true, true) => n.max(floor).div_ceil(pass).saturating_mul(pass),
                        (true, false) => n.max(floor),
                        _ => n,
                    }
                }
            };
            let want = rem[i];
            let n = if want <= share.saturating_add(floor / 2) {
                want
            } else if first || share >= floor {
                share
            } else {
                0
            };
            if n == 0 || share == 0 {
                continue;
            }
            first = false;
            plan[i] = n;
            budget = match budget {
                Budget::Tokens(t) => Budget::Tokens(t.saturating_sub(n)),
                Budget::Micros(us, c) => {
                    let spent = c * n as f64 * (fed[i] as f64 + n as f64 / 2.0 + DEPTH_K);
                    Budget::Micros((us - spent).max(0.0), c)
                }
            };
        }
        plan
    }

    /// main's budget (engine.rs `Batch::budget` at d6b7051, unchanged since
    /// ab5568c), as a function of its inputs.
    fn budget_main(
        floor: usize,
        ms: usize,
        cost: Option<f64>,
        decode_ms: f64,
        idle: bool,
    ) -> Budget {
        if floor == 0 {
            return Budget::Tokens(usize::MAX);
        }
        let cost = cost.filter(|&c| c > 0.0);
        match (cost, idle, ms) {
            (_, true, 0) | (None, true, _) => Budget::Tokens(super::IDLE_CHUNK.max(floor)),
            (Some(c), true, _) => Budget::Micros(super::IDLE_MS * 1e3, c),
            (_, false, 0) => Budget::Tokens(floor),
            (None, false, _) => Budget::Tokens(super::SEED_CHUNK.max(floor)),
            (Some(c), false, _) => Budget::Micros((ms as f64 - decode_ms).max(0.0) * 1e3, c),
        }
    }

    fn same(a: Budget, b: Budget) -> bool {
        match (a, b) {
            (Budget::Tokens(x), Budget::Tokens(y)) => x == y,
            (Budget::Micros(u, c), Budget::Micros(v, d)) => {
                u.to_bits() == v.to_bits() && c.to_bits() == d.to_bits()
            }
            _ => false,
        }
    }

    /// The rollback: with AXON_STEP_COST=legacy (and the default fifo
    /// admission, `admit_gate`'s fifo arm with 2 lanes at any slot count:
    /// `legacy_alone_is_main_admission`) every scheduling decision is
    /// main's (d6b7051, unchanged at fcde887), bit for bit,
    /// over 20,000 random states: the budget (the same timings fed through
    /// `StepCost` as through main's `cost` / `decode_ms`) and the plan
    /// (pass charge 0), against frozen copies of main's code; the fifo
    /// admission gate is `pre.len() < 2`, main's `MAX_PREFILLING`.
    #[test]
    fn legacy_reproduces_main_decisions() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n.max(1)
        };
        for it in 0..20_000 {
            let floor = [0, 1, 16, 64, 256, 512][rnd(6) as usize];
            let ms = [0, 20, 50, 60, 100][rnd(5) as usize];
            let p = ChunkPolicy {
                floor,
                step_ms: ms,
                step_tokens: [0, 300][rnd(2) as usize],
                legacy: true,
            };
            // A history of feeds and decode steps, through both models.
            let (mut cost, mut old, mut dec) = (StepCost::default(), None::<f64>, 0.0);
            for _ in 0..rnd(6) {
                let (d, n) = (rnd(30_000) as usize, 1 + rnd(2000) as usize);
                let us = 100.0 + rnd(200_000) as f64 / 7.0;
                cost.learn_feed(d, n, us);
                let mid = d as f64 + n as f64 / 2.0 + DEPTH_K;
                let c = us / (n as f64 * mid);
                if n >= super::LEARN_MIN {
                    old = Some(old.map_or(c, |a| 0.5 * a + 0.5 * c));
                }
                let (r, secs) = (1 + rnd(200) as usize, rnd(90_000) as f64 / 1e6);
                cost.learn_decode(r, secs * 1e3);
                dec = secs * 1e3;
            }
            let rows = rnd(4) as usize * rnd(150) as usize;
            let (b, pass_us) = two_pass_budget(&p, &cost, rows);
            assert_eq!(pass_us, 0.0);
            assert!(
                same(b, budget_main(floor, ms, old, dec, rows == 0)),
                "it {it}"
            );
            let n = 1 + rnd(4) as usize;
            let fed: Vec<usize> = (0..n).map(|_| rnd(20_000) as usize).collect();
            let rem: Vec<usize> = (0..n).map(|_| 1 + rnd(9000) as usize).collect();
            let st: Vec<usize> = (0..n).map(|_| rnd(STARVE as u64 + 2) as usize).collect();
            let pass = [1024, 1152][rnd(2) as usize];
            let idle = rows == 0;
            let most = step_prompts(&p, &cost, rows);
            assert_eq!(most, usize::MAX, "legacy: no bound on prompts a step");
            let a = plan_charged(b, pass_us, most, floor, pass, &fed, &rem, &st, idle);
            assert_eq!(
                a,
                plan_main(b, floor, pass, &fed, &rem, &st, idle),
                "it {it}"
            );
            assert_eq!(starve(&a, &rem, &st, idle).len(), n);
            let pre: Vec<usize> = (0..rnd(4)).map(|_| rnd(9000) as usize).collect();
            assert_eq!(
                admit_gate(AdmitOrder::Fifo, 2, 0, 0, &pre, rnd(9000) as usize),
                pre.len() < 2
            );
        }
    }
}

#[cfg(test)]
mod victim_tests {
    use super::{Victim, victim};

    /// The inversion case: an old long prompt L (id 1) still prefilling and
    /// holding rows, a young request Z (id 2) decoding and short of a page:
    /// Z gives way (preempted), L keeps its rows.
    #[test]
    fn older_prefill_keeps_its_rows() {
        assert_eq!(victim(2, &[2], &[(1, true)]), Victim::Preempt(0));
        // A younger prefill does give its rows up, for any older grower.
        assert_eq!(victim(2, &[2], &[(1, true), (3, true)]), Victim::Unfeed(1));
        assert_eq!(victim(1, &[1, 2], &[(3, true)]), Victim::Unfeed(0));
        // A younger prefill holding nothing: the youngest running goes.
        assert_eq!(victim(1, &[1, 4, 2], &[(3, false)]), Victim::Preempt(1));
        // The oldest request of all never gives way.
        assert_eq!(victim(1, &[1], &[(3, false)]), Victim::Stuck);
        assert_eq!(victim(2, &[2], &[]), Victim::Stuck);
    }

    /// A pool of 25 pages (64 positions), an old 1200-token prompt L (id 1,
    /// 19 pages) fed 64 a step, a young request Z (id 2) decoding a token a
    /// step from a 600-token history (10 pages): (L's refeeds, L finished,
    /// Z's tokens) after 400 steps, `rule` deciding who gives way.
    fn simulate(rule: impl Fn(u64, &[u64], &[(u64, bool)]) -> Victim) -> (usize, bool, usize) {
        let (total, need) = (25usize, |n: usize| n.div_ceil(64));
        let (l_len, mut l_fed, mut l_done, mut refeeds) = (1200usize, 0usize, false, 0);
        let (mut z_len, mut z_running, mut z_tokens) = (600usize, true, 0);
        let used = |l_fed: usize, l_done: bool, z_len: usize, z_running: bool| {
            (if l_done { 0 } else { need(l_fed) }) + if z_running { need(z_len) } else { 0 }
        };
        for _ in 0..400 {
            if z_running && used(l_fed, l_done, z_len + 1, true) > total {
                let pre = if l_done { vec![] } else { vec![(1, l_fed > 0)] };
                match rule(2, &[2], &pre) {
                    Victim::Unfeed(_) => {
                        l_fed = 0;
                        refeeds += 1;
                    }
                    Victim::Preempt(_) => z_running = false,
                    Victim::Stuck => panic!("stuck"),
                }
            }
            if z_running && used(l_fed, l_done, z_len + 1, true) <= total {
                z_len += 1;
                z_tokens += 1;
            }
            if !l_done {
                let next = (l_fed + 64).min(l_len);
                if used(next, false, z_len, z_running) <= total {
                    l_fed = next;
                }
                // Fed whole: it decodes and finishes (its pages free).
                l_done = l_fed == l_len;
            }
            if !z_running && used(l_fed, l_done, z_len, true) <= total {
                z_running = true; // fed again, it goes on
            }
        }
        (refeeds, l_done, z_tokens)
    }

    /// The rule: Z gives way, L is never fed again from 0 and finishes, then
    /// Z goes on. The old rule (any prefill holding rows gives way) refeeds
    /// L over and over and it never finishes.
    #[test]
    fn grow_rule_makes_progress_without_refeeding_the_old_prompt() {
        let (refeeds, l_done, z_tokens) = simulate(victim);
        assert_eq!(refeeds, 0, "L fed again");
        assert!(l_done, "L finished");
        assert!(z_tokens > 200, "Z went on: {z_tokens}");
        let old = |_: u64, running: &[u64], pre: &[(u64, bool)]| match pre.iter().position(|p| p.1)
        {
            Some(j) => Victim::Unfeed(j),
            None if running.len() > 1 => Victim::Preempt(running.len() - 1),
            None => Victim::Stuck,
        };
        let (refeeds, l_done, _) = simulate(old);
        assert!(
            refeeds > 1 && !l_done,
            "the old rule: {refeeds} refeeds, L done {l_done}"
        );
    }
}

#[cfg(test)]
mod trace_tests {
    use super::StepTrace;
    use std::time::Instant;

    /// AXON_SERVE_TRACE's lines are JSON with every field, a step without
    /// a decode pass has nulls for its decode fields.
    #[test]
    fn step_trace_lines_are_json() {
        let path = std::env::temp_dir().join(format!("axon-trace-{}.jsonl", std::process::id()));
        let f = std::fs::File::create(&path).unwrap();
        let mut tr = StepTrace {
            out: std::io::LineWriter::new(f),
            t0: Instant::now(),
            feeds: 2,
            fed: 700,
            feed_ms: 31.25,
            dec: Some((128, 128 * 790 + 64, 18.6)),
        };
        let v: serde_json::Value = serde_json::from_str(&tr.line(5.0, 52.5, 2, 128)).unwrap();
        assert_eq!(v["fed"], 700);
        assert_eq!(v["feeds"], 2);
        assert_eq!(v["rows"], 128);
        assert_eq!(v["depth"], 790.5);
        assert_eq!(v["dec_ms"], 18.6);
        assert_eq!(v["ms"], 52.5);
        assert_eq!(v["running"], 128);
        tr.dec = None;
        let v: serde_json::Value = serde_json::from_str(&tr.line(5.0, 30.0, 1, 0)).unwrap();
        assert_eq!(v["rows"], 0);
        assert!(v["dec_ms"].is_null() && v["depth"].is_null());
        assert_eq!(v["prefilling"], 1);
        drop(tr);
        std::fs::remove_file(&path).unwrap();
    }
}
