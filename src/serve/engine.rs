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
//! is sized to a time budget (`--prefill-ms`: a step's chunk and decode
//! together take about that long), at least `--prefill-chunk` tokens; with
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
use super::{Event, Job, TextOut};
use crate::lm::{Backend, BatchLm, RowHead, RowOut, SeqRow, StepError};
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
/// A step's time budget for its prefill chunk and decode together, in ms
/// (default, `--prefill-ms`; 0: chunks of exactly `--prefill-chunk`).
pub const PREFILL_MS: usize = 60;
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
    /// The step's budget in ms (0: chunks of `floor`).
    pub ms: usize,
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
    /// Prefill cost per token per position attended (+ DEPTH_K), µs:
    /// a moving average over feeds of at least LEARN_MIN tokens, once
    /// measured.
    cost: Option<f64>,
    /// The last decode step's time.
    decode_ms: f64,
    /// Most tokens of one of `feed_slot`'s passes (`BatchLm::pass_max`):
    /// an idle piece is a whole number of them (no extra partial passes).
    pass: usize,
    log: bool,
    generated: Arc<AtomicU64>,
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
            cost: None,
            decode_ms: 0.0,
            log,
            generated,
        }
    }

    pub fn idle(&self) -> bool {
        self.seqs.is_empty() && self.pre.is_empty()
    }

    pub fn has_free(&self) -> bool {
        self.free() > 0
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
    pub fn admit(&mut self, job: Job, out: TextOut<'t>) {
        let p = &job.prompt;
        let last = p.len() - 1;
        let (slot, common) = self.free_slot(p).expect("admit: a free slot");
        let reuse = common.min(last);
        if super::debug() {
            eprintln!(
                "axon serve: #{} prefill: chunked, slot {slot}, {} tok ({reuse} cached), {} decoding",
                job.id,
                p.len(),
                self.seqs.len()
            );
        }
        self.busy[slot] = true;
        self.touch(slot);
        self.held[slot].truncate(reuse);
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
    }

    /// What this step may feed: everything without chunking; with
    /// nothing decoding (nobody to stall) IDLE_MS worth (IDLE_CHUNK before
    /// a cost is known), so requests arriving meanwhile still start soon;
    /// else `floor` tokens (no time budget), or what fits in the budget
    /// beside the last decode step at the measured cost (SEED_CHUNK before
    /// one is known).
    fn budget(&self) -> Budget {
        let ChunkPolicy { floor, ms } = self.policy;
        if floor == 0 {
            return Budget::Tokens(usize::MAX);
        }
        let idle = self.seqs.is_empty();
        let cost = self.cost.filter(|&c| c > 0.0);
        match (cost, idle, ms) {
            (_, true, 0) | (None, true, _) => Budget::Tokens(IDLE_CHUNK.max(floor)),
            (Some(c), true, _) => Budget::Micros(IDLE_MS * 1e3, c),
            (_, false, 0) => Budget::Tokens(floor),
            (None, false, _) => Budget::Tokens(SEED_CHUNK.max(floor)),
            (Some(c), false, _) => Budget::Micros((ms as f64 - self.decode_ms).max(0.0) * 1e3, c),
        }
    }

    /// Feed this step's prompt (`budget`, split by `plan`); prompts then
    /// in their slots join the running requests.
    fn prefill(&mut self) -> Result<(), String> {
        if self.pre.is_empty() {
            return Ok(());
        }
        let budget = self.budget();
        let rem: Vec<usize> = self
            .pre
            .iter()
            .map(|f| f.seq.hist.len() - 1 - f.fed)
            .collect();
        let fed: Vec<usize> = self.pre.iter().map(|f| f.fed).collect();
        let starved: Vec<usize> = self.pre.iter().map(|f| f.starved).collect();
        let idle = self.seqs.is_empty();
        let plan = plan(
            budget,
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
        for (i, &n) in plan.iter().enumerate() {
            if n > 0 {
                self.feed(i, n)?;
            }
        }
        let mut i = 0;
        while i < self.pre.len() {
            let f = &self.pre[i];
            if f.fed == f.seq.hist.len() - 1 {
                let mut f = self.pre.remove(i);
                f.seq.prompt_ms = f.t0.elapsed().as_secs_f64() * 1e3;
                f.seq.t1 = Instant::now();
                self.seqs.push(f.seq);
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Feed the next `n` tokens of prefill `i`'s prompt into its slot.
    fn feed(&mut self, i: usize, n: usize) -> Result<(), String> {
        let f = &mut self.pre[i];
        let s = f.seq.slot;
        let toks = &f.seq.hist[f.fed..f.fed + n];
        let t = Instant::now();
        self.net.feed_slot(&mut self.pool, s, toks, f.fed)?;
        let mid = f.fed as f64 + n as f64 / 2.0 + DEPTH_K;
        let c = t.elapsed().as_secs_f64() * 1e6 / (n as f64 * mid);
        self.held[s].extend_from_slice(toks);
        f.fed += n;
        // Only long enough feeds teach the cost (a short one costs more per
        // token: a pass's fixed part).
        if n >= LEARN_MIN {
            self.cost = Some(self.cost.map_or(c, |a| 0.5 * a + 0.5 * c));
        }
        Ok(())
    }

    /// A slot for the S1 path to run `prompt` in, chosen as `admit`
    /// chooses (`best_slot`: the longest prefix of it, else the least
    /// recently used), `cache` (a `slot_cache` of the pool) pointed at it,
    /// and the tokens the slot holds (the engine reuses as many as the
    /// prompt shares); the slot holds nothing meanwhile. The batch is idle.
    pub fn lend_solo(
        &mut self,
        prompt: &[u32],
        cache: &mut <Backend<'a> as crate::lm::Lm>::Cache,
    ) -> Result<Vec<u32>, String> {
        assert!(self.idle(), "lend_solo: the batch is running");
        let (slot, _) = self.free_slot(prompt).expect("lend_solo: a free slot");
        self.net.move_slot_cache(&self.pool, cache, slot)?;
        self.solo = slot;
        self.touch(slot);
        Ok(std::mem::take(&mut self.held[slot]))
    }

    /// The slot the S1 path runs in (`lend_solo`).
    pub fn solo_slot(&self) -> usize {
        self.solo
    }

    /// The S1 path ended (without a hand-over): its slot holds the rows of
    /// `hist`.
    pub fn end_solo(&mut self, hist: Vec<u32>) {
        self.held[self.solo] = hist;
    }

    /// Take over a request `generate` was running on `cache` (the S1 path,
    /// in the slot `lend_solo` gave it, so its rows 0..hist.len() - 1 are
    /// already there and `BatchLm::adopt` copies nothing).
    pub fn adopt(
        &mut self,
        mut seq: Seq<'t>,
        cache: &<Backend<'a> as crate::lm::Lm>::Cache,
    ) -> Result<(), String> {
        let n = seq.hist.len() - 1;
        let slot = self.solo;
        assert!(!self.busy[slot], "adopt: the S1 slot is busy");
        self.net.adopt(&mut self.pool, slot, cache, n)?;
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
    /// running request and each one's next token.
    pub fn step(&mut self) -> Result<(), String> {
        self.reap();
        self.prefill()?;
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
        self.decode_ms = t.elapsed().as_secs_f64() * 1e3;
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

    /// Each slot's state, for GET /slots and /metrics (read only).
    pub fn slot_states(&self) -> Vec<super::ops::SlotState> {
        let mut v: Vec<super::ops::SlotState> = (0..self.slots)
            .map(|s| super::ops::SlotState {
                held: self.held[s].len(),
                busy: self.busy[s],
                request: None,
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
        for s in pre.chain(self.seqs.drain(..)).collect::<Vec<_>>() {
            self.busy[s.slot] = false;
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
/// another.
fn plan(
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

/// The free slot (not `busy`) whose `held` tokens share the longest prefix
/// with `prompt`, and that length. Ties go to the least recently `used`
/// slot (then the lowest): a request, batched or alone, evicts the
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
        .max_by_key(|&(s, n)| (n, std::cmp::Reverse((used[s], s))))
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
