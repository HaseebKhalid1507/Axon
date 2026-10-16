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
//! nothing else runs goes the S1 way (`Engine::job`: `generate` on the
//! engine's own cache, with CUDA graphs and drafting); if another request
//! arrives meanwhile, its rows are copied into a slot (`BatchLm::adopt`) and
//! it carries on in the steps.
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
    t0: Instant,
}

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
/// A step's prompt tokens when nothing decodes (as hybrid.rs STREAM_BATCH).
const IDLE_CHUNK: usize = 4096;

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
    pub seqs: Vec<Seq<'t>>,
    /// Requests being prefilled, oldest first.
    pre: Vec<Prefill<'t>>,
    policy: ChunkPolicy,
    /// Prefill cost per token per position attended (+ DEPTH_K), µs:
    /// a moving average over chunks of at least the floor, once measured.
    cost: Option<f64>,
    /// The last decode step's time.
    decode_ms: f64,
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
        Batch {
            net,
            pool,
            slots,
            ctx,
            held: vec![Vec::new(); slots],
            busy: vec![false; slots],
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

    /// A free slot: the one holding the longest prefix of `prompt` (its
    /// length), or (no prompt) any.
    fn free_slot(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        (0..self.slots)
            .filter(|&s| !self.busy[s])
            .map(|s| {
                let n = self.held[s]
                    .iter()
                    .zip(prompt)
                    .take_while(|(a, b)| a == b)
                    .count();
                (s, n)
            })
            .max_by_key(|&(s, n)| (n, std::cmp::Reverse(s)))
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
        self.busy[slot] = true;
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
            t0: now,
        });
    }

    /// Prompt tokens this step may feed: with nothing decoding (nobody to
    /// stall) IDLE_CHUNK, so requests arriving meanwhile can still start
    /// between chunks; all of them without chunking; else `floor`, or with
    /// a time budget what fits in it beside the last decode step at the
    /// measured cost, at the oldest prompt's depth (the first chunk
    /// SEED_CHUNK), at least `floor`.
    fn budget(&self) -> usize {
        let ChunkPolicy { floor, ms } = self.policy;
        if floor == 0 {
            return usize::MAX;
        }
        if self.seqs.is_empty() {
            return IDLE_CHUNK.max(floor);
        }
        if ms == 0 {
            return floor;
        }
        match self.cost {
            None => SEED_CHUNK.max(floor),
            Some(c) => {
                // n tokens from depth d cost c * n * (d + n / 2 + K).
                let left_us = (ms as f64 - self.decode_ms).max(0.0) * 1e3;
                let dk = self.pre[0].fed as f64 + DEPTH_K;
                let n = (dk * dk + 2.0 * left_us / c).sqrt() - dk;
                (n as usize).max(floor)
            }
        }
    }

    /// Feed this step's prompt tokens (`budget`): first, shortest
    /// remainder first, the prompts (not the oldest) that finish within
    /// half of it, so a short prompt isn't held up behind a long one; then
    /// the oldest prompts first with the rest (a prompt whose remainder is
    /// at most a floor more than what is left fed whole: no last sliver of
    /// a pass of its own). The oldest gets at least half
    /// of each step: it can't starve. Prompts then in their slots join the
    /// running requests.
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
        let mut plan = vec![0; rem.len()];
        let mut short = budget / 2;
        let mut by_len: Vec<usize> = (1..rem.len()).collect();
        by_len.sort_by_key(|&i| rem[i]);
        for i in by_len {
            if rem[i] <= short {
                plan[i] = rem[i];
                short -= rem[i];
            }
        }
        let mut left = budget - plan.iter().sum::<usize>();
        for i in 0..rem.len() {
            if left == 0 {
                break;
            }
            let want = rem[i] - plan[i];
            let n = if want <= left.saturating_add(self.policy.floor) {
                want
            } else {
                left
            };
            plan[i] += n;
            left = left.saturating_sub(n);
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
        // Only chunks of a long prompt teach the cost (a short prompt's
        // feed costs more per token: a pass's fixed part).
        if n >= self.policy.floor.max(1) {
            self.cost = Some(self.cost.map_or(c, |a| 0.5 * a + 0.5 * c));
        }
        Ok(())
    }

    /// Take over a request `generate` was running on `cache` (the S1 path):
    /// its rows 0..hist.len() - 1 copied into a free slot.
    pub fn adopt(
        &mut self,
        mut seq: Seq<'t>,
        cache: &<Backend<'a> as crate::lm::Lm>::Cache,
    ) -> Result<(), String> {
        let n = seq.hist.len() - 1;
        let (slot, _) = self.free_slot(&[]).expect("adopt: a free slot");
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
