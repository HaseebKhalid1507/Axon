//! `axon serve`'s continuous batching (stage S2-lite, specs/r6-pick.md §4):
//! many requests decode together, one token each per step, over a pool of
//! KV slots on the GPU (`BatchLm`, cuda/qwen2/slots.rs).
//!
//! The engine thread runs a loop: take new requests, prefill one waiting
//! request's prompt into a free slot (its whole prompt, a step of its own),
//! then one step with a row per running request (its last token at its
//! position, the greedy pick or its sampling candidates back), each request's
//! next token picked on the host (`Sampler::pick`, its own seeded rng), its
//! text sent, and finished requests' slots freed. A request admitted while
//! nothing else runs goes the S1 way (`Engine::job`: `generate` on the
//! engine's own cache, with CUDA graphs and drafting); if another request
//! arrives meanwhile, its rows are copied into a slot (`BatchLm::adopt`) and
//! it carries on in the steps.
//!
//! Every row's results are the bits the one-sequence calls give
//! (`BatchLm`'s contract), and the host does per request what `generate`
//! does per token (a seeded pick from the same candidates, the same stop
//! rules), so a request's tokens are the same whatever else runs with it.

use super::api::Usage;
use super::{Event, Job, TextOut};
use crate::lm::{Backend, BatchLm, RowHead, RowOut, SeqRow};
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
    log: bool,
    generated: Arc<AtomicU64>,
}

impl<'a, 't> Batch<'a, 't> {
    pub fn new(
        net: &'a Backend<'a>,
        pool: <Backend<'a> as BatchLm>::Pool,
        slots: usize,
        ctx: usize,
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
            log,
            generated,
        }
    }

    pub fn idle(&self) -> bool {
        self.seqs.is_empty()
    }

    pub fn has_free(&self) -> bool {
        self.free() > 0
    }

    /// Free slots.
    pub fn free(&self) -> usize {
        self.busy.iter().filter(|b| !**b).count()
    }

    /// Whether a request of `prompt` tokens plus up to `n_gen` more fits a
    /// slot.
    pub fn fits(&self, prompt: usize, n_gen: usize) -> bool {
        prompt + n_gen <= self.ctx
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

    /// Prefill `job`'s prompt (all but its last token, which is its first
    /// row) into a free slot; it decodes from the next step on.
    pub fn admit(&mut self, job: Job, out: TextOut<'t>) -> Result<(), String> {
        let t0 = Instant::now();
        let p = &job.prompt;
        let last = p.len() - 1;
        let (slot, common) = self.free_slot(p).expect("admit: a free slot");
        let reuse = common.min(last);
        self.busy[slot] = true;
        self.held[slot].truncate(reuse);
        if reuse < last {
            self.net
                .feed_slot(&mut self.pool, slot, &p[reuse..last], reuse)?;
        }
        self.held[slot] = p[..last].to_vec();
        let sampler = job.sample.map(Sampler::new);
        self.seqs.push(Seq {
            hist: p.clone(),
            ids: Vec::new(),
            sampler,
            out,
            cached: reuse,
            prompt_ms: t0.elapsed().as_secs_f64() * 1e3,
            t1: Instant::now(),
            slot,
            job,
        });
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
        while i < self.seqs.len() {
            if self.seqs[i].job.cancel.load(Ordering::Relaxed) {
                let s = self.seqs.swap_remove(i);
                self.finish(s, End::Gone);
            } else {
                i += 1;
            }
        }
    }

    /// One step: a row per running request, then each one's next token.
    pub fn step(&mut self) -> Result<(), String> {
        self.reap();
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
        let outs = self.net.step(&mut self.pool, &rows)?;
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

    /// A device error: every running request gets it.
    pub fn fail_all(&mut self, e: &str) {
        for s in self.seqs.drain(..) {
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
