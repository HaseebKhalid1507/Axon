//! `axon serve`'s continuous batching on the GPU (stage S2-lite of
//! specs/r4-serve.md, brief specs/r6-pick.md §4): a pool of slots and the
//! multi-sequence step.
//!
//! The pool is one cache of `slots` x `ctx` positions: slot s holds a
//! sequence's rows s * ctx.. (`kv_args_at` gives a slot's layer pointers).
//! A prompt is fed into its slot by the batched passes as `feed_batch`
//! (`Place::Seq` with the slot's first row); a step runs one token of each
//! of many sequences together (`Place::Rows`: per token its position and
//! cache row, prefill.cu "Placement"), its attention the decode kernel per
//! query on its own slot's rows. Every kernel does per token what the
//! one-sequence pass does on the same position and rows, so each row's
//! cache rows, logits, picks and candidates are the bits `Lm` gives that
//! sequence alone (`forward_argmax`, `forward_cands`), whatever else is in
//! the step: the server's output doesn't depend on its load.
//!
//! The step's head: the greedy rows take `argmax_b` (verify_argmax's
//! pick), the sampled ones sample.cu with a penalty per row (S's
//! SP_PER_ROW layout) and the step's largest k (a row wanting fewer keeps
//! the first ones: the k best are a prefix of any larger k's); past
//! CAND_MAX (top_k off) a row's logits come back and the host selects, as
//! `forward_cands` does.
//!
//! A step's kernels depend only on its row counts (rows, greedy, sampled
//! on the device): each such shape is captured into a CUDA graph on first
//! use (its inputs copied in before it runs), as the decode step's is, so
//! a step costs one launch (AXON_CUDA_GRAPHS=0: kernel by kernel).

use super::batch::{BatchIn, HEAD_SUB, Place, RowAt, passes};
use super::*;
use crate::lm::{BatchLm, RowHead, RowOut, SeqRow, StepError};
use std::collections::HashMap;
use std::rc::Rc;

/// sample.cu SP_PER_ROW: S[1] when each row has its own penalty.
const SP_PER_ROW: u32 = u32::MAX;

/// A pool of `slots` sequences of up to `ctx` positions each.
pub struct CudaPool<'g> {
    c: CudaCache<'g>,
    slots: usize,
    ctx: usize,
    /// Most rows per step (logits scratch rows).
    rows_max: usize,
    /// Captured steps by (rows, greedy rows, + device-sampled rows): a
    /// step's kernels and their arguments depend on nothing else (its
    /// tokens, places and S are copied in before the graph runs).
    /// Each with the step count of its last use: the least recently used
    /// one goes when one more is needed past MAX_GRAPHS.
    graphs: HashMap<(usize, usize, usize), (Graph<'g>, u64)>,
    steps: u64,
}

/// Captured steps kept at most.
const MAX_GRAPHS: usize = 64;

impl<'g> CudaQwen2<'g> {
    /// VRAM a pool of `slots` x `ctx` (`pool_bytes_kv`) and the
    /// single-request cache, a view of a slot (`slot_view`: its scratch,
    /// no KV of its own), take with caches of type `kv` (the model's, or
    /// serve's `--kv q8` hint).
    pub fn serve_bytes_kv(&self, kv: CacheKv, slots: usize, ctx: usize) -> usize {
        self.cache_bytes_kv(kv, ctx) - self.kv_cache_bytes(kv, ctx)
            + self.pool_bytes_kv(kv, slots, ctx)
    }

    /// A cache of slot `slot`'s rows of `p` (`Lm`'s calls on it read and
    /// write that slot's KV bytes, its own scratch and graphs beside), or
    /// why there's no room for its scratch. The single-request path of
    /// `axon serve` runs in a slot this way (`retarget_view` picks which,
    /// per request): no cache beside the pool, and
    /// its rows are the slot's for the batch to take over (`adopt_cache`
    /// copies nothing).
    pub fn slot_view(&self, p: &CudaPool<'g>, slot: usize) -> Result<CudaCache<'g>, String> {
        if slot >= p.slots {
            return Err(format!("slot {slot} of a {}-slot pool", p.slots));
        }
        if p.c.tab.is_some() {
            return Err("slot_view of a paged pool (a view is plain rows)".into());
        }
        let shared = (p.c.kv.clone(), p.c.rows, slot * p.ctx);
        self.make_cache(p.ctx, p.c.mode, p.ctx, Some(shared))
    }

    /// VRAM a pool of `slots` x `ctx` takes with caches of type `kv`
    /// (`pool` checks it is free).
    fn pool_bytes_kv(&self, kv: CacheKv, slots: usize, ctx: usize) -> usize {
        self.cache_bytes_kv(kv, ctx) - self.kv_cache_bytes(kv, ctx)
            + self.kv_cache_bytes(kv, slots * ctx)
            + slots.saturating_sub(HEAD_SUB) * self.vocab * 4
            + self.batch_for(ctx) * 8
    }

    /// A pool of `slots` sequences of up to `ctx` positions, in the model's
    /// KV type, or why there's no room for it.
    pub fn pool(&self, slots: usize, ctx: usize) -> Result<CudaPool<'g>, String> {
        if slots == 0 || ctx == 0 {
            return Err("a pool needs at least one slot and position".into());
        }
        let max_ctx = slots
            .checked_mul(ctx)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| format!("{slots} slots of {ctx} positions: too many"))?;
        if self.first > 0 {
            return Err("no pool on a split model's GPU half".into());
        }
        let mut c = self.try_cache_at(max_ctx, self.kv, ctx)?;
        let g = self.gpu;
        let b = c.b.as_mut().expect("batch scratch");
        let rows_max = slots.max(HEAD_SUB);
        if rows_max > b.max {
            return Err(format!(
                "{slots} slots: more than a pass's {} tokens",
                b.max
            ));
        }
        b.rp = Some((g.alloc(b.max * 8)?, g.host_alloc(b.max * 8)?));
        if rows_max > HEAD_SUB {
            b.logits = g.alloc(rows_max * self.vocab * 4)?;
        }
        b.logits_rows = true;
        // S for every row with as many penalty ids as a row can have (the
        // distinct tokens of a window of at most ctx), made up front: it is
        // never reallocated (the graphs hold its address).
        self.samp_bufs(&mut c, 8 + rows_max * (2 + ctx.min(self.vocab)), rows_max)?;
        Ok(CudaPool {
            c,
            slots,
            ctx,
            rows_max,
            graphs: HashMap::new(),
            steps: 0,
        })
    }

    /// `BatchLm::feed_slot`: the passes of `feed_batch`, into the slot.
    pub fn feed_slot(
        &self,
        p: &mut CudaPool<'g>,
        slot: usize,
        tokens: &[u32],
        start: usize,
    ) -> Result<(), String> {
        if slot >= p.slots || start + tokens.len() > p.ctx {
            return Err(format!(
                "feed_slot: slot {slot} positions {start}..{} (pool {} x {})",
                start + tokens.len(),
                p.slots,
                p.ctx
            ));
        }
        if let Some(t) = tokens.iter().find(|&&t| t as usize >= self.vocab) {
            return Err(format!("token {t} out of range"));
        }
        let max = p.pass_max();
        let base = slot * p.ctx;
        for r in passes(tokens.len(), max) {
            let input = BatchIn::Tokens(&tokens[r.clone()]);
            let place = Place::Seq {
                start: start + r.start,
                base,
            };
            self.pass_at(&mut p.c, input, place, Head::No, None, 0, &mut |_| Ok(()))?;
        }
        self.gpu.sync()
    }

    /// `BatchLm::step`.
    pub fn step_rows(
        &self,
        p: &mut CudaPool<'g>,
        rows: &[SeqRow],
    ) -> Result<Vec<RowOut>, StepError> {
        let nb = rows.len();
        if nb == 0 {
            return Ok(Vec::new());
        }
        let bad = |e: String| Err(StepError::Input(e));
        if nb > p.rows_max {
            return bad(format!("{nb} rows: the pool takes {}", p.rows_max));
        }
        for r in rows {
            if r.slot >= p.slots || r.pos >= p.ctx || r.token as usize >= self.vocab {
                return bad(format!(
                    "row: token {} at {} of slot {} (pool {} x {})",
                    r.token, r.pos, r.slot, p.slots, p.ctx
                ));
            }
            if let RowHead::Sample { spec, recent } = r.head {
                if recent.last() != Some(&r.token) {
                    return bad("a sampled row's recent must end with its token".into());
                }
                if spec.k == 0 {
                    return bad("a sampled row wants at least one candidate".into());
                }
            }
        }
        self.step_checked(p, rows).map_err(StepError::Device)
    }

    /// `step_rows` after its checks.
    fn step_checked(&self, p: &mut CudaPool<'g>, rows: &[SeqRow]) -> Result<Vec<RowOut>, String> {
        let nb = rows.len();
        // Greedy rows first, then the ones sampled on the device, then
        // those selected on the host (the order changes no row's bits).
        let kind = |r: &SeqRow| match r.head {
            RowHead::Argmax => 0,
            RowHead::Sample { spec, .. } if spec.k <= CAND_MAX => 1,
            RowHead::Sample { .. } => 2,
        };
        let mut order: Vec<usize> = (0..nb).collect();
        order.sort_by_key(|&i| kind(&rows[i]));
        let ng = rows.iter().filter(|r| kind(r) == 0).count();
        let ns = ng + rows.iter().filter(|r| kind(r) == 1).count();
        let tokens: Vec<u32> = order.iter().map(|&i| rows[i].token).collect();
        let at: Vec<RowAt> = order
            .iter()
            .map(|&i| RowAt {
                pos: rows[i].pos as u32,
                base: (rows[i].slot * p.ctx) as u32,
            })
            .collect();
        let sampled: Vec<(CandSpec, &[u32])> = order[ng..ns]
            .iter()
            .map(|&i| match rows[i].head {
                RowHead::Sample { spec, recent } => (spec, recent),
                RowHead::Argmax => unreachable!(),
            })
            .collect();
        let kmax = sampled.iter().map(|s| s.0.k).max().unwrap_or(1);
        if !sampled.is_empty() {
            self.samp_upload_rows(&mut p.c, &sampled, kmax)?;
        }
        let tab = p.c.tab.as_ref().map(|t| t.len() / 4);
        let b = p.c.b.as_mut().expect("batch scratch");
        self.load_input(b, BatchIn::Tokens(&tokens))?;
        b.upload_rows(&at, tab)?;
        // The kernels: the step's graph (made on its first use).
        let c = &mut p.c;
        let mut pass = || {
            self.pass_at(
                c,
                BatchIn::Loaded(nb),
                Place::Rows,
                Head::Sample,
                Some((ng, ns)),
                0,
                &mut |_| Ok(()),
            )
        };
        if self.graphs {
            let key = (nb, ng, ns);
            p.steps += 1;
            if !p.graphs.contains_key(&key) {
                if p.graphs.len() >= MAX_GRAPHS {
                    let lru = p
                        .graphs
                        .iter()
                        .min_by_key(|(_, (_, t))| *t)
                        .map(|(k, _)| *k);
                    p.graphs.remove(&lru.unwrap());
                }
                let g = self.gpu.capture(pass)?;
                p.graphs.insert(key, (g, 0));
            }
            let (g, used) = p.graphs.get_mut(&key).unwrap();
            *used = p.steps;
            // SAFETY: the graph's buffers are the pool's and the model's,
            // both borrowed here; the pool owns the graph, and the buffers
            // it captured are never reallocated while it lives (S is made
            // at its largest in `pool`).
            unsafe { g.launch()? };
        } else {
            pass()?;
        }
        let mut out: Vec<Option<RowOut>> = vec![None; nb];
        let b = p.c.b.as_mut().expect("batch scratch");
        if ng > 0 {
            for (z, pick) in b.picks(ng)?.into_iter().enumerate() {
                out[order[z]] = Some(RowOut::Pick(pick));
            }
        }
        if ns > ng {
            let cs = self.samp_read(&mut p.c, kmax, self.vocab, ns - ng)?;
            for (j, mut c) in cs.into_iter().enumerate() {
                c.truncate(sampled[j].0.k);
                out[order[ng + j]] = Some(RowOut::Cands(c));
            }
        }
        let v = self.vocab;
        for (z, &i) in order.iter().enumerate().skip(ns) {
            let RowHead::Sample { spec, recent } = rows[i].head else {
                unreachable!()
            };
            let b = p.c.b.as_ref().expect("batch scratch");
            p.c.host_out.copy_from(&b.logits, z * v * 4, v * 4)?;
            let hb = &p.c.host_out.bytes_mut()?[..v * 4]; // waits for the step and copy
            let mut l: Vec<f32> = hb
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect();
            out[i] = Some(RowOut::Cands(sample::cands(&mut l, &spec, recent)));
        }
        Ok(out.into_iter().map(|o| o.expect("every row")).collect())
    }

    /// S for a step's sampled rows (in their order): the largest k,
    /// per-row penalties (sample.cu SP_PER_ROW) and penalty ids (row j's:
    /// the distinct tokens among the last `last_n` of its `recent`, as
    /// `sample::cands` takes them).
    fn samp_upload_rows(
        &self,
        c: &mut CudaCache<'g>,
        rows: &[(CandSpec, &[u32])],
        kmax: usize,
    ) -> Result<(), String> {
        assert!((1..=CAND_MAX).contains(&kmax));
        let nr = rows.len();
        let ids: Vec<Vec<u32>> = rows
            .iter()
            .map(|(spec, recent)| {
                if spec.penalty != 1.0 && spec.last_n > 0 {
                    sample::penalty_ids(recent, spec.last_n)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let need = 4 + 4 + 2 * nr + ids.iter().map(Vec::len).sum::<usize>();
        let s = c.samp.as_mut().expect("a pool's sampling scratch");
        // Never past it: rows <= rows_max, ids per row <= min(ctx, vocab).
        if need > s.cap || nr > s.rows {
            return Err(format!(
                "S of {need} u32s for {nr} rows: past the pool's scratch"
            ));
        }
        let hb = s.host_s.bytes_mut()?;
        let mut put = |i: usize, v: u32| hb[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        s.params = None;
        put(0, 0);
        put(1, 0);
        // S, from u32 4 on.
        put(4, kmax as u32);
        put(5, SP_PER_ROW);
        put(6, nr as u32);
        put(7, 0);
        let base = 8 + 2 * nr;
        let mut off = 0;
        for (j, w) in ids.iter().enumerate() {
            let pen = if w.is_empty() { 1.0 } else { rows[j].0.penalty };
            put(8 + nr + j, pen.to_bits());
            for (i, &id) in w.iter().enumerate() {
                if id as usize >= self.vocab {
                    return Err(format!("token {id} out of range"));
                }
                put(base + off + i, id);
            }
            off += w.len();
            put(8 + j, off as u32);
        }
        s.host_s.copy_to(&s.s, 0, need * 4)
    }

    /// Point `c`, a view of `p` (`slot_view`), at slot `slot`'s rows. Its
    /// captured steps hold the old slot's addresses: they go (the next
    /// steps are captured again; the same kernels, so the same bits).
    pub fn retarget_view(
        &self,
        p: &CudaPool<'g>,
        c: &mut CudaCache<'g>,
        slot: usize,
    ) -> Result<(), String> {
        if !Rc::ptr_eq(&c.kv, &p.c.kv) || slot >= p.slots {
            return Err(format!("retarget: not a view of this pool, or slot {slot}"));
        }
        if c.tab.is_some() || p.c.tab.is_some() {
            return Err("retarget: a paged cache or pool (a view is plain rows)".into());
        }
        let row0 = slot * p.ctx;
        if c.row0 != row0 {
            c.row0 = row0;
            c.drop_graphs();
        }
        Ok(())
    }

    /// `BatchLm::adopt`: positions 0..n of `src` (a cache of this model,
    /// the pool's KV type) into slot `slot`, enqueued; the bytes copied (0:
    /// `src` is that slot's view).
    pub fn adopt_cache(
        &self,
        p: &mut CudaPool<'g>,
        slot: usize,
        src: &CudaCache<'g>,
        n: usize,
    ) -> Result<usize, String> {
        if slot >= p.slots || n > p.ctx || n > src.max_ctx {
            return Err(format!(
                "adopt: {n} positions into slot {slot} (pool {} x {})",
                p.slots, p.ctx
            ));
        }
        if src.mode != p.c.mode || src.net != self.id {
            return Err("adopt: a cache of another model or KV type".into());
        }
        if src.tab.is_some() || p.c.tab.is_some() {
            return Err("adopt: a paged cache (row copies are of plain rows)".into());
        }
        if Rc::ptr_eq(&src.kv, &p.c.kv) {
            // A view of the pool (`slot_view`): its rows are in place if
            // it is this slot's, and copying another slot's view's would
            // be a copy within the pool, which nothing needs.
            return match src.row0 == slot * p.ctx {
                true => Ok(0),
                false => Err(format!("adopt: another slot's view into slot {slot}")),
            };
        }
        let kvd = self.kv_dim();
        // (bytes per row, per layer K/V array rows) of each region: the
        // rows, then (q8) their scales.
        let regions: Vec<(usize, usize)> = match p.c.mode {
            CacheKv::F16 => vec![(kvd * 2, 0)],
            CacheKv::Q8 { group } => vec![(kvd, 0), (kvd / group * 4, 1)],
        };
        let nl = self.layers.len();
        // The copy reads src's rows from row 0: a plain cache, or a view of
        // another pool's slot 0.
        if src.row0 != 0 {
            return Err("adopt: a view of another pool's slot past 0".into());
        }
        let (ns, np) = (src.rows, p.c.rows);
        let mut src_at = 0;
        let mut dst_at = 0;
        let mut copied = 0;
        for (row, _) in regions {
            for a in 0..2 * nl {
                src.kv.copy_into(
                    src_at + a * ns * row,
                    &p.c.kv,
                    dst_at + (a * np + slot * p.ctx) * row,
                    n * row,
                )?;
                copied += n * row;
            }
            src_at += 2 * nl * ns * row;
            dst_at += 2 * nl * np * row;
        }
        Ok(copied)
    }
}

impl<'g> BatchLm for CudaQwen2<'g> {
    type Pool = CudaPool<'g>;

    fn new_pool(&self, slots: usize, ctx: usize) -> Result<CudaPool<'g>, String> {
        self.pool(slots, ctx)
    }

    fn feed_slot(
        &self,
        pool: &mut CudaPool<'g>,
        slot: usize,
        tokens: &[u32],
        start: usize,
    ) -> Result<(), String> {
        CudaQwen2::feed_slot(self, pool, slot, tokens, start)
    }

    fn step(&self, pool: &mut CudaPool<'g>, rows: &[SeqRow]) -> Result<Vec<RowOut>, StepError> {
        self.step_rows(pool, rows)
    }

    fn slot_cache(&self, pool: &CudaPool<'g>, slot: usize) -> Result<CudaCache<'g>, String> {
        self.slot_view(pool, slot)
    }

    fn move_slot_cache(
        &self,
        pool: &CudaPool<'g>,
        cache: &mut CudaCache<'g>,
        slot: usize,
    ) -> Result<(), String> {
        self.retarget_view(pool, cache, slot)
    }

    fn adopt(
        &self,
        pool: &mut CudaPool<'g>,
        slot: usize,
        cache: &CudaCache<'g>,
        n: usize,
    ) -> Result<(), String> {
        self.adopt_cache(pool, slot, cache, n).map(|_| ())
    }

    fn pass_max(&self, pool: &CudaPool<'g>) -> usize {
        pool.pass_max()
    }
}

impl CudaPool<'_> {
    /// `BatchLm::pass_max`: the batch scratch's tokens (PREFILL_BATCH, or
    /// PLAN_BATCH when the bigger passes don't fit; `feed_slot` splits by it).
    pub fn pass_max(&self) -> usize {
        self.c.b.as_ref().expect("batch scratch").max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::Kernels;
    use crate::gguf::Model;
    use crate::sample::{SampleArgs, Sampler};

    /// Slot `slot`'s first `n` positions of every layer, as bytes (f16 K
    /// and V rows; q8 rows, then their scales).
    fn slot_bytes(net: &CudaQwen2, c: &CudaCache, base: usize, n: usize) -> Vec<u8> {
        let all: Vec<u8> = c.kv.read().unwrap();
        let kvd = net.kv_dim();
        let rows: Vec<usize> = match c.mode {
            CacheKv::F16 => vec![kvd * 2],
            CacheKv::Q8 { group } => vec![kvd, kvd / group * 4],
        };
        let mut out = Vec::new();
        let mut at = 0;
        for row in rows {
            for a in 0..2 * net.n_layers {
                let s = at + (a * c.rows + base) * row;
                out.extend_from_slice(&all[s..s + n * row]);
            }
            at += 2 * net.n_layers * c.rows * row;
        }
        out
    }

    /// Prompt-like ids: a mix of a few repeats (the penalty bites) and
    /// spread-out ids.
    fn ids(n: usize, seed: u32, vocab: usize) -> Vec<u32> {
        let mut s = seed | 1;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                if i % 5 == 3 {
                    (s % 50) + 100
                } else {
                    s % (vocab as u32 - 300) + 200
                }
            })
            .collect()
    }

    fn models() -> Vec<Model> {
        [crate::testutil::model(), crate::testutil::qwen3_model()]
            .into_iter()
            .flatten()
            .collect()
    }

    fn sampler(top_k: usize, penalty: f32, seed: u64) -> Sampler {
        let a = SampleArgs {
            temp: Some(0.9),
            top_k: Some(top_k),
            repeat_penalty: Some(penalty),
            repeat_last_n: Some(48),
            seed: Some(seed),
            ..Default::default()
        };
        Sampler::new(a.resolve(0.8).unwrap())
    }

    /// Checkpoint 1: a step of nb rows of one slot (consecutive positions:
    /// a verify pass in a pool) == `verify_cands` / `verify_argmax` on a
    /// plain cache, candidates and picks bit for bit, and the slot's KV
    /// bytes == the cache's; on f16 and q8 caches, qwen2 and qwen3.
    #[test]
    fn pool_rows_of_one_slot_match_verify() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let km = Kernels::module(&g).unwrap();
        if !km.mma() {
            return;
        }
        for m in models() {
            for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
                let ctx = 160;
                let net = CudaQwen2::with_kv(&g, &km, &m, ctx, kv, false).unwrap();
                let v = net.vocab;
                let prompt = ids(61, 7, v);
                for (nb, slot) in [(1usize, 0usize), (5, 2), (16, 1)] {
                    let toks = ids(nb, 99 + nb as u32, v);
                    let start = prompt.len();
                    let mut recent = prompt.clone();
                    recent.extend_from_slice(&toks);
                    let spec = CandSpec {
                        k: 40,
                        penalty: 1.2,
                        last_n: 64,
                    };
                    // Reference: a plain cache.
                    let mut c = net.new_cache(ctx);
                    net.feed_batch(&prompt, 0, &mut c);
                    let want_c = net.verify_cands(&toks, start, &mut c, &spec, &recent);
                    let want_g = net.verify_argmax(&toks, start, &mut c);
                    let want_kv = slot_bytes(&net, &c, 0, start + nb);
                    // The pool: the same prompt in `slot`, then one step.
                    let mut p = net.pool(3, ctx).unwrap();
                    net.feed_slot(&mut p, slot, &prompt, 0).unwrap();
                    let n = recent.len();
                    let rows: Vec<SeqRow> = (0..nb)
                        .map(|b| SeqRow {
                            token: toks[b],
                            pos: start + b,
                            slot,
                            head: RowHead::Sample {
                                spec,
                                recent: &recent[..n - nb + b + 1],
                            },
                        })
                        .collect();
                    let got = net.step_rows(&mut p, &rows).unwrap();
                    let what = format!("{} {kv:?} nb {nb} slot {slot}", m.arch);
                    for (b, o) in got.iter().enumerate() {
                        assert_eq!(*o, RowOut::Cands(want_c[b].clone()), "{what} row {b}");
                        if let RowOut::Cands(c) = o {
                            let bits = |c: &[Cand]| -> Vec<(u32, u32)> {
                                c.iter().map(|c| (c.logit.to_bits(), c.id)).collect()
                            };
                            assert_eq!(bits(c), bits(&want_c[b]), "{what} row {b} bits");
                        }
                    }
                    let base = slot * ctx;
                    assert!(
                        slot_bytes(&net, &p.c, base, start + nb) == want_kv,
                        "{what}: KV bytes"
                    );
                    // Greedy rows (argmax_b), the same positions again (rows
                    // rewritten with the same bits).
                    let rows: Vec<SeqRow> = rows
                        .iter()
                        .map(|r| SeqRow {
                            head: RowHead::Argmax,
                            ..*r
                        })
                        .collect();
                    let got = net.step_rows(&mut p, &rows).unwrap();
                    let want: Vec<RowOut> = want_g.iter().map(|&t| RowOut::Pick(t)).collect();
                    assert_eq!(got, want, "{what}: greedy");
                    assert!(
                        slot_bytes(&net, &p.c, base, start + nb) == want_kv,
                        "{what}: KV bytes after the greedy step"
                    );
                }
            }
        }
    }

    /// One request's head in the slot tests.
    #[derive(Clone, Copy)]
    enum Kind {
        Greedy,
        /// top_k, repeat penalty, seed.
        Sampled(usize, f32, u64),
    }

    /// The request's tokens and per-step results run alone through `Lm`
    /// (`forward_argmax` / `forward_cands`, the decode step), and the
    /// cache's KV bytes after it.
    fn solo(
        net: &CudaQwen2,
        ctx: usize,
        prompt: &[u32],
        kind: Kind,
        steps: usize,
    ) -> (Vec<RowOut>, Vec<u32>, Vec<u8>) {
        let mut c = net.new_cache(ctx);
        let n = prompt.len();
        if n > 1 {
            net.feed_batch(&prompt[..n - 1], 0, &mut c);
        }
        let mut hist = prompt.to_vec();
        let mut sm = match kind {
            Kind::Greedy => None,
            Kind::Sampled(k, pen, seed) => Some(sampler(k, pen, seed)),
        };
        let mut outs = Vec::new();
        for _ in 0..steps {
            let (tok, pos) = (hist[hist.len() - 1], hist.len() - 1);
            let (o, next) = match sm.as_mut() {
                None => {
                    let t = net.forward_argmax(tok, pos, &mut c);
                    (RowOut::Pick(t), t)
                }
                Some(s) => {
                    let spec = s.cfg.spec();
                    let cs = net.forward_cands(tok, pos, &mut c, &spec, &hist);
                    let t = s.pick(&cs);
                    (RowOut::Cands(cs), t)
                }
            };
            outs.push(o);
            hist.push(next);
        }
        let kv = slot_bytes(net, &c, 0, hist.len() - 1);
        (outs, hist, kv)
    }

    /// Checkpoints 2 and 3: requests of different lengths in slots taken in
    /// a shuffled order, rows in a different shuffled order every step,
    /// greedy and sampled (penalties and k per row, one with top_k off: the
    /// host selects) mixed, one request handed over from a plain cache
    /// (`adopt`) partway, one joining late: every request's results and
    /// tokens == running it alone, and each slot's KV bytes == its solo
    /// cache's; f16 and q8, qwen2 and qwen3.
    #[test]
    fn pool_steps_match_solo_in_any_order() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let km = Kernels::module(&g).unwrap();
        if !km.mma() {
            return;
        }
        let kinds = [
            Kind::Greedy,
            Kind::Sampled(40, 1.1, 1),
            Kind::Sampled(1, 1.0, 2),
            Kind::Sampled(256, 1.3, 3),
            Kind::Greedy,
            Kind::Sampled(0, 1.05, 4),
        ];
        let lens = [5usize, 37, 70, 1, 100, 23];
        let steps = 24;
        for m in models() {
            for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
                let ctx = 160;
                let net = CudaQwen2::with_kv(&g, &km, &m, ctx, kv, false).unwrap();
                let v = net.vocab;
                let nr = kinds.len();
                let prompts: Vec<Vec<u32>> =
                    (0..nr).map(|i| ids(lens[i], 1000 + i as u32, v)).collect();
                let refs: Vec<_> = (0..nr)
                    .map(|i| solo(&net, ctx, &prompts[i], kinds[i], steps))
                    .collect();
                // The handed-over request runs alone in a plain cache
                // (adopt copies its rows), or in a view of its slot
                // (`slot_view`, as serve's: adopt copies nothing).
                for via_view in [false, true] {
                    let slots = nr + 1;
                    let mut p = net.pool(slots, ctx).unwrap();
                    let mut perm = 0x9e37_79b9u32;
                    let mut rnd = |n: usize| {
                        perm ^= perm << 13;
                        perm ^= perm >> 17;
                        perm ^= perm << 5;
                        perm as usize % n
                    };
                    // Slot of each request: a shuffle of the slots.
                    let mut slot_of: Vec<usize> = (0..slots).collect();
                    for i in (1..slots).rev() {
                        slot_of.swap(i, rnd(i + 1));
                    }
                    // Request 1 starts alone in a plain cache (5 steps), then
                    // is adopted; request 5 joins at step 9.
                    let (handed, late, handed_at, late_at) = (1usize, 5usize, 5usize, 9usize);
                    let mut hist: Vec<Vec<u32>> = prompts.clone();
                    let mut sms: Vec<Option<Sampler>> = kinds
                        .iter()
                        .map(|k| match *k {
                            Kind::Greedy => None,
                            Kind::Sampled(k, pen, seed) => Some(sampler(k, pen, seed)),
                        })
                        .collect();
                    let mut done = vec![0usize; nr];
                    // The handed-over request's first steps, alone.
                    let mut c = match via_view {
                        false => net.new_cache(ctx),
                        true => net.slot_view(&p, slot_of[handed]).unwrap(),
                    };
                    {
                        let h = &mut hist[handed];
                        let n = h.len();
                        net.feed_batch(&h[..n - 1], 0, &mut c);
                        for _ in 0..handed_at {
                            let (tok, pos) = (h[h.len() - 1], h.len() - 1);
                            let s = sms[handed].as_mut().unwrap();
                            let spec = s.cfg.spec();
                            let cs = net.forward_cands(tok, pos, &mut c, &spec, h);
                            assert_eq!(RowOut::Cands(cs.clone()), refs[handed].0[done[handed]]);
                            h.push(s.pick(&cs));
                            done[handed] += 1;
                        }
                    }
                    for i in 0..nr {
                        if i != handed && i != late {
                            let n = prompts[i].len();
                            net.feed_slot(&mut p, slot_of[i], &prompts[i][..n - 1], 0)
                                .unwrap();
                        }
                    }
                    for step in 0..steps + late_at {
                        if step == 2 {
                            let n = hist[handed].len() - 1;
                            let copied = net.adopt_cache(&mut p, slot_of[handed], &c, n).unwrap();
                            assert_eq!(copied == 0, via_view, "adopt copies iff not a view");
                        }
                        if step == late_at {
                            let n = prompts[late].len();
                            net.feed_slot(&mut p, slot_of[late], &prompts[late][..n - 1], 0)
                                .unwrap();
                        }
                        let mut live: Vec<usize> = (0..nr)
                            .filter(|&i| done[i] < steps)
                            .filter(|&i| i != handed || step >= 2)
                            .filter(|&i| i != late || step >= late_at)
                            .collect();
                        for i in (1..live.len()).rev() {
                            live.swap(i, rnd(i + 1));
                        }
                        if live.is_empty() {
                            break;
                        }
                        let specs: Vec<Option<CandSpec>> = live
                            .iter()
                            .map(|&i| sms[i].as_ref().map(|s| s.cfg.spec()))
                            .collect();
                        let rows: Vec<SeqRow> = live
                            .iter()
                            .zip(&specs)
                            .map(|(&i, spec)| {
                                let h = &hist[i];
                                SeqRow {
                                    token: h[h.len() - 1],
                                    pos: h.len() - 1,
                                    slot: slot_of[i],
                                    head: match spec {
                                        None => RowHead::Argmax,
                                        Some(spec) => RowHead::Sample {
                                            spec: *spec,
                                            recent: &h[h.len() - spec.last_n.clamp(1, h.len())..],
                                        },
                                    },
                                }
                            })
                            .collect();
                        let outs = net.step_rows(&mut p, &rows).unwrap();
                        for (&i, o) in live.iter().zip(outs) {
                            let what = format!(
                                "{} {kv:?} view {via_view} request {i} step {}",
                                m.arch, done[i]
                            );
                            assert_eq!(o, refs[i].0[done[i]], "{what}");
                            let next = match (&o, sms[i].as_mut()) {
                                (RowOut::Pick(t), None) => *t,
                                (RowOut::Cands(cs), Some(s)) => s.pick(cs),
                                _ => panic!("{what}: a result of another head"),
                            };
                            hist[i].push(next);
                            done[i] += 1;
                        }
                    }
                    for i in 0..nr {
                        let what = format!("{} {kv:?} view {via_view} request {i}", m.arch);
                        assert_eq!(done[i], steps, "{what}");
                        assert_eq!(hist[i], refs[i].1, "{what}: tokens");
                        let n = hist[i].len() - 1;
                        assert!(
                            slot_bytes(&net, &p.c, slot_of[i] * ctx, n) == refs[i].2,
                            "{what}: KV bytes"
                        );
                    }
                }
            }
        }
    }

    /// `slot_view_matches_a_cache`'s calls on cache `c`: its picks.
    fn run<'g>(net: &CudaQwen2<'g>, prompt: &[u32], c: &mut CudaCache<'g>) -> Vec<u32> {
        let mut out = Vec::new();
        net.feed_batch(&prompt[..60], 0, c);
        let mut h = prompt[..61].to_vec();
        for _ in 0..12 {
            let t = net.forward_argmax(h[h.len() - 1], h.len() - 1, c);
            out.push(t);
            h.push(t);
        }
        // A drafting pass: the last pick, then guesses (most of them
        // wrong).
        let mut guess = vec![h[h.len() - 1]];
        guess.extend_from_slice(&prompt[61..68]);
        let v = net.verify_argmax(&guess, h.len() - 1, c);
        assert!(
            v.iter().zip(&guess[1..]).any(|(a, b)| a != b),
            "a guess is rejected"
        );
        out.extend_from_slice(&v);
        // Back to the prompt's prefix (reuse of 40 rows), the rest fed
        // again.
        net.feed_batch(&prompt[40..89], 40, c);
        out.push(net.forward_argmax(prompt[89], 89, c));
        out
    }

    /// A slot view is never paged: `set_tab` refuses slot 0's view (row0 0)
    /// as any other's and a pool with a live view, `retarget_view` refuses a
    /// view with a table, and `slot_view` a paged pool (a table and a view
    /// offset never both apply).
    #[test]
    fn paged_views_refused() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let km = Kernels::module(&g).unwrap();
        let net = CudaQwen2::with_kv(&g, &km, &m, 128, CacheKv::F16, false).unwrap();
        let mut p = net.pool(2, 128).unwrap();
        for slot in [0, 1] {
            let mut v = net.slot_view(&p, slot).unwrap();
            let t = g.upload(&[0u32, 1]).unwrap();
            let set = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.set_tab(Some(t))));
            assert!(set.is_err(), "set_tab on slot {slot}'s view");
            v.tab = Some(g.upload(&[0u32, 1]).unwrap());
            assert!(
                net.retarget_view(&p, &mut v, 1 - slot).is_err(),
                "retarget of a paged view"
            );
        }
        {
            let _v = net.slot_view(&p, 0).unwrap();
            let t = g.upload(&[0u32, 1, 2, 3]).unwrap();
            let set =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.c.set_tab(Some(t))));
            assert!(set.is_err(), "set_tab on a pool with a live view");
        }
        p.c.set_tab(Some(g.upload(&[0u32, 1, 2, 3]).unwrap()));
        assert!(net.slot_view(&p, 0).is_err(), "slot_view of a paged pool");
    }

    /// A view of a slot (`slot_view`, serve's single-request cache) runs
    /// the one-sequence calls as a cache of its own: prefill, the decode
    /// step (its graphs), a verify pass with rejected guesses (drafting),
    /// and a prefix reuse fed again at a position: the same picks, and the
    /// slot's KV bytes == the plain cache's; the other slots stay as they
    /// were; `adopt_cache` of it into its slot copies nothing, into another
    /// slot is refused. f16 and q8, qwen2 and qwen3.
    #[test]
    fn slot_view_matches_a_cache() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let km = Kernels::module(&g).unwrap();
        for m in models() {
            for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
                let ctx = 192;
                let net = CudaQwen2::with_kv(&g, &km, &m, ctx, kv, false).unwrap();
                let prompt = ids(90, 77, net.vocab);
                let mut plain = net.new_cache(ctx);
                let want = run(&net, &prompt, &mut plain);
                let want_kv = slot_bytes(&net, &plain, 0, 90);
                let slots = 3;
                let mut p = net.pool(slots, ctx).unwrap();
                let other = ids(30, 5, net.vocab);
                net.feed_slot(&mut p, 2, &other, 0).unwrap();
                let other_kv = slot_bytes(&net, &p.c, 2 * ctx, 30);
                assert!(slot_bytes(&net, &p.c, 2 * ctx, 30) == other_kv);
                let mut row = Vec::new();
                net.save_kv(&plain, 0, 90, &mut row).unwrap();
                let mut moved: Option<CudaCache> = None;
                for slot in [0, 1] {
                    let what = format!("{} {kv:?} slot {slot}", m.arch);
                    // Every other slot's rows, whole, before the view runs.
                    let others: Vec<(usize, Vec<u8>)> = (0..slots)
                        .filter(|&o| o != slot)
                        .map(|o| (o, slot_bytes(&net, &p.c, o * ctx, ctx)))
                        .collect();
                    // Slot 0's view, then the same view pointed at slot 1
                    // (`retarget_view`: its graphs captured again).
                    let mut c = match moved.take() {
                        None => net.slot_view(&p, slot).unwrap(),
                        Some(mut c) => {
                            net.retarget_view(&p, &mut c, slot).unwrap();
                            c
                        }
                    };
                    assert_eq!(run(&net, &prompt, &mut c), want, "{what}: picks");
                    assert!(
                        slot_bytes(&net, &p.c, slot * ctx, 90) == want_kv,
                        "{what}: KV bytes"
                    );
                    for (o, b) in &others {
                        assert!(
                            slot_bytes(&net, &p.c, o * ctx, ctx) == *b,
                            "{what}: slot {o}'s bytes"
                        );
                    }
                    assert_eq!(net.adopt_cache(&mut p, slot, &c, 90), Ok(0), "{what}");
                    assert!(slot_bytes(&net, &p.c, slot * ctx, 90) == want_kv, "{what}");
                    assert!(net.adopt_cache(&mut p, 2, &c, 90).is_err(), "{what}");
                    // kvsave's rows of a view: the slot's, and no further.
                    let mut img = Vec::new();
                    net.save_kv(&c, 0, 90, &mut img).unwrap();
                    assert!(img == row, "{what}: saved image");
                    net.save_kv(&c, 0, ctx, &mut Vec::new()).unwrap();
                    assert!(
                        net.save_kv(&c, 0, ctx + 1, &mut Vec::new()).is_err(),
                        "{what}"
                    );
                    assert!(
                        net.save_kv(&c, ctx, ctx + 1, &mut Vec::new()).is_err(),
                        "{what}"
                    );
                    let big = vec![0u8; img.len() * 4];
                    assert!(
                        net.load_kv(&mut c, ctx - 10, ctx + 10, &mut &big[..])
                            .is_err(),
                        "{what}: load past the slot"
                    );
                    moved = Some(c);
                }
                assert!(net.slot_view(&p, slots).is_err());
                // A plain cache's adopt copies its rows (what the view's
                // count of 0 is told apart from).
                assert!(net.adopt_cache(&mut p, 2, &plain, 90).unwrap() > 0);
                assert!(slot_bytes(&net, &p.c, 2 * ctx, 90) == want_kv);
            }
        }
    }

    /// A wide pool on a 128-wide-head model (qwen3 0.6B; Llama 3.2 1B
    /// too, hd 64): 20 slots of 1152 positions, prompts up to 1100 tokens,
    /// 20 rows per step (more than att_per_query, so the per-query
    /// attention runs in several launches, and more than HEAD_SUB, so the
    /// head's slices split the greedy and the sampled rows, one selected on
    /// the host): every request's results and tokens == running it alone,
    /// and the slots' KV bytes == the solo caches'.
    #[test]
    fn pool_wide_long_steps_match_solo() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let km = Kernels::module(&g).unwrap();
        if !km.mma() {
            return;
        }
        let ms: Vec<Model> = [
            crate::testutil::qwen3_model(),
            crate::testutil::llama_model(),
        ]
        .into_iter()
        .flatten()
        .collect();
        let (slots, ctx, steps) = (20usize, 1152usize, 3usize);
        for m in ms {
            let net = CudaQwen2::with_kv(&g, &km, &m, ctx, CacheKv::F16, false).unwrap();
            let v = net.vocab;
            if net.head_dim == 128 {
                assert!(slots > super::super::batch::att_per_query(128));
            }
            assert!(slots > HEAD_SUB);
            let kinds: Vec<Kind> = (0..slots)
                .map(|i| match i % 4 {
                    0 | 2 => Kind::Greedy,
                    1 => Kind::Sampled(40, 1.1, i as u64),
                    _ if i == 3 => Kind::Sampled(0, 1.0, 99),
                    _ => Kind::Sampled(7, 1.3, i as u64),
                })
                .collect();
            let lens: Vec<usize> = (0..slots).map(|i| 1100 - (i * 53) % 1090).collect();
            let prompts: Vec<Vec<u32>> = (0..slots)
                .map(|i| ids(lens[i], 7000 + i as u32, v))
                .collect();
            let refs: Vec<_> = (0..slots)
                .map(|i| solo(&net, ctx, &prompts[i], kinds[i], steps))
                .collect();
            let mut p = net.pool(slots, ctx).unwrap();
            // Slot of request i: reversed.
            let slot_of = |i: usize| slots - 1 - i;
            for (i, pr) in prompts.iter().enumerate() {
                net.feed_slot(&mut p, slot_of(i), &pr[..pr.len() - 1], 0)
                    .unwrap();
            }
            let mut hist = prompts.clone();
            let mut sms: Vec<Option<Sampler>> = kinds
                .iter()
                .map(|k| match *k {
                    Kind::Greedy => None,
                    Kind::Sampled(k, pen, seed) => Some(sampler(k, pen, seed)),
                })
                .collect();
            for step in 0..steps {
                let specs: Vec<Option<CandSpec>> = sms
                    .iter()
                    .map(|s| s.as_ref().map(|s| s.cfg.spec()))
                    .collect();
                let rows: Vec<SeqRow> = (0..slots)
                    .map(|i| {
                        let h = &hist[i];
                        SeqRow {
                            token: h[h.len() - 1],
                            pos: h.len() - 1,
                            slot: slot_of(i),
                            head: match &specs[i] {
                                None => RowHead::Argmax,
                                Some(spec) => RowHead::Sample {
                                    spec: *spec,
                                    recent: &h[h.len() - spec.last_n.clamp(1, h.len())..],
                                },
                            },
                        }
                    })
                    .collect();
                let outs = net.step_rows(&mut p, &rows).unwrap();
                for (i, o) in outs.into_iter().enumerate() {
                    let what = format!("{} request {i} (len {}) step {step}", m.arch, lens[i]);
                    assert_eq!(o, refs[i].0[step], "{what}");
                    let next = match (&o, sms[i].as_mut()) {
                        (RowOut::Pick(t), None) => *t,
                        (RowOut::Cands(cs), Some(s)) => s.pick(cs),
                        _ => panic!("{what}: a result of another head"),
                    };
                    hist[i].push(next);
                }
            }
            for i in 0..slots {
                assert_eq!(hist[i], refs[i].1, "{} request {i}: tokens", m.arch);
                let n = hist[i].len() - 1;
                assert!(
                    slot_bytes(&net, &p.c, slot_of(i) * ctx, n) == refs[i].2,
                    "{} request {i}: KV bytes",
                    m.arch
                );
            }
        }
    }

    /// Pool steps of 17..64 rows (past gemvm16: the products split between
    /// gemvm32 and gemm2's partial tiles, `takes_gemvm`; the head in one
    /// product over all rows; samp_tail in launches of HEAD_SUB rows), on
    /// the 0.5B and Llama 1B, mixed greedy and sampled rows: every
    /// request's results and tokens == running it alone, under each
    /// gemvm routing (AXON_GEMVM_MAX 16, 32, by shape).
    #[test]
    fn pool_steps_past_gemvm16_match_solo() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let km = Kernels::module(&g).unwrap();
        if !km.mma() {
            return;
        }
        let ms: Vec<Model> = [crate::testutil::model(), crate::testutil::llama_model()]
            .into_iter()
            .flatten()
            .collect();
        let (slots, ctx, steps) = (64usize, 256usize, 2usize);
        for m in ms {
            let mut net = CudaQwen2::with_kv(&g, &km, &m, ctx, CacheKv::F16, false).unwrap();
            let v = net.vocab;
            let kinds: Vec<Kind> = (0..slots)
                .map(|i| match i % 3 {
                    0 => Kind::Greedy,
                    1 => Kind::Sampled(40, 1.1, i as u64),
                    _ => Kind::Sampled(5, 1.0, i as u64),
                })
                .collect();
            let lens: Vec<usize> = (0..slots).map(|i| 20 + (i * 37) % 200).collect();
            let prompts: Vec<Vec<u32>> = (0..slots)
                .map(|i| ids(lens[i], 9000 + i as u32, v))
                .collect();
            let refs: Vec<_> = (0..slots)
                .map(|i| solo(&net, ctx, &prompts[i], kinds[i], steps))
                .collect();
            for vm in [16usize, 32, 0] {
                net.bf.vm_max = vm;
                for nb in [17usize, 33, 48, 64] {
                    let mut p = net.pool(slots, ctx).unwrap();
                    for (i, pr) in prompts.iter().enumerate().take(nb) {
                        net.feed_slot(&mut p, i, &pr[..pr.len() - 1], 0).unwrap();
                    }
                    let mut hist: Vec<Vec<u32>> = prompts[..nb].to_vec();
                    let mut sms: Vec<Option<Sampler>> = kinds[..nb]
                        .iter()
                        .map(|k| match *k {
                            Kind::Greedy => None,
                            Kind::Sampled(k, pen, seed) => Some(sampler(k, pen, seed)),
                        })
                        .collect();
                    for step in 0..steps {
                        let specs: Vec<Option<CandSpec>> = sms
                            .iter()
                            .map(|s| s.as_ref().map(|s| s.cfg.spec()))
                            .collect();
                        let rows: Vec<SeqRow> = (0..nb)
                            .map(|i| {
                                let h = &hist[i];
                                SeqRow {
                                    token: h[h.len() - 1],
                                    pos: h.len() - 1,
                                    slot: i,
                                    head: match &specs[i] {
                                        None => RowHead::Argmax,
                                        Some(spec) => RowHead::Sample {
                                            spec: *spec,
                                            recent: &h[h.len() - spec.last_n.clamp(1, h.len())..],
                                        },
                                    },
                                }
                            })
                            .collect();
                        let outs = net.step_rows(&mut p, &rows).unwrap();
                        for (i, o) in outs.into_iter().enumerate() {
                            let what =
                                format!("{} vm {vm} nb {nb} request {i} step {step}", m.arch);
                            assert_eq!(o, refs[i].0[step], "{what}");
                            let next = match (&o, sms[i].as_mut()) {
                                (RowOut::Pick(t), None) => *t,
                                (RowOut::Cands(cs), Some(s)) => s.pick(cs),
                                _ => panic!("{what}: a result of another head"),
                            };
                            hist[i].push(next);
                        }
                    }
                    for i in 0..nb {
                        assert_eq!(hist[i], refs[i].1, "{} vm {vm} nb {nb} request {i}", m.arch);
                    }
                }
            }
        }
    }

    /// Step cost of a pool vs the decode step, the C curve (MEASURE, not a
    /// check; `--ignored --nocapture` to see). `AXON_STEP_MODEL=path`
    /// (default the 0.5B); `AXON_STEP_DEPTHS` (default "600,4096"),
    /// `AXON_STEP_NB` (default "1,2,4,8,16,24,32,48,64"), `AXON_STEP_PROF`
    /// (the nb's profiled per kernel, default "1,8,16,32,64"; "" none),
    /// `AXON_STEP_VM` (gemvm's batch limits `BatchFns::vm_max` to compare,
    /// default the build's).
    /// Per depth: one pool of max(nb) slots (fewer if it doesn't fit), every
    /// slot fed `depth` tokens; then nb rows of distinct slots at position
    /// `depth`, greedy, graphs on (as serve), ms per step = median of 5 runs
    /// of 40 steps, against the decode step (forward_argmax) at that depth.
    /// The profile: one pass without graphs, an event after each `on`
    /// segment, summed per name over the layers (3rd of 3 rounds).
    #[test]
    #[ignore]
    fn pool_step_cost() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        // AXON_STEP_MODEL: one model file or several, comma-separated (one
        // job for a whole round).
        let paths: Vec<Option<String>> = match std::env::var("AXON_STEP_MODEL") {
            Ok(v) => v.split(',').map(|p| Some(p.trim().to_string())).collect(),
            Err(_) => vec![None],
        };
        for path in paths {
            let m = match &path {
                Some(p) => crate::gguf::load_with(p, Some(false)).unwrap(),
                None => match crate::testutil::model() {
                    Some(m) => m,
                    None => return,
                },
            };
            let list = |var: &str, def: &str| -> Vec<usize> {
                std::env::var(var)
                    .unwrap_or(def.into())
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim().parse().unwrap())
                    .collect()
            };
            let depths = list("AXON_STEP_DEPTHS", "600,4096");
            let nbs = list("AXON_STEP_NB", "1,2,4,8,16,24,32,48,64");
            let profs = list("AXON_STEP_PROF", "1,8,16,32,64");
            // gemvm's batch limits to compare (the last one profiled), default
            // the build's.
            let vms = match list("AXON_STEP_VM", "") {
                v if v.is_empty() => vec![0],
                v => v,
            };
            // AXON_STEP_KM: the kernel builds to run in turn per depth, e.g.
            // "plain,AXON_G2_ZFILL,plain,AXON_G2_ZFILL" (prefill.cu g2_copy).
            let kms: Vec<String> = std::env::var("AXON_STEP_KM")
                .unwrap_or("plain".into())
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            // Each entry "plain", or defines joined by '+' (e.g. "GVM_PF2+GVM_MINB=4").
            let mods: Vec<(String, crate::cuda::Module)> = kms
                .iter()
                .map(|k| {
                    let mut opts = vec!["--std=c++17".to_string(), "--ftz=false".to_string()];
                    if k != "plain" {
                        opts.extend(k.split('+').map(|d| format!("-D{d}")));
                    }
                    let o: Vec<&str> = opts.iter().map(|s| s.as_str()).collect();
                    let m = crate::cuda::Module::load(&g, crate::cuda::KERNELS, &o)
                        .unwrap()
                        .0;
                    (k.clone(), m)
                })
                .collect();
            let time = |f: &mut dyn FnMut()| {
                f();
                let mut v: Vec<f64> = (0..5)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        for _ in 0..40 {
                            f();
                        }
                        t.elapsed().as_secs_f64() * 1e3 / 40.0
                    })
                    .collect();
                v.sort_by(f64::total_cmp);
                v[2]
            };
            let name = path.as_deref().unwrap_or("default 0.5B");
            let gi = g.info().unwrap();
            eprintln!("C curve: {name} ({}), {} ({} SMs)", m.arch, gi.name, gi.sms);
            for (&depth, kname) in depths.iter().flat_map(|d| kms.iter().map(move |k| (d, k))) {
                let km = &mods.iter().find(|m| &m.0 == kname).unwrap().1;
                eprintln!("kernels: {kname}");
                let ctx = (depth + 64).next_multiple_of(64);
                let mut net = CudaQwen2::new(&g, km, &m, ctx).unwrap();
                net.graphs = true;
                let prompt: Vec<u32> = (0..depth)
                    .map(|i| 1000 + (i * 7919 % 20000) as u32)
                    .collect();
                let d = {
                    let mut c = net.new_cache(ctx);
                    net.feed_batch(&prompt, 0, &mut c);
                    time(&mut || {
                        std::hint::black_box(net.forward_argmax(5, depth, &mut c));
                    })
                };
                let want = *nbs.iter().chain(&profs).max().unwrap();
                let mut slots = want;
                let mut p = loop {
                    match net.pool(slots, ctx) {
                        Ok(p) => break p,
                        Err(e) if slots > 1 => {
                            eprintln!("  pool of {slots} x {ctx}: {e}");
                            slots /= 2;
                        }
                        Err(e) => panic!("{e}"),
                    }
                };
                for s in 0..slots {
                    net.feed_slot(&mut p, s, &prompt, 0).unwrap();
                }
                eprintln!(
                    "depth {depth}: decode step (forward_argmax) {d:.3} ms; pool {slots} x {ctx}"
                );
                eprintln!("| depth | nb | gemvm max | ms/step | x solo | rows/s |");
                let rows = |nb: usize| -> Vec<SeqRow> {
                    (0..nb)
                        .map(|s| SeqRow {
                            token: 5,
                            pos: depth,
                            slot: s,
                            head: RowHead::Argmax,
                        })
                        .collect()
                };
                for &nb in nbs.iter().filter(|&&nb| nb <= slots) {
                    let rs = rows(nb);
                    for &vm in &vms {
                        net.bf.vm_max = vm;
                        // (The graphs hold the kernels picked when made.)
                        p.graphs.clear();
                        let t = time(&mut || {
                            std::hint::black_box(net.step_rows(&mut p, &rs).unwrap());
                        });
                        eprintln!(
                            "| {depth} | {nb} | {vm} | {t:.3} | {:.2} | {:.0} |",
                            t / d,
                            nb as f64 * 1e3 / t
                        );
                    }
                }
                net.bf.vm_max = *vms.last().unwrap();
                for &nb in profs.iter().filter(|&&nb| nb <= slots) {
                    let tokens = vec![5u32; nb];
                    let at: Vec<RowAt> = (0..nb)
                        .map(|s| RowAt {
                            pos: depth as u32,
                            base: (s * p.ctx) as u32,
                        })
                        .collect();
                    for round in 0..3 {
                        let tab = p.c.tab.as_ref().map(|t| t.len() / 4);
                        let b = p.c.b.as_mut().unwrap();
                        net.load_input(b, BatchIn::Tokens(&tokens)).unwrap();
                        b.upload_rows(&at, tab).unwrap();
                        let mut evs = vec![("start", g.record().unwrap())];
                        net.pass_at(
                            &mut p.c,
                            BatchIn::Loaded(nb),
                            Place::Rows,
                            Head::Sample,
                            Some((nb, nb)),
                            0,
                            &mut |n| {
                                evs.push((n, g.record()?));
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
                        let segs: Vec<String> = agg
                            .iter()
                            .filter(|a| a.1 >= 0.005)
                            .map(|a| format!("{} {:.3}", a.0, a.1))
                            .collect();
                        eprintln!(
                            "prof depth {depth} nb {nb}: {total:.3} ms (no graph, {} events): {}",
                            evs.len() - 1,
                            segs.join(", ")
                        );
                    }
                }
            }
        }
    }
}
