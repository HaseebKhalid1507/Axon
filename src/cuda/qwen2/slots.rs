//! `axon serve`'s continuous batching on the GPU (stage S2-lite of
//! specs/r4-serve.md, brief specs/r6-pick.md §4): a pool of slots and the
//! multi-sequence step.
//!
//! The pool is one cache whose rows are pages of 64 positions (specs
//! r11 §2, pages.rs): slot s's position t is at row
//! `tab[s * cap + t / 64] * 64 + t % 64`, `tab` the pool's page table on
//! the device (fixed address: graphs hold it; contents change) and
//! `PageMap` its host side. A slot holds pages only for the positions it
//! has (`ensure` before a feed or step writes past them, `release` when
//! its rows are dropped), so `ctx` is a slot's cap, not a reservation.
//! With AXON_PAGED_KV=0 the pool is plain: slot s holds rows s * ctx..
//! (the layout before paging). A prompt is fed into its slot by the
//! batched passes as `feed_batch` (`Place::Seq` with the slot's first
//! table entry, or row); a step runs one token of each
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
use super::pages::{self, NoPages, PAGE, PageMap};
use super::*;
use crate::lm::{BatchLm, PoolPages, RowHead, RowOut, SeqRow, StepError};
use std::collections::HashMap;
use std::rc::Rc;

/// sample.cu SP_PER_ROW: S[1] when each row has its own penalty.
const SP_PER_ROW: u32 = u32::MAX;

/// A pool of `slots` sequences of up to `ctx` positions each.
pub struct CudaPool<'g> {
    c: CudaCache<'g>,
    slots: usize,
    ctx: usize,
    /// Paged: which pages each slot holds (`c.tab` its device mirror; the
    /// last page, `sink`, is in no slot: table entries of positions a slot
    /// doesn't hold point there). None: plain rows (AXON_PAGED_KV=0).
    pm: Option<PageMap>,
    /// Most rows per step (logits scratch rows).
    rows_max: usize,
    /// Captured steps by (rows, greedy rows, + device-sampled rows, the
    /// attention's queries per launch `BatchBufs::att_z`): a step's kernels
    /// and their arguments depend on nothing else (its tokens, places and S
    /// are copied in before the graph runs).
    /// Each with the step count of its last use: the least recently used
    /// one goes when one more is needed past MAX_GRAPHS.
    graphs: HashMap<(usize, usize, usize, usize), (Graph<'g>, u64)>,
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
    #[cfg_attr(not(test), allow(dead_code))] // serve takes `slot_view_bare`
    pub fn slot_view(&self, p: &CudaPool<'g>, slot: usize) -> Result<CudaCache<'g>, String> {
        self.slot_view_with(p, slot, true)
    }

    /// `slot_view` of a paged pool without batch scratch of its own (serve's
    /// single-request cache: `BatchLm::slot_cache`): its batched passes
    /// (prefill, verify) run on the pool's, lent while it runs
    /// (`lend_scratch` .. `return_scratch`), as the pool and the view never
    /// run at once. A plain pool's view has its own (as before paging).
    pub fn slot_view_bare(&self, p: &CudaPool<'g>, slot: usize) -> Result<CudaCache<'g>, String> {
        self.slot_view_with(p, slot, p.pm.is_none())
    }

    /// The pool's batch scratch, lent to `c` (its bare view: `slot_view_bare`;
    /// a view with scratch of its own: nothing to do). The pool's own passes
    /// can't run until `return_scratch`. The buffers move, their addresses
    /// don't: the pool's captured steps hold them still.
    pub fn lend_scratch(&self, p: &mut CudaPool<'g>, c: &mut CudaCache<'g>) -> Result<(), String> {
        if c.b.is_some() {
            return Ok(());
        }
        if !Rc::ptr_eq(&c.kv, &p.c.kv) {
            return Err("lend_scratch: not a view of this pool".into());
        }
        let mut b =
            p.c.b
                .take()
                .ok_or("lend_scratch: the pool's scratch is lent")?;
        // A one-sequence cache's head writes one row per launch.
        b.logits_rows = false;
        c.b = Some(b);
        Ok(())
    }

    /// `lend_scratch` undone (a no-op for a view with scratch of its own).
    pub fn return_scratch(
        &self,
        p: &mut CudaPool<'g>,
        c: &mut CudaCache<'g>,
    ) -> Result<(), String> {
        if p.c.b.is_some() {
            return Ok(());
        }
        if !Rc::ptr_eq(&c.kv, &p.c.kv) {
            return Err("return_scratch: not a view of this pool".into());
        }
        let mut b =
            c.b.take()
                .ok_or("return_scratch: the view holds no scratch")?;
        b.logits_rows = true;
        p.c.b = Some(b);
        Ok(())
    }

    fn slot_view_with(
        &self,
        p: &CudaPool<'g>,
        slot: usize,
        batch: bool,
    ) -> Result<CudaCache<'g>, String> {
        if slot >= p.slots {
            return Err(format!("slot {slot} of a {}-slot pool", p.slots));
        }
        if p.pm.is_none() {
            let shared = (p.c.kv.clone(), p.c.rows, slot * p.ctx);
            return self.make_cache(p.ctx, p.c.mode, p.ctx, Some(shared), true);
        }
        // Paged: the pool's arrays from row 0 and its table from the slot's
        // first entry. Its positions go to the cap whatever the pool's rows
        // (the caller `reserve`s pages for those the `Lm` calls write: past
        // them, the table points at the sink page).
        let shared = (p.c.kv.clone(), p.c.rows, 0);
        let mut c = self.make_cache(p.ctx.min(p.c.rows), p.c.mode, p.ctx, Some(shared), batch)?;
        c.max_ctx = p.ctx;
        c.tab = p.c.tab.clone();
        c.tab0 = p.base(slot);
        Ok(c)
    }

    /// VRAM a pool of `slots` x `ctx` takes with caches of type `kv`
    /// (`pool` checks it is free).
    fn pool_bytes_kv(&self, kv: CacheKv, slots: usize, ctx: usize) -> usize {
        if pages::paged_env() {
            return self.pool_paged_bytes(kv, slots, ctx, slots * ctx.div_ceil(PAGE));
        }
        self.cache_bytes_kv(kv, ctx) - self.kv_cache_bytes(kv, ctx)
            + self.kv_cache_bytes(kv, slots * ctx)
            + slots.saturating_sub(HEAD_SUB) * self.vocab * 4
            + self.batch_for(ctx) * 8
    }

    /// `serve_bytes_kv` of a paged pool (`pool_paged_bytes`).
    /// (The view is bare: `slot_view_bare`, the pool's batch scratch lent.)
    pub fn serve_paged_bytes(&self, kv: CacheKv, slots: usize, ctx: usize, pages: usize) -> usize {
        self.cache_bytes_kv(kv, ctx)
            - self.kv_cache_bytes(kv, ctx)
            - self.batch_bytes(self.batch_for(ctx), ctx)
            + self.pool_paged_bytes(kv, slots, ctx, pages)
    }

    /// Bytes of one page (PAGE positions of every layer) of a `kv` cache.
    pub fn page_bytes(&self, kv: CacheKv) -> usize {
        self.kv_cache_bytes(kv, PAGE)
    }

    /// VRAM a paged pool of `pages` pages for `slots` slots of up to `ctx`
    /// positions takes (`pool_paged`; its scratch is sized by `ctx`).
    pub fn pool_paged_bytes(&self, kv: CacheKv, slots: usize, ctx: usize, pages: usize) -> usize {
        self.cache_bytes_kv(kv, ctx) - self.kv_cache_bytes(kv, ctx)
            + self.kv_cache_bytes(kv, (pages + 1) * PAGE)
            + slots * ctx.div_ceil(PAGE) * 4
            + slots.saturating_sub(HEAD_SUB) * self.vocab * 4
            + self.batch_for(ctx) * 8
    }

    /// A pool of `slots` sequences of up to `ctx` positions, in the model's
    /// KV type, or why there's no room for it: paged with a page for every
    /// position of every slot (AXON_PAGE_SHUFFLE: in a shuffled order), or
    /// plain with AXON_PAGED_KV=0.
    pub fn pool(&self, slots: usize, ctx: usize) -> Result<CudaPool<'g>, String> {
        let pages = pages::paged_env().then(|| slots.saturating_mul(ctx.div_ceil(PAGE)));
        self.pool_with(slots, ctx, pages, pages::shuffle_env())
    }

    /// A paged pool of `pages` pages for `slots` slots of up to `ctx`
    /// positions each (`pages` None: plain, every slot its ctx rows);
    /// `shuffle`: pages handed out in a seeded random order.
    pub fn pool_with(
        &self,
        slots: usize,
        ctx: usize,
        pages: Option<usize>,
        shuffle: Option<u64>,
    ) -> Result<CudaPool<'g>, String> {
        if slots == 0 || ctx == 0 || pages == Some(0) {
            return Err("a pool needs at least one slot, position and page".into());
        }
        let cap = ctx.div_ceil(PAGE);
        let rows = match pages {
            None => slots.checked_mul(ctx),
            // The sink page after the pool's.
            Some(n) => n.checked_add(1).and_then(|n| n.checked_mul(PAGE)),
        };
        let max_ctx = rows
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| format!("{slots} slots of {ctx} positions: too many"))?;
        if self.first > 0 {
            return Err("no pool on a split model's GPU half".into());
        }
        let mut c = self.try_cache_at(max_ctx, self.kv, ctx)?;
        let g = self.gpu;
        let pm = match pages {
            None => None,
            Some(n) => {
                // Every entry at the sink until a slot takes pages.
                let tab = g.upload(&vec![n as u32; slots * cap])?;
                c.tab = Some(Rc::new(tab));
                Some(PageMap::new(slots, cap, n, shuffle))
            }
        };
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
            pm,
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
        if p.c.b.is_none() {
            return Err("feed_slot: the pool's scratch is lent (`return_scratch` first)".into());
        }
        let short = self.reserve(p, slot, start + tokens.len())?;
        if short > 0 {
            return Err(format!(
                "feed_slot: slot {slot} positions {start}..{}: {short} KV pages short",
                start + tokens.len()
            ));
        }
        #[cfg(debug_assertions)]
        p.debug_tab();
        let max = p.pass_max();
        let base = p.base(slot);
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
        if p.c.b.is_none() {
            return bad("the pool's scratch is lent (`return_scratch` first)".into());
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
        // Each row's page (a new one every PAGE positions).
        for r in rows {
            match self.reserve(p, r.slot, r.pos + 1) {
                Ok(0) => {}
                Ok(short) => {
                    return bad(format!(
                        "row at {} of slot {}: {short} KV pages short",
                        r.pos, r.slot
                    ));
                }
                Err(e) => return Err(StepError::Device(e)),
            }
        }
        #[cfg(debug_assertions)]
        p.debug_tab();
        self.step_checked(p, rows).map_err(StepError::Device)
    }

    /// Pages for slot `slot`'s positions 0..hi: Ok(0) once it holds them
    /// (its new table entries copied to the device, ordered before any
    /// later work), or the pages short (nothing taken). A plain pool holds
    /// every slot's ctx rows: Ok(0).
    pub fn reserve(&self, p: &mut CudaPool<'g>, slot: usize, hi: usize) -> Result<usize, String> {
        if slot >= p.slots || hi > p.ctx {
            return Err(format!(
                "reserve: slot {slot} positions ..{hi} (pool {} x {})",
                p.slots, p.ctx
            ));
        }
        let Some(m) = p.pm.as_mut() else {
            return Ok(0);
        };
        if m.retired() {
            // Released pages are free once nothing in flight reads them.
            self.gpu.sync()?;
            m.flush();
        }
        let r = match m.ensure(slot, hi) {
            Ok(r) => r,
            Err(NoPages { short }) => return Ok(short),
        };
        if !r.is_empty() {
            let tab = p.c.tab.as_ref().expect("a paged pool's table");
            let e = slot * m.cap() + r.start;
            tab.write_at(e * 4, &m.pages(slot)[r])?;
        }
        Ok(0)
    }

    /// Slot `slot` keeps the pages of positions 0..lo (a paged pool; plain:
    /// nothing changes): the others are free once no work in flight can
    /// read them, and their table entries point at the sink.
    pub fn release(&self, p: &mut CudaPool<'g>, slot: usize, lo: usize) -> Result<(), String> {
        if slot >= p.slots {
            return Err(format!("release: slot {slot} of {}", p.slots));
        }
        let Some(m) = p.pm.as_mut() else {
            return Ok(());
        };
        let (keep, had) = (PageMap::need(lo), m.pages(slot).len());
        if keep >= had {
            return Ok(());
        }
        m.release(slot, lo);
        let tab = p.c.tab.as_ref().expect("a paged pool's table");
        let sink = vec![m.total() as u32; had - keep];
        // Synchronous: the step in flight is done too, so the pages are free.
        tab.write_at((slot * m.cap() + keep) * 4, &sink)?;
        m.flush();
        Ok(())
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
                base: p.base(rows[i].slot) as u32,
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
        let pos_max = rows.iter().map(|r| r.pos).max().unwrap_or(0);
        b.att_z = self.att_rows(b, nb, pos_max);
        let att_z = b.att_z;
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
            let key = (nb, ng, ns, att_z);
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
        let paged = |t: &Option<Rc<DevBuf<'g>>>| t.as_ref().map(Rc::as_ptr);
        if paged(&c.tab) != paged(&p.c.tab) {
            return Err("retarget: a view of a paged pool takes its table only".into());
        }
        // Paged: the slot's first table entry; plain: its first row.
        let (at, now) = match p.pm {
            Some(_) => (&mut c.tab0, p.base(slot)),
            None => (&mut c.row0, slot * p.ctx),
        };
        if *at != now {
            *at = now;
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
        if Rc::ptr_eq(&src.kv, &p.c.kv) && p.pm.is_some() {
            // A view of the paged pool: in place if it is this slot's (its
            // positions in the slot's pages, which `reserve` took).
            if src.tab0 != p.base(slot) {
                return Err(format!("adopt: another slot's view into slot {slot}"));
            }
            let held = p.pm.as_ref().map_or(0, |m| m.pages(slot).len());
            if held < PageMap::need(n) {
                return Err(format!(
                    "adopt: slot {slot} holds {held} pages, not {n} positions"
                ));
            }
            return Ok(0);
        }
        if src.tab.is_some() {
            return Err("adopt: a paged cache of another pool".into());
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
        // Runs of positions (first, first row in the pool, count): one,
        // or a page each through the table.
        let runs: Vec<(usize, usize, usize)> = match p.pm.is_some() {
            false => vec![(0, slot * p.ctx, n)],
            true => {
                let short = self.reserve(p, slot, n)?;
                if short > 0 {
                    return Err(format!("adopt: {short} KV pages short"));
                }
                let m = p.pm.as_ref().expect("paged");
                (0..n)
                    .step_by(PAGE)
                    .map(|t| (t, m.pages(slot)[t / PAGE] as usize * PAGE, PAGE.min(n - t)))
                    .collect()
            }
        };
        let (ns, np) = (src.rows, p.c.rows);
        let mut src_at = 0;
        let mut dst_at = 0;
        let mut copied = 0;
        for (row, _) in regions {
            for a in 0..2 * nl {
                for &(t, r, k) in &runs {
                    src.kv.copy_into(
                        src_at + (a * ns + t) * row,
                        &p.c.kv,
                        dst_at + (a * np + r) * row,
                        k * row,
                    )?;
                    copied += k * row;
                }
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

    fn new_pool_paged(
        &self,
        slots: usize,
        ctx: usize,
        pages: usize,
    ) -> Result<CudaPool<'g>, String> {
        self.pool_with(slots, ctx, Some(pages), pages::shuffle_env())
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
        self.slot_view_bare(pool, slot)
    }

    fn lend_scratch(&self, pool: &mut CudaPool<'g>, c: &mut CudaCache<'g>) -> Result<(), String> {
        CudaQwen2::lend_scratch(self, pool, c)
    }

    fn return_scratch(&self, pool: &mut CudaPool<'g>, c: &mut CudaCache<'g>) -> Result<(), String> {
        CudaQwen2::return_scratch(self, pool, c)
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

    fn reserve(&self, pool: &mut CudaPool<'g>, slot: usize, hi: usize) -> Result<usize, String> {
        CudaQwen2::reserve(self, pool, slot, hi)
    }

    fn release(&self, pool: &mut CudaPool<'g>, slot: usize, lo: usize) -> Result<(), String> {
        CudaQwen2::release(self, pool, slot, lo)
    }

    fn pages(&self, pool: &CudaPool<'g>) -> Option<PoolPages> {
        pool.page_info()
    }
}

impl CudaPool<'_> {
    /// Slot `slot`'s first table entry (paged), or first row (plain): a
    /// `Place::Seq` / `RowAt` base.
    fn base(&self, slot: usize) -> usize {
        match &self.pm {
            Some(m) => slot * m.cap(),
            None => slot * self.ctx,
        }
    }

    /// Debug builds: the allocator's invariants hold and the device table
    /// is its mirror (a slot's entries its pages, then the sink).
    #[cfg(debug_assertions)]
    fn debug_tab(&self) {
        let (Some(m), Some(t)) = (&self.pm, &self.c.tab) else {
            return;
        };
        m.check();
        let tab: Vec<u32> = t.read().expect("the page table");
        for (s, row) in tab.chunks(m.cap()).enumerate() {
            let l = m.pages(s);
            assert_eq!(&row[..l.len()], l, "slot {s}'s table row");
            assert!(
                row[l.len()..].iter().all(|&e| e as usize == m.total()),
                "slot {s}'s table past its pages"
            );
        }
    }

    /// `BatchLm::pages`.
    pub fn page_info(&self) -> Option<PoolPages> {
        let m = self.pm.as_ref()?;
        Some(PoolPages {
            size: PAGE,
            free: m.free(),
            total: m.total(),
            held: (0..self.slots).map(|s| m.pages(s).len()).collect(),
        })
    }

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

    /// The pool layouts the slot tests run on: plain rows
    /// (AXON_PAGED_KV=0), pages handed out in order, and shuffled pages
    /// (AXON_PAGE_SHUFFLE) with a few spare.
    #[derive(Clone, Copy, Debug)]
    enum Lay {
        Plain,
        Paged,
        Shuffled(u64),
    }

    const LAYS: [Lay; 3] = [Lay::Plain, Lay::Paged, Lay::Shuffled(0x5eed)];

    fn pool_of<'g>(net: &CudaQwen2<'g>, slots: usize, ctx: usize, lay: Lay) -> CudaPool<'g> {
        let pages = slots * ctx.div_ceil(PAGE);
        match lay {
            Lay::Plain => net.pool_with(slots, ctx, None, None),
            Lay::Paged => net.pool_with(slots, ctx, Some(pages), None),
            Lay::Shuffled(s) => net.pool_with(slots, ctx, Some(pages + 3), Some(s)),
        }
        .unwrap()
    }

    /// Slot `slot`'s first `n` positions of every layer (a paged slot's:
    /// those it holds pages for, at most n), gathered through its pages:
    /// `slot_bytes`' layout.
    fn pool_bytes(net: &CudaQwen2, p: &CudaPool, slot: usize, n: usize) -> Vec<u8> {
        let Some(m) = &p.pm else {
            return slot_bytes(net, &p.c, slot * p.ctx, n);
        };
        let pages = m.pages(slot);
        let n = n.min(pages.len() * PAGE);
        let all: Vec<u8> = p.c.kv.read().unwrap();
        let (c, kvd) = (&p.c, net.kv_dim());
        let rows: Vec<usize> = match c.mode {
            CacheKv::F16 => vec![kvd * 2],
            CacheKv::Q8 { group } => vec![kvd, kvd / group * 4],
        };
        let mut out = Vec::new();
        let mut at = 0;
        for row in rows {
            for a in 0..2 * net.n_layers {
                for t in (0..n).step_by(PAGE) {
                    let r = pages[t / PAGE] as usize * PAGE;
                    let s = at + (a * c.rows + r) * row;
                    out.extend_from_slice(&all[s..s + PAGE.min(n - t) * row]);
                }
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
                let cases = LAYS
                    .iter()
                    .flat_map(|&l| [(1usize, 0usize), (5, 2), (16, 1)].map(|x| (l, x)));
                for (lay, (nb, slot)) in cases {
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
                    let mut p = pool_of(&net, 3, ctx, lay);
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
                    let what = format!("{} {kv:?} {lay:?} nb {nb} slot {slot}", m.arch);
                    for (b, o) in got.iter().enumerate() {
                        assert_eq!(*o, RowOut::Cands(want_c[b].clone()), "{what} row {b}");
                        if let RowOut::Cands(c) = o {
                            let bits = |c: &[Cand]| -> Vec<(u32, u32)> {
                                c.iter().map(|c| (c.logit.to_bits(), c.id)).collect()
                            };
                            assert_eq!(bits(c), bits(&want_c[b]), "{what} row {b} bits");
                        }
                    }
                    assert!(
                        pool_bytes(&net, &p, slot, start + nb) == want_kv,
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
                        pool_bytes(&net, &p, slot, start + nb) == want_kv,
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
                let cases = LAYS.iter().flat_map(|&l| [(l, false), (l, true)]);
                for (lay, via_view) in cases {
                    let slots = nr + 1;
                    let mut p = pool_of(&net, slots, ctx, lay);
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
                        true => {
                            // The `Lm` calls take no pages: reserved first.
                            let n = prompts[handed].len() + handed_at;
                            assert_eq!(net.reserve(&mut p, slot_of[handed], n), Ok(0));
                            net.slot_view(&p, slot_of[handed]).unwrap()
                        }
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
                                "{} {kv:?} {lay:?} view {via_view} request {i} step {}",
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
                        let what = format!("{} {kv:?} {lay:?} view {via_view} request {i}", m.arch);
                        assert_eq!(done[i], steps, "{what}");
                        assert_eq!(hist[i], refs[i].1, "{what}: tokens");
                        let n = hist[i].len() - 1;
                        assert!(
                            pool_bytes(&net, &p, slot_of[i], n) == refs[i].2,
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

    /// Views and tables: `set_tab` (tests' tables) refuses a view and a
    /// pool with a live view; a plain pool's view never gets a table
    /// (`retarget_view` refuses one); a paged pool's view shares the pool's
    /// table from its slot's first entry, and `retarget_view` moves it.
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
        let p = pool_of(&net, 2, 128, Lay::Plain);
        for slot in [0, 1] {
            let mut v = net.slot_view(&p, slot).unwrap();
            let t = g.upload(&[0u32, 1]).unwrap();
            let set = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.set_tab(Some(t))));
            assert!(set.is_err(), "set_tab on slot {slot}'s view");
            v.tab = Some(Rc::new(g.upload(&[0u32, 1]).unwrap()));
            assert!(
                net.retarget_view(&p, &mut v, 1 - slot).is_err(),
                "retarget of a paged view of a plain pool"
            );
        }
        let mut p = pool_of(&net, 2, 128, Lay::Plain);
        {
            let _v = net.slot_view(&p, 0).unwrap();
            let t = g.upload(&[0u32, 1, 2, 3]).unwrap();
            let set =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.c.set_tab(Some(t))));
            assert!(set.is_err(), "set_tab on a pool with a live view");
        }
        let p = pool_of(&net, 2, 128, Lay::Paged);
        let mut v = net.slot_view(&p, 1).unwrap();
        assert!(Rc::ptr_eq(
            v.tab.as_ref().unwrap(),
            p.c.tab.as_ref().unwrap()
        ));
        assert_eq!((v.tab0, v.row0, v.max_ctx), (2, 0, 128));
        net.retarget_view(&p, &mut v, 0).unwrap();
        assert_eq!(v.tab0, 0);
        let t = g.upload(&[0u32, 1]).unwrap();
        let set = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| v.set_tab(Some(t))));
        assert!(set.is_err(), "set_tab on a paged pool's view");
        let mut plain = net
            .slot_view(&pool_of(&net, 2, 128, Lay::Plain), 0)
            .unwrap();
        assert!(net.retarget_view(&p, &mut plain, 1).is_err());
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
                for lay in LAYS {
                    let mut p = pool_of(&net, slots, ctx, lay);
                    let other = ids(30, 5, net.vocab);
                    net.feed_slot(&mut p, 2, &other, 0).unwrap();
                    let other_kv = pool_bytes(&net, &p, 2, 30);
                    assert!(pool_bytes(&net, &p, 2, 30) == other_kv);
                    let mut row = Vec::new();
                    net.save_kv(&plain, 0, 90, &mut row).unwrap();
                    let mut moved: Option<CudaCache> = None;
                    for slot in [0, 1] {
                        let what = format!("{} {kv:?} {lay:?} slot {slot}", m.arch);
                        // The `Lm` calls take no pages: the slot's, reserved.
                        assert_eq!(net.reserve(&mut p, slot, ctx), Ok(0), "{what}");
                        // Every other slot's rows, whole, before the view runs.
                        let others: Vec<(usize, Vec<u8>)> = (0..slots)
                            .filter(|&o| o != slot)
                            .map(|o| (o, pool_bytes(&net, &p, o, ctx)))
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
                            pool_bytes(&net, &p, slot, 90) == want_kv,
                            "{what}: KV bytes"
                        );
                        for (o, b) in &others {
                            assert!(
                                pool_bytes(&net, &p, *o, ctx) == *b,
                                "{what}: slot {o}'s bytes"
                            );
                        }
                        assert_eq!(net.adopt_cache(&mut p, slot, &c, 90), Ok(0), "{what}");
                        assert!(pool_bytes(&net, &p, slot, 90) == want_kv, "{what}");
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
                    assert!(pool_bytes(&net, &p, 2, 90) == want_kv);
                }
            }
        }
    }

    /// `t` fed into slot `slot` from position `at` in uneven chunks
    /// (`chunk` first).
    fn feed_chunks<'g>(
        net: &CudaQwen2<'g>,
        p: &mut CudaPool<'g>,
        slot: usize,
        t: &[u32],
        at: usize,
        chunk: usize,
    ) {
        let mut i = 0;
        for n in [chunk, 1, 64, 63, 200].iter().cycle() {
            let e = (i + n).min(t.len());
            net.feed_slot(p, slot, &t[i..e], at + i).unwrap();
            i = e;
            if i == t.len() {
                break;
            }
        }
    }

    /// 8 greedy decode steps on `c` from token `t` at `pos`.
    fn cont<'g>(net: &CudaQwen2<'g>, c: &mut CudaCache<'g>, mut t: u32, pos: usize) -> Vec<u32> {
        (0..8)
            .map(|i| {
                t = net.forward_argmax(t, pos + i, c);
                t
            })
            .collect()
    }

    /// Paged pool life (spec §3.1.2), four model families (gemma3: head
    /// 256, the split kernel, its 512-position window across pages), over
    /// shuffled pages: prompts fed in uneven chunks, a slot's pages
    /// released and taken by another, a prefix kept and the rest fed again,
    /// then steps of both slots: results, tokens and KV bytes == running
    /// each alone. Out of pages: `reserve` says how many are short and
    /// takes none, `feed_slot` and `step_rows` refuse before any work.
    #[test]
    fn paged_release_refeed_four_families() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let km = Kernels::module(&g).unwrap();
        let ms: Vec<Model> = [
            crate::testutil::model(),
            crate::testutil::qwen3_model(),
            crate::testutil::llama_model(),
            crate::testutil::gemma3_model(),
        ]
        .into_iter()
        .flatten()
        .collect();
        let (ctx, steps) = (768usize, 6usize);
        for m in ms {
            let kvs: &[CacheKv] = match m.arch.as_str() {
                "qwen2" => &[CacheKv::F16, CacheKv::Q8 { group: 0 }],
                _ => &[CacheKv::F16],
            };
            for &kv in kvs {
                let net = CudaQwen2::with_kv(&g, &km, &m, ctx, kv, false).unwrap();
                let v = net.vocab;
                let what = format!("{} {kv:?}", m.arch);
                let (a, b, c) = (ids(601, 11, v), ids(300, 12, v), ids(421, 13, v));
                let ref_a = solo(&net, ctx, &a, Kind::Greedy, steps);
                let ref_c = solo(&net, ctx, &c, Kind::Greedy, steps);
                // A's 10 pages + C's 7 + one each to grow, and 2 spare.
                let need = PageMap::need(601 + steps) + PageMap::need(421 + steps);
                let mut p = net.pool_with(3, ctx, Some(need + 2), Some(0xfeed)).unwrap();
                feed_chunks(&net, &mut p, 0, &a[..600], 0, 37);
                feed_chunks(&net, &mut p, 1, &b, 0, 100);
                // Slot 1 gone: C takes its pages; A keeps 200 and refeeds.
                net.release(&mut p, 1, 0).unwrap();
                feed_chunks(&net, &mut p, 2, &c[..420], 0, 129);
                net.release(&mut p, 0, 200).unwrap();
                assert_eq!(p.page_info().unwrap().held[0], 4, "{what}");
                feed_chunks(&net, &mut p, 0, &a[200..600], 200, 65);
                // Out of pages.
                let before = p.page_info().unwrap();
                let short = net.reserve(&mut p, 1, ctx).unwrap();
                assert_eq!(short, PageMap::need(ctx) - before.free, "{what}");
                assert_eq!(p.page_info().unwrap(), before, "{what}: nothing taken");
                assert!(net.feed_slot(&mut p, 1, &b, 0).is_err(), "{what}");
                assert_eq!(p.page_info().unwrap(), before, "{what}");
                let mut hist = [a.clone(), c.clone()];
                for step in 0..steps {
                    let rows: Vec<SeqRow> = [(0usize, 0usize), (1, 2)]
                        .iter()
                        .map(|&(i, slot)| SeqRow {
                            token: hist[i][hist[i].len() - 1],
                            pos: hist[i].len() - 1,
                            slot,
                            head: RowHead::Argmax,
                        })
                        .collect();
                    let outs = net.step_rows(&mut p, &rows).unwrap();
                    for (i, o) in outs.into_iter().enumerate() {
                        let r = [&ref_a, &ref_c][i];
                        assert_eq!(o, r.0[step], "{what} request {i} step {step}");
                        let RowOut::Pick(t) = o else { unreachable!() };
                        hist[i].push(t);
                    }
                }
                assert_eq!(hist[0], ref_a.1, "{what}: tokens");
                assert_eq!(hist[1], ref_c.1, "{what}: tokens");
                assert!(
                    pool_bytes(&net, &p, 0, 600 + steps) == ref_a.2,
                    "{what}: A's KV"
                );
                assert!(
                    pool_bytes(&net, &p, 2, 420 + steps) == ref_c.2,
                    "{what}: C's KV"
                );
                // A step whose row needs a page none is free for: refused.
                let free = p.page_info().unwrap().free;
                let hog = net.reserve(&mut p, 1, free * PAGE).unwrap();
                assert_eq!(hog, 0, "{what}");
                let row = SeqRow {
                    token: 5,
                    pos: free * PAGE,
                    slot: 1,
                    head: RowHead::Argmax,
                };
                let before = p.page_info().unwrap();
                assert!(
                    matches!(net.step_rows(&mut p, &[row]), Err(StepError::Input(_))),
                    "{what}: a step past the pages"
                );
                assert_eq!(p.page_info().unwrap(), before, "{what}");
            }
        }
    }

    /// kvsave over pages (spec §3.1.5): a paged slot's image is a plain
    /// cache's (the file format doesn't change); it restores into a plain
    /// cache and into a slot of a differently shuffled pool, and both
    /// continue as the slot does.
    #[test]
    fn paged_kvsave_round_trip() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let km = Kernels::module(&g).unwrap();
        for m in models() {
            for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
                let ctx = 320;
                let net = CudaQwen2::with_kv(&g, &km, &m, ctx, kv, false).unwrap();
                let what = format!("{} {kv:?}", m.arch);
                let prompt = ids(151, 21, net.vocab);
                let n = 150;
                let mut plain = net.new_cache(ctx);
                net.feed_batch(&prompt[..n], 0, &mut plain);
                let mut want_img = Vec::new();
                net.save_kv(&plain, 0, n, &mut want_img).unwrap();
                let pa = |seed| net.pool_with(3, ctx, Some(16), Some(seed)).unwrap();
                let mut p = pa(1);
                net.feed_slot(&mut p, 2, &ids(70, 3, net.vocab), 0).unwrap();
                net.feed_slot(&mut p, 1, &prompt[..n], 0).unwrap();
                let v = net.slot_view(&p, 1).unwrap();
                let mut img = Vec::new();
                net.save_kv(&v, 0, n, &mut img).unwrap();
                assert!(img == want_img, "{what}: image");
                assert!(net.save_kv(&v, 0, 193, &mut Vec::new()).is_err(), "{what}");
                let want = cont(&net, &mut plain, prompt[n], n);
                let mut c2 = net.new_cache(ctx);
                net.load_kv(&mut c2, 0, n, &mut &img[..]).unwrap();
                assert_eq!(
                    cont(&net, &mut c2, prompt[n], n),
                    want,
                    "{what}: restored into a plain cache"
                );
                let mut q = pa(2);
                net.feed_slot(&mut q, 0, &ids(130, 4, net.vocab), 0)
                    .unwrap();
                assert_eq!(net.reserve(&mut q, 2, n + 8), Ok(0));
                let mut w = net.slot_view(&q, 2).unwrap();
                net.load_kv(&mut w, 0, n, &mut &img[..]).unwrap();
                assert_eq!(
                    cont(&net, &mut w, prompt[n], n),
                    want,
                    "{what}: restored into another pool"
                );
                let mut v = v;
                assert_eq!(net.reserve(&mut p, 1, n + 8), Ok(0));
                assert_eq!(
                    cont(&net, &mut v, prompt[n], n),
                    want,
                    "{what}: the slot itself"
                );
            }
        }
    }

    /// serve's single-request cache over a paged pool (`slot_view_bare`):
    /// no batch scratch of its own; with the pool's lent it runs the
    /// one-sequence calls (prefill, decode graphs, a verify with rejected
    /// guesses, a prefix refed) with a plain cache's picks and KV bytes;
    /// the pool refuses its own passes while it is lent, and after
    /// `return_scratch` its steps run as before (the same bits as alone).
    /// f16 and q8, qwen2 and qwen3.
    #[test]
    fn bare_view_borrows_the_pool_scratch() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let km = Kernels::module(&g).unwrap();
        for m in models() {
            for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
                let ctx = 192;
                let net = CudaQwen2::with_kv(&g, &km, &m, ctx, kv, false).unwrap();
                let what = format!("{} {kv:?}", m.arch);
                // 91 tokens: `run` uses the first 90, the pool's step after the loan
                // the 91st (at position 90). ids' first 90 are `run`'s prompt.
                let prompt = ids(91, 77, net.vocab);
                let mut plain = net.new_cache(ctx);
                let want = run(&net, &prompt, &mut plain);
                let want_kv = slot_bytes(&net, &plain, 0, 90);
                let mut p = pool_of(&net, 3, ctx, Lay::Shuffled(9));
                let mut v = net.slot_view_bare(&p, 1).unwrap();
                assert!(v.b.is_none(), "{what}: a bare view");
                assert_eq!(net.reserve(&mut p, 1, ctx), Ok(0));
                net.lend_scratch(&mut p, &mut v).unwrap();
                assert!(
                    net.feed_slot(&mut p, 0, &prompt[..5], 0).is_err(),
                    "{what}: lent"
                );
                assert_eq!(run(&net, &prompt, &mut v), want, "{what}: picks");
                assert!(pool_bytes(&net, &p, 1, 90) == want_kv, "{what}: KV bytes");
                net.return_scratch(&mut p, &mut v).unwrap();
                assert!(
                    v.b.is_none() && p.c.b.as_ref().unwrap().logits_rows,
                    "{what}"
                );
                // The pool's own step after the loan: slot 1 goes on from 90.
                let (refs, _, _) = solo(&net, ctx, &prompt[..91], Kind::Greedy, 1);
                let row = SeqRow {
                    token: prompt[90],
                    pos: 90,
                    slot: 1,
                    head: RowHead::Argmax,
                };
                // Positions 40..90 were refed by `run`: the rows are prompt's.
                assert_eq!(net.step_rows(&mut p, &[row]).unwrap(), refs, "{what}: step");
                // A plain pool's view has its own scratch: lending is a no-op.
                let mut q = pool_of(&net, 2, ctx, Lay::Plain);
                let mut w = net.slot_view_bare(&q, 0).unwrap();
                assert!(w.b.is_some(), "{what}");
                net.lend_scratch(&mut q, &mut w).unwrap();
                assert!(q.c.b.is_some(), "{what}");
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
                    pool_bytes(&net, &p, slot_of(i), n) == refs[i].2,
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
    /// gemvm routing (AXON_GEMVM_MAX 16, 32, by shape); the attention in
    /// launches of att_per_query rows (AXON_ATT_ONE=0), and in as few as the
    /// partials hold (`att_rows`), at 256 positions (64 rows: 2 launches)
    /// and in a paged pool of 2048 (1 launch).
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
            for (vm, one, cap) in [
                (16usize, true, ctx),
                (32, true, ctx),
                (0, true, ctx),
                (0, false, ctx),
                (0, true, 2048),
            ] {
                net.bf.vm_max = vm;
                net.bf.att_one = one;
                for nb in [17usize, 33, 48, 64] {
                    let mut p = match cap {
                        c if c == ctx => net.pool(slots, ctx).unwrap(),
                        c => net.pool_with(slots, c, Some(slots * 5), None).unwrap(),
                    };
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
                        let att_z = p.c.b.as_ref().unwrap().att_z;
                        match (one, cap) {
                            (false, _) => assert_eq!(att_z, 0),
                            (true, 2048) => assert_eq!(att_z, nb, "one launch"),
                            (true, _) => assert!(att_z > 0 && att_z <= nb),
                        }
                        for (i, o) in outs.into_iter().enumerate() {
                            let what = format!(
                                "{} vm {vm} one {one} cap {cap} nb {nb} request {i} step {step}",
                                m.arch
                            );
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
                        assert_eq!(
                            hist[i], refs[i].1,
                            "{} vm {vm} one {one} cap {cap} nb {nb} request {i}",
                            m.arch
                        );
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
    /// default the build's), `AXON_STEP_ONE` (`BatchFns::att_one` 1 / 0 in
    /// turn, e.g. "1,0,1,0"; default the build's), `AXON_STEP_CAP` (a paged
    /// pool of slots of this many positions holding only the pages the
    /// depth needs, as serve's; default a plain pool of depth + 64).
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
            let ones: Vec<Option<bool>> = match list("AXON_STEP_ONE", "") {
                v if v.is_empty() => vec![None],
                v => v.into_iter().map(|o| Some(o != 0)).collect(),
            };
            let cap = list("AXON_STEP_CAP", "").first().copied();
            // The GPU's temperature and SM clock now (thermal honesty: A/Bs
            // are interleaved, the start of each row logged).
            let temp = || -> String {
                std::process::Command::new("nvidia-smi")
                    .args([
                        "--query-gpu=temperature.gpu,clocks.sm",
                        "--format=csv,noheader",
                    ])
                    .output()
                    .map(|o| {
                        String::from_utf8_lossy(&o.stdout)
                            .trim()
                            .replace('\n', "; ")
                    })
                    .unwrap_or_else(|_| "?".into())
            };
            // AXON_STEP_KM: the kernel builds to run in turn per depth, e.g.
            // "plain,SOME_DEFINE,plain,SOME_DEFINE" (prefill.cu experiments).
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
                let pctx = cap.map_or(ctx, |c| c.max(ctx));
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
                    let p = match cap {
                        None => net.pool(slots, ctx),
                        Some(_) => net.pool_with(slots, pctx, Some(slots * ctx / PAGE), None),
                    };
                    match p {
                        Ok(p) => break p,
                        Err(e) if slots > 1 => {
                            eprintln!("  pool of {slots} x {pctx}: {e}");
                            slots /= 2;
                        }
                        Err(e) => panic!("{e}"),
                    }
                };
                for s in 0..slots {
                    net.feed_slot(&mut p, s, &prompt, 0).unwrap();
                }
                eprintln!(
                    "depth {depth}: decode step (forward_argmax) {d:.3} ms; pool {slots} x {pctx}{}; gpu {}",
                    if cap.is_some() { " paged" } else { "" },
                    temp()
                );
                eprintln!(
                    "| depth | nb | gemvm max | att one | att_z | ms/step | x solo | rows/s | gpu at start |"
                );
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
                    for (&vm, &one) in vms.iter().flat_map(|v| ones.iter().map(move |o| (v, o))) {
                        net.bf.vm_max = vm;
                        if let Some(o) = one {
                            net.bf.att_one = o;
                        }
                        // (The graphs hold the kernels picked when made.)
                        p.graphs.clear();
                        let t0 = temp();
                        let t = time(&mut || {
                            std::hint::black_box(net.step_rows(&mut p, &rs).unwrap());
                        });
                        eprintln!(
                            "| {depth} | {nb} | {vm} | {} | {} | {t:.3} | {:.2} | {:.0} | {t0} |",
                            net.bf.att_one as u8,
                            p.c.b.as_ref().unwrap().att_z,
                            t / d,
                            nb as f64 * 1e3 / t
                        );
                    }
                }
                net.bf.vm_max = *vms.last().unwrap();
                if let Some(o) = *ones.last().unwrap() {
                    net.bf.att_one = o;
                }
                for &nb in profs.iter().filter(|&&nb| nb <= slots) {
                    let tokens = vec![5u32; nb];
                    let at: Vec<RowAt> = (0..nb)
                        .map(|s| RowAt {
                            pos: depth as u32,
                            base: p.base(s) as u32,
                        })
                        .collect();
                    for round in 0..3 {
                        let tab = p.c.tab.as_ref().map(|t| t.len() / 4);
                        let b = p.c.b.as_mut().unwrap();
                        net.load_input(b, BatchIn::Tokens(&tokens)).unwrap();
                        b.upload_rows(&at, tab).unwrap();
                        b.att_z = net.att_rows(b, nb, depth);
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
