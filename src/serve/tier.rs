//! `axon serve --cache-ram`: the KV RAM tier (docs/design/agent-workload.md
//! §3, stage B). A conversation's KV pages are written to host memory as
//! they fill (write-behind, a budget per engine step, and the rest before
//! a slot gives them up), kept there in `kvstore::PageStore` (keyed by the
//! token prefix, LRU over unheld leaves, capped in bytes), and on admission
//! the longest stored prefix of a prompt is copied back into its slot
//! before the rest is prefilled.
//!
//! Bytes move, nothing is recomputed: a page's bytes are a function of the
//! token prefix (the model's identity is the process's: the store dies with
//! it), and the store compares a page's tokens on every hit, so a restored
//! slot holds exactly the bytes a prefill would have written and continues
//! bit for bit. `--cache-ram 0` (the default) makes none of this: no store,
//! no host memory, no copies, main's scheduling.
//!
//! Only whole pages (`PAGE` positions) are stored: a sequence's tail is
//! prefilled again. A prompt's last token is always fed (its logits), so a
//! restore stops at the whole pages below it (agent-workload.md §2.1).

use super::ops::Metrics;
use crate::cuda::HostBuf;
use crate::kvsave::{KvIo, PAGE};
use crate::kvstore::{Key, PageStore, ROOT};
use crate::lm::{Backend, BackendCache, BatchLm};
use std::io::{self, Read};
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

/// `--cache-ram auto`'s ceiling: half the RAM available after the model,
/// at most this. Host memory is only taken as pages are stored.
pub const AUTO_CEIL: u64 = 8 << 30;

/// Pinned host memory is allocated this many bytes at a time (at least a
/// page), as the store fills.
const SLAB_BYTES: usize = 64 << 20;

/// Bytes the write-behind copies per engine step while requests decode
/// (the rest wait for the next steps, or for the slot to give them up):
/// ~0.6 ms at 25 GB/s (ESTIMATED), about 1 % of a 60 ms step.
pub const STEP_BYTES: usize = 16 << 20;

/// `--cache-ram BYTES|auto`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheRam {
    /// Bytes (0: off).
    Bytes(u64),
    Auto,
}

impl CacheRam {
    /// `auto`, or bytes with an optional K/M/G/T suffix (powers of 1024).
    pub fn parse(s: &str) -> Result<CacheRam, String> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("auto") {
            return Ok(CacheRam::Auto);
        }
        let (num, shift) = match s.char_indices().last() {
            Some((i, c)) if c.is_ascii_alphabetic() => {
                let shift = match c.to_ascii_uppercase() {
                    'K' => 10,
                    'M' => 20,
                    'G' => 30,
                    'T' => 40,
                    'B' => 0,
                    _ => return Err(format!("--cache-ram {s}: BYTES[K|M|G|T] or auto")),
                };
                (&s[..i], shift)
            }
            _ => (s, 0),
        };
        num.parse::<u64>()
            .ok()
            .and_then(|n| n.checked_mul(1u64 << shift))
            .map(CacheRam::Bytes)
            .ok_or_else(|| format!("--cache-ram {s}: BYTES[K|M|G|T] or auto"))
    }

    /// The store's size in bytes given `avail` bytes of RAM available now
    /// (after the model): `auto` is half of it, at most `AUTO_CEIL` (0 if
    /// unknown); a size past `avail` is refused (pinned memory can't be
    /// swapped out).
    pub fn bytes(self, avail: Option<u64>) -> Result<u64, String> {
        match self {
            CacheRam::Bytes(0) => Ok(0),
            CacheRam::Auto => Ok(avail.map_or(0, |a| (a / 2).min(AUTO_CEIL))),
            CacheRam::Bytes(n) => match avail {
                Some(a) if n > a => Err(format!(
                    "--cache-ram {}: more than the {} of RAM available beside the model \
                     (pinned memory can't be swapped out); a smaller size, or auto",
                    gib(n),
                    gib(a)
                )),
                _ => Ok(n),
            },
        }
    }
}

pub fn gib(n: u64) -> String {
    format!("{:.2} GiB", n as f64 / (1u64 << 30) as f64)
}

/// The page store and, per sequence (a pool slot, or the one cache), the
/// keys of the whole pages of what it holds that are stored, in order. A
/// sequence's last stored page is held in the store (so a chain a slot is
/// still writing never loses its leaf, and so none of its prefix).
pub(super) struct Tier<P> {
    pub store: PageStore<P>,
    /// Image bytes of a page.
    pub page: usize,
    keys: Vec<Vec<Key>>,
    metrics: Option<Arc<Metrics>>,
}

impl<P> Tier<P> {
    fn new(cap: usize, page: usize, seqs: usize, metrics: Option<Arc<Metrics>>) -> Tier<P> {
        let pages = cap / page.max(1);
        if let Some(m) = &metrics {
            m.kv_ram_cap.store((pages * page) as u64, Relaxed);
        }
        Tier {
            store: PageStore::new(pages * page),
            page,
            keys: vec![Vec::new(); seqs],
            metrics,
        }
    }

    /// Pages of sequence `s` stored.
    pub fn stored(&self, s: usize) -> usize {
        self.keys[s].len()
    }

    /// Positions of `head`'s longest stored prefix (whole pages; no LRU
    /// change).
    pub fn peek(&self, head: &[u32]) -> usize {
        self.store.peek_len(head)
    }

    /// Sequence `s`'s stored pages are `keys` now (the hold moves to its
    /// last one).
    fn set_keys(&mut self, s: usize, keys: Vec<Key>) {
        if let Some(&k) = keys.last() {
            self.store.hold(k);
        }
        if let Some(&k) = self.keys[s].last() {
            self.store.release(k);
        }
        self.keys[s] = keys;
    }

    /// Sequence `s` keeps the first `pages` of its stored pages.
    pub fn truncate(&mut self, s: usize, pages: usize) {
        if pages < self.keys[s].len() {
            let k = self.keys[s][..pages].to_vec();
            self.set_keys(s, k);
        }
    }

    /// Sequence `s` gains the stored page `key` (the next of its pages).
    fn push(&mut self, s: usize, key: Key) {
        let mut k = self.keys[s].clone();
        k.push(key);
        self.set_keys(s, k);
    }

    /// Sequence `s`, whose first `have` positions hold `head`'s, takes the
    /// keys of `head`'s stored prefix it holds already (no copy needed).
    fn adopt_held(&mut self, s: usize, ks: &[Key], have: usize) {
        let n = ks.len().min(have / PAGE);
        if n > self.keys[s].len() {
            self.set_keys(s, ks[..n].to_vec());
        }
    }

    fn publish(&self, written: usize, restored: usize) {
        if let Some(m) = &self.metrics {
            m.kv_ram_bytes.store(self.store.used() as u64, Relaxed);
            m.kv_ram_pages.store(self.store.len() as u64, Relaxed);
            m.kv_ram_written.fetch_add(written as u64, Relaxed);
            m.kv_ram_restored.fetch_add(restored as u64, Relaxed);
            if restored > 0 {
                m.kv_ram_hits.fetch_add(1, Relaxed);
            }
        }
    }
}

/// How the arena's page-locked memory is taken (`AXON_KV_RAM_ALLOC`): all
/// of it, at the size asked for, before the server starts (`Start`), or by
/// a thread from server start on, the tier growing as slabs arrive
/// (`Thread`). Never inside an engine step: a slab's `cuMemHostAlloc` took
/// 61.8 ms per 64 MiB on a 4090 pod (MEASURED, 1027-100925-fbcb), and grown
/// lazily in steps it cost +21.5 % ITL p99 there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alloc {
    Start,
    Thread,
}

impl Alloc {
    /// AXON_KV_RAM_ALLOC=start|thread (default: start).
    pub fn from_env() -> Alloc {
        match std::env::var("AXON_KV_RAM_ALLOC").as_deref() {
            Ok("thread") => Alloc::Thread,
            _ => Alloc::Start,
        }
    }
}

/// A slab pinned by the allocator thread: (index, address), or why it
/// stopped.
type SlabMsg = Result<(usize, usize), String>;

/// Page-locked host memory in fixed-size page slots: `most` pages in slabs
/// of `per`, and a free list of slots. Copies go straight between a slot's
/// device rows and a page slot, on the stream: the host never touches the
/// bytes, so stream order is all the synchronisation there is (a slot freed
/// by an eviction and written again is written after every copy enqueued
/// before), and the bytes are never zeroed (a slot is copied from only
/// after a copy into it).
pub(super) struct Arena {
    /// By slab index; None until pinned (`Alloc::Thread`).
    slabs: Vec<Option<HostBuf<'static>>>,
    free: Vec<u32>,
    page: usize,
    per: usize,
    most: usize,
    /// Slabs the allocator thread pins, as each is ready (dropped with
    /// slabs in it only at shutdown: those few are left to the process's
    /// exit).
    rx: Option<std::sync::mpsc::Receiver<SlabMsg>>,
    /// Slabs in `slabs` so far.
    ready: usize,
    t0: Instant,
    /// Seconds from start until every slab was pinned.
    pub done_s: Option<f64>,
    log: bool,
}

impl Arena {
    /// Pages of slab `k`.
    fn slab_pages(&self, k: usize) -> usize {
        self.per.min(self.most - k * self.per)
    }

    fn new(
        net: &Backend,
        page: usize,
        most: usize,
        how: Alloc,
        log: bool,
    ) -> Result<Arena, String> {
        let per = (SLAB_BYTES / page).max(1);
        let n = most.div_ceil(per);
        let mut a = Arena {
            slabs: (0..n).map(|_| None).collect(),
            free: Vec::new(),
            page,
            per,
            most,
            rx: None,
            ready: 0,
            t0: Instant::now(),
            done_s: None,
            log,
        };
        match how {
            Alloc::Start => {
                for k in 0..n {
                    let b = net.host_alloc(a.slab_pages(k) * page)?;
                    a.adopt(k, b);
                }
                a.done_s = Some(a.t0.elapsed().as_secs_f64());
            }
            Alloc::Thread => {
                let al = net
                    .host_allocator()
                    .ok_or("no page-locked memory on this backend")?;
                let sizes: Vec<usize> = (0..n).map(|k| a.slab_pages(k) * page).collect();
                let (tx, rx) = std::sync::mpsc::channel::<SlabMsg>();
                std::thread::Builder::new()
                    .name("axon-kv-ram".into())
                    .spawn(move || {
                        for (k, &len) in sizes.iter().enumerate() {
                            let r = al.alloc(len);
                            let stop = r.is_err();
                            if let Err(std::sync::mpsc::SendError(Ok((_, addr)))) =
                                tx.send(r.map(|addr| (k, addr)))
                            {
                                // Nobody takes it any more (the server is
                                // stopping): give it back.
                                // SAFETY: ours, never adopted, no copy.
                                unsafe { al.free(addr) };
                                return;
                            }
                            if stop {
                                return;
                            }
                        }
                    })
                    .map_err(|e| format!("the KV RAM tier's allocator thread: {e}"))?;
                a.rx = Some(rx);
            }
        }
        Ok(a)
    }

    /// Slab `k`'s memory is `b`: its slots are free.
    fn adopt(&mut self, k: usize, b: HostBuf<'static>) {
        let base = k * self.per;
        let n = self.slab_pages(k);
        self.slabs[k] = Some(b);
        self.free.extend((base..base + n).rev().map(|i| i as u32));
        self.ready += 1;
    }

    /// Take the slabs the allocator thread has pinned since the last call.
    fn poll(&mut self, net: &Backend) {
        let Some(rx) = &self.rx else {
            return;
        };
        let mut got = Vec::new();
        let mut end = None;
        loop {
            match rx.try_recv() {
                Ok(Ok(s)) => got.push(s),
                Ok(Err(e)) => {
                    end = Some(Some(e));
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    end = Some(None);
                    break;
                }
            }
        }
        for (k, addr) in got {
            let len = self.slab_pages(k) * self.page;
            // SAFETY: a live `host_allocator` allocation of `len` bytes,
            // sent once, owned by nobody else; page slots are written
            // before they are read.
            let b = unsafe { net.host_adopt(addr, len) };
            self.adopt(k, b);
        }
        if self.ready == self.slabs.len() && self.done_s.is_none() {
            let s = self.t0.elapsed().as_secs_f64();
            self.done_s = Some(s);
            self.rx = None;
            if self.log {
                eprintln!(
                    "axon serve: KV RAM tier: {} pinned in {s:.1} s (a thread, from start)",
                    gib(self.bytes() as u64)
                );
            }
        } else if let Some(e) = end {
            self.rx = None;
            eprintln!(
                "axon serve: KV RAM tier: pinned {} of {} ({}); it stays that size",
                gib(self.bytes() as u64),
                gib((self.most * self.page) as u64),
                e.unwrap_or_else(|| "its thread ended".into())
            );
        }
    }

    /// A free page slot (None: every pinned one in use).
    fn take(&mut self, net: &Backend) -> Option<u32> {
        self.poll(net);
        self.free.pop()
    }

    /// Slot `i`'s slab and byte offset.
    fn at(&mut self, i: u32) -> (&mut HostBuf<'static>, usize) {
        let (s, k) = (i as usize / self.per, i as usize % self.per);
        let b = self.slabs[s].as_mut().expect("a slot of a pinned slab");
        (b, k * self.page)
    }

    /// Bytes pinned so far.
    pub fn bytes(&self) -> usize {
        (0..self.slabs.len())
            .filter(|&k| self.slabs[k].is_some())
            .map(|k| self.slab_pages(k) * self.page)
            .sum()
    }
}

/// The tier of a slot pool (`Batch`): pages in pinned host memory.
pub(super) struct PoolTier {
    pub t: Tier<u32>,
    arena: Arena,
    /// The slot the write-behind looks at first next (round robin).
    next: usize,
}

impl PoolTier {
    /// A tier of `cap` bytes for `slots` slots of `pool`, or None: no pages
    /// (`cap` under a page), or no page image on this backend; `how`: how
    /// its page-locked memory is taken (all of it, now or by a thread from
    /// now on); Err: it couldn't be.
    pub fn new(
        net: &Backend,
        pool: &<Backend<'_> as BatchLm>::Pool,
        slots: usize,
        cap: u64,
        how: Alloc,
        log: bool,
        metrics: Option<Arc<Metrics>>,
    ) -> Result<Option<PoolTier>, String> {
        let Some(page) = net.page_bytes(pool) else {
            return Ok(None);
        };
        let Ok(cap) = usize::try_from(cap) else {
            return Ok(None);
        };
        let t = Tier::new(cap, page, slots, metrics);
        let most = t.store.cap() / page;
        if most == 0 {
            return Ok(None);
        }
        let arena = Arena::new(net, page, most, how, log)?;
        let p = PoolTier { t, arena, next: 0 };
        p.publish_pinned();
        Ok(Some(p))
    }

    /// How the memory was taken, for the banner.
    pub fn how(&self) -> String {
        match self.arena.done_s {
            Some(s) => format!("pinned at start in {s:.1} s"),
            None => "pinned by a thread from start".into(),
        }
    }

    fn publish_pinned(&self) {
        if let Some(m) = &self.t.metrics {
            m.kv_ram_pinned.store(self.arena.bytes() as u64, Relaxed);
        }
    }

    /// Store up to `most` of slot `s`'s pages not yet stored (whole pages
    /// of `held`, the tokens whose rows it holds), in order; a page some
    /// other sequence stored already costs no copy (and no budget). The
    /// pages copied (stops early when the store has no room).
    pub fn write(
        &mut self,
        net: &Backend,
        pool: &<Backend<'_> as BatchLm>::Pool,
        s: usize,
        held: &[u32],
        most: usize,
    ) -> Result<usize, String> {
        let mut copied = 0;
        let whole = held.len() / PAGE;
        while self.t.stored(s) < whole && copied < most {
            let i = self.t.stored(s);
            let toks = &held[i * PAGE..(i + 1) * PAGE];
            let parent = self.t.keys[s].last().copied().unwrap_or(ROOT);
            if let Some(k) = self.t.store.child(parent, toks) {
                self.t.push(s, k);
                continue;
            }
            let room = self.t.store.room(parent, self.t.page);
            self.arena.free.extend(self.t.store.take_evicted());
            if !room {
                break;
            }
            let Some(a) = self.arena.take(net) else {
                break;
            };
            let (buf, at) = self.arena.at(a);
            net.page_out(pool, s, i, buf, at)?;
            match self.t.store.insert(parent, toks, a, self.t.page) {
                Some(k) => self.t.push(s, k),
                // A hash collision with another prefix: keep what's there.
                None => {
                    self.arena.free.push(a);
                    break;
                }
            }
            copied += 1;
        }
        if copied > 0 {
            self.t.publish(copied, 0);
        }
        self.publish_pinned();
        Ok(copied)
    }

    /// The page-level eviction hook (PR C's `PrefixIndex::pop_lru_leaf`,
    /// agreed on the r10 thread): the GPU page `page`, the block `tokens`
    /// after the prefix `parent`, is about to be reused. Already stored
    /// (the usual case: pages are written behind as they fill): touched,
    /// no copy. Else, if its parent is stored (or it is a first page),
    /// copied (D2H, ordered before any later reuse of the page) and
    /// stored; else dropped (an orphan; its prefix is prefilled again).
    /// Whether the store has it after.
    #[allow(dead_code)] // PR C3 calls it (its page-level eviction); the seam agreed on the r10 thread
    pub fn spill_page(
        &mut self,
        net: &Backend,
        pool: &<Backend<'_> as BatchLm>::Pool,
        parent: Key,
        tokens: &[u32],
        page: u32,
    ) -> Result<bool, String> {
        if self.t.store.child(parent, tokens).is_some() {
            return Ok(true);
        }
        let room = self.t.store.room(parent, self.t.page);
        self.arena.free.extend(self.t.store.take_evicted());
        if !room {
            return Ok(false);
        }
        let Some(a) = self.arena.take(net) else {
            return Ok(false);
        };
        let (buf, at) = self.arena.at(a);
        net.page_out_id(pool, page, buf, at)?;
        match self.t.store.insert(parent, tokens, a, self.t.page) {
            Some(_) => {
                self.t.publish(1, 0);
                Ok(true)
            }
            None => {
                self.arena.free.push(a);
                Ok(false)
            }
        }
    }

    /// The write-behind of an engine step: up to `bytes` of copies over
    /// every slot's unstored pages, round robin from where the last one
    /// stopped. `held`: each slot's tokens.
    pub fn write_some(
        &mut self,
        net: &Backend,
        pool: &<Backend<'_> as BatchLm>::Pool,
        held: &[Vec<u32>],
        bytes: usize,
    ) -> Result<(), String> {
        let mut left = bytes.div_ceil(self.t.page).max(1);
        let n = held.len();
        for k in 0..n {
            let s = (self.next + k) % n;
            left -= self.write(net, pool, s, &held[s], left)?;
            if left == 0 {
                self.next = s;
                return Ok(());
            }
        }
        Ok(())
    }

    /// The pages of `head`'s longest stored prefix (held: nothing evicts
    /// them until `restored`), and their slots.
    pub fn lookup(&mut self, head: &[u32]) -> Vec<(Key, u32)> {
        let ks = self.t.store.lookup(head);
        ks.iter()
            .map(|&k| {
                self.t.store.hold(k);
                (k, *self.t.store.page(k).expect("looked up"))
            })
            .collect()
    }

    /// Slot `s` holds the first `have` positions of the prefix `pages` (a
    /// `lookup`) stores, and pages for its first `to` pages: copy in what
    /// it lacks of those, then let `pages` go. The slot's stored pages are
    /// then `pages[..to]` (or its own, if longer).
    pub fn restore(
        &mut self,
        net: &Backend,
        pool: &<Backend<'_> as BatchLm>::Pool,
        s: usize,
        pages: &[(Key, u32)],
        have: usize,
        to: usize,
    ) -> Result<(), String> {
        let from = have / PAGE;
        let mut res = Ok(());
        for (i, &(_, a)) in pages.iter().enumerate().take(to).skip(from) {
            let (buf, at) = self.arena.at(a);
            res = net.page_in(pool, s, i, buf, at);
            if res.is_err() {
                break;
            }
        }
        let keys: Vec<Key> = pages.iter().map(|p| p.0).collect();
        if res.is_ok() {
            self.t.adopt_held(s, &keys, (to * PAGE).max(have));
            self.t.publish(0, (to * PAGE).saturating_sub(have));
        }
        for &k in &keys {
            self.t.store.release(k);
        }
        res
    }
}

/// The tier of the one cache (no pool: the CPU, `--parallel 1`): pages in
/// ordinary memory, through the backend's cache image (`KvIo`).
pub(super) struct CacheTier {
    pub t: Tier<Box<[u8]>>,
}

impl CacheTier {
    /// A tier of `cap` bytes for `cache`, or None: no image of it, or
    /// `cap` under a page.
    pub fn new(
        net: &Backend,
        cache: &BackendCache,
        cap: u64,
        metrics: Option<Arc<Metrics>>,
    ) -> Option<CacheTier> {
        let ident = net.kv_ident(cache)?;
        let row = ident.get("row_bytes")?.as_u64()? as usize;
        let t = Tier::new(usize::try_from(cap).ok()?, row * PAGE, 1, metrics);
        (t.store.cap() > 0).then_some(CacheTier { t })
    }

    /// Store the whole pages of `hist` (the tokens whose rows `cache`
    /// holds) not stored yet.
    pub fn write(&mut self, net: &Backend, cache: &BackendCache, hist: &[u32]) -> io::Result<()> {
        let (from, to) = (self.t.stored(0), hist.len() / PAGE);
        if from >= to {
            return Ok(());
        }
        let mut img = Vec::with_capacity((to - from) * self.t.page);
        net.kv_save(cache, from * PAGE, to * PAGE, &mut img)?;
        let mut copied = 0;
        for (i, b) in (from..to).zip(img.chunks_exact(self.t.page)) {
            let toks = &hist[i * PAGE..(i + 1) * PAGE];
            let parent = self.t.keys[0].last().copied().unwrap_or(ROOT);
            if let Some(k) = self.t.store.child(parent, toks) {
                self.t.push(0, k);
                continue;
            }
            if !self.t.store.room(parent, self.t.page) {
                break;
            }
            drop(self.t.store.take_evicted());
            match self.t.store.insert(parent, toks, b.into(), self.t.page) {
                Some(k) => self.t.push(0, k),
                None => break,
            }
            copied += 1;
        }
        self.t.publish(copied, 0);
        Ok(())
    }

    /// `cache` holds `have` positions of `head` (the prompt but its last
    /// token): copy the rest of `head`'s longest stored prefix in. The
    /// positions it holds after (>= `have`).
    pub fn restore(
        &mut self,
        net: &Backend,
        cache: &mut BackendCache,
        head: &[u32],
        have: usize,
    ) -> io::Result<usize> {
        self.t.truncate(0, have / PAGE);
        let ks = self.t.store.lookup(head);
        if ks.len() * PAGE <= have {
            self.t.adopt_held(0, &ks, have);
            return Ok(have);
        }
        let from = have / PAGE;
        let mut src = Pages {
            pages: ks[from..]
                .iter()
                .map(|k| &self.t.store.page(*k).expect("looked up")[..])
                .collect(),
            at: 0,
        };
        net.kv_load(cache, from * PAGE, ks.len() * PAGE, &mut src)?;
        self.t.adopt_held(0, &ks, ks.len() * PAGE);
        self.t.publish(0, ks.len() * PAGE - have);
        Ok(ks.len() * PAGE)
    }
}

/// Pages' images read one after another.
struct Pages<'a> {
    pages: Vec<&'a [u8]>,
    at: usize,
}

impl Read for Pages<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while let Some(p) = self.pages.first() {
            if self.at < p.len() {
                let n = buf.len().min(p.len() - self.at);
                buf[..n].copy_from_slice(&p[self.at..self.at + n]);
                self.at += n;
                return Ok(n);
            }
            self.pages.remove(0);
            self.at = 0;
        }
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_ram_parses() {
        assert_eq!(CacheRam::parse("0"), Ok(CacheRam::Bytes(0)));
        assert_eq!(CacheRam::parse("auto"), Ok(CacheRam::Auto));
        assert_eq!(CacheRam::parse("6G"), Ok(CacheRam::Bytes(6 << 30)));
        assert_eq!(CacheRam::parse("512m"), Ok(CacheRam::Bytes(512 << 20)));
        assert_eq!(CacheRam::parse("1234"), Ok(CacheRam::Bytes(1234)));
        assert!(CacheRam::parse("6X").is_err());
        assert!(CacheRam::parse("G").is_err());
        assert!(CacheRam::parse("-1").is_err());
        assert!(CacheRam::parse("99999999999T").is_err());
    }

    #[test]
    fn cache_ram_sizes() {
        let g = 1u64 << 30;
        assert_eq!(CacheRam::Bytes(0).bytes(Some(g)), Ok(0));
        assert_eq!(CacheRam::Auto.bytes(Some(10 * g)), Ok(5 * g));
        assert_eq!(CacheRam::Auto.bytes(Some(100 * g)), Ok(AUTO_CEIL));
        assert_eq!(CacheRam::Auto.bytes(None), Ok(0));
        assert_eq!(CacheRam::Bytes(2 * g).bytes(Some(3 * g)), Ok(2 * g));
        assert!(CacheRam::Bytes(4 * g).bytes(Some(3 * g)).is_err());
        assert_eq!(CacheRam::Bytes(4 * g).bytes(None), Ok(4 * g));
    }

    #[test]
    fn pages_read_in_order() {
        let (a, b) = (vec![1u8, 2, 3], vec![4u8, 5]);
        let mut r = Pages {
            pages: vec![&a, &b],
            at: 0,
        };
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4, 5]);
    }

    /// Holds follow a sequence's last stored page: a chain being written
    /// never loses its leaf, and truncating lets the old leaf go.
    #[test]
    fn keys_hold_the_leaf() {
        let mut t: Tier<u8> = Tier::new(3, 1, 2, None);
        let a: Vec<u32> = (0..3 * PAGE as u32).collect();
        let mut parent = ROOT;
        for b in a.chunks_exact(PAGE) {
            parent = t.store.insert(parent, b, 0, 1).unwrap();
            t.push(0, parent);
        }
        // Full; the leaf is held: nothing can go.
        let other: Vec<u32> = (1000..1000 + PAGE as u32).collect();
        assert!(!t.store.room(ROOT, 1));
        t.truncate(0, 1);
        // The old leaf goes, then the middle page; the first is held.
        assert!(t.store.room(ROOT, 1));
        t.store.insert(ROOT, &other, 0, 1).unwrap();
        assert_eq!(t.store.take_evicted().len(), 1);
        assert_eq!(t.peek(&a), 2 * PAGE);
        t.truncate(0, 0);
        assert!(t.store.room(ROOT, 1));
    }
}
