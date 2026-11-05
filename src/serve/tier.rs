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
use crate::cuda::{HostAllocator, HostBuf};
use crate::kvsave::{KvIo, PAGE};
use crate::kvstore::{Key, PageStore, ROOT};
use crate::lm::{Backend, BackendCache, BatchLm};
use std::io::{self, Read};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// `--cache-ram auto`'s ceiling: half the RAM available after the model,
/// at most this. On the GPU the whole size is page-locked (unswappable)
/// from server start, used or not (`Alloc`).
pub const AUTO_CEIL: u64 = 8 << 30;

/// The arena is page-locked this many bytes at a time (at least a page),
/// all of it from server start (`Alloc`).
const SLAB_BYTES: usize = 64 << 20;

/// Bytes the write-behind copies per engine step while requests decode
/// (the rest wait for the next steps, or for the slot to give them up):
/// ~0.6 ms at 25 GB/s (ESTIMATED), about 1 % of a 60 ms step.
pub const STEP_BYTES: usize = 16 << 20;

/// Bytes the idle flush copies per piece (#170: the engine loop checks for
/// a new request between pieces, so one waits for at most one): 19 pages
/// of a 7B f16, ≈ 2.5 ms at 28 GB/s, ≈ 10 ms at bella's slow 2D copy
/// (ESTIMATED, docs/design/kv-ram-tier-exclusive.md §2.5).
pub const IDLE_BYTES: usize = 64 << 20;

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

/// A tier error (`e`, in its `what`: a spill, the write-behind or a
/// restore) turned the KV RAM tier off (its owner has dropped it): one
/// line on stderr, the same on every path, and /metrics' tier gauges go
/// to 0 (its counters stay).
pub(super) fn off(what: &str, e: &str, metrics: Option<&Metrics>) {
    let line = format!("axon serve: --cache-ram: {what}: {e}; the KV RAM tier is off from here");
    eprintln!("{line}");
    if let Some(m) = metrics {
        m.kv_ram_off();
        #[cfg(test)]
        m.tier_faults.said(line, m);
    }
}

/// What an injected fault's error says (`Faults`).
#[cfg(test)]
pub const FAULT: &str = "an injected KV RAM tier fault (a test's)";

/// Tests (#172): faults injected into the KV RAM tier's copies, and the
/// lines `off` said, per server (`Metrics::tier_faults`: the one handle a
/// test's `Server` and the engine's tier both hold). None of it is built
/// outside tests. A copy is a pool's `page_out` / `page_in` (a page), or
/// the one cache's `kv_save` / `kv_load` (all the pages a write or a
/// restore moves), each made for one `Op`. The copy an armed fault hits
/// fails instead of running: a failed copy in leaves the rows it was to
/// fill as they were (another conversation's), so a reply that used them
/// anyway would differ.
#[cfg(test)]
#[derive(Default)]
pub struct Faults {
    /// The fault armed: the op, and its copies until one fails, it counted.
    armed: Mutex<Option<(Op, usize)>>,
    /// The copy the armed fault failed: its op, and the slot it was for
    /// (the one cache: 0; `PoolTier::spill_page`'s, no slot's: usize::MAX).
    hit: Mutex<Option<(Op, usize)>>,
    /// The lines `off` said on stderr, each with the tier's counters then
    /// (`Counters`).
    said: Mutex<Vec<(String, Counters)>>,
    /// Times the engine waited for a request with nothing to do (`idle`).
    idles: std::sync::atomic::AtomicU64,
    /// #170: the idle flush's piece, in bytes (0: `IDLE_BYTES`).
    idle_bytes: std::sync::atomic::AtomicUsize,
    /// #170: what the engine loop did while its batch was idle (`Ev`).
    events: Mutex<Vec<Ev>>,
    /// #170: the idle piece (counted from the start) after which the
    /// engine waits for `unpark` (0: none).
    park_at: std::sync::atomic::AtomicUsize,
    /// Whether the engine waits (`park_at`), and its wake-up.
    parked: Mutex<bool>,
    unparked: std::sync::Condvar,
    /// #170: the write-behind finds no free page slot in the arena.
    starve: std::sync::atomic::AtomicBool,
}

/// Tests (#170): what the engine loop did while its batch was idle, in
/// order (`Faults::events`).
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ev {
    /// A piece of the idle flush ran; the tier's pages written after it.
    Piece(u64),
    /// It took a request off the channel.
    Job,
}

/// Tests: what a tier copy is made for, as the off line names it (`off`'s
/// `what`). An armed fault counts only its op's copies, so the copies of
/// another (a write-behind's before a spill's) never take it.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// A slot's pages before it gives its rows up (`Batch::spill`,
    /// `PoolTier::spill_page`).
    Spill,
    /// The pages that filled meanwhile: an engine step's
    /// (`Batch::write_behind`), or the one cache's after a request
    /// (`Engine::run`).
    WriteBehind,
    /// A stored prefix copied in (`Batch::restore`, `Engine::job`).
    Restore,
}

#[cfg(test)]
impl Op {
    /// The off line's name of it.
    pub fn what(self) -> &'static str {
        match self {
            Op::Spill => "spill",
            Op::WriteBehind => "write-behind",
            Op::Restore => "restore",
        }
    }
}

/// Tests: the KV RAM tier's /metrics counters (pages written, tokens
/// restored, hits, pages spilled).
#[cfg(test)]
pub type Counters = [u64; 4];

#[cfg(test)]
impl Faults {
    /// The `n`-th copy made for `op` from now fails (1: the next); the
    /// fault armed before is gone.
    pub fn fail(&self, op: Op, n: usize) {
        *self.armed.lock().unwrap_or_else(|p| p.into_inner()) = (n > 0).then_some((op, n));
    }

    /// The copy the last fault failed (`hit`), if one did.
    pub fn hit(&self) -> Option<(Op, usize)> {
        *self.hit.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Times the engine waited for a request with nothing to do (every
    /// 100 ms while it waits): once it is past the count a test read after
    /// a reply, what the engine does after a reply (a pool's idle flush,
    /// the one cache's write, a batch's last write-behind) is done.
    /// Acquire, against `idle`'s Release: what the engine wrote before it
    /// waited (its metrics, `said`) is visible to a test that saw the count
    /// move, by the memory model rather than x86's ordering.
    pub fn idles(&self) -> u64 {
        self.idles.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The idle flush's pieces are `bytes` from now (a page: a few pages
    /// of backlog are as many pieces).
    pub fn set_idle_bytes(&self, bytes: usize) {
        self.idle_bytes.store(bytes, Relaxed);
    }

    /// The idle flush's piece (`set_idle_bytes`, else `IDLE_BYTES`).
    pub(super) fn idle_bytes(&self) -> usize {
        match self.idle_bytes.load(Relaxed) {
            0 => IDLE_BYTES,
            n => n,
        }
    }

    /// What the engine loop did while idle so far.
    pub fn events(&self) -> Vec<Ev> {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// The engine waits after the `n`-th piece of the idle flush (counted
    /// from the start) until `unpark` (30 s at most: a failed test still
    /// stops).
    pub fn park_after(&self, n: usize) {
        self.park_at.store(n, Relaxed);
    }

    /// Wait until the engine is parked (`park_after`); panics after 30 s.
    pub fn wait_parked(&self) {
        let p = self.parked.lock().unwrap_or_else(|p| p.into_inner());
        let (p, _) = self
            .unparked
            .wait_timeout_while(p, std::time::Duration::from_secs(30), |p| !*p)
            .unwrap_or_else(|p| p.into_inner());
        assert!(*p, "the engine never parked: {:?}", self.events());
    }

    /// The parked engine goes on.
    pub fn unpark(&self) {
        *self.parked.lock().unwrap_or_else(|p| p.into_inner()) = false;
        self.unparked.notify_all();
    }

    /// A piece of the idle flush ran (`m`: the tier's counters now); the
    /// one to park after waits here.
    pub(super) fn piece(&self, m: &Metrics) {
        let n = {
            let mut e = self.events.lock().unwrap_or_else(|p| p.into_inner());
            e.push(Ev::Piece(m.kv_ram_written.load(Relaxed)));
            e.iter().filter(|e| matches!(e, Ev::Piece(_))).count()
        };
        if n == self.park_at.load(Relaxed) {
            let mut p = self.parked.lock().unwrap_or_else(|p| p.into_inner());
            *p = true;
            self.unparked.notify_all();
            let (mut p, _) = self
                .unparked
                .wait_timeout_while(p, std::time::Duration::from_secs(30), |p| *p)
                .unwrap_or_else(|p| p.into_inner());
            *p = false;
        }
    }

    /// The engine loop, idle, took a request off the channel.
    pub(super) fn job(&self) {
        self.events
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(Ev::Job);
    }

    /// The write-behind finds no free page slot in the arena while `on`
    /// (it copies nothing: the idle flush can't progress).
    pub fn starve(&self, on: bool) {
        self.starve.store(on, Relaxed);
    }

    /// The engine waits for a request with nothing to do (`idles`).
    pub(super) fn idle(&self) {
        self.idles
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// What `off` said so far, each line with the counters then.
    pub fn off_lines(&self) -> Vec<(String, Counters)> {
        self.said.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// `off` said `line`; `m`: the counters now.
    fn said(&self, line: String, m: &Metrics) {
        let c = [
            &m.kv_ram_written,
            &m.kv_ram_restored,
            &m.kv_ram_hits,
            &m.kv_ram_spilled,
        ]
        .map(|c| c.load(Relaxed));
        self.said
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((line, c));
    }

    /// A copy for `op` into or out of slot `slot` is to be made: Err if it
    /// is the one armed.
    fn fault(&self, op: Op, slot: usize) -> Result<(), String> {
        let mut a = self.armed.lock().unwrap_or_else(|p| p.into_inner());
        let Some((o, n)) = a.as_mut() else {
            return Ok(());
        };
        if *o != op {
            return Ok(());
        }
        *n -= 1;
        if *n > 0 {
            return Ok(());
        }
        *a = None;
        *self.hit.lock().unwrap_or_else(|p| p.into_inner()) = Some((op, slot));
        Err(format!("{FAULT}: a copy for a {}", op.what()))
    }
}

/// The page store and, per sequence (a pool slot, or the one cache), the
/// keys of the whole pages of what it holds that are stored, in order (a
/// chain from ROOT). While a sequence holds their rows those pages are
/// class 0 in the store (`resident`: the GPU has their bytes), so the store
/// evicts them first; the sequence then lets go of the evicted ones
/// (`drain`), and its rows stay. A page whose rows no sequence holds any
/// more is class 1, RAM's own (docs/design/kv-ram-tier-exclusive.md §2).
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

    /// Sequence `s`'s stored pages are `keys` now (a stored chain from
    /// ROOT): those it no longer names lose it as a device holder, the new
    /// ones gain it.
    fn set_keys(&mut self, s: usize, keys: Vec<Key>) {
        let old = &self.keys[s];
        let c = old.iter().zip(&keys).take_while(|(a, b)| a == b).count();
        for &k in &old[c..] {
            self.store.resident(k, false);
        }
        for &k in &keys[c..] {
            self.store.resident(k, true);
        }
        self.keys[s] = keys;
    }

    /// Sequence `s` keeps the first `pages` of its stored pages: the rest
    /// are RAM's own now, unless another sequence holds them too.
    pub fn truncate(&mut self, s: usize, pages: usize) {
        if pages < self.keys[s].len() {
            for &k in &self.keys[s][pages..] {
                self.store.resident(k, false);
            }
            self.keys[s].truncate(pages);
            if let Some(m) = &self.metrics {
                m.kv_ram_gpu_pages.store(self.store.dups() as u64, Relaxed);
            }
        }
    }

    /// Sequence `s` gains the stored page `key` (the next of its pages).
    fn push(&mut self, s: usize, key: Key) {
        self.store.resident(key, true);
        self.keys[s].push(key);
    }

    /// The pages the store evicted since the last call (their memory goes
    /// back to its owner). An evicted page is a leaf, so it is the last of
    /// each sequence's keys that names it (a `make_room` that took a leaf
    /// and then its parent took the last two): those let go of it. Their
    /// rows stay; the write-behind or their spill copies it again.
    fn drain(&mut self) -> Vec<P> {
        let ev = self.store.take_evicted();
        if ev.is_empty() {
            return Vec::new();
        }
        let gone: std::collections::HashSet<Key> = ev.iter().map(|e| e.0).collect();
        for ks in &mut self.keys {
            while ks.last().is_some_and(|k| gone.contains(k)) {
                ks.pop();
            }
        }
        ev.into_iter().map(|(_, p)| p).collect()
    }

    /// Sequence `s`'s keys before they are extended: its last stored page
    /// still the child of the one before it (`drain` keeps it so; else, a
    /// bug, they are cut to their longest stored chain).
    fn revalidate(&mut self, s: usize) {
        let ks = &self.keys[s];
        let link = |i: usize| match i {
            0 => ROOT,
            _ => ks[i - 1],
        };
        let n = ks.len();
        if n == 0 || self.store.parent_of(ks[n - 1]) == Some(link(n - 1)) {
            return;
        }
        if cfg!(debug_assertions) {
            panic!("sequence {s}: its keys name an unstored page");
        }
        let v = (0..n)
            .take_while(|&i| self.store.parent_of(ks[i]) == Some(link(i)))
            .count();
        let cut = self.keys[s].split_off(v);
        for k in cut {
            if self.store.parent_of(k).is_some() {
                self.store.resident(k, false);
            }
        }
    }

    /// Debug builds: the sequences' keys are stored chains from ROOT, and
    /// each page's device holders are the sequences naming it (none for
    /// any other page).
    fn check(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        let mut holders: std::collections::HashMap<Key, u32> = Default::default();
        for (s, ks) in self.keys.iter().enumerate() {
            let mut parent = ROOT;
            for &k in ks {
                assert_eq!(
                    self.store.parent_of(k),
                    Some(parent),
                    "sequence {s}: its keys aren't a stored chain"
                );
                *holders.entry(k).or_default() += 1;
                parent = k;
            }
        }
        for (&k, &n) in &holders {
            assert_eq!(self.store.gpu(k), n, "{k:x}: device holders");
        }
        assert_eq!(self.store.dups(), holders.len(), "class-0 pages");
    }

    /// Sequence `s`, whose first `have` positions hold `head`'s, takes the
    /// keys of `head`'s stored prefix it holds already (no copy needed).
    fn adopt_held(&mut self, s: usize, ks: &[Key], have: usize) {
        let n = ks.len().min(have / PAGE);
        if n > self.keys[s].len() {
            self.set_keys(s, ks[..n].to_vec());
        }
    }

    /// Tests: the write-behind is starved (`Faults::starve`).
    #[cfg(test)]
    fn starved(&self) -> bool {
        (self.metrics.as_ref()).is_some_and(|m| m.tier_faults.starve.load(Relaxed))
    }

    /// Tests: a copy for `op` (of slot `slot`) is to be made; Err instead
    /// if a fault was armed for it (`Faults`).
    #[cfg(test)]
    fn fault(&self, op: Op, slot: usize) -> Result<(), String> {
        self.metrics
            .as_ref()
            .map_or(Ok(()), |m| m.tier_faults.fault(op, slot))
    }

    fn publish(&self, written: usize, restored: usize) {
        if let Some(m) = &self.metrics {
            m.kv_ram_bytes.store(self.store.used() as u64, Relaxed);
            m.kv_ram_pages.store(self.store.len() as u64, Relaxed);
            m.kv_ram_gpu_pages.store(self.store.dups() as u64, Relaxed);
            m.kv_ram_written.fetch_add(written as u64, Relaxed);
            m.kv_ram_restored.fetch_add(restored as u64, Relaxed);
            if restored > 0 {
                m.kv_ram_hits.fetch_add(1, Relaxed);
            }
        }
    }
}

/// How the arena's page-locked memory is taken (`AXON_KV_RAM_ALLOC`): by a
/// thread from server start on, the tier growing as slabs arrive (`Thread`,
/// the default), or all of it, at the size asked for, before the server
/// starts (`Start`). Never inside an engine step: a slab's `cuMemHostAlloc`
/// took 61.8 ms per 64 MiB on a 4090 pod (MEASURED, 1027-100925-fbcb), and
/// grown lazily in steps it cost +21.5 % ITL p99 there. Thread vs start
/// (4090, 1027-134559-a975): no inter-step gap over 0.3 ms either way, ITL
/// p99 +1.4 % vs +1.9 % over main at 6G, start-up at 20G 5.1 s vs 16.4 s
/// (main 2.0 s).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alloc {
    Start,
    Thread,
}

impl Alloc {
    /// AXON_KV_RAM_ALLOC=thread|start (default: thread).
    pub fn from_env() -> Alloc {
        match std::env::var("AXON_KV_RAM_ALLOC").as_deref() {
            Ok("start") => Alloc::Start,
            _ => Alloc::Thread,
        }
    }
}

/// What the allocator thread and the engine share: slabs pinned and not
/// adopted yet, (index, address); why it stopped early; whether it's done;
/// and whether the arena is gone (then the thread frees what it pins, and
/// the arena's drop frees what it left here: nothing leaks either way).
#[derive(Default)]
struct Pinning {
    ready: Vec<(usize, usize)>,
    err: Option<String>,
    done: bool,
    closed: bool,
}

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
    /// The allocator thread's slabs (`Alloc::Thread`, until all are in),
    /// and an allocator to free the ones nobody adopted at drop.
    pinning: Option<(Arc<Mutex<Pinning>>, HostAllocator)>,
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
            pinning: None,
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
                let back = net.host_allocator().expect("as above");
                let sizes: Vec<usize> = (0..n).map(|k| a.slab_pages(k) * page).collect();
                let sh = Arc::new(Mutex::new(Pinning::default()));
                let sh2 = sh.clone();
                std::thread::Builder::new()
                    .name("axon-kv-ram".into())
                    .spawn(move || {
                        for (k, &len) in sizes.iter().enumerate() {
                            let r = al.alloc(len);
                            let mut p = sh2.lock().unwrap_or_else(|e| e.into_inner());
                            match r {
                                // The arena is gone: give it back.
                                // SAFETY: ours, never adopted, no copy.
                                Ok(addr) if p.closed => {
                                    unsafe { al.free(addr) };
                                    return;
                                }
                                Ok(addr) => p.ready.push((k, addr)),
                                Err(e) => {
                                    p.err = Some(e);
                                    p.done = true;
                                    return;
                                }
                            }
                        }
                        sh2.lock().unwrap_or_else(|e| e.into_inner()).done = true;
                    })
                    .map_err(|e| format!("the KV RAM tier's allocator thread: {e}"))?;
                a.pinning = Some((sh, back));
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
    pub fn poll(&mut self, net: &Backend) {
        let Some((sh, _)) = &self.pinning else {
            return;
        };
        let (got, err, done) = {
            let mut p = sh.lock().unwrap_or_else(|e| e.into_inner());
            (std::mem::take(&mut p.ready), p.err.take(), p.done)
        };
        for (k, addr) in got {
            let len = self.slab_pages(k) * self.page;
            // SAFETY: a live `host_allocator` allocation of `len` bytes,
            // handed over once, owned by nobody else; page slots are
            // written before they are read.
            let b = unsafe { net.host_adopt(addr, len) };
            self.adopt(k, b);
        }
        if self.ready == self.slabs.len() && self.done_s.is_none() {
            let s = self.t0.elapsed().as_secs_f64();
            self.done_s = Some(s);
            self.pinning = None;
            if self.log {
                eprintln!(
                    "axon serve: KV RAM tier: {} pinned in {s:.1} s (a thread, from start)",
                    gib(self.bytes() as u64)
                );
            }
        } else if done {
            self.pinning = None;
            eprintln!(
                "axon serve: KV RAM tier: pinned {} of {} ({}); it stays that size",
                gib(self.bytes() as u64),
                gib((self.most * self.page) as u64),
                err.unwrap_or_else(|| "its thread ended".into())
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

impl Drop for Arena {
    /// The allocator thread stops (it frees what it pins from now on), and
    /// the slabs it pinned that were never adopted are freed here.
    fn drop(&mut self) {
        if let Some((sh, al)) = self.pinning.take() {
            let mut p = sh.lock().unwrap_or_else(|e| e.into_inner());
            p.closed = true;
            for (_, addr) in p.ready.drain(..) {
                // SAFETY: pinned by our thread, never adopted, no copy.
                unsafe { al.free(addr) };
            }
        }
    }
}

/// The tier of a slot pool (`Batch`): pages in pinned host memory.
pub(super) struct PoolTier {
    pub t: Tier<u32>,
    arena: Arena,
    /// Tests: what `write`'s copies are for (`Faults`): a spill, unless
    /// `write_some` (the write-behind) set it for the call.
    #[cfg(test)]
    op: Op,
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
        let p = PoolTier {
            t,
            arena,
            #[cfg(test)]
            op: Op::Spill,
        };
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

    /// Adopt the slabs the allocator thread has pinned (`Alloc::Thread`)
    /// and show them in /metrics: the engine calls it every iteration, so
    /// pinning shows (and finishes) on an idle server too. Whether it grew.
    pub fn poll(&mut self, net: &Backend) -> bool {
        if self.arena.pinning.is_none() {
            return false;
        }
        let had = self.arena.bytes();
        self.arena.poll(net);
        self.publish_pinned();
        self.arena.bytes() > had
    }

    fn publish_pinned(&self) {
        if let Some(m) = &self.t.metrics {
            m.kv_ram_pinned.store(self.arena.bytes() as u64, Relaxed);
        }
    }

    /// The spill of slot `s`, whose rows are about to go: store all its
    /// pages not yet stored (whole pages of `held`, the tokens whose rows
    /// it holds), in order, evicting to make room (class 0 first: other
    /// slots' duplicates); a page some other sequence stored already costs
    /// no copy. The pages copied (stops early when the store has no room).
    pub fn write(
        &mut self,
        net: &Backend,
        pool: &<Backend<'_> as BatchLm>::Pool,
        s: usize,
        held: &[u32],
    ) -> Result<usize, String> {
        let r = self.write_pages(net, pool, s, held, usize::MAX, true);
        if let (Ok(n), Some(m)) = (&r, &self.t.metrics) {
            m.kv_ram_spilled.fetch_add(*n as u64, Relaxed);
        }
        self.t.check();
        r
    }

    /// Store up to `most` of slot `s`'s pages not yet stored, in order,
    /// `evict`ing to make room (a spill) or into free room only (the
    /// write-behind: it stops at a full store, before copying). The pages
    /// copied; a page found stored (another sequence's) costs none.
    fn write_pages(
        &mut self,
        net: &Backend,
        pool: &<Backend<'_> as BatchLm>::Pool,
        s: usize,
        held: &[u32],
        most: usize,
        evict: bool,
    ) -> Result<usize, String> {
        // Tests: what its copies are for (a spill, but in `write_some`).
        #[cfg(test)]
        let op = std::mem::replace(&mut self.op, Op::Spill);
        self.t.revalidate(s);
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
            let room = match evict {
                true => self.t.store.room(parent, self.t.page),
                false => self.t.store.has_room(self.t.page),
            };
            let freed = self.t.drain();
            self.arena.free.extend(freed);
            if !room {
                break;
            }
            // Tests: no page slot free to the write-behind (`Faults::starve`).
            #[cfg(test)]
            if op == Op::WriteBehind && self.t.starved() {
                break;
            }
            let Some(a) = self.arena.take(net) else {
                break;
            };
            let (buf, at) = self.arena.at(a);
            // Tests: an injected fault fails the copy instead (`Faults`).
            #[cfg(test)]
            if let Err(e) = self.t.fault(op, s) {
                self.arena.free.push(a);
                return Err(e);
            }
            if let Err(e) = net.page_out(pool, s, i, buf, at) {
                self.arena.free.push(a);
                return Err(e);
            }
            let k = match evict {
                true => self.t.store.insert(parent, toks, a, self.t.page),
                false => self.t.store.insert_behind(parent, toks, a, self.t.page),
            };
            match k {
                Some(k) => self.t.push(s, k),
                // A hash collision with another prefix: keep what's there.
                None => {
                    self.arena.free.push(a);
                    break;
                }
            }
            copied += 1;
        }
        self.t.publish(copied, 0);
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
        let freed = self.t.drain();
        self.arena.free.extend(freed);
        if !room {
            return Ok(false);
        }
        let Some(a) = self.arena.take(net) else {
            return Ok(false);
        };
        let (buf, at) = self.arena.at(a);
        #[cfg(test)]
        if let Err(e) = self.t.fault(Op::Spill, usize::MAX) {
            self.arena.free.push(a);
            return Err(e);
        }
        if let Err(e) = net.page_out_id(pool, page, buf, at) {
            self.arena.free.push(a);
            return Err(e);
        }
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

    /// The write-behind (an engine step's, or a piece of the idle flush):
    /// up to `bytes` of copies of the slots' unstored pages, slots in
    /// `order` (the next to give their rows up first), into the store's
    /// free room only: a full store copies nothing. `held`: each slot's
    /// tokens. Its progress: the pages slots now name that they didn't
    /// (copied, or found stored), 0 if it could do nothing.
    pub fn write_some(
        &mut self,
        net: &Backend,
        pool: &<Backend<'_> as BatchLm>::Pool,
        held: &[Vec<u32>],
        order: &[usize],
        bytes: usize,
    ) -> Result<usize, String> {
        let mut left = bytes.div_ceil(self.t.page).max(1);
        let mut progress = 0;
        for &s in order {
            #[cfg(test)]
            {
                self.op = Op::WriteBehind;
            }
            let had = self.t.stored(s);
            let r = self.write_pages(net, pool, s, &held[s], left, false);
            progress += self.t.stored(s).saturating_sub(had);
            left -= r?;
            if left == 0 {
                break;
            }
        }
        self.t.check();
        Ok(progress)
    }

    /// Whether the write-behind has work: a slot holds a whole page not
    /// stored, and the store has free room for a page (#170's idle flush;
    /// a full store has none). `held`: each slot's tokens.
    pub fn backlog(&self, held: &[Vec<u32>]) -> bool {
        self.t.store.has_room(self.t.page)
            && held
                .iter()
                .enumerate()
                .any(|(s, h)| self.t.stored(s) < h.len() / PAGE)
    }

    /// Debug builds: slot `s`'s keys are the keys of its rows' first pages
    /// (`held`: its tokens), the invariant the bytes rest on (§2.2).
    pub fn check_slot(&self, s: usize, held: &[u32]) {
        if cfg!(debug_assertions) {
            let ks = &self.t.keys[s];
            let want = crate::kvstore::keys(&held[..(ks.len() * PAGE).min(held.len())]);
            assert!(
                *ks == want,
                "slot {s}: its stored pages aren't its rows' pages"
            );
        }
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
            // The whole page goes back, rows below `have` too (the same
            // bytes): #164's copy-on-write precondition, page `i` must be
            // this slot's alone (a shared partial page is copied first).
            #[cfg(test)]
            if let Err(e) = self.t.fault(Op::Restore, s) {
                res = Err(e);
                break;
            }
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
        self.t.check();
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
        #[cfg(test)]
        self.t.fault(Op::WriteBehind, 0).map_err(io::Error::other)?;
        net.kv_save(cache, from * PAGE, to * PAGE, &mut img)?;
        let mut copied = 0;
        for (i, b) in (from..to).zip(img.chunks_exact(self.t.page)) {
            let toks = &hist[i * PAGE..(i + 1) * PAGE];
            let parent = self.t.keys[0].last().copied().unwrap_or(ROOT);
            if let Some(k) = self.t.store.child(parent, toks) {
                self.t.push(0, k);
                continue;
            }
            let room = self.t.store.room(parent, self.t.page);
            drop(self.t.drain());
            if !room {
                break;
            }
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
        #[cfg(test)]
        self.t.fault(Op::Restore, 0).map_err(io::Error::other)?;
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

    thread_local! {
        /// Pages `store_seq` saw evicted on this thread (the fuzz's count).
        static EVICTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// `n` pages of tokens from `seed`.
    fn toks(seed: u32, n: usize) -> Vec<u32> {
        (0..(n * PAGE) as u32).map(|i| seed * 100_000 + i).collect()
    }

    /// Store `t`'s pages after what sequence `s` names (as a spill does:
    /// `room`, `drain`, `insert`, `push`); the pages it now names.
    fn store_seq(t: &mut Tier<u8>, s: usize, toks: &[u32]) -> usize {
        while t.stored(s) < toks.len() / PAGE {
            let i = t.stored(s);
            let b = &toks[i * PAGE..(i + 1) * PAGE];
            let parent = t.keys[s].last().copied().unwrap_or(ROOT);
            if let Some(k) = t.store.child(parent, b) {
                t.push(s, k);
                continue;
            }
            let room = t.store.room(parent, 1);
            let n = t.drain().len();
            EVICTED.with(|e| e.set(e.get() + n));
            if !room {
                break;
            }
            let k = t.store.insert(parent, b, 0, 1).expect("room made");
            t.push(s, k);
            t.check();
        }
        t.check();
        t.stored(s)
    }

    /// A sequence's pages are class 0 while it names them: evicted first
    /// (newest first, a leaf at a time), and it lets go of them (`drain`);
    /// truncated, they are class 1.
    #[test]
    fn keys_are_class0_and_drain_pops_them() {
        let mut t: Tier<u8> = Tier::new(5, 1, 2, None);
        let (a, b) = (toks(1, 3), toks(2, 2));
        assert_eq!(store_seq(&mut t, 0, &a), 3);
        assert_eq!(t.store.dups(), 3);
        // Sequence 1's rows go (a spill, then truncate): its pages are RAM's.
        assert_eq!(store_seq(&mut t, 1, &b), 2);
        t.truncate(1, 0);
        assert_eq!((t.store.dups(), t.stored(1)), (3, 0));
        t.check();
        // Full. A new page evicts sequence 0's leaf (class 0) before b's
        // older class-1 pages, and sequence 0 lets go of it.
        let c = toks(3, 1);
        assert_eq!(store_seq(&mut t, 1, &c), 1);
        assert_eq!(t.stored(0), 2, "the evicted leaf popped");
        assert_eq!(t.peek(&a), 2 * PAGE);
        assert_eq!(t.peek(&b), 2 * PAGE, "class 1 kept");
        // Re-storing sequence 0's page evicts sequence 1's (class 0,
        // newest) rather than b.
        assert_eq!(store_seq(&mut t, 0, &a), 3);
        assert_eq!(t.stored(1), 0);
        assert_eq!(t.peek(&b), 2 * PAGE);
        t.check();
        // Truncated: sequence 0's tail is class 1.
        t.truncate(0, 1);
        assert_eq!(t.store.dups(), 1);
        t.check();
    }

    /// Two sequences naming one page: it has two device holders, and is
    /// class 1 only once neither names it; a `make_room` that takes a
    /// chain's leaf and then its parent pops both.
    #[test]
    fn shared_pages_and_a_two_page_eviction() {
        let mut t: Tier<u8> = Tier::new(4, 1, 3, None);
        let shared = toks(1, 2);
        let mut a = shared.clone();
        a.extend(toks(2, 1));
        assert_eq!(store_seq(&mut t, 0, &a), 3);
        assert_eq!(store_seq(&mut t, 1, &shared), 2, "found stored: no copy");
        assert_eq!(t.store.used(), 3);
        let k0 = t.keys[1][0];
        assert_eq!(t.store.gpu(k0), 2);
        t.truncate(0, 0);
        assert_eq!(t.store.gpu(k0), 1);
        t.truncate(1, 0);
        assert_eq!((t.store.gpu(k0), t.store.dups()), (0, 0));
        t.check();
        // Sequence 2 names a's chain again, then 3 pages of room are made
        // in a 4-page store: one make_room takes a's leaf, then its parent
        // (a leaf by then); both popped, the first page stays.
        assert_eq!(store_seq(&mut t, 2, &a), 3);
        assert!(t.store.room(ROOT, 3));
        assert_eq!(t.drain().len(), 2);
        assert_eq!(t.stored(2), 1, "both evicted pages popped");
        t.check();
    }

    /// A restore (`lookup`, held, then `adopt_held`) makes the pages it
    /// moved class 0, the slot's again: a second spill of the slot copies
    /// nothing, and they are the first evicted when room is needed.
    #[test]
    fn a_restore_down_classes() {
        let mut t: Tier<u8> = Tier::new(3, 1, 2, None);
        let a = toks(1, 3);
        assert_eq!(store_seq(&mut t, 0, &a), 3);
        t.truncate(0, 0);
        assert_eq!(t.store.dups(), 0, "RAM's own");
        let ks = t.store.lookup(&a);
        for &k in &ks {
            t.store.hold(k);
        }
        t.adopt_held(1, &ks, ks.len() * PAGE);
        for &k in &ks {
            t.store.release(k);
        }
        assert_eq!(t.store.dups(), 3);
        t.check();
        let used = t.store.used();
        assert_eq!(store_seq(&mut t, 1, &a), 3);
        assert_eq!(t.store.used(), used, "the second spill copied nothing");
        // Full: a new page evicts the restored chain's leaf (class 0).
        let b = toks(2, 1);
        assert_eq!(store_seq(&mut t, 0, &b), 1);
        assert_eq!(t.stored(1), 2, "its leaf evicted and popped");
        t.check();
    }

    /// Fable's case: a slot restoring its own chain holds the pages it
    /// looked up; another slot's spill that needs room meanwhile can't
    /// evict them, class 0 though they are, and the restore failing
    /// (nothing adopted, the holds let go) leaves the books balanced.
    #[test]
    fn a_restores_holds_beat_class0_eviction() {
        let mut t: Tier<u8> = Tier::new(3, 1, 2, None);
        let a = toks(1, 3);
        assert_eq!(store_seq(&mut t, 0, &a), 3);
        t.truncate(0, 1);
        let ks = t.store.lookup(&a);
        assert_eq!(ks.len(), 3);
        for &k in &ks {
            t.store.hold(k);
        }
        // Sequence 1's spill: no page is evictable, so nothing is stored.
        assert_eq!(store_seq(&mut t, 1, &toks(2, 1)), 0);
        assert_eq!(t.stored(0), 1);
        t.check();
        // The restore failed: the holds go, the slot keeps what it had.
        for &k in &ks {
            t.store.release(k);
        }
        assert_eq!(t.stored(0), 1);
        t.check();
        // Then room is made from a's class-1 tail, not its class-0 head.
        assert_eq!(store_seq(&mut t, 1, &toks(2, 1)), 1);
        assert_eq!((t.stored(0), t.peek(&a)), (1, 2 * PAGE));
        t.check();
    }

    /// A seeded Tier fuzz (docs/design/kv-ram-tier-exclusive.md §3): 3
    /// sequences on 4 conversations sharing prefixes, at a capacity that
    /// evicts several leaves of one chain in one go. Random stores (spill
    /// style), truncates, adoptions of a lookup's prefix and write-behind
    /// style stores into free room; after each, `check` (stored chains,
    /// device holders), and no sequence names an unstored page.
    #[test]
    fn fuzz() {
        let mut evicted = 0;
        for seed in 1..=64u64 {
            match std::panic::catch_unwind(|| fuzz_one(seed)) {
                Ok(n) => evicted += n,
                Err(e) => {
                    let m = e
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    panic!("tier fuzz, seed {seed}: {m}");
                }
            }
        }
        eprintln!("tier fuzz: {evicted} pages evicted");
        assert!(evicted > 0);
    }

    fn fuzz_one(seed: u64) -> usize {
        let mut st = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut rnd = move |n: usize| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st % n as u64) as usize
        };
        let conv = |c: usize| -> Vec<u32> {
            let mut t = toks(if c < 2 { 1 } else { 10 + c as u32 }, 2);
            t.extend(toks(20 + c as u32, 4));
            t
        };
        let mut t: Tier<u8> = Tier::new(7, 1, 3, None);
        // What each sequence holds the rows of (tokens).
        let mut held: Vec<Vec<u32>> = vec![Vec::new(); 3];
        EVICTED.with(|e| e.set(0));
        for _ in 0..500 {
            let s = rnd(3);
            match rnd(6) {
                0 => {
                    // New rows: a conversation's first n pages (its keys
                    // cut to what still matches, as admit's spill does).
                    let n = 1 + rnd(6);
                    let c = conv(rnd(4))[..n * PAGE].to_vec();
                    let common = held[s].iter().zip(&c).take_while(|(a, b)| a == b).count();
                    t.truncate(s, common / PAGE);
                    held[s] = c;
                }
                1 | 2 => {
                    store_seq(&mut t, s, &held[s]);
                }
                3 => {
                    // The write-behind: free room only.
                    while t.stored(s) < held[s].len() / PAGE {
                        let i = t.stored(s);
                        let b = &held[s][i * PAGE..(i + 1) * PAGE];
                        let parent = t.keys[s].last().copied().unwrap_or(ROOT);
                        if let Some(k) = t.store.child(parent, b) {
                            t.push(s, k);
                            continue;
                        }
                        let Some(k) = t.store.insert_behind(parent, b, 0, 1) else {
                            break;
                        };
                        t.push(s, k);
                    }
                    assert!(
                        t.store.take_evicted().is_empty(),
                        "the write-behind evicted"
                    );
                }
                4 => {
                    // A restore: the stored prefix of the rows, held while
                    // looked up, adopted.
                    let ks = t.store.lookup(&held[s]);
                    for &k in &ks {
                        t.store.hold(k);
                    }
                    t.adopt_held(s, &ks, ks.len() * PAGE);
                    for &k in &ks {
                        t.store.release(k);
                    }
                }
                _ => {
                    // The rows go (a reclaim: spill, then forget them).
                    store_seq(&mut t, s, &held[s]);
                    t.truncate(s, 0);
                    held[s].clear();
                }
            }
            assert!(t.drain().is_empty(), "an eviction not drained");
            t.check();
            for (s, h) in held.iter().enumerate() {
                let ks = &t.keys[s];
                assert_eq!(
                    *ks,
                    crate::kvstore::keys(&h[..ks.len() * PAGE]),
                    "sequence {s}: its keys aren't its rows' pages"
                );
            }
        }
        EVICTED.with(|e| e.get())
    }
}
