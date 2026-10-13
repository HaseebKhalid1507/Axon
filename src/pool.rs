//! Persistent worker pool for the forward pass.
//!
//! A token runs ~100 parallel matvecs with a few microseconds of serial work
//! between them. Rayon puts its threads to sleep in those gaps, and waking them
//! cost ~30-60 us per matvec on this machine, about a quarter of decode time.
//! Here the workers stay up and spin between jobs, so starting one costs ~1 us.
//!
//! - Chunks are claimed from a shared counter, so fast and efficiency cores
//!   balance themselves.
//! - Each worker reports completion in its own cache line; the caller polls
//!   them. No shared "done" counter bouncing between cores.
//! - A worker spins for SPIN_FOR after its last job, then parks, so an idle
//!   axon doesn't burn CPU. Past YIELD_AFTER it yields while it spins, so a
//!   late job's preempted thread can get its core back.
//! - Adaptive participation: a job uses only the first `active` workers. When
//!   the caller keeps waiting far longer than its own work took (a worker got
//!   preempted holding a chunk, i.e. the machine is busy), `active` steps down;
//!   after a long quiet streak it probes back up. Workers left out park.
//!   AXON_POOL_ADAPT=0 turns this off.
//! - Fewer threads than cores by default. AXON_THREADS sets the spawned count,
//!   an upper bound on `active`.

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Mutex, OnceLock};
use std::thread::Thread;
use std::time::{Duration, Instant};

const SPIN_FOR: Duration = Duration::from_micros(200);
/// Waiting longer than this means the job is late (or there's no next job
/// yet): keep spinning, but yield now and then.
const YIELD_AFTER: Duration = Duration::from_micros(30);

/// The epoch word is `seq << ACTIVE_BITS | participating workers`, so a worker
/// learns whether it's in a job from the same atomic load that publishes it.
const ACTIVE_BITS: u32 = 16;
const ACTIVE_MASK: u64 = (1 << ACTIVE_BITS) - 1;

thread_local! {
    /// True while this thread is running a chunk (worker threads always are).
    /// A nested `run` from inside a chunk would deadlock, so it runs serially.
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[repr(align(128))]
struct Padded<T>(T);

/// The job being run: a borrowed closure called once per chunk index.
/// Only valid while `run` is executing; `run` doesn't return until every
/// participating worker has finished with it.
#[derive(Clone, Copy)]
struct JobRef {
    f: *const (dyn Fn(usize) + Sync),
    n_chunks: usize,
}

struct Shared {
    /// Epoch word (see ACTIVE_BITS), bumped by the caller to publish a new job
    /// (SeqCst store); workers Acquire it.
    epoch: Padded<AtomicU64>,
    /// Next chunk to claim.
    next: Padded<AtomicUsize>,
    /// Written before `epoch` is bumped, read after it's observed, and only by
    /// workers that epoch says participate.
    job: UnsafeCell<JobRef>,
    /// done[w] = last epoch word worker w finished (only set when it took part).
    done: Vec<Padded<AtomicU64>>,
    /// sleeping[w]: worker w is parked or about to park; the caller must unpark it.
    sleeping: Vec<Padded<AtomicBool>>,
    /// Set if a chunk panicked on a worker thread.
    panicked: AtomicBool,
}

// The UnsafeCell is only written by the caller (holding `Pool::lock`) between
// jobs, while every worker that could read it is idle; workers read it only
// after Acquiring an epoch that published it and names them as participants.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

fn participants(word: u64) -> usize {
    (word & ACTIVE_MASK) as usize
}

pub struct Pool {
    shared: &'static Shared,
    workers: Vec<Thread>,
    /// One job at a time (also makes `run` safe to call from several threads,
    /// e.g. the test harness), and the participation policy state.
    lock: Mutex<Adapt>,
    adapt: bool,
    /// Mirror of the policy's active count (caller included), for `active()`.
    /// Single writer (the caller, under `lock`), read rarely: deliberately not
    /// padded onto its own cache line.
    active: AtomicUsize,
    /// Lowest `active` seen since start, for `min_active()`. Same single writer.
    min_active: AtomicUsize,
}

impl Pool {
    /// The process-wide pool, started on first use.
    pub fn global() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(|| Pool::new(default_threads()))
    }

    /// Total threads that can run chunks, including the caller.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Threads currently taking part in jobs, including the caller.
    pub fn active(&self) -> usize {
        self.active.load(Relaxed)
    }

    /// Lowest `active()` since the pool started (how far adaptation backed off).
    pub fn min_active(&self) -> usize {
        self.min_active.load(Relaxed)
    }

    /// Adaptation counters since start, for AXON_POOL_DEBUG. They live in the
    /// policy state `run` already holds the lock for: no extra hot-path atomics.
    /// Takes that lock, so don't call it from inside a chunk of `run`.
    pub fn stats(&self) -> PoolStats {
        self.lock.lock().unwrap_or_else(|p| p.into_inner()).stats
    }

    fn new(threads: usize) -> Pool {
        Pool::with(threads, adapt_enabled())
    }

    fn with(threads: usize, adapt: bool) -> Pool {
        // The epoch word has ACTIVE_BITS for the participant count.
        let n = (threads.max(1) - 1).min(ACTIVE_MASK as usize);
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            epoch: Padded(AtomicU64::new(0)),
            next: Padded(AtomicUsize::new(0)),
            job: UnsafeCell::new(JobRef {
                f: &|_| {},
                n_chunks: 0,
            }),
            done: (0..n).map(|_| Padded(AtomicU64::new(0))).collect(),
            sleeping: (0..n).map(|_| Padded(AtomicBool::new(false))).collect(),
            panicked: AtomicBool::new(false),
        }));
        let workers = (0..n)
            .map(|w| {
                std::thread::Builder::new()
                    .name(format!("axon-{w}"))
                    .spawn(move || worker(shared, w))
                    .expect("spawn pool worker")
                    .thread()
                    .clone()
            })
            .collect();
        Pool {
            shared,
            workers,
            lock: Mutex::new(Adapt::new(n)),
            adapt,
            active: AtomicUsize::new(n + 1),
            min_active: AtomicUsize::new(n + 1),
        }
    }

    /// Call `f(c)` for every c in 0..n_chunks, spread over the pool, and return
    /// when all calls are done. A `run` from inside a chunk runs serially.
    pub fn run(&self, n_chunks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_chunks == 0 {
            return;
        }
        if IN_POOL.get() {
            (0..n_chunks).for_each(f);
            return;
        }
        let mut adapt = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let s = self.shared;
        let active = adapt.active;
        if active == 0 {
            // No workers (AXON_THREADS=1). Release the lock first: a nested
            // `run` from `f` would otherwise deadlock on it.
            drop(adapt);
            (0..n_chunks).for_each(f);
            return;
        }
        // Publish the job. Every worker that took part in the previous job has
        // marked it done (we waited), and the rest never read `job` or `next`
        // for an epoch that left them out, so nobody is reading them now.
        // SAFETY: the transmute only erases the closure's lifetime. Only workers
        // w < active dereference the pointer, and only until they store
        // done[w] = word, which we wait for below before returning, so it never
        // outlives `f`.
        let f_static: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        unsafe {
            *s.job.get() = JobRef {
                f: f_static,
                n_chunks,
            }
        };
        s.next.0.store(0, Relaxed);
        let seq = (s.epoch.0.load(Relaxed) >> ACTIVE_BITS) + 1;
        let word = (seq << ACTIVE_BITS) | active as u64;
        s.epoch.0.store(word, SeqCst);
        // Wake parked participants only; left-out workers stay parked.
        let mut woke = false;
        for (w, t) in self.workers[..active].iter().enumerate() {
            if s.sleeping[w].0.load(SeqCst) {
                t.unpark();
                woke = true;
            }
        }
        // The caller works too.
        let t0 = Instant::now();
        IN_POOL.set(true);
        let mine = catch_unwind(AssertUnwindSafe(|| claim_chunks(s, f, n_chunks)));
        IN_POOL.set(false);
        let t1 = Instant::now();
        for d in &s.done[..active] {
            let mut spins = 0u32;
            while d.0.load(Acquire) != word {
                // A worker is late, probably preempted holding a chunk: give
                // our core away now and then instead of spinning a timeslice.
                spins = spins.wrapping_add(1);
                if spins.is_multiple_of(64) && t1.elapsed() > YIELD_AFTER {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
            }
        }
        // Clear the worker-panic flag before re-raising our own panic, or a
        // stale flag would fail the next, unrelated run.
        let worker_panicked = s.panicked.swap(false, Relaxed);
        // Only judge clean runs where the caller did at least one chunk: a
        // wake-up from park or a panic unwinding makes a long tail that says
        // nothing about preemption, and with no chunk of our own there's no
        // yardstick.
        if self.adapt
            && !woke
            && !worker_panicked
            && let Ok(&k) = mine.as_ref()
            && k > 0
        {
            let (own, tail) = (t1 - t0, t1.elapsed());
            let stall = is_stall(own, tail);
            adapt.stats.judged += 1;
            if stall {
                adapt.stats.stalls += 1;
                adapt.stats.stall_us += (tail - own).as_micros() as u64;
            }
            let a = adapt.record(stall) + 1;
            self.active.store(a, Relaxed);
            self.min_active.fetch_min(a, Relaxed);
        }
        drop(adapt);
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        if worker_panicked {
            panic!("a pool worker panicked while running a chunk");
        }
    }

    /// Pin participation to `workers` (test use; pair with adaptation off).
    #[cfg(test)]
    fn force_active(&self, workers: usize) {
        let mut a = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        a.active = workers.min(a.max);
        self.active.store(a.active + 1, Relaxed);
        self.min_active.fetch_min(a.active + 1, Relaxed);
    }
}

/// Claim and run chunks until none are left; returns how many we ran.
fn claim_chunks(s: &Shared, f: &(dyn Fn(usize) + Sync), n: usize) -> usize {
    let mut k = 0;
    loop {
        let c = s.next.0.fetch_add(1, Relaxed);
        if c >= n {
            return k;
        }
        f(c);
        k += 1;
    }
}

fn worker(s: &'static Shared, w: usize) {
    IN_POOL.set(true);
    let mut seen = 0;
    let mut spin_for = SPIN_FOR;
    loop {
        // Wait for a new epoch: spin for a while (yielding once it's been a
        // while), then park.
        let start = Instant::now();
        let mut spins = 0u32;
        let word = loop {
            let e = s.epoch.0.load(Acquire);
            if e != seen {
                break e;
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(64) {
                let waited = start.elapsed();
                if waited > spin_for {
                    s.sleeping[w].0.store(true, SeqCst);
                    // Re-check after announcing we're going to sleep: either we
                    // see the new epoch here, or the caller sees `sleeping` and
                    // unparks us (an unpark before park makes park return at
                    // once). A caller that leaves us out doesn't unpark us; we
                    // see its epoch whenever we next wake.
                    let e = s.epoch.0.load(SeqCst);
                    if e == seen {
                        std::thread::park();
                    }
                    s.sleeping[w].0.store(false, SeqCst);
                    continue;
                }
                if waited > YIELD_AFTER {
                    std::thread::yield_now();
                    continue;
                }
            }
            std::hint::spin_loop();
        };
        seen = word;
        if w >= participants(word) {
            // Left out of this job: don't touch `job`, `next`, `done` or
            // `panicked`. The caller doesn't wait for us and may reuse `job`
            // at any moment. Park until a caller that wants us unparks us.
            spin_for = Duration::ZERO;
            continue;
        }
        spin_for = SPIN_FOR;
        // SAFETY: `word` names us as a participant, so the caller that
        // published it (job written before its SeqCst epoch store, which we
        // Acquired) waits for done[w] == word before returning or writing
        // `job` again. `f` is live until we store done below.
        let job = unsafe { *s.job.get() };
        let f = unsafe { &*job.f };
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| claim_chunks(s, f, job.n_chunks))) {
            // Leak the payload: if its Drop panicked, this thread would die and
            // the caller would wait on our done slot forever.
            std::mem::forget(payload);
            s.panicked.store(true, Relaxed);
        }
        s.done[w].0.store(word, Release);
    }
}

/// A tail (caller waiting after its own chunks) this much longer than the
/// caller's own work means someone sat on a chunk: a preempted worker, not
/// imbalance. The caller did ≥1 chunk, so 4x is ≥4 chunks' worth, past what
/// an E-core vs P-core speed gap explains.
const STALL_RATIO: u32 = 4;
/// ...and at least this long: above park/unpark and cache-miss jitter, far
/// below a scheduler slice (~1-3 ms), which is what a real stall costs.
const STALL_MIN: Duration = Duration::from_micros(50);
/// Stalls within FORGET_AFTER clean runs of each other needed to step down.
/// One isolated stall (a kernel thread, an interrupt storm) shouldn't cost a
/// thread.
const STALLS_TO_DROP: u32 = 2;
/// Clean runs after which pending stalls are forgotten (~2 tokens of decode).
const FORGET_AFTER: u32 = 256;
/// Clean runs before probing one step up (~17 tokens of decode).
const PROBE_BASE: u32 = 2048;
/// Probe backoff cap (~270 tokens, a few seconds), so a steadily busy machine
/// pays a failed probe (one stall, a few ms) at most every couple of seconds,
/// and one whose load went away starts climbing back within a few seconds.
const PROBE_MAX: u32 = 16 * PROBE_BASE;
/// A stall within this many runs of a step up means the probe failed: step
/// straight back down and wait twice as long before the next probe.
const PROBE_GRACE: u32 = 256;

fn is_stall(own: Duration, tail: Duration) -> bool {
    tail > STALL_MIN && tail > own * STALL_RATIO
}

/// Adaptation counters since start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Runs the policy looked at (no wake-up, no panic, caller ran a chunk).
    pub judged: u64,
    /// Judged runs whose tail was a stall.
    pub stalls: u64,
    /// Time those stalls cost: tail beyond the caller's own work, in us.
    pub stall_us: u64,
    pub step_downs: u64,
    pub probes: u64,
    pub failed_probes: u64,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} judged, {} stalls ({:.1} ms lost), {} down, {} probes ({} failed)",
            self.judged,
            self.stalls,
            self.stall_us as f64 / 1e3,
            self.step_downs,
            self.probes,
            self.failed_probes
        )
    }
}

/// Participation policy: a pure state machine fed one stall/no-stall verdict
/// per run. Counts are workers (caller excluded).
#[derive(Clone, Debug)]
struct Adapt {
    max: usize,
    min: usize,
    active: usize,
    /// Stalls not yet acted on.
    stalls: u32,
    /// Runs since the last stall or step up.
    clean: u32,
    probe_after: u32,
    /// Last change was a step up still inside PROBE_GRACE.
    probing: bool,
    stats: PoolStats,
}

impl Adapt {
    fn new(max: usize) -> Adapt {
        Adapt {
            max,
            min: max.min(1),
            active: max,
            stalls: 0,
            clean: 0,
            probe_after: PROBE_BASE,
            probing: false,
            stats: PoolStats::default(),
        }
    }

    /// Feed one run's verdict; returns the new active worker count.
    fn record(&mut self, stall: bool) -> usize {
        if stall {
            self.clean = 0;
            self.stalls += 1;
            if self.probing {
                self.probing = false;
                self.probe_after = (self.probe_after * 2).min(PROBE_MAX);
                self.stats.failed_probes += 1;
                self.step_down();
            } else if self.stalls >= STALLS_TO_DROP {
                self.step_down();
            }
        } else {
            self.clean = self.clean.saturating_add(1);
            if self.clean == FORGET_AFTER {
                self.stalls = 0;
            }
            if self.probing && self.clean >= PROBE_GRACE {
                // Probe held (and so we're climbing, or back at full): the
                // backoff only applies to a level that keeps failing.
                self.probing = false;
                self.probe_after = PROBE_BASE;
            }
            if self.clean >= self.probe_after && self.active < self.max {
                self.active += 1;
                self.clean = 0;
                self.stalls = 0;
                self.probing = true;
                self.stats.probes += 1;
            }
        }
        self.active
    }

    fn step_down(&mut self) {
        if self.active > self.min {
            self.active -= 1;
            self.stats.step_downs += 1;
        }
        self.stalls = 0;
    }
}

/// AXON_POOL_ADAPT=0 pins participation to every spawned thread.
fn adapt_enabled() -> bool {
    !matches!(
        std::env::var("AXON_POOL_ADAPT").as_deref().map(str::trim),
        Ok("0")
    )
}

/// AXON_THREADS if set, else the core count minus a quarter (min 1), leaving
/// room for the OS and other programs so spinning workers aren't preempted.
fn default_threads() -> usize {
    if let Some(n) = std::env::var("AXON_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return n.max(1);
    }
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    (cores - cores / 4).max(1)
}

/// Split `data` into chunks of `chunk` items and call `f(chunk_index, chunk)`
/// for each, in parallel on the global pool.
pub fn for_each_chunk_mut<T: Send>(
    data: &mut [T],
    chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    assert!(chunk > 0);
    let len = data.len();
    let base = SendPtr(data.as_mut_ptr());
    Pool::global().run(len.div_ceil(chunk), &|c| {
        let start = c * chunk;
        let n = chunk.min(len - start);
        // SAFETY: chunk c covers [start, start + n), disjoint from every other
        // chunk, and inside `data`, which is borrowed mutably for the whole run.
        let ys = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), n) };
        f(c, ys);
    });
}

/// `for_each_chunk_mut`, but chunks are handed out last first. When later
/// chunks take longer (causal attention: later tokens see more positions),
/// starting the longest ones first keeps a long chunk from starting at the
/// end while every other thread waits.
pub fn for_each_chunk_mut_rev<T: Send>(
    data: &mut [T],
    chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    assert!(chunk > 0);
    let len = data.len();
    let n_chunks = len.div_ceil(chunk);
    let base = SendPtr(data.as_mut_ptr());
    Pool::global().run(n_chunks, &|i| {
        let c = n_chunks - 1 - i;
        let start = c * chunk;
        let n = chunk.min(len - start);
        // SAFETY: as in for_each_chunk_mut; c runs over 0..n_chunks once each.
        let ys = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), n) };
        f(c, ys);
    });
}

struct SendPtr<T>(*mut T);
impl<T> SendPtr<T> {
    // A method (not field access) so closures capture the whole Sync wrapper,
    // not the raw pointer inside it.
    fn get(&self) -> *mut T {
        self.0
    }
}
unsafe impl<T: Send> Sync for SendPtr<T> {}
unsafe impl<T: Send> Send for SendPtr<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_chunk_runs_exactly_once() {
        let pool = Pool::new(6);
        for n in [0, 1, 5, 6, 7, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, &|c| {
                hits[c].fetch_add(1, Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "n={n}");
        }
    }

    #[test]
    fn writes_are_visible_after_run() {
        let mut v = vec![0u64; 10_007];
        for round in 1..=50u64 {
            for_each_chunk_mut(&mut v, 37, |c, ys| {
                for (j, y) in ys.iter_mut().enumerate() {
                    *y = round * 1_000_000 + (c * 37 + j) as u64;
                }
            });
            assert!(
                v.iter()
                    .enumerate()
                    .all(|(i, &y)| y == round * 1_000_000 + i as u64)
            );
        }
    }

    #[test]
    fn survives_parking() {
        // Long enough between jobs that every worker parks, then must be woken.
        let pool = Pool::new(4);
        for _ in 0..5 {
            let count = AtomicUsize::new(0);
            pool.run(64, &|_| {
                count.fetch_add(1, Relaxed);
            });
            assert_eq!(count.load(Relaxed), 64);
            std::thread::sleep(SPIN_FOR * 3);
        }
    }

    #[test]
    fn concurrent_callers_serialize() {
        let pool = Pool::new(4);
        std::thread::scope(|sc| {
            for t in 0..4 {
                let pool = &pool;
                sc.spawn(move || {
                    for _ in 0..200 {
                        let sum = AtomicUsize::new(0);
                        pool.run(17, &|c| {
                            sum.fetch_add(c + t, Relaxed);
                        });
                        assert_eq!(sum.load(Relaxed), (0..17).sum::<usize>() + 17 * t);
                    }
                });
            }
        });
    }

    #[test]
    fn nested_run_is_serial_not_deadlock() {
        let count = AtomicUsize::new(0);
        Pool::global().run(8, &|_| {
            Pool::global().run(5, &|_| {
                count.fetch_add(1, Relaxed);
            });
        });
        assert_eq!(count.load(Relaxed), 40);
    }

    #[test]
    fn nested_run_without_workers() {
        // One thread, no workers: the serial path must not hold the lock
        // while it runs `f` (this hung under AXON_THREADS=1).
        let pool = Pool::new(1);
        let count = AtomicUsize::new(0);
        pool.run(8, &|_| {
            pool.run(5, &|_| {
                count.fetch_add(1, Relaxed);
            });
        });
        assert_eq!(count.load(Relaxed), 40);
    }

    #[test]
    fn double_panic_does_not_poison_next_run() {
        let pool = Pool::new(4);
        for _ in 0..20 {
            // Every chunk panics, so the caller and workers all do.
            let r = catch_unwind(AssertUnwindSafe(|| pool.run(64, &|_| panic!("all"))));
            assert!(r.is_err());
            let count = AtomicUsize::new(0);
            pool.run(10, &|_| {
                count.fetch_add(1, Relaxed);
            });
            assert_eq!(count.load(Relaxed), 10);
        }
    }

    #[test]
    fn worker_panic_reaches_the_caller() {
        let pool = Pool::new(4);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(1000, &|c| {
                if c == 777 {
                    panic!("boom");
                }
            })
        }));
        assert!(r.is_err());
        // The pool still works afterwards.
        let count = AtomicUsize::new(0);
        pool.run(100, &|_| {
            count.fetch_add(1, Relaxed);
        });
        assert_eq!(count.load(Relaxed), 100);
    }

    /// Index of the pool worker running this, if it is one.
    fn worker_index() -> Option<usize> {
        std::thread::current()
            .name()
            .and_then(|n| n.strip_prefix("axon-"))
            .and_then(|i| i.parse().ok())
    }

    #[test]
    fn partial_participation_leaves_others_alone() {
        let pool = Pool::with(6, false);
        pool.force_active(2);
        assert_eq!(pool.active(), 3);
        let s = pool.shared;
        let before: Vec<u64> = s.done.iter().map(|d| d.0.load(Acquire)).collect();
        for n in [1, 7, 100, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let outsider = AtomicBool::new(false);
            pool.run(n, &|c| {
                if worker_index().is_some_and(|w| w >= 2) {
                    outsider.store(true, Relaxed);
                }
                hits[c].fetch_add(1, Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "n={n}");
            assert!(!outsider.load(Relaxed), "a left-out worker ran a chunk");
        }
        for (w, b) in before.iter().enumerate().skip(2) {
            assert_eq!(s.done[w].0.load(Acquire), *b, "worker {w} done slot");
        }
        // Left-out workers park rather than spin.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !(2..5).all(|w| s.sleeping[w].0.load(SeqCst)) {
            assert!(Instant::now() < deadline, "left-out workers never parked");
            std::thread::sleep(Duration::from_millis(1));
        }
        // Bringing them back wakes them and everything still adds up.
        pool.force_active(5);
        for _ in 0..50 {
            let sum = AtomicUsize::new(0);
            pool.run(97, &|c| {
                sum.fetch_add(c, Relaxed);
            });
            assert_eq!(sum.load(Relaxed), (0..97).sum::<usize>());
        }
        let word = s.epoch.0.load(Acquire);
        assert!(s.done.iter().all(|d| d.0.load(Acquire) == word));
    }

    #[test]
    fn adapt_quiet_machine_stays_full() {
        let mut a = Adapt::new(17);
        for _ in 0..100_000 {
            assert_eq!(a.record(false), 17);
        }
        // Isolated stalls, spaced out, never cost a thread.
        for _ in 0..50 {
            assert_eq!(a.record(true), 17);
            for _ in 0..FORGET_AFTER {
                assert_eq!(a.record(false), 17);
            }
        }
    }

    #[test]
    fn adapt_steps_down_to_floor_and_probes_back_up() {
        let mut a = Adapt::new(5);
        assert_eq!(a.record(true), 5);
        assert_eq!(a.record(false), 5);
        assert_eq!(a.record(true), 4);
        for _ in 0..20 {
            a.record(true);
        }
        assert_eq!(a.active, 1, "floor is caller + 1 worker");
        // A quiet streak probes one step up.
        for _ in 0..PROBE_BASE - 1 {
            assert_eq!(a.record(false), 1);
        }
        assert_eq!(a.record(false), 2);
        // Probe fails: a single stall steps back down and doubles the wait.
        assert_eq!(a.record(true), 1);
        for _ in 0..2 * PROBE_BASE - 1 {
            assert_eq!(a.record(false), 1);
        }
        assert_eq!(a.record(false), 2);
        // Probe holds past the grace period: backoff resets, and a quiet
        // machine climbs all the way back.
        let mut runs = 0;
        while a.active < 5 {
            a.record(false);
            runs += 1;
            assert!(runs < 4 * PROBE_BASE as usize, "slow to recover");
        }
        assert_eq!(a.probe_after, PROBE_BASE);
        for _ in 0..10_000 {
            assert_eq!(a.record(false), 5);
        }
    }

    #[test]
    fn adapt_backoff_is_capped() {
        let mut a = Adapt::new(3);
        a.record(true);
        a.record(true);
        a.record(true);
        a.record(true);
        assert_eq!(a.active, 1);
        for _ in 0..20 {
            while !a.probing {
                a.record(false);
            }
            a.record(true);
        }
        assert_eq!(a.probe_after, PROBE_MAX);
        assert_eq!(a.active, 1);
        assert_eq!(a.stats.probes, 20);
        assert_eq!(a.stats.failed_probes, 20);
        assert_eq!(a.stats.step_downs, 2 + 20);
    }

    #[test]
    fn adapt_backoff_resets_once_load_goes_away() {
        let mut a = Adapt::new(3);
        for _ in 0..4 {
            a.record(true);
        }
        for _ in 0..20 {
            while !a.probing {
                a.record(false);
            }
            a.record(true);
        }
        assert_eq!((a.active, a.probe_after), (1, PROBE_MAX));
        // Quiet from here: one capped wait, then a probe per PROBE_BASE.
        let mut runs = 0;
        while a.active < 3 {
            assert_eq!(a.record(false), a.active);
            runs += 1;
        }
        assert_eq!(runs, PROBE_MAX + PROBE_BASE);
        for _ in 0..PROBE_GRACE {
            a.record(false);
        }
        assert_eq!(a.probe_after, PROBE_BASE, "back at full: base backoff");
        assert!(!a.probing);
        // A later busy spell starts from the base wait again.
        a.record(true);
        a.record(true);
        assert_eq!(a.active, 2);
        for _ in 0..PROBE_BASE - 1 {
            a.record(false);
        }
        assert_eq!(a.record(false), 3);
    }

    #[test]
    fn adapt_with_no_workers_is_inert() {
        let mut a = Adapt::new(0);
        for i in 0..10_000 {
            assert_eq!(a.record(i % 3 == 0), 0);
        }
    }

    #[test]
    fn stall_verdict() {
        let us = Duration::from_micros;
        assert!(!is_stall(us(20), us(5)));
        assert!(!is_stall(us(1), us(40)), "short tails are noise");
        assert!(
            !is_stall(us(500), us(1500)),
            "long chunks, proportional tail"
        );
        assert!(is_stall(us(20), us(2000)));
    }

    #[test]
    fn panics_with_adaptation_on() {
        let pool = Pool::with(4, true);
        for i in 0..40 {
            if i % 2 == 0 {
                // Workers sit on their chunk for 2 ms while the caller drains
                // the rest in microseconds: a stall every time.
                let count = AtomicUsize::new(0);
                pool.run(64, &|_| {
                    if worker_index().is_some() {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    count.fetch_add(1, Relaxed);
                });
                assert_eq!(count.load(Relaxed), 64);
            } else {
                let r = catch_unwind(AssertUnwindSafe(|| {
                    pool.run(64, &|c| {
                        if c % 7 == 3 {
                            panic!("adapt boom");
                        }
                    })
                }));
                assert!(r.is_err());
            }
        }
        assert!(pool.active() < pool.threads(), "stalls never stepped down");
        assert!(pool.active() >= 2);
        assert_eq!(pool.min_active(), pool.active());
        let st = pool.stats();
        assert!(st.judged >= st.stalls && st.stalls >= 2, "{st}");
        // Loss is the tail beyond the caller's own work; on a starved CPU (tests
        // in parallel on few cores) the caller's work stretches too, so only
        // check that some loss was recorded.
        assert!(st.stall_us > 0, "2 ms stalls: {st}");
        assert_eq!(st.step_downs as usize, pool.threads() - pool.active());
        let count = AtomicUsize::new(0);
        pool.run(100, &|_| {
            count.fetch_add(1, Relaxed);
        });
        assert_eq!(count.load(Relaxed), 100);
    }
}
