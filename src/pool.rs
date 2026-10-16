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
//! - Two kinds of job, each with its own participation level (`Kind`):
//!   prefill's compute-bound matmuls, and one token's memory-bound decode
//!   matvecs, which stop gaining from threads long before they run out of
//!   cores and lose to the slowest core's tail. Decode uses at most
//!   `decode_start` threads (stepping down from there on stalls, probing back
//!   up to it), and a step down in one kind never throttles the other.
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

/// What a job is, for the participation policy (each kind has its own).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Prefill and everything else: compute-bound, wants every thread.
    Wide = 0,
    /// One token's matvecs and attention: memory-bound, settles lower.
    Decode = 1,
}

pub struct Pool {
    shared: &'static Shared,
    workers: Vec<Thread>,
    /// One job at a time (also makes `run` safe to call from several threads,
    /// e.g. the test harness), and the participation policy state per `Kind`.
    lock: Mutex<[Adapt; 2]>,
    adapt: bool,
    /// Mirror of each kind's active count (caller included), for `active()`.
    /// Single writer (the caller, under `lock`), read rarely: deliberately not
    /// padded onto its own cache line.
    active: [AtomicUsize; 2],
    /// Lowest `active` seen since start, for `min_active()`. Same single writer.
    min_active: [AtomicUsize; 2],
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

    /// Threads currently taking part in wide jobs, including the caller.
    pub fn active(&self) -> usize {
        self.active_in(Kind::Wide)
    }

    /// Threads currently taking part in `kind` jobs, including the caller.
    pub fn active_in(&self, kind: Kind) -> usize {
        self.active[kind as usize].load(Relaxed)
    }

    /// Lowest `active()` since the pool started (how far adaptation backed off).
    pub fn min_active(&self) -> usize {
        self.min_active_in(Kind::Wide)
    }

    /// Lowest `active_in(kind)` since the pool started.
    pub fn min_active_in(&self, kind: Kind) -> usize {
        self.min_active[kind as usize].load(Relaxed)
    }

    /// Adaptation counters since start (both kinds), for AXON_POOL_DEBUG.
    /// They live in the policy state `run` already holds the lock for: no
    /// extra hot-path atomics. Takes that lock, so don't call it from inside
    /// a chunk of `run`.
    pub fn stats(&self) -> PoolStats {
        let a = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let (w, d) = (a[0].stats, a[1].stats);
        PoolStats {
            judged: w.judged + d.judged,
            stalls: w.stalls + d.stalls,
            stall_us: w.stall_us + d.stall_us,
            step_downs: w.step_downs + d.step_downs,
            probes: w.probes + d.probes,
            failed_probes: w.failed_probes + d.failed_probes,
            decode_active: a[1].active + 1,
            decode_min: self.min_active[1].load(Relaxed),
        }
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
        // Decode is capped at its own level (a probe up never passes it:
        // over a long generation probing back to every thread lost the
        // gain). With adaptation off every kind uses every thread, except
        // that an explicit AXON_DECODE_THREADS still caps decode.
        let d = if adapt {
            decode_start(n + 1) - 1
        } else {
            decode_threads_env().map_or(n, |t| t.clamp(1, n + 1) - 1)
        };
        Pool {
            shared,
            workers,
            lock: Mutex::new([Adapt::new(n), Adapt::new(d)]),
            adapt,
            active: [AtomicUsize::new(n + 1), AtomicUsize::new(d + 1)],
            min_active: [AtomicUsize::new(n + 1), AtomicUsize::new(d + 1)],
        }
    }

    /// Call `f(c)` for every c in 0..n_chunks, spread over the pool, and return
    /// when all calls are done. A `run` from inside a chunk runs serially.
    pub fn run(&self, n_chunks: usize, f: &(dyn Fn(usize) + Sync)) {
        self.run_in(Kind::Wide, n_chunks, f)
    }

    /// `run` as a `kind` job: with that kind's participation.
    pub fn run_in(&self, kind: Kind, n_chunks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_chunks == 0 {
            return;
        }
        if IN_POOL.get() {
            (0..n_chunks).for_each(f);
            return;
        }
        let mut policy = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let adapt = &mut policy[kind as usize];
        let s = self.shared;
        let active = adapt.active;
        if active == 0 {
            // No workers (AXON_THREADS=1). Release the lock first: a nested
            // `run` from `f` would otherwise deadlock on it.
            drop(policy);
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
            self.active[kind as usize].store(a, Relaxed);
            self.min_active[kind as usize].fetch_min(a, Relaxed);
        }
        drop(policy);
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
        for kind in [Kind::Wide, Kind::Decode] {
            self.force_active_in(kind, workers);
        }
    }

    /// `force_active` for one kind of job.
    #[cfg(test)]
    fn force_active_in(&self, kind: Kind, workers: usize) {
        let mut p = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let (k, a) = (kind as usize, &mut p[kind as usize]);
        a.active = workers.min(a.max);
        self.active[k].store(a.active + 1, Relaxed);
        self.min_active[k].fetch_min(a.active + 1, Relaxed);
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
    /// Threads in decode jobs now and at least (caller included).
    pub decode_active: usize,
    pub decode_min: usize,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} judged, {} stalls ({:.1} ms lost), {} down, {} probes ({} failed); \
             decode {} active now, {} at least",
            self.judged,
            self.stalls,
            self.stall_us as f64 / 1e3,
            self.step_downs,
            self.probes,
            self.failed_probes,
            self.decode_active,
            self.decode_min
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
    /// Up to `max` workers, starting at all of them.
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

/// Most threads (caller included) decode jobs use, of `threads`:
/// AXON_DECODE_THREADS if set; on a hybrid CPU with both kinds of core
/// allowed, as many as there are physical fast cores (`hybrid_fast_cores`;
/// SMT siblings count once); else all. With AXON_POOL_ADAPT=0 only an explicit
/// AXON_DECODE_THREADS applies. One token's matvecs are memory-bound: past the cores that saturate
/// DRAM, a slow core only adds a tail every fork/join waits for.
fn decode_start(threads: usize) -> usize {
    let threads = threads.max(1);
    if let Some(n) = decode_threads_env() {
        return n.clamp(1, threads);
    }
    match hybrid_fast_cores() {
        Some(p) => p.clamp(1, threads),
        None => threads,
    }
}

/// AXON_DECODE_THREADS, if set to a number.
fn decode_threads_env() -> Option<usize> {
    std::env::var("AXON_DECODE_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
}

/// The performance cores this process may run on, if it may also run on
/// efficiency cores (Linux: /sys/devices/cpu_core and cpu_atom, and
/// Cpus_allowed_list); None on a homogeneous CPU or elsewhere. Also None
/// when SMT is on for those P-cores (cpu_core lists both hyperthreads of a
/// core; `physical_cores` counts them once): on an i5-13420H (4 P-cores x 2
/// SMT + 4 E-cores) a cap at the 4 physical or the 8 logical P CPUs both
/// measured 7-10% below no cap (tg128, 1.5B and 3B), so SMT-on hybrids keep
/// every thread.
fn hybrid_fast_cores() -> Option<usize> {
    let read = |p: &str| std::fs::read_to_string(p).ok();
    let status = read("/proc/self/status")?;
    let allowed = cpu_list(
        status
            .lines()
            .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?,
    );
    let mine = |path: &str| -> Option<Vec<usize>> {
        let cpus = cpu_list(&read(path)?);
        Some(cpus.into_iter().filter(|c| allowed.contains(c)).collect())
    };
    let (p, e) = (
        mine("/sys/devices/cpu_core/cpus")?,
        mine("/sys/devices/cpu_atom/cpus")?,
    );
    let siblings = |cpu: usize| {
        let t = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
        read(&format!("{t}/core_cpus_list")).or_else(|| read(&format!("{t}/thread_siblings_list")))
    };
    let cores = physical_cores(&p, siblings);
    (!p.is_empty() && !e.is_empty() && cores == p.len()).then_some(cores)
}

/// How many physical cores the logical `cpus` are on: cpus whose sibling
/// list (`siblings(cpu)`, a Linux cpu list; None if unknown) has the same
/// lowest cpu share a core; a cpu with no list is a core of its own.
fn physical_cores(cpus: &[usize], siblings: impl Fn(usize) -> Option<String>) -> usize {
    let mut cores: Vec<usize> = cpus
        .iter()
        .map(|&c| {
            siblings(c)
                .and_then(|s| cpu_list(&s).into_iter().min())
                .unwrap_or(c)
        })
        .collect();
    cores.sort_unstable();
    cores.dedup();
    cores.len()
}

/// A Linux cpu list ("0-7,9,12-15"); malformed parts are skipped.
fn cpu_list(s: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for part in s.trim().split(',') {
        let mut ends = part
            .trim()
            .splitn(2, '-')
            .map(|x| x.trim().parse::<usize>());
        match (ends.next(), ends.next()) {
            (Some(Ok(a)), None) => v.push(a),
            (Some(Ok(a)), Some(Ok(b))) if a <= b && b - a < 1 << 16 => v.extend(a..=b),
            _ => {}
        }
    }
    v
}

/// Split `data` into chunks of `chunk` items and call `f(chunk_index, chunk)`
/// for each, in parallel on the global pool.
pub fn for_each_chunk_mut<T: Send>(
    data: &mut [T],
    chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    for_each_chunk_mut_in(Kind::Wide, data, chunk, f)
}

/// `for_each_chunk_mut` as a `kind` job.
pub fn for_each_chunk_mut_in<T: Send>(
    kind: Kind,
    data: &mut [T],
    chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    assert!(chunk > 0);
    let len = data.len();
    let base = SendPtr(data.as_mut_ptr());
    Pool::global().run_in(kind, len.div_ceil(chunk), &|c| {
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

/// A raw pointer pool chunks share to write disjoint parts of one buffer.
pub(crate) struct SendPtr<T>(pub(crate) *mut T);
impl<T> SendPtr<T> {
    // A method (not field access) so closures capture the whole Sync wrapper,
    // not the raw pointer inside it.
    pub(crate) fn get(&self) -> *mut T {
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

    /// Each kind of job has its own participation: decode jobs use only
    /// their workers, wide jobs keep theirs, and a decode step down leaves
    /// wide jobs alone.
    #[test]
    fn kinds_participate_apart() {
        let pool = Pool::with(6, false);
        pool.force_active_in(Kind::Decode, 2);
        assert_eq!(pool.active_in(Kind::Decode), 3);
        assert_eq!(pool.active_in(Kind::Wide), 6);
        for n in [1, 7, 100, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let outsider = AtomicBool::new(false);
            pool.run_in(Kind::Decode, n, &|c| {
                if worker_index().is_some_and(|w| w >= 2) {
                    outsider.store(true, Relaxed);
                }
                hits[c].fetch_add(1, Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "n={n}");
            assert!(
                !outsider.load(Relaxed),
                "a left-out worker ran a decode chunk"
            );
            // Wide jobs in between wake everyone and still add up.
            let sum = AtomicUsize::new(0);
            pool.run(n, &|c| {
                sum.fetch_add(c, Relaxed);
            });
            assert_eq!(sum.load(Relaxed), (0..n).sum::<usize>());
        }
        assert_eq!(pool.active_in(Kind::Wide), 6);
        let word = pool.shared.epoch.0.load(Acquire);
        assert!(pool.shared.done.iter().all(|d| d.0.load(Acquire) == word));

        // The policy: a decode level with its own ceiling steps down on its
        // own and never probes past that ceiling.
        let mut p = [Adapt::new(7), Adapt::new(4)];
        assert_eq!(p[1].active, 4);
        for _ in 0..10 {
            p[1].record(true);
        }
        assert_eq!((p[0].active, p[1].active), (7, 1));
        for _ in 0..PROBE_MAX * 8 {
            p[1].record(false);
        }
        assert_eq!(p[1].active, 4);
        // SMT: hyperthread siblings count as one core.
        let smt = |c: usize| Some(format!("{}-{}", c & !1, c | 1));
        assert_eq!(physical_cores(&[0, 1, 2, 3, 4, 5, 6, 7], smt), 4);
        assert_eq!(physical_cores(&[0, 2, 3], smt), 2);
        assert_eq!(physical_cores(&[0, 1, 2, 3], |_| None), 4);
        let list = |c: usize| Some(if c < 2 { "0,1".into() } else { format!("{c}") });
        assert_eq!(physical_cores(&[0, 1, 2, 3], list), 3);
        assert!((1..=64).all(|t| (1..=t).contains(&decode_start(t))));
        assert_eq!(cpu_list("0-3,8, 10-11\n"), [0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(cpu_list(""), [] as [usize; 0]);
        assert_eq!(cpu_list("x,5-2,7"), [7]);
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
                // Workers sit on their chunk for 10 ms while the caller drains
                // the rest in microseconds: a stall every time, even when the
                // caller is starved too (qemu, tests sharing 2 vCPUs). Twice:
                // after a slow unwind the workers have parked, and a run that
                // wakes them isn't judged; the second finds them spinning.
                for _ in 0..2 {
                    let count = AtomicUsize::new(0);
                    pool.run(64, &|_| {
                        if worker_index().is_some() {
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        count.fetch_add(1, Relaxed);
                    });
                    assert_eq!(count.load(Relaxed), 64);
                }
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
        assert!(st.stall_us > 0, "10 ms stalls: {st}");
        assert_eq!(st.step_downs as usize, pool.threads() - pool.active());
        let count = AtomicUsize::new(0);
        pool.run(100, &|_| {
            count.fetch_add(1, Relaxed);
        });
        assert_eq!(count.load(Relaxed), 100);
    }
}
