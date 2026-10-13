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
//!   axon doesn't burn CPU.
//! - Fewer threads than cores by default: a spinning worker that gets
//!   preempted stalls everyone waiting on it. AXON_THREADS overrides.

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use std::sync::{Mutex, OnceLock};
use std::thread::Thread;
use std::time::{Duration, Instant};

const SPIN_FOR: Duration = Duration::from_micros(200);

thread_local! {
    /// True while this thread is running a chunk (worker threads always are).
    /// A nested `run` from inside a chunk would deadlock, so it runs serially.
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[repr(align(128))]
struct Padded<T>(T);

/// The job being run: a borrowed closure called once per chunk index.
/// Only valid while `run` is executing; `run` doesn't return until every
/// worker has finished with it.
#[derive(Clone, Copy)]
struct JobRef {
    f: *const (dyn Fn(usize) + Sync),
    n_chunks: usize,
}

struct Shared {
    /// Bumped by the caller to publish a new job (Release); workers Acquire it.
    epoch: Padded<AtomicUsize>,
    /// Next chunk to claim.
    next: Padded<AtomicUsize>,
    /// Written before `epoch` is bumped, read after it's observed.
    job: UnsafeCell<JobRef>,
    /// done[w] = last epoch worker w finished.
    done: Vec<Padded<AtomicUsize>>,
    /// sleeping[w]: worker w is parked or about to park; the caller must unpark it.
    sleeping: Vec<Padded<AtomicBool>>,
    /// Set if a chunk panicked on a worker thread.
    panicked: AtomicBool,
}

// The UnsafeCell is only written by the caller (holding `Pool::lock`) between
// jobs, while every worker is idle; workers read it only after Acquiring the
// epoch that published it.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    shared: &'static Shared,
    workers: Vec<Thread>,
    /// One job at a time; also makes `run` safe to call from several threads
    /// (e.g. the test harness).
    lock: Mutex<()>,
}

impl Pool {
    /// The process-wide pool, started on first use.
    pub fn global() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(|| Pool::new(default_threads()))
    }

    /// Total threads that run chunks, including the caller.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    fn new(threads: usize) -> Pool {
        let n = threads.max(1) - 1;
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            epoch: Padded(AtomicUsize::new(0)),
            next: Padded(AtomicUsize::new(0)),
            job: UnsafeCell::new(JobRef {
                f: &|_| {},
                n_chunks: 0,
            }),
            done: (0..n).map(|_| Padded(AtomicUsize::new(0))).collect(),
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
            lock: Mutex::new(()),
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
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let s = self.shared;
        if self.workers.is_empty() {
            (0..n_chunks).for_each(f);
            return;
        }
        // Publish the job. Workers are all idle (the previous run waited for
        // them), so nobody is reading `job` or `next` right now.
        // SAFETY: the transmute only erases the closure's lifetime. Workers use
        // the pointer only until they mark this epoch done, and we wait for
        // that below before returning, so it never outlives `f`.
        let f_static: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        unsafe {
            *s.job.get() = JobRef {
                f: f_static,
                n_chunks,
            }
        };
        s.next.0.store(0, Relaxed);
        let epoch = s.epoch.0.load(Relaxed) + 1;
        s.epoch.0.store(epoch, SeqCst);
        for (w, t) in self.workers.iter().enumerate() {
            if s.sleeping[w].0.load(SeqCst) {
                t.unpark();
            }
        }
        // The caller works too.
        IN_POOL.set(true);
        let mine = catch_unwind(AssertUnwindSafe(|| claim_chunks(s, f, n_chunks)));
        IN_POOL.set(false);
        for d in &s.done {
            let mut spins = 0u32;
            while d.0.load(Acquire) != epoch {
                // If that worker was preempted (more threads than free cores),
                // give up our core instead of spinning a whole timeslice.
                spins = spins.wrapping_add(1);
                if spins.is_multiple_of(10_000) {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
            }
        }
        // Clear the worker-panic flag before re-raising our own panic, or a
        // stale flag would fail the next, unrelated run.
        let worker_panicked = s.panicked.swap(false, Relaxed);
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        if worker_panicked {
            panic!("a pool worker panicked while running a chunk");
        }
    }
}

fn claim_chunks(s: &Shared, f: &(dyn Fn(usize) + Sync), n: usize) {
    loop {
        let c = s.next.0.fetch_add(1, Relaxed);
        if c >= n {
            break;
        }
        f(c);
    }
}

fn worker(s: &'static Shared, w: usize) {
    IN_POOL.set(true);
    let mut seen = 0;
    loop {
        // Wait for a new epoch: spin for a while, then park.
        let start = Instant::now();
        let mut spins = 0u32;
        let epoch = loop {
            let e = s.epoch.0.load(Acquire);
            if e != seen {
                break e;
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(64) && start.elapsed() > SPIN_FOR {
                s.sleeping[w].0.store(true, SeqCst);
                // Re-check after announcing we're going to sleep: either we see
                // the new epoch here, or the caller sees `sleeping` and unparks
                // us (and an unpark before park makes park return at once).
                let e = s.epoch.0.load(SeqCst);
                if e == seen {
                    std::thread::park();
                }
                s.sleeping[w].0.store(false, SeqCst);
                continue;
            }
            std::hint::spin_loop();
        };
        seen = epoch;
        // SAFETY: published before `epoch` (which we Acquired) and not modified
        // until every worker, including us, has stored done = epoch.
        let job = unsafe { *s.job.get() };
        let f = unsafe { &*job.f };
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| claim_chunks(s, f, job.n_chunks))) {
            // Leak the payload: if its Drop panicked, this thread would die and
            // the caller would wait on our done slot forever.
            std::mem::forget(payload);
            s.panicked.store(true, Relaxed);
        }
        s.done[w].0.store(epoch, Release);
    }
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
}
