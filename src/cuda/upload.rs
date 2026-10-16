//! The weight upload through a ring of page-locked staging buffers.
//!
//! A pageable `cuMemcpyHtoD` out of the file mapping runs at ~11-14 GB/s:
//! the driver stages it through its own small pinned buffer, one chunk at a
//! time, with one CPU copy in between. Here `threads` workers copy the
//! weights into `slots` page-locked buffers of `slot` bytes each, and the
//! calling thread DMAs each full slot to the device on a copy stream while
//! the workers fill the next ones: the CPU copies run in parallel and the
//! DMA runs at up to the link's pinned rate (`cuda::tests::h2d_bandwidth`).
//! The workers read the file with pread (no page faults in a mapping).
//! Measured: on a Gen4 x16 4090 host the 7B upload went 423 -> 291 ms; on
//! a laptop whose memory bandwidth is the limit (page cache -> staging ->
//! DMA is three passes over DRAM, as the driver's own path is) about the
//! same rate as before, faster only by the faults pread avoids.
//!
//! The device bytes are the same as `DevBuf::write_at`'s for every piece;
//! the gaps between pieces (arena alignment padding) are written as zeros.

use super::{DevBuf, Gpu, HostBuf, Pinned, Stream, SyncEvent};
use std::sync::{Condvar, Mutex};

/// Where a piece's bytes come from.
#[derive(Clone, Copy)]
pub enum Src<'a> {
    /// In memory (a file mapping, the anonymous copy, a side copy).
    Mem(&'a [u8]),
    /// `len` bytes of the file at this offset, read with pread: the kernel
    /// copies them out of the page cache, no page tables to fill.
    File(&'a std::fs::File, u64),
}

/// `len` bytes for the arena at `off`.
#[derive(Clone, Copy)]
pub struct Piece<'a> {
    pub off: usize,
    pub len: usize,
    pub src: Src<'a>,
}

impl Piece<'_> {
    fn end(&self) -> usize {
        self.off + self.len
    }
}

/// Ring geometry: staging buffers, bytes each, copy threads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub slots: usize,
    pub slot: usize,
    pub threads: usize,
}

impl Shape {
    /// The default: 8 x 8 MB from 1 GiB to upload, 4 x 4 MB below (pinning
    /// costs ~0.2-0.3 ms per MB, which a small model's upload doesn't win
    /// back), half the CPUs up to 8 threads. AXON_UPLOAD_RING=
    /// "slots,MB,threads" sets it. None (the driver's pageable copies, as
    /// before) with AXON_UPLOAD_RING=0, or below `MIN_BYTES`.
    /// AXON_UPLOAD_RING asks for the ring ("slots,MB,threads"): use it
    /// whatever is in the page cache.
    pub fn forced() -> bool {
        std::env::var("AXON_UPLOAD_RING").is_ok_and(|v| v != "0")
    }

    pub fn pick(total: usize) -> Option<Shape> {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        let big = total >= 1 << 30;
        let mut s = Shape {
            slots: if big { 8 } else { 4 },
            slot: if big { 8 << 20 } else { 4 << 20 },
            threads: (cpus / 2).clamp(1, 8),
        };
        match std::env::var("AXON_UPLOAD_RING").as_deref() {
            Ok("0") => return None,
            Ok(v) => {
                let p: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                // Sane sizes only: 2..=64 slots of 1..=256 MB, 1..=64 threads.
                if let [a, b, c] = p[..]
                    && (2..=64).contains(&a)
                    && (1..=256).contains(&b)
                    && (1..=64).contains(&c)
                {
                    s = Shape {
                        slots: a,
                        slot: b << 20,
                        threads: c,
                    };
                }
            }
            Err(_) => {}
        }
        (total >= MIN_BYTES).then_some(s)
    }
}

/// How far ahead of the copies the helper runs `pre` (read-ahead).
const PRE_AHEAD: usize = 256 << 20;

/// Below this the ring's setup (pinning its buffers) costs more than it saves.
const MIN_BYTES: usize = 64 << 20;

/// A raw pointer the workers may share: each writes only its own stripe of
/// a slot no DMA is reading (`Ring::upload`'s protocol).
#[derive(Clone, Copy)]
struct SlotPtr(*mut u8);
// SAFETY: see above; the pointee is page-locked host memory owned by the
// ring, alive for the whole scope the workers run in.
unsafe impl Send for SlotPtr {}
// SAFETY: as above.
unsafe impl Sync for SlotPtr {}

/// The page-locked staging memory and the slots' events.
pub struct Ring<'g> {
    shape: Shape,
    mem: Staging<'g>,
    events: Vec<SyncEvent<'g>>,
    /// The last upload's seconds spent waiting for the copies, and for
    /// the DMAs (AXON_UPLOAD_TRACE).
    pub waits: std::cell::Cell<(f64, f64)>,
}

/// `slots * slot` page-locked bytes: an anonymous mapping in transparent
/// huge pages, registered with the driver (pinning 2 MB pages is ~5x
/// cheaper than cuMemHostAlloc's 4 KB ones: 10 vs 56 ms for 64 MB,
/// `cuda::tests::h2d_bandwidth`), else cuMemHostAlloc.
enum Staging<'g> {
    /// Field order: unregistered before it's unmapped. The bytes start
    /// `at` bytes into the mapping.
    Registered {
        _pin: Pinned<'g>,
        map: memmap2::MmapMut,
        at: usize,
    },
    Alloc(HostBuf<'g>),
}

impl Staging<'_> {
    fn base(&mut self) -> Result<*mut u8, String> {
        Ok(match self {
            // SAFETY: the offset stays inside the mapping (`Ring::new`).
            Staging::Registered { map, at, .. } => unsafe { map.as_mut_ptr().add(*at) },
            Staging::Alloc(b) => b.bytes_mut()?.as_mut_ptr(),
        })
    }
}

const HUGE: usize = 2 << 20;

/// Shared between the uploading thread and the workers.
struct State {
    /// Chunks below this may be filled (their slot's last DMA ran).
    free_upto: usize,
    /// Stripes of each slot's current chunk filled so far.
    filled: Vec<usize>,
    /// Stop: an error on the uploading thread, or a worker panicked.
    abort: bool,
    /// A worker's error.
    err: Option<String>,
}

/// Sets `abort` if a worker unwinds, so nobody waits for its stripes.
struct AbortOnUnwind<'a>(&'a Mutex<State>, &'a Condvar);
impl Drop for AbortOnUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            if let Ok(mut s) = self.0.lock() {
                s.abort = true;
                s.err
                    .get_or_insert("weight upload: a copy thread panicked".into());
            }
            self.1.notify_all();
        }
    }
}

/// Copy the arena bytes [a, b) into `out`: the pieces' bytes where they
/// are, zeros between them. `pieces` sorted by offset, non-overlapping.
fn fill(out: &mut [u8], a: usize, b: usize, pieces: &[Piece]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    debug_assert_eq!(out.len(), b - a);
    let mut at = a; // next byte of out to write
    let mut i = pieces.partition_point(|p| p.end() <= a);
    while at < b {
        let Some(p) = pieces.get(i).filter(|p| p.off < b) else {
            out[at - a..].fill(0);
            return Ok(());
        };
        if p.off > at {
            out[at - a..p.off - a].fill(0);
            at = p.off;
        }
        let end = p.end().min(b);
        let dst = &mut out[at - a..end - a];
        match p.src {
            Src::Mem(s) => dst.copy_from_slice(&s[at - p.off..end - p.off]),
            Src::File(f, fo) => f.read_exact_at(dst, fo + (at - p.off) as u64)?,
        }
        at = end;
        i += 1;
    }
    Ok(())
}

impl<'g> Ring<'g> {
    /// Pin the staging memory (the slow part, see `Staging`).
    pub fn new(gpu: &'g Gpu, shape: Shape) -> Result<Ring<'g>, String> {
        let len = shape.slots * shape.slot;
        let mem = match Self::registered(gpu, len, shape.threads) {
            Some(m) => m,
            None => Staging::Alloc(gpu.host_alloc(len)?),
        };
        let events = (0..shape.slots)
            .map(|_| gpu.new_event())
            .collect::<Result<_, _>>()?;
        Ok(Ring {
            shape,
            mem,
            events,
            waits: Default::default(),
        })
    }

    /// `len` bytes of huge-page memory, faulted in by `threads` threads,
    /// then registered. None if that fails (no THP needed: 4 KB pages work,
    /// slower).
    fn registered(gpu: &'g Gpu, len: usize, threads: usize) -> Option<Staging<'g>> {
        if std::env::var("AXON_UPLOAD_HOSTALLOC").is_ok_and(|v| v == "1") {
            return None;
        }
        // One huge page more, to start on a huge-page boundary.
        let mut map = memmap2::MmapMut::map_anon(len + HUGE).ok()?;
        // 4 KB pages unless AXON_UPLOAD_THP=1: huge pages pin ~5x faster,
        // but their direct compaction (THP defrag=madvise) stalled for
        // 40-270 ms on a busy host with a big page cache.
        if std::env::var("AXON_UPLOAD_THP").as_deref() == Ok("1") {
            let _ = map.advise(memmap2::Advice::HugePage);
        }
        let at = (map.as_ptr() as usize).next_multiple_of(HUGE) - map.as_ptr() as usize;
        let mem = &mut map[at..at + len];
        let per = len.div_ceil(threads).next_multiple_of(HUGE);
        std::thread::scope(|sc| {
            for c in mem.chunks_mut(per) {
                // Only for speed: without a thread the registration
                // faults the pages in itself.
                let _ = std::thread::Builder::new().spawn_scoped(sc, move || c.fill(0));
            }
        });
        // SAFETY: the bytes stay mapped while the registration lives: the
        // two are dropped together, the registration first (`Staging`).
        let pin = unsafe { gpu.register(map.as_ptr().add(at), len, false) }.ok()?;
        Some(Staging::Registered { _pin: pin, map, at })
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Write `pieces` (arena offset, bytes; sorted, non-overlapping) to
    /// `dst` on `stream`, zeros in the gaps between them, and wait for it.
    /// `staged(i)` runs on this thread once piece i's bytes are all in the
    /// ring (its source isn't read again), in order. `pre(i)` runs on a
    /// helper thread ahead of the copies (e.g. filling a mapping's page
    /// tables); it should be cheap to skip.
    pub fn upload(
        &mut self,
        stream: &Stream,
        dst: &DevBuf,
        pieces: &[Piece],
        staged: &mut dyn FnMut(usize),
        pre: &(dyn Fn(usize) + Sync),
    ) -> Result<(), String> {
        let Some(first) = pieces.first().map(|p| p.off) else {
            return Ok(());
        };
        let end = pieces.iter().map(Piece::end).max().unwrap();
        if end > dst.len() {
            return Err(format!(
                "upload of arena bytes {first}..{end} into a {}-byte buffer",
                dst.len()
            ));
        }
        let Shape {
            slots,
            slot,
            threads,
        } = self.shape;
        let n_chunks = (end - first).div_ceil(slot);
        let base = self.mem.base()?;
        // SAFETY: slot i is bytes [i * slot, (i + 1) * slot) of the staging memory.
        let ptrs: Vec<SlotPtr> = (0..slots)
            .map(|i| SlotPtr(unsafe { base.add(i * slot) }))
            .collect();
        let state = Mutex::new(State {
            free_upto: slots,
            filled: vec![0; slots],
            abort: false,
            err: None,
        });
        let cv = Condvar::new();
        // Where the copies are: the helper stays ahead of it.
        let copying = std::sync::atomic::AtomicUsize::new(0);
        let chunk = |k: usize| {
            let a = first + k * slot;
            (a, (a + slot).min(end))
        };
        // Units of work: stripe w of chunk k is unit k * threads + w, taken
        // in order by whichever worker is free.
        let next = std::sync::atomic::AtomicUsize::new(0);
        let r = std::thread::scope(|sc| {
            // The helper: `pre` for each piece the copies haven't passed,
            // at most `PRE_AHEAD` bytes ahead of them (read-ahead mustn't
            // push what's about to be copied out of a small page cache).
            // Optional: without it the copies just read on their own.
            let _ = std::thread::Builder::new().spawn_scoped(sc, || {
                use std::sync::atomic::Ordering::Relaxed;
                for (i, p) in pieces.iter().enumerate() {
                    loop {
                        let c = copying.load(Relaxed);
                        if c == usize::MAX {
                            return;
                        }
                        let at = first + c * slot;
                        if p.end() <= at {
                            break; // passed: skip it
                        }
                        if p.off <= at + PRE_AHEAD {
                            pre(i);
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                }
            });
            let mut started = 0;
            for _ in 0..threads {
                let (state, cv, ptrs, next) = (&state, &cv, &ptrs, &next);
                let w = std::thread::Builder::new().spawn_scoped(sc, move || {
                    let _guard = AbortOnUnwind(state, cv);
                    loop {
                        let u = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let (k, w) = (u / threads, u % threads);
                        if k >= n_chunks {
                            return;
                        }
                        {
                            let mut s = state.lock().unwrap();
                            while !s.abort && k >= s.free_upto {
                                s = cv.wait(s).unwrap();
                            }
                            if s.abort {
                                return;
                            }
                        }
                        let (a, b) = chunk(k);
                        let len = b - a;
                        let (sa, sb) = (len * w / threads, len * (w + 1) / threads);
                        if sb > sa {
                            // SAFETY: slot k % slots has `slot` >= len bytes; this
                            // unit alone writes [sa, sb) of it, and no DMA reads
                            // the slot (k < free_upto: its last DMA finished).
                            let out = unsafe {
                                std::slice::from_raw_parts_mut(ptrs[k % slots].0.add(sa), sb - sa)
                            };
                            if let Err(e) = fill(out, a + sa, a + sb, pieces) {
                                let mut s = state.lock().unwrap();
                                s.abort = true;
                                s.err.get_or_insert(format!(
                                    "weight upload: reading the model: {e}"
                                ));
                                cv.notify_all();
                                return;
                            }
                        }
                        let mut s = state.lock().unwrap();
                        s.filled[k % slots] += 1;
                        if s.filled[k % slots] == threads {
                            cv.notify_all();
                        }
                    }
                });
                // Units are taken by whichever worker is free: fewer
                // threads than asked only cost speed; none is an error.
                started += w.is_ok() as usize;
            }
            let r = if started == 0 {
                Err("weight upload: couldn't start a copy thread".to_string())
            } else {
                self.drive(
                    stream, dst, n_chunks, &chunk, &ptrs, &state, &cv, &copying, pieces, staged,
                )
            };
            copying.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
            if r.is_err() {
                state.lock().unwrap_or_else(|e| e.into_inner()).abort = true;
                cv.notify_all();
            }
            r
        });
        // On every path: no DMA from the staging memory may outlive this
        // call (cuMemHostAlloc'd staging is freed by a drop that only waits
        // for the main stream).
        let synced = stream.sync();
        r?;
        synced
    }

    /// The uploading thread's side of `upload`: DMA each chunk once its
    /// stripes are in, recycle slots as their DMAs finish.
    #[allow(clippy::too_many_arguments)]
    fn drive(
        &self,
        stream: &Stream,
        dst: &DevBuf,
        n_chunks: usize,
        chunk: &dyn Fn(usize) -> (usize, usize),
        ptrs: &[SlotPtr],
        state: &Mutex<State>,
        cv: &Condvar,
        copying: &std::sync::atomic::AtomicUsize,
        pieces: &[Piece],
        staged: &mut dyn FnMut(usize),
    ) -> Result<(), String> {
        let Shape { slots, threads, .. } = self.shape;
        let mut next_staged = 0;
        let (mut wf, mut wd) = (0.0, 0.0);
        for k in 0..n_chunks {
            copying.store(k, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::Instant::now();
            {
                let mut s = state.lock().unwrap();
                while !s.abort && s.filled[k % slots] < threads {
                    s = cv.wait(s).unwrap();
                }
                if s.abort {
                    return Err(s
                        .err
                        .clone()
                        .unwrap_or("weight upload: a copy thread failed".into()));
                }
                s.filled[k % slots] = 0;
            }
            wf += t.elapsed().as_secs_f64();
            let (a, b) = chunk(k);
            // SAFETY: the slot's b - a bytes were written by the workers
            // (filled == threads, synchronized by the mutex) and stay
            // untouched until this DMA's event fired (free_upto below).
            unsafe { stream.h2d(dst, a, ptrs[k % slots].0, b - a)? };
            stream.record(&self.events[k % slots])?;
            while next_staged < pieces.len() && pieces[next_staged].end() <= b {
                staged(next_staged);
                next_staged += 1;
            }
            if k >= 1 {
                // Chunk k - 1's DMA done: its slot can take chunk k - 1 + slots.
                let t = std::time::Instant::now();
                self.events[(k - 1) % slots].sync()?;
                wd += t.elapsed().as_secs_f64();
                let mut s = state.lock().unwrap();
                s.free_upto = k - 1 + slots + 1;
                cv.notify_all();
            }
        }
        while next_staged < pieces.len() {
            staged(next_staged);
            next_staged += 1;
        }
        self.waits.set((wf, wd));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Piece, Src, fill};

    #[test]
    fn fill_copies_pieces_and_zeroes_gaps() {
        let p1 = [1u8; 10];
        let p3 = [3u8; 7];
        // p2 from a file: 5 bytes at offset 3 of it.
        let path = std::env::temp_dir().join(format!("axon-fill-{}", std::process::id()));
        std::fs::write(&path, [0, 0, 0, 2, 2, 2, 2, 2, 0]).unwrap();
        let f = std::fs::File::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let pieces = [
            Piece {
                off: 4,
                len: 10,
                src: Src::Mem(&p1),
            },
            Piece {
                off: 20,
                len: 5,
                src: Src::File(&f, 3),
            },
            Piece {
                off: 25,
                len: 7,
                src: Src::Mem(&p3),
            },
        ];
        let mut want = [0u8; 40];
        want[4..14].fill(1);
        want[20..25].fill(2);
        want[25..32].fill(3);
        for a in 0..40 {
            for b in a..=40 {
                let mut out = vec![9u8; b - a];
                fill(&mut out, a, b, &pieces).unwrap();
                assert_eq!(out, want[a..b], "[{a}, {b})");
            }
        }
        // Past the end of the file: an error.
        let short = [Piece {
            off: 0,
            len: 9,
            src: Src::File(&f, 3),
        }];
        assert!(fill(&mut [0u8; 9], 0, 9, &short).is_err());
    }
}
