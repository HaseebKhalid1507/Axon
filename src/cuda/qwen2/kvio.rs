//! A cache's rows to and from host memory (kvsave): the bytes of
//! positions lo..hi in the image layout of `crate::kvsave` (64-position
//! pages; per page, per region, every layer's K then V rows), staged
//! through page-locked memory, one strided copy (`cuMemcpy2DAsync`) per
//! region per page. Nothing numeric runs: a restored cache holds the bytes
//! the saved one did, so it continues bit for bit.

use super::*;
use crate::kvsave::{PAGE, segments};
use std::io::{self, Read, Write};

/// Most bytes staged per round trip through pinned memory.
const STAGE: usize = 64 << 20;

/// A cache's rows in device memory: `kv` holds, per region, `2 * nl`
/// arrays of `max_ctx` rows (`CudaCache::kv`'s layout); `base` is the
/// row a sequence starts at (a pool slot's `slot * ctx`), and `len` its
/// positions (a slot's ctx: its rows end there, the next slot's begin).
/// Paged (`tab`, a sequence's page list): position t is at row
/// `tab[t / PAGE] * PAGE + t % PAGE` (base 0), a page's rows contiguous in
/// each array as before.
pub(crate) struct Rows<'a, 'g> {
    pub kv: &'a DevBuf<'g>,
    pub max_ctx: usize,
    pub base: usize,
    pub len: usize,
    pub tab: Option<Vec<u32>>,
    /// Bytes per row of each region.
    pub regions: Vec<usize>,
    pub arrays: usize,
}

impl Rows<'_, '_> {
    /// Image bytes of one position.
    pub fn row_bytes(&self) -> usize {
        self.arrays * self.regions.iter().sum::<usize>()
    }

    /// Enqueue the copies of positions lo..hi (inside one page) between the
    /// device and `host` at `at`, in image order; `to_dev`: host to device.
    pub(crate) fn page(
        &self,
        host: &mut HostBuf,
        at: usize,
        lo: usize,
        hi: usize,
        to_dev: bool,
    ) -> Result<(), String> {
        let n = hi - lo;
        let first = match &self.tab {
            None => self.base + lo,
            Some(t) => t[lo / PAGE] as usize * PAGE + lo % PAGE,
        };
        let (mut img, mut reg) = (at, 0);
        for &row in &self.regions {
            let off = reg + first * row;
            let shape = (n * row, self.arrays);
            let pitch = self.max_ctx * row;
            if to_dev {
                host.copy_2d_to(img, n * row, self.kv, off, pitch, shape)?;
            } else {
                host.copy_2d_from(img, n * row, self.kv, off, pitch, shape)?;
            }
            img += self.arrays * n * row;
            reg += self.arrays * self.max_ctx * row;
        }
        Ok(())
    }

    /// Pages of lo..hi grouped into rounds of at most STAGE bytes:
    /// (first segment, end segment) index ranges, and the staging size.
    fn rounds(&self, segs: &[(usize, usize)]) -> (Vec<(usize, usize)>, usize) {
        let rb = self.row_bytes();
        let per = (STAGE / (PAGE * rb)).max(1);
        let rounds: Vec<_> = (0..segs.len())
            .step_by(per)
            .map(|i| (i, (i + per).min(segs.len())))
            .collect();
        let most = rounds
            .iter()
            .map(|&(a, b)| segs[a..b].iter().map(|s| (s.1 - s.0) * rb).sum::<usize>())
            .max()
            .unwrap_or(0);
        (rounds, most)
    }

    /// Positions lo..hi's image to `out`.
    pub fn save(&self, gpu: &Gpu, lo: usize, hi: usize, out: &mut dyn Write) -> io::Result<()> {
        self.check(lo, hi)?;
        if lo == hi {
            return Ok(());
        }
        let segs = segments(lo, hi);
        let (rounds, most) = self.rounds(&segs);
        let mut host = gpu.host_alloc(most).map_err(io::Error::other)?;
        let rb = self.row_bytes();
        for (a, b) in rounds {
            let mut at = 0;
            for &(s, e) in &segs[a..b] {
                self.page(&mut host, at, s, e, false)
                    .map_err(io::Error::other)?;
                at += (e - s) * rb;
            }
            out.write_all(&host.bytes_mut().map_err(io::Error::other)?[..at])?;
        }
        Ok(())
    }

    /// Positions lo..hi from their image in `src`. Enqueued: the copies
    /// are ordered before any later work on the stream (the staging buffer
    /// waits for them before it is refilled or freed).
    pub fn load(&self, gpu: &Gpu, lo: usize, hi: usize, src: &mut dyn Read) -> io::Result<()> {
        self.check(lo, hi)?;
        if lo == hi {
            return Ok(());
        }
        let segs = segments(lo, hi);
        let (rounds, most) = self.rounds(&segs);
        let mut host = gpu.host_alloc(most).map_err(io::Error::other)?;
        let rb = self.row_bytes();
        for (a, b) in rounds {
            let n: usize = segs[a..b].iter().map(|s| (s.1 - s.0) * rb).sum();
            src.read_exact(&mut host.bytes_mut().map_err(io::Error::other)?[..n])?;
            let mut at = 0;
            for &(s, e) in &segs[a..b] {
                self.page(&mut host, at, s, e, true)
                    .map_err(io::Error::other)?;
                at += (e - s) * rb;
            }
        }
        // Drop waits for the copies (HostBuf), so the bytes have landed.
        drop(host);
        Ok(())
    }

    pub(crate) fn check(&self, lo: usize, hi: usize) -> io::Result<()> {
        let rows_ok = match &self.tab {
            None => self.base + hi <= self.max_ctx,
            Some(t) => {
                hi.div_ceil(PAGE) <= t.len()
                    && t[..hi.div_ceil(PAGE)]
                        .iter()
                        .all(|&p| (p as usize + 1) * PAGE <= self.max_ctx)
            }
        };
        if lo > hi || hi > self.len || !rows_ok {
            return Err(io::Error::other(format!(
                "positions {lo}..{hi} of a {}-position sequence at row {} of a {}-row cache",
                self.len, self.base, self.max_ctx
            )));
        }
        Ok(())
    }
}

impl<'g> CudaQwen2<'g> {
    /// Bytes per row of each region of a `mode` cache: the rows, then (q8)
    /// their scales (`adopt_cache`'s walk).
    pub(crate) fn kv_regions(&self, mode: CacheKv) -> Vec<usize> {
        let kvd = self.kv_dim();
        match mode {
            CacheKv::F16 => vec![kvd * 2],
            CacheKv::Q8 { group } => {
                let g = if group == 0 { self.head_dim } else { group };
                vec![kvd, kvd / g * 4]
            }
        }
    }

    /// `c`'s rows (a whole cache: base 0; a slot view: its slot's, and
    /// no further; a paged pool's slot view: through its table row, read
    /// back from the device: only the pages the slot holds are its).
    pub(crate) fn cache_rows<'a>(&self, c: &'a CudaCache<'g>) -> io::Result<Rows<'a, 'g>> {
        let tab = match &c.tab {
            None => None,
            Some(t) => {
                let all: Vec<u32> = t.read().map_err(io::Error::other)?;
                let n = c.max_ctx.div_ceil(PAGE);
                let row = all
                    .get(c.tab0..(c.tab0 + n).min(all.len()))
                    .ok_or_else(|| io::Error::other("a paged cache's table past the pool's"))?;
                // A pool's slot view: the sink's entries (positions the slot
                // holds no page for) end the list.
                let sink = (c.rows / PAGE - 1) as u32;
                let held = row.iter().copied().take_while(|&p| !c.view || p != sink);
                Some(held.collect())
            }
        };
        Ok(Rows {
            kv: &c.kv,
            max_ctx: c.rows,
            base: if tab.is_some() { 0 } else { c.row0 },
            len: c.max_ctx,
            tab,
            regions: self.kv_regions(c.mode),
            arrays: 2 * self.layers.len(),
        })
    }

    /// Image bytes of one position of a `c` cache.
    pub(crate) fn cache_row_bytes(&self, c: &CudaCache) -> usize {
        2 * self.layers.len() * self.kv_regions(c.mode).iter().sum::<usize>()
    }

    /// The KV type `c` stores, as kvsave's identity names it.
    pub(crate) fn cache_kv_name(&self, c: &CudaCache) -> String {
        match c.mode {
            CacheKv::F16 => "f16".into(),
            CacheKv::Q8 { group } => {
                format!("q8/{}", if group == 0 { self.head_dim } else { group })
            }
        }
    }

    pub(crate) fn gpu(&self) -> &'g Gpu {
        self.gpu
    }

    /// Whether it runs the fast numerics mode's module (`Module::fast`).
    pub(crate) fn fast(&self) -> bool {
        self.fast
    }

    /// Its module's source and options hash (`Module::src_hash`).
    pub(crate) fn src_hash(&self) -> u128 {
        self.src_hash
    }

    /// Count one launch of a fast-set kernel (`#ifdef AXON_FAST` code, or a
    /// product routed to one in fast mode). Every such launch site calls it,
    /// so the fast-vs-exact gate (`cuda::qwen2::tests::numerics`) can tell
    /// a graded arm that ran a fast kernel from one that ran only exact
    /// ones. No kernel calls it yet: none exists (`numerics::fast_kernels`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn fast_launched(&self) {
        debug_assert!(self.fast, "a fast-set kernel launched from an exact module");
        self.fast_launches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Fast-set kernel launches so far (`fast_launched`).
    #[cfg_attr(not(test), allow(dead_code))] // the fast-vs-exact gate reads it
    pub(crate) fn fast_launches(&self) -> u64 {
        self.fast_launches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn kv_dims(&self) -> (usize, usize, usize) {
        (self.layers.len(), self.kv_dim(), self.head_dim)
    }

    /// `c`'s image of positions lo..hi to `out`.
    pub(crate) fn save_kv(
        &self,
        c: &CudaCache<'g>,
        lo: usize,
        hi: usize,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        if c.net != self.id {
            return Err(io::Error::other("a cache of another model"));
        }
        self.cache_rows(c)?.save(self.gpu, lo, hi, out)
    }

    /// Positions lo..hi of `c` from their image in `src`.
    pub(crate) fn load_kv(
        &self,
        c: &mut CudaCache<'g>,
        lo: usize,
        hi: usize,
        src: &mut dyn Read,
    ) -> io::Result<()> {
        if c.net != self.id {
            return Err(io::Error::other("a cache of another model"));
        }
        self.cache_rows(c)?.load(self.gpu, lo, hi, src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Median seconds of `f` over `n` runs (after one warm-up).
    fn time(gpu: &Gpu, n: usize, mut f: impl FnMut()) -> f64 {
        f();
        gpu.sync().unwrap();
        let mut t: Vec<f64> = (0..n)
            .map(|_| {
                let t0 = Instant::now();
                f();
                gpu.sync().unwrap();
                t0.elapsed().as_secs_f64()
            })
            .collect();
        t.sort_by(f64::total_cmp);
        t[n / 2]
    }

    /// Median seconds of `f` over `n` runs (after one warm-up): (to enqueue
    /// its copies, until they are done).
    fn time_enq(gpu: &Gpu, n: usize, mut f: impl FnMut()) -> (f64, f64) {
        f();
        gpu.sync().unwrap();
        let (mut e, mut t): (Vec<f64>, Vec<f64>) = (0..n)
            .map(|_| {
                let t0 = Instant::now();
                f();
                let e = t0.elapsed().as_secs_f64();
                gpu.sync().unwrap();
                (e, t0.elapsed().as_secs_f64())
            })
            .unzip();
        e.sort_by(f64::total_cmp);
        t.sort_by(f64::total_cmp);
        (e[n / 2], t[n / 2])
    }

    /// One 1D copy of `n` bytes between `host` + `at` and `dev` + `off`
    /// (`to_dev`: host to device), enqueued on the stream.
    fn copy_1d(host: &HostBuf, to_dev: bool, at: usize, dev: &DevBuf, off: usize, n: usize) {
        assert!(
            at + n <= host.len && off + n <= dev.len,
            "1D copy out of bounds"
        );
        let g = host.gpu;
        g.bind().unwrap();
        host.pending.set(true);
        let (h, d) = (host.p as usize + at, dev.ptr + off as u64);
        // SAFETY: in bounds of both (asserted); pinned and device memory
        // outlive the copy (the caller syncs before either is dropped).
        let r = unsafe {
            if to_dev {
                (g.d.cuMemcpyHtoDAsync_v2)(d, h as *const _, n, g.stream)
            } else {
                (g.d.cuMemcpyDtoHAsync_v2)(h as *mut _, d, n, g.stream)
            }
        };
        crate::cuda::ffi::check(g.d, r, "1D copy").unwrap();
    }

    /// One 2D device-to-device copy of `height` rows of `width` bytes, row i
    /// from `src` + `off` + i * `pitch` to `dst` + `at` + i * `dpitch`.
    fn copy_2d_dd(
        src: &DevBuf,
        (off, pitch): (usize, usize),
        dst: &DevBuf,
        (at, dpitch): (usize, usize),
        (width, height): (usize, usize),
    ) {
        let end = |o: usize, p: usize| o + (height - 1) * p + width;
        assert!(
            end(off, pitch) <= src.len && end(at, dpitch) <= dst.len,
            "2D D2D out of bounds"
        );
        let g = src.gpu;
        g.bind().unwrap();
        use crate::cuda::ffi;
        let c = ffi::Memcpy2D {
            src_type: ffi::CU_MEMORYTYPE_DEVICE,
            src_dev: src.ptr + off as u64,
            src_pitch: pitch,
            dst_type: ffi::CU_MEMORYTYPE_DEVICE,
            dst_dev: dst.ptr + at as u64,
            dst_pitch: dpitch,
            width,
            height,
            ..Default::default()
        };
        // SAFETY: every row in bounds of both (asserted); both outlive it.
        let r = unsafe { (g.d.cuMemcpy2DAsync_v2)(&c, g.stream) };
        ffi::check(g.d, r, "2D D2D copy").unwrap();
    }

    /// WP2 checkpoint 1 (docs/design/kv-ram-tier-exclusive.md §5.1): other
    /// shapes for a 64-position page's copy than `Rows::page`'s one 2D copy
    /// (`arrays` rows of 64 * `row` bytes, `ctx * row` apart on the device).
    /// `host` gets the image layout (a page's arrays back to back). Per row:
    /// the time to enqueue the copies, and until they are done.
    /// - 56 x 1D: one 1D copy per array row (a loop in `Rows::page`).
    /// - 1D per page, scattered: one 3.5 MiB copy per page between a
    ///   contiguous device buffer and host slots in a stride permutation
    ///   (the RAM tier's arena slots: a session's pages aren't contiguous).
    /// - D2D: a page's rows gathered into (scattered from) a 64 MiB device
    ///   staging buffer, by one 2D copy or 56 1D copies per page, no kernel.
    /// - gather / scatter path: D2D per page and 1D per page through the
    ///   staging buffer, 18 pages a fill (everything on the one stream).
    fn copy_paths(
        g: &Gpu,
        dev: &DevBuf,
        host: &HostBuf,
        (ctx, arrays, row): (usize, usize, usize),
        tag: &str,
    ) {
        let w = 64 * row;
        let pb = w * arrays;
        let np = ctx / 64;
        let bytes = np * pb;
        let report = |what: &str, (e, s): (f64, f64), b: usize| {
            let what = format!("{what}{tag}");
            eprintln!(
                "BW {what:<34} {:8.2} ms {:7.2} GB/s  (enqueue {:7.2} ms)",
                s * 1e3,
                b as f64 / s / 1e9,
                e * 1e3
            )
        };
        let n = 7;
        // A page's array a: `a * ctx * row + p * w` on the device.
        let arr = |p: usize, a: usize| a * ctx * row + p * w;
        for to_dev in [false, true] {
            let d = if to_dev { "H2D" } else { "D2H" };
            let s = time_enq(g, n, || {
                for p in 0..np {
                    for a in 0..arrays {
                        copy_1d(host, to_dev, p * pb + a * w, dev, arr(p, a), w);
                    }
                }
            });
            report(&format!("{d} 56 x 1D per 64-pos page"), s, bytes);
            let s = time_enq(g, 31, || {
                for a in 0..arrays {
                    copy_1d(host, to_dev, a * w, dev, arr(0, a), w);
                }
            });
            report(&format!("{d} 56 x 1D, one page"), s, pb);
        }
        // Host slot of page p: a stride permutation of 0..np.
        let k = (np / 3 + 1..).find(|&k| gcd(k, np) == 1).unwrap();
        let slot = |p: usize| (p * k) % np * pb;
        for to_dev in [false, true] {
            let d = if to_dev { "H2D" } else { "D2H" };
            let s = time_enq(g, n, || {
                for p in 0..np {
                    copy_1d(host, to_dev, slot(p), dev, p * pb, pb);
                }
            });
            report(&format!("{d} 1D per page, scattered host"), s, bytes);
        }
        let stage = g.alloc(64 << 20).unwrap();
        let fill = (64 << 20) / pb;
        let st = |p: usize| p % fill * pb;
        for gather in [true, false] {
            let d = if gather { "gather" } else { "scatter" };
            let s = time_enq(g, n, || {
                for p in 0..np {
                    if gather {
                        copy_2d_dd(dev, (arr(p, 0), ctx * row), &stage, (st(p), w), (w, arrays));
                    } else {
                        copy_2d_dd(&stage, (st(p), w), dev, (arr(p, 0), ctx * row), (w, arrays));
                    }
                }
            });
            report(&format!("D2D 2D per page ({d})"), s, bytes);
            let s = time_enq(g, n, || {
                for p in 0..np {
                    for a in 0..arrays {
                        if gather {
                            dev.copy_into(arr(p, a), &stage, st(p) + a * w, w).unwrap();
                        } else {
                            stage.copy_into(st(p) + a * w, dev, arr(p, a), w).unwrap();
                        }
                    }
                }
            });
            report(&format!("D2D 56 x 1D per page ({d})"), s, bytes);
        }
        // The whole path, a fill at a time: out = gather then D2H per page
        // to its host slot; in = H2D per page from its slot, then scatter.
        for out in [true, false] {
            let path = |pages: usize| {
                for f0 in (0..pages).step_by(fill) {
                    let ps = f0..(f0 + fill).min(pages);
                    for p in ps.clone() {
                        if out {
                            copy_2d_dd(
                                dev,
                                (arr(p, 0), ctx * row),
                                &stage,
                                (st(p), w),
                                (w, arrays),
                            );
                        } else {
                            copy_1d(host, true, slot(p), &stage, st(p), pb);
                        }
                    }
                    for p in ps {
                        if out {
                            copy_1d(host, false, slot(p), &stage, st(p), pb);
                        } else {
                            copy_2d_dd(
                                &stage,
                                (st(p), w),
                                dev,
                                (arr(p, 0), ctx * row),
                                (w, arrays),
                            );
                        }
                    }
                }
            };
            let d = if out {
                "D2H gather path"
            } else {
                "H2D scatter path"
            };
            report(
                &format!("{d} (2D D2D + 1D)"),
                time_enq(g, n, || path(np)),
                bytes,
            );
            report(&format!("{d}, one page"), time_enq(g, 31, || path(1)), pb);
        }
        g.sync().unwrap();
    }

    /// The GPU's performance state, clocks and display (nvidia-smi), for
    /// the log: bella's 2D copies ran at 7.1 GB/s in some runs and at 28 in
    /// others, the same binary (docs/design/kv-ram-tier-exclusive.md §5.1).
    fn gpu_state(when: &str) {
        let q = "--query-gpu=pstate,clocks.gr,clocks.mem,clocks.video,display_active,\
                 clocks_event_reasons.active";
        match std::process::Command::new("nvidia-smi")
            .args([q, "--format=csv,noheader"])
            .output()
        {
            Ok(o) => eprintln!(
                "GPU state ({when}): {}",
                String::from_utf8_lossy(if o.status.success() {
                    &o.stdout
                } else {
                    &o.stderr
                })
                .trim()
            ),
            Err(e) => eprintln!("GPU state ({when}): nvidia-smi: {e}"),
        }
    }

    /// A one-block kernel spinning on a stream of its own until `stop` is
    /// set (or `most_ns` pass): the GPU busy, as it is while serving, for
    /// copies timed on the main stream meanwhile (`Spin`).
    const SPIN: &str = r#"
extern "C" __global__ void bw_spin(const volatile unsigned int* stop, unsigned long long most_ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (*stop == 0 && t - t0 < most_ns);
}
"#;

    /// `SPIN` running (`Spin::start`) until `stop`.
    struct Spin<'g> {
        s: crate::cuda::Stream<'g>,
        flag: DevBuf<'g>,
        _m: crate::cuda::Module<'g>,
    }

    impl<'g> Spin<'g> {
        fn start(gpu: &'g Gpu) -> Spin<'g> {
            let m = crate::cuda::Module::load(gpu, SPIN, &crate::cuda::KERNEL_OPTS)
                .unwrap()
                .0;
            let flag = gpu.alloc(4).unwrap();
            flag.write(&[0u32]).unwrap();
            let s = gpu.new_stream().unwrap();
            let k = m.function("bw_spin").unwrap();
            // SAFETY: bw_spin's two parameters; it reads flag's 4 bytes,
            // which live until `stop` has waited for it.
            let args = [flag.arg(), crate::cuda::Arg::U64(60_000_000_000)];
            unsafe { gpu.launch_on(&s, &k, (1, 1, 1), (32, 1, 1), 0, &args) }.unwrap();
            drop(k);
            // Clocks follow load with a lag.
            std::thread::sleep(std::time::Duration::from_millis(300));
            Spin { s, flag, _m: m }
        }

        /// Stop it, and wait until it has.
        fn stop(self) {
            // The main stream's copy runs beside the kernel on `s`.
            self.flag.write(&[1u32]).unwrap();
            self.s.sync().unwrap();
        }
    }

    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    /// `Rows`: unaligned lo..hi at a nonzero base, two regions (q8's
    /// shape), an image over STAGE (several staging rounds): the image is
    /// kvsave's layout, and a load puts back exactly those rows, nothing
    /// else.
    #[test]
    fn rows_round_trip_multi_round() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let (arrays, regions, max_ctx, base) = (48, vec![512usize, 64], 3000, 200);
        let total: usize = regions.iter().map(|r| arrays * max_ctx * r).sum();
        let src: Vec<u8> = (0..total).map(|i| (i * 131 + i / 7) as u8).collect();
        let dev = g.alloc(total).unwrap();
        dev.write(&src).unwrap();
        let rows = |kv| Rows {
            tab: None,
            kv,
            max_ctx,
            base,
            len: max_ctx - base,
            regions: regions.clone(),
            arrays,
        };
        let (lo, hi) = (37, 2750);
        let r = rows(&dev);
        assert!((hi - lo) * r.row_bytes() > STAGE, "more than one round");
        let mut img = Vec::new();
        r.save(&g, lo, hi, &mut img).unwrap();
        // Each region's (offset, row), for the reference walks.
        let regs: Vec<(usize, usize)> = regions
            .iter()
            .scan(0, |off, &row| {
                let o = *off;
                *off += arrays * max_ctx * row;
                Some((o, row))
            })
            .collect();
        let mut want = Vec::new();
        for (s, e) in segments(lo, hi) {
            for &(reg, row) in &regs {
                for a in 0..arrays {
                    let off = reg + (a * max_ctx + base + s) * row;
                    want.extend_from_slice(&src[off..off + (e - s) * row]);
                }
            }
        }
        assert!(img == want, "image layout");
        let dev2 = g.alloc(total).unwrap();
        dev2.zero().unwrap();
        rows(&dev2).load(&g, lo, hi, &mut &img[..]).unwrap();
        let back: Vec<u8> = dev2.read().unwrap();
        let mut expect = vec![0u8; total];
        for &(reg, row) in &regs {
            for a in 0..arrays {
                let off = reg + (a * max_ctx + base + lo) * row;
                let n = (hi - lo) * row;
                expect[off..off + n].copy_from_slice(&src[off..off + n]);
            }
        }
        assert!(back == expect, "rows loaded back");
        // Nothing to copy: no staging at all.
        r.save(&g, 5, 5, &mut Vec::new()).unwrap();
    }

    /// D2H and H2D bandwidth for a 7B-shaped slot (28 layers, kv_dim 512,
    /// f16: 1 KiB rows, 56 arrays) of `AXON_BW_CTX` (default 32768)
    /// positions: contiguous, one 2D copy for the slot, one per 64-position
    /// page, and the other page copy shapes of `copy_paths`.
    /// `cargo test kvio::tests::bandwidth -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bandwidth() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let ctx: usize = std::env::var("AXON_BW_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32768);
        let (arrays, row) = (56, 1024);
        let bytes = arrays * ctx * row;
        let info = g.info().unwrap();
        eprintln!(
            "device {} ({}), slot {ctx} positions = {:.3} GiB",
            info.name,
            info.arch(),
            bytes as f64 / (1u64 << 30) as f64
        );
        gpu_state("start");
        let dev = g.alloc(bytes).unwrap();
        let mut host = g.host_alloc(bytes).unwrap();
        let gbs = |s: f64| bytes as f64 / s / 1e9;
        let n = 7;
        let report =
            |what: &str, s: f64| eprintln!("BW {what:<34} {:8.2} ms {:7.2} GB/s", s * 1e3, gbs(s));
        let s = time(&g, n, || host.copy_from(&dev, 0, bytes).unwrap());
        report("D2H contiguous", s);
        let s = time(&g, n, || host.copy_to(&dev, 0, bytes).unwrap());
        report("H2D contiguous", s);
        let rows = Rows {
            tab: None,
            kv: &dev,
            max_ctx: ctx,
            base: 0,
            len: ctx,
            regions: vec![row],
            arrays,
        };
        let s = time(&g, n, || {
            host.copy_2d_from(0, ctx * row, &dev, 0, ctx * row, (ctx * row, arrays))
                .unwrap()
        });
        report("D2H one 2D copy", s);
        let s = time(&g, n, || {
            host.copy_2d_to(0, ctx * row, &dev, 0, ctx * row, (ctx * row, arrays))
                .unwrap()
        });
        report("H2D one 2D copy", s);
        for page in [16, 64, 256] {
            let s = time(&g, n, || {
                for p in 0..ctx / page {
                    let at = p * page * row * arrays;
                    host.copy_2d_from(
                        at,
                        page * row,
                        &dev,
                        p * page * row,
                        ctx * row,
                        (page * row, arrays),
                    )
                    .unwrap();
                }
            });
            report(&format!("D2H 2D per {page}-pos page"), s);
            let s = time(&g, n, || {
                for p in 0..ctx / page {
                    let at = p * page * row * arrays;
                    host.copy_2d_to(
                        at,
                        page * row,
                        &dev,
                        p * page * row,
                        ctx * row,
                        (page * row, arrays),
                    )
                    .unwrap();
                }
            });
            report(&format!("H2D 2D per {page}-pos page"), s);
        }
        copy_paths(&g, &dev, &host, (ctx, arrays, row), "");
        gpu_state("after the copies, idle");
        // Again with the GPU busy (`Spin`): today's 2D page copy and the
        // other shapes.
        let spin = Spin::start(&g);
        gpu_state("busy");
        for to_dev in [false, true] {
            let s = time(&g, n, || {
                for p in 0..ctx / 64 {
                    let (at, off, w) = (p * 64 * row * arrays, p * 64 * row, 64 * row);
                    if to_dev {
                        host.copy_2d_to(at, w, &dev, off, ctx * row, (w, arrays))
                    } else {
                        host.copy_2d_from(at, w, &dev, off, ctx * row, (w, arrays))
                    }
                    .unwrap();
                }
            });
            let d = if to_dev { "H2D" } else { "D2H" };
            report(&format!("{d} 2D per 64-pos page (busy)"), s);
        }
        copy_paths(&g, &dev, &host, (ctx, arrays, row), " (busy)");
        gpu_state("busy, at the end");
        spin.stop();
        // The RAM tier's costs (serve --cache-ram): a pinned slab's
        // allocation (cuMemHostAlloc + zeroing), and one page's D2H / H2D
        // alone (the write-behind's unit), each a median of n.
        for mib in [64usize, 256] {
            let t = time(&g, n, || drop(g.host_alloc(mib << 20).unwrap()));
            eprintln!(
                "BW host_alloc {mib:>4} MiB (+ free)          {:8.2} ms",
                t * 1e3
            );
            // SAFETY: never read: allocated and freed.
            let t = time(&g, n, || {
                drop(unsafe { g.host_alloc_uninit(mib << 20) }.unwrap())
            });
            eprintln!(
                "BW host_alloc_uninit {mib:>4} MiB (+ free)   {:8.2} ms",
                t * 1e3
            );
            let al = g.host_allocator();
            let t = time(&g, n, || {
                let a = al.alloc(mib << 20).unwrap();
                // SAFETY: ours, never adopted, no copy.
                unsafe { al.free(a) };
            });
            eprintln!(
                "BW HostAllocator {mib:>4} MiB (+ free)      {:8.2} ms",
                t * 1e3
            );
        }
        let page = 64 * row * arrays;
        let s = time(&g, 31, || {
            host.copy_2d_from(0, 64 * row, &dev, 0, ctx * row, (64 * row, arrays))
                .unwrap()
        });
        eprintln!(
            "BW D2H one 64-pos page ({:.2} MiB)       {:8.3} ms {:7.2} GB/s",
            page as f64 / (1u64 << 20) as f64,
            s * 1e3,
            page as f64 / s / 1e9
        );
        let s = time(&g, 31, || {
            host.copy_2d_to(0, 64 * row, &dev, 0, ctx * row, (64 * row, arrays))
                .unwrap()
        });
        eprintln!(
            "BW H2D one 64-pos page                  {:8.3} ms {:7.2} GB/s",
            s * 1e3,
            page as f64 / s / 1e9
        );
        // The file path's staging, to and from RAM (pre-touched: no page
        // faults) and, with AXON_BW_FILE, a file.
        let mut img = vec![1u8; bytes];
        let s = time(&g, 3, || rows.save(&g, 0, ctx, &mut &mut img[..]).unwrap());
        report("save() to RAM (staged)", s);
        let s = time(&g, 3, || rows.load(&g, 0, ctx, &mut &img[..]).unwrap());
        report("load() from RAM (staged)", s);
        drop(img);
        if let Ok(f) = std::env::var("AXON_BW_FILE") {
            let t0 = Instant::now();
            let mut w = std::io::BufWriter::new(std::fs::File::create(&f).unwrap());
            rows.save(&g, 0, ctx, &mut w).unwrap();
            w.into_inner().unwrap().sync_all().unwrap();
            report("save() to file + fsync", t0.elapsed().as_secs_f64());
            let s = time(&g, 3, || {
                let mut r = std::io::BufReader::new(std::fs::File::open(&f).unwrap());
                rows.load(&g, 0, ctx, &mut r).unwrap()
            });
            report("load() from file (page cache)", s);
            let _ = std::fs::remove_file(&f);
        }
    }
}
