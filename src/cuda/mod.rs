//! CUDA backend plumbing: the driver API and NVRTC loaded at run time
//! (ffi.rs), a device with its context and stream (`Gpu`), device buffers
//! (`DevBuf`), kernels compiled by NVRTC for the device and cached on disk
//! (`Module`), the kernels in kernels.cu and decode.cu (`Kernels`), and the
//! Qwen2 decoder running them (qwen2.rs).
//!
//! No link-time CUDA dependency: without an NVIDIA driver `Gpu::open`
//! returns Err and everything else in axon works as before.
//!
//! Thread model: one context per device (the primary one, shared by every
//! `Gpu` on it), made current on the calling thread by every operation
//! (`bind`), and one stream per `Gpu`. `Gpu` holds raw driver handles, so it
//! is !Send and !Sync, and everything on it (buffers, modules, graphs, the
//! GPU model) borrows it: all of it stays on the thread that opened it, by
//! construction. That is the invariant; making `Gpu` Send would be sound
//! (bind on every entry point already makes no thread-affinity assumption)
//! but nothing needs it: generation and chat run on one thread.
//!
//! Streams: the stream is non-blocking (unordered with the legacy default
//! stream), as graph capture requires, so every copy and memset is enqueued
//! on it; the synchronous `DevBuf` calls wait for the stream afterwards.

mod ffi;
pub mod qwen2;
pub mod upload;

use ffi::{CUcontext, CUdevice, CUdeviceptr, CUfunction, CUmodule, CUstream, Driver, check};
use std::ffi::{CString, OsString, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Orders stream captures against `Pinned`'s drop. Its cuCtxSynchronize
/// on any thread invalidates a capture in progress on another thread's
/// stream, THREAD_LOCAL mode or not: the capture's next launch fails with
/// CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, and the sync itself fails
/// (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED), so the memory was unpinned
/// without waiting for copies from it. Measured on sm_89: 67-76 of 300
/// captures broken; cuMemHostUnregister alone, cuMemHostRegister,
/// allocation, module loads and stream syncs elsewhere broke none
/// (`gpu_capture_survives_pinning_elsewhere`). So a capture holds this
/// shared and the drop holds it exclusively: it waits for the captures in
/// flight (each a few hundred launches), and captures wait for it.
static CAPTURE_GATE: RwLock<()> = RwLock::new(());

/// `CAPTURE_GATE` shared (a capture). Poison is ignored: it guards no data.
fn capture_gate() -> RwLockReadGuard<'static, ()> {
    CAPTURE_GATE.read().unwrap_or_else(|e| e.into_inner())
}

/// `CAPTURE_GATE` exclusive (no capture runs on any thread).
fn no_capture() -> RwLockWriteGuard<'static, ()> {
    CAPTURE_GATE.write().unwrap_or_else(|e| e.into_inner())
}

/// Why `Gpu::open` fails off Linux.
pub const LINUX_ONLY: &str = "CUDA support is Linux-only for now";

/// kernels.cu, decode.cu, prefill.cu, stream.cu, sample.cu, hopper.cu and gemma.cu, compiled
/// at run time as one module. hopper.cu is empty except for the sm_90a target (`Target::Cubin90a`).
const KERNELS: &str = concat!(
    include_str!("kernels.cu"),
    include_str!("decode.cu"),
    include_str!("prefill.cu"),
    include_str!("stream.cu"),
    include_str!("sample.cu"),
    include_str!("hopper.cu"),
    include_str!("gemma.cu")
);

/// A CUDA device: its retained primary context and a stream to run on.
/// Dropping it destroys the stream and releases the context; `DevBuf`s and
/// `Module`s borrow it, so they're gone first.
pub struct Gpu {
    d: &'static Driver,
    dev: CUdevice,
    ctx: CUcontext,
    stream: CUstream,
}

/// What `Gpu::info` reports.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    /// Compute capability (major, minor): sm_{major}{minor}.
    pub cc: (u32, u32),
    pub sms: u32,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub bus_bits: u32,
    /// Peak memory clock in kHz (0 if the driver doesn't say).
    pub mem_clock_khz: u32,
}

impl DeviceInfo {
    /// The NVRTC architecture name, e.g. "sm_120".
    pub fn arch(&self) -> String {
        format!("sm_{}{}", self.cc.0, self.cc.1)
    }

    /// Theoretical memory bandwidth in GB/s: two transfers per clock (DDR)
    /// over the whole bus. None if the driver reports no clock or width.
    pub fn bandwidth_gbs(&self) -> Option<f64> {
        (self.mem_clock_khz > 0 && self.bus_bits > 0)
            .then(|| 2.0 * self.mem_clock_khz as f64 * 1e3 * (self.bus_bits / 8) as f64 / 1e9)
    }
}

/// The threads `preinit` started, joined by `settle`.
static PREINIT: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>> =
    std::sync::Mutex::new(Vec::new());

/// Start the driver's init (cuInit, device 0's primary context: ~0.2 s)
/// and NVRTC's load on background threads, so they overlap what the caller
/// does before `Gpu::open` (reading the GGUF, building the tokenizer). Best
/// effort: an error is left for `Gpu::open` to report (`ffi::driver` keeps
/// it). The context's retain is never released: the process keeps the
/// primary context it's about to use anyway. Call `settle` before exiting
/// without opening the GPU (exit mustn't race a thread inside the driver).
/// Tests: the next `n` graph instantiations fail as out of memory
/// (`inject_graph_oom`), so the fallback path runs on any host without
/// starving the GPU.
#[cfg(test)]
static INJECT_GRAPH_OOM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn inject_graph_oom(n: usize) {
    INJECT_GRAPH_OOM.store(n, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
fn take_injected_graph_oom() -> bool {
    use std::sync::atomic::Ordering::SeqCst;
    INJECT_GRAPH_OOM
        .fetch_update(SeqCst, SeqCst, |n| n.checked_sub(1))
        .is_ok()
}

pub fn preinit() {
    if !cfg!(target_os = "linux") {
        return;
    }
    let mut ts = PREINIT.lock().unwrap();
    ts.push(std::thread::spawn(|| {
        let Ok(d) = ffi::driver() else { return };
        let mut dev: CUdevice = 0;
        // SAFETY: writes a device handle to a valid CUdevice.
        if unsafe { (d.cuDeviceGet)(&mut dev, 0) } != 0 {
            return;
        }
        let mut ctx: CUcontext = std::ptr::null_mut();
        // SAFETY: dev is a valid device; writes the context handle.
        let _ = unsafe { (d.cuDevicePrimaryCtxRetain)(&mut ctx, dev) };
    }));
    // NVRTC's soname depends on the driver's version: this waits for cuInit
    // and then loads the library while the context is being made.
    ts.push(std::thread::spawn(|| {
        let _ = ffi::nvrtc();
    }));
}

/// Wait for `preinit`'s threads (none: returns at once; never for the
/// calling thread itself).
pub fn settle() {
    let ts = match PREINIT.lock() {
        Ok(mut g) => std::mem::take(&mut *g),
        Err(_) => return,
    };
    let me = std::thread::current().id();
    for t in ts {
        if t.thread().id() != me {
            let _ = t.join();
        }
    }
}

impl Gpu {
    /// Device 0.
    pub fn open() -> Result<Gpu, String> {
        Self::open_device(0)
    }

    /// Device `ordinal`: Err with the reason if there's no driver, no such
    /// device, or its context can't be had.
    pub fn open_device(ordinal: usize) -> Result<Gpu, String> {
        if !cfg!(target_os = "linux") {
            return Err(LINUX_ONLY.into());
        }
        // The driver would make us wait for `preinit`'s work anyway.
        settle();
        let d = ffi::driver()?;
        let mut count: c_int = 0;
        // SAFETY: writes the device count to a valid c_int.
        let r = unsafe { (d.cuDeviceGetCount)(&mut count) };
        check(d, r, "cuDeviceGetCount")?;
        if count <= 0 {
            return Err("no CUDA device".into());
        }
        if ordinal >= count as usize {
            return Err(format!("no CUDA device {ordinal}: there are {count}"));
        }
        let mut dev: CUdevice = 0;
        // SAFETY: ordinal < count; writes a device handle to a valid CUdevice.
        let r = unsafe { (d.cuDeviceGet)(&mut dev, ordinal as c_int) };
        check(d, r, "cuDeviceGet")?;
        let mut ctx: CUcontext = std::ptr::null_mut();
        // SAFETY: dev is a valid device; writes the context handle.
        let r = unsafe { (d.cuDevicePrimaryCtxRetain)(&mut ctx, dev) };
        check(d, r, "cuDevicePrimaryCtxRetain")?;
        // From here on Drop undoes what's been done; stream is set below.
        let mut gpu = Gpu {
            d,
            dev,
            ctx,
            stream: std::ptr::null_mut(),
        };
        gpu.bind()?;
        let mut stream: CUstream = std::ptr::null_mut();
        // A non-blocking stream, as graph capture needs: nothing runs on the
        // legacy default stream. Every copy and memset (DevBuf::write/read/
        // zero, HostBuf copies) is enqueued on this stream, so it's ordered
        // with the kernels launched here by stream order alone.
        // SAFETY: the context is current; writes the stream handle.
        let r = unsafe { (d.cuStreamCreate)(&mut stream, ffi::CU_STREAM_NON_BLOCKING) };
        check(d, r, "cuStreamCreate")?;
        gpu.stream = stream;
        Ok(gpu)
    }

    /// Make this device's context current on the calling thread.
    fn bind(&self) -> Result<(), String> {
        // SAFETY: ctx is the retained primary context, live until Drop.
        let r = unsafe { (self.d.cuCtxSetCurrent)(self.ctx) };
        check(self.d, r, "cuCtxSetCurrent")
    }

    pub fn info(&self) -> Result<DeviceInfo, String> {
        let d = self.d;
        let attr = |a: c_int, what: &str| -> Result<u32, String> {
            let mut v: c_int = 0;
            // SAFETY: dev is valid; writes one c_int.
            let r = unsafe { (d.cuDeviceGetAttribute)(&mut v, a, self.dev) };
            check(d, r, what)?;
            Ok(v.max(0) as u32)
        };
        let mut name = [0u8; 256];
        // SAFETY: writes at most name.len() bytes, NUL-terminated.
        let r =
            unsafe { (d.cuDeviceGetName)(name.as_mut_ptr().cast(), name.len() as c_int, self.dev) };
        check(d, r, "cuDeviceGetName")?;
        let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
        let mut total = 0usize;
        // SAFETY: writes one size_t.
        let r = unsafe { (d.cuDeviceTotalMem_v2)(&mut total, self.dev) };
        check(d, r, "cuDeviceTotalMem_v2")?;
        self.bind()?;
        let (mut free, mut ctx_total) = (0usize, 0usize);
        // SAFETY: our context is current; writes two size_t.
        let r = unsafe { (d.cuMemGetInfo_v2)(&mut free, &mut ctx_total) };
        check(d, r, "cuMemGetInfo_v2")?;
        Ok(DeviceInfo {
            name: String::from_utf8_lossy(&name[..end]).into_owned(),
            cc: (
                attr(ffi::ATTR_CC_MAJOR, "cuDeviceGetAttribute(cc major)")?,
                attr(ffi::ATTR_CC_MINOR, "cuDeviceGetAttribute(cc minor)")?,
            ),
            sms: attr(ffi::ATTR_SM_COUNT, "cuDeviceGetAttribute(SM count)")?,
            total_bytes: total as u64,
            free_bytes: free as u64,
            bus_bits: attr(ffi::ATTR_MEMORY_BUS_BITS, "cuDeviceGetAttribute(bus width)")?,
            mem_clock_khz: attr(
                ffi::ATTR_MEMORY_CLOCK_KHZ,
                "cuDeviceGetAttribute(memory clock)",
            )?,
        })
    }

    /// `len` bytes of uninitialized device memory.
    pub fn alloc(&self, len: usize) -> Result<DevBuf<'_>, String> {
        let mut ptr: CUdeviceptr = 0;
        if len > 0 {
            self.bind()?;
            // SAFETY: context current; writes the device address.
            let r = unsafe { (self.d.cuMemAlloc_v2)(&mut ptr, len) };
            check(self.d, r, "cuMemAlloc_v2")?;
        }
        Ok(DevBuf {
            gpu: self,
            ptr,
            len,
        })
    }

    /// A device copy of `data`.
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<DevBuf<'_>, String> {
        let b = self.alloc(size_of_val(data))?;
        b.write(data)?;
        Ok(b)
    }

    /// Wait for everything launched on the stream; Err if a kernel failed.
    pub fn sync(&self) -> Result<(), String> {
        self.bind()?;
        // SAFETY: stream is live until Drop.
        let r = unsafe { (self.d.cuStreamSynchronize)(self.stream) };
        check(self.d, r, "cuStreamSynchronize")
    }

    /// Launch `f` on the stream with a grid of `grid` blocks of `block`
    /// threads and `shared` bytes of dynamic shared memory. Asynchronous:
    /// faults show up at the next `sync` or copy.
    ///
    /// Err if `f` belongs to another `Gpu`'s module. (Its module is loaded:
    /// `f` borrows it.)
    ///
    /// # Safety
    /// `args` match the kernel's parameter list in number and type; every
    /// buffer the kernel touches is live until the launch completes and
    /// large enough for what it accesses with these arguments.
    pub unsafe fn launch(
        &self,
        f: &Func,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        args: &[Arg],
    ) -> Result<(), String> {
        if !std::ptr::eq(f.module.gpu, self) {
            return Err(format!(
                "launch of {}: its module is on another Gpu",
                f.name
            ));
        }
        // SAFETY: the caller's contract, which is launch_with's.
        unsafe { self.launch_with(f, grid, block, shared, args, false) }
    }

    /// `launch`, and with `overlap` allowed to start before the previous
    /// kernel on the stream finishes: programmatic dependent launch, if
    /// `f`'s module supports it (`Module::pdl`; otherwise `overlap` is
    /// ignored). The kernel must then execute `griddepcontrol.wait` (decode.cu
    /// `pdl_wait`) before touching memory the previous kernel reads or writes.
    ///
    /// # Safety
    /// As `launch`; with `overlap`, the kernel waits as described.
    pub unsafe fn launch_with(
        &self,
        f: &Func,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        args: &[Arg],
        overlap: bool,
    ) -> Result<(), String> {
        if !std::ptr::eq(f.module.gpu, self) {
            return Err(format!(
                "launch of {}: its module is on another Gpu",
                f.name
            ));
        }
        // SAFETY: the caller's contract; our stream.
        unsafe { self.launch_raw(self.stream, f, grid, block, shared, args, overlap) }
    }

    /// `launch` on another stream of this `Gpu` (`Gpu::new_stream`).
    ///
    /// # Safety
    /// As `launch`, the buffers live until the work on `s` completes.
    pub unsafe fn launch_on(
        &self,
        s: &Stream,
        f: &Func,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        args: &[Arg],
    ) -> Result<(), String> {
        if !std::ptr::eq(f.module.gpu, self) || !std::ptr::eq(s.gpu, self) {
            return Err(format!("launch of {}: another Gpu's", f.name));
        }
        // SAFETY: the caller's contract; s is ours and live (borrowed).
        unsafe { self.launch_raw(s.s, f, grid, block, shared, args, false) }
    }

    /// `launch_with` on `stream` (ours).
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_raw(
        &self,
        stream: CUstream,
        f: &Func,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        args: &[Arg],
        overlap: bool,
    ) -> Result<(), String> {
        self.bind()?;
        // Debug builds: the kernel takes exactly args.len() parameters (a
        // missing one is read from past the array: undefined behaviour,
        // not an error). Needs the driver's cuFuncGetParamInfo (12.4+).
        // A query that fails for another reason than "no such parameter"
        // (CUDA_ERROR_INVALID_VALUE), e.g. a context an earlier kernel left
        // in CUDA_ERROR_ILLEGAL_ADDRESS, is that error, not a count mismatch.
        #[cfg(debug_assertions)]
        if let Some(info) = ffi::param_info() {
            const INVALID_VALUE: ffi::CUresult = 1;
            let (mut off, mut size) = (0usize, 0usize);
            let n = args.len();
            // SAFETY: f is a loaded kernel; off and size are written.
            let mut query = |i: usize| unsafe { info(f.f, i, &mut off, &mut size) };
            let last = if n == 0 {
                ffi::CUDA_SUCCESS
            } else {
                query(n - 1)
            };
            let past = query(n);
            for r in [last, past] {
                if r != ffi::CUDA_SUCCESS && r != INVALID_VALUE {
                    check(self.d, r, &format!("cuFuncGetParamInfo ({})", f.name))?;
                }
            }
            assert!(
                last == ffi::CUDA_SUCCESS && past == INVALID_VALUE,
                "launch of {} with {n} arguments: not its parameter count",
                f.name
            );
            // And each argument's size is its parameter's: a U32 where the
            // kernel takes a pointer has the driver copy 8 bytes from a
            // 4-byte value, so the pointer's high half is garbage. Each
            // parameter is copied from its own address, so only that one
            // is wrong. Sizes only: cuFuncGetParamInfo reports no types, so
            // u32 / i32 / f32, and Ptr / U64, can't be told apart.
            for (i, a) in args.iter().enumerate() {
                // SAFETY: as above; i < the parameter count (checked).
                let r = unsafe { info(f.f, i, &mut off, &mut size) };
                check(
                    self.d,
                    r,
                    &format!("cuFuncGetParamInfo {} param {i}", f.name),
                )?;
                assert!(
                    size == a.size(),
                    "launch of {}: argument {i} is {a:?}, {} bytes; the parameter has {size}",
                    f.name,
                    a.size()
                );
            }
        }
        // cuLaunchKernelEx copies each parameter from these addresses before returning.
        let mut ptrs: Vec<*mut c_void> = args.iter().map(Arg::addr).collect();
        let mut attr = ffi::LaunchAttr {
            id: ffi::LAUNCH_ATTR_PROGRAMMATIC,
            pad: 0,
            value: [1, 0, 0, 0, 0, 0, 0, 0],
        };
        let pdl = overlap && f.module.pdl;
        let cfg = ffi::LaunchConfig {
            grid: [grid.0, grid.1, grid.2],
            block: [block.0, block.1, block.2],
            shared,
            stream,
            attrs: if pdl { &mut attr } else { std::ptr::null_mut() },
            n_attrs: pdl as u32,
        };
        // SAFETY: f's module is loaded on this context (f borrows it, checked
        // above) and args match its signature (caller); ptrs points to args'
        // live values; cfg and attr are live for the call; stream is ours.
        let r = unsafe {
            (self.d.cuLaunchKernelEx)(&cfg, f.f, ptrs.as_mut_ptr(), std::ptr::null_mut())
        };
        check(self.d, r, &format!("cuLaunchKernelEx {}", f.name))
    }
}

impl Gpu {
    /// A TMA descriptor (CUtensorMap, for `Arg::Blob128`) of a 2-D byte
    /// tensor at device address `base`: `dims` = (bytes per row, rows),
    /// rows `stride` bytes apart, copied in boxes of `boxd` = (bytes, rows),
    /// with the 128-byte smem swizzle or none. Boxes past the tensor's edge
    /// read zeros. Err without cuTensorMapEncodeTiled (driver older than
    /// CUDA 12.0) or if the driver rejects the shape (`base` must be 16-byte
    /// aligned, `stride` a multiple of 16, box sides <= 256, the box row a
    /// multiple of 16 bytes and <= 128 with the swizzle).
    pub fn tensor_map_2d(
        &self,
        base: u64,
        dims: (u64, u64),
        stride: u64,
        boxd: (u32, u32),
        swizzle128: bool,
    ) -> Result<[u64; 16], String> {
        #[repr(C, align(64))]
        struct Map([u64; 16]);
        let enc = ffi::tensor_map_encoder().ok_or("the driver has no cuTensorMapEncodeTiled")?;
        let mut m = Map([0; 16]);
        let gdim = [dims.0, dims.1];
        let gstride = [stride];
        let bdim = [boxd.0, boxd.1];
        let estride = [1u32, 1];
        self.bind()?;
        // SAFETY: m is 128 bytes, 64-byte aligned (as CUtensorMap); the
        // arrays have rank (2) entries, strides rank - 1. The driver only
        // encodes base into the map: nothing is accessed until a kernel
        // uses it.
        let r = unsafe {
            enc(
                &mut m.0,
                ffi::TENSOR_MAP_DATA_TYPE_UINT8,
                2,
                base as *mut c_void,
                gdim.as_ptr(),
                gstride.as_ptr(),
                bdim.as_ptr(),
                estride.as_ptr(),
                0, // CU_TENSOR_MAP_INTERLEAVE_NONE
                if swizzle128 {
                    ffi::TENSOR_MAP_SWIZZLE_128B
                } else {
                    ffi::TENSOR_MAP_SWIZZLE_NONE
                },
                ffi::TENSOR_MAP_L2_PROMOTION_256B,
                0, // CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE (zeros)
            )
        };
        check(
            self.d,
            r,
            &format!("cuTensorMapEncodeTiled dims {dims:?} stride {stride} box {boxd:?}"),
        )?;
        Ok(m.0)
    }

    /// `len` bytes of page-locked host memory, for stream-ordered copies.
    pub fn host_alloc(&self, len: usize) -> Result<HostBuf<'_>, String> {
        let mut p: *mut c_void = std::ptr::null_mut();
        if len > 0 {
            self.bind()?;
            // SAFETY: context current; writes the host address. Flags 0:
            // portable to this context only, which is all we use.
            let r = unsafe { (self.d.cuMemHostAlloc)(&mut p, len, 0) };
            check(self.d, r, "cuMemHostAlloc")?;
            // SAFETY: p is a fresh allocation of len bytes.
            unsafe { std::ptr::write_bytes(p.cast::<u8>(), 0, len) };
        }
        Ok(HostBuf {
            gpu: self,
            p: p.cast(),
            len,
            pending: std::cell::Cell::new(false),
        })
    }

    /// Record everything `f` launches on the stream into a graph instead of
    /// running it. `f` must only launch kernels and make stream-ordered
    /// async copies (no sync, no alloc/free): those would end the capture
    /// with an error. The capture is ended on every path out.
    pub fn capture(&self, f: impl FnOnce() -> Result<(), String>) -> Result<Graph<'_>, String> {
        /// Ends the capture (and frees its partial graph) if `f` panics or
        /// returns early, so the stream is usable again.
        struct Capturing<'a>(&'a Gpu, bool);
        impl Capturing<'_> {
            /// End the capture; the graph (null if it was invalidated).
            fn end(&mut self) -> (ffi::CUresult, ffi::CUgraph) {
                self.1 = false;
                let mut g: ffi::CUgraph = std::ptr::null_mut();
                // SAFETY: ends the capture begun on our stream; writes the graph.
                let r = unsafe { (self.0.d.cuStreamEndCapture)(self.0.stream, &mut g) };
                (r, g)
            }
        }
        impl Drop for Capturing<'_> {
            fn drop(&mut self) {
                if self.1 && self.0.bind().is_ok() {
                    let (_, g) = self.end();
                    if !g.is_null() {
                        // SAFETY: the partial graph, never instantiated.
                        unsafe { (self.0.d.cuGraphDestroy)(g) };
                    }
                }
            }
        }
        self.bind()?;
        // Held until the capture has ended (early exits: `cap` drops first).
        let gate = capture_gate();
        // SAFETY: our stream, not capturing (captures don't nest here: the
        // stream is only captured inside this function, and a capture that
        // didn't finish was ended by `Capturing`).
        let r = unsafe {
            (self.d.cuStreamBeginCapture_v2)(self.stream, ffi::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        };
        check(self.d, r, "cuStreamBeginCapture")?;
        let mut cap = Capturing(self, true);
        f()?; // early return / panic: `cap` ends the capture
        let (r, g) = cap.end();
        drop(gate);
        check(self.d, r, "cuStreamEndCapture")?;
        let mut exec: ffi::CUgraphExec = std::ptr::null_mut();
        #[cfg(test)]
        if take_injected_graph_oom() {
            // SAFETY: the captured graph, never instantiated.
            unsafe { (self.d.cuGraphDestroy)(g) };
            return Err(
                "cuGraphInstantiateWithFlags: CUDA_ERROR_OUT_OF_MEMORY: injected (test)".into(),
            );
        }
        // SAFETY: g is the captured graph; writes the executable graph.
        let r = unsafe { (self.d.cuGraphInstantiateWithFlags)(&mut exec, g, 0) };
        // SAFETY: the graph itself isn't needed once instantiated (or failed).
        unsafe { (self.d.cuGraphDestroy)(g) };
        check(self.d, r, "cuGraphInstantiateWithFlags")?;
        Ok(Graph { gpu: self, exec })
    }

    /// A timing event recorded on the stream now.
    pub fn record(&self) -> Result<Event<'_>, String> {
        self.bind()?;
        let mut e: ffi::CUevent = std::ptr::null_mut();
        // SAFETY: context current; writes the event. Flags 0: timing on.
        let r = unsafe { (self.d.cuEventCreate)(&mut e, 0) };
        check(self.d, r, "cuEventCreate")?;
        let ev = Event { gpu: self, e };
        // SAFETY: our event and stream.
        let r = unsafe { (self.d.cuEventRecord)(ev.e, self.stream) };
        check(self.d, r, "cuEventRecord")?;
        Ok(ev)
    }
}

impl Gpu {
    /// Another non-blocking stream on this device: work on it runs
    /// concurrently with the main stream's, ordered by `SyncEvent`s only.
    pub fn new_stream(&self) -> Result<Stream<'_>, String> {
        self.bind()?;
        let mut s: CUstream = std::ptr::null_mut();
        // SAFETY: context current; writes the stream handle.
        let r = unsafe { (self.d.cuStreamCreate)(&mut s, ffi::CU_STREAM_NON_BLOCKING) };
        check(self.d, r, "cuStreamCreate")?;
        Ok(Stream { gpu: self, s })
    }

    /// An event for ordering streams (no timing).
    pub fn new_event(&self) -> Result<SyncEvent<'_>, String> {
        self.bind()?;
        let mut e: ffi::CUevent = std::ptr::null_mut();
        // SAFETY: context current; writes the event.
        let r = unsafe { (self.d.cuEventCreate)(&mut e, ffi::CU_EVENT_DISABLE_TIMING) };
        check(self.d, r, "cuEventCreate")?;
        Ok(SyncEvent { gpu: self, e })
    }

    /// Record `e` on the main stream: it fires once what's enqueued so far ran.
    pub fn record_main(&self, e: &SyncEvent) -> Result<(), String> {
        self.bind()?;
        // SAFETY: our event and stream.
        let r = unsafe { (self.d.cuEventRecord)(e.e, self.stream) };
        check(self.d, r, "cuEventRecord")
    }

    /// Make the main stream's later work wait for `e`'s last record.
    pub fn wait_main(&self, e: &SyncEvent) -> Result<(), String> {
        self.bind()?;
        // SAFETY: our event and stream.
        let r = unsafe { (self.d.cuStreamWaitEvent)(self.stream, e.e, 0) };
        check(self.d, r, "cuStreamWaitEvent")
    }

    /// Page-lock `len` bytes of host memory at `p` for DMA at full speed
    /// (cuMemHostRegister); unlocked when the `Pinned` drops. `read_only`
    /// for memory mapped read-only (the GPU only reads it then).
    ///
    /// # Safety
    /// p..p + len is mapped and stays mapped while the `Pinned` lives.
    pub unsafe fn register(
        &self,
        p: *const u8,
        len: usize,
        read_only: bool,
    ) -> Result<Pinned<'_>, String> {
        self.bind()?;
        let flags = if read_only {
            ffi::CU_MEMHOSTREGISTER_READ_ONLY
        } else {
            0
        };
        // SAFETY: the caller's; no device mapping, this context.
        let r = unsafe { (self.d.cuMemHostRegister_v2)(p as *mut c_void, len, flags) };
        check(self.d, r, "cuMemHostRegister")?;
        Ok(Pinned { gpu: self, p })
    }
}

/// Several devices on one driver thread (pipeline parallel,
/// `qwen2::pipeline`). Each `Gpu` keeps its own stream; work is ordered
/// across them with `SyncEvent`s (`record_main` on one, `wait_main` on
/// another: `cuStreamWaitEvent` takes another context's event).
impl Gpu {
    /// The device's ordinal (as `open_device` took it).
    pub fn ordinal(&self) -> usize {
        self.dev as usize
    }

    /// Whether `other` is the same device (`open_device` twice: the same
    /// primary context, so a copy between them is a device-to-device copy).
    pub fn same_device(&self, other: &Gpu) -> bool {
        self.dev == other.dev
    }

    /// Whether AXON_P2P=1 asks for peer access between devices
    /// (`enable_peer`). Off by default: the P2P path has not run on hardware
    /// that has it yet (the 2x4090 test host had none); the host-staged copy
    /// is the tested one.
    pub fn p2p_setting() -> bool {
        std::env::var("AXON_P2P").is_ok_and(|v| v == "1")
    }

    /// With AXON_P2P=1 (`p2p_setting`), let this device's kernels and copies
    /// read and write `other`'s memory directly (P2P over PCIe or NVLink).
    /// True if on (or the same device), false if not asked for, the devices
    /// can't, or the driver refuses (a note on stderr): copies are then
    /// staged through host memory by the driver (`copy_peer` works either
    /// way). Never an Err: a refused peer mapping only costs speed.
    pub fn enable_peer(&self, other: &Gpu) -> bool {
        if self.same_device(other) {
            return true;
        }
        if !Self::p2p_setting() {
            return false;
        }
        let (a, b) = (self.ordinal(), other.ordinal());
        let refused = |what: String| {
            eprintln!(
                "axon: note: no P2P from GPU {a} to GPU {b} ({what}); copies go through host memory"
            );
            false
        };
        let mut can: c_int = 0;
        // SAFETY: both devices are valid; writes one c_int.
        let r = unsafe { (self.d.cuDeviceCanAccessPeer)(&mut can, self.dev, other.dev) };
        if let Err(e) = check(self.d, r, "cuDeviceCanAccessPeer") {
            return refused(e);
        }
        if can == 0 {
            return refused("the devices can't".into());
        }
        if let Err(e) = self.bind() {
            return refused(e);
        }
        // SAFETY: our context is current; other.ctx is a live context.
        let r = unsafe { (self.d.cuCtxEnablePeerAccess)(other.ctx, 0) };
        if r == ffi::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            return true;
        }
        match check(self.d, r, "cuCtxEnablePeerAccess") {
            Ok(()) => true,
            Err(e) => refused(e),
        }
    }

    /// Copy `n` bytes from `src` at `src_off` (on any device) to `dst` at
    /// `dst_off` (on any device), enqueued on this `Gpu`'s stream: ordered
    /// after the work enqueued here, not waited for. The caller orders it
    /// with the other device's stream (`record_main` / `wait_main`).
    pub fn copy_peer(
        &self,
        dst: &DevBuf,
        dst_off: usize,
        src: &DevBuf,
        src_off: usize,
        n: usize,
    ) -> Result<(), String> {
        let ok = |o: usize, len: usize| o.checked_add(n).is_some_and(|end| end <= len);
        if !ok(src_off, src.len) || !ok(dst_off, dst.len) {
            return Err(format!(
                "peer copy of {n} bytes from {src_off} of {} to {dst_off} of {}",
                src.len, dst.len
            ));
        }
        if n == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: n bytes in bounds of both allocations (checked), each in
        // its own Gpu's live context; they outlive the stream work
        // (cuMemFree waits for it).
        let r = unsafe {
            (self.d.cuMemcpyPeerAsync)(
                dst.ptr + dst_off as u64,
                dst.gpu.ctx,
                src.ptr + src_off as u64,
                src.gpu.ctx,
                n,
                self.stream,
            )
        };
        check(self.d, r, "cuMemcpyPeerAsync")
    }
}

/// Host memory page-locked by `Gpu::register`.
pub struct Pinned<'g> {
    gpu: &'g Gpu,
    p: *const u8,
}

impl Drop for Pinned<'_> {
    fn drop(&mut self) {
        if self.gpu.bind().is_ok() {
            // Both calls below would break another thread's capture (and
            // the sync would fail, unpinning under a live copy).
            let _gate = no_capture();
            // A copy on any stream may still use it (e.g. a streamed
            // prefill's KV write-back, if a panic unwinds past the pass):
            // wait for the whole context, not just the main stream.
            // SAFETY: our context is current (bind); a plain wait.
            unsafe { (self.gpu.d.cuCtxSynchronize)() };
            // SAFETY: registered by us, unregistered once.
            unsafe { (self.gpu.d.cuMemHostUnregister)(self.p as *mut c_void) };
        }
    }
}

/// An extra stream (`Gpu::new_stream`), destroyed on drop after its work.
pub struct Stream<'g> {
    gpu: &'g Gpu,
    s: CUstream,
}

impl Stream<'_> {
    /// Record `e` here.
    pub fn record(&self, e: &SyncEvent) -> Result<(), String> {
        self.gpu.bind()?;
        // SAFETY: our event and stream.
        let r = unsafe { (self.gpu.d.cuEventRecord)(e.e, self.s) };
        check(self.gpu.d, r, "cuEventRecord")
    }

    /// Make this stream's later work wait for `e`'s last record.
    pub fn wait(&self, e: &SyncEvent) -> Result<(), String> {
        self.gpu.bind()?;
        // SAFETY: our event and stream.
        let r = unsafe { (self.gpu.d.cuStreamWaitEvent)(self.s, e.e, 0) };
        check(self.gpu.d, r, "cuStreamWaitEvent")
    }

    /// Wait for everything enqueued here.
    pub fn sync(&self) -> Result<(), String> {
        self.gpu.bind()?;
        // SAFETY: our stream.
        let r = unsafe { (self.gpu.d.cuStreamSynchronize)(self.s) };
        check(self.gpu.d, r, "cuStreamSynchronize")
    }

    /// Enqueue a copy of `n` host bytes at `src` to `dst` + `off`.
    ///
    /// # Safety
    /// src..src + n stays readable and unchanged until the copy ran (it is
    /// asynchronous only from page-locked memory; from pageable memory the
    /// driver stages it before returning).
    pub unsafe fn h2d(
        &self,
        dst: &DevBuf,
        off: usize,
        src: *const u8,
        n: usize,
    ) -> Result<(), String> {
        if off.checked_add(n).is_none_or(|e| e > dst.len) {
            return Err(format!("copy of {n} bytes at {off}: out of bounds"));
        }
        self.gpu.bind()?;
        // SAFETY: in bounds on the device (checked); the host side is the caller's.
        let r = unsafe {
            (self.gpu.d.cuMemcpyHtoDAsync_v2)(dst.ptr + off as u64, src.cast(), n, self.s)
        };
        check(self.gpu.d, r, "cuMemcpyHtoDAsync_v2")
    }

    /// Enqueue a copy of `n` bytes from `src` + `off` to host memory at `dst`.
    ///
    /// # Safety
    /// dst..dst + n stays writable and untouched by the host until the copy ran.
    pub unsafe fn d2h(
        &self,
        dst: *mut u8,
        src: &DevBuf,
        off: usize,
        n: usize,
    ) -> Result<(), String> {
        if off.checked_add(n).is_none_or(|e| e > src.len) {
            return Err(format!("copy of {n} bytes at {off}: out of bounds"));
        }
        self.gpu.bind()?;
        // SAFETY: in bounds on the device (checked); the host side is the caller's.
        let r = unsafe {
            (self.gpu.d.cuMemcpyDtoHAsync_v2)(dst.cast(), src.ptr + off as u64, n, self.s)
        };
        check(self.gpu.d, r, "cuMemcpyDtoHAsync_v2")
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        if self.gpu.bind().is_ok() {
            // SAFETY: our stream, destroyed once after its work.
            unsafe {
                (self.gpu.d.cuStreamSynchronize)(self.s);
                (self.gpu.d.cuStreamDestroy_v2)(self.s);
            }
        }
    }
}

/// An event ordering work across streams (`Gpu::new_event`).
pub struct SyncEvent<'g> {
    gpu: &'g Gpu,
    e: ffi::CUevent,
}

impl SyncEvent<'_> {
    /// Block the host until the last record fired.
    pub fn sync(&self) -> Result<(), String> {
        self.gpu.bind()?;
        // SAFETY: our event.
        let r = unsafe { (self.gpu.d.cuEventSynchronize)(self.e) };
        check(self.gpu.d, r, "cuEventSynchronize")
    }
}

impl Drop for SyncEvent<'_> {
    fn drop(&mut self) {
        if self.gpu.bind().is_ok() {
            // SAFETY: our event, destroyed once (pending records complete
            // asynchronously: the driver allows destroying a recorded event).
            unsafe { (self.gpu.d.cuEventDestroy_v2)(self.e) };
        }
    }
}

/// A recorded CUDA event (see `Gpu::record`).
pub struct Event<'g> {
    gpu: &'g Gpu,
    e: ffi::CUevent,
}

impl Event<'_> {
    /// Milliseconds from `self` to `later`, once `later` has happened.
    pub fn ms_to(&self, later: &Event) -> Result<f32, String> {
        let d = self.gpu.d;
        self.gpu.bind()?;
        // SAFETY: both events are live and recorded.
        let r = unsafe { (d.cuEventSynchronize)(later.e) };
        check(d, r, "cuEventSynchronize")?;
        let mut ms = 0f32;
        // SAFETY: as above; writes one float.
        let r = unsafe { (d.cuEventElapsedTime)(&mut ms, self.e, later.e) };
        check(d, r, "cuEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for Event<'_> {
    fn drop(&mut self) {
        if self.gpu.bind().is_ok() {
            // SAFETY: our event, destroyed once.
            unsafe { (self.gpu.d.cuEventDestroy_v2)(self.e) };
        }
    }
}

/// An instantiated CUDA graph (`Gpu::capture`), destroyed on drop.
pub struct Graph<'g> {
    gpu: &'g Gpu,
    exec: ffi::CUgraphExec,
}

impl Graph<'_> {
    /// Replay the graph on the stream (asynchronous).
    ///
    /// # Safety
    /// Every buffer the captured kernels and copies touch is still live.
    pub unsafe fn launch(&self) -> Result<(), String> {
        self.gpu.bind()?;
        // SAFETY: our executable graph and stream; buffers live (caller).
        let r = unsafe { (self.gpu.d.cuGraphLaunch)(self.exec, self.gpu.stream) };
        check(self.gpu.d, r, "cuGraphLaunch")
    }
}

impl Drop for Graph<'_> {
    fn drop(&mut self) {
        if self.gpu.bind().is_ok() {
            // cuGraphExecDestroy doesn't wait: finish a replay in flight first.
            let _ = self.gpu.sync();
            // SAFETY: our executable graph, destroyed once, not running.
            unsafe { (self.gpu.d.cuGraphExecDestroy)(self.exec) };
        }
    }
}

/// Page-locked host memory (`Gpu::host_alloc`), zeroed, freed on drop, for
/// asynchronous stream-ordered copies. While a copy may be in flight the
/// host doesn't touch the bytes: `bytes_mut` waits for the stream first.
pub struct HostBuf<'g> {
    gpu: &'g Gpu,
    p: *mut u8,
    len: usize,
    /// A copy was enqueued since the last wait.
    pending: std::cell::Cell<bool>,
}

impl HostBuf<'_> {
    /// The bytes, once no copy to or from them is in flight (waits for the
    /// stream if one may be).
    pub fn bytes_mut(&mut self) -> Result<&mut [u8], String> {
        if self.pending.get() {
            self.gpu.sync()?;
            self.pending.set(false);
        }
        if self.len == 0 {
            return Ok(&mut []);
        }
        // SAFETY: our allocation of len bytes, borrowed mutably; no copy is
        // using it (waited for above).
        Ok(unsafe { std::slice::from_raw_parts_mut(self.p, self.len) })
    }

    /// Forget the copy in flight: the caller waited for the stream since it
    /// was enqueued (another buffer's `bytes_mut`, a `sync`), so the next
    /// `bytes_mut` needn't wait again.
    ///
    /// # Safety
    /// The stream has been synchronized after the last copy to or from us.
    pub unsafe fn assume_synced(&self) {
        self.pending.set(false);
    }

    /// Ok if `n` bytes at `off` fit in a `len`-byte device buffer and in us.
    fn check_range(&self, off: usize, n: usize, len: usize) -> Result<(), String> {
        if n > self.len || off.checked_add(n).is_none_or(|end| end > len) {
            return Err(format!("copy of {n} bytes at {off}: out of bounds"));
        }
        Ok(())
    }

    /// Enqueue a copy of our first `n` bytes to `dst` + `off`.
    pub fn copy_to(&self, dst: &DevBuf, off: usize, n: usize) -> Result<(), String> {
        self.check_range(off, n, dst.len)?;
        self.gpu.bind()?;
        self.pending.set(true);
        // SAFETY: in bounds of both (checked); pinned memory stays allocated
        // while self lives, and isn't touched by the host until the stream
        // has been waited for (`pending`, Drop).
        let r = unsafe {
            (self.gpu.d.cuMemcpyHtoDAsync_v2)(
                dst.ptr + off as u64,
                self.p.cast(),
                n,
                self.gpu.stream,
            )
        };
        check(self.gpu.d, r, "cuMemcpyHtoDAsync_v2")
    }

    /// Enqueue a copy of `n` bytes from `src` + `off` into our first bytes.
    pub fn copy_from(&mut self, src: &DevBuf, off: usize, n: usize) -> Result<(), String> {
        self.check_range(off, n, src.len)?;
        self.gpu.bind()?;
        self.pending.set(true);
        // SAFETY: as copy_to.
        let r = unsafe {
            (self.gpu.d.cuMemcpyDtoHAsync_v2)(
                self.p.cast(),
                src.ptr + off as u64,
                n,
                self.gpu.stream,
            )
        };
        check(self.gpu.d, r, "cuMemcpyDtoHAsync_v2")
    }
}

impl HostBuf<'_> {
    /// Enqueue `height` rows of `width` bytes from `src`, row i at
    /// `off + i * pitch` there, to row i at `at + i * host_pitch` here (one
    /// strided copy: a cache's rows of every layer at once).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_2d_from(
        &mut self,
        at: usize,
        host_pitch: usize,
        src: &DevBuf,
        off: usize,
        pitch: usize,
        (width, height): (usize, usize),
    ) -> Result<(), String> {
        self.copy_2d(false, at, host_pitch, src, off, pitch, width, height)
    }

    /// `copy_2d_from` the other way: our rows to `dst`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_2d_to(
        &self,
        at: usize,
        host_pitch: usize,
        dst: &DevBuf,
        off: usize,
        pitch: usize,
        (width, height): (usize, usize),
    ) -> Result<(), String> {
        self.copy_2d(true, at, host_pitch, dst, off, pitch, width, height)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_2d(
        &self,
        to_dev: bool,
        at: usize,
        host_pitch: usize,
        dev: &DevBuf,
        off: usize,
        pitch: usize,
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        // The end of the last row, if every row fits its pitch.
        let end = |o: usize, p: usize| {
            (width <= p)
                .then(|| {
                    (height - 1)
                        .checked_mul(p)?
                        .checked_add(o)?
                        .checked_add(width)
                })
                .flatten()
        };
        match (end(at, host_pitch), end(off, pitch)) {
            (Some(h), Some(d)) if h <= self.len && d <= dev.len => {}
            _ => {
                return Err(format!(
                    "2D copy of {height} x {width} bytes (host {at} pitch {host_pitch} of {}, \
                     device {off} pitch {pitch} of {}): out of bounds",
                    self.len, dev.len
                ));
            }
        }
        self.gpu.bind()?;
        self.pending.set(true);
        let (host, devp) = (self.p as usize + at, dev.ptr + off as u64);
        let c = if to_dev {
            ffi::Memcpy2D {
                src_type: ffi::CU_MEMORYTYPE_HOST,
                src_host: host,
                src_pitch: host_pitch,
                dst_type: ffi::CU_MEMORYTYPE_DEVICE,
                dst_dev: devp,
                dst_pitch: pitch,
                width,
                height,
                ..Default::default()
            }
        } else {
            ffi::Memcpy2D {
                src_type: ffi::CU_MEMORYTYPE_DEVICE,
                src_dev: devp,
                src_pitch: pitch,
                dst_type: ffi::CU_MEMORYTYPE_HOST,
                dst_host: host,
                dst_pitch: host_pitch,
                width,
                height,
                ..Default::default()
            }
        };
        // SAFETY: every row in bounds of both (checked); pinned memory and
        // device memory stay allocated while the copy may run (as copy_to).
        let r = unsafe { (self.gpu.d.cuMemcpy2DAsync_v2)(&c, self.gpu.stream) };
        check(self.gpu.d, r, "cuMemcpy2DAsync_v2")
    }
}

impl Drop for HostBuf<'_> {
    fn drop(&mut self) {
        if !self.p.is_null() && self.gpu.bind().is_ok() {
            // Copies in flight may still use it.
            let _ = self.gpu.sync();
            // SAFETY: our allocation, freed once.
            unsafe { (self.gpu.d.cuMemFreeHost)(self.p.cast()) };
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // Nothing to do about an error here; the context goes away anyway.
        let _ = self.bind();
        if !self.stream.is_null() {
            // SAFETY: our stream, destroyed once; pending work completes first.
            unsafe { (self.d.cuStreamDestroy_v2)(self.stream) };
        }
        // SAFETY: balances the retain in open_device.
        unsafe { (self.d.cuDevicePrimaryCtxRelease_v2)(self.dev) };
        // Leave no context current on this thread: the released one may be
        // destroyed (last reference), and a dangling current context would
        // be used by any later CUDA call here.
        // SAFETY: a null context is valid: it unbinds.
        unsafe { (self.d.cuCtxSetCurrent)(std::ptr::null_mut()) };
    }
}

/// Plain-old-data element types for copies to and from the device: any bit
/// pattern is a valid value, no padding.
///
/// # Safety
/// Only for types where every byte pattern of `size_of::<T>()` is a valid T.
pub unsafe trait Pod: Copy + Default {}
// SAFETY: primitive integers and floats: every bit pattern is valid.
unsafe impl Pod for u8 {}
// SAFETY: as above.
unsafe impl Pod for u16 {}
// SAFETY: as above.
unsafe impl Pod for u32 {}
// SAFETY: as above.
unsafe impl Pod for i32 {}
// SAFETY: as above.
unsafe impl Pod for u64 {}
// SAFETY: as above.
unsafe impl Pod for f32 {}

/// `len` bytes of device memory, freed on drop.
pub struct DevBuf<'g> {
    gpu: &'g Gpu,
    ptr: CUdeviceptr,
    len: usize,
}

impl DevBuf<'_> {
    pub fn len(&self) -> usize {
        self.len
    }

    /// A kernel argument: the device address.
    pub fn arg(&self) -> Arg {
        Arg::Ptr(self.ptr)
    }

    /// A kernel argument: the device address `off` bytes in. Panics past the
    /// end (a launch with it would read out of bounds).
    pub fn arg_at(&self, off: usize) -> Arg {
        assert!(
            off <= self.len,
            "arg_at({off}) past a {}-byte buffer",
            self.len
        );
        Arg::Ptr(self.ptr + off as u64)
    }

    /// Copy `data` to the start of the buffer. Synchronous: enqueued on the
    /// stream after the work already launched there, then waited for.
    pub fn write<T: Pod>(&self, data: &[T]) -> Result<(), String> {
        self.write_at(0, data)
    }

    /// `write` at byte offset `off`.
    pub fn write_at<T: Pod>(&self, off: usize, data: &[T]) -> Result<(), String> {
        let n = size_of_val(data);
        if off.checked_add(n).is_none_or(|end| end > self.len) {
            return Err(format!(
                "upload of {n} bytes at {off} into a {}-byte buffer",
                self.len
            ));
        }
        if n == 0 {
            return Ok(());
        }
        self.gpu.bind()?;
        // SAFETY: n bytes into our allocation at off (in bounds, checked)
        // from data's n bytes, which stay borrowed until the sync below has
        // seen the copy finish.
        let r = unsafe {
            (self.gpu.d.cuMemcpyHtoDAsync_v2)(
                self.ptr + off as u64,
                data.as_ptr().cast(),
                n,
                self.gpu.stream,
            )
        };
        check(self.gpu.d, r, "cuMemcpyHtoDAsync_v2")?;
        self.gpu.sync()
    }

    /// The whole buffer as Ts (its length must be a multiple of T's size),
    /// after the work launched so far finishes.
    #[cfg_attr(not(test), allow(dead_code))] // tests and benchmarks
    pub fn read<T: Pod>(&self) -> Result<Vec<T>, String> {
        let sz = size_of::<T>();
        if !self.len.is_multiple_of(sz) {
            return Err(format!(
                "{}-byte buffer read as {sz}-byte elements",
                self.len
            ));
        }
        let mut out = vec![T::default(); self.len / sz];
        if self.len > 0 {
            self.gpu.bind()?;
            // SAFETY: len bytes from our allocation into out's len bytes
            // (T is Pod so any bytes are valid Ts), stream-ordered after the
            // kernels; out isn't touched until the sync has seen it finish.
            let r = unsafe {
                (self.gpu.d.cuMemcpyDtoHAsync_v2)(
                    out.as_mut_ptr().cast(),
                    self.ptr,
                    self.len,
                    self.gpu.stream,
                )
            };
            check(self.gpu.d, r, "cuMemcpyDtoHAsync_v2")?;
            self.gpu.sync()?;
        }
        Ok(out)
    }

    /// Copy `n` bytes at `off` to `dst` at `dst_off`, enqueued on the
    /// stream (ordered after the work launched there, not waited for).
    pub fn copy_into(
        &self,
        off: usize,
        dst: &DevBuf,
        dst_off: usize,
        n: usize,
    ) -> Result<(), String> {
        let ok = |o: usize, len: usize| o.checked_add(n).is_some_and(|end| end <= len);
        if !ok(off, self.len) || !ok(dst_off, dst.len) {
            return Err(format!(
                "copy of {n} bytes from {off} of {} to {dst_off} of {}",
                self.len, dst.len
            ));
        }
        if n == 0 {
            return Ok(());
        }
        self.gpu.bind()?;
        // SAFETY: n bytes in bounds of both allocations (checked), which
        // outlive the stream work (cuMemFree waits for it).
        let r = unsafe {
            (self.gpu.d.cuMemcpyDtoDAsync_v2)(
                dst.ptr + dst_off as u64,
                self.ptr + off as u64,
                n,
                self.gpu.stream,
            )
        };
        check(self.gpu.d, r, "cuMemcpyDtoDAsync_v2")
    }

    /// Set every byte to zero (synchronous, stream-ordered).
    pub fn zero(&self) -> Result<(), String> {
        if self.len == 0 {
            return Ok(());
        }
        self.gpu.bind()?;
        // SAFETY: len bytes of our allocation.
        let r = unsafe { (self.gpu.d.cuMemsetD8Async)(self.ptr, 0, self.len, self.gpu.stream) };
        check(self.gpu.d, r, "cuMemsetD8Async")?;
        self.gpu.sync()
    }
}

impl Drop for DevBuf<'_> {
    fn drop(&mut self) {
        if self.ptr != 0 && self.gpu.bind().is_ok() {
            // SAFETY: our allocation, freed once. cuMemFree waits for work
            // using it to finish.
            unsafe { (self.gpu.d.cuMemFree_v2)(self.ptr) };
        }
    }
}

/// A kernel argument, passed by value.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// A device address (`T*` in the kernel).
    Ptr(u64),
    U64(u64),
    #[cfg_attr(not(test), allow(dead_code))] // tests and benchmarks
    U32(u32),
    I32(i32),
    F32(f32),
    /// 128 bytes by value: a CUtensorMap (`Gpu::tensor_map_2d`), the
    /// kernel's `const __grid_constant__ TMap` (hopper.cu).
    Blob128([u64; 16]),
}

impl Arg {
    /// Bytes of the value (the kernel parameter's size: debug builds check
    /// it in `Gpu::launch_raw`).
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    fn size(&self) -> usize {
        match self {
            Arg::Ptr(_) | Arg::U64(_) => 8,
            Arg::U32(_) | Arg::I32(_) | Arg::F32(_) => 4,
            Arg::Blob128(_) => 128,
        }
    }

    /// Address of the value, as cuLaunchKernel's parameter array wants it.
    fn addr(&self) -> *mut c_void {
        match self {
            Arg::Ptr(v) | Arg::U64(v) => v as *const u64 as *mut c_void,
            Arg::U32(v) => v as *const u32 as *mut c_void,
            Arg::I32(v) => v as *const i32 as *mut c_void,
            Arg::F32(v) => v as *const f32 as *mut c_void,
            Arg::Blob128(v) => v as *const [u64; 16] as *mut c_void,
        }
    }
}

/// A kernel of a loaded `Module`, which it borrows (so it can't outlive it),
/// and its name for error messages.
pub struct Func<'m> {
    f: CUfunction,
    name: String,
    module: &'m Module<'m>,
}

impl Func<'_> {
    /// Allow up to `bytes` of dynamic shared memory per block (over the
    /// default 48 KiB needs this opt-in).
    pub fn set_max_shared(&self, bytes: usize) -> Result<(), String> {
        let g = self.module.gpu;
        g.bind()?;
        // SAFETY: a function of a loaded module; a plain attribute write.
        let r = unsafe {
            (g.d.cuFuncSetAttribute)(
                self.f,
                ffi::FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes as c_int,
            )
        };
        check(g.d, r, &format!("cuFuncSetAttribute {}", self.name))
    }

    /// (registers per thread, local memory bytes per thread): spills show
    /// up as local memory.
    #[cfg_attr(not(test), allow(dead_code))] // tuning tests
    pub fn regs_local(&self) -> Result<(i32, i32), String> {
        let g = self.module.gpu;
        g.bind()?;
        let (mut regs, mut local) = (0, 0);
        // SAFETY: a function of a loaded module; writes one int each.
        let r = unsafe { (g.d.cuFuncGetAttribute)(&mut regs, ffi::FUNC_ATTR_NUM_REGS, self.f) };
        check(g.d, r, "cuFuncGetAttribute")?;
        // SAFETY: as above.
        let r = unsafe {
            (g.d.cuFuncGetAttribute)(&mut local, ffi::FUNC_ATTR_LOCAL_SIZE_BYTES, self.f)
        };
        check(g.d, r, "cuFuncGetAttribute")?;
        Ok((regs, local))
    }
}

/// How `Module::load` got its code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    /// From the cache.
    Hit,
    /// Compiled: no cache entry yet.
    Miss,
    /// Compiled: the entry was unreadable, damaged or wouldn't load (it was replaced).
    Corrupt,
    /// Compiled: no cache directory (no absolute $XDG_CACHE_HOME or $HOME).
    Off,
}

/// Compiled CUDA code loaded on a `Gpu`; unloaded on drop.
pub struct Module<'g> {
    gpu: &'g Gpu,
    m: CUmodule,
    /// Its kernels may be launched overlapping their predecessor
    /// (`Gpu::launch_with`): see `pdl_ok`.
    pdl: bool,
    /// The architecture its code was compiled for (10 * major + minor; 0
    /// if unknown): int8 tensor cores (mma.sync) need 80.
    arch: i32,
    /// Built with AXON_EXT_TYPES (`Kernels::module_ext`): its decode
    /// kernels run Q4_0, Q4_1, Q5_1 and Q5_K too.
    ext: bool,
    /// Compiled for sm_90a (hopper.cu's kernels are in it) and the driver
    /// has cuTensorMapEncodeTiled.
    hopper: bool,
}

/// Whether programmatic dependent launch is safe and works for code
/// compiled for `arch` (10 * major + minor of the cubin or PTX loaded) on a
/// device of `cc` with a CUDA `driver`, `env` being $AXON_CUDA_PDL:
/// - the code was compiled for sm_90+: decode.cu emits its
///   `griddepcontrol.wait` only then (`__CUDA_ARCH__ >= 900`), and without
///   the wait an overlapped kernel could read its predecessor's output
///   early (a PTX fallback for an older arch has none);
/// - the device is sm_90+ (it was compiled for, but say so);
/// - the driver is CUDA 12.3+, the first to keep programmatic launches in
///   stream capture (older ones fail the capture);
/// - AXON_CUDA_PDL isn't "0".
fn pdl_ok(arch: i32, cc: i32, driver: (i32, i32), env: Option<&str>) -> bool {
    arch >= 90 && cc >= 90 && driver >= (12, 3) && env != Some("0")
}

/// $XDG_CACHE_HOME/axon/cuda, else ~/.cache/axon/cuda.
pub fn cache_dir() -> Option<PathBuf> {
    cache_dir_from(std::env::var_os("XDG_CACHE_HOME"), std::env::var_os("HOME"))
}

/// `cache_dir` from those variables' values. Relative paths (and empty
/// ones) are ignored, as the XDG Base Directory spec says.
fn cache_dir_from(xdg: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    let abs = |v: Option<OsString>| v.map(PathBuf::from).filter(|p| p.is_absolute());
    let base = match abs(xdg) {
        Some(x) => x,
        None => abs(home)?.join(".cache"),
    };
    Some(base.join("axon").join("cuda"))
}

/// What decides the bits this device's kernels compute, as a string
/// (kvsave's identity: a cache image is only loaded where it was made):
/// the kernel sources and options, the device's architecture and
/// AXON_HOPPER (the sm_90a build). Not the driver or NVRTC version: the
/// xarch audit measured the same bits across them (`driver_tag`).
pub fn numerics_tag(gpu: &Gpu) -> Result<String, String> {
    let src = fnv128(&[KERNELS.as_bytes(), KERNEL_OPTS.join("\n").as_bytes()]);
    Ok(format!(
        "kernels {src:032x} {} hopper {}",
        gpu.info()?.arch(),
        hopper_env()
    ))
}

/// The driver's CUDA version, "13.0" (kvsave records it, for people).
pub fn driver_tag(gpu: &Gpu) -> Result<String, String> {
    let d = ffi::driver_version(gpu.d)?;
    Ok(format!("{}.{}", d.0, d.1))
}

/// FNV-1a, 128-bit: a stable hash (unlike std's, which may change between
/// Rust releases) for cache keys and the entries' checksums.
fn fnv128(parts: &[&[u8]]) -> u128 {
    let mut h: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    for p in parts {
        // Length first, so ("ab", "c") and ("a", "bc") differ.
        for &b in (p.len() as u64).to_le_bytes().iter().chain(p.iter()) {
            h ^= b as u128;
            h = h.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
        }
    }
    h
}

/// A cache entry: this magic, the payload's fnv128, the cubin.
const MAGIC: &[u8; 10] = b"AXONCUBIN\x01";

fn encode_entry(cubin: &[u8]) -> Vec<u8> {
    let mut v = MAGIC.to_vec();
    v.extend_from_slice(&fnv128(&[cubin]).to_le_bytes());
    v.extend_from_slice(cubin);
    v
}

/// The cubin of a well-formed entry, None if it's damaged.
fn decode_entry(bytes: &[u8]) -> Option<&[u8]> {
    let rest = bytes.strip_prefix(MAGIC.as_slice())?;
    let (sum, cubin) = rest.split_at_checked(16)?;
    (!cubin.is_empty() && sum == fnv128(&[cubin]).to_le_bytes()).then_some(cubin)
}

/// Temp files older than this in the cache dir are leftovers of a process
/// that died between write and rename.
const STALE_TMP: std::time::Duration = std::time::Duration::from_secs(3600);

/// Write `bytes` to `path` atomically: a temp file in the same directory,
/// then a rename, so a reader never sees half an entry. Temp names are
/// unique per process (pid) and per call (a counter). Creates the directory
/// private (0700: what's in it gets loaded as code) and sweeps stale temp
/// files. Best effort: a cache that can't be written only costs a compile
/// next time.
fn store(path: &Path, bytes: &[u8]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let Some(dir) = path.parent() else { return };
    if create_private_dir(dir).is_err() {
        return;
    }
    sweep_stale_tmp(dir, STALE_TMP);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!(".tmp.{}.{seq}", std::process::id()));
    let ok = std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, path).is_ok();
    if !ok {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// `dir` and its missing parents, created with mode 0700 on Unix.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    let mut b = std::fs::DirBuilder::new();
    b.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut b, 0o700);
    b.create(dir)
}

/// Remove `.tmp.*` files in `dir` last modified more than `age` ago.
fn sweep_stale_tmp(dir: &Path, age: std::time::Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let old = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|el| el > age);
        if old && e.file_name().to_string_lossy().starts_with(".tmp.") {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// What NVRTC compiles kernels to for a device: architectures as
/// 10 * major + minor (sm_120 = 120).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    /// Machine code for sm_XY: runs on that major with an equal or higher minor.
    Cubin(i32),
    /// PTX for compute_XY, which the driver compiles for any newer device.
    Ptx(i32),
    /// Machine code for sm_90a: sm_90's plus the arch-specific Hopper
    /// instructions (wgmma, TMA: hopper.cu). Runs only on compute
    /// capability 9.0 (H100, H200, H20, GH200).
    Cubin90a,
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Target::Cubin(a) => write!(f, "sm_{a}"),
            Target::Ptx(a) => write!(f, "compute_{a}"),
            Target::Cubin90a => write!(f, "sm_90a"),
        }
    }
}

/// The best `Target` for a device of architecture `cc` given the ones NVRTC
/// supports (`archs`): sm_90a for a 9.0 device if `hopper` ($AXON_HOPPER
/// isn't "0") and NVRTC is 12.0+ (the first with arch-specific targets);
/// else its own cubin; else a cubin of the newest older minor of its major
/// (binary compatible); else PTX of the newest older architecture, for the
/// driver to compile (a GPU newer than NVRTC). Err if the GPU is older than
/// everything NVRTC still supports.
fn pick_target(cc: i32, archs: &[i32], nvrtc: (i32, i32), hopper: bool) -> Result<Target, String> {
    if cc == 90 && hopper && nvrtc.0 >= 12 && archs.contains(&90) {
        return Ok(Target::Cubin90a);
    }
    if archs.contains(&cc) {
        return Ok(Target::Cubin(cc));
    }
    let older = || archs.iter().copied().filter(|&a| a < cc);
    if let Some(a) = older().filter(|&a| a / 10 == cc / 10).max() {
        return Ok(Target::Cubin(a));
    }
    if let Some(a) = older().max() {
        return Ok(Target::Ptx(a));
    }
    Err(format!(
        "NVRTC {}.{} can't compile for sm_{cc}: the oldest it supports is sm_{}; \
         use an older CUDA toolkit for this GPU",
        nvrtc.0,
        nvrtc.1,
        archs.iter().min().map_or("?".into(), |a| a.to_string())
    ))
}

/// Err if NVRTC is from a newer CUDA major than the driver supports: what it
/// builds may not load there.
fn check_versions(driver: (i32, i32), nvrtc: (i32, i32)) -> Result<(), String> {
    if nvrtc.0 > driver.0 {
        return Err(format!(
            "NVRTC {}.{} needs a CUDA {} driver, have {}.{}; update the NVIDIA driver \
             or point CUDA_PATH at a CUDA {} toolkit",
            nvrtc.0, nvrtc.1, nvrtc.0, driver.0, driver.1, driver.0
        ));
    }
    Ok(())
}

/// The NVRTC to build for a device of architecture `cc` (10 * major +
/// minor) on a CUDA `driver`, and its `Target` (`hopper`: as `pick_target`):
/// the first candidate (`ffi::nvrtc_where`, the driver's CUDA major first)
/// that isn't newer than the driver (`check_versions`) and supports the
/// device (`pick_target`). So a GPU the newest toolkit dropped (Pascal,
/// Volta under CUDA 13) gets an older one that is also installed. Err: each
/// loaded NVRTC's refusal.
fn nvrtc_for(
    cc: i32,
    driver: (i32, i32),
    hopper: bool,
) -> Result<(&'static ffi::Nvrtc, Target), String> {
    let target = std::cell::Cell::new(None);
    let n = ffi::nvrtc_where(|n| {
        let nv = ffi::nvrtc_version(n)?;
        check_versions(driver, nv)?;
        target.set(Some(pick_target(cc, &ffi::nvrtc_archs(n)?, nv, hopper)?));
        Ok(())
    })?;
    Ok((n, target.get().expect("set for the NVRTC accepted")))
}

/// Whether sm_90 builds Hopper's sm_90a code: AXON_HOPPER isn't "0".
fn hopper_env() -> bool {
    std::env::var("AXON_HOPPER").ok().as_deref() != Some("0")
}

/// Compile CUDA C++ `src` with NVRTC `n` (`opts` include the architecture)
/// to a cubin, or for a `Ptx` target NUL-terminated PTX (as
/// cuModuleLoadData takes it). Err carries the compile log.
fn compile(n: &ffi::Nvrtc, src: &str, opts: &[String], target: Target) -> Result<Vec<u8>, String> {
    compile_logged(n, src, opts, target).map(|(image, _)| image)
}

/// `compile`, and the compile log of a successful build (ptxas notes, e.g.
/// wgmma serialization: tests check it's empty).
fn compile_logged(
    n: &ffi::Nvrtc,
    src: &str,
    opts: &[String],
    target: Target,
) -> Result<(Vec<u8>, String), String> {
    let csrc = CString::new(src).map_err(|_| "kernel source has a NUL byte")?;
    let copts: Vec<CString> = opts
        .iter()
        .map(|o| CString::new(o.as_str()).map_err(|_| "NVRTC option has a NUL byte"))
        .collect::<Result<_, _>>()?;
    let optp: Vec<*const std::ffi::c_char> = copts.iter().map(|c| c.as_ptr()).collect();
    let mut prog: ffi::NvrtcProgram = std::ptr::null_mut();
    // SAFETY: csrc and the name are NUL-terminated; no headers.
    let r = unsafe {
        (n.nvrtcCreateProgram)(
            &mut prog,
            csrc.as_ptr(),
            c"axon.cu".as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    ffi::check_nvrtc(n, r, "nvrtcCreateProgram")?;
    /// Destroys the program on every path out.
    struct Prog<'a>(&'a ffi::Nvrtc, ffi::NvrtcProgram);
    impl Drop for Prog<'_> {
        fn drop(&mut self) {
            // SAFETY: a program nvrtcCreateProgram made, destroyed once.
            unsafe { (self.0.nvrtcDestroyProgram)(&mut self.1) };
        }
    }
    let p = Prog(n, prog);
    // SAFETY: optp holds optp.len() NUL-terminated strings alive in copts.
    let r = unsafe { (n.nvrtcCompileProgram)(p.1, optp.len() as c_int, optp.as_ptr()) };
    let mut len = 0usize;
    // SAFETY: writes one size_t.
    let mut log = if unsafe { (n.nvrtcGetProgramLogSize)(p.1, &mut len) } == 0 && len > 0 {
        let mut buf = vec![0u8; len];
        // SAFETY: buf has the len bytes the log needs (NUL included).
        let r = unsafe { (n.nvrtcGetProgramLog)(p.1, buf.as_mut_ptr().cast()) };
        match r {
            ffi::NVRTC_SUCCESS => String::from_utf8_lossy(&buf[..len - 1]).into_owned(),
            _ => String::new(), // the compile error itself still goes out
        }
    } else {
        String::new()
    };
    if r != ffi::NVRTC_SUCCESS {
        if log.trim().is_empty() {
            log = "(no log)".into();
        }
        let e = ffi::check_nvrtc(n, r, "nvrtcCompileProgram").unwrap_err();
        return Err(format!("{e} [{}]:\n{}", opts.join(" "), log.trim_end()));
    }
    let ptx = matches!(target, Target::Ptx(_));
    let (size_fn, get_fn, what) = if ptx {
        (n.nvrtcGetPTXSize, n.nvrtcGetPTX, "nvrtcGetPTX")
    } else {
        (n.nvrtcGetCUBINSize, n.nvrtcGetCUBIN, "nvrtcGetCUBIN")
    };
    let mut len = 0usize;
    // SAFETY: writes one size_t.
    let r = unsafe { size_fn(p.1, &mut len) };
    ffi::check_nvrtc(n, r, what)?;
    let mut image = vec![0u8; len];
    // SAFETY: image has the len bytes the size call asked for (for PTX,
    // its terminating NUL included).
    let r = unsafe { get_fn(p.1, image.as_mut_ptr().cast()) };
    ffi::check_nvrtc(n, r, what)?;
    if image.is_empty() || (ptx && image.last() != Some(&0)) {
        return Err(format!("NVRTC produced no code for {target}"));
    }
    Ok((image, log))
}

impl<'g> Module<'g> {
    /// `src` compiled for this GPU with `opts` (beyond the architecture),
    /// through the kernel cache at `cache_dir()`.
    pub fn load(
        gpu: &'g Gpu,
        src: &str,
        opts: &[&str],
    ) -> Result<(Module<'g>, CacheStatus), String> {
        Self::load_in(gpu, src, opts, cache_dir().as_deref())
    }

    /// `load` with the cache in `dir` (None: no cache). Picks the NVRTC
    /// and the target (`nvrtc_for`).
    pub fn load_in(
        gpu: &'g Gpu,
        src: &str,
        opts: &[&str],
        dir: Option<&Path>,
    ) -> Result<(Module<'g>, CacheStatus), String> {
        let cc = gpu.info()?.cc;
        let (n, target) = nvrtc_for(
            (cc.0 * 10 + cc.1) as i32,
            ffi::driver_version(gpu.d)?,
            hopper_env(),
        )?;
        Self::load_for(gpu, n, src, opts, dir, target)
    }

    /// `load_in` with NVRTC `n` for `target`. Entries are keyed by a hash of
    /// the source, options, target and NVRTC version. One that can't be
    /// read, fails its checksum or won't load is compiled again and replaced,
    /// never an error. Fresh code is stored only once it has loaded, so code
    /// the driver rejects is never cached.
    fn load_for(
        gpu: &'g Gpu,
        n: &ffi::Nvrtc,
        src: &str,
        opts: &[&str],
        dir: Option<&Path>,
        target: Target,
    ) -> Result<(Module<'g>, CacheStatus), String> {
        let (mut m, status) = Self::load_cached(gpu, n, src, opts, dir, target)?;
        let arch = match target {
            Target::Cubin(a) | Target::Ptx(a) => a,
            Target::Cubin90a => 90,
        };
        m.hopper = target == Target::Cubin90a && ffi::tensor_map_encoder().is_some();
        let cc = gpu.info()?.cc;
        let env = std::env::var("AXON_CUDA_PDL").ok();
        m.arch = arch;
        m.pdl = pdl_ok(
            arch,
            (cc.0 * 10 + cc.1) as i32,
            ffi::driver_version(gpu.d)?,
            env.as_deref(),
        );
        Ok((m, status))
    }

    /// `load_for` without setting `pdl`.
    fn load_cached(
        gpu: &'g Gpu,
        n: &ffi::Nvrtc,
        src: &str,
        opts: &[&str],
        dir: Option<&Path>,
        target: Target,
    ) -> Result<(Module<'g>, CacheStatus), String> {
        let nv = ffi::nvrtc_version(n)?;
        let mut all = vec![format!("--gpu-architecture={target}")];
        all.extend(opts.iter().map(|s| s.to_string()));
        let (joined, ver) = (all.join("\n"), format!("nvrtc {}.{}", nv.0, nv.1));
        let tname = target.to_string();
        let key = fnv128(&[
            src.as_bytes(),
            joined.as_bytes(),
            tname.as_bytes(),
            ver.as_bytes(),
        ]);
        let fresh = || -> Result<(Module<'g>, Vec<u8>), String> {
            let image = compile(n, src, &all, target)?;
            let m = Self::from_image(gpu, &image).map_err(|e| {
                let d = ffi::driver_version(gpu.d).unwrap_or((0, 0));
                format!(
                    "{e}; NVRTC {}.{} built it for {target}, the driver is CUDA {}.{}",
                    nv.0, nv.1, d.0, d.1
                )
            })?;
            Ok((m, image))
        };
        let Some(dir) = dir else {
            return Ok((fresh()?.0, CacheStatus::Off));
        };
        let path = dir.join(format!("{key:032x}.cubin"));
        let status = match std::fs::read(&path) {
            Ok(bytes) => match decode_entry(&bytes) {
                Some(image) => match Self::from_image(gpu, image) {
                    Ok(m) => return Ok((m, CacheStatus::Hit)),
                    Err(_) => CacheStatus::Corrupt,
                },
                None => CacheStatus::Corrupt,
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => CacheStatus::Miss,
            Err(_) => CacheStatus::Corrupt,
        };
        let (m, image) = fresh()?;
        store(&path, &encode_entry(&image));
        Ok((m, status))
    }

    /// Whether this module's kernels may overlap their predecessor
    /// (`pdl_ok` for the target it was compiled for).
    #[cfg(test)]
    pub fn pdl(&self) -> bool {
        self.pdl
    }

    /// Whether its code has the int8 tensor-core kernels (built for sm_80+).
    pub fn mma(&self) -> bool {
        self.arch >= 80
    }

    /// Whether its code has hopper.cu's kernels (built for sm_90a) and the
    /// driver can make their TMA descriptors (`Gpu::tensor_map_2d`).
    pub fn hopper(&self) -> bool {
        self.hopper
    }

    /// Whether the decode kernels run the extra types (`Kernels::module_ext`).
    pub fn ext(&self) -> bool {
        self.ext
    }

    /// Turn programmatic dependent launch off for this module (tests: PDL
    /// on vs off).
    #[cfg(test)]
    pub fn disable_pdl(&mut self) {
        self.pdl = false;
    }

    /// Load a cubin, or NUL-terminated PTX (compiled by the driver).
    fn from_image(gpu: &'g Gpu, image: &[u8]) -> Result<Module<'g>, String> {
        gpu.bind()?;
        let mut m: CUmodule = std::ptr::null_mut();
        // SAFETY: context current; image is an in-memory cubin the driver
        // parses (and rejects if malformed) before returning, or PTX text
        // (cache entries pass a checksum; PTX from `compile` ends in a NUL).
        let r = unsafe { (gpu.d.cuModuleLoadData)(&mut m, image.as_ptr().cast()) };
        check(gpu.d, r, "cuModuleLoadData")?;
        Ok(Module {
            gpu,
            m,
            pdl: false,
            arch: 0,
            ext: false,
            hopper: false,
        })
    }

    /// The `extern "C"` kernel `name`.
    pub fn function(&self, name: &str) -> Result<Func<'_>, String> {
        let c = CString::new(name).map_err(|_| "kernel name has a NUL byte")?;
        let mut f: CUfunction = std::ptr::null_mut();
        self.gpu.bind()?;
        // SAFETY: the module is loaded; c is NUL-terminated.
        let r = unsafe { (self.gpu.d.cuModuleGetFunction)(&mut f, self.m, c.as_ptr()) };
        check(self.gpu.d, r, &format!("cuModuleGetFunction {name}"))?;
        Ok(Func {
            f,
            name: name.into(),
            module: self,
        })
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        if self.gpu.bind().is_ok() {
            // SAFETY: our module, unloaded once; no Func of it is left (they
            // borrow it), so nothing launches its kernels after this.
            unsafe { (self.gpu.d.cuModuleUnload)(self.m) };
        }
    }
}

/// The dequantization kernel of kernels.cu (G0's bit-exactness checks), from
/// a `Module` made by `Kernels::module`, which also holds decode.cu's.
#[cfg_attr(not(test), allow(dead_code))] // tests and benchmarks
pub struct Kernels<'m> {
    gpu: &'m Gpu,
    dequant_rows: Func<'m>,
    /// The module's `ext` (the extra types' dequant compiled in).
    ext: bool,
}

/// `Kernels::mla_module`'s source: kernels.cu's dequantization, moe.cu's
/// router logits and add, mla.cu.
const MLA_KERNELS: &str = concat!(
    include_str!("kernels.cu"),
    include_str!("moe.cu"),
    include_str!("mla.cu")
);

/// NVRTC options for kernels.cu beyond the architecture. Denormals kept,
/// as on the CPU (the defaults, spelled out: they are part of the cache key).
const KERNEL_OPTS: [&str; 2] = ["--std=c++17", "--ftz=false"];
/// `KERNEL_OPTS` plus the extra weight types (Q4_0, Q4_1, Q5_1, Q5_K, Q2_K, Q3_K, IQ4_NL) in the
/// decode kernels' type switch.
const KERNEL_OPTS_EXT: [&str; 3] = ["--std=c++17", "--ftz=false", "-DAXON_EXT_TYPES"];

/// Whether `m` has a matrix type only `Kernels::module_ext` runs.
pub fn needs_ext(m: &crate::gguf::Model) -> bool {
    use crate::quant::{IQ4_NL, Q2_K, Q3_K, Q4_0, Q4_1, Q5_1, Q5_K};
    m.tensors
        .iter()
        .any(|t| matches!(t.qtype.id, Q4_0 | Q4_1 | Q5_1 | Q5_K | Q2_K | Q3_K | IQ4_NL))
}

impl<'m> Kernels<'m> {
    /// kernels.cu compiled for `gpu` (or from the cache): main's four
    /// quant types (Q4_K, Q6_K, Q5_0, Q8_0) in the decode kernels.
    pub fn module(gpu: &Gpu) -> Result<Module<'_>, String> {
        Ok(Module::load(gpu, KERNELS, &KERNEL_OPTS)?.0)
    }

    /// `module` with every supported quant type in the decode kernels
    /// (AXON_EXT_TYPES). Same bits for the four (same code per type); a
    /// separate build so models without the extra types keep the smaller
    /// kernels.
    pub fn module_ext(gpu: &Gpu) -> Result<Module<'_>, String> {
        let mut m = Module::load(gpu, KERNELS, &KERNEL_OPTS_EXT)?.0;
        m.ext = true;
        Ok(m)
    }

    /// moe.cu (the MoE router, `crate::moe`) on its own: the dense
    /// models' module stays as it is.
    pub fn moe_module(gpu: &Gpu) -> Result<Module<'_>, String> {
        Ok(Module::load(gpu, include_str!("moe.cu"), &KERNEL_OPTS)?.0)
    }

    /// mla.cu (deepseek2's MLA attention and router, `cuda::qwen2::mla`)
    /// on its own, after kernels.cu's dequantization (every type) and
    /// moe.cu's router logits: the other modules stay as they are.
    /// Debug builds compile it with MLA_DEBUG (mla.cu's traps: a router
    /// with no finite score), their own cache entry.
    pub fn mla_module(gpu: &Gpu) -> Result<Module<'_>, String> {
        let mut opts = KERNEL_OPTS_EXT.to_vec();
        if cfg!(debug_assertions) {
            opts.push("-DMLA_DEBUG");
        }
        Ok(Module::load(gpu, MLA_KERNELS, &opts)?.0)
    }

    /// The module model `m` needs: `module_ext` if it has an extra type.
    pub fn module_for<'g>(gpu: &'g Gpu, m: &crate::gguf::Model) -> Result<Module<'g>, String> {
        if needs_ext(m) {
            Self::module_ext(gpu)
        } else {
            Self::module(gpu)
        }
    }

    /// The kernels of `m`, which must be `Kernels::module`'s.
    #[cfg_attr(not(test), allow(dead_code))] // tests and benchmarks
    pub fn new(m: &'m Module<'_>) -> Result<Kernels<'m>, String> {
        Ok(Kernels {
            gpu: m.gpu,
            dequant_rows: m.function("dequant_rows")?,
            ext: m.ext,
        })
    }

    /// Rows `rows` of a matrix of type `qt` with `n` values per row, whose
    /// bytes are `w`, dequantized to f32: bit for bit `quant::dequant` of
    /// each row, concatenated.
    #[cfg_attr(not(test), allow(dead_code))] // tests and benchmarks
    pub fn dequant_rows(
        &self,
        qt: u32,
        w: &DevBuf,
        n: usize,
        rows: &[u32],
    ) -> Result<Vec<f32>, String> {
        use crate::quant::{
            F16, F32, IQ4_NL, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0,
        };
        let q = crate::gguf::qtype(qt);
        if ![
            F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL,
        ]
        .contains(&qt)
        {
            return Err(format!("dequant_rows: unsupported type {}", q.name));
        }
        if [Q4_0, Q4_1, Q5_1, Q5_K, Q2_K, Q3_K, IQ4_NL].contains(&qt) && !self.ext {
            return Err(format!(
                "dequant_rows: type {} needs Kernels::module_ext",
                q.name
            ));
        }
        if n == 0 || !n.is_multiple_of(q.block_size) || n > 65_535 * 256 {
            return Err(format!("dequant_rows: bad row length {n} for {}", q.name));
        }
        // One block per row in grid x, at most 2^31 - 1.
        if rows.len() > i32::MAX as usize {
            return Err(format!(
                "dequant_rows: {} rows, at most 2^31 - 1",
                rows.len()
            ));
        }
        let rb = crate::quant::row_bytes(qt, n);
        if let Some(&r) = rows.iter().find(|&&r| (r as usize + 1) * rb > w.len()) {
            return Err(format!(
                "dequant_rows: row {r} is past the {}-byte matrix",
                w.len()
            ));
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let idx = self.gpu.upload(rows)?;
        let out = self.gpu.alloc(rows.len() * n * 4)?;
        const BLOCK: u32 = 256;
        // SAFETY: the args match dequant_rows' signature; every row read is
        // inside w (checked above) and out holds rows.len() * n floats.
        unsafe {
            self.gpu.launch(
                &self.dequant_rows,
                (rows.len() as u32, (n as u32).div_ceil(BLOCK), 1),
                (BLOCK, 1, 1),
                0,
                &[
                    Arg::I32(qt as i32),
                    w.arg(),
                    Arg::U64(rb as u64),
                    Arg::I32(n as i32),
                    idx.arg(),
                    out.arg(),
                ],
            )?;
        }
        out.read()
    }
}

/// One line about the GPU for `axon info`: name, architecture, memory and
/// theoretical bandwidth, or why there's none; and if no NVRTC can build
/// for it (`nvrtc_for`), that it's unusable and why.
pub fn summary() -> String {
    let usable = |i: &DeviceInfo| {
        let d = ffi::driver_version(ffi::driver()?)?;
        nvrtc_for((i.cc.0 * 10 + i.cc.1) as i32, d, hopper_env()).map(|_| ())
    };
    match Gpu::open().and_then(|g| g.info()) {
        Ok(i) => {
            const GIB: f64 = (1u64 << 30) as f64;
            let mut s = format!(
                "gpu: {} | {}, {} SMs | {:.1} GiB ({:.1} free)",
                i.name,
                i.arch(),
                i.sms,
                i.total_bytes as f64 / GIB,
                i.free_bytes as f64 / GIB
            );
            if let Some(bw) = i.bandwidth_gbs() {
                s.push_str(&format!(" | ~{bw:.0} GB/s"));
            }
            if let Err(e) = usable(&i) {
                s.push_str(&format!(" | unusable: {e}"));
            }
            s
        }
        Err(e) => format!("gpu: none ({e})"),
    }
}

/// The code `Module::load` builds for `gpu` (`nvrtc_for`'s target), as
/// NVRTC names it ("sm_89", "sm_90a", "compute_90"); for PTX, which the
/// driver compiles, with the driver's version; and the architecture it is
/// for (10 * major + minor, `Module::mma`'s). For the identity tuple.
pub fn code_target(gpu: &Gpu) -> Result<(String, i32), String> {
    let cc = gpu.info()?.cc;
    let d = ffi::driver_version(gpu.d)?;
    let (_, target) = nvrtc_for((cc.0 * 10 + cc.1) as i32, d, hopper_env())?;
    Ok(match target {
        Target::Ptx(a) => (format!("{target} PTX, driver {}.{}", d.0, d.1), a),
        Target::Cubin(a) => (target.to_string(), a),
        Target::Cubin90a => (target.to_string(), 90),
    })
}

#[cfg(test)]
mod tests;
