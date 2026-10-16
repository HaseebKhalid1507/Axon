//! The CUDA driver API and NVRTC, loaded at run time with dlopen/dlsym.
//!
//! Nothing here is linked: the same binary runs on a machine without an
//! NVIDIA driver or toolkit, where loading just returns Err. Only the entry
//! points axon calls are resolved, by their versioned ABI names (the ones
//! cuda.h's `#define cuMemAlloc cuMemAlloc_v2` etc. map to), so behaviour
//! doesn't depend on which header a driver was built against. The libraries
//! are loaded once per process and never closed: the tables are `'static`.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUevent = *mut c_void;
pub type NvrtcResult = c_int;
pub type NvrtcProgram = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const NVRTC_SUCCESS: NvrtcResult = 0;
/// `CU_STREAM_NON_BLOCKING`: not ordered with the legacy default stream,
/// which stream capture requires; every copy goes on the stream itself.
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 2;
pub const CU_MEMHOSTREGISTER_READ_ONLY: c_uint = 8;
/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`: capture only fails on unsafe
/// calls made by this thread.
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;

/// `CUmemorytype` values (cuda.h).
pub const CU_MEMORYTYPE_HOST: c_uint = 1;
pub const CU_MEMORYTYPE_DEVICE: c_uint = 2;

/// `CUDA_MEMCPY2D` (cuda.h): `height` rows of `width` bytes, each side at
/// its own pitch.
#[repr(C)]
#[derive(Default)]
pub struct Memcpy2D {
    pub src_x: usize,
    pub src_y: usize,
    pub src_type: c_uint,
    pub src_host: usize,
    pub src_dev: CUdeviceptr,
    pub src_array: usize,
    pub src_pitch: usize,
    pub dst_x: usize,
    pub dst_y: usize,
    pub dst_type: c_uint,
    pub dst_host: usize,
    pub dst_dev: CUdeviceptr,
    pub dst_array: usize,
    pub dst_pitch: usize,
    pub width: usize,
    pub height: usize,
}

/// `CUlaunchAttribute` (cuda.h): an id and a 64-byte value union.
#[repr(C)]
pub struct LaunchAttr {
    pub id: c_uint,
    pub pad: c_uint,
    pub value: [u64; 8],
}

/// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`: the kernel may
/// start before the previous one on the stream ends; it waits for it with
/// `griddepcontrol.wait` (sm_90+).
pub const LAUNCH_ATTR_PROGRAMMATIC: c_uint = 6;

/// `CUlaunchConfig` (cuda.h).
#[repr(C)]
pub struct LaunchConfig {
    pub grid: [c_uint; 3],
    pub block: [c_uint; 3],
    pub shared: c_uint,
    pub stream: CUstream,
    pub attrs: *mut LaunchAttr,
    pub n_attrs: c_uint,
}

/// `CUfunction_attribute` values (cuda.h).
pub const FUNC_ATTR_NUM_REGS: c_int = 4;
pub const FUNC_ATTR_LOCAL_SIZE_BYTES: c_int = 3;
pub const FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

/// `CUdevice_attribute` values used for `DeviceInfo` (cuda.h).
pub const ATTR_SM_COUNT: c_int = 16;
pub const ATTR_MEMORY_CLOCK_KHZ: c_int = 36;
pub const ATTR_MEMORY_BUS_BITS: c_int = 37;
pub const ATTR_CC_MAJOR: c_int = 75;
pub const ATTR_CC_MINOR: c_int = 76;

/// Declares a table of entry points: a struct with one `unsafe extern "C"`
/// fn pointer per symbol, named as the C symbol, and a loader resolving all
/// of them (any missing one is an Err naming it).
macro_rules! api {
    ($(#[$doc:meta])* $table:ident {
        $( $(#[$m:meta])* fn $name:ident($($ty:ty),* $(,)?) -> $ret:ty; )*
    }) => {
        $(#[$doc])*
        #[allow(non_snake_case)]
        pub struct $table {
            $( $(#[$m])* pub $name: unsafe extern "C" fn($($ty),*) -> $ret, )*
        }

        impl $table {
            /// # Safety
            /// `lib` is a live dlopen handle whose symbols have the C
            /// signatures declared in the table.
            unsafe fn resolve(lib: *mut c_void) -> Result<$table, String> {
                Ok($table {
                    $( $name: {
                        let p = sym(lib, concat!(stringify!($name), "\0"))?;
                        // SAFETY: p is the non-null address of this symbol,
                        // whose C signature the table declares (caller's contract).
                        unsafe {
                            std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($ty),*) -> $ret>(p)
                        }
                    }, )*
                })
            }
        }
    };
}

api! {
    /// The CUDA driver API (libcuda.so.1).
    Driver {
        fn cuInit(c_uint) -> CUresult;
        fn cuDriverGetVersion(*mut c_int) -> CUresult;
        fn cuGetErrorName(CUresult, *mut *const c_char) -> CUresult;
        fn cuGetErrorString(CUresult, *mut *const c_char) -> CUresult;
        fn cuDeviceGetCount(*mut c_int) -> CUresult;
        fn cuDeviceGet(*mut CUdevice, c_int) -> CUresult;
        fn cuDeviceGetName(*mut c_char, c_int, CUdevice) -> CUresult;
        fn cuDeviceGetAttribute(*mut c_int, c_int, CUdevice) -> CUresult;
        fn cuDeviceTotalMem_v2(*mut usize, CUdevice) -> CUresult;
        fn cuDevicePrimaryCtxRetain(*mut CUcontext, CUdevice) -> CUresult;
        fn cuDevicePrimaryCtxRelease_v2(CUdevice) -> CUresult;
        fn cuCtxSetCurrent(CUcontext) -> CUresult;
        fn cuCtxSynchronize() -> CUresult;
        #[cfg_attr(not(test), allow(dead_code))] // tests: Drop unbinds
        fn cuCtxGetCurrent(*mut CUcontext) -> CUresult;
        fn cuMemGetInfo_v2(*mut usize, *mut usize) -> CUresult;
        fn cuMemAlloc_v2(*mut CUdeviceptr, usize) -> CUresult;
        fn cuMemFree_v2(CUdeviceptr) -> CUresult;
        fn cuMemcpyHtoDAsync_v2(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
        fn cuMemcpyDtoHAsync_v2(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
        // serve: a sequence's KV rows between caches.
        fn cuMemcpyDtoDAsync_v2(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
        fn cuMemsetD8Async(CUdeviceptr, u8, usize, CUstream) -> CUresult;
        // kvsave: a cache's rows to and from host memory, one strided copy
        // per region.
        fn cuMemcpy2DAsync_v2(*const Memcpy2D, CUstream) -> CUresult;
        fn cuMemHostAlloc(*mut *mut c_void, usize, c_uint) -> CUresult;
        fn cuMemFreeHost(*mut c_void) -> CUresult;
        fn cuMemHostRegister_v2(*mut c_void, usize, c_uint) -> CUresult;
        fn cuMemHostUnregister(*mut c_void) -> CUresult;
        fn cuStreamWaitEvent(CUstream, CUevent, c_uint) -> CUresult;
        fn cuModuleLoadData(*mut CUmodule, *const c_void) -> CUresult;
        fn cuModuleUnload(CUmodule) -> CUresult;
        fn cuModuleGetFunction(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
        // Kernel attributes: the GEMMs' dynamic shared memory over 48 KiB.
        fn cuFuncSetAttribute(CUfunction, c_int, c_int) -> CUresult;
        fn cuFuncGetAttribute(*mut c_int, c_int, CUfunction) -> CUresult;
        fn cuLaunchKernelEx(
            *const LaunchConfig,
            CUfunction,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CUresult;
        fn cuStreamCreate(*mut CUstream, c_uint) -> CUresult;
        fn cuStreamDestroy_v2(CUstream) -> CUresult;
        fn cuStreamSynchronize(CUstream) -> CUresult;
        // CUDA graphs: the captured decode step.
        fn cuStreamBeginCapture_v2(CUstream, c_int) -> CUresult;
        fn cuStreamEndCapture(CUstream, *mut CUgraph) -> CUresult;
        fn cuGraphInstantiateWithFlags(*mut CUgraphExec, CUgraph, u64) -> CUresult;
        fn cuGraphLaunch(CUgraphExec, CUstream) -> CUresult;
        fn cuGraphExecDestroy(CUgraphExec) -> CUresult;
        fn cuGraphDestroy(CUgraph) -> CUresult;
        // Events, to time kernels (tests and benchmarks).
        fn cuEventCreate(*mut CUevent, c_uint) -> CUresult;
        fn cuEventRecord(CUevent, CUstream) -> CUresult;
        #[cfg_attr(not(test), allow(dead_code))]
        fn cuEventSynchronize(CUevent) -> CUresult;
        #[cfg_attr(not(test), allow(dead_code))]
        fn cuEventElapsedTime(*mut f32, CUevent, CUevent) -> CUresult;
        fn cuEventDestroy_v2(CUevent) -> CUresult;
        // Pipeline parallel (cuda::qwen2::pipeline): hidden states between
        // devices, over P2P when the devices allow it, else staged by the
        // driver through host memory.
        fn cuDeviceCanAccessPeer(*mut c_int, CUdevice, CUdevice) -> CUresult;
        fn cuCtxEnablePeerAccess(CUcontext, c_uint) -> CUresult;
        fn cuMemcpyPeerAsync(
            CUdeviceptr,
            CUcontext,
            CUdeviceptr,
            CUcontext,
            usize,
            CUstream,
        ) -> CUresult;
    }
}

/// `CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`: `cuCtxEnablePeerAccess` on a
/// pair that already has it (not a failure).
pub const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: CUresult = 704;

api! {
    /// NVRTC, CUDA's run-time compiler (libnvrtc.so).
    Nvrtc {
        fn nvrtcVersion(*mut c_int, *mut c_int) -> NvrtcResult;
        fn nvrtcGetNumSupportedArchs(*mut c_int) -> NvrtcResult;
        fn nvrtcGetSupportedArchs(*mut c_int) -> NvrtcResult;
        fn nvrtcGetErrorString(NvrtcResult) -> *const c_char;
        fn nvrtcCreateProgram(
            *mut NvrtcProgram,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> NvrtcResult;
        fn nvrtcCompileProgram(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;
        fn nvrtcGetProgramLogSize(NvrtcProgram, *mut usize) -> NvrtcResult;
        fn nvrtcGetProgramLog(NvrtcProgram, *mut c_char) -> NvrtcResult;
        fn nvrtcGetPTXSize(NvrtcProgram, *mut usize) -> NvrtcResult;
        fn nvrtcGetPTX(NvrtcProgram, *mut c_char) -> NvrtcResult;
        fn nvrtcGetCUBINSize(NvrtcProgram, *mut usize) -> NvrtcResult;
        fn nvrtcGetCUBIN(NvrtcProgram, *mut c_char) -> NvrtcResult;
        fn nvrtcDestroyProgram(*mut NvrtcProgram) -> NvrtcResult;
    }
}

#[cfg(target_os = "linux")]
mod dl {
    use std::ffi::{c_char, c_int, c_void};

    pub const RTLD_NOW: c_int = 2;

    unsafe extern "C" {
        // glibc's dynamic loader interface (in libc proper since 2.34; std
        // links libdl on older glibc).
        pub fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        pub fn dlerror() -> *mut c_char;
    }
}

/// The last dlerror() message, or a fallback.
#[cfg(target_os = "linux")]
fn dl_error() -> String {
    // SAFETY: dlerror returns null or a NUL-terminated string valid until the
    // next dl* call on this thread; it's copied out at once.
    unsafe {
        let e = dl::dlerror();
        if e.is_null() {
            "unknown dlerror".into()
        } else {
            CStr::from_ptr(e).to_string_lossy().into_owned()
        }
    }
}

/// dlopen the first of `names` that loads; Err lists why each one failed.
#[cfg(target_os = "linux")]
pub fn open_lib(names: &[String]) -> Result<*mut c_void, String> {
    let mut why = Vec::new();
    for n in names {
        let Ok(c) = std::ffi::CString::new(n.as_str()) else {
            continue;
        };
        // SAFETY: c is NUL-terminated. Loading runs the library's
        // initializers, which for libcuda/libnvrtc have no preconditions.
        let h = unsafe { dl::dlopen(c.as_ptr(), dl::RTLD_NOW) };
        if !h.is_null() {
            return Ok(h);
        }
        why.push(dl_error());
    }
    Err(why.join("; "))
}

#[cfg(not(target_os = "linux"))]
pub fn open_lib(_names: &[String]) -> Result<*mut c_void, String> {
    Err(crate::cuda::LINUX_ONLY.into())
}

/// Address of `name` (NUL-terminated) in `lib`, Err if it isn't exported.
#[cfg(target_os = "linux")]
fn sym(lib: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    // SAFETY: lib is a live handle (callers), name is NUL-terminated (the
    // api! macro appends it).
    let p = unsafe { dl::dlsym(lib, name.as_ptr().cast()) };
    if p.is_null() {
        return Err(format!(
            "missing symbol {}: {}",
            name.trim_end_matches('\0'),
            dl_error()
        ));
    }
    Ok(p)
}

#[cfg(not(target_os = "linux"))]
fn sym(_lib: *mut c_void, _name: &str) -> Result<*mut c_void, String> {
    Err(crate::cuda::LINUX_ONLY.into())
}

/// The driver API, initialized (`cuInit(0)` succeeded), or why not. The
/// tables hold only fn pointers, so they're Send + Sync as they are; the
/// library handles are never closed, so the pointers stay valid.
pub fn driver() -> Result<&'static Driver, String> {
    static D: OnceLock<Result<Driver, String>> = OnceLock::new();
    let l = D.get_or_init(|| {
        let lib =
            open_lib(&["libcuda.so.1".into()]).map_err(|e| format!("no NVIDIA driver: {e}"))?;
        // SAFETY: lib is the CUDA driver, whose exports have the
        // signatures in `Driver` (cuda.h, versioned names).
        let d = unsafe { Driver::resolve(lib)? };
        // SAFETY: cuInit with flags 0 is the documented first call.
        let r = unsafe { (d.cuInit)(0) };
        check(&d, r, "cuInit")?;
        Ok(d)
    });
    l.as_ref().map_err(Clone::clone)
}

/// The CUDA version the driver supports, (major, minor).
pub fn driver_version(d: &Driver) -> Result<(i32, i32), String> {
    let mut v: c_int = 0;
    // SAFETY: writes one c_int.
    let r = unsafe { (d.cuDriverGetVersion)(&mut v) };
    check(d, r, "cuDriverGetVersion")?;
    Ok((v / 1000, v % 1000 / 10))
}

/// cuTensorMapEncodeTiled (CUDA 12.0+): fills a 128-byte CUtensorMap.
pub type TensorMapEncodeTiled = unsafe extern "C" fn(
    *mut [u64; 16], // tensorMap (64-byte aligned)
    c_uint,         // CUtensorMapDataType
    c_uint,         // tensorRank
    *mut c_void,    // globalAddress
    *const u64,     // globalDim[rank]
    *const u64,     // globalStrides[rank - 1] (bytes)
    *const c_uint,  // boxDim[rank]
    *const c_uint,  // elementStrides[rank]
    c_uint,         // CUtensorMapInterleave
    c_uint,         // CUtensorMapSwizzle
    c_uint,         // CUtensorMapL2promotion
    c_uint,         // CUtensorMapFloatOOBfill
) -> CUresult;

pub const TENSOR_MAP_DATA_TYPE_UINT8: c_uint = 0;
pub const TENSOR_MAP_SWIZZLE_NONE: c_uint = 0;
pub const TENSOR_MAP_SWIZZLE_128B: c_uint = 3;
pub const TENSOR_MAP_L2_PROMOTION_256B: c_uint = 3;

/// The driver's cuTensorMapEncodeTiled (TMA descriptors for the Hopper
/// kernels), None if the driver doesn't export it (older than CUDA 12.0):
/// optional, unlike `Driver`'s entry points, whose absence is an error.
pub fn tensor_map_encoder() -> Option<TensorMapEncodeTiled> {
    static F: OnceLock<Option<TensorMapEncodeTiled>> = OnceLock::new();
    *F.get_or_init(|| {
        driver().ok()?;
        // The loaded driver's handle again (dlopen refcounts; never closed).
        let lib = open_lib(&["libcuda.so.1".into()]).ok()?;
        let p = sym(lib, "cuTensorMapEncodeTiled\0").ok()?;
        // SAFETY: p is the driver's cuTensorMapEncodeTiled, whose C
        // signature (cuda.h) TensorMapEncodeTiled declares.
        Some(unsafe { std::mem::transmute::<*mut c_void, TensorMapEncodeTiled>(p) })
    })
}

/// Where libnvrtc may be: the soname matching the driver's CUDA major
/// first (a newer NVRTC's output may not load on an older driver), then the
/// dev symlink and the other sonames (CUDA 13, 12); each on the library
/// path, then in the toolkit's lib64 under $CUDA_PATH and the usual prefixes.
fn nvrtc_candidates(driver_major: Option<i32>) -> Vec<String> {
    let mut names: Vec<String> = driver_major
        .map(|m| format!("libnvrtc.so.{m}"))
        .into_iter()
        .collect();
    for n in ["libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"] {
        if !names.iter().any(|x| x == n) {
            names.push(n.into());
        }
    }
    let mut dirs = Vec::new();
    if let Some(p) = std::env::var_os("CUDA_PATH") {
        dirs.push(std::path::PathBuf::from(p).join("lib64"));
    }
    dirs.push("/opt/cuda/lib64".into());
    dirs.push("/usr/local/cuda/lib64".into());
    // By name first: the driver's soname anywhere beats a mismatched one.
    let mut v = Vec::new();
    for n in &names {
        v.push(n.clone());
        v.extend(
            dirs.iter()
                .map(|d| d.join(n).to_string_lossy().into_owned()),
        );
    }
    v
}

/// NVRTC, or why not: the first library on `nvrtc_candidates` that loads.
pub fn nvrtc() -> Result<&'static Nvrtc, String> {
    nvrtc_where(|_| Ok(()))
}

/// The first NVRTC on `nvrtc_candidates` (in its order) that `ok` accepts.
/// Each distinct library is loaded once, only when every one before it was
/// refused (so with CUDA 13 and 12 toolkits both installed, a GPU NVRTC 13
/// dropped still gets the 12), and never closed. `ok` must not call back
/// into `nvrtc`. Err: "NVRTC not found: .." (why each name didn't load) if
/// none loads, else every loaded one's refusal, then why the others didn't
/// load ("not loaded: ..").
pub fn nvrtc_where(ok: impl Fn(&Nvrtc) -> Result<(), String>) -> Result<&'static Nvrtc, String> {
    /// The candidates, the next one to try, those loaded (by handle) and
    /// why the others didn't load.
    struct Libs {
        names: Vec<String>,
        next: usize,
        loaded: Vec<(usize, &'static Nvrtc)>,
        why: Vec<String>,
    }
    static LIBS: Mutex<Option<Libs>> = Mutex::new(None);
    let mut g = LIBS.lock().unwrap_or_else(|e| e.into_inner());
    let libs = g.get_or_insert_with(|| Libs {
        names: nvrtc_candidates(driver().and_then(driver_version).ok().map(|v| v.0)),
        next: 0,
        loaded: Vec::new(),
        why: Vec::new(),
    });
    let mut refused = Vec::new();
    for i in 0.. {
        // Load the next distinct library when the loaded ones are used up.
        while i == libs.loaded.len() && libs.next < libs.names.len() {
            let name = &libs.names[libs.next];
            libs.next += 1;
            let lib = match open_lib(std::slice::from_ref(name)) {
                Ok(h) => h,
                Err(e) => {
                    libs.why.push(e);
                    continue;
                }
            };
            // The same file by another name (the dev symlink, a path).
            if libs.loaded.iter().any(|&(h, _)| h == lib as usize) {
                continue;
            }
            // SAFETY: lib is NVRTC, whose exports have the signatures in `Nvrtc`.
            match unsafe { Nvrtc::resolve(lib) } {
                Ok(n) => libs.loaded.push((lib as usize, Box::leak(Box::new(n)))),
                Err(e) => libs.why.push(format!("{name}: {e}")),
            }
        }
        let Some(&(_, n)) = libs.loaded.get(i) else {
            break;
        };
        match ok(n) {
            Ok(()) => return Ok(n),
            Err(e) => refused.push(e),
        }
    }
    if libs.loaded.is_empty() {
        return Err(format!("NVRTC not found: {}", libs.why.join("; ")));
    }
    let mut e = refused.join("; ");
    if !libs.why.is_empty() {
        // Where else it looked, so a refusal shows what was searched.
        e.push_str(&format!("; not loaded: {}", libs.why.join("; ")));
    }
    Err(e)
}

/// NVRTC's version, (major, minor).
pub fn nvrtc_version(n: &Nvrtc) -> Result<(i32, i32), String> {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: writes two c_ints.
    let r = unsafe { (n.nvrtcVersion)(&mut major, &mut minor) };
    check_nvrtc(n, r, "nvrtcVersion")?;
    Ok((major, minor))
}

/// The architectures NVRTC compiles for, as 10 * major + minor (75, 120, ...).
pub fn nvrtc_archs(n: &Nvrtc) -> Result<Vec<i32>, String> {
    let mut k: c_int = 0;
    // SAFETY: writes one c_int.
    let r = unsafe { (n.nvrtcGetNumSupportedArchs)(&mut k) };
    check_nvrtc(n, r, "nvrtcGetNumSupportedArchs")?;
    let mut v = vec![0 as c_int; k.max(0) as usize];
    if !v.is_empty() {
        // SAFETY: v has the k entries nvrtcGetNumSupportedArchs reported.
        let r = unsafe { (n.nvrtcGetSupportedArchs)(v.as_mut_ptr()) };
        check_nvrtc(n, r, "nvrtcGetSupportedArchs")?;
    }
    Ok(v)
}

/// Ok, or Err("call: NAME: description") for a failed driver call.
pub fn check(d: &Driver, r: CUresult, call: &str) -> Result<(), String> {
    if r == CUDA_SUCCESS {
        return Ok(());
    }
    let text = |f: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult| {
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: f writes a pointer to a static NUL-terminated string, or
        // fails on an unknown code (p stays null).
        let ok = unsafe { f(r, &mut p) } == CUDA_SUCCESS && !p.is_null();
        // SAFETY: on success p points to a static C string.
        ok.then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    };
    let name = text(d.cuGetErrorName).unwrap_or_else(|| format!("CUresult {r}"));
    match text(d.cuGetErrorString) {
        Some(s) => Err(format!("{call}: {name}: {s}")),
        None => Err(format!("{call}: {name}")),
    }
}

/// Ok, or Err("call: description") for a failed NVRTC call.
pub fn check_nvrtc(n: &Nvrtc, r: NvrtcResult, call: &str) -> Result<(), String> {
    if r == NVRTC_SUCCESS {
        return Ok(());
    }
    // SAFETY: nvrtcGetErrorString returns a static string for any code.
    let p = unsafe { (n.nvrtcGetErrorString)(r) };
    let s = if p.is_null() {
        format!("nvrtcResult {r}")
    } else {
        // SAFETY: non-null, static and NUL-terminated.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    };
    Err(format!("{call}: {s}"))
}
