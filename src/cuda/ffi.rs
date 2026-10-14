//! The CUDA driver API and NVRTC, loaded at run time with dlopen/dlsym.
//!
//! Nothing here is linked: the same binary runs on a machine without an
//! NVIDIA driver or toolkit, where loading just returns Err. Only the entry
//! points axon calls are resolved, by their versioned ABI names (the ones
//! cuda.h's `#define cuMemAlloc cuMemAlloc_v2` etc. map to), so behaviour
//! doesn't depend on which header a driver was built against. The libraries
//! are loaded once per process and never closed: the tables are `'static`.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type NvrtcResult = c_int;
pub type NvrtcProgram = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const NVRTC_SUCCESS: NvrtcResult = 0;
/// `CU_STREAM_DEFAULT`: a blocking stream, ordered with the legacy default
/// stream, so the synchronous copies and memset (which run there) are ordered
/// with the kernels launched on it.
pub const CU_STREAM_DEFAULT: c_uint = 0;

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
        #[cfg_attr(not(test), allow(dead_code))] // tests: Drop unbinds
        fn cuCtxGetCurrent(*mut CUcontext) -> CUresult;
        fn cuMemGetInfo_v2(*mut usize, *mut usize) -> CUresult;
        fn cuMemAlloc_v2(*mut CUdeviceptr, usize) -> CUresult;
        fn cuMemFree_v2(CUdeviceptr) -> CUresult;
        fn cuMemcpyHtoD_v2(CUdeviceptr, *const c_void, usize) -> CUresult;
        fn cuMemcpyDtoH_v2(*mut c_void, CUdeviceptr, usize) -> CUresult;
        // Stream-ordered copies, for G2's decode loop (pinned host buffers).
        #[allow(dead_code)]
        fn cuMemcpyHtoDAsync_v2(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
        #[allow(dead_code)]
        fn cuMemcpyDtoHAsync_v2(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
        fn cuMemsetD8_v2(CUdeviceptr, u8, usize) -> CUresult;
        fn cuModuleLoadData(*mut CUmodule, *const c_void) -> CUresult;
        fn cuModuleUnload(CUmodule) -> CUresult;
        fn cuModuleGetFunction(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
        fn cuLaunchKernel(
            CUfunction,
            c_uint, c_uint, c_uint,
            c_uint, c_uint, c_uint,
            c_uint,
            CUstream,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CUresult;
        fn cuStreamCreate(*mut CUstream, c_uint) -> CUresult;
        fn cuStreamDestroy_v2(CUstream) -> CUresult;
        fn cuStreamSynchronize(CUstream) -> CUresult;
        // CUDA graphs, for G2's captured decode step: resolved now so a
        // driver without them fails at open, not mid-generation.
        #[allow(dead_code)]
        fn cuStreamBeginCapture_v2(CUstream, c_int) -> CUresult;
        #[allow(dead_code)]
        fn cuStreamEndCapture(CUstream, *mut CUgraph) -> CUresult;
        #[allow(dead_code)]
        fn cuGraphInstantiateWithFlags(*mut CUgraphExec, CUgraph, u64) -> CUresult;
        #[allow(dead_code)]
        fn cuGraphLaunch(CUgraphExec, CUstream) -> CUresult;
        #[allow(dead_code)]
        fn cuGraphExecDestroy(CUgraphExec) -> CUresult;
        #[allow(dead_code)]
        fn cuGraphDestroy(CUgraph) -> CUresult;
    }
}

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

/// NVRTC, or why not.
pub fn nvrtc() -> Result<&'static Nvrtc, String> {
    static N: OnceLock<Result<Nvrtc, String>> = OnceLock::new();
    let l = N.get_or_init(|| {
        let major = driver().and_then(driver_version).ok().map(|v| v.0);
        let lib =
            open_lib(&nvrtc_candidates(major)).map_err(|e| format!("NVRTC not found: {e}"))?;
        // SAFETY: lib is NVRTC, whose exports have the signatures in `Nvrtc`.
        unsafe { Nvrtc::resolve(lib) }
    });
    l.as_ref().map_err(Clone::clone)
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
