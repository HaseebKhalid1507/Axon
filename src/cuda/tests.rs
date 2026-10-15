use super::{
    Arg, CacheStatus, DevBuf, Gpu, KERNEL_OPTS, KERNELS, Kernels, Module, Target, cache_dir,
    cache_dir_from, check_versions, compile, encode_entry, ffi, pdl_ok, pick_target,
};
use crate::gguf::Model;
use crate::quant::{self, F16, F32, Q4_K, Q5_0, Q6_K, Q8_0};
use std::path::PathBuf;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The first index where `got` and `want` differ in bits, if any.
fn first_diff(got: &[f32], want: &[f32]) -> Option<usize> {
    assert_eq!(got.len(), want.len());
    (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits())
}

/// GPU dequant of `rows` of a matrix (bytes `data`, type `qt`, `n` per row)
/// must equal `quant::dequant` of each row bit for bit.
fn assert_rows_match(
    g: &Gpu,
    k: &Kernels,
    what: &str,
    qt: u32,
    data: &[u8],
    n: usize,
    rows: &[u32],
) {
    let w: DevBuf = g.upload(data).unwrap();
    let got = k.dequant_rows(qt, &w, n, rows).unwrap();
    let rb = quant::row_bytes(qt, n);
    let mut want = vec![0f32; n];
    for (j, &r) in rows.iter().enumerate() {
        let r = r as usize;
        quant::dequant(qt, &data[r * rb..(r + 1) * rb], &mut want);
        if let Some(i) = first_diff(&got[j * n..(j + 1) * n], &want) {
            panic!(
                "{what} ({}) row {r} value {i}: gpu {:e} ({:#x}) vs cpu {:e} ({:#x})",
                crate::gguf::qtype(qt).name,
                got[j * n + i],
                got[j * n + i].to_bits(),
                want[i],
                want[i].to_bits()
            );
        }
    }
}

/// First, last and 6 random rows of every tensor of `m`; returns (tensors,
/// rows) checked and the types seen.
fn check_every_tensor(g: &Gpu, m: &Model) -> (usize, usize, Vec<&'static str>) {
    let km = Kernels::module(g).unwrap();
    let k = Kernels::new(&km).unwrap();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (mut rows_checked, mut types) = (0, Vec::new());
    for t in &m.tensors {
        let n = t.dims[0] as usize;
        let n_rows = (t.n_elements as usize / n) as u32;
        let mut rows = vec![0, n_rows - 1];
        rows.extend((0..6).map(|_| (rng.next() % n_rows as u64) as u32));
        rows.sort_unstable();
        rows.dedup();
        assert_rows_match(g, &k, &t.name, t.qtype.id, m.tensor_data(t), n, &rows);
        rows_checked += rows.len();
        if !types.contains(&t.qtype.name) {
            types.push(t.qtype.name);
        }
    }
    (m.tensors.len(), rows_checked, types)
}

#[test]
fn gpu_dequant_bitwise_every_tensor() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let (tensors, rows, types) = check_every_tensor(&g, &m);
    eprintln!("{tensors} tensors, {rows} rows bit-identical; types {types:?}");
    for want in ["Q4_K", "Q6_K", "Q5_0", "Q8_0", "F32"] {
        assert!(types.contains(&want), "{want} not covered: {types:?}");
    }
}

/// The same on the 1.5B (other shapes: 1536 wide, 8960 FFN), if it's there.
#[test]
fn gpu_dequant_bitwise_every_tensor_1_5b() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let m = crate::gguf::load_with(path.to_str().unwrap(), Some(false)).unwrap();
    let (tensors, rows, types) = check_every_tensor(&g, &m);
    eprintln!("1.5B: {tensors} tensors, {rows} rows bit-identical; types {types:?}");
}

/// Every type on random blocks: all 65536 f16 values (NaN payloads,
/// subnormals, -0), random f32 bits, and quant blocks of random bytes with
/// finite scales. (NaN/inf scales are left out: x86 and NVIDIA give NaNs
/// from arithmetic different payloads, and no real weight has them.)
#[test]
fn gpu_dequant_bitwise_synthetic() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let k = Kernels::new(&km).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    let all_f16: Vec<u8> = (0..=u16::MAX).flat_map(|h| h.to_le_bytes()).collect();
    assert_rows_match(&g, &k, "all f16", F16, &all_f16, 1 << 16, &[0]);

    let mut f32s: Vec<u8> = (0..4096)
        .flat_map(|_| (rng.next() as u32).to_le_bytes())
        .collect();
    for (i, v) in [
        f32::NAN,
        -0.0,
        f32::INFINITY,
        1e-45,
        f32::from_bits(0x7fa0_0001),
    ]
    .iter()
    .enumerate()
    {
        f32s[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    assert_rows_match(&g, &k, "random f32", F32, &f32s, 1024, &[0, 1, 2, 3]);

    // (type, offsets of the block's f16 scales)
    let quants: [(u32, &[usize]); 4] =
        [(Q5_0, &[0]), (Q8_0, &[0]), (Q4_K, &[0, 2]), (Q6_K, &[208])];
    for (qt, scales) in quants {
        let q = crate::gguf::qtype(qt);
        let (n, n_rows) = (1024, 64u32);
        let mut data: Vec<u8> = (0..quant::row_bytes(qt, n) * n_rows as usize)
            .map(|_| rng.next() as u8)
            .collect();
        for blk in data.chunks_mut(q.block_bytes) {
            for &o in scales {
                if blk[o + 1] & 0x7c == 0x7c {
                    blk[o + 1] &= !0x40; // exponent 31 -> 15
                }
            }
        }
        let rows: Vec<u32> = (0..n_rows).rev().collect();
        assert_rows_match(&g, &k, "random blocks", qt, &data, n, &rows);
    }
}

#[test]
fn gpu_dequant_rejects_bad_requests() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let k = Kernels::new(&km).unwrap();
    let w = g.upload(&[0u8; 144 * 2]).unwrap();
    assert!(k.dequant_rows(Q4_K, &w, 256, &[0, 1]).is_ok());
    assert!(
        k.dequant_rows(Q4_K, &w, 256, &[2])
            .unwrap_err()
            .contains("past")
    );
    assert!(k.dequant_rows(Q4_K, &w, 100, &[0]).is_err());
    assert!(
        k.dequant_rows(2, &w, 32, &[0])
            .unwrap_err()
            .contains("Q4_0")
    );
}

#[test]
fn gpu_buffers_round_trip() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let v: Vec<u32> = (0..1000u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    let b = g.upload(&v).unwrap();
    assert_eq!(b.len(), 4000);
    assert_eq!(b.read::<u32>().unwrap(), v);
    b.zero().unwrap();
    assert!(b.read::<f32>().unwrap().iter().all(|x| x.to_bits() == 0));
    assert!(b.write(&[0u8; 4001]).is_err());
    assert!(g.alloc(6).unwrap().read::<f32>().is_err());
    let empty = g.alloc(0).unwrap();
    assert_eq!(empty.read::<u8>().unwrap(), Vec::<u8>::new());
}

#[test]
fn gpu_info_is_sane() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let i = g.info().unwrap();
    eprintln!("{i:?} -> {}", super::summary());
    assert!(!i.name.is_empty());
    assert!(i.cc.0 >= 3 && i.sms > 0);
    assert!(i.free_bytes > 0 && i.free_bytes <= i.total_bytes);
    assert!(super::summary().starts_with(&format!("gpu: {} | {}", i.name, i.arch())));
}

/// Opening what isn't there is an Err with a reason, never a panic: a
/// missing library, a device past the last (or no driver at all).
#[test]
fn gpu_open_errors_are_graceful() {
    let e = ffi::open_lib(&["libaxon-no-such-lib.so.1".into()]).unwrap_err();
    if cfg!(target_os = "linux") {
        assert!(e.contains("libaxon-no-such-lib.so.1"), "{e}");
    }
    let e = Gpu::open_device(1 << 20).err().expect("no device 1048576");
    assert!(!e.is_empty());
    eprintln!("open_device(1 << 20): {e}");
}

#[test]
fn gpu_compile_error_shows_log() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let src = "extern \"C\" __global__ void k(float* p) { p[0] = undefined_thing; }";
    let e = Module::load_in(&g, src, &[], None)
        .err()
        .expect("compile error");
    assert!(
        e.contains("nvrtcCompileProgram") && e.contains("undefined_thing"),
        "{e}"
    );
}

/// A scratch cache dir under the real one, removed on drop.
struct TempCache(PathBuf);
impl TempCache {
    fn new(name: &str) -> TempCache {
        let base = cache_dir()
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/axon-cuda"));
        let d = base.join(format!("test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        TempCache(d)
    }
}
impl Drop for TempCache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn gpu_kernel_cache_hit_miss_corrupt() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let dir = TempCache::new("cache");
    let src = "extern \"C\" __global__ void iota(float* p, float s, unsigned n) { if (threadIdx.x < n) p[threadIdx.x] = s * threadIdx.x; }";
    let load = || {
        let (m, st) = Module::load_in(&g, src, &["--std=c++17"], Some(&dir.0)).unwrap();
        // The module works, whichever way it came.
        let f = m.function("iota").unwrap();
        let out = g.alloc(32 * 4).unwrap();
        let args = [out.arg(), super::Arg::F32(0.5), super::Arg::U32(32)];
        // SAFETY: iota(float*, float, unsigned) writes p[0..32] for n = 32.
        unsafe { g.launch(&f, (1, 1, 1), (64, 1, 1), 0, &args).unwrap() };
        g.sync().unwrap();
        let v: Vec<f32> = out.read().unwrap();
        assert!(
            v.iter().enumerate().all(|(i, &x)| x == 0.5 * i as f32),
            "{v:?}"
        );
        st
    };
    let entry = || {
        let files: Vec<PathBuf> = std::fs::read_dir(&dir.0)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(files.len(), 1, "one entry, no temp files left: {files:?}");
        files.into_iter().next().unwrap()
    };
    assert_eq!(load(), CacheStatus::Miss);
    assert_eq!(load(), CacheStatus::Hit);
    let path = entry();
    assert!(path.extension().is_some_and(|e| e == "cubin"));

    // A flipped byte, a truncated file, a good checksum over something that
    // isn't a cubin: each is recompiled and replaced.
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0x55;
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(load(), CacheStatus::Corrupt);
    assert_eq!(load(), CacheStatus::Hit);
    std::fs::write(&path, &bytes[..5]).unwrap();
    assert_eq!(load(), CacheStatus::Corrupt);
    assert_eq!(load(), CacheStatus::Hit);
    std::fs::write(&path, encode_entry(b"\x7fELF not really a cubin")).unwrap();
    assert_eq!(load(), CacheStatus::Corrupt);
    assert_eq!(load(), CacheStatus::Hit);
    assert_eq!(entry(), path);

    // Unreadable (a directory where the file goes), and not replaceable:
    // still compiles every time.
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir_all(path.join("x")).unwrap();
    assert_eq!(load(), CacheStatus::Corrupt);
    assert_eq!(load(), CacheStatus::Corrupt);
    assert_eq!(entry(), path);

    // Other options are another entry.
    let (_, st) = Module::load_in(&g, src, &[], Some(&dir.0)).unwrap();
    assert_eq!(st, CacheStatus::Miss);
    let (_, st) = Module::load_in(&g, src, &[], None).unwrap();
    assert_eq!(st, CacheStatus::Off);
}

const IOTA: &str = "extern \"C\" __global__ void iota(float* p, float s, unsigned n) \
                    { if (threadIdx.x < n) p[threadIdx.x] = s * threadIdx.x; }";

/// Run `iota` from `m` and check what it wrote.
fn run_iota(g: &Gpu, m: &Module) {
    let f = m.function("iota").unwrap();
    let out = g.alloc(32 * 4).unwrap();
    // SAFETY: iota(float*, float, unsigned) writes p[0..32] for n = 32.
    unsafe {
        g.launch(
            &f,
            (1, 1, 1),
            (32, 1, 1),
            0,
            &[out.arg(), Arg::F32(2.0), Arg::U32(32)],
        )
    }
    .unwrap();
    let v: Vec<f32> = out.read().unwrap();
    assert!(
        v.iter().enumerate().all(|(i, &x)| x == 2.0 * i as f32),
        "{v:?}"
    );
}

/// The whole kernel module compiles (NVRTC, no GPU needed) for the oldest
/// architectures NVRTC still supports - sm_70 and sm_75, where it has them -
/// which take the paths without tensor cores (the split attention, gemvn),
/// and for sm_80 (the oldest with them). Skipped without NVRTC.
#[test]
fn nvrtc_kernels_compile_for_old_archs() {
    let n = match ffi::nvrtc() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let archs = ffi::nvrtc_archs(n).unwrap();
    let want: Vec<i32> = [70, 75, 80]
        .into_iter()
        .filter(|a| archs.contains(a))
        .collect();
    assert!(want.contains(&80), "NVRTC archs {archs:?}");
    for a in want {
        let t = Target::Cubin(a);
        let mut opts = vec![format!("--gpu-architecture={t}")];
        opts.extend(KERNEL_OPTS.iter().map(|s| s.to_string()));
        let image = compile(KERNELS, &opts, t).unwrap_or_else(|e| panic!("sm_{a}: {e}"));
        eprintln!("sm_{a}: {} bytes of cubin", image.len());
    }
}

#[test]
fn gpu_target_choice() {
    let archs = [75, 80, 86, 87, 89, 90, 100, 103, 110, 120, 121];
    let v = (13, 4);
    assert_eq!(pick_target(120, &archs, v), Ok(Target::Cubin(120)));
    // A newer minor of a known major: the older minor's cubin runs on it.
    assert_eq!(pick_target(88, &archs, v), Ok(Target::Cubin(87)));
    assert_eq!(pick_target(122, &archs, v), Ok(Target::Cubin(121)));
    // A newer major than NVRTC knows: PTX for the driver to compile.
    assert_eq!(pick_target(130, &archs, v), Ok(Target::Ptx(121)));
    assert_eq!(pick_target(130, &[75, 80], v), Ok(Target::Ptx(80)));
    // Older than anything NVRTC supports.
    let e = pick_target(61, &archs, v).unwrap_err();
    assert!(
        e.contains("sm_61") && e.contains("sm_75") && e.contains("13.4"),
        "{e}"
    );
    assert_eq!(Target::Ptx(89).to_string(), "compute_89");

    assert!(check_versions((13, 4), (13, 4)).is_ok());
    assert!(check_versions((13, 0), (13, 4)).is_ok());
    assert!(check_versions((13, 2), (12, 8)).is_ok());
    let e = check_versions((12, 4), (13, 4)).unwrap_err();
    assert!(
        e.starts_with("NVRTC 13.4 needs a CUDA 13 driver, have 12.4"),
        "{e}"
    );
}

#[test]
fn gpu_cache_dir_follows_xdg() {
    let os = |s: &str| Some(std::ffi::OsString::from(s));
    let want = |s: &str| Some(PathBuf::from(s));
    assert_eq!(cache_dir_from(os("/x/c"), os("/h")), want("/x/c/axon/cuda"));
    // Relative or empty $XDG_CACHE_HOME is ignored (XDG spec).
    assert_eq!(
        cache_dir_from(os("rel/c"), os("/h")),
        want("/h/.cache/axon/cuda")
    );
    assert_eq!(
        cache_dir_from(os(""), os("/h")),
        want("/h/.cache/axon/cuda")
    );
    assert_eq!(cache_dir_from(None, os("/h")), want("/h/.cache/axon/cuda"));
    assert_eq!(cache_dir_from(None, os("")), None);
    assert_eq!(cache_dir_from(os("rel"), None), None);
}

/// `store` makes the directory 0700, leaves no temp file, and sweeps temp
/// files older than an hour (and only those).
#[test]
fn gpu_cache_store_hygiene() {
    let dir = TempCache::new("store");
    let d = dir.0.join("sub");
    super::store(&d.join("a.cubin"), b"one");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&d).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700, "{mode:o}");
    }
    let (stale, fresh) = (d.join(".tmp.1.1"), d.join(".tmp.2.2"));
    std::fs::write(&stale, b"x").unwrap();
    std::fs::write(&fresh, b"x").unwrap();
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600);
    std::fs::File::options()
        .write(true)
        .open(&stale)
        .unwrap()
        .set_modified(old)
        .unwrap();
    super::store(&d.join("b.cubin"), b"two");
    let mut names: Vec<String> = std::fs::read_dir(&d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, [".tmp.2.2", "a.cubin", "b.cubin"]);
    assert_eq!(std::fs::read(d.join("b.cubin")).unwrap(), b"two");
}

/// Code the driver won't load is an Err and never cached; code that loads
/// is stored. PTX for an older architecture goes through the driver's
/// compiler and caches like a cubin.
#[test]
fn gpu_only_loadable_code_is_cached() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let dir = TempCache::new("loadable");
    let cc = g.info().unwrap().cc;
    let cc = (cc.0 * 10 + cc.1) as i32;
    let archs = ffi::nvrtc_archs(ffi::nvrtc().unwrap()).unwrap();
    let files = || std::fs::read_dir(&dir.0).map_or(0, |r| r.count());

    // A cubin of another major can't run here.
    let other = *archs.iter().find(|&&a| a / 10 != cc / 10).unwrap();
    let e = Module::load_for(&g, IOTA, &[], Some(&dir.0), Target::Cubin(other))
        .err()
        .expect("foreign cubin loads");
    assert!(
        e.contains("cuModuleLoadData") && e.contains(&format!("sm_{other}")),
        "{e}"
    );
    assert_eq!(files(), 0, "an unloadable cubin was cached");

    let ptx = *archs.iter().filter(|&&a| a < cc).max().unwrap();
    for want in [CacheStatus::Miss, CacheStatus::Hit] {
        let (m, st) = Module::load_for(&g, IOTA, &[], Some(&dir.0), Target::Ptx(ptx)).unwrap();
        assert_eq!(st, want, "compute_{ptx}");
        run_iota(&g, &m);
    }
    assert_eq!(files(), 1);
}

/// A kernel only launches on the `Gpu` its module is on; dropping the last
/// `Gpu` leaves no context current on the thread.
#[test]
fn gpu_launch_checks_owner_and_drop_unbinds() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let h = Gpu::open().unwrap();
    {
        let (m, _) = Module::load_in(&g, IOTA, &[], None).unwrap();
        run_iota(&g, &m);
        let f = m.function("iota").unwrap();
        // SAFETY: never runs: the owner check fails first.
        let e = unsafe { h.launch(&f, (1, 1, 1), (1, 1, 1), 0, &[]) }.unwrap_err();
        assert!(e.contains("another Gpu"), "{e}");
    }
    drop(h);
    drop(g);
    let d = ffi::driver().unwrap();
    let mut ctx: ffi::CUcontext = std::ptr::dangling_mut();
    // SAFETY: writes one context handle.
    let r = unsafe { (d.cuCtxGetCurrent)(&mut ctx) };
    ffi::check(d, r, "cuCtxGetCurrent").unwrap();
    assert!(ctx.is_null(), "a context is still current after Drop");
}

/// Programmatic dependent launch only for code compiled for sm_90+ (decode.cu
/// emits its waits only there), on an sm_90+ device, with a 12.3+ driver
/// (capture keeps PDL edges from 12.3 on), unless AXON_CUDA_PDL=0.
#[test]
fn gpu_pdl_gating() {
    assert!(pdl_ok(120, 120, (13, 0), None));
    assert!(pdl_ok(90, 120, (12, 3), Some("1")));
    // A PTX fallback for an older arch: no griddepcontrol.wait in it.
    assert!(!pdl_ok(89, 120, (13, 0), None));
    assert!(!pdl_ok(86, 86, (13, 0), None));
    // Drivers before 12.3 fail captures with programmatic launches.
    assert!(!pdl_ok(120, 120, (12, 2), None));
    assert!(!pdl_ok(90, 90, (12, 0), None));
    assert!(!pdl_ok(120, 120, (13, 0), Some("0")));
    // This machine's module agrees with its target.
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let m = Kernels::module(&g).unwrap();
    let cc = g.info().unwrap().cc;
    let d = ffi::driver_version(ffi::driver().unwrap()).unwrap();
    let cc = (cc.0 * 10 + cc.1) as i32;
    let off = std::env::var("AXON_CUDA_PDL").ok();
    let want = cc >= 90 && d >= (12, 3) && off.as_deref() != Some("0");
    assert_eq!(m.pdl(), want);
    // A module compiled as PTX for an older arch never overlaps.
    let n = ffi::nvrtc().unwrap();
    let archs = ffi::nvrtc_archs(n).unwrap();
    if let Some(&old) = archs.iter().filter(|&&a| a < 90).max() {
        let (m, _) = Module::load_for(&g, IOTA, &[], None, Target::Ptx(old)).unwrap();
        assert!(!m.pdl());
    }
}

/// A capture whose body panics or fails is ended, so the stream works again.
#[test]
fn gpu_capture_unwinds_cleanly() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let (m, _) = Module::load_in(&g, IOTA, &[], None).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = g.capture(|| panic!("boom"));
    }));
    assert!(r.is_err());
    run_iota(&g, &m);
    assert!(g.capture(|| Err("nope".into())).is_err());
    run_iota(&g, &m);
    // And a good capture still replays.
    let out = g.alloc(32 * 4).unwrap();
    let f = m.function("iota").unwrap();
    // SAFETY: iota writes out[0..32]; out outlives the graph.
    let graph = g
        .capture(|| unsafe {
            g.launch(
                &f,
                (1, 1, 1),
                (32, 1, 1),
                0,
                &[out.arg(), Arg::F32(3.0), Arg::U32(32)],
            )
        })
        .unwrap();
    // SAFETY: out is live.
    unsafe { graph.launch().unwrap() };
    let v: Vec<f32> = out.read().unwrap();
    assert_eq!(v[31], 93.0);
}

/// Page-locking and unlocking host memory on another thread (as a hybrid
/// model's setup and teardown do) never breaks a capture in progress here:
/// without `CAPTURE_GATE`, `Pinned`'s drop there broke captures here
/// (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED; sm_89: this test failed within
/// 0.2 s).
#[test]
fn gpu_capture_survives_pinning_elsewhere() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let (m, _) = Module::load(&g, IOTA, &[]).unwrap();
    let f = m.function("iota").unwrap();
    let out = g.alloc(32 * 4).unwrap();
    let (stop, started) = (AtomicBool::new(false), AtomicBool::new(false));
    std::thread::scope(|s| {
        s.spawn(|| {
            let h = Gpu::open().unwrap();
            let buf = vec![0u8; 1 << 16];
            while !stop.load(Ordering::Relaxed) {
                // SAFETY: buf outlives the Pinned, dropped right away.
                drop(unsafe { h.register(buf.as_ptr(), buf.len(), false) }.unwrap());
                started.store(true, Ordering::Relaxed);
            }
        });
        while !started.load(Ordering::Relaxed) {
            std::thread::yield_now();
        }
        let r = (0..200).try_for_each(|_| {
            // SAFETY: iota writes out[0..32]; out outlives the graph.
            g.capture(|| {
                (0..50).try_for_each(|_| unsafe {
                    g.launch(
                        &f,
                        (1, 1, 1),
                        (32, 1, 1),
                        0,
                        &[out.arg(), Arg::F32(3.0), Arg::U32(32)],
                    )
                })
            })
            .map(drop)
        });
        stop.store(true, Ordering::Relaxed);
        r.unwrap();
    });
}

/// H2D throughput for streaming weights (PR A): pageable Vec, cuMemHostAlloc,
/// cuMemHostRegister'd anonymous mapping (like model.rs's r8 arena) and a
/// registered unaligned heap block (like a KV buffer). Prints GB/s.
#[test]
#[ignore = "benchmark: cargo test --release h2d_bench -- --ignored --nocapture"]
fn h2d_bench() {
    let Ok(g) = Gpu::open() else { return };
    let n = 256usize << 20;
    let dev = g.alloc(n).unwrap();
    let s = g.new_stream().unwrap();
    let e = g.new_event().unwrap();
    let time = |src: *const u8, what: &str| {
        let mut best = f64::MAX;
        for _ in 0..4 {
            let t = std::time::Instant::now();
            // SAFETY: src has n readable bytes for the duration (synced below).
            unsafe { s.h2d(&dev, 0, src, n).unwrap() };
            s.record(&e).unwrap();
            e.sync().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        eprintln!("{what}: {:.1} GB/s", n as f64 / best / 1e9);
    };
    let v = vec![7u8; n];
    time(v.as_ptr(), "pageable Vec");
    let mut h = g.host_alloc(n).unwrap();
    h.bytes_mut().unwrap().fill(3);
    let hp = h.bytes_mut().unwrap().as_ptr();
    time(hp, "cuMemHostAlloc");
    let mut map = memmap2::MmapMut::map_anon(n).unwrap();
    let _ = map.advise(memmap2::Advice::HugePage);
    map.fill(5);
    time(map.as_ptr(), "anon mmap pageable");
    let t = std::time::Instant::now();
    // SAFETY: map lives until after pin drops.
    let pin = unsafe { g.register(map.as_ptr(), n, false).unwrap() };
    eprintln!(
        "register 256 MiB: {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    time(map.as_ptr(), "anon mmap registered");
    drop(pin);
    let heap = vec![9u8; n + 100];
    let p = heap[37..].as_ptr();
    // SAFETY: heap lives until after pin2 drops.
    match unsafe { g.register(p, n, false) } {
        Ok(pin2) => {
            time(p, "unaligned heap registered");
            drop(pin2);
        }
        Err(e) => eprintln!("unaligned register: {e}"),
    }
    // D2H into a registered block.
    let mut back = vec![0u8; n];
    // SAFETY: back lives until after pin3 drops.
    let pin3 = unsafe { g.register(back.as_ptr(), n, false).unwrap() };
    let t = std::time::Instant::now();
    // SAFETY: back is writable, synced below.
    unsafe { s.d2h(back.as_mut_ptr(), &dev, 0, n).unwrap() };
    s.sync().unwrap();
    eprintln!(
        "d2h registered: {:.1} GB/s",
        n as f64 / t.elapsed().as_secs_f64() / 1e9
    );
    drop(pin3);
    assert_eq!(back[12345], 9);
}

/// Host-to-device bandwidth: pageable (`write_at`, the weight upload's
/// path) vs page-locked (`HostBuf::copy_to`), 1 GB in 32 MB copies, best of
/// 3. Startup measurement; prints only. Run with --ignored --nocapture.
#[test]
#[ignore]
fn h2d_bandwidth() {
    let Ok(g) = Gpu::open() else {
        eprintln!("no GPU: skipped");
        return;
    };
    const CHUNK: usize = 32 << 20;
    const TOTAL: usize = 1 << 30;
    let dst = g.alloc(TOTAL).unwrap();
    let src = vec![7u8; TOTAL];
    let mut pinned = g.host_alloc(CHUNK).unwrap();
    pinned.bytes_mut().unwrap().fill(7);
    let gbs = |f: &mut dyn FnMut()| {
        (0..3)
            .map(|_| {
                let t = std::time::Instant::now();
                f();
                TOTAL as f64 / t.elapsed().as_secs_f64() / 1e9
            })
            .fold(0.0, f64::max)
    };
    let pageable = gbs(&mut || {
        for off in (0..TOTAL).step_by(CHUNK) {
            dst.write_at(off, &src[off..off + CHUNK]).unwrap();
        }
    });
    let pinned_dma = gbs(&mut || {
        for off in (0..TOTAL).step_by(CHUNK) {
            pinned.copy_to(&dst, off, CHUNK).unwrap();
        }
        g.sync().unwrap();
    });
    // The weight upload's real source: a fresh mapping of the model file
    // (page cache warm after the first round), page faults included, and
    // with the page tables filled first (MADV_POPULATE_READ, not timed).
    let path = crate::testutil::model_path().unwrap();
    let file = std::fs::File::open(path).unwrap();
    let n = (file.metadata().unwrap().len() as usize).min(TOTAL) / CHUNK * CHUNK;
    let mapped = |populate: bool| {
        (0..3)
            .map(|_| {
                // SAFETY: read-only map of a file nobody writes.
                let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
                if populate {
                    let _ = map.advise(memmap2::Advice::PopulateRead);
                }
                let t = std::time::Instant::now();
                for off in (0..n).step_by(CHUNK) {
                    dst.write_at(off, &map[off..off + CHUNK]).unwrap();
                }
                n as f64 / t.elapsed().as_secs_f64() / 1e9
            })
            .fold(0.0, f64::max)
    };
    let (file_gbs, populated) = (mapped(false), mapped(true));
    eprintln!(
        "h2d: pageable {pageable:.1} GB/s, pinned {pinned_dma:.1} GB/s, file mapping \
         {file_gbs:.1} GB/s, populated file mapping {populated:.1} GB/s ({n} bytes)"
    );
}
