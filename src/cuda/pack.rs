//! Precompiled kernels: the pack. A release build carries the product's
//! kernel modules (`super::product_modules`) compiled ahead of time for the
//! common GPU architectures (`axon kernels build`), so `--device cuda` runs
//! on a machine with only the NVIDIA driver: no CUDA toolkit, no NVRTC.
//!
//! Each entry is NVRTC's output for one module (its `module_hash`: source
//! and options) and one `Target`, made by the same calls `Module::load`
//! makes at run time (`super::nvrtc_opts`, `super::compile`), so with the
//! same NVRTC it is the same code, byte for byte: the entry records the
//! fnv128 of NVRTC's output, which a test compares with a fresh compile.
//! Stored raw (a cubin, or PTX ending in a NUL) or wrapped in a fatbin that
//! NVIDIA's `fatbinary` compressed (`--compress-mode=size`, ~1/7 the size),
//! which `cuModuleLoadData` unpacks in the driver. The pick of a target for
//! a device (`pick`) is `pick_target`'s over the pack's targets.
//!
//! Where it comes from: embedded in the `axon` binary at build time
//! (`--cfg axon_kernel_pack`, src/main.rs, `set_embedded`), or a file named
//! by AXON_KERNEL_PACK_FILE. AXON_KERNEL_PACK: unset or 1 uses it when it
//! has the device's code, else NVRTC; 0 never (NVRTC only: the kill
//! switch); `only` refuses NVRTC (a driver-only machine on purpose).
//!
//! Format, little endian: a 32-byte header (magic `AXONPAK\x01`, NVRTC
//! major and minor as u32, the entry count as u32, 12 zero bytes), the
//! entries (80 bytes each: module u128, target kind u8 (0 cubin, 1 PTX,
//! 2 sm_90a), form u8 (0 raw, 1 fatbin), arch u16, 4 zero bytes, offset u64,
//! length u64, NVRTC output's fnv128, stored bytes' fnv128, 8 zero bytes),
//! then the images, each at a 16-byte boundary; entries with the same code
//! (a fast module is its exact twin's code today) share one image.

use super::{Target, ffi, fnv128};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const MAGIC: &[u8; 8] = b"AXONPAK\x01";
const HEADER: usize = 32;
const ENTRY: usize = 80;
/// More entries than any pack has (14 modules x a few targets): a bound on
/// what a damaged header can make `parse` allocate.
const MAX_ENTRIES: usize = 1 << 14;
/// A fatbin's first four bytes (0xBA55ED50, little endian).
const FATBIN_MAGIC: [u8; 4] = [0x50, 0xED, 0x55, 0xBA];
const ELF_MAGIC: &[u8; 4] = b"\x7fELF";

/// How an entry's image is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// NVRTC's output as it was: a cubin, or PTX ending in a NUL.
    Raw,
    /// A fatbin (`fatbinary`, compressed) holding NVRTC's output.
    Fatbin,
}

/// One module's code for one target.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    /// `module_hash` of the source and options it was built from.
    pub module: u128,
    pub target: Target,
    pub form: Form,
    /// The image's place in the pack.
    off: usize,
    len: usize,
    /// fnv128 of NVRTC's output (before any fatbin): what a fresh compile
    /// with the same NVRTC must hash to.
    #[cfg_attr(not(test), allow(dead_code))] // the bit-identity tests
    pub raw_fnv: u128,
    /// fnv128 of the stored bytes: `Pack::image` checks it.
    img_fnv: u128,
}

/// A parsed pack over its bytes.
pub struct Pack<'a> {
    bytes: &'a [u8],
    nvrtc: (i32, i32),
    entries: Vec<Entry>,
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

fn u128_at(b: &[u8], at: usize) -> u128 {
    u128::from_le_bytes(b[at..at + 16].try_into().unwrap())
}

impl<'a> Pack<'a> {
    /// The pack in `bytes`, or why it isn't one: every entry's image must
    /// lie inside the bytes after the entry table, look like its form (an
    /// ELF cubin, PTX ending in a NUL, a fatbin) and be the only one for
    /// its module and target. Images' checksums are checked when they're
    /// used (`image`), not here: a start-up reads only the header.
    pub fn parse(bytes: &'a [u8]) -> Result<Pack<'a>, String> {
        let bad = |why: String| Err(format!("not a valid kernel pack: {why}"));
        if bytes.len() < HEADER || &bytes[..8] != MAGIC {
            return bad("no AXONPAK\\x01 header".into());
        }
        let nvrtc = (u32_at(bytes, 8), u32_at(bytes, 12));
        let count = u32_at(bytes, 16) as usize;
        let table_end = count
            .checked_mul(ENTRY)
            .and_then(|t| t.checked_add(HEADER))
            .filter(|&e| count <= MAX_ENTRIES && e <= bytes.len());
        let Some(table_end) = table_end else {
            return bad(format!("{count} entries don't fit {} bytes", bytes.len()));
        };
        if nvrtc.0 > 1000 || nvrtc.1 > 1000 {
            return bad(format!("NVRTC version {}.{}", nvrtc.0, nvrtc.1));
        }
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let e = &bytes[HEADER + i * ENTRY..HEADER + (i + 1) * ENTRY];
            let arch = u16::from_le_bytes([e[18], e[19]]) as i32;
            let target = match (e[16], arch) {
                (_, a) if !(30..=2000).contains(&a) => {
                    return bad(format!("entry {i}: architecture {a}"));
                }
                (0, a) => Target::Cubin(a),
                (1, a) => Target::Ptx(a),
                (2, 90) => Target::Cubin90a,
                (k, a) => return bad(format!("entry {i}: target kind {k} (arch {a})")),
            };
            let form = match e[17] {
                0 => Form::Raw,
                1 => Form::Fatbin,
                f => return bad(format!("entry {i}: form {f}")),
            };
            let (off, len) = (u64_at(e, 24), u64_at(e, 32));
            let end = off
                .checked_add(len)
                .filter(|&end| end <= bytes.len() as u64);
            if off < table_end as u64 || len == 0 || end.is_none() {
                return bad(format!(
                    "entry {i}: image at {off}+{len} outside {table_end}..{}",
                    bytes.len()
                ));
            }
            let (off, len) = (off as usize, len as usize);
            let img = &bytes[off..off + len];
            let looks = match (form, target) {
                (Form::Fatbin, _) => img.starts_with(&FATBIN_MAGIC),
                (Form::Raw, Target::Ptx(_)) => img.last() == Some(&0),
                (Form::Raw, _) => img.starts_with(ELF_MAGIC),
            };
            if !looks {
                return bad(format!("entry {i}: the image isn't a {form:?} {target}"));
            }
            let entry = Entry {
                module: u128_at(e, 0),
                target,
                form,
                off,
                len,
                raw_fnv: u128_at(e, 40),
                img_fnv: u128_at(e, 56),
            };
            if entries
                .iter()
                .any(|x: &Entry| x.module == entry.module && x.target == entry.target)
            {
                return bad(format!("entry {i}: a second {target} image of one module"));
            }
            entries.push(entry);
        }
        Ok(Pack {
            bytes,
            nvrtc: (nvrtc.0 as i32, nvrtc.1 as i32),
            entries,
        })
    }

    /// The NVRTC that built it, (major, minor): the driver must support
    /// its CUDA major (`check_versions`).
    pub fn nvrtc(&self) -> (i32, i32) {
        self.nvrtc
    }

    #[cfg_attr(not(test), allow(dead_code))] // tests
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Every target some entry is for, sorted.
    pub fn targets(&self) -> Vec<Target> {
        let mut t: Vec<Target> = self.entries.iter().map(|e| e.target).collect();
        t.sort();
        t.dedup();
        t
    }

    /// The entry of `module` (`module_hash`) for `target`, if any.
    pub fn find(&self, module: u128, target: Target) -> Option<&Entry> {
        self.entries
            .iter()
            .find(|e| e.module == module && e.target == target)
    }

    /// `e`'s stored bytes, Err if they fail their checksum (a damaged file).
    pub fn image(&self, e: &Entry) -> Result<&'a [u8], String> {
        let img = &self.bytes[e.off..e.off + e.len];
        match fnv128(&[img]) == e.img_fnv {
            true => Ok(img),
            false => Err(format!("the {} image fails its checksum", e.target)),
        }
    }

    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }
}

/// A copy of an image in 8-byte-aligned memory for `cuModuleLoadData`:
/// an embedded pack's bytes have no alignment of their own.
pub struct Aligned {
    words: Vec<u64>,
    len: usize,
}

impl Aligned {
    pub fn new(b: &[u8]) -> Aligned {
        let mut words = vec![0u64; b.len().div_ceil(8)];
        for (w, c) in words.iter_mut().zip(b.chunks(8)) {
            let mut x = [0u8; 8];
            x[..c.len()].copy_from_slice(c);
            *w = u64::from_ne_bytes(x);
        }
        Aligned {
            words,
            len: b.len(),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `words` holds at least `len` initialized bytes (len.div_ceil(8)
        // u64s), any byte pattern is a valid u8, and the slice borrows self.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }
}

/// AXON_KERNEL_PACK's choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The pack when it has the device's code, else NVRTC (the default).
    On,
    /// NVRTC only (AXON_KERNEL_PACK=0).
    Off,
    /// The pack only (AXON_KERNEL_PACK=only).
    Only,
}

/// The mode a value of AXON_KERNEL_PACK asks for.
pub fn mode_from(v: Option<&str>) -> Result<Mode, String> {
    match v {
        None | Some("" | "1") => Ok(Mode::On),
        Some("0") => Ok(Mode::Off),
        Some("only") => Ok(Mode::Only),
        Some(v) => Err(format!("AXON_KERNEL_PACK takes 0, 1 or only, got '{v}'")),
    }
}

/// AXON_KERNEL_PACK's mode.
pub fn mode() -> Result<Mode, String> {
    mode_from(std::env::var("AXON_KERNEL_PACK").ok().as_deref())
}

static EMBEDDED: OnceLock<&'static [u8]> = OnceLock::new();

/// The pack built into this binary (src/main.rs calls it first thing when
/// built with `--cfg axon_kernel_pack`). Later calls do nothing.
pub fn set_embedded(bytes: &'static [u8]) {
    let _ = EMBEDDED.set(bytes);
}

/// The pack to use and where it is from (AXON_KERNEL_PACK_FILE's file,
/// else the embedded one), None if neither; Err if it can't be read or
/// isn't valid. Read and parsed once.
pub fn active() -> Result<Option<&'static Pack<'static>>, String> {
    static ACTIVE: OnceLock<Result<Option<Pack<'static>>, String>> = OnceLock::new();
    ACTIVE
        .get_or_init(|| {
            let bytes: &'static [u8] = match std::env::var_os("AXON_KERNEL_PACK_FILE") {
                Some(f) => {
                    let b = std::fs::read(&f).map_err(|e| {
                        format!("AXON_KERNEL_PACK_FILE {}: {e}", Path::new(&f).display())
                    })?;
                    Box::leak(b.into_boxed_slice())
                }
                None => match EMBEDDED.get() {
                    Some(b) => b,
                    None => return Ok(None),
                },
            };
            Pack::parse(bytes).map(Some)
        })
        .as_ref()
        .map(Option::as_ref)
        .map_err(Clone::clone)
}

/// Where `active`'s pack is from, for people.
pub fn active_source() -> String {
    match std::env::var_os("AXON_KERNEL_PACK_FILE") {
        Some(f) => format!("file {}", Path::new(&f).display()),
        None if EMBEDDED.get().is_some() => "built in".into(),
        None => "none".into(),
    }
}

/// The target of `targets` (a pack's) for a device of architecture `cc`,
/// by `pick_target`'s rule: sm_90a for a 9.0 device if `hopper`; else its
/// own cubin; else a cubin of the newest older minor of its major (binary
/// compatible); else PTX of the newest older architecture (the driver
/// compiles it: a GPU newer than the pack). None: nothing here runs on it.
pub fn pick(cc: i32, hopper: bool, targets: &[Target]) -> Option<Target> {
    let has = |t: Target| targets.contains(&t);
    if cc == 90 && hopper && has(Target::Cubin90a) {
        return Some(Target::Cubin90a);
    }
    if has(Target::Cubin(cc)) {
        return Some(Target::Cubin(cc));
    }
    let older = |ptx: bool| {
        targets.iter().filter_map(move |&t| match (t, ptx) {
            (Target::Cubin(a), false) | (Target::Ptx(a), true) if a < cc => Some(a),
            _ => None,
        })
    };
    if let Some(a) = older(false).filter(|&a| a / 10 == cc / 10).max() {
        return Some(Target::Cubin(a));
    }
    older(true).max().map(Target::Ptx)
}

/// A target as `axon kernels build --targets` names it: sm_XY, sm_90a or
/// compute_XY.
pub fn parse_target(s: &str) -> Result<Target, String> {
    let num = |d: &str| d.parse::<i32>().ok().filter(|a| (30..=2000).contains(a));
    let t = match s {
        "sm_90a" => Some(Target::Cubin90a),
        _ => match (s.strip_prefix("sm_"), s.strip_prefix("compute_")) {
            (Some(d), _) => num(d).map(Target::Cubin),
            (_, Some(d)) => num(d).map(Target::Ptx),
            _ => None,
        },
    };
    t.ok_or_else(|| format!("unknown target '{s}' (sm_XY, sm_90a or compute_XY)"))
}

/// The targets a release pack holds: Turing (RTX 20), Ampere (A100, RTX
/// 30), Ada (RTX 40), Hopper (both builds: sm_90a by default,
/// AXON_HOPPER=0's sm_90) and Blackwell desktop (RTX 50) cubins, and
/// compute_120 PTX for GPUs newer than all of them.
pub const RELEASE_TARGETS: &str = "sm_75,sm_80,sm_86,sm_89,sm_90,sm_90a,sm_120,compute_120";

/// An image to store: NVRTC's output for (module, target) entries.
struct Built {
    keys: Vec<(u128, Target)>,
    target: Target,
    raw: Vec<u8>,
}

/// The pack of every product module (`product_modules`, each from its
/// `min_arch` on) for `targets`, compiled by NVRTC `n` on `jobs` threads,
/// each image wrapped in a compressed fatbin by `fatbinary` if given (temp
/// files in `work`), else stored raw. `log` hears progress. NVRTC only,
/// no GPU.
pub fn build(
    n: &ffi::Nvrtc,
    targets: &[Target],
    fatbinary: Option<&Path>,
    work: &Path,
    jobs: usize,
    log: &(dyn Fn(&str) + Sync),
) -> Result<Vec<u8>, String> {
    let nv = ffi::nvrtc_version(n)?;
    let archs = ffi::nvrtc_archs(n)?;
    for &t in targets {
        let ok = match t {
            Target::Cubin(a) | Target::Ptx(a) => archs.contains(&a),
            Target::Cubin90a => archs.contains(&90) && nv.0 >= 12,
        };
        if !ok {
            return Err(format!("NVRTC {}.{} can't build {t}", nv.0, nv.1));
        }
    }
    let arch = |t: Target| match t {
        Target::Cubin(a) | Target::Ptx(a) => a,
        Target::Cubin90a => 90,
    };
    let mods = super::product_modules();
    let todo: Vec<(usize, Target)> = (0..mods.len())
        .flat_map(|i| targets.iter().map(move |&t| (i, t)))
        .filter(|&(i, t)| arch(t) >= mods[i].min_arch)
        .collect();
    log(&format!(
        "NVRTC {}.{}: {} modules x {} targets = {} compiles on {jobs} threads",
        nv.0,
        nv.1,
        mods.len(),
        targets.len(),
        todo.len()
    ));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::Mutex::new(Vec::<(u128, Target, Vec<u8>)>::new());
    let failed = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(i, t)) = todo.get(k) else { break };
                    let m = &mods[i];
                    let t0 = std::time::Instant::now();
                    match super::compile(n, m.src, &super::nvrtc_opts(t, &m.opts), t) {
                        Ok(img) => {
                            log(&format!(
                                "[{}/{}] {t} {}: {} bytes, {:.0} s",
                                k + 1,
                                todo.len(),
                                m.name,
                                img.len(),
                                t0.elapsed().as_secs_f64()
                            ));
                            let h = super::module_hash(m.src, &m.opts);
                            done.lock().unwrap().push((h, t, img));
                        }
                        Err(e) => failed.lock().unwrap().push(format!("{t} {}: {e}", m.name)),
                    }
                }
            });
        }
    });
    let failed = failed.into_inner().unwrap();
    if !failed.is_empty() {
        return Err(failed.join("\n"));
    }
    // One image per distinct code (a fast module is its exact twin's today).
    let mut built: Vec<Built> = Vec::new();
    let mut done = done.into_inner().unwrap();
    done.sort_by_key(|a| (a.0, a.1));
    for (h, t, raw) in done {
        match built.iter_mut().find(|b| b.target == t && b.raw == raw) {
            Some(b) => b.keys.push((h, t)),
            None => built.push(Built {
                keys: vec![(h, t)],
                target: t,
                raw,
            }),
        }
    }
    let mut images = Vec::with_capacity(built.len());
    for (i, b) in built.iter().enumerate() {
        let stored = match fatbinary {
            Some(fb) => (Form::Fatbin, wrap_fatbin(fb, work, i, b.target, &b.raw)?),
            None => (Form::Raw, b.raw.clone()),
        };
        images.push(stored);
    }
    Ok(write(nv, &built, &images))
}

/// `raw` (NVRTC's output for `t`) as a compressed fatbin: `fatbinary`
/// `fb` run on a temp file in `work` (removed after).
fn wrap_fatbin(fb: &Path, work: &Path, i: usize, t: Target, raw: &[u8]) -> Result<Vec<u8>, String> {
    let (kind, sm, body) = match t {
        Target::Cubin(a) => ("elf", a.to_string(), raw),
        Target::Cubin90a => ("elf", "90a".to_string(), raw),
        // fatbinary takes the PTX text, without NVRTC's NUL.
        Target::Ptx(a) => ("ptx", a.to_string(), raw.strip_suffix(&[0]).unwrap_or(raw)),
    };
    let inp = work.join(format!("img{i}.{kind}"));
    let out = work.join(format!("img{i}.fatbin"));
    std::fs::write(&inp, body).map_err(|e| format!("{}: {e}", inp.display()))?;
    let r = std::process::Command::new(fb)
        .arg("--64")
        .arg(format!("--create={}", out.display()))
        .args(["--compress-all", "--compress-mode=size"])
        .arg(format!(
            "--image3=kind={kind},sm={sm},file={}",
            inp.display()
        ))
        .output();
    let _ = std::fs::remove_file(&inp);
    let r = r.map_err(|e| format!("{}: {e}", fb.display()))?;
    if !r.status.success() {
        return Err(format!(
            "{} failed ({}): {}",
            fb.display(),
            r.status,
            String::from_utf8_lossy(&r.stderr).trim()
        ));
    }
    let img = std::fs::read(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let _ = std::fs::remove_file(&out);
    if !img.starts_with(&FATBIN_MAGIC) {
        return Err(format!("{} made no fatbin for {t}", fb.display()));
    }
    Ok(img)
}

/// The pack's bytes: header, entries (sorted by module and target), images.
fn write(nvrtc: (i32, i32), built: &[Built], images: &[(Form, Vec<u8>)]) -> Vec<u8> {
    let mut keys: Vec<(u128, Target, usize)> = built
        .iter()
        .enumerate()
        .flat_map(|(i, b)| b.keys.iter().map(move |&(h, t)| (h, t, i)))
        .collect();
    keys.sort();
    let align = |n: usize| n.div_ceil(16) * 16;
    let mut offs = Vec::with_capacity(images.len());
    let mut at = align(HEADER + keys.len() * ENTRY);
    for (_, img) in images {
        offs.push(at);
        at = align(at + img.len());
    }
    let mut v = Vec::with_capacity(at);
    v.extend_from_slice(MAGIC);
    v.extend_from_slice(&(nvrtc.0 as u32).to_le_bytes());
    v.extend_from_slice(&(nvrtc.1 as u32).to_le_bytes());
    v.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    v.resize(HEADER, 0);
    for &(h, t, i) in &keys {
        let (kind, arch) = match t {
            Target::Cubin(a) => (0u8, a),
            Target::Ptx(a) => (1, a),
            Target::Cubin90a => (2, 90),
        };
        let form = match images[i].0 {
            Form::Raw => 0u8,
            Form::Fatbin => 1,
        };
        v.extend_from_slice(&h.to_le_bytes());
        v.extend_from_slice(&[kind, form]);
        v.extend_from_slice(&(arch as u16).to_le_bytes());
        v.extend_from_slice(&[0; 4]);
        v.extend_from_slice(&(offs[i] as u64).to_le_bytes());
        v.extend_from_slice(&(images[i].1.len() as u64).to_le_bytes());
        v.extend_from_slice(&fnv128(&[&built[i].raw]).to_le_bytes());
        v.extend_from_slice(&fnv128(&[&images[i].1]).to_le_bytes());
        v.extend_from_slice(&[0; 8]);
    }
    for (i, (_, img)) in images.iter().enumerate() {
        v.resize(offs[i], 0);
        v.extend_from_slice(img);
    }
    v.resize(at, 0);
    v
}

/// `fatbinary` from the CUDA toolkit: $CUDA_PATH/bin, /opt/cuda/bin,
/// /usr/local/cuda/bin, then $PATH. None if none is there.
pub fn find_fatbinary() -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::env::var_os("CUDA_PATH")
        .map(|p| PathBuf::from(p).join("bin"))
        .into_iter()
        .collect();
    dirs.push("/opt/cuda/bin".into());
    dirs.push("/usr/local/cuda/bin".into());
    if let Some(p) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&p));
    }
    dirs.into_iter()
        .map(|d| d.join("fatbinary"))
        .find(|f| f.is_file())
}

/// What `p` holds, for `axon kernels list`: NVRTC, size, entries, modules
/// and targets.
pub fn describe(p: &Pack) -> String {
    let mut mods: Vec<u128> = p.entries.iter().map(|e| e.module).collect();
    mods.sort();
    mods.dedup();
    let product = super::product_modules();
    let known = mods
        .iter()
        .filter(|&&h| {
            product
                .iter()
                .any(|m| super::module_hash(m.src, &m.opts) == h)
        })
        .count();
    let packed = p.entries.iter().all(|e| e.form == Form::Fatbin);
    format!(
        "NVRTC {}.{}, {:.1} MB ({}), {} entries: {} modules ({} of this build's {}), targets {}",
        p.nvrtc.0,
        p.nvrtc.1,
        p.len() as f64 / 1e6,
        if packed {
            "compressed fatbins"
        } else {
            "raw images"
        },
        p.entries.len(),
        mods.len(),
        known,
        product.len(),
        p.targets()
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    )
}

/// `axon kernels`' usage.
const USAGE: &str = "usage:
  axon kernels list [PACK]   the precompiled kernels this binary uses (or file PACK holds),
                             and where this machine's GPU gets its kernels from
  axon kernels build OUT [--targets LIST] [--fatbinary PATH|none] [-j N]
                             compile this build's kernel modules with NVRTC (no GPU needed)
                             into pack OUT (default targets: sm_75,sm_80,sm_86,sm_89,sm_90,
                             sm_90a,sm_120,compute_120; images compressed by the CUDA
                             toolkit's fatbinary, found in $CUDA_PATH/bin, /opt/cuda/bin,
                             /usr/local/cuda/bin or $PATH; none: stored raw, ~7x larger;
                             -j: compiles at once, default 2)";

/// `axon kernels ARGS...` (`args`: what follows `kernels`), or why it failed.
pub fn kernels_cmd(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("list") if args.len() <= 2 => list(args.get(1).map(Path::new)),
        Some("build") if args.len() >= 2 => build_cmd(&args[1..]),
        Some("-h" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.to_string()),
    }
}

/// `axon kernels list [FILE]`.
fn list(file: Option<&Path>) -> Result<(), String> {
    match file {
        Some(f) => {
            let b = std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
            let p = Pack::parse(&b)?;
            println!("{}: {}", f.display(), describe(&p));
        }
        None => match active()? {
            Some(p) => println!("precompiled kernels ({}): {}", active_source(), describe(p)),
            None => println!("precompiled kernels: none in this build"),
        },
    }
    match mode()? {
        Mode::On => {}
        Mode::Off => println!("AXON_KERNEL_PACK=0: not used (NVRTC only)"),
        Mode::Only => println!("AXON_KERNEL_PACK=only: no NVRTC"),
    }
    println!("{}", super::summary());
    Ok(())
}

/// `axon kernels build OUT [--targets LIST] [--fatbinary PATH|none] [-j N]`.
fn build_cmd(args: &[String]) -> Result<(), String> {
    let out = PathBuf::from(&args[0]);
    let (mut targets, mut fb, mut jobs) = (RELEASE_TARGETS.to_string(), None, 2usize);
    let mut i = 1;
    while i < args.len() {
        let v = args
            .get(i + 1)
            .ok_or_else(|| format!("{} needs a value\n{USAGE}", args[i]))?;
        match args[i].as_str() {
            "--targets" => targets = v.clone(),
            "--fatbinary" => fb = Some(v.clone()),
            "-j" => {
                jobs = v
                    .parse()
                    .ok()
                    .filter(|&j| j > 0)
                    .ok_or_else(|| format!("-j takes a positive number, got '{v}'"))?
            }
            f => return Err(format!("unknown argument '{f}'\n{USAGE}")),
        }
        i += 2;
    }
    let targets: Vec<Target> = targets
        .split(',')
        .map(parse_target)
        .collect::<Result<_, _>>()?;
    let fb: Option<PathBuf> = match fb.as_deref() {
        Some("none") => None,
        Some(p) => Some(PathBuf::from(p)),
        None => Some(find_fatbinary().ok_or(
            "no fatbinary (the CUDA toolkit's) found: set CUDA_PATH, or --fatbinary none \
             to store the images uncompressed",
        )?),
    };
    let n = ffi::nvrtc()?;
    let work = out.with_extension("work");
    std::fs::create_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    let log = |s: &str| eprintln!("axon kernels: {s}");
    let r = build(n, &targets, fb.as_deref(), &work, jobs, &log);
    let _ = std::fs::remove_dir_all(&work);
    let bytes = r?;
    let p = Pack::parse(&bytes)?;
    let tmp = out.with_extension("tmp");
    std::fs::write(&tmp, &bytes)
        .and_then(|()| std::fs::rename(&tmp, &out))
        .map_err(|e| format!("{}: {e}", out.display()))?;
    println!("{}: {}", out.display(), describe(&p));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A raw cubin-looking image of `n` bytes, distinct per `seed`.
    fn elf(n: usize, seed: u8) -> Vec<u8> {
        let mut v = ELF_MAGIC.to_vec();
        v.extend((0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)));
        v
    }

    fn ptx(text: &str) -> Vec<u8> {
        let mut v = text.as_bytes().to_vec();
        v.push(0);
        v
    }

    /// A pack of `items` ((module, target, image) each; equal images share).
    fn pack_of(items: &[(u128, Target, Vec<u8>)]) -> Vec<u8> {
        let mut built: Vec<Built> = Vec::new();
        for (h, t, raw) in items {
            match built.iter_mut().find(|b| b.target == *t && &b.raw == raw) {
                Some(b) => b.keys.push((*h, *t)),
                None => built.push(Built {
                    keys: vec![(*h, *t)],
                    target: *t,
                    raw: raw.clone(),
                }),
            }
        }
        let images: Vec<(Form, Vec<u8>)> =
            built.iter().map(|b| (Form::Raw, b.raw.clone())).collect();
        write((13, 4), &built, &images)
    }

    fn sample() -> Vec<u8> {
        pack_of(&[
            (1, Target::Cubin(89), elf(100, 1)),
            (2, Target::Cubin(89), elf(100, 1)), // same code as module 1: shared
            (1, Target::Cubin90a, elf(70, 2)),
            (1, Target::Ptx(120), ptx("// ptx\n.version 9.0\n")),
            (3, Target::Cubin(75), elf(5, 3)),
        ])
    }

    /// What `write` stores, `parse` reads back: every entry, its image, the
    /// NVRTC version; equal images are stored once; images are 16-aligned.
    #[test]
    fn pack_round_trips() {
        let b = sample();
        let p = Pack::parse(&b).unwrap();
        assert_eq!(p.nvrtc(), (13, 4));
        assert_eq!(p.entries().len(), 5);
        assert_eq!(
            p.targets(),
            vec![
                Target::Cubin(75),
                Target::Cubin(89),
                Target::Ptx(120),
                Target::Cubin90a
            ]
        );
        let img = |h, t| p.image(p.find(h, t).unwrap()).unwrap();
        assert_eq!(img(1, Target::Cubin(89)), &elf(100, 1)[..]);
        assert_eq!(img(2, Target::Cubin(89)), &elf(100, 1)[..]);
        assert_eq!(img(1, Target::Cubin90a), &elf(70, 2)[..]);
        assert_eq!(img(3, Target::Cubin(75)), &elf(5, 3)[..]);
        assert!(img(1, Target::Ptx(120)).ends_with(b".version 9.0\n\0"));
        assert!(p.find(3, Target::Cubin(89)).is_none());
        let (a, b2) = (
            p.find(1, Target::Cubin(89)).unwrap(),
            p.find(2, Target::Cubin(89)).unwrap(),
        );
        assert_eq!((a.off, a.len), (b2.off, b2.len), "one image for equal code");
        assert_eq!(a.raw_fnv, fnv128(&[&elf(100, 1)]));
        assert!(p.entries().iter().all(|e| e.off % 16 == 0));
        // The table, then the 4 distinct images, each padded to 16 bytes.
        let imgs = [104usize, 74, 21, 9].map(|n| n.div_ceil(16) * 16);
        assert_eq!(
            b.len(),
            (HEADER + 5 * ENTRY).div_ceil(16) * 16 + imgs.iter().sum::<usize>()
        );
        assert_eq!(
            Aligned::new(img(1, Target::Cubin(89))).bytes(),
            &elf(100, 1)[..]
        );
        assert_eq!(Aligned::new(&[1, 2, 3]).bytes(), &[1, 2, 3]);
        assert!(describe(&p).contains("NVRTC 13.4"), "{}", describe(&p));
    }

    /// Damage is caught: a flipped image byte at `image` (its checksum),
    /// and at `parse` every truncation, a bad magic, out-of-range offsets
    /// and lengths, a huge count, unknown kinds and forms, an image that
    /// isn't its form, a duplicate entry.
    #[test]
    fn pack_rejects_damage() {
        let good = sample();
        let p = Pack::parse(&good).unwrap();
        let e = *p.find(1, Target::Cubin90a).unwrap();
        let mut b = good.clone();
        b[e.off + 10] ^= 1;
        let p2 = Pack::parse(&b).unwrap();
        assert!(p2.image(&e).unwrap_err().contains("checksum"));
        assert!(p2.image(p2.find(1, Target::Cubin(89)).unwrap()).is_ok());
        for n in 0..good.len() {
            // A cut inside the table or an image is refused; after the last
            // image (padding) it still parses.
            let last = p.entries().iter().map(|e| e.off + e.len).max().unwrap();
            assert_eq!(Pack::parse(&good[..n]).is_ok(), n >= last, "cut at {n}");
        }
        let entry = |i: usize| HEADER + i * ENTRY;
        let mutate = |f: &dyn Fn(&mut Vec<u8>)| {
            let mut b = good.clone();
            f(&mut b);
            Pack::parse(&b).map(|_| ())
        };
        let refused = |what: &str, f: &dyn Fn(&mut Vec<u8>)| {
            let r = mutate(f);
            assert!(r.is_err(), "{what}: accepted");
            assert!(r.unwrap_err().contains("not a valid kernel pack"), "{what}");
        };
        refused("magic", &|b| b[0] = b'X');
        refused("count huge", &|b| {
            b[16..20].copy_from_slice(&u32::MAX.to_le_bytes())
        });
        refused("count past the end", &|b| {
            b[16..20].copy_from_slice(&(MAX_ENTRIES as u32).to_le_bytes())
        });
        refused("nvrtc", &|b| {
            b[8..12].copy_from_slice(&5000u32.to_le_bytes())
        });
        refused("kind", &|b| b[entry(0) + 16] = 7);
        refused("form", &|b| b[entry(0) + 17] = 2);
        refused("arch 0", &|b| {
            b[entry(0) + 18..entry(0) + 20].copy_from_slice(&0u16.to_le_bytes())
        });
        refused("90a not at 90", &|b| {
            let i = (0..5).find(|&i| b[entry(i) + 16] == 2).unwrap();
            b[entry(i) + 18..entry(i) + 20].copy_from_slice(&89u16.to_le_bytes())
        });
        refused("offset in the table", &|b| {
            b[entry(0) + 24..entry(0) + 32].copy_from_slice(&40u64.to_le_bytes())
        });
        refused("offset past the end", &|b| {
            let n = b.len() as u64;
            b[entry(0) + 24..entry(0) + 32].copy_from_slice(&n.to_le_bytes())
        });
        refused("length wraps", &|b| {
            b[entry(0) + 32..entry(0) + 40].copy_from_slice(&u64::MAX.to_le_bytes())
        });
        refused("length 0", &|b| {
            b[entry(0) + 32..entry(0) + 40].copy_from_slice(&[0; 8])
        });
        refused("a cubin that isn't ELF", &|b| {
            let off = u64_at(b, entry(0) + 24) as usize;
            b[off] = 0;
        });
        refused("a fatbin that isn't one", &|b| b[entry(0) + 17] = 1);
        refused("PTX without its NUL", &|b| {
            let i = (0..5).find(|&i| b[entry(i) + 16] == 1).unwrap();
            let (off, len) = (u64_at(b, entry(i) + 24), u64_at(b, entry(i) + 32));
            b[(off + len - 1) as usize] = b'x';
        });
        refused("duplicate entry", &|b| {
            let (e0, e1) = (b[entry(0)..entry(1)].to_vec(), entry(1));
            b[e1..e1 + ENTRY].copy_from_slice(&e0);
        });
    }

    /// Random damage never panics `parse`, and whatever it accepts has
    /// every image inside the bytes (`image` slices without panicking).
    #[test]
    fn pack_parse_survives_random_damage() {
        let good = sample();
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut accepted = 0;
        for _ in 0..20_000 {
            let mut b = good.clone();
            for _ in 0..1 + rnd() % 4 {
                let at = (rnd() as usize) % b.len();
                // Mostly the header and table, where the structure is.
                let at = if rnd() % 4 == 0 {
                    at
                } else {
                    at % (HEADER + 5 * ENTRY)
                };
                b[at] = rnd() as u8;
            }
            if rnd() % 8 == 0 {
                b.truncate((rnd() as usize) % (b.len() + 1));
            }
            if let Ok(p) = Pack::parse(&b) {
                accepted += 1;
                for e in p.entries() {
                    assert!(e.off + e.len <= b.len());
                    let _ = p.image(e);
                }
            }
        }
        assert!(accepted > 0, "some damage (an image byte) still parses");
    }

    /// `pick` is `pick_target`'s rule over a pack's targets: for every
    /// device a release pack covers it picks what NVRTC 13.4 (whose archs
    /// include them all) picks; newer majors get the PTX, an sm_121 the
    /// sm_120 cubin, and a GPU older than the pack (or an sm_100 below its
    /// PTX) nothing, so NVRTC builds for it.
    #[test]
    fn pack_pick_follows_pick_target() {
        let targets: Vec<Target> = RELEASE_TARGETS
            .split(',')
            .map(|t| parse_target(t).unwrap())
            .collect();
        let nvrtc_archs = [75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, 121];
        for cc in [75, 80, 86, 89, 90, 120] {
            for hopper in [true, false] {
                let want = super::super::pick_target(cc, &nvrtc_archs, (13, 4), hopper).unwrap();
                assert_eq!(
                    pick(cc, hopper, &targets),
                    Some(want),
                    "sm_{cc} hopper {hopper}"
                );
            }
        }
        assert_eq!(pick(87, true, &targets), Some(Target::Cubin(86)));
        assert_eq!(pick(121, true, &targets), Some(Target::Cubin(120)));
        assert_eq!(pick(130, true, &targets), Some(Target::Ptx(120)));
        assert_eq!(pick(100, true, &targets), None);
        assert_eq!(pick(70, true, &targets), None);
        assert_eq!(pick(61, true, &targets), None);
        // Without sm_90a, a Hopper takes the sm_90 cubin; without either, nothing.
        let no90a: Vec<Target> = targets
            .iter()
            .copied()
            .filter(|&t| t != Target::Cubin90a)
            .collect();
        assert_eq!(pick(90, true, &no90a), Some(Target::Cubin(90)));
        assert_eq!(pick(90, true, &[Target::Cubin(89)]), None);
        assert_eq!(pick(89, true, &[]), None);
    }

    #[test]
    fn pack_mode_and_targets_parse() {
        assert_eq!(mode_from(None), Ok(Mode::On));
        assert_eq!(mode_from(Some("1")), Ok(Mode::On));
        assert_eq!(mode_from(Some("")), Ok(Mode::On));
        assert_eq!(mode_from(Some("0")), Ok(Mode::Off));
        assert_eq!(mode_from(Some("only")), Ok(Mode::Only));
        assert!(mode_from(Some("yes")).unwrap_err().contains("0, 1 or only"));
        assert_eq!(parse_target("sm_89"), Ok(Target::Cubin(89)));
        assert_eq!(parse_target("sm_90a"), Ok(Target::Cubin90a));
        assert_eq!(parse_target("compute_120"), Ok(Target::Ptx(120)));
        for bad in ["sm_", "sm_x", "compute_9", "sm_90b", "89", "sm_99999"] {
            assert!(parse_target(bad).is_err(), "{bad}");
        }
        for t in RELEASE_TARGETS.split(',') {
            assert_eq!(parse_target(t).unwrap().to_string(), t);
        }
    }

    /// `code_for`'s choice: the pack's target when it has the device's code
    /// and the driver is new enough; else NVRTC, with the pack's reason
    /// ahead of NVRTC's when neither builds; AXON_KERNEL_PACK=only never
    /// asks NVRTC, =0 never looks at the pack.
    #[test]
    fn pack_code_choice() {
        use super::super::{Code, code_from};
        let b = pack_of(&[(1, Target::Cubin(89), elf(10, 1))]);
        let p: &'static Pack<'static> = Box::leak(Box::new(
            Pack::parse(Box::leak(b.into_boxed_slice())).unwrap(),
        ));
        let no_nvrtc = || Err("NVRTC not found".to_string());
        let never = || -> Result<_, String> { panic!("NVRTC asked") };
        // The driver's release is read only for the "too old" line.
        fn unasked() -> Option<String> {
            panic!("driver release asked")
        }
        fn r570() -> Option<String> {
            Some("570.86.15".into())
        }
        fn unknown() -> Option<String> {
            None
        }
        let new = ((13, 5), unasked as fn() -> Option<String>);
        let old = ((12, 8), r570 as fn() -> Option<String>);
        let pick = |r: Result<(Code, Target), String>| match r {
            Ok((Code::Pack(_), t)) => Ok(format!("pack {t}")),
            Ok((Code::Nvrtc(_), t)) => Ok(format!("nvrtc {t}")),
            Err(e) => Err(e),
        };
        assert_eq!(
            pick(code_from(Mode::On, Ok(Some(p)), 89, new, true, never)),
            Ok("pack sm_89".into())
        );
        assert_eq!(
            pick(code_from(Mode::Only, Ok(Some(p)), 89, new, true, never)),
            Ok("pack sm_89".into())
        );
        // A driver older than the pack's CUDA major and no NVRTC: one line,
        // the driver found, the branch needed, the way out.
        let too_old = "NVIDIA driver 570.86.15 (CUDA 12.8) is too old: axon's GPU kernels need \
                       R580+ (CUDA 13); update the driver, or run on the CPU with --device cpu";
        assert_eq!(
            pick(code_from(Mode::On, Ok(Some(p)), 89, old, true, no_nvrtc)),
            Err(too_old.into())
        );
        assert_eq!(
            pick(code_from(Mode::Only, Ok(Some(p)), 89, old, true, never)),
            Err(format!("AXON_KERNEL_PACK=only: {too_old}"))
        );
        assert_eq!(
            pick(code_from(
                Mode::On,
                Ok(Some(p)),
                89,
                ((11, 4), unknown),
                true,
                no_nvrtc
            )),
            Err(
                "the NVIDIA driver (CUDA 11.4) is too old: axon's GPU kernels need R580+ \
                 (CUDA 13); update the driver, or run on the CPU with --device cpu"
                    .into()
            )
        );
        assert_eq!(
            pick(code_from(Mode::On, Ok(Some(p)), 100, new, true, no_nvrtc)),
            Err("no precompiled kernels for sm_100 (the pack has sm_89); NVRTC not found".into())
        );
        assert_eq!(
            pick(code_from(Mode::Only, Ok(Some(p)), 100, new, true, never)),
            Err(
                "AXON_KERNEL_PACK=only: no precompiled kernels for sm_100 (the pack has sm_89)"
                    .into()
            )
        );
        assert_eq!(
            pick(code_from(Mode::Only, Ok(None), 89, new, true, never)),
            Err("AXON_KERNEL_PACK=only: this build has no precompiled kernels".into())
        );
        assert_eq!(
            pick(code_from(Mode::On, Ok(None), 89, new, true, no_nvrtc)),
            Err("this build has no precompiled kernels; NVRTC not found".into())
        );
        // No pack: an old driver is NVRTC's business (check_versions).
        assert_eq!(
            pick(code_from(
                Mode::On,
                Ok(None),
                89,
                ((12, 8), unasked),
                true,
                no_nvrtc
            )),
            Err("this build has no precompiled kernels; NVRTC not found".into())
        );
        assert_eq!(
            pick(code_from(
                Mode::On,
                Err("bad pack".into()),
                89,
                new,
                true,
                no_nvrtc
            )),
            Err("bad pack; NVRTC not found".into())
        );
        // The kill switch: no pack reason, NVRTC's own error.
        assert_eq!(
            pick(code_from(Mode::Off, Ok(Some(p)), 89, old, true, no_nvrtc)),
            Err("NVRTC not found".into())
        );
    }

    /// The driver's release from /proc/driver/nvidia/version (proprietary
    /// and open kernel modules), and the one-line refusal of a driver older
    /// than CUDA 12, whose API axon can't load (`ffi::driver`).
    #[test]
    fn pack_old_driver_lines() {
        use super::super::{old_api, old_driver};
        let closed = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.86.15  Tue Jan  7 \
                      23:03:26 UTC 2025\nGCC version:  gcc version 12.2.0\n";
        let open = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  620.70.06  Release \
                    Build  (notroot@)  Sat Nov  7 14:23:54 UTC 2026\nGCC version:  Selected \
                    multilib: .;@m64\n";
        assert_eq!(
            ffi::parse_driver_release(closed).as_deref(),
            Some("570.86.15")
        );
        assert_eq!(
            ffi::parse_driver_release(open).as_deref(),
            Some("620.70.06")
        );
        for bad in [
            "",
            "GCC version: 12.2.0\n",
            "NVRM version: NVIDIA UNIX x86_64 Kernel Module\n",
            "NVRM version: x 570 .5 5.x\n",
        ] {
            assert_eq!(ffi::parse_driver_release(bad), None, "{bad:?}");
        }
        assert_eq!(
            old_api((11, 4), Some("470.256.02"), 13).as_deref(),
            Some(
                "NVIDIA driver 470.256.02 (CUDA 11.4) is too old: axon's GPU kernels need R580+ \
                 (CUDA 13); update the driver, or run on the CPU with --device cpu"
            )
        );
        // A build without a pack needs CUDA 12 (with an NVRTC of its own).
        assert_eq!(
            old_api((11, 8), None, 12).as_deref(),
            Some(
                "the NVIDIA driver (CUDA 11.8) is too old: axon's GPU kernels need R525+ \
                 (CUDA 12); update the driver, or run on the CPU with --device cpu"
            )
        );
        assert_eq!(old_api((12, 0), Some("525.60.13"), 13), None);
        assert!(old_driver((14, 0), None, 15).contains("need a CUDA 15 driver;"));
        // One line, always.
        for s in [
            old_driver((12, 8), Some("570.86.15"), 13),
            old_driver((11, 0), None, 12),
        ] {
            assert!(!s.contains('\n'), "{s}");
        }
    }

    /// The module the test model runs (`module_spec` as `module_for_mode`
    /// picks it), from a pack holding NVRTC's output for this GPU's target
    /// (written by `write`, as `build` does) and compiled by that NVRTC
    /// here: the entry's checksum is the fresh compile's, both modules have
    /// the same flags (architecture, Hopper, PDL, mode, FP8, `src_hash`), and
    /// the model computes the same logits with each, bit for bit, through
    /// batched prefill and decode. A pack without the module, or with its
    /// image damaged, falls back to NVRTC for the same target. Skipped
    /// without NVRTC (it is the reference).
    #[test]
    fn gpu_pack_module_is_nvrtcs() {
        use super::super::{CacheStatus, Module, module_hash, module_spec, nvrtc_for, nvrtc_opts};
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let Some(m) = crate::testutil::model_with(Some(false)) else {
            return;
        };
        let cc = g.info().unwrap().cc;
        let driver = ffi::driver_version(g.d).unwrap();
        let (n, t) = match nvrtc_for(
            (cc.0 * 10 + cc.1) as i32,
            driver,
            super::super::hopper_env(),
        ) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let ext = super::super::needs_ext(&m);
        let (src, opts) = module_spec(false, false, ext, crate::numerics::Numerics::Exact);
        let h = module_hash(src, &opts);
        let raw = super::super::compile(n, src, &nvrtc_opts(t, &opts), t).unwrap();
        let nv = ffi::nvrtc_version(n).unwrap();
        let one = |raw: &[u8], key: u128| {
            let b = Built {
                keys: vec![(key, t)],
                target: t,
                raw: raw.to_vec(),
            };
            write(nv, std::slice::from_ref(&b), &[(Form::Raw, raw.to_vec())])
        };
        let bytes = one(&raw, h);
        let p = Pack::parse(&bytes).unwrap();
        assert_eq!(p.find(h, t).unwrap().raw_fnv, fnv128(&[&raw]));
        let (mut mp, st) = Module::load_packed(&g, &p, src, &opts, None, t).unwrap();
        assert_eq!(st, CacheStatus::Pack);
        let (mut mn, st) = Module::load_for(&g, n, src, &opts, None, t).unwrap();
        assert_eq!(st, CacheStatus::Off);
        // As `module_for_mode` marks it.
        (mp.ext, mn.ext) = (ext, ext);
        let flags = |m: &Module| (m.arch, m.hopper, m.pdl, m.fast, m.f8, m.src_hash);
        assert_eq!(flags(&mp), flags(&mn), "{t}");
        let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
        let ids = tok.encode("The capital of France is Paris, and the capital of Japan is");
        let trace = |km| greedy_bits(&g, km, &m, &ids);
        assert!(
            trace(&mp) == trace(&mn),
            "{t}: the pack's module computes other bits"
        );
        // No entry for the module, or a damaged one: NVRTC builds it for `t`.
        for bytes in [one(&raw, h ^ 1), {
            let mut b = one(&raw, h);
            let e = *Pack::parse(&b).unwrap().find(h, t).unwrap();
            b[e.off + e.len / 2] ^= 0x40;
            b
        }] {
            let p = Pack::parse(&bytes).unwrap();
            if mode() == Ok(Mode::Only) {
                assert!(Module::load_packed(&g, &p, src, &opts, None, t).is_err());
                continue;
            }
            let (mf, st) = Module::load_packed(&g, &p, src, &opts, None, t).unwrap();
            assert_eq!(st, CacheStatus::Off, "compiled, no cache dir");
            assert_eq!(flags(&mf), flags(&mn));
        }
    }

    /// The logits' bits of 8 greedy decode steps of `m` on module `km`
    /// after a batched prefill of `ids`.
    fn greedy_bits<'g>(
        g: &'g super::super::Gpu,
        km: &'g super::super::Module<'g>,
        m: &crate::gguf::Model,
        ids: &[u32],
    ) -> Vec<u32> {
        use crate::Lm;
        let net = super::super::qwen2::CudaQwen2::new(g, km, m, 128).unwrap();
        let mut c = net.new_cache(ids.len() + 10);
        let (&last, rest) = ids.split_last().unwrap();
        let pb = net.prefill_batch();
        for (i, b) in rest.chunks(pb).enumerate() {
            net.feed_batch(b, i * pb, &mut c);
        }
        let (mut tk, mut out) = (last, Vec::new());
        for s in 0..8 {
            let l = net.forward(tk, rest.len() + s, &mut c);
            tk = crate::argmax(&l);
            out.extend(l.iter().map(|x| x.to_bits()));
        }
        out
    }

    /// A release pack (AXON_TEST_KERNEL_PACK names its file) holds, for
    /// this GPU's target, NVRTC's exact output for every product module:
    /// each entry's checksum is a fresh compile's with the NVRTC here (when
    /// it is the pack's version; else skipped), and each loads on this GPU
    /// through `load_packed`. Skipped without the variable or NVRTC.
    #[test]
    fn gpu_release_pack_is_nvrtcs() {
        use super::super::{CacheStatus, Module, module_hash, nvrtc_opts, product_modules};
        let Some(f) = std::env::var_os("AXON_TEST_KERNEL_PACK") else {
            eprintln!("skipping: AXON_TEST_KERNEL_PACK names no pack file");
            return;
        };
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let bytes = std::fs::read(&f).unwrap();
        let p = Pack::parse(&bytes).unwrap();
        let cc = g.info().unwrap().cc;
        let cc = (cc.0 * 10 + cc.1) as i32;
        let t = pick(cc, super::super::hopper_env(), &p.targets())
            .unwrap_or_else(|| panic!("the pack has no code for sm_{cc}"));
        let driver = ffi::driver_version(g.d).unwrap();
        let n = super::super::nvrtc_target(t, driver).ok();
        let same = n.filter(|&n| ffi::nvrtc_version(n).ok() == Some(p.nvrtc()));
        if same.is_none() {
            eprintln!(
                "NVRTC {}.{} isn't here: checking that the entries load, not their bytes",
                p.nvrtc().0,
                p.nvrtc().1
            );
        }
        let mods = product_modules();
        let mut checked = 0;
        for md in mods.iter().filter(|md| cc >= md.min_arch) {
            let h = module_hash(md.src, &md.opts);
            let e = p
                .find(h, t)
                .unwrap_or_else(|| panic!("{t} {}: not in the pack", md.name));
            if let Some(n) = same {
                let raw = super::super::compile(n, md.src, &nvrtc_opts(t, &md.opts), t).unwrap();
                assert_eq!(e.raw_fnv, fnv128(&[&raw]), "{t} {}: other bytes", md.name);
                checked += 1;
            }
            let (m, st) = Module::load_packed(&g, &p, md.src, &md.opts, None, t).unwrap();
            assert_eq!(st, CacheStatus::Pack, "{t} {}", md.name);
            assert_eq!(m.src_hash, h);
        }
        eprintln!("{t}: {checked} entries the same bytes as NVRTC's here; all load");
    }

    /// The pack holds every module the product loads: `product_modules`
    /// has each `module_spec` input `module_for_mode` can pass (FP8 only in
    /// fast mode and not with gpt-oss) and each `moe_spec` input, once.
    #[test]
    fn product_modules_cover_the_loaders() {
        use crate::numerics::Numerics::{Exact, Fast};
        let have: Vec<u128> = super::super::product_modules()
            .iter()
            .map(|m| super::super::module_hash(m.src, &m.opts))
            .collect();
        let mut want = Vec::new();
        for oss in [false, true] {
            for ext in [false, true] {
                for n in [Exact, Fast] {
                    let (s, o) = super::super::module_spec(oss, false, ext, n);
                    want.push(super::super::module_hash(s, &o));
                }
            }
        }
        for ext in [false, true] {
            let (s, o) = super::super::module_spec(false, true, ext, Fast);
            want.push(super::super::module_hash(s, &o));
        }
        for mla in [false, true] {
            for n in [Exact, Fast] {
                let (s, o) = super::super::moe_spec(mla, n);
                want.push(super::super::module_hash(s, &o));
            }
        }
        let mut d = have.clone();
        d.sort();
        d.dedup();
        assert_eq!(d.len(), have.len(), "each module once");
        want.sort();
        assert_eq!(d, want);
    }
}
