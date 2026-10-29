//! KV save/restore: a sequence's cache rows to a file and back, exactly.
//!
//! A cache's bytes are a pure function of the token prefix (decode ==
//! batched prefill == verify, bit for bit), so copying them is exact: a
//! restored cache continues as the uninterrupted one would. They are only
//! that function *for one identity*: the model file, the KV type, the RoPE
//! table, the backend and its numerics (kernel sources, GPU architecture,
//! driver; the numerics mode, `crate::numerics`). A file records its identity in the header; one made under
//! another identity is never loaded as bytes, only as tokens, which are
//! prefilled again. A file is never wrong, only slower.
//!
//! # File format `AXKV` v1
//!
//! `b"AXKV"`, version u32, header length u32, the header (JSON: `identity`,
//! null when the backend has no cache image, e.g. a CPU/GPU split), then
//! records, appended over time:
//!
//! `kind u8, len u64, payload[len], check u64` (`Hasher` of the payload).
//! - `KV` (1): `lo u64, hi u64`, then positions lo..hi's image.
//! - `STATE` (2): JSON, the conversation (`chat::Chat`'s state) with the
//!   positions the file's KV records cover (`kv`): records 0..a, a..b, ...
//!   tile 0..kv.
//!
//! Reading stops at the first torn or failing record: a crash mid-append
//! loses the last save, not the file. The last STATE wins.
//!
//! # Image layout
//!
//! Positions lo..hi split at multiples of `PAGE` (64); per page, per region
//! (f16: the rows; q8: the int8 rows, then the f32 scales), per layer K then
//! V, the page's rows. A page of every layer is the unit the host-RAM store
//! (`kvstore`) keeps, and one strided copy per region moves it.

use crate::cuda::qwen2::{CudaCache, CudaQwen2};
use crate::lm::{Backend, BackendCache, CacheInner, Inner, Lm};
use crate::model::{KvCache, Qwen2};
use crate::numerics::Numerics;
use serde_json::{Value, json};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Positions per page of an image.
pub const PAGE: usize = 64;

/// File format version.
const VERSION: u32 = 1;
const MAGIC: &[u8; 4] = b"AXKV";
const KIND_KV: u8 = 1;
const KIND_STATE: u8 = 2;

/// Bump when the CPU's (or the host side's) cache bits change for the same
/// model and settings: files from before are then prefilled again. The
/// GPU's kernels are covered by their source hash (`cuda::numerics_tag`).
pub const NUMERICS: u32 = 1;

/// lo..hi split at multiples of PAGE.
pub fn segments(lo: usize, hi: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut s = lo;
    while s < hi {
        let e = ((s / PAGE + 1) * PAGE).min(hi);
        v.push((s, e));
        s = e;
    }
    v
}

/// A fast streaming 64-bit hash (4 lanes of multiply-rotate over 8-byte
/// words; not cryptographic): files' checksums and the model's identity.
#[derive(Clone)]
pub struct Hasher {
    h: [u64; 4],
    buf: [u8; 32],
    n: usize,
    total: u64,
}

const K: [u64; 4] = [
    0x9e37_79b9_7f4a_7c15,
    0xc2b2_ae3d_27d4_eb4f,
    0x1656_67b1_9e37_79f9,
    0x85eb_ca77_c2b2_ae63,
];

impl Default for Hasher {
    fn default() -> Hasher {
        Hasher {
            h: K,
            buf: [0; 32],
            n: 0,
            total: 0,
        }
    }
}

impl Hasher {
    #[inline(always)]
    fn block(h: &mut [u64; 4], b: &[u8]) {
        for (i, w) in b.chunks_exact(8).enumerate() {
            let w = u64::from_le_bytes(w.try_into().expect("8 bytes"));
            h[i] = (h[i] ^ w).wrapping_mul(K[i]).rotate_left(31);
        }
    }

    pub fn update(&mut self, mut d: &[u8]) {
        self.total += d.len() as u64;
        if self.n > 0 {
            let t = (32 - self.n).min(d.len());
            self.buf[self.n..self.n + t].copy_from_slice(&d[..t]);
            self.n += t;
            d = &d[t..];
            if self.n < 32 {
                return;
            }
            let b = self.buf;
            Self::block(&mut self.h, &b);
            self.n = 0;
        }
        let (blocks, rest) = d.as_chunks::<32>();
        for b in blocks {
            Self::block(&mut self.h, b);
        }
        self.buf[..rest.len()].copy_from_slice(rest);
        self.n = rest.len();
    }

    pub fn finish(&self) -> u64 {
        let mut h = self.h;
        let mut tail = [0u8; 32];
        tail[..self.n].copy_from_slice(&self.buf[..self.n]);
        Self::block(&mut h, &tail);
        let mut x = self.total ^ (self.n as u64).rotate_left(56);
        for (i, v) in h.iter().enumerate() {
            x = (x ^ v.rotate_left(i as u32 * 16)).wrapping_mul(K[(i + 1) % 4]);
            x ^= x >> 29;
        }
        x
    }

    pub fn of(d: &[u8]) -> u64 {
        let mut h = Hasher::default();
        h.update(d);
        h.finish()
    }
}

/// Hashes what goes through it.
struct HashW<'a, W: Write> {
    w: &'a mut W,
    h: Hasher,
    n: u64,
}

impl<W: Write> Write for HashW<'_, W> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        let k = self.w.write(b)?;
        self.h.update(&b[..k]);
        self.n += k as u64;
        Ok(k)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }
}

struct HashR<'a, R: Read> {
    r: &'a mut R,
    h: Hasher,
}

impl<R: Read> Read for HashR<'_, R> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        let k = self.r.read(b)?;
        self.h.update(&b[..k]);
        Ok(k)
    }
}

/// The model file's identity: its length and a hash of every byte, in
/// hex. Hashing a multi-GB file takes a while, so the result is kept in
/// ~/.cache/axon/model-ids by (path, length, modification time).
pub fn model_id(path: &Path) -> io::Result<String> {
    let canon = std::fs::canonicalize(path)?;
    // A safetensors checkpoint, named by its directory or by a shard in it
    // (`safetensors::dir_of`): every file's id, in order, so a changed
    // tokenizer.json or a second shard changes it too.
    // (By the path as given: an HF cache's shard is a symlink to a blob.)
    if let Some(dir) = path.to_str().and_then(crate::safetensors::dir_of) {
        let mut h = Hasher::default();
        for f in crate::safetensors::files(&std::fs::canonicalize(dir)?)? {
            h.update(one_file_id(&std::fs::canonicalize(&f)?)?.as_bytes());
            h.update(b"\n");
        }
        return Ok(format!("{:016x}-st", h.finish()));
    }
    one_file_id(&canon)
}

/// `model_id` of one file (`canon`: canonical).
fn one_file_id(canon: &Path) -> io::Result<String> {
    let meta = std::fs::metadata(canon)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let key = format!("{}\t{}\t{mtime}", canon.display(), meta.len());
    let store = crate::cuda::cache_dir().and_then(|d| d.parent().map(|p| p.join("model-ids")));
    if let Some(s) = &store
        && let Ok(t) = std::fs::read_to_string(s)
        && let Some(id) = t
            .lines()
            .find_map(|l| l.strip_prefix(&key)?.strip_prefix('\t'))
    {
        return Ok(id.to_owned());
    }
    let id = format!("{:016x}-{}", hash_file(canon, meta.len())?, meta.len());
    if let Some(s) = &store {
        let _ = std::fs::create_dir_all(s.parent().expect("a parent"));
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(s)
        {
            let _ = writeln!(f, "{key}\t{id}");
        }
    }
    Ok(id)
}

/// `Hasher` of each 256 MiB of the file (on several threads), then of
/// those hashes in order.
fn hash_file(path: &Path, len: u64) -> io::Result<u64> {
    const CHUNK: u64 = 256 << 20;
    let n = len.div_ceil(CHUNK).max(1) as usize;
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 8);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out = vec![0u64; n];
    let parts: Vec<io::Result<Vec<(usize, u64)>>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads.min(n))
            .map(|_| {
                sc.spawn(|| -> io::Result<Vec<(usize, u64)>> {
                    let mut f = File::open(path)?;
                    let mut buf = vec![0u8; 8 << 20];
                    let mut got = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n {
                            return Ok(got);
                        }
                        f.seek(SeekFrom::Start(i as u64 * CHUNK))?;
                        let mut left = CHUNK.min(len - i as u64 * CHUNK) as usize;
                        let mut h = Hasher::default();
                        while left > 0 {
                            let k = left.min(buf.len());
                            f.read_exact(&mut buf[..k])?;
                            h.update(&buf[..k]);
                            left -= k;
                        }
                        got.push((i, h.finish()));
                    }
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("hash thread"))
            .collect()
    });
    for p in parts {
        for (i, h) in p? {
            out[i] = h;
        }
    }
    let mut h = Hasher::default();
    for v in out {
        h.update(&v.to_le_bytes());
    }
    Ok(h.finish())
}

/// What a cache's bytes depend on in the model: the file (hash and
/// length), its architecture and the RoPE tables the backends build from
/// it (a RoPE override changes them without changing the file).
pub fn model_ident(m: &crate::gguf::Model, path: &Path, head_dim: usize) -> io::Result<Value> {
    let mut h = Hasher::default();
    if head_dim > 0 {
        for f in crate::model::rope_freqs(m, head_dim)
            .iter()
            .chain(&crate::model::rope_freqs_swa(m, head_dim))
        {
            h.update(&f.to_bits().to_le_bytes());
        }
    }
    Ok(json!({
        "model": model_id(path)?,
        "arch": m.arch,
        "rope": format!("{:016x}", h.finish()),
    }))
}

/// A backend that can image its cache (copy rows lo..hi out and in).
pub trait KvIo: Lm {
    /// What `cache`'s bytes depend on beyond the model (`model_ident`): the
    /// backend, its numerics, the KV type and shape, and `row_bytes` (image
    /// bytes per position). None: no image (the CPU/GPU split, several
    /// GPUs): a saved session is prefilled again.
    fn kv_ident(&self, cache: &Self::Cache) -> Option<Value>;

    /// `cache`'s (`kv_ident`) image of positions lo..hi to `out`.
    fn kv_save(
        &self,
        cache: &Self::Cache,
        lo: usize,
        hi: usize,
        out: &mut dyn Write,
    ) -> io::Result<()>;

    /// Positions lo..hi of `cache` from their image (`kv_save`'s bytes).
    fn kv_load(
        &self,
        cache: &mut Self::Cache,
        lo: usize,
        hi: usize,
        src: &mut dyn Read,
    ) -> io::Result<()>;

    /// (layers, kv_dim, head_dim).
    fn kv_shape(&self) -> (usize, usize, usize);
}

/// The CPU's arch class: what of its instruction sets changes the bits.
fn cpu_class() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        // FMA contraction is the one ISA choice that changes the bits.
        let fma = if is_x86_feature_detected!("fma") {
            " fma"
        } else {
            ""
        };
        format!("x86_64{fma}")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::env::consts::ARCH.into()
    }
}

impl KvIo for Qwen2<'_> {
    fn kv_ident(&self, c: &KvCache) -> Option<Value> {
        // deepseek2's K-only MLA cache has no image format yet: a saved
        // session is prefilled again.
        if self.is_mla() {
            return None;
        }
        let (layers, kvd, hd) = self.kv_shape();
        let (name, size) = c.elem();
        Some(json!({
            "backend": "cpu",
            "class": cpu_class(),
            "numerics": format!("cpu {NUMERICS}"),
            "kv": name,
            "layers": layers, "kv_dim": kvd, "head_dim": hd,
            "row_bytes": 2 * layers * kvd * size,
        }))
    }

    fn kv_save(&self, c: &KvCache, lo: usize, hi: usize, out: &mut dyn Write) -> io::Result<()> {
        check_range(lo, hi, c.max_ctx())?;
        let row = self.kv_shape().1 * c.elem().1;
        let arrays = c.arrays();
        for (s, e) in segments(lo, hi) {
            for a in &arrays {
                out.write_all(&a[s * row..e * row])?;
            }
        }
        Ok(())
    }

    fn kv_load(&self, c: &mut KvCache, lo: usize, hi: usize, src: &mut dyn Read) -> io::Result<()> {
        check_range(lo, hi, c.max_ctx())?;
        let row = self.kv_shape().1 * c.elem().1;
        let mut arrays = c.arrays_mut();
        for (s, e) in segments(lo, hi) {
            for a in &mut arrays {
                src.read_exact(&mut a[s * row..e * row])?;
            }
        }
        Ok(())
    }

    fn kv_shape(&self) -> (usize, usize, usize) {
        (
            self.n_layers,
            self.n_kv_heads * self.head_dim,
            self.head_dim,
        )
    }
}

fn check_range(lo: usize, hi: usize, max_ctx: usize) -> io::Result<()> {
    if lo > hi || hi > max_ctx {
        return Err(io::Error::other(format!(
            "positions {lo}..{hi} of a {max_ctx}-position cache"
        )));
    }
    Ok(())
}

impl<'g> KvIo for CudaQwen2<'g> {
    fn kv_ident(&self, c: &CudaCache<'g>) -> Option<Value> {
        // A split model's GPU half has no image of its own (the CPU half's
        // rows are elsewhere); deepseek2's K-only MLA cache has no image
        // format yet (as the CPU's `Qwen2`): a saved session is prefilled
        // again (#159).
        if self.first > 0 || self.mla.is_some() {
            return None;
        }
        let (layers, kvd, hd) = self.kv_shape();
        let num = crate::cuda::numerics_tag(self.gpu(), self.fast()).ok()?;
        let driver = crate::cuda::driver_tag(self.gpu()).unwrap_or_default();
        let class = self.gpu().info().ok()?.arch();
        Some(json!({
            "backend": "cuda",
            "class": class,
            "numerics": format!("{num} host {NUMERICS}"),
            // Not compared: drivers and NVRTC versions measured to give the
            // same bits (xarch audit).
            "info": { "driver": driver },
            "kv": self.cache_kv_name(c),
            "layers": layers, "kv_dim": kvd, "head_dim": hd,
            "row_bytes": self.cache_row_bytes(c),
        }))
    }

    fn kv_save(
        &self,
        c: &CudaCache<'g>,
        lo: usize,
        hi: usize,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        self.save_kv(c, lo, hi, out)
    }

    fn kv_load(
        &self,
        c: &mut CudaCache<'g>,
        lo: usize,
        hi: usize,
        src: &mut dyn Read,
    ) -> io::Result<()> {
        self.load_kv(c, lo, hi, src)
    }

    fn kv_shape(&self) -> (usize, usize, usize) {
        self.kv_dims()
    }
}

impl KvIo for Backend<'_> {
    /// The model's (`Qwen2`'s, `CudaQwen2`'s) and, in the fast numerics
    /// mode, `numerics_mode` ("fast/1.1"): a session saved in one mode is
    /// restored into the other as tokens (prefilled again), never as bytes.
    /// Exact mode adds nothing, so files saved before the modes still load.
    fn kv_ident(&self, cache: &BackendCache) -> Option<Value> {
        let mut v = match (&self.0, &cache.0) {
            (Inner::Cpu(n), CacheInner::Cpu(c)) => n.kv_ident(c),
            (Inner::Cuda(n), CacheInner::Cuda(c)) => n.kv_ident(c),
            _ => None,
        }?;
        if let (Numerics::Fast, Value::Object(o)) = (self.numerics(), &mut v) {
            o.insert("numerics_mode".into(), json!(self.numerics().tag()));
        }
        Some(v)
    }

    fn kv_save(
        &self,
        cache: &BackendCache,
        lo: usize,
        hi: usize,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        match (&self.0, &cache.0) {
            (Inner::Cpu(n), CacheInner::Cpu(c)) => n.kv_save(c, lo, hi, out),
            (Inner::Cuda(n), CacheInner::Cuda(c)) => n.kv_save(c, lo, hi, out),
            _ => Err(io::Error::other("this backend has no cache image")),
        }
    }

    fn kv_load(
        &self,
        cache: &mut BackendCache,
        lo: usize,
        hi: usize,
        src: &mut dyn Read,
    ) -> io::Result<()> {
        match (&self.0, &mut cache.0) {
            (Inner::Cpu(n), CacheInner::Cpu(c)) => n.kv_load(c, lo, hi, src),
            (Inner::Cuda(n), CacheInner::Cuda(c)) => n.kv_load(c, lo, hi, src),
            _ => Err(io::Error::other("this backend has no cache image")),
        }
    }

    fn kv_shape(&self) -> (usize, usize, usize) {
        match &self.0 {
            Inner::Cpu(n) => n.kv_shape(),
            Inner::Cuda(n) => n.kv_shape(),
            // No image (`kv_ident` is None): nothing depends on the shape.
            // A MoE model (attention on the GPU, experts on the CPU) has no
            // cache image yet either: chat's session save/restore is off.
            Inner::Hybrid(_) | Inner::Pipeline(_) | Inner::Moe(_) => (0, 0, 0),
        }
    }
}

/// The full identity of a cache image: the model's and the backend's
/// (`KvIo::kv_ident`), or None if the backend has none.
pub fn identity(model: &Value, backend: Option<Value>) -> Option<Value> {
    let mut b = backend?;
    let (Value::Object(o), Value::Object(m)) = (&mut b, model) else {
        return None;
    };
    for (k, v) in m {
        o.insert(k.clone(), v.clone());
    }
    o.insert("format".into(), json!(VERSION));
    Some(b)
}

/// The part of an identity that must match: all but `info` (recorded for
/// people, e.g. the driver version, which doesn't change the bits).
fn compared(v: &Value) -> Value {
    let mut v = v.clone();
    if let Value::Object(o) = &mut v {
        o.remove("info");
    }
    v
}

/// The hash binding a KV record of positions ..hi to the tokens whose
/// cache it is (`tokens[..hi]`): a record of other tokens never loads.
fn token_hash(tokens: &[u32]) -> u64 {
    let mut h = Hasher::default();
    h.update(b"axkv tokens");
    for t in tokens {
        h.update(&t.to_le_bytes());
    }
    h.finish()
}

/// What the file holds, as last written or read by this process.
struct Saved {
    /// The KV records cover positions 0..kv, the cache of tokens[..kv].
    kv: usize,
    tokens: Vec<u32>,
    /// Where the last good STATE ends: the file's length after our last
    /// save, or what a read found good (a torn or orphaned tail after it
    /// is cut off before the next append).
    len: u64,
    /// `len` came from a read: bytes past it are a tail to cut off. After
    /// our own save they can only be another writer's (a rewrite), so the
    /// file is then rewritten, never cut at our old length.
    from_read: bool,
    /// The file's (device, inode) then: a file replaced by another writer
    /// (a rename over it) is rewritten, not appended to.
    id: Option<(u64, u64)>,
    /// Bytes of STATE records no longer current, and of the current one.
    stale: u64,
    last_state: u64,
}

/// Errors that mean "this filesystem can't lock", not "someone holds it":
/// no locks (ENOLCK), unsupported (EOPNOTSUPP), no permission, read-only.
fn lock_unsupported(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::Unsupported
            | io::ErrorKind::PermissionDenied
            | io::ErrorKind::ReadOnlyFilesystem
    ) || matches!(e.raw_os_error(), Some(37 | 95 | 13 | 30))
}

/// The file's (device, inode), to tell a file replaced under us from ours.
fn file_id(m: &std::fs::Metadata) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some((m.dev(), m.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = m;
        None
    }
}

/// `p` made absolute with symlinks resolved, so `./F`, `F` and a link to it
/// are one session; for a file not there yet, its directory's canonical
/// path with the name appended (else `p` as given).
pub fn canonical(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    let dir = match p.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    match (std::fs::canonicalize(dir), p.file_name()) {
        (Ok(d), Some(n)) => d.join(n),
        _ => p.to_path_buf(),
    }
}

/// A session file (`axon chat --session`).
pub struct Session {
    pub path: PathBuf,
    /// `model_ident`.
    model: Value,
    saved: Option<Saved>,
    /// The advisory lock on `path` + ".lock" (`lock`), held while we live.
    lock: Option<File>,
}

/// What `Session::save` did.
#[derive(Debug, Default)]
pub struct SaveInfo {
    /// Rewrote the file (else appended).
    pub rewrote: bool,
    /// KV positions written.
    pub kv_written: usize,
    /// Bytes written.
    pub bytes: u64,
}

/// A KV record in a file.
#[derive(Clone, Copy, Debug)]
struct Rec {
    lo: usize,
    hi: usize,
    /// `token_hash` of the tokens ..hi.
    tokens: u64,
    /// Offset and length of the payload, and its check.
    at: u64,
    n: u64,
    check: u64,
}

/// A file's records (`Session::read`).
pub struct Loaded {
    /// The last good STATE (with `kv`).
    pub state: Value,
    /// Why the file's cache can't be loaded here (another identity, or
    /// none), if it can't.
    ident_err: Option<String>,
    /// The KV records before the end of the last good STATE.
    recs: Vec<Rec>,
    /// Where the last good STATE ends.
    good_end: u64,
    /// The file's (device, inode) when read.
    id: Option<(u64, u64)>,
    stale: u64,
    last_state: u64,
}

/// Bytes of a record's framing: kind, length, check.
const FRAME: u64 = 17;
/// lo, hi and the token hash before a KV record's image.
const KV_HEAD: u64 = 24;
/// Largest header read.
const MAX_HEADER: u64 = 1 << 20;

/// Rewrite the file once its stale STATE records pass this many bytes.
const MAX_STALE: u64 = 64 << 20;

fn record(w: &mut impl Write, kind: u8, payload: &[u8]) -> io::Result<()> {
    w.write_all(&[kind])?;
    w.write_all(&(payload.len() as u64).to_le_bytes())?;
    w.write_all(payload)?;
    w.write_all(&Hasher::of(payload).to_le_bytes())
}

type SaveFn<'a> = dyn FnMut(usize, usize, &mut dyn Write) -> io::Result<()> + 'a;
type LoadFn<'a> = dyn FnMut(usize, usize, &mut dyn Read) -> io::Result<()> + 'a;

/// A KV record of positions lo..hi of the cache of `tokens` (..hi; `row`
/// image bytes each).
fn kv_record(
    w: &mut impl Write,
    tokens: &[u32],
    lo: usize,
    row: usize,
    save: &mut SaveFn,
) -> io::Result<()> {
    let hi = tokens.len();
    let n = ((hi - lo) * row) as u64;
    w.write_all(&[KIND_KV])?;
    w.write_all(&(KV_HEAD + n).to_le_bytes())?;
    let mut hw = HashW {
        w,
        h: Hasher::default(),
        n: 0,
    };
    hw.write_all(&(lo as u64).to_le_bytes())?;
    hw.write_all(&(hi as u64).to_le_bytes())?;
    hw.write_all(&token_hash(tokens).to_le_bytes())?;
    save(lo, hi, &mut hw)?;
    if hw.n != KV_HEAD + n {
        return Err(io::Error::other(format!(
            "a KV image of {} bytes, {} expected",
            hw.n - KV_HEAD,
            n
        )));
    }
    let c = hw.h.finish();
    w.write_all(&c.to_le_bytes())
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// A new file for writing, readable by its owner only (it holds the
/// conversation).
fn create_private(p: &Path) -> io::Result<File> {
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    o.open(p)
}

impl Session {
    pub fn new(path: PathBuf, model: Value) -> Session {
        Session {
            path,
            model,
            saved: None,
            lock: None,
        }
    }

    /// Take the session's advisory exclusive lock, held until the Session
    /// drops. It is a sidecar, `F.lock` (0600), because rewrites replace F
    /// itself; it is never removed (unlinking a lock file races with the
    /// next process opening it). Err(WouldBlock) if another holder has it
    /// (another process, or another Session in this one): two writers would
    /// rewrite each other's saves. Ok(Some(why)) if this filesystem can't
    /// lock (no flock, read-only, no permission): the session runs
    /// unlocked, which `load` keeps safe (see `save`).
    pub fn lock(&mut self) -> io::Result<Option<String>> {
        if self.lock.is_some() {
            return Ok(None);
        }
        let mut p = self.path.clone().into_os_string();
        p.push(".lock");
        let p = PathBuf::from(p);
        let mut o = std::fs::OpenOptions::new();
        o.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
        let opened = match o.open(&p) {
            // flock needs no write access: an existing read-only lock file
            // (another user's directory, a read-only share) still locks.
            Err(e) if e.kind() == io::ErrorKind::PermissionDenied => File::open(&p).map_err(|_| e),
            r => r,
        };
        let unavailable = |e: io::Error| -> io::Result<Option<String>> {
            if lock_unsupported(&e) {
                Ok(Some(format!(
                    "session lock unavailable on this filesystem ({e}); running without a lock"
                )))
            } else {
                Err(e)
            }
        };
        let f = match opened {
            Ok(f) => f,
            Err(e) => return unavailable(e),
        };
        match f.try_lock() {
            Ok(()) => {
                self.lock = Some(f);
                Ok(None)
            }
            Err(std::fs::TryLockError::WouldBlock) => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "session {} is in use by another axon process (or another session in this one)",
                    self.path.display()
                ),
            )),
            Err(std::fs::TryLockError::Error(e)) => unavailable(e),
        }
    }

    /// Set the model identity (`model_ident`) once it is known: the lock is
    /// taken before the model loads, so a session in use fails fast.
    pub fn set_model(&mut self, model: Value) {
        self.model = model;
    }

    pub fn model(&self) -> &Value {
        &self.model
    }

    /// Save `state` (JSON; this adds `kv`) with the cache's positions
    /// 0..n holding `tokens` (n = its length), their image written by
    /// `save(lo, hi, out)`, under `ident` (None: tokens only, `kv` 0).
    /// Appends the new positions if the file holds a prefix of them (after
    /// cutting off anything past its last good STATE), else rewrites it
    /// (through a temporary file renamed over it).
    pub fn save(
        &mut self,
        ident: Option<&Value>,
        tokens: &[u32],
        save: &mut SaveFn,
        state: &Value,
    ) -> io::Result<SaveInfo> {
        let n = if ident.is_some() { tokens.len() } else { 0 };
        let row = ident.and_then(|i| i["row_bytes"].as_u64()).unwrap_or(0) as usize;
        let mut state = state.clone();
        state["kv"] = json!(n);
        let st = serde_json::to_vec(&state).map_err(io::Error::other)?;
        // Appending is an optimisation only. The contract (never restore a
        // wrong cache) rests on `load`: it tiles records whose hash covers
        // the whole token prefix they cache, so bytes written by another
        // writer never load for our tokens whatever this check misses (a
        // same-length rewrite in place). `from_read` and the file id only
        // keep another writer's saves from being cut (less data lost).
        let cur = std::fs::metadata(&self.path).ok();
        let cur_len = cur.as_ref().map(|m| m.len());
        let cur_id = cur.as_ref().and_then(file_id);
        let append = self.saved.as_ref().filter(|s| {
            s.kv <= n
                && tokens[..s.kv] == s.tokens[..s.kv]
                && cur_len.is_some_and(|c| c == s.len || (s.from_read && c > s.len))
                && cur_id == s.id
                && s.stale + s.last_state <= MAX_STALE
        });
        let mut info = SaveInfo::default();
        let (from, stale, before) = match append {
            Some(s) => {
                let mut f = std::fs::OpenOptions::new().write(true).open(&self.path)?;
                // A torn or orphaned tail (a save that never finished)
                // goes: what follows must come right after the last STATE.
                if cur_len != Some(s.len) {
                    f.set_len(s.len)?;
                }
                f.seek(SeekFrom::Start(s.len))?;
                let mut w = BufWriter::with_capacity(1 << 20, f);
                if n > s.kv {
                    kv_record(&mut w, &tokens[..n], s.kv, row, save)?;
                }
                record(&mut w, KIND_STATE, &st)?;
                w.into_inner().map_err(|e| e.into_error())?.sync_data()?;
                (s.kv, s.stale + s.last_state, s.len)
            }
            None => {
                info.rewrote = true;
                let mut tmp = self.path.clone().into_os_string();
                tmp.push(".tmp");
                let tmp = PathBuf::from(tmp);
                // A leftover one may have other permissions: start afresh.
                let _ = std::fs::remove_file(&tmp);
                let mut w = BufWriter::with_capacity(1 << 20, create_private(&tmp)?);
                let hdr =
                    serde_json::to_vec(&json!({ "identity": ident })).map_err(io::Error::other)?;
                w.write_all(MAGIC)?;
                w.write_all(&VERSION.to_le_bytes())?;
                w.write_all(&(hdr.len() as u32).to_le_bytes())?;
                w.write_all(&hdr)?;
                if n > 0 {
                    kv_record(&mut w, &tokens[..n], 0, row, save)?;
                }
                record(&mut w, KIND_STATE, &st)?;
                w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
                std::fs::rename(&tmp, &self.path)?;
                // The rename itself, durable.
                let dir = match self.path.parent() {
                    Some(d) if !d.as_os_str().is_empty() => d,
                    _ => Path::new("."),
                };
                if let Ok(d) = File::open(dir) {
                    let _ = d.sync_all();
                }
                (0, 0, 0)
            }
        };
        let meta = std::fs::metadata(&self.path)?;
        let len = meta.len();
        info.kv_written = n - from;
        info.bytes = len - before;
        self.saved = Some(Saved {
            kv: n,
            tokens: tokens[..n].to_vec(),
            len,
            from_read: false,
            id: file_id(&meta),
            stale,
            last_state: FRAME + st.len() as u64,
        });
        Ok(info)
    }

    /// The file's records, its identity checked against `ident` (None:
    /// the backend has no image). Ok(None) if there is no file. Err only if
    /// it isn't a session file or holds no conversation.
    pub fn read(&self, ident: Option<&Value>) -> io::Result<Option<Loaded>> {
        let f = match File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let meta = f.metadata()?;
        let len = meta.len();
        let mut r = BufReader::with_capacity(1 << 16, f);
        let mut head = [0u8; 12];
        r.read_exact(&mut head)
            .map_err(|_| bad("not an axon session file"))?;
        if &head[..4] != MAGIC {
            return Err(bad("not an axon session file"));
        }
        let ver = u32::from_le_bytes(head[4..8].try_into().expect("4"));
        if ver != VERSION {
            return Err(bad(format!(
                "session file version {ver}; this axon reads {VERSION}"
            )));
        }
        let hl = u32::from_le_bytes(head[8..12].try_into().expect("4")) as u64;
        if hl > MAX_HEADER || hl > len - 12 {
            return Err(bad("a damaged session file header"));
        }
        let mut hdr = vec![0u8; hl as usize];
        r.read_exact(&mut hdr)?;
        let hdr: Value = serde_json::from_slice(&hdr).map_err(|e| bad(format!("header: {e}")))?;
        let file_ident = &hdr["identity"];
        let row = file_ident["row_bytes"].as_u64().unwrap_or(0);
        let mut pos = 12 + hl;
        let mut recs = Vec::new();
        let mut state: Option<(Value, u64)> = None;
        let (mut stale, mut good_end) = (0, 0);
        // Records until the end, or the first torn or failing one.
        loop {
            let mut fr = [0u8; 9];
            if pos + FRAME > len || r.read_exact(&mut fr).is_err() {
                break;
            }
            let n = u64::from_le_bytes(fr[1..9].try_into().expect("8"));
            if n > len - pos - FRAME {
                break;
            }
            let at = pos + 9;
            match fr[0] {
                KIND_KV if n >= KV_HEAD => {
                    let mut h = [0u8; 24];
                    r.read_exact(&mut h)?;
                    let w =
                        |i: usize| u64::from_le_bytes(h[i * 8..i * 8 + 8].try_into().expect("8"));
                    let (lo, hi) = (w(0), w(1));
                    if hi < lo || (hi - lo).checked_mul(row) != Some(n - KV_HEAD) {
                        break;
                    }
                    r.seek_relative((n - KV_HEAD) as i64)?;
                    let mut c = [0u8; 8];
                    r.read_exact(&mut c)?;
                    recs.push(Rec {
                        lo: lo as usize,
                        hi: hi as usize,
                        tokens: w(2),
                        at,
                        n,
                        check: u64::from_le_bytes(c),
                    });
                }
                KIND_STATE => {
                    let mut p = vec![0u8; n as usize];
                    let mut c = [0u8; 8];
                    r.read_exact(&mut p)?;
                    r.read_exact(&mut c)?;
                    if Hasher::of(&p) != u64::from_le_bytes(c) {
                        break;
                    }
                    let Ok(v) = serde_json::from_slice(&p) else {
                        break;
                    };
                    if let Some((_, l)) = state {
                        stale += l;
                    }
                    state = Some((v, FRAME + n));
                    good_end = at + n + 8;
                }
                _ => break,
            }
            pos = at + n + 8;
        }
        let Some((state, last_state)) = state else {
            return Err(bad("no conversation in the session file"));
        };
        // Records after the last STATE belong to a save that never finished.
        recs.retain(|r| r.at < good_end);
        let ident_err = if file_ident.is_null() {
            Some("the file holds no cache".into())
        } else {
            match ident {
                None => Some("this backend has no cache image".into()),
                Some(i) if compared(i) != compared(file_ident) => {
                    Some(format!("it was made under {}", diff(file_ident, i)))
                }
                Some(_) => None,
            }
        };
        Ok(Some(Loaded {
            state,
            ident_err,
            recs,
            good_end,
            id: file_id(&meta),
            stale,
            last_state,
        }))
    }

    /// Load the cache of `tokens` (positions 0..tokens.len(), the state's
    /// `kv`) from `l` through `load(lo, hi, src)`: records of exactly these
    /// tokens tiling 0..kv (the latest one per start), each record's hash
    /// checked as it streams. Err(why) if the identity differs, the records
    /// don't cover them, or one fails (the rows may then hold anything:
    /// prefill them again). On success the file is the session's (`save`
    /// appends to it).
    pub fn load(&mut self, l: &Loaded, tokens: &[u32], load: &mut LoadFn) -> io::Result<()> {
        if let Some(e) = &l.ident_err {
            return Err(io::Error::other(e.clone()));
        }
        let kv = tokens.len();
        let mut tiles = Vec::new();
        let mut end = 0;
        while end < kv {
            let next = l
                .recs
                .iter()
                .rev()
                .find(|r| {
                    r.lo == end
                        && r.hi > end
                        && r.hi <= kv
                        && r.tokens == token_hash(&tokens[..r.hi])
                })
                .ok_or_else(|| {
                    io::Error::other("its cache records don't cover the conversation")
                })?;
            tiles.push(*next);
            end = next.hi;
        }
        let mut f = BufReader::with_capacity(1 << 20, File::open(&self.path)?);
        for r in tiles {
            f.seek(SeekFrom::Start(r.at))?;
            let mut take = (&mut f).take(r.n);
            let mut hr = HashR {
                r: &mut take,
                h: Hasher::default(),
            };
            let mut head = [0u8; KV_HEAD as usize];
            hr.read_exact(&mut head)?;
            load(r.lo, r.hi, &mut hr)?;
            if hr.h.finish() != r.check {
                return Err(bad(format!(
                    "cache record {}..{} fails its check",
                    r.lo, r.hi
                )));
            }
        }
        self.saved = Some(Saved {
            kv,
            tokens: tokens.to_vec(),
            len: l.good_end,
            from_read: true,
            id: l.id,
            stale: l.stale,
            last_state: l.last_state,
        });
        Ok(())
    }

    /// Forget what the file holds: the next `save` rewrites it.
    pub fn forget(&mut self) {
        self.saved = None;
    }
}

/// The identity fields that differ, for a message: "key file (now ours)".
fn diff(file: &Value, ours: &Value) -> String {
    let (Value::Object(a), Value::Object(b)) = (compared(file), compared(ours)) else {
        return "another identity".into();
    };
    let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
    keys.sort();
    keys.dedup();
    let d: Vec<String> = keys
        .into_iter()
        .filter(|k| a.get(*k) != b.get(*k))
        .map(|k| {
            let s = |v: Option<&Value>| v.map_or("-".into(), |v| v.to_string());
            format!("{k} {} (now {})", s(a.get(k)), s(b.get(k)))
        })
        .collect();
    d.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_split_at_pages() {
        assert_eq!(segments(0, 0), vec![]);
        assert_eq!(segments(0, 64), vec![(0, 64)]);
        assert_eq!(
            segments(10, 200),
            vec![(10, 64), (64, 128), (128, 192), (192, 200)]
        );
        assert_eq!(segments(64, 65), vec![(64, 65)]);
    }

    #[test]
    fn hasher_streams() {
        let d: Vec<u8> = (0..1000u32).map(|i| (i * 7 + i / 13) as u8).collect();
        let one = Hasher::of(&d);
        for cut in [0, 1, 7, 31, 32, 33, 500, 999, 1000] {
            let mut h = Hasher::default();
            h.update(&d[..cut]);
            h.update(&d[cut..]);
            assert_eq!(h.finish(), one, "cut {cut}");
        }
        let mut e = d.clone();
        e[999] ^= 1;
        assert_ne!(Hasher::of(&e), one);
        assert_ne!(Hasher::of(&d[..999]), one);
        assert_ne!(Hasher::of(&[0; 8]), Hasher::of(&[0; 9]));
    }

    /// A fake cache of `toks`: position p's image is `row` bytes of
    /// (token, p, byte index).
    fn fake_save(
        row: usize,
        toks: &[u32],
    ) -> impl FnMut(usize, usize, &mut dyn Write) -> io::Result<()> {
        move |lo, hi, w| {
            for (p, &t) in toks.iter().enumerate().take(hi).skip(lo) {
                let b: Vec<u8> = (0..row)
                    .map(|i| (t as usize * 7 + p * 31 + i) as u8)
                    .collect();
                w.write_all(&b)?;
            }
            Ok(())
        }
    }

    fn image(row: usize, toks: &[u32]) -> Vec<u8> {
        let mut v = Vec::new();
        fake_save(row, toks)(0, toks.len(), &mut v).unwrap();
        v
    }

    /// Read `s`'s file and load its cache for `toks[..kv]` (kv from the
    /// state): (state, image bytes) or the load's error.
    fn reload(
        s: &mut Session,
        ident: &Value,
        toks: &[u32],
        row: usize,
    ) -> (Value, io::Result<Vec<u8>>) {
        let l = s.read(Some(ident)).unwrap().unwrap();
        let kv = l.state["kv"].as_u64().unwrap() as usize;
        let mut got = Vec::new();
        let r = s.load(&l, &toks[..kv], &mut |lo, hi, src| {
            let mut b = vec![0u8; (hi - lo) * row];
            src.read_exact(&mut b)?;
            got.extend(b);
            Ok(())
        });
        (l.state, r.map(|_| got))
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("axon-kvsave-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(tag);
        let _ = std::fs::remove_file(&p);
        p
    }

    const ROW: usize = 40;

    fn toks() -> (Vec<u32>, Vec<u32>) {
        let a: Vec<u32> = (0..300).collect();
        let mut b = a.clone();
        for t in &mut b[100..] {
            *t += 1000;
        }
        (a, b)
    }

    fn save(s: &mut Session, ident: &Value, toks: &[u32], n: u32) -> SaveInfo {
        s.save(
            Some(ident),
            toks,
            &mut fake_save(ROW, toks),
            &json!({ "n": n }),
        )
        .unwrap()
    }

    /// Saves append while the tokens extend, rewrite otherwise; a new
    /// reader sees the last state and appends too; another identity loads
    /// no bytes (but `info` isn't compared); a flipped byte fails the load.
    #[test]
    fn file_appends_rewrites_and_checks() {
        let ident = json!({"model": "x", "row_bytes": ROW, "info": {"driver": "1"}});
        let path = scratch("f.axkv");
        let (t, other) = toks();
        let mut s = Session::new(path.clone(), json!({}));
        let i1 = save(&mut s, &ident, &t[..100], 1);
        assert!(i1.rewrote && i1.kv_written == 100);
        let i2 = save(&mut s, &ident, &t[..250], 2);
        assert!(!i2.rewrote && i2.kv_written == 150, "{i2:?}");
        let i3 = save(&mut s, &ident, &t[..250], 3);
        assert!(!i3.rewrote && i3.kv_written == 0);

        let mut r = Session::new(path.clone(), json!({}));
        let newer_driver = json!({"model": "x", "row_bytes": ROW, "info": {"driver": "2"}});
        let (st, img) = reload(&mut r, &newer_driver, &t, ROW);
        assert_eq!(st["n"], 3);
        assert_eq!(img.unwrap(), image(ROW, &t[..250]));
        // The reader's session appends.
        assert!(!save(&mut r, &ident, &t[..260], 4).rewrote);
        // Other tokens: rewritten.
        assert!(save(&mut r, &ident, &other[..120], 5).rewrote);
        let (st, img) = reload(
            &mut Session::new(path.clone(), json!({})),
            &ident,
            &other,
            ROW,
        );
        assert_eq!(st["n"], 5);
        assert_eq!(img.unwrap(), image(ROW, &other[..120]));

        // Another identity: tokens only.
        let o = json!({"model": "y", "row_bytes": ROW});
        let (_, e) = reload(&mut Session::new(path.clone(), json!({})), &o, &other, ROW);
        assert!(e.unwrap_err().to_string().contains("model"));

        // A flipped KV byte: the load fails (the caller prefills again).
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes.len() / 2;
        bytes[at] ^= 0x40;
        std::fs::write(&path, &bytes).unwrap();
        let (_, e) = reload(
            &mut Session::new(path.clone(), json!({})),
            &ident,
            &other,
            ROW,
        );
        assert!(e.is_err());
        let _ = std::fs::remove_file(&path);
    }

    /// A save torn at any byte: a new process loads the save before it,
    /// saves on (the torn tail cut off, then appended to), and the next
    /// process loads that latest state exactly.
    #[test]
    fn torn_tail_then_new_process_saves_on() {
        let ident = json!({"model": "x", "row_bytes": ROW});
        let path = scratch("torn.axkv");
        let (t, _) = toks();
        let mut s = Session::new(path.clone(), json!({}));
        save(&mut s, &ident, &t[..100], 1);
        let good = std::fs::metadata(&path).unwrap().len();
        save(&mut s, &ident, &t[..250], 2);
        let len = std::fs::metadata(&path).unwrap().len();
        let whole = std::fs::read(&path).unwrap();
        for cut in [1, 9, 30, len - good - 1] {
            std::fs::write(&path, &whole[..(len - cut) as usize]).unwrap();
            let mut r = Session::new(path.clone(), json!({}));
            let (st, img) = reload(&mut r, &ident, &t, ROW);
            assert_eq!(st["n"], 1, "cut {cut}");
            assert_eq!(img.unwrap(), image(ROW, &t[..100]), "cut {cut}");
            let i = save(&mut r, &ident, &t[..260], 3);
            assert!(!i.rewrote, "cut {cut}: appended after the cut");
            let (st, img) = reload(&mut Session::new(path.clone(), json!({})), &ident, &t, ROW);
            assert_eq!(st["n"], 3, "cut {cut}");
            assert_eq!(img.unwrap(), image(ROW, &t[..260]), "cut {cut}");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A complete KV record with no STATE after it (a save cut between
    /// them) of other tokens at the same positions: never loaded. A new
    /// process cuts it off; and even appended after (an old writer), the
    /// records are matched by their tokens, not their positions.
    #[test]
    fn orphan_records_never_load() {
        let ident = json!({"model": "x", "row_bytes": ROW});
        let path = scratch("orphan.axkv");
        let (t, other) = toks();
        let orphan = |path: &Path| {
            let mut f = std::fs::OpenOptions::new().append(true).open(path).unwrap();
            kv_record(&mut f, &other[..250], 100, ROW, &mut fake_save(ROW, &other)).unwrap();
        };
        let mut s = Session::new(path.clone(), json!({}));
        save(&mut s, &ident, &t[..100], 1);
        let good = std::fs::metadata(&path).unwrap().len();
        orphan(&path);
        let mut r = Session::new(path.clone(), json!({}));
        let (st, img) = reload(&mut r, &ident, &t, ROW);
        assert_eq!(
            (st["n"].clone(), img.unwrap()),
            (json!(1), image(ROW, &t[..100]))
        );
        save(&mut r, &ident, &t[..250], 2);
        let (st, img) = reload(&mut Session::new(path.clone(), json!({})), &ident, &t, ROW);
        assert_eq!(
            (st["n"].clone(), img.unwrap()),
            (json!(2), image(ROW, &t[..250]))
        );

        // Without the cut: a writer appending after the orphan.
        let mut s = Session::new(path.clone(), json!({}));
        save(&mut s, &ident, &t[..100], 1);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), good);
        orphan(&path);
        s.saved.as_mut().unwrap().len = std::fs::metadata(&path).unwrap().len();
        assert!(!save(&mut s, &ident, &t[..250], 2).rewrote);
        let (st, img) = reload(&mut Session::new(path.clone(), json!({})), &ident, &t, ROW);
        assert_eq!(
            (st["n"].clone(), img.unwrap()),
            (json!(2), image(ROW, &t[..250]))
        );
        // Tokens no record holds: nothing loads.
        let mut s2 = Session::new(path.clone(), json!({}));
        let l = s2.read(Some(&ident)).unwrap().unwrap();
        let mut third = t.clone();
        third[150] = 7;
        assert!(s2.load(&l, &third[..250], &mut |_, _, _| Ok(())).is_err());
        let _ = std::fs::remove_file(&path);
    }

    /// Two writers on one file (no lock taken): after B rewrites it
    /// longer, A's next save rewrites it rather than cut it at A's old
    /// length, and the file loads A's latest state. A leftover .tmp with
    /// other permissions doesn't make the file readable by others.
    #[test]
    fn second_writer_then_save_rewrites() {
        let ident = json!({"model": "x", "row_bytes": ROW});
        let path = scratch("two.axkv");
        let (t, other) = toks();
        let mut a = Session::new(path.clone(), json!({}));
        save(&mut a, &ident, &t[..100], 1);
        let len_a = std::fs::metadata(&path).unwrap().len();
        let mut b = Session::new(path.clone(), json!({}));
        let (st, img) = reload(&mut b, &ident, &t, ROW);
        assert_eq!(
            (st["n"].clone(), img.unwrap()),
            (json!(1), image(ROW, &t[..100]))
        );
        // Another conversation from the start: B rewrites the file.
        let mut third = other.clone();
        third[5] = 4242;
        assert!(save(&mut b, &ident, &third[..250], 2).rewrote);
        assert!(std::fs::metadata(&path).unwrap().len() > len_a);
        let mut tmp = path.clone().into_os_string();
        tmp.push(".tmp");
        std::fs::write(&tmp, b"leftover").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644)).unwrap();
        }
        let i = save(&mut a, &ident, &t[..150], 3);
        assert!(i.rewrote, "{i:?}");
        let (st, img) = reload(&mut Session::new(path.clone(), json!({})), &ident, &t, ROW);
        assert_eq!(
            (st["n"].clone(), img.unwrap()),
            (json!(3), image(ROW, &t[..150]))
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let _ = std::fs::remove_file(&path);
    }

    /// The safety property itself: another writer rewrites the file in
    /// place to exactly our length with other tokens (so neither the length
    /// nor the file id tells), we append and save our state; a reload of our
    /// state must not hand back the other writer's prefix: its load fails
    /// (the caller prefills again). Never a wrong cache.
    #[test]
    fn same_length_foreign_rewrite_never_restores_wrong_state() {
        let ident = json!({"model": "x", "row_bytes": ROW});
        let path = scratch("same-len.axkv");
        let other_path = scratch("same-len-b.axkv");
        let (t, _) = toks();
        let mut foreign = t.clone();
        foreign[3] = 999;
        let mut a = Session::new(path.clone(), json!({}));
        save(&mut a, &ident, &t[..100], 1);
        let mut b = Session::new(other_path.clone(), json!({}));
        save(&mut b, &ident, &foreign[..100], 1);
        let bytes = std::fs::read(&other_path).unwrap();
        assert_eq!(bytes.len() as u64, std::fs::metadata(&path).unwrap().len());
        std::fs::write(&path, &bytes).unwrap(); // in place: same inode
        assert!(
            !save(&mut a, &ident, &t[..150], 2).rewrote,
            "appended: nothing tells"
        );
        let (st, img) = reload(&mut Session::new(path.clone(), json!({})), &ident, &t, ROW);
        assert_eq!(st["n"], 2);
        let e = img.expect_err("must not load another writer's prefix");
        assert!(e.to_string().contains("don't cover"), "{e}");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&other_path);
    }

    #[test]
    fn canonical_names_one_session() {
        let dir = scratch("canon-dir");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("s.axkv");
        // Not there yet: the directory's canonical path plus the name.
        let before = canonical(&f);
        assert_eq!(before, std::fs::canonicalize(&dir).unwrap().join("s.axkv"));
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(canonical(&f), before);
        assert_eq!(canonical(&dir.join(".").join("s.axkv")), before);
        #[cfg(unix)]
        {
            let link = dir.join("link.axkv");
            std::os::unix::fs::symlink(&f, &link).unwrap();
            assert_eq!(canonical(&link), before);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One process per session: a second lock fails with a clear message
    /// until the first Session drops; a lock file we can only read still
    /// locks (flock needs no write access).
    #[test]
    fn lock_refuses_a_second_holder() {
        let path = scratch("locked.axkv");
        let mut x = Session::new(path.clone(), json!({}));
        assert_eq!(x.lock().unwrap(), None);
        assert_eq!(x.lock().unwrap(), None); // held already: fine
        let mut y = Session::new(path.clone(), json!({}));
        let e = y.lock().unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::WouldBlock);
        assert!(
            e.to_string().contains("in use by another axon process"),
            "{e}"
        );
        drop(x);
        assert_eq!(y.lock().unwrap(), None);
        drop(y);
        let mut lock = path.clone().into_os_string();
        lock.push(".lock");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&lock, std::fs::Permissions::from_mode(0o400)).unwrap();
            let mut z = Session::new(path.clone(), json!({}));
            assert_eq!(z.lock().unwrap(), None);
            let e = Session::new(path.clone(), json!({})).lock().unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::WouldBlock);
            drop(z);
            std::fs::set_permissions(&lock, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let _ = std::fs::remove_file(lock);
    }

    /// Damaged headers are errors, never panics or huge allocations.
    #[test]
    fn hostile_headers() {
        let path = scratch("hostile.axkv");
        let mut f = Vec::new();
        f.extend(MAGIC);
        f.extend(VERSION.to_le_bytes());
        for hl in [u32::MAX, 1 << 21, 5] {
            let mut g = f.clone();
            g.extend(hl.to_le_bytes());
            g.extend(b"{}");
            std::fs::write(&path, &g).unwrap();
            assert!(
                Session::new(path.clone(), json!({})).read(None).is_err(),
                "hl {hl}"
            );
        }
        // A good header, then garbage: no conversation.
        let mut g = f.clone();
        g.extend(2u32.to_le_bytes());
        g.extend(b"{}");
        g.extend([
            KIND_STATE, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1, 2, 3,
        ]);
        std::fs::write(&path, &g).unwrap();
        assert!(Session::new(path.clone(), json!({})).read(None).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn not_a_session_file() {
        let path = scratch("junk");
        std::fs::write(&path, b"hello world, not a session").unwrap();
        assert!(Session::new(path.clone(), json!({})).read(None).is_err());
        let _ = std::fs::remove_file(&path);
        assert!(Session::new(path, json!({})).read(None).unwrap().is_none());
    }
}
