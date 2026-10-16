//! KV save/restore: a sequence's cache rows to a file and back, exactly.
//!
//! A cache's bytes are a pure function of the token prefix (decode ==
//! batched prefill == verify, bit for bit), so copying them is exact: a
//! restored cache continues as the uninterrupted one would. They are only
//! that function *for one identity*: the model file, the KV type, the RoPE
//! table, the backend and its numerics (kernel sources, GPU architecture,
//! driver). A file records its identity in the header; one made under
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
use crate::lm::{Backend, BackendCache, Lm};
use crate::model::{KvCache, Qwen2};
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
    let meta = std::fs::metadata(&canon)?;
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
    let id = format!("{:016x}-{}", hash_file(&canon, meta.len())?, meta.len());
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

/// The CPU's arch class: the instruction sets its kernels pick from.
fn cpu_class() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let f = [
            ("avx2", is_x86_feature_detected!("avx2")),
            ("fma", is_x86_feature_detected!("fma")),
            ("f16c", is_x86_feature_detected!("f16c")),
        ];
        let on: Vec<&str> = f.iter().filter(|x| x.1).map(|x| x.0).collect();
        format!("x86_64 {}", on.join(" "))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::env::consts::ARCH.into()
    }
}

impl KvIo for Qwen2<'_> {
    fn kv_ident(&self, c: &KvCache) -> Option<Value> {
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
        // rows are elsewhere).
        if self.first > 0 {
            return None;
        }
        let (layers, kvd, hd) = self.kv_shape();
        let num = crate::cuda::numerics_tag(self.gpu()).ok()?;
        let class = self.gpu().info().ok()?.arch();
        Some(json!({
            "backend": "cuda",
            "class": class,
            "numerics": format!("{num} host {NUMERICS}"),
            "kv": self.cache_kv_name(c),
            "layers": layers, "kv_dim": kvd, "head_dim": hd,
            "row_bytes": self.cache_rows(c).row_bytes(),
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
    fn kv_ident(&self, cache: &BackendCache) -> Option<Value> {
        match (self, cache) {
            (Backend::Cpu(n), BackendCache::Cpu(c)) => n.kv_ident(c),
            (Backend::Cuda(n), BackendCache::Cuda(c)) => n.kv_ident(c),
            _ => None,
        }
    }

    fn kv_save(
        &self,
        cache: &BackendCache,
        lo: usize,
        hi: usize,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        match (self, cache) {
            (Backend::Cpu(n), BackendCache::Cpu(c)) => n.kv_save(c, lo, hi, out),
            (Backend::Cuda(n), BackendCache::Cuda(c)) => n.kv_save(c, lo, hi, out),
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
        match (self, cache) {
            (Backend::Cpu(n), BackendCache::Cpu(c)) => n.kv_load(c, lo, hi, src),
            (Backend::Cuda(n), BackendCache::Cuda(c)) => n.kv_load(c, lo, hi, src),
            _ => Err(io::Error::other("this backend has no cache image")),
        }
    }

    fn kv_shape(&self) -> (usize, usize, usize) {
        match self {
            Backend::Cpu(n) => n.kv_shape(),
            Backend::Cuda(n) => n.kv_shape(),
            // No image (`kv_ident` is None): nothing depends on the shape.
            Backend::Hybrid(_) | Backend::Pipeline(_) => (0, 0, 0),
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

/// What the file holds, as last written or read by this process.
struct Saved {
    /// The KV records cover positions 0..kv, the cache of tokens[..kv].
    kv: usize,
    tokens: Vec<u32>,
    /// The file's length after it (an append checks nobody changed it).
    len: u64,
    /// Bytes of STATE records no longer current, and of the current one.
    stale: u64,
    last_state: u64,
}

/// A session file (`axon chat --session`).
pub struct Session {
    pub path: PathBuf,
    /// `model_ident`.
    model: Value,
    saved: Option<Saved>,
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

/// A file's records (`Session::read`).
pub struct Loaded {
    /// The last STATE (with `kv`).
    pub state: Value,
    /// The file's identity matches and its KV records cover the state's
    /// `kv` positions; else why not.
    pub usable: Result<usize, String>,
    /// KV records: (lo, hi, offset of the payload, payload length, check).
    recs: Vec<(usize, usize, u64, u64, u64)>,
    len: u64,
    stale: u64,
    last_state: u64,
}

/// Bytes of a record's framing: kind, length, check.
const FRAME: u64 = 17;

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

/// A KV record of positions lo..hi (`row` image bytes each).
fn kv_record(
    w: &mut impl Write,
    lo: usize,
    hi: usize,
    row: usize,
    save: &mut SaveFn,
) -> io::Result<()> {
    let n = ((hi - lo) * row) as u64;
    w.write_all(&[KIND_KV])?;
    w.write_all(&(16 + n).to_le_bytes())?;
    let mut hw = HashW {
        w,
        h: Hasher::default(),
        n: 0,
    };
    hw.write_all(&(lo as u64).to_le_bytes())?;
    hw.write_all(&(hi as u64).to_le_bytes())?;
    save(lo, hi, &mut hw)?;
    if hw.n != 16 + n {
        return Err(io::Error::other(format!(
            "a KV image of {} bytes, {} expected",
            hw.n - 16,
            n
        )));
    }
    let c = hw.h.finish();
    w.write_all(&c.to_le_bytes())
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

impl Session {
    pub fn new(path: PathBuf, model: Value) -> Session {
        Session {
            path,
            model,
            saved: None,
        }
    }

    pub fn model(&self) -> &Value {
        &self.model
    }

    /// Save `state` (JSON; this adds `kv`) with the cache's positions
    /// 0..n holding `tokens` (n = its length), their image written by
    /// `save(lo, hi, out)`, under `ident` (None: tokens only, `kv` 0).
    /// Appends the new positions if the file holds a prefix of them, else
    /// rewrites it (through a temporary file renamed over it).
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
        let cur_len = std::fs::metadata(&self.path).map(|m| m.len()).ok();
        let append = self.saved.as_ref().filter(|s| {
            s.kv <= n
                && tokens[..s.kv] == s.tokens[..s.kv]
                && cur_len == Some(s.len)
                && s.stale + s.last_state <= MAX_STALE
        });
        let mut info = SaveInfo::default();
        let (from, stale, before) = match append {
            Some(s) => {
                let f = std::fs::OpenOptions::new().append(true).open(&self.path)?;
                let mut w = BufWriter::with_capacity(1 << 20, f);
                if n > s.kv {
                    kv_record(&mut w, s.kv, n, row, save)?;
                }
                record(&mut w, KIND_STATE, &st)?;
                w.into_inner().map_err(|e| e.into_error())?.sync_data()?;
                (s.kv, s.stale + s.last_state, s.len)
            }
            None => {
                info.rewrote = true;
                let tmp = self.path.with_extension("axkv-tmp");
                let f = File::create(&tmp)?;
                let mut w = BufWriter::with_capacity(1 << 20, f);
                let hdr =
                    serde_json::to_vec(&json!({ "identity": ident })).map_err(io::Error::other)?;
                w.write_all(MAGIC)?;
                w.write_all(&VERSION.to_le_bytes())?;
                w.write_all(&(hdr.len() as u32).to_le_bytes())?;
                w.write_all(&hdr)?;
                if n > 0 {
                    kv_record(&mut w, 0, n, row, save)?;
                }
                record(&mut w, KIND_STATE, &st)?;
                w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
                std::fs::rename(&tmp, &self.path)?;
                (0, 0, 0)
            }
        };
        let len = std::fs::metadata(&self.path)?.len();
        info.kv_written = n - from;
        info.bytes = len - before;
        self.saved = Some(Saved {
            kv: n,
            tokens: tokens[..n].to_vec(),
            len,
            stale,
            last_state: FRAME + st.len() as u64,
        });
        Ok(info)
    }

    /// The file's records, checked against `ident` (None: the backend has
    /// no image). Ok(None) if there is no file. Err only if it isn't a
    /// session file or holds no conversation.
    pub fn read(&self, ident: Option<&Value>) -> io::Result<Option<Loaded>> {
        let f = match File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let len = f.metadata()?.len();
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
        let hl = u32::from_le_bytes(head[8..12].try_into().expect("4")) as usize;
        let mut hdr = vec![0u8; hl];
        r.read_exact(&mut hdr)?;
        let hdr: Value = serde_json::from_slice(&hdr).map_err(|e| bad(format!("header: {e}")))?;
        let file_ident = &hdr["identity"];
        let row = file_ident["row_bytes"].as_u64().unwrap_or(0);
        let mut pos = 12 + hl as u64;
        let mut recs = Vec::new();
        let mut state: Option<(Value, u64)> = None;
        let mut stale = 0;
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
                KIND_KV if n >= 16 => {
                    let mut lohi = [0u8; 16];
                    r.read_exact(&mut lohi)?;
                    let lo = u64::from_le_bytes(lohi[..8].try_into().expect("8"));
                    let hi = u64::from_le_bytes(lohi[8..].try_into().expect("8"));
                    if hi < lo || (hi - lo).checked_mul(row) != Some(n - 16) {
                        break;
                    }
                    r.seek_relative((n - 16) as i64)?;
                    let mut c = [0u8; 8];
                    r.read_exact(&mut c)?;
                    recs.push((lo as usize, hi as usize, at, n, u64::from_le_bytes(c)));
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
                }
                _ => break,
            }
            pos = at + n + 8;
        }
        let Some((state, last_state)) = state else {
            return Err(bad("no conversation in the session file"));
        };
        let kv = state["kv"].as_u64().unwrap_or(0) as usize;
        // The records tiling 0..kv, in order (later ones belong to a save
        // whose STATE didn't make it).
        let mut end = 0;
        let mut tiling = Vec::new();
        for &rc in &recs {
            if end == kv {
                break;
            }
            if rc.0 != end {
                break;
            }
            end = rc.1;
            tiling.push(rc);
        }
        let usable = if kv == 0 || file_ident.is_null() {
            Err("the file holds no cache".into())
        } else if Some(file_ident) != ident {
            Err(match ident {
                None => "this backend has no cache image".into(),
                Some(i) => format!("it was made under {}", diff(file_ident, i)),
            })
        } else if end != kv {
            Err("its cache records are incomplete".into())
        } else {
            Ok(kv)
        };
        Ok(Some(Loaded {
            state,
            usable,
            recs: tiling,
            len,
            stale,
            last_state,
        }))
    }

    /// Load `l`'s positions 0..kv (`usable`) through `load(lo, hi, src)`,
    /// checking each record's hash as it streams. Err(why) if one fails (the
    /// rows may then hold anything: prefill them again). On success the
    /// file is the session's (`save` may append to it).
    pub fn load(&mut self, l: &Loaded, tokens: &[u32], load: &mut LoadFn) -> io::Result<()> {
        let kv = *l.usable.as_ref().map_err(|e| io::Error::other(e.clone()))?;
        let mut f = BufReader::with_capacity(1 << 20, File::open(&self.path)?);
        for &(lo, hi, at, n, check) in &l.recs {
            f.seek(SeekFrom::Start(at))?;
            let mut take = (&mut f).take(n);
            let mut hr = HashR {
                r: &mut take,
                h: Hasher::default(),
            };
            let mut lohi = [0u8; 16];
            hr.read_exact(&mut lohi)?;
            load(lo, hi, &mut hr)?;
            if hr.h.finish() != check {
                return Err(bad(format!("cache record {lo}..{hi} fails its check")));
            }
        }
        self.saved = Some(Saved {
            kv,
            tokens: tokens[..kv].to_vec(),
            len: l.len,
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

/// The identity fields that differ, for a message: "key file vs ours".
fn diff(file: &Value, ours: &Value) -> String {
    let (Value::Object(a), Value::Object(b)) = (file, ours) else {
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

    /// A fake cache: position p's image is `row` bytes of (p, byte index).
    fn fake_save(row: usize) -> impl FnMut(usize, usize, &mut dyn Write) -> io::Result<()> {
        move |lo, hi, w| {
            for p in lo..hi {
                let b: Vec<u8> = (0..row).map(|i| (p * 31 + i) as u8).collect();
                w.write_all(&b)?;
            }
            Ok(())
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("axon-kvsave-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(tag);
        let _ = std::fs::remove_file(&p);
        p
    }

    /// Saves append while the tokens extend, rewrite otherwise; reads see
    /// the last state; a torn tail or a corrupt byte never loads wrong
    /// bytes; another identity loads no bytes.
    #[test]
    fn file_appends_rewrites_and_checks() {
        let row = 40;
        let ident = json!({"model": "x", "row_bytes": row});
        let path = scratch("f.axkv");
        let mut s = Session::new(path.clone(), json!({}));
        let toks: Vec<u32> = (0..300).collect();
        let st = |n: usize| json!({ "n": n });
        let i1 = s
            .save(Some(&ident), &toks[..100], &mut fake_save(row), &st(1))
            .unwrap();
        assert!(i1.rewrote && i1.kv_written == 100);
        let i2 = s
            .save(Some(&ident), &toks[..250], &mut fake_save(row), &st(2))
            .unwrap();
        assert!(!i2.rewrote && i2.kv_written == 150, "{i2:?}");
        let i3 = s
            .save(Some(&ident), &toks[..250], &mut fake_save(row), &st(3))
            .unwrap();
        assert!(!i3.rewrote && i3.kv_written == 0);
        let full_len = std::fs::metadata(&path).unwrap().len();

        let mut r = Session::new(path.clone(), json!({}));
        let l = r.read(Some(&ident)).unwrap().unwrap();
        assert_eq!(l.state["n"], 3);
        assert_eq!(l.usable, Ok(250));
        let mut got = Vec::new();
        r.load(&l, &toks, &mut |lo, hi, src| {
            let mut b = vec![0u8; (hi - lo) * row];
            src.read_exact(&mut b)?;
            got.push((lo, hi, b));
            Ok(())
        })
        .unwrap();
        let mut want = Vec::new();
        fake_save(row)(0, 250, &mut want).unwrap();
        assert_eq!(
            got.iter().flat_map(|g| g.2.clone()).collect::<Vec<_>>(),
            want
        );
        // The reader's session appends.
        let i4 = r
            .save(Some(&ident), &toks[..260], &mut fake_save(row), &st(4))
            .unwrap();
        assert!(!i4.rewrote, "{i4:?}");
        // Other tokens: rewritten.
        let mut other = toks.clone();
        other[5] = 999;
        let i5 = r
            .save(Some(&ident), &other[..120], &mut fake_save(row), &st(5))
            .unwrap();
        assert!(i5.rewrote);
        assert_eq!(r.read(Some(&ident)).unwrap().unwrap().usable, Ok(120));

        // Another identity: tokens only.
        let o = json!({"model": "y", "row_bytes": row});
        let l = r.read(Some(&o)).unwrap().unwrap();
        assert!(
            l.usable.as_ref().unwrap_err().contains("model"),
            "{:?}",
            l.usable
        );
        assert!(r.read(None).unwrap().unwrap().usable.is_err());

        // Torn tail: the last save is lost, the one before stands.
        let mut s = Session::new(path.clone(), json!({}));
        s.save(Some(&ident), &toks[..100], &mut fake_save(row), &st(1))
            .unwrap();
        s.save(Some(&ident), &toks[..250], &mut fake_save(row), &st(2))
            .unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        for cut in [1, 9, 30] {
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.set_len(len - cut).unwrap();
            let l = Session::new(path.clone(), json!({}))
                .read(Some(&ident))
                .unwrap()
                .unwrap();
            assert_eq!(l.state["n"], 1, "cut {cut}");
            assert_eq!(l.usable, Ok(100), "cut {cut}");
        }
        // A save after the torn tail rewrites (the length isn't ours).
        assert!(
            s.save(Some(&ident), &toks[..260], &mut fake_save(row), &st(3))
                .unwrap()
                .rewrote
        );

        // A flipped KV byte: the load fails (the caller prefills again).
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes.len() / 2;
        bytes[at] ^= 0x40;
        std::fs::write(&path, &bytes).unwrap();
        let mut r = Session::new(path.clone(), json!({}));
        let l = r.read(Some(&ident)).unwrap().unwrap();
        assert_eq!(l.usable, Ok(260));
        let e = r.load(&l, &toks, &mut |lo, hi, src| {
            let mut b = vec![0u8; (hi - lo) * row];
            src.read_exact(&mut b)
        });
        assert!(e.is_err());
        assert!(full_len > 0);
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
