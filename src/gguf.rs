//! GGUF reader: header, metadata and tensor table, parsed from the in-memory file.
//!
//! The file is untrusted input. Every length and count is checked against the bytes
//! that are actually left before anything is allocated, and every tensor's data range
//! is checked to lie inside the file, so a truncated or malformed file is an
//! `InvalidData` error, never a panic or a giant allocation.

use std::fs::File;
use std::io::{self, Read};

const GGUF_MAGIC: u32 = 0x46554747;
/// GGML_MAX_DIMS.
const MAX_DIMS: u32 = 4;
/// `general.alignment` when the file doesn't set it.
const DEFAULT_ALIGNMENT: u64 = 32;
/// Arrays of arrays are legal GGUF; nothing real nests deeper than this.
const MAX_ARRAY_DEPTH: u32 = 4;

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// A cursor over the file bytes that refuses to read past the end.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn left(&self) -> u64 {
        (self.buf.len() - self.pos) as u64
    }

    fn take(&mut self, n: u64, what: &str) -> io::Result<&'a [u8]> {
        if n > self.left() {
            return Err(bad(format!(
                "truncated or corrupt file: {what} needs {n} bytes at offset {}, only {} left",
                self.pos,
                self.left()
            )));
        }
        let s = &self.buf[self.pos..self.pos + n as usize];
        self.pos += n as usize;
        Ok(s)
    }

    fn array<const N: usize>(&mut self, what: &str) -> io::Result<[u8; N]> {
        Ok(self.take(N as u64, what)?.try_into().unwrap())
    }

    fn u8(&mut self, what: &str) -> io::Result<u8> {
        Ok(self.array::<1>(what)?[0])
    }
    fn u32(&mut self, what: &str) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }
    fn u64(&mut self, what: &str) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.array(what)?))
    }

    fn string(&mut self, what: &str) -> io::Result<String> {
        let len = self.u64(what)?;
        let bytes = self.take(len, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| bad(format!("{what} is not valid UTF-8")))
    }

    /// `n` items of at least `min_size` bytes each must fit in what's left.
    fn check_count(&self, n: u64, min_size: u64, what: &str) -> io::Result<()> {
        if n.checked_mul(min_size).is_none_or(|b| b > self.left()) {
            return Err(bad(format!(
                "corrupt file: {what} count {n} can't fit in the {} bytes left",
                self.left()
            )));
        }
        Ok(())
    }

    fn val(&mut self, t: u32, depth: u32) -> io::Result<Val> {
        const W: &str = "metadata value";
        Ok(match t {
            0 => Val::U8(self.u8(W)?),
            1 => Val::I8(self.u8(W)? as i8),
            2 => Val::U16(u16::from_le_bytes(self.array(W)?)),
            3 => Val::I16(i16::from_le_bytes(self.array(W)?)),
            4 => Val::U32(self.u32(W)?),
            5 => Val::I32(i32::from_le_bytes(self.array(W)?)),
            6 => Val::F32(f32::from_le_bytes(self.array(W)?)),
            7 => Val::Bool(self.u8(W)? != 0),
            8 => Val::Str(self.string("metadata string")?),
            9 => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(bad("corrupt file: metadata arrays nested too deep"));
                }
                let et = self.u32(W)?;
                let n = self.u64(W)?;
                self.check_count(n, val_min_size(et)?, "metadata array")?;
                let mut a = Vec::with_capacity((n as usize).min(4096));
                for _ in 0..n {
                    a.push(self.val(et, depth + 1)?);
                }
                Val::Arr(a)
            }
            10 => Val::U64(self.u64(W)?),
            11 => Val::I64(i64::from_le_bytes(self.array(W)?)),
            12 => Val::F64(f64::from_le_bytes(self.array(W)?)),
            _ => {
                return Err(bad(format!(
                    "corrupt file: unknown metadata value type {t}"
                )));
            }
        })
    }
}

/// Smallest encoding of a metadata value of type `t`, in bytes.
fn val_min_size(t: u32) -> io::Result<u64> {
    Ok(match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        8 | 10..=12 => 8,
        9 => 12, // element type + count
        _ => {
            return Err(bad(format!(
                "corrupt file: unknown metadata value type {t}"
            )));
        }
    })
}

#[derive(Debug, Clone)]
pub enum Val {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Val>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Val::U8(v) => write!(f, "{v}"),
            Val::I8(v) => write!(f, "{v}"),
            Val::U16(v) => write!(f, "{v}"),
            Val::I16(v) => write!(f, "{v}"),
            Val::U32(v) => write!(f, "{v}"),
            Val::I32(v) => write!(f, "{v}"),
            Val::F32(v) => write!(f, "{v:.6}"),
            Val::Bool(v) => write!(f, "{v}"),
            Val::Str(v) => write!(f, "{v}"),
            Val::Arr(v) => write!(f, "[{} items]", v.len()),
            Val::U64(v) => write!(f, "{v}"),
            Val::I64(v) => write!(f, "{v}"),
            Val::F64(v) => write!(f, "{v}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QType {
    pub id: u32,
    pub name: &'static str,
    pub block_size: usize,
    pub block_bytes: usize,
    pub bits: f32,
}

const QTYPES: &[QType] = &[
    QType {
        id: 0,
        name: "F32",
        block_size: 1,
        block_bytes: 4,
        bits: 32.0,
    },
    QType {
        id: 1,
        name: "F16",
        block_size: 1,
        block_bytes: 2,
        bits: 16.0,
    },
    QType {
        id: 2,
        name: "Q4_0",
        block_size: 32,
        block_bytes: 18,
        bits: 4.5,
    },
    QType {
        id: 3,
        name: "Q4_1",
        block_size: 32,
        block_bytes: 20,
        bits: 5.0,
    },
    QType {
        id: 6,
        name: "Q5_0",
        block_size: 32,
        block_bytes: 22,
        bits: 5.5,
    },
    QType {
        id: 7,
        name: "Q5_1",
        block_size: 32,
        block_bytes: 24,
        bits: 6.0,
    },
    QType {
        id: 8,
        name: "Q8_0",
        block_size: 32,
        block_bytes: 34,
        bits: 8.5,
    },
    QType {
        id: 9,
        name: "Q8_1",
        block_size: 32,
        block_bytes: 36,
        bits: 9.0,
    },
    QType {
        id: 10,
        name: "Q2_K",
        block_size: 256,
        block_bytes: 84,
        bits: 2.6,
    },
    QType {
        id: 11,
        name: "Q3_K",
        block_size: 256,
        block_bytes: 110,
        bits: 3.4,
    },
    QType {
        id: 12,
        name: "Q4_K",
        block_size: 256,
        block_bytes: 144,
        bits: 4.5,
    },
    QType {
        id: 13,
        name: "Q5_K",
        block_size: 256,
        block_bytes: 176,
        bits: 5.5,
    },
    QType {
        id: 14,
        name: "Q6_K",
        block_size: 256,
        block_bytes: 210,
        bits: 6.6,
    },
    QType {
        id: 15,
        name: "Q8_K",
        block_size: 256,
        block_bytes: 292,
        bits: 9.1,
    },
    QType {
        id: 30,
        name: "BF16",
        block_size: 1,
        block_bytes: 2,
        bits: 16.0,
    },
];

pub fn qtype(id: u32) -> QType {
    QTYPES
        .iter()
        .find(|q| q.id == id)
        .copied()
        .unwrap_or(QType {
            id,
            name: "??",
            block_size: 1,
            block_bytes: 1,
            bits: 0.0,
        })
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub qtype: QType,
    pub offset: u64,
    pub n_elements: u64,
    pub n_bytes: u64,
    #[allow(dead_code)] // parsed from the name; nothing reads it yet
    pub layer: i32, // -1 = global, 0..N = layer index
    #[allow(dead_code)]
    pub kind: String, // "attn_q", "ffn_up", "token_embd", etc.
}

pub struct Model {
    pub arch: String,
    pub metadata: Vec<(String, Val)>,
    pub tensors: Vec<Tensor>,
    pub data: memmap2::Mmap,
    pub data_offset: u64,
    /// `data` maps the file (true) or is the anonymous copy (false).
    file_backed: bool,
    /// Byte ranges of the tensors handed back by `release`, sorted. On the
    /// anonymous copy (zeroed there) `tensor_data` refuses tensors that
    /// overlap one.
    released: std::sync::Mutex<Vec<std::ops::Range<usize>>>,
    any_released: std::sync::atomic::AtomicBool,
}

/// `release` only drops whole aligned blocks of this size inside released
/// tensors (and the padding between them): a multiple of every page size
/// Linux uses (4/16/64 KB), so other tensors' bytes are never touched. On
/// the anonymous copy it's 2 MB instead: that copy asks for huge pages, and
/// dropping part of one splits it, so the tensors still read from the rest
/// of it would go back to 4 KB pages (dTLB misses on every decode token).
const RELEASE_ALIGN: usize = 64 << 10;
const RELEASE_ALIGN_HUGE: usize = 2 << 20;

impl Model {
    /// In bounds for every tensor of a loaded model: `parse` checks the ranges.
    fn tensor_range(&self, t: &Tensor) -> std::ops::Range<usize> {
        let s = (self.data_offset + t.offset) as usize;
        s..s + t.n_bytes as usize
    }

    /// The tensor's bytes. Panics if they were `release`d from the anonymous
    /// copy (they'd read as zeros).
    pub fn tensor_data(&self, t: &Tensor) -> &[u8] {
        let r = self.tensor_range(t);
        if !self.file_backed && self.any_released.load(std::sync::atomic::Ordering::Acquire) {
            let rel = self.released.lock().unwrap();
            assert!(
                !rel.iter().any(|x| x.start < r.end && r.start < x.end),
                "tensor {} was released after repacking; load the model again to read it",
                t.name
            );
        }
        &self.data[r]
    }

    /// Whether `data` maps the file (vs the anonymous copy, see `map_weights`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn file_backed(&self) -> bool {
        self.file_backed
    }

    /// Hand back the memory of a tensor that was copied elsewhere and won't be
    /// read from here again (MADV_DONTNEED, whole aligned blocks only). A
    /// block may span several released tensors and the padding between them
    /// (bytes no tensor owns), so small tensors next to each other go too;
    /// it never holds a byte of a tensor that wasn't released. On a file
    /// mapping the pages just leave this mapping and a later read faults them
    /// back in from the file. On the anonymous copy they come back as zeros,
    /// so `tensor_data` panics on a released tensor from then on.
    pub fn release(&self, t: &Tensor) {
        let r = self.tensor_range(t);
        if r.is_empty() {
            return;
        }
        let mut rel = self.released.lock().unwrap();
        let i = rel.partition_point(|x| x.start < r.start);
        if rel.get(i) == Some(&r) {
            return;
        }
        rel.insert(i, r);
        if !self.file_backed {
            self.any_released
                .store(true, std::sync::atomic::Ordering::Release);
        }
        // The run of released tensors around this one, joined across gaps
        // that hold no tensor bytes.
        let (mut lo, mut hi) = (i, i);
        while lo > 0 && self.unowned(rel[lo - 1].end, rel[lo].start) {
            lo -= 1;
        }
        while hi + 1 < rel.len() && self.unowned(rel[hi].end, rel[hi + 1].start) {
            hi += 1;
        }
        let base = self.data.as_ptr() as usize;
        let align = if self.file_backed {
            RELEASE_ALIGN
        } else {
            RELEASE_ALIGN_HUGE
        };
        let start = (base + rel[lo].start).next_multiple_of(align) - base;
        let end = (base + rel[hi].end) / align * align - base;
        if start >= end {
            return;
        }
        // SAFETY: [start, end) is inside `data` and inside released tensors
        // or bytes no tensor owns; nothing reads those again (on the
        // anonymous copy `tensor_data` enforces that; the slices handed out
        // before were all consumed by their owners before `release`). Blocks
        // dropped by an earlier call are simply dropped again. On a file
        // mapping the bytes don't change at all.
        let _ = unsafe {
            self.data
                .unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, start, end - start)
        };
    }

    /// Whether `t` was handed back by `release`.
    #[cfg(test)]
    pub fn is_released(&self, t: &Tensor) -> bool {
        let r = self.tensor_range(t);
        self.released.lock().unwrap().contains(&r)
    }

    /// Whether bytes a..b of `data` (between two released tensors) belong
    /// to no tensor.
    fn unowned(&self, a: usize, b: usize) -> bool {
        a >= b
            || !self.tensors.iter().any(|t| {
                let r = self.tensor_range(t);
                !r.is_empty() && r.start < b && a < r.end
            })
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| if let Val::U32(n) = v { Some(*n) } else { None })
    }
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| {
                if let Val::Str(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            })
    }
    pub fn n_layers(&self) -> u32 {
        self.meta_u32(&format!("{}.block_count", self.arch))
            .unwrap_or(0)
    }
    pub fn embed_dim(&self) -> u32 {
        self.meta_u32(&format!("{}.embedding_length", self.arch))
            .unwrap_or(0)
    }
    pub fn n_heads(&self) -> u32 {
        self.meta_u32(&format!("{}.attention.head_count", self.arch))
            .unwrap_or(0)
    }
    pub fn n_kv_heads(&self) -> u32 {
        self.meta_u32(&format!("{}.attention.head_count_kv", self.arch))
            .unwrap_or(0)
    }
    pub fn ctx_len(&self) -> u32 {
        self.meta_u32(&format!("{}.context_length", self.arch))
            .unwrap_or(0)
    }
    pub fn ffn_dim(&self) -> u32 {
        self.meta_u32(&format!("{}.feed_forward_length", self.arch))
            .unwrap_or(0)
    }
    pub fn total_params(&self) -> u64 {
        self.tensors.iter().map(|t| t.n_elements).sum()
    }
    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.n_bytes).sum()
    }
}

/// The whole file, in memory. By default it's copied into anonymous memory with
/// transparent huge pages requested: decode walks the whole model per token, and 2 MB
/// pages cut the TLB misses of 4 KB file pages (llama.cpp's `--no-mmap` effect).
///
/// The copy is not shared between processes and can't be dropped under memory
/// pressure the way page cache can, so it's skipped (file mmap instead) when the file
/// is over 60% of `MemAvailable`. `AXON_MMAP=1` always maps the file, `AXON_MMAP=0`
/// always copies.
///
/// Assumes Linux >= 6.7 for 2 MB-aligned anonymous mappings; on older kernels the
/// first and last partial 2 MB of the copy use 4 KB pages, which costs nothing much.
/// Returns the mapping and whether it maps the file. `copy` overrides the
/// choice (Some(true): copy, Some(false): map the file).
fn map_weights(file: &File, copy: Option<bool>) -> io::Result<(memmap2::Mmap, bool)> {
    let len = file.metadata()?.len() as usize;
    let env = std::env::var("AXON_MMAP");
    let copy = match (copy, env.as_deref()) {
        (Some(c), _) => c,
        (None, Ok("0")) => true,
        (None, Ok(_)) => false,
        (None, Err(_)) => match mem_available() {
            Some(avail) if len as u64 > avail / 10 * 6 => {
                eprintln!(
                    "note: model is {} MB, {} MB available; mapping the file instead of \
                     copying it (AXON_MMAP=0 forces the copy)",
                    len >> 20,
                    avail >> 20
                );
                false
            }
            _ => true,
        },
    };
    if !copy {
        // SAFETY: the file is opened read-only and not modified while mapped.
        return Ok((unsafe { memmap2::Mmap::map(file)? }, true));
    }
    let mut anon = memmap2::MmapMut::map_anon(len.max(1))?;
    #[cfg(target_os = "linux")]
    {
        // Advise before touching any page. On THP=madvise (most distro defaults) this is
        // what turns huge pages on; with defrag=defer+madvise it also opts this region
        // into direct compaction at fault time, so a fragmented box loads slower but
        // still gets huge pages up front. Failure only costs speed.
        let _ = anon.advise(memmap2::Advice::HugePage);
    }
    let mut f = file;
    f.read_exact(&mut anon[..len]).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            io::Error::new(io::ErrorKind::UnexpectedEof, "file shrank while loading")
        } else {
            e
        }
    })?;
    Ok((anon.make_read_only()?, false))
}

/// `MemAvailable` from /proc/meminfo, in bytes.
fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

pub fn load(path: &str) -> io::Result<Model> {
    load_with(path, None)
}

/// `load`, with the copy-vs-map choice forced (`copy`: see `map_weights`).
pub fn load_with(path: &str, copy: Option<bool>) -> io::Result<Model> {
    let file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    let (data, file_backed) = map_weights(&file, copy)?;
    let h = parse(&data[..len.min(data.len())])?;
    Ok(Model {
        arch: h.arch,
        metadata: h.metadata,
        tensors: h.tensors,
        data,
        data_offset: h.data_offset,
        file_backed,
        released: Default::default(),
        any_released: Default::default(),
    })
}

/// Everything in a GGUF file but the tensor data.
struct Header {
    arch: String,
    metadata: Vec<(String, Val)>,
    tensors: Vec<Tensor>,
    data_offset: u64,
}

fn parse(buf: &[u8]) -> io::Result<Header> {
    let mut r = Reader { buf, pos: 0 };

    let magic = r
        .u32("GGUF magic")
        .map_err(|_| bad("not a GGUF file (too short)"))?;
    if magic != GGUF_MAGIC {
        return Err(bad("not a GGUF file (bad magic)"));
    }
    let ver = r.u32("GGUF version")?;
    if ver != 2 && ver != 3 {
        return Err(bad(if matches!(ver.swap_bytes(), 2 | 3) {
            "big-endian GGUF files are not supported".to_string()
        } else {
            format!("unsupported GGUF version {ver} (axon reads versions 2 and 3)")
        }));
    }
    let n_tensors = r.u64("tensor count")?;
    let n_meta = r.u64("metadata count")?;
    // A key/value pair is at least a key length, a type and one byte of value; a
    // tensor info at least a name length, n_dims, a type and an offset.
    r.check_count(n_meta, 8 + 4 + 1, "metadata")?;
    r.check_count(n_tensors, 8 + 4 + 4 + 8, "tensor")?;

    let mut metadata = Vec::new();
    for _ in 0..n_meta {
        let key = r.string("metadata key")?;
        let vt = r.u32("metadata type")?;
        let val = r.val(vt, 0)?;
        metadata.push((key, val));
    }

    let arch = metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| {
            if let Val::Str(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })
        .unwrap_or_default();

    // The spec says u32; other unsigned types are accepted if the value fits.
    let alignment = match metadata.iter().find(|(k, _)| k == "general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some((_, v)) => {
            let a = match v {
                Val::U8(a) => *a as u64,
                Val::U16(a) => *a as u64,
                Val::U32(a) => *a as u64,
                Val::U64(a) if *a <= u32::MAX as u64 => *a,
                _ => return Err(bad(format!("general.alignment must be a u32, got {v}"))),
            };
            if !a.is_power_of_two() {
                return Err(bad(format!(
                    "general.alignment must be a power of two, got {a}"
                )));
            }
            a
        }
    };

    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for i in 0..n_tensors {
        let name = r
            .string("tensor name")
            .map_err(|e| bad(format!("tensor #{i}: {e}")))?;
        let nd = r.u32("tensor n_dims")?;
        // 0 dims is a scalar: one element (the empty product).
        if nd > MAX_DIMS {
            return Err(bad(format!(
                "tensor {name}: {nd} dimensions (at most {MAX_DIMS})"
            )));
        }
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(r.u64("tensor dims")?);
        }
        let qt = qtype(r.u32("tensor type")?);
        let offset = r.u64("tensor offset")?;
        let n_elements = dims
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d))
            .ok_or_else(|| bad(format!("tensor {name}: element count overflows ({dims:?})")))?;
        // Unknown types (qtype "??") have no known size; only their offset is checked.
        let n_bytes = if qt.name == "??" {
            0
        } else {
            if n_elements % qt.block_size as u64 != 0 {
                return Err(bad(format!(
                    "tensor {name}: {n_elements} elements is not a whole number of {} blocks",
                    qt.name
                )));
            }
            (n_elements / qt.block_size as u64)
                .checked_mul(qt.block_bytes as u64)
                .ok_or_else(|| bad(format!("tensor {name}: size overflows")))?
        };
        if offset % alignment != 0 {
            return Err(bad(format!(
                "tensor {name}: data offset {offset} is not aligned to {alignment}"
            )));
        }

        let parts: Vec<&str> = name.split('.').collect();
        let (layer, kind) = if parts[0] == "blk" {
            (
                parts
                    .get(1)
                    .and_then(|p| p.parse::<i32>().ok())
                    .unwrap_or(-1),
                parts.get(2..).unwrap_or_default().join("."),
            )
        } else {
            (-1, name.clone())
        };

        tensors.push(Tensor {
            name,
            dims,
            qtype: qt,
            offset,
            n_elements,
            n_bytes,
            layer,
            kind,
        });
    }

    let data_offset = (r.pos as u64).next_multiple_of(alignment);
    let file_len = buf.len() as u64;
    for t in &tensors {
        let end = data_offset
            .checked_add(t.offset)
            .and_then(|s| s.checked_add(t.n_bytes));
        if end.is_none_or(|e| e > file_len) {
            return Err(bad(format!(
                "tensor {}: data at {}+{} ({} bytes) runs past the end of the file \
                 ({file_len} bytes); the file is truncated or corrupt",
                t.name, data_offset, t.offset, t.n_bytes
            )));
        }
    }

    Ok(Header {
        arch,
        metadata,
        tensors,
        data_offset,
    })
}

/// A model built in memory: `meta` (U32, F32, Str and arrays of Str values) and
/// zero-filled tensors `(name, dims, type id)`, written as a GGUF and parsed.
#[cfg(test)]
pub fn build_for_test(meta: &[(&str, Val)], tensors: &[(&str, &[u64], u32)]) -> Model {
    fn str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }
    fn val(b: &mut Vec<u8>, v: &Val) {
        match v {
            Val::U32(x) => b.extend_from_slice(&x.to_le_bytes()),
            Val::F32(x) => b.extend_from_slice(&x.to_le_bytes()),
            Val::Str(s) => str(b, s),
            Val::Arr(a) => {
                b.extend_from_slice(&8u32.to_le_bytes());
                b.extend_from_slice(&(a.len() as u64).to_le_bytes());
                a.iter().for_each(|x| val(b, x));
            }
            _ => unimplemented!("build_for_test: {v:?}"),
        }
    }
    fn type_id(v: &Val) -> u32 {
        match v {
            Val::U32(_) => 4,
            Val::F32(_) => 6,
            Val::Str(_) => 8,
            _ => 9,
        }
    }
    let mut b = Vec::new();
    for x in [GGUF_MAGIC, 3] {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for x in [tensors.len() as u64, meta.len() as u64] {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for (k, v) in meta {
        str(&mut b, k);
        b.extend_from_slice(&type_id(v).to_le_bytes());
        val(&mut b, v);
    }
    let mut offset = 0u64;
    for (name, dims, ty) in tensors {
        str(&mut b, name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        dims.iter()
            .for_each(|d| b.extend_from_slice(&d.to_le_bytes()));
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        let q = qtype(*ty);
        let n: u64 = dims.iter().product();
        offset += (n / q.block_size as u64 * q.block_bytes as u64).next_multiple_of(32);
    }
    b.resize(b.len().next_multiple_of(32) + offset as usize, 0);
    let h = parse(&b).expect("build_for_test: parse");
    let mut data = memmap2::MmapMut::map_anon(b.len()).unwrap();
    data.copy_from_slice(&b);
    Model {
        arch: h.arch,
        metadata: h.metadata,
        tensors: h.tensors,
        data: data.make_read_only().unwrap(),
        data_offset: h.data_offset,
        file_backed: false,
        released: Default::default(),
        any_released: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A little-endian GGUF writer for building test files in memory.
    #[derive(Default)]
    struct W(Vec<u8>);

    impl W {
        fn u32(mut self, v: u32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn u64(mut self, v: u64) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn str(self, s: &str) -> Self {
            let mut w = self.u64(s.len() as u64);
            w.0.extend_from_slice(s.as_bytes());
            w
        }
        fn header(ver: u32, n_tensors: u64, n_meta: u64) -> Self {
            W::default()
                .u32(GGUF_MAGIC)
                .u32(ver)
                .u64(n_tensors)
                .u64(n_meta)
        }
        /// One F32 tensor info: name, 1 dim of `n`, type 0, `offset`.
        fn tensor(self, name: &str, n: u64, offset: u64) -> Self {
            self.str(name).u32(1).u64(n).u32(0).u64(offset)
        }
        fn pad_to(mut self, align: usize, data: usize) -> Vec<u8> {
            let n = self.0.len().next_multiple_of(align) + data;
            self.0.resize(n, 0);
            self.0
        }
    }

    fn err(buf: &[u8]) -> String {
        match parse(buf) {
            Ok(_) => panic!("parsed a malformed file"),
            Err(e) => {
                assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{e}");
                e.to_string()
            }
        }
    }

    fn valid() -> W {
        W::header(3, 2, 1)
            .str("general.architecture")
            .u32(8)
            .str("qwen2")
            .tensor("blk.0.attn_norm.weight", 8, 0)
            .tensor("output_norm.weight", 8, 32)
    }

    #[test]
    fn parses_a_valid_file() {
        let buf = valid().pad_to(32, 64);
        let h = parse(&buf).expect("valid file");
        assert_eq!(h.arch, "qwen2");
        assert_eq!(h.tensors.len(), 2);
        assert_eq!(h.data_offset % 32, 0);
        assert_eq!(
            (h.tensors[0].layer, h.tensors[0].kind.as_str()),
            (0, "attn_norm.weight")
        );
        assert_eq!(h.tensors[1].n_bytes, 32);
    }

    #[test]
    fn truncated_files() {
        let buf = valid().pad_to(32, 64);
        assert!(err(&buf[..2]).contains("not a GGUF"));
        assert!(err(&[]).contains("not a GGUF"));
        // Every cut inside the header or the data is an error, never a panic.
        for n in 4..buf.len() {
            err(&buf[..n]);
        }
        assert!(err(&buf[..buf.len() - 1]).contains("past the end"));
    }

    #[test]
    fn bad_magic_and_version() {
        let mut buf = valid().pad_to(32, 64);
        buf[0] = b'X';
        assert!(err(&buf).contains("bad magic"));
        for v in [0u32, 1, 4, u32::MAX] {
            let buf = W::header(v, 0, 0).0;
            assert!(err(&buf).contains(&format!("version {v}")));
        }
        assert!(err(&W::header(3u32.swap_bytes(), 0, 0).0).contains("big-endian"));
    }

    #[test]
    fn huge_lengths_and_counts() {
        // A key claiming 2^60 bytes must not be allocated.
        let buf = W::header(3, 0, 1).u64(1 << 60).pad_to(32, 64);
        assert!(err(&buf).contains("metadata key"));
        let buf = W::header(3, 0, 1).str("k").u32(8).u64(u64::MAX).0;
        assert!(err(&buf).contains("metadata string"));
        let buf = W::header(3, 0, 1).str("k").u32(9).u32(4).u64(1 << 62).0;
        assert!(err(&buf).contains("metadata array count"));
        assert!(err(&W::header(3, u64::MAX, 0).0).contains("tensor count"));
        assert!(err(&W::header(3, 0, 1 << 40).0).contains("metadata count"));
        // Arrays of arrays of arrays...
        let mut w = W::header(3, 0, 1).str("k").u32(9);
        for _ in 0..64 {
            w = w.u32(9).u64(1);
        }
        assert!(err(&w.pad_to(8, 64)).contains("nested"));
    }

    #[test]
    fn bad_tensors() {
        // Data range past the end of the file.
        let buf = W::header(3, 1, 0).tensor("t", 8, 1 << 20).pad_to(32, 64);
        assert!(err(&buf).contains("past the end"));
        let buf = W::header(3, 1, 0)
            .tensor("t", 8, u64::MAX - 31)
            .pad_to(32, 64);
        assert!(err(&buf).contains("past the end"));
        // Element count overflow.
        let buf = W::header(3, 1, 0)
            .str("t")
            .u32(2)
            .u64(1 << 40)
            .u64(1 << 40)
            .u32(0)
            .u64(0)
            .pad_to(32, 64);
        assert!(err(&buf).contains("overflows"));
        // Too many dims.
        let buf = W::header(3, 1, 0).str("t").u32(5).pad_to(32, 64);
        assert!(err(&buf).contains("dimensions"));
        // Misaligned offset.
        let buf = W::header(3, 1, 0).tensor("t", 1, 4).pad_to(32, 64);
        assert!(err(&buf).contains("aligned"));
        // Non-UTF-8 name.
        let mut w = W::header(3, 1, 0).u64(2);
        w.0.extend_from_slice(&[0xff, 0xfe]);
        let buf = w.u32(1).u64(1).u32(0).u64(0).pad_to(32, 64);
        assert!(err(&buf).contains("UTF-8"));
        // Partial quant block: 31 elements of Q8_0 (32 per block).
        let buf = W::header(3, 1, 0)
            .str("t")
            .u32(1)
            .u64(31)
            .u32(8)
            .u64(0)
            .pad_to(32, 64);
        assert!(err(&buf).contains("whole number"));
        // 0 dims: a scalar, one element.
        let buf = W::header(3, 1, 0)
            .str("s")
            .u32(0)
            .u32(0)
            .u64(0)
            .pad_to(32, 32);
        let t = &parse(&buf).unwrap().tensors[0];
        assert_eq!((t.n_elements, t.n_bytes), (1, 4));
        // "blk" alone is a name, not a crash.
        let buf = W::header(3, 1, 0).tensor("blk", 1, 0).pad_to(32, 32);
        assert_eq!(parse(&buf).unwrap().tensors[0].layer, -1);
    }

    #[test]
    fn honours_general_alignment() {
        let meta = |w: W, a: u32| w.str("general.alignment").u32(4).u32(a);
        let buf = meta(W::header(3, 1, 1), 64)
            .tensor("t", 4, 64)
            .pad_to(64, 128);
        let h = parse(&buf).expect("alignment 64");
        // The 90-byte header rounds up to 128; the default 32 would give 96.
        assert_eq!(h.data_offset, 128);
        let buf = meta(W::header(3, 1, 1), 64)
            .tensor("t", 4, 32)
            .pad_to(64, 128);
        assert!(err(&buf).contains("aligned to 64"));
        let buf = meta(W::header(3, 0, 1), 48).pad_to(8, 0);
        assert!(err(&buf).contains("power of two"));
        // Wider unsigned types are fine if the value fits; other types aren't.
        let u64_meta = |a: u64| {
            let mut w = W::header(3, 0, 1).str("general.alignment").u32(10);
            w = w.u64(a);
            w.pad_to(8, 0)
        };
        assert_eq!(parse(&u64_meta(32)).unwrap().data_offset % 32, 0);
        assert!(err(&u64_meta(1 << 40)).contains("must be a u32"));
        let buf = W::header(3, 0, 1)
            .str("general.alignment")
            .u32(8)
            .str("32")
            .pad_to(8, 0);
        assert!(err(&buf).contains("must be a u32"));
    }
}
