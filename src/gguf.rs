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
/// String arrays at least this long are parsed on several threads
/// (`Reader::strings`).
const PAR_STRINGS: usize = 16384;
/// Arrays of arrays are legal GGUF; nothing real nests deeper than this.
const MAX_ARRAY_DEPTH: u32 = 4;
/// Most metadata array elements in a file, all arrays together. Each one
/// is a `Val` in memory (32 bytes) however small its encoding (a u8 is 1
/// byte), so the file-size check alone let a 150 MB header ask for 4.8 GB.
/// Real vocabularies have up to ~262k tokens, in 3-4 arrays; this is 268 MB
/// of `Val`s at most.
const MAX_META_ELEMS: u64 = 1 << 23;

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// A cursor over the file bytes that refuses to read past the end.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    /// Metadata array elements still allowed (`MAX_META_ELEMS`).
    elems_left: u64,
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

    /// An array of `n` strings (the vocabulary, the merges: ~150k each):
    /// their bounds found in one pass, then the Strings made on up to 4
    /// threads. The same values as reading them one by one; with both a
    /// truncated and an invalid UTF-8 string, the truncation is reported.
    fn strings(&mut self, n: usize) -> io::Result<Val> {
        const W: &str = "metadata string";
        // `n` was checked against the bytes left, not against memory: a
        // corrupt count only grows these as strings are really read.
        let mut spans = Vec::with_capacity(n.min(1 << 18));
        for _ in 0..n {
            let len = self.u64(W)?;
            let at = self.pos;
            self.take(len, W)?;
            spans.push((at, len as usize));
        }
        let buf = self.buf;
        let make = |part: &[(usize, usize)]| -> io::Result<Vec<Val>> {
            (part.iter())
                .map(|&(at, len)| {
                    let b = &buf[at..at + len];
                    match std::str::from_utf8(b) {
                        Ok(s) => Ok(Val::Str(s.to_owned())),
                        Err(_) => Err(bad(format!("{W} is not valid UTF-8"))),
                    }
                })
                .collect()
        };
        let threads = std::thread::available_parallelism()
            .map_or(1, |t| t.get())
            .clamp(1, 4);
        let per = n.div_ceil(threads).max(PAR_STRINGS / 2);
        let parts: Vec<io::Result<Vec<Val>>> = std::thread::scope(|sc| {
            // A part whose thread didn't start is made here, in order.
            let hs: Vec<_> = (spans.chunks(per))
                .map(|c| {
                    (
                        c,
                        std::thread::Builder::new().spawn_scoped(sc, move || make(c)),
                    )
                })
                .collect();
            (hs.into_iter())
                .map(|(c, h)| match h {
                    Ok(h) => h.join().expect("string parse thread"),
                    Err(_) => make(c),
                })
                .collect()
        });
        let mut a = Vec::with_capacity(spans.len());
        for p in parts {
            a.extend(p?);
        }
        Ok(Val::Arr(a))
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
                if n > self.elems_left {
                    return Err(bad(format!(
                        "corrupt file: metadata arrays hold more than {MAX_META_ELEMS} values"
                    )));
                }
                self.elems_left -= n;
                if et == 8 && n >= PAR_STRINGS as u64 {
                    return self.strings(n as usize);
                }
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

impl Val {
    /// The value of a numeric (32/64-bit integer or float) entry.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Val::U32(n) => Some(*n as f64),
            Val::I32(n) => Some(*n as f64),
            Val::U64(n) => Some(*n as f64),
            Val::I64(n) => Some(*n as f64),
            Val::F32(n) => Some(*n as f64),
            Val::F64(n) => Some(*n),
            _ => None,
        }
    }
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
}

const QTYPES: &[QType] = &[
    QType {
        id: 0,
        name: "F32",
        block_size: 1,
        block_bytes: 4,
    },
    QType {
        id: 1,
        name: "F16",
        block_size: 1,
        block_bytes: 2,
    },
    QType {
        id: 2,
        name: "Q4_0",
        block_size: 32,
        block_bytes: 18,
    },
    QType {
        id: 3,
        name: "Q4_1",
        block_size: 32,
        block_bytes: 20,
    },
    QType {
        id: 6,
        name: "Q5_0",
        block_size: 32,
        block_bytes: 22,
    },
    QType {
        id: 7,
        name: "Q5_1",
        block_size: 32,
        block_bytes: 24,
    },
    QType {
        id: 8,
        name: "Q8_0",
        block_size: 32,
        block_bytes: 34,
    },
    QType {
        id: 9,
        name: "Q8_1",
        block_size: 32,
        block_bytes: 36,
    },
    QType {
        id: 10,
        name: "Q2_K",
        block_size: 256,
        block_bytes: 84,
    },
    QType {
        id: 11,
        name: "Q3_K",
        block_size: 256,
        block_bytes: 110,
    },
    QType {
        id: 12,
        name: "Q4_K",
        block_size: 256,
        block_bytes: 144,
    },
    QType {
        id: 13,
        name: "Q5_K",
        block_size: 256,
        block_bytes: 176,
    },
    QType {
        id: 14,
        name: "Q6_K",
        block_size: 256,
        block_bytes: 210,
    },
    QType {
        id: 15,
        name: "Q8_K",
        block_size: 256,
        block_bytes: 292,
    },
    QType {
        id: 20,
        name: "IQ4_NL",
        block_size: 32,
        block_bytes: 18,
    },
    QType {
        id: 30,
        name: "BF16",
        block_size: 1,
        block_bytes: 2,
    },
    QType {
        id: 39,
        name: "MXFP4",
        block_size: 32,
        block_bytes: 17,
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
    pub layer: i32,   // -1 = global, 0..N = layer index
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
    /// Tensors `load` rewrote (llama's reordered q/k rows) when `data` maps
    /// the file: their bytes live here, each in its own anonymous mapping,
    /// and `tensor_data` reads them from it (the mapping stays the file's,
    /// shared and read-only). Empty otherwise (the copy is rewritten in place).
    side: Vec<Side>,
    /// The file `data` maps, if it does (`tensor_file`).
    file: Option<File>,
    /// `read_ahead` started its thread.
    ahead: bool,
}

/// A rewritten tensor's bytes outside a file mapping (`Model::side`).
/// Never dropped while the model lives: a file mapping's released pages
/// fault back in with the same bytes, so slices taken before a `release`
/// stay valid there (tests run a per-row model next to a repacking one on
/// one mapping); dropped side pages would read as zeros through them.
struct Side {
    /// The tensor's byte range in `data`.
    range: std::ops::Range<usize>,
    /// Anonymous, read-only, range.len() bytes (at least 1).
    map: memmap2::Mmap,
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
        if let Some(s) = self.side_of(&r) {
            return &s.map[..r.len()];
        }
        if !self.file_backed && self.any_released.load(std::sync::atomic::Ordering::Acquire) {
            // Unlocked before the panic: a caught one leaves the lock usable.
            let gone =
                (self.released.lock().unwrap().iter()).any(|x| x.start < r.end && r.start < x.end);
            assert!(
                !gone,
                "tensor {} was released after repacking; load the model again to read it",
                t.name
            );
        }
        &self.data[r]
    }

    /// The side copy holding bytes `r` of `data`, if any.
    fn side_of(&self, r: &std::ops::Range<usize>) -> Option<&Side> {
        if self.side.is_empty() || r.is_empty() {
            return None;
        }
        self.side.iter().find(|s| s.range == *r)
    }

    /// Where `t`'s bytes (`tensor_data`'s) can be read with pread: the
    /// mapped file and the offset in it. None on the anonymous copy and for
    /// a rewritten tensor (its bytes are in a side copy, not the file).
    pub fn tensor_file(&self, t: &Tensor) -> Option<(&File, u64)> {
        let r = self.tensor_range(t);
        let f = self.file.as_ref().filter(|_| self.file_backed)?;
        self.side_of(&r).is_none().then_some((f, r.start as u64))
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
    /// so `tensor_data` panics on a released tensor from then on. A tensor
    /// with a side copy (`side`) keeps it (see `Side`); only its untouched
    /// file pages go.
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
        let Some(std::ops::Range { start, end }) =
            release_span(base, rel[lo].start..rel[hi].end, align)
        else {
            return;
        };
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

    /// Fill in the page tables for `t`'s bytes on a file mapping
    /// (MADV_POPULATE_READ: from the page cache, reading the file where it
    /// isn't there), so a copy out of them doesn't stop at a page fault
    /// every 64 KB. Nothing on the anonymous copy. Best effort.
    pub fn prefault(&self, t: &Tensor) {
        let r = self.tensor_range(t);
        if self.file_backed && !r.is_empty() {
            let _ = self
                .data
                .advise_range(memmap2::Advice::PopulateRead, r.start, r.len());
        }
    }

    /// The fraction of the tensor data's file pages in the page cache
    /// (mincore), or None on the anonymous copy or if it can't be told.
    /// ~1 ms for a 5 GB file.
    pub fn resident_fraction(&self) -> Option<f64> {
        if !self.file_backed || !cfg!(target_os = "linux") {
            return None;
        }
        unsafe extern "C" {
            fn mincore(addr: *mut std::ffi::c_void, len: usize, vec: *mut u8) -> i32;
        }
        let page = page_size();
        let start = self.data_offset as usize / page * page;
        let len = self.data.len().checked_sub(start).filter(|&l| l > 0)?;
        let mut v = vec![0u8; len.div_ceil(page)];
        // SAFETY: [start, start + len) lies in our mapping and starts on a
        // page boundary (the mapping does); v has a byte per page.
        let r = unsafe {
            mincore(
                self.data.as_ptr().add(start).cast_mut().cast(),
                len,
                v.as_mut_ptr(),
            )
        };
        (r == 0).then(|| v.iter().filter(|&&b| b & 1 != 0).count() as f64 / v.len() as f64)
    }

    /// Whether the background read-ahead (`read_ahead`) was started.
    pub fn reading_ahead(&self) -> bool {
        self.ahead
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
/// The whole `align` blocks (in absolute addresses) inside bytes `r` of a
/// buffer at address `base`, as offsets into it, or None if there are none.
/// Aligned first, then made relative: `base` need not be aligned, and a run
/// ending inside base's own block has no block to give back (subtracting
/// base from its rounded-down end would underflow).
fn release_span(
    base: usize,
    r: std::ops::Range<usize>,
    align: usize,
) -> Option<std::ops::Range<usize>> {
    let s = (base + r.start).next_multiple_of(align);
    let e = (base + r.end) / align * align;
    (s < e).then(|| s - base..e - base)
}

fn map_weights(file: &File, copy: Option<bool>) -> io::Result<(memmap2::Mmap, bool)> {
    let len = file.metadata()?.len() as usize;
    if copy_choice(copy, len, true) {
        Ok((copy_file(file, len)?, false))
    } else {
        // SAFETY: the file is opened read-only and not modified while mapped.
        Ok((unsafe { memmap2::Mmap::map(file)? }, true))
    }
}

/// Whether to copy a `len`-byte file (`map_weights`): `copy` if given, else
/// AXON_MMAP's, else `wanted` unless it's over 60% of `MemAvailable`.
fn copy_choice(copy: Option<bool>, len: usize, wanted: bool) -> bool {
    match (copy, std::env::var("AXON_MMAP").as_deref()) {
        (Some(c), _) => c,
        (None, Ok("0")) => true,
        (None, Ok(_)) => false,
        (None, Err(_)) if !wanted => false,
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
    }
}

/// The first `len` bytes of `file` in anonymous memory, transparent huge
/// pages requested (`map_weights`).
fn copy_file(file: &File, len: usize) -> io::Result<memmap2::Mmap> {
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
    anon.make_read_only()
}

/// The system page size (sysconf), 4096 if it can't be told.
fn page_size() -> usize {
    unsafe extern "C" {
        fn sysconf(name: i32) -> std::ffi::c_long;
    }
    const SC_PAGESIZE: i32 = 30; // Linux
    // SAFETY: sysconf has no preconditions.
    let p = unsafe { sysconf(SC_PAGESIZE) };
    if p > 0 { p as usize } else { 4096 }
}

/// `MemAvailable` from /proc/meminfo, in bytes.
pub(crate) fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn load(path: &str) -> io::Result<Model> {
    load_with(path, None)
}

/// `load`, with the copy-vs-map choice forced (`copy`: see `map_weights`).
pub fn load_with(path: &str, copy: Option<bool>) -> io::Result<Model> {
    let file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    let (data, file_backed) = map_weights(&file, copy)?;
    let h = parse(&data[..len.min(data.len())])?;
    let mut m = Model::new(h, data, file_backed);
    m.unpermute_rope()?;
    if file_backed {
        m.file = Some(file);
    }
    Ok(m)
}

/// Load a model to run it (`axon gen` / `chat`): the file is mapped and
/// parsed, `wants_copy` is asked (and may check the model), then the file
/// is copied into anonymous huge pages (`map_weights`) if `copy` says so,
/// else (None) if `wants_copy` said the huge pages would pay,
/// i.e. the weights are read in place per token rather than repacked
/// elsewhere once (then a copy is a second pass over the file and twice the
/// memory while loading); AXON_MMAP and the `MemAvailable` guard apply to
/// that choice as in `load`. A model left mapped is read ahead
/// (`Model::read_ahead`). It stays mapped for the whole run: rewriting the
/// file in place meanwhile (not replacing it) can kill the process with
/// SIGBUS, as with any mapped file; AXON_MMAP=0 copies instead.
/// `wants_copy` says what the model wants: `Want(bool)`, the copy that
/// pays or not (the choice above); `MapOnly`, mapped whatever `copy` and
/// AXON_MMAP say (a MoE model: a copy beside its r8 arena would hold the
/// experts twice).
pub fn load_to_run(
    path: &str,
    copy: Option<bool>,
    wants_copy: impl FnOnce(&Model) -> CopyWish,
) -> io::Result<Model> {
    let file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    // SAFETY: the file is opened read-only and not modified while mapped.
    let data = unsafe { memmap2::Mmap::map(&file)? };
    let h = parse(&data[..len.min(data.len())])?;
    let mut m = Model::new(h, data, true);
    // Always asked (the caller may check the model there), used if `copy`
    // is None.
    let copy = match wants_copy(&m) {
        CopyWish::MapOnly => false,
        CopyWish::Want(wanted) => copy.unwrap_or_else(|| copy_choice(None, len, wanted)),
    };
    if copy {
        // Same bytes (parse checked every tensor range against len).
        m.data = copy_file(&file, len)?;
        m.file_backed = false;
    }
    // After the copy-vs-map choice (in place on a copy, side copies on the
    // file mapping) and before the read-ahead (the file pages it hands
    // back are never read again).
    m.unpermute_rope()?;
    if !copy {
        m.read_ahead(&file);
        m.file = Some(file);
    }
    Ok(m)
}

/// `load_to_run`'s copy-vs-map wish of a model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CopyWish {
    /// Whether a copy pays (reads in place), subject to `copy` / AXON_MMAP.
    Want(bool),
    /// Always mapped.
    MapOnly,
}

/// Bytes per read-ahead request. One MADV_WILLNEED reads at most the
/// device's read-ahead window (max of read_ahead_kb and the largest I/O;
/// 1 MB or more on NVMe) and drops the rest silently; on a device with a
/// smaller window the pages skipped are read when first touched instead.
const READ_AHEAD_CHUNK: usize = 1 << 20;

impl Model {
    /// Start reading the file into the page cache on a thread, in the order
    /// the backends read it: the small tensors (norms, biases: read one by
    /// one before the matrices), then the embedding and output head, then
    /// layer by layer. Each request is MADV_WILLNEED on a mapping of its
    /// own: the kernel queues the reads and returns, so the disk stays busy
    /// while the caller does other work (CUDA init, parsing, the
    /// tokenizer), and on a warm page cache it's a lookup per page. With a
    /// cold page cache the weights otherwise come in one page fault at a
    /// time. Not when the file is over half of `MemAvailable`: then pages
    /// read far ahead could be dropped again before they're used (a split
    /// model hands its uploaded part back as it goes). AXON_READAHEAD=0
    /// turns it off. Best effort: errors are ignored.
    fn read_ahead(&mut self, file: &File) {
        if std::env::var("AXON_READAHEAD").is_ok_and(|v| v == "0")
            || mem_available().is_none_or(|a| self.data.len() as u64 > a / 2)
        {
            return;
        }
        let Ok(file) = file.try_clone() else { return };
        let base = self.data_offset as usize;
        let mut spans: Vec<(usize, usize, usize)> = self
            .tensors
            .iter()
            .filter(|t| t.n_bytes > 0)
            .map(|t| {
                let rank = if t.n_bytes < (1 << 20) {
                    0
                } else if let Some(l) = t.name.strip_prefix("blk.") {
                    2 + l
                        .split('.')
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or(0)
                } else {
                    1
                };
                let s = base + t.offset as usize;
                (rank, s, s + t.n_bytes as usize)
            })
            .collect();
        spans.sort_unstable();
        self.ahead = std::thread::Builder::new()
            .name("axon-readahead".into())
            .spawn(move || {
                // SAFETY: read-only, never written through; only advised.
                let Ok(map) = (unsafe { memmap2::Mmap::map(&file) }) else {
                    return;
                };
                for (_, s, e) in spans {
                    // advise_range rounds the start down to a page.
                    let (mut at, e) = (s, e.min(map.len()));
                    while at < e {
                        let n = READ_AHEAD_CHUNK.min(e - at);
                        if map.advise_range(memmap2::Advice::WillNeed, at, n).is_err() {
                            return;
                        }
                        at += n;
                    }
                }
            })
            .is_ok();
    }
}

impl Model {
    fn new(h: Header, data: memmap2::Mmap, file_backed: bool) -> Model {
        Model {
            arch: h.arch,
            metadata: h.metadata,
            tensors: h.tensors,
            data,
            data_offset: h.data_offset,
            file_backed,
            released: Default::default(),
            any_released: Default::default(),
            side: Vec::new(),
            file: None,
            ahead: false,
        }
    }

    /// llama: attn_q / attn_k rows back in rotate-half order
    /// (`rope_unpermute_plan`). On a file mapping each reordered tensor gets
    /// its own anonymous copy (`Model::side`, kept for the model's life; the
    /// mapping stays the file's, shared and read-only) and its file pages,
    /// never read again, go back right away; on the anonymous copy, in
    /// place. Nothing to do for other architectures.
    fn unpermute_rope(&mut self) -> io::Result<()> {
        let perm = rope_unpermute_plan(&self.arch, &self.metadata, &self.tensors);
        if perm.is_empty() {
            return Ok(());
        }
        let base = self.data_offset as usize;
        if self.file_backed {
            for &(i, heads, sub) in &perm {
                let t = &self.tensors[i];
                let s = base + t.offset as usize;
                let range = s..s + t.n_bytes as usize;
                let mut m = memmap2::MmapMut::map_anon(range.len().max(1))?;
                m[..range.len()].copy_from_slice(&self.data[range.clone()]);
                unpermute_rows(&mut m[..range.len()], t, heads, sub);
                self.side.push(Side {
                    range,
                    map: m.make_read_only()?,
                });
            }
            let rewritten: Vec<usize> = (self.side.iter())
                .filter_map(|sc| {
                    (self.tensors.iter()).position(|t| self.tensor_range(t) == sc.range)
                })
                .collect();
            for i in rewritten {
                self.release(&self.tensors[i]);
            }
        } else {
            let placeholder = memmap2::MmapMut::map_anon(1)?.make_read_only()?;
            let mut w = std::mem::replace(&mut self.data, placeholder).make_mut()?;
            apply_rope_unpermute(&mut w, base, &self.tensors, &perm);
            self.data = w.make_read_only()?;
        }
        Ok(())
    }
}

/// Tensors whose rows `load` reorders, with their head counts: on a
/// `llama` GGUF, attn_q and attn_k (weights and any biases). llama.cpp's
/// converter permutes those rows per head so that its "normal" RoPE, which
/// turns adjacent pairs (2i, 2i + 1), gives the Hugging Face model's rotate-
/// half RoPE. Putting the rows back makes the model a NEOX-RoPE one like
/// qwen2's (pair i is dims i and i + head_dim / 2, same frequency), so every
/// kernel applies as is. Queries and keys of a head get the same
/// permutation, which leaves every q.k (and so the whole model) unchanged;
/// values aren't touched. Empty for other architectures or if the head
/// counts are missing (`check_supported` reports that).
///
/// deepseek2 (MLA) turns only a 64-dim sub-head the same "normal" way: the
/// last `rope.dimension_count` rows of each `key_length_mla` query head
/// (attn_q / attn_q_b) and of the one shared key (attn_kv_a_mqa, after the
/// `kv_lora_rank` latent rows). Those rows get the same reordering, inside
/// their range only (`Sub`).
fn rope_unpermute_plan(arch: &str, metadata: &[(String, Val)], tensors: &[Tensor]) -> Vec<Perm> {
    let num = |k: &str| {
        metadata
            .iter()
            .find(|(n, _)| n == &format!("{arch}.{k}"))
            .and_then(|(_, v)| v.as_f64())
            .filter(|&v| v >= 1.0 && v.fract() == 0.0)
            .map(|v| v as usize)
    };
    let mut out = Vec::new();
    if arch == "deepseek2" {
        let (Some(nh), Some(rope), Some(lat)) = (
            num("attention.head_count"),
            num("rope.dimension_count"),
            num("attention.kv_lora_rank"),
        ) else {
            return out;
        };
        let Some(hk) = num("attention.key_length_mla") else {
            return out;
        };
        if rope > hk || !rope.is_multiple_of(2) {
            return out;
        }
        for (i, t) in tensors.iter().enumerate() {
            let (heads, sub) = match t.kind.as_str() {
                "attn_q.weight" | "attn_q_b.weight" => (nh, (hk - rope, rope)),
                "attn_kv_a_mqa.weight" => (1, (lat, rope)),
                _ => continue,
            };
            if t.layer >= 0 {
                out.push((i, heads, Some(sub)));
            }
        }
        return out;
    }
    if arch != "llama" {
        return out;
    }
    let (Some(nh), Some(nkv)) = (num("attention.head_count"), num("attention.head_count_kv"))
    else {
        return out;
    };
    for (i, t) in tensors.iter().enumerate() {
        let heads = match t.kind.as_str() {
            "attn_q.weight" | "attn_q.bias" => nh,
            "attn_k.weight" | "attn_k.bias" => nkv,
            _ => continue,
        };
        if t.layer >= 0 {
            out.push((i, heads, None));
        }
    }
    out
}

/// Rows off..off + len of each head (`rope_unpermute_plan`'s deepseek2
/// sub-heads); None: the whole head.
type Sub = Option<(usize, usize)>;

/// A tensor to reorder: its index, its head count, the rows of each head.
type Perm = (usize, usize, Sub);

/// Undo the converter's per-head row permutation (see
/// `rope_unpermute_plan`) on each planned tensor of `data`: within a head of
/// hd rows, GGUF row 2i + j holds the original row j * hd / 2 + i. Rows are
/// whole quant blocks (dims[0] values each), so this moves bytes only.
/// Tensors whose shape doesn't split into heads of an even size are left as
/// they are (`check_supported` rejects them).
fn apply_rope_unpermute(data: &mut [u8], base: usize, tensors: &[Tensor], plan: &[Perm]) {
    for &(i, heads, sub) in plan {
        let t = &tensors[i];
        let s = base + t.offset as usize;
        unpermute_rows(&mut data[s..s + t.n_bytes as usize], t, heads, sub);
    }
}

/// `apply_rope_unpermute` on one tensor's bytes `b`. Left as is (and
/// rejected by `check_supported`) if its shape doesn't split into whole
/// rows of whole quant blocks and heads of an even size. With `sub`
/// (off, len), only rows off..off + len of each head move, among
/// themselves.
fn unpermute_rows(b: &mut [u8], t: &Tensor, heads: usize, sub: Sub) {
    let rows = match t.dims.len() {
        1 => t.dims[0],
        n if n >= 2 => t.dims[1..].iter().product(),
        _ => return,
    } as usize;
    // A 1-D tensor (a bias) has one value per row.
    let cols = if t.dims.len() == 1 { 1 } else { t.dims[0] };
    if t.n_bytes == 0
        || rows == 0
        || !rows.is_multiple_of(heads)
        || (t.dims.len() >= 2 && !cols.is_multiple_of(t.qtype.block_size as u64))
    {
        return;
    }
    let hd = rows / heads;
    let rb = t.n_bytes as usize / rows;
    let (off, len) = sub.unwrap_or((0, hd));
    if !len.is_multiple_of(2)
        || off + len > hd
        || rb * rows != t.n_bytes as usize
        || b.len() != t.n_bytes as usize
    {
        return;
    }
    let mut tmp = Vec::with_capacity(len * rb);
    for head in b.chunks_exact_mut(hd * rb) {
        let part = &mut head[off * rb..(off + len) * rb];
        tmp.clear();
        tmp.extend_from_slice(part);
        for (r, src) in tmp.chunks_exact(rb).enumerate() {
            let (i, j) = (r / 2, r % 2);
            let d = j * len / 2 + i;
            part[d * rb..(d + 1) * rb].copy_from_slice(src);
        }
    }
}

/// Everything in a GGUF file but the tensor data.
struct Header {
    arch: String,
    metadata: Vec<(String, Val)>,
    tensors: Vec<Tensor>,
    data_offset: u64,
}

fn parse(buf: &[u8]) -> io::Result<Header> {
    let mut r = Reader {
        buf,
        pos: 0,
        elems_left: MAX_META_ELEMS,
    };

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

    // No two tensors may share bytes. Writers lay tensors out one after
    // another; overlapping ranges (a type or size changed in place) broke
    // `Model::release`'s bookkeeping: a repacked tensor's released bytes
    // were another tensor's too, and reading that one panicked.
    let mut spans: Vec<(u64, u64, usize)> = (tensors.iter().enumerate())
        .filter(|(_, t)| t.n_bytes > 0)
        .map(|(i, t)| (t.offset, t.offset + t.n_bytes, i))
        .collect();
    spans.sort_unstable();
    for w in spans.windows(2) {
        if w[1].0 < w[0].1 {
            return Err(bad(format!(
                "tensors {} and {} overlap in the file; it is corrupt",
                tensors[w[0].2].name, tensors[w[1].2].name
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
    build_for_test_with(meta, tensors, &[])
}

/// `build_for_test` with tensor i's bytes from data[i] (exactly its size;
/// tensors past data.len() stay zero).
#[cfg(test)]
pub fn build_for_test_with(
    meta: &[(&str, Val)],
    tensors: &[(&str, &[u64], u32)],
    data: &[Vec<u8>],
) -> Model {
    let b = gguf_bytes_for_test(meta, tensors, data);
    let h = parse(&b).expect("build_for_test: parse");
    let mut data = memmap2::MmapMut::map_anon(b.len()).unwrap();
    data.copy_from_slice(&b);
    let perm = rope_unpermute_plan(&h.arch, &h.metadata, &h.tensors);
    apply_rope_unpermute(&mut data, h.data_offset as usize, &h.tensors, &perm);
    Model {
        arch: h.arch,
        metadata: h.metadata,
        tensors: h.tensors,
        data: data.make_read_only().unwrap(),
        data_offset: h.data_offset,
        file_backed: false,
        released: Default::default(),
        any_released: Default::default(),
        side: Vec::new(),
        file: None,
        ahead: false,
    }
}

/// The GGUF file `build_for_test_with` parses (tests that hand a fixture
/// to llama.cpp write it out).
#[cfg(test)]
pub fn gguf_bytes_for_test(
    meta: &[(&str, Val)],
    tensors: &[(&str, &[u64], u32)],
    data: &[Vec<u8>],
) -> Vec<u8> {
    fn str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }
    fn val(b: &mut Vec<u8>, v: &Val) {
        match v {
            Val::U32(x) => b.extend_from_slice(&x.to_le_bytes()),
            Val::I32(x) => b.extend_from_slice(&x.to_le_bytes()),
            Val::F32(x) => b.extend_from_slice(&x.to_le_bytes()),
            Val::Bool(x) => b.push(u8::from(*x)),
            Val::Str(s) => str(b, s),
            Val::Arr(a) => {
                b.extend_from_slice(&a.first().map_or(8, type_id).to_le_bytes());
                b.extend_from_slice(&(a.len() as u64).to_le_bytes());
                a.iter().for_each(|x| val(b, x));
            }
            _ => unimplemented!("build_for_test: {v:?}"),
        }
    }
    fn type_id(v: &Val) -> u32 {
        match v {
            Val::U32(_) => 4,
            Val::I32(_) => 5,
            Val::F32(_) => 6,
            Val::Bool(_) => 7,
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
    let mut at = Vec::with_capacity(tensors.len());
    for (name, dims, ty) in tensors {
        str(&mut b, name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        dims.iter()
            .for_each(|d| b.extend_from_slice(&d.to_le_bytes()));
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        let q = qtype(*ty);
        let n: u64 = dims.iter().product();
        let bytes = n / q.block_size as u64 * q.block_bytes as u64;
        at.push((offset as usize, bytes as usize));
        offset += bytes.next_multiple_of(32);
    }
    let base = b.len().next_multiple_of(32);
    b.resize(base + offset as usize, 0);
    for (d, &(off, len)) in data.iter().zip(&at) {
        assert_eq!(d.len(), len, "build_for_test_with: tensor bytes");
        b[base + off..base + off + len].copy_from_slice(d);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The row reordering on its own: head sizes 64 and 128, a quantized
    /// matrix (Q8_0 rows of 2 blocks) and a 1-D F32 bias, against the
    /// converter's permutation applied to known rows (HF row j*hd/2 + i is
    /// stored at 2i + j; undoing it must give the HF order back).
    #[test]
    fn unpermute_rows_undoes_the_converter() {
        for (hd, heads) in [(64usize, 3usize), (128, 2)] {
            let rows = hd * heads;
            // HF order: row r's bytes all equal r (mod 251), so the order is visible.
            for (dims, qt, rb) in [
                (vec![64u64, rows as u64], qtype(crate::quant::Q8_0), 2 * 34),
                (vec![rows as u64], qtype(crate::quant::F32), 4),
            ] {
                let hf: Vec<u8> = (0..rows).flat_map(|r| vec![(r % 251) as u8; rb]).collect();
                // The converter's order: GGUF row h*hd + 2i + j <- HF row h*hd + j*hd/2 + i.
                let mut gguf = vec![0u8; hf.len()];
                for h in 0..heads {
                    for i in 0..hd / 2 {
                        for j in 0..2 {
                            let (src, dst) = (h * hd + j * hd / 2 + i, h * hd + 2 * i + j);
                            gguf[dst * rb..(dst + 1) * rb]
                                .copy_from_slice(&hf[src * rb..(src + 1) * rb]);
                        }
                    }
                }
                let t = Tensor {
                    name: "blk.0.attn_q.weight".into(),
                    n_elements: dims.iter().product(),
                    n_bytes: hf.len() as u64,
                    dims,
                    qtype: qt,
                    offset: 0,
                    layer: 0,
                    kind: "attn_q.weight".into(),
                };
                unpermute_rows(&mut gguf, &t, heads, None);
                assert_eq!(gguf, hf, "hd {hd} type {}", qt.name);
            }
        }
    }

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

    /// `release`'s block arithmetic on unaligned bases: runs ending inside
    /// the base's own block (which underflowed before), runs with no whole
    /// block, aligned and unaligned bases, and the span always inside the run.
    #[test]
    fn release_span_unaligned_base() {
        const A: usize = 2 << 20;
        // Base 4 KiB past a 2 MiB boundary: a run ending before the next
        // boundary has no whole block.
        let base = 7 * A + 4096;
        assert_eq!(release_span(base, 0..1000, A), None);
        assert_eq!(release_span(base, 100..A - 4096 - 1, A), None);
        // Exactly up to the next boundary: still no whole block.
        assert_eq!(release_span(base, 0..A - 4096, A), None);
        // One whole block.
        assert_eq!(
            release_span(base, 0..2 * A - 4096, A),
            Some(A - 4096..2 * A - 4096)
        );
        // Aligned base.
        assert_eq!(release_span(8 * A, 0..A, A), Some(0..A));
        assert_eq!(release_span(8 * A, 1..A, A), None);
        assert_eq!(release_span(8 * A, 5..3 * A + 7, A), Some(A..3 * A));
        // Empty run.
        assert_eq!(release_span(base, 50..50, A), None);
        for base in [0, 4096, A - 4096, A + 123, 3 * A] {
            for (s, e) in [(0, 10), (0, A), (A / 2, 5 * A / 2), (17, 4 * A + 3)] {
                if let Some(r) = release_span(base, s..e, A) {
                    assert!(s <= r.start && r.end <= e && r.start < r.end);
                    assert_eq!((base + r.start) % A, 0);
                    assert_eq!((base + r.end) % A, 0);
                }
            }
        }
    }

    /// `release` on an in-memory model (anonymous, base not 2 MiB-aligned
    /// in general) of small tensors: no panic under overflow checks, and
    /// the released tensors are recorded.
    #[test]
    fn release_small_anonymous_model() {
        let names: Vec<String> = (0..6).map(|i| format!("t{i}")).collect();
        let ts: Vec<(&str, &[u64], u32)> = names
            .iter()
            .map(|n| (n.as_str(), &[64u64][..], 0))
            .collect();
        let m = build_for_test(&[("general.architecture", Val::Str("x".into()))], &ts);
        for t in &m.tensors {
            m.release(t);
            assert!(m.is_released(t));
        }
    }

    /// `resident_fraction`: all of a file just written and read is in the
    /// page cache; the anonymous copy has no answer.
    #[test]
    fn resident_fraction_of_a_read_file() {
        let path = std::env::temp_dir().join(format!("axon-resident-{}.gguf", std::process::id()));
        std::fs::write(&path, valid().pad_to(32, 64 + (1 << 20))).unwrap();
        let p = path.to_str().unwrap();
        let mapped = load_with(p, Some(false)).unwrap();
        let _ = mapped.data.iter().fold(0u8, |a, &b| a ^ b); // touch every page
        let f = mapped.resident_fraction().expect("a file mapping");
        let copied = load_with(p, Some(true)).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(f > 0.99, "{f}");
        assert_eq!(copied.resident_fraction(), None);
    }

    /// `load_to_run`: `MapOnly` maps even when `copy` (or so AXON_MMAP=0,
    /// which is only read without it) asks for the copy; `Want` follows it.
    #[test]
    fn load_to_run_map_only_never_copies() {
        let path = std::env::temp_dir().join(format!("axon-maponly-{}.gguf", std::process::id()));
        std::fs::write(&path, valid().pad_to(32, 64 + 4096)).unwrap();
        let p = path.to_str().unwrap();
        let mapped = load_to_run(p, Some(true), |_| CopyWish::MapOnly).unwrap();
        let copied = load_to_run(p, Some(true), |_| CopyWish::Want(false)).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(mapped.file_backed());
        assert!(!copied.file_backed());
    }

    /// A string array long enough for the threaded parse (`Reader::strings`)
    /// reads as it would one by one; a bad string in it is an error.
    #[test]
    fn parses_a_long_string_array() {
        let n = PAR_STRINGS * 2 + 7;
        let texts: Vec<String> = (0..n)
            .map(|i| format!("t{i}é{}", "x".repeat(i % 13)))
            .collect();
        let file = |bad: Option<usize>| {
            let mut w = W::header(3, 2, 2)
                .str("general.architecture")
                .u32(8)
                .str("qwen2")
                .str("tokenizer.ggml.tokens")
                .u32(9)
                .u32(8)
                .u64(n as u64);
            for (i, t) in texts.iter().enumerate() {
                w = w.str(t);
                if bad == Some(i) {
                    let l = w.0.len();
                    w.0[l - 1] = 0xff; // the last byte of this string
                }
            }
            w.tensor("blk.0.attn_norm.weight", 8, 0)
                .tensor("output_norm.weight", 8, 32)
                .pad_to(32, 64)
        };
        let h = parse(&file(None)).expect("valid file");
        let Some((_, Val::Arr(a))) = h
            .metadata
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.tokens")
        else {
            panic!("no array")
        };
        assert_eq!(a.len(), n);
        assert!(
            a.iter()
                .zip(&texts)
                .all(|(v, t)| matches!(v, Val::Str(s) if s == t))
        );
        assert_eq!(h.tensors.len(), 2);
        for i in [0, PAR_STRINGS + 3, n - 1] {
            assert!(
                err(&file(Some(i))).contains("not valid UTF-8"),
                "bad string {i}"
            );
        }
        let f = file(None);
        assert!(err(&f[..f.len() / 2]).contains("truncated"));
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

    /// Metadata arrays share one element budget (`MAX_META_ELEMS`): a u8
    /// array is 1 byte per element in the file but a 32-byte `Val` in
    /// memory, so 150 MB of header aborted on a 4 GB address-space cap.
    #[test]
    fn metadata_element_budget() {
        let u8_array = |w: W, key: &str, n: u64| {
            let mut w = w.str(key).u32(9).u32(0).u64(n);
            w.0.resize(w.0.len() + n as usize, 7);
            w
        };
        let one = u8_array(W::header(3, 0, 1), "a", MAX_META_ELEMS + 1).pad_to(8, 0);
        assert!(err(&one).contains("more than"));
        // Cumulative: 1000 values, then one array that fits the file but
        // not what is left of the budget.
        let w = u8_array(W::header(3, 0, 2), "a", 1000);
        let two = u8_array(w, "b", MAX_META_ELEMS - 999).pad_to(8, 0);
        assert!(err(&two).contains("more than"));
        let w = u8_array(W::header(3, 0, 2), "a", 1000);
        let ok = u8_array(w, "b", 1000).pad_to(8, 0);
        assert_eq!(parse(&ok).expect("2000 values").metadata.len(), 2);
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

    /// Tensors sharing bytes are refused (a fuzzed tied embedding retyped
    /// to F16 in place overlapped the matrices after it, which were
    /// repacked and released; reading the embedding then panicked).
    /// Touching ranges and zero-size tensors are fine.
    #[test]
    fn overlapping_tensors_refused() {
        let buf = W::header(3, 2, 0)
            .tensor("a", 16, 0)
            .tensor("b", 8, 32)
            .pad_to(32, 96);
        assert!(err(&buf).contains("tensors a and b overlap"));
        // b inside a, listed first.
        let buf = W::header(3, 2, 0)
            .tensor("b", 8, 32)
            .tensor("a", 24, 0)
            .pad_to(32, 96);
        assert!(err(&buf).contains("overlap"));
        let buf = W::header(3, 3, 0)
            .tensor("a", 8, 0)
            .tensor("b", 8, 32)
            .tensor("z", 0, 32)
            .pad_to(32, 64);
        assert_eq!(parse(&buf).expect("touching").tensors.len(), 3);
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
