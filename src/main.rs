use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read};

const GGUF_MAGIC: u32 = 0x46554747;

// ── reading primitives ──

fn read_u8(r: &mut impl Read) -> io::Result<u8> { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0]) }
fn read_u32(r: &mut impl Read) -> io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn read_u64(r: &mut impl Read) -> io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn read_i32(r: &mut impl Read) -> io::Result<i32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(i32::from_le_bytes(b)) }
fn read_f32(r: &mut impl Read) -> io::Result<f32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b)) }

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ── GGUF values ──

#[derive(Debug)]
enum Val {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32),
    Bool(bool), Str(String), Arr(Vec<Val>), U64(u64), I64(i64), F64(f64),
}

impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Val::U8(v) => write!(f, "{v}"), Val::I8(v) => write!(f, "{v}"),
            Val::U16(v) => write!(f, "{v}"), Val::I16(v) => write!(f, "{v}"),
            Val::U32(v) => write!(f, "{v}"), Val::I32(v) => write!(f, "{v}"),
            Val::F32(v) => write!(f, "{v}"), Val::Bool(v) => write!(f, "{v}"),
            Val::Str(v) => write!(f, "\"{v}\""), Val::Arr(v) => write!(f, "[{} items]", v.len()),
            Val::U64(v) => write!(f, "{v}"), Val::I64(v) => write!(f, "{v}"),
            Val::F64(v) => write!(f, "{v}"),
        }
    }
}

fn read_val(r: &mut impl Read, t: u32) -> io::Result<Val> {
    match t {
        0 => Ok(Val::U8(read_u8(r)?)),
        1 => Ok(Val::I8(read_u8(r)? as i8)),
        2 => { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(Val::U16(u16::from_le_bytes(b))) }
        3 => { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(Val::I16(i16::from_le_bytes(b))) }
        4 => Ok(Val::U32(read_u32(r)?)),
        5 => Ok(Val::I32(read_i32(r)?)),
        6 => Ok(Val::F32(read_f32(r)?)),
        7 => Ok(Val::Bool(read_u8(r)? != 0)),
        8 => Ok(Val::Str(read_string(r)?)),
        9 => {
            let et = read_u32(r)?;
            let n = read_u64(r)? as usize;
            let mut a = Vec::with_capacity(n.min(4096));
            for _ in 0..n { a.push(read_val(r, et)?); }
            Ok(Val::Arr(a))
        }
        10 => Ok(Val::U64(read_u64(r)?)),
        11 => { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(Val::I64(i64::from_le_bytes(b))) }
        12 => { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(Val::F64(f64::from_le_bytes(b))) }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown val type {t}"))),
    }
}

// ── quantization types ──

#[derive(Debug, Clone, Copy)]
pub struct QType {
    pub id: u32,
    pub name: &'static str,
    pub block_size: usize,
    pub block_bytes: usize,
}

const QTYPES: &[QType] = &[
    QType { id: 0,  name: "F32",    block_size: 1,   block_bytes: 4 },
    QType { id: 1,  name: "F16",    block_size: 1,   block_bytes: 2 },
    QType { id: 2,  name: "Q4_0",   block_size: 32,  block_bytes: 18 },
    QType { id: 3,  name: "Q4_1",   block_size: 32,  block_bytes: 20 },
    QType { id: 6,  name: "Q5_0",   block_size: 32,  block_bytes: 22 },
    QType { id: 7,  name: "Q5_1",   block_size: 32,  block_bytes: 24 },
    QType { id: 8,  name: "Q8_0",   block_size: 32,  block_bytes: 34 },
    QType { id: 9,  name: "Q8_1",   block_size: 32,  block_bytes: 36 },
    QType { id: 10, name: "Q2_K",   block_size: 256, block_bytes: 84 },
    QType { id: 11, name: "Q3_K",   block_size: 256, block_bytes: 110 },
    QType { id: 12, name: "Q4_K",   block_size: 256, block_bytes: 144 },
    QType { id: 13, name: "Q5_K",   block_size: 256, block_bytes: 176 },
    QType { id: 14, name: "Q6_K",   block_size: 256, block_bytes: 210 },
    QType { id: 15, name: "Q8_K",   block_size: 256, block_bytes: 292 },
];

fn qtype(id: u32) -> QType {
    QTYPES.iter().find(|q| q.id == id).copied()
        .unwrap_or(QType { id, name: "??", block_size: 1, block_bytes: 1 })
}

// ── tensor descriptor ──

#[derive(Debug)]
pub struct TensorDesc {
    pub name: String,
    pub dims: Vec<u64>,
    pub qtype: QType,
    pub offset: u64,      // byte offset from the start of the tensor data section
    pub n_elements: u64,
    pub n_bytes: u64,
}

// ── the loaded model ──

pub struct Model {
    pub arch: String,
    pub metadata: Vec<(String, Val)>,
    pub tensors: Vec<TensorDesc>,
    pub data: memmap2::Mmap,          // the whole file, memory-mapped
    pub data_offset: u64,             // where the tensor data section starts in the file
}

impl Model {
    /// Get the raw bytes for a tensor (still quantized)
    pub fn tensor_data(&self, t: &TensorDesc) -> &[u8] {
        let start = (self.data_offset + t.offset) as usize;
        let end = start + t.n_bytes as usize;
        &self.data[start..end]
    }

    /// Look up a metadata value by key
    pub fn meta(&self, key: &str) -> Option<&Val> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        match self.meta(key)? { Val::U32(v) => Some(*v), _ => None }
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        match self.meta(key)? { Val::F32(v) => Some(*v), _ => None }
    }

    /// Find a tensor by name
    pub fn tensor(&self, name: &str) -> Option<&TensorDesc> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

// ── loader ──

pub fn load(path: &str) -> io::Result<Model> {
    let file = File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    // parse the header from a buffered reader over the same file
    let f2 = File::open(path)?;
    let mut r = BufReader::new(f2);

    let magic = read_u32(&mut r)?;
    if magic != GGUF_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("not a GGUF file (magic: 0x{magic:08X})")));
    }

    let _version = read_u32(&mut r)?;
    let n_tensors = read_u64(&mut r)?;
    let n_metadata = read_u64(&mut r)?;

    // metadata
    let mut metadata = Vec::new();
    for _ in 0..n_metadata {
        let key = read_string(&mut r)?;
        let vtype = read_u32(&mut r)?;
        let value = read_val(&mut r, vtype)?;
        metadata.push((key, value));
    }

    let arch = metadata.iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| if let Val::Str(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_default();

    // tensor directory
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims { dims.push(read_u64(&mut r)?); }
        let qt = qtype(read_u32(&mut r)?);
        let offset = read_u64(&mut r)?;

        let n_elements: u64 = dims.iter().product();
        let n_bytes = (n_elements as usize / qt.block_size * qt.block_bytes) as u64;

        tensors.push(TensorDesc { name, dims, qtype: qt, offset, n_elements, n_bytes });
    }

    // the tensor data starts at the next alignment boundary (32 bytes) after the header
    // we figure out where the reader is now by summing what we've read
    // easier: the first tensor's data is at file offset = (current_pos aligned to 32) + tensor[0].offset
    // GGUF spec: tensor data starts at alignment(32) after the header
    let header_end = r.stream_position()
        .unwrap_or_else(|_| {
            // fallback: compute from mmap by scanning for the alignment gap
            // the tensor data section is aligned to 32 bytes
            let first_offset = tensors.first().map(|t| t.offset).unwrap_or(0);
            // search backwards from the expected position
            let approx = mmap.len() as u64 - tensors.iter().map(|t| t.n_bytes).sum::<u64>();
            approx - first_offset
        });
    let alignment = 32u64;
    let data_offset = (header_end + alignment - 1) / alignment * alignment;

    Ok(Model { arch, metadata, tensors, data: mmap, data_offset })
}

use std::io::Seek;

// ── main ──

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: axon <model.gguf>");
        std::process::exit(1);
    }

    let model = load(&args[1])?;

    println!("axon v{}", env!("CARGO_PKG_VERSION"));
    println!();

    // architecture summary
    let n = &model.arch;
    let ctx = model.meta_u32(&format!("{n}.context_length")).unwrap_or(0);
    let emb = model.meta_u32(&format!("{n}.embedding_length")).unwrap_or(0);
    let layers = model.meta_u32(&format!("{n}.block_count")).unwrap_or(0);
    let heads = model.meta_u32(&format!("{n}.attention.head_count")).unwrap_or(0);
    let kv_heads = model.meta_u32(&format!("{n}.attention.head_count_kv")).unwrap_or(0);

    println!("{n} | {layers}L / {emb}d / {heads}h (kv {kv_heads}) | ctx {ctx}");
    println!();

    // tensor summary
    let total_params: u64 = model.tensors.iter().map(|t| t.n_elements).sum();
    let total_bytes: u64 = model.tensors.iter().map(|t| t.n_bytes).sum();
    let gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("{} tensors | {:.2}B params | {:.2} GB", model.tensors.len(), total_params as f64 / 1e9, gb);
    println!();

    // verify mmap: read the first 16 bytes of the first tensor
    if let Some(t) = model.tensors.first() {
        let data = model.tensor_data(t);
        let preview: Vec<String> = data[..16.min(data.len())].iter().map(|b| format!("{b:02x}")).collect();
        println!("first tensor: {} ({}, {} bytes)", t.name, t.qtype.name, t.n_bytes);
        println!("  mmap[0..16]: {}", preview.join(" "));
        println!("  data is {} ({})", if data.iter().all(|&b| b == 0) { "all zeros — BAD" } else { "non-zero — OK" },
            if data.len() > 0 { "mapped" } else { "empty" });
    }

    Ok(())
}
