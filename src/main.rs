use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

/// GGUF magic: "GGUF" in little-endian
const GGUF_MAGIC: u32 = 0x46554747;

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_bool(r: &mut impl Read) -> io::Result<bool> {
    Ok(read_u8(r)? != 0)
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// GGUF value types
#[derive(Debug)]
enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl std::fmt::Display for GGUFValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGUFValue::U8(v) => write!(f, "{v}"),
            GGUFValue::I8(v) => write!(f, "{v}"),
            GGUFValue::U16(v) => write!(f, "{v}"),
            GGUFValue::I16(v) => write!(f, "{v}"),
            GGUFValue::U32(v) => write!(f, "{v}"),
            GGUFValue::I32(v) => write!(f, "{v}"),
            GGUFValue::F32(v) => write!(f, "{v}"),
            GGUFValue::Bool(v) => write!(f, "{v}"),
            GGUFValue::String(v) => write!(f, "\"{v}\""),
            GGUFValue::Array(v) => write!(f, "[{} items]", v.len()),
            GGUFValue::U64(v) => write!(f, "{v}"),
            GGUFValue::I64(v) => write!(f, "{v}"),
            GGUFValue::F64(v) => write!(f, "{v}"),
        }
    }
}

fn read_value(r: &mut impl Read, vtype: u32) -> io::Result<GGUFValue> {
    match vtype {
        0 => Ok(GGUFValue::U8(read_u8(r)?)),
        1 => Ok(GGUFValue::I8(read_u8(r)? as i8)),
        2 => { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(GGUFValue::U16(u16::from_le_bytes(b))) }
        3 => { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(GGUFValue::I16(i16::from_le_bytes(b))) }
        4 => Ok(GGUFValue::U32(read_u32(r)?)),
        5 => Ok(GGUFValue::I32(read_i32(r)?)),
        6 => Ok(GGUFValue::F32(read_f32(r)?)),
        7 => Ok(GGUFValue::Bool(read_bool(r)?)),
        8 => Ok(GGUFValue::String(read_string(r)?)),
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                arr.push(read_value(r, elem_type)?);
            }
            Ok(GGUFValue::Array(arr))
        }
        10 => Ok(GGUFValue::U64(read_u64(r)?)),
        11 => { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(GGUFValue::I64(i64::from_le_bytes(b))) }
        12 => { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(GGUFValue::F64(f64::from_le_bytes(b))) }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown GGUF value type {vtype}"))),
    }
}

/// Quantization type names
fn qtype_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1",
        6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
        14 => "Q6_K", 15 => "Q8_K", 16 => "IQ2_XXS", 17 => "IQ2_XS",
        18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S",
        22 => "IQ2_S", 23 => "IQ4_XS", 24 => "I8", 25 => "I16",
        26 => "I32", 27 => "I64", 28 => "F64", 29 => "IQ1_M",
        _ => "??",
    }
}

/// Bytes per block for a quantization type (block_size, bytes_per_block)
fn qtype_block(t: u32) -> (usize, usize) {
    match t {
        0 => (1, 4),    // F32
        1 => (1, 2),    // F16
        2 => (32, 18),  // Q4_0
        3 => (32, 20),  // Q4_1
        6 => (32, 22),  // Q5_0
        7 => (32, 24),  // Q5_1
        8 => (32, 34),  // Q8_0
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        _ => (1, 1),
    }
}

#[derive(Debug)]
struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    qtype: u32,
    offset: u64,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: axon <model.gguf>");
        std::process::exit(1);
    }

    let f = File::open(&args[1])?;
    let mut r = BufReader::new(f);

    // Header
    let magic = read_u32(&mut r)?;
    if magic != GGUF_MAGIC {
        eprintln!("not a GGUF file (magic: 0x{magic:08X})");
        std::process::exit(1);
    }

    let version = read_u32(&mut r)?;
    let n_tensors = read_u64(&mut r)?;
    let n_metadata = read_u64(&mut r)?;

    println!("axon v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("GGUF v{version}  |  {n_tensors} tensors  |  {n_metadata} metadata keys");
    println!();

    // Metadata
    let mut arch = String::new();
    let mut ctx_len: u64 = 0;
    let mut embed_dim: u64 = 0;
    let mut n_layers: u64 = 0;
    let mut n_heads: u64 = 0;
    let mut n_kv_heads: u64 = 0;
    let mut vocab_size: u64 = 0;

    println!("── metadata ──");
    for _ in 0..n_metadata {
        let key = read_string(&mut r)?;
        let vtype = read_u32(&mut r)?;
        let value = read_value(&mut r, vtype)?;

        // extract key architecture params
        match key.as_str() {
            "general.architecture" => if let GGUFValue::String(ref s) = value { arch = s.clone(); },
            k if k.ends_with(".context_length") => if let GGUFValue::U32(v) = value { ctx_len = v as u64; },
            k if k.ends_with(".embedding_length") => if let GGUFValue::U32(v) = value { embed_dim = v as u64; },
            k if k.ends_with(".block_count") => if let GGUFValue::U32(v) = value { n_layers = v as u64; },
            k if k.ends_with(".attention.head_count") => if let GGUFValue::U32(v) = value { n_heads = v as u64; },
            k if k.ends_with(".attention.head_count_kv") => if let GGUFValue::U32(v) = value { n_kv_heads = v as u64; },
            k if k.ends_with(".vocab_size") => if let GGUFValue::U32(v) = value { vocab_size = v as u64; },
            _ => {}
        }

        // print interesting ones
        let dominated = key.starts_with("tokenizer.ggml.tokens")
            || key.starts_with("tokenizer.ggml.scores")
            || key.starts_with("tokenizer.ggml.token_type");
        if !dominated {
            println!("  {key}: {value}");
        }
    }

    println!();
    println!("── architecture ──");
    println!("  arch:       {arch}");
    println!("  context:    {ctx_len}");
    println!("  embed:      {embed_dim}");
    println!("  layers:     {n_layers}");
    println!("  heads:      {n_heads} (kv: {n_kv_heads})");
    if vocab_size > 0 { println!("  vocab:      {vocab_size}"); }

    // Tensor info
    println!();
    println!("── tensors ({n_tensors}) ──");

    let mut tensors = Vec::new();
    let mut total_params: u64 = 0;
    let mut total_bytes: u64 = 0;

    for _ in 0..n_tensors {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(&mut r)?);
        }
        let qtype = read_u32(&mut r)?;
        let offset = read_u64(&mut r)?;

        let n_elements: u64 = dims.iter().product();
        let (block_size, bytes_per_block) = qtype_block(qtype);
        let n_bytes = (n_elements as usize / block_size) * bytes_per_block;

        total_params += n_elements;
        total_bytes += n_bytes as u64;

        tensors.push(TensorInfo { name, dims, qtype, offset });
    }

    // print first 10 and last 5
    for (i, t) in tensors.iter().enumerate() {
        if i < 10 || i >= tensors.len() - 5 {
            let shape: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
            println!("  {:60} {:>6} [{}]", t.name, qtype_name(t.qtype), shape.join("x"));
        } else if i == 10 {
            println!("  ... ({} more) ...", tensors.len() - 15);
        }
    }

    println!();
    let gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let params_b = total_params as f64 / 1e9;
    println!("  total: {params_b:.2}B params, {gb:.2} GB");

    Ok(())
}
