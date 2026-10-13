use std::fs::File;
use std::io::{self, BufReader, Read, Seek};

const GGUF_MAGIC: u32 = 0x46554747;

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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

fn read_val(r: &mut impl Read, t: u32) -> io::Result<Val> {
    match t {
        0 => Ok(Val::U8(read_u8(r)?)),
        1 => Ok(Val::I8(read_u8(r)? as i8)),
        2 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(Val::U16(u16::from_le_bytes(b)))
        }
        3 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(Val::I16(i16::from_le_bytes(b)))
        }
        4 => Ok(Val::U32(read_u32(r)?)),
        5 => Ok(Val::I32(read_i32(r)?)),
        6 => Ok(Val::F32(read_f32(r)?)),
        7 => Ok(Val::Bool(read_u8(r)? != 0)),
        8 => Ok(Val::Str(read_string(r)?)),
        9 => {
            let et = read_u32(r)?;
            let n = read_u64(r)? as usize;
            let mut a = Vec::with_capacity(n.min(4096));
            for _ in 0..n {
                a.push(read_val(r, et)?);
            }
            Ok(Val::Arr(a))
        }
        10 => Ok(Val::U64(read_u64(r)?)),
        11 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(Val::I64(i64::from_le_bytes(b)))
        }
        12 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(Val::F64(f64::from_le_bytes(b)))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown val type {t}"),
        )),
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
}

impl Model {
    pub fn tensor_data(&self, t: &Tensor) -> &[u8] {
        let s = (self.data_offset + t.offset) as usize;
        &self.data[s..s + t.n_bytes as usize]
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

pub fn load(path: &str) -> io::Result<Model> {
    let file = File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let f2 = File::open(path)?;
    let mut r = BufReader::new(f2);

    let magic = read_u32(&mut r)?;
    if magic != GGUF_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a GGUF file",
        ));
    }
    let _ver = read_u32(&mut r)?;
    let n_tensors = read_u64(&mut r)?;
    let n_meta = read_u64(&mut r)?;

    let mut metadata = Vec::new();
    for _ in 0..n_meta {
        let key = read_string(&mut r)?;
        let vt = read_u32(&mut r)?;
        let val = read_val(&mut r, vt)?;
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

    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_string(&mut r)?;
        let nd = read_u32(&mut r)?;
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(read_u64(&mut r)?);
        }
        let qt = qtype(read_u32(&mut r)?);
        let offset = read_u64(&mut r)?;
        let n_elements: u64 = dims.iter().product();
        let n_bytes = (n_elements as usize / qt.block_size * qt.block_bytes) as u64;

        let parts: Vec<&str> = name.split('.').collect();
        let (layer, kind) = if parts[0] == "blk" {
            (parts[1].parse::<i32>().unwrap_or(-1), parts[2..].join("."))
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

    let data_offset = r.stream_position().unwrap_or(0).div_ceil(32) * 32;
    Ok(Model {
        arch,
        metadata,
        tensors,
        data: mmap,
        data_offset,
    })
}
