//! Hugging Face checkpoints: a directory with `config.json`, `tokenizer.json`
//! (+ `tokenizer_config.json`) and `model.safetensors` or the shards named by
//! `model.safetensors.index.json`, loaded into a `gguf::Model` that the rest of
//! Axon can't tell from the GGUF `convert_hf_to_gguf.py --outtype f16` makes of
//! the same checkpoint (FP8 plan, stage A):
//!
//! - `config.json` becomes the GGUF metadata keys (`metadata`), the tokenizer
//!   files the `tokenizer.ggml.*` keys (`tokenizer_metadata`); the
//!   pre-tokenizer's regex is recognized, never executed.
//! - HF tensor names become GGUF names (`gguf_name`), shapes are reversed.
//! - BF16 matrices become F16 and every vector F32, as the converter makes
//!   them (bf16 -> f32 -> f16 round to nearest even), in one anonymous arena
//!   (`Model::aux`); F16 / F32 tensors already in that form and 32-byte
//!   aligned in their file are read in place from the mapped shard.
//!
//! The files are untrusted input: the header length, every offset and size
//! and every JSON shape are checked before anything is allocated from them.
//! Supported: qwen2, qwen3 and llama checkpoints with a byte-level BPE
//! tokenizer whose pre-tokenizer Axon knows; anything else is an error that
//! names what it is.

use crate::gguf::{AUX, Model, Shard, Source, Tensor, Val};
use serde_json::{Map, Value};
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Hugging Face's own limit on the JSON header (safetensors `MAX_HEADER_SIZE`).
const MAX_HEADER: u64 = 100_000_000;
/// The most shards an index may name.
const MAX_SHARDS: usize = 4096;
/// The largest vocabulary accepted (the biggest real ones are ~262k).
const MAX_VOCAB: u64 = 1 << 22;

/// A safetensors element type: its name in the header and its size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dtype {
    pub name: &'static str,
    /// Bits per element (F4: 4).
    pub bits: u64,
    /// The `gguf::QType` id tensors of this type get when they're left as
    /// they are (GGML's where there is one; Axon-private ids >= 1100 else).
    pub qtype: u32,
}

/// Axon-private tensor type ids (never in a GGUF; `gguf::qtype` names them).
pub const QT_F8_E4M3: u32 = 1100;
pub const QT_F8_E5M2: u32 = 1101;
pub const QT_F8_E8M0: u32 = 1102;
pub const QT_F4: u32 = 1103;
pub const QT_U8: u32 = 1104;
pub const QT_U16: u32 = 1105;
pub const QT_U32: u32 = 1106;
pub const QT_U64: u32 = 1107;
pub const QT_BOOL: u32 = 1108;

const DTYPES: &[Dtype] = &[
    Dtype {
        name: "F64",
        bits: 64,
        qtype: 28,
    },
    Dtype {
        name: "F32",
        bits: 32,
        qtype: 0,
    },
    Dtype {
        name: "F16",
        bits: 16,
        qtype: 1,
    },
    Dtype {
        name: "BF16",
        bits: 16,
        qtype: 30,
    },
    Dtype {
        name: "I64",
        bits: 64,
        qtype: 27,
    },
    Dtype {
        name: "I32",
        bits: 32,
        qtype: 26,
    },
    Dtype {
        name: "I16",
        bits: 16,
        qtype: 25,
    },
    Dtype {
        name: "I8",
        bits: 8,
        qtype: 24,
    },
    Dtype {
        name: "U64",
        bits: 64,
        qtype: QT_U64,
    },
    Dtype {
        name: "U32",
        bits: 32,
        qtype: QT_U32,
    },
    Dtype {
        name: "U16",
        bits: 16,
        qtype: QT_U16,
    },
    Dtype {
        name: "U8",
        bits: 8,
        qtype: QT_U8,
    },
    Dtype {
        name: "BOOL",
        bits: 8,
        qtype: QT_BOOL,
    },
    Dtype {
        name: "F8_E4M3",
        bits: 8,
        qtype: QT_F8_E4M3,
    },
    Dtype {
        name: "F8_E5M2",
        bits: 8,
        qtype: QT_F8_E5M2,
    },
    Dtype {
        name: "F8_E8M0",
        bits: 8,
        qtype: QT_F8_E8M0,
    },
    Dtype {
        name: "F4",
        bits: 4,
        qtype: QT_F4,
    },
];

/// One tensor of a shard's header, offsets absolute in the file.
#[derive(Clone, Debug)]
pub struct StTensor {
    pub name: String,
    pub dtype: Dtype,
    /// PyTorch order (`[out, in]`).
    pub shape: Vec<u64>,
    pub begin: u64,
    pub end: u64,
}

/// A shard's header: its tensors (in data order) and `__metadata__`.
pub struct Header {
    pub tensors: Vec<StTensor>,
    pub metadata: Vec<(String, String)>,
}

/// Parse a whole safetensors file `buf`: `[u64 LE header_len][JSON][data]`.
/// Every check is made before anything is sized from the file's numbers:
/// the header fits (and is at most 100 MB), it's a JSON object of
/// `{dtype, shape, data_offsets}` objects, each range lies inside the data
/// and holds exactly shape x dtype bytes. Overlapping ranges are legal and
/// accepted (nothing here writes through them).
pub fn parse_header(buf: &[u8]) -> io::Result<Header> {
    let Some(len) = buf.get(..8) else {
        return Err(bad(format!(
            "not a safetensors file ({} bytes, no header length)",
            buf.len()
        )));
    };
    let n = u64::from_le_bytes(len.try_into().unwrap());
    if n > MAX_HEADER {
        return Err(bad(format!(
            "safetensors header length {n} is over the {MAX_HEADER}-byte limit"
        )));
    }
    if n > buf.len() as u64 - 8 {
        return Err(bad(format!(
            "truncated safetensors file: the header needs {n} bytes, {} left",
            buf.len() - 8
        )));
    }
    let data_start = 8 + n;
    let data_len = buf.len() as u64 - data_start;
    let v: Value = serde_json::from_slice(&buf[8..data_start as usize])
        .map_err(|e| bad(format!("safetensors header is not valid JSON: {e}")))?;
    let Value::Object(obj) = v else {
        return Err(bad("safetensors header is not a JSON object"));
    };
    let mut tensors = Vec::with_capacity(obj.len());
    let mut metadata = Vec::new();
    for (name, v) in &obj {
        if name == "__metadata__" {
            let Value::Object(m) = v else {
                return Err(bad("safetensors __metadata__ is not an object"));
            };
            for (k, v) in m {
                let s = match v {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                metadata.push((k.clone(), s));
            }
            continue;
        }
        let e = |m: &str| bad(format!("safetensors tensor {name}: {m}"));
        let Value::Object(t) = v else {
            return Err(e("not an object"));
        };
        let dt = match t.get("dtype") {
            Some(Value::String(s)) => s,
            _ => return Err(e("no dtype string")),
        };
        let dtype = *DTYPES
            .iter()
            .find(|d| d.name == dt)
            .ok_or_else(|| e(&format!("unknown dtype {dt}")))?;
        let shape = match t.get("shape") {
            Some(Value::Array(a)) if a.len() <= 4 => a
                .iter()
                .map(|d| d.as_u64().ok_or_else(|| e("shape holds a non-integer")))
                .collect::<io::Result<Vec<u64>>>()?,
            Some(Value::Array(a)) => {
                return Err(e(&format!("{} dimensions (at most 4)", a.len())));
            }
            _ => return Err(e("no shape array")),
        };
        let (begin, end) = match t.get("data_offsets") {
            Some(Value::Array(a)) if a.len() == 2 => match (a[0].as_u64(), a[1].as_u64()) {
                (Some(b), Some(e)) => (b, e),
                _ => return Err(e("data_offsets are not integers")),
            },
            _ => return Err(e("no data_offsets pair")),
        };
        if begin > end || end > data_len {
            return Err(e(&format!(
                "data_offsets [{begin}, {end}] are outside the {data_len} data bytes; \
                 the file is truncated or corrupt"
            )));
        }
        let bytes = shape
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d))
            .and_then(|n| n.checked_mul(dtype.bits))
            .ok_or_else(|| e(&format!("element count overflows ({shape:?})")))?;
        if !bytes.is_multiple_of(8) || bytes / 8 != end - begin {
            return Err(e(&format!(
                "{} bytes for shape {shape:?} of {dt}, which needs {}",
                end - begin,
                bytes as f64 / 8.0
            )));
        }
        tensors.push(StTensor {
            name: name.clone(),
            dtype,
            shape,
            begin: data_start + begin,
            end: data_start + end,
        });
    }
    tensors.sort_by(|a, b| (a.begin, &a.name).cmp(&(b.begin, &b.name)));
    Ok(Header { tensors, metadata })
}

/// The checkpoint directory `path` names, if it names one: a directory, or
/// a `.safetensors` file in one. None: a GGUF (or nothing) as before.
pub fn dir_of(path: &str) -> Option<PathBuf> {
    let p = Path::new(path);
    if p.is_dir() {
        return Some(p.to_path_buf());
    }
    if p.extension().is_some_and(|e| e == "safetensors") && p.is_file() {
        return Some(match p.parent() {
            Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
            _ => PathBuf::from("."),
        });
    }
    None
}

fn read_json(path: &Path) -> io::Result<Value> {
    let b = std::fs::read(path)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    serde_json::from_slice(&b).map_err(|e| bad(format!("{}: not valid JSON: {e}", path.display())))
}

fn read_json_opt(path: &Path) -> io::Result<Option<Value>> {
    if path.is_file() {
        read_json(path).map(Some)
    } else {
        Ok(None)
    }
}

/// The shard files of `dir`, in order: the index's (sorted by name, i.e.
/// `model-00001-of-0000N` first) or `model.safetensors`; each with the
/// tensor names the index puts in it (None without an index).
fn shard_files(dir: &Path) -> io::Result<Vec<(PathBuf, Option<Vec<String>>)>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.is_file() {
        let one = dir.join("model.safetensors");
        if !one.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "{}: no model.safetensors or model.safetensors.index.json",
                    dir.display()
                ),
            ));
        }
        return Ok(vec![(one, None)]);
    }
    let v = read_json(&index)?;
    let Some(Value::Object(wm)) = v.get("weight_map") else {
        return Err(bad(format!("{}: no weight_map object", index.display())));
    };
    let mut files: Vec<(String, Vec<String>)> = Vec::new();
    for (name, f) in wm {
        let Value::String(f) = f else {
            return Err(bad(format!(
                "{}: weight_map entry {name} is not a file name",
                index.display()
            )));
        };
        // A plain file name in the directory, nothing else.
        if f.is_empty() || f.contains('/') || f.contains('\\') || f == "." || f == ".." {
            return Err(bad(format!(
                "{}: weight_map entry {name}: '{f}' is not a file name in the directory",
                index.display()
            )));
        }
        match files.iter_mut().find(|x| x.0 == *f) {
            Some(x) => x.1.push(name.clone()),
            None => files.push((f.clone(), vec![name.clone()])),
        }
    }
    if files.is_empty() || files.len() > MAX_SHARDS {
        return Err(bad(format!(
            "{}: weight_map names {} files (1 to {MAX_SHARDS})",
            index.display(),
            files.len()
        )));
    }
    files.sort();
    files
        .into_iter()
        .map(|(f, names)| {
            let p = dir.join(&f);
            if !p.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "shard {f} (tensor {} in model.safetensors.index.json) is missing from {}",
                        names[0],
                        dir.display()
                    ),
                ));
            }
            Ok((p, Some(names)))
        })
        .collect()
}

/// Every file of the checkpoint in `dir` that the model depends on, in a
/// fixed order: config.json, the tokenizer files, the index (if any) and
/// the shards. What `identity::model_digest` and `kvsave::model_id` hash.
pub fn files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut v = vec![dir.join("config.json"), dir.join("tokenizer.json")];
    for f in [
        "tokenizer_config.json",
        "chat_template.jinja",
        "chat_template.json",
        "model.safetensors.index.json",
    ] {
        let p = dir.join(f);
        if p.is_file() {
            v.push(p);
        }
    }
    v.extend(shard_files(dir)?.into_iter().map(|s| s.0));
    Ok(v)
}

// ---------------------------------------------------------------------------
// config.json -> GGUF metadata

/// The GGUF architecture of a `model_type` (or `architectures[0]`), or why not.
fn arch_of(cfg: &Value) -> Result<&'static str, String> {
    let mt = cfg.get("model_type").and_then(Value::as_str).unwrap_or("");
    let hf = (cfg.get("architectures").and_then(|a| a.get(0)))
        .and_then(Value::as_str)
        .unwrap_or("");
    Ok(match (mt, hf) {
        ("qwen2", _) | ("", "Qwen2ForCausalLM") => "qwen2",
        ("qwen3", _) | ("", "Qwen3ForCausalLM") => "qwen3",
        ("llama" | "mistral", _) | ("", "LlamaForCausalLM" | "MistralForCausalLM") => "llama",
        (
            "qwen3_moe" | "deepseek_v2" | "deepseek_v3" | "kimi_k2" | "gemma3" | "gemma3_text"
            | "gpt_oss",
            _,
        ) => {
            return Err(format!(
                "model_type {mt} is not supported in safetensors form yet (stage A loads \
                 qwen2, qwen3 and llama checkpoints); use its GGUF"
            ));
        }
        _ => {
            return Err(format!(
                "model_type '{}' is not supported (qwen2, qwen3, llama)",
                if mt.is_empty() { hf } else { mt }
            ));
        }
    })
}

/// A config.json number that must be a whole u32.
fn cfg_u32(cfg: &Value, k: &str) -> Result<Option<u32>, String> {
    match cfg.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) if n <= u32::MAX as u64 => Ok(Some(n as u32)),
            _ => Err(format!("config.json {k} = {v} is not a whole number")),
        },
    }
}

fn cfg_f64(cfg: &Value, k: &str) -> Result<Option<f64>, String> {
    match cfg.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("config.json {k} = {v} is not a number")),
    }
}

/// The RoPE parameters: transformers >= 5's `rope_parameters`, else the
/// top-level `rope_theta` and `rope_scaling`, as one object.
fn rope_params(cfg: &Value) -> Map<String, Value> {
    let mut p = Map::new();
    if let Some(Value::Object(s)) = cfg.get("rope_scaling") {
        p.extend(s.clone());
    }
    if let Some(t) = cfg.get("rope_theta") {
        p.insert("rope_theta".into(), t.clone());
    }
    if let Some(Value::Object(r)) = cfg.get("rope_parameters") {
        let r = match r.get("full_attention") {
            Some(Value::Object(f)) => f,
            _ => r,
        };
        p.extend(r.clone());
    }
    p
}

/// What `config.json` (`cfg`) says, as the GGUF keys convert_hf_to_gguf.py
/// writes for it (the ones Axon reads: model.rs, rope.rs), plus the
/// `rope_freqs.weight` values a Llama 3 RoPE needs. Optional keys missing
/// from the config are missing here too, so `check_supported` sees what it
/// would see in the GGUF. `name`: general.name (the directory's).
pub fn metadata(cfg: &Value, name: &str) -> Result<Meta, String> {
    if !cfg.is_object() {
        return Err("config.json is not a JSON object".into());
    }
    let a = arch_of(cfg)?;
    if let Some(q) = cfg.get("quantization_config").filter(|q| !q.is_null()) {
        let m = q.get("quant_method").and_then(Value::as_str).unwrap_or("?");
        return Err(format!(
            "quantized checkpoint (quantization_config.quant_method {m}): not supported yet \
             (FP8 lands in stages B-C of the FP8 plan); use a GGUF"
        ));
    }
    if let Some(act) = cfg.get("hidden_act").and_then(Value::as_str)
        && act != "silu"
    {
        return Err(format!("hidden_act {act} is not supported (silu)"));
    }
    if cfg.get("use_sliding_window") == Some(&Value::Bool(true)) {
        return Err("use_sliding_window true is not supported".into());
    }
    let mut m: Vec<(String, Val)> = vec![
        ("general.architecture".into(), Val::Str(a.into())),
        ("general.name".into(), Val::Str(name.into())),
    ];
    let k = |s: &str| format!("{a}.{s}");
    let u32s = [
        ("block_count", "num_hidden_layers"),
        ("context_length", "max_position_embeddings"),
        ("embedding_length", "hidden_size"),
        ("feed_forward_length", "intermediate_size"),
        ("attention.head_count", "num_attention_heads"),
        ("attention.head_count_kv", "num_key_value_heads"),
    ];
    for (g, h) in u32s {
        if let Some(v) = cfg_u32(cfg, h)? {
            m.push((k(g), Val::U32(v)));
        }
    }
    let rp = rope_params(cfg);
    let rp = Value::Object(rp);
    if let Some(t) = cfg_f64(&rp, "rope_theta")? {
        m.push((k("rope.freq_base"), Val::F32(t as f32)));
    }
    if let Some(e) = cfg_f64(cfg, "rms_norm_eps")? {
        m.push((k("attention.layer_norm_rms_epsilon"), Val::F32(e as f32)));
    }
    let head_dim = cfg_u32(cfg, "head_dim")?;
    if let Some(h) = head_dim {
        m.push((k("attention.key_length"), Val::U32(h)));
        m.push((k("attention.value_length"), Val::U32(h)));
    }
    let rope_dim = match head_dim {
        Some(h) => Some(h),
        None => match (
            cfg_u32(cfg, "hidden_size")?,
            cfg_u32(cfg, "num_attention_heads")?,
        ) {
            (Some(d), Some(n)) if n > 0 => Some(d / n),
            _ => None,
        },
    };
    if a == "llama" {
        if let Some(v) = cfg_u32(cfg, "vocab_size")? {
            m.push((k("vocab_size"), Val::U32(v)));
        }
        if let Some(r) = rope_dim {
            m.push((k("rope.dimension_count"), Val::U32(r)));
        }
    }
    // RoPE scaling (convert_hf_to_gguf.py's TextModel.set_gguf_parameters).
    let ty = (rp.get("rope_type").or(rp.get("type")))
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase);
    let factor = cfg_f64(&rp, "factor")?;
    let mut freqs = None;
    match (ty.as_deref(), factor) {
        (None | Some("default"), _) => {}
        (Some("linear"), Some(f)) => {
            m.push((k("rope.scaling.type"), Val::Str("linear".into())));
            m.push((k("rope.scaling.factor"), Val::F32(f as f32)));
        }
        (Some("yarn"), Some(f)) => {
            m.push((k("rope.scaling.type"), Val::Str("yarn".into())));
            m.push((k("rope.scaling.factor"), Val::F32(f as f32)));
            let orig = cfg_u32(&rp, "original_max_position_embeddings")?
                .ok_or("RoPE scaling yarn without original_max_position_embeddings")?;
            m.push((k("rope.scaling.original_context_length"), Val::U32(orig)));
            for (g, h) in [
                ("rope.scaling.yarn_ext_factor", "extrapolation_factor"),
                ("rope.scaling.yarn_attn_factor", "attention_factor"),
                ("rope.scaling.yarn_beta_fast", "beta_fast"),
                ("rope.scaling.yarn_beta_slow", "beta_slow"),
            ] {
                let v = match (cfg_f64(&rp, h)?, h) {
                    (None, "attention_factor") => cfg_f64(&rp, "attn_factor")?,
                    (v, _) => v,
                };
                if let Some(v) = v {
                    m.push((k(g), Val::F32(v as f32)));
                }
            }
        }
        (Some("llama3"), _) => {
            let dim = rope_dim.ok_or("RoPE scaling llama3 without a head size")?;
            freqs = Some(llama3_rope_factors(&rp, dim as usize)?);
        }
        (Some(t), _) => {
            return Err(format!("RoPE scaling type {t} is not supported"));
        }
    }
    Ok(Meta {
        arch: a.to_string(),
        keys: m,
        rope_freqs: freqs,
    })
}

/// `metadata`'s answer.
pub struct Meta {
    pub arch: String,
    pub keys: Vec<(String, Val)>,
    /// `rope_freqs.weight` (Llama 3 RoPE scaling), if the model has one.
    pub rope_freqs: Option<Vec<f32>>,
}

/// Llama 3's per-frequency RoPE divisors (`rope_freqs.weight`), computed
/// as convert_hf_to_gguf.py's LlamaModel.generate_extra_tensors does it:
/// in f32, as torch does (the python scalars are rounded to f32 at each
/// operation with a tensor).
fn llama3_rope_factors(rp: &Value, dim: usize) -> Result<Vec<f32>, String> {
    if dim == 0 || !dim.is_multiple_of(2) || dim > 4096 {
        return Err(format!("RoPE scaling llama3: bad head size {dim}"));
    }
    let base = cfg_f64(rp, "rope_theta")?.unwrap_or(10000.0) as f32;
    let factor = cfg_f64(rp, "factor")?.unwrap_or(8.0);
    let low = cfg_f64(rp, "low_freq_factor")?.unwrap_or(1.0);
    let high = cfg_f64(rp, "high_freq_factor")?.unwrap_or(4.0);
    let old = cfg_f64(rp, "original_max_position_embeddings")?.unwrap_or(8192.0);
    let low_wl = (old / low) as f32;
    let high_wl = (old / high) as f32;
    let out = (0..dim / 2)
        .map(|i| {
            let e = (2 * i) as f32 / dim as f32;
            let freq = 1.0f32 / base.powf(e);
            let wavelen = (2.0 * std::f64::consts::PI) as f32 / freq;
            if wavelen < high_wl {
                1.0
            } else if wavelen > low_wl {
                factor as f32
            } else {
                let smooth = (old as f32 / wavelen - low as f32) / (high - low) as f32;
                1.0 / ((1.0 - smooth) / factor as f32 + smooth)
            }
        })
        .collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// tokenizer.json -> tokenizer.ggml.*

/// The pre-tokenizer regexes Axon knows, byte for byte as tokenizer.json
/// holds them, and the GGUF `tokenizer.ggml.pre` each one is (the converter
/// identifies them by a token hash; same answer for these). The regex is
/// never run: Axon's own expanded pattern for that name is (tokenizer.rs).
const PRE_PATTERNS: &[(&str, &str, bool)] = &[
    // Qwen2 / Qwen2.5 / Qwen3: digits one at a time.
    (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        "qwen2",
        false,
    ),
    // Llama 3: digits in threes; merges ignored for whole-vocab words.
    (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        "llama-bpe",
        true,
    ),
];

/// The tokenizer files of a checkpoint.
pub struct TokFiles {
    pub tokenizer: Value,
    pub config: Option<Value>,
    /// chat_template.jinja (or chat_template.json's), used when
    /// tokenizer_config.json has no chat_template.
    pub chat_template: Option<Value>,
}

/// An added token's text by content lookup (`SpecialVocab`'s first match).
fn added_id(added: &[Value], content: &str) -> Option<u64> {
    added
        .iter()
        .find(|a| a.get("content").and_then(Value::as_str) == Some(content))
        .and_then(|a| a.get("id").and_then(Value::as_u64))
}

/// A tokenizer_config `*_token` entry: a string or `{content}`.
fn token_text(v: Option<&Value>) -> Option<&str> {
    match v? {
        Value::String(s) => Some(s),
        Value::Object(o) => o.get("content").and_then(Value::as_str),
        _ => None,
    }
}

/// Converter's `does_token_look_special`.
fn looks_special(t: &str) -> bool {
    ["<pad>", "<mask>", "<2mass>", "[@BOS@]"].contains(&t)
        || (t.starts_with("<|") && t.ends_with("|>"))
        || (t.starts_with("<\u{ff5c}") && t.ends_with("\u{ff5c}>"))
        || (t.starts_with("<unused") && t.ends_with('>'))
        || (t.starts_with("<start_of_") && t.ends_with('>'))
        || (t.starts_with("<end_of_") && t.ends_with('>'))
}

/// The `tokenizer.ggml.*` keys (and `tokenizer.chat_template`) the converter
/// writes for a byte-level BPE `tokenizer.json`, with `vocab_size` from
/// config.json (`cfg`): every id below it has a token ("[PAD<i>]",
/// UNUSED, where the vocabulary has none). Refused: SentencePiece-style
/// BPE (byte fallback, Metaspace: the GGUF carries scores this lacks), a
/// normalizer other than NFC (which, as llama.cpp, is not applied), and any
/// pre-tokenizer not in `PRE_PATTERNS`.
pub fn tokenizer_metadata(tf: &TokFiles, cfg: &Value) -> Result<Vec<(String, Val)>, String> {
    let tok = &tf.tokenizer;
    let model = tok.get("model").ok_or("tokenizer.json has no model")?;
    let mtype = model.get("type").and_then(Value::as_str).unwrap_or("?");
    if mtype != "BPE" {
        return Err(format!(
            "tokenizer.json model type {mtype} is not supported (BPE)"
        ));
    }
    let dec = tok.get("decoder");
    let dec_ty = dec.and_then(|d| d.get("type")).and_then(Value::as_str);
    if model.get("byte_fallback") == Some(&Value::Bool(true)) || dec_ty != Some("ByteLevel") {
        return Err(
            "this tokenizer is SentencePiece-style BPE (byte fallback / no ByteLevel decoder): \
             it needs tokenizer.model's scores; use the GGUF"
                .into(),
        );
    }
    match tok.get("normalizer") {
        None | Some(Value::Null) => {}
        Some(n) if n.get("type").and_then(Value::as_str) == Some("NFC") => {}
        Some(n) => {
            return Err(format!(
                "tokenizer normalizer {} is not supported (none, NFC)",
                n.get("type").and_then(Value::as_str).unwrap_or("?")
            ));
        }
    }
    let (pre, ignore_merges) = recognize_pre(tok.get("pre_tokenizer"))?;
    if model
        .get("ignore_merges")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        != ignore_merges
    {
        return Err(format!(
            "tokenizer.json ignore_merges disagrees with the {pre} pre-tokenizer"
        ));
    }

    // Tokens: the vocabulary, then the added tokens (by id).
    let Some(Value::Object(vocab)) = model.get("vocab") else {
        return Err("tokenizer.json model.vocab is not an object".into());
    };
    let added: &[Value] = match tok.get("added_tokens") {
        Some(Value::Array(a)) => a,
        _ => &[],
    };
    let mut max_id = 0u64;
    let mut ids: Vec<(u64, &str, Option<bool>)> = Vec::with_capacity(vocab.len() + added.len());
    for (t, id) in vocab {
        let id = id
            .as_u64()
            .ok_or_else(|| format!("tokenizer vocab {t}: bad id"))?;
        ids.push((id, t, None));
    }
    for a in added {
        let (Some(id), Some(t)) = (
            a.get("id").and_then(Value::as_u64),
            a.get("content").and_then(Value::as_str),
        ) else {
            return Err("tokenizer added_tokens entry without id/content".into());
        };
        let special = a.get("special").and_then(Value::as_bool).unwrap_or(false);
        ids.push((id, t, Some(special)));
    }
    for &(id, _, _) in &ids {
        max_id = max_id.max(id);
    }
    let n = match cfg_u32(cfg, "vocab_size")? {
        Some(v) => v as u64,
        None => max_id + 1,
    };
    if ids.is_empty() || max_id >= n || n > MAX_VOCAB {
        return Err(format!(
            "tokenizer: ids up to {max_id} for a vocabulary of {n} (at most {MAX_VOCAB})"
        ));
    }
    // The vocabulary first, then the added tokens over it (an added token
    // is typed as one even where the vocabulary has the same text).
    let mut tokens: Vec<Option<(String, i32)>> = vec![None; n as usize];
    for (id, t, special) in ids {
        let e = &mut tokens[id as usize];
        let ty = match special {
            None => 1,
            Some(true) => 3,
            Some(false) if looks_special(t) => 3,
            Some(false) => 4,
        };
        if special.is_none() && e.is_some() {
            return Err(format!("tokenizer: two vocabulary entries with id {id}"));
        }
        let text = match ty {
            4 => t.replace('\u{2581}', " "),
            _ => t.to_string(),
        };
        *e = Some((text, ty));
    }
    let (texts, types): (Vec<Val>, Vec<Val>) = (tokens.into_iter().enumerate())
        .map(|(i, t)| match t {
            Some((s, ty)) => (Val::Str(s), Val::I32(ty)),
            None => (Val::Str(format!("[PAD{i}]")), Val::I32(5)),
        })
        .unzip();

    let merges: Vec<Val> = match model.get("merges") {
        Some(Value::Array(a)) => a
            .iter()
            .map(|m| match m {
                Value::String(s) => Ok(Val::Str(s.clone())),
                Value::Array(p) if p.len() == 2 => match (p[0].as_str(), p[1].as_str()) {
                    (Some(l), Some(r)) => {
                        let enc = |s: &str| s.replace(' ', "\u{120}");
                        Ok(Val::Str(format!("{} {}", enc(l), enc(r))))
                    }
                    _ => Err("tokenizer merge pair is not two strings".to_string()),
                },
                _ => Err("tokenizer merges entry is not a string or a pair".to_string()),
            })
            .collect::<Result<_, _>>()?,
        _ => Vec::new(),
    };

    let mut out: Vec<(String, Val)> = vec![
        ("tokenizer.ggml.model".into(), Val::Str("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Val::Str(pre.into())),
        ("tokenizer.ggml.tokens".into(), Val::Arr(texts)),
        ("tokenizer.ggml.token_type".into(), Val::Arr(types)),
        ("tokenizer.ggml.merges".into(), Val::Arr(merges)),
    ];
    out.extend(special_tokens(tf, cfg, added, n)?);
    out.extend(chat_templates(tf)?);
    Ok(out)
}

/// The pre-tokenizer's GGUF name and whether its BPE ignores merges, or
/// a refusal that shows the pattern.
fn recognize_pre(p: Option<&Value>) -> Result<(&'static str, bool), String> {
    let seq = match p {
        Some(v) if v.get("type").and_then(Value::as_str) == Some("Sequence") => {
            v.get("pretokenizers").and_then(Value::as_array)
        }
        _ => None,
    };
    let unsupported = |what: String| format!("pre-tokenizer {what} is not supported");
    let Some([split, bl]) = seq.map(Vec::as_slice) else {
        return Err(unsupported(format!(
            "{} (a Split regex then ByteLevel)",
            p.map_or("<none>".to_string(), |v| v
                .to_string()
                .chars()
                .take(200)
                .collect())
        )));
    };
    let ok_bl = bl.get("type").and_then(Value::as_str) == Some("ByteLevel")
        && bl.get("use_regex") == Some(&Value::Bool(false))
        && bl.get("add_prefix_space") != Some(&Value::Bool(true));
    let re = (split.get("type").and_then(Value::as_str) == Some("Split"))
        .then(|| split.get("pattern")?.get("Regex")?.as_str())
        .flatten();
    let ok_split = split.get("behavior").and_then(Value::as_str) == Some("Isolated")
        && split.get("invert") != Some(&Value::Bool(true));
    let (Some(re), true, true) = (re, ok_bl, ok_split) else {
        return Err(unsupported(
            "sequence (Axon knows a Split{Regex, Isolated} then ByteLevel{use_regex: false})"
                .into(),
        ));
    };
    match PRE_PATTERNS.iter().find(|p| p.0 == re) {
        Some(&(_, name, ig)) => Ok((name, ig)),
        None => Err(unsupported(format!(
            "regex {re:?} (known: Qwen2's, Llama 3's)"
        ))),
    }
}

/// A TemplateProcessing entry's special token id.
fn sp(v: Option<&Value>) -> Option<&str> {
    v.and_then(|x| x.get("SpecialToken"))
        .and_then(|s| s.get("id"))
        .and_then(Value::as_str)
}

/// bos / eos / unk / pad ids and add_bos, as gguf-py's SpecialVocab finds
/// them: tokenizer_config.json's `*_token` strings looked up in the added
/// tokens, then config.json's `*_token_id` integers; `add_bos_token` from
/// tokenizer.json's post-processor, else tokenizer_config.json.
fn special_tokens(
    tf: &TokFiles,
    cfg: &Value,
    added: &[Value],
    n: u64,
) -> Result<Vec<(String, Val)>, String> {
    let tc = tf.config.as_ref();
    let get = |k: &str| {
        tc.and_then(|c| token_text(c.get(k)))
            .filter(|s| !s.is_empty())
    };
    let cls = get("cls_token");
    let mut bos = get("bos_token").or(cls);
    let mut eos = get("eos_token").or(get("sep_token"));
    let mut add_bos: Option<bool> = None;
    if let Some(pp) = tf.tokenizer.get("post_processor").filter(|p| !p.is_null()) {
        let procs: Vec<&Value> = match pp.get("processors").and_then(Value::as_array) {
            Some(a) => a.iter().collect(),
            None => vec![pp],
        };
        for p in procs {
            match p.get("type").and_then(Value::as_str) {
                Some("ByteLevel") => {
                    add_bos.get_or_insert(false);
                }
                Some("TemplateProcessing") => {
                    let single = p.get("single").and_then(Value::as_array);
                    let Some(s) = single.filter(|s| s.len() > 1) else {
                        continue;
                    };
                    if let Some(first) = sp(s.first()) {
                        if tc.is_none() || bos.is_none() {
                            bos = Some(first);
                        }
                        add_bos = Some(Some(first) == bos || Some(first) == cls);
                    }
                    if let Some(last) = sp(s.last()) {
                        if tc.is_none() {
                            eos = Some(last);
                        } else if Some(last) != eos {
                            return Err(format!(
                                "tokenizer post-processor appends {last}: not supported"
                            ));
                        }
                    }
                }
                Some(t) => return Err(format!("tokenizer post-processor {t} is not supported")),
                None => {}
            }
        }
    }
    if add_bos.is_none() {
        add_bos = tc
            .and_then(|c| c.get("add_bos_token"))
            .and_then(Value::as_bool);
    }
    let mut out = Vec::new();
    for (typ, text, key) in [
        ("bos", bos, "bos_token_id"),
        ("eos", eos, "eos_token_id"),
        ("unk", get("unk_token"), "unknown_token_id"),
        ("pad", get("pad_token"), "padding_token_id"),
    ] {
        let id = text
            .and_then(|t| added_id(added, t))
            .or_else(|| cfg.get(format!("{typ}_token_id")).and_then(Value::as_u64))
            .filter(|&id| id < n);
        if let Some(id) = id {
            out.push((format!("tokenizer.ggml.{key}"), Val::U32(id as u32)));
        }
    }
    if let Some(b) = add_bos {
        out.push(("tokenizer.ggml.add_bos_token".into(), Val::Bool(b)));
    }
    Ok(out)
}

/// `tokenizer.chat_template` (and named `tokenizer.chat_template.<name>`
/// ones for a list of templates).
fn chat_templates(tf: &TokFiles) -> Result<Vec<(String, Val)>, String> {
    let ct = tf
        .config
        .as_ref()
        .and_then(|c| c.get("chat_template"))
        .or(tf.chat_template.as_ref());
    Ok(match ct {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![("tokenizer.chat_template".into(), Val::Str(s.clone()))],
        Some(Value::Array(a)) => {
            let mut v = Vec::new();
            for t in a {
                let (Some(name), Some(tpl)) = (
                    t.get("name").and_then(Value::as_str),
                    t.get("template").and_then(Value::as_str),
                ) else {
                    return Err("chat_template list entry without name/template".into());
                };
                let key = match name {
                    "default" => "tokenizer.chat_template".to_string(),
                    n => format!("tokenizer.chat_template.{n}"),
                };
                v.push((key, Val::Str(tpl.into())));
            }
            v
        }
        Some(_) => return Err("chat_template is not a string or a list".into()),
    })
}

// ---------------------------------------------------------------------------
// tensors

/// The GGUF name of a qwen2 / qwen3 / llama HF tensor, None for one Axon
/// doesn't know (refused) and Some("") for one it skips.
pub fn gguf_name(hf: &str) -> Option<String> {
    match hf {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        _ => {}
    }
    let rest = hf.strip_prefix("model.layers.")?;
    let (n, rest) = rest.split_once('.')?;
    let n: u32 = n.parse().ok().filter(|_| !n.starts_with('+'))?;
    if rest.ends_with("rotary_emb.inv_freq") {
        return Some(String::new());
    }
    let g = match rest {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{n}.{g}"))
}

/// The type a tensor becomes, as `convert_hf_to_gguf.py --outtype f16`
/// makes it: vectors, norms and routers F32, other matrices F16 (F32 ones
/// stay F32 here: lossless, and the CPU runs them). None: left as it is.
fn target(gguf: &str, t: &StTensor) -> Option<u32> {
    use crate::quant::{F16, F32};
    let vector =
        t.shape.len() <= 1 || gguf.ends_with("_norm.weight") || gguf.contains(".ffn_gate_inp.");
    match t.dtype.name {
        "BF16" | "F16" if vector => Some(F32),
        "BF16" | "F16" => Some(F16),
        "F32" => Some(F32),
        _ => None,
    }
}

/// BF16 -> F16 for every BF16 bit pattern, as the converter makes it:
/// torch's bf16 -> f32 (exact), then numpy's f32 -> f16 (`astype`): round
/// to nearest even, overflow to infinity (`quant::f32_to_f16`), except
/// that a NaN keeps its top ten payload bits unquieted (numpy's
/// `npy_floatbits_to_halfbits`; a BF16 NaN always has one of them set).
/// Checked against numpy for all 65536 patterns (`bf16_table_matches_numpy`).
fn bf16_to_f16(h: u16) -> u16 {
    if h & 0x7f80 == 0x7f80 && h & 0x7f != 0 {
        return (h & 0x8000) | 0x7c00 | ((h & 0x7f) << 3);
    }
    crate::quant::f32_to_f16(f32::from_bits((h as u32) << 16))
}

fn bf16_to_f16_table() -> Vec<u16> {
    (0..=u16::MAX).map(bf16_to_f16).collect()
}

/// One conversion: `n` elements of `from` at `src` in file `file`, into
/// `dst` of `aux` as `to`.
struct Job {
    file: usize,
    src: u64,
    n: usize,
    from: &'static str,
    to: u32,
    dst: usize,
}

/// Elements per conversion piece (a few MB: the threads share big tensors).
const PIECE: usize = 1 << 21;

/// Convert `job` from `bytes` (its source) into `out`.
fn convert(job: &Job, bytes: &[u8], out: &mut [u8], tab: &[u16]) {
    use crate::quant::F16;
    let u16s = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]));
    match (job.from, job.to == F16) {
        ("BF16", true) => {
            for (o, h) in out.chunks_exact_mut(2).zip(u16s) {
                o.copy_from_slice(&tab[h as usize].to_le_bytes());
            }
        }
        ("BF16", false) => {
            for (o, h) in out.chunks_exact_mut(4).zip(u16s) {
                o.copy_from_slice(&((h as u32) << 16).to_le_bytes());
            }
        }
        ("F16", false) => {
            for (o, h) in out.chunks_exact_mut(4).zip(u16s) {
                o.copy_from_slice(&crate::quant::f16_to_f32(h).to_le_bytes());
            }
        }
        // Same type: a copy (the source was misaligned).
        _ => out.copy_from_slice(bytes),
    }
}

/// Run the conversions on every core (`files[j.file]` read with pread),
/// filling `aux`.
fn run_jobs(files: &[&File], jobs: &[Job], aux: &mut [u8]) -> io::Result<()> {
    // Pieces in dst order, then split into disjoint slices of aux.
    let mut pieces: Vec<(Job, usize)> = Vec::new();
    for j in jobs {
        let (sb, db) = (bytes_per(j.from), bytes_per_q(j.to));
        let mut at = 0;
        while at < j.n {
            let k = PIECE.min(j.n - at);
            pieces.push((
                Job {
                    file: j.file,
                    src: j.src + (at * sb) as u64,
                    n: k,
                    from: j.from,
                    to: j.to,
                    dst: j.dst + at * db,
                },
                k * db,
            ));
            at += k;
        }
    }
    pieces.sort_by_key(|p| p.0.dst);
    let mut work: Vec<(Job, &mut [u8])> = Vec::with_capacity(pieces.len());
    let mut rest = aux;
    let mut at = 0;
    for (j, len) in pieces {
        let (_, r) = std::mem::take(&mut rest).split_at_mut(j.dst - at);
        let (d, r) = r.split_at_mut(len);
        rest = r;
        at = j.dst + len;
        work.push((j, d));
    }
    // Biggest first is moot (pieces are ~equal); in order keeps reads sequential.
    work.reverse();
    let tab = bf16_to_f16_table();
    let work = std::sync::Mutex::new(work);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| -> io::Result<()> {
                    let mut buf = Vec::new();
                    loop {
                        let Some((j, out)) = work.lock().unwrap().pop() else {
                            return Ok(());
                        };
                        buf.resize(j.n * bytes_per(j.from), 0);
                        files[j.file].read_exact_at(&mut buf, j.src)?;
                        convert(&j, &buf, out, &tab);
                    }
                })
            })
            .collect();
        hs.into_iter()
            .try_for_each(|h| h.join().expect("safetensors conversion thread"))
    })
}

fn bytes_per(dtype: &str) -> usize {
    match dtype {
        "F32" => 4,
        _ => 2,
    }
}

fn bytes_per_q(qt: u32) -> usize {
    match qt {
        crate::quant::F16 => 2,
        _ => 4,
    }
}

/// The checkpoint's config.json and what it and the tokenizer files say
/// as GGUF metadata (`metadata` + `tokenizer_metadata`): everything but
/// the tensors.
pub fn checkpoint_metadata(dir: &Path) -> io::Result<(Value, Meta)> {
    let cfg = read_json(&dir.join("config.json"))?;
    let name = dir
        .canonicalize()
        .ok()
        .and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
    if dir.join("hf_quant_config.json").is_file() {
        return Err(bad(
            "quantized checkpoint (hf_quant_config.json: NVIDIA ModelOpt): not supported yet \
             (FP8 / NVFP4 land in stages B-D of the FP8 plan); use a GGUF",
        ));
    }
    let mut meta = metadata(&cfg, &name).map_err(bad)?;
    let tf = TokFiles {
        tokenizer: read_json(&dir.join("tokenizer.json"))?,
        config: read_json_opt(&dir.join("tokenizer_config.json"))?,
        chat_template: match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(s) => Some(Value::String(s)),
            Err(_) => read_json_opt(&dir.join("chat_template.json"))?
                .and_then(|v| v.get("chat_template").cloned()),
        },
    };
    meta.keys
        .extend(tokenizer_metadata(&tf, &cfg).map_err(bad)?);
    Ok((cfg, meta))
}

/// Load the checkpoint in `dir`. `check` sees the model once its tensor
/// table and metadata are built and before any weight is read (`main`'s
/// `check_supported`), as `gguf::load_to_run`'s `wants_copy`; the shards
/// are always mapped (the converted tensors live in the anonymous `aux`
/// arena), so its answer is not used. `fill` false: the tensors that need
/// converting aren't (their bytes can't be read; `axon info`).
pub fn load(
    dir: &Path,
    check: impl FnOnce(&Model) -> crate::gguf::CopyWish,
    fill: bool,
) -> io::Result<Model> {
    let (cfg, meta) = checkpoint_metadata(dir)?;
    let Meta {
        arch,
        keys: mut metadata,
        rope_freqs,
    } = meta;
    // Shards: mapped, headers parsed, tensors named.
    let shard_list = shard_files(dir)?;
    let mut maps: Vec<Shard> = Vec::with_capacity(shard_list.len());
    let mut tensors: Vec<Tensor> = Vec::new();
    let mut jobs: Vec<Job> = Vec::new();
    let mut aux_len = 0usize;
    let mut seen = std::collections::HashSet::new();
    let mut st_meta: Vec<(String, Val)> = Vec::new();
    for (si, (path, names)) in shard_list.iter().enumerate() {
        let file = File::open(path)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        // SAFETY: opened read-only; not modified while mapped (as a GGUF).
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let fname = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let h = parse_header(&map).map_err(|e| bad(format!("{fname}: {e}")))?;
        if let Some(names) = names {
            for n in names {
                if !h.tensors.iter().any(|t| t.name == *n) {
                    return Err(bad(format!(
                        "tensor {n}: model.safetensors.index.json puts it in {fname}, which \
                         doesn't hold it"
                    )));
                }
            }
        }
        for (k, v) in h.metadata {
            if !st_meta
                .iter()
                .any(|(x, _)| *x == format!("safetensors.{k}"))
            {
                st_meta.push((format!("safetensors.{k}"), Val::Str(v)));
            }
        }
        for t in h.tensors {
            if !seen.insert(t.name.clone()) {
                return Err(bad(format!("tensor {} is in two shards", t.name)));
            }
            let g = gguf_name(&t.name).ok_or_else(|| {
                bad(format!(
                    "tensor {} ({fname}) is not a {arch} tensor Axon knows",
                    t.name
                ))
            })?;
            if g.is_empty() {
                continue;
            }
            let dims: Vec<u64> = t.shape.iter().rev().copied().collect();
            let n_elements: u64 = dims.iter().product();
            let to = target(&g, &t);
            let (qtype, shard, offset, n_bytes) = match to {
                Some(q) if q == t.dtype.qtype && t.begin.is_multiple_of(32) => {
                    (q, si as u16, t.begin, t.end - t.begin)
                }
                Some(q) => {
                    let n_bytes = n_elements * bytes_per_q(q) as u64;
                    let dst = aux_len;
                    aux_len = (aux_len + n_bytes as usize).next_multiple_of(64);
                    jobs.push(Job {
                        file: si,
                        src: t.begin,
                        n: n_elements as usize,
                        from: t.dtype.name,
                        to: q,
                        dst,
                    });
                    (q, AUX, dst as u64, n_bytes)
                }
                None => (t.dtype.qtype, si as u16, t.begin, t.end - t.begin),
            };
            let (layer, kind) = match g.strip_prefix("blk.").and_then(|r| r.split_once('.')) {
                Some((l, k)) => (l.parse().unwrap_or(-1), k.to_string()),
                None => (-1, g.clone()),
            };
            tensors.push(Tensor {
                name: g,
                dims,
                qtype: crate::gguf::qtype(qtype),
                offset,
                n_elements,
                n_bytes,
                layer,
                kind,
                shard,
                hf: Some(crate::gguf::HfTensor {
                    name: t.name,
                    dtype: t.dtype.name,
                }),
            });
        }
        maps.push(Shard { map, file });
    }
    // transformers' default for these models is untied.
    let tied = (cfg.get("tie_word_embeddings").and_then(Value::as_bool)).unwrap_or(false);
    if !tied && !tensors.iter().any(|t| t.name == "output.weight") {
        return Err(bad(
            "lm_head.weight is missing and tie_word_embeddings is false",
        ));
    }
    let mut freq_vals = None;
    if let Some(f) = rope_freqs {
        let n_bytes = f.len() as u64 * 4;
        let dst = aux_len;
        aux_len = (aux_len + n_bytes as usize).next_multiple_of(64);
        tensors.push(Tensor {
            name: "rope_freqs.weight".into(),
            dims: vec![f.len() as u64],
            qtype: crate::gguf::qtype(crate::quant::F32),
            offset: dst as u64,
            n_elements: f.len() as u64,
            n_bytes,
            layer: -1,
            kind: "rope_freqs.weight".into(),
            shard: AUX,
            hf: None,
        });
        freq_vals = Some((dst, f));
    }
    metadata.extend(st_meta);
    // Tensors in GGUF order: globals, then by layer (stable within).
    tensors.sort_by_key(|t| t.layer);

    let source = Source::Safetensors {
        dir: dir.to_path_buf(),
        files: files(dir)?,
    };
    let mut maps = maps.into_iter();
    let first = maps.next().expect("at least one shard");
    let mut m = Model::from_safetensors(arch, metadata, tensors, first, maps.collect(), source);
    let _ = check(&m);
    if aux_len > 0 && fill {
        let mut aux = memmap2::MmapMut::map_anon(aux_len.next_multiple_of(2 << 20))?;
        #[cfg(target_os = "linux")]
        {
            // Before the first touch (as gguf::copy_file). Failure only costs speed.
            let _ = aux.advise(memmap2::Advice::HugePage);
        }
        let files: Vec<&File> = m.shard_files();
        run_jobs(&files, &jobs, &mut aux)?;
        if let Some((dst, f)) = freq_vals {
            for (i, v) in f.iter().enumerate() {
                aux[dst + 4 * i..dst + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        drop(files);
        m.set_aux(aux.make_read_only()?);
    }
    Ok(m)
}

/// Why `m` can't run on the GPU, if it can't: a safetensors checkpoint's
/// converted F16 matrices have no GPU kernels.
pub fn gpu_refusal(m: &Model) -> Option<String> {
    matches!(m.source, Source::Safetensors { .. }).then(|| {
        "a safetensors checkpoint runs on the CPU only for now (--device cpu): its BF16/F16 \
         matrices have no GPU kernels; GPU support for safetensors weights comes with the FP8 \
         stage B"
            .to_string()
    })
}

#[cfg(test)]
#[path = "safetensors_tests.rs"]
mod tests;
