use super::*;
use crate::gguf::{self, Val};
use crate::quant::{F16, F32};
use serde_json::json;

/// FNV-1a 64 of bytes.
fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325, |x, &c| {
        (x ^ c as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// The BF16 -> F16 table is numpy's (what convert_hf_to_gguf.py writes):
/// FNV-1a of numpy's 65536 results (`np.arange(65536, dtype=np.uint32) << 16`
/// viewed as f32, `.astype(np.float16)`, little-endian u16s), computed with
/// numpy 2.x; torch's bf16 -> f32 -> numpy path gives the same table.
#[test]
fn bf16_table_matches_numpy() {
    let t = bf16_to_f16_table();
    let b: Vec<u8> = t.iter().flat_map(|h| h.to_le_bytes()).collect();
    assert_eq!(fnv(&b), 0xe1bc_8cd0_d393_4b25);
    // Spot checks: round to nearest even, overflow, subnormals, NaN payloads.
    assert_eq!(bf16_to_f16(0x3f80), 0x3c00); // 1.0
    assert_eq!(bf16_to_f16(0x4780), 0x7c00); // 65536 -> inf
    assert_eq!(bf16_to_f16(0x7f81), 0x7c08); // sNaN stays signalling (numpy)
    assert_eq!(bf16_to_f16(0x7fc0), 0x7e00);
    assert_eq!(bf16_to_f16(0x3380), 0x0001); // 2^-24
    assert_eq!(bf16_to_f16(0x3300), 0x0000); // 2^-25: tie to even
}

/// A safetensors file: header (padded to 8 with spaces, as HF writes it),
/// then each tensor's bytes in order.
fn st_bytes(ts: &[(&str, &str, Vec<u64>, Vec<u8>)], meta: Option<Value>) -> Vec<u8> {
    let mut h = Map::new();
    if let Some(m) = meta {
        h.insert("__metadata__".into(), m);
    }
    let mut at = 0u64;
    for (name, dt, shape, b) in ts {
        h.insert(
            name.to_string(),
            json!({"dtype": dt, "shape": shape, "data_offsets": [at, at + b.len() as u64]}),
        );
        at += b.len() as u64;
    }
    let mut j = serde_json::to_vec(&Value::Object(h)).unwrap();
    while !j.len().is_multiple_of(8) {
        j.push(b' ');
    }
    let mut out = (j.len() as u64).to_le_bytes().to_vec();
    out.extend(j);
    for t in ts {
        out.extend(&t.3);
    }
    out
}

/// A raw file: header length, header bytes, data.
fn raw(n: u64, header: &[u8], data: &[u8]) -> Vec<u8> {
    let mut v = n.to_le_bytes().to_vec();
    v.extend(header);
    v.extend(data);
    v
}

fn err(b: &[u8]) -> String {
    match parse_header(b) {
        Ok(_) => panic!("parsed: {:?}", String::from_utf8_lossy(b)),
        Err(e) => {
            assert_eq!(e.kind(), io::ErrorKind::InvalidData);
            e.to_string()
        }
    }
}

#[test]
fn header_parses_and_checks_every_range() {
    let f = st_bytes(
        &[
            ("a", "BF16", vec![2, 3], vec![1; 12]),
            ("b", "F32", vec![], vec![2; 4]),
            ("c", "F8_E4M3", vec![5], vec![3; 5]),
            ("d", "F4", vec![2, 4], vec![4; 4]),
        ],
        Some(json!({"format": "pt", "n": 3})),
    );
    let h = parse_header(&f).unwrap();
    let names: Vec<&str> = h.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["a", "b", "c", "d"]);
    let start = 8 + u64::from_le_bytes(f[..8].try_into().unwrap());
    assert_eq!((h.tensors[0].begin, h.tensors[0].end), (start, start + 12));
    assert_eq!(h.tensors[3].end, f.len() as u64);
    assert_eq!(
        h.metadata,
        [("format".into(), "pt".into()), ("n".into(), "3".into())]
    );

    // Truncated header, a 2 GB header length, past the end, not JSON, not
    // an object, bad entries.
    assert!(err(&[1, 2, 3]).contains("no header length"));
    assert!(err(&raw(2 << 30, b"{}", &[])).contains("limit"));
    assert!(err(&raw(64, b"{}", &[])).contains("truncated"));
    assert!(err(&raw(4, b"{\"a\"", &[])).contains("not valid JSON"));
    assert!(err(&raw(2, b"[]", &[])).contains("not a JSON object"));
    assert!(err(&raw(4, b"null", &[])).contains("not a JSON object"));
    let one = |t: Value, data: &[u8]| {
        let j = serde_json::to_vec(&json!({ "t": t })).unwrap();
        raw(j.len() as u64, &j, data)
    };
    let e = err(&one(
        json!({"dtype": "F16", "shape": [2], "data_offsets": [0, 8]}),
        &[0; 4],
    ));
    assert!(e.contains("outside"), "{e}");
    let e = err(&one(
        json!({"dtype": "F16", "shape": [2], "data_offsets": [4, 0]}),
        &[0; 4],
    ));
    assert!(e.contains("outside"), "{e}");
    let e = err(&one(
        json!({"dtype": "F16", "shape": [3], "data_offsets": [0, 4]}),
        &[0; 4],
    ));
    assert!(e.contains("needs 6"), "{e}");
    let e = err(&one(
        json!({"dtype": "Q4", "shape": [2], "data_offsets": [0, 4]}),
        &[0; 4],
    ));
    assert!(e.contains("unknown dtype Q4"), "{e}");
    let e = err(&one(
        json!({"dtype": "U8", "shape": [1,1,1,1,1], "data_offsets": [0, 1]}),
        &[0],
    ));
    assert!(e.contains("5 dimensions"), "{e}");
    let big = u64::MAX / 2;
    let e = err(&one(
        json!({"dtype": "F32", "shape": [big, 4], "data_offsets": [0, 4]}),
        &[0; 4],
    ));
    assert!(e.contains("overflows"), "{e}");
    let e = err(&one(
        json!({"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}),
        &[0; 4],
    ));
    assert!(e.contains("non-integer"), "{e}");
    let e = err(&one(
        json!({"dtype": "F32", "shape": [1], "data_offsets": [0]}),
        &[0; 4],
    ));
    assert!(e.contains("data_offsets"), "{e}");
    let e = err(&one(json!([1, 2]), &[0; 4]));
    assert!(e.contains("not an object"), "{e}");
    // The ranges must tile the data exactly (the reference reader's
    // `Metadata::validate`): overlaps, gaps, a nonzero start and trailing
    // bytes are all refused.
    let tiles = |t: Value, data: usize| {
        let j = serde_json::to_vec(&t).unwrap();
        parse_header(&raw(j.len() as u64, &j, &vec![0; data])).map(|h| h.tensors.len())
    };
    let u8s = |b: u64, e: u64| json!({"dtype": "U8", "shape": [e - b], "data_offsets": [b, e]});
    assert_eq!(
        tiles(json!({"x": u8s(0, 4), "y": u8s(4, 6), "z": u8s(6, 6)}), 6).unwrap(),
        3
    );
    assert_eq!(tiles(json!({}), 0).unwrap(), 0);
    for (what, t, n) in [
        ("overlap", json!({"x": u8s(0, 4), "y": u8s(1, 3)}), 4),
        ("same range", json!({"x": u8s(0, 4), "y": u8s(0, 4)}), 4),
        ("gap", json!({"x": u8s(0, 2), "y": u8s(3, 4)}), 4),
        ("late start", json!({"x": u8s(1, 4)}), 4),
        ("trailing", json!({"x": u8s(0, 4)}), 5),
        ("no tensors, data", json!({}), 1),
    ] {
        let e = tiles(t, n).unwrap_err().to_string();
        assert!(e.contains("tile") || e.contains("cover"), "{what}: {e}");
    }
}

/// Review MUST 1 (#102): a small file can't ask for a big arena. Before the
/// tiling check, N tensors sharing one range each got their own converted
/// copy: a 1 GB file could ask for N x its size of anonymous memory. Now
/// such a header is refused before anything is allocated.
#[test]
fn shared_ranges_cant_inflate_the_arena() {
    let n = 5000;
    let mut h = Map::new();
    for l in 0..n {
        h.insert(
            format!("model.layers.{l}.mlp.up_proj.weight"),
            json!({"dtype": "BF16", "shape": [64, 64], "data_offsets": [0, 8192]}),
        );
    }
    let j = serde_json::to_vec(&Value::Object(h)).unwrap();
    let e = parse_header(&raw(j.len() as u64, &j, &[0; 8192]))
        .err()
        .unwrap();
    assert!(e.to_string().contains("without gaps or overlaps"), "{e}");
    // And through the loader, as a checkpoint.
    let d = Dir::new("bomb");
    write_tiny(&d, 1);
    d.put("model.safetensors", &raw(j.len() as u64, &j, &[0; 8192]));
    let e = load_dir(&d.0).err().unwrap().to_string();
    assert!(e.contains("without gaps or overlaps"), "{e}");
}

/// Review SHOULD 6 (#102): a checkpoint has one cache id however it's named
/// (its directory or a shard in it), and it covers every file: a changed
/// tokenizer.json or second shard changes it.
#[test]
fn model_id_covers_the_whole_checkpoint() {
    let d = Dir::new("mid");
    write_tiny(&d, 2);
    let id = || crate::kvsave::model_id(&d.0).unwrap();
    let a = id();
    let b = crate::kvsave::model_id(&d.0.join("model-00001-of-00002.safetensors")).unwrap();
    assert_eq!(a, b);
    let mut t = tiny_tokenizer();
    t["added_tokens"][2]["content"] = json!("<tool_call2>");
    d.json("tokenizer.json", &t);
    let c = id();
    assert_ne!(a, c);
    let f = d.0.join("model-00002-of-00002.safetensors");
    let mut s2 = std::fs::read(&f).unwrap();
    let last = s2.len() - 1;
    s2[last] ^= 1;
    std::fs::write(&f, &s2).unwrap();
    assert_ne!(c, id());
}

/// Review SHOULD 5 (#102): the JSON files are capped like the header.
#[test]
fn oversized_json_is_refused() {
    let d = Dir::new("bigjson");
    write_tiny(&d, 1);
    let f = std::fs::File::create(d.0.join("tokenizer.json")).unwrap();
    f.set_len(MAX_JSON + 1).unwrap(); // sparse: no disk
    let e = load_dir(&d.0).err().unwrap().to_string();
    assert!(
        e.contains("over the") && e.contains("tokenizer.json"),
        "{e}"
    );
}

/// Every prefix of a valid file and every single-byte corruption of its
/// header is an error or a parse, never a panic.
#[test]
fn header_never_panics() {
    let f = st_bytes(
        &[
            ("a", "BF16", vec![2, 3], vec![1; 12]),
            ("b", "F32", vec![1], vec![2; 4]),
        ],
        None,
    );
    for n in 0..f.len() {
        let _ = parse_header(&f[..n]);
    }
    let hl = 8 + u64::from_le_bytes(f[..8].try_into().unwrap()) as usize;
    for i in 0..hl {
        for x in [0u8, b'9', b'"', b'}', b'-', 0xff] {
            let mut g = f.clone();
            g[i] = x;
            let _ = parse_header(&g);
        }
    }
}

/// GPT-2's byte -> char map (the byte-level vocabulary's alphabet).
fn byte_chars() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = vec![' '; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap();
    }
    out
}

const QWEN2_RE: &str = PRE_PATTERNS[0].0;

/// A byte-level BPE tokenizer.json: the 256 bytes, a few merges, three
/// added tokens (ids 256..259) and one gap before `vocab` (config).
fn tiny_tokenizer() -> Value {
    let bc = byte_chars();
    let mut vocab = Map::new();
    for (i, c) in bc.iter().enumerate() {
        vocab.insert(c.to_string(), json!(i));
    }
    vocab.insert("he".into(), json!(259));
    vocab.insert("ll".into(), json!(260));
    json!({
        "version": "1.0",
        "added_tokens": [
            {"id": 256, "content": "<|endoftext|>", "special": true, "normalized": false},
            {"id": 257, "content": "<|im_start|>", "special": true, "normalized": false},
            {"id": 258, "content": "<tool_call>", "special": false, "normalized": false},
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN2_RE}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "model": {"type": "BPE", "byte_fallback": false, "vocab": vocab, "merges": ["h e", ["l", "l"]]},
    })
}

/// The tiny qwen2 (dim 64, 2 heads / 1 kv, ffn 128, 2 layers, vocab 264)
/// as HF tensors of BF16 values from a fixed LCG: (HF name, PyTorch shape,
/// bf16 values).
fn tiny_weights() -> Vec<(String, Vec<u64>, Vec<u16>)> {
    let mut s = 0x1234_5678u32;
    let mut rnd = |n: usize, scale: f32, bias: f32| -> Vec<u16> {
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let x = ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * scale + bias;
                // Truncation keeps the low bits varied; any bf16 is fine.
                (x.to_bits() >> 16) as u16
            })
            .collect()
    };
    let mut v = vec![(
        "model.embed_tokens.weight".to_string(),
        vec![264, 64],
        rnd(264 * 64, 2.0, 0.0),
    )];
    for l in 0..2 {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        v.push((p("input_layernorm.weight"), vec![64], rnd(64, 0.5, 1.0)));
        v.push((
            p("self_attn.q_proj.weight"),
            vec![64, 64],
            rnd(64 * 64, 0.4, 0.0),
        ));
        v.push((p("self_attn.q_proj.bias"), vec![64], rnd(64, 0.2, 0.0)));
        v.push((
            p("self_attn.k_proj.weight"),
            vec![32, 64],
            rnd(32 * 64, 0.4, 0.0),
        ));
        v.push((p("self_attn.k_proj.bias"), vec![32], rnd(32, 0.2, 0.0)));
        v.push((
            p("self_attn.v_proj.weight"),
            vec![32, 64],
            rnd(32 * 64, 0.4, 0.0),
        ));
        v.push((p("self_attn.v_proj.bias"), vec![32], rnd(32, 0.2, 0.0)));
        v.push((
            p("self_attn.o_proj.weight"),
            vec![64, 64],
            rnd(64 * 64, 0.4, 0.0),
        ));
        v.push((
            p("post_attention_layernorm.weight"),
            vec![64],
            rnd(64, 0.5, 1.0),
        ));
        v.push((
            p("mlp.gate_proj.weight"),
            vec![128, 64],
            rnd(128 * 64, 0.4, 0.0),
        ));
        v.push((
            p("mlp.up_proj.weight"),
            vec![128, 64],
            rnd(128 * 64, 0.4, 0.0),
        ));
        v.push((
            p("mlp.down_proj.weight"),
            vec![64, 128],
            rnd(64 * 128, 0.4, 0.0),
        ));
        v.push((
            p("self_attn.rotary_emb.inv_freq"),
            vec![16],
            rnd(16, 1.0, 0.0),
        ));
    }
    v.push(("model.norm.weight".into(), vec![64], rnd(64, 0.5, 1.0)));
    v
}

fn tiny_config() -> Value {
    json!({
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_act": "silu",
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true,
        "vocab_size": 264, "bos_token_id": 256, "eos_token_id": 256,
        "use_sliding_window": false, "torch_dtype": "bfloat16",
    })
}

/// A scratch directory, removed on drop.
struct Dir(PathBuf);
impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("axon-st-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
    fn put(&self, name: &str, b: &[u8]) {
        std::fs::write(self.0.join(name), b).unwrap();
    }
    fn json(&self, name: &str, v: &Value) {
        self.put(name, &serde_json::to_vec(v).unwrap());
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The tiny checkpoint in `d`, in `shards` files (an index when > 1).
fn write_tiny(d: &Dir, shards: usize) {
    d.json("config.json", &tiny_config());
    d.json("tokenizer.json", &tiny_tokenizer());
    d.json(
        "tokenizer_config.json",
        &json!({"add_bos_token": false, "bos_token": null, "eos_token": "<|endoftext|>",
                "chat_template": "{{ messages }}"}),
    );
    let w = tiny_weights();
    let per = w.len().div_ceil(shards);
    let mut wm = Map::new();
    for (i, part) in w.chunks(per).enumerate() {
        let ts: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = part
            .iter()
            .map(|(n, s, v)| {
                (
                    n.as_str(),
                    "BF16",
                    s.clone(),
                    v.iter().flat_map(|h| h.to_le_bytes()).collect(),
                )
            })
            .collect();
        let f = if shards == 1 {
            "model.safetensors".to_string()
        } else {
            format!("model-{:05}-of-{shards:05}.safetensors", i + 1)
        };
        for t in &ts {
            wm.insert(t.0.into(), json!(f));
        }
        d.put(&f, &st_bytes(&ts, Some(json!({"format": "pt"}))));
    }
    if shards > 1 {
        d.json(
            "model.safetensors.index.json",
            &json!({"metadata": {}, "weight_map": wm}),
        );
    }
}

/// The GGUF F16 file convert_hf_to_gguf.py would write for the tiny
/// checkpoint, built here independently of the loader: matrices bf16 ->
/// f32 -> f16 (`quant::f32_to_f16`; no NaNs here), vectors bf16 -> f32.
fn tiny_gguf(meta: &[(String, Val)]) -> Model {
    let mut list: Vec<(String, Vec<u64>, u32)> = Vec::new();
    let mut data = Vec::new();
    for (n, shape, v) in tiny_weights() {
        let g = gguf_name(&n).unwrap();
        if g.is_empty() {
            continue;
        }
        let dims: Vec<u64> = shape.iter().rev().copied().collect();
        let f = |h: &u16| f32::from_bits((*h as u32) << 16);
        if shape.len() == 1 {
            data.push(v.iter().flat_map(|h| f(h).to_le_bytes()).collect());
            list.push((g, dims, F32));
        } else {
            data.push(
                v.iter()
                    .flat_map(|h| crate::quant::f32_to_f16(f(h)).to_le_bytes())
                    .collect(),
            );
            list.push((g, dims, F16));
        }
    }
    let meta: Vec<(&str, Val)> = meta.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let l: Vec<(&str, &[u64], u32)> = list.iter().map(|t| (t.0.as_str(), &t.1[..], t.2)).collect();
    gguf::build_for_test_with(&meta, &l, &data)
}

fn load_dir(d: &Path) -> io::Result<Model> {
    load(d, |_| gguf::CopyWish::MapOnly, true)
}

/// The tiny checkpoint loads as its GGUF F16: same tensors (names, dims,
/// types, bytes), same metadata, same logits bit for bit; one file or
/// three shards.
#[test]
fn tiny_checkpoint_is_its_gguf_f16() {
    for shards in [1, 3] {
        let d = Dir::new(&format!("tiny{shards}"));
        write_tiny(&d, shards);
        let m = load_dir(&d.0).unwrap();
        crate::model::check_supported(&m).unwrap();
        let g = tiny_gguf(&m.metadata);
        assert_eq!(m.arch, "qwen2");
        assert_eq!(m.tensors.len(), g.tensors.len());
        for t in &g.tensors {
            let s = m.tensors.iter().find(|x| x.name == t.name).unwrap();
            assert_eq!(
                (s.dims.clone(), s.qtype.id),
                (t.dims.clone(), t.qtype.id),
                "{}",
                t.name
            );
            assert_eq!(m.tensor_data(s), g.tensor_data(t), "{}", t.name);
            assert_eq!(s.shard, gguf::AUX);
        }
        let ids = [256u32, 10, 72, 101, 108, 108, 111, 263];
        let run = |m: &Model| {
            let net = crate::model::Qwen2::new(m);
            let mut c = crate::Lm::new_cache(&net, 16);
            let v: Vec<Vec<f32>> = (ids.iter().enumerate())
                .map(|(p, &t)| crate::Lm::forward(&net, t, p, &mut c))
                .collect();
            v
        };
        let (a, b) = (run(&m), run(&g));
        for (x, y) in a.iter().zip(&b) {
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(x), bits(y), "{shards} shards");
        }
        assert!(matches!(m.source, gguf::Source::Safetensors { .. }));
        assert!(gpu_refusal(&m).unwrap().contains("--device cpu"));
    }
}

#[test]
fn tiny_metadata_and_tokenizer() {
    let d = Dir::new("tinymeta");
    write_tiny(&d, 1);
    let m = load_dir(&d.0).unwrap();
    let get = |k: &str| m.metadata.iter().find(|x| x.0 == k).map(|x| x.1.clone());
    let u = |k: &str| match get(k) {
        Some(Val::U32(n)) => n,
        v => panic!("{k}: {v:?}"),
    };
    assert_eq!(u("qwen2.block_count"), 2);
    assert_eq!(u("qwen2.attention.head_count_kv"), 1);
    assert_eq!(u("tokenizer.ggml.bos_token_id"), 256);
    assert_eq!(u("tokenizer.ggml.eos_token_id"), 256);
    assert!(matches!(get("qwen2.rope.freq_base"), Some(Val::F32(f)) if f == 1e6));
    assert!(matches!(
        get("tokenizer.ggml.add_bos_token"),
        Some(Val::Bool(false))
    ));
    assert!(matches!(get("tokenizer.chat_template"), Some(Val::Str(s)) if s == "{{ messages }}"));
    assert!(get("qwen2.attention.key_length").is_none());
    assert!(matches!(get("safetensors.format"), Some(Val::Str(s)) if s == "pt"));
    let Some(Val::Arr(tokens)) = get("tokenizer.ggml.tokens") else {
        panic!()
    };
    let Some(Val::Arr(types)) = get("tokenizer.ggml.token_type") else {
        panic!()
    };
    assert_eq!(tokens.len(), 264);
    let tt = |i: usize| match (&tokens[i], &types[i]) {
        (Val::Str(s), Val::I32(t)) => (s.clone(), *t),
        _ => panic!(),
    };
    assert_eq!(tt(33), ("!".into(), 1));
    assert_eq!(tt(0), ("\u{100}".into(), 1));
    assert_eq!(tt(256), ("<|endoftext|>".into(), 3));
    assert_eq!(tt(258), ("<tool_call>".into(), 4));
    assert_eq!(tt(259), ("he".into(), 1));
    assert_eq!(tt(261), ("[PAD261]".into(), 5));
    let Some(Val::Arr(merges)) = get("tokenizer.ggml.merges") else {
        panic!()
    };
    assert!(matches!(&merges[1], Val::Str(s) if s == "l l"));
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    assert_eq!(
        tok.encode_special("hello<|im_start|>"),
        [259, 260, 111, 257]
    );
    assert_eq!(tok.encode_special("hello<tool_call>"), [259, 260, 111, 258]);
}

/// A checkpoint `edit`ed from the tiny one fails to load with `want` in the error.
fn refused(tag: &str, edit: impl FnOnce(&Dir), want: &str) {
    let d = Dir::new(tag);
    write_tiny(&d, 2);
    edit(&d);
    match load_dir(&d.0) {
        Ok(_) => panic!("{tag}: loaded"),
        Err(e) => assert!(e.to_string().contains(want), "{tag}: {e}"),
    }
}

#[test]
fn bad_checkpoints_are_clean_errors() {
    let cfg = |f: fn(&mut Map<String, Value>)| {
        move |d: &Dir| {
            let mut c = tiny_config();
            f(c.as_object_mut().unwrap());
            d.json("config.json", &c);
        }
    };
    let tok = |f: fn(&mut Value)| {
        move |d: &Dir| {
            let mut t = tiny_tokenizer();
            f(&mut t);
            d.json("tokenizer.json", &t);
        }
    };
    refused(
        "shard",
        |d| std::fs::remove_file(d.0.join("model-00002-of-00002.safetensors")).unwrap(),
        "is missing",
    );
    refused(
        "trav",
        |d| {
            let mut ix = read_json(&d.0.join("model.safetensors.index.json")).unwrap();
            ix["weight_map"]["model.norm.weight"] = json!("../x.safetensors");
            d.json("model.safetensors.index.json", &ix);
        },
        "not a file name",
    );
    refused(
        "elsewhere",
        |d| {
            let mut ix = read_json(&d.0.join("model.safetensors.index.json")).unwrap();
            ix["weight_map"]["model.norm.weight"] = json!("model-00001-of-00002.safetensors");
            d.json("model.safetensors.index.json", &ix);
        },
        "doesn't hold it",
    );
    refused(
        "trunc",
        |d| d.put("model-00002-of-00002.safetensors", &[9, 0, 0]),
        "no header length",
    );
    refused(
        "arch",
        cfg(|c| {
            c.insert("model_type".into(), json!("phi3"));
        }),
        "model_type 'phi3' is not supported",
    );
    refused(
        "moe",
        cfg(|c| {
            c.insert("model_type".into(), json!("qwen3_moe"));
        }),
        "not supported in safetensors form yet",
    );
    refused(
        "fp8",
        cfg(|c| {
            c.insert("quantization_config".into(), json!({"quant_method": "fp8"}));
        }),
        "quant_method fp8",
    );
    refused(
        "modelopt",
        |d| d.json("hf_quant_config.json", &json!({})),
        "ModelOpt",
    );
    refused(
        "act",
        cfg(|c| {
            c.insert("hidden_act".into(), json!("gelu"));
        }),
        "hidden_act gelu",
    );
    refused(
        "swa",
        cfg(|c| {
            c.insert("use_sliding_window".into(), json!(true));
        }),
        "use_sliding_window",
    );
    refused(
        "rope",
        cfg(|c| {
            c.insert(
                "rope_scaling".into(),
                json!({"type": "dynamic", "factor": 2.0}),
            );
        }),
        "RoPE scaling type dynamic",
    );
    refused(
        "linear-nofactor",
        cfg(|c| {
            c.insert("rope_scaling".into(), json!({"type": "linear"}));
        }),
        "RoPE scaling linear without a factor",
    );
    refused(
        "untied",
        cfg(|c| {
            c.insert("tie_word_embeddings".into(), json!(false));
        }),
        "lm_head.weight is missing",
    );
    refused(
        "layers",
        cfg(|c| {
            c.insert("num_hidden_layers".into(), json!(-2));
        }),
        "not a whole number",
    );
    refused(
        "cfgjson",
        |d| d.put("config.json", b"[1,"),
        "not valid JSON",
    );
    refused(
        "spm",
        tok(|t| {
            t["model"]["byte_fallback"] = json!(true);
        }),
        "tokenizer.model",
    );
    refused(
        "wp",
        tok(|t| {
            t["model"]["type"] = json!("WordPiece");
        }),
        "model type WordPiece",
    );
    refused(
        "nfkc",
        tok(|t| {
            t["normalizer"] = json!({"type": "NFKC"});
        }),
        "normalizer NFKC",
    );
    refused(
        "re",
        tok(|t| {
            t["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!("\\s+");
        }),
        "regex \"\\\\s+\"",
    );
    refused(
        "bl",
        tok(|t| {
            t["pre_tokenizer"] = json!({"type": "ByteLevel", "use_regex": true});
        }),
        "pre-tokenizer",
    );
    refused(
        "ig",
        tok(|t| {
            t["model"]["ignore_merges"] = json!(true);
        }),
        "ignore_merges",
    );
    refused(
        "vocab",
        tok(|t| {
            t["added_tokens"][0]["id"] = json!(9999);
        }),
        "ids up to 9999",
    );
    refused(
        "tensor",
        |d| {
            let b = st_bytes(
                &[(
                    "model.layers.0.mlp.up_proj.bias",
                    "BF16",
                    vec![2],
                    vec![0; 4],
                )],
                None,
            );
            d.put("model-00002-of-00002.safetensors", &b);
            let mut ix = read_json(&d.0.join("model.safetensors.index.json")).unwrap();
            ix["weight_map"] =
                json!({"model.layers.0.mlp.up_proj.bias": "model-00002-of-00002.safetensors"});
            d.json("model.safetensors.index.json", &ix);
        },
        "is not a qwen2 tensor Axon knows",
    );
}

/// An FP8 tensor in a checkpoint without a quantization_config: loads
/// (named in `axon info`), refused by check_supported.
#[test]
fn unconverted_types_are_refused_by_check_supported() {
    let d = Dir::new("f8");
    write_tiny(&d, 1);
    let up = "model.layers.0.mlp.up_proj.weight";
    let w = tiny_weights();
    let ts: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = (w.iter())
        .map(|(n, sh, v)| match n.as_str() {
            n if n == up => (n, "F8_E4M3", vec![128, 64], vec![0x38; 128 * 64]),
            n => (
                n,
                "BF16",
                sh.clone(),
                v.iter().flat_map(|h| h.to_le_bytes()).collect(),
            ),
        })
        .collect();
    d.put("model.safetensors", &st_bytes(&ts, None));
    let m = load_dir(&d.0).unwrap();
    let e = crate::model::check_supported(&m).unwrap_err().to_string();
    assert!(
        e.contains("F8_E4M3") && e.contains("blk.0.ffn_up.weight"),
        "{e}"
    );
}

#[test]
fn names_map_to_gguf() {
    assert_eq!(
        gguf_name("model.layers.12.self_attn.o_proj.weight").unwrap(),
        "blk.12.attn_output.weight"
    );
    assert_eq!(
        gguf_name("model.layers.0.self_attn.k_norm.weight").unwrap(),
        "blk.0.attn_k_norm.weight"
    );
    assert_eq!(gguf_name("lm_head.weight").unwrap(), "output.weight");
    assert_eq!(
        gguf_name("model.layers.3.self_attn.rotary_emb.inv_freq").unwrap(),
        ""
    );
    assert!(gguf_name("model.layers.x.mlp.up_proj.weight").is_none());
    assert!(gguf_name("model.layers.+1.mlp.up_proj.weight").is_none());
    assert!(gguf_name("model.layers.007.mlp.up_proj.weight").is_none());
    assert!(gguf_name("model.layers.0.mlp.up_proj.weight").is_some());
    assert!(gguf_name("model.layers.1.mlp.up_proj.bias").is_none());
    assert!(gguf_name("model.vision_tower.weight").is_none());
}

// ---------------------------------------------------------------------------
// Real checkpoints (models/hf/*), skipped when absent: they're the stage A
// gate's models (docs/log/2026-10-19-safetensors-loader.md), not on every
// test host.

fn models() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models")
}

/// `rel` under models/, or None (skipped, said on stderr).
fn opt(rel: &str) -> Option<PathBuf> {
    let p = models().join(rel);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skipping: models/{rel} is not here");
        None
    }
}

/// The keys model.rs, rope.rs, moe.rs and tokenizer.rs read, as one list.
const READ_KEYS: &[&str] = &[
    "general.architecture",
    "{a}.block_count",
    "{a}.context_length",
    "{a}.embedding_length",
    "{a}.feed_forward_length",
    "{a}.attention.head_count",
    "{a}.attention.head_count_kv",
    "{a}.attention.key_length",
    "{a}.attention.value_length",
    "{a}.attention.layer_norm_rms_epsilon",
    "{a}.rope.freq_base",
    "{a}.rope.dimension_count",
    "{a}.rope.scaling.type",
    "{a}.rope.scaling.factor",
    "{a}.rope.scaling.original_context_length",
    "{a}.rope.scale_linear",
    "{a}.attention.scale",
    "{a}.attention.sliding_window",
    "{a}.expert_count",
    "{a}.vocab_size",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.chat_template",
];

fn same_val(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Arr(x), Val::Arr(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_val(p, q))
        }
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::F32(x), Val::F32(y)) => x.to_bits() == y.to_bits(),
        (x, y) => x.as_f64().is_some() && x.as_f64() == y.as_f64(),
    }
}

/// Every key Axon reads that the GGUF has, synthesized with the same
/// value, and none it doesn't have (`except`: keys whose GGUF value came
/// from a different revision of the files, checked by hand in the log).
fn metadata_parity(dir: &Path, gguf_path: &Path, except: &[&str]) -> (Model, Model) {
    let (_, meta) = checkpoint_metadata(dir).unwrap();
    let g = gguf::load_with(gguf_path.to_str().unwrap(), Some(false)).unwrap();
    let a = g.arch.clone();
    assert_eq!(meta.arch, a);
    let get = |m: &[(String, Val)], k: &str| m.iter().find(|x| x.0 == k).map(|x| x.1.clone());
    let mut bad = Vec::new();
    for k in READ_KEYS {
        let k = k.replace("{a}", &a);
        if except.contains(&k.as_str()) {
            continue;
        }
        match (get(&meta.keys, &k), get(&g.metadata, &k)) {
            (None, None) => {}
            (Some(x), Some(y)) if same_val(&x, &y) => {}
            (x, y) => bad.push(format!("{k}: {x:?} vs {y:?}")),
        }
    }
    assert!(bad.is_empty(), "{}: {bad:#?}", dir.display());
    let kv: Vec<(&str, Val)> = meta
        .keys
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();
    (gguf::build_for_test(&kv, &[]), g)
}

/// The tokenizer parity corpus: the 50 logits-gate prompts, the fixture
/// texts and edge cases (decomposed accents, which NFC would compose and
/// neither side does; specials; long runs of spaces, digits, newlines;
/// emoji; mixed scripts).
fn corpus() -> Vec<String> {
    let mut v: Vec<String> = crate::testutil::read_json("tokens.json")
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["text"].as_str().unwrap().to_string())
        .collect();
    v.extend(
        std::fs::read_to_string(crate::testutil::fixtures().join("natural_text.txt"))
            .unwrap()
            .split("\n\n")
            .map(str::to_string),
    );
    v.extend(GATE_PROMPTS.iter().map(|s| s.to_string()));
    v.extend(
        [
            "e\u{301}cole cafe\u{301} n\u{303}",
            "école café ñ",
            "<|endoftext|><|im_start|>system\n<|im_end|><tool_call>x</tool_call>",
            "    \n\n\n\t\t  x  1234567890 12 123 1234 3.14159 1,000,000",
            "🚀🦙👩‍👩‍👧 ✅ 😶‍🌫️",
            "東京は日本の首都です。서울은 한국의 수도입니다. Москва — столица.",
            "I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you",
            "",
            " ",
            "\u{0}\u{1}\u{7f}",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    v
}

/// The logits gate's prompt set (scripts/oracle/llama_logits.py PROMPTS).
pub(crate) const GATE_PROMPTS: &[&str] = &[
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "The three laws of thermodynamics are",
    "import numpy as np\nimport matplotlib.pyplot as plt\n\n",
    "Q: What is 17 times 23?\nA:",
    "The quick brown fox",
    "In 1969, Neil Armstrong",
    "To make a cup of tea, first",
    "SELECT name, age FROM users WHERE",
    "The mitochondria is",
    "Dear Sir or Madam,\n\nI am writing to",
    "The best way to learn a new language is",
    "fn main() {\n    let v: Vec<i32> =",
    "Photosynthesis converts",
    "The stock market crashed in",
    "Water boils at",
    "Shakespeare wrote",
    "The largest planet in our solar system is",
    "<|im_start|>user\nWhat is the speed of light?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a haiku about autumn.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nTranslate 'good morning' into German.<|im_end|>\n<|im_start|>assistant\n",
    "Le chat est assis sur",
    "Der schnellste Weg nach Berlin ist",
    "東京は日本の",
    "A neural network consists of",
    "The recipe calls for two cups of flour,",
    "#include <stdio.h>\n\nint main(void) {\n",
    "Monday, Tuesday, Wednesday,",
    "1, 1, 2, 3, 5, 8, 13,",
    "The opposite of hot is",
    "My favorite color is blue because",
    "The French Revolution began in",
    "In machine learning, overfitting means",
    "She opened the door and saw",
    "The theory of relativity states that",
    "Breaking news: scientists have discovered",
    "To install the package, run",
    "Roses are red, violets are",
    "The function returns None if",
    "Mount Everest is located in",
    "A haiku has three lines with",
    "The integral of x squared is",
    "Once the server starts, it listens on port",
    "The primary colors are red, yellow, and",
    "He was born in 1879 in Ulm, Germany, and",
    "git commit -m \"",
    "The patient presented with a fever and",
    "Chapter 1\n\nIt was a bright cold day in April, and",
    "The square root of 144 is",
];

fn tokenizer_parity(a: &Model, b: &Model) -> usize {
    let (ta, tb) = (
        crate::tokenizer::Tokenizer::from_gguf(a),
        crate::tokenizer::Tokenizer::from_gguf(b),
    );
    assert_eq!(ta.vocab_len(), tb.vocab_len());
    assert_eq!(
        (ta.eos(), ta.bos(), ta.eog()),
        (tb.eos(), tb.bos(), tb.eog())
    );
    let c = corpus();
    for s in &c {
        assert_eq!(ta.encode_prompt(s), tb.encode_prompt(s), "{s:?}");
        assert_eq!(ta.encode_special(s), tb.encode_special(s), "{s:?}");
    }
    c.len()
}

/// Gate 2 on Qwen2.5-0.5B-Instruct: the checkpoint's synthesized keys are
/// its converted GGUF's, and the two tokenizers agree token for token.
#[test]
fn qwen25_metadata_and_tokenizer_match_gguf() {
    let (Some(d), Some(g)) = (
        opt("hf/Qwen2.5-0.5B-Instruct"),
        opt("Qwen2.5-0.5B-Instruct-F16.gguf"),
    ) else {
        return;
    };
    let (a, b) = metadata_parity(&d, &g, &[]);
    let n = tokenizer_parity(&a, &b);
    eprintln!("qwen2.5: {n} texts token for token");
}

/// The same on Qwen3-0.6B's files against Qwen/Qwen3-0.6B-GGUF's Q8_0
/// (metadata only: the weights aren't downloaded). The GGUF was made
/// before Qwen updated the chat template: that key is not compared.
#[test]
fn qwen3_metadata_and_tokenizer_match_gguf() {
    let (Some(d), Some(g)) = (opt("hf/Qwen3-0.6B-meta"), opt("Qwen3-0.6B-Q8_0.gguf")) else {
        return;
    };
    let (a, b) = metadata_parity(&d, &g, &["tokenizer.chat_template"]);
    tokenizer_parity(&a, &b);
}

/// Llama-3.2-1B-Instruct's files (unsloth's ungated copy) against the
/// Q4_K_M GGUF: metadata, the tokenizer, and the computed Llama 3 RoPE
/// factors against the GGUF's `rope_freqs.weight`, bit for bit. That GGUF
/// (bartowski's) predates the converter writing `add_bos_token`; Axon's
/// default for llama-bpe is the `true` synthesized here, and
/// `tokenizer_parity` checks the BOS both sides add.
#[test]
fn llama32_metadata_tokenizer_and_rope_freqs_match_gguf() {
    let (Some(d), Some(g)) = (
        opt("hf/Llama-3.2-1B-Instruct-meta"),
        opt("Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ) else {
        return;
    };
    let except = ["tokenizer.chat_template", "tokenizer.ggml.add_bos_token"];
    let (a, b) = metadata_parity(&d, &g, &except);
    // Not compared with that GGUF (it lacks the key), but pinned: the
    // converter's SpecialVocab sets it from the TemplateProcessing BOS.
    let add_bos = a
        .metadata
        .iter()
        .find(|x| x.0 == "tokenizer.ggml.add_bos_token");
    assert!(matches!(add_bos, Some((_, Val::Bool(true)))), "{add_bos:?}");
    tokenizer_parity(&a, &b);
    let (_, meta) = checkpoint_metadata(&d).unwrap();
    let f = meta.rope_freqs.unwrap();
    let t = b
        .tensors
        .iter()
        .find(|t| t.name == "rope_freqs.weight")
        .unwrap();
    let want: Vec<u32> = (b.tensor_data(t).chunks_exact(4))
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(f.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want);
}

/// Every tensor of checkpoint `d`, converted, is byte for byte the
/// converter's GGUF F16 tensor of the same name (dims, type, bytes) as Axon
/// loads it, and there are no others. For llama that's after Axon undoes
/// the converter's q/k row permutation on the GGUF: the checkpoint's rows
/// are taken as they are.
fn tensors_match(d: &Path, g: &Path) {
    let m = load_dir(d).unwrap();
    let g = gguf::load_with(g.to_str().unwrap(), Some(false)).unwrap();
    assert_eq!(m.tensors.len(), g.tensors.len());
    for t in &g.tensors {
        let s =
            (m.tensors.iter().find(|x| x.name == t.name)).unwrap_or_else(|| panic!("{}", t.name));
        assert_eq!((&s.dims, s.qtype.id), (&t.dims, t.qtype.id), "{}", t.name);
        assert!(
            m.tensor_data(s) == g.tensor_data(t),
            "{}: bytes differ",
            t.name
        );
    }
}

/// Gate 1's premise on Qwen2.5-0.5B-Instruct.
#[test]
fn qwen25_tensors_match_gguf_f16() {
    let (Some(d), Some(g)) = (
        opt("hf/Qwen2.5-0.5B-Instruct"),
        opt("Qwen2.5-0.5B-Instruct-F16.gguf"),
    ) else {
        return;
    };
    tensors_match(&d, &g);
}

/// The same on Llama-3.2-1B-Instruct's weights (review #102 SHOULD 4):
/// attn_q / attn_k included, so taking a checkpoint's rows unpermuted is
/// right; rope_freqs.weight computed == the converter's.
#[test]
fn llama32_tensors_match_gguf_f16() {
    let (Some(d), Some(g)) = (
        opt("hf/Llama-3.2-1B-Instruct"),
        opt("Llama-3.2-1B-Instruct-F16.gguf"),
    ) else {
        return;
    };
    tensors_match(&d, &g);
}

/// Gate 3: `check_supported` refuses a checkpoint exactly as it refuses
/// the GGUF F16 of the same checkpoint (same message), for each of
/// today's metadata refusals reachable from a config.json: the
/// synthesized keys are the converter's, so the check can't tell them
/// apart. (The new, loader-level refusals: `bad_checkpoints_are_clean_errors`.)
#[test]
fn check_supported_refuses_as_for_the_gguf() {
    type Edit = fn(&mut Map<String, Value>);
    let cases: &[(&str, Edit, &str)] = &[
        (
            "linear",
            |c| {
                c.insert(
                    "rope_scaling".into(),
                    json!({"type": "linear", "factor": 2.0}),
                );
            },
            "RoPE scaling (factor 2)",
        ),
        (
            "yarn-high",
            |c| {
                c.insert("rope_scaling".into(), json!({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32, "beta_fast": -1.0}));
            },
            "out of range",
        ),
        (
            "kv-missing",
            |c| {
                c.remove("num_key_value_heads");
            },
            "head_count_kv is missing",
        ),
        (
            "kv-3",
            |c| {
                c.insert("num_key_value_heads".into(), json!(3));
            },
            "not a multiple",
        ),
        (
            "odd-head",
            |c| {
                c.insert("head_dim".into(), json!(33));
            },
            "odd",
        ),
        (
            "heads-split",
            |c| {
                c.insert("num_attention_heads".into(), json!(3));
                c.insert("num_key_value_heads".into(), json!(1));
            },
            "doesn't split",
        ),
        (
            "layers-missing",
            |c| {
                c.remove("num_hidden_layers");
            },
            "block_count is missing",
        ),
        (
            "layers-more",
            |c| {
                c.insert("num_hidden_layers".into(), json!(3));
            },
            "blk.2",
        ),
    ];
    for (tag, edit, want) in cases {
        let d = Dir::new(&format!("cs-{tag}"));
        write_tiny(&d, 1);
        let mut c = tiny_config();
        edit(c.as_object_mut().unwrap());
        d.json("config.json", &c);
        let m = load_dir(&d.0).unwrap();
        let g = tiny_gguf(&m.metadata);
        let (a, b) = (
            crate::model::check_supported(&m),
            crate::model::check_supported(&g),
        );
        let (a, b) = (a.unwrap_err().to_string(), b.unwrap_err().to_string());
        assert_eq!(a, b, "{tag}");
        assert!(a.contains(want), "{tag}: {a}");
    }
}

/// Fuzz finding (1019-023221-b0fe, mutant `st-config-rope_scaling={'rope_type': 'llama3'}`):
/// a computed `rope_freqs.weight` was read by `check` (`check_supported` ->
/// `rope_freqs`) before the aux arena existed, and `gen` panicked. Now the
/// computed tensors are in place before `check`, in a header-only load too,
/// and the model runs or is refused, never panics.
#[test]
fn llama3_rope_on_a_checkpoint_is_readable_before_conversion() {
    let d = Dir::new("llama3rope");
    write_tiny(&d, 2);
    let mut c = tiny_config();
    c["rope_scaling"] = json!({"rope_type": "llama3"});
    d.json("config.json", &c);
    let mut seen = None;
    let m = load(
        &d.0,
        |m| {
            let ok = crate::model::check_supported(m).map_err(|e| e.to_string());
            seen = Some((ok, crate::model::rope_freqs(m, 32)));
            gguf::CopyWish::MapOnly
        },
        true,
    )
    .unwrap();
    let (ok, freqs) = seen.unwrap();
    assert_eq!(freqs, crate::model::rope_freqs(&m, 32));
    assert_eq!(freqs.len(), 16);
    let h = load(&d.0, |_| gguf::CopyWish::MapOnly, false).unwrap();
    assert_eq!(freqs, crate::model::rope_freqs(&h, 32));
    if ok.is_ok() {
        let net = crate::model::Qwen2::new(&m);
        let mut c = crate::Lm::new_cache(&net, 4);
        assert_eq!(crate::Lm::forward(&net, 1, 0, &mut c).len(), 264);
    }
    // A header-only load's converted tensors are an error message, not garbage.
    let t = h
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up.weight")
        .unwrap();
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.tensor_data(t).len()));
    assert!(e.is_err());
}

// ---------------------------------------------------------------------------
// FP8 checkpoints (docs/design/fp8-weights.md §2, §5): compressed-tensors
// per-channel e4m3 with dynamic per-token activations.

/// RedHatAI's FP8-dynamic quantization_config (VERIFIED on the 0.5B / 7B).
fn fp8_qcfg() -> Value {
    let q = |strategy: &str, dynamic: bool| {
        json!({"actorder": null, "block_structure": null, "dynamic": dynamic, "group_size": null,
               "num_bits": 8, "observer": null, "observer_kwargs": {}, "strategy": strategy,
               "symmetric": true, "type": "float"})
    };
    json!({"config_groups": {"group_0": {"input_activations": q("token", true), "output_activations": null,
            "targets": ["Linear"], "weights": q("channel", false)}},
           "format": "float-quantized", "ignore": ["lm_head"], "kv_cache_scheme": null,
           "quant_method": "compressed-tensors", "quantization_status": "compressed"})
}

/// The tiny FP8 model's tensors: hidden 128 (whole 128-blocks per row),
/// ffn 256, every linear F8_E4M3 (random finite codes) with BF16 scales
/// [rows, 1], the rest BF16. (name, dtype, shape, bytes).
fn tiny_fp8_weights() -> Vec<(String, &'static str, Vec<u64>, Vec<u8>)> {
    let mut seed = 99u64;
    let mut s = 0x2468_ace0u32;
    let mut bf = |n: usize, scale: f32, bias: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let x = ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * scale + bias;
                ((x.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect()
    };
    let mut v = vec![(
        "model.embed_tokens.weight".to_string(),
        "BF16",
        vec![264u64, 128],
        bf(264 * 128, 2.0, 0.0),
    )];
    for l in 0..2 {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        v.push((
            p("input_layernorm.weight"),
            "BF16",
            vec![128],
            bf(128, 0.5, 1.0),
        ));
        v.push((
            p("post_attention_layernorm.weight"),
            "BF16",
            vec![128],
            bf(128, 0.5, 1.0),
        ));
        for (b, n) in [
            ("self_attn.q_proj", 128),
            ("self_attn.k_proj", 64),
            ("self_attn.v_proj", 64),
        ] {
            v.push((
                p(&format!("{b}.bias")),
                "BF16",
                vec![n],
                bf(n as usize, 0.2, 0.0),
            ));
        }
        for (b, rows, k) in [
            ("self_attn.q_proj", 128u64, 128u64),
            ("self_attn.k_proj", 64, 128),
            ("self_attn.v_proj", 64, 128),
            ("self_attn.o_proj", 128, 128),
            ("mlp.gate_proj", 256, 128),
            ("mlp.up_proj", 256, 128),
            ("mlp.down_proj", 128, 256),
        ] {
            v.push((
                p(&format!("{b}.weight")),
                "F8_E4M3",
                vec![rows, k],
                crate::quant::fp8::test_codes((rows * k) as usize, &mut seed),
            ));
            v.push((
                p(&format!("{b}.weight_scale")),
                "BF16",
                vec![rows, 1],
                bf(rows as usize, 0.004, 0.006),
            ));
        }
    }
    v.push((
        "model.norm.weight".into(),
        "BF16",
        vec![128],
        bf(128, 0.5, 1.0),
    ));
    v
}

/// The tiny FP8 checkpoint in `d`, `edit` applied to its tensors first.
fn write_tiny_fp8(d: &Dir, edit: impl FnOnce(&mut Vec<(String, &'static str, Vec<u64>, Vec<u8>)>)) {
    let mut c = tiny_config();
    let o = c.as_object_mut().unwrap();
    o.insert("hidden_size".into(), json!(128));
    o.insert("intermediate_size".into(), json!(256));
    o.insert("quantization_config".into(), fp8_qcfg());
    d.json("config.json", &c);
    d.json("tokenizer.json", &tiny_tokenizer());
    d.json(
        "tokenizer_config.json",
        &json!({"add_bos_token": false, "bos_token": null, "eos_token": "<|endoftext|>",
                "chat_template": "{{ messages }}"}),
    );
    let mut w = tiny_fp8_weights();
    edit(&mut w);
    let ts: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = w
        .iter()
        .map(|(n, t, s, b)| (n.as_str(), *t, s.clone(), b.clone()))
        .collect();
    d.put(
        "model.safetensors",
        &st_bytes(&ts, Some(json!({"format": "pt"}))),
    );
}

/// The tiny FP8 checkpoint on the CPU (fp8-ref) gives the logits of the GGUF
/// whose matrices are F32 holding e4m3 * scale (the exact dequant: HF F32's
/// weights), bit for bit; its matrices are F8_E4M3 read in place with an
/// F32 scale per row; it runs on the GPU (no refusal).
#[test]
fn tiny_fp8_checkpoint_is_its_exact_dequant() {
    use crate::quant::fp8::e4m3_to_f32;
    let d = Dir::new("fp8");
    write_tiny_fp8(&d, |_| {});
    let m = load_dir(&d.0).unwrap();
    crate::model::check_supported(&m).unwrap();
    assert!(is_fp8(&m) && gpu_refusal(&m).is_none());
    let q = m
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .unwrap();
    assert_eq!((q.qtype.id, q.shard), (QT_F8_E4M3, 0));
    let sc = m
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.scale")
        .unwrap();
    assert_eq!((sc.qtype.id, sc.n_elements), (F32, 128));
    // The GGUF of the dequantized weights.
    let w = tiny_fp8_weights();
    let bf = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    };
    let mut list: Vec<(String, Vec<u64>, u32)> = Vec::new();
    let mut data: Vec<Vec<u8>> = Vec::new();
    for (n, dt, shape, b) in &w {
        if n.ends_with("_scale") {
            continue;
        }
        let g = gguf_name(n).unwrap();
        let dims: Vec<u64> = shape.iter().rev().copied().collect();
        if *dt == "F8_E4M3" {
            let s = bf(&w.iter().find(|x| x.0 == format!("{n}_scale")).unwrap().3);
            let k = shape[1] as usize;
            let v: Vec<u8> = b
                .iter()
                .enumerate()
                .flat_map(|(i, &c)| (e4m3_to_f32(c) * s[i / k]).to_le_bytes())
                .collect();
            data.push(v);
            list.push((g, dims, F32));
        } else if shape.len() == 1 {
            data.push(bf(b).iter().flat_map(|f| f.to_le_bytes()).collect());
            list.push((g, dims, F32));
        } else {
            data.push(
                bf(b)
                    .iter()
                    .flat_map(|&f| crate::quant::f32_to_f16(f).to_le_bytes())
                    .collect(),
            );
            list.push((g, dims, F16));
        }
    }
    let meta: Vec<(&str, Val)> = m
        .metadata
        .iter()
        .filter(|(k, _)| k != FP8_KEY)
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();
    let l: Vec<(&str, &[u64], u32)> = list.iter().map(|t| (t.0.as_str(), &t.1[..], t.2)).collect();
    let g = gguf::build_for_test_with(&meta, &l, &data);
    let ids = [256u32, 10, 72, 101, 108, 108, 111, 263];
    let run = |m: &Model| -> Vec<Vec<u32>> {
        let net = crate::model::Qwen2::new(m);
        let mut c = crate::Lm::new_cache(&net, 16);
        let mut v: Vec<Vec<u32>> = (ids.iter().enumerate())
            .map(|(p, &t)| {
                crate::Lm::forward(&net, t, p, &mut c)
                    .iter()
                    .map(|f| f.to_bits())
                    .collect()
            })
            .collect();
        let mut c2 = crate::Lm::new_cache(&net, 16);
        v.push(
            crate::Lm::prefill_logits(&net, &ids, 0, &mut c2)
                .iter()
                .map(|f| f.to_bits())
                .collect(),
        );
        v
    };
    assert_eq!(run(&m), run(&g));
}

/// FP8 checkpoints Axon doesn't run, refused by name.
#[test]
fn fp8_checkpoints_refused_cleanly() {
    let qcfg = |f: fn(&mut Value)| {
        move |d: &Dir| {
            let mut c = read_json(&d.0.join("config.json")).unwrap();
            f(&mut c["quantization_config"]);
            d.json("config.json", &c);
        }
    };
    type Case<'a> = (&'a str, Box<dyn Fn(&Dir)>, &'a str);
    let cases: Vec<Case> = vec![
        (
            "static",
            Box::new(qcfg(|q| {
                q["config_groups"]["group_0"]["input_activations"]["dynamic"] = json!(false)
            })),
            "dynamic",
        ),
        (
            "tensor",
            Box::new(qcfg(|q| {
                q["config_groups"]["group_0"]["weights"]["strategy"] = json!("tensor")
            })),
            "strategy",
        ),
        (
            "group",
            Box::new(qcfg(|q| {
                q["config_groups"]["group_0"]["weights"]["group_size"] = json!(128)
            })),
            "group_size",
        ),
        (
            "block",
            Box::new(qcfg(|q| {
                q["config_groups"]["group_0"]["weights"]["block_structure"] = json!([128, 128])
            })),
            "block_structure",
        ),
        (
            "int",
            Box::new(qcfg(|q| {
                q["config_groups"]["group_0"]["weights"]["type"] = json!("int")
            })),
            "type",
        ),
        (
            "kv",
            Box::new(qcfg(|q| q["kv_cache_scheme"] = json!({"num_bits": 8}))),
            "kv_cache_scheme",
        ),
        (
            "ignore",
            Box::new(qcfg(|q| {
                q["ignore"] = json!(["lm_head", "model.layers.0.mlp.down_proj"])
            })),
            "ignore",
        ),
        (
            "method",
            Box::new(qcfg(|q| q["quant_method"] = json!("fp8"))),
            "quant_method fp8",
        ),
        (
            "format",
            Box::new(qcfg(|q| q["format"] = json!("pack-quantized"))),
            "format",
        ),
    ];
    for (tag, edit, want) in cases {
        let d = Dir::new(&format!("fp8{tag}"));
        write_tiny_fp8(&d, |_| {});
        edit(&d);
        match load_dir(&d.0) {
            Ok(_) => panic!("{tag}: loaded"),
            Err(e) => assert!(e.to_string().contains(want), "{tag}: {e}"),
        }
    }
    type Edit = fn(&mut Vec<(String, &'static str, Vec<u64>, Vec<u8>)>);
    let tensor_cases: [(&str, Edit, &str); 8] = [
        (
            "noscale",
            |w| w.retain(|t| t.0 != "model.layers.1.mlp.up_proj.weight_scale"),
            "without its weight_scale",
        ),
        (
            "shape",
            |w| {
                let t = w
                    .iter_mut()
                    .find(|t| t.0 == "model.layers.0.self_attn.q_proj.weight_scale")
                    .unwrap();
                // 64 scales (of 1.0) for q_proj's 128 rows (hidden 128),
                // well-formed bytes: the count is what's wrong.
                t.2 = vec![1, 64];
                t.3 = [0x80u8, 0x3f].repeat(64);
            },
            "blk.0.attn_q.scale: shape [64, 1], want one scale per row (128)",
        ),
        (
            "grid",
            |w| {
                // 128 scales (the bytes unchanged), but as a grid: not one
                // per row.
                let t = w
                    .iter_mut()
                    .find(|t| t.0 == "model.layers.0.self_attn.q_proj.weight_scale")
                    .unwrap();
                t.2 = vec![16, 8];
            },
            "blk.0.attn_q.scale: shape [8, 16], want one scale per row (128)",
        ),
        (
            "bytes",
            |w| {
                // The header's shape and the data disagree.
                let t = w
                    .iter_mut()
                    .find(|t| t.0 == "model.layers.0.self_attn.q_proj.weight_scale")
                    .unwrap();
                t.3.truncate(2);
            },
            "q_proj.weight_scale: 2 bytes for shape",
        ),
        (
            "nan",
            |w| {
                w.iter_mut()
                    .find(|t| t.0 == "model.layers.0.mlp.down_proj.weight")
                    .unwrap()
                    .3[17] = 0x7f
            },
            "NaN code",
        ),
        (
            "negscale",
            |w| {
                let t = w
                    .iter_mut()
                    .find(|t| t.0 == "model.layers.0.mlp.gate_proj.weight_scale")
                    .unwrap();
                t.3[1] |= 0x80;
            },
            "finite and >= 0",
        ),
        (
            "bf16linear",
            |w| {
                let t = w
                    .iter_mut()
                    .find(|t| t.0 == "model.layers.1.self_attn.o_proj.weight")
                    .unwrap();
                t.1 = "BF16";
                t.3 = t.3.iter().flat_map(|_| [0u8, 0x3c]).collect();
                w.retain(|t| t.0 != "model.layers.1.self_attn.o_proj.weight_scale");
            },
            "every layer linear is F8_E4M3",
        ),
        (
            "orphan",
            |w| {
                w.push((
                    "model.layers.0.input_layernorm.weight_scale".into(),
                    "BF16",
                    vec![1],
                    vec![0, 0x3c],
                ))
            },
            "not a qwen2 tensor",
        ),
    ];
    for (tag, edit, want) in tensor_cases {
        let d = Dir::new(&format!("fp8t{tag}"));
        write_tiny_fp8(&d, edit);
        match load_dir(&d.0) {
            Ok(_) => panic!("{tag}: loaded"),
            Err(e) => assert!(e.to_string().contains(want), "{tag}: {e}"),
        }
    }
    // FP8 tensors without a quantization_config.
    let d = Dir::new("fp8nocfg");
    write_tiny_fp8(&d, |_| {});
    let mut c = read_json(&d.0.join("config.json")).unwrap();
    c.as_object_mut().unwrap().remove("quantization_config");
    d.json("config.json", &c);
    let e = load_dir(&d.0).err().expect("loaded").to_string();
    assert!(e.contains("without a quantization_config"), "{e}");
}
