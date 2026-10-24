//! The public API from outside the crate (#118): what examples/embed.rs and
//! an embedder use, compiled against the library as another crate would.
//! `Backend` and `BackendCache` are opaque, `Lm` is sealed, `Device`,
//! `KvType` and `GenStats` are non-exhaustive (the `compile_fail` doc tests
//! in src/lm.rs and src/lib.rs check the other side: what no longer
//! compiles).
//!
//! The model tests use models/qwen2.5-0.5b-q4_k_m.gguf (CONTRIBUTING.md);
//! `AXON_SKIP_MODEL_TESTS=1` skips them, `AXON_SKIP_GPU_TESTS=1` the CUDA one.

use axon::{
    Backend, BackendCache, Device, GenOpts, GenStats, KvType, Lm, Model, Tokenizer, Utf8Stream,
};

const MODEL: &str = "models/qwen2.5-0.5b-q4_k_m.gguf";
const PROMPT: &str = "The capital of France is";
const N: usize = 12;

fn model_path() -> Option<String> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL);
    if p.exists() {
        return Some(p.to_str().expect("utf-8 path").to_string());
    }
    if std::env::var_os("AXON_SKIP_MODEL_TESTS").is_some() {
        eprintln!("skipping: {MODEL} not present and AXON_SKIP_MODEL_TESTS is set");
        return None;
    }
    panic!("{MODEL} is missing (CONTRIBUTING.md), or set AXON_SKIP_MODEL_TESTS=1");
}

/// The names an embedder needs, with their types as the API gives them
/// (compile-time coverage: never run).
#[allow(dead_code)]
fn surface(m: &Model, net: &Backend, cache: &mut BackendCache, tok: &Tokenizer) {
    let _: Result<Model, String> = axon::load("x.gguf", Device::Cpu);
    let _: Result<Backend, String> = Backend::new(m, Device::Cpu, 64, None, KvType::F16, false);
    let _: usize = net.ctx(64);
    let _: usize = net.vocab();
    let _: usize = net.prefill_batch();
    let _: bool = net.decode_streams();
    let _: BackendCache = net.new_cache(64);
    let _: Vec<f32> = net.forward(1, 0, cache);
    let _: u32 = net.forward_argmax(1, 0, cache);
    net.feed_batch(&[1, 2], 0, cache);
    let _: Vec<u32> = net.verify_argmax(&[1, 2], 0, cache);
    let _: Vec<std::ops::Range<usize>> = net.prompt_chunks(2, 0, cache);
    let _: Vec<f32> = net.prefill_logits(&[1, 2], 0, cache);
    let _: Vec<u32> = tok.encode_prompt("x");
    let _: Vec<u32> = tok.eog();
    let _: String = tok.decode(&[1]);
    let _: Vec<u8> = tok.decode_bytes(&[1]);
    let _: String = Utf8Stream::default().push(b"x");
    let _: (&str, u32, u32) = (m.arch(), m.n_layers(), m.ctx_len());
    let st: GenStats = GenStats::default();
    let _: (&Vec<u32>, usize, usize, usize, usize) =
        (&st.ids, st.passes, st.decided, st.drafted, st.accepted);
    let _: Result<(), String> = axon::serve(&[]).map_err(|e| e.to_string());
}

/// `Device` and `KvType` from outside: parsed as the CLI parses them, and
/// matched with a wildcard arm (they're non-exhaustive).
#[test]
fn device_and_kv_parse() {
    assert_eq!(Device::parse("cpu"), Ok(Device::Cpu));
    assert_eq!(Device::parse("cuda"), Ok(Device::Cuda));
    assert!(Device::parse("tpu").is_err());
    assert_eq!(KvType::parse(""), Ok(KvType::F16));
    assert_eq!(KvType::parse("q8"), Ok(KvType::Q8));
    assert!(KvType::parse("f8").is_err());
    let name = |d: Device| match d {
        Device::Cpu => "cpu",
        Device::Cuda => "cuda",
        _ => "other",
    };
    assert_eq!(name(Device::Cpu), "cpu");
    let bytes = |k: KvType| match k {
        KvType::F16 => 2,
        KvType::F32 => 4,
        KvType::Q8 => 1,
        _ => 0,
    };
    assert_eq!(bytes(KvType::F32), 4);
}

/// examples/embed.rs's path (load, tokenize, prefill, `generate`) on one
/// device, and the same tokens from a decode loop over `Lm`'s own calls.
fn gen_on(device: Device) {
    let Some(path) = model_path() else { return };
    let model = axon::load(&path, device).expect("load");
    let tok = Tokenizer::from_gguf(&model);
    let ids = tok.encode_prompt(PROMPT);
    let ctx = ids.len() + N + 1;
    let net = match Backend::new(&model, device, ctx, None, KvType::F16, false) {
        Ok(n) => n,
        Err(e) if device == Device::Cuda && std::env::var_os("AXON_SKIP_GPU_TESTS").is_some() => {
            eprintln!("skipping: no usable GPU ({e}) and AXON_SKIP_GPU_TESTS is set");
            return;
        }
        Err(e) => panic!("Backend::new: {e}"),
    };
    assert!(net.ctx(ctx) > ids.len());
    assert!(net.vocab() >= tok.vocab_len());

    let mut cache = net.new_cache(ctx);
    let logits = net.prefill_logits(&ids, 0, &mut cache);
    let stop: Vec<u32> = Vec::new();
    let mut streamed = Vec::new();
    let st = axon::generate(
        &net,
        &mut cache,
        &ids,
        logits,
        GenOpts::greedy(N, &stop),
        &mut |_, _| Vec::new(),
        &mut |id| {
            streamed.push(id);
            Ok(())
        },
    )
    .expect("generate");
    assert_eq!(st.ids.len(), N);
    assert_eq!(st.ids, streamed);

    // The same greedy tokens from `feed_batch` + `forward_argmax` steps.
    let mut c2 = net.new_cache(ctx);
    net.feed_batch(&ids[..ids.len() - 1], 0, &mut c2);
    let mut t = *ids.last().unwrap();
    let mut got = Vec::new();
    for i in 0..N {
        t = net.forward_argmax(t, ids.len() - 1 + i, &mut c2);
        got.push(t);
    }
    assert_eq!(
        got, st.ids,
        "generate == forward_argmax steps on {device:?}"
    );
    let text = tok.decode(&st.ids);
    assert!(!text.is_empty());
    eprintln!("{device:?}: {PROMPT}{text}");
}

#[test]
fn gen_cpu() {
    gen_on(Device::Cpu);
}

#[test]
fn gen_cuda() {
    gen_on(Device::Cuda);
}
