//! The public API from outside the crate (#118): what examples/embed.rs and
//! an embedder use, compiled against the library as another crate would.
//! `Backend` and `BackendCache` are opaque, `Lm` is sealed, `Device`,
//! `KvType`, `Numerics` and `GenStats` are non-exhaustive (the
//! `compile_fail` doc tests in src/lm.rs and src/lib.rs check the other
//! side: what no longer compiles).
//!
//! The model tests use models/qwen2.5-0.5b-q4_k_m.gguf (CONTRIBUTING.md);
//! `AXON_SKIP_MODEL_TESTS=1` skips them, `AXON_SKIP_GPU_TESTS=1` the CUDA one.

use axon::{
    Backend, BackendCache, BackendOptions, Device, GenOpts, GenStats, KvType, Lm, Model, Numerics,
    Tokenizer, Utf8Stream,
};

const MODEL: &str = "models/qwen2.5-0.5b-q4_k_m.gguf";
/// The second model of the swap test (another architecture: qwen3);
/// `AXON_TEST_QWEN3` overrides it, as for the crate's qwen3 tests.
const MODEL2: &str = "models/Qwen3-0.6B-Q8_0.gguf";
const PROMPT: &str = "The capital of France is";
const N: usize = 12;

fn model_path(model: &str) -> Option<String> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(model);
    if p.exists() {
        return Some(p.to_str().expect("utf-8 path").to_string());
    }
    if std::env::var_os("AXON_SKIP_MODEL_TESTS").is_some() {
        eprintln!("skipping: {model} not present and AXON_SKIP_MODEL_TESTS is set");
        return None;
    }
    panic!("{model} is missing (CONTRIBUTING.md), or set AXON_SKIP_MODEL_TESTS=1");
}

/// The names an embedder needs, with their types as the API gives them
/// (compile-time coverage: never run).
#[allow(dead_code)]
fn surface(m: &Model, net: &Backend, cache: &mut BackendCache, tok: &Tokenizer) {
    let _: Result<Model, String> = axon::load("x.gguf", Device::Cpu);
    let _: Result<Backend, String> = Backend::new(m, BackendOptions::default());
    let o: BackendOptions = BackendOptions::default()
        .device(Device::Cuda)
        .max_ctx(64)
        .gpu_layers(12)
        .gpu_layers(None)
        .gpu_layers(Some(3))
        .kv(KvType::Q8)
        .fit(true)
        .numerics(Numerics::Fast);
    let _: Result<Backend, String> = Backend::new(m, o);
    let _: Numerics = net.numerics();
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
    assert_eq!(Numerics::parse("exact"), Ok(Numerics::Exact));
    assert_eq!(Numerics::parse("fast"), Ok(Numerics::Fast));
    assert!(Numerics::parse("approx").is_err());
    assert_eq!(Numerics::default(), Numerics::Exact);
    let mode = |n: Numerics| match n {
        Numerics::Exact => "exact",
        Numerics::Fast => "fast",
        _ => "other",
    };
    assert_eq!(mode(Numerics::Fast), "fast");
}

/// examples/embed.rs's path (load, tokenize, prefill, `generate`) of the
/// model at `file` on one device, and the same tokens from a decode loop
/// over `Lm`'s own calls: the generated ids, or None (skipped). `extra`
/// context beyond what the run needs and the KV type are the backend's
/// options. The model and its backend are dropped on return.
fn gen_on(file: &str, device: Device, extra: usize, kv: KvType) -> Option<Vec<u32>> {
    let path = model_path(file)?;
    let model = axon::load(&path, device).expect("load");
    let tok = Tokenizer::from_gguf(&model);
    let ids = tok.encode_prompt(PROMPT);
    let ctx = ids.len() + N + 1;
    let o = BackendOptions::default()
        .device(device)
        .max_ctx(ctx + extra)
        .kv(kv);
    let net = match Backend::new(&model, o) {
        Ok(n) => n,
        Err(e) if device == Device::Cuda && std::env::var_os("AXON_SKIP_GPU_TESTS").is_some() => {
            eprintln!("skipping: no usable GPU ({e}) and AXON_SKIP_GPU_TESTS is set");
            return None;
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
    eprintln!("{device:?} {}: {PROMPT}{text}", model.arch());
    Some(st.ids)
}

/// An embedder's model swap, as a determinism check (not a leak test:
/// nothing here measures memory): a backend, dropped, then a second
/// `Backend::new` with another model and other options (a longer context;
/// on CUDA a q8 cache) in the same process, which generates; then the
/// first model again with its first options, which must give the same
/// tokens as the first time, so nothing the second load set process-wide
/// changed it.
fn swap_on(device: Device) {
    let Some(first) = gen_on(MODEL, device, 0, KvType::F16) else {
        return;
    };
    let second = std::env::var("AXON_TEST_QWEN3").unwrap_or_else(|_| MODEL2.to_string());
    // q8 is CUDA-only; the CPU's f16/f32 choice is process-wide (f16 here).
    let kv = if device == Device::Cuda {
        KvType::Q8
    } else {
        KvType::F16
    };
    if gen_on(&second, device, 1000, kv).is_none() {
        return;
    }
    let again = gen_on(MODEL, device, 0, KvType::F16).expect("the first model, loaded again");
    assert_eq!(
        again, first,
        "a reload after a swap gives the same tokens ({device:?})"
    );
}

#[test]
fn gen_cpu() {
    gen_on(MODEL, Device::Cpu, 0, KvType::F16);
}

#[test]
fn gen_cuda() {
    gen_on(MODEL, Device::Cuda, 0, KvType::F16);
}

#[test]
fn second_backend_cpu() {
    swap_on(Device::Cpu);
}

#[test]
fn second_backend_cuda() {
    swap_on(Device::Cuda);
}
