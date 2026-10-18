//! Helpers shared by the unit tests.

use crate::gguf::{self, Model};
use std::path::{Path, PathBuf};

pub const MODEL: &str = "models/qwen2.5-0.5b-q4_k_m.gguf";
/// The qwen3 test model (`AXON_TEST_QWEN3` overrides it).
pub const QWEN3_MODEL: &str = "models/Qwen3-0.6B-Q8_0.gguf";
/// The llama test model (`AXON_TEST_LLAMA` overrides it): head_dim 64,
/// 4 query heads per kv head, tied output, Llama 3 RoPE factors.
pub const LLAMA_MODEL: &str = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
/// The optional SentencePiece test model (`AXON_TEST_MISTRAL` overrides it).
pub const MISTRAL_MODEL: &str = "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf";
pub const GEMMA3_MODEL: &str = "models/google_gemma-3-1b-it-Q8_0.gguf";

/// Reference data made by `scripts/oracle/make_fixtures.py`, committed to the repo.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Path of the test model (`AXON_TEST_MODEL` overrides MODEL; fixture-based
/// tests like matches_ollama only make sense for MODEL), or None to skip.
///
/// A missing model is a test failure, not a silent pass. CI doesn't have the
/// 470 MB file, so it opts out explicitly with `AXON_SKIP_MODEL_TESTS=1`.
pub fn model_path() -> Option<PathBuf> {
    let path = match std::env::var_os("AXON_TEST_MODEL") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL),
    };
    if !path.exists() {
        if std::env::var_os("AXON_SKIP_MODEL_TESTS").is_some() {
            eprintln!("skipping: {MODEL} not present and AXON_SKIP_MODEL_TESTS is set");
            return None;
        }
        panic!(
            "{MODEL} is missing. Download it (see CONTRIBUTING.md), or set \
             AXON_SKIP_MODEL_TESTS=1 to skip the tests that need it."
        );
    }
    Some(path)
}

/// The real model, for tests that need it.
pub fn model() -> Option<Model> {
    model_with(None)
}

/// `model`, copied into anonymous memory (Some(true)) or file-mapped
/// (Some(false)); None picks as `gguf::load` does.
pub fn model_with(copy: Option<bool>) -> Option<Model> {
    let path = model_path()?;
    Some(gguf::load_with(path.to_str().expect("utf-8 path"), copy).expect("load model"))
}

/// The qwen3 test model, or None to skip (`AXON_SKIP_MODEL_TESTS`, as
/// `model_path`).
pub fn qwen3_model() -> Option<Model> {
    let path = match std::env::var_os("AXON_TEST_QWEN3") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join(QWEN3_MODEL),
    };
    if !path.exists() {
        if std::env::var_os("AXON_SKIP_MODEL_TESTS").is_some() {
            eprintln!("skipping: {QWEN3_MODEL} not present and AXON_SKIP_MODEL_TESTS is set");
            return None;
        }
        panic!(
            "{} is missing. Download Qwen3-0.6B-Q8_0.gguf (Qwen/Qwen3-0.6B-GGUF), or set \
             AXON_SKIP_MODEL_TESTS=1 to skip the tests that need it.",
            path.display()
        );
    }
    Some(gguf::load(path.to_str().expect("utf-8 path")).expect("load qwen3 model"))
}

/// The llama test model, or None to skip (`AXON_SKIP_MODEL_TESTS`, as
/// `model_path`).
pub fn llama_model() -> Option<Model> {
    let path = llama_model_path()?;
    Some(gguf::load(path.to_str().expect("utf-8 path")).expect("load llama model"))
}

/// Path of the llama test model, or None to skip (`AXON_SKIP_MODEL_TESTS`,
/// as `model_path`).
pub fn llama_model_path() -> Option<PathBuf> {
    let path = match std::env::var_os("AXON_TEST_LLAMA") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join(LLAMA_MODEL),
    };
    if !path.exists() {
        if std::env::var_os("AXON_SKIP_MODEL_TESTS").is_some() {
            eprintln!("skipping: {LLAMA_MODEL} not present and AXON_SKIP_MODEL_TESTS is set");
            return None;
        }
        panic!(
            "{} is missing. Download Llama-3.2-1B-Instruct-Q4_K_M.gguf \
             (bartowski/Llama-3.2-1B-Instruct-GGUF), or set AXON_SKIP_MODEL_TESTS=1 to skip \
             the tests that need it.",
            path.display()
        );
    }
    Some(path)
}

/// The Mistral test model (`AXON_TEST_MISTRAL` overrides the path):
/// Mistral-7B-Instruct-v0.3 Q4_K_M, a SentencePiece vocabulary with
/// `[INST]` control tokens. Optional (4.4 GB): None, with a note, when it's
/// missing.
pub fn mistral_model() -> Option<Model> {
    let path = match std::env::var_os("AXON_TEST_MISTRAL") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join(MISTRAL_MODEL),
    };
    if !path.exists() {
        eprintln!(
            "skipping: {} not present (bartowski/Mistral-7B-Instruct-v0.3-GGUF, optional)",
            path.display()
        );
        return None;
    }
    Some(gguf::load(path.to_str().expect("utf-8 path")).expect("load mistral model"))
}

/// Path of the Gemma 3 test model (`AXON_TEST_GEMMA3` overrides it):
/// bartowski's google_gemma-3-1b-it Q8_0. Optional (1.1 GB): None, with a
/// note, when it's missing.
pub fn gemma3_model_path() -> Option<PathBuf> {
    let path = match std::env::var_os("AXON_TEST_GEMMA3") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join(GEMMA3_MODEL),
    };
    if !path.exists() {
        eprintln!(
            "skipping: {} not present (bartowski/google_gemma-3-1b-it-GGUF, optional)",
            path.display()
        );
        return None;
    }
    Some(path)
}

/// The Gemma 3 test model (`gemma3_model_path`), or None to skip.
pub fn gemma3_model() -> Option<Model> {
    let path = gemma3_model_path()?;
    Some(gguf::load(path.to_str().expect("utf-8 path")).expect("load gemma3 model"))
}

/// The qwen3moe test models (`AXON_TEST_MOE` / `AXON_TEST_MOE_SMALL`
/// override them). MOE_MODEL: mradermacher's Qwen3-30B-A3B-Instruct-Pruned-2B
/// Q4_K_M, 3 layers with the 30B's exact shapes and type mix (128 experts,
/// top-8, Q6_K down in layer 2); gibberish out, every kernel exercised.
/// MOE_SMALL_MODEL: Qwen3-MOE-4x0.6B Q4_K_M (4 experts, top-2, 28 layers),
/// coherent text. Optional (1.5 / 1.0 GB): None, with a note, when missing.
pub const MOE_MODEL: &str = "models/Qwen3-30B-A3B-Instruct-Pruned-2B.Q4_K_M.gguf";
pub const MOE_SMALL_MODEL: &str = "models/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2.Q4_K_M.gguf";

/// `MOE_MODEL` (small: `MOE_SMALL_MODEL`)'s path, or None to skip.
pub fn moe_model_path(small: bool) -> Option<PathBuf> {
    let (var, rel) = match small {
        false => ("AXON_TEST_MOE", MOE_MODEL),
        true => ("AXON_TEST_MOE_SMALL", MOE_SMALL_MODEL),
    };
    let path = match std::env::var_os(var) {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join(rel),
    };
    if !path.exists() {
        eprintln!("skipping: {} not present (optional)", path.display());
        return None;
    }
    Some(path)
}

/// The qwen3moe test model (`moe_model_path`), or None to skip.
pub fn moe_model(small: bool) -> Option<Model> {
    let path = moe_model_path(small)?;
    Some(gguf::load(path.to_str().expect("utf-8 path")).expect("load moe model"))
}

pub fn read_json(rel: &str) -> serde_json::Value {
    let path = fixtures().join(rel);
    let s = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The GPU, for tests that need one, or None to skip.
///
/// Like `model_path`: no usable GPU (no driver, no device, or it won't
/// open) is a test failure, not a silent pass. Machines without one (CI)
/// opt out explicitly with `AXON_SKIP_GPU_TESTS=1`.
///
/// GPU tests run one at a time: the calling test holds a process-wide lock
/// until its thread ends (libtest runs each test on its own thread). Two
/// reasons. On cards under 16 GiB (a 6 GB laptop 4050, a 12 GB 5070 Ti)
/// tests that load models at once run out of VRAM. And on 24 GiB RTX 4090
/// pods (EPYC hosts with the AMD IOMMU on), GPU tests in flight together in
/// one process lose the whole context to CUDA_ERROR_ILLEGAL_ADDRESS when
/// a process starts on the host (even a non-CUDA one), every later GPU test
/// with it (docs/log/2026-10-18-cuda-concurrency.md: no fault with one GPU
/// test at a time). `AXON_TEST_GPU_SERIAL=0` turns it off (hunting:
/// cuda::qwen2::tests::gpu_concurrency_matrix_hunt), `=1` is the default.
pub fn gpu() -> Option<crate::cuda::Gpu> {
    match crate::cuda::Gpu::open() {
        Ok(g) => {
            serialize_gpu(&g);
            Some(g)
        }
        Err(e) if std::env::var_os("AXON_SKIP_GPU_TESTS").is_some() => {
            eprintln!("skipping: no usable GPU ({e}) and AXON_SKIP_GPU_TESTS is set");
            None
        }
        Err(e) => {
            panic!("no usable GPU: {e}. Set AXON_SKIP_GPU_TESTS=1 to skip the tests that need one.")
        }
    }
}

/// Take the GPU test lock for the rest of this thread (see `gpu`).
/// Re-entrant: a thread already holding it does nothing.
fn serialize_gpu(g: &crate::cuda::Gpu) {
    use std::cell::RefCell;
    use std::sync::{Mutex, MutexGuard, OnceLock};
    static LOCK: Mutex<()> = Mutex::new(());
    static ON: OnceLock<bool> = OnceLock::new();
    thread_local!(static HELD: RefCell<Option<MutexGuard<'static, ()>>> = const { RefCell::new(None) });
    let on = *ON.get_or_init(|| {
        let total = g.info().map_or(u64::MAX, |i| i.total_bytes);
        let on = std::env::var("AXON_TEST_GPU_SERIAL").as_deref() != Ok("0");
        if on {
            // Straight to stderr: libtest captures eprintln! inside tests.
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "testutil: GPU has {:.2} GiB: serializing GPU tests (AXON_TEST_GPU_SERIAL=0 to disable)",
                total as f64 / (1u64 << 30) as f64
            );
        }
        on
    });
    if on {
        HELD.with(|h| {
            let mut h = h.borrow_mut();
            if h.is_none() {
                *h = Some(LOCK.lock().unwrap_or_else(|e| e.into_inner()));
            }
        });
    }
}

/// Two GPU tests never hold the GPU at once, whatever the card (unless
/// AXON_TEST_GPU_SERIAL=0): the second waits for the first's thread to end.
#[test]
fn gpu_tests_never_overlap() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    if std::env::var("AXON_TEST_GPU_SERIAL").as_deref() == Ok("0") {
        return;
    }
    let (held, first_in) = (AtomicBool::new(false), AtomicBool::new(false));
    std::thread::scope(|s| {
        s.spawn(|| {
            let Some(_g) = gpu() else {
                first_in.store(true, Ordering::SeqCst);
                return;
            };
            held.store(true, Ordering::SeqCst);
            first_in.store(true, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(300));
            held.store(false, Ordering::SeqCst);
            // The lock goes when this thread ends, just after.
        });
        while !first_in.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        s.spawn(|| {
            if gpu().is_some() {
                assert!(
                    !held.load(Ordering::SeqCst),
                    "a second GPU test got the GPU while the first held it"
                );
            }
        });
    });
}

#[test]
fn gpu_lock_is_reentrant() {
    let (Some(_a), Some(_b)) = (gpu(), gpu()) else {
        return;
    };
}
