//! Helpers shared by the unit tests.

use crate::gguf::{self, Model};
use std::path::{Path, PathBuf};

pub const MODEL: &str = "models/qwen2.5-0.5b-q4_k_m.gguf";

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
pub fn gpu() -> Option<crate::cuda::Gpu> {
    match crate::cuda::Gpu::open() {
        Ok(g) => Some(g),
        Err(e) if std::env::var_os("AXON_SKIP_GPU_TESTS").is_some() => {
            eprintln!("skipping: no usable GPU ({e}) and AXON_SKIP_GPU_TESTS is set");
            None
        }
        Err(e) => {
            panic!("no usable GPU: {e}. Set AXON_SKIP_GPU_TESTS=1 to skip the tests that need one.")
        }
    }
}
