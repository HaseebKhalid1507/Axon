# Contributing to Axon

## Setup

```bash
git clone git@github.com:HaseebKhalid1507/Axon.git
cd Axon
cargo build
cargo test
```

## The test model

Some tests run the real model and need it at `models/qwen2.5-0.5b-q4_k_m.gguf`
(491 MB, git-ignored). It's Qwen's own Q4_K_M build of Qwen2.5-0.5B-Instruct:

```bash
mkdir -p models
curl -L -o models/qwen2.5-0.5b-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
sha256sum models/qwen2.5-0.5b-q4_k_m.gguf
# 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
```

The qwen3 tests also need Qwen3-0.6B (Qwen's Q8_0 build, 639 MB) at
`models/Qwen3-0.6B-Q8_0.gguf` (`AXON_TEST_QWEN3` points them at another qwen3
file); the GPU tests use `models/Qwen3-4B-Q4_K_M.gguf` too if it's there:

```bash
curl -L -o models/Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
sha256sum models/Qwen3-0.6B-Q8_0.gguf
# 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
```

Without them those tests fail on purpose, so a missing model can't pass for a
working one. To skip them, set `AXON_SKIP_MODEL_TESTS=1`. CI does this.

## The GPU tests

The CUDA tests (`src/cuda/`) need an NVIDIA GPU with its driver
(`libcuda.so.1`) and NVRTC from the CUDA toolkit (`libnvrtc.so`, found on the
library path, in `$CUDA_PATH/lib64`, `/opt/cuda/lib64` or
`/usr/local/cuda/lib64`). Nothing CUDA is linked: axon builds and runs
without them. Like a missing model, a missing GPU makes those tests fail; to
skip them, set `AXON_SKIP_GPU_TESTS=1`. CI does this.

Kernels are compiled for the GPU on first use and cached in
`$XDG_CACHE_HOME/axon/cuda` (else `~/.cache/axon/cuda`); deleting it is safe.

The tests compare axon against reference data in `tests/fixtures/`: gguf-py's
dequantizer, Hugging Face's tokenizer, and Ollama's greedy output on CPU.
`scripts/oracle/make_fixtures.py` regenerates it (setup in its header).

## Before submitting

```bash
cargo fmt
cargo clippy
cargo test
```

## Pull requests

Open a PR against `main`. CI must pass. Haseeb merges.
