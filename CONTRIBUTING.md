# Contributing to Axon

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems go through
[SECURITY.md](SECURITY.md), not public issues.

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
curl -fL -o models/qwen2.5-0.5b-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
sha256sum models/qwen2.5-0.5b-q4_k_m.gguf
# 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
```

The qwen3 tests also need Qwen3-0.6B (Qwen's Q8_0 build, 639 MB) at
`models/Qwen3-0.6B-Q8_0.gguf` (`AXON_TEST_QWEN3` points them at another qwen3
file); the GPU tests use `models/Qwen3-4B-Q4_K_M.gguf` too if it's there:

```bash
curl -fL -o models/Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
sha256sum models/Qwen3-0.6B-Q8_0.gguf
# 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
```

The llama tests need Llama-3.2-1B-Instruct (bartowski's Q4_K_M, 808 MB) at
`models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (`AXON_TEST_LLAMA` points them at another
llama file); the GPU tests use the 3B and Llama-3.1-8B Q4_K_M builds too if they're there:

```bash
curl -fL -o models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
sha256sum models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
# 6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83
```

With a GPU, the FP8 checkpoint tests also need RedHatAI's FP8 build of
Qwen2.5-0.5B-Instruct (9 files, 877 MiB), a Hugging Face directory at
`models/hf/Qwen2.5-0.5B-Instruct-FP8-dynamic` (`AXON_TEST_FP8_MODEL` points them at
another copy). `curl -f` fails on an HTTP error instead of saving the error page,
and every file has its sha256 (each line should say `OK`):

```bash
d=models/hf/Qwen2.5-0.5B-Instruct-FP8-dynamic; mkdir -p $d
for f in added_tokens.json config.json generation_config.json merges.txt model.safetensors \
         special_tokens_map.json tokenizer_config.json tokenizer.json vocab.json; do
  curl -fL -o $d/$f https://huggingface.co/RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic/resolve/main/$f
done
(cd $d && sha256sum -c) <<'EOF'
58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b  added_tokens.json
3fae208d3ae1a0faeea0767ccb30d3d744f413fd3d74bb9c9a04c36c6f716d94  config.json
b87d5b053ff14b30842cb40fc4f7bf08b09047fa27ed65fb942ea6943e9a895b  generation_config.json
8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5  merges.txt
5c804f9499421e548f50f5db893035377b2c8b807ecdef1226710ab001f3b2b0  model.safetensors
76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd  special_tokens_map.json
7e88129d9769a0b14b1587a7d5e829fe93ac0e1511636471fdfc0811951418e6  tokenizer_config.json
913950e4971737031da511cdd1b410daae4566f62eb845b3975bca5a102323d8  tokenizer.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json
EOF
```

Without them those tests fail on purpose, so a missing model can't pass for a
working one. To skip them, set `AXON_SKIP_MODEL_TESTS=1`. CI does this. To skip
only the FP8 checkpoint's tests, set `AXON_SKIP_FP8_MODEL_TESTS=1`.

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
