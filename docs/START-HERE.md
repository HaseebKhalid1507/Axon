# Start here

A short orientation for new collaborators. Everything under "What works on main" was checked against
the source at main `5f96039`. Anything that isn't on main yet is marked **planned** or **in flight**.

## What Axon is

Axon is an LLM inference engine for GGUF models (Qwen2, Qwen3, Llama 3), written from scratch in Rust.
It runs on the CPU and on NVIDIA GPUs. It doesn't use ggml, BLAS, binding crates, or a CUDA toolkit at
build time. The CPU path is a GGUF reader, integer dot-product kernels (scalar, AVX2, AVX-VNNI), a
spinning worker pool and a transformer forward pass. On the GPU, CUDA C++ kernels are compiled by NVRTC
at run time and launched through the driver API. Both libraries are loaded with `dlopen`. `src/` holds
about 42k lines of Rust and 4.7k of CUDA, tests included (`wc -l`). Single-stream speed is ahead of
llama.cpp and ollama in 45 of 45 cells on RunPod RTX 4090 / RTX 5090 / H100 SXM, and ahead of ollama
in 23 of 24 cells on an RTX 4050 laptop ([benchmarks-gpu.md](benchmarks-gpu.md)).

## What works on main

| area | on main | where |
|---|---|---|
| Architectures | `qwen2`, `qwen3` (dense), `llama` (Llama 3.x, including RoPE frequency factors). Other architectures, MoE and RoPE scaling are refused with an error. | `model.rs` (`ARCHS`) |
| Tensor types | CPU: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K. GPU: every weight matrix must be one of the eight quantized types. | `quant.rs`, `cuda/kernels.cu` |
| CPU SIMD | One baseline x86-64 binary. It picks scalar, AVX2 or AVX-VNNI kernels at run time with the same output bits as a `target-cpu` build (`AXON_SIMD` forces a set). Q4_K/Q6_K/Q5_0/Q8_0 are repacked into 8-row interleaved layouts. | `quant.rs`, `quant/r8.rs`, `quant/r8q.rs`, `mv.rs`, `vmath.rs` |
| CUDA | `--device cuda`, Linux. Loads `libcuda.so.1` and NVRTC at run time, compiles the kernels for the GPU on first use and caches them. Int8 tensor-core GEMMs and attention need sm_80+; older GPUs fall back to dp4a decode kernels for batches. A GPU is refused if the installed NVRTC can't target it (`pick_target`). One token is a captured CUDA graph, replayed. | `cuda/mod.rs`, `cuda/ffi.rs`, `cuda/qwen2.rs` |
| Commands | `info`, `gen`, `logits` (top-K next-token logits), `chat` (terminal, chat template, KV reuse across turns), `serve` (OpenAI-compatible HTTP). | `main.rs`, `chat.rs`, `serve.rs` |
| Sampling | Greedy, or temperature / top-k / top-p / min-p / repeat penalty with a seed (llama.cpp's order). On the GPU the penalty and top-k (k ≤ 256) run on the device. | `sample.rs`, `cuda/sample.cu` |
| Speculative decoding | Prompt lookup (`--draft K`) and a draft model (`--draft-model small.gguf`). Both are lossless: same output with or without, greedy or seeded sampling. | `spec.rs`, `draft.rs` |
| CPU + GPU split | A model too big for VRAM puts its first layers on the CPU and the rest plus the head on the GPU (`--gpu-layers N`). For prompts of ≥ 32 tokens the CPU layers' weights are streamed through the GPU. | `hybrid.rs`, `cuda/qwen2/stream.rs` |
| KV cache | f16 by default; `--kv q8` (GPU only) halves it with slightly different results. | `kv.rs` |
| Serving | `axon serve`: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, SSE streaming. Runs **one request at a time** (stage S1); the KV cache carries over between requests for shared prefixes. | `serve.rs`, `serve/` |

**Not on main:** tool calling. `serve` refuses `tools`, `functions` and messages carrying `tool_calls`
with a 400 (`serve/api.rs`). Also absent: concurrent request batching, multiple GPUs, Gemma, and
SentencePiece-tokenizer models (Mistral, Llama 2).

The full flag and environment reference is in the [README](../README.md#usage).

## The design contract: bit identity

Every optimized path gives exactly the same bits as the plain path on the same device:

- **decode == batched prefill == speculative verify.** A token's logits and KV-cache rows are
  identical whether it was fed one at a time (`forward`), in a prompt batch of any size or position
  (`feed_batch`), or inside a verify pass (`verify_argmax`, `verify_cands`). The GPU batched kernels
  follow the decode kernels' "canonical order": exact integer block sums, then one fixed-order fma
  chain per output. See the headers of `cuda/decode.cu` ("Row dot products"), `cuda/prefill.cu` and
  `cuda/qwen2/batch.rs`.
- **SIMD == scalar** on the CPU, at any thread count or chunking. One hardware property changes CPU
  bits: whether the CPU has FMA (`vmath.rs`).
- **Deterministic.** Every float reduction has a fixed order and uses no atomics (`cuda/decode.cu`
  header). The same input gives the same bytes on every run.
- **GPU ≠ CPU bit for bit.** The two agree to f32 rounding: the GPU splits its float sums differently.
  The tests bound the gap (KL, top-token agreement) instead of requiring equality.

Why this matters:

- Chat's KV reuse equals a fresh prefill, and `serve`'s prefix reuse equals running the request alone.
  Greedy `/v1/completions` output is byte-identical to `axon gen`.
- Speculative decoding is lossless by construction: accepted tokens are exactly what plain decoding
  would have picked.
- A kernel change can be checked with md5s of generated text instead of tolerances. Changes are
  gated this way.
- It extends to future work. Continuous batching, pipeline-parallel prefill and prefill/decode
  disaggregation can't drift numerically if every layer still runs the same per-row arithmetic
  ([design/multi-gpu.md](design/multi-gpu.md)).

Cross-machine evidence (MEASURED, main `f8d8a86`): CPU output from 12 `gen` runs (4 models × 3 prompts)
had matching md5s across two machines and five kernel-path runs. **Parabellum** (dev laptop: Intel Core
Ultra 9 275HX with AVX-VNNI, plus an RTX 5070 Ti Laptop GPU, sm_120, 12 GB) ran the VNNI, AVX2 and
scalar paths. **Jaded** (homelab box: AVX2-only i7-9750H, GTX 1050 sm_61) ran the AVX2 and scalar
paths.

The tests also compare against three independent oracles in `tests/fixtures/`: gguf-py's dequantizer,
Hugging Face's tokenizer, and Ollama's greedy output.

## Source map

Suggested reading order: `gguf.rs` → `quant.rs` → `model.rs` → `main.rs`, then `lm.rs` → `cuda/`.

| file | what it is |
|---|---|
| `src/main.rs` | CLI: argument parsing, backend choice, `run_lm` (prefill in batches, then generate). |
| `src/gguf.rs` | GGUF reader: header, metadata, tensor table. Treats the file as untrusted input. |
| `src/quant.rs` | Block formats, dequantization, `dot_*_q8` kernels (scalar, AVX2, AVX-VNNI). |
| `src/quant/r8.rs`, `r8q.rs` | 8-row interleaved repacks (Q4_K/Q6_K; Q5_0/Q8_0) with integer-first GEMV/GEMM kernels. |
| `src/model.rs` | CPU transformer: `hidden()` (one token, numbered steps) and `feed_batch` (a prompt batch). |
| `src/pool.rs` | Persistent spinning worker pool; balances P- and E-cores. |
| `src/mv.rs`, `src/vmath.rs` | Run-time multiversioning (baseline vs x86-64-v3), and FMA/exp/softmax/SiLU with identical bits on SIMD and scalar paths. |
| `src/kv.rs` | KV cache storage (f16 / f32; q8 lives on the GPU). |
| `src/tokenizer.rs` | GPT-2 byte-level BPE with the qwen2 and llama-bpe pre-tokenizers. |
| `src/sample.rs` | Sampling (candidates from the backend, then the draw) and the generic `generate` loop. |
| `src/spec.rs`, `src/draft.rs` | Prompt-lookup drafting; draft-model drafting with its own KV cache. |
| `src/lm.rs` | The `Lm` trait (`forward`, `feed_batch`, `verify_argmax`, `*_cands`) and `Backend` (CPU / GPU / split). |
| `src/hybrid.rs` | CPU+GPU layer split (first layers on the CPU), plus `most_gpu_layers` placement. |
| `src/chat.rs`, `src/lineedit.rs` | `axon chat`: templates (ChatML, Llama 3, Qwen3 thinking) and the raw-mode line editor. |
| `src/serve.rs` | `axon serve` engine thread and job queue (S1: one request at a time). |
| `src/serve/http.rs`, `serve/api.rs` | Hand-written HTTP/1.1 server; OpenAI request/response mapping. |
| `src/cuda/ffi.rs` | Driver API + NVRTC via `dlopen`/`dlsym` (nothing linked). |
| `src/cuda/mod.rs` | `Gpu`, `DevBuf`, NVRTC compile + on-disk kernel cache, target choice (`pick_target`). |
| `src/cuda/qwen2.rs` | `CudaQwen2`: weights in GGUF block format in one VRAM arena, decode via CUDA graphs, VRAM planning. |
| `src/cuda/qwen2/batch.rs` | Batched passes (prefill, verify) on int8 tensor cores, bit-equal to decode. |
| `src/cuda/qwen2/stream.rs` | Streaming the CPU layers' weights through the GPU for prefill of a split model. |
| `src/cuda/kernels.cu` | Dequant kernels, bit-identical to `quant.rs`. |
| `src/cuda/decode.cu` | One-token kernels: GEMV (dp4a), RMSNorm, RoPE, attention; defines the canonical order. |
| `src/cuda/prefill.cu` | Batched kernels: `gemm2*` (mma.sync int8), tensor-core attention, `gemv_ref`. |
| `src/cuda/sample.cu`, `stream.cu` | Device-side penalty + top-k; inverse of the CPU repack for streaming. |
| `src/**/tests.rs`, `src/testutil.rs` | Bit-identity, oracle and end-to-end tests; test model paths. |
| `scripts/bench.py`, `sweep.py` | CPU benchmark harness ([BENCH.md](../BENCH.md)). |
| `scripts/bench_serve.py`, `serve_check.py` | Serving benchmark; end-to-end check of `serve` against `gen`. |
| `scripts/oracle/make_fixtures.py` | Regenerates `tests/fixtures/`. |

## Build and test

From [CONTRIBUTING.md](../CONTRIBUTING.md). Requires Rust 1.89 or newer; CI pins 1.95.0.

```bash
cargo build --release          # same binary for CPU and GPU; nothing CUDA needed at build time
cargo fmt && cargo clippy && cargo test
```

- Model tests need `models/qwen2.5-0.5b-q4_k_m.gguf`, `models/Qwen3-0.6B-Q8_0.gguf` and
  `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`. Download commands and sha256 sums are in CONTRIBUTING.
  A missing model fails its tests on purpose. Set `AXON_SKIP_MODEL_TESTS=1` to skip them.
- GPU tests need a driver and NVRTC. Set `AXON_SKIP_GPU_TESTS=1` to skip them. CI sets both skip
  variables and also runs the suite on the scalar path and under a qemu Nehalem CPU, so no AVX
  instruction can slip into the baseline build.
- Kernel cache: `$XDG_CACHE_HOME/axon/cuda` (else `~/.cache/axon/cuda`); it is safe to delete.

## In flight (not on main)

Each item is a branch or design, not merged:

- **S2 continuous batching for `serve`.** Many requests share one decode step (slot KV cache, per-row
  positions, per-row sampling). Aggregate target ESTIMATED at ~5× single-stream on 7B at 8 clients.
- **Gemma 3.** Head dim 256, sliding-window layers, GeGLU, 262k vocab. Main refuses `gemma` today.
- **Startup round 2.** Pinned staging for weight upload and overlap of upload with prefill, to cut GPU
  time-to-first-token.
- **Multi-GPU, pipeline parallel first.** Stage 1 is in progress ([design/multi-gpu.md](design/multi-gpu.md)).
- **Hopper wgmma/TMA prefill path for H100.** Design only; the first pod experiment is pending
  ([design/hopper.md](design/hopper.md)).
- **Old-GPU (sm_61, Pascal) fixes**, tested on Jaded's GTX 1050. Main has no tested path for sm_61.

## Other docs

- [benchmarks-gpu.md](benchmarks-gpu.md): RunPod RTX 4090 / RTX 5090 / H100 SXM and RTX 4050 laptop
  results, with caveats.
- [design/multi-gpu.md](design/multi-gpu.md): pipeline parallel, then tensor parallel; bit-identity analysis.
- [design/hopper.md](design/hopper.md): why the H100 trails on prefill, and the wgmma plan.
- [roadmap.md](roadmap.md): priorities, and what is not planned.
- [../BENCH.md](../BENCH.md): CPU numbers and the Parabellum GPU board; [../README.md](../README.md):
  full usage.
