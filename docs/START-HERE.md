# Start here

A short orientation for new collaborators. Everything under "What works on main" was checked against
the source at main `5f96039`; the Architectures, CUDA and Serving rows and the multi-GPU row were updated at
main `a591e17` from the merged PRs (#33, #34, #35, #38, #42), and the table was brought up to main `39624db`
(2026-10-25) from the PRs merged since (listed in [history.md](history.md), Phases 17–20). Anything that isn't on
main yet is marked **planned** or **in flight**.

**North star (owner, 10-26): run Synaps on Axon**, many warm agent sessions and subagents on one box, then many
users: [roadmap](roadmap.md#mission-and-positioning), [design/agent-workload.md](design/agent-workload.md).

How Axon is developed, what was tried (including what failed) and where it stands:
[methodology.md](methodology.md), [history.md](history.md),
[experiments-rejected.md](experiments-rejected.md), [results.md](results.md).

## What Axon is

**Mission: make LLM inference cheap and fast, and run the best open models.** Cheap means wasting no
hardware, from a laptop to a datacenter: the most tokens per GPU, per watt and per dollar. Fast means as close
to the hardware limit as the math allows, for one stream and for thousands. The best models means the frontier
open models, not only the ones that fit a laptop.

Axon is an LLM inference engine for GGUF models (Qwen2, Qwen3, Llama 3, Gemma 3, and the Qwen3 MoE, gpt-oss and
DeepSeek V2/V3 / Kimi K2 architectures) and Hugging Face safetensors checkpoints, written from scratch in Rust.
It runs on the CPU and on NVIDIA GPUs. It doesn't use ggml, BLAS, binding crates, or a CUDA toolkit at
build time. The CPU path is a GGUF reader, integer dot-product kernels (scalar, AVX2, AVX-VNNI), a
spinning worker pool and a transformer forward pass. On the GPU, CUDA C++ kernels are compiled by NVRTC
at run time and launched through the driver API. Both libraries are loaded with `dlopen`. The repo holds
about 99k lines of Rust and 8.4k of CUDA, tests included (`git ls-files` + `wc -l` at `39624db`; 738 `#[test]`s
in `src/`). Single-stream speed is ahead of llama.cpp and ollama in 45 of 45 cells on RunPod RTX 4090 / RTX 5090 /
H100 SXM (main 7f42c02), and ahead of ollama in 23 of 24 cells on an RTX 4050 laptop
([benchmarks-gpu.md](benchmarks-gpu.md)). **Serving at datacenter concurrency is behind:** on an H100 SXM (10-19)
Axon is the fastest at one stream but 2.0–3.8× behind vLLM / SGLang at 128 concurrent requests
([results.md](results.md#serving-runpod-h100-sxm-main-c15c90e-2026-10-19)).

## What works on main

| area | on main | where |
|---|---|---|
| Architectures | `qwen2`, `qwen3` (dense), `gemma3` (1B, 4B; 27B refused; no q8 KV, split, streaming or multi-GPU for it), `llama` (Llama 3.x, including RoPE frequency factors; SentencePiece-tokenizer `llama` models such as Mistral 7B v0.3 and Llama 2, with Mistral's `[INST]` chat format). `qwen3moe` (Qwen3-30B-A3B's architecture), `gpt-oss` (gpt-oss-20b, MXFP4 experts) and `deepseek2` (MLA; DeepSeek V2/V3, Kimi K2, Moonlight): on the CPU, or with `--device cuda` as llama.cpp's `--cpu-moe` layout (attention, router, shared experts and head on the GPU, routed experts on the CPU; #67, #98, #89). deepseek2's routed experts can also be resident on the GPU, layer by layer, same bits (`--gpu-experts N\|auto`, opt-in; #127). YaRN as an opt-in override (#49). Other architectures are refused with an error. | `model.rs` (`ARCHS`), `model/moe.rs`, `model/deepseek2.rs`, `moe.rs` |
| Model files | GGUF, or a Hugging Face checkpoint directory of a `qwen2` / `qwen3` / `llama` model (BF16 / F16 / F32; BF16 runs as the converter's F16 GGUF, bit for bit). CPU only for checkpoints (#102). | `gguf.rs`, `safetensors.rs` |
| Tensor types | CPU: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL. GPU: every weight matrix must be one of the eleven quantized types. | `quant.rs`, `quant/k23.rs`, `quant/iq4nl.rs`, `cuda/kernels.cu` |
| CPU SIMD | One baseline x86-64 binary. It picks scalar, AVX2 or AVX-VNNI kernels at run time with the same output bits as a `target-cpu` build (`AXON_SIMD` forces a set). Q4_K/Q6_K/Q5_0/Q8_0 are repacked into 8-row interleaved layouts. | `quant.rs`, `quant/r8.rs`, `quant/r8q.rs`, `mv.rs`, `vmath.rs` |
| CUDA | `--device cuda`, Linux. Loads `libcuda.so.1` and NVRTC at run time, compiles the kernels for the GPU on first use and caches them. Int8 tensor-core GEMMs and attention need sm_80+; older GPUs fall back to dp4a decode kernels for batches. A GPU is refused if no installed NVRTC can target it (older NVRTCs are tried, #39). One token is a captured CUDA graph, replayed. On H100/H200 (sm_90a) the fused gate+up prefill GEMM runs on wgmma + TMA (#33). | `cuda/mod.rs`, `cuda/ffi.rs`, `cuda/qwen2.rs` |
| Commands | `info`, `gen`, `logits` (top-K next-token logits), `chat` (terminal, chat template, KV reuse across turns), `serve` (OpenAI-compatible HTTP). | `cli.rs`, `chat.rs`, `serve.rs` |
| Library | `axon::load`, `Backend::new` (`BackendOptions`), `Tokenizer`, the sealed `Lm` trait, `generate` (greedy, optional lossless drafting) and `serve`, in-process; `Backend` is opaque. 0.x, not tagged, not yet hardened for embedders (#118) (#117, #124). | `lib.rs`, `generate.rs`, `examples/embed.rs` |
| Sampling | Greedy, or temperature / top-k / top-p / min-p / repeat penalty with a seed (llama.cpp's order). On the GPU the penalty and top-k (k ≤ 256) run on the device. | `sample.rs`, `cuda/sample.cu` |
| Speculative decoding | Prompt lookup (`--draft K`) and a draft model (`--draft-model small.gguf`). Both are lossless: same output with or without, greedy or seeded sampling. | `spec.rs`, `draft.rs` |
| CPU + GPU split | A model too big for VRAM puts its first layers on the CPU and the rest plus the head on the GPU (`--gpu-layers N`). For prompts of ≥ 32 tokens the CPU layers' weights are streamed through the GPU. | `hybrid.rs`, `cuda/qwen2/stream.rs` |
| KV cache | f16 by default; `--kv q8` (GPU only) halves it with slightly different results. In `serve` on one GPU, a paged pool: 64-position pages shared by every slot, admission and preemption by pages (#64, #71). | `kv.rs`, `cuda/qwen2/pages.rs` |
| Serving | `axon serve`: `/v1/chat/completions` (with OpenAI tool calling: `tools`, `tool_choice`, `tool_calls`, for Qwen3, Qwen2.5, Llama 3.1/3.2 and gpt-oss templates), `/v1/completions`, `/v1/models`, `/health`, SSE streaming. On one GPU, decode rows of concurrent requests run in one batched step (`--parallel N`; #38), each request's output equal to running it alone; long prompts are fed in chunks between the decode steps (#45). Also OpenAI's Responses API and ollama's native API (#70; [serve-apis.md](serve-apis.md)), and `--api-key`, `/metrics`, `/slots` (#61). Each step's prompt chunk is sized from the measured step cost (`--step-ms 60`); admission is first come first served, shortest-first opt-in (`--admit-order sjf`; #131). On the CPU, a CPU/GPU split, a multi-GPU pipeline or a MoE model, requests run one at a time. The KV cache carries over for shared prefixes. Default context: the model's trained length where it fits (#42). | `serve.rs`, `serve/`, `serve/engine.rs`, `serve/tools.rs` |
| Multi-GPU | `--gpus 0,1`: pipeline parallel, layers split across GPUs in order, bit-identical to one GPU. P2P is opt-in and untested (`AXON_P2P=1`). No tensor parallel (#34). | `cuda/qwen2/pipeline.rs` |

**Not on main:** tensor parallel, FP8 weights, qwen3moe / gpt-oss experts on the GPU (deepseek2's can be, #127), streaming MoE experts for prefill, `serve` slot save/restore, batched
speculative decoding in `serve` (#90 closed by its kill rule), a decode-shaped Hopper GEMM. The legacy `functions`
field is still a 400.

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

Suggested reading order: `gguf.rs` → `quant.rs` → `model.rs` → `cli.rs` + `generate.rs`, then `lm.rs` → `cuda/`.

| file | what it is |
|---|---|
| `src/lib.rs` | The crate root: module list and the public library API (`load`, `Backend`, `Tokenizer`, `Lm`, `generate`, `serve`; `examples/embed.rs`). |
| `src/main.rs` | The `axon` binary: calls `axon::cli::main`, nothing else. |
| `src/cli.rs` | CLI: argument parsing, backend choice, `run_lm` (prefill in batches, then generate). |
| `src/generate.rs` | `generate`, the greedy / drafted decode loop (`sample.rs` for sampling), and `argmax`. |
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
| `src/serve.rs`, `src/serve/engine.rs` | `axon serve`: the engine thread, the batched pool (continuous batching, chunked prefill, page-based admission) and the single-stream path. |
| `src/serve/http.rs`, `serve/api.rs` | Hand-written HTTP/1.1 server; OpenAI request/response mapping. |
| `src/cuda/ffi.rs` | Driver API + NVRTC via `dlopen`/`dlsym` (nothing linked). |
| `src/cuda/mod.rs` | `Gpu`, `DevBuf`, NVRTC compile + on-disk kernel cache, target choice (`pick_target`). |
| `src/cuda/qwen2.rs` | `CudaQwen2`: weights in GGUF block format in one VRAM arena, decode via CUDA graphs, VRAM planning. |
| `src/cuda/qwen2/batch.rs` | Batched passes (prefill, verify) on int8 tensor cores, bit-equal to decode. |
| `src/cuda/qwen2/stream.rs` | Streaming the CPU layers' weights through the GPU for prefill of a split model. |
| `src/cuda/qwen2/pages.rs`, `slots.rs` | The paged KV pool: page lists, the device page table, slot views. |
| `src/moe.rs`, `src/cuda/qwen2/moe.rs`, `mla.rs` | MoE on the GPU with the experts on the CPU; deepseek2's MLA attention. |
| `src/safetensors.rs` | Hugging Face checkpoint directories as GGUF tensors and metadata. |
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
- Where the suite's time goes, how to time it, and the fast / full test tiers for PR rounds:
  [testing.md](testing.md).

## In flight (not on main)

As of main `39624db` (2026-10-25). None of these is on main; each is open work:

- **The H100 serving gap**, the roadmap's proposed order (from an outside review; the owner decides;
  [roadmap](roadmap.md#next-proposed-order-outside-review-10-19-owner-decides), [reviews/](reviews/README.md)):
  - **H100 step anatomy** (item 1 (a)): the kit is draft PR #125 (`AXON_SERVE_TRACE`, gemmh2 variants,
    `pool_feed_cost`), rehearsed on bella; the pod session waits for approval.
  - **The shortest-first default on the H100** (#139): #131 shipped the measured step budget and kept
    `--admit-order sjf` opt-in after it missed its kill rule on the laptop cells; one H100 session at mix C = 64 /
    128 decides whether it becomes the default. Not booked.
  - Later in the order: an exact Hopper GEMM (item 2), FP8 under `fast` if that GEMM misses (item 4).
- **One MoE resident on the GPU, next steps** (item 3; PRs A–C are on main for deepseek2):
  - the **Moonlight bench of record** against llama.cpp (`--n-cpu-moe`), since #127's ×2.06 / ×3.40 was against
    Axon's own CPU-expert path and TAINTED;
  - **Qwen3-30B-A3B step 0**: the resident path for `qwen3moe`, which has none yet.
- **Hardening:** #135 (a dense CPU load under a memory limit is OOM-killed instead of refused). #137 (an explicit
  `--parallel N` silently serving fewer slots) is fixed by #141: it's refused at start.
- **Library API for embedders:** the rest of #118 (stderr from library code, env-only config, leaks, `serve`'s
  exits). The v0.1.0 tag is Haseeb's call.
- **Determinism modes** (`--numerics exact | fast`): an ADR merged as PROPOSED
  ([design/determinism-modes.md](design/determinism-modes.md)); nothing is built until it is decided.
- **Batched speculative decoding in `serve`:** the first attempt (#90) is closed; a second (latency-guarded gate v2,
  then multi-row verify) is designed ([research/spec-gate-v2-20261024.md](research/spec-gate-v2-20261024.md)), below
  the line in the proposed order.
- **FP8 weights** (safetensors stage B), after stage A (#102).

## Other docs

- [benchmarks-gpu.md](benchmarks-gpu.md): RunPod RTX 4090 / RTX 5090 / H100 SXM and RTX 4050 laptop
  results, the serving boards (RTX 4090 and H100 SXM vs vLLM / SGLang), with caveats.
- [design/multi-gpu.md](design/multi-gpu.md): pipeline parallel, then tensor parallel; bit-identity analysis.
- [design/hopper.md](design/hopper.md): why the H100 trails on prefill, and the wgmma plan.
- [design/determinism-modes.md](design/determinism-modes.md): the `exact` / `fast` numerics ADR (PROPOSED).
- [determinism.md](determinism.md): what is bit-identical, and the identity tuple behind `system_fingerprint`.
- [serve-apis.md](serve-apis.md): `axon serve`'s routes (chat completions, completions, Responses, ollama).
- [testing.md](testing.md): the test tiers and what to run when.
- [roadmap.md](roadmap.md): priorities, and what is not planned.
- [methodology.md](methodology.md): how work is picked, built, reviewed, gated and benchmarked.
- [hardening.md](hardening.md): fuzzers, the multi-hour serve soak and the model-zoo sweep: method and results.
- [history.md](history.md): every round and PR, with why, how it was verified, and before/after numbers.
- [experiments-rejected.md](experiments-rejected.md): NO-GOs and regressions, with numbers. Read before starting a round.
- [results.md](results.md): current standing vs ollama and llama.cpp, with dates, commits and hosts.
- [research/](research/README.md): the round-10 research reports and scope.
- [reviews/](reviews/README.md): outside reviews (curated; the reviewer's opinions, Haseeb's decisions).
- [log/](log/README.md): one engineering-log entry per PR, NO-GO, board or merge train.
- [../BENCH.md](../BENCH.md): CPU numbers and the Parabellum GPU board; [../README.md](../README.md):
  full usage.
