# Axon 0.1.0: release notes (draft)

> **Draft.** Not tagged, not published. Tagging, the date and the GitHub release are the maintainer's call; this
> page is what the release would say. Full list of changes: [CHANGELOG.md](../CHANGELOG.md).

## In one sentence

Equal to the fastest int4 engines at one stream; the lowest ITL p99 on the mixed long-prompt workload at C = 128
(that cell only; worst on the short-prompt workload from C = 8 up); reproducible bits; 1.8–3.8× behind on
throughput at C = 128 today. (All on one board: Qwen2.5-7B on an H100 SXM, 2026-10-19; [README](../README.md#where-it-stands).)

## What it is

LLM inference written from scratch in Rust for the CPU and NVIDIA GPUs: one binary, no Python, no CUDA toolkit at
build time (CUDA C++ kernels compiled at run time by NVRTC, launched through the driver API), no ggml, no BLAS.
Since 0.1 it is also a library: `axon::load`, `Backend::new`, `Tokenizer`, the `Lm` trait, `axon::generate`
(greedy, with optional lossless drafting; sampling stays the CLI's) and `axon::serve` let a Rust program run a model
in-process ([examples/embed.rs](../examples/embed.rs)). Not yet hardened for embedders: see Known limits.

- **Models:** Qwen2 / Qwen2.5, Qwen3, Llama 3.x, Mistral 7B, Gemma 3 1B/4B (dense, CPU and CUDA); qwen3moe,
  gpt-oss-20b and DeepSeek V2/V3 / Kimi K2 / Moonlight architectures (attention on the GPU, experts on the CPU);
  Hugging Face safetensors (CPU). Quantizations from Q2_K to F32, plus MXFP4 and IQ4_NL.
- **Commands:** `info`, `gen`, `logits`, `chat` (with exact session save/restore) and `serve` (OpenAI chat and
  completions with tool calling, `/v1/responses`, the ollama API, continuous batching, chunked prefill, paged KV,
  metrics).
- **GPUs:** sm_61 to sm_120, Hopper wgmma/TMA prefill, CPU/GPU split, multi-GPU pipeline, 8-bit KV.

## The contract

Decode == batched prefill == speculative verify, bit for bit, on each device; reusing a cache == a fresh prefill;
greedy output doesn't change with drafting, batching or prompt chunking. Every response carries a
`system_fingerprint` of what decides its bits ([docs/determinism.md](determinism.md)). The tests check this
against three independent oracles (gguf-py, Hugging Face's tokenizer, Ollama's greedy output).

## Numbers, honestly

- **Single stream:** ahead of llama.cpp and ollama in all 45 cells of the 2026-10-18 RunPod board (RTX 4090,
  RTX 5090, H100 SXM; Qwen2.5 0.5B–7B and Qwen3 8B Q4_K_M; pp560 / pp2100 / tg128); the thinnest margins are
  +3 % (7B and Qwen3 8B decode on the 4090). [results.md](results.md).
- **Serving on an H100 SXM** (Qwen2.5-7B, 2026-10-19, against vLLM 0.32.0 and SGLang 0.5.22 with AWQ and FP8; *mix*
  = chat plus 2k–7.6k-token prompts, *classic* = 519-token prompts, 512 out):
  - ahead at one stream: *mix* C = 1 287 tok/s vs 284 SGLang-AWQ (inside one pod's uncertainty) and 235 vLLM-AWQ;
    ITL p99 lowest of the five at C = 1 on both workloads;
  - ITL p99 lowest at *mix* C = 128 only (77.5 ms vs 126–320 ms): Axon's steps there are slow but even;
  - ITL p99 **highest** of the five on *classic* from C = 8 up (C = 128: 62.0 vs 20.2–39.7 ms) and on *mix* at
    C = 8 (58.2 vs 7.6–10.7 ms); second at *mix* C = 32 and 64;
  - aggregate throughput **2.0–3.8× behind on *mix* and 1.8–3.2× on *classic* at C = 128** (*mix* 1,868 vs
    3,770–7,146 tok/s), behind at every C ≥ 8, with worse TPOT and TTFT p99 at C ≥ 32;
  - cold start 1.0 s with the shipped kernel cache, vs 42–71 s.

  Axon has no FP8 path, so the rows run different weights. [README](../README.md#where-it-stands),
  [results.md](results.md#serving-runpod-h100-sxm-main-c15c90e-2026-10-19).

## Known limits

- Throughput at high concurrency on datacenter GPUs (above); the plan is in [roadmap.md](roadmap.md).
- MoE experts run on the CPU; no MoE CPU/GPU layer split, multi-GPU or MoE draft model.
- Gemma 3 on the GPU must fit whole; 27B is refused.
- CUDA is Linux only. Built and tested on x86-64 Linux only (scalar, AVX2 and AVX-VNNI kernels picked at run
  time); other targets have scalar code paths but no CI.
- The library API is 0.x, small, and not hardened for embedders
  ([#118](https://github.com/HaseebKhalid1507/Axon/issues/118)): library calls print notes to stderr; `AXON_*`
  variables are read once per process (the f16/f32 KV choice is process-wide); each CUDA `Backend::new` leaks its
  context and kernels; `axon::serve` exits the process on a usage error (2) or an engine failure (70); `generate` is
  greedy only.
- One maintainer, no external users yet: please open issues.

## Install

```bash
git clone https://github.com/HaseebKhalid1507/Axon.git && cd Axon
cargo build --release            # target/release/axon
cargo run --release --example embed -- model.gguf "The capital of France is" 32
```

Rust 1.89 or newer. For an NVIDIA GPU: the driver and an NVRTC (`libnvrtc.so`) at run time.

## For the maintainer (not done by this PR)

- [ ] Pick the commit (main after the lib split merges) and the date; fill both into CHANGELOG.md.
- [ ] `git tag -s v0.1.0 <sha> && git push origin v0.1.0`.
- [ ] GitHub release from this page. No `cargo publish` under the name `axon`: it's taken on crates.io
      (aep/axon); pick a package name first (`[lib] name = "axon"` can stay).
