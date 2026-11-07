<h1 align="center">Axon</h1>

<h3 align="center">LLM inference from scratch in Rust, on the CPU and on NVIDIA GPUs</h3>

<p align="center">
  <a href="https://github.com/HaseebKhalid1507/Axon/actions/workflows/ci.yml"><img src="https://github.com/HaseebKhalid1507/Axon/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/rust-1.89%2B-orange.svg" alt="Rust 1.89+"></a>
</p>

<p align="center">
  Run Qwen, Llama, Gemma, gpt-oss and DeepSeek-architecture models from one binary, on a laptop CPU or a datacenter
  GPU, with outputs that don't depend on batching, prompt chunking or drafting.
</p>

**Mission: make LLM inference cheap and fast, and run the best open models.** Cheap means wasting no hardware, from
a laptop to a datacenter: the most tokens per GPU, per watt and per dollar. Fast means as close to the hardware
limit as the math allows, for one stream and for thousands. The best models means the frontier open models, not
only the ones that fit a laptop.

**New here? Start with [docs/START-HERE.md](docs/START-HERE.md).**

---

## Install

Any x86-64 CPU (other targets build too, scalar only); NVIDIA GPUs on Linux. There are no release binaries yet:
build it with Rust 1.89 or newer.

```bash
git clone https://github.com/HaseebKhalid1507/Axon.git && cd Axon
cargo build --release          # one binary for the CPU and the GPU
./target/release/axon --version
```

Nothing CUDA is needed to build. For `--device cuda`, Axon loads the NVIDIA driver (CUDA 12 or newer) and NVRTC at
run time, compiles its kernels on first use and caches them. Details: [docs/models.md](docs/models.md#nvidia-gpus).

## Quick start

```bash
mkdir -p models
curl -fL -o models/qwen2.5-0.5b-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

./target/release/axon gen models/qwen2.5-0.5b-q4_k_m.gguf "The capital of France is" -n 32
./target/release/axon chat models/qwen2.5-0.5b-q4_k_m.gguf
./target/release/axon gen models/qwen2.5-0.5b-q4_k_m.gguf "The capital of France is" -n 32 --device cuda
```

`gen` streams the text to stdout and prints prompt and decode speed to stderr. `chat` is a conversation.

Serve it, and talk to it with any OpenAI or ollama client:

```bash
./target/release/axon serve models/qwen2.5-0.5b-q4_k_m.gguf --device cuda --port 8080
curl -N localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"stream":true}'
```

## Commands

| Command | What it does |
|---|---|
| `axon info <model>` | Print a summary of the model and exit |
| `axon gen <model> "<prompt>"` | Generate, greedy unless `--temp` > 0 |
| `axon logits <model> "<prompt>"` | The top-K next-token logits after the prompt |
| `axon chat <model>` | Talk to an instruct model in the terminal |
| `axon serve <model>` | The HTTP server: the OpenAI and ollama APIs |

Every flag and its default: [docs/usage.md](docs/usage.md). The `AXON_*` variables:
[docs/environment.md](docs/environment.md).

## What it does

- **CPU and GPU, one binary.** On the CPU it picks scalar, AVX2 or AVX-VNNI kernels at run time. On NVIDIA GPUs it
  replays each token as a captured CUDA graph, with int8 tensor cores for prompts (sm_80+). A model too big for the
  GPU is split across the CPU and the GPU, and `--gpus 0,1` splits one across GPUs. [More](docs/models.md#hardware)
- **The model families.** Qwen2 / Qwen2.5, Qwen3, Llama 3.x (and SentencePiece `llama` models such as Mistral 7B
  v0.3), Gemma 3 1B and 4B. Mixture of experts: Qwen3 MoE (Qwen3-30B-A3B), gpt-oss-20b, and the DeepSeek V2/V3 / Kimi
  K2 architecture (`deepseek2`, run on Moonlight-16B-A3B; nothing larger yet), with attention on the GPU and the
  experts on the CPU; Qwen3 MoE's and `deepseek2`'s experts can stay on the GPU too (`--gpu-experts`).
  [More](docs/models.md#architectures)
- **GGUF and safetensors, FP8 included.** GGUF files are read in place: F32, F16, Q4_0 to Q8_0, Q2_K to Q6_K and
  IQ4_NL (and gpt-oss's MXFP4 experts). Hugging Face checkpoint directories load too: BF16 / F16 / F32 on the CPU,
  and per-channel FP8 (`*-FP8-dynamic`) on the CPU and the GPU, as Q8_0 by default or natively on sm_89+ with
  `--numerics fast`. [More](docs/models.md#model-files-gguf-and-safetensors)
- **Bit-exact fast paths.** Batched prefill gives exactly the logits of feeding one token at a time, every SIMD
  kernel matches its scalar twin, and in `serve` on one GPU a request gets the same output batched as alone. Every
  response names what produced it (`system_fingerprint`). [More](docs/determinism.md)
- **`axon serve`.** OpenAI chat completions, completions and the Responses API, plus ollama's API, streamed or
  whole. Tool calling for Qwen3, Qwen2.5, Llama 3.1 / 3.2 and gpt-oss. On one GPU, continuous batching over a paged
  KV pool, with `--api-key`, `/metrics` and `/slots`. [More](docs/serving.md)
- **KV that outlives the request.** `chat --session` saves a conversation. A 29.8k-position 7B session restores in
  0.41–0.45 s instead of a 3.16 s re-prefill (RTX 4090). In `serve`, opt-in for now: a KV RAM tier (`--cache-ram`)
  and shared prefix pages (`AXON_SHARE_KV=1`). [More](docs/serving.md#operations-auth-metrics-slots)
- **Speculative drafting, lossless.** Prompt lookup (`--draft K`) or a small draft model (`--draft-model`): the same
  output with or without. With a draft model (K = 3), greedy decode ran +37 % on Qwen2.5-7B, +44 % on Qwen3-8B and
  +63 % on Llama-3.1-8B. Sampled, it gains little or nothing. [More](docs/usage.md#speculative-drafting)
- **Refuses instead of dying.** A load that can't fit in the RAM available (the host's or a cgroup limit's) is
  refused with the figures and a way out, not OOM-killed. A run capped within ~48 MiB above that edge can still be
  killed (#217). [More](docs/usage.md#memory-limits)
- **A library too.** `axon::load`, `Backend`, `Tokenizer`, `Lm`, `generate` and `serve`, in-process from Rust
  (0.x, not published yet). [More](docs/library.md)
- **Almost no dependencies.** Five crates. No ggml, no BLAS, no binding crates, no CUDA toolkit at build time: the
  driver and NVRTC are loaded with `dlopen`. [More](docs/how-it-works.md)

## Works with

`axon serve` speaks the APIs these clients already use. Each has been run against it:

- **OpenAI SDKs:** the `openai` Python library, or any OpenAI client with `base_url="http://127.0.0.1:8080/v1"`.
- **ollama clients:** the `ollama` CLI and the `ollama` Python library (`/api/chat`, `/api/generate`, `/api/tags`, ...).
- **codex:** through a `wire_api = "responses"` provider, or `codex --oss --local-provider ollama`.
- **opencode:** through `ollama-ai-provider-v2`. **pi:** through `openai-responses`.
- **Synaps:** a Synaps session on gpt-oss-20b, with working tool calls
  ([log](docs/log/2026-10-20-gpt-oss-tools.md)). Running Synaps on Axon is the
  [north star](docs/roadmap.md#mission-and-positioning).
- **crush:** with pi, opencode and codex, it ran coding tasks on Qwen2.5-7B through `axon serve` for the agent
  traffic captures ([log](docs/log/2026-11-04-agent-harness-traces.md)).

How to point each one at Axon: [docs/serving.md](docs/serving.md#the-responses-api-and-the-ollama-api). Runs and
transcripts: [docs/log/2026-10-17-serve-responses-ollama.md](docs/log/2026-10-17-serve-responses-ollama.md).

## Benchmarks

One RunPod RTX 4090, Q4_K_M GGUFs, tok/s, medians of 3 interleaved rounds. `pp2100` is a ~2,100-token prompt,
`tg128` decoding 128 tokens. **Bold** marks where Axon is behind.

| model | test | Axon | llama.cpp | ollama | Axon / llama.cpp |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | pp2100 | 111,300 | 62,816 | 51,584 | 1.77× |
| | tg128 | 1,042 | 875 | 671 | 1.19× |
| Qwen2.5-7B | pp2100 | 13,632 | 12,064 | 11,655 | 1.13× |
| | tg128 | 186 | 181 | 167 | 1.03× |
| Qwen3-8B | pp2100 | 12,450 | 11,710 | 11,340 | 1.06× |
| | tg128 | 170 | 165 | 153 | 1.03× |
| Qwen3-30B-A3B (MoE) | pp2100 | 12,212 | 9,289 | – | 1.31× |
| | tg128 | 219.7 | 252.5 | – | **0.87×** |
| Moonlight-16B-A3B (MoE) | pp2100 | 7,004 | 12,272 | – | **0.57×** |
| | tg128 | 272.9 | 285.5 | – | **0.96×** |

- **Dense rows:** Axon main `7f42c02` against llama.cpp `ff5a233` (`llama-bench -ngl 99`) and ollama 0.40.6,
  2026-10-18. On that board Axon led both in all 45 cells (RTX 4090, RTX 5090 and H100 SXM × five models × three
  tests). Its thinnest margins are the 4090 cells above: 7B and 8B decode (1.03×) and 8B pp2100 (1.06×).
- **MoE rows:** Axon main `8508a75`, every expert on the GPU, against llama.cpp `4413a19` at its best placement
  (also all on the GPU), 2026-10-26, job `1026-215154-40d7`, no ollama run. Since then #154 brought the 30B's
  decode to 241.6 against 254.6 (0.95×, job `1027-020521-6db3`).
- **Behind elsewhere:** serving at datacenter concurrency. On an H100 SXM (Qwen2.5-7B, 2026-10-19) Axon is the
  fastest of five engine and weight rows at one stream, but vLLM and SGLang push 1.8–3.8× its tokens at 128
  concurrent requests (two workloads). Also behind: Gemma 3 prefill (no tensor-core attention at its head size),
  MoE prefill with the experts on the CPU, and CPU decode on 1.5B–8B (tied with ollama, not ahead):
  [the full list](docs/results.md#known-losses-and-gaps-as-of-3014eeb).
- None of these boards ran on today's main. Merges since then changed some cells (prefill: #92, #100; MoE decode:
  #154).

Every board with its method, caveats and hardware: [docs/results.md](docs/results.md),
[docs/benchmarks-gpu.md](docs/benchmarks-gpu.md) and [BENCH.md](BENCH.md).

## Why Axon

- **Cheap means every machine counts.** The CPU and consumer GPUs are first-class targets, not fallbacks, and one
  binary covers them all.
- **Reproducible bits.** In the default `exact` numerics a reply doesn't change with what it's batched with, how the
  prompt is chunked or whether drafting is on, and its `system_fingerprint` names what produced it.
  [More](docs/determinism.md)
- **One binary, a one-second start.** No Python, no CUDA toolkit. With the shipped kernel cache it starts in 1.0 s,
  against 42–71 s for vLLM and SGLang on the same H100.
- **Measured in the open.** Each board names its commit, hardware and job, and what lost or was dropped is written
  down too: [docs/experiments-rejected.md](docs/experiments-rejected.md).

**Not for you if:**
- you serve many concurrent users on datacenter GPUs. Use vLLM or SGLang: on that H100 they push 1.8–3.8× Axon's
  tokens at 128 requests, with lower per-token latency from 8 up, and they have far more models, multi-node and
  LoRA, with years of production use.
- you need tensor parallel, or a model outside the families above.
- you want a settled tool. Axon is 0.x, has one maintainer and no external users yet.

## Docs

- **Start here:** [docs/START-HERE.md](docs/START-HERE.md): what's on main, the design contract, a map of the source.
- **Use it:** [usage](docs/usage.md), [models and hardware](docs/models.md), [serving](docs/serving.md) and its
  [routes and fields](docs/serve-apis.md), [environment variables](docs/environment.md), [the library](docs/library.md).
- **How it works:** [docs/how-it-works.md](docs/how-it-works.md), [determinism](docs/determinism.md),
  [long context](docs/long-context.md), [design notes](docs/design/).
- **Numbers:** [results](docs/results.md), [GPU boards](docs/benchmarks-gpu.md), [CPU boards](BENCH.md).
- **The project:** [roadmap](docs/roadmap.md), [methodology](docs/methodology.md), [history](docs/history.md),
  [testing](docs/testing.md), [hardening](docs/hardening.md), the [CHANGELOG](CHANGELOG.md).

## Contributing

Contributions from people and from their AI agents are welcome. The work queue is the issue board.

```bash
cargo build && cargo test      # the model tests need the GGUFs listed in CONTRIBUTING.md
AXON_SKIP_MODEL_TESTS=1 AXON_SKIP_GPU_TESTS=1 cargo test     # or skip them
```

- **[AGENTS.md](AGENTS.md)** first: how to claim an `agent-ready` issue, the hard rules (bit identity, the md5 gate
  against `main`, no new crates), how to measure, and the evidence every PR must include.
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: setup, test models, oracles, the checks to run.

## License

MIT, see [LICENSE](LICENSE). Parts of the kernels and the tokenizer follow llama.cpp/ggml (MIT); test fixtures
contain rows of Qwen2.5 weights (Apache-2.0). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
