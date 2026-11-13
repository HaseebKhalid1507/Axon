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

**Mission: Axon is for everyone, the standard for running local AI**, as Synaps should be the standard agent runtime:
inference made cheap and fast. For now the target is the best local backend for coding agents on NVIDIA PCs,
Linux then Windows, installed with one command ([roadmap](docs/roadmap.md)).

**New here? Start with [docs/START-HERE.md](docs/START-HERE.md).**

---

## Install

Linux x86-64, one command; it puts `axon` in `~/.local/bin` after checking the release tarball's SHA-256:

```bash
curl -fsSL https://raw.githubusercontent.com/HaseebKhalid1507/Axon/main/install.sh | sh
```

It runs on any x86-64 CPU, and on an NVIDIA GPU with nothing but the driver, **R580 or newer** (CUDA 13): the GPU
kernels are built in for RTX 20 through RTX 50, A100 and H100. On an older driver, `--device cuda` stops with one
line naming the driver it found and that it needs R580+: update the driver, or run on the CPU (`--device cpu`, the
default). Details: [docs/usage.md](docs/usage.md#install-a-release-linux-x86-64).

From source (any x86-64 CPU, other targets scalar only; NVIDIA GPUs on Linux), with Rust 1.89 or newer:

```bash
git clone https://github.com/HaseebKhalid1507/Axon.git && cd Axon
cargo build --release          # one binary for the CPU and the GPU: ./target/release/axon
```

Nothing CUDA is needed to build. A source build's `--device cuda` loads the driver (CUDA 12 or newer) and the CUDA
toolkit's NVRTC at run time, compiles its kernels on first use and caches them. Details:
[docs/models.md](docs/models.md#nvidia-gpus).

## Quick start

```bash
axon pull qwen2.5-0.5b         # 0.49 GB into ~/.local/share/axon/models; axon pull --list names the others
axon gen qwen2.5-0.5b "The capital of France is" -n 32
axon chat qwen2.5-0.5b
axon gen qwen2.5-0.5b "The capital of France is" -n 32 --device cuda
```

`gen` streams the text to stdout and prints prompt and decode speed to stderr. `chat` is a conversation. Any GGUF on
Hugging Face: `axon pull OWNER/REPO/FILE.gguf`; a path to a `.gguf` works wherever a name does.

Serve it, and talk to it with any OpenAI or ollama client:

```bash
axon serve qwen2.5-0.5b --device cuda --port 8080
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
| `axon pull <name>` | Download a GGUF from Hugging Face, checked against its SHA-256 |

Every flag and its default: [docs/usage.md](docs/usage.md). The `AXON_*` variables:
[docs/environment.md](docs/environment.md).

## What it does

- **CPU and GPU, one binary.** AVX2 and AVX-VNNI on the CPU, CUDA graphs and tensor cores on NVIDIA GPUs. Big dense
  Qwen and Llama models split across the CPU and GPUs. [More](docs/models.md#hardware)
- **The model families.** Qwen2.5, Qwen3, Llama 3, Mistral and Gemma 3, plus the mixture-of-experts Qwen3-30B-A3B,
  gpt-oss-20b and Moonlight, DeepSeek V3's architecture. [More](docs/models.md#architectures)
- **GGUF and safetensors, FP8 included.** GGUFs from Q2_K to F16. Hugging Face checkpoints: BF16 on the CPU,
  FP8-dynamic on the CPU and the GPU. [More](docs/models.md#model-files-gguf-and-safetensors)
- **Bit-exact fast paths.** By default, batching, prompt chunking and drafting never change a reply's bits, and every
  response names what produced it. [More](docs/determinism.md)
- **`axon serve`.** The OpenAI chat, Responses and ollama APIs, with tool calling. Continuous batching for dense
  models on one GPU; MoE serves one request at a time. [More](docs/serving.md)
- **KV that outlives the request.** `chat --session` restores a 29.8k-token 7B conversation in 0.41–0.45 s instead
  of a 3.16 s re-prefill, on an RTX 4090. [More](docs/usage.md#sessions)
- **Speculative drafting, lossless.** Prompt lookup or a draft model, same output. A draft model sped up greedy 7B
  and 8B decode by 37–63 %; sampled, it gains little. [More](docs/usage.md#speculative-drafting)
- **Refuses instead of dying.** A load that won't fit in RAM or its cgroup limit is refused with the numbers, not
  OOM-killed. One with under ~48 MiB to spare can still be killed. [More](docs/usage.md#memory-limits)
- **A library too.** Load, place and run a model in-process from Rust: `axon::load`, `Backend`, `generate`, `serve`.
  Version 0.x, not on crates.io yet. [More](docs/library.md)
- **Almost no dependencies.** Five crates. No ggml, no BLAS, no CUDA toolkit at build time: the driver and NVRTC
  load at run time. [More](docs/how-it-works.md)

## Works with

`axon serve` speaks the APIs these clients already use. Each has been run against it:

- **OpenAI SDKs:** the `openai` Python library, or any OpenAI client with `base_url="http://127.0.0.1:8080/v1"`.
- **ollama clients:** the `ollama` CLI and the `ollama` Python library.
- **codex:** through a `wire_api = "responses"` provider, or `codex --oss --local-provider ollama`.
- **opencode:** through `ollama-ai-provider-v2`. **pi:** through `openai-responses`.
- **Synaps:** a Synaps session on gpt-oss-20b, with working tool calls
  ([log](docs/log/2026-10-20-gpt-oss-tools.md)). Running Synaps on Axon waits for a
  local Synaps profile ([roadmap](docs/roadmap.md#shelved-for-now)).
- **crush:** with pi, opencode and codex, it ran coding tasks on Qwen2.5-7B through `axon serve` for the agent
  traffic captures ([log](docs/log/2026-11-04-agent-harness-traces.md)).

How to point each one at Axon: [docs/serving.md](docs/serving.md#the-responses-api-and-the-ollama-api). Runs and
transcripts: [docs/log/2026-10-17-serve-responses-ollama.md](docs/log/2026-10-17-serve-responses-ollama.md).

## Benchmarks

Tokens per second on an RTX 4090 (RunPod) unless marked, medians of interleaved rounds. `pp2100` is a ~2,100-token
prompt, `tg128` decoding 128 tokens. **Bold** marks where Axon is behind.

| model | test | Axon | llama.cpp | ollama | Axon / llama.cpp |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | pp2100 | 111,300 | 62,816 | 51,584 | 1.77× |
| | tg128 | 1,042 | 875 | 671 | 1.19× |
| Qwen2.5-7B | pp2100 | 13,632 | 12,064 | 11,655 | 1.13× |
| | tg128 | 186 | 181 | 167 | 1.03× |
| Qwen3-8B | pp2100 | 12,450 | 11,710 | 11,340 | 1.06× |
| | tg128 | 170 | 165 | 153 | 1.03× |
| Gemma 3 4B | pp2100 | 8,782 | 17,920 | 3,436 | **0.49×** |
| | tg128 | 257 | 221 | 97 | 1.16× |
| Qwen3-30B-A3B (MoE) | pp2100 | 12,212 | 9,289 | – | 1.31× |
| | tg128 | 219.7 | 252.5 | – | **0.87×** |
| Moonlight-16B-A3B (MoE) | pp2100 | 7,004 | 12,272 | – | **0.57×** |
| | tg128 | 272.9 | 285.5 | – | **0.96×** |
| gpt-oss-20b (MoE) † | pp512 | 265 | 995 | – | **0.27×** |
| | tg128 | 46.6 | 23.4 | – | 1.99× |

Dense rows: 2026-10-18, llama.cpp `ff5a233`, ollama 0.40.6. Gemma: #35's own run, 10-16, ollama 0.35. MoE rows:
10-26, llama.cpp `4413a19`, every expert on the GPU; #154 has since brought the 30B's decode to 0.95×. † A laptop
RTX 5070 Ti, experts on the CPU for both engines, Axon's prompt 486 tokens (#98, 10-18). None of these ran on
today's main. Method, caveats and every known loss:
[docs/results.md](docs/results.md#known-losses-and-gaps-as-of-3014eeb).

**Serving:** on an H100 SXM (Qwen2.5-7B, 10-19) Axon is level with the fastest int4 engines at one stream (+1–3 %,
inside one pod's noise). From 8 concurrent requests up vLLM and SGLang, on AWQ and FP8 weights, lead: 1.8–3.8×
Axon's tokens at 128. On an RTX 4090, vLLM-AWQ is 13–24 % ahead at 8–32 requests (Qwen2.5-7B, 10-18).
[Serving boards](docs/results.md#serving-runpod-h100-sxm-main-c15c90e-2026-10-19)

## Why Axon

- **Cheap means every machine counts.** The CPU and consumer GPUs are first-class targets, not fallbacks, and one
  binary covers them all.
- **Reproducible bits.** By default a reply doesn't change with what it's batched with, how the prompt is chunked
  or whether drafting is on, and its `system_fingerprint` names what produced it. [More](docs/determinism.md)
- **One binary, a one-second start.** No Python. Once its kernels are cached it starts in 1.0 s, against 42–71 s
  for vLLM and SGLang on the same H100. The first GPU run compiles them with NVRTC, from a CUDA toolkit for now (a
  start with only the driver is [planned](docs/roadmap.md#week-1-11-12-to-11-18)).
- **Measured in the open.** Each board names its commit, hardware and job, and what lost or was dropped is written
  down too: [docs/experiments-rejected.md](docs/experiments-rejected.md).

**Not for you if:**
- you serve many concurrent users. Use vLLM or SGLang: they push 1.8–3.8× Axon's tokens at 128 requests on an
  H100 and vLLM-AWQ is 13–24 % ahead at 8–32 on an RTX 4090, and they have far more models, multi-node and LoRA, with
  years of production use.
- you need tensor parallel, batched MoE serving, or a model outside the families above.
- you want a settled tool. Axon is 0.x, has one maintainer and no external users yet.

## Docs

- **Start here:** [docs/START-HERE.md](docs/START-HERE.md): what's on main, the design contract, a map of the source.
- **Use it:** [usage](docs/usage.md), [models and hardware](docs/models.md), [serving](docs/serving.md) and its
  [routes and fields](docs/serve-apis.md), [environment variables](docs/environment.md),
  [the library](docs/library.md).
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
