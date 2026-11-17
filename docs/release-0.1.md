# Axon 0.1.0: release notes (draft)

> **Draft.** Not tagged, not published. Tagging, the date and the GitHub release are the maintainer's call. The
> release's notes are this page from "In one sentence" to the end of "Known limits": not this banner, not the
> maintainer's list at the end. Full list of changes: [CHANGELOG.md](../CHANGELOG.md).
>
> The release candidate is main `9f3630c`'s binary (2026-11-14), from the release workflow's dry run; the boards
> below run on it. Main has merged only scripts and docs since (#235, #276): the dry run on main `1a1d751`
> (2026-11-17) built the same `axon`, byte for byte. The tag can't be `9f3630c`: this PR (#270) and the CHANGELOG's
> date land after it, so the tag is a descendant with no change under `src/`, whose binary should be the same
> bytes. If a PR that changes `src/` merges first (#271, #274 are open), the binary differs. The maintainer checks that before publishing
> (releasing.md, "The boards' binary"); if it differs, the board lines say they measured the candidate, not the
> release binary.

## In one sentence

One Linux x86-64 binary, installed with one command, that runs GGUF models on the CPU or on an NVIDIA GPU with only
the driver, serves them OpenAI-style to coding agents, and gives the same output bits however it batches, drafts or
reuses a cache (except under `--numerics fast`, and on a CPU/GPU split that streams its prefill, which is on by
default when a model doesn't fit the GPU: [The contract](#the-contract)). The numbers below are each tied to a date,
a commit and a GPU; none of the older ones were re-run on the candidate.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/HaseebKhalid1507/Axon/main/install.sh | sh
axon pull qwen2.5-coder-7b              # axon pull --list: the names
axon chat qwen2.5-coder-7b --device cuda       # an NVIDIA GPU; without --device cuda, the CPU
axon serve qwen2.5-coder-7b --device cuda --port 8080
```

- **What it installs:** `axon` in `~/.local/bin` and the license notices in `~/.local/share/doc/axon`, from the
  latest GitHub release, after checking the tarball's SHA-256 against the release's `.sha256`. Run again, it
  changes nothing unless the release's binary differs ([usage.md](usage.md#install-a-release-linux-x86-64)).
- **What it needs:** Linux x86-64 with glibc 2.28 or newer (RHEL 8, Debian 10, Ubuntu 20.04). For `--device cuda`,
  only the NVIDIA driver, R580 or newer (CUDA 13); no CUDA toolkit. Without a GPU it runs on the CPU.
- **From source:** `cargo build --release` (Rust 1.89 or newer, Cargo.toml's `rust-version`; CI builds with 1.95.0).
  A source build has no precompiled kernels: on the GPU it compiles them with NVRTC (`libnvrtc.so`, from a CUDA
  toolkit) on first use ([models.md](models.md#nvidia-gpus)).

## What it is

LLM inference written from scratch in Rust for the CPU and NVIDIA GPUs: one binary, no Python, no ggml, no BLAS,
nothing CUDA needed to build. Its CUDA C++ kernels are launched through the driver API. A release binary carries
them precompiled; a source build compiles them with NVRTC.

- **Models:** Qwen2 / Qwen2.5, Qwen3, Llama 3.x, Mistral 7B, Gemma 3 1B/4B (dense, CPU and CUDA); qwen3moe,
  gpt-oss-20b and DeepSeek V2/V3 / Kimi K2 / Moonlight architectures (attention on the GPU, routed experts on the
  CPU; for qwen3moe and deepseek2, `--gpu-experts N|auto` keeps them resident on the GPU, opt-in: #127, #146;
  gpt-oss's stay on the CPU, since its expert biases and MXFP4 experts aren't supported there yet). Hugging Face
  safetensors (CPU), and FP8 checkpoints (RedHatAI's `*-FP8-dynamic`, as Q8_0 by default, #177). Quantizations
  from Q2_K to F32, plus MXFP4 and IQ4_NL ([models.md](models.md)).
- **GPUs:** sm_61 to sm_120 (sm_61 and sm_70 with a CUDA 12 toolkit's NVRTC: Known limits), Hopper wgmma/TMA
  prefill, CPU/GPU split, multi-GPU pipeline, 8-bit KV.
- **Commands:** `info`, `gen`, `logits`, `chat` (with exact session save/restore), `serve`, `pull` and `kernels`.
- **`axon serve`:** OpenAI chat and completions, `/v1/responses` and the ollama API; continuous batching, chunked
  prefill, a paged KV cache, metrics ([serving.md](serving.md)). For agents:
  - **tool calls** on Qwen2.5 and Qwen3 (Hermes), Llama 3.1/3.2 and gpt-oss (harmony), with prompts from each
    GGUF's own chat template, token for token (#246); other chat formats answer a tool request with a 400. Calls
    stream as they are generated (#260; `AXON_TOOL_STREAM=0` sends each call whole);
  - **shared prefix pages, on by default** (#228): requests with a common prompt prefix share its KV pages, same
    bits; `AXON_SHARE_KV=0` turns them off. Under `--numerics fast` (so under `--datacenter`) they are off unless
    `AXON_SHARE_KV=1`;
  - **a KV RAM tier**, opt-in (`--cache-ram BYTES|auto`, #166, #201, #226): pages the GPU evicts go to RAM, and an
    evicted session resumes from there instead of prefilling again;
  - **constrained tool calls**, opt-in (`AXON_CONSTRAIN=1`, #264): on Qwen2.5-family models a tool call is held to
    valid JSON.
- **The GPU with only the driver** (#227): the release binary carries its kernels precompiled for sm_75, sm_80,
  sm_86, sm_89, sm_90/sm_90a and sm_120 (RTX 20/30/40/50, A100, H100), plus PTX that the driver compiles for newer
  GPUs. They are NVRTC 13.4.92's output for the same source: the same code a run-time compile with that NVRTC
  gives (other NVRTC versions were measured identical on sm_89 only; [models.md](models.md#nvidia-gpus)), and
  nothing compiles on first run. `axon kernels list` says where this GPU's kernels come from.
- **`axon pull NAME | OWNER/REPO/FILE.gguf`** (#231): GGUFs from Hugging Face, resumable, checked against their
  SHA-256 before they are moved into the models directory (`~/.local/share/axon/models`). About ten names for local
  coding agents (`axon pull --list`), each a file at a pinned commit; `gen`, `chat`, `serve` and `info` take the
  name where they took a path.
- **A library:** `axon::load`, `Backend::new`, `Tokenizer`, the `Lm` trait, `axon::generate` (greedy, with
  optional lossless drafting; sampling stays the CLI's) and `axon::serve` let a Rust program run a model in-process
  ([examples/embed.rs](../examples/embed.rs)). Not yet hardened for embedders: see Known limits.

## The contract

Decode == batched prefill == speculative verify, bit for bit, on each device; reusing a cache (and, since #228, a
shared prefix page) == a fresh prefill; greedy output doesn't change with drafting, batching or prompt chunking.
Every response carries a `system_fingerprint` of what decides its bits ([docs/determinism.md](determinism.md)). Two
exceptions. `--numerics fast` (opt-in) trades this for tolerance and has its own fingerprint. A CPU/GPU split that
streams its CPU layers for prefill (on by default when there's VRAM for it; `AXON_STREAM=0` turns it off) runs a
prompt of 32 tokens or more down another path than a shorter one, so there decode and prefill, or a cached prefix
and a fresh prefill, can differ; its fingerprint says it streams. The tests and the md5 gates check the bits;
correctness is checked against three independent oracles (gguf-py, Hugging Face's tokenizer, Ollama's greedy
output).

## Numbers, honestly

Each line names its date, its commit and its GPU. None of the older boards were re-run on the candidate; the two
boards on `9f3630c` are below them.

- **Single stream, RunPod, main `7f42c02` (2026-10-18):** ahead of llama.cpp (master `ff5a233`) and ollama 0.40.6
  in all 45 cells (RTX 4090, RTX 5090, H100 SXM; Qwen2.5 0.5B–7B and Qwen3 8B Q4_K_M; pp560 / pp2100 / tg128). The
  thinnest margins are +3 %: 7B and Qwen3 8B decode on the 4090 (186 vs 181 and 170 vs 165 tok/s).
  [results.md](results.md#gpu-runpod-rtx-4090--rtx-5090--h100-sxm-main-7f42c02-2026-10-18-1411--1508). That is
  parity with a small lead, and both engines have moved since. llama.cpp ran as `llama-bench -ngl 99 -r 1` at its
  defaults (`-ub 512`, flash attention at its default); the board on `9f3630c` gives it `-fa 1` and its best of
  `-ub 512` and `4096`, so the two can disagree.
- **Serving on an H100 SXM, main `c15c90e` (2026-10-19)** (Qwen2.5-7B, against vLLM 0.32.0 and SGLang 0.5.22 with
  AWQ and FP8; *mix* = chat plus 2k–7.6k-token prompts, *classic* = 519-token prompts, 512 out):
  - level with SGLang-AWQ at one stream (*mix* C = 1: 287 vs 284 tok/s, inside one pod's uncertainty), ahead of
    vLLM-AWQ (235); ITL p99 lowest of the five at C = 1 on both workloads;
  - ITL p99 lowest at *mix* C = 128 only (77.5 ms vs 126–320 ms): Axon's steps there are slow but even;
  - ITL p99 **highest** of the five on *classic* from C = 8 up (C = 128: 62.0 vs 20.2–39.7 ms) and on *mix* at
    C = 8 (58.2 vs 7.6–10.7 ms); second at *mix* C = 32 and 64;
  - aggregate throughput **2.0–3.8× behind on *mix* and 1.8–3.2× on *classic* at C = 128** (*mix* 1,868 vs
    3,770–7,146 tok/s), behind at every C ≥ 8, with worse TPOT and TTFT p99 at C ≥ 32;
  - cold start 1.0 s with its kernel cache compiled ahead and copied to the pod, vs 42–71 s.

  At `c15c90e` Axon had no FP8 path, so the rows run different weights; FP8 checkpoints load since #177, as Q8_0
  by default, and this board wasn't re-run. [results.md](results.md#serving-runpod-h100-sxm-main-c15c90e-2026-10-19).
  This tier is [shelved](roadmap.md#shelved-for-now).
- **Agent replay, bella (RTX 5070 Ti Laptop), main `d8a296c` (2026-10-27):** a 10-minute window of agent traffic
  (20 sessions, prompts p50 17.2k tokens, scaled to 0.15× and clipped at 32k), Qwen2.5-7B Q4_K_M, f16 KV, against
  llama-server `4413a19`: turn TTFT p50 0.122 vs 1.06 s, ITL p99 119 vs 611 ms. llama-server was configured by
  Axon's side; it re-prefilled 1.9 % of its prompt tokens, Axon 10.3 %. [log](log/2026-10-27-agent-replay.md). The
  board below re-runs it against seven llama-server configurations chosen to favor it, and its R supersedes this
  line.
- **The boards on `9f3630c`** (bella, the release candidate's own binary, its kernel pack only; roadmap week 1,
  item 7). **Pending:** they run on bella once a smoke of the same scripts has passed, one cell per job, a run lost
  to foreign GPU load alone run again (bounded). Each is stated here only once its log is in this PR. The rules were
  written down before any board ran ([log](log/2026-11-14-release-0.1-prep.md)).
  - *Single stream:* axon vs llama-bench, upstream's official build of llama.cpp master (b12387, 2026-11-14) and
    bella's `4413a19`, the weekly pod board's protocol with the run order rotated by round. Each cell is read
    against the faster of the two builds; ahead or behind only past 2 % with the rounds' ranges apart. Job `<sb>`.
  - *Agent replay:* the cell above. llama-server first runs seven configurations chosen to favor it, from llama.cpp's
    source; the one with the lowest turn TTFT p50 is its pick. Then axon vs that configuration, 3 rounds not lost to
    foreign GPU load (at most 8), the engine order alternating. **Stop rule:** R = llama-server's median turn TTFT p50
    over axon's, over the rounds where both engines are clean. If llama-server is faster in every round, or the rounds'
    ratios include 1, the one-sentence summary above says so. If R ≤ 1.5, the edge is thinner than claimed, and MoE in
    the paged pool goes first in the roadmap. The tag goes ahead either way. Jobs `<sweep>`, `<board>`.

## Known limits

- **Throughput at high concurrency on datacenter GPUs** (above); that tier is shelved.
- **Mixture-of-experts models:** routed experts run on the CPU unless `--gpu-experts` keeps them on the GPU
  (opt-in, qwen3moe and deepseek2; gpt-oss's experts always run on the CPU); `axon serve` runs one request at a
  time for them, without paged KV, shared pages or the RAM tier; no CPU/GPU layer split, multi-GPU or MoE draft
  model. gpt-oss-20b prefills at 0.27× llama.cpp, both with its experts on the CPU (`-ot exps=CPU` for llama.cpp;
  bella, #98, 2026-10-19).
- **Structured output:** `response_format` JSON is refused with a 400 (#116); constrained decoding covers only
  Hermes tool calls on Qwen2.5-family models, opt-in.
- **Gemma 3** on the GPU must fit whole; 27B is refused.
- **Platforms:** the release binary is Linux x86-64 only. CUDA is Linux only. Built and tested on x86-64 Linux
  only (scalar, AVX2 and AVX-VNNI kernels picked at run time); other targets have scalar code paths but no CI. Not
  in 0.1: Windows, vision, embeddings, auto-update, signing.
- **GPUs the precompiled kernels don't cover:** data-center Blackwell (B200/B300: sm_100, sm_103), Volta and
  Pascal, and drivers before R580, need NVRTC from a CUDA toolkit, as a source build does (Volta and Pascal a CUDA
  12 one). Without one, `--device cuda` stops with the reason; it never falls back to the CPU silently.
- **The release isn't signed.** `install.sh` checks the tarball against the release's own `.sha256`: that catches a
  broken download, not a release replaced by someone with push access.
- **The library API is 0.x, small, and not hardened for embedders**
  ([#118](https://github.com/HaseebKhalid1507/Axon/issues/118)): library calls print notes to stderr; engine
  tuning variables (`AXON_*`) are read inside library calls, some once at first use and others at every `load` /
  `Backend::new`, so changing them between loads changes behaviour; `AXON_GPUS` is the only way to pick a GPU; the
  f16/f32 KV choice is process-wide; drafting is refused on a split with `AXON_STREAM_DECODE=1`; each CUDA
  `Backend::new` leaks its context and kernels; `axon::serve` exits the process on a usage error (2) or an engine
  failure (70); `generate` is greedy only.
- **One maintainer, no external users yet:** please open issues.

## For the maintainer (not done by this PR)

The procedure, its checks and what each refusal means: [releasing.md](releasing.md).

- [ ] Before the tag, one docs-only commit: the date in place of `unreleased` in CHANGELOG.md's `## [0.1.0]`
      heading, and the "pending tag" wording dropped from README.md and docs/ (releasing.md, "Before the tag"):
      the tarball ships README.md, so the tag's must already read as released. The binary doesn't change
      (nothing under `src/` includes a doc).
- [ ] Tag the commit: `git tag -s v0.1.0 <sha> && git push origin v0.1.0`. The tag's run builds the tarball and
      makes a draft release titled "Axon v0.1.0 (checked; publish with scripts/release/publish.sh)" once the tag
      still names its commit.
- [ ] `scripts/release/publish.sh --check v0.1.0`: it prints the tarball's SHA-256. Then the hand checks in
      releasing.md (pack bytes, md5 gate, a clean machine) on the draft's own tarball.
- [ ] The draft tarball's `axon` has the SHA-256 the boards measured, `2c19415bf329b30f…` (releasing.md, "The
      boards' binary"). If not, the board lines say "measured on the candidate `9f3630c`, not the release binary".
- [ ] The draft's notes: this page from "In one sentence" to the end of "Known limits" (not the banner, not this
      list), with no `<sb>`, `<sweep>`, `<board>` or "Pending" left in it, keeping the job's "SHA-256 of the
      tarball: …" and "Built from commit …" lines, each exactly once, and the title. A board with no reading by
      the tag (its queue still running, no untainted run, NO_VERDICT) doesn't hold the tag: its sub-bullet goes,
      and the bullet says which board wasn't measured on the candidate and why, in one line. The 10-27 agent line
      then stands as written, its caveat included. Nothing from the failed 11-15 run is cited.
- [ ] Publish: `scripts/release/publish.sh v0.1.0 <SHA-256>`, with the SHA-256 written next to the checks.
- [ ] No `cargo publish` under the name `axon`: it's taken on crates.io (aep/axon); pick a package name first
      (`[lib] name = "axon"` can stay).
