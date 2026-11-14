# Using Axon

The command line in full: the commands, every flag and its default, chat, and the memory checks. Models and
hardware: [models.md](models.md). `axon serve`: [serving.md](serving.md). Environment variables:
[environment.md](environment.md). Axon as a Rust library: [library.md](library.md).

## Install a release (Linux x86-64)

```sh
curl -fsSL https://raw.githubusercontent.com/HaseebKhalid1507/Axon/main/install.sh | sh
axon pull qwen2.5-coder-7b        # or axon pull --list
axon chat qwen2.5-coder-7b --device cuda
```

Until v0.1.0 is published there is no release to install (the script says so): build from source, as in the
[README](../README.md#install).

`install.sh` puts `axon` in `~/.local/bin` (`AXON_INSTALL_DIR`) from the latest GitHub release
(`AXON_VERSION=vX.Y.Z` for another), after checking the tarball's SHA-256 against the release's `.sha256`, and
the license notices in `~/.local/share/doc/axon`; run again, it replaces the binary only if the release's differs.
The release binary needs glibc 2.28+ (RHEL 8, Debian 10, Ubuntu 20.04 and newer) and, for `--device cuda`, only the
NVIDIA driver, R580 or newer (CUDA 13): its GPU kernels are built in for Turing (RTX 20), Ampere, Ada, Hopper and
Blackwell RTX (sm_120) GPUs, with PTX for newer ones (models.md, NVIDIA GPUs); other GPUs (B200-class sm_100,
Volta, Pascal) need the CUDA toolkit's NVRTC, as a source build does (Volta and Pascal a CUDA 12 one: NVRTC 13
dropped them). On a driver older than R580, `--device cuda` uses a
CUDA 12 toolkit's NVRTC if there is one, and otherwise stops with one line (and axon's usual hint), for example

```
axon: --device cuda: NVIDIA driver 570.86.15 (CUDA 12.8) is too old: axon's GPU kernels need R580+ (CUDA 13); update the driver, or run on the CPU with --device cpu
try 'axon --help'
```

(exit code 2, as for every refusal).

Without a GPU it runs on the CPU (the default). To uninstall: `rm ~/.local/bin/axon` and
`rm -r ~/.local/share/doc/axon`, and `~/.local/share/axon/models` for the pulled models. Publishing a release:
[releasing.md](releasing.md).

## Quick start

Build, download a model, `gen`, `chat` and `serve`: the [README's quick start](../README.md#quick-start). `gen`
streams the text to stdout and prints prompt and decode speed to stderr; `chat` is a conversation: type a message,
get a reply, keep going.

## Generation

### Sampling

Sampling: greedy, or temperature / top-k / top-p / min-p / repeat penalty with a seed
(llama.cpp's order, ollama's defaults). `gen` continues raw text, greedy unless `--temp`
is above 0 (reproducible output and benchmarks); `chat` talks to an Instruct model
through its chat template (ChatML, Llama 3's, Mistral's `[INST]` or Gemma 3's) and samples by default (`--temp 0` for greedy). On the GPU
the penalty and top-k (up to 256) run on the device, so only k candidates come back per
token; with top-k 0 (off: the whole vocab, as llama.cpp) the logits come back instead.

### Speculative drafting

- Prompt-lookup drafting (`--draft K`, off by default): speculative decoding that guesses
  the next tokens from earlier text (prompt and output) and verifies them in one batched
  pass. Lossless: the same output as without it (greedy, or sampled with the same seed).
- Draft-model drafting (`--draft-model small.gguf`): a small model of the same vocabulary
  (Qwen2.5-0.5B for Qwen2.5-7B, Qwen3-0.6B for Qwen3-8B, Llama-3.2-1B for Llama-3.1-8B)
  guesses the next tokens, the model verifies them in one batched pass; equally lossless.
  Default K=3 with a draft model. Greedy tg128 medians in its A/B at K=3: +37% Qwen2.5-7B,
  +44% Qwen3-8B, +63% Llama-3.1-8B; sampled, little or none (below). Next to a model split
  across the CPU and GPU (32B on a 12 GB card), greedy gen and chat default to K=7: the split's
  step is bound by reading the CPU layers' weights from RAM, which a verify pass does once for
  all of its tokens, so drafting pays far more there ([design/spec-split.md](design/spec-split.md)).

### KV cache

KV cache: f16 by default; `--kv q8` (GPU only) stores int8 with a scale per position and
head, half the memory, with slightly different results.

### Sessions

Sessions: `chat --session FILE` keeps a conversation in a file, saved after every reply.
At the next start its KV cache is loaded back if the model, KV type, backend and GPU
match (for an FP8 checkpoint in fast mode, also whether fp8.cu ran it), else the saved
tokens are prefilled again; either way the reply continues as if never stopped (on a
CPU/GPU split a re-prefilled cache is close, not exact). Measured:
a 29.8k-position 7B session restores in 0.41-0.45 s against a 3.16 s re-prefill (RTX 4090).

### Long context

Long context (opt-in): `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768` (llama.cpp's
flags) takes Qwen2.5 / Qwen3 from 32k to 128k by YaRN RoPE scaling. It changes the output at
every length (measured +8 % perplexity at 2k-8k on Qwen2.5-7B), so only for prompts past the
trained context: [docs/long-context.md](long-context.md).

## Commands

```
axon <model.gguf>                         same as axon info
axon info <model.gguf>                    print a summary and exit
                                          (everywhere: a checkpoint directory instead, CPU)
axon gen <model.gguf> "<prompt>" [-n N] [--draft K] [--draft-model PATH] [--device D]
          [--ctx N] [--gpu-layers N] [--gpus L] [--gpu-experts N|auto] [--kv T] [--numerics M]
                                          generation (default 32 tokens; greedy unless --temp)
axon logits <model.gguf> "<prompt>" [-k K] [--numerics M]
                                          top-K next-token logits (default 10)
axon chat <model.gguf> [--system TEXT] [--no-think] [--ctx N] [--forget-to F] [-n N]
          [--draft K] [--draft-model PATH] [--device D] [--gpu-layers N] [--gpus L]
          [--gpu-experts N|auto] [--kv T] [--numerics M] [--session F]
                                          talk to an Instruct model (sampled; --temp 0: greedy)
sampling (gen, chat): --temp T --top-k K --top-p P --min-p P
          --repeat-penalty P --repeat-last-n N --seed S   (axon --help)
axon serve <model.gguf> [--host H] [--port P] [--ctx N] [--draft K] [--draft-model PATH]
          [--device D] [--gpu-layers N] [--kv T] [--parallel N] [--step-ms MS]
          [--admit-order sjf|fifo] [--numerics M] [--datacenter] [...]
                                          OpenAI-compatible HTTP server (serving.md)
axon kernels list [PACK]                  the precompiled GPU kernels and where this GPU's
                                          come from (models.md, NVIDIA GPUs)
axon kernels build OUT [--targets L] [--fatbinary P|none] [-j N]
                                          compile a kernel pack with NVRTC (release builds)
axon pull NAME | OWNER/REPO/FILE.gguf      download a GGUF from Hugging Face, checked against its
                                          SHA-256 (models.md, Getting models); axon pull --list
axon --help | --version
```

## Flags and defaults

Defaults, as `axon --help` gives them:

- `--draft K`: 0 (off); try 7. With `--draft-model`: 3 (0 = off); gen and chat decoding
  greedily: 7 when the model is split across the CPU and GPU. The gain is in greedy decoding (`--temp 0`); sampled, it's little or none (measured +0.8 % on 7B at temp 0.8, K=3; slower at K 5 and 7).
- `--draft-model PATH`: none. The draft model runs on the same `--device` (on `cuda` all of
  it on the GPU; its memory is set aside before the model is placed). A model whose token
  table isn't the start of the main model's, or whose end-of-generation tokens differ, is
  refused. gen, chat: `--draft-model auto` picks the smallest GGUF in the model's directory,
  at most 1/8 of its size, that can draft for it (none found: drafting off, said on stderr).
  Opt-in: drafting is never turned on without the flag.
- `--device D`: `cpu`, or `cuda`. A model too big for the GPU's memory is split
  automatically: the last layers that fit on the GPU, the first ones on the CPU.
- `--gpu-layers N` (with `--device cuda`): the last N layers and the output head on the
  GPU, the rest on the CPU (0 = all on the CPU). Default: all of them if they fit with
  256 MiB to spare, else as many as fit with another 640 MiB left for the desktop; an
  explicit N only needs the 256 MiB.
- `--gpu-experts N|auto` (gen, chat; `AXON_GPU_EXPERTS` for serve too; deepseek2 and
  qwen3moe with `--device cuda`): the routed experts of N MoE layers resident on the GPU instead of the
  CPU, smallest layers first, as many as fit after the context's cache (`auto`: all that
  fit). Default 0. Same bits as on the CPU (the experts' GPU kernels are twins of the CPU's
  r8 kernels); where that can't hold (`AXON_REPACK=0`, a host whose FMA choice differs from
  the GPU twins', a GPU below sm_80, or a weight type with no twin) it keeps them on the CPU
  and says why, as it does for gpt-oss (no MXFP4 twin yet). qwen3moe's bits are
  checked on Qwen3-MOE-4x0.6B and a pruned Qwen3-30B-A3B (the 30B's expert shapes); the full
  Qwen3-30B-A3B is untested end to end until its pod bench (`auto` vs `0`, text and speed) runs.
  On the CPU and with dense models the flag does nothing. `axon serve` takes it as
  `AXON_GPU_EXPERTS` and serves a MoE model one request at a time.
- `--gpus L` (gen, chat; `AXON_GPUS=L` for serve too): the CUDA devices to run on. One
  (`--gpus 1`) runs the model there. Several (`--gpus 0,1`) split its layers across them in
  order (pipeline parallel): the first GPU runs the embedding and first layers, the last one
  the rest and the output head, and each token's hidden state (one row of floats) is copied
  from one to the next. The output is the same, bit for bit, as on one GPU. It doesn't make
  one stream faster (decode is the sum of the stages: measured 1.00x one 4090 on 32B and 7B)
  but holds models bigger than one card, and prefill overlaps the stages (1.46x on 7B,
  1.55x on 32B for a 2098-token prompt on 2x RTX 4090). Layers go evenly if they fit, else
  in proportion to each card's free memory; `AXON_PP_CUTS` sets them. The copies go through
  host memory (the driver stages them; ~10 us per token on PCIe 4.0). `AXON_P2P=1` asks
  for direct peer-to-peer copies instead: UNTESTED, no P2P-capable machine has run it yet.
  `axon serve` runs a pipeline one request at a time.
- `--kv T`: `f16`, or `q8` (cuda only; on a split, the GPU layers' cache, so more
  layers fit).
- `--numerics M` (gen, logits, chat, serve): `exact`, or `fast`. `exact` is the
  bit-identity contract ([docs/determinism.md](determinism.md)); `fast` may run
  kernels that change the float order (graded against exact, not equal) under its own
  identity, and promises tolerance on every path, never bits (batching, cache reuse,
  drafting). Its only fast kernels are an FP8 checkpoint's on the GPU (#177); elsewhere `fast` runs the exact
  kernels, under the fast identity, and says so once.
  `serve --datacenter` implies `--numerics fast` (and, today, nothing else); an explicit
  `--numerics` wins. Why: [docs/design/determinism-modes.md](design/determinism-modes.md).
- `--ctx N` (chat): the model's trained context, at most 8192. On cuda, if the
  whole model doesn't fit with it, a shorter context if at least half of it (or 4096)
  fits, else a split. For `serve`, see [serving.md](serving.md): its default is the whole trained context.
  gen: none (the cache holds the prompt and `-n`); `--ctx N` places the model and sizes its
  cache for N tokens, as chat would, to measure a long context's speed.
- `--forget-to F` (chat): 0.5. When the conversation outgrows `--ctx`, the oldest exchanges
  are forgotten until the rest, with the new message, fills at most F of the context (the
  newest exchange stays unless the new message needs its room). The next turns append to
  the cache until it's full again; a reply reserves room for itself (`-n`, or a quarter of
  the context if unlimited), so with the defaults 0.5 leaves a quarter of the context to grow
  into. `min`, 0 or 1: only as many as make room for the reply (the old behaviour).
  Ctrl-C before a reply's first token puts the forgotten exchanges back.
- `--rope-scaling yarn --rope-scale F [--yarn-orig-ctx N]` (gen, logits, chat, serve): off.
  With it the context is N x F (`--ctx` up to that); also `--yarn-beta-fast` (32),
  `--yarn-beta-slow` (1), `--yarn-mscale` (0.1 ln F + 1; not llama.cpp's `--yarn-attn-factor`). See
  [docs/long-context.md](long-context.md) for when to use it and what it costs.
- `-n N`: gen 32 tokens; chat no limit per reply.
- Sampling: `--temp` 0 for gen, 0.8 for chat; `--top-k` 40 (0 = off, all tokens),
  `--top-p` 0.9, `--min-p` 0 (off), `--repeat-penalty` 1.1 (serve: 1, off),
  `--repeat-last-n` 64, `--seed` a new one each run (gen prints it).

Special-token text in a prompt (`<|im_end|>` and the like) is encoded as plain
text, never as the control token.

## Chat

`chat` keeps the whole conversation in one KV cache, so each turn only runs the
model over the new message, not the history. On a terminal: `/reset` forgets the
conversation, `/system TEXT` sets a new system message, `/help` lists commands and
keys. The prompt is a line editor: arrows, Ctrl-Left/Right or Alt-B/F by word,
Home/End or Ctrl-A/E, Ctrl-W/U/K to delete a word back or to the start/end, Ctrl-Y
to put it back, Up/Down for this session's history, Ctrl-L to clear the screen. A
line ending in `\` (or Alt-Enter) goes on to a new line of the same message, and
pasted text stays one message, line breaks and all. Ctrl-C stops a reply or clears
the line, never quits; Ctrl-D on an empty line (or `/exit`) quits. Keys typed while
a reply streams wait for the next prompt instead of mixing into the reply. The
terminal is in raw mode while chat runs, so Ctrl-Z and Ctrl-\ do nothing. Piped,
each line of stdin is a message and each reply one block of stdout. When the
conversation outgrows `--ctx` (default: the model's, at most 8192), the oldest exchanges are forgotten,
down to half the context by default (`--forget-to`). Forgetting shifts every position, so what's kept is
prefilled again; forgetting only what makes room would pay that near-full prefill on every turn once the
context is full, while half leaves room for the next several turns to append to the cache. The cache after
forgetting is the same, byte for byte, as a fresh prefill of what's kept.

Llama 3 uses the official 3.1/3.2 template (no tools): a system block that always
opens with the template's "Cutting Knowledge Date: December 2023 / Today Date: 26 Jul
2024" lines (its default date, so a conversation doesn't depend on the day) and then
`--system`'s text if any, headers around each message, `<|eot_id|>` ending each turn.

Qwen3 follows its own template: no system message unless `--system` gives one, and
the model reasons first (`<think>...</think>`, shown as it streams; Ollama's default
too). `--no-think` opens each reply with an empty reasoning block instead (the
template's `enable_thinking=false`). Earlier replies stay in the conversation
without their reasoning, as the template renders them. Qwen recommends sampling
for thinking mode (temperature 0.6, top-p 0.95, top-k 20), as `chat` does by default with
its own defaults (`--temp 0.8 --top-k 40 --top-p 0.9`); with `--temp 0` (greedy) long
reasoning can loop.

## Memory limits

A load that can't fit in the RAM available (the host's, or the tightest cgroup limit) is refused at
load with the figures and the way out (exit 2), instead of being OOM-killed: a dense CPU model, a split's CPU half,
an FP8 checkpoint, a checkpoint's converted tensors (#133, #142, #180, #206, #215; below).
Except near the edge: within ~48 MiB above a refusal a run can still be OOM-killed (the checks' tolerance lets it
through, and the CUDA module's memory isn't counted: #217).

**Memory limits (CPU).** Before a model is built for the CPU, its repacked weights and the KV cache of the
first 8192 tokens (or `--ctx`, if shorter) are checked against the RAM available: the host's `MemAvailable`,
or the room under the tightest cgroup memory limit (a container, a systemd scope). A model that can't fit is
refused with a message naming the binding limit and the ways out (a GPU or more `--gpu-layers`, when the model
can take them, or the longest `--ctx` that fits), instead of being OOM-killed. A longer `--ctx` whose full KV cache wouldn't fit only prints a
note: that cache is taken as the context fills. Mixture-of-experts models have the same check for their experts.
The CPU half of a `--gpu-layers` split has it too, for its layers and their KV cache, checked before the GPU
half's upload, with the upload's own host memory counted: its page-locked staging ring (16 or 64 MiB, or the
size `AXON_UPLOAD_RING` sets), or, if larger, the biggest matrix it converts to Q8_0 on the host (an FP8
checkpoint's head: 138 MiB on the 0.5B). The upload is done before the CPU half is built, so the split needs the
larger of the two. When the model's bytes are in RAM (a copied file, or a checkpoint's
converted tensors), the GPU half's share, which goes back as it's uploaded, is credited to the CPU half. A
safetensors checkpoint's tensors that load converts (BF16 made F16 or F32: the BF16 model's, an FP8
checkpoint's embedding, head and vectors) are checked before they're made, on every device. Not checked yet: the
CUDA kernels' module, about 25 MiB of host memory loaded after that check and before the split's, and the
conversion's page tables, about 1 MiB (#217). A split that streams
prefill through the GPU page-locks the CPU layers' whole KV cache when the cache is made; if the RAM available
can't hold that, the cache doesn't stream (a note says so) and the run goes on at CPU prefill speed.
