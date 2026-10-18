# Axon

LLM inference for Qwen2, Qwen3, Llama 3 and Gemma 3 GGUF models on the CPU and on NVIDIA
GPUs, written from scratch in Rust.

**Mission: make LLM inference cheap and fast, and run the best open models.** Cheap means wasting no
hardware, from a laptop to a datacenter: the most tokens per GPU, per watt and per dollar. Fast means as close
to the hardware limit as the math allows, for one stream and for thousands. The best models means the frontier
open models, not only the ones that fit a laptop.

**New here? Start with [docs/START-HERE.md](docs/START-HERE.md).**

No ggml, no BLAS, no binding crates, no CUDA toolkit at build time: a GGUF reader,
quantized integer dot-product kernels (scalar, AVX2, AVX-VNNI), a spinning worker
pool and a transformer forward pass on the CPU; on the GPU, CUDA C++ kernels that
NVRTC compiles at run time, launched through the driver API (both libraries loaded
with dlopen). About 67k lines of Rust and 6.5k of CUDA, tests included.

Every optimized path is **bit-identical** to the plain reference path on its
device: batched prefill produces exactly the logits that feeding one token at a
time does (on the CPU, and on the GPU against the GPU's own one-token step), and
each SIMD kernel exactly what its scalar twin does. The GPU is deterministic and
close to the CPU but not bit-identical to it (its float sums are split and ordered
differently; `src/cuda/qwen2.rs`). The tests check this, and check against three
independent oracles (gguf-py's dequantizer, Hugging Face's tokenizer, Ollama's
greedy output).

## Status

Works today, honestly scoped:

- Architectures: `qwen2`, `qwen3` (dense) and `llama` (Llama 3.x). The tests use
  Qwen2.5-0.5B-Instruct (Q4_K_M), Qwen3 0.6B (Q8_0) and Llama 3.2 1B Instruct, and the
  GPU tests also the Qwen2.5 1.5B, 3B and 7B Instruct, Qwen3 4B and 8B, and Llama 3.2 3B
  and Llama 3.1 8B Instruct Q4_K_M builds; Qwen3 1.7B (Q4_K_M) runs too. Llama 3's RoPE
  frequency factors and the llama-bpe tokenizer with its BOS token are handled. `llama`
  models with a SentencePiece tokenizer (Mistral 7B Instruct v0.3, Llama 2, TinyLlama) work
  too; Mistral's `[INST]` chat format is built in. `gemma3` (Gemma 3 1B and 4B instruct;
  head size 256, sliding-window layers, GeGLU) runs on the CPU and the GPU with its
  `<start_of_turn>` chat format; on the GPU the whole model must fit (no CPU/GPU split, no
  `--kv q8` yet), and its local layers keep a full-length KV cache. 27B is refused.
  `qwen3moe` (Qwen3-30B-A3B and its relatives: softmax top-k routing, no shared expert)
  runs on the CPU (`--device cpu`: every expert repacked for the r8 kernels, stage 1, see
  [the log](docs/log/2026-10-17-moe-stage1.md)) or, with `--device cuda`, with its attention,
  norms, routers and head on the GPU and its experts on the CPU (llama.cpp's `--cpu-moe`
  layout; stage 2, [log](docs/log/2026-10-17-moe-stage2.md)). Both: decode == batched
  prefill == verify bit for bit, logits within llama.cpp's own CPU/CUDA spread; no layer
  split (`--gpu-layers`), multi-GPU or MoE draft model. `axon info` prints its experts and
  bytes per token. Streaming the experts through the GPU for long prompts is next
  ([roadmap](docs/roadmap.md#mixture-of-experts)). `gpt-oss` (gpt-oss-20b, ggml-org's MXFP4
  GGUF) runs on the CPU only for now (stage 0, [log](docs/log/2026-10-18-gpt-oss-s0.md)): MXFP4
  experts read in place, attention sinks, the 128-token sliding window on alternate layers,
  YaRN, the o200k tokenizer and the harmony chat format (`axon chat` shows the `final`
  channel; `axon serve` returns the `analysis` channel as `reasoning_content`; no tools
  yet); `--device cuda` is refused until stage 1. Other architectures, other
  mixture-of-experts models and RoPE scaling (a factor other than 1; Gemma 3's linear
  scaling and YaRN, below, excepted) are refused with an error.
- Tensor types: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and
  IQ4_NL on the CPU, so every Q2_K and Q3_K_S/M/L GGUF runs (IQ4_NL is llama.cpp's fallback
  for their rows that aren't a multiple of 256, e.g. the 0.5B's 896), and MXFP4 for
  gpt-oss's experts (CPU only). On the GPU every weight matrix must be one of the eleven
  quantized types (F16/F32 matrices are refused there). Others load for `info` but can't run.
- Sampling: greedy, or temperature / top-k / top-p / min-p / repeat penalty with a seed
  (llama.cpp's order, ollama's defaults). `gen` continues raw text, greedy unless `--temp`
  is above 0 (reproducible output and benchmarks); `chat` talks to an Instruct model
  through its chat template (ChatML, Llama 3's, Mistral's `[INST]` or Gemma 3's) and samples by default (`--temp 0` for greedy). On the GPU
  the penalty and top-k (up to 256) run on the device, so only k candidates come back per
  token; with top-k 0 (off: the whole vocab, as llama.cpp) the logits come back instead.
- CPU: any x86-64 (others build too, scalar only). The default build is baseline x86-64 and
  picks its AVX2 / AVX-VNNI code at runtime, with the same output bits as a `-C target-cpu`
  build (see `src/mv.rs`). Attention and SwiGLU use fused multiply-adds when the CPU has
  FMA and a separate multiply and add when it doesn't (`src/vmath.rs`), so logits depend on
  that one CPU property (on x86: whether the CPU has FMA; other targets fuse only when
  built with `fma` enabled), and are identical across paths (`AXON_SIMD`), batch sizes and
  thread counts for a given answer to it.
- GPU (`--device cuda` or `AXON_DEVICE=cuda`): NVIDIA, Linux only. Nothing CUDA is needed
  to build. At run time axon loads the driver (`libcuda.so.1`) and NVRTC (`libnvrtc.so`,
  `.so.13` or `.so.12`, from the library path or the toolkit's `lib64` under `$CUDA_PATH`,
  `/usr/local/cuda` or `/opt/cuda`), compiles its kernels for the GPU on first use and
  caches them in `~/.cache/axon/cuda` (`$XDG_CACHE_HOME/axon/cuda`). The driver must
  support CUDA 12 or newer, and at least NVRTC's CUDA major version. A GPU newer than
  NVRTC gets PTX for the driver to compile. If the first NVRTC found can't target the GPU
  (NVRTC 13 dropped Pascal and Volta), the next one installed is tried, so a GTX 1050
  (sm_61) runs on NVRTC 12.x beside 13; a GPU no installed NVRTC can target is refused.
  The int8 tensor-core GEMMs (batched prefill, speculative verify) and the tensor-core
  attention need sm_80 or newer; older GPUs run batches through the decode kernels
  (dp4a): correct, not fast. Without a usable GPU, `--device cuda` stops with the reason;
  it never falls back to the CPU silently.
- CPU + GPU: a model too big for the GPU is split (the first layers on the CPU, the rest
  and the output head on the GPU; `--gpu-layers N` to choose). For prompts (batches of
  32 tokens or more) the CPU layers' weights are streamed through the GPU, chunk by chunk,
  instead of running those layers on the CPU (`AXON_STREAM=0`: off).
- KV cache: f16 by default; `--kv q8` (GPU only) stores int8 with a scale per position and
  head, half the memory, with slightly different results.
- Sessions: `chat --session FILE` keeps a conversation in a file, saved after every reply.
  At the next start its KV cache is loaded back if the model, KV type, backend and GPU
  match, else the saved tokens are prefilled again; either way the reply continues as if
  never stopped (on a CPU/GPU split a re-prefilled cache is close, not exact). Measured:
  a 29.8k-position 7B session restores in 0.41-0.45 s against a 3.16 s re-prefill (RTX 4090).
- Long context (opt-in): `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768` (llama.cpp's
  flags) takes Qwen2.5 / Qwen3 from 32k to 128k by YaRN RoPE scaling. It changes the output at
  every length (measured +8 % perplexity at 2k-8k on Qwen2.5-7B), so only for prompts past the
  trained context: [docs/long-context.md](docs/long-context.md).
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
  all of its tokens, so drafting pays far more there (`docs/design/spec-split.md`).
- Serving: `axon serve`, an OpenAI-compatible HTTP server (below) with tool calling. On one GPU it
  batches the decode steps of concurrent requests (`--parallel N`) and feeds new prompts in chunks
  between them, over a paged KV cache (64-position pages shared by every slot: a slot holds only the
  positions it has; no extra cache for a request running alone); elsewhere it
  runs one request at a time. Operations: `--api-key`, `GET /metrics` (Prometheus), `GET /slots`,
  `--max-prompt-tokens`, `--slot-idle-ms`.
- Rust 1.89 or newer.

Not on main yet: tensor parallel, saving and restoring `serve` slots, the `/v1/responses` and ollama-native APIs,
and the MoE paths on the GPU (#67). What's in flight and why: [docs/roadmap.md](docs/roadmap.md). How Axon is built and measured, and
what was tried and dropped: [docs/methodology.md](docs/methodology.md),
[docs/history.md](docs/history.md), [docs/experiments-rejected.md](docs/experiments-rejected.md).

## Quickstart

```bash
git clone https://github.com/HaseebKhalid1507/Axon.git
cd Axon
cargo build --release

mkdir -p models
curl -L -o models/qwen2.5-0.5b-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

./target/release/axon gen models/qwen2.5-0.5b-q4_k_m.gguf "The capital of France is" -n 32
./target/release/axon chat models/qwen2.5-0.5b-q4_k_m.gguf

# the same on an NVIDIA GPU (the build is the same; the driver and NVRTC are found at run time)
./target/release/axon gen models/qwen2.5-0.5b-q4_k_m.gguf "The capital of France is" -n 32 --device cuda
```

`gen` streams the text to stdout and prints prompt/decode speed to stderr.
`chat` is a conversation: type a message, get a reply, keep going.

## Usage

```
axon <model.gguf>                         same as axon info
axon info <model.gguf>                    print a summary and exit
axon gen <model.gguf> "<prompt>" [-n N] [--draft K] [--draft-model PATH] [--device D]
          [--ctx N] [--gpu-layers N] [--gpus L] [--kv T]
                                          generation (default 32 tokens; greedy unless --temp)
axon logits <model.gguf> "<prompt>" [-k K]  top-K next-token logits (default 10)
axon chat <model.gguf> [--system TEXT] [--no-think] [--ctx N] [--forget-to F] [-n N]
          [--draft K] [--draft-model PATH] [--device D] [--gpu-layers N] [--gpus L] [--kv T]
          [--session F]
                                          talk to an Instruct model (sampled; --temp 0: greedy)
sampling (gen, chat): --temp T --top-k K --top-p P --min-p P
          --repeat-penalty P --repeat-last-n N --seed S   (axon --help)
axon serve <model.gguf> [--host H] [--port P] [--ctx N] [--draft K] [--draft-model PATH]
          [--device D] [--gpu-layers N] [--kv T] [--parallel N] [...]
                                          OpenAI-compatible HTTP server (below)
axon --help | --version
```

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
- `--ctx N` (chat): the model's trained context, at most 8192. On cuda, if the
  whole model doesn't fit with it, a shorter context if at least half of it (or 4096)
  fits, else a split. For `serve`, see below: its default is the whole trained context.
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
  [docs/long-context.md](docs/long-context.md) for when to use it and what it costs.
- `-n N`: gen 32 tokens; chat no limit per reply.
- Sampling: `--temp` 0 for gen, 0.8 for chat; `--top-k` 40 (0 = off, all tokens),
  `--top-p` 0.9, `--min-p` 0 (off), `--repeat-penalty` 1.1 (serve: 1, off),
  `--repeat-last-n` 64, `--seed` a new one each run (gen prints it).

Special-token text in a prompt (`<|im_end|>` and the like) is encoded as plain
text, never as the control token.

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

### Serving: `axon serve`

```bash
axon serve models/qwen2.5-7b-instruct-q4_k_m.gguf --device cuda --port 8080
curl -N localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"stream":true}'
```

An OpenAI-compatible HTTP/1.1 server (any OpenAI client works with
`base_url="http://127.0.0.1:8080/v1"`): `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/completions` (raw text, no template, as `gen`), `POST /v1/responses` and the
ollama API (`/api/chat`, `/api/generate`, ...; below), `GET /health`, `GET /metrics`,
`GET /slots` (below); streamed
(server-sent events, `data: [DONE]` at the end; `stream_options.include_usage` adds a
usage chunk) or whole. It takes the device, `--ctx`, `--kv`, `--gpu-layers` and `--draft`
flags of `chat`, plus `--host` (default 127.0.0.1), `--port` (default 8080), `--alias NAME`
(another model name to answer to), `--max-tokens N` (a cap on any request's generation;
default: up to the context), `--cors [ORIGIN]` (answer CORS preflights and allow ORIGIN,
default `*`, to call the server from web pages; off by default, so an arbitrary page can't
use your local model), `--max-conns N` / `--max-queue N` (defaults 256 / 128),
`--parallel N` (requests decoded at once on the GPU, default 16 with paged KV, 8 without; 1: one at a time),
`--prefill-ms MS` / `--prefill-chunk N` (how much prompt a batched step feeds while others
decode: up to what fits in MS ms beside their decode step, at least N tokens; defaults 60 /
256; `--prefill-ms 0`: exactly N tokens; `--prefill-chunk 0`: whole prompts), `--quiet`
(no per-request line on stderr). `GET /health` is 503 until the model is loaded.

- Context (`--ctx N`, per request): by default the model's trained context (32768 for
  Qwen2.5, 40960 for Qwen3, 131072 for Llama 3.1), so agent harnesses' 10k-40k-token prompts
  work without flags. With the whole model on one GPU (placed automatically, or every layer
  by `--gpu-layers N`), if the weights leave too little GPU memory for it (the KV cache of
  `--kv`'s type), the longest that fits (a multiple of 256; placed automatically, at least
  4096, else the model is split). Over several GPUs (`AXON_GPUS`), the longest the pipeline
  fits, at least 4096, else 8192. On the CPU, the longest whose KV cache fits in free RAM (a
  multiple of 1024, at least 8192). A CPU/GPU split (automatic or `--gpu-layers`) keeps 8192,
  as before: a long cache there takes GPU memory from the layers. A shorter context than
  trained is said once on stderr, with why and the flag to change it. An explicit `--ctx N`
  is used as given (up to the trained length). A `--draft-model` is sized for the main
  model's context (at most 32768 by default; a split's 8192) and drafts up to it.
- Many requests at once (`--device cuda`, the whole model on the GPU): up to `--parallel`
  requests decode together, one token each per step over the model's weights (continuous
  batching: a request joins as soon as a slot is free, and leaves when done). While others
  decode, a new prompt is prefilled in chunks, up to `--prefill-ms` / `--prefill-chunk` worth
  per step in between their steps (shortest remaining prompt first; one passed over 4
  steps in a row goes first), so a long prompt doesn't stop the others' streams while it
  prefills (on a 4090, 7B, a 24k-token prompt: their worst token gap ~70 ms instead of
  2.4 s); with nothing decoding, prompts are fed one after another, shortest remaining
  first, each to its end, in pieces of 250-500 ms at full speed (250 ms worth rounded up
  to whole prefill passes; 4096 tokens until the server has measured a chunk's cost); a
  prompt that 4 shorter, younger ones have finished ahead of goes first. A long prompt sent
  to an idle server alone takes the single-stream path below and is fed whole: a request
  arriving during that prefill waits for it. The KV cache is paged: a pool of 64-position
  pages shared by every slot, a slot holding pages only for the positions it has, so the
  context is a slot's cap, not memory reserved for it (`AXON_PAGED_KV=0`: the old pool of
  slots reserved at the full context each). The pool is the GPU memory left beside the
  weights and scratch (at most a page per position of every slot), up to 16 requests at once
  by default (`--parallel N`: N); the context is the model's (or `--ctx N`), at most what the
  pages hold. Placed automatically, the whole model stays on the GPU if it fits beside an
  8192-token cache, whatever `--ctx` says (the pool takes what is left). A request starts
  when a slot is free and the pages for its prompt are (with a page per running request to
  grow into); else it waits in the queue. Idle slots keep their prefixes (a later request
  sharing one reuses its rows) until their pages are needed, least recently used first.
  When decode runs out of pages, the youngest prompt still being prefilled gives its pages
  back, then the youngest running request is preempted (fed again later, the same output);
  the oldest never is. The banner says the pool's shape (`KV: P pages x 64 = T tokens over
  up to S slots (cap C)`), `GET /slots` each slot's `pages_used` and the pool's
  `pages_free` / `pages_total`. Measured on a 12 GB laptop GPU at the default 32768: Qwen2.5-7B
  Q4_K_M gets 1537 pages (98k tokens: 6 agents at 15k each; `--kv q8`: 2983 pages, 12), where
  the reserved pool held 3 slots; Qwen2.5-14B Q4_K_M with `--kv q8 --ctx 32768` stays on the
  GPU with ~15k tokens of pages instead of splitting. The request running alone runs in a slot's pages (its own
  scratch beside it, no cache of its own), picked as a batched request's is: the slot
  holding the longest prefix of its prompt if that prefix is at least half the slot's
  tokens or more than half the prompt's, else the one idle the longest (a shared <bos> or
  system prompt doesn't evict a conversation). With `AXON_PAGED_KV=0`, without `--ctx` or
  `--parallel` the context comes first: as many slots (2 to 8) as fit with the whole
  context; if not even 2 do, 2 or more with the longest context they fit with, down to
  32768 (or the whole context, if shorter); below that, one request at a time with the
  whole context. `--parallel N` alone: N slots of the longest context that fits them (down
  to 1024 positions). `--ctx N`: never shortened; the slots that fit with it (up to
  `--parallel`), else one at a time. The server says on stderr what it chose and why; the
  context it serves is a slot's (`/v1/models` `meta.n_ctx`; longer prompts get a 400
  `context_length_exceeded`). The Q6_K prefill copy (int8 Q6_K matrices for prefill's tensor
  cores, same results; 1.02 GiB on 7B Q4_K_M) is made after the caches and slots, only from
  the memory they leave (`AXON_Q6K8=0` never makes it).
  A request running alone takes the single-stream path (CUDA graphs, `--draft`), and
  moves into the batch when a second one arrives (its rows are already in its slot: nothing
  is copied). Every request's output is exactly what it
  gets alone, whatever else runs with it (greedy, and sampled with a `seed`): the batched
  kernels are bit-identical per row to the one-sequence ones. On the CPU, with
  `--gpu-layers` (a CPU/GPU split) and with a multi-GPU pipeline (several `AXON_GPUS`) the
  server runs one request at a time; an explicit `--parallel N` (N > 1) there is an error.
- Requests run each exactly as `gen` / `chat` would run it (greedy `/v1/completions` output
  is byte-identical to `axon gen`'s). The KV cache carries over between requests: a prompt
  that starts with tokens a cache (or a slot) already holds (a chat follow-up resending the
  conversation, a shared system prompt) only runs the model over the rest
  (`usage.prompt_tokens_details.cached_tokens`).
- Chat prompts are framed as `axon chat` frames them (ChatML, or Llama 3's template with its
  BOS and system block; Qwen2.5's default system message when none is given; Qwen3 earlier
  replies without their reasoning), and end on the same tokens (Llama 3: `<|eot_id|>` too).
  `/v1/completions` tokenizes and stops as `gen` does (BOS for Llama 3). Qwen3 thinks
  unless `chat_template_kwargs: {"enable_thinking": false}` (or `--no-think` as the server
  default); the reasoning comes as `reasoning_content`, apart from `content`
  (`--think-inline` leaves the `<think>` block in `content`).
- Sampling: `temperature` (0 = greedy), `top_p`, `seed`, and the common extensions `top_k`,
  `min_p`, `repetition_penalty` (or `repeat_penalty`) and `repeat_last_n`; `max_tokens` (or
  `max_completion_tokens`), `stop` (up to 16 strings, not included in the output). What a
  request leaves out comes from the server's sampling flags, else `chat`'s defaults
  (temperature 0.8, top-k 40, top-p 0.9) but with no repeat penalty (1.0, as OpenAI and
  llama-server: `chat`'s 1.1 would quietly penalize the punctuation and indentation of code
  and JSON; `--repeat-penalty` or a request's `repetition_penalty` turns it on). For Qwen3's
  thinking mode Qwen recommends temperature 0.6, top_p 0.95, top_k 20 (greedy can loop on
  long reasoning).
  `presence_penalty` / `frequency_penalty` other than 0, `n` > 1, `logprobs`, legacy
  `functions`, `logit_bias` and JSON `response_format` are refused with a 400 rather than
  ignored.
- Tool calling: see below.
- `finish_reason` is `stop` (end token or stop string) or `length`; `usage` counts tokens;
  a `timings` object gives prompt and generation speed (llama-server's field names).
- A client that disconnects stops its generation within one token.
- Limits: 16 KiB of request head, 1 MiB of body (`Content-Length` or chunked), 256
  connections, 128 queued requests (more: 503); keep-alive and pipelining; idle connections
  close after 30 s. No TLS or HTTP/2: put a reverse proxy in front to expose it beyond
  localhost.
- If the engine fails on a request (a CUDA error is sticky: every later request would fail
  too), the server logs the error and exits with code 70, for a supervisor to restart it.

#### Operations: auth, metrics, slots

- `--api-key KEY` (or `AXON_API_KEY`, which keeps the key out of `ps`): every route but
  `GET /health` needs `Authorization: Bearer KEY` (compared in constant time); without it a
  401 with an OpenAI-shaped error (`code: "invalid_api_key"`) and `WWW-Authenticate: Bearer`.
  CORS preflights (`OPTIONS`) stay open: browsers send them without credentials, and they
  answer with headers only. A request with a body but without the key gets the 401 right
  after its headers, whatever the route (`/health` and `OPTIONS` are open only without a
  body), before a 413 for a body past 1 MiB and without `100 Continue`; its body is never
  buffered or parsed, and the connection closes (after dropping what the client already
  sent, at most 2 s / 4 MiB, so the 401 isn't lost to a reset; an upload still going after
  that can be reset before the client reads the 401). `--api-key` wins over `AXON_API_KEY` (the variable isn't
  read then); the variable is trimmed (a secret file's trailing newline), and set but blank
  or not UTF-8 is an error. stderr says `api key: on (from AXON_API_KEY|--api-key)`, never
  the key. Still plain HTTP: put TLS in front of it beyond a trusted LAN.
- `GET /metrics`: Prometheus text format. `axon_requests_total{route,status}`,
  `axon_generated_tokens_total`, `axon_prompt_tokens_total`,
  `axon_prompt_cached_tokens_total` (prefix reuse), `axon_requests_in_flight`,
  `axon_queue_depth` (in flight, not in a slot), `axon_slots_busy` / `axon_slots_free`,
  `axon_slot_idle_evictions_total`, `axon_prompt_refused_total`, and the histograms
  `axon_ttft_seconds` (from the request's body read, so tokenizing and queueing count, to
  its first text) and `axon_itl_seconds` (between a request's
  text events, about one token each). No host-RAM KV store hits yet: serve has no store.
- `GET /slots`: JSON, `{"ctx", "slots": [{"id", "busy", "held_tokens", "request_id",
  "age_ms"}], "queue_depth"}`. `held_tokens`: the prefix whose KV rows the slot holds (what
  a later prompt can reuse); `request_id` is the `#n` of the stderr log. A request running
  alone is in the slot it runs in (on the CPU, a split or a pipeline, the one cache is slot 0).
- `--max-prompt-tokens N`: a longer prompt is refused before prefill with a 400
  (`code: "context_length_exceeded"`, as a prompt past the context). It counts the whole
  rendered prompt: the chat template's tokens and the tool schemas too; exactly N is allowed.
- `--slot-idle-ms MS` (at least 1000: delayed ACKs and retransmits pause a live client's
  acknowledgements for hundreds of ms): a stream whose client stops reading is cancelled (as
  a disconnect: its slot freed for the next request). A client that reads slowly isn't
  affected. A stopped one can't be told from a slow one until its receive buffer is full, so
  the clock starts only then: on Linux (x86, arm, riscv64, s390x, loongarch64), once the
  unacknowledged bytes stop moving for MS ms. With a default ~128 KiB receive window that is
  roughly 500+ tokens of SSE after the client stopped (MEASURED, bella CPU, 0.5B,
  `--slot-idle-ms 1000`, a client with default buffers: 792 tokens and 6.3 s from its last
  read to the slot freed, and in another run the autotuned buffers absorbed a whole
  1900-token reply; with a 4 KiB `SO_RCVBUF`, 158-202 tokens and 1.7 s). A reply shorter
  than that just finishes into the buffers. Elsewhere only a write blocked for MS ms (at most the
  60 s write timeout) is detected, after the server's send buffer fills too, and stderr says
  so at startup.
- A request's `priority` field is accepted and ignored (requests run in arrival order).

#### Tool calling

`tools`, `tool_choice` and tool turns in the history (`role: "tool"` results, assistant
messages with `tool_calls` and null or absent `content`) work on `/v1/chat/completions`, so
agent harnesses that talk to a local OpenAI-compatible endpoint can run on axon:

```bash
axon serve models/Qwen3-8B-Q4_K_M.gguf --device cuda --ctx 32768 --kv q8 --alias qwen3-8b
```

- The tools are rendered as the model's own chat template does (the GGUF's
  `tokenizer.chat_template`, checked against transformers' rendering in tests). Qwen3 and
  Qwen2.5 get the Hermes `# Tools` / `<tools>` block in the system message, call with
  `<tool_call>{"name": ..., "arguments": ...}</tool_call>`, and see results as
  `<tool_response>` blocks. Llama 3.1/3.2 get the tools in the first user message, call with
  a JSON reply (`{"name": ..., "parameters": ...}`) and see results as `ipython` turns (one
  call per message, as the template requires). Other chat formats answer a tool request
  with a 400. A request without tools or tool turns is rendered exactly as before.
- The reply's calls come back as OpenAI `tool_calls`, with `finish_reason: "tool_calls"`.
  Streamed: a call is sent as soon as its block closes, as two `delta.tool_calls` chunks: the
  first carries `index`, `id`, `type` and `function.name`, the second `function.arguments`.
  Text outside the blocks streams as `content`. Nothing of a block reaches `content`
  while it is still being decided, and blank lines that only separate text from calls are
  dropped. Not streamed: `message.tool_calls`, with `content` null if there was no text.
  Call ids are `call_` plus 24 hex digits.
- A block that isn't a JSON object with a string `"name"` (invalid JSON, or the reply was
  cut off mid-call) is not a call: it is returned as `content`, tags and all. So
  `arguments` is always valid JSON. A string `arguments` value is used as the arguments if
  it is itself JSON, else returned as its JSON string literal. Qwen3's reasoning still
  comes as `reasoning_content`.
- The parser reads tags, not prose or JSON. `<tool_call>` written in ordinary text opens a
  block (which comes back as content if it isn't a call), and a `</tool_call>` inside a
  JSON string value ends the block early (the call then fails to parse and is returned
  as content).
- The format is chosen per model family from the vocabulary (ChatML with or without
  `<think>`, or Llama 3's headers), not read from the GGUF's template. Models of a known
  family with a different tool format, such as Qwen3-Coder's XML `<function=...>` calls,
  are not supported.
- `tool_choice`: `auto` (default); `none` (the tools are not shown to the model, and its
  reply is plain content); `required` or `{"type": "function", "function": {"name": ...}}`.
  The last two start the reply with the opening of a call (the named one: through its
  `"arguments": `), so the model must call. Qwen3 then skips its reasoning. Without
  `tools` they are a 400.
  `parallel_tool_calls`, `strict`, `store`, `reasoning_effort` and `prompt_cache_key` are
  accepted and ignored.
- Context: an agent's system prompt and tool schemas can be 15-25k tokens; the default
  context is the model's trained length where it fits (above), so no `--ctx` is needed.
  `--kv q8` halves the cache's memory (more slots, or the whole context on a smaller GPU). The prefix cache works across tool turns: the next
  request resends the history with the results appended, and only the tokens after the
  previous reply's start are run again, as long as no other request evicted the prefix in
  between. There is one KV cache, so a harness's side requests (title generation,
  summaries) or a second client make the next turn prefill the system prompt and tools again.
- Harnesses: anything speaking OpenAI Chat Completions, e.g. Synaps (`provider.local.url`),
  opencode and crush (an OpenAI-compatible provider whose base URL is
  `http://127.0.0.1:8080/v1`), and pi (an `openai-completions` model in `models.json`).
  Codex CLI and pi's `openai-responses` speak the Responses API, and ollama clients the
  ollama API: both are served too (below).

#### The Responses API and the ollama API

`POST /v1/responses` (OpenAI's Responses API, what codex speaks) and ollama's native API
(`POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `POST /api/show`, `GET /api/ps`,
`GET /api/version`, `GET /`) are served next to chat completions, so a client written for
OpenAI or for ollama works unchanged. Each request is translated into the chat-completions
request and run the same way: the same `--api-key` gate, size limits, queue, batching and
cache, and greedy output identical to `/v1/chat/completions` for the same messages (a test
checks it on every route). [docs/serve-apis.md](docs/serve-apis.md) lists every field.

```bash
# codex, against a local provider (~/.codex/config.toml, or a CODEX_HOME of its own):
#   model = "qwen3-8b"
#   model_provider = "axon"
#   [model_providers.axon]
#   name = "axon"
#   base_url = "http://127.0.0.1:8080/v1"
#   wire_api = "responses"
codex exec "Read hello.txt and tell me the magic number"
# or codex's own ollama provider:
CODEX_OSS_BASE_URL=http://127.0.0.1:8080/v1 codex exec --oss --local-provider ollama -m qwen3-8b "..."
# any ollama client:
OLLAMA_HOST=http://127.0.0.1:8080 ollama run qwen3-8b
```

- `/v1/responses`: `input` (a string, or `message`, `function_call`,
  `function_call_output` and `reasoning` items), `instructions`, function `tools` (other
  tool types, such as codex's `web_search` and `namespace`, are dropped), `tool_choice`,
  `stream` (OpenAI's events, `response.created` through `response.completed`),
  `max_output_tokens`, sampling, `usage`, and `previous_response_id` (responses with
  `store`, the default, are kept in memory: the last 256). Structured output, background
  responses and conversations are a 400.
- ollama: `stream` defaults to true (NDJSON); `options` carries the sampling settings
  (`num_predict` is the token limit; `num_ctx` and the runtime options are ignored: the
  server's `--ctx` applies); `think`; tool calls with object arguments; ollama's counts and
  durations. The model answers to its file name and `--alias`, with or without `:latest`.
  `format` (JSON / structured output), images and embeddings are refused with a clear error.

### Environment

| variable | effect |
|---|---|
| `AXON_THREADS=N` | worker threads (default: cores minus a quarter) |
| `AXON_SIMD=scalar\|avx2\|vnni` | force a kernel set (default: the best the CPU has) |
| `AXON_DECODE_THREADS=N` | CPU: at most N threads per decode-step job (default: the P-core count on a hybrid Intel CPU without SMT, else every thread; prefill always uses every thread) |
| `AXON_POOL_ADAPT=0` | keep every worker in every job (no adaptive drop-out under load) |
| `AXON_POOL_DEBUG=1` | `gen`: print thread-pool stats at the end |
| `AXON_MMAP=0\|1` | CPU: `0`: copy the model into huge-page memory; `1`: map the file. Default: map it when every weight matrix is repacked (Q4_K/Q6_K/Q5_0/Q8_0 models: the repack holds them in huge pages, so a copy would only be a second pass over the file), else copy unless the file is over 60% of available memory. A mapped model stays mapped while axon runs: overwriting the GGUF in place meanwhile (e.g. `curl -o` onto the same file) can crash it with SIGBUS; use `AXON_MMAP=0`, or write a new file and rename it over the old one. The GPU always maps (it reads the file once), and so does a mixture-of-experts model (`qwen3moe`, `AXON_MMAP=0` included: a copy beside its repacked experts would hold them twice) |
| `AXON_READAHEAD=0` | `gen`/`chat`: don't read a mapped model into the page cache ahead of use on a background thread (on by default unless the file is over half of available memory) |
| `AXON_PREFETCH=N` | decode: software prefetch distance in bytes (default 8192, 0 = off, max 1 MiB) |
| `AXON_DECODE_ROWS=1\|2\|4` | decode: rows per kernel call (default 4; K-quants use at most 2) |
| `AXON_REPACK=0` | CPU: run every matrix in place from the GGUF's blocks instead of repacking Q4_K/Q6_K/Q5_0/Q8_0 for the faster kernels (for comparisons) |
| `AXON_DRAFT=K` | same as `--draft K` (gen, chat, serve): prompt-lookup speculative decoding, up to K draft tokens per step (default 0 = off; try 7). Lossless: same output as without it (greedy, or sampled with the same seed) |
| `AXON_DRAFT_MODEL=PATH` | same as `--draft-model PATH` (gen, chat, serve) |
| `AXON_DEVICE=cpu\|cuda` | same as `--device` (default `cpu`) |
| `AXON_GPU_LAYERS=N` | same as `--gpu-layers N` (with `--device cuda`) |
| `AXON_GPUS=L` | same as `--gpus L` (gen, chat, serve; with `--device cuda`) |
| `AXON_PP_CUTS=K[,K2..]` | `--gpus` pipeline: the first layer of each GPU after the first (default: even if it fits, else by free memory) |
| `AXON_P2P=1` | `--gpus` pipeline: try direct peer-to-peer copies between the GPUs. UNTESTED (no P2P-capable box has run it); default: copies staged through host memory by the driver (tested) |
| `AXON_KV=f16\|f32\|q8` | KV cache type (default f16; `f32` CPU only, for A/B; `q8` cuda only, as `--kv q8`) |
| `AXON_STREAM=0` | split model: don't stream the CPU layers' weights through the GPU for prefill batches (default on: they are copied over once per prompt chunk, and those layers then get the GPU's numerics) |
| `AXON_STREAM_MIN=N` | split model: smallest prefill batch streamed (default 32). Streaming pins the CPU layers' KV cache in RAM |
| `AXON_STREAM_DECODE=1` | split model, experiment: stream one-token decode steps too (slower; refused with `--draft`) |
| `AXON_CUDA_GRAPHS=0` | cuda: launch each token's kernels one by one instead of replaying a captured CUDA graph (same results, slower) |
| `AXON_MOE_PROFILE=1` | cuda, mixture-of-experts model: every 64 decode steps, print per layer the GPU segment (attention + router; synced for the profile, so a little slower), the copy down, the CPU experts and the copy up |
| `AXON_CUDA_PDL=0` | cuda: no programmatic dependent launch (used only on sm_90+ with a CUDA 12.3+ driver) |
| `AXON_ATT_MQ=0` | cuda, tensor cores: a one-sequence batched pass of few queries (a prompt chunk, a draft verify) runs attention per query (`attn_blk_b`, ≤ 16 queries at head size 128, ≤ 48 at 64) and above that the flash kernel, instead of `attn_blk_mq` (16 queries per K/V tile read). Same output bits |
| `AXON_ATT_MQ_MAX=N` | cuda, tensor cores: most queries of a one-sequence pass that `attn_blk_mq` takes before the flash kernel (default 32 at head size 128, 48 at 64). Parsed as an integer; anything else, the default. Clamped up to the per-query limit (16 at head size 128, 48 at 64). Large values turn the flash kernel off for one-sequence passes of up to N tokens (and prompts are cut into the plain passes): slower past the crossover, same bits |
| `AXON_GATE_UP=0` | cuda: run the FFN's gate and up products (both Q4_K) as two GEMMs instead of one fused launch in batched passes (same bits) |
| `AXON_HOPPER=0` | cuda, H100/H200 (compute capability 9.0): compile the kernels for plain sm_90 instead of sm_90a, which turns the wgmma/TMA kernels off (same bits) |
| `AXON_GEMMH=0\|1` | cuda, sm_90a: `0` keeps every batched Q4_K product on the mma.sync GEMMs; `1` sends every one of them to the wgmma kernel. Default: the fused gate+up and the large Q4_K products (>= 3584 rows, at least one full wave of blocks) use wgmma (same bits either way) |
| `AXON_Q6K8=0\|1` | cuda: `0` never makes the int8 copy of Q6_K matrices that speeds up prefill (sm_80+); `1` makes it whenever it fits beside the cache. Default: only if 768 MiB stay free beside it and the cache, and never on a split (same bits either way) |
| `AXON_SKIP_MODEL_TESTS=1` | tests: skip the ones that need the real model |

## Numbers

tok/s, median of 3. `pp560` / `pp2100` are prompt evaluation over ~560 / ~2100
tokens, `tg128` decode over 128 tokens, `@1` one thread. Compare engines within one
machine. Methods, ranges and more runs are in [BENCH.md](BENCH.md) (CPU, dev laptop GPU)
and [docs/benchmarks-gpu.md](docs/benchmarks-gpu.md) (RunPod, RTX 4050 laptop, caveats).

Datacenter and desktop GPUs (RunPod, main @ f8d8a86, 2026-10-15): axon ahead of both
llama.cpp 4413a19 (`llama-bench -ngl 99`) and ollama 0.35.0 in all 45 cells (3 GPUs ×
Qwen2.5 0.5B/1.5B/3B/7B and Qwen3 8B Q4_K_M × pp560/pp2100/tg128). Qwen2.5 7B, axon /
llama.cpp / ollama, and axon's range over llama.cpp across the five models:

| GPU | 7B pp2100 | 7B tg128 | pp2100 vs llama.cpp | tg128 vs llama.cpp |
|---|---|---|---|---|
| RTX 4090 | 13842 / 11937 / 12313 | 185 / 176 / 174 | 1.09-1.90x | 1.05-1.31x |
| RTX 5090 | 18609 / 15416 / 15832 | 308 / 280 / 279 | 1.21-1.94x | 1.10-1.46x |
| H100 SXM | 12534 / 9691 / 9711 | 311 / 241 / 247 | 1.19-2.10x | 1.22-1.34x |

The H100's prefill trails the 4090's: in this run axon's int8 GEMMs used `mma.sync`, not Hopper's
`wgmma`. Since #33 the fused gate+up and the large Q4_K products run on `wgmma` on sm_90a (+12 % on a
7B prefill pass with `AXON_GEMMH=1`, H100 PCIe; the new default routing itself is not yet measured on an H100; [docs/design/hopper.md](docs/design/hopper.md)). pp560 gaps are wider (up to
15x) but mostly measure the other engines' fixed per-request cost; quote pp2100.

Parabellum, the dev laptop (from BENCH.md): main @ 0851798 (`--device cuda`, whole model on the GPU) vs ollama 0.35.0
and llama.cpp 4413a19 (CUDA build, `llama-bench -ngl 99`), same GGUF files, on an
NVIDIA GeForce RTX 5070 Ti Laptop GPU (the laptop part, not the desktop card; sm_120,
12 GB), 2026-10-15:

| model | engine | pp560 | pp2100 | tg128 |
|---|---|---|---|---|
| Qwen2.5 0.5B Q4_K_M | axon 0851798 | 44779.7 | 44681.0 | 825.4 |
| | ollama 0.35.0 | 14288.9 | 42593.6 | 583.2 |
| | llama.cpp | 25333.7 | 42530.7 | 692.9 |
| Qwen2.5 1.5B Q4_K_M | axon 0851798 | 21353.3 | 20168.4 | 419.7 |
| | ollama 0.35.0 | 12157.8 | 17303.7 | 370.5 |
| | llama.cpp | 13025.7 | 17320.2 | 405.9 |
| Qwen2.5 3B Q4_K_M | axon 0851798 | 10800.9 | 10273.0 | 255.9 |
| | ollama 0.35.0 | 7515.5 | 9415.9 | 229.4 |
| | llama.cpp | 7563.1 | 9302.3 | 206.1 |
| Qwen2.5 7B Q4_K_M | axon 0851798 | 5040.2 | 4898.1 | 118.0 |
| | ollama 0.35.0 | 4227.1 | 4800.2 | 111.3 |
| | llama.cpp | 4412.0 | 4105.1 | 112.0 |
| Qwen3 0.6B Q8_0 | axon 0851798 | 35792.7 | 31853.1 | 646.0 |
| | ollama 0.35.0 | 20215.7 | 30777.8 | 472.5 |
| | llama.cpp | 20178.9 | 30827.2 | 541.6 |
| Qwen3 4B Q4_K_M | axon 0851798 | 8206.4 | 7780.6 | 204.6 |
| | ollama 0.35.0 | 6008.5 | 7213.9 | 172.6 |
| | llama.cpp | 6264.7 | 7091.3 | 163.4 |
| Qwen3 8B Q4_K_M | axon 0851798 | 4697.4 | 4512.1 | 110.0 |
| | ollama 0.35.0 | 3806.2 | 4248.5 | 104.0 |
| | llama.cpp | 3919.1 | 3791.3 | 105.7 |

main has since gained the round-5 prefill changes (#24, 29c98a7): +2-23% pp2100 in
that commit's own measurements, depending on the model (most on the small ones);
the table predates them.

CPU, an older run (from BENCH.md: main @ 74fe85e, 2026-10-13, before the GPU backend;
Qwen2.5-0.5B-Instruct Q4_K_M, Parabellum's Intel Core Ultra 9 275HX, 18 threads, quiet machine):

| engine | pp560 | pp2100 | tg128 | tg128@1 |
|---|---|---|---|---|
| axon 74fe85e | 1294.7 | 1068.8 | 163.8 | 29.9 |
| ollama 0.35.0 | 1027.3 | 961.2 | 161.6 | 34.4 |

How to reproduce the CPU runs (`scripts/bench.py`) and the loaded-machine numbers
are in BENCH.md.

## How it works

Read in this order:

1. **`src/gguf.rs`**: the file format. A header, key/value metadata, a tensor
   table, then the raw tensor bytes, used in place (never converted).
2. **`src/quant.rs`**: block formats and dequantization, then the scalar dot
   products (`dot_*_q8`, quantized weights times 8-bit activations). Read those;
   the SIMD versions below them compute the same sums and can wait.
3. **`src/model.rs`, `hidden()`**: the whole transformer for one token in
   numbered steps (norm, QKV, RoPE, KV cache, attention, output projection,
   SwiGLU FFN, residuals). Then `feed_batch`, the same math for 64 prompt
   tokens at once, reading each weight row once per batch.
4. **`src/main.rs`, `run_lm`**: encode the prompt, prefill in batches, then
   `generate`: `forward`, pick the next token (argmax, or `sample.rs`), repeat.

`pool.rs` (the worker pool) and `tokenizer.rs` (byte-level BPE) can be read
any time.

The GPU, after that:

5. **`src/lm.rs`**: the `Lm` trait (`forward`, `feed_batch`, `verify_argmax`, ...)
   that generation, chat and the server are written against, and `Backend`, which
   holds the CPU, GPU or split model the command line picked.
6. **`src/cuda/qwen2.rs`** and **`decode.cu`**: the model on the GPU. Weights are
   uploaded once in their GGUF block formats; one token is a fixed sequence of kernels
   per layer, captured once into a CUDA graph and replayed every token.
7. **`src/cuda/qwen2/batch.rs`** and **`prefill.cu`**: batched passes (prefill,
   speculative verify) on int8 tensor cores, ordered so that every token gets the same
   bits as the one-token step. `src/cuda/mod.rs` and `ffi.rs` load the driver and NVRTC.
8. **`src/hybrid.rs`**: a model split across devices: the first layers on the CPU, the
   rest and the head on the GPU, one hidden-state copy between them per token; for
   prompts, the CPU layers' weights can be streamed through the GPU instead.

## Docs

- [docs/START-HERE.md](docs/START-HERE.md): what Axon is, what's on main, the bit-identity
  contract, a map of the source tree
- [docs/serve-apis.md](docs/serve-apis.md): `axon serve`'s routes and the fields each takes
  (chat completions, completions, the Responses API, the ollama API)
- [docs/determinism.md](docs/determinism.md): what is bit-identical (and what isn't), the
  identity tuple behind every response's `system_fingerprint` (`GET /identity` on `axon serve`)
- [docs/benchmarks-gpu.md](docs/benchmarks-gpu.md): GPU scoreboards with method and caveats
- [docs/roadmap.md](docs/roadmap.md): what's next, and what isn't planned
- [docs/methodology.md](docs/methodology.md): how changes are picked, built, reviewed, gated and benchmarked
- [docs/history.md](docs/history.md): the engineering log, round by round, with before/after numbers
- [docs/experiments-rejected.md](docs/experiments-rejected.md): what was tried and didn't work, with numbers
- [docs/results.md](docs/results.md): current standing vs ollama and llama.cpp, with dates and commits
- [docs/research/](docs/research/README.md): research reports behind the roadmap
- [docs/log/](docs/log/README.md): one entry per PR (the convention for new work)
- [docs/long-context.md](docs/long-context.md): YaRN (`--rope-scaling yarn`) past the trained
  context: when to use it, llama.cpp agreement, retrieval and short-context cost
- [docs/design/](docs/design/): [multi-GPU](docs/design/multi-gpu.md) and
  [Hopper](docs/design/hopper.md) designs

## Contributing

Contributions from people and from their AI agents are welcome. The work queue is the issue board.

- **[AGENTS.md](AGENTS.md)** first: how to claim an `agent-ready` issue, the hard rules (bit identity,
  the md5 gate against `main`, no new crates), how to measure, and the evidence every PR must include.
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: setup, test models, oracles, the checks to run.

## License

MIT, see [LICENSE](LICENSE). Parts of the kernels and the tokenizer follow
llama.cpp/ggml (MIT); test fixtures contain rows of Qwen2.5 weights
(Apache-2.0). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
