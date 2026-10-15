# Axon

LLM inference for Qwen2, Qwen3 and Llama 3 GGUF models on the CPU and on NVIDIA
GPUs, written from scratch in Rust.

No ggml, no BLAS, no binding crates, no CUDA toolkit at build time: a GGUF reader,
quantized integer dot-product kernels (scalar, AVX2, AVX-VNNI), a spinning worker
pool and a transformer forward pass on the CPU; on the GPU, CUDA C++ kernels that
NVRTC compiles at run time, launched through the driver API (both libraries loaded
with dlopen). About 41k lines of Rust and 4.7k of CUDA, tests included.

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
  frequency factors and the llama-bpe tokenizer with its BOS token are handled.
  Other architectures, mixture-of-experts models and RoPE scaling (a factor other than 1)
  are refused with an error.
- Tensor types: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K on the CPU. On the
  GPU every weight matrix must be one of the eight quantized types (F16/F32 matrices are
  refused there). Others load for `info` but can't run.
- Sampling: greedy, or temperature / top-k / top-p / min-p / repeat penalty with a seed
  (llama.cpp's order, ollama's defaults). `gen` continues raw text, greedy unless `--temp`
  is above 0 (reproducible output and benchmarks); `chat` talks to an Instruct model
  through its chat template (ChatML, or Llama 3's) and samples by default (`--temp 0` for greedy). On the GPU
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
  NVRTC gets PTX for the driver to compile; one older than every architecture NVRTC
  supports is refused.
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
- Prompt-lookup drafting (`--draft K`, off by default): speculative decoding that guesses
  the next tokens from earlier text (prompt and output) and verifies them in one batched
  pass. Lossless: the same output as without it (greedy, or sampled with the same seed).
- Serving: `axon serve`, an OpenAI-compatible HTTP server (below).
- Rust 1.89 or newer.

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
axon gen <model.gguf> "<prompt>" [-n N] [--draft K] [--device D] [--gpu-layers N] [--kv T]
                                          generation (default 32 tokens; greedy unless --temp)
axon logits <model.gguf> "<prompt>" [-k K]  top-K next-token logits (default 10)
axon chat <model.gguf> [--system TEXT] [--no-think] [--ctx N] [-n N] [--draft K]
          [--device D] [--gpu-layers N] [--kv T]
                                          talk to an Instruct model (sampled; --temp 0: greedy)
sampling (gen, chat): --temp T --top-k K --top-p P --min-p P
          --repeat-penalty P --repeat-last-n N --seed S   (axon --help)
axon serve <model.gguf> [--host H] [--port P] [--ctx N] [--draft K] [--device D]
          [--gpu-layers N] [--kv T] [...]  OpenAI-compatible HTTP server (below)
axon --help | --version
```

Defaults, as `axon --help` gives them:

- `--draft K`: 0 (off); try 7.
- `--device D`: `cpu`, or `cuda`. A model too big for the GPU's memory is split
  automatically: the last layers that fit on the GPU, the first ones on the CPU.
- `--gpu-layers N` (with `--device cuda`): the last N layers and the output head on the
  GPU, the rest on the CPU (0 = all on the CPU). Default: all of them if they fit with
  256 MiB to spare, else as many as fit with another 640 MiB left for the desktop; an
  explicit N only needs the 256 MiB.
- `--kv T`: `f16`, or `q8` (cuda only; on a split, the GPU layers' cache, so more
  layers fit).
- `--ctx N` (chat, serve): the model's trained context, at most 8192. On cuda, if the
  whole model doesn't fit with it, a shorter context if at least half of it (or 4096)
  fits, else a split.
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
conversation outgrows `--ctx` (default: the model's, at most 8192), the oldest exchanges are forgotten.

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
`POST /v1/completions` (raw text, no template, as `gen`), `GET /health`; streamed
(server-sent events, `data: [DONE]` at the end; `stream_options.include_usage` adds a
usage chunk) or whole. It takes the device, `--ctx`, `--kv`, `--gpu-layers` and `--draft`
flags of `chat`, plus `--host` (default 127.0.0.1), `--port` (default 8080), `--alias NAME`
(another model name to answer to), `--max-tokens N` (a cap on any request's generation;
default: up to the context), `--cors [ORIGIN]` (answer CORS preflights and allow ORIGIN,
default `*`, to call the server from web pages; off by default, so an arbitrary page can't
use your local model), `--max-conns N` / `--max-queue N` (defaults 256 / 128), `--quiet`
(no per-request line on stderr). `GET /health` is 503 until the model is loaded.

- One request at a time: requests queue and run in arrival order, each exactly as `gen` /
  `chat` would run it (greedy `/v1/completions` output is byte-identical to `axon gen`'s).
  The KV cache carries over between requests: a prompt that starts with the previous
  request's tokens (a chat follow-up resending the conversation, a shared system prompt)
  only runs the model over the rest (`usage.prompt_tokens_details.cached_tokens`).
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
  `presence_penalty` / `frequency_penalty` other than 0, `n` > 1, `logprobs`, `tools`,
  `logit_bias` and JSON `response_format` are refused with a 400 rather than ignored.
- `finish_reason` is `stop` (end token or stop string) or `length`; `usage` counts tokens;
  a `timings` object gives prompt and generation speed (llama-server's field names).
- A client that disconnects stops its generation within one token.
- Limits: 16 KiB of request head, 1 MiB of body (`Content-Length` or chunked), 256
  connections, 128 queued requests (more: 503); keep-alive and pipelining; idle connections
  close after 30 s. No TLS or HTTP/2: put a reverse proxy in front to expose it beyond
  localhost.
- If the engine fails on a request (a CUDA error is sticky: every later request would fail
  too), the server logs the error and exits with code 70, for a supervisor to restart it.

### Environment

| variable | effect |
|---|---|
| `AXON_THREADS=N` | worker threads (default: cores minus a quarter) |
| `AXON_SIMD=scalar\|avx2\|vnni` | force a kernel set (default: the best the CPU has) |
| `AXON_POOL_ADAPT=0` | keep every worker in every job (no adaptive drop-out under load) |
| `AXON_POOL_DEBUG=1` | `gen`: print thread-pool stats at the end |
| `AXON_MMAP=0\|1` | CPU: `0`: copy the model into huge-page memory; `1`: map the file. Default: map it when every weight matrix is repacked (Q4_K/Q6_K/Q5_0/Q8_0 models: the repack holds them in huge pages, so a copy would only be a second pass over the file), else copy unless the file is over 60% of available memory. A mapped model stays mapped while axon runs: overwriting the GGUF in place meanwhile (e.g. `curl -o` onto the same file) can crash it with SIGBUS; use `AXON_MMAP=0`, or write a new file and rename it over the old one. The GPU always maps (it reads the file once) |
| `AXON_READAHEAD=0` | `gen`/`chat`: don't read a mapped model into the page cache ahead of use on a background thread (on by default unless the file is over half of available memory) |
| `AXON_PREFETCH=N` | decode: software prefetch distance in bytes (default 8192, 0 = off, max 1 MiB) |
| `AXON_DECODE_ROWS=1\|2\|4` | decode: rows per kernel call (default 4; K-quants use at most 2) |
| `AXON_REPACK=0` | CPU: run every matrix in place from the GGUF's blocks instead of repacking Q4_K/Q6_K/Q5_0/Q8_0 for the faster kernels (for comparisons) |
| `AXON_DRAFT=K` | same as `--draft K` (gen, chat, serve): prompt-lookup speculative decoding, up to K draft tokens per step (default 0 = off; try 7). Lossless: same output as without it (greedy, or sampled with the same seed) |
| `AXON_DEVICE=cpu\|cuda` | same as `--device` (default `cpu`) |
| `AXON_GPU_LAYERS=N` | same as `--gpu-layers N` (with `--device cuda`) |
| `AXON_KV=f16\|f32\|q8` | KV cache type (default f16; `f32` CPU only, for A/B; `q8` cuda only, as `--kv q8`) |
| `AXON_STREAM=0` | split model: don't stream the CPU layers' weights through the GPU for prefill batches (default on: they are copied over once per prompt chunk, and those layers then get the GPU's numerics) |
| `AXON_STREAM_MIN=N` | split model: smallest prefill batch streamed (default 32). Streaming pins the CPU layers' KV cache in RAM |
| `AXON_STREAM_DECODE=1` | split model, experiment: stream one-token decode steps too (slower; refused with `--draft`) |
| `AXON_CUDA_GRAPHS=0` | cuda: launch each token's kernels one by one instead of replaying a captured CUDA graph (same results, slower) |
| `AXON_CUDA_PDL=0` | cuda: no programmatic dependent launch (used only on sm_90+ with a CUDA 12.3+ driver) |
| `AXON_GATE_UP=0` | cuda: run the FFN's gate and up products (both Q4_K) as two GEMMs instead of one fused launch in batched passes (same bits) |
| `AXON_Q6K8=0\|1` | cuda: `0` never makes the int8 copy of Q6_K matrices that speeds up prefill (sm_80+); `1` makes it whenever it fits beside the cache. Default: only if 768 MiB stay free beside it and the cache, and never on a split (same bits either way) |
| `AXON_SKIP_MODEL_TESTS=1` | tests: skip the ones that need the real model |

## Numbers

tok/s, median of 3. `pp560` / `pp2100` are prompt evaluation over ~560 / ~2100
tokens, `tg128` decode over 128 tokens, `@1` one thread. One machine: treat the
numbers as relative. Methods, ranges and more runs are in [BENCH.md](BENCH.md).

GPU (from BENCH.md): main @ 0851798 (`--device cuda`, whole model on the GPU) vs ollama 0.35.0
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
Qwen2.5-0.5B-Instruct Q4_K_M, Intel Core Ultra 9 275HX, 18 threads, quiet machine):

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): test model, oracles, checks to run.

## License

MIT, see [LICENSE](LICENSE). Parts of the kernels and the tokenizer follow
llama.cpp/ggml (MIT); test fixtures contain rows of Qwen2.5 weights
(Apache-2.0). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
