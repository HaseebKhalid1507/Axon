# Axon

CPU-only LLM inference for Qwen2, Qwen3 and Llama 3 GGUF models, written from scratch in Rust.

No ggml, no BLAS, no bindings: a GGUF reader, quantized integer dot-product
kernels (scalar, AVX2, AVX-VNNI), a spinning worker pool and a transformer
forward pass, about 7k lines. Every optimized path is **bit-identical** to the
plain reference path: batched prefill produces exactly the logits that feeding
one token at a time does, and each SIMD kernel exactly what its scalar twin
does. The tests check this, and check against three independent oracles
(gguf-py's dequantizer, Hugging Face's tokenizer, Ollama's greedy output).

## Status

Works today, honestly scoped:

- Architectures: `qwen2`, `qwen3` (dense) and `llama` (Llama 3.x). Tested with Qwen2.5-0.5B-Instruct; also runs the
  1.5B, 3B and 7B Instruct Q4_K_M builds, Qwen3 0.6B (Q8_0), 1.7B, 4B and 8B (Q4_K_M), and
  Llama 3.2 1B / 3B and Llama 3.1 8B Instruct (Q4_K_M; Llama 3's RoPE frequency factors, the
  llama-bpe tokenizer with its BOS token).
  Other architectures are refused with an error.
- Tensor types: Q4_K, Q5_0, Q6_K, Q8_0, F16, F32 (what Qwen's Q4_K_M files use). Others load for `info` but can't run.
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
```

`gen` streams the text to stdout and prints prompt/decode speed to stderr.
`chat` is a conversation: type a message, get a reply, keep going.

## Usage

```
axon <model.gguf>                         same as axon info
axon info <model.gguf>                    print a summary and exit
axon gen <model.gguf> "<prompt>" [-n N]   generation (default 32 tokens; greedy unless --temp)
axon logits <model.gguf> "<prompt>" [-k K]  top-K next-token logits (default 10)
axon chat <model.gguf> [--system TEXT] [--no-think] [--ctx N] [-n N] [--draft K]
                                          talk to an Instruct model
sampling (gen, chat): --temp T --top-k K --top-p P --min-p P
          --repeat-penalty P --repeat-last-n N --seed S   (axon --help)
axon serve <model.gguf> [--host H] [--port P] [--ctx N] [--device D] [--kv T] [--draft K]
                                          OpenAI-compatible HTTP server
axon --help | --version
```

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
conversation outgrows `--ctx` (default 8192), the oldest exchanges are forgotten.

Llama 3 uses the official 3.1/3.2 template (no tools): a system block that always
opens with the template's "Cutting Knowledge Date: December 2023 / Today Date: 26 Jul
2024" lines (its default date, so a conversation doesn't depend on the day) and then
`--system`'s text if any, headers around each message, `<|eot_id|>` ending each turn.

Qwen3 follows its own template: no system message unless `--system` gives one, and
the model reasons first (`<think>...</think>`, shown as it streams; Ollama's default
too). `--no-think` opens each reply with an empty reasoning block instead (the
template's `enable_thinking=false`). Earlier replies stay in the conversation
without their reasoning, as the template renders them. Qwen recommends sampling
for thinking mode; axon is greedy, which can loop on long reasoning.

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
- Chat prompts are framed as `axon chat` frames them (ChatML; Qwen2.5's default system
  message when none is given; Qwen3 earlier replies without their reasoning). Qwen3 thinks
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
| `AXON_DRAFT=K` | gen: prompt-lookup speculative decoding, up to K draft tokens per step (default 0 = off; try 7). Lossless: same output as without it (greedy, or sampled with the same seed) |
| `AXON_SKIP_MODEL_TESTS=1` | tests: skip the ones that need the real model |

## Numbers

From [BENCH.md](BENCH.md) (main @ 74fe85e, Qwen2.5-0.5B-Instruct Q4_K_M,
Intel Core Ultra 9 275HX, 18 threads, quiet machine, tok/s, median of 3):

| engine | pp560 | pp2100 | tg128 | tg128@1 |
|---|---|---|---|---|
| axon 74fe85e | 1294.7 | 1068.8 | 163.8 | 29.9 |
| ollama 0.35.0 | 1027.3 | 961.2 | 161.6 | 34.4 |

`pp` is prompt evaluation, `tg128` decode over 128 tokens, `@1` one thread.
Ranges, the loaded-machine numbers and how to reproduce them (`scripts/bench.py`)
are in BENCH.md. One machine, one model: treat them as relative.

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
4. **`src/main.rs`, `run_lm`**: encode the prompt, prefill in batches, then the
   greedy loop: `forward`, argmax, repeat.

`pool.rs` (the worker pool) and `tokenizer.rs` (byte-level BPE) can be read
any time.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): test model, oracles, checks to run.

## License

MIT, see [LICENSE](LICENSE). Parts of the kernels and the tokenizer follow
llama.cpp/ggml (MIT); test fixtures contain rows of Qwen2.5 weights
(Apache-2.0). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
