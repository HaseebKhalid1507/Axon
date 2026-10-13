# Axon

CPU-only LLM inference for Qwen2 GGUF models, written from scratch in Rust.

No ggml, no BLAS, no bindings: a GGUF reader, quantized integer dot-product
kernels (scalar, AVX2, AVX-VNNI), a spinning worker pool and a transformer
forward pass, about 7k lines. Every optimized path is **bit-identical** to the
plain reference path: batched prefill produces exactly the logits that feeding
one token at a time does, and each SIMD kernel exactly what its scalar twin
does. The tests check this, and check against three independent oracles
(gguf-py's dequantizer, Hugging Face's tokenizer, Ollama's greedy output).

## Status

Works today, honestly scoped:

- Architecture: `qwen2`. Tested with Qwen2.5-0.5B-Instruct; also runs the 1.5B, 3B and 7B
  Instruct Q4_K_M builds. Other architectures are refused with an error.
- Tensor types: Q4_K, Q5_0, Q6_K, Q8_0, F16, F32 (what Qwen's Q4_K_M files use). Others load for `info` but can't run.
- Sampling: greedy only. No chat template yet: `gen` continues raw text.
- CPU: any x86-64 (others build too, scalar only). The default build is baseline x86-64 and
  picks its AVX2 / AVX-VNNI code at runtime, with the same output bits as a `-C target-cpu`
  build (see `src/mv.rs`). Attention and SwiGLU use fused multiply-adds when the CPU has
  FMA and a separate multiply and add when it doesn't (`src/vmath.rs`), so logits depend on
  that one CPU property: deterministic per CPU class (FMA or not), identical across paths
  (`AXON_SIMD`), batch sizes and thread counts within it.
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
```

`gen` streams the text to stdout and prints prompt/decode speed to stderr.

## Usage

```
axon <model.gguf>                         browse the model in a terminal UI
axon info <model.gguf>                    print a summary and exit
axon gen <model.gguf> "<prompt>" [-n N]   greedy generation (default 32 tokens)
axon logits <model.gguf> "<prompt>" [-k K]  top-K next-token logits (default 10)
axon --help | --version
```

Special-token text in a prompt (`<|im_end|>` and the like) is encoded as plain
text, never as the control token.

### Environment

| variable | effect |
|---|---|
| `AXON_THREADS=N` | worker threads (default: cores minus a quarter) |
| `AXON_SIMD=scalar\|avx2\|vnni` | force a kernel set (default: the best the CPU has) |
| `AXON_POOL_ADAPT=0` | keep every worker in every job (no adaptive drop-out under load) |
| `AXON_POOL_DEBUG=1` | `gen`: print thread-pool stats at the end |
| `AXON_MMAP=0\|1` | `0`: copy the model into huge-page memory; `1`: map the file. Default: copy, unless the file is over 60% of available memory |
| `AXON_PREFETCH=N` | decode: software prefetch distance in bytes (default 8192, 0 = off, max 1 MiB) |
| `AXON_DECODE_ROWS=1\|2\|4` | decode: rows per kernel call (default 4; K-quants use at most 2) |
| `AXON_DRAFT=K` | gen: prompt-lookup speculative decoding, up to K draft tokens per step (default 0 = off; try 7). Lossless: same output as plain greedy |
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
any time; `viz.rs` is the TUI.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): test model, oracles, checks to run.

## License

MIT, see [LICENSE](LICENSE). Parts of the kernels and the tokenizer follow
llama.cpp/ggml (MIT); test fixtures contain rows of Qwen2.5 weights
(Apache-2.0). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
