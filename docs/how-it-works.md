# How Axon works

What Axon is built from, the contract every optimized path keeps, and the order to read the source in. The
file-by-file table is the [source map](START-HERE.md#source-map); the bit-identity contract in full is
[determinism.md](determinism.md).

## Built from scratch


No ggml, no BLAS, no binding crates, no CUDA toolkit at build time: a GGUF reader,
quantized integer dot-product kernels (scalar, AVX2, AVX-VNNI), a spinning worker
pool and a transformer forward pass on the CPU; on the GPU, CUDA C++ kernels that
NVRTC compiles at run time, launched through the driver API (both libraries loaded
with dlopen). About 121k lines of Rust and 9.9k of CUDA, tests included (904 `#[test]`s in `src/`; counted at `3014eeb`).

## Bit-identical paths

Every optimized path is **bit-identical** to the plain reference path on its
device: batched prefill produces exactly the logits that feeding one token at a
time does (on the CPU, and on the GPU against the GPU's own one-token step), and
each SIMD kernel exactly what its scalar twin does. The GPU is deterministic and
close to the CPU but not bit-identical to it (its float sums are split and ordered
differently; `src/cuda/qwen2.rs`). The tests check this, and check against three
independent oracles (gguf-py's dequantizer, Hugging Face's tokenizer, Ollama's
greedy output).

## Reading the source

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
4. **`src/cli.rs`, `run_lm`**: encode the prompt, prefill in batches, then
   `generate` (`src/generate.rs`): `forward`, pick the next token (argmax, or `sample.rs`), repeat.
   (`src/lib.rs` is the crate root and the public API; `src/main.rs` only calls `axon::cli::main`.)

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
