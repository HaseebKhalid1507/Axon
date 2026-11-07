# Axon as a library

`cargo build` builds the `axon` library and the `axon` command line on top of it. The library's API is small:
`axon::load` a GGUF, `Backend::new(&model, BackendOptions::default().device(..).max_ctx(..))` to place it on the CPU
or a GPU, the model's own `Tokenizer`, the `Lm` trait
(caches, prefill, decode steps), `axon::generate` (the greedy loop behind `axon gen`, streaming each token to a
callback; sampling stays the CLI's in 0.1) and `axon::serve` (`axon serve`, in-process).
[examples/embed.rs](../examples/embed.rs) loads a model and decodes N tokens greedily, the same tokens `axon gen` prints.
The API takes device, KV type and GPU layers as arguments and never reads `AXON_DEVICE` / `AXON_KV` /
`AXON_GPU_LAYERS`; the example reads them as `axon gen` does (`scripts/embed_check.sh` checks the two agree):

```bash
cargo run --release --example embed -- models/qwen2.5-0.5b-q4_k_m.gguf "The capital of France is" 32
```

The API is 0.x and will change, and it isn't hardened for embedders yet
([#118](https://github.com/HaseebKhalid1507/Axon/issues/118)): library calls print notes to stderr; engine
tuning variables are read inside library calls, some once at first use (`AXON_SIMD`, `AXON_THREADS`, the decode
config) and others again at every `load` / `Backend::new` (`AXON_MMAP`, `AXON_REPACK`, `AXON_Q6K8`,
`AXON_GATE_UP`, `AXON_CUDA_GRAPHS`, `AXON_PP_CUTS`), so changing the environment between loads changes behaviour;
`AXON_GPUS` is the only way to pick the GPU(s) in 0.1 (read at each `Backend::new`); a CUDA `Backend::new` leaks a
little; and `axon::serve` exits the process on a
usage error (2) or an engine failure (70). A bad `AXON_SIMD`, `AXON_DECODE_ROWS` or `AXON_PREFETCH` is an error
from `load` / `Backend::new` (and from `axon gen` / `logits` / `chat` / `serve`, exit 2; `axon info` runs no kernels
and doesn't check), not a panic. `Backend` and `BackendCache` are opaque, `Lm` is sealed, and `Device`,
`KvType`, `GenStats` and `BackendOptions` are non-exhaustive, so new backends, devices, stats and options don't
break 0.1.x callers. The crate's package name on crates.io isn't settled (`axon` is taken);
it isn't published.
