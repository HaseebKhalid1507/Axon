# Third-party notices

Axon is MIT-licensed (see `LICENSE`). It contains or is derived from the
following third-party material.

## llama.cpp / ggml (MIT)

`src/quant.rs` follows ggml's block layouts (`ggml/src/ggml-common.h`), its
dequantization and dot-product math (`ggml/src/ggml-quants.c`,
`ggml/src/ggml-cpu/arch/x86/quants.c`). `src/tokenizer.rs` follows llama.cpp's
qwen2 pre-tokenizer regex and special-token splitting (`src/llama-vocab.cpp`).
Source: https://github.com/ggml-org/llama.cpp

```
MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen2.5 weights in test fixtures (Apache-2.0)

`tests/fixtures/golden/*.raw` (and the `.f32`/`.json` made from them) are
single rows of quantized weights taken from Qwen/Qwen2.5-0.5B-Instruct-GGUF,
Copyright (c) Alibaba Cloud, licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0 (model card:
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF). They are unmodified
byte ranges of that file. Only use fixtures from models whose license allows
redistribution.

`tests/fixtures/gen/*.json` (Ollama greedy outputs) and
`tests/fixtures/tokens.json` (Hugging Face tokenizer output) are model outputs
and token ids recorded by `scripts/oracle/make_fixtures.py`.

## Rust dependencies

Their notices ship with their sources on crates.io. A release tarball includes the license files of every crate
linked into its binary in `THIRD_PARTY_LICENSES.txt` (`scripts/release/rust_licenses.py`). That script runs on
every pull request (CI) and fails if a linked crate's license expression isn't satisfied by this list: for
`A OR B` one of them must be on it, for `A AND B` both. None is copyleft; keep it that way.

Allowed licenses (`scripts/release/rust_licenses.py`): MIT, Apache-2.0, Apache-2.0 WITH LLVM-exception, Unlicense, Zlib, BSL-1.0, Unicode-3.0.

## Rust standard library (MIT OR Apache-2.0)

Every Rust binary links the Rust standard library (std, core, alloc and the crates it builds in, such as
compiler_builtins, hashbrown, addr2line, gimli, miniz_oxide and rustc-demangle) statically. Its copyright notices:
https://github.com/rust-lang/rust/blob/master/COPYRIGHT. A release tarball ships the building toolchain's
`COPYRIGHT-library.html` as `RUST_STD_COPYRIGHT.html`.

## GPU kernels in release binaries (NVIDIA CUDA Toolkit)

A release binary carries its GPU kernels precompiled (the kernel pack,
`docs/models.md`, "NVIDIA GPUs"): Axon's own CUDA sources (`src/cuda/*.cu`)
compiled by NVIDIA's NVRTC and compressed with `fatbinary`, both from the CUDA
Toolkit, as any CUDA application's compiled kernels are; device code the
compiler builds in from its CUDA headers and libdevice is NVIDIA's, under the
CUDA Toolkit EULA (https://docs.nvidia.com/cuda/eula/). No NVIDIA library is
shipped: `--device cuda` uses the NVIDIA driver installed on the machine.
