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

Crates in `Cargo.lock` are under MIT, Apache-2.0, Zlib, Unlicense, BSL-1.0 or
Unicode-3.0 (`unicode-ident`) licenses; none is copyleft. Their notices ship
with their sources on crates.io; a binary distribution should include them
(e.g. generated with `cargo about`).
