# Changelog

All notable changes to Axon. Versions follow [SemVer](https://semver.org/); before 1.0 the library API and the CLI
can change in any minor release. Each entry names its pull requests (#N on GitHub); the engineering log (`docs/log/`) has the
measurements behind them.

## [0.1.0] - unreleased (draft; not tagged)

The first version: everything merged to `main` from the start of the project (2026-10-10) through PR #113
(main `aecc49d`, 2026-10-20), 103 pull requests, plus the library split below. Release notes:
[docs/release-0.1.md](docs/release-0.1.md). The tag and the date are the maintainer's call.

### Library
- The crate is now a library (`axon`) plus the `axon` binary on top of it. `src/lib.rs` is the crate root and the
  public API: `load`, `Model`, `Backend`, `BackendCache`, `Device`, `KvType`, `Tokenizer`, `Utf8Stream`, the `Lm`
  trait, `generate`, `GenOpts` (`GenOpts::greedy`; sampling stays crate-internal in 0.1), `GenStats`, and `serve`.
  The command line moved from `src/main.rs` to `src/cli.rs` (a new file: git doesn't track it as a rename) and the
  decode loop and its tests to `src/generate.rs`; `examples/embed.rs` shows the API. No change for the CLI: same
  greedy bits, same output, same tests (14 renamed: `tests::*` → `cli::tests::*` or `generate::tests::*`).
- The KV type is an argument: `Backend::new` (its `BackendOptions::kv`) reads no `AXON_KV` (the CLI and `examples/embed.rs` parse it
  with `KvType::parse` and pass it), `KvType::F32` takes effect on the CPU (it used to be silently f16 through the
  library), and a type that conflicts with the process's earlier f16/f32 choice is an error.
- `Model` is read-only outside the crate: its fields and the methods that hand out or release weight bytes are
  crate-private (a safe caller could zero weights a live `Backend` borrows); accessors `arch`, `meta_u32`,
  `n_layers`, `embed_dim`, `n_heads`, `n_kv_heads`, `ctx_len`, `total_params`, `total_bytes`. No `memmap2` type in
  the API.
- `generate` refuses drafting (`max_draft > 0`) on a backend whose decode steps stream (`Lm::decode_streams`:
  `AXON_STREAM_DECODE=1` on a CPU/GPU split), where drafting wouldn't be lossless; the CLI and `serve` already
  refused the pair.
- `Backend` and `BackendCache` are opaque (#118 A): no variants to match and no fields, so the model types inside
  (CPU, GPU, split, pipeline, mixture-of-experts) and their geometry can't be reached or written from outside the
  crate, and a new kind is not a breaking change. `Lm` is sealed (implemented by the crate's models only).
  `Device`, `KvType` and `GenStats` are `#[non_exhaustive]` (a match outside the crate needs a `_` arm;
  `GenStats::default` stays).
- `Backend::new(&model, BackendOptions)`: device, context, GPU layers, KV type and fit are a `#[non_exhaustive]`
  options struct with `Default` and chainable setters (`BackendOptions::default().device(Device::Cuda).max_ctx(8192)`),
  not six positional arguments, so new options (#118 item 2) aren't breaking. `Lm::forward_cands` / `verify_cands`
  are `#[doc(hidden)]` until the sampler is exported (#118 item 8).
- A bad `AXON_SIMD` (unknown, or a kernel set the CPU lacks), `AXON_DECODE_ROWS` or `AXON_PREFETCH` is an error
  from `load` and `Backend::new` (the CLI: its usual message, exit 2) instead of a panic inside a later call (#118 B).
- Not hardened for embedders yet (stderr notes, env-only config, a CUDA leak per `Backend::new`, `serve` exits the
  process): #118.

### Models and formats
- Qwen2 / Qwen2.5 GGUFs on the CPU from the first commit; Qwen3 dense on the CPU and CUDA (#18); Llama 3.x
  (#22, which also carried the sm_89 capture fix) and its prompt framing in `serve` (#23); SentencePiece tokenizer
  and Mistral 7B's chat format (#32); Gemma 3 1B / 4B on the CPU and CUDA (#35).
- Mixture of experts: qwen3moe on the CPU (#63), then attention and router on the GPU with CPU experts (#67, #76);
  gpt-oss-20b with MXFP4, o200k and harmony on the CPU (#78), then on the GPU with CPU experts (#98);
  DeepSeek V2/V3 / Kimi K2 / Moonlight (MLA, DeepSeek routing) on the CPU (#79), then GPU MLA with CPU routed
  experts (#89); one router kernel and a `MoeKind` enum (#104).
- Quantization: Q5_K and Q4_0 (#22); Q2_K, Q3_K and IQ4_NL on the CPU and CUDA (#72); clear refusals for i-quants
  and repacked GGUFs (#83).
- Hugging Face safetensors checkpoints (BF16/F16/F32) on the CPU, run as the GGUF F16 of the same weights (#102).
- YaRN context extension, opt-in: `--rope-scaling yarn` (#49).

### CPU
- Interleaved quantized kernels, FMA attention, vectorized exp, f16 KV cache (#2); bench harness, pool under load,
  kernels, attention, decode, hygiene, prompt-lookup drafting (#1).
- Prefill: Q8_0/Q5_0 GEMM fix, fewer pool joins, batch 128 (#20); decode pool participation on hybrid CPUs (#32).
- Startup: map instead of copy when repacked, CUDA init at t = 0, read-ahead, faster upload (#21); pinned upload
  ring, faster tokenizer build and GGUF string parse (#41).
- **Behaviour change (#142):** a dense model whose repacked weights plus the KV cache of its first 8192 tokens (or
  `--ctx`, if shorter) don't fit in the RAM available (host `MemAvailable` or the tightest cgroup limit, #133) is
  refused at load with a message naming the limit (exit 2), instead of being OOM-killed. A longer `--ctx` that
  wouldn't fit whole prints a note. The MoE RAM refusal's figures are now GiB, as the dense one's, and a cgroup
  limit is printed in MiB.

### CUDA
- GPU backend: foundations (#7), decode (#8), batched prefill and speculative verify (#10), an 8-bit KV cache
  `--kv q8` (#11), the canonical block order and tensor-core GEMMs (#12), shared-score batched attention and
  1024-token passes (#15), tensor-core flash attention (#19), r5 prefill epilogues (#24), narrow-row decode
  kernels (#30).
- Placement: CPU/GPU split `--gpu-layers` (#13), streaming the CPU layers' weights for prefill (#16), pipeline
  parallel over several GPUs `--gpus 0,1` (#34), older NVRTC fallback for old GPUs (#39).
- Hopper (sm_90a) wgmma + TMA prefill GEMMs (#33); long-context decode (#50) and prefill (#56) attention;
  batched-decode step cost and gemm2 partial tiles (#81), fused q|k|v and gate|up launches at nb ≤ 16 (#86);
  multi-query block attention (#84); prompt passes sized to the GEMM waves (#92, #97); a 128 × 64 Q4_K/Q5_K GEMM
  tile (#100).
- Paged KV: every cache kernel through a page table (#64, #68), then the pool allocates pages and `serve` plans,
  admits and preempts by pages (#71).

### Generation
- Sampling: temperature, top-k, top-p, min-p, repeat penalty, seeds, on the CPU and CUDA (#17).
- Speculative decoding with a draft model `--draft-model` (#27), and on the CPU/GPU split (#52); greedy output
  unchanged by drafting.
- Identity tuple and `system_fingerprint`: every output says which bits it is (#51).

### `axon chat`
- Terminal chat with the model's template (#4), line editing (#6), exact KV save/restore `--session` (#48, #54),
  forgetting down to half the context when full `--forget-to` (#75).

### `axon serve`
- OpenAI-compatible HTTP server (#22); tool calling (#26); continuous batching of decode rows (#38); the trained
  context by default (#42); chunked prefill `--prefill-chunk` (#45, #55, #57); one cache for the single-request
  path (#60); ops: `--api-key`, `/metrics`, `/slots`, `--max-prompt-tokens`, `--slot-idle-ms` (#61, #62);
  `/v1/responses` and the ollama API (#70); gpt-oss tool calling via harmony (#111).
- Fixes: metrics counted before the response is written (#93); a request leaves in-flight before its client has
  the response (#113).
- **Behaviour change (#141):** an explicit `--parallel N` that doesn't fit in GPU memory at start now exits 2 with
  the numbers (it used to start with fewer slots, one request at a time at worst, said only with logging on). A
  supervisor that restarts on exit will keep exiting until `--parallel`, `--ctx` or `--kv` changes; leave
  `--parallel` out for the automatic sizing, which still falls back (and now says so, also under `--quiet`).

### Hardening, tests and CI
- Fuzz-found fixes for GGUF, tokenizer, serve and CLI (#37); soak harness and model-zoo sweep (#83).
- Test suite: qemu-robust pool test (#14), attention sweep skipped under qemu (#5), PTX arch choice on A100 (#31),
  frozen natural-text fixture (#36), GPU tests serialized on small cards (#40, #43) and on every card (#82), GPU
  OOM test tolerant of other processes (#46), slot-idle test bounds (#87), one CPU reference per input and the
  fast/full tiers (#80), graded logit tests with a repacked reference (#105), Gemma 3 near-tie handling (#107).
- Dropped the model viewer and ratatui: 61 → 27 crates (#9).

### Docs
- README, BENCH.md, START-HERE, GPU benchmarks, methodology, engineering history, rejected experiments, results
  (#25, #28, #47, #88, #91, #103); roadmap (#29, #44, #59, #65, #66, #69, #73, #96); drafting and placement note
  (#58); ILLEGAL_ADDRESS investigation (#53); test speed (#85); contributor contract, code of conduct, security
  policy (#74, #77); determinism-modes ADR (#101).
