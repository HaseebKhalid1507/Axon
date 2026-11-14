# Environment variables

The `AXON_*` variables and what each changes. The command-line flags are in [usage.md](usage.md), and `axon serve`'s
in [serving.md](serving.md).

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
| `AXON_GPU_EXPERTS=N\|auto` | same as `--gpu-experts` (gen, chat, serve; default 0) |
| `AXON_REPACK=0` | CPU: run every matrix in place from the GGUF's blocks instead of repacking Q4_K/Q6_K/Q5_0/Q8_0 for the faster kernels (for comparisons) |
| `AXON_DRAFT=K` | same as `--draft K` (gen, chat, serve): prompt-lookup speculative decoding, up to K draft tokens per step (default 0 = off; try 7). Lossless: same output as without it (greedy, or sampled with the same seed) |
| `AXON_DRAFT_MODEL=PATH` | same as `--draft-model PATH` (gen, chat, serve) |
| `AXON_MODELS_DIR=DIR` | `axon pull`'s models directory, where `gen`/`chat`/`serve`/`info` also find pulled models by name (default `$XDG_DATA_HOME/axon/models`, else `~/.local/share/axon/models`) |
| `HF_TOKEN=TOKEN` | `axon pull`: a Hugging Face access token, for gated or private repos (passed to curl on its stdin; needs curl 7.58+, which doesn't send it on to another host on a redirect) |
| `AXON_DEVICE=cpu\|cuda` | same as `--device` (default `cpu`) |
| `AXON_GPU_LAYERS=N` | same as `--gpu-layers N` (with `--device cuda`) |
| `AXON_GPUS=L` | same as `--gpus L` (gen, chat, serve; with `--device cuda`) |
| `AXON_PP_CUTS=K[,K2..]` | `--gpus` pipeline: the first layer of each GPU after the first (default: even if it fits, else by free memory) |
| `AXON_P2P=1` | `--gpus` pipeline: try direct peer-to-peer copies between the GPUs. UNTESTED (no P2P-capable box has run it); default: copies staged through host memory by the driver (tested) |
| `AXON_KV=f16\|f32\|q8` | KV cache type (default f16; `f32` CPU only, for A/B; `q8` cuda only, as `--kv q8`) |
| `AXON_STREAM=0` | split model: don't stream the CPU layers' weights through the GPU for prefill batches (default on: they are copied over once per prompt chunk, and those layers then get the GPU's numerics) |
| `AXON_STREAM_MIN=N` | split model: smallest prefill batch streamed (default 32). Streaming pins the CPU layers' KV cache in RAM (all of it, when the cache is made; not if the RAM available can't hold it) |
| `AXON_STREAM_DECODE=1` | split model, experiment: stream one-token decode steps too (slower; refused with `--draft`) |
| `AXON_CUDA_GRAPHS=0` | cuda: launch each token's kernels one by one instead of replaying a captured CUDA graph (same results, slower) |
| `AXON_MOE_PROFILE=1` | cuda, mixture-of-experts model: every 64 decode steps, print per layer the GPU segment (attention + router; synced for the profile, so a little slower), the copy down, the CPU experts and the copy up |
| `AXON_CUDA_PDL=0` | cuda: no programmatic dependent launch (used only on sm_90+ with a CUDA 12.3+ driver) |
| `AXON_ATT_MQ=0` | cuda, tensor cores: a one-sequence batched pass of few queries (a prompt chunk, a draft verify) runs attention per query (`attn_blk_b`, ≤ 16 queries at head size 128, ≤ 48 at 64) and above that the flash kernel, instead of `attn_blk_mq` (16 queries per K/V tile read). Same output bits |
| `AXON_ATT_MQ_MAX=N` | cuda, tensor cores: most queries of a one-sequence pass that `attn_blk_mq` takes before the flash kernel (default 32). Parsed as an integer; anything else, the default. Small values send passes of more than N queries to the flash kernel (0: all of them). Large values turn the flash kernel off for one-sequence passes of up to N tokens (and prompts are cut into the plain passes): slower past the crossover, same bits |
| `AXON_WAVE_SPLIT=0` | cuda, `gen`/`logits`: split a prompt of several batched passes into equal passes (the old split) instead of sizing them to fill this GPU's GEMM waves (e.g. pp2100 on an RTX 4090: 1088 + 1012, not 1050 + 1050, +3 % prompt eval on 7B/8B; same bits). `AXON_PASS_TRACE=1` prints the pass sizes |
| `AXON_GATE_UP=0` | cuda: run the FFN's gate and up products (both Q4_K) as two GEMMs instead of one fused launch in batched passes (same bits) |
| `AXON_KERNEL_PACK=0\|only` | cuda: `0`: don't use the precompiled kernels a release binary carries, compile with NVRTC as a source build does (the kill switch); `only`: never NVRTC, fail when the pack has no code for the GPU (a driver-only machine on purpose). Default: the pack when it has the GPU's code and the driver supports its CUDA major, else NVRTC ([models.md](models.md#nvidia-gpus)) |
| `AXON_KERNEL_PACK_FILE=PATH` | cuda: use the kernel pack in PATH (`axon kernels build`) instead of the one built into the binary. A file that can't be read or isn't a valid pack is one warning, and NVRTC compiles the kernels (with `AXON_KERNEL_PACK=only`, an error). A pack is code that runs on the GPU: point this only at packs you built |
| `AXON_HOPPER=0` | cuda, H100/H200 (compute capability 9.0): compile the kernels for plain sm_90 instead of sm_90a, which turns the wgmma/TMA kernels off (same bits) |
| `AXON_GEMM_TILE=64` | cuda, sm_80+: keep every batched Q4_K / Q5_K product (and the fused gate+up) on the 64 × 64 `gemm2s` tile. Default (unset or any other value): the 128 × 64 `gemm2t` tile where its waves cost less (`tile_t`: the big products at > 64 tokens; not batches of ≤ 64 tokens, nor, e.g., a 512-row k / v on a 128-SM GPU). Same bits either way |
| `AXON_GEMMH=0\|1` | cuda, sm_90a: `0` keeps every batched Q4_K product on the mma.sync GEMMs; `1` sends every one of them to the wgmma kernel. Default: the fused gate+up and the large Q4_K products (>= 3584 rows, at least one full wave of blocks) use wgmma (same bits either way) |
| `AXON_Q6K8=0\|1` | cuda: `0` never makes the int8 copy of Q6_K matrices that speeds up prefill (sm_80+); `1` makes it whenever it fits beside the cache. Default: only if 768 MiB stay free beside it and the cache, and never on a split (same bits either way) |
| `AXON_FP8_NATIVE=0` | cuda, FP8 checkpoint, `--numerics fast`: run the Q8_0 form converted at load, as exact mode does, instead of fp8.cu's kernels (the fingerprint says `weights=fp8-q8`). A `chat --session` file records whether fp8.cu ran (the KV identity has `weights: fp8-native`, #198), so a session saved with it set and restored without it, or the other way round, prefills its saved tokens again instead of loading their KV bytes. With it set, `gpu_fp8_fast_vs_exact_sanity`, `graded_in_two_passes_is_graded_at_once` and `gpu_fp8_graded_vs_fp8_ref_1000` fail (they have no native FP8 to grade), as does the by-hand `gpu_fp8_load_diag`; the FP8 kernel tests don't read it |
| `AXON_ADMIT=fifo\|sjf` | `serve`: same as `--admit-order` (any case; default sjf); any other value refuses to start |
| `AXON_STEP_COST=legacy` | `serve`: the step budget before the measured cost model (`--step-ms` less the last decode step); with `AXON_ADMIT=fifo` and `--step-ms 60`, the scheduling before #131, decision for decision |
| `AXON_API_KEY=KEY` | `serve`: same as `--api-key KEY`, kept out of `ps`; `--api-key` wins ([serving.md](serving.md#operations-auth-metrics-slots)) |
| `AXON_SHARE_KV=1` | `serve`, the GPU's paged pool: requests share the KV pages of a common prompt prefix, same bits (opt-in, off by default; [serving.md](serving.md#operations-auth-metrics-slots)) |
| `AXON_TOOL_STREAM=0` | `serve` (exactly `0`): tool calls go out whole once decided (two chunks; on `/v1/responses` at the reply's end), and a `<tool_call>` block that isn't valid JSON is returned as content, as before #260, streamed or not. Default: a call streams as it is generated, its id and name first, then its arguments, on chat completions and Responses; and a call whose arguments go bad or are cut off is still a call (invalid `arguments`), on streamed and whole (`"stream": false`) replies alike ([serving.md](serving.md#tool-calling)) |
| `AXON_PAGED_KV=0` | `serve`: the old pool of slots each reserved at the full context, instead of 64-position pages shared by every slot |
| `AXON_KV_PAGES=N` | `serve`: size the paged pool to N pages (at most what fits), for testing a server under memory pressure |
| `AXON_KV_RAM_ALLOC=start` | `serve --cache-ram` on the GPU: pin the whole tier before the server answers, instead of in 64 MiB pieces while it runs |
| `AXON_SERVE_TRACE=FILE` | `serve`: append a JSON line per batched step to FILE (its prompt feeds, its decode pass, its wall time); off by default, no output changes |
| `AXON_SKIP_MODEL_TESTS=1` | tests: skip the ones that need the real model |
| `AXON_SKIP_GPU_TESTS=1` | tests: skip the ones that need a GPU (CI sets it; [CONTRIBUTING.md](../CONTRIBUTING.md)) |
| `AXON_SKIP_FP8_MODEL_TESTS=1` | tests: skip the ones that need the FP8 checkpoint (`models/hf/Qwen2.5-0.5B-Instruct-FP8-dynamic`); without it, or `AXON_SKIP_MODEL_TESTS`, a missing checkpoint fails them on a host with a GPU |
