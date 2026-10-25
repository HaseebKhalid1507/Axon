# One MoE layer's experts on the GPU, CPU-exact (PR #122)

- Date: 2026-10-24. Branch: feat/moe-resident-b (stacked on PR A #120, merged as 6e3b19c, which is merged in
  at 45888a5). Status: open; every gate of record passed (2026-10-25).
- Pick / design: Fable 10-19 §3 #3; spec `r16-pick4-moe-resident.md` §6 PR B. Also #121 items 1–4 (shady's
  follow-ups on #120).

Labels: **MEASURED** (job cited), **CODE**, **ESTIMATED**.

## What changed

Test-only: no model path launches the new kernels yet (that's PR C). Bits of every existing kernel are unchanged
(SASS below).

- **`src/cuda/moe_res.cu` (new, in `kernels_src!`'s base list):**
  - `moe_sort`: groups the nb·k (token, slot) pairs by expert, in ascending pair order within each expert, the
    same order as the CPU's counting sort. It cuts each expert's run into 64-token tiles. One block, warp ballots.
  - `moe_gather`: copies each pair's quantized token into sorted order.
  - `moe_gemm_tw_{q4_k,q5_0,q8_0,q6_k}`: grouped twin GEMMs over the tile list (`blockIdx.x` = tile). They're
    PR A's `gemm2_t<…, TW = true>`. `gemm2_t` gets a defaulted `zt` (token-tile override); every existing
    instantiation keeps `zt = -1`, so their SASS doesn't change.
  - `swiglu_quant_tw`: `swiglu_twin` then `quant_warp`, the CPU's `glu` then `QuantizedAct::for_types`.
  - `moe_sum`: the CPU's `combine` in rank order (`y = w0·d0`, then `y + wi·di`, a multiply then an add).
  - `moe_gemv_tw`: the decode expert GEMV over the k slots, in `gemv_tw`'s numerics. It runs on any arch.
- **`src/cuda/qwen2/experts.rs` (`#[cfg(test)]`):** `CudaExperts::upload(l)`, then `layer` (nb 1: GEMV; more:
  grouped), `batched` (with GPU event timing) and `layer_gemv` (token by token, the nb-invariance reference).
- **#121 items 1–4:**
  1. Host refusal `twin_gemm_ok(cc)`: twin GEMMs need sm_80+. Below that, gemm2's body is `__trap()`, which would
     poison the context. Unit test `twin_refusals`. On sm < 8.0, `gpu_twin_gemm2s_matches_cpu_r8` asserts the
     refusal and prints `SKIPPED (refused, as designed)`; before, it returned silently.
  2. Type contract: `twin_type(qt)` (Q4_K / Q5_0 / Q8_0 / Q6_K only, unit-tested against the other types) guards
     the runtime-typed `gemv_tw` path. `gemm2_t<…, TW = true>` `static_assert`s the twinned types (Q2_K and the MIN1
     types would have taken the plain epilogue silently).
  3. Epilogues: `gpu_twin_epilogues_match_cpu` checks bias, mode 1, and both together, on `gemv_tw` and every
     `gemm2s_*_tw`. All are bit-equal to `init + (row sum + bias)`.
  4. `fused`: `CudaExperts` reads `vmath::fused()` once at upload, and every `swiglu_twin` launch of the layer
     passes that field. The tests assert it equals the host's.
  - Item 5 (`--gpu-experts` refuses or falls back under `AXON_REPACK=0` or a mismatched F) belongs to PR C, where
    the flag is added. It stays open in #121.
- **`deepseek2::build_only` + `Qwen2::new_experts_of` (test):** repack only the layers under test, so the
  Moonlight test needs ~1.2 GB of r8 arena, not 11 GB.

## Why

PR B is the expert layer itself, checked bit for bit against the CPU before any placement or plumbing (PR C) can
make the model use it.

## How it was verified

**Gates of record: 265f604** (code; review round 1). b2f714e is a test-only commit on top (the bias-refusal test), gated by its own two jobs. Later commits are docs.

- [x] `rustfmt --edition 2024 --check src/lib.rs`
- [x] clippy-driver `-D warnings`, lib and `--test` (265f604 and b2f714e, local)
- [x] CI `check` (head)

| check | where (device / arch) | job | passed / failed / ignored |
|---|---|---|---|
| experts + twin tests, **release then debug-assertion** build: Moonlight layers 1 / 3 / 26, qwen3moe pruned 30B-A3B layers 0–2 (layer 2 **Q6_K down**) + 4x0.6B layers 0 (Q6_K) / 3, fixture; nb 1/6/48/510/1024, nb 1 forced through the grouped path, nb invariance | bella sm_120 (`AXON_TEST_DS2=1`) | **1024-174754-77d4** | PASS 9 / 0 / 1, both builds (bench ignored) |
| same, release (qwen3moe + fixture; Moonlight not on the node) | avante sm_89 | **1024-174755-c99f** | PASS 9 / 0 / 1 |
| same, release (decode GEMV; grouped path refused; gemm2s twin test `SKIPPED (refused, as designed)`) | jade sm_61 | **1024-174754-565e** | PASS 9 / 0 / 1 |
| fast tier, release, shard 1/2 | bella / avante | **1024-174755-2580** | PASS 278 / 0 / 30 |
| fast tier, release, shard 2/2 | bella / avante | **1024-174755-a30d** | PASS 358 / 0 / 18 |
| fast tier, debug assertions + overflow checks, CPU queue (unsharded) | bella / avante | **1024-174755-34f3** | PASS 652 / 0 / 48 |
| `expert_biases_refused_before_upload` + `twin_refusals`, b2f714e, with GPU (upload itself refuses) | avante sm_89 | **1025-012258-d4c6** | PASS 2 / 0 / 0 |
| same, CPU queue (the check before any upload; no GPU) | bella CPU | **1025-013756-5e5a** | PASS 2 / 0 / 0 |
| §5 bench | bella sm_120, exclusive | **1025-022441-83db** | PASS (≤ 5 ms at 510: 2.283 / 2.126 ms), CLEAN |
| `AXON_SIMD=scalar` | n/a: no CPU kernel changed | | |

Round 0 (9d85b1f) gates, superseded: 4105, b997, 8d37, 5c02, 4b95, 6c2b, 8e2c PASS. 06d5 was frozen by the bella hold and never finished; the round-1 suites replace it.

SASS of the existing kernels vs main 6e3b19c (nvcc 13 offline with the NVRTC opts; cuobjdump -sass per function, addresses stripped):

| target | existing identical | new |
|---|---|---|
| sm_120 plain / ext | 164 / 164, both | 9 |
| sm_89 plain / ext | 164 / 164, both | 9 |
| sm_75 plain / ext (jade's sm_61 isn't in this nvcc) | 164 / 164, both | 9 |
| sm_120 `KERNELS_OSS` ext | 168 / 168 | 9 |

New: `moe_sort`, `moe_gather`, `moe_gemm_tw_{q4_k,q5_0,q8_0,q6_k}`, `swiglu_quant_tw`, `moe_sum`, `moe_gemv_tw`.

## Results

**Gate of record: job 1025-022441-83db** (`--timing bella`, binary **265f604**, FOREIGN_GPU max 0 % / mean 0 %, 1 sample, 7.7 s: **CLEAN**). `cuda_experts_moonlight_bench`, Moonlight-16B-A3B Q4_K_M, one layer's routed experts (64 experts, k 6, dim 2048, ffn 1408), random hidden states routed by the CPU's router. GPU: CUDA events around the grouped kernels, median of 5 × 10 reps after 3 warm-ups. CPU: `moe_layer_batch` wall time, median of 3, default threads, same host (MEASURED).

| layer (down type) | nb | GPU ms | CPU ms | GPU / CPU speedup |
|---|---|---|---|---|
| 1 (Q8_0 down) | 48 | 1.612 | 6.16 | 3.8× |
| 1 (Q8_0 down) | **510** | **2.283** | 22.31 | 9.8× |
| 1 (Q8_0 down) | 1024 | 3.672 | 46.07 | 12.5× |
| 3 (Q5_0 down) | 48 | 1.350 | 4.84 | 3.6× |
| 3 (Q5_0 down) | **510** | **2.126** | 22.91 | 10.8× |
| 3 (Q5_0 down) | 1024 | 3.456 | 44.23 | 12.8× |

**Spec gate (≤ 5 ms per layer at nb 510): PASS**, 2.283 / 2.126 ms. It is per-layer kernel time, not a forward pass: router, attention and graph overhead come with the model plumbing (PR C).

History, for honesty:
- **1025-004736-1325** (r2, 265f604) ran but is TAINTED (FOREIGN_GPU mean 10 %: zen-bin 7 %, Xorg 3 %). Its numbers are consistent with the clean run: 510 → 2.232 / 2.091 ms, speedups 3.6–12.2×. Not the gate.
- **1025-021642-5e98** (r3) was submitted without `AXON_TEST_DS2=1`, so the Moonlight test skipped (0.3 s, no BENCH lines). It does **not** count.
- The earlier 1024-164700-b533 / 1024-171406-2860 never ran (bella held; superseded).

## What didn't work / notes

- The first fixture run (jade, 1020-164016-5dbf, 10-20) failed in the test itself, not in the numerics. The CPU
  repack of the in-memory fixture released its expert bytes before the GPU upload read them (gguf.rs: "released
  after repacking"). The fix is to upload first, then build the CPU experts. Its Moonlight "ok" was a skip
  (AXON_TEST_DS2 unset).
- 00205a2 ("WIP, not gated") describes the 10-20 state. It's superseded by the gates above; history is left as is.
- Moonlight runs only on bella: the remote nodes have no `models/` link for this worktree. avante and jade run the
  fixture.
- **Track kill rule (Fable):** if resident decode is < 1.5× the CPU-expert decode on bella, the track stops after
  this PR. It's decided in PR C's first bench: tg128 on bella, `--gpu-experts auto` vs 0.
