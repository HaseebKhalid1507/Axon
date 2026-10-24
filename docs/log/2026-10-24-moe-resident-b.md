# One MoE layer's experts on the GPU, CPU-exact (PR #122)

- Date: 2026-10-24. Branch: feat/moe-resident-b (stacked on PR A #120, merged as 6e3b19c, which is merged in
  at 45888a5). Status: open (gates pending).
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

| check | where (device / arch) | job | passed / failed / ignored |
|---|---|---|---|
| experts + twin tests, release, **Moonlight layers 1 / 3 / 26** + fixture, nb 1/6/48/510/1024 + nb invariance | bella sm_120 (AXON_TEST_DS2=1) | **1024-164652-4105** | PASS 8 / 0 / 1 (the bench is ignored) |
| same, debug assertions + overflow checks | bella sm_120 (AXON_TEST_DS2=1) | **1024-164653-b997** | PASS 8 / 0 / 1 |
| same, release (fixture; Moonlight skips: no models link on the node) | avante sm_89 | **1024-164730-8d37** | PASS 8 / 0 / 1 |
| same, release (fixture via decode GEMV; batched refused; gemm2s twin test prints `SKIPPED (refused, as designed)`) | jade sm_61 | **1024-164653-5c02** | PASS 8 / 0 / 1 |
| fast tier, release, shard 1/2 | avante sm_89 | **1024-165504-4b95** | PASS 270 / 0 / 30 |
| fast tier, release, shard 2/2 | bella sm_120 | **1024-165504-06d5** | pending (bella held) |
| fast tier, debug, CPU queue, shard 1/2 | bella | **1024-164654-6c2b** | PASS 367 / 0 / 18 |
| fast tier, debug, CPU queue, shard 2/2 | bella | **1024-164654-8e2c** | PASS 284 / 0 / 30 |
| `twin_refusals` (host refusals, no GPU) | every run above | | PASS |
| `AXON_SIMD=scalar` | n/a: no CPU kernel changed | | |

SASS of the existing kernels vs main 6e3b19c (nvcc 13 offline with the NVRTC opts; cuobjdump -sass per function, addresses stripped):

| target | existing identical | new |
|---|---|---|
| sm_120 plain / ext | 164 / 164, both | 9 |
| sm_89 plain / ext | 164 / 164, both | 9 |
| sm_75 plain / ext (jade's sm_61 isn't in this nvcc) | 164 / 164, both | 9 |
| sm_120 `KERNELS_OSS` ext | 168 / 168 | 9 |

New: `moe_sort`, `moe_gather`, `moe_gemm_tw_{q4_k,q5_0,q8_0,q6_k}`, `swiglu_quant_tw`, `moe_sum`, `moe_gemv_tw`.

## Results

Pending: bella was held at 17:21 with the bench queued (exclusive 1024-164700-b533; non-exclusive fallback 1024-171406-2860). Gate: ≤ 5 ms per layer at nb 510.

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
