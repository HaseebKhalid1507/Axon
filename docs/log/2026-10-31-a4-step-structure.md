# WP-A4 (i): a layer's Q4_K q, k (and v) in one launch for decode-sized batches (exact, same bits)

- Date: 2026-10-31. Branch: feat/a4-step-structure. Base: main fde94b6. This is WP-A4 / WP-C1 of the beat-vLLM scope
  (`specs/beat-vllm-scope.md` §6), in exact mode, on consumer GPUs first. Design and results:
  [a4-step-structure.md](../design/a4-step-structure.md).

Labels: **MEASURED** (job cited), **CODE**.

## What changed

- **`prefill.cu`:** `gemm2x_q4_k` and `gemm2xt_q4_k` put two or three Q4_K products of the same activations into one grid, mode 0. In
  `gemm2x` every tile is gemm2s's; in `gemm2xt`, q's tiles are gemm2t's. Each output comes from the same `gemm2_t`
  instantiation as its own launch, with the same stage order and epilogue. Only the blockIdx → (matrix, row tile) map changes. The
  map uses a new `yt` argument of `gemm2_t`, the row-tile twin of the existing `zt`.
- **`batch.rs`:** `bgemm_g2x` fuses q|k|v, or else q|k, at 17 ≤ nb ≤ 128 (`G2X_MAX_NB`). It fuses only when the small matrices' blocks fit
  the SMs that q's last wave leaves idle, so they add no wave. q keeps the tile `bgemm` would give it. `AXON_QKV_FUSE=0` turns it off.
- **Tests:** `gpu_qkv_one_launch_matches_three` (new) compares against main's `bgemm` routing, byte for byte. Two tests are extended:
  `gpu_batched_products_match_gemv_bits` and `gpu_products_match_reference_bits` now cover the decode batch sizes, and
  `pool_steps_past_gemvm16_match_solo` goes up to 128 rows with the fusion on and off.
- **Harnesses (test builds only):** `pool_step_cost` prints `HOST`, the per-step host time inside `step_checked`, and
  `cuda_gemm_tile_bench` takes `AXON_BENCH_SHAPE`.
- **Dropped:** (ii), the 32-row `down` tiles. The M2 micro-bench measured them 7–15 % slower than gemm2s
  (`1030-130922-c603`). They were built in `7b32b3a` and removed in `05b48b6`.

## Numbers (bella, RTX 5070 Ti Laptop sm_120, 7B Q4_K_M, graphed pool step; MEASURED, `1031-045802-1654`)

| depth | nb 16 | nb 32 | nb 64 | nb 128 |
|---|---|---|---|---|
| 600 | −0.1 % | **−2.3 %** | −2.3 % | −2.2 % |
| 256 | 0.0 % | −1.7 % | −1.6 % | −2.0 % |

These are medians, candidate vs main. The A/A control (the candidate with `AXON_QKV_FUSE=0`) is within −0.7 to +0.9 % of main.
The job was flagged TAINTED by a single Xorg sample at 54 %, with a mean of 0.9 %; the lead accepted it with disclosure. nb 128 is
informational (M3). The drift-paired estimate at nb 32 d600 is −1.4 % (design note §7.2).
