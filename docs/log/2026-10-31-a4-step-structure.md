# WP-A4: a layer's Q4_K / Q5_K down on gemm2t from nb 32 (exact, same bits); q|k|v fusion and 32-row tiles dropped

- Date: 2026-10-31. Branch: feat/a4-step-structure. Base: main fde94b6. This is WP-A4 / WP-C1 of the beat-vLLM scope
  (`specs/beat-vllm-scope.md` §6), in exact mode, on consumer GPUs first. Design and results:
  [a4-step-structure.md](../design/a4-step-structure.md) (§7, §8).

Labels: **MEASURED** (job cited), **CODE**, **OPINION**.

## What ships

- **`batch.rs`:** a layer's Q4_K / Q5_K `down` keeps gemm2s up to `DOWN_T_MIN_NB` = 31 tokens instead of
  `TILE_T_MIN_NB` = 64. Past it, `tile_t_from`'s wave model picks gemm2t where it pays (`bgemm_at`; the layer's two
  `down` calls). `AXON_DOWN_TILE_T_MIN_NB` overrides it, and 64 restores the old routing. Every other product keeps 64.
- **Bits:** gemm2t and gemm2s give the same bits per output (`gpu_batched_products_match_gemv_bits`: both equal the
  decode GEMV; extended to nb 17..129). `gpu_down_tile_t_matches_bgemm` (new) checks the moved route against the old one
  byte for byte, in modes 0 and 1. `down_tile_t_floor_routes` (new, CPU) pins what moves.
- **What moves (CODE, the wave model):** on bella (46 SMs), downs of 3072 rows and up at nb 32..64; the 7B's is measured.
  On a 4090 (128 SMs), an H100 (132) and a 5090 (170), no down of a hidden size ≤ 8192 moves. Smaller cards move more
  (from 1408 rows on 20 SMs), unmeasured.
- **Kept from the experiments (tests only):** `pool_steps_past_gemvm16_match_solo` up to 128 rows, and the
  `AXON_BENCH_SHAPE` filter of `cuda_gemm_tile_bench`.

## Numbers (bella, RTX 5070 Ti Laptop sm_120, 7B Q4_K_M, graphed pool step; MEASURED, `1031-065003-733e`)

Pre-registered run (note §8): arms M (main), A (fusion on), T (A plus the floor at 31), F (fusion off, the A/A control).
Each run started at 54–55 °C. The statistic is the median of paired deltas against the M runs on either side. TAINTED by one
Xorg sample at 41 % (mean 1.2 %); the lead accepted the run, because its noise term is the in-job F control.

| depth | nb | T vs A (the floor) | A vs M (the fusion) | S_F |
|---|---|---|---|---|
| 600 | 16 | −0.11 % | +0.01 % | 0.31 % |
| 600 | **32** | **−2.47 %** | **−1.80 %** | 0.96 % |
| 600 | 64 | −2.53 % | −0.84 % | 6.30 % |
| 600 | 128 | −0.10 % | −2.44 % | 1.29 % |
| 256 | 32 | −2.62 % | −1.90 % | 0.71 % |
| 256 | 64 | −2.29 % | −3.24 % | 0.48 % |

The floor's gain is taken as T against A. It was measured with the fusion on, and the shipped head has no fusion; the two
touch different products, so the gain is assumed to carry over (unmeasured on the final head).

## NO-GO

- **(i) q|k|v in one launch** (`gemm2x_q4_k` / `gemm2xt_q4_k`, built in `7b32b3a`, removed in `fe20964`). The first A/B
  (`1031-045802-1654`) was INCONCLUSIVE: −2.3 % by median, but not more than 2 × main's spread, and −1.4 % drift-paired.
  The pre-registered rerun gave −1.80 % at nb 32 d600, which is not beyond its noise term (2 S_F = 1.93 %). Lead ruling
  10-31: removed.
- **(ii) 32-row `down` tiles** (built in `7b32b3a`, removed in `05b48b6`). The M2 micro-bench (`1030-130922-c603`)
  measured them 7–15 % slower than gemm2s and 35–41 % slower than gemm2t. That bench is also where the gemm2t lead came
  from: 145 vs 184 µs for the 7B's down at nb 32. The bench was TAINTED (foreign GPU load mean 7.1 %, max 39 %). Its kernels
  ran interleaved and their per-round ranges don't overlap: the slowest gemm2t round beats the fastest gemm2s round (0.82 at
  nb 32, 0.89 at nb 64).
