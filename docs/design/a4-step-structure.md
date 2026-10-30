# WP-A4 / WP-C1: decode-step structure, exact mode, consumer GPUs first

**Status: Checkpoint 0 (design, no code).** Branch `feat/a4-step-structure` off `origin/main` `fde94b6`. Scope: the
spec's WP-A4 (i) and (ii) (`specs/beat-vllm-scope.md` §6 Track A, the same package as Track C's WP-C1), in **exact**
mode, built and judged on bella (RTX 5070 Ti Laptop, sm_120, 46 SMs, 12 GB). (iii), the lm_head on the fast GEMM, is
out of scope. Hopper (the spec's H100 kill rule) comes after this, on a pod the lead books.

Labels: **MEASURED** (a cited log or spec table), **CODE** (file:line on `fde94b6`), **ESTIMATED** (arithmetic
shown), **OPINION**.

## 0. Summary

- (i) **q|k|v in one launch** for decode-sized batches (17 ≤ nb ≤ 128): one grid over q's, k's and v's row tiles. The
  k and v tiles fill the idle SMs of q's last wave.
- (ii) **32-row tiles for `down`** where the wave model says they pay. On bella that is nb ≤ 64 only (see §2.2).
- No per-output reduction order changes. Every output is computed by the same device code as on main (same
  `g2_compute` instantiation, same stage order over K, same epilogue). Only the blockIdx → (matrix, row tile) map and
  the tile's row count change. `NUMERICS_VERSION_EXACT` stays 1 and the fingerprint is unchanged. Nothing goes behind
  `--numerics fast`.
- On bella the gain is smaller than the spec's H100 figure, and part of the spec's (ii) does not apply there: main
  already routes `q`, `o`, Q4_K `down` and `gate_up` to the 128-row gemm2t tile at nb 128 on 46 SMs (§2.2). ESTIMATED
  gain is up to −6.7 % at nb 32 and −2 % at nb 128 (§3). The A/B bar in §4 is set at about 60 % of those estimates,
  the same ratio as the spec's (−10 % bar for a −17 % estimate).

## 1. Where the step's time goes on bella

MEASURED, bella, 7B Q4_K_M, `pool_step_cost` profile at d600 (log `1018-090100-a107`, main `1a55a4b` of 10-18; older
than `fde94b6`, so the A/B re-measures main). The step is 24.4 ms with the graph (A/A over 6 runs: 24.32–24.48 ms,
±0.3 %). The ungraphed profile below totals 25.04 ms, in ms per step over 28 layers:

| nb | q | k | v (v6 + v4) | wo | gate_up | down6 (14 Q6_K layers) | down4 (14 Q4_K layers) | attn | head |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 1.29 | 0.40 | 0.47 | 1.08 | 8.96 | 5.20 | 2.46 | 2.43 | 1.72 |
| 64 | 1.20 | 1.42 | 0.69 | 1.16 | 8.44 | 4.50 | 2.86 | 6.62 | 2.18 |

MEASURED (GGUF headers): the 7B's `attn_v` and `ffn_down` are Q6_K on 14 of 28 layers and Q4_K on the rest. `q`, `k`,
`o`, `gate` and `up` are all Q4_K. `down` is 31 % of the nb 32 step (7.66 ms). The Q6_K `down` costs 2.1× the Q4_K one
for 1.46× the bytes.

Routing on bella at the A/B's batch sizes, from the code (CODE `batch.rs`: `bgemm` 1121, `tile_t` 383, `takes_gemvm`
2654, `bgemm_x` 2580; `layer`'s qkv block 1855–1868; `ffn_half`'s `down` 2555/2562):

| product (7B) | nb 32 | nb 64 | nb 128 |
|---|---|---|---|
| q (3584 rows) | gemm2s 64×64, 56 CTAs | gemm2s, 56 | **gemm2t 128×64, 56** (`tile_t` true at 46 SMs) |
| k, v (512 rows) | **gemvm32** (`takes_gemvm`: 4·8 < 46) | gemm2s, 8 each | gemm2s, 16 each (`tile_t` false) |
| down Q4_K (3584 rows) | gemm2s, 56 | gemm2s, 56 | **gemm2t, 56** |
| down Q6_K | gemm2s Q6_K (register-staged), 56 | same | same (64×128 needs 1.5 CTAs/SM: not taken) |

Occupancy (CODE `qwen2.rs:111` `gemm2_smem_bm`): gemm2s Q4_K 64×64×3 stages uses 87.8 KB of shared memory and
gemm2s Q6_K 83.3 KB. sm_120 (like sm_89) has 100 KB of shared memory per SM, so both run **1 CTA/SM**, not the
H100's 2. That is why the spec's H100 arithmetic (144 CTAs = one wave of 264 slots) does not carry over.

## 2. What changes, and why no bit does

The contract this rests on: in exact mode, every product kernel gives each output (row r, token z) the same bits:
gemv, gemvm8/16/32, gemvmx, gemm2s, gemm2 and gemm2t. Main already switches between them by nb and SM count with
"results don't depend on it" (CODE `tile_t`, `takes_gemvm`, `wave_passes` docs). That is tested by
`pool_steps_past_gemvm16_match_solo` (vm_max 16/32/0, so the 7B-shaped k/v go to gemvm32 or gemm2s at the same nb)
and `gpu_batched_products_match_gemv_bits`. For the gemm2 family the reason is structural (CODE `prefill.cu:1838`
`g2_compute`, 2174 kernel body):

- Each output's float chain is `acc` over the K stages s = 0..nst−1, in order, inside one thread. Per stage it takes
  one f32 step (`fmaf(dk, d·S − dmin·M, acc)`, or the TW twin's mul-then-add) of **integer-exact** mma sums (S, M:
  int32, order-free).
- Then one epilogue (bias, mode 0 store or mode 1 residual add).
- BM, BN, the CTA's place in the grid and the neighbouring rows and tokens enter none of it. Padding tokens (z ≥ nb)
  are zero-filled and never stored.

So a change that keeps, per output, the same `QT`, the same `TW` flag, the same stage order and the same epilogue
cannot change a bit. Both changes are of that kind.

### 2.1 (i) q|k|v as one launch

**Today** (`layer`, `batch.rs:1855`): for nb ≤ 16, `bgemm_x` already runs q|k|v as one gemvmx launch. For nb > 16
that path declines and q, k, v are three `bgemm` launches (two kernels when v is Q6_K).

**Change:** a new `bgemm_g2x` that `layer` tries for 17 ≤ nb ≤ 128 before falling back to the three `bgemm` calls.
It launches one grid whose blockIdx.y covers q's row tiles, then k's, then v's. The large CTAs go first, so the
k/v CTAs are dispatched into q's tail wave. A small segment table (weights, rows, bias, out per matrix) is passed as
kernel args, as gemvmx passes its three matrices (CODE `bgemm_x` 2580).

Precedent: `moe_res.cu`'s grouped expert GEMMs already launch several matrices' tiles in one grid through the same
kernel body (the `zt` tile-list parameter, `prefill.cu:2174`).

- **Same type only.** All segments of a launch share `QT`, `cols` and `rb` (`bgemm_x`'s checks).
  - On Q4_K-v layers, q|k|v is one launch.
  - On Q6_K-v layers, q|k is one launch and v runs alone, as today.
  - Fusing mixed-type segments (a kernel holding both the Q4_K and the Q6_K paths) is a possible follow-up, only if
    the A/B shows the Q6_K-v layers are where the remaining time is.
- **Tile per segment, as main picks it.** At nb ≤ 64 every segment is a gemm2s 64×64 tile, i.e. the existing gemm2s
  instantiation with a segment lookup at the top. At nb > 64, when `tile_t` gives q the 128-row gemm2t tile (bella,
  nb 128), the kernel holds both instantiations, WR = 4 for q's tiles and WR = 2 for k/v's, and branches on the
  segment. Shared memory is the max of the two (gemm2s 87.8 KB vs gemm2t 80.5 KB: still 1 CTA/SM).
- **k/v at nb 17..32 move from gemvm32 into the fused gemm2s grid.** The bits are the same (the contract above,
  tested at vm_max 16 vs 32). Taking this route only pays when the k/v CTAs fit in q's tail. The router fuses only if
  `fused CTAs ≤ ⌈q CTAs / sms⌉ · sms`, i.e. k/v add no wave. Otherwise it keeps today's launches.
  - On bella at nb 32: q has 56 CTAs, so 1 wave + 10 with 36 SMs idle in the tail. k|v add 16 CTAs, so fused is 72
    CTAs, still 2 waves.
  - On the 4090 (128 SMs) at nb 32: q's 56 CTAs leave 72 SMs idle, so the fusion fits there too. The spec's WP-C1
    warning ("k/v already take gemvm32 at nb ≤ 32") is about the gain, not about bits.
- **Prefill** passes of 17..128 tokens take the same `layer` code and so get (i) too. That is how the md5 gate
  exercises it (§5). Passes over 128 tokens are untouched, so `gemm_cost`/`wave_passes` keep their model.
- Kill switch `AXON_QKV_FUSE=0` (a `BatchFns` field, so tests can flip it in-process the way they flip `vm_max`).

Launches per layer for 17 ≤ nb ≤ 128: 3 → 1 on Q4_K-v layers, 3 → 2 on Q6_K-v layers.

### 2.2 (ii) 32-row tiles for `down`

**Today** (`ffn_half`, `batch.rs:2555` mode 0 and `2562` mode 1): `down` (7B: 3584 rows × 18944 cols) is one
`bgemm`: 56 gemm2s CTAs at nb ≤ 64, and gemm2t (also 56) at nb 128 on bella.

**Change:** a 32×64 tile.

- **Q4_K / Q5_K** (raw-staged): the existing kernel template with WR = 1 (BM 32; 8 warps along tokens, BN = 64·NT,
  NT = 1). `g2_copy` already takes BM. `g2_compute` is per warp and 32 rows (`row 32 wr + 16 mi + g + 8(e>>1)`) and
  does not change. 112 CTAs for the 7B; shared memory 71.3 KB, so still 1 CTA/SM on sm_120.
- **Q6_K** (register-staged, step (ii-b)): `g2_item`/`g2_load_w`/`g2_store_w` assume 64 rows (`static_assert
  G2_THREADS == 4·G2_BM`, `prefill.cu:1443`; `WR == 2 || RAW`, 2185). (ii-b) gives them a BM parameter: at BM 32 each
  thread loads half an item's rows, and the meta layout offsets use BM. The dequant values written to shared memory
  and the compute are byte-identical; only which thread copies which row changes. The 14 Q6_K `down` layers are
  68 % of `down`'s nb 32 time (5.20 of 7.66 ms), so (ii) without (ii-b) leaves most of the gain behind.
- **Router:** a `tile_r32(rows, nb, sms)` predicate built like `tile_t`, on the same `wave_cost` and `PARTIAL_WAVE`
  model. It takes 32 rows when the model says the waves cost less. A per-MAC factor for the 32-row tile
  (`TILE_R32_TIME`, like `TILE_T_TIME`) starts at 100 (no penalty) and is fitted from the A/B's profile, not
  guessed.
  - On bella the model picks 32-row tiles at nb ≤ 64 (56 CTAs: 1.45 waves → 112 half-CTAs: 1.30, ESTIMATED −10 % of
    `down`).
  - At nb 128 it keeps gemm2t: 224 quarter-size CTAs cost 2.45 vs gemm2t's 2.38 (ESTIMATED, in gemm2s-CTA waves).
  - So the spec's "(ii) at nb ≤ 128" becomes "nb ≤ 64" on bella. The H100 (2 CTAs/SM, 132 SMs) is where the nb 128
    half applies; that is the follow-up pod run.
- Applies to `down` only (scope). `o` has the same 3584-row shape and could reuse the predicate later.
- Kill switch `AXON_DOWN_R32=0`.

## 3. ESTIMATED gains on bella

Wave arithmetic uses the repo's own fitted model (CODE `wave_cost`; `PARTIAL_WAVE` 30, fitted on 4090 prefill).
Times are scaled from §1's MEASURED profile (main `1a55a4b`).

| cell | (i) | (ii) | total | of step |
|---|---|---|---|---|
| nb 32 d600 (24.4 ms) | k+v ride q's tail: −0.87 ms (all of k+v, upper bound) | down4 −10 % (−0.25) + down6 −10 % (−0.52, needs ii-b) | **≈ −1.6 ms** | **−6.7 %** |
| nb 128 d256 | q|k|v: k+v in q's gemm2t tail ≈ −1.1 gemm2s-waves per Q4_K-v layer, −0.54 per Q6_K-v layer, at ≈ 29 µs per wave (nb 64 q) ≈ −0.67 ms | none (router keeps gemm2t) | **≈ −0.7 ms** | **≈ −2 %** (step ≈ 35 ms, ESTIMATED: unmeasured at nb 128 on bella) |

OPINION: (i)'s nb 32 figure is an upper bound because it assumes the k/v CTAs are fully hidden in q's tail. Gemvm32's
k at 14 µs/layer is already cheap.

## 4. Pre-registered A/B rule

The spec's kill rule (H100): nb 128 d600 step not ≤ 13.0 ms (−10 % of 14.55) → revert the tile change; keep the
fusion if it alone wins ≥ 0.8 ms. Adapted to bella:

- **Harness:** `cuda::qwen2::slots::tests::pool_step_cost` (`slots.rs:2418`, `#[ignore]`).
  - Settings: `AXON_STEP_MODEL` = the 7B Q4_K_M, f16 KV, graphs on (the harness default); `AXON_STEP_PROF` for the
    per-kernel split.
  - Main `fde94b6` and the candidate run as two binaries in **one** `axon-run --detach` timing job, alternating
    A B A B A B (3 each), cool-gated (the harness logs temperature and clock per row).
  - The statistic is the median ms/step per cell. A/A noise is the spread of main's 3 runs in the same job (10-18
    reference: ±0.3 %).
- **Deciding cells:**
  - nb 32, d600.
  - nb 128, d256. 128 slots × 704 positions of f16 KV (5.2 GB) do not fit beside the 7B (4.7 GB) in 12 GB, so nb 128
    runs at d256 (pool 128 × 320).
- **Regression cells:** nb 16 d600 (the gemvmx path; must not move), nb 64 d600, and solo `forward_argmax`.
- **Pass:** candidate ≤ 0.96 × main at nb 32 (−4 %, ≈ 60 % of the ESTIMATED −6.7 %), and ≤ 0.99 × main at nb 128
  (−1 %, ≈ 50 % of −2 %, above the ±0.3 % noise). Each gain must exceed 2× the job's A/A spread. No regression cell
  may be more than 0.5 % worse.
- **Partial:** if the combined candidate misses the nb 32 bar, rerun with `AXON_DOWN_R32=0`.
  - If (i) alone gives ≥ 2 % at nb 32 (≈ 60 % of its −3.6 %), keep (i) and revert (ii).
  - If (ii) alone (`AXON_QKV_FUSE=0`) gives ≥ 2 %, keep (ii) and revert (i).
  - A change that fails its own bar is reverted, not tuned until it passes.
- **Kill:** if neither part reaches 2 % at nb 32 and neither reaches 1 % at nb 128, stop WP-C1 on consumer GPUs and
  record it in `experiments-rejected.md`. No 4090/H100 pod is asked for it.
- **Secondary, not deciding:** `scripts/serve_load.py load` at C = 8/16/32 (7B, `--parallel 32`, a small `--ctx`
  that fits), agg tok/s and ITL p50/p99 vs main, reported in the PR.

## 5. Bit-identity tests (all through `axon-run --detach` on bella)

Existing tests, which must stay green and are extended where noted:

1. `cuda::qwen2::tests::gpu_batched_products_match_gemv_bits` (`tests.rs:3031`): batched products == per-token gemv
   bits. Extended to nb ∈ {17, 24, 32, 33, 48, 64, 65, 96, 128}, run with the fused qkv launch and 32-row `down`
   tiles forced on.
2. `cuda::qwen2::slots::tests::pool_steps_past_gemvm16_match_solo` (`slots.rs:2309`): pool rows == solo under vm_max
   16/32/0. Extended with nb 65 and 128 and with both kill switches on and off.
3. `pool_steps_match_solo_in_any_order` (`slots.rs:1202`), `pool_wide_long_steps_match_solo` (`slots.rs:2174`),
   `pool_rows_of_one_slot_match_verify` (`slots.rs:1060`): row order, long contexts, f16 and q8 KV, verify rows.

New tests:

4. `gpu_qkv_one_launch_matches_three` (`tests.rs`): per test model (0.5B Q4_K_M with Q4_K and Q6_K v layers,
   Llama 1B with no biases, qwen3 0.6B Q8_0 as a register-staged type at BM 64), the qkv buffer from one `bgemm_g2x`
   launch == three `bgemm` launches, **byte for byte**. Covered: every nb in test 1's set; both segment layouts (q|k|v,
   q|k + v); the WR 4 + WR 2 mixed grid (nb 65..128, forced on with `tile_t`).
5. `gpu_down_rows32_matches_rows64` (`tests.rs`): `down` through 32-row tiles == the 64-row gemm2s and the gemm2t
   result, byte for byte, for Q4_K and Q6_K (ii-b), epilogue mode 0 (into `up`) and mode 1 (residual into `x`), nb in
   test 1's set.

Gates:

6. **md5 gate vs main** (AGENTS.md "The md5 gate"): greedy `axon gen` on the 0.5B, 1.5B and 7B. Prompt prefill
   passes of 17..128 tokens take (i), so this exercises the fused launch end to end.
7. **Serve identity:** `serve_load.py texts` at `--c 1` and `--c 8`, against main's `--c 1` digest. Caveat: its 16
   requests keep nb ≤ 16, so the steps take gemvmx and not the new code. Tests 2–3 are what cover pool steps of
   17..128 rows.

`NUMERICS_VERSION_EXACT` is not bumped. If any of tests 1–7 shows one differing bit, the change is a bug, not a
numerics change, and does not ship.

## 6. Plan after review

- Checkpoint 1: (i) at nb ≤ 64 (one gemm2s instantiation plus the segment table), with tests 1, 2 and 4, then the
  A/B. This is the cheapest part and the most informative.
- Checkpoint 2: the WR 4 + WR 2 mixed grid (nb 65..128).
- Checkpoint 3: (ii) for Q4_K, then (ii-b) for Q6_K, with test 5.
- Then the full A/B (§4), fmt/clippy/CPU suites, the md5 and serve gates, and the PR (template: methodology,
  benchmarks, models, hardware, gates).
- Pods (4090 re-board, H100 nb 128 kill rule) only if the lead books them.
