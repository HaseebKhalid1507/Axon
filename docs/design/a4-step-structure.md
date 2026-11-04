# WP-A4 / WP-C1: decode-step structure, exact mode, consumer GPUs first

**Status: Checkpoint 4, the PR (#178).** What ships: a layer's Q4_K / Q5_K `down` takes gemm2t from nb 32 instead of
nb 65 (`DOWN_T_MIN_NB` = 31), −2.5 % per 7B pool step at nb 32 on bella, same bits (§8). (i), q|k|v in one launch, is
**dropped**: the pre-registered rerun put it at −1.80 %, not beyond its noise term (2 S_F = 1.93 %, §8.1), and the lead ruled it out (§8.2). Its
code was removed in `fe20964`. (ii) was dropped by M2 with no A/B (§7.1); its code was removed in `05b48b6`. CP0 review: shady
7/10 BUILD-WITH-MUSTS (commit comment on `70f5fd0`). Its MUSTs M1–M3 are folded in below (§4, §5). Branch `feat/a4-step-structure` off `origin/main` `fde94b6`. Scope: the
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
  gain for CP1's scope is up to −4.1 % at nb 32 and −1.8 % at nb 128 (§3), against the `fde94b6` baseline. The
  A/B bars in §4 are 60 % of the estimates, the spec's own ratio (−10 % bar for a −17 % estimate).

## 1. Where the step's time goes on bella

MEASURED, bella, 7B Q4_K_M, `pool_step_cost` on main `fde94b6` (job `1030-124131-7858`, two runs in one job). Run 1
had no foreign GPU load (`FOREIGN_GPU sample max=0%`); run 2 had Xorg/kitty at 8 % mean, 14 % max. Graphed ms/step:

| cell | run 1 (clean) | run 2 (8 % foreign) | A/A spread | solo `forward_argmax` |
|---|---|---|---|---|
| nb 32 d600 (pool 64 × 704) | 22.995 | 23.171 | 0.77 % | 12.29 / 10.46 |
| nb 32 d256 (pool 128 × 320) | 21.724 | 22.100 | 1.7 % | 10.66 / 10.82 |
| nb 128 d256 | 38.751 | 38.947 | 0.51 % | (as above) |

A pool of 128 × 704 does not fit (CUDA OOM, MEASURED), so nb 128 runs at d256. The ungraphed per-kernel profile, run
1, ms per step summed over 28 layers:

| cell | total | q | k | v6 + v4 | wo | gate_up | down6 (14 Q6_K layers) | down4 (14 Q4_K layers) | attn | head |
|---|---|---|---|---|---|---|---|---|---|---|
| nb 32 d600 | 23.43 | 1.23 | 0.46 | 0.28 + 0.23 | 1.17 | 8.45 | 3.55 | 2.65 | 2.15 | 2.25 |
| nb 128 d256 | 39.59 | 2.11 | 0.70 | 0.45 + 0.34 | 2.12 | 13.52 | 5.71 | 4.77 | 4.63 | 3.30 |

(CP0's table came from main `1a55a4b` of 10-18, log `1018-090100-a107`. These replace it.)

MEASURED (GGUF headers): the 7B's `attn_v` and `ffn_down` are Q6_K on 14 of 28 layers and Q4_K on the rest. `q`, `k`,
`o`, `gate` and `up` are all Q4_K. `down` is 26 % of the nb 32 d600 profile (6.20 of 23.43 ms). The Q6_K `down` costs
1.34× the Q4_K one for 1.46× the bytes.

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

**Dropped (§7).** M2's micro-bench measured the 32-row tiles 7–15 % slower than gemm2s, the tile the router gives the 7B's
`down` at nb 32 and 64. They were 35–41 % slower than gemm2t. Kernels, router, switch and test were removed in `05b48b6`. The design
below is kept as the record.

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
  57 % of `down`'s nb 32 time (3.55 of 6.20 ms, `fde94b6`), so (ii) without (ii-b) leaves most of the gain behind.
- **Router:** a `tile_r32(rows, nb, sms)` predicate built like `tile_t`, on the same `wave_cost` and `PARTIAL_WAVE`
  model. It takes 32 rows when the model says the waves cost less. A per-MAC factor for the 32-row tile
  (`TILE_R32_TIME`, like `TILE_T_TIME`) is a 100 placeholder in CP1's code, with the router off; it is set from
  the micro-bench (§4 step 1) before the router is ever on by default.
  - On bella the model picks 32-row tiles at nb ≤ 64 (56 CTAs: 1.45 waves → 112 half-CTAs: 1.30, ESTIMATED −10 % of
    `down`).
  - At nb 128 it keeps gemm2t: 224 quarter-size CTAs cost 2.45 vs gemm2t's 2.38 (ESTIMATED, in gemm2s-CTA waves).
  - So the spec's "(ii) at nb ≤ 128" becomes "nb ≤ 64" on bella. The H100 (2 CTAs/SM, 132 SMs) is where the nb 128
    half applies; that is the follow-up pod run.
- Applies to `down` only (scope). `o` has the same 3584-row shape and could reuse the predicate later.
- Switch `AXON_DOWN_R32`: unset or 0 is off (CP1's default), 3 takes gemm2r (3 stages), 2 gemm2r2 (2 stages, 2
  CTAs/SM).

## 3. ESTIMATED gains on bella

Wave arithmetic uses the repo's own fitted model (CODE `wave_cost`; `PARTIAL_WAVE` 30, fitted on 4090 prefill). Times
are from §1's MEASURED `fde94b6` profile, run 1. CP1 builds (i) and (ii) for Q4_K only (no ii-b), so the Q6_K `down`
and the Q6_K `v` gain nothing here.

| cell | (i) | (ii), Q4_K only | total | of step |
|---|---|---|---|---|
| nb 32 d600 (22.995 ms) | k (0.46) + v4 (0.23) ride q's tail: −0.69 ms, upper bound (v6 runs alone) | down4 −10 % (−0.26) | **≈ −0.95 ms** | **−4.1 %** |
| nb 128 d256 (38.751 ms) | k + v4 in q's gemm2t tail, ≈ −0.7 ms (CP0's wave estimate; k + v4 measure 1.03 ms, upper bound) | none (`tile_r32` stops at nb 64) | **≈ −0.7 ms** | **≈ −1.8 %** |

OPINION: both (i) figures are upper bounds. They come from the ungraphed profile, so they include launch gaps the graph
already removes (review point), and they assume the k/v CTAs are fully hidden in q's tail. The −10 % for (ii) is the
wave model at `TILE_R32_TIME` = 100. Shady's point stands: a 32-row CTA stages the same activation tile for half the
rows, so its true per-MAC cost is unknown. M2's micro-bench measures it before the router is used (§4).

## 4. Pre-registered A/B rule

The spec's kill rule (H100): nb 128 d600 step not ≤ 13.0 ms (−10 % of 14.55) → revert the tile change; keep the
fusion if it alone wins ≥ 0.8 ms. Adapted to bella, with CP0 review's M2 and M3. All of it runs in **one**
`axon-run --detach --timing bella --low` job (`.a4bin/ab.sh`, committed before submission and not edited after):

**Step 1: the (ii) micro-bench (M2).** `cuda_gemm_tile_bench` (candidate binary) with `AXON_BENCH_SHAPE=down`,
`AXON_BENCH_NB=32,64`, 7 rounds: the 7B's Q4_K down (3584 × 18944) on gemm2s (today's tile), gemm2r (32 rows, 3
stages, 1 CTA/SM) and gemm2r2 (32 rows, 2 stages, 47.5 KB, 2 CTAs/SM), the median µs per launch in isolation. Rule:

- A 32-row variant **qualifies** if its median is ≤ 0.95 × gemm2s's at **both** nb 32 and nb 64. The best qualifying
  variant is (ii)'s; its ratio at nb 32 becomes `TILE_R32_TIME`.
- If neither qualifies, (ii) is **dropped with no A/B**: its kernels and router are removed before the PR, and (ii-b)
  never starts. The A/B rows below that carry (ii) are then reported but decide nothing.

**Step 2: the pool A/B.** `cuda::qwen2::slots::tests::pool_step_cost`, the 7B Q4_K_M, f16 KV, graphs on,
`AXON_STEP_DEPTHS=600,256`, `AXON_STEP_NB=16,32,64,128` (nb 128 only fits at d256). Four configurations:

- **M**: main `fde94b6`'s test binary.
- **A**: the candidate, defaults ((i) on, (ii) off).
- **B**: the candidate, `AXON_DOWN_R32=3` ((i) + (ii) with gemm2r).
- **C**: the candidate, `AXON_DOWN_R32=2` ((i) + (ii) with gemm2r2).

Three rounds, the order rotated each round (M A B C, B C M A, C M A B). The statistic is the median ms/step per cell
over the three rounds. The A/A spread of a cell is (max − min) / median of M's three runs in this job.

- **M3, nb 128 is provisional.** If M's nb 128 d256 spread is over 0.5 %, nb 128 decides nothing, and that is
  reported. The baseline job already measured 0.51 % between a clean run and a run with 8 % foreign load (§1), so this
  is a real risk.
- **Pass for (i)** (A vs M): ≤ 0.982 × M at nb 32 d600 (−1.8 %, 60 % of the ESTIMATED −3.0 %), and more than 2 × the
  A/A spread of that cell. At nb 128 d256, if M3 allows: ≤ 0.99 × M (−1 %, ≈ 55 % of −1.8 %).
- **Pass for (ii)** (the qualifying variant's config vs A, the share (ii) adds on top of (i)): ≤ 0.995 × A at nb 32
  **and** nb 64 d600. (ii)'s ESTIMATED −1.1 % of the step is close to the noise, so its step bar is "measurably better,
  never worse" and its real bar is the micro-bench. (ii-b) for Q6_K starts only if (ii)-Q4_K passes **both**.
- **Regression cells:** nb 16 at both depths (the gemvmx path, which CP1 does not touch), and solo `forward_argmax`.
  No regression cell may be more than 0.5 % worse than M, and nb 16 must not move beyond its A/A spread.
- **Kill:** if (i) misses its nb 32 bar and nb 128 decides nothing or misses too, stop WP-C1 on consumer GPUs and
  record it in `experiments-rejected.md`. No pod is asked for it. A change that fails its own bar is reverted, not
  tuned until it passes.
- **Changed from CP0 and why:** CP0's bars (−4 % at nb 32 combined, ≥ 2 % for either part alone) were 60 % of
  estimates made from the `1a55a4b` profile, assuming (ii-b) and that all of v fuses. The `fde94b6` baseline and the
  CP1 scope (Q4_K only; v6 stays alone) lower the estimates, so the bars are recomputed by the same 60 % rule. They are
  set before any candidate run.

**Step 3: host time per step (for A4 vs A6 ordering).** The candidate's `pool_step_cost` prints, per cell, a `HOST`
line. It is the mean time `step_checked` spends on the CPU outside the GPU's work: from its entry to the graph launch's
return, plus its tail after the picks' read (test builds only: `#[cfg(test)]` counters, zero cost in the product).
Compare it with the step's ms: that is the host gap A6 (iii) could overlap.

**Secondary, not deciding:** `scripts/serve_load.py load` at C = 8/16/32 vs main, in the PR.

## 5. Bit-identity tests (all through `axon-run --detach` on bella)

What each test compares against (M1): every reference below is the code main runs, or the per-token GEMV, never a
forced variant of the new kernel.

1. `cuda::qwen2::tests::gpu_batched_products_match_gemv_bits`: every batched product kernel == the per-token decode
   GEMV, bit for bit, on the 0.5B's real tensors and synthetic shapes. Extended: nb ∈ {1, 6, 7, 8, 13, 17, 24, 31, 32,
   33, 48, 64, 65, 66, 77, 96, 128, 129}. `gpu_products_match_reference_bits` (against `gemv_ref`, the order
   written out) uses the same list. (CP1 also put gemm2r/gemm2r2 in `gemm2_names`; they went with (ii).)
2. `cuda::qwen2::slots::tests::pool_steps_past_gemvm16_match_solo`: pool rows == each request run alone (decode GEMV),
   0.5B and Llama 1B. Extended to 128 slots, nb up to 128, and the switches as configurations: vm_max 16 and 32 (fuse
   on), then by shape with fuse off and fuse on. The Llama's q and k (and v on 8 layers) are Q4_K, so it takes the
   fused launch. (CP1's r32 configurations went with (ii).)
3. `pool_steps_match_solo_in_any_order`, `pool_wide_long_steps_match_solo`, `pool_rows_of_one_slot_match_verify`:
   unchanged, must stay green.

New tests:

4. `gpu_qkv_one_launch_matches_three` (`tests.rs`): the q|k|v buffer from one `gemm2x`/`gemm2xt` launch == the
   launches **main's `bgemm`** makes for each matrix at that nb on this GPU, byte for byte over the whole buffer
   (gemvm32 for k/v where `takes_gemvm` picks it at 17..32, gemm2s above, gemm2t for q where `tile_t` picks it). Both
   fused variants at every nb, plus the router's own choice (`bgemm_g2x`, falling back to `bgemm`). Shapes: the 7B's
   (3584 + 512 + 512 × 3584), an odd one (1000 + 136 + 72 × 1536), and two matrices (333 + 77 × 768); with and
   without biases; nb ∈ {17, 24, 32, 33, 48, 64, 65, 96, 128}. Transitively against the GEMV: `bgemm`'s kernels at
   these sizes are test 1's.
5. *(Removed with (ii) in `05b48b6`.)* `gpu_down_rows32_matches_rows64` (`tests.rs`): gemm2r and gemm2r2 == gemm2s (and gemm2t), byte for byte, Q4_K,
   epilogue mode 0 (plain, bias) and mode 1 (residual), the same nb set. Shapes: the 7B's down (3584 × 18944) and odd
   ones whose K stage counts (74, 1, 3, 4, 5, 14) leave the last pass through the 3- or 2-slot ring partial. A K-quant
   stage is one whole 256-wide block, so that is the only partial stage there is. (ii-b)'s Q6_K shapes join this test
   only if (ii-b) is ever built.

Gates:

6. **md5 gate vs main** (AGENTS.md "The md5 gate", `.a4bin/md5.sh`): greedy `axon gen -n 64` on 11 (model, device,
   kv) configs (0.5B, Llama 1B, 7B, Llama 3.1 8B, Qwen3 0.6B, gemma3 1B; cpu, cuda, cuda `--kv q8`) × 3 prompts.
   `gen` decodes at nb 1, so decode never reaches (i). The final gate adds prompt P2 (~70 tokens), which prefills in
   one pass inside `bgemm_g2x`'s 17..128 range. That runs the fused launch on the Q4_K-q/k models (Llama 1B, 7B,
   8B). P1 (6 tokens) and P3 (~700) do not reach it. Tests 1, 2 and 4 remain the proof for decode batch sizes.
7. **Serve identity:** `serve_load.py texts` at `--c 1` and `--c 8`, against main's `--c 1` digest. Its 16 requests
   keep nb ≤ 16 (gemvmx), so it doesn't reach the new code either.

`NUMERICS_VERSION_EXACT` is not bumped. If any of tests 1–7 shows one differing bit, the change is a bug, not a
numerics change, and does not ship.

## 6. Plan

- **Done (CP1 → PR):** (i) at nb 17..128, in both tile layouts (gemm2x: every tile is gemm2s's; gemm2xt: q's tiles are gemm2t's), as one
  launch, Q4_K only, on by default, off with `AXON_QKV_FUSE=0`. (ii) is built, micro-benched, dropped and removed (§7).
- **Not done:** (ii-b) Q6_K is never started (M2). The full release, debug and scalar suites are not run on the final head. The serve
  identity gate (test 7) is not run either: it can't reach (i) at nb ≤ 16.
- **Next, by §7's data:** A6 (iii)'s host overlap has at most ≈ 0.1 % of a graphed pool step to win inside
  `step_checked`. A6's case rests on the serve loop's between-step host time and on feed passes (A6 (i)/(ii)), which
  `pool_step_cost` doesn't see. One lead from the micro-bench, not built: gemm2t runs the 7B's `down` 21 % faster than
  gemm2s at nb 32 in isolation, with the same bits (§7). `TILE_T_MIN_NB` = 64 keeps gemm2s there.
- Pods (4090, the H100 kill rule) only if the lead books them.

## 7. Results (bella, `05b48b6`'s code paths; MEASURED)

### 7.1 M2 micro-bench: (ii) dropped

Job `1030-130922-c603` (CP1's `ab.sh`, step 1), `cuda_gemm_tile_bench`, `AXON_BENCH_SHAPE=down`, the 7B's Q4_K down
(3584 × 18944), 7 rounds, median µs per launch. The job was TAINTED: foreign GPU load mean 7.1 %, max 39 %
(zen-bin). These are same-job ratios, and the margins are far outside any drift.

| nb | gemm2s (today's tile) | gemm2t | gemm2r (32 rows, 3 st) | gemm2r2 (32 rows, 2 st) | best r32 / gemm2s |
|---|---|---|---|---|---|
| 32 | 183.8 | 145.0 | 203.8 | 195.9 | **1.066** |
| 64 | 196.8 | 162.2 | 223.9 | 227.2 | **1.138** |

The rule required ≤ 0.95 at both nb, and neither variant qualifies. By the pre-registered rule (§4), (ii) is dropped with no
A/B and removed (`05b48b6`), and (ii-b) never starts. This is shady's point, measured: a 32-row CTA stages the same activation
tile for half the rows. Registers: gemm2r 132, gemm2r2 128, gemm2s 185, no spills.

### 7.2 The pool A/B: (i)

`c603`'s step 2 (rotated M A B C) is not used. Its first round was cold, and main-vs-main swung 35 %. The lead reran (i) alone
with `.a4bin/ab2.sh` (job `1031-045802-1654`, `--timing bella --low`): one discarded warm-up, then
`M A A M F M A A M F M A A M`. M is main `fde94b6`. A is the candidate (`7b32b3a`, defaults: (i) on, (ii) off; the same
code paths as `05b48b6`). F is the candidate with `AXON_QKV_FUSE=0`, an A/A control that must equal M. 7B Q4_K_M,
f16 KV, graphs on. Each cell is the median of 5 × 40-step windows inside one run. The table shows the median over runs (M 6, A 6, F 2).

**Taint, disclosed:** the job was flagged TAINTED by its max sample (Xorg 54 %, one screen wake). The mean foreign
load was 0.9 % over 188 samples. The lead accepted the run, because the in-job F control measures the noise floor directly.

| depth | nb | M ms/step | M spread (6 runs) | A ms/step | A vs M | F vs M | A, drift-paired per AA block |
|---|---|---|---|---|---|---|---|
| 600 | 16 | 18.709 | 5.4 % | 18.683 | −0.1 % | −0.1 % | +1.9, −0.0, −0.3 |
| 600 | 32 | 22.439 | 4.6 % | 21.922 | **−2.3 %** | +0.0 % | −1.7, −2.6, +0.1 |
| 600 | 64 | 26.792 | 7.3 % | 26.188 | −2.3 % | −0.7 % | −0.6, −2.3, −1.2 |
| 600 | 128 | 42.832 | 7.1 % | 41.873 | −2.2 % | +0.9 % | +0.3, −2.6, −2.6 |
| 256 | 16 | 17.604 | 3.9 % | 17.599 | −0.0 % | +0.4 % | +0.1, −0.0, −0.8 |
| 256 | 32 | 21.445 | 8.2 % | 21.086 | −1.7 % | +0.1 % | −1.6, −1.4, −2.6 |
| 256 | 64 | 24.084 | 6.8 % | 23.694 | −1.6 % | −0.2 % | −1.7, −1.6, −2.4 |
| 256 | 128 | 37.559 | 7.2 % | 36.791 | −2.0 % | +0.3 % | −2.1, −2.0, −1.1 |

"Drift-paired" is each `A A` pair's mean against the mean of the two Ms around it (blocks 1–4, 6–9, 11–14). M's spread
is mostly a monotonic drift over the 18-minute job: at nb 32 d600, M1 21.67 ms rises to M14 22.67 ms. The ABBA order is there
to cancel exactly that. F against its neighbouring Ms is +0.4 / 0.0 % at nb 32 d600, and within −0.8 to +2.9 % per block over
all cells.

**Ruling (lead, 10-31, final): (i) is INCONCLUSIVE.** The lead first accepted the run as a pass on the median. That
acceptance was withdrawn when the second clause below was pointed out.

1. The median passes: −2.3 % against the −1.8 % bar. But the gain is not more than 2 × the A/A spread of the cell (M's 4.6 %), and
   the drift-paired estimate is −1.4 %, under the bar. So the pre-registered rule neither passes nor kills (i).
2. (ii) is dropped by M2 (§7.1).
3. nb 128 is informational only. Under M3 it decides nothing, because M's spread is over 0.5 %.
4. A rerun is pre-registered in §8.

**The second clause, in detail.** §4's pass for (i) also asks for "more than 2 × the A/A spread of that cell".
Read literally, as M's max − min over its 6 runs, that spread is 4.6 %, and the clause fails. M's spread is a drift that the
design cancels. A drift-free A/A, F against its neighbouring Ms (≤ 0.4 % at that cell), meets the clause by a wide margin. On the
drift-paired statistic, nb 32 d600 is −1.4 % (one of three blocks is +0.1 %) and nb 32 d256 is −1.9 %. Over the 6 cells where (i) fires, the drift-paired mean is
≈ −1.6 %. The pattern is consistent: nb 16 (no fusion) gives 0.0 to −0.1 % by median, and every cell where it fires gives −1.6 to −2.3 %.

### 7.3 Host time per step (A4 vs A6)

The candidate's `HOST` lines, the same job (test-only instrumentation in `step_checked`, removed in `fe20964` with (i); it is
in `05b48b6` for A6 to reuse): the host time `step_checked` spends outside the GPU's work, from entry to the graph
launch's return plus its tail after the picks' read. Median over runs, ms/step:

| depth | nb 16 | nb 32 | nb 64 | nb 128 | step (A) |
|---|---|---|---|---|---|
| 600, A | 0.021 | 0.021 | 0.017 | 0.020 | 18.7–41.9 ms |
| 600, F | 0.016 | 0.018 | 0.019 | 0.019 | |
| 256, A | 0.018 | 0.017 | 0.022 | 0.021 | 17.6–36.8 ms |
| 256, F | 0.021 | 0.019 | 0.022 | 0.023 | |

So 0.016–0.023 ms of a 17.6–42.8 ms graphed step, ≤ 0.13 %. Inside the pool step, the host is not the cost. A6 (iii)'s
host overlap can only win host time spent outside `step_checked`: the serve loop's scheduling and row building between
steps. A6 (i)/(ii)'s feed passes are not seen here either, so A6 needs its own serve-level measurement. On this
data, A4 (i)'s −1.4 to −2.3 % is already larger than anything A6 (iii) could hide inside the step.

## 8. CP3: the rerun, pre-registered (10-31, before submission)

**Why.** §7.2 is inconclusive. M's raw spread was a thermal climb, so this run sets the start temperature of every
run and takes its noise term from the F control, not from M's spread. It also adds the gemm2t lead from §7.1.

**New switch, candidate only: `AXON_DOWN_TILE_T_MIN_NB`** (`BatchFns::down_t_min_nb`, default `TILE_T_MIN_NB` = 64,
which is main's routing).
- It is the most tokens a layer's Q4_K or Q5_K `down` keeps on gemm2s, whatever `tile_t`'s waves say (`tile_t_from`, `bgemm_at`).
- It routes only between gemm2s and gemm2t. Main already proves those bit-identical per output:
  `gpu_batched_products_match_gemv_bits` checks both `gemm2s_q4_k` and `gemm2t_q4_k` (`gemm2_names`) == the decode
  GEMV, bit for bit. This branch extends it to nb 17..129, including 32, 33, 48 and 64, at Q4_K widths up to the 7B's 18944, with all
  four epilogues.
- The new test `gpu_down_tile_t_matches_bgemm` checks the route itself. It compares `bgemm_at` with t_min 16 and 31 against
  main's `bgemm`, byte for byte, on the 7B's down and on a shape that takes gemm2t at any SM count. It covers mode 0 with bias and mode 1
  (the residual add), nb ∈ {17, 24, 31, 32, 33, 48, 63, 64}, and asserts the route actually moved.
- On bella the 7B's 3584-row `down` takes gemm2t at nb 32..64 once the floor is 31 (CODE `wave_cost`: 28 gemm2t blocks at 82 % beat 56 gemm2s).
  Q6_K `down` (14 of 28 layers) is untouched. ESTIMATED gain from §7.1: (183.8 − 145.0) µs × 14 layers = −0.54 ms
  of ≈ 21.9 ms at nb 32 d600, about −2.5 %, an upper bound from isolated launches.

**Arms, all in one `--timing bella --low` job** (`.a4bin/ab3.sh`, sha256 below):
- **M** is main `fde94b6`.
- **A** is the candidate with defaults (fused q|k|v).
- **T** is A plus `AXON_DOWN_TILE_T_MIN_NB=31`.
- **F** is the candidate with `AXON_QKV_FUSE=0`, the A/A control.

`pool_step_cost`, 7B Q4_K_M, f16 KV, graphs on, `AXON_STEP_DEPTHS=600,256`, `AXON_STEP_NB=16,32,64,128`.

**Order:** one discarded warm-up (M), then `M`, then 6 blocks `a b c M`. Each block is one permutation of A T F, mirrored in
pairs: ATF FTA, TFA AFT, FAT TAF. Each arm runs 6 times and sits in each in-block position twice. **Cooldown before every run:**
30 s, then wait until `nvidia-smi` reads ≤ 55 °C, with a cap of 300 s. A run that hits the cap is kept and logged `COOL CAPPED`, and
the start temperature of every run is reported.

**Statistic** (`.a4bin/ab3_stat.py`, sha256 below):
- **Paired delta:** each candidate run against the mean of its block's two M runs, d_X = X / mean(M_before, M_after) − 1. Per cell, the
  median over the arm's 6 runs.
- **T against A:** per block, d_TA = T / A − 1, the same block. The median over the 6 blocks.
- **Noise term:** S_F = max(d_F) − min(d_F), the range of F's 6 paired deltas at that cell. It replaces M's raw spread.
- **Bars at nb 32 d600:**
  - **A passes** if median d_A ≤ −1.8 % **and** |median d_A| > 2 S_F.
  - **T passes** (against A) if median d_TA ≤ −2.0 % **and** |median d_TA| > 2 S_F.
  - Otherwise it fails. Nothing is tuned and rerun to pass.
- **Regression cells:** nb 16 at both depths, where neither change fires. A median d_A or d_T beyond 2 S_F is reported as a regression.
- **nb 128:** informational. Other cells are reported but decide nothing.
- **Outcomes:**
  - If A passes, (i) ships on by default.
  - If A fails, #178 closes as a NO-GO for (i), with a docs entry.
  - T is a separate decision. If it passes, its default (31) goes into a follow-up PR with its own design paragraph. If it fails, the
    switch stays an experiment knob or is removed.

**Scripts, frozen at submission:** `ab3.sh` sha256 `a22043383c01c48f…` and `ab3_stat.py` sha256 `10146c962f37ca5a…`, both
attached to #178. The candidate binary is the `19c3e75` test build.

**Gates for the new code** (`axon-run --detach`, release and debug, the candidate head): `gpu_down_tile_t_matches_bgemm`,
`gpu_batched_products_match_gemv_bits`, `tile_t_picks`, and §5's tests 1–4. Plus the md5 gate against main at the candidate head, which
runs with the default routing.

### 8.1 Results (job `1031-065003-733e`, MEASURED)

Run as pre-registered: scripts unchanged (sha256 as above), `--timing bella --low`, candidate `19c3e75`. Every run started
at 54–55 °C after a 60–156 s wait; none hit the cap. **Taint, disclosed:** flagged TAINTED by one Xorg sample at 41 % (mean
foreign load 1.2 % over 656 samples). The lead accepted the run under the pre-registered rule, whose noise term is the
in-job F control, so the taint is already in S_F.

Median ms/step over runs (M 7, A T F 6 each), and `ab3_stat.py`'s paired medians:

| depth | nb | M | A | T | F | d_A | d_T | d_TA (T vs A) | d_F | S_F |
|---|---|---|---|---|---|---|---|---|---|---|
| 600 | 16 | 18.106 | 18.129 | 18.107 | 18.111 | +0.01 % | −0.07 % | −0.11 % | −0.04 % | 0.31 % |
| 600 | **32** | 21.615 | 21.237 | 20.709 | 21.768 | **−1.80 %** | −4.17 % | **−2.47 %** | +0.64 % | **0.96 %** |
| 600 | 64 | 25.961 | 25.366 | 24.713 | 25.989 | −0.84 % | −3.23 % | −2.53 % | +1.53 % | 6.30 % |
| 600 | 128 | 40.787 | 39.662 | 39.617 | 40.759 | −2.44 % | −2.74 % | −0.10 % | +0.16 % | 1.29 % |
| 256 | 16 | 17.425 | 17.412 | 17.391 | 17.408 | −0.05 % | −0.19 % | −0.16 % | −0.06 % | 2.03 % |
| 256 | 32 | 20.422 | 20.029 | 19.511 | 20.399 | −1.90 % | −4.51 % | −2.62 % | −0.08 % | 0.71 % |
| 256 | 64 | 23.105 | 22.337 | 21.834 | 23.108 | −3.24 % | −5.37 % | −2.29 % | +0.03 % | 0.48 % |
| 256 | 128 | 35.659 | 35.115 | 35.085 | 35.565 | −1.54 % | −1.57 % | −0.09 % | −0.22 % | 0.57 % |

- **A at nb 32 d600: FAIL.** −1.80 % meets the −1.8 % bar but not the noise clause (|d_A| > 2 S_F = 1.93 %).
- **T against A at nb 32 d600: PASS.** −2.47 %, beyond both −2.0 % and 2 S_F.
- **Regression cells (nb 16):** none; every median is within 2 S_F of 0.
- The floor moves only nb 32..64, and the data agree. T against A is −2.3 to −2.6 % at nb 32 and 64, and −0.1 % at nb 16 and
  128, where T and A run the same kernels.

### 8.2 Ruling (lead, 10-31) and what ships

The pre-registered outcomes said a failed A closes #178 and a passing T goes to a follow-up PR. The lead ruled under CP0's
keep-one-part rule instead:

1. (i) is removed in a new commit (`fe20964`): `gemm2x_q4_k`, `gemm2xt_q4_k`, gemm2_t's `yt`, `bgemm_g2x`,
   `AXON_QKV_FUSE`, and `gpu_qkv_one_launch_matches_three`. There is no default-off code. The tests that protect existing
   routes stay: the extended nb lists of `gpu_batched_products_match_gemv_bits` and `gpu_products_match_reference_bits`, and
   `pool_steps_past_gemvm16_match_solo` up to 128 rows.
2. The floor ships as the default in the same PR: `DOWN_T_MIN_NB` = 31, with `AXON_DOWN_TILE_T_MIN_NB` as the override
   (64 restores the old routing).
3. **Assumption, stated:** T was measured on top of A. The shipped head has the floor without the fusion. The two touch
   different products (q, k, v against `down`) and different launches, so the floor's gain is taken as T against A,
   −2.5 % at nb 32 d600 (−2.6 % at d256, −2.3 to −2.5 % at nb 64), not T against M (−4.2 %). This is unmeasured on the
   final head; the md5 gate there checks bits, not speed.

### 8.3 Route safety on other GPUs (CODE `tile_t_from`, `wave_cost`)

At nb ≤ 64 there is one token tile, so the route depends only on (rows, SMs), not on nb. The question is which `down`
row counts (= hidden size) the waves send to gemm2t once the floor is 31. Swept over hidden sizes 128..8192 (every
multiple of 128, which covers every Qwen2.5, Qwen3, Llama 3, Mistral and Gemma 3 dense model):

| GPU (SMs) | downs moved to gemm2t at nb 32..64 |
|---|---|
| RTX 4090, sm_89 (128) | **none** (the first is 8320 rows) |
| H100, sm_90 (132) | **none** (the first is 8576). Q4_K at nb ≤ 64 isn't on gemmh2 there (`gemmh_plain` needs ≥ 132 blocks; the 7B's down has 28), so it takes the same `tile_t_from` path. |
| RTX 5090, sm_120 (170) | **none** (the first is 11008) |
| bella, 5070 Ti Laptop (46) | 3072 and up (Llama 3B, the 7B and 8B, Qwen 14B/32B, Gemma 12B/27B). The 7B is measured; the others are not. |
| 20 / 58 / 76 / 84 SMs | 1408 / 3840 / 4992 / 5504 and up (every multiple of 128 from there to 8192). Unmeasured. |

`down_tile_t_floor_routes` (CPU test) pins the 4090, H100 and 5090 rows of the table, and bella's 7B move. Routes don't
change on a 4090 or an H100, so the lead's rule needs no `--timing 4090` job, and nothing changes there to measure.

**Unmeasured cards, OPINION:** the smaller cards' moves come from the same wave model, which predicted this one well.
For the 7B's down on 46 SMs it gives gemm2t/gemm2s = 0.82, and the micro-bench measured 0.79 at nb 32 and 0.82 at nb 64.
The micro-bench (`1030-130922-c603`) was TAINTED: foreign GPU load mean 7.1 %, max 39 %. Its kernels ran
interleaved, in a shuffled order each round, and the per-round ranges don't overlap. Even the slowest gemm2t round beats
the fastest gemm2s round (0.82 at nb 32, 0.89 at nb 64), so the ratios are taken to hold. The step gain itself was
measured separately, in the pool A/B (`1031-065003-733e`).
`TILE_T_TIME` itself was fitted on a 20-SM RTX 4050. The floor is not restricted by SM count. If a smaller card regresses,
`AXON_DOWN_TILE_T_MIN_NB=64` is the switch back.

**Known modelling gap:** `gemm_cost` (the prefill wave planner, passes > 32 tokens) still prices every Q4_K product with the
64 floor. That includes `down` in a pass of 33..64 tokens, which now runs gemm2t on bella. It only steers how a long prompt is
split into passes, and its passes are at least a quarter of the batch max (288 tokens at 1152). Bits are unaffected.

### 8.4 Final-head gates (`fe20964`)

Bit tests (release and debug), the md5 gate against main (11 configs × 4 prompts: P4, about 53 tokens, prefills in one pass of
33..64 tokens, so the moved route runs on the 7B and the 8B), and the full release and debug suites, all through
`axon-run --detach` on bella. The jobs and results are in #178's §6.

