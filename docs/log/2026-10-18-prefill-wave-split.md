# Prompt passes sized to the GEMM waves (`AXON_WAVE_SPLIT`, r15 pick 1)

- Date: 2026-10-18. Branch: feat/prefill-wave-split (off main 50cd609; main merged in at 711cbce and b90d808, e1ce382).
- PR #92. Spec: r15-4090-prefill.md pick 1 (Yoru). Review: Shady sa_513 (SHIP-AFTER-FIXES 6.5/10). M1 and S2–S4 are
  fixed in 2e83f5a.
- Host-only change: no kernel change, no numerics change, no `NUMERICS_VERSION` bump.
- **Scope: `gen` / `logits` only.** `serve` and `chat` feed the prompt to the net in `prefill_batch` chunks. Each
  chunk is one pass, so they see no change. The follow-up (stacked PR #97, feat/prefill-wave-split-serve) hands them the
  whole un-fed prompt.

## What changed
- **`batch::wave_passes`** (`prompt_passes`, for n > the batch max and start + n < `FA_LONG`) chooses the pass
  lengths with the least modelled GEMM time, under these rules:
  - The pass count is `n.div_ceil(max)`, as before.
  - No pass is longer than max, so the scratch and the memory plan are unchanged.
  - No pass is shorter than `wave_floor` = max / 4 (288 tokens at 1152).
  - Every pass but the last is a whole number of 64-token tiles.
  - A split must model ≥ 3 % less time than the equal split (`WAVE_MARGIN` 0.97, the same as `FA_MARGIN`).
    Otherwise the equal split stays.
- **`batch::gemm_cost`** sums, over the products, waves × a block's MACs. It uses `bgemm`'s grids:
  - gemm2s, 64 × 64 (rows × tokens);
  - sm_90a gemmh2, 128 × 64, where `gemmh_plain` picks it;
  - register-staged types: 64 × 128 at ≥ 1.5 blocks per SM, else 64 × 64;
  - fused gate + up: 32 rows of each × 64 (gemmh2: 64).

  All of these run 1 block per SM. A partial last wave costs `PARTIAL_WAVE` = 30 % of a full wave plus 70 % × the
  fraction of SMs it fills (see "Model calibration" below).
- **`CudaQwen2::wave_gemms`** lists the real per-layer products, including `ffn_half`'s fuse rule. It returns None
  without the int8 GEMMs (sm < 80) or for MoE models, and then the old split stays.
- The SM count comes from `cuDeviceGetAttribute`. Nothing is hardcoded.
- `AXON_WAVE_SPLIT=0` gives the old equal split, with no rebuild. `AXON_PASS_TRACE=1` prints each prompt's split.
  Both are read when the model is built.

## Why
r15 §4 pick 1. On 128 SMs, 2100 = 1050 + 1050 makes 17 token tiles per pass:

| products | blocks | waves of work | waves run |
|---|---:|---:|---:|
| q / wo / down | 952 | 7.44 | 8 |
| k / v | 136 | 1.06 | 2 |
| gate + up | 10,064 | 78.6 | 79 |

1088 + 1012 is 17 full tiles plus 16 tiles, which gives exactly 7, 1 and 74 waves.

## Measured: RTX 4090 (MEASURED, Case, 2026-10-18 21:01–21:13 EDT)
- **Host:** RunPod secure pod1, RTX 4090 (128 SMs), driver 595.91.07, CUDA 13.2, EPYC 7452.
- **Binary:** `axon-0b2f029` for both arms. `AXON_WAVE_SPLIT=0` is the old split.
- **Method:** `ab_wave.py`, `gen` prompt-eval tok/s at exact token counts. 7 rounds interleaved, order alternating,
  3 s cooldown. Medians (min–max), n = 7 per arm.
- **Raw data:** pod4090-92.log, pod4090-92-ab-wave.jsonl, pod4090-wave.log.

| model | pp | old (equal split) | new (wave split) | Δ | splits |
|---|---|---|---|---|---|
| Qwen2.5-7B | 2100 | 13255.5 (13180–13289) | 13656.6 (13617–13733) | **+3.03 %** | [1050,1050] → [1088,1012] |
| Qwen2.5-7B | 1153 | 12876.2 | 12825.9 | −0.39 % | [577,576] → [704,449] |
| Qwen2.5-7B | 1777 | 13357.2 | 13590.5 | +1.75 % | [889,888] → [1024,753] |
| Qwen2.5-7B | 3000 | 13276.3 | 13428.2 | +1.14 % | [1000×3] → [1152,1152,696] |
| Qwen3-8B | 2100 | 12184.3 (12057–12209) | 12653.6 (12546–12782) | **+3.85 %** | [1050,1050] → [1088,1012] |
| Qwen3-8B | 1153 | 11949.0 | 12140.9 | +1.61 % | [577,576] → [704,449] |
| Qwen3-8B | 1777 | 12643.9 | 12728.6 | +0.67 % | [889,888] → [1024,753] |
| Qwen3-8B | 3000 | 12409.4 | 12511.4 | +0.82 % | [1000×3] → [1024,1024,952] |

- At pp2100 the two arms' ranges don't overlap, for both models.
- pp3000 is under 1.5 % for both models, as the model predicted.
- **On the 4090, 1777 is not a control:** the two arms split it differently.
- The pod split every prompt exactly as `wave_passes_shapes` predicted.

**r15 kill A/B** (`cuda_prefill_profile`, `AXON_PROF_LOOP=20`): the sum of the two pass medians, 5 rounds
interleaved (pod4090-waveprof.log).

| model | 1050 + 1050 | 1088 + 1012 | Δ | per round | spec expects |
|---|---|---|---|---|---|
| 7B | 158.30 ms | 153.99 ms | **−2.72 %** | 2.51 / 2.69 / 1.82 / 2.76 / 3.54 | ≥ 2.5 % |
| Qwen3-8B | 173.02 ms | 166.24 ms | **−3.92 %** | 4.12 / 3.92 / 3.90 / 3.97 / 4.48 | ≥ 3.5 % |

The kill rule is ≥ 1.5 % at pp2100. **PASS.** r15 estimated +3.7 % (7B) and +5 % (8B); the measurement is +3.0 % and
+3.9 %.

**bella (1018-163243-7684): run, discarded.** The controls drifted −5.4 % and the ranges were ~40 % wide (the GPU
was under desktop load). Its numbers aren't used.

**Unmeasured on H100 and 5090.** Their rows below are modelled only.

## Model calibration and the pass floor (review M1, 2e83f5a)
The first model was r15's, where a partial wave costs a whole one, with a 1 % margin. It overrated uneven splits: the
4090's 7B pp1153 split modelled −3.8 % of GEMM time and measured −0.39 % end to end. It also chose degenerate tails
on unmeasured GPUs: [1088, 65] (H100 SXM 7B pp1153), [64, 1089] (bella), and [1152 ×5, 240] (5090 pp6000).

**Partial-wave cost.** I fitted it to the 8 e2e cells above. Predicted gain = 0.87 (r15's GEMM share) × modelled
GEMM change, with a partial wave costing α + (1 − α) × its fill:

| α (PARTIAL_WAVE) | rms error (points) |
|---:|---:|
| 1.0 (r15) | 1.82 |
| 0.6 | 0.97 |
| 0.3 | **0.63** |
| 0 (tile padding only) | 0.99 |

α = 0.3 is used (fit/fit.py in the scratch dir).

**No per-pass constant.** The pass count is fixed, so a constant cost per pass adds the same amount to every split
and can't change the choice. Short passes do pay for their latency-bound kernels (norms, quantize, RoPE, attention:
time ≈ max(latency, nb × rate)). That is a cost of passes below some length, so a floor models it:
`wave_floor` = max / 4. The shortest pass ever measured was 449 tokens.

**Margin 0.97.** The fitted model's error is ~0.6 points, and its worst cell (7B pp1153) is off by ~1.6 points.

**Before → after**, modelled GEMM change at batch max 1152 ("=" = the equal split stays). Full table:
`wave_passes_shapes` output.

| GPU (SMs) | 7B pp1153 | 7B pp2100 | 7B pp3000 | 7B pp6000 | 8B pp1153 | 8B pp2100 | 8B pp3000 | 8B pp6000 |
|---|---|---|---|---|---|---|---|---|
| RTX 4090 (128) | [704,449] −3.8 → = | [1088,1012] −4.1 → **[1088,1012] −3.6** | −1.1 → = | [..,496] −1.5 → = | [704,449] −4.2 → = | [1088,1012] −5.6 → **[1088,1012] −4.0** | −1.1 → = | −1.7 → = |
| RTX 5090 (170) | −2.1 → = | [1152,948] −4.1 → [1152,948] −3.6 | [1152,1152,696] −5.8 → same −3.0 | [..,240] −5.9 → [..,1024,368] −3.3 | = | [1152,948] −4.1 → −3.5 | −6.0 → −3.2 | [..,240] −7.6 → [..,768,624] −3.9 |
| H100 SXM (132) | **[1088,65]** −6.2 → = | [1088,1012] −2.9 → −3.2 | −3.7 → = | −5.5 → [1152,1152,1024,896,896,880] −3.2 | [704,449] −11.1 → −3.7 | [1088,1012] −6.6 → −4.4 | −1.4 → = | −1.4 → = |
| H100 PCIe (114) | [704,449] −7.6 → = | [1152,948] −6.8 → −4.4 | −1.4 → = | −1.4 → = | [832,321] −1.5 → = | [1088,1012] −3.0 → −3.2 | −3.3 → = | −3.3 → = |
| bella (46) | **[64,1089]** −2.0 → = | [1088,1012] −3.1 → −3.3 | −1.8 → = | [..,240] −2.4 → = | = | [1088,1012] −2.8 → −3.2 | −2.5 → = | −2.6 → = |
| avante (20) | = | [1152,948] −3.5 → −3.4 | −1.8 → = | −2.1 → = | = | [1152,948] −3.6 → −3.4 | −2.2 → = | −2.4 → = |

- The pp2100 picks survive everywhere, including the 4090's 1088 + 1012.
- No degenerate tails remain.
- On the 4090, the 7B cells at pp1153, 1777 and 3000 (measured −0.4 / +1.8 / +1.1 %) are now under the margin.
- Percentages are relative to each model's own baseline. The new baseline is lower, so some rows show a larger change
  than before.

## Bits
Pass boundaries are outside the identity (determinism.md). The switch is default-on, so every existing
batched-prefill bit test runs the new split wherever it applies. That includes `prefill_bits` and
`gpu_prefill_logits_match_forward_big`, which compare against per-token decode.

`gpu_prefill_wave_split_bits` checks that the wave split gives the same bits as the equal split (KV bytes and logits):
- at 6 fixed lengths, plus the first 2 lengths up to 4600 whose split differs on the GPU under test;
- for one of those, again from position 37 (start > 0);
- against per-token decode at 1777 tokens on 0.5B.

It flips the `wave_split` field, not the environment variable. It fails as vacuous if no length up to 4600 splits
differently on the GPU it runs on.

`wave_passes_shapes` checks the invariants, the floor and the margin, that max 64 and max 63 return None, and pins
the 4090 cells.

## Gates
**At 0b2f029.** md5 runs `gen` vs main on 6 models × 5 prompts, including prompts of ~2100 and ~1780 tokens that take
more than one pass.

| gate | job | result |
|---|---|---|
| release suite, shard 2/2 | 1018-163238-a844 | PASS |
| release suite, shard 1/2 (avante) | 1018-163238-d187 | FAIL on `serve::tests::serve_api_key` only: the /metrics race fixed by #93 |
| d187's shard rerun at 711cbce (main merged) | 1018-193811-9a1a | PASS |
| `serve::` debug at 711cbce | 1018-193812-0bea | PASS |
| debug suite | ee59, 033e | PASS |
| big-model bits on bella | 1018-163242-49f2 | PASS |
| md5 vs main 50cd609 | 1018-163243-9a15 | PASS |

**At 2e83f5a** (main e1ce382 merged; host-only change, so the minimal set): ALL PASS

| gate | job | result |
|---|---|---|
| bit tests (`gpu_prefill bits wave_passes passes_ gpu_q8_ cuda::qwen2::moe::`), release | 1018-221436-9e33 | PASS, 52/0 (avante) |
| same, debug | 1018-221437-12e0 | PASS, 52/0 |
| big-model bits on bella (`AXON_BIG_MODELS=1`) | 1018-221437-c3f0 | PASS, 2/0 (bella, 46 SMs; wave split == equal on 0.5B/1.5B/7B/8B, incl. start 37) |
| md5 vs main e1ce382 | 1018-221437-1aa6 | PASS, 0 of 30 differ, 60/60 rc 0 (cand ran wave splits e.g. [1088, 1013] on the ~2100-token prompts) |
| clippy (bin + tests), rustfmt | local | PASS |

**4090 re-check of 2e83f5a (MEASURED, Case, RunPod 4090, `axon-2e83f5a` both arms, ab_wave.py 5 rounds
interleaved, medians; pod4090-2e83f5a-ab.log): PASS.**

| model | pp | old (equal) | new (wave) | Δ | splits |
|---|---|---|---|---|---|
| Qwen2.5-7B | 2100 | 13286.8 (13101–13359) | 13638.0 (13528–13695) | **+2.64 %** | [1050,1050] → [1088,1012] |
| Qwen2.5-7B | 1153 | 12872.7 (12818–12880) | 12834.1 (12710–12860) | −0.30 % | same split (control) |
| Qwen2.5-7B | 3000 | 13339.7 (13253–13359) | 13357.4 (13316–13368) | +0.13 % | same split (control) |
| Qwen2.5-7B | 6000 | 12559.3 (12549–12601) | 12532.1 (12520–12581) | −0.22 % | same split (control) |
| Qwen3-8B | 2100 | 12190.0 (12170–12245) | 12687.3 (12641–12857) | **+4.08 %** | [1050,1050] → [1088,1012] |
| Qwen3-8B | 1153 | 11922.9 (11791–11960) | 11929.8 (11807–11968) | +0.06 % | same split (control) |
| Qwen3-8B | 3000 | 12405.8 (12358–12425) | 12405.9 (12317–12474) | +0.00 % | same split (control) |
| Qwen3-8B | 6000 | 11548.2 (11521–11576) | 11551.0 (11520–11567) | +0.02 % | same split (control) |

As predicted: the pp2100 win holds, and every cell the fitted model now declines runs the same split in both arms
(controls within ±0.3 %).

gates: ALL PASS at 2e83f5a
