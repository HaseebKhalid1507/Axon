# attn_blk_nb: batched decode attention for the serve pool (nb ≥ 32), exact (T1, PR #151)

**Status: the pre-registered 4090 kill rule FAILED as written (×1.23 against a ≥ 1.5× bar).** Following the
plan, the decision passes to the other pre-registered criterion, the H100 confirm. If that misses too, the line is
killed and moves to experiments-rejected. Nothing is merged on the 4090 result.

## What changed

- `prefill.cu attn_blk_nb_*` (f16, q8, q8g32; hd 64 / 128). Grid (n_kv, S, rows) × 4 warps.
  - Each CTA streams one row's 64-position blocks in order through 2 cp.async stages.
  - Per block it runs `attn_blk_t`'s NW = 4 code.
  - S = 1: the CTA runs `attn_cmb_dev`'s fold itself, then `quantize_b`. There are no partials.
  - S > 1: `attn_blk_b`'s partials, then `attn_cmb_b`.
- `batch.rs`: pool steps with ≥ 32 rows on tensor cores take it.
  - `nb_split` picks S (`AXON_ATT_NB_S` forces it).
  - `AXON_ATT_NB=0` restores `attn_blk_b`.
  - nb ≤ 16 is untouched.
- The bit-identity argument is in the PR body and the kernel comment.

## Why

- The H100 anatomy (#125) found the pool step's attention dominating at depth: 18.7 of 32.6 ms at 7B nb 128 d4096, at 48 % of DRAM peak.
- `attn_blk_b` is a one-warp CTA with 32 KB of static smem (3 warps/SM on sm_89), 16 rows per launch, and serial load → mma.

## Verification (all MEASURED)

| gate | job | result |
|---|---|---|
| `cuda_attn_nb_exact` (act bits vs `attn_blk_b`+`attn_cmb_b`; S 1/2/3/7, f16/q8/q8g32, hd 64/128, plain + paged, ragged depths), `pool_wide_long_steps_match_solo` (40 rows, f16 + q8g32), `pool_steps_past_gemvm16_match_solo` | 1026-212457-46f0 (pod1, sm_89) | 3 passed |
| fast tier, release / debug | 1026-212457-209b / 1026-212457-95ab | rc 0 / rc 0 |
| gen md5 (0.5B / 7B / Qwen3-8B × f16 / q8 × 3 prompts) + serve pool md5 (48 concurrent greedy requests; base / cand / cand `AXON_ATT_NB=0`; 0.5B, Qwen3-0.6B, 7B × f16 / q8) | 1026-212457-608a (bella) | gen 0 of 18 differ; serve 0 differing of 288 requests |

## Numbers

**Kernel**, clean RTX 4090 (pod1, job 1026-221123-4546), `cuda_attn_pool_bench`, 7B geometry, us per layer:

| kv | depth | nb | old attn_blk_b + cmb | old GB/s | new | new GB/s | speedup |
|---|---|---|---|---|---|---|---|
| f16 | 600 | 32 | 32.8 | 1202* | 19.5 | 2024* | ×1.68 |
| f16 | 600 | 64 | 101.4 | 777 | 52.2 | 1508* | ×1.94 |
| f16 | 600 | 128 | 238.6 | 660 | 184.4 | 854 | ×1.29 |
| f16 | 4096 | 32 | 354.3 | 758 | 292.1 | 919 | ×1.21 |
| f16 | 4096 | 64 | 708.3 | 758 | 583.8 | 920 | ×1.21 |
| **f16** | **4096** | **128** | **1415.2** | **759** | **1153.0** | **931** | **×1.23 (kill cell: FAIL, bar ×1.5)** |
| q8 | 600 | 32 / 64 / 128 | 55.3 / 108.5 / 260.1 | 312–374 | 27.6 / 50.2 / 100.4 | 735–810 | ×2.00 / 2.16 / 2.59 |
| q8 | 4096 | 32 / 64 / 128 | 303.1 / 601.1 / 1205.5 | 457–461 | 170.8 / 326.7 / 626.7 | 811–884 | ×1.77 / 1.84 / 1.92 |

\* Above the 4090's ~1008 GB/s: those working sets (39–79 MB) are partly L2-resident (72 MB of L2).

- **Post hoc, not part of the pre-registered rule:**
  - At d4096 the new kernel streams 931 GB/s, ~92 % of the 4090's ~1008 GB/s. The old one was already at 759 GB/s (75 %).
  - So on this proxy the ceiling for any kernel was ~1.33×, and 1.5× was physically unreachable.
  - The 1.5× bar came from the H100's 48 %-of-peak figure, which doesn't carry over to a 4090.
  - We record this as an explanation, not a waiver.
- q8 shows the most gain: the old kernel was latency- or dequant-bound at 312–461 GB/s. In the new one, 4 warps share each block's dequant while the next block's copy is in flight.

**Step**: `pool_step_cost` 7B on the same 4090, graphs on, ms per step, AXON_ATT_NB 0 → 1, two interleaved rounds:

| depth | nb | NB=0 | NB=1 | Δ |
|---|---|---|---|---|
| 600 | 32 | 9.658 / 9.693 | 9.387 / 9.416 | −2.9 % |
| 600 | 64 | 11.980 / 12.033 | 11.439 / 11.465 | −4.6 % |
| 600 | 128 | 17.285 / 17.348 | 16.018 / 16.063 | −7.4 % |
| 4096 | 32 | 17.863 / 17.838 | 16.241 / 16.258 | −9.0 % |
| 4096 | 64 | 28.428 / 28.438 | 25.082 / 25.114 | −11.8 % |

d4096 nb 128 doesn't fit in 24 GB: the harness halved the pool to 64.

**No-regression A/B** (perf_ab, 5 rounds, same job), cand / base:

| model | tg128 | pp512 | pp2100 |
|---|---|---|---|
| 0.5B | ×0.998 | ×1.009 | ×0.993 |
| 7B | ×1.004 | ×0.996 | ×0.999 |

All within ±1 %. nb ≤ 16 runs unchanged code.

**Baseline on avante** (RTX 4050 Laptop, job 1026-211017-82d0):
- "both" exceeds attn + cmb measured alone (d4096 nb 128: 8173 vs 7709 us).
- Our reading, not measured: the combine measured alone rereads one launch's 15 MB of partials while they're L2-hot. In the real interleaving, the KV stream has evicted them.
- That partials round trip is what S = 1 removes.

## Next: H100 confirm (pre-registered; decides the line)

- `pool_step_cost` 7B nb 128 d4096 ≤ 24 ms and d600 ≤ 14.5 ms.
- Mix C = 128 aggregate ≥ +10 %.
- Both with AXON_ATT_NB = 0 / 1 interleaved in one binary.
- Same session: an `nb_split` S sweep at nb 32 (128 CTAs < 132 SMs).
