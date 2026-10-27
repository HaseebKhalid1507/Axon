# attn_blk_nb: batched decode attention for the serve pool (nb ≥ 32), exact (T1, PR #151)

**Verdict: KILL RULE MISSED. The 4090 clause failed (×1.23 against ≥ ×1.5), and so did all three H100 clauses
(below). Ruling (Fable, read-only referee, relayed by the lead): MERGE, default on, with the miss recorded.** The
principle: "a kill rule bounds spend; the contract bounds merge". The kernel keeps the bit-identity contract (every
bit gate passes) and has no measured regression.

Recorded as the ruling asks:
1. The 4090 clause (×1.23) missed its "else stop", and the H100 session was booked anyway. That was the lead's
   override of the pick's "don't book an H100 before a 4090/bella proxy has passed its kill rule", and it cost
   ~$0.93.
2. The mix C = 128 "+10 %" bar was a spec error: the mix is prefill-bound (ITL p50 ≈ 57 ms vs a ~15 ms decode step,
   finding 2 below), so a decode-attention change couldn't move it by 10 %. Future rules should gate on the term
   the change actually affects.
3. No T1 follow-ups (narrow-product tiles, q8 re-measure, the `nb_split` fix from finding 1) without a fresh pick.

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

## H100 confirm (pre-registered; MISSED)

- Job 1026-225032-3dfd: H100 80GB HBM3 SXM (132 SMs), secure, pod1. 480 s of stages.
- Session cost **$0.933** (axon-pods ledger: booked 22:50, down 23:04).
- One binary throughout (src == f797eaf), with AXON_ATT_NB 0 / 1 interleaved.

`pool_step_cost` 7B, ms per step (graphs), two rounds:

| depth | nb | NB=0 | NB=1 | Δ | pre-registered bar | verdict |
|---|---|---|---|---|---|---|
| 600 | 32 | 9.595 / 9.655 | 9.692 / 9.682 | +0.6 % | | |
| 600 | 64 | 11.533 / 11.594 | 11.093 / 11.087 | −4.1 % | | |
| 600 | 128 | 15.712 / 15.794 | 14.551 / 14.544 | −7.6 % | ≤ 14.5 | **MISS by 0.05** |
| 4096 | 32 | 13.184 / 13.187 | 13.287 / 13.282 | +0.8 % | | |
| 4096 | 64 | 18.494 / 18.497 | 15.296 / 15.292 | −17.3 % | | |
| 4096 | 128 | 29.247 / 29.305 | 24.489 / 24.492 | −16.3 % | ≤ 24.0 | **MISS by 0.49** |

Mix C = 128 (sd_client, the showdown's prompt set and SLO), rounds ordered NB 0, 1, 1, 0:

| cell | agg tok/s | goodput | slo_ok | ITL p50 ms | TTFT p50 ms |
|---|---|---|---|---|---|
| NB=0 r1 | 1802.8 | 329.2 | 0.188 | 56.86 | 3150 |
| NB=1 r1 | 1819.1 | 225.9 | 0.133 | 56.64 | 3241 |
| NB=1 r2 | 1837.9 | 253.8 | 0.133 | 56.75 | 3064 |
| NB=0 r2 | 1820.3 | 148.9 | 0.086 | 57.20 | 3044 |

- Agg median: 1811.6 → 1828.5, **×1.009 against a bar of ≥ ×1.10: MISS.**
- Goodput and slo_ok swing between rounds of the same arm (148.9–329.2), too much to read a difference from.

**Baselines moved.** The bars were derived from #125's anatomy numbers: nb 128 32.6 ms at d4096 and 19.5 ms at d600
(events, no graphs), and the ~18.6 ms showdown step. This session's NB=0 baseline on the same harness is 29.3 / 15.7 ms.

**Finding 1 (MEASURED): `nb_split`'s heuristic is wrong on Hopper at nb 32.** It takes S > 1 when CTAs < SMs (128 <
132). Kernel speedup over attn_blk_b + cmb by forced S (`AXON_ATT_NB_S`), us per layer in parentheses:

| kv | depth | S=1 | S=2 | S=3 | S=5 |
|---|---|---|---|---|---|
| f16 | 600 | **×1.40** (25.2) | ×0.92 | ×0.92 | ×0.98 |
| f16 | 4096 | **×1.43** (110.2) | ×0.99 | ×0.99 | ×0.97 |
| q8 | 600 | **×1.10** (37.3) | ×0.83 | ×0.85 | ×1.04 |
| q8 | 4096 | ×0.94 (186.8) | ×0.76 | ×0.91 | **×1.15** (152.3) |

- On f16, S = 1 wins at both depths.
- At step level (`pool_step_cost` nb 32, NB=1, S forced to 1):
  - d600: 9.366 ms against NB=0's 9.595 / 9.655.
  - d4096: 11.908 ms against NB=0's 13.18, about −10 %.
- With the default S, NB=1 is +0.6 % / +0.8 % at nb 32. The S choice explains the default's flat nb 32 cells.
- Code is not changed here.

**Finding 2 (MEASURED): mix C = 128 isn't decode-bound.**
- ITL p50 is ≈ 57 ms on both arms.
- A pure decode step at nb 128 is 14.5–15.8 ms (d600).
- So most of each serve step goes to prefill chunks under the step budget, and the attention kernel barely moves the mix aggregate.

Kernel level on the H100 (`cuda_attn_pool_bench` default grid, same job; old attn_blk_b + cmb → new, us per layer):

| kv | depth | nb 32 (default S = 3) | nb 64 | nb 128 |
|---|---|---|---|---|
| f16 | 600 | 34.6 → 35.9 (×0.96) | 73.8 → 38.2 (×1.93) | 145.5 → 75.5 (×1.93) |
| f16 | 4096 | 159.3 → 163.2 (×0.98) | 315.2 → 192.7 (×1.64) | 631.0 → 445.2 (×1.42; 1702 → 2413 GB/s) |
| q8 | 600 | 39.9 → 38.1 (×1.05) | 83.0 → 43.5 (×1.91) | 171.3 → 76.8 (×2.23) |
| q8 | 4096 | 175.9 → 150.9 (×1.17) | 348.5 → 215.9 (×1.61) | 696.0 → 412.0 (×1.69) |

## Cost

- 4090 pod time: the bits job 46f0 and timing job 4546 (pod1).
- H100: $0.933 (job 3dfd, ~14 min booked, 8 min of stages).
- Kit prewarmed on bella (NVRTC 13.2 / 13.3 / 13.4 sm_90a, no on-pod compile: "kernel cache 1 -> 4", nothing compiled).
- bella rehearsal before booking: 1026-223829-01f2.

## Review round (shady 8/10), f41f2ec

- **MUST.** At the H100's nb 32 (128 CTAs < 132 SMs), the default S = 3 split was ×0.96 / ×0.98 of attn_blk_b + cmb, and the step was +0.6 / +0.8 %.
  - `nb_split` now returns None below rows × kv heads = SMs, and the step keeps main's attn_blk_b there.
  - attn_blk_nb runs only at S = 1, where every measured cell is a win.
  - `AXON_ATT_NB_S` still forces a split, for experiments only.
  - The fix is argued from the dispatch, not re-timed on an H100.
- `AXON_ATT_NB` now also accepts off / false / no. Both env switches are read once per module.
- The "2 CTAs per SM" comment is corrected: at 69–75 KB the kernel fits one CTA per SM on sm_86/89/120.
- `cuda_attn_nb_exact` adds GQA groups 8 and 6 and position 0.
- The clippy `manual_checked_ops` warning in that test is fixed.
