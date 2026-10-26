# H100 SXM step anatomy: results (2026-10-26)

The session [2026-10-24-h100-anatomy.md](2026-10-24-h100-anatomy.md) prepared, run once. Roadmap item 1(a)
([Next: proposed order](../roadmap.md#next-proposed-order-outside-review-10-19-owner-decides)), outside review 10-19
§2.4, plus the research's E0' and the shortest-first spec's t0. All numbers MEASURED on one RunPod H100 80GB HBM3 (SXM,
secure, sm_90, 132 SMs, driver 595.91, NVRTC 13.2), kit binaries at 9efdcf2 (#119's at ecdb1d0), 7B Q4_K_M.
Raw: [gist](https://gist.github.com/HaseebKhalid1507/8edbbf8cf1531a9041b409777ed27209) (tables, `cells.jsonl`,
`cost.json`, stage times, driver log, hardware); bella `~/.cache/axon-h100-anatomy/pod-pod1/results/`.

## Session

Launch 1026-125601. All 11 stages rc 0, 554 cells, `an_check` PASS. `ncu` was denied: RunPod containers keep the
performance counters admin-only (`RmProfilingAdminOnly`), the risk the plan named, so no stall reasons.

**Cost:** driver `cost.json` paid 7.58 min / $0.504; the axon-pods ledger says $0.533 for the pod's 7 min (it rounds
up per minute and adds disk). Useful 4.78 min, setup 2.78 min (create → driver start), waste 0.369. The plan
estimated ≈ 18 min / ~$1.20 likely, ≤ $2.00 expected worst, $2.99 hard stop. The stages took 4.8 min against an
11.4 min estimate (the H100 ran them about twice as fast as the rehearsal host). Paid for by the 10-26 12:00–24:00
allowance window.

## GEMM (`cuda_gemmh_bench`, median TMAC/s of real MACs)

| shape | nb | gemm2s | gemm2t | gemmh2 (exact) | gemmh2 no drain (INEXACT) | gemmh2 magic S | exact / no drain | magic / exact |
|---|---|---|---|---|---|---|---|---|
| gate 3584×18944 | 128 | 81.1 | 86.6 | 79.0 | 83.9 | 83.5 | 0.942 | 1.057 |
| gate | 1024 | 94.4 | **119.4** | **118.9** | **119.4** | 118.8 | 0.996 | 0.999 |
| q/wo 3584×3584 | 256 | 75.8 | 94.3 | 82.9 | 91.3 | 91.7 | **0.908** | 1.106 |
| q/wo | 1024 | 89.3 | 99.2 | 96.9 | 97.9 | 97.1 | 0.990 | 1.002 |
| down 18944×3584 | 1024 | 96.8 | 109.3 | 105.4 | 105.5 | 104.9 | 0.999 | 0.995 |
| kv 3584×512 | 1024 | 78.8 | 51.7 | 47.5 | 52.4 | 52.7 | **0.906** | 1.109 |
| gate+up fused | 1024 | 90.6 (gemm2s_gu) | — | 112.9 | 114.5 | 114.0 | 0.986 | 1.010 |

The full table (nb 1 … 1024, every kernel) is in the gist. Reference ceilings, `cuda_gemm_bench`: `mma.sync` int8
peak 325 TMAC/s, f16 279.

- **Exact vs no drain:** 0.94–1.00 at every shape for nb ≥ 128, except two cells at 0.906 / 0.908 (q/wo nb 256,
  kv nb 1024).
  - Those two are the same cells where the magic variant beat the exact kernel by 10–11 %, so they look like slow
    samples of the exact kernel rather than a drain cost.
  - The lowest overall is down nb 8 at 0.878, a size the product runs on gemv/gemvm, not gemmh2.
  - The no-drain variant still waits per block, so this bounds the drain's arithmetic, not that wait.
- **The no-drain GEMM is itself 119.4 TMAC/s** at gate nb 1024, the same as the exact kernel and as `gemm2t`
  (mma.sync).
- **Magic-S:** bit-identical in all 12 kernel cases (4 adversarial, including the fused gate+up) and on 134 M
  device pairs. Speed vs exact is 0.97–1.06 everywhere except those same two outlier cells.
- **I2F throughput:** I2F 33.8 results/clk/SM, magic 38.1. The research note cited 16.

## Decode step (`pool_step_cost`, 7B, graphs on, ms per step)

| depth | nb 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| 600 | 4.64 | 5.46 | 9.60 | 11.55 | **15.72** | 27.39 |
| 4096 | 4.84 | 6.50 | 13.18 | 18.48 | **29.22** | 53.80 |

The solo decode step (forward_argmax) is 3.23 / 3.32 ms. The review's showdown figure was 18.6 ms at nb 128 and
depth ~790, which falls between the two rows.

Per kernel (events, no graphs, so totals are about 20 % above the graphed step), nb 128:

| depth | total | GEMM | attention (attn + combine) | norms / elementwise | head | narrow products (q, k, v, wo, down) |
|---|---|---|---|---|---|---|
| 600 | 19.47 | 10.97 (56 %) | 5.47 (28 %) | 0.91 | 1.10 | 6.30 (q 0.77, k 0.81, v 0.79, wo 0.76, down 3.18) |
| 4096 | 32.62 | 10.91 (33 %) | 18.70 (57 %) | 0.91 | 1.09 | 6.30 |

At nb 256, depth 4096, attention is 37.4 of 59.7 ms (63 %).

## Serve, mix C = 128 (sd_client, the showdown's prompt set and SLO)

| build | agg tok/s | TTFT p50 / p99 ms | ITL p50 / p99 ms | goodput | slo_ok |
|---|---|---|---|---|---|
| this branch (two-pass steps), `AXON_SERVE_TRACE` on | 1,879 | 1,844 / 13,713 | 58.2 / 78.8 | 694 | 0.32 |
| #119 hybrid step (ecdb1d0) | 1,822 (−3 %) | 10,222 / 15,679 | 28.2 / 40.5 | 211 | 0.11 |

The first row reproduces the 10-19 showdown cell (1,868 tok/s, ITL p50 57.9).

Step trace of the first row, 1,047 steps, 38.04 s wall:

| part | seconds |
|---|---|
| prompt feeds (716 passes, 244,896 tokens, 467 per step at p50) | 21.87 |
| decode passes | 15.69 |
| host work | **0.45 (1.2 %)** |
| idle | 0.03 |

- 524 of the 1,047 steps ran two passes (a feed plus a decode).
- Their decode passes cost 9.73 s. The decode step at ≥ 0.9 × C rows is 20.9 ms at p50.

#119's `pool_hybrid_cost`: at the same token budget, fusing saves 30–35 % at chunk 128 (nb 32 / 64), 10–26 % at 256, 11–17 % at
512. At the same chunk the saving is −4.5 to +15.7 %: negative at chunk 256 for nb 64 and 128, and at 512 for nb 128.

## Prefill pass fixed cost (`pool_feed_cost`)

| depth | t0 | per token |
|---|---|---|
| 0 | 5.34 ms | 57.3 µs |
| 600 | 5.69 ms | 58.2 µs |
| 4096 | 7.35 ms | 64.7 µs |

The shortest-first spec estimated t0 at 8–10 ms on an H100, derived from the C = 1 short-prompt TTFT, so it is the
cost of a whole short-prompt pass. Measured, a 64-token pass costs 9.96 ms (depth 0), which is inside that range.
The spec's ~1 s per 104-short wave in two-pass mode therefore stands, and so does its "on the edge without #119" note.

## What the numbers decide

Read against the pre-registered rules in the kit's PLAN.md (axon-devstack `runbooks/h100-anatomy`), filled there.
In short:

- **Exactness contract:** cleared on Hopper (within 10 %).
- **Item 2, E0'(b):** the no-drain kernel is under 180 TMAC/s, so the restructure comes first.
- **M6 split-K:** the narrow products are 6.3 ms of the nb 128 step (≥ 3 ms), so M6 goes before the round-2 GEMM.
- **Attention at depth:** it is now the largest term at depth 4,096, which the review's ranking didn't have.
- **The review's "~25 % is neither kernel" is wrong as stated:** host time is 1.2 %. The scheduler cost is real, but
  it sits inside the passes: 524 two-pass steps and 467-token chunks.
- **#119 as configured doesn't pay at C = 128 on Hopper:** −3 % aggregate, TTFT ×5.5, goodput −70 %, against
  ITL ÷2.
- **Shortest-first:** a short pass costs ~10 ms, which confirms the spec's estimate. A wave of shorts in two-pass
  mode takes ~1 s, and short TTFT p50 ~0.3 s is on the edge. #119 as configured doesn't help at C = 128, so 1(c) goes
  ahead without waiting for it and is judged by its own A/B.
