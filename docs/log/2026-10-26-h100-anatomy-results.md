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

- **Exact vs no drain:** the pre-registered rule is "within 10 %", with no nb scope. Measured over every cell:
  - 0.94–1.00 at nb ≥ 128, except 0.906 (kv nb 1024) and 0.908 (q/wo nb 256). q/wo nb 512 is 0.945.
  - The lowest is **0.878 at down nb 8**, outside 10 %.
  - Restricting the verdict to nb ≥ 128 (the sizes that route to gemmh2; nb 8 runs on gemv/gemvm) is a **scope
    chosen after the data**, stated here as such.
  - On q/wo at nb 256–512 and kv at nb 1024, both the no-drain and the magic variant beat the exact kernel, which
    looks systematic, not like noise. It is **unexplained and needs a rerun** of those cells.
  - The no-drain variant still waits per block, so this bounds the drain's arithmetic, not that wait.
- **The no-drain GEMM is itself 119.4 TMAC/s** at gate nb 1024, the same as the exact kernel and as `gemm2t`
  (mma.sync).
- **Magic-S:** bit-identical in all 12 kernel cases (3 of them adversarial: 3584×192, 18944×128 and the fused
  gate+up 3584×256) and on 134 M device pairs. Speed vs exact:
  - 0.97–1.06 on gate / down / gate+up at nb ≥ 128, and 0.999 at gate nb 1024;
  - **1.06–1.11 on q/wo at nb 256–512 and kv at nb 1024**, plus 1.083 at down nb 8.
  - It may be a free exact win of 6–11 % on the narrow shapes: **open, untested** until those cells are rerun.
- **I2F throughput:** I2F 33.8 results/clk/SM, magic 38.1. The research note cited 16.

## Decode step (`pool_step_cost`, 7B, graphs on, ms per step)

| depth | nb 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| 600 | 4.64 | 5.46 | 9.60 | 11.55 | **15.72** | 27.39 |
| 4096 | 4.84 | 6.50 | 13.18 | 18.48 | **29.22** | 53.80 |

The solo decode step (forward_argmax) is 3.23 / 3.32 ms. The review's showdown figure was 18.6 ms at nb 128 and
depth ~790, which falls between the two rows.

Per kernel, nb 128 (events, no graphs). Caveats:
- The totals are above the graphed step: +24 % at depth 600 (19.47 vs 15.72 ms), +12 % at depth 4096.
- The segments include quantize work that the class split counts as "other": gate_up 5.21 + narrow 6.30 =
  11.51 ms, above the GEMM class's 10.97.
- "attention" below is attn + combine. The attention class including rope_kv is 5.70 / 18.93 ms.

| depth | total | GEMM | attention (attn + combine) | norms / elementwise | head | narrow products (q, k, v, wo, down) |
|---|---|---|---|---|---|---|
| 600 | 19.47 | 10.97 (56 %) | 5.47 (28 %) | 0.91 | 1.10 | 6.30 (q 0.77, k 0.81, v 0.79, wo 0.76, down 3.18) |
| 4096 | 32.62 | 10.91 (33 %) | 18.70 (57 %) | 0.91 | 1.09 | 6.30 |

At nb 256, depth 4096, attention is 37.4 of 59.7 ms (63 %).

For M6's size: at nb 128 on the narrow shapes, `gemm2s` (64 × 64 tiles, more CTAs) beats gemmh2 by 1.5–1.6×:
q/wo 70.1 vs 46.4, down 81.6 vs 51.3 TMAC/s. On kv at nb 1024 it is 1.66× (78.8 vs 47.5).

## Serve, mix C = 128 (sd_client, the showdown's prompt set and SLO)

| build | agg tok/s | TTFT p50 / p99 ms | ITL p50 / p99 ms | goodput | slo_ok |
|---|---|---|---|---|---|
| this branch (two-pass steps), `AXON_SERVE_TRACE` on | 1,879 | 1,844 / 13,713 | 58.2 / 78.8 | 694 | 0.32 |
| #119 hybrid step (ecdb1d0) | 1,822 (−3 %) | 10,222 / 15,679 | 28.2 / 40.5 | 211 | 0.11 |

The first row reproduces the 10-19 showdown cell (1,868 tok/s, ITL p50 57.9).

Caveats on the comparison:
- #119 was **closed NO-GO on 10-25** (`2026-10-25-reject-hybrid-step.md`). This cell is consistent with that verdict.
- Only one run per arm, the arms are on different bases (9efdcf2 vs ecdb1d0), and only the first arm was traced.
- So "−3 %" is within single-run noise at C = 128. TTFT ×5.5 and goodput −70 % are large enough to stand.

Step trace of the first row, 1,047 steps, 38.04 s wall:

| part | seconds |
|---|---|
| prompt feeds (716 passes, 244,896 tokens, 467 per step at p50) | 21.87 |
| decode passes | 15.69 |
| host work | **0.45 (1.2 %)** |
| idle | 0.03 |

- 524 of the 1,047 steps ran two passes (a feed plus a decode).
- Their decode passes cost 9.73 s. The decode step at ≥ 0.9 × C rows is 20.9 ms at p50.

Against the outside review's model (§2.2):
- Its model: 251k prompt tokens at a lone prompt's 13.8k tok/s = 18.2 s, plus 556 ideal decode steps × 18.6 ms =
  10.3 s.
- Measured: feeds 21.87 s and decode 15.69 s, **≈ 9.1 s (24 %) above that model**.
- The excess is inside the GPU passes, through the mechanisms the review named: two passes per step (524 of them)
  and small chunks (feeds at 11.2k tok/s vs 13.8k).
- This **confirms the review's ~25 % in size and cause**. Host work is negligible (1.2 %); the review didn't claim
  otherwise.

#119's `pool_hybrid_cost`: at the same token budget, fusing saves 30–35 % at chunk 128 (nb 32 / 64), 10–26 % at 256, 11–17 % at
512. At the same chunk the saving is −4.5 to +15.7 %: negative at chunk 256 for nb 64 and 128, and at 512 for nb 128.

## Prefill pass fixed cost (`pool_feed_cost`)

| depth | t0 | per token |
|---|---|---|
| 0 | 5.34 ms | 57.3 µs |
| 600 | 5.69 ms | 58.2 µs |
| 4096 | 7.35 ms | 64.7 µs |

**Correction to 7643217:** that commit read "t0 is under the estimate, so 1(c) doesn't need #119". It compared the
wrong quantity.
- The spec derived its "t0 ≈ 8–10 ms" from a 14 ms short-prompt TTFT and used it as the cost of a whole
  short-prompt pass (104 × t0 ≈ 1 s per wave, spec §4.5).
- Measured: a 64-token pass costs 9.96 ms (depth 0). In the serve trace, the 25 single-feed steps of ≤ 80 tokens
  took a median **9.79 ms** of feed.
- So the spec's estimate holds. About 1 s per 104-short wave sits between the rule's two outcomes (0.5 vs 1.5 s);
  short TTFT p50 ~0.3 s is on the edge.

## What the numbers decide

Read against the pre-registered rules in the kit's PLAN.md (axon-devstack `runbooks/h100-anatomy`), filled there.

- **Exactness contract:** cleared on Hopper within 10 % for nb ≥ 128. That scope was chosen after the data: down
  nb 8 is 0.878, and q/wo / kv need a rerun.
- **Item 2, E0'(b):** the kill trigger **fired** (no-drain 119.4 < 180). The rule's two branches were "structure
  first" or "go to item 4". **Owner's choice after the trigger (Jawz, lead): item 2 continues structure-first.**
  Reason: no-drain equals exact (119.4 vs 118.9), so dropping exactness (item 4) recovers nothing at nb 1024.
- **M6 split-K:** the narrow products are 6.3 ms of the nb 128 step (≥ 3 ms, robust to the event and quantize
  overheads), so M6 goes before the round-2 GEMM. gemm2s's 1.5–1.66× on narrow shapes says more CTAs pay there.
- **Attention at depth:** it is the largest term at depth 4,096 (57 %), which the review's ranking didn't have.
- **The review's ~25 % scheduler term is confirmed** (≈ 24 %, inside the passes: two passes per step and small
  chunks). Host work is 1.2 %.
- **#119:** closed NO-GO on 10-25, and this cell is consistent with that. Reopening it would need its own proposal.
- **Shortest-first (1(c)):** the spec's ~10 ms per short pass holds, which puts ~1 s per wave on the edge.
  Independently of t0, 1(c) doesn't wait for #119 because #119 is closed.
