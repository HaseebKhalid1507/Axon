# T2b: the MoE decode step: fewer, wider expert launches, sum + residual fused, router top-k by shuffles

- Date: 2026-10-26. Branch: feat/moe-decode-step, off main 1e01712 (main 0c23a87 merged in at a016e0c).
  Status: **kill rule MISSED; merges default-on per methodology step 9 (#155); the line closes.**
- Design note: `specs/t2b-moe-decode-step.md` (foreman workspace). Diagnosis: job 1026-222632-d7c1 (CUPTI decode
  traces on an RTX 4090).

## Kill rule (pre-registered, written before any timing)
- Qwen3-30B-A3B all-resident tg128 on a clean RTX 4090 must reach **≥ 252.5 tok/s** (llama.cpp `4413a19`'s median
  in #144's bench of record). Same board script, interleaved, 3 rounds. Else stop after 4 builder-days and write a
  NO-GO.
- No regression > 2 % on Moonlight (tg128 / pp512 / pp2100) or on dense models.

## What changed
- `moe_res.cu`:
  - `moe_gu_tw` (gate and up of a slot in one launch, two rows of each per block);
  - `moe_gemv2_tw` (down, two rows per block);
  - `moe_sum_add` (the weighted sum, then x += y).
- `experts.rs`: the decode path (nb 1) uses them when gate and up share a type and the row counts are even.
  `enqueue_add` is the residual-fused variant.
- `cuda/qwen2/moe.rs` (qwen3moe `moe_step`): a resident layer adds into x itself, and the next segment skips its
  `moe_add`.
- `moe.cu` `moe_router`: the k argmax rounds by warp shuffles and one cross-warp fold (2 barriers a round instead of
  a 7-level block tree).
- Moonlight's MLA path gets the new GEMV kernels through `enqueue` (unchanged call); `mla.rs` and `mla.cu` are not
  touched.

## Why
d7c1, Qwen3-30B-A3B all-resident decode on the 4090: Axon 4.57 ms GPU busy per step vs llama.cpp 3.80.
- Expert GEMVs: 1.81 vs 1.31 ms (60 % vs 84 % of bandwidth).
- Expert glue launches: 0.32 vs ≈ 0.10.
- Router: 0.47 vs 0.25.

## How it was verified / results
Bits (MEASURED, code 19931aa vs main 1e01712): ALL PASS.
- Expert / twin / q3res / ds2 / moe / oss tests, 49/0/1 each: bella release 1026-224704-d556 (with Moonlight), bella
  debug 1026-224705-d585, avante sm_89 1026-224705-503d. They cover the CPU vs GPU check at nb 1 through the new
  kernels, and `enqueue_add` == x + y.
- md5 1026-224705-6959: cpu 99/99, cuda 117/117, cuda `--kv q8` 117/117 identical. That includes Moonlight,
  4x0.6B and the pruned 30B at `--gpu-experts auto` and at a partial N, each equal to main at N = 0.

Timing (MEASURED, RTX 4090 pod, CLEAN, 3 interleaved rounds; candidate 19931aa vs base main 1e01712 vs llama.cpp
4413a19; raw logs in [this gist](https://gist.github.com/HaseebKhalid1507/658a01e192cc7dbb616028becb9f80d4)):

| model, all resident | metric | candidate | base | cand / base | llama.cpp | cand / llama |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B (6db3) | tg128 | 241.6 | 222.1 | ×1.088 | 254.6 | **×0.95** |
| | pp512 | 11,329 | 11,291 | ×1.003 | 8,851 | ×1.28 |
| | pp2100 | 13,016 | 13,000 | ×1.001 | 9,041 | ×1.44 |
| Moonlight (1ea0) | tg128 | 272.0 | 270.9 | ×1.004 | 289.8 | ×0.94 |
| | pp512 | 9,746 | 9,728 | ×1.002 | 11,696 | ×0.83 |
| | pp2100 | 7,486 | 7,473 | ×1.002 | 12,220 | ×0.61 |

Merge re-gate at d99e400 (code a016e0c = 19931aa + main 0c23a87), MEASURED, bella, all rc 0
([gate logs](https://github.com/HaseebKhalid1507/Axon/tree/5c7bb3917328aeb500027db7ab14798d1b589e80/2026-10/t2b-d99e400-subset)):
- `cuda:: kv pool:: moe::` + Moonlight: release 1027-023104-dddc and debug 1027-023104-465b, both 249/0/39.
- md5 vs main 0c23a87, 1027-023052-3dbb: cpu 99/99, cuda 117/117, cuda `--kv q8` 117/117 identical.

## Verdict
- **Kill rule MISSED:** 241.6 < 252.5. The no-regression half passes (Moonlight ×1.002–1.004; dense kernels untouched, md5
  identical).
- **Methodology step 9 (#155):** the code is exact (md5 and bits gates, text identical across configs), +8.8 % on the
  decode step it targets, and regresses no cell past 2 %. So it merges default-on, with the miss recorded here, in the PR
  and in roadmap item 3. The line closes: no follow-ups without a fresh pick.
- **Remaining decode gap to llama.cpp on the 30B: ~5 % (×0.95, ≈ 0.2 ms per step).** By d7c1's breakdown (ESTIMATED):
  - the expert GEMVs, still below llama.cpp's bandwidth;
  - the router (logits and top-k are separate launches);
  - `qkv_rope` at ~61 % of bandwidth.

## What didn't work
The target. Wider expert GEMVs, the fused sum + residual and the shuffle top-k bought +8.8 %, where the estimate was
+25–30 %. The router fusion and the `qkv_rope` loads weren't attempted within the line's spend.
