# T2b: the MoE decode step: fewer, wider expert launches, sum + residual fused, router top-k by shuffles

- Date: 2026-10-26. Branch: feat/moe-decode-step, off main 1e01712. Status: draft, gates and the kill-rule bench
  pending.
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

Timing: the 4090 board job 1027-012527-1ea0 is pending. The kill rule is applied as written when it lands.
