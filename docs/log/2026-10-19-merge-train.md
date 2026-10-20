# Merge train 1019-1542: #71 + #84 + #98 landed as one tree  (docs-only record)

- Date: 2026-10-19 (gates 15:44 → 18:25, landed 18:49 EDT). Train tip: `07c6782` = main `3575503` (#100) + #71 + #84
  + #98. Landed as e1faa2e (#71), b8a7ec7 (#84), c15c90e (#98). Status: merged; main c15c90e's tree is identical to
  07c6782 (`git diff 07c6782 c15c90e` is empty), so no post-merge run was needed.
- Pick / design: the H100 showdown's condition (Haseeb, 10-19 02:28: "once the PRs land"). The three PRs touch the
  same files (`src/cuda/qwen2/batch.rs`, `serve/`, `cuda/qwen2.rs`); gating them together, on the tree that would
  become main, was cheaper than three serial rebases with three full gates. Tool: axon-devstack `axon-train`.

## What landed

| PR | what | its own headline (from the PR body) |
|---|---|---|
| #71 paged KV stages 2+3 | the pool allocates 64-position pages (`PageMap`, per-slot page lists, a device page table); serve plans its KV by pages, admits by pages, reclaims idle prefixes and preempts by recompute (youngest first). An out-of-memory CUDA graph capture drops the other graphs, retries once, else runs the step kernel by kernel (same bits). Kill switch `AXON_PAGED_KV=0` | MEASURED, bella exclusive (1019-105853-7408): 7B f16 at 32k, 7 × 15k agents + 600: aggregate ~+10 % (72.6 / 70.4 → 79.1 / 79.2 tok/s), ITL p50 up (more streams decode at once); C=1 / C=8 unchanged within the round spread; md5 0/45 |
| #84 attn_blk_mq | multi-query block attention: one CTA loads each K/V tile once for 16 queries of one sequence; used for one-sequence passes of ≤ 32 queries (`AXON_ATT_MQ=0` = old dispatch) | the spec's kill rule **failed** at 128 queries (0.76× attn_fa_b on a 4090); shipped for ≤ 32. MEASURED on bella: serve admit at chunk 32 TTFT −6.4 %, decoders' ITL p99 −8.6 %; gen with a draft model at 21.7k +20.4 % |
| #98 gpt-oss stage 1 | attention (sinks, alternating 128-token window), norms, router and head on the GPU; MXFP4 experts on the CPU | MEASURED, bella: tg128 46.6 vs 23.4 tok/s (1.99× llama.cpp `-ot exps=CPU`); prefill 265 vs 995 (**0.27×**); `gen` md5 vs main 0 of 106 differ (CUDA f16, q8 KV and CPU rows) |

## Gates (on the train tip 07c6782, all MEASURED)

| gate | result | job |
|---|---|---|
| release suite, 3 shards (bella) | PASS 229 + 162 + 240, 0 failed | 1019-154447-d4fa, -d498, -40fd |
| debug-assertion suite, 3 shards (bella) | PASS, 0 failed | 1019-154447-21df, -6728, -8388 |
| md5 of `gen`, main vs train tip, same host | 0 of 30 differ | 1019-154447-7817 |
| sm_89 bit tests (avante) | PASS 61 / 0 | 1019-161853-1f3c |
| CI on each PR | green (#84 retargeted to main so CI would run) | GitHub |

Not counted: be57 (rc 101) was a missing-models `--src`, not code; 0d8d / 1adb / 64db / 6c09 were killed because
`train-*` jobs claim the full RAM family and couldn't place on avante, and were replaced by 1f3c.

## Serve after the train (MEASURED)

An exclusive avante A/B, ecb53b7 (before #97 / #100 / #102 and this train) vs c15c90e, job **1019-203211-9269**,
0.5B Q4_K_M, `--parallel 4`, C = 1 / 2 / 4, 3 rounds: classic 333 / 570 / 1,067 vs 333 / 571 / 1,067 tok/s, mix
306 / 482 / 793 vs 306 / 474 / 791. No regression. Details: [2026-10-19-h100-showdown.md](2026-10-19-h100-showdown.md#pre-run-ab-did-the-merge-train-regress-serve).

## The other train of the day

Train 1018-2343 landed #89 (deepseek2 stage 1, 479495c) and #92 (wave-sized prompt passes, ecb53b7) at 02:34 EDT,
the tree equal to its tip; its 7 jobs passed, including the md5 job (…5394). Rule recorded the same night: after a
train lands, no post-merge run (the tree was already gated).

## What didn't work

- The first #71 train (`c26ffe0`, main + #71 @ 9a1d281) went stale: #71's exclusive bench (1018-233201-7bb1) found a
  real crash (a lazily captured graph hit CUDA_ERROR_OUT_OF_MEMORY with 0.24 GiB left after the page plan, and the
  engine exited). The fix (OOM-safe capture, sampling buffers in the plan) changed #71, so that train's remaining jobs
  were killed and the train rebuilt on the fixed head.
