# qwen3moe experts resident on the GPU: `--gpu-experts` for Qwen3-30B-A3B and relatives (PR E)

- Date: 2026-10-26. Branch: feat/moe-resident-qwen3, off main be13f1f. Status: draft, gates pending.
- Track: roadmap item 3 (Qwen3-30B-A3B as the headline). Step 0 notes: the PR body, §1.

Labels: **MEASURED** (job cited), **ESTIMATED**, **CODE**.

## What changed
- `cuda/qwen2/moe.rs`: the qwen3moe passes enqueue a resident layer's experts on the router's output (`mb.r`) into
  `mb.y`, where the host round trip would have put the CPU's y. `moe_step` (decode) does it inside the layer's
  segment graph; `moe_tokens` (batched prefill, verify) after the router. The kernels are the existing twins
  (`CudaExperts::enqueue`).
- `moe.rs`: placement accepts qwen3moe, probing its scratch (`moe_bufs`) for the ctx-first budget; every cache gets
  `ResBufs`. gpt-oss stays refused (MXFP4 has no twin; expert biases).
- `model.rs`: `Qwen2::build_skip`. The CPU half builds no experts for the resident layers (`Weight::absent`, no
  arena), and their host bytes are released, as for deepseek2.
- Help text and README.

## Why
- On bella, qwen3moe decode spends most of each step in CPU experts. On the pruned 30B-A3B (the 30B's expert shapes)
  that is 0.24–0.32 ms per layer in r8, MEASURED (s0q3b, job 1026-022525-5052, record). Over 48 layers that is
  ≈ 13–15 ms per token (ESTIMATED).
- The full 30B fits ≈ 25 of 48 expert layers on bella at 4k (ESTIMATED from the GGUF header; 2.5 GB peak card use
  measured with every expert on the CPU).

## How it was verified
Gates pending: see the PR body (gatewatch `moerese-b9a4093`).

## What didn't work / open
- The full Qwen3-30B end-to-end number isn't measurable on bella. The 18.6 GB file doesn't stay in the page cache
  (job 5052: up to 399k major faults per run, residency drifting 26–98 %). It waits for a pod bench.
