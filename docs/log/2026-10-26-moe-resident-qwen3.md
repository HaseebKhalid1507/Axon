# qwen3moe experts resident on the GPU: `--gpu-experts` for Qwen3-30B-A3B and relatives (PR E)

- Date: 2026-10-26. Branch: feat/moe-resident-qwen3, off main be13f1f; main 680a2c5 merged in at 111456a, the
  merge-ready head. Review: shady 8/10 APPROVE-WHEN-GREEN; SHOULDs done (f2148c6, bf70daa).
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

## How it was verified (MEASURED; the PR body has the job links)

Gates of record, at 111456a, all on bella:

| gate | job | result |
|---|---|---|
| resident suite + Moonlight, release | 1026-062648-e18c | PASS 29/0/1 |
| full release, 2 shards | group 551e | PASS 438/0/22 + 266/0/29 |
| full debug, 2 shards | group 8dfb | PASS 438/0/22 + 266/0/29 |
| md5 vs main 680a2c5 | 1026-062650-e264 | pending |

Earlier rounds:
- **b9a4093:**
  - resident suite 41/0/1 on bella (release, debug) and avante;
  - jade 37/4: the 4 are pre-existing `moe::tests::*_pruned` tests refused by #142's RAM check under jade's 6 GB.
    They were not run on sm_61 and are covered on sm_89 / sm_120;
  - fast tier 683/0/50 and 701/0/50;
  - md5 vs be13f1f all identical (cpu 99/99, cuda 108/108, cuda q8 108/108).
- **f2148c6:** q3res tests 3/0/0, release and debug.

A/B of record: 1026-075028-1c02 (111456a vs main 680a2c5, `--timing bella`), **CLEAN** (max 5 %, mean 0.5 %).
Record rule: spread < 10 %, ≤ 1,000 major faults; every run had 0 major faults. Text is identical across main /
`--gpu-experts 0` / `auto`.

| model | metric | main | `0` | `auto` | auto / main |
|---|---|---:|---:|---:|---:|
| Qwen3-MOE-4x0.6B | tg512 | 171.4 | 173.2 | 427.6 | ×2.49 |
| Qwen3-MOE-4x0.6B | pp2100 | 1,978 | 2,172 (noise: ranges overlap) | 27,785 | ×14.0 |
| pruned Qwen3-30B-A3B | tg512 | 760.8 (11 % spread: not a record) | 685.6 (not a record) | 951.8 (0.1 %) | ×1.25, indicative only |
| pruned Qwen3-30B-A3B | pp2100 | 8,563 | 8,422 (−1.6 %, records) | 90,793 | ×10.6 |

The earlier A/B 63a6 (b9a4093 vs be13f1f) was TAINTED (max 32 %) and is superseded. The full Qwen3-30B-A3B end to
end waits for a pod bench (`auto` vs `0`, text and speed).

## What didn't work / open
- The full Qwen3-30B end-to-end number isn't measurable on bella. The 18.6 GB file doesn't stay in the page cache
  (job 5052: up to 399k major faults per run, residency drifting 26–98 %). It waits for a pod bench.
