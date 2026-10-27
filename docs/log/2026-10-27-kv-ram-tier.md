# The KV RAM tier: `axon serve --cache-ram` (PR #166)

- Date: 2026-10-27. Branch: feat/kv-ram-tier. Base: main 0c23a87, then merged with 6f06544 (head d349483).
  Status: open (draft). Rule §6 row B **not passed** on the first clean A/B (the ITL p99 clause; below).
- Pick / design: agent workload track, PR B ([design/agent-workload.md](../design/agent-workload.md) §3, §6 row B; #158).
  Rule A (#160's f16 board) did not deprioritise it: main re-prefills 10.3 % at speed 1 and 47.5 % at speed 4.

## What changed
- `--cache-ram BYTES|auto` (default 0: off; none of the new code runs). `src/kvstore.rs`'s page store, unused
  since #48, wired into serve (`src/serve/tier.rs`):
  - **Writes.** A slot's whole 64-position pages are copied to host RAM as they fill: a write-behind of at most
    16 MiB per engine step while requests decode, and the rest before the slot gives its rows up (reclaim, admit,
    lend_solo, unfeed, preempt).
  - **Restore.** On admission, the longest stored prefix of the prompt below its last token is copied back, and
    only the rest is prefilled.
- **CUDA pool.** Pages sit in a pinned arena of fixed-size page slots, in 64 MiB `cuMemHostAlloc` slabs allocated
  as the store fills. One `cuMemcpy2DAsync` per region per page goes through the slot's page list
  (`CudaQwen2::pool_page_io`), all on the main stream.
- **CPU and `--parallel 1`.** The same through `KvIo` on the one cache.
- **Off** where the cache has no image (split, pipeline, MoE, deepseek2).
- **PR C seam.** `PoolTier::spill_page` / `BatchLm::page_out_id` (by physical page id), agreed with #164.
  Not called yet.
- **Sizing.** `auto` = min(½ RAM available after the model, 8 GiB). A size past the RAM available is refused.
- `/metrics` gains `axon_kv_ram_*`; README serve flags.
- Scripts: `scripts/kv_resume_bench.py` (the TTFT-resume micro-bench).

## Why
§0 of the design: 97.6 % of an agent's prompt tokens are a prefix the previous request already had, and sessions
idle for minutes. Main drops an idle slot's prefix on reclaim or slot reuse, and the next turn re-prefills it all.
#160's board (bella 7B f16, agent-32k):
- **Speed 1:** main's TTFT-resume p50 is 4.72 s in the 2–10 min bucket.
- **Speed 4:** main's TTFT-resume p50 is 15.9 s, with 47.5 % of prompt tokens re-prefilled.
- **llama-server**, with its 8 GiB RAM prompt cache, re-prefills 1.9 %.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| release suite, 3 shards, at d349483 | 726 passed / 0 failed / 55 ignored | 1027-083426-55e2, -0f60, -64a4 (bella) |
| debug-assertion suite, 3 shards, at d349483 | 726 / 0 / 55 | 1027-084942-d6b0, 1027-084943-d527, -0871 (bella) |
| md5 of `gen`, main 0c23a87 vs db6e6e3: 4 models × 3 prompts × cuda f16 / cuda q8 / cpu | 30/30 equal | 1027-034836-d86a (RTX 4090) |
| serve md5: 7 sessions × 3 turns, main vs `--cache-ram 0` vs `1G`: 7B f16, 7B q8, gemma-3-1b f16, Qwen3-0.6B q8, 0.5B cpu | 42/42 each | d86a; 7B cells 1027-040338-78d8 (harness fix rerun) |
| clippy -D warnings (lib + tests), rustfmt | clean | bella |
| new tests: `pool_pages_restore_exactly` (KV bytes + 20 greedy tokens == uninterrupted; f16/q8 × plain/paged/shuffled × qwen2.5/qwen3, gemma3 f16), `serve_{cpu,cuda}_cache_ram` (replies == `--cache-ram 0`, solo and batch admit, paged/plain/q8/one-at-a-time) | pass | in the suites above |

## Results
TTFT-resume micro-bench, **RTX 4090 pod (sm_89, driver 595.91.07, PCIe gen 1 x16 reported idle), job
1027-100925-fbcb, 0 % foreign GPU load**, d5a2899 vs main 0c23a87. 3 rounds interleaved, medians.
- **Setup.** 7B Q4_K_M f16, `--ctx 32768 --parallel 3`, which caps the paged pool at 98,304 tokens, bella's pool.
  Candidate `--cache-ram 6G`.
- **Resume cell.** One 25k session; 3 others push it out of the pool; 120 s idle; its next turn with 157 new
  tokens.
- **ITL cell.** 2 streams of 256 tokens decoding while a fresh 25k prompt prefills.

All MEASURED:

| metric | main | `--cache-ram 6G` | Δ |
|---|---|---|---|
| TTFT-resume p50 (25.6k-token prompt) | 2.647 s [2.646–2.648] | **0.149 s** [0.148–0.149] | −94 % (17.8×) |
| prompt tokens cached on resume | 0 of 26,692 | 26,496 of 26,692 | the tail under a page is re-fed |
| ITL p50 (510 samples / round) | 6.41 ms | 6.42 ms | +0.1 % |
| **ITL p99** | **68.67 ms** [67.85–70.77] | **83.44 ms** [77.22–83.77] | **+21.5 %** |
| reply md5 (every round, both) | 6948a467… | 6948a467… | same |
| pinned host memory at the end | — | 6,440,878,080 B (the 6 GiB cap: 1,755 pages) | |

Bandwidth on the same pod, same job (MEASURED, a 25.6k-position 7B-shaped slot):
- **Whole slot:** D2H 26.4 GB/s, H2D 26.3 GB/s; per 64-position 2D page, 26.0 GB/s.
- **One page alone (3.5 MiB):** D2H 0.187 ms, H2D 0.146 ms.
- **`host_alloc` (with free):** 64 MiB 61.8 ms, 256 MiB 252 ms.

On bella (TAINTED run 1027-034307-2c0a, indicative only), the same 64 MiB allocation was 13.9 ms.

### Rule §6 row B, clause by clause, as written
> TTFT-resume p50 (gap 2–10 min, 25k) ≤ 0.3 s on bella 7B f16 *and* replay md5 == main *and* aggregate / ITL p99
> within 2 % of main at speed 4; else keep the code behind `--cache-ram 0` and record why.

| clause | evidence | verdict |
|---|---|---|
| TTFT-resume p50 ≤ 0.3 s | 0.149 s on the 4090 with bella's pool geometry (the foreman ruled this valid to undraft; the bella cell of record, 1027-122443-5258, is PENDING) | **pass (4090)**; bella PENDING |
| replay md5 == main | replay 1027-122430-5f88 PENDING. Supporting only: every micro-bench reply md5 equal, serve md5 42/42 in every cell | PENDING |
| aggregate / ITL p99 within 2 % at speed 4 | replay 5f88 PENDING. The micro-bench's ITL p99, which the foreman counts as an ITL step: **+21.5 %** (83.44 vs 68.67 ms; every candidate round, 77.2–83.8, above every main round, 67.9–70.8) | **FAIL** |

**Verdict: the rule does not pass.** The code stays behind `--cache-ram 0` (the default) and the PR stays a
draft. Per the pre-registered stop, nothing was tuned after this result.

### Why ITL p99 rose (analysis; the cause is ESTIMATED, the inputs MEASURED)
- **Not the restore.** The micro-bench restores the session before the ITL cell starts, and the ITL cell's
  prompts share no stored prefix. The H2D restore never runs during a measured inter-token gap.
- **Not the write-behind.** It is capped at 16 MiB per step = 4 pages × 0.187 ms ≈ 0.75 ms per step (MEASURED
  page D2H), about 1 % of a 68 ms chunked step. That can't add ~15 ms at p99. ITL p50 didn't move either.
- **Not a forced flush.** Every reclaimed slot had been fully written when its solo request ended.
- **The leading cause: pinned-slab growth inside engine steps.**
  - The arena grows lazily, one 64 MiB slab (18 pages) at a time, and each slab costs **61.8 ms** on this pod.
  - The arena ended at its cap in every candidate round (6,440,878,080 B = 1,755 pages).
  - Before the ITL cell, the store held the 4 sessions, about 418 + 3 × 414 = 1,660 pages ≈ 93 slabs (ESTIMATED
    from the token counts).
  - So the last ≈ 5 slabs (≈ 81 pages) were allocated while the fresh prompt's pages were written behind, inside
    the ITL cell. Each allocation stalls one engine step by about 62 ms, and with 2 decoding streams a stall hits
    2 of the 510 ITL samples.
  - ≈ 5 stalls × 2 = up to ≈ 10 samples, about 2 %, enough to move the p99 (index int(0.99 × 509) = 503, the 7th-largest sample) off the
    chunked-prefill steps (main's p99, ~68 ms) and into the stalled ones.
- **Not confirmed sample by sample.** The bench keeps only the per-round p50/p99. Confirming it needs
  `AXON_SERVE_TRACE` step times, or the ITL tail samples, from the candidate server.
- **The replay will likely hit the same cost.** The replay job 5f88 runs `--cache-ram 20G`, which grows ≈ 325
  slabs × ≈ 62 ms ≈ 20 s of step stalls over a 600 s replay (ESTIMATED). It is very likely to show the same p99
  effect for the same reason.

### What a fix would take (not built)
1. **Allocate the arena eagerly at start** (the size asked for, so no pinned memory the user didn't ask for).
   - About 10 lines in `tier::Arena` plus a banner line.
   - Startup cost ≈ 98 slabs × 62 ms ≈ 6 s on this pod, ≈ 1.4 s on bella (ESTIMATED from the measured
     per-slab cost).
   - No allocation in any step after that. The alternative, pre-growing a slab on idle engine iterations, keeps
     the memory lazy but still stalls steps under sustained load (the speed-4 replay has few idle iterations).
2. **Optionally, skip `host_alloc`'s zeroing for the arena.** Every page slot is written before it is read.
   That shortens the startup cost; it needs a split measurement (pinning vs memset), which doesn't exist yet.
3. **Re-gate.** Rerun the micro-bench with `AXON_SERVE_TRACE` on the candidate (one 4090 timing job, ≈ 16 min)
   to confirm the cause and the fix. Then the replay (5f88's script) and the bella cell.

The H2D restore needs no change for this result. Chunking or overlapping it (a copy stream) would only matter
when a restore lands while other slots decode, which this cell doesn't exercise. The replay measures that case.

## What didn't work
- **The first bella timing run** (1027-034307-2c0a) was TAINTED: desktop browser, foreign GPU max 96 %. It was
  superseded.
- **The first md5 job's serve harness** sent traffic to a loading server (503). Fixed (wait for `/health` and
  `/v1/models` 200) and rerun.
- **Lazy pinned growth inside steps:** the ITL p99 cost above.
