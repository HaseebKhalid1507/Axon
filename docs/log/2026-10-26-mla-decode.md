# cuda/mla: faster MLA decode kernels, same bits (S2.2 + S2.3)  (PR #145)

- Date: 2026-10-26. Branch: perf/mla-decode. Base: main be13f1f. Status: open. Code 2be8bfe (round 2, Shady's review 1); bit gates green; the perf run of record is TAINTED, clean rerun pending.
- Pick / design: spike's round-2 plan (`~/.cache/axon-moeres/spec-res-r2.md`, S2.2 and S2.3, the bit-neutral
  variants). The CUPTI diagnosis there: at Moonlight `--gpu-experts auto` on bella, a decode step takes 8.25 ms in
  Axon and 6.57 ms in llama.cpp. `mla_heads_mv` accounts for +1.08 ms of the gap and `mla_attn` at short positions
  for +0.30 ms.

## What changed
- `mla_heads_mv` (`src/cuda/mla.cu`):
  - A block of rpb rows copies three things into shared memory first: the rows' bytes (16-byte loads where
    aligned), the per-block factors deq computes for every element, and its tokens' inputs. The factors are Q5_0 /
    Q8_0 `d`, Q4_K `d1 = d * sc` and `m1 = dmin * m`, and Q6_K `ds`; each is computed once per block, by the same
    expression deq uses.
  - Each weight is then deq's last float step on those factors.
  - The type switch is resolved at compile time. The extra types call `deq` itself.
  - A lone token (the decode step) skips the 7 idle chains and trees. It runs 4 rows per warp with their 4 xor
    trees transposed: 6 shuffles instead of 20, the same sums.
  - rpb: the largest power of two up to 32 with at most 4.6 KB of row bytes (Moonlight: k_b 32 rows, v_b 16).
- `mla_attn`:
  - Heads per block (hpb) are chosen per token by the work: the largest power of two up to 8 whose block count
    (n_h / hpb × splits × tokens) still reaches the SM count. A short position or a lone decode token gets more
    blocks with fewer heads each.
  - A flat grid covers every token (`attn_blocks`, proven by a test).
  - With fewer than 8 heads per block, all 8 warps score rows, through shared memory. With 8, main's own scoring
    loop runs.
  - A decode step below 4 splits (position < 512) double-buffers its K tiles with cp.async (sm_80+, 2 tiles of
    shared memory). It has its own set of decode graphs, because the host picks this by position.
  - Split points, the tile size and every per-head float operation are unchanged.
- main's two kernels are kept as `mla_heads_mv_ref` / `mla_attn_ref` in test builds only (`src/cuda/mla_ref.cu`,
  `Kernels::mla_ref_module`).

## Bits
No float operation changes, and neither does its order or its operands:
- **`mla_heads_mv`:** each output is the same per-lane fma chain over c = lane, lane + 32, … with the same weight
  value, then the same xor tree. The transposed tree pairs the same lanes in the same order.
- **`mla_attn`:** hpb only decides which block and threads compute a head. A row's score is the same chain and tree,
  read back as lane r = row r. Each column pair is the same rescale and the same fma over the tile's rows in order.
  Double buffering changes when bytes arrive, not what they are.

Tests that compare bits with main's kernels:
- `mla_heads_mv_equals_main`: types Q5_0, Q8_0, Q4_K, Q6_K, F16, F32, and Q4_0 (deq at run time); Moonlight's shapes
  and odd ones; 1–17 tokens; rows per block 1, 3, 8, 16, 32 and the host's; weights at 16-, 4- and 2-byte alignment.
- `mla_attn_equals_main`: 16-, 64-, 8- and 2-head shapes; a batch over 6 splits; lone tokens at 14 positions read
  from P; every hpb mode; with and without double buffering; partials included.
- `attn_blocks_cover_the_hpb_rule`.

## Why / Results
Kernel times (`mla_decode_kernel_bench`, one token, Moonlight shapes, a 4096-row cache's 32 splits launched; host
timing of 400 back-to-back launches, MEASURED):

| kernel | avante sm_89: main → cand µs | bella sm_120 (GPU shared with another job, at 22cfa18: same kernels, before the transposed tree): main → cand µs |
|---|---|---|
| heads_mv k_b (Q5_0 8192 × 128) | 49.0 → 16.7 | 20.7 → 7.2 |
| heads_mv v_b (Q4_K 2048 × 512) | 55.0 → 12.5 | 20.1 → 5.3 |
| attn pos 8 | 8.2 → 5.7 | 6.0 → 3.1 |
| attn pos 64 | 34.1 → 8.0 | 27.9 → 6.8 |
| attn pos 136 | 51.9 → 11.9 | 48.1 → 9.7 |
| attn pos 300 | 52.2 → 16.1 | 46.3 → 10.1 |
| attn pos 1000 (no dbuf) | 52.5 → 33.6 | 45.2 → 21.8 |
| attn pos 4000 (no dbuf) | 104.2 → 104.3 | 61.0 → 55.2 |

End to end (job 1026-042831-fd06, `--timing bella`, **TAINTED**: foreign GPU max 32 %, mean 10.6 %; arms interleaved, 3
rounds, medians): Moonlight `--gpu-experts auto`, main be13f1f → 2be8bfe, llama.cpp `-t 8 -ncmoe 1`:

| point | base | cand | Δ | llama.cpp | cand / llama |
|---|---|---|---|---|---|
| tg128, 6-token prompt | 115.2 | 143.7 | +24.8 % | 169.3 | 0.849 |
| tg128 after a 2,101-token prompt | 114.9 | 137.4 | +19.6 % | 163.8 (`-d 2048`) | 0.839 |
| pp2101 | 1,461 | 1,782 | +22.0 % | | |

Per decode step (CUPTI, same job): `mla_heads_mv` 1.361 → 0.417 ms, `mla_attn` 0.897 → 0.208 ms; busy 7.78 → 6.25 ms, span
8.76 → 7.23 ms. No visible hitch at position 512 (the second graph set's capture): the steps there are within the spread
(median 7.08 ms, the longest anywhere 7.81 ms at 547).

Per round (TAINTED), short tg128 base / cand / llama: 115.15 / 142.77 / 159.87, 119.05 / 147.16 / 172.87,
114.97 / 143.74 / 169.30. Long tg128: 114.94 / 138.79 / 163.76, 115.24 / 135.32 / 163.18, 114.32 / 137.40 / 163.81.
pp2101 base / cand: 1,465.9 / 1,781.5, 1,452.9 / 1,791.8, 1,461.4 / 1,774.0.

Kill rule (≥ 145 tok/s, 0.85 × llama.cpp's 171): **NOT MET on a tainted run.** Median 143.7 < 145, and 2 of 3 rounds are below
(142.8, 147.2, 143.7). Clean rerun 1026-054428-1455 pending; its result is recorded here whichever way it lands.

Prefill: batched passes run the same rewritten kernels (the staged NT = 8 rows, hpb by work, no double buffering), with the
same bits (`*_equals_main` at nb > 1; 739c: decode == prefill == verify). pp2101 +22 % (tainted).

## What didn't work
- Double buffering at every position: the 2-tile shared memory leaves one block per SM. At 1000 and 4000 that cost
  33.6 → 44.7 and 104 → 135 µs on sm_89, so it only applies below 4 splits.
- hpb 1 everywhere: best at the shortest positions, 2× slower at 4000. Hence the rule.
- A first heads_mv rewrite with rows in a dynamically indexed array: it spilled to local memory and was slower than
  plain staging. A `__launch_bounds__(256, 3)` cap on mla_attn spilled (24 B) and lost 20 % at long positions.

## How it was verified (2be8bfe)
| gate | result | job |
|---|---|---|
| mla kernel bit tests vs main's kernels | 7/0/1 (sm_89); 2b18a69: 7/0/1 sm_61, 16/0/1 sm_120 (2a4b) | 1026-042321-dd18; 1026-040422-0963 |
| decode == prefill == verify, resident == CPU (ds2 fixture + Moonlight) | 16/0/1 | 1026-042322-739c |
| fast tier release / debug | 683/0/51, 701/0/51 | 1026-042323-541b, 1026-042323-6fb7 |
| Moonlight md5 vs main (0, auto): md5ml 12/12, cross-512 gen 6/6 (prompts of 449–487 tokens, -n 128, incl. `--draft 7`), serve across 512 4/4 | identical | 1026-042322-ef23 (gen), 1026-044417-c660 (serve) |

Harness note: ef23 exited 1 because its serve step called `axon serve --gpu-experts`, which serve doesn't take (placement is
`AXON_GPU_EXPERTS`). Its gen pairs are 6/6 identical; c660 is the fixed serve rerun. sm_120 bits at 2be8bfe come from those
Moonlight runs and 739c; the kernel bit tests ran on sm_89 (dd18). The 2b18a69 → 2be8bfe kernel delta is `__align__(16)` only.
| clippy -D warnings, rustfmt, nvcc 0 spills | clean | local |
