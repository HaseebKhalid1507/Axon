# cuda/mla: faster MLA prefill (batched heads GEMM, shared K tiles), same bits  (PR #147)

- Date: 2026-10-26. Branch: perf/mla-prefill (stacked on #145; main 680a2c5 plain-merged in). Status: open. Code 2852486 (main 3f04dbc merged in); every gate green; the kill rule MET
  in a clean window. Review: shady 8/10 APPROVE (round 2).
- Pick / design: spike's round-2 plan (`~/.cache/axon-moeres/spec-res-r2.md`, S2.5). His trace 8bc8: at pp512
  `mla_heads_mv` + `mla_attn` take 159 of 232 ms of GPU busy time; at pp2100, 1,170 of 1,420 ms. Kill rule, fixed before
  building: pp2100 at `--gpu-experts auto` ≥ 2,800 tok/s (0.6 × llama.cpp's 4,680), in clean windows.

## The S2 decode kill rule (#145), from the clean rerun's job 1026-054428-1455
Job 1455 (`perf3.sh`, `--timing bella`, main be13f1f vs 2be8bfe vs llama.cpp `-t 8 -ncmoe 1`, 3 rounds, order rotated).
**TAINTED:** foreign GPU max 82 %, mean 13.0 %; the max is one kitty/Xorg spike at 07:51, near the end.

| round | short tg128: base / cand / llama.cpp | long tg128 (after pp2101): base / cand / llama.cpp `-d 2048` | pp2101: base / cand |
|---|---|---|---|
| 1 | 120.73 / 150.02 / 177.99 | 118.61 / 139.48 / 168.89 | 1,510.9 / 1,834.6 |
| 2 | 121.44 / 150.96 / 173.28 | 118.70 / 139.85 / 169.53 | 1,526.5 / 1,847.7 |
| 3 | 126.27 / 152.00 / 173.00 | 118.91 / 139.03 / 150.05 | 1,509.2 / 1,844.3 |
| median | 121.4 / **151.0** / 173.3 | 118.7 / 139.5 / 168.9 | 1,510.9 / 1,844.3 (+22 %) |

**Verdict:** ≥145 tok/s clause MET on a tainted run: all 3 rounds ≥150, median 151.0. Foreign load can only slow our
arm, so the absolute clause holds a fortiori. Ratio 0.87× llama.cpp, same run, also tainted. No clean window has been
obtained on bella's desktop.

## What changed
- `mla_heads_mm`: the batched per-head products (absorb `Wk_b q_nope`, expand `Wv_b o`) as a register-tiled f32 GEMM.
  The 32 lane chains per output are computed in bit-reversed order and folded into main's xor-tree shape, so the bits
  are the same by construction. The weights come from staged rows and scale tables, reused over 64-token tiles. It runs
  when the tiles are full enough: nb / ceil(nb / 64) ≥ 32 tokens per tile for k_b (128 columns), ≥ 58 for v_b (512);
  otherwise `mla_heads_mv` runs.
- `mla_attn_b` (sm_80+): one split per block for 32 (token, head) pairs. The K tile is converted to f32 once and
  shared by all of them; every pair keeps main's per-tile operations in main's order.
- main's kernels are kept as `*_ref` in test builds. `mla_heads_mm_equals_main`, `mla_attn_b_equals_main` (+ 128
  heads), `heads_mm_dispatch`, and `mla_prefill_kernel_bench`.

## How it was verified (2852486)
| gate | result | job |
|---|---|---|
| kernel bits vs main's kernels + bench | 12/0/0 sm_89; 21/0/0 sm_120 (with ds2 decode == prefill == verify, Moonlight); 4/0/0 sm_61 | 1026-091354-524e, 1026-091354-11d9, 1026-091355-4cf4 |
| Moonlight md5 vs main 3f04dbc (0, auto): md5ml 14/14 (incl. a 3,961-token prompt), cross-512 gen 6/6, serve 4/4 | identical | 1026-091355-5af9 |
| fast tier debug / release | 708/0/52, 690/0/52 | 1026-091356-7d02, 1026-091356-da6b |
| clippy -D warnings, rustfmt, nvcc 0 spills | clean | local |

## Results
Kernel level (`mla_prefill_kernel_bench`, vs main 680a2c5's kernels, 3 interleaved rounds, median), 2,100 tokens, bella
sm_120: heads k_b 3,930 → 1,964 µs, v_b 3,273 → 2,046 µs, attention 19,576 → 11,223 µs (avante sm_89: 11,130 → 5,163,
8,448 → 5,445, 54,748 → 33,018).

End to end, job 1026-091356-73bb (`--timing bella`, **clean**: foreign GPU 0 %). Moonlight `--gpu-experts auto`
placed 26/26 expert layers on the GPU here, so llama-bench ran `-t 8 -ngl 99 -ncmoe 0`. Placement depends on the VRAM
free at the time: #145's runs got 25/26 (`-ncmoe 1`), so its 0.87× decode ratio and this prefill ratio aren't
comparable. Axon's prompts are 509 / 2,105 tokens, llama-bench's 512 / 2,100.

| round | pp2105 base / cand | llama pp2100 | pp509 base / cand | llama pp512 | tg128 base / cand / llama |
|---|---|---|---|---|---|
| 1 | 2,129.4 / 3,212.6 | 4,988.1 | 3,415.9 / 4,242.9 | 5,017.8 | 158.8 / 157.8 / 193.7 |
| 2 | 2,130.4 / 3,211.7 | 4,703.9 | 3,423.7 / 4,248.3 | 5,252.3 | 149.0 / 158.0 / 170.7 |
| 3 | 2,136.6 / 3,168.5 | 4,709.5 | 3,421.5 / 4,327.4 | 5,226.3 | 156.7 / 158.1 / 183.6 |
| median | 2,130 → 3,212 (+51 %) | 4,710 | 3,422 → 4,248 (+24 %) | 5,226 | 156.7 → 158.0 |

Kill rule (pp2100 ≥ 2,800 tok/s and ≥ 0.6 × llama.cpp, clean): **MET**. 3,212 = 0.68 × llama.cpp's median 4,710
(0.64 × its best round, 4,988).

## What didn't work / follow-ups
- `mla_attn_b` with 256 threads and 4 pairs per score warp: slower on both GPUs (3 interleaved rounds: avante 42.2 vs
  33.0 ms, bella 18.5 vs 11.2 ms at 2,100 tokens).
- The GEMM at nb ≥ 16 (round 1): slower than `mla_heads_mv` below its crossover. Replaced by the tokens-per-tile rule.
- The v_b fill (58) is sm_120's crossover; sm_89's is ~49. On avante the rule keeps `mla_heads_mv` at 48 / 96 / 112
  tokens where the GEMM is 3 / 4 / 16 % faster. That's a missed win, not a regression; a per-architecture fill is a
  follow-up.
