# MoE-resident PR D: the bench of record vs llama.cpp, Qwen3-30B-A3B + Moonlight on an RTX 4090 (docs PR)

- Date: 2026-10-26. Branch: docs/moe-resident-bench (docs only). Track: roadmap item 3, r16 pick-4 spec PR D.
- Code under test: main `8508a75` (#145, #146, #147) on the RTX 4090 pod; the superseded Parabellum boards ran
  main `39624db`, then three TAINTED `8508a75` re-benches. Board:
  [benchmarks-gpu.md](../benchmarks-gpu.md#mixture-of-experts-on-an-rtx-4090-moonlight-16b-a3b-and-qwen3-30b-a3b-vs-llamacpp-2026-10-26-bench-of-record).

Labels: **MEASURED** (job cited), **ESTIMATED**, **CODE**.

## Bench of record: RTX 4090 pod, main 8508a75 (MEASURED, job 1026-215154-40d7, CLEAN)

Qwen3-30B-A3B and Moonlight-16B-A3B, Axon `--gpu-experts` vs llama.cpp `4413a19` `--n-cpu-moe`, 3 interleaved
rounds. Tables and method are on the board; raw data in
[this gist](https://gist.github.com/HaseebKhalid1507/bf2abed2de5b1cc0a1a68510ef46f399).

- **Qwen3-30B-A3B, all on the GPU:** tg128 219.7 vs 252.5 (**×0.87, behind**); pp512 11,022 vs 8,372 (×1.32);
  pp2100 12,212 vs 9,289 (×1.31). Roadmap item 3's kill rule reads "30B-A3B pp2100 < llama.cpp `--cpu-moe` on the same pod → not a headline". This board didn't run `--cpu-moe` (every expert on the CPU); it compared against llama.cpp's **best** placement (`--n-cpu-moe 0`: everything on the GPU), which is stricter. The rule still passes: 12,212 vs 9,289.
- **Moonlight, all on the GPU:** 272.9 vs 285.5 (×0.96), 9,035 vs 11,176 (×0.81), 7,004 vs 12,272 (×0.57). Behind
  on all three.
- **Partial offload, on this EPYC 75F3 pod:** Axon's CPU-expert path is slower than llama.cpp's (Moonlight N = 12 vs
  `-ncmoe 14`: tg 79 vs 96, pp2100 673 vs 1,330). On Parabellum (Core Ultra 9 275HX), Axon led partial-offload decode 1.46–1.68× at llama.cpp's best thread count (TAINTED boards, but consistent across runs), so this gap is not general.
- **Text:** identical across all Axon placements on both models.
- **Superseded run:** 1026-211155-a0f6 had every llama.cpp row at rc 127 (no CUDA 12 libraries on the pod image) and
  still passed. The harness now preflights llama-bench and fails on any rc ≠ 0.
- Everything below this point is the earlier Parabellum work (main 39624db, then TAINTED 8508a75 re-benches),
  kept as history.

## What was measured, and why

The spec's PR D asks for two things. First, Moonlight at `--gpu-experts 0 / 6 / 12 / auto` against llama.cpp
`--n-cpu-moe (26 − N)` at the same placement: tg128, pp512 and pp2100, 3 interleaved rounds, on bella. Second, its
accept rule: `auto` must reach +50 % decode and +100 % pp512 over N = 0. The 4090 N = 26 point is skipped; it needs
spend approval.

## Results (MEASURED, job 1025-225107-e28c, TAINTED overall: max 46 %, mean 3.1 %; 17 of 72 runs tainted)

| N (resident) | tg128 axon / llama | pp512 axon / llama | pp2100 axon / llama |
|---|---|---|---|
| 0 | 54.5 / 31.6 = 1.72× (llama 2 of 3 runs tainted) | 594 / 796 = 0.75× | 590 / 674 = 0.87× |
| 6 | 66.6 / 41.2 = 1.62× | 703 / 997 = 0.70× | 699 / 892 = 0.78× |
| 12 | 74.0 / 51.4 = 1.44× | 1,014 / 1,382 = 0.73× | 839 / 1,203 = 0.70× |
| auto (25 / 25 / 26) | **121.7 / 171.0 = 0.71×** (all clean) | **2,386 / 5,321 = 0.45×** | **1,588 / 4,681 = 0.34×** |

Per-run values and per-run taint are on the board.

- **Accept rule (auto vs N = 0): PASS.** tg128 ×2.23, pp512 ×4.02.
- **Head-to-head: llama.cpp wins at full residency.** Decode is 29 % behind; prefill is 2.2× (pp512) to 2.9×
  (pp2100) behind. Axon's prefill at `auto` *falls* from pp512 to pp2100 (2,386 → 1,588); llama.cpp barely moves
  (5,321 → 4,681).
- **Axon's decode lead at low N** comes from its CPU expert path, and partly from thread count: llama-bench
  defaults to `-t 4`, Axon uses 8 decode / 24 prefill threads. A thread-matched rerun is queued (below).
- **Placement variance:** `auto` placed 25 layers in rounds 1–2 and 26 in round 3, because the desktop's VRAM
  share at load varied. llama.cpp ran each round at that round's count.

## How it was run

`benchd.sh` is in the [gist](https://gist.github.com/HaseebKhalid1507/7c37456c09d035b98d2f2b7d7b21bf51) with
`res.tsv` and the job log; commands and flags are on the board. Engine order alternates by round, with a 5 s pause
before each run, and per-run taint comes from `nvidia-smi pmon` (axon-run's rule).

## What didn't work

- The first submission (1025-222603-40ed) died after 0.2 s. `benchd.sh` had `local … met=$3 … f=…-$met` on one
  line, which is unbound under `set -u`. The foreman split the line and resubmitted it as e28c.

## Open, and next

- **Thread-matched head-to-head** (jobs 1026-011428-edef + 1026-052430-6c71; table on the board): llama.cpp at
  `-t 8` (its best for decode) and `-t 24`, interleaved with Axon, N = 0 / 6 / 12, 3 rounds.
  - Axon still leads decode, 1.68× / 1.52× / 1.46×; llama.cpp leads prefill (Axon at 0.71–0.84×).
  - `-t 24` collapses llama.cpp's decode to 9–17 tok/s.
  - edef is TAINTED (Xorg, max 53 %). Its six runs inside the two FOREIGN_GPU spike windows are replaced by their
    rerun in 6c71, which is CLEAN (max 0 %), and marked ‡ on the board. The first rerun, 8bc8, was tainted again and
    isn't used.
- **Re-bench on main 8508a75: three runs, all TAINTED** (1026-102643-eeb9, 1026-124838-2f63, 1026-205041-3204;
  desktop load: Xorg, kitty, obs; foreign mean 14–25 %).
  - Their accept ratios agree: tg128 ×2.14–2.37, pp512 ×3.73–4.88. None is a record.
  - The 39624db tables are superseded.
  - The clean record is the 4090 pod board; a quiet-night bella rerun is queued by the foreman.
- **Why `auto` trails:** CUPTI traces of both engines (jobs edef, 8bc8; kernel durations GPU-side) put the gap in
  the MLA kernels (`mla_heads_mv`, `mla_attn`), not in the twin experts. Decode: +1.38 of +2.07 ms busy. Prefill:
  68–82 % of Axon's GPU time. #145 took the decode half; the prefill half is open. Exactness costs about 4 % of a decode step
  (ESTIMATED) and nothing measurable in prefill.
- Qwen3-30B-A3B residency is #146; its step 0 went ahead on per-layer records (pruned 30B-A3B).
