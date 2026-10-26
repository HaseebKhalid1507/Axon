# MoE-resident PR D: the Moonlight bench of record vs llama.cpp (docs PR)

- Date: 2026-10-26. Branch: docs/moe-resident-bench (docs only). Track: roadmap item 3, r16 pick-4 spec PR D.
- Code under test: main `39624db` (`--gpu-experts`, PRs #120 / #122 / #127). Board:
  [benchmarks-gpu.md](../benchmarks-gpu.md#mixture-of-experts-moonlight-16b-a3b-resident-experts-vs-llamacpp-parabellum-2026-10-26).

Labels: **MEASURED** (job cited), **ESTIMATED**, **CODE**.

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

- **Thread-matched head-to-head** (llama.cpp `-t 8` and `-t 24`, interleaved with Axon, N = 0 / 6 / 12): pending
  bella job. The result will be added here as an Update line.
- **Why `auto` trails:** the per-kernel traces (CUPTI) of one Axon decode step and pp512 against llama.cpp's at the
  same placement are pending. They feed the spec for the resident kernels' round 2, which has to keep the twin
  contract (bits equal to the CPU path) under `exact`.
- Qwen3-30B-A3B inherits these kernels. Its step 0 waits on that diagnosis.
