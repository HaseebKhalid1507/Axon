# The FP8 gate off the GPU test lock: GPU fast vs GPU exact in the suites, the run of record nightly (#232)

- Date: 2026-11-13. Branch: dec/fp8-graded-lock. Base: main 4f347a3. Status: open
- Pick / design: #232 (filed from #225's gates). Lead rulings 11-13: the CPU reference before the lock; the
  1000-position grade nightly and by hand on FP8 PRs; after the sizing below, the always-on test is GPU fast vs GPU
  exact (sanity only), and a failing run of record means look, not block.

## What changed
Test-only, plus docs.
- **`gpu_fp8_fast_vs_exact_sanity` (new, every suite).**
  - The FP8 0.5B checkpoint's fast mode against its exact mode (the Q8_0 form, D1) on the same GPU, every arm of the
    #173 walk at `Sizes::FP8` (>= 1000 positions an arm). The floor (exact with a q8 KV cache) is printed only.
  - No statistics: `assert_sane_vs_exact` (finite logits, max |dlogit| < 12), a fast kernel launched in every arm,
    and the routes (the FP8 tensor-core GEMM for every batched prefix, `ATT_NB_MIN` for the load arm).
  - Its doc comment says what it can't catch: a bug fast and exact share (loader, FP8 scales, head, attention), both
    modes drifting from true FP8 math, and accuracy losses inside the |dlogit| bound.
  - It replaces `gpu_fp8_graded_vs_fp8_ref`. The debug build's `Sizes::FP8_DEBUG` run is gone, and so is the
    provisional `Sizes::FP8_FAST`: both builds run the new test.
- **`gpu_fp8_graded_vs_fp8_ref_1000` (`#[ignore]`, the run of record).** It grades against the CPU's fp8-ref with
  the statistical bars, unchanged.
  - Its CPU nets run first, without the GPU test lock (`numerics::reference`), and keep the reference's logits. It
    then takes the lock for the GPU half only (`Reference::grade`).
  - `testutil::gpu_cc` tells beforehand, without the lock, whether the GPU half will run.
  - It runs nightly on main (axon-nightly, devstack) and by hand on PRs that touch the FP8 kernels or their routing
    (docs/testing.md, PR template row).
- **`graded_in_two_passes_is_graded_at_once` (new).** `reference` + `Reference::grade` give `grade_sized`'s `Graded`
  field for field, f64 sums bit for bit. It uses three GPU nets of the checkpoint (exact, exact q8 KV, fast) on every
  arm and branch of the walk.
- `walk()` is the one skeleton for both. The interleaved `grade_sized` makes the same calls in the same order as
  before for every caller.

## Why
#232: the old test took the lock in its first line and kept it through its CPU nets. On packed pods that was over
2700 s, and 138 GPU tests of #225's suite failed on the lock timeout behind it.

The CPU-first restructure fixes the lock. The probe below holds the lock for 11.9 s, where the old test held it for
45+ minutes. But the CPU half itself was 33 minutes at about half the positions, so no statistical grade fits an
always-on suite:
- **Why the CPU reference is slow (#244).** The CPU's float matmul (F32, F16, the FP8 checkpoint's F8S) is a
  serial scalar dot: `dot_row`'s `mulss`/`addss` chain, one accumulator. It re-dequantizes every row for each token
  of a batch.
  - In ef0b's phase timing the two CPU nets cost the same: the reference 1966.8 s, the floor 1972.6 s, the GPU net
    15.0 s, the tallies 73.7 s, of 4031 s.
  - Counted work: 13.76 T MAC in 3,939 s, 3.5 G MAC/s on 5 threads. 6d27 gives 4.67 G MAC/s on 4 threads.
- **Cost by size** (work counted from `walk()`, per CPU net: body tokens × 357.8 M MAC, head rows × 136.1 M MAC;
  120 s on a packed pod buys 0.42–0.56 T MAC). The cheapest all-arm CPU-reference run (one window an arm) is
  600–800 s. Even one net, no verify rows, one window an arm, is 200–270 s.
- **False fails with no bug** (#245). These are today's bars on windows drawn from ef0b's per-position dump, 2000
  draws a size:

  | windows dec/bat/cac, load steps | drawn without replacement | bootstrap |
  |---|---|---|
  | 1/1/1, 2 | 55.4 % | 55.4 % |
  | 9/4/11, 13 (the provisional `FP8_FAST`) | 16.9 % | 35.4 % |
  | 15/6/17, 20 | 1.4 % | 34.8 % |
  | 19/7/21, 25 (`FP8`) | the run (PASS) | 34.7 % |

  No size fits both 120 s and 1/1000. The run of record's own bars flag about 1 in 3 harmless numerics changes, so
  its failure reads as look, not block (docs/testing.md).

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| the CPU-first test at a provisional ~520 positions an arm, packed pod1 (`--nocapture`) | pass; CPU reference 1980.7 s without the lock, 0.1 s waiting, GPU half 11.9 s holding it | 1113-013544-6d27 |
| the measure build (phase timing + per-position dump), `Sizes::FP8`, packed pod2 | pass, every arm; 4031 s | 1113-013522-ef0b |
| build, clippy `-D warnings`, rustfmt | see the PR's gates table | — |
| fast tier release (the whole suite, nothing skipped), debug suite, the equivalence test | see the PR's gates table | — |

## What didn't work
- **Grading a smaller sample against the CPU in the suites** (the provisional `FP8_FAST`, about 520 positions an
  arm). It costs 33 min of CPU on a packed pod, and it false-fails 17–35 % of the time (table above).
- **A pod for the equivalence test.** Two pod2 runs failed in 0.2–0.3 s on "FP8 checkpoint … is missing"
  (1113-014516-4219, 1113-014736-b647): pod2's test-binary tree lacked the checkpoint. The rerun is pinned to bella.
  A devstack ask (the forum) adds the new tests to axon-run's `CKPT_TESTS`.

## Follow-ups
- #244: the CPU float matmul (needs a spec). It is a bit-preserving SIMD across rows; after v0.1.0 on the roadmap.
- #245: the graded bars, redesigned around paired, window-clustered discordance (needs a spec; no threshold changes
  now).
- devstack (case): the nightly `nightly-<sha7>-fp8gate` job; `CKPT_TESTS` entries for
  `gpu_fp8_fast_vs_exact_sanity`, `graded_in_two_passes_is_graded_at_once` and `gpu_fp8_graded_vs_fp8_ref_1000`.
