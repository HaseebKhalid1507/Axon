# The FP8 gate off the GPU test lock: GPU fast vs GPU exact in the suites, the run of record by hand (#232)

- Date: 2026-11-13. Branch: dec/fp8-graded-lock. Base: main 4f347a3. Status: open
- Pick / design: #232 (filed from #225's gates). Lead rulings 11-13: the CPU reference before the lock; the
  1000-position grade nightly and by hand on FP8 PRs; after the sizing below, the always-on test is GPU fast vs GPU
  exact (sanity only), and a failing run of record means look, not block. Lead rulings on shady's review (7/10,
  comment 6849413937): by hand only until devstack's nightly is deployed; the run of record on sm_120 and sm_89;
  the sanity test gets loose deterministic bars (the "sanity only" ruling revised); look-not-block until #245.

## What changed
Test-only, plus docs.
- **`gpu_fp8_fast_vs_exact_sanity` (new, every suite).**
  - The FP8 0.5B checkpoint's fast mode against its exact mode (the Q8_0 form, D1) on the same GPU, every arm of the
    #173 walk at `Sizes::FP8` (>= 1000 positions an arm). The floor is exact with a q8 KV cache.
  - No statistics: `assert_sane_vs_exact` (finite logits, max |dlogit| < 12), a fast kernel launched in every arm,
    and the router's table (`f8_route` gives the FP8 tensor-core GEMM for every batched prefix's size, `ATT_NB_MIN`
    for the load arm).
  - **Loose bars (review round 1):** top-1 >= 80 % an arm, KL mean <= 3× the floor's, verify rows >= 85 %
    (`numerics::loose_bars`, a CPU unit test `loose_bars_rules`). b356 on a 4090: top-1 90.5–94.8 %, KL means
    0.73–1.56× the floor's, rows 92.9 %. Deterministic for a build on a GPU, so they don't flake; the bound of 12
    alone only caught garbage (decode's max was 8.65 on a correct build).
  - Its doc comment says what it can't catch: a bug fast and exact share (loader, FP8 scales, head, attention), both
    modes drifting from true FP8 math, accuracy losses inside the loose bars, and which kernels ran (the route
    checks read the router's table; the batched and cached arms' launches are met by their decode steps). Kernel
    precision on every PR is `gpu_fp8_gemm_bits_and_anchor` and `gpu_fp8_gemv_bits_and_anchor`.
  - It replaces `gpu_fp8_graded_vs_fp8_ref`. The debug build's `Sizes::FP8_DEBUG` run is gone, and so is the
    provisional `Sizes::FP8_FAST`: both builds run the new test.
- **`gpu_fp8_graded_vs_fp8_ref_1000` (`#[ignore]`, the run of record).** It grades against the CPU's fp8-ref with
  the statistical bars, unchanged.
  - Its CPU nets run first, without the GPU test lock (`numerics::reference`), and keep the reference's logits. It
    then takes the lock for the GPU half only (`Reference::grade`).
  - `testutil::gpu_cc` tells beforehand, without the lock, whether the GPU half will run.
  - It runs by hand on PRs that touch the FP8 kernels, their routing or the FP8 load path, on sm_120 (bella, the e4m3
    `mma.sync`) and sm_89 (a 4090 pod, two f16 mmas: #190's path). Not nightly yet: devstack's axon-nightly has no
    job for it (asked 11-13); docs/testing.md says nightly once it's deployed.
  - Its route asserts (pure) run before its CPU half, so a routing regression fails at once, not 30–60 min in.
- **`graded_in_two_passes_is_graded_at_once` (new).** `reference` + `Reference::grade` give `grade_sized`'s `Graded`
  field for field, f64 sums bit for bit. It uses three GPU nets of the checkpoint (exact, exact q8 KV, fast) on every
  arm of the walk, with the cached arm on windows of its own (`cached_windows: Some`, as `Sizes::FP8`); the `None`
  branch is only the one-pass `grade`'s and isn't run two-pass. It assumes fast mode is deterministic run to run
  (its doc comment: a fast kernel with atomics would fail it as if the passes differed). Since review round 1,
  `Reference::grade` also asserts that every kept verify pick was read.
  - **Update: its first gate run caught a difference** (release shard 1, 1113-031552-6419, pod1; the debug run
    1113-032053-efeb too). Each arm's floor `Agree::last_kl` (the previous position's KL, a sequence's running state)
    ended at its last position's KL in one pass and at 0.0 in two. Nothing else differed. The cause was
    `ArmGrade::start`, which reset both nets' `last_kl` at every sequence start in every pass, so `Reference::grade`'s
    pass zeroed the floor's, which only the `reference` pass tallies. `walk` now resets only the nets the pass runs
    (`Nets::start`), so each pass leaves the other nets' tallies as it found them.
  - No graded value depended on it. `last_kl` is read only by the same net's next tally, as an `over_half` entry's
    previous-position KL, and the floor's tallies all happen in the `reference` pass with the same resets as in one
    pass. The earlier runs' verdicts (6d27, ef0b) stand.
- `walk()` is the one skeleton for both. The interleaved `grade_sized` makes the same calls on each net in the same
  order as before for every caller. Across nets one order moved: main allocated all three verify caches and then ran
  the three verifies; now each net allocates and verifies in turn (no bits depend on allocation addresses).

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

  No size fits both 120 s and 1/1000. A bootstrap suggests the run of record's own bars flag about 1 in 3 harmless
  numerics changes, so until #245 lands its failure reads as look, not block (docs/testing.md).

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| the CPU-first test at a provisional ~520 positions an arm, packed pod1 (`--nocapture`) | pass; CPU reference 1980.7 s without the lock, 0.1 s waiting, GPU half 11.9 s holding it | 1113-013544-6d27 |
| the measure build (phase timing + per-position dump), `Sizes::FP8`, packed pod2 | pass, every arm; 4031 s | 1113-013522-ef0b |
| build, clippy `-D warnings`, rustfmt | see the PR's gates table | — |
| the equivalence test, first run (head 54f1005) | fail: the floor's `last_kl` (above); fixed in `walk` | 1113-031552-6419, 1113-032053-efeb |
| fast tier release (the whole suite, nothing skipped), debug suite, the equivalence test | see the PR's gates table | — |
| review round 1 (the loose bars, the route split, the `picks` assert): release shard 1, debug FP8 on a GPU | see the PR's gates table | — |
| the new tests on sm_120, and the run of record at head on sm_120 and sm_89 | see the PR's gates table | — |

## What didn't work
- **Grading a smaller sample against the CPU in the suites** (the provisional `FP8_FAST`, about 520 positions an
  arm). It costs 33 min of CPU on a packed pod, and it false-fails 17–35 % of the time (table above).
- **A pod for the equivalence test.** Two pod2 runs failed in 0.2–0.3 s on "FP8 checkpoint … is missing"
  (1113-014516-4219, 1113-014736-b647): pod2's test-binary tree lacked the checkpoint. A rerun pinned to bella
  (1113-032053-d70b) was dropped while bella was held: release shard 1 ran the test on pod2 instead
  (1113-035842-6480, PASS), since the shard's filters stage the checkpoint. A devstack ask (the forum) adds the new
  tests to axon-run's `CKPT_TESTS`: until then a `--exact --ignored` run of `_1000` on a pod stages no checkpoint.
- **The docs first called the nightly live** (shady, review round 1): devstack's axon-nightly had no FP8 job. They
  now say by hand only until it's deployed.

## Follow-ups
- #244: the CPU float matmul (needs a spec). It is a bit-preserving SIMD across rows; after v0.1.0 on the roadmap.
- #245: the graded bars, redesigned around paired, window-clustered discordance (needs a spec; no threshold changes
  now).
- devstack (case): the nightly `nightly-<sha7>-fp8gate` job; `CKPT_TESTS` entries for
  `gpu_fp8_fast_vs_exact_sanity`, `graded_in_two_passes_is_graded_at_once` and `gpu_fp8_graded_vs_fp8_ref_1000`;
  mergeready asking for the run of record's job IDs when a diff touches the trigger list (docs/testing.md).
