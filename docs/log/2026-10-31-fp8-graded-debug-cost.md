# The FP8 graded gate's cost in debug builds  (PR #179)

- Date: 2026-10-31. Branch: fix/fp8-graded-debug-cost. Base: main fde94b6. Status: open
- Pick / design: lead, 10-31, after A4's debug shard 1/3 (1031-080835-d64f) failed on the GPU test lock

## What changed
Test-only. `cuda::qwen2::tests::fp8::gpu_fp8_graded_vs_fp8_ref` (#177) grades at `Sizes::FP8` (>= 1000 positions
per arm) in a release build, as before, and at a new `Sizes::FP8_DEBUG` in a debug-assertions build: every arm,
every batched prefix (24 / 160 / 520) and route, the load arm's pool at 32 rows (`ATT_NB_MIN`, the pool's
tensor-core attention), on 150 / 147 / 98 / 128 positions. Its per-arm position assertion is 1000 in release, 98
in debug.

**The split (5b4f24a, lead ruling 10-31).** The statistical bars (`Top1::Binomial`'s top-1 floor, the KL mean,
the outlier count, the verify rows) are graded in the release build only, unchanged, at >= 1000 positions per arm:
the run of record. A debug-assertions build doesn't grade them: it runs every arm and route for the checks only it
makes, asserts a fast kernel ran in every arm, and keeps the sanity backstops on the net under test
(`assert_sane`: every logit finite, and the whole-vocabulary max |dlogit| < 12). It prints the grading numbers
labelled "debug build: info only, not graded". Finiteness is a new count, `Agree::nonfinite`: `max_abs` can't
show a NaN, since `f32::max` drops it. Unit test: `cuda::qwen2::tests::sane_backstops`.

## Why
At `Sizes::FP8` the test held the GPU test lock for over 2700 s (`AXON_TEST_GPU_LOCK_TIMEOUT`) in A4's debug shard
1/3 on bella (1031-080835-d64f, debug binary of fe20964, 4 test threads, job 3,078 s): the 60 GPU tests queued
behind it failed on the lock timeout (testutil.rs:372); the test itself passed. Alone (`--test-threads 1`, #177's
fp8tests.sh, 1030-031016-38fe) all 13 FP8 tests together took 1,476 s in release and 1,656 s in debug, so the
gate was always most of that and near the timeout once it shares the CPU. The cost is the two CPU nets (the
fp8-ref and the e4m3-activation floor) over about 14k tokens each; the debug size runs about a quarter of them.

Why a smaller debug size and not `#[cfg_attr(debug_assertions, ignore)]`: a debug build checks things on these
paths that a release build never does: every kernel launch's argument count against the kernel's
(cuFuncGetParamInfo, `cuda/mod.rs`), the pool's page table (`slots.rs`), the fast counter's assertion, overflow.
Ignoring the test would drop those checks for the FP8 fast routes (the batched GEMM, the pool steps). A
larger debug run adds nothing statistical: the debug build adds checks, not other arithmetic, so it would grade
the same numbers as release. The release run (>= 1000 positions per arm) stays the binding accuracy gate.

Why the debug build stopped grading (the motivating evidence): 567edfe's first debug run of the test alone,
**1031-115732-b712**, failed on the bars, not on a check: `0.5B FP8 cuda fast vs fp8-ref [batched prefill]: top-1
agreement 137/147 (floor 142, 2 SE 4.4)` (288.6 s). This was the risk the first version named: the binomial
rule's allowance at 147 positions is 4.4 tokens, and a 5-token gap at that size is noise. The bars are a
statistical test sized for >= 1000 positions; at 98-150 they would either fail on noise or have to be loosened.
The lead ruled not to loosen them: the debug job checks what only debug checks, the release job grades.
The release run backs this (MEASURED, 9e47, the same batched arm at 1029 positions): the fast path agrees on
974/1029 (94.7 %), its floor on 970/1029 (94.3 %). At n = 147 and p = 0.947, SE ≈ 2.7 tokens: b712's 137/147
(93.2 %) is 0.8 SE below the fast path's own release rate (139.1 expected). The outlier was the debug **floor**:
142/147 (96.6 %) against its release rate of 94.3 % (138.6 expected), 1.2 SE above. The small subset drew a lucky
floor; the fast path was in line.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| the test alone, debug, 567edfe (grading at 147 positions) | **FAILED** on top-1, batched arm 137/147 vs floor 142 (2 SE 4.4), 288.6 s: the motivating evidence for the split | 1031-115732-b712 |
| A4's debug shard 1/3 filter list, 567edfe | killed: the old binary would fail the same test | 1031-115733-027c |
| the test alone, release, 567edfe (release path unchanged by 5b4f24a) | **PASS**, 1,362 s; every arm graded (batched 974/1029 vs floor 970) | 1031-115732-9e47 |
| the test + `sane_backstops` alone, debug, 5b4f24a (`--nocapture`, wall time) | **PASS**, 2/2, 259 s | 1031-120743-e8dc |
| A4's debug shard 1/3 filter list (d64f's argv), default threads, 5b4f24a | **PASS**, 234 passed / 0 failed / 13 ignored, 1,581 s, no lock timeouts | 1031-120743-fac0 |
| CI | green at 13c6e32 | GitHub Actions |
| clippy -D warnings (lib + tests), rustfmt | clean at 5b4f24a | local |

## Results
- The test's debug wall time: before, over 2,700 s holding the GPU test lock in d64f (4 threads; 60 GPU tests
  failed on the lock timeout); after, **259 s** alone (e8dc).
- The shard that broke (d64f's filter list, 4 threads): 234 passed / 0 failed in 1,581 s, no lock timeouts (fac0).
- Release unchanged: 1,362 s alone, the binding accuracy run (9e47).

## What didn't work
567edfe's first version kept the statistical bars in debug at the smaller size; it failed on noise (b712, above).

## Follow-ups (shady's SHOULDs, 10-31; test code, so not in this PR: they would force a re-gate)
- Keep the top-20 max |dlogit| < 6 bound in `assert_sane` (a per-position bound, not a statistic; e8dc's worst
  was 3.43, release's 4.09).
- Assert `nonfinite == 0` in `assert_graded_with` too, so release catches a NaN explicitly rather than through
  `sum_kl <= 2 * floor` happening to be false.
- State the `MOE_DEBUG` exception next to "same arithmetic" (the MoE router module only; not on this path).
