# The FP8 graded gate's cost in debug builds  (PR #TBD)

- Date: 2026-10-31. Branch: fix/fp8-graded-debug-cost. Base: main fde94b6. Status: open
- Pick / design: lead, 10-31, after A4's debug shard 1/3 (1031-080835-d64f) failed on the GPU test lock

## What changed
Test-only. `cuda::qwen2::tests::fp8::gpu_fp8_graded_vs_fp8_ref` (#177) grades at `Sizes::FP8` (>= 1000 positions
per arm) in a release build, as before, and at a new `Sizes::FP8_DEBUG` in a debug-assertions build: every arm,
every batched prefix (24 / 160 / 520) and route, the load arm's pool at 32 rows (`ATT_NB_MIN`, the pool's
tensor-core attention), on 150 / 147 / 98 / 128 positions. Its per-arm position assertion is 1000 in release, 98
in debug. The grading bars (`Top1::Binomial`, KL, outliers) are unchanged in both.

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

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| the test alone, debug (`--nocapture`, wall time) | PENDING | |
| the test alone, release | PENDING | |
| A4's debug shard 1/3 filter list (d64f's argv), default threads | PENDING | |
| clippy -D warnings (lib + tests), rustfmt | clean | local |

## Results
PENDING (the debug wall time before → after).

## What didn't work
n/a
