# GPU tests serialize on every card; CUDA_ERROR_ILLEGAL_ADDRESS on packed 4090 pods  (PR #82)

- Date: 2026-10-18. Branch: fix/cuda-concurrency. Base: main 5a73f63. Status: open.
- Pick / design: foreman bug hunt (r10): context-poisoning ILLEGAL_ADDRESS on RunPod 4090 pods, hidden by
  axon-node's POISON-RERUN ("culprit none (did not reproduce alone)", rc 0). Follows #53 (illaddr, NO-GO).

## What changed
- `src/testutil.rs`: `testutil::gpu()` takes the process-wide GPU test lock on every card. Before, it took it
  only on cards under 16 GiB, so on 24 GiB 4090s libtest ran up to 4 GPU tests at once. `AXON_TEST_GPU_SERIAL=0`
  still turns the lock off, for hunting. The new test `gpu_tests_never_overlap` checks that a second GPU test
  waits for the first. It fails on main on any card of 16 GiB or more and passes here.
- `src/cuda/mod.rs`: in debug builds, the launch check (#68 compares the argument count against
  `cuFuncGetParamInfo`) now also compares each argument's size against its parameter's (`Arg::size`). Example:
  a `U32` passed where the kernel takes a pointer.
- `src/cuda/qwen2/tests.rs`: two `#[ignore]` hunting aids, `gpu_concurrency_matrix_hunt` and its child
  `gpu_concurrency_hunt_child` (env-driven, see below). No change to non-test code paths in release builds.

## Why
The bug: a job's CUDA context is poisoned (`CUDA_ERROR_ILLEGAL_ADDRESS` at the next sync), and every later
GPU test in that process fails with it. Seen in 21 axon-run logs since 10-16, all on 4090 pods, across 7
RunPod pods and drivers 570, 580 and 595.

What every one of those runs had in common: several GPU tests in flight in one process. `serialize_small_gpu`
serialized only cards under 16 GiB. That is why bella (12 GB), avante (6 GB) and jade (3 GB) never showed it.

MEASURED on pods, main 5a73f63 test binary. set10 = the prefill_batchings family, gpu_bits_llama_3b_8b and
natural_long_{0_5b,1_5b}:

| run | config | result |
|---|---|---|
| 906f, cf9e, fe45 | set10, 4 test threads, packed pod1/pod2 | FAULT at 130 s, ~100 s, ~300 s |
| 9eb3 | set10, 4 threads, alone on pod3, coredump env | pass |
| 78f9 | set10, `--test-threads 1`, packed pod1 | 10/10 pass (707 s) |
| b223 | set10, 4 threads, AXON_CUDA_GRAPHS=0, packed pod1 | pass |
| 59f1 | memcheck, prefill_batchings_0_5b, serial, pod1 | 0 errors (1495 s) |

The hunting harness narrows the trigger. Three threads run 1200-token per-token decode graphs on mapped
1.5B, 1B and 0.5B models. Next to them, one "aggressor" thread does one thing in a loop for 75 to 150 s:

| aggressor | where | result |
|---|---|---|
| in-process: model churn (fresh or long-lived thread), ctx retain/release, module load/unload, 256 MB alloc/free, pin/unpin, host_alloc, a whole prefill_bits | pod2, alone | all clean (~20 min) |
| `proc`: child processes, 12 s 0.5B decode | pod2 / pod1 | FAULT 108 s, FAULT 22 s |
| `proc-ctx`: child processes that only open the GPU | pod1 | FAULT 61 s |
| `spawn`: `sh -c "sleep 0.2"` children (no CUDA at all) | pod1 | FAULT 120 s |
| `proc`, ONE victim thread | pod1 | 2 x 125 s clean |
| `proc`, AXON_CUDA_GRAPHS=0 | pod2 | 2 x 125 s clean |
| `spawn`, exclusive pod3 (Ryzen 7950X, no IOMMU) | pod3 | 2 x 150 s clean |
| `spawn` | bella (sm_120) | 2 x 150 s clean |

Guard canaries around every DevBuf (`AXON_ALLOC_GUARD`, experiment only, not in the PR): none was smashed,
and the faults went on regardless (449e).

By host: pod1 (EPYC 7452, AMD IOMMU on) faulted in 10 of 50 jobs and pod2 (EPYC 75F3, AMD IOMMU on) in 7 of
36. Both are packed. pod3 (Ryzen, no IOMMU, never packed) had 0 faults in 16 jobs. The coredump env
(`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`) suppressed the fault each time it was set (here and in #53), so the
faulting kernel and address are still unnamed.

Reading: the fault needs two things at once. First, more than one stream of decode-graph work in flight in
the process. Second, a process being created on the host. The process doesn't have to touch CUDA, and doesn't
even have to be our child: packed co-tenant suites are enough. It shows up on the IOMMU-enabled EPYC pods.
Nothing Axon does in-process triggers it. memcheck and the guards find no out-of-bounds access. ESTIMATED,
not proven: below Axon (driver, IOMMU or host), not an Axon buffer bug. Axon itself is single-threaded on the
GPU outside tests, so a test harness that never overlaps GPU tests removes the half of the trigger Axon
controls. With one GPU thread nothing faulted, but the evidence is short runs, not a proof of zero.

## How it was verified
MEASURED at 323770d (pods show UTC; the dates/times here are EDT):

| gate | job | host | result |
|---|---|---|---|
| release suite, shard 1/2/3, default env (now serial) | 8500 / ea6b / 3e7b | pod2 / pod1 / pod1 | 246/0, 163/0, 105/0 (514 pass, 0 fail, no POISON) |
| debug-assertion suite (+ the size check), shard 1/2/3 | de1c / 59c9 / 4ba3 | pod1 / pod1 / pod4 | 246/0, 163/0, 105/0 (514 pass, 0 fail) |
| `gpu_tests_never_overlap` on main + the test only | dfda | pod1 | rc 101: fails on main (the second test got the GPU while the first held it) |
| same test on this branch | 8500 (rel), de1c (dbg) | pod2, pod1 | pass |
| set10 (the faulting set), default 4 threads, packed pod1 | 1990, 5258 | pod1 (EPYC, IOMMU, 2 co-tenants each) | 10/10, 10/10 (main: FAULT in 3 of 4) |
| set10 runs 3-5 | f026 / f287 / e592 | pod2 / pod1 / pod1 (EPYC, IOMMU) | 10/10 each: 5/5 clean vs main 3/4 faulted |
| release suite after merging main c8e34fd (54c348d) | bb4c / b6b6 / 303d | pod4 / pod2 / pod1 | 553 pass, 0 fail |
| debug + release suites after merging main ac358d4 (097249f) | see PR #82 | pods or bella | see PR |
| md5 of gen, main vs candidate (3 models x 3 prompts x cuda/cpu) | 4309 | bella | 18 same, 0 differ |
| clippy -D warnings (bin + tests, debug cfg), rustfmt --check | local | bella | clean |

axon-node exits rc 102 for any poisoned context from 04:22 EDT on (devstack 7f14147). 5258 and dfda started
after that, so 5258's rc 0 is clean evidence by itself. The other gates have no POISON-RERUN line in their
logs and 0 failures, so they are clean too.

## What didn't work / open
- The faulting kernel isn't named: the coredump env hides the fault, and memcheck finds nothing.
- Untested: an IOMMU-enabled host without packing; IOMMU passthrough (`iommu=pt`) on the same host.
- Untested: whether production `axon serve` (one GPU thread) can fault on an IOMMU host with process churn.
  A short 1-thread run was clean.
