# CUDA_ERROR_ILLEGAL_ADDRESS on packed 4090 pods: investigation, root cause not found  (NO-GO, docs only)

- Date: 2026-10-16. Branch: fix/illaddr. Base: main 4a68be3. Status: open (docs only, no code change)
- Pick: foreman's request to debug the ILLEGAL_ADDRESS in job 1016-162856-7a9e (PR #42 test binary on pod4).

## What changed
Nothing in the code. This entry records what the fault is and isn't, how to reproduce it, and the tools
that worked on a RunPod pod, so the next person starts from here.

## The problem
Since pod packing started (~16:00 on 10-16, `axon-node` GPU slots), whole test suites on RTX 4090 pods
(sm_89, driver 580.167.08) sometimes die with `CUDA_ERROR_ILLEGAL_ADDRESS`. The error is sticky, so every
later GPU test in the process fails too (49 to 143 cascade failures). There were 9 incidents in axon-run's
logs on 10-16 between 16:28 and 17:48: 7a9e, 72b5, 4513, ba9f, 4588, fd69, 0dbd, aa3c and e154, on pods 1, 2
and 4. None appear in any earlier log. The binaries differed: main a591e17, kvsave, yarn and spec32, in
release and debug builds. The test that saw the fault first also differed every time:
gpu_q8_qwen3, gpu_bits_llama_3b_8b, gpu_prefill_batchings_bit_identical_{0_5b,q8_1_5b},
gpu_products_match_reference_bits, and pipeline_{same_device_matches_cuda(_1_5b),verify_matches_forward_argmax,one_stage_is_cuda}.
The test that observes a sticky error is the first one to synchronize afterwards, so it need not be the one
that caused it.

## What the fault depends on (MEASURED, pod5 = RTX 4090, held for this)
- **It needs concurrency.** axon-node's POISON-RERUN reran every poisoned test of fd69 alone, each in a fresh
  process: 79/79 passed. The kvsave builder's split of 72b5's failures behaved the same way, and it is
  intermittent. gpu_bits_llama_3b_8b alone passed (6ee3). The other 116 together, multi-threaded, faulted
  again with a different first test (4588). Then the 12 q8 tests with --test-threads=1 passed (bf82), and
  the remaining 74 at the default 4 threads also passed (e581).
- **Reproduced on pod5 with the main binary** (6f88bb, the PR #42 test binary). The run was
  `--test-threads 4` of gpu_prefill_batchings_bit_identical_{0_5b,1_5b,llama} + gpu_bits_llama_3b_8b, with other
  CUDA processes on the card, 60 s per round.
  - 4 hits in about 90 rounds: p0a r1, pairA r5, vicN r7 and vicN r19. Each hit came 10 to 35 s into the
    process, with all live threads in the per-token decode loop of `prefill_with` (tests.rs:3086,
    `net.feed`: the decode step's CUDA graph replay).
  - 0 hits in the 6 rounds with no other process on the card. That is too few rounds to clear the
    "alone" case.
  - All 4 hits happened while a compute-sanitizer process was running on the card as a co-tenant. Rounds
    without one: 0 hits in about 20.
  - With CUDA_LAUNCH_BLOCKING=1: 1 hit (vicLB r8). All 4 threads got
    `cuGraphLaunch: CUDA_ERROR_ILLEGAL_ADDRESS`. So the fault lies within one decode step: either the
    8-byte H2D copy of {token, pos} from pinned `host_params`, or a kernel in the step's captured graph.
  - With CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1: 0 hits in about 35 rounds, so no GPU core dump was ever
    written. That is not significant either way.
  - With AXON_CUDA_GRAPHS=0 + CUDA_LAUNCH_BLOCKING=1, which would put the kernel's name in the error
    (`cuLaunchKernelEx <name>`): 0 hits in 29 rounds. Graphs on over about the same period: 1 hit in 27
    (LAUNCH_BLOCKING) and 2 in 40 (plain). At ~5 % a round that is not significant; graphs-only is
    possible, not shown.
  - Pod5 logs, including every hit's full log and the memcheck logs: `~/.cache/axon-illaddr/pod5-logs/` on bella.
- **Under memcheck, one test thread (the 0.5B prefill test, card shared):** the run died with
  CUDA_ERROR_LAUNCH_FAILED (719) at the same decode-step sync, tests.rs:3086, and memcheck reported no
  invalid access. The fault was not an instrumented kernel load or store. memcheck of gpu_q8_qwen3 and the
  first iterations of gpu_q8_qwen3_4b: 0 errors.
- **Trivially memory-safe workloads never faulted** under the same co-tenants. These were Python ctypes
  programs with 4 threads in one context (`~/.cache/axon-illaddr/victim*.py`), about 50 minutes of runs and
  roughly 136k work units in total. They covered: graphs of 8 or 300 kernel nodes replayed
  with alloc/free churn; Axon's decode-step API sequence (cuMemHostAlloc'd 8-byte params H2D, graph replay,
  600 KB D2H, sync, then cuMemFreeHost/cuMemFree per "cache"); that sequence plus a thread doing the upload
  ring's cuMemHostRegister / H2D / cuCtxSynchronize / cuMemHostUnregister / munmap cycle; and a kernel that
  indexes by the uploaded params, with every step's value checked afterwards (no stale or corrupt params).

## Ruled out (by experiment or by reading)
- **MPS.** pod compute mode is Default and no MPS daemon runs, so separate processes have separate contexts.
- **cuMemFree not waiting.** freewait.py showed cuMemFree waiting for a still-running kernel on our
  non-blocking stream (1001 ms for a 1000 ms spin). capfree.py showed the same while another thread holds a
  THREAD_LOCAL stream capture with nodes in it.
- **Deterministic out-of-bounds access.** memcheck in serial: see above.
- **Kernel local-memory resizing.** The largest local frame is 40 B (rope_table), under the default stack, so
  no resize ever happens.
- **PDL.** It is sm_90+ only.
- **Kernel cache races.** Entries are written to a temp file and renamed, and carry checksums.
- **HostBuf reuse while a copy is in flight.** `pending` makes the host wait for the stream first.
- **Ring slot recycling.** It is event-ordered, and the ring syncs its stream before returning.
- **cp.async left in flight at kernel exit.** attn_blk_t, attn_fa_bt and gemm2_t all wait for every real
  copy before they exit.
- **VRAM-dependent batch / Q6_K-copy decisions.** No geometry depends on them unsafely, and the hits
  happened with VRAM free.

## Not done / next steps
- Name the faulting kernel or copy. The recipe needs no code change: `AXON_CUDA_GRAPHS=0
  CUDA_LAUNCH_BLOCKING=1`. Every launch is then synchronous and `check()` puts the kernel's name in the
  error. Run fast.sh's victim set with it next to a heavy co-tenant (`compute-sanitizer` on another test,
  or spinner.py). If the error lands on `embed` / `rope_table`, the first kernels after the params copy,
  suspect the copy.
- Try the pipeline tests (4 of the 9 first-faulters) as victims. They open two `Gpu`s, and so two streams,
  in one context.
- Packing (axon-node), until this is understood: the incidents began with packing and need concurrency. Two
  options: run GPU test suites exclusively (`AXON_NODE_EXCLUSIVE=1`), or set `AXON_TEST_GPU_SERIAL=1` for
  packed jobs (one GPU test at a time per process). POISON-RERUN already turns a cascade into "culprit none,
  N/N pass alone". Keep it, and record "culprit none" runs as this issue, not as branch failures.

## Tools that worked on a pod (no CUDA toolkit in the image)
- compute-sanitizer: the pip wheel `nvidia-cuda-sanitizer-api` lacks TreeLauncherSubreaper and fails with
  "Target application terminated before first instrumented API call". Use the redist tarball instead:
  `cuda_sanitizer_api-linux-x86_64-13.0.85-archive.tar.xz` (13.0 matches driver 580).
- cuda-gdb: the redist tarball `cuda_gdb-linux-x86_64-13.0.85-archive.tar.xz` runs in the pod container
  (ptrace works; ASLR can't be disabled).
- Run the store binary by hand like axon-node does: `cd` to the manifest symlink, then
  `XDG_CACHE_HOME=/root/axon-node/cache LD_LIBRARY_PATH=/root/axon-node/lib/nvrtc-13.0`.
- Scripts: `~/.cache/axon-illaddr/` (fast.sh victim loop, san.sh, gdb.sh, starter.sh, victim*.py, NOTES.md).
