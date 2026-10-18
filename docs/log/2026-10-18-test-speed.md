# Test-suite speed: one CPU reference per input, a fast tier, and a deterministic slot-idle test  (PR #80)

- Date: 2026-10-18. Branch: feat/test-speed. Base: main a032773 (the timings were taken on main 5a73f63, before the
  rebase onto #67/#70/#74; those merges touch none of the timed tests' helpers). Status: open
- Headline: this PR's code saves ~166 s of the release suite (-7.6 %, MEASURED, exclusive 4090). The rest of the
  measured -224 s is main's run paying ~55 s of cold kernel cache, which isn't this PR. The bigger lever is the
  fast tier for review rounds: 1209 s, i.e. 55 % of main's full release suite.
- Pick / design: foreman's "test-suite speed" track. The test queue is the bottleneck: 49 GPU-host hours of jobs
  in 24 h, 22 h of them full release suites (237 jobs) and 17 h debug-assertion suites.

## What changed
Test code, scripts and docs only. The non-test release binary is byte-identical to main's.

- `src/cuda/qwen2/tests.rs`: `trace` + `compare_trace` (and `compare_each!`), `compare_forced2`,
  `continuations_by`, `Agree::tally`. Several helpers grade two or more models (GPU, CPU per-row kernels, hybrid
  splits, streamed splits) against the same CPU reference over the same inputs. Until now they recomputed that
  reference once per model. Now it runs once, and each model under test gets exactly the calls it got before
  (same caches, same tokens, same positions), so every tally and assertion sees the same numbers. Call sites:
  `agreement` (logits_match), `long_context`, `2000_tokens`, `natural_long`, `continuations_match`, gemma3's
  `gemma3_vs_cpu` and long test. `llama_mapped` checks the file's path instead of loading a throwaway copy.
- `src/hybrid/tests.rs`: `mixed_graded` and `stream_graded` run the CPU reference once for their four
  comparisons (CPU rows, then k = 1, n/2, n-1), instead of four times.
- `src/serve/tests.rs`: `serve_slot_idle_frees_a_stalled_stream` no longer flakes under load (see Why).
- `scripts/test-tier.sh`, `scripts/test-fast-skip.txt`, `docs/testing.md`: a fast tier for intermediate review
  rounds and a full tier for the final one.

## Why
Measured per-test times first: one exclusive 4090 (pod3, 6.8 vCPU, AXON_THREADS=8), `--test-threads 1`,
libtest `--report-time --format json`. A measurement-only build links a constructor that sets
`RUSTC_BOOTSTRAP=1`, because axon-run forwards only `AXON_*`. Main's release suite takes 2195.6 s there: 513
passed, 35 ignored. The same suite packed on a pod with 4 test threads takes ~2400 s.

What didn't explain it:
- **Model load / repack per test.** Uncontended, a copy load is 0.20-0.27 s and `Qwen2::new` 0.13-0.40 s
  (0.5B, Qwen3 0.6B, Llama 1B). The 8-26 s from a first probe were a cold page cache plus a packed pod.
  Process-wide model fixtures would save seconds per suite and add RSS, so they weren't done.
- **NVRTC per process.** `Kernels::module` is 0.02 s with the disk cache warm. A fresh pod pays it once per
  module (the cold-cache confound below).
- **Test threads.** `pool::Pool` is process-wide and runs one job at a time, so the CPU forwards of all test
  threads take turns, and on a packed pod threads mostly fight over vCPUs. Packed with 4 threads is no faster
  than serial and exclusive. CPU work therefore costs suite time directly, whatever the thread count.

What did: CPU reference passes repeated for every model compared against them. `hybrid_logits_match_cpu_*`
and `stream_logits_match_cpu_*` ran the CPU reference 4x, the `gpu_logits_match_cpu_*`, natural-long and
long-context tests 2x, `continuations_match` 3x. Those 22 tests took 453 s, 21 % of the suite.

The flake: `serve_slot_idle_frees_a_stalled_stream` failed once in main's debug suite on a packed pod
(1018-014631-dd84: "never filled"). The test calls the client's receive window full once the client's queue
stops growing and the engine makes 2 more tokens. With `--slot-idle-ms 300` the server evicts the stream 300 ms
after the window fills. On a loaded host (debug build, a pool shared with other tests' CPU forwards, a packed pod)
one token step can take longer than that, so the stream is gone before the 2 tokens come. After that nothing
grows again, and the loop waits out its 2 minutes. Now, with the flag, the server's own eviction
(`axon_slot_idle_evictions_total` > 0) also counts as the window being full: the client's queue can't grow
once the stream is evicted. The fill time is still the client's last queue change, so the bounds on how fast
the slot is freed are unchanged. No timeout was raised.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| release suite, tree 0e3f4cb, AXON_TEST_GPU_SERIAL=1 | 543 passed, 0 failed, 35 ignored; rc 0; no POISON-RERUN (pod2 e81870b2b3d8, packed, 2415.9 s) | 1018-032116-538f |
| debug-assertion suite, tree 0e3f4cb, AXON_TEST_GPU_SERIAL=1 | 543 passed, 0 failed, 35 ignored; rc 0 (bella, 3183.3 s) | 1018-032116-2f3a |
| test count, `--list` main a032773 vs branch | 578 = 578 (35 ignored both) | local `--list` |
| non-test release binary, main vs branch | byte-identical, md5 697d88a026572ebf2651732eed6d51ef | local build, same flags |
| md5 of `gen` (0.5B, cuda, -n 64), same host (pod2) | identical, 22703eeab624b382bb353f6cb44859a4 both; rc 0 | 1018-022915-1235 / 1018-022937-5c6d |
| clippy -D warnings (bin + tests), rustfmt | clean on 0e3f4cb; clippy also clean on the merge with main cf8dee4 | local |
| slot-idle x20 in a CPU-loaded slice (pod2, 8 pool threads, 4 test threads, + 5 CPU-heavy tests), main / branch | 0/20 failed / 0/20 failed | 1018-021925-f066 / 1018-021949-7daa |
| slot-idle stress x10, 1 pool thread + 64 spinning threads (pod3 9ec6dd14dd25), main / branch | **10/10 failed "never filled"** / **0/10 failed** (freed 0.50-1.70 s after the fill, bound 2.8 s) | 1018-031600-534b / 1018-032125-50cf |
| slot-idle stress x10, 8 pool threads + 64 spinning threads (pod3), main / branch | 10/10 failed "never filled" / **10/10 failed**: the window detection now works, but the slot was freed 4.00-4.20 s after the fill in 8 runs and 7.8-7.9 s in 2, past the test's 2.8 s bound | 1018-030316-d4af / 1018-044914-c136 |

The stress runs link `stress.o`, a measurement-only constructor: AXON_REPEAT=N runs the test binary N times,
and AXON_HOG=K starts K spinning threads in each run, which gives a loaded host on demand. Under that load main
fails every time, so the old test isn't flaky in the random sense. It fails deterministically once a token step
is slower than the gap the server leaves (~300 ms between the window filling and the eviction, during which the
test needs 2 more tokens). An idle host steps far faster than that (0.5B CPU decode: ~18 ms per token in release on a pod, a few times that in
debug). So main passes there, and in a packed suite it only fails when other tests' CPU forwards (one shared
pool) and the pod's neighbours slow the engine enough: once in main's debug suite (1018-014631-dd84), 0 of 20
in the milder slice above. GitHub CI never runs it at all (`AXON_SKIP_MODEL_TESTS=1`). Debug assertions matter
only in that they make each step slower. With the branch, the 1-pool-thread stress passes 10/10 inside the
unchanged timing bound.

**Update 2026-10-18 (after merge):** the fix covers the 1-pool-thread stress but NOT the 8-pool-thread one. Under 8
pool threads plus 64 spinning threads the branch also fails 10/10 (1018-044914-c136). It no longer fails on "never
filled". It fails on the freeing bound instead: freed 4.00-4.20 s after the window filled (8 runs) and 7.8-7.9 s
(2 runs), against 2.8 s. So #80 made nothing worse (main fails 10/10 there too, earlier in the test), but the test
isn't green under that stress. The follow-up investigates the ~4.1 s.

## Results
Release suite, exclusive 4090 pod3 (same host for both), `--test-threads 1`, MEASURED:

| | main 5a73f63 | branch | |
|---|---|---|---|
| the 22 changed tests (this PR's code) | 452.7 s | 286.4 s | **-166.3 s, -7.6 % of main's suite** |
| `gpu_batched_products_match_gemv_bits` + `gpu_capture_survives_pinning_elsewhere` | 72.8 s | 17.9 s | -55 s, not this PR: main's run compiled kernels into the pod's cold NVRTC cache, the branch's found them there |
| the other 489 tests | 1669.9 s | 1667.2 s | -0.2 % (noise) |
| whole suite | 2195.6 s | 1971.6 s | -224 s, -10.2 % as measured |

Method: one exclusive 4090 pod (pod3 9ec6dd14dd25, 6.8 vCPU, AXON_THREADS=8) for both runs, `--test-threads 1`,
libtest `--report-time --format json`. The per-test times are libtest's, the suite's is libtest's
`exec_time`. One run each (a docs-and-scripts round, no rerun of main's exclusive timing). The gate-of-record
suites above ran packed or on bella with AXON_TEST_GPU_SERIAL=1 and aren't timing evidence.

Jobs: 1018-005814-52ea (main), 1018-014201-c3d2 (branch).

Top 20 by main's time (release, same host, s):

| test | main | branch | |
|---|---|---|---|
| cuda::qwen2::tests::gpu_bits_llama_3b_8b | 231.8 | 231.9 | +0% |
| cuda::tests::nvrtc_kernels_compile_for_sm90a | 84.2 | 83.4 | -1% |
| draft::tests::draft_model_output_unchanged_cpu | 64.4 | 61.2 | -5% |
| cuda::tests::nvrtc_kernels_compile_for_old_archs | 63.0 | 63.2 | +0% |
| cuda::qwen2::tests::gpu_q8_qwen3_4b | 60.6 | 61.3 | +1% |
| cuda::qwen2::tests::gpu_logits_match_cpu_natural_long_1_5b | 59.1 | 43.0 | -27% |
| hybrid::tests::hybrid_real_model_sampling | 57.1 | 57.2 | +0% |
| cuda::qwen2::tests::gpu_prefill_batchings_bit_identical_qwen3_4b | 51.0 | 50.9 | -0% |
| cuda::qwen2::tests::gpu_q8_qwen3 | 46.5 | 47.2 | +1% |
| cuda::qwen2::tests::gpu_batched_products_match_gemv_bits | 44.9 | 2.8 | (cold kernel cache on main's run) |
| cuda::qwen2::tests::gpu_prefill_batchings_bit_identical_q8_1_5b | 41.2 | 41.3 | +0% |
| cuda::qwen2::slots::tests::pool_wide_long_steps_match_solo | 40.6 | 41.6 | +2% |
| hybrid::tests::stream_logits_match_cpu_1_5b | 39.8 | 21.3 | -47% |
| hybrid::tests::hybrid_logits_match_cpu_1_5b | 38.7 | 21.8 | -44% |
| cuda::qwen2::tests::gpu_q8_llama | 33.5 | 33.4 | -0% |
| cuda::qwen2::tests::gpu_yarn_matches_reference_and_keeps_bits | 32.7 | 30.8 | -6% |
| hybrid::tests::hybrid_prefill_batchings_bit_identical_llama | 32.6 | 33.0 | +1% |
| draft::tests::chat_with_draft_model_unchanged | 32.3 | 25.2 | -22% (untouched: noise) |
| hybrid::tests::hybrid_prefill_batchings_bit_identical_qwen3 | 31.3 | 31.1 | -1% |
| hybrid::tests::hybrid_logits_match_cpu_llama | 30.1 | 16.4 | -46% |

Debug-assertion build, main, top 20 (exclusive 4090 pod2, t1, 1018-005834-c83c, 1171 s for the 20): it has the
same order as release, and the GPU-bound tests cost about the same in both builds (gpu_bits_llama_3b_8b 241 s,
nvrtc sm90a 129 s, draft cpu 86 s, hybrid_real_model_sampling 78 s). So debug assertions aren't what makes the
debug suite cost a GPU slot for ~40 min: it's the same GPU work run a second time. That's why the fast tier
runs the debug build on the CPU queue.

The tiers, from the measured per-test times (so ESTIMATED as whole-tier numbers). One baseline: main's full release
suite on the exclusive 4090, 2195.6 s.

| per review round | GPU-slot time (serial, exclusive 4090) | vs main's release suite |
|---|---|---|
| before: release + debug suite, both on GPU hosts | 2195.6 s + roughly the same again | 200 % |
| full tier, this branch (final round): release + debug + md5 | 1971.6 s + debug | ~180 % |
| fast tier, this branch: release minus the skip list | 1209 s, plus the debug suite on a CPU slot (main's took 698 s packed, 1018-014631-dd84) | 55 % |

The skip list is 763 s of the branch's release suite (39 %). Its entries are NOT redundant. Each is the only test
of some shape or weight-type combination (`scripts/test-fast-skip.txt` says which): head size 128 with GQA 3/4/6
and K-quant weights (Llama 3B/8B, Qwen2.5 1.5B batchings and grading), Qwen3 with K-quant weights (4B), and the
NVRTC builds for sm_70/75/80/90a. Between review rounds the fast tier covers head size 128 only through Qwen3 0.6B
(GQA 2, Q8_0) and the 1.5B verify and q8 tests. The NVRTC builds come back in whenever the branch or the working
tree changes `src/cuda/*.cu`, `src/cuda/mod.rs` or `src/cuda/ffi.rs`. `gpu_bits_llama_3b_8b` is also a witness
in the open pod ILLEGAL_ADDRESS investigation (sa_315).

## What didn't work / wasn't done
- Process-wide model and `Qwen2` fixtures (OnceLock): no gain worth the RSS (loads are 0.2 s). Also unsafe
  with copied models, because `Qwen2::new` releases what it repacks.
- Caching CPU references across tests (the same 0.5B/1.5B/Qwen3/Llama traces are recomputed by the gpu, hybrid
  and stream test families): would save ~100 s more, but at ~120 MB per model held for the process, and keyed by
  model contents (the yarn test changes a model in place). Left for later.
- Shortening any prompt or position range: none qualified. The 1200-token batchings must cross the max batch
  and the attention tiles, and the near-tie statistics were measured on these exact inputs.
- `gpu_bits_llama_3b_8b` (232 s, 11 % of the suite) is real GPU work: 4 attention paths x 7 batchings over 1200
  tokens on 3B and 8B. It's in the fast tier's skip list, untouched otherwise.
- The NVRTC cross-arch tests (147 s) could compile their archs in parallel threads. That doesn't help a
  multi-threaded suite run, where they already overlap other tests. Skip-listed instead.
- Found on the way: the ILLEGAL_ADDRESS cascade on pods (1018-001615-48ee; owned by sa_313). `axon-run --shard`
  ignores `--dry-run` and submits for real (the tier script refuses that combination).
