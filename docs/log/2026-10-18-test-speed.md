# Test-suite speed: one CPU reference per input, a fast tier, and a deterministic slot-idle test  (PR #80)

- Date: 2026-10-18. Branch: feat/test-speed. Base: main a032773 (the timings were taken on main 5a73f63, before the
  rebase onto #67/#70/#74; those merges touch none of the timed tests' helpers). Status: open
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
| release suite (rebased tree) | pending | 1018-022946-e1f5 |
| debug-assertion suite (rebased tree) | pending | 1018-022946-32cb |
| test count, `--list` main vs branch | 578 = 578 (35 ignored both) | local `--list` |
| non-test release binary, main vs branch | byte-identical, md5 697d88a026572ebf2651732eed6d51ef | local build, same flags |
| md5 of `gen` (0.5B, cuda, -n 64), same host | pending | 1018-022915-1235 / 1018-022937-5c6d |
| clippy -D warnings (bin + tests), rustfmt | clean | local |
| flake, 20 runs each under CPU load, main vs branch | pending | 1018-021925-f066 / 1018-021949-7daa |

## Results
Release suite, exclusive 4090 pod3 (same host for both), `--test-threads 1`, MEASURED:

| | main 5a73f63 | branch | |
|---|---|---|---|
| whole suite | 2195.6 s | 1971.6 s | -10.2 % |
| the 22 changed tests | 452.7 s | 286.4 s | -37 % |
| the other 491 tests | 1742.7 s | 1685.0 s | -3 % (noise, plus main's run compiling kernels into a cold pod cache: `gpu_batched_products_match_gemv_bits` 44.9 -> 2.8 s, `gpu_capture_survives_pinning_elsewhere` 28.0 -> 15.1 s) |

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

The tiers (from the measured per-test times, so ESTIMATED as a whole-tier number):

| per review round | GPU-slot time (serial, exclusive 4090) |
|---|---|
| before: release + debug suite on GPU | ~2200 s + ~2200-2400 s |
| fast tier, branch: release minus the skip list | 1209 s (-45 % vs main's release), plus the debug suite on a CPU slot (main's took 698 s packed, 1018-014631-dd84) |
| full tier (final round): release + debug + md5 | as before, minus this PR's 10 % |

The skip list is 763 s of the branch's release suite (39 %). Every entry repeats a contract that the fast tier
checks on a smaller model: Llama 3B/8B, Qwen3 4B, the Qwen2.5 1.5B siblings. It also holds the two NVRTC
cross-arch builds, which come back in when a branch touches `src/cuda/*.cu` or `src/cuda/mod.rs`.

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
