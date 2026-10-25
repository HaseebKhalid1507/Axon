# testutil: test loaders map as the product does  (PR #134)

- Date: 2026-10-25. Branch: fix/testutil-map. Base: main b2068e7; merged with bdc80f8 (main fb749cc + #136) and
  then main 5bd21c3 (= bdc80f8 + #136's docs commit: same code tree, different commit). Status: open
- Pick / design: issue #132 (Shady's review of #128). Follow-up filed: #138.

## What changed
- `gguf::load` (test-only: the binary never calls it) loads as `axon` does for the CPU: `load_to_run` with
  `model::cpu_copy_wish`. A MoE model is always mapped; a dense one is copied only when it reads its matrices in
  place (AXON_REPACK=0, types that don't repack), under the usual 60 % / AXON_MMAP rule. Every default test model
  repacks, so all of them are now mapped instead of copied.
- `testutil::model_with(None)` uses it. So do `model()`, `qwen3_model`, `llama_model`, `mistral_model`,
  `gemma3_model` and the direct `gguf::load` callers.
- `model::cpu_copy_wish(m)` is the one CPU load policy: `crate::load`, `cli::open_model_with` and `gguf::load`
  call it (it was the same `match` three times). Moved, not changed.
- **Read-after-release guard, per test model.** `Model::strict_release` is a `#[cfg(test)]` field set only by
  `gguf::load`, i.e. exactly the loads that used to copy. On those models `tensor_data` refuses a released
  tensor on the file mapping too, as the copy did (where a released tensor reads as zeros). That covers rewritten
  tensors as well: llama/mistral q/k live in side copies on a mapping. Releases made after the flag is set are kept
  in a test-only list (`strict_released`) and checked *before* the side-copy return. Explicit
  `load_with(_, Some(false))` loads keep main's copy-only check: about 23 GPU tests build several nets from one
  mapped Model on purpose (#138). Non-test builds (release and debug binaries) run main's code: everything new is
  `#[cfg(test)]`.
- Tests: `testutil::load_tests::loaders_map_as_the_product_does` (replaces #128's `moe_test_models_are_mapped`)
  checks the helpers map, `load_moe` maps, and the one explicit copy-path test (`load_with(_, Some(true))` and
  `load_with(_, None)` copy the 0.5B and hold the mapping's bytes, one at a time).
  `released_tensor_refused_on_a_mapping` checks a strict model refuses a released `ffn_down` and a released llama
  `attn_q` (side copy), while an explicit mapping reads a released tensor back. The other explicit copy tests are
  unchanged (`llama_load_paths_agree`, `release_keeps_embedding_on_copy_and_mmap`,
  `load_to_run_map_only_never_copies`).

## Why
`gguf::load` copied any file under 60 % of `MemAvailable` into anonymous memory. So Mistral-7B (4.4 GB) and the
other dense helpers held a second copy of the file next to their r8 arena, which the product never does. That's an
OOM-kill risk in memory-capped test jobs, and RAM the scheduler has to claim for nothing.

## How it was verified
Gates of record at the final head (f571b14, with main 5bd21c3 merged):

| gate | result | job |
|---|---|---|
| focused `release_` + `testutil::load_tests` (with the llama arm), release / debug | PASS 7 / 0 / 0 each | 1025-153745-2460, 1025-153745-157b |
| full release suite, 2 shards, bella sm_120, `--ram 16000` | pending | 1025-153839-2b70, 1025-153839-b71e (group 1025-153839-7ad8) |
| full debug-assertion suite, 2 shards, bella, `--ram 16000` | pending | 1025-153839-2112, 1025-153839-ce69 (group 1025-153839-f7c1) |
| md5 of `gen` output vs main 5bd21c3, 4 models × 3 prompts × cpu + cuda f16, bella | pending | 1025-153839-7d39 (base: axon-bdc80f8; `git diff bdc80f8 5bd21c3 -- src` is empty) |
| clippy -D warnings (lib, lib tests, bin), rustfmt | clean | local |

Earlier rounds, kept for the record:

| commit | gate | result | job |
|---|---|---|---|
| 43bc8b7 | debug fast tier (CPU) | PASS 656 / 0 / 48 | 1025-054658-1dcb |
| 43bc8b7 | release fast tier | killed (rc 143) at 1009 s by the lead for bella memory starvation, not a test failure; rerun 1025-061646-1db1 dropped unstarted | 1025-054658-c346 |
| 43bc8b7 | md5 vs b2068e7 | 24/24 | 1025-054657-ad4c |
| 6fa7533 | debug fast tier (CPU) | FAIL 656 / 1 (global guard, see below) | 1025-063402-cf93 |
| 6fa7533 | release fast tier | dropped before start (same failure expected) | 1025-063402-4bbe |
| 6fa7533 | md5 vs b2068e7 | 24/24 | 1025-063401-9f19 |
| 5583215 | focused `release_` + load_tests, release / debug | 6 / 0 / 0 each | 1025-065516-bfea, 1025-065517-03c0 |
| 5583215 | release fast tier | FAIL 616 / 23 (global guard, see below) | 1025-065721-7ada |
| 5583215 | debug fast tier (CPU: those GPU tests skip) | PASS 656 / 0 / 48 | 1025-065721-1c4a |
| 5583215 | repro: `hybrid_logits_match_cpu_0_5b` | panics on `blk.1.attn_q.bias` after the test's second CPU net | 1025-085554-48d3 |
| main fb749cc | product check: hybrid 0.5B/qwen3/llama, gemma3, MoE pruned `gen --device cuda`, with and without AXON_MMAP=0 | 10/10 rc 0, no panic, output identical per model | 1025-085708-49c2 |
| 942dc0c | focused (4 formerly failing GPU tests + release_ + load_tests), release / debug | 9 / 0 / 0 each | 1025-090053-8d97, 1025-090056-922c |
| 942dc0c | release fast tier / md5 | cancelled (rc 130), superseded by the 1ca1703 round | 1025-090159-950b, 1025-090158-53fc |
| 942dc0c | debug fast tier (CPU) | PASS 657 / 0 / 48 | 1025-090159-a17c |
| 1ca1703 (942dc0c + bdc80f8) | full release, 2 shards | PASS 667 / 0 / 48 | [29e9](https://github.com/HaseebKhalid1507/Axon/blob/81011aced49ba7f650a4a9b9f6a92c9de2e90505/2026-10/testmap-main2/1025-130813-29e9.log), [433b](https://github.com/HaseebKhalid1507/Axon/blob/81011aced49ba7f650a4a9b9f6a92c9de2e90505/2026-10/testmap-main2/1025-130813-433b.log) |
| 1ca1703 | full debug, 2 shards | PASS 667 / 0 / 48 | [3108](https://github.com/HaseebKhalid1507/Axon/blob/81011aced49ba7f650a4a9b9f6a92c9de2e90505/2026-10/testmap-main2/1025-130813-3108.log), [d72c](https://github.com/HaseebKhalid1507/Axon/blob/81011aced49ba7f650a4a9b9f6a92c9de2e90505/2026-10/testmap-main2/1025-130813-d72c.log) |
| 1ca1703 | md5 vs bdc80f8 | 24/24 | [e600](https://github.com/HaseebKhalid1507/Axon/blob/81011aced49ba7f650a4a9b9f6a92c9de2e90505/2026-10/testmap-main2/1025-130814-e600.log) |

## Results
Bits: the md5 of `gen` output is identical vs main in every round. Non-test code changes only by moving the
`cpu_copy_wish` match.

Memory, MEASURED, one run each: axon-run's per-tick PSS of the job's process tree, debug fast tier on bella's CPU
queue, 4 test threads. Co-tenants differed between the runs, so treat this as indicative:

| run | tests | peak PSS | learned anon | learned model working set |
|---|---|---|---|---|
| f298497 (main's loaders) 1025-054002-f395 | 664 | 11461 MB | 9372 MB | 13938 MB |
| 43bc8b7 (mapped) 1025-054658-1dcb | 656 | 9765 MB | 8987 MB | 15471 MB |

Anonymous memory goes down and page cache charged to the job goes up (reclaimable). Mapped loads also start
`read_ahead` (MADV_WILLNEED over the file), which `load_with` never did: new page-cache I/O per test load, nearly
free on a warm cache.

## What didn't work
- **A global guard, `STRICT_RELEASE = cfg!(any(test, debug_assertions))` (6fa7533).** It refused released tensors on
  every mapping in every test and debug build. It broke `release_keeps_embedding_on_copy_and_mmap`, which reads a
  released tensor on a mapping on purpose (debug 1025-063402-cf93: 656 / 1). It would also have made debug `axon`
  binaries stricter than release ones.
- **The same guard with that one test patched (5583215).** The test was switched to expect the panic, with a
  `#[cfg(test)] tensor_data_unchecked` to keep checking the release-mode property. The GPU release suite then failed
  23 tests (1025-065721-7ada: 616 / 23): hybrid, stream, gemma3, the identity split, and MoE/oss GPU graded. Each
  loads explicitly mapped and builds several nets from one Model by design. That's not a product bug: `--device
  cuda` always maps, and main gives identical output with AXON_MMAP=0 (1025-085708-49c2). It's not a false positive
  either: `release` records exact tensor ranges, and only the madvise is rounded.
- **The per-Model flag at 942dc0c** still returned side copies before the check, so a llama/mistral q/k read after
  release went unguarded (round-2 review). Closed at f571b14.
