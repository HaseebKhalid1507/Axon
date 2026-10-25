# testutil: test loaders map as the product does  (PR #134)

- Date: 2026-10-25. Branch: fix/testutil-map. Base: main b2068e7. Status: open
- Pick / design: issue #132 (Shady's review of #128).

## What changed
- `gguf::load` (test-only: `cfg_attr(not(test), allow(dead_code))`; the binary never calls it) is now
  `load_to_run` with the product's CPU wish, the same closure as `crate::load` / `cli::open_model_with`: a MoE
  model `MapOnly`; a dense one `Want(model::reads_matrices_in_place(m))`. So a dense model that repacks (every
  default test model) is mapped, and one that reads its matrices in place (AXON_REPACK=0, types that don't repack)
  is copied under the usual 60 % / AXON_MMAP rule, as `axon gen --device cpu` does.
- `testutil::model_with(None)` uses it (was `load_with(_, None)`, the copy path). `qwen3_model`, `llama_model`,
  `mistral_model`, `gemma3_model` and the tests that call `gguf::load` directly follow automatically.
- `model::cpu_copy_wish(m)` is the one CPU policy: `crate::load`, `cli::open_model_with` and `gguf::load` call it
  (review round: it was the same `match` three times).
- **Read-after-release guard, tests and debug builds** (`gguf::STRICT_RELEASE = cfg!(any(test, debug_assertions))`):
  `Model::release` records, and `tensor_data` refuses, a released tensor on a file mapping too. Before, only the
  anonymous copy refused (there the released pages read as zeros); on a mapping they fault back in with the right
  bytes, so with the test models now mapped a read-after-release bug would have passed every repack test (chat, draft,
  serve, sample, r8q, model) while the product's copy path (AXON_MMAP=0, models that read in place) reads zeros.
  Release builds: `STRICT_RELEASE` is false, the same code as before. New test
  `testutil::load_tests::released_tensor_refused_on_a_mapping`.
- `testutil::load_tests::moe_test_models_are_mapped` (#128) becomes `loaders_map_as_the_product_does`: the
  helpers map (MoE always; dense when they repack), and the one explicit copy-path test: `load_with(_, Some(true))`
  and `load_with(_, None)` (when the RAM available is > 4 GB) copy the 0.5B and hold the mapping's bytes, tensor by
  tensor. The other explicit copy tests stay: `llama_load_paths_agree`, `release_keeps_embedding_on_copy_and_mmap`,
  `load_to_run_map_only_never_copies`.

## Why
`gguf::load` copied any file under 60 % of `MemAvailable` into anonymous memory: Mistral-7B (4.4 GB) and the other
dense helpers held a second copy of the file next to their r8 arena, which the product never does. Not a false
refusal (dense models don't run the MoE check), but an OOM-kill risk in memory-capped test jobs (`--ram 6000`
scopes, jade's cap) and RAM the scheduler has to claim for nothing.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| debug-assertion suite (fast tier, CPU) @43bc8b7 | PASS 656 / 0 / 48 | 1025-054658-1dcb (bella, CPU queue) |
| release suite (fast tier) @43bc8b7 | not a result: killed (rc 143) at 1009 s by the lead, for bella memory starvation (per the review), not a test failure; its rerun 1025-061646-1db1 was dropped from the queue unstarted when the review round changed test code | 1025-054658-c346 |
| md5 of `gen`, main vs candidate, same host @43bc8b7 | 24/24 identical (4 models × 3 prompts × cpu, cuda f16) | 1025-054657-ad4c (bella) |
| debug-assertion suite @6fa7533 | FAIL 656 / 1: `release_keeps_embedding_on_copy_and_mmap` read a released tensor on the mapping on purpose (the new guard's first catch); release job 1025-063402-4bbe dropped | 1025-063402-cf93 (avante) |
| focused `release_` + `testutil::load_tests` @5583215 (the test fixed, `#[cfg(test)] tensor_data_unchecked`) | PASS 6 / 0 / 0 release and debug | 1025-065516-bfea, 1025-065517-03c0 |
| release suite (fast tier) @5583215 | pending | 1025-065721-7ada (bella, --ram 16000) |
| debug-assertion suite (fast tier, CPU) @5583215 | pending | 1025-065721-1c4a (avante, --ram 16000) |
| md5 @6fa7533 (valid for 5583215: test-only commit) | pending | 1025-063401-9f19 (bella) |
| clippy -D warnings (lib, tests, bin), rustfmt | clean | local |

## Results
Bits: the release binary's behaviour is unchanged: `gguf::load` is dead code outside tests, `cpu_copy_wish` is the
same `match` moved, and `STRICT_RELEASE` is false without debug assertions. md5 above.

Memory, MEASURED, one run each, axon-run's samples of the job's process tree (PSS every tick, bella CPU queue, debug
fast tier, 4 test threads; same host class, different co-tenants, so treat as indicative):

| run | tests | peak PSS | learned anon | learned model working set |
|---|---|---|---|---|
| f298497 (#133: main's loaders) 1025-054002-f395 | 664 | 11461 MB | 9372 MB | 13938 MB |
| 43bc8b7 (this PR, mapped) 1025-054658-1dcb | 656 | 9765 MB | 8987 MB | 15471 MB |

As expected: anonymous memory down (-385 MB anon, -1.7 GB peak PSS), page cache charged to the job up (+1.5 GB working
set, reclaimable). `load_to_run` also starts `read_ahead` (MADV_WILLNEED over the file) on every mapped load, which
`load_with` didn't: new page-cache I/O per test load, nearly free on a warm cache.

## What didn't work
n/a.
