# testutil: test loaders map as the product does  (PR #PRNUM)

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
- `testutil::load_tests::moe_test_models_are_mapped` (#128) becomes `test_loaders_map_as_the_product_does`: the
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
| release suite (fast tier) | REL | 1025-054658-1dcb (bella) |
| debug-assertion suite (fast tier, CPU) | DBG | 1025-054658-c346 (CPU) |
| md5 of `gen`, main vs candidate, same host | MD5 | 1025-054657-ad4c (bella) |
| clippy -D warnings (lib, tests, bin), rustfmt | clean | local |

## Results
Bits: test-only change; the axon binary's code paths are untouched (`gguf::load` is dead code outside tests).
PEAKS

## What didn't work
n/a.
