# serve: no N+1 cache, the single-request path runs in slot 0  (PR #PRNUM)

- Date: 2026-10-17. Branch: feat/serve-one-cache. Base: main cf68e9d (after #57). Status: open
- Pick / design: r10 round C2 (specs/r10-scope.md §4.4, §4.6 row C2; roadmap Scale #2).

## What changed
- `CudaCache` (cuda/qwen2.rs): the KV buffer is an `Rc<DevBuf>` with `rows` (rows per K/V array) and `row0`
  (the row position 0 is at). A plain cache: `rows = max_ctx`, `row0 = 0`, so its layout and every kernel
  argument are what they were. `kv_args_at` and `cache_rows` (kvio) use `rows`/`row0`; `try_cache_at` became
  `make_cache` with an optional shared KV (no KV allocated, not zeroed again).
- `CudaQwen2::slot_view(pool, slot)` (slots.rs) and `BatchLm::slot_cache`: a cache whose KV is a pool slot's rows
  (same device memory), with scratch and CUDA graphs of its own. `adopt_cache` of a pool's own view into its slot
  copies nothing (into another slot: refused).
- `axon serve` (serve.rs `engine`): the pool is made first, and the single-request cache is `slot_cache(0)`
  instead of a separate `new_cache(ctx)`. The batch lends slot 0's token history to the engine for the solo run
  (`Batch::lend_solo` / `end_solo`), so prefix reuse works across solo and batched requests in that slot; a
  hand-over is `adopt` into slot 0 (bookkeeping only). With no pool (one request at a time, CPU, split,
  pipeline) nothing changes.
- Memory planning: `serve_bytes_kv` counts the solo path's scratch but not a second KV cache, so `plan_slots`
  fits N caches, not N + 1. The stderr note no longer says "2 slots and the single-request cache are 3 caches".
- `--parallel` help, README serve paragraph, results/roadmap/experiments-rejected rows updated.

## Why
A pool of N slots cost N + 1 KV caches: the request running alone (CUDA graphs, drafting) had its own cache
beside the pool. On a 12 GB card that extra cache is a whole slot: Qwen2.5-7B f16 at 32768 is 1.75 GiB a cache,
so bella served one request at a time at the trained context (MEASURED in #42).

Two ways were on the table: a view of slot 0 (keeps graphs and drafting at C=1), or no solo path once a pool
exists (the first request takes 1-row batch steps). Measured the second as a one-line switch on this branch
(bella, 7B, `--ctx 8192`, C=1, 3 rounds, pre-rebase binaries, 1017-041612-9431): **88.5 tok/s vs 103.5 main /
102.4 view, -14 %** (no CUDA graph of the solo decode step, no prompt-lookup drafting). Not within noise, so the
view it is.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| serve:: avante (sm_89) release | RELAVANTE | 1017-042200-3d54 |
| serve:: avante debug-assertions | DBGAVANTE | 1017-042201-36ec |
| serve:: bella (sm_120) release | RELBELLA | 1017-042201-25e8 |
| serve:: bella debug-assertions | DBGBELLA | 1017-042201-741e |
| full release suite, `--shard 3` | SHARDS | group 1017-042451-5252 |
| md5 of `gen`, main cf68e9d vs candidate, bella: CUDA f16, CUDA `--kv q8`, CPU (5 models x 3 prompts, -n 64) | 0/15, 0/15, 0/15 differ | 1017-042159-14aa |
| serve vs gen at C=1 (`scripts/serve_check.py`, 7B, bella): default (2 slots), `--kv q8` (5 slots), `--ctx 8192` (8 slots) | ALL OK x3 | 1017-042200-691a |
| new test `slot_view_matches_a_cache` (prefill, graphed decode, verify with rejected guesses, prefix re-feed in a view of slots 0 and 1: picks and KV bytes == a plain cache; other slots untouched; adopt no-op) and `pool_steps_match_solo_in_any_order` with the hand-over from a view | pass (avante) | 1017-041012-94ab |
| clippy -D warnings (bin + tests), rustfmt | clean | |

## Results
Slots (bella, RTX 5070 Ti Laptop 12 GB, Qwen2.5-7B Q4_K_M, startup banner; MEASURED, 1017-042200-5fdc):

| config | main cf68e9d | this branch |
|---|---|---|
| f16 `--ctx 32768` | 1 (one request at a time) | **2** (and the Q6_K prefill copy still made) |
| f16 default (no flags; trained 32768) | 1 | **2** |
| q8 `--ctx 32768` | 4 | **5** |

C=1 serve decode, client-side streamed tok/s, 256 tokens greedy, 3 interleaved rounds (each the median of 3
requests), cool-gated, same host (MEASURED, 1017-042200-5fdc):

| config | main | branch |
|---|---|---|
| `--ctx 8192` (both pooled: main's own solo cache vs slot 0's view) | 101.3 | 102.7 |
| `--ctx 32768` (main: one at a time; branch: 2 slots, solo in the view) | 101.4 | 102.8 |

Within noise (round spread ~±2 tok/s). Every request's text was identical across binaries.

## Not done / notes
- The view's scratch (attention partials, batch buffers; ~0.4 GiB on 7B at 32k) is still separate from the
  pool's; sharing it is possible but touches the batch passes. KV was the slot's worth.
- The solo path always uses slot 0. A request alone whose prefix sits in another slot re-prefills (main kept
  the solo cache's own history instead; same bits either way: cache reuse == fresh prefill).
