# serve: no N+1 cache, the single-request path runs in slot 0  (PR #60)

- Date: 2026-10-17. Branch: feat/serve-one-cache. Base: main cf68e9d (after #57). Status: open (PR #60)
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
| serve:: avante (sm_89) release | 71 passed, 0 failed, 2 ignored | 1017-042200-3d54 |
| serve:: avante debug-assertions | 71 / 0 / 2 | 1017-042201-36ec |
| serve:: bella (sm_120) release | 71 / 0 / 2 | 1017-042201-25e8 |
| serve:: bella debug-assertions | 71 / 0 / 2 | 1017-042201-741e |
| full release suite, `--shard 3` | PASS: 471 passed, 0 failed, 35 ignored (avante + bella) | group 1017-042451-5252 |
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
- Q6_K prefill copy at `--kv q8 --ctx 32768`: not made with 5 slots ("0.79 GiB free beside the caches"), and
  not made on main's 4 slots either (same note, 1017-042200-5fdc). At f16 32768 it is made with 2 slots.

## Review round 1 (shady, 2a4eed6: SHIP-AFTER-FIXES)
- Slot choice (`engine::best_slot`): ties now go to the highest free slot, so slot 0 is taken last. At first
  ties went to slot 0, which let any batched admit with no prefix match evict the request-alone's prefix there.
  On main the solo cache was out of reach of batch traffic; now batch traffic takes slot 0 only when every
  other slot is busy or slot 0 holds the longest match. Unit test `best_slot_prefers_prefix_then_keeps_slot_0`.
- kvsave rows of a view (`kvio::Rows::len`): `save_kv`/`load_kv` on a slot view are bounded by the slot
  (`row0 + max_ctx`), not the pool's rows, so they can't reach the next slot. Tested in
  `slot_view_matches_a_cache` (save of 0..ctx works, ..ctx+1 and a load past the slot are refused; a view's
  saved image == a plain cache's).
- `adopt_cache` returns the bytes copied: 0 for the slot's own view, >0 for a plain cache (both asserted, also
  in `pool_steps_match_solo_in_any_order`); the copy path refuses a source with `row0 != 0`.
- `slot_view_matches_a_cache` also asserts its verify pass rejected a guess, and that every other slot's rows
  are byte-identical before and after a view runs.
- New serve test `serve_solo_and_batch_share_slot_0`, 2 slots: (a) a request the batch admits into slot 0, then
  a request alone sharing its prefix: cached_tokens covers the prefix and the reply == alone; (b) alone ->
  hand-over -> finishes batched -> a request alone sharing its prefix: the same.
- `layer_kv` (pipeline.rs, tests) documents that on a view it dumps the whole pool.

## Rebased on #61 (serve ops pack)
- `GET /slots` (#61) published the solo cache as a separate `solo` entry. Here the solo path runs in slot 0, so
  that entry is folded into `slots[0]`: `Engine::run` publishes slot 0 busy with its request and its held tokens
  after (`Metrics::publish_slot0`, replacing `publish_solo`); `SlotsView` has no `solo`, so a request alone is
  counted once; with one request at a time the one cache is `slots[0]` (a free slot 0 before any request).
  `serve_cuda_slots` now runs a request alone first (slot 0 busy, slot 1 free, 1 busy / 1 free in /metrics, its
  prefix held in slot 0 after); `serve_metrics_slots_max_prompt` and the slot-idle test read `slots[0]`; the
  `slot_counts` unit test covers the fold. README and #61's log entry (an Update line) say so.
