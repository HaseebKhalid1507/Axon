# Pool attention: grid y by the rows' depth, not the slot cap  (PR #N, gates pending)

- Date: 2026-10-18. Branch: perf/attn-grid-depth. Base: origin/feat/paged-kv-pool 5d6be54 (#71) + origin/main
  2b89410 merged in (6fa800e; README conflict only). Stacked on #71. Status: open, gates pending.
- Pick / design: the lead left by the r13 pick 3 NO-GO (docs/log/2026-10-18-attn-one-launch.md).

## Why this branch is on #71, not main

The sizing line (`min(max_split, nblk_max)` in `attn_half`) is on main too. But main's serve pool is plain: a slot of
32768 positions costs its whole KV, so a 7B on bella gets 2–3 such slots. The 16–64 slots of 32768 positions that make
the empty CTAs matter exist only with #71's paged pool (16 slots by default; slot = a cap, pages on demand). The A/B
harness also needs #71's `pool_with` to build such a pool at bella's VRAM.

## What changed

- `att_grid(nblk_max, pos_max)` (batch.rs): attn_blk_b's grid y = min(max_split, nblk_max), clamped to the
  deepest query's blocks rounded up to a power of two. A pool step computes it on the host from its rows' positions
  (already on the host for `upload_rows`; no device sync) and stores it in `BatchBufs::att_gy`. The step's graph key
  becomes (nb, greedy, device-sampled, att_gy): power-of-two buckets keep it to ≤ 9 graphs per step shape over a
  32768 cap (LRU of 64 as before). One-sequence per-query passes (verify, ≤ att_per_query tokens, not captured)
  size it by start + nb.
- `AXON_ATTN_GRID_CAP=1` (`BatchFns::grid_cap`): the old sizing, for A/Bs.
- Tests: `pool_steps_past_gemvm16_match_solo` also runs a paged pool of 2048-position slots with the grid by depth
  (asserts grid y < the cap sizing, ≥ the blocks needed) and with `grid_cap`; pool == solo in both.
- `pool_step_cost`: `AXON_STEP_GCAP` (arms interleaved in one process), `AXON_STEP_CAP` (a paged pool of slots of that
  many positions holding only the depth's pages, as serve), GPU temperature + SM clock per row, profile rows of
  paged pools at the right table base.

## Bits: why any grid y gives the same results

decode.cu attn_blk_t: a CTA with `blockIdx.y >= nblk` returns before any work. Otherwise it loops
`for (b = blockIdx.y; b < nblk; b += gridDim.y)`, and each block b writes its own partial (part_o/part_ml at
`hb = head * nblk_max + b`), with nothing carried between iterations. The q8 new-row store goes to the CTA with
`blast % gridDim.y == blockIdx.y`, which exists for any gridDim.y ≥ 1. attn_cmb_b folds the partials in block order,
independent of the grid. So grid y changes which CTA computes a block, never what it computes. Existing proofs:
`gpu_decode_attention_long` (decode at grid y 1, 7 and 10 000 → clamped to nblk_max, i.e. far more CTAs than blocks,
== batched prefill logits, at 8191–32511 in a 32768 cache, f16 and q8) and `gpu_batched_attention_matches_decode_bits`
(attn_blk_b at grid y 1–46 == the decode partials == attn_fa_b, all layouts incl. q8).

## Estimate before building (ESTIMATED)

7B (4 kv heads), bella (46 SMs, max_split = 8 · 46 / 4 = 92), slot cap 32768 (nblk_max 512), depth 600 (10 blocks):
grid y 92, of which 10 are live per (kv head, query).

| nb | CTAs / layer | empty / layer | empty / step (28 layers) |
|---|---|---|---|
| 16 | 5 888 | 5 248 | 147 k |
| 32 | 11 776 | 10 496 | 294 k |
| 64 | 23 552 | 20 992 | 588 k |

An empty CTA still reads rp (its position) and returns, and it holds a slot first. attn_blk_b at head size 128 has
~32 KiB of static shared memory (K and V tiles), so about 3 CTAs fit per SM: ~138 on bella. At ~0.5 µs per empty
CTA (dispatch + one load + exit) that's ~3.6 ns per CTA GPU-wide: nb 32 ≈ 1.1 ms, nb 64 ≈ 2.1 ms per step. The
floor, ignoring the shared-memory limit (~0.5 ns per CTA), is 0.15 / 0.29 ms. Both clear the 0.1 ms bar. Corroborating
(one-launch arm, no graph, 1018-090100-a107): bella 7B attention at depth 600 costs 2.43 ms at nb 32 and 6.62 ms at
nb 64. That's 2.7× for 2× the KV bytes, and ~330 GB/s at nb 64.

## How it is verified (pending: job ids)

| gate | job | host | status |
|---|---|---|---|
| bit tests, release (`gpu_batched_attention_matches_decode_bits gpu_decode_attention_long pool_steps_past_gemvm16_match_solo gpu_paged_split_and_batched_bits serve_identity`) | 1018-203538-4d30 | avante | pending |
| the same, debug assertions | 1018-203538-915c | bella | pending |
| gen md5 main 2b89410 vs 5f8e745, 6 models x 5 prompts, -n 64 | 1018-203539-2f4a | bella | pending |
| A/B 7B, depth 600/4096, nb 16/32/64, cap 32768, x5 interleaved (+ cap-arm profile) | 1018-203539-8d76 | bella `--timing` | pending |
| A/B 1.5B, the same | 1018-203541-262a | avante `--timing` | pending |
| clippy -D warnings (bin + tests), rustfmt | local | — | pass @5f8e745 |

Not submitted yet: the full release / debug suites (to follow if the A/B passes).

## Results

Pending.

## What didn't work

- r13 pick 3 (one launch for all rows): NO-GO; see its log. This lead came out of its profile.
- Not done here: the solo decode step has the same sizing (grid y = min(max_split, nblk_max) for a 32768 cache:
  ~328 empty CTAs per layer at depth 600, ~9 k per token, ESTIMATED ~0.05–0.1 ms of ~10 ms). Its graphs are per
  cache, so it would need the same keying. A follow-up if this one pays.
