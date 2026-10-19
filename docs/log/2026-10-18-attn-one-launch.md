# One attention launch for all pool rows (r13 pick 3)  (NO-GO)

- Date: 2026-10-18. Branch: feat/attn-one-launch (code at 1a55a4b, not for merge). Base: origin/feat/paged-kv-pool
  665115c (#71) + origin/main 953b886 merged in (a85aa1c; for #81's `pool_step_cost`). Status: rejected, no PR.
- Pick / design: specs r13-perf-pick §4. Hypothesis: the pool's attn_blk_b runs once per att_per_query (16 for head
  size 128, 48 for 64) rows, each launch latency-bound, so one launch over all nb rows should save ~1.1 ms at 7B nb 64
  depth 600 (ESTIMATED from 4090 profiles). Kill rule: pool_step_cost at nb 64 depth 600, interleaved one-launch vs
  today: drop it if it saves < 0.5 ms, or if the partials growth costs > 1 % of #71's pages at the default ctx on bella.

## What changed (on the branch only)

- `att_rows` (batch.rs): no new memory. The partials buffer already holds `min(max, att_per_query) x nblk_max(ctx)`
  blocks per head. A pool step at depth d needs only `ceil((d + 1) / 64)` blocks per query, so the same buffer holds
  `parts / need` queries. The step takes the fewest launches that fit, splits the rows evenly, and spaces their
  partials `parts / att_z` blocks apart (the stride is only a layout; the kernel and combine are unchanged, so the
  bits are the same). At serve's paged cap of 32768, nb 64 is one launch at depth 600 and at 4096.
- Pool graphs are keyed by `att_z` too. `AXON_ATT_ONE=0` restores the per-att_per_query launches.
- Tests: `gpu_batched_attention_matches_decode_bits` with partial strides nblk_max + 0 / 1 / 37;
  `pool_steps_past_gemvm16_match_solo` with the switch off, on (2 launches at nb 64) and on with a 2048 paged pool
  (asserts 1 launch).
- `pool_step_cost`: `AXON_STEP_ONE` (on/off interleaved in one process), `AXON_STEP_CAP` (a paged pool like serve's,
  holding only the pages the depth needs), GPU temperature and SM clock at the start of each row.

The kill rule's memory clause passes by construction: 0 bytes added.

## How it was verified

- Smoke, bella, 1018-090100-8e1f: the three tests above pass (rc 0).
- Timing, exclusive, one process per model, arms interleaved 0,1,0,1,0,1 (each a median of 5 x 40 graph steps):
  bella RTX 5070 Ti Laptop sm_120 46 SMs (1018-090100-a107: 7B + 0.5B, then a 7B reversed-order pass 1,0,1,0),
  avante RTX 4050 Laptop sm_89 20 SMs (1018-090129-1f7e: 1.5B + 0.5B). Pool: paged, cap 32768, the depth's pages
  only. GPU at the start of each row: bella 71–86 C, 2422–2700 MHz; avante 82–88 C, 1995–2430 MHz. Clocks stayed
  close between the arms of a cell; the separation is in the run-to-run spread below.
- The 7B at depth 4096 fit only 16 slots on bella (32 x 4160 positions needs 7.56 GiB, 5.30 free); the 1.5B fit 32 on avante.

## Results (MEASURED; ms/step, median over the interleaved rounds; Δ = one-launch − today)

Launches today: nb / 16 for head size 128 (7B, 1.5B); one launch for the 0.5B up to nb 48, then two (48 + 16).
So every nb 16 cell, and the 0.5B's nb 32 cells, are controls: the same launches, only the stride differs.

| host | model | pass | depth | nb | today | one launch | Δ ms | Δ % | rounds |
|---|---|---|---|---|---|---|---|---|---|
| bella | 7B | 1 | 600 | 16 | 20.845 | 20.878 | +0.033 | +0.2 % | 3+3 |
| bella | 7B | 1 | 600 | 32 | 24.466 | 24.335 | −0.131 | −0.5 % | 3+3 |
| bella | 7B | 1 | 600 | 64 | 29.370 | 29.424 | +0.054 | +0.2 % | 3+3 |
| bella | 7B | 2 (rev) | 600 | 32 | 24.742 | 24.460 | −0.282 | −1.1 % | 2+2 |
| bella | 7B | 2 (rev) | 600 | 64 | 29.616 | 29.558 | −0.058 | −0.2 % | 2+2 |
| bella | 7B | 1 | 4096 | 16 | 28.688 | 28.636 | −0.052 | −0.2 % | 3+3 |
| bella | 0.5B | 1 | 600 | 16 | 3.296 | 3.310 | +0.014 | +0.4 % | 3+3 |
| bella | 0.5B | 1 | 600 | 32 | 4.777 | 4.765 | −0.012 | −0.3 % | 3+3 |
| bella | 0.5B | 1 | 600 | 64 | 5.966 | 5.851 | −0.115 | −1.9 % | 3+3 |
| bella | 0.5B | 1 | 4096 | 16 | 4.796 | 4.776 | −0.020 | −0.4 % | 3+3 |
| bella | 0.5B | 1 | 4096 | 32 | 8.166 | 8.107 | −0.059 | −0.7 % | 3+3 |
| bella | 0.5B | 1 | 4096 | 64 | 13.116 | 13.491 | +0.375 | +2.9 % | 3+3 |
| avante | 1.5B | 1 | 600 | 16 | 10.662 | 10.692 | +0.030 | +0.3 % | 3+3 |
| avante | 1.5B | 1 | 600 | 32 | 14.578 | 14.643 | +0.065 | +0.4 % | 3+3 |
| avante | 1.5B | 1 | 600 | 64 | 19.732 | 19.736 | +0.004 | +0.0 % | 3+3 |
| avante | 1.5B | 1 | 4096 | 16 | 21.173 | 21.196 | +0.023 | +0.1 % | 3+3 |
| avante | 1.5B | 1 | 4096 | 32 | 35.381 | 35.819 | +0.438 | +1.2 % | 3+3 |
| avante | 0.5B | 1 | 600 | 16 | 5.439 | 5.413 | −0.026 | −0.5 % | 3+3 |
| avante | 0.5B | 1 | 600 | 32 | 7.496 | 7.493 | −0.003 | −0.0 % | 3+3 |
| avante | 0.5B | 1 | 600 | 64 | 9.928 | 10.031 | +0.103 | +1.0 % | 3+3 |
| avante | 0.5B | 1 | 4096 | 16 | 9.750 | 9.819 | +0.069 | +0.7 % | 3+3 |
| avante | 0.5B | 1 | 4096 | 32 | 16.899 | 17.061 | +0.162 | +1.0 % | 3+3 |
| avante | 0.5B | 1 | 4096 | 64 | 28.585 | 29.178 | +0.593 | +2.1 % | 3+3 |

Noise: the controls (same launches) land within ±0.7 %. The depth losses are outside every round's spread
(bella 0.5B 4096 nb 64: today 13.04–13.13, one launch 13.39–13.51; avante 1.5B 4096 nb 32: 35.38 vs 35.78–35.84;
avante 0.5B 4096 nb 64: 28.58–28.60 vs 29.16–29.19).

**Kill rule: FAILS.** 7B nb 64 depth 600 on bella: +0.054 ms (pass 1), −0.058 ms (pass 2), against a required
−0.5 ms. It is flat at the cell the pick was aimed at. Memory clause: passes (0 bytes).

## Why it didn't pay

Profile lines (no graph, per-segment events summed over layers). They ran with the switch's last value, so they show
the one-launch arm only; there is no per-arm attention split.

| host | model | depth | nb | attn + combine | step | share |
|---|---|---|---|---|---|---|
| bella | 7B | 600 | 32 | 2.431 + 0.241 | 25.04 | 11 % |
| bella | 7B | 600 | 64 | 6.617 + 0.534 | 30.67 | 23 % |
| bella | 0.5B | 600 | 64 | 1.084 + 0.149 | 6.77 | 18 % |
| bella | 0.5B | 4096 | 64 | 7.655 + 0.854 | 13.76 | 62 % |
| avante | 1.5B | 600 | 64 | 7.465 + 0.643 | 20.23 | 40 % |
| avante | 1.5B | 4096 | 32 | 23.085 + 2.196 | 36.69 | 69 % |
| avante | 0.5B | 4096 | 64 | 20.448 + 2.509 | 29.97 | 77 % |

- The premise was occupancy measured on a 128-SM 4090: one per-16 launch at depth 600 has ~640 live one-warp CTAs
  (4 kv heads x 10 blocks x 16), about 5 per SM there. On bella's 46 SMs that is already ~14 per SM, and on avante's
  20 SMs (1.5B: 2 x 10 x 16) 16 per SM. The per-16 launches were not starved on these cards, so merging them buys
  nothing at depth 600 (ESTIMATED from the grid shapes; consistent with the flat cells).
- At depth 4096, one launch is 1–3 % slower. The cause is not isolated; one-launch-only profiles can't settle it.
  Candidates (UNMEASURED): 4x the KV streams and 4x the partials in flight per layer before the combine (0.5B 4096
  nb 64: ~15 MB of partials per layer vs ~3.8 MB, against an L2 of tens of MB), and DRAM locality with 64 sequences
  read at once instead of 16.
- Attention reads the 7B's KV at about a third of bandwidth at depth 600: 64 x 601 positions x 56 KiB ≈ 2.2 GB in 6.6 ms
  ≈ 330 GB/s (ESTIMATED bytes; MEASURED time, no graph). So bytes aren't the limit there, and neither, per the table,
  is the number of launches. A lead, unmeasured: with serve's 32768 cap, grid y is max_split = 92 (bella 7B) or 184
  (0.5B), and nblk_max = 512. So at depth 600 only 10 of the 92 CTAs per (kv head, query) have a block; the
  rest exit at once (~89 % empty CTAs in both arms). The old harness's pool (ctx = depth + 64) had grid y 11, which
  hid this. Sizing grid y by the rows' depth (bits-safe per the "any grid y gives the same bits" contract, keyed in
  the graph like att_z) is the next thing to measure.

## What would have to be true for it to win

- Many SMs and shallow depth, so that one per-16 launch underfills the GPU: the 4090 (128 SMs) or larger, at depth
  ≤ ~1k, nb 32–64. That's where the spec's −1.1 ms was estimated, and it is **not measured** (pods out of budget).
  Retry only if a 4090 run of `AXON_STEP_ONE=0,1,0,1 AXON_STEP_CAP=32768 AXON_STEP_DEPTHS=600 AXON_STEP_NB=32,64`
  shows ≥ 0.5 ms at 7B nb 64.
- A depth-gated version (one launch only below some depth) isn't worth it on bella or avante. Its best measured
  cells are bella 7B 600 nb 32 −0.13 / −0.28 ms (−0.5 / −1.1 %) and bella 0.5B 600 nb 64 −0.12 ms (−1.9 %). The
  target cell (7B 600 nb 64) is flat, and avante gains nowhere. All of that is under the 0.5 ms bar.

## What didn't work

- The pick itself, as above. The code stays on feat/attn-one-launch (1a55a4b) for a 4090 retry; the stride-generalised
  bit test and `pool_step_cost`'s `AXON_STEP_CAP` / `AXON_STEP_ONE` / temperature logging are worth keeping if it is retried.
