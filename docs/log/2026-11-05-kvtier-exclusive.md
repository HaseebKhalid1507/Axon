# An exclusive KV RAM tier, and #170's idle flush in pieces (WP2)

- Date: 2026-11-05. Branch: feat/kvtier-exclusive. Base: main c54a730 (merged in at 84861fc). Status: draft PR,
  **the gate's bella cells PENDING** (below). Design: [kv-ram-tier-exclusive.md](../design/kv-ram-tier-exclusive.md)
  (revision 3), reviewed at revision 0 by shady (7/10 REVISE) and Fable (BUILD-WITH-CHANGES).
- Pick: foreman (axon), plan `specs/agent-workload-plan.md` §3 row 2 (Fable). Builder: joestar.

## Why
#166's tier (`--cache-ram`, opt-in) is inclusive: pages are copied to RAM as they fill, so the tier holds about its
own size, the GPU pool's pages included. On bella 6 GiB is 1,755 pages against a 1,570–1,744-page pool, so it only
helps when the work written during a conversation's gap lands between the two (≈ 11–185 pages). Rule B's bella cell
missed its TTFT-resume clause (1031-063509-f763: 6,592 of 26,496 tokens restored, 7.05 s against main's 8.68 s),
and #166 shipped opt-in. The plan's next step is an exclusive tier: RAM keeps what the GPU doesn't. #170, found in
#166's review: the flush at the end of a request running alone had no budget and could hold the next request for
the whole backlog.

## What changed
- **`kvstore` (3e4a306, checkpoint 1):** two eviction classes in one tree. Class 0 holds copies of pages a GPU slot
  still holds; it is evicted first, newest first. Class 1 holds RAM's own pages, evicted least recently used.
  Also `has_room` / `insert_behind`, an insert that never evicts, used by the write-behind, and the keys an eviction
  dropped (for the tier's per-slot lists). A seeded fuzz runs 64 seeds, about 8,800 evictions, checking the tree's
  invariants after every op.
- **The tier's policy (c9702e9, checkpoint 2):** a slot's stored pages are class 0. An eviction pops them from every
  slot's list at once (`Tier::drain`). The write-behind copies only into free room, slots in reclaim order (the next
  to give its rows up first). A spill moves the pages past what the slot keeps to class 1 without copying them
  again. A restore marks what it moved as class 0. `Tier::check` and `check_slot` assert the lists against the
  slots' rows in debug and test builds.
- **#170 (c9702e9):** `end_solo`'s flush is gone. An idle batch with a backlog writes it in 64 MiB pieces, and the
  engine loop takes a waiting request between two pieces. A piece that copies nothing stalls the flush until the
  batch steps, admits or the tier grows, so an idle server never spins. A stop returns at once.
- **Fix (a41d028), found by the new serve test (a):** a spill that gives nothing up copies nothing. A
  conversation's next turn in its own slot used to copy its whole unstored chain before its first token. On a full
  tier that is ≈ 269 pages (987 MB, ≈ 35 ms at 28 GB/s) per turn, evicting RAM's own pages to hold duplicates (design §2.3; the
  lead: in scope).
- **/metrics:** `axon_kv_ram_gpu_pages` (class 0) and `axon_kv_ram_spilled_pages_total`.
- **The step trace (626051c, checkpoint 3):** `AXON_SERVE_TRACE` lines carry the tier's copies since the line
  before (`spilled`/`spill_ms`, `restored`/`restore_ms`, `wb`/`wb_ms`). An admission's copies land on the next
  step's line and explain the gap before it. The design asked for this (§2.7) and checkpoint 2 left it out.
- **Tests:** `Tier` unit tests and a fuzz; #181's off-path tests kept green (the Spill case is retargeted to a
  2-page tier at turn 3); `serve_cuda_cache_ram_idle_flush` (design §3's serve tests (a)–(c)) through test-only
  knobs on `tier::Faults` (piece size, a park after piece k, the loop's events, a starved arena). The release
  binary has no knob symbols and no fault strings.

## The copy path (checkpoint 1): `Rows::page` as it is
Bella's 2D page copy ran at 7.1 GB/s in f763 and in two other runs of ten, and at 28 GB/s in the rest. The plan
priced a fix: a 56 × 1D loop per page, or a staged gather. MEASURED on bella (`kvio::tests::bandwidth`, 8c7cf14,
D2H / H2D GB/s; all three runs TAINTED by one Xorg sample; read anyway on the lead's call, design §2.8.2):

| row | 7732, 25.6k (53.5 % foreign) | 139c, 25.6k (8.3 %) | 0d10, 32k (10.9 %) | busy (139c / 0d10) |
|---|---|---|---|---|
| contiguous | 28.63 / 28.68 | 28.64 / 28.66 | 28.64 / 28.68 | |
| 2D per 64-pos page (`Rows::page`) | 28.07 / 28.27 | 28.11 / 28.26 | 28.05 / 28.26 | 28.31 / 28.39, 28.30 / 28.39 |
| 56 × 1D per page (the loop) | 16.72 / 17.45 | 16.76 / 17.30 | 16.41 / 17.26 | 18.15 / 18.52, 17.94 / 18.60 |
| staged gather / scatter path | 25.54 / 25.57 | 25.45 / 25.58 | 25.58 / 25.65 | 25.75 / 25.86, 25.84 / 25.92 |

- **The loop: rejected.** It is enqueue-bound (host time 101 of its 105 ms at 32k: ≈ 3.5 µs per copy, 28,672 copies), under
  20 GB/s even with the GPU at full clocks.
- **The staged gather: rejected.** It costs pool pages and is slower than today's copy.
- **`Rows::page` stays.** No copy-path PR.
- **What didn't work:** the hypothesis that an idle desktop leaves the GPU in a low clock state and slows 2D
  copies. Both -r2 runs started in P8 with the desktop near idle and were fast from their first row. The slow mode
  stayed unmeasured; the resume cell's trace gives the rate its restores got (below).

## Verification
| gate | job | result |
|---|---|---|
| the cache-RAM serve tests (off path, idle flush), a41d028, bella | 1105-115704-823c | 6 / 0 |
| `serve::` `kvstore::` `cuda::qwen2::slots::` release, a41d028, bella | 1105-115704-aeee | 192 / 0 (9 ignored) |
| same, debug assertions + overflow checks, bella | 1105-115846-69b9 | 192 / 0 (9 ignored) |
| md5 vs main c54a730, a41d028 (gate offload: pod1, RTX 4090) | 1105-115953-59cd | gen: 30 pairs, 0 differ. serve (#166's `serve_md5.py`, 7 sessions × 3 turns, evicting): 42 / 42 same in each of 5 cells (7B f16, 7B q8, gemma-3-1b f16, Qwen3-0.6B q8, CPU 0.5B), the arms being the candidate at default flags and with `--cache-ram 1G`; the tier arm restored 13,618–13,888 tokens a cell and gave main's digest |
| release / debug / md5 at 626051c (the trace fields) | 1105-124048-55ca / -b052, 1105-124051-8cde | PENDING |
| clippy `-D warnings` (test, bin), rustfmt | local | clean |

## The gate (design §6, verbatim; bella, §7)
PENDING: the s4 and s1 replays (`agent_replay.py run`, main's copy c54a730 sha256 34b1910b…, board trace
713285ba…) and the resume cell (`kv_resume_bench-440fe18.py`, 3 rounds, `--gap 120`), main c54a730 vs 626051c with
`--cache-ram 6G`.
