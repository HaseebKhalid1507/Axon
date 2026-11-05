# An exclusive KV RAM tier, and #170's idle flush in pieces (WP2)

- Date: 2026-11-05. Branch: feat/kvtier-exclusive. Base: main c54a730 (merged in at 84861fc). Status: draft PR #201;
  **the gate: s4 PASS, s1 PROVISIONAL (rerun queued), the resume cell PENDING** (below). Design: [kv-ram-tier-exclusive.md](../design/kv-ram-tier-exclusive.md)
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
  step's line. **Correction (after s4):** the `*_ms` fields are the host's time to *enqueue* the copies (they run
  async on the stream; s4: 11,745 pages "restored" in 15 ms), not their duration. The copies' device time shows in
  the next step's `ms` outside `feed_ms` and `dec_ms` (below). The design asked for this (§2.7) and checkpoint 2 left it out.
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
  stayed unmeasured. The s4 replay's restores ran in the fast mode (≈ 28.8 GB/s, ESTIMATED from the trace, below).

## Verification
| gate | job | result |
|---|---|---|
| the cache-RAM serve tests (off path, idle flush), a41d028, bella | 1105-115704-823c | 6 / 0 |
| `serve::` `kvstore::` `cuda::qwen2::slots::` release, a41d028, bella | 1105-115704-aeee | 192 / 0 (9 ignored) |
| same, debug assertions + overflow checks, bella | 1105-115846-69b9 | 192 / 0 (9 ignored) |
| md5 vs main c54a730, a41d028 (gate offload: pod1, RTX 4090) | 1105-115953-59cd | gen: 30 pairs, 0 differ. serve (#166's `serve_md5.py`, 7 sessions × 3 turns, evicting): 42 / 42 same in each of 5 cells (7B f16, 7B q8, gemma-3-1b f16, Qwen3-0.6B q8, CPU 0.5B), the arms being the candidate at default flags and with `--cache-ram 1G`; the tier arm restored 13,618–13,888 tokens a cell and gave main's digest |
| release at 626051c (the trace fields), bella | 1105-124048-55ca | 192 / 0 (9 ignored) |
| same, debug, bella | 1105-124048-b052 | 192 / 0 (9 ignored) |
| md5 vs main c54a730, 626051c (gate offload: pod1, RTX 4090) | 1105-124051-8cde | gen: 30 pairs, 0 differ. serve: 42 / 42 same in each of the 5 cells |
| the replay script's smoke (0.5B, 60 s window, 1G; an earlier summary) | 1105-124212-b8b6 | ran end to end; all 6 runs TAINTED (desktop load, mean 13–25 %), one md5: no timing read (rc 1, SHORT) |
| clippy `-D warnings` (test, bin), rustfmt | local | clean |

## The gate (design §6, verbatim; bella, §7)
The s4 and s1 replays: `agent_replay.py run` (main's copy c54a730, sha256 34b1910b…), board trace 713285ba…,
`--window-s 600 --scale 0.15 --clip 32768`, the board's axon server (`--ctx 32768 --parallel 16 --kv f16`), main
c54a730 vs 626051c `--cache-ram 6G`, 3 rounds B C / C B / B C, a fresh server per run (`replayexcl.sh`, staged as
`replayexcl-cef04a517ea0`). Medians over the clean rounds; per-run values in the job logs.

**s4, 1105-124635-28ee: PASS, clean.** MEASURED, bella, 2026-11-05 12:50–13:25. Job foreign GPU max 12 %, mean
0.1 % (not TAINTED); 3 / 3 rounds clean; every run's foreign max ≤ 16 %.

| s4 | main | cand | |
|---|---|---|---|
| ITL p99 (ms) | 139.51 | 136.75 | **0.980 × main (≤ 1.25): PASS** |
| re-prefill share | 0.2532 | **0.0372** | **≤ 5 %: PASS** |
| md5, all 6 runs | afec23b4a89f | afec23b4a89f | **SAME** |
| re-prefilled tokens | 739,721 | 108,758 | INFO |
| output tok/s | 49.18 | 92.22 | INFO |
| goodput tok/s | 8.21 | 20.69 | INFO |
| TTFT new / resume / turn p50 (s) | 18.46 / 16.85 / 0.268 | 6.49 / 11.15 / 0.167 | INFO (s4 overloads both) |
| wall (s) | 403.7–408.0 | 209.2–216.3 | INFO: main falls behind the trace, cand nearly keeps up |
| preempted | 0 | 0 | |

The step trace (§6 step 2; INFO, the ITL clause passed so there is no miss to attribute): no stall over 171 ms while
sequences decoded in any of the 6 runs. Step p99: main 139.5 / 140.6 / 140.2 ms, cand 129.6 / 136.7 / 136.8 ms.
Steps over 171 ms: main 1 / 3 / 1, cand 3 / 3 / 3, all prompt feeds (feed 121–299 ms); two of cand r0's also carry
a restore (483 and 260 pages, 62 and 33 ms). Per cand run the tier spilled 8,860–9,534 pages, restored
10,248–11,745 and wrote behind 1,897–1,914.

The copy rate, ESTIMATED from the trace: a step's `ms` less `feed_ms` and `dec_ms` is ≤ 0.8 ms in every one of
main's 15,036 steps; in cand's it exceeds 20 ms only on lines that carry copies, and scales with their pages.
Attributing that time to the copies: restore-only lines (n = 15–27 a run) **28.8 GB/s median in each run**,
spill-only lines (n = 19–20) 27.7–27.9 GB/s: the fast mode of the copy-path table above, not the 7.1 GB/s slow mode. Not
explained: on lines with both (≈ 250 spilled, ≈ 255 restored) the time is the restore's alone.

**s1, 1105-124636-2d32: PROVISIONAL (SHORT: 2 clean rounds of 3).** bella, 13:25–15:12, job TAINTED (Xorg, max
99 %, mean 1.2 %). Round level, agent_replay's rule (a run's foreign SM mean > 5 % or max > 25 %): cand r0 (1.7 / 49
%), cand r2 (1.3 / 50 %) and cand r2-a1 (2.2 / 44 %) tripped the max; their rounds' main runs (16, 25, 19 %) were
clean but aren't read alone. The job's two extra rounds were spent (r0-a1, r2-a1), so round 2 has no clean pair.
Clean rounds r0-a1 and r1:

| s1 (2 clean rounds) | main | cand | |
|---|---|---|---|
| ITL p99 (ms) | 126.91 (127.05, 126.77) | 123.05 (122.73, 123.36) | 0.970 × main (≤ 1.05) |
| md5, the 4 clean runs | afec23b4a89f | afec23b4a89f | same (all 10 runs give it) |
| re-prefill share | 0.1429 | 0.0372 | INFO |
| re-prefilled tokens | 417,494 | 108,715 | INFO |
| TTFT resume p50 (s) | 0.746 | 0.267 | INFO (the ≤ 0.3 s clause is the resume cell's) |
| TTFT new / turn p50 (s) | 5.20 / 0.126 | 3.87 / 0.118 | INFO |
| goodput tok/s | 16.80 | 19.95 | INFO |
| output tok/s | 32.99 | 32.99 | INFO (the trace's pace at s1) |

The rule asks for 3 clean rounds, so s1 is rerun as **1105-151540-c76b** (`kvexcl-626051c-replay-s1-r2`, the same
script and inputs; 2d32's outputs moved aside to `out-s1.2d32` so the rerun's summary can't read them). Decided
before it runs: c76b is read **on its own**; 2d32 stays PROVISIONAL and is not pooled with it. If c76b is SHORT too,
that is reported, not pooled. Arithmetic, not a reading: with two of three values per binary known, a third clean
round can only move the s1 ITL ratio within 0.966–0.973.

**Resume cell, 1105-124636-68e9: PENDING** (`kv_resume_bench-440fe18.py`, 3 rounds, `--gap 120`, after
`kvio::tests::bandwidth` at 25.6k).
