# An exclusive KV RAM tier, and #170's idle flush in pieces (WP2)

- Date: 2026-11-05. Branch: feat/kvtier-exclusive. Base: main c54a730 (merged in at 84861fc). Status: draft PR #201;
  **the gate's five clauses PASS: s4 on bella; s1 and the resume cell on an RTX 4090 (the lead's reroute, so 4090
  numbers)**. `--cache-ram` stays opt-in in this PR (verdict below). Design: [kv-ram-tier-exclusive.md](../design/kv-ram-tier-exclusive.md)
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

**Moved off bella (the lead, 11-05 18:30).** The resume cell (68e9) and the s1 rerun (c76b) waited 5 h 44 min and
3 h 15 min behind bella's desktop GPU load. The lead rerouted both by hand to a secure RTX 4090 pod (pod1). Note §7
says "bella only", so their numbers are **4090 numbers**, labelled as such, never compared with bella's.

**Resume cell, 1105-124636-68e9 (pod1, RTX 4090): VOID, 6 of 6 rounds, rc 3; a harness bug.** Every round read
"not evicted: 26544 of 26692 prompt tokens still in the GPU pool (pool 5288 pages, pressure +8, 4959 kept)": the
resume was a prefix-cache hit (0.088 s on both binaries) and the candidate restored nothing (0 restores in its
traces). The bench sends pressure sessions until the floor page counts of the session and everything after it exceed
the pool the server logged, but `kv_resume_bench.py` @440fe18 also stopped after 8 of them. On bella's
1,570–1,744-page pools the loop needs 0–1, so the cap never bound. The 4090's pool is 5,288 pages: the session
(414 pages), its 3 others and 8 pressure sessions keep 4,959, so nothing had to leave and the check voided every
round, as it should. **Fix (harness only), fe5117d:** the cap follows the pool (pool / a session's pages + 1, 8 at
least): 13 there. a169 sent 9 in every round, 5,371 > 5,288 pages kept (MEASURED; 5,372 estimated beforehand).
With bella's pools the loop sends the same sessions as before.
`resumeexcl.sh` also stops printing `restored × 3.5 MiB / restore_ms` as a rate (that field is enqueue time, above).
It prints the step-time estimate instead, on lines whose only copy is the restore. Rerun: **1105-191445-a169**
(`kv_resume_bench-fe5117d.py`, same binaries, `--timing 4090`), below. 68e9's bandwidth rows (MEASURED, 4090,
25.6k): 2D per 64-pos page 12.83 / 13.12 GB/s D2H / H2D, contiguous 12.92 / 13.26: every layout ≈ 13 GB/s, idle
and busy alike (the pod's own host link; about half of bella's 28 GB/s fast mode).

**s1 rerun, 1105-151540-c76b (pod1, RTX 4090): PASS, 3 clean rounds of 3.** 2026-11-05 19:11–20:14; job
foreign GPU max 0 %, every run `tainted=False`. Read on its own, as decided before it ran; 2d32 (bella) stays
PROVISIONAL and is not pooled with it. Medians of 3 (raw in brackets):

| s1, 4090 | main | cand | |
|---|---|---|---|
| ITL p99 (ms) | 11.24 (11.24, 11.24, 11.19) | 11.52 (11.68, 11.52, 11.45) | **1.025 × main (≤ 1.05): PASS** |
| md5, all 6 runs | 2aee881bcf75 | 2aee881bcf75 | **SAME** (per host: bella's runs give afec23b4a89f) |
| re-prefilled tokens / share | 21,291 / 0.0073 | 7,359 / 0.0025 | INFO |
| TTFT resume p50 (s) | 0.066 (0.071, 0.066, 0.064) | 0.092 (0.102, 0.092, 0.091) | INFO: +26 ms, below |
| TTFT turn / new p50 (s) | 0.063 / 1.463 | 0.068 / 1.402 | INFO |
| goodput / output tok/s | 33.1 / 33.15 | 32.95 / 33.15 | INFO |

Per cand run (identical in all 3): 1,755 pages pinned (6,440,878,080 bytes), 9,780 written (1,755 by the write-behind,
the tier's size, then 8,025 by spills), 15,724 tokens restored in 21 hits.

**Resume cell rerun, 1105-191445-a169 (pod1, RTX 4090): PASS, 3 of 3 rounds valid.** 2026-11-05 20:14–20:33, foreign
GPU 0 of 111 samples over 25 %. Every round: pool 5,288 pages, pressure +9 (5,371 pages kept), main's
resume found 0 tokens cached (a full re-prefill), the candidate's restored all 26,496.

| resume, 4090 (25k session evicted, `--gap 120`) | main | cand | |
|---|---|---|---|
| TTFT resume (s) | 2.834 (2.834, 2.834, 2.836) | **0.204** (0.204, 0.203, 0.209) | **≤ 0.3 s: PASS**; 13.9× |
| tokens cached / restored | 0 / – | 26,496 / 26,496 | |
| ITL p99 while a 25k prompt prefills (ms) | 89.23 | 90.77 | INFO: +1.7 % (the bench's write-behind cell) |
| ITL p50 / p99, decode only (ms) | 6.638 / 7.633 | 6.651 / 7.757 | INFO |
| startup (s) | 2.007 | 2.062 | INFO: 6 GiB pinned at start |
| md5 | 78722066e051 | 78722066e051 | SAME |

Each restore (414 pages) was enqueued in 1.6–2.7 ms. Its step line also carries the idle flush's 1,739 write-behind
pages from the gap, so the trace gives no rate. At the 4090's MEASURED 0.280 ms per page (H2D), the copies are
≈ 116 ms of the 204 (ESTIMATED).

**The s1 TTFT-resume p50 on the 4090 (+26 ms; INFO, not a clause): the tail spill.** Each request paired across the
binaries (the replay is deterministic), its median over the 3 rounds (`s1_per_request.py` in the gate scripts):
- Resumes: 14 of 34 are slower by > 10 ms (+14 to +135 ms), 1 is faster; per-request median +4.6 ms. Every slow
  resume computed the same tokens as main's, so none of it is re-prefill. Turns: 21 of 127 slower, 7 faster.
  New requests computed 4.4 % fewer tokens (169,085 → 161,661: shared prefixes restored) and got 52.6 ms faster.
- 26 of the 105 requests that give up ≥ 1 page at admission are slower by > 10 ms. Only 1 of the 35 that give up
  nothing is. A request gives up a page when its previous turn's generated tail crossed a page boundary: the replay's
  next prompt carries the trace's text, not the model's reply, so the slot keeps only the previous prompt.
- 15 of those 26 cost 0.85–1.10× (median 0.90) a D2H copy of their **whole chain** at the 4090's 0.286 ms per page
  (254–255 pages: 72.6–72.9 ms against +64 to +77 measured; 485 pages, #143: 138.7 ms against +135). The rest are partial.
- The code agrees. `spill` → `PoolTier::write` copies every unstored whole page from the first unstored one
  (`write_pages`: `stored(s)..held.len()/PAGE`), the pages the slot keeps included, because the tree stores a page
  only under its stored parent. With the tier full, a chain's class-0 copies are the first evicted by other spills,
  so its next tail spill copies the whole chain again, before the request's first token. 8,025 spilled pages a run is
  ≈ 2.3 s of copies at 0.286 ms (ESTIMATED), to restore ≈ 246 pages' worth of tokens.
- This is the rest of a41d028's defect. That fix skips a spill that gives up nothing; a spill that gives up one page
  still copies the chain. Fixed in e1df27a (below), the s1 and s4 cells rerun.
- Against the resume cell's 13.9×: the win comes when the session was evicted (a169), the cost when it wasn't. On a
  24 GB card at s1 almost nothing is evicted (main re-prefills 21,291 tokens in 10 min), so the copies are mostly
  cost. On bella the pool is short and restores dominate (2d32: resume p50 0.746 → 0.267 s). Bella pays the same
  copies at 28 GB/s: ≈ 33 ms per 254-page chain (ESTIMATED).

## Verdict: the rule, and `--cache-ram auto`
- **The five clauses, as read: all PASS.** ≤ 0.3 s: 0.204 s (a169, 4090). Re-prefill ≤ 5 %: 3.72 % (28ee, bella).
  s1 ITL p99 ≤ 1.05×: 1.025 (c76b, 4090; bella's PROVISIONAL 2d32 read 0.970). s4 ITL p99 ≤ 1.25×: 0.980 (28ee,
  bella). Replay md5 == main: same in every clean run of every job.
- **The rule's outcome is "Pass → default `auto` (≤ 8 GiB), paged GPU pool only". This PR does not flip it.**
  Note §7 and §10.3: the flip needs Haseeb's answer to the plan's Q2 (pinning 8 GiB of bella's RAM) and is not part
  of this PR's gate. `--cache-ram` stays opt-in here.
- **Open for the flip (the lead's calls, not re-reading the rule):**
  - Two clauses were read on a 4090. Note §7 says "bella only (the gate is bella's cell)", and the lead rerouted them.
    Bella's resume cell is ESTIMATED at 0.21–0.32 s in its fast copy mode and 0.37–0.49 s in its slow one: a169's
    88 ms outside the copies, scaled by bella / 4090 (1.73× the decode step, 3.06× the full prefill, f763 vs a169),
    plus 414 pages at 28 or 7 GB/s. So the 4090 does not settle bella's ≤ 0.3 s. Rerunning the cell on bella takes
    ≈ 20 min.
  - The tail spill (above): `auto` would have put it on every large-VRAM server, where it is mostly cost. Fixed in
    e1df27a, gates rerun (below).

## The tail-spill fix (e1df27a) and one more gate round (the lead's ruling, 11-05 ~21:00)
- **The rule.** A spill stores the pages a slot gives up only if the page before them is stored, in the slot's keys
  or by another sequence (`Tier::orphans`, a `peek_len` walk with no LRU change). Otherwise it drops them, as with
  no tier. It never copies a page the slot keeps. A reclaim (`keep` 0) gives up everything, so it still stores the
  whole chain; a41d028's give-nothing-up case is the same rule's limit.
- **Bits.** A dropped page is just not stored, and restores read stored pages only, so replies stay main's bits (the
  serve test below compares; the md5 gates check).
- **What dropping can cost.** With the kept prefix unstored, everything the slot gives up is dropped, not only a
  reply's tail: a different conversation admitted to a slot that shares only a system prompt with the old one loses
  the old one's pages, unless that prompt is stored. Shared pages usually are: a stored page with children is
  evicted only after them. The rerun's s1 / s4 re-prefill tokens measure it; the resume cell's session is evicted by
  reclaims (`keep` 0), which still store everything.
- **Tests.** `tier::tests::spill_orphans` (the rule: kept pages stored, another sequence's parent, the s1 case after
  eviction, a reclaim), and the fuzz's admissions now spill by it, `check` after each.
  `serve::tests::serve_cuda_cache_ram_tail_spill`: (a) the write-behind starved, P's reply ends a page past X, and Q
  (X, then other text) gives up 1 page. It must copy nothing; fc7bb13 copies the whole chain. (b) X stored by the
  idle flush, then starved: Q2 gives up P2's pages under X's stored last page, which alone are stored, and R (P2's
  next turn) restores them. Every reply == `--cache-ram 0`'s.
- **Before / after (bella):** the test on fc7bb13 + the test alone, 1105-205357-b9a0 (expected FAIL at (a)); after:
  in the release and debug suites below (`serve::` holds both tests). PENDING.
- **The gate round (round 2; it replaces 626051c's)** at 9829783 (e1df27a, main 7a2c9cf merged in 26649ae, one
  test fix), against main 7a2c9cf, the same harnesses (`agent_replay` sha256 34b1910b…, the trace, `serve_md5.py`,
  `kv_resume_bench.py` @fe5117d; `resumeexcl.sh` now writes per host, because bella's and the 4090's resume cells
  share one staged dir). The bar for the fix (the lead's): s1 resume p50 ≤ ≈ 1.10× main with re-prefill tokens still
  well below main's, and the resume cell's 13.9× unchanged. All PENDING:

  | cell | job | host |
  |---|---|---|
  | release `serve:: kvstore:: cuda::qwen2::slots:: cuda::qwen2::kvio::` | 1105-210552-7515 | bella |
  | debug (debug-assertions, overflow-checks), the same filters | 1105-210552-110a | bella |
  | md5 vs main, default flags (gen + serve) | 1105-210601-ed07 | bella / pods |
  | replay s4 | 1105-210655-4117 | `--timing bella` |
  | replay s1 | 1105-211308-268a | `--timing 4090` (pod1, secure, warmed) |
  | resume cell | 1105-211420-0618 | `--timing 4090` (pod1, after s1) |
  | resume cell, bella's (the number `auto` needs) | 1105-211421-8e35 | `--timing bella`, no fallback |

## What didn't work
- 68e9: the bench's fixed pressure cap (8 sessions) never evicted anything on the 4090's 5,288-page pool. All 6
  rounds were VOID; fixed in fe5117d.
- 2d32: Xorg load on bella tainted 3 candidate runs. SHORT, 2 clean rounds of 3; rerun as c76b.
- a41d028 fixed only half of the next-turn spill: one given-up page still copied the whole chain (above; e1df27a).
