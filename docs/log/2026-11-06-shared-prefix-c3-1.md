# Shared-prefix KV pages, C3.1: duplicates swapped, refed prefills claim again, C3's nits  (PR #216)

- Date: 2026-11-06. Branch: feat/shared-prefix-c3.1. Base: main 2aec1e5 (#207, C3, sharing opt-in), merged up to
  1065204 (#215) in `f1271c7`. Code `c167272`, `2ce5d12` (tests only). Status: open; sharing stays opt-in
  (`AXON_SHARE_KV=1`).
- Pick / design: issues #212, #213, #214 (shady's #207 review follow-ups), design note
  [docs/design/shared-prefix-pages.md](../design/shared-prefix-pages.md). Timing registered for #210 and #211 on the
  PR (comment 6693952063) and run outside its merge path (lead's ruling, 11-06): results on #210 / #211.

## What changed
- **#213: a duplicate page is swapped for the index's.** Two prompts prefilling one prefix side by side each write
  their own copy of a block; the one behind finds it indexed (`Ins::Have`). C3 kept the copy and stopped indexing
  that slot (a page is indexed only under a parent page its slot holds), so its later pages never entered the
  index. `Batch::index_pages` now swaps the slot's copy for the index's page (`BatchLm::replace_pages` →
  `PageMap::replace`; refcounts and holds kept in step, the copy freed) and indexes the rest under it. Debug builds
  compare the two pages' bytes before the swap (`same_page_bytes`); `check_share` asserts no duplicate is kept.
- **#212: a preempted or unfed prefill claims the shared prefix again.** C3 refed such a request from 0.
  `Prefill::refeed` marks it; at its next feed `Batch::refeed_claim` claims the index's pages of its tokens so far
  through `claim_prefix`, as admission does, and feeds only the rest. A claim maps live pages and takes none from
  the free list, so it needs no room check.
- **#214, the five nits:** `share_env`'s doc moved onto its item; `release_rows` returns an `Err` naming the page
  instead of asserting; `busy_pages` marks with generation stamps (no per-call vector of every page; on the stamp's
  wrap every mark is cleared and 0 is skipped); admission walks the index with the prompt's block keys computed once
  (`Job::block_keys`, `PrefixIndex::peek_keyed`); `reclaim` passes over an idle slot whose pages are all shared
  while another would give a page up (`reclaim_victim`).
- With sharing off every changed path takes main's decisions.

## Round 1 → 2: two test bugs
Release 1106-142607-6a44 (`c167272`) failed 219 / 2. Both were the tests':
- `busy_pages_stamp_wraps` asserted a final stamp `< 8`; from `u32::MAX - 1` twelve calls end at 11 (MAX, the wrap's
  1, then 2..=11). It also couldn't catch a wrap without the clear: its first call overwrote every held page's
  stale mark. The rewrite keeps stale marks 0 and 1 (the wrapping call's two possible stamps) on the held pages
  through a no-busy first call and asserts the marks after the wrap and the exact final stamp (12). A scratch copy
  of `busy_pages` with three wrong wraps (no clear; no guard; clear but stamp 0): the new test fails all three, the
  old one passed two.
- `serve_share_dup_swapped`'s fixture was under its own preconditions: prefix 361 tokens (needs ≥ 448), and the
  probes' shared whole pages 576 / 512 (needs ≥ prefix + 192 = 553). Prefix ×32 and suffix lines ×18: 481 tokens,
  768 / 704 (≥ 673), counted with Qwen2.5's `tokenizer.json` (which reproduces the old 361).
The failing-first run was redone on `bc9d181` with the new fixture, so it fails at the swap / refeed assertions.

## Gates
Round 2, head `2ce5d12` (product code `c167272`), bella (RTX 5070 Ti Laptop, sm_120):

| gate | result | job |
|---|---|---|
| touched modules' tests (`cuda::qwen2::pages:: cuda::qwen2::slots:: serve:: lm:: kvio:: kvsave::`), release | PASS 221 / 0 / 10 ignored | 1106-160457-782d |
| same, debug assertions + overflow checks (`check_share` leak check and E3 after every step) | PASS 221 / 0 / 10 ignored | 1106-160700-6e64 |
| failing first: `bc9d181` (tests only) + `2ce5d12`'s fixture | FAILS as intended, 0 / 2, at the targeted assertions | 1106-160902-78aa |
| md5 v3 vs main `2aec1e5` (axon-podjob) | PASS: gen 0 of 30 cells differ; serve 973 / 973 replies equal | 1106-160457-e00e |

- The failing-first run fails where the fixes act. `serve_share_dup_swapped`: the probes' cached tokens are
  `[448, 704]` against `[768, 704]` (the first probe gets only the prefix's 7 pages: its suffix pages were never
  indexed). `serve_share_pressure`: `cache-ram 0: again: cached [0, 256, 284, 0, 0, 266], 3 preempted or unfed`:
  the three refed requests prefilled from 0. The "0 pages swapped" / "0 refed claims" in those messages prove
  nothing: the counters aren't wired at `bc9d181`. The cached tokens carry the proof (shady).
- md5's pressure section preempted in every undersized arm (2,000-token replies, 4 slots), so it gated #212's path:
  preempted / unfed per arm, qwen2.5-0.5b f16 at 128 pages: main 25 / 14, off 26 / 14, on 9 / 50, shuffle 10 / 51,
  tier 17 / 60; at 192: 1 / 10, 1 / 9, 1 / 22, 1 / 22, 1 / 22. Qwen3-0.6B q8 at 128: 3 / 15, 5 / 14, 3 / 45, 3 / 45,
  16 / 60; at 192: 1 / 9, 14 / 9, 1 / 21, 31 / 21, 1 / 22. Main at its full pool: 0 / 0. Every reply equal to main's.
- Main and cand-off differ at equal bits (Qwen3 at 192 pages: 1 vs 14 preempted). By the code, sharing off takes
  main's decisions (`refeed_claim` returns at once without the index; `reclaim_victim` with `gives` true for every
  slot is main's least recently used idle slot). INFO probe 1106-165421-dd4c (`pressvar.sh`, bella, rc 0) says it is
  the harness's arrival order (4 client threads). Same workload; per round main undersized then cand-off, three
  rounds interleaved. Preempted / unfed / cached prompt tokens:

  | cell | main A | off A | main B | off B | main C | off C | replies = main full pool |
  |---|---|---|---|---|---|---|---|
  | Qwen3-0.6B q8, 192 pages | 1 / 9 / 1308 | 1 / 9 / 1308 | 1 / 9 / 1308 | 1 / 9 / 981 | 14 / 9 / 1308 | 14 / 9 / 981 | 84 / 84 |
  | qwen2.5-0.5b f16, 128 pages | 25 / 14 / 0 | 25 / 14 / 0 | 25 / 14 / 0 | 20 / 14 / 0 | 25 / 14 / 0 | 26 / 14 / 0 | 84 / 84 |

  Main alone goes from 1 to 14 preempted at 192 pages, and off matches it in each round (1, 1; 1, 1; 14, 14): the
  md5 run's 1 vs 14 is that spread. At 128 pages main is 25 three times and off 25 / 20 / 26; main didn't move in
  that cell, so it doesn't show main's spread by itself. Off's 981 cached tokens at 192 pages is main's own count
  at that cell in the md5 run (e00e). Unfed counts equal in every run of a cell.

Merge regate, head `f1271c7` (origin/main 1065204 merged in: #215's RAM pre-load checks touch `src/lm.rs`, README and
CHANGELOG, as this PR does; the conflict was CHANGELOG only, both entries kept). bella, `--on bella`:

| gate | result | job |
|---|---|---|
| this PR's modules and #215's (`… kvsave:: model:: hybrid:: gguf:: qwen2::stream safetensors:: cuda::upload`, 402 tests), release | PASS 389 / 0 / 13 ignored | 1106-182016-f597 |
| same, debug assertions + overflow checks | PASS 389 / 0 / 13 ignored | 1106-182016-04d7 |
| md5 v3 vs main `1065204` (axon-podjob) | PASS: gen 0 of 30 cells differ; serve 973 / 973 replies equal | 1106-182017-58d2 |

- The pressure section preempted in every undersized arm again. Preempted / unfed (main, off, on, shuffle, tier):
  qwen2.5-0.5b f16 at 128 pages 27 / 14, 25 / 14, 3 / 45, 33 / 45, 32 / 60; at 192: 1 / 10, 1 / 9, 1 / 22, 1 / 22,
  29 / 21. Qwen3-0.6B q8 at 128: 5 / 14, 21 / 14, 52 / 59, 3 / 45, 37 / 44; at 192: 1 / 9, 1 / 9, 31 / 21, 1 / 21,
  1 / 22. Main at its full pool: 0 / 0. The counts moved from round 2's, by arrival order (above); the bits didn't.
- The CPU serve section's cached tokens move by a token or two between runs, main's too (13082 in e00e, 13083 here;
  the branch 13083, 13081): arrival order again, every reply equal.

Timing: registered on the PR (comment 6693952063) before any timing job, run on a secure RTX 4090 outside the PR's
merge path; results on #211 (subagent start) and #210 (W-sys warm/GB). A pass makes the default flip a follow-up PR.
The W-sys script's own smoke on bella (job 1106-161219-9c15: 0.5B model, 8 sessions, a 60 s window) checked its
plumbing with the real binaries: pool size, `/metrics`, md5, the verdict lines (W0 untestable there by design: 8
small sessions fit whole).
Every registered clause passed. #211 (job 1106-173522-297a, verdict #211 comment 6698163790): the 8 sessions' TTFT
p50, cand/main, 0.591 idle (≤ 0.80) and 0.798 busy (≤ 0.86); the leader's 1.000 / 0.899 (≤ 1.05); md5 equal every
round, 0 errors. On main's normal rounds alone (some rounds main's RAM tier gave no sibling the prefix) T1 is 0.591 /
0.802. #210 W-sys (job 1106-170837-edc9, #210 comment 6697988287): warm/GB 3.01× (≥ 1.4), W0 main warm 1 / 64, W2 md5
equal; but both servers were queue-bound (median turn TTFT 33 s main, 16 s cand), so the ratio compares 1 warm
session with 3–4. Pod time $1.05 (secure RTX 4090, 17:02–18:11).

## Review
- shady 8/10 SHIP-AFTER-FIXES at `bfa311e` (PR comment 6697783054): no code bug. Checked hunk by hunk against
  `2aec1e5` that sharing off takes main's decisions (the proof dd4c is consistent with, not a proof of).
- MUST, done: the timing handoff on the issues, since after the merge only they are read. #210 comment 6697951310,
  #211 comment 6697951572: the registration, the lead's 19:49 UTC W-sys approval, the lead's ruling (the default
  flip is a separate PR, clauses unchanged; in the PR body from 21:13 UTC, before the round's first timing job at
  22:08 UTC), the candidate `c167272` with `AXON_SHARE_KV=1`, the job ids.
- SHOULDs:
  - `index_pages` holds index pages before `replace_pages` applies the swap; an `Err` between leaves holds the
    slot's table doesn't map (fatal today, so nothing lingers). #218, with the nit that a refeed-only step counts
    as `moved` (skips `unfeed_youngest` for one step).
  - `axon_prompt_shared_tokens_total` now counts refeed claims, generated tokens included. README and CHANGELOG
    fixed here; the field's doc comment and the HELP text are code: #219 (reword, or count refeeds apart).
  - The failing-first messages' "0 pages swapped" / "0 refed claims" counters aren't wired at `bc9d181`: no longer
    cited (above).
  - Debug runs with sharing on cover only the 8 `serve_share_*` tests and the unit fuzzers: the flip PR's gate list
    gets the debug suite with `AXON_SHARE_KV=1` (in the #210 / #211 comments).

## Lessons
- A count a harness's client threads can reorder (preemptions, cached tokens) is not a decision gate: compare it
  against the base's own spread over a few runs before calling a difference a change. The bits are the gate.
- A bound in a test is a derivation: write it as the exact value with the steps in the message, not an inequality
  picked by eye (the `< 8` was off by the wrap's skipped 0 and the first call).
- A test whose setup is overwritten by its own first step proves nothing about that setup: check it against a
  mutant of the code it guards before trusting it.
- Size a fixture against its own asserts before the gate (a tokenizer count costs seconds; the gate costs an hour of
  queue).
