# Shared-prefix KV pages, C1 + C2: refcounted pages and a prefix index, no behaviour change  (PR #164)

- Date: 2026-11-04 (C1 + C2 written 2026-10-27). Branch: feat/shared-prefix-pages. Base: main af281d9 (#166, #181,
  #180 merged in at 5564c69). Status: open.
- Pick / design: agent workload row C ([docs/design/agent-workload.md](../design/agent-workload.md) §2, §6 row C;
  design note [docs/design/shared-prefix-pages.md](../design/shared-prefix-pages.md)). The foreman's plan split the
  PR: C1 + C2 land now with no behaviour change; C3 (the engine: claim, index, evict) and C4 (in-flight dedupe) are
  follow-up PRs.

## What changed
- **C1, `PageMap` (src/cuda/qwen2/pages.rs).** Each page counts the slot lists that hold it (`refs`), and the prefix
  index can pin it (`pin` / `unpin`; pinned with no holder = "warm"). `map(slot, pages)` appends another slot's live
  pages. `release` retires a page only once no slot holds it and it isn't pinned. `check()` asserts one state per
  page (free, retired or live), refs == occurrences, live ⇔ held or pinned, no page twice in a slot.
- **C1, `PrefixIndex` (src/serve/prefix.rs, new, host-only).** Keyed by `kvstore::chain`; tokens and parent are
  compared on every hit, so a hash collision is a miss, never an alias. Holds mirror the pool's refs; LRU over unheld
  leaves (`pop_lru_leaf -> Evicted { key, parent, tokens, page }`); `walk_prompt` capped below the prompt's last
  position; `unindex_subtree` for in-place continuation.
- **C2, the pool plumbing (src/lm.rs, src/cuda/qwen2/slots.rs).** `BatchLm::{slot_pages, map_pages, pin_page,
  unpin_page}` (+ `same_page_bytes` in test and debug builds), `PoolPages { warm, excl }`. `map_pages` validates (live
  pages, no duplicates, within the cap; nothing changes on Err).
- **The immutability rule (E3): a page shared by two slots or pinned by the index is never written again.** Debug
  builds assert it on every write into a pool page (`PageMap::assert_writable` in `feed_slot`, `step_rows`, `adopt`).
- **Merged with #166 (the RAM tier).** No textual conflict; the semantic one is #166's page copies, a write path
  into pool pages (`pool_page_io` to the device: `page_in`) and the by-id copy-out C3's eviction will use
  (`pool_page_id_io`: `page_out_id`). Both now check the rule in debug builds: `page_in` writes only a page its slot
  may write; `page_out_id` reads only a live page (held, or warm), so C3 must copy a page out before `unpin_page`
  can retire it. New GPU test `page_copies_respect_sharing`.
- **CI fix.** `PageMap::writable` was dead code in a release build under `-D warnings` (its only non-test caller,
  `assert_writable`, is debug-only): `#[cfg_attr(not(debug_assertions), allow(dead_code))]` until C3 calls it.
- **Docs.** The design note moved from `specs/` (a directory only this PR created) to `docs/design/`, with a status
  header and a paragraph on #166; CHANGELOG.

## Why no behaviour changes
Nothing in the engine calls `map_pages`, `pin_page` or `unpin_page` (they are `#[allow(dead_code)]` until C3), so
every live page has refs 1 and no pin. Then `ensure` hands out the same pages as main (it only sets refs = 1), and
`release` retires the same pages in the same order (each drops to refs 0, unpinned, and is pushed to `retire` as
main's `retire.extend(drain(keep..).rev())` did). The new checks are debug-only. What release builds add is
allocator bookkeeping: a refs store per page `ensure` takes, and a decrement and compare per page `release` gives
back. `BatchLm::pages` computes exactly main's fields again. The engine calls it at least 3 times per step: twice
in `publish` (directly and through `slot_states`), once in `grow`, and twice more per prefill chunk in `feedable`.
- **Corrected after review (shady, round 1).** At `0b22e9c` `BatchLm::pages` also counted `warm` and `excl`: an
  O(total pages) scan and a second Vec on every call. This entry and the PR said that ran "on admission and
  planning, not per token", which was wrong: it ran on every engine step. Nothing in release read the counts, so they
  are `PageMap` methods now (tests read them) and C3 adds them where admission needs them. The timing A/B runs on
  the head without them.

## How it was verified
All on bella (RTX 5070 Ti Laptop, sm_120), at `0b22e9c` (code identical to `5564c69`; `0b22e9c` is docs only), main
`af281d9` as the base.
- The touched modules' tests (`cuda::qwen2::pages::`, `cuda::qwen2::slots::`, `serve::`, `lm::`; 193, of which 9
  are `#[ignore]` benches), release and debug assertions + overflow checks.
- md5 gate (`md5.sh`, axon-podjob): greedy `gen -n 64`, main vs this head interleaved in one job, CPU / CUDA f16 /
  CUDA `--kv q8` × qwen2.5-0.5b, Qwen3-0.6B, Llama-3.2-1B, gemma-3-1b × 3 prompts (one ~600 tokens).
- Serve identity (`serveid.sh`, axon-podjob): the C=1 and C=8 reply digests of main, main `--cache-ram 2G`, and this
  head plain / `AXON_PAGE_SHUFFLE=1` / `AXON_PAGED_KV=0` / `--cache-ram 2G`, Qwen2.5 1.5B and 7B Q4_K_M.
- clippy `-D warnings` (lib, tests, bin; debug and release cfgs), rustfmt, CI.

## Results
| gate | job | result |
|---|---|---|
| touched modules' tests, release | 1104-171647-06a8 | PASS 184 / 0 failed / 9 ignored |
| same, debug assertions + overflow checks | 1104-171647-9b0d | PASS 184 / 0 failed / 9 ignored |
| md5 `gen` vs main | 1104-170831-4345 | PASS 33 / 33 identical, 0 nonzero rc |
| serve identity vs main | 1104-170831-522e | PASS 16 / 16 runs print main's C=1 digest (1.5B `5e320d8f66a1cca5`, 7B `393d877192f2e4d7`) |
| serve timing A/B, `--timing bella` | 1104-170832-990d | dropped before it started (round 1 changed the code) |
| CI (`check`) | GitHub Actions run 41921677516 | PASS |

The first release / debug test jobs (1104-170829-af78, 1104-170830-b3e9) were killed by the foreman, not failed: the
scratch submit script claimed no RAM, the scheduler's default (4000 MB) throttled the release run at MemoryHigh (PSI
73 %), and both were resubmitted with 10093 / 18000 MB as the -r2 jobs above. The script now claims 10500 / 18000.

Round 2, at `64f7660` (shady's review, round 1: `BatchLm::pages` back to main's fields, `page_state`, the claims
corrected). The serve identity gate gained a second turn per conversation (`turns.py`: 16 conversations over 8
slots, so a second turn's prefix comes back from the RAM tier when it is on) and prints the tier's restore
counters on the `--cache-ram` runs, failing at 0 restored (round 1's `texts` prompts are under a page: it can't
have restored any).

| gate | job | result |
|---|---|---|
| touched modules' tests (194), release | 1104-175445-a26e | PASS 185 / 0 failed / 9 ignored |
| same, debug assertions + overflow checks | 1104-175446-5782 | PASS 185 / 0 failed / 9 ignored |
| md5 `gen` vs main | 1104-175446-fe6f | PASS 33 / 33 identical, 0 nonzero rc |
| serve identity vs main, two turns, restore counts | 1104-175447-f4e7 | PASS 16 / 16 runs print main's C=1 digest (1.5B `8dd1d2cae1681575`, 7B `9eb689790183d2ff`); every `--cache-ram` run restored (1.5B main 878 / cand 880 tokens, 7B 880 / 876; 14 hits, 33 pages written each) |
| serve timing A/B, round 1 (3 rounds), RTX 4090 pod | 1104-175448-a7de | C=1, C=8 pass; C=16 misses on overlap (tok/s −0.36 %, ITL p50 +0.36 %, ranges disjoint): rerun |
| serve timing A/B, rerun (10 rounds), same pod | 1104-183406-47bd | PASS, every cell (FOREIGN_GPU max 0 %). C=16 tok/s 1534.20 [1527.80–1539.60] vs main 1532.55 [1528.40–1543.30] (+0.11 %), ITL p50 8.27 vs 8.27, ranges overlapping; C=1 and C=8 tok/s and ITL p50 within 0.11 %; C=8 TTFT p50 +4.99 % (ranges overlap) |
| A/A control (main vs main, 10 rounds), same pod | 1104-183510-8351 | main vs main, every cell within the rule. C=16 tok/s 1534.10 [1519.20–1537.00] vs 1534.95 [1532.80–1537.90], ITL p50 8.27 vs 8.25; C=1 TTFT p50 −3.03 % |
| host tests `pages::` + `prefix::` | local | 14 / 0 / 0 |
| clippy (debug and release cfgs), rustfmt | local | clean |
| CI (`check`) at `64f7660` / `5a68e4a` | runs 41929354749 / 41929482616 | PASS / PASS |

Logs: [sharekv-64f7660-gates](https://github.com/HaseebKhalid1507/Axon/pull/164#issuecomment-6644863023),
[sharekv-64f7660-timing](https://github.com/HaseebKhalid1507/Axon/pull/164#issuecomment-6645130290),
[sharekv-64f7660-timing-r2](https://github.com/HaseebKhalid1507/Axon/pull/164#issuecomment-6645842192) (47bd, 8351).

- The digests differ from round 1's because the workload does (two turns). `RESTORE … restored_pages=` in the
  serve gate's log is the script's restored tokens / 64, rounded down, not a counter. Restored tokens vary run to run
  (876–880 for the same 14 hits: write-behind depends on timing), so the gate checks > 0, never equality with main.
- **The timing A/B ran on a 4090, not on bella.** It was submitted `--timing bella`, waited 24 min behind desktop GPU
  load, and the foreman rerouted it to a RunPod secure 4090 (`pod1`: driver 595.91.07, CUDA / NVRTC 13.2, EPYC 7763,
  exclusive, FOREIGN_GPU max 0 %). The A/B is relative and interleaved on one card, so it stands; its absolute
  numbers aren't bella's. Acceptance (shady's, stated before the run): per cell, cand/main of agg tok/s and ITL p50
  in [0.98, 1.02] and the [min–max] ranges overlapping; ITL p99 and TTFT p50 overlapping or within ±5 %; a miss is
  rerun once on the same pod. The rerun (`servetime-r2.sh`) runs 10 rounds, starts each binary once untimed first
  (round 0: NVRTC compiles before any timed round; the pod's own kernel cache can't be pre-warmed for it, as the
  script sets `XDG_CACHE_HOME` to its job directory) and prints every run. The A/A control is the same script with
  main on both sides: the spread between server starts of one binary.
- **Timing verdict: no measurable cost.** The rerun passes every cell. The A/A control puts round 1's C=16 miss
  inside one binary's own spread between server starts: round 1's candidate range at C=16, tok/s 1520.60–1536.90 and
  ITL p50 8.26–8.27, lies within main's A/A range alone (1519.20–1537.00, 8.25–8.28). It was round 1's main that ran
  high (1539.40 median, against 1532.55 and 1534.10 / 1534.95 for main in the rerun and both A/A legs).
- Review round 2 (shady, 8.5/10 SHIP-AFTER-FIXES at `5a68e4a`): its MUSTs were this timing gate (met by 47bd, with
  the A/A 8351) and a body and log behind their results (this entry, and the PR body). No issue covers C1 + C2 on
  their own; the foreman waived the one-issue-per-PR rule for this split.

Not covered by the serve gate: an undersized pool (reclaim, a short `make_room`, victim, preempt). Its pool is
512 pages = 8 × 4096, a full fit, and the CLI has no pool-size knob (`Opts::kv_pages` is test-only); the suites'
`serve_paged_pressure` (40 pages) covers those paths. Listed for C3 (#187).

## What didn't work
- CI on 89f977f failed on the release build's dead-code lint. The local clippy ran at opt-level 0 only, where
  `debug_assertions` is on, so it never saw the release configuration. The scratch clippy script now runs both
  (`-C debug-assertions=off` for lib, tests and bin) and reproduced CI's error without the attribute.
