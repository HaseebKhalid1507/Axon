# Shared-prefix KV pages, C3: the engine claims, indexes and evicts shared pages  (PR #207)

- Date: 2026-11-06. Branch: feat/shared-prefix-c3. Base: main ad0c65e (#201's exclusive KV RAM tier and its tail-spill
  drop rule included). Status: open.
- Pick / design: issue #187 (agent workload row C, stage C3), design note
  [docs/design/shared-prefix-pages.md](../design/shared-prefix-pages.md) §3.3–3.7 ("What C3 built" in its header
  lists where the code differs), spec [docs/design/agent-workload.md](../design/agent-workload.md) §6 row C. Builds on
  #164 (C1 + C2: the refcounted pinnable `PageMap`, the host `PrefixIndex`, the `BatchLm` share calls).

## What changed
- **Claim** (`Batch::claim_prefix` over `plan_claim`, src/serve/engine.rs). `admit` and `lend_solo` both go through
  it. The index's `walk_prompt` gives the pages of the prompt's whole blocks below its last token. If the slot
  already holds as much, it continues in place (main's path). Otherwise it keeps its leading pages that match the
  walk's, releases the rest and maps the index's pages (`BatchLm::map_pages`). Phase 1, no copy-on-write:
  - A continuation mid-way into a page that other slots hold too is fed again from that page's start.
  - A continuation into an indexed page this slot alone holds: the page leaves the index, with every block after it
    (`unindex_at` → `unindex_subtree`), and is written in place.
- **Index** (`index_pages`). A slot's whole pages are offered to the index once all their rows are written: fed,
  decoded, restored from the RAM tier, or written by the request running alone. New blocks are pinned and held.
  A duplicate (the index has the block at another page) stays the slot's own, and so do the slot's later pages. So
  every slot holding an indexed page holds its ancestors' too, and the warm set evicts leaf by leaf.
- **Evict** (`evict_warm`, `let_go`). `make_room` and `reclaim` take the index's least recently used warm leaf first
  (`pop_lru_leaf`), then an idle slot's pages (whose indexed ones turn warm and go next). Order per leaf, with #201's
  tier: `PoolTier::spill_page` (the page still pinned, so live; `page_out_id` refuses a dead page) →
  `Tier::drop_resident(key)` → `unpin_page`. `Tier::keep_resident` / `drop_resident` are new (#201's design §2.6
  names the seam): a stored key whose bytes the index keeps on the GPU stays class 0 after its slot lets go. With
  sharing on, a slot giving up rows stores none of its indexed pages (they stay on the GPU until evicted); a page
  the index lacks spills as before.
- **Admission's room** (sharing on): every page no running request holds and the claim doesn't (free, warm, idle
  slots'), counted once (`BatchLm::busy_pages`, O(pages held), one bitmap a call). With sharing off, main's formula.
- **`AXON_SHARE_KV=0`**: off; every path is main's. **`AXON_KV_PAGES=N`**: an N-page pool (at most what fits), the
  undersized-pool identity gate's knob (#187: the CLI had none).
- **Counts.** `/slots` `pages_warm` / `pages_shared`. `/metrics` `axon_kv_pages_warm`, `axon_kv_pages_shared`,
  `axon_prompt_shared_tokens_total`, `axon_pages_evicted_total`. They are the index's O(1) counters, not `PoolPages`
  fields (#164 review: `BatchLm::pages` runs ≥ 3× a step).
- **Debug checks** (#164 review's carried items):
  - E3 on the request running alone: `lend_solo` asserts the positions its view writes are in pages only it holds,
    unpinned.
  - After every step, the leak check (`check_share`): `PageMap::check` (one state a page, `refs` == occurrences in
    slot lists), the index's holds of each page == its `refs`, every indexed page pinned, each slot's offered keys its
    rows' chain.
- **Tests** (10 new): `serve::engine::claim_tests::claim_prefix_cases`, `serve::prefix::tests::{peek_is_walk_without_the_touch,
  warm_and_shared_counts}`, `serve::tier::tests::the_index_keeps_pages_class0`, `serve::tests::{kv_pages_env_parses,
  serve_share_matches_off, serve_share_fanout_idle, serve_share_pressure, serve_share_solo_handover,
  serve_share_cancel_frees}`.

## How it was verified
Bella (RTX 5070 Ti Laptop, sm_120) for correctness. Timing on an RTX 4090 secure pod (the lead's allowance, ≤ $2 a
round), main `ad0c65e` as the base in every cell (the postmerge build of main).
- The touched modules' tests (`cuda::qwen2::pages::`, `cuda::qwen2::slots::`, `serve::`, `lm::`, `kvio::`,
  `kvsave::`: 225), release, and debug assertions + overflow checks (the leak check and the E3 checks run there).
  clippy `-D warnings` (lib, tests, bin; debug and release cfgs), rustfmt, CI.
- md5 (`md5share.sh`, axon-podjob): (1) greedy `gen -n 64`, main vs this head vs this head with
  `AXON_PAGE_SHUFFLE=1`, CUDA f16 / CUDA `--kv q8` / CPU × qwen2.5-0.5b, Qwen3-0.6B, gemma-3-1b, Qwen2.5-7B × 3
  prompts; (2) serve (`serve_md5_share.py`): 7 sessions × 3 turns over 4 slots, every prompt opening with the same
  600-word system text, greedy and seeded-sampled. Every reply == main's for: sharing on (default),
  `AXON_PAGE_SHUFFLE=1`, `AXON_SHARE_KV=0`, `--cache-ram 1G`, and an undersized pool `AXON_KV_PAGES=48` (alone, with
  `--cache-ram 1G`, with the shuffle).
- Replay md5 under the shuffle (`replayshare.sh` via `replaymd5.sh`, bella): `agent_replay.py run` (main's copy),
  board trace, s4, one round; main vs this head with `AXON_PAGE_SHUFFLE=1`, both `--cache-ram 6G`.
- Timing (design §6 row C; #187's clauses, read as fixed before any data, in the PR body §4):
  - `replayshare.sh`, s4 and s1, 3 rounds each, both binaries `--cache-ram 6G` (main = main + WP2 at the same
    size): TTFT-new p50 ≤ 0.6× (s4), s1 ITL p99 ≤ 1.02×.
  - warm/GB cand/main ≥ 1.4, on a cell picked by the lead's 11-06 ruling, recorded before any timing data
    (PR comment 6685131445). The canonical s4 cell capped it at 18 / 14 = 1.29×: main + WP2 already kept 12 / 14 /
    14 of its 18 turn-sessions warm in #201's bella runs. `warmladder.sh`:
    - Base only (main + WP2, 6G), up the #186 ladder `--sessions` 8, 16, 24, 32, speed 4, 3 rounds a rung.
    - The rung is the smallest where base's median warm ÷ turn-sessions ≤ 0.60.
    - Then the A/B at that rung, 3 rounds interleaved.
    - No rung ≤ 0.60 up to 32: the clause is untestable on the 4090, and the default rests on the other clauses.
  - `fanshare.sh`: the fan-out cell, idle and busy: siblings' incremental TTFT p50 ≤ 0.25× main, leader ≤ 1.05×.
  - `servetime-r2.sh` (#164's): C=1 decode tok/s within 2 %.

## Results
| gate | head | job | result |
|---|---|---|---|
| touched modules' tests, release | `71b1a60` | 1106-073530-3f6b | **FAIL** 213 / 2 / 10: two test bugs (below) |
| same, debug assertions + overflow checks | `71b1a60` | 1106-073715-f411 | not run: dropped from the queue (its binary had both tests) |
| CI (`check`) | `71b1a60`..`d01c249` | runs 42219303641, 42219322630, 42220176786 | **FAIL**: `peek_is_walk_without_the_touch` (host-only) |
| touched modules' tests, release (r2) | `76b8f0b` | 1106-090453-76cf | **PASS** 215 / 0 / 10 |
| same, debug assertions + overflow checks (r2) | `76b8f0b` | 1106-090603-c6bb | **PASS** 215 / 0 / 10 |
| `serve_share_pressure` ×3, `--nocapture`, `AXON_SERVE_DEBUG=1` (diagnostic, jade GTX 1050) | `76b8f0b` | 1106-090454-2e5f | **PASS** 3 / 3 runs. Every chunked admission of round 2 claimed the prefix (256, 266 or 284 of 285 tokens cached); after the round 3 requests a server report 0 cached, against 4–5 preemptions + unfeeds in that round (every server of every run): the zeros are victims reset after admission, as diagnosed |
| md5: gen + serve | `71b1a60` (= `76b8f0b`'s code) | 1106-073531-7cb0 | **PASS**: gen 0 of 30 cells differ (90 runs, rc 0); serve every reply == main: 7B f16 147/147, 7B q8 147/147, gemma3-1B f16 84/84, Qwen3-0.6B q8 147/147, CPU 0.5B 42/42, in every variant |
| replay md5 under `AXON_PAGE_SHUFFLE` | `71b1a60` (= `76b8f0b`'s code) | 1106-073531-d9e7 | **PASS**: MD5 SAME, 174 requests each, 0 errors, the round clean |
| timing (4090 pod) | | not yet submitted | after the correctness gates |

### r1's two failures (3f6b): both test bugs, fixed in `76b8f0b` (tests only)
- `serve::prefix::tests::peek_is_walk_without_the_touch` (prefix.rs:521, left `Some(2)`, right `Some(1)`;
  deterministic, also CI's failure). The test checked `peek == walk` with a `walk` on the same index whose LRU order
  it then asserted, and `walk` touches (as documented): A became younger than B and B's leaf went first. The index
  does what its docs say; the test now peeks one copy of the index and walks another (and checks both orders).
  Host-only: it should have run locally before the gate (it does now, with `pages::` and `claim_tests`).
- `serve::tests::serve_share_pressure` (tests.rs:7397, `cache-ram 0: request 1 again: 0 cached`). The test asserted
  that every request of its second round reports the prefix cached. The round runs under the pressure the test
  builds (40 pages, kv_grow 0), so requests are preempted or unfed, and `preempt` / `unfeed` set `seq.cached = 0`:
  the request is fed from 0 again (as on main; claiming its pages again on refeed is not in C3, design header). The
  only other writer of `cached` is admission. So a request short of the prefix was a victim. The round now
  allows at most its `preempted` + `prefills_unfed` (from `/slots`, before and after) short of the prefix, and at
  least one at it: the oldest request never gives way (`victim`). Its replies were == sharing off (that assert, the
  one before, passed). The diagnostic job prints each run's cached counts, victims and admissions.
  It bore this out (2e5f, jade, 3 runs × 2 servers): `AXON_SERVE_DEBUG`'s admission lines show every chunked
  admission of round 2 with the prefix claimed (256, 266 or 284 of 285 tokens cached); the test's line after the
  round shows 3 requests at 0 cached and 4–5 preemptions + unfeeds in every one, e.g. run 1 without the tier:
  admissions #7–#12 cached 266, 266, 284, 256, 256, 256; after, `[0, 0, 266, 0, 284, 266]`, 4 preempted or unfed.
- **md5 and replay md5 stay at `71b1a60`.** `76b8f0b` changes `#[cfg(test)]` code only (`git diff 71b1a60 76b8f0b
  -- src`: prefix.rs's `mod tests` and tests.rs). The two release binaries' `.text`, `.rodata`, `.data`,
  `.data.rel.ro`, `.eh_frame`, `.gcc_except_table` and `.init_array` are byte-identical (sha256 of each section);
  only `.strtab`, the symbol-name table, differs (134 bytes).

## Verdict
Pending the gates. Rule (#187): if a timing clause fails, sharing ships off by default (`AXON_SHARE_KV` opt-in).
