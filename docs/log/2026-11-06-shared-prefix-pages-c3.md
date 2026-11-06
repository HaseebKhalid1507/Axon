# Shared-prefix KV pages, C3: the engine claims, indexes and evicts shared pages  (PR #207)

- Date: 2026-11-06. Branch: feat/shared-prefix-c3. Base: main ad0c65e (#201's exclusive KV RAM tier and its tail-spill
  drop rule included), merged up to 996a5ac (#206) at `a843282`. Status: open; sharing opt-in after the timing round
  (Verdict); shady's review 8/10 SHIP-AFTER-FIXES, fixes below (r4).
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
- **`AXON_SHARE_KV=1`: opt-in, off by default** (`afb16be`, after the timing round failed two of #187's clauses;
  Verdict below). Unset or anything but `1`: off, every path is main's. **`AXON_KV_PAGES=N`**: an N-page pool (at most what fits), the
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
  serve_share_cancel_frees}`; `afb16be` adds `serve::tests::share_kv_is_opt_in`, and `abb5f97` (shady's SHOULD 2)
  `serve::tests::serve_share_unindex_in_place`: the claim that takes a page its slot alone holds out of the index
  and writes the rest of it in place (`unindex_at`) runs (a test-only count, `Metrics::unindexed` under
  `#[cfg(test)]`, goes 0 → 1 at that claim), the request reports every token it shares cached (in place from
  mid-page), and every reply, the original prompt's again after it too, is the one with sharing off.

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
- At `afb16be` (sharing opt-in) the same md5 job as `md5share-v2.sh`: every arm that ran sharing on by default
  sets `AXON_SHARE_KV=1`, and the default (off) and the undersized pool off are arms of their own.
- **Which gate proves which** (shady's SHOULD 3). The undersized-pool md5 arms (`AXON_KV_PAGES=48`, in 7cb0 and 4cc6)
  reported `preempted 0, unfed 0` in every model and arm: they gate the index's warm eviction (223–272 pages evicted
  an arm) and the reclaim of idle slots, not preemption, whatever the script's header meant them to. Reply identity
  under preemption and unfeed rests on `serve_share_pressure`: a 40-page pool with `kv_grow` 0, every reply == the
  one with sharing off, and a preemption asserted on each of its servers (with and without the tier). It runs in
  every release and debug suite here (r2, r3, r4) and ran 3 × on jade (2e5f: 4–5 preemptions + unfeeds a round).
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
| timing: s4, warm/GB ladder, fan-out, servetime, s1 (4090 pod) | `76b8f0b` | 1106-094454-b75c, 1106-094455-4ac1, 1106-094455-e8ca, 1106-094456-10cf, 1106-094456-7274 | **two clauses FAIL** (table below) |
| touched modules' tests + `share_kv_is_opt_in`, release (r3) | `afb16be` | 1106-120316-feea | **PASS** 216 / 0 / 10 |
| same, debug assertions + overflow checks (r3) | `afb16be` | 1106-120317-e24a | **PASS** 216 / 0 / 10 |
| md5 (`md5share-v2.sh`: gen + serve, sharing off by default and opt-in) | `afb16be` | 1106-120317-4cc6 | **PASS**: gen 0 of 30 cells differ (90 runs, rc 0). Serve, every reply == main's: 7B f16 189/189 and 7B q8 189/189 (9 arms × 21), gemma3-1B f16 84/84 (4 arms), Qwen3-0.6B q8 189/189, CPU 0.5B 42/42 (default and `=1`). The default (off) arm is main's decision for decision too: shared/evicted absent and the same cached-token count as main in every GPU cell (7B 11,118 = 11,118; gemma 11,135; 0.6B 11,118), as is `AXON_SHARE_KV=0`. The `=1` arms share (1,280–10,880 tokens) and evict (48-page pool: 223–272 pages) with the same replies |
| CI (`check`) | `bcb27f4` (`afb16be` + docs) | Actions run 42258164917 | **PASS** |
| touched modules' tests + `serve_share_unindex_in_place` (217), release (r4: main `996a5ac` merged, shady's test) | `abb5f97` | 1106-130147-494d | pending |
| same, debug assertions + overflow checks (r4) | `abb5f97` | 1106-130338-def1 | pending |
| md5, replay md5, timing at the merge head | `abb5f97` | | not rerun: waived by the lead (#206 changes only load-time RAM checks, none of serve, pages or slots; `abb5f97` is `#[cfg(test)]` only) |

`abb5f97` changes no release bits: `git diff a843282 abb5f97 -- src` is `#[cfg(test)]` items only (serve.rs, serve/ops.rs,
serve/engine.rs) and serve/tests.rs. The release binaries of `a843282` (a clean worktree) and `abb5f97`: `.text`,
`.rodata`, `.data`, `.eh_frame`, `.gcc_except_table`, `.init_array` byte-identical (sha256 each); `.data.rel.ro`
differs in 91 4-byte words, each a panic location's line number moved by +4, +6 or +7 (the test-only lines above it).

### Timing (RTX 4090 pod, `76b8f0b` vs main `ad0c65e`)
Pod `pod1` (zmu6vy2elvx0ft): RTX 4090 24 GB, secure RunPod, driver 595.91.07, power limit 450 W. Exclusive (`--timing`:
each job waited for 120 s of 0 % foreign GPU load), both binaries' NVRTC caches warmed first. Every round clean
(0 TAINTED), 0 errors. Up 2 h 15 min, $2.03; terminated by `downer.sh` at 11:55 once the last job ended.

| clause (PR §4, fixed before any data) | job | main | this head | cand/main | verdict |
|---|---|---|---|---|---|
| s4 TTFT-new p50 ≤ 0.6× (3 rounds) | b75c | 1.382 s | 1.386 s | 1.003 | **FAIL** |
| warm/GB ≥ 1.4× at the ladder's rung | 4ac1 | base warm 7 / 7, 14 / 14, 22 / 22, 29 / 29 turn-sessions at 8 / 16 / 24 / 32 | (not run: no rung) | | **UNTESTABLE** (`WARMLADDER UNTESTABLE`, exit 3) |
| s1 ITL p99 ≤ 1.02× (3 rounds) | 7274 | 10.54 ms | 10.54 ms | 1.000 | PASS |
| fan-out idle: siblings' (TTFT − leader's) p50 ≤ 0.25× | e8ca | 1.886 s | 0.493 s | 0.261 | **FAIL** |
| fan-out idle: leader TTFT p50 ≤ 1.05× | e8ca | 0.549 s | 0.551 s | 1.004 | PASS |
| fan-out busy (4 decoding): siblings' (TTFT − leader's) p50 ≤ 0.25× | e8ca | 1.339 s | 0.656 s | 0.490 | **FAIL** |
| fan-out busy: leader TTFT p50 ≤ 1.05× | e8ca | 0.742 s | 0.720 s | 0.969 | PASS |
| C=1 decode agg tok/s within 2 % (5 rounds + a warm one) | 10cf | 181.00 | 181.00 | 1.000 | PASS |
| md5 == main (every timing run's replies) | all | | | | PASS: s4 and s1 `2aee881bcf75` on all 12 runs, fan-out SAME over 5 rounds × 2 variants |

INFO (not clauses):
- s4: re-prefilled tokens 1,665 → 577 (−65 %); goodput 125.96 → 126.87 tok/s; ITL p99 78.68 → 78.49 ms; TTFT-resume
  p50 0.064 → 0.063 s; warm sessions 18 / 18 on both. s1: re-prefill 3,213 → 2,125 tokens; TTFT-new 0.999×.
- Fan-out: siblings' TTFT p50 2.434 → 1.045 s idle (0.43×), 2.082 → 1.375 s busy (0.66×). Each sibling after the
  leader prefilled 54 of the 6,006 shared tokens (busy: the first sibling prefilled 4,854–4,918 of them); main
  prefilled 6,006, 6,006, ~3.1–3.4k, ~0.9–1.5k, then 54–246.
- servetime C=8 / C=16: agg tok/s 1.000 / 1.000, ITL p99 1.001 / 1.000.
- The harness's own `CLAUSE s4 warm/GB … FAIL` line (0.696 both) is not a clause: §4 moved warm/GB off the
  canonical s4 cell ("The canonical s4 cell caps this clause at 18 / 14 = 1.29× whatever the feature does (§8), so it
  moves to a cell with headroom"), to the ladder's rung. The ladder found none, so warm/GB is untestable on the 4090
  (§4 rule 5): neither a pass nor a fail. Base kept every turn-session warm at every rung: the 4090's pool is 5,288
  pages (338k positions) against bella's 1,722.

**Which TTFT-new clause was registered.** #187's gate line reads "TTFT-new p50 for subagent starts ≤ 0.6×". The PR's
§4 registered it on s4: "**s4 TTFT-new p50 ≤ 0.6×**, on the canonical s4 cell (the board trace's own sessions)". That
text is in every revision of the PR body from its creation (11-06 12:39 UTC; the timing jobs were submitted at
14:44 UTC) and in this log since `9610f6d` (07:38 EST). So it isn't a harness default: it counts. The fan-out
cell (spec §6 row C, re-registered 10-27 in #160) has its own clause, the siblings' increment ≤ 0.25×, also in §4.
The s4 clause was unreachable on its cell, and I should have checked that before registering it. Rebuilding the cell
from the replay's own text builder at the board's flags shows why. Its 13 new sessions are forks. Each prompt's
longest prefix shared with any earlier prompt is 1,134 tokens for 9 of them, 954, 1,122 and 8,192 for three, and 0
for two. Their prompts are ~16.2k tokens (p50; compacted to half the 32k budget), so the p50 new prompt has 15,063
tokens nothing can share. Sharing cuts its prefill by ~7 % at most (ESTIMATED ceiling ≈ 0.93×). The s4 cell measures
the claim path's cost on new sessions (1.003×: none), not a saving. Per §4 ("I don't argue a miss away") it stands as
a FAIL. The outcome doesn't hang on it: the fan-out clause fails on its own, in both variants.

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
**Sharing ships off by default; `AXON_SHARE_KV=1` turns it on** (`afb16be`). The rule (#187, PR §4): "If any clause
fails, sharing ships off by default (`AXON_SHARE_KV=1` opt-in)". Two failed: the fan-out siblings' increment (idle
0.261×, busy 0.490×, against ≤ 0.25×) and s4's TTFT-new p50 (1.003×, against ≤ 0.6×). warm/GB was untestable on the
4090 (the ladder found no rung). s1 ITL p99, C=1 decode, the leader guards and every md5 passed. Off, the server
runs main's admission, eviction and spills, decision for decision (the old kill switch's path, md5-gated in 7cb0's
`AXON_SHARE_KV=0` arm and again at `afb16be`). On, it is as measured above: same bits, same decode, 65 % fewer
re-prefilled tokens at s4, the fan-out's siblings at 0.43× (idle) and 0.66× (busy) main's TTFT.

Follow-ups (not in this PR), filed after shady's review:
- #210: warm/GB, #187's headline clause, untestable on the 4090: the ladder again where the pool fills
  (`AXON_KV_PAGES=1722` on the 4090, or bella), its rung rule and ceiling registered before it runs.
- #211: a subagent-start TTFT cell whose new sessions share a real system prompt and tool catalog, its ceiling worked
  out before it runs. The fan-out clause had s4's flaw too (shady): with sharing on, each sibling after the leader
  prefilled 54 of the 6,006 shared tokens, so the 0.261× is the siblings' own 1,110-token deltas and their
  scheduling, which sharing can't cut. That doesn't change the verdict above; it shapes the next registration.
- #212: a preempted or unfed prefill claims the shared prefix again when it is fed again.
- #213: one duplicate page (`Ins::Have`) stops its slot from indexing any of its later pages (shady's SHOULD 1;
  compounds with #212 under pressure).
- #188: C4, in-flight prefix dedupe.
