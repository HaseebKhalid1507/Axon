# An exclusive KV RAM tier (WP2, with #170): design note, checkpoint 0

- Status: **checkpoint 0, design only, no product code.** Branch `feat/kvtier-exclusive`, off b21a80c (#181's head;
  main is merged in as its own commit once #181 lands). Author: joestar. Foreman: axon. Reviewers: Fable, shady.
- Plan: `specs/agent-workload-plan.md` (Fable, 2026-11-04), §3 row 2 ("Exclusive RAM tier (B2) + #170") and the
  RAM-tier paragraph of §1. Builds on #166 (`--cache-ram`, merged, opt-in, default 0), #172/#181 (the tier's off path,
  tested), #170 (the idle flush has no budget). Design context: [agent-workload.md §3](agent-workload.md).
- Labels: MEASURED (a job id or file is given), ESTIMATED (arithmetic from measured inputs, shown), DECISION (wants
  a reviewer's yes before code).
- Line numbers are at e325181 (this branch's first commit). The plan's numbers (`tier.rs:477–533`, ...) predate #181.

## 0. Summary

#166's tier is **inclusive**. The write-behind copies every page the GPU holds into RAM, and each slot holds its
chain's leaf (`Tier::set_keys`, tier.rs:284), so RAM fills with copies of what the GPU already has. On bella the
6 GiB tier is 1,755 pages and the GPU pool is 1,570–1,744, so the tier held about the pool's own content. The idle
session's tail was the LRU and went first: **1031-063509-f763: TTFT-resume 8.3 → 7.0 s, 6,592 of 26,496 tokens
restored** (MEASURED). Useful capacity was ≈ tier − pool where it should be tier + pool.

The change, all of it in the store's eviction order and the tier's bookkeeping (no byte changes):

1. **Two LRU classes in `PageStore`** (§2.1). A page that some slot still holds on the GPU is *class 0*, a
   duplicate: it is evicted first, newest first, which is tail-first within a slot. A page only RAM has is *class 1*:
   plain LRU, as today, evicted only when no class-0 page is evictable. The slot's leaf hold becomes this residency
   mark, a soft hold.
2. **Write-behind into free room only**, in reclaim order (§2.3). It never evicts. Spills (a slot giving up rows) may
   evict, class 0 first.
3. **`write` re-validates the slot's keys** against the store before extending, because a class-0 page can now be
   evicted under a slot (§2.2).
4. **`restore` down-classes what it moved in** (§2.4). The copies stay as class 0, evicted first, and a later spill of
   the same slot copies nothing.
5. **#170: the idle flush is budgeted** (64 MiB per idle loop iteration) and checks the queue between pieces (§2.5).
   `end_solo` no longer flushes.

Bits: unchanged by construction (§3). The tier's off path from #181 is unchanged and its tests must still pass (§4).

**The gate (verbatim, §6)** has a resume clause of ≤ 0.3 s. My bella arithmetic (§5) puts the exclusive tier's
resume at **≈ 0.30–0.36 s, at the copy floor**. The reason is bella's MEASURED 2D page copy rate, 7.1 GB/s
(0.53 ms per page, 219 ms for the 414-page session), not tier policy. §5.2 proposes measuring that first and names
the remedy (a device gather, then one contiguous copy at the MEASURED 28.6 GB/s). It is a scope question for the
foreman (§10, Q1).

## 1. The problem, measured

**Mechanics (main at e325181).** `PoolTier::write` (tier.rs:636) stores slot `s`'s unstored whole pages and
`push`es each key. `set_keys` (tier.rs:284) holds the slot's last stored key, and `PageStore::evictable`
(kvstore.rs:160) is `children == 0 && holds == 0`. So every slot's chain is pinned at its leaf (and, since parents
outlive children, all the way to the root) for as long as the slot holds those rows. `Batch::write_behind`
(engine.rs:748) runs the copies at 16 MiB per step while requests run (`STEP_BYTES`), and **without a budget** as
soon as the batch is idle (`step`, engine.rs:1294–1304: #170). A slot giving up rows calls `spill(s, keep)`
(engine.rs:722) from `reclaim`, `admit`, `lend_solo`, `unfeed`, `preempt` and `end_solo`. That writes whatever is
still missing and truncates the slot's keys, which releases the hold.

What the GPU holds is therefore always in RAM too, and pinned. RAM's own content (pages no slot holds) is only what
fits in `tier − Σ held`, and it is the LRU class. In the bella cell, session A (414 pages) went idle and the
pressure sessions (2,066 pages) wrote behind as they filled. Those newer pages pushed A's tail out of RAM before A
was reclaimed from the GPU: 103 of A's 414 pages survived (6,592 tokens).

**The working set (ESTIMATED, ideal).** This is a page-level LRU over the agent-32k replay's own request stream:
`agent_replay.plan()` as the board runs it, window 600 s, scale 0.15, clip 32768, 174 requests, 2,921,348 prompt
tokens and 5,164 distinct 64-token pages. Concurrency is ignored, so it is optimistic. Script: `ws.py`, attached to
the PR; it reads the never-committed trace and prints numbers only.

| capacity (pages) | what it models | ideal re-prefill |
|---|---|---|
| 1,537 | bella's f16 pool alone (main) | 9.97 % |
| 1,755 | the inclusive 6 GiB tier (≈ max(pool, tier)) | 6.82 % |
| 2,707 | pool + an exclusive 4 GiB tier | 0.75 % |
| **3,292** | **pool + an exclusive 6 GiB tier** | **0.15 %** |
| 3,877 | pool + an exclusive 8 GiB tier | 0.15 % |
| 7,388 | pool + 20 GiB (the 4090 run's tier) | 0 % |

The measured main is far above its ideal: 47.5 % at s4 and 10.3 % at s1 on the f16 board
(`~/.cache/axon-agent/board-f16-table-clean.md`, MEASURED), against 9.97 % ideal. Slot granularity, preemption and
concurrent sessions cost more than LRU arithmetic shows, so the table ranks the options; it does not predict the
gate. What it does show is that an inclusive 6 GiB tier cannot reach ≤ 5 % even ideally (6.82 %), while an
exclusive one has ≈ 30× headroom ideally.

## 2. Design

### 2.1 `PageStore`: two classes, one tree

The store is a tree. Pages chain from `ROOT`, only unheld leaves are evictable, and parents outlive children
(kvstore.rs:231–264). That stays. Each `Entry` gains a residency count:

```rust
struct Entry<P> {
    // ... as today: parent, tokens, page, bytes, children, holds, last
    /// Sequences (slots, or the one cache) whose GPU rows hold this page
    /// (`Tier::keys`). > 0: class 0, a duplicate of GPU rows, evicted first.
    gpu: u32,
}
```

The single `lru: BTreeSet<(u64, Key)>` becomes two sets of the same entries, by class: `dup` (evictable class 0)
and `lru` (evictable class 1). `make_room` (kvstore.rs:231) evicts `dup.iter().next_back()` (newest first) until it
is empty, then `lru.iter().next()` (oldest first, as today). `touch`, `hold`, `release` and `remove` move an entry
between the sets by `(evictable, gpu > 0)`, in the places that move it in and out of `lru` today. New calls:

- `resident(key, on: bool)`: `gpu += 1` / `-= 1`, re-filing the entry if its class changes.
- `insert_behind(parent, tokens, page, bytes) -> Option<Key>`: `insert`, except it may only use **free room**. It
  returns None when `used + bytes > cap` and evicts nothing (§2.3).
- `take_evicted() -> Vec<(Key, P)>`: today it returns the pages; it also returns their keys, for §2.2's eager pop.

**Why newest-first within class 0 (DECISION, Q2).** Among duplicates, the useful one is the copy of the slot the GPU
gives up *next*, because its spill then copies nothing. `reclaim` (engine.rs:1209) takes the least recently used idle
slot. The slots it takes last are the busy and recent ones, whose pages are also the most recently written or
touched. So "newest `last` first" approximates "the duplicate least likely to be needed soon". Within one slot, the
tail page has the newest tick (pages are written in position order), and only a leaf is evictable anyway, so this is
**tail-first**, as the plan asks. The alternative, LRU in class 0 too, would drop the next victim's copies first and
turn its reclaim into a full burst (§5.3).

**What the tree costs.** A slot's resident pages are a prefix of its chain (rows 0..held). A class-0 page with a
stored child (a RAM-only extension of the same prefix, e.g. a longer turn of the same conversation spilled from
another slot) is not a leaf and stays: a duplicate that can't go. These are bounded by the GPU-resident pages that
have a RAM-only extension. Freeing them would orphan the extension. That trade is the price of a tree store, and
the plan's fallback has it too. A gauge exposes it (§2.7).

### 2.2 The tier's per-sequence keys: soft holds, re-validated

`Tier::keys[s]` (tier.rs:249) keeps meaning "the stored pages of what sequence `s` holds, in order". What changes:

- `set_keys(s, new)` marks residency instead of holding the leaf: `resident(k, false)` for the old keys not in `new`,
  `resident(k, true)` for the new ones not in the old. Prefix order is kept, so this is a diff of two prefixes of
  one chain: O(changed). `truncate`, `push` and `adopt_held` go through it as today. The **hard** holds stay: the
  parent during `insert`/`room`, and `lookup`'s pages until `restore` lets them go (tier.rs:772–824). A copy in flight
  can never lose its page.
- **Eager pop on eviction.** A class-0 page can now be evicted while `s` still holds the rows. The evicted page is a
  leaf, so it has no stored child, so it is the *last* key of every `keys[s]` containing it: a chain's later key is
  a stored child of the earlier one. After every store call that can evict (`room`, `insert`, `insert_behind`),
  `Tier` drains `take_evicted()`. For each evicted key, every `s` whose `keys[s].last()` is that key pops it. That is
  O(sequences) per eviction (≤ 16 slots on the board's config). The arena gets the page's slot back, as
  `self.arena.free.extend(...)` does today.
- **`write` re-validates before extending** (the plan's requirement). It checks that `keys[s].last()` is still
  stored as the child of `keys[s][len-2]` with the slot's tokens; the eager pop makes this always true, so it is a
  `debug_assert!`, and in release builds a check that truncates `keys[s]` to its longest still-stored prefix. Then
  it extends from there. A page that the eager pop dropped from `keys[s]` is simply unstored again. The write-behind
  copies it again if free room appears, and the slot's spill copies it if the slot gives the rows up.
- Debug invariant, checked after each `Tier` call in debug builds:
  `Σ_entries gpu == Σ_s keys[s].len()`, and every `keys[s]` is a stored chain from `ROOT`.

### 2.3 Copies out: write-behind into free room, spill may evict

- **Write-behind** (`write_some`, tier.rs:746) uses `insert_behind`, so free room only. It never evicts class 0
  (a duplicate for a duplicate gains nothing) and never class 1 (RAM's own content beats a copy of GPU rows).
  Its order changes from round robin to **reclaim order**: idle slots by `Batch::used` ascending, the next
  `reclaim` victim first, then busy slots. Today's round robin can spend the free room on a busy slot that won't be
  reclaimed for minutes. With a full tier it copies nothing, so there is no write/evict loop by construction.
- **Spill** (`Batch::spill`, engine.rs:722, call sites unchanged as the plan says) is the copy that must happen: the
  rows are about to be released. It uses `room` + `insert` as today, which may evict class 0 (newest first), then
  class 1 (LRU). Then `truncate(s, keep / PAGE)` turns the pages past `keep` from class 0 into class 1
  (`resident(k, false)`), with no copy when they were written behind. That transition is the point of the design: a
  page leaves the GPU and RAM keeps it, as RAM's own.
- **`end_solo`** (engine.rs:1233) calls `spill(solo, usize::MAX)` today. That is not a give-up: the slot keeps its
  rows, so it is a write-behind of the whole S1 request on the request's last step (#170's second case). It is
  replaced by nothing: the slot's keys are already its stored pages, and the idle flush (§2.5) copies the rest.
  `#[cfg(test)] Faults::idle` (#181) then counts only when the engine has nothing left to flush, so #181's tests keep
  "the engine is done with the reply" (§4).

### 2.4 Copies in: `restore` down-classes what it moved

`PoolTier::restore` (tier.rs:786) copies pages `have/PAGE..to` into the slot. Then `adopt_held` makes them the
slot's keys, which with §2.2 marks them resident: **class 0**. The bytes stay in RAM. They are the first thing
evicted when room is needed, and if the slot is reclaimed again before that, its spill copies nothing.

The plan allows "drops or down-classes". Dropping is down-classing plus evicting at once, which frees room nobody
asked for yet and makes the next spill of that conversation pay the full copy. Down-class dominates it. The
fallback (§8) is "move on restore", which drops. On a failed restore, nothing changes from #181: the slot keeps
`have`, `tier_failed("restore")`, and the looked-up pages are released.

### 2.5 #170: the idle flush, budgeted and checked against the queue

- `Batch::step` (engine.rs:1294) always writes behind `STEP_BYTES` (16 MiB). The `usize::MAX` branch for an idle
  batch goes.
- The engine loop (serve.rs:1504–1517) blocks on `recv_timeout(100 ms)` only when the batch is idle, nothing waits,
  **and the tier has no backlog**. Backlog means a slot holds an unstored whole page and the store has free room for
  one: `PoolTier::backlog() -> bool`, O(slots). With backlog it calls `b.write_behind(IDLE_BYTES)`, where
  `IDLE_BYTES = 64 << 20` (the plan's number), then `jobs.try_recv()`. A job goes to `waiting` and the loop admits it
  on the same iteration; otherwise the loop runs again. A new arrival waits at most one piece: 64 MiB is 18 pages,
  ≈ 9.5 ms at bella's MEASURED 2D rate (§5.1). Today it waits for the whole backlog. Up to one pool's worth of pages
  is ≈ 0.9 s on bella (1,744 × 0.51 ms, ESTIMATED), not #170's 0.2 s, which was the 4090's rate.
- Free-room-only write-behind also bounds the backlog. Once the tier is full there is nothing to flush.
- The one-cache path (`Engine::run`'s `CacheTier::write` after each request: the CPU, or `--parallel 1`) is unchanged.
  It is not #170's case, and §2.1/§2.2 apply to it through the shared `Tier<P>`/`PageStore` with one sequence.

### 2.6 The C3 seam (#164's page-level eviction, gojo)

`PoolTier::spill_page(net, pool, parent, tokens, page)` (tier.rs:702; agreed on the r10 thread, not yet called) stays
as written. Under exclusivity it also has to end the page's GPU residency. Proposal: when C3 evicts a GPU page, it
calls `spill_page` and then `Tier::drop_resident(key)` (`resident(key, false)` for the slots whose keys hold it,
truncating them there), before `unpin_page`. With shared pages (refs > 1) the residency count equals the refcount, so
the page is class 1 only when the last GPU reference goes. Posted on the r10 thread. C3 lands after WP2 (plan §3).

### 2.7 /metrics

- New gauge `axon_kv_ram_gpu_pages`: class-0 pages (duplicates). RAM's useful content is `axon_kv_ram_pages −
  axon_kv_ram_gpu_pages`, which the board reports next to warm/GB.
- New counter `axon_kv_ram_spilled_pages_total`: pages copied by a spill (the burst copies on the admit path),
  separate from `axon_kv_ram_written_pages_total` (all copies out). This is what explains an ITL p99 move (§5.3).
- `Metrics::kv_ram_off` zeroes the new gauge with the others and keeps the counters (#181's rule). #181's tests
  assert the gauges and counters they know. The new ones are added to the same assertions.

## 3. Bit-exactness

- The tier moves **bytes only**. `page_out`/`page_in` are `cuMemcpy2DAsync`s of whole pages (`kvio::Rows::page`),
  and nothing numeric runs. A restored page holds the bytes the engine computed for those tokens at those positions.
  In the default (exact) mode those bytes don't depend on how the rows were computed: "with its prefix taken from
  the cache (a chat follow-up, a slot reused) or prefilled fresh: yes", decode or chunked prefill alike
  ([determinism.md](../determinism.md), the table under "Same bits?"). Main's slot reuse already relies on that, and
  #166's gates checked it for the tier (md5 vs main, gen 0/30 differ, serve 21/21 × 5 cells, 1031-070656-f427). The
  exceptions are unchanged: a streamed CPU/GPU split has no page image (`kv_ident` None, the tier is off there), and
  `fast` mode promises no bits with or without a tier.
- Exclusivity changes **which** pages RAM keeps and **when** they are copied, never their bytes, the kernels, or the
  order of a sequence's own computation. A page is used only after `lookup` matched its key *and its tokens*
  (kvstore.rs:266–281; a collision keeps what is there), and only after the whole restore succeeded
  (#181: trusted whole or not at all).
- The residency count only orders eviction. A wrong count (a bug) can change which page is evicted, and so speed,
  never the bytes a reply is computed from. The debug invariant (§2.2) catches it in the unit tiers.
- Tests that pin it (checkpoint 1/2, deterministic, no model where possible):
  - kvstore: class-0 evicted before class 1, newest first and leaf only; `insert_behind` never evicts; the
    class 0 → 1 → 0 transitions under `resident`; a chain's interior class-0 page stays while it has a child.
  - tier: a slot's leaf evicted as class 0, then the eager pop, then `write` re-copies; restore down-classes and a
    second spill copies 0 pages; the debug invariant through random op sequences (a seeded fuzz over
    insert/insert_behind/resident/lookup/release on `PageStore<Vec<u8>>`).
  - serve: `--cache-ram` vs `--cache-ram 0` replies bit-identical through a reclaim + restore cycle (#166's
    `serve_cuda_cache_ram` / `serve_cpu_cache_ram` pattern), and a resume under pressure restoring **all** of A's
    pages (the bella cell's shape at 0.5B scale: a tier the size of the pool, pressure past the pool).

## 4. Failure modes

| failure | behaviour | covered by |
|---|---|---|
| a copy fails (spill, write-behind, restore; `page_out`/`page_in` Err) | tier off, the server serves on as with `--cache-ram 0` (`Batch::tier_failed`, `tier::off`): unchanged | #181's three tests, unchanged, must pass; plus a spill fault after an eager pop (the re-validation path) |
| a restore whose `make_room` → `reclaim` → spill fails | tier off mid-restore, the slot keeps `have` (#181's fix) | `serve_cuda_cache_ram_off_in_reclaim` |
| `keys[s]` names an evicted page | impossible after the eager pop (`debug_assert`); release builds truncate to the stored prefix in `write` | tier unit test |
| residency count wrong | eviction order only, never bytes (§3); the debug invariant fails the unit tiers | fuzz test |
| the tier fills with class 1 | spills evict class-1 LRU (today's behaviour); write-behind stops (free room only) | kvstore test |
| interior duplicates pile up | bounded by resident pages with RAM-only extensions; visible as `axon_kv_ram_gpu_pages` | board scrape |
| spill burst on the admit path | the victim's unstored pages, ≈ 0.53 ms per page on bella; counted in `spilled_pages_total` | §5.3, ITL clauses |
| pinned RAM | unchanged from #166 (the allocator thread, `auto`'s ½-available / 8 GiB cap, the cgroup-aware check) | #166's tests |
| restore larger than the pool | `make_room` short, a partial restore of `to` pages (unchanged) | #166's tests |

## 5. Bella arithmetic, before building

### 5.1 Inputs (MEASURED, 1031-063509-f763's bandwidth phase, 7B shapes, bella, 2026-10-31)

| copy | rate | per 64-position page (3.50 MiB) |
|---|---|---|
| contiguous D2H / H2D, one 1.37 GiB copy | 28.62 / 28.67 GB/s | 0.13 ms |
| 2D, one per 64-position page (the tier's `page_out`/`page_in`) | 7.14 / 6.95 GB/s | **0.514 / 0.528 ms** |
| one page alone (latency-bound) | 3.60 / 3.48 GB/s | 1.02 / 1.05 ms |

A page is one `cuMemcpy2DAsync` of 56 rows (28 layers × K/V) of 64 KiB (`kvio::Rows::page`, kvio.rs:41). On this
GPU a 2D copy runs at ≈ 7.1 GB/s however wide its rows are: the whole slot as *one* 2D copy measured 7.15 GB/s too.
Contiguous runs at 28.6 GB/s. The plan's "414 pages × 0.19 ms ≈ 80 ms" is the 4090's rate. **On bella a 25k-token
spill or restore is ≈ 213–219 ms.** agent-workload §3's "≤ 0.1 s from RAM on bella (67 ms per 32k slot, #48)" does
not match f763. Checkpoint 1 re-measures it first (`kvio::tests::bandwidth`, ≈ 5 min on bella).

### 5.2 The resume cell (ESTIMATED)

`kv_resume_bench.py` @440fe18 at bella's defaults: A is 26,481 prompt tokens, its chain 414 pages (26,496
positions); the resume prompt is 26,692; the pool 1,744 pages; 2,066 pressure pages; the tier 1,755 pages.

- Capacity. A is reclaimed first (it is the LRU idle slot), so it becomes the oldest class 1. Class 1 needs at most
  A's 414 pages plus the pressure pages reclaimed, 2,066 + 414 − 1,744 = 736, for 1,150 pages in all, ≤ 1,755.
  **All 414 of A's pages survive** (inclusive: 103).
- The resume's own reclaim. It needs 414 + 4 pages from a pool full of idle pressure sessions. The victims (LRU
  pressure slots) were written behind while the tier had free room (1,755 − 414 = 1,341 pages of it). Newest-first
  class-0 eviction drops later sessions' copies first, so the victims' copies are likely still stored, and their
  spill is metadata only. ESTIMATED 0–0.2 s of burst, most likely 0.
- TTFT-resume ≈ restore 414 × 0.528 ms = **219 ms** + the 196-token tail prefilled at depth 26.5k (≈ 60–110 ms,
  ESTIMATED from the 32k long-context pp of 2,682 tok/s, docs/results.md) + one decode step (≈ 13 ms, f763's
  itl p50) ≈ **0.30–0.34 s**. A consistency check, not a calibration: on the 4090 the same restore is
  414 × 0.19 ms = 79 ms, and #166 MEASURED a 0.149 s resume there (plan §1). That leaves ≈ 70 ms for the tail and
  the step on a faster card, which is consistent with the 60–110 ms above for bella.
- **So the ≤ 0.3 s clause is at bella's copy floor, independent of the tier's policy.** The remedy, if checkpoint 1's
  re-measure confirms 7 GB/s: copy pages with a device gather. A kernel copies the 56 rows of N pages into a
  contiguous device staging buffer (≥ 64 MiB, D2D at hundreds of GB/s), then one contiguous D2H at 28.6 GB/s; H2D is
  the reverse, a scatter. ESTIMATED 414 pages in ≈ 55–65 ms, which puts the cell at ≈ 0.15–0.2 s. Bytes only (§3:
  a copy kernel moves bytes, so the bits are unchanged). Its own commit, measured with a new
  `kvio::tests::bandwidth` row before anything uses it. **Scope: Q1.**

### 5.3 The replay clauses (ESTIMATED)

- **f16 s4 re-prefill ≤ 5 %.** The ideal at pool + exclusive 6 GiB is 0.15 % (§1). Main measures ≈ 4.8× its ideal at
  pool-only (47.5 vs 9.97 %), and the 4090's 20G run measured 73.8 → 0.1 % (1027-155046-1ed0). ESTIMATED 0.5–5 %.
  It passes unless slot granularity costs as much on top of the tier as it does on the pool alone.
- **s4 ITL p99 ≤ 1.25× main** (main 137 ms MEASURED on the board → ≤ 171 ms). A step that runs a spill burst stalls
  every running decode. A 25k-token victim with nothing written behind is ≈ 213 ms on bella, over the bound alone. The
  clause fails if such steps reach ≈ 1 % of the ITL samples. The run has ≈ 174 admissions and a few running streams,
  so ≈ 500 affected samples out of tens of thousands. That is near 1 %, ESTIMATED. The mitigations are §2.3's
  reclaim-ordered write-behind, which keeps the next victim's copy, and the gather path (÷ 4).
  `spilled_pages_total` and `AXON_SERVE_TRACE` show which one bit.
- **s1 ITL p99 ≤ 1.05× main** (main 119 ms → ≤ 125 ms). At s1 the pool holds most of the working set (re-prefill
  10.3 % on main), reclaims are rare, and idle-flush pieces are ≤ 9.5 ms on an idle engine. ESTIMATED to pass.
- **replay md5 == main**: by §3.

## 6. The gate (pre-registered; verbatim from the plan, §3 row 2, kill rule)

> ≤ 0.3 s *and* re-prefill ≤ 5 % *and* s1 ITL p99 ≤ 1.05× main *and* s4 ITL p99 ≤ 1.25× main *and* replay md5 ==
> main; else `--cache-ram` stays opt-in and the fallback is "no write-behind, spill at eviction, move on restore"
> (simpler, bursty: 414 pages × 0.19 ms ≈ 80 ms per 25k reclaim). Pass → default `auto` (≤ 8 GiB) **for the paged
> GPU pool only**

The plan's metric column, as expectations, read but not gated: "rule B's bella cell with pressure
(`kv_resume_bench.py` @440fe18): TTFT-resume p50 25k, 2–10 min **8.3 s → ≤ 0.3 s**; agent-32k **f16 s4 re-prefill
47.5 % → ≤ 5 %, TTFT-turn 17 s → ≤ 0.5 s, goodput 5.5 → ≥ 20 tok/s**".

How each clause is read (all main vs candidate in the same job, bella, §7):

| clause | from | read |
|---|---|---|
| ≤ 0.3 s | resume cell, `ttft_resume` per round | median of the candidate's non-VOID rounds (3) |
| re-prefill ≤ 5 % | replay s4, `reprefill_share` (clamped) | median of the candidate's 3 clean runs |
| s1 ITL p99 ≤ 1.05× main | replay s1, `itl_p99_ms` | median cand / median main |
| s4 ITL p99 ≤ 1.25× main | replay s4, `itl_p99_ms` | median cand / median main |
| replay md5 == main | replay s1 and s4, `md5` | every clean run of both binaries, one md5 per speed |

Also required, not part of the rule: the unit tiers (`serve:: kvstore:: cuda::qwen2::slots::`, release and debug) and
md5 vs main at default flags (gen + serve, `md5.sh` through `axon-podjob`). With `--cache-ram 0` the product path must
be main's. Default flags don't reach the tier, so this proves the change is inert when off.

## 7. Bench methodology

- **Host and rules.** bella only (the gate is bella's cell), through `axon-podjob --on bella --timing` (exclusive
  GPU, cool gate), desktop idle. main = the merge base at gate time, built with the same recipe
  (`axon-rustc-build`). Fresh server per run, interleaved, one warm-up request per server (not measured), `/metrics`
  and `/slots` scraped after each run.
  **TAINTED = rerun**: a run whose foreign GPU load trips `agent_replay.py`'s rule (foreign SM mean > 5 % or max >
  25 %) is never read. That round is rerun, both binaries. The plan's budget is 1.5× the nominal time.
- **Resume cell.** The f763 script (`timing-bella2.sh`) with the candidate's flag:
  `kv_resume_bench-440fe18.py --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --trace --bin base=MAIN
  --bin cand=CAND:--cache-ram,6G --rounds 3 --gap 120`, at the script's defaults (`--tokens 25000 --others 3 --new 157
  --ctx 32768 --itl-streams 4`) and bella's server defaults (`--parallel 16`, a pool sized from free VRAM:
  1,570–1,744 pages). It is preceded by `kvio::tests::bandwidth` (AXON_BW_CTX=25600), whose rates go in the PR. A
  round whose session wasn't evicted is VOID (the script exits 3) and is rerun, never counted. INFO: restored
  tokens, `axon_kv_ram_*`, the trace's inter-step gaps. The exclusive tier has no time-based eviction, so the gap
  (2–10 min in the plan) changes nothing; 120 s is the cell of record. ≈ 20 min (f763: 1,175 s).
- **Replay.** `scripts/agent_replay.py run` (main's copy at gate time; its sha goes in the PR) against servers the
  gate script starts, as #166's `replay-pressure.sh` did (a scratch script; the gate's copy goes on the PR).
  Trace `board-trace-v1.json` (sha256 713285ba54487e17…, never committed), `--window-s 600 --scale 0.15
  --clip 32768` (the board's). Server = the board's axon cell: `serve Qwen2.5-7B-Instruct-Q4_K_M.gguf --device cuda --ctx 32768 --parallel 16 --kv f16`, plus `--cache-ram 6G` for
  the candidate. Speeds 1 and 4, 3 rounds each, order B C / C B / B C. ≈ 10 min a run (board wall 597–604 s,
  MEASURED), so 12 runs ≈ 2 h 10 min with server starts.
- **Total bella time per gate set**: ≈ 2.5 h nominal, ≈ 3.75 h with the taint budget, plus the unit tiers
  (393 s release, 414 s debug, MEASURED at #181's 1104-152814-7b76 / -d474) and md5 vs main (168 s at #166's
  1031-070656-f427).
- **RAM on bella.** 30.7 GiB MemTotal, shared with Haseeb's desktop. Jobs run in `app-axonjobs.slice` with a
  MemoryMax per claim. The gates pin **≤ 6 GiB** (`--cache-ram 6G`). Claim ≈ 16 GB per bench job (the 7B mmap
  4.4 GB + 6 GiB pinned + the process and the Python client) and ≥ 18 GB for the debug unit tier (#166/#181's OOM
  history). The plan's default flip (`auto`, ≤ 8 GiB) needs Haseeb's answer to the plan's Q2 ("May `axon serve` pin
  8 GiB of bella's RAM permanently...?") and is not part of this PR's gate.

## 8. If the gate fails

The rule says `--cache-ram` stays opt-in and the fallback is "no write-behind, spill at eviction, move on restore".
In this design the fallback is a configuration of the same code, so it costs a build, not a rewrite:
`STEP_BYTES = IDLE_BYTES = 0` (no write-behind, no idle flush), and `restore` drops what it moved (`resident` →
evict). The fallback gets its own A/B with the same rule before it is proposed. If only the ≤ 0.3 s clause fails and
§5.2's floor explains the gap, the PR says so with the numbers, and the gather path (Q1) is the next step. The rule
is not re-read after the fact.

## 9. Effort estimate

| step | content | ESTIMATED |
|---|---|---|
| checkpoint 1 | bandwidth re-measure on bella (§5.1); `PageStore` classes + `insert_behind` + evicted keys; the kvstore tests and the fuzz | 0.5 day |
| checkpoint 2 | `Tier` residency, eager pop, `write` re-validation, reclaim-ordered write-behind, `restore` down-class, `end_solo`, #170's loop, metrics; tier and serve tests; #181's tests green | 1 day |
| (Q1) gather path | the copy kernel, `kvio` rows, a bandwidth row, its tests (only if in scope) | +0.5–1 day |
| gates | unit tiers rel + dbg, md5, the resume cell, the replay board (§7), ≈ 4–5 h of bella | 1 day elapsed |
| review | 2 rounds (shady), log entry, PR body | 0.5 day |

≈ 3 days elapsed without the gather path and ≈ 3.5–4 with it (the plan says 1.5 rounds). The engine-side diff is
small: kvstore ≈ 120 lines, tier ≈ 150, engine/serve ≈ 60, ops ≈ 20. The tests are most of it (≈ 400 lines).

## 10. Questions for review

1. **The 0.3 s clause against bella's 2D copy floor** (§5.2): put the gather path in WP2's scope, gated with it? Or
   gate WP2 on the policy alone and accept that it may miss the 0.3 s clause on copy bandwidth? (My proposal:
   checkpoint 1 measures first. If the 2D rate is still ≈ 7 GB/s, the gather path goes in, as its own commit, before
   the gate set runs.)
2. **Class-0 order**: newest first (§2.1, my proposal), or LRU, or explicit next-victim priority (the tier tracks
   `Batch::used`)?
3. **Pinned size**: the plan's Q2 to Haseeb (8 GiB). The gates assume 6 GiB.
4. **#48's 67 ms per 32k slot vs f763's 2D rates**: which one holds today? Checkpoint 1's first job answers it.
5. **The one cache** (CPU, `--parallel 1`): exclusivity applies through the shared store, but the default flip is
   "for the paged GPU pool only". It stays opt-in there. Agreed?
