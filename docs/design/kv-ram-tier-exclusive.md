# An exclusive KV RAM tier (WP2, with #170): design note, checkpoint 0, revision 2

- Status: **revision 2 (2026-11-05): checkpoint 1's first bandwidth run, read PROVISIONALLY (§2.8.1), no product code
  yet.** Branch `feat/kvtier-exclusive`, main c54a730 merged in (84861fc). Author: joestar. Foreman: axon. Reviews of
  revision 0 (ac29b61): shady 7/10 REVISE (commit comment 208187552), Fable BUILD-WITH-CHANGES
  (`specs/kvexcl-cp0-review.md`). Every point either review raised is answered in place; §11 lists where, §12 what
  revision 2 changed. The foreman's decisions are final and recorded in §10.
- Plan: `specs/agent-workload-plan.md` (Fable, 2026-11-04), §3 row 2 ("Exclusive RAM tier (B2) + #170") and the
  RAM-tier paragraph of §1. Builds on #166 (`--cache-ram`, merged, opt-in, default 0), #172/#181 (the tier's off path,
  tested), #170 (the idle flush has no budget). Design context: [agent-workload.md §3](agent-workload.md).
- Labels: MEASURED (a job id or file is given), ESTIMATED (arithmetic from measured inputs, shown), DECISION (the
  foreman's, §10).
- Line numbers are at e325181 (src/serve is af281d9's, main). c54a730 changed src/serve only by `mod prefix;`
  (serve.rs:39), so serve.rs's numbers below line 39 are one higher there. The plan's numbers (`tier.rs:477–533`, ...)
  predate #181.

## 0. Summary

#166's tier is **inclusive**. The write-behind copies every page the GPU holds into RAM, and each slot holds its
chain's leaf (`Tier::set_keys`, tier.rs:284), so RAM fills with copies of what the GPU already has. On bella the
6 GiB tier is 1,755 pages and the GPU pool is 1,608–1,744, so the tier held about the pool's own content. The idle
session's tail was the LRU and went first. **1031-063509-f763: TTFT-resume medians 8.68 → 7.05 s, 6,592 of 26,496
tokens restored** (MEASURED). Useful capacity was ≈ tier − pool where it should be tier + pool.

The change. Items 1–5 are the store's eviction order and the tier's bookkeeping, with no byte changes:

1. **Two LRU classes in `PageStore`** (§2.1). A page whose bytes are in the GPU pool is *class 0*, a duplicate. It is
   evicted first, newest first, which is tail-first within a slot. A page only RAM has is *class 1*: plain LRU, as
   today, evicted only when no class-0 page is evictable. The slot's leaf hold becomes this residency mark, a soft hold.
2. **Write-behind into free room only**, checked before the copy, in reclaim order (§2.3). It never evicts. A spill
   (a slot giving up rows) may evict, class 0 first.
3. **`write` re-validates the slot's keys** against the store before extending them. A class-0 page can now be evicted
   while its slot still holds it (§2.2).
4. **`restore` down-classes what it moved in** (§2.4). The copies stay as class 0, so a later spill of the same slot
   copies nothing while they last.
5. **#170: the idle flush is budgeted** (64 MiB a piece), checks the queue between pieces, and blocks when a piece
   makes no progress (§2.5). `end_solo` no longer flushes.
6. **The copy path** (§2.8, DECISION Q1). A page moved at bella's contiguous rate instead of its 2D rate. Its own
   gated commit, in its own PR that lands first, chosen by checkpoint 1's measurements. **Revision 2, PROVISIONAL
   (§2.8.1):** checkpoint 1's first run (TAINTED) rejects the loop and puts today's 2D copy at the contiguous rate
   (28.1 GB/s). Bella's 2D copy is bimodal, 28 or 7.1 GB/s depending on machine state, the same binary. No copy-path
   PR until the rerun measures the slow mode, idle and busy.

**Steady state (§2.3, Fable's H1).** The agent replay's working set is 5,164 distinct pages against a 1,755-page tier,
so the tier is full within about the first third of a run and stays full. From then on, the write-behind and the idle
flush copy nothing. Every reclaim spills the victim's unstored pages, and the class-0 copies drain until a victim's
spill is about its whole chain. **Items 1–4 carry the capacity clauses. The ITL clauses depend on the copy rate
(item 6). The write-behind protects them only while the tier warms up.**

Bits: unchanged by construction (§3). The tier's off path from #181 is unchanged in substance. One of its test cases
targets the `end_solo` flush that item 5 removes, so it must be pointed at a spill that still copies (§4).

**The gate (verbatim, §6)** does not move. On bella's MEASURED slow-mode 2D page copy (7.1 GB/s, 0.53 ms a page),
the resume comes out at **≈ 0.30–0.34 s** and the s4 ITL p99 at **≈ 360–420 ms**, both failing on the copy floor
whatever the policy does. At the contiguous rate (28.6 GB/s; 2D's own rate in the fast mode) they come out at
**≈ 0.14–0.19 s** and **≈ 150–215 ms**: a pass and a coin flip (§5, all ESTIMATED). Hence item 6, and the order:
measure the copy shapes first (checkpoint 1), then the policy.

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
`agent_replay.plan()` as the board runs it, window 600 s, scale 0.15, clip 32768. That is 174 requests, 2,921,348
prompt tokens and 5,164 distinct 64-token pages. A request's prompt has median 17,211 tokens (269 pages; p25 16,309,
p75 19,219), and a request adds a median of **2 never-sent pages** (p75 5, p90 15, mean 17.9). Concurrency is ignored,
so the numbers are optimistic. Scripts: `ws.py` / `ws3.py`, attached to the PR. They read the never-committed trace
and print numbers only. The GPU pool is the board's axon cell: 1,608–1,609 pages, MEASURED in
`board-f16-1027-050316/axon-f16-1860{1,4,7}.log`. f763's resume cell had 1,744.

| capacity (pages) | what it models | ideal re-prefill |
|---|---|---|
| 1,608 | the board's f16 pool alone (main) | 8.83 % |
| 1,744 | f763's pool alone | 6.97 % |
| 1,755 | the inclusive 6 GiB tier (≈ max(pool, tier)) | 6.82 % |
| 2,778 | board pool + an exclusive 4 GiB tier | 0.69 % |
| **3,363** | **board pool + an exclusive 6 GiB tier** | **0.15 %** |
| 3,948 | board pool + an exclusive 8 GiB tier | 0.15 % |

The measured main is above its ideal, much more so at s4. The f16 board measured 47.5 % at s4 and 10.3 % at s1
(`~/.cache/axon-agent/board-f16-table-clean.md`, MEASURED) against 8.83 % ideal: 5.4× at s4, 1.17× at s1. Slot
granularity, preemption and concurrent sessions cost more than LRU arithmetic shows, so the table ranks the options
and does not predict the gate. What it does show: an inclusive 6 GiB tier cannot reach ≤ 5 % even ideally (6.82 %),
while an exclusive one has ≈ 30× headroom ideally.

## 2. Design

### 2.1 `PageStore`: two classes, one tree

The store is a tree. Pages chain from `ROOT`, only unheld leaves are evictable, and parents outlive children
(kvstore.rs:231–264). That stays. Each `Entry` gains a residency count:

```rust
struct Entry<P> {
    // ... as today: parent, tokens, page, bytes, children, holds, last
    /// Holders on the device of this page's bytes: a pool slot's rows in
    /// WP2 (`Tier::keys`), an indexed GPU page under C3 (§2.6).
    /// > 0: class 0, a duplicate of bytes the GPU pool has, evicted first.
    gpu: u32,
}
```

**The invariant (Fable's H3): class 0 ⇔ the page's bytes are in the GPU pool, whatever holds them there.** In WP2 the
only holders are slots' rows, so `gpu` = the number of `keys[s]` that contain the key. C3 adds a second kind of
holder (§2.6).

The single `lru: BTreeSet<(u64, Key)>` becomes two sets of the same entries, by class: `dup` (evictable class 0)
and `lru` (evictable class 1). `make_room` (kvstore.rs:231) evicts `dup.iter().next_back()` (newest first) until it
is empty, then `lru.iter().next()` (oldest first, as today). `touch`, `hold`, `release` and `remove` move an entry
between the sets by `(evictable, gpu > 0)`, in the places that move it in and out of `lru` today. New calls:

- `resident(key, on: bool)`: `gpu += 1` / `-= 1`, re-filing the entry if its class changes.
- `has_room(bytes) -> bool`: `used + bytes <= cap`. It evicts nothing; the write-behind asks it before copying (§2.3).
- `insert_behind(parent, tokens, page, bytes) -> Option<Key>`: `insert` restricted to **free room**. It returns None
  when `used + bytes > cap` and evicts nothing.
- `take_evicted() -> Vec<(Key, P)>`: today it returns the pages; it also returns their keys, for §2.2's eager pop.

**Newest-first within class 0 (DECISION Q2).** Among duplicates, the useful one is the copy of the slot the GPU gives
up *next*, because its spill then copies nothing. `reclaim` (engine.rs:1209) takes the least recently used idle slot.
The slots it takes last are the busy and recent ones, whose pages are also the most recently written, restored or
touched. `lookup` touches a restored chain leaf to root (kvstore.rs:277), so every page of it, the leaf included, is
newer than every older chain's. So "newest
`last` first" approximates "the duplicate least likely to be needed soon". Within one slot the tree forces tail-first
whatever the ticks (only a leaf is evictable). Newest-first also concentrates evictions on few slots, which keeps the
number of burst-affected ITL samples small: p99 counts samples, not burst size. The alternatives are worse or not
needed yet. LRU-first would eat the next victim's copies first. An explicit next-victim priority is newest-first in
effect, but costs re-keying on every `touch(slot)`. It is deferred: the trace line's per-reclaim spill count (§2.7)
shows whether bursts spread across slots, and only then is it revisited.

**What the tree costs.** A slot's resident pages are a prefix of its chain (rows 0..held). A class-0 page with a
stored child stays: it is not a leaf, so it is a duplicate that can't go. An example is a RAM-only extension of the
same prefix, such as a longer turn of the same conversation spilled from another slot. These pages are bounded by
the GPU-resident pages that have a RAM-only extension. Freeing them would orphan the extension. That trade is the
price of a tree store, and the plan's fallback has it too. A gauge exposes it (§2.7).

### 2.2 The tier's per-sequence keys: soft holds, re-validated

`Tier::keys[s]` (tier.rs:249) keeps meaning "the stored pages of what sequence `s` holds, in order". What changes:

- `set_keys(s, new)` marks residency instead of holding the leaf: `resident(k, false)` for the old keys not in `new`,
  `resident(k, true)` for the new ones not in the old. Prefix order is kept, so this is a diff of two prefixes of
  one chain: O(changed). `truncate`, `push` and `adopt_held` go through it as today. The **hard** holds stay: the
  parent during `insert`/`room`, and `lookup`'s pages until `restore` lets them go (tier.rs:772–824). A copy in flight
  can never lose its page.
- **Eager pop on eviction.** A class-0 page can now be evicted while `s` still holds the rows. An evicted page is a
  leaf with no stored child, so it is the *last* key of every `keys[s]` containing it: a chain's later key is a stored
  child of the earlier one. Every store call that can evict (`room`, `insert`) is followed by `Tier` draining
  `take_evicted()` into a set `E`. Then every `s` runs `while keys[s].last() ∈ E { pop }`. That loop doesn't depend on
  the order of evictions: a `make_room` that takes a chain's leaf and then its parent pops both. The pop doesn't call
  `resident(false)`, because the entry and its count are already gone. The cost is O(sequences + evicted) per drain
  (≤ 16 slots on the board's config). The arena gets the pages' slots back, as `self.arena.free.extend(...)` does today.
- **`write` re-validates before extending** (the plan's requirement). It checks that `keys[s].last()` is still
  stored as the child of `keys[s][len-2]` with the slot's tokens. The eager pop makes this always true, so it is a
  `debug_assert!`. In release builds it is a check that truncates `keys[s]` to its longest still-stored prefix. Then
  `write` extends from there. A page that the eager pop dropped from `keys[s]` is simply unstored again: the
  write-behind copies it again if free room appears, and the slot's spill copies it if the slot gives the rows up.
- **Debug invariants**, checked after each `Tier` call in debug builds (and by the fuzz, §3):
  - `Σ_entries gpu == Σ_s keys[s].len()`;
  - every key in every `keys[s]` is stored, and `keys[s]` is a chain from `ROOT`;
  - `gpu(parent) ≥ gpu(child)`: residency only goes down along a chain, so a class-0 leaf's parent goes back to
    `dup`, never to `lru`.
- **The invariant that protects the bytes** is `keys[s] == kvstore::keys(&held[s])[..keys[s].len()]`, not the count.
  A wrong count changes only the eviction order (§3). It is checked at the `Batch` level (it needs `held`) in debug
  builds after `spill`, `write_behind`, `restore` and `release`. No path makes the two diverge today, and no new path
  does.

### 2.3 Copies out: write-behind into free room, spill may evict

- **Write-behind** (`write_some`, tier.rs:746) asks `has_room(page)` **before** each `page_out` and stops at the first
  no. Then `insert_behind`. A full tier therefore costs no D2H. Without the check it would cost one wasted page copy
  per slot per step, up to 16 × 0.5 ms. The write-behind never evicts class 0 (a duplicate for a duplicate gains
  nothing) and never class 1 (RAM's own content beats a copy of GPU rows). Its order changes from round robin to
  **reclaim order**: idle slots by `Batch::used` ascending, the next `reclaim` victim first, then busy slots. Today's
  round robin can spend the free room on a busy slot that won't be reclaimed for minutes. There is no write/evict loop
  by construction.
- **Spill** (`Batch::spill`, engine.rs:722; call sites unchanged, as the plan says) is the copy that must happen,
  because the rows are about to be released. It uses `room` + `insert` as today, which may evict class 0 (newest
  first), then class 1 (LRU). Then `truncate(s, keep / PAGE)` turns the pages past `keep` from class 0 into class 1
  (`resident(k, false)`), with no copy for the ones already written behind or restored. That transition is the point
  of the design: a page leaves the GPU and RAM keeps it, as RAM's own.
  **Checkpoint 2 change (found by serve test (a), §3):** a spill whose `keep` covers every whole page the slot holds
  (`admit` / `lend_solo` of a conversation's next turn in its own slot) gives nothing up, and copies nothing. Today's
  `spill` copies the slot's whole unstored chain whatever `keep` is. Under the inclusive tier that was nearly free,
  because the write-behind had already stored those pages. Under this design, on a full tier, the write-behind
  copies nothing, so a next turn's admission would copy its conversation's whole chain (steady state: ≈ 269 pages,
  ≈ 0.13 s at 28 GB/s) before its first token, evicting class 0 and then class 1 to hold duplicates of rows the GPU
  keeps. A partial give-up still copies the kept parents, because a stored page needs its stored parent.
- **Steady state (Fable's H1, shady's M4).** These are the facts the design has to live with:
  1. **The tier fills early.** The working set (5,164 pages) is ≫ the tier (1,755). A page is copied the first time
     the GPU holds it, so the tier fills once ≈ 1,755 / 5,164 = 34 % of the distinct pages have been computed, about
     the first third of a run (ESTIMATED: arrivals taken as uniform). It then stays full. Free room appears only
     transiently, and the next spill fills what it freed.
  2. **The free-room copies stop.** `insert_behind` and the idle flush copy nothing.
  3. **The class-0 copies drain.** Every reclaim spills the victim's unstored pages and evicts that many class-0
     pages to make room, newest first, so from the most recently admitted slots. Those slots re-copy them at their own
     reclaim. The burst grows each generation, u_{n+1} ≈ r_new + u_n, until it reaches the chain. Restores add class 0
     (r pages per admission), and reclaims turn the victim's chain into class 1, so the class-0 count falls by ≈ u per
     admission and empties within tens of admissions (ESTIMATED).
  4. **A steady-state spill is ≈ the victim's whole chain**, median 269 pages, and not its new work: a request adds a
     median of 2 new pages (§1).

  So: the capacity clauses pass or fail on the policy (§1's sim), and **the ITL clauses are set by the copy rate**.
  The write-behind doesn't protect them; reclaim order matters only while the tier warms up (as in the resume cell,
  §5.2). Fixing that is out of WP2. A write-behind that may evict later victims' duplicates needs ≈ 128 MiB a step at s4
  (Fable, ESTIMATED), and a copy stream of its own to cost no step time: WP2b (§8).
- **`end_solo`** (engine.rs:1233) calls `spill(solo, usize::MAX)` today. That is not a give-up: the slot keeps its
  rows, so it is a write-behind of the whole S1 request on the request's last step (#170's second case). It is
  replaced by nothing. The slot's keys are already its stored pages, and the idle flush (§2.5) copies the rest while
  there is free room. On a full tier an S1 request isn't written behind before its reclaim, which is consistent with
  steady state above. The case of #181's tests that targets this spill moves (§4).

### 2.4 Copies in: `restore` down-classes what it moved

`PoolTier::restore` (tier.rs:786) copies pages `have/PAGE..to` into the slot. Then `adopt_held` makes them the
slot's keys, which with §2.2 marks them resident: **class 0**. The bytes stay in RAM. They are the first thing
evicted when room is needed (newest first). If the slot is reclaimed again before that happens, its spill copies
nothing.

The plan allows "drops or down-classes". Dropping is down-classing plus evicting at once: it frees room nobody asked
for yet and makes the next spill of that conversation pay the full copy. Down-class dominates it: the same copies, and
strictly more zero-copy reclaims. On a failed restore nothing changes from #181: the slot keeps `have`,
`tier_failed("restore")` runs, and the looked-up pages are released. `restore`'s precondition comment (tier.rs:800–802:
page `i` must be this slot's alone, #164's copy-on-write) stays as it is. That is where the immutability rule of the
plan's E3 meets WP2, and WP2's code doesn't touch it.

### 2.5 #170: the idle flush, budgeted, checked against the queue, never spinning

- `Batch::step` (engine.rs:1294) always writes behind `STEP_BYTES` (16 MiB). The `usize::MAX` branch for an idle
  batch goes.
- **Backlog**, defined exactly: some slot holds an unstored whole page, *and* the store has free room for one page
  without evicting (`has_room`). That is `PoolTier::backlog() -> bool`, O(slots). A full tier with pages unstored has
  no backlog.
- **The loop** (serve.rs:1504–1517). It blocks on `recv_timeout(100 ms)` when the batch is idle, nothing waits, and
  either there is no backlog or the tier is `stalled` (below). Otherwise it runs one piece, `b.flush_idle(IDLE_BYTES)
  -> usize` with `IDLE_BYTES = 64 << 20` (the plan's number), then `jobs.try_recv()`:
  - `Ok(job)`: to `waiting`, admitted on the same iteration.
  - `Err(Disconnected)`: return, as the blocking branch's `Err(_)` does.
  - `Err(Empty)` with `stop` set: return. Shutdown doesn't wait for a backlog; the tier dies with the process.
  - `Err(Empty)` otherwise: the loop runs again.

  A new arrival waits at most one piece. Under `div_ceil(..).max(1)` (tier.rs:753) 64 MiB is 19 pages: ≈ 9.8 ms at
  bella's 2D rate, ≈ 2.5 ms at the contiguous rate. Today it waits for the whole backlog, up to a pool's worth:
  ≈ 1,744 × 0.514 ≈ 0.9 s on bella (ESTIMATED), not #170's 0.2 s, which was the 4090's rate.
- **Liveness (shady's M2).** `flush_idle` returns its progress: pages copied plus pages adopted through `child()` hits.
  A piece with **zero** progress sets `stalled`, and the loop then blocks as if there were no backlog. `stalled`
  clears when the batch next steps or admits, i.e. when something changed. Two cases make no progress while
  `backlog()` holds:
  - the arena can't grow: `Alloc::Thread` stopped short ("pinned X of Y; it stays that size", tier.rs:514), so
    `arena.take()` is None on every piece;
  - a key collision: `insert` keeps what is there.

  Without the rule the engine spins at 100 % CPU on an idle server, and in the collision case it also wastes a
  `page_out` per turn. With it, the collision case costs at most one wasted copy per step.
- **`Faults::idle` (#181, serve.rs:1511–1512) counts each blocking wait**, so it fires both with no backlog and when
  stalled. In #181's tests the tier is 64 MiB, ≈ 85 pages at 0.5B (768 KiB a page), and it never fills: "idle" still
  means "the tail is flushed", and their counters hold.
- The one-cache path (`Engine::run`'s `CacheTier::write` after each request: the CPU, or `--parallel 1`) is unchanged.
  It is not #170's case, and §2.1/§2.2 apply to it through the shared `Tier<P>`/`PageStore` with one sequence.

### 2.6 The C3 seam (#164's page-level eviction, gojo)

The invariant is §2.1's: class 0 ⇔ the bytes are in the GPU pool, whatever holds them. C3 changes who holds them, so
its two events must keep the count, not just eviction (Fable's H3):

- **Reclaim keeps indexed pages warm** under C3: the slot is released, and its indexed pages stay on the GPU (plan §3
  row 3). WP2's `truncate(s, 0)` would then `resident(k, false)` the slot's keys and make RAM's copies class 1 while
  the bytes are still in VRAM, which is inclusive again for the warm set. C3 must move the residency from `keys[s]` to
  the prefix index at reclaim: `resident(k, true)` for the index's holder before the slot's `resident(k, false)`, so
  the net is unchanged.
- **GPU eviction**: `pop_lru_leaf` → `PoolTier::spill_page(net, pool, parent, tokens, page)` (tier.rs:702; agreed on
  the r10 thread, not yet called) → `Tier::drop_resident(key)` (`resident(key, false)` for the index's holder), before
  `unpin_page`. #164's C1 + C2 landed in 0e014cd as specified here: `src/serve/prefix.rs` returns an `Evicted { key,
  parent, tokens, page }` from `pop_lru_leaf` and `unindex_subtree`, to be copied out before the unpin. The engine
  takes it in C3, so nothing in WP2 changes.
- **Shared pages** (refs > 1): `gpu` counts device holders, so a page becomes class 1 only when its last holder on the
  GPU goes.

The field is named for that: `gpu` = holders on the device, not "slots". Posted on the r10 thread; C3 lands after WP2
(plan §3).

### 2.7 /metrics and the trace

- New gauge `axon_kv_ram_gpu_pages`: class-0 pages (duplicates). RAM's useful content is `axon_kv_ram_pages −
  axon_kv_ram_gpu_pages`, which the board reports next to warm/GB. In steady state it is expected near 0 (§2.3). If
  it stays ≫ 0 with small spills, H1's drain argument is wrong and the write-behind matters.
- New counter `axon_kv_ram_spilled_pages_total`: pages copied by a spill (the burst copies on the admit path),
  separate from `axon_kv_ram_written_pages_total` (all copies out).
- `AXON_SERVE_TRACE`'s admission line gains `spilled N restored M` (pages): the spill burst and the restore of that
  admission's step. That attributes a step over 171 ms to the copies or to something else (§5.3, §6), and answers
  Q2's "spread bursts" question per reclaim.
- `Metrics::kv_ram_off` zeroes the new gauge with the others and keeps the counters (#181's rule). #181's tests
  assert the gauges and counters they know; the new ones are added to the same assertions.

### 2.8 The copy path (DECISION Q1: in scope, its own gated commit and PR, landing first)

Two gate clauses depend on the copy rate (§5.2, §5.3), and the 2D rate fails both on its own. `Rows::page`
(kvio.rs:41–66) issues one `cuMemcpy2DAsync` per region per page: 56 rows (28 layers × K/V) of 64 KiB, `max_ctx * row`
apart on the device. In f763 that ran at ≈ 7.1 GB/s however the rows were cut (§5.1). The question is which page
copy shape runs near the contiguous 28.6 GB/s. (Revision 2: in 7 of 10 bella runs the 2D copy itself does, §2.8.1.)

**Checkpoint 1 measures first** (`kvio::tests::bandwidth`, bella, `--timing`, 7B shapes, a 25,600-position slot as
f763, and 32,768 for #48's slot). Every new row also prints the time to *enqueue* its copies, because the engine
thread pays that. A 56-copies-per-page loop issues 23,184 copies for a 414-page restore. The rows:

| row | shape | what it decides |
|---|---|---|
| 2D per 64-pos page (exists) | `Rows::page` today | the re-measure of f763's 7.1 GB/s |
| 56 × 1D per 64-pos page | one `cuMemcpyAsync` per 64 KiB row, host in image layout (Fable) | ≥ 20 GB/s: the fix is a loop in `Rows::page`, no kernel |
| 1D per page, scattered host | one 3.5 MiB copy per page, device contiguous, host slots in a stride permutation (shady: arena slots aren't contiguous within a session) | the host side of any gather path |
| D2D 2D / 56 × 1D per page | a page's rows gathered into (scattered from) a 64 MiB device staging buffer by driver copies | whether a gather needs a kernel of ours |
| gather path / scatter path | D2D per page into staging, then 1D per page to scattered host slots (18 pages a fill; restore the reverse) | the whole no-kernel gather, end to end |

**The decision rule** (the foreman's, extended to the no-kernel gather):
1. If 56 × 1D per page is ≥ 20 GB/s both ways (with an enqueue time well under the copy time), the fix is a loop in
   `Rows::page`. No staging, no kernel.
2. Else, if the gather and scatter paths are ≥ 20 GB/s, the fix is driver copies through a 64 MiB device staging
   buffer.
3. Else a gather kernel (D2D into the staging buffer, one contiguous D2H per fill; H2D the reverse), with its own
   bandwidth row.

Either way the copy moves bytes only, so the bits don't change (§3). md5 vs main and #166's bit-identity serve tests
gate it. Any staging buffer is accounted for: 64 MiB is 18.3 pool pages (≈ 1.1 % of 1,744), allocated before the pool
is sized, so the pool loses 19 pages. The resume cell reports its pool count. The copy-path PR lands before the policy
with its own gates (unit tiers, md5, the bandwidth rows), which keeps the gate's attribution clean: main has no tier
by default, so main's ITL is unaffected by it.

### 2.8.1 Checkpoint 1's first rows (1104-164604-7732, TAINTED): the provisional reading

MEASURED, bella, `AXON_BW_CTX=25600` (400 pages), 28da662, 2026-11-05 11:11. TAINTED: FOREIGN_GPU max 99 %, mean
53.5 % over its 2 samples (Xorg 66 %, zen 33 %). GB/s, medians of 7 (31 for one page); enqueue = host time to issue a
row's copies:

| row | D2H | H2D | enqueue, 400 pages |
|---|---|---|---|
| contiguous, one 1.37 GiB copy | 28.63 | 28.68 | |
| 2D per 64-pos page (`Rows::page` today) | **28.07** | **28.27** | |
| 2D, one page alone | 26.88 | 27.20 | |
| 56 × 1D per page | 16.72 | 17.45 | 83.8 / 80.5 ms |
| 56 × 1D, one page | 16.52 | 17.04 | 0.05 ms |
| 1D per page, scattered host slots | 28.02 | 28.29 | 0.37 / 0.38 ms |
| D2D 2D per page, gather / scatter | 299.1 / 298.4 | | 0.58 / 0.57 ms |
| D2D 56 × 1D per page, gather / scatter | 42.5 / 42.2 | | 34.6 ms |
| gather path / scatter path (2D D2D + 1D, 18 pages a fill) | 25.54 | 25.57 | 0.88 / 0.87 ms |

**The rule, read against the taint:**
1. **Rule 1, the loop: rejected, robustly.** 16.7 / 17.5 GB/s is under 20, and its enqueue is 95 % of its copy time
   (≈ 3.7 µs of driver time per `cuMemcpyAsync`, 22,400 of them). The taint can't plausibly lift it past 20: the
   enqueue is host CPU time, and the one-page row, which isn't enqueue-bound, runs at 16.5–17.0 GB/s (≈ 3.9 µs per
   64 KiB copy on the engine). In the same run today's 2D path is 1.6× faster.
2. **Rule 2, staged driver copies: it passes (25.5 / 25.6 GB/s), but in this run it loses to today's 2D path**
   (28.07 / 28.27 GB/s), which also needs no staging and costs no pool pages (§2.8's 19). This holds under any taint
   correction: the 2D row is at 98 % of the contiguous rate, and the staged path is a contiguous-shape PCIe leg plus a
   D2D leg on the same stream, so nothing can lift it above the 2D path.
3. **So in this run the copy path is the third option: `Rows::page` as it is.**

**f763's 7.1 GB/s was not the taint, and this run doesn't overturn it.** Ten bella runs of these rows since 10-16
(§5.1's table) put the 2D page copy at 27.6–28.4 GB/s in seven and at 7.14–7.18 GB/s in three: 05d7 (10-30 21:53),
3742 (10-31 05:12) and f763 (10-31 06:37). Contiguous ran at 27.9–28.8 GB/s in all ten. The same binary (92cb99a)
ran fast at 10-31 02:25 (ef2b) and slow at 05:12 (3742), so the cause is machine state, not code. All three slow runs
had an idle desktop after their first sample (0–1.5 % foreign GPU). In all seven fast runs kitty, Xorg or zen was
active (1.8–31 %). **Hypothesis (ESTIMATED, n = 10, correlation only):** with the desktop idle, the GPU stays in a
low performance state for copy-only work. A 2D copy is then limited by the copy engine's clock (≈ ¼ rate) and a
linear copy by PCIe. If so:
- `--timing` waits for a quiet GPU, and §7's cells run with the desktop idle. That selects for the slow mode: the
  untainted rerun is likely to land in it, and so is the resume cell. That cell restores after a 120 s idle gap.
- Serving's admission steps run among its own kernels. They may never see the slow mode.

Nothing has measured any other shape in the slow mode, or any shape with the GPU busy, and those decide the copy path
(§5.2, §5.3), not the taint. The bench gains both (8c7cf14, test only):
- `GPU state` lines: nvidia-smi's pstate, clocks, `display_active` and clock event reasons, at the start, idle after
  the cold rows, and twice during the busy pass.
- **A busy pass**: a one-block kernel spins on a second stream (32 threads, no memory traffic) while the 2D page
  rows and every `copy_paths` row run again, marked `(busy)`.

**PROVISIONAL (DECISION: it changes §10.1, so it needs the foreman's yes):**
- **The loop is out**, final, whatever the rerun shows.
- **No copy-path PR now. `Rows::page` stays.** The policy (checkpoints 1–2) is built now. It doesn't depend on the
  copy path: it calls `page_out` / `page_in`, and the copy path sits below them in `Rows::page`.
- **The untainted rerun (`-r2`, with the busy pass) settles it:**
  - (a) 2D per page ≥ 20 GB/s cold and busy: the copy path is `Rows::page` as is. FINAL.
  - (b) slow cold, ≥ 20 GB/s busy: the copy is kept for the replay's admission steps. The resume cell's cold restore
    is the open question. Its trace (`restored` and the admission step's ms, §2.7) gives the rate the restore got,
    and the copy path is decided on that number, before the gate is read.
  - (c) slow busy too: the copy path is whichever shape is fastest in that state, among the scattered 1D page copy
    behind a device gather, the staged path (if its D2D leg holds up) and a gather kernel (rule 3). It goes in its
    own PR first, as §10.1 says.
- **If the rerun lands in the fast mode again**, the slow mode stays unmeasured. The resume cell then reads its own
  restore rate as in (b), and I report it rather than re-read the rule.

Jobs (8c7cf14, `--timing bella`): **1105-112352-139c** (`kvexcl-8c7cf14-bw25600-r2`, the rerun of record) and
1105-112353-0d10 (`kvexcl-8c7cf14-bw32768`, #48's slot; it replaces 1104-164605-53d3, which never ran). This section
turns PROVISIONAL → FINAL, or not, on 139c's rows, and the PR body quotes it.

## 3. Bit-exactness

- The tier moves **bytes only**. `page_out`/`page_in` are copies of whole pages (`kvio::Rows::page`: 2D today; 1D or
  staged after §2.8), and nothing numeric runs. A restored page holds the bytes the engine computed for those tokens at
  those positions. In the default (exact) mode those bytes don't depend on how the rows were computed: "with its
  prefix taken from the cache (a chat follow-up, a slot reused) or prefilled fresh: yes", decode or chunked prefill
  alike ([determinism.md](../determinism.md), the table under "Same bits?"). Main's slot reuse already relies on that,
  and #166's gates checked it for the tier (md5 vs main, gen 0/30 differ, serve 21/21 × 5 cells, 1031-070656-f427).
  The exceptions are unchanged. A streamed CPU/GPU split has no page image (`kv_ident` None, so the tier is off there),
  and `fast` mode promises no bits with or without a tier.
- Exclusivity changes **which** pages RAM keeps and **when** they are copied. It never changes their bytes, the
  kernels, or the order of a sequence's own computation. A page is used only after `lookup` matched its key *and its
  tokens* (kvstore.rs:266–281; a collision keeps what is there), and only after the whole restore succeeded (#181:
  trusted whole or not at all).
- The residency count only orders eviction. A wrong count (a bug) can change which page is evicted, and so speed,
  never the bytes a reply is computed from. The bytes rest on `keys[s]` matching `held[s]`'s chain (§2.2's
  Batch-level check).

**Tests** (checkpoints 1–2; deterministic, no model where possible):

- **kvstore** (checkpoint 1):
  - class 0 is evicted before class 1, newest first and leaf only;
  - `has_room` and `insert_behind` never evict;
  - the class 0 → 1 → 0 transitions under `resident`;
  - a chain's interior class-0 page stays while it has a child;
  - `take_evicted` returns the keys.
  - The seeded `PageStore<Vec<u8>>` fuzz over insert / insert_behind / resident / lookup / release.
- **Tier fuzz (shady's M3).** `Tier<P>` with P a CPU stand-in for an arena slot (no GPU), ≥ 3 sequences whose chains
  share prefixes (so `gpu` ≥ 2 on shared pages). Random `set_keys` / `push` / `truncate` / `adopt_held` / stores through
  `room` + `insert` and `insert_behind` / `lookup` + release, including `make_room` calls that evict several leaves of
  one chain in one go. After every op, §2.2's three invariants hold, and no `keys[s]` names an unstored key. Seeded and
  reproducible; the seed is printed on failure.
- **Tier unit tests:**
  - a slot's leaf is evicted as class 0, then the eager pop, then `write` re-copies it;
  - a restore down-classes, and a second spill of the slot copies 0 pages;
  - (Fable) a `Restore` fault after a spill that evicted class 0 under the restoring slot's own chain: the eager pop
    and `lookup`'s hard holds must not conflict, and the tier goes off with the slot at `have`.
  - `keys_hold_the_leaf` (tier.rs:977) asserts today's leaf hold, the opposite of the new behaviour. It is replaced by
    `keys_are_class0_and_drain_pops_them` (checkpoint 2's name): a slot's stored pages are class 0 and evictable
    (newest first) while it holds them, and
    class 1 after `truncate`.
- **Built in checkpoint 2 (a41d028):** the unit tests above, as `keys_are_class0_and_drain_pops_them`,
  `shared_pages_and_a_two_page_eviction` (two holders; one `make_room` takes a leaf, then its parent),
  `a_restore_down_classes`, `a_restores_holds_beat_class0_eviction` (Fable's case) and a seeded `fuzz` (64 seeds, 3
  sequences on 4 conversations sharing prefixes, 7 pages; `check` and keys == the rows' pages after every op). The
  serve tests (a)–(c) below are one test, `serve_cuda_cache_ram_idle_flush`.
- **Serve, #170 and the new loop (shady's M3).** cfg(test) knobs on `tier::Faults` (as built: the piece size, a park
  after piece k, the loop's events, and a starved arena):
  - `idle_bytes`: one page a piece, so a few pages of backlog are many pieces;
  - a barrier that parks the engine after piece k until the test releases it. Event order is checked, not timing.

  The tests:
  - (a) **#170 itself**: a request R sent while the engine is parked after piece 1 of a multi-piece backlog is
    admitted after at most one more piece (the event log: `piece 1, [piece 2], admit R`). As built the loop looks
    between pieces, so it is none: `Job, Piece(1), Job`, then the rest of the backlog a page a piece.
  - (b) **the off path inside the idle loop**: a `WriteBehind` fault armed for piece 2's copy. As built, R comes
    after the off line, not between pieces 1 and 2: a request arriving there is taken before piece 2 runs (a), and
    its own steps' write-behind would take the fault instead. R's reply == the `--cache-ram 0` run's; the off line
    says write-behind; the gauges are 0; the counters stay at piece 1's; no piece runs after the fault.
  - (c) **no progress blocks**: a knob that starves the arena (`take` → None). With a backlog, `Faults::idles` still
    advances (the loop blocks), and flush attempts stay ≤ 1 per step. Then the server stops with the backlog pending
    and returns within the test's timeout.
- **Serve, bits:** `--cache-ram` vs `--cache-ram 0` replies are bit-identical through a reclaim + restore cycle (#166's
  `serve_cuda_cache_ram` / `serve_cpu_cache_ram` pattern), on the copy path of §2.8. SHOULD (the resume cell is the
  gate): a resume under pressure restoring **all** of A's pages, the bella cell's shape at 0.5B scale.

## 4. Failure modes, and #181's tests

**#181's tests under the new rules (shady's M1).** revision 0 said the three pass unchanged. That is wrong for one case:

- `serve_cuda_cache_ram_off`'s first pool fault (`Op::Spill`, turn 1, n 1, "B's spill after it ran alone") fires in
  `end_solo`'s `spill(solo, usize::MAX)`, which §2.3 deletes. With the 64 MiB tier, the idle flush then copies B's
  pages as `Op::WriteBehind`, and no later spill in the test has an unstored page left to copy. Turn 1 gets no off
  line and `hit` is not `Spill`, so the test fails as written, for both the paged and the plain pool.
- **Retarget (checkpoint 2, as built):** the `Spill` case runs with a 2-page tier (`TierFault::pages`), which A's
  first turn fills through the idle flush. With no free room there is no write-behind and no idle flush, so B's and
  C's pages stay unstored. Turn 2 (C, alone) takes A's slot, the least recently used: its spill of A's third page
  finds no room, because `room` holds the parent and A's first page isn't a leaf. Turn 3 (A's turn 2, alone) takes
  B's slot. `lend_solo` spills before it looks anything up, so A's pages (class 1 since turn 2's truncate) aren't
  held, and B's first page evicts A's leaf. The fault is armed for that copy: `Op::Spill`, turn 3, n 1. The other
  pool faults (`Restore` turn 3, `WriteBehind` and `Restore` turn 6) don't depend on `end_solo` and stay.
- `serve_cuda_cache_ram_off_in_reclaim` and the CPU case pass under the new rules (shady walked steps 1–5). V is
  unstored because the tier is full, not because leaves are held, and Q's leaf still goes. Their comments are
  updated to say why.

| failure | behaviour | covered by |
|---|---|---|
| a copy fails (spill, write-behind, idle flush, restore) | tier off; the server serves on as with `--cache-ram 0` (`Batch::tier_failed`, `tier::off`): unchanged | #181's tests (one case retargeted, above); §3 (b) for the idle loop; a spill fault after an eager pop (the re-validation path) |
| a restore whose `make_room` → `reclaim` → spill fails | tier off mid-restore; the slot keeps `have` (#181's fix) | `serve_cuda_cache_ram_off_in_reclaim` |
| a restore fault after a spill evicted class 0 under the restoring slot's chain | as above; the hard holds keep the looked-up pages | §3 tier unit test (Fable) |
| the idle flush can't progress (arena short, key collision) | `stalled`: the loop blocks; no spin | §3 (c) |
| stop or a closed channel during a backlog | the loop returns; no flush on the way out | §3 (c) |
| `keys[s]` names an evicted page | impossible after the eager pop (`debug_assert`); release builds truncate to the stored prefix in `write` | tier fuzz, unit test |
| residency count wrong | eviction order only, never bytes (§3); the debug invariants fail the unit tiers | tier fuzz |
| `keys[s]` ≠ `held[s]`'s chain | the one that could touch bytes; none known | §2.2's Batch-level debug check |
| the tier fills with class 1 | spills evict class-1 LRU (today's behaviour); write-behind stops (free room only, checked before copying) | kvstore test |
| interior duplicates pile up | bounded by resident pages with RAM-only extensions; visible as `axon_kv_ram_gpu_pages` | board scrape |
| spill burst on the admit path | in steady state ≈ the victim's chain (§2.3), at the copy rate; counted in `spilled_pages_total` and the trace line | §5.3, ITL clauses |
| pinned RAM | unchanged from #166 (the allocator thread, `auto`'s ½-available / 8 GiB cap, the cgroup-aware check) | #166's tests |
| restore larger than the pool | `make_room` short; a partial restore of `to` pages (unchanged) | #166's tests |

## 5. Bella arithmetic, before building

### 5.1 Inputs (MEASURED, 1031-063509-f763's bandwidth phase, 7B shapes, bella, 2026-10-31)

Revision 2: **f763 caught bella's slow 2D mode** (§2.8.1). The 2D rows below hold in that state only. In the other
state, seven of ten runs, a 2D page copy runs at the contiguous rate. The table after this section has all ten runs.

| copy | rate | per 64-position page (3.50 MiB) |
|---|---|---|
| contiguous D2H / H2D, one 1.37 GiB copy | 28.62 / 28.67 GB/s | 0.13 ms |
| 2D, one per 64-position page (the tier's `page_out`/`page_in`) | 7.14 / 6.95 GB/s | **0.514 / 0.528 ms** |
| one page alone (latency-bound) | 3.60 / 3.48 GB/s | 1.02 / 1.05 ms |

The per-page values are derived: 205.66 / 400 and 211.19 / 400 pages. f763 as a job was labelled TAINTED (one Xorg
sample at 99 % at its start, where the bandwidth phase runs; the rest ≤ 14 %). The 2D rate is the same 7.1 GB/s in all
six 2D rows, against a steady 28.6 GB/s contiguous, which a transient spike doesn't explain. Even so, checkpoint 1's
clean `--timing` job is the number of record. In f763's state a 2D copy ran at ≈ 7.1 GB/s
however wide its rows were; the whole slot as *one* 2D copy measured 7.15 GB/s too. The 4090 pod did 26.0 GB/s per 2D
page (#166's 3f0e). The plan's "414 pages × 0.19 ms ≈ 80 ms" is the 4090's rate. **In that state a 25k-token spill or
restore takes ≈ 213–219 ms on the 2D path. In the fast state it takes ≈ 54 ms** (414 × 3.5 MiB at 28.07 GB/s, 7732).

Every bella run of these rows (MEASURED; GB/s; foreign GPU = the job's samples after its first, which is 96–99 % in
every run):

| job | start | binary | contiguous D2H | one 2D copy | 2D per page D2H / H2D | foreign GPU after the first sample |
|---|---|---|---|---|---|---|
| 1016-173320-f60d | 10-16 17:53 | | 28.39 | 28.39 | 28.11 / 27.90 | |
| 1027-034307-2c0a | 10-27 09:49 | d5a2899 | 28.60 | 28.60 | 28.23 / 28.24 | 4–6 % (kitty, Xorg) |
| 1027-160248-e437 | 10-27 18:32 | 92cb99a | 28.64 | 28.64 | 28.34 / 28.25 | 2–4 % (Xorg) |
| 1027-185035-5c99 | 10-27 19:01 | 92cb99a | 28.64 | 28.64 | 28.35 / 28.25 | 1–5 % (kitty) |
| 1030-115644-7f15 | 10-30 12:15 | 92cb99a | 27.89 | 27.95 | 27.62 / 26.94 | 7–9 % (zen, kitty, Xorg) |
| **1030-195012-05d7** | 10-30 21:53 | 92cb99a | 28.75 | **7.18** | **7.17 / 6.97** | **0.0 %** |
| 1031-015256-ef2b | 10-31 02:25 | 92cb99a | 28.01 | 27.80 | 27.69 / 27.11 | 9–31 % (zen, Xorg, kitty) |
| **1031-045801-3742** | 10-31 05:12 | 92cb99a | 28.78 | **7.19** | **7.18 / 7.08** | **0–1 %** |
| **1031-063509-f763** | 10-31 06:37 | 92cb99a | 28.62 | **7.15** | **7.14 / 6.95** | **0–1.5 %** |
| 1104-164604-7732 | 11-05 11:11 | 28da662 | 28.63 | 28.62 | 28.07 / 28.27 | (2 samples: Xorg 66 %, zen 33 %) |

**#48's "67 ms per 32k slot" vs f763 (revision 0's Q4) is answered by arithmetic.** A 32,768-position 7B slot is
32,768 × 56 × 1 KiB = 1.88 GB, and 1.88 GB / 67 ms = 28.0 GB/s: the contiguous rate. Both numbers hold; they measure
different copy shapes. Checkpoint 1 measures the 32k slot as well.

What's still unmeasured is the rate of every shape other than these (§2.8's rows). The 55–65 ms for a 414-page copy
"at the contiguous rate" below is ESTIMATED from the one 1.37 GiB copy. The real shapes are N page copies (1D, or
staged), and checkpoint 1 measures them.

### 5.2 The resume cell (ESTIMATED)

`kv_resume_bench.py` @440fe18 at bella's defaults: A is 26,481 prompt tokens, its chain 414 pages (26,496 positions);
the resume prompt is 26,692 tokens; the pool 1,744 pages; 2,066 pressure pages; the tier 1,755 pages. This cell is the
warm-up regime of §2.3: the tier has free room while A and the first pressure sessions are written.

- **Capacity.** A is reclaimed first (it is the LRU idle slot), so it becomes the oldest class 1. Everything ever
  reclaimed, A included, is 2,066 + 414 − 1,744 = 736 pages, so class 1 ≤ 736. For a spill to evict class 1 the tier
  must be full with no class 0 left. Class 1 ≤ 736 means class 0 ≥ 1,755 − 736 = 1,019 whenever the tier is full, so
  there is always class 0 to evict first. **All 414 of A's pages survive** (inclusive: 103).
- **The resume's own reclaim** needs 414 + 4 pages from a pool full of idle pressure sessions. The victims are the
  LRU pressure slots, the earliest ones. They were written behind while the tier still had free room (1,755 − 414 =
  1,341 pages of it). Newest-first class-0 eviction drops later sessions' copies first, so the victims' copies are
  likely still stored and their spill is metadata only. ESTIMATED 0–0.2 s of burst on the 2D path, most likely 0.
- **TTFT-resume** = restore + the 196-token tail + one decode step + the client:
  - restore: 414 × 0.528 ms = **219 ms** on the 2D path;
  - tail, prefilled at depth 26.5k: 64–98 ms. The low end is f763's base average, 26,692 / 8.68 s = 3,075 tok/s
    (MEASURED). The high end is 2,000 tok/s, below bella's 32k prefill *average* of 2,682 tok/s (r9,
    `docs/research/r10-scope.md:32`), because the marginal token at 26.5k costs more than the average;
  - one decode step: ≈ 13 ms (f763's itl p50);
  - the client: 5–10 ms.

  Sum: **≈ 0.30–0.34 s on the 2D path**. At the contiguous rate the restore is 414 × 3.5 MiB / 28.6 GB/s = 53 ms,
  55–65 ms with per-page overheads, so ≈ 0.14–0.19 s.
- A consistency check, not a calibration: on the 4090 the same restore is 414 × 0.19 ms = 79 ms, and #166 MEASURED a
  0.149 s resume there (plan §1). That leaves ≈ 70 ms for the tail and the step on a faster card, consistent with the
  figures above for bella.
- **So in the slow 2D mode the ≤ 0.3 s clause fails on bella's copy floor, whatever the policy does.** Hence §2.8.
  In the fast mode (7732) the 2D path *is* the contiguous rate, so ≈ 0.14–0.19 s. Which mode the resume cell's
  restore sees is §2.8.1's open question.

### 5.3 The replay clauses (ESTIMATED; the mechanism is the admission step, Fable's H2)

- **f16 s4 re-prefill ≤ 5 %.** The ideal at board pool + exclusive 6 GiB is 0.15 % (§1). Main runs 5.4× its ideal at
  s4 (47.5 / 8.83 %). The 4090's 20G run measured 73.8 → 0.1 % (1027-155046-1ed0). ESTIMATED **0.2–1 %**. It fails
  only if slot granularity costs ≥ 30× the ideal on top of the tier.
- **s4 ITL p99 ≤ 1.25× main** (main 137 ms MEASURED on the board → ≤ 171 ms). Shady's M4 and Fable's H2, redone with
  arithmetic:
  1. **ITL samples per run ≈ 19.7k.** The trace fixes the output: main's six f16 runs emitted 19,925–19,928 tokens
     each (`output_tok_s × wall_s` in `board-f16-1027-050316/run-axon-f16-s{1,4}-r*.json`, MEASURED). So a run has
     ≈ 19,925 − 174 ≈ 19.75k ITL samples for any engine; a faster engine shortens the wall instead. That corrects
     Fable's ≈ 60k.
  2. **Admissions with copies: up to 174.** At s4 that is 13 new, 5 resume and 156 turns; under pressure most turns
     find their slot reclaimed (main re-prefilled 47.5 % of prompt tokens). Each admission's step runs, on one stream:
     the victim's spill D2H, the restore H2D, the first prefill chunk, and every decoder's row (`admit`, engine.rs
     995–1008, then `step`).
  3. **Stalled decoders per admission step: 4–8** (ESTIMATED; Fable 6–8). That gives 174 × 4–8 ≈ 700–1,400 affected
     samples, **3.5–7 % of 19.75k**, ≫ 1 %. The p99 therefore sits *on* admission steps, around their 80–86th
     percentile. The 4090's 1ed0 differs: there `--parallel 3` kept few decoders per admission (Fable: ≈ 0.6 % of
     samples) and the p99 fell, 68.2 → 63.8 ms (MEASURED, docs/log/2026-10-27-kv-ram-tier.md).
  4. **Copies per admission** (§2.3): a restore of the session's chain, median 269 pages, less what the slot still
     holds; and a steady-state spill of the victim's chain, ≈ 269 pages.

  The verdicts per copy path, with the chunk at 80–137 ms (main's p99 is a chunked re-prefill step):

  | copy path | spill + restore | admission step | vs ≤ 171 ms |
  |---|---|---|---|
  | 2D, slow mode (f763) | 269 × 0.514 + 269 × 0.528 = 138 + 142 ms | ≈ 360–420 ms | **FAIL** |
  | contiguous rate (0.13–0.15 ms/page; 2D in the fast mode, 7732) | 35–40 + 35–40 ms | ≈ 150–215 ms | **marginal, a coin flip** |

  On the contiguous path the clause passes if victims' copies often survive (newest-first helps, Q2) or if restores
  are shorter than chains. If it fails and the trace shows admission-step stalls (`spilled`/`restored` on the steps
  over 171 ms), the rule's else branch applies. The next step is then WP2b, async copies: a second stream, the
  restored request parked a step, and the victim's pages released at the copy's event. A policy change is not the
  next step (§8).
- **s1 ITL p99 ≤ 1.05× main** (main 119 ms → ≤ 125 ms).
  - At s1 the pool holds more of the working set: main re-prefilled 10.3 %, 1.17× its ideal.
  - The restores replace main's re-prefills of ≈ 10.3 % × 2.92M ≈ 301k tokens, ≈ 4.7k pages in all. They fall on an
    ESTIMATED 20–60 admission steps with copies (the 34 resumes, plus the turns whose slot was reclaimed), with 1–3
    decoders each: ≈ 20–180 samples, **0.1–0.9 % of 19.75k**.
  - That is under 1 %, so the p99 stays on main's step mix. ESTIMATED to pass, with the 2D path's top end close to
    1 %. The trace attributes it.
- **replay md5 == main**: by §3.

**Expectations** (Fable §4, falsifiable, ESTIMATED):

| measure | expected |
|---|---|
| re-prefill | ≤ 1 % |
| TTFT-resume | 0.14–0.19 s at a contiguous-rate copy (2D in the fast mode, or a fast path), 0.30–0.34 s at 7.1 GB/s |
| s4 ITL p99 | 150–215 ms at a contiguous-rate copy, > 300 ms at 7.1 GB/s |
| s1 ITL p99 | within 1.05× main |
| md5 | == main |
| `axon_kv_ram_gpu_pages` | near 0 in steady state |

## 6. The gate (pre-registered; verbatim from the plan, §3 row 2, kill rule)

> ≤ 0.3 s *and* re-prefill ≤ 5 % *and* s1 ITL p99 ≤ 1.05× main *and* s4 ITL p99 ≤ 1.25× main *and* replay md5 ==
> main; else `--cache-ram` stays opt-in and the fallback is "no write-behind, spill at eviction, move on restore"
> (simpler, bursty: 414 pages × 0.19 ms ≈ 80 ms per 25k reclaim). Pass → default `auto` (≤ 8 GiB) **for the paged
> GPU pool only**

The rule stays word for word, including "414 pages × 0.19 ms", now known to be the 4090's number. Its bar does not
move because a burst was priced wrong.

The plan's metric column, as expectations, read but not gated: "rule B's bella cell with pressure
(`kv_resume_bench.py` @440fe18): TTFT-resume p50 25k, 2–10 min **8.3 s → ≤ 0.3 s**; agent-32k **f16 s4 re-prefill
47.5 % → ≤ 5 %, TTFT-turn 17 s → ≤ 0.5 s, goodput 5.5 → ≥ 20 tok/s**". The column is quoted as written. Its 8.3 s is
f763's base round 0; the medians are 8.68 → 7.05 s (§0).

**Only the order of measurement changes** (DECISION, Fable §4):
1. **Checkpoint 1's bandwidth rows** on bella (§2.8), ≈ 5 min, before any policy code. They decide the copy path.
2. **`AXON_SERVE_TRACE` step durations in the s4 replay**, read alongside `itl_p99_ms`. The count of steps over
   171 ms, and `spilled`/`restored` per admission (§2.7), attribute a miss to the admission step (H2) or to something
   else.
3. **The resume cell last**, as the rule says.

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
  GPU, cool gate), desktop idle. main = the merge base at gate time, built with the same recipe (`axon-rustc-build`).
  Fresh server per run, interleaved, one warm-up request per server (not measured), with `/metrics` and `/slots`
  scraped after each run. **TAINTED = rerun**: a run whose foreign GPU load trips `agent_replay.py`'s rule (foreign SM
  mean > 5 % or max > 25 %) is never read. That round is rerun for both binaries. The plan's budget is 1.5× the
  nominal time.
- **Checkpoint 1** (the copy shapes). `kvio::tests::bandwidth` (ignored test), `axon-run --timing bella`, at
  `AXON_BW_CTX=25600` (f763's slot: 400 pages) and 32768 (#48's), medians of 7 runs per row (31 for the one-page
  rows). A TAINTED job is rerun as `-r2`.
- **Resume cell.** The f763 script (`timing-bella2.sh`) with the candidate's flag:
  `kv_resume_bench-440fe18.py --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --trace --bin base=MAIN
  --bin cand=CAND:--cache-ram,6G --rounds 3 --gap 120`. It runs at the script's defaults (`--tokens 25000 --others 3
  --new 157 --ctx 32768 --itl-streams 4`) and bella's server defaults (`--parallel 16`, a pool sized from free VRAM:
  1,570–1,744 pages, less a staging buffer's 19 if §2.8 needs one). It is preceded by `kvio::tests::bandwidth`
  (AXON_BW_CTX=25600), whose rates go in the PR. A round whose session wasn't evicted is VOID (the script exits 3) and
  is rerun, never counted. INFO: restored tokens, `axon_kv_ram_*`, the trace's inter-step gaps. The exclusive tier has
  no time-based eviction, so the gap (2–10 min in the plan) changes nothing; 120 s is the cell of record. ≈ 20 min
  (f763: 1,175 s).
- **Replay.** `scripts/agent_replay.py run` (main's copy at gate time; its sha goes in the PR), against servers the
  gate script starts, as #166's `replay-pressure.sh` did (a scratch script; the gate's copy goes on the PR).
  - Trace: `board-trace-v1.json` (sha256 713285ba54487e17…, never committed), with `--window-s 600 --scale 0.15
    --clip 32768` (the board's).
  - Server: the board's axon cell, `serve Qwen2.5-7B-Instruct-Q4_K_M.gguf --device cuda --ctx 32768 --parallel 16
    --kv f16`, plus `--cache-ram 6G` for the candidate. `AXON_SERVE_TRACE` is on for both binaries at s4 (§6, step 2).
  - Runs: speeds 1 and 4, 3 rounds each, order B C / C B / B C. ≈ 10 min a run (board wall 597–604 s, MEASURED), so
    12 runs ≈ 2 h 10 min with server starts.
- **Total bella time per gate set**: ≈ 2.5 h nominal, ≈ 3.75 h with the taint budget. On top come the unit tiers
  (393 s release, 414 s debug, MEASURED at #181's 1104-152814-7b76 / -d474) and md5 vs main (168 s at #166's
  1031-070656-f427).
- **RAM on bella.** 30.7 GiB MemTotal, shared with Haseeb's desktop. Jobs run in `app-axonjobs.slice` with a
  MemoryMax per claim. The gates pin **≤ 6 GiB** (`--cache-ram 6G`). Claim ≈ 16 GB per bench job (the 7B mmap 4.4 GB,
  6 GiB pinned, the process and the Python client) and ≥ 18 GB for the debug unit tier (#166/#181's OOM history). The
  plan's default flip (`auto`, ≤ 8 GiB) needs Haseeb's answer to the plan's Q2 ("May `axon serve` pin 8 GiB of
  bella's RAM permanently...?") and is not part of this PR's gate.

## 8. If the gate fails

The rule says `--cache-ram` stays opt-in. **It stays opt-in, and the fallback is not run** (Fable's H1). In steady
state the fallback ("no write-behind, spill at eviction, move on restore") behaves like this design: the write-behind
copies nothing once the tier is full, and every reclaim spills. It would differ only during warm-up, so its own A/B
would spend a gate set to learn that. If a fallback is ever built: `STEP_BYTES = IDLE_BYTES = 0` is *not* "no
write-behind" in today's code, because `write_some` copies `div_ceil(..).max(1)` = one page for 0 bytes (tier.rs:753).
It needs a `bytes == 0` guard.

What a failure points to:
- **Only the s4 ITL clause fails, with admission-step stalls in the trace**: WP2b, async copies (§5.3).
- **Only ≤ 0.3 s fails**: the copy path's MEASURED rate (§2.8) and the trace's restore time go in the PR, with the gap
  they explain.
- **The capacity clause fails** (re-prefill > 5 %): the sim's ranking was wrong for the real engine. `/slots` and the
  `axon_kv_ram_*` scrape say where the pages went.

The rule is not re-read after the fact.

## 9. Effort estimate (Fable §5's, agreed)

| step | content | ESTIMATED |
|---|---|---|
| checkpoint 1 | the copy-shape rows on bella (§2.8); `PageStore` classes, `has_room`, `insert_behind`, evicted keys; the kvstore tests and fuzz | 0.5 day |
| copy path (Q1) | its own PR: a loop in `Rows::page` (hours), or the staged gather (driver copies or a kernel: a day); bandwidth rows, md5, its gates | 0.5–1 day |
| checkpoint 2 | `Tier` residency, eager pop, `write` re-validation, reclaim-ordered write-behind, `restore` down-class, `end_solo`, #170's loop with the liveness rule, metrics and trace; the Tier fuzz, tier and serve tests; #181's tests green (one case retargeted) | 1 day |
| gates | unit tiers rel + dbg, md5, the resume cell, the replay (§7): ≈ 4–5 h of bella | 1 day elapsed |
| review | 2 rounds (shady), log entry, PR body | 0.5 day |

≈ 3 builder-days, 3.5–4 days elapsed (the plan priced 1.5 rounds: the policy alone at the 4090's copy rate, without
the copy path or the fuzz and serve tests that soft holds need). The engine-side diff is small: kvstore ≈ 140 lines,
tier ≈ 170, engine/serve ≈ 80, ops ≈ 20. The tests are most of it (≈ 600 lines).

## 10. Decisions (the foreman's, final)

1. **Q1, the copy path: in scope**, as its own gated commit and PR that lands first. Checkpoint 1 measures first,
   including the 56 × 1D per page row and the scattered 3.5 MiB page shape. If 1D per page is ≥ 20 GB/s, the fix is a
   loop in `Rows::page` and there is no gather kernel (§2.8's rule). *Revision 2 proposes (PROVISIONAL, wants the
   foreman's yes): the loop is out; no copy-path PR until the `-r2` rerun measures bella's slow 2D mode, idle and
   busy; the policy is built meanwhile (§2.8.1).*
2. **Q2, class-0 order: newest first** (§2.1).
3. **Q3, pinned size:** the gates run at 6 GiB. `auto` (≤ 8 GiB) waits for Haseeb's answer to the plan's Q2.
4. **Q4, #48 vs f763:** answered by arithmetic (§5.1); both hold. Checkpoint 1 measures the 32k slot too.
5. **Q5, the one cache** (CPU, `--parallel 1`): stays opt-in. The default flip is for the paged GPU pool only.
6. **The gate stays verbatim.** Only the order of measurement changes (§6): the bandwidth rows, then the s4 trace's
   step durations with spilled pages per admission, then the resume cell.

## 11. Revision 1: what changed since ac29b61, by review point

| point | where |
|---|---|
| shady M1: #181's `Op::Spill` turn-1 case loses its target | §4 (retarget), §2.3 |
| shady M2: the idle flush can spin; stop and a closed channel | §2.5 (liveness, the loop's three exits), §3 (c), §4 |
| shady M3: test #170, a fault in the idle loop, no-progress, the Tier fuzz | §3 (serve (a)–(c), the Tier fuzz) |
| shady M4: §5.3 credited the write-behind | §2.3 (steady state), §5.3 (redone with arithmetic) |
| shady numbers: 8.68 → 7.05 medians; the sim's pool (1,608–1,609, re-run); class 1 ≤ 736; one resume range; 2,682 tok/s's source; Q4 by arithmetic; `max(1)` in §8; 19 pages | §0, §1, §5.1, §5.2, §2.5, §8 |
| shady SHOULDs: `has_room` before `page_out`; order-free eager pop; the Batch-level `keys == chain(held)` check; `keys_hold_the_leaf`'s replacement | §2.1, §2.2, §2.3, §3 |
| Fable H1: steady state is the fallback; no fallback A/B | §0, §2.3, §8 |
| Fable H2: s4 ITL p99 = the admission step; WP2b | §5.3 (with the ITL sample count corrected to ≈ 19.75k from the board's JSONs), §8 |
| Fable H3: class 0 ⇔ bytes in the GPU pool; C3 at reclaim; `gpu` = device holders; tier.rs:800's comment | §2.1, §2.4, §2.6 |
| Fable: `backlog()` and `Faults::idle` defined; the Restore-after-own-chain-eviction case | §2.5, §3, §4 |
| Fable: `spilled`/`restored` per admission on the trace line; staging-buffer accounting | §2.7, §2.8, §7 |
| Foreman: Q1 in scope (own PR, first), Q2 newest-first, Q5 opt-in, gate verbatim, measurement order | §10, §2.8, §6 |

## 12. Revision 2 (2026-11-05): checkpoint 1's first run, and main merged

| what | where |
|---|---|
| 1104-164604-7732's rows (TAINTED), read against the taint: the loop rejected, the staged path below today's 2D | §2.8.1 |
| Bella's 2D copy is bimodal (28 vs 7.1 GB/s, same binary, machine state); all ten runs tabulated; f763's numbers hold for the slow mode only | §2.8.1, §5.1 |
| PROVISIONAL: no copy-path PR now, `Rows::page` stays, the policy is built; the `-r2` rerun (1105-112352-139c) decides by (a)/(b)/(c) | §2.8.1, §10.1 |
| The bench's GPU-state lines and busy pass (8c7cf14, test only) | §2.8.1 |
| The 32k slot: 1104-164605-53d3 (28da662, never ran) replaced by 1105-112353-0d10 (8c7cf14) | §2.8.1 |
| main c54a730 merged (84861fc): only `mod prefix;` in src/serve; #164 C1 + C2's `Evicted` matches §2.6 | header, §2.6 |
