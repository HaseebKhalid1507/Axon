# PR C: shared-prefix KV pages: design note (checkpoint 0)

- Author: gojo (sharekv track), for Jawz (foreman). Date: 2026-10-27. Status: **design only, no engine code, no gate spend.**
- Base read: origin/main `0c23a87` (#155). Spec: [docs/design/agent-workload.md](../docs/design/agent-workload.md) §0, §2.1, §2.2, §6 row C (#158).
- Branch `feat/shared-prefix-pages`, worktree `~/.cache/axon-wt-sharekv`, scratch `~/.cache/axon-sharekv/`.
  (The repo keeps designs in `docs/design/`; this file moves there when the PR opens.)
- Labels: **CODE** = read on main at file:line; **MEASURED** = someone's run with a log; **ESTIMATED** = arithmetic shown.
  Nothing in this note was run.

---

## 0. TL;DR

1. **Exactness holds on every pooled path on main**, with one extra rule the spec leaves out: **an indexed page or a
   page with refs > 1 is immutable.** The spec's condition 3 ("a slot writes only at positions ≥ its fed length") is not
   enough, because main *shrinks* `fed`: `admit` truncates to `reuse` and keeps the page holding it
   (engine.rs `admit`: `held.truncate(reuse)` + `release(slot, reuse)`), and the S1 path does the same
   (serve.rs `job`: `self.hist.truncate(reuse)`). A re-admitted slot then rewrites positions inside a page it kept,
   which may be shared by then. §3.4 fixes it at claim time.
2. **Only the GPU pool exists to share in**: `BatchLm for Backend` is CUDA-only (lm.rs:1058, "batching many requests is
   GPU-only"). There is no CPU pool and no paged CPU attention. MoE models (qwen3moe, gpt-oss, **deepseek2/MLA**) get
   no pool (`moe_parallel_refusal`, serve.rs:1621; `is_moe`, model/moe.rs:122), and mla.rs asserts `c.tab.is_none()`.
   A split gets no pool either (`pool_with`: `first > 0`). **The pooled archs are qwen2, qwen3, llama and gemma3
   (gemma3 f16 only).**
3. **q8 groups never cross a position, so they never cross a page.** `kv_q8_store` quantizes groups of `KG` along the
   head dimension of *one* row; the scale index is `((row)·kv_dim + kvh·HD)/KG` (decode.cu:1309, prefill.cu `att_load_q8`).
   `ksh` is the model's `attn_k.bias` (qwen2.rs:1576), a constant.
4. **B and C are not independent in code.** They share the engine's admit and evict path (`Batch::admit`, `admit_check`,
   `make_room`, `reclaim`, `idle_pages`, `lend_solo`) and both extend `BatchLm` / `PoolPages` in lm.rs. Their data
   structures are independent: PageMap for C, PageStore for B. §6 proposes the seam.
5. **The §6 row C TTFT clause is ESTIMATED unreachable for most shared spans, even by a perfect implementation.** The
   leader's own prefill sits on every sibling's critical path, so siblings 2–8 TTFT p50 ≤ 0.25× main needs a shared
   span S ≳ 18× the delta, about 19k tokens at a 1,024-token delta (§8.2). Measuring the *increment over the leader's
   TTFT* gives the intended rule, with a threshold near S ≥ 3×d. The rule is pre-registered, so it should be fixed
   **before A fixes the cell**.
6. **From an idle server the fan-out burst never reaches in-flight dedupe.** Sibling 1 runs the S1 path alone
   (fifo: `!b.idle() || waiting.len() > 1` at serve.rs:1436, sjf: serve.rs:1470) and is handed over only at decode (`Handover::due` sits in the
   token callback). Siblings 2–8 therefore arrive after its whole prompt exists and hit the index. Dedupe fires only
   when siblings are admitted while the batch is running. The bench needs both variants (§8.1).
7. Estimate: **~6 builder-days (5–7.5)**, against the spec's 4–6 (§9).

---

## 1. What's on main (the parts C changes)

| piece | where (CODE) | what it does today |
|---|---|---|
| `PageMap` | src/cuda/qwen2/pages.rs | host allocator. Every page is in exactly one place: the free list, the retire list, or one slot's list. A slot's list grows at its end (`ensure`) and shrinks from its end (`release`). `check()` asserts "no page twice". `AXON_PAGE_SHUFFLE=seed` pops a seeded random free page (xorshift). |
| pool | src/cuda/qwen2/slots.rs `CudaPool` | one cache whose rows are pages; `tab[s·cap + i]` is the device table; unheld entries point at the **sink** page (`total`). `reserve` → `ensure` + one synchronous `tab.write_at`; `release` → table entries to sink + `flush`. `debug_tab` asserts table == PageMap. |
| page → row | decode.cu:57 `pg_row`, :1466 `att_rb`; prefill.cu:27–50 `rp_tab/rp_row/rp_base` | row = `tab[pos/64]·64 + pos%64`. Every attention variant (tile f16/q8 prefill, `attn_blk_nb` #151, split decode, gemma3 `attn_split_w_256_pg` / `_bw_256_pg`, gpt-oss `oss.cu`) goes through these. |
| reads past the length | decode.cu:1452 `att_load`, :1478 `att_load_q8` | `cp.async` with src-size 0 for rows ≥ `lim`: **those rows are never read**, they're zero-filled in shared memory. Bytes past a reader's `np` cannot enter arithmetic. |
| engine | src/serve/engine.rs `Batch` | `held[s]` = tokens whose rows slot s holds; `best_slot` = longest prefix of a free slot (`counts`: ≥ half of held or > half of the prompt), else LRU; `admit` reuses `common.min(len−1)`; `admit_check` counts `free + idle_pages + held[slot]−keep`; `make_room` → `reclaim` (LRU idle slot drops **everything**); `grow` → `victim` (unfeed youngest prefill, else preempt youngest running, oldest never); `finish` leaves the slot idle *holding* its rows. |
| S1 path | serve.rs `Engine::job`, engine.rs `lend_solo / end_solo / adopt` | one request alone runs `generate` (graphs, drafting) on a view of a slot; `lend_solo` reserves prompt + n_gen pages; hand-over (adopt, copies nothing) only from the token callback, i.e. **after** the prompt is in. |
| kvstore | src/kvstore.rs | `chain(parent, 64 tokens)`, `keys`, `ROOT`; `PageStore<P>` (byte cap, LRU over unheld leaves, `hold/release`, tokens compared on lookup). Not wired (B wires it). |

Page size: `PAGE = ATT_BLK = 64` (pages.rs:18, qwen2.rs:75), `kvsave::PAGE = 64` (kvsave.rs:43), `static_assert(ATT_BLK ==
PG_ROWS)` (decode.cu:1396). The spec's condition 1 matches the code.

---

## 2. Exactness: a page's bytes are a function of (identity, its prefix tokens) only

**Claim.** On a pool, the K/V (and q8 scale) bytes at position t are `F(identity, tokens[0..=t])`. If that holds, a page
written by request A has exactly the bytes request B would have written for the same 64-token block after the same
prefix. Who wrote the page then doesn't matter.

Checked per path (CODE unless marked):

| ingredient | why it is a function of the prefix only | evidence |
|---|---|---|
| per-token ops (norms, matmuls, RoPE, KV store) | fixed reduction order by shape and position, never by batch (the contract) | determinism.md; tests `gpu_prefill_batchings_bit_identical_*`, `pool_steps_match_solo_in_any_order`, `pool_wide_long_steps_match_solo`, `paged_release_refeed_four_families` |
| attention for query t | reads rows of positions [lo(t), t] of its own sequence through the table; the physical page id only selects addresses | `pg_row` / `att_rb`; **MEASURED**: `AXON_PAGE_SHUFFLE` identity runs gave one digest `393d877192f2e4d7` over plain, paged, shuffled, C=1 and C=8 (#64/#71, cited by the spec) |
| rows past the reader's length | never read (zero-fill, src-size 0) | decode.cu:1452, :1478; prefill.cu:480–491, :644–653 (`np_max` = this pass's own last position) |
| batched decode attention `attn_blk_nb` (#151) | exact by construction, S = 1 regime | #151, `pool_steps_*` tests |
| **gemma3 sliding window** | `lo = win_lo(np, win)`: a function of the position; the window is translated per position through the table; no ring buffer and no out-of-window page release on the GPU pool | gemma.cu:103–117, :233–249; `gpu_paged_gemma_window_bits`; the "512-position window across pages" case in `paged_release_refeed_four_families` |
| **q8 KV scales** | groups of KG along the head dim of one row; one scale per (row, kv head, group); no group spans two positions | decode.cu:1309–1340; scale address `((rb + r)·kv_dim + kvh·HD)/KG` (prefill.cu `att_load_q8`) |
| q8 K shift `ksh` | `attn_k.bias`, a model constant | qwen2.rs:1576 |
| RoPE | static tables (incl. YaRN); nothing on main depends on sequence length (no dynamic NTK) | rope.rs |
| **MLA (deepseek2)** | n/a: MoE ⇒ no pool; mla.rs:1189/1321 `assert!(c.tab.is_none())`. If MLA ever gets a pool, its latent rows are also per-position (re-check then). | model/moe.rs:122, serve.rs:1621 |
| gpt-oss windows/sinks, qwen3moe | n/a: MoE ⇒ no pool | same |
| CPU | n/a: no pool, no paged KV; serve runs one request at a time on its own `KvCache` | lm.rs:1058 |
| splits | n/a: `pool_with` refuses `first > 0` | slots.rs `pool_with` |
| per-request options | none touch the forward pass: `Job` carries prompt, n_gen, sampling, stops, think/channels | serve.rs:239 |
| one identity per pool | one process, one model, one `--kv`, one `--rope-scaling` | spec §2.1 condition 2 |

**What sharing must add.** Four conditions, all host-side and all checkable:

- **E1. Key = tokens.** A page is found only by `kvstore::chain` over its whole prefix, with each block's tokens and
  parent key compared on hit. A 64-bit collision gives a miss, never an alias.
- **E2. Written before read.** A page is indexed only once all 64 rows are written by committed tokens: `held.len() ≥
  64(k+1)`, never draft rows. The engine is single-stream and `feed_slot` ends with `gpu.sync()` (slots.rs:381), so
  any later read is ordered after the write. The spec's "only pages completed in earlier steps" is safe but stricter
  than needed: earlier in the *same* step is fine too.
- **E3. Immutable once shared (new, see TL;DR 1).** A position may be written only if its page has
  `refs == 1 && !indexed`. Shared and indexed pages are read-only. Debug builds assert this in `feed_slot`,
  `step_rows` and `lend_solo`, for every page a write range touches.
- **E4. The page holding `len − 1` is private** (spec condition 3): shared pages stop at the whole pages below
  `min(common, len − 1)`. The last prompt token is re-fed as on main.

**A free runtime proof in tests.** If two slots independently write the same block (a sibling prefilled before the
leader indexed it), the index gets a second page for a key it already holds. In test and debug builds, read both pages
back and **assert byte equality**. Every serve test that runs then checks the theorem itself, not only the outputs.

---

## 3. Design

### 3.1 Data structures

**`PageMap` (pages.rs): refcounts and pins.**
```rust
pub struct PageMap {
    cap, pages: Vec<Vec<u32>>, free, retire, total, rng,   // as on main
    refs: Vec<u32>,     // occurrences of the page in slot lists
    pinned: Vec<bool>,  // the prefix index holds it ("warm" when refs == 0)
}
// page state: free | retired | live (refs > 0 || pinned)
fn ensure(slot, hi)          // fresh pages, refs = 1 (as today)
fn map(slot, &[u32])         // append live pages to a slot's list, refs += 1 (list must end page-aligned)
fn release(slot, lo)         // pages past need(lo): refs -= 1; refs == 0 && !pinned -> retire
fn pin(p) / unpin(p)         // unpin with refs == 0 -> retire
fn writable(slot, page_idx) -> bool  // refs == 1 && !pinned
fn check()                   // each page in exactly one state; refs[p] == occurrences; no slot past cap
```
`AXON_PAGE_SHUFFLE` is unchanged: it only picks *which* free page `ensure` pops.

**`PrefixIndex` (new, host-only; proposed home `src/serve/prefix.rs` so the pool stays token-agnostic).** This is
kvstore's design with page ids as payload:
```rust
pub struct PrefixIndex {
    root: Key,                      // kvstore::ROOT today; stage F: ROOT ^ H(tenant_salt)
    map: HashMap<Key, Node>,        // Node { parent: Key, tokens: Box<[u32]>, page: u32, children: u32, holds: u32, last: u64 }
    lru: BTreeSet<(u64, Key)>,      // evictable = children == 0 && holds == 0 (unheld warm leaves)
    tick: u64,
}
fn walk(&mut self, tokens: &[u32]) -> Vec<(Key, u32)>    // verified chain, touched deepest-first
fn insert(&mut self, parent: Key, block: &[u32], page: u32) -> Ins  // New(key) | Have(key, page) | Collision
fn hold(key) / release(key)                                // mirrors slot maps; == PageMap refs (debug-checked)
fn pop_lru_leaf(&mut self) -> Option<Evicted { key, parent, tokens, page }>   // B's spill hook point
fn unindex_subtree(&mut self, key) -> Vec<u32>             // for in-place continuation (§3.4)
```
It reuses `kvstore::{chain, keys, ROOT}` read-only, so **C does not edit kvstore.rs** (B owns it). Size ESTIMATED:
about 1,537 nodes for the 7B f16 bella pool, at roughly 350 B each (256 B of tokens + map overhead), which is
≈ 0.5 MB of host RAM.

**Per-request state** (engine `Seq`/`Prefill`): `keys: Vec<Key>`, the chain of the slot's indexed pages, extended
incrementally so a page boundary costs one `chain` over 64 tokens (µs, ESTIMATED).

### 3.2 `BatchLm` additions (lm.rs; CUDA impl in slots.rs, `Backend` dispatch)

```rust
fn map_pages(&self, pool, slot, pages: &[u32]) -> Result<(), String>;    // PageMap::map + one table write
fn pin_page / unpin_page(&self, pool, page: u32) -> Result<(), String>;
fn slot_pages(&self, pool, slot) -> Vec<u32>;                            // page ids (B needs this too: D2H)
fn copy_page(&self, pool, src: u32, slot, rows: usize) -> Result<(), String>;  // COW: ensure 1 fresh page, D2D rows
```
`PoolPages` gains `warm` (pinned, refs 0) and `excl: Vec<usize>` (per slot, pages with refs == 1). For COW,
`cuda/mod.rs` needs a **device-to-device 2D copy**: `copy_2d` there is host↔device only (mod.rs:1127). A page is
`2·n_layers` strided pieces per region (the layer arrays are contiguous with stride `rows·row_bytes`; see the
`adopt_cache` loop), so `cuMemcpy2DAsync` does it in one call for f16 and two for q8. The spec's "one D2D ≈ 10 µs" holds
only with that 2D copy. Done the `adopt_cache` way it is 56 (f16) / 112 (q8) memcpys for a 7B, ≈ 0.2–0.6 ms
(ESTIMATED at 3–5 µs host cost per call).

### 3.3 Where pages enter the index (pin)

A page is inserted once it is full of committed rows:
- `Batch::feed`, for each page boundary the new `fed` crosses;
- `step_inner`, when `held[s].len() % 64 == 0` after a row (**decode-generated pages are indexed too**: the next turn's
  prompt is prompt + reply + delta, the 97.6 % of §0);
- `adopt` / `end_solo`, for every full page of `hist` not yet indexed (the S1 path indexes nothing while it runs, which
  is fine because nothing else runs).

`Ins::Have(key, other)` with `other != own page` means a duplicate. Phase 1 leaves the own page unindexed and asserts
byte equality in tests (§2). An optional later step *swaps* the slot's entry to `other` and frees its own, which is
exact by §2 and saves memory.

### 3.4 Admit: `claim_prefix(slot, prompt) -> reuse` (used by `admit` **and** `lend_solo`)

```
last   = len − 1
own    = common(held[slot], prompt)            // best_slot's slot and prefix, as today
walk   = index.walk(prompt[..last])            // whole pages with 64(k+1) ≤ last  (E4)
if min(own, last) ≥ 64·|walk|:                 // continue in place (main's path, zero copy)
    reuse = min(own, last)
    p = page holding reuse (if reuse % 64 != 0 and p < held pages):
        refs == 1 && pinned  -> unindex_subtree(p) (its warm descendants free); write in place
        refs  > 1            -> COW (copy rows 0..reuse%64 into a fresh page)  [or round reuse down to 64·p: phase 1]
else:                                          // the index knows more than the slot
    release(slot, 0); map_pages(slot, walk pages); reuse = 64·|walk|; hold each key
fed = reuse; cached = reuse
```
`best_slot` **still matters**, against spec §2.2 ("stops mattering"). The partial tail page (≤ 63 rows) is never in a
whole-page index and lives only in its slot, and continuing in place is main's zero-copy path. It stays the tiebreaker.

**COW vs round-down.** COW saves at most 63 tokens of prefill per admission: ≤ 63/(63 + 1,024) ≈ 6 % of a
1,024-delta sibling's prefill, ≈ 3 % on average (ESTIMATED). It matters most for identical prompts and resumes with no
new tokens. Proposal: phase 1 rounds down, which is exact and needs no new copy primitive; COW comes in a separate
piece (C2b) behind the same gates and is dropped if the cell passes without it.

### 3.5 In-flight dedupe (prefill once, join many)

- **Cover.** At admission and every step, prefill j's cover is the max over *older* prefills i (`id_i < id_j`) of
  `min(common(prompt_i, prompt_j), last_i, last_j)` rounded down to pages. If the cover exceeds `j.fed` by ≥ 1 page and
  the leader hasn't fed it yet, j is **parked**.
- **Parked.** `plan_charged` gives it 0 tokens. It doesn't count against `lanes` (`admit_gate` sees its `rem` as its
  private part only). `admit_check` counts only its private pages. It holds a slot.
- **Joining.** Before planning each step, a parked j walks the index from `j.fed` and `map_pages` whatever the leader
  has indexed since; `j.fed` and `j.cached` advance. Once the leader's `fed ≥ cover`, j feeds its own remainder
  (partial page + delta).
- **Liveness.** A prefill waits only on an *older* one, so the oldest never waits and there are no cycles. If the leader
  is cancelled, finished, unfed or preempted, its indexed pages **stay**: they are held by joiners or warm. The cover is
  recomputed. The oldest former joiner now has no cover and feeds the rest itself; younger joiners see it as their new
  leader. This is the spec's "first joiner takes over", and it falls out of the rule with no special case.
- The leader preempted by recompute re-admits through `claim_prefix` and hits its own indexed pages, so preemption
  stops throwing away whole prefixes.
- Chunks stay non-page-aligned (`plan_charged` unchanged). A page mid-write is never joined, and the joiner feeds a
  partial page itself. No scheduler-alignment change.

### 3.6 Eviction, accounting, preemption

`make_room(slot, hi)` loops:
1. `reserve`; if short:
2. evict `index.pop_lru_leaf()` (warm, refs 0) → `unpin_page` → retire/free. **B's spill hook goes here**
   (`Evicted{key,parent,tokens,page}` → D2H into PageStore before the page is reused).
3. else `reclaim` the LRU idle slot: `release(s, 0)`. Its pinned pages become warm, its unpinned tail frees. Go to 2.
4. else short (the caller's `victim` path).

Leaf LRU with deepest-first touch consumes a cold session **tail-first**. Main drops the whole LRU idle session; this
keeps its head, so a resume re-prefills only the evicted tail. Shared roots (system prompt, tools) are touched by every
user and stay. This is a **warm/GB lever even without cross-session sharing**.

**Accounting (spec "count only pages the slot holds alone").** A page is freeable for an admission iff, after that
admission maps its own prefix, no busy slot holds it and neither does the request itself. So
`have = free + retired + warm(not on my path) + Σ_idle s≠slot excl[s](not on my path) + own pages past keep`, and
`want = need(last + grow) − mapped`. `others` stays as is (`held` counts every mapped entry). `victim` should prefer
victims with `excl > 0`. Preempting a request whose pages are all shared frees nothing, and the loop moves to the next
one. Progress argument as on main: once every non-oldest request has yielded and all warm pages are evicted, the oldest
holds ≤ cap ≤ total.

**Cancel / finish.** `finish` leaves the slot idle and holding, as today. `reap` → `finish(Gone)` the same. A cancelled
or preempted prefill's `release` decrements refs, and the index's `holds` mirror decrements with it. Leak invariant
(debug, after every step): `Σ refs == Σ |slot lists|`, `index.holds(key) == refs(page)` for indexed pages, and
`free + retired + live == total`.

### 3.7 Kill switch and observability

- `AXON_SHARE_KV=0` gives today's behaviour bit for bit *and* decision for decision: no index, no `map`, and refs are
  always 1. Proposed default: on, once gated. Outputs are identical either way; only scheduling and memory change.
- `/slots`: `pages_warm`, `pages_shared`. `/metrics`: `axon_prompt_shared_tokens_total` (tokens mapped from the index,
  in-flight joins included) and `axon_pages_evicted_total`. `cached_tokens` in usage includes mapped tokens. This gives
  A's re-prefill counter what it needs.

---

## 4. Interaction with B (RAM tier) and A (replay)

**Shared code (flagged; posted to the Axon r10 forum thread):**

| function / file | B touches it for | C touches it for |
|---|---|---|
| engine.rs `Batch::admit`, `lend_solo` | restore on admit (H2D before prefill) | `claim_prefix` (map indexed pages) |
| engine.rs `admit_check`, `idle_pages`, `uncached` | RAM hits lower `want` | refcount-exclusive accounting, mapped pages |
| engine.rs `make_room`, `reclaim` | evict-to-RAM instead of drop | warm-leaf eviction (page-level) |
| lm.rs `BatchLm`, `PoolPages` | page ids for D2H/H2D | `map_pages`, pins, `slot_pages`, `excl/warm` |
| cuda/mod.rs | maybe pinned/async copies | D2D 2D copy (COW only) |
| serve/ops.rs `/slots`, `/metrics` | RAM tier fields | warm/shared fields |
| kvstore.rs | owns it (PageStore with pinned buffers) | **read-only** use of `chain/keys/ROOT` |

**The seam I propose.** C turns eviction page-level and exposes `pop_lru_leaf → Evicted{key, parent, tokens, page}`.
B spills each evicted page D2H into PageStore under the **same key** (the same `chain`, the same `ROOT`). On admit the
chain walk goes GPU index → B's PageStore → prefill. B restores into fresh pages, and C indexes them as if fed. Write-
through (B's s-B plan) dedups by key (`PageStore::insert` returns the stored key for a page already there), so a page
shared by 8 slots is written once. If B lands first with slot-granular spill (`reclaim` → D2H every whole page of the
idle slot), C later moves the hook from `reclaim` to `pop_lru_leaf`, a small rebase. Either order works if we agree on
the `Evicted` shape now. **One of us should add `BatchLm::slot_pages`** (both need it).

**Identity.** The GPU index lives as long as the process does, so `ROOT` stays constant (spec condition 2). B's RAM
tier lives as long as the process too. The disk tier (E) needs `ROOT = H(identity)`. Make `root` a field of
`PrefixIndex` now, so E and F (tenant salt) only change a constructor argument.

**A.** The replay md5 of every reply is the gate of record (§5.3). `axon gen` md5 is necessary but doesn't exercise
the pool at all.

---

## 5. Tests the gates need

### 5.1 Host-only (no GPU, both suites)
- `pagemap_share_fuzz`: random ensure / map / release / pin / unpin / flush, checked against `check()` (refs ==
  occurrences, one state per page, a failed ensure takes nothing), LIFO and shuffled.
- `prefix_index_*`: walk capped below `len − 1`; a forced collision (same key, other tokens) is a miss; LRU never evicts
  an inner or held node; a chain never loses its root; `unindex_subtree` frees warm descendants only.
- `claim_prefix_cases`: reuse mid-page in a pinned refs-1 page (unindex, in place), in a refs > 1 page (round down /
  COW), whole prompt cached (page with `len − 1` private), `len − 1` exactly on a page boundary, index longer than own
  slot, own slot longer than index.
- `join_cover_rule`: waits only on older; leader cancelled → oldest joiner feeds, younger re-target; no cycle in a
  random arrival/cancel fuzz.
- `best_slot` tests unchanged; `legacy_reproduces_main_decisions` with `AXON_SHARE_KV=0`.

### 5.2 GPU, pool-level (slots.rs / paged_tests.rs; `AXON_TEST_GPU_SERIAL=1` on pods)
- `shared_pages_match_solo_four_families`: slot A feeds P; slot B maps A's whole pages and feeds its own delta. B's picks,
  candidates and KV bytes == solo. qwen2 f16 + q8, qwen3, llama, gemma3 with the prefix > 512 so the **window spans
  shared and private pages**. Under `AXON_PAGE_SHUFFLE`.
- `shared_page_immutable`: snapshot a shared page's bytes, run the other slot's feeds and steps, and check the bytes are
  unchanged. A debug `#[should_panic]` covers a feed aimed at a shared page.
- `cow_tail_matches_solo` (only if C2b ships): common ends mid-page; copy, continue, compare with solo; the source is
  untouched.
- `duplicate_pages_byte_equal`: two slots write the same block independently, and the bytes are equal (§2's runtime
  proof).

### 5.3 Serve-level (serve/tests.rs) and md5
- `serve_identity` with `AXON_SHARE_KV=1` vs `0`: every reply identical, greedy and seeded, f16 and q8, with
  `AXON_PAGE_SHUFFLE` set.
- `serve_fanout_join` (busy variant): 2 background decoders, then 8 siblings sharing a 1k-token prefix plus distinct
  deltas within 100 ms. Each reply == alone, and `shared_tokens ≥ 7·⌊S/64⌋·64`.
- `serve_fanout_idle`: the same from idle (S1 leader, hand-over, then index hits).
- `serve_share_cancel_leak`: cancel the leader mid-prefill, then a joiner. The rest == alone. At idle, the leak
  invariant holds, and evicting every warm page gives `free == total`.
- `serve_share_pressure`: `serve_paged_pressure` with sharing on. Preemption with shared pages, no deadlock, no leak,
  preempted counter > 0.
- `serve_share_solo_handover`: the S1 path maps an indexed prefix, then hands over. == alone.
- md5 vs main: the AGENTS.md `gen` sets (CPU, cuda f16, cuda q8; + gemma3), necessary but they don't touch the pool.
  **Gate of record: A's replay md5 of every reply on one trace**, main vs cand (sharing on), cand
  `AXON_SHARE_KV=0`, and cand under `AXON_PAGE_SHUFFLE`. Suites: release, debug-assertions, scalar, clippy, fmt.

---

## 6. Bench cell (§6 row C)

**The cell as pre-registered:** 8 siblings within 100 ms; prompt = S (A's measured p50 shared span) + a **1,024-token
private delta** (A doesn't separate it); 600-token replies; Qwen2.5-7B Q4_K_M, `serve --parallel 16`, bella, f16 (q8
second). Gates: siblings 2–8 TTFT p50 ≤ 0.25× main (or, if S < 4·1,024, "siblings 2–8 prefill no shared-page tokens"
from the counters) **and** warm/GB ≥ 1.4× main on agent-32k at speed 4 **and** md5 == main under `AXON_PAGE_SHUFFLE`
**and** C=1 decode within 2 %. It waits for A's measured S.

### 6.1 Two variants needed (TL;DR 6)
- **idle**: the cell as written. Sibling 1 runs S1, and siblings 2–8 hit the index after hand-over. This measures §3.3–3.4.
- **busy**: the same burst on top of 2–4 decoding sessions, so all 8 are admitted into the running batch. This
  measures §3.5 dedupe.

### 6.2 The 0.25× clause, ESTIMATED
Model: prefill throughput R tokens/s whatever the split; fifo with 2 lanes (`lanes_for`: fifo → 2); T = (S + d)/R,
t = (d + 32)/R (32 = mean partial page).
- main: the leader takes T, then siblings finish in pairs at T + 2T, + 4T, + 6T, + 7T. The p50 of siblings 2–8 (4th of
  7) is ≈ **5T**.
- C: the leader takes T, then deltas in pairs. p50 ≈ **T + 4t**.
- ratio ≈ (T + 4t) / 5T. With d = 1,024: S = 4k → 0.37; S = 10k → 0.28; S = 20k → 0.24. **≤ 0.25 needs S ≳ 19k.**
  Below that, a perfect C fails the clause, because the leader's prefill is a floor both share.
- The **incremental** form, (siblings' TTFT − leader's TTFT) p50, gives ratio ≈ t/T = (d + 32)/(S + d). That is ≤ 0.25
  iff S ≳ 3.1·d, which matches the spec's own "S < 4× delta" fallback threshold. I think that's the intended rule.
  **Recommend re-registering it in A's PR, before the cell is fixed.**
- Scale ambiguity: agent-32k runs the trace at s = 0.15. If S is A's p50 shared span *scaled*, it is ~0.3–3k
  (ESTIMATED from the spec's 2–20k range), which is below 4×d, so the fallback clause applies. A should state whether
  S is scaled.

ESTIMATED prompt tokens computed at S = 10k, d = 1,024: main 8 × 11,024 = 88.2k; C 11,024 + 7 × (1,024 + ≤ 63)
≈ 18.6k (−79 %).

### 6.3 Other cells
agent-32k at speed 1 and 4 (warm/GB, TTFT-resume, re-prefill tokens), and C=1 decode `tg128` main vs cand interleaved.
C adds host work only at page boundaries, so ESTIMATED noise.

---

## 7. What's wrong or loose in the spec against the code

1. **§2.1 condition 3 is incomplete.** `fed` shrinks: `admit` truncates to `reuse` and keeps the page holding it
   (engine.rs `admit`), and so do `Engine::job` (`hist.truncate(reuse)`), `unfeed` and `preempt`. A slot then rewrites
   positions inside a page it kept. If that page is shared or indexed, this corrupts another request. Fix: E3 (indexed
   or refs > 1 ⇒ immutable), enforced in `claim_prefix` (§3.4).
2. **§2.2 "`best_slot` stops mattering (any free slot)".** The partial tail (≤ 63 rows) lives only in its slot, and
   in-place continuation is zero-copy. Keep `best_slot` as the tiebreaker.
3. **§2.1 COW "one D2D ≈ 10 µs".** True only with a D2D 2D copy, which cuda/mod.rs lacks (`copy_2d` is host↔device). A
   page is 2·n_layers strided pieces per region; with `adopt_cache`-style memcpys it's ≈ 0.2–0.6 ms (ESTIMATED).
4. **§2.2 dedupe "pages completed in earlier steps"**: stricter than needed (single stream, `feed_slot` syncs). Not
   wrong. **More important:** from idle the fan-out burst never reaches dedupe (S1 leader, hand-over only at decode;
   serve.rs:1436 / :1470 and `Handover::due` in the token callback). §6 row C needs a busy variant.
5. **§6 row C TTFT clause**: ESTIMATED unreachable for S ≲ 19k at d = 1,024 (§6.2). The incremental form fits the
   spec's own 4× threshold.
6. **§6 kill rule "if shared < 2k, C becomes dedupe-only"**: in this design dedupe *is* refcounts + index (joiners map
   indexed pages), so dedupe-only saves only warm retention (≈ 0.5 builder-day). Warm retention (page-level LRU instead
   of whole-slot `reclaim`) is itself a warm/GB lever independent of cross-session sharing. Suggest the fallback be
   "warm index off" (pages unpin at refs 0), not a different code path.
7. **"B and C are independent in code"**: not at the engine. See §4's table.
8. **§1.4 / §4 `--slot-idle-ms`**: it is a stalled-*client* cancel, off by default (serve.rs:156, "a stream whose
   client stops reading for this long is cancelled"). It has nothing to do with KV retention; prefixes are dropped only
   by `reclaim` under page pressure. §4's "Idle policy: `--slot-idle-ms` stops meaning forget" builds on a wrong reading
   (B/D's concern, flagged for them).
9. **§2.2 "GPU-resident radix index"**: it is a host-resident `HashMap` indexing GPU pages. Wording only.
10. **Dispatch question "attn over pages on CPU"**: there is no CPU pool or paged CPU attention on main. MLA has no pool.
    gemma3 has no q8 on the GPU (qwen2.rs `plan`). So "q8 across a shared boundary" applies to qwen2/qwen3/llama only,
    and is a non-issue there (§2).

Verified correct: PAGE = ATT_BLK = 64 = kvsave::PAGE; `admit` re-feeds `len − 1`; no pool on splits or MoE; 7B page =
3.5 MiB f16 (28 L × 4 KV heads × 128 × 2 × 2 B × 64); `admit_check` / `reclaim` / `idle_pages` count held pages as
freeable.

---

## 8. Open questions (for Jawz)

1. Index home: `src/serve/prefix.rs` (host-only, engine-owned; pool tracks refs and pins) vs pages.rs. I propose the
   engine side.
2. Index decode-generated pages? I propose yes (resume = prompt + reply + delta).
3. COW in PR C or phase-2? I propose round-down first (C3), COW as C2b if the numbers want it.
4. `AXON_SHARE_KV` default on after gates (bits identical either way)?
5. Order vs B: who lands the `make_room` / `admit` seam first? Agree on the `Evicted` shape and on `slot_pages` now.
6. The fan-out cell: add the busy variant, the incremental TTFT form, and a statement of whether S is trace-scaled.
   This goes to joestar (#160) *before* the cell is fixed.
7. Duplicate-page swap (§3.3): later, or never?

---

## 9. Builder-day estimates (ESTIMATED)

| piece | content | days |
|---|---|---|
| C0 | this note | 0.5 (done) |
| C1 | PageMap refs + pins + `check`; `PrefixIndex`; host fuzz/unit tests | 1.0 |
| C2 | `BatchLm`/`Backend` plumbing (`map_pages`, pins, `slot_pages`, `PoolPages`), `debug_tab` with sharing, GPU pool tests (4 families, immutability, duplicate-byte-equality) | 1.0 |
| C2b | COW: D2D 2D copy in cuda/mod.rs + `copy_page` + test (optional) | 0.5 |
| C3 | engine: `claim_prefix` (admit + `lend_solo`), indexing points, accounting, page-level eviction, preempt/unfeed/cancel with refs, `AXON_SHARE_KV`, metrics/slots, serve tests | 1.5 |
| C4 | in-flight dedupe: cover rule, parked joiners in plan/lanes/admit_check, liveness, tests | 1.0 |
| C5 | gates: release + dbg + scalar suites, clippy/fmt, md5 sets, replay md5 ×3 (on/off/shuffle); fix rounds | 0.75 |
| C6 | bench: fan-out idle + busy, agent-32k speed 1/4 (warm/GB), C=1 decode A/B; log entry; PR body | 1.0 |
| **total** | | **≈ 6 (5–7.5)**; without C2b ≈ 5.75 |

The spec says 4–6. The extra is the immutability fix (§7.1), the second bench variant and the leak instrumentation.
Gate queue time is not included.
