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
bookkeeping: a refs store per page `ensure` takes, a decrement and compare per page `release` gives back, and the
`warm` / `excl` counts every `BatchLm::pages` call now computes (a pass over the page map, on admission and
planning, not per token). The timing A/B below measures it.

## How it was verified
(pending: gates at the head below)

## Results
(pending)

## What didn't work
- CI on 89f977f failed on the release build's dead-code lint. The local clippy ran at opt-level 0 only, where
  `debug_assertions` is on, so it never saw the release configuration. The scratch clippy script now runs both
  (`-C debug-assertions=off` for lib, tests and bin) and reproduced CI's error without the attribute.
