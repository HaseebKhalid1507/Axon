# Tests for the KV RAM tier's off path, and a fix  (issue #172)

- Date: 2026-11-04. Branch: test/kvtier-off-path. Base: main f458f48. Status: open (draft)
- Pick: foreman (axon), issue #172. Builder: joestar.

## What changed
Mostly tests, plus one fix. Since #166, a KV RAM tier error (`--cache-ram`) turns the tier off rather than failing
the engine. On the pool path that is `Batch::tier_failed` (from a spill, the write-behind or a restore); on the
one-cache path, `Engine::run` (its write) and `Engine::job` (its restore). /metrics then shows the tier off
(`Metrics::kv_ram_off`: gauges 0, counters kept). No test drove any of it.

- **A test-only fault hook**, `serve::tier::Faults`, held in `Metrics::tier_faults` because that is the one
  handle a test's `Server` and the engine's tier both hold. `fail(op, n)` arms the n-th next copy made for `op`,
  one of `Op::Spill`, `Op::WriteBehind` or `Op::Restore` (the off line's names), and only that op's copies count.
  A copy is a pool's `page_out` / `page_out_id` / `page_in` (one page each), or the one cache's `kv_save` /
  `kv_load` (all the pages one write or restore moves). The copy it hits fails **instead of running**. A failed
  copy in therefore leaves the rows it was to fill as they were (another conversation's rows), so a reply built
  on them would differ from the reference. (A fault after the copy would leave the right rows in place and hide
  that bug.) `hit()` records the op and the slot of the copy it failed. Every line of the hook is
  `#[cfg(test)]`; the product's lines at the copy sites are main's, verbatim.
- **One off line** (`tier::off`, product code): `axon serve: --cache-ram: {what}: {e}; the KV RAM tier is off
  from here`, where `what` is `spill`, `write-behind` or `restore`, on every path. The single-cache path used to
  print `...: {e}; off from here` and the pool path didn't name the operation. Under `cfg(test)` it also
  records the line and the tier's counters at that moment; the tests' stderr check reads that record.
- **Fix** (6acd20f): `Batch::restore` checks the tier again after `make_room` (below).
- **Tests** (`serve/tests.rs`):
  - `serve_cpu_cache_ram_off` (the one cache, CPU) and `serve_cuda_cache_ram_off` (a paged pool, a plain pool
    and `--parallel 1`). They use `cache_ram_case`'s conversations (A, B, C; A alone; B, C; A again, with a page
    of new text, and with D decoding on a pool: a batched admission), then B and A again. Each run fails the
    tier once, mid-conversation:
    - pool: B's spill after it ran alone; A's turn-2 restore after its first page came back (n = 2: a partial
      restore must not be trusted); the batch's write-behind (A's turn 3 admitted while D decodes).
    - one cache: the write after A's turn 3 (its earlier turns came back from RAM); A's turn-2 restore, while
      the cache holds C's rows.

    Each run asserts:
    - every reply, D's included, is bit-identical to the `--cache-ram 0` run's (greedy), with the same prompt
      and completion token counts;
    - the server answers the later requests;
    - exactly one off line, equal to the expected one: the armed op's name, and the fault's message, which
      names the op its copy was made for. A fault can only fire on its op's copies, so the line proves where it
      landed;
    - /metrics: `axon_kv_ram_capacity_bytes`, `_pinned_bytes`, `_bytes` and `_pages` are 0; the counters equal
      what they were when the tier failed (kept, and unchanged since); pages were written before the failure,
      and a hit happened before it for the turn-3 faults.
  - `serve_cuda_cache_ram_off_in_reclaim`: the bug below.

## Why
#172: the off path shipped untested. A tier that fails must leave the server as `--cache-ram 0` would.

## Found and fixed: a tier error in a restore's reclaim exited the server
`Batch::restore` makes room for the stored pages it copies in: `make_room` -> `reclaim` -> `spill(victim)`. If
that spill's copy failed, `tier_failed` dropped the tier, and `restore` then ran `self.tier.as_mut().expect("on")`,
which panicked. `or_exit("an admission")` turned the panic into `exit(70)` for the whole server, the opposite of
#166's contract.

The repro, `serve_cuda_cache_ram_off_in_reclaim`, uses a paged pool of 10 KV pages (3 slots of 640 positions) and
a 5-page tier. Each step is asserted through /slots (pages per slot, pages free, held tokens) and /metrics (the
tier's counters and pages):
1. R (3 whole pages) and Q (2) run alone in slots 0 and 1. Both are stored, which fills the tier.
2. V (2 whole pages) runs alone in slot 2. There is no room (no stored page is evictable), so none of V's pages
   is stored.
3. W (R's first sentences, under a page) runs in slot 0. R's stored pages are now no slot's.
4. D (a counting prompt over 4 pages; prompt plus reply 6 pages) runs alone in slot 1. Q's stored pages are now
   no slot's, and D's reservation takes all the free pages.
5. R's next turn arrives while D decodes. D hands over and keeps the 5 or 6 pages of its rows, whenever the
   hand-over comes. R's turn is admitted to slot 0. Its restore of R's 3 stored pages lacks pages and reclaims V,
   and V's spill has room now (Q's last page is evicted). The armed spill fault fails that copy.

The test then asserts that every reply, cached tokens included, equals the `--cache-ram 0` run's; that the
failed copy was V's (`hit() == (Spill, slot 2)`: an idle slot that isn't the admitted one, so a reclaim's);
that the tier's counters were `[5, 0, 0]` when it failed (nothing restored before it); and that D was still
decoding when R's turn returned (a batched admission, not a request alone).

- **Before the fix** (658ef08, MEASURED on bella, job 1104-144238-8244): every precondition passed, then
  `--cache-ram: spill: ... a copy for a spill; the KV RAM tier is off from here`, the panic `on` at
  engine.rs:794 (the `expect`), `axon serve: the engine failed on an admission: on; exiting`, **exit 70**.
- **The fix** (6acd20f): after `make_room`, `let Some(t) = self.tier.as_mut() else { return Ok(have) }`. The
  slot keeps `have`, as after a failed copy in; the pages `make_room` reserved are the prefill's.
- **Audit** of the other sites after a call that can turn the tier off (spill, write_behind, restore, and
  make_room/reclaim through spill): admit and lend_solo reach the tier again only through restore, which checks
  it first; reclaim, feedable, grow, end_solo, unfeed, preempt and step use no tier afterwards; cached_len and
  poll_tier match on it. The one cache's `Engine::run` and `Engine::job` drop it and don't use it afterwards. No
  other site.

## Zero code in release (MEASURED: product binaries, main f458f48 vs d93920f, same rustc recipe)
- `strings`: neither the fault message nor `a copy for a` is in either binary. `nm -C`: neither binary has a
  `Faults`, `tier_faults`, `tier::Op`, `off_lines` or `Faults::hit`/`fail` symbol.
- `PoolTier::write`, `PoolTier::restore`, `CacheTier::write` and `CacheTier::restore` disassemble (objdump,
  addresses masked) to the **same instructions** as main's: 1213, 465, 1184 and 528 instructions.
  (`write_some` is inlined in both.)
- The only differences come from the `off` refactor and the fix: `tier::off` (new, 242 B), `Batch::spill`
  391 -> 323 B, `Engine::run` 8115 -> 8412 B, and `Batch::restore` 1170 -> 1105 B (the panic path is gone).
  `.text` 4498549 -> 4499093 B (+544), `.rodata` the same size.

## What didn't work first
- **Faults keyed by direction** (copy out / copy in, a556543). The write-behind's copies out took a fault armed
  for a spill: the first repro run's off line said `write-behind` (1104-093038-ca19). Faults are now keyed by
  the op the copy is made for.
- **The first repro's page arithmetic** (9 pages, a short D). D's `adopt` released the pages its run alone had
  reserved, so R's restore always found free pages and never reached `reclaim`. The rebuilt repro gives D a
  prompt over 4 pages, so its rows hold the pages whenever it hands over, and R 3 stored pages.
- **The one cache's write-behind fault had nothing to copy**: A's turn 3 ended inside A's last stored page,
  so the write after it copied nothing (1104-093038-2b56 / 8d80: `the off lines: []`). A's turn 3 now adds 10
  sentences (over a page), and the test asserts that turn is at least a page past turn 2's rows.
- Binding the copy's result (`let r = ...`, with a `cfg(test)` / `cfg(not(test))` pair) changed
  `PoolTier::write`'s stack frame in release (+16 B frame, +9 B code). Fixed by making the `cfg(test)` check a
  statement of its own before main's line (5bc698a).

## Gates
See the PR body (job ids and verdicts).
