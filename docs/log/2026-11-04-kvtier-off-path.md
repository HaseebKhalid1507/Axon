# Tests for the KV RAM tier's off path, and a fix  (issue #172)

- Date: 2026-11-04. Branch: test/kvtier-off-path. Base: main a5836cf (round 1: f458f48). Status: open, ready
  for review (PR #181). Round 1 reviewed by shady, 8/10 SHIP-WITH-FIXES on 14c95a7; round 2 (bd58fe2, merged
  with main as 1e0f248) answers it.
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
  that bug.) `hit()` records the op and the slot of the copy it failed. `idles()` counts the times the engine
  waited for a request with nothing to do (one `cfg(test)` statement where the engine loop waits): once it is
  past the count read after a reply, the reply's tail is done (a pool's `end_solo` spill, the one cache's write,
  a batch's last write-behind). Every line of the hook is `#[cfg(test)]`; the product's lines at the copy sites
  are main's, verbatim.
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
      restore must not be trusted; alone, through `lend_solo`); with A's turn 3 admitted while D decodes, the
      batch's write-behind, and A's restore in that admission (n = 2, through `admit`, which takes restore's
      return as the positions cached; `lend_solo` discards it).
    - one cache: the write after A's turn 3 (its earlier turns came back from RAM); A's turn-2 restore, while
      the cache holds C's rows.

    Each turn is waited out (`idles()`) before the next is sent or a fault armed, so a fault can only fail a
    copy of its own turn (round 1 armed straight after a reply, so A's `end_solo` spill could take B's fault).
    Each run asserts:
    - every reply, D's included, is bit-identical to the `--cache-ram 0` run's (greedy), with the same prompt
      and completion token counts; from the turn the tier is off on (the restore's own turn; the one after a
      write's) the cached tokens too: the GPU rows are `--cache-ram 0`'s, and a restore trusted in part would
      count the page that landed;
    - the off line comes in the armed turn: none after the turns before it, one after it and every later turn;
    - the counters the off line saw are the turn before's written pages (the failed copy was the turn's first
      out, or one in), all of them for a restore fault (a failed restore counts nothing) or a fault by turn 3,
      and with A's turn-3 hit for the turn-6 write faults;
    - A's first turn stored 2 pages or more, all still in the tier before the armed turn, so `Restore` with
      n = 2 is "the first page lands, the second fails";
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

## Zero code in release (MEASURED: product binaries, same rustc recipe, axon-rustc-build)
Re-measured in round 2: round 1's per-function disassemblies were overwritten in a loop, so the files saved for
d93920f were the last function's (89 B, a symbol not in either binary), and its counts included 4 header lines
each. The comparison is now a script (`relcmp.py`, attached to the PR) that keeps every function's masked
disassembly. Masked: instruction addresses, absolute targets, `%rip` displacements, ThinLTO's `.llvm.<n>`
suffixes, the crate hash and index of anonymous constants, GOT slot offsets.

- **No hook in the binary** (main f458f48 vs d93920f, main a5836cf vs 1e0f248): `strings` finds neither the fault
  message nor `a copy for a`; `nm -C` finds no `Faults`, `tier_faults`, `tier::Op`, `off_lines`, `idles` symbol.
- **The four hooked functions are main's instructions:**

  | function | f458f48 / d93920f | a5836cf / 1e0f248 |
  |---|---|---|
  | `PoolTier::write` | 1209 / 1209, identical | 1209 / 1209, identical but pool offsets |
  | `PoolTier::restore` | 461 / 461, identical | 461 / 461, identical |
  | `CacheTier::write` | 1180 / 1180, identical | 1180 / 1180, identical but pool offsets |
  | `CacheTier::restore` | 524 / 524, identical | 524 / 524, identical but pool offsets |

  "Pool offsets": the same instructions, with 4 operands pointing at a different offset into the codegen unit's
  anonymous constant pool (it gained entries ahead of them). The pointed-at data was not compared; it includes
  panic Locations, whose line numbers moved with the test-only lines above them.
- **Every function of `serve::tier`, `serve::engine::Batch` and `serve::Engine`** (31 at the new head): the same
  instructions, except `Engine::run` (1417 -> 1487 instructions), `Batch::restore` (267 -> 256), `Batch::spill`
  (87 -> 80) and the new `tier::off` (51). Those are the off-line refactor (the two `Engine::run` branches and
  `tier_failed` call `tier::off` instead of `eprintln!` + `kv_ram_off`) and the fix (`restore`'s `expect` and
  its panic path are gone). The engine loop (`serve::engine`, Engine::job inlined) is -83 B: `Engine::job`'s
  off branch, the same refactor.
- **.text, accounted for in full** (symbol sizes from `nm -S` over the whole binary, plus the padding between
  functions, add up to the `.text` delta exactly):

  | | f458f48 -> d93920f | a5836cf -> 1e0f248 |
  |---|---|---|
  | serve code this PR changes: `tier::off` +242, `Engine::run` +297, `Batch::restore` -65, `Batch::spill` -68, `serve::engine` -83 | +323 B | +323 B |
  | other functions (below) | +188 B | -1,465 B |
  | alignment padding between functions | +33 B | +86 B |
  | **.text** | **+544 B** | **-1,056 B** |

  The "other functions" are codegen-unit side effects, not code the PR adds. The build is deterministic
  (a5836cf built twice: byte-identical, sha256 da14f0ad...). The lib's 4 codegen units keep their split (the same
  CGU names; `tier::off` lands in cgu.2 with the tier; per-CGU `nm -S` of the lib rlibs). Every symbol in cgu.0 is
  the same size. Every other changed symbol is a generic or std instance in cgu.1/2/3, which hold the changed
  serve code, or one that moved between them: std sort (`drift::sort` x4, -576 B each), `hashbrown`
  `reserve_rehash` / `HashMap::remove`, the `kvstore::PageStore::make_room` instances (-246 / -201 B), `serde_json`
  index impls, `crossterm` command impls. The likely cause is LLVM's inlining choices changing with a unit's
  contents. I didn't check that function by function. The set differs per base: against f458f48 it was
  `CudaQwen2::build` +284, `Map::fold` -309, two `Vec::from_iter` +175, `safetensors::run_jobs` +68,
  `Ring::upload` -38, tokenizer `bpe_word` -8 / `from_gguf` +16. None of them is code this PR touches.

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
- **Round 1's disassembly files** were overwritten per function in a shell loop, so the d93920f pair kept on
  disk was the last one (89 B, a symbol in neither binary), and the counts included objdump's 4 header lines.
  Redone with `relcmp.py` above. The verdict (the hooked functions are main's instructions) holds.
- **Round 1's faults weren't pinned to a turn.** A fault armed right after a reply could take that reply's
  tail: a pool's `end_solo` spill (Spill@1 could have hit A's spill, not B's), the one cache's write (Write@6
  could race C's). The tests passed, but they couldn't tell those apart. `idles()` and the per-turn asserts fix
  that.
- Binding the copy's result (`let r = ...`, with a `cfg(test)` / `cfg(not(test))` pair) changed
  `PoolTier::write`'s stack frame in release (+16 B frame, +9 B code). Fixed by making the `cfg(test)` check a
  statement of its own before main's line (5bc698a).

## No md5 gate
n/a: the md5 set runs `axon gen`, which never reaches `serve/`. The serve code that changed in release is the
tier's error branches (above). `Engine::run` did change (+297 B), but only in its off branch, and that branch runs
only after a tier error, with `--cache-ram` given. Without `--cache-ram`, no changed line runs.

## Gates
All on bella (RTX 5070 Ti Laptop sm_120, driver 615.78.08). `serve:: kvstore:: cuda::qwen2::slots::`,
`--test-threads 1`, nothing skipped:
- Round 1 at d93920f: release 1104-145037-6285 and debug 1104-145037-1ee9 (168 passed / 0 failed / 9 ignored each),
  the repro after the fix 1104-145038-8b2a (pass), and before the fix, at 658ef08, 1104-144238-8244 (exit 70:
  the bug). Earlier rounds at 5bc698a (1104-093038-2b56 / 8d80 / ca19) failed on the hook's direction keying and
  the empty write (above).
- Round 2 at 1e0f248: the three off-path tests 5 times each, release (1104-152814-8b5f: 15 passed, 0 failed);
  release 1104-152814-7b76 and debug 1104-152814-d474 (verdicts in the PR body).
