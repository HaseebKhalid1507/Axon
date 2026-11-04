# Tests for the KV RAM tier's off path  (issue #172)

- Date: 2026-11-04. Branch: test/kvtier-off-path. Base: main 9948129. Status: open (draft)
- Pick: foreman (axon), issue #172. Builder: joestar.

## What changed
Mostly test-only. Since #166, a KV RAM tier error (`--cache-ram`) turns the tier off rather than failing the
engine. On the pool path that is `Batch::tier_failed` (from a spill, the write-behind or a restore); on the
one-cache path, `Engine::run` (its write) and `Engine::job` (its restore). /metrics then shows the tier off
(`Metrics::kv_ram_off`: gauges 0, counters kept). No test drove any of it.

- **A test-only fault hook**, `serve::tier::Faults`, held in `Metrics::tier_faults` because that is the one
  handle a test's `Server` and the engine's tier both hold. `fail_out(n)` / `fail_in(n)` arm the n-th next
  copy out / in: a pool's `page_out` / `page_out_id` / `page_in` (one page each), or the one cache's `kv_save` /
  `kv_load` (all the pages one write or restore moves). The copy it hits fails **instead of running**. A failed
  copy in therefore leaves the rows it was to fill as they were (another conversation's rows), so a reply built
  on them would differ from the reference. (A fault after the copy would leave the right rows in place and hide
  that bug.) Every line of the hook is `#[cfg(test)]`, and the product's lines at the copy sites are main's,
  verbatim.
- **One off line** (`tier::off`, product code): `axon serve: --cache-ram: {what}: {e}; the KV RAM tier is off
  from here`, where `what` is `spill`, `write-behind` or `restore`, on every path. The single-cache path used to
  print `...: {e}; off from here` and the pool path didn't name the operation. Under `cfg(test)` it also
  records the line and the tier's counters at that moment; the tests' stderr check reads that record.
- **Tests** (`serve/tests.rs`):
  - `serve_cpu_cache_ram_off` (the one cache, CPU) and `serve_cuda_cache_ram_off` (a paged pool, a plain pool
    and `--parallel 1`). They use `cache_ram_case`'s conversations (A, B, C; A alone; B, C; A again with D
    decoding on a pool, a batched admission), then B and A again. Each run fails the tier once,
    mid-conversation:
    - pool: B's spill after it ran alone; A's turn-2 restore after its first page came back (n = 2: a partial
      restore must not be trusted); the batch's write-behind (A's turn 3 admitted while D decodes).
    - one cache: the write after A's turn 3 (its earlier turns came back from RAM); A's turn-2 restore, while
      the cache holds C's rows.

    Each run asserts:
    - every reply, D's included, is bit-identical to the `--cache-ram 0` run's (greedy), with the same prompt
      and completion token counts;
    - the server answers the later requests;
    - exactly one off line, naming the expected operation and the injected fault;
    - /metrics: `axon_kv_ram_capacity_bytes`, `_pinned_bytes`, `_bytes` and `_pages` are 0; the counters equal
      what they were when the tier failed (kept, and unchanged since); pages were written before the failure,
      and a hit happened before it for the turn-3 faults.
  - `serve_cuda_cache_ram_off_in_reclaim`: the bug below.

## Why
#172: the off path shipped untested. A tier that fails must leave the server as `--cache-ram 0` would.

## Zero code in release (MEASURED: product binaries, main 9948129 vs this branch, same rustc recipe)
- `strings`: the fault message is in neither binary. `nm -C`: no `Faults`, `tier_faults`, `fail_out`, `fail_in`
  or `off_lines` symbol in either.
- `PoolTier::write`, `PoolTier::restore`, `CacheTier::write` and `CacheTier::restore` disassemble (objdump,
  addresses masked) to the **same instructions** as main's: 1213, 465, 1184 and 528 instructions.
- The only differences come from the `off` refactor: `tier::off` (new, 242 B), `Batch::spill` 391 -> 323 B,
  `Batch::restore` 1170 -> 1175 B, `Engine::run` 8115 -> 8412 B. `.text` +608 B, `.rodata` the same size.
- What didn't work first: binding the copy's result (`let r = ...`, with a `cfg(test)` / `cfg(not(test))` pair)
  changed `PoolTier::write`'s stack frame in release (+16 B frame, +9 B code). Fixed by making the `cfg(test)`
  check a statement of its own before main's line (5bc698a).

## Found: a tier error in a restore's reclaim exits the server
`Batch::restore` -> `make_room` -> `reclaim` -> `spill(victim)`. If that spill's copy fails, `tier_failed` drops
the tier, and `restore` then runs `self.tier.as_mut().expect("on")` (engine.rs), which panics. `or_exit("an
admission")` turns the panic into `exit(70)` for the whole server, the opposite of #166's contract. It needs a
batched admission (another request decoding) whose restore lacks pages, and an idle reclaim victim holding whole
pages the tier didn't store when it ended (the tier was full) and has room for now. The test builds exactly
that, and asserts each step through /slots and /metrics (doc comment). Reported to the foreman before any fix
(scope decision).

## Gates
See the PR body (job ids and verdicts).
