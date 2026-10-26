# serve: refuse an explicit --parallel N that doesn't fit  (PR #141)

- Date: 2026-10-26. Branch: fix/parallel. Base: main 39624db, merged up to 161e787 (docs only). Status: open
- Pick / design: issue #137 (found by okarin while gating #136). The `--help` default fix was asked for by the foreman,
  since it's the same flag.

## What changed
- `serve`: an explicit `--parallel N` whose slots don't fit in GPU memory at start is **refused** instead of served
  with fewer slots (one request at a time at worst, said only with logging on). The engine sends a `Refused` (the
  message) instead of the loaded model, `serve::start` returns it as an `InvalidInput` error carrying it
  (`is_refused`), and `serve::run` (the CLI, and the public `axon::serve`, which calls it) exits 2 with it through
  `crate::fail`, as the CPU/split and MoE `--parallel` refusals do. So the library still exits the process on this
  refusal through `axon::serve`, as that function's doc says of every usage error. `start` itself is crate-private;
  a typed, matchable start error for embedders is #118's.
- The message: what was asked; why the paged pool didn't fit, when it was tried (one `--ctx` sequence's pages don't
  fit beside the model and N slots' scratch, or not even 1024 tokens' worth; the GiB free, when it could be read);
  the GiB N reserved slots need at the plan's context and the GiB free; how many fit; the fixes (a smaller `--ctx`,
  `--kv q8`, fewer `--parallel`, or leave `--parallel` out). The planner's own note isn't repeated.
- Also refused when `--parallel` was given: a pool or request view that fails to allocate, and (tests only) a paged
  pool asked for with `kv_pages` that wasn't planned (the #137 failure mode of `serve_paged_pressure`).
- `--parallel` left out (automatic sizing): unchanged fallback. Falling back to one request at a time, a failed pool,
  the paged pool falling back to reserved slots (printed only when the start isn't refused), and an explicit
  `--parallel N` whose context was shortened are printed even under `--quiet`.
- `Opts::paged` (tests): force reserved slots (None: `AXON_PAGED_KV`, as before).
- `axon --help`: `--parallel` defaults to 16 with paged KV (the default; `PAGED_PARALLEL`) and 8 with
  `AXON_PAGED_KV=0`; it said 8. The README already said 16. A test checks the numbers against the constants.
- CHANGELOG ("Behaviour change (#141)"), README, the lib.rs doc of `serve` (this refusal is an exit-2 case),
  docs/results.md and START-HERE (#137 no longer an open limitation).

## Why
Job 1025-085841-ee61 (#137): `serve_paged_pressure` asked for `parallel 6`, `kv_pages 40` and came up with one slot
and no page pool, silently (`log: false`), because free VRAM was short at start. A production `--parallel 6` under
`--quiet` behaves the same way: 6× less concurrency, nothing said. Of the issue's options, (a) (refuse) was chosen by
the foreman. It matches how `--parallel` is refused on the CPU/split and for MoE (`moe_parallel_refusal`, #127): only
an explicit value is refused, never the automatic sizing.

## How it was verified
| gate | result | job |
|---|---|---|
| focused (refusal predicate, end-to-end refusal, MoE predicate, help defaults, `serve_paged_pressure`) @b78c1aa, release | PASS 5 / 0 / 0 | [1025-233655-7409](https://github.com/HaseebKhalid1507/Axon/blob/a20256f50710d009408eca869729a3d618bd8717/2026-10/parallel-b78c1aa/1025-233655-7409.log) (avante sm_89) |
| same, debug assertions | PASS 5 / 0 / 0 | [1025-233656-033e](https://github.com/HaseebKhalid1507/Axon/blob/a20256f50710d009408eca869729a3d618bd8717/2026-10/parallel-b78c1aa/1025-233656-033e.log) (avante) |
| fast tier, release @b78c1aa | PASS 673 / 0 / 50 (18 filtered) | [1025-233657-d706](https://github.com/HaseebKhalid1507/Axon/blob/a20256f50710d009408eca869729a3d618bd8717/2026-10/parallel-b78c1aa/1025-233657-d706.log) (bella sm_120) |
| fast tier, debug assertions (CPU) @b78c1aa | PASS 691 / 0 / 50 | [1025-233657-bbb0](https://github.com/HaseebKhalid1507/Axon/blob/a20256f50710d009408eca869729a3d618bd8717/2026-10/parallel-b78c1aa/1025-233657-bbb0.log) (bella) |
| md5 of `gen` output vs main 39624db, 4 models × 3 prompts, cpu + cuda f16 @ad27921 | 24 / 24 identical (not rerun at b78c1aa: that diff only moves a stderr line, drops an unknown figure from a message and edits a doc) | 1025-230929-1a0e (bella) |
| CLI demo, main vs candidate @ad27921 | main serves 25 slots silently for `--parallel 256`; candidate exits 2 with the numbers | 1025-230929-0a5d (bella) |
| clippy -D warnings (lib, lib tests, bin), rustfmt | clean | local |

## Results
CLI demo, MEASURED on bella (job 1025-230929-0a5d, round 1 code ad27921): `axon serve` qwen2.5-0.5b `--device cuda
--ctx 32768 --quiet`, reserved slots. `--parallel 256`: main served 25 slots silently; the candidate exits 2 with
"--parallel 256: 256 reserved slots of 32768 tokens need 96.71 GiB of GPU memory beside the model, 10.32 GiB is free;
25 fit. …". `--parallel` left out: 8 slots on both. No speed claim. Bits: md5 of `gen` output vs main, 24/24.

## What didn't work
- A `#[cfg(test)]` panic in place of the exit (round 0): the in-process test then didn't run the production path.
  Replaced by the `Refused` error from `start` (review round 1).
