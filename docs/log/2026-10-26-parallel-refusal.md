# serve: refuse an explicit --parallel N that doesn't fit  (PR #PRNUM)

- Date: 2026-10-26. Branch: fix/parallel. Base: main 39624db. Status: open
- Pick / design: issue #137 (found by okarin while gating #136); the --help default fix was asked by the foreman on
  the same flag.

## What changed
- `serve`: an explicit `--parallel N` whose slots don't fit in GPU memory at start is **refused** (exit 2, like the
  CPU/split and MoE `--parallel` refusals) instead of served with fewer slots. The message gives what was asked, how
  many fit, the bytes N slots need and the bytes free, the plan's reason, and the fixes (a smaller `--ctx`,
  `--kv q8`, fewer `--parallel`, or leave `--parallel` out). Pure decision: `serve::parallel_refusal`.
- Also refused when `--parallel` was given: a pool or request view that fails to allocate, and (tests only) a paged
  pool asked for with `kv_pages` that wasn't planned (the #137 failure mode of `serve_paged_pressure`).
- `--parallel` left out (automatic sizing): unchanged fallback. Falling back to one request at a time, a failed pool,
  and the paged plan falling back to reserved slots are now printed even under `--quiet`.
- In tests the server runs in the test process, so the refusal panics the engine thread (`start` returns an error)
  instead of exiting the test binary (`serve::refuse_start`).
- `Opts::paged` (tests): force reserved slots (None: `AXON_PAGED_KV`, as before).
- `axon --help`: `--parallel` defaults to 16 with paged KV (the default; `PAGED_PARALLEL`) and 8 with
  `AXON_PAGED_KV=0`; it said 8. The README already said 16.

## Why
Job 1025-085841-ee61 (#137): `serve_paged_pressure` asked for `parallel 6`, `kv_pages 40` and came up with one slot
and no page pool, silently (`log: false`), because free VRAM was short at start. A production `--parallel 6` under
`--quiet` behaves the same way: 6x less concurrency, nothing said. Option (a) of the issue (refuse) was chosen by the
foreman; it matches how `--parallel` is refused on the CPU/split and for MoE (`moe_parallel_refusal`, #127): only
an explicit value is refused, never the automatic sizing.

## How it was verified
GATES

## Results
No speed claim: start-up decisions only. Bits: the md5 of `gen` output vs main (serve's numerics are untouched).

## What didn't work
n/a.
