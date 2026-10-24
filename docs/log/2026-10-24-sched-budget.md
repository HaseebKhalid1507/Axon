# Shortest-first admission and a measured step budget for `axon serve` (PR #TBD)

- Date: 2026-10-24. Branch: feat/sched-budget. Base: feat/hybrid-step (#119) ecdb1d0 (stacked; main 6e3b19c below it).
  Status: open (draft, gates pending)
- Pick / design: roadmap "Next" item 1 (c) and (d); spec `specs/sched-sjf-token-budget.md` (Okarin, 10-20).

## What changed
- `src/serve/engine.rs`: `StepCost` (a pass's fixed µs, µs per prompt-token unit, decode-only ms per row-count
  bucket, µs per hybrid-pass token unit; learned from the steps that run). `--step-ms` (default 50, `--prefill-ms`
  kept as its alias) turns it into the two-pass chunk (`two_pass_budget`: the modelled decode step at this row
  count instead of the last step's time; `plan_charged` charges a pass per prompt fed) and the hybrid pass's prompt
  tokens (`fused_share`: grows with rows and the step's time, whole 64-token tiles, at least `--prefill-chunk`;
  #119 had a fixed 256 tokens with the rows in them). `--step-tokens N` pins it.
- Admission: `admit_gate` (long prompts take one of `--prefill-lanes` lanes, default slots/16 in 2..8; short ones,
  fed whole in a step, never wait for a lane, only for about two steps' worth of short prompts already queued),
  `admit_key` (aged past `--admit-age-ms` 3000 first, then the shortest uncached prompt, then arrival).
- `src/serve.rs`: `--admit-order sjf` (default) pulls up to max(slots, 16) requests off the channel and admits
  several per engine iteration; `fifo` runs the old loop verbatim. `AXON_ADMIT=fifo` sets it from the environment.
- `/metrics`: `axon_step_seconds`, `axon_step_seconds_rows16` (steps that fed prompt beside rows), cost gauges,
  lanes, waiting count; `/slots` lists the waiting requests.
- Rollback without a revert: `--admit-order fifo --step-ms 60 --prefill-lanes 2` with `AXON_STEP_COST=legacy`.

## Why
The spec's reading of the H100 showdown (run h100-1019-2058): the mix C=128 short-prompt TTFT (p50 2.4-2.9 s, max
14.7 s for 60-token prompts) comes from FIFO, one admission per iteration and 2 prefill lanes, not from kernel
speed; and #119's fixed 256-token pass leaves 128 prompt tokens per step at 128 rows (ESTIMATED to regress mix
C=128). See the spec §0-§1.

## How it was verified
See the PR body (gates table, job IDs). Bits: the identity suites, md5 of `gen` vs main and vs #119, serve digests
across six arms; the rollback: `legacy_reproduces_119_decisions` (20,000 random states, every decision bit-equal
to a frozen copy of #119's code).

## Results
Pending (PR body §5 and an Update line here).

## What didn't work
- A 6 ms seed for the pass's fixed part (the spec's): on a 0.5B a 512-token feed is ~15 ms, so the seed would skew
  the learned per-token cost by ~40 %. The fixed part is 0 until a < 64-token feed teaches it (one does, in the
  first step that feeds a short prompt).
- The spec's "first refusal ends the round": an aged long prompt refused for a lane would then block every short
  prompt behind it. A refusal by class (lane / short budget) skips to the next candidate; a refusal for a slot or
  pages still ends the round, so nothing overtakes on pages.
