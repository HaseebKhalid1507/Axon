# Shortest-first admission and a measured step budget for `axon serve` (PR #126)

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

## Review fixes (shady 7/10, 10-24)
- `pass_us` was an unguarded EMA: one stalled short feed could pin two-pass prefill at `floor` tokens a step until
  the next short feed (in hybrid, maybe never). Now a sample is at most 4x the estimate and half a step, and the
  estimate is read bounded by twice the fastest decode-only step (one pass over the weights, almost no token
  work); decode-only steps run whenever no prompt waits, so the bound holds in hybrid too. Test
  `pass_us_outlier_recovers`: a 10x sample, then normal ones; the step's short-prompt count is back within 3
  (within 5 when the outlier is the first sample, nothing to cap it against).
- The short/long class came from the current plan (`fused_share` over `pre`), so it flipped as a round admitted
  prompts (on the H100's numbers a 300-token prompt: long with nothing prefilling, short with two long ones).
  Now from the step's capacity (one endless prompt at depth 0). Test `short_class_stable_across_a_round`.
- Two-pass steps feed one pass per short prompt; the short gate now counts passes too (2 steps' worth).
- A pages refusal no longer ends the round at once: short prompts that fit the pages left still start, until the
  refused request has waited 2x `--admit-age-ms` (then it ends the round). `pages_block_bounds_the_wait`.
- Cold start: before any short feed, a pass is charged the fastest decode step (measured; still no fixed seed),
  so a cold wave of short prompts isn't fed in one long step. `cold_pass_cost_is_the_decode_step`.
- `axon_step_cost_legacy` gauge; `serve_rollback_flags_reach_the_engine` checks the rollback arm's lanes and
  legacy flag on /metrics.

## What didn't work
- A 6 ms seed for the pass's fixed part (the spec's): on a 0.5B a 512-token feed is ~15 ms, so the seed would skew
  the learned per-token cost by ~40 %. Unmeasured, it is the fastest decode step (above).
- The spec's "first refusal ends the round": an aged long prompt refused for a lane would then block every short
  prompt behind it. A refusal by class (lane / short budget) skips to the next candidate; a refusal for a slot or
  pages still ends the round, so nothing overtakes on pages.
