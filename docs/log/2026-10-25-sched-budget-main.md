# Shortest-first admission and a measured step budget for `axon serve`, on main (PR #TBD)

- Date: 2026-10-25. Branch: feat/sched-budget-main. Base: main d6b7051. Status: open (draft, gates pending)
- Pick / design: roadmap "Next" item 1 (c) and (d); spec `specs/sched-sjf-token-budget.md` (Okarin, 10-20).
- Supersedes #126 (the same work stacked on #119's hybrid step; #119 failed its kill rules and was closed: the
  roadmap's "else revert it and keep (c)/(d)"). This is (c)+(d) in the two-pass path only: no fused pass, no
  `fused_share`, no `AXON_HYBRID`.

## What changed
- `src/serve/engine.rs`: `StepCost` (a pass's fixed µs, µs per prompt-token unit, decode-only ms per row-count
  bucket; learned from the steps that run). `--step-ms` (default 50; `--prefill-ms` kept as its alias) turns it
  into the chunk (`two_pass_budget`: the modelled decode step at this row count instead of the last step's time;
  `plan_charged` charges a pass per prompt fed). `--step-tokens N` pins it.
- Admission: `admit_gate` (long prompts take one of `--prefill-lanes` lanes, default slots/16 in 2..8; short ones,
  fed whole in a step, never wait for a lane, only for about two steps' worth of short prompts already queued, by
  tokens and by passes), `admit_key` (aged past `--admit-age-ms` 3000 first, then the shortest uncached prompt,
  then arrival). The short bound is the step's capacity (`two_pass_capacity`), not the current plan.
- `src/serve.rs`: `--admit-order sjf` (default) pulls up to max(slots, 16) requests off the channel and admits
  several per engine iteration; `fifo` runs main's loop verbatim. `AXON_ADMIT=fifo` sets it from the environment.
  After a refusal for pages, short prompts that fit still start until the refused request has waited 2x the age
  bound (`pages_block`).
- `/metrics`: `axon_step_seconds`, `axon_step_seconds_rows16` (steps that fed prompt beside rows), cost gauges,
  `axon_step_cost_legacy`, lanes, waiting count; `/slots` lists the waiting requests.
- Rollback without a revert: `--admit-order fifo` with `AXON_STEP_COST=legacy` (`--step-ms 60`, main's default
  budget, and `--prefill-lanes 2` past 32 slots) = main's decisions, bit for bit
  (`legacy_reproduces_main_decisions`, 20,000 random states against frozen copies of main's `budget` and `plan`).

## Why
The spec's reading of the H100 showdown (run h100-1019-2058): the mix C=128 short-prompt TTFT (p50 2.4-2.9 s, max
14.7 s for 60-token prompts) comes from FIFO, one admission per iteration and 2 prefill lanes, not from kernel
speed; and main's step lands 8 ms over the 50 ms ITL SLO (`--prefill-ms 60`). See the spec §0-§1.

## How it was verified
See the PR body (gates table, job IDs). Bits: the identity suites, md5 of `gen` vs main, serve digests across
arms (main, cand, cand rollback, cand lanes 8 / age 0).

## Results
Pending at this head (PR body §5 and an Update line here). Before the restack, on the #119-stacked build with
`AXON_HYBRID=0` (the same two-pass code path), avante 0.5B mix C=32, 3 rounds, clean (job 1024-212712-7985):
sjf50 vs roll (main's scheduling): aggregate +1.5 %, short TTFT p90 -50 %, ITL p99 54.7 vs 68.0 ms, step p50
(rows >= 16) 50.5 vs 58.0 ms.

## Review fixes carried over (shady 7/10 on #126)
- `pass_us` bounded: a sample is at most 4x the estimate and half a step; the estimate is read bounded by twice
  the fastest decode-only step. `pass_us_outlier_recovers`: a 10x sample, then normal ones; the step's
  short-prompt count is back within 3 (within 5 when the outlier is the first sample).
- Short/long class from the step's capacity, not the current plan: `short_class_stable_across_a_round`.
- Two-pass short gate counts passes; shorts past a pages refusal (bounded by 2x the age bound); cold start charges
  the fastest decode step per pass (measured, no seed); `serve_rollback_flags_reach_the_engine`.

## What didn't work
- #119's fused pass (see its PR): the 4090 mix C=64 cell gave +0.65 % against the +15 % its kill rule needed.
  With it, sjf50 doubled ITL p99 against roll in hybrid mode (0.5B 25.9 -> 56.7 ms, 7B C=16 26.5 -> 53.2 ms: the
  fused pass sized up to 50 ms against #119's fixed 256 tokens); in two-pass mode the same arms lower it.
- A 6 ms seed for the pass's fixed part (the spec's): on a 0.5B a 512-token feed is ~15 ms, so the seed would skew
  the learned per-token cost by ~40 %. Unmeasured, it is the fastest decode step.
- The spec's "first refusal ends the round": an aged long prompt refused for a lane would then block every short
  prompt behind it. A refusal by class skips to the next candidate; a slot refusal ends the round.
