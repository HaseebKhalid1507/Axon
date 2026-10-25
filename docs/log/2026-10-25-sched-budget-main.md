# Shortest-first admission and a measured step budget for `axon serve`, on main (PR #131)

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
| gate | result | job ID |
|---|---|---|
| fast tier, release (a18ff8d) | 642 passed, 0 failed, 49 ignored | avante 1025-011101-7b35 |
| debug-assertion suite, CPU (a18ff8d) | 660 passed, 0 failed, 49 ignored | avante 1025-011102-d0a2 |
| debug `serve::` + `cuda::qwen2::slots::`, GPU (a18ff8d) | 142 passed, 0 failed, 7 ignored | avante 1025-011102-6e5e |
| `serve::` release, rollback env (a18ff8d) | 131 passed, 1 failed (`serve_sjf_short_ahead_of_long`: test bug, below) | avante 1025-011103-2a37 |
| `serve::` release, rollback env (38bca6d) | **132 passed, 0 failed, 4 ignored** | avante 1025-024210-ace4 |
| debug `serve::` + `slots::`, GPU (38bca6d) | pending (queued) | avante 1025-024210-b160 |
| md5 of `gen`, main d6b7051 vs a18ff8d, same host | 0 of 20 pairs differ | bella 1025-011103-bbeb |
| serve digests (main, cand, rollback, lanes 8 / age 0, step-tokens 64; 0.5B, 1.5B) | 1 distinct digest over 10 runs, each model | bella 1025-011104-fadf |
| clippy -D warnings (bin + test), rustfmt | clean | local |
| pure engine tests (incl. `legacy_reproduces_main_decisions`) | 23 passed (+ `pages_block_bounds_the_wait`) | local |

**The one failure, a test bug.** `serve_sjf_short_ahead_of_long` pins `--step-tokens 256` so the long prompts need ~20 steps, but it never set `legacy`, and `Sched::default()` reads `AXON_STEP_COST` from the environment. Under the rollback env the legacy budget (main's) returns before `--step-tokens` is read, by design. So the long prompts were fed in a few large time-budget chunks, and their last chunk landed in the step that finished the short request. Before the fix, on avante with the roll env, 4 of 5 runs failed (1025-023649-d087, -9908, -8e2c, -6a69; -7753 passed), each with the short request's done time equal to the first long first-token time (221–223 ms). The fix in 38bca6d is test only: `legacy = false` plus an assert that `axon_step_cost_legacy` is 0, 7000-token long prompts (~27 steps against ~5), and no 30 ms sleep. The property is unchanged. After the fix, 8 of 8 pass (roll env 1025-024017-e871, -7a4f, -8563, 1025-024018-2c82, -0c52; default env 1025-024018-cc30, 1025-024019-5eae, -48cb): the short request is done at ~70 ms, the first long first token at ~600–630 ms.

## Results
MEASURED, avante (RTX 4050 Laptop), clean (foreign GPU 0 %), at a18ff8d. roll = main's scheduling.
- `serve_sched_ab` mix, C=32, 3 rounds (0.5B 1025-012508-948a, 1.5B 1025-012609-5381). Short TTFT p90: main 226.6 / 1602.8 ms, sjf50 135.5 / 386.4 ms. Aggregate is within ±3 % everywhere (sjf50 vs main: +0.0 % / +2.8 %). slo_ok at 1.5B: 0.55 (main) → 0.92 (sjf50).
- sd_client mix, 0.5B, 2 rounds (1025-022130-32ae). Short TTFT p99 at C=16 / 32: main 712 / 1462 ms, sjf60 120 / 146 ms. Short p50 32–42 → 56–58 ms. Aggregate within ±2 %.
- Watch: at 1.5B, `--step-ms 50` has ITL p99 91.7 vs 75.6 ms (main) and step p50 58 ms (+16 % over target). The `--prefill-chunk 256` floor binds there.
- Kill-rule verdict: waits for the spec's bella cells (7B C=16, 1.5B C=32). They keep coming back TAINTED by the desktop and are deferred.

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
