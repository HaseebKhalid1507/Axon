# Hybrid step rejected (NO-GO, PR #119 closed)

- Date: 2026-10-25. Branch: docs/reject-hybrid-step (this write-up); the code is on `feat/hybrid-step` (#119, closed,
  not merged; kept). Base: main d6b7051.
- Pick / design: roadmap item 1 (b), from Fable's 10-19 review; design `docs/research/research-sched.md` P1.

Labels: **MEASURED** (job / pod run cited), **ESTIMATED**.

## What was tried
- A serve step became one pass: the decode rows and the prompts' chunks together, instead of a chunk pass then a
  decode step. `--prefill-chunk` became the pass's total tokens, rows included, rounded up to 64-token tiles.
  `AXON_HYBRID=0` kept the old two-pass code.
- Bits: unchanged, by identity A. The correctness gates passed at c2bd9f6: md5 0/20, serve A/B one digest × 6, the
  suites.

## Kill rules (written before the build)
| rule | cell | result | needed |
|---|---|---|---|
| roadmap item 1 (b) | mix C = 64, 7B Q4_K_M, clean secure RTX 4090 pod (10-25; `~/.cache/axon-p4090/results/out/mix64-ecdb1d0`) | agg 1,699 vs main 1,688 tok/s (**+0.65 %**), ITL p99 39.0 vs 72.3 ms | ≥ +15 % with ITL p99 ≤ today's |
| #119 §8 | 7B loop C = 8 | clean RTX A5000 **+6.0 %** (bella TAINTED: +2.9 / +6.4 %) | ≥ +8 % |

Both MEASURED and both missed, so it's rejected.

## Results (MEASURED, mix C = 64, medians of 2 rotated rounds, 4090)
| arm | agg tok/s | ITL p99 ms | ITL p50 | TTFT p50 ms | TTFT p99 | slo_ok |
|---|---|---|---|---|---|---|
| main 6e3b19c (`--prefill-ms 60`) | 1,688 | 72.3 | 16.1 | 486 | 5,785 | 0.696 |
| hybrid ecdb1d0 (`AXON_HYBRID=1`) | 1,699 | 39.0 | 26.6 | 1,734 | 6,620 | 0.68 |
| #119 two-pass (`AXON_HYBRID=0`) | 1,657 | 72.9 | 16.1 | 502 | 5,811 | 0.692 |

Also MEASURED, 7B loop C = 8 on a clean A5000:
- agg 456.2 vs 430.4 (+6.0 %);
- admit ITL-during p99 96.2 vs 157.5 (−39 %);
- long-prompt loop +11.3 %.

## What it bought, and why it lost
- **Bought:** smoother decode. ITL p99 −46 % at C = 64, and admit ITL-during p99 −39 %.
- **Lost:**
  - throughput is flat (+0.65 %);
  - TTFT p50 is 3.6× worse at C = 64;
  - ITL p50 is higher (26.6 vs 16.1 ms), because every decode row now waits for a pass that includes a chunk.
- **Why (ESTIMATED, not profiled):**
  - At C = 64 the step isn't bound by the second weight read that the merge saves.
  - With the rows counted in the budget, 64 rows leave a 256-token pass about 192 prompt tokens, so prompts are
    admitted more slowly.
- **A future variant must:**
  - keep the decode rows out of the prompt budget, or size the pass from the measured step cost (item 1 (d));
  - first show a weight-read-bound step in the step anatomy (item 1 (a));
  - put TTFT p50 in its kill rule.

## Next
(c) shortest-first admission and (d) the measured step budget continue in two-pass mode via #126, restacked onto
main.
