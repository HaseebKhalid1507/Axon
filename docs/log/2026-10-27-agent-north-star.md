# The north star on the roadmap: run Synaps on Axon (agent workload track)  (docs only)

- Date: 2026-10-27. Branch: docs/agent-north-star. Base: main d8a296c (#152). Status: open.
- Pick / design: Haseeb set the north star on 10-26 ("run Synaps on Axon: many concurrent sessions, each spawning
  lots of subagents, with sessions staying warm in cache, then multiple users"). Fable wrote the track spec, which
  the lead asked to apply.

## What changed
- New `docs/design/agent-workload.md`: Fable's track spec, copied from the foreman's workspace
  (`specs/agent-workload.md`). Changed:
  - a header line (owner set 10-26 for the north star and the ranking; the design itself stays PROPOSED);
  - relative links for the repo location (determinism.md, multi-gpu.md, the #71 / H100-anatomy / W4A8-probe logs,
    history for #48, the roadmap);
  - a re-count note in §0.
  The §0 aggregate table is numbers and structure only. I read it line by line and found no message text, no
  names, no paths beyond the log file pattern, and no credentials.
- `docs/roadmap.md`:
  - the north-star paragraph (design §7) opens "Mission and positioning";
  - "Next: proposed order": an "owner set 10-26" note and rows **2A** (PR A, in shelved item 2's slot) and
    **2B, 2C** (two tracks, ahead of item 4), each with the design's costs and pre-registered rules. Item 4 is
    marked as behind them and blocked on the ADR. The shelved item 2 row stays as it was, since item 4's text
    refers to it;
  - "Agent-harness workloads" drops "(later: not scheduled)" and takes the design's §7 rows (anchors updated);
  - the two "below the line" mentions of stage B and shared-prefix pages now point at 2B / 2C.
- `docs/START-HERE.md`: one sentence pointing at the north star.

## Sources and checks
- The design's §0 numbers were re-counted from the `Token Usage` lines of `~/.synaps-cli/synaps.log.2026-10-2{0..7}`,
  numbers only. The re-count found 22,838 requests (the 10-27 log is live) against the spec's 22,795. The prompt
  p50 / p90 / p99 (165,532 / 410,485 / 809,911) and the 97.6 % cached share match. The output and new-token
  percentiles are within the extra requests (output p50 351, p90 1,543, p99 5,554, mean 700; new tokens p50 1,045).
  The gap, fan-out and lifetime rows weren't re-counted (they need the dispatch lines); they stay as the spec gives
  them.
- #48's 67 ms page restore: `docs/history.md` (#48 row). The 57 % attention share at depth 4,096:
  `log/2026-10-26-h100-anatomy-results.md`. Item 2 shelved: `log/2026-10-26-w4a8-probe.md`.
- Relative links and anchors in README, docs/, docs/log, docs/design, docs/research, docs/reviews: script (GitHub slug
  rules, negative-tested), 0 bad.
- No cargo, builds or model runs (docs only).
