# Outside reviews

Reviews of the project by someone outside the build loop (so far: Fable, a consultant model working read-only on
the repo, the logs and the job records). They are curated copies: personal and financial specifics are removed and
marked, editor's notes flag claims that read differently against the published numbers, and everything else is the
reviewer's text with its own MEASURED / CODE / ESTIMATED / OPINION labels. **The opinions are the reviewer's;
decisions are Haseeb's.** A review is input to the [roadmap](../roadmap.md), not a plan.

| date | review | tree | what it is |
|---|---|---|---|
| 2026-10-19 01:14 EDT | [the big picture](2026-10-19-outside-review-bigpicture.md) | `05fb70c` + open PRs | what's built, standing vs llama.cpp / ollama / vLLM / SGLang / TRT-LLM, mission fit, roadmap critique, risks at 10× scope, the dev process, 15 hard questions, top 5 for two weeks |
| 2026-10-19 ~23:00 EDT | [next steps after the H100 showdown](2026-10-19-outside-review-next-steps.md) | `764792f` | scorecard against the first review, a decomposition of the H100 serving gap, a ranked next five with kill rules, the determinism-modes ADR given the H100 data, process leaks, 3 hard questions |

## Citation note

[design/determinism-modes.md](../design/determinism-modes.md) (#101) cites the first review as
`fable-bigpicture-20261019` with section numbers (§1, §2, §5 risk 1, §5 risk 4, §7, §8 recommendation 3). That file
is [2026-10-19-outside-review-bigpicture.md](2026-10-19-outside-review-bigpicture.md); its section numbers are
unchanged. The ADR itself is not edited (it is Haseeb's decision).

## What came of them

- First review, recommendation 1 (run the mission benchmark): the H100 serving showdown,
  [log/2026-10-19-h100-showdown.md](../log/2026-10-19-h100-showdown.md). Recommendation 3 (the determinism-cost ADR):
  #101, merged as PROPOSED.
- Second review, the ranked five: the roadmap's
  [proposed order](../roadmap.md#next-proposed-order-outside-review-10-19-owner-decides).
