# Docs brought up to main 764792f, plus the H100 serving showdown  (docs only)

- Date: 2026-10-19. Branch: docs/refresh-20261019. Base: main 764792f (#101). Status: open.
- Pick / design: maintainer request ("update all of the relevant axon docs") after the 10-19 merge train
  (#71, #84, #98) and the H100 serving showdown (run h100-1019-2058). The last refresh was
  [2026-10-18-results-refresh.md](2026-10-18-results-refresh.md) at 4507c75.

## What changed
- New entries: [2026-10-19-h100-showdown.md](2026-10-19-h100-showdown.md) (method, results, cost and waste, what
  went wrong, the pre-run A/B), [2026-10-19-merge-train.md](2026-10-19-merge-train.md) (train 1019-1542 and its
  gates), this file. Dated "Update" lines on the #71, #84 and #98 entries (merged).
- `docs/benchmarks-gpu.md`: new section "Concurrency: H100 SXM, Qwen2.5-7B (2026-10-19)" with the setup, the
  quant caveat, the cost, the headline and every `sd_table.py` table (aggregate, goodput, SLO fraction, TTFT, ITL,
  TPOT, completion lengths, control pass) for both workloads. The intro now names the two serving sections and says
  ESTIMATED is labelled.
- `docs/results.md`: main 764792f; a "Serving: RunPod H100 SXM" section; caveat-table rows for every merge since
  4507c75 (#79, #81, #83, #86 before the 7f42c02 board; #92/#97, #100, #71, #84, #89/#98, #102 after it; #82/#93
  and the docs PRs as combined rows); #89 and #98 in "New models and types"; known losses as of 764792f (H100
  serving first; MoE prefill; #90).
- `README.md`: opening line (deepseek2, safetensors), line counts, "Not on main yet", a "Serving on an H100 SXM"
  table in Numbers with an honest reading, a note that the 10-18 board predates #92 / #100, the ADR in the docs list.
- `docs/roadmap.md`: "on main" = 764792f; the mission table's datacenter row now cites the board (the rest of #101's
  section unchanged); new section **Next: proposed order (outside review 10-19, owner decides)**: the H100 gap
  and the review's decomposition, then its ranked five (step anatomy + scheduler fixes, exact Hopper GEMM + M6
  split-K, one MoE resident, FP8 under `fast` gated on the GEMM, adoption floor), each with gain, cost, kill rule and
  what it unblocks; the review's ADR recommendation; a two-pod re-run; below-the-line items (incl. #90's second
  attempt, marked in Scale item 10 and the agent-harness section); "Recently landed" for 10-18/19; done marks on
  Q2_K/Q3_K, MoE stage 2, safetensors stage A, paged KV 2–3, serve ops, YaRN + chat halves, gpt-oss stage 1; Scale
  item 10 → #90 closed, second attempt with a depth-aware gate; the frontier table's deepseek2 / qwen3moe cells.
- `docs/experiments-rejected.md`: #90 (batched speculative decoding, the three kill-rule runs) under Speculation;
  #84's 128-query design under GPU attention.
- `docs/history.md`: Phase 19 (4507c75 → 764792f, 18 merges), #90's close, the H100 board.
- `docs/START-HERE.md`: what Axon is (architectures, line counts, the serving gap), the "What works on main" rows
  (architectures, model files, KV cache, serving), the source map (serve engine, pages, MoE/MLA, safetensors),
  "Not on main", "In flight" as of 764792f, other docs.
- `docs/log/README.md`: history now runs to 10-19; boards, trains and refreshes get entries; where to start.
- New `docs/reviews/`: README (what they are, the ADR citation note), curated copies of the two 10-19 outside reviews
  (`2026-10-19-outside-review-bigpicture.md`, the one #101's ADR cites by section; `2026-10-19-outside-review-next-steps.md`).
  Curation removes personal and financial specifics (marked) and adds **[Editor]** notes where a claim reads
  differently against the published tables; labels, citations and section numbers are the reviewer's. Linked from
  README, START-HERE, the log index, history and the roadmap.
- Not edited: `docs/design/determinism-modes.md` (the ADR; Haseeb's decision).

## Sources
- Outside reviews: `reviews/fable-bigpicture-20261019.md` and `reviews/fable-nextsteps-20261019.md` in the foreman's
  workspace (read in full; curated copies in docs/reviews/).
- PR bodies and comments: `gh pr view N` for #79, #81, #82, #83, #86, #88, #89, #90 (and its close comment), #91, #92,
  #93, #96, #97, #98, #100, #101, #102, #71, #84. Where a PR body and its log entry differ, the PR body was used.
- H100: `~/.cache/axon-h100sd/pod-pod1/results/h100-1019-2058/` (`config.json`, `cells.jsonl`, `events.jsonl`,
  `cost.json`, `driver.log`, `gpu.csv`, `engines/*.log`), `pod-pod1/{setup-times.tsv, setup.log}`, the `sd_table.py`
  output, and axon-devstack `runbooks/h100-serving-showdown.md` (§1–5, 7, 8, 10, and its A/B note).
- Train: `axon-run ls --all` (job ids), `git diff 07c6782 c15c90e` (empty).
- Line counts at 764792f: `git ls-files '*.rs' | xargs wc -l` 89,986; `'*.cu' '*.cuh'` 8,013; `grep -rc '#\[test\]' src`
  677.

## How it was verified
| check | result |
|---|---|
| table medians re-derived from `cells.jsonl` for spot cells (Axon mix C = 1 / 128, classic C = 128; vLLM-AWQ / FP8 mix C = 128) | equal to `sd_table.py` |
| every ratio and range in the prose recomputed from the tables | by hand |
| relative links and anchors in the changed files | script, see the PR |
| cargo, builds, model runs | none (docs only) |

## Claims narrowed against the tables
The run's first summary (foreman's log, 10-19 21:37) said three things the tables don't support as written:
- "Best ITL p99 at C ≥ 32" holds only at mix C = 128. At mix C = 32 / 64, SGLang-FP8 is lower (24.3 / 65.8 vs 68.9 /
  74.0 ms), and on classic Axon's ITL p99 is the highest from C = 32 up.
- "3–4× behind at high concurrency": mix C = 128 is 2.0–3.8× (3.1× vs vLLM-AWQ, 3.8× vs vLLM-FP8); classic is
  1.8–3.2× (2.7× vs vLLM-AWQ).
- #33's wgmma GEMM is on main and takes some batched Q4_K products on sm_90a (those that fill a wave of 128 × 64
  tiles, and the fused gate+up), so "no wgmma" is written as "no decode-shaped wgmma / Marlin-class GEMM".
