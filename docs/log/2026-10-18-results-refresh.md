# results.md, history.md and README brought up to main 4507c75  (docs only)

- Date: 2026-10-18. Branch: docs/results-2026-10-18. Base: main 4507c75 (#87). Status: open.
- Pick / design: maintainer request. `docs/results.md` still said "Main is a591e17" and its caveat table stopped
  at #42; `docs/history.md` stopped at a591e17 with #45, #46 and lcdec listed as open.

## What changed
- `docs/results.md`:
  - Header: main `4507c75` (2026-10-18 06:05 EDT). The "none of the boards was run on it" banner stays.
  - Caveat table: one row per merged PR since a591e17 that can change speed or output on some path (#45, #48,
    #49, #50, #51, #52, #55, #56, #57, #60, #61, #63, #64, #67, #70, #72, #75, #78), each effect copied from
    the PR body and labelled **MEASURED** or **not measured**. One combined row for code with no speed claim and
    unchanged md5 (#54, #62, #68, #76), one for docs/tests/scripts (#46, #47, #53, #58, #59, #65, #66, #69, #73,
    #74, #77, #80, #85, #87).
  - Long context: the "lcdec in flight, +12 % to +30 % on a 5090 pod" sentence was wrong after #50 (merged; it
    withdrew those pod3 numbers). Replaced with #50's own bella numbers and #56's prefill-at-depth comparison.
    The table itself is unchanged and still from r9.
  - New section "New models and types since the boards": #63, #67, #78, #72 vs llama.cpp, from their PRs.
  - Known losses: as of 4507c75; added #72's CPU prefill and GPU losses, #67's MoE prefill vs `--cpu-moe`,
    #55's bella mid-admit ITL p99, and that MoE / gpt-oss have only dev-model or CPU numbers.
- `docs/history.md`: Phase 17 (a591e17 → 69adfa2, 10-16 16:55 → 10-17 17:22) and Phase 18 (the night of
  10-17/18, 5a73f63 → 4507c75), one row per PR. Status updates on #45, #46, lcdec and the r10 tracks; #45's row
  now gives the merged version's numbers next to the first revision's.
- `README.md`: the opening line names Qwen3 MoE and gpt-oss; serve's bullet names the Responses and ollama APIs;
  "Not on main yet" no longer lists `/v1/responses`, the ollama API or MoE on the GPU (#70, #67 merged); line
  counts (`git ls-files '*.rs'` / `'*.cu' '*.cuh'` + `wc -l` at 4507c75: 76,182 and 7,096); docs/testing.md
  in the docs list. The model and tensor-type lists were already current (#67, #72, #78 updated them).
- `docs/log/README.md`: says history.md now summarizes the merges to 10-18.

## Sources
`git log --first-parent a591e17..origin/main` (36 merges, #46 … #87) and `gh pr view N --json body` for each.
Where a PR body and its log entry differ, the PR body was used. No number was computed for this page except the
line counts; ranges are the PR's own min/max or the union of its rows.

## How it was verified
| check | result |
|---|---|
| every new number traced to a PR body | by hand, row by row |
| relative links and anchors in the changed files | script, see the PR |
| cargo, model runs | none (docs only) |
