# Engineering history docs (PR #47, docs only)

- Date: 2026-10-16. Branch: docs/engineering-log. Base: main a591e17. Status: open
- Pick / design: Haseeb asked for a record in the repo of what was done, why, how it was measured and what
  happened, including what failed.

## What changed
New: `docs/methodology.md`, `docs/history.md`, `docs/experiments-rejected.md`, `docs/results.md`,
`docs/research/` (six round-10 research reports + `r10-scope.md` + an index), `docs/log/` (this convention).
Links from `docs/START-HERE.md` and `README.md`. The stale "not on main" lines in both (Gemma, batching,
multi-GPU) are corrected. No code changes.

## Why
Until now the reasoning and the numbers lived in the foreman's workspace (PROGRESS.md, specs, builders'
NOTES.md, scoreboards) and in PR bodies. None of that is versioned with the code, and the rejected
experiments weren't written down in the repo at all.

## How it was verified
Every number was taken from a PR body, `git log`, PROGRESS.md, a spec, a builder's NOTES.md or a scoreboard
file, and cross-checked where two sources covered the same thing. Where sources disagree or a gate was
pending at merge time, the docs say so inline. No builds, tests or GPU runs (docs only).

## Results
n/a.

## What didn't work / open
- No board has been run on main a591e17. `results.md` lists the merges since the last boards.
- Not verified here: the forum's critic threads beyond what PRs and PROGRESS.md quote; PR comments other than
  those on #18, #32, #35 and #36.
