# Contributor contract: AGENTS.md, issue forms, PR template, labels  (PR #74)

- Date: 2026-10-17. Branch: docs/contrib-agents. Base: main 5a73f63. Status: open
- Pick / design: Haseeb, to open Axon to outside contributors (people and their agents) with the issue board as the work queue.

## What changed
Docs and config only, no Rust. `AGENTS.md` (workflow, hard rules, measuring, required PR evidence with an
example adapted from #60, GPU gates, review), `.github/ISSUE_TEMPLATE/` (change request, bug, config with blank
issues off), `.github/pull_request_template.md` (nine required evidence sections), `.github/labels.md` and
`scripts/sync-labels.sh` (not run), README "Contributing".

## Why
The repo goes public later. Outside agents follow written rules literally, so the rules the project already
works by (bit identity, md5 gate, interleaved A/B with pp and tg, maintainer-only merge) are written down in one
place, and every PR has to carry the evidence a maintainer needs to rerun it.

## How it was verified
| gate | result |
|---|---|
| relative links and anchors in the changed files | all resolve (script in the PR body) |
| issue forms parse as YAML | ok |
| `bash -n scripts/sync-labels.sh`; labels.md and script list the same names and colors | ok |
| cargo / models / GPU | n/a: no code changed |

## Not done
- Labels not created on GitHub, repo settings unchanged: Haseeb runs `scripts/sync-labels.sh` when he decides.
- The `run-gpu` automation is a later PR; AGENTS.md describes only that maintainers run the GPU gates.
