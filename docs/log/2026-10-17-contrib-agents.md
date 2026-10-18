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
| `bash -n scripts/sync-labels.sh`; labels.md, the script and the arguments it passes to `gh` (stubbed) have the same names, colors and descriptions; every description ≤ 100 characters (GitHub's cap) | ok, 13 labels |
| cargo / models / GPU | n/a: no code changed |

## Not done
- Labels not created on GitHub, repo settings unchanged: Haseeb runs `scripts/sync-labels.sh` when he decides.
- The `run-gpu` automation is a later PR; AGENTS.md describes only that maintainers run the GPU gates.

## Update 2026-10-17: review round 1 (shady, SHIP-AFTER-FIXES 8/10)
- `run-gpu` runs on a throwaway rented GPU pod, never on maintainer machines (Haseeb's 3f56d9d, then the script
  synced). Its description was shortened from the review's 102 characters to 100 to fit GitHub's limit.
- Label descriptions are now identical in labels.md and the script (`claimed`, `arch`, `run-gpu` had drifted);
  `bug` added, since bug.yml applies it. The table's backticks were dropped so the text matches byte for byte.
- The example PR body uses placeholders for driver, NVRTC, distro and kernel, which #60's log doesn't record.
- The md5 table is the same in AGENTS.md and the PR template; the template has one checkbox per line.
- change-request.yml: the gates field uses a placeholder, so an empty submit is blocked.
- AGENTS.md: ask for an extension on the issue if 48 h isn't enough.
