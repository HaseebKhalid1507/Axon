#!/usr/bin/env bash
# Create or update the issue-board labels described in .github/labels.md.
# Idempotent: `gh label create --force` updates an existing label's color and description.
# Needs `gh` logged in with write access. Usage: scripts/sync-labels.sh [OWNER/REPO]
set -euo pipefail

repo="${1:-HaseebKhalid1507/Axon}"

# name|color|description  (keep in sync with .github/labels.md)
labels=(
  "agent-ready|0e8a16|Spec, scope and gates written. Claimable: comment 'claiming'."
  "needs-spec|fbca04|Proposal only. Not claimable until a maintainer writes the spec."
  "claimed|5319e7|Assigned. PR due within 48 h of assignment, or the claim lapses."
  "perf|d93f0b|Speed or memory. Needs base-vs-candidate numbers, pp and tg."
  "arch|1d76db|A model architecture: loading, forward pass, tokenizer, template."
  "quant|c5def5|Tensor / quant types and their kernels."
  "serve|006b75|axon serve: HTTP API, scheduler, slots, batching."
  "gpu|b60205|CUDA path. GPU gates required (maintainer-run)."
  "cpu|0052cc|CPU path: SIMD kernels, worker pool, repacks."
  "good-first-issue|7057ff|Small and well-bounded; a good first PR."
  "run-gpu|e99695|Maintainer-applied: run GPU gates on maintainer hardware."
  "blocked|000000|Waiting on another issue, PR or decision. Don't start or claim."
)

for entry in "${labels[@]}"; do
  IFS='|' read -r name color desc <<<"$entry"
  echo "label: $name"
  gh label create "$name" --repo "$repo" --color "$color" --description "$desc" --force
done
