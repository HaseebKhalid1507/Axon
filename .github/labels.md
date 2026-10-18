# Labels

The issue board is the work queue. `scripts/sync-labels.sh` creates or updates these labels (a maintainer
runs it; it uses `gh label create --force`). Edit this table and the script together: names, colors and
descriptions must match exactly, and GitHub caps a description at 100 characters.

| label | color | meaning |
|---|---|---|
| `agent-ready` | `#0e8a16` | Spec, scope and acceptance gates written. Claimable: comment 'claiming'. |
| `needs-spec` | `#fbca04` | Proposal only. Not claimable until a maintainer writes the spec. |
| `claimed` | `#5319e7` | Assigned to someone. A PR is due within 48 h of assignment, or the claim lapses. |
| `perf` | `#d93f0b` | Speed or memory. Needs base-vs-candidate numbers, pp and tg. |
| `arch` | `#1d76db` | A model architecture: loading, forward pass, tokenizer, chat template. |
| `quant` | `#c5def5` | Tensor / quant types and their kernels. |
| `serve` | `#006b75` | axon serve: HTTP API, scheduler, slots, batching. |
| `gpu` | `#b60205` | CUDA path. GPU gates required (maintainer-run). |
| `cpu` | `#0052cc` | CPU path: SIMD kernels, worker pool, repacks. |
| `good first issue` | `#7057ff` | Small and well-bounded; a good first PR. |
| `run-gpu` | `#e99695` | PRs only, maintainer-applied: run the GPU gates on a throwaway rented GPU pod once the diff is read. |
| `bug` | `#d73a4a` | Wrong output, crash, refusal, or a slowdown vs an earlier commit. |
| `blocked` | `#000000` | Waiting on another issue, PR or decision. Don't start or claim. |

Rules of thumb:

- An issue has exactly one of `needs-spec`, `agent-ready`, `claimed` (or none, once closed).
- Area labels (`perf`, `arch`, `quant`, `serve`, `gpu`, `cpu`) combine freely.
- Only maintainers add `agent-ready`, `claimed` and `run-gpu`.
- A `bug` is not claimable until a maintainer triages it into `agent-ready`.
