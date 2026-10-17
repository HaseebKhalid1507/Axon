# Roadmap: a mixture-of-experts section  (docs only)

- Date: 2026-10-17. Branch: docs/roadmap-moe. Base: main 3c3da61 (after #58). Status: open.
- Pick / design: maintainer request to put the MoE plan on the roadmap in one place.

## What changed
- `docs/roadmap.md`: a new "Mixture-of-experts" section, placed before "Not planned". It covers why MoE matters,
  the plan in order (Qwen3-30B-A3B, then gpt-oss-20b, then DeepSeek-class / Qwen3-235B-A22B as "not scheduled"),
  the ordering and dependencies, the per-device exactness contract, and the MoE approaches that are not planned.
  The MoE bullet under "Not planned" pointed at "Scale item 7 below", which was wrong (the item is above). It now
  links to the new section, and so does Scale item 7's status cell.
- `README.md`: the "mixture-of-experts models … are refused" line now links to the roadmap section.

## Why
The MoE plan was spread over one Scale table row, one "Not planned" bullet and two research notes. Now one
section has it.

## Sources
Every number is copied from these notes, with their labels (all speed figures are ESTIMATED; none is MEASURED):
- `docs/research/r10-scope.md`: §1.3 (decode arithmetic, 35–55 tok/s; prefill streaming), §1.4 rows B2 and B6
  (scope, bits risk, builder-hours, gates, dependencies), §5 rank 7, §6 dependency graph.
- `docs/research/research-bigmodels.md`: §4.3 (gpt-oss-20b ~45 tok/s with every expert on the CPU), §5 P1
  (exactness per device, as the dense split's), §6 dead ends (lossy shortcuts, experts from disk, CPU attention).
- `docs/design/multi-gpu.md` and #34 for the pipeline parallel the "later" models would need.

Not from the sources: the DeepSeek-class (MLA) / Qwen3-235B-A22B line. It came from the maintainer's brief and is
labeled "not scheduled, no scope or estimates". The sources say nothing about non-GGUF formats, so the section
doesn't either. The ordering paragraph uses r10's own reason for the order: rank 7 comes after C2 and paged KV
because the ranking puts agent-swarm unblockers on 12 GB first. r10 lists B2's dependencies as "none on serve; the
streamer; not before Gemma 3 settles". It records no technical dependency on paged KV.

## Verification
Docs only. Every relative link and anchor in the changed files was checked to resolve. CI.
