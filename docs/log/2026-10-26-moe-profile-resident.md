# AXON_MOE_PROFILE with resident experts: no more `inf / NaN` (fix)

- Date: 2026-10-26. Branch: fix/moe-profile-resident. Track: roadmap item 3 (T2b's diagnosis needed it).

## What changed
`cuda/qwen2/moe.rs`:
- `moe_step` counts every MoE layer's decode step, resident layers included. A resident layer's experts are inside
  its GPU segment, so its CPU columns stay 0.
- The report's lines come from `profile_lines`, which skips a layer with no step counted.
- Nothing changes without `AXON_MOE_PROFILE=1`.

## Why
The step count only grew in the host round trip, which a resident layer (`--gpu-experts`) skips. Its 0 steps made
`profile_report` divide by zero: `gpu inf, down NaN, …` for every partial placement. MEASURED in job
1026-222632-d7c1 (Qwen3-30B-A3B at `--gpu-experts 25`). With every layer resident the report never printed, because
the last layer's round trip triggered it.

## How it was verified
- `moe_profile_report_resident_layer`: finite numbers, the empty layer skipped, the sum equal to the parts.
- A GPU smoke run: `--gpu-experts` partial and `auto` with `AXON_MOE_PROFILE=1`, the report finite and printed.
- The fast tier and the md5 gate; job ids are in the PR.
