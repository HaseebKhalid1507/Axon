# AXON_MOE_PROFILE with resident experts: no more `inf / NaN` (fix)

- Date: 2026-10-26. Branch: fix/moe-profile-resident. Track: roadmap item 3 (T2b's diagnosis needed it).

## What changed
`cuda/qwen2/moe.rs`, profiling only (`AXON_MOE_PROFILE=1`):
- **`prof_step`** is the profiler's single write point. Per MoE layer and decode step it adds the synced GPU
  segment time, the host round trip's four CPU times (none for a resident layer, whose CPU columns stay 0), and
  the step count. It returns "report now" on the last layer's `PROFILE_EVERY`-th step. Only `moe_step` calls it.
- **`moe_round_trip`** returns its four times instead of adding them. So a 1-token batched pass (`moe_tokens`: a
  1-token prompt, verify or suffix) no longer adds CPU time to the profile without a step (review round 1).
- **`profile_lines`:** the report as lines; the footer counts the layers it shows.
- Nothing changes without `AXON_MOE_PROFILE=1`.

## Why
The step count only grew in the host round trip, which a resident layer (`--gpu-experts`) skips. Its 0 steps made
`profile_report` divide by zero: `gpu inf, down NaN, …` for every partial placement. MEASURED in job
1026-222632-d7c1 (Qwen3-30B-A3B at `--gpu-experts 25`). With every layer resident the report never printed, because
the last layer's round trip triggered it.

## How it was verified
- **`moe_profile_counts_every_layer`:** runs `prof_step` over `PROFILE_EVERY` decode steps of 3 layers, for
  resident-only, offloaded-only and mixed placements. Every layer counts every step, the report fires once on the
  last layer's last step, and its lines are finite with resident layers' CPU columns 0.
- **`moe_profile_lines_sum`:** the report's arithmetic.
- **GPU smoke:** `AXON_MOE_PROFILE=1`, 4x0.6B, `--gpu-experts 14 / auto / 0`. The report prints in all three
  placements (`profsmoke.sh` shows the last report of each run) with no `inf` / `NaN`.
- The fast tier and the md5 gate; job ids are in the PR.
