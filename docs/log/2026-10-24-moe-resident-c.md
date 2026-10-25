# Resident MoE experts in the model: `--gpu-experts N|auto|0` (PR #127)

- Date: 2026-10-24 (updated 10-25). Branch: feat/moe-resident-c, base `main`. PR B (#122) is merged to main as b2068e7;
  main was merged into this branch at c3cce85 (b2068e7) and 3c7ca02 (fb749cc). No rebase.
  Status: open, draft. Gates 8/8 PASS at 18427b2; review round 1 fixes at b20c301 (gates pending); **the kill-rule
  bench has no clean measurement yet** (the clean run c579 was parked for desktop GPU taint).
- Pick / design: Fable 10-19 §3 #3; spec `r16-pick4-moe-resident.md` §6 PR C; #121 item 5.

Labels: **MEASURED** (job cited), **CODE**, **ESTIMATED**.

## What changed

- **Default `--gpu-experts 0`: today's behaviour, the same code path.** It stays the default until the kill-rule bench
  passes.
- **The flag:** `--gpu-experts N|auto` (gen, chat; `AXON_GPU_EXPERTS` everywhere, serve too). It keeps deepseek2's
  routed experts of N MoE layers (auto: as many as fit) resident on the GPU.
  - Placement is ctx first: a probe cache of the planned context, then the free VRAM, less the experts' scratch and
    512 MiB headroom.
  - Smallest layers first, ties to the lower layer (`experts::place`).
- **CPU arena:** built without the resident layers (`Qwen2::new_experts_except`: their experts are `Weight::absent`,
  never read). Their tensors are uploaded from the mapping before the build, then released (b20c301). The RAM check
  (`check_ram_keep`) sums the kept layers' own repacked bytes.
- **Decode (`mla_step`):** a resident layer skips the host round trip. Its experts (`CudaExperts::enqueue`: no sync,
  no allocation) go into the same per-layer graph as the shared experts. They read the router's output in `mb.r`
  and write `mb.y`, which the next segment adds, exactly where the CPU's `y` was added. So the float ops on `x` are
  the same, in the same order.
- **Batched passes (`mla_tokens`):** the same, without graphs.
- **Refusals that fall back to 0, with the reason in the banner:**
  - non-deepseek2 MoE;
  - sm < 8.0 (`twin_gemm_ok`);
  - `AXON_REPACK=0`, or a twin F that isn't the host's `vmath::fused()` (`twin_identity`: #121 item 5).
  - The bits can't move, so the identity doesn't change and `NUMERICS_VERSION` stays.
- **Banner:** `moe: … , N experts x M layers on the CPU (top-k); R of 26 expert layers resident on the GPU (X GiB)`.

## Why

Step 0 (S0.3, MEASURED 10-19): CPU experts are 11.85 of ~15.5 ms per Moonlight decode step on bella. Keeping
experts on the GPU removes the round trip and the CPU work for those layers.

## How it was verified

Gates of record for the review, at **18427b2**: 8/8 PASS (MEASURED; [logs](https://github.com/HaseebKhalid1507/Axon/tree/121f334862d2e43f904b4d4cbe336b6b3425fdba/2026-10/moeresc-18427b2)).

| check | host | job | result |
|---|---|---|---|
| experts / twin / ds2_resident / fixture bits / load_tests | jade sm_61 | 1025-054105-b03b | PASS 16/0/1; resident paths SKIPPED-refused (sm_61), visibly |
| same, resident fixture | avante sm_89 | 1025-054105-444b | PASS 16/0/1; fixture 0/1/2/auto bits equal, 4 option sets |
| same + Moonlight 0/6/12/auto (`AXON_TEST_DS2=1`) | bella sm_120 | 1025-054105-d006 | PASS 16/0/1; bits equal, auto 24 of 26 layers (8.12 GiB) |
| full release, 2 shards | bella | group 1025-054126-6631 | PASS 660/0/48 |
| full debug assertions, 2 shards | bella | group 1025-054126-9148 | PASS 660/0/48 |
| md5 vs main b2068e7 | bella | 1025-054126-42be | 30/30 dense cuda; Moonlight 3/3, main = `--gpu-experts 0` = `auto` |

Review round 1 (shady 8/10, at 18427b2). Fixes at **b20c301**:
- The resident layers' host expert bytes are released after the upload, and the CPU half no longer builds those
  experts at all. Before, it built in-place views over them, which would read released bytes as soon as anyone
  released them, or under #134's STRICT_RELEASE.
- The Moonlight gate asserts that N layers were placed (N(6)/N(12)) and that auto places at least one. Before, it
  could pass with none placed.
- `check_ram_keep` uses the kept layers' own bytes instead of the average. The average under-counted, because
  placement moves the smallest layers to the GPU.
- serve refuses `--parallel N > 1` for MoE before the placement. A MoE backend serves one request at a time, so the
  refusal used to come only after GBs of experts were uploaded.

Gates at b20c301: res bella 1025-083916-0d50; full release (group 2f06) and debug (group 596a); md5 vs main fb749cc
(8b7a); the res suite with #134 merged (strict release) on bella d1d2 and avante b482. Pending when this was written.

## Results: the kill-rule bench

The rule: resident decode must be ≥ 1.5× the CPU-expert decode on bella, or the track stops.
Moonlight-16B-A3B Q4_K_M, `gen -n 128`, ctx 4096, greedy, `--gpu-experts 0` vs `auto`, 3 interleaved rounds.

- **No clean measurement yet.** The clean run 1025-054130-c579 (18427b2, `--timing bella`) was parked by the lead
  because of desktop GPU load. It will be rerun at b20c301 when bella is quiet.
- The only number so far is **TAINTED**: 1024-191040-267a (0685079; foreign GPU mean 9.2 %, max 16 %). Medians:
  decode 51.97 → 108.95 tok/s (2.10×), prompt eval 308.7 → 862.0 tok/s (2.79×), text identical, auto 24 of 26
  layers. It shows the effect is large; it is not the rule's verdict.

## What didn't work / open

- The per-layer graphs aren't merged across runs of resident layers. Each MoE layer still has two graph launches (its
  segment, then shared + resident experts), but no host sync. Merging them is the next lever if launch overhead shows
  up in the profile.
- qwen3moe / gpt-oss residency: refused for now (`moe_step` has no resident path; gpt-oss's expert biases aren't
  supported, see #122).
- `moe_sort` is one block: fine at pass sizes up to the cache's batch max.
- serve: a MoE model serves one request at a time (`--parallel N > 1` is refused before placement). A future MoE slot
  pool must budget each slot's KV and `ResBufs` in `place_resident`.
