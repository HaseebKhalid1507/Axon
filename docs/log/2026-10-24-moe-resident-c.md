# Resident MoE experts in the model: `--gpu-experts N|auto|0` (PR #PRNUM)

- Date: 2026-10-24. Branch: feat/moe-resident-c, stacked on PR B (#122, feat/moe-resident-b; merged in at 73f5c79).
  Status: open, draft (gates and the kill-rule bench pending: bella held).
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
- **CPU arena:** built without the resident layers (`Qwen2::new_experts_except`; RAM check `check_ram_keep`).
  Their tensors are uploaded from the mapping before the repack.
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

GATES

## Results: the kill-rule bench

BENCH

## What didn't work / open

- The per-layer graphs aren't merged across runs of resident layers. Each MoE layer still has two graph launches (its
  segment, then shared + resident experts), but no host sync. Merging them is the next lever if launch overhead shows
  up in the profile.
- qwen3moe / gpt-oss residency: refused for now (`moe_step` has no resident path; gpt-oss's expert biases aren't
  supported, see #122).
- `moe_sort` is one block: fine at pass sizes up to the cache's batch max.
- serve with several caches: each cache gets its own expert scratch (`ResBufs`), but the placement only probes for
  one cache of the planned context.
