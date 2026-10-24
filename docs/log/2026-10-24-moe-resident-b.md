# One MoE layer's experts on the GPU, CPU-exact (PR #PRNUM)

- Date: 2026-10-24. Branch: feat/moe-resident-b (stacked on PR A #120, merged as 6e3b19c, which is merged in
  at 45888a5). Status: open (gates pending).
- Pick / design: Fable 10-19 §3 #3; spec `r16-pick4-moe-resident.md` §6 PR B. Also #121 items 1–4 (shady's
  follow-ups on #120).

Labels: **MEASURED** (job cited), **CODE**, **ESTIMATED**.

## What changed

Test-only: no model path launches the new kernels yet (that's PR C). Bits of every existing kernel are unchanged
(SASS below).

- **`src/cuda/moe_res.cu` (new, in `kernels_src!`'s base list):**
  - `moe_sort`: groups the nb·k (token, slot) pairs by expert, in ascending pair order within each expert, the
    same order as the CPU's counting sort. It cuts each expert's run into 64-token tiles. One block, warp ballots.
  - `moe_gather`: copies each pair's quantized token into sorted order.
  - `moe_gemm_tw_{q4_k,q5_0,q8_0,q6_k}`: grouped twin GEMMs over the tile list (`blockIdx.x` = tile). They're
    PR A's `gemm2_t<…, TW = true>`. `gemm2_t` gets a defaulted `zt` (token-tile override); every existing
    instantiation keeps `zt = -1`, so their SASS doesn't change.
  - `swiglu_quant_tw`: `swiglu_twin` then `quant_warp`, the CPU's `glu` then `QuantizedAct::for_types`.
  - `moe_sum`: the CPU's `combine` in rank order (`y = w0·d0`, then `y + wi·di`, a multiply then an add).
  - `moe_gemv_tw`: the decode expert GEMV over the k slots, in `gemv_tw`'s numerics. It runs on any arch.
- **`src/cuda/qwen2/experts.rs` (`#[cfg(test)]`):** `CudaExperts::upload(l)`, then `layer` (nb 1: GEMV; more:
  grouped), `batched` (with GPU event timing) and `layer_gemv` (token by token, the nb-invariance reference).
- **#121 items 1–4:**
  1. Host refusal `twin_gemm_ok(cc)`: twin GEMMs need sm_80+. Below that, gemm2's body is `__trap()`, which would
     poison the context. Unit test `twin_refusals`. On sm < 8.0, `gpu_twin_gemm2s_matches_cpu_r8` asserts the
     refusal and prints `SKIPPED (refused, as designed)`; before, it returned silently.
  2. Type contract: `twin_type(qt)` (Q4_K / Q5_0 / Q8_0 / Q6_K only, unit-tested against the other types) guards
     the runtime-typed `gemv_tw` path. `gemm2_t<…, TW = true>` `static_assert`s the twinned types (Q2_K and the MIN1
     types would have taken the plain epilogue silently).
  3. Epilogues: `gpu_twin_epilogues_match_cpu` checks bias, mode 1, and both together, on `gemv_tw` and every
     `gemm2s_*_tw`. All are bit-equal to `init + (row sum + bias)`.
  4. `fused`: `CudaExperts` reads `vmath::fused()` once at upload, and every `swiglu_twin` launch of the layer
     passes that field. The tests assert it equals the host's.
  - Item 5 (`--gpu-experts` refuses or falls back under `AXON_REPACK=0` or a mismatched F) belongs to PR C, where
    the flag is added. It stays open in #121.
- **`deepseek2::build_only` + `Qwen2::new_experts_of` (test):** repack only the layers under test, so the
  Moonlight test needs ~1.2 GB of r8 arena, not 11 GB.

## Why

PR B is the expert layer itself, checked bit for bit against the CPU before any placement or plumbing (PR C) can
make the model use it.

## How it was verified

GATES

## Results

BENCH

## What didn't work / notes

- The first fixture run (jade, 1020-164016-5dbf, 10-20) failed in the test itself, not in the numerics. The CPU
  repack of the in-memory fixture released its expert bytes before the GPU upload read them (gguf.rs: "released
  after repacking"). The fix is to upload first, then build the CPU experts. Its Moonlight "ok" was a skip
  (AXON_TEST_DS2 unset).
- 00205a2 ("WIP, not gated") describes the 10-20 state. It's superseded by the gates above; history is left as is.
- Moonlight runs only on bella: the remote nodes have no `models/` link for this worktree. avante and jade run the
  fixture.
- **Track kill rule (Fable):** if resident decode is < 1.5× the CPU-expert decode on bella, the track stops after
  this PR. It's decided in PR C's first bench: tg128 on bella, `--gpu-experts auto` vs 0.
