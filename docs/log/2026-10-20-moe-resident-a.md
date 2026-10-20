# Twin numerics primitives for resident MoE experts (PR #120)

- Date: 2026-10-20. Branch: feat/moe-resident-a. Base: main ab5568c (merged in at a21b121; the work itself was
  written on 05fb70c). Status: open (all gates green at a21b121, 2026-10-20 17:26).
- Pick / design: Fable's next-steps review 10-19, §3 #3 ("one MoE resident on the GPU"); spec
  `r16-pick4-moe-resident.md`, §6 PR A (step 0 in the same spec: S0.1, S0.3, S0.4 PASS).

Labels: **MEASURED** (job cited), **CODE**, **ESTIMATED**.

## What changed

No model behaviour change: nothing outside the tests launches the new kernels.

- **`src/cuda/moe_twin.cu` (new, in `kernels_src!`'s base list):** `exp_twin` / `swiglu_twin`, op-for-op ports
  of `vmath::exp_f::<F>` / `swiglu1_f::<F>`. F (the host's `vmath::fused()`) is a kernel argument: FMAs with it,
  a multiply then an add without, as on the CPU. Every float op is an explicit `_rn` intrinsic, so nvcc can't
  contract. Test handles: `exp_twin_k`, `swiglu_twin_k`.
- **Twin flag `TW` (CPU-order epilogue), default off:**
  - decode (`decode.cu`): `blk_q4_k` / `dot_rows_t` / `gv_chain_tw` take `TW`; new kernel `gemv_tw` covers Q4_K,
    Q6_K, Q5_0 and Q8_0, giving each row the r8 kernels' sum.
  - prefill (`prefill.cu`): `g2_compute` and `gemm2_t` take `TW` (after main's `BN` / `WR` parameters). New
    kernels `gemm2s_{q4_k,q5_0,q8_0,q6_k,q6k8}_tw` give each output r8 / r8q `gemm`'s sum for its token. The
    float steps are a multiply then an add, in the CPU's order: Q4_K `acc + ((d·dk)·S − (dmin·dk)·M)`, Q6_K
    `acc + (dk·dw)·isum`, Q5_0 / Q8_0 `acc + (dw·d0)·isum`.
  - Q6_K is included for qwen3moe / Qwen3-30B-A3B (Q6_K down in Q4_K_M).
- **`src/quant.rs`:** `#[cfg(test)] QuantizedAct::views()`, so the tests can compare the GPU quantizer's bytes.
- **Tests (`src/cuda/qwen2/tests/twin.rs`, 4 new):**
  - `gpu_twin_quant_matches_cpu`: decode and batched quantizer bits vs `QuantizedAct`.
  - `gpu_twin_exp_swiglu_match_cpu`: 1.25M inputs, F true and false, plus `vmath::swiglu`.
  - `gpu_twin_gemv_matches_cpu_r8`: twin gemv vs CPU `r8` / `r8q` `gemv`.
  - `gpu_twin_gemm2s_matches_cpu_r8`: twin gemm2s vs CPU `gemm` at nb 1 / 7 / 48 / 129 / 1024.
  - Shapes: Moonlight and Qwen3-30B-A3B expert shapes plus ragged ones.
  - Each test also checks that the *plain* kernel differs somewhere, so a twin that silently fell back to the plain
    epilogue would fail.
  - sm < 8.0 (jade, sm_61): `gpu_twin_gemm2s_matches_cpu_r8` skips, because the gemm2 family has no tensor cores
    there. The gemv twin runs and passes.

## Why

Resident experts are worth having only if they don't change the bits. With twins, which layers sit in VRAM
(`--ctx`, free memory, `--gpu-experts N`) never changes an output, so residency stays out of the identity and
`NUMERICS_VERSION` doesn't move (spec §3–4). PR A is the numerics alone, testable on synthetic matrices, before
any expert-layer kernel exists.

## How it was verified

| gate | result | job ID / log |
|---|---|---|
| twin tests, bella sm_120 | PASS 4/0/0 | 1020-162850-7c8a |
| twin tests, avante sm_89 | PASS 4/0/0 | 1020-162853-5647 |
| twin tests, jade sm_61 | PASS 4/4 (gemm2s skipped: sm < 8.0) | 1020-162900-ea13 |
| fast tier release suite (2 shards) | PASS 289/0/31 (avante) + 342/0/16 (bella) | 1020-162900-74db, 1020-165512-1f8b (resubmit of 6eb0, OOM-killed twice by its scope during parallel chat:: tests) |
| fast tier debug-assertion suite, CPU queue (2 shards) | PASS 381/0/18 + 266/0/29 | group 1020-162900-73f3 (75b7, 8b31) |
| md5 of `gen`, main ab5568c vs a21b121, same job, cuda, 6 dense models × 5 prompts, -n 64 | 30/30 identical | 1020-162854-8999 |
| SASS: nvcc offline, NVRTC opts, main ab5568c's sources vs a21b121's (cuobjdump -sass, addresses stripped) | existing kernels identical: 156/156 on sm_120 and sm_89 (plain and EXT) and on sm_75 (stand-in for jade: nvcc 13 has no sm_61); `KERNELS_OSS` 160/160 on sm_120; 8 new kernels | local, `~/.cache/axon-moeres/cu/a21/` |
| clippy-driver -D warnings (lib, `--test`), rustfmt --check | clean | local |

Earlier, on the 05fb70c base (wip3, before the merge), the twin tests passed 4/4 on jade (1019-013243-96dd),
avante (1019-013243-6cba) and bella (1019-013244-12d4). The plain kernels differ from the CPU there: gemv1 on
10,529 rows, gemm2s on 252,574 outputs. So the tests have teeth.

## Results

n/a: no behaviour change, nothing to time. Per-kernel speed of the twins vs the plain kernels is measured in
PR B, where they run on real expert shapes.

## Track kill rule (Fable)

If resident decode is < 1.5× the CPU-expert decode on bella, the track stops at PR B. It isn't decided here.
PR B measures per-layer GPU time; the decode comparison needs PR C's plumbing.

## What didn't work / open

- Launched on sm < 8.0, `gemm2s_*_tw` compiles to an empty body (`#if __CUDA_ARCH__ >= 800`, the same as every
  gemm2 kernel) and would leave its output unwritten. Nothing launches it in this PR. PR B/C must refuse twin GEMMs
  below sm_80 at the call site: take the per-token `gemv_tw` there, or keep those experts on the CPU.
