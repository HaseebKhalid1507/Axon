# Hopper (H100) prefill: why it trails, and a wgmma/TMA path

**Status: design. The first pod experiment (§6, E0–E2) is pending; nothing Hopper-specific is on main.**

Written against main `f8d8a86`; the files it cites are unchanged in kind on `5f96039`. Labels:
**MEASURED** (the RunPod board in [../benchmarks-gpu.md](../benchmarks-gpu.md), earlier Axon profiling
runs, or the cited external microbenchmark), **CODE** (read in the tree), **ESTIMATED** (arithmetic
shown).

## 1. Summary

1. **The gap is the GEMM, and the GEMM is instruction-issue-bound.** The cause is not tensor math,
   clocks alone, or power. 7B pp2100 is 6.5 GMAC per token, so the H100 sustains 81.5 TMAC/s
   end-to-end, against 90 on the 4090 and 121 on the 5090 (MEASURED tok/s × MACs).
   - Per SM·GHz the H100 is *ahead* (48 vs 38 / 37 tok/s). It loses only on raw SM-clock count:
     132 SMs × 1.98 GHz against 128 × 2.82 (4090) and 170 × 2.96 (5090).
   - Hopper's tensor pipe does 4× Ada's int8 MACs per SM-clock, and Axon's kernel never uses that.
     Each 64×64×256 stage issues ~1400 non-tensor instructions (MEASURED from SASS) against ~290 clocks
     of tensor work at wgmma rate.
   - The H100 runs at **7.6% of its int8 peak**, against 23–27% on the consumer cards.
2. **mma.sync on Hopper is capped at ~65% of wgmma throughput.** MEASURED by Luo et al. 2024,
   arXiv:2402.13499, on H800: int8 m16n8k32 at 978 TOPS vs wgmma m64n256k32 at 1449 TOPS. That cap
   still sits ~8.5× above what Axon reaches, so instruction count is the binding constraint today.
   wgmma matters because it reads both operands from shared memory via descriptors (no fragment loads
   or nibble unpacking in the math warps), and TMA removes the copy-issue cost.
3. **A bit-identical Hopper GEMM exists** (§4). The integer block sums stay exact in int32, and the
   float epilogue is unchanged.
4. **ESTIMATED result:** Q4_K GEMMs go from ~85 to 180–300 TMAC/s, and 7B pp2100 from 12.5k to
   **22–32k tok/s** (1.8–2.6×), above the 5090's 18.6k. Decode is untouched.
   - It is a datacenter-only path: `sm_90a` (H100 / H200 / H20 / GH200).
   - Parabellum's sm_120 can't run it, so local work is compile-only and runs happen on pods.

## 2. Roofline (7B pp2100)

MACs per token (CODE shapes; dim 3584, ffn 18944, 28 layers): 6.53 G for the projections, plus
0.21 G of attention at a mean position of 1050.

| GPU | SMs × clock | int8 dense peak | 7B pp2100 tok/s | e2e TMAC/s | % of peak | tok/s per SM·GHz |
|---|---|---:|---:|---:|---:|---:|
| H100 SXM | 132 × 1980 MHz | 1070 (wgmma) | 12,534 | 81.5 | 7.6% | **48.0** |
| RTX 4090 | 128 × 2820 | 370 | 13,842 | 90.0 | 24% | 38.3 |
| RTX 5090 | 170 × 2962 | 516 | 18,609 | 121 | 23% | 37.0 |
| RTX 5070 Ti Laptop (Parabellum) | 46 × ~2350 | 116.8 (measured) | 4,898 | 31.8 | 27% | 45.3 |

Where the H100's ~168 ms per 2100 tokens goes (ESTIMATED by scaling the MEASURED split on Parabellum,
which is 90% GEMM, 7% attention and 3% elementwise):
- elementwise ~2.6 ms;
- attention ~5 ms;
- **GEMM ≈ 160 ms, 95% of the pass, at ≈ 85 TMAC/s**;
- launch gaps < 1%.

Power is not the limit: the H100 ran at 409 W of 700 W, at a flat 1980 MHz (MEASURED).

Issue budget per warp per 64×64×256 stage (MEASURED on SASS):

| item | count |
|---|---:|
| tensor clocks, Ada-class | ~1150 |
| tensor clocks, Hopper mma.sync | ~445 |
| tensor clocks, wgmma rate | ~290 |
| non-tensor issue per warp | 590–700 |
| non-tensor issue per sub-partition (2 warps) | ~1400 |

The non-tensor work is integer scale multiplies, nibble unpacks, shared-memory loads, async copies with
their addressing, and the epilogue. **The tensor pipe idles at least 3 of every 4 clocks.**

## 3. NVRTC can emit wgmma, setmaxnreg and TMA: requirements

- **Arch string `sm_90a`.** NVRTC ≥ 12.0 accepts it, but `nvrtcGetSupportedArchs` reports `90`.
  `pick_target` (CODE, `cuda/mod.rs`) must therefore special-case cc 9.0. wgmma, setmaxnreg and
  `cp.async.bulk.tensor` are rejected for plain `sm_90`, so the source goes behind
  `__CUDA_ARCH_FEAT_SM90_ALL`.
  - An `sm_90a` cubin loads only on cc 9.0.
  - Blackwell (sm_100 / sm_120) has no wgmma, so this is strictly a Hopper path. `gemm2` stays
    everywhere else.
- **PTX ISA 8.0 = CUDA 12.0** covers everything needed. NVRTC emits a cubin, and `check_versions`
  already enforces NVRTC major ≤ driver major (CODE). The H100 pod (driver 570.195, NVRTC 12.8)
  qualifies.
- **TMA:**
  - `cuTensorMapEncodeTiled` is one more optional `dlsym` in `ffi.rs`. If it is missing, the Hopper
    path is disabled.
  - Q4_K weights: a u8 rank-2 map with box {144 B, rows}.
  - Activations: a 128-B swizzled box.
  - Each map goes to the kernel by value as a 64-B-aligned `__grid_constant__` struct, which needs a
    new 128-byte kernel-argument kind.

**Pitfalls, each with a planned test:**
1. ptxas "wgmma … serialized" (C7510) silently costs 3–5×. The compile test fails if the NVRTC log
   contains "serialized".
2. A missing `fence.proxy.async` between producer `st.shared` and wgmma reads gives rare,
   nondeterministic bit mismatches.
3. Wrong smem-descriptor fields give garbage without a fault. Test against `gemv_ref` first. int8
   wgmma is K-major only, which our weights and activations already are.
4. `setmaxnreg` register budget: 2 × 128 × 232 + 128 × 40 ≤ 64K.
5. mbarrier parity bugs hang the GPU. Develop under a timeout, and test odd stage counts and ragged
   shapes.

## 4. Bit identity: what stays fixed

The contract (CODE, `prefill.cu` header and `gemv_ref`) is defined per output and per 256-K block *b*:

- `S = Σ_j sc_j · sumi_j` and `M = Σ_j m_j · bsum_j` (integers);
- `v = d·S − dmin·M`;
- `acc = fma(dk_b, v, acc)`, with *b* ascending from 0.

Any exact way of computing the integers is allowed.

**Integer part.** Q4_K's 6-bit sub-block scale times a 4-bit quant doesn't fit int8. So:
- Split `sc = lo4 + 16·hi2`.
- The producer pre-scales the quants to u8: `q·lo4 ≤ 225`, `q·hi2 ≤ 45`.
- Run two accumulation passes with one exact `D *= 16` between them: `S = 16·D_hi + D_lo`.
- wgmma accumulates int32 exactly, and |S| ≤ 30.7 M, the same bound as today.

Other formats:
- Q5_K uses a 3+3 split.
- Q6_K (int8 copy) takes 4 passes of 2-bit scale pieces with a Horner combine.

The cost is 2.06× the tensor work for Q4_K and 4.06× for Q6_K. That is still ≥ 3× today's effective
rate.

**Float part: byte-for-byte the same code.** The same `h2f`, `cvt.rn`, `__fmul_rn` / `__fsub_rn` /
`__fmaf_rn` in the same order, `--ftz=false`, no fast-math. TMA and wgmma only move bytes and multiply
integers. **No float op is added or removed**, so `gemv_ref` equality holds by construction, and
decode == prefill == verify survives.

Formats with 32-value blocks (Q8_0, Q5_0, Q4_0, Q4_1, Q5_1) need a float op every 32 K, so wgmma gains
nothing there. They stay on `gemm2`.

## 5. Kernel sketch: `gemmh_*` (sm_90a only)

- Warp-specialized CTA of 384 threads: 1 producer warpgroup and 2 consumer warpgroups.
- CTA tile of 128 rows × 64 tokens, `wgmma.m64n64k32` per consumer warpgroup, 128-K stages in a
  3-stage mbarrier ring, ~160 KB of shared memory.
- **Producer:** one TMA per stage for the raw weights (not one per 144-B row, a mistake an earlier
  attempt made at −55%) and one for the activations. It then unpacks and pre-scales the weights, at
  ~15% of issue, and finishes with a proxy fence and an arrive.
- **Consumers, ping-pong:** one warpgroup runs its hi/lo wgmma batch while the other runs the previous
  block's integer-to-float epilogue.
- **Budget per 128×64×256 block (ESTIMATED):**
  - ~1060 tensor clocks against ~820 issue slots, so it is tensor-bound;
  - expect 55–65% tensor utilization;
  - L2 bandwidth is the next wall, at ~300 TMAC/s.
- **Rates (ESTIMATED):**
  - Q4_K: 180–300 TMAC/s.
  - Q6_K copy: 100–160 TMAC/s.
  - 7B overall: GEMM 160 ms → 57–85 ms, so **22.7k–32.5k tok/s**. Decode (311 tok/s) is unchanged.
- **Host:** use `gemmh` when the module is Hopper, the batch is larger than the small-batch threshold,
  and the type is Q4_K / Q5_K / Q6_K. The fused gate/up kernel gets a twin.

## 6. Staged plan and gates

**E0 + E1 + E2: one H100 pod, ~1 h at the pod's $3.99/h.** The test binary is built locally, and NVRTC
compiles on the pod.

- **E0, baseline profile (15 min).** Run `cuda_gemm_bench` and `cuda_prefill_profile` (existing
  tests), with a clock/power trace alongside. Predictions:
  - `mma_peak` 650–720 TMAC/s. If it is ≥ 950, Hopper mma.sync is full-rate and §1.2 is wrong; §1.1
    still stands.
  - Q4_K gate/up at 80–100 TMAC/s.
  - GEMM ≥ 90% of the pass.
  - A flat 1980 MHz, < 500 W, no throttling.
- **E1, `hopper_check` (25 min).** A TMA 2-D load, one mbarrier and one int8 wgmma from descriptors,
  checked against a CPU integer GEMM. **Gate:** compiles with no "serialized" warning, loads, and the
  bits are equal.
- **E2, `wg_peak` (15 min).** A wgmma loop on static shared memory. Prediction: ≥ 1000 TMAC/s.
- **Go/no-go:**
  - **GO** if the GEMM share is ≥ 85%, E1 passes, and `wg_peak` ≥ 1.3 × `mma_peak`.
  - If `mma_peak` ≈ `wg_peak`, still go, because TMA and smem-sourced operands remove the issue
    overhead either way, but expect the 180 TMAC/s floor.
  - **NO-GO** only if E1 can't be made to pass through NVRTC and the dlopen'd driver.

**S1, plumbing (local, ~1 day).** An optional `cuTensorMapEncodeTiled`, an `sm_90a` target (with an
`AXON_HOPPER=0` opt-out), `Module::hopper()`, and the 128-byte argument kind. `sm_90a` is added to the
compile-only test of old architectures. **Gate:** existing tests stay green on sm_120, and sm_90a
compiles clean.

**S2, `gemmh_q4_k` v1 (pod, ~2 h).** One consumer warpgroup, 64×64, 2 stages. Bit tests over ragged
rows, token counts and K sizes, plus prefill, verify and PDL on/off identity. **Gate:** bits equal
everywhere and ≥ 120 TMAC/s. Otherwise stop and write it up.

**S3, full kernel (pod, ~3 h).** Ping-pong, 3 stages, the Q6_K and gate/up twins, then the scoreboard.
**Gate:**
- 7B pp2100 ≥ 20k tok/s and Qwen3-8B ≥ 18k;
- decode within ±2%;
- all bit tests pass;
- no change on the 4090 or 5090.

**S4, tuning (optional).** 2-CTA cluster multicast of activations and 128×128 tiles. Target: 7B pp2100
≥ 25k.

## 7. Risks

- L2 bandwidth may cap Q4_K at ~250 TMAC/s before issue does. The fix is S4 multicast.
- Register spills in the consumer warpgroup. The fallback computes M with scalar integer ops, or uses
  smaller tiles.
- 4-pass Q6_K may land near today's rate. That is acceptable, since Q6_K is ~15% of 7B's MACs.
- A driver without `cuTensorMapEncodeTiled` disables the path and falls back to `gemm2`.
- Every run needs pod time, so batch runs behind compile-only gates on Parabellum.
- B200 (sm_100a) would need a tcgen05 twin. The producer/consumer structure and the §4 arithmetic carry
  over; the mma instructions don't.
