# Exact W4A8 GEMM on Hopper: how to get from 84–93 to ~200 TMAC/s

Research brief for roadmap item 2 ("Hopper exact GEMM, round 2"). 2026-10-24, by Joestar (subagent). Read-only on
the Axon repo at main (`src/cuda/hopper.cu`, `prefill.cu`, `decode.cu`, `docs/design/hopper.md`,
`docs/design/determinism-modes.md`, `docs/roadmap.md`, `docs/reviews/2026-10-19-outside-review-next-steps.md`,
`docs/experiments-rejected.md`, `docs/history.md` #33). No builds, no jobs.

Labels: **MEASURED** (someone measured it; cited), **CODE** (read in source; file cited), **ESTIMATED** (arithmetic
shown), **OPINION** (mine). Every external claim names the file or paper I read. Where I could not read it, it says so.

---

## 0. Summary (the ten lines)

1. Nobody ships an *exact-integer* W4A8 wgmma GEMM. CUTLASS's Hopper mixed-input supports `{fp16,bf16}×{int8,int4,int2}`
   and `fp8×int4` only (CODE, example 55 README); vLLM's Machete W4A8 is still `# TODO: Support W4A8 when ready` (CODE,
   `generate.py`); TensorRT-LLM's "W4A8" on Hopper is `e4m3 × int4` through CUTLASS (float accumulate), and its QServe port
   is `mma.sync.m16n8k32.s8` (CODE). LiquidGEMM (ByteDance, arXiv:2509.01229) is the only published wgmma W4A8-int8
   kernel; it folds int8 *group* scales into the weights before the MMA, which Q4_K's 6-bit scales cannot do.
2. **The exact-Q4_K ceiling on an H100 SXM is ~475 TMAC/s useful**, not 1070: two u8 passes (lo/hi scale split) plus the
   mins term cost 2.25× tensor work (ESTIMATED; the 2.06× in hopper.md §4 omits putting M on the tensor core).
3. **The per-256 drain costs issue slots, not float throughput**: ~9 full-rate ops per output per block ≈ 50 % of the SM's
   issue bandwidth at full tensor rate (ESTIMATED §3.2). That is the structural cost of the contract on Hopper and it
   caps a perfect kernel at ~60 % tensor busy ≈ 285 TMAC/s. 200 TMAC/s = 42 % tensor busy.
4. **`gemmh2` has a hidden conversion-unit cost:** it still does one real `I2F` per output per block (`i2f_rn(S)`, CODE
   `hopper.cu:838`), and I2F runs at 16/clk/SM (1/8 of FFMA) on sm_90 (NVIDIA CUDA C Programming Guide throughput
   table; cited from memory, see §3.3). The tree already avoids this for M (`gm_i2f`, CODE `prefill.cu:839-846`). A 3+3
   scale split makes S convertible the same way, bit-identically (§3.3, proof).
5. **Recommended design (§4):** CTA 128 rows × 64 tokens, K-stage = one 256-superblock, 1 producer warpgroup (TMA,
   bsum pieces) + 2 consumer warpgroups (`m64n64k32` RS, 18 wgmma per block: 8 lo + 8 hi + 2 for M), per-k32 commit
   groups with 4 rotating A-fragment slots so the unpack overlaps the warpgroup's own wgmmas, magic-number conversions,
   4-stage mbarrier ring, 2-CTA cluster multicast of the activation tile in round two. Registers ≈ 222/thread (fits 232).
6. **Fixed-chunk split-K (M6)** is cheap and has precedents: CUTLASS stream-K defaults to `ReductionMode::Deterministic`
   (a lock-ordered serial chain over splits, CODE); Marlin's `global_reduce` is lock-ordered too but rounds partials to
   fp16 (CODE); QQQ reduces int32 partials (CODE). Recommend chunk = 4 blocks (1024 K), partials in fp32, a separate
   ordered reduce kernel that also runs the epilogue (§5).
7. **Expected: 180–260 TMAC/s on 3584×18944 at nb 1024, median ~210.** Confidence that it clears the roadmap's 180
   kill gate: ~55 % (ESTIMATED; every previous Hopper estimate in this project overshot 2×, §6).
8. **Where the 2× to Marlin-class comes from:** 2.25× tensor passes (the Q4_K format, not the contract) ≈ 55 %; the
   drain's issue load (contract + format) ≈ 25 %; today's kernel structure (no warp specialization, full `wg_wait<0>`
   per block, I2F, M on CUDA cores) ≈ 20 % (OPINION on the arithmetic in §7).
9. **Top risks:** the unexplained 3× between the issue model and `gemmh2`'s measured 84 TMAC/s (§6); register spills at
   n64 with four accumulator sets; L2 at ≥ 250 TMAC/s without multicast; M6 touching every product kernel.
10. **Plan:** half a pod-day of profiling that can kill the idea for $5 (E0': `gemmh2` with magic-I2F and with the drain
    removed), then 4.5 builder-days (§8). Kill: 3584×18944 at nb 1024 < 180 TMAC/s after day 5, or the drain-free upper
    bound itself < 180 at day 0.5.

---

## 1. What Axon has today (CODE)

- `gemmh2_q4_k` (`src/cuda/hopper.cu:682-900`): two *independent* warpgroups, each 64 rows × 64 tokens, RS wgmma
  `m64n64k32.u8.s8` with the A fragments unpacked from the raw GGUF 144-byte rows in registers (`alo[8][4]`,
  `ahi[8][4]`, scales split `lo4 + 16·hi2`, pre-scaled `q·lo4 ≤ 225`, `q·hi2 ≤ 45`). Thread 0 issues TMA inline (no
  producer warpgroup). Per 256-K block: 16 wgmmas, `wg_commit`, M by 4 dp4a + IMAD per output on CUDA cores while the
  wgmmas run, then `wg_wait<0>` (full drain), then the epilogue per output: `S = 16·dhi + dlo` (IMAD),
  `i2f_rn(S)` (**real I2F**), `gm_i2f(M)` (magic add/sub), two `__fmul_rn`, `__fsub_rn`, `__fmaf_rn` with `dk`.
  4-stage ring, `GEMMH2_SMEM` ≈ 4 × (16 KB + 18 KB) + barriers.
- MEASURED on an H100 PCIe (114 SMs; `experiments-rejected.md:50`, `history.md` #33): gate nb 1050 gemm2s 70.2,
  gemmh 56.6 (SS, unpack to smem), gemmh1 38.7, **gemmh2 83.8 TMAC/s**; wgmma peak probes 620–707 TMAC/s, mma.sync 254.
  On the H100 SXM showdown (10-19) the end-to-end prefill rate was ~100 TMAC/s (review §2.2).
- The contract per output and block b (`decode.cu:4-15`, `prefill.cu:2-16`, `gemv_ref`): exact integers
  `S = Σ_j sc_j·sumi_j`, `M = Σ_j m_j·bsum_j`; `v = fsub(fmul(d, i2f(S)), fmul(dmin, i2f(M)))`; `acc = fma(dk_b, v, acc)`
  ascending from +0. "Any exact way of computing the integers is allowed" (hopper.md §4).

---

## 2. State of the art on Hopper (what each one actually does)

| kernel | types / pipe | dequant | reduce | H100 number | order deterministic? | label |
|---|---|---|---|---|---|---|
| **Marlin** (IST-DASLab, `marlin_cuda_kernel.cu`) | W4A16, fp16 `mma.sync.m16n8k16` | in registers (lop3 + fp16 magic-number), weights pre-shuffled | striped partition over K across SMs; `global_reduce` serial in L2, lock-ordered by `slice_idx` (`barrier_acquire(&locks[slice_col], slice_idx)`), partials **rounded to fp16** in the C buffer, fp32 add | Axon's showdown: vLLM-AWQ 29.8k tok/s on a lone 7.6k prompt ≈ 215 TMAC/s ≈ 43 % of the fp16 pipe (MEASURED, review §2.2; whether vLLM 0.32 picked Marlin or Machete for AWQ on sm_90 is **unverified**) | yes run-to-run on one GPU; the stripe layout depends on the SM count, and the fp16 round trip is lossy | CODE (marlin.cu:162-180, 510-560, 655-682) |
| **Machete** (vLLM `csrc/quantization/machete`) | W4A16 / W8A16 on wgmma, CUTLASS 3.5.1 mixed-input (example 55) | in registers → RS wgmma; computes `Yᵀ = Wᵀ Xᵀ` so the narrow type is operand A ("only A and C can be sourced from registers"); pre-shuffled so one 128-bit LDS feeds the fragment; interleaved upconvert | CUTLASS `KernelTmaWarpSpecialized{Pingpong,Cooperative}`; tile heuristic `(128,256)` / `(128,128)` cluster `(2,1,1)` for M > 128, `(128,16..64)` for small M (CODE `generate.py:510-533`) | "competitive with FP16 at batch ≥ 128" on H100 (blog, figure only; ESTIMATED 250–350 TMAC/s from that phrase); +29 % / +32 % e2e over Marlin on Llama 3.1 70B (MEASURED, blog) | no split-K in the heuristic I read → data-parallel, deterministic; **W4A8 (int8 act) still `# TODO`** (CODE `generate.py:603`) | CODE + MEASURED (blog) |
| **QQQ** (vLLM `marlin/qqq/marlin_qqq_gemm_kernel.cu` v0.6.3) | W4A8, `mma.sync.m16n8k16.s32.s8.s8` | per-group: int4 → int8 in registers via fp16 `hfma2` with the group scale (lop3 + 2 hfma2 + pack), per-channel: `q & 0xf0f0f0f0` | Marlin's striped split-K; **`global_reduce` works on INT32** (exact, commutative) then per-channel fp32 × per-token scales in the epilogue | paper: 3.67× / 3.29× over FP16 GEMM, A100 (MEASURED by them); LiquidGEMM found QQQ slower than QServe on H800 | int32 partials → bit-exact in any order; one int32 accumulator across all of K (group scale is int8-representable, unlike Q4_K's) | CODE + MEASURED (paper abstract) |
| **QServe** (MIT, arXiv:2405.04532; TRT-LLM port `qserveGemmPerGroup.cu`) | W4A8KV4, `mma.sync.m16n8k32.s8` | "progressive quantization": int4 × int8 group scale → int8 (range clamped to ±119), "subtraction after multiplication", compute-aware weight reordering, register-level parallelism | none (data-parallel), int32 accumulators across K | 1.2× / 1.4× TRT-LLM throughput on A100 / L40S (MEASURED, abstract); no wgmma → capped at the mma.sync rate on Hopper (Luo: 978 vs 1449 TOPS) | deterministic | CODE (TRT-LLM `qserveGemmPerGroup.cu:140-150`, no `wgmma`, no `split`) |
| **LiquidGEMM** (ByteDance/SJTU, arXiv:2509.01229) | W4A8 int8 on wgmma, H800 | LiquidQuant: rotate int8 into u8 range before u4 quant, dequant = **IMAD + XOR per 4 elements** in registers; "dual-MMA packed layout" (one 16-byte `ldmatrix`/LDS feeds two k32 steps); **1 Load WG + 2 Compute WGs**, each Compute WG dequants then immediately issues RS wgmma; CUTLASS ping-pong skeleton, TMA; group size 64 with int8 group scales folded before the MMA | not stated; int32 accumulate over K, per-channel fp scale in the epilogue | **2.90×** over QServe at batch 256 (7B/13B/70B), 1.12–1.63× over TRT-LLM W4A16/W8A8/FP8; no absolute TOPS in the text I read (figures only) | deterministic as far as the paper says | MEASURED (paper) |
| **TensorRT-LLM** `fpA_intB_gemm` sm90 | int4 × {fp16, bf16, **e4m3**}, CUTLASS mixed-input, `KernelTmaWarpSpecialized{Pingpong,Cooperative}`, cluster 1×1 … 2×2 | upconvert in registers (CUTLASS) | data-parallel in the template I read | — | float accumulate; its "W4A8" is FP8-activation, not exact-int | CODE (`fpA_intB_gemm_template_sm90.h`) |
| **FlashInfer** `flashinfer/gemm` | fp8 / mxfp8 / fp4 / bf16 only | — | — | — | no int8-activation W4 path | CODE (`gemm/__init__.py`) |
| **CUTLASS 3** mixed input (example 55) | `{fp16,bf16}×{int8,int4,int2}`, `fp8×int4`; narrow type through registers (RS), K-major, group size a multiple of tile-K | in registers | stream-K / split-K schedulers with `ReductionMode::Deterministic` default | — | **no int8×int4→s32** | CODE (README) |

Microbenchmarks that bound the design (MEASURED, Luo et al. arXiv:2402.13499, H800, Tables VIII and X):
- int8 wgmma `m64n256k32`: **RS 1447.9 vs SS 1448.7 TOPS** of 1513 peak. RS costs nothing.
- fp16 wgmma by N (SS/RS, dense): n256 728/732, n128 728/725, **n64 720/720, n32 477/710**. n64 is full rate; n32 is
  only full rate in RS mode. (The int8 rows weren't in the extract I read; the shape dependence is structural, so I take
  it to transfer.)
- mma.sync int8 `m16n8k32` on H800: 978 TOPS vs wgmma 1449 (already in hopper.md §1.2).

Take-aways for Axon: every fast Hopper W4 kernel (Machete, LiquidGEMM, TRT-LLM) is **RS wgmma with the narrow operand
as A, dequant in registers, warp-specialized producer/consumer, TMA, and no per-group float drain** (int8-foldable group
scales or fp16 upconvert with a single accumulator chain). Axon's roadmap item 2 wording ("RS, as Marlin does") is
right; the part none of them have is our per-256 drain, and that is the whole problem (§3).

---

## 3. The arithmetic of an exact Q4_K superblock on wgmma

### 3.1 Tensor work (ESTIMATED)

H100 SXM: 132 SMs × 1.98 GHz × 4096 int8 MAC/clk = **1071 TMAC/s** at the clock the pod held (hopper.md §2; the
datasheet's 1979 TOPS = 989 TMAC/s is at a lower clock).

Per output and 256-K block the exact kernel issues, per warpgroup `m64nNk32`:
- 8 k32 steps × 2 passes for S (`q·lo` and `q·hi`, since 4-bit × 6-bit = 10 bits > u8) = 16 wgmma;
- 2 wgmma for M = Σ_j m_j·bsum_j: A = mins (≤ 63, u8), B = the token's 8 `bsum_j` split into `lo6 + 64·hi7` int8
  pieces (gemmh2 already builds exactly these, `hopper.cu:770-775`), K = 8 useful of 32, two accumulators
  (`D_Mlo`, `D_Mhi`; `M = 64·D_Mhi + D_Mlo`). **This moves M off the CUDA cores** (gemmh2: ~176 issue slots per warp per
  block) at the cost of +12.5 % tensor time. Alternative with zero prep: a third full pass with A = `m_j` broadcast
  over the sub-block (+50 % tensor). The padded form is better.

So the multiplier is **18/8 = 2.25×** and the useful ceiling is **1071 / 2.25 = 476 TMAC/s** at 100 % tensor busy.
200 TMAC/s is 42 % tensor busy; 285 is 60 %.

Could a one-pass form beat it? With one u8 pass and the per-sub-block `sumi_j` kept separate you need 8 accumulators
per block (256 regs at n64: impossible) or a drain after every k32 and 8 IMADs per output per block on CUDA cores.
Those IMADs alone are 8/128 = 0.0625 clk per output-block per SM against 0.0625 clk of tensor time: exactly 1:1, the
FMA pipe saturates before the tensor pipe does, and the per-k32 wait kills the async pipeline. (This is `gemm2`'s
mma.sync structure, MEASURED MIO/latency-bound on sm_89, ADR §2.1.) **Two passes is the right call; the 2.25× is the
price of the GGUF format, not of the contract.**

### 3.2 The drain: issue slots, not FLOPs (ESTIMATED)

Per output per block the contract needs: combine `S` (1 IMAD, or 0 with §3.3), 2 int→float conversions, 2 fmul,
1 fsub, 1 fma, ~0.5 LDS for `dk` ≈ **8–9 full-rate instructions**. At 100 % tensor rate an SM finishes
4096 / (256 × 2.25) = 7.1 output-blocks per clock, so the drain wants **64 instructions per clock per SM = 50 % of the
SM's 128 issue slots** (4 sub-partitions × 32 lanes, 1 warp-instruction per clock each). Add the weight unpack (§3.4,
12–23 %) and the wgmma issue (3 %) and a perfect kernel is at 65–76 % issue utilization at 100 % tensor, which in
practice means ~60 % tensor busy ≈ **285 TMAC/s is the exact-Q4_K ceiling** on this design. That is the contract's
Hopper cost in one number: the drain is ~half the SM's instruction bandwidth.

What it is *not*: it is not FP32 throughput (9 ops at 128/clk = 7 % of tensor time), not smem, not L2 (§4.4).

### 3.3 The conversion-unit trap, and the exact fix

`__int2float_rn` compiles to `I2F.F32.S32`, which runs on the conversion/XU pipe at **16 results per clock per SM** on
cc 8.x/9.0 ("all other type conversions: 16", CUDA C Programming Guide, Arithmetic Instructions throughput table —
**cited from memory; the docs page needs JS and I could not fetch it; verify with one microbench in E0'**). That is 1/8
of FFMA. Two I2F per output-block = 0.125 XU-clk per output-block per SM against 0.14 clk of tensor time: **the drain's
conversions alone would occupy the XU 89 % of the time at full tensor rate**. `gemmh2` already dodges half of it:
`gm_i2f(M)` is the magic-number form (`prefill.cu:839-846`, "instead of a slow I2F"), but `i2f_rn(S)` is a real I2F
(`hopper.cu:838`), so each consumer warp spends 32 × 8 = 256 XU-clk per block against 512 tensor-clk, and the two
warpgroups together 512 of every 1024: 50 % XU occupancy on a kernel that is already issue-bound (OPINION: a plausible
part of the 3× gap in §6).

**Make S convertible with the magic trick, bit-identically.** Split the 6-bit sub-block scale 3+3 instead of 4+2:
`sc = lo3 + 8·hi3`, pre-scaled quants `q·lo3 ≤ 105`, `q·hi3 ≤ 105` (both u8; hopper.md §4 already uses 3+3 for
Q5_K). Bounds: `|sumi_j| ≤ 32·15·127 = 60,960`, so `|D_lo|, |D_hi| ≤ 8·7·60,960 = 3.41 M < 2²² = 4.19 M`, and
`|M| ≤ 8·63·4,064 = 2.05 M < 2²²`. Then:

```
flo  = __fsub_rn(__int_as_float(0x4B400000 + D_lo),  12582912.0f);   // 1.5·2^23 magic, exact for |x| < 2^22
f8hi = __fsub_rn(__int_as_float(0x4D400000 + D_hi), 100663296.0f);   // 1.5·2^26 magic, ulp 8: exact 8·D_hi
fS   = __fadd_rn(f8hi, flo);                                          // == i2f_rn(S), S = 8·D_hi + D_lo
```

`flo` and `f8hi` are exact floats (both integers are below 2²², and 8·D_hi is a float with ulp 8 in [2²⁶, 2²⁷)); the
real sum is exactly `S`, so one correctly-rounded `fadd` returns `rn(S)`, which is by definition what `cvt.rn.f32.s32`
returns. **Bit-identical to `i2f_rn(S)` for every input the format can produce**, including |S| ≥ 2²⁴ where the
conversion rounds. Cost: 2 IADD + 3 FADD at full rate instead of 1 IMAD + 1 I2F at 1/8 rate. This adds float ops to
the code but not to the *result*; hopper.md §4's "no float op added" is a sufficient condition, not the contract. Gate
it with `gpu_products_match_reference_bits` plus one adversarial fixture at the bounds (`sc = 63`, `q = 15`,
`x = ±127` everywhere) so the proof is exercised, not trusted. CODE precedent: `gm_i2f` has been on main for M.

### 3.4 Unpack cost with a load-time Hopper repack (ESTIMATED)

RS `m64nNk32.u8` holds 16 A bytes per thread per k32 step; per block and pass that is 128 u8 = 32 registers, 64 for
both passes (gemmh2's `alo`/`ahi`). From the raw GGUF layout gemmh2 does 16 LDS.32 + 2 LDS.128 and per packed u32:
1–2 extract + 2 IMUL (one 32-bit IMUL scales four bytes at once since 15·7 = 105 < 256) ≈ 135 instructions per warp
per block. A load-time repack to a Hopper-fragment order (as Marlin/Machete/LiquidGEMM all do; bit content unchanged,
"any exact way" applies) makes it 4 LDS.128 + the same ALU ≈ 120, and puts the 8 scale bytes and 8 min bytes where
one LDS.128 fetches them. The repack is a one-time CPU pass like `r8.rs` (Axon already repacks Q4_K for the CPU path,
CODE START-HERE) or a tiny device kernel at upload. Against 576 tensor-clk per warpgroup-block (n64) the unpack is
**21–23 % of a consumer warp's issue**; at n128 it would be half that, but n128 doesn't fit in registers (§4.2).

---

## 4. Recommended kernel: `gemmh3_q4_k` (sm_90a)

### 4.1 Shape and roles

- **CTA = 384 threads: 1 producer warpgroup (setmaxnreg 40) + 2 consumer warpgroups (setmaxnreg 232)**, 1 CTA/SM.
- **CTA tile 128 rows × 64 tokens; K-stage = one 256-superblock** (TMA boxes: raw weights `{144 B, 128 rows}`, activations
  2 × `{128 B, 64}` swizzle-128B as today, plus the 64 tokens' `bsum` pieces as a 64 × 32-byte K-major tile the producer
  writes from registers behind a `fence.proxy.async`). Stage = 18 + 16 + 2 KB = 36 KB; **4 stages = 144 KB** (+ `dk`
  and barriers), under the 227 KB limit, leaving room for a 128×32 token-tile variant to coexist.
- **Consumer warpgroup w owns rows 64w..64w+63 × all 64 tokens**: `wgmma.m64n64k32.s32.u8.s8` RS (A = unpacked weights
  from registers, B = activations from smem), **18 wgmmas per block**: 8 lo, 8 hi, 2 for M (A = mins broadcast into the
  first 8 K-lanes of a k32 fragment, zeros elsewhere; B = the bsum-piece tile).
- **Producer**: one thread issues the three TMA loads per stage and `expect_tx`; 64 threads build the bsum lo/hi pieces
  and `dk` for the 64 tokens (gemmh2's `t < 64` block, `hopper.cu:760-778`), write them to the stage, fence, arrive.
  Producer issue ≈ 30–40 slots per 1152 tensor-clk: negligible.

### 4.2 Registers (the binding constraint; ESTIMATED)

Per consumer thread at n64: `acc` fp32 32 + `D_lo` 32 + `D_hi` 32 + `D_Mlo` 32 + `D_Mhi` 32 (a 64×64 s32 fragment is
32 regs; the two M accumulators cannot share one fragment, since the ×64 on the hi piece has to be applied after the
MMA) + rotating A slots 4 × (4 lo + 4 hi) = 32 + descriptors/addresses/scales ~30 ≈ **222–230 of 232**. Tight. Fallbacks in order: (a) M via
one accumulator plus a scalar fix (`D_M = Σ m_j·lo_j`, hi part by 8 IMAD/output on CUDA cores: +32 slots/warp-block),
(b) n32 token tiles (RS n32 is full rate per Luo Table X; registers halve; grid doubles), (c) M back on dp4a
(gemmh2's path). **n128 per warpgroup is out**: four 64-reg accumulator sets + fragments = 320 regs.

### 4.3 The per-block schedule (what changes vs gemmh2)

```
for block b:
  wait full[s]                                   (mbarrier parity)
  for j in 0..7:                                 (k32 steps; A slot = j & 3)
     if j >= 4: wgmma.wait_group<N>  so that slot j&3's previous wgmmas retired   ← PTX: A regs must not change before wait
     unpack lo/hi fragments of step j into slot j&3   (4 LDS.128 total per block, ~15 ALU per step)
     wgmma.fence; wgmma lo(j) -> D_lo; wgmma hi(j) -> D_hi; commit_group
  wgmma M_lo, M_hi (B = bsum tile); commit; wait_group<0>
  arrive empty[s]                                 (the stage can be refilled now; the epilogue needs only registers + dk)
  epilogue over 32 outputs: fS (5 ops, §3.3), fM (2 ops), fmul, fmul, fsub, fma(dk)   ≈ 300 issue slots
```

Per-k32 commit groups let the **unpack overlap the warpgroup's own in-flight wgmmas** (LiquidGEMM's "dequantize then
immediately MMA"), which gemmh2 cannot do (its single `wg_wait<0>` per block serialises unpack → 16 wgmma → epilogue).
The epilogue is the warpgroup's tensor-idle window (~300 slots); the other warpgroup's 576-clk wgmma burst fills it
(ping-pong). Per-warpgroup cycle ≈ 576 tensor + ~40 sync + 300 epilogue ≈ 920 clk; two warpgroups demand 1152 tensor-clk
per ~920 → tensor-bound in the model, which is the right side to be on. Issue per sub-partition per 1152 clk:
2 × (18 + 120 + 300 + 40) + 35 ≈ **990 of 1152 = 86 %** (ESTIMATED; this is why I do not believe 100 % tensor and why
§6 derates).

Pitfalls carried from hopper.md §3 all apply: ptxas C7510 "serialized" (the rotating-slot pattern is statically
unrolled, so register disjointness is provable; keep the compile test that greps the NVRTC log), `fence.proxy.async`
after the producer's `st.shared` of the bsum tile, mbarrier parity on odd stage counts, `setmaxnreg` sums
2 × 128 × 232 + 128 × 40 = 64,512 ≤ 65,536.

### 4.4 Bandwidth (ESTIMATED)

Per CTA block (128 × 64 × 256): TMA 34 KB; wgmma B reads 18 × 2 KB × 2 warpgroups = 72 KB; smem total ≈ 106 KB per
1152 clk = 92 B/clk of 128 (72 %; fine, and the SS variant that stages unpacked A would add 64 KB and bust it, which is
what `gemmh` measured). **L2→SM at 100 % tensor: 34 KB × 132 SMs × 1.98 GHz / 1152 = 7.7 TB/s — above the H100's L2**
(hopper.md: ~5.5–7 TB/s); at the 40–60 % tensor we are aiming for it is 3.1–4.6 TB/s, OK. Round two: a 2-CTA cluster
(two row tiles, same 64 tokens) multicasting the activation tile cuts it to 5.9 TB/s at 100 %. DRAM is not a factor
above nb ≈ 64 (weights 4.68 GB per pass).

### 4.5 Tile table and routing for the 7B shapes (CTAs = ⌈rows/128⌉ × ⌈nb/64⌉ × splits; 132 SMs)

| product (rows × K) | K blocks | nb ≤ 32 | nb 64 | nb 128 | nb 512 | nb 1024 |
|---|---|---|---|---|---|---|
| gate+up fused (2 × 18944 × 3584) | 14 | GEMV / gemm2s (memory-bound) | 296 CTAs (2.2 waves) | 592 (4.5) | 2368 (18) | 4736 (36) |
| down (3584 × 18944) | 74 | GEMV with M6 split | 28 → **split 8** = 224 | 56 → **split 4** = 224 | 224 → split 2 = 448 | 448 (3.4 waves; split 2 or persistent to kill the 40 % tail) |
| q / o (3584 × 3584) | 14 | GEMV with M6 split | 28 → split 4 = 112 (chunks {4,4,4,2}) | 56 → split 2–4 | 224 | 448 |
| k / v (512 × 3584) | 14 | GEMV with M6 split | 4 → split 4 = 16 | 8 → split 4 = 32 | 32 → split 2 | 64 → split 2 (1.8 GMAC: ~10 µs, don't over-engineer) |

Ragged token tails (nb = 100 → two 64-tiles, 28 % waste) and nb ≤ 48 get the **128 × 32 variant** (RS n32 is full
rate; half the registers; same code with N as a template parameter). Fused gate+up keeps gemmh2's twin structure
(warpgroup 0 = gate rows, 1 = up rows, SwiGLU through smem at the end).

### 4.6 An alternative worth one experiment: the "u8 twin" (OPINION)

hopper.md §4 already plans an int8 copy for Q6_K. A **pre-scaled u8 twin of Q4_K** (two bytes per weight: `q·lo3`
and `q·hi3`, with scales/mins per row-block alongside) removes the unpack entirely: TMA lands A tiles in the swizzled
wgmma layout, SS wgmma, consumer work = the drain only. Cost: 2 B/weight = +14 GB VRAM for 7B, 3.6× the weight bytes
per pass (hurts the nb ≤ 128 step: 14 GB / 3.35 TB/s = 4.2 ms, worse than today's 1.4 ms weight read), so it is a
prefill-only kernel for nb ≥ ~512 on cards with VRAM to spare, and never the 70B answer. Not the recommendation; a
one-day fallback if the RS unpack turns out to be the stall (it is a bit-exact kernel too).

---

## 5. Fixed-chunk split-K that stays bit-exact (the ADR's M6)

**Precedents (CODE):**
- CUTLASS 3 stream-K / split-K (`sm90_tile_scheduler_stream_k.hpp:496-556`, `tile_scheduler_params.h:461`):
  `ReductionMode::Deterministic` is the **default**. Each split with `K_idx > 0` spins `wait_eq(lock, K_idx)` until the
  preceding splits have added into the fp32 workspace, adds its accumulators (`BlockStripedReduce::reduce`), then
  `arrive_inc(lock, k_tile_count)`; the final split `load_add`s the workspace and runs the epilogue. `Nondeterministic`
  only waits for the first split's store (`wait_lt(lock, 1)`) and lets the rest add in arrival order. So the cost of
  determinism in CUTLASS is a serial chain of (splits − 1) tile read-add-writes on the critical path, nothing else.
- Marlin (`marlin_cuda_kernel.cu:510-560, 673-677`): `barrier_acquire(&locks[slice_col], slice_idx)`, reduce,
  `barrier_release`: the same lock-ordered serial chain "in L2 cache", fp32 adds but **fp16 storage of partials**
  (lossy, and the stripe count depends on the `sms` argument, so bits differ across GPUs).
- QQQ (`marlin_qqq_gemm_kernel.cu:615-618`): the same chain on **int32** partials, exact regardless of order, because
  QQQ's whole K is one integer accumulator. Axon's integer part has the same property *within* a superblock; across
  superblocks the float chain fixes the order.
- Stream-K (Osama et al., PPoPP 2023) is the general form; its fixup is the same workspace + barrier pattern. I did not
  re-read the paper; the CUTLASS implementation above is the primary source here.

**Design for Axon (OPINION):**
- Redefine the canonical order once as a two-level chain with **chunk = 4 blocks (1024 K)**: `acc_c` runs blocks
  `4c..4c+3` from +0 in order; `out = (((acc_0 + acc_1) + acc_2) + …)` in chunk order. It is a function of K only, so
  identities A/B/C/G hold at any split, and the unsplit kernel pays `⌈K/1024⌉ − 1` extra fadds per output (≤ 18 for
  down; nothing). K = 3584 → chunks {4,4,4,2}; K = 18944 → 18 full + 1 of 2 blocks.
- A split CTA computes a **contiguous chunk range** and writes its fp32 partial (already summed left-to-right over its
  chunks, which is associatively identical to the global chain since `((a+b)+c)+d = P(a,b,c)+d`). Partials go to a
  workspace `[splits][nb][rows]` fp32 (nb 128 × 3584 rows × 4 splits = 7.3 MB; L2-resident).
- **Reduce in a second kernel**, not with locks: it sums the partials in split order and runs `gm_pre`/`gm_put`
  (bias, residual modes). Cost: one launch (~3 µs) + reading 7 MB (~5 µs) versus the 3+ ms those products take today
  at nb 128. Rationale: CUDA-graph replay and PDL make spinning locks a liability; a kernel pair is trivially
  deterministic and testable. Locks (CUTLASS-style) would save the launch and are a later micro-optimisation.
- Atomics (`red.global.add.f32`) would save the ordered pass and the workspace, and are what `fast` mode could use
  (ADR §4); the cost of *not* using them here is ~10 µs per split product (ESTIMATED), invisible next to the 18.6 ms step.
- The decode GEMVs (`decode.cu`), `gemm2*` and `gemmh*` all adopt the chunked chain in one PR with one
  `NUMERICS_EXACT` bump and the md5 re-pins (ADR M6). This is the one part of the plan that touches every kernel.

---

## 6. Expected performance and confidence

| | model (ESTIMATED) | why I don't believe the model |
|---|---|---|
| tensor peak | 1071 TMAC/s | — |
| exact Q4_K useful ceiling (2.25×) | 476 | — |
| issue-feasible ceiling (§3.2) | ~285 (60 % tensor) | — |
| gemmh3 as specified, no multicast | 240–330 | **hopper.md's model predicted 180–300 for gemmh2 and it measured 84 on the PCIe card.** My issue model gives gemmh2 ~80 % tensor; reality was ~25 %. A 3× gap of unknown cause (candidates: XU occupancy §3.3, the single-WG serial chain with only 8 warps/SM of latency hiding, ptxas scheduling of RS fragments, mbarrier phasing) |
| **what I'd bet on** | **180–260, median ~210** on 3584×18944 at nb 1024 (H100 SXM) | derated 1.5× against the model for the unexplained part that warp specialization + overlap don't address |
| P(≥ 180, the kill gate) | ~55 % | |
| P(≥ 250) | ~30 % | needs multicast and a clean profile |
| nb = 128 products with M6 | 56 → 224 CTAs; GEMM share of the step 8.4 → ~4.5–5.5 ms | roadmap's number; depends on the same kernel rate |
| 7B pp2100 | 13.8k → 22–27k tok/s | GEMM 95 % of the pass; attention untouched |

The one measurement that collapses this uncertainty is E0' (§8, day 0.5): `gemmh2` with (a) the I2F swapped for the
§3.3 conversion (bit-identical, so the existing tests gate it) and (b) the drain removed entirely (inexact bound).
If (b) is < 180 TMAC/s the problem isn't the drain and the restructure is the whole bet; if (a) ≈ (b) ≫ 84 the XU was
most of it and the gate is nearly in hand.

---

## 7. Where the 2× to Marlin-class comes from (ESTIMATED shares, OPINION on the split)

Marlin/Machete-class on H100: ~215 TMAC/s measured for vLLM-AWQ (fp16 pipe, 43 % of 495), Machete "competitive with
FP16" (~250–350 on the fp16 pipe). An exact Q4_K kernel at the same *efficiency* of its pipe would be 0.43 × 476 ≈
**205 TMAC/s**: the roadmap's 200 target is literally "Marlin efficiency on the exact-Q4_K ceiling". The gap between
that and the int8 pipe's 1071:

1. **2.25× tensor passes** (6-bit sub-block scales don't fit u8; mins need their own pass): −55 % of peak. Property of
   GGUF Q4_K; a requantized format with int8-foldable group scales (QServe/LiquidGEMM) or an FP8 checkpoint doesn't pay
   it. **Not a contract cost**, and `--numerics fast` doesn't remove it either: without requantizing the weights there is
   no one-pass kernel for Q4_K, exact or not. (That is why item 4's FP8 kernel is the right `fast` alternative.)
2. **The per-256 drain's issue load** (~50 % of issue at full tensor): caps tensor busy at ~60 %. Contract + format
   together (a per-256 fp32 chain is what the format's `d`, `dmin` per superblock demand; the contract only fixes its
   order). ≈ −25 % relative.
3. **Today's kernel structure**: no producer warpgroup, full `wg_wait<0>` per block, M on dp4a, one real I2F per output,
   raw-layout unpack: the remaining ~2× between 84–93 and ~200. This is what round two removes.
4. **Not** the reduction order: the GEMM has 36 waves at nb 1024 and needs no split-K; where it needs one (nb ≤ 128,
   narrow products), a fixed-chunk split is deterministic at ~10 µs of cost (§5).

---

## 8. Five-builder-day plan (one kernel builder, ~$25 of H100 SXM) and the kill rule

| day | work | gate / output |
|---|---|---|
| **0.5** (pod, ~1 h, $5) | **E0'**: `cuda_gemm_bench` on the SXM for gemm2s / gemmh2 at the four 7B shapes × nb {64,128,512,1024}; ncu on gemmh2 (`sm__pipe_tensor_cycles_active`, `sm__inst_executed_pipe_xu`, issue-active, top stall reasons); two gemmh2 variants built locally: **(a)** `i2f_rn(S)` → §3.3 conversion with the 3+3 split (bit tests must pass), **(b)** drain removed (inexact upper bound; never routed). One I2F-throughput microbench to confirm 16/clk/SM. | The drain's true cost on Hopper (roadmap item 1(a) asked for exactly this). **Early kill: (b) < 180 TMAC/s → the restructure is required before any of §4 pays; proceed only with day 1–2 scoped as "structure first".** |
| **1–2** (local compile-only + 1 pod h) | `gemmh3_q4_k` v1: producer warpgroup (TMA + bsum tile + dk), 2 consumers n64, M on tensor, §3.3 conversions, per-k32 commit groups with 4 A slots, 4 stages, load-time Hopper repack of Q4_K (device kernel at upload, behind `Module::hopper()`). Bit tests: `gpu_products_match_reference_bits`, batched-vs-gemv, ragged rows/tokens/K, PDL on/off, plus the adversarial-bounds fixture. No "serialized" in the NVRTC log. | **≥ 140 TMAC/s** on gate+up nb 1024 (1.5× today) and bits equal everywhere; else fall back through §4.2 (a)→(c) for half a day, then stop and write up. |
| **3** (local + 1 pod h) | **M6**: chunk-4 canonical chain in `decode.cu` GEMVs, `gemm2*`, `gemmh*`, `gemv_ref`; split-K partials + ordered reduce kernel with the epilogue modes; `NUMERICS_EXACT` bump; md5 and `gpu_products_match_reference_bits` re-pinned; routing table §4.5. | decode == prefill == verify across every split factor; H100 7B nb 128 step **−10 %** (roadmap's M6 gate) else revert M6 alone. |
| **4** (1 pod h) | 2-CTA cluster multicast of the activation tile; 128×32 variant for ragged/small nb; fused gate+up twin; Q6_K twin only if time (4 passes, hopper.md §4). | gate+up nb 1024 ≥ day-2 + 15 %; down/q/o at nb 128 on the split path faster than gemm2s. |
| **5** (1 pod h) | Scoreboard: 7B pp2100 and Qwen3-8B, `pool_step_cost` nb 1–256, the two serve C = 128 cells; consumer cubins byte-identical; write-up in `docs/log/`. | **Kill rule (roadmap item 2, kept):** 3584×18944 at nb 1024 **< 180 TMAC/s on the H100 SXM** → stop, record in `experiments-rejected.md` with the E0' numbers, go to item 4's FP8 `fast` kernel. Pass: ≥ 180 and pp2100 ≥ 20k tok/s. |

Top 3 risks (OPINION):
1. **The unexplained 3× (§6).** If E0'(b) (no drain) is itself ~120, the stall is in the RS/TMA/mbarrier structure and
   warp specialization is a bet, not a fix. Mitigation: E0' first, ncu stall reasons before writing a line of gemmh3.
2. **Register spills at n64 with four s32 accumulator sets + rotating A slots** (≈ 225 of 232). Mitigation: §4.2's
   fallbacks, in order; `-Xptxas -v` equivalent via `nvrtcGetProgramLog` on every compile; a spill is a fail.
3. **M6's blast radius**: every product kernel, every md5 reference, the CPU/GPU graded tests, decode graphs. Mitigation:
   land M6 as its own PR after the kernel is measured, with the chunk constant in one header and `NUMERICS_EXACT` bumped
   once; keep the un-split path the default on consumer cards (bits change there too; one re-pin).
   Runner-up: L2 bandwidth above ~250 TMAC/s without multicast (§4.4), and NVRTC/ptxas serialising the rotating-slot
   wgmmas (C7510; the compile-log test catches it).

---

## 9. Sources read

- Axon: `src/cuda/hopper.cu` (gemmh2 682–900, wgmma/TMA helpers 1–120), `src/cuda/prefill.cu` (header 1–16,
  `gm_i2f` 839–846), `src/cuda/decode.cu:413, 470-474`; `docs/design/hopper.md`; `docs/design/determinism-modes.md`
  (§2, M6 table); `docs/roadmap.md` "Next: proposed order"; `docs/reviews/2026-10-19-outside-review-next-steps.md` §2;
  `docs/experiments-rejected.md:50`; `docs/history.md` #33; commit `d39111b` message (PCIe numbers).
- LiquidGEMM: Hu, Xiao, Sun et al., arXiv:2509.01229 (HTML v1; §1, §3, §4 pipeline and layout, §5 numbers).
- Machete: Neural Magic / Red Hat Developer, "Introducing Machete, a mixed-input GEMM kernel optimized for NVIDIA Hopper
  GPUs" (2024-10-14); vLLM `csrc/quantization/machete/generate.py` (main; tile heuristic 510–533, W4A8 TODO 603).
- Marlin: IST-DASLab `marlin/marlin_cuda_kernel.cu` (master); paper arXiv:2408.11743 not re-read.
- QQQ: Zhang et al. arXiv:2406.09904 (abstract); vLLM v0.6.3 `csrc/quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu`.
- QServe: Lin, Tang, Yang et al. arXiv:2405.04532 (abstract); TensorRT-LLM `cpp/tensorrt_llm/kernels/qserveGemmPerGroup.cu`.
- TensorRT-LLM: `cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/` listing and `fpA_intB_gemm_template_sm90.h`.
- CUTLASS: `examples/55_hopper_mixed_dtype_gemm/README.md`; `include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp`;
  `include/cutlass/gemm/kernel/tile_scheduler_params.h`.
- FlashInfer: `flashinfer/gemm/__init__.py` and directory listing.
- Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", arXiv:2402.13499 (Tables VI–X).
- **Could not verify online:** the CUDA C Programming Guide throughput table (JS-only page); PTX ISA wording on A-register
  reuse before `wgmma.wait_group` (from memory: undefined behaviour to modify them earlier); which vLLM backend served
  AWQ on the H100 showdown; Machete's absolute TFLOPS (figures only); Stream-K paper text.
