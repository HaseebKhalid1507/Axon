# FP8 weights end to end (WP-A2): design note

**Status: accepted for CP1** (review shady 7/10 CHANGES on 89d53a9; lead rulings 2026-10-29, final, folded in below and
pre-registered in §6). Plan: `beat-vllm-scope.md` §6 WP-A2; detail:
`fp8-safetensors.md` §1.2, §2, §3.2–3.6, §3.8–3.9, Stage B (the "spec"). Why now: A1 on an H100 came back VALID with
Track A OPEN (#174, job 1029-160204-d540). The FP8 wgmma GEMM ran 482 TMAC/s at gate nb 1024 (4.17× gemmh2) and
split-K down ran 241 at nb 128. On main: the safetensors loader (#102) and `--numerics exact|fast` (#173, ADR
[determinism-modes.md](determinism-modes.md)). Labels: CODE (main 452a7d4), MEASURED, ESTIMATED, OPINION.

## 1. Scope
In scope: per-channel e4m3 checkpoints (`RedHatAI/Qwen2.5-{0.5B,7B}-Instruct-FP8-dynamic`, compressed-tensors,
`float-quantized`). Their layout, VERIFIED from the 0.5B header and both configs: each linear is `weight` F8_E4M3
`[out,in]` plus `weight_scale` BF16 `[out,1]`, with `w = fp8·s`. Weights use `strategy channel`, activations
`strategy token, dynamic`, and the config has `ignore ["lm_head"]` and `kv_cache_scheme null`. embed, lm_head, norms
and biases are BF16. The 7B is 8.71 GB in 2 shards; the 0.5B is 0.90 GB.
Out of scope: block-128 and per-tensor scales (Stage C, though the kernels take a scale stride, §3), NVFP4, the Hopper
wgmma kernel (WP-A3: a routing seam only), a fused gate+up FP8 GEMM, split-K, and the CPU Q8_0 fast path.

## 2. Decisions
**D1. Exact mode runs the spec's FP8 → Q8_0-at-load fallback (§3.5). It doesn't refuse.** Reasons:
(a) the default mode must run the file (sm_61/80 too), and #173's exact-vs-fast run (launches moved, bug detectors)
needs a same-file exact net; the graded *reference* is the CPU fp8-ref (§6 B, ruling 2). (b) It adds no kernels: the result is a Q8_0 model run by today's exact kernels, so identities A–L hold
by construction, and the same path serves sm_61/sm_80 and fast mode below sm_89. (c) Its loss becomes a number
(the spec's "fallback vs native") instead of an adjective. GGUF exact bits can't move (D4).
**D2. Activations: e4m3 per TOKEN (ruling 3).** It matches the checkpoint's calibration (`input_activations.strategy
token`), vLLM's `cutlass_scaled_mm`, and A1's single-epilogue kernel (482 TMAC/s): the scales factor out of K, so no
promotion FMA. Mechanism: in a fast FP8 module, `quant_warp` (14 call sites, unchanged) also stores its 8 values as
f32 (`xf`, masked to `e < n`: the K = 896 tail is half a 256-block); then `quant_f8_tok` (one CTA per token) takes
`amax` over `e < n`, `s = __fdiv_rn(amax, 448)`, `q = cvt.rn.satfinite.e4m3x2.f32(__fdiv_rn(x, s))`, `s == 0 → q = 0`.
**Layout (the A3 seam, fixed now):** `qa` u8 `[nb][K]` at each token's act row (stride `astride`, 16-B aligned: a TMA 2D
box, as gemmh2 maps act today); `das` f32 `[nb]` contiguous, token-major (TMA 1D). If per-128 is ever chosen:
`da` f32 `[K/128][nb]`, MN-major per k-block (DeepGEMM's), next to `das`. **Pre-registered rule:** per-128 groups
replace per-token only if the CPU emulation (D7) shows per-token missing the HF bar (§6 A) and per-128 meeting it;
both are measured on the 0.5B before any GPU kernel. A per-128 GEMM on Hopper (promotion FMA) is then A3's to time.
**D3. The head and embedding are converted to Q8_0 at upload,** from the loader's F16 copy of the BF16 (spec §3.8).
That needs no new kernel (the Q8_0 GEMV/GEMM/embed exist), the tied 0.5B keeps one copy, and the 7B saves 1.1 GB
of VRAM. The identity records `head=q8`. Both modes share these kernels.
**D4. GGUF bits stay unchanged by construction.** The FP8 kernels sit in a new `fp8.cu`, compiled only into a
`KERNELS_F8` source variant with `-DAXON_F8`, **only for an FP8 checkpoint in `fast` on sm ≥ 89** (ruling 4). Exact
mode (and fast below sm_89) loads the plain module: same text, options and cache key as main, so the Q8_0 fallback
is today's kernels. `act_bytes` grows only in that module's nets; every routing branch keys on `qt == F8_E4M3`.
**D5. GEMM numerics (fast only), A1's form.** 64-row tiles; each output's 4 f32 accumulators carry the whole K
through chained `mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32`; epilogue `y = (acc·ws[r])·das[t]` (two `fmul.rn`), then
`gm_store` (bias, residual, stride). `-DAXON_F8_PROMOTE` (an FADD into CUDA-core f32 every 128 K) exists only for the
case where the f64 anchor (§3) shows the in-mma accumulation narrower than f32.
**D6. GEMV numerics for nb ≤ 16, CUDA cores (fast only).** One warp per row; lane l owns bytes 4l..4l+4 of each
128-byte chunk (coalesced); `acc_l = fma(w, a, acc_l)` over its K/32 values in order (e4m3×e4m3 products are exact
in f32); xor tree (16, 8, 4, 2, 1); `y = (Σ·ws[r])·das[t]`. Weights are read once for all nb tokens. Same operands as
D5, another order: the ADR's fast mode allows it, and decode is graded on its own.
**D7. CPU reference `fp8-ref` = exact dequant, f32 activations, F16 head** (spec §3.6 numerics, not its AVX2
kernel): F8_E4M3 joins the F32/F16 float fallback (CODE `quant.rs:2749`: its order, SIMD == scalar as F16's), the row
scale applied to each dot; weights read in place from the mmap. It is HF F32's twin (validated at KL ≤ 1e-4, §6 A),
the graded gate's reference (§6 B), and exact on the CPU. Emulation (`Fp8Cpu { act, head_q8 }`; CLI
`AXON_FP8_CPU_ACT=f32|e4m3-tok|e4m3-128`, `AXON_FP8_CPU_HEAD=f16|q8`): activations round-trip through the Rust twin
of D2's quantizer, the head through the Q8_0 converter: the fast path's precision in the CPU's order. Head on/off
splits the KL into head vs GEMM (review SHOULD 3). Slow prefill (f32 FMA): not a product path.

## 3. Kernels and their references (ruling 5)
Every FP8 kernel has two checks: **bits** against a reference of its own order (tiling, staging, odd K, scale
indexing), and an **independent f64 anchor**: a Rust f64 GEMM of the dequantized values, `E = ws·das·Σ w·a` (exact
e4m3 products, f64 sum), with the pre-registered tolerance **`|y − E| ≤ 2⁻²³·(K/32 + 40)·ws·das·Σ|w·a|`**: f32
recursive summation over ≤ K/32 chained steps plus ≤ 32 in-instruction terms, the tree and two scale roundings, ×2
slack. A narrower in-mma accumulator (Hopper's ~14 bits) fails it at K ≥ 896: that is what it is for.
| kernel (file) | used for | bits reference | anchor shapes |
|---|---|---|---|
| `quant_f8_tok` + `quant_warp` xf (`fp8.cu`, `decode.cu` `#ifdef AXON_F8`) | every F8 matrix's input | Rust twin `quant::quantize_act_f8` (integer RNE, no shared code): 256 codes, ties, satfinite, subnormals, the `n = 896` tail, 10⁶ random rows | `|q·s − x| ≤ s·½ulp_e4m3(x/s)` |
| `gemv_f8` (`fp8.cu`), nb 1..16, D6 | decode `gemv` seam, `bgemm` small | Rust `quant::gemv_f8_ref` (D6's order, `mul_add`) | nb {1,2,3,8,16} × rows {64,128,576,896,4864} × K {128,896,4864} |
| `gemm_f8<BN 64/128>` (`fp8.cu`), nb > 16, D5: K stage 256 B ×3 cp.async, raw rows (stride `cols` B), half stage for odd 128-block counts (K 896 = 7) | `bgemm`, unfused gate/up + `swiglu_quant_b` | `gemm_ref_f8`: one warp per 16×8 tile straight from global, fragment loads **written separately** from `gemm_f8`'s (no shared helper) | nb {17,24,64,65,128,160,520,1024} × rows {64,128,576,1024} × K {128,896,4864}; each token alone and in every n position |
| FP8→Q8_0, F16→Q8_0 converter (`quant.rs`, Rust, in the upload ring workers) | D1 fallback, D3 head | `quantize_q8_0` of the exact f32 dequant; GPU arena read back == CPU bytes | — |
Plus `gemv_f8` at nb 16 vs `gemm_f8` at nb 17 (the same 16 tokens + 1), within 2× the tolerance. Weight-scale stride
args `(rsh, rstride, kmask)`, per channel `(0,1,0)`, so Stage C changes arguments only. Hopper seam: `bgemm`'s F8 branch
is `match f8_route(nb, module) { Gemv, MmaSync, Wgmma }`, `Wgmma` → `MmaSync` until WP-A3 (sm_90 runs `mma.sync`).
Every FP8 launch calls `fast_launched()`; decode graphs count at capture (§6 B says where).

## 4. Routing and identity
| FP8 checkpoint | GPU sm ≥ 89 | GPU sm < 89 | CPU |
|---|---|---|---|
| `--numerics exact` (default) | Q8_0 at load, plain module: `weights=fp8-q8 act=q8 head=q8` | same | `weights=fp8-ref act=f32 head=f16` |
| `--numerics fast` | native, F8 module: `weights=fp8-native act=e4m3-tok head=q8` | fp8-q8, plain module, FAST_NOTE | fp8-ref plus FAST_NOTE |
The `weights/act/head` fields appear **only for FP8 checkpoints**, so GGUF tuples gain nothing.
`numerics::fast_kernels(net)` (CODE `numerics.rs:98`) is true only for fp8-native. `NUMERICS_VERSION_FAST` goes 1 → 2
by the ADR's "a fast kernel bumps it": GGUF fast fingerprints change, their bits don't. `gpu_refusal` stays for BF16
safetensors and lifts for FP8.

## 5. Files
- `src/safetensors.rs` (+`_tests.rs`): accept exactly this compressed-tensors config, refusing others by name: static
  `input_scale`, strategy tensor/block/group, e5m2, `kv_cache_scheme`, other targets/ignores. `weight_scale` →
  `blk.N.<kind>.scale` as an F32 vector (exact). F8 matrices are read in place. The loader also scans for the NaN
  codes `0x7f/0xff`, refuses `in % 128 != 0`, and checks scale shape and finiteness. FP8 fixtures go into
  `fuzz_gguf.py`'s safetensors mode.
- `src/gguf.rs` (qtype 1100 sizing, scale lookup), `src/quant.rs` (block_info F8 = (1,1), e4m3 decode, twin
  quantizer, converters, `gemv_f8_ref`, float fallback), `src/model.rs` (Weight row scale, D7, check_supported).
- `src/cuda/fp8.cu` (new), `src/cuda/decode.cu` (`quant_warp` xf store), `src/cuda/mod.rs` (`KERNELS_F8`, `-DAXON_F8`, sm ≥ 89),
  `src/cuda/upload.rs` (converting `Src`), `src/cuda/qwen2.rs` (`Mat.ws`, plan: native vs Q8_0 by mode and arch, head,
  `act_bytes`, decode `gemv` seam), `src/cuda/qwen2/batch.rs` (`bgemm` F8 branch, the `ffn_half` unfused path).
- `src/identity.rs`, `src/numerics.rs`, `src/lm.rs` (refusal site), tests in `src/cuda/qwen2/tests.rs` (products)
  and `tests/numerics.rs` (FP8 graded run), plus `testutil::fp8_model()` (skips when absent).
- `scripts/oracle/hf_logits.py` (new: `tf` = §6 A's teacher-forced gate; torch + transformers + safetensors, its own
  exact dequant, `--dtype f32|bf16`), `src/cli.rs` (`AXON_LOGITS_EACH`, ~10 lines). `docs/log/2026-10-xx-fp8-weights.md`, `docs/determinism.md`, README support table.
The hot files (`batch.rs`, `*.cu`, `lm.rs`) get a forum CLAIM at CP1 start.

## 6. Gates (pre-registered; rulings 1, 2, 5, 6)
**A. Accuracy vs HF (binding).** `scripts/oracle/hf_logits.py tf`: 32 windows of natural text
(`tests/fixtures/natural_text.txt`, `numerics_dnll.py`'s windowing), ~200 tokens each, ids from axon's tokenizer given to
every side (`AXON_LOGITS_IDS`). HF runs one forward per window on the exact f32 dequant (**F32**) and on its BF16 cast
(**BF16**, = compressed-tensors' decompress). Per position: KL(P_HF ‖ P_axon) on the partition {axon's top-64, rest}
(a lower bound; both rest masses reported), top-1 (raw), ΔNLL of the next token; means with SE clustered by window.
Arms, each binding and each ≥ 5000 positions: **decode** (`AXON_LOGITS_PREFIX=1`: `gemv_f8`) and **batched**
(`AXON_LOGITS_EACH=1`, new: each position its own batched pass, positions ≥ 16 so `gemm_f8`). **Bars, fp8-native on the
0.5B, against HF BF16 and HF F32 both:** KL mean ≤ 0.01, top-1 ≥ 97 %. **Stop line:** KL mean > 0.02 → stop, report.
**Near ties:** none excused; positions with HF top-2 margin < 0.1 nats are counted in, and listed with their count.
Reported beside, same positions: Q4_K_M 0.5B's own top-1 and KL vs HF (is 97 % reachable?), fp8-q8 (exact), fp8-ref,
the D7 emulations (tok / 128, head f16 / q8), and the 50 greedy picks (diagnostic only). **fp8-ref validation:**
KL(HF F32 ‖ fp8-ref) ≤ 1e-4 (BF16 beside); only then may fp8-ref stand in for the 7B reference, and the 7B gate then
reads "vs HF-F32-equivalent"; a booked pod adds a few-prompt 7B HF spot check.
**B. #173 graded gate, FP8 form (ruling 2).** Reference `r` = CPU fp8-ref; net under test = GPU fp8-native (fast);
floor `f` = CPU fp8-ref with the fast path's precision (e4m3-tok acts, q8 head), so `ac` = KL(f ‖ r) measures the
same precision change the net makes, and `assert_graded`'s mean-KL bound is 2× it (its other bounds unchanged). All
4 arms; fast launches > 0 asserted **per arm**: decode on a fresh net (its graph is captured inside the arm), the
batched arm's 24/160/520 prefixes each through `gemm_f8` (asserted per prefix). The Q8_0 exact model is not this
gate's reference (it carries requant loss); exact-vs-fast stays as the launches-moved/bug-detector run only.
**C. Kernel checks:** §3, release + debug. **D. Exact mode:** the FP8 checkpoint's fp8-q8 net passes the existing
bit-identity tests. **E. md5:** GGUF `gen` main vs branch, exact **and** fast (`axon-podjob md5.sh`). **F.** Loader
acceptance/refusals + fuzz. CP1 = A (0.5B) + B–F on bella, then a draft PR.
**CP2 (7B):** A on the 7B (pod HF BF16, or fp8-ref if validated). Speed on bella: one `--timing bella` job, 7B FP8 fast vs
Q4_K_M interleaved, pp512 / pp2048 / tg128, medians. Full suites, clippy `-D warnings`, rustfmt, CI, sm_89 run of B + C.
B can't run on the 7B (three nets), so it stays on the 0.5B. **H100 decode bar (ruling 6), a kernel-efficiency bar:**
FP8 tg128 ≥ 0.9 × Q4_K_M tg128 × (Q4_K_M bytes/token ÷ FP8 bytes/token), bytes measured (weights read incl. the q8
head, plus KV at 128): ≈ 0.9 × 314 × 4.68/7.68 ≈ **172 tok/s**. It is not a beat-vLLM bar; any such claim needs a
vLLM-FP8 tg128 measured beside it. ESTIMATED on bella: tg ≈ 0.6× Q4_K_M; pp 0.9–1.3× (R2).

## 7. Pods (I stage and rehearse on bella; you book)
4090 (sm_89), one session of about 45 min ≈ **$0.67**: (1)+(4) on the 0.5B, then the 7B HF BF16 reference (15 GB
in BF16: too much RAM for bella under the 8 GB rule). If avante (sm_89) is back up, it runs (1)+(4) for $0.
H100, about 20 min ≈ **$1.33**: tg128 7B FP8 vs Q4_K_M (+ vLLM-FP8 tg128 if a beat-vLLM claim is wanted). Total ≈ $2.0 of the $7.50. Models go in the kit by explicit copy (ruling 7).
If fp8-ref passes its 1e-4 validation, the 7B fp8-ref on bella's CPU is the 7B reference (≈ 7 GB file-backed + 2.2 GB anon).

## 8. Risks
- R1: settled by ruling 2 (§6 B): the floor is the same precision change emulated on the CPU, the reference the true
  model. Residual risk: the GPU's reorder noise on top of the floor exceeds 2× it; then I report, not loosen.
- R2: GeForce `mma` e4m3 with f32 accumulate is half the int8 rate (4090 sheet 330 vs 660 TOPS; ESTIMATED the same
  on sm_120). r17 found the GEMM MIO-bound, so this may not bind. pp might not beat Q4_K_M on bella.
- R3: mma semantics misread (operand order, fragment map, saturation, neighbour coupling): the bits reference can
  share the mistake; the f64 anchor (§3) can't. Neighbour coupling: tested by each token alone in every n position.
- R4: settled by ruling 1 (≥ 5000 teacher-forced positions per arm). R5: settled by ruling 6 (172 tok/s).
- R6: `axon-run --models` stages `.gguf` files only. The safetensors directories go into the kit with an explicit
  rsync (rehearsed on bella).
- R7: RAM. The fallback converts inside the ring workers, so there's no 8.1 GB anonymous Q8_0 arena for the 7B.
- R8: HF in BF16 rounds the dequantized weights (12 → 8 significant bits), so HF F32 is the faithful reference.
  I gate on BF16 as the scope says, and report F32 beside it.

## 9. Rollback
Everything sits behind `qt == F8_E4M3` or an FP8-only module. Reverting the PR brings back today's refusal
(`safetensors.rs:519`). No GGUF kernel, buffer or fingerprint changes except fast's counter (one constant).
