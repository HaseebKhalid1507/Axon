# FP8 weights end to end (WP-A2): design note

**Status: proposed** (checkpoint 0, for review before code). Plan: `beat-vllm-scope.md` §6 WP-A2; detail:
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
(a) #173's gate needs it. `graded_vs_exact` (CODE `tests/numerics.rs:581`) grades the net against **the same file
in exact mode on the same device**, with an exact-vs-exact floor. A refusing exact mode leaves the FP8 gate with no
reference. (b) It adds no kernels: the result is a Q8_0 model run by today's exact kernels, so identities A–L hold
by construction, and the same path serves sm_61/sm_80 and fast mode below sm_89. (c) Its loss becomes a number
(the spec's "fallback vs native") instead of an adjective. GGUF exact bits can't move (D4).
**D2. Activations are e4m3 per token per 128-K group (spec §3.2), not the checkpoint's per-token.** Every fused
quantize site (`quant_warp`: 14 call sites, 6 decode.cu + 4 prefill.cu for qwen2) already works one warp per
256-block, so two 128-groups per warp drop in inside `quant_warp` itself: no call-site change, no full-row pass. The groups are finer than the
checkpoint's calibration, and they're Stage C's tile. Cost: the GEMM folds `da[kb]` in once per 128-K block, which
it does anyway (D5). A1(a) was specified with one epilogue, so A3 has to re-measure with promotion (R5).
**D3. The head and embedding are converted to Q8_0 at upload,** from the loader's F16 copy of the BF16 (spec §3.8).
That needs no new kernel (the Q8_0 GEMV/GEMM/embed exist), the tied 0.5B keeps one copy, and the 7B saves 1.1 GB
of VRAM. The identity records `head=q8`. Both modes share these kernels.
**D4. GGUF bits stay unchanged by construction.** The FP8 kernels sit in a new `fp8.cu`, compiled only into FP8
models' modules (a `KERNELS_F8` source variant plus `-DAXON_F8`, like `KERNELS_OSS`). The f8 write in `quant_warp`
is under `#ifdef AXON_F8`. `act_bytes` grows only for FP8 models, and every routing branch keys on `qt == F8_E4M3`.
GGUF modules preprocess to main's text, which the md5 gate checks.
**D5. GEMM numerics (fast only).** Per 128-K block: `D = 0`, then 4 chained `mma.sync.m16n8k32.row.col.f32.e4m3.e4m3`,
then `acc = fma(da[t][kb], D, acc)` on the CUDA cores, blocks in ascending order. The epilogue is `y = acc·ws[r]`,
then `gm_store` (bias, residual, stride). Promotion every 128 K is DeepGEMM's guard against narrow in-mma
accumulation. Fast mode doesn't need it bit-wise, but a fixed order is what makes a reference-bits test possible.
A `-DAXON_F8_ZK` variant (D zeroed per k32) is kept for R3.
**D6. GEMV numerics for nb ≤ 16, CUDA cores (fast only).** One warp per row. Lane l owns bytes 4l..4l+4 of each
128-block. Per block it computes `p = Σ₄ w·a` (an fma chain from 0; e4m3×e4m3 products are exact in f32), then
`acc_l = fma(da[kb], p, acc_l)`. After the last block an xor tree (16, 8, 4, 2, 1) sums the lanes, then `·ws[r]`.
Weights are read once for all nb tokens. The order differs from D5's on purpose: the ADR's fast mode allows it, and
the gate grades decode separately from batched.
**D7. CPU reference path = exact dequant, f32 activations** (spec §3.6 numerics, without the AVX2 kernel).
F8_E4M3 joins the F32/F16 float fallback (CODE `quant.rs:2749`, its order and SIMD==scalar as F16's), and the row
scale is applied to the dot. Weights are read in place from the mmap. Uses: (i) an Axon-side HF-equivalent reference,
validated against HF on the 0.5B; (ii) the diagnostic `AXON_FP8_CPU_ACT=e4m3`, which round-trips activations
through the Rust twin of D2's quantizer, so the activation-quant loss is measured before any GPU kernel exists (R1).
Exact on the CPU, so the `cpu` arms work. Slow prefill (f32 FMA), so it's not a product path.

## 3. Kernels and their references
| kernel (file) | used for | reference, test (bits) |
|---|---|---|
| `quant_warp` f8 view (`decode.cu`, `#ifdef AXON_F8`): per 128-group `s=__fdiv_rn(amax,448)`, `cvt.rn.satfinite.e4m3x2.f32(__fdiv_rn(x,s))`, `s==0→0` | every activation an F8 matrix reads | Rust twin `quant::quantize_act_f8` (integer RNE): all 256 codes round-trip, ties, satfinite, subnormals, 10⁶ random rows GPU == CPU |
| `gemv_f8` (`fp8.cu`), nb 1..16, D6 | decode `gemv` seam, `bgemm` small | Rust `quant::gemv_f8_ref` (D6's order, `mul_add`). nb {1,2,3,8,16} × rows {64,128,576,896,4864} × K {128,896,4864} |
| `gemm_f8<BN 64/128>` (`fp8.cu`), nb > 16, D5: 64-row tiles, K stage 256 B ×3 cp.async, raw rows (stride `cols` B, 16-B aligned), half stage for an odd block count (K 896 = 7 blocks) | `bgemm`, unfused gate/up + `swiglu_quant_b` | `gemm_ref_f8` (`fp8.cu`): one warp per 16×8 tile from global, the same mma chain. nb {17,24,64,65,128,160,520,1024} × rows {64,128,576,1024} × K {128,896,4864}, each token alone and in every n position (the spec §4 batch-invariance check) |
| FP8→Q8_0 and F16→Q8_0 converter (`quant.rs`, Rust, in the upload ring workers) | D1 fallback, D3 head | CPU bytes == `quantize_q8_0` of the exact f32 dequant (e4m3 × bf16 = 12 bits, exact); the GPU arena read back == CPU bytes |
Weight scale lookup: `ws(r, kb) = s[(r >> rsh)·rstride + (kb & kmask)]`, per channel `(0,1,0)` (spec §3.3), so Stage C
only changes arguments. Hopper seam: `bgemm`'s F8 branch is `match f8_route(nb, module) { Gemv, MmaSync, Wgmma }`.
`Wgmma` returns `None` (falls back to `MmaSync`) until WP-A3. `mma.sync` e4m3 runs on sm_89+, so H100 runs FP8 in A2.
Every FP8 launch calls `fast_launched()`. Note: decode graphs count at capture, not at replay.

## 4. Routing and identity
| FP8 checkpoint | GPU sm ≥ 89 | GPU sm < 89 | CPU |
|---|---|---|---|
| `--numerics exact` (default) | Q8_0 at load: `weights=fp8-q8 act=q8 head=q8` | same | `weights=fp8-ref act=f32 head=f16` |
| `--numerics fast` | native: `weights=fp8-native act=e4m3-128 head=q8` | fp8-q8 plus FAST_NOTE | fp8-ref plus FAST_NOTE |
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
- `src/cuda/fp8.cu` (new), `src/cuda/decode.cu` (f8 view), `src/cuda/mod.rs` (`KERNELS_F8`, `-DAXON_F8`, sm ≥ 89),
  `src/cuda/upload.rs` (converting `Src`), `src/cuda/qwen2.rs` (`Mat.ws`, plan: native vs Q8_0 by mode and arch, head,
  `act_bytes`, decode `gemv` seam), `src/cuda/qwen2/batch.rs` (`bgemm` F8 branch, the `ffn_half` unfused path).
- `src/identity.rs`, `src/numerics.rs`, `src/lm.rs` (refusal site), tests in `src/cuda/qwen2/tests.rs` (products)
  and `tests/numerics.rs` (FP8 graded run), plus `testutil::fp8_model()` (skips when absent).
- `scripts/oracle/hf_logits.py` (new: `dump-hf`, transformers + compressed-tensors, `--dtype bf16|f32`, axon's ids,
  writes `llama_logits.py`'s JSON). `docs/log/2026-10-xx-fp8-weights.md`, `docs/determinism.md`, README support table.
The hot files (`batch.rs`, `*.cu`, `lm.rs`) get a forum CLAIM at CP1 start.

## 6. Gates
**CP1 (0.5B, bella sm_120):** (1) the §3 bit tests, release + debug. (2) Loader acceptance and refusals, plus fuzz.
(3) Accuracy: `hf_logits.py` on bella's CPU (a venv in scratch). KL(axon-cuda-fast ‖ HF BF16) top-20 mean ≤ 0.01
and top-1 ≥ 49/50 (97 % of 50, raw; near-ties at zoo's `TIE` 0.1 are listed, not excused). KL > 0.02 means stop
and report. Also reported: fp8-q8 exact, fp8-ref, and `AXON_FP8_CPU_ACT=e4m3` against HF BF16 and HF F32.
(4) `graded_vs_exact` on the FP8 0.5B, fast vs exact on bella, all 4 arms, with fast launches > 0 in each
(batched: 24/160/520 all go through `gemm_f8`). (5) The exact-mode FP8 model passes the existing bit-identity tests
(`gpu_prefill_batchings_bit_identical` and friends). (6) md5: GGUF `gen` main vs branch, exact **and** fast, via
`axon-podjob md5.sh`. Then a draft PR.
**CP2 (7B):** accuracy (3) against the pod's HF BF16. Speed on bella: one `--timing bella` job with 7B FP8 fast and
7B Q4_K_M interleaved, pp512, pp2048 and tg128, medians, MEASURED. Full suites release + debug, clippy
`-D warnings` (bin + tests), rustfmt, CI, and the sm_89 run of (1)+(4). The graded gate can't run on the 7B (three
nets, ~24 GB of VRAM), so (4) stays on the 0.5B.
ESTIMATED on bella: tg ≈ 0.6× Q4_K_M (7.68 vs 4.68 GB read per token at equal DRAM efficiency); pp 0.9–1.3×
(no unpack is worth +25–27 %, MEASURED r17; GeForce f32-accumulate FP8 runs at half the int8 rate, R2).

## 7. Pods (I stage and rehearse on bella; you book)
4090 (sm_89), one session of about 45 min ≈ **$0.67**: (1)+(4) on the 0.5B, then the 7B HF BF16 reference (15 GB
in BF16: too much RAM for bella under the 8 GB rule). If avante (sm_89) is back up, it runs (1)+(4) for $0.
H100, about 20 min ≈ **$1.33**: tg128 7B FP8 vs Q4_K_M. Total ≈ $2.0 of the $7.50. An option that saves the 7B
reference pod: if D7 matches HF to KL ≤ 1e-4 on the 0.5B, the 7B fp8-ref on bella's CPU becomes the 7B reference
(about 7 GB file-backed plus 2.2 GB anon), and the pod run confirms it. Your call.

## 8. Risks
- **R1 (biggest): the graded gate's relative bound.** `assert_graded` asks for mean KL ≤ 2× an exact-vs-exact floor
  (f16 vs q8 KV), which measures a *reordering*. e4m3 activations are a *precision* change. ESTIMATED: about 5× the
  RMS error of exact's int8 per-32 activations (3 mantissa bits vs ~7), so fast vs exact may fail 2× floor while
  every bug-detector bound passes (flips < tie, ≤ 1 position > 0.5, max abs < 6). The ADR left this bound open
  (`numerics.rs` header, ADR §7). Before any GPU kernel exists, D7's emulation measures it. If it bites, I stop and
  report the numbers with a proposed `Floor` for precision changes. I won't loosen the gate myself.
- R2: GeForce `mma` e4m3 with f32 accumulate is half the int8 rate (4090 sheet 330 vs 660 TOPS; ESTIMATED the same
  on sm_120). r17 found the GEMM MIO-bound, so this may not bind. pp might not beat Q4_K_M on bella.
- R3: the mma result for (m,n) could depend on its fragment neighbours (spec §4, untested). The §3 test catches it,
  and `AXON_F8_ZK` is the fallback.
- R4: top-1 on 50 positions is coarse: 49/50 is the bar, so one near-tie flip fails it. Natural-text ΔNLL
  (`numerics_dnll.py`, ≥ 30 windows) is reported beside it.
- R5: the H100 bar doesn't match the scope's own arithmetic. 0.9 × 314 = 283 tok/s needs 2.17 TB/s (65 % of HBM3),
  while the scope's "floor ≈ 250" (0.80×) sits below its own kill line. Please settle the bar before booking.
  Separately, A3 inherits D2's per-128 promotion, which the A1 probe didn't time.
- R6: `axon-run --models` stages `.gguf` files only. The safetensors directories go into the kit with an explicit
  rsync (rehearsed on bella).
- R7: RAM. The fallback converts inside the ring workers, so there's no 8.1 GB anonymous Q8_0 arena for the 7B.
- R8: HF in BF16 rounds the dequantized weights (12 → 8 significant bits), so HF F32 is the faithful reference.
  I gate on BF16 as the scope says, and report F32 beside it.

## 9. Rollback
Everything sits behind `qt == F8_E4M3` or an FP8-only module. Reverting the PR brings back today's refusal
(`safetensors.rs:519`). No GGUF kernel, buffer or fingerprint changes except fast's counter (one constant).
