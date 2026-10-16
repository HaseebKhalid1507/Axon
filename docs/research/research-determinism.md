# research: determinism — batch-invariant inference and Axon's bit-identity contract as a product feature

Read-only research note (Chrollo), 2026-10-16. Axon tree read at the dev checkout (working tree 0851798; main e998d6f
per brief), specs r4-serve / r4-decode-bw / r5-decode / r5-long-prefill / r8-unpack / r9-pick / r10-scope,
`research-r8-bitident.md`, PROGRESS.md; llama.cpp `4413a19` (`ggml/src/ggml-cuda/`); vLLM `main` (fetched raw:
`docs/features/batch_invariance.md`, `vllm/model_executor/determinism/batch_invariant.py`,
`vllm/v1/attention/backends/flash_attn.py`); SGLang blog 2025-09-22; Thinking Machines blog 2025-09-10. PDFs under
a scratch directory (not in the repo). Labels: **MEASURED** (Axon log/spec or a paper's measurement), **CLAIMED**
(paper's own number, not reproduced), **ESTIMATED** (my arithmetic, shown), **CODE** (file:line).

---

## 1. The problem in Axon's terms

Axon's kernels promise (CODE prefill.cu:1–17, decode.cu:4–15, 1049–1084): decode == batched prefill == speculative
verify bit for bit, a request batched with others == alone, run-to-run identical; every GEMM output is an exact
integer block sum followed by one fixed `fma` chain over blocks in order (decode.cu:271–297, `gemv_ref`); attention is
64-position blocks folded in ascending order (`att_fold_w/att_fold_o`, decode.cu:1319–1339). This is exactly what the
2025–26 literature calls *batch invariance* plus the stronger *prefill–decode / chunk / prefix-reuse equivalence* that
Vosti (§2) finds even vLLM's and SGLang's deterministic modes fail on some models.

The question is what the contract costs and what it is worth. **MEASURED cost so far on bella:** 7B decode GEMVs run at
602 GB/s of a 648 GB/s ceiling (93 %; r5-decode §0), qkv 515 / wo 568; 7B prefill GEMM at MMQ parity (29.7 vs 29.3
TMAC/s, r4-prefill-big §1.4) against a llama.cpp kernel that *does* use stream-K; decode attention at 32k is
latency-bound at 350 GB/s (r9 §0) with a fix (r9 #1) that keeps the order. The one place the contract measurably bites is
the **CPU**: r8 (free float order) 1.5B pp560 734 vs bit-identical r8-bi 458–574 vs head 343–417 (PROGRESS 336–342,
388–391, 426, 525; noisy). Known forbidden speedups: split-K GEMV (r4-decode-bw §3), split-KV / 128-position blocks /
f16 accumulate in prefill attention (r5-long-prefill §4.1–4.2, ≤ 2 % e2e), stream-K GEMM. Compared with what vLLM and
SGLang pay for the same promise (25–63 %, §2–3), Axon's determinism is close to free on the GPU; the work is to (a)
recover the few percent the contract still holds and (b) turn the property into a tested, exposed guarantee.

---

## 2. Literature map

Read in full (PDF, method + numbers): TML blog, LLM-42, CoRun, Vosti, TBIK, MarginGate, vLLM docs/source, SGLang blog.
Abstract + key sections: the rest.

| # | Paper | The idea | Numbers (hardware) |
|---|---|---|---|
| 1 | **Defeating Nondeterminism in LLM Inference** — He / Thinking Machines, blog, 2025-09-10 | Run-to-run determinism of kernels ≠ *batch invariance*; the reduction order of a token must not depend on batch size or on how many of its sequence-mates are co-processed. Recipe: one tile config for all M (no split-K), RMSNorm one row per core, attention with **fixed split-size** (not fixed split-count) split-KV so chunked prefill and decode reduce identically. Stream-K is worse: not even batch-*position*-invariant. | MEASURED: Qwen3-235B, 1000 greedy completions → 80 unique without, 1 with. Perf: Qwen3-8B, 1000 req, vLLM default 26 s, batch-invariant 55 s, with improved attention 42 s (**+62 %**). Triton batch-invariant GEMM ≈ 20 % below cuBLAS. (H100-class.) |
| 2 | **Towards Deterministic Inference in SGLang** — LMSYS blog, 2025-09-22 | TML ops + fixed split-KV size in FlashInfer/FA3/Triton, chunked-prefill truncation aligned to the split size, CUDA graphs, radix cache, seeded non-greedy sampling. | MEASURED (Qwen3-8B, H100): slowdown 24–55 %, avg **34.35 %** (FlashInfer/FA3); CUDA graphs 2.8× over the naive integration. "Recommend for debugging and reproducibility." |
| 3 | **LLM-42: Enabling Determinism with Verified Speculation** — Gond, Kamath, Ramjee, Panwar, SOSP 2026, arXiv:2601.17768 | Decode on the fast nondeterministic path, periodically *verify* a window with fixed-shape batch-invariant kernels, roll back on mismatch (speculative decoding turned into a determinism mechanism). | MEASURED (Llama-3.1-8B, H100 PCIe): batch-invariant Triton GEMM 194 vs cuBLAS 527 TFLOPS (**−63 %**); SGLang deterministic mode −24…−36 % throughput; one deterministic request in a batch of 11 collapses throughput 56 %. LLM-42 within 3 % of nondeterministic; recomputation overhead 6.8 % at window 512. |
| 4 | **CoRun: Padding is Simple and Efficient for Deterministic LLM Inference** — Zhao et al., arXiv:2608.14376 (2026) | Most kernels are *position*-invariant though not batch-invariant: pad decode to one fixed captured shape, isolate prefill, keep fast kernels. | MEASURED (vLLM 0.25, Qwen3-235B/DeepSeek-V3/Hy3): batch-invariant kernels raise a 32-request decode step 19.4 → 88.4 ms, throughput −74 %; CoRun 15–324 % over BI. Ablation: disabling split-KV alone costs **15–41 %** on ragged batches, 1–9 % on uniform lengths. |
| 5 | **Vosti: Specifying, Implementing, and Verifying Deterministic LLM Inference** — Qin et al. (Duke), arXiv:2609.38981 (2026) | Formal spec: same prompt + sampler state → bitwise-identical logits per position across batching, chunking, prefill/decode split, KV reuse/eviction. Verus proof of the engine, Triton kernel analyzer. | MEASURED test table (Llama-3.1-8B / Gemma3-4B): vLLM default passes 177/296 batch, 49/90 chunk, 53/384 prefill–decode; **vLLM batch-invariant on Gemma3-4B: 193/296, 0/90, 5/384**; SGLang det-Triton 3/384 prefill–decode. Vosti 1.36–3.19× vLLM-BI decode throughput on H200, still slower than default modes. |
| 6 | **Deterministic Inference across Tensor Parallel Sizes (TBIK)** — Zhang et al., arXiv:2511.17826 (2025/26) | Tree-based invariant GEMM + all-reduce so TP=1 training and TP=k rollout agree bitwise (RL training–inference mismatch). | MEASURED (H20 TP=4): batch-invariant ops +10–33 %, TBIK another +5–30 %, total **+22–63 %** latency vs BF16 cuBLAS. |
| 7 | **MarginGate** — Chu, Zhou, Zhang, arXiv:2605.30218 (2026) | Batch-induced token flips are sparse (0.3–1.3 % of decode steps on MATH500/GSM8K/HumanEval); verify only low top-1/top-2 margin steps. | CLAIMED: Llama-3.1-8B flips 0.48 % of steps; cuts LLM-42's latency increment 2.0–2.2×. |
| 8 | **Non-Determinism of "Deterministic" LLM Settings** — Atil et al., arXiv:2408.04667 (2024/25) | Empirical: 5 LLMs × 8 tasks × 10 runs at temperature 0. | MEASURED: accuracy varies up to 15 % across runs, best–worst gap up to 70 %; no model gives identical strings. |
| 9 | **Understanding and Mitigating Numerical Sources of Nondeterminism (LayerCast)** — Yuan et al., arXiv:2506.09501 (2025) | Batch size, GPU count and GPU version change reasoning outputs; LayerCast = BF16 weights, FP32 compute. | MEASURED: DeepSeek-R1-Distill-Qwen-7B up to 9.15 % accuracy std across configs, 9k-token length swings. |
| 10 | **Accelerating the Mitigation of LLM Inference Nondeterminism Across GPU Architectures** — Cooper et al., arXiv:2609.25624 (2026) | Fused-upcast FP32 GEMMs with a reduction order that is a function of the problem shape only (not SM count) → cross-GPU bit identity. | CLAIMED 1.17–3.1× faster than LayerCast; still an FP32 path. Shows the SM-count dependence is the cross-GPU culprit. |
| 11 | **HEAL** — Zhu et al. (UCLA), arXiv:2606.21023 (2026) | Cross-GPU divergence from downcasts at kernel boundaries; INT16 Q/K/V + error-compensated 16-bit GEMM. | CLAIMED: FP32 pipeline 4.2–13.1× TPOT; HEAL 40–109 %. |
| 12 | **Greedy Decoding Is Not Precision-Invariant** — Du et al., TMLR 2026, arXiv:2609.26621 | BF16 vs FP16 greedy diverge on 49–100 % of prompts; FP32 lm_head on low-margin steps recovers much at < 4 % cost. | MEASURED on A10G/L4/A100. |
| 13 | **The Integer Alibi** — Chen, arXiv:2608.13756 (2026) | vLLM CUTLASS vs Triton INT8 GEMM: each bit-reproducible, 0/64 agreement. INT32 dot is exact and order-free, so the divergence is entirely in the float epilogue (scale application). | MEASURED on Qwen3-1.7B/8B. **Directly Axon's design argument**: integer core + canonical float epilogue. |
| 14 | **Taming Bitwise Behavior in GPU Kernels with Tensor Core** — Yang et al. (GT/Meta), arXiv:2609.11356 (2026) | GEMMDesc: a formal descriptor of reduction order; compiler enforcement; matches cuBLAS bit-for-bit on GB300/H100. | CLAIMED: pinning order costs "about 20 %" of GEMM throughput in prior work; their kernels within 10 % of free-order on 19/27 shapes. |
| 15 | **Hawkeye: Reproducing GPU-Level Non-Determinism** — Badash, Boneh, Komargodski, Srivastava, arXiv:2603.20421 (2026) | Reverse-engineers tensor-core f16/bf16/fp8 accumulation (rounding, subnormals, alignment at max exponent −132/−133 Hopper) so a CPU reproduces mma bit-exactly. | MEASURED: 100 % bit-exact on 100k random 16×16 tiles, Ampere/Hopper/Ada. Relevant to Axon's f16 `mma.sync` attention: internal accumulation is architecture-specific. |
| 16 | **TOPLOC** — Ong et al. (Prime Intellect), arXiv:2501.16007 (2025) | Verifiable inference by locality-sensitive hashing of top-k final-layer activations every 32 tokens (8 B/token); tolerates nondeterminism by thresholds. | CLAIMED 100 % detection of model/prompt/precision swaps, 1000× smaller proofs than storing activations. |
| 17 | **TensorCommitments** — Baser et al., arXiv:2602.12630 (2026) | Cryptographic commitment to inference (Terkle trees); verifier needs no GPU. | CLAIMED 29–67× faster than Verkle; still heavy. |
| 18 | **Stream-K** — Osama, Merrill, Cecka, Garland, Owens, PPoPP 2023, arXiv:2301.03598 | Split the MAC-loop iteration space evenly across CTAs, fix up partial tiles. The decomposition depends on SM count and problem shape. | The source of llama.cpp's MMQ/FA tail-efficiency; inherently not batch-invariant (ref. 1). |
| 19 | **Impacts of floating-point non-associativity on reproducibility for HPC and DL** — Shanmugavelu et al., SC-W 2024, arXiv:2408.05148 | Survey + measurements of atomics-induced variability in PyTorch/HPC. | Background. |
| 20 | **FoldAttention** — Achanta, arXiv:2609.33410 (2026) | Fixed softmax reference instead of online max → partials add without rescale; gates key/value reads. | CLAIMED 1.36–2.30× decode on H100 at matched error; quotes the "determinism tax" of FA3 backward at up to 38 %. Lossy (skips keys) — opt-in territory only. |
| 21 | **Vitamin-E** — Yang et al., arXiv:2603.15042 (2026) | GPU spatial sharing that keeps kernels' parallel structure independent of transient resource state. | Background: timing may vary, structure must not. |
| 22 | **Re-feeding Is Not Replaying** — Matteson, arXiv:2606.15621 (2026) | Re-feeding a transcript prefix ≠ the KV state the model passed through: changes token-credit estimates 14–28 pp above the replica floor. | MEASURED on stock engines. Axon's prefix reuse == fresh prefill makes this noise zero by construction. |

Threads through the map: (i) 2024–25 established the damage (8, 9, 12); (ii) Sep 2025 gave the recipe and the price
(1, 2: 34–62 %); (iii) 2026 is about *reducing the price* without the kernels — speculation (3, 7), scheduling (4),
specification (5), TP (6); (iv) cross-GPU identity is a separate, harder problem (10, 11, 15); (v) verification wants
exactness it can't get (16, 17, 22). Axon sits at the end of (ii) having never paid the price, which is why (iii) is
irrelevant to it and (v) is an opportunity.

---

## 3. What production engines do

**llama.cpp (CUDA, `4413a19`)** — not batch-invariant, not prefill==decode, not hardware-portable, and says so:
* `tools/server/README.md:589`: `cache_prompt` — "logits are **not** guaranteed to be bit-for-bit identical for different
  batch sizes (prompt processing vs. token generation) … can cause nondeterministic results."
* GEMM path switches by token count: `mmvq.cuh:3` `MMVQ_MAX_BATCH_SIZE 8` → `ggml-cuda.cu:1834–1835` mmvq for ≤ 8
  tokens, MMQ above. Two kernels, two float orders ⇒ decode ≠ prefill.
* MMQ uses **stream-K**: `mmq.cuh:1443–1449` grid = `nsm` blocks unless tile efficiency ≥ 90 %, fixup kernel
  `mul_mat_q_stream_k_fixup` (1243) when `ntiles_dst % grid != 0`; the K split point of a CTA is `kbc = blockIdx.x ·
  (… ntx·nty·blocks_per_ne00)/gridDim.x` (1068) — a function of the token count *and* the SM count. Same shape on a
  46-SM and a 128-SM GPU reduces in different orders.
* Flash attention: `fattn-common.cuh:1145–1190` stream-K on Ada+ (`should_use_stream_k` returns true for `cc >=
  ADA_LOVELACE`), block count `min(max_blocks_per_sm·nsm, …)`, `flash_attn_stream_k_fixup_*` (731, 815). Non-stream-K
  path: `parallel_blocks` picked by an efficiency search over `ntiles_dst` (= batch × heads) and `nsm`
  (1196–1215), merged by `flash_attn_combine_results` (924). Both depend on batch and hardware.
* Deterministic run-to-run on one machine with the same batch schedule, yes (no atomics in the hot kernels: the only
  `atomicAdd`s are `top-k.cu`, `count-equal.cu`); invariant to batch composition, no.

**vLLM (`main`, Oct 2026)** — `VLLM_BATCH_INVARIANT=1`, documented beta (`docs/features/batch_invariance.md`):
* Ampere/ROCm: Triton persistent matmul overrides for `aten::mm/addmm/matmul/linear` (`batch_invariant.py:1318–1328`);
  Hopper/Blackwell: cuBLASLt with `CUBLAS_WORKSPACE_CONFIG=:16:8`, `CUBLASLT_WORKSPACE_SIZE=1` — "the only source of batch
  variance is split-k, which we disable via the workspace config" (1330–1335). Doc item 5: tile configs may still
  follow the runtime batch size (breakable CUDA graphs) — i.e. they rely on M-tile choice not changing the K order.
* Attention: FlashAttention with `num_splits=1` when batch-invariant (`flash_attn.py:862–863, 1705`) — **no split-KV at
  all**, decode attention serial over the KV length per (query, head); AOT scheduling disabled (829–831).
* Softmax/log-softmax/mean/RMSNorm/bmm replaced by Triton batch-invariant kernels; TF32 off; LoRA shrink in 8 fixed K
  chunks; TP all-reduce pinned to the 1-stage custom kernel, NCCL pinned to Tree/Simple/1 channel (1391–1426).
* Cost: not stated in the doc ("may impact performance"); third-party MEASURED: +62 % (TML, ref 1), −74 % throughput at
  32 ragged requests (CoRun, ref 4), 1.36–3.19× slower decode than Vosti on H200 (ref 5).

**SGLang** — `--enable-deterministic-inference`: TML ops, fixed split-KV size per backend (FlashInfer
`fixed_split_size`, FA3 `num_splits=1`, Triton fixed decode split), chunked-prefill truncation aligned to the split size
(blog §Technical Details). MEASURED 24–55 % slower, avg 34.35 %, Qwen3-8B H100. Vosti finds its Triton backend fails
prefill–decode equivalence (3/384).

**TensorRT-LLM** — no batch-invariant mode in its docs that I could verify; not claimed here.

**Axon (for comparison, CODE)** — one float order for every GEMM path (`gemv`, `gemvn`, `gemvm`, `gemm2`, `gemv_ref`,
prefill.cu:1–17, decode.cu:271–297), chosen at nb ≤ 16 / > 16 (`batch.rs:371` "Same bits either way") because the
integer part is exact and only the per-block float epilogue is order-defined; attention split at a **fixed 64-position
size** with a sequential fold (decode.cu:1052–1073) = exactly the TML "fixed split-size" strategy, shared by prefill's
`attn_fa_b` (register fold) and decode's `attn_blk` + `attn_cmb` (partials + fold); no atomics anywhere in arithmetic
(r4-decode-bw §5; the only atomics are tickets/timers). Tests: `gpu_products_match_reference_bits`
(cuda/qwen2/tests.rs:2544), `feed_batch_matches_feed*` (model.rs:3516–3558), md5 gates, serve HTTP C=1/4/8 == solo
(PROGRESS 873, 926). Sampling: seeded xoshiro, llama.cpp's order, exact radix top-k on device (sample.rs:2–30).

---

## 4. Proposals for Axon, ranked

### P1. Make the contract a product: identity tuple, `system_fingerprint`, and a published determinism spec (do first)
* **Change**: (a) `serve.rs` / `engine.rs`: compute an *identity tuple* at load = (GGUF file hash, backend {cuda|cpu|hybrid},
  KV type, GPU arch class, kernel-source hash from the NVRTC cache key, axon version, sampler version); expose it in
  `/v1/models` and as the OpenAI `system_fingerprint` field on every response (clients already parse it). (b) A
  `docs/determinism.md` stating the guarantee in Vosti's terms (§2 #5): for a fixed tuple, (prompt ids, params, seed) →
  identical token sequence regardless of concurrency, batch position, prefill chunking, prefix-cache hit/miss, drafting,
  slot, restart. State the exclusions (r4-serve §6): CPU ≠ CUDA, f16 ≠ q8, hybrid decode = CPU numerics, arch class.
  (c) A `serve_identity` test as specified in r4-serve §6 (randomized batch composition, chunk sizes 16..512,
  prefix-cache on/off, drafting on/off, forced eviction) — the current gates cover C=1/4/8 == solo (PROGRESS 873) but
  not chunking/eviction permutations.
* **Gain**: the differentiator becomes checkable. No other local engine can state this (llama.cpp disclaims it,
  README:589); vLLM/SGLang charge 25–60 % for a weaker version.
* **Bits**: unchanged. **Risk**: low; the test may find a real hole (good). **Builder-hours**: 10–16.
* **Gate**: `serve_identity` passes 100 % over ≥ 200 randomized schedules on 7B and 0.5B, f16 and q8; fingerprint
  stable across restarts, different across `--kv` and arch.

### P2. Exact output attestation: response hash + optional activation commitment
* **Change**: per response, `attestation = H(identity_tuple ‖ prompt_ids ‖ params ‖ seed ‖ output_ids)`; opt-in
  `commit: true` also returns a TOPLOC-style commitment (top-128 (index, value) of the final hidden state every 32
  tokens, 8 B/token, ref 16) — but **exact**, since a replay on the same tuple reproduces the bytes. `axon verify
  <trace.json>` replays and diffs with zero tolerance. Hook: the head slices already pass through `samp_tail`/`argmax_b`
  (batch.rs HEAD_SUB); the hidden row is in `act` before the head GEMV.
* **Use cases** (concrete): agent replay (re-run a Synaps/opencode session from its trace and get the same tool calls —
  ref 22 measures 14–28 pp credit noise on engines that can't); eval reproducibility (ref 8: 15 % accuracy swings
  vanish; a benchmark run is a hash); regression testing of Axon itself (every PR's md5 gates, promoted to a user
  feature); verifiable outputs for a remote Axon box (a client with the same tuple re-derives the commitment).
* **Bits**: unchanged. **Risk**: low; must document that attestation proves *this tuple produced this*, not
  cross-hardware truth (§6 Q1). **Hours**: 12–20. **Gate**: 1000 replays, 0 mismatches; commitment ≤ 8 B/token.

### P3. Deterministic fixed-K-chunk split-K for narrow GEMVs (a one-time redefinition of the canonical order)
* **Idea**: redefine the chain as two-level with a **fixed chunk size in blocks** (TML's fixed split-size, applied to K):
  `acc_c = chain over blocks [cC, (c+1)C)`, then `acc = ((acc_0 + acc_1) + acc_2) + …` in order. Any kernel (decode
  GEMV, `gemvm`, `gemm2`) computes the same two-level sum, so decode == prefill holds; the split is a function of K
  only, so batch invariance holds. Decode may then run chunks on different blocks and combine in a tiny second kernel
  (or in the last-arriving block, order fixed by chunk index, not arrival). C = 8 blocks (2048 K values for K-quants,
  256 for Q8_0) gives 7B K=3584 → 2 chunks, 18944 → 10.
* **Where it pays (MEASURED shapes)**: only matrices with fewer rows than resident blocks: 7B k/v (512 rows), 0.5B
  k/v (128), 1.5B k/v (256) and the whole 0.5B/1.5B/3B set that r4-decode-bw §2 found at 272–284 GB/s. 7B: qkv+wo =
  462 MB/token at 515/568 GB/s → 0.86 ms; at gate_up's 620 GB/s → 0.745 ms: **ESTIMATED −0.11 ms of 7.77 = 1.5 %**.
  3B: small-N 0.69 ms → ~0.35 ms at 600 GB/s: ESTIMATED up to 9 % (before r5 P1's partial fix, so call it 4–8 %).
  0.5B: bounded by ~115 launches × 3–4 µs fixed latency (r4-serve §1: ≥ 60 % of the step is fixed) → ESTIMATED 3–6 %.
* **Bits**: *changes the GPU md5s once* (CPU unaffected: CPU/GPU agree to tolerance, not bits — decode.cu:286–287).
  Must land in decode.cu, prefill.cu (`gemvn/gemvm/gemm2` epilogues), `gemv_ref`, and `gpu_products_match_reference_bits`
  together (r4-decode-bw §5 spells out this exact condition). The two-level sum is at least as accurate (shorter chains).
* **Risk**: medium — every GEMM kernel's epilogue; the r5 history shows "more blocks" is not free (persistent GEMV lost).
  **Hours**: 30–50. **Gate**: tg128 on 0.5B/1.5B/3B ≥ +3 % (≥ 5 interleaved rounds, cool-gated), 7B/8B ≥ −0.5 %;
  `gpu_decode_matches_batched`, `feed_batch_matches_feed`, md5 suite regenerated once and then frozen.
* Rank 3 not 1 because the yield on the flagship 7B/8B is ≤ 2 % and the order change is a one-time break of every
  stored md5 (document it in the identity tuple's kernel hash — P1 makes that harmless).

### P4. Hierarchical fold in decode attention — only after r9 #1, only if `attn_cmb` is bandwidth-bound
* **Idea**: keep 64-position blocks; fold them sequentially inside fixed chunks of F blocks (e.g. F = 4 → 256 positions)
  within one warp, write one partial per chunk; the combine folds chunks in order. Same two-level trick as P3, mirrored
  in prefill's register fold (`attn_fa_b` just re-associates at chunk boundaries — cheap).
* **Gain**: partial traffic per kv-head per block = 7 q-heads × 130 × 4 B = 3.6 KB vs 32 KB of KV (11 %, ESTIMATED from
  the shapes): 7B@32k ≈ 207 MB written + read per token ≈ 0.6–0.7 ms at 600 GB/s; F = 4 cuts it 4× → **ESTIMATED ≤ 0.5
  ms of 13.8 ms (≤ 3.6 %) at 32k, ~0 at 8k**. r9 #1 (order-preserving, ESTIMATED 5.3 → 3.3 ms) comes first and may
  make this moot. **Bits**: GPU md5 changes once; decode/prefill mirrored. **Hours**: 16–24. **Gate**: tg@32k 7B/0.5B
  ≥ +2 % over r9 #1's result; tg128 flat.

### P5. Cross-architecture identity audit (cheap, informative)
* Run the md5 suite on bella (sm_120), avante (sm_89), a 4090 pod (sm_89), A100 (sm_80), H100 (sm_90) and *diff
  across hosts* — today's rule is base-vs-candidate same host (BUILDER-RULES). Expect: GEMM outputs identical (integer +
  IEEE `fma/fmul/fdiv` are architecture-exact); attention may differ via `ex2.approx.ftz` (decode.cu:1095–1099) and the
  f16 `mma.sync` internal accumulation (Hawkeye, ref 15: alignment at −132 vs −133 on Hopper). PROGRESS 906 (two
  tolerance tests fail on every sm_89 host but pass on bella) says a difference exists.
* **Change**: none in kernels; 2–4 h + a pod. If attention is the only divergence, replacing `ex2.approx` by a
  software exp2 with a fixed polynomial is a known, bounded change (vmath.rs did it on the CPU) and would make the tuple
  key "GPU, any sm_80+" instead of "arch class" — worth knowing before marketing cross-host replay (P2).

### P6. Opt-in `--numerics fast`: **CPU only**
* **Where it pays**: the CPU K-quant prefill, MEASURED on branches: 1.5B pp560 734 (r8, free order) vs 458–574 (r8-bi)
  vs 343–417 (head); research-r8-bitident §3–4: the bit-identical r8-bi keeps ~40 % of r8's uop cut. That is a 1.3–1.6×
  CPU prefill on ≥ 1.5B, the single largest thing the contract holds anywhere in Axon.
* **Where it does not pay (GPU)**: split-K (P3's ≤ 2 % on 7B is recoverable *deterministically*); stream-K GEMM (llama.cpp
  has it and is at kernel parity, r4-prefill-big §1.4); prefill split-KV (≤ 2 % e2e, r5-long-prefill §4.1); f16
  accumulate / 128-blocks (≤ 1.5 % 7B, accuracy risk, §4.2); adaptive split-KV in decode (P4's ≤ 3.6 %); lossy attention
  (FoldAttention-style key skipping, ref 20: CLAIMED 1.4–2.3× on H100 BF16 — a quality change, and at 32k attention is
  38 % of the 7B step, so even 2× = ~19 % tg; the only GPU item with a double-digit ceiling, and it is not a
  "numerics" mode but an approximation mode).
* **Recommendation**: ship nothing non-deterministic on the GPU; ship `--numerics fast` for the CPU backend as a
  documented second identity tuple (fingerprint differs), default off, after r8-bi lands. **Hours**: the r8 branch
  exists; 8–12 to gate it behind the flag and the tuple. **Gate**: CPU 1.5B/3B/7B pp560 ≥ +25 % vs r8-bi; KL vs
  bit-identical path ≤ 1e-3 (PROGRESS 346: N3's 2.3e-3–5.8e-3 was accepted as noise-level).

---

## 5. Dead ends

1. **LLM-42-style decode-verify-rollback and MarginGate** (refs 3, 7): they exist to get determinism out of a fast
   nondeterministic path. Axon's fast path *is* the canonical path; adding a verifier buys nothing and costs 6.8 %
   recomputation (ref 3 MEASURED).
2. **CoRun-style padded fixed-shape decode + isolated prefill** (ref 4): Axon's kernels are batch-invariant per row, so
   padding to a fixed shape is pure waste; CoRun falls below BI at padding > 64 for a single request (ref 4 §F).
3. **Pinning one GEMM tile config for all M** (TML / vLLM): necessary when the mma accumulates floats; irrelevant for
   Axon, where the tile and the instruction (dp4a vs `mma.sync` int8) do not touch the bits. Keep `gemvm`/`gemm2`
   switching freely; the Integer Alibi paper (ref 13) is the formal argument — and its warning: the *epilogue* must be
   fixed, which Axon's `gemv_ref` does and vLLM's CUTLASS/Triton pair did not.
4. **`num_splits=1` decode attention** (vLLM's batch-invariant attention): serial over the KV length per row — Axon's
   fixed-64 split + fold is strictly better and already deterministic; never regress to it.
5. **Stream-K GEMM with "deterministic fixup"**: the fixup order can be made fixed by partial index, but the *split
   points* are still functions of grid size (SM count) and token count (llama.cpp mmq.cuh:1068): batch-dependent by
   construction. A fixed-split-size variant degenerates into P3.
6. **FP32 everywhere / LayerCast / HEAL** (refs 9–11): the cross-GPU problem they solve comes from bf16/fp16 float
   accumulation; Axon's GEMMs are integer-exact with f32 epilogues already, so there is nothing to upcast there.
7. **Cryptographic proof of inference** (ref 17): orders of magnitude too slow for a local engine; exact replay (P2) is
   the honest verification Axon can offer.
8. **Atomics-based fusions** (last-block tickets in r4-decode-bw Stage 2): fine for *who* runs an epilogue, never for
   arithmetic — already Axon's rule (r4-decode-bw §5); the dec2 attempt lost on speed anyway (PROGRESS 847).

---

## 6. Open questions a 1-hour experiment could settle

1. **Cross-arch md5** (P5): are bella's and a 4090's / A100's md5s equal? Which kernel diverges first (dump logits of
   layer 0 after attention vs after the GEMMs)? Decides whether "GPU arch class" stays in the identity tuple.
2. **How much of the 7B 515 GB/s qkv rate is parallelism?** Run `cuda_gemv_bench` on the 512×3584 k/v shape with the
   rows split across 2× and 4× as many blocks writing *separate* partials (no combine, bits ignored, bench only): if GB/s
   does not rise, P3 is dead on 7B and only the 0.5B/1.5B case remains.
3. **Is `attn_cmb` bandwidth- or latency-bound at 32k?** `%globaltimer` stamps (r5-decode's timeline patch) on 7B at pos
   32511: in-kernel chain time vs bytes. Decides P4 after r9 #1.
4. **Does the serve path already pass Vosti's four tests?** Script: same 50 prompts (a) alone, (b) in random batches,
   (c) chunked at 16/64/512, (d) continued after eviction and re-admission; compare *logits* (not tokens) via
   `samp_read` with top-k 0. If 4/4 pass today, P1 is documentation + a test harness, not engineering.
5. **`ex2.approx.ftz` vs a polynomial exp2 in the fold**: micro-bench the attention kernel with a 2-ulp software
   exp2 (vmath.rs's `exp8` ported) — if the cost is < 2 % of attention, cross-arch identity becomes cheap to buy.
