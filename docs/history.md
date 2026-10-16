# Engineering history

A chronological log of what was built, why, how it was checked and what it measured, from the first
commit to main `a591e17` (2026-10-16 16:22 EDT), plus the open PRs at that point. Sources: the PR
bodies (`gh pr view`), `git log`, the foreman's log (PROGRESS.md), the round specs, the builders'
NOTES.md and the scoreboards. None of those except the PRs and git history are in the repo. The method
behind all of this is in [methodology.md](methodology.md); everything that was rejected is collected,
with numbers, in [experiments-rejected.md](experiments-rejected.md).

Conventions:

- Dates and times are EDT (UTC−4), the merge-commit time on main. GitHub shows UTC, so a PR merged at
  22:41 EDT shows the next day there.
- "bella" is the dev laptop: Core Ultra 9 275HX + RTX 5070 Ti Laptop, sm_120, 12 GB. BENCH.md and START-HERE call
  it Parabellum. Other hosts are named in [methodology.md](methodology.md#hardware).
- Numbers are tok/s unless stated, and MEASURED unless marked ESTIMATED. "A → B" is before → after on
  the same host in the same run. Where a run was on a hot or shared machine, only the ratio means
  anything; that is noted.
- Status is one of merged, rejected, superseded, closed, open.

## Phase 0: the CPU engine before the loop (2026-10-10 → 10-13, commits to main, no PRs)

| date | commit(s) | what | result (bella CPU, 0.5B Q4_K_M) |
|---|---|---|---|
| 10-10 | 7f605e0 … 648024b | repo, GGUF parser, mmap loading, a ratatui model viewer | — |
| 10-12 | fe65766 | qwen2.5-0.5B forward pass end to end; "The capital of France is" matches ollama 16/16 tokens, max \|Δlogprob\| 0.097 | 2.9 tok/s, one core |
| 10-12 | 699e36e | reference data committed (gguf-py rows, HF token ids, ollama greedy paths); `matches_ollama` test (0.15 logprob) | — |
| 10-12 | 4026a79, 377d53d, e4f10bc | rows split across cores (rayon); 8-bit activations + integer dots; AVX2/AVX-VNNI kernels with runtime dispatch | decode 2.9 → 23.4 → ~50 → ~52 |
| 10-12 | 39f8883, af8b57e | fused q/k/v and gate/up passes; persistent spin-then-park pool replaces rayon (rayon's wake-ups cost 30–60 µs per matvec) | ~52 → ~72 → ~103 (ollama CPU ~74 alongside) |
| 10-12 | 5a1b81c … 323547c | branch-free rounding, per-head attention tasks, RoPE table, fused SwiGLU, vectorized abs-max, O(n) streaming output, fast argmax | ~102 → ~136 → ~168 (n=64) |
| 10-13 | 0f55cb3 … 74fe85e | prefill without the output head; `feed_batch` (batched prefill, bit-identical to `feed`); 4-token and 2×4 tile kernels; x86-64-v3 build; VNNI unpack-once | pp560 ~146 → ~229 → ~300 → ~494 → ~593 |

Every step kept logits bit-identical or within the ollama check. **Baseline at 74fe85e** (BENCH.md, quiet,
3 rounds, 18 threads): axon pp560 1295 / pp2100 1069 / tg128 164 / tg128@1 29.9; ollama 0.35.0 1027 / 961 /
162 / 34.4. With 6 busy processes alongside, axon fell to pp560 370 / tg128 32.

## Phase 1: CPU engine, the overnight gauntlet (2026-10-13)

The brief (PROGRESS.md): beat ollama 0.35's bundled llama.cpp and llama.cpp master on bella's CPU, one
piece at a time. Each piece got a builder, a harsh critic and an interleaved bench against the baseline
binary, and output had to stay bit-identical. Pieces that weren't faster were reverted.

### #1 Overnight: bench harness, pool under load, kernels, attention, decode, hygiene, drafting — merged 10-13 15:09
- **Why:** the baseline lost single-thread decode to ollama (29.9 vs 34.4) and collapsed under load (pp560 −71 %,
  tg128 −80 % with 6 busy processes).
- **What (17 commits):**
  - `scripts/bench.py`, an interleaved harness, and later v2 with `sweep.py`
  - K-quant pair kernels (unpack once per row pair)
  - V-sum for all q heads of a kv head in one pass
  - weights in anonymous huge pages
  - an adaptive pool (drops a worker after stalls, probes back up)
  - long-context decode attention in (kv head, t-block) tasks
  - multi-row decode matvecs with unsigned lanes and 8 KiB software prefetch
  - a fix for a nested pool run that deadlocked under `AXON_THREADS=1`
  - GGUF validation and CLI hygiene
  - lossless prompt-lookup drafting (`--draft K`, off by default)
  - a baseline x86-64 build with runtime dispatch, and a qemu Nehalem CI step
- **Verified:** every commit had tests, clippy, fmt and `gen` md5 on 3 prompts at 0.5B and 1 at 1.5B equal to 74fe85e.
  CI caught an i8 overflow in a test under debug assertions (fixed in aa9be77). From then on the verify also ran
  the debug build.
- **Results (bella CPU, 18 threads):**
  - Pool under 6 busy processes: pp560 363 → 744, tg128 29 → 125. AXON_THREADS=24 quiet: 596 → 994.
  - Decode kernels: 0.5B tg128@1 31.8 → 58.5, 1.5B 20.3 → 23.7. tg128 at 18 threads flat (DRAM-bound).
  - Attention: pp2100 768 → 832 (+8.3 %), light load. Decode attention at 2k: tg128 108.4 → 129.0.
  - Huge pages: tg128@1 28.8 → 31.5.
  - Each engine at its best thread count (sweep at aa9be77): 0.5B pp560 1345 vs ollama 1037 vs llama.cpp
    597; tg128 192 / 163 / 65. 1.5B pp560 382 / **397** / 338, tg128 71.8 / 67.7 / 34.0. The one loss is
    1.5B prefill.
- **Rejected along the way:** token tiling in attention, pool policies r2–r5, parallel exp, the long-prefill query
  tiles, decode V-sum whole-line tasks, verify-batch kernels, hard P-core pinning, and the critic's
  per-line prefetch. Details in [experiments-rejected.md](experiments-rejected.md#cpu-overnight-2026-10-13).
- **Bug found and fixed:** b8a04b6 turned drafting on by default although its message, the PR body and
  BENCH.md said off. It ran that way from ~03:37 until it was found at 14:15 and fixed in b6ebad3.
- **Status:** merged (Haseeb).

### #3 r8bi: bit-identical interleaved K-quant kernels — closed 10-13 18:17
- **Why:** research-r8-bitident.md found that a bit-identical version of the r8 interleave was feasible,
  estimated at ~1.2× on K-quant prefill.
- **Result (bella, 1.5B, 18 threads, 5 rounds):** pp560 417 → 470 (+13 %), tg128@1 23.3 → 29.3 (+26 %), but
  pp2100 362 → 346 (−4 %, noisy) and tg128 74.5 → 72.0 (−3 %). An earlier round reached pp560 +27 %; why the
  later rounds lost that was never pinned down.
- **Status:** closed when #2 merged, by Haseeb's rule ("if #2 merges, close #3"). Branch `r8bi` kept.

## Phase 2: numerics (2026-10-13)

### #2 numerics: interleaved kernels, FMA attention + vectorized exp, f16 KV — merged 10-13 18:14
- **Why:** on every model ≥ 1.5B the K-quant kernels were the engine (q4_k pairs = 59 % of 1.5B prefill), and
  1.5B prefill lost to ollama. Long-context prefill was FP-bound in attention, where the bit-identity rule
  forbade FMA. At 8k context, decode spent ~206 MB/token on the f32 KV cache.
- **What:** r8 8-row interleaved Q4_K/Q6_K and r8q Q5_0/Q8_0 with integer-first GEMV/GEMM, FMA in attention and a
  vectorized exp (0.90 ulp max), an f16 KV cache, and a huge-page arena with T1 prefetch. Runtime dispatch was
  kept. Opt-outs: `AXON_REPACK=0`, `AXON_KV=f32`.
- **Verified:** teacher-forced A/B over 2560 positions in 8 documents:
  - FMA + exp: every ΔNLL within ~1.3 SE of zero.
  - f16 KV: 0.5B ΔNLL +0.0011 ± 0.0006, top-1 96.4 %; 1.5B −0.0023 ± 0.0031.
  - `matches_ollama` still passes at 0.15.
  - After the rebase, 16 path configs and 24 gen md5s equal the pre-rebase build.
- **Results (bella CPU, 18 threads):**
  - 1.5B pp560 400 → 734 (r8); 0.5B pp560 1376 → 1728 (r8q); 0.5B pp8400 603 → 733 and tg128@8k 85 → 97 (f16 KV).
  - Single-thread decode 0.5B 53.6 → 82.5, 1.5B 21.0 → 29.7 (arena).
  - vs main: 0.5B pp560 1053 → 1539, 1.5B 397 → 685. tg128 at 18 threads unchanged (DRAM-bound).
- **Status:** merged by Haseeb. This changed the contract from "logits never change" to "deterministic per
  build/CPU class, with opt-outs" ([methodology](methodology.md#the-bit-identity-contract-and-why-it-shapes-every-decision)).

## Phase 3: chat (2026-10-13)

| PR | merged | what / why | verified | status |
|---|---|---|---|---|
| #4 chat | 10-13 18:28 | `axon chat`: one KV cache per conversation (a turn feeds only new tokens), ChatML exactly as Qwen2.5's template, Ctrl-C stops a reply | 7 tests: every reply equals a fresh prefill of the same history, bit for bit; 107 tests; gen md5 = #2's | merged |
| #5 CI | 10-13 22:03 | since #2, CI took 21–26 min; the qemu step alone was 18 min, 40 s of it one attention sweep. That test is skipped under qemu only | CI step 18 min → ~1 min | merged |
| #6 line editing | 10-13 22:03 | the first real session showed escape junk on arrow keys and keys typed during a reply echoed *into* the reply | 138 tests, PTY-driven checks, `stty -g` restored | merged |
| #9 drop viz | 10-13 22:35 | the viewer was unused; ratatui brought 35 of 61 crates | 61 → 27 crates, binary 3.29 → 3.07 MB | merged |

## Phase 4: GPU G0–G3 (2026-10-13 → 10-14)

### #7 GPU foundations (G0 + G1) — merged 10-13 22:03
- **What:** libcuda and NVRTC loaded with `dlopen` (no crates, no toolkit at build time), kernels compiled at run
  time and cached, dequant kernels bit-identical to the CPU, and `trait Lm` as the backend seam.
- **Verified:** 122 tests; dequant bit-exact on every tensor of 0.5B and 1.5B plus all 65,536 f16 values; all 41
  FFI signatures checked against the CUDA 13.4 headers by the critic. Cost: `axon info` +200 ms.

### #8 GPU decode (G2) — merged 10-13 22:11
- **What:** weights in VRAM in GGUF block format, activations quantized exactly like the CPU, dp4a GEMVs,
  flash-decoding attention, one CUDA graph per token, and PDL only where the loaded code supports it.
- **Verified:** GEMVs within 1e-6 relative of the CPU. GPU vs CPU top-1 97.0 % on 0.5B, which is the
  CPU-vs-CPU level. Deterministic across graph and PDL on/off. 172 tests.
- **Results (bella, vs ollama, hot GPU):** decode 530 / 264 / 132 / 52.1 vs 520 / 249 / 122 / 51.1 (0.5B–7B).
  Prompt eval was 10–18× behind (701 vs 8573 on 0.5B).
- **Missed:** the target was 7B ≥ 100 tok/s. It reached that only in short bursts on a cool GPU (101–106); the
  laptop hits its power cap within seconds.

### #10 Batched prefill and verify on tensor cores (G3) — merged 10-14 05:49
- **Why:** prompt eval was the G2 gap.
- **What:** int8 `mma.sync` GEMMs that reproduce decode's per-lane float order exactly, so batched == decode bit
  for bit at any batch and position.
- **Results (bella, noisy, ≥ 73 °C):**
  - pp560 0.5B 743 → 11069, 7B 65 → 1497. That is still 40–50 % of ollama and llama.cpp (7B 3158 / 3313).
  - Decode ahead on all four models (7B 89.5 vs 80.6 / 77.9).
  - `--draft 7` on 7B: code rewrite 67 → 275.
- **Diagnosis:** GEMMs at ~20 % of int8 peak, because the exact decode-order epilogue is ALU-bound. A
  llama.cpp-style inexact epilogue was only 10–19 % faster, so it was rejected and the contract was redesigned
  instead (#12).

## Phase 5: offload and hybrid (2026-10-14)

Haseeb asked for a model-bigger-than-VRAM path "without AirLLM" and "with AirLLM". A Fable review
(forum, 2026-10-13 23:3x) found that streaming the weights for prefill is what llama.cpp already does
(CUDA op offload for batch ≥ 32), and that AirLLM's distinctive move, streaming for decode, loses. The plan
was revised to `specs/g4-offload.md`.

### #13 CPU/GPU layer split (`--gpu-layers`) — merged 10-14 08:06
- **What:** first layers on the CPU; the last layers, norm and head on the GPU; auto placement by binary search on
  the real VRAM need.
- **Verified:** all-CPU and all-GPU splits equal each backend bit for bit. Mixed splits are graded within the
  CPU's own spread. The critic found that the foreman's rebase had broken the build (fixed in 7c85dad).
- **Results (bella, shared machine):** 32B at 28/64 layers decodes at 4.27 vs llama.cpp 3.98 (same `-ngl`). 14B at
  44/48 decodes at 30.1 vs 19.0. Prefill trailed llama.cpp's default op-offload mode (32B pp560 47.7 vs 208),
  which led to #16.

### #16 Stream the CPU layers' weights to the GPU for prefill — merged 10-14 16:38
- **What:** for batches ≥ 32, each CPU layer's repacked weights are pinned in place, streamed through a 1–2 layer
  VRAM ring, un-permuted on the GPU and run on the batched kernels. Decode and verify keep the plain split.
- **Verified:** after a streamed prefill, both halves' KV equal the all-GPU model's bits. Verify never streams.
  Chat gets a graded test (98.5 % top-1).
- **Results (bella, 32B Q4_K_M, 28/64 layers):** pp560 ~390 vs plain split 56–60 vs llama.cpp `-ngl 28` 211–212;
  pp2100 ~407 vs 264–286. Decode 4.8–5.8 vs 3.5–3.7.
- **Rejected inside it:** `AXON_STREAM_DECODE=1` (stream every token), 2.07 tok/s on 32B. Left as an env-only
  experiment.

## Phase 6: 8-bit KV cache (2026-10-14)

### #11 `--kv q8` on the GPU — merged 10-14 16:38
- **Why:** memory. The 14B didn't fit at 8k chat context in f16.
- **What:** int8 KV with one f32 scale per position and kv head, and a fixed key shift after RoPE. Without the
  shift, Qwen's |k| ≈ 200 bias channels set every row's scale and q8 was 5–15× noise. Batched q8 == decode q8 bit for bit.
- **Results (bella):** 14B runs fully on the GPU at ~7.9k context (prefill 372, decode 42.1). Speed vs f16: decode
  unchanged, prefill 4–7 % slower. Accuracy: 7B q8-vs-f16 KL at the CPU-vs-CPU level; 0.5B/1.5B a few times
  higher.
- **Status:** merged. A memory feature, not a speed one. Round 9 later measured that it gains nothing at 32k,
  because decode attention was latency-bound (see [results](results.md#long-context-gpu-bella)).

## Phase 7: GPU kernel rounds (GEMM, decode)

### r1 → #12 gemm2: canonical block order, 64-row tensor-core GEMMs — merged 10-14 06:12
- **Pick:** Fable r1 pick, "GEMM tile/contract redesign first" (forum). **Design:** `r1-gemm2.md`.
- **What:** a new canonical GPU order (exact integer sum per block, scale once, one fma chain over blocks) for
  decode, gemvn, gemvm and the new 64×128 / 64×64 `mma.sync` GEMMs. GPU logits change once; CPU unchanged.
- **Verified:** `gpu_products_match_reference_bits` across types, widths, odd rows, 8 batch sizes and every
  epilogue. GPU-vs-CPU closeness improved (2000-token KL 4.7e-5 → 2.1e-5).
- **Results (bella, throttled):**
  - GEMM nb512: 18.8 → 40.7 TMAC/s (q/wo).
  - End to end pp560 ×1.66–2.49 (7B 1010 → 2510).
  - Decode A/B, 8 rounds: 7B ×1.045, 0.5B ×1.099. An earlier version had cost 7B decode ~8 %; it was fixed before merge.

### r2 → #15 shared-score batched attention, 1024-token passes — merged 10-14 16:38
- **Why:** long prompts spent about half of prefill in attention, recomputing q·k for every split.
- **Results (bella, 85–88 °C):** pp560 vs llama.cpp 116 / 105 / 95 / 96 % (0.5B–7B). pp2100 56 / 66 / 79 / 86 %.
  Decode flat. GPU md5s unchanged.

### r4 → in #22: prefill epilogues, int8 Q6_K copy, fused gate/up — merged 10-15 12:20
- **Design:** `r4-prefill-big.md`. The 7B GEMMs were at MMQ parity and power-bound (sustained 0.6–0.68× burst).
  The waste was a trailing 1-token forward (6 %), the fp32 swiglu round trip, and a 52-token tail pass.
- **Results (bella):** the whole #22 stack vs main 53bbd12, 3 rounds: 7B pp560 +8.9 %, q3-4B pp560 +8.9 %, tg flat. #22's stages besides this one don't touch prefill GEMMs, but there is no p7-only number in the PR.
- **Rejected inside it:** stage 3 `gemm3` (TMA): −2 % / −55 %.
- **Also rejected:** the r4 decode track (persistent GEMVs + fused epilogues) cost 7B decode −14.5 % in the
  integration chain and was dropped from #22. See [rejected](experiments-rejected.md#gpu-decode).

### r5 → #24 prefill: swiglu outside the up GEMM, batched epilogue loads — merged 10-15 17:39
- **Design:** `r5-long-prefill.md`. Attention had stopped being the long-prompt problem (7–20 % of a pass).
- **Results (bella, 5 rounds, ≤ 60 °C starts):** pp2100 0.5B +21.8 / +23.0 %, 1.5B Q5_K_M +19.4 %, q3-0.6B +13 %,
  7B +2.3–2.7 %. tg flat. md5 17/17. P3 dropped (within spread), P4 not attempted.

### r5 decode → #30 rmsnorm_quant_v2 and gate_up4 — merged 10-16 04:11
- **Design:** `r5-decode.md`. 7B decode GEMVs were already at 93 % of the 648 GB/s ceiling and the rest was the
  140 W cap. Small launches cost ~6 % of the 7B step.
- **Results (bella, 5 rounds, builder's runs vs 0851798):** tg128 0.5B 781 → 822 (+5.2 %), 1.5B +4.0 %, q3-0.6B
  +3.0 %, 7B +2.4 % (paired +0.2 %). Bits identical (8-model test, 42/42 md5 on 4090 pods).
- **Caveat:** 1.5B decode on bella is bistable across processes (343–413 tok/s). P3a dropped (no gain).

### r8 GEMM pipeline (mbarrier stages, producer warp) — rejected 10-16 08:55
- **Design:** `r8-unpack.md`, which found the GEMM not unpack-bound but losing ~30 % to barrier lockstep and
  cp.async issue in an L2-resident ablation. Predicted +20–25 % on bella.
- **Result (bella):** gate/q shapes −3…−28 %, 3B e2e pp −8.5…−11 %. NO-GO by its own rule. Branch
  `feat/r8-gemm-pipeline`, opt-in `AXON_GEMM2M`, no PR.

## Phase 8: attention

### r3 → #19 tensor-core flash attention, new canonical order — merged 10-14 22:41
- **Why (MEASURED, stage 0):** at pp2100 attention was 63.6 % of a 0.5B prefill pass and 37.5 % of 7B's, on CUDA
  cores at ~5 TFLOP/s. **Design:** `r3-long-prefill.md`, option (a).
- **What:** `mma.sync` f16 attention for decode *and* prefill in one new canonical order, so batched == decode still
  holds. The new prefill kernel runs at 43–47 TFLOP/s.
- **Verified:** bit tests over f16/q8/q8g32, NaN-poisoned cache past ctx, >1024 positions. One graded test,
  `assert_graded`, was loosened after one chaotic position reached KL 1.2. The measurements are in the test
  comment, and an injected mask bug still fails it.
- **Results (bella vs llama.cpp, ≤ 80 °C starts):** pp2100 1.10 / 1.09 / 1.05 / 1.13× (0.5B–7B), was 0.57–0.83×.
  pp560 1.53–1.01×, tg128 1.06–1.21×. The next scoreboard (881d7b6) still had ties with ollama at 7B pp560, 8B
  pp560 and 4B pp2100.

### r9 #2 lcpre: ping-pong prefill attention — rejected 10-16 (afternoon)
- RTX 5090: hd-128 shapes 2.4–3.0 % *slower* per CTA; avante +2–5 %. NO-GO by the rule (< 2 % on the 5090).
  Nothing committed. See [rejected](experiments-rejected.md#gpu-attention).

### r9 #1 lcdec: long-context decode attention — in flight (no PR at a591e17)
- **Why:** r9-pick measured that axon loses decode at 32k: 7B 72.4 vs llama.cpp 76.7, 0.5B 413 vs 464.
- **Status from the builder's NOTES (not on main):** 4-warp `attn_blk` plus a two-phase `attn_cmb`. Pod3 RTX 5090
  paired ratios at 32k: 7B f16 +12.0 %, 7B q8 +23.4 %, 0.5B +30 %, q3-8B +14.6 %; md5 40/40 equal. A −1.5 %
  regression at tg128 pos 8 on 0.5B was being fixed. **Not merged; numbers unconfirmed on bella.**

## Phase 9: sampling and speculation

### #17 sampling — merged 10-14 18:16
- **Why:** axon was greedy-only, and greedy chat replies loop.
- **What:** temperature, top-k, top-p, min-p, repeat penalty and seed, in llama.cpp's order with ollama's defaults
  for chat (`gen` stays greedy). A fused CUDA penalty + radix top-k kernel. Drafting stays lossless under
  sampling: one random number per decided token, so the same seed gives the same tokens with drafting on or off.
- **Verified:** drafting == no drafting over 300 seeds × K, a chi-square over 30k draws, and GPU kernel == CPU
  filter on 1,848 adversarial rows. The critic's blocker (`--repeat-last-n 0` + draft panicked) was fixed.
- **Cost:** CUDA 0.5B −1.4 %, 7B −0.4 %, CPU noise.

### #27 draft-model speculative decoding — merged 10-16 02:33
- **Pick:** r6-pick #1. Single-stream 7B/8B could no longer be widened by kernel work, only by fewer weight passes
  per token. Estimated +30–55 %.
- **Results (bella, 5 rounds, ≤ 60 °C starts, tg128 medians, K=3):**

  | model (drafter) | main | draft model |
  |---|---|---|
  | Qwen2.5-7B (0.5B) | 104.5 | 142.8 (+36.6 %) |
  | Qwen3-8B (0.6B) | 103.7 | 149.3 (+44.0 %) |
  | Llama-3.1-8B (3.2-1B) | 109.3 | 178.4 (+63.3 %) |

  Sampled (temp 0.8): +0.8 % at K=3, −20 % / −29 % at K=5/7. That missed its +10 % gate, so the gain is greedy only.
- **Verified:** md5 with a drafter == without, 135/135 (K 3/5/7) and 45/45. Suites 343/0 on a 4090.

Prompt-lookup drafting (`--draft K`) landed in #1 and stayed off by default: low-repetition text loses
(story −4 % on 0.5B).

## Phase 10: models

| PR | merged | what | why / verified | notes |
|---|---|---|---|---|
| #18 Qwen3 dense | 10-14 18:16 | q/k RMSNorm fused into the attention prologue; `/no_think` template; also a `gguf::release` underflow fix | an independent f64 reference forward catches injected norm bugs; matches llama.cpp QWEN3 + HF template; qwen2 md5 unchanged. v5 chosen over v3 (v3's norm handshake cost 2–3 %) | 4B decode 144.9 vs llama.cpp 123.7 / ollama 124.7 (pre-gemm2 base) |
| #22 (q5k) | 10-15 12:20 | Q5_K, Q4_0, Q5_1, Q4_1 on CPU and CUDA | dequant bit-exact vs ggml; existing decode SASS unchanged (separate module) | CPU paths for these types are per-row (slow), unlike r8 types |
| #22 (llama) | 10-15 12:20 | Llama 3.x: un-permuted q/k at load, rope_freqs, llama-bpe tokenizer | tokenization == llama.cpp; CPU 1B pp512 311 vs ollama 275 | |
| #23 serve Llama framing | 10-15 13:05 | `/v1/completions` dropped BOS and ignored `<\|eot_id\|>` on Llama; chat endpoint only knew ChatML | serve == gen/chat on Qwen 0.5B and Llama 1B, CPU + CUDA | |
| #32 SentencePiece + Mistral | 10-16 10:38 | a port of llama.cpp's SPM; Mistral `[INST]` framing (a deliberate hybrid of two references) | 28/28 CPU md5; Mistral tests with the model present | Mistral + tools → 400 |
| #35 Gemma 3 1B/4B | 10-16 15:11 | head dim 256, sliding window, GeGLU, 262k vocab, chat format | tokenizer == llama.cpp on 42 texts; misses vs llama.cpp are near-ties in an f64 reference; md5 vs main for other models: CPU 21/21, CUDA 27/27 (PR comment, final head); suites 409/0 rel + dbg on avante | 4090: pp2100 trails llama.cpp (1B 32748 vs 40504, 4B 8782 vs 17920): no tensor-core attention at hd 256. q8, split, streaming and 27B refused |

## Phase 11: serving

| PR | merged | what | why | verified / numbers |
|---|---|---|---|---|
| #22 (serve S1) | 10-15 12:20 | `axon serve`, OpenAI-compatible, one request at a time; KV carried over for shared prefixes | `specs/r4-serve.md`: kernels already batch-invariant, so S1 first, S2 later | byte-identical to gen on CPU and CUDA; 0.5B GPU stream 706 vs llama-server 560 / ollama 484 (builder, bella) |
| #23 | 10-15 13:05 | Llama framing in serve | see Phase 10 | |
| #26 tool calling | 10-16 04:11 | `tools`, `tool_choice`, `tool_calls` (streamed and whole) for Qwen3, Qwen2.5 Hermes, Llama 3.1 | agent harnesses got a 400 on `tools` | rendering == transformers' rendering of the GGUF templates; requests without tools byte-identical to main; crush, pi, Synaps answered a read-file loop on Qwen3-8B. opencode + Qwen3-8B didn't finish (cause unverified) |
| #38 S2-lite continuous batching | 10-16 10:38 | decode rows of many requests in one step; per-row positions, cache rows and sampling; FIFO, whole-prompt prefill as its own step | r6-pick #2: estimated 7B at 8 clients ~5.5× | identity: batched == solo 6/6, md5 22/22. bella 7B C=16 355 vs 80 (4.4×), 0.5B 2978 vs 620 (4.5×). 4090 7B C=8 740, C=16 1017. Missed the 7B C=8 5× and 0.5B 3.5k targets |
| #42 serve default ctx | 10-16 16:22 | default context = trained length where it fits; slot planning; deferred Q6_K copy | r9 #3a: agent prompts > 8k got a 400 | release/debug 415/0 (4090), md5 0/15 differ CUDA and CPU, ~20k-token requests answered. **Behaviour change:** bella 7B now serves one request at a time at 32k by default (was 8 × 8k). The 12 GB fit check on bella was pending at merge |
| #45 chunked prefill | open | prompts fed in ≤ 512-token chunks between decode steps | r9 #3b: a 24k admit froze every other stream for 2.45 s (7B, 4090) | 4090: worst inter-token gap 2455 → 84 ms (7B), 354 → 21 ms (0.5B); admitted request's TTFT +21 % / +83 %. Single-user unchanged |

## Phase 12: startup

| PR | merged | what | numbers (bella, TTFT process start → first token) |
|---|---|---|---|
| #21 startup | 10-15 07:57 | CPU maps instead of copying when every matrix is repacked; CUDA init at t=0 on threads; read-ahead; prefaulted upload; `_exit` after gen. Design: `r4-startup.md` | CPU 7B warm 5.44 s → 1.74–2.30 s (ollama cold load 3.73); q3-4B 1.47 → 0.89 s; GPU 7B warm 732 → 624 ms. md5 identical. **Behaviour change:** overwriting a mapped GGUF in place can SIGBUS (`AXON_MMAP=0` copies) |
| #41 startup round 2 | 10-16 11:50 | pinned upload ring, used only when ≥ 75 % of the GGUF is in page cache; faster tokenizer build and GGUF string parse | warm 7B 614 → 533 ms, q3-8B 637 → 545; cold unchanged (a forced ring was 35–40 ms *slower* cold, hence the residency rule). Layer-wise upload/prefill overlap not done |

## Phase 13: CPU, later rounds

| PR | merged | what | why | numbers |
|---|---|---|---|---|
| #20 CPU prefill | 10-15 08:36 | Q8_0/Q5_0 GEMM per-chunk fix (59 → 157–171 GMAC/s per group call), 17 → 10 pool joins per layer, E-core tail split, batch 128 | CPU board 881d7b6 gaps: P/E prefill imbalance, Q8_0 prefill, transposes | bella 16 threads vs 881d7b6: pp512 0.5B ×1.57, q3-0.6B ×1.82, 1.5B ×1.15, 7B ×1.07; tg ~flat. CPU suites 275/0 |
| #32 decode pool cap | 10-16 10:38 | decode gets its own pool level, capped at the P-cores on non-SMT hybrids | memory-bound matvecs saturate DRAM before all cores; E-core chunks add a tail | bella tg 1.017–1.098× (builder's runs). On avante (SMT hybrid) any cap lost 7–10 %, so SMT parts get no cap. Timing A/B of the exact merged tree pending |

## Phase 14: multi-GPU and Hopper

### #33 Hopper sm_90a wgmma + TMA prefill GEMMs — merged 10-16 05:49
- **Why:** on the RunPod board the H100's prefill trailed even the 4090; axon didn't use wgmma. Design: `r7-hopper.md`.
- **E0 on an H100 PCIe ($0.77):** wgmma int8 peak ~620–707 TMAC/s vs mma.sync 254. But the best plain GEMM reached
  only 84 TMAC/s, so the spec's ≥ 120 gate was **not met**. The fused gate+up kernel was +18 %. 7B pass
  9263 → 10376 (+12.0 %). H100 md5 10/10 identical.
- **Shipped:** fused gate+up on wgmma by default on sm_90a; plain products only where measured faster. Consumer
  cubins byte-identical. The default routing itself was not run on an H100.

### #34 pipeline parallel over several GPUs (`--gpus 0,1`) — merged 10-16 07:30
- **Design:** `r7-multigpu.md`, pipeline first.
- **Results (2×4090, no P2P):** 7B prefill 1.46×, 32B 1.55×, decode 1.00×. md5 168/168 equal to one GPU.
- **Rejected:** tensor parallel. The all-reduce cost 2.24 ms/token of comms against a 2 ms gate. P2P is opt-in
  and untested (no P2P box). vLLM TP2 on the same pod beat axon single-stream by ~1.5× (65–70 vs 43, 32B AWQ).

## Phase 15: hardening and test infrastructure

| PR | merged | what / why |
|---|---|---|
| #14 | 10-14 06:21 | pool adaptation test flaked under qemu on 2 vCPUs (workers park during the slow unwind) → test-only fix |
| #31 | 10-16 04:11 | `gpu_pdl_gating` picked compute_89 PTX on an sm_80 A100 → newest arch ≤ the device. Main's suite on A100: 342/1, only this test |
| #36 | 10-16 08:18 | two sm_89 accuracy tests failed after #28 rewrote the README: the tests tokenized the live README. Now a frozen fixture (5f96039's text) |
| #37 hardening | 10-16 08:51 | roadmap item 2 (#29). GGUF fuzzer (~1800 mutants/model) and serve fuzzer (183 checks) found 9 bugs: an O(n²) BPE (1 MB pre-token: no answer in 120 s → 0.3 s), slow-loris 503s, disconnects not cancelling, overlapping tensors panic, metadata allocation abort, RST on early errors, TLS bytes held 30 s, non-UTF-8 argv panic, `gen -n` overflow. 1 h CPU soak: 5121 requests, 0 unexpected |
| #39 old GPUs | 10-16 07:16 | with CUDA 13 + 12 NVRTCs installed, axon picked 13, which can't target sm_61 → fall back to an older NVRTC. The sm_61 llama test failure was the CPU reference misjudging a real near-tie (f64 check) → near-tie rule uses either side's margin |
| #40, #43 | 10-16 10:39, 14:36 | GPU tests serialized on cards < 8 GiB, then < 16 GiB: concurrent tests ran out of VRAM (false failures on avante, then bella) |
| #46 | open | `gpu_too_big_is_an_error` retries its free-VRAM checks when other processes share the GPU |

## Phase 16: docs and tooling

- **Docs PRs:** #25 (README + BENCH for the GPU backend, 10-15), #28 (START-HERE, GPU benchmarks, multi-GPU and
  Hopper designs, roadmap, 10-16), #29 (hardening as priority 2), #44 (roadmap Scale section from `r10-scope.md`).
- **Tooling (outside the repo, no PRs):** `bench.py`/`sweep.py` (in repo, #1); `scoreboard.py` and
  `cpu_scoreboard.py`; `axon-q` (10-15) → `axon-run` (10-16); `axon-pods` with a budget guard (10-15); the
  heartbeat and watchdogs. Described in [methodology.md](methodology.md#orchestration-tools-outside-the-repo).
- **Scoreboards:** GPU bella 0851798 (10-15 13:06); RunPod 4090/5090/H100 at f8d8a86 (10-15 23:53); avante GPU and
  jade CPU at f8d8a86 (10-16 00:23, 01:11); CPU bella 881d7b6 (10-15) and f8d8a86 (10-15 19:02); long context
  (r9, 10-16). Current standing: [results.md](results.md).

## r10 tracks (launched 2026-10-16 ~16:00, not on main at a591e17)

From the research reports ([research/](research/README.md)): `spec32` (speculation on the 32B split), `fingerprint`
(identity tuple, `system_fingerprint`, determinism doc), `kvsave` (exact KV save/restore, `chat --session`) and
`yarn` (`--rope-scaling yarn`). Plus, still in flight: #45 chunked prefill, lcdec, and axon-run pod packing.
