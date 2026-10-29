# ADR: what bit-identity costs, and making it a mode

**Status: accepted** (2026-10-29, Haseeb: "lets try fast"). §7 records the answers to its four questions. Built: M1,
M2, M3, M5 and M8 (the mode scaffolding, no kernels; PR #PRNUM, [log](../log/2026-10-29-numerics-modes.md), "As built"
under §4.7). Deferred: M4 (CPU fast prefill) and M6 (exact split-K; its 4090 attempt, #148, was killed on its kill rule:
[experiments-rejected.md](../experiments-rejected.md)). M7 waits on a measured gap (§4.6). Written against `origin/main`
`05fb70c` (2026-10-19). Answers the outside review's questions "Is bit-identity a product or a dev tool? If it's a dev tool, why
is it a global constraint instead of a mode?" and "what is the plan for tensor parallel with fixed-order reductions,
and have you costed it against NCCL?" (`fable-bigpicture-20261019`, §5 risk 1, §7 and §8 recommendation 3), and its
mission-drift question, whose answer is the new "Mission and positioning" section of [roadmap.md](../roadmap.md).

Labels: **MEASURED** (a cited log, PR or spec table), **CODE** (file:line on `05fb70c`), **ESTIMATED** (arithmetic
shown), **OPINION** (mine). Numbers from the overnight workspace specs (`r16.md`, `r17-gemm-ablation.md`) are MEASURED
there and cited as such; they are not in this repo.

---

## 0. Decision in one paragraph

Keep the contract as the **default for everything a person runs by hand** (`gen`, `chat`, `serve` on one card) and make
it a **named mode, `--numerics exact`**, with a sibling **`--numerics fast`** that may use order-changing kernels. Fast
mode is a second identity (its own `NUMERICS_VERSION` line and fingerprint), gated by tolerance against exact mode
instead of by md5, and it is what a datacenter deployment turns on. The measured cost of the contract on today's
kernels is small (§2: the prefill GEMM's binding resource is the SM's L1/MIO path, not the reduction order; wave tails are
already taken; decode is at the DRAM wall on consumer cards), so **nothing in the fast kernel set exists yet and nothing
should be built for it until a measured gap names it** (§4.6). What the mode buys today is a place to put split-K
GEMVs, stream-K, NCCL-style all-reduce and FP8 tensor-core accumulation when those gaps are measured, without
re-litigating the contract each time, and a stated answer to "why not vLLM" that doesn't depend on the contract.

---

## 1. The contract today

### 1.1 The identities Axon guarantees, and where each is held in place

"Same bits" below means the same logits (and so the same tokens, KV bytes and sampled output for a seed) for one
identity tuple (`src/identity.rs`; [determinism.md](../determinism.md)). Every test is CODE.

| # | identity | holds for | test (file:line) |
|---|---|---|---|
| A | **decode == batched prefill.** A token fed alone (`forward`) and the same token inside a `feed_batch` of any size, at any batch position, write the same KV rows and logits | CPU, CUDA, hybrid, MoE | `feed_batch_matches_feed{,_qwen3,_llama,_fixture}` `src/model.rs:4640-4682` (body `feed_batch_matches_feed_on` `:5006`); MoE `src/moe/tests.rs:163`; GPU `gpu_prefill_batchings_bit_identical_{qwen3,llama,qwen3_4b,0_5b,1_5b,q2k_q3k,q8_0_5b,q8_1_5b}` `src/cuda/qwen2/tests.rs:1578,1752,1835,3832,3843,4421,5668,5680` |
| B | **decode == speculative verify.** `verify_argmax` / `verify_cands` row *b* equals `forward` on token *b*, also when the cache holds a rejected earlier draft | all backends | `verify_batch_matches_forward` `src/cli.rs:1832`; `verify_argmax_matches_forward_argmax` `:1866`; GPU `gpu_verify_matches_forward_argmax_qwen3{,_4b}` `src/cuda/qwen2/tests.rs:1589,1846`. Contract text: `src/lm.rs:48-50` |
| C | **every product kernel == one canonical order.** Decode GEMV, `gemvn`, `gemvm`, the tensor-core GEMMs and the fused gate/up all equal `gemv_ref` (the order written out one thread per output) for every weight type, odd widths, every epilogue | CUDA | `gpu_products_match_reference_bits` `src/cuda/qwen2/tests.rs:2957`; `gpu_batched_products_match_gemv_bits` `:2805`; `gpu_swiglu_epilogue_and_gate_up_match_reference` `:3064`. Order defined at `src/cuda/decode.cu:4-15` ("Row dot products"), `src/cuda/prefill.cu:2-16`, `gemv_ref` `src/cuda/decode.cu:886`, `gv_chain` `:923`; attention fold `att_fold_w/att_fold_o` `:1603-1615` |
| D | **cache reuse == fresh prefill** (chat follow-ups, cut replies, stopped replies, with and without drafting) | CPU, CUDA, hybrid | `turns_match_fresh_prefill{,_cuda,_hybrid}` `src/chat.rs:1519,1655,2011`; `chat_with_draft_model_unchanged` `src/draft.rs:927` |
| E | **drafting == no drafting**, greedy (prompt lookup and a draft model) | CPU, CUDA, hybrid | `drafting_matches_greedy{,_cuda,_hybrid}` `src/cli.rs:1902-1926`; `draft_model_output_unchanged_{cpu,cuda,cuda_big}` `src/draft.rs:611,621,642` |
| F | **sampled drafting == plain sampling, token for token, for a seed** (not just in distribution; §4.4 explains why that is stronger than the literature's claim) | sampler (model-independent), plus the real CPU model | `drafting_matches_plain_sampling` `src/sample.rs:1180` (300 seeds × K ∈ {1,3,7} × oracle/wrong/random/lookup drafts); `drafting_distribution_chi_square` `:1218`; the real-model case `:1261` |
| G | **batched with others == alone**: any slot, any position, joining or leaving mid-generation, chunked prefill interleaved with decode, handed over from a solo cache | CUDA pool and `serve` | `pool_steps_match_solo_in_any_order` `src/cuda/qwen2/slots.rs:719`; `serve_batched_matches_solo` `src/serve/tests.rs:1659` |
| H | **same request → same fingerprint → same text** across restarts, streaming, prefix-cached, under load; fingerprint carries `NUMERICS_VERSION` | `serve` | `serve_identity` `src/serve/tests.rs:2417`; `serve_identity_cpu` `:2462`; numerics field asserted at `:2368`; `src/identity.rs:65,90` |
| I | **pipeline parallel == one GPU** (every layer whole on one device, f32 hops) | multi-GPU | by construction ([multi-gpu.md](multi-gpu.md) §3.1); MEASURED 2×4090 decode 1.00× (roadmap item 4). Testable on one device by opening it twice |
| J | **paged == unpaged** (page tables change addresses, not order) | CUDA | `gen` md5 identical, #64/#68 (MEASURED, [log](../log/2026-10-17-paged-kv-s01.md)) |
| K | **CPU: SIMD == scalar, any thread count, AVX2 == AVX-VNNI**; one hardware bit in the tuple (FMA), one opt-out (`AXON_REPACK=0`) | CPU | md5 gate on the CPU, 20/20 across an AVX2 and an AVX-VNNI host ([determinism.md](../determinism.md) table); CI's qemu Nehalem step ([methodology.md](../methodology.md) gates) |
| L | **the md5 regression net**: main vs candidate `gen` outputs, greedy, same host, 11–42 cells across models, CPU + CUDA, short and ~610-token prompts | every PR | [methodology.md](../methodology.md) "Gates"; recent example `docs/log/2026-10-18-c-curve-2.md:63` (20/20) |

### 1.2 What is explicitly *not* an identity, and the tolerance machinery that already exists for it

| pair | relation | gate (CODE) |
|---|---|---|
| CPU vs GPU | graded, not equal: top-1 within 2 % of the CPU's self-agreement and ≥ 94 % (≥ floor on natural text), every top-1 flip below the reference's own spread, mean KL ≤ 2× the CPU's self-KL, ≤ 1 position with KL > 0.5, none > 2 | `assert_graded` / `assert_graded_floor` `src/cuda/qwen2/tests.rs:255-310`; absolute fallback when the CPU's two kernel sets agree too well (`:370-386`): mean KL ≤ 5e-3, max 0.2, flips < 0.5); used by `gpu_logits_match_cpu_*` `:338,354,389,1556,1730,2150,2191,2244` and `gpu_continuations_match_cpu_*` |
| streamed CPU/GPU split, prefill ≥ 32 tokens vs decode | different numerics, same identity string says so (`prefill streams >=32`) | `src/hybrid.rs:98-100,346,499`; [determinism.md](../determinism.md) "What is NOT guaranteed" |
| f16 vs q8 KV | different identity | `kv` field, `src/identity.rs:100` |
| sm_61 vs sm_80+; sm_89 vs sm_120 | different / unaudited; `gpu` field keys the exact `sm_XY` | [determinism.md](../determinism.md) |
| across `NUMERICS_VERSION` | different identity by rule | `src/identity.rs:55-65` |

**Why this matters for the ADR (OPINION):** the fast-mode gate in §4.2 is *not* new machinery. It is `assert_graded`,
which already decides whether a GPU kernel is "the same model" as the CPU, pointed at exact mode instead of the CPU. The
project has been running a tolerance-gated second numerics (the GPU vs the CPU) since G3; a fast mode is a third.

---

## 2. What it costs

The contract forbids any kernel whose float-operation order depends on anything but the model's shapes and the token's
position (`src/cuda/decode.cu:13-15`). Six techniques are on the forbidden list. For each: what it would buy on Axon's
kernels, with the number and its label. Where the honest answer is "little", it says so.

### 2.1 Split-K / stream-K in the prefill GEMM

- **What binds the GEMM today (MEASURED, r17 ablation on sm_89, `r17-gemm-ablation.md` §0, §4).** Replacing every mma by
  LOP3s: −4 / 0 / −5 % (the tensor pipe is idle capacity). Removing the cp.async fill: +34 / +54 / +29 % (gate / down /
  gu). The activation fill alone: +19 / +32 / +16 %. Per-stage ALU: +25 % when all of it goes. The verdict: the loop is
  **MIO/L1-path and latency bound at 2 warps per scheduler, not throughput bound on any unit**. The full kernel runs at
  33 % of `mma_peak` on avante and 27–33 % on the 4090 (r16 §0: gate_up 83 TMAC/s, down 102, peak 307).
- **What split-K changes.** Split-K parallelises the K loop across CTAs to fill the machine when there are too few
  output tiles. 7B gate_up at nb 1024 has (2×18944/64) × (1024/64) = 9,472 CTAs for 128 SMs: 74 waves. There is no
  occupancy problem to solve; split-K would only add a fixup pass. **ESTIMATED gain: ≤ 0 % on the measured shapes.** Where
  split-K does matter is narrow products (k/v projections, 512 rows) at *small* nb, which is the decode case, §2.2.
- **What stream-K changes.** Stream-K removes wave quantisation (the partial last wave). #92 (`AXON_WAVE_SPLIT`, r15 pick
  1) sizes the prompt passes to whole waves and took **+2.6 / +4.1 % pp2100 on the 4090** (MEASURED, PR #92 body, cited
  in r16 §1). What remains for stream-K is the residual tail inside each pass plus the per-CTA fill/tail that r16 pick 2
  (persistent CTAs, order-preserving) targets at **+2–5 % e2e ESTIMATED**. Stream-K's *additional* gain over an
  order-preserving persistent kernel: **ESTIMATED ≤ 2 %**, and llama.cpp's MMQ, which uses stream-K, measured at parity
  with Axon's `gemm2` (29.7 vs 29.3 TMAC/s, bella, r4-prefill-big §1.4, MEASURED).
- **Cost of the contract here: ESTIMATED ≤ 2 % of prefill**, and the two picks that keep the order (128-row tile,
  aligned block sums: r17 §5, ESTIMATED +8–15 % on the kernel) are worth more than what the forbidden tools would add.
  r16 §1's table says the same: "stream-K / split-K: buys only wave tails, which #92 already took. Not worth it."

### 2.2 Split-K in the decode GEMV (batch 1 and small nb)

- **Consumer cards (MEASURED):** 7B decode GEMVs at 602 of 648 GB/s on bella (93 %, r5-decode §0); 4090 186 tok/s ×
  4.68 GB ≈ 870 GB/s = 86 % of 1008 (ESTIMATED from the board). Nothing to gain.
- **H100 SXM (MEASURED 311 tok/s; ESTIMATED 1.46 TB/s = 43 % of 3.35 TB/s).** This is the one batch-1 cell where a
  split-K GEMV could matter: a 512-row k/v projection over 132 SMs is latency-bound, and splitting K is the standard fix.
  But llama.cpp, which has no contract, sits at 34 % on the same card (review §2), and `hopper.md` §1 finds the H100 pass
  **issue-bound**, not order-bound. So the 43 % → ~85 % gap (≈ 2× tok/s) is an upper bound on *everything* wrong on
  Hopper at batch 1, of which the reduction order is one uncosted part. **ESTIMATED share attributable to the
  contract: unknown, bounded above by ~15 % of the step** (the k/v/q+wo GEMVs are ~17 % of the 7B step on the 4090 at
  nb 1, `docs/log/2026-10-18-c-curve.md` per-kernel table: 0.93 + 0.36 of 7.53 ms; the split-K case is only the
  narrow ones).
- **And it is recoverable without a fast mode.** `research-determinism.md` P3: a two-level chain with a fixed chunk
  size in K (chunk = 8 blocks), summed in index order, is a function of K only, so decode == prefill == verify and batch
  invariance all hold; decode may then run the chunks on different CTAs and combine in chunk order. Cost: one
  deliberate NUMERICS bump, every product kernel changed together, `gpu_products_match_reference_bits` re-pinned. The
  same idea is multi-gpu.md §3.3's shard-invariant tree. **The contract forbids *arrival-order* and *SM-count-dependent*
  splits, not fixed splits.** That distinction is the most important sentence in this section.

### 2.3 llama.cpp's per-32 float epilogue

- MEASURED: on G3's tiles the inexact epilogue was only 10–19 % faster, and the gap to llama.cpp turned out to be GEMM
  efficiency, fixed by redefining the canonical order in #12 (`docs/experiments-rejected.md:43`). r16 §1: against today's
  `gemm2` it is "a wash" (r4 §1.4); r17 §2: `NO_EPI` (the float epilogue removed entirely) is +2 / +4 / −8 %, inside the
  noise band. **Cost of the contract: ESTIMATED 0–4 % of the GEMM kernel, ≈ 0–3 % e2e prefill.** Not worth a second
  kernel set on its own, and (r16 §1) if adopted it would be a *new fixed order*, i.e. still exact-mode material, not
  fast-mode material.

### 2.4 Tensor parallel with NCCL all-reduce vs a fixed-order reduce

Axon has no TP; everything here is ESTIMATED from multi-gpu.md §3.2–3.3 and NCCL's documented behaviour.

- **Why NCCL breaks the contract (OPINION, from NCCL's documented behaviour).** NCCL picks algorithm (ring, tree,
  CollNet) and protocol (LL, LL128, Simple) by message size and topology. A decode all-reduce is `dim × 4 B` (14–32 KB);
  a prefill all-reduce of 1024 tokens is 1000× larger. Different sizes can take different algorithms, hence different
  per-element reduction orders, hence **decode ≠ prefill**, and in a ring the order in which a chunk's partials are
  summed depends on the chunk's owner rank, so **TP-N ≠ TP-1**. Pinning `NCCL_ALGO`/`NCCL_PROTO` makes it run-to-run
  stable in practice but is not an API guarantee, and the ring order still differs from the single-GPU chain.
- **What a fixed-order reduce costs (ESTIMATED).**
  - *Decode (latency-bound, 14–32 KB).* NCCL's own choice at this size is a tree or LL ring; a hand-written
    rank-ordered tree over P2P copies is the same number of hops. multi-gpu.md §3.2 prices 128 syncs per token at
    1.9 ms (P2P on) / 4.5 ms (off) of a 14.8–17.4 ms 32B step on 2×4090; the arithmetic is the same whichever
    engine's all-reduce, because the cost is the hop latency, not the add. **Contract cost at decode ≈ 0.**
  - *Prefill (bandwidth-bound).* A ring moves 2(N−1)/N × M bytes per GPU; a rank-ordered reduce-scatter + all-gather
    moves the same bytes but must land the N−1 partials before summing each owner chunk in rank order (extra buffer
    of (N−1) × M/N per GPU) or sum sequentially in rank order with pipelining (one extra hop of latency). On PCIe
    2×4090 the 1024-token all-reduce traffic is ~107 ms of a ~0.7 s pass (§3.2); **a 10–20 % penalty on the reduce is
    ESTIMATED 1.5–3 % of the prefill pass**. On NVLink the all-reduce share is smaller still.
  - *The shard-invariant chain* (§3.3): ~7 extra fadds per output per matrix, plus a one-time bits change on one GPU
    too. **ESTIMATED < 0.5 % of a GEMV.** It makes TP-1/2/4/8 bits equal, which NCCL can never offer.
- **Honest ceiling.** multi-gpu.md §4 already concedes "beating vLLM TP on NVLink H100/A100 for one stream" is not
  claimed, for reasons that have nothing to do with the order (mature kernels, ~5 µs all-reduce). **The contract's
  share of the TP gap is ESTIMATED single-digit percent; the TP gap itself is "no TP exists".** Review §7 Q9 asks "if
  it is 2×, the contract dies at the first multi-GPU node": it is not 2×; it is the add order of two or eight partial
  sums.

### 2.5 FP8 weights and activations: tensor-core accumulate orders

Axon has no FP8 path. This is the one technique where the contract is a *design* constraint rather than a few percent.

- **Why today's trick stops working.** Exactness today rests on int8 × int8 → int32 block sums being exact *whatever
  the hardware's internal order* (dp4a and `mma.sync` agree because integers associate), with one float chain over the
  blocks afterwards (`src/cuda/decode.cu:4-15`). An FP8 tensor-core mma accumulates in float inside the instruction, in
  an order and (on Hopper) a precision that is undocumented and architecture-specific (Hawkeye, arXiv:2603.20421, cited
  in determinism.md; DeepSeek V3's per-128-K promotion to CUDA cores exists because Hopper's FP8 accumulator is
  narrower than f32). A CUDA-core decode GEMV cannot reproduce that order.
- **The exact-mode option (OPINION, uncosted):** one instruction family for every nb: decode runs the same FP8 `mma`
  with M padded (nb 1 → 16 or 64 rows of padding), same K-chunk order, so decode == prefill holds because the *same
  instruction* does the accumulation. Decode is DRAM-bound and the tensor pipe is idle (§2.1), so the wasted MACs are
  free in principle; what is not known is whether an mma-based GEMV reaches the 93 % of DRAM the dp4a GEMV does.
  **ESTIMATED cost: 0 % if it does, up to the gap if it doesn't; one pod hour to find out.** Cross-architecture identity
  for FP8 is then *per `sm_XY`*, which the tuple already keys on.
- **The fast-mode option:** cuBLASLt-class or hand-written split-K FP8 GEMMs with whatever accumulation is fastest,
  graded by tolerance. This is what every datacenter engine does. **If Axon builds FP8, it needs the mode to exist
  first** so the FP8 round can ship the fast path and grade the exact path against it, instead of blocking on the
  uncosted exact path.

### 2.6 Batch-size-dependent kernels in serving

- Axon already switches kernels by nb: decode GEMV at 1, `gemvm`/`gemvn` at small nb, `gemm2` 64×64 or 64×128 tiles
  above (`src/cuda/qwen2/batch.rs:795-803`). The contract permits that *because* every one reproduces `gemv_ref`'s
  order (identity C). So the cost is not "no batch-dependent kernels"; it is "every new kernel must reproduce one order",
  which rules out kernels that reduce across CTAs by arrival (split-K with atomics, stream-K) and anything that
  accumulates in a different precision per path (f16 accumulate in attention: r5-long-prefill §4.2, ≤ 1.5 %
  MEASURED-bounded).
- **The C-curve (MEASURED, 4090, 7B):** after #81's routing fix the nb 32 step is 9.81 ms vs 6.53 at nb 1 (1.81×;
  `docs/log/2026-10-18-c-curve.md`). What's left at depth: attention at 4096 is 18.6 of 31.8 ms at nb 64 and already at
  ~80 % of DRAM (bytes, not order); the pool's nb 1 step is 1.20× solo because of separate q/k/v GEMMs and `gemvm8` at
  ~700 GB/s (kernel shape, order-preserving). Against vLLM: −13 / −21 / −24 % at C 8/16/32, on different weights (AWQ
  int4, Marlin) and an engine with async scheduling. **Share of that gap attributable to the contract: ESTIMATED ≤ 5
  points**; the rest is attention bytes, host-side sampling and the single engine thread (review §5 risk 4), none of
  which the contract touches. Flash-decoding's split-KV at *fixed* split size is, like §2.2, an exact-mode option
  (`research-determinism.md` P4, ESTIMATED ≤ 3.6 % at 32k).

### 2.7 The one place the contract is measured to bite: the CPU

MEASURED (research-determinism §1, PROGRESS night 1): 1.5B pp560 CPU prefill 734 tok/s with a free float order (r8) vs
458–574 bit-identical (r8-bi) vs 343–417 before either. **1.3–1.6× on CPU prefill**, the largest cost the contract holds
anywhere. The review's thesis ("the contract was chosen when the target was a laptop CPU") inverts the evidence: the
CPU is where it costs most, and the GPU is where it is nearly free.

### 2.8 Summary table

| technique | forbidden today? | buys (label) | verdict |
|---|---|---|---|
| stream-K prefill GEMM | yes | ≤ 2 % after #92 and r16 pick 2 (ESTIMATED; MMQ parity MEASURED) | not worth a mode on its own |
| split-K prefill GEMM | yes | ≈ 0 (74 waves of CTAs; MIO-bound, MEASURED r17) | nothing to buy |
| split-K decode GEMV, narrow rows | arrival-order: yes; fixed-chunk: **no** | H100 batch 1: part of a shared, uncosted 43 % → ~85 % gap (ESTIMATED ≤ 15 % of step); consumer: ~0 (93 % of DRAM MEASURED) | do P3 in exact mode first |
| per-32 float epilogue | yes (as llama.cpp does it) | 0–4 % of the GEMM kernel (MEASURED wash / NO_EPI +2–4 %) | no |
| NCCL all-reduce (TP) | yes | ≈ 0 at decode, 1.5–3 % of prefill vs a rank-ordered reduce (ESTIMATED) | fixed-order TP is fine; fast mode may use NCCL for the convenience, not the speed |
| FP8 tensor-core accumulation | de facto (no exact design) | the whole FP8 path on Hopper/Blackwell | **the real reason to have a mode** |
| batch-dependent kernels in serve | only arrival-order ones | ≤ 5 points of the 13–24 % vLLM gap (ESTIMATED) | no |
| CPU free float order | yes | 1.3–1.6× CPU prefill (MEASURED) | fast mode, CPU, when anyone asks for CPU prefill |

---

## 3. What it buys

| property | mechanism | who notices |
|---|---|---|
| **Lossless speculative decoding, token for token** | identity B + F: verify rows equal decode, one RNG draw per decided token, keep iff the draw is the draft | every user of `--draft`: +37–63 % greedy tg128 (MEASURED, review §1) with *zero* quality question to answer. Engines without B need the Leviathan accept/reject proof and still change bits (§4.4) |
| **Exact chat / serve cache** | identity D + G + J: KV bytes are a pure function of the token prefix | a saved session restores and continues bit for bit (29.8k positions in 0.41 s vs 3.16 s re-prefill, MEASURED #48); a slot paged, chunked, preempted or restored continues as if uninterrupted. llama-server's `cache_prompt`, `--cache-ram`, `--ctx-shift` each carry a "not reproducible" caveat |
| **md5 as the regression net** | identity L | every kernel PR is gated by `gen` md5s, not tolerances: a wrong mask is a different hash, not a KL of 0.3 that someone argues about. 88 PRs in 9 days with no silent numerics regression (methodology §"What it does well") |
| **Debuggability** | A + B + C | a GPU bug reproduces on the CPU reference at the same position (G3's masks, #19's sm<80 fallback, #35's GeGLU were all found this way); decode vs prefill divergence is a test failure, not a heisenbug |
| **Reproducible serving** | identity H | `system_fingerprint` is a real key: a client caching on `(fingerprint, prompt, params, seed)` gets exact hits; an eval run is replayable; a batched reply equals the solo reply, so load does not change answers |
| **Cross-run, cross-host replay within an identity** | tuple keys | an RTX 4050 and a 4090 agree 20/20 (MEASURED); a rented box returns the laptop's bytes |

**Is any of that customer value, or is it a dev tool? (OPINION.)** Both, in this proportion:

1. **Dev tool first, and it is the project's velocity.** The md5 gate and the CPU reference are why an agent-built
   kernel engine with 668 `unsafe` sites has had no silent numerics regression. Remove the contract and every kernel PR
   needs a tolerance argument and a critic who can tell a 2e-3 KL regression from noise. That is the cost vLLM and SGLang
   pay, and it is why their deterministic modes fail Vosti's equivalence tests (determinism.md "How this compares").
2. **Customer value, real but narrow, in three places:**
   - **Lossless speculation and exact caches are product features whose *value* is speed**, and the contract is the
     cheapest way to get them without a quality argument. The customer sees +37–63 % and a 0.4 s session restore, not the
     contract.
   - **Reproducibility at the API** (`system_fingerprint` that means something, batched == solo) is what evaluation
     pipelines, agent harnesses that replay, and regulated deployments ask for. It is the one claim in Axon that vLLM
     ships only as a beta mode at −25 to −62 % and SGLang at −34 % (CLAIMED, their numbers, determinism.md). Nobody has
     yet paid for it here, and the review is right that no customer is named.
   - **Datacenter operators do not pay for it at the top of the C-curve.** They pay for tokens per dollar. At that tier
     the contract is worth keeping only if it is free, and §2 says it is nearly free *today* and stops being free at FP8
     and at the narrow-GEMV end of Hopper decode.
3. So: **a product feature in local/agent/eval use, a dev tool everywhere, and a liability only where FP8 begins.** That
   is exactly the shape that wants a mode with a default per entry point, not a global constraint and not a removal.

---

## 4. Proposal: modes

### 4.1 The modes

```
--numerics exact     today's contract: identities A–L. Default for gen, chat, serve.
--numerics fast      order-changing kernels allowed. Identities B', D', G' (below) by tolerance,
                     E/F lossless by construction (§4.4). Default for `serve --datacenter`.
```

No third tier. The research note's "approximation" ideas (lossy attention, expert deferral, pre-gating) change the
*model*, not the arithmetic order, and belong under a different name if ever (`--approx`), with a KL gate against the
model, never under `--numerics`. Keeping `--numerics` to two values keeps the identity tuple and the gate matrix at two
columns. **(OPINION.)** If a third tier appears later it should be `exact-xarch` (cross-architecture exact: software
`exp2`, no `ex2.approx`, research-determinism P5), which is *stricter* than exact, not looser.

What `fast` is allowed to do, by category, each item opt-in by its own kernel-routing rule and only once a measured
gap justifies it (§4.6):

| category | allowed in `fast` | still required in `fast` |
|---|---|---|
| product kernels | split-K / stream-K with arrival-order or SM-count-dependent fixup; per-32 float epilogue; FP8/FP4 tensor-core accumulation; cuBLASLt-style library GEMMs | run-to-run determinism for a *fixed* (nb, placement, GPU) is **not** required; the identity says `fast` and the fingerprint is honest about it |
| attention | split-KV with SM-count-dependent splits; f16 accumulate in prefill attention | same mask, same RoPE, same KV layout: fast mode never changes *what* is computed, only the float order and precision of accumulation |
| multi-GPU | NCCL (or an equivalent) all-reduce, any algorithm | rank-count in the tuple |
| sampler, tokenizer, templates, stop handling | **unchanged**: the sampler is model-independent and costs nothing | identity F's machinery is shared |
| CPU | free float order (r8), per-32 scaling | SIMD == scalar is dropped too; the tuple says `fast` |

What `fast` is **not**: a different model, a different quantisation, a different KV type, a different context policy.
Those stay flags with their own tuple fields.

### 4.2 Gates

| gate | exact | fast |
|---|---|---|
| md5 of `gen` vs main, same host | **kept, unchanged** (identity L) | not applicable across a kernel change; kept for *no-kernel-change* PRs (fast md5 should still match main when nothing in the fast set changed) |
| A, B, D, G (decode == prefill == verify == batched, cache == fresh) | bits (existing tests) | **tolerance vs exact mode on the same device**: `assert_graded(fast, exact)` with the CPU-vs-CPU floor replaced by **exact-vs-exact across two placements the project already accepts as "the same model"** (the GPU's f16 vs q8 KV, or sm_89 vs sm_120). Concretely: top-1 within 2 points of that floor and ≥ 94 %, flips below the floor's spread, mean KL ≤ 2× floor, ≤ 1 position > 0.5, none > 2 (`src/cuda/qwen2/tests.rs:255-310`, unchanged code, new reference) |
| products vs `gemv_ref` (C) | bits | **relative error** per output ≤ 2⁻¹⁶ × ‖row‖₁·‖act‖∞ bound for int8-sourced kernels (ESTIMATED: a float reassociation of ≤ 74 block partials in f32), tolerance per kernel documented at its routing site; FP8 kernels get their own bound |
| E/F (drafting lossless) | token-for-token vs no drafting | **distribution** test: `drafting_distribution_chi_square` (`src/sample.rs:1218`) already does this on the sampler; add the same chi-square on the real model in fast mode (§4.4). Greedy: top-1 agreement ≥ the A/B floor, not equality |
| vs llama.cpp | top-20 prefill logits agreement (zoo sweep, hardening.md) | **same, plus KL and top-1 on a fixed 2k-token corpus per model class** (the GPU-vs-CPU corpus, `gpu_logits_match_cpu_2000_tokens` `:2191`), reported in the PR table; fast mode must sit inside llama.cpp's own CPU/CUDA spread (the MoE gate's wording, roadmap item 7) |
| teacher-forced ΔNLL (methodology "Numerics-changing work") | on a NUMERICS bump | **on every fast-kernel PR**, with clustered SE, against exact |
| speed | A/B interleaved (unchanged) | A/B interleaved, **and fast must beat exact by its kill rule** (§4.6), else the kernel does not ship |
| identity H (fingerprint) | unchanged | `serve_identity` variant: the fingerprint in fast mode differs from exact, is stable across restarts for a fixed placement, and is **not** promised stable across GPU count or driver when the fast set uses NCCL/library kernels (documented in the tuple's `numerics` value) |

**What the md5 gate loses and how it is replaced (OPINION).** Nothing, for exact mode. For fast mode the net is three
tests instead of one hash: the graded tolerance vs exact (catches a wrong kernel, since a *wrong* kernel moves dozens of
positions > 0.5, per the comment at `tests.rs:288-299`), the chi-square on drafting, and ΔNLL. That is the same net the
GPU-vs-CPU gate has used since G3, so the critic already knows how to read it.

### 4.3 Cache keys and `NUMERICS_VERSION`

Today: `numerics=1` (`src/identity.rs:65`), one integer, bumped on any bit change on any backend.

Proposed: **two counters and a mode tag, one tuple line.**

```
numerics=exact/1           today's line becomes this (SCHEME stays axon-identity/1: the field's value changes,
                           not the format; "1" alone is read as "exact/1" for one release)
numerics=fast/1            the fast kernel set, its own counter
```

Rules:
- `NUMERICS_VERSION_EXACT` bumps exactly as the rule at `src/identity.rs:55-64` says today. **Nothing about exact mode's
  cadence changes.**
- `NUMERICS_VERSION_FAST` bumps when a fast-set kernel changes bits *in fast mode*. A change to a shared kernel (one
  used by both modes) bumps both.
- Shared code that doesn't touch arithmetic (tokenizer, templates, stop strings, think handling) is in both counters by
  the existing rule, since it changes "what turns a request into tokens or text".
- **Cache keys:** every on-disk and in-RAM artifact that is a function of the bits carries the full `numerics` line:
  - AXKV session files (`chat --session`): the header already records the identity and a file made under another
    identity is loaded as tokens, never as bytes (`src/kvsave.rs:6-10,14`), so once the `numerics` line carries the
    mode a fast-mode session refuses to restore bytes into an exact-mode process and vice versa, with no new code. *Within* fast mode a
    restored cache is still "the same model" by tolerance, and continuing from it is the same promise as continuing a
    batched request: it is not bit-equal to a fresh prefill and the fingerprint already says so.
  - The NVRTC module cache (`~/.cache/axon/cuda`): fast-set kernels compile under a distinct define (`-DAXON_FAST`),
    which is already part of the module key (r17 §1 used `AXON_NVRTC_DEFINES` the same way).
  - Server-side prefix cache / slots: a mode is a process-wide setting, so slots never mix modes. Per-request mode
    selection (the review's "`seed`-style opt-in") is **rejected**: a slot's KV written by fast kernels cannot be
    continued exactly by exact kernels, so a per-request switch would need two caches per slot or a re-prefill. Run two
    processes.
  - `system_fingerprint` / `GET /identity`: unchanged mechanics, the `numerics` value differs. A client replaying by
    fingerprint across modes gets a miss, which is correct.
- `axon info` prints both predicted fingerprints per placement (`src/cli.rs:395-415` already loops placements).

### 4.4 Speculative decoding in fast mode stays lossless: how

Axon's accept rule (`src/sample.rs:23-30`, `generate` `:512-600`) is not the Leviathan/Chen draft-vs-target ratio test.
It is: *draw the position's token from the target's candidates with the position's own uniform; keep the draft iff the
draw equals it; otherwise emit the draw*. Every decided token consumes exactly one RNG number whether or not a draft was
in flight. Consequences:

1. **The emitted sequence is a sample from the target's distribution at every position, by construction, for any
   drafter, including an adversarial one.** The drafter never enters the probability; it only decides whether the next
   position's logits are already in hand (verify row) or need a forward. In exact mode that gives token-for-token
   equality with plain sampling (identity F) because verify rows equal decode rows.
2. **In fast mode, verify row *i* ≠ decode row *i* in the low bits.** The sampled token at position *i* is then drawn
   from `p_verify(i)`, which is the fast-mode model evaluated through the batched kernel path. That is a legitimate
   evaluation of the model under fast numerics, exactly as a batched request's logits are under batch-invariance-free
   engines. So the output is **exactly distributed as "the fast-mode model sampled through the verify path"**, and the
   *only* gap from plain fast-mode decoding is `p_verify` vs `p_decode`, which is identity B' and gated by tolerance
   (§4.2). Lossless-in-distribution holds with respect to the kernel path that produced the sample; there is no
   drafter-induced bias to prove away, because the rule never used the drafter's distribution.
3. **Rejection leaves the cache consistent:** on a reject the queue is cleared (`:551`) and the emitted token is run at
   its position, by `forward_cands` or as row 0 of the next `verify_cands` (`:560-575`), overwriting the KV row verify
   wrote for the wrong guess. Accepted tokens keep verify's KV
   rows, which in fast mode are tolerance-equal, not bit-equal, to what decode would have written. That is the same
   state a chunked prefill leaves in any batch-variant engine.
4. **Gate:** `drafting_distribution_chi_square` on the real model in fast mode (30k seeds on the sampler today; on a
   model, 2k seeds × first two tokens is enough for a χ² at p > 0.01), plus greedy top-1 agreement with no-draft decoding
   at the A/B floor. Acceptance rate is reported but not gated (a fast verify path that disagrees with decode at near-ties
   lowers acceptance slightly: ESTIMATED < 1 point at the 94 % top-1 floor).
5. **Batched speculative decoding (#90)** inherits all of this: rows are independent, and the rule is per row.

### 4.5 Defaults

| entry point | default | why (OPINION) |
|---|---|---|
| `axon gen` | `exact` | a CLI run is a reproducibility tool; the md5 gate runs through it |
| `axon chat` | `exact` | session files, cache == fresh prefill, drafting; a person's own machine where the contract is free |
| `axon serve` (one card, LAN) | `exact` | the agent-harness case: replay, exact prefix cache, `system_fingerprint` that means something; the C-curve gap is not the contract (§2.6) |
| `axon serve --datacenter` (new flag bundle: implies `--numerics fast`, and later TP / FP8 / NCCL as they exist) | `fast` | tokens per dollar; FP8 has no exact path; the fingerprint says `fast` so no client is misled |
| `axon info` | prints both | — |
| tests, md5 gate, scoreboards | `exact` unless the test is a fast-mode test | the regression net stays bit-exact |

`--numerics` is accepted by every entry point, so a datacenter can run exact and a laptop can run fast; the defaults
are about what a person expects when they don't ask. `--datacenter` is a bundle name, not a mode, so it can grow
(`--parallel`, `--kv q8`, TP degree) without the numerics flag growing.

### 4.6 The rule that keeps the fast set empty until it earns a kernel

**No fast-set kernel is built on a prediction.** Each one needs: (1) a measured gap in exact mode on the mission cell
(an H100 / B200 at C ≥ 32, or FP8), (2) an exact-mode attempt or a written reason none exists (as §2.5 for FP8), and
(3) a kill rule, e.g. "fast GEMV on H100 batch 1 ≥ +10 % tok/s vs exact after P3 lands, else delete". This is r16 §1's
"not worth it" column made into process. Without it the mode becomes a place to dump unvalidated kernels, which is the
review's §5 risk 1 with the sign flipped.

### 4.7 Migration plan: PR-sized steps

| PR | change | bits | size | gate |
|---|---|---|---|---|
| **M1** `--numerics` flag, tuple line `numerics=exact/1`, `NUMERICS_VERSION_{EXACT,FAST}` constants, `fast` **refused** ("no fast kernels on this backend yet"), `axon info` prints both, docs: this ADR linked from determinism.md | none (the canonical text changes; `serve_identity` and `identity.rs:683,709` tests updated; **this is a fingerprint change for every client and deserves its own release note**) | 4–8 h | md5 20/20; all identity tests |
| **M2** `--datacenter` bundle on `serve` (implies `fast` once it exists; today: refused, or exact with a warning; decide at M1 review) | none | 2 h | serve tests |
| **M3** Test scaffolding: `Mode`-parameterised `assert_graded(fast, exact)` helper, the exact-vs-exact floor computed from f16-vs-q8 or sm_89-vs-sm_120 (whichever is available on the host), the fast-mode χ² on the real model, ΔNLL script hook | none | 8–12 h | runs green in exact-vs-exact (sanity), skipped where `fast` is refused |
| **M4** The CPU fast set: the r8 free-order prefill branch behind the flag (research-determinism P6), `cpu` tuple field says `fast` | fast only | 8–12 h | CPU 1.5B/3B/7B pp560 ≥ +25 % vs exact (MEASURED 734 vs 458–574 says it clears); KL ≤ 1e-3 vs exact |
| **M5** NVRTC cache key carries `-DAXON_FAST`; a kvsave test that a cross-mode session file restores as tokens, not bytes | none in exact | 2–4 h | kvsave tests |
| **M6** *(exact mode, not fast)* P3 fixed-chunk split-K chain, all product kernels, one NUMERICS_EXACT bump, measured on an H100 at batch 1 | exact bits change once | 2–3 days + 1 pod h | `gpu_products_match_reference_bits` re-pinned; md5 references re-pinned; H100 7B tg128 ≥ +5 % or revert |
| **M7** *(first GPU fast kernel, only after a measured gap)* candidates in order of expected value: FP8 wgmma GEMM on Hopper (needs the FP8 weight path first: review §4), then split-K narrow GEMV if M6 leaves a gap, then NCCL all-reduce in TP (after TP exists in exact form, multi-gpu.md §3.3) | fast only | per kernel | §4.2 fast column + its kill rule |
| **M8** Roadmap: "Not planned" bullet on split-KV / NCCL / atomics rewritten as "not in exact mode" and pointing here | docs | 1 h | — |

M1–M3 and M5 are the whole mode and cost under a week with no kernel work. M4 is the one fast kernel set that already
exists on a branch. M6 is exact-mode work that removes the strongest single argument for a GPU fast set. M7 is gated on
measurement by rule.

**As built (2026-10-29, PR #PRNUM; [log](../log/2026-10-29-numerics-modes.md)).** Where the build differs from the plan
above, and why:

- **M1: `fast` is accepted, not refused.** With no fast kernel on a backend, `--numerics fast` runs the exact kernels,
  prints one notice (`no fast kernels on this backend yet; running exact kernels; identity says fast`) and identifies
  as fast everywhere: `numerics=fast/1` in the tuple and the fingerprint, `-DAXON_FAST` in its NVRTC modules' options
  (their own kernel-cache entries), `numerics_mode` in its kvsave identity. Why not refuse: (1) M2's
  `serve --datacenter` implies fast; refusing fast would make the bundle refuse to start until a kernel lands, so no
  deployment could adopt the flag now and keep it; (2) M3 and M5 need a fast mode that runs, or their tests are
  skipped everywhere and the first fast kernel lands on untested scaffolding; (3) it is honest: a fast fingerprint
  promises tolerance, never exact bits, so a client that gets exact bits under it was promised less than it got, and
  the first fast kernel bumps `NUMERICS_VERSION_FAST`, which tells fingerprint-caching clients. The notice keeps anyone
  from reading `fast` as faster today.
- **M1: the mode is a `Backend`'s** (`BackendOptions::numerics`, public; `axon::Numerics`, non-exhaustive), and the
  CLI's per process. Exact modules' NVRTC options are main's, byte for byte, so their cache keys, cubins and bits are
  too (the md5 gate checks the bits). The `numerics` field's JSON (`GET /identity`) is now the string `"exact/1"` (was
  the number 1). Nothing in Axon reads the field back (fingerprints are compared as hashes), so §4.3's "'1' alone is
  read as 'exact/1' for one release" has no reader to apply to.
- **M2: `--datacenter` implies only `--numerics fast` today**; an explicit `--numerics` wins over it, so a datacenter
  can serve exact.
- **M3: the floor** is the cross-arch one (exact on a second GPU of another architecture in the same process) when the
  host has one, else this GPU's f16 vs q8 KV cache, both exact; on the CPU, the r8 vs per-row kernels. No Axon host has
  two architectures in one box today, so the cross-arch branch is written but not yet exercised. The fast-mode χ² on
  the real model (§4.4) is not in this step: with no fast kernel, fast's verify rows are exact's, so the existing
  identity-F tests already cover it bit for bit; it belongs to the first fast kernel's PR, next to the kernel that
  breaks B. The ΔNLL hook is `scripts/numerics_dnll.py` (through `axon logits` with `AXON_LOGITS_NLL=1`) plus the same
  measurement in-process in the graded tests.
- **M5: exact sessions are unchanged.** A fast-mode backend adds `numerics_mode` to its kvsave identity; exact adds
  nothing, so session files saved before the modes still restore as bytes. Draft models always run exact.

---

## 5. Positioning (the mission-drift question)

The review asks what changed between "never GPU backends, own the ≤1B CPU niche" (the 10-13 internal review) and
"datacenters", and what the mission will be in six more days. The honest answer (OPINION): the early note described the
*hardware the project had*, not the mission; the mission statement in roadmap.md has said "from a laptop to a
datacenter" since the 10-17 refresh. What drifted is the *evidence base* (every board is ≤ 8B on one card), not the
stated goal. The fix is to write down what the two tiers are and which contract each runs under, so the next review can
check the work against the words. The section below is proposed for roadmap.md, above "The thinking behind the order".

> ### Mission and positioning
>
> Haseeb's words, in order: **"make inference cheap and fast"**, **"run the best models"**, and **"run at datacenters
> without wasting resources: squeeze every drop."**
>
> Two tiers serve those, under two numerics modes ([determinism-modes.md](determinism-modes.md)):
>
> | tier | hardware | what "cheap and fast" means there | numerics | who it competes with |
> |---|---|---|---|---|
> | **Local and agent** | one consumer GPU, a laptop, a CPU box; one person or one agent swarm | the fastest single stream, lossless speculation, an exact cache that survives restarts, replies that don't change under load | `exact` (default) | llama.cpp, ollama. Axon's edge: measured single-stream speed on NVIDIA and a contract nobody else ships by default |
> | **Datacenter** | H100/B200 nodes, FP8 weights, TP, C ≥ 32 | tokens per dollar at goodput; TTFT/ITL p99 under load; frontier MoE models | `fast` (default under `--datacenter`), with `exact` available for eval and replay | vLLM, SGLang, TRT-LLM. Axon's edge is not yet measured there; the first datacenter board (review §8 rec. 1) is what makes this row honest |
>
> "Run the best models" is one list for both tiers (Qwen3-30B-A3B, gpt-oss, DeepSeek V3 / Kimi K2), but the first tier
> runs them split across CPU and GPU and the second runs them resident at FP8. Both tiers share one engine, one
> tokenizer and one sampler; they differ in the kernel set the numerics flag selects and in what the gate measures:
> bits in `exact`, tolerance plus a board against vLLM/SGLang in `fast`.
>
> **What Axon refuses to build:** lossy approximations sold as the model (expert deferral, pre-gated routing, key
> skipping) under either mode; platforms it can't measure on (Metal, ROCm, NEON) until a machine exists; a third
> numerics tier. **What it will not claim:** "squeezes every drop" until the datacenter board is measured on the
> hardware the mission names.

**How exact and fast map onto the two tiers (OPINION, with the §2 numbers):** on the local tier the contract is free
within ~2 % on the GPU and is the product (speculation, cache, replay); on the datacenter tier the contract is free
until FP8 and TP, which is where the mode begins to earn its keep. Nothing in the mission requires giving up the
contract where it is free, and nothing in the contract should block FP8 where it is not.

---

## 6. Consequences

- **Positive:** every future kernel round answers "which mode?" instead of "does the contract allow it?"; FP8 and TP
  have a home; the md5 net is untouched for the mode that uses it; `system_fingerprint` becomes more truthful, not
  less (it says `fast` when it is).
- **Negative:** one more identity line changes every fingerprint once (M1); the test matrix gains a column that only
  runs where a fast set exists; two numerics counters to keep honest; the temptation to build fast kernels before the
  gap is measured (§4.6 is the brake).
- **Rejected alternatives:** (a) *per-request mode* (review's "`seed`-style"): a slot's KV cannot be shared across modes,
  so it is two processes in disguise (§4.3). (b) *Drop the contract globally*: throws away the regression net that is the
  project's velocity and the two features that are measured product wins, for ≤ 2 % on the kernels that exist. (c)
  *Keep it global and never do FP8*: forfeits the datacenter tier by construction. (d) *Three tiers*: a looser third tier
  is an approximation mode, not a numerics mode; a stricter one (cross-arch exact) is a possible future `exact-xarch`.

## 7. Open questions for Haseeb, answered

Answered 2026-10-29 (Haseeb: "lets try fast"; the lead's answers to the four questions):

1. M1 changes every fingerprint once. Ship it with the next NUMERICS bump that would happen anyway, or now?
   **Now.** It ships with a release note (CHANGELOG.md): every fingerprint changes once, no output bit does.
2. `serve --datacenter` as the name of the bundle, or `--profile datacenter`? **`serve --datacenter`.**
3. Is CPU fast prefill (M4, the one measured 1.3–1.6×) worth 8–12 h now, given nobody runs CPU prefill at scale?
   **No: deferred, out of scope.**
4. M6 (fixed-chunk split-K in exact mode) is a one-time exact-bits change. Approve the principle before the H100 hour?
   **Deferred, out of scope** (its 4090 attempt, #148, already failed its kill rule).
