# Multi-GPU: pipeline parallel first, tensor parallel second

**Status: design. Stage 1 (pipeline parallel) is in progress; nothing multi-GPU is on main yet.**

Written against main `f8d8a86`, and its code references were rechecked on `5f96039`. Labels:
**CODE** (read in the tree), **MEASURED** (a cited run), **ESTIMATED** (arithmetic shown). Every
number for a rented multi-GPU box is ESTIMATED by scaling single-GPU measurements. The experiment in
§5 is meant to replace them.

## 1. Summary

- **Build pipeline parallel (PP) first.** It is `hybrid.rs` with a GPU in place of the CPU. The tree
  already has layer-range GPU models (`CudaQwen2::new_layers` with a `first` layer, `Input::Hidden`,
  `BatchIn::Resident`), a single hidden-state hop per stage boundary, and a one-thread driver that can
  hold several `Gpu`s (CODE). PP is **bit-identical to one GPU by construction**: every layer runs
  whole, on one device, with the same kernels.
- What PP buys: models that don't fit one card (72B on 2×A100-80 or 4×4090; 32B at long context on
  2×4090) and **KV room for serving**, because the weights are held once instead of once per replica.
  What it doesn't buy: single-stream decode speed (≈ −0.3%, ESTIMATED).
- **Tensor parallel (TP)** is the only option that makes one stream faster. On PCIe 4090s the gain for
  32B is ESTIMATED at 1.4–1.6×, not 2×, because of 128 all-reduces per token. On NVLink A100 it is
  ~1.5× for 32B and ~1.8× for 72B (ESTIMATED).
- TP changes the reduction order of `wo` and `ffn_down`. Within one TP configuration, decode ==
  prefill == verify still holds. **TP-N bits ≠ TP-1 bits** unless we adopt the shard-invariant order
  of §3.3, which makes 1/2/4/8-GPU bits equal at about zero cost.
- **Honest caveat:** for a model that fits one card, two independent replicas (data parallel) beat PP
  and need no code. Most of the multi-GPU lead over vLLM would be Axon's single-GPU lead carried over,
  plus KV room.

## 2. Shapes and cost model

Shapes (CODE, GGUF metadata):

| model | dim | heads / kv | head dim | ffn | layers | Q4_K_M size |
|---|---:|---|---:|---:|---:|---:|
| Qwen2.5 7B | 3584 | 28 / 4 | 128 | 18944 | 28 | 4.68 GB |
| Qwen2.5 32B | 5120 | 40 / 8 | 128 | 27648 | 64 | 19.85 GB |
| Qwen2.5 72B | 8192 | 64 / 8 | 128 | 29568 | 80 | ~47.4 GB |

Step cost: on Parabellum's RTX 5070 Ti Laptop GPU (672 GB/s spec, 648 GB/s measured ceiling), 7B
decode takes 8.47 ms per token and its GEMVs run at 93% of the ceiling. About 1.3 ms of that is not
GEMV. A verify pass of 8 tokens costs 1.45× one forward (all MEASURED).

Scaled, ESTIMATED: a 4090 (1008 GB/s) does 7B in ~5.6 ms (~175 tok/s) and 32B in ~23.8 ms (~42 tok/s).
For 72B under PP on A100-80s, ≈ 47.4 / (0.9 · 2039) + 3 ms ≈ 28.8 ms.

Links (ESTIMATED; there are no multi-GPU logs yet):
- Pinned PCIe 4.0 x16: ~25 GB/s. MEASURED 25.4 GB/s host-to-device on Parabellum, via the
  `h2d_bandwidth` test.
- P2P small copy + event: ~10–15 µs. Without P2P the driver stages through RAM: ~30–40 µs and
  ≤ 12 GB/s.
- NVLink A100: ~250 GB/s per direction, small hop ~5–8 µs.

## 3. Options

### 3.1 Pipeline parallel (layer split)

Stage *s* holds layers `[k_s, k_{s+1})`. Stage 0 holds the embedding. The last stage holds the final
norm, the head and the sampling. Data moved:
- Per token, one copy of the hidden state per boundary: 7B 14 KB, 32B 20 KB, 72B 32 KB.
- Per prefill batch of T tokens, T·dim·4 B per boundary (32B, T = 1024: 21 MB ≈ 1 ms on PCIe).
- KV: each stage keeps its own layers' cache, so total KV bytes don't change.

**Single-stream decode:** the stages run one after another. 32B on 2×4090 ≈ 23.8 ms + 2 hops ≈ 23.9 ms
(ESTIMATED). 72B Q4_K_M on 2×A100-80 ≈ 35 tok/s (ESTIMATED).

**Capacity** (ESTIMATED):
- 72B Q4_K_M needs 47.4 GB + KV, so 2×A100-80 or 4×4090. 72B Q3_K_M (~37.7 GB) fits 2×4090 with
  ~10 GB left for KV.
- 32B on 2×4090 under PP keeps ~28 GB for KV, against ~8 GB for two replicas. At 256 KB/token that
  is ~27 concurrent 4k contexts instead of 8.

**Serving:** with two or more step batches in flight (requires S2 continuous batching, in flight),
stage 1 works on batch A while stage 0 works on batch B. Aggregate throughput then approaches 2× one
GPU at the same per-stream latency. For a model that fits one card this equals data parallel; for one
that doesn't, it is the only option. The hop cost at decode barely grows with batch size (32B, 16 rows:
320 KB ≈ 25 µs).

**One long prompt:** cut it into micro-batches of `prefill_batch` tokens (1152 on the GPU, CODE). Stage
1 runs chunk *i* while stage 0 runs chunk *i+1*. The bits don't change, because results don't depend on
batching (the contract). ESTIMATED ≈ 1.8× on prompts of two or more chunks.

**Bit identity: exact.** Each layer's kernels, inputs and KV rows are those of the single-GPU run, and
the hop copies f32 bits. It can be tested on **one** device: open device 0 twice (two `Gpu`s, two
stages; the peer copy becomes a device-to-device copy).

**Balance:** each stage's bytes are its layers' weights + KV + scratch, from `CudaQwen2::vram_need`
(already a pure function, CODE). The search has the shape of `hybrid::most_gpu_layers`, per device.
Uneven cards get uneven cuts.

### 3.2 Tensor parallel (Megatron-style)

The split, across N GPUs:
- q/k/v and gate/up are split by output rows. Each GPU attends over its own heads; a GQA group stays
  whole, so N must divide `n_kv_heads` (7B: 4; 32B and 72B: 8).
- `wo` and `ffn_down` are split by input columns. Each GPU produces a partial sum, followed by an
  all-reduce: **two per layer**.
- Norms, residuals, embedding and head are replicated. KV is split N ways.

**Kernel constraint (CODE, `decode.cu` "Row dot products"):** the input to `wo`/`down` is quantized to
Q8_K in 256-value blocks, so shard boundaries must fall on 256-column multiples. 32B with N = 2 or 4
divides cleanly. 7B with N = 4 and 72B with N = 4 need uneven shards, which cost at most one block of
imbalance.

**Communication** (N = 2): each GPU sends 2 × dim × 4 B per layer per token. For 32B that is 128 syncs
and 2.6 MB per token. A 1024-token prefill moves ~2.7 GB per GPU, ≈ 107 ms at 25 GB/s. That is only
partly hideable, because each all-reduce depends on the GEMM before it.

**Decode latency, 32B, N = 2 (ESTIMATED):**

| box | GEMV per GPU | 128 syncs | other | step | vs 1 GPU |
|---|---:|---:|---:|---:|---|
| 2×4090, P2P on (15 µs) | 10.6 ms | 1.9 ms | 2.3 ms | 14.8 ms | 42 → 68 tok/s (1.6×) |
| 2×4090, P2P off (35 µs) | 10.6 ms | 4.5 ms | 2.3 ms | 17.4 ms | 57 tok/s (1.4×) |
| 2×A100 NVLink (7 µs) | 5.2 ms | 0.9 ms | 1.5 ms | 7.6 ms | 1.5× (72B: 1.8×) |

7B with TP2 on 4090s comes out at ≈ 1.25×, which is not worth the work. The host enqueues ~4 driver
calls per sync (copy, record, wait, graph segment), ≈ 2–3 ms of CPU per token, so it must stay at
least one token ahead of the GPU.

### 3.3 Bit identity under TP, precisely

The canonical order (CODE: `decode.cu` "Row dot products"; `prefill.cu`'s `gemm2` and `gemv_ref`
follow it) is one fma chain over a row's 256-blocks in order, then the residual add. A row-sharded `wo`
computes chain(blocks 0..B/2) on GPU 0 and chain(B/2..B) on GPU 1, then adds them. That rounds
differently from the single chain. So:

- **decode == prefill == verify inside a TP-N run: holds.** Every path takes the same per-shard chain
  and the same partial-sum order: a fixed tree over GPUs, never NCCL's ring. It is deterministic
  across runs.
- **TP-N == TP-1: does not hold** with the current chain. (GPU == CPU doesn't hold today either.)
- Chat cache reuse, lossless drafting and serve identity all hold within one TP configuration.

**The fix, cheap: a shard-invariant order.**
- Replace the single chain with V = 8 fixed chunks of blocks, summed as a balanced tree
  `((c0+c1)+(c2+c3)) + ((c4+c5)+(c6+c7))`, then the residual.
- TP-N with N dividing 8 gives each GPU a subtree, and the all-reduce sums the N partials in the same
  tree. **TP-1, 2, 4 and 8 then give identical bits.**
- Cost: ~7 extra fadds per output per matrix, plus a one-time, deliberate bits change on a single GPU
  too (every GPU md5 re-pinned).
- `gv_chain` (decode), the gemvn/gemvm/gemm2 epilogues and `gemv_ref` change together;
  `gpu_products_match_reference_bits` keeps them equal.
- Decide before stage 2; don't do it in stage 1.

### 3.4 Combinations

- **PP × TP** (4×4090 as 2 stages × TP2): 32B gets ~1.6× single-stream, and 72B fits. A stage is
  written so that it can later be a TP group.
- **CPU + GPUs:** `hybrid.rs`'s CPU half becomes stage 0 of the pipeline.
- **Draft model on GPU 1, target on GPU 0:** §4, item 4.

## 4. Where this could beat vLLM on PCIe consumer boxes

Speculative parts are labeled.

1. **PP with many in-flight sequences, no NCCL.** vLLM's default on 2 GPUs is TP. On 4090s without
   P2P, its two all-reduces per layer go through host memory (ESTIMATED 4–6 ms per token at 32B). Axon
   PP pays one 20 KB hop per token (~30 µs) at any batch size and keeps its single-GPU kernel lead.
   Expected: at 8–32 clients, 32B on 2×4090 delivers ≈ 2× single-GPU Axon in aggregate. vLLM's PP
   bubbles at low concurrency, and ours would too. **Speculative:** that vLLM TP on no-P2P 4090s is as
   slow as the arithmetic says. Measure it (§5, step 6).
2. **KV room.** Weights are held once: ~3.5× the concurrent 4k contexts on 2×4090 for 32B. Any PP
   engine gets this. It is the honest reason PP beats data parallel for serving.
3. **Micro-batched prefill and prefill/decode disaggregation without numerics drift.** Because Axon is
   batch-invariant, a prompt prefilled in any chunking on any identical GPU writes the bytes decode
   would have written. GPU 0 can prefill new requests while GPU 1 decodes running ones. The KV moves
   once: 32B, 1024 tokens = 268 MB ≈ 11–22 ms on PCIe. vLLM also disaggregates; our edge is only that
   correctness comes free. **Speculative:** whether p99 inter-token latency matters more to users
   than aggregate throughput. With two GPUs, disaggregation gives less throughput than PP when prefill
   is a small share of the work.
4. **Draft model on a second GPU.** The draft model guesses its next K tokens optimistically while the
   target verifies. ESTIMATED ≈ 1.5× on top of speculation when acceptance is high (code, RAG), ≈ 1.0×
   when low. Bits unchanged. **Speculative:** depends on the acceptance distribution.
5. **TP that reproduces single-GPU output** (§3.3). A rented 2-GPU server would return the same bytes
   as a laptop with the same GPU model. NCCL's reduction order doesn't allow vLLM to promise this. No
   speed value, but real value wherever outputs must be reproducible.

**Not claimed:** beating vLLM TP on NVLink H100/A100 for one stream (their all-reduce is ~5 µs there
and their kernels are mature), or large-batch prefill throughput on H100 ([hopper.md](hopper.md)).

## 5. Cheap first experiment (2×4090 pod, ≤ 3 h at ~$0.68–1.78/h)

This needs no Axon multi-GPU code. Each step has a prediction to check:

1. `nvidia-smi topo -m`, link width and generation; a small probe of `cuDeviceCanAccessPeer` /
   `cuCtxEnablePeerAccess`. Prediction: **P2P off** on most consumer pods.
2. Hop microbenchmark: `cuMemcpyPeerAsync` of 20 KB, 320 KB, 21 MB and 268 MB, each with an event
   wait, 1000× each, P2P on and off. Prediction: 20 KB at 10–15 µs on / 30–40 µs off; 21 MB ≈ 1 ms on
   / ~2 ms off.
3. TP all-reduce loop: 128 × (20 KB peer copy each way + event + a one-block add kernel) per "token".
   Prediction: 1.5–2.5 ms per token with P2P on, 3.5–6 ms off.
4. Data-parallel baseline: two `axon gen` processes (`CUDA_VISIBLE_DEVICES=0` / `1`), 7B and 32B,
   alone and together. Prediction: 7B tg ~175, 32B tg ~42, and both-at-once within 3% of alone.
5. PP-hop proxy: `--gpu-layers` on one 4090 with a one-layer CPU half. The hybrid's per-token
   host-to-device hop is an upper bound on a PP hop. Prediction: the 32B step rises ≤ 0.3 ms.
6. Reference: `vllm serve` a 32B AWQ model with `--tensor-parallel-size 2` (and `-pp 2`), at 1 and 8
   clients. Prediction without P2P: ~25–35 tok/s at 1 client; 150–250 aggregate at 8.

**Go:** build stage 1 if step 2 is ≤ 50 µs per hop and step 4 confirms the single-GPU numbers within
15%. Build stage 2 for PCIe only if step 3 is ≤ 2 ms per token. Otherwise stage 2 targets NVLink
A100/H100, where step 3 should be ~1 ms.

## 6. Build plan

### Stage 1: PP (ESTIMATED 25–35 engineer-hours; no kernel changes). In progress.

- `cuda/ffi.rs`: add the peer-access and peer-copy entry points. They aren't on main yet.
- `cuda/mod.rs`: `Gpu::enable_peer`, `copy_peer` and a cross-context event wait. All `Gpu`s stay on
  the one driver thread.
- `cuda/qwen2.rs`: `new_layers` gains an end layer and a head flag, so a middle stage carries no
  head; resident input for decode (batched already has `BatchIn::Resident`).
- New `pipeline.rs`: `PipelineQwen2` implementing `Lm`. Decode runs the stages in turn with peer
  copies and events. `feed_batch` enqueues chunk *i* on stage *s* after stage *s−1*'s chunk *i*, so
  chunks overlap. Includes a pure, tested placement function `pipeline_cuts`.
- CLI: a GPU-list flag (e.g. `--gpus 0,1`) and `Backend::Pipeline`. `serve` needs no change, since it
  is generic over `Lm`.
- Later (stage 1b, after S2 lands): two in-flight step batches, so `serve` overlaps stages. The 2×
  aggregate comes from this step.

Tests (one device, opened twice):
- Pipeline logits and per-layer KV bytes bit-equal to `CudaQwen2`, for 2 and 3 stages and several cuts.
- Batched and verify paths over the pipeline equal one call, at chunk sizes 3, 7 and 64.
- Placement balance, including uneven free memory.
- `gen` md5s with a one-GPU pipeline equal today's.

Gates on 2×4090 (ESTIMATED):
- 32B decode ≥ 0.97× one 4090.
- 72B Q3_K_M ≥ 20 tok/s.
- `serve` at 8 clients on 32B ≥ 1.7× one GPU (after 1b).

### Stage 2: TP2 (ESTIMATED 60–90 engineer-hours; kernel and host work). Planned.

(a) The §3.3 order change, alone and on a single GPU, as its own PR with an A/B (expect ±0%) and the
md5 re-pin. (b) Sharded upload by rank (row ranges for q/k/v/gate/up, block-column ranges for wo/down).
(c) A fixed-order `tp_reduce_add` kernel, with graphs per (GPU, layer) segment. (d) A `TpGroup` with the
same surface as `CudaQwen2`, so a pipeline stage can be a group. Gate on a pod: 32B TP2 decode ≥ 1.4×
single with P2P on. Otherwise TP is declared NVLink-only and the PCIe story stays PP.

## 7. Risks

- Stage 1b depends on the shape of S2's batching API. Stage 1 stays on `Lm` so it can land
  independently.
- A driver-staged peer copy may order against the sender's stream differently than a P2P copy does.
  The same-device test can't see this; only the pod run can.
- VRAM headroom constants (`HEADROOM` and the desktop margin) are per device today; PP must apply them
  per stage.
- Opening a device's context costs time. Only device 0's context is pre-initialized today; other
  devices pay at open.
