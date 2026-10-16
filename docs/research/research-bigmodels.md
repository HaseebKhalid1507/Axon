# research: bigmodels — dense 14B–70B and MoE on a 12 GB card (CPU/GPU hybrid, weight streaming, experts on the CPU)

Read-only research note, 2026-10-16. Axon main `e998d6f` (exported to a scratch directory),
llama.cpp `4413a19` (2026-10-12), ollama `main` (`llm/llama_server.go`, fetched from GitHub). Papers: PDFs under
a scratch directory (not in the repo). Labels: **MEASURED** (Axon logs/specs, or a paper's own measurement on the
hardware named), **ESTIMATED** (my arithmetic, shown), **CLAIMED** (paper's headline, not reproduced), **CODE**
(file:line). No cargo, no model runs.

Machine constants used throughout (all MEASURED unless marked): bella = RTX 5070 Ti Laptop 12 GB (11.2 GiB free),
648 GB/s VRAM ceiling; 30 GB DDR5, CPU effective ≈ **71 GB/s** on a dense GEMV (7B Q4_K_M CPU decode 15.1–16.6 tok/s ×
4.68 GB; sb-cpu-881d7b6.md, PROGRESS 136); CPU int8 prefill ≈ **3.4 TFLOP/s** (7B pp512 222 tok/s × 15.2 GFLOP/token,
ESTIMATED from the MEASURED rate); PCIe Gen5 x8 **25.4 GB/s pinned** H2D, ~8–12 GB/s pageable (g4-offload.md);
Core Ultra 9 275HX: AVX2 + AVX-VNNI, **no AVX-512, no AMX**.

---

## 1. The problem in Axon's terms

- **32B dense on 12 GB** (MEASURED, PROGRESS 682): 28/64 layers on the GPU, decode **4.79 tok/s** (llama.cpp 3.66),
  pp560 328–390 and pp2100 ~407 tok/s streamed (llama.cpp 211 / 264–286). The 36 CPU layers are ~10.8 GB of weights:
  10.8 / 71 GB/s = **152 ms** of an observed 209 ms step. No kernel changes that; only fewer RAM bytes per token do.
- **Streaming decode is 2× slower than the CPU computing** (MEASURED, hybrid.rs:88–91 note). The reason is structural
  (§4, dead end 1): PCIe DMA and the CPU's GEMV read the *same* DRAM, so on bella every byte of a RAM-resident weight
  costs ≥ 1/75 GB/s whichever device consumes it, and PCIe alone is 3× slower than that.
- **14B at 32k** drops to 17 tok/s because a 3.2 GiB q8 slot is *reserved* (r10 §1.2); that is B1 (paged KV), not
  this topic — nothing here changes that diagnosis.
- **MoE is refused** (CODE model.rs:493–494). Qwen3-30B-A3B (18.6 GB Q4_K_M, 17.6 GB of it experts) and gpt-oss-20b
  (12.1 GB, 10.2 GB of MXFP4 experts) are the two models a 12 GB + 30 GB box can run at "30B-class" quality, and
  llama.cpp's `--cpu-moe` / `--fit` made experts-on-CPU the default way to run them.
- **70B on 2×24 GB** runs under PP (bit-identical by construction) but is unmeasured (B3).

The gap to close is therefore not "stream better" but **"read fewer RAM bytes per generated token"**: MoE sparsity
(1.1 GB/token instead of 10.8), VRAM-resident subsets of experts, and speculative decoding (several tokens per weight
pass). Everything below is ranked by that lens, under the bit-identity contract.

---

## 2. Literature map

Read in full (method + evaluation): Fiddler, KTransformers, "Not All Models Suit Expert Offloading", HOBBIT (§2–4),
ProMoE (§2–4), MoE-Lightning (§3–4), PowerInfer (§1–3), LLM in a flash (§1–3). Abstract + skim: the rest.

### 2.1 Dense offloading and activation sparsity

| paper | idea | numbers (hardware) | fit for Axon |
|---|---|---|---|
| **FlexGen** (Sheng et al., ICML 2023, arXiv 2303.06865) | Offload weights/KV/activations across GPU/CPU/disk; zig-zag block schedule; huge batches to amortize weight I/O; CPU attention option | OPT-175B on one T4 16 GB + 208 GB RAM: 1 token/s at effective batch 144 (MEASURED by the paper) | Throughput-only; batch=1 latency is worse than CPU compute. Lesson kept: *amortize a weight pass over many rows* → speculative verify rows (P3). |
| **Deja Vu** (Liu et al., ICML 2023, arXiv 2310.17157) | "Contextual sparsity": lookahead MLP predictors pick the attention heads and MLP neurons to compute per token | OPT-175B latency > 2× lower vs FasterTransformer on A100 (CLAIMED); up to 85 % sparsity on OPT | **Lossy** (skipped neurons are an approximation; accuracy "no degradation" within benchmark noise). ReLU-era OPT; not Qwen. |
| **PowerInfer** (Song et al., SOSP 2024, arXiv 2312.12456) | Hot neurons on GPU, cold on CPU, online predictors choose which neurons to compute; neuron-aware sparse operators | RTX 4090: 13.20 tok/s avg quantized, 8.32 non-quantized on 40–70B ReLU models, up to 11.69× over llama.cpp (MEASURED by the paper) | **Lossy** via predictors; requires ReLU-family sparsity (paper's own Fig: SwiGLU LLaMA2-13B only 43 % sparse vs 97 % for OPT-30B). Qwen2.5/3 are SwiGLU → not applicable. |
| **PowerInfer-2** (Xue et al., 2024, arXiv 2406.06282) | Same idea on a phone: neuron-cluster granularity, NPU for prefill, I/O pipeline from flash | Mixtral 47B on a 24 GB phone 11.68 tok/s (CLAIMED) | Same lossy predictor dependency; mobile I/O design. Not applicable. |
| **LLM in a flash** (Alizadeh et al., ACL 2024, arXiv 2312.11514) | Flash → DRAM streaming of FFN neurons with a *window* of recent tokens' active neurons kept resident; row-column bundling so up/down rows for one neuron are one read; low-rank ReLU predictor | OPT-6.7B / Falcon-7B with half the model in DRAM: 4–5× (CPU) and 20–25× (GPU) over naive loading on an M1 Max (MEASURED by the paper) | Predictor is lossy (Table 1: "marginal" zero-shot impact); ReLU models. The exact part (bundling consecutive rows into one DMA) is already how Axon streams whole matrices. |

### 2.2 MoE offloading (CPU experts, GPU experts, prediction, pipelines)

| paper | idea | numbers (hardware) | fit for Axon |
|---|---|---|---|
| **Fiddler** (Kamahori et al., ICLR 2025, arXiv 2402.07033) | Non-expert weights on GPU, experts in RAM; **per expert per layer, run on the CPU unless its token count makes transfer+GPU cheaper** (Alg. 1: `cpu_lat(s) > gpu_lat + transfer_lat` → GPU). Popular experts pinned on GPU by offline profile. AVX512-BF16 CPU kernel. | Mixtral-8x7B fp16 (90 GB) on RTX 6000 24 GB + 48-core Xeon (PCIe3) and RTX 6000 Ada + 112-core Xeon: 1.26× llama.cpp single-batch, 1.30× long prefill, 11.57× beam search (MEASURED by the paper). Weight transfer is 2–5× the GPU compute time per expert (Fig. 7). Popularity placement gains only 3–5 pp hit rate over random (App. C). | **Exactly the plan for B2's decode path**, and the right prefill rule once CPU==GPU bits (P2). The device choice depends on routing counts → breaks "batched == alone" unless the two devices agree bit for bit. |
| **KTransformers** (Chen et al., SOSP 2025, DOI 10.1145/3731569.3764843) | Shared experts + attention on GPU, routed experts on CPU with **AMX tile-layout int4/int8 kernels** (switch to AVX-512 when tokens-per-expert ≤ 4); CUDA graphs around the CPU call via a lock-free queue; **Expert Deferral**: 3 of 8 routed experts' outputs are added one layer late so the CPU overlaps the GPU's next attention | Dual Xeon 8452Y (220 GB/s/socket) + A100/RTX 4080: 21.3 TFLOPS CPU MoE kernel (3.98× PyTorch); decode 1.25–1.76× llama.cpp fp16, 1.77–1.93× quantized; deferral +33 % decode throughput, accuracy drop ≤ 0.5 % (MEASURED by the paper) | AMX is absent on bella (AVX-VNNI only → Axon's `r8q` VNNI path is already the equivalent). CUDA-graph-around-CPU is worth copying. **Deferral changes the model** (O_k = I_k + S_k + R^def_{k−1}(I_{k−1}) + R^imm_k) → lossy → dead end 4. |
| **MoE-Lightning** (Cao et al., ASPLOS 2025, arXiv 2411.11217) | CGOPipe: CPU-GPU-I/O pipeline with paged expert weights and **attention on the CPU** (KV never leaves RAM); Hierarchical Roofline Model to pick micro-batch sizes | Mixtral 8x7B on a T4 16 GB: up to 10.3× FlexGen-class systems (MEASURED by the paper), batch throughput | Throughput regime (big batches). CPU attention is wrong on bella (KV on the 648 GB/s side, 71 GB/s RAM). The HRM way of reasoning (bytes per level) is what §4 uses. |
| **MoE-Infinity** (Xue et al., 2024, arXiv 2401.14361) | Batch-1 "activation traces" drive expert cache replacement and prefetch; experts stream from host | 3.1–16.7× per-token latency vs vLLM/Ollama/DeepSpeed on DeepSeek/Mixtral (CLAIMED) | Exact (prediction only orders fetches). Trace machinery is heavier than LRU for small gain (§5.2 of the routing paper: LRU is 67–75 % of optimal at ρ = 2–3). |
| **Pre-gated MoE** (Hwang et al., ISCA 2024, arXiv 2308.12066) | Train a *pre-gate* in layer N−1 that selects layer N's experts, so expert fetch overlaps compute | 23 % overhead vs all-in-GPU-memory, 4.2× less peak GPU memory (CLAIMED) | **Requires fine-tuning the model** → not applicable to stock GGUFs. |
| **EdgeMoE** (Yi et al., TMC 2025, arXiv 2308.14352) | Experts on external storage, fetched on activation; expert-wise bitwidth adaptation; preloading by prediction | Mobile devices; speedups over naive (CLAIMED) | Bitwidth adaptation is lossy; the exact half (fetch-on-activation) is what llama.cpp's MoE cache does. |
| **HOBBIT** (Tang et al., 2024, arXiv 2411.01433; on llama.cpp) | Mixed precision: cache-miss experts replaced by int4 copies when their gate score is low; **next-layer prediction by feeding the current layer's input to the next layers' gates** (96 % next-layer accuracy on Mixtral, Fig. 7); multi-dimensional cache | Up to 9.93× decode vs offloading baselines on RTX 4090 and Jetson Orin (MEASURED by the paper); loading one Mixtral fp16 expert over PCIe4 ≫ its 3 ms compute | Mixed precision = lossy (dead end 5). The **skip-gate predictor is free and exact as a prefetch hint** (P7). |
| **ProMoE** (Song et al., 2024, arXiv 2410.22134; on llama.cpp + transformers) | Learned per-layer MLP predictor from layer-i hidden state to layer-(i+k) experts ("stride"), run on the CPU; chunked prefetch; reordered execution to hide misses. Defines GoodPred = accuracy × fetch-rate | Avg 2.20× prefill / 2.07× decode over reactive caching on consumer GPUs (CLAIMED). Skip-gate predictor only 66.9 % avg on Qwen1.5/2-MoE, learned 84.7 % (MEASURED by the paper) | Exact (prefetch only). On bella decode is PCIe/RAM *bytes*-bound, not latency-bound (§4), so prediction hides ≤ the per-layer sync (~5–10 %). Low priority. |
| **Klotski** (Fang et al., 2025, arXiv 2502.06888) | Expert-aware multi-batch pipeline: order batches so the expert set loaded for layer L is reused across batches; I/O-compute planner; correlation prefetcher | Up to 85× throughput vs offloading baselines (CLAIMED), batch regime | Throughput-only; single-user bella gains nothing. |
| **Not All Models Suit Expert Offloading** (Liang et al., ICLR 2026, arXiv 2505.16056) | Two metrics: SRP (how well a fixed expert set covers a token segment) and SCH (oracle cache hit rate at cache ratio ρ = slots / active experts). 20 models analysed | **Qwen3 (30B-A3B) is a group-1 model: SRP 54.1 at m = 16** (Table 2), with the SCH knee at **ρ ≈ 2**; LRU reaches 56–75 % of the oracle hit rate at ρ = 1–3 (Table 4); shared experts and tight load balance *lower* consistency (MEASURED by the paper, routing traces) | The one paper that answers "is an expert cache worth it for Qwen3": yes, with a knee at ~16 cached experts per layer (2.2 GB on bella); see §4.2. gpt-oss (32 experts, top-4) is not in their set. |
| **HybriMoE** (Zhong et al., DAC 2025, arXiv 2504.05897; on KTransformers) | Dynamic intra-layer CPU/GPU split of the routed experts by current load; impact-driven inter-layer prefetch; score-based cache | 1.33× prefill / 1.70× decode over KTransformers (CLAIMED; 4090-class) | Exact (misses computed on the CPU) — the Fiddler dynamic rule, again needing CPU==GPU bits for Axon. |
| **DAOP** (DATE 2025, DOI 10.23919/date64628.2025.10992741) / **DALI** (2026, arXiv 2602.03495) | Data-aware CPU/GPU expert assignment + predictive pre-calculation; DALI: runtime 0-1 assignment by greedy, residual-based prefetch prediction, workload-aware cache | Speedups over KTransformers-class on local PCs (CLAIMED) | Same family as Fiddler/HybriMoE; refinements of the placement decision. |
| **MoE-SpeQ** (2025, arXiv 2511.14102), **SPICE** (2026, arXiv 2608.21240) | A *draft model* predicts future tokens → future experts → prefetch; SPICE adds low-rank expert surrogates for low-confidence misses | 2.34× (Phi-MoE) and 3.12× TPOT (DS-V2-Lite, Qwen2-57B) (CLAIMED) | Surrogates are lossy. The draft-predicts-experts idea is interesting only because Axon already runs a draft model; but see §4.2: bytes, not latency, bound bella. |
| **MoE-CORE** (2026, arXiv 2610.01950) | Alternating full-layer expert buffers in prefill; decode: non-uniform per-layer cache capacity, routing-history replacement, cross-layer prefetch; "executes router-selected experts exactly" by default, optional lossy substitution | DeepSeek-V4-Flash TPOT 38–45 ms vs 1269 ms for vLLM prefetch under an 84 GB cap (CLAIMED) | Validates two of our choices: stage whole expert layers for prefill (Axon's streamer), exact-by-default with lossy as opt-in. |
| **Cloud-Grade SLOs for Local MoE** (2026, arXiv 2606.10493) | "Stream-loading prefill" (SLP): experts streamed through the GPU per prompt chunk; AVX-512 FP8 GEMV; PD disaggregation in one node | 1,200 tok/s prefill on DeepSeek-V3 (dual-socket + consumer GPU), 28 tok/s decode INT4 (CLAIMED) | SLP is Axon's streamer applied to experts — the plan in r10 §1.3. |
| **SeqMoE** (2026, arXiv 2609.12978) | Expert prediction as sequence modelling (multi-step, multi-layer), job-sequencing prefetch, probabilistic Belady eviction, **graph-compatible (sync-free) offload runtime** | 45 % residency → 96.97 % hit rate, 80 % of full-load speed (CLAIMED) | The sync-free runtime point matters (llama.cpp's cache syncs per layer, §3.2). Prediction stack is heavy. |
| **ATSInfer** (2026, arXiv 2607.10183), **FreeToken** (2026, arXiv 2608.16157), **MawForge** (2026, arXiv 2607.09686) | Tensor-granularity placement with load-aware dynamic transfer (ATSInfer: +1.94× prefill, +3.29× decode on consumer boxes, CLAIMED); whole-stack edge MoE serving (FreeToken: 35B on an 8 GB laptop GPU, CLAIMED); on-demand expert materialization from disk on unified memory (MawForge: "effective as a bounded mechanism, not as a cache-maximization policy") | The 2026 consensus: static whole-layer `-ngl` is dead; placement is per tensor/expert and adaptive. | Axon's hybrid is still a single layer cut (hybrid.rs:1–3); per-tensor placement is the step B2 forces anyway. |
| **Memory-Sovereign Inference: Output-Exact Execution** (2026, arXiv 2608.23805) | A falsifiable *exactness certificate* for storage-backed MoE inference (Qwen3-Next): token IDs, full logit rows and route events equal a zero-cache oracle; LRU-64 expert cache | Exact under an 11 GB host / 24 GB device cap; async 8-window path at 32 % of the blocking path's wall time (MEASURED by the paper, one model/device) | Someone else now sells what Axon's contract already is. Cite it when positioning B2: exact expert offload is a differentiator vs lossy systems. |

---

## 3. What production engines do

### 3.1 llama.cpp (`4413a19`)

**Placement flags.** `-ot/--override-tensor <regex>=<buft>` (CODE common/arg.cpp:2756–2761); `-cmoe/--cpu-moe` pushes
`{"\.ffn_(up|down|gate|gate_up)_(ch|)exps", cpu_buft}` (arg.cpp:2762–2768, common.h:1239, 1247–1249);
`-ncmoe/--n-cpu-moe N` adds one override per layer `blk\.<i>\.ffn_…exps` for i < N (arg.cpp:2769–2777,
common.h:1251–1257); `-ncffn/--n-cpu-ffn N` does the same for dense FFNs (arg.cpp:2789–2798). The override buft is the
plain CPU buffer type, so with mmap (default) the experts stay in **pageable** mapped pages; the pinned
`CUDA_Host` buft is only used for CPU tensors when not mmapped (llama-model.cpp:1136–1150 adds the host buft;
:1916 uses mmap when `is_default_buft`). Hence the folk advice `--no-mmap` with `-ot`: it turns the per-op uploads
from ~8–12 GB/s into ~25 GB/s on bella.

**`--fit` (default on) is MoE-aware.** `common_params_fit_impl` first measures memory with *all* experts moved to the
CPU (fit.cpp:616–646, pattern `blk\.\d+\.ffn_(up|down|gate_up|gate)_(ch|)exps`), then fills devices with dense
weights and overflows per layer with a fraction enum `ATTN < UP < GATE < MOE` (fit.cpp:18–24, 493–530, 560–585), i.e.
it ends up at "attention + dense on GPU, experts of the first k layers on CPU" automatically.

**Decode path with CPU experts.** The scheduler assigns an op to the backend of its weight (ggml-backend.cpp:1105–1127);
`MUL_MAT_ID` over a host weight runs on the CPU backend: `ggml_compute_forward_mul_mat_id` (ggml-cpu.c:1554–1731):
one barrier (1669), then for each expert with rows, chunks of 16 rows handed out by a per-expert atomic counter
(1686–1731), no barrier between experts; the tiled path takes over when profitable (1679). The hidden state and ids
cross PCIe as scheduler inputs (ggml-backend.cpp:1823–1863). The GPU parts are captured as CUDA graphs per split
subgraph, keyed by the split's first node (ggml-cuda.cu:2677–2679, 4634–4660) — the KTransformers claim that
llama.cpp "disables CUDA graphs" with CPU splits is dated.

**Prefill path with CPU experts = per-op offload.** For batches ≥ 32 tokens (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32,
ggml-cuda.cu:6037; `get_op_batch_size` returns `ne[2]` = tokens for `MUL_MAT_ID`, :5848–5861; `offload_op`
:5863–5867) the scheduler moves the op to the GPU and copies the host weight. The copy callback
`llama_context::sched_copy_experts` (llama-context.cpp:2758–2830, registered at :711) reads the ids back
(`ggml_backend_tensor_get_async` + `ggml_backend_synchronize`), marks used experts, and uploads **only the used
experts**, consecutive ones in one `tensor_set_async`, plus 512 B of padding so MMQ never reads NaNs (:2815–2830). The
copies go on the compute stream with no double buffering; from pageable memory they also block the host.

**GPU MoE kernels.** Dispatch in `ggml_cuda_mul_mat_id` (ggml-cuda.cu:1992–2030): `ne2 ≤ MMVQ_MAX_BATCH_SIZE` and a
per-type cap (`get_mmvq_mmid_max_batch`, mmvq.cu:150–188; e.g. Q2_K 7, MXFP4 7 on Turing+) → `mul_mat_vec_q` with ids
(mmvq.cu:600–720: `channel_x = ids[channel_dst]`), fused gate+up+GLU for `MUL_MAT_ID` (ggml-cuda.cu:1721, 1859);
larger → MMQ with ids: `mm_ids_helper` (mmid.cu:27–120) sorts token slots by expert into a compact src1 with
`expert_bounds`, activations are quantized to q8_1 *after* the gather (mmq.cu:285–346), one grouped kernel; otherwise
the host-side fallback that syncs, sorts on the CPU and loops experts (ggml-cuda.cu:2040–2140; asserted never under
graph capture, :1961–1990). Routing: `ggml_argsort_top_k` on the probabilities (llama-graph.cpp:2237), fused on CUDA
by `topk-moe.cu` (warp softmax + top-k + optional renorm).

**Experimental GPU expert cache: `--moe-cache-mib N`** (arg.cpp:2778–2788; `llama_moe_cache`, src/llama-moe-cache.cpp).
Per device, layers whose experts are all host-resident share banks of `n_slots` experts per tensor (:243–330) with one
LRU over (layer, expert) keys (:16–120, `plan`: hits touched first, sorted misses take the LRU head so uploads
coalesce). Per MoE layer per token the graph does a `get_rows(slot_map, ids)` whose host-weight input triggers the copy
callback (llama-graph.cpp:2492–2530, :1665 `mul_mat_id(cached_experts, cur, slots)`); the callback reads the ids,
**synchronizes the device** (:548–550), plans, uploads the missing experts (:553–567), then copies the updated slot
map. Disabled for ubatches > 32 tokens or when `n_tokens × n_expert_used > n_slots` (:196, :462–473); large batches
instead copy cached experts device-to-device and upload the rest (`copy_experts`, :480–511, called from
sched_copy_experts :2815–2826). Not compatible with pipeline parallelism (llama-context.cpp:517–523). **Exact**:
every expert still runs on the GPU from identical bytes; the cache only changes where they come from. Hit/miss stats
are logged at exit (:583–600) — a free hit-rate instrument for the §6 experiment.

### 3.2 ollama (`main`, Oct 2026)

ollama now **launches `llama-server` as a subprocess** and inherits its `-ngl auto` / `--fit` placement
(llm/llama_server.go:13–14, 427–434: `num_gpu` > 0 → `-ngl N`, 0 → `-ngl 0`, −1 → nothing). It passes
`--cache-type-k/-v` (:417–419) and `--flash-attn` (:422, 626–650); grepping the launcher finds **no `--cpu-moe`,
`-ot` or `--moe-cache-mib` passthrough** — users get fit's MoE-aware overflow but no control over it.

### 3.3 vLLM / SGLang / TensorRT-LLM (docs)

vLLM: CPU offload exists (`--cpu-offload-gb`, per-layer weight swapping) and is documented as a last resort; MoE
expert parallelism is multi-GPU only; no CPU expert compute. SGLang has integrated KTransformers as a CPU-expert
backend (lmsys blog 2025-10-22, "CPU/GPU hybrid expert parallelism"), i.e. the Fiddler/KTransformers design is now a
datacenter-grade option too. TensorRT-LLM: no host offload path worth citing for consumer boxes.

### 3.4 What Axon has today (CODE, `e998d6f`)

Single layer cut, CPU first (hybrid.rs:1–26, `choose_placement` :242–276); streamer for CPU layers at batch ≥ 32
(`STREAM_MIN` hybrid.rs:81; stream.rs:1–29: pinned in place via `Gpu::register`, 2 × 16 MiB staging, 3 streams,
un-permute kernel, ring of ≥ one layer + biggest matrix, KV rows copied back to the f16 CPU cache); streamed prefill
gives the CPU layers' positions **GPU numerics** (stream.rs:26–29) — the hybrid's contract is already "compare against
a fresh prefill done the same way". CPU weights `r8`/`r8q` repack: integer-first per super-block, then one f32 step
per (row, super-block) in block order, *no FMA* (r8.rs:1–40: `acc += (d·a.d)·f32(sumi) − (dmin·a.d)·f32(summ)`); GPU
canonical chain: `v = d·S − dmin·M; acc = fma(dk, v, acc)` (decode.cu:323–345) — "not the CPU's order". This
difference is the whole obstacle to dynamic expert placement (P2).

---

## 4. Bandwidth arithmetic for bella (ESTIMATED unless marked)

### 4.1 Shapes

| | Qwen3-30B-A3B Q4_K_M | gpt-oss-20b (ggml-org GGUF) |
|---|---|---|
| layers / experts / top-k | 48 / 128 / 8, no shared expert, `norm_topk_prob` | 24 / 32 / 4, softmax over the top-4 only |
| expert matrices | gate, up: 2048×768; down: 768×2048 → 4.72 M params | gate_up fused 2880×5760, down 2880×2880 → 24.9 M params |
| bytes per expert | 17.6 GB / 6144 = **2.86 MB** (0.606 B/param; file 18.6 GB minus ~1.0 GB non-expert) | MXFP4 4.25 bit → **13.2 MB** (+ bf16 biases) |
| expert bytes total | 17.6 GB | 10.2 GB |
| expert bytes per token | 8 × 48 × 2.86 = **1.10 GB** | 4 × 24 × 13.2 = **1.27 GB** |
| non-expert weights | ≈ 1.0 GB (attn 0.54, embed 0.18, head 0.26, routers 0.05) | ≈ 1.9 GB |
| KV per token (f16) | 48 × 2 × 4 × 128 × 2 = 96 KiB (3 GiB at 32k; 1.6 q8) | 48 KiB; half the layers SWA-128 → ~24 KiB effective |

### 4.2 Decode, Qwen3-30B-A3B

| layout | RAM bytes/token | VRAM bytes/token | step | tok/s | exact? |
|---|---|---|---|---|---|
| A. all experts on CPU (r10 B2 as written) | 1.10 GB → 15.5 ms at 71 GB/s | 1.0 GB + KV@8k 0.75 GiB → 2.8 ms | + 48 hops × 2 × ~15 µs ≈ 1.5 ms (README 152: ~10 µs pipeline hop) ⇒ **~20 ms** | **~50** (40–55 across 65–75 GB/s) | yes: experts always on the CPU |
| B. static per-layer: experts of 16/48 layers resident (6.0 GB VRAM; VRAM budget: 1.0 + 6.0 + KV 1.6 q8@32k + ring/scratch 1.5 ≈ 10.1 GiB) | 0.73 GB → 10.3 ms | 1.0 + 0.37 GB → 2.1 ms (+KV) | ⇒ **~14 ms** | **~70** | yes: a fixed per-layer device, as hybrid today |
| C. LRU expert cache, misses **streamed** to the GPU, all experts computed on the GPU (llama.cpp `--moe-cache-mib` model) | 0 (CPU idle) | hits from VRAM: ≤ 1.7 ms | misses over PCIe: 384 × (1−h) × 2.86 MB / 25.4 GB/s = **43 (1−h) ms** + 3 ms | h = 0.5: 40; h = 0.75: 70; **h = 0.85: 100** | yes by construction |
| D. LRU cache, hits on GPU, **misses computed on the CPU** (Fiddler/HybriMoE) | 1.10 (1−h) GB | hits | h = 0.75: 3.9 + 1.3 + 1.5 + hops 1.5 ⇒ **~8.5 ms** | **~110** | **only with CPU==GPU bits (P2)** |

Where h comes from: the routing-consistency paper puts Qwen3 in its highest-consistency group with the oracle cache
knee at ρ ≈ 2 (16 slots per layer, 2.2 GB) and LRU at ~67 % of oracle there (Table 4, a toy baseline; MEASURED on
traces). With the ~6 GB bella can spare (ρ ≈ 5.5, 44 slots per layer) a realistic LRU hit rate is **0.6–0.8** — to be
MEASURED with llama.cpp's own cache stats (§6 Q2) before anyone builds a cache. Reading the table:

- **C never beats B unless h > 0.85.** The demand-fetch cache is exact and adaptive, but PCIe at 25 GB/s is the slow
  side of this machine; a static placement that reads the same bytes from RAM at 71 GB/s wins at any plausible h.
- **Prediction/prefetch does not change the bytes.** At decode a layer's attention is ~0.1 ms, so prefetching layer
  l+1's experts under layer l hides at most the per-layer sync (~1–2 ms/token, 5–10 %). The lever is the *hit rate*
  (cache size, policy), not the predictor. ProMoE/HOBBIT/SeqMoE matter on PCIe-latency-bound boxes with big GPUs.
- **D is the only layout above 100 tok/s**, and it is reachable only if an expert's output is the same bits on both
  devices. That is proposal P2.

Comparison point: llama.cpp `--cpu-moe` on DDR5 laptops is reported by users at 20–30 tok/s for 30B-A3B (CLAIMED,
community; not measured here). Its CPU `mul_mat_id` is also RAM-bound, so the expected gap to layout A is the same
1.25–1.7× Axon already shows on CPU decode (sb-cpu), plus whatever `--fit` keeps resident.

### 4.3 Decode, gpt-oss-20b

All experts on CPU: 1.27 GB / 71 GB/s = 17.9 ms + GPU 1.9 GB → 3 ms + 24 hops → **~22 ms, ~45 tok/s**. With 16 of 24
layers' experts resident (6.8 GB; budget 1.9 + 6.8 + KV@32k 0.75 + scratch 1.0 ≈ 10.5 GiB): RAM 8 × 4 × 13.2 MB =
0.42 GB → 6 ms; VRAM 1.9 + 0.85 GB → 4.3 ms; ⇒ **~11–12 ms, 80–90 tok/s**. Needs the MXFP4 dot (E8M0 block scale
× FP4 E2M1, 32-value blocks), attention sinks and the 128-token sliding window in every attention kernel (B6).

### 4.4 Prefill with streamed experts (both models)

Per prompt chunk (≤ 4096 tokens) the streamer must bring every expert not already resident: Qwen3 17.6 GB / 25.4 GB/s
= **0.69 s** (0.46 s with 6 GB resident); gpt-oss 10.2 GB → 0.40 s (0.13 s with 6.8 GB resident). GPU compute for a
4096-token chunk: Qwen3 experts 4096 × 8 × 9.44 MFLOP × 48 = 14.9 TFLOP + dense/attention ≈ 8 TFLOP; at the 40–60
TFLOP/s Axon's GEMMs reach (7B pp8k 3999 tok/s × 15.2 GFLOP = 61 TFLOP/s MEASURED-derived; grouped small-M GEMMs
lower) ⇒ 0.4–0.6 s, overlapped with the copy ⇒ **≈ 0.7–0.8 s per 4096 ⇒ 5000–5800 tok/s ceiling, 3000–4500 realistic**.
A 560-token prompt still pays the whole 0.69 s ⇒ ~700 tok/s.

**The MoE break-even is not 32 tokens.** CPU experts cost N × 8 × 48 × 2.8 µs = **N × 1.07 ms** (9.44 MFLOP per
token-expert at 3.4 TFLOP/s); streaming costs 0.69 s + N × 0.12 ms ⇒ break-even **N ≈ 730 tokens** (≈ 480 with 6 GB
resident). Dense layers stream at 32 because every token uses every byte; an MoE token uses 1/16 of the bytes the
stream must move. Per expert the Fiddler rule gives the same answer: upload (113 µs) beats CPU when an expert has
> ~40 tokens in the chunk; a 4096 chunk averages 256 per expert (all GPU), a 560 chunk averages 35 (mixed). Any rule
keyed on *chunk* counts makes bits depend on co-batched rows → P2 again, or a per-request rule (prompt length) with
the hybrid's existing "same way" caveat.

### 4.5 Dense 32B: split vs. streaming, and the shared-DRAM floor

CPU half (36 layers, 10.8 GB): compute 152 ms at 71 GB/s (MEASURED-derived) vs. stream 10.8 / 25.4 = 425 ms ⇒ the
measured ~2× loss of `AXON_STREAM_DECODE=1`. Mixing them does not help: the DMA engine and the CPU cores read the same
DDR5, so **t ≥ (RAM bytes/token) / 75 GB/s + (VRAM bytes/token) / 648 GB/s** for any split of the work. Only three
things lower the dense 32B step: fewer RAM bytes (more layers on the GPU: q8 KV, paged KV so the cache isn't a 32k
reservation; or a smaller quant — different numerics), more tokens per pass (speculative verify, P3), or a second GPU.
Attention-on-GPU/FFN-on-CPU per layer (`--n-cpu-ffn`) moves the same bytes; it only changes where the KV sits (all on
the GPU, costing weight room) — not a win on 12 GB.

### 4.6 70B on 2×24 GB (PP)

Llama 3.3 70B Q4_K_M 42.5 GB over 2×4090: 2 × 21.25 GB / (0.93 × 1008 GB/s) = 45 ms + 2 hops + ~3 ms ⇒ **~48 ms,
~21 tok/s**; 2×5090 (1792 GB/s): ~28 ms, **~35 tok/s**. KV room 48 − 42.5 − ~1.2 scratch ≈ 4.3 GB ⇒ 13.7k tokens f16 /
~26k q8 at 320 KiB/token. 72B Q4_K_M (47.4 GB) does not leave KV room; Q3_K_M 37.7 GB does (~10 GB). TP is out
(MEASURED 2.24 ms all-reduce loop per token on no-P2P 4090s, PROGRESS 888). A 1B draft adds ESTIMATED +40–60 % greedy
(the verify pass costs 1.45× one forward at nb = 8, MEASURED on bella; the same kernels run on the pod).

---

## 5. Proposals for Axon, ranked

| # | proposal | change | gain (basis) | bit identity | risk | builder-h | gate |
|---|---|---|---|---|---|---|---|
| **P1** | **B2 core: MoE with per-layer expert placement.** New `moe` block in model.rs (`ffn_gate_inp`, 3-D `ffn_{gate,up,down}_exps`); router GEMV + softmax + **exact top-k (score desc, index asc) + renorm** on the GPU (one small kernel; batched rows independent); per layer either (i) experts resident on the GPU: a `mul_mat_id`-style path = sort rows by expert (an `mm_ids_helper` twin), per-expert `gemm2`/`gemvm` over gathered rows, weighted sum **in top-k rank order**; or (ii) experts on the CPU: a gathered 8-expert GEMV over the `r8` repack with **two barriers per layer** (gate+up stage, down stage — not 24 barriers), the D2H of (normed hidden, ids, weights) and H2D of the sum folded into one sync each way. Placement by VRAM: fill from the last layer with resident experts, as `most_gpu_layers` does for layers (hybrid.rs:168–184). Prefill: CPU experts below ~700 tokens per request, streamed experts above (the existing streamer, per expert tensor; ring ≥ one layer of experts = 366 MB) | 30B-A3B decode **~50 tok/s** all-CPU-experts, **~70** with 16 layers resident; prefill 3000–4500 tok/s on long prompts (§4.2, §4.4) | **Per device, as hybrid's**: a layer's experts always run on one fixed device for decode; prefill ≥ threshold streams them to the GPU → same relaxation the dense hybrid already documents (stream.rs:26–29). Decode == verify (both CPU path). Top-k exactness: GPU router logits come from the canonical chain → deterministic → the select is a pure function. | medium: new arch, 3-D tensors, per-layer device mixing; the CPU gather kernel's barrier count decides whether 15 ms becomes 25 | 80–120 | logits vs llama.cpp within its CPU/CUDA spread; `feed == feed_batch` on CPU-expert layers below the stream threshold; chat reuse == fresh (same way); bench vs `llama-server --cpu-moe` and `--fit` on bella: tg128, pp560, pp2100; tg ≥ 45 at 0 resident layers |
| **P2** | **Cross-device exact expert block.** Make the CPU `r8`/`r8q` f32 step the GPU's (`v = d·S − dmin·M` by fmul/fmul/fsub, `acc = fma(dk, v, acc)` with AVX2 FMA; Q6_K `fma(dk·d, f32(S), acc)`), re-pin CPU md5s once; one shared Q8_K activation quantizer (same amax→d, same `rint(x·inv_d)`); one shared SiLU with a bit-exact `expf` (the N3 CPU `vexp` polynomial ported to CUDA with fma); the 8-way weighted sum in rank order with fma on both. Only the expert FFN crosses devices in an MoE (attention/norm/rope stay on the GPU), so this is tractable where the dense hybrid's full CPU==GPU is not | Unlocks layout D (**~110 tok/s**, §4.2), Fiddler's per-expert prefill rule (§4.4) and an LRU cache whose misses run on the CPU — all exact. Also makes CPU-expert prefill == streamed-expert prefill bits, restoring `feed == feed_batch` for MoE at every batch size | **Creates** identity where none exists; a deliberate one-time bits change on the CPU path (its md5 suite re-pinned). GPU bits unchanged | medium-high: `expf` and the quantizer are the hard parts; a `gpu_products_match_cpu_bits` test is the proof | 30–50 | unit test: CPU vs GPU per-row outputs of one Q4_K and one Q6_K matrix bit-equal for 10k random Q8_K activations; then the whole expert block (gate/up/silu·up/down) bit-equal; then P1 with the dynamic rule on, md5s identical for all-CPU vs mixed placement |
| **P3** | **Speculative decoding on the 32B split, tuned for a slow verifier.** Drafting already runs on hybrid (`hybrid+drafter 6/6`, PROGRESS 859) but was never measured on 32B; the solo gains (+37–63 % at K = 3) assumed a verify of 1.45× a forward on an all-GPU model. On the split the verify's CPU half reads the weights once for all rows (bytes 152 ms vs compute (K+1) × 10.8 ms at 3.4 TFLOP/s) ⇒ verify(8) ≈ 1.1–1.3× a token while the 0.5B draft on the GPU costs ~1 ms/token ⇒ K = 6–8 is right, not 3 | **4.79 → 9–14 tok/s greedy** (ESTIMATED: α = 0.5–0.7 on code, K = 7 ⇒ 2.0–3.1 accepted per 1.15–1.3× step). The cheapest big win in this theme | **none**: verify == decode per token is the existing contract (CPU rows for CPU layers, GPU for GPU) | low; sampled decoding gains ~0 (as on main) — greedy/agent loops only | 4–8 (K policy per backend, a bench) | tokens with/without draft equal on 32B; tg ≥ 9 tok/s greedy on a code prompt, bella cool-gated; K sweep {3, 5, 7, 9} recorded |
| **P4** | **GPU expert cache, exact variant (llama.cpp's `--moe-cache-mib` shape).** LRU banks per layer group, demand-fetch over pinned PCIe, slot map on the GPU, device-side `get_rows` of slots → the `mul_mat_id` path reads banks. Decode value only if h > 0.85 (§4.2 C); the **prefill value is unconditional**: a chunk's streamer skips resident experts (llama.cpp `copy_experts`), cutting the 0.69 s floor by the resident fraction and lowering the CPU/stream break-even (§4.4) | decode: 0 to +40 % depending on h (MEASURE first, §6 Q2); prefill: −35 % stream time at 6 GB resident; short-prompt TTFT 0.69 → 0.45 s | yes by construction (same bytes, same GPU kernels); the cache state changes only *where* bytes come from | medium; per-layer sync for the ids (llama.cpp syncs 48×/token — fold the ids readback into the existing hop) | 30–40 (after P1) | hit-rate log ≥ 0.7 on a 20k agent trace at 6 GB; md5 identical with cache on/off; pp560 TTFT ≤ 0.5 s |
| **P5** | **B6 gpt-oss-20b**: MXFP4 expert dot (32-value blocks, E8M0 scale; integer-first like the others), attention sinks (one extra logit per head in the softmax), 128-token SWA on alternate layers, 16/24 layers' experts resident | **80–90 tok/s** decode (§4.3), the second most-asked model | per-device as P1; sinks and SWA are new numerics tuples (new md5s) | medium: three new kernel features touch every attention kernel | 60–90 (after P1) | logits vs llama.cpp CUDA; tg ≥ 70 on bella; SWA layer KV == full-attention KV on the last 128 positions |
| **P6** | **B3 70B pod measurement** + draft: Llama 3.3 70B Q4_K_M on 2×4090 (and 2×5090 if offered), tg128/pp2100, `--kv q8` at 16k, md5 vs 1-GPU A100-80; then `--draft` Llama-3.2-1B | documented **~21 tok/s** (4090) / **~35** (5090) + 40–60 % with a draft (ESTIMATED §4.6) | none (PP is exact; drafting is exact) | none | 4–8 + a pod | numbers on record; md5 equality across cuts |
| **P7** | **Expert prefetch hint (skip-gate predictor)**: feed layer l's post-attention hidden through layer l+1's router to start the next layer's CPU gather (or PCIe fetch) before layer l's experts finish — HOBBIT's 96 % (Mixtral) / ProMoE's 66.9 % (Qwen MoEs) next-layer accuracy; only orders work, never changes which experts run | hides the per-layer sync: **+5–10 %** on layouts A/B/D | none (a hint) | low | 8–12 (after P1) | tg +5 % with the hint on, identical tokens |

Ordering rationale: P3 first because it is nearly free and attacks the one number users see (4.8 tok/s); P1 is the
round; P2 decides whether B2 is "a 70 tok/s model" or "a 110 tok/s model with exact dynamic placement" and should be
prototyped (its unit test) *during* P1's design, since the CPU gather kernel's f32 step is written then anyway.

---

## 6. Dead ends (for Axon, on this hardware)

1. **Streaming weights for decode, in any mix.** PCIe (25.4 GB/s) and the CPU (71 GB/s) read the same DDR5; the floor
   is RAM bytes / 75 GB/s and the CPU alone already sits at 95 % of it. MEASURED 2× loss (hybrid.rs:88–91). Keep
   `AXON_STREAM_DECODE` as the experiment flag it is.
2. **Activation-sparsity offload (PowerInfer, Deja Vu, LLM in a flash).** Needs ReLU-family models (SwiGLU LLaMA2-13B
   is 43 % sparse per PowerInfer's own table; Qwen2.5/3 are SwiGLU) and online predictors whose misses are silently
   dropped neurons → lossy by construction. Could only ever be an opt-in mode, and no supported model benefits.
3. **Pre-gated MoE** — requires fine-tuning the model. **EdgeMoE bitwidth adaptation, HOBBIT mixed precision, SPICE
   low-rank surrogates, MoE-CORE substitution** — misses served by a different expert or a different quant: lossy.
4. **KTransformers' Expert Deferral** — adds R^def_{k−1}(I_{k−1}) into layer k: a different function, ≤ 0.5 %
   accuracy drop CLAIMED. Not Axon's model.
5. **AMX / AVX-512 kernels** (KTransformers, Fiddler, the 2026 SLO paper's FP8 GEMV) — bella's 275HX has neither; the
   VNNI `r8q` path is the right instrument, and decode is RAM-bound anyway so kernel FLOPs barely matter there.
6. **CPU attention (MoE-Lightning, FlexGen)** — right when the GPU is a T4 and RAM is 192 GB; wrong when the KV fits
   a 648 GB/s GPU and RAM is 71 GB/s.
7. **Throughput-regime pipelines (FlexGen, Klotski, MoE-Gen)** — batch tens to hundreds of sequences; a 12 GB/30 GB
   laptop serving one to a few agents has no such batch.
8. **Tensor parallel on PCIe** for 32B/70B — MEASURED over the 2 ms gate (PROGRESS 888).
9. **NVMe streaming of >RAM dense models (AirLLM)** — 72B Q4_K_M on bella: 23 GB/token over a ~5 GB/s drive ⇒ ~0.2
   tok/s floor (research-airllm.md §3). Only MoE with expert-level on-demand loading from disk (MawForge, FreeToken)
   makes >RAM models move, and the two models that matter here fit in RAM.
10. **Expert prediction as the main lever at decode** — on bella the expert path is bytes-bound (§4.2); prediction is
    a 5–10 % hint (P7), not a round.

---

## 7. Open questions a 1-hour experiment settles

1. **CPU==GPU dot bits.** Over one Q4_K and one Q6_K matrix with 10k random Q8_K activation rows: GPU `gv_chain`
   outputs vs a CPU `r8` variant whose f32 step is rewritten to the GPU's op order (fmul, fmul, fsub, fma). Count
   mismatches. Zero ⇒ P2's hardest half is a one-line change; non-zero ⇒ find which op (likely the f16→f32 scale
   product rounding) and whether cvt order fixes it.
2. **Qwen3-30B-A3B LRU hit rate and llama.cpp baselines on bella** (needs the GGUF; GPU time, so a scheduled slot):
   `llama-bench -m Qwen3-30B-A3B-Q4_K_M -ot exps=CPU` with and without `--no-mmap`, `-ncmoe {48, 32, 24}`, `--fit` default; then
   `llama-cli --moe-cache-mib 6000` on a 20k agent-style prompt + 256 tokens and read `llama_moe_cache: ubatch <= 8:
   hits/misses` at exit. This gives h for §4.2 C/D directly and the number P1 must beat.
3. **32B split + draft.** `axon gen` 32B with `--draft Qwen2.5-0.5B`, K ∈ {3, 5, 7, 9}, a 300-token code prompt,
   128 greedy tokens, cool-gated: tok/s per K. Confirms or kills P3 before anyone designs anything.
4. **DRAM sharing.** Run the CPU 7B decode while a pinned 4 GB H2D copy loops: if 7B tok/s drops by the copy's
   share of 75 GB/s, §4.5's floor is confirmed and dead end 1 is closed for good.
5. **Barrier cost of the expert gather.** A pool micro-bench: 24 GEMVs of 768×2048 Q4_K per "layer" with (a) one
   barrier per GEMV, (b) two per layer. The difference × 48 is the budget P1's CPU kernel design must respect.
6. **Hop latency.** D2H 8 KB + sync + H2D 8 KB round trip on bella at 48 repetitions: is it the ~10 µs the pipeline
   measured or the 30–40 µs a staged copy costs? Sets the 1–3 ms term in §4.2.
