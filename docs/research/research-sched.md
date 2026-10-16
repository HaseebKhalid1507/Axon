# research: serving concurrent requests / agent swarms — batching, scheduling, fairness, batched speculation

Read-only research note (Chrollo), 2026-10-16. Axon main `e998d6f` read in a detached worktree; the chunked-prefill
branch `feat/serve-chunked-prefill` at `8eae48e` (the builder's worktree and notes); llama.cpp
`4413a19`. Labels: **CODE** (file:line), **MEASURED** (Axon log/spec or a paper's own measurement), **ESTIMATED** (arithmetic shown),
**CLAIMED** (a paper's headline, not reproduced).

---

## 1. The problem in Axon's terms

- S2-lite batches one decode row per running request (CODE engine.rs:181–254, slots.rs). A new request's **whole prompt is a step of its
  own** (engine.rs:120–148 `admit` → `feed_slot`), so every decode row stalls for it: 20k tokens at 2682 tok/s (MEASURED, 7B at 32k
  depth) is a **7 s ITL spike**. Closed-loop `bench_serve.py` at C=8 on a 4090 gave 740 tok/s (4.3×) vs the 5.5× estimate; r10 §4.2
  attributes ~18 % of wall to those whole-prompt steps (ESTIMATED).
- The chunked-prefill branch fixes the stall but as **two passes per iteration**: `prefill()` feeds ≤ 512 tokens via `feed_slot`, then
  `step()` runs the decode rows (CODE axon-wt-chunk engine.rs `step`: `self.prefill()?` then the row step). Weights are read twice per
  iteration; the decode rows do not ride in the chunk's pass. Pod bench (MEASURED, 4090, 7B, 23.9k admit beside 4 decoders): idle ITL
  6.55 ms; c512 ITL-during p50 62.5 / p99 83.9 ms, TTFT 2989 ms (+21 % over 2464 unchunked); c128 31.5 / 45.2 ms, TTFT 5901 ms;
  c32 30.6 / 44.2 ms, TTFT 22.9 s. **The per-chunk cost has a floor (~25 ms 7B at 24k depth)**: few-query `attn_fa_b` over a long KV.
- Drafting (`--draft-model`, K=3: +37 % 7B / +44 % q3-8B / +63 % Llama-8B greedy, MEASURED; +0.8 % sampled) runs **only on the solo
  path** (engine.rs:10–14); batched steps never speculate. Acceptance is exact-match (`sampler pick == draft`), not rejection sampling.
- Fairness: FIFO admission off a bounded channel; oldest prefill first (`pre[0]`); slot choice = longest common prefix, ties to the
  lowest index (engine.rs:117 `Reverse(s)`); a request that fits no slot waits for `b.idle()` (serve.rs:797–810). The served context is
  the slot's (serve.rs:722–724, `Loaded.ctx = served`) and `n_gen = min(max_tokens, ctx − prompt)` (serve.rs:1336–1340), so
  `fits_slot` (serve.rs:423–425) is false **only when `n_gen == 0`** — the starvation path is effectively dead code today, not a live hazard.

---

## 2. Literature map

Read in full (pdftotext): Sarathi-Serve, Sarathi, TurboSpec/SmartSpec, Synergy of SD and batching, VTC, DistServe, MagicDec, Autellix,
DLPM, DeepSpeed-FastGen. Abstract + known method for the rest.

### 2.1 Iteration-level scheduling and chunked prefill

| paper | idea | numbers (hardware) |
|---|---|---|
| **Orca** (Yu et al., OSDI 2022) | Iteration-level scheduling (return/admit per step) + selective batching (batch the GEMMs, not attention). The ancestor of every continuous-batching engine, Axon's S2-lite included. | CLAIMED 36.9× throughput vs FasterTransformer at equal latency, GPT-3 175B on A100s. |
| **Sarathi** (Agrawal et al., arXiv 2308.16369, 2023) | Split prefill into chunks; *piggyback* decode tokens onto a chunk's pass so decodes ride the compute-bound GEMM for ~free. | CLAIMED decode throughput 10× (LLaMA-13B, A6000), 1.33× e2e; pipeline bubbles −6.3×. |
| **Sarathi-Serve** (Agrawal et al., OSDI 2024, arXiv 2403.02310) | **Stall-free batching**: a per-iteration *token budget*; decode rows first, the remainder filled with prefill chunks (Alg. 1). Token budget chosen by TBT SLO vs chunking overhead (§4.3): smaller chunks waste GEMM efficiency and *re-read the KV in attention* per chunk. | MEASURED: Mistral-7B 1×A100 budget 256/512; Yi-34B 2×A100 budget 512/2048; **chunk 512 costs up to ~25 % prefill overhead, 2048 ≈ negligible** (§5.4.1, Fig. 14). CLAIMED capacity 2.6× (Mistral-7B) / 3.7× (Yi-34B) vs vLLM under TBT SLO; 5.6× for Falcon-180B PP. |
| **DeepSpeed-FastGen** (Holmes et al., arXiv 2401.08671, 2024) | **Dynamic SplitFuse**: long prompts split, short prompts *fused* to exactly fill a fixed token budget, so every forward has the same size (a flat latency curve). | CLAIMED 2.3× effective throughput, 2× mean latency, 3.7× lower P95 token latency vs vLLM (A100). |
| **POD-Attention** (Kamath et al., arXiv 2410.18038, 2024; ASPLOS 2025) | One attention kernel that co-schedules prefill (compute-bound) and decode (bandwidth-bound) CTAs on the same SM for hybrid batches. | CLAIMED attention up to 59 % faster (mean 28 %) vs separate kernels. |

### 2.2 Disaggregation

| paper | idea | numbers |
|---|---|---|
| **DistServe** (Zhong et al., OSDI 2024, arXiv 2401.09670) | Prefill and decode on *different GPUs*, each with its own parallelism; KV shipped over NVLink. Motivation: hybrid batching only *mitigates* prefill–decode interference (§2.3). | CLAIMED 7.4× more requests or 12.6× tighter SLO vs vLLM (OPT-13B/66B/175B, A100 NVLink). |
| **Splitwise** (Patel et al., ISCA 2024, arXiv 2311.18677) | Phase splitting across *machines*, possibly heterogeneous (old GPUs for decode). | CLAIMED 1.4× throughput at 20 % lower cost, or 2.35× at same cost/power (A100/H100 clusters). |
| **Mooncake** (Qin et al., FAST 2025, arXiv 2407.00079) | KVCache-centric disaggregation; CPU/DRAM/SSD as a disaggregated KV store; early rejection under overload. | CLAIMED up to 525 % throughput in simulation, 75 % more real requests (Kimi). |
| **semi-PD** (arXiv 2504.19867, 2025) | Disaggregated *computation* on one GPU (SM-level partition of the two phases) with *unified* storage. The only form of "disaggregation" that even applies to one GPU. | CLAIMED 1.27–2.58× lower mean latency (DeepSeek), 1.55–1.72× more requests in SLO (Llama). |

**Why not on one laptop GPU (the asked question).** Disaggregation buys three things: no interference, independent parallelism per
phase, and hardware specialization. On bella none exists: (i) there is one 46-SM device, so "prefill GPU" and "decode GPU" are the same
SMs — semi-PD's SM partition is the only variant, and partitioning 46 SMs leaves decode with too few SMs to pull 648 GB/s (the decode
GEMV already needs most SMs in flight to saturate bandwidth; r5-decode) and prefill with too few to be compute-bound; (ii) the KV
transfer that DistServe hides on NVLink has no counterpart — the KV is already where it must be; (iii) the interference it removes is
exactly what Sarathi's token budget bounds on one GPU. DistServe itself concedes chunked prefill "alleviates" the interference
(§2.3) and argues for disaggregation only where the *cluster* has spare GPUs. r10 §4.6 already lists it as "not a round"; this note agrees
and adds: the one idea worth stealing is semi-PD's *asynchrony*, which on one GPU degenerates to the hybrid step in §4 P1.

### 2.3 Preemption, fairness, QoE, migration

| paper | idea | numbers |
|---|---|---|
| **FastServe** (Wu et al., arXiv 2305.05920, 2023) | Preemption per output token; *skip-join MLFQ* — input length (known) picks the initial queue; KV swapped to host under pressure. | CLAIMED 31.4× / 17.9× throughput at equal mean / tail latency vs vLLM-era Orca. |
| **VTC** (Sheng et al., OSDI 2024, arXiv 2401.00588) | Fairness in continuous batching: per-client *virtual token counter* of service = w_p·input + w_q·output (they use 1:2, OpenAI pricing); admit the client with the smallest counter; *counter lift* for returning clients. | Proven: service gap between backlogged clients ≤ 2·max(w_p·L_input, w_q·M) (Thm 4.4), within 2× of the lower bound. A10G/A100, LLaMA-2-7B/70B. |
| **DLPM — Locality-aware Fair Scheduling** (Cao et al., arXiv 2501.14312, 2025) | VTC ignores prefix locality. **Deficit Longest Prefix Match**: sort waiting requests by matched prefix, admit while the client's deficit counter > 0, replenish a quantum Q_u when all are ≤ 0; charge only *extend* (uncached) tokens. | Proven gap ≤ 2(U + Q_u). CLAIMED up to 2.87× throughput over VTC, 7.18× lower per-client latency distributed (A10). |
| **Andes** (Liu et al., arXiv 2404.16283, 2024) | QoE = did the user receive tokens at ≥ reading pace; preempt requests already ahead of pace, favour those behind. | CLAIMED 4.7× QoE at equal GPUs or −61 % GPUs at equal QoE. |
| **Llumnix** (Sun et al., OSDI 2024, arXiv 2406.03243) | Live migration of requests + KV across *instances* for load balance, defragmentation, priorities. | CLAIMED tail latency ↓ an order of magnitude; 36 % cost. Multi-instance only — no Axon analogue. |

### 2.4 Speculative decoding under batching

| paper | idea | numbers |
|---|---|---|
| **SpecInfer** (Miao et al., ASPLOS 2024, arXiv 2305.09781) | Token *trees* from several small drafters, verified in one tree-attention pass. | CLAIMED 1.5–2.8× distributed, 2.6–3.5× offloading. |
| **Medusa** (Cai et al., ICML 2024, arXiv 2401.10774) | Extra decoding heads on the target; tree attention; needs training. | CLAIMED 2.2× (Medusa-1, lossless), 2.3–3.6× (Medusa-2). |
| **EAGLE / EAGLE-2 / EAGLE-3** (Li et al., ICML 2024 arXiv 2401.15077; EMNLP 2024 arXiv 2406.16858; NeurIPS 2025 arXiv 2503.01840) | A one-layer feature-level drafter on the target's hidden states; EAGLE-2 context-aware dynamic trees; EAGLE-3 direct token prediction with multi-layer feature fusion. All need a trained head per target. | CLAIMED 2.7–3.5× (70B); 3.05–4.26×; up to 6.5×, and **1.38× throughput at batch 64 in SGLang**. |
| **Lookahead decoding** (Fu et al., ICML 2024, arXiv 2402.02057) | Jacobi/n-gram pool, no drafter; trades log(FLOPs) per step for fewer steps. | CLAIMED up to 1.8× MT-bench. Compute-hungry: wrong regime for a 46-SM part. |
| **Synergy of SD and batching** (Su et al., arXiv 2310.18813, 2023) | MEASURED on RTX 3090/4090/A100 (OPT-1.3B/6.7B, Llama-7B): **optimal K shrinks with batch**; at b=32 the best K ≤ 2; at b=8 on a 3090 K=3–4; b=4 OPT-6.7B K=2 (A100) / 3 (4090). Profile once, map b → K. | Fig. 1, Table 1 model T = T_S(b,K) + T_L(b,K+1). |
| **BASS** (Qian et al., arXiv 2404.15778, 2024) | Batched SD with ragged per-sequence draft lengths and attention kernels for it. | MEASURED 7.8B on 1×A100, b=8: 5.8 ms/token/sequence, 1.1k tok/s aggregate, 2.15× over regular decoding; GPU utilization 15.8 %. |
| **TurboSpec / SmartSpec** (Liu et al., arXiv 2406.14066 v3 2025; v1 title "Optimizing Speculative Decoding … Using Goodput") | Closed-loop control: predict *goodput* = accepted tokens / batch latency from a profiled latency model (linear in batched tokens) and the running acceptance rate; pick K per step, **disable SD** when it loses. | MEASURED Llama-2-7B + 160M on H100/L40S: at acceptance 0.5, **batch ≥ 16 makes SD useless on an H100** (§6.1, Fig. 9a); best K 3 at acc 0.5, 7 at acc 0.9 (Fig. 10). Up to ~2× throughput on L40S for low-rate workloads. |
| **MagicDec** (Chen et al., arXiv 2408.11049, 2024; ICLR 2025) | At long context the *KV read*, not the weights, is the decode bottleneck, and it grows with batch; a drafter with a *sparse/small KV* (StreamingLLM or top-k self-speculation) keeps T_draft/T_target small, so SD speeds up *more* at larger batch. | CLAIMED 2.51× Llama-3.1-8B at batch 32–256, 8×H100, 16k–32k inputs. |
| **Nightjar** (arXiv 2512.22420, 2025) | Multi-armed-bandit K per batch size; disables SD under load and *offloads the draft model to CPU* to give its VRAM back to the KV pool. | CLAIMED +14.76 % throughput, −20.18 % latency vs static SD. |
| "An Interpretable Latency Model for Speculative Decoding in LLM Serving" (arXiv 2605.15051, 2026) | Surfaced in search; **not read**, title only. | — |

### 2.5 Agent workloads: prefix sharing, program-level scheduling, tool latency

| paper | idea | numbers |
|---|---|---|
| **SGLang / RadixAttention** (Zheng et al., NeurIPS 2024) | Radix-tree prefix cache shared across requests; cache-aware scheduling (longest-match first). The basis of DLPM and KVFlow. | CLAIMED up to 6.4× throughput on agentic/program workloads. |
| **Hydragen** (Juravsky et al., arXiv 2402.05099, 2024) | Exact attention with a *shared prefix*: prefix attention batched across sequences as a matrix–matrix product, suffix per sequence, combined by log-sum-exp. | CLAIMED up to 32× CodeLlama-13b throughput; prefix 1k→16k costs < 15 % with large batch. |
| **Parrot** (Lin et al., OSDI 2024, arXiv 2405.19888) | *Semantic Variables*: the app exposes its request DAG; the server does dataflow analysis, prefix sharing, batching of dependent calls. | CLAIMED up to an order of magnitude e2e. |
| **Autellix** (Luo et al., arXiv 2502.13965, 2025) | Programs (agents) as first-class: **PLAS** (program-level attained service: the agent that has consumed less cumulative service goes first, MLFQ-style, with anti-starvation) and ATLAS for DAG agents; preemption at call granularity; HoL blocking at both call and program level measured. | CLAIMED 4–15× program throughput at equal latency vs vLLM FCFS. |
| **KVFlow** (arXiv 2507.07400, 2025) | Workflow-aware eviction: each agent gets *steps-to-execution*; evict the cache furthest from reuse, not LRU; prefetch the next agent's KV from CPU in the background. | CLAIMED 1.83× single workflow, 2.19× many concurrent workflows vs SGLang hierarchical cache. |

---

## 3. What production engines do

### 3.1 llama-server (`tools/server/server-context.cpp` at `4413a19`; `server.cpp` is only the entry point)

- **Loop**: `update_slots()` :3158 → `pre_decode()` :3272 builds *one* `batch` per iteration, then :3232–3254 splits it into `n_batch`
  pieces and calls `decode()` :4150 (`llama_decode`, which internally splits again into `n_ubatch` micro-batches = one forward each); a KV
  allocation failure halves `n_batch` and retries (:4233–4238).
- **Composition (Sarathi-style hybrid)**: generating slots' sampled tokens (+ their speculative drafts) go in first via
  `handle_last_sampled_token` (:543, called :3469–3471); then, if `cont_batching` (default true, `common/common.h:573`), prompts fill the
  rest **up to `n_batch`** (:3481–3486; the per-token loop :3956–3966 breaks at `batch.size() < n_batch`). Slots are iterated in index
  order — the lowest-indexed prefilling slot eats the budget first (:3484 `iterate(slots, …)`); a prompt that cannot split waits for a
  batch that fits it whole (:3833–3838).
- **Defaults**: `n_batch 2048`, `n_ubatch 512` (`common/common.h:456–457`). So a 20k prompt is fed 2048 tokens per iteration, in 4
  forwards of 512; the decode rows ride in the first ubatch and then **wait ~3 more forwards** — llama-server's ITL during an admit is
  worse than Axon's chunk-512 branch at defaults (ESTIMATED: ~0.5 s per iteration on bella 7B at ~4000 tok/s). Users lower `-b`.
- **Slot selection** `get_available_slot` :1688–1790: explicit `id_slot`; else LCP similarity `lcp/len > --slot-prompt-similarity`
  (default 0.1, `common.h:703`) :1707–1757; else **LRU** by `t_last_used` :1760–1778. If the chosen slot would lose > 50 % of its context
  (`f_keep < 0.5`) its cache is saved to the host prompt cache (`--cache-ram`) :1744–1748.
- **Queue**: FIFO `queue_tasks`; a task with no free slot is **deferred** and retried after the next step (:2754–2770). No fairness, no
  priority beyond `id_slot`.
- **Speculation per slot**: `n_draft_max` = remaining context/−2 and remaining tokens (:521–540); `draft.n_max` default 3
  (`common.h:329`); `use_spec_rejection()` = temp > 0 (:508–510) → **rejection sampling** at temp > 0 (Axon uses exact match, which is
  why Axon's sampled gain is +0.8 % and llama.cpp's is larger — but Axon's is deterministic per seed with or without a drafter; llama's
  changes the sampled sequence). Drafts for all generating slots are produced in one `common_speculative_draft` call (:3414–3418) and
  the verify rows are simply appended to the same batch (:3469–3471): batched speculation falls out of the hybrid batch for free, with no
  budget policy (every slot drafts n_max regardless of batch size).
- `README.md:589`: `cache_prompt` not bit-reproducible across batch sizes — the contract Axon sells.

### 3.2 vLLM / SGLang / TensorRT-LLM (from papers/docs, not local)

- **vLLM** (V1 engine): chunked prefill on by default with `max_num_batched_tokens` (Sarathi's token budget; docs default 2048 online
  /8192 offline for recent versions), `max_num_seqs`; FCFS with `priority` scheduling optional; preemption by recompute or CPU swap;
  speculative decoding (n-gram, draft model, EAGLE/EAGLE-3, Medusa) with a fixed `num_speculative_tokens` — TurboSpec's dynamic
  control is the paper, not the default; prefix caching by hashed blocks. Rejection sampling at temp > 0.
- **SGLang**: RadixAttention prefix cache; scheduler policy `lpm` (longest prefix match, cache-aware) default, `fcfs`, `dfs-weight`;
  chunked prefill (`chunked_prefill_size`, 8192 default on big GPUs); EAGLE-3 speculation in batches (the 1.38× at batch 64 claim).
- **TensorRT-LLM**: in-flight batching with `max_num_tokens` budget (chunked context), `guaranteed_no_evict` vs `max_utilization`
  scheduling, draft-target and EAGLE speculation, KV reuse by block hash.

Common ground: *every* engine converged on Sarathi's token budget + hybrid batch; none ships fairness by default; speculation is static
and left to the operator.

---

## 4. Proposals for Axon, ranked

Cost basis used throughout (bella, 7B Q4_K_M unless noted). MEASURED: decode step 8.47 ms sustained; nb=8 step 1.45× a forward (dp4a
path, g3 NOTES); prefill 3999 tok/s at 8k depth, 2682 at 32k; C=16 closed-loop 355 tok/s (= 45 ms per 16-row iteration incl. prefill
steps); 0.5B C=16 2978 tok/s. 4090: idle ITL 6.55 ms, chunk-512 at 24k depth ≈ 56 ms over idle, chunk-128 ≈ 25 ms over idle (the
floor). ESTIMATED where marked.

### P1. Hybrid step: the prefill chunk and the decode rows in **one pass** (Sarathi piggybacking, Axon-style)

**Change.** Today's branch: `feed_slot(chunk)` then `step(rows)` — two passes. Make `BatchLm::step` accept rows of a prefilling slot
alongside the decode rows: `Place::Rows` already allows "a sequence's rows of a pass in ascending positions" (CODE batch.rs:142–153;
`step` doc lm.rs:171–173), and the kernels are token-major except attention. The one real change is in `pass_at` (batch.rs:1147–1215):
when the pass is `Rows` and contains a run of ≥ `att_per_query(hd)` rows of **one slot at consecutive positions**, launch `attn_fa_b`
for that run (it takes `start, nb` and a base — the `Place::Seq` path already does, with `base` = slot × ctx) and `attn_blk_b` for the
remaining decode rows. Two launches, same kernels as today. Host side: `prefill()` builds `SeqRow`s with `RowHead::None` (no logits) for
the chunk's tokens except its last when it completes the prompt; the head runs only on the decode rows + completed prompts (HEAD_SUB
slices already skip rows without a head? — if not, pad the head to the rows that want one; `argmax_b`/`samp_tail` per 16-row slice,
batch.rs "HEAD_SUB").

**Gain.** Saves one weight read + one set of per-row decode attention per iteration during an admit: ESTIMATED bella 7B, chunk 256:
separate passes ≈ 8.5 (decode, C≤8 ~10–12) + 64 ms (256/4000) ≈ 75 ms → fused ≈ 66 ms (the decode rows' GEMV disappears into the
chunk's GEMM, which is compute-bound at nb ≥ 128); **−12 % iteration time, −0.4 ms/decode-row**. At chunk 128: 8.5 + 32 → ~34 ms,
−20 %. It is also what makes P2's small budgets affordable, and it brings the batch's GEMMs to the `gemm2` tensor-core path (nb > 16)
for the decode rows' share for free. Aggregate at C=8 closed-loop: the ~18 % prefill share (r10 §4.2) becomes ~12–14 % (ESTIMATED).

**Bit identity.** Preserved. GEMM: `gemvm` (nb ≤ 16) and `gemm2` (nb > 16) are pinned "same bits either way" (bgemm test; r4 §6.1), so
decode rows computed inside a 300-row pass equal their 8-row pass. Attention: the FA kernel for the chunk rows is the same kernel a
`feed_slot` of that chunk runs; the decode rows take `attn_blk_b` exactly as today; both are pinned equal to each other (r4 §6.2, the
`feed_batch_matches_feed` family). KV writes are per row. Risk: *none numeric*; the test is `serve_chunked_prefill_matches_solo` with the
fused pass, plus `step` rows == `feed_slot` + `step` bytes.

**Risk.** Medium engineering: `pass_at`'s attention branch gains a split; the head must skip chunk rows; graphs are keyed by (rows,
greedy, sampled) — a hybrid pass has a new shape per (chunk length, rows), so **do not capture graphs for hybrid passes** (chunk passes
are 30–130 ms; launch overhead ~1 ms is noise). Builder-hours **20–30** on top of the chunk branch.

**Gate.** (a) bytes: a slot prefilled by hybrid passes with random chunk sizes 16..512 while 1..7 other rows decode == `feed_slot` whole,
KV-byte exact, tokens exact; (b) bella 7B, 4 decoders + one 20k admit, chunk 256: ITL-during p50 ≤ 1.15× (chunk-pass time alone),
i.e. the decode rows cost ≤ 15 % of the iteration; aggregate C=8 closed-loop ≥ +8 % over the two-pass branch.

### P2. Token-budget scheduler with a load-adaptive chunk (the answer to "what chunk size and policy")

**Change** (`engine.rs` only, after P1). Per iteration a budget `B` tokens (`--prefill-chunk` becomes `--step-budget`): decode rows first,
`B − rows` to prefills. Pick `B` by what the iteration is for:
- **decode rows present**: `B = B_itl` (default **256 on bella, 512 on 4090-class**; see numbers below);
- **no decode rows** (nothing to stall): `B = 4096` (= `STREAM_BATCH`; the whole prompt if it fits) — the solo-admit case keeps today's
  TTFT; the branch's c128/c32 TTFT explosions (5.9 s / 22.9 s MEASURED) come from chunking with nobody to protect;
- **prefill choice**: shortest-remaining-first among prefills (`last − fed`), with an age cap (a prefill older than `max_wait` goes first).
  This is FastServe's skip-join insight — input length is known — applied where it matters for agents: a 300-token tool-result turn
  (its 20k prefix cached) slips ahead of another agent's 20k cold prompt instead of queuing behind 40 of its chunks. Bound: with 256
  budget the 20k prompt is delayed by at most the sum of the short ones, each ≤ 1 iteration.

**Why 256 on bella** (ESTIMATED from MEASURED rates). ITL-during = hybrid iteration ≈ chunk/pp_rate + attention floor:
| budget | 7B at 8k depth (4000 tok/s) | 7B at 24k–32k depth (~2700 tok/s + ~25 ms floor scaled to bella ≈ 35 ms) | prefill passes for 20k |
|---|---|---|---|
| 512 | ~130 ms | ~225 ms | 40 |
| 256 | ~65 ms | ~130 ms | 79 |
| 128 | ~35 ms | ~80 ms (floor-dominated) | 157 |
Decode p50 at C=4 ≈ 10 ms, so the C1 gate "p99 ≤ 2× p50" is unreachable at 7B with any chunk the GEMM can use (it needs ≤ ~32 tokens,
which is the 22.9 s-TTFT regime). Replace it with an absolute bound: **ITL p99 ≤ 100 ms during an admit** (Andes' "reading pace" is
~5–10 tok/s; 100 ms keeps streaming smooth) — 256 meets it at 8k depth, borderline at 32k until the few-query long-KV attention
(feat/long-prefill-attn, sa_174 area) lowers the floor; 512 fails it at 7B on bella but is right on a 4090 (56 ms MEASURED). Sarathi-Serve
picked 256 for Mistral-7B on an A100 under a strict TBT SLO (§5.2) — same order, 3× the SMs.

**Throughput side.** Fewer, larger chunks win: a 20k prompt at 256 costs 79 iterations each paying the attention floor and GEMM
tile underfill (gemm2 tiles are 64/128 wide, r4 §4.5); Sarathi measured ≤ 25 % prefill overhead at 512 vs ≈ 0 at 2048 (34B, A100). On
bella at 7B the GEMM is compute-bound from ~128 tokens, so 256 → 512 is ESTIMATED ≤ 10 % prefill-throughput difference; the adaptive
rule (4096 when alone) recovers the rest. Net: **256 bounds p99 ITL at ≤ ~10 % aggregate cost vs 512; 128 costs ~25 % and buys little
until the attention floor moves.**

**Bits:** none (chunking is invisible by contract). **Hours:** 8–12. **Gate:** `bench_serve_admit.py` on bella: ITL p99 ≤ 100 ms during a
20k admit beside 4 decoders at 8k; TTFT of a solo 20k admit == unchunked ± 3 %; shortest-first: a 300-token turn admitted during another
agent's 20k admit gets TTFT ≤ 0.5 s (vs ≥ 5 s FIFO).

### P3. Fairness for agent swarms: deficit counters over *uncached* work, LRU eviction, honest refusal

**Policy** (the answer to "what fairness policy fits 20k prompts then short turns"): **DLPM-lite**.
- Client key: request `user` field, else API key (M2), else peer IP. (A swarm from one harness is one client unless it sets `user` —
  document that.)
- Service charged = `w_p × fed_tokens + w_q × generated`, with `fed` = tokens actually prefilled (the cached prefix is free, as DLPM
  charges *extend* tokens — VTC would bill an agent 20k again every turn for a prefix that costs Axon nothing). w_p : w_q = 1 : 2 (VTC's
  choice) is wrong for this box: a prefill token costs ~0.25 ms, a decode token ~8.5 ms/rows ≈ 1–4 ms → **w_p : w_q ≈ 1 : 8** by
  measured cost (ESTIMATED; make it `--fair-weights`).
- Admission: among waiting requests that fit, prefer the client with the largest remaining deficit; when all clients ≤ 0 replenish a
  quantum Q (e.g. 4k tokens-equivalent). Within a client, FIFO. Combined with P2's shortest-first for *prefill order*, a 20k cold prompt
  cannot monopolize the budget: it drains its client's deficit and other clients' short turns go first until the quantum resets.
- Eviction (M3c): when all free slots hold live prefixes, evict the slot with the oldest `last_used` *and the smallest held prefix*
  (cheapest to rebuild) — not the lowest index (engine.rs:117). With M1's host store the eviction cost drops to ~0.1 s, and KVFlow's
  point stands: for a swarm the right eviction key is "furthest from reuse", which the server can approximate by per-client turn cadence.
- The oversize path (serve.rs:797–810): since `fits_slot` is false only for `n_gen == 0`, delete the path and refuse `max_tokens: 0`
  with a 400 (or return an empty completion); the honest 400 for prompts past `n_ctx` already exists (serve.rs:1329–1331). Nothing
  to age.

**Bits:** none. **Hours:** 10–16 (counters, keys, a `/slots` view of deficits). **Gate:** `bench_serve.py` scenario: 8 agents, one of
them firing 20k prompts back-to-back, others 300-token turns on cached prefixes: the short turns' TTFT p99 ≤ 3× their solo TTFT (vs
unbounded under FIFO); the heavy agent still completes (no starvation: its deficit is replenished every quantum).

### P4. Batched speculative decoding — only with a budget, only greedy rows, only after the per-row costs are measured

**When it pays (the answer for C=2–8).** Model one round at C rows, K drafts each, acceptance p per token (Axon's solo gain at K=3 implies
p ≈ 0.6 on 7B: 2.37 tokens/round = 1 + p + p² + p³; ESTIMATED):
- cost(round) = T_draft(C, K) + T_verify(C × (K+1)); tokens(round) = C × (1 + p + … + p^K).
- no-spec cost per token = T_step(C) / C.
MEASURED anchors on bella: T_step(1) = 8.5 ms, T_step(8) ≈ 12.3 ms (1.45×); T_step(16) is **not measured** decode-only (C=16 closed-loop
= 45 ms/iteration includes prefill steps; §4.2 of r10 hints ~1.85× at 8 on a 4090 and worse per row at 16). Two cases:
| C, K | verify rows | verify (ESTIMATED) | draft (0.5B batched, ESTIMATED) | ms/token spec | ms/token no-spec | verdict |
|---|---|---|---|---|---|---|
| 2, 3 | 8 | 12.3 ms (MEASURED ratio) | 3 × ~1.6 + 1.5 ≈ 6.3 ms | 18.6 / 4.74 = **3.9** | 9.5 / 2 = 4.75 | **+20 %** |
| 4, 3 | 16 | 17 ms if nb16 ≈ 2.0× (unknown) | 3 × ~2 + 1.5 ≈ 7.5 | 24.5 / 9.5 = **2.6** | 10.5 / 4 = 2.6 | break-even |
| 4, 1 | 8 | 12.3 | ~1.6 + 1.5 ≈ 3.1 | 15.4 / 6.4 = **2.4** | 2.6 | +8 % |
| 8, 1 | 16 | 17 (if 2.0×) | ~3.6 | 20.6 / 12.8 = 1.6 | 12.3 / 8 = 1.54 | ≈ 0 |
| 8, ≥2 | ≥ 24 | past `SMALL` → `gemm2`, cost ~linear | — | — | — | **loses** |
So: **pays at C ≤ 2–3 with K=3, marginal at C=4 (K=1–2), off at C ≥ 8** — the same shape as Synergy (Fig. 1: optimal K falls to ≤ 2 by
b=32 on a 3090, and bella has less compute per GB/s than a 3090's 936 GB/s / 82 SMs) and TurboSpec (SD off at batch ≥ 16 on an H100 at
acc 0.5; bella's crossover is earlier because its verify rows stop being free sooner). r10 C4's rule "rows × (K+1) ≤ 16" is the right
*shape*; the numbers above say the threshold should be chosen from a measured T_step(nb) curve, not assumed free up to 16.

Two Axon-specific caveats the papers do not have:
1. **Sampled rows gain nothing** (+0.8 % MEASURED at temp 0.8) because acceptance is exact-match, not rejection sampling — and
   rejection sampling would break "same tokens with or without a drafter" (it preserves the distribution, not the sequence). So draft
   **only rows with temp 0 / top-k 1**; agents that sample are plain decode rows. This also shrinks the budget problem: at C=8 with 3
   greedy rows, 3 × (K+1) = 12 ≤ 16 fits.
2. **Long context kills it in the current kernel**: in `Place::Rows` every row is a separate `attn_blk_b` walk of its slot's KV
   (batch.rs:1195–1215). At 32k a row's attention is 5.3 ms (MEASURED), so K+1 verify rows of one slot cost (K+1) × 5.3 ms of attention
   — a 7B at 32k with K=3 pays 21 ms of attention per sequence per round against an 8.5 ms GEMV: SD loses at any C. MagicDec's lesson
   inverted: the fix is a **multi-query decode attention** — the K+1 (≤ 16) queries of one slot share one KV walk (load a 64-position
   block once, run each query's mma chain on it). Same per-(query, head) order ⇒ same bits. This is the same kernel the few-query
   prefill floor (P2) and `feat/long-ctx-decode` want; it is the real prerequisite for C4 on agents' contexts.

**Change.** A `BatchLm` pool for the drafter (a 0.5B pool of C slots — its ctx must track the target's; ~56 KiB/tok × 0.14 for 0.5B);
`Batch::step` becomes: (a) drafter batched step for the greedy rows under budget (one 0.5B `step` with K sequential rows — K launches of
a C-row step, ~1.6–2 ms each ESTIMATED from 0.5B C=16 at 5.4 ms/step); (b) one target step with 1 + k_r rows per sequence (ascending
positions in the same slot, as `Place::Rows` permits); (c) per row accept the longest matching prefix exactly as `sample::generate` does;
rejected rows' KV is overwritten next step (positions absolute). Budget: `Σ_r (1 + k_r) ≤ N*` where N* is the largest nb whose
measured T_step(nb)/T_step(1) ≤ 1.5 (bella today: 8); distribute k_r round-robin among greedy rows with acceptance ≥ 0.5 over their last
32 drafts (TurboSpec's per-request acceptance tracking, FastServe-style no starvation of the budget).

**Bits:** none — verify rows == decode rows is the standing contract; drafting changes which rows run, never their values. **Hours:**
30–50 (+ the multi-query attention kernel, 20–30, if long contexts are the target). **Gate:** tokens per request identical with and
without `--draft-model` under C=4 mixed load; bella 7B greedy code prompts: C=2 ≥ +20 % per stream, C=4 ≥ +5 %, C=8 aggregate within
2 % of no-spec (the policy turned itself off). Kill if `pool_step_cost` (slots.rs:880, `#[ignore]`) shows nb=16 ≥ 2.5× nb=1 on bella:
then only C ≤ 2 ever pays and C4 is not a round.

### P5. Shared-prefix attention for the swarm (Hydragen-style), after C5

Once C5 (shared-prefix pages) lands, the 12 agents' 20k shared system prompt is stored once but still *read* 12 times per decode step
(12 × 20k × 56 KiB = 13 GiB of KV traffic per step ≈ 20 ms at 648 GB/s — more than the whole GEMV). Hydragen batches the sequences'
queries against the shared prefix in one pass (one read). In Axon's terms: `attn_blk_b` over the shared pages with the 12 slots' queries
as a multi-query group (P4's kernel again), then each slot's private suffix, then the canonical fold. **Bits:** the per-(query, head)
block order is the prefix blocks then the suffix blocks, which is the order a solo sequence walks — same bits if the fold is the same
chain (it is the existing `attn_cmb_b` fold over block partials). ESTIMATED gain for the 12-agent swarm at 20k shared + 5k private:
attention per step 12 × 25k → 20k + 12 × 5k = 80k block-rows (−68 %), step ≈ 8.5 + 4.5 vs 8.5 + 16 ms → **~1.9× aggregate** at C=12.
Hours 30–40; gate: identity with shared vs private pages; swarm bench ITL p50.

---

## 5. Dead ends

- **Prefill/decode disaggregation on one GPU** (DistServe/Splitwise/Mooncake): nothing to disaggregate onto (§2.2). semi-PD's SM partition
  on 46 SMs starves both phases; the hybrid pass (P1) delivers the asynchrony it seeks.
- **Rejection-sampling verification** (llama.cpp at temp > 0, vLLM, EAGLE): distribution-preserving, not sequence-preserving — breaks
  "with drafter == without" and "deterministic run to run" for sampled requests. Acceptable only as an explicit opt-in mode; the win is
  real (llama.cpp's sampled SD gains), but Axon's contract is the product.
- **EAGLE/Medusa heads**: need a trained head per target model; Axon serves arbitrary GGUFs. Only if a community head exists for the
  exact target (none for Qwen2.5-7B-Instruct GGUF users in practice). Lookahead decoding: compute-hungry (log FLOPs per step) on a
  46-SM part whose decode is bandwidth-bound — the Synergy/TurboSpec regime says no.
- **Preemption by KV swap to host for latency** (FastServe, vLLM `--swap-space`): 1.75 GiB per 32k 7B slot at 25 GB/s = 70 ms each way;
  as a *latency* tool at ≤ 8 slots it loses to simply not stalling (P1/P2). As a *capacity* tool it is M1 (eviction to host), already planned.
- **Per-token QoE preemption (Andes)**: pauses requests already "ahead of reading pace" to favour those behind. Agents do not read;
  they wait for the full turn. Wrong objective for the swarm; keep for a human-chat preset at most.
- **Llumnix-style migration**: needs ≥ 2 instances.
- **llama-server's default `-b 2048`** as a model: it is the thing Axon beats on ITL during admits; do not copy the 2048 budget when rows
  are decoding.
- **The "oversize request runs alone when idle" path**: dead code (`fits_slot` false only for `n_gen == 0`); delete rather than fix (P3).
- **C1's gate "ITL p99 ≤ 2× p50"**: not attainable at 7B on bella with any GEMM-efficient chunk; replace with an absolute bound (P2).

---

## 6. Open questions a 1-hour experiment settles

1. **T_step(nb) on bella, decode-only** — run `pool_step_cost` (slots.rs:880, `AXON_STEP_MODEL=<7B>`, `--ignored --nocapture`): ms/step
   at nb ∈ {1, 2, 4, 8, 16}, graphs on/off. It fixes P4's budget threshold N*, confirms or kills C4, and is C6's table.
2. **Hybrid vs two-pass iteration time** without writing P1: time `feed_slot(256)` alone, `step(4 rows)` alone, and a `feed_batch(260)`
   on a plain cache at the same depth (the fused pass's cost proxy, since GEMMs are token-major). The difference is P1's gain.
3. **Few-query attention floor on bella**: `feed_slot` of 32 / 128 / 512 tokens at depths 8k / 24k / 32k on 7B; the depth-dependent
   constant is the chunk floor and tells whether 256 or 512 is the right bella default before the long-KV attention work lands.
4. **Acceptance rate of 0.5B→7B on agent traffic**: run `gen --draft-model` on a 2k-token tool-call transcript continuation (JSON
   tool args, code edits) vs the prose bench; the `accepted/drafted` stat. Agent output (JSON, repeated identifiers) is where p ≥ 0.7
   is plausible and where P4's table shifts toward paying at C=4.
5. **Does `attn_fa_b` accept a slot base in `Place::Rows`?** Read `fa_shape`/`attn_fa_b` args (batch.rs:1160–1180 pass `start` and
   kv pointers from `kv_args_at`) — if the kv pointers can already be a slot's (as `feed_slot` uses), P1's attention split is a
   dispatch change, not a kernel change; 30 minutes of reading settles P1's hour estimate.
