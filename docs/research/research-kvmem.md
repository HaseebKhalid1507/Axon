# research: KV-cache memory for many concurrent sessions

Read-only research note (Chrollo), 2026-10-16. Axon read at `e998d6f` (a detached worktree),
branch `feat/long-ctx-decode` at `fb4ed4a` checked for attention-tile changes (none: `ATT_BLK` is still 64),
PR #42 via `gh pr view`, llama.cpp `4413a19` (`tools/server`, `src/llama-kv-cache.cpp`, `src/llama-context.cpp`,
`common/arg.cpp`). Papers read from arXiv PDFs in a scratch directory (not in the repo) (full text where marked
**READ**, abstract + key tables otherwise). Labels: **CODE** (file:line), **MEASURED** (paper or Axon log/spec),
**ESTIMATED** (arithmetic shown), **CLAIMED** (a paper's claim, not reproduced).

---

## 1. The problem in Axon's terms

- A pool is one `CudaCache` of `slots × ctx` rows per layer array (CODE slots.rs:5–7, 70–84; `kv_args_at`
  qwen2.rs:1344–1350). A slot **reserves** `ctx` positions whether it uses 300 or 32k. Beside the pool the solo
  path keeps its own full cache (`Engine.cache = net.new_cache(ctx)`, CODE serve.rs:681) — **N+1 caches**.
- MEASURED (PR #42 e2e, bella 11.2 GiB free): 7B Q4_K_M f16 KV at 32768 → "one request at a time (2 slots don't
  fit)"; `--kv q8` → 3 slots. ESTIMATED why: 4.36 GiB weights + 1.02 q6k8 copy + 1.75 (+1 cache) + 2 × 1.75 slots
  + ~0.75 scratch/headroom = 11.6 > 11.2; drop the +1 → 9.9 → 2 slots; drop the q6k8 copy too → 3.
- Eviction is destructive: the free slot with the longest common prefix wins (CODE engine.rs:104–118); a loser's
  prefix is overwritten (`held[slot] = p[..last]`, engine.rs:129–134). No host copy, no file, no save on exit.
  A 12-agent swarm on 1–3 slots re-prefills 15–30k tokens per turn: 5–12 s at the MEASURED 2682 tok/s (r9 §0).
- A 14B at `--ctx 32768` becomes a CPU/GPU split for a 3.2 GiB q8 reservation while agents use 10–25k:
  decode 17 tok/s MEASURED vs ~55 all-GPU ESTIMATED (r10 §1.2).
- What Axon has that nobody else does: cache bytes are a pure function of the token prefix (prefill.cu header,
  decode.cu "Tensor-core attention", r4 §6). Every byte-copying scheme below is therefore **exact** — a restored
  or paged slot continues bit for bit — and llama.cpp's own README says its `cache_prompt` is not
  (tools/server/README.md:589).

---

## 2. Literature map

**Paged / virtual memory**

1. **Efficient Memory Management for LLM Serving with PagedAttention** (vLLM), SOSP 2023, arXiv 2309.06180 —
   **READ**. Fixed-size KV blocks + per-sequence block tables; copy-on-write sharing; preemption by swap or
   recompute. MEASURED (A100, OPT-13B/66B/175B, LLaMA-13B): 2–4× throughput over Orca-class systems; prior
   systems waste 60–80 % of KV memory to reservation + fragmentation (Fig. 2). Block size: 16 default; 16–128
   all good on ShareGPT, large blocks hurt on short-prompt Alpaca (§7.2). §7.3: swapping is slow at small blocks
   (many small PCIe copies); "recomputation overhead is never higher than 20 % of swapping's latency"; blocks
   16–64 make the two comparable. The paper's own paged kernel was 20–26 % slower than FasterTransformer's.
2. **vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention**, ASPLOS 2025,
   arXiv 2405.04437 — **READ**. Keep KV *virtually contiguous*; back it on demand with CUDA VMM
   (`cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` / `cuMemSetAccess`) so unmodified attention kernels run.
   MEASURED (A100): paged FA2/FlashInfer prefill kernels up to 37 % / 42 % slower than non-paged, paged decode up
   to 12 %, 7–13 % more instructions; stock CUDA maps 2 MB pages only (they patched the driver for 64 KB);
   API latency per 2 MB page: `cuMemCreate` 29 µs, `cuMemMap` 2 µs, `cuMemSetAccess` 38 µs (Table 3); mapping
   hidden by doing it one iteration ahead. Prefill up to 1.97× FA2-paged at the same memory.
3. **FlashInfer**, arXiv 2501.01005 (MLSys 2025) — abstract only. Block-sparse/paged KV format behind FA-style
   kernels; the reference for "paged without the penalty". Not needed for Axon's conclusions.

**Prefix caching**

4. **SGLang: Efficient Execution of Structured LM Programs** (RadixAttention), NeurIPS 2024, arXiv 2312.07104 —
   **READ** §3. Radix tree over token sequences → KV, token-granularity pages, LRU on leaves with refcounts, cache
   and running requests share one pool; cache-aware (DFS-order) scheduling reaches 96 % of the optimal hit rate
   on average. MEASURED (A10G/A100): up to 6.4× throughput, hit rates 50–99 % on few-shot/multi-turn benches.
5. **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**, FAST 2025 / ACM ToS 2025,
   arXiv 2407.00079 — **READ** §2–4. Hash-chained block keys (`PrefixHash(tokens, B)`), DRAM/SSD tiers, RDMA
   transfer engine. Kimi trace: avg input 8,590 tokens, output 182; LRU beats LFU; hit rate 34 % → 55 % from
   1k to 50k blocks then flat; >50 % of blocks never reused, a few hit tens of thousands of times. CLAIMED
   up to 525 % throughput under SLO; 75 % more requests in production.
6. **KVCache Cache in the Wild** (Alibaba), arXiv 2506.02634 — abstract. Production characterisation: reuse is
   skewed; single-turn reuse as important as multi-turn; per-category reuse time is predictable. Supports an
   LRU-by-last-use store sized by RAM, not by slot count.
7. **vLLM v1 Automatic Prefix Caching** (docs, `docs/design/prefix_caching.md`) — read. Hash(parent hash, block
   tokens, extra) per *full* block; sha256 default since v0.11 (collisions); free queue is LRU with refcounts;
   `cache_salt` per request for multi-tenant isolation.
8. **TensorRT-LLM KV cache system** (docs `features/kvcache.md`) — read. Blocks pack all layers; power-of-two
   tokens per block; radix search tree; prioritised LRU with per-token-range retention priorities; evicted
   primary blocks copied to secondary (host) memory and stay in the tree.

**Offload / save-restore / transfer**

9. **CachedAttention / AttentionStore: Cost-Efficient LLM Serving for Multi-turn Conversations**, USENIX ATC
   2024, arXiv 2403.19708 — **READ** §2–3. Host-RAM + disk store of per-session KV; **layer-wise pre-loading**
   overlapped with the previous layer's compute; async saving; scheduler-aware prefetch; decoupled positional
   encoding (saves pre-RoPE K so truncated histories can be re-RoPE'd — lossy relative to a fresh prefill).
   MEASURED (4×A100, LLaMA-65B): prefill 2K = 360 ms vs loading its 5 GB KV from host = 192 ms over PCIe Gen4
   x16 at ~26 GB/s; TTFT −87 %, prefill throughput ×7.8, cost −70 % CLAIMED.
10. **InfiniGen: Efficient Generative Inference with Dynamic KV Cache Management**, OSDI 2024, arXiv 2406.19707 —
    abstract + §1. Offload-based decode: speculate next layer's important keys with partial weights, prefetch
    only those. CLAIMED up to 3.0× over FlexGen-style offload. Lossy (top-k keys).
11. **CacheGen: KV Cache Compression and Streaming**, SIGCOMM 2024, arXiv 2310.07240 — **READ** §1–2, §5.
    Delta-encode KV across adjacent tokens, layer-sensitive quantisation, arithmetic coding; stream with adaptive
    bitrate. MEASURED: 3.5–4.3× smaller than 8-bit quant at equal quality; TTFT 3.2–3.7× lower vs quant
    baseline; vs 8-bit "nearly lossless" quant still 1.67–1.81× faster load. Lossy by design.
12. **LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference**, arXiv 2510.09665 — **READ**
    §2–5. Chunk-granular (256-token) store/load through a staging GPU buffer, layer-wise async loading, tiered
    CPU/disk/remote. Table 1 (RCCL): 64 KB messages 4 GB/s, 1 MB 30 GB/s, 16 MB+ 49 GB/s — **transfer
    granularity decides bandwidth**. torch.save-style paths "sub-1 GB/s". CLAIMED up to 15× throughput with vLLM.
13. **Comparative Characterization of KV Cache Management Strategies** (vLLM/InfiniGen/…), arXiv 2604.05012;
    **eLLM** (elastic runtime + KV memory), arXiv 2506.15155; **SwiftCache** (multi-turn reload cost over
    NVLink peers), arXiv 2606.16135 — abstracts only; 2025–26 confirmations that reload latency, not capacity,
    is the multi-turn bottleneck once KV leaves HBM.

**KV quantisation**

14. **KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization**, ICML 2024, arXiv 2402.02750 — **READ** §3–4.
    K per-channel, V per-token, group 32, full-precision residual window R=128. MEASURED (fake quant, Llama-2-13B,
    Table 1): 16-bit CoQA 66.37 / TruthfulQA 29.53; **4-bit per-token K and V: 66.48 / 29.51 (no loss)**; 2-bit
    per-token both: 52.93 / 24.98; 2-bit K-channel V-token: 63.53 / 28.60. Peak memory 2.6× lower, throughput
    2.35–3.47× (A100, batch-size driven).
15. **KVQuant: Towards 10 Million Context Length LLM Inference**, NeurIPS 2024, arXiv 2401.18079 — **READ** §3–4.
    Pre-RoPE per-channel K, per-token V, non-uniform codebooks, 1 % outliers sparse. MEASURED Wikitext-2 PPL
    LLaMA-7B (Table 1, baseline 5.68): **int4 per-token no groups 5.98 (+0.30)**; ATOM-4bit g128 5.77 (+0.09);
    **FlexGen-4bit g64 5.73 (+0.05)**; KVQuant-4bit 5.72; 3-bit uniform 10.87; KVQuant-3bit-1 % 5.75; int2 11779.
    On 13B/30B/65B the same ordering, gaps shrink with size.
16. **QServe: W4A8KV4 Quantization and System Co-design**, arXiv 2405.04532 (MLSys 2025) — **READ** §3–4, Fig. 16.
    SmoothAttention: scale K outlier channels by λ_i = max|K_i|^0.5 (RoPE-pair constrained), fold into W_q/W_k.
    Ablation on Llama-2-7B Wikitext-2 (FP16 5.47): W4A8KV8 after clip 5.66 → **+KV4 5.80 (+0.14)** →
    +SmoothAttention 5.75 (+0.09 net for KV4). KV4 is per-head per-token asymmetric.
17. **Atom: Low-bit Quantization for Efficient and Accurate LLM Serving**, MLSys 2024, arXiv 2310.19102 —
    **READ** §5.4. Ablation (LLaMA-7B Wikitext-2): "+ quantizing KV cache to INT4 (g128, asymmetric)
    6.04 → 6.16 (**+0.12**)".
18. **KV Cache is 1 Bit Per Channel: Coupled Quantization**, NeurIPS 2024, arXiv 2405.03917 — §3. Learned
    multi-channel centroids (CQ-4c8b etc.), Fisher-guided; 1 bit/channel with modest PPL loss CLAIMED. Needs
    offline calibration per model and a codebook lookup in the kernel.
19. **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** (Google), arXiv 2504.19874 —
    **READ** §1–2. Random rotation + per-coordinate MSE quantiser + 1-bit QJL on the residual; CLAIMED quality
    neutrality at 3.5 bits/channel, >5× KV reduction, perfect needle-in-haystack. Data-oblivious, no calibration.
20. **llama.cpp PR #21038 "rotate activations for better quantization"** (ggerganov, merged 2026-04-01) — read
    the PR body and the resulting code (`src/llama-kv-cache.cpp:315–360`: `attn_rot_k/v`, 64×64 normalised
    Hadamard, on for quantised KV types with head_dim % 64 == 0, `LLAMA_ATTN_ROT_DISABLE`). MEASURED
    (`llama-perplexity`, 128 chunks): **Qwen3-8B f16 KV 7.3203; q8_0 7.3172 → 7.3195; q5_0 7.3793 → 7.3323;
    q4_0 7.6451 (+0.32) → 7.5012 (+0.18)**. Qwen3-0.6B q4_0: 62.0 → 46.3 vs 13.67 f16 (catastrophic either way);
    q5_1 61.7 → 14.1. Gemma3-4B q4_0 7.8535 → 7.7928 vs 7.6905. Note: llama.cpp's q4_0 KV on Qwen is bad
    *without* outlier handling — Qwen's giant near-constant K channels (decode.cu:1052–1057 explains the same
    thing for Axon's q8 and fixes it with a key shift).
21. **InnerQ**, arXiv 2602.23200 — abstract. Group along the inner (head) dimension so dequant aligns with the
    dot product; tuning-free. Matches what Axon's q8 already does (group = head_dim or 32 along the row).

**Lossy eviction / compression (opt-in only)**

22. **H2O: Heavy-Hitter Oracle**, NeurIPS 2023, arXiv 2306.14048 — §1–3. Keep recent + top accumulated-attention
    tokens, 20 % budget; CLAIMED 29× throughput vs DeepSpeed/HF, 3× vs FlexGen (OPT-6.7B/30B). Needs attention
    probabilities exported every step.
23. **SnapKV: LLM Knows What You are Looking for Before Generation**, NeurIPS 2024, arXiv 2404.14469 — §1, §4.
    After prefill, keep per-head top prefix positions chosen by the last-window attention (pooled); CLAIMED 3.6×
    decode speed, 8.2× memory at 16k, "negligible" NIAH loss. The prompt's KV is altered: output ≠ full attention.
24. **Efficient Streaming LMs with Attention Sinks** (StreamingLLM), ICLR 2024, arXiv 2309.17453 — §1–3. Keep 4
    sink tokens + a sliding window; positions assigned *within the cache*, not by original index; CLAIMED 22.2×
    over sliding-window recompute at stable PPL. Breaks "cache == fresh prefill" by construction.
25. **ShadowKV**, arXiv 2410.21465 — title/abstract only: low-rank K on GPU, V on CPU, sparse retrieval. Lossy
    decode-time retrieval, same family as InfiniGen/Quest.

---

## 3. What production engines do

### llama.cpp (`4413a19`)

- **Slots and the KV pool.** `-np N` slots; `--kv-unified` (`-kvu`, arg.cpp:1727–1733) makes one cell array
  shared by all sequences (`n_stream = unified ? 1 : n_seq_max`, llama-kv-cache.cpp:84); then any one sequence
  may use the whole pool (`n_ctx_seq = n_ctx`, llama-context.cpp:379–383) — otherwise each stream is
  `n_ctx / n_seq_max` cells padded to 256, a fixed reservation like Axon's. `-np -1` (auto) → 4 slots + unified
  (server.cpp:166–173). `--kv-unified-per-slot N` caps a slot inside the unified pool and, with no `-c`, sizes the
  pool to `n_parallel × N` (arg.cpp:1653–1658; server.cpp:176–186; server-context.cpp:1356–1372, 4551–4557).
  Cost of unified: attention runs over `n_kv = pad(used_max_p1, 256)` of the *whole* pool with a seq mask
  (llama-kv-cache.cpp:1278–1290) — a sequence pays for everyone's live cells. Axon's design (addresses per slot,
  attention over its own `np`) avoids that.
- **Slot selection and reuse.** `--slot-prompt-similarity` LCP fraction else LRU (server-context.cpp:1708–1775);
  `cache_prompt` keeps `n_past = LCP(slot.prompt, new prompt)` (3602–3604); `--cache-reuse N` additionally shifts
  matching chunks ≥ N tokens by a RoPE delta (`seq_rm` + `seq_add`, 3633–3679) — **not** a fresh-prefill cache.
  Context shift (`--ctx-shift`, 3277–3315) discards `n_left/2` after `n_keep` and shifts.
- **Host-RAM store, `--cache-ram N` (MiB, default 8192).** `server_prompt_cache` (server-task.cpp:1692–1900):
  whole-slot state blobs (`llama_state_seq_get_data_ext`, server-context.cpp:308–332) saved when a slot is about
  to lose > 50 % of its context (`f_keep < 0.5`, 1748–1752) or by LRU choice (1757–1778); load picks the entry with
  the best (keep, sim) with `f_keep ≥ 0.25` (server-task.cpp:1796–1830), restores the whole blob
  (`llama_state_seq_set_data_ext`, 1835) and the normal LCP logic then trims; entries that are prefixes of the new
  prompt are deleted (1739–1749); size- and token-limited FIFO eviction (1873–1890). `--cache-idle-slots` saves
  idle slots to the store on the next task (server-context.cpp:1562–1575). The state writer is synchronous
  `ggml_backend_tensor_get` per layer range into a host `std::vector` (llama-context.cpp: `llama_io_write_buffer::
  write_tensor` 2921–2936 batches the gets; `llama_io_write_file` 3017–3024 copies through a temp buffer) —
  pageable memory, no pinned path, no overlap with compute.
- **Files, `--slot-save-path DIR`.** `POST /slots/{id}?action=save|restore|erase` (server-context.cpp:5298–5327;
  save task 2916–2957 via `llama_state_seq_save_file`, restore 2958–3000 via `llama_state_seq_load_file`, plus
  context checkpoints). File = tokens + per-layer K rows then V rows (`state_write_data`, llama-kv-cache.cpp:
  2248–2345; V transposed per element row when `v_trans`). README example: 1745 tokens, 14.3 MB, save 49.9 ms,
  restore 42.9 ms (README.md:1150–1195) = 8.2 KiB/token, ~290 MB/s — a file path, not a PCIe path.
- **KV types.** `-ctk/-ctv` ∈ {f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1} (arg.cpp:304–314); since
  PR #21038 quantised K/V are Hadamard-rotated per 64/128-wide block before storage (llama-kv-cache.cpp:
  321–356; file:line references PR #21038's comment URLs), rotation undone on the attention output.

### vLLM, SGLang, TensorRT-LLM (docs/papers)

- vLLM v1: 16-token blocks, hash-chained prefix cache over full blocks (sha256), refcount + LRU free queue,
  `cache_salt`; preemption by recompute (default) or swap (`--swap-space`); no per-session files.
- SGLang: token-granular pages, radix tree, LRU on leaves, cache-aware scheduling; HiCache (2025) adds host/disk
  tiers behind the same tree (docs; not read in full).
- TensorRT-LLM: blocks pack all layers; radix tree; prioritised LRU with per-token-range retention; evicted
  blocks go to secondary (host) memory and stay searchable.

None of the three promises batch-invariant bits; prefix hits are "almost a free lunch and won't change model
outputs" in vLLM's words, which is true only up to kernel reduction order.

---

## 4. Proposals for Axon, ranked

Common facts used below (all ESTIMATED from MEASURED parts unless marked): 7B f16 KV = 56 KiB/token, q8 ≈ 0.53×
(r10 §0); 32k slot = 1.75 GiB f16 / 0.93 q8; PCIe Gen5 x8 pinned H2D 25.4 GB/s MEASURED (brief) → 1.75 GiB in
**74 ms**, 0.93 GiB in 39 ms; D2H assumed symmetric (not measured — gate it). 300-token turn delta = 16.4 MiB →
0.65 ms. `MemAvailable` ~17 GB → ~9 f16 32k slots, ~20 agents at 15k used.

### P1. Host-RAM KV store on eviction + save/restore (= r10 rank 4, L1/M1), page-granular and append-only

**Change.** `CudaQwen2::save_rows(pool, slot, lo..hi) -> into pinned host page(s)` and `restore_rows`: for the f16
region, 2·nl strided chunks per 64-position page (each `64 × kv_dim × 2` B at pitch `max_ctx × row`), i.e. **one
`cuMemcpy2DAsync` per page per direction** (add `cuMemcpy2DAsync_v2` to ffi.rs; `DevBuf::copy_into` at
cuda/mod.rs:1136–1164 shows the pattern); q8 adds the scales region the same way (`adopt_cache`, slots.rs:
337–372, already walks these regions — it is the restore code for the GPU→GPU case). Host page = 64 tokens × all
layers (7B f16 3.5 MiB, q8 1.9 MiB; Qwen3-8B 9 MiB), keyed by a hash chain over 64-token blocks of the token
prefix (vLLM/Mooncake style: `h_b = H(h_{b-1}, tokens[64b..64b+64])`), refcounted, LRU by last use, capped by
`--cache-ram MiB` (llama-server's name). Store lives in `Batch` (engine.rs) beside `held`. Three hooks:

1. **write-through**: when a slot completes a 64-block (every 64 generated tokens, and at the end of each
   `feed_slot`), enqueue its page D2H on a copy stream (`Gpu::new_stream` + `SyncEvent` exist; stream.rs uses
   them). Cost per page 3.5 MiB / 25 GB/s = 0.14 ms, hidden under the next step. The tail (< 64 tokens) is saved
   on finish/eviction only. Eviction then costs ~0 and `save on exit` is instantaneous.
2. **admit**: longest prefix over free slots (as now) *and* over the store; if the store wins, restore the
   matching pages into the slot (`n/64` strided copies, ≥ 1.9 MiB each — LMCache Table 1 and vLLM §7.3 both say
   this granularity is what reaches link bandwidth; 64 KiB copies would run at ~4 GB/s), then `feed_slot` the rest.
   A hash hit on a *different* session's prefix is also a hit (shared system prompt + tool schemas).
3. **HTTP/CLI**: `POST /slots/:id?action=save|restore|erase` + `--slot-save-path` (llama-server's shapes so
   existing clients work; file = tokens + pages, f16/q8 bytes as stored, header with model id, KV mode, arch, so
   a restore refuses a mismatched tuple); `chat --session FILE`.

**Gain.** Swarm turn: 5–12 s re-prefill → 74 ms (f16 32k) or 39 ms (q8) + the suffix's prefill (~0.1 s for 300
tokens). 20k shared prefix for 12 agents: paid once, then 45 ms per agent instead of 7.5 s. Restart: 128k session
back in ~0.3 s instead of ~2 min (r10 §2.4). ESTIMATED throughout from the 25.4 GB/s and 2682 tok/s MEASURED figures.

**Bit identity.** Preserved: bytes move, nothing is recomputed; the restored slot's next token equals the
uninterrupted run's. The hash is a lookup key only — on a hit, compare the stored token block to the prompt's
(the store keeps tokens) so a collision cannot inject foreign bytes.

**Risk.** Low–medium: `admit` is rewritten by C1 (chunked prefill) too — land after C1. Pinned RAM: cap the pinned
store (default ¼ of `MemAvailable`); `cuMemHostAlloc` of several GiB is fine on a 30 GB box. `Gpu: !Send` means
the copies are issued from the engine thread — async on a side stream, fine.

**Builder-hours.** 24–32 (+8 for files and `chat --session`, +6 for layer-wise overlap below if wanted).

**Gate.** (a) identity: for 20 prompts/seeds, run continuously vs {evict after prompt, restore, continue} and
vs {save to file, restart process, restore}: token sequences equal, slot bytes equal (`slot_bytes` test helper
in slots.rs). (b) bella 7B f16 `--ctx 32768`: restore of a 32k slot ≤ 100 ms, q8 ≤ 60 ms; D2H bandwidth
measured and recorded. (c) `serve_load.py`: 12 clients, 20k shared prefix + 5k private, 3 slots: TTFT p50 ≤ 1 s
(main: ≥ 6 s). (d) `--kv q8` passes the same suite.

**Second phase (optional): layer-wise overlap.** CachedAttention/LMCache restore layer *l* while layer *l−1* of the
suffix prefill runs. `pass_at` launches layers in order on one stream; put restores on the copy stream and make
layer *l*'s first kernel wait on event *l*. Hides up to 74 ms behind a ~100 ms suffix prefill (TTFT 174 → ~100 ms).
Worth 6–8 h only after P1 is measured.

### P2. Drop the N+1 cache (= r10 rank 2, C2)

**Change.** Either a `CudaCache` *view* at slot 0's base (same pointers → graphs and drafting keep working;
`adopt` becomes bookkeeping) or no solo path once a pool exists. Nothing in this research changes r10's analysis;
it is listed because P3/P4 inherit its memory. **Gain** 7B f16 32k: 1 → 2 slots (3 without the q6k8 copy);
q8 3 → 5 (ESTIMATED above). **Bits** none. **Hours** 8–16. **Gate** `serve::` suite; `serve_check.py` C=1 within
noise of `gen`; bella 7B f16 `--ctx 32768` reports ≥ 2 slots.

### P3. Slots that grow: virtual-contiguous KV with CUDA VMM (vAttention) — the cheap half of "paged"

**Change.** Keep the pool's layout and every kernel. Reserve the pool's address range with
`cuMemAddressReserve` (one range: `2·nl` arrays × `slots × ctx` rows, plus q8 scales, as today), round `ctx` so a
slot's range in each array is 2 MiB-aligned (7B f16 rows are 1 KiB → `ctx` multiple of 2048; 14B/Qwen3-8B 2 KiB
rows → 1024; q8 halves the rows so doubles the multiple), and map physical 2 MiB chunks (`cuMemCreate` once at
start into a free list, `cuMemMap` + `cuMemSetAccess` per chunk) to a slot's range in each array only as the
slot's `held` length crosses a 2 MiB boundary; unmap on finish when the pool is short. New ffi entries:
`cuMemGetAllocationGranularity`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`,
`cuMemRelease`, `cuMemAddressFree`. `pool_bytes`/`plan_slots` change meaning: `--ctx` becomes a cap, `--parallel`
a cap, admission is by free physical chunks. Zero-fill a chunk when mapped (today `c.kv.zero()` at creation,
qwen2.rs:1570–1572; stale rows are never read). Map one step ahead of need (vAttention §5.3): on 7B a growth step
is 56 chunks × (2 + 38) µs ≈ 2.2 ms ESTIMATED from vAttention Table 3, off the critical path.

**Gain.** The reservation problem is gone without a kernel change: a 14B `--ctx 32768` server whose agents sit at
10–25k keeps everything on the GPU (r10 §1.2: 17 → ~55 tok/s ESTIMATED); 7B at 15k per agent: ~7 conversations on
12 GB (6.3 GiB / 0.82) less the VMM waste below. Oversize-request starvation ends (any slot may grow to `ctx`).
Internal fragmentation: ≤ 2 MiB per (layer, K/V) per slot = 112 MiB worst / ~56 MiB mean per 7B slot (ESTIMATED:
`2·28 × 2 MiB`); at 8 live slots ~0.45 GiB mean — the price of touching no kernel. 14B: 96 arrays → 96–192 MiB
per slot. (Stock CUDA offers no 64 KB granularity; vAttention needed a driver patch.)

**Bit identity.** Preserved trivially: identical bytes at identical virtual addresses; CUDA graphs (slots.rs:
49, 222–240, addresses baked) stay valid because virtual addresses never move.

**Risk.** Medium-low. Unknowns: driver cost of `cuMemSetAccess` on sm_120 laptop (vAttention measured A100);
whether `cuMemMap` may alias one physical handle at two virtual ranges (if yes, a read-only shared system prefix
can be aliased into every slot at 2 MiB granularity — C5 without copy-on-write kernels; verify, §6). Fallback is
P4.

**Builder-hours.** 16–24. **Gate.** `serve_identity` with slots grown/shrunk in random orders == contiguous pool,
bytes equal; 14B Q4_K_M `--ctx 32768` on bella stays all-GPU with a 15k agent prompt and decodes ≥ 45 tok/s;
`nvidia-smi` shows pool memory tracking used tokens ±0.5 GiB; growth-step latency logged ≤ 5 ms.

### P4. Block-table paging at 64 positions (= r10 rank 5, B1/C3) — the full version

**Change.** r4 §4.2's table, unchanged by this research: `rope_kv_b`/`rope_kv_norm_b`/`kv_q8_b` store at
`tab[seq][P>>6]*64 + (P&63)`; `attn_blk_b_{64,128}` (+q8) read block *b* at `tab[seq][b]*64`; `attn_fa_b{,s}`
tiles at `tab[seq][b]*64` with per-sequence CTA row ranges; `attn_cmb_b` takes `P[z]`; everything token-major
(`embed_b`, norms, `bgemm`, heads) untouched. Host: free list, refcounts, block tables uploaded with the step's
tokens; CLI path through an identity table (`tab[s][b] = b`), which is itself the test.

**Does block size 64 align it for free? Yes.** CODE decode.cu:1126–1163: "Positions go in blocks of ATT_BLK = 64";
step 2 computes block *b*'s scores from rows `64b..`, steps 4–8 are *within* the block, step 9 folds "blocks in
ascending order, only those with 64b < np"; prefill walks the same 64-row tiles (`attn_fa_b`, prefill.cu:
387–435, `t0 = b * ATT_BLK`), and the decode kernels + `attn_combine_blk` fold the same partials in the same order.
The q8 rule is per (position, kv-head) group (decode.cu:1042–1051); RoPE and KV stores are per row. Nothing in
the arithmetic references a row's *address*, only its *position*; a page table changes addresses of whole
64-row blocks and nothing else ⇒ same bits, by construction, for f16 and q8, decode and prefill. The in-flight
`feat/long-ctx-decode` (fb4ed4a) keeps `ATT_BLK 64` (4 warps per block, 64-output combine) — still aligned.
The one path that is *not* 64-aligned is the pre-sm_80 `attn_split` (decode.cu:1420–1436: `chunk = ceil(np /
nsplit)`, tiles of 128) on the 1050; it already gives different bits from the tensor-core path and would need
per-64-row address translation inside its tile loop (slower there, but that GPU is a fallback).

**Kernel cost.** One `tab` load (4 B) per 64-row block per warp against 16–32 KiB of K+V streamed per block:
ESTIMATED < 1 % on attention; vAttention's 7–13 % instruction overhead came from 16-token blocks and per-element
index math in FA2/FlashInfer, not from this shape. Register pressure: one register.

**Gain over P3.** Internal waste 3.5 MiB per 7B slot instead of ~56–112 MiB; **copy-on-write prefix sharing**
(C5) and a GPU-resident prefix cache (refcount-0 pages kept until needed, SGLang/vLLM style) become possible:
12 agents × 20k shared prefix = 13 GiB of KV → 1.1 GiB (ESTIMATED r10 C5). With P1 in place the shared prefix is
already *cheap* (45 ms restore per agent); P4+C5 makes it *free* and memory-shared.

**Bits** none (above). **Risk** the broadest kernel touch since S2 (every cache-indexing kernel in prefill.cu /
decode.cu and q8 twins). **Hours** 40–60 (+40–60 for C5). **Gate** `serve_identity` with shuffled page tables ==
contiguous, token- and byte-exact; attention µs with the identity table within 1 % of today (`cuda_timeline`);
pool-invariant fuzz (no page double-use) in debug builds; 7B f16 `--ctx 32768` on bella: ≥ 6 live 15k agents.

Ordering between P3 and P4: P3 first if the team wants the 14B win in a fortnight; P4 is the end state and
subsumes P3 (the VMM reserve can remain as the physical allocator under the page pool, which is what eLLM argues).

### P5. Opt-in lower-bit KV: `--kv q4v` (K q8, V q4) first, `--kv q4` second — after r9 #1, as a new identity tuple

**Change.** A `CacheKv::Q4 { k: u8|u4, group }` mode; V rule Q4_1-like per 32 values (min + scale, asymmetric,
RNE, f32 ops correctly rounded, dequant `min + d·q` folded in the fixed fma chain exactly as the q8 `d·q` is);
K stays q8 with the key shift in `q4v`; in full `q4`, K keeps the shift (it removes Qwen's near-constant giant
channels — the thing that wrecks llama.cpp's un-rotated q4_0 on Qwen, PR #21038's 0.6B/8B tables) and optionally
a fixed 64×64 Hadamard per head (llama.cpp's recipe; rotation is a fixed-order small matmul so the bits stay a
function of the prefix). Kernels: `kv_q4_store`, `att_cvt_q4` twins of the q8 ones (decode.cu:1077–1101 and the
`att_cvt_q8` tile converter), the `Q8Fns` table pattern (qwen2.rs:207–268) extended.

**Expected quality (MEASURED elsewhere, not on Axon).** V-only 4-bit per-token g32 ≈ no loss (KIVI Table 1:
4-bit per-token K+V = 16-bit on CoQA/TruthfulQA; V has no outlier structure, KIVI §3.2). K at 4-bit with g32–g64
grouping: +0.05 (FlexGen g64) to +0.12/+0.14 (Atom g128, QServe KV4 pre-Smooth) PPL on LLaMA/Llama-2-7B;
llama.cpp q4_0 (g32, symmetric, no outlier handling) on Qwen3-8B **+0.32, +0.18 rotated**; 5-bit +0.06/+0.01.
So: `q4v` ≈ q8 quality; full `q4` on Qwen needs the shift (and probably rotation) to land near +0.1–0.2 PPL, and
small Qwen (≤ 1.5B) should refuse it. Bytes: `q4v` K 0.52× + V 0.31× → **0.41× f16** (vs q8 0.53×); `q4` 0.31×.
7B 128k: 7.0 → 3.7 (q8) → 2.9 (`q4v`) → 2.2 GiB (`q4`).

**Speed.** None today: decode attention at 32k is latency-bound at 350 of 648 GB/s and q8 gains nothing (r9 fact
2, MEASURED 70.9 vs 72.4). Only after r9 #1 makes it bandwidth-bound does fewer bytes buy time (branch: q8
+13.8 % MEASURED); then `q4` could be worth ~+10 % more at 32k (ESTIMATED, half the bytes of q8 on a ~1/3 share
of the step).

**Bit identity.** Preserved *within the mode* (same structure as q8: per-row quant, fixed-order dequant; batched
== solo; prefill == decode). A third numerics tuple to test (f16 / q8 / q4), which is the real cost.

**Risk** high on quality, moderate on work. **Hours** 20–28 for `q4v`, +12 for full `q4` with rotation.
**Gate** KL(f16 ‖ mode) on 7B and 14B at 8k and 32k: `q4v` ≤ 1.5× q8's; `q4` ≤ 3× q8's and a 32k needle test
passes; tg@32k ≥ q8 + 8 % after r9 #1; md5 of the identity suite within the mode. **Verdict:** keep B5's "likely
not a round" until P3/P4 land; P1–P4 solve the 12 GB cases losslessly.

### P6. Fastest exact save/restore path on PCIe Gen5 x8 (what P1 should implement, specifically)

1. **Pinned host memory** (`cuMemHostAlloc`, mod.rs:463–470): the 25.4 GB/s figure is pinned; pageable copies
   stage through the driver at roughly half that (not measured here — gate it).
2. **Large transfers**: ≥ 1.9 MiB per call (one page × all layers via `cuMemcpy2DAsync`), never per-layer 64 KiB
   rows (LMCache Table 1: 64 KB → 4 GB/s, 1 MB → 30 GB/s, 16 MB+ → link rate; vLLM §7.3).
3. **Two copy engines, full duplex**: evict (D2H) and restore (H2D) on two streams at once — ~75 ms for the pair,
   not 150 ms (ESTIMATED; consumer Blackwell has ≥ 2 copy engines — verify with `cuDeviceGetAttribute
   ASYNC_ENGINE_COUNT`, §6).
4. **Write-through** (P1 hook 1) so eviction never waits on a D2H; saves cost 0.14 ms per filled page.
5. **Overlap restore with the suffix prefill** layer by layer (P1 phase 2).
6. **NVMe tier**: files are page-aligned dumps of the host pages (`O_DIRECT` optional); a Gen4 NVMe at ~5 GB/s
   reads a 1.75 GiB slot in ~0.35 s (ESTIMATED; bella's drive not characterised). GPUDirect Storage is not worth
   chasing on GeForce; RAM → GPU is the hot path, disk → RAM the cold one.
7. **Compression is not on the exact path.** CacheGen-style delta/arithmetic coding is lossy; lossless
   byte-level compression of f16/q8 KV is ~1.0–1.1× and costs CPU — skip.

---

## 5. Dead ends (for Axon)

- **Context shift / K-shift / `--cache-reuse` chunk shifting** (llama-server 3277–3315, 3633–3679) and
  **CachedAttention's decoupled-RoPE truncation**: the shifted cache is not what a fresh prefill writes; breaks
  "chat reuse == fresh" by design. r10 L4 already decided; this research agrees.
- **StreamingLLM** (sinks + window with cache-relative positions): same reason, plus it changes every output
  past the window. Fine as a paper, wrong for an engine whose selling point is exactness.
- **H2O / SnapKV / InfiniGen / ShadowKV / Quest-class eviction and sparse retrieval**: lossy per token; need
  attention probabilities or key centroids exported from flash-style kernels Axon doesn't export; produce
  per-row irregular token sets (a page table with holes plus a mask). Acceptable only as an explicitly lossy
  `--kv-evict` research flag with its own identity tuple — and for agent workloads (tool schemas, code, JSON)
  dropping tokens is the wrong trade. Not a round.
- **2-bit / 1-bit KV (KIVI-2, CQ, KVQuant-2, TurboQuant)**: per-channel K needs transposed/group-of-tokens
  layouts that conflict with per-row stores and paging at 64 (KIVI's residual window is literally a different
  layout for the last R tokens); CQ needs per-model calibration; the quality numbers at 2 bits (KIVI 2-bit
  per-token: CoQA 66 → 53) are not what a "beats llama.cpp" engine ships by default. Revisit only if 128k on
  6 GB cards becomes a target.
- **KV offload as a decode-time memory extension** (InfiniGen/FlexGen style): 7 GiB of 128k KV over 25 GB/s is
  280 ms per token (r10 §2.4). Host RAM is a *store*, never the attention operand.
- **Unified ragged pool, llama.cpp style** (seq-masked attention over the whole pool): a sequence's attention
  cost scales with the pool's occupancy (llama-kv-cache.cpp:1278–1290) — the opposite of Axon's per-slot `np`.
- **Hash-only prefix matching without token verification**: vLLM moved to sha256 for a reason; Axon keeps the
  tokens per page and compares, so any fast hash (xxhash-class) is safe.
- **Whole-slot blobs picked by LCP fraction (llama.cpp's `server_prompt_cache`)**: simple, but it duplicates the
  shared prefix per session and can't serve a cross-session prefix hit without loading someone else's whole
  state. Page-granular hash chains (P1) cost the same code and fix both.

---

## 6. Open questions a 1-hour experiment could settle

1. **D2H bandwidth and duplex on bella**: `cuMemcpyDtoHAsync` of 1.75 GiB pinned, then D2H ‖ H2D on two streams;
   read `CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT`. Decides P6 items 1 and 3 (expect ~25 GB/s each way).
2. **`cuMemcpy2DAsync` strided page copy rate**: 56 rows × 64 KiB per 7B page vs 56 separate `DtoH` calls vs one
   contiguous 3.5 MiB copy. Decides P1's page size (64 vs 256 tokens per host page).
3. **VMM on sm_120**: `cuMemGetAllocationGranularity` (is it 2 MiB?), `cuMemMap`+`cuMemSetAccess` latency per
   chunk, and **whether one physical handle maps at two virtual ranges** (read-only shared prefix for free).
   Decides P3 and whether C5 has a kernel-free form.
4. **Paged attention overhead**: patch `attn_blk_b_128` to read block addresses through an identity table and
   time a 32k single-row step (`cuda_timeline`): expect < 1 %. Decides whether P4 needs any kernel tuning.
5. **q4 KV quality on Qwen with Axon's key shift**: a Python fake-quant of the dumped f16 cache (Q4_1 g32 V only;
   then K with and without the shift and with a 64×64 Hadamard) → KL vs f16 logits on 7B at 8k, compared to the
   PR #21038 q4_0 numbers. Decides whether P5's `q4` is +0.1 or +0.3 PPL territory before any CUDA is written.
6. **Pageable vs pinned restore**: the same 1.75 GiB from a `Vec<u8>` vs `HostBuf`. Decides whether the store
   must be pinned or can be pageable with a pinned staging ring.
