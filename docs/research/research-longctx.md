# research: longer sessions — context past the trained length, long chats, session persistence

Read-only recon (Chrollo), Axon at `e998d6f` (a detached worktree), llama.cpp at `4413a19`,
GGUF probes of the local model files with a 30-line metadata reader (not in the repo).
Labels: **CODE** (file:line), **MEASURED** (paper table or Axon spec/log), **ESTIMATED** (arithmetic shown),
**CLAIMED** (paper's claim, not reproduced). Papers read in full (pdftotext): YaRN, Qwen2.5 report, Qwen2.5-1M
report, StreamingLLM, LongRoPE2, SelfExtend, DCA, RULER. Others: abstract + the cited section.

---

## 1. The problem in Axon's terms

- **Hard wall at the trained length.** `--ctx` ≤ `ctx_len` (CODE serve.rs:653–659, main.rs:874–882); any GGUF
  `rope.scaling.factor ≠ 1` or `attn_factor ≠ 1` is refused (CODE model.rs:499–511, 535–548). Qwen2.5 7B/14B/32B are
  32768, Qwen3 40960 (PROBED, §3.4) and **none of the Qwen GGUFs carries a `rope.scaling.*` key** (PROBED): the
  "128k with YaRN" of the model cards is an operator flag in llama.cpp/vLLM, not file metadata. An agent at 33k gets
  a 400 `context_length_exceeded` (CODE serve.rs:1275–1281).
- **The chat window re-prefills on nearly every turn once full.** `Chat::reply` drops the minimal `k` exchanges
  (CODE chat.rs:584–608); `drop_oldest` resets `fed` to the system block (chat.rs:506–532), so the whole post-system
  history is fed again. At a 32k 7B window that is ~24k tokens (ctx − `want` = 32k − 8k) at 2682 tok/s MEASURED
  (r9 §0) ≈ **9 s per exchange** (ESTIMATED), repeating every exchange.
- **No persistence.** A restart loses every slot; a 25k agent prefix re-costs 9–12 s TTFT; a 128k session would
  re-cost ~2 min (r10 §2.4 ESTIMATED). llama-server has had `/slots/:id?action=save|restore` since 2024.
- **The contract is the asset.** Axon's cache bytes are a pure function of (token prefix, model, kv type, rope
  config, backend). llama.cpp's own README says `cache_prompt` is not bit-reproducible across batch sizes
  (CODE llama.cpp tools/server/README.md:589). Anything proposed below must keep that function pure.

## 2. Literature map

### 2.1 RoPE rescaling (frequency-side)

| # | paper | idea | numbers | hw |
|---|---|---|---|---|
| P1 | **Position Interpolation** — Chen, Wong, Chen, Tian (Meta), 2023, arXiv 2306.15595 | Scale positions by 1/s so all positions land inside the trained range; needs ~1000 fine-tune steps. | LLaMA 7B–65B to 32k with PI + fine-tune (CLAIMED). Non-fine-tuned PI at s=4 on LLaMA-7B: PPL 7.09 at 2k vs 4.05 baseline (MEASURED, YaRN Table 5). | A100 |
| P2 | **NTK-aware / Dynamic NTK** — bloc97 and emozilla, Reddit r/LocalLLaMA, June 2023 (cited by P3 as (bloc97, 2023a/b), (emozilla, 2023)) | Change the base so high-frequency pairs extrapolate and low-frequency interpolate; "dynamic" = s = max(1, len/L) recomputed per forward. | Non-fine-tuned LLaMA-7B, s=4: PPL 4.27 at 2k, >101 at 8k (NTK-aware blows up at the boundary) (MEASURED, YaRN Table 5). | — |
| P3 | **YaRN** — Peng, Quesnelle, Fan, Shippole, 2023, arXiv 2309.00071 (ICLR 2024) — *read in full* | "NTK-by-parts": per-pair ramp between extrapolation (pairs with ≥ β=32 rotations in L) and interpolation (≤ α=1 rotation), plus an attention temperature √(1/t) = 0.1·ln(s)+1 applied to q and k. Also Dynamic-YaRN (§3.4). | **Non-fine-tuned LLaMA-7B (Table 5, MEASURED):** static s=4: PPL 4.19/3.77/3.65 at 2k/4k/8k vs baseline 4.05 at 2k; static s=16: 4.61 at 2k; **dynamic**: 4.05/3.67/3.65/3.33/3.45 at 2k–32k (identical to baseline below L, equal or better than static beyond). Fine-tuned Llama-2 7B s=32: PPL 2.37 at 128k (Table 1). | A100 (fine-tunes: 400 steps, bs 64) |
| P4 | **LongRoPE** — Ding et al. (Microsoft), 2024, arXiv 2402.13753 | Evolutionary search over per-dim rescale factors + a start-token bias; progressive 256k → 2M. | LLaMA2-7B to 2048k with passkey ≥ 90% (CLAIMED); short-context recovered by a second rescale factor set for ≤ 4k (a *dual* factor table switched by length). | A100 |
| P5 | **LongRoPE2** — Shang, Zhang et al. (Microsoft), 2025, arXiv 2502.20082 — *read in full* | Hypothesis: YaRN/NTK fail because higher RoPE dims are *under-trained*, so the true critical dim is lower than the theory's; search factors with a "needle-driven" PPL; mixed-window training keeps short-context quality. | LLaMA3-8B → 128k effective on RULER while keeping >98.5 % short-context, with 10B tokens of training (CLAIMED, Fig. 1). States LLaMA3.1-8B's YaRN-based 128k "falls short" of its effective length on RULER. | A100 |
| P6 | **Llama 3 rope scaling** — "The Llama 3 Herd of Models", 2024, arXiv 2407.21783; HF config `rope_scaling {type: llama3, factor 8, low_freq_factor 1, high_freq_factor 4, original 8192}` | A per-pair divisor between 1 and 8 by wavelength (λ < 2048: 1; λ > 8192: 8; smooth between), **no temperature**. llama.cpp's converter writes it as the `rope_freqs.weight` tensor (CODE llama.cpp conversion/llama.py:306–326). | Llama 3.1 8B RULER 95.5/93.8/91.6/87.4/84.7/77.0 at 4k–128k, effective 32k (MEASURED, RULER Table 3). | — |
| P7 | **Qwen2.5 Technical Report** — Qwen team, 2024, arXiv 2412.15115 — *read in full* | Trained 4k → 32k with ABF (base 1e4 → 1e6). For >32k inference: **DCA + YaRN attention scaling** (not the frequency part). Model card recipe for other engines: static `rope_scaling {type yarn, factor 4, original_max_position_embeddings 32768}`; "vLLM only supports static YARN … potentially impacting performance on shorter texts … add only when processing long contexts". | **Qwen2.5-7B-Instruct RULER (Table 16, MEASURED): 96.7/95.1/93.7/89.4 at 4k–32k; with DCA+YaRN 82.3 @64k, 55.1 @128k; without: 74.5 @64k, 31.4 @128k.** 14B: 86.7/78.1 vs 82.3/53.0. 32B: 90.3/82.0 vs 85.3/57.7. LV-Eval 7B: 48.0/41.1 @64k/128k with, 33.1/13.6 without (Table 17). "DCA+YaRN does not change the model behavior within 32K" (their deployment, see §2.2 P9). | Their vLLM fork |
| P8 | **Qwen3 Technical Report** — 2025, arXiv 2505.09388; Qwen3-8B model card | Native 32,768; card: `max_position_embeddings` 40,960 = 32k output reserve + 8k prompt. YaRN factor 4, orig 32768 → 131,072 "validated". "All the notable open-source frameworks implement static YaRN … potentially impacting performance on shorter texts … if the average context length does not exceed 32,768 tokens, we do not recommend enabling YaRN"; "if your typical context is 65,536, set factor 2.0". llama.cpp line given verbatim: `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`. | — (no RULER table for the YaRN setting in the card) | — |

### 2.2 Attention-side extrapolation (no rescaling of trained frequencies)

| # | paper | idea | numbers | hw |
|---|---|---|---|---|
| P9 | **Dual Chunk Attention (DCA)** — An et al., 2024, arXiv 2402.17463 (ICML 2024) — *read in full*; and **Qwen2.5-1M report** — 2025, arXiv 2501.15383 — *read in full* | Keys get position `k mod s` (chunk-local); queries are roped three ways (intra: `q mod s`; successive: `q mod s + s_local`, clamped; inter: constant `s−1`) and attention picks the q variant by chunk relation; three flash-attention passes merged by LSE. Relative distances never exceed the trained length. Qwen adds YaRN's **temperature only**, √(1/t)=0.1 ln(s)+1, s = inference length / training length. | Training-free Llama2-70B "ChunkLlama": 94 % of gpt-3.5-16k on long tasks, 192k with high passkey (CLAIMED). Qwen2.5-7B-Instruct with DCA+YaRN: **passkey > 80 % at 1M** despite 32k training (CLAIMED, 1M report §3). RULER 64k/128k numbers as P7. **Key property**: for positions below the chunk length (vLLM: `chunk_size − local_size`) key positions equal the plain ones and queries only see intra-chunk keys → bit-identical there. | 2×A100-80G |
| P10 | **StreamingLLM / attention sinks** — Xiao, Tian, Chen, Han, Lewis, 2023, arXiv 2309.17453 (ICLR 2024) — *read in full* | The first ~4 tokens absorb attention mass ("sinks"); window attention collapses when they are evicted; keep 4 sinks + a sliding window and the model streams indefinitely with stable PPL. Explicitly: does **not** extend the usable context ("we do not expand the attention window"); the quality-correct baseline is *sliding window with re-computation*. | Up to 22.2× faster than re-computation (MEASURED, their Fig/§4); re-computation is what Axon does today, once per drop. | A100 |
| P11 | **LM-Infinite** — Han et al., 2023/2024, arXiv 2308.16137 (NAACL 2024) | Λ-shaped mask (global start + local window) plus a distance ceiling (relative positions clipped at L). | PPL stable to very long inputs; 2.7× decode speedup, 7.5× memory (CLAIMED); retrieval limited to the window unless the later "top-k middle tokens" variant is used. | — |
| P12 | **SelfExtend** — Jin et al., 2024, arXiv 2401.01325 (ICML 2024) — *read in full* | Grouped attention (`floor(pos/G)`) for far tokens + normal attention for the neighbor window `w_n`; two attention passes merged. No training. | Llama-2-7B-chat 4k → 16k/25k; passkey 100 % within the extended range; LongBench SE-Llama-2-7B-chat-16k averages in the Table 3 rows (MEASURED). Cost: 2× attention. | A100 |
| P13 | **InfLLM** — Xiao et al., 2024, arXiv 2402.04617 (NeurIPS 2024) | Sliding window + block memory in CPU RAM; blocks summarized by representative tokens; top-k blocks retrieved per step and given local positions. | Mistral-7B / Llama-3-8B to 1024k with ~100 % passkey (CLAIMED). Lossy (retrieved blocks share one distance); a retrieval system, not a cache mode. | A100 |

### 2.3 Evaluation

| # | paper | what it measures | numbers used above |
|---|---|---|---|
| P14 | **RULER** — Hsieh et al. (NVIDIA), 2024, arXiv 2404.06654 — *read in full* | 13 synthetic tasks (NIAH variants, variable tracking, aggregation, QA) at 4k–128k; "effective length" = longest length scoring ≥ Llama2-7B's 85.6 at 4k. | Table 3: Qwen2 72B claimed 128k, effective 32k (96.9 … 79.8 @64k, 53.7 @128k); Llama3.1 8B effective 32k; Mistral-v0.2 7B 49.0 @64k, 13.8 @128k (MEASURED). Only 50 % of models keep performance at their claimed length. |
| P15 | **LongBench** — Bai et al., 2023, arXiv 2308.14508 (ACL 2024); **LongBench v2** — 2024, arXiv 2412.15204 | Real tasks, bilingual, mostly < 32k (v1); v2: 8k–2M, multiple choice. | Used by P12 (Table 3) and P7 (LongBench-Chat column: Qwen2.5-7B 7.42). |
| P16 | **Needle-in-a-haystack** — Kamradt, 2023 (GitHub `gkamradt/LLMTest_NeedleInAHaystack`; not a paper) | Single-fact retrieval at depth × length. | The passkey task of P3/P9/P12 is its ancestor (Mohtashami & Jaggi 2023). Too easy to be a quality gate alone (P14 §1; P12 §4.3 say so). |

### 2.4 History compression (lossy)

| # | paper | idea | why it matters here |
|---|---|---|---|
| P17 | **H2O** — Zhang et al., 2023, arXiv 2306.14048 (NeurIPS 2023); **SnapKV** — Li et al., 2024, arXiv 2404.14469 (NeurIPS 2024) | Evict KV rows by accumulated / observation-window attention score; keep recent + heavy hitters. | Cache becomes a function of the *generation path* and of the eviction schedule, not of the prefix; prefill-vs-decode eviction differs. Breaks contract 1. Opt-in only, and only after B1 (paged KV) exists to hold holes. |
| P18 | **LLMLingua / LongLLMLingua** — Jiang et al., 2023, arXiv 2310.05736 / 2310.06839 (EMNLP 2023 / ACL 2024) | Token-side prompt compression with a small LM's perplexity; 2–20× (CLAIMED). | Token-side = contract-safe (the compressed prompt is just tokens). A harness feature, not an engine one. |
| P19 | **Recursively Summarizing … Long-Term Dialogue Memory** — Wang et al., 2023, arXiv 2308.15022; **MemGPT** — Packer et al., 2023, arXiv 2310.08560 | Summarize old turns into a memory block the model reads; MemGPT: the model manages its own context ("OS" paging) with a recall store. | The agent-harness pattern (§3.4). Token-side; keeps the contract. |

### 2.5 2025–2026 follow-ups

- **STRING** — "Why Does the Effective Context Length of LLMs Fall Short?", 2024/2025, arXiv 2410.18745: effective
  length is typically < half the trained length because of the left-skewed relative-position distribution; shifting
  well-trained positions over the ineffective ones at inference improves long-context scores without training
  (CLAIMED). Relevant diagnosis: Qwen2.5-7B's own RULER drops 96.7 → 89.4 *inside* 32k (P7).
- **Base of RoPE Bounds Context Length** — Men et al., 2024, arXiv 2405.14591: a lower bound on the base for a given
  length; "superficial" long-context ability from OOD-only fixes. Supports treating static YaRN on a 1e6-base model as
  a mitigation, not a guarantee.
- **CoPE: Clipped RoPE** — Li, Ren, Yuille, Wang, 2026, arXiv 2602.05258: soft-clip the low-frequency pairs; unifies
  OOD-mitigation and "semantic" views; training-time method.
- **Q-ROAR** — Qiao, Huang, 2025, arXiv 2509.14391: PI/YaRN on *quantized* weights amplifies outliers; proposes
  rescaling per weight group. Direct relevance: Axon runs Q4_K_M; any YaRN quality gate must be run on the GGUF, not
  on bf16 numbers from a card.
- Qwen3 (P8) and Qwen2.5-1M (P9) are the production recipes of 2025; LongRoPE2 (P5) is the current
  research recipe (needs training, so not an inference-engine option).

## 3. What production engines do

### 3.1 llama.cpp (`4413a19`)

- **Flags** (CODE common/arg.cpp:2337–2401): `--rope-scaling {none,linear,yarn}`, `--rope-scale N` (sets
  `rope_freq_scale = 1/N`), `--rope-freq-scale`, `--rope-freq-base`, `--yarn-orig-ctx N`, `--yarn-ext-factor`
  (default −1 → 1.0 when yarn, else 0), `--yarn-attn-factor`, `--yarn-beta-fast` (32), `--yarn-beta-slow` (1).
  Defaults from GGUF `rope.scaling.*` keys when present (CODE src/llama-context.cpp:200–224, 251–256).
- **mscale** (CODE src/llama-context.cpp:259–300): `yarn_attn_factor = get_mscale(1/freq_scale) = 0.1·ln(factor)+1`,
  then divided by the same quantity because the kernel multiplies it back in (ops.cpp:6197 / rope.cu:34; PR #17945);
  net kernel mscale = 0.1·ln(factor)+1 on **both** q and k → logits × (0.1·ln s + 1)². With yarn and a custom scale,
  `n_ctx_train` is rewritten to `n_ctx_orig_yarn / freq_scale` (CODE src/llama-context.cpp:4110–4113) — that is how
  `-c 131072` becomes legal.
- **Ramp** (CODE ggml/src/ggml.c:4479–4491; ggml-cpu/ops.cpp:6179–6200; ggml-cuda/rope.cu:15–41):
  `corr_dim(n_rot) = n_dims·ln(n_ctx_orig/(n_rot·2π)) / (2·ln base)`, `low = floor(corr_dim(β_fast))`,
  `high = ceil(corr_dim(β_slow))`; `ramp(i) = 1 − clamp((i/2 − low)/(high − low), 0, 1)`;
  `θ = θ·freq_scale·(1 − ramp·ext) + θ·ramp·ext`; `cos/sin · mscale`. All f32; θ accumulated by repeated
  multiplication by `theta_scale` (ops.cpp:6214) — Axon's f64 table will differ in low bits (the existing spread).
- **GGUF keys** (CODE gguf-py/gguf/constants.py:265–275; conversion/base.py:1633–1654): the converter writes
  `rope.scaling.type/factor/original_context_length` (+ ext/attn/beta if present) **only if the HF config has
  `rope_type: yarn`**. Qwen's configs don't (§3.4), hence no keys.
- **Context shift** (CODE tools/server/server-context.cpp:3273–3335): only while *generating* and only with
  `--context-shift`; `n_keep` (+1 for BOS, capped at n_ctx − 4), `n_discard = n_left/2`, `seq_rm` then `seq_add(…, −n_discard)`
  = a K-cache RoPE delta in place; tokens compacted; `slot.truncated = true`. Without the flag the slot stops with
  `STOP_TYPE_LIMIT` (server-context.cpp:2029–2036). A *prompt* ≥ n_ctx is a 400 `ERROR_TYPE_EXCEED_CONTEXT_SIZE`
  (server-context.cpp:3578–3585) — prompt truncation was removed. `--cache-reuse N` shifts matching chunks on edits.
- **Persistence**: `POST /slots/:id?action=save|restore|erase` under `--slot-save-path` (README.md:1150–1195;
  server-context.cpp:2920–2945 `llama_state_seq_save_file` with the token list; magic `GGSQ` version 4,
  include/llama.h:48–49; per-layer K then V cell ranges, `v_trans` flag, src/llama-kv-cache.cpp:2248–2290).
  `--cache-ram MiB` (default 8192; arg.cpp:1719) keeps evicted slots' caches in host RAM; a slot about to lose
  > 50 % of its context is saved first, LRU (server-context.cpp:1750–1800, 1492–1502). `--cache-idle-slots`.
  ollama launches llama-server with `--cache-ram 0` and `--kv-unified` (ollama llm/llama_server.go:398).

### 3.2 ollama (main, read from GitHub)

- Clamps `num_ctx` to the GGUF's trained length (CODE server.go:111–113) — **no YaRN path at all**.
- Chat level: drops whole messages from the front, keeping every system message and the last message
  (CODE server/prompt.go:21–76, `chatPrompt`).
- Prompt level: if the rendered prompt still exceeds `num_ctx − 1` and context shift is enabled, keeps `num_keep`
  (+BOS) and the last `num_ctx − (num_ctx − num_keep)/2` tokens ("free roughly half", CODE llm/llama_server.go:299–348);
  otherwise a 400 (`promptTooLongMessage`).
- Generation overflow: passes `--context-shift` (+ `--keep`) to llama-server (llama_server.go:815–822).

### 3.3 vLLM (main, read from GitHub)

- `get_rope` dispatches `rope_scaling.rope_type == "yarn"` to `YaRNScalingRotaryEmbedding` (CODE
  vllm/model_executor/layers/rotary_embedding/__init__.py:243–283): **static**; `cos/sin · mscale` with
  `mscale = 0.1·ln(factor)+1` unless `attention_factor` / `mscale,mscale_all_dim` given; `truncate` (floor/ceil)
  default True (yarn_scaling_rope.py:26–46, 76–85). Same math as HF `_compute_yarn_parameters`
  (transformers modeling_rope_utils.py:345–486) and as llama.cpp. Enabled by `--hf-overrides` / `--rope-scaling`
  JSON (Qwen3 card) and `--max-model-len 131072`.
- DCA: `dual_chunk_rope.py` (keys `pos mod chunk_len`, five q variants) remains in main; the DCA attention backend
  is not in the current tree listing — the DCA path Qwen validated lives in Qwen's fork / older V0.
- Prefix caching by hashed blocks; no context shift; no session files.

### 3.4 Probe: which local GGUFs carry scaling metadata (PROBED, read-only)

| file | arch | `context_length` | `rope.freq_base` | `rope.scaling.*` | `rope_freqs.weight` |
|---|---|---:|---:|---|---|
| Qwen2.5-{0.5B,7B,14B,32B}-Instruct Q4_K_M | qwen2 | 32768 | 1e6 | **none** | no |
| Qwen3-{4B,8B} Q4_K_M | qwen3 | 40960 | 1e6 | **none** | no |
| Meta-Llama-3.1-8B-Instruct Q4_K_M | llama | 131072 | 5e5 | none (encoded as the tensor) | yes (Axon reads it, model.rs:387–400) |
| Mistral-7B-Instruct-v0.3 | llama | 32768 | 1e6 | none | no |

### 3.5 Agent harnesses (what they need from the server)

- **opencode** (CODE packages/core/src/session/compaction.ts:40–60, 90–100, 211–233, 911–918): auto-compacts when
  its estimate ≥ `window − max(10 %·window, 16 000)`; on a provider "too long" rejection (`reason: overflow`) it retries
  at 0.7 / 0.5 / 0.35 of the rejected size; summary is an LLM call with a fixed Markdown template, `keep` recent
  tokens verbatim. Needs: an honest 400, a correct `n_ctx` (Axon: `/v1/models` meta, serve.rs:1207) and
  `usage.prompt_tokens`.
- **Claude Code** (docs, code.claude.com/docs/en/prompt-caching): request = stable prefix (system, project context)
  + conversation; compaction rewrites the conversation and is counted as an "expected rebuild" of the prompt cache.
  The pattern: *the server caches the prefix; the client decides what the prefix is.*
- **ollama** truncates on behalf of the client (§3.2) — the behavior harness authors complain about (silent loss).

## 4. Proposals for Axon, ranked

### L1 — KV save/restore (bit-exact session persistence) — **first**

**Change.** `CudaQwen2::save_slot(pool, slot, n) -> SlotImage` / `load_slot(pool, slot, &SlotImage)`: the loop of
`adopt_cache` (CODE cuda/qwen2/slots.rs:337–377) with `HostBuf::copy_from`/`copy_to` at offsets (mod.rs:960–990;
today `copy_from` lands at offset 0 only — add an `at` argument) instead of `DevBuf::copy_into`. Per region
(F16: rows of `kv_dim·2` B; Q8: rows of `kv_dim` B then scales `kv_dim/group·4` B), per layer K then V,
`n` rows from `slot·ctx`. One `cuMemcpy2DAsync` per region (src pitch `max_ctx·row`, dst pitch `n·row`, height
`2·n_layers`) replaces `2·n_layers` calls. CPU caches (`Layers<Half>`, kv.rs; hybrid's CPU-layer KV, hybrid.rs:100–108)
are already host slices: `memcpy`. Serve: `held[slot]` (engine.rs) is the token prefix; `Batch::admit` already reuses
by longest common prefix, so a restored slot is just a pre-filled `held` + bytes. Chat: `Chat` state = `hist`,
`sys`, `turns`, `msgs`, `system`, `fed`, think flags, and the sampler's xoshiro state `[u64; 4]` (sample.rs:204;
"one number per pick", :251–256) so sampled continuations match too.

**File format `AXKV` v1** (a header, then tokens, then regions in the layout above):
`{magic, version, model_id (GGUF basename + size + hash of the tensor-info table), arch, n_layers, kv_dim, head_dim,
kv_mode (F16 | Q8{group}), rope {base, scaling_type, factor, orig_ctx, mscale, beta_fast, beta_slow},
backend ("cuda sm_120" + kernel-source hash | "cpu avx2"), n_tokens, tokens: [u32; n], sampler: Option<[u64;4]+cfg>}`.
**Any header mismatch → ignore the bytes, keep the tokens, re-prefill.** A stale file is never wrong, only slow.
The backend tag exists because `rope_table`'s f64 `sincos` is "not guaranteed bit-identical" GPU vs CPU
(CODE decode.cu:306–310); the rope block exists because L2 changes every cache byte.

**API.** `POST /slots/:id?action=save|restore|erase` with `{filename}` under `--slot-save-path` (llama-server's
exact shape so existing clients work); `--cache-ram MiB` = the same image kept in pinned host RAM, LRU keyed by
token prefix, filled when a slot is evicted (= r10 M1); `axon chat --session FILE` saves on exit, restores on start.

**Gain.** 32k 7B f16 = 1.75 GiB: **69 ms** H2D at 25.4 GB/s MEASURED pinned (ESTIMATED: 1.75·2³⁰/25.4·10⁹) vs
12.1 s re-prefill MEASURED; from NVMe ≈ +0.6–0.9 s at 2–3 GB/s. 128k q8 3.7 GiB: 146 ms vs ~2 min.

**Bit-identity.** Preserved by construction: bytes are copied; attention reads only rows < n; stale rows ≥ n are
the same situation as today's slot reuse (`held[slot].truncate(reuse)`). Nothing numeric runs.

**Risk.** Low. Hazards: saving before the stream drained (use `HostBuf::bytes_mut`'s wait); the q8 scale region
offsets (reuse `kv_args_at`'s arithmetic); a pool `ctx` smaller than `n` on restore (refuse). Pinned RAM budget on a
30 GB box: cap `--cache-ram` (llama's 8192 MiB default).

**Builder-hours.** 20–30 (+8 for files and `chat --session`), as r10.

**Gate.** (1) `restore_equals_continuous`: prefill P (≥ 8k tokens), greedy-generate 64, record ids + md5 of the slot's
KV region; separately prefill P, save, drop the pool, new pool, restore into a *different* slot while other slots are
busy, generate 64 → ids and md5 equal; repeat under `--kv q8` and with a seeded sampler. (2) restore ≤ 100 ms for
32k 7B f16 on bella from RAM. (3) a header with a flipped `rope.factor` falls back to re-prefill and still produces
the continuous-run ids.

### L2 — YaRN as an explicit flag (Qwen2.5 / Qwen3 to 128k) — **second**

**The exact recipe (Qwen2.5-7B-Instruct).** `rope_type yarn, factor 4, original_max_position_embeddings 32768,
beta_fast 32, beta_slow 1, truncate true, attention_factor = 0.1·ln 4 + 1 = 1.138629` on both q and k
(logits × 1.29648). For hd 128, base 1e6, orig 32768 (ESTIMATED from the formulas in §3.1/§3.3):
`low = floor(23.596) = 23`, `high = ceil(39.651) = 40`. Pairs 0–23 (wavelength ≤ 900) keep their frequency;
pairs 40–63 (wavelength ≥ 35k, < 1 rotation in 32k) are divided by 4; 24–39 blend linearly. Context becomes 131072.
Qwen3-8B: the card says orig 32768 (not the GGUF's 40960); with 40960 the dims are 24/41 — pass `--yarn-orig-ctx 32768`
as the card does; Axon should default `orig` to the GGUF `context_length` exactly as llama.cpp does, and print it.
For 64k sessions use factor 2 (P8's advice): `low/high` unchanged, mscale 1.0693.

**Change.** `rope_freqs` (CODE model.rs:387–400) already applies a per-pair divisor (Llama 3's `rope_freqs.weight`)
— YaRN's frequency part is **the same shape**: a computed divisor vector `d_i ∈ [1, factor]` from (base, hd, orig,
β). Add it there, from a `RopeScaling` struct filled by the CLI (`--rope-scaling yarn --rope-scale N --yarn-orig-ctx M
[--yarn-attn-factor --yarn-beta-fast --yarn-beta-slow]`, llama.cpp's names) or from GGUF keys when a converter wrote
them (`rope.scaling.type = yarn`, `.factor`, `.original_context_length`, …). The magnitude scale is one multiply in
the three table builders: CPU `rope_table_of` (model.rs:1408–1418), `rope_table` (decode.cu:311–320), `rope_table_b`
(prefill.cu:49–55) — `(sin·m, cos·m)`; every consumer (`qkv_rope`, `warp_norm_rope`, `rope_kv_b`, `rope_kv_norm_b`)
reads the table unchanged. gemma3's `rope_swa` table stays unscaled. `check_supported` (model.rs:499–548) stops
refusing a factor when the flag supplies it; `ctx_len` becomes `orig × factor` (llama-context.cpp:4110–4113's rule);
`/v1/models` reports the new `n_ctx`. Pool/cache sizing is untouched (r10 §0: 128k q8 fits a 7B on 12 GB).

Alternative placement of mscale: fold m² into the attention softmax scale instead of the table — identical logits,
unscaled keys in the cache (slightly kinder to q8 ranges), but touches every attention kernel's scale constant; the
table placement mirrors llama.cpp/vLLM/HF bit-structure, so the logits comparison is tighter. Recommend the table.

**Does it change outputs below 32k? Yes — at every position ≥ 1.** Pairs 24–63 rotate by different angles at every
position, and all logits are × 1.296. md5s change with the flag on. Magnitude on the actual model is **unmeasured**;
the closest MEASURED data: non-fine-tuned LLaMA-7B static s=4 PPL +3.5 % at 2k (4.05 → 4.19) and dynamic = baseline
(P3 Table 5); Qwen's own wording for Qwen2.5/Qwen3 is "potentially impacting performance on shorter texts … not
recommended if the average context ≤ 32k" (P7, P8). Note Qwen's Table 16 row "DCA+YaRN does not change the model
behavior within 32K" is about **DCA + temperature** in their fork (identity below L by construction), **not** about
the static recipe the card gives llama.cpp/vLLM users. No public RULER number for *static* YaRN on Qwen2.5-7B was
found; the DCA numbers (82.3 @64k, 55.1 @128k) are the ceiling to compare against, not the expectation.

**Explicit flag or automatic?** Explicit, off by default, with one convenience:
- `--ctx N > trained` on an arch without scaling → an error that prints the exact flags (not a silent switch).
- `--rope-scaling yarn` without `--rope-scale` → factor = `ceil(ctx / orig)` (smallest that fits; P8's "factor 2 for
  64k"), printed at startup. Deterministic from flags, so md5s are a function of the command line.
- *Dynamic* YaRN (s = len/L per forward, the paper's recommendation for non-fine-tuned models) is **not** an option
  for Axon: every stored key's rotation would depend on the current length (P3 §3.4 says so: "cache the kv-embeddings
  before applying rotary"), so a prefix's cache is no longer a function of the prefix; re-roping the cache or storing
  un-roped keys and roping at attention time is a kernel rewrite for a benefit the explicit flag already gives.
  A chat-only "flip to static s when the session first crosses L, with one re-prefill" keeps the contract (the cache
  is then a function of (prefix, s) with s fixed for the session's life) and is the one automatic policy worth having
  — in `axon chat`, never in `serve` where `n_ctx` must be stable for the harness.

**Gain.** Qwen2.5 7B/14B/32B and Qwen3 run to 128k; 7B at 128k q8 all-GPU on bella: ESTIMATED 34 tok/s today, 65
after r9 #1 (r10 §2.4). Zero per-token cost (same kernels, one multiply in a 64-element table per position).

**Bit-identity.** Contract preserved (decode, batched prefill and verify all read the one table; the flag is global);
new bits when on, by design. Off: md5s unchanged (gate).

**Risk.** Low-medium. The f64 table vs llama.cpp's f32 accumulated θ: at pos 100k the f32 error is ~pos·ulp — llama's
is the noisier side; compare against HF/vLLM f32 too if the spread test fails. `check_supported`'s `linear`/`yarn`
strings with factor 1 (model.rs:546) must keep working. Q-ROAR (§2.5) warns quantized weights react worse to PI than
bf16 — test on the Q4_K_M, not on a card number.

**Builder-hours.** 12–20 (r10). **Gate.** (1) logits vs `llama-server --rope-scaling yarn --rope-scale 4
--yarn-orig-ctx 32768` at positions 40k and 100k within the GPU-vs-llama spread the tests use; (2) md5 suite unchanged
with the flag off; (3) passkey/NIAH on 7B Q4_K_M at 64k and 100k ≥ llama.cpp's own score with the same flags (the
honest bar: Axon matches the recipe, the recipe's quality is Qwen's claim); (4) PPL at 8k with/without the flag,
reported (not gated) so the "below 32k" cost is a number in PROGRESS.md.

### L3 — Chat truncates in halves and says so before, not after — **cheap, do with L1**

**Change.** `Chat::reply` (CODE chat.rs:584–608): when `k > 0` is needed, keep dropping until `kept(k) + user + close
+ want ≤ ctx/2` (free half the window), not the minimum. Report `dropped` **before** the turn runs (main.rs:1111–1116
prints it after) and once per episode: "forgetting the first N exchanges (M tokens); re-reading K tokens, ~S s".
**Gain.** One ~6 s re-prefill (≈16k tokens at ≥ 2682 tok/s, ESTIMATED, upper bound since shallower prefill is faster)
per ~16k tokens of new conversation instead of ~9 s per exchange. **Bits.** None (re-prefill == fresh, the contract;
`full_context_drops_oldest` tests adjust). **Hours.** 2–4. **Gate.** 40-turn script at `--ctx 2048` re-prefills ≤ 3
times; chat tests pass.

### L4 — `/compact` in chat: summarize what is about to be forgotten (opt-in, token-side, lossy)

**What chat should do at the limit instead of silently dropping.** Three honest options, in order:
(a) L3's announced half-drop (default; exact history for the kept half, nothing invented);
(b) `/compact` (manual) or `--compact auto`: before dropping, ask the model for a ≤ 400-token structured summary of
the exchanges about to go (prompt appended at the end of the still-cached history → ~50-token prefill + ~400-token
decode ≈ 5 s at 85 tok/s MEASURED-based), then rebuild `hist` = system + "[earlier conversation, summarized] …" as
the first user turn + the kept half, re-prefill (≈ 6 s). opencode's template (§3.5) is a good shape; the summary is
plain tokens, so cache == fresh prefill still holds;
(c) `/save FILE` of the transcript (text) — always available, zero model cost.
Never: K-shift/context-shift (r10 L4; stitched positions, non-reproducible cache), silent truncation (ollama's
complaint), or KV eviction (P17: cache depends on the path).
**Serve**: keep the 400; add `n_ctx`, `prompt_tokens` and `max_tokens` as structured fields in the error body so
harnesses don't guess (opencode guesses with 0.7/0.5/0.35 shrink steps). **Hours.** 6–10. **Bits.** None.

### L5 — DCA + causal temperature (identity below the trained length) — **later, if static YaRN's short-text cost is unacceptable**

**Idea.** Qwen's validated recipe (P9): keys stored with position `k mod s_chunk` (a pure function of `k`: contract
holds), queries roped three ways, attention per 64-position block selects the q variant by the block's chunk
relation (`s_chunk` and `s_local` multiples of 64 make it block-uniform except the successive-chunk window edge),
three partial (M, L, O) folds merged exactly as the split-K fold already is (decode.cu "Tensor-core attention" fold
order). Temperature as a **per-query-position** scalar `0.1·ln(max(1, (pos+1)/L)) + 1` folded into the softmax
scale: a function of the query's own position, so decode == prefill == verify and it is exactly 1 below L.
**Property.** Bit-identical to today below the chunk length (keys and queries unchanged there; choose `s_chunk − s_local` = 32768 for Qwen2.5 — Qwen's own chunk/local values for the 128k models are not published in P7/P9, only the 1M models' config is) — the thing static YaRN cannot give.
**Cost.** Every attention kernel (prefill.cu `attn_fa_b`, decode.cu `attn_split_*`, q8 twins) learns a q-variant
select; 3 q ropes per token (cheap). 40–60 h. **Gate.** md5s unchanged below 32k with the mode on; passkey at 64k/128k
vs Qwen's DCA numbers (82.3/55.1 RULER are the reference, passkey > 80 % at 1M CLAIMED).

## 5. Dead ends

- **Context shift / K-shift** (llama.cpp `--context-shift`, server-context.cpp:3273–3335; ollama's half-discard): the
  cache after a shift is not the cache a fresh prefill writes (positions moved, text stitched); llama.cpp itself
  disables it for multimodal and shared prompts. Axon's re-prefill *is* StreamingLLM's "sliding window with
  re-computation" baseline, the quality-correct one; L3 amortizes its cost to ~0 per token.
- **StreamingLLM sink + window as a mode**: streams forever but, per its authors, does not extend usable context;
  for "long sessions" the user wants the earlier turns, not stable perplexity without them.
- **Dynamic YaRN / Dynamic NTK**: needs un-roped keys or a cache re-rope on every length change (P3 §3.4). Breaks the
  prefix-function property; the explicit static flag plus a chat-only one-time flip covers the use.
- **KV eviction (H2O/SnapKV) by default**: cache depends on attention scores along the generation path, so a batched
  request ≠ the same request alone and prefill ≠ decode eviction. Opt-in only, and pointless before B1 (paged KV).
- **KV offload to host for 128k decode**: 7 GiB f16 read per token over 25 GB/s = 280 ms/token (r10 §2.4). Storage
  only (L1), never a decode path.
- **q4 KV to make 128k f16-cheap**: r10 B5's verdict stands; q8 already fits 128k on 12 GB.
- **InfLLM-style retrieval memory**: a retrieval system with its own positions (lossy); a harness/RAG concern.
- **LongRoPE/LongRoPE2 factor search**: needs a search over the model (hours of GPU) and, for LongRoPE2, training;
  the per-dim factor *table* it produces would load through the same `rope_freqs` divisor path as YaRN — worth
  accepting as a file (`rope_freqs.weight`-like) if someone publishes one for Qwen, not worth computing in Axon.

## 6. Open questions a 1-hour experiment could settle (builder, not this agent)

1. **The short-text cost of static YaRN on the real artifact**: `llama-perplexity` on Qwen2.5-7B-Instruct Q4_K_M at
   8k with and without `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`; also factor 2. Gives the number
   the cards only hint at, on the quantized weights (Q-ROAR's point).
2. **Does static YaRN actually retrieve at 64k/100k on the GGUF?** Passkey at 5 depths × {48k, 64k, 100k} with
   llama-server + the flags; factor 2 at 64k vs factor 4. Sets L2's gate (3) threshold with evidence.
3. **Save/restore timing**: `cuMemcpy2DAsync` (1 call/region) vs the per-layer loop for 1.75 GiB D2H→pinned→NVMe
   and back; confirms the 69 ms and the NVMe share.
4. **L3's amortization**: the 40-turn script at `--ctx 2048` on main vs the halves patch; count re-prefills and
   wall time.
5. **f64 vs f32 rope tables at 100k**: dump llama.cpp's and Axon's (sin, cos) at pos 100 000 for pairs 24–63;
   bounds the spread before anyone chases a "mismatch" that is llama's f32 accumulation.
