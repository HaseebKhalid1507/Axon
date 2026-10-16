# r10 scope — four product themes: bigger models, longer sessions, multi-user, concurrent requests

Strategist note (Yoru). Read-only on origin/main `e998d6f`, PR #42 (`fix/serve-default-ctx`), llama.cpp `4413a19`
(`tools/server`), the r4/r7/r9/g4 specs and PROGRESS.md. Probes: GGUF metadata of the local models (no gguf-py
deps: a 30-line reader), `free`, `nvidia-smi`. No cargo, no models run. Labels: **CODE** (file:line on `e998d6f`),
**MEASURED** (a cited log, PR or spec), **ESTIMATED** (arithmetic shown).

Hardware: bella (RTX 5070 Ti Laptop 12 GB, 11.2 GiB free at start per #42, 30 GB RAM of which ~17 GB
`MemAvailable` right now, PCIe Gen5 x8 pinned 25.4 GB/s MEASURED), avante (4050 6 GB), jade (1050 3 GB),
RunPod 4090 / 5090 / A100 / H100 on demand.

## 0. The one table everything below leans on: KV bytes per token

f16, K+V = layers × 2 × n_kv × hd × 2 B (CODE: GGUF metadata read today; r4-serve §1 for the small ones).
q8 ≈ 0.53× (int8 + an f32 scale per head, CODE `CacheKv::Q8`, qwen2.rs:192–197).

| model | weights Q4_K_M | KiB/tok f16 | 8k | 32k f16 / q8 | 128k f16 / q8 | fits 12 GB all-GPU at |
|---|---:|---:|---:|---|---|---|
| Qwen2.5 7B | 4.68 GB | 56 | 0.44 GiB | 1.75 / 0.93 GiB | 7.0 / 3.7 GiB | 32k f16 (1 slot today), 128k q8 (ESTIMATED: 4.4 + 3.7 + 0.5 scratch = 8.6 GiB) |
| Qwen3 8B | ~5.0 GB | 144 | 1.1 | 4.5 / 2.4 (40960: 5.6 / 3.0) | — | 40k q8 |
| Llama 3.1 8B | 4.9 GB | 128 | 1.0 | 4.0 / 2.1 | 16 / 8.5 | 45568 f16 (MEASURED in #42) |
| Qwen2.5 14B | 8.99 GB | 192 | 1.5 | 6.0 / 3.2 | — | 8k f16 only (MEASURED: "14B fits at 8k chat"); 32k needs a split: q8 37/48 layers 17.0 tok/s, f16 29/48 11.3 (MEASURED, PROGRESS 733) |
| Qwen2.5 32B | 19.85 GB | 256 | 2.0 | 8.0 / 4.2 | — | never; hybrid, KV of CPU layers in RAM |
| Qwen3-30B-A3B (MoE) | ~18.6 GB | 96 | 0.75 | 3.0 / 1.6 | — | experts on CPU (§1.3) |
| Llama 3.x 70B / Qwen 72B | 42.5 / 47.4 GB | 320 | 2.5 | 10 / 5.3 | — | 2×24 GB only (§1.5) |

Trained lengths (probed): Qwen2.5 7B/14B/32B 32768, Qwen3-8B 40960, Llama 3.1 8B 131072. **No Qwen2.5 or Qwen3
GGUF carries a `rope.scaling.*` key** — the YaRN-to-128k that Qwen's model card describes is an operator override in
llama.cpp (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`), not file metadata.

Long-context speed on bella, 7B (MEASURED, r9 §0): decode 94.8 tok/s at 8k, 85.8 at 16k, **72.4 at 32k** (llama.cpp
76.7); attention costs 5.3 ms of the 13.8 ms step at 32k and runs at 350 GB/s of a 648 GB/s ceiling (latency-bound,
so `--kv q8` gains nothing today: 70.9). Prefill of 32k: 12.1 s (2682 tok/s), attention ≈ 48 % of it. PR
`feat/long-ctx-decode` (sa_174) has q8 tg@32k +13.8 % MEASURED, f16 ~+2 %.

---

## 1. Bigger models

### 1.1 What exists on main

- **CPU/GPU split** (`hybrid.rs`): CPU runs the first k layers, GPU the rest and the head; one hidden-state hop per
  token (CODE hybrid.rs:1–26). Automatic placement by free VRAM with `DESKTOP_MARGIN` 640 MiB (hybrid.rs:44,
  `choose_placement` :242). KV of CPU layers lives in RAM, f16 (hybrid.rs:100–108).
- **Streamed prefill** (`cuda/qwen2/stream.rs`): for batches ≥ `STREAM_MIN` 32 (hybrid.rs:81) the CPU layers' weights are
  un-permuted from the pinned repack and double-buffered through a VRAM ring, so those layers prefill at GPU speed;
  the new KV rows are copied back to the CPU cache (stream.rs:1–29). Chunks of `STREAM_BATCH` 4096 (hybrid.rs:86).
  `AXON_STREAM_DECODE=1` streams one-token steps too: ~2× slower than the CPU computing them (hybrid.rs:88–91).
- **q8 KV** (`--kv q8`, CODE `CacheKv::Q8`, qwen2.rs:192–197; on splits too, PROGRESS 733).
- **Pipeline parallel** `--gpus 0,1` (README 141–156): bit-identical to one GPU, decode 1.00× one 4090, prefill
  1.46× (7B) / 1.55× (32B) on 2×4090 (MEASURED, PROGRESS 888). `serve` on a pipeline is one request at a time
  (CODE serve.rs:687–697). P2P untested. TP: all-reduce loop 2.24 ms/token on no-P2P 4090s, over the 2 ms gate →
  not for PCIe (MEASURED, PROGRESS 888; vLLM TP2 32B AWQ 65–70 tok/s vs axon 43).
- **MoE refused** with an error (README 23–24; `check_supported`, model.rs:499–548 refuses RoPE scaling too).
- Serve's context default for a split is `min(trained, 8192)` (#42: `split_ctx`); a long cache there takes layers.

MEASURED on bella, 32B Q4_K_M hybrid (PROGRESS 681–734): pp560 328–390, pp2100 ~407 tok/s streamed (llama.cpp
`-ngl` 211 / 264–286), decode 4.79 tok/s (llama.cpp 3.66); stream off: pp560 56, pp2100 47. 14B as in §0.

### 1.2 What users hit

- A 12 GB card and a 14B: fine at 8k, but an agent needs 32k, and at 32k the model becomes a split and decode drops
  from an ESTIMATED ~60 tok/s all-GPU (8.99 GB / 0.93 × 648 GB/s ≈ 15 ms) to **17 tok/s** (MEASURED, q8 37/48).
  The cause is a 3.2 GiB cache **reserved** for a 32k slot while an agent turn uses 10–25k.
- 32B on 12 GB works and beats llama.cpp, but 4.8 tok/s is a reading speed, not an agent speed. The CPU half
  (36 of 64 layers, ~11 GB of weights over ~75 GB/s RAM) bounds it: ESTIMATED floor 11 / 75 ≈ 150 ms per token.
  No kernel work changes that; only fewer CPU bytes per token do (MoE, or a second GPU).
- MoE users (Qwen3-30B-A3B, gpt-oss-20b) get a refusal; llama.cpp's `--cpu-moe` / `-ot exps=CPU` made "experts on
  CPU, attention on GPU" the default way to run these on 8–16 GB cards.
- 24 GB × 2 pod users want 70B. It already runs (PP), but no one has measured it.

### 1.3 What llama.cpp / ollama / vLLM do, and where Axon can win

- llama.cpp: `-ngl N` split (CPU first layers), op-offload for batches ≥ 32 (copies CPU weights per op; the idea
  behind Axon's streamer, g4-offload Fable note), `-ot <regex>=CPU` / `--cpu-moe` / `--n-cpu-moe N` / `--n-cpu-ffn N`
  (CODE llama.cpp `tools/server/README.md:80–84`), `-ctk/-ctv` in q8_0 / q4_0 / q5 / iq4_nl. ollama: `num_gpu`,
  `OLLAMA_KV_CACHE_TYPE q8_0|q4_0`, no expert offload control. vLLM: no CPU offload worth using on consumer boxes.
- **Where Axon wins today:** streamed prefill (32B pp2100 407 vs 286) and split decode (4.79 vs 3.66), both MEASURED.
- **Where Axon can win on MoE:** llama.cpp runs the CPU experts' prefill on the CPU or per-op offload; Axon's
  streamer already double-buffers whole layers' weights through a VRAM ring under compute. Streaming the experts
  per prompt chunk makes MoE prefill GPU-speed: 17.5 GB of experts at 25 GB/s = 0.7 s per 4096-token chunk
  (ESTIMATED ≈ 3000–5000 tok/s ceiling vs llama.cpp's CPU-expert prefill of a few hundred). Decode is RAM-bound on
  both: 30B-A3B reads 8 of 128 experts × 48 layers ≈ 1.0 GB per token (ESTIMATED from the shapes: 3 × 2048 × 768
  params × 8 experts × ~0.56 B × 48) → 13 ms at 75 GB/s + ~1 ms GPU attention + 96 PCIe hops ≈ 1.5 ms (hybrid's
  hop is ~10 µs MEASURED on the pipeline, README 152) → **ESTIMATED 35–55 tok/s**, against llama.cpp's reported
  20–30 on comparable DDR5 laptops (not measured here; measure it in the round).

### 1.4 Rounds (ranked within the theme)

| # | round | gain (basis) | bits risk | effort (builder-h) | gate | depends on |
|---|---|---|---|---|---|---|
| B1 | **Paged KV for the pool** (block table per slot; a slot takes pages as it grows; `--ctx` is a cap, not a reservation) — shared with §4 | 14B agent on 12 GB stays **all-GPU** at 10–25k used tokens (2.0–4.7 GiB f16 vs 6 GiB reserved): ESTIMATED decode 17 → ~55 tok/s (3×). 7B: ~7 agents at 15k each on one card instead of 1–2 slots (§4) | **none by construction**: paging changes addresses of 64-position blocks, never their order (r4 §4.2, decode.cu "Tensor-core attention") — but it is the broadest kernel touch since S2 (every cache-indexing kernel in prefill.cu/decode.cu and the q8 twins) | 40–60 | `serve_identity` with shuffled page tables == contiguous, token- and KV-byte-exact; 14B `--ctx 32768` on bella stays all-GPU at a 15k agent prompt and decodes ≥ 45 tok/s | #42 merged, C2 (drop the N+1 cache) first |
| B2 | **MoE: Qwen3-30B-A3B** — router + top-k on the GPU, attention/norms/shared on the GPU, experts on the CPU for decode (new CPU expert-GEMV over the repacked layout, 8 experts gathered, fixed summation order), experts **streamed** through the GPU for prefill (the existing streamer, per expert matrix) | 30B-class quality on 12 GB at ESTIMATED 35–55 tok/s decode, 3000+ tok/s prefill (§1.3) | **medium**: a new architecture (`moe` block, `ffn_gate_inp`, `ffn_*_exps` 3-D tensors), per-layer device mixing (today the split is a single cut), and a top-k that must be an exact select (ties by index) so decode == prefill. Contract holds per device as hybrid's does | 80–120 (two weeks, as the roadmap said) | logits vs llama.cpp within its own CPU/CUDA spread on 30B-A3B; `feed_batch == feed` on the MoE layers; chat cache reuse == fresh prefill; bench vs `llama-server --cpu-moe` (decode, pp560, pp2100) on bella | none on serve; the streamer; not before Gemma 3 settles (shared `check_supported`) |
| B3 | **70B / 72B on 2×24 GB: measure, document, fix what breaks** (Llama 3.3 70B Q4_K_M 42.5 GB → ~4 GiB left for KV on 2×4090 = 12k f16 / 24k q8; 72B Q3_K_M 37.7 GB) | A documented, measured path. ESTIMATED 70B decode ≈ 42.5 / (0.93 × 1008) + 3 ≈ 48 ms → ~21 tok/s on 2×4090 | none (PP is bit-identical by construction) | 4–8 + a $5 pod | tg128/pp2100 at 2 cuts, md5 vs 1-GPU on A100-80 (fits in one), `--kv q8` at 16k | PP on main (done) |
| B4 | **Multi-GPU stage 1b**: the pool over a pipeline, two step batches in flight — shared with §4 | 32B serve on 2×4090 ≈ 2× one GPU's aggregate at 8+ clients (ESTIMATED, design/multi-gpu.md §3.1); 27 concurrent 4k contexts instead of 8 (KV room) | none (same kernels; the hop copies f32) | 40–60 | pod: 32B C=8 aggregate ≥ 1.7× one 4090; `serve_identity` on a 2-stage one-device pipeline | S2 pool (done), B1 optional, C1 (chunked prefill) for the micro-batches |
| B5 | **q4 KV experiment** (K q8 / V q4 first, as the community runs llama.cpp) | Only one 12 GB case needs it: 14B at a reserved 32k slot all-GPU (3.2 → ~1.8 GiB). B1 solves that case without a quality loss, and q8 isn't even faster yet (r9 fact 2) | **high on quality**, new identity tuple; no speed until decode attention is bandwidth-bound | 16–24 | KL vs f16 on 14B at 8k/32k ≤ 2× q8's; tg@32k gain ≥ +10 % over q8 after r9 #1 | r9 #1 (long-ctx decode) landed; after B1 — likely **not a round** |
| B6 | gpt-oss-20b (MXFP4 tensor type, attention sinks, alternating 128-sliding-window) | Second most-asked MoE | medium (new quant + sinks in every attention kernel) | 60–90 | as B2 | B2, Gemma 3's SWA |

The "obvious" play is q4 KV ("half the cache again"). It is the wrong one on this hardware: the 14B user's problem
is a *reserved* 32k slot, and B1 fixes that for every model at zero quality cost, while B5 buys 1.4 GiB on one
model and a third numerics tuple to test.

---

## 2. Longer sessions

### 2.1 What exists on main

- Contexts up to the trained length: `--ctx` ≤ `ctx_len` (serve.rs:653–659; main.rs:874–882 for chat, default
  `min(trained, 8192)` = `DEFAULT_CHAT_CTX`, main.rs:775). #42 makes serve's default the trained length where it fits.
- RoPE: base from the GGUF; Llama 3's `freq_factors` (model.rs:384); Gemma 3's linear scale; **any other scaling
  factor ≠ 1 is refused** (model.rs:499–511, 535–548: "linear"/"yarn" type strings are accepted only with factor 1).
- Prefix reuse: serve keeps the previous request's cache and feeds only the new suffix (serve.rs:478–495 in the
  solo path; engine.rs:104–134 per slot). MEASURED: an agent loop's turn 2 fed 298 of 4705 tokens (r9 §0).
- Chat's window: `Chat::reply` forgets the oldest exchanges until `kept(k) + user + close + want ≤ ctx` with `want =
  max_reply.min(ctx/4)` (chat.rs:571–607); `drop_oldest` shifts positions, so **everything after the system block
  is fed again** (chat.rs:506–532, `self.fed = self.fed.min(self.sys)`). Exact (re-prefill == fresh, the contract), but
  at a 32k 7B window that is a ~9 s re-prefill of ~24k tokens — and `k` is minimal, so it happens on nearly every
  exchange once the window is full.
- No save/restore of any cache; a restart re-prefills. `/v1/models` reports `n_ctx` (#42) so harnesses can plan.

### 2.2 What users hit

- An agent at 30–40k on a 32k model: a 400 `context_length_exceeded`. All the harnesses named (opencode, pi, codex,
  crush, Synaps) compact their own history on that error; the server's job is the honest 400 and a correct `n_ctx`.
  What they want next is **more than 32k** (Qwen2.5 "128k with YaRN"; Qwen3 40k → 128k the same way).
- A chat user at hour three: the first messages vanish (expected) and, in `axon chat`, the turn that crosses the
  window pauses ~9 s (unexpected).
- Anyone restarting the server loses every slot's prefix: the next agent turn on a 25k context is a 9–12 s TTFT
  (MEASURED prefill rates: 7B 2682 tok/s at 32k depth).

### 2.3 What llama.cpp / ollama / vLLM do

- **YaRN**: ggml's rope applies the YaRN ramp (`beta_fast` 32, `beta_slow` 1, `ext_factor`, `attn_factor` = 0.1·ln(s)+1
  magnitude scale) from CLI flags or from GGUF keys when a converter writes them (`tools/server/README.md:59–67`).
  Qwen's GGUFs don't carry them (§0), so llama.cpp users pass the flags; a 7B then runs to 131072.
- **Context shift** (`--ctx-shift`, server-context.cpp:3277–3310): on overflow discard `n_left/2` tokens after the
  `n_keep` prefix and **shift the K cache in place by a RoPE delta**. No re-prefill, but the shifted cache is not the
  cache a fresh prefill would write (positions moved; the model sees a stitched history). llama.cpp's own README
  warns `cache_prompt` is not bit-reproducible across batch sizes on most backends (README.md:589).
  `--cache-reuse N` extends this to prompt edits (shift chunks ≥ N). ollama truncates (`num_ctx`, keeps the system
  prompt) and otherwise relies on the same shift.
- **Persistence**: `POST /slots/:id?action=save|restore|erase` to a file under `--slot-save-path`
  (README.md:1150–1195; 14 MB for 1745 tokens in their example = a 7B-class cache, 50 ms save / 43 ms restore);
  `--prompt-cache FILE` for the CLI; `--cache-ram N` (default 8192 MiB) keeps evicted slots' caches in host RAM
  (server-context.cpp:1492–1502, 1751–1800: a slot about to lose > 50 % of its context is saved, LRU).
- vLLM: prefix caching by hashed blocks, no context shift, no session files.

**Where Axon can win:** exactness. Axon's cache bytes are a pure function of the token prefix (the contract:
batched == one at a time, chat reuse == fresh prefill). So a saved cache restored into any slot, on any identical
GPU, after any restart, continues the conversation **bit for bit** — llama.cpp cannot promise that, and context
shift breaks it on purpose. Don't build a K-shift; build save/restore and make truncation cheap.

### 2.4 Cost model, 7B on 12 GB (ESTIMATED from MEASURED parts)

| context | KV f16 / q8 | decode step (today) | after r9 #1 + q8 | prefill from empty |
|---|---|---|---|---|
| 32k | 1.75 / 0.93 GiB | 13.8 ms (72 tok/s, MEASURED) | ~10.2 ms (~98) | 12.1 s (MEASURED) |
| 64k | 3.5 / 1.9 | 8.5 + 2×5.3 = 19 ms (52) | 8.5 + 2×1.7 = 12 (83) | GEMM 2× + attention 4×: ~12.6 + 23 ≈ 36 s |
| 128k | 7.0 / 3.7 (q8 only all-GPU) | 8.5 + 4×5.3 = 30 ms (34) | 8.5 + 4×1.7 = 15 (65) | ~25 + 93 ≈ **2 min** |

Attention scales with the context per row and is paid per row (never shared across slots), so 128k sessions are a
one-at-a-time affair on 12 GB and the 2-minute prefill is paid once — which is why persistence (L1) ranks above
YaRN (L2): without save/restore, a 128k session is lost at every restart and re-costs 2 minutes.

Paged or offloaded KV for 128k on 12 GB: decode attention reads the whole KV every token (7 GiB f16 at 128k); over
PCIe at 25 GB/s that is 280 ms per token. Offloading KV to host is not a decode option; it is a *storage* option
(L1). q8 on the GPU is the 128k option, and it fits.

### 2.5 Rounds

| # | round | gain (basis) | bits risk | effort | gate | depends on |
|---|---|---|---|---|---|---|
| L1 | **KV save/restore**: `BatchLm::save_slot / load_slot` (f16/q8 bytes as stored, per layer, to pinned host RAM); an LRU host store keyed by token prefix, `--cache-ram MiB` (llama-server's name); `POST /slots/:id?action=save|restore|erase` to files (llama-server's shape, so existing clients work); `chat --session FILE` save on exit / restore on start. Shared with §3 (M1 is the eviction half) | A 32k 7B slot restores in ESTIMATED 75 ms from RAM (1.75 GiB at 25.4 GB/s pinned MEASURED) or ~0.6 s from NVMe, against a 12 s re-prefill; 128k: 150 ms vs 2 min | **none**: bytes are copied; a restored slot's next token equals the uninterrupted run's. Test it exactly that way | 20–30 (+8 for files and `chat --session`) | restored-slot tokens == continuous-run tokens, KV bytes equal; restore ≤ 100 ms for 32k 7B f16 on bella; survives `--kv q8` | S2 pool (done); C1 (chunked prefill) merges first (both edit `admit`) |
| L2 | **YaRN as a CLI override** (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx N`, llama.cpp's flag names and ramp constants; also read the GGUF keys when present), CPU + GPU rope tables, `ctx_len` raised to `orig × factor` | Qwen2.5 7B/14B/32B and Qwen3 to 128k; on 12 GB a 7B at 128k q8 fits (§0) and decodes at ESTIMATED 34 tok/s today, 65 after r9 #1 | **none for the contract** (a deployment flag; decode == prefill since both read one table), **new bits** when on; the magnitude scale touches q and k before attention (one more multiply in `rope_*` kernels and the CPU path) | 12–20 | logits vs llama.cpp `--rope-scaling yarn` at positions 40k / 100k within the GPU-vs-llama spread the tests already use; needle-in-haystack at 64k and 100k passes on 7B; md5s unchanged with the flag off | r9 #1 (so 128k decode is usable); L1 (so a 128k session is worth having) |
| L3 | **Chat's window slides in halves**: when the window is full drop exchanges until ≥ ctx/2 is free, not the minimum; show it once | One ~9 s re-prefill per ~16k tokens of conversation instead of per exchange (ESTIMATED from the 2682 tok/s rate) | none (re-prefill == fresh, by contract; `full_context_drops_oldest` tests adjust) | 2–4 | the existing chat tests at a small ctx; a 40-turn script at `--ctx 2048` re-prefills ≤ 3 times | none |
| L4 | **Not a round: context shift / K-shift.** | — | breaks "cache reuse == fresh prefill" by design | — | — | Decided: harnesses compact; chat truncates in halves (L3); the 400 is honest |

---

## 3. Multi-user

### 3.1 What exists on main

- Pool of `--parallel` slots (default 8), each a `ctx`-position KV region of one cache (slots.rs:1–16, 36–58);
  `pool_for` shrinks slots then context to fit (serve.rs:828–853); #42 replaces that with `plan_slots`.
- Admission: FIFO off a bounded channel (`--max-queue` 128, 503 when full); the first waiting request that fits a
  slot is admitted, its **whole prompt one step of its own** (engine.rs:120–148 `admit`, serve.rs:783–795).
- Slot choice: the free slot with the longest common prefix (engine.rs:104–118); a slot that loses is **overwritten**
  — its prefix is gone (engine.rs:129–134 `held[slot] = p[..last]`). No host copy.
- A request that fits no slot (prompt + max_tokens > slot ctx) runs alone "once the batch is idle" (serve.rs:797–810).
  **Under steady load that is never** — a starvation path for long requests (the S2 critic's HOL note, PROGRESS 889).
- Cancel on disconnect (engine.rs:167–178), `--max-tokens`, `--max-conns` / `--max-queue`, `--cors`, `GET /health`
  (503 until loaded). A `generated` counter (`AtomicU64`, engine.rs:66) and `Server::waiting()` / `conns()`
  (serve.rs:893–900) exist but no endpoint reports them. No auth: "put a reverse proxy in front" (README 274).
- Per-request log line with prompt/cached/decode timings (engine.rs:293–311).

### 3.2 What users hit

- A Synaps-style swarm: 6–12 agents, each a conversation of 15–30k tokens, against a 12 GB 7B. Slots: **1 today at
  32k f16** (#42 e2e), 3 with q8. Every agent turn beyond the slot count lands on a slot holding someone else's
  prefix → a 5–12 s re-prefill (MEASURED rates) and, worse, it evicts *that* agent's prefix in turn: the swarm
  thrashes. (The shared system prompt + tool schemas, 15–25k tokens, is re-prefilled per agent too — see C5.)
- Whoever exposes serve on a LAN gets no auth and no view of what the server is doing (`/health` only).
- A chatty client with `max_tokens` unset can hold a slot for a whole context's worth of generation; one with a huge
  prompt can't be refused by size, only by `--ctx`.

### 3.3 What llama.cpp / ollama / vLLM do

- llama-server: `-np N` slots, unified or per-slot KV (`--kv-unified`, `--kv-unified-per-slot`), slot selection by
  LCP similarity else LRU (server-context.cpp:1720–1775), **`--cache-ram`** saves an evicted slot's cache to host
  RAM and reloads on a prefix match (§2.3), `--api-key` (Bearer), `/metrics` Prometheus, `/slots` (queue and slot
  state), `/props`, per-request `n_keep`, `t_max_predict_ms`, `--slot-prompt-similarity`. ollama: `OLLAMA_NUM_PARALLEL`,
  `OLLAMA_MAX_QUEUE`, keep-alive, no auth, no metrics. vLLM: `--api-key`, `/metrics` (rich), request priority, LRU
  prefix cache with block sharing, preemption by recompute or swap to CPU (`--swap-space`).
- **Where Axon can win:** restore is exact (§2.3), and its cost on this box is 75 ms per 32k slot. llama.cpp's
  `--cache-ram` carries the same ~bytes but a restored slot there isn't guaranteed to continue identically. The
  swarm case is where it matters: eviction stops being a 12 s penalty and becomes a 0.1 s one.

### 3.4 Rounds

| # | round | gain (basis) | bits risk | effort | gate | depends on |
|---|---|---|---|---|---|---|
| M1 | **Host-RAM KV on eviction** (L1's store, driven by `admit`): before a slot's prefix is overwritten, save it if the store has room (`--cache-ram`, default a quarter of `MemAvailable`, LRU by last use); on admit, longest prefix over free slots ∪ store; restore then feed the rest | A 12-agent swarm on a 3-slot 7B q8 server: per-turn prefill ESTIMATED 5–12 s → 0.1 s + the new suffix (~300 tokens, 0.15 s). 17 GB `MemAvailable` holds ~9 × 32k 7B f16 caches or ~18 q8 | none (bytes copied; test = restored == continuous) | 20–30 (the L1 store) | `serve_load.py`-style run: 12 clients, 20k shared-ish prompts, 3 slots: p50 TTFT ≤ 1 s vs ≥ 6 s on main; identity suite with forced eviction | C1 merged (admit changes) |
| M2 | **Serve ops pack**: `--api-key` (Bearer, constant-time compare; `/health` stays open), `GET /metrics` (Prometheus text: requests, generated tokens, prompt/cached tokens, queue depth, slots busy/free, store hits, TTFT/ITL histograms), `GET /slots` (per-slot held tokens, busy, request id, age), `--max-prompt-tokens`, `--slot-idle-ms` (a paused client frees its slot), request `priority` accepted and ignored for now | What a LAN deployment needs on day one; llama-server parity on the operational surface | none | 12–20 | fuzz the new routes with `fuzz_serve.py`; `curl -H 'Authorization: Bearer'` tests; metrics scrape parses with `prometheus_client` | none |
| M3 | **Fairness fixes**: (a) aging for requests that fit no slot — after `--max-wait-ms` stop admitting, drain, run it alone, resume; (b) admission order shortest-prompt-first among the waiting when the batch is prefill-bound, FIFO otherwise; (c) when `--parallel` slots all hold live prefixes, prefer evicting the slot with the oldest `last_used`, not the lowest index (engine.rs:117 `Reverse(s)`) | No request starves; TTFT p99 drops under a mixed agent load (ESTIMATED; measure with `bench_serve.py` scenario 4) | none | 8–12 | a load test with one oversize request among 8 agents completes it within `max_wait`; identity unchanged | C1 (chunked prefill makes (b) mostly moot: the stall it avoids is the one C1 removes) |
| M4 | **Not a round: multi-model serving** (ollama-style load/unload). | — | — | — | — | one process per model; a proxy routes |

---

## 4. Concurrent requests

### 4.1 What exists on main

- S2-lite: one step = one row per running request, `Place::Rows` per-row positions, attention per query over its
  slot (slots.rs:1–16), per-row sampling with the step's largest k (slots.rs:17–22), CUDA graphs keyed by (rows,
  greedy, sampled) with an LRU of 64 (slots.rs:40–54, 222–230). GEMMs: `gemvm8` / `gemvm16` read the weights once per
  step for nb ≤ 16 (`SMALL`, batch.rs:25, 594–601), `gemm2` tensor cores above.
- **N+1 caches**: `Engine.cache = net.new_cache(ctx)` (serve.rs:679) beside the pool; the S1 path runs the first
  request there with graphs and drafting and `adopt` copies its rows into a slot when a second arrives
  (engine.rs:150–165, serve.rs:801–809). #42 shrinks that cache to a slot's length; it still costs a full slot.
  PROGRESS 954: dropping it lets 7B f16 run 2 × 32k on 12 GB beside the q6k8 copy.
- Whole-prompt prefill as its own step (engine.rs:120–133): every other row waits for it.
- Chunked prefill (r9 #3b, sa_178) in flight on `feat/serve-chunked-prefill` (no commits yet at `d1c2c78`).
- Drafting (`--draft`, `--draft-model`) only on the solo path (engine.rs:10–14); batched steps never speculate.
- Hybrid, pipeline and CPU backends: one request at a time (`new_pool` errs, lm.rs:560–565).

MEASURED (PROGRESS 873, 926): 4090 7B C=1 171 tok/s (= main), **C=8 740 (4.3×)**, C=16 1017 (5.9×); bella 7B C=16
355 (4.4× of 80.1), 0.5B C=16 2978. r6's estimate was 5.5× at C=8, 8.5× at C=16.

### 4.2 Why C=8 got 4.3× and not 5.5× (ESTIMATED decomposition, to be confirmed by one profile)

The 5.5× came from "an nb-row step costs 1.45× a forward" (the nb=8 verify ratio MEASURED on bella's dp4a path)
and counted decode steps only. At C=8 on a 4090: 740 tok/s = 8 rows per 10.8 ms, i.e. 1.85× the 5.85 ms forward.
Three pieces:

1. **Prefill steps in the wall clock.** `bench_serve.py` is closed-loop: when a request ends (128 tokens ≈ 1.4 s at
   C=8) its replacement's 560-token prompt is a step of its own (≈ 40 ms on a 4090 at 13.8k tok/s). Eight of those
   per 1.4 s ≈ 0.32 s → **~18 % of wall** with every decode row stalled. That alone turns 5.5× into ~4.5×.
2. **gemvm8 isn't 1.45× on a 4090.** The 1.45 was bella (dp4a, 648 GB/s, 46 SMs). On a 4090 the forward is faster
   (1008 GB/s) but the 8-row epilogue and per-row attention + head (`argmax_b` / `samp_tail` per 16-row `HEAD_SUB`
   slice) are not bandwidth work; ESTIMATED 1.5–1.6× there. Profile it (`cuda_timeline` on a C=8 step).
3. Host work between steps: `step()` syncs, copies rows in, reads picks out, the host samples (engine.rs:213–246);
   at 10.8 ms steps that is a few percent, not the gap.

So chunked prefill (C1) is a *throughput* round too, not only an ITL one: it doesn't remove prefill compute, but it
removes the stall of the other rows during it (ESTIMATED +10–15 % aggregate at C=8 closed-loop). The remaining gap
is the kernel ratio in (2), which a batched-decode profile should settle before anyone tunes `gemvm`.

### 4.3 Long context changes the picture: batching is attention-bound there

Attention is per row and never amortized across slots. 7B at 32k: GEMV ~8.5 ms (shared by all rows at nb ≤ 16)
+ 5.3 ms **per row** (MEASURED single-row cost): C=4 ≈ 8.5×1.3 + 4×5.3 ≈ 32 ms per step → 125 tok/s aggregate,
31 per stream; C=8 ≈ 53 ms → 150 aggregate, 19 per stream (ESTIMATED). After r9 #1 brings the row to ~3.3 ms
(and ~1.7 with q8): C=8 → 37 ms / 216 tok/s, or 25 ms / 320 tok/s. **The long-context decode attention round is the
concurrency round for agents**, ahead of any scheduler work; it is in flight (sa_174, q8 +13.8 % MEASURED).

### 4.4 Max slots at 32k on 12 GB, 7B (ESTIMATED from §0; 11.2 GiB free, 4.36 GiB weights, ~0.5 GiB scratch, 1.0 GiB q6k8 copy optional)

| | f16 (1.75 GiB/slot) | q8 (0.93 GiB/slot) |
|---|---|---|
| today, N+1 caches, with the q6k8 copy | 1 slot (MEASURED #42) | 3 slots (MEASURED #42) |
| C2: no +1 cache | 2 (3 without the q6k8 copy) | 5 |
| B1: paged, agents at ~15k used | ~7 conversations (6.3 GiB / 0.82 GiB) | ~14 |
| B1 + C5 shared system prefix (20k shared, 5k private each) | ~20 | ~40 |

### 4.5 What llama.cpp / vLLM do

llama-server: `-np N` with a unified KV pool, `-ub` chunked prompt batches mixed with decode (`-cb`), speculative
decoding per slot (`--draft-max`), no bit-identity promise across batch sizes (README.md:589). vLLM: paged attention
with block sharing, chunked prefill on by default, speculative decoding in batches (n-gram, EAGLE), `--max-num-seqs`.
**Axon's standing edge:** batched == solo bits, already paid for (r4 §6), and a single-stream lead of 5–46 % (README
tables). It loses today on slot count (fixed reservations, N+1) and on the whole-prompt stall.

### 4.6 Rounds

| # | round | gain (basis) | bits risk | effort | gate | depends on |
|---|---|---|---|---|---|---|
| C1 | **Chunked prefill** (in flight, sa_178): a waiting prompt enters ≤ 512 tokens per step (`feed_slot` at a position), interleaved with the decode rows | ITL during a 20k admit: a 7 s stall (20k / 2682) → steps of ≤ ~0.2 s (ESTIMATED from the 512-token pass ≈ 0.11–0.19 s); aggregate +10–15 % at C=8 closed-loop (§4.2 item 1) | **low**: rows never mix; a chunk is `feed_slot` at a position, which is `feed_batch`'s contract (any chunking gives the same rows) | 30–50 | `bench_serve.py` scenario 4: one 20k admit during 4 decoding clients, ITL p99 ≤ 2× p50; identity suite with random chunk sizes 16..512 | #42 (serve.rs pool planning) |
| C2 | **Drop the N+1 cache**: the solo path runs in slot 0's memory (a `CudaCache` view at the slot's base: same pointers, so graphs and drafting keep working and `adopt` becomes a bookkeeping no-op), or — simpler — no solo path once a pool exists and the first request takes a 1-row step (loses CUDA graphs' ~5 % on 7B and the drafter at C=1; measure which) | 7B f16 on 12 GB: 1 → 2–3 slots at 32k; q8 3 → 5 (ESTIMATED §4.4). Every card gains one slot's worth | none (the view aliases bytes; the no-solo variant is S2's contract) | 8–16 | `serve::` suite; `serve_check.py` C=1 within noise of `gen`; a 7B f16 `--ctx 32768` server on bella reports ≥ 2 slots | #42 |
| C3 | = B1 **paged KV** | §4.4: slots by tokens used | none by construction (addresses only) | 40–60 | as B1 | C2 |
| C4 | **Batched speculative decoding**: draft-model rows for every running request in one verify step (`verify_cands` rows per slot), budget policy: K such that rows × (K+1) ≤ 16 (`SMALL`, the gemvm regime where extra rows are ~free), off above | At C ≤ 4 with a draft: per-stream ESTIMATED +30–50 % (the solo gains were +37–63 % greedy MEASURED; the verify rows ride the same weight pass); at C ≥ 8 nothing (policy off) | **none**: lossless by the verify contract (`verify_cands` rows == decode rows) | 30–50 | tokens with and without drafting equal per request under load; C=4 7B greedy code prompts ≥ +25 % per stream, C=16 aggregate unchanged | C1 (step scheduling), the draft model on main (done) |
| C5 | **Shared-prefix pages** (copy-on-write over C3): a system prompt + tool schemas held once, every agent's slot points at the same pages until it diverges | A 12-agent swarm with a 20k shared prefix: 12 × 20k × 56 KiB = 13 GiB of KV collapses to 1.1 GiB; prefill of the shared part paid once per server lifetime (ESTIMATED) | none (pages are read-only until copied; order unchanged) | 40–60 | identity with shared vs private pages; swarm load test TTFT p50 ≤ 0.3 s after the first agent | C3 |
| C6 | **Batched-decode profile** (half a day, no code): `cuda_timeline` on a C=8 step on bella and a 4090; where the 1.85× goes (§4.2 item 2); whether `attn_blk_b` per query or the head slices dominate | Decides whether a `gemvm` or head round exists at all | — | 4–6 | a table: per-kernel µs at nb 1 / 8 / 16 | none; do it inside C1's bench time |

Not rounds: **"more slots via a smaller default ctx"** (#42 decided context first, and agents need it); **prefill /
decode disaggregation on one GPU** (nothing to disaggregate onto); **tensor parallel on PCIe** (MEASURED 2.24 ms of
all-reduce per token, over the gate).

---

## 5. Merged ranking — top 10 rounds

Order: what unblocks an agent swarm on a 12 GB card first; exact bits always; cheap-and-wide before deep.

| rank | round | theme(s) | gain (basis) | bits risk | builder-h | status |
|---|---|---|---|---|---|---|
| 1 | **C1 Chunked prefill** | concurrent, multi-user | 20k admit: 7 s stall → ≤ 0.2 s steps; +10–15 % aggregate at C=8 (ESTIMATED) | low | 30–50 | in flight (sa_178) |
| 2 | **C2 Drop the N+1 cache** | concurrent, bigger | 7B f16 32k on 12 GB: 1 → 2–3 slots; q8 3 → 5 (ESTIMATED) | none | 8–16 | planned; after #42 |
| 3 | **r9 #1 Long-context decode attention** | concurrent, longer | tg@32k 72 → ~85 f16, ~98 q8; C=8 at 32k 150 → 216–320 tok/s (ESTIMATED; q8 +13.8 % MEASURED on the branch) | low | 18–28 | in flight (sa_174) |
| 4 | **L1/M1 KV save/restore + host-RAM store on eviction** | multi-user, longer | eviction 5–12 s → ~0.1 s (ESTIMATED from 25.4 GB/s MEASURED); sessions survive restarts, bit for bit | none | 20–30 (+8) | planned |
| 5 | **B1/C3 Paged KV** | bigger, concurrent, multi-user | 14B agents all-GPU on 12 GB (17 → ~55 tok/s ESTIMATED); ~7 7B agents per card; ends the oversize-request starvation | none by construction, broad kernel touch | 40–60 | planned |
| 6 | **M2 Serve ops pack** (`--api-key`, `/metrics`, `/slots`, request caps, idle timeout) | multi-user | LAN-ready; llama-server operational parity | none | 12–20 | planned |
| 7 | **B2 MoE: Qwen3-30B-A3B**, experts on CPU, streamed for prefill | bigger | 30B-class on 12 GB at ESTIMATED 35–55 tok/s decode, 3000+ tok/s prefill | medium (new arch, exact top-k) | 80–120 | planned |
| 8 | **L2 YaRN override** (+ L3 chat halves, 2–4 h) | longer | Qwen2.5/Qwen3 to 128k; 7B 128k q8 fits 12 GB, 34 → 65 tok/s after rank 3 (ESTIMATED) | new bits when on; none off | 12–20 | planned |
| 9 | **B4 Multi-GPU stage 1b** (pool over the pipeline) + **B3** 70B pod measurement | bigger, concurrent | 32B serve 2×4090 ≈ 2× aggregate (ESTIMATED); 70B at ~21 tok/s documented | none | 40–60 (+8) | design done (axon `docs/design/multi-gpu.md` §6) |
| 10 | **C4 Batched speculative decoding** | concurrent | +30–50 % per stream at C ≤ 4 with a draft (ESTIMATED from MEASURED solo gains); off at C ≥ 8 | none | 30–50 | planned |

Below the line: C5 shared-prefix pages (after 5; the swarm's real memory win), M3 fairness (after 1), C6 profile
(inside 1's bench), B5 q4 KV (likely not a round), B6 gpt-oss-20b (after 7).

## 6. Dependency graph

```
#42 serve default ctx ──┬──► C1 chunked prefill ──┬──► L1/M1 save/restore + host store ──► L2 YaRN 128k
                        │                         ├──► C4 batched speculation               ▲
                        │                         └──► M3 fairness                          │
                        └──► C2 drop N+1 cache ──► B1/C3 paged KV ──► C5 shared-prefix pages │
                                                        │                                   │
r9 #1 long-ctx decode attention ────────────────────────┼──► B5 q4 KV experiment (if ever) ──┘
                                                        │
PP on main ──► B3 70B pod measure ──► B4 stage 1b ◄── S2 pool (done); C1 for micro-batches
streamer on main ──► B2 MoE (after Gemma 3 settles) ──► B6 gpt-oss-20b
M2 ops pack: no dependencies
```

## 7. What to watch

- **C1 and L1 both rewrite `admit`.** Land C1 first; L1's store hooks the eviction point C1 defines.
- **C2's "view" variant vs "no solo path":** the solo path is where CUDA graphs and the draft model live. If the
  view is awkward, measure what C=1 loses without them (7B: ~5 % graphs; the drafter is +37 % greedy) before
  choosing — C4 brings drafting back to the pool later anyway.
- **B1 is the kernel-wide change.** Keep the old entry points as thin wrappers with an identity page table (r4 §9.1)
  so the CLI path and its md5 suite don't move; the identity table *is* the test.
- **Thermal honesty on bella:** every concurrency number here is a 4090 or a cool-gated bella run; sustained C=16
  on a 120 W laptop will throttle. Report SM clocks with serve benches (r4 §9.5).
- **The research tracks** (PROGRESS 957–963) map onto this: (3) KV memory → B1/C5, (4) big models on small VRAM →
  B2, (5) chunked prefill / scheduling → C1/M3, (6) long sessions → L1/L2. Point them at these rounds' gates.
