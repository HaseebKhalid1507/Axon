# Roadmap

Short and opinionated. "On main" means merged at `e998d6f`. Everything else is planned or in flight.
Estimates come from the design notes and are labeled.

## The thinking behind the order

Single-stream decode on a 7B/8B model is near the hardware limit. On Parabellum (RTX 5070 Ti Laptop),
7B decode GEMVs run at 93% of the measured 648 GB/s ceiling, and the 7B prefill GEMMs are power-bound
at parity with llama.cpp's MMQ kernels (MEASURED). More kernel tuning won't move single-stream 7B much.
Two things will:

- **fewer full-weight passes per token:** speculation. The draft model is now on main.
- **more useful rows per weight pass:** batching many requests (S2), and later multiple GPUs.

Speed is not the only bar. Nobody runs Axon in production yet, so hardening against
hostile files, bad requests and long-running load comes right after batching (item 2).

The exception is the H100, where prefill is issue-bound at 7.6% of its int8 peak. That gap is real
and has a specific fix ([design/hopper.md](design/hopper.md)).

## Recently landed

- **Draft-model speculative decoding** (`--draft-model`, #27). Lossless, with K = 3 by default.
  Greedy tg128 medians were +37% on Qwen2.5-7B, +44% on Qwen3-8B and +63% on Llama-3.1-8B at K = 3
  (MEASURED in the go/no-go A/B, recorded in `src/draft.rs`). Sampled at temp 0.8 the gain was
  +0.8% on 7B (README).

## Priorities

| # | item | why | status |
|---|---|---|---|
| 1 | **S2: continuous batching in `serve`.** Slot KV cache, per-row positions and sampling, a FIFO scheduler; whole-prompt prefill as its own step. | The largest product gap: ollama and llama-server already serve parallel requests, while `serve` ran one at a time. MEASURED (#38): 7B 4090 C=8 740 tok/s (4.3×), C=16 1017 (5.9×); bella C=16 355 (4.4×). Batched output is identical to solo. Why 4.3× and not the 5.5× estimate: whole-prompt prefill steps stall every row (the "Scale" section, C1). | done |
| 2 | **Real-world hardening.** Fuzz the GGUF reader (truncated files, bad offsets, oversized dims, size overflow) and the HTTP API (malformed JSON, huge bodies, bad UTF-8, invalid params, clients that disconnect mid-stream, many idle connections). Edge prompts: empty, at and over the context limit, odd Unicode. A model-zoo sweep of GGUFs from different converters and quant mixes: each one either matches llama.cpp or is refused with a clear error. Multi-hour soak with many clients, watching RAM/VRAM for leaks. Clean failure on OOM and GPU errors. | Speed is measured; robustness is not. Axon has no users yet and has never served real traffic, while llama.cpp has years of edge cases behind it. Gate: no panic, no out-of-bounds, no hang on any fuzz input; every refusal is a clear error; the server stays up; the soak shows flat memory. Fuzz harness and soak script live in the repo. | in progress (#37: the fuzz-found fixes) |
| 3 | **Startup round 2.** Pinned staging for the weight upload, upload/prefill overlap, faster tokenizer and GGUF parsing. | Time to first token. ESTIMATED 7B warm short-prompt TTFT 624 → ~480 ms, from the measured 25.4 GB/s pinned upload. No change to output bits. | done (#41) |
| 4 | **Multi-GPU, pipeline parallel (stage 1).** | Models bigger than one card, and KV room for serving. Bit-identical to one GPU by construction (MEASURED on 2×4090: decode 1.00×, prefill 1.46–1.55×). Stage 1b (serve pool over the pipeline, ~2× aggregate ESTIMATED) is in the "Scale" section. | stage 1 done ([design](design/multi-gpu.md)) |
| 5 | **Hopper wgmma/TMA prefill.** | H100 prefill trails the 4090 and 5090. ESTIMATED 7B pp2100 12.5k → 22–32k tok/s. Gated on a ~1-hour pod experiment. | design; experiment pending ([design](design/hopper.md)) |
| 6 | **Gemma 3** (1B/4B/12B). | The most-requested missing family. Medium bit-identity risk: new attention variants (head dim 256, sliding window) must keep decode == batched. | 1B/4B done (#35); 27B refused |
| 7 | **Old GPUs (sm_61, Pascal).** | Fixes for Pascal cards, tested on Jaded's GTX 1050. Main has no tested sm_61 path; it refuses a GPU the installed NVRTC can't target. | in flight |
| 8 | **SentencePiece tokenizer** (Mistral 7B v0.3, Llama 2, TinyLlama) on the existing `llama` path. | Cheap coverage, no kernel risk. A port of llama.cpp's SPM tokenizer, checked token for token against llama-server; Mistral's `[INST]` chat format included. | done |
| 9 | **CPU re-board on main, then what it shows.** | The last full CPU scoreboard predates several merged CPU changes. Remaining items (pool spin, P/E-core tail) are ESTIMATED at a few percent. | planned |
| 10 | **GPU decode: long-context attention; locked-clock trial.** | ESTIMATED +3% decode at 4k context. A locked SM clock may recover part of the 9% laptop power-cap cost (MEASURED cost; the fix is untested). The long-context part is Scale item 3. | planned |

Also open:
- **Tool calling: `/v1/responses` (codex) and the ollama-native API.** Chat-completions tool calling
  is on main; these two remain (ESTIMATED 700-1000 and 300-400 lines).
- **Qwen3 0.6B Q8_0 long prefill on sm_89.** The one cell Axon loses on the RTX 4050
  ([benchmarks-gpu.md](benchmarks-gpu.md)).

## Scale: bigger models, longer sessions, multi-user, concurrent requests

Four themes Haseeb asked for by name. The full scope (code references, the KV cost model per model and card,
what llama-server / ollama / vLLM do, gates) is the r10-scope note in the overnight workspace; this is the cut.
The short version: on a 12 GB card, a 7B serves **one** 32k slot today (MEASURED in #42), an agent swarm thrashes the prefix cache, a 14B at 32k becomes a split at 17 tok/s because a slot is a
*reservation*, and nothing survives a restart. The fixes compound, so the order matters more than any one item.

Where Axon can beat llama-server and vLLM on this hardware is exactness: cache bytes are a pure function of the
token prefix, so a slot saved, restored, paged or chunked continues bit for bit. llama.cpp's `--cache-ram`,
`--ctx-shift` and `cache_prompt` all come with a "not reproducible" caveat in its own README.

| # | item | why | status |
|---|---|---|---|
| 1 | **Chunked prefill in `serve`.** A waiting prompt enters ≤ 512 tokens per step, interleaved with the decode rows, instead of a whole-prompt step that stalls everyone. | A 20k agent prompt is a 7 s freeze for every other client today (MEASURED 2682 tok/s at depth). ESTIMATED: steps of ≤ 0.2 s instead, and +10–15 % aggregate at 8 closed-loop clients, since those stalls are most of the gap between the measured 4.3× and the estimated 5.5×. Low bits risk: a chunk is `feed_slot` at a position. Gate: ITL p99 ≤ 2× p50 during a 20k admit; identity with random chunk sizes. | in flight |
| 2 | **Drop the N+1 cache.** A pool of N slots costs N+1 KV caches: the request running alone has its own (`serve.rs` `Engine.cache`). Alias it to slot 0's memory, or retire the solo path once a pool exists. | 7B f16 at 32k on 12 GB: 1 → 2–3 slots; `--kv q8` 3 → 5 (ESTIMATED from the cache sizes; 1 and 3 are MEASURED in #42). No bits change. 8–16 h. MEASURED on bella: f16 1 → 2, q8 4 → 5 (the solo path in a view of a slot; C=1 decode unchanged). | PR open (feat/serve-one-cache) |
| 3 | **Long-context decode attention** (one KV block per warp, a wider combine, then the q8 path). | Decode at 32k is where the lead breaks (MEASURED 7B 72.4 vs llama.cpp 76.7), and attention is paid per row, so it is also the concurrency bottleneck for agents: 8 clients at 32k are ESTIMATED 150 tok/s aggregate today, 216–320 after. q8 tg@32k +13.8 % MEASURED on the branch. | in flight |
| 4 | **KV save/restore.** Slot ↔ pinned host RAM (`--cache-ram`, LRU by prefix) on eviction, `POST /slots/:id?action=save\|restore` to files in llama-server's shape, `chat --session FILE`. | More conversations than slots is the swarm's normal state; eviction costs a 5–12 s re-prefill today and ESTIMATED ~0.1 s after (1.75 GiB at the MEASURED 25.4 GB/s). A restored slot continues bit for bit; llama-server can't promise that. 20–30 h. Gate: restored == uninterrupted, tokens and KV bytes. | planned, after 1 |
| 5 | **Paged KV.** A slot takes 64-position pages as it grows; `--ctx` is a cap, not a reservation. | The widest payoff of the list: a 14B agent on 12 GB stays all-GPU at 10–25k used tokens (ESTIMATED 17 → ~55 tok/s), a 7B card holds ~7 agents at 15k instead of 1–2 slots, and requests that fit no slot stop starving. Bit-identical by construction: paging changes addresses, never order (`docs/design` to come; r4-serve §4.2). The broadest kernel touch since S2 — keep the old entry points as identity-table wrappers. 40–60 h. | planned, after 2 |
| 6 | **Serve ops: `--api-key`, `/metrics`, `/slots`, `--max-prompt-tokens`, idle-slot timeout.** | What exposing `serve` on a LAN needs; parity with llama-server's operational surface. No bits risk. 12–20 h. | planned |
| 7 | **MoE with the experts on the CPU** (Qwen3-30B-A3B first): router, attention and shared weights on the GPU, experts on the CPU for decode, experts streamed through the GPU for prefill (the hybrid's streamer, per expert). | 30B-class quality on 12 GB. ESTIMATED decode 35–55 tok/s (1.0 GB of active experts per token over 75 GB/s RAM) and 3000+ tok/s prefill, where llama.cpp's `--cpu-moe` prefills on the CPU. Medium bits risk: a new architecture and an exact top-k. 80–120 h. Gate: logits within llama.cpp's own CPU/CUDA spread; cache reuse == fresh prefill. gpt-oss-20b (MXFP4, sinks) after. | stages 0–1 in review: qwen3moe loads and runs on the CPU only (exact, at the memory roofline); GPU attention + CPU experts (stage 2) and streamed prefill (stage 3) next ([plan](#mixture-of-experts), [log](log/2026-10-17-moe-stage1.md)) |
| 8 | **YaRN RoPE as an override** (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx N`, llama.cpp's names), plus `chat` forgetting in halves. | No Qwen2.5/Qwen3 GGUF carries scaling keys (checked): 128k is a flag in llama.cpp and a refusal here. A 7B at 128k q8 fits 12 GB (3.7 GiB) and decodes at ESTIMATED 34 tok/s today, 65 after item 3; the 2-minute prefill is why item 4 comes first. New bits only when on. 12–20 h. `chat` today re-prefills ~24k tokens on nearly every exchange once a 32k window is full (`drop_oldest` keeps the minimum): dropping to half is 2–4 h. | planned, after 3 and 4 |
| 9 | **Multi-GPU stage 1b**: the pool over a pipeline, two step batches in flight; and a measured 70B on 2×24 GB. | 32B serve on 2×4090 ≈ 2× one GPU's aggregate (ESTIMATED, [design](design/multi-gpu.md) §3.1); 70B Q4_K_M already runs under PP (ESTIMATED ~21 tok/s on 2×4090, 12k f16 / 24k q8 of KV) but no one has measured it: a $5 pod. 40–60 h + 8. | design done |
| 10 | **Batched speculative decoding.** Draft rows for every running request in one verify step, on while rows × (K+1) ≤ 16 (the weight-read-once regime), off above. | +30–50 % per stream at ≤ 4 clients with a draft model (ESTIMATED from the MEASURED solo gains), nothing lost at 16. Lossless by the verify contract. 30–50 h. | planned, after 1 |

Below the line: shared-prefix pages (copy-on-write over item 5: a swarm's 20k system prompt held once; the swarm's
real memory win), fairness (aging for oversize requests, shortest-first admission), a batched-decode kernel profile
(inside item 1's bench), q4 KV (under "Not planned", below).

Dependencies: #42 → 1 → {4, 10}; #42 → 2 → 5 → shared-prefix pages; 3 → 8; 4 → 8; stage 1 (done) → 9; the
streamer (done) → 7. Item 6 depends on nothing.

## Mixture-of-experts

**Why.** A mixture-of-experts model activates a few experts per token, so most of its weights sit idle on any
given step. Qwen3-30B-A3B runs about 3B parameters per token: 30B-class quality, with a per-token read a 12 GB card
plus system RAM can handle. Axon runs `qwen3moe` on the CPU (stages 0–1, [log](log/2026-10-17-moe-stage1.md));
the GPU paths below are what's left. llama.cpp's `--cpu-moe` has made
"experts on the CPU, attention on the GPU" the usual way to run these models on 8–16 GB cards.

**The plan, in order.**

1. **Qwen3-30B-A3B** (Scale item 7; r10 B2). Router + top-k on the GPU, with an exact select (ties broken by
   index) so decode == prefill. Attention, norms and the shared weights also on the GPU. Experts on the CPU for
   decode: a new CPU expert GEMV over the repacked layout, 8 experts gathered, fixed summation order. Experts
   **streamed** through the GPU for prefill: the existing streamer, per expert matrix.
   - Speed on bella, ESTIMATED: decode 35–55 tok/s (1.0 GB of active experts per token over 75 GB/s RAM),
     prefill 3000+ tok/s on long prompts. llama.cpp's `--cpu-moe` prefills those experts on the CPU.
   - Cost: 80–120 builder-hours. Bits risk: medium. It is a new architecture (`ffn_gate_inp` and 3-D
     `ffn_*_exps` tensors), each layer mixes devices (today's split is a single cut), and the top-k must be exact.
   - Gates: logits vs llama.cpp within llama.cpp's own CPU/CUDA spread on 30B-A3B; `feed_batch == feed` on the MoE
     layers; chat cache reuse == fresh prefill; a bench against `llama-server --cpu-moe` (decode, pp560, pp2100)
     on bella.
2. **gpt-oss-20b** (r10 B6), after item 1 and Gemma 3's sliding window. New pieces: the MXFP4 tensor type,
   attention sinks, and an alternating 128-token sliding window. 60–90 builder-hours, medium bits risk (a new quant
   type, and sinks in every attention kernel); gates as item 1. ESTIMATED ~45 tok/s decode with every expert on the
   CPU ([research-bigmodels](research/research-bigmodels.md) §4.3).
3. **Later, not scheduled:** DeepSeek-class models (MLA attention) and Qwen3-235B-A22B on multi-GPU pods over the
   pipeline parallel from #34. No scope or estimates exist for these yet.

**Ordering.** MoE is rank 7 in r10's merged ranking. It comes after C2 (one cache instead of N+1, rank 2) and
paged KV (rank 5), because that ranking puts what unblocks an agent swarm on a 12 GB card first. In r10's
dependency graph, MoE's own prerequisites are the streamer (on main) and Gemma 3 settling, since both edit
`check_supported`. Nothing in `serve` is a prerequisite, and the serve ops pack (Scale item 6) has no dependencies,
so it can run in parallel.

**Exactness.** Same rule as the CPU/GPU split: exact per device. Decode always runs a given layer's experts on
the same device. Prefill streams them through the GPU, which relaxes bit-identity in the same way the dense split
already documents.

**Not planned for MoE** ([research-bigmodels](research/research-bigmodels.md) §6):
- Lossy shortcuts, which change the model's output: expert deferral, mixed-precision or low-rank stand-ins for
  missed experts, and pre-gated routing (that one also needs fine-tuning).
- Loading experts from disk on demand. Both target models fit in RAM.
- Attention on the CPU. The KV cache belongs on the 648 GB/s side.

Sources: [r10-scope](research/r10-scope.md) §1.3–1.4, §5–6; [research-bigmodels](research/research-bigmodels.md)
§4–6; [design/multi-gpu.md](design/multi-gpu.md).

## Not planned, and why

- **Split-KV or f16-accumulate attention, NCCL-style ring all-reduce, atomics in reductions.** Each
  would break the bit-identity contract (decode == prefill == verify, deterministic output). Multi-GPU
  will use fixed-order reductions instead.
- **More single-stream 7B GEMM tuning on consumer GPUs.** The kernels are at bandwidth or power limits
  (above). An earlier experimental GEMM ("gemm3") was a no-go. Hopper is the exception.
- **AVX-512 and ARM/NEON CPU kernels.** No machine is available to measure them on. A
  correctness-only round under emulation is possible, but it couldn't make speed claims.
- **macOS and Windows builds.** CPU-only ports of dlopen, mmap and thread affinity can't be verified
  here, and Metal would be a new backend.
- **Mixture-of-experts** (Qwen3-30B-A3B, gpt-oss-20b) as a GPU-resident model: they don't fit a 12 GB card.
  With the experts on the CPU they do: see [Mixture-of-experts](#mixture-of-experts), above.
- **Context shift (llama.cpp's K-shift on overflow).** It re-positions cached keys in place, so the cache is no
  longer what a fresh prefill writes: it breaks "cache reuse == fresh prefill" by design. Agent harnesses compact
  their own history on a 400; `chat` truncates. Save/restore (Scale item 4) is the exact alternative.
- **q4 KV cache.** The one 12 GB case it would help (14B at a reserved 32k slot) is solved by paged KV at no
  quality cost, and q8 isn't bandwidth-bound yet (MEASURED: `--kv q8` is −2 % at 32k today). Revisit after the
  long-context attention round, as an experiment with a KL gate.
- **Beating vLLM tensor parallel on NVLink H100/A100 for one stream.** Their all-reduce is ~5 µs there
  and their kernels are mature. Axon's multi-GPU case is PCIe consumer boxes and reproducible output
  ([design/multi-gpu.md](design/multi-gpu.md) §4).
