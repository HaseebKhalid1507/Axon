# Roadmap

Short and opinionated. "On main" means merged at `69adfa2` (2026-10-17). Everything else is planned or in flight.
Estimates come from the design notes and are labeled.

**Mission: make LLM inference cheap and fast, and run the best open models.** Cheap means wasting no
hardware, from a laptop to a datacenter: the most tokens per GPU, per watt and per dollar. Fast means as close
to the hardware limit as the math allows, for one stream and for thousands. The best models means the frontier
open models, not only the ones that fit a laptop. Every item below serves one of those three.

## Mission and positioning

Haseeb's words, in order: **"make inference cheap and fast"**, **"run the best models"**, and **"run at datacenters
without wasting resources: squeeze every drop."**

Two tiers serve those, under two numerics modes ([design/determinism-modes.md](design/determinism-modes.md), proposed):

| tier | hardware | what "cheap and fast" means there | numerics | who it competes with |
|---|---|---|---|---|
| **Local and agent** | one consumer GPU, a laptop, a CPU box; one person or one agent swarm | the fastest single stream, lossless speculation, an exact cache that survives restarts, replies that don't change under load | `exact` (default) | llama.cpp, ollama. Axon's edge: measured single-stream speed on NVIDIA ([benchmarks-gpu.md](benchmarks-gpu.md)) and a contract nobody else ships by default ([determinism.md](determinism.md)) |
| **Datacenter** | H100/B200 nodes, FP8 weights, tensor parallel, C ≥ 32 | tokens per dollar at goodput; TTFT and ITL p99 under load; frontier MoE models resident | `fast` (default under `--datacenter`), with `exact` available for eval and replay | vLLM, SGLang, TensorRT-LLM. Axon's edge is not yet measured there: the one concurrency cell (4090, int4) trails vLLM 13–24 % at C 8–32 ([log](log/2026-10-18-c-curve.md)), and no H100 board at C ≥ 8 exists. The first datacenter board is what makes this row honest |

"Run the best models" is one list for both tiers (Qwen3-30B-A3B, gpt-oss, DeepSeek V3 / Kimi K2), but the first tier
runs them split across CPU and GPU and the second runs them resident, at FP8 once that path exists. Both tiers share
one engine, one tokenizer and one sampler; they differ in the kernel set the numerics flag selects and in what the gate
measures: bits in `exact`, tolerance plus a board against vLLM/SGLang in `fast`. The bit-identity contract is the
local tier's product and every tier's regression net; it is a cost only where FP8 and tensor parallel begin, and the
mode is how it stops being a global constraint there without being dropped where it is free (ESTIMATED ≤ 2 % on
today's GPU kernels; the ADR has the numbers).

**What Axon refuses to build:** lossy approximations sold as the model (expert deferral, pre-gated routing, key
skipping) under either mode; platforms it can't measure on (Metal, ROCm, NEON) until a machine exists; a third
numerics tier. **What it will not claim:** "squeezes every drop" until a datacenter board is measured on the hardware
the mission names.

## The thinking behind the order

Single-stream decode on a 7B/8B model is near the hardware limit. On Parabellum (RTX 5070 Ti Laptop),
7B decode GEMVs run at 93% of the measured 648 GB/s ceiling, and the 7B prefill GEMMs are power-bound
at parity with llama.cpp's MMQ kernels (MEASURED). More kernel tuning won't move single-stream 7B much.
Two things will:

- **fewer full-weight passes per token:** speculation. The draft model is now on main.
- **more useful rows per weight pass:** batching many requests (S2), and later multiple GPUs.

Speed is not the only bar. Nobody runs Axon in production yet, so hardening against
hostile files, bad requests and long-running load comes right after batching (item 2).

The exception is the H100, where prefill was issue-bound at 7.6% of its int8 peak. A first wgmma round is on main
(#33: +12 % 7B prefill on an H100 PCIe, MEASURED); the plain products are still issue-bound ([design/hopper.md](design/hopper.md)).

## Long-term goals

These are not priorities, and nothing here jumps the queue below. They are where the mission points (the best
models, cheaply and fast), so choices made along the way shouldn't close them off.

- **Run Kimi K2** (1T parameters, 32B active per token, the DeepSeek V3 architecture). It is the largest open
  model and the test of whether Axon scales past a 12 GB card. Every piece it needs is useful on its own, and
  each is listed under [Frontier models](#frontier-models-100b-to-1t):
  - experts on the GPU (MoE stages 2–3, already planned);
  - MLA attention and DeepSeek routing (the `deepseek2` architecture), tried first on a small DeepSeek-architecture
    model that fits Parabellum. **Stage 0 (CPU-only exact reference) is in** (2026-10-18, on Moonlight-16B-A3B,
    K2's routing and tokenizer at 1/60 of its size; [log](log/2026-10-18-deepseek2-s0.md)); **stage 1 (GPU MLA
    attention, router and shared experts, routed experts on the CPU) is in** ([log](log/2026-10-18-deepseek2-s1.md));
  - Kimi's tokenizer (`kimi-k2`): **in**, token for token with Hugging Face's on a varied corpus;
  - 2–3-bit weight types. At Q4_K_M the model is ≈ 600 GB; at 2–3 bits it is ≈ 250–380 GB. Q2_K and Q3_K (and
    IQ4_NL, their files' fallback type) run on the CPU and CUDA (feat/q2k-q3k, [log](log/2026-10-17-q2k-q3k.md));
  - an expert cache across GPU, RAM and disk, and multi-GPU (pipeline parallel is on main, #34);
  - the hardware to test it at full size: a multi-GPU or large-RAM pod, a few dollars to tens of dollars an hour.

  YaRN, which K2 uses for its long context, is already on main (#49).

## Recently landed

Newest first, as of `69adfa2`; the numbers are the PRs' own MEASURED headlines.

- **Paged KV, stages 0–1** (#64, #68): every cache kernel through a page table; same bits, same speed.
- **One cache instead of N+1 in `serve`** (#60): bella 7B at 32k serves 2 slots instead of 1 (q8: 5 instead of 4).
- **Serve ops** (#61, #62): `--api-key`, `GET /metrics`, `GET /slots`, `--max-prompt-tokens`, `--slot-idle-ms`.
- **MoE stages 0–1** (#63): `qwen3moe` loads and runs on the CPU, exact.
- **Chunked prefill in `serve`** (#45, #55, #57): other streams' ITL p99 during a 24k admit 2392 → 70 ms (7B, 4090).
- **Draft model next to a CPU/GPU split** (#52): 32B on bella, greedy 2–3.5× faster, sampled 1.7–3.3×.
- **Long-prompt prefill attention** (#56): 7B pp16k +5.5 %, pp32k +3.4 % on bella, bit-identical.
- **Long-context decode attention** (#50): 7B tg@32k +3.3 % f16, +17.6 % `--kv q8` (bella).
- **Identity tuple and `system_fingerprint`** (#51; [determinism.md](determinism.md)).
- **YaRN RoPE override** (#49): Qwen 32k → 128k, opt-in ([long-context.md](long-context.md), [log](log/2026-10-16-yarn.md)).
- **KV save/restore, stage A** (#48, #54): `chat --session`; a 29.8k-position session restores in 0.41–0.45 s vs 3.16 s.
- **Gemma 3 1B/4B** (#35), CPU and CUDA.

## Priorities

| # | item | why | status |
|---|---|---|---|
| 1 | **S2: continuous batching in `serve`.** Slot KV cache, per-row positions and sampling, a FIFO scheduler; whole-prompt prefill as its own step. | The largest product gap: ollama and llama-server already serve parallel requests, while `serve` ran one at a time. MEASURED (#38): 7B 4090 C=8 740 tok/s (4.3×), C=16 1017 (5.9×); bella C=16 355 (4.4×). Batched output is identical to solo. Why 4.3× and not the 5.5× estimate: whole-prompt prefill steps stall every row (the "Scale" section, C1). | done (#38) |
| 2 | **Real-world hardening.** Fuzz the GGUF reader (truncated files, bad offsets, oversized dims, size overflow) and the HTTP API (malformed JSON, huge bodies, bad UTF-8, invalid params, clients that disconnect mid-stream, many idle connections). Edge prompts: empty, at and over the context limit, odd Unicode. A model-zoo sweep of GGUFs from different converters and quant mixes: each one either matches llama.cpp or is refused with a clear error. Multi-hour soak with many clients, watching RAM/VRAM for leaks. Clean failure on OOM and GPU errors. | Speed is measured; robustness is not. Axon has no users yet and has never served real traffic, while llama.cpp has years of edge cases behind it. Gate: no panic, no out-of-bounds, no hang on any fuzz input; every refusal is a clear error; the server stays up; the soak shows flat memory. Fuzz harness and soak script live in the repo. | fuzzers and the fuzz-found fixes done (#37: `scripts/fuzz_gguf.py`, `scripts/fuzz_serve.py`, 9 findings fixed). Soak + model-zoo sweep: PR #83 (`scripts/soak.py`, `scripts/zoo.py`, [hardening.md](hardening.md)). Soaks: 1.5B 3.2 h ×2 on sm_89 and 7B 1.1 h on sm_120 pass; CPU 0.5B 4 h on bella passes and 2 h on jade fails the RSS slope on a single 8 MiB step. 0 failing requests in about 223k. Zoo: 18 files, 8 match llama.cpp, 1 within llama.cpp's CPU/CUDA spread, 9 clean refusals (unclear "??" refusals fixed). Left: OOM / GPU-error injection, multi-day runs |
| 3 | **Startup round 2.** Pinned staging for the weight upload, upload/prefill overlap, faster tokenizer and GGUF parsing. | Time to first token. ESTIMATED 7B warm short-prompt TTFT 624 → ~480 ms, from the measured 25.4 GB/s pinned upload. No change to output bits. | done (#41) |
| 4 | **Multi-GPU, pipeline parallel (stage 1).** | Models bigger than one card, and KV room for serving. Bit-identical to one GPU by construction (MEASURED on 2×4090: decode 1.00×, prefill 1.46–1.55×). Stage 1b (serve pool over the pipeline, ~2× aggregate ESTIMATED) is in the "Scale" section. | stage 1 done (#34) ([design](design/multi-gpu.md)) |
| 5 | **Hopper wgmma/TMA prefill.** | H100 prefill trails the 4090 and 5090. ESTIMATED 7B pp2100 12.5k → 22–32k tok/s. Gated on a ~1-hour pod experiment. | first round done (#33): fused gate+up on wgmma, 7B prefill 9263 → 10376 tok/s (+12.0 %) on an H100 PCIe (MEASURED, `AXON_GEMMH=1`). The plain products reach 84 TMAC/s (issue-bound), so the 22–32k estimate is not reached; the default routing itself was not run on an H100 ([design](design/hopper.md)) |
| 6 | **Gemma 3** (1B/4B/12B). | The most-requested missing family. Medium bit-identity risk: new attention variants (head dim 256, sliding window) must keep decode == batched. | 1B/4B done (#35); 27B refused |
| 7 | **Old GPUs (sm_61, Pascal).** | Fixes for Pascal cards, tested on Jaded's GTX 1050. Main has no tested sm_61 path; it refuses a GPU the installed NVRTC can't target. | done (#39): with NVRTC 13 and 12.9 installed, jade's GTX 1050 (sm_61) now falls back to 12.9 and generates (main refused); 25/25 CUDA tests pass there (MEASURED) |
| 8 | **SentencePiece tokenizer** (Mistral 7B v0.3, Llama 2, TinyLlama) on the existing `llama` path. | Cheap coverage, no kernel risk. A port of llama.cpp's SPM tokenizer, checked token for token against llama-server; Mistral's `[INST]` chat format included. | done (#32) |
| 9 | **CPU re-board on main, then what it shows.** | The last full CPU scoreboard predates several merged CPU changes. Remaining items (pool spin, P/E-core tail) are ESTIMATED at a few percent. | planned |
| 10 | **GPU decode: long-context attention; locked-clock trial.** | ESTIMATED +3% decode at 4k context. A locked SM clock may recover part of the 9% laptop power-cap cost (MEASURED cost; the fix is untested). The long-context part is Scale item 3. | long-context attention done (#50, Scale item 3); locked-clock trial planned |

Also open:
- ~~**Tool calling: `/v1/responses` (codex) and the ollama-native API.**~~ Done
  (docs/serve-apis.md, log 2026-10-17-serve-responses-ollama). Left: `format` / structured
  output (needs grammar-constrained sampling), embeddings.
- **Q2_K and Q3_K.** CPU and CUDA kernels for Q2_K, Q3_K and IQ4_NL (the fallback type in the same files);
  the 2–3-bit frontier models need them ([Long-term goals](#long-term-goals)). **In review** (feat/q2k-q3k,
  [log](log/2026-10-17-q2k-q3k.md)); follow-ups: a repacked (r8) CPU layout (CPU prefill on 7B Q2_K /
  Q3_K_M is 41% / 22% behind llama.cpp at pp2100, MEASURED) and gemvm small-batch kernels.
- **MoE stage 2** (attention and router on the GPU, experts on the CPU). **In flight** (PR #67); its status
  lives in Scale item 7 and [Mixture-of-experts](#mixture-of-experts).
- **Native safetensors, then FP8 / NVFP4 weights** (the FP8 plan, `specs/fp8-safetensors.md`).
  Stage A, the loader (Hugging Face checkpoint directories of qwen2 / qwen3 / llama, BF16 run as the
  GGUF F16, CPU only), is **in review** ([log](log/2026-10-19-safetensors-loader.md)). Next: B, FP8
  per-channel dense on the GPU (sm_89 / sm_120); C, FP8 128×128 block scales (DeepSeek-style, MoE
  experts from per-expert tensors); D, NVFP4.
- **Qwen3 0.6B Q8_0 long prefill on sm_89.** The one cell Axon loses on the RTX 4050
  ([benchmarks-gpu.md](benchmarks-gpu.md)).
- **Gemma 3 27B (head_dim 128) on CUDA.** Needs the head-128 windowed attention kernels and the
  1/sqrt(dim/heads) query scale; refused at load today (CPU and GPU).

## Scale: bigger models, longer sessions, multi-user, concurrent requests

Four themes Haseeb asked for by name. The full scope (code references, the KV cost model per model and card,
what llama-server / ollama / vLLM do, gates) is the r10-scope note in the overnight workspace; this is the cut.
The short version, as first written: on a 12 GB card, a 7B served **one** 32k slot (MEASURED in #42; two since #60), an agent swarm thrashes the prefix cache, a 14B at 32k becomes a split at 17 tok/s because a slot is a
*reservation*, and nothing survived a restart (`chat --session` does now, #48; `serve` doesn't yet). The fixes compound, so the order matters more than any one item.

Where Axon can beat llama-server and vLLM on this hardware is exactness: cache bytes are a pure function of the
token prefix, so a slot saved, restored, paged or chunked continues bit for bit. llama.cpp's `--cache-ram`,
`--ctx-shift` and `cache_prompt` all come with a "not reproducible" caveat in its own README.

| # | item | why | status |
|---|---|---|---|
| 1 | **Chunked prefill in `serve`.** A waiting prompt enters ≤ 512 tokens per step, interleaved with the decode rows, instead of a whole-prompt step that stalls everyone. | A 20k agent prompt is a 7 s freeze for every other client today (MEASURED 2682 tok/s at depth). ESTIMATED: steps of ≤ 0.2 s instead, and +10–15 % aggregate at 8 closed-loop clients, since those stalls are most of the gap between the measured 4.3× and the estimated 5.5×. Low bits risk: a chunk is `feed_slot` at a position. Gate: ITL p99 ≤ 2× p50 during a 20k admit; identity with random chunk sizes. | done (#45, #55, #57): during a 24k admit, the other streams' ITL p99 went 2392 → 70 ms (7B, RTX 4090, MEASURED in #45; p50 59 ms) and a short prompt arriving meanwhile gets its first token in 111 ms instead of 2.1 s; the long prompt's own TTFT is +48 %. #55/#57: per-prompt depth budgets and idle fairness |
| 2 | **Drop the N+1 cache.** A pool of N slots costs N+1 KV caches: the request running alone has its own (`serve.rs` `Engine.cache`). Alias it to slot 0's memory, or retire the solo path once a pool exists. | 7B f16 at 32k on 12 GB: 1 → 2–3 slots; `--kv q8` 4 → 5 (ESTIMATED from the cache sizes; 1 and 4 are MEASURED in #42). No bits change. 8–16 h. MEASURED on bella: f16 1 → 2, q8 4 → 5 (the solo path in a view of a slot; C=1 decode unchanged). | done (#60): bella 7B at 32k, f16 1 → 2 slots, `--kv q8` 4 → 5; C=1 decode 101.4 → 102.8 tok/s, within noise (MEASURED) ([log](log/2026-10-17-serve-one-cache.md)) |
| 3 | **Long-context decode attention** (one KV block per warp, a wider combine, then the q8 path). | Decode at 32k is where the lead breaks (MEASURED 7B 72.4 vs llama.cpp 76.7), and attention is paid per row, so it is also the concurrency bottleneck for agents: 8 clients at 32k are ESTIMATED 150 tok/s aggregate today, 216–320 after. q8 tg@32k +13.8 % MEASURED on the branch. | done (#50): bella 7B tg@32k f16 +3.3 % (the +12 % gate not met: attention is near the DRAM ceiling), `--kv q8` +17.6 %, 0.5B f16 +10 % (MEASURED). Next (planned): fold the combine, ESTIMATED ≈ +4 % |
| 4 | **KV save/restore.** Slot ↔ pinned host RAM (`--cache-ram`, LRU by prefix) on eviction, `POST /slots/:id?action=save\|restore` to files in llama-server's shape, `chat --session FILE`. | More conversations than slots is the swarm's normal state; eviction costs a 5–12 s re-prefill today and ESTIMATED ~0.1 s after (1.75 GiB at the MEASURED 25.4 GB/s). A restored slot continues bit for bit; llama-server can't promise that. 20–30 h. Gate: restored == uninterrupted, tokens and KV bytes. | stage A done (#48, #54): AXKV session files and `chat --session`, exact; a 29.8k-position 7B f16 session restores in 0.41–0.45 s vs a 3.16 s re-prefill (RTX 4090, MEASURED). Stage B (`serve`: `--cache-ram`, `/slots` save/restore) planned |
| 5 | **Paged KV.** A slot takes 64-position pages as it grows; `--ctx` is a cap, not a reservation. | The widest payoff of the list: a 14B agent on 12 GB stays all-GPU at 10–25k used tokens (ESTIMATED 17 → ~55 tok/s), a 7B card holds ~7 agents at 15k instead of 1–2 slots, and requests that fit no slot stop starving. Bit-identical by construction: paging changes addresses, never order (`docs/design` to come; r4-serve §4.2). The broadest kernel touch since S2 — keep the old entry points as identity-table wrappers. 40–60 h. | stages 0–1 done (#64, #68): every cache kernel reads through a page table (null = identity); `gen` md5 identical, C=1 speed within noise on bella (tg128 0.997×, 30k-prefix decode 1.000×, MEASURED) ([log](log/2026-10-17-paged-kv-s01.md)). Stages 2–3 (the pool allocates pages; serve plans, admits, reclaims and preempts by pages) in review (#71, [log](log/2026-10-17-paged-kv-s23.md)); stage 4 (preempt-by-store, shared prefix pages) after item 4 |
| 6 | **Serve ops: `--api-key`, `/metrics`, `/slots`, `--max-prompt-tokens`, idle-slot timeout.** | What exposing `serve` on a LAN needs; parity with llama-server's operational surface. No bits risk. 12–20 h. | planned |
| 7 | **MoE with the experts on the CPU** (Qwen3-30B-A3B first): router, attention and shared weights on the GPU, experts on the CPU for decode, experts streamed through the GPU for prefill (the hybrid's streamer, per expert). | 30B-class quality on 12 GB. ESTIMATED decode 35–55 tok/s (1.0 GB of active experts per token over 75 GB/s RAM) and 3000+ tok/s prefill, where llama.cpp's `--cpu-moe` prefills on the CPU. Medium bits risk: a new architecture and an exact top-k. 80–120 h. Gate: logits within llama.cpp's own CPU/CUDA spread; cache reuse == fresh prefill. gpt-oss-20b (MXFP4, sinks) after. | stages 0–1 on main (CPU only, exact); stage 2 in review: GPU attention + router, CPU experts (exact, graded vs the CPU; 30B decode ESTIMATED 46–51 tok/s at 8k from the Pruned-2B's per-layer timings); streamed prefill (stage 3) next ([plan](#mixture-of-experts), [log 1](log/2026-10-17-moe-stage1.md), [log 2](log/2026-10-17-moe-stage2.md)) |
| 8 | **YaRN RoPE as an override** (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx N`, llama.cpp's names), plus `chat` forgetting in halves. | No Qwen2.5/Qwen3 GGUF carries scaling keys (checked): 128k is a flag in llama.cpp and a refusal here. A 7B at 128k q8 fits 12 GB (3.7 GiB) and decodes at ESTIMATED 34 tok/s today, 65 after item 3; the 2-minute prefill is why item 4 comes first. New bits only when on. 12–20 h. `chat` today re-prefills ~24k tokens on nearly every exchange once a 32k window is full (`drop_oldest` keeps the minimum): dropping to half is 2–4 h. | planned, after 3 and 4 |
| 9 | **Multi-GPU stage 1b**: the pool over a pipeline, two step batches in flight; and a measured 70B on 2×24 GB. | 32B serve on 2×4090 ≈ 2× one GPU's aggregate (ESTIMATED, [design](design/multi-gpu.md) §3.1); 70B Q4_K_M already runs under PP (ESTIMATED ~21 tok/s on 2×4090, 12k f16 / 24k q8 of KV) but no one has measured it: a $5 pod. 40–60 h + 8. | design done |
| 10 | **Batched speculative decoding.** Draft rows for every running request in one verify step, on while rows × (K+1) ≤ 16 (the weight-read-once regime), off above. | +30–50 % per stream at ≤ 4 clients with a draft model (ESTIMATED from the MEASURED solo gains), nothing lost at 16. Lossless by the verify contract. 30–50 h. | planned (item 1 is done) |

Below the line: shared-prefix pages (copy-on-write over item 5: a swarm's 20k system prompt held once; the swarm's
real memory win; [Agent-harness workloads](#agent-harness-workloads-later-not-scheduled) item 2), fairness (aging for oversize requests, shortest-first admission), a batched-decode kernel profile
(inside item 1's bench), q4 KV (under "Not planned", below).

Also done, not yet in the table: **item 6, serve ops** (#61, #62: `--api-key`, `/metrics`, `/slots`,
`--max-prompt-tokens`, `--slot-idle-ms`; [log](log/2026-10-17-serve-ops.md),
[follow-up](log/2026-10-17-serve-ops-followup.md)), and **item 8's YaRN override** (#49: a pass-key needle 18/18 up to 128k
with it, 9/18 without; +7.7–8.0 % perplexity at 2k–8k on 7B, so opt-in; [log](log/2026-10-16-yarn.md)). Item 8's other half,
`chat` forgetting in halves, is in review (`chat --forget-to F`, default 0.5: once the window is full, the oldest
exchanges go until the rest fills half of it, so the next turns append to the cache instead of each re-prefilling
it; cache == fresh prefill of what's kept; [log](log/2026-10-18-chat-halves.md)). Both status
cells sit next to item 7's, which PR #67 edits, so they are updated here until #67 merges.

Dependencies: 1 (done) → {4 stage B, 10}; 2 (done) → 5 (stages 0–1 done) → shared-prefix pages; 3 (done) and 4
→ the rest of 8; stage 1 (done) → 9; the streamer (done) → 7.

## Agent-harness workloads (later: not scheduled)

On the roadmap, not prioritised: nothing here jumps the queue above. Agent harnesses (Synaps, and others such as
opencode, pi and codex) talk to `serve` through its OpenAI- or ollama-compatible API, and their load doesn't look
like a chat benchmark. **Why:** several concurrent agent sessions on one consumer GPU, each with a ~15–20k-token
system prompt plus tool definitions that the sessions share, long multi-turn histories, tool-call turns, and long
idle gaps while the human reads. Everything below is ESTIMATED or unmeasured; no number in this section is MEASURED
on this workload.

What works today: OpenAI `tool_calls` (streamed), per-slot prefix reuse, continuous batching (#38), a long prompt's
admission no longer stalls the other streams (Scale item 1), and paged KV (#71, once merged).

| # | item | why | status |
|---|---|---|---|
| 1 | **Agent-workload benchmark.** N concurrent sessions, a shared ~20k system prompt, tool-call turns, idle gaps. Report TTFT, ITL p50/p99 and aggregate tok/s on 7B, Qwen3-8B, gpt-oss-20b and Qwen3-30B-A3B on bella (12 GB). | Nothing on this workload is measured yet, so this comes first: it sets the baseline the other items are judged against. | later: not scheduled |
| 2 | **Shared prefix pages across sessions.** A radix/prefix cache over #71's paged pool (like vLLM's automatic prefix caching or SGLang's RadixAttention): the common system prompt is prefilled once and stored once, with copy-on-write at the point where a session diverges. | The biggest expected win for this workload (ESTIMATED): every session after the first skips a ~20k prefill and the pages it would have held. This is the "shared-prefix pages" entry below the line in [Scale](#scale-bigger-models-longer-sessions-multi-user-concurrent-requests). Depends on #71. | later: not scheduled |
| 3 | **Idle-session KV offload to host RAM.** | This is Scale item 4 (KV save/restore, stage B: `--cache-ram`), not a separate project. The agent motivation: a session idles for minutes while the human reads, and its slot is better spent on a session that is generating. | later: not scheduled (tracked as Scale item 4) |
| 4 | **Fast first turn for MoE models.** Stream the experts through the GPU for prefill. | Today's prefill with the experts on the CPU is ~265 tok/s on gpt-oss-20b (MEASURED, bella, job 1018-092300-3a16, pp486), so a 20k agent prompt is about a minute to first token (ESTIMATED from that rate). The plan is item 1 of the design note `specs/gptoss-prefill.md`, the same work as MoE stage 3 ([Mixture-of-experts](#mixture-of-experts)). | later: not scheduled (tracked under MoE stage 3) |

Batched speculative decoding for the decode side is #90 (Scale item 10); it isn't repeated here.

**Capacity on bella at ~25k context per session** (ESTIMATED: arithmetic from KV bytes per token, paged and
`--kv q8`, against the VRAM left after the weights; not measured): a 14B ≈ 1 session, Qwen3-8B ≈ 3, Qwen2.5-7B ≈ 8,
an MoE with the experts in RAM ≈ 6, with the slow first turn of item 4. These counts store each session's copy of
the system prompt separately; item 2 is what changes them.

## Mixture-of-experts

**Why.** A mixture-of-experts model activates a few experts per token, so most of its weights sit idle on any
given step. Qwen3-30B-A3B runs about 3B parameters per token: 30B-class quality, with a per-token read a 12 GB card
plus system RAM can handle. Axon runs `qwen3moe` on the CPU (stages 0–1, [log](log/2026-10-17-moe-stage1.md))
and with attention + routers on the GPU and the experts on the CPU (stage 2, [log](log/2026-10-17-moe-stage2.md));
streaming the experts through the GPU for prefill (stage 3) is what's left of item 1. llama.cpp's `--cpu-moe` has made
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
   **Stage 0 done** (CPU only, exact: MXFP4, sinks, the window, YaRN, o200k, harmony;
   [log](log/2026-10-18-gpt-oss-s0.md)). **Stage 1 in review**: #67's split (attention, sinks, window, router and
   head on the GPU, experts on the CPU; [log](log/2026-10-18-gpt-oss-s1.md)): bella tg128 46.6 tok/s, 1.99x
   llama.cpp `-ot exps=CPU`, but prefill 0.27x (its experts run on the CPU; llama.cpp ships big-batch expert GEMMs
   to the GPU). Next: an MXFP4 GPU kernel, bit-exact with the CPU's, for streamed-expert prefill and the whole model
   on 24 GB cards.
3. **Later:** the 100B–1T models (Qwen3-235B, Qwen3-Coder-480B, DeepSeek V3, Kimi K2). See
   [Frontier models](#frontier-models-100b-to-1t), below.

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
- Loading experts from disk on demand. Both target models fit in RAM; for models that don't, see
  [Frontier models](#frontier-models-100b-to-1t) item 4.
- Attention on the CPU. The KV cache belongs on the 648 GB/s side.

Sources: [r10-scope](research/r10-scope.md) §1.3–1.4, §5–6; [research-bigmodels](research/research-bigmodels.md)
§4–6; [design/multi-gpu.md](design/multi-gpu.md).

## Frontier models (100B to 1T)

**Why it's here.** People judge an engine's top end by the largest open models. None of them runs on Parabellum
today. Two of them need new architecture work, and none fits the machine (30 GB RAM, 12 GB VRAM, ~49 GB of free
disk). This section is the plan for when building it makes sense: what's missing, in what order, and what should
trigger each step. Nothing here is scheduled.

**The models.** Sizes are ≈ 4.8 bits per weight × parameters, the Q4_K_M average (calculated; none downloaded).
"Read per token" is the weight bytes one decode step reads at that size.

| model | kind | Q4_K_M | read per token | in Axon today | missing |
|---|---|---|---|---|---|
| Qwen3-235B-A22B | MoE | ≈ 141 GB | ≈ 13 GB | `qwen3moe`, CPU only (#63) | experts on the GPU (MoE stages 2–3); a box with ≥ 160 GB RAM |
| Qwen3-Coder-480B-A35B | MoE | ≈ 288 GB | ≈ 21 GB | `qwen3moe`, CPU only | as above, ≥ 320 GB RAM |
| Llama 3.1 405B | dense | ≈ 243 GB | ≈ 243 GB | `llama`; pipeline parallel (#34) should run it on a 4×80 GB pod (untested) | tensor parallel, to make it fast |
| DeepSeek V3 / R1 (671B, 37B active) | MoE + MLA | ≈ 403 GB | ≈ 22 GB | refused | MLA attention, DeepSeek routing, ≥ 450 GB RAM or a GPU node |
| Kimi K2 (1T, 32B active) | DeepSeek V3 architecture | ≈ 600 GB | ≈ 19 GB | refused | same as DeepSeek, plus Kimi's tokenizer and 2–3-bit types; ≥ 650 GB at Q4_K_M ([long-term goal](#long-term-goals)) |
| DeepSeek V4.1 Flash (1017) | MoE + new attention, vision | 444.7 GB (MEASURED: published GGUF shards) | ≈ 9 GB | refused (`deepseek41`) | a new architecture ([below](#deepseek-v41-flash)), experts on the GPU, a ≥ 500 GB node |

**What's missing, in build order.**

1. **Experts on the GPU.** MoE stages 2–3, already planned ([Mixture-of-experts](#mixture-of-experts), above).
   Three of the five rows depend on it, and so do both DeepSeek-architecture rows once item 2 exists.
   *Trigger:* already next in the MoE plan.
2. **MLA attention and DeepSeek routing** (the `deepseek2` GGUF architecture, which covers V2, V3 and Kimi K2).
   MLA caches one compressed latent per token per layer (576 values at DeepSeek V3's shapes) instead of full K and
   V. Decode can fold the up-projections into the query ("absorbed" form), which reads far less KV per step.
   The routing adds shared experts, sigmoid scores with a per-expert bias, and a grouped top-k (no published GGUF
   carries the group count, and llama.cpp doesn't apply it: refused until one does).
   - **Cleanups owed:** byte-level BPE vocabularies don't split USER_DEFINED tokens (`</think>`, harmony
     markers) out of plain text as llama.cpp and HF do. (Done in #104: `model::moe::Moe`'s router rule is one
     `enum Router { Gate(Gate), Oai(Box<Oai>) }`, the GPU routers one kernel, `moe_router`, and `MoeKind`.)
   - **Status (2026-10-18): stage 0 done** ([log](log/2026-10-18-deepseek2-s0.md), design note
     `specs/r12-deepseek2.md`): the absorbed form on the CPU for decode, prefill and verify alike (bit for bit),
     a K-only 576-wide cache, DeepSeek V3 YaRN's attention scale, the `kimi-k2` and `deepseek-v3` tokenizers, the
     Kimi and DeepSeek V3 chat formats. **Stage 1 done** ([log](log/2026-10-18-deepseek2-s1.md)): `--device cuda`
     with MLA attention (own kernel module, latent cache, attention in 128-row splits), router, shared experts
     and the dense lead on the GPU, the routed experts on the CPU (the #67 split), one code path for decode and
     batched passes. Next: `--kv q8` for the latent cache, tensor-core attention for long prefills, then the
     generic MoE stages (experts streamed through the GPU, experts resident in VRAM).
   - **Exactness:** the absorbed and expanded forms round differently, so decode, prefill and verify must pick one
     order and keep it, the same decode == prefill rule as every other kernel.
   - **Dev model that fits Parabellum:** Moonlight-16B-A3B-Instruct (Q4_K_M 10.54 GB): the current absorbed GGUF
     layout, Kimi K2's routing and tokenizer. (Every published DeepSeek-V2-Lite GGUF is the legacy non-MLA layout,
     which Axon refuses with a re-convert message.) DeepSeek V3 and Kimi K2 then run on a pod.
   - **Gates:** logits vs llama.cpp within its own CPU/CUDA spread on V2-Lite; decode == prefill == verify; tok/s vs
     llama.cpp on the same box.
   - **Cost:** no estimate yet. Scope it with a design note first, as r11 did for MoE.
   - *Trigger:* MoE stage 3 is done, and someone asks for DeepSeek-class models (or they become the default for
     people running models locally).
3. **Tensor parallel over PCIe, with fixed-order reductions.** Today's multi-GPU is pipeline parallel: the GPUs
   take turns, so one stream decodes at one GPU's bandwidth over the whole model. Splitting every layer across the
   GPUs lets them all read at once. The all-reduce has to sum in a fixed order (no ring, no atomics) to keep the
   exactness contract; that costs speed against NCCL, and it is why "beat vLLM on NVLink" stays under
   [Not planned](#not-planned-and-why).
   - ESTIMATED for 405B on 4×H100: pipeline ≤ 14 tok/s (243 GB ÷ 3.35 TB/s), tensor parallel up to ~4× that before
     the all-reduce cost.
   - *Trigger:* multi-GPU stage 1b (Scale item 9) is done, and a measured 70B or 405B pipeline run shows decode is
     the gap.
4. **Experts from host RAM or disk, cached on the GPU.** For MoE models bigger than the box: hot experts stay on
   the GPU, the rest in pinned RAM, the cold tail on NVMe. Exact by construction, because caching changes where an
   expert is read from, never which experts run. Prior work: MoE-Infinity, MoE-CORE, MawForge
   ([research-bigmodels](research/research-bigmodels.md) §2.2).
   - ESTIMATED on Parabellum from NVMe alone (~5 GB/s): Qwen3-235B ≈ 0.4 tok/s. That's not useful, so this is a
     pod and workstation feature, not a laptop one.
   - *Trigger:* after item 1, when a target model is bigger than the machine's RAM.
5. **Hardware and disk.** Parabellum can't hold any of these files. Full-size gates need pods with large disks; a
   network volume means each file downloads once.
   - *Trigger:* whenever a step above needs a full-size gate.

**DeepSeek V4.1 Flash.** <a id="deepseek-v41-flash"></a>Added 2026-10-17 at Haseeb's request
([HF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)). A target, not a plan: we deal with it
when MoE on the GPU and the DeepSeek V3 work above are done.

- **What it is** (MEASURED from the HF `config.json` and a published GGUF's header): 40 layers, width 5120,
  384 routed experts + 1 shared, 6 used per token, expert FFN 2304, 1M context (YaRN ×16). The attention is
  not V3's MLA: 64 query heads over 1 KV head of 512 dims (64 roped), a query LoRA (1280), a grouped output
  LoRA (8 groups, rank 1024), a 128-token sliding window, per-layer KV compression ratios, and a sparse-attention
  indexer (32 heads × 128, top-512). Also new: 4-way hyper-connections normalised by 20 Sinkhorn iterations,
  "engram" n-gram hash embeddings at layers 1 and 14, `sqrtsoftplus` gating, a SwiGLU clamp, 3 draft layers
  (MTP/DSpark) and a vision tower. Native weights are FP8 (32×32 blocks, ue8m0 scales) with FP4 experts:
  510 GB of safetensors. GGUFs exist with `general.architecture = deepseek41`: Q4_K_M 444.7 GB, Q2_K 264.5 GB.
- **Reference:** our llama.cpp checkout (2026-10-12) has `deepseek4` but not `deepseek41`. Find the
  implementation that produced those GGUFs before designing anything; it is the oracle.
- **What Axon needs:** the `deepseek41` architecture (attention, indexer, compression, hyper-connections, engram,
  routing), MoE experts on the GPU (stages 2–3), multi-GPU for a 445 GB file, and FP8/FP4 types only if we want
  the native weights rather than GGUF. Text first; vision later. The draft layers could feed `--draft`.
- **Exactness:** the indexer's top-k must break ties by position and pick the same keys in decode, prefill and
  verify; the Sinkhorn loop runs a fixed number of iterations in a fixed order; compressed and uncompressed layers
  each keep one fold order. Same decode == prefill rule as everything else.
- **Hardware:** Parabellum can't download it (≈ 35 GB free disk). Q4_K_M needs 8×80 GB (640 GB) or 4×H200 (564 GB,
  tight once KV is added) with pipeline parallel, or a ≥ 512 GB RAM server with the experts on the CPU, plus a
  ≥ 500 GB network volume so the file downloads once.
- **Cost of one full-size gate:** ESTIMATED from our account's prices (H100 $2.89/h): 8×H100 ≈ $23/h, so a
  2-hour session (download + gates + bench) ≈ $50. That is above the current $38 guard cap; raise it for that run.
- **Read per token:** ESTIMATED ≈ 15B active weights (≈ 10B expert, ≈ 5B attention) × 4.8 bits ≈ 9 GB.
- **Dev model:** no small `deepseek41` sibling is known yet; check at the time. Otherwise prune to 2–3 layers and
  a few experts on a big-disk pod, as the Pruned-2B did for MoE, and gate kernels against the reference there.
- *Trigger:* MoE stage 3 and the DeepSeek V3 item (2) are done.

**Why this order.** Item 1 is already planned and unlocks most of the table. Item 2 unlocks the two biggest
models. Item 3 matters for dense giants. Item 4 only matters past RAM.

**Decode ceilings, batch 1** (ESTIMATED: bandwidth ÷ bytes read per token; real numbers land below these):

| model | where | ceiling |
|---|---|---|
| Qwen3-235B-A22B | Parabellum, weights from NVMe (~5 GB/s) | ≈ 0.4 tok/s |
| Llama 3.1 405B | Parabellum, weights from NVMe | ≈ 0.02 tok/s (one token per ~50 s) |
| Qwen3-235B-A22B | server, experts on CPU, assumed ~300 GB/s RAM | ≈ 20 tok/s |
| DeepSeek V3 | server, experts on CPU, assumed ~300 GB/s RAM | ≈ 14 tok/s |
| Llama 3.1 405B | 4×H100, pipeline parallel (today's code) | ≤ 14 tok/s |
| Llama 3.1 405B | 4×H100, tensor parallel (item 3) | up to ~50 tok/s before the all-reduce cost |
| DeepSeek V4.1 Flash | 8×H100, pipeline parallel, Q4_K_M | ≤ ~370 tok/s (≈ 9 GB ÷ 3.35 TB/s) |
| DeepSeek V4.1 Flash | server, experts on CPU (≈ 6 GB/token), assumed ~300 GB/s RAM | ≈ 50 tok/s |

## Not planned, and why

- **Split-KV or f16-accumulate attention, NCCL-style ring all-reduce, atomics in reductions: not in `exact`
  mode.** Each would break the bit-identity contract (decode == prefill == verify, deterministic output). Multi-GPU
  in exact mode uses fixed-order reductions. Where they are allowed, and what they are measured to buy (little, on
  today's kernels), is the subject of [design/determinism-modes.md](design/determinism-modes.md).
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
  ([design/multi-gpu.md](design/multi-gpu.md) §4). Exact tensor parallel with fixed-order reductions is planned
  ([Frontier models](#frontier-models-100b-to-1t) item 3); it trades some speed for that contract.
