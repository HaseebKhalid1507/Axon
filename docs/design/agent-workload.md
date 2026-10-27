# Agent workload track: run Synaps on Axon

- Author: Fable. Date: 2026-10-27. Status: **north star: owner, 10-26** (Haseeb set it, and agreed to queue the
  track after the H100 anatomy); **ranking: lead (Jawz), 10-27** (§8, applied in the
  [roadmap](../roadmap.md#next-proposed-order-outside-review-10-19-owner-decides)); the design below is PROPOSED
  (each PR still needs its go). Base read: origin/main `d8a296c` (#152). Copied into the repo from the foreman's
  workspace (`specs/agent-workload.md`) on 2026-10-27; then corrected after review (PR #158, round 1: exactness conditions, capacity labels, kill rules, privacy, the
  llama.cpp quote).
- North star (Haseeb, 10-26): **"Run Synaps on Axon: many concurrent sessions, each spawning lots of subagents,
  with sessions staying warm in cache, then multiple users."** Stage 1: one box, one power user's swarm.
  Stage 2: multi-user, multi-tenant: the datacenter tier the roadmap already has.
- Every number is MEASURED (source named) or ESTIMATED (reasoning given). Nothing here was run on a GPU by me.

## 0. What the workload actually is (MEASURED from Synaps' own logs, structure only)

Source: `~/.synaps-cli/synaps.log.2026-10-2{0..7}` (8 days, 22,795 API requests; per-request
`Token Usage: input | output | cache_read | cache_create` lines, request-start timestamps, keep-warm and subagent
dispatch lines) and the `subagent-history/` file layout (62 parents, 1,347 subagent files). Counts and timestamps
only; no message text was read, copied or left the machine. (Re-count for the repo copy, 2026-10-27 00:10 EDT,
from the `Token Usage` lines only: 22,838 requests, because the 10-27 log was still being written; the prompt
percentiles and the 97.6 % cached share came out the same, and the rest within those extra requests: output p90
1,543, mean 700; new tokens p90 3,933.) Token counts are Claude's tokenizer (Qwen's run
±10–15 %, ESTIMATED; stage A re-tokenizes the synthetic filler, so the replay is in the target model's tokens).

| structure | value | what it says |
|---|---|---|
| prompt tokens per request (cache_read + cache_create + input) | p50 165k, p90 410k, p99 810k | far past any local model's context: the replay must scale or clip (§1.2) |
| of which already cached (cache_read) | **97.6 % of all prompt tokens**; p50 162k | each request is the previous one plus ~1k tokens: the win is *keeping the prefix*, not prefill speed |
| new tokens per request (cache_create) | p50 1,044, p90 3,925, p99 27k | the per-turn delta (tool result + reply); the uncached `input` is p50 2 tokens |
| output tokens | p50 351, p90 1,544, p99 5,554, mean 699; 17.7 % > 1,000 | decode-heavy for an "agent" load; ITL matters |
| gap between request starts | p50 4.5 s, p90 21 s, p99 258 s; 77 % < 10 s, 3.8 % > 60 s, 0.36 % > 10 min | mostly tool-execution gaps; idle slots must stay warm across seconds, sometimes minutes |
| requests in flight | p50 2, p90 4, p99 8, max 10 | today's concurrency is small; the *live-prefix* count is what's large |
| sessions seen per day | 6–36 | each with a 100–340k-token prefix Synaps pays Anthropic to keep warm (710 keep-warm periods, prefix p50 211k) |
| subagent fan-out | per parent p50 2, p90 24, max 363; 144 starts + 177 resumes in 8 days | bursts of siblings sharing a system prompt (p50 1.9k chars) and tool catalog; depth 1 (0 nested dispatches) |
| subagent lifetime | p50 142 messages ≈ 70 requests, p90 ~350 | a subagent is itself a long-lived growing session |
| request duration (Claude) | p50 6.6 s, p90 31 s; stream opens p50 1.8 s | the bar a local engine is measured against |

Three decisions follow. (1) **Intra-session prefix retention dominates** (97.6 %); cross-session sharing (system
prompt + tools) is second and sized in stage A (ESTIMATED 2–20k tokens; not measured because that needs the
tokenized prefixes hashed, which is stage A's job). (2) Warmth, not peak throughput, is the product: at 7B on bella a
25k re-prefill is ≈ 10 s (ESTIMATED from the [#71 log](../log/2026-10-17-paged-kv-s23.md)'s 7 × 15k prompts in 43 s, prefill-bound ≈ 2.4k tok/s batched: 25k / 2.43k = 10.3 s)
against a 67 ms page restore from host RAM (MEASURED, #48, bella, 32k slot; [history](../history.md)). (3) The fan-out burst (24 siblings)
is the one place prefill-once-share-many pays on day one.

## 1. Measure first (stage A: no engine changes)

### 1.1 Trace extraction: `scripts/agent_trace.py` (runs on bella, output is numbers only)

Input: `synaps.log.*` plus `sessions/*.json` and `subagent-history/*/sa_*.json` **read only for structure**: role
sequence, per-block type (text / tool_use / tool_result / thinking), per-block token length (tokenized locally,
then discarded), parent/child links, timestamps. Output: one trace file per day, kept **outside the repo**, one
record per request:

```
{ "t_ms": int,              // arrival, relative to the start of that day's trace
  "session": int, "parent": int|null, "depth": 0|1,   // opaque ids
  "prompt": int,            // total prompt tokens
  "shared": [{"with": "sys"|"tools"|"session", "tokens": int}],  // prefix shared with the system prompt pool / tool catalog / the session's own previous request
  "new": int,               // tokens not in any prior prefix
  "output": int,
  "gap_prev_ms": int,       // since this session's previous request
  "fanout": int }           // siblings dispatched in the same burst
```
**Privacy (the repo is public).** Raw traces never go in the repo: arrival times alone are a record of when the
owner works. What is committed is the two scripts and a small **synthetic** example trace (generated from a seed,
not derived from a real day). Any real trace shared beyond bella has its dates removed (no date in the file name
or contents) and each day shifted to start at `t_ms = 0`. CI checks the committed example for strings: no field
longer than 16 bytes (a length check, not a hash).

Filler: at replay, every token span is synthetic text of the same length (a seeded word list re-tokenized until
the count matches), with the shared spans byte-identical across the requests that share them. No Synaps text
enters the repo; a trace carries no strings but the enum above.

### 1.2 Replay: `scripts/agent_replay.py` (an OpenAI-API client; works against any server)

- Scale knobs: `--scale s` multiplies every span (prompt, new, output) and `--clip ctx` drops the oldest
  "session" span to fit, as Synaps' compaction does. Two canonical cells: **agent-32k** (s = 0.15: prompt p50 ≈ 25k,
  fits a 32k 7B) and **agent-128k** (s = 0.6 under YaRN; H100 / 24 GB only). `--speed k` compresses gaps by k
  (k = 1 is real time; k = 4 the stress cell).
- Open loop by arrival time (a late reply doesn't delay the next arrival within a session beyond the trace's gap),
  so a slow engine shows as queueing, not as a shorter run.
- Determinism: `max_tokens` = the trace's output length, `temperature 0`, fixed seed; the replay records the md5 of
  every reply so an engine change is checked for bits against main on the same trace (the exactness gate for every
  later PR in this track).

### 1.3 Metrics (all reported per cell, 3 rounds, medians, foreign-GPU-load label as `axon-run --timing`)

| metric | definition |
|---|---|
| **TTFT-new** | first byte for a request whose session has no prior request (a new subagent): p50 / p99 |
| **TTFT-resume** | first byte for a request after a session gap > 30 s (the human read, or a long tool): p50 / p99, split by gap bucket (30 s–2 min, 2–10 min, > 10 min) |
| **TTFT-turn** | first byte for the ordinary next turn (gap < 30 s) |
| **goodput** | output tokens/s summed over requests whose TTFT ≤ 2 s and ITL p99 ≤ 100 ms (the SLO; same shape as the H100 board's `slo_ok`) |
| **ITL p99** | inter-token latency over all streamed tokens |
| **warm/GB** | sessions whose next turn gets TTFT-turn ≤ 0.5 s, divided by the KV bytes used (GPU + RAM tiers), at steady state |
| **re-prefill tokens** | prompt tokens the engine computed that an ideal cache would not have (from `/metrics` on Axon, from the client-side timing elsewhere): the direct measure of §2–3 |

### 1.4 Baseline (bella: RTX 5070 Ti Laptop 12 GB, 30 GB RAM; largest sensible model)

Model: **Qwen2.5-7B-Instruct Q4_K_M** (fits all-GPU with 98k f16 / 191k q8 tokens of pages, MEASURED #71), both
`--kv f16` and `--kv q8`. Secondary: Qwen3-8B (144 KiB/token f16: 8 KV heads × 36 layers, ESTIMATED from the
shapes, ~2.6× the 7B's bytes, so ~38k tokens of pages) and Qwen3-30B-A3B with CPU experts (prefill too slow for
a 25k prompt: [roadmap](../roadmap.md#agent-harness-workloads) Agent item 6; include one cell to show it). Engines:

| engine | config | what it has that matters |
|---|---|---|
| Axon main | `serve --parallel 16` default paged pool; `--slot-idle-ms` default | per-slot prefix reuse (longest-prefix slot, `best_slot`), chunked prefill, paged KV; **no** cross-slot sharing, idle prefixes are *dropped* on reclaim (`reclaim`, LRU), nothing survives restart |
| llama.cpp `llama-server` | `-np 4..8 --cache-reuse 256 --slot-save-path`, `cache_prompt: true` | per-slot prompt cache with chunk reuse (KV shift); its README says of `cache_prompt` that "the logits are **not** guaranteed to be bit-for-bit identical for different batch sizes (prompt processing vs. token generation) enabling this option can cause nondeterministic results" (`tools/server/README.md`, llama.cpp 4413a19), slot save/restore to files on request |
| ollama | `OLLAMA_NUM_PARALLEL 4`, `keep_alive` | one KV per parallel slot, prefix reuse per slot; no save |

Report every metric in §1.3 for agent-32k at speed 1 and 4, both KV types. Expected shape (ESTIMATED, to be
replaced): all three engines are fine while live sessions ≤ slots; beyond that Axon re-prefills on every resume,
and llama-server / ollama do too. The baseline's job is to put a number on **re-prefill tokens** and TTFT-resume.

**vLLM / SGLang comparison (ESTIMATED cost):** one RunPod H100 SXM session, Axon + vLLM (APC on) + SGLang
(RadixAttention) on Qwen2.5-7B, agent-32k and agent-128k, prebuilt binaries and a network volume as the showdown
did: ≈ $4–6 (the 10-19 showdown's board cost $3.09 MEASURED for three engines; two cells × two engines more).
Not before stage C exists: on main the result is known in shape (Axon has no cross-request sharing; vLLM/SGLang do).

## 2. Shared-prefix KV pages

### 2.1 Is reuse exact by construction?

Yes, with four conditions, and the code already states the invariant: cache bytes at position t are a function of
the identity and tokens[0..=t] only (`kvsave.rs` header; [docs/determinism.md](../determinism.md): "the order of every float
operation depends on the model's shapes and the token's position, never on the batch"; tests
`gpu_prefill_batchings_bit_identical_*`, `turns_match_fresh_prefill*`, `serve_batched_matches_solo`,
`pool_steps_match_solo_in_any_order`). Chunked prefill is `feed_slot` at a position (engine.rs header), and a
request admitted onto a slot already reuses `held[slot]` tokens and feeds the rest, so "resume from a cached prefix
written by another request" is the same operation as today's "reuse this slot's own prefix", which the contract
and `serve_identity` already cover. What sharing adds is only *who wrote the page*, and the bytes don't depend on
that. The conditions:

1. **Page alignment.** Share whole pages only: 64 positions at a 64-aligned start (PAGE = ATT_BLK; kvstore stores
   whole pages already). Attention folds 64-position blocks ascending and a tile is loaded whole from `tab[b] × 64`
   (r11 §2.4 table), so a page's physical id never enters the arithmetic: MEASURED by `AXON_PAGE_SHUFFLE` identity
   runs (#64/#71: one digest `393d877192f2e4d7` across plain, paged, shuffled, C=1 and C=8).
2. **One identity per pool.** Page sharing (§2.2) needs no identity in the key *today*: one `serve` process runs
   one model with one identity, and there is no `PageMap` on a split at all (`pool_with`: "no pool on a split
   model's GPU half"), nor for a MoE model (serve runs it one request at a time, #127). Identity matters for the
   **RAM and disk tiers** (§3), which outlive a process: see "Identity" there.
3. **No write to a shared page.** A slot writes K/V only at positions ≥ its fed length, and admission always
   re-feeds the prompt's last token (`admit`: `reuse = common.min(p.len() - 1)`), which writes position `len − 1`.
   So **shared pages stop at `min(common, len − 1)`**: only whole pages below it are shared, and the page holding
   `len − 1` is always private or copied, even when the whole prompt is cached (identical siblings, or a resume with
   no new tokens). If the shared prefix ends
   mid-page, the tail page is **copied** (one D2D of 64 rows × layers: 3.5 MiB f16 for 7B, ≈ 10 µs at 400+ GB/s,
   ESTIMATED) before the slot continues: copy-on-write at the divergence point, at page granularity. If the shared
   prefix ends on a page boundary, nothing is copied; the next page is private.
4. **Reads only ever cover filled positions** (attention masks by `np`), so a shared page holding positions the
   reader hasn't reached (a longer sibling's) is never read past the reader's length. Already true for the sink
   page design.

So the exactness gate for stage C is the existing one: the replay md5 of every reply equals main's, with sharing
on, shuffled pages, and a sibling arriving mid-prefill of the prefix.

What §2.2 must also cover (not exactness, but PR C's correctness):
- **The solo path.** A request running alone goes through `Engine::job` → `generate` (graphs, drafting) on a slot
  view, then `adopt`. PR C either shares on that path too or turns sharing off there; the md5 gate covers both.
- **Page accounting.** `admit_check`, `reclaim` and `idle_pages` count a slot's held pages as freeable. With
  refcounts, a page frees only at refs 0, so they must count only pages the slot holds alone.
- **Preemption of the prefilling owner.** If the request writing a prefix is preempted (by recompute) while joiners
  wait for its pages, the joiners must not wait forever: the first joiner takes over and feeds the rest of the prefix
  itself. Liveness only; the bytes are the same whoever writes them.

### 2.2 Design (over `PageMap` + `kvstore`)

- **Key = token-chain hash** (`kvstore::chain`: `key_b = H(key_{b-1}, tokens of block b)`, tokens compared on hit so
  a collision can't alias). "Content-hashed" means hashing the tokens that determine the bytes, not the bytes:
  hashing GPU bytes would cost a readback for nothing.
- **Refcounts in `PageMap`:** `refs[page]`; a slot's list may point at pages other slots also hold; `release`
  decrements and only `retire`s at zero; `check` extends to "sum of refs == occurrences". A GPU-resident radix
  index `key → page` (one `HashMap`, parent links as kvstore's) covers pages held by ≥ 1 slot *or* kept idle
  (refs 0 but indexed: the "warm" set, LRU among unheld leaves, kvstore's rule, so a chain never loses its root).
- **Admit:** compute the prompt's keys; walk the indexed chain, **capped at the whole pages below
  `min(common, len − 1)`** (§2.1 condition 3); `ensure` only the pages past it; the page holding the cap is private
  (copied if it was cached). `best_slot` stops mattering for the prefix (any free slot), which also removes today's "longest
  prefix slot" coupling.
- **In-flight dedupe (prefill once, join many):** a request whose prefix is being prefilled by another request
  *waits for those pages* instead of prefilling them itself: it is admitted as "joined at k" and takes only pages
  **completed in earlier steps**. Chunks are not page-aligned on main (`plan_charged` sizes a chunk from a µs budget,
  and the first chunk starts at the reuse point, any length), so a page still being written is never joined; the
  joiner feeds the partial page itself (bytes are chunk-invariant, so this is exact). Aligning chunks to pages would
  be a scheduler change with its own gate. This is the
  fan-out burst's win and costs no new kernel.
- **Eviction order:** free → retire → unheld warm leaves (LRU) → (stage B) spill to RAM instead of forgetting.
- Kill switch `AXON_SHARE_KV=0` (today's per-slot behaviour, bit for bit, as `AXON_PAGED_KV=0` did).

Memory effect (ESTIMATED, 7B f16 on bella, 1,537 pages): a 10k shared system+tools prefix held once instead of per
session is 157 pages per extra session; at 8 sessions that is 1,100 pages, 71 % of the pool. With 2k shared it is
31 pages per session: small. Stage A decides which world we are in.

## 3. Tiered KV: GPU → pinned RAM → disk, evict to a lower tier, never drop

- **Tier 1 GPU pages** (§2). **Tier 2 pinned host RAM** (`--cache-ram GB`): kvstore's `PageStore` with pinned
  buffers, written D2H when a page is evicted from the GPU *or* write-through as it fills (the s-B plan in
  `kvstore.rs`; write-through costs a 3.5 MiB D2H per 64 tokens per slot, ≈ 0.14 ms at 25 GB/s, ESTIMATED:
  the D2H rate is unmeasured and assumed equal to the 25.4 GB/s pinned **H2D** upload MEASURED on bella
  ([research-kvmem](../research/research-kvmem.md)); negligible against a 60 ms step). **Tier 3 disk** (`--cache-dir`): AXKV page
  files (the #48 format), written when RAM evicts, read on a RAM miss. Restart: the disk index is rebuilt from the
  files' headers; sessions survive a `serve` restart, which nothing does today.
- **Identity (the tiers outlive a process).** `kvstore::ROOT` is a constant today, so pages are not keyed by model
  identity; that is safe only while one process holds one identity in memory. The disk tier (restart survival) and
  FP8 mode (§5) must add it: `ROOT = H(identity)` (fingerprint + `kvsave::NUMERICS` + `cuda::numerics_tag`), or an
  identity check per file as AXKV sessions do. And there is no image to store where `kv_ident` is `None`: every
  CPU/GPU split's GPU half (`first > 0`; on a split that streams prefill the same prefix has two byte-images,
  CPU-decoded vs GPU-prefilled, [determinism.md](../determinism.md)) and deepseek2's MLA cache (84e5307). The tiers
  are off there until those have images.
- Restore on admit: longest chain across tiers; disk → RAM → GPU in page order, pipelined with the prefill of the
  remainder (the first chunk waits for the pages below it only).
- **Restore cost targets** (7B f16, 25k positions = 1.37 GiB): from RAM **≤ 0.1 s** on bella (ESTIMATED from the
  MEASURED 67 ms for a 32k slot page-wise, #48); from disk **≤ 0.6 s** on bella (the 4090 file restore of 29.8k
  positions took 0.41–0.45 s f16 / 0.22–0.23 s q8, MEASURED #48 ≈ 3.7 GB/s; bella's NVMe is unmeasured, so the
  target is 0.6 and stage B measures it). Against a ≈ 10 s re-prefill (ESTIMATED §0) either tier is a ~17–100× TTFT
  win for a resumed session; the stage-A baseline makes it MEASURED.
- **Capacity (bytes per token, from the shapes; pages at 64 tokens; q8 = 0.53× f16 per roadmap).** Every cell is
  **ESTIMATED from the shapes** (bytes per token × the tier's size) except the two cells marked MEASURED (#71's
  bella pool):

| card / tier | 7B f16 (56 KiB/tok) | 7B q8 | Qwen3-8B f16 (144 KiB/tok) | Qwen3-30B-A3B f16 (96 KiB/tok, 48 L × 4 KV) |
|---|---|---|---|---|
| bella GPU 12 GB (after Q4_K_M weights) | **98k tok MEASURED** (#71) ≈ 4 sessions @25k | **191k MEASURED** ≈ 7 | ~38k ESTIMATED ≈ 1 | experts in RAM: ~60k ESTIMATED |
| bella pinned RAM, 16 of 30 GB | 300k ≈ 12 sessions | 565k ≈ 22 | 117k ≈ 4 | 175k ≈ 7 |
| bella disk, 40 GB of the ~49 free | 750k ≈ 30 | 1.4M ≈ 56 | 290k ≈ 11 | 437k ≈ 17 |
| 24 GB card (4090; ~17 GiB of pages ESTIMATED) | 320k ≈ 12 | 600k ≈ 24 | 124k ≈ 5 | ~75k (18.6 GB weights) ≈ 3 |
| 80 GB H100 (72 GiB of pages) | ≈ 1.35M ESTIMATED (72 GiB / 56 KiB) ≈ 54; the 10-19 showdown ran a 1.05M *configured* pool (128 × 8k slots), not the card's capacity | ≈ 2.5M ≈ 101 | 520k ≈ 20 | 630k ≈ 25 |

Sessions at 25k tokens each, no sharing; with a shared 10k prefix each session costs 15k, ×1.67 the counts
(ESTIMATED). The **warm/GB** metric at 7B f16 is 0.73 sessions/GB (q8 1.4; with 10k shared 1.2 / 2.3), which is
what the stage-A table verifies against the ideal.
- RAM tier interactions: pinned RAM is taken from what the CPU-expert MoE path and the mmap'd weights need; the
  planner refuses `--cache-ram` past "available after the model" (the cgroup-aware RAM check from #133 applies).

## 4. Scheduling for agents

Today (engine.rs, #131): FIFO admission (sjf opt-in), 2 prefill lanes, a measured step budget (`--step-ms 60`),
reclaim idle prefixes LRU, preempt-by-recompute youngest-first, oldest never preempted, GROW_TOKENS 1,024 reserved
per admission. Additions, all policy in `engine.rs`, no kernel:

- **Priority classes.** `priority: interactive | background` per request (OpenAI body extension and
  `X-Axon-Priority`; default from a per-API-key setting). Synaps tags its main session interactive and
  subagents background (one line in its backend). Effect: (a) admission order = class, then the #131 key (age, then
  shortest uncached); (b) an interactive request's prefill chunk is budgeted first within `--step-ms`;
  (c) preemption victim = youngest *background* first; (d) the warm set keeps interactive sessions' pages in the
  GPU tier ahead of background ones (eviction LRU within class, background first).
- **Fairness.** Weighted shares by class (default 2:1 of the step's prompt tokens and of decode rows when
  contended), with #131's `--admit-age-ms` aging so background never starves (the stage-A trace has 24-wide bursts;
  the gate below is on their p99).
- **Prefix-aware admission.** (a) In-flight dedupe (§2.2): siblings of a prefix being prefilled are admitted
  as joiners, cost 0 prompt tokens until the prefix exists. (b) Admission cost = *uncached* tokens, as #131's sjf
  already keys on `uncached()`; with sharing that count is after the cross-session chain walk. (c) A burst with a
  shared prefix is admitted together once the prefix is in, up to the pages for their deltas.
- **Idle policy.** `--slot-idle-ms` stops meaning "forget": an idle session's pages go to RAM after it (§3), and
  back on its next turn; the slot (a table row + a logits row) is what frees, not the prefix.

## 5. Stage 2: multi-user, multi-tenant

- **Tenant = API key** (`--api-key` exists, #61; stage 2 makes it a keyring with a tenant id per key).
- **Isolation of cached prefixes.** A prefix-cache hit is observable as TTFT (the prompt-cache side channel:
  cross-user cache hits leak that someone else sent the same prefix). Rule: **a page is shareable only within its
  tenant**: the chain root is `ROOT ⊕ H(tenant_salt)` (a per-tenant 64-bit salt from the keyring), so two tenants'
  identical prompts never meet in the index, in any tier. Operator-declared **public prefixes** (`--public-prefix
  FILE`: a system prompt the operator ships to everyone) are the one cross-tenant pool, salted with the public salt,
  and their hit/miss is not a secret. Optional `--ttft-floor-ms` pads interactive TTFT to a floor for tenants that
  ask for it (cost: latency; off by default).
- **Quotas** per tenant: GPU pages, RAM pages, disk bytes, requests in flight, tokens/min; a tenant over quota
  queues, never evicts another tenant's warm set. `/metrics` and `/slots` gain a tenant label.
- **Auth:** keys in a file with tenant ids and classes; stage 2a keeps bearer keys, 2b adds mTLS or an OIDC
  introspection hook (choose when a named user asks).
- **Composition with the datacenter items.** FP8 (`--numerics fast`) is a different identity: its pages live in a
  separate key space, so a tenant switching modes re-prefills; nothing else changes. TP: pages are per-rank slices
  of the same positions, keyed by the same chain, each rank its own PageMap (the [multi-gpu design](multi-gpu.md)'s per-rank cache).
  The H100 C=128 gap (attention 57 % of the nb 128 step at depth 4,096; MEASURED 10-26, [log](../log/2026-10-26-h100-anatomy-results.md)) is orthogonal: sharing
  removes *prefill* and *memory*, not decode attention; on an H100 it raises the warm-session count (≈ 1.35M tokens
  of pages: with a shared 10k prefix, 25k sessions go from ≈ 54 to ≈ 90, ESTIMATED) and cuts TTFT on the *agent*
  board (F). It does not move the 10-19 showdown board: that board salts every prompt so no request shares a prefix
  ([log](../log/2026-10-19-h100-showdown.md)).

## 6. PR sequence, kill rules, cost

| PR | what | cost (ESTIMATED) | kill / decision rule (pre-registered) |
|---|---|---|---|
| **A** | `scripts/agent_trace.py`, `scripts/agent_replay.py`, traces for 8 days, baseline board on bella (Axon main, llama-server, ollama; 7B f16 + q8; agent-32k at speed 1 and 4). **No engine changes.** | 2 builder-days, $0; the pod comparison deferred to C | A ships regardless (it's the ruler). Re-prefill tokens include those recomputed after a preemption; A states how it counts them (server counters or client-side) before the baseline runs. Decision rule for B/C: if TTFT-resume p50 on main ≤ 1.0 s *and* re-prefill tokens < 10 % of prompt tokens, B/C are deprioritised below item 3; if the measured cross-session shared prefix < 2k tokens, C becomes dedupe-only (in-flight joiners), not a cross-session index |
| **B** | kvstore wired: RAM tier (`--cache-ram`), evict-to-RAM instead of reclaim-drop, restore on admit; `/slots` save/restore; restart survival from the RAM tier's disk spill (stage A's AXKV) | 3–4 builder-days, $0 (bella) | TTFT-resume p50 (gap 2–10 min, 25k) ≤ 0.3 s on bella 7B f16 *and* replay md5 == main *and* aggregate / ITL p99 within 2 % of main at speed 4; else keep the code behind `--cache-ram 0` and record why |
| **C** | shared-prefix pages: refcounted `PageMap`, chain index, COW tail, in-flight dedupe, `AXON_SHARE_KV` | 4–6 builder-days, $0; + $4–6 for the H100 vLLM/SGLang cells | fan-out cell, fixed in A's PR before C starts: 8 siblings arriving within 100 ms, each prompt = the shared prefix (A's measured p50 shared span) + a **private delta of A's measured p50 `new` (1,024 tokens if A doesn't separate it)**, 600-token replies. Gate: siblings 2–8 TTFT p50 ≤ 0.25× main (if the shared span is < 4× the delta, this clause becomes "siblings 2–8 prefill no shared-page tokens", from the re-prefill counter) *and* warm/GB ≥ 1.4× main **on agent-32k at speed 4** *and* md5 == main under `AXON_PAGE_SHUFFLE` *and* C=1 decode within 2 %. Else revert to dedupe-only, which ships only if md5 == main and C=1 decode is within 2 % |
| **D** | priority classes, weighted fairness, prefix-aware admission | 2–3 builder-days, $0 | interactive TTFT p50 under 16 background subagents ≤ 0.5× fifo *and* background goodput ≥ 0.9× fifo *and* background TTFT p99 ≤ 2× fifo; else ship classes as opt-in like sjf |
| **E** | disk tier proper (`--cache-dir`, index rebuild, restore pipelined with prefill) | 2 builder-days | 25k from disk ≤ 0.6 s on bella; else RAM-only stays the product |
| **F** | stage 2: tenants, salts, quotas, keyring; H100 agent board (agent-128k) vs vLLM/SGLang | 5–7 builder-days + ~$10 H100 | Isolation: in a 10k-request adversarial replay (tenant B replays tenant A's prefixes), **0 cross-tenant cache hits** (the hard gate: per-tenant cached-token counters, and a debug assert that a page's salt matches the reader's), and B's TTFT median on A's prefixes within the round-to-round spread of B's TTFT median on fresh prefixes of the same lengths (n = 1,000 each, 3 rounds). Quotas: a breach never evicts another tenant's pages. Overhead: tenancy on vs off, C=1 decode and agent-32k aggregate within 2 %. H100 agent board (agent-128k vs vLLM APC and SGLang RadixAttention): published either way; the README claims the agent tier only if Axon's warm/GB and TTFT-resume p50 are at least as good as the best other's, else it lists them as losses |

Order: A → B → C → D → E → F; B and C are independent in code (B wires kvstore, C changes PageMap) and can run
as two tracks after A. Total stage 1 (A–E): ~13–17 builder-days, < $10. Stage 2 (F): one more week and ~$10.

## 7. Roadmap insertion (applied in [roadmap.md](../roadmap.md))

Paragraph, after "Mission and positioning":

> **North star (owner, 10-26): run Synaps on Axon.** Many concurrent agent sessions, each spawning subagents, with every
> session's prefix staying warm, first for one power user on one box, then for many users. The workload is
> MEASURED from Synaps' own logs (8 days, 22.8k requests): 97.6 % of prompt tokens are a prefix the previous
> request already had, turns arrive 4.5 s apart (p50) with 1k new tokens and 350 output tokens, subagents fan out
> 2 (p50) to 24 (p90) at a time, and sessions idle for minutes. So the product is warmth and exact resume, not
> peak prefill: shared, refcounted prefix pages over #71's pool, a GPU → pinned RAM → disk KV tier that evicts
> instead of dropping, and priority classes in the scheduler. Track spec: [design/agent-workload.md](agent-workload.md).

Rows for the "Agent-harness workloads" table (replacing "later: not scheduled"):

| # | item | why | status |
|---|---|---|---|
| 1 | **Agent trace + replay harness; baseline board** (PR A) | the ruler for the track; no engine change; defines TTFT-new, TTFT-resume, goodput, ITL p99, warm/GB | **scheduled first** |
| 2 | **Shared-prefix pages** (PR C: refcounts, chain index, COW tail, prefill-once-join-many) | fan-out bursts and the swarm's memory; exact by construction (§2.1) | after A |
| 3 | **Tiered KV** (PR B/E: `--cache-ram`, `--cache-dir`, evict to a tier, restore on admit, restart survival) | TTFT-resume ≈ 10 s → ≤ 0.3 s (ESTIMATED from #48's MEASURED 67 ms / 0.45 s restores) | after A, parallel with 2 |
| 4 | **Agent scheduling** (PR D: priority classes, fairness, prefix-aware admission) | the main session stays interactive under a background swarm | after 2 |
| 5 | **Multi-tenant** (PR F: per-tenant salts, quotas, keyring; H100 agent board) | stage 2; composes with FP8 / TP as separate key spaces | after 4 |
| 6 | Fast first turn for MoE models | unchanged (MoE stage 3) | tracked under MoE |

## 8. Where it ranks against "Next" items 1–5

- Item 1 (H100 anatomy + scheduler fixes): (a) measured, (b) rejected, (c)(d) landed; only #139 is open. Done enough.
- Item 2 (Hopper exact GEMM round 2): **shelved 10-26** by its probe rule ([log](../log/2026-10-26-w4a8-probe.md)). Its slot is empty.
- Item 3 (MoE resident): PRs A–C merged; the 30B decode diagnosis is a kernel builder's job and does not contend
  with this track (engine.rs / pages.rs / kvstore.rs vs cuda/moe_res.cu).
- Item 4 (FP8 under `fast`): waits on Haseeb's ADR decision; nothing started.
- Item 5 (adoption floor): done except the tag.

**Ranking (lead, Jawz, 10-27; the owner set the north star 10-26 and agreed to queue it after the H100 anatomy):**
the agent track takes item 2's vacated slot: **A now** (2 days, $0, no engine risk, and it is the
first time the mission's "agent swarm" tier gets a measured number), then **B and C as two tracks** at the
position item 4 holds today; item 4 moves behind them until the ADR is decided, because it is blocked anyway. Item 3
continues in parallel (different files, different builder). The reason this outranks FP8 on the mission's own
terms: the owner's named workload is this one, the local tier is where Axon's exactness is a product advantage no
one else ships (llama.cpp's README says `cache_prompt` logits are "**not** guaranteed to be bit-for-bit identical" and "can cause nondeterministic results"; vLLM/SGLang don't claim bits), and the cost
is a tenth of item 4's. What should *not* move: item 3's kill rules, #139, and the ADR decision itself.
