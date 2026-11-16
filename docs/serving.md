# Serving: `axon serve`

Everything about `axon serve`: what it is, its flags, its operations (auth, metrics, slots, the KV RAM tier and
shared prefix pages), tool calling, and the Responses and ollama APIs with the clients run against them. Every
route and the fields each takes: [serve-apis.md](serve-apis.md). The rest of the command line:
[usage.md](usage.md).

## What it is

`axon serve` is an OpenAI-compatible HTTP server (below) with tool calling. On one GPU it
batches the decode steps of concurrent requests (`--parallel N`) and feeds new prompts in chunks
between them, over a paged KV cache (64-position pages shared by every slot: a slot holds only the
positions it has; no extra cache for a request running alone); elsewhere it
runs one request at a time. Besides chat completions it speaks OpenAI's Responses API (what codex
uses) and ollama's native API ([docs/serve-apis.md](serve-apis.md)). Operations: `--api-key`,
`GET /metrics` (Prometheus), `GET /slots`, `--max-prompt-tokens`, `--slot-idle-ms`.
`--cache-ram BYTES|auto|off` (opt-in, default off; `auto` is the smaller of 4 GiB and a quarter of the
RAM available after the model, off under a binding cgroup limit) keeps conversations' KV pages in pinned host RAM
and copies them back on the next turn, bit for bit. Copies of pages the GPU pool still holds are
written only into free room and are the first evicted, so the tier's useful capacity is about its
size, not its size minus the pool's. An evicted 25k-token session resumes in 0.141 s against a
2.713 s re-prefill on an RTX 4090, and in 0.36–0.51 s against 8.6–8.8 s on bella (two TAINTED reads)
([design](design/kv-ram-tier-exclusive.md) §6, §14). It stays opt-in until #233 (the first request waits
3.2–4.0 s for the pinning) and #234 (the write-behind costs 2.3 % tok/s at C=16 on a 4090) are fixed.
Requests share the KV pages of a common prompt prefix, same bits (on by default; `AXON_SHARE_KV=0` turns it
off; off by default under `--numerics fast`'s FP8 kernels, where `AXON_SHARE_KV=1` turns it on; #207, #216;
[below](#operations-auth-metrics-slots)).

## Starting a server, and its flags

```bash
axon serve models/qwen2.5-7b-instruct-q4_k_m.gguf --device cuda --port 8080
curl -N localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"stream":true}'
```

An OpenAI-compatible HTTP/1.1 server (any OpenAI client works with
`base_url="http://127.0.0.1:8080/v1"`): `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/completions` (raw text, no template, as `gen`), `POST /v1/responses` and the
ollama API (`/api/chat`, `/api/generate`, ...; below), `GET /health`, `GET /metrics`,
`GET /slots` (below); streamed
(server-sent events, `data: [DONE]` at the end; `stream_options.include_usage` adds a
usage chunk) or whole. It takes the device, `--ctx`, `--kv`, `--gpu-layers` and `--draft`
flags of `chat`, plus `--host` (default 127.0.0.1), `--port` (default 8080), `--alias NAME`
(another model name to answer to), `--max-tokens N` (a cap on any request's generation;
default: up to the context), `--cors [ORIGIN]` (answer CORS preflights and allow ORIGIN,
default `*`, to call the server from web pages; off by default, so an arbitrary page can't
use your local model), `--max-conns N` / `--max-queue N` (defaults 256 / 128),
`--parallel N` (requests decoded at once on the GPU, default 16 with paged KV, 8 without; 1: one at a time;
an explicit N that doesn't fit in GPU memory at start is refused, the automatic sizing falls back and says so),
`--prefill-chunk N` (at least N prompt tokens a batched step feeds while others decode;
default 256; `--prefill-chunk 0`: whole prompts), `--step-ms MS` (a step's target time
while requests decode, default 80: the prompt tokens it feeds beside their decode step are
what fits at the cost the server measures as steps run, per pass, per prompt token and per
decode row count, and a step feeds at most the prompts that fit; 0: exactly `--prefill-chunk`
tokens; `--prefill-ms` is its old name, still accepted), `--step-tokens N`
(pin a step's prompt tokens; this also lifts the bound on prompts per step),
`--admit-order sjf|fifo` (`sjf`, default: shortest prompt first, several per step; `fifo`,
`AXON_ADMIT=fifo`: first come first served, one per step). With `--step-ms 80` this default
passed a pre-registered rule on an H100 SXM, mix C=128 against fifo at 60 ms: aggregate
+7.6 %, goodput +60.5 %, ITL p99 99.9 ms, C=64 aggregate +4.6 %, identical outputs
([#165](https://github.com/HaseebKhalid1507/Axon/issues/165)). `--admit-age-ms MS` (sjf: a
request waiting this long starts before shorter ones, default 3000; 0 removes the starvation
bound, a long prompt can then wait as long as shorter ones keep arriving, and `serve` says so),
`--prefill-lanes N` (prompts prefilling at once: with sjf, prompts longer than a step's chunk,
default parallel/16 within 2..8, and short ones never wait for a lane; with fifo, all of them,
default 2). The scheduling of before: `--admit-order fifo --step-ms 60`, and for the fixed
budget of before #131 (`--step-ms` less the last decode step) add `AXON_STEP_COST=legacy`.
`--no-constrain` (tool calls not held to the tools offered and to JSON:
[Tool calling](#tool-calling)), `--quiet`
(no per-request line on stderr). `GET /health` is 503 until the model is loaded.

- Context (`--ctx N`, per request): by default the model's trained context (32768 for
  Qwen2.5, 40960 for Qwen3, 131072 for Llama 3.1), so agent harnesses' 10k-40k-token prompts
  work without flags. With the whole model on one GPU (placed automatically, or every layer
  by `--gpu-layers N`), if the weights leave too little GPU memory for it (the KV cache of
  `--kv`'s type), the longest that fits (a multiple of 256; placed automatically, at least
  4096, else the model is split). Over several GPUs (`AXON_GPUS`), the longest the pipeline
  fits, at least 4096, else 8192. On the CPU, the longest whose KV cache fits in free RAM (a
  multiple of 1024, at least 8192). A CPU/GPU split (automatic or `--gpu-layers`) keeps 8192,
  as before: a long cache there takes GPU memory from the layers. A shorter context than
  trained is said once on stderr, with why and the flag to change it. An explicit `--ctx N`
  is used as given (up to the trained length). A `--draft-model` is sized for the main
  model's context (at most 32768 by default; a split's 8192) and drafts up to it.
- Many requests at once (`--device cuda`, the whole model on the GPU): up to `--parallel`
  requests decode together, one token each per step over the model's weights (continuous
  batching: a request joins as soon as a slot is free, and leaves when done). While others
  decode, a new prompt is prefilled in chunks, up to `--step-ms` / `--prefill-chunk` worth
  per step in between their steps (shortest remaining prompt first; one passed over 4
  steps in a row goes first), so a long prompt doesn't stop the others' streams while it
  prefills (on a 4090, 7B, a 24k-token prompt: their worst token gap ~70 ms instead of
  2.4 s); with nothing decoding, prompts are fed one after another, shortest remaining
  first, each to its end, in pieces of 250-500 ms at full speed (250 ms worth rounded up
  to whole prefill passes; 4096 tokens until the server has measured a chunk's cost); a
  prompt that 4 shorter, younger ones have finished ahead of goes first. A long prompt sent
  to an idle server alone takes the single-stream path below and is fed whole: a request
  arriving during that prefill waits for it. The KV cache is paged: a pool of 64-position
  pages shared by every slot, a slot holding pages only for the positions it has, so the
  context is a slot's cap, not memory reserved for it (`AXON_PAGED_KV=0`: the old pool of
  slots reserved at the full context each). The pool is the GPU memory left beside the
  weights and scratch (at most a page per position of every slot), up to 16 requests at once
  by default (`--parallel N`: N); the context is the model's (or `--ctx N`), at most what the
  pages hold. Placed automatically, the whole model stays on the GPU if it fits beside an
  8192-token cache, whatever `--ctx` says (the pool takes what is left). A request starts
  when a slot is free and the pages for its prompt are (with a page per running request to
  grow into); else it waits in the queue. Idle slots keep their prefixes (a later request
  sharing one reuses its rows) until their pages are needed, least recently used first.
  When decode runs out of pages, the youngest prompt still being prefilled gives its pages
  back, then the youngest running request is preempted (fed again later, the same output);
  the oldest never is. The banner says the pool's shape (`KV: P pages x 64 = T tokens over
  up to S slots (cap C)`), `GET /slots` each slot's `pages_used` and the pool's
  `pages_free` / `pages_total`. Measured on a 12 GB laptop GPU at the default 32768: Qwen2.5-7B
  Q4_K_M gets 1537 pages (98k tokens: 6 agents at 15k each; `--kv q8`: 2983 pages, 12), where
  the reserved pool held 3 slots; Qwen2.5-14B Q4_K_M with `--kv q8 --ctx 32768` stays on the
  GPU with ~15k tokens of pages instead of splitting. The request running alone runs in a slot's pages (its own
  scratch beside it, no cache of its own), picked as a batched request's is: the slot
  holding the longest prefix of its prompt if that prefix is at least half the slot's
  tokens or more than half the prompt's, else the one idle the longest (a shared <bos> or
  system prompt doesn't evict a conversation). With `AXON_PAGED_KV=0`, without `--ctx` or
  `--parallel` the context comes first: as many slots (2 to 8) as fit with the whole
  context; if not even 2 do, 2 or more with the longest context they fit with, down to
  32768 (or the whole context, if shorter); below that, one request at a time with the
  whole context. `--parallel N` alone: N slots of the longest context that fits them (down
  to 1024 positions). `--ctx N`: never shortened; the slots that fit with it (up to
  `--parallel`), else one at a time. The server says on stderr what it chose and why; the
  context it serves is a slot's (`/v1/models` `meta.n_ctx`; longer prompts get a 400
  `context_length_exceeded`). The Q6_K prefill copy (int8 Q6_K matrices for prefill's tensor
  cores, same results; 1.02 GiB on 7B Q4_K_M) is made after the caches and slots, only from
  the memory they leave (`AXON_Q6K8=0` never makes it). `AXON_SERVE_TRACE=FILE` (off by
  default; no output changes) appends a JSON line per batched step to FILE: its prompt feeds
  (passes, tokens, ms), its decode pass (rows, mean depth, ms) and its wall time.
  A request running alone takes the single-stream path (CUDA graphs, `--draft`), and
  moves into the batch when a second one arrives (its rows are already in its slot: nothing
  is copied). Every request's output is exactly what it
  gets alone, whatever else runs with it (greedy, and sampled with a `seed`): the batched
  kernels are bit-identical per row to the one-sequence ones. On the CPU, with
  `--gpu-layers` (a CPU/GPU split), with a multi-GPU pipeline (several `AXON_GPUS`) and with
  a mixture-of-experts model the server runs one request at a time; an explicit
  `--parallel N` (N > 1) there is an error (for a MoE model, before its experts are placed).
- Requests run each exactly as `gen` / `chat` would run it (greedy `/v1/completions` output
  is byte-identical to `axon gen`'s). The KV cache carries over between requests: a prompt
  that starts with tokens a cache (or a slot) already holds (a chat follow-up resending the
  conversation, a shared system prompt) only runs the model over the rest
  (`usage.prompt_tokens_details.cached_tokens`).
- Chat prompts are framed as `axon chat` frames them: as the GGUF's own chat template does,
  token for token (ChatML, Llama 3's headers with BOS, Gemma, DeepSeek, harmony, Kimi, Mistral;
  the template's default system message when none is given, e.g. Qwen2.5's or Qwen2's; earlier
  replies of a thinking model without their reasoning, as Qwen3's and R1's templates drop it;
  see [usage.md](usage.md#chat)), and end on the same tokens (Llama 3: `<|eot_id|>` too).
  Tools need a template that renders them (Qwen2.5, Qwen3, Llama 3.1/3.2, gpt-oss): a request
  with tools, or a tool turn, for a model whose template has none (Qwen2, Qwen1.5, Llama 3.0) is
  a 400, as ollama refuses it.
  `/v1/completions` tokenizes and stops as `gen` does (BOS for Llama 3). Qwen3 thinks
  unless `chat_template_kwargs: {"enable_thinking": false}` (or `--no-think` as the server
  default); the reasoning comes as `reasoning_content`, apart from `content`
  (`--think-inline` leaves the `<think>` block in `content`).
- Sampling: `temperature` (0 = greedy), `top_p`, `seed`, and the common extensions `top_k`,
  `min_p`, `repetition_penalty` (or `repeat_penalty`) and `repeat_last_n`; `max_tokens` (or
  `max_completion_tokens`), `stop` (up to 16 strings, not included in the output). What a
  request leaves out comes from the server's sampling flags, else `chat`'s defaults
  (temperature 0.8, top-k 40, top-p 0.9) but with no repeat penalty (1.0, as OpenAI and
  llama-server: `chat`'s 1.1 would quietly penalize the punctuation and indentation of code
  and JSON; `--repeat-penalty` or a request's `repetition_penalty` turns it on). For Qwen3's
  thinking mode Qwen recommends temperature 0.6, top_p 0.95, top_k 20 (greedy can loop on
  long reasoning).
  `presence_penalty` / `frequency_penalty` other than 0, `n` > 1, `logprobs`, legacy
  `functions`, `logit_bias` and JSON `response_format` are refused with a 400 rather than
  ignored.
- Tool calling: see below.
- `finish_reason` is `stop` (end token or stop string) or `length`; `usage` counts tokens;
  a `timings` object gives prompt and generation speed (llama-server's field names).
- A client that disconnects stops its generation within one token.
- Limits: 16 KiB of request head, 1 MiB of body (`Content-Length` or chunked), 256
  connections, 128 queued requests (more: 503); keep-alive and pipelining; idle connections
  close after 30 s. No TLS or HTTP/2: put a reverse proxy in front to expose it beyond
  localhost.
- If the engine fails on a request (a CUDA error is sticky: every later request would fail
  too), the server logs the error and exits with code 70, for a supervisor to restart it.

## Operations: auth, metrics, slots

- `--api-key KEY` (or `AXON_API_KEY`, which keeps the key out of `ps`): every route but
  `GET /health` needs `Authorization: Bearer KEY` (compared in constant time); without it a
  401 with an OpenAI-shaped error (`code: "invalid_api_key"`) and `WWW-Authenticate: Bearer`.
  CORS preflights (`OPTIONS`) stay open: browsers send them without credentials, and they
  answer with headers only. A request with a body but without the key gets the 401 right
  after its headers, whatever the route (`/health` and `OPTIONS` are open only without a
  body), before a 413 for a body past 1 MiB and without `100 Continue`; its body is never
  buffered or parsed, and the connection closes (after dropping what the client already
  sent, at most 2 s / 4 MiB, so the 401 isn't lost to a reset; an upload still going after
  that can be reset before the client reads the 401). `--api-key` wins over `AXON_API_KEY` (the variable isn't
  read then); the variable is trimmed (a secret file's trailing newline), and set but blank
  or not UTF-8 is an error. stderr says `api key: on (from AXON_API_KEY|--api-key)`, never
  the key. Still plain HTTP: put TLS in front of it beyond a trusted LAN.
- `GET /metrics`: Prometheus text format. `axon_requests_total{route,status}`,
  `axon_generated_tokens_total`, `axon_prompt_tokens_total`,
  `axon_prompt_cached_tokens_total` (prefix reuse), `axon_requests_in_flight`,
  `axon_queue_depth` (in flight, not in a slot), `axon_slots_busy` / `axon_slots_free`,
  `axon_slot_idle_evictions_total`, `axon_prompt_refused_total`,
  `axon_tool_calls_dropped_total` (tool calls begun but returned as no call: a `<tool_call>` block or
  Llama JSON reply that isn't a JSON object with a string name, as content, or a gpt-oss call whose arguments
  never opened as an object, as reasoning), `axon_tool_calls_invalid_total` (calls returned whose `arguments`
  aren't a JSON object: a streamed call cut off or gone bad, #260, or a non-object value); constrained
  tool calls' (below): `axon_constraint_tokens_total` (tokens picked inside a held call),
  `axon_constraint_fallback_total` (held positions decided on the whole logits row, the device's candidates
  holding too few allowed tokens), `axon_constraint_trigger_rejected_total` (a `<tool_call>` written out in a
  token that goes on with text no call starts with: left free), `axon_constraint_stalls_total` (held calls
  closed the shortest way after they stalled: their arguments cut, each also a stderr line),
  `axon_constraint_unheld_total` (tools requests left unconstrained: the grammar couldn't hold their tools'
  names or the call their prompt opened; each reason also a stderr line, once),
  `axon_constraint_mask_cache_clears_total` and the gauge `axon_constraint_mask_cache_peak_entries` (how full
  the grammars' token-mask caches get: 16 MiB each, started over when full) and
  `axon_constraint_mask_seconds_total`; and the histograms
  `axon_ttft_seconds` (from the request's body read, so tokenizing and queueing count, to
  its first text) and `axon_itl_seconds` (between a request's
  text events, about one token each). The scheduler's (#131): `axon_waiting_requests`,
  `axon_prefill_lanes` / `axon_prefill_lanes_busy`, `axon_step_prompt_tokens`, the histograms `axon_step_seconds`
  and `axon_step_seconds_rows16` (steps that fed prompt beside running rows), and the step cost model's
  `axon_step_cost_*`. The KV RAM tier's and shared pages' are below.
  A request is counted in `axon_requests_total` before its first byte is sent, and out of
  `axon_requests_in_flight` and its slot before its last byte: a client that has the whole
  response, then reads /metrics, never sees its own request still running.
- `GET /slots`: JSON, `{"ctx", "slots": [{"id", "busy", "held_tokens", "request_id",
  "age_ms"}], "queue_depth", "waiting": [{"id", "prompt_tokens", "age_ms"}], "graphs_evicted_oom",
  "steps_eager_oom"}`. `held_tokens`: the prefix whose KV rows the slot holds (what
  a later prompt can reuse); `request_id` is the `#n` of the stderr log. A request running
  alone is in the slot it runs in (on the CPU, a split or a pipeline, the one cache is slot 0).
  `waiting`: the requests the engine holds for a slot (with a pool), not the ones running; empty on an idle
  server (#268: a request that had run alone stayed listed until the next one came).
  With the paged pool, each slot adds `pages_used`, and the object `pages_free`, `pages_total`, `page_tokens`,
  `preempted` and `prefills_unfed`. The two OOM counters: CUDA graphs given up, and steps run kernel by kernel, for
  lack of VRAM (never fatal, the same bits).
- `--max-prompt-tokens N`: a longer prompt is refused before prefill with a 400
  (`code: "context_length_exceeded"`, as a prompt past the context). It counts the whole
  rendered prompt: the chat template's tokens and the tool schemas too; exactly N is allowed.
- `--slot-idle-ms MS` (at least 1000: delayed ACKs and retransmits pause a live client's
  acknowledgements for hundreds of ms): a stream whose client stops reading is cancelled (as
  a disconnect: its slot freed for the next request). A client that reads slowly isn't
  affected. A stopped one can't be told from a slow one until its receive buffer is full, so
  the clock starts only then: on Linux (x86, arm, riscv64, s390x, loongarch64), once the
  unacknowledged bytes stop moving for MS ms. With a default ~128 KiB receive window that is
  roughly 500+ tokens of SSE after the client stopped (MEASURED, bella CPU, 0.5B,
  `--slot-idle-ms 1000`, a client with default buffers: 792 tokens and 6.3 s from its last
  read to the slot freed, and in another run the autotuned buffers absorbed a whole
  1900-token reply; with a 4 KiB `SO_RCVBUF`, 158-202 tokens and 1.7 s). A reply shorter
  than that just finishes into the buffers. Elsewhere only a write blocked for MS ms (at most the
  60 s write timeout) is detected, after the server's send buffer fills too, and stderr says
  so at startup.
- **Shared prefix KV pages** (the GPU's paged pool; on by default, `AXON_SHARE_KV=0` turns them off and
  `AXON_SHARE_KV=1` turns them on; unset or empty is the default; any other value, `false` or `off` say, refuses
  to start, naming the valid ones, #267). The start line says on or off and what decided it: the default, the
  variable, or fast kernels' default (below); it prints under `--quiet` too when the variable is set to `0` or `1`, or fast kernels
  turned sharing off. A request whose
  prompt starts with whole pages (64 positions) another request already computed, the same
  tokens after the same prefix, maps those pages instead of computing them again, and prefills
  only the rest: subagents that share a system prompt and tool catalog hold one copy of it. Same
  bits: a page is reused only under the same tokens and the same parent page, so its rows are the
  ones the request would compute (exact kernels: under `--numerics fast` on an FP8 checkpoint the
  rows change with how a prompt is chunked, so sharing is off there by default; `AXON_SHARE_KV=1` turns it on,
  within fast mode's tolerance, as a cached prefix is; [determinism](determinism.md)).
  **What a client can learn:** `usage.prompt_tokens_details.cached_tokens` counts the pages a
  request mapped, so it now includes pages another client computed. A request learns how many
  64-token blocks at the start of its prompt someone else sent recently, as with any shared
  prompt cache (before, only an idle slot's last conversation could tell it that). On a server
  shared by several users who shouldn't learn about each other's prompts (one `--api-key` for
  all of them, say), set `AXON_SHARE_KV=0`. Pages enter the index as they fill; a request writes only pages
  it alone holds (a prompt that continues mid-way into a shared page is fed again from that page's
  start). When a slot gives pages up, the indexed ones stay on the GPU (warm) for a later prompt,
  until room is needed: then the least recently used warm leaf goes first (with `--cache-ram`, to
  the RAM tier), before an idle slot's pages. `GET /slots` adds `pages_warm` and `pages_shared`;
  `/metrics` `axon_kv_pages_warm`, `axon_kv_pages_shared`, `axon_prompt_shared_tokens_total`
  (admissions' prompt tokens mapped from the index), `axon_refeed_shared_tokens_total` (tokens a preempted
  or unfed request maps from the index again when it is fed again: its prompt and what it generated so far;
  apart since #219, so preemption doesn't inflate the prompt count) and `axon_pages_evicted_total` (warm
  pages given up for room). `AXON_KV_PAGES=N` sizes the pool to N pages (at most what fits), for testing a server
  under memory pressure. If the server exits with an engine failure while sharing is on, the line names
  `AXON_SHARE_KV=0`: restart with it to rule sharing out. The win is subagent fan-out: eight subagents on a 6,006-token prefix reach their first
  token in 0.591× the time (#211); the agent board's Synaps-shaped trace shares only 4–7 % of its prompt tokens,
  so it gains little there.
- `--cache-ram BYTES|auto|off` (default 0: off): a KV RAM tier. A conversation's KV pages (64
  positions, every layer) are copied to host memory as they fill, so when its slot goes to
  another conversation they are kept there instead of dropped; its next turn copies the
  longest stored prefix back and prefills only the rest. Same bits: the rows are moved, not
  recomputed (a tail under 64 positions is prefilled again). `auto` is the smaller
  of 4 GiB and a quarter of the RAM available after the model (the host's `MemAvailable`, or a
  memory cgroup's room if that is less). It is **off when a cgroup limit is the tighter bound**
  (a container, a systemd scope with `MemoryMax`): pinned memory counts against the limit and
  can't be reclaimed. The banner says the size, or why it is off. BYTES (K/M/G suffixes) sets
  the size, under a cgroup too; `0` or `off` (the default) turns the tier off. On the GPU
  the **whole size is page-locked (unswappable) RAM from server start**, used or
  not: a thread pins it in 64 MiB pieces while the server already runs (or, with
  `AXON_KV_RAM_ALLOC=start`, all of it before it answers). `auto` therefore pins up to 4 GiB
  at start. With the GPU's paged pool, pages are copied as they fill only into the tier's free room
  (while requests decode, at most 16 MiB a step; an idle server copies the rest in 64 MiB pieces and
  takes a new request between two), and a slot's pages not yet stored are copied when it gives its
  rows up. The copies of pages the pool still holds are evicted first (newest first), then RAM's own
  pages (least recently used), so its useful capacity is about `--cache-ram` (a 7B f16 position is
  56 KiB). The one cache (the CPU, `--parallel 1`) stores its new pages after each request. Size it to
  what you keep warm, and leave the rest of the machine room: under a
  cgroup memory limit, pinned memory can't be reclaimed (the CPU path's pages are ordinary
  memory, taken as they are stored). Off where the cache has
  no page image (a CPU/GPU split, several GPUs, MoE, deepseek2). It lives as long as the
  process (surviving a restart is a later stage). `/metrics` adds `axon_kv_ram_*`
  (`axon_kv_ram_gpu_pages`: stored pages the GPU pool still holds; `axon_kv_ram_spilled_pages_total`:
  pages copied as a slot gave its rows up), and `AXON_SERVE_TRACE`'s step lines its copies.
- A request's `priority` field is accepted and ignored (requests start in arrival order, or
  shortest prompt first by default; `--admit-order fifo` for arrival order).

## Tool calling

`tools`, `tool_choice` and tool turns in the history (`role: "tool"` results, assistant
messages with `tool_calls` and null or absent `content`) work on `/v1/chat/completions`, so
agent harnesses that talk to a local OpenAI-compatible endpoint can run on axon:

```bash
axon serve models/Qwen3-8B-Q4_K_M.gguf --device cuda --ctx 32768 --kv q8 --alias qwen3-8b
```

- The tools are rendered as the model's own chat template does (the GGUF's
  `tokenizer.chat_template`, checked against transformers' rendering in tests). Qwen3 and
  Qwen2.5 get the Hermes `# Tools` / `<tools>` block in the system message, call with
  `<tool_call>{"name": ..., "arguments": ...}</tool_call>`, and see results as
  `<tool_response>` blocks. Llama 3.1/3.2 get the tools in the first user message, call with
  a JSON reply (`{"name": ..., "parameters": ...}`) and see results as `ipython` turns (one
  call per message, as the template requires). gpt-oss (harmony) gets the tools as
  TypeScript in the developer message (`# Tools` / `namespace functions { ... }`), calls on
  the commentary channel (`<|channel|>commentary to=functions.NAME <|constrain|>json
  <|message|>{args}`), and sees results as `functions.NAME` messages; its analysis channel
  stays `reasoning_content`. Its calls are read from the special tokens only: text the
  model writes (say, quoting the call syntax) is never a call. A call is made as soon as
  its arguments are a whole JSON object, and that ends the reply (one call per reply; the
  model's `<|call|>` isn't waited for). A recipient in the role section counts only right
  after the role (`assistant to=functions.NAME`). A reply that opens no channel header is
  the answer, streamed and cut by `stop` strings as without tools. Past calls are
  rendered in the template's history form
  (`<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>{args}<|call|>`).
  Other chat formats answer a tool request with a 400. A request without tools or tool
  turns is rendered exactly as before.
- The reply's calls come back as OpenAI `tool_calls`, with `finish_reason: "tool_calls"`
  (`"length"` if the token limit cut the reply). Text outside the blocks streams as `content`,
  and blank lines that only separate text from calls are dropped. Not streamed:
  `message.tool_calls`, with `content` null if there was no text. Call ids are `call_` plus
  24 hex digits.
- **Streamed, a call goes out as it is generated** (#260), as OpenAI and llama-server send it:
  a `delta.tool_calls` chunk with `index`, `id`, `type` and `function.name` as soon as the
  call's name is read and its arguments open with `{`, then a chunk of `function.arguments`
  per piece of text. On `/v1/responses` the call is its own item at that point
  (`response.output_item.added`, then `response.function_call_arguments.delta` events). So a
  call that runs on (a looping argument, and no `max_tokens`) is visible from its first
  tokens; before, nothing of it went out until its block closed, and a client with an idle
  timeout hung up (codex's is 300 s: #260's runaway call was silent for 340 s).
- **Which blocks are calls.** Up to that point a block is held, and decided as soon as its text
  can't begin a JSON object any more: then it is not a call, and it comes back as `content`,
  tags and all, from there on. A block whose object closes without arguments that open with
  `{` (a string or array `arguments`, or the name after them) is decided whole: a call if it
  is a JSON object with a string `"name"`, else content. A string `arguments` value is used as
  the arguments if it is itself JSON, else returned as its JSON string literal.
- **Once a call's head is out it is a call**, as in OpenAI's API, where `arguments` is the
  model's text and clients check it. Arguments that stop being JSON (an unescaped quote, a raw
  newline, a bad escape: #259's failures) are returned as written up to the block's end, less
  the object's closing `}`. A reply cut off mid-call is a call with what was written. Either
  way `arguments` isn't a valid JSON object, and the call is counted in
  `axon_tool_calls_invalid_total`. A harness then reports the call as failed to the model,
  which can try again; before, the block came back as text, which ended codex's turn. This
  holds **streamed or not**: a whole (`"stream": false`) reply decides its calls the same way,
  so the two always agree (`AXON_TOOL_STREAM=0` restores the old rule for both). On
  `/v1/responses` a call the token limit cut (the reply's last item, its arguments not a JSON
  object, in an `incomplete` response) has `status: "incomplete"`, as OpenAI marks it; any
  other call is `completed`. A stop string that hits inside a call's arguments also ends it
  with what was written, with `finish_reason: "tool_calls"`; on `/v1/responses` that call is
  `completed` (the finish is `stop`, not the token limit). Other text after the call's object
  ends the block there: `<tool_call>` opens the next call (#259's "a valid object followed by
  a second `<tool_call>`"), anything else is content, less the block's own `</tool_call>` if it
  comes later (the first one before the next `<tool_call>`), so a call's own stray
  `</tool_call>` doesn't show in `content`. A block that never becomes a call still comes back
  as content, tags and all, and so does a second stray `</tool_call>`.
- **Calls are held to the tools offered and to JSON** (constrained decoding, #259), on the
  Hermes models (Qwen2.5 and Qwen3 templates), by default. From the reply's `<tool_call>` (its
  token, or the text spelled out; not inside Qwen3's reasoning) to its `</tool_call>`, every
  token is picked from those that keep the call one of
  `{"name": NAME, "arguments": OBJECT}`: NAME one of the request's tool names (so a model
  can't call `apply_patch` when it wasn't offered: #259's codex runs did, 6 times), OBJECT
  any JSON object, nested at most 64 deep, with no special token (EOS, `<|im_end|>`, a second
  `<tool_call>`) inside a string. After the call the reply is free again, so it can make
  more calls or write text. `tool_choice` `required` and named start the reply inside the
  call (the prompt opened it), held from its first token. What doesn't change: requests
  without tools, `tool_choice: "none"`, text before a call, and every token of a reply that
  makes no call. Held calls can't run on for ever (#272, where 13 of 15 malformed calls
  were runaways that hit `max_tokens`):
  - **Whitespace** outside strings is one space, or one or two newlines and at most 20
    spaces or tabs, as llama.cpp's grammars bound it, in the call's frame and its arguments.
  - **A stalled call is cut off.** When the last 2,048 tokens of a call repeat one block of
    at most 256 tokens (`false, false, ...`, a string of `\n\n\n...`, `--exclude=*.pyc
    --exclude=*.pyc ...`, a list in a loop, a 72-token block of code 53 times), it is closed
    the shortest way from there: the string ends, then each open array and object, then
    `</tool_call>`, and the reply ends after it (only its stop token is allowed), so a model
    in a loop can't open the next call and stall again. **The client gets a well-formed
    call whose arguments are cut where it stalled:** a file written short, a command run
    short. The close can also invent its last piece: after a `,` an array needs one more
    item and gets one digit (a number is the shortest JSON value), an object one more
    member (`"": 0`), a number one more digit. Each stall is a line on stderr, whatever the
    log level (`axon serve: #12: a held tool call stalled after 2077 tokens (its last 2048
    repeat one block of 72 tokens): closed the shortest way, so its arguments are cut there
    and may end with an invented item; the reply ends after it (AXON_CONSTRAIN_STALL=0 turns
    this off)`), and counts in `axon_constraint_stalls_total`. In the acceptance runs' 6,333
    calls 11 of the 13 runaways trip it (their loops are 2,144 to 50,831 tokens' worth of
    text); the other two, a `, false` loop and a list loop, ended by themselves first, after
    1,963 and 623. The other calls stay far from it (383 at most in a block repeated). Data
    that repeats one block 8 times or more trips it: 256 identical `<div class="cell"></div>`
    lines after about 2,085 tokens (205 cells written, the closing tags lost), 100 identical
    log lines after about 2,070, a 32×32 matrix of zeros after about 2,070, 150 identical
    JSON objects after about 2,060. Data that doesn't repeat a block isn't cut, however few
    tokens it uses (400 dice rolls, a 30×20 tile map of 4 tile ids, a 2,048-bit string).
    **`AXON_CONSTRAIN_STALL=0`** turns the closing off alone: a call in a loop then runs to
    `max_tokens`, its name and JSON still held (the grammar's hash in the fingerprint says
    `stall=off`). Until Fable's review of #274 two more rules held a call: an array kept its
    first item's kind (it rewrote real arguments without a trace: `[["Alice", 30]]`,
    `["$qty", 250]`, `[1.5, null, 2.0]`), and a call whose last 1,024 tokens were at most 8
    distinct ones was cut (it caught one runaway more, and cut 400 dice rolls); both are
    gone. When in doubt the rule doesn't fire.
    **What it misses: a loop that breaks off and starts over.** Replayed with seeds 1 and 2,
    one of the 13 (Qwen2.5-7B, a pi `edit`) repeats `'convert_to_all_except', ` (6 tokens)
    up to 284 times in a string, ends the string, writes `"newText"` again in the same
    object (JSON allows a repeated key), copies the line's first 65 to 91 tokens and loops
    again. No stretch repeats one block for 2,048 tokens (1,711 at most), so the call runs
    on to `max_tokens` and arrives cut (`finish_reason` `length`), as it would without the
    constraint; with greedy decoding it stalls (after 2,158 tokens). The few-ids rule caught
    it at 1,024 tokens (not at 2,048), but cut 15 of the 74 legit cases. A looser cycle rule
    would have to fire when 90 % of 2,048 tokens equal the token a block back (seed 2's loop
    reaches 91 %), and that cuts sparse data: a 40×30 tile map with 10 % walls, a 60×20
    one-hot CSV, a 24×24 0/1 mask. So the miss stays; the harness bar counts such a call as
    malformed.

  Valid isn't right: where the model would have written an unescaped quote, the string ends
  there and the call goes on from it, so the call parses but may not be what was meant (the
  harness tells the model, which can retry). What it can't write whole: more blank space than
  llama.cpp allows, or more than 2,048 tokens of one block written over and over (the call is
  cut, above); with greedy decoding a retry stops at the same place, so such a file can't be
  written through a held call (`AXON_CONSTRAIN_STALL=0` or `--no-constrain` can). Per-tool argument schemas are #259's next step (PR C).
  `max_tokens` can still cut a call short (a call with invalid arguments, as above). The
  identity changes with it: `NUMERICS_VERSION_EXACT` is 2 (every fingerprint moved), and a
  tools request's `system_fingerprint` names the grammar
  ([determinism.md](determinism.md#the-identity-tuple-and-system_fingerprint)).
  **`--no-constrain`** or **`AXON_CONSTRAIN=0`** turns it off: the server then runs as before
  it, token for token, and its identity says `no-constrain`. `AXON_CONSTRAIN=1` (or unset) is
  on; any other value refuses to start. The banner ends with `constrained tool calls
  (hermes-json <hash>)` when it's on; a Hermes model it can't hold (a vocabulary without
  every byte as a token, or no `<tool_call>` token) gets a line saying why.
- **`AXON_TOOL_STREAM=0`** (exactly `0`) turns this off: every call goes out once its block is decided, as
  two chunks (id and name, then all of `function.arguments`; on `/v1/responses`, after the
  text when the reply ends), and a block that isn't a JSON object with a string `"name"`
  (invalid JSON, or cut off mid-call) is returned as `content`, so `arguments` is always
  valid JSON. A call whose `</tool_call>` the model left out ends where the next `<tool_call>`
  begins, if its block is a call up to there (#259). The ollama routes always work this way
  (an ollama call is a whole object).
- Qwen3's reasoning still comes as `reasoning_content`. gpt-oss: its calls stream the same
  way once their arguments open with `{` (after nothing but blanks). A call whose arguments
  never do (a number, an array, nothing, other text first) is not a call: its text comes back as `reasoning_content` (it was on the
  commentary channel), never as `content`, and is counted in `axon_tool_calls_dropped_total`.
  With `AXON_TOOL_STREAM=0` that also holds for a call that ends without a whole JSON
  object (a header, its `<|call|>`, the token limit).
- The parser reads tags and JSON, not prose. `<tool_call>` written in ordinary text opens a
  block (which comes back as content if it isn't a call). In a call's arguments, a
  `</tool_call>` inside a JSON string is the string's text, so a call writing a file that
  mentions the tag stays whole; anywhere else it ends the block. A string the model leaves
  open usually goes bad at the newline Hermes writes before the tag, so the tag still ends
  the block. If the model also leaves out that newline, the tag stays string text and the
  open string swallows what follows: the rest of the reply goes into the call's (invalid)
  arguments, and a later `<tool_call>` in it is lost. That is rare. #266 tracks the fix:
  arguments that have gone bad end at the next `<tool_call>`. With
  `AXON_TOOL_STREAM=0` the tag ends the block wherever it is, as before: inside a string, the
  block fails to parse and is returned as content.
- The format is chosen per model family from the vocabulary (ChatML with or without
  `<think>`, or Llama 3's headers), not read from the GGUF's template. Models of a known
  family with a different tool format, such as Qwen3-Coder's XML `<function=...>` calls,
  are not supported.
- `tool_choice`: `auto` (default); `none` (the tools are not shown to the model, and its
  reply is plain content); `required` or `{"type": "function", "function": {"name": ...}}`.
  The last two start the reply with the opening of a call (the named one: through its
  `"arguments": `), so the model must call. Qwen3 and gpt-oss then skip their reasoning.
  gpt-oss: a named choice opens `<|channel|>commentary to=functions.NAME <|constrain|>json
  <|message|>` (the form the model writes calls in) and keeps every tool in the prompt.
  `required` with one tool is that tool's named choice; with several it is a 400, since a
  prefill can't force one of them (constrained decoding of harmony calls: #116).
  Without `tools` they are a 400.
  `parallel_tool_calls`, `strict`, `store`, `reasoning_effort` and `prompt_cache_key` are
  accepted and ignored.
- Context: an agent's system prompt and tool schemas can be 15-25k tokens; the default
  context is the model's trained length where it fits (above), so no `--ctx` is needed.
  `--kv q8` halves the cache's memory (more slots, or the whole context on a smaller GPU). The prefix cache works across tool turns: the next
  request resends the history with the results appended, and only the tokens after the
  previous reply's start are run again, as long as no other request evicted the prefix in
  between. There is one KV cache, so a harness's side requests (title generation,
  summaries) or a second client make the next turn prefill the system prompt and tools again.
- Harnesses: anything speaking OpenAI Chat Completions, e.g. Synaps (`provider.local.url`),
  opencode and crush (an OpenAI-compatible provider whose base URL is
  `http://127.0.0.1:8080/v1`), and pi (an `openai-completions` model in `models.json`).
  Codex CLI and pi's `openai-responses` speak the Responses API, and ollama clients the
  ollama API: both are served too (below).

## The Responses API and the ollama API

`POST /v1/responses` (OpenAI's Responses API, what codex speaks) and ollama's native API
(`POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `POST /api/show`, `GET /api/ps`,
`GET /api/version`, `GET /`) are served next to chat completions, so a client written for
OpenAI or for ollama works unchanged. Each request is translated into the chat-completions
request and run the same way: the same `--api-key` gate, size limits, queue, batching and
cache, and greedy output identical to `/v1/chat/completions` for the same messages (a test
checks it on every route). [docs/serve-apis.md](serve-apis.md) lists every field.

```bash
# codex, against a local provider (~/.codex/config.toml, or a CODEX_HOME of its own):
#   model = "qwen3-8b"
#   model_provider = "axon"
#   [model_providers.axon]
#   name = "axon"
#   base_url = "http://127.0.0.1:8080/v1"
#   wire_api = "responses"
codex exec "Read hello.txt and tell me the magic number"
# or codex's own ollama provider:
CODEX_OSS_BASE_URL=http://127.0.0.1:8080/v1 codex exec --oss --local-provider ollama -m qwen3-8b "..."
# any ollama client:
OLLAMA_HOST=http://127.0.0.1:8080 ollama run qwen3-8b
```

- `/v1/responses`: `input` (a string, or `message`, `function_call`,
  `function_call_output` and `reasoning` items), `instructions`, function `tools` (other
  tool types, such as codex's `web_search` and `namespace`, are dropped), `tool_choice`,
  `stream` (OpenAI's events, `response.created` through `response.completed`),
  `max_output_tokens`, sampling, `usage`, and `previous_response_id` (responses with
  `store`, the default, are kept in memory: the last 256). Structured output, background
  responses and conversations are a 400.
- ollama: `stream` defaults to true (NDJSON); `options` carries the sampling settings
  (`num_predict` is the token limit; `num_ctx` and the runtime options are ignored: the
  server's `--ctx` applies); `think`; tool calls with object arguments; ollama's counts and
  durations. The model answers to its file name (or, given by a name `axon pull`
  resolved, such as `axon serve qwen3-8b`, that name) and `--alias`, with or without
  `:latest`.
  `format` (JSON / structured output), images and embeddings are refused with a clear error.
