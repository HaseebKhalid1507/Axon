# 2026-10-17: `axon serve` speaks the Responses API and the ollama API

Branch `feat/serve-apis`. Roadmap "Also open": tool calling on `/v1/responses` (codex) and the
ollama-native API.

## What changed

- **`src/serve/responses.rs` (new): `POST /v1/responses`.** The body is translated into a
  `/v1/chat/completions` body and parsed by `api::parse`, so it gets the same validation and
  the same `GenReq`. Input items map to chat messages: `instructions` becomes a leading
  system message; message items with text parts become messages; a run of `reasoning`,
  assistant `message` and `function_call` items becomes one assistant turn with
  `reasoning_content`, `content` and `tool_calls`; `function_call_output` becomes a `tool`
  message. Flat `function` tools become chat tools. Other tool types (codex always sends
  `web_search` and a `namespace`) are dropped. The translation uses tools.rs's
  order-preserving `J`, so tool schemas render with the request's key order, as on chat
  completions. Answers come back whole or as OpenAI's SSE events with `sequence_number`. A
  reply is `[reasoning?, message?, function_call...]`, and sent back as input it renders
  exactly as the chat-completions history would. `previous_response_id` is supported (it
  was cheap): an in-memory store of the last 256 `store: true` responses, capped at 64 MiB.
  An unknown id is a 400 `previous_response_not_found`.
- **`src/serve/ollama.rs` (new):** `/api/chat` (NDJSON stream by default, `think`,
  `options`, tool calls with object arguments), `/api/generate` (chat, or `raw` as
  `/v1/completions`), `/api/tags`, `/api/show`, `/api/ps`, `/api/version` (0.14.1: codex's
  ollama provider needs at least 0.13.4 for the Responses API), and `GET /` (the CLI
  heartbeat). The ollama "load" request (no prompt or messages) gets an immediate
  `done_reason: "load"` reply. `/api/embed`, `/api/embeddings` and the model-management
  routes return 501 with a reason. `format`, images, `suffix`, `template` and `context` are
  400. Errors use ollama's `{"error": ...}` shape.
- **`src/serve.rs`:** `completion()` now only parses. The rest moved, unchanged, into
  `run_request(c, req, sh, GenReq, Out, arrived)`. `Out` (Chat, Text, Responses, Ollama)
  picks the error shape, the param name in size errors, the id prefix, and the writer at
  the end. `whole()` and `stream()` for chat and completions are untouched. Route table:
  `/v1/responses`, plus `ollama::methods`/`serves`/`route`. `Loaded` gained `arch` and
  `params`, and `Shared` gained `path` (for `/api/show` and `/api/tags`). engine.rs, the
  engine loop and `plan_slots` were not touched.
- `src/serve/ops.rs`: new `/metrics` route labels. `scripts/fuzz_serve.py`: `--apis` cases,
  run in every mode: bad shapes, 60 type-swapped mutations, the describe and refused routes.
  It also accepts ollama's error shape. Fixed a stale check: `/slots` has had no `solo`
  entry since #60 (d657757).
- Docs: new docs/serve-apis.md (every route and field), the README serve section, and the
  roadmap.

## Why

Codex speaks only the Responses API, and much of the local-model ecosystem speaks ollama's
API: the ollama CLI and libraries, Open WebUI, and opencode through ollama-ai-provider-v2.
Neither could use axon before this change. The design rule was **no new engine path**. Each
route becomes the existing request type, so the auth gate, size limits, queue, batching,
cache and determinism are all shared.

## Shapes checked against

- A capture of codex 0.161.0's real request, using a dummy server: `tools` holds `function`,
  `namespace` and `web_search`; `input` holds `developer` and user message items with
  `input_text` parts; it also sends `reasoning: {summary: auto}`, `store: false` and
  `include: [reasoning.encrypted_content]`.
- codex-rs `codex-api/src/sse/responses.rs`: the events it handles. `response.created` reads
  `response.id`; `output_item.done` parses a `ResponseItem`; `response.completed` reads
  `usage` with `input_tokens`, `output_tokens`, `total_tokens` and optional details. For
  `response.incomplete`, any reason other than `interrupted` is an error to codex.
- codex-rs `protocol/src/models.rs`: `ResponseItem` Message, Reasoning (`summary` is
  required) and FunctionCall (`arguments` is a string, plus `call_id`).
- codex-rs `ollama/`: the minimum version 0.13.4, `/api/tags`, `/api/version` and `/api/pull`.
- openai-python 3.x: typed `responses.create` and `responses.stream`, plus
  `get_final_response()`, run in e2e. ollama-python 0.6.3: `list`, `show`, `ps`, `chat`
  (with tools, streamed, with `think`), `generate` and `embed`.

## Verification

| gate | result | job |
|---|---|---|
| unit tests (responses/ollama parse, shapes, bad bodies, deep nesting, store bound) | 21/21 | in the serve:: runs |
| new serve:: tests v1 (first run) | 13/14: `serve_responses_tool_loop` failed. Response ids (`resp-{n}{created}`) collided between two test servers started in the same second, so the process-wide store returned another server's conversation. Fixed: `responses::fresh_id`, a process-wide sequence | 1017-172040-871c |
| md5 `gen` + serve, main 54a75cb vs candidate (h1) | 26/26 same | 1017-172756-1d14 (bella) |
| fuzz `--apis` / `--ops` / plain (v3) | 115/0, 0 findings, 298/0; rc 0 | 1017-174323-b3ea |
| fuzz `--ops` on main's binary with main's script | 3 findings, rc 1. All three are `KeyError('solo')` / `slots:shape`: main's script is stale since #60. This is not a regression; the script is fixed on this branch | 1017-173240-ffb3 |
| first fuzz (h1, unfixed script) | apis 115/0, plain 298/0, ops rc 1 from the same 3 `solo` findings | 1017-172816-c5dc |
| clippy -D warnings (bin + tests), rustfmt | clean | |

Greedy determinism: `serve_apis_match_chat_completions` sends the same messages to
`/v1/chat/completions`, `/v1/responses` (whole and streamed), `/api/chat` (whole and
streamed) and `/api/generate`. All must give the same text, completion tokens and prompt
tokens, and a forced tool call must be the same call on every route. It covers 0.5B and
Qwen3-0.6B (thinking off) on the CPU, and 0.5B on CUDA. On bella (Qwen3-8B, CUDA, q8 KV),
e2e checked 2 prompts on all 4 routes: identical text and token counts (MEASURED, job 48ac).

## Real harness runs (bella, Qwen3-8B Q4_K_M, `--ctx 32768 --kv q8`, through axon-run)

Transcripts: `~/.cache/axon-apis/transcripts/`.

| client | route(s) | result (job 1017-172755-48ac, binary h1) |
|---|---|---|
| codex exec, custom provider `wire_api = "responses"` | /v1/responses | PASS: ran `cat hello.txt`, answered 4817 |
| codex exec `--oss --local-provider ollama` (`CODEX_OSS_BASE_URL`) | /api/version, /api/tags, /v1/responses | PASS: reasoning streamed, `cat hello.txt`, 4817 |
| pi, `openai-responses` api | /v1/responses | PASS: 4817 (call plus answer: 2 requests in the serve log) |
| openai-python SDK | /v1/responses | PASS: tool loop via `previous_response_id`; streamed loop with items sent back (cache 707/730 reused); reasoning item |
| ollama-python 0.6.3 | /api/chat, /api/generate, tags/show/ps, embed | PASS: tool loop whole and streamed with thinking; generate; embed gives a clear 501 |
| `ollama` CLI (`OLLAMA_HOST`) | /, /api/tags, /api/show, /api/generate, /api/chat | `list` and `run` OK. `show` failed: the CLI sends `"model": ""` beside `"name"`. Fixed in fe084e2; rerun pending (9fe3) |
| opencode + ollama-ai-provider-v2 | /api/chat | did not run: opencode's managed service port was taken. Harness quirk; the earlier tools harness passes `--standalone`. Rerun pending (9fe3) |

## What didn't work / limits

- `format` / JSON-schema output is refused. It needs grammar-constrained sampling, which
  axon doesn't have. `text.format` other than text on Responses is refused too.
- `output_tokens_details.reasoning_tokens` is the reasoning text re-encoded (+2 for the
  think markers): the engine reports text, not a per-channel token count.
- Responses function calls are streamed when the reply ends, after the message item. The
  tool parser decides a call whole, so each call's arguments come in one delta.
- `/api/show` reports the served context (`--ctx`), not the trained context, as
  `<arch>.context_length`. Clients size their prompts by it.
- `num_ctx` is ignored: one server, one context.

## Review round 1 (Shady, SHIP-AFTER-FIXES 7/10)

- Response ids were `resp-<seq>-<start>`: guessable, and with `store` on by default any key
  holder could continue another client's conversation. Now `resp-` + 128 bits from
  /dev/urandom + the process sequence number (unique even with equal random bits). The docs
  say a stored response is readable by whoever holds its id.
- Three errors in `run_request` (no chat template 400, queue full 503, shutting down 503)
  still used `api_err`, the OpenAI shape, on the ollama routes: now `out.err`.
  `serve_caps_and_slow_reader` asserts the queue-full 503 is exactly
  `{"error": "too many queued requests"}` on /api/chat (and OpenAI's on /v1/responses), and
  the fuzzer checks /api/* errors are exactly `{"error": "..."}`.
- docs: ollama's sampling defaults (top_k 40, top_p 0.9, repeat_penalty 1.1, num_ctx 2048)
  are not applied.
- A message item's `status` follows the response's (`incomplete` when cut off);
  `reasoning_tokens` is counted from the reasoning text.
- NITs: `serves()` compares methods exactly (split, not substring); the GET routes answer
  consistently while loading (above); the store sizes a conversation without printing it.
