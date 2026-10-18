# `axon serve`: routes and fields

One HTTP/1.1 server, one model, four request APIs. Every generation route turns its body
into the chat-completions request (`src/serve/api.rs` `GenReq`) and runs it the one way
(`run_request` in `src/serve.rs`): the same `--api-key` gate (checked before the body is
read), the same 1 MiB body limit, `--max-prompt-tokens` and context checks, the same queue,
batching, prefix cache and engine. Only the answer's shape differs. Greedy output is
identical across routes for the same messages (`serve_apis_match_chat_completions`).

| route | method | module |
|---|---|---|
| `/v1/chat/completions` | POST | api.rs, tools.rs |
| `/v1/completions` | POST | api.rs |
| `/v1/responses` | POST | responses.rs |
| `/v1/models`, `/identity`, `/health`, `/metrics`, `/slots` | GET, HEAD | serve.rs, ops.rs |
| `/api/chat`, `/api/generate`, `/api/show` | POST | ollama.rs |
| `/api/tags`, `/api/ps`, `/api/version`, `/` | GET, HEAD | ollama.rs |
| `/api/embed`, `/api/embeddings` | POST | 501: not supported |
| `/api/pull`, `/api/push`, `/api/create`, `/api/copy`, `/api/delete` | POST (DELETE) | 501: one model per server |

Chat completions and completions: the field table is the module doc of `src/serve/api.rs`;
tool calling is in the README ("Tool calling").

## `POST /v1/responses`

Checked against codex's client (`codex-rs/codex-api/src/sse/responses.rs`, the
`ResponseItem` types in `codex-rs/protocol/src/models.rs`, codex 0.161.0 on the wire) and the
openai-python SDK's typed `responses.create` / `responses.stream`.

| field | axon |
|---|---|
| `input` | a string (one user message), or items: `message` (or a bare `{role, content}`; roles user, system, developer = system, assistant; content a string or `input_text` / `output_text` / `text` / `refusal` parts, joined with newlines), `function_call` (`call_id`, `name`, `arguments` string), `function_call_output` (`call_id`, `output` string or text parts), `reasoning` (its `content` text, else its `summary`; encrypted-only items carry nothing). A run of reasoning, assistant message and function_call items is one assistant turn. Images, files, audio, `item_reference` and other item types: 400 |
| `instructions` | a leading system message (not carried over by `previous_response_id`) |
| `tools` | `function` tools (`name`, `description`, `parameters`, `strict`), rendered as chat tools in the request's key order. Other types (`web_search`, `namespace`, `custom`, `local_shell`, `file_search`, ...) are dropped and not echoed |
| `tool_choice` | `auto`, `none`, `required`, `{"type": "function", "name": ...}`; hosted-tool choices and `allowed_tools`: 400 |
| `stream` | server-sent events (below) |
| `max_output_tokens` | the generation limit (at least 1) |
| `temperature`, `top_p` | as on chat completions; axon's extensions (`top_k`, `min_p`, `seed`, `repetition_penalty`, `repeat_last_n`, `stop`) are passed through too |
| `reasoning.effort` | `"none"`: Qwen3 answers without thinking; other values: the model's default. `reasoning.summary` is ignored |
| `previous_response_id` | continues a stored response (its input and output, not its instructions). Responses with `store` (default true) are kept in memory: the last 256, at most 64 MiB, until the server restarts. Unknown: 400 `previous_response_not_found` |
| `store` | false: not kept (codex sends false and resends the items). A stored response is readable by anyone who holds its id (and, with `--api-key`, the key): ids are `resp-` + 128 random bits, unguessable, but not tied to a client; send `store: false` for conversations other key holders must not continue |
| `text.format` | `text` only; `json_schema` / `json_object`: 400 |
| `background`, `conversation`, `prompt`, `top_logprobs` > 0 | 400 |
| `parallel_tool_calls`, `metadata`, `user`, `include`, `truncation`, `prompt_cache_key`, `service_tier`, `safety_identifier`, `client_metadata`, `max_tool_calls`, `stream_options` | accepted; `metadata`, `user` and `parallel_tool_calls` are echoed, the rest ignored (`truncation` is always `disabled`: a prompt past the context is a 400) |

The response object: `id` (`resp-...`), `object: "response"`, `created_at`, `status`
(`completed`; `incomplete` with `incomplete_details.reason: "max_output_tokens"` when cut
off by the limit or the context; `failed` on an engine error), `model`, `output`, `usage`
(`input_tokens`, `input_tokens_details.cached_tokens`, `output_tokens`,
`output_tokens_details.reasoning_tokens` (the reasoning text encoded again, plus the
`<think>`/`</think>` markers: the engine reports text, not tokens per channel, so this equals
the generated count unless the model wrote a split the tokenizer wouldn't make),
`total_tokens`), `system_fingerprint`, and the echoed request fields.

`output` is `[reasoning?, message?, function_call...]`: a `reasoning` item (`summary: []`,
`content: [{"type": "reasoning_text", ...}]`) when the model thought, a `message` item
(`output_text` part; `status` `incomplete` when the response is) when there is text (or
nothing else), then one `function_call` item per
call (`id` `fc_...`, `call_id` `call_...`, `arguments` a JSON string). Sent back as input,
these render as the chat-completions history of the same reply.

Stream events, each `event: <type>` + `data: {...}` with `type` and `sequence_number`:
`response.created`, `response.in_progress`; per item `response.output_item.added`, for a
message `response.content_part.added`, `response.output_text.delta`...,
`response.output_text.done`, `response.content_part.done`; for reasoning
`response.reasoning_text.delta`..., `response.reasoning_text.done`; for a call
`response.function_call_arguments.delta` (the whole arguments) and `.done`; then
`response.output_item.done`; last `response.completed` or `response.incomplete` with the
whole response. An engine error: `error`, then `response.failed`. Function calls are sent
when the reply ends, after the message item (the parser decides each call whole).

## The ollama API

Field names follow ollama's `docs/api.md`. Every error on an `/api/*` path is
`{"error": "..."}` and nothing else: the routes' 400/404/501, the 503s (loading, queue full,
shutting down), a wrong method (405, with `Allow`), an unknown `/api/...` route (404), and
what the HTTP layer answers before routing once the request line is read (401 without the
key, 413, a broken body). Only a request whose head doesn't parse (431, a malformed request
line), whose path isn't known, gets OpenAI's shape.

**Sampling defaults are axon's, not ollama's.** ollama fills in `top_k` 40, `top_p` 0.9,
`repeat_penalty` 1.1 and `num_ctx` 2048 (plus a Modelfile's `PARAMETER`s) when a request
leaves them out; axon doesn't. An option left out takes `axon serve`'s flag (`--temp`,
`--top-k`, `--top-p`, `--repeat-penalty`, ...; their defaults: `axon chat`'s, with the repeat
penalty off), and the context is the server's `--ctx`. To get ollama's sampling, send those
values in `options`.

`POST /api/chat`

| field | axon |
|---|---|
| `model` | the file name or `--alias`, with or without `:latest`; anything else: 404 |
| `messages` | `role` (system, user, assistant, tool), `content` (string), `thinking` (an earlier reply's reasoning), `tool_calls` (`function.name`, `function.arguments` object), `tool_name` (ignored). `images`: 400 |
| `tools` | as on chat completions (`{"type": "function", "function": {...}}`) |
| `stream` | default true: NDJSON (`application/x-ndjson`), one object per line |
| `think` | false: no thinking; true or a level: think (Qwen3). The reply's reasoning is `message.thinking` |
| `format` | `"json"` or a schema: 400 (no grammar-constrained sampling) |
| `options` | `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `repeat_penalty`, `repeat_last_n`, `presence_penalty` / `frequency_penalty` (0 only, as on chat completions), `num_predict` (the limit; -1 / -2: up to the context; 0: 400). `num_ctx` and the runtime options (`num_gpu`, `num_thread`, `mirostat`, ...) are ignored: the server's `--ctx` and flags apply |
| `keep_alive` | ignored (the model stays loaded) |

Each line: `{"model", "created_at", "message": {"role": "assistant", "content", "thinking"?,
"tool_calls"?: [{"id", "function": {"index", "name", "arguments": {...}}}]}, "done": false}`;
the last: `"done": true`, `done_reason` (`stop`, `length`), `total_duration`,
`load_duration` (0), `prompt_eval_count` (the whole prompt, cached tokens included),
`prompt_eval_duration`, `eval_count`, `eval_duration` (nanoseconds). A call whose arguments
aren't a JSON object gets `{}`. No messages: ollama's load request, answered at once with
`done_reason: "load"`.

`POST /api/generate`: `prompt` with `system` as a chat, or `raw: true` (the text as
`/v1/completions` takes it, no template); `stream`, `think`, `options` as above; the text
is `response` (reasoning: `thinking`). `suffix`, `template`, `context` and `images`: 400.
No prompt: the load request.

While the model loads, `GET /`, `/api/version`, `/api/tags` and `/api/ps` all answer 200:
`/api/tags` lists the model from its file (size, date; no digest, family or size label
yet), and `/api/ps` lists nothing (no model is loaded yet), as ollama does with a model on
disk and none in memory. `/api/show`, `/api/chat` and `/api/generate` are 503 until it has
loaded.

`GET /api/tags` and `/api/ps`: the model under each of its names, with `size` (the file),
`digest` (the identity's model digest), `modified_at` (the file's), `details` (`format`
gguf, `family` the architecture, `parameter_size`, `quantization_level` the most common
tensor type); `/api/ps` adds `context_length` (the served context), `size_vram`,
`expires_at`. `POST /api/show` (`model` or `name`): `details`, `model_info`
(`general.architecture`, `general.parameter_count`, `<arch>.context_length` = the served
context, which is what clients size their prompts by), `capabilities` (`completion`,
`tools` with a chat template, `thinking` for Qwen3). `GET /api/version`: `0.14.1`, the
ollama API level implemented (codex's ollama provider needs at least 0.13.4 for the
Responses API). `GET /` answers `Ollama is running` (the CLI's heartbeat).

## Clients run against it

See the engineering log (docs/log/2026-10-17-serve-responses-ollama.md) for the runs and
transcripts: codex `exec` (a custom `wire_api = "responses"` provider, and `--oss
--local-provider ollama`), pi (`openai-responses`), opencode (`ollama-ai-provider-v2`), the
`ollama` CLI, and the openai and ollama Python libraries.
