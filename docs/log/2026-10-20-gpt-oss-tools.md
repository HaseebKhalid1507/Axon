# serve: tool calling for gpt-oss (harmony)  (PR #111)

- Date: 2026-10-20. Branch: feat/oss-tools. Base: main cb53181 (merged 6f1efb5 in, no code change). Status: open
- Pick / design: issue #108 (agent harnesses on gpt-oss-20b got `400 tools: tool calling is not supported
  for this model's chat format`). Follows #26 (tool calling, Qwen/Llama) and #70 (Responses, ollama), on the
  harmony framing of [2026-10-18-gpt-oss-s0.md](2026-10-18-gpt-oss-s0.md).

## What changed
- `src/serve/tools.rs`: `Family::Harmony`. `harmony()` renders a tool request as gpt-oss's GGUF template does:
  - system message with the template's tools line ("Calls to these tools must go to the commentary channel:
    'functions'.");
  - the developer message's `# Tools` / `## functions` / `namespace functions { ... }`, with
    `ts_namespace` / `ts_type` porting the template's `render_tool_namespace` / `render_typescript_type`:
    nested objects, arrays, string enums as unions, optional `?`, descriptions as `//` comments, defaults,
    type lists, `oneOf`, `nullable`, and the template's whitespace quirks;
  - calls as `<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>{args}<|call|>`;
  - results as `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>{content|tojson}<|end|>`.
- History follows the template, not the issue's sketch (it differs in three places):
  - the recipient sits in the role section (`assistant to=functions.X<|channel|>commentary json`), not in the
    channel header;
  - a tool result's content is `tojson`'d (a JSON string literal);
  - a call message's analysis (`reasoning_content`, else its content) is kept only while no later
    assistant message is a final answer; other reasoning is dropped;
  - also, only a message's first call is rendered, a result is named after the last call, and a system
    message other than the first is not rendered.
- `src/chat.rs`: `Channels::with_tools`. It reads the role section too, and a message addressed `to=X` (in
  either header) is a call: its text is `Hidden` and `Channels::call()` names the recipient. Without tools,
  behaviour is unchanged (`axon chat` never sets it).
- `src/serve.rs`: with tools, `TextOut::harmony` writes the call into the content in its history form (stop
  strings don't apply to it). `tools::Parser` (`Kind::Harmony`, Hermes' block machinery with these markers)
  turns it into a `Call`. Every endpoint already wired for tools gets it: chat completions (stream and whole),
  Responses and ollama `/api/chat`. `--think-inline` keeps the raw stream and doesn't parse calls.
- `tool_choice`:
  - `none`: no tools rendered, plain chat;
  - `auto`: normal;
  - `required`: the prompt ends `<|start|>assistant to=functions.`;
  - named: `... to=functions.NAME<|channel|>commentary json<|message|>`. Both skip the analysis, like Qwen3's
    forced calls. `Channels` is primed with the prompt's last message head, so a prefilled call is
    announced.
- Malformed arguments (not JSON), or a block with no name, come back as content, verbatim, the same policy as
  the other families. Arguments are always valid JSON.
- `scripts/harmony_tool_ref.py` regenerates `tests/fixtures/tool_templates_harmony.json`: 8 cases, each a
  request and the prompt jinja2 renders from the GGUF's own template, with transformers' setup and the
  date fixed to `HARMONY_DATE`.

## Why
gpt-oss-20b is the most capable model bella runs (~45 tok/s decode, experts on the CPU). Without tools, no
agent harness can drive it. The done condition: `synaps-axon oss` (profile axon-oss, port 18442, `--kv f16`)
runs a Synaps session with working tool calls.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| unit: tools + harmony (render fixtures byte-exact, TS types, parser every split, channels to calls) | 16 passed | 1020-030043-0780 (bella, CPU) |
| real model: gpt-oss-20b serve tool loop (CUDA) | passed, 14.5 s | 1020-030102-6c02 (bella) |
| fast tier, release (big-model skips as the fast tier), GPU | 619 passed, 0 failed, 47 ignored | 1020-044910-f145 (bella) |
| fast tier, debug assertions, CPU, 3 shards | 637 passed, 0 failed, 47 ignored | group 1020-043143-a805 (avante) |
| `serve::` suite, debug assertions, GPU | 116 passed, 1 failed (`serve_metrics_slots_max_prompt`, a pre-existing race: #112) | 1020-030717-aa65 (bella) |
| that test alone x10, candidate and main debug, interleaved | 10/10 each | 1020-045611-6e9f |
| `serve::` suite rerun, same binary | 117 passed, 0 failed, 2 ignored | 1020-045611-3396 |
| gpt-oss model tests (`oss gptoss harmony chat::`), release, GPU | 43 passed, 0 failed, 2 ignored | 1020-030717-b206 |
| md5 of `gen`, main cb53181 vs 231d34e, same host (8 models cuda f16 x 4 prompts, 4 cuda q8 x 3, 4 cpu x 3, one ~600-token prompt) | 56/56 equal | 1020-030718-7eaf |
| serve A/B, main vs candidate, raw bodies (ids/times masked): 17 requests on Qwen3-8B, Llama 3.1 8B, qwen2.5-0.5b, Qwen3-0.6B, Llama 3.2 1B (with and without tools, every tool_choice, streamed, Responses, ollama); 7 without tools on gpt-oss | 17/17 x5, 7/7 | 1020-030718-7eaf |
| merge of main 6f1efb5 (00b8a71): `axon` binary byte-identical to 231d34e; quick md5 (5 models cuda, q8, cpu) | identical binary; 29/29 equal | 1020-054115-f4d5 |
| e2e: Synaps headless (`synaps chat --profile axon-oss`), gpt-oss-20b `--kv f16 --ctx 32768` on :18442 | ran `uname -sr && nproc && date +%Y` via its bash tool and answered from the output, correctly | 1020-030754-405a |
| clippy -D warnings (bin + tests), rustfmt | clean | |

## Results
Functional change, no speed claim. Requests without tools keep their prompt, tokens and output (serve A/B
above). Tool requests on the Qwen/Llama families are byte-identical to main. MEASURED on bella, e2e
(Synaps, gpt-oss-20b, cuda + CPU experts): turn 1 prompt 1724 tok in 21.9 s, 55 tok at 30.7 tok/s; turn 2
1200 of 1265 prompt tokens from the cache, 81 tok at 30.7 tok/s. Turn 2 is shorter than turn 1 because of what
Synaps sends, not axon. Its request log (1020-094444-6662, a logging proxy in front of the server): turn 1 sent
system 751 chars, user 1761 chars, 11 tools; turn 2 sent system 751, user **186**, the assistant call, the tool result
(29 chars), and the same 11 tools. Synaps shrinks the earlier user turn (likely by stripping the ambient context it
injected), so turn 2 is a different, shorter prompt that shares the first ~1200 tokens.

## What didn't work / not done
- Built-in tools (`browser`, `python`) are not offered: the template's `builtin_tools` kwarg has no OpenAI
  field. A call to such a recipient still parses, named after the full recipient.
- Parallel calls: `<|call|>` ends the reply (EOG), so a reply makes at most one call. A history with several
  calls in one message renders only the first, as the template does.
- Malformed arguments return as content (the existing policy). They are not handed back as a raw-string
  `arguments`.

## Review round 1 (shady, 7/10, on c39f7a4) → 6bcb9a3
- **MUST 1, forgery.** The parser matched the call's text, so a final answer that spelled the call syntax with
  ordinary tokens (a quoted web page, "repeat this") came out as a `tool_calls` entry. Calls now come only from
  the special tokens: `TextOut` hands the parser one string with the text escaped (`\u{1}` → `\u{1}0`) and each
  call framed out of band (`\u{1}1 recipient \u{1}2 args \u{1}3`), so nothing the model writes can frame one.
  A `<|call|>` inside the arguments' text no longer ends a call. Tests: `harmony_parser` (forged text,
  the escape char, every split), `harmony_calls_from_channels` (forged final text as plain tokens through the real
  tokenizer and splitter, every split), `serve_gptoss_tool_calls` (gpt-oss-20b asked to repeat the call syntax:
  no `bash` call on chat completions, Responses or `/api/chat`, streamed or whole).
- **MUST 2, required.** The prompt ended `to=functions.`, but o200k splits `to=functions.get_weather` as ` to`,
  `=functions`, `.get`, `_weather`, so a bare `.` token is off-distribution. It now ends `to=functions` and the
  model writes `.NAME`. `render_harmony` checks the split with axon's tokenizer, and the real model must
  call an offered tool for two `required` prompts.
- SHOULDs:
  - only an `assistant` message to `functions.NAME` is a call: a hallucinated `functions.x to=assistant` result,
    `python` and `browser.*` stay reasoning;
  - a named choice renders every tool, so the developer message, and the cache with it, matches auto's; the
    prefill forces the call;
  - a reply with no channel header is the answer, as without tools;
  - a malformed call returns the model's arguments' text, never axon's framing;
  - real-model `tools: []` with tool history;
  - the e2e captures what Synaps sends (a logging proxy in front of the server).
- Gates at 6bcb9a3: oss 1020-063226-3966, serve:: dbg 1020-063226-b46c, fast tier rel 1020-063227-51d9 (includes
  `serve::` release), fast tier dbg 1020-063228-e29d/e306/74b2, md5 + serve A/B 1020-063228-d66d, e2e 1020-063229-6424.

## Review round 2 (shady, 8/10 APPROVE, on 6bcb9a3) → d916bf8
- SHOULD 1, truncation. `<|call|>` is no longer a stop token for harmony tool requests (the splitter must
  see it). A call is a call only when it is closed by its marker (`<|call|>`, or `<|end|>`). A call cut off by
  the token limit is never one: its arguments' text is content, blank arguments are nothing, never `{}`.
  TextOut writes a separate cut frame for it.
- SHOULD 2, arguments must be a JSON object. `42`, `[1]`, strings and null are content.
- SHOULD 3, one call per reply. The first closed call ends generation: TextOut reports it like a stop string,
  so the engine stops exactly there on both the solo and the batched path. Why stop rather than demote later
  calls to content: a model that keeps going after `<|end|>` usually hallucinates the tool's result
  (`functions.x to=assistant ...`) and then answers from it. That text is wrong as content too, and it costs
  tokens. Stopping also matches the history renderer (one call per assistant message), so the next turn's
  prompt is the template's.
- SHOULD 4 (edge cases of a reply with no header) is #114.
- Gates (delta; 6bcb9a3's full set still covers everything else): unit (CPU) 1020-065333-b744; unit + gpt-oss
  model tests + `serve_gptoss_tool_calls`, `serve::` debug, md5 + serve A/B in one bella job, 1020-065334-6933.

## Round 2 regression: `required` on gpt-oss-20b → fixed back to `to=functions.`
Round 1's MUST 2 change (end the `required` prefill at `to=functions`, o200k's pretoken boundary) failed on the
real model. With "Hi there" and `tool_choice: required`, gpt-oss-20b took `functions` as the whole recipient, went
on to `<|channel|>analysis`, and greeted: `finish_reason: stop`, no call. MEASURED, `serve_gptoss_tool_calls` at
6bcb9a3, release 1020-063226-3966 and debug 1020-063226-b46c, the same reply both times (prompt 155 tok, 112 from
the cache). It is the model's choice, not the cache: the prompt is the rendered one, prefill included, and
`render_harmony` checks those tokens.
The prefill ends with the `.` again. The lone `.` token is off the trained split, but the model can only continue
with a name. The real-model test now has four `required` prompts (a greeting, a time question, a joke, arithmetic),
each of which must call an offered tool. `render_harmony` checks that the prompt ends with the prefill tokens and that
the primed splitter makes a call from a name and finds none when a header follows directly.

## `required` with several tools: a 400 (#116); one tool is the named choice
The `.` didn't hold either. At 56fc0d1 (job 1020-081323-5660, MEASURED), gpt-oss-20b got "Hi there" with
`get_weather` + `get_time` and `required`. It left `to=functions.` with `<|channel|>analysis`, decided "No function
needed", and answered "Hello!" (finish `stop`). The rest of that job passed: md5 identical, serve A/B 17/17 ×5 and 7/7,
`serve::` debug 119/1 (only this case failed). A text prefill can't force a choice among several tools: the model can
always leave it with a special token.
Forcing it needs constrained decoding: mask the logits after `to=functions.` to the offered names. Greedy picks are
made inside the backends' step (`lm::RowOut::Pick`, a device argmax on the batched CUDA path) and in both generate
loops, which is the shared hot path, so it is out of this PR's scope. Filed as #116. Until then:
- `required` with one tool is that tool's named choice (its prefill: the arguments follow);
- `required` with several tools on gpt-oss is a 400 (`tool_choice`), whose message points to naming the tool or `auto`.
Also: a header token (`<|channel|>` / `<|message|>`) inside a call's arguments now cuts the call (not a call).

## Forced calls: the model's call form, and a call is made when its object is whole
b104d07's `required` (one tool) case failed on "Tell me a joke." (job 1020-084724-fd79); the rest passed: md5
identical, serve A/B 17/17 ×5 and 7/7, `serve::` debug 119/1, and the Synaps e2e at 56fc0d1 (1020-081324-9a19).
The raw tokens (`--think-inline`, streamed one token per chunk; job 1020-093424-74a0) after the named prefill
`<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>`:
```text
{" name ":" get _weather "," params ":{" city ":" ?? "} } We need to respond to user: "Tell me a joke." We don't need
weather. Just a joke. <|end|> <|start|> assistant <|channel|> final <|message|> Why don’t scientists trust atoms? ...
```
That is a wrapped object, `{"name": ..., "params": {...}}`, then reasoning text **with no header**, then `<|end|>`.
`<|end|>` closed the "call", whose arguments weren't JSON, and they came back as content: the reasoning leaked.
In auto mode the model writes its calls as `<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}`.
Prefill forms side by side (job 1020-093636-b9e8, 5 prompts, MEASURED):
| prefill | arguments the model writes |
|---|---|
| A: `assistant to=functions.X<|channel|>commentary json<|message|>` (the template's history form) | `{"name":"get_weather","params"/"arguments"/"json":...}` every time, wrapped; on the haiku prompt, `??…` to the token limit |
| B: `assistant<|channel|>commentary to=functions.X <|constrain|>json<|message|>` (the model's call form) | `{"city":"London"}` / `{"city":"Paris"}`, the tool's own schema, every time |
| C: `assistant to=functions.X<|channel|>commentary <|constrain|>json<|message|>` | wrapped, like A |

Fixes:
- The named prefill (so `required` with one tool too) is B.
- `TextOut` holds a call's arguments and makes the call as soon as they are a whole JSON object. Nothing valid follows
  an object's `}`, so a well-formed call never needs its `<|call|>`, and the reply ends there (one call per reply).
  On B the model's `{"city":"London"}?<|end|>...` makes the call at the `}`.
- A call that ends any other way (its marker, a header, the token limit) with arguments that aren't a whole object is
  no call. Its text is **reasoning** (it was on the commentary channel, which is reasoning without tools), never
  content, and the reply goes on unless the token was `<|call|>`. What follows goes where its channel says.
- Tests through the real tokenizer and splitter, at every split point:
  - complete object then a header, text or a second call: one call;
  - incomplete arguments then `<|end|>`, a header, `<|call|>` or the limit: no call, the text is reasoning;
  - analysis after a dropped call is reasoning;
  - the joke reply replayed token by token makes one call.

## Review round 3 (shady, 8/10, on 7acbd59) → this commit
- Docs: README says bad gpt-oss calls go to `reasoning_content`, a call ends at the whole object, and the
  prefill strings. The PR body's What changed / Design notes are rewritten for the current code.
- `ArgScan`: the arguments' JSON state (string, escape, depth) is tracked as tokens arrive, O(1) a byte. The text
  is parsed once, when the top-level object closes, instead of after every `}`, which was quadratic on a big
  file-writing call, inside the engine loop.
- Tests:
  - `arg_scan`: `}`/`]` in strings, `\"` and `\\`, nesting, unicode, at every two-point split;
  - the same arguments through `TextOut`, tokenized normally and one token per character: the call is made at
    the closing token, never earlier, and one token short is no call.
- `axon_tool_calls_dropped_total` in /metrics, and a line on stderr with AXON_SERVE_DEBUG=1, for each dropped call.
- Gates at 7acbd59: delta 1020-094441-ff27 (gpt-oss 38/0, `serve::` debug 120/0/2, md5 0 of 56 differ, serve A/B
  17/17 ×5 and 7/7) and Synaps e2e 1020-094444-6662 (bash called, answer from its output), both pass.
