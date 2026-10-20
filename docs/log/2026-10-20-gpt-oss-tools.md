# serve: tool calling for gpt-oss (harmony)  (PR #111)

- Date: 2026-10-20. Branch: feat/oss-tools. Base: main cb53181. Status: open
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
| fast tier, release | PENDING | |
| fast tier, debug assertions (CPU, 3 shards) | PENDING | |
| `serve::` suite, debug assertions, GPU | PENDING | |
| gpt-oss model tests (`oss gptoss harmony chat::`) | PENDING | |
| md5 of `gen`, main vs candidate, same host | PENDING | |
| serve A/B with tools, main vs candidate (Qwen3-8B, Llama 3.1 8B, + 3 small, gpt-oss without tools) | PENDING | |
| e2e: Synaps headless, profile axon-oss, gpt-oss-20b `--kv f16 --ctx 32768` | PENDING | |
| clippy -D warnings (bin + tests), rustfmt | clean | |

## Results
Functional change, no speed claim. Requests without tools keep their prompt, tokens and output (serve A/B
above). Tool requests on the Qwen/Llama families are byte-identical to main.

## What didn't work / not done
- Built-in tools (`browser`, `python`) are not offered: the template's `builtin_tools` kwarg has no OpenAI
  field. A call to such a recipient still parses, named after the full recipient.
- Parallel calls: `<|call|>` ends the reply (EOG), so a reply makes at most one call. A history with several
  calls in one message renders only the first, as the template does.
- Malformed arguments return as content (the existing policy). They are not handed back as a raw-string
  `arguments`.
