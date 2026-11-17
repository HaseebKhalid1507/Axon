# serve /v1/responses: codex's goal item is a tool message, not a 400  (PR #278)

- Date: 2026-11-17. Branch: fix/responses-no-call-id. Base: main 1a1d751. Status: open
- Pick / design: issue #273 (eval track, from #235's acceptance runs); the foreman's 11-17 ruling: take the issue's first
  suggestion, bug fix only.

## What changed
`src/serve/responses.rs`, `items()`: a `function_call_output` item without a `call_id` but with a non-empty `name`
string becomes a tool message with no `tool_call_id`: `{"role": "tool", "content": output}`. A `call_id` that is null
counts as absent.
- **No `tool_call_id`, not even the item's `id`.** At 7a6c175 the message took the item's `id` as its `tool_call_id`.
  That followed the issue's "a template that requires ids can get the item's id", as the foreman relayed it. shady's
  review (NIT) showed it was dead data, and misleading:
  - `tool_call_id` names the call a tool message answers, and the goal answers none.
  - No template in Axon reads the field (`tools::TMsg` has no id), and only the stored conversation kept it.
  - A template that does render call ids (Mistral's `[TOOL_RESULTS]`, for one) would emit a dangling id.

  43efbb3 drops it.
- With neither a `call_id` nor a `name`, the item is still a 400 (param `input`). The message now names both fields:
  "a function_call_output item needs a 'call_id' string (or a 'name', for an output that answers no call)".
- A `call_id` that is present but not a string, or an empty or non-string `name`, is still a 400.
- An item with a `call_id` string takes the same branch as before and gives the same message: role, `tool_call_id` =
  `call_id`, content, in that order. Its `name`, `id` and anything else are still not read.
- Docs: the module doc and `docs/serve-apis.md`'s `input` row.

## Why
codex 0.169 appends an item of its own to `input` after the model calls its `create_goal` tool:

```json
{"type":"function_call_output","id":"fco_01a1dd99-167a-7410-8a4c-4f7dea6c71f1","name":"goal_objective",
 "namespace":"agent","output":"Write unit tests for ledger/validate.py ...",
 "internal_chat_message_metadata_passthrough":{"turn_id":"01a1dd99-06d3-7f00-a2bf-319820b5dbd1","create_time":1794719159.9305158}}
```

It has no `call_id`. Axon answered with a 400, and codex ended the task with it. #273 has the counts: 1 to 4 of 14 tasks
per arm in jobs 1113-010041-0dce and 1114-192659-82f7. llama-server 4413a19 refuses the item as well
(`Cannot determine type of 'item'`).

Where the item sits: all 6 codex rollouts from #235's runs that carry it (Qwen2.5-7B, the llama and axon-constrain
arms) put it right after a `function_call_output`. One of them has it three times in a row, which is codex retrying
into the 400. Every one has the same six keys. So in every template the goal follows a tool call. For gpt-oss that
matters: Harmony names a tool message after the last assistant call (`functions.<name>`, as llama.cpp's gpt-oss
template does with `last_tool_call`), and refuses one with no call before it. The goal renders as
`functions.<the last call>`, which is the same rule chat completions follows.

## How it was verified
Code at 43efbb3 (the review round). 7a6c175's suites, e2e and mutants were killed before they finished: the round
changed `responses.rs`, and the e2e script had the bug the review found (below).

| gate | result | job ID / log |
|---|---|---|
| build: release bin, release and debug-assertion test bins (rustc, lib layout) at 43efbb3 | OK | bella `~/.cache/axon-eval/logs/buildall-43efbb3.log` |
| clippy `-D warnings` (lib, tests, bin), rustfmt | clean | same log |
| targeted, debug-assertion build at c559c64: `serve::responses::`, `serve::tools::`, `serve::template_tests::`, `serve::api::` (jade, CPU) | 44 passed | 1117-093156-5057 |
| release suite, 3 shards (jade, CPU: GPU tests skip themselves) | PENDING | 1117-100032-edd3, -14b5, -1565 (group 9c4a) |
| debug-assertion suite, 3 shards (jade, CPU) | PENDING | 1117-100307-343e, -26e6, -d5bb (group 1ec3) |
| codex over the wire: codex 0.169.0 against a stub, the goal item as sent (bella, CPU, no model) | PENDING | 1117-100614-67b4 |
| `/v1/responses` e2e differential, main 1a1d751 vs 43efbb3, with codex's wire request (jade, CPU, 3 models) | PENDING | submitted when the wire check is done (gatewatch `eval-43efbb3-e2e`) |
| mutants M1–M5, the candidate as control (`mutants.sh`, jade, CPU) | PENDING | 1117-101610-fcdd |
| md5 of `gen` | not run: `gen` never reaches `responses.rs` | |

**Bit identity for requests main accepts.** These are exactly the requests where every `function_call_output` has a
`call_id` string. They take the first match arm, `Some(J::Str(id))`, which is main's `.and_then(J::str)`. They build
the same three keys in the same order, and nothing else in `items()` or `parse()` changed. Tests:
- `outputs_with_a_call_id_are_unchanged` pins that message. It runs six variants of the item: bare, with an `id`, with
  a `name`, with an empty name and a numeric id, with a numeric name and a null id, and with all of codex's fields.
  Each is tried alone and after a user message.
- `translates_to_chat` and `previous_response_id_continues` still pass unchanged.
- The serve md5 harnesses on bella (`serve_md5.py` from #166 and its `-share` versions) only drive `/v1/completions`
  and `/v1/chat/completions`, so they never reach this code. The e2e differential (`resp_e2e.py`, below) is the serve
  md5 for this path.

**The goal item.** `codex_goal_item_is_a_tool_message` takes the item byte for byte from the rollout of job
1114-192659-82f7 (axon-constrain arm, write-tests). It sends a codex-shaped request: a user message, `create_goal`'s
call, its output, then the goal item.
- The request parses to the same `ToolReq` as the chat request with a plain tool message (no `tool_call_id`) in that
  place. `tools::render` reads only the `ToolReq`, so every template renders the goal as that tool message, bit for bit.
- The stored history holds exactly `{"role": "tool", "content": goal}`. That holds for the item as sent, without its
  `id`, and with `"call_id": null` (the review round).
- Without its `name`, the item is a 400 with param `input`, and the message names both fields.
- `bad_shapes_are_400` gets five more shapes: codex's item without its name, a null `call_id` with a null `name`, an
  empty `name`, a numeric `name`, and a numeric `call_id` with a name.

**Codex over the wire** (`codex_wire.py` / `codex_wire.sh`; the review's SHOULD 3). The unit test's item comes from
codex's rollout, which is codex's own serialization, not the request body. So `codex exec` (0.169.0, bella's, the
version of the runs) runs a write-tests prompt against a stub Responses server. The stub answers the first request
that offers `create_goal` with a call to it, and every later request with "Done.". The check passes if a later
request carries a `function_call_output` with a non-empty `name` and no `call_id`. That request is kept as it went
over the wire and replayed in the e2e (W1). No model, no GPU.

**e2e differential** (`resp_e2e.py` / `resp_e2e.sh`, in the eval scratch dir; sha256 in the PR body). It starts
`axon serve` from main 1a1d751 and from the candidate on CPU, one after the other, on Qwen2.5-0.5B (Qwen2.5 Hermes
tools; context 32768, for W1), Qwen3-0.6B (Qwen3) and Llama-3.2-1B (Llama 3.1 tools). Each gets the same greedy
requests, 40 tokens:
- **A, requests main accepts.** Tools with no history; a call and its output; that output with codex's extra fields;
  two calls, one output given as text parts; the call and output streamed; instructions with no tools; a stored
  conversation continued by `previous_response_id`. Each must be 200 on both, and the candidate must equal main byte
  for byte, ids and timestamps dropped, every SSE event of a stream included.
- **A7, an output with neither field.** 400 on both, with the same type, param and code. The message is this PR's to
  change, so the candidate's must name `'name'`.
- **B, the goal.** The goal item with its `id`, without it, and streamed. Main must answer 400 and the candidate 200.
  The candidate's answer must equal its answer to the chat-completions twin: same prompt tokens, reasoning, text and
  calls.
- **B4.** The goal stored (`store: true`), then continued by `previous_response_id`: 200 twice on the candidate.
- **C.** The goal item without its `name` is a 400 on both.
- **W1, codex's wire request** (from the wire check), on Qwen2.5-0.5B only, with only `stream` turned off and
  `max_output_tokens` set to 16. Main must answer 400, naming `call_id`, and the candidate 200.

**Mutants** (sources at bella `~/.cache/axon-eval/mut/M*.rs`, each a copy of 43efbb3's `responses.rs` with one hunk
changed; release test builds, `serve::responses::`):

| mutant | the one change | must fail |
|---|---|---|
| M1-main | main's arm back (`call_id` required) | `codex_goal_item_is_a_tool_message` |
| M2-no-name-needed | `(None \| Some(J::Null), _) => None` | `bad_shapes_are_400`, `codex_goal_item_is_a_tool_message` |
| M3-any-call-id | `(_, Some(J::Str(n))) if !n.is_empty() => None` | `bad_shapes_are_400` |
| M4-name-first | that arm before the `call_id` arm | `outputs_with_a_call_id_are_unchanged` |
| M5-no-null-call-id | `(None, Some(J::Str(n))) ...` (no `\| Some(J::Null)`) | `codex_goal_item_is_a_tool_message` |

M5 replaces 7a6c175's M5, which undid the item-`id` fallback the review round removed.

## Review
shady 8/10 SHIP-AFTER-FIXES on 16a5d7a ([comment 6951583424](https://github.com/HaseebKhalid1507/Axon/pull/278#issuecomment-6951583424)):
- **MUST, fixed in the script.** The e2e's A7 compared the whole 400 body, message included, and this PR changes that
  message. 1117-093917-0144 would have failed on every model. It was killed while still queued.
- **SHOULD 1, fixed.** A1–A6 must be 200, not just equal.
- **SHOULD 2, fixed.** A null `call_id` with a name is now tested; it kills the mutant without `| Some(J::Null)` (M5).
- **SHOULD 3, done.** Codex over the wire, above.
- **NITs.** No `tool_call_id` (above); the stored-goal continuation (B4); the mutants published (above).

## Results
PENDING: the gates above (gatewatch `eval-43efbb3-gates`). Not a
performance change: no pp/tg numbers apply.

## What didn't work / not done
- Harmony (gpt-oss) still refuses a tool message with no assistant call before it, from either API. llama.cpp's
  template does the same. codex's goal always follows a call (above), so this doesn't come up. If it ever did, the
  item's `name` could stand in for the call's name in Harmony, but that is a template change and outside this fix.
- llama-server keeps refusing the item. Under codex, Axon now gets further than llama.cpp does.
