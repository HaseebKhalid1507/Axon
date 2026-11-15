# Coding-agent acceptance test: opencode, codex and pi on Axon and llama-server (PR #235)

- Date: 2026-11-13; measurement fixes 11-14 and 11-15; rerun on main 9f3630c (#262, #264, #228) 11-14/15.
  Branch: feat/eval. Status: open (PR #235).
- Script: `scripts/harness_accept.py` (stdlib only), run through `scripts/harness_accept.sh` with axon-podjob.

## Method

- **Tasks and checks:** 14 small coding tasks on a generated repo (`scripts/examples/harness-repo/make_repo.py`),
  each with an automatic check. `selfcheck` shows every check failing on the untouched repo and passing on a
  reference fix.
- **Harnesses:** opencode 2.2.7, codex 0.169.0 and pi 0.85.1 (every reported run; the job log's `ACCEPT` line
  records them), each under bubblewrap with a scratch HOME.
- **Servers:**
  - `axon serve --kv q8 --parallel 4 --ctx 32768`.
  - llama.cpp 4413a19 with `-ngl 999 -fa on --jinja -b 2048 -ub 2048 -kvu --cache-reuse 256 -ctk/-ctv q8_0 -np 4`,
    32k per sequence.
  - Both get the model card's sampler (Qwen2.5: temp 0.7, top-k 20, top-p 0.8, min-p 0, repeat penalty off). No
    harness sends temperature or top_p.
  - **Arms** (`--engines`, 11-14): `axon` is main's defaults (since the merge: tool calls streamed, #262; shared
    prefix KV pages on, #228); `axon-constrain` adds `AXON_CONSTRAIN=1` (#264) and fails to start unless its start
    line names the grammar; `axon-nostream` is #262's kill switch; `llama`. One server at a time, same model file.
- **Proxy:** each harness talks to the server through a recording proxy. Per request it records TTFT to the first
  token, the server's own prefill timings, tool calls parsed, calls that came back as text, and arguments that aren't
  valid JSON.

## Measurement fixes (11-14, commit 0d0b0bb)

Found while triaging the Qwen2.5-7B runs. None of them changes a pass count. They change which task a request is
counted under, and whether the engine keeps working after a harness has gone.

- **The proxy now passes a hang-up to the engine.** It used to sit blocked on the engine's stream and never notice
  that the harness had gone, so a reply the engine held back kept generating. In codex job 0dce, Axon's unclosed
  `<tool_call>` (#260) ran 40 s past codex's retry and slowed the retry to half speed.
  - Now a watcher per request closes the engine's connection on the harness's EOF, and both engines cancel. The
    record says so: `cut: client`, counted as `hup`.
  - A chunked read cut mid-stream raised `IncompleteRead`, which isn't an `OSError`, so the record was lost. It's
    caught now.
- **A request counts in its own task.** Records used to be taken by time, so a request still open when its task
  ended was counted in the next task's row. That is how codex's runaway in write-tests showed up under pyproject.
  - Now the proxy waits up to 20 s at a task's end for that task's own requests, then cuts what's left
    (`cut: drain`, counted as `cut_end`). Records are taken by their own task label.
- **Compactions are counted per run.** pi and codex compact their own history near the context's end: a summary
  request, then the conversation restarted as system + summary + recent messages. That leaves no cached prefix past
  the system prompt, on any engine.
  - Each task run counts the harness's compactions from its own files: pi's session `compaction` entries, codex's
    rollout `compacted` items, opencode's `compaction` rows.
  - A **cold prompt** is a request that fed at least 4000 uncached prompt tokens. A cold prompt is attributed to a
    compaction if it starts after the last tool-carrying request before the compaction's timestamp and no later
    than the first one after it. That covers the summary request and the restarted conversation.
  - SUMMARY shows `cmp`, `cold`, `c_cmp` (cold, attributed to a compaction) and `c_1st` (cold, a task's first
    request).
  - opencode compacts without cold prompts: its summary request extends the cached conversation, and it restarts
    with a short prompt that reuses the system prefix.
- **`rescore DIR`** applies the per-task attribution and the compaction counts to runs made before the fixes. The
  tables below are rescored. 4 codex rows and 6 pi rows moved one request each, always into the next task; no
  totals changed.
- **Self-test:**
  - `selfcheck` covers each fix: a hang-up while the engine holds a streamed reply and a whole one; a request that
    outlives its task; a request still open when its task is drained; compaction counts in each harness's file
    layout; cold attribution on pi's pattern.
  - A mutant without the watcher fails the hang-up checks, and one whose drain doesn't wait fails the outlives check.
  - Job 1114-063505-7685 (jade, CPU, sandboxed): SELFCHECK PASS.

## Malformed calls after #262 (11-14, commit 335ce25)

- **What changed in the engine.** Since #262 Axon streams a call once its head is out, so arguments that go bad
  come back as a call whose `arguments` don't parse (what llama-server and OpenAI do), not as text. And
  `axon_tool_calls_dropped_total` now also counts the Hermes blocks returned as text, which the proxy counts too.
- **The count.** `malf` per task is the proxy's (calls as text + calls whose arguments don't parse), or Axon's own
  `dropped + invalid` where that is higher. The two views overlap since #262, so they are never added. Runs from
  before the merge give the same counts as before.
- **The rate** for #259's 1 % bar comes from the proxy alone, the same on every engine: malformed replies / tool-call
  replies. A tool-call reply has a parsed call or a call left as text; a malformed one has a call left as text or a
  call whose arguments don't parse. It's the denominator the 3.1 % below used.
- **Self-test:** two interleaved streamed calls with one broken; a broken call on `/v1/responses`; the arms'
  environments (inherited `AXON_CONSTRAIN`/`AXON_TOOL_STREAM` dropped) and start-line check; the count and rate
  rules on rows shaped like each engine's.

## Qwen2.5-7B before the merge (Axon `3014eeb`)

Jobs 1113-010041-0dce (codex), 1113-010041-c777 (pi), 1114-063612-b2e0 (opencode). bella's RTX 5070 Ti Laptop (12 GB).

| harness | engine | tasks passed | malformed calls | malformed replies / tool-call replies | timeouts | TTFT p50 (ms) | wall p50 (s) | compactions | cold prompts (attributed to a compaction) |
|---|---|---|---|---|---|---|---|---|---|
| codex | axon | 1 / 14 | 9 | 9 / 255 (3.5 %) | 2 | 71 | 8.2 | 4 | 6 (4) |
| codex | llama | 1 / 14 | 0 | 0 / 275 | 1 | 111 | 69.3 | 2 | 3 (2) |
| opencode | axon | 2 / 14 | 4 | 4 / 157 (2.5 %) | 0 | 86 | 20.6 | 3 | 3 (0) |
| opencode | llama | 3 / 14 | 0 | 0 / 170 | 0 | 135 | 58.4 | 6 | 6 (1) |
| pi | axon | 4 / 14 | 0 | 0 / 266 | 3 | 81 | 11.6 | 33 | 30 (30) |
| pi | llama | 2 / 14 | 1 | 1 / 213 (0.5 %) | 2 | 250 | 19.7 | 8 | 6 (6) |
| **all** | **axon** | **7 / 42** | **13** | **13 / 678 (1.9 %)** | | | | | |
| **all** | **llama** | **6 / 42** | **1** | **1 / 658 (0.15 %)** | | | | | |

No crashes or context overflows on either engine. One HTTP error the proxy lost (fixed at f9b5e85, below): llama-server
refused codex's goal item in pyproject (#273). With 14 tasks and one round, a pass-count gap under about ±4 means
nothing.

- **Codex, malformed 9 vs 0 (#259, #260):**
  - 8 are `<tool_call>` blocks whose JSON doesn't parse: unescaped quotes in shell commands (4), `\(` escapes (2),
    a raw newline (1), and a missing `</tool_call>` between two calls (1). Axon returns such a block as text, so
    codex treats it as the final answer and stops, after 1–6 requests (wall p50 8.2 s against llama's 69.3 s).
  - The 9th is a call that never closed: 26093 tokens over 340 s, held back by Axon, so codex timed out and retried
    (#260).
  - llama.cpp builds a lazy grammar from the tools, which forces valid JSON once `<tool_call>` appears.
  - Axon's model also called `apply_patch`, a tool codex didn't offer, 6 times.
- **pi, 30 vs 6 cold 16k-token prompts:**
  - Every one comes right after a pi compaction, on both engines. Before the first compaction in those three tasks,
    no Axon request after the first fed more than 971 uncached tokens (194 requests).
  - Axon had more compactions because the model looped to the 32k window in 3 tasks there, against 1 on llama.
  - Not an Axon bug, so not filed.
- **The roadmap's constrained-JSON check** ("defer if the harness test shows under 1 % malformed calls") is
  answered: **don't defer.**
  - Axon codex: 8 invalid-JSON calls in 255 tool-call replies, 3.1 %. Counting the runaway: 9 of 255, 3.5 %. With pi
    (0 of 266): 9 of 521, 1.7 %.
  - llama-server: 0 of 275 on codex.
  - The 6 calls to an unoffered tool aren't in these counts.
  - #259 is the fix.

## Qwen2.5-7B after the merge (main 9f3630c)

Jobs 1114-192659-82f7 (codex), 1114-192700-a2e2 (opencode), 1114-192701-202e (pi), script 335ce25, binary
`axon-a6f7f18` (the merge's `src/`, identical to main 9f3630c's). Three arms per job, one server at a time, same
model file and sampler as above. Rescored at f9b5e85 (no request moved).

| harness | arm | tasks passed | malformed calls | malformed replies / tool-call replies | timeouts | TTFT p50 (ms) | wall p50 (s) | compactions | cold prompts (attributed to a compaction) |
|---|---|---|---|---|---|---|---|---|---|
| codex | axon | 0 / 14 | 363 | 362 / 1152 (31.4 %) | 5 | 65 | 261.4 | 8 | 9 (8) |
| codex | axon-constrain | 1 / 14 | 2 | 2 / 347 (0.6 %) | 2 | 62 | 22.1 | 4 | 5 (4) |
| codex | llama | 1 / 14 | 0 | 0 / 103 | 0 | 84 | 9.1 | 1 | 2 (1) |
| opencode | axon | 3 / 14 | 5 | 4 / 181 (2.2 %) | 0 | 48 | 18.0 | 5 | 1 (0) |
| opencode | axon-constrain | 1 / 14 | 3 | 3 / 408 (0.7 %) | 2 | 45 | 54.3 | 20 | 5 (0) |
| opencode | llama | 1 / 14 | 2 | 2 / 236 (0.8 %) | 0 | 111 | 70.9 | 5 | 6 (0) |
| pi | axon | 3 / 14 | 16 | 16 / 321 (5.0 %) | 5 | 105 | 48.4 | 67 | 66 (65) |
| pi | axon-constrain | 4 / 14 | 10 | 10 / 178 (5.6 %) | 4 | 162 | 7.4 | 50 | 47 (46) |
| pi | llama | 2 / 14 | 2 | 2 / 237 (0.8 %) | 2 | 137 | 20.3 | 35 | 34 (33) |

What the malformed replies are, from the request records (`finish`, `cut`) and the harnesses' own session files:

| | axon | axon-constrain | llama |
|---|---:|---:|---:|
| tasks passed, all harnesses | 6 / 42 | 6 / 42 | 4 / 42 |
| malformed replies / tool-call replies | 382 / 1654 (23.1 %) | 15 / 933 (1.6 %) | 4 / 576 (0.7 %) |
| JSON syntax errors (invalid escapes, unescaped quotes, raw newlines) | 378 | **0** | 0 |
| cut short (`max_tokens`, incomplete, hang-up) | 4 | 13 | 2 |
| a call written as prose, no `<tool_call>` | 0 | 2 | 2 |
| tasks with at least one malformed reply | 15 / 42 | 7 / 42 | 3 / 42 |
| codex: calls to a tool it didn't offer (`apply_patch`, rollouts' `unsupported call`) | 23 | 29 | 0 |
| turn timeouts | 10 | 8 | 2 |

No crashes or context overflows. HTTP errors: codex's goal item (#273) was refused once on axon-constrain
(write-tests) and in 4 tasks on llama (from codex's output: the proxy lost those 4 records, fixed at f9b5e85).

- **Codex since #262 retries a broken call instead of stopping.** A call whose arguments don't parse now comes back
  as a call (before: as text, which codex took as the final answer). Codex answers `failed to parse function
  arguments: invalid escape …`, and the model sends the identical call again: 164 of cli-bug's 199 calls repeat the
  one before. cli-bug, stubs and pyproject took 122, 122 and 106 broken calls. That is the 31.4 %, the wall p50 of
  261 s (8.2 s before the merge) and the 5 timeouts.
- **The constrained arm has no syntax errors left.** Its 13 cut-short replies are runaways that the generic JSON
  grammar allows:
  - ~25k tokens of whitespace between tokens (codex stubs);
  - `"edits": [{…}, false, false, …]` to pi's `max_tokens` (pi stubs, 7 replies);
  - an endless `\n` run inside a string (codex spec-func, in a call to the unoffered `apply_patch`).

  llama.cpp's grammar bounds whitespace and follows each tool's schema (#272). The name enum is PR B's (#259).
- **Pass counts don't move** (6, 6, 4 of 42). Qwen2.5-7B finishes at most 4 of 14 tasks on any arm, which is a model
  limit. The constraint is about reliability and cost (retry loops, timeouts, GPU hours), not the pass rate.
- **Run-to-run spread is large.** llama-server was the same binary in both rounds, and still went from 3 to 1
  opencode passes and from 275 to 103 codex tool-call replies. One loop can dominate a per-reply rate (cli-bug's 122),
  so read the tasks with at least one malformed reply next to it.
- **pi compactions** (67 axon, 50 constrained, 35 llama): every cold prompt but one follows a compaction, as before
  the merge. The model looped to the 32k window more often on the Axon arms. Not an Axon bug.

**Decision for #259 (comment 6895714846): constrained JSON is needed now.**
- Unconstrained, every harness is over the roadmap's 1 % bar: codex 31.4 %, opencode 2.2 %, pi 5.0 %.
- The constrained arm removes every syntax error at no cost in passes.
- Recommended: default-on for the Hermes family, with `AXON_CONSTRAIN=0` as the kill switch; PR B's name enum;
  #272's whitespace bound and argument types.

## Measurement fixes (11-15, commits a1bf02f and f9b5e85)

- **llama-server's errors were lost.** Its error object carries a numeric `"code": 400`. The proxy built the error
  string as `code + ": " + message`, which raised, so the request was never recorded: 4 in 82f7, 1 in 0dce, all
  codex's goal item (#273). The code is `str()`ed now; selfcheck case `err_int`.
- **The SUMMARY's `(all)` rows** printed `dropped 0 invalid 0` beside 362 invalid calls. Axon's `/metrics` deltas are
  floats, and the sum took ints only. Each task row had them right.
- **New counts:**
  - `malf_cut`: malformed replies cut by `max_tokens`, an incomplete response or a hang-up, i.e. runaways rather
    than syntax errors. `rescore` computes it for old runs.
  - `tc_unoffered`: calls naming a tool the request didn't offer. It walks codex's namespace tools and is counted by
    the proxy alike on every engine. Runs before f9b5e85 show `?`; their codex counts above come from the rollouts.
- **Self-test:** job 1115-090855-d282 (jade, sandboxed), SELFCHECK PASS.

## Pending

- Qwen3-8B, `axon` and `llama` (no constrained arm: #264 holds Qwen2.5's Hermes calls only, and Qwen3's start
  after `<think>`), turn timeout 600 s for thinking, binary `axon-a6f7f18`, one job per harness:
  - codex 1115-090345-4036 (script a1bf02f: llama's 400s are lost there; codex's output has them);
  - opencode 1115-090828-1d9e and pi 1115-090829-adba (script f9b5e85).
- gpt-oss-20b on 7 tasks (experts on the CPU, slow), after Qwen3-8B.

## Raw

- Run dirs on bella: `~/.cache/axon-eval/accept/qwen2.5-7b-{codex-1114-041519,pi-1114-051800,opencode-1114-110413}`
  (before the merge) and `qwen2.5-7b-{codex-1114-225214,opencode-1115-002814,pi-1115-012835}` (after). Each holds
  results.jsonl, requests.jsonl, results-rescored.jsonl, rescore.txt, the engine logs and the harnesses' session
  files.
- The job logs are in `~/.cache/axon-run/logs/`.
