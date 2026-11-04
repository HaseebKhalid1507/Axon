# Agent harness traces + the N_warm ladder + #182 (PR A2)

- Date: 2026-11-04. Branch: feat/agent-harness-traces. Base: main af281d9 (merged in, `a3da697`). Status: open.
- Pick / design: the agent-workload plan (Fable, 11-04), WP1's second half (A2) and §4's N_warm ladder; issue #182
  (shady's round-3 review of #160). The capture half is joestar's, from 10-27 (`62fa040`, `b434675`), written to his
  spec "agent-harness traces beyond Synaps" (Checkpoint 0). The spec is local, not in the repo, so this log stands in
  for it.
- No engine changes. Scripts and this log only.

## What changed
**Capture (joestar, 10-27)**
- `scripts/agent_proxy.py` is a recording proxy in front of `axon serve`. Per request it appends one JSON line of
  numbers: route, token counts, timings, tool-call counts and per-capture HMAC digests of the message prefixes. No
  text is kept.
- `scripts/agent_trace_proxy.py` turns a capture into a #160 board trace. It recovers sessions from digest prefixes
  (main, side, compact and subagent requests), runs `agent_trace.py --check` and deletes the raw capture. It refuses
  git trees.
- `scripts/agent_capture.sh HARNESS [ROUNDS]` is one job: axon serve (7B Q4_K_M, ctx 32k, q8, parallel 4) behind the
  proxy, plus one harness driving tasks in a synthetic repo. Each harness gets a scratch HOME / XDG dirs, its own
  config and a dummy key. The real configs (`~/.codex` etc.) are never touched.
  - Tasks: bugfix, feature, resume, explain, compact, and subagent where the harness has subagents.
  - Harnesses: pi, opencode, codex, crush.
- `scripts/examples/harness-repo/make_repo.py` builds the synthetic "ledger" project, with one planted bug.
- `scripts/examples/harness-selftest/` is a self-test with a fake upstream; no model and no harness.

**This PR (11-04)**
- **`agent_replay.py --sessions N`**: the N_warm ladder (run, board: `--sessions 8,16,24,32`; `agent_board.sh`:
  `AXON_AGENT_SESSIONS`).
  - **The session order:** primed sessions (started before the window) and new ones, interleaved in proportion.
    Each class is ordered by its first request in the window.
  - **N ≤ S** keeps the first N sessions and only their primes.
  - **N > S** adds copies in the same order. Copy j of a session keeps the original's arrivals, gaps, lengths,
    compactions and max_tokens, but has its own fresh filler. It shares only what two real sessions share: the
    system/tool pool, and, for a forked session, the prefix of copy j of its parent. Copy 0 is the session itself.
  - **N = S** is therefore the canonical cell, byte for byte. Checked on the board trace: 174 requests and every
    prompt equal.
  - **`table`** gives each ladder cell its own block and prints N_warm per speed / KV / engine. N_warm is the
    largest N whose median TTFT-turn p50 is ≤ 0.5 s and whose median re-prefill share (clamped) is ≤ 10 %, the plan's
    definition. ollama has no re-prefill number, so it gets "none".
- **#182, all four items:**
  1. **`ollama_check` allows only the layer types it understands:** model, template, params, license and system.
     Any other layer (an adapter, a projector) fails the board. `meta.json` records every layer (type, digest, size)
     and the params blob's contents, which are the sampling settings ollama adds to a request.
  2. **No blob is trusted by name.** Every blob the manifest names (config and layers) must have the manifest's
     size, which costs nothing to check, and must hash to its own digest. MEASURED on bella: 2.6 s for axon-q7
     (with the 7B's 4.7 GB weights), 0.5 s for axon-q15, 0.2 s for axon-oracle.
  3. **`meta.json` is written at the start** (`state: checking`), then after the checks (`running`), after each
     engine start and run, and at the end (`ended`). Writes are atomic: a temp file, then a rename. A capped or
     killed board keeps its hashes, config and the list of finished runs. The kill smoke now asserts this for each
     case.
  4. **Harness provenance:** `meta.json` and each `run` result carry `harness`: the commit of the tree
     `agent_replay.py` runs from, the tracked files that differ from it, and the script's own sha256, which also
     identifies a copy outside git. The board prints it at start.
- **Tests:**
  - **`scripts/agent_replay_selftest.py`** is offline, stdlib only and takes about 0.05 s. It covers the ladder
    (order, subsets, copies, pool and fork sharing, own text, N = S), every `ollama_check` refusal, the `table` and
    N_warm reading, and provenance. CI runs it next to `--check-committed`; it also passes on Python 3.12, CI's
    ubuntu-24.04 version.
  - **The smoke** (`agent_replay_smoke.sh`) runs the self-test first. It now also runs an Axon ladder of S/2 and
    2S sessions on the same short cell and checks each N's session count, prompt counts and errors.
- **Capture dry run (`scripts/examples/harness-selftest/capture_dry.sh`):** the self-test, then each installed
  harness's real binary against `fake_upstream.py --text-only`, through `agent_capture.sh`'s smoke mode. It uses no
  model and no GPU.
  - **Pass condition:** the harness finishes its task (rc 0), and its requests reach the proxy and convert. The
    converter's privacy check runs inside. A harness that isn't installed is reported SKIPPED.
  - **Smoke mode now converts** the capture as well; before, it stopped once the requests reached the proxy.
  - **The fake's `/v1/responses` was too thin for codex 0.166.0** (bella upgraded it from 0.163.0 on 10-31). Codex
    rejected a text delta without its output item ("OutputTextDelta without active item") and a completed response
    without `usage.total_tokens`, then retried 5 times and failed. The fake now sends the full event sequence, as
    Axon's `src/serve/responses.rs` already does: item added, part added, delta, done events, then completed with
    `total_tokens`.
- **Docs:** the scripts' references to the uncommitted spec now point to this log.

## Why
- **The ladder is the headline "sessions sustained at SLO" number**, plan §4. Today's estimates, from the plan (NOT
  MEASURED): f16 ≈ 5, q8 ≈ 9. Both sit below agent-32k's S = 20 sessions, so the readings that matter now come from
  real sessions only. Synthetic copies only come into play from N = 21 on.
- **Why the order is stratified:** by first arrival alone, the first sessions in the window are the primed ones,
  because they were already running. On the board trace (agent-32k, scale 0.15, clip 32,768, speed 1), from
  `plan()`, no engine run:

| N | requests | primed / sessions | primed tokens | prompt tokens | prompt p50 |
|---|---|---|---|---|---|
| 8, by first arrival (dropped) | 93 | 6 / 8 | 118,469 | 1,881,711 | 19,164 |
| **8, stratified** | **65** | **3 / 8** | **49,545** | **1,206,316** | **18,036** |
| 16, stratified | 147 | 6 / 16 | 118,469 | 2,470,184 | 17,703 |
| 20 (= S, the canonical cell) | 174 | 7 / 20 | 133,347 | 2,921,348 | 17,205 |
| 24, stratified | 199 | 9 / 24 | 168,664 | 3,353,628 | 17,114 |
| 32, stratified | 272 | 11 / 32 | 209,995 | 4,675,814 | 17,205 |

  First-arrival order would have put 89 % of the cell's primed tokens into N = 8. That is exactly the range where
  the plan expects N_warm to fall, so the ladder would have read pessimistic there.

## The 10-27 captures (MEASURED, joestar; not rerun here)
Job per harness on bella, harness `b434675`, axon `6f06544` (main then), Qwen2.5-7B-Instruct Q4_K_M, ctx 32k, q8,
parallel 4, 3 rounds. Harness versions then: pi 0.85.1, opencode 2.1.1, codex 0.163.0, crush 0.98.1. Gatewatch
`harness-captures-b434675`: ALL PASS.

| harness | job | wall | requests | sessions (subagent) | prefix reuse | mean prompt | tool calls (as text) | tests OK |
|---|---|---|---|---|---|---|---|---|
| pi | 1027-164944-15ed | 820 s | 103 | 38 (25) | 43.6 % | 6,955 | 108 (6) | 3 / 9 |
| opencode | 1027-164944-a63e | 903 s | 246 | 40 (24) | 95.3 % | 10,072 | 224 (13) | 1 / 9 |
| codex | 1027-164945-a435 | 214 s | 59 | 17 (5) | 89.4 % | 7,078 | 39 (5) | 0 / 9 |
| crush | 1027-164946-6b56 | 1,507 s | 474 | 291 (267) | 87.1 % | 8,467 | 367 (2) | 3 / 9 |

- **Prefix reuse** is cached / prompt tokens as the proxy saw them. "tests OK" counts bugfix, feature and resume
  over 3 rounds. The 7B mostly can't finish these tasks.
- **The captures are traffic shapes, not a harness ranking.** One task timed out (rc 124) for pi and one for crush.
- **pi's low reuse is the resume task:** 562,900 prompt tokens at 35.3 %, with 7 compactions.
- The traces are in `~/.cache/axon-agent/traces/harness-*.json` (numbers only, never in the repo).

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| `agent_replay_selftest.py` (offline) | local PASS (Python 3.14 and 3.12); in CI and inside the smoke | CI, smoke job |
| regression: `table` on the 10-27 board dirs, and canonical `plan()` prompts, main vs this head | identical (local, no engine) | — |
| mutants: 9 deliberate bugs in the new code (ladder order / text / fork / primes, hashing, sizes, layers, N_warm, provenance) | 9 / 9 caught by the self-test (local) | — |
| `ollama_check` on bella's real store (mirror, 3 gate models) | all pass; timings above (local, file reads only) | — |
| smoke: extract → replay (axon + ollama 0.40.13, 1.5B) → table, plus the ladder (axon, S/2 and 2S) | see PR body | PR body |
| kill smoke (4 cases), plus `meta.json` kept at kill | see PR body | PR body |
| capture dry run: self-test plus pi 0.85.1, opencode 2.2.5, codex 0.166.0, crush 0.99.0 against the fake | local PASS (4 harnesses driven); gate job in the PR body | PR body |
| CI (`--check-committed`, fmt, clippy, tests) | see PR body | PR body |

## Risks and what wasn't measured
- **No ladder board has run.** N_warm is a definition plus a test on synthetic run results. ESTIMATED cost: at speed
  1, each N is the 600-s window per engine per round (Axon s1 was 640 s MEASURED on 10-27). 4 N × 3 engines × 3
  rounds is ≥ 6.5 h, and longer for an engine that overloads at N = 32. Split it by engine or round.
- **Copies are phase-locked:** copy j arrives at the same instants as its original. Above S this makes bursts more
  correlated than independent users would, so N_warm above 20 is likely pessimistic. Below S there are no copies. If
  a reading lands near a boundary above 20, a jittered variant (one seeded offset per copy) settles it.
- **The 10-27 captures are not rerun** against the 10-31 harness versions. The dry run proves the current binaries
  still drive the pipeline end to end. It does not check that their traffic shape is unchanged.
- **The full-blob hash adds ~3 s per board** on the 7B. That's noise next to a 220-min board.

## Next
- WP6: the ladder board (f16, q8; speed 1; N = 8, 16, 24, 32) with the desktop idle or labelled.
- Replay the 4 harness traces as board cells, so the cache numbers aren't only Synaps-shaped.
