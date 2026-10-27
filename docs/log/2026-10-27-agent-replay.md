# Agent workload PR A: trace extractor, replay client, baseline board (2026-10-27)

Spec: [design/agent-workload.md](../design/agent-workload.md) §1 and §6, row A. No engine changes. PR #160.
**Status: board PENDING** (bella `--timing bella` jobs `1027-014629-7829` f16 and `1027-014630-9cb7` q8). This entry
gets the board table and the §6 B/C verdict when they finish.

## What landed
- **`scripts/agent_trace.py`**: Synaps logs → a numbers-only trace.
  - Sessions are recovered as Anthropic cache chains: cache_read(n+1) = read(n) + create(n). That holds for 97.1 % of
    22.8k requests, MEASURED over 8 log days.
  - Pools and forks come from where a new chain's cached prefix ends; subagents from dispatch lines.
  - `--check` (no string value > 16 bytes), `--anonymize-time` (no date or time of day; each day starts at t = 0),
    and `--synthetic`.
  - **The repo is public: no trace from real logs is ever committed, not even a numbers-only one.** The script
    refuses to write one inside a git tree. The repo carries only `scripts/examples/agent-trace-synthetic.json`.
- **`scripts/agent_replay.py`**: `run` (one cell against a URL), `board` (engines in interleaved rounds), `fanout`
  (row C's cell, idle and busy variants) and `table`.
  - Filler: synthetic single-token words from the target GGUF's vocabulary.
  - Requests: OpenAI `/v1/completions`, streamed.
  - The metrics of §1.3.

## Trace structure (MEASURED, Claude tokens, 8 days)
- **Prompts:** p50 / p90 / p99 165.9k / 412k / 812k; 97.6 % of prompt tokens cached.
- **Per-turn deltas:** `new` p50 1,110; output p50 353.
- **Gaps and sessions:** per-session gap p50 8.2 s; 662 chains, 279 of them subagent chains.
- **A new subagent's shared prefix:** p50 6,006 (p25 0, p75 6,363).

## Fan-out cell: row C re-registered 10-27, before any C data
- **What changed:** the spec's row C fan-out gate is re-registered in this PR (A owns the cell; no C data exists).
  The new text is in the spec's §6 row C.
- **Why: the leader's prefill is on every sibling's critical path.** The absolute form, "siblings 2–8 TTFT p50 ≤ 0.25×
  main", needs a shared span S ≳ 19k at a ~1k delta even for a perfect build. ESTIMATED with 2 fifo lanes: ratio ≈
  (T + 4t) / 5T (PR C design note, "spec corrections" 5).
- **Fix: the incremental form.** "(siblings 2–8 TTFT − the leader's TTFT) p50 ≤ 0.25× main's same quantity" keeps the
  bar's intent. It is ≈ (d + 32) / (S + d), ESTIMATED 0.16 at the cell's S and d. The spec's own short-span fallback
  is kept: if S < 4× d, the clause becomes "siblings 2–8 prefill no shared-page tokens".
- **Why a second variant: dedupe never fires from an idle server.** Sibling 1 runs the S1 path alone and is handed
  over only at decode (spec corrections 6).
  - **busy** is the same burst while 4 unrelated sessions (2,048-token prompts, 4,000-token replies) decode.
  - Both variants are gates.
- **S and d are unscaled:** S = 6,006 and d = 1,110, A's measured p50s, Claude tokens taken as target-model tokens.
  The cell isolates the burst, so agent-32k's s = 0.15 is not applied. Scaled, S would be ~900 tokens, below 4× d,
  and the clause would degenerate to the fallback.
- **Other clauses unchanged:** warm/GB ≥ 1.4× main on agent-32k at speed 4, md5 == main under `AXON_PAGE_SHUFFLE`,
  C=1 decode within 2 %.
- **Harness:** `agent_replay.py fanout --url … --model GGUF [--busy 4]`, 5 rounds of fresh text, medians. It runs when
  C benches, not in this PR's board.
