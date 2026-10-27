# Agent workload PR A: trace extractor, replay client, baseline board (2026-10-27)

Spec: [design/agent-workload.md](../design/agent-workload.md) §1 and §6, row A. No engine changes. PR #160.
**Status:**
- **f16 board: done with gaps** (job `1027-014629-7829`, 15 of 18 runs; table below).
- **Rule A verdict: B/C are NOT deprioritised.**
- **q8 board: pending** (job `1027-081924-51c9`).

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

## Baseline board, f16 (MEASURED, bella RTX 5070 Ti Laptop 12 GB, driver 615.78; job `1027-014629-7829`)

### Setup and gaps
- **Model and cell:** Qwen2.5-7B-Instruct Q4_K_M, agent-32k. 174 requests, 20 sessions, prompt p50 17.2k tokens. Speed
  1 is a 600 s window, speed 4 the same window in 150 s.
- **Rounds:** interleaved axon → llama-server → ollama.
- **KV budget:**
  - Axon d8a296c's paged pool is 102,976 tokens (5.9 GB).
  - llama-server 4413a19: `-c 98304 -np 3 -kvu --cache-reuse 256 -fa on`, `--cache-ram` 8192 MiB default,
    `--cache-idle-slots` on.
  - ollama 0.40.6: `NUM_PARALLEL=3`, `CONTEXT_LENGTH=32768`, FA on; 29/29 layers on the GPU.
- **The job was killed by its own `--max-min 200` cap at 08:23 (rc 143), not by a failure.** Every completed run had
  0 request errors. A full board takes ~220–240 min; `scripts/agent_board.sh` now says so and asks for
  `--max-min ≥ 270`.
- **What completed:**
  - Axon: all 3 rounds × both speeds.
  - llama-server: speed 1 all 3 rounds, speed 4 rounds 0–1 (**n = 2**).
  - ollama: rounds 0–1 at both speeds (**n = 2**).
- **Taint:** axon-run marked the whole job TAINTED (FOREIGN_GPU max 36 %, mean 1.1 %: kitty 23 %, Xorg 13 %). The
  per-run labels below come from each run's own `nvidia-smi pmon` sample (TAINTED = mean > 5 % or max > 25 %). The
  tainted runs fall inside the clean runs' ranges.

**speed 1** (median of rounds; [min–max])

| metric | axon f16 (n=3: r0,1,2) | llama f16 (n=3: r0,1,2) | ollama f16 (n=2: r0,1) |
|---|---|---|---|
| TTFT-new p50 s | 4.12 [4.05–4.28] | 5.99 [5.98–6.1] | 4.9 [4.89–4.91] |
| TTFT-new p99 s | 8.65 [8.62–13.1] | 32.5 [29.3–33.1] | 10.3 [8.28–12.3] |
| TTFT-resume p50 s | 0.276 [0.26–0.281] | 0.951 [0.905–1.03] | 1.03 [1.01–1.06] |
| TTFT-resume p99 s | 15.2 [15.2–15.2] | 22.5 [22.3–24.2] | 16.5 [16.5–16.5] |
| resume 30s-2m p50 | 0.224 [0.205–0.255] | 0.884 [0.877–1.01] | 0.86 [0.84–0.88] |
| resume 2-10m p50 | 4.72 [4.7–4.72] | 10.9 [10.8–11.5] | 5.23 [5.18–5.28] |
| resume >10m p50 | 7.08 [7.04–7.57] | 20.5 [20.4–20.6] | 7.06 [6.98–7.15] |
| TTFT-turn p50 s | 0.122 [0.118–0.123] | 1.06 [0.749–1.07] | 0.439 [0.42–0.458] |
| goodput tok/s | 17.4 [17.4–17.7] | 13 [12.6–13.3] | 9 [8.97–9.04] |
| ITL p99 ms | 119 [119–122] | 611 [607–633] | 232 [228–235] |
| warm/GB | 2.54 [2.37–2.54] | 0.887 [0.71–0.887] | 1.51 [1.42–1.6] |
| re-prefill share | 0.103 [0.103–0.119] | 0.019 [0.0186–0.0194] | n/a |
| output tok/s | 33 [33–33] | 32.4 [32.4–32.7] | 17.6 [17.5–17.7] |
| wall s | 604 [604–604] | 616 [610–616] | 600 [600–600] |
| errors | 0 [0–0] | 0 [0–0] | 0 [0–0] |
| tainted runs | 1/3 (r0) | 1/3 (r0) | 1/2 (r1) |
| reply md5 stable over rounds | yes | no (3) | no (2) |

**speed 4** (median of rounds; [min–max])

| metric | axon f16 (n=3: r0,1,2) | llama f16 (n=2: r0,1) | ollama f16 (n=2: r0,1) |
|---|---|---|---|
| TTFT-new p50 s | 30.1 [29.5–30.5] | 44.9 [44.6–45.3] | 7.9 [7.81–8] |
| TTFT-new p99 s | 62.7 [56.4–63.5] | 217 [216–219] | 26 [25.8–26.2] |
| TTFT-resume p50 s | 15.9 [15.5–25.1] | 12.1 [12.1–12.1] | 14.8 [13.8–15.7] |
| TTFT-resume p99 s | 52.3 [42.9–52.7] | 191 [190–192] | 22.1 [21.7–22.4] |
| resume 30s-2m p50 | 7.27 [6.94–25.1] | 12.1 [12.1–12.1] | 9.94 [9.77–10.1] |
| resume 2-10m p50 | 15.5 [15.5–15.9] | 8.88 [8.86–8.9] | 14.8 [13.8–15.7] |
| resume >10m p50 | n/a | n/a | n/a |
| TTFT-turn p50 s | 17 [12.5–20.5] | 25.5 [25.5–25.5] | 3.43 [3.25–3.61] |
| goodput tok/s | 5.49 [5.21–6.03] | 2.15 [2.12–2.18] | 5.97 [5.79–6.16] |
| ITL p99 ms | 137 [137–140] | 2.29e+03 [2.27e+03–2.31e+03] | 359 [352–366] |
| warm/GB | 0.677 [0.677–0.678] | 0.266 [0.177–0.355] | 0.266 [0.177–0.355] |
| re-prefill share | 0.475 [0.443–0.477] | 0.344 [0.344–0.345] | n/a |
| output tok/s | 33.4 [33.1–35.1] | 13.7 [13.4–14] | 47 [46.5–47.4] |
| wall s | 597 [568–601] | 1.45e+03 [1.43e+03–1.48e+03] | 224 [223–226] |
| errors | 0 [0–0] | 0 [0–0] | 0 [0–0] |
| tainted runs | 0/3 | 1/2 (r0) | 0/2 |
| reply md5 stable over rounds | yes | no (2) | no (2) |

### Reading the table
- **ollama stops early.** It ended ~40 % of replies before `max_tokens` (`out_tokens_short` 69–72 of 174; EOS on
  filler; Axon and llama-server: 0). Its output tok/s, goodput and speed-4 wall time are therefore not comparable to
  the other two.
- **Reply md5 is stable across rounds only on Axon.** llama-server and ollama vary run to run at temperature 0.
- **Speed 4 overloads every engine.** The 150-s window takes 568–601 s on Axon, 1,427–1,482 s on llama-server and
  ~224 s on ollama (shorter replies). That cell measures queueing.
- **Axon's output rate is the same ~33 tok/s at speed 1 and speed 4,** with 47 % re-prefill at speed 4: prefill, not
  decode, is the bound there. (Observation, not part of rule A.)
- **llama-server re-prefills 1.9 % at speed 1 against Axon's 10.3 %.** Its default host-RAM prompt cache
  (`--cache-ram` / `--cache-idle-slots`) restores idle prefixes, which is what B would add to Axon. Axon still has the
  lower TTFT-resume and TTFT-turn at speed 1.

## Rule A (§6 row A) on Axon main: B/C are NOT deprioritised
The rule: if TTFT-resume p50 ≤ 1.0 s **and** re-prefill < 10 % of prompt tokens, B/C drop below item 3.

| Axon main, f16 | TTFT-resume p50 | re-prefill share | rule |
|---|---|---|---|
| speed 1 (n = 3) | 0.276 s [0.260–0.281] ✓ | **10.3 % [10.3–11.9 %]** ✗ (every round > 10 %) | fails |
| speed 4 (n = 3) | **15.9 s [15.5–25.1]** ✗ | **47.5 % [44.3–47.7 %]** ✗ | fails |

- **Verdict:** both clauses must hold, so B/C stay where they are.
  - At speed 4 the rule fails on both clauses by far.
  - At speed 1 it fails on re-prefill by a narrow margin: 0.3 pp at the median, and the lowest round is 10.31 %.
- **Robust to the killed runs:** the rule reads only Axon main, which has all 3 rounds at both speeds.
- **The second rule (C becomes dedupe-only if the shared prefix < 2k):** a new subagent's shared prefix is p50 6,006
  tokens, so C stays a cross-session index.

## Does the f16 board need a rerun?
**Recommendation: no full rerun for A.** Ship A with the baseline cells labelled n = 2 where they are.
- The B/C verdict depends only on Axon's n = 3 cells.
- The n = 2 baseline cells are context, not gates.
- Rerun f16 (≈ 4 h of exclusive bella) only if a later B/C gate compares against a baseline cell.
- The q8 board (`1027-081924-51c9`, `--max-min 270`) should give n = 3 everywhere for q8.
