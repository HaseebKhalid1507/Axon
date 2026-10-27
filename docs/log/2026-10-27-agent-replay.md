# Agent workload PR A: trace extractor, replay client, baseline board (2026-10-27)

Spec: [design/agent-workload.md](../design/agent-workload.md) §1 and §6, row A. No engine changes. PR #160.

**Status:**
- **f16 board: done with gaps.** Job `1027-014629-7829` ran 15 of 18 runs. Its ollama column is **non-comparable**
  (the chat-template bug, below).
- **Rule A, f16: fails, so B/C stay.** q8 cannot change that under the combination pre-registered below.
- **q8 board:** pending, on the fixed harness.

## What landed
- **`scripts/agent_trace.py`**: Synaps logs → a numbers-only trace.
  - Sessions are recovered as Anthropic cache chains: cache_read(n+1) = read(n) + create(n). That holds for 97.1 % of
    22.8k requests, MEASURED over 8 log days.
  - Pools and forks come from where a new chain's cached prefix ends; subagents from dispatch lines.
  - Modes:
    - `--check`: no string value > 16 bytes.
    - `--check-committed`: CI; every committed trace must be `"synthetic": true` and pass `--check`.
    - `--anonymize-time`: no date or time of day; each day starts at t = 0.
    - `--synthetic`: writes the example trace.
  - **The repo is public: no trace from real logs is ever committed, not even a numbers-only one.** The script
    refuses to write one inside a git tree (the check follows symlinks). The repo carries only
    `scripts/examples/agent-trace-synthetic.json`.
- **`scripts/agent_replay.py`**:
  - Subcommands: `run` (one cell against a URL), `board` (engines in interleaved rounds), `fanout` (row C's cell,
    idle and busy) and `table`.
  - Wrapper scripts: `scripts/agent_board.sh` (the board job, with its measured duration) and
    `scripts/agent_board_kill_smoke.sh` (no engine outlives a killed board).
- **CI:** `ci.yml` runs `agent_trace.py --check-committed`.

## Trace structure (MEASURED, Claude tokens, 8 days)
- **Prompts:** p50 / p90 / p99 165.9k / 412k / 812k; 97.6 % of prompt tokens cached.
- **Per-turn deltas:** `new` p50 1,110; output p50 353.
- **Gaps and sessions:** per-session gap p50 8.2 s; 662 chains, 279 of them subagent chains.
- **A new subagent's shared prefix:** p50 6,006 (p25 0, p75 6,363).

## Cell, engines and how §1.3 is measured
- **Cell agent-32k:**
  - Scale 0.15, clip 32,768, taken from the 10-min trace window with the most sessions: 174 requests, 20 sessions,
    13 new, 7 primed (133k tokens), prompt p50 17.2k.
  - Speed 1 is 600 s; speed 4 is the same window in 150 s.
  - Filler: single-token words from the GGUF vocab. Every engine must count a fresh 512-word probe as exactly 512
    prompt tokens at start, and every axon / llama-server request's own prompt count must equal what was sent; a
    mismatch fails the board.
- **Engines:**
  - **Axon:** `serve --ctx 32768 --parallel 16 --kv`, default paged pool. Its size is the KV budget for every engine.
  - **llama-server:** `-np <budget // 32768> -c <np × 32768> -kvu --cache-reuse 256 -fa on`, `cache_prompt`.
  - **ollama:** its own instance, `NUM_PARALLEL = budget // 32768`, `CONTEXT_LENGTH 32768`, FA, KV type. Since the
    fix it is sent prompts through `/api/generate` with `raw: true`.
- **Where §1.3 departs from the spec:**
  - **Re-prefill** comes from each response's usage, not `/metrics`. Axon: `prompt_tokens − cached_tokens`.
    llama-server: `timings.prompt_n`. ollama (raw): `prompt_eval_count`. The ideal is the tokens not in any prefix
    sent earlier in send order (16-token blocks). Reported both clamped at 0 per request and net.
  - **warm/GB** divides by the engine's configured **GPU** KV budget, not "KV bytes used (GPU + RAM tiers)". So it
    leaves out llama-server's 8 GiB `--cache-ram` host prompt cache (default on, with `--cache-idle-slots`) and the
    8,192 MiB prompt cache ollama's runner logs. Both engines have a host tier that Axon lacks; the denominator
    doesn't count it.
- **Preemption** (row A asks how it is counted):
  - On a preempted request, Axon's `preempt` / `unfeed` set `cached = 0`, so usage counts the re-fed prompt. It
    misses the lost first feed and the re-fed generated tokens: an undercount.
  - llama-server doesn't preempt (it queues).
  - **f16 board: 0 preemptions.** No `preempted after` line in any of the three Axon serve logs.
  - From the fix on, every run records Axon's `/slots` `preempted` and `prefills_unfed` deltas.

## Baseline board, f16 (MEASURED; job `1027-014629-7829`)
**Host:** bella, RTX 5070 Ti Laptop 12 GB, driver 615.78, Core Ultra 9 275HX, 30 GB RAM, CachyOS 7.2.11.

### Gaps and caveats
- **The job was killed by its own `--max-min 200` cap (rc 143), not by a failure.**
  - Completed: Axon all 3 rounds; llama-server speed 4 n = 2; ollama n = 2.
  - A full board takes ~220–240 min: submit with `--max-min ≥ 270`.
- **ollama is NON-COMPARABLE in this board.** Its `/v1/completions` applied the Qwen chat template:
  - The 24,508-token prime was evaluated as 24,537 tokens (+29), and later prompts as +5.
  - The model then answered a chat turn and stopped at `<|im_end|>`: never more than 146 decoded tokens, ~41 % of
    replies short, about half of Axon's output tokens.
  - Every ollama cell, not only throughput, comes from a lighter, different workload. (The first version of this log
    said "EOS on filler"; that was wrong.)
  - The harness now uses `/api/generate` `raw: true` and asserts prompt counts.
- **Taint correction.** The first version said "the tainted runs fall inside the clean runs' ranges". **That was
  false.**
  - Axon speed 1 r0 (tainted) vs the clean r1 / r2: re-prefill 11.89 % vs 10.31 / 10.31; warm/GB 2.371 vs 2.54;
    TTFT-new 4.277 vs 4.046–4.125; ITL p99 121.95 vs 118.99–119.34.
  - llama-server speed 1 r0 (tainted) vs clean: TTFT-turn 0.749 vs 1.056–1.068; goodput 13.35 vs 12.63–13.0.
  - Rule A below reads the clean rounds.
- **Samples are per metric:** TTFT-new n = 13 per run, so its "p99" is the max. TTFT-resume n = 34 at speed 1 and
  5 at speed 4. The 2–10 min bucket is n = 4, the > 10 min bucket n = 1.

**speed 1** (median of rounds; [min–max])

| metric | axon f16 (n=3: r0,1,2) | llama f16 (n=3: r0,1,2) | ollama f16 (n=2: r0,1) NON-COMPARABLE: chat template |
|---|---|---|---|
| TTFT-new p50 s | 4.12 [4.05–4.28] | 5.99 [5.98–6.1] | 4.9 [4.89–4.91] |
| TTFT-new p99 s (max when n < 100) | 8.65 [8.62–13.1] | 32.5 [29.3–33.1] | 10.3 [8.28–12.3] |
|   n new | 13 [13–13] | 13 [13–13] | 13 [13–13] |
| TTFT-resume p50 s | 0.276 [0.26–0.281] | 0.951 [0.905–1.03] | 1.03 [1.01–1.06] |
| TTFT-resume p99 s (max when n < 100) | 15.2 [15.2–15.2] | 22.5 [22.3–24.2] | 16.5 [16.5–16.5] |
|   n resume | 34 [34–34] | 34 [34–34] | 34 [34–34] |
| resume 30s-2m p50 | 0.224 [0.205–0.255] | 0.884 [0.877–1.01] | 0.86 [0.84–0.88] |
|   n 30s-2m | 29 [29–29] | 29 [29–29] | 29 [29–29] |
| resume 2-10m p50 | 4.72 [4.7–4.72] | 10.9 [10.8–11.5] | 5.23 [5.18–5.28] |
|   n 2-10m | 4 [4–4] | 4 [4–4] | 4 [4–4] |
| resume >10m p50 | 7.08 [7.04–7.57] | 20.5 [20.4–20.6] | 7.06 [6.98–7.15] |
|   n >10m | 1 [1–1] | 1 [1–1] | 1 [1–1] |
| TTFT-turn p50 s | 0.122 [0.118–0.123] | 1.06 [0.749–1.07] | 0.439 [0.42–0.458] |
|   n turn | 127 [127–127] | 127 [127–127] | 127 [127–127] |
| goodput tok/s | 17.4 [17.4–17.7] | 13 [12.6–13.3] | 9 [8.97–9.04] |
| ITL p99 ms | 119 [119–122] | 611 [607–633] | 232 [228–235] |
| warm/GB (GPU KV budget) | 2.54 [2.37–2.54] | 0.887 [0.71–0.887] | 1.51 [1.42–1.6] |
| re-prefill share (clamped) | 0.103 [0.103–0.119] | 0.019 [0.0186–0.0194] | n/a |
| re-prefill share (net) | n/a | n/a | n/a |
| preempted (axon /slots) | n/a | n/a | n/a |
| output tok/s | 33 [33–33] | 32.4 [32.4–32.7] | 17.6 [17.5–17.7] |
| replies short of max_tokens | 0 [0–0] | 0 [0–0] | 71 [70–72] |
| prompt-count mismatches | n/a | n/a | n/a |
| wall s | 604 [604–604] | 616 [610–616] | 600 [600–600] |
| errors | 0 [0–0] | 0 [0–0] | 0 [0–0] |
| tainted runs | 1/3 (r0) | 1/3 (r0) | 1/2 (r1) |
| reply md5 stable over rounds | yes | no (3) | no (2) |

**speed 4** (median of rounds; [min–max])

| metric | axon f16 (n=3: r0,1,2) | llama f16 (n=2: r0,1) | ollama f16 (n=2: r0,1) NON-COMPARABLE: chat template |
|---|---|---|---|
| TTFT-new p50 s | 30.1 [29.5–30.5] | 44.9 [44.6–45.3] | 7.9 [7.81–8] |
| TTFT-new p99 s (max when n < 100) | 62.7 [56.4–63.5] | 217 [216–219] | 26 [25.8–26.2] |
|   n new | 13 [13–13] | 13 [13–13] | 13 [13–13] |
| TTFT-resume p50 s | 15.9 [15.5–25.1] | 12.1 [12.1–12.1] | 14.8 [13.8–15.7] |
| TTFT-resume p99 s (max when n < 100) | 52.3 [42.9–52.7] | 191 [190–192] | 22.1 [21.7–22.4] |
|   n resume | 5 [5–5] | 5 [5–5] | 5 [5–5] |
| resume 30s-2m p50 | 7.27 [6.94–25.1] | 12.1 [12.1–12.1] | 9.94 [9.77–10.1] |
|   n 30s-2m | 4 [4–4] | 4 [4–4] | 4 [4–4] |
| resume 2-10m p50 | 15.5 [15.5–15.9] | 8.88 [8.86–8.9] | 14.8 [13.8–15.7] |
|   n 2-10m | 1 [1–1] | 1 [1–1] | 1 [1–1] |
| resume >10m p50 | n/a | n/a | n/a |
|   n >10m | 0 [0–0] | 0 [0–0] | 0 [0–0] |
| TTFT-turn p50 s | 17 [12.5–20.5] | 25.5 [25.5–25.5] | 3.43 [3.25–3.61] |
|   n turn | 156 [156–156] | 156 [156–156] | 156 [156–156] |
| goodput tok/s | 5.49 [5.21–6.03] | 2.15 [2.12–2.18] | 5.97 [5.79–6.16] |
| ITL p99 ms | 137 [137–140] | 2.29e+03 [2.27e+03–2.31e+03] | 359 [352–366] |
| warm/GB (GPU KV budget) | 0.677 [0.677–0.678] | 0.266 [0.177–0.355] | 0.266 [0.177–0.355] |
| re-prefill share (clamped) | 0.475 [0.443–0.477] | 0.344 [0.344–0.345] | n/a |
| re-prefill share (net) | n/a | n/a | n/a |
| preempted (axon /slots) | n/a | n/a | n/a |
| output tok/s | 33.4 [33.1–35.1] | 13.7 [13.4–14] | 47 [46.5–47.4] |
| replies short of max_tokens | 0 [0–0] | 0 [0–0] | 69.5 [69–70] |
| prompt-count mismatches | n/a | n/a | n/a |
| wall s | 597 [568–601] | 1.45e+03 [1.43e+03–1.48e+03] | 224 [223–226] |
| errors | 0 [0–0] | 0 [0–0] | 0 [0–0] |
| tainted runs | 0/3 | 1/2 (r0) | 0/2 |
| reply md5 stable over rounds | yes | no (2) | no (2) |

## Rule A: pre-registered reading (written 10-27, after f16 and before any q8 data), then the verdict

### How rule A is read
- **Speeds:** it reads **speed 1**. Speed 4 overloads every engine (a 150-s window takes 568–1,482 s), so its
  TTFT-resume and re-prefill measure queueing more than cache loss. Speed 4 is reported, not read by the rule.
- **Rounds:** the median over the untainted rounds (all rounds if fewer than 2 are clean).
- **KV types:** B/C are deprioritised only if the rule holds for **both f16 and q8**. Either failing keeps them.
  Consequence, stated plainly: since f16 fails, **q8 cannot flip the verdict**; the q8 board is the baseline record.

### Verdict
| Axon main, f16, speed 1 | TTFT-resume p50 | re-prefill (clamped / net) | rule |
|---|---|---|---|
| clean rounds r1, r2 | 0.281 / 0.276 s ✓ | **10.31 % / 10.27 %** (both rounds) ✗ | **fails** |
| all rounds (for reference) | 0.276 [0.260–0.281] ✓ | 10.3 % [10.3–11.9] ✗ | fails |

- **B/C are NOT deprioritised:** f16 fails on re-prefill (≥ 10 % in every round, clean or not). The margin is
  narrow, 0.27–0.31 pp; it is not noise, since r1 and r2 are identical.
- **Speed 4, for reference:** 15.9 s and 47.5 %.
- **The second rule (C becomes dedupe-only if the shared prefix < 2k):** p50 6,006 tokens, so C stays a
  cross-session index.
- **The f16 rerun question:** the verdict reads only Axon cells, which are complete, so no full rerun for A. Rerun
  only if a later gate compares against a baseline cell.

## Fan-out cell: row C re-registered 10-27, before any C data
- **What changed:** the spec's row C fan-out gate is re-registered in this PR (A owns the cell; no C data exists).
  The new text is in the spec's §6 row C.
- **Why: the leader's prefill is on every sibling's critical path.** The absolute form needs S ≳ 16k at d = 1,024
  (≈ 17.2k at d = 1,110) even for a perfect build. ESTIMATED with 2 fifo lanes: ratio ≈ (T + 4t) / 5T, where
  T = (S + d)/R and t = (d + 32)/R. So ≤ 0.25 iff S + d ≥ 16 (d + 32). The first version said "≳ 19k".
- **Fix: the incremental form.** (siblings 2–8 TTFT − the leader's TTFT) p50 ≤ 0.25× main's same quantity,
  ≈ (d + 32)/(S + d) ≈ 0.16 at S = 6,006 and d = 1,110 (A's p50s, unscaled). The spec's short-span fallback is kept.
- **Added after the review, still before any C data:**
  - A guard: the leader's TTFT p50 ≤ 1.05× main's, so the gate can't be passed by slowing the leader.
  - The leader is the first-sent sibling.
- **Two variants, both gates:** idle, and busy (4 decoding sessions), since dedupe never fires from an idle server.
- **Unchanged:** warm/GB ≥ 1.4× main on agent-32k speed 4, md5 under `AXON_PAGE_SHUFFLE`, C=1 decode within 2 %.
- **Harness:** `agent_replay.py fanout [--busy 4]`. It runs when C benches.

## Harness bugs found on the way (all fixed)
- **Smoke 1:** the wrong model name to `axon serve`; llama-server `-np 8` overflowing a unified pool; the summary
  crashing on an error result.
- **Engine leak:** a llama-server started in its own session survived axon-run's killpg of the board and held
  10 GB of bella's VRAM for 37 min.
  - Engines now stay in the job's process group, get `PR_SET_PDEATHSIG`, and are tree-killed by signal handlers,
    at exit and after a failed start.
  - A leftover on our port is killed only if it is ours with the exact command line.
  - Checked by `agent_board_kill_smoke.sh`.
- **ollama's chat template** (above).
- **The ideal-cache hit** is now assigned at send time, not at thread start (≤ 0.04 pp on f16).
