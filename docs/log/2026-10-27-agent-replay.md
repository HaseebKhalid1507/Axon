# Agent workload PR A: trace extractor, replay client, baseline board (2026-10-27)

Spec: [design/agent-workload.md](../design/agent-workload.md) §1 and §6, row A. No engine changes. PR #160.

**Status:**
- **f16 board: done with gaps.** Job `1027-014629-7829` ran 15 of 18 runs. Its ollama column is **non-comparable**
  (the chat-template bug, below).
- **Rule A, f16: fails, so B/C stay.** q8 cannot change that under the combination pre-registered below.
- **q8 board: done** (job `1027-083752-1e62`). Rule A passes on q8, but the pre-registered combination needs both KV
  types, so the **combined verdict is: B/C are NOT deprioritised.** ollama q8 speed 1 is rerunning (all 3 rounds
  were tainted).

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


## Baseline board, q8 (MEASURED; job `1027-083752-1e62`, fixed harness 8588874 + 1ba07d9)
**Host and setup:** bella, as for f16, at `--kv q8`. Axon's pool was 190,336 / 193,408 / 192,832 tokens across the
rounds. llama-server ran `-c 163840 -np 5`; ollama `NUM_PARALLEL=5`.

**Prompt checks:**
- Every engine passed the 512-word tokenization check.
- 0 prompt-count mismatches on all 18 runs.
- ollama's `prompt_eval_count` equals the sent prompt on every request (1,018 of 1,018), so the raw path sends
  exactly our prompts.

**The job:** rc 0, `BOARD END ok`. axon-run labelled the whole job TAINTED (FOREIGN_GPU max 49 %, mean 1.2 % over
287 samples).

### Taint: axon-run's over-threshold samples mapped to cells
The methodology's taint rule is the **harness's per-run flag**: `nvidia-smi pmon` every ~5 s over the run, TAINTED
if the run's mean > 5 % or any sample > 25 %. Applied as written:

| cell (engine, speed, round) | harness per-run flag | axon-run samples over threshold in the cell |
|---|---|---|
| llama s1 r0 | **TAINTED** (max 27) | 12:47:47–12:49:49: 4 windows with mean > 5 % |
| llama s4 r0 | **TAINTED** (max 27) | 12:50:49–12:54:53: 5 windows with mean > 5 % |
| ollama s1 r0 | **TAINTED** (max 98) | 13:02:58: max 49 % (zen-bin), mean 9.8 % |
| ollama s1 r1 | **TAINTED** (max 44) | 13:54:05: max 26 % (Xorg 17 %, zen-bin 9 %) |
| ollama s1 r2 | **TAINTED** (max 28) | 14:40:02: mean 5.4 % (zen-bin) |
| llama s1 r1 | clean (mean 0.7, max 15) | 13:37:10: one 30-s window, mean 7.7 %, max 24 % |
| llama s4 r2 | clean (mean 1.0, max 11) | 14:36:24: one 30-s window, mean 5.6 % |

- **Clean throughout** (no over-threshold sample by either sampler): Axon all 6 cells; llama s1 r2 and s4 r1; ollama
  s4 r0, r1, r2.
- **Under the written rule:**
  - Tainted cells are **missing rounds**, not averaged in.
  - Axon is n = 3 at both speeds; llama-server is n = 2 at both (r1, r2); ollama s4 is n = 3.
  - **ollama s1 has no clean round.** A targeted rerun of only that cell (3 rounds) is queued.
- **A stricter reading** (any axon-run window over threshold taints the cell) would leave llama s1 with r2 only and
  s4 with r1 only (n = 1). It is not the written rule; it is stated so the reader can see the difference. Rule A
  reads only Axon, which is clean under both.

### ollama's engine-side aborts
- **ollama aborts 3–5 requests per cell mid-stream** with "prediction aborted, token repeat limit reached". This is
  ollama's own repetition guard firing on filler-conditioned output: the same requests (k 22419, 22425, 22491…) abort
  in every round. **It is not a harness bug.** Axon and llama-server have no such guard and decode to `max_tokens`.
- **How they are counted:**
  - They are errors and contribute 0 to goodput.
  - In this board's runs the summary dropped them from the TTFT / ITL samples, which can flatter ollama. Bounds from
    the per-request records, putting each aborted request at 0 or at +∞:
    - TTFT-turn p50 moves by ≤ 0.03 s at speed 1 and ≤ 0.25 s at speed 4;
    - TTFT-new p50 at speed 4 could be up to +1.5 s worse (1 aborted new request);
    - TTFT-resume: no aborted requests.
  - From d7f5fd3 on, an aborted request keeps its TTFT and ITL in the latency stats, counts as an error, and never
    counts toward goodput. The rerun uses that.
- **Re-prefill correction:** ollama's `prompt_eval_count` is the whole prompt even when it is cached. MEASURED: it
  equals the sent prompt on all 1,018 requests while TTFT-turn stays ~0.4 s. So it can't measure re-prefill;
  ollama's re-prefill is n/a. The harness first read it as recomputed tokens (0.93 share, wrong); that is fixed.
- **llama-server's net re-prefill is negative at speed 4** (−4 %): `--cache-reuse 256` reuses KV chunks outside the
  common prefix by shifting them (not bit-reproducible, per its README), so it computes less than the prefix ideal.
  The clamped share is 0.73 %.

### Board q8, clean rounds only (`agent_replay.py table --clean-only`)

**speed 1** (median of rounds; [min–max])

| metric | axon q8 (n=3: r0,1,2) | llama q8 (n=2: r1,2) |
|---|---|---|
| TTFT-new p50 s | 4.14 [4.13–4.23] | 9.51 [8.52–10.5] |
| TTFT-new p99 s (max when n < 100) | 7.38 [7.38–7.4] | 52.8 [49.2–56.4] |
|   n new | 13 [13–13] | 13 [13–13] |
| TTFT-resume p50 s | 0.14 [0.136–0.142] | 0.798 [0.768–0.829] |
| TTFT-resume p99 s (max when n < 100) | 5.57 [5.38–5.59] | 21.3 [16.3–26.3] |
|   n resume | 34 [34–34] | 34 [34–34] |
| resume 30s-2m p50 | 0.182 [0.18–0.194] | 0.82 [0.81–0.829] |
|   n 30s-2m | 29 [29–29] | 29 [29–29] |
| resume 2-10m p50 | 0.076 [0.076–0.081] | 0.8 [0.768–0.833] |
|   n 2-10m | 4 [4–4] | 4 [4–4] |
| resume >10m p50 | 0.04 [0.038–0.04] | 0.226 [0.174–0.277] |
|   n >10m | 1 [1–1] | 1 [1–1] |
| TTFT-turn p50 s | 0.109 [0.107–0.115] | 0.647 [0.637–0.657] |
|   n turn | 127 [127–127] | 127 [127–127] |
| goodput tok/s | 20.7 [20.4–20.8] | 10.9 [9.98–11.8] |
| ITL p99 ms | 119 [117–119] | 916 [858–974] |
| warm/GB (GPU KV budget) | 2.55 [2.55–2.59] | 1.1 [1–1.2] |
| re-prefill share (clamped) | 0.0127 [0.0127–0.0127] | 0.0075 [0.0075–0.0075] |
| re-prefill share (net) | 0.0123 [0.0123–0.0123] | 0.0072 [0.0072–0.0072] |
| preempted (axon /slots) | 0 [0–0] | n/a |
| output tok/s | 33 [33–33] | 29.8 [29.7–30] |
| replies short of max_tokens | 0 [0–0] | 0 [0–0] |
| prompt-count mismatches | 0 [0–0] | 0 [0–0] |
| wall s | 604 [604–604] | 668 [664–672] |
| errors (all kinds) | 0 [0–0] | 0 [0–0] |
|   engine aborts mid-stream (latency counted, no goodput) | n/a | n/a |
| tainted runs | 0/3 | 0/2 |
| reply md5 stable over rounds | yes | no (2) |

**speed 4** (median of rounds; [min–max])

| metric | axon q8 (n=3: r0,1,2) | llama q8 (n=2: r1,2) | ollama q8 (n=3: r0,1,2) |
|---|---|---|---|
| TTFT-new p50 s | 6.5 [6.48–6.64] | 9.27 [6.64–11.9] | 6.53 [6.11–6.71] |
| TTFT-new p99 s (max when n < 100) | 14 [14–14] | 31.2 [25–37.4] | 19.1 [18.1–20.1] |
|   n new | 13 [13–13] | 13 [13–13] | 12 [12–12] |
| TTFT-resume p50 s | 0.102 [0.098–0.159] | 5.51 [4.53–6.49] | 4.37 [3.48–5.14] |
| TTFT-resume p99 s (max when n < 100) | 11.7 [1.77–11.7] | 27.6 [20.4–34.9] | 6.41 [5.98–7.82] |
|   n resume | 5 [5–5] | 5 [5–5] | 5 [5–5] |
| resume 30s-2m p50 | 0.098 [0.097–0.102] | 1.12 [0.644–1.59] | 4.37 [3.48–5.14] |
|   n 30s-2m | 4 [4–4] | 4 [4–4] | 4 [4–4] |
| resume 2-10m p50 | 0.052 [0.046–0.159] | 5.51 [4.53–6.49] | 0.513 [0.234–0.658] |
|   n 2-10m | 1 [1–1] | 1 [1–1] | 1 [1–1] |
| resume >10m p50 | n/a | n/a | n/a |
|   n >10m | 0 [0–0] | 0 [0–0] | 0 [0–0] |
| TTFT-turn p50 s | 0.159 [0.158–0.163] | 7.24 [5.61–8.88] | 1.74 [1.57–1.82] |
|   n turn | 156 [156–156] | 156 [156–156] | 152 [152–153] |
| goodput tok/s | 26.1 [25.7–29.1] | 7.97 [7.68–8.27] | 13.4 [13.3–14.2] |
| ITL p99 ms | 150 [150–152] | 1.2e+03 [1.09e+03–1.3e+03] | 373 [371–382] |
| warm/GB (GPU KV budget) | 3.06 [3.06–3.1] | 0.3 [0.2–0.401] | 1.2 [1.2–1.4] |
| re-prefill share (clamped) | 0.0127 [0.0075–0.0127] | 0.0073 [0.0052–0.0094] | n/a |
| re-prefill share (net) | 0.0123 [0.0072–0.0123] | -0.0403 [-0.0424–-0.0383] | n/a |
| preempted (axon /slots) | 0 [0–0] | n/a | n/a |
| output tok/s | 101 [101–104] | 45.1 [44.1–46.1] | 70 [68.3–70.3] |
| replies short of max_tokens | 0 [0–0] | 0.5 [0–1] | 0 [0–1] |
| prompt-count mismatches | 0 [0–0] | 0 [0–0] | 0 [0–0] |
| wall s | 197 [192–197] | 441 [431–451] | 265 [264–277] |
| errors (all kinds) | 0 [0–0] | 0 [0–0] | 5 [4–5] |
|   engine aborts mid-stream (latency counted, no goodput) | n/a | n/a | n/a |
| tainted runs | 0/3 | 0/2 | 0/3 |
| reply md5 stable over rounds | yes | no (2) | no (3) |

## Rule A on f16 + q8 (pre-registered reading: speed 1, clean rounds, both KV types must pass)
| Axon main, speed 1, clean rounds | TTFT-resume p50 (≤ 1.0 s) | re-prefill (< 10 %) | rule |
|---|---|---|---|
| f16 (r1, r2) | 0.281 / 0.276 s ✓ | **10.31 % / 10.27 %** (clamped / net) ✗ | fails |
| q8 (r0, r1, r2) | 0.140 [0.136–0.142] s ✓ | 1.27 % / 1.23 % ✓ | passes |
| **combined** | | | **fails: B/C are NOT deprioritised** |

- **q8 passes clearly:** 1.86× the pages for the same 20 sessions.
- **f16 fails narrowly,** and under the pre-registered combination f16 decides.
- **What this means:**
  - At Axon's default f16 KV, ~10 % of prompt tokens are recomputed because whole slots are reclaimed. q8's larger
    pool hides the loss.
  - B (evict to RAM) attacks exactly that loss, and so does C's page-level retention.
- **Other #160 rules:**
  - The shared prefix is p50 6,006 ≥ 2k, so C stays a cross-session index.
  - Row A "ships regardless" (it is the ruler).
  - Row C's fan-out gate is re-registered and is not exercised here.

## Next steps
- **q8 ollama speed 1:** targeted rerun `1027-145644-71fd` (3 rounds, numbered r3–r5, `--timing bella --low
  --max-min 55`).
- **f16, deferred:** the ollama column is non-comparable (chat template) and llama s4 is n = 1 clean. A targeted
  rerun (ollama s1 + s4 ×3 on the raw path, llama s4 ×2; ≈ 2 h) is needed only if a B/C gate compares against a
  baseline. Rule A doesn't.
