# Hardening: fuzzing, soak and the model zoo

Roadmap item 2. Speed is measured everywhere else in these docs; this page is about whether Axon survives what
real clients and real files do to it. Three tools, all stdlib Python in `scripts/`:

| tool | what it throws at Axon | gate |
|---|---|---|
| `fuzz_gguf.py` (#37) | ~1800 mutants per model: truncations, bad offsets, oversized dims, size overflow, hostile metadata | no panic, no OOB, no hang; every refusal is one clear line |
| `fuzz_serve.py` (#37, #61) | malformed JSON, huge bodies, bad UTF-8, invalid params, slow-loris, disconnects, auth, `/metrics`, stalled readers | a 4xx with a JSON message (overload: 503); the server stays up and `/health` answers |
| `soak.py` (this page) | hours of a realistic mixed load | below |
| `zoo.py` (this page) | GGUFs from the wild, from different converters and quant mixes | each one matches llama.cpp or is refused with a clear error |

## Soak (`scripts/soak.py`)

```
scripts/soak.py <axon> <model.gguf> --hours H [--clients N] [--slots S] [--ctx C] [--out DIR] [-- serve flags]
```

It starts `axon serve` itself (`--ctx C`, `--parallel S` when S > 1), so it knows the pid, and runs N client
threads until the deadline. Each client picks from a weighted mix and waits up to 1 s between requests:

| kind | weight | what |
|---|---:|---|
| chat | 30 | short chat, 60 % streamed; half greedy, half sampled (random temperature, top_p, seed) |
| agent | 14 | a long system prompt shared by every client (¼ of the context) + 1–3 short turns: the prefix cache across slots |
| tool | 10 | a `get_weather` call (`tool_choice` auto, required or named), then the tool-result turn, streamed |
| completion | 8 | `/v1/completions`, streamed or not |
| disconnect | 8 | a stream the client closes after 1–20 chunks (the cancel path) |
| invalid | 7 | truncated JSON, bad UTF-8, `temperature -1`, `max_tokens -5`, a 404 route; plus `n 3` and `logit_bias`, which are refused today because they are *unsupported*, not invalid (a 200 there is class `accepted-unsupported`: reported, not failing, so the case moves when the feature ships) |
| long | 6 | a prompt of 35–55 % of the context |
| slow-reader | 4 | a stream read 64 bytes at a time with 20–100 ms pauses through a 4 KiB `SO_RCVBUF`. Since this PR's review round: ~1500 tokens (~300 KB of SSE, well past the kernel socket buffers, so the server's writer really blocks), and with several slots a concurrent probe (greedy, 4 tokens) must get its first bytes within `--probe-ttft-s` (30 s), else `slow-blocks-others` (failing). **Harness changed, not yet exercised on a GPU soak** (the runs below used `max_tokens` 96, ~30 KB, which the socket buffers absorb: they tested a slow *client*, not server write back-pressure) |
| oversize | 3 | a prompt of 1.3× the context: must be a 4xx |
| burst | every 2 min | client 0 fires 3× the slot count of streams at once (503 back-pressure allowed) |

A client that gets a 503 backs off 1–3 s, as a real one would.

**Priming** (default; `--no-prime` skips it). Before the reference identity run, the harness sends every batch mix
the slots can form: for nb = 1..S concurrent streams, with ng = 0..nb of them greedy, twice each. On CUDA the
batched decode step is captured as a CUDA graph per (rows, greedy rows, sampled rows) on first use
(`cuda/qwen2/slots.rs`; at most `MAX_GRAPHS` = 64 kept, LRU). Each instantiated graph takes driver memory, so a mix
first seen hours into a soak shows up as a permanent VRAM step that isn't a leak. Priming makes these one-time
allocations before the memory window starts. The bounds are unchanged. Priming was added after the first avante
run. It did **not** remove that run's 4 MiB VRAM step: the primed rerun shows the same step at 27.6 min (see
below). So the step isn't, or isn't only, the batch-mix graph cache. Priming stays because it is cheap (80
requests, 6 s).

Every 30 s one CSV row (`DIR/samples.csv`; also printed as `CSV,...` so a remote job log keeps it): server RSS and
VmHWM (`/proc/PID/status`), VRAM (`nvidia-smi --query-compute-apps` for the server pid; the whole card is only
recorded, never judged, unless `--vram-card`), open fds, threads, request count, whole-request latency p50/p99 of
the successful requests, TTFT p50/p99 (first chunk with content), and counts by class: `ok`, `4xx-expected`,
`503`, `cancelled` (our disconnects), and the failing classes `5xx`, `unexpected-4xx` (a valid request refused),
`unexpected-200` (an invalid one accepted), `dropped` (a connection error on a valid request), `bad-body` (a
200 without `finish_reason`/`usage`, a stream without `[DONE]`, `tool_choice=required` without a call),
`timeout`, `other-status`, `503-undocumented` (a 503 whose body isn't one of the back-pressure messages
`too many queued requests` / `too many connections`) and `slow-blocks-others`. Since the review round the row
also has `RssAnon` and `RssFile` (`/proc/PID/status`) next to VmRSS.

**Greedy identity.** Four fixed greedy requests (two chats, a forced tool call and a third-of-the-context prompt)
run before the load, every 15 min under full load, and after it. Every run must return the same text, tool
calls and token counts as the first. Under load this also checks batch invariance: the reference ran alone, the
spot checks run next to 12 other clients.

**Verdict** (`DIR/summary.json`; exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE). Revised in the review round (below):

- the server is alive at the end, and its log has no `panicked`;
- no request in a failing class. 503s must carry a back-pressure message; their fraction of all requests is
  printed in the verdict, not capped;
- every identity run matches;
- after the load stops and keep-alives time out (35 s), fds and threads are back at the pre-load baseline
  (+2 fds of slack), and `/slots` shows no busy slot and an empty queue;
- server-side cancellation is reported: client disconnects vs the engine's `cancelled after N tokens` log lines.
  A disconnect that lands after the generation ended has nothing to cancel, so the ratio is not judged;
- **memory.** A least-squares line through the RSS series against time, over the samples after a 20-minute
  warm-up. The series is `RssAnon` (anonymous memory, where a leak lives) when sampled, else VmRSS. The slope is
  projected over 24 h with ±2 standard errors from the residuals: **PASS** if growth + 2 SE < bound, **FAIL**
  if growth − 2 SE > bound, **INCONCLUSIVE** in between. The bound is `max(64 MiB, 2 % of the median RSS)`;
  VRAM (per pid) uses the same rule with 16 MiB. Memory is **judged only over at least 4 h after the warm-up**.
  A shorter window is reported as NOT-JUDGED with what it would be, and doesn't decide the run.

Why the bound: the soak is meant to catch a leak that matters for a server left up for days. A 1 KiB/request
leak at ~1–7 requests/s is 85–600 MiB a day and fails. **Detection floor:** anything under 64 MiB/day passes by
construction. On avante (~25k requests/h) that is about 100 bytes per request, so this is not a proof of zero
growth. 2 % of RSS keeps the bound meaningful for a big CPU model. VRAM is allocated up front, so 16 MiB/day is
measurement slack.

Why the review round changed it: a fitted slope responds to a single step Δ at fraction f of a window T with
about 6·Δ·f·(1−f)/T. Projected over 24 h, a short run amplifies a one-off allocator step 24/T times; jade's FAIL
and bella-7B's PASS were the same artifact with opposite signs. The SE and the 4 h minimum stop a short run from
deciding either way. One limit remains. The SE assumes independent residuals, and a step makes them
autocorrelated, so a clean step can still give a confident wrong slope: jade's is 137 ± 11 MiB/day. The harness smoke run in the review round shows the same thing from the other side. On jade, 0.5B, with
`--min-judge-h 0.05` forced so a 5-minute window would be judged (1018-041451-3d5e), RssAnon rose 3.2 MiB during
warm-up. The fit projected that to 710 MiB/day with a ±2 SE interval of [321, 1099], so the run FAILED with
apparent confidence: early warm-up growth is autocorrelated and not linear, and the SE can't see that. **The
SE band alone does not protect the verdict; the ≥ 4 h minimum window does.** That smoke is a harness test, not a
result. It confirmed that the RssAnon/RssFile columns are recorded and that the slots-idle and cancel checks
run and pass. That is why
a longer window and RssAnon (not VmRSS, where file-backed pages being reclaimed and refaulted under memory
pressure look like growth) both matter.

## Model zoo (`scripts/zoo.py`)

```
scripts/zoo.py fetch --dir D
scripts/zoo.py run --axon BIN --llama-server LLAMA_CPU_SERVER --dir D --models-dir models --out zoo.json
scripts/zoo.py table zoo.json
```

For every file: `axon info`, `axon gen -n 64` greedy (CPU), then top-20 next-token logprobs from `axon logits`
and from llama.cpp's CPU `llama-server` given the same token ids, on 16 prompts of
`scripts/oracle/llama_logits.py`, plus llama.cpp's greedy 64 tokens after the gen prompt. `match`: mean
KL(axon ‖ llama.cpp) ≤ 0.02 and every top-1 disagreement is a near-tie (llama.cpp's own top-2 within 0.1
nats). A refusal must exit non-zero with exactly one `axon: ...` line under 400 characters naming what is
unsupported, with no panic, in both `gen` and `logits`.

The KL gate (0.02 on 16 prompts) is tighter than llama.cpp's own CPU-vs-CUDA agreement on gemma3 (KL 3.3e-2 and
3.4e-2 on the two gemma3 files below). By this gate llama.cpp's CUDA build would itself be `drift` there.
Treat a `drift` row as "compare against llama.cpp's spread" (`scripts/oracle/llama_logits.py`), not as a finding.

## Soak results (2026-10-17/18, MEASURED)

**Provenance.** The server is `axon serve` from main 54a75cb; this PR's code change only rewords refusals and
isn't exercised by the soak. The runs used three copies of the harness. All of them have the same original
verdict logic, defaults and fixed thresholds:

- 4f83, 45bd: no 503 back-off, no `summarise`, no priming.
- 4700: back-off and the identity 503 retry.
- d942: priming added.
- 9552: everything above.

None of them had the review-round additions: the slope ±2 SE verdict, RssAnon, the 503 body check, the 1500-token
slow reader with probe, cancel counting and slots-idle. **The memory columns below are re-judged from each run's
saved CSV rows with the revised logic (`soak.py rejudge LOG`)**. That is pure analysis of the same samples on
VmRSS: none of these runs sampled RssAnon.

"Requests" counts every request the clients sent, including expected 4xx (invalid/oversize, ~9 %),
self-cancelled disconnects (~7 %) and 503s; "ok" is the successful ones. "failing": every request in a failing
class. Latency is the whole request, successful requests only, as the median of the 30 s windows' p50/p99. That
understates the true p99. The p99 of about 30 s is the slow-reader requests (96 tokens drained at ~1 KB/s by the
client). "max excl. slow-reader" is the slowest successful request of any other kind.

| host | backend / model | hours | clients / slots / ctx | requests / ok | failing | 503 (fraction) | identity | RSS growth/24 h ±2 SE (bound) | VRAM growth/24 h ±2 SE (bound 16) | memory (revised rule) | lat p50 / p99 / max excl. slow-reader | TTFT p50 | fds, threads before → after | other checks | job |
|---|---|---:|---|---:|---:|---:|---|---|---|---|---|---:|---|---|---|
| avante (RTX 4050 Laptop, sm_89, 6 GB) | cuda, Qwen2.5-1.5B-Instruct Q4_K_M | 3.22 | 12 / 4 / 4096 | 79,656 / 67,270 | 0 | 0 | 14/14 | 5.4 [4.8, 5.9] (64) | 2.9 [1.4, 4.3]; one +4 MiB step at 22.5 min | NOT-JUDGED (2.86 h < 4 h); would be PASS | 313 ms / 30.3 s / 3.0 s | 28 ms | 39, 7 → 39, 7 | pass | 1017-171509-4f83 |
| avante, primed rerun | same | 3.22 | 12 / 4 / 4096 | 80,879 / 68,368 | 0 | 0 | 14/14 | −61.8 [−70.1, −53.4] (64) | 8.4 [6.0, 10.8]; same step at 27.6 min | NOT-JUDGED (2.86 h); would be PASS | 294 ms / 30.3 s / 2.9 s | 26 ms | 39, 7 → 39, 7 | pass | 1017-195822-d942 |
| bella (RTX 5070 Ti Laptop, sm_120, 12 GB) | cuda, Qwen2.5-7B-Instruct Q4_K_M | 1.12 | 16 / 4 / 4096 | 20,357 / 17,144 | 0 | 0 | 6/6 | 45.8 **[−74.3, 166.0]** (64) | 0.0 [0.0, 0.0] | NOT-JUDGED (0.76 h); **would be INCONCLUSIVE** | 2.0 s / 31.0 s / 7.9 s | 1.39 s | 39, 7 → 39, 7 | pass | 1017-171510-45bd |
| jade (i7-9750H, 6C/12T, 7 GB) | cpu, Qwen2.5-0.5B-Instruct Q4_K_M | 2.02 | 6 / 1 / 2048, `--max-queue 4` | 5,627 / 2,650 | 0 | 2,216 (39 %) | 9/9 | 137.4 [126.0, 148.8] (64) | n/a | NOT-JUDGED (1.66 h); **would be FAIL** | 12.1 s / 21.4 s / 30.2 s | 11.1 s | 5, 3 → 5, 3 | pass | 1017-175220-4700 |
| bella CPU (8 threads) | cpu, Qwen2.5-0.5B-Instruct Q4_K_M | 4.02 | 6 / 1 / 2048, `--max-queue 4` | 36,941 / 30,672 | 0 | 544 (1.5 %) | 17/17 | 10.0 [−9.2, 29.1] (64) | n/a | NOT-JUDGED (3.66 h after warm-up); would be PASS | 1.0 s / 30.7 s / 13.0 s | 0.77 s | 5, 10 → 5, 10 | pass | 1017-195600-9552 |

Under the original rule (slope × 24 h < bound, no minimum window) these were PASS, PASS, PASS, **FAIL**, PASS.

**What the revised rule says, honestly:** none of the five runs has 4 h after its warm-up, so **no run's
memory verdict is judged**. That includes bella CPU: 4.02 h minus the 20-min warm-up is 3.66 h. The
non-memory checks pass on all five: 0 failing of 223,460 requests (186,104 ok), every identity run matched, no
panic, the server stayed up, fds and threads back at baseline. What the memory data would say:

- **bella-7B is INCONCLUSIVE.** Its "PASS" at 45.8 MiB/day sat on a 1.1 h run, and ±2 SE spans −74 to +166
  MiB/day. It doesn't support a memory claim either way.
- **jade would still FAIL.** RSS rose 6 MiB in two steps (531.7 → 533.7 MiB at 72 min, → 537.9 at 81 min), then
  stayed at 537.6–537.9 for 40 min. Steps make the residuals small and autocorrelated, so the SE is tight
  (±11) and wrong in the same direction as the slope. The bella CPU run of the same binary, model and mix (6.6× the
  requests over 4 h) grows 10 MiB/day, its last quarter 1.6 MiB *below* its first, while VmRSS swings ±25 MiB
  (406–457) as the allocator takes and returns memory. Our reading is a one-off allocator step, not a leak.
  That is a reading, not a result: jade sampled only VmRSS, and it ran under memory pressure, where file-backed
  pages being reclaimed and refaulted show up in VmRSS. The RssAnon column added in the review round would
  settle it. A 4 h jade replicate (1017-220905-24da) never started: axon-node refused it because MemAvailable
  was under its 2560 MB floor.
- **avante VRAM would pass, but the verdict depends on when the step lands.** Both 3.2 h runs show exactly one
  +4 MiB step in the server's own VRAM (at 22.5 and 27.6 min), then nothing for about 3 h. Near the start of the
  window, a step barely moves the slope. The same step at the midpoint would give ~45 MiB/day and FAIL the
  16 MiB bound. Priming every batch mix didn't remove it, so the batch graph cache (`cuda/qwen2/slots.rs`, ≤ 64
  graphs) is not the whole cause. Other candidates: the solo path's long-position decode graph (`cmb64`), or a
  driver pool growing on a first rare shape. Not identified.

**503s.** On the CPU runs (1 slot, a 4-deep queue, 6 clients and bursts of 3) they are back-pressure by design.
39 % on jade is a load the box can't keep up with, not a soak of steady service. These runs did not check
503 bodies; the revised harness does (class `503-undocumented`). The GPU runs never filled their 128-deep queue.

**Server-side cancellation.** Only the bella CPU run kept its whole serve log: 2,505 `cancelled after N tokens`
lines for 2,588 client disconnects (0.97). The rest are disconnects that came after the generation ended. The
revised harness reports this ratio for every run.

Latency, for scale: 1.5B on the 4050 laptop serves 6–7 requests/s at a p50 of about 0.3 s. A 7B on the 5070 Ti Laptop runs
16 clients at a p50 of 2.0 s. A slot holds up to ~75 % of a 4096 context (the `long` prompts).

**Not covered:** a judged memory verdict (no run has 4 h after warm-up; the next soak should run ≥ 4.5 h on jade
or a pod); RssAnon sampling, the 1500-token slow reader with its probe, the 503 body check and cancel counting
on a real soak (implemented in the review round, smoke-tested on jade CPU only: one slot, so no probe);
multi-day runs (the longest is 4 h); pod GPUs; `--kv q8`; `--draft`; a split
(`--gpu-layers`) or pipeline server; Qwen3 reasoning streams; `--api-key`/`--slot-idle-ms` under load (fuzzed in
#61, not soaked); thousands of idle connections (`fuzz_serve.py`); OOM and GPU-error injection (still open
under item 2).

## Model-zoo results (2026-10-18, MEASURED)

18 files, 5.7 GB downloaded plus 6 already in `models/`. Axon at this PR merged with main 45d4dce (commit
0bf1d35, which includes #72's Q2_K/Q3_K/IQ4_NL), CPU, against llama.cpp b11580 (4413a19) CPU, 16 prompts, top-20
(job 1018-053457-037d). Earlier runs: on main before the fix 1017-171548-0e4e; at this PR before #72
1017-175220-02a3, where IQ4_NL and the two Q2_K/Q3_K_M files were the expected refusals. GGUF version,
`general.quantization_version` and `tokenizer.ggml.pre` come from each file's header. TheBloke's TinyLlama (late
2023) predates the pre-tokenizer key and runs anyway. sha256 are in the PR body.

| file | source | MB | GGUF v / quant v / pre | tensor types | expect | verdict | top-1 | KL mean | gen chars = llama.cpp | refusal / note |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| qwen2.5-0.5b-q4_k_m.gguf | Qwen | 491 | 3 / 2 / qwen2 | F32:121 Q5_0:133 Q8_0:13 Q4_K:12 Q6_K:12 | run | match | 14/16 | 2.4e-03 | 143 | Qwen official, Q4_K_M |
| Qwen3-0.6B-UD-Q4_K_XL.gguf | unsloth | 405 | 3 / 2 / qwen2 | F32:113 Q4_K:110 Q5_K:30 Q6_K:37 IQ4_XS:20 | refuse | refused |  |  |  | tensor blk.3.attn_q.weight: type IQ4_XS (23) is not supported; this i-quant or ternary type is not supported: use a K-quant file (Q4_K_M, Q5_K_M, Q6_K) or Q8_0 |
| Qwen3-0.6B-Q4_1.gguf | unsloth | 409 | 3 / 2 / qwen2 | F32:113 Q4_1:196 Q6_K:1 | run | match | 16/16 | 1.2e-03 | 182 | Q4_1 |
| Qwen3-0.6B-IQ4_NL.gguf | unsloth | 382 | 3 / 2 / qwen2 | F32:113 Q6_K:1 IQ4_NL:196 | run | match | 16/16 | 4.9e-04 | 87 | IQ4_NL (supported since #72) |
| Qwen3-0.6B-BF16.gguf | unsloth | 1198 | 3 / 2 / qwen2 | F32:113 BF16:197 | refuse | refused |  |  |  | tensor token_embd.weight: type BF16 (30) is not supported; use the F16 or Q8_0 file |
| Qwen2.5-0.5B-Instruct-Q6_K.gguf | lmstudio-community | 506 | 3 / 2 / qwen2 | F32:121 Q8_0:145 Q6_K:24 | run | match | 16/16 | 2.9e-03 | 72 | Q6_K |
| Qwen2.5-0.5B-Instruct.i1-Q4_K_M.gguf | mradermacher | 398 | 3 / 2 / qwen2 | F32:121 Q5_0:132 Q8_0:13 Q4_K:12 Q6_K:12 | run | match | 16/16 | 5.0e-03 | 348 | imatrix Q4_K_M |
| Qwen2.5-0.5B-Instruct.i1-IQ2_M.gguf | mradermacher | 329 | 3 / 2 / qwen2 | F32:121 Q5_0:24 Q8_0:1 IQ4_NL:120 IQ3_S:3 IQ2_S:21 | refuse | refused |  |  |  | tensor blk.0.ffn_down.weight: type IQ3_S (21) is not supported (the file has unsupported tensors: 3 IQ3_S, 21 IQ2_S); this i-quant or ternary type is not supported: use a K-quant file (Q4_K_M, Q5_K_M, Q6_K) or Q8_0 |
| Qwen2.5-0.5B-Instruct.i1-Q4_0_4_4.gguf | mradermacher | 352 | 3 / 2 / qwen2 | F32:121 Q8_0:1 Q4_0_4_4:168 | refuse | refused |  |  |  | tensor blk.0.attn_q.weight: type Q4_0_4_4 (31) is not supported; it is an ARM repacked layout that llama.cpp has dropped: use the Q4_0 file |
| SmolLM2-360M-Instruct-Q8_0.gguf | bartowski | 386 | 3 / 2 / smollm | F32:65 Q8_0:225 | refuse | refused |  |  |  | pre-tokenizer smollm is not supported (qwen2, llama-bpe, gpt-4o) |
| tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | TheBloke | 669 | 3 / 2 / - | F32:45 Q4_K:135 Q6_K:21 | run | match | 16/16 | 1.4e-03 | 279 | 2023 converter (TheBloke) |
| gpt2.Q8_0.gguf | QuantFactory | 178 | 3 / 2 / gpt-2 | F32:99 Q8_0:50 | refuse | refused |  |  |  | architecture 'gpt2' is not supported (qwen2, qwen3, llama, gemma3, qwen3moe, gpt-oss) |
| gemma-3-1b-it-Q4_K_M.gguf | ggml-org | 806 | 3 / 2 / default | F32:157 Q5_0:117 Q8_0:14 Q4_K:39 Q6_K:13 | run | match | 16/16 | 6.4e-03 | 334 | ggml-org gemma3 |
| google_gemma-3-1b-it-Q4_K_M.gguf | bartowski | 806 | 3 / 2 / default | F32:157 Q5_0:117 Q8_0:14 Q4_K:39 Q6_K:13 | run | drift | 15/16 | 1.5e-02 | 350 | bartowski gemma3 |
| Llama-3.2-1B-Instruct-Q4_K_M.gguf | bartowski | 808 | 3 / 2 / llama-bpe | F32:34 Q4_K:96 Q6_K:17 | run | match | 16/16 | 3.3e-03 | 130 | llama3.2 |
| Qwen2.5-0.5B-Instruct-Q2_K.gguf | bartowski | 339 | 3 / 2 / qwen2 | F32:121 Q5_0:24 Q8_0:1 Q3_K:24 IQ4_NL:120 | run | match | 16/16 | 2.1e-03 | 121 | Q2_K file: Q3_K + IQ4_NL fallback tensors (896 columns aren't whole 256-blocks); since #72 |
| Qwen2.5-0.5B-Instruct-Q3_K_M.gguf | bartowski | 355 | 3 / 2 / qwen2 | F32:121 Q5_0:46 Q5_1:2 Q8_0:1 Q4_K:23 Q5_K:1 IQ4_NL:96 | run | match | 16/16 | -1.1e-03 | 330 | Q3_K_M (+ IQ4_NL fallback tensors); since #72 |
| Qwen3-0.6B-Q8_0.gguf | Qwen | 639 | 3 / 2 / qwen2 | F32:113 Q8_0:197 | run | match | 15/16 | 3.3e-03 | 326 | Qwen official Qwen3 Q8_0 |

**Verdicts (MEASURED, 0bf1d35):**
- **11 `match`.** Since #72 this includes unsloth's IQ4_NL (16/16, KL 5e-4) and both bartowski Q2_K/Q3_K_M
  files (16/16 each). Those two hold 120 IQ4_NL tensors alongside their K-quants (896 columns aren't whole
  256-blocks), and the "Q2_K" file has no Q2_K tensors at all, only Q3_K and IQ4_NL.
- **1 `drift`**, within llama.cpp's own spread (below).
- **6 clean refusals.**
- **0 bugs.**

A KL mean slightly below 0 (Q3_K_M, −1.1e-3) is the estimator: KL over axon's top 20, with llama.cpp's logprob
floored at its own 20th. That can undershoot when the two top-20 sets differ, so read it as ≈ 0.

**The gemma3 `drift` row, against llama.cpp's own CPU-vs-CUDA spread** (50 prompts of
`scripts/oracle/llama_logits.py`, bella, same token ids, job 1017-175220-02a3):

| file | pair | top-1 | KL mean | KL max |
|---|---|---:|---:|---:|
| bartowski google_gemma-3-1b-it-Q4_K_M | llama CPU vs **axon** | 48/50 | 1.29e-2 | 0.217 |
| | llama CPU vs llama CUDA | 48/50 | 3.31e-2 | 0.815 |
| ggml-org gemma-3-1b-it-Q4_K_M | llama CPU vs **axon** | 49/50 | 1.60e-2 | 0.198 |
| | llama CPU vs llama CUDA | 48/50 | 3.44e-2 | 0.479 |
| Qwen qwen2.5-0.5b-instruct-q4_k_m | llama CPU vs **axon** | 46/50 | 2.80e-3 | 0.038 |
| | llama CPU vs llama CUDA | 43/50 | 4.27e-3 | 0.052 |

On every file, Axon is closer to llama.cpp's CPU than llama.cpp's own CUDA build is. The bartowski gemma3 row is
`drift` only by the zoo's fixed 0.02/0.1-nat gate on 16 prompts (one top-1 flip, at a 0.17-nat gap). Against
llama.cpp's spread it is a match, and the zoo gate is tighter than llama.cpp's own CPU/CUDA agreement on gemma3.

**What the zoo found:** before this PR, every i-quant or repacked file was refused as `type ?? (20) is not
supported`, and an "IQ2_M" file's first unsupported tensor was IQ4_NL, which a user can't connect to the file's
name. That covered 6 of the 18 files at the time (both IQ files, UD-Q4_K_XL, Q4_0_4_4, and the two Q2_K/Q3_K files;
since #72, IQ4_NL and the Q2_K/Q3_K files run). The refusal
now names the type, lists every unsupported type in a mixed file, and says which file to get instead (the log
entry has the before/after). Clean refusals left as they are: SmolLM2's `smollm` pre-tokenizer (a llama-arch
file Axon could run once that regex is added) and gpt2's architecture.
