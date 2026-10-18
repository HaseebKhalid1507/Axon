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
| invalid | 7 | truncated JSON, bad UTF-8, `temperature -1`, `n 3`, `max_tokens -5`, `logit_bias`, a 404 route |
| long | 6 | a prompt of 35–55 % of the context |
| slow-reader | 4 | a stream read 64 bytes at a time with 20–100 ms pauses through a 4 KiB `SO_RCVBUF` |
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
`timeout`, `other-status`.

**Greedy identity.** Four fixed greedy requests (two chats, a forced tool call and a third-of-the-context prompt)
run before the load, every 15 min under full load, and after it. Every run must return the same text, tool
calls and token counts as the first. Under load this also checks batch invariance: the reference ran alone, the
spot checks run next to 12 other clients.

**Verdict** (`DIR/summary.json`, exit 0 = PASS):

- the server is alive at the end, and its log has no `panicked`;
- no request in a failing class; 503 only as back-pressure (`too many queued requests` / `too many connections`);
- every identity run matches;
- flat memory: a least-squares line through RSS against time, over the samples after a 20-minute warm-up, its
  slope projected over 24 h must be under `max(64 MiB, 2 % of the median RSS)`. VRAM (per pid): under 16 MiB
  per 24 h.
- after the load stops and keep-alives time out (35 s), fds and threads are back at the pre-load baseline
  (+2 fds of slack).

Why that memory bound: the soak's job is to catch a leak that matters for a server left up for days. A leak
of 1 KiB per request at the soak's rate (~1–5 requests/s on a GPU) is 85–420 MiB a day, well over the bound,
so it fails; the allocator's own drift (glibc arenas settling, a few MiB in the first minutes, which the
warm-up excludes) stays far under it. 2 % of RSS keeps the bound meaningful for a 7B on the CPU (GBs of RSS)
without failing a 0.5B for its first-touch pages. VRAM is allocated up front (weights, KV slots, scratch), so
any growth at all is a bug; 16 MiB/day is measurement slack.

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

## Soak results (2026-10-17/18, MEASURED)

Harness `scripts/soak.py` at this PR. Server: `axon serve` from main 54a75cb (the soak doesn't exercise this PR's
code change, which only rewords refusals). "failing": every request in a failing class (5xx, unexpected
4xx/200, dropped, bad body, timeout). Latency is the whole request, successful requests only, as the median of
the 30 s windows' p50/p99. The p99 of about 30 s is the slow-reader requests, which take about 30 s by
construction. The "max excl. slow-reader" column is the slowest successful request of any other kind.

| host | backend / model | hours | clients / slots / ctx | requests | failing | 503 | identity | RSS slope (24 h, bound) | VRAM (24 h, bound 16) | lat p50 / p99 / max excl. slow-reader | TTFT p50 | fds, threads before → after | verdict | job |
|---|---|---:|---|---:|---:|---:|---|---|---|---|---:|---|---|---|
| avante (RTX 4050 Laptop, sm_89, 6 GB) | cuda, Qwen2.5-1.5B-Instruct Q4_K_M | 3.22 | 12 / 4 / 4096 | 79,656 | 0 | 0 | 14/14 | 0.23 MiB/h (5.4, 64) | 2168 → 2172 at 22.5 min, then flat (2.9) | 313 ms / 30.3 s / 3.0 s | 28 ms | 39, 7 → 39, 7 | **PASS** | 1017-171509-4f83 |
| avante, primed rerun | same | 3.22 | 12 / 4 / 4096 | 80,879 | 0 | 0 | 14/14 | −2.57 MiB/h (−62, 64) | 2168 → 2172 at 27.6 min, then flat (8.4) | 294 ms / 30.3 s / 2.9 s | 26 ms | 39, 7 → 39, 7 | **PASS** | 1017-195822-d942 |
| bella (RTX 5090, sm_120, 12 GB) | cuda, Qwen2.5-7B-Instruct Q4_K_M | 1.12 | 16 / 4 / 4096 | 20,357 | 0 | 0 | 6/6 | 1.91 MiB/h (45.8, 64) | 7180 flat (0.0) | 2.0 s / 31.0 s / 7.9 s | 1.39 s | 39, 7 → 39, 7 | **PASS** | 1017-171510-45bd |
| jade (i7-9750H, 6C/12T, 7 GB) | cpu, Qwen2.5-0.5B-Instruct Q4_K_M | 2.02 | 6 / 1 / 2048, `--max-queue 4` | 5,627 | 0 | 2,208 | 9/9 | **5.73 MiB/h (137, 64)** | n/a | 12.1 s / 21.4 s / 30.2 s | 11.1 s | 5, 3 → 5, 3 | **FAIL** (memory slope only) | 1017-175220-4700 |
| bella CPU (8 threads) | cpu, Qwen2.5-0.5B-Instruct Q4_K_M | 4.02 | 6 / 1 / 2048, `--max-queue 4` | 36,941 | 0 | 544 | 17/17 | 0.42 MiB/h (10.0, 64) | n/a | 1.0 s / 30.7 s / 13.0 s | 0.77 s | 5, 10 → 5, 10 | **PASS** | 1017-195600-9552 |

The 503s are the documented back-pressure: one CPU slot plus a 4-deep queue, 6 clients and bursts of 3. Every
503 carried `too many queued requests`, and clients back off 1–3 s. A GPU run never filled its 128-deep queue.

**Reading the two FAIL-shaped results:**

- **jade, RSS: FAIL by the criterion.** Its RSS sat at 529.8–531.9 MiB for the first 80 min. It stepped to 537.9
  MiB at 81 min, then stayed at 537.6–537.9 for the last 40 min. A least-squares line through a single
  late step over a 1.7 h window projects to 137 MiB/day, so the run fails. The bella CPU run of the same binary,
  model and mix (4.02 h, 36,941 requests, 6.6× jade's) has a slope of 0.42 MiB/h (10 MiB/day), with
  the last quarter 1.6 MiB *below* the first. Its RSS also moves in ±25 MiB swings (406–457 MiB) as the
  allocator takes and returns memory, so an 8 MiB step is well inside what glibc does on its own. Our reading:
  jade's step is a one-off allocator step, not a leak. The verdict stays FAIL: the criterion was set before the
  data, and a short run can't tell a step from growth. A 4 h jade replicate (1017-220905-24da) never started:
  axon-node refused it because MemAvailable was under its 2560 MB floor (other jobs were using jade's RAM).
- **avante, VRAM: passes, with an unexplained one-time step.** Both 3.2 h runs show exactly one +4 MiB step in
  the server's own VRAM, at 22.5 min and at 27.6 min, then nothing for about 3 h. Priming every batch mix first
  didn't remove it, so the batch graph cache (`cuda/qwen2/slots.rs`, bounded at 64 graphs) is not the whole
  cause. Other candidates are the solo path's decode graph for long positions (`cmb64`, captured on first
  use), or the driver growing a pool on a first rare shape. Two flat 3 h tails don't look like a leak. The
  slope check passes in both runs (2.9 and 8.4 MiB/day against 16). Not identified.

Latency, for scale: 1.5B on the 4050 laptop serves 6–7 requests/s at a p50 of about 0.3 s. A 7B on the 5090 runs
16 clients at a p50 of 2.0 s. A slot holds up to ~75 % of a 4096 context (the `long` prompts).

**Not covered:** multi-day runs (the longest is 4 h); pod GPUs; `--kv q8`; `--draft`; a split
(`--gpu-layers`) or pipeline server; Qwen3 reasoning streams; `--api-key`/`--slot-idle-ms` under load (fuzzed in
#61, not soaked); thousands of idle connections (`fuzz_serve.py`); OOM and GPU-error injection (still open
under item 2).

## Model-zoo results (2026-10-18, MEASURED)

18 files, 5.7 GB downloaded plus 6 already in `models/`. Axon at this PR, CPU, against llama.cpp b11580
(4413a19) CPU, 16 prompts, top-20 (job 1017-175220-02a3; the run on main before the fix: 1017-171548-0e4e).
GGUF version, `general.quantization_version` and `tokenizer.ggml.pre` come from each file's header.
TheBloke's TinyLlama (late 2023) predates the pre-tokenizer key and runs anyway. sha256 are in the PR body.

| file | source | MB | GGUF v / quant v / pre | tensor types | expect | verdict | top-1 | KL mean | gen chars = llama.cpp | refusal / note |
|---|---|---:|---|---|---|---|---:|---:|---:|---|
| qwen2.5-0.5b-q4_k_m.gguf | Qwen | 491 | 3 / 2 / qwen2 | F32:121 Q5_0:133 Q8_0:13 Q4_K:12 Q6_K:12 | run | match | 14/16 | 2.4e-03 | 143 | Qwen official, Q4_K_M |
| Qwen3-0.6B-UD-Q4_K_XL.gguf | unsloth | 405 | 3 / 2 / qwen2 | F32:113 Q4_K:110 Q5_K:30 Q6_K:37 IQ4_XS:20 | refuse | refused |  |  |  | tensor blk.3.attn_q.weight: type IQ4_XS (23) is not supported; i-quants (IQ*) and ternary (TQ*) types are not supported: use a K-quant file (Q4_K_M, Q5_K_M, Q6_K) or Q8_0 |
| Qwen3-0.6B-Q4_1.gguf | unsloth | 409 | 3 / 2 / qwen2 | F32:113 Q4_1:196 Q6_K:1 | run | match | 16/16 | 1.2e-03 | 182 | Q4_1 |
| Qwen3-0.6B-IQ4_NL.gguf | unsloth | 382 | 3 / 2 / qwen2 | F32:113 Q6_K:1 IQ4_NL:196 | refuse | refused |  |  |  | tensor blk.0.attn_q.weight: type IQ4_NL (20) is not supported; i-quants (IQ*) and ternary (TQ*) types are not supported: use a K-quant file (Q4_K_M, Q5_K_M, Q6_K) or Q8_0 |
| Qwen3-0.6B-BF16.gguf | unsloth | 1198 | 3 / 2 / qwen2 | F32:113 BF16:197 | refuse | refused |  |  |  | tensor token_embd.weight: type BF16 (30) is not supported; use the F16 or Q8_0 file |
| Qwen2.5-0.5B-Instruct-Q6_K.gguf | lmstudio-community | 506 | 3 / 2 / qwen2 | F32:121 Q8_0:145 Q6_K:24 | run | match | 16/16 | 2.9e-03 | 72 | Q6_K |
| Qwen2.5-0.5B-Instruct.i1-Q4_K_M.gguf | mradermacher | 398 | 3 / 2 / qwen2 | F32:121 Q5_0:132 Q8_0:13 Q4_K:12 Q6_K:12 | run | match | 16/16 | 5.0e-03 | 348 | imatrix Q4_K_M |
| Qwen2.5-0.5B-Instruct.i1-IQ2_M.gguf | mradermacher | 329 | 3 / 2 / qwen2 | F32:121 Q5_0:24 Q8_0:1 IQ4_NL:120 IQ3_S:3 IQ2_S:21 | refuse | refused |  |  |  | tensor blk.0.attn_q.weight: type IQ4_NL (20) is not supported (the file has unsupported tensors: 3 IQ3_S, 120 IQ4_NL, 21 IQ2_S); i-quants (IQ*) and ternary (TQ*) types are not supported: use a K-quant file (Q4_K_M, Q5_K_ |
| Qwen2.5-0.5B-Instruct.i1-Q4_0_4_4.gguf | mradermacher | 352 | 3 / 2 / qwen2 | F32:121 Q8_0:1 Q4_0_4_4:168 | refuse | refused |  |  |  | tensor blk.0.attn_q.weight: type Q4_0_4_4 (31) is not supported; it is an ARM repacked layout that llama.cpp has dropped: use the Q4_0 file |
| SmolLM2-360M-Instruct-Q8_0.gguf | bartowski | 386 | 3 / 2 / smollm | F32:65 Q8_0:225 | refuse | refused |  |  |  | pre-tokenizer smollm is not supported (qwen2, llama-bpe) |
| tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | TheBloke | 669 | 3 / 2 / - | F32:45 Q4_K:135 Q6_K:21 | run | match | 16/16 | 1.4e-03 | 279 | 2023 converter (TheBloke) |
| gpt2.Q8_0.gguf | QuantFactory | 178 | 3 / 2 / gpt-2 | F32:99 Q8_0:50 | refuse | refused |  |  |  | architecture 'gpt2' is not supported (qwen2, qwen3, llama, gemma3, qwen3moe) |
| gemma-3-1b-it-Q4_K_M.gguf | ggml-org | 806 | 3 / 2 / default | F32:157 Q5_0:117 Q8_0:14 Q4_K:39 Q6_K:13 | run | match | 16/16 | 6.4e-03 | 334 | ggml-org gemma3 |
| google_gemma-3-1b-it-Q4_K_M.gguf | bartowski | 806 | 3 / 2 / default | F32:157 Q5_0:117 Q8_0:14 Q4_K:39 Q6_K:13 | run | drift | 15/16 | 1.5e-02 | 350 | bartowski gemma3 |
| Llama-3.2-1B-Instruct-Q4_K_M.gguf | bartowski | 808 | 3 / 2 / llama-bpe | F32:34 Q4_K:96 Q6_K:17 | run | match | 16/16 | 3.3e-03 | 130 | llama3.2 |
| Qwen2.5-0.5B-Instruct-Q2_K.gguf | bartowski | 339 | 3 / 2 / qwen2 | F32:121 Q5_0:24 Q8_0:1 Q3_K:24 IQ4_NL:120 | refuse | refused |  |  |  | tensor blk.0.attn_q.weight: type IQ4_NL (20) is not supported (the file has unsupported tensors: 24 Q3_K, 120 IQ4_NL); i-quants (IQ*) and ternary (TQ*) types are not supported: use a K-quant file (Q4_K_M, Q5_K_M, Q6_K) o |
| Qwen2.5-0.5B-Instruct-Q3_K_M.gguf | bartowski | 355 | 3 / 2 / qwen2 | F32:121 Q5_0:46 Q5_1:2 Q8_0:1 Q4_K:23 Q5_K:1 IQ4_NL:96 | refuse | refused |  |  |  | tensor blk.0.attn_q.weight: type IQ4_NL (20) is not supported; i-quants (IQ*) and ternary (TQ*) types are not supported: use a K-quant file (Q4_K_M, Q5_K_M, Q6_K) or Q8_0 |
| Qwen3-0.6B-Q8_0.gguf | Qwen | 639 | 3 / 2 / qwen2 | F32:113 Q8_0:197 | run | match | 15/16 | 3.3e-03 | 326 | Qwen official Qwen3 Q8_0 |

**Verdicts:** 8 `match`; 1 `drift`, which is within llama.cpp's own spread (below); 9 clean refusals; 0 bugs. The
two Q2_K / Q3_K_M files are refused as expected on main, because the Q2_K/Q3_K kernels are in PR #72. These two
0.5B files would still be refused after #72: they also hold 120 IQ4_NL tensors (896 columns aren't whole
256-blocks, so llama.cpp falls back to IQ4_NL), and the "Q2_K" file holds no Q2_K tensors at all, only Q3_K and
IQ4_NL.

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
name. That covered 6 of the 18 files (both IQ files, UD-Q4_K_XL, Q4_0_4_4, and the two Q2_K/Q3_K files). The refusal
now names the type, lists every unsupported type in a mixed file, and says which file to get instead (the log
entry has the before/after). Clean refusals left as they are: SmolLM2's `smollm` pre-tokenizer (a llama-arch
file Axon could run once that regex is added) and gpt2's architecture.
