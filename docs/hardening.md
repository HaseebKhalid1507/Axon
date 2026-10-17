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

RESULTS
