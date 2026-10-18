# `chat` forgets in halves  (PR #75)

- Date: 2026-10-18. Branch: feat/chat-halves. Base: main 5a73f63. Status: open
- Pick / design: roadmap Scale item 8, second half ("`chat` forgetting in halves").

## What changed
- `src/chat.rs`: `Chat::forget_to` (default `chat::FORGET_TO` = 0.5). In `reply()`, when the context is full,
  the fewest oldest exchanges that make room are found as before (`k_min`). With `forget_to > 0`, more of the
  oldest exchanges go, until what's kept plus the new message fills at most `forget_to × ctx`. The newest
  exchange is never taken to reach the target; only `k_min` can take it, as before. Exchanges are only ever
  dropped whole, through the same `drop_oldest`, so Mistral's re-framing (the system message moves into the new
  first exchange) applies unchanged. `forget_to = 0` gives the old behaviour exactly.
- `src/main.rs`: `axon chat --forget-to F`, a fraction from 0 to 1, or `min` (= 0). It's in `axon --help`, the
  chat usage line and README.
- Tests: `full_context_drops_oldest` now runs at ctx 160 and 320 with forget_to 0 / 0.5 / 0.25 over 16
  exchanges. Every reply equals a fresh greedy pass over its history, and after each forgetting turn the cache
  image (`kv_save` of positions 0..fed) equals a fresh prefill of the kept history, byte for byte. When a target
  is set, what's kept fits it. At 320, half forgets on fewer turns and prefills fewer tokens than min. New
  `session_after_forgetting`: a session saved right after a halving (the file is rewritten, since its prefix
  changed) loads in a new `Chat` with the same fed, tokens and cache bytes, and 12 exchanges reloaded at every
  halving give the uninterrupted conversation's replies. `mistral_framing_and_replies`' full-context loop runs
  with forget_to 0 and 0.5 over 8 exchanges, and also checks each reply against a fresh pass. New ignored
  benches: `forget_bench` (0.5B) and `forget_bench_7b`.

## Why
Forgetting shifts every position after the system block, so the whole kept history is prefilled again. With the
minimum dropped, the window is full again right after, so every turn after the window first fills pays a
near-full re-prefill. The roadmap's figure is ~24k tokens per exchange at a 32k window. Forgetting down to half
pays one re-prefill of ~half the window, and the next several turns append to a warm cache.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| release suite (3 shards, 4090 pods) | **PASS** 514 passed / 0 failed / 37 ignored. Shard 2 hit the CUDA ILLEGAL_ADDRESS cascade on pod1, starting in `gpu_prefill_batchings_bit_identical_q8_0_5b` (code untouched here); axon-run's retry of those 28 tests passed 28/28. All `chat::` tests pass, new and extended ones included | group 1018-003131-4eea (f872, 73e4, 90c3) |
| debug-assertion suite (4090 pod) | pending (running) | 1018-003316-4224 |
| md5 `gen` main 5a73f63 vs candidate, bella, cuda + cpu, 5 models × 3 prompts | pending (bella queue) | 1018-003503-003d |
| chat md5 on bella, cuda + cpu, 0.5B `--ctx 512 --temp 0`, 16 messages: main vs `--forget-to min` (must match) | pending (bella queue) | 1018-003503-003d |
| clippy -D warnings (bin + tests) | clean | local |
| rustfmt --check | clean | local |

## Results
Methodology: `chat::tests::forget_bench` / `forget_bench_7b`, ignored tests added in this PR. Each is a 40-exchange scripted greedy chat with the default Qwen system message (10 topics cycled, each message numbered). forget_to 0 (main's behaviour, through the same code path) and 0.5 run interleaved, 2 rounds, in one process on one GPU. Per turn the bench records `Reply::prompt_tokens`, the prefill time, and the TTFT, timed from the `reply()` call to the first emitted token. "Turns per re-prefill" = turns from the first forgetting turn to the end ÷ the number of forgetting turns. The counts are deterministic: both rounds gave the same ones.

| run | host | model file | ctx / -n | config | forgetting turns (of 40) | turns per re-prefill | prefilled tok, 40 turns | TTFT mean / p95, all turns (ms) | TTFT mean / p95 after the window filled (ms) | job |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B | 4090 pod2, **shared** (packed) | qwen2.5-0.5b-q4_k_m.gguf | 2048 / 160 | min (main) | 24 (first at turn 13) | 1.17 | 42,488 | 24.5 / 39.7 · 22.6 / 37.7 | 31.7 / 40.9 · 29.6 / 39.2 | 1018-003138-4b40 |
| 0.5B | 4090 pod2, **shared** (packed) | qwen2.5-0.5b-q4_k_m.gguf | 2048 / 160 | half | 5 (first at turn 13) | 5.60 | 5,640 (**7.5× fewer**) | 6.9 / 10.5 · 5.8 / 10.5 | 7.6 / 14.9 · 5.1 / 10.6 | 1018-003138-4b40 |
| 0.5B | 4090 pod, exclusive | qwen2.5-0.5b-q4_k_m.gguf | 2048 / 160 | min / half | pending | | | | | 1018-003140-0097 |
| 7B | 4090 pod, exclusive | Qwen2.5-7B-Instruct-Q4_K_M.gguf | 8192 / 400 | min / half | pending | | | | | 1018-012407-85c0 |
| 7B | bella, exclusive | Qwen2.5-7B-Instruct-Q4_K_M.gguf | 8192 / 400 | min / half | pending | | | | | 1018-003141-6ad6 |

All MEASURED; the TTFT cells give both rounds (r0 · r1). The 0.5B timings come from a packed pod, so treat them as a relative A/B (interleaved in one process); the exclusive reruns are pending. The token counts don't depend on the host.

Commands: `axon-run --vram 2500 -- axon-test chat::tests::forget_bench --exact --ignored --nocapture`; the same with `--timing 4090`; `axon-run --timing bella|4090 --big-models --vram 9000 -- axon-test chat::tests::forget_bench_7b --exact --ignored --nocapture`.

At a 32k window the same arithmetic applies: min re-prefills ~24k tokens (the window less the ctx/4 reply budget) on nearly every turn once the window is full, while half re-prefills at most ~16k once every several turns (ESTIMATED by the same arithmetic; the 7B rows above measure 8k).

## What didn't work / notes
- Counting the reply budget (`want`) in the target was rejected. With unlimited replies `want` is ctx/4, so
  "half" would really keep about a quarter of the window of history. The target counts what's actually kept
  (history plus the new message); `want` still decides when forgetting happens at all.
- The roadmap's item 8 status cell sits next to item 7's, which PR #67 edits, so this PR updates the paragraph
  under the table (as the roadmap asks until #67 merges), not the cell.
