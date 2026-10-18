# `chat` forgets in halves  (PR #75)

- Date: 2026-10-18. Branch: feat/chat-halves. Base: main 5a73f63. Status: open (gates of record green at 5cae03a)
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

### After Shady's review (5cae03a)
- **Ctrl-C during the re-prefill** used to lose everything the turn had forgotten, up to half the conversation.
  Now a stop before the reply's first token puts the forgotten exchanges back (`Chat::undo_forget`). The cache
  holds the shifted history from where it began (`fed` after the drops, at most the system block), so `fed` is
  lowered to that point and the next turn prefills from there; it forgets afresh if it has to. `r.dropped` is 0
  after an undo.
- `full_context_drops_oldest` is now generic over the backend, and `full_context_drops_oldest_cuda` runs it on
  `CudaQwen2` (ctx 320, forget_to 0 and 0.5). Before every turn, the same message is stopped at each prefill
  batch and then at its first token. Each time the conversation must come back unchanged and the cache must still
  equal a fresh prefill of `hist[..fed]`, and the test requires that at least one forgetting turn was stopped at
  its first token.
- The bench also runs with -n unlimited (the CLI default) and reports the max TTFT and the forgetting turns'
  median TTFT; a p95 over 40 turns hides 2–5 forgetting turns.
- The stats line says "forgot the oldest N exchanges (down to half the context)" (or "to fit" under min). The help
  lines are wrapped to the file's width, and help/README say that `--forget-to 1` == min and that with -n unlimited
  a reply reserves ctx/4, so 0.5 leaves a quarter of the context to grow into. A comment at the target explains
  why it leaves out the reply's room (`close`, `want`).

## Why
Forgetting shifts every position after the system block, so the whole kept history is prefilled again. With the
minimum dropped, the window is full again right after, so every turn after the window first fills pays a
near-full re-prefill. The roadmap's figure is ~24k tokens per exchange at a 32k window. Forgetting down to half
pays one re-prefill of ~half the window, and the next several turns append to a warm cache.

## How it was verified
Round 1, code 955ee0f (the binaries for this round were built before a help-text-only edit):
| gate | result | job ID / log |
|---|---|---|
| release suite (3 shards, 4090 pods) | PASS 514 / 0 / 37 ignored. Shard 2 hit the pod CUDA poison (ILLEGAL_ADDRESS cascade, first `gpu_prefill_batchings_bit_identical_q8_0_5b`); its 28 failures passed when axon-run re-ran them in a fresh process, so under the new rule that shard is **not clean evidence**. Every `chat::` test passed in the first run | group 1018-003131-4eea (f872, 73e4, 90c3); binary `axon-test-rel`, 955ee0f source minus a help-text wording edit |
| debug-assertion suite, first run (4090 pod) | 418 / 96 failed: POISON-RERUN (first `gpu_prefill_batchings_bit_identical_qwen3`). All 96 passed alone in fresh processes. Not clean evidence | 1018-003316-4224 |
| md5 `gen` main 5a73f63 vs candidate (`axon-cand`, 955ee0f = HEAD code), bella RTX 5070 Ti Laptop (sm_120), 5 models × 3 prompts, -n 64 greedy | cuda 0/15 differ, cpu 0/15 differ | 1018-003503-003d |
| chat md5, bella, 0.5B `--ctx 512 --temp 0 -n 64`, 16 messages | main == `--forget-to min` on cuda and cpu (same stdout, 11 forgetting turns each); default: 4 forgetting turns. Its stdout happened to match too: these one-line questions don't depend on the forgotten history, so the strong evidence is the unit tests | 1018-003503-003d |
| clippy -D warnings (bin + tests), rustfmt --check | clean | local |


Round 2, code 5cae03a (Shady's fixes; the gates of record):

| gate (code 5cae03a) | result | job (binary, host) |
|---|---|---|
| `chat::` release tests, `AXON_TEST_GPU_SERIAL=1` | **PASS** 21 / 0 / 2 ignored (the benches), no POISON-RERUN; includes `full_context_drops_oldest`, `full_context_drops_oldest_cuda`, `session_after_forgetting`, `mistral_framing_and_replies` | 1018-033525-7682 (`axon-test-5cae03a`, 4090 pod3) |
| debug-assertion suite (gate of record), `AXON_TEST_GPU_SERIAL=1` | **PASS** 515 / 0 / 37 ignored, no POISON-RERUN | 1018-033814-8723 (`axon-test-dbg-5cae03a`, 4090 pod4) |
| md5 `gen` main 5a73f63 vs 5cae03a, 5 models × 3 prompts, -n 64 greedy | **cuda 0/15 differ, cpu 0/15 differ** | 1018-033814-bf37 (`axon-cand-5cae03a`, bella RTX 5070 Ti Laptop sm_120) |
| chat md5, history-dependent prompts (10 × "remember favourite number #i" + "list every favourite number I told you"), 0.5B `--ctx 512 --temp 0 -n 64` | **cuda and cpu: main == `--forget-to min`** (`0f51987f8e9d`, 6 forgetting turns each); **default differs** (`c251bcf2efe5`, 3 forgetting turns) | 1018-033814-bf37 (bella) |
| clippy -D warnings (bin + tests), rustfmt --check | clean, at 5cae03a and on the merge with main ac358d4 (`git merge-tree` clean, tree 32b33be, not pushed) | local |

## Results
Methodology: `chat::tests::forget_bench` / `forget_bench_7b`, ignored tests added in this PR. Each is a 40-exchange scripted greedy chat with the default Qwen system message (10 topics cycled, each message numbered). forget_to 0 (main's behaviour, through the same code path) and 0.5 run interleaved, 2 rounds, in one process on one GPU, under exclusive placement. Per turn the bench records `Reply::prompt_tokens`, the prefill ms, and the TTFT, timed from the `reply()` call to the first emitted token. "Steady" = turns after the window first filled that didn't forget. Medians are over both rounds; the token counts were identical in both rounds. Binary: `axon-test-bench`, 955ee0f source minus a help-text wording edit (chat.rs identical to HEAD 3406e77). Per-turn rows (`FBT`) are in the job logs.

| model, host | ctx / -n | config | turns that forgot (of 40; first at) | exchanges dropped per forgetting turn, median (max) | turns per re-prefill | tokens prefilled, 40 turns | forgetting turn: prompt tok / prefill ms / TTFT ms, median | steady turn: prompt tok / TTFT ms, median (n per run) | TTFT median, all turns after the fill | job |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B `qwen2.5-0.5b-q4_k_m.gguf`, 4090 pod | 2048 / 160 | min (main) | 24 (13) | 1 (2) | 1.17 | 42,488 | 1726 / 16.2 / 16.3 | 26 / 3.7 (4) | 16.1 ms | 1018-003140-0097 |
| same | 2048 / 160 | **half** | 5 (13) | 6 (7) | 5.60 | 5,640 (**7.5× fewer**) | 954 / 8.7 / 8.8 | 25 / 3.7 (23) | **3.7 ms** | 1018-003140-0097 |
| 7B `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4090 pod | 8192 / 400 | min (main) | 19 (22) | 1 (2) | 1.00 | 141,761 | 7394 / 586.0 / 586.0 | none (0) | 586.0 ms | 1018-012407-85c0 |
| same | 8192 / 400 | **half** | 2 (22) | 10.5 (11) | 9.50 | 8,774 (**16× fewer**) | 3910 / 292.9 / 293.1 | 25 / 19.2 (17) | **19.2 ms** | 1018-012407-85c0 |
| 7B, bella (RTX 5070 Ti Laptop, sm_120) | 8192 / 400 | min (main) | 18 (22) | 1 (2) | 1.06 | 137,989 | 7646 / 2071.7 / 2071.8 | 26 / 33.9 (1) | 2070.8 ms | 1018-003141-6ad6 |
| same | 8192 / 400 | **half** | 2 (22) | 10.5 (11) | 9.50 | 8,802 (**15.7× fewer**) | 3924 / 1047.2 / 1053.6 | 25 / 32.5 (17) | **32.7 ms** | 1018-003141-6ad6 |

All MEASURED. Mean / p95 TTFT over all 40 turns (r0): 0.5B 11.5 / 17.5 → 4.3 / 8.8 ms; 7B 4090 287.7 / 601.8 → 31.3 / 20.7 ms; 7B bella 950.1 / 2083.2 → 81.7 / 38.3 ms. p95 is nearest-rank, so with half, the 2 forgetting turns of 40 sit above it; see the median column for their cost. In short: a forgetting turn under half costs about half of one under min, and it comes every 5.6–9.5 turns instead of every turn. On bella the greedy text differs slightly from the 4090's (sm_120 vs sm_89 bits, as expected), hence the slightly different token counts.

Commands: `axon-run --timing 4090 --vram 2500 -- axon-test-bench chat::tests::forget_bench --exact --ignored --nocapture`; `axon-run --timing 4090|bella --big-models --vram 9000 -- axon-test-bench chat::tests::forget_bench_7b --exact --ignored --nocapture`. Stats: `fbstats.py` over the `FBT` lines. An earlier non-exclusive 0.5B run (1018-003138-4b40, shared pod) gave the same counts.

### Round 2 bench (5cae03a, -n as given and -n unlimited)

Bench at 5cae03a, `chat::tests::forget_bench` + `forget_bench_7b` in one `--timing 4090` job (1018-033527-1588, pod4, exclusive). Setup: 40 scripted greedy exchanges; min and half interleaved, 2 rounds each, both with identical token counts. Medians are over both rounds; times are ms.

| model, ctx, -n | config | forgetting turns (of 40) | dropped per forgetting turn, median (max) | turns per re-prefill | tokens prefilled | forgetting turn: prompt tok / prefill / TTFT, median | steady turns: TTFT median (n per run) | TTFT median after the fill | TTFT max |
|---|---|---|---|---|---|---|---|---|---|
| 0.5B, 2048, 160 | min | 24 | 1 (2) | 1.17 | 42,488 | 1726 / 16.2 / 16.2 | 3.7 (4) | 16.1 | 17.6 |
| | half | 5 | 6 (7) | 5.60 | 5,640 (7.5× fewer) | 954 / 8.7 / 8.8 | 3.7 (23) | 3.7 | 8.9 |
| 0.5B, 2048, **unlimited** | min | 21 | 1 (4) | 1.81 | 21,269 | 1064 / 9.5 / 9.6 | 3.6 (17) | 4.6 | 15.3 |
| | half | 15 | 2 (5) | 2.53 | 11,277 (1.9× fewer) | 628 / 6.7 / 6.8 | 3.7 (23) | 3.7 | 15.2 |
| 7B Q4_K_M, 8192, 400 | min | 19 | 1 (2) | 1.00 | 141,761 | 7394 / 583.8 / 583.8 | none (0) | 583.8 | 616.2 |
| | half | 2 | 10.5 (11) | 9.50 | 8,774 (16× fewer) | 3910 / 292.7 / 292.9 | 19.1 (17) | 19.1 | 294.8 |
| 7B Q4_K_M, 8192, **unlimited** | min | 28 | 1 (2) | 1.04 | 169,850 | 6086 / 474.5 / 474.6 | 18.6 (1) | 474.5 | 477.8 |
| | half | 7 | 4 (5) | 4.14 | 26,020 (6.5× fewer) | 3699 / 273.4 / 273.6 | 17.3 (22) | 17.8 | 303.1 |

All MEASURED. With the CLI default (-n unlimited), a reply reserves ctx/4, so halving leaves only a quarter of the window to grow into. The measured result: 7B goes from a re-prefill on every turn (1.04) to one every 4.14 turns, with 6.5× fewer tokens prefilled (Shady estimated 4–5). 0.5B gains less, 1.81 → 2.53 and 1.9× fewer tokens (his estimate 2.7). Its greedy unlimited replies run long, ~1000 tokens, so even min forgets half the window at a time. The forgetting turns are where TTFT is paid: under half they cost about half of min's (273.6 vs 474.6 ms on 7B unlimited) and come 4× less often. The TTFT max is the same order under both configs, because a single forgetting turn sets it.

What that means at 32k (ESTIMATED from the measured 8k rows above): with -n 400, a forgetting turn under min re-prefilled a median of 7394 of 8192 positions (90 %, 7B 4090 row), and under half 3910 (48 %). At a 32k window that's ~29.5k vs ~15.6k tokens, and under min it happens on nearly every turn. With -n unlimited (the CLI default), a reply reserves ctx/4, so min keeps at most ~24k of 32k (the roadmap's figure) and re-prefills that on nearly every turn. Half keeps at most 16k, leaving ctx/4 = 8k for later turns to append into before the next forget. Measured under the defaults (-n unlimited, round 2 table): 4.14 turns per re-prefill on 7B at 8192, 2.53 on 0.5B at 2048.

## What didn't work / notes
- Counting the reply budget (`want`) in the target was rejected. With unlimited replies `want` is ctx/4, so
  "half" would really keep about a quarter of the window of history. The target counts what's actually kept
  (history plus the new message); `want` still decides when forgetting happens at all.
- The roadmap's item 8 status cell sits next to item 7's, which PR #67 edits, so this PR updates the paragraph
  under the table (as the roadmap asks until #67 merges), not the cell.
