# `chat` forgets in halves  (PR #TBD)

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
RESULTS_GATES

## Results
RESULTS_BENCH

## What didn't work / notes
- Counting the reply budget (`want`) in the target was rejected. With unlimited replies `want` is ctx/4, so
  "half" would really keep about a quarter of the window of history. The target counts what's actually kept
  (history plus the new message); `want` still decides when forgetting happens at all.
- The roadmap's item 8 status cell sits next to item 7's, which PR #67 edits, so this PR updates the paragraph
  under the table (as the roadmap asks until #67 merges), not the cell.
