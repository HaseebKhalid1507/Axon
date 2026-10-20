# Hybrid step: the prompt chunk and the decode rows in one pass  (PR #TBD)

- Date: 2026-10-20. Branch: feat/hybrid-step. Base: main aecc49d, then ab5568c merged in (#117). Status: open (draft)
- Pick / design: Fable's 10-19 next-steps review, §3 item 1 (b); research-sched.md P1 (Sarathi piggybacking).
  P2 (load-adaptive budget, shortest-first admission, MAX_PREFILLING ~C/16) is left for a later PR.

## What changed
- While requests decode, a serve step is now **one pass** of their decode rows and the prompts' chunks
  (`BatchLm::step_fused`, `CudaQwen2::step_fused`). Before, each step was a chunk's own pass (`feed_slot`) and then
  the decode step: two reads of every weight.
- `src/cuda/qwen2/batch.rs`: `Hybrid` (in `BatchBufs::hyb`, read by `pass_hooked` on `Place::Rows`) lists the
  pass's flash runs (one `attn_fa_b` launch per chunk, the launch `feed_slot` gives it: `attn_flash`, factored out
  of the one-sequence path) and `head0`, so the head runs only on the decode rows. Per-query attention takes the
  rows after the runs (decode rows, and chunks of 32 tokens or fewer).
- `src/cuda/qwen2/slots.rs`: `step_fused` does the checks and page reservations and runs without a CUDA graph
  (its shape changes every step; decode-only steps keep their graphs). It falls back to `feed_slot` + `step_rows`
  when there are no rows, or on MoE / MLA layers.
- `src/serve/engine.rs`: `plan_fused` / `join_fused`. `--prefill-chunk N` (default 256) is the hybrid pass's
  tokens **in all, the rows included**, rounded up to whole 64-token tiles (N < 64: N prompt tokens beside the
  rows). The order is as before (shortest remaining first, the starved first). A prompt's last chunk carries its
  first decode row, so TTFT doesn't lose a step.
- **Kill switch `AXON_HYBRID=0`** (`Opts::hybrid`, `ChunkPolicy::hybrid`) runs the two-pass code of before,
  untouched; `--prefill-ms` applies only there. CLI help, README and the `engine.rs` header say so.

## Why
- H100 showdown (10-19): about 25 % of mix C=128 wall time was neither prefill nor decode kernel time, and the
  structural cause is two passes per step (Fable 10-19 §2.2).
- Step 0 (the kill test, `pool_hybrid_cost`, interleaved per-rep A/B). MEASURED on bella, 7B Q4_K_M, job
  1020-130047-1741, foreign GPU load 0 %. Saving of one fused pass of B tokens vs the decode step + a chunk of
  B − nb, as min..max over nb 4/16/32/64:

| budget B | saved % (median per nb) | worst rep |
|---|---|---|
| 128 | 19.6 .. 32.9 | 17.3 |
| 256 | 11.0 .. 22.9 | 9.6 |
| 512 | 9.4 .. 14.6 | 8.3 |

  Against today's real step (decode + a chunk of B, and the extra tile it pays), per unit (prefill token or
  decode row) at nb 16: 256: 251.8 → 206.1 µs (−18 %); 512: 213.6 → 189.7 µs (−11 %). avante's 0.5B / 1.5B
  (jobs ecfa / 087c) agree.
- **The tile cliff** (same run): a pass that crosses a 64-token GEMM tile pays the whole tile. feed(132) costs 1.223
  vs 0.826 per token at 128 (gemm_cost). So fusing a 256-token chunk with 4 rows (260 tokens) gives the gain back
  ("same chunk": −1.1..+1.4 %). Hence the budget counts the rows and stays on a tile boundary.

## How it was verified
| gate | result | job ID |
|---|---|---|
| bits: `pool_hybrid_steps_match_feed_whole` (new; random chunks 16..512, 1..7 decode rows greedy/sampled/host-selected, two prompts in a pass, KV bytes + tokens == fed whole, f16/q8 × plain/paged/shuffled × qwen2/qwen3) | pass | in fast-rel and serve-slots-dbg below |
| `serve_chunked_prefill_matches_solo` with hybrid on (budget 16 and 256, f16/q8) and off | pass | same |
| fast tier, release (c2bd9f6) | 626 passed / 0 failed | 1020-142608-c863 (avante) |
| debug-assertion suite, CPU, 3 shards | 215 / 243 / 186 passed, 0 failed | 1020-142608-c50d |
| serve:: + slots::, debug-assertion, GPU | 132 / 0 | 1020-142608-359e (bella) |
| serve:: release with AXON_HYBRID=0 | 121 / 0 | 1020-142609-e6d2 (bella) |
| md5 of `gen`, main ab5568c vs c2bd9f6, same host | 0 of 20 pairs differ | 1020-142609-36c1 (bella) |
| serve identity A/B (main, cand hybrid on, cand hybrid off; 2 runs each; 4 decoders + 4 long prompts of ~1k–6k tokens arriving while they decode, then a cache-reuse wave) | 1 digest over 6 runs, 0.5B and 1.5B | 1020-142615-cbd6 (bella) |
| clippy -D warnings (lib+bin, tests), rustfmt | clean | local, 1ba683e |

After c2bd9f6 the source changes are bench code only (`serve_hybrid_ab` in `src/serve/tests.rs`).

## Results
Pending: `serve_hybrid_ab` (interleaved; on / off at main's `--prefill-ms 60` / off at `--prefill-ms 0`):
admit of a 20k-token prompt beside 4 decoders and a C=8 closed loop. 1.5B on avante 1020-152517-9729, 7B on bella
1020-152517-f575. Gate (from the review): ITL-during p50 ≤ 1.15× the chunk pass alone; C=8 aggregate ≥ +8 % over
AXON_HYBRID=0 on bella 7B; ITL p99 not worse. Kill rule: < +8 % → this goes to experiments-rejected.md.

Smoke, 0.5B, bella, 1 round, admit 4k (job 1020-143100-be68; not a timing run, not evidence): ITL during p50
8.7 ms (on) vs 13.6 (off) / 10.5 (off-ms0); p99 10.7 vs 55.0 / 14.3; C=8 loop 2705 vs 2692 tok/s (+0.5 %), ITL p99
8.0 vs 13.9 ms.

## What didn't work
- Step 0 v1 harness (separately timed medians): the two tainted 7B runs disagreed by ~22 points at chunk 256.
  The interleaved per-rep harness (v2) fixed it.
- `serve_hybrid_ab` v1 sized the admit's context as long + 2048. The "20k" prompt is 23.9k tokens by the tokenizer,
  so the server returned 400 (job cfdc). It is now sized from the tokenizer's count.
