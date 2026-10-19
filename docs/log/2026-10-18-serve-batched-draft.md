# Batched speculative decoding in `axon serve` (r14 stage 1)  (PR #90; kill rule PENDING)

Branch `feat/serve-batched-draft`, spec `specs/r14-batched-spec.md` (stage 1: greedy, plain drafter pool,
goodput-gated K). Status: **kill rule PENDING.** Run 1 (job 1018-101159-4afd) lost its d600 cell to a harness bug. It
showed +18.7 % at C=4 d4k and a gate that drafts where it shouldn't at 16k. The gate is fixed in 3541ea2.
Run 2 (job 1018-204023-96fd, every cell) decides.

## What changed

- `axon serve --draft-model D` with continuous batching (`--parallel N`, GPU): the batch now drafts.
  Before, only a request running alone drafted (`generate` with a `ModelDrafter`); every batched step
  was one token per request.
- `src/serve/drafting.rs` (new): `DraftPool` — the draft model's own pool of slots beside the main
  model's, slot s mirroring slot s; per-slot `cached` tokens (LCP catch-up), per-request acceptance EMAs
  (β 1/16; a new request starts at the server's EMA, 1 before any guess was seen) and the gate.
- The gate (no baked constants): at startup, before the first request, steps of both pools are timed at
  row counts {1, C, 2C, 3C, 4C, C(K+1)} (drafter {1, C, 2C}) × depths {512, 4096, 16384} clamped to the
  slot context (3 timed reps after one untimed), fitted by least squares to t(nb, Σd) = a + b·nb + c·Σd,
  then corrected online per row count by every measured step (EMA of measured/predicted, β 1/8). Per
  step it scores Kmax ∈ 0..=K: K_i = min(Kmax, cap_i), the lowest-α requests' last guesses dropped while
  α^K_i is worth less than the row's cost at the round's rate, goodput = Σ E_i / (draft steps + verify
  step); Kmax = 0 is the plain step, unchanged. `AXON_DRAFT_CALIB=0` skips calibration (K always).
  Every round's step shapes are captured as CUDA graphs at startup too.
- A round (`Batch::round`): max K_i drafter steps (the first also catches each request's drafter slot
  up: the main model's correction token, or the last guess of a fully accepted round; a bigger gap is a
  feed), one main-model step with K_i + 1 rows per request at their positions, then the host walks the
  picks exactly as `generate` does (stop ids, stop strings, `max_tokens`, cancellation per token).
- Prefill chunks are mirrored into the drafter's slot in the same step (inside `--prefill-ms`); the
  single-request path's drafter now runs on a view of its slot of the drafter pool (a hand-over copies
  nothing on either model). The drafter pool's context is the longest that fits up to
  min(slot context, the drafter's context); past it requests draft nothing.
- `BatchLm::new_pool_rows` (additive): a pool whose steps take more rows than it has slots.
- `/metrics`: `axon_draft_rounds_total{k}`, `axon_draft_tokens_total`, `axon_draft_accepted_total`,
  `axon_draft_alpha`, `axon_draft_step_us_total`, `axon_draft_verify_us_total`.

## Why

r14-alpha.md MEASURED α = 0.756 (E = 2.76 tokens per pass at K=3) for 0.5B → 7B on agent-style prompts;
r13 §0's MEASURED step costs put batched drafting at +41 % (C=4, d600), lower with depth, negative at 16k
with today's verify (each verify row reads its slot's KV again). The gate is what makes it safe at every
depth: where drafting loses it must pick K=0, today's step.

## Bits

Greedy only (sampled requests take no guesses in stage 1; their rows are as before). A verify row's pick
is `verify_argmax`'s bit for bit (the pool's contract), and the walk is `generate`'s, so a drafted
request's bytes equal the plain server's. Tested: `pool_rows_of_many_slots_match_verify` (3 slots × 1/4/2
rows interleaved in one step == per-slot `verify_argmax` / `verify_cands`, KV bytes included, f16/q8,
qwen2/qwen3), `serve_batched_draft_matches_plain` (1.5B + 0.5B, 4 slots, 16-token chunks, 13 requests
twice — prefix reuse —, stop strings, caps mid-round, a cancelled stream, sampled requests mixed in:
bodies, token counts and finish reasons equal the server without a draft model; gate forced and
calibrated), and the bench's md5 of every reply off / on / main.

## How it is measured (kill rule, spec §5)

One exclusive bella job (1018-101159-4afd, binary at 545ac02, harness `bench_spec.py` + `analyze.py` in the
builder's scratch): serve Qwen2.5-7B-Instruct Q4_K_M + qwen2.5-0.5b Q4_K_M, draft off / on (gate on, K=3) /
main 953b886, interleaved per rep, 3 reps, medians, GPU temperature and clocks logged per run. Cells: depth
~600 (the prompts as they are; --ctx 2048 --parallel 8), ~4k and ~16k (a synthetic per-client session
history before the system prompt, warmed once untimed so it sits in the client's slot, as an agent's
history does; padding length measured against the real tokenizer at startup). C=4 and C=8 clients (16k:
C=4 only, 8 slots of 16k don't fit in 12 GB), each sending its share of r14-alpha's 40 synthetic agent
prompts back to back, greedy, max_tokens 128. Per run: aggregate tok/s (completion tokens / wall), per-request
ITL = server decode_ms / tokens (p50, p99 over requests), raw inter-chunk gap p99 (reported, bounded
separately at ≤ 2.5× plain, not a kill criterion), the md5 of every reply (must agree off / on / main).

Kill if on vs off < +10 % aggregate at C=4 d600, or per-request ITL p99 regresses > 15 %, or the gate picks
K>0 somewhere and loses by > 3 %. 16k is reported; the gate must pick K=0 there.

## Measurements

### Run 1: job 1018-101159-4afd (bella, exclusive night mode, binary 545ac02, main 953b886)

**MEASURED.** Medians of 3 reps (`analyze.py`), Qwen2.5-7B-Instruct Q4_K_M + qwen2.5-0.5b Q4_K_M, greedy,
128 tokens, 40 synthetic agent prompts. GPU 72-83 C, SM 2.36-2.63 GHz, ~139 W through the run.
ITL is per request: server decode_ms / tokens. The gap column is the raw inter-chunk gap p99 across all
streams.

| cell | C | prompt tok | mode | agg tok/s | ITL p50 ms | ITL p99 ms | gap p99 ms | rounds k=0/1/2/3 (3 reps) | kept/drafted | α (end) |
|---|---|---|---|---|---|---|---|---|---|---|
| d4k | 4 | 4276 | off | 225.3 | 16.13 | 19.90 | 78.1 | | | |
| d4k | 4 | 4276 | **on** | **267.4** | 13.53 | 19.34 | 104.5 | 0/971/480/215 | 6795/9326 | 0.92 |
| d4k | 4 | 4276 | main | 225.0 | 16.23 | 19.77 | 79.3 | | | |
| d4k | 8 | 4276 | off | 316.2 | 23.11 | 29.37 | 99.8 | | | |
| d4k | 8 | 4276 | **on** | **323.4** | 22.24 | 30.20 | 131.0 | 151/445/291/107 | 6305/8719 | 0.83 |
| d4k | 8 | 4276 | main | 316.8 | 23.24 | 29.30 | 99.9 | | | |
| d16k | 4 | 16736 | off | 151.8 | 23.55 | 28.99 | 112.9 | | | |
| d16k | 4 | 16736 | **on** | **149.5** | 23.86 | **45.06** | 145.2 | 1609/1019/39/47 | 3022/3838 | 0.79 |
| d16k | 4 | 16736 | main | 151.9 | 23.54 | 29.28 | 112.7 | | | |

| cell | C | on vs off agg | ITL p99 | raw gap p99 |
|---|---|---|---|---|
| d4k | 4 | **+18.7 %** | −2.8 % | 1.34× |
| d4k | 8 | +2.3 % | +2.8 % | 1.31× |
| d16k | 4 | −1.5 % | **+55.4 %** | 1.29× |

- **The d600 cell has no data.** It was a harness bug, not a server failure. The padding calibration request
  was ~1400 tokens of padding plus a prompt, which came to 2453 tokens against that cell's --ctx 2048, so
  every d600 run errored out. The harness now calibrates with a shorter padding.
- **md5:** 120 of 120 (cell, C, prompt) replies were identical across off, on and main, in every rep. The
  greedy bytes are unchanged at 7B.
- **Off vs main:** off and main agree to within 0.3 %, so the plain path costs nothing on this branch.
- **d16k, run by run:**
  - In rep 0 the gate ran plain steps throughout: 1112 k=0 rounds, α 0.23. The −0.8 % against off is within
    the off reps' own spread (150.7-152.2).
  - In reps 1 and 2 it mostly picked K=1 (α 0.79 / 0.90) and lost 2.2 % and 0.7 %.
  - Why: from the fitted costs, the gate predicted about +16 % for K=1 at C=4 / 16k. The drafted rounds
    actually took about 1.25-1.35× what their steps predicted: the catch-up feeds after plain rounds and the
    host work between the draft steps aren't in the step fits.
  - Why rep 0 differed: the first verified guess was rejected, and the server's α was then set to that
    single observation (0). After that, no request ever drafted, so nothing could lift it.
- **d16k ITL p99 (+55 %):** this is not one stalled verify; the raw gap p99 is only 1.29×, under the 2.5×
  bound. At 16k a drafted round costs about 36-45 ms: the plain step's 23 ms, plus a drafter step, plus four
  more verify rows that each read the 16k-token KV again. A request whose guesses keep missing pays that
  round time for each single token it gets. Each request's α starts at the server's high EMA and adapts with
  β 1/16, so a low-acceptance request drafts for a while before the gate stops it. The p99 over 40 requests
  is that worst request.

### Verdict after run 1

**Not decided.** The C=4 d600 cell the rule names first has no data. Of the cells that ran:
- **d4k: passes.** C=4 is +18.7 % (bar +10 %) with ITL p99 −2.8 % (bar +15 %). C=8 is +2.3 % with ITL p99
  +2.8 %. Neither cell loses where the gate drafted.
- **d16k: the gate is wrong there.** It costs 1.5 %, under the 3 % kill clause, but the spec requires K=0 at
  16k, and in reps 1-2 the gate drafted. Per-request ITL p99 regressed 55 %.

### Gate fix (3541ea2) and run 2

- **Round time is learned.** Each drafted round's measured time is divided by its steps' predicted time.
  An EMA of that ratio, kept per most-guesses value, scales the gate's prediction. The steps' fits can't
  see the catch-up feeds and the host work between steps; this correction does. With run 1's numbers it
  turns 16k off (unit test `gate_learns_round_overhead`).
- **The server's α starts from a prior of 1.** It moves only by the EMA, so one early rejection can no
  longer pin it at 0.
- **Each request's first round always guesses once.** That is one row, about 0.05 % of a 128-token request
  at 16k. It keeps α and the round correction measured, so a gate that turned drafting off can still see
  when it pays again.

Run 2 is job 1018-204023-96fd (bella, binary 3541ea2, main 7f42c02): every cell, d600 included, with the
harness fixed. PENDING.

### Overhead with every round plain

With all rounds k=0 (rep 0 at 16k) the leftover overhead is mostly the drafter's copy of each prompt's
prefill. That is about 500 new tokens per request at depth 16k, roughly 1 % of a request's decode time
(ESTIMATED). Feeding the drafter lazily, only once the gate drafts again, would remove it, at the price of
a catch-up feed inside the first drafted round. That is a follow-up; it was not done here.

## Gates

| gate | job | result |
|---|---|---|
| new tests (pool multi-slot rows, serve drafted == plain, batched == solo, gate units), e0d36b2 | 1018-100700-e959 (avante) | PASS 8/0 |
| fast tier debug-assertions (CPU), 545ac02 | 1018-102219-c41c | PASS 579/0 |
| fast tier release, 545ac02 | 1018-102219-57d9 | PASS |
| full tier release, 73ebfff, 2 shards | 1018-135324-5a23, -7e38 (group 5177) | PASS 580/0 (40 ignored) |
| full tier debug-assertions, 73ebfff, 2 shards | 1018-135324-8bba, -1c95 (group 8fca) | PASS 580/0 (40 ignored) |
| md5 gen cuda, main 7f42c02 vs 73ebfff (5 models × 3 prompts) | 1018-135332-8711 | 15/15 identical |
| md5 gen cpu, main 7f42c02 vs 73ebfff | 1018-135332-c975 | 15/15 identical |
| new tests, 3541ea2 (gate fix) | 1018-204024-43a7 | pending |
| fast tier rel / dbg, 3541ea2 | 1018-204055-dd09 / 1018-204055-5dda | pending |
| kill rule run 2, 3541ea2 | 1018-204023-96fd | pending |

## Caveats

- Greedy only. Sampled requests take no guesses in the batch (stage 3: the lazy pick walk).
- Verify rows of one slot each read the slot's KV again (`Place::Rows`, no multi-query sharing): at deep
  contexts drafting loses and the gate's answer is K=0. Stage 2 (the mq-rows verify on #84's kernel) is
  what the agent-depth case needs.
- The drafter pool's context is what fits beside the main model's pool; the single-request path drafts on
  a view of that pool, so a long request running alone may now stop drafting earlier than before when VRAM
  is tight (speed only; output unchanged).
- The gate's cost model is linear in rows; if a kernel switch at some row count (gemvm32 / gemm2) shows,
  the per-row-count online correction absorbs it, and the calibration line on stderr shows the fit.
