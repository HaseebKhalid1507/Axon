# Batched speculative decoding in `axon serve` (r14 stage 1)  (PR #90; kill rule PENDING)

Branch `feat/serve-batched-draft`, spec `specs/r14-batched-spec.md` (stage 1: greedy, plain drafter pool,
goodput-gated K). Status: **kill rule PENDING** — the bella timing job 1018-101159-4afd is queued (bella in
day mode). GO / NO-GO is decided from it; nothing below claims a speedup yet.

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

PENDING (job 1018-101159-4afd).

## Gates so far

| gate | job | result |
|---|---|---|
| new tests (pool multi-slot rows, serve drafted == plain, batched == solo, gate units), e0d36b2 | 1018-100700-e959 (avante) | PASS 8/0 |
| fast tier debug-assertions (CPU), 545ac02 | 1018-102219-c41c | PASS 579/0 |
| fast tier release, 545ac02 | 1018-102219-57d9 | queued |
| full tier rel + dbg, md5 gen cuda/cpu vs main 7f42c02 | see PR body | submitted |

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
