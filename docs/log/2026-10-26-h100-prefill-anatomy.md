# H100 prefill anatomy of the mix C=128 cell, and #139's config cells (PR #157)

- Date: 2026-10-26. Branch: feat/feed-anatomy. Base: main d8a296c (rebuilt on main after #151 before booking).
  Status: open (draft). Session NOT booked: the rule below is written first.
- Pick / design: lead (Jawz) 10-26: one ~$2 H100 SXM session for the prefill anatomy of the mix C=128 serve cell and
  #139's config sweep. Background: on the H100 (1026-225032-3dfd) mix C=128 ITL p50 ≈ 57 ms against a ~15 ms pure
  decode step, so a serve step is mostly prompt feeds under `--step-ms 60`. Fable's pick
  (`specs/pick-20261026-fable.md` §(d)) estimates the prefill GEMM at ~70 % of a chunk (unverified) and names
  bucketed graph capture of feeds as the one scheduler item worth a look. The 10-26 anatomy
  (`2026-10-26-h100-anatomy-results.md`): feeds 21.9 s of a 38.0 s cell, 716 passes, t0 5.3–7.4 ms.

## What the session measures (one `--timing h100` job, ≤ 28 min)
- **(a) Per-kernel share of a feed pass** (`pool_feed_anatomy`, new, test-only, 7B, paged pool as serve):
  - for n 16–768 tokens at depth 0–6,000: the pass's wall time (`feed_slot` + sync, what a serve step pays);
  - its kernels: the same pass captured in a CUDA graph and replayed;
  - split into GEMM (q/k/qkv/wo/gate/gate_up/down), attention (rope_kv/attn/combine) and other, by an event
    after each segment.
- **(b) Launch share:** wall − graph replay = launch + gaps, the most a graph-captured feed could save. A bucketed
  scheme would also have to pass the position through a buffer. Enqueue time is reported too.
- **In-situ weighting:**
  - every server runs with `AXON_SERVE_TRACE`;
  - the anatomy is weighted by the fifo60 C=128 cells' feed passes (sizes MEASURED from the trace; depths
    ESTIMATED from the prompt set: shorts at 0, long-prompt chunks token-weighted over the long prompts);
  - cross-checked by comparing the grid's predicted feed time with the trace's measured one.
- **(c) #139:**
  - arms fifo/sjf × `--step-ms` 50/60/80, C=64 and C=128, 2 rounds;
  - arm order rotated, and reversed in round 2; C order alternating per round; one server per arm-round;
  - the showdown's client and SLO, with a fixed salt per (round, C) so every arm does identical work;
  - text and length are checked equal across arms (BITS).

## Decision rule (pre-registered; first match wins)
On the fifo60 C=128 cells' feed passes (the shipped default), as shares of feed-pass wall time:
1. **GEMM ≥ 65 %** → the next pick is a **new prefill-GEMM proposal with a measured upper-bound gate**.
2. **launch + gaps ≥ 25 %** → **graph-captured (bucketed) feeds**.
3. **else** → **the scheduler config**, decided by #139's cells below.

#139 (spec `sched-sjf-token-budget` §8.2, verbatim thresholds). The deciding cell is C=128 at `--step-ms 60`; the
other cells are reported.
- **sjf becomes the default** iff, against fifo at the same `--step-ms`:
  - short TTFT p50 ≤ 0.5×,
  - long TTFT max ≤ 1.5×,
  - aggregate ≥ −3 %,
  - and "Together": sjf's short TTFT p50 ≤ 500 ms and aggregate ≥ 1,868 × 0.97.
- **`--step-ms 50` becomes the default** iff fifo50 against fifo60 at C=128 has ITL p50 ≤ 50 ms, slo_ok ≥ 0.5 and
  aggregate ≥ −3 %.
- **`--step-ms 80`** is reported only.
- **BITS** must hold across all arms, or the comparisons are void.

## Session plan
- Kit `~/.cache/axon-feedanat/h100` (`sess_h100.sh`, `anat.py`, `sd/`, `kcache-sm90a/`, SHA256SUMS): prebuilt
  binaries, sm_90a kernel cache for NVRTC 13.2/13.3/13.4.
- One job: `axon-run --timing h100 --big-models --src KIT --max-min 28`. A session budget (24 min of stages)
  skips arm-rounds rather than overrun.
- `axon-pods up h100 --models sd --hours 0.6`: the dead-man ends it at 36 min, ≤ $2.39 at $3.99/h.
- Cap $2.50. ESTIMATED ≈ 20 min paid, ≈ $1.35: anatomy ~2 min, 12 arm-rounds × ~70 s, ~3 min of pod setup.
- Bella rehearsal (2 arms × 2 rounds, timeouts ×3): job 1026-232230-4945.

## Results
Pending (the session is not booked).
