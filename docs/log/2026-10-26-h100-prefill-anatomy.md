# H100 prefill anatomy of the mix C=128 cell, and #139's config cells (PR #157)

- Date: 2026-10-26. Branch: feat/feed-anatomy. Base: main d8a296c (merged with main 0c23a87, #151 included, at 86953be: the binaries the session ran).
  Status: open. Session ran 10-27 (job 1027-030631-09ae, after the rule below was committed and posted).
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
- Bella rehearsal (2 arms × 2 rounds, timeouts ×3): job 1026-232230-4945, rc 0.
  - Every stage ran; BITS held on 768 of 768 requests.
  - The grid predicted the traced feed time to 1.00× (141.4 vs 141.9 s).
  - Projected to the H100 (bella's C=128 cells are ~3.3× the H100's 38 s): ≈ 14–15 min of stages, inside the 24-min
    budget and `--max-min 28`.
- **Fixed after the rehearsal (ff6f1c3):**
  - The harness classed the Q4_K / Q6_K v and down products (segments `v4`, `v6`, `down4`, `down6`) as "other". That
    made the weighted GEMM share 46 % instead of 78 %, and gave the wrong branch of the rule. Every segment is now
    mapped explicitly: an unknown one panics in the test and stops `anat.py`, which recomputes the classes itself.
  - Wall, graph replay and enqueue were timed in separate blocks, and drift landed on wall − graph (−14 to +12 %);
    they are now timed in turn every repetition. Checked on bella: 1027-014519-e6bd, 40 grid rows; launch + gaps
    0–2 % at most points (13 % at one).
  - The grid now goes to n = 1024: H100 passes are ~470 tokens at p50 under 60 ms, more at 80.

## Results

**Session.** Job 1027-030631-09ae, `--timing h100` on pod2: RunPod H100 80GB HBM3 (SXM, secure, sm_90, 132 SMs,
driver 595.91.07). Binaries and kit at 86953be (main 0c23a87 + the harness), 7B Q4_K_M
(`65b8fcd92af6b4fe…`). rc 0 after 885 s of stages: anatomy 51 s, then 12 arm-rounds of 62–78 s. No stage was
skipped or timed out. FOREIGN_GPU max 0 %. Kernel cache 1 → 4 entries (the kit's; nothing compiled on the pod).
**Cost:** pod up 03:09:38, down 03:27:00 (17.4 min); ledger **$1.399** (cap $2.50).
Raw log, kit scripts and SHA256SUMS: https://gist.github.com/HaseebKhalid1507/aa8b1af9b814d181d28ab9980320c495

**Bits:** 768 of 768 requests identical (text_sha, completion_tokens) across the 6 arms.

### (a)/(b) Feed pass anatomy (`pool_feed_anatomy`, MEASURED)
Shares of a feed pass's wall time: the kernels are the graph-replayed pass, split by the event profile's class shares;
launch + gaps = wall − graph replay.

| depth | n | wall | enqueue | graph | GEMM % | attention % | other % | launch+gaps % |
|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 6.93 | 0.88 | 6.43 | 75 | 8 | 9 | 7 |
| 0 | 64 | 10.28 | 0.73 | 9.68 | 82 | 6 | 7 | 6 |
| 0 | 128 | 12.32 | 0.75 | 11.73 | 83 | 6 | 6 | 5 |
| 0 | 256 | 19.14 | 0.74 | 18.52 | 87 | 5 | 5 | 3 |
| 0 | 384 | 30.13 | 0.78 | 29.50 | 90 | 4 | 4 | 2 |
| 0 | 512 | 34.71 | 0.78 | 34.08 | 90 | 4 | 4 | 2 |
| 0 | 768 | 50.48 | 0.78 | 49.88 | 91 | 4 | 4 | 1 |
| 0 | 1024 | 66.12 | 0.78 | 65.48 | 91 | 4 | 4 | 1 |
| 1000 | 16 | 6.92 | 0.65 | 6.41 | 75 | 9 | 9 | 7 |
| 1000 | 64 | 11.37 | 0.74 | 10.78 | 75 | 14 | 6 | 5 |
| 1000 | 128 | 13.52 | 0.73 | 12.92 | 77 | 13 | 6 | 4 |
| 1000 | 256 | 20.25 | 0.74 | 19.63 | 82 | 9 | 5 | 3 |
| 1000 | 384 | 31.11 | 0.77 | 30.49 | 87 | 7 | 4 | 2 |
| 1000 | 512 | 35.94 | 0.77 | 35.31 | 87 | 7 | 4 | 2 |
| 1000 | 768 | 52.49 | 0.78 | 51.87 | 88 | 7 | 4 | 1 |
| 1000 | 1024 | 68.34 | 0.77 | 67.70 | 88 | 7 | 4 | 1 |
| 2000 | 16 | 7.11 | 0.64 | 6.60 | 73 | 11 | 9 | 7 |
| 2000 | 64 | 12.52 | 0.73 | 11.92 | 69 | 21 | 6 | 5 |
| 2000 | 128 | 14.62 | 0.73 | 13.58 | 70 | 18 | 5 | 7 |
| 2000 | 256 | 21.35 | 0.74 | 20.71 | 79 | 14 | 5 | 3 |
| 2000 | 384 | 32.28 | 0.77 | 31.66 | 84 | 11 | 4 | 2 |
| 2000 | 512 | 37.19 | 0.78 | 36.57 | 84 | 10 | 4 | 2 |
| 2000 | 768 | 54.73 | 0.78 | 54.06 | 84 | 11 | 4 | 1 |
| 2000 | 1024 | 70.56 | 0.77 | 69.93 | 86 | 9 | 4 | 1 |
| 4000 | 16 | 7.35 | 0.64 | 6.84 | 72 | 13 | 8 | 7 |
| 4000 | 64 | 14.69 | 0.74 | 14.08 | 60 | 31 | 5 | 4 |
| 4000 | 128 | 16.84 | 0.74 | 16.23 | 63 | 28 | 5 | 4 |
| 4000 | 256 | 23.45 | 0.74 | 22.83 | 72 | 21 | 4 | 3 |
| 4000 | 384 | 34.58 | 0.77 | 33.95 | 78 | 16 | 4 | 2 |
| 4000 | 512 | 39.63 | 0.77 | 39.01 | 79 | 15 | 4 | 2 |
| 4000 | 768 | 59.11 | 0.78 | 58.42 | 78 | 17 | 3 | 1 |
| 4000 | 1024 | 74.93 | 0.77 | 74.27 | 81 | 15 | 4 | 1 |
| 6000 | 16 | 7.85 | 0.66 | 7.33 | 68 | 17 | 8 | 7 |
| 6000 | 64 | 16.78 | 0.75 | 16.18 | 53 | 39 | 4 | 4 |
| 6000 | 128 | 18.79 | 0.74 | 18.18 | 57 | 36 | 4 | 3 |
| 6000 | 256 | 25.58 | 0.76 | 24.95 | 66 | 27 | 4 | 2 |
| 6000 | 384 | 36.95 | 0.77 | 36.32 | 74 | 21 | 3 | 2 |
| 6000 | 512 | 42.06 | 0.78 | 41.43 | 75 | 20 | 4 | 2 |
| 6000 | 768 | 63.46 | 0.77 | 62.78 | 73 | 23 | 3 | 1 |
| 6000 | 1024 | 79.73 | 0.77 | 79.10 | 77 | 19 | 3 | 1 |

**Weighted over the fifo60 C=128 cells' feed passes** (pass sizes MEASURED from the serve trace; depths ESTIMATED
from the prompt set): **GEMM 78.4 %, attention 15.0 %, other 4.3 %, launch + gaps 2.3 %.** Calibration: the grid
predicts 41.32 s of feed time for those cells' passes, and the trace measured 44.78 s (ratio 1.08).

### (c) #139 cells (medians of 2 rounds; one server per arm-round, fixed salt per (round, C), the showdown SLO)

| C | arm | rounds | agg tok/s (spread) | short TTFT p50 | p99 | long TTFT p99 | long max | ITL p50 | ITL p99 | TPOT p50 | slo_ok | goodput |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | fifo60 | 2 | 1,635.1 (±2 %) | 269.8 | 5,532.1 | 6,614.7 | 6,614.7 | 19.4 | 73.2 | 33.5 | 0.688 | 1,170.1 |
| 64 | sjf60 | 2 | 1,625.8 (±0 %) | 118.9 | 5,712.5 | 6,394.5 | 6,394.5 | 19.0 | 72.2 | 31.3 | 0.684 | 1,161.2 |
| 64 | fifo50 | 2 | 1,497.4 (±1 %) | 490.5 | 6,499.4 | 8,154.5 | 8,154.4 | 28.0 | 61.2 | 37.2 | 0.676 | 1,036.2 |
| 64 | sjf50 | 2 | 1,527.8 (±1 %) | 106.8 | 6,136.9 | 7,163.4 | 7,163.4 | 22.9 | 60.5 | 35.0 | 0.664 | 1,031.3 |
| 64 | fifo80 | 2 | 1,669.7 (±1 %) | 253.2 | 5,041.1 | 6,002.5 | 6,002.5 | 17.9 | 98.4 | 32.7 | 0.715 | 1,242.8 |
| 64 | sjf80 | 2 | 1,697.3 (±2 %) | 151.2 | 4,715.1 | 5,760.4 | 5,760.4 | 17.5 | 96.6 | 32.6 | 0.742 | 1,336.4 |
| 128 | fifo60 | 2 | 1,845.5 (±1 %) | 2,320.3 | 14,063.1 | 15,618.2 | 15,618.2 | 55.9 | 80.0 | 58.0 | 0.279 | 551.7 |
| 128 | sjf60 | 2 | 1,885.0 (±0 %) | 953.5 | 15,609.7 | 14,773.4 | 14,773.4 | 53.1 | 78.2 | 50.5 | 0.358 | 746.2 |
| 128 | fifo50 | 2 | 1,570.0 (±2 %) | 8,860.5 | 17,589.5 | 19,637.8 | 19,637.8 | 47.0 | 62.8 | 46.0 | 0.102 | 163.3 |
| 128 | sjf50 | 2 | 1,678.8 (±0 %) | 5,624.6 | 22,697.2 | 16,987.8 | 16,987.8 | 47.0 | 69.5 | 47.2 | 0.293 | 500.0 |
| 128 | fifo80 | 2 | 1,984.8 (±0 %) | 861.7 | 12,040.6 | 13,514.8 | 13,514.8 | 31.0 | 101.1 | 53.2 | 0.367 | 792.5 |

### Step traces (`AXON_SERVE_TRACE`, s)

| arm | round | C | steps | wall | feed passes | prompt tokens | feeds | decode | host | 2-pass steps | tokens per pass p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fifo50 | 1 | 64 | 924 | 23.98 | 436 | 122,385 | 12.23 | 11.40 | 0.32 | 400 | 313 |
| fifo50 | 1 | 128 | 1367 | 45.33 | 921 | 242,643 | 25.09 | 19.41 | 0.75 | 876 | 291 |
| fifo50 | 2 | 128 | 1350 | 46.19 | 933 | 243,950 | 25.32 | 19.99 | 0.79 | 891 | 286 |
| fifo50 | 2 | 64 | 904 | 23.65 | 428 | 120,988 | 11.99 | 11.26 | 0.36 | 383 | 321 |
| fifo60 | 1 | 64 | 902 | 22.05 | 336 | 122,365 | 10.59 | 11.10 | 0.33 | 267 | 477 |
| fifo60 | 1 | 128 | 1057 | 38.76 | 743 | 242,688 | 22.35 | 15.54 | 0.82 | 550 | 392 |
| fifo60 | 2 | 128 | 1056 | 39.07 | 744 | 244,340 | 22.42 | 15.78 | 0.82 | 560 | 390 |
| fifo60 | 2 | 64 | 862 | 21.55 | 330 | 120,677 | 10.46 | 10.74 | 0.32 | 268 | 495 |
| fifo80 | 1 | 64 | 898 | 21.52 | 262 | 122,385 | 10.12 | 11.02 | 0.35 | 202 | 422 |
| fifo80 | 1 | 128 | 1019 | 36.28 | 541 | 242,684 | 20.49 | 14.91 | 0.83 | 401 | 400 |
| fifo80 | 2 | 128 | 979 | 36.10 | 548 | 244,367 | 20.60 | 14.67 | 0.79 | 395 | 400 |
| fifo80 | 2 | 64 | 847 | 21.18 | 261 | 120,677 | 10.06 | 10.71 | 0.38 | 195 | 428 |
| sjf50 | 1 | 64 | 901 | 23.30 | 413 | 121,629 | 11.84 | 11.11 | 0.32 | 353 | 350 |
| sjf50 | 1 | 128 | 1240 | 42.90 | 880 | 241,623 | 24.25 | 17.74 | 0.76 | 765 | 321 |
| sjf50 | 2 | 128 | 1222 | 42.67 | 871 | 242,584 | 24.22 | 17.54 | 0.76 | 758 | 328 |
| sjf50 | 2 | 64 | 895 | 23.36 | 409 | 120,677 | 11.76 | 11.18 | 0.37 | 355 | 358 |
| sjf60 | 1 | 64 | 901 | 21.95 | 323 | 121,186 | 10.54 | 11.03 | 0.34 | 252 | 517 |
| sjf60 | 1 | 128 | 1094 | 38.13 | 683 | 241,643 | 21.56 | 15.74 | 0.73 | 529 | 477 |
| sjf60 | 2 | 128 | 1040 | 38.08 | 690 | 241,876 | 21.77 | 15.37 | 0.83 | 528 | 475 |
| sjf60 | 2 | 64 | 869 | 21.91 | 324 | 120,677 | 10.51 | 10.96 | 0.39 | 255 | 513 |
| sjf80 | 1 | 64 | 871 | 21.22 | 257 | 120,902 | 10.02 | 10.82 | 0.34 | 177 | 761 |
| sjf80 | 1 | 128 | 999 | 35.65 | 530 | 241,669 | 20.26 | 14.53 | 0.79 | 356 | 707 |
| sjf80 | 2 | 128 | 984 | 35.72 | 532 | 241,869 | 20.28 | 14.60 | 0.77 | 361 | 695 |
| sjf80 | 2 | 64 | 834 | 20.78 | 249 | 120,674 | 9.87 | 10.55 | 0.32 | 179 | 773 |

## Verdicts (the pre-registered rule, applied as written)
- **Anatomy: branch 1.** GEMM is 78.4 % ≥ 65 % of the feed pass. Next pick: **a new prefill-GEMM proposal with a
  measured upper-bound gate.** Launch + gaps is 2.3 %, far under the 25 % that branch 2 (graph-captured feeds)
  needed: capturing feeds would save at most ~2 % of feed time.
  - A brief for the pick went to the lead (outside this PR). Its proposed gate: before any build, a measured upper
    bound (the W4A8 probe skeleton at the serve's M = 384 / 512 on fused gate_up and down, Q6_K included, with a
    measured exact epilogue), and a weighted exact speedup ≥ 1.34× (= feeds −20 %, ~−11 % of the C=128 cell's wall).
  - The GEMM time by shape at n = 512: gate_up 52 %, down 29 % (the Q6_K layers 15 %), q/wo 13 %, k/v 4.5 %; all
    GEMMs at 102 TMAC/s, against a 325 TMAC/s mma.sync peak.
- **#139, deciding cell C=128 at `--step-ms 60`:**
  - (c) clause, sjf against fifo: short TTFT p50 0.41×, long TTFT max 0.95×, aggregate +2.1 %: PASS.
  - "Together": sjf's short TTFT p50 is 953.5 ms, over the 500 ms bound (aggregate 1,885 ≥ 1,812 passes): **FAIL**.
  - **sjf stays opt-in.**
- **#139 `--step-ms 50`:** fifo50 against fifo60 at C=128: ITL p50 47.0 ms (passes), slo_ok 0.102 (needs ≥ 0.5),
  aggregate −14.9 %: **FAIL**. 60 stays the default.
- **#139 `--step-ms 80`** (reported only, per the rule): fifo80 against fifo60 at C=128 is +7.5 % aggregate, slo_ok
  0.367 vs 0.279, ITL p99 101.1 vs 80.0 ms. sjf80 meets every (c) and "Together" threshold at C=128 (short p50 173 ms,
  aggregate 2,013). The rule makes no default change on 80. A default of `--step-ms 80` (with or without sjf) would
  need its own pre-registered decision.

## Observed, not pre-registered
- **Tile cliff at n = 384.** A 384-token pass costs 29.5 ms of GEMM against 32.7 ms at 512: 85 against 102 TMAC/s
  over all GEMMs, with down at 77 against 103 and q/wo at 63 against 85. fifo60 C=128 passes run at p50 ≈ 390 tokens.
- sjf at C=128, against fifo at the same step: short TTFT p99 is higher (15.6 vs 14.1 s at 60) and long TTFT p99 lower
  (14.8 vs 15.6 s). Goodput is +35 % at 60 (746 vs 552), 3.1× at 50 (500 vs 163) and +2.5 % at 80 (812 vs 793).
