# Outside review: next steps after the H100 showdown (2026-10-19 ~23:00 EDT)

> **Outside review, curated for the repo.** Written by Fable, an outside consultant model, read-only, at main
> `764792f`. The opinions and rankings are the reviewer's; decisions are Haseeb's. Labels and citations are the
> reviewer's own. Curation: personal and financial specifics (account balance, credit amounts, a budget split,
> hardware-purchase advice, details about the owner's machines) are removed and marked *[omitted]*; editor's notes
> marked **[Editor]** flag where a claim reads differently against the published tables
> ([log/2026-10-19-h100-showdown.md](../log/2026-10-19-h100-showdown.md)). Everything else is the review's text. The
> previous review it scores against: [2026-10-19-outside-review-bigpicture.md](2026-10-19-outside-review-bigpicture.md).
> Paths under `~/.cache` and the overnight workspace (`PROGRESS.md`, `specs/`) are the reviewer's sources on the
> project's machines; they are not in the repo.

Read-only. Tree: `origin/main` `764792f`. Sources: my 10-19 01:14 review ([`fable-bigpicture-20261019.md`](2026-10-19-outside-review-bigpicture.md)), the H100
showdown raw data (`~/.cache/axon-h100sd/pod-pod1/results/h100-1019-2058/{cells,requests}.jsonl`, `cost.json`,
`config.json`, `engines/axon.log`), `PROGRESS.md` (through 10-19 22:10), `docs/design/determinism-modes.md` (#101),
`docs/log/*`, `src/serve/engine.rs`, `src/cuda/qwen2/batch.rs`, `~/.cache/axon-run/jobs-done/*.json` (1,998 job records),
`specs/fp8-safetensors.md`, `specs/r16-pick4-moe-resident.md`. Labels: **MEASURED** (source cited), **CODE** (file:line),
**ESTIMATED** (arithmetic shown), **OPINION**.

## Executive summary

1. Four of my five recommendations were acted on in 22 h: the mission benchmark ran ($3.09), the determinism ADR shipped
   (#101), gate logs went public, kernel micro-rounds were paused. The MoE-to-the-end item is at step 0; `lib.rs` did not happen.
2. The H100 showdown is the first honest datacenter number and it says: **Axon is a 3.1–3.8× throughput deficit at
   C=128 against vLLM/SGLang** (mix 1,868 vs 5,748–7,146 tok/s), while winning TTFT at C=1 and ITL p99 at every C≥32.
   **[Editor]** Against the tables: 3.1–3.8× is vs vLLM-AWQ and the two FP8 rows; vs SGLang-AWQ it is 2.0× (mix) and
   1.8× (classic, where the range is 1.8–3.2×). Axon's ITL p99 is the lowest only at mix C=128; at mix C=32 / 64
   SGLang-FP8 is lower (24.3 / 65.8 vs 68.9 / 74.0 ms), and on classic Axon's ITL p99 is the highest from C=32 up.
3. The gap decomposes (MEASURED+ESTIMATED, §2) as: prefill GEMM 2.7× slower (13.8k vs 37.7k tok/s on a 7.6k prompt), the
   nb=128 decode step 3.3× slower (18.6 vs 5.1–5.7 ms, 6× above its own DRAM floor), and ~25 % of wall time in
   two-pass-per-step scheduling overhead that vLLM doesn't pay. Order of blame: kernel (≈65 %), scheduler (≈25 %), quant format (≈10 %).
4. None of that is the bit-identity contract *as such*; it is Hopper kernel engineering (int8 `mma.sync` at ~10 % of the
   H100's int8 pipe) plus the engine running decode and prefill as separate passes. The contract becomes the blocker only at FP8.
5. Next five, ranked: (1) a $5 H100 step-anatomy + two scheduler fixes; (2) Hopper exact W4A8 wgmma GEMM round 2 with a
   hard kill rule; (3) one MoE resident on the GPU; (4) FP8 under `--numerics fast` only if (2) misses; (5) `lib.rs` + v0.1.
6. Adopt the mode scaffolding of #101 now (M1–M3, M5, under a week, no kernels); build no fast kernel until step (2) reports.
7. Process: 1,998 jobs, **1,179 h of queue wait against 277 h of run**, 37 % of jobs killed, every bella timing A/B on
   10-18/19 tainted at least once, 50 % pod waste on the showdown. Fix the gate host, cap concurrent tracks, persist pod state.
8. *[omitted: a budget-allocation recommendation; the process changes it rests on are in §5.]*

---

## 1. Scorecard against the 10-19 review

| # | recommendation (10-19 01:14) | status | evidence |
|---|---|---|---|
| 1 | Run the mission benchmark once, honestly, publish it | **done, narrower than asked** | H100 SXM, Qwen2.5-7B only, 5 engines × 5 C × 2 workloads × 2 rounds, 120 cells, 0 drops, $3.09 (MEASURED `cost.json`, PROGRESS 21:37). Missing vs the ask: Qwen3-30B-A3B ("MoE phase cut", PROGRESS 03:15) and the shared-prefix agent workload. Int4 arm is AWQ vs Q4_K_M, not the same weights (documented) |
| 2 | Finish one MoE to GPU-resident experts; park the others | **step 0 passed, nothing on main** | S0.1/S0.4 PASS 02:50; S0.3 MEASURED 06:16: CPU experts are 11.85 of ~15.5 ms per Moonlight decode step (76 %); PR A (twin kernels, 4/4 tests on 3 hosts) sits uncommitted in `~/.cache/axon-wt-moeres`. Meanwhile #89 (deepseek2 s1) and #98 (gpt-oss s1) landed: the breadth I argued against, but they were already built, so landing them was cheaper than parking |
| 3 | Write the determinism-cost ADR | **done, and better than asked** | #101, 454 lines, every technique costed with labels, merged PROPOSED (`764792f`). My "compounds with batch size" thesis was answered with numbers (§2.6: ≤ 5 points of the C-curve gap). §2 below revisits that with the H100 data |
| 4 | Make gates reproducible by someone who isn't bella | **partly** | `origin/gate-logs` branch exists (3 train logs pushed), PR comments, opt-in agent trailer, `docs/gpu-ci-options.md` decided *against* a self-hosted runner (fork code next to keys). No GPU CI; every gate still runs on the private queue |
| 5 | Freeze ≤8B kernel micro-rounds; `lib.rs`; FP8 spike | **partly** | r17 pick 1 (#100) finished and merged (+15.4 % 3B pp1050 on avante, MEASURED job 4c86; the "+22–28 % per MAC" is the kernel-level figure), then "kernel rounds r18+ NOT allowed" (03:40). FP8 got a 740-line spec and the safetensors loader (#102) merged. `lib.rs`: not started |

**What I got wrong (OPINION, with the numbers):**
- I wrote "expect to lose" the batch-1 cell. Axon **won** C=1 against vLLM (287 vs 235–246 mix) and tied SGLang-AWQ
  (304 vs 296 classic, 287 vs 284 mix: within 1–3 %, i.e. noise). The honest reading: at batch 1 Axon equals the best
  datacenter int4 engine on an H100; the win is against vLLM's per-step overhead, not against its kernels.
- I framed the contract as "a tax that compounds with batch size". The ADR's rebuttal (the prefill GEMM is MIO-bound on
  sm_89; order costs ≤ 2 %) is correct *on consumer cards*. On the H100 the question is still open and nobody has run the
  one experiment that settles it (§2.4). I was too confident; the ADR was measured on the wrong card.
- I predicted the single engine thread + host sampling would be "the step" at C=128 on a 0.5B. On the 7B it is not
  visible in the showdown data; the structural cost is the *two passes per step* (§2.3), which I didn't name.

---

## 2. What the H100 result really says

### 2.1 Headline (MEASURED, `cells.jsonl`, median of rounds)

| cell | Axon | vLLM-AWQ | SGLang-AWQ | vLLM-FP8 | SGLang-FP8 |
|---|---|---|---|---|---|
| mix C=1 agg tok/s | **287** | 235 | 284 | 239 | 246 |
| mix C=128 agg | 1,868 | 5,748 | 3,770 | 7,146 | 7,135 |
| classic C=128 agg | 4,485 | 12,190 | 8,288 | 14,205 | 14,380 |
| mix C=128 goodput | 513 | 4,953 | 2,645 | 7,057 | 7,129 |
| mix C=128 ITL p99 ms | **77.5** | 205 | 320 | 140 | 126 |
| mix C=128 TTFT p99 ms | 14,513 | 3,062 | 5,266 | 2,114 | 1,920 |

Both Axon rounds agree within ±2 % at C=64/128 (±23 % at C=8 mix is the small-sample workload, both engines). The control
pass (no sleep/memory-saver) matches the parked pass within 4 %, so parking didn't rig the others.

### 2.2 Decomposing the gap: per-request data

**Prefill rate (MEASURED, `requests.jsonl`, mix C=1 round 1, a lone request on an idle server):**

| engine | 7.65k-token prompt TTFT | tok/s | 4.07k prompt TTFT | tok/s |
|---|---|---|---|---|
| Axon Q4_K_M | 555 ms | **13.8k** | 296 ms | 13.75k |
| vLLM-AWQ (Marlin) | 257 ms | 29.8k | — | — |
| SGLang-AWQ | 389 ms | 19.7k | — | — |
| vLLM-FP8 | 203 ms | **37.7k** | 105 ms | 38.8k |
| SGLang-FP8 | 170 ms | 45k | — | — |

ESTIMATED work rate: 7B ≈ 6.53 GMAC/token in projections (`hopper.md` §2) + ~0.7 GMAC attention at mean position 3.8k
→ ~7.2 GMAC/token. Axon: 7,651 × 7.2 G / 0.555 s ≈ **100 TMAC/s = 10 % of the H100's 989 TMAC/s int8 dense pipe**.
vLLM-FP8 ≈ 270 TMAC/s (27 % of the FP8 pipe, same peak). vLLM-AWQ ≈ 215 TMAC/s = **43 % of the FP16 pipe** (495).
So Marlin, on a pipe with half Axon's peak, does 2.2× Axon's work rate. The ceiling for an exact W4A8 kernel at Marlin's
efficiency is ~400 TMAC/s, 4× today. This is the single biggest term: 2.7× on every prefill token, and every prompt
token in the mix workload (251k of them per C=128 cell) pays it.

**Decode step at nb=128 (MEASURED, classic C=128, ITL p50 = the steady decode step, mean depth ≈ 790):**

Axon **18.6 ms**; vLLM-AWQ 5.1; vLLM-FP8 5.7; SGLang-FP8 4.9. ESTIMATED memory floor for Axon's bytes: weights 4.68 GB +
KV 128 × 790 × 57,344 B = 5.8 GB → 10.5 GB / 3.35 TB/s = **3.1 ms**. Axon runs at 6× its floor; vLLM at ~1.6× of its
(slightly larger) floor. GEMM MACs at nb=128 are 128 × 6.53 G = 836 GMAC; at Axon's own prefill rate that is 8.4 ms, so
roughly half the 18.6 ms is the GEMM running *slower than prefill* and the other half is everything else. Why the GEMM
is slower at nb=128 than at nb=7,651 (CODE `batch.rs:316-333`, `tile_t`): 128 tokens = 2 token tiles of 64; the q/wo/down
products (3,584 rows) launch 28 × 2 = **56 CTAs on 132 SMs** (42 % of one wave), k/v (512 rows) launch **8 CTAs**, each
with a serial K loop of 3,584–18,944. Only gate/up (592 CTAs) fills the card. On the 4090 at nb 64 the same three
products were 5.2 of 8.1 ms of GEMM time (MEASURED `2026-10-18-c-curve.md` per-kernel table). This is exactly the shape
the ADR's M6 (fixed-chunk split-K, *exact mode*) is for.

**The scheduler term (ESTIMATED from the mix C=128 cell):** wall 38.1 s. Prompt tokens 251k at 13.8k tok/s = 18.2 s.
Decode: 256 requests × 278 tokens / 128 rows ≈ 556 steps × 18.6 ms = 10.3 s. Sum 28.5 s; measured 38.1 s → **~9.6 s
(25 %) is neither prefill nor decode kernel time.** The same arithmetic for vLLM-FP8: 251k / 37.7k = 6.7 s + 556 × 5.7 ms
= 3.2 s = 9.9 s vs 10.0 s measured: **~0 overhead**. Where Axon's 9.6 s goes (CODE):
- `engine.rs:391-420` `prefill()` then the decode step: **two GPU passes per step**. vLLM puts the 128 decode rows into
  the prefill chunk's forward, so the decode rows ride the weight read the prefill already paid. Axon pays a full
  18.6 ms step on top of every prefill chunk. Identity A (decode == batched prefill at any batch position, ADR §1.1)
  means merging them is **bit-compatible by construction**: this is a scheduler change, not a numerics one.
- `PREFILL_MS = 60` (`engine.rs:83`): the chunk is sized to 60 ms minus the last decode step, so at nb=128 it is
  ~35–40 ms ≈ 500 tokens, which runs the GEMM at 8 token tiles instead of 120; and `MAX_PREFILLING = 2` (`:205`) serialises
  the 24 long prompts. A 7.5k prompt needs ~15 steps ≈ 0.9 s by itself; queued two at a time behind 128 arrivals → TTFT
  p99 14.5 s (MEASURED long7.5k p90 10.6 s, max 12.0 s).
- 416 of 512 requests are 59-token prompts and still wait a median **2.5 s** for a first token (MEASURED). There is no
  shortest-first admission; a 59-token prompt costs <1 % of a step.

**Goodput collapse (slo_ok 0.26) is capacity, not a tail:** the SLO is ITL ≤ 50 ms and Axon's mix C=128 step is 58 ms
at p50 (p99 77). Every request fails ITL by the same ~8 ms. `--prefill-ms 40` would pass ITL and push TTFT further out.
The budget knob is also why Axon's ITL p99 is the best in the table; that is a real scheduler property worth keeping.

### 2.3 Blame by category (OPINION on the arithmetic above)

| category | share of the C=128 gap | evidence |
|---|---|---|
| **kernel**: Q4_K int8 `mma.sync` GEMM at 10 % of the Hopper int8 pipe; small-M occupancy at nb ≤ 128 on 132 SMs | ≈ 65 % | 2.7× prefill, 2× of the 3.3× decode step |
| **scheduler**: separate prefill and decode passes, 60 ms chunk budget at nb=128, 2 concurrent prefills, FIFO admission | ≈ 25 % | the 9.6 s residual; TTFT of short prompts |
| **quant format**: Q4_K's 256-superblock scale structure forces an int32→f32 drain every 256 K (CODE `decode.cu:9-12`), 8 wgmma k32 instructions per drain; FP8/AWQ-Marlin accumulate the whole K in one chain | ≈ 10 %, **UNMEASURED on Hopper** | `hopper.md` §2: 1,400 non-tensor issue slots per 290 tensor clocks; the 10-16 gemmh2 probe reached 84 TMAC/s "issue/latency-bound at 12 % of wgmma peak" (`experiments-rejected.md`) |
| the bit-identity contract per se | small, inside the "quant format" row | the drain is the contract's only Hopper-specific cost; the ADR costed it on sm_89 where it is free |
| sm_90 features unused | folded into "kernel" | TMA/wgmma exist only for the fused gate+up (`gemmh2`), routed by `gemmh_plain` (`batch.rs:42`); never for q/k/v/wo/down at nb ≤ 128; no FP8; no cluster/DSMEM |

### 2.4 What to measure next, cheaply (≈ $5 of H100, one session, prebuilt binaries)

1. **`pool_step_cost` on the H100 at nb 1/16/32/64/128/256, depth 600 and 4,096, per kernel** (the harness exists,
   `2026-10-18-c-curve.md`; it has only ever run on a 4090 at nb ≤ 64). This splits the 18.6 ms into GEMM / attention /
   head / elementwise and tells whether M6 split-K or the Hopper GEMM is the first kernel.
2. **`cuda_gemm_bench` on 7B shapes at nb 64…1,024 for `gemm2s` / `gemm2t` / `gemmh2`**, with one deliberately *wrong*
   variant: `gemmh2` with the per-256 drain removed (accumulate the whole K in int32, single epilogue). Its speed is the
   **upper bound of what the contract costs on Hopper**; if it is within 10 % of the exact kernel, the contract is
   exonerated on this card and the gap is pure kernel engineering. Nobody has this number; the ADR's §2 needs it.
3. **One `serve` step trace at mix C=128** (events around `prefill()` and the step; `/metrics` already counts steps):
   confirms the two-pass accounting in §2.2 before anyone builds the merged pass.
4. Re-run only the two Axon C=128 cells after each fix (≈ 2 min of pod each) rather than the 120-cell matrix.

---

## 3. Ranked next five

Weighing: the datacenter gap is 3–4× and measured; "best models" is at step 0 with a passing kill rule; hardening/adoption
has zero external users to harden for. The order below puts the cheapest decisive measurement first, the one hard
kernel second, the first real model third, FP8 fourth *because it depends on the second's verdict*, and adoption fifth.

### 1. H100 step anatomy + two scheduler fixes (exact mode)
- **Why:** §2.4's measurements decide steps 2 and 4; the scheduler term is 25 % of the gap and needs no numerics change.
- **What:** (a) the three measurements above; (b) decode rows join the prefill chunk's pass (one `feed` per step; identity
  A guarantees bits); (c) shortest-first admission for prompts under ~256 tokens and `MAX_PREFILLING` scaled with slots
  (2 → ~C/16); (d) chunk budget expressed as tokens-per-step at the measured cost rather than a fixed 60 ms.
- **Expected gain (ESTIMATED):** removing the separate decode pass saves ≈ (18.6 − 128 tokens at prefill rate ≈ 9 ms) ×
  556 steps ≈ 5 s of 38 s → **+15 % mix C=128 throughput**; wider chunks and 4–8 concurrent prefills recover part of the
  remaining ~4.6 s → **+25–40 % total**; short-prompt TTFT p50 2.5 s → < 0.3 s; goodput ×3–5 at C=64 (slo_ok 0.74 → ≥ 0.95).
  Still ~2.5× behind vLLM-FP8.
- **Cost:** 2–3 builder-days; **$5** H100 (two 30-min sessions, prebuilt). Bella/avante suffice for the bits gates.
- **Kill rule:** merged pass must give mix C=64 agg ≥ +15 % on a 4090 with ITL p99 ≤ today's; else revert and keep (c)/(d).
- **Unblocks:** a reliable per-kernel target for step 2; every later C ≥ 32 number.

### 2. Hopper exact GEMM, round 2: W4A8 wgmma with register dequant, integer superblock reduce, TMA weights; plus M6 fixed-chunk split-K for the narrow products
- **Why:** the 2.7× prefill and ~2× of the decode-step gap. The 10-16 `gemmh` probe tried SS wgmma with weights unpacked
  into smem and stalled at 84 TMAC/s; an RS design (dequant in registers, as Marlin does) was the "untried next step"
  (`experiments-rejected.md`, r4 stage 3 note). The per-256 drain is a *fixed* order, so the kernel stays exact (ADR §2.2:
  "the contract forbids arrival-order and SM-count-dependent splits, not fixed splits"). Split-K with fixed chunks
  (ADR M6) turns 56 CTAs into 224+ for q/wo/down at nb ≤ 128.
- **Expected gain (ESTIMATED):** Marlin's 43 % of its pipe on Axon's 2×-wider int8 pipe = 400 TMAC/s; assume half:
  **200 TMAC/s** → prefill 13.8k → ~27k tok/s (within 10 % of vLLM-AWQ, 0.7× vLLM-FP8); nb=128 GEMM 8.4 → ~4.5 ms; with
  step 1, classic C=128 4,485 → **~9–10k** (vLLM-AWQ 12,190). M6 alone on today's kernel: ESTIMATED −3 ms of the 18.6 ms
  step (q/k/v/wo/down from ~40 % to full-wave occupancy).
- **Cost:** 6–10 builder-days of one strong kernel builder (not a swarm), **$40** of H100 in ~10 one-hour sessions,
  plus the M6 NUMERICS_EXACT bump and re-pinned md5s (ADR M6: 2–3 days, included).
- **Kill rule:** after 5 builder-days, gate 3,584 × 18,944 at nb 1,024 **< 180 TMAC/s on the H100** (2× today's 84–93)
  → stop, record in `experiments-rejected.md`, and go to step 4's fast-mode kernel instead. M6 separately: H100 7B
  nb=128 step −10 % or revert.
- **Unblocks:** every datacenter claim; resident-MoE prefill (step 3) on the same GEMM; 32B/70B on H100s; makes
  "squeeze every drop" a sentence that could become true.

### 3. One MoE resident on the GPU: Moonlight-16B-A3B now, Qwen3-30B-A3B as the headline (PR A → D of `r16-pick4`)
- **Why:** "run the best models" has zero GPU-resident MoE tokens on main after three stage-1 architectures. Step 0
  passed every kill rule (S0.1 N(8k)=25/26 expert layers fit bella; S0.4 expert GEMMs 2.1–2.5 ms/layer on bella vs a
  14.6 ms kill; S0.3 CPU experts = 76 % of the decode step, MEASURED 06:16). PR A exists.
- **Expected gain (ESTIMATED from S0.3):** step 15.5 ms → 3.65 ms GPU + ~2.5 ms resident experts ≈ **6 ms → ~160 tok/s**
  Moonlight decode on bella (today 62.4; **[Editor]** #89's own bench, a different job, measured 54.0); prefill ×3–5 (experts stop being a CPU GEMM). On a 24 GB card
  Qwen3-30B-A3B Q4_K_M (≈ 18 GB) fully resident: first real ≥30B-class model with a measured number vs llama.cpp and vLLM.
- **Cost:** ~5 builder-days (the spec's own figure), **$10** (one 32–48 GB pod session for 30B-A3B at ctx 8k; a 5090 at
  ~$0.9/h or an A6000/L40S).
- **Kill rule:** resident decode < 1.5× the CPU-expert decode on bella → stop at PR B; 30B-A3B pp2100 < llama.cpp
  `--cpu-moe` on the same pod → not a headline.
- **Unblocks:** the K2/DeepSeek template (experts resident per layer, the same twin kernels), gpt-oss-20b on 24 GB
  (`gptoss-prefill.md` item 1 becomes the same code), an honest "best models" row.

### 4. FP8 on Hopper under `--numerics fast` (M1–M3, M5 first; then the FP8 W8A8 wgmma GEMM), gated on step 2's verdict
- **Why:** the ADR is right that FP8 is "the real reason to have a mode" (§2.5): no exact int block sums exist. It is also
  what DeepSeek V3 / Kimi K2 ship natively (128×128 block scales, `fp8-safetensors.md` §VERIFIED), so "run the best
  models as published" needs it regardless of the throughput race. Stage A (loader) is on main.
- **Sequencing (OPINION):** if step 2 hits its kill rule, FP8-fast is the *replacement* kernel and jumps to #2; if step
  2 succeeds, FP8 is still needed for the K2 path but stops being a throughput argument on dense 7B (FP8 reads 1.8× the
  bytes of Q4_K_M; at nb=128 both are compute-bound on pipes of equal peak). Either way: the **mode scaffolding (M1–M3,
  M5) now** (§4 below), the **Hopper FP8 kernel after step 2 reports**, and the spec's stage B (FP8 on sm_89/sm_120
  consumer cards) **demoted**: it serves no mission cell.
- **Expected gain (ESTIMATED):** parity-class prefill with vLLM-FP8 (30–38k tok/s) if the kernel reaches their 27 % of
  peak; the K2/DeepSeek path loads lab checkpoints without a GGUF requant.
- **Cost:** 8–12 builder-days + **$40** of H100; the tolerance gates exist (`assert_graded`), the χ² and ΔNLL hooks are M3.
- **Kill rule:** FP8-fast GEMM < 1.5× the exact int8 GEMM of step 2 on the same shapes → the second numerics isn't
  worth its test column for dense models; keep it only for block-scaled checkpoints.
- **Unblocks:** DeepSeek V3 / Kimi K2 stage 2 at FP8, an FP8 board, the TP/NCCL discussion (irrelevant until a 2-GPU
  exact TP exists).

### 5. Adoption floor: `lib.rs`, a tagged v0.1, the honest board on the README, one sentence for "why not vLLM"
- **Why:** 764 commits, 100 PRs, zero external users, zero external bug reports. The showdown gave the project its
  first honest positioning sentence (OPINION, from §2.1): *"equal to the fastest int4 engines at batch 1, best ITL tail
  under load, reproducible bits; 3× behind on throughput at C=128 today."* A release with that sentence attracts the
  user who values the first three; waiting for the throughput parity that step 2 may or may not deliver attracts nobody.
- **Cost:** 3 builder-days, $0. `lib.rs` is a split, not a rewrite (the 10-13 review's top item, twice deferred).
- **Kill rule:** none; it is hygiene. Measurable outcome: one external issue or PR within 30 days.
- **Unblocks:** embedding (Synaps on Axon without a subprocess), a second human reading the code, the bus-factor question.

**Below the line, deliberately:** batched speculative decoding (#90 closed by its rule, correctly; its d16k loss is the
depth-aware gate problem the closing comment names); KV stage B (`--cache-ram`) and shared-prefix pages (the agent
workload is still unmeasured, so their payoff is unmeasured); big-model offload (`g4-offload`: a laptop story);
Gemma 27B / sm_61 / CPU fast prefill (M4); cross-platform. Any of them can jump the queue the day a named user asks.

---

## 4. The determinism-modes ADR, given the H100 numbers

**Adopt the mode now; build nothing in it yet (OPINION, consistent with the ADR's own §4.6).**

- **Now (this week):** M1 (`--numerics` flag, `numerics=exact/1` tuple line, `fast` refused), M2 (`--datacenter` bundle),
  M3 (graded-tolerance scaffolding), M5 (NVRTC key, kvsave cross-mode test). Under a week, no kernels. Do M1 before any
  external user exists: the fingerprint changes once, and the showdown is a reason to publish a release note anyway.
  Answer the ADR's Q1 "ship with the next NUMERICS bump" → yes, and M6 (step 2 above) is that bump.
- **Not now:** M4 (CPU fast prefill). The ADR asks whether anyone runs CPU prefill at scale; nobody does.
- **`--numerics fast` as a deployment default: later, and only after step 2 reports.** Today the fast set is empty, and
  §2.3 says the measured gap is mostly kernel engineering that is legal in exact mode. Turning on an empty mode buys a
  fingerprint change and nothing else.
- **Which fast kernels would close the most gap first, if the exact round fails its kill rule:**
  1. **A Hopper GEMM with free accumulation order** (W4 or W8 weights × 8-bit activations, f32 accumulate over the whole
     K, no per-256 drain; FP8 W8A8 is the natural one since the loader already handles the checkpoints). That is the
     vLLM/SGLang kernel class and the 2.7× prefill term. Everything else is second-order on these numbers.
  2. **Split-K with arrival-order fixup for q/k/v/wo/down at nb ≤ 128** — but the *fixed-chunk* exact version (M6) should
     be tried first and probably captures most of it.
  3. Not worth a fast kernel on this evidence: split-KV attention (attention is ≤ 2 ms of the 18.6 ms step at 8k ctx),
     NCCL (no TP exists), the per-32 epilogue (ADR §2.3: a wash).
- **One correction to the ADR (OPINION):** §2.6's "≤ 5 points of the vLLM gap is the contract" was estimated on a 4090
  curve that stopped at nb 64. The H100 nb=128 step is 6× its DRAM floor and the Hopper-specific cost of the per-256
  drain has never been measured (§2.4 item 2). Until it is, the ADR's "nearly free on the GPU" should read "nearly free
  on sm_89/sm_120; unmeasured on sm_90".

---

## 5. Process: where time and money leak

**MEASURED from `~/.cache/axon-run/jobs-done` (1,998 job records, 10-16 → 10-19) and PROGRESS.md:**

| leak | number | what it means |
|---|---|---|
| Queue wait | **1,179 h of cumulative wait vs 277 h of run**; median 18 min, p75 65 min, p90 144 min, max 11.3 h | the fleet is 4.3× oversubscribed in wall-clock terms; builders idle ("all idle waiting on gates", 10-18 08:52) and then spawn new tracks to fill the time, which deepens the queue |
| bella as the gate host | 620 jobs, 91 run-h, **750 wait-h**; timing jobs median 26 min, p90 **218 min** wait | bella is a desktop in daily use: a daytime VRAM cap, owner holds, foreign load from the desktop session. 191 job logs mention foreign/tainted GPU; 3 TAINTED + 2 INCONCLUSIVE A/Bs on 10-18/19, each re-run (#90 ran its kill rule three times) |
| Killed jobs | **732 of 1,998 (37 %)**: 655 killed in queue (superseded by a new head or a stale train), 77 after starting (19 h of run lost) | every push to a PR re-gates the full tier; a train goes STALE whenever main moves (two trains rebuilt on 10-19) |
| Failing runs | 895 jobs ended non-zero; 53 h of run on them | includes the hidden `ILLEGAL_ADDRESS` poison era (13 pod logs "weaker than reported", fixed by #82) and OOM-kills |
| Laptop fleet | avante and jade both died on battery 10-19 05:44–08:53; jade OOM ×2 on a 3 GB debug suite; bella RAM thrash ×2; the scheduler daemon itself oomd-killed twice (06:54, 08:17) | three incidents in one night that no datacenter engine's CI would have |
| Pods | $65.23 total; showdown waste **51 %** ($1.50 of $2.96: boot+setup 390 s, cold-start 315 s, wait-setup 335 s, control 153 s); board waste 30 %; 10-18 night: $57 across 4 pods with 11 of 25 queued jobs pod-eligible | every pod re-downloads models, re-creates venvs, re-captures graphs; the "fake vLLM" rehearsal (05:36) nearly burned the paid run |
| Review cycle | #71: opened 10-17, landed 10-19 18:49 after a crash fix found by a *tainted* bench; full tier = rel + dbg shards + md5 (one md5 set "~4h43m", 02:57) + Shady review + fixes + regate + train | ~2 days floor per non-trivial PR, most of it waiting |
| Rounds killed at step 0 or after build | r13 pick 1, #94, #95, attn-one-launch, #90, r16 picks 1–2: six kills in 48 h | the kill rules work; what they reveal is that six tracks were opened on the laptop tier while the mission cell was unmeasured |

**Three concrete changes (OPINION):**

1. **Take timing and full-tier gates off the desktop.** One non-desktop GPU host runs every `--timing` job and every full
   tier; bella runs only sm_120-specific bits checks and day-mode smoke. *[omitted: how to provision that host.]*
   Expected: tainted A/Bs → 0, timing p90 wait 218 → < 30 min, the owner-hold class disappears.
2. **Gate per train, not per push; cap concurrent tracks at three; every track names its mission cell.** Fast tier
   (bits + clippy + fixtures) on every push; the full rel/dbg/md5 tier only on the train tip. Kill-in-queue rate 33 % →
   < 10 %. A track must say which showdown cell it moves (or which model it makes run) and its kill rule before a
   worktree is made; "≤ 8B dense kernel micro-round" is not a cell until §2.4 is measured.
3. **Pods keep state and run one rehearsed script.** A RunPod network volume (≈ $0.07/GB/month; 100 GB = $7/month)
   holding models, venvs, the NVRTC cache and the showdown kit removes boot+setup, cold-start and wait-setup (≈ 1,040
   of the showdown's 1,352 wasted seconds). Rehearsals run the *real* engines on bella (the 05:36 lesson), and
   `cost.json` is published with every run. Target waste ≤ 20 %.

*[omitted: a budget-allocation subsection (credit, balance, how to split it across pods). The rules that close it are kept:]*
- Hard rules: pods never compile, never idle > 5 min, never run a "board" without a one-line
  question, and every run's waste percentage goes in PROGRESS.

---

## 6. The three hardest questions for Haseeb

1. **Are you willing to spend the whole loop on one kernel for two to four weeks?** The 3–4× is a Marlin/CUTLASS-class
   GEMM problem on Hopper. Two Hopper attempts (gemmh, gemmh1) failed and the third (gemmh2) reached 12 % of wgmma
   peak and shipped only for gate+up. The swarm process that lands eight PRs a day is the wrong shape for this: it
   needs one builder, one card, ten sessions, and nothing else landing. If the answer is no, the datacenter tier is a
   benchmark you run, not a product you ship, and the roadmap should say so.
2. **What does the first paying user buy: the ITL p99 Axon wins, or the tokens-per-dollar it loses by 3×?** The showdown
   produced one unambiguous datacenter win (tail latency at C ≥ 32) and one unambiguous loss. A product can be built on
   the win (interactive agents, strict latency SLOs, reproducible bits) *today*; the loss takes step 2 to close. Picking
   neither is picking the loss.
3. **What is the burn per unit of mission progress, and what would you stop?** Six days, 1,998 jobs, 1,179 h of queue
   wait, $65 of pods, ~100 PRs, plus the agent time. Of the last ten merges, one (#71) changed the mission
   number by making C=128 possible; the rest moved laptop cells or added architectures with no GPU-resident experts.
   Name the three tracks you would close tomorrow, and the one you would fund with the entire budget. If you can't name
   them, the roadmap is a list, not a plan.

---

*Fable. Read-only; nothing built or run. All numbers from the cited files as of 2026-10-19 ~23:00 EDT. Arithmetic is
shown where a number is ESTIMATED; per-kernel H100 numbers do not exist yet and §2.4 is how to get them for $5.*
