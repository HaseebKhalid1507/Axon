# Experiments that didn't make it

Every NO-GO, every regression caught before merge, and every target that was missed, with the numbers and
the reason. Read this before starting a round: most of these took hours of builder time and GPU time to
learn. Sources: the foreman's log (PROGRESS.md), the builders' NOTES.md, the round specs and the PR bodies.
Hosts and protocol are in [methodology.md](methodology.md); dates are EDT, 2026.

"Code" says where the attempt lives now: **deleted**, **branch** (kept, not merged), **opt-in** (on main
behind a flag) or **parked**.

## CPU (overnight, 2026-10-13)

All on bella's CPU, Qwen2.5 0.5B Q4_K_M unless noted, 18 threads, `bench.py` interleaved medians.

| experiment | numbers | why it lost / lesson | code |
|---|---|---|---|
| **Attention token tiling** (2 or 4 query tokens share each K/V row load) | 4 rounds, pp2100: TB=1 783, TB=2 730, TB=4 695; base 756 | V reads dominate (7× per token), so sharing K saves ≤ 1–1.5 %. The tiles cost more than that | deleted; replaced by the V-sum-for-all-heads design that shipped in #1 |
| **Pool policy r2**: 50 ms windows, step down on loss rate | light load (a browser playing video): pp2100 672–702 vs adapt=0 739–759. `--load 6`: pp560 579 vs pool r1 744 | overreacted to light load | deleted |
| **Pool policy r4**: hill-climb on wall time per tick | quiet pp560 971 vs base 1011; `--load 6` tg 85 vs pool r1 151; `--load 12` tg 31 vs 101 | lost everywhere. Policies r2–r5 all lost to r1; what shipped is r1 plus observability (r6) | deleted |
| **Critic's decode prefetch fixes** (per-line prefetch bursts, chunk cap) | dec4: 1 thread 56.6 → 49.2. dec5 (branch-free bursts): 1 thread 44.9 vs r2 51.2; 18 threads tied | data-dependent hint loops, and the cap turned prefetch off at 18 threads. Shipped r2's kernels plus a 1 MiB clamp on `AXON_PREFETCH` | deleted |
| **Hard P/E-core pinning** (`AXON_PIN=1`) | +3–4 % quiet prefill, **−19 %** under `--load 6` | pinning fights the scheduler when the machine is shared. The nested-run hang it exposed was fixed and shipped | parked (not on main) |
| **#9b parallel exp** above 2048 positions | pp2100 823 → 791 (−4 %); **tg128@8k 79.8 → 64.4 (−19 %)** | the extra exp pass plus a fork/join costs more than the serial exp | deleted |
| **#12 long-context prefill query tiles** (2 queries per task above 3k) | pp2100 866.7 vs 866.2; pp8400 388 vs 417 (−7 %, ranges overlap) | perf at pp8400: long prefill is FP-compute-bound in attention (pool closure 31.5 %, QK 25.8 %), not K/V-bandwidth-bound. Bit identity forbade FMA there, which led to N3 in #2 | deleted |
| **Prompt-lookup low-acceptance gate** | an offline simulator replaying the exact greedy streams: no gate beats "no gate" under the "worst prompt ≥ −1.5 %" rule. 8 rounds, K=7: story 194 → 186 (−4 %) | story text loses inside the forward passes, not to the gate. Drafting stays **off by default** (`--draft 7` opt-in) | gate deleted; drafting opt-in |
| **#14 decode V-sum whole-line tasks** | tg@8k: head 80.8 / vsum 77.9 / split2 81.9 / split1 81.5; 1.5B 37.9 / 37.9 | decode at 8k is DRAM-bound (the critic's arithmetic); line sharing and misaligned KV weren't the cost | deleted |
| **#16 verify-batch kernels** (q5_0/q8_0 batch kernels, ~20–40 % fewer uops) | spec bench, 6 rounds, K0/K7: prose 220/218, story 172/174, code 424/412. Verify nb4 ~8.4 ms both | at T=8 the batch is compute-bound on the E-cores (half-width SIMD) while the P-cores wait. What did help was finer chunks: 16 chunks per thread for nb ≤ 9 (b6ebad3, shipped) | deleted |
| **r8bi** (bit-identical interleaved K-quant, PR #3) | 1.5B, 5 rounds: pp560 +13 %, tg@1 +26 %, but pp2100 −4 %, tg128 −3 %. An earlier round had +27 % pp560; the loss since round 3 was never pinned down (LLC misses ~+20 %) | mixed, and superseded by #2's numerics r8 (1.5B pp560 734) | branch `r8bi`, PR #3 closed |
| **N2 r8q, first layout** | Q5_0 at 272 B per 8-row block vs 176 B: +55 % decode bytes | layout too fat. The compact r2 (176 B) shipped in #2 | deleted |

**Regressions caught before merge (night 1):**
- The numerics branch cost 0.5B tg@1 −11 % (27.9 → 24.9) and r8bi round 2 cost 1.5B tg@1 −13 %. Root cause:
  4 KB pages (dTLB misses 3K → 1.0M) plus L1-filling T0 prefetch bursts. Fixed with a 2 MB huge-page arena and
  T1 prefetch, which then gave +41–54 % (in #2).
- b8a04b6 turned drafting on by default against its own message. Found by a benchmark at 14:15 and reverted
  (b6ebad3).
- A test built activations with `as i8 - 127`, which panics under debug assertions. CI caught it; the verify
  then ran release only. Since then the debug suite is a gate.

## GPU prefill (GEMM)

| experiment | when / host | numbers | why / lesson | code |
|---|---|---|---|---|
| **Inexact (llama.cpp-style) epilogue on G3's tiles** | 10-14, bella | only 10–19 % faster than the exact epilogue | the gap to llama.cpp was GEMM efficiency (~20 % of int8 peak), not exactness. Fixed by redefining the canonical order (#12) | not built further |
| **gemm2 smem padding (G2_ROW 272) / precomputed copy pointers (G2Copy)** | 10-14, bella | gemm2s gate 42.1 → 40.9 TMAC/s | slower | deleted |
| **gemm2 v3, first version** | 10-14, bella | 7B decode ×0.908 (8 rounds) | the new canonical order cost decode until the scale loads were hoisted and the R chains run on lanes of warp 0 in parallel. Fixed before #12 merged (7B ×1.045) | fixed |
| **r4 stage 3 `gemm3`** (TMA / `cp.async.bulk`, one request per row) | 10-15, bella | −2 % / −55 % | per-row TMA request overhead. A tensor-map TMA (2 requests per stage) was the untried next step | p7 branch, not on main |
| **r5 P3: per-shape 4-warp attention rule** | 10-15, bella | +0.6–1.9 % end to end (inside the spread); kernel-level q3-8B shape 1.045 (−4 %) | failed the spec's go | deleted |
| **r8 GEMM pipeline**: mbarrier stages (gemm2m) + producer warp (gemm2p) | 10-16, bella, job 1016-054529-8777 | sustained TMAC/s vs gemm2s: gate nb559 −16 % / −28 %, q nb1050 −5 % / −13 %, down nb1050 **+4.5 %** / −3 %. E2E 3B pp560 10661 → 9759 / 9659 (−8.5 / −9.4 %), q3-8B −5.4 / −11.4 %. tg unchanged | NO-GO by its own rule (< +6 %). The ablation behind the design copied from an L2-resident buffer; the real kernel streams from DRAM and is bound by copy latency, so the barrier wasn't the critical path. Untried: deeper stages, a 2-warp producer, and measuring on H100 first | branch `feat/r8-gemm-pipeline`, `AXON_GEMM2M` (default 0), no PR |
| **"Unpack-bound" hypothesis** (r8-unpack spec) | 10-16, bella SASS + ablation | an int8 repack of the weights gains ~0 on bella; nibble unpack, epilogue and M-term are free; scale IMADs cost 17 % | the GEMM wasn't unpack-bound. Amortizing scales is impossible under the contract | design only |
| **Hopper `gemmh` (SS wgmma, weights unpacked into smem)** and the S2 speed gate | 10-16, H100 PCIe | gate nb1050: gemm2s 70.2, gemmh 56.6, gemmh1 38.7, gemmh2 83.8 TMAC/s against a ≥ 120 gate; kv 512 rows: gemmh2 21.6 vs gemm2s 35.6 | smem-bound (SS) and issue/latency-bound (RS) at ~12 % of wgmma peak. The fused gate+up wgmma (+18 %) shipped (#33); plain products are routed only where measured faster | gemmh/gemmh1 in tree as probes; routing in #33 |
| **M6: fixed-chunk two-level K chain in every GPU product + split-K for the narrow small-nb products** (NUMERICS_VERSION 2, PR #148) | 10-26, RTX 4090 pod (jobs 1026-202840-3e4b, 1026-204007-20fc) | vs main ae1e26b, 5 rounds: 7B pp512 ×0.725, pp2100 ×0.738, tg128 ×0.965; 0.5B pp ×0.86, tg128 ×0.937. 7B `pool_step_cost` d600 (3 rounds): nb 128 17.31 → 19.80 ms (**+14 %**; the rule's −10 % target is an H100 number, the 4090 never got near it), nb 32 +4 %, nb 1/8 +0.3 % | the kill rule failed on its 4090 no-regression clause (all six cells worse than −2 %), so the H100 was never booked. The chunked order (extra accumulators, local-memory chunk totals, new spills) is paid by every large-nb kernel; whether split-K saved anything on the narrow products is unknown (no `AXON_KSPLIT=0` run; ESTIMATED, not profiled). An f64-graded gemma3 test from the work is kept as an idea (its bars weren't written blind). [log](log/2026-10-26-m6-split-k.md) | branch `perf/m6-split-k`, PR #148 closed |
| **Hopper W4A8 round 2, upper-bound probe**: warp-specialized persistent TMA + RS wgmma Q4_K × Q8_K, integer-only accumulate, NO exact epilogue (deliberately inexact; T3, PR #150) | 10-26, H100 SXM (job 1026-222543-3f73) | gate 3584 × 18944 nb 1024: **224.8 TMAC/s** (224.3–224.9) against the pre-registered ≥ 240; gemmh2 116.5, gemm2t 118.5 in the same rounds; probe / gemmh2 1.80–2.04× on all 4 shapes × nb 128 / 1024; every bit check PASS; wgmma RS-n64 peak ~965 raw | below the bound that, less the ≥ 25 % exact-epilogue issue cost, was needed to reach the roadmap's 180 (240 × 0.75): round 2 is **shelved per the rule**. The structure alone is ~1.9× gemmh2; an exact kernel on it would land at ≤ ~169 (ESTIMATED: 224.8 × 0.75). First run died on an int4 load of the 4-byte-aligned activation sums (fixed). ≈ $2.40 of H100 (0.60 pod-hours). [log](log/2026-10-26-w4a8-probe.md) | branch `feat/t3-w4a8-probe`, PR #150 closed |

## GPU decode

| experiment | when / host | numbers | why / lesson | code |
|---|---|---|---|---|
| **r4 decode track** (persistent GEMV grid, rmsnorm fused into wo/down epilogues, gate_up quantize with per-row atomic tickets) | 10-15, bella | stacked in night 3: 7B tg128 **89.8 → 76.8 (−14.5 %)**. Bisected to this stage alone: default −14.3 %, all off −3.1 %, `PERSIST=1` 24.5 tok/s | dropped from #22, then reworked (below) | — |
| **dec2 rework** (interleaved norm epilogue, next-wave L2 prefetch without the persistent loop, smem back to 4096 B) | 10-15, bella, vs 53bbd12 | 7B: base 117.1 / host code only 117.2 / kernels off 113.4 (−3.2 %) / persistent 109.7 (−6.3 %) / norm fused 109.9 / next-wave prefetch 110.0 (−6.1 %) / all on 99.9 (−14.7 %). 3B: off −6.6 %, prefetch −12.3 %. vs main 4da08e1: every variant 100.4–110.9 vs 117.2 | per-kernel times were equal; the loss was in the kernel-to-kernel PDL overlap. A persistent grid blocks the next kernel's blocks from becoming resident, a norm in the wo tail adds 8 µs where a separate 5.8 µs norm overlapped, and per-row atomic tickets serialize. The smem-carveout hypothesis was tested and rejected. **Don't retry** (r5-decode §5) | deleted |
| **r5-decode "do not retry" list** | 10-15, bella | host/readback pipelining ≤ 0.8 %; unrolling `down` 0…−17 %; 4 rows per block on 7B shapes or `down` 0…−9 %; 4-lane Q4_K 0 % isolated, sustained and in-graph; dropping `down`'s L2 prefetch costs ~1.2 µs on 1.5B | 7B GEMVs are at 93 % of the DRAM ceiling; the rest is the 140 W cap, which no kernel change buys back | design notes |
| **r5 decode P3a** (batched partial loads in `attn_cmb`) | 10-15/16, bella | two attempts, no kernel gain | — | dropped from #30 |
| **gate_up4 at 64 registers on 7B** | 10-15, bella | 7B −8 % (spills) | the narrow-row rule keeps R=2 on 7B; only rows < 1 KiB use gate_up4 (#30) | rule in #30 |
| **lcdec exploration: grid y = blocks** | 10-16, bella | 0.5B pos-8 step with a 32k cache: 1.207 → 1.472 ms | killed; empty CTAs at short positions. The 4-warp `attn_blk` (in flight) avoids it | deleted |
| **gemvm16 tuning: two blocks of prefetch (`GVM_PF2`), 4 resident blocks per SM (`GVM_MINB=4`)** | 10-18, RTX 4090 (job 1018-034145-34d7) | 7B pool step, depth 600, ms at nb 1 / 8 / 16: plain 6.50 / 7.15 / 8.70; PF2 6.88 / 7.67 / 9.46; MINB4 6.80 / 7.22 / 10.01; both 7.27 / 8.09 / 10.66 | more data in flight per warp cost more occupancy (registers) than it bought; gemvm16's ~620 GB/s is per-block work, not latency hiding. **Don't retry these knobs**; the lead is in fewer, fuller launches (#86) | removed (#86) |
| **Gate\|up on gemm2s_q4_k_gu at nb 9–16** (r13 pick 1) instead of the fused gemvmx | 10-18, bella sm_120 (1018-060449-b71d), avante RTX 4050 sm_89 (1018-060449-ca07) | `cuda_gu_bench`, 7B gate\|up shape, weights cycled from DRAM. gemm2s_q4_k_gu ÷ gemvmx time at nb 4 / 8 / 9 / 12 / 16: bella 1.63 / 1.54 / 1.44 / 1.35 / 1.16 (second pass 1.65 / 1.56 / 1.43 / 1.36 / 1.15); avante 1.09 / 1.11 / 1.11 / 1.12 / 1.19. bella 7B pool step at nb 16: 21.32 / 21.43 ms (gemvmx) vs 21.93 / 22.05 (gemm2), +0.6 ms | gemm2 is slower at every nb ≤ 16 on both cards, so the kill rule (≥ 12 % faster, ≥ 0.3 ms step) fails. The premise compared gemm2 at nb 32 (2.83 ms) with gemvm16 at nb 16 (3.4 ms). But gemm2's cost is not flat below its 64-token tile; likely because the zero-filled padding columns still pay the tile's MMA and smem work (not profiled). And gemvmx16 at nb 16 already fused gate and up. **The 4090 was not measured** (pods out of budget). Retry only if a 4090 run shows gemm2 ahead at nb 16 | branch feat/c-curve-3 only (`AXON_GEMVM_GU_MIN`), not merged |

## GPU attention

| experiment | when / host | numbers | why / lesson | code |
|---|---|---|---|---|
| **swiglu_quant_b with 128-thread blocks** (r3) | 10-14, bella | e2e no gain, slightly slower everywhere | — | reverted |
| **Loosened `assert_graded`** (r3, #19): not a rejection but a judgement call | 10-14, bella | one chaotic position KL 1.219 (next worst 0.107) | the test was loosened to allow KL > 0.5 at ≤ 1 % of positions, none > 2; an injected mask bug gives 14–48 positions > 0.5, so it still catches real bugs | on main |
| **lcpre: ping-pong prefill attention** (two warp groups, PV of block b−1 during scores of b) | 10-16, RTX 5090 pod3 + avante | 5090: 7B 1121@16k 1836.9 → 1885.4 µs (+2.6 % slower), @32k +3.0 %; q3-8B +2.4 / +2.8 %. avante sm_89: −2.3…−4.6 % (faster). p_64 on 0.5B: 5090 −10.5 %, avante +9 % | NO-GO (rule: ≥ 2 % gain on the 5090). The wins and losses of the 64-row variants track grid geometry (CTAs vs SMs), not ping-pong. Lead left: pick rows per CTA from (CTAs, SMs); estimated −8 % attention on bella's 46 SMs, unmeasured on bella | deleted (uncommitted worktree) |
| **attn_blk_mq at 128 queries** (r12 perf pick 3, #84): one CTA loads each K/V tile once for 16 queries of one sequence, aimed at chunk-128 admission at depth | 10-18, RTX 4090 pod (1018-043837-1ff9) | mq + attn_cmb_b over 128 queries at 24k: 1964 µs vs attn_fa_b 1486 µs = **0.76×** (vs attn_fa_bs 969 µs: 0.49×); kill rule was ≥ 1.8× | the per-(query, head, block) partials (512 B of f32 per row-block at hd 128, written and read back by the fold) cost ~3.5× the K/V the tile saves, while attn_fa_b folds in registers; the spec's estimate counted K/V, not partials. A no-partials chained-fold variant is written up in #84, not built | shipped only for ≤ 32 queries of one sequence (#84, `AXON_ATT_MQ=0` off), where it is faster: bella serve admit at chunk 32 TTFT −6.4 %, ITL p99 −8.6 % |

## Offload, KV, memory

| experiment | when / host | numbers | why / lesson | code |
|---|---|---|---|---|
| **Stream weights for decode** (`AXON_STREAM_DECODE=1`, AirLLM's move) | 10-14, bella, 32B 28/64 | 2.07 tok/s vs 4.8–5.8 plain split decode | PCIe and the CPU read the same DDR5; per-token streaming can't beat computing on the CPU (also research-bigmodels §6 "dead end 1") | opt-in env experiment on main |
| **Plain split prefill without streaming** | 10-14, bella, 32B | pp560 47.7 vs llama.cpp op-offload 208 | not a rejection: it is why #16 exists | `AXON_STREAM=0` |
| **q8 KV without the key shift** | 10-14, bella | 5–15× the noise of f16 | Qwen's \|k\| ≈ 200 bias channels set every row's scale | replaced by the shift (#11) |
| **Hadamard rotation of K / K+V for q8** | 10-14, bella | 0.5B @2048 KL 4.9e-2 → 1.2e-2, but short contexts worse / noisy | small and inconsistent | deleted |
| **q8 KV for speed** | 10-14 (#11) and 10-16 (r9), bella | decode unchanged, prefill −4–7 % (#11). At 32k: 7B tg 70.9 (q8) vs 72.4 (f16) | half the bytes, same time: decode attention was latency-bound at 350 of 648 GB/s. lcdec (in flight) measures q8 +13.8–23 % once attention is restructured | `--kv q8` stays a memory feature |
| **Serve pool of N slots costs N+1 caches** | 10-16, bella (#42) | 7B: 2 slots fit only at 30720, not 32768 | a known gap, not an experiment; follow-up "drop the N+1 cache" (roadmap Scale #2) | fixed by feat/serve-one-cache (docs/log/2026-10-17-serve-one-cache.md) |

## Startup

| experiment | when / host | numbers | why / lesson | code |
|---|---|---|---|---|
| **Tokenizer speedup, round 1** | 10-15, bella | gain within noise | conflicted with the llama/serve tokenizer work; redone in #41 (57–66 → 26–36 ms) | dropped from #21 |
| **Pinned upload ring, always on** | 10-16, bella | ring forced vs ring off, cold: 7B 830 → 868 ms, 8B 854 → 889 ms (+35–40 ms); warm it wins (7B 612 → 532 ms) | a cold load has ~30 % of the file resident at upload time, so the preads are disk-bound and lose to the fault-driven copy. Shipped with a ≥ 75 % residency rule (the threshold is ESTIMATED) | rule in #41 |
| **MADV_WILLNEED ahead of the ring's preads** | 10-16, bella | cold 7B 871 → 890 ms | worse | deleted |
| **Register the private mapping / populate + register** | 10-16, bella | register 910 ms; 1 GB populate 120 ms + register 154 ms | slower than the ring | deleted |
| **0.5B with the ring** | 10-16, bella | 282 → 295 ms (pin cost) in an earlier build | fixed in a later build (0.5B regression gone) | — |

## CPU pool and models, later rounds

| experiment | when / host | numbers | why / lesson | code |
|---|---|---|---|---|
| **Decode cap at physical P-cores on an SMT hybrid** | 10-16, avante (i5-13420H) | tg128 1.5B: no cap 59.19, 4 physical P 53.43, 8 logical P 52.92; 3B 29.32 / 27.13 / 26.47 (**−7…−10 %**) | the cap helps only non-SMT hybrids (bella). Shipped with no cap on SMT-on hybrids (#32) | rule in #32 |
| **Splitting the VNNI dpbusd chain in two** (#20) | 10-15, bella P-core | no gain | already ~1 vpdpbusd per cycle | not included |
| **E-core tail split on small models** (#20 run_split) | 10-15, bella | +5.2 % 1.5B, +2.3 % 3B, but −1 % 0.5B, −0.6…+0.8 % on 0.6B/1.7B | kept: net positive | on main |
| **Qwen3 norm with a block handshake (v3)** | 10-14, bella | 4B: v3 2–3 % slower than no-norm; v5 (norm in the attention prologue) 1.015× | v5 shipped (#18) | deleted |

## Speculation

| experiment | when / host | numbers | why / lesson | code |
|---|---|---|---|---|
| **Draft model, sampled** (temp 0.8) | 10-16, bella | 7B tg128: main 115.4, K=3 116.3 (+0.8 %), K=5 92.7 (−19.7 %), K=7 81.7 (−29.2 %); acceptance 25.6 / 16.7 / 12.5 % | missed the +10 % gate. Acceptance is exact-match per seed, not rejection sampling (research-sched §4 P4) | draft model shipped (#27); the gain is greedy only |
| **Prompt lookup, sampled** | 10-16, bella | 7B 115.4 → 107.2 (−7.1 %) | low acceptance under sampling | off by default |
| **Draft model K=5/7 greedy** | 10-16, bella | 7B +32 % / +23 % vs +37 % at K=3 | each guess costs a drafter step | default K=3 with a drafter |
| **Batched speculative decoding in `serve`, stage 1** (r14, #90): draft rows for every greedy request in the batch, K (0–3) per request per step from a goodput gate (step costs fitted at startup, corrected online, per-request α) | 10-18/19, bella exclusive; run 1 1018-101159-4afd (545ac02), run 2 1018-204023-96fd TAINTED, **run 3 1019-013704-5285** (3541ea2, foreign GPU max 19 %, mean 0.4 %) | 7B Q4_K_M + 0.5B drafter, agent prompts, greedy, medians of 3. Agg tok/s off → on: d600 C=4 328.9 → **409.8 (+24.6 %)**, C=8 466.1 → 491.4 (+5.4 %); d4k C=4 276.9 → **331.7 (+19.8 %)**, C=8 391.0 → 396.3 (+1.4 %; ITL p99 +14.3 %); **d16k C=4 185.0 → 179.0 (−3.2 %), ITL p99 23.8 → 39.9 ms (+67.6 %)**, α 0.54. md5 120/120 identical across off / on / main (run 1) | kill rule (spec §5): ITL p99 ≤ +15 % and no drafting cell > 3 % slower both **failed** at d16k. The short-context wins are real; the gate is what failed: at depth it keeps drafting when plain decoding is faster, and the 3541ea2 fix (learned round time, α prior 1, one probe per fresh request) didn't drive K to 0. Lesson (first write-up, **corrected 10-24**): "the gate needs depth in its model" is wrong: the gate already fitted `t = a + b·nb + c·Σd` at depths 512 / 4k / 16k (CODE). The post-mortem in [research/spec-gate-v2-20261024.md](research/spec-gate-v2-20261024.md) finds (ESTIMATED from the branch's logs): optimistic priors relearned on every server restart (the harness restarts per run, so each run was the learning transient); no ITL term, so mixed-K rounds made K=0 rows pay a K=1 round (ITL 1.68× predicted, 1.68× MEASURED); and serve verify re-reads each slot's KV per row (`attn_blk_b`), so a round costs ~(1+K)× plain at depth and drafting can't win at 16k at any α. A gate's best outcome at 16k is 0 %; a win at depth needs the multi-row verify (stage 2) | branch `feat/serve-batched-draft` (PR #90 closed 10-19, not merged); a second attempt with a depth-aware gate is Scale item 10 |

## Serving scheduler

| experiment | when / host | numbers | why / lesson | code |
|---|---|---|---|---|
| **Hybrid step** (roadmap item 1 (b), #119): a serve step is **one pass** of the decode rows and the prompts' chunks instead of a chunk pass then a decode step; `--prefill-chunk` = the pass's tokens rows included, rounded to 64-token tiles; bits unchanged (identity A; md5 0/20, serve A/B one digest ×6) | 10-20 bella / avante / RTX A5000 pod; **kill-rule cell 10-25 on a clean secure RTX 4090 pod** (sm_89; `mix64-ecdb1d0`, 2 rotated rounds) | **mix C = 64, 7B Q4_K_M, `--parallel 64 --ctx 8192` (MEASURED, medians of 2):** main 6e3b19c agg **1,688** tok/s, ITL p99 72.3 ms, ITL p50 16.1, TTFT p50 486 ms, TTFT p99 5,785, slo_ok 0.696; hybrid (ecdb1d0, `AXON_HYBRID=1`) agg **1,699 (+0.65 %)**, ITL p99 **39.0 (−46 %)**, ITL p50 26.6 (+65 %), TTFT p50 **1,734 (3.6× worse)**, TTFT p99 6,620, slo_ok 0.68; `AXON_HYBRID=0` 1,657 (−1.8 %), ITL p99 72.9, TTFT p50 502. **7B loop C = 8 (MEASURED):** clean A5000 agg 456.2 vs 430.4 (+6.0 %), ITL p99 54.8 vs 63.9, TTFT p50 161 vs 188; admit ITL-during p99 96.2 vs 157.5 (**−39 %**); long-prompt loop +11.3 %; bella runs TAINTED +2.9 / +6.4 %. 1.5B avante loop +2.1 %, ITL p99 −49 % | **Both kill rules failed:** roadmap item 1 needs mix C = 64 agg ≥ +15 % with ITL p99 ≤ today's (got +0.65 %); the PR's §8 needed loop C = 8 ≥ +8 % (got +6.0 % clean). What it buys is smoother decode (ITL p99 −46 %, admit ITL-during p99 −39 %) paid with prompt latency (TTFT p50 3.6× worse at C = 64) and flat throughput. Why (ESTIMATED, not profiled): at C = 64 the step isn't bound by the second weight read the merge saves (64 decode rows' attention at depth and the chunk's own compute dominate), and with the budget counting the rows, 64 rows leave a 256-token pass ~192 prompt tokens, so admission slows and TTFT grows. A future variant would have to (1) not charge decode rows against the prompt budget (or size the pass from measured step cost: item 1 (d)), (2) show a step that *is* weight-read-bound before claiming throughput (step anatomy, item 1 (a)), and (3) be judged on the same mix C = 64 cell with TTFT p50 in the kill rule, not just agg and ITL p99 | branch `feat/hybrid-step` (PR #119 closed 10-25, not merged); (c) shortest-first admission and (d) the measured step budget landed in two-pass mode in #131 (10-25, superseding #126; [log](log/2026-10-25-sched-budget-main.md)): (d) is the default (`--step-ms 60`); (c) is opt-in (`--admit-order sjf`) after missing its kill-rule clause "short TTFT p50 ≤ 0.5× fifo" on the clean bella cells (1.5B C=32 0.56×, sdab15 C=32 0.79×), pending the H100 cells (#139) |

## Multi-GPU, serving targets

| experiment | when / host | numbers | why / lesson | code |
|---|---|---|---|---|
| **Tensor parallel on PCIe without P2P** | 10-16, 2×4090 pod | all-reduce loop 2.90 ms/token, ~2.24 ms of it comms, against a 2 ms gate | not worth it without P2P. Pipeline parallel shipped (#34); P2P is opt-in and **untested** | not built |
| **S2 batching targets** | 10-16, 4090 + bella | 7B C=8 4.3× (target 5×); 0.5B C=8 below the 3.5k gate; bella 7B C=16 4.4× | shipped anyway. The estimate counted decode steps only; closed-loop clients spend ~18 % of wall time in whole-prompt prefill steps (ESTIMATED, #44) | #38 merged; #45 chunked prefill open |
| **Chunked prefill, small chunks** (#45) | 10-16, 4090 pod1 | 7B: chunk 128 ITL 31.5 ms, TTFT +139 %; chunk 32 30.6 ms, TTFT +828 %; default 512: 62.5 ms, +21 % | a pass at ~24k depth has a floor (~25 ms 7B, ~10 ms 0.5B, ESTIMATED), whatever its token count; suspected few-query `attn_fa_b` latency | default 512 in #45 (open) |

## Process failures (cost time, not code)

- **Dead loop, ~6 h** (10-13 03:45 → 09:55): the foreground hit a 512-tool-call cap and nothing woke it.
  Fixes: the cap was raised, bench-waiter agents became event sources, and later the heartbeat.
- **Desktop OOM kills ×2** (10-13 11:19, 11:55): `/tmp` (tmpfs) at 11–14 GB plus 3 builders loading models in
  parallel, plus (the second time) a leaking browser tab. Since then: no `/tmp` for anything big, and a model lock.
- **A 32B CPU-only bench OOM-killed** (10-14 02:51). Rule: no 32B CPU runs while builders are active.
- **Rebase reported clean without compiling** (#13); **a verify on a live worktree** during a rebase (g4a). Both are now rules.
- **Lock hog / GPU at 25 %** (10-15): a 2.5 h chain held the model lock as one job, and the exclusive locks
  serialized CPU-bound suites. That led to one step per lock, a shared test lane, and then axon-run.
- **A README-dependent test** broke when the docs changed (#36): tests now read a frozen fixture.
