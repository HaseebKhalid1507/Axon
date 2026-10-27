# Design note (Checkpoint 0): a production W4A8 prefill path on Hopper

**Status: proposed; pre-flight passed (10-27). Nothing is built, nothing is booked; Shady reviews before the build.** Written 2026-10-27 on `feat/r11-w4a8-gate` (dd1e7fa, base
main 0c23a87), after rule P (#165) came back GO. Input: [../log/2026-10-27-w4a8-rule-p.md](../log/2026-10-27-w4a8-rule-p.md).
Labels: **MEASURED** (job cited), **ESTIMATED** (arithmetic shown), **CODE** (file on the branch or main), **OPINION**.

**Lead's decisions (Jawz, 10-27), recorded here:**
- **Restart item 2: yes, conditionally** on a free local check of how much feed GEMM time falls in passes ≥ T.
  Done: §4, "Pre-flight". Every T ∈ {128, 160, 192} keeps the ESTIMATE's low end above the +8 % bar, so the build
  may proceed. Shady reviews this note first.
- **No new numerics tier: accepted** as the contract (§2).
- **Q6 copy option (A): acceptable** if the bits stay identical, the VRAM is accounted for, and it sits behind the
  same flag (§3.3).

---

## 0. In one paragraph

- **What:** turn the measured exact probe (`probe_w4a8x_n64`, `probe_w6a8x_n64`) into the kernels the serve's
  prefill uses on sm_90a, for Q4_K and Q6_K products on passes of at least T tokens. T is a threshold measured in
  the first session.
- **Numerics:** the exact variants are **bit-identical to gemmh2_q4_k and gemm2(s)_q6k8 on every output**
  (MEASURED, H100, job 1027-084257-71f3). So this is **not a numerics change**:
  - no new tier and no `NUMERICS_VERSION` bump;
  - no identity-string change;
  - the md5 gate is the correctness proof.
- **Rollout:** behind `AXON_W4A8=1` (off by default) until a pre-registered end-to-end serve A/B on an H100 says GO;
  after that, on by default for sm_90a.
- **ESTIMATED effect:** about +13.5 % aggregate at mix C = 128. The low end is **+8.7 to +10.4 %**, re-derived
  from the measured bound on pass sizes (§4); the high end is +18 %. Kill bar +8 %.
- **Cost:** 3–5 builder-days and ≤ $10 of H100 (cap).

## 1. Scope: which products, which types, which hardware

**In (v1):**

| product (7B shapes) | weight type | today's kernel at nb ≥ ~256 (CODE, batch.rs `bgemm`) | exact probe vs it, nb 384 / 512 (MEASURED) | serve epilogue |
|---|---|---|---|---|
| fused gate + up, 3584 → 2 × 18,944 | Q4_K | `gemmh2_q4_k_gu` | 1.689× / 1.701× | silu(gate) · up |
| down, 18,944 → 3584 (Q4_K layers) | Q4_K | `gemmh2_q4_k` | 1.554× / 1.561× | mode 1: out += W x (+ bias) |
| down (Q6_K layers of Q4_K_M) | Q6_K (by its prefill copy) | `gemm2s_q6k8` / `gemm2_q6k8` | 1.110× / 1.206× | mode 1 |
| q and wo, 3584 → 3584 | Q4_K | `gemmh2_q4_k` | 1.581× / 1.659× | q: mode 0 (+ bias); wo: mode 1 |

- **One plain Q4_K kernel covers down4, q and wo** (rows ≥ 3584: `gemmh_plain`'s row rule). The gu kernel and the Q6
  kernel are separate entry points of the same template.
- **Models:** any whose prefill products are Q4_K / Q6_K at ≥ 3584 rows on an sm_90a card: the Q4_K_M family
  (Qwen2.5, Qwen3, Llama 3.x).
  - The tests and the A/B use Qwen2.5-7B-Instruct-Q4_K_M, the anatomy's model.
  - Other models are covered by the bits tests and by the md5 gate's model set.

**Out (v1), each with its reason:**
- **k / v (512 rows):** 28 × ceil(nb/64) tiles is under one wave at the serve's nb. Today's routing already keeps
  them off gemmh2 (`gemmh_plain`). They count at 1 in the estimate.
- **Q5_K, Q8_0, Q4_0, the IQ types:** no measurement. Each needs its own unpack and rule. A separate change if a
  model mix asks for it.
- **The decode / verify GEMVs and batches < T:** they keep today's kernels. Same bits either way (§2), so routing by
  T is a performance choice only.
- **MoE expert GEMMs:** different grids (#154 track).

## 2. Exactness: the contract stays the default one

**The contract (accepted by the lead, 10-27):**
- The W4A8 path is a **faster route to the same bits**, under the default (exact) contract.
- Every output equals today's kernel's output for the same product: `gemmh2_q4_k_gu`, `gemmh2_q4_k` (modes 0 / 1,
  with or without bias) and `gemm2(s)_q6k8`. Hence it also equals `gemv_ref`'s order (identity C).
- There is no new `--numerics` tier, no `NUMERICS_VERSION` bump and no identity field. Routing (`AXON_W4A8`, the
  threshold T, the GPU) can't change bits.
- **Held in place, every round:**
  - the bits tests on the H100 on every output (ragged shapes plus the 7B shapes);
  - the md5 gate on the H100, `AXON_W4A8=1` vs `=0` vs main, n/N identical;
  - at Checkpoint 2, text_sha and completion_tokens equal for every request.
- **Any difference is a bug.** It is fixed, never toleranced away. If the contract can't be kept, the change doesn't
  merge.

Why it holds:

- **What the contract requires (CODE, `docs/determinism.md`; identity C in `docs/design/determinism-modes.md`):**
  every product kernel equals `gemv_ref`'s order. That is: an exact integer block sum, then one fixed float chain
  over the 256-blocks in order.
- **gemmh2 already is W4A8.** Q4_K weights times Q8_K int8 activations, integer S and M per 256-block, then
  `v = d·S − dmin·M; acc = fma(dk, v, acc)`.
- **The exact probe does the same per block.**
  - What differs is only how the integers are formed: RS wgmma from registers, a persistent warp-specialized CTA,
    M on the tensor core, and Q6_K's scales as digit planes.
  - Integer sums are exact and order-free, so the float chain sees the same inputs in the same order.
- **MEASURED:** 0 mismatches against gemmh2_q4_k and gemm2s_q6k8 on every output:
  - five ragged shapes per kernel (1–17 blocks, rows and tokens off the tile, 1–7 CTAs, the stage ring wrapping
    across tiles);
  - every timed 7B cell at nb 384 and 512.
- **So the production path is identity-C material, in the exact (default) contract:**
  - **no `--numerics` tier.** The ADR's fast mode is still "proposed", and nothing here needs it;
  - **no `NUMERICS_VERSION` bump** and no identity field;
  - routing by T (or by `AXON_W4A8`) can't change bits;
  - **the md5 gate is the proof:** `gen` and `serve` text with `AXON_W4A8=1` vs `=0` vs main, same H100, must be
    n/N identical. Any difference is a bug, not a tolerance question.
- **What the build must reproduce exactly (the probe skipped these):**
  - **gu's tail:** `__fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), u)` on the same two floats.
    - Same compile options (KERNEL_OPTS, `--ftz=false`), so `expf` lowers the same way.
    - Pairing as gemmh2_q4_k_gu: a 128-row tile of 64 gate rows (consumer 1) and the same 64 rows of up (consumer
      2), with gate's acc exchanged through smem at the tile end.
  - **mode 1 and bias:** `gm_pre` / `gm_put`'s ops. Every load of `out` happens before any store, as gemm2's
    epilogue does it (prefill.cu).
  - **The Q6 copy's d:** f32 = h2f of the f16, as Q6K8.
- **What would break it (OPINION):** any split of K across CTAs, any atomics, or a per-tile (instead of per-256)
  drain. That last one is the skeleton's cheat, and it is why the skeleton stays test-only.

## 3. Design deltas from the probe

1. **Kernel file.**
   - New `src/cuda/w4a8.cu` in KERNELS, inside `#if __CUDA_ARCH_FEAT_SM90_ALL`: off sm_90a it compiles to nothing,
     as hopper.cu does. Entry points `w4a8_q4_k`, `w4a8_q4_k_gu`, `w4a8_q6`.
   - The probe stays test-only. The skeletons are deleted from the product file.
   - Shared hot files are claimed on the forum before editing (`src/cuda/*.cu`, `batch.rs`, `qwen2.rs`, `mod.rs`).
2. **Routing (`bgemm` and the gu call site in `batch.rs`).**
   - Taken when `AXON_W4A8` is on, the module is `hopper()`, nb ≥ T, and the product is in §1's table.
   - Grid = #SMs (persistent). Tensor maps are built per call, as gemmh2's are today.
   - `gemm_cost`'s wave model gains the persistent kernel's cost, so the wave-split planner sizes passes right.
3. **The Q6_K weight copy: the one real open choice.** Small passes (< T) still read the Q6K8 copy's scales
   (`gemm2s_q6k8`), and the W4A8 kernel reads digits. Two options:
   - **(A) Extend the copy:** `[q 256][sc 16][d 4][digits 24][pad 4]` = 304 B a block.
     - Q6K8_BYTES becomes a parameter of `g2_load_w` (the bytes and order are unchanged, so the bits are too).
     - Two TMA boxes of 160 + 144 B, because a box is ≤ 256 elements wide.
     - VRAM: +16 B per 288-B block, i.e. +5.6 % of the Q6 copy; 7B ≈ +55 MB (ESTIMATED: 13 Q6 down layers ×
       3584 × 74 blocks × 16 B).
   - **(B) Keep Q6K8 as is**, and have the producer's 3 builder warps compute the digits from the 16 scales of each
     staged row, writing them to smem next to the raw rows.
     - No layout change and no shared-file risk.
     - About 2048 digit decompositions per stage on 96 threads. The producer has idle issue slots, but Q6 is
       already the weak shape.
   - **Pre-registered pick:** time (B) in session P1. **Keep (B) if its exact down6 is within 3 % of the probe's
     PW6 numbers (0.2941 / 0.2904 ms) at nb 384 / 512; else build (A).**
   - **(A)'s conditions (lead, 10-27).** All three must hold, checked in P3:
     - **bits:** `gemm2s_q6k8` / `gemm2_q6k8` read the 304-B copy and give the same bits as with the 288-B copy (md5
       on / off / main, plus the product bits tests);
     - **VRAM accounted:** `make_q6k8`'s size check uses 304 B a block (`q6k8_fits`), and the load banner's "Q6_K
       int8 copy" line reports the new size;
     - **behind the same flag:** the 304-B layout is made only with `AXON_W4A8=1` on an sm_90a module. Otherwise
       the 288-B Q6K8 copy is unchanged.
4. **Threshold T.**
   - Session P1 sweeps nb ∈ {64, 96, 128, 192, 256, 320, 384, 448, 512, 768, 1024} on all four products.
   - T is the smallest nb at and above which the exact W4A8 kernel is ≥ 1.05× today's kernel at *every* swept nb,
     per product (per-product T allowed).
   - ESTIMATED T ≈ 128–192 for the Q4_K products: the 10-26 skeleton was 1.68× gemmh2 at gate nb 128, × f 0.836
     ≈ 1.40. Q6 is likely higher, since it is 1.11× at 384.
5. **Q6_K performance headroom (not v1's gate, a follow-up).**
   - Down6 is 1.11–1.21×, against 1.55–1.70× on the Q4_K products.
   - The three-plane form costs 3 tensor passes. Two balanced radix-8 planes cover only [−27, 36], and Q6_K's
     int8 scales span [−128, 127] (llama.cpp's Q6_K quantizer maps each block's largest sub-block scale to −128).
   - Leads: a 128-token tile for the Q6 kernel, or a per-tensor scale-range check that picks 2 planes where they
     fit. Measured later; not in this build.

## 4. Kill rules (pre-registered here, before any build)

**Checkpoint 1: kernels (end of builder-day 3).** Production kernels (gu with silu·up, plain with mode 0/1 + bias,
Q6 under the §3.3 choice) on the H100, in a probe-style bench against today's kernels, interleaved, 7 rounds, at
nb 384 and 512.
- **Bits:** every output identical to `gemmh2_q4_k_gu` / `gemmh2_q4_k` (mode 0 and 1, with and without bias) /
  `gemm2(s)_q6k8`.
- **Speed:** S_weighted with #165's shares (k/v at 1) ≥ **1.40** at both nb, measured directly on the production
  kernels (no factor). The probe measured 1.503 / 1.543; 1.40 lets the production epilogues cost up to ~7 % of that.
- **Either fails → stop.** Record it in `experiments-rejected.md`. No integration and no e2e spend.

**Checkpoint 2: end to end, the build's GO / KILL.** The rule is fixed now, with its ESTIMATE:

- **Setup:**
  - H100 SXM, Qwen2.5-7B-Instruct-Q4_K_M, the #139 / #165 serve mix kit, C = 64 and C = 128, 2 rounds.
  - Arms interleaved and order-reversed per round, fixed salt per (round, C).
  - Main (merge base) vs the candidate with `AXON_W4A8=1`, both at main's scheduler defaults of that day.
- **Primary:** aggregate tok/s at C = 128, the median of the 2 rounds.
- **ESTIMATE: +13.5 %. Range: +8.7 to +18.2 %, with the low end re-derived from a measured bound (below).**
  Derivation:
  - **MEASURED inputs:**
    - GEMM is 78.4 % of feed time (#157);
    - feeds are 57.5 % of wall (fifo60 C = 128, 44.77 s of 77.83 s over 2 rounds) and 56.8 % at sjf80;
    - the exact kernels' S_weighted is 1.50–1.54 (#165, INFO lines).
  - **ESTIMATED inputs:**
    - production keeps 95 % of the kernel speedup, giving S ≈ 1.45 (low end 1.35, high end 1.52);
    - coverage c, the share of feed GEMM time in passes ≥ T: 0.85 for the point estimate. That lies inside the
      measured bound below.
  - **Arithmetic:** aggregate = 1/(1 − feed_share × 0.784 × c × (1 − 1/S)) − 1.
    - point: 1/(1 − 0.576 × 0.784 × 0.85 × 0.310) − 1 = **+13.5 %**;
    - high: c 1.0, S 1.52 → +18.2 %;
    - low: S 1.35 at the bound's minimum c (table).
  - **Pre-flight (done 10-27, local, $0): the measured bound on c.**
    - **The data that exists.** The H100 serve traces' per-pass sizes were not kept: they lived on pods 2 and 3,
      and only logs reached `gate-logs`. So c can't be read off a histogram. What the H100 job logs do give, per
      cell (job 1027-030631-09ae):
      - feed passes N, prompt tokens and the p50 pass size: fifo60 C = 128 743 / 744 passes, 242,688 / 244,340
        tokens, p50 392 / 390; sjf80 C = 128 530 / 532 passes, 241,669 / 241,869 tokens, p50 707 / 695;
      - the per-n GEMM ms grid (ANATSEG, n 16–1024);
      - for fifo60, the grid's GEMM total over the cell's real passes: 0.784 × 41.32 s / 2 = 16.20 s.
    - **The bound.** A linear program over the pass-size histogram (sizes 1–1152) finds the minimum and maximum c
      that every histogram consistent with those numbers allows. It allows 10 passes of slack on the p50 split:
      anat.py takes the p50 over steps, and 0.1–0.4 % of steps run two feed passes. The worst case piles half the
      passes just under T.
    - Script: `coverage_lp.py` in the builder's scratch dir. Bella's traces were not used: the step budget sizes
      passes from the host's measured cost, so bella's passes are smaller.

    | T | fifo60 C = 128: c ≥ | low end (S 1.35) | sjf80 C = 128: c ≥ | low end (S 1.35) |
    |---|---|---|---|---|
    | 128 | 0.759 | **+9.74 %** | 0.817 | **+10.42 %** |
    | 160 | 0.720 | **+9.20 %** | 0.787 | **+9.99 %** |
    | 192 | 0.681 | **+8.66 %** | 0.756 | **+9.57 %** |
    | 256 (for reference) | 0.612 | +7.71 % | 0.717 | +9.02 % |
    | 320 (for reference) | 0.580 | +7.28 % | 0.696 | +8.74 % |

    - The maximum c is 0.97–1.0 at every T. sjf80's bound omits the GEMM-total constraint (none was published for
      it), so it is the looser of the two.
    - **Reading:** at every T the lead asked about (128, 160, 192), even the worst-case distribution keeps the low end
      above +8 %, on both arms. Rule S passed (#165), so sjf80 is likely main's default by Checkpoint 2. **Pre-flight
      passes: proceed.**
    - **Added after the bound, stricter, not a change to Checkpoint 2.** If P1's sweep sets T > 192 for any product,
      re-run the bound at that T on main's default arm before P3. If the low end falls under +8 %, stop before any
      integration spend and report.
    - **Closing the gap.** P1 and P4 print a per-pass size histogram into the job log (the serve trace's `fed` per
      feed pass), so it lands in `gate-logs`. That turns this bound into a measurement before Checkpoint 2 is read.
  - **As first written** (kept for the record): coverage 0.85 assumed, range +9.6 to +17.2 %.
- **GO (merge; default on for sm_90a at nb ≥ T)** requires all of:
  - C = 128 aggregate ≥ **+8 %**;
  - C = 64 aggregate ≥ **+3 %**;
  - C = 128 ITL p99 ≤ main × 1.05, and short-prompt TTFT p50 ≤ main × 1.05;
  - **bits:** text_sha and completion_tokens equal for every request across the arms; plus the md5 gate (on / off /
    main) n/N identical on the H100.
- **KILL:** any clause fails.
  - The code is not merged. Record it in `experiments-rejected.md`.
  - "Hopper exact int8 GEMM" is closed. Item 4's FP8 rule then measures against today's gemmh2.
  - **A bits failure is a bug, not a KILL:** fix it and rerun once; a second bits failure is a KILL.
- **Why +8 %:** it sits under the estimate's low end (+9.6 % as first written; +8.66 % at the bound's worst case,
  fifo60 T = 192) because every Hopper estimate before #165 overshot.
  #165's own estimates undershot the measurement, so the bar is not tightened either. The bar is not lowered after
  the data.
- **Reported, not gating** (methodology: both pp and tg):
  - `gen` pp2100 and pp560 on the H100 with the flag on vs off (OPINION, not derived: +15–25 % pp2100; attention's share
    of a deep pass on the H100 is unmeasured for this cell);
  - tg128, which must be unchanged within noise (decode never takes the path).

## 5. Testing without an sm_90a machine at home

- **On bella (sm_120) and avante (sm_89), what runs:**
  - **The NVRTC sm_90a compile + ptxas check** (no GPU needed). The kernels are in the cubin, there are no wgmma
    serialization notes, and spills are 0. It compiles with a unique symbol, because NVRTC returns an *empty* log for
    a source it has compiled before, which would make the check vacuous (found on this branch).
  - **CPU unit tests:**
    - the Q6 digit encoding (all 256 scales);
    - the integer references (`probe_w6a8_ref`-style) and the copy layout's invariants (TMA alignment, smem
      carve-up, ≤ 227 KiB);
    - a pure routing function (`(qt, rows, nb, sms, hopper, flag) → kernel`) tested off-GPU.
  - **The whole existing suite and the md5 gate vs main.** Off sm_90a the new file compiles to nothing and routing
    requires `module.hopper()`, so bella / avante bits are unchanged by construction; the md5 gate proves it there
    (n/N).
- **What can't run there:** every launch of the new kernels (wgmma, TMA tensor maps, setmaxnreg, mbarrier parity).
  Those run only on H100 pods, staged as #165's job was:
  - prebuilt binaries, prewarmed sm_90a cache;
  - child processes with time limits;
  - memcheck on tiny launches first;
  - bits per kernel;
  - then benches with only the kernels that passed.
  - Each round is rehearsed on bella first (the sm_90a cells skip there).
- **Portable fallback: yes, and it is today's code.**
  - nb < T, sm_90a without the flag: gemmh2 / gemm2s / gemm2 on sm_90a; gemm2 / gemm2t / gemvn on sm_80+ and below.
  - The W4A8 path is bit-identical to them, so the fallback is the *same identity*, not a degraded mode. A host
    without sm_90a loses only speed.
- **No portable twin of the new kernel.** An mma.sync emulation on sm_89 / sm_120 would test a different kernel.
  The risks are Hopper-only mechanisms (descriptors, TMA boxes, barrier phases, register reallocation), and the
  integer algebra is already pinned by the CPU references and the H100 bits tests. OPINION: 1–2 days for little.
- **The sm_120 story (bella, RTX 50xx): none in this build.**
  - wgmma is sm_90a-only (CODE: hopper.cu's guard; PTX ISA). sm_120 has TMA and mma.sync (including its
    block-scaled FP8 / FP4 forms) but no wgmma and no tcgen05 / TMEM (those are sm_100's).
  - The probe's win comes from async register-sourced wgmma hiding the unpack. On consumer cards the prefill GEMM is
    MIO / L1-path bound, not tensor bound (MEASURED, r17 ablation on sm_89: removing every mma changed −5 to 0 %).
    So the remedy doesn't transfer.
  - A consumer-card prefill gain needs its own measured-gap proposal.
- **Other Hopper parts:** H200 / GH200 / H20 (sm_90a) are covered by construction. Re-measure before claiming.
- **B200 (sm_100):** needs a tcgen05 rewrite; out of scope.

## 6. Pod plan and cap

H100 SXM secure, $3.99/h. Each session follows the builder rules:
- prebuilt kit with SHA256SUMS and a prewarmed `kcache-sm90a` (3 NVRTC versions);
- a bella rehearsal first;
- every job queued before `axon-pods up`;
- a dead-man of ≤ 0.7 h;
- the pod downed by the script.

| session | content | ESTIMATED pod time | ESTIMATED $ |
|---|---|---|---|
| P1 (day 2–3) | bring-up: memcheck tiny, bits on all entry points (modes 0/1, bias, gu), the nb sweep for T, the §3.3 (B)-vs-(A) timing, **Checkpoint 1**; one fifo60 + one sjf80 C = 128 serve cell on main with the per-pass histogram printed to the log (+~2 min) | ~12 min + ~4 boot | 1.10 |
| P2 (contingency) | one fix round of P1 | ~12 + 4 | 1.10 |
| P3 (day 4–5) | integration: H100 GPU test tier `full` with `AXON_W4A8=1` (`AXON_TEST_GPU_SERIAL=1`), md5 on / off / main | ~25 + 4 | 1.95 |
| P4 | **Checkpoint 2** e2e A/B (2 rounds × C 64 / 128 × 2 arms) + pp / tg board; per-pass histograms in the log | ~15 + 4 | 1.25 |
| P5 | review-round re-gate (P3's suite on the merge-ready head) | ~25 + 4 | 1.95 |
| **total** | | | **≈ 7.35** |

- **Cap: $10 for the build.** Spend past $10 needs the lead's go in writing.
- **Spend stops at the first failed checkpoint:** a Checkpoint 1 KILL caps it at ≈ $2.20 (P1 + P2).
- **Builder time: ESTIMATED 3–5 builder-days.**
  - day 1: the gu and mode-1 / bias epilogues and the routing function;
  - day 2: tests and local gates, then rehearsal and P1;
  - day 3: Checkpoint 1;
  - days 4–5: integration, P3 / P4, the PR.
- That is less than item 2's 6–10 days because the exact kernel already exists, already measured and bit-identical.

## 7. Decisions (lead, 10-27) and what's open

1. **Restart item 2:** yes, conditional on the pre-flight. The pre-flight passed (§4): the low end is +8.66 % to
   +10.42 % at T ≤ 192.
2. **No new numerics tier:** accepted as the contract (§2).
3. **Q6 copy:** (B) first, with the pre-registered switch to (A). (A) is acceptable under §3.3's three conditions.
4. **Open:** Shady's review of this note before any build. T itself, from P1's sweep; T > 192 triggers the §4
   re-check.
