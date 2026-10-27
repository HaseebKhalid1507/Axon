# Design note (Checkpoint 0): a production W4A8 prefill path on Hopper

**Status: proposed; pre-flight passed (10-27); round-2 review MUSTs (#165, Shady 6442748632 / 6442789303) addressed. Nothing is built, nothing is booked.** Written 2026-10-27 on `feat/r11-w4a8-gate` (dd1e7fa, base
main 0c23a87), after rule P (#165) came back GO. Input: [../log/2026-10-27-w4a8-rule-p.md](../log/2026-10-27-w4a8-rule-p.md).
Labels: **MEASURED** (job cited), **ESTIMATED** (arithmetic shown), **CODE** (file on the branch or main), **OPINION**.

**Lead's decisions (Jawz, 10-27), recorded here:**
- **Restart item 2: yes, conditionally** on a free local check of how much feed GEMM time falls in passes ≥ T.
  Done: §4, "Pre-flight". Every T ∈ {128, 160, 192} keeps the ESTIMATE's low end above the +8 % bar, *if S above
  nb 512 holds at 1.35* (an ESTIMATED bound; §4 lists its assumptions). The post-P1 re-run (§4) is the real stop.
- **Rulings on the round-2 review (Jawz, 10-27):**
  - the stop rule reads the LOWER of fifo60 and sjf80 (M9);
  - every serve cell pins `--admit-order` and `--step-ms` explicitly (M4);
  - the post-P1 re-run is unconditional, on P1's measured S at nb {384, 512, 768, 1024} (M8);
  - the LP script and its pinned input are committed (M7);
  - the slack's source is stated and P1 measures the H100's two-pass rate (M10);
  - the bound is labelled ESTIMATED;
  - M1–M3, M5 and M6 are fixed.
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
  - the md5 gate and the bits tests are the correctness proof, with a W4A8 hit counter showing the path actually
    ran (§2).
- **Rollout:** behind `AXON_W4A8=1` (off by default) until a pre-registered end-to-end serve A/B on an H100 says GO;
  after that, on by default for sm_90a.
- **ESTIMATED effect:** +13.5 % aggregate at mix C = 128 (fifo60; +13.3 % at sjf80). Range +8.7 % (a bound under
  assumptions, §4) to +18.2 %. At a GEMM share of 72.6 % instead of 78.4 % (§4), the point is +12.4 %. Kill bar +8 %.
- **Cost:** 3–5 builder-days and ≤ $10 of H100 for the whole line (cap; §6, from the regular window allowance, no
  piggy).

## 1. Scope: which products, which types, which hardware

**In (v1):**

| product (7B shapes) | weight type | today's kernel on a feed pass on sm_90a (CODE, main batch.rs) | exact probe vs it, nb 384 / 512 (MEASURED, job 1027-084257-71f3) | serve epilogue |
|---|---|---|---|---|
| fused gate + up, 3584 → 2 × 18,944 | Q4_K | `gemmh2_q4_k_gu` whenever `!takes_gemvm` (nb > 32), from the gu fuse at the gate call site | 1.689× / 1.701× | silu(gate) · up |
| down, 18,944 → 3584 (Q4_K layers) | Q4_K | `gemmh2_q4_k` where `gemmh_plain` (≥ one wave of 128 × 64 tiles: nb ≥ 257 on 132 SMs), else gemm2t / gemm2s | 1.554× / 1.561× | mode 1: out += W x (+ bias) |
| down (Q6_K layers of Q4_K_M) | Q6_K (by its prefill copy) | `gemm2s_q6k8` / `gemm2_q6k8` (64 × 128 where 2 ctas ≥ 3 sms) | 1.110× / 1.206× | mode 1 |
| q and wo, 3584 → 3584 | Q4_K | as down (Q4_K) | 1.581× / 1.659× | q: mode 0 (+ bias); wo: mode 1 |

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
- **Decode steps, verify passes, and feed passes < T:** they keep today's kernels.
  - **Routing rule:** the W4A8 path is taken only on a feed (prefill or chunk) pass with nb ≥ T. The pass kind is
    passed down from the feed caller, so a pool decode step never takes it, whatever its row count (up to C = 128).
  - **Why it's needed (CODE):** on main a decode step at C = 128 with nb > 32 already reaches `gemmh2_q4_k_gu` through
    the same gu fuse. An nb-only rule with T ≤ 128 would route decode steps too. That would change ITL (a gating
    clause) and is outside the estimate, which covers feed time only.
  - **Tests:**
    - a routing unit test (pass kind × nb × product → kernel);
    - on the H100, a C = 128 pool run of decode-only steps whose W4A8 hit counter (§2) stays 0.
  - Same bits either way (§2), so the routing rule is a performance and ITL choice only. W4A8 for decode steps would
    be a later change, with its own rule.
- **MoE expert GEMMs:** different grids (#154 track).

## 2. Exactness: the contract stays the default one

**The contract (accepted by the lead, 10-27):**
- The W4A8 path is a **faster route to the same bits**, under the default (exact) contract.
- Every output equals today's kernel's output for the same product: `gemmh2_q4_k_gu`, `gemmh2_q4_k` (modes 0 / 1,
  with or without bias) and `gemm2(s)_q6k8`. Hence it also equals `gemv_ref`'s order (identity C).
- There is no new `--numerics` tier, no `NUMERICS_VERSION` bump and no identity field. Routing (`AXON_W4A8`, the
  threshold T, the GPU) can't change bits.
- **Held in place, every round:**
  - the bits tests on the H100 on every output: ragged shapes, the 7B shapes, and every nb cell of P1's sweep.
    - The `w4a8_*` entry points join `gpu_products_match_reference_bits` and
      `gpu_batched_products_match_gemv_bits`, compared with `gemv_ref`, not only with gemmh2.
  - the md5 gate on the H100, `AXON_W4A8=1` vs `=0` vs main, n/N identical;
  - at Checkpoint 2, text_sha and completion_tokens equal for every request.
- **Every gate must show the path ran (M1).** A green gate that never launched a W4A8 kernel proves nothing.
  - **Hit counter:** a process-global launch counter per entry point (`w4a8_q4_k`, `w4a8_q4_k_gu`, `w4a8_q6`).
    `AXON_W4A8_STATS=1` prints it at exit (`W4A8 HITS q4_k <n> gu <n> q6 <n>`).
  - **Which entry points must hit, per test (Shady S8).** The pinned 7B case is Qwen2.5-7B-Instruct-Q4_K_M, the
    anatomy's and the serve's model. Its products are gate_up Q4_K (`w4a8_q4_k_gu`); q, wo and the Q4_K down layers
    (`w4a8_q4_k`); and the Q6_K down layers (`w4a8_q6`, given the Q6 copy, made at load on an 80 GB card).

    | test / gate | model | must hit (> 0) | must be 0 |
    |---|---|---|---|
    | md5 gate, on-arm (`gen`, the md5 prompt set incl. the ~600-token prompt, `AXON_W4A8_STATS=1`) | Qwen2.5-7B-Instruct-Q4_K_M | gu, q4_k, q6 | — |
    | md5 gate, on-arm, the set's small models (0.5B / 1.5B / Llama-1B) | each | none required (bits only) | — |
    | new `gpu_w4a8_feed_batch_matches_feed_7b` (H100, `AXON_W4A8_T=33`): `feed_batch` of a 600-token prompt vs `feed` per token, logits + KV bit-equal | Qwen2.5-7B-Instruct-Q4_K_M | gu, q4_k, q6 (counted around the `feed_batch` call) | 0 during the per-token `feed` calls |
    | new `gpu_w4a8_pool_routing_7b` (H100, `AXON_W4A8_T=33`): pool C = 128, one 600-token prompt fed in chunks while 127 slots decode | Qwen2.5-7B-Instruct-Q4_K_M | gu, q4_k, q6 on `Place::Seq` (feed) passes | **0 on every `Place::Rows` (decode) pass** (per-pass counter deltas) |
    | `gpu_prefill_batchings_bit_identical_*`, `pool_steps_match_solo_in_any_order`, `serve_batched_matches_solo` (H100 `full` tier, `AXON_W4A8_T=33`) | their existing small models | bits only, no hit requirement (their shapes are mostly below the routing rows; decode-dominated) | — |
    | Checkpoint 2 candidate arm, every cell | Qwen2.5-7B-Instruct-Q4_K_M | gu, q4_k, q6 (a cell with 0 on any of them is void) | — |

  - **The H100 `full` tier** runs with `AXON_W4A8=1 AXON_W4A8_T=33`, a forced threshold just above the gemvm limit,
    so every eligible feed pass takes the path.
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
- **MEASURED (job 1027-084257-71f3):** 0 mismatches against gemmh2_q4_k and gemm2(s)_q6k8 on every output:
  - five ragged shapes per kernel (1–17 blocks, rows and tokens off the tile, 1–7 CTAs, the stage ring wrapping
    across tiles) against `gemmh2_q4_k` / `gemm2s_q6k8`;
  - every timed 7B cell at nb 384 and 512, each against its own base kernel: `gemm2s_q6k8` at nb 384, `gemm2_q6k8` at
    512 (batch.rs's pick on 132 SMs);
  - the gu tail (silu·up), mode 1 and bias were never run by the probe; Checkpoint 1 covers them.
- **So the production path is identity-C material, in the exact (default) contract:**
  - **no `--numerics` tier.** The ADR's fast mode is still "proposed", and nothing here needs it;
  - **no `NUMERICS_VERSION` bump** and no identity field;
  - routing by T (or by `AXON_W4A8`) can't change bits;
  - **the proof is the bits tests plus the md5 gate, with the hit counter showing the path ran** (above). `gen` and
    `serve` text with `AXON_W4A8=1` vs `=0` vs main, same H100, must be n/N identical. Any difference is a bug, not a
    tolerance question.
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
     - The first 276 bytes are `q6k_to_i8`'s, unchanged. Q6K8's 12 bytes of padding become 24 bytes of digits plus
       4 of padding.
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
   - **A product that never qualifies (M11).** If a product has no swept nb from which it stays ≥ 1.05× up to
     1024, P1 prints `W4A8 T <product> none`.
     - That product is **not routed**: it keeps today's kernel, and counts at S = 1 in Checkpoint 1's S_weighted and
       in the post-P1 re-run.
     - This is a valid table, not a missing input and not a rerun reason. Checkpoint 1's bar applies to what ships.
     - Down6 (1.11× / 1.21× in the probe) is the likely case.
   - Every swept cell is bit-checked against its base kernel on every output, not only timed. The ragged tests stop
     at nb 200 and 17 blocks; the sweep covers 64–1024 on the 7B shapes.
   - Per-product T goes into the post-P1 re-run per product (§4). The pre-P1 LP bound uses one T for all GEMM time,
     so its table is read at the *largest* T it might face.
   - ESTIMATED T ≈ 128–192 for the Q4_K products: the 10-26 skeleton was 1.68× gemmh2 at gate nb 128, × f 0.836
     ≈ 1.40. Q6 is likely higher, since it is 1.11× at 384.
5. **Q6_K performance headroom (not v1's gate, a follow-up).**
   - Down6 is 1.11–1.21×, against 1.55–1.70× on the Q4_K products.
   - The three-plane form costs 3 tensor passes. Two balanced radix-8 planes cover only [−27, 36], and Q6_K's
     int8 scales span [−128, 127] (llama.cpp's Q6_K quantizer maps each block's largest sub-block scale to −128).
   - Leads: a 128-token tile for the Q6 kernel, or a per-tensor scale-range check that picks 2 planes where they
     fit. Measured later; not in this build.

## 4. Kill rules (pre-registered here, before any build)

**Checkpoint 1: kernels (end of builder-day 3, session P1).** Production kernels (gu with silu·up, plain with
mode 0/1 + bias, Q6 under the §3.3 choice) on the H100, in a probe-style bench against today's kernels, interleaved,
7 rounds, at nb 384, 512, 768 and 1024. The gate reads 384 and 512; 768 and 1024 feed the post-P1 re-run.
- **Bits:** every output identical to `gemmh2_q4_k_gu` / `gemmh2_q4_k` (mode 0 and 1, with and without bias) /
  `gemm2(s)_q6k8`.
- **Speed:** S_weighted with #165's shares (k/v at 1) ≥ **1.40** at both nb, measured directly on the production
  kernels (no factor). The probe measured 1.503 / 1.543; 1.40 lets the production epilogues cost up to ~7 % of that.
- **Either fails → stop.** Record it in `experiments-rejected.md`. No integration and no e2e spend.
- **One attempt (M3, M12).** The attempt is P1's first run.
  - If it completes the Checkpoint 1 cells with every bit check green, its speed numbers are final: a speed miss is
    a KILL, with no fix round.
  - **A bits failure follows Checkpoint 2's rule (M12; lead, 10-27):**
    - run 1's speed table is **published**;
    - after the fix, P2 reruns the cells;
    - **the speed bar must hold in both runs.** Run 1's table counts as measured, including any product whose bits
      failed. A run 1 that missed on speed is a KILL whatever P2 shows, so a fix can't buy a fresh speed draw.
  - **The only rerun (session P2)** is for a P1 that produced no valid Checkpoint 1 table, for one of these written
    reasons:
    - (i) bring-up: a launch fault, a sanitizer error, a bits mismatch (under M12's rule above), or a missing
      `W4A8 S` / `PASSHIST` line (`W4A8 T <p> none` is not missing, M11). If run 1 produced no speed table at all
      (a fault before the cells), there is nothing from run 1 to hold;
    - (ii) infrastructure: the pod was lost, the job was killed by the host, FOREIGN_GPU > 0 % (TAINTED), or the
      kit failed before the first cell.
  - P2 runs once. A second failure for any reason is a KILL.
  - A P1 that has a valid table but misses on speed gets no P2.

**Checkpoint 1, P1 void and one rerun (lead's ruling, 10-27, after P1's data; pre-registered before the rerun is
submitted).**
- **P1 (job 1027-151227-e203) is VOID, not a KILL.** Its W4A8 timings didn't measure what the rule names ("measured
  directly on the production kernels"):
  - the W4A8 arm's test launcher (`launch_w4a8_k`, tests.rs @ 441ebee) called `g.info()` inside every timed launch;
    the baseline arm's launchers didn't;
  - in the bench's memory context that call costs **0.89 ms median** (p90 0.99, max 4.0) on bella;
  - routed through the same per-launch call, a baseline kernel goes from 0.036 to 0.405 ms at nb 64, flat and jittery
    like P1's W4A8 arm (job 1027-154351-fd3a, `cuda_w4a8_harness_infocost`).
  - Bits were clean and stand.
- **One rerun is authorized** (P1R), on these terms:
  - **Kernel code byte-identical to P1:** `src/cuda/*.cu` and the KERNELS list unchanged since 441ebee. The job prints
    the KERNELS source sha256 and the loaded cubin's sha256; the kit's cubins must have P1's SHA256SUMS entries. Only
    the test harness changes.
  - **Same cells, same 7 rounds, same 1.40 bar at nb 384 / 512, same bits checks.** The session stage only. P1's serve
    cells ran main's binary through other launchers, so its `PASSHIST` / `PASSMULTI` / `FEEDSHARE` lines are reused
    for the post-P1 re-run.
  - **Validity check first, printed before the verdict:**
    - every W4A8 cell's round-to-round spread (max / min − 1 over the 7 rounds) must be < 5 %;
    - W4A8 time must scale with work: t(gate_up, nb 1024) / t(q/wo, nb 64) ≥ 10.
    - If either fails, P1R is VOID again and the line stops, with no third run.
  - **Whatever S comes out is final.** PASS proceeds to the post-P1 re-run. A miss is a KILL, with no further rerun
    for any reason (M12's bits rule still applies to a bits failure; nothing else).
  - **Cap $2:** `--hours 0.5`. Booked only after a bella rehearsal of the fixed harness passes.

**Checkpoint 2: end to end, the build's GO / KILL.** The rule is fixed now, with its ESTIMATE:

- **Setup (pinned now, M4):**
  - H100 SXM, Qwen2.5-7B-Instruct-Q4_K_M, the #165 serve kit: `sd_client.py` sha256 a82b55c178a2fa9b…,
    `prompts-mix-v2.jsonl` c3ff5fb849cd7dc9… (feat/feed-anatomy). C = 64 and C = 128, **3 rounds** per arm
    (the TTFT clause needs ≥ 3; every other clause reads rounds 1–2, exactly as registered).
  - **Two scheduler configs, both run, each with explicit flags in every cell, whatever main's default is that day:**
    `--admit-order fifo --step-ms 60` and `--admit-order sjf --step-ms 80`.
  - In each config: main (merge base) vs the candidate (`AXON_W4A8=1`, `AXON_W4A8_STATS=1`). Arms are interleaved and
    order-reversed per round, with a fixed salt per (round, C).
  - If main's default changes before P4, the ESTIMATE is re-stated for it. The bar is not.
- **Definitions (the kit's, CODE `sd_client.py`):**
  - **aggregate** = `agg_tok_s` = Σ completion tokens / wall, wall = first send → last stream end, per cell;
  - **short-prompt TTFT p50** = `ttft_short_ms_p50`, over kinds `short` and `classic` (the set's shorts are 36–48
    prompt tokens);
  - **ITL p99** = `itl_ms` p99, all streams pooled.
- **Primary:** aggregate tok/s at C = 128, the median of rounds 1 and 2. **Each clause must hold in both configs**
  (the lower config decides, as in M9).
- **ESTIMATE: +13.5 % (fifo60) / +13.3 % (sjf80). Range: +8.7 % (an ESTIMATED bound, below) to +18.2 %.**
  Derivation:
  - **Inputs from the H100 feed-anatomy job 1027-030631-09ae** (gate-logs `2026-10/feedanat-86953be-h100`):
    - **feeds / wall, C = 128, 2 rounds (MEASURED, trace):** fifo60 44.77 of 77.83 s = 0.5752; sjf80 40.54 of
      71.37 s = 0.5680.
    - **GEMM share of feed time, 78.4 % (partly ESTIMATED).** The per-kernel split is measured. Its weighting is
      not: the depths come from the prompt set, and the grid under-predicts the traced feed time (41.32 s vs 44.78 s,
      ratio 1.08). If that 8 % gap is all non-GEMM, the share is 72.6 %, and the point estimate drops to +12.4 %.
    - **The exact kernels' S_weighted, 1.50–1.54** (MEASURED, job 1027-084257-71f3, INFO lines).
  - **ESTIMATED inputs:**
    - production keeps 95 % of the kernel speedup, giving S ≈ 1.45 (low end 1.35, high end 1.52);
    - the low end's S of 1.35 is already below Checkpoint 1's 1.40 bar, so it is conservative for any build that
      passes Checkpoint 1 (S5);
    - coverage c, the share of feed GEMM time in passes ≥ T: 0.85 for the point estimate, inside the bound below.
  - **Arithmetic:** aggregate = 1/(1 − feed_share × gemm_share × c × (1 − 1/S)) − 1.
    - point: 1/(1 − 0.576 × 0.784 × 0.85 × 0.310) − 1 = **+13.5 %**;
    - high: c 1.0, S 1.52 → +18.2 %;
    - low: the bound below.
  - **Pre-flight (done 10-27, local, $0): an ESTIMATED bound on the GEMM-time fraction saved.**
    - **The data that exists.** The H100 serve traces' per-pass sizes were not kept: they lived on pods 2 and 3,
      and only logs reached `gate-logs`. From job 1027-030631-09ae, per C = 128 cell:
      - feed passes N, prompt tokens and the p50 pass size: fifo60 743 / 744 passes, 242,688 / 244,340 tokens,
        p50 392 / 390; sjf80 530 / 532 passes, 241,669 / 241,869 tokens, p50 707 / 695;
      - the per-n GEMM ms grid, n 16–1024;
      - for fifo60, the grid's GEMM total over the cell's real passes, 0.784 × 41.32 s / 2 = 16.20 s.
    - **The script:** [`scripts/w4a8_coverage_lp.py`](../../scripts/w4a8_coverage_lp.py), on this branch (M7).
      - `bound` mode: a **minimum-ratio LP** (Charnes–Cooper) over every pass-size histogram consistent with those
        numbers. It minimises the saved fraction r = Σ_{s ≥ T} x_s g(s)(1 − 1/S(s)) / Σ x_s g(s) itself, not the
        numerator of that ratio.
      - **Pinned input:** the job log at gate-logs commit 4cbe7f3, path
        `2026-10/feedanat-86953be-h100/1027-030631-09ae.log`, sha256 `55efde1e…e745ae303`. The script refuses other
        bytes.
      - `selftest` reproduces the registered +9.74 / +9.20 / +8.66 %.
      - Bella's traces are not used for sizes: the step budget sizes passes from the host's measured cost.
    - **Assumptions of the bound (S6). It is ESTIMATED, not measured:**
      1. g(n) is linear between the 8 grid points; below 16 it is g(16); above 1024 it is ∝ n (as anat.py). Passes
         are 1–1152 tokens. Shady checked a 1536 / 2048 cap and nothing changed.
      2. GEMM ms is depth-independent (to ~1 % on the grid; MEASURED).
      3. **Slack on the p50 split, 10 passes per half (M10).** anat.py takes the p50 over steps (fed / feeds). The
         two-feed-pass rate is taken from **bella's** traces of the same serve code (2 / 1925 to 8 / 1915 steps,
         0.1–0.4 %), not the H100's. P1 measures the H100 rate (§6), and the post-P1 re-run uses its measured passes,
         so after P1 this is no longer an assumption.
      4. **S = 1.35 for every covered pass, including nb > 512.** #165 measured nb 384 / 512 only.
         - Supporting, ESTIMATED: the 10-26 probe skeleton at nb 1024 was 1.93–1.97× gemmh2 on gate / down4 / q-wo.
           × f 0.836 (10-27) that gives ≈ 1.6.
         - Down6 at nb > 512 was never run.
         - **This assumption decides the low end** (table below). P1 measures it.
      5. The pass mix is main 0c23a87's, one T for all GEMM time, and decode excluded (§1's routing rule).
      6. **Optimistic at small nb (Shady S12).** One S_w per pass treats the uncovered k / v / qkv at the nb 512
         share (~5 %). In the pinned ANATSEG that share is 15.1 % of GEMM time at n 128, 5.8 % at 384 and 2.7 % at
         1024, so the bound flatters passes just above T. `p1` mode is per product (uncovered classes at S = 1), so
         the real stop is unaffected. The `gate`, `down` and `qkv` classes read ~0.087 ms at every n ≥ 64: marker
         events, not GEMMs (under 1 %); `gate` is counted in gate_up's numerator.
    - **Bound at the registered assumptions** (slack 10, S 1.35 everywhere). The lower arm is read (M9):

    | T | fifo60 r ≥ | fifo60 low end | sjf80 r ≥ | sjf80 low end | lower arm |
    |---|---|---|---|---|---|
    | 128 | 0.1967 | +9.74 % | 0.2118 | +10.42 % | **+9.74 %** |
    | 160 | 0.1868 | +9.20 % | 0.2040 | +9.99 % | **+9.20 %** |
    | 192 | 0.1767 | +8.66 % | 0.1961 | +9.57 % | **+8.66 %** |
    | 256 (reference) | 0.1587 | +7.71 % | 0.1852 | +8.99 % | +7.71 % (misses) |

    - **Sensitivity, lower arm, low end** (script output, `bound` mode):

    | T | slack 10, S>512 1.35 | slack 40 | slack 80 | slack 10, S>512 1.25 | slack 10, S>512 1.15 |
    |---|---|---|---|---|---|
    | 128 | +9.74 % | +9.47 % | +9.12 % | +7.46 % | +4.74 % |
    | 160 | +9.20 % | +8.89 % | +8.49 % | +7.24 % | +4.60 % |
    | 192 | +8.66 % | +8.31 % | **+7.86 %** | **+7.06 %** | **+4.49 %** |

    - **Reading:**
      - The pre-flight clears +8 % at T ≤ 192 **only if** S above nb 512 is ≥ about 1.31 (interpolated between the 1.25 and 1.35 columns), and the slack is
        ≤ about 65 passes.
      - The first is the deciding assumption, which is why the post-P1 re-run is unconditional and reads P1's
        measured S at nb 768 / 1024.
      - **The pre-flight passes as an ESTIMATE. The real stop is the re-run.**
  - **Post-P1 re-run: unconditional, before P3 (M8, pre-registered now).**
    - **Inputs from P1's log,** in the format the script's `p1` mode parses:
      - `W4A8 S <product> nb <nb> <S>`: production W4A8 vs today's kernel, median of 7 interleaved rounds,
        bit-checked, at every Checkpoint-1 nb (384, 512, 768, 1024) and every swept nb;
      - `W4A8 T <product> <T>|none`: per product; `none` means not routed, S = 1 (M11);
      - `PASSHIST` / `PASSMULTI` / `FEEDSHARE` for one fifo60 and one sjf80 C = 128 serve cell on main, with
        explicit flags.
    - **The computation:**
      - **saved fraction:** r = Σ_passes Σ_p g_p(n)(1 − 1/S_p(n)) / Σ_passes g_all(n).
        - g_p per product comes from the anatomy's ANATSEG classes. k / v and the rest count at 1.
        - S_p is linear between P1's measured nb; it is 1 below T_p and held at S(1024) above.
      - **passes:** P1's measured histograms. Steps with two or more feed passes are split worst case.
      - **aggregate:** 1/(1 − feed_share × **0.726** × r) − 1. The share is the conservative 72.6 % from above; 78.4 %
        is reported too.
    - **Rule:** the lower of fifo60 and sjf80 must be ≥ +8 %, else **STOP before P3** and record it. A missing input
      is a STOP for the P1 attempt, which then falls under Checkpoint 1's rerun rule.
      - Output: `W4A8 P1 LOW END x %`, then `W4A8 P1 STOP RULE: PROCEED|STOP`.
      - The kill bar is not changed by this. It is an earlier, cheaper stop.
  - **As first written** (kept for the record): coverage 0.85 assumed, range +9.6 to +17.2 %. At 300e6c3 the
    script was in scratch and the bound was called "measured"; that was corrected here.
- **GO (merge; default on for sm_90a at nb ≥ T)** requires all of:
  - C = 128 aggregate ≥ **+8 %**;
  - C = 64 aggregate ≥ **+3 %**;
  - C = 128 ITL p99 ≤ main × 1.05 (rounds 1–2, as registered);
  - **C = 128 short-prompt TTFT p50 (lead's ruling, 10-27, before any data):**
    - main and the candidate run interleaved in the same session, alternating rounds, ≥ 3 rounds per arm;
    - the candidate's median over its rounds ≤ main's median × **min(max(1.05, 1 + main's round-to-round spread),
      1.15)**. The 1.15 ceiling is the lead's (10-27, before any data), after Shady's S11: with 3 rounds the spread
      tracks one outlier;
    - main's spread and the resulting bar are **published next to the verdict**, per config;
    - spread = max / min − 1 of main's own per-round short TTFT p50, in that session, config and C, from main's
      rounds only;
  - every clause in both scheduler configs;
  - the candidate arm's W4A8 hit counters > 0 in every cell (a cell with 0 hits is void);
  - **bits:** text_sha and completion_tokens equal for every request across the arms; plus the md5 gate (on / off /
    main) n/N identical on the H100.
- **KILL:** any clause fails.
  - The code is not merged. Record it in `experiments-rejected.md`.
  - "Hopper exact int8 GEMM" is closed. Item 4's FP8 rule then measures against today's gemmh2.
  - **One attempt; the rerun rules are written now (M3):**
    - **Infrastructure** (the pod lost, the job killed by the host, FOREIGN_GPU > 0 %, the kit failing before the
      first cell, or a voided cell): rerun once, same binaries, unchanged. The rerun's numbers stand alone.
    - **A bits failure** (any request's text_sha / completion_tokens differ, or md5 n/N < N): it's a bug, so fix it
      and rerun once (P3 + P4 again).
      - The first run's speed numbers are **published either way**.
      - **GO then needs the speed clauses to hold in both runs.** A first run that missed on speed is a KILL whatever
        the rerun shows, so a fix can't buy a fresh speed draw.
      - A second bits failure is a KILL.
    - Nothing else gets a rerun.
- **Why +8 %:** it sits under the estimate's low end (+9.6 % as first written; +8.66 % at the bound's worst case,
  fifo60 T = 192) because every Hopper estimate before #165 overshot.
  #165's own estimates undershot the measurement, so the bar is not tightened either. The bar is not lowered after
  the data.
- **Expected noise (S3; MEASURED, main's binaries, C = 128):**
  - **aggregate:** within-session spread over 2 rounds is ±0–3 % (jobs 1027-030631-09ae and 1027-084258-1ce4).
  - **Short TTFT:** the TTFT clause carries its own in-session noise allowance (main's spread, above). A
    cross-session comparison of the same binaries is not used for it.
  - P4 prints per-round values for every clause, so a borderline result is visible.
- **Reported, not gating** (methodology: both pp and tg):
  - `gen` pp2100 and pp560 on the H100 with the flag on vs off (OPINION, not derived: +15–25 % pp2100; attention's share
    of a deep pass on the H100 is unmeasured for this cell);
  - tg128, which must be unchanged within noise. Single-stream decode is nb 1 (gemvm / GEMV), and §1's routing rule
    keeps every decode step off the path.

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
    - **That proves only that the off-path is untouched. It proves nothing about W4A8.**
- **Ongoing cost:** every later PR that touches `w4a8.cu` or its routing needs an H100 gate session (P3-like, about
  $1.95), with the hit counters, before merge. Bella / avante can't gate it.
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
- `--hours 0.7` (dead-man 42 min);
- the pod downed by the script.

**How the money works (CODE, `axon-pods help`, M5):**
- **The window.** The allowance is $10 per 12 h window ([00:00, 12:00) and [12:00, 24:00) America/New_York), shared
  with every lane.
- **The reservation.** `up` reserves $/h × max(--hours, 1), so **every H100 session reserves $3.99** whatever its
  dead-man. `up` refuses unless window-left = $10 + min(0, piggy) − spent − live reservations is ≥ $3.99 and the
  balance is ≥ $5. At `down` the reservation is replaced by the actual spend.
- **The guard** terminates every live pod when a window's spend reaches $10.
- **No piggy.** Every session below comes from the regular window allowance. None is a piggy session, so none needs
  Haseeb told in advance, and `--piggy` is never passed.
- **Booking condition.** A session is booked only when `axon-pods budget` shows window-left ≥ $3.99 + $0.50 at
  booking time. If not, it waits for a later window; it is never squeezed in.
- **Clean 4090 timing lanes go first** (the policy's order; Shady S10). If an `axon-run --timing 4090` job is queued
  for the window, this line's session defers until that job has run, or until a later window.

| session | content | ESTIMATED pod time | ESTIMATED spend | window |
|---|---|---|---|---|
| P1 (day 2–3) | bring-up (memcheck tiny); bits on all entry points (modes 0/1, bias, gu) and on every swept nb; the nb sweep for T; (B)-vs-(A) timing; **Checkpoint 1** cells at nb 384 / 512 / 768 / 1024; one fifo60 + one sjf80 C = 128 serve cell on main (explicit flags) printing `PASSHIST` / `PASSMULTI` / `FEEDSHARE`, which also measures the H100's two-feed-pass rate | ~16 min + ~4 boot | 1.35 | W1 |
| P2 (only under Checkpoint 1's rerun rule) | the one rerun of P1 | ~16 + 4 | 1.35 | W2 (a later window) |
| (A) session (only if §3.3 picks (A)) | (A)'s bring-up, bits and down6 timing | ~12 + 4 | 1.10 | the window after P1 / P2 |
| — | **post-P1 re-run** (local, $0) → PROCEED / STOP | — | 0 | — |
| P3 (day 4–5) | integration: H100 GPU `full` tier (`AXON_W4A8=1 AXON_W4A8_T=33`, `AXON_TEST_GPU_SERIAL=1`, hit counters asserted), md5 on / off / main with stats | ~25 + 4 | 1.95 | W3 |
| P4 | **Checkpoint 2:** 2 configs × 2 arms × 3 rounds × C 64 / 128, plus the pp / tg board and per-pass histograms | ~20 + 4 | 1.60 | W4, or W3 if window-left ≥ $4.49 after P3 is down |
| P5 | review-round re-gate (P3's suite on the merge-ready head) | ~25 + 4 | 1.95 | W5 |

W1–W5 are successive windows that meet the booking condition, not fixed dates. Each gets at most one session of
this line, except W3 for P3 + P4 as stated.

**Spend against the $10 cap (the whole line):**

| path | sessions | ESTIMATED spend |
|---|---|---|
| clean | P1, P3, P4, P5 | **$6.85** |
| + P2 rerun | + P2 | $8.20 |
| + (A) fallback | + (A) session | $9.30 |
| + Checkpoint 2 infrastructure rerun | + P4 again | clean + $1.60 = $8.45; with P2 and (A): **$10.90** |
| + Checkpoint 2 bits rerun | + P3, P4 again | $12.85 |

- **Any path whose spend would pass $10 needs the lead's go in writing before the session that crosses it is
  booked.** That covers P2 + (A) + an infrastructure rerun ($10.90) and every bits-rerun path ($12.85).

- **Spend stops at the first failed checkpoint:**
  - a Checkpoint 1 KILL ends the line at $1.35, or $2.70 with P2;
  - a post-P1 STOP ends it at the same point, before P3.
- **Builder time: ESTIMATED 3–5 builder-days.**
  - day 1: the gu and mode-1 / bias epilogues and the routing function;
  - day 2: tests and local gates, then rehearsal and P1;
  - day 3: Checkpoint 1;
  - days 4–5: integration, P3 / P4, the PR.
- That is less than item 2's 6–10 days because the exact kernel already exists, already measured and bit-identical.

## 7. Decisions (lead, 10-27) and what's open

1. **Restart item 2:** yes, conditional on the pre-flight. The pre-flight passed as an ESTIMATED bound (§4): +8.66 %
   to +9.74 % on the lower arm at T ≤ 192, assuming S ≥ 1.35 above nb 512. The unconditional post-P1 re-run decides.
2. **No new numerics tier:** accepted as the contract (§2).
3. **Q6 copy:** (B) first, with the pre-registered switch to (A). (A) is acceptable under §3.3's three conditions.
4. **Round-2 rulings:**
   - the stop rule reads the lower of fifo60 and sjf80;
   - every cell has explicit scheduler flags;
   - the post-P1 re-run is unconditional, on S at nb 384–1024;
   - the script and its input are committed and pinned;
   - the slack's source is stated, and P1 measures the H100's two-pass rate.
5. **Open:** Shady's round 3; T itself, from P1's sweep.
