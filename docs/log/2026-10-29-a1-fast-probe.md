# WP-A1: the H100 probe of two `--numerics fast` GEMM candidates, staged (2026-10-29)

Track A's first step in the beat-vLLM plan (`specs/beat-vllm-scope.md` §6 WP-A1, Jawz workspace). It's a $4 probe of the two fast-mode GEMM
candidates on the T3 skeleton, with a kill rule. This entry covers the **build and the staging**. **No H100 has run it:
RunPod is on HOLD**, and the kit stops at "ready to book". Branch `feat/a1-fast-probe`; the code is test-only
(`src/cuda/probe_fast.cu` is never in KERNELS: tests.rs builds it as a module of its own). Nothing reaches a product path.
`--numerics fast` (approved 10-29) isn't needed here: the probe makes no product routing decision.

Labels: **MEASURED** (job / command cited), **CODE** (file), **ESTIMATED** (arithmetic shown).

## Pre-registered rules: issue #174 (filed 2026-10-29 17:34 UTC, before any booking)

Verbatim from the plan:
- FP8 gate nb 1024 **< 350** TMAC/s → Track A closed.
- W4A16 gate nb 1024 **< 230** → drop W4A16.
- down nb 128 with split-K **< 180** for both candidates → classic M1 out (record; continue mix-only if FP8 passed).

#174 also fixes, before any data:
- the derivations;
- the rule cells (each family's better tile; rule 3 takes the best split-K or stream-K variant);
- the tolerance bounds;
- validity: VOID unless the device is an H100 SXM with 132 SMs, the rule cells spread < 5 %, t(gate 1024) / t(q/wo 128) ≥ 10, and gemmh2 at
  gate nb 1024 is within ±15 % of T3's 116.5.

**Amendment 1 (#174, 2026-10-29, before any data; after shady's pre-booking review, 6/10 FIX-THEN-BOOK):**
- **V3 is wave-aware:** the bar is ≥ 0.5 × the tiles-per-SM ratio per variant (9 for 128×64, 4.5 for 128×128 / 256×64).
  The old bar of 10 VOIDed a healthy run (below).
- Only counting (tol PASS) cells feed the rule cells, V2 and V3.
- V1 is "H100 SXM" concretely: not NVL / PCIe, and power.limit ≥ 690 W (or the SXM5 name when that can't be read).
- FP8 gets a third tile on the wider instruction (`probe_fp8n128_128x128`, m64n128k32). A rule 1 KILL closes Track A
  only if that tile counted; otherwise it reads "KILL at n64 only" and Track A stays open.
- The fallback merge rule is written down.

The thresholds and derivations are unchanged.

`scripts/a1/a1_verdict.py` applies all of it to the job's log; `scripts/a1/a1_battery.py` proves it on 14 synthetic logs.

## What was built (CODE)

| kernel | instruction | CTA tile | stage / ring / smem | per 256-K block (SASS, NVRTC 13.4) |
|---|---|---|---|---|
| `probe_fp8_128x64` | SS `wgmma.m64n64k32.f32.e4m3.e4m3` | 128 rows × 64 tok | 48 KiB × 4 = 197,696 B | 8 QGMMA, 4 UTMALDG |
| `probe_fp8_128x128` | same, 2 per k32 step | 128 × 128 | 64 KiB × 3 = 197,680 B | 16 QGMMA, 4 UTMALDG |
| `probe_fp8n128_128x128` | SS `wgmma.m64n128k32.f32.e4m3.e4m3`, 1 per k32 step | 128 × 128 | same as above | 8 QGMMA.64x128x32, 4 UTMALDG |
| `probe_w4a16_128x64` | RS `wgmma.m64n64k16.f32.f16.f16` | 128 × 64 | 52 KiB × 4 = 214,080 B | 16 HGMMA, 6 UTMALDG, 96 HFMA2 + 32 HADD2 / thread |
| `probe_w4a16_256x64` | same, 2 row blocks / warpgroup | 256 × 64 | 72 KiB × 3 = 222,256 B | 32 HGMMA, 6 UTMALDG, 192 HFMA2 + 64 HADD2 / thread |

All five kernels: **168 registers, 0 spill, 0 stack**, and no `Potential Performance Loss` (C75xx) note.

**The shared skeleton** is T3's: a persistent grid of #SMs and 384-thread CTAs.
- One producer warpgroup (setmaxnreg 40). Thread 0 issues every TMA; nothing is built by threads.
- Two consumer warpgroups (setmaxnreg 232). They accumulate in f32 over the CTA's whole K range and run one epilogue.
- **Schedules:** tiles (ksplit 1), split-K (ksplit S) or stream-K (ksplit 0) through one `a1_next`, the same in
  producer and consumers. Partials are combined with `atomicAdd` (fast mode's arrival-order fixup) after
  `a1_zero_f32`.

**FP8** (`probe_fp8_*`, `probe_fp8n128_*`):
- Both operands come by TMA, SW128, K-major (FP8 wgmma has no transpose).
- The n128 tile: the m64n128 f32 accumulator fragment is exactly the two n64 fragments end to end (registers 32..63 =
  tokens 64..127). The B descriptor at token 0 with SBO 1024 already spans 128 rows. So it is the 128×128 kernel
  with one instruction per k32 step instead of two, and A is read from smem once: 0.023 B/MAC instead of 0.031.
  Production FP8 GEMMs use N ≥ 128; without it, a KILL could have measured the n64 instruction.
- Epilogue: `acc · ws[row] · xs[token]`.
- Synthetic e4m3 codes (exponent field 3..12, never NaN) and per-row / per-token scales.

**W4A16** (`probe_w4a16_*`):
- Q4_K is repacked on the host (`a1_q4f_pack`). Each consumer thread reads its 8 nibbles per sub-block, for both of
  its rows, as one conflict-free uint4 per 4 sub-blocks.
- The per-32 half2 (s, −z) = (fp16(d·sc), fp16(−dmin·m)) are rounded once from the exact f32 products.
- In registers:
  - LOP3 `(v & 0x000f000f) | 0x64006400` gives 1024 + n; HSUB2 −1024 makes it the exact n.
  - The high nibbles go through `0x00f000f0`, then HFMA2 ×1/16 −64 gives the exact n.
  - One HFMA2 then makes n·s − z.
- The A-fragment slots rotate as in T3: 4 slots and `wait<3>` at 128 rows; 2 slots and `wait<1>` at 256 rows, where 4
  slots spilled.

**Why two tiles per candidate (ESTIMATED).** The operands' L2→SM traffic is bw/T + bx/R bytes per MAC.
- At T3's 128×64:
  - FP8 at the 350 bar needs 1/64 + 1/128 = 0.0234 B/MAC × 350 T = **8.2 TB/s**;
  - W4A16 at the 230 bar needs 0.625/64 + 2/128 = 0.0254 × 230 T = **5.8 TB/s**.
- T3's int8 probe pulled ≈ 4.6 TB/s at its 224.8 (43,008 B per 2.10 M MACs). hopper.md §7 already named L2 as the wall
  near 250.
- The second tiles cut this to **5.5** (FP8 128×128) and **4.0** TB/s (W4A16 256×64), with the same instructions.
- Without them, a KILL could be a measurement of the tile, not the candidate. Each rule takes the family's better
  tile.

**The ceilings:**
- `a1_peak_fp8_ss_n64`, `a1_peak_fp8_ss_n128` (SS e4m3) and `a1_peak_f16_rs_n64` (RS f16) are measured in the same
  job, at 2 WG/SM. They are recorded, not rule inputs.
- `hopper.cu`'s int8 SS n64 measured ≈ 700 TMAC/s on an H100 PCIe (r8). SS n64 sits near the 128 B/clk smem limit at
  full rate (8 KiB per 262 k MACs per CTA step).

## Harness (CODE, tests.rs `a1_*`); lessons applied

- **P1's void** (a driver call in the timed launcher): every launch is an `A1Plan` built before timing. Maps,
  function lookups and smem attributes come first. `fire()` is `Gpu::launch` per launch and nothing else:
  cuCtxSetCurrent (a thread-local swap, no sync), the parameter-pointer Vec and cuLaunchKernelEx. That is the path the
  baselines and production take. (#174's first text said "cuLaunchKernel only", which was inaccurate; amended.)
- **T3's misaligned load:** `a1_layout` checks, before every plan:
  - TMA bases and strides (16 B);
  - SW128 destinations (1024 B) and no-swizzle destinations (128 B);
  - barriers (8 B);
  - expect-tx = the stage's boxes;
  - smem ≤ 227 KiB.

  `a1_layout_pinned` ties the constants to the .cu's `#define`s.
- **A hang must not eat the hour:** every mbarrier wait traps after 4 s (`%globaltimer`).
  - The first compile had a printf in that path. ptxas then serialized every wgmma: C7510, "wgmma pipeline crossing
    function boundary". That is exactly hopper.md pitfall 1. The trap is now printf-free and a test forbids
    `printf(`.
- **Session** (`cuda_a1_session`): stages in child processes, each with a limit, within a 40-min budget.
  1. The sm_90a compile check (its own process too: a failure is recorded and the session goes on, because the kernels
     come from the kit's cubin).
  2. The sanitizer self-check (memcheck over `gpu_rmsnorm_matches_cpu`).
  3. **memcheck on each variant's tiny shapes, first.**
  4. Tiny shapes, then ragged shapes (every output: 1–74 K-blocks; split-K 2 / 3 / 14; stream-K cut inside tiles; more
     CTAs than blocks).
  5. The peaks.
  6. The bench, with a fallback per variant and then baselines only.

  How failures are handled:
  - A **fault** (CUDA error, trap, memcheck error, timeout) drops that variant from the bench.
  - A **tolerance miss** doesn't: its cells carry `tol FAIL`, and a FAIL number doesn't count for the rules.
- **Bench** (`cuda_a1_bench`):
  - The cells and nb are the plan's. The variants are added at nb 128 (split-K + stream-K on down, q/wo and kv;
    stream-K on gate, for information).
  - `gemm2s_q4_k` and `gemmh2_q4_k` run in the same interleaved rounds, on the same Q4_K weights and activations
    that W4A16 is repacked from.
  - Timing is T3's method: per round a warm launch, then 3 launches between events; median of 9 rounds (printed to 2
    decimals).
  - The outputs are NaN-filled before the rounds, so an output no launch writes reads as a tol FAIL.
  - Cells under 10 µs per launch, and every kv nb 128 cell, are marked `~` host-launch-sensitive (the 3 fires partly
    time host enqueue). No rule reads one.
  - **Caveat (the same method as T3, so comparable):** back-to-back launches keep the weights L2-warm where they fit.
    W4A16's gate (≈ 42 MB) and Q4_K's (38 MB) fit H100's 50 MB L2; FP8's 68 MB doesn't.
- **Job** (`scripts/a1/a1_h100.sh`): an nvidia-smi trace (clocks, power, temperature, clock-event reasons, every 500 ms)
  runs over the session, and the verdict summarises it over the bench windows (`A1 CLOCKS`, recorded). The device line
  (name, power limits) goes into the log as `A1 SMI`, which V1 reads. **The result is the `A1 VERDICT` line.**
  `A1 H100 DONE rc 0` means a verdict was printed, with any stage failures listed; rc 1 means there is no verdict.
  - Tolerance check: CPU f64 reference at 4096 sampled outputs plus the corners. The bounds are FP8 rel-rms 0.1 /
    rel-max 0.5 (loose: Hopper FP8 accumulation keeps ~14 bits) and W4A16 0.005 / 0.03.

## Verification so far

| check | result | where |
|---|---|---|
| NVRTC 13.4 sm_90a compile (`nvrtc_probe_fast_compiles_sm90a`, ptxas -v) | PASS: 5 × 168 regs, 0 spill, no C75xx | bella, no GPU |
| `scripts/a1/a1_battery.py`: a1_verdict.py on 14 synthetic logs (healthy at T3-like scaling, host-bound, tol-fail in one family, both KILL scopes, fallback, NVL / PCIe / capped SXM, rehearsal-like, garbage) | PASS: 14 / 14 as expected | bella, no GPU |
| `a1_cell_line_pinned`: the bench's cell line format = the battery's FORMAT line | PASS | bella, no GPU |
| `a1_w4a16_fragments_match_dequant`: the kernel's W4A16 register path emulated on the CPU (its smem addresses, LOP3 / HSUB2 / HFMA2 in fp16, the wgmma A-fragment map) for R 128 and 256 vs `quant::dequant` | PASS: every weight produced exactly once, within fp16 rounding | bella, no GPU |
| `a1_sched_covers`: tiles / split-K / stream-K cover every (tile, block) once, never empty | PASS | bella, no GPU |
| `a1_e4m3_decode`, `a1_layout_pinned` | PASS | bella, no GPU |
| clippy -D warnings (lib + tests), rustfmt | clean | bella |
| bella rehearsal of the H100 kit through `axon-podjob` | see the PR (job id) | bella |
| sm_90a tiny / ragged / bench | **PENDING: needs an H100 (RunPod HOLD)** | — |

## Pre-booking review (shady, 6/10 FIX-THEN-BOOK) and what it changed

- **V3 would have VOIDed a healthy run (the review's M1).**
  - The kernels are persistent, so a cell's time scales with tiles per CTA, not MACs. q/wo nb 128 is one partial wave.
  - T3 (MEASURED, job 1026-222543-3f73) read t(gate 1024) / t(q/wo 128) = 15.5 against a wave ratio of 17.9.
  - The bigger tiles I added (128×128, 256×64) have a wave ratio of 9, so a healthy run reads ≈ 7.8 < 10. Shady's
    synthetic log got `VOID (V3 scaling fp8_128x128 7.8 < 10; ...)` from the old script.
  - The bar is now ≥ 0.5 × the wave ratio: 1.7× under T3's healthy fraction, and ≥ 4.5× over P1's ≈ 1. The old
    script still prints VOID on the battery's HEALTHY log; the new one prints VALID.
- **A tol-FAIL cell could VOID the other family (M2).** `best_tiles()` fell back to non-counting cells, and V3 looped
  over every variant. Now only counting cells feed V2, V3 and the rule cells (battery: TOL-FAIL W4A16).
- **The rehearsal can't reach the candidate paths of the verdict (M3).** That's why the battery exists, with its
  format tied to the Rust formatter.
- **What a rule 1 KILL closes** (the review's SHOULD 8 and the lead's call): the n128 tile, above.

## Caught locally on the way (no GPU needed)

- **A printf in the hang watchdog serialized every wgmma.** ptxas said: "C7510: wgmma.mma_async instructions are
  serialized due to wgmma pipeline crossing function boundary at a function call". That is hopper.md pitfall 1, a
  silent 3–5×. The watchdog now traps without a call, and `a1_layout_pinned` forbids `printf(`.
- **The 256-row W4A16 spilled 132 B** with T3's 4 A-fragment slots. ptxas allocates the consumer within the
  launch-bound 168 registers, not the setmaxnreg 232. At 2 row blocks each commit group is already 4 wgmmas deep, so
  2 slots keep the same in-flight depth: 0 spill.
- **NVRTC 13.4's own compile cache blinds compile-log checks.** A cache hit returns the cubin with an *empty* log.
  - Measured on bella: the same source twice gives 5, then 0 `ptxas info` lines; with `CUDA_CACHE_DISABLE=1`, 5.
  - The cache is keyed on the module's PTX, so a unique `-D` the source never reads still hits (ec121ac tried it;
    0.11 s, empty log).
  - The probe's check compile now appends a uniquely named empty kernel (the PTX changes, so it always misses) and
    requires the ptxas blocks (5dbb56c). With the dumped module: 4 of 4 probe blocks, and 0 on an identical re-run.
  - **main's Hopper compile check (`src/cuda/tests.rs` ≈ L486, "wgmma serialized") has the same blind spot.** It
    compiles identical KERNELS + opts every run. This is not fixed here (out of scope); it was posted to the project
    forum.

## First-run risks on sm_90a (OPINION, with what guards each)

1. **The W4A16 A-fragment layout** (`m64nNk16` f16 from registers) is taken from the PTX ISA and checked by CPU
   emulation only.
   - If the layout is misread, the tolerance fails. The *speed* is unaffected (a K permutation is the same instruction
     stream), but a FAIL number doesn't count.
   - Guard: the ragged stage reports it within seconds.
2. **FP8 accumulation precision** at K 18944. The 0.1 rel-rms bound assumes ≤ a few % error; DeepSeek-V3 reports ~2 %
   at K 4096. The measured errors are printed either way.
3. **TMA boxes of 32 B** (the W4A16 scales) and **256-row boxes** (the 256×64 nibbles and scales) are new. T3 used
   144 B × 128 and 128 B swizzled. The driver encodes them; memcheck runs on them first.
4. **SS n64 smem bandwidth** caps FP8 128×64 near the SS peak. The peak stage measures it.
5. **The atomics' cost** (REDG per output per range) at nb 128 is part of the split-K number on purpose.
6. **The n128 tile** (`probe_fp8n128_128x128`) is a new instruction shape on the same TMA boxes as `probe_fp8_128x128`.
   A wrong B span or fragment order would fail tolerance on the ragged shapes (the rule then reads "KILL at n64 only"
   if the n64 tiles are under the bar, never a false close).
7. **The pod's NVRTC version.** The kit ships 13.2 / 13.3 / 13.4 cubins of all three modules, as P1 did, and the job
   lists any cache entry compiled on the pod.
