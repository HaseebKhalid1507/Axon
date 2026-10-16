# research: attention kernels on consumer Blackwell (sm_120) — long-context decode and prefill

Read-only research round, 2026-10-16. Tree read: Axon main e998d6f (exported with `git show` to a scratch directory),
the lcdec worktree (fb4ed4a) and both builders' NOTES, llama.cpp 4413a19 (2026-10-12, local checkout),
FlashAttention-2 `flash_fwd_launch_template.h` (GitHub main). Papers: read in full = Lean Attention, FlashDecoding++,
FlashInfer (pdftotext); abstracts + known method for the rest. Labels: **MEASURED** / **ESTIMATED** (arithmetic shown) /
**CLAIMED** / **CODE** (file:line).

---

## 1. The problem in Axon's terms

MEASURED (r9-pick §0, bella, medians of 2 rounds): at depth 32512, 7B Q4_K_M tg **72.4 vs llama.cpp 76.7 (−6 %)**, 0.5B
**413 vs 464 (−11 %)**; Qwen3-8B (8 kv heads) +11 %. Both losers have ≤ 4 kv heads. `--kv q8` buys nothing on main (70.9).

MEASURED (lcdec NOTES, `cuda_profile` ctx 32768 pos 32511): 7B `attn_blk_128` **114.7 µs/layer** in every variant tried
(1 warp, 4 warps, gy = nblk) = "~580 GB/s on 66.6 MB"; `attn_cmb` 32 → 19 µs after the builder's rewrite; 0.5B attn 32 µs,
combine 24 → 14 µs. q8 7B attn 143 → 70 µs with 4 warps; q8 tg@32k +13.8 %.

The arithmetic that reframes it (ESTIMATED): `attn_blk` also **writes** a partial per (head, 64-block):
7B = 28 heads × 508 blocks × 130 floats × 4 B = **7.4 MB/layer = 11 % of the KV bytes**. (66.6 + 7.4) MB / 114.7 µs =
**645 GB/s — the kernel is already at r4's 648 GB/s ceiling.** The f16 kernel has no headroom left; what Axon pays
over llama.cpp is (a) the partial traffic itself and (b) the serial combine that re-reads it. Marginal cost 8k → 32k
(r9 table): Axon 3.26 ms, llama.cpp 2.2 ms for the same 1.39 GB of KV ⇒ llama.cpp's marginal rate is **≈ 630 GB/s**
(at the ceiling), Axon's 427. Per layer per 24k positions the 38 µs gap splits ≈ 27 µs combine (main) + 9 µs partial
bytes. With lcdec's 19 µs combine the remaining 7B gap is ESTIMATED ≈ 0.6 ms/token ≈ parity.

Prefill: 7B pp32k ties llama.cpp (2682 vs 2566); attention ≈ 48 % of the pass at 56–69 % of the 64.8 TFLOP/s f16→f32
`mma.sync` peak (r5 §2 MEASURED). lcpre: ping-pong (FA3-style) is **−2.4…−3 % on a 5090, +2…5 % on sm_89** → NO-GO;
`bs_128` (4 warps, 2 CTAs/SM) is **+20 % on the 7B shape on the 5090, −12 % on avante**; bella (46 SMs, 5.39 waves)
unmeasured at 16k–32k.

---

## 2. Literature map

| # | paper | the idea | numbers (label) | hardware |
|---|---|---|---|---|
| 1 | **FlashAttention** (Dao et al., 2022, arXiv 2205.14135, NeurIPS) | Tiling + online softmax; never materialize S; IO-optimal for SRAM size | 3× GPT-2 training speedup (CLAIMED) | A100 |
| 2 | **FlashAttention-2** (Dao, 2023, arXiv 2307.08691, ICLR'24) | Fewer non-matmul FLOPs (one rescale per tile, not per row), parallelize over seq-len for low batch, split work so each warp owns its rows (no smem exchange for O) | 50–73 % of peak fwd; ~2× FA1 (CLAIMED). CODE: hd128 on sm86/89 uses **64×64 tiles, 4 warps** for causal ("square is fastest"), 128×32 non-causal (2 CTAs/SM); split-KV decode: 64×128 tiles, 4 warps, `num_splits` by SM count, tree combine | A100, sm86/89 |
| 3 | **FlashDecoding** (Dao, Haziza, Massa, Sizov, PyTorch blog Oct 2023, no paper) | Split-KV: parallelize decode over the KV length, write (m, l, o) partials, reduce in a second kernel | up to 8× faster decode at 64k vs FA2 on A100 (CLAIMED) | A100 |
| 4 | **FlashDecoding++** (Hong et al., 2023, arXiv 2311.01282, MLSys'24) | "Unified max": a pre-estimated per-layer max φ lets partial softmaxes skip the rescale (asynchronous), fall back to the synchronized path on overflow; flat-GEMM double buffering; heuristic dataflow | 1.37× avg over FlashDecoding, up to 4.86× vs HF (CLAIMED); synchronized partial-softmax update costs ~18.8 % of attention (Llama2-7B, A100, MEASURED by them) | A100, RTX 3090, MI210 |
| 5 | **Lean Attention** (Sanovar et al., 2024, arXiv 2405.10480, v2 Jan 2025) [read] | Stream-K style decomposition of decode attention: equal "LeanTile" counts per CTA across heads/batch, the online-softmax merge treated as an associative reduction with a fixup; targets 100 % SM occupancy whatever the problem size | 1.73× avg, up to 2.18× at 256k vs FlashDecoding (CLAIMED) | A100/H100 |
| 6 | **FlashInfer** (Ye et al., 2025, arXiv 2501.01005, MLSys'25) [read] | Block-sparse KV format; JIT templates; **plan-based, load-balanced persistent kernel compatible with CUDA graphs** (fixed grid, workspace offsets); explicitly **no atomic aggregation — deterministic reduction order for identical length info** (§3.3.1) | 29–69 % ITL reduction vs compiler backends, 28–30 % long-context latency (CLAIMED) | H100/A100 |
| 7 | **FlashAttention-3** (Shah et al., 2024, arXiv 2407.08608, NeurIPS'24) | Warp-specialized producer/consumer with TMA; **2-warpgroup ping-pong** so one group's softmax overlaps the other's GEMMs; intra-warpgroup GEMM/softmax pipelining; FP8 with block quant + incoherent processing | 740 TFLOP/s f16 = 75 % of H100 (MEASURED by them). Needs wgmma async — not on sm_120 | H100 |
| 8 | **ThunderKittens** (Spector et al., 2024, arXiv 2410.20399) | 16×16 register tiles, a CTA template for overlapping async ops across warps (load/compute/store workers), grid-level launch hiding | matches FA3 on attention inference; attention bwd +10–40 % (CLAIMED) | H100 (4090 ports exist) |
| 9 | **POD-Attention** (Kamath et al., ASPLOS'25, arXiv 2410.18038) | Co-schedule a prefill CTA (compute-bound) and a decode CTA (bandwidth-bound) on the same SM for hybrid batches | attention up to +59 % (mean +28 %) vs independent kernels (CLAIMED) | A100/H100 |
| 10 | **SageAttention** (Zhang et al., 2024, arXiv 2410.02367, ICLR'25) | INT8 Q·Kᵀ (K smoothing), **PV in FP16 with FP16 accumulate** (2× rate on GeForce), per-block scales | 2.1× FA2 OPS on RTX 4090 (CLAIMED) | RTX 4090 |
| 11 | **SageAttention2** (2024, arXiv 2411.10958) / **SageAttention3** (2025, arXiv 2505.11594) | INT4 QK per-thread quant + FP8 PV, two-level accumulation; SA3: microscaling **FP4 on Blackwell** tensor cores | SA2 3× FA2 on 4090; SA3 **1038 TOPS on RTX 5090 = 5× the fastest FA on 5090** (CLAIMED) | 4090, **5090 (sm_120)** |
| 12 | **MpFA** (2026, arXiv 2609.33135) | QK NVFP4 + PV FP8 on Blackwell; finds fully-FP4 attention fails from non-matmul costs (online quant, smem traffic, softmax contention); rank-one compensation | 2.81× e2e over BF16 FA4 at 16k–128k (CLAIMED) | B200 |
| 13 | **INT-FlashAttention** (2024, arXiv 2409.16997) | Fully INT8 attention in the FA forward on Ampere | +72 % speed vs f16 FA (CLAIMED) | A100 |
| 14 | **Atom** (Zhao et al., MLSys'24, arXiv 2310.19102) | W4A4 + **KV4** with per-group asymmetric quant; attention kernel dequantizes KV tiles in-register before the mma | up to 7.7× throughput vs FP16 (CLAIMED) | A100 |
| 15 | **QServe** (Lin et al., 2024, arXiv 2405.04532) | W4A8**KV4**; the key insight: CUDA-core dequant is the bottleneck → "progressive" dequant, SmoothAttention for KV4 | 1.2–1.4× TensorRT-LLM (CLAIMED) | A100/L40S |
| 16 | **KIVI** (Liu et al., 2024, arXiv 2402.02750, ICML'24) | K per-channel, V per-token 2-bit asymmetric quant with a small f16 "residual" window; a fused dequant-attention kernel | 2.6× peak memory ↓, 2.35–3.47× throughput ↑ (CLAIMED) | A100 |
| 17 | **Hardware-Efficient Attention for Fast Decoding** (Zadouri, Strauss, Dao, 2025, arXiv 2505.21487) | GTA/GLA: raise arithmetic intensity per KV byte; shows GQA decode is KV-byte-bound and that the kernel's job is purely bytes/s | GLA kernel 2× FlashMLA in spec-decode (CLAIMED) | H100 |
| 18 | **Defeating Nondeterminism in LLM Inference** (Thinking Machines, blog Sept 2025) | Batch-invariant kernels; for split-KV: **fix the split *size*, not the split count**, so the reduction order never depends on batch/shape | — (engineering note) | H100 |
| 19 | **CoRun** (2026, arXiv 2608.14376) / **Vosti** (2026, arXiv 2609.38981) | Deterministic inference: CoRun = isolated prefill + fixed-shape batched decode via CUDA graphs (batch-invariant kernels cost >2× latency); Vosti = formally verified bitwise-identical logits across batching/chunking/paging | CoRun +15–324 % throughput vs batch-invariant kernels (CLAIMED) | H100 |
| 20 | **MOIRA** (2026, arXiv 2610.04313) | Self-planning sparse decode: each CTA derives its work from page-list lengths inside the graph | approximate (sparse) — not for Axon's default | H100 |

Takeaways for sm_120: every fast dense kernel on consumer parts (FA2 sm86/89, llama.cpp, SageAttention on 4090/5090)
is **4 warps per CTA, 64-row KV tiles, 2 CTAs/SM, cp.async double buffering**; the only >1.3× levers on GeForce beyond
that are precision ones (f16-accumulate PV, INT8/FP4) which break the contract. For decode, the state of the art
(FlashInfer, Lean, llama.cpp) is a **persistent CTA walking a contiguous KV range with an in-register online softmax
and a handful of partials per head**, reduced in a fixed order.

---

## 3. What production engines do

### llama.cpp (4413a19), the kernel that beats Axon at 32k
- **Kernel choice** `fattn.cu:638-665`: on Ada+ with f16 KV, `Q->ne[1]==1` uses the **vec** kernel only when
  `!(gqa_ratio > 4 && (D >= 256 || K->ne[1] >= 8192))` (`:645-647`) → 7B/0.5B (gqa 7) at ≥ 8192 run the **MMA f16**
  kernel; `fattn.cu:266-267` picks `ncols2 = 8` for gqa > 4, `:170-174` `ncols1 = 8/ncols2 = 1` for one query.
- **Config** `fattn-mma-f16.cuh:59` `(128,128, ncols 8)`: 128 threads (4 warps), occupancy 2, **nbatch_fa = 128 KV rows
  per rescale**, K/V tiles 64 half2 wide, **nstages 2 (cp.async)**, Q in registers; `:39` same for hd 64.
- **Operand layout** `:673-674`: for `cols_per_warp == 8` the mma is `mma(KQ_C, K_A, Q_B)` — **K tile is the A operand
  (16 KV rows per m-tile), the 8 GQA heads are the N dimension**. Per 64 positions at hd 128 that is 32 + 32 = 64 mma
  per CTA; Axon's Q-as-A layout (`decode.cu:1301-1317`, `:1378-1393`) issues **128** for the same 64 positions with 9/16
  of each m-tile padding (group 7). The 4 warps split the 128-row batch (`np = 4`, `:609`), each 32 rows, and merge
  maxima/rowsums through smem once per batch.
- **Parallelization** `fattn-common.cuh:1145-1186`: Ada+ always uses **stream-K** (`:1156-1158`): `nblocks =
  min(max_blocks_per_sm × nsm, ntiles_KV × ntiles_dst)` rounded to a multiple of `ntiles_dst` (`:1173-1185`);
  `max_blocks_per_sm` from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (`:1137`). ESTIMATED for bella 7B at 32k:
  smem per CTA = 128 × 128 × 4 B + mask = 65.8 KB (`:2033-2043`) → 2 × 65.8 > 128 KB ⇒ **1 CTA/SM, 46 CTAs**, each walking
  ≈ 254 × 4 / 46 ≈ 22 contiguous 128-row tiles (`fattn-mma-f16.cuh:1899-1908`, kbc ranges) with a running online
  softmax in registers; **≈ 12 partials per head** instead of Axon's 508.
- **Fixup** `fattn-common.cuh:729-809`: one block per output tile, **a sequential fold from `b_last-1` down to
  `b_first`** (`:788-805`) — fixed order, no atomics, deterministic for a fixed shape (but shape-dependent: the
  block count depends on `Q->ne[1]`, so it is not batch-invariant in Axon's sense).
- `flash_attn_mask_to_KV_max` (`:671-723`, launched `:1122-1131`) scans the mask so the kernel stops at the last
  valid position, not the padded `ne11`.
- 2026 addition: a mask-driven **sparse gather** path for decode (`fattn.cu:128-152`, `K->ne[1] >= 4096`) — only pays
  when the mask is sparse (SWA/tree); dense causal gains nothing.
- Prefill `fattn-mma-f16.cuh:62` `(128,128, ncols 64)`: 128 threads, occupancy 2, nbatch_fa 64, nstages 2 → **4 warps,
  2 CTAs/SM, 64-position tiles** = Axon's `attn_fa_bs_128` geometry but with a K/V alternating prefetch (`:629-637`,
  `:981-994`: wait K(b) → issue V(b) → QK → wait V(b) → issue K(b+1) → PV).

### vLLM / SGLang / TensorRT-LLM (docs)
- vLLM default decode on NVIDIA: FlashAttention-2/3 split-KV (`num_splits` by SM count) or FlashInfer (persistent
  plan, `:3.3.1` above); `--enable-batch-invariant` mode (Sept 2025) uses fixed-size KV splits and loses throughput
  (CoRun: >2× latency, −74 % throughput CLAIMED).
- SGLang: FlashInfer by default (deterministic mode since 2025 uses fixed split sizes), Triton decode kernel as
  fallback (split-KV with `num_kv_splits` fixed per launch).
- TensorRT-LLM: XQA kernels for GQA decode (heads packed as the mma N dimension, KV int8/fp8 dequant in-kernel —
  the same layout llama.cpp uses); FP8/INT8 KV cache as the long-context lever.

---

## 4. Proposals for Axon, ranked

### A. Decode: fold the partials *while* `attn_blk` runs ("chase fold"), contract-preserving — **rank 1**
**Change.** `src/cuda/decode.cu`: add fold CTAs to the `attn_blk_*` launch (lowest `blockIdx.y`, one CTA per query head:
`n_heads` CTAs × `hd` threads, 7B 28, 0.5B 14). Attention CTAs store the block's `(o, m, l)` as now, then
`__threadfence()` + write a per-(head, block) ready flag = `np` (monotonic: no reset between steps, works inside the
graph). Fold CTAs run `attn_cmb_dev`'s exact loop (`decode.cu:2038-2068`, superchunk scan + `att_fold_o` chain in
ascending block order) but spin on the flag of each block before consuming it, then `__fdiv_rn`, `y`, `quant_warp`.
Drop the separate `attn_cmb` launch for the f16/q8 tensor-core path (`qwen2.rs:1812-1821`); `attn_cmb_b` (batched,
#38) untouched. Builder: 12–20 h.
**Gain (ESTIMATED).** The combine disappears from the critical path (fold throughput ≫ block arrival rate: one block
per head lands every ~0.2 µs; one fold step is ~10 cycles): 7B −19 µs/layer (lcdec c5) … −32 (main) × 28 = **−0.5 … −0.9
ms/token at 32k → 72 → 76–78 tok/s**; 0.5B −14 … −24 µs × 24 = **−0.34 … −0.58 ms of 2.42 → 480–540 tok/s** (llama 464).
At pos 8 the fold has 1 block: cost ≈ the 28 extra tiny CTAs (≤ 2 µs/step ESTIMATED; gate below).
**Bits.** Identical: the same `att_fold_w/att_fold_o` chain in the same order; which CTA runs it and when is not
arithmetic (contract `decode.cu:1151-1157`).
**Risk.** Medium: forward progress (fold CTAs must be scheduled with, not instead of, attention CTAs — 28 CTAs of 128
threads vs 3 × 46 attention slots; use `blockIdx.y < n_fold` and `__launch_bounds__`; PDL `griddepcontrol.wait` already
in `PDL_START`); the q8 "own block stores new row then re-reads scales" path (`:1923-1940`) unaffected; the
`gpu_decode_matches_batched*` and `graph*` suites catch any ordering bug.
**Gate.** `cuda_profile` ctx 32768 pos 32511: attn+combine 7B ≤ 120 µs/layer (from 134), 0.5B ≤ 36 (from 46);
pos 8 step ≤ +0.5 %; `gen` tg@32k 7B ≥ 76, 0.5B ≥ 470 (≥ 5 interleaved rounds); bit suites + `gpu_decode_attention_long`.

### B. Decode: keep the partials L2-resident (cache-policy hints), contract-preserving — **rank 2 (1-h experiment first)**
**Change.** `decode.cu:1198-1202` `cp16z`: add `createpolicy.fractional.L2::evict_first` + `cp.async.cg.shared.global.
L2::cache_hint` for the KV stream (and `att_load_q8`); store `part_o/part_ml` with `st.global.L2::evict_last` (or
default). 7.4 MB (7B) / 1.9 MB (0.5B) of partials vs a 36 MB L2 (r4 MEASURED): if the KV stream stops evicting them, the
next layer's stores overwrite dirty lines in place and the write-back to DRAM mostly vanishes; the combine (or A's fold)
reads L2 hits. **Gain (ESTIMATED):** up to the 11 % partial bytes: 7B −9 µs/layer (−0.25 ms, +2 %), 0.5B −3 µs; plus a
faster combine. **Bits:** none change. **Risk:** low; r4 found `cudaAccessPolicyWindow` dead on this GPU, but PTX
`createpolicy` is a different mechanism (sm_80+). 2–4 h. **Gate:** `attn_blk` in-kernel ≤ 106 µs at 32k on 7B
(= 66.6 MB / 630 GB/s); kill if < 3 %.

### C. Prefill: SM-count-aware CTA geometry + pass sizing for wave fill, contract-preserving — **rank 3**
**Change.** (i) `batch.rs:1154` `small` rule: choose `attn_fa_bs_128` (4 warps, 2 CTAs/SM) vs `attn_fa_b_128` from
`(CTAs, sms)` and shape, measured on bella — FA2 picks 64×64/4-warp on sm86/89 and lcpre measured bs +20 % on the 7B
shape on the 5090 (geometry, not ping-pong). (ii) `batch.rs:1565 passes()` / `qwen2.rs:91 PREFILL_BATCH`: size passes
so `n_kv × ceil(nb·group/rows)` ≈ k × 46: 7B today 1121 tok → 248 CTAs = 5.39 waves (last wave 39 % full); 1042 tok →
228 = 4.96 waves. **Gain (ESTIMATED):** (ii) −8 % attention at 16k–32k ≈ **pp32k +3–4 %**; (i) if bella behaves like
the 5090 on 7B, attention −15 % ≈ pp32k +7 %. **Bits:** identical (`passes` doc: "Results don't depend on it"; rows
are independent, `prefill.cu:366-377`). **Risk:** low. 4–8 h + bench. **Gate:** lcpre's `cuda_fa_bench` (c3-test,
`AXON_FA_NZ=1121,1042,1016`) on bella, 7B/q3-8B/0.5B shapes at 16k and 32k; then pp8k/16k/32k ≥ 5 rounds.

### D. Prefill: 4-warp CTA with llama.cpp's alternating K/V prefetch (3 tiles = 48 KB, 2 CTAs/SM) — **rank 4**
**Change.** `prefill.cu:378-503` `attn_fa_bt`: a variant `<128, 4, ST=…>` whose load schedule is: wait K(b) → issue
V(b) → `att_scores` → softmax/fold → wait V(b) → issue K(b+1) → `att_pv`. Smem 3 tiles (48 KB) so 2 CTAs/SM stay
resident (`bs_128` today has ST = 1, no prefetch). The scheduler then interleaves two independent CTAs — the "free
ping-pong" FA2 relies on at 2 CTAs/SM, without named barriers or the register cost that sank lcpre's `attn_fa_p_128`
(255 regs). **Gain (ESTIMATED):** tensor efficiency 60 → 70 % → attention −14 %, **pp32k +5–7 %** on hd 128. **Bits:**
identical (load timing only). **Risk:** medium (register budget at 242 for `bs_128`; measure, do not guess). 10–20 h.
**Gate:** `cuda_fa_bench` 7B shape 1121@32k on bella ≤ 0.87 × `attn_fa_b_128`; q3-8B/0.5B no worse; bits suites.

### E. Decode: K-as-A / Vᵀ-as-A operand layout (llama.cpp's), halves the mma count — **rank 5, blocked on a test**
Only pays where the kernel is not DRAM-bound: the q8 path (70 µs for 33 MB = 475 GB/s, convert + mma bound) and hd 64.
**Bits:** the contract's row-independence argument (`decode.cu:1151-1154`) would have to hold with the roles swapped
(a column's result from column j of B only, and the k-reduction identical under operand swap) — plausible, **unverified**.
Run the 1-h experiment in §6 first; if it passes, 10–16 h for the q8 kernels. Gain ESTIMATED q8 7B@32k attn 70 → ~55 µs
(bytes/648 GB/s = 51), +3 % tok/s on q8.

### F. Opt-in only: two-level fold (superblocks of 8 blocks) — a contract change
`attn_blk` CTA owns a contiguous superblock (double-buffered loop, fold in registers from (−inf, 0, 0)), one partial per
512 positions; combine chain 64 long; prefill folds the superblock partial into a top-level state kept in smem (the
8-warp hd-128 kernel has no registers for a second O). Partial bytes ÷ 8, combine ÷ 8: ESTIMATED 7B −24 µs/layer (+5 %),
0.5B +14 % — but A already hides the combine and B the bytes, and this re-keys every GPU md5, the q8 kernels,
`attn_blk_b`/`attn_cmb_b` and the verify path. 30–50 h. Do only if A+B leave > 3 % on the table.

### G. Opt-in only (`--attn fast`): f16-accumulate PV (SageAttention) / INT8 QK
2× tensor rate on GeForce for PV; breaks bits and the KL gate; r5 §4.2 notes Qwen2's bias-dominated keys. Not default.

---

## 5. Dead ends

- **Ping-pong warp groups in `attn_fa_bt`** (r5 P4): MEASURED −2.4…−3 % on sm_120 (lcpre). The register cap (255) and the
  extra barrier eat the overlap; sm_89 +2–5 % does not transfer. Superseded by 2 CTAs/SM (C/D).
- **Lean Attention / stream-K with atomic or shape-dependent aggregation**: order depends on CTA count → not
  batch-invariant; llama.cpp's own fixup order depends on `Q->ne[1]`. Axon's fixed 64-block partials are exactly the
  "fixed split size" rule of Thinking Machines (#18) — keep it.
- **FlashDecoding++ unified max**: changes numerics, and Axon's fold is already asynchronous-safe; the 18.8 % it
  saves is the rescale, which Axon pays only in the (hidden by A) combine.
- **Larger blocks / f16 accumulate / FP8-FP4 (SageAttention3, MpFA)**: contract + accuracy; sm_120 has no FP4 path that
  is bit-compatible with the f16 canonical order.
- **FA3/ThunderKittens producer-consumer with TMA/wgmma**: sm_90a only; on sm_120 the tools are cp.async + mma.sync.
- **More `gy`/one block per warp** (r9 Stage 1a): MEASURED no gain (114.7 µs in all variants), and gy = nblk_max cost
  +22 % at pos 8 with a 32k cache — the kernel is DRAM-bound; killed by the builder, correctly.
- **4 warps per block for hd 64**: MEASURED worse at 8192 (13.7 vs 11.9 µs); 0.5B stays single-warp.
- **KV int4 / KIVI 2-bit (Atom, QServe)**: bytes halve again (0.5B/7B tg@32k would gain ~1.4 ms on 7B ESTIMATED), but
  it is a new cache format that must round-trip prefill/decode identically and llama.cpp compatibility (q4_0 KV) is a
  separate format; keep as a later `--kv q4` round after q8 is the documented long-context default.
- **POD-Attention**: serving-level; relevant to r9 #3b (chunked prefill co-running with decode), not to single-stream
  tg. Note for that round: a prefill chunk's `attn_fa_b` already overlaps decode attention across SMs if launched on a
  second stream; POD's SM-level fusion is a 40+ h item.
- **Persistent megakernel / CUDA-graph gaps**: r5 §1.3 measured 1–2 µs per launch; A removes one launch per layer
  anyway.

---

## 6. Open questions a 1-hour experiment settles

1. **Are the partial stores on the critical path?** Bench `attn_blk_128` at 32k with the `part_o` stores compiled out
   (scratch only): expect 114.7 → ~103 µs if the 645 GB/s arithmetic is right. Decides B's ceiling.
2. **L2 cache hints** (B): `createpolicy` evict_first on `cp16z`, evict_last on the partial stores; `cuda_profile`
   attn + combine at 32k, 7B and 0.5B.
3. **Transposed-mma determinism** (E): extend `mma_f16_is_deterministic` — compute S = Q·Kᵀ with Q as A vs Sᵀ = K·Qᵀ with K
   as A for the same rows/columns, compare bits over random f16 inputs incl. Qwen2-scale bias channels; same for
   P·V vs Vᵀ·Pᵀ.
4. **bella prefill geometry** (C): lcpre's `c3-test cuda_fa_bench`, `AXON_FA_NZ=1121,1042,1016`, kernels b/bs, 7B and
   q3-8B shapes at 16k/32k, 5 interleaved rounds — the one bella number the NO-GO verdict lacks.
5. **Where do 7B's 5.3 ms go?** `cuda_timeline` at pos 32511: attn 3.2 + combine 0.9 ms (main) sum to 4.1 ms; the
   remaining ~1.2 ms is clocks/thermal or exposed gaps — it changes whether A+B reach the +12 % gate or "only" parity.
6. **The right bar for q8**: `llama-bench -fa 1 -ctk q8_0 -ctv q8_0 -d 32512` on 7B/0.5B; Axon's q8 +13.8 % (lcdec) is
   compared today against llama.cpp's f16 KV.

---

## Summary (5 lines)

f16 decode attention is already at the DRAM ceiling once its own 11 % partial-store traffic is counted (645 GB/s);
the 7B/0.5B deficit at 32k is the serial `attn_cmb` (≈ 70 % of the gap) plus those partial bytes (≈ 25 %), while
llama.cpp's stream-K CTAs carry ~12 partials per head and fold them in a tiny fixup. Rank 1: fold the partials in
flight inside the `attn_blk` launch (same order, same bits: hides 0.5–0.9 ms/token on 7B, 0.3–0.6 ms on 0.5B); rank 2:
L2 cache hints to keep partials resident; prefill: 4-warp CTAs at 2 CTAs/SM with K/V alternating prefetch and
pass sizes that fill 46 SMs — the geometry FA2 and llama.cpp both use on consumer parts — not ping-pong.
