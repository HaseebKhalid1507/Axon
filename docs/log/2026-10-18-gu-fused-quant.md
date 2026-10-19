# The FFN hid quantize inside the fused gate + up launch (`AXON_GU_QUANT`, r15 pick 2) (PR #94)

- Date: 2026-10-18. Branch: feat/gu-fused-quant, off main 2b89410. Status: draft, gates and measurement pending.
- Spec: r15-4090-prefill.md §4 pick 2. Evidence for building it: r15-profile.md §5–7.
- Kernel and host change. Bits are unchanged by construction, so there's no `NUMERICS_VERSION` bump. It merges
  cleanly with #92 (feat/prefill-wave-split): the hunks don't overlap (checked with `git merge-tree`).

## What changed
- **prefill.cu `gemm2s_q4_k_gu`** takes four new arguments: `qact`, `qstride`, `want_k` and `cnt`. When `cnt` is
  null the kernel behaves exactly as before. Otherwise `gu_quant_tail` runs after the hid stores:
  - Each thread issues `__threadfence()`; this is belt and braces, since the barrier plus thread 0's fence already
    order the stores. Then comes a barrier, and thread 0 does `__threadfence()` followed by
    `atomicInc(cnt[x · ffn/256 + y/8], 7)`, which wraps 7 → 0 on the 8th arrival.
  - The 8 blocks of one 256-row block (`blockIdx.y / 8`) and one 64-token tile (`blockIdx.x`) all hit the same
    counter. The block that sees 7 is last: its thread 0 fences again (acquire) before the barrier. Its
    `atomicInc` has already put the counter back to 0.
  - That block's 8 warps then run `quant_warp` on the tile's ≤ 64 tokens × 256 values. They load with `__ldcg`
    (L2, so never a stale L1 line), and write into `qact`.
  - This is the same function on the same floats as `quantize_b` (ffn % 256 == 0, so every value is valid). The
    atomic only decides which block does the computing.
  - The kernel also checks rows % 256 == 0 on the device (`cnt && (rows & 255) == 0`). The host checks it too.
  - `out` lost its `__restrict__`, because the tail reads it back. ptxas on sm_89 still reports 193 registers before
    and after, with 0 spills, plus one shared int.
- **batch.rs `ffn_half`**, fused branch (gemm2s path; the sm_90a gemmh2 path is unchanged):
  - With `gu_quant` on and ffn % 256 == 0, the kernel quantizes into **b.gate**. That buffer is free here, because
    the gate product never goes to memory in the fused launch, and nb × act_stride ≤ nb × ffn × 4.
  - The down product reads hid from b.gate, and `quantize_b` doesn't run.
  - It can't write into b.act: other blocks of the same launch are still reading the dim-wide input activations
    there, and the two layouts overlap.
- **`BatchBufs::gu_cnt`**: one u32 per 64-token tile and 256 ffn rows, allocated and zeroed with the batch scratch
  (counted in `batch_bytes`). Every launch leaves it zero again, so repeated launches need no reset. Batched passes
  aren't graph-captured. It assumes one launch uses the counters at a time: the scratch's passes run on one stream,
  and `gemm2s_q4_k_gu` launches without PDL.
- **`AXON_GU_QUANT=0`** (`CudaQwen2::gu_quant`) restores `quantize_b`, with no rebuild. Every other shape and path
  keeps today's code: ffn % 256 ≠ 0, the unfused products, the swiglu epilogue, gemmh2 and gemma.

## Why
r15 profile: `quantize_b` re-reads the whole f32 hid, which is 80 MB per layer on 7B and mostly comes from DRAM. In
the pass profile ("swiglu") it costs 5.1–5.6 % on avante sm_89 3B, 3.2 % on bella 3B, 2.3–2.5 % on bella 7B and
2.0 % on bella 4B. The 4090 7B share is still not measured (ESTIMATED 2.5–5 %). In the fused kernel, the last block
of each group reads 64 × 1 KB that 136 consecutive CTAs (about 1 MB) have just written, while it is still in L2.

## Bits
Same function, same floats, same output layout and stride. Only the buffer address and the computing block change.
Tests:
- `gpu_swiglu_epilogue_and_gate_up_match_reference` adds 512×768 and 1024×256. In both, the kernel's own quantize
  must equal `quantize_b` of its f32 output byte for byte, over two launches, with every counter zero after each
  launch. nb runs 1–129, so it includes partial token tiles.
- **`gpu_gu_quant_matches_quantize_b_many_waves`** (new, fast tier): 4096 × 1024 at nb 1000–1095. That is
  128 × 16–17 blocks over many waves, so they finish genuinely out of order. It runs 20 launches, and after each:
  qact == `quantize_b(out)` byte for byte, every counter is zero, and out has the same bits as launch 0. GPU
  against GPU, with no `gemv_ref`.
- `prefill_bits` (0.5B, 1.5B, Q8 KV, and the other models that call it) adds `prefill_logits` with `gu_quant` off
  from 0 and from 300. Every case must match the per-token reference's logits and KV.
- `gpu_prefill_logits_match_forward_*` run with it on (the default).
- **`gpu_gu_quant_repeat_stress`** (new, ignored): the same 2100-token prompt (two passes, each with a partial
  tile) prefilled 200× on 3B. Every run's hash of logits + KV must equal the `quantize_b` run's, with the counters
  zero after each run. It fails if the model doesn't take the fused path.

The risk is a missing or wrong fence, which would show up as rare nondeterministic mismatches (hopper.md pitfall 2
is the same class). That's what the 200× stress on avante (sm_89) and bella (sm_120) is for. The 4090 run named in
the spec isn't done: there are no pods for this round.

## Verification and measurement
Review: Shady 8/10 MERGE-AFTER-GATES. SHOULD 1–4 and NITs 1–6 applied in b8c7d8c; SHOULD 5 is the follow-up below.
The a17b7fd gate jobs were all cancelled before they started and resubmitted at the new head.

Binaries are at a17b7fd. clippy `-D warnings` (bin + tests) and rustfmt: PASS, locally.

| gate | job | status |
|---|---|---|
| fast tier, release, `--shard 2` | group 1018-195144-1610 (64fa, 792d) | PENDING |
| full debug-assertion suite, GPU, `--shard 2` | group 1018-195148-1974 (0cc5, 8766) | PENDING |
| bella: rel bit suite + 7B/8B `prefill_logits_match_forward_big` + 200× stress (3B) + md5 vs main 2b89410 (6 models × 5 prompts) | 1018-195152-9505 | PENDING |
| avante: 200× stress (3B) | 1018-195152-495c | PENDING |
| A/B avante 3B, `--timing avante`, nb 1050 at 0, 7 rounds interleaved | 1018-195159-357b | PENDING |
| A/B bella 7B, `--timing bella` (waits for night mode), nb 1050 at 0, 7 rounds interleaved | 1018-195159-fc68 | PENDING |

A/B method: `cuda_prefill_profile` with `AXON_PROF_GUQ_AB=7`. It uses the same process and the same cache. Each
round runs off and on in alternating order, each with one warm-up pass, one callback-free pass (GPU events) and
one evented pass, and reports medians of gate_up, swiglu, their sum, down and the pass. The kill rule (spec): gate_up
+ swiglu must fall by ≥ 60 % of the old swiglu time. The job prints that ratio.

## Results
- pp (prefill pass, A/B above): PENDING on both hosts. The spec ESTIMATES 4090 7B at +2–3.5 %, bella 7B at
  ~+1.5–2 % and avante 3B at ~+4 %.
- tg: doesn't apply. Decode never runs `gemm2s_q4_k_gu` (batched passes over `VM_MAX` tokens only), so it isn't
  measured.
- The kill rule is checked from the A/B's own "kill rule" line. If gate_up + swiglu don't fall by ≥ 60 % of the
  old swiglu time, the pick dies and this PR becomes a NO-GO entry.

## What didn't work / open
- Quantizing into b.act (as `quantize_b` does) is unsafe here: other blocks of the same launch still read the
  input activations from it. So it goes into b.gate instead.
- Follow-up (review SHOULD 5): once #92's wave split lands, rerun `gpu_gu_quant_repeat_stress` on main. That
  covers the 1088 + 1012 pass sizes, which put the partial tile in a different place.
- Reading the A/B: in the ON arm the "swiglu" row is only the profile callback's event overhead. No kernel runs
  there.
- Not measured: the 4090 (spec step 0 share, A/B and the stress test). No pods this round.
