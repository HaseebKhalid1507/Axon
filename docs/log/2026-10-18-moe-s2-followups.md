# 2026-10-18: MoE stage 2 follow-ups (Shady's final review of #67)

Five small fixes from the #67 review (MERGE 9/10). No output change: greedy bits are identical to main
(md5 below). Hunks kept small because feat/gpt-oss and feat/deepseek2 stack on this code.

## What changed, and why

1. **`moe_step` checks the paged table in debug builds.** Dense `step_with` calls `c.debug_covers(0, pos + 1)`;
   `moe_step` didn't, so a paged overrun in MoE decode went unnoticed in debug builds. It calls it now, right after
   the `pos < max_ctx` assert, the same place as the dense step.
2. **A dense batched pass on a MoE model returns an Err.** `pass_hooked` with no hook on an `Ffn::Moe` layer used to
   reach `self.layer` → `LayerW::dense()` → panic. It now returns
   `Err("a mixture-of-experts model's batched pass needs its experts (crate::moe: moe_tokens), not the dense pass")`
   before it enqueues anything (the check runs first in the function, so no half-run pass is left on the stream).
   Test: `cuda::qwen2::moe::tests::moe_gpu_dense_pass_is_an_err` calls `pass_with` (what a stacked branch would
   use) on the 4x0.6B and gets the Err.
3. **`moe_check_ram` counts only what the host holds.** The split (`crate::moe::MoeQwen3`, attention on the GPU) keeps
   just the experts in the host's r8 arena. The attention, embedding and head stay in the mapping or on the GPU. The
   check still billed `dense_bytes` to RAM, which is why it refused the Pruned-2B on jade. `check_ram(m, dense)`:
   `dense` is true for `--device cpu` (unchanged: experts + dense + 1.5 GiB headroom) and false for the split
   (experts + 1.5 GiB headroom). The refusal message names which one it means ("the CPU model" / "the CPU experts").
   Test: `ram_need_counts_the_arena_only_when_repacking` was extended with the split's need and a case where RAM
   holds the experts but not the dense weights: the split passes and the CPU model is refused.
4. **`moe_step` takes `Head::Argmax` only.** With `Head::No`, the last segment only added the last layer's FFN into x,
   and the next step's embedding overwrites x anyway. Its only caller (`MoeQwen3::step`) passes `Argmax`. I picked
   the assert, not removing the parameter: it's one line, and the signature that the stacked branches call stays
   the same. The graph table keeps its `seg * 4 + head` layout. Test: `moe::tests::moe_gpu_step_refuses_head_no`.
5. **One banner line per MoE run.** `axon gen` printed `moe: <describe>` at load (lm.rs) and again
   `device: cuda + cpu experts, <describe>, kv …` at the end (main.rs). I kept the load-time `moe:` line, since serve
   and chat print it too, and moved the kv type onto it (`moe: …, kv f16`). I dropped main.rs's `Backend::Moe`
   block. No README or doc quotes either line (grep: only `identity.rs`'s `cuda+cpu experts`, which is the identity
   string and isn't changed).

## RAM need, before → after (from `axon info` sizes; formula = `ram_need`, headroom 3 << 29 B = 1.61 GB)

RAMNUMBERS

## Verification

EVIDENCE

## What didn't work

- An associated `const` with a `&str` inside `impl<'g> CudaQwen2<'g>` is refused by rustc
  (`elided_lifetimes_in_associated_constant`, deny by default). The message became a module-level const instead.
