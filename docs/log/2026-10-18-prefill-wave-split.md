# Prompt passes sized to the GEMM waves (`AXON_WAVE_SPLIT`, r15 pick 1)

- Date: 2026-10-18. Branch: feat/prefill-wave-split, off main 50cd609. Status: draft, measurement pending.
- Spec: r15-4090-prefill.md pick 1 (Yoru). Host-only change: no kernel, no numerics, no `NUMERICS_VERSION` bump.

## What changed
- **`batch::wave_passes`**: for a prompt that needs several batched passes (n > the batch max, start + n <
  `FA_LONG`), choose the pass lengths with the least modelled GEMM time. The pass count stays
  `n.div_ceil(max)` and no pass is longer than max, so the batch scratch and memory plan are untouched. All passes
  but the last are whole 64-token tiles, and the last is over `VM_MAX` tokens, so every pass stays on the tiled
  GEMMs the model describes. A split must model at least 1 % (`WAVE_MARGIN`) less than the equal split, or the
  equal split stays. Ties go to fewer token tiles, then the more even split, then the shorter last pass
  (2100 → 1088 + 1012).
- **`batch::gemm_cost`**: one pass's GEMM time is Σ over the products of ⌈blocks / SMs⌉ × a block's MACs. It uses
  `bgemm`'s grids: Q4_K / Q5_K gemm2s 64 × 64 (rows × tokens), the sm_90a `gemmh2` 128 × 64 where `gemmh_plain`
  picks it, the register-staged types (Q6_K and its int8 copy, Q8_0, …) 64 × 128 when there are ≥ 1.5 blocks per SM,
  else 64 × 64, and the fused gate + up as 32 rows of each × 64 (gemmh2: 64). All of these kernels run 1 block per
  SM. A partial last wave costs a whole wave.
- **`CudaQwen2::wave_gemms`**: each layer's products as `layer` / `ffn_half` launch them, including `ffn_half`'s
  `fuse` rule. Returns None without the int8 GEMMs (sm < 80) or for MoE layers; both keep the old split.
- The SM count is `BatchFns::sms`, from `cuDeviceGetAttribute(SM count)`. Nothing is hardcoded.
- `AXON_WAVE_SPLIT=0` gives the old equal split (`batch::passes`), with no rebuild. `AXON_PASS_TRACE=1` prints each
  prompt's pass sizes to stderr. `verify` / `verify_cands`, pipelines and slots keep `passes`.

## Why
r15 §4 pick 1: on 128 SMs, 2100 = 1050 + 1050 makes 17 token tiles per pass. q / wo / down then run 952 blocks
(7.44 waves), k / v 136 (1.06 waves), and gate + up 10,064 (78.6 waves). With 1088 + 1012, pass A is 17 full tiles
and pass B is 16 tiles, which gives exactly 7, 1 and 74 waves. The model reproduces r15's 4090 numbers exactly:
7B 282.6 → 270.9 and Qwen3-8B 207.0 → 195.5 block-waves (64 × 64 × dim units, per layer, summed over the passes).

## The wave model per GPU (ESTIMATED, `wave_passes_shapes` output)
GEMM block-waves per layer at batch max 1152, equal split → wave split. llama.cpp's Q4_K_M layout is used, with v
and down in Q6_K in half the layers (Q6_K int8 copy tiles). H100 rows use the gemmh2 routing.

| GPU (SMs) | 7B pp2100 | 7B pp1153 | 7B pp1777 | 7B pp3000 | 8B pp2100 | 8B pp1153 | 8B pp1777 | 8B pp3000 |
|---|---|---|---|---|---|---|---|---|
| bella 5070 Ti L (46) | 1088+1012, −3.1 % | 64+1089, −2.0 % | = | 1152+1152+696, −1.8 % | 1088+1012, −2.8 % | = | = | 1088+1024+888, −2.5 % |
| RTX 4090 (128) | 1088+1012, −4.1 % | 704+449, −3.8 % | 1024+753, −4.1 % | −1.1 % | 1088+1012, −5.6 % | −4.2 % | −1.8 % | −1.1 % |
| RTX 5090 (170) | 1152+948, −4.1 % | −2.1 % | −2.9 % | −5.8 % | 1152+948, −4.1 % | = | −4.2 % | −6.0 % |
| H100 PCIe (114) | 1152+948, −6.8 % | −7.6 % | −4.7 % | −1.4 % | 1088+1012, −3.0 % | −1.5 % | = | −3.3 % |
| H100 SXM (132) | 1088+1012, −2.9 % | 1088+65, −6.2 % | = | −3.7 % | 1088+1012, −6.6 % | −11.1 % | −5.8 % | −1.4 % |

"=" means the equal split stays. bella's absolute block-waves at pp2100: 7B 762.3 → 738.9, 8B 556.0 → 540.5. GEMM
is ~85-90 % of the pass (r15 §2), so the e2e prediction is roughly 0.85-0.9× the GEMM figure: on bella, 7B ~+2.7 %
and 8B ~+2.4 % at pp2100. Zero-filled partial tiles are cheaper than full ones (their loads are skipped), so expect
less. r15's 5090 figure (−1.9 %) is for 1088 + 1012. The search here finds 1152 + 948 for −4.1 %.

## Bits
Pass boundaries don't change bits (determinism.md: "with the prompt prefilled in chunks: yes"). Default-on means
every existing batched-prefill bit test now runs the wave split wherever it applies. They compare against
per-token decode (`prefill_bits` from 1200 tokens, `gpu_prefill_logits_match_forward_big`).
`gpu_prefill_wave_split_bits` (new) checks wave split == equal split, KV bytes and logit bits, at
1153/1300/1777/2100/2304/2900 tokens on 0.5B and 1.5B (7B and Qwen3-8B with `AXON_BIG_MODELS=1`). It also checks
== per-token decode at 1777 on 0.5B, and fails if no length's split differs on the GPU it runs on (so it can't pass
vacuously). The md5 gate adds ~2100- and ~1780-token prompts, so `gen` runs more than one pass.

## Verification and measurement
Gates at 0b2f029 (branch head 711cbce merges main 2b89410): release suite group 1018-163238-7f0e, a844 PASS and
d187 FAIL on `serve::tests::serve_api_key` only, which is the /metrics race fixed by #93 and not this change. d187's
shard was rerun at 711cbce as 1018-193811-9a1a, with `serve::` in debug as 1018-193812-0bea (both pending). The
debug suite (group 1018-163238-f2c9: ee59, 033e) PASS. Big-model bits on bella (1018-163242-49f2) PASS. md5 vs main
(1018-163243-9a15) PASS. That makes 5/6 PASS. The bella A/B is 1018-163243-7684. bella A/B: same binary, `AXON_WAVE_SPLIT=0` vs `1`, interleaved with alternating order,
7 rounds, 3 s cooldown, medians, exact token counts. 7B and Qwen3-8B at pp2100 (waves differ for both), pp3000
(both), pp1153 (7B only) and pp1777 (neither: the control). Kill rule (r15): ≥ 1.5 % at the predicted lengths.

Results: PENDING.
