# gemma3 in-place CPU window "mismatch" is a reference near-tie  (PR #107)

- Date: 2026-10-20. Branch: fix/gemma3-inplace-window. Base: main 17b8f6e. Status: open
- Pick / design: issue #106 (`gemma3_matches_reference_forward` fails with `AXON_REPACK=0`, window 6).

## What changed
`src/model/gemma3_tests.rs` only (no engine code):
- `gemma3_matches_reference_forward` builds both CPU models itself, `Qwen2::new_with(m, false)` (in place,
  per-row kernels) and `Qwen2::new_with(m, true)` (r8/r8q repack), each from its own load (repacking
  releases the file's tensors), whatever `AXON_REPACK` says. Before, it built one model via `Qwen2::new`
  and the in-place build was only covered when someone set the env.
- The top-token assert is margin-aware: the model's top token is the reference's, or the two are within
  `TOP_TIE` = 0.5 logits of each other in the reference and in the model. The |dlogit| < 1.0 bound, the
  "the window changes the reference by > 3" check and the prefill == decode bit check are unchanged.

## Why
Diagnosis on bella, scratch probe job 1020-014608-3a07 (per-layer dumps of the residual stream and the
attention output at the last position, decode, Gemma 3 1B Q4_K_M, prompt of #106, window 6):

- Reference at the failing position: 7001 at 17.984, 506 at 17.637, a **0.348** logit margin. In place
  scores them 17.894 / 17.911 (picks 506, by 0.017). Repacked scores them 17.829 / 17.650 (picks 7001).
  max |dlogit| vs the f64 reference: in place 0.885, repacked 0.567. Both are under the test's own 1.0
  bound, which is about three times the margin. The tie is inside the noise the test already allows.
- In place vs repacked, relative L2 per layer: 8e-8 at layer 0 (the windowed attention included); then
  attention output 3e-6 at layer 1 and 6e-4 at layer 2 (residual 6e-5, 1.5e-3); 1e-3 to 2e-2 through
  layer 25. It grows smoothly with no step.
  The same growth shows with no window (layer 3: 5e-4, layer 5: 2.7e-3) and with no jump at the global
  layers (5, 11, 17, 23). At every layer both builds sit about equally far from the reference
  (5e-3 to 3e-2). The two builds' activation rounding differs (per-row vs r8 kernels, `identity::cpu`).
- Where it starts (second probe, job 1020-020313-3178: every position and layer, bit for bit, with int8
  activation-quant and f16 KV flips counted). The two builds first differ at **position 0, layer 0, in the
  q/k/v matvec**: identical inputs, outputs about 1 ulp apart (max rel 2e-7) on 75-85 % of elements. That is
  the per-row vs r8 accumulation order, which determinism.md already lists as changing the low bits. Rounding
  cliffs then amplify it: 1 f16 K and 2 V flips at layer 0, an int8 activation flip at layers 2 and 3, k/v
  5e-5 apart at layer 3, and the first > 1e-4 at position 0, layer 3 (the FFN input, 58 int8 flips). All of
  this happens at position 0, which a 6-token window cannot mask, and the trace is identical for window None
  and window 6. The window only decides which of the already-diverged KV rows the last position reads.
- Window bound: the reference at window 5 / 7 is 3.50 / 7.56 logits (max) from in place and 3.61 / 7.92
  from repacked, against 0.88 / 0.57 at window 6. Both builds mask exactly at window 6.

So there is no masking, window-bound, KV-index or local/global bug in the in-place path. The exact
top-token assert was stricter than the test's own numeric bound.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| gemma3 tests (`gemma3` filter, CPU + GPU), `AXON_REPACK=0`, skipping `gpu_gemma3_logits_match_cpu` | pending | 1020-020400-354a |
| (first run without the skip: 10/11, only `gpu_gemma3_logits_match_cpu` failed, "mean KL"; main 17b8f6e fails it the same way, see below) | 10 passed, 1 failed | 1020-015203-7661; main: 1020-015644-d707 |
| gemma3 tests, default env | 11 passed, 0 failed, 4 ignored (avante) | 1020-015204-56a2 |
| fast tier, release | pending | 1020-015209-5584 |
| fast tier, debug assertions (CPU queue) | pending | 1020-015209-85ae |
| md5 of `gen`, main 17b8f6e vs candidate, same host | pending | 1020-015205-dedd |
| clippy -D warnings (bin + test), rustfmt | clean | local, rustc 1.95.0 |

## Results
No engine change, so no bits change: the md5 A/B checks the default (r8) CPU path, the in-place CPU
path and CUDA (f16 and q8 KV) on gemma3 1B (Q4_K_M, Q8_0) and 4B Q4_K_M. No perf numbers, since no
engine code changed.

`gpu_gemma3_logits_match_cpu` under `AXON_REPACK=0` is not #106. Its graded CPU reference ("rows") is built
by `Qwen2::new`, so under that env it is the same in-place model as the one under test, the spread is 0
("cpu rows vs cpu: KL 0") and the mean-KL check trips. Main fails it identically (job 1020-015644-d707).
PR #105 (fix/graded-ref-repack) builds those references repacked explicitly.

## What didn't work
n/a. The hypotheses checked and ruled out (window bound off by one, a mask or ring-index difference in
the in-place attention, local/global layer selection) are the bullets above.
