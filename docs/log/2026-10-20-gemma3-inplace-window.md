# gemma3 in-place CPU window "mismatch" is a reference near-tie  (PR #PRNUM)

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
- In place vs repacked, relative L2 per layer: 8e-8 at layer 0 (the windowed attention included), attention output then
  3e-6 at layer 1 and 6e-4 at layer 2 (residual 6e-5, 1.5e-3), then 1e-3 to 2e-2 through layer 25. It grows smoothly with no step.
  The same growth shows with no window (layer 3: 5e-4, layer 5: 2.7e-3) and with no jump at the global
  layers (5, 11, 17, 23). At every layer both builds sit about equally far from the reference
  (5e-3 to 3e-2). The two builds' activation rounding differs (per-row vs r8 kernels, `identity::cpu`).
- Window bound: the reference at window 5 / 7 is 3.50 / 7.56 logits (max) from in place and 3.61 / 7.92
  from repacked, against 0.88 / 0.57 at window 6. Both builds mask exactly at window 6.

So there is no masking, window-bound, KV-index or local/global bug in the in-place path. The exact
top-token assert was stricter than the test's own numeric bound.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| gemma3 tests (`gemma3` filter, CPU + GPU), `AXON_REPACK=0` | pending | 1020-015203-7661 |
| gemma3 tests, default env | pending | 1020-015204-56a2 |
| fast tier, release | pending | 1020-015209-5584 |
| fast tier, debug assertions (CPU queue) | pending | 1020-015209-85ae |
| md5 of `gen`, main 17b8f6e vs candidate, same host | pending | 1020-015205-dedd |
| clippy -D warnings (bin + test), rustfmt | clean | local, rustc 1.95.0 |

## Results
No engine change, so no bits change: the md5 A/B checks the default (r8) CPU path, the in-place CPU
path and CUDA (f16 and q8 KV) on gemma3 1B (Q4_K_M, Q8_0) and 4B Q4_K_M. No perf numbers, since no
engine code changed.

## What didn't work
n/a. The hypotheses checked and ruled out (window bound off by one, a mask or ring-index difference in
the in-place attention, local/global layer selection) are the bullets above.
