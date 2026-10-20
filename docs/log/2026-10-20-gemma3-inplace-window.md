# gemma3 in-place CPU window "mismatch" is a reference near-tie  (PR #107)

- Date: 2026-10-20. Branch: fix/gemma3-inplace-window. Base: main 17b8f6e (main cb53181 merged in). Status: open
- Pick / design: issue #106 (`gemma3_matches_reference_forward` fails with `AXON_REPACK=0`, window 6).

## What changed
`src/model/gemma3_tests.rs` only (no engine code):
- `gemma3_matches_reference_forward` builds both CPU models itself, `Qwen2::new_with(m, false)` (in place,
  per-row kernels) and `Qwen2::new_with(m, true)` (r8/r8q repack), each from its own load (repacking
  releases the file's tensors), whatever `AXON_REPACK` says. Before, it built one model via `Qwen2::new`
  and the in-place build was only covered when someone set the env.
- The top-token assert is margin-aware: the model's top token is the reference's, or the two are within
  `TOP_TIE` = 0.5 logits of each other in the reference and in the model. The comment derives the band
  from the measured misses and says what it costs: where the reference gap is under 0.5 (the last
  position, in both window configs) the check accepts either of the top two.
- Review round (#107, Shady): so that the tie band can't hide a window bug, at window 6 each build must be
  `NEIGHBOR_MIN` = 2.0 logits from the **window-5 and window-7 references** wherever those differ from window
  6's. Measured: 2.92-5.63 and 3.88-7.92, against at most 0.885 from window 6's (probe 4, job 1020-031341-ba0c).
  The checked positions go from [3, 10] to [3, 5, 6, 7, 8, 9, 10], so the window is exercised at 6 positions
  instead of 1. The reference's top-2 gap is printed at every position. Mutation check: with the window bound
  made off by one low (effectively window 5) **and** the |dlogit| bound disabled, the neighbour check alone
  fails the test ("pos 5: only 0.5629 from the window 5 reference", job 1020-031948-9e15).
- Unchanged: the |dlogit| < 1.0 bound, the "the window changes the reference by > 3" check and the
  prefill == decode bit check.

## Why
Diagnosis on bella, scratch probe job 1020-014608-3a07 (per-layer dumps of the residual stream and the
attention output at the last position, decode, Gemma 3 1B Q4_K_M, prompt of #106, window 6):

- Reference at the failing position: 7001 at 17.984, 506 at 17.637, a **0.348** logit margin. In place
  scores them 17.894 / 17.911 (picks 506, by 0.017). Repacked scores them 17.829 / 17.650 (picks 7001).
  max |dlogit| vs the f64 reference: in place 0.885, repacked 0.567. Both are under the test's own 1.0
  bound, which is about three times the margin. The tie is inside the noise the test already allows.
- In place vs repacked, relative L2 per layer: 8e-8 at layer 0 (the windowed attention included); then
  attention output 3e-6 at layer 1 and 6e-4 at layer 2 (residual 6e-5, 1.5e-3); 1e-3 to 2e-2 through
  layer 25. It grows smoothly with no step at the global layers (5, 11, 17, 23). Without the window the
  divergence at this position starts later and smaller: residual 2.1e-7 at layer 1 (6.4e-5 with window 6)
  and attention output 1.4e-6 at layer 2 (5.8e-4 with window 6). So the window makes the computation more
  sensitive to rounding, plausibly because dropping the early keys sharpens the softmax (not verified). It is
  still not a build bug: at layers 0-2 the two builds are equally far from the reference (rel(ip,ref) =
  rel(rp,ref) to 3 digits), and at every layer they are about equally far (5e-3 to 3e-2). The two builds'
  activation rounding differs (per-row vs r8 kernels, `identity::cpu`).
- Where it starts (second probe, job 1020-020313-3178: every position and layer, bit for bit, with int8
  activation-quant and f16 KV flips counted). The two builds first differ at **position 0, layer 0, in the
  q/k/v matvec**: identical inputs, outputs about 1 ulp apart (max rel 2e-7) on 75-85 % of elements. That is
  the per-row vs r8 accumulation order, which determinism.md already lists as changing the low bits. Rounding
  cliffs then amplify it: 1 f16 K and 2 V flips at layer 0, an int8 activation flip at layers 2 and 3, k/v
  5e-5 apart at layer 3, and the first > 1e-4 at position 0, layer 3 (the FFN input, 58 int8 flips). This
  shows the kernels' rounding is enough to start the divergence before any position a window could mask.
  It says nothing about the window itself: the probe prints at most 70 lines per window, all at position 0.
- Why in place is at 0.88 and repacked at 0.57 here (third probe, job 1020-021440-ff16: 4 prompts, every
  position, both windows, 70 points per build per window, max |dlogit| against the reference):

  | | in place median / max | repacked median / max | in place > repacked | top-1 misses ip / rp |
  |---|---|---|---|---|
  | window None | 0.597 / 2.17 | 0.572 / 3.22 | 37/70 | 1 / 1 |
  | window 6 | 0.852 / 5.04 | 0.831 / 4.31 | 39/70 | 6 / 3 |

  Points over 1.0: 40 in place, 42 repacked. The builds are indistinguishable in distribution. The in-place
  window-6 median is 0.85, so 0.88 is close to typical, and repacked's 0.57 at this position is below its own
  median (0.83). A 6-token window on a model trained with 512 raises both builds' drift from the f64
  reference (median 0.60 to 0.85). On a code prompt both builds go past the test's 1.0 bound, even with no
  window (up to 2.17 in place, 3.22 repacked), and both miss top-1 there at a 0.36 gap. The bound holds for
  the test's own prompt only, for both builds. That's filed as **#110**, not widened here. The probe 3 source
  was recreated and re-run, and its output is identical to the original's (job 1020-031340-bcbd).
- Window bound: at the last position the reference at window 5 / 7 is 3.50 / 7.56 logits (max) from in
  place and 3.61 / 7.92 from repacked, against 0.88 / 0.57 at window 6. Probe 4 extends this to every
  position: 2.92-5.63 from window 5 (positions 5-10) and 3.88-7.92 from window 7 (positions 6-10). Both
  builds mask exactly at window 6, which is now a test assertion (`NEIGHBOR_MIN`).

So there is nothing to fix in the window path (mask, window bound, KV index, local/global selection). The
exact top-token assert was stricter than the test's own numeric bound. Probe sources and logs:
https://gist.github.com/HaseebKhalid1507/54afc02fac2fca96e1e301f701a938f1

## How it was verified
Final gates on 976e44f (review round: the neighbour check). md5 base: main cb53181 (unchanged since).

| gate | result | job ID / log |
|---|---|---|
| gemma3 tests (`gemma3` filter, CPU + GPU), `AXON_REPACK=0` | 11 passed, 0 failed, 4 ignored (avante) | 1020-032930-c24d |
| gemma3 tests, default env | 11 passed, 0 failed, 4 ignored (avante) | 1020-032931-631c |
| fast tier, release | 614 passed, 0 failed, 47 ignored, 18 filtered (bella) | 1020-032935-3a34 |
| fast tier, debug assertions (CPU queue) | pending | 1020-032935-905f |
| md5 of `gen`, main cb53181 vs candidate, same host | pending | 1020-032934-c46b |
| clippy -D warnings (bin + test), rustfmt | clean | local, rustc 1.95.0 |

The `AXON_REPACK=0` run is redundant for `gemma3_matches_reference_forward`, which builds both CPU models
itself. It's kept for the other gemma3 tests, the GPU-vs-CPU graded one among them, which use `Qwen2::new`
and so follow the env.

Earlier rounds. a2d2bee (main cb53181 merged in): gemma3 `AXON_REPACK=0` 11 passed (1020-022535-ba49),
default 11 passed (1020-022535-4ac4), fast release 614 passed (1020-022536-6678), fast debug 632 passed
(1020-022536-1fbd), md5 vs cb53181 30/30 (1020-022536-0e4f). 2dfeadd (md5 base 17b8f6e): gemma3
`AXON_REPACK=0` 10 passed with `gpu_gemma3_logits_match_cpu` skipped (1020-020400-354a), default 11 passed
(1020-015204-56a2), fast debug passed (1020-015209-85ae), md5 30/30 (1020-021243-187c).

`gpu_gemma3_logits_match_cpu` under `AXON_REPACK=0` is not #106. Its graded CPU reference ("rows") is built
by `Qwen2::new`, so under that env it is the same in-place model as the one under test, the spread is 0
("cpu rows vs cpu: KL 0") and the mean-KL check trips. Main 17b8f6e fails it identically (job 1020-015644-d707).
PR #105 (merged at cb53181, now in this branch) builds those references repacked explicitly.

## What didn't work
n/a. The hypotheses checked and ruled out (window bound off by one, a mask or ring-index difference in
the in-place attention, local/global layer selection) are the bullets above. The CPU path's general drift
from the f64 reference on code prompts is a separate question: #110.
