# Graded logit tests: the reference is built repacked explicitly  (PR #105)

- Date: 2026-10-20. Branch: fix/graded-ref-repack. Base: main dc80363. Status: open. `gemma3_matches_reference_forward` under AXON_REPACK=0 is #106 (separate)
- Pick / design: found during #104's review gates. `oss_gpu_logits_graded_model` failed on main and on #104 alike
  when it was run with `AXON_REPACK=0`.

## What changed
Tests only. The graded logit tests grade a backend (GPU, hybrid, stream, the MoE split, chat reuse) against
the CPU. Every bound is relative to `ac`, the CPU's per-row kernels (`rows`, `Qwen2::new_with(m, false)`)
against its default repacked kernels (`cpu`). `cpu` was built with `Qwen2::new(m)`, which reads AXON_REPACK.
- All 17 such references are now `Qwen2::new_with(m, true)`:
  - `cuda/qwen2/tests.rs` ×12, including `gemma3_vs_ref` and `gpu_gemma3_logits_match_cpu`, which the
    first `AXON_REPACK=0` run caught;
  - `hybrid/tests.rs` ×2;
  - `moe/tests.rs`, `moe/oss_tests.rs` and `chat.rs` ×1 each.
- `assert_graded_floor` (so `assert_graded` too) and `assert_graded_gemma` first call `assert_ref_spread`. A
  reference with no spread (positions > 0, max |dlogit| 0) fails with "reference spread is 0: cpu and rows are
  the same model (AXON_REPACK=0? ...)", not on the first relative bound it breaks.
- New test: `ref_spread_zero_fails_by_name`.
- No non-test code changed.

## Why
With `AXON_REPACK=0`, `cpu` and `rows` were the same model, so `ac`'s spread was exactly 0 and
`ag.sum_kl <= 2 x ac.sum_kl` failed as an opaque "mean KL". It wasn't nondeterminism. MEASURED on bella, gpt-oss-20b,
main 764792f:

| env | rows-vs-cpu KL | gpu-vs-cpu KL | result | job |
|---|---|---|---|---|
| `AXON_REPACK=0` | 0.00 (198/198) | 1.20e-3 (197/198) | FAIL | 1020-004332-e3d1 |
| default | 2.43e-3 (195/198) | 2.63e-3 (195/198) | PASS | 1020-004332-a0e9 |

The two KLs move together because the reference itself changed: with the env set it was the in-place model.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| affected tests, default env, bella (97a99af) | 37 passed, 5 ignored | 1020-010138-c875 |
| affected tests, `AXON_REPACK=0`, bella (97a99af) | 2 failed: an unconverted gemma3 ref (caught by the new check, fixed in f8e6b10) and #106 | 1020-010139-4ab6 |
| affected tests, `AXON_REPACK=0`, f8e6b10, #106's test skipped | 36 passed, 5 ignored | 1020-014803-a36f |
| gemma3 graded tests, default env, f8e6b10 | 3 passed | 1020-014803-4317 |
| fast tier (`scripts/test-tier.sh fast`, `TIER_SHARD=3`): release | 202 + 225 + 186 passed | 1020-010139-6bb9 / c8d0 / 5ec4 |
| fast tier: debug assertions (CPU queue) | 162 + 229 + 240 passed | 1020-010139-67fa / ccf9 / bc9b |
| `ref_spread_zero_fails_by_name` at af34f3e | passed (release; debug 1020-011056-86b3) | 1020-010631-de37 |
| md5 of `gen` | n/a: no non-test code changed | |
| clippy -D warnings (bin + test), rustfmt | clean | (local) |

The affected set is `logits_match_cpu continuations_match_cpu cuda_kv_q8_accuracy gemma3_ graded
gpu_yarn_matches_reference`, 42 tests. That covers every caller of the changed references, plus a few
gemma3/serve tests caught by the filter.

## Results
No numbers move under the default env: `Qwen2::new` already gave `new_with(m, true)` there. Under
`AXON_REPACK=0`, the graded tests now grade against the repacked CPU, as they do by default.

## What didn't work
An absolute KL floor (`max(2 x ac, bound)`) was considered and dropped. Justifying a bound from data needs a
spread per model under both kernel sets. An explicit reference makes the spread real in every environment, and
the named check catches a degenerate reference, whatever the cause.
