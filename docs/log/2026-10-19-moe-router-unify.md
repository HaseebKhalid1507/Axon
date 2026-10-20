# One MoE router kernel + MoeKind enum  (PR #104)

- Date: 2026-10-19. Branch: refactor/moe-router-kind. Base: main 764792f. Status: open; review-fix gates
  (fdd7816) passed.
- Pick / design: issue #99, the follow-up to Shady's review of #98 (three copies of the k-round tree argmax, two
  MoE dispatch styles in `MoeQwen3`).

## What changed
- `src/cuda/moe.cu`: `moe_router` replaces `moe_topk` (qwen3moe), `moe_topk_oai` (gpt-oss) and mla.cu's
  `mla_topk` (deepseek2). It takes a gate (`MOE_GATE_SOFTMAX` / `SIGMOID` / `OAI`), a selection bias (none,
  `exp_probs_b`, the router's logit bias), the renorm and the scale. The k-round tree argmax (ties to the lower
  index) and the no-eligible-score guard (expert 0; `__trap()` under `MOE_DEBUG`, which an unknown gate also hits)
  appear once. `MLA_DEBUG` is merged into `MOE_DEBUG`.
- `cuda::qwen2::moe`: `MoeKind { Qwen3, Oai, Mla }` replaces `moe_route`'s `LayerW::oss` arm and #89's
  `xf`/`xb` side path + `MoeQwen3::new_mla`. `MoeQwen3::run` is the one dispatch for decode and batched passes.
  `launch_router` is shared by `moe_route` and `mla_layer`. `MoeKind::new` errs on an arch without a kind.
  `Kernels::moe_module_for` picks the router module.
- `model::moe`: `Moe { gate_fn, oai }` -> `rule: Router { Gate(Gate), Oai(Box<Oai>) }` (the roadmap debt from #79).
- Tests: the three kernel tests launch through `launch_router` and the `GATE_*` constants.
  `moe::router_legacy_tests::router_bits_match_legacy` (`#[ignore]`, GPU, release builds) compares `moe_router`
  against main's three kernels (`tests/fixtures/moe_router_legacy.cu`) bit for bit.
- No flag, no default changed. The MLA path no longer allocates a one-token `MoeBufs` stub.

## Why
Debt that grows with each MoE family: three copies of the selection loop, each with its own NaN guard (the
qwen3moe one had none: an all-NaN router wrote `p[INT_MAX]` out of shared memory), and two dispatch styles.

## How it was verified
Hosts: bella (RTX 5070 Ti Laptop 12 GB, driver 615.71.09, NVRTC 13.4), avante (RTX 4050 Laptop 6 GB, driver
610.43.02, NVRTC 13.3), jade (GTX 1050 3 GB, driver 580.159.04, NVRTC 13.0).

| gate | result | job ID / log |
|---|---|---|
| fast tier (moe/mla/oss/gptoss/deepseek2), release, 0fac171 | 49 passed, 1 ignored | 1019-222212-6a02 (avante) |
| fast tier, debug assertions, 0fac171 | passed | 1019-222212-81ce (avante) |
| release suite, `--shard 3`, 0fac171 | 162 + 229 + 240 passed, 0 failed | 1019-222224-4620 / 8c37 / bed9 |
| debug-assertion suite, `--shard 3`, 0fac171 | 162 + 229 + 240 passed, 0 failed | 1019-222224-487e / c8c4 / f330 |
| `moe_router` vs main's kernels, 280 configs, 2,809,856 slots, 0fac171 | 0 differ (sm_61, sm_89) | 1019-222728-91cd (jade), 1019-222818-e4d6 (avante) |
| md5 of `gen`, main vs candidate, same host, 0fac171 | 28/28 equal | 1019-222233-1991 (bella) |
| clippy -D warnings (bin + test), rustfmt | clean | (local) |
| after review fixes, fdd7816: fast tier release / debug (bella, the real models) | 49 passed, 2 ignored / same | 1020-002429-907c / 1020-002430-c70d |
| fdd7816: `router_bits_match_legacy` | 0 of 2,809,856 differ | 1020-002430-e156 (avante) |
| fdd7816: Moonlight + gpt-oss model tests with `AXON_TEST_DS2=1 AXON_REPACK=0` | 7 passed, 1 env failure (below) | 1020-002431-db39 (bella) |
| `oss_gpu_logits_graded_model`, main vs fdd7816, default env | both pass, identical numbers | 1020-004332-a0e9 / 1020-004333-4d84 (bella) |
| the same under `AXON_REPACK=0`, main | fails as db39 does (pre-existing) | 1020-004332-e3d1 (bella) |
| fdd7816: md5 of `gen`, main vs candidate, same host (P5 natural ~2k-token prompt added) | 32/32 equal | 1020-002431-81be (bella) |
| fdd7816: md5 again, plus CPU with the default repack for qwen3moe and gpt-oss | 38/38 equal | 1020-004401-2b1e (bella) |

## Results
MEASURED, avante `--timing avante`, Qwen3-MOE-4x0.6B Q4_K_M, `gen -n 256 --device cuda`, 3 interleaved rounds:
decode main 101.50 / 100.87 / 98.98 tok/s (mean 100.45), candidate 102.30 / 101.61 / 96.71 (mean 100.21):
-0.2 %, inside the ±2 % spread between rounds. No change. pp wasn't timed: the 17-token prompt is too short to
mean anything, and the router is a per-token kernel that's the same in prefill. gpt-oss and Moonlight weren't
timed: they don't fit avante's 6 GB card and 15 GB of RAM, and bella is a shared desktop.

`oss_gpu_logits_graded_model` fails under `AXON_REPACK=0` on main too. `Qwen2::new` reads AXON_REPACK, so the
test's CPU reference and its in-place "rows" model become the same model, the reference spread is 0, and the
mean-KL check (`<= 2 x` the reference) can't pass. With the default env, main and #104 print the same numbers
(KL 2.63e-3 / 2.43e-3, 195/198, the same flip margins). That fix goes in a separate PR off main.

## What didn't work
Nothing was dropped. Kind and scratch are still two enums that have to agree by convention (`run` has an
`unreachable!` arm for a mismatch). Putting the scratch inside the kind would make a mismatch unrepresentable,
but it's left as a follow-up.
