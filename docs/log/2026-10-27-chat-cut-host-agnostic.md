# tests: planner-vs-old-cut checks that don't depend on the GPU's SM count  (PR #163)

- Date: 2026-10-27. Branch: fix/chat-cut-host-agnostic. Base: main 0c23a87. Status: open
- Pick / design: foreman (Jawz), after main 0c23a87's release gate failed on the RTX 4090 pod (job 1027-021156-1252).

## What changed
Test code only (`src/chat.rs` tests, `src/cuda/qwen2/tests.rs`, `src/serve/tests.rs`, `src/testutil.rs`).
- `chat::tests::long_turn_matches_fresh_prefill_cuda`: the net is wrapped in `Cuts`, an `Lm` that forwards every
  method to the GPU model and logs the positions each `feed_batch` / `prefill_logits` call fed. The test now asserts
  that the ranges `Chat::reply` really fed equal `prompt_chunks(n, start, cache)` (the cache's planner) on both turns.
  Before, it only called `prompt_chunks` itself after the turn, so it never saw what chat fed. The long turn is
  ~1230 tokens instead of ~1530, a length (asserted) where no planner choice can be the old cut on any GPU, and there
  the planner-vs-old check is a hard assert.
- `cuda::qwen2::tests::gpu_prompt_chunks_bits`: at 2099 and 1799 tokens, a NOTE instead of `assert_ne!` when the
  planner's cut equals the old one on this GPU. A third prefix (from 899: 1200 / 1201 tokens) is in the
  always-differs range and asserts it for both serve's way and chat's way, and runs the same bit checks
  (== per-token decode).
- `serve::tests::serve_cuda_long_prompt_matches_gen`: the prompt is ~1200 tokens (was ~2000), asserted in the
  always-differs range, so the real `Engine::job` path keeps a hard not-the-old-cut assert (on the reference net's
  plan) on every GPU. (Review fix, e55daec: d5f85cc had made it a NOTE.)
- `testutil::planner_cut_differs(plan, old, sms, what)`: returns whether the cuts differ, and prints
  `NOTE: <what>: planner cut == prefill_batch cut on this GPU (N SMs), [...]; the planner-vs-old check is vacuous here`
  when they don't.

## Why
`long_turn_matches_fresh_prefill_cuda` failed on the 4090 (128 SMs) at `assert_ne!(rs, old)`: turn 0's 1532 tokens
were cut `[0..1152, 1152..1532]` by `prompt_passes`, the same as `chunk_ranges(1532, 1152)`. Every check before that
passed (reply == fresh prefill). The planner is host-dependent by design: `wave_split_passes` → `wave_passes` takes
the split with the least `gemm_cost`, which counts GEMM waves on the GPU's SMs. When the remainder after a full
1152-token pass is at least `wave_floor` (max / 4 = 288), "1152 + rest" is one of its candidates, and on the 4090 it
won for 1532. On bella (sm_120) it didn't, so the test passed there. Not a correctness bug: pass boundaries don't
change bits (determinism.md), and the reply matched.

Why a length in max + 1 ..= max + wave_floor − 1 (1153..=1439 at the batch max 1152, 1025..=1279 at 1024) can't give
the old cut on any GPU (CODE: src/cuda/qwen2/batch.rs `prompt_passes`, `wave_passes`, `fa_passes`, `passes`):
- the old cut is `max + (n − max)`, two passes with the last under `wave_floor`;
- `passes` (AXON_WAVE_SPLIT=0, or no split worth it) gives two equal passes of ~n/2 < max;
- `wave_passes` (only at start + n < FA_LONG) makes two passes and only takes splits whose last pass is ≥
  `wave_floor` (`if last < floor || last > max { continue }`), so the first pass is ≤ n − 288 < max;
- `fa_passes` below FA_LONG (8192) is `passes`. From FA_LONG on (a turn after a long history) it returns
  `passes_n(n, np)` for np in np0 ..= np0 + max(np0 / 8, 1), here 2 or 3 *equal* passes (3 when the 128-token
  tiles don't grow, n > 1280). Equal passes are never max + a short rest.
No SM count enters any of these bounds. The one way to get the old cut is a cache with no batch scratch
(`prompt_chunks` falls back to `chunk_ranges`), which these tests' caches always have.

## How it was verified
Gates of the three touched tests (filters `chat::tests::long_turn_matches_fresh_prefill_cuda
cuda::qwen2::tests::gpu_prompt_chunks_bits serve::tests::serve_cuda_long_prompt_matches_gen`, `--nocapture
--test-threads 1`, AXON_PASS_TRACE=1). The CPU suites weren't run locally: the diff touches only these GPU tests and
`testutil`, and CI runs the CPU suites (release, debug, scalar). Final gates are at e55daec (in the PR body, §6);
the first round, at d5f85cc:

| gate | host | result | job |
|---|---|---|---|
| release | RTX 4090 pod (sm_89, 128 SMs) | 3 / 0 / 0 | 1027-032020-7777 |
| release | avante RTX 4050 Laptop (sm_89, 20 SMs) | 3 / 0 / 0 | 1027-032210-df4f |
| debug assertions | avante | 3 / 0 / 0 | 1027-032211-3263 |
| clippy -D warnings (tests), rustfmt | bella | clean | |
| md5 of `gen` | n/a | test-only diff: no non-test code changed | |

## Results
The cuts the planner gave (AXON_PASS_TRACE, MEASURED, job 1027-032020-7777, RTX 4090, 128 SMs):
- chat turn 0: 1228 tokens fed `[0..614, 614..1228]` (equal passes), the same on avante (20 SMs); chat's recorded
  cuts == the planner's on both turns.
- `gpu_prompt_chunks_bits` (serve's way): 2099 → 1088 + 1011, 1799 → 960 + 839, 1200 (from 899) → 640 + 560 on the
  4090; 600 + 600 for 1200 on avante. All differ from the old cut, so no NOTE fired on either.
- For comparison, main's test on the same 4090 (job 1027-021156-1252): 1532 → 1152 + 380, the old cut.
- serve at d5f85cc (~2000 tokens): 2004 → 1002 + 1002 on both. At e55daec it is ~1200 tokens: see the PR body.

## What didn't work
- Keeping ~1530 tokens and making the planner-vs-old check a NOTE alone: on the 4090 the test would then prove
  nothing about "chat uses the planner" at that length. A length where the cuts must differ keeps the proof on every
  host. The NOTE stays only for the bits test's 2099 / 1799 cases, whose lengths are tied to its other checks.
- A NOTE for serve's ~2000-token prompt (d5f85cc): review found it left `Engine::job`'s path without a hard
  not-the-old-cut check on hosts where the cuts coincide; moved into the range instead (e55daec).
