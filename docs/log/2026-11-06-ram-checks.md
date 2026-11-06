# RAM pre-load checks: FP8 without the repack, the split checked before its upload, refusals name the way out (#183, #184, #185)

- Date: 2026-11-06. Branch: fix/ram-checks. Base: main ad0c65e. Status: open (PR #206).
- Follows #142 (the CPU model's RAM refusal) and #180 (#143, the split's CPU half). Three issues filed from #180's
  review and gates.

## What changed
- **#183, `model::dense_ram_need_keep`: FP8's arena counted with `AXON_REPACK=0`.** The loop returned before the
  arena when the repack was off. `Qwen2::new_layers`' `build_skip` gives every F8_E4M3 matrix an arena slot (its F8S
  rows) whether the repack is on or not. So with `AXON_REPACK=0` the FP8 0.5B's whole ~352 MiB arena was missing from
  #142's CPU check and #180's split check. The estimate now counts a matrix when `repack || F8_E4M3`, which is
  `build_skip`'s own rule.
- **#184, `HybridQwen2::build`: the split's check runs before the GPU half's upload.** #180 ran it after the upload,
  right before the CPU half. The upload page-locks a staging ring and, on a copy of the file, still holds the GPU half's
  bytes, so a cap that the upload itself didn't fit in was an OOM kill rather than a refusal. The check now runs after
  the CUDA context and the module are made, so `avail()` reads what they leave, and before any tensor goes up. Its need
  (`model::GpuHalfHost`) changes in two ways:
  - **`staging`** (`CudaQwen2::upload_staging`): the most the upload holds at once, the larger of two phases that
    never overlap. First the bytes `upload::Ring::new` page-locks, which is `upload::Shape::pick` of the plan's total:
    4 × 4 MiB, or 8 × 8 MiB for 1 GiB or more, none under 64 MiB, or `AXON_UPLOAD_RING`'s size. Then, after the
    ring is freed, the matrices the plan converts to Q8_0 on the host (`plan.conv`: an FP8 checkpoint's split, which
    takes the Q8_0 path), one `rows × Q8_0` buffer at a time, each allocated while its source is still held and
    dropped after its write: the largest of them counts. On the FP8 0.5B that's the head, 151936 × 952 B
    (138 MiB) against a 16 MiB ring; on a 7B FP8, ~552 MiB. (The 0.5B's head is its own `lm_head`: the file carries
    it though the config ties it, and it's made F16 in the loader's anonymous arena, 260 MiB, as a 7B's is.) #206's
    first version counted the ring alone (shady's review, MUST 1): at 23 GPU layers the head's 260 MiB credit then
    covered the one-layer CPU half, the need was the 16 MiB ring, under the tolerance, so no RAM figure refused it
    and the head's conversion was an OOM kill. The staging is added to the need; both phases are over
    before the CPU half is built, so adding it overstates the peak by at most its size. It's counted even when the
    upload then skips the ring (a cold page cache).
  - **`freed`** (`Model::release_gain`): on a copy of the file (or the loader's arena), the GPU half's bytes that
    `release` hands back as each tensor is uploaded. These are the whole 2 MiB blocks of each run of the GPU half's
    tensors, joined across bytes no tensor owns, exactly as `release` drops them. On a file mapping it's 0, because
    those pages are page cache. A safetensors checkpoint's converted tensors (AUX: a BF16 head made F16) are
    anonymous, so they're credited too, on a default load (the 7B FP8's untied head: ~1039 MiB). It's credited
    against the CPU half's need, not the staging, because the ring is pinned, and each conversion is allocated,
    while those bytes are still held.
  The refusal names both: `(its repacked weights … GiB, the KV cache of … tokens … GiB, the GPU half's upload staging
  0.02 GiB, less the GPU half's … GiB of the copy, handed back as it goes up)`.
- **#185, `model::ram_remedy`: the refusal names its ways out**, after `(bound by …)`:
  - When the weights fit without the cache: the longest `--ctx`, in steps of 256, whose KV floor fits beside them,
    computed as the verdict is (with the staging and the credit), so always under the `--ctx` refused, and none when
    the staging alone is over the room.
  - The CPU model: `to fit: a GPU (--device cuda runs the model there, or with --gpu-layers N part of it)`, as far as
    the model can take it (`model::GpuWays`, after shady's review): a BF16/F16 safetensors checkpoint runs on the CPU
    only (`safetensors::gpu_refusal`), so its refusal names the `--ctx`, else `more free RAM (the model runs on the CPU
    only)`; gemma3 doesn't split (`hybrid::split_refusal`, now shared with `build` and `plan`), so its refusal names
    `--device cuda` alone.
  - A split chosen with `--gpu-layers`: `more --gpu-layers than G (fewer layers on the CPU), if the GPU has room`.
  - `--gpu-layers 0`: `--gpu-layers N (N layers on the GPU)`.
  - An automatic split (`hybrid::auto_gpu_layers`): `The split was picked to fit the GPU's memory (G of N layers
    there), so RAM is now the limit; to fit: a smaller --ctx, which also leaves the GPU room for more layers`.

  The staging's plan, which now runs before the check, keeps the split's `{g}/{n} layers on the GPU: …, or fewer
  --gpu-layers` wrapper on its own refusals (the FP8 module's, `gpu_type_ok`'s).
  `check_ram_split` takes `auto` (`CpuHalf { k, n, auto, gpu }`), and `HybridQwen2::build`'s `margin` parameter became
  `auto` (the margin is derived from it, as before).
- README (CPU memory limits), CHANGELOG.

## Why A and not B (#184)
The lead's call, after a checkpoint. The CUDA context and NVRTC are made at CLI start, on a background thread
(`cuda::preinit`), long before the model is read. A floor before that (option B) needs the context's host memory
measured per driver. It covers only caps below it, about 0.2 GiB, which is inferred from #180's caps and has no
output to confirm it. Option A covers what the split itself commits before the check (the ring, a copy's GPU half),
with no new measurement. B stays a possible follow-up in #184.

## How it was verified
Each new test of a fix fails before it. The expect-fail tree is the candidate with the fixes reverted and the tests
kept: #183's rule back to main's early return, `ram_remedy` returning nothing, and the split's check moved back after
the upload with no GPU-half terms (main's order). It keeps the new `release_gain` helper, so that helper's own test
passes there.
- `model::ram_dense_tests::fp8_arena_counted_without_repack` (bella, FP8 0.5B): the estimate's arena is the built
  one (`Qwen2::arena_len`, through the r8 or F8S rows) for k = 1, n/2, n, with the repack on and off.
  `safetensors::tests::tiny_fp8_ram_arena_without_repack` does the same on the tiny checkpoint, so CI runs it.
- `model::ram_dense_tests::dense_need_and_refusal_wording`, `long_ctx_warns_not_refuses`,
  `split_refusal_wording_and_cap`: #185's exact strings and figures. The `--ctx` figure is the largest fitting step
  (`fits(4608) && !fits(4864)`). #184's math: 2400 + 512 − 512 + 64 MiB refused a byte under the edge and through
  at it. A credit larger than the CPU half doesn't pay for the ring.
- `hybrid::tests::hybrid_ram_refused_before_the_upload` (GPU, the 0.5B at 12/24): refused with no RAM. When the RAM
  is read, 0 tensors have been released (each goes back right after its upload), and none after the refusal. At main's
  order 146 are: the 12 GPU layers and the head. The message names the 16 MiB ring. Then a copy of Qwen3-0.6B at one
  GPU layer and ctx 2048: the credit is positive and at most the GPU half's bytes, and the half is refused a byte under
  the edge and built at it. On bella the credit was 12 MiB (of the GPU half's 15 MiB), less than the CPU half's 219 MiB
  (mostly its KV floor), so the edge moves with both terms. A mutant with no credit (`freed: 0`) is refused at the
  edge (bella 1106-093340-399d); without the ring the message check above fails.
  - The first version did the copy on the 0.5B at 12/24 and ctx 512, and failed on bella (1106-084830-8699, "the
    half past the tolerance"). The test was wrong, not the check. On a copy, a CPU half commits only its growth, one
    matrix in flight and its KV (`dense_load_bytes`): ~6 MiB there. The credit was ~250 MiB: 12 layers plus the
    0.5B's untied 138 MiB head, which goes up with any GPU half. That left the 16 MiB ring, under the job's 34 MiB
    tolerance on bella (32 MiB + 8 CPUs x 256 KiB), so no RAM figure could refuse that half. By the same figures no
    split of a copied 0.5B (f16 KV) can be refused: its largest CPU half (one GPU layer, the KV floor at 8192) is
    ~95 MiB against a ~147 MiB credit, and both are real (the head's copy goes back as it's uploaded). The test now
    checks its premise (ring > 0, credit < the CPU half, need > tolerance) and prints the figures.
- `gguf::tests::release_gain_is_what_release_drops` (Linux, mincore): on an anonymous model, the resident bytes that
  releasing a run of four 1.5 MiB tensors and one apart drops equal `release_gain`, and it's 0 on a file mapping.
- `hybrid::tests::hybrid_ram_check_on_real_splits`: the edge now includes the ring. At main's order the 0.5B at one
  GPU layer is built a byte under the edge (the ring isn't weighed).

After shady's review (round 4, dfbf2a4), each new test fails on a second expect-fail tree: dfbf2a4 with the
staging back to the ring alone, `GpuWays` always both, the plan's errors unwrapped and the `--ctx` figure back to
0a2b680's.
- `hybrid::tests::hybrid_fp8_staging_counts_the_q8_conversions` (GPU, the FP8 0.5B): at 1, 12 and 23 GPU layers the
  staging is at least the head's Q8_0 bytes (151936 × 952 = 138 MiB) and every GPU-half FP8 matrix's. At 23 the
  credit (the head's F16 bytes) is over the CPU half and the need is the staging alone, exactly: the credit's rest is
  never set against it. The split is refused a byte under the edge and built at it, and the built CPU half is the
  estimate. (r4's version asserted a need over the head's Q8_0 bytes, strictly; the need is those bytes exactly, so it
  failed on bella, 1106-105216-a4d5: the test's off-by-one, not the check's.)
- `hybrid::tests::hybrid_split_plan_refusal_keeps_its_wording` (GPU, sm_89 and up): the FP8 module refuses a split
  in the staging's `plan`, and the message keeps `{g}/{n} layers on the GPU: …, or fewer --gpu-layers`.
- `model::ram_dense_tests::remedy_names_only_what_the_model_takes`: a GGUF qwen3's refusal names the GPU whole and in
  part, gemma3's `--device cuda` alone, a CPU-only model's the `--ctx`, else more free RAM.
- `safetensors::tests::tiny_bf16_ram_refusal_names_no_gpu`: the tiny BF16 checkpoint is CPU-only (`GpuWays` none)
  and its refusal names `--ctx 768`, else more free RAM; the tiny FP8 checkpoint's names the GPU.
- `model::ram_dense_tests::split_refusal_wording_and_cap`: + a credit over the CPU half with the staging alone over
  the room: no `--ctx` is named (0a2b680's figure was `--ctx 8960`, past the 8192 refused).

Under a real cap (shady's nit; `fp8cap.sh` through `axon-podjob`, bella 1106-105219-0faf): the FP8 0.5B at
`--gpu-layers 23 --numerics exact` (the Q8_0 path), `gen -n 8`, each run in its own `systemd-run --user --scope -p
MemoryMax=… -p MemorySwapMax=0`, for main ad0c65e, 0a2b680 (#206 before the review: the ring alone) and dfbf2a4
(c5c015e's CLI binary is byte-identical to it: c5c015e changes tests only). The smallest cap that runs, bisected in
8 MiB steps, was 909 MiB for all three. Then all three at every 16 MiB cap from 589 to 941 MiB:

| cap (MiB) | main | 0a2b680 | dfbf2a4 |
|---|---|---|---|
| 589–765 | OOM | OOM | OOM |
| 781–861 | OOM | OOM | **refused** |
| 877–893 | OOM | OOM | OOM |
| 909 | OOM | runs | runs |
| 925–941 | runs | runs | runs |

- The candidate refuses at six caps where #206's first version is OOM-killed, and nowhere that version runs. Every
  run's text is the uncapped run's.
- At 877 and 893 the check passes and the run is OOM-killed: that's the tolerance (34 MiB in that job's 8-CPU slot),
  which lets a load within it through, as #142's check always has. Both caps are within 32 MiB of the smallest that
  runs.
- At 765 and below, all three are killed before the check reads anything. Where the refusals stop (861 refused, 877
  not, against a need of 138 MiB and a 34 MiB tolerance) puts what's charged to the cgroup when the check runs at
  757–773 MiB. Before it come the loader's F16 copies of the embedding and the head (anonymous, 2 × 260 MiB), which
  `safetensors.rs` makes without reading the RAM, and the CUDA context. That's inferred from those figures: the OOM
  runs' stderr wasn't kept. It's outside #183–#185 (option B would cover the context, not the loader).
- main ran at 909 in the bisect and was OOM-killed at 909 in the sweep: its peak varies by a few MiB at the edge.

### Gates
Product code dfbf2a4 (after shady's review); c5c015e fixes the FP8 staging test (test-only).
| gate | result | job |
|---|---|---|
| c5c015e, tier filter (172 tests), release, GPU (bella), `--nocapture` | PASS 169 / 0 / 3 | 1106-113105-5c70 |
| the same, debug assertions + overflow checks | PASS 169 / 0 / 3 | 1106-113105-184a |
| the same filter, release, `--cpu` (bella, GPU hidden) | PASS 169 / 0 / 3 | 1106-113106-4ebe |
| the same, debug | PASS 169 / 0 / 3 | 1106-113106-47e4 |
| the pure tests, release (jade, CPU; `AXON_SKIP_FP8_MODEL_TESTS=1`, so `fp8_arena_counted_without_repack` skips there) | PASS 37 / 0 / 0 (`model::ram_dense_tests safetensors::tests:: gguf::tests::release_gain`) | 1106-113106-b2a2 |
| the five new tests on the second expect-fail tree | FAIL as expected: 5 of 5, each for its own reason | 1106-113107-1a3e |
| a real cap, main vs 0a2b680 vs dfbf2a4 (above) | PASS | 1106-105219-0faf |
| dfbf2a4, tier filter, release, GPU (r4) | FAIL 168 / 1 / 3: the staging test's strict `>` (above) | 1106-105216-a4d5 |
| dfbf2a4, the pure tests (jade, r4) | FAIL 36 / 1 / 0: the FP8 checkpoint isn't on jade and the skip variable wasn't set (the job, not the code) | 1106-105330-f46b |
| r4's debug, `--cpu` and expect-fail jobs: killed or dropped once a4d5's failure was understood | n/a | 1106-105216-703c, 1106-105217-a602, 1106-105217-8f30, 1106-105317-f31c |
| clippy `-D warnings` (harness at c5c015e, binary at dfbf2a4: c5c015e is test-only), rustfmt | clean | local |

Before the review: product code e6704b7; 0a2b680 fixes the copy test (test-only).
| gate | result | job |
|---|---|---|
| the new tests' pure parts, release (jade, CPU) | PASS 40 / 0 / 0 (`gguf:: model::ram_dense_tests safetensors::tests::tiny_fp8`) | 1106-072522-4e45 |
| e6704b7, tier filter, release, GPU (bella) | FAIL 164 / 1 / 3: the copy test's premise (above) | 1106-084830-8699 |
| e6704b7, the same filter, `--cpu`, release / debug | PASS 165 / 0 / 3 each | 1106-074156-e184, 1106-072834-211d |
| 0a2b680, tier filter (168 tests), release, GPU (bella) | PASS 165 / 0 / 3 | 1106-093339-ff24 |
| the same, debug assertions + overflow checks | PASS 165 / 0 / 3 | 1106-093621-9fa8 |
| the same filter, release, `--cpu` (bella, GPU hidden) | PASS 165 / 0 / 3 | 1106-093622-e79d |
| the same, debug | PASS 165 / 0 / 3 | 1106-093623-41f8 |
| the new tests at main's code (expect-fail tree, 15 tests) | FAIL as expected: 7 fail (2 for #183, 2 for #184, 3 for #185); the 8 that pass are older tests and `release_gain`'s, whose helper the tree keeps | 1106-093622-b200 |
| the copy test with no credit (mutant) | FAIL as expected: refused at the edge | 1106-093340-399d |
| clippy `-D warnings` (harness, binary), rustfmt | clean | local |

## What didn't work / notes
- The first run of `fp8_arena_counted_without_repack` (bella `1106-064626-f708`) failed with "no FP8 arena built".
  `Qwen2::r8_arena` finds the shared arena through a repacked matrix, and FP8 layers hold F8S rows instead. The tests
  now read the arena through either (`arena_len`, cfg(test)). `r8_arena`, which the streamer pins, is unchanged.
- Not covered: a cap below what's committed before the check (option B above covers the CUDA context's share; on
  the FP8 0.5B the loader's F16 copies come first too: the real cap above, 765 MiB and below). #180's `splitcap.sh`
  demo (a GGUF split) wasn't rerun; the real cap ran the FP8 split, the case shady's review was about.
- r4's jade job failed on the missing FP8 checkpoint because the job didn't export `AXON_SKIP_FP8_MODEL_TESTS=1`
  (#193 made a missing checkpoint a failure); r5's did.
