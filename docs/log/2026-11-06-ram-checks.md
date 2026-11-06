# RAM pre-load checks: FP8 without the repack, the split checked before its upload, refusals name the way out (#183, #184, #185)

- Date: 2026-11-06. Branch: fix/ram-checks. Base: main ad0c65e. Status: open (PR @@PR@@).
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
  - **`staging`** (`CudaQwen2::upload_staging`): the bytes `upload::Ring::new` page-locks, which is
    `upload::Shape::pick` of the plan's total. That's 4 × 4 MiB, or 8 × 8 MiB for 1 GiB or more, and none under
    64 MiB or with `AXON_UPLOAD_RING=0`. It's added to the need. The ring is freed before the CPU half is built, so
    adding it overstates the peak by at most its size, and that slack also covers the upload's smaller host costs.
    It's counted even when the upload then skips the ring (a cold page cache).
  - **`freed`** (`Model::release_gain`): on a copy of the file (or the loader's arena), the GPU half's bytes that
    `release` hands back as each tensor is uploaded. These are the whole 2 MiB blocks of each run of the GPU half's
    tensors, joined across bytes no tensor owns, exactly as `release` drops them. On a file mapping it's 0, because
    those pages are page cache. It's credited against the CPU half's need, not the ring's, because the ring is pinned
    while the copy still holds those bytes.
  The refusal names both: `(its repacked weights … GiB, the KV cache of … tokens … GiB, the GPU half's upload staging
  0.02 GiB, less the GPU half's … GiB of the copy, handed back as it goes up)`.
- **#185, `model::ram_remedy`: the refusal names its ways out**, after `(bound by …)`:
  - When the weights fit without the cache: the longest `--ctx`, in steps of 256, whose KV floor fits beside them.
  - The CPU model: `to fit: a GPU (--device cuda runs the model there, or with --gpu-layers N part of it)`.
  - A split chosen with `--gpu-layers`: `more --gpu-layers than G (fewer layers on the CPU), if the GPU has room`.
  - `--gpu-layers 0`: `--gpu-layers N (N layers on the GPU)`.
  - An automatic split (`hybrid::auto_gpu_layers`): `The split was picked to fit the GPU's memory (G of N layers
    there), so RAM is now the limit; to fit: a smaller --ctx, which also leaves the GPU room for more layers`.

  With #184 the remedy's `--ctx` is computed with the staging added and the copy's credit given.
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
Each new test fails before its fix. The expect-fail tree is the candidate with the fixes reverted and the tests
kept: #183's rule back to main's early return, `ram_remedy` returning nothing, and the split's check moved back after
the upload with no GPU-half terms (main's order).
- `model::ram_dense_tests::fp8_arena_counted_without_repack` (bella, FP8 0.5B): the estimate's arena is the built
  one (`Qwen2::arena_len`, through the r8 or F8S rows) for k = 1, n/2, n, with the repack on and off.
  `safetensors::tests::tiny_fp8_ram_arena_without_repack` does the same on the tiny checkpoint, so CI runs it.
- `model::ram_dense_tests::dense_need_and_refusal_wording`, `long_ctx_warns_not_refuses`,
  `split_refusal_wording_and_cap`: #185's exact strings and figures. The `--ctx` figure is the largest fitting step
  (`fits(4608) && !fits(4864)`). #184's math: 2400 + 512 − 512 + 64 MiB refused a byte under the edge and through
  at it. A credit larger than the CPU half doesn't pay for the ring.
- `hybrid::tests::hybrid_ram_refused_before_the_upload` (GPU, the 0.5B at 12/24): refused with no RAM. When the RAM
  is read, 0 tensors have been released (each goes back right after its upload), and none after the refusal. At main's
  order, 12 layers' worth are. The message names the 16 MiB ring. Then a copy of Qwen3-0.6B at one GPU layer and ctx
  2048: the credit is positive and at most the GPU half's bytes, and the half is refused a byte under the edge and
  built at it. The credit (one layer, ~16 MiB) is less than the CPU half's need (~220 MiB, mostly its KV floor), so
  the edge moves with both terms. A mutant with no credit (`freed: 0`) is refused at the edge; without the ring the
  message check above fails.
  - The first version did the copy on the 0.5B at 12/24 and ctx 512, and failed on bella (1106-084830-8699, "the
    half past the tolerance"). The test was wrong, not the check. On a copy, a CPU half commits only its growth, one
    matrix in flight and its KV (`dense_load_bytes`): ~6 MiB there. The credit was ~250 MiB: 12 layers plus the
    0.5B's untied 138 MiB head, which goes up with any GPU half. That left the 16 MiB ring, under bella's 38 MiB
    tolerance (32 MiB + 24 CPUs x 256 KiB), so no RAM figure could refuse that half. By the same figures no split of a
    copied 0.5B (f16 KV) can be refused: its largest CPU half (one GPU layer, the KV floor at 8192) is ~95 MiB against
    a ~147 MiB credit, and both are real (the head's copy goes back as it's uploaded). The test now checks its premise (ring > 0, credit < the CPU half, need > tolerance) and prints the
    figures.
- `gguf::tests::release_gain_is_what_release_drops` (Linux, mincore): on an anonymous model, the resident bytes that
  releasing a run of four 1.5 MiB tensors and one apart drops equal `release_gain`, and it's 0 on a file mapping.
- `hybrid::tests::hybrid_ram_check_on_real_splits`: the edge now includes the ring. At main's order it builds one
  byte short (the ring isn't weighed).

### Gates (e6704b7)
| gate | result | job |
|---|---|---|
| the new tests' pure parts, release (jade, CPU) | PASS 40 / 0 / 0 (`gguf:: model::ram_dense_tests safetensors::tests::tiny_fp8`) | 1106-072522-4e45 |
| tier filter, release, GPU (bella) | pending | 1106-072820-c2df |
| the same, debug assertions + overflow checks | pending | 1106-072821-c793 |
| tier filter, release, `--cpu` (bella, GPU hidden) | pending | 1106-072833-36f6 |
| the same, debug | pending | 1106-072834-211d |
| the new tests at main's code (expect-fail tree) | pending | 1106-072821-8945 |
| clippy `-D warnings` (harness, binary), rustfmt | clean | local |

## What didn't work / notes
- The first run of `fp8_arena_counted_without_repack` (bella `1106-064626-f708`) failed with "no FP8 arena built".
  `Qwen2::r8_arena` finds the shared arena through a repacked matrix, and FP8 layers hold F8S rows instead. The tests
  now read the arena through either (`arena_len`, cfg(test)). `r8_arena`, which the streamer pins, is unchanged.
- Not covered: a cap below the CUDA context's own host memory (option B above), and a real cgroup run of the new
  order. The ordering is shown with injected RAM. #180's `splitcap.sh` demo wasn't rerun.
