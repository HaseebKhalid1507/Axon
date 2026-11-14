# 2026-11-12: precompiled kernel pack (dist track, PR #227)

**What:** a release `axon` carries its CUDA kernels precompiled (the *kernel pack*), so `--device cuda` runs on a
machine with only the NVIDIA driver (no toolkit, no `libnvrtc`). `axon kernels build OUT` compiles every product
module with NVRTC for sm_75/80/86/89/90/90a/120 + compute_120 PTX (no GPU needed), each image a fatbin compressed
by `fatbinary`; `--cfg axon_kernel_pack` + `AXON_BUILD_KERNEL_PACK` embeds it (src/main.rs). `Module::load` takes
the pack's image when it covers the GPU and the driver, else NVRTC for the same target. Kill switch
`AXON_KERNEL_PACK=0`; `=only` refuses NVRTC; `AXON_KERNEL_PACK_FILE` loads a pack file.

**Why (a) the pack, not (b) shipping libnvrtc:** 14 module variants (cache keys) per target, 8 distinct images
(every fast cubin equals its exact twin), 106 compiles over 8 targets. libnvrtc + builtins is 133 MB and the first
run would still compile 1-2 min per module; the pack is a few tens of MB inside the binary, with no compile at
start-up and no NVIDIA blob to redistribute.

**Bits:** an entry is NVRTC's output for the same source, options and target as the run-time compile; same NVRTC
version, same bytes (each entry records the hash of NVRTC's raw output). Checked by `gpu_release_pack_is_nvrtcs`,
`gpu_pack_module_is_nvrtcs` (logits bit for bit) and the md5 gate (main on NVRTC vs the pack with
`AXON_KERNEL_PACK=only`): results in the PR.

**Driver floor:** the pack needs R580+ (CUDA 13). An older driver without a CUDA 12 toolkit gets one line from
`--device cuda`: the driver version (`/proc/driver/nvidia/version`), "need R580+ (CUDA 13)", and `--device cpu`;
a pre-CUDA-12 driver, whose API axon can't load, gets the same line instead of a missing symbol.

**Review (shady, 7/10, 11-13):** `axon kernels build .` would have run `remove_dir_all(".")` after the build
(`with_extension` does nothing to a path with no file name). OUT must now name a file that isn't a directory
(refused before NVRTC is looked up), and the build works in a new `.OUT.axon-kernels-PID.work` beside it
(`create_dir`: an existing one is an error), the only thing it removes. Also: an unusable pack warns once before
NVRTC takes over; `axon kernels list` exits non-zero on a pack from another version of the kernels; a fatbin's
own size (headerSize + fatSize) must lie inside its image; reserved bytes must be zero. The headline claim gets
its clean-machine run on a driver-only RTX 4090, md5s against main's NVRTC output (PR body).
