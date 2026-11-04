# The CPU half of a `--gpu-layers` split refused when it can't fit in the RAM available  (PR #180)

- Date: 2026-11-04. Branch: fix/split-ram-check. Base: main e04b808, merged up to 9948129 (#157). Status: open
- Pick / design: issue #143 (follow-up of #142, which refused CPU-only dense loads; #133 the cgroup bound).

## What changed
- **The split's CPU half is checked** (`model::check_ram_split`, called in `HybridQwen2::build`). A split builds its
  CPU half as `Qwen2::new_layers(m, repack, k)`: layers 0..k (k = n − `--gpu-layers`), their repacked matrices in
  one r8 arena, their KV cache. The output head is the GPU's (the CPU half only gets it when k = n, `--gpu-layers 0`),
  and the embedding is read from the model's own bytes, never repacked. `dense_ram_need_keep(m, repack, elem, k)` is
  #142's `dense_ram_need` restricted to that: layers below k, the head iff k ≥ n, the KV of those k layers.
  `dense_ram_need` is now `dense_ram_need_keep(.., usize::MAX)`: same figures. The refusal is #142's
  (`dense_verdict`, same tolerance, same KV floor of min(ctx, 8192) positions, the long-context note), with the split
  named:
  `qwen2: the CPU half of the split (18 of 36 layers) needs about … GiB of RAM (its repacked weights … GiB, the KV
  cache of … tokens … GiB) and only … GiB is available (bound by cgroup …)`.
  With `--gpu-layers 0` (no GPU half): `the CPU model (--gpu-layers 0: all N layers on the CPU) needs about …`.
- **When:** after the GPU half's upload, right before the CPU half is built, so `avail()` sees what the CUDA context
  and the GPU half's upload already hold in host memory, and (on a copy of the file) the GPU's weights handed back.
  Exit 2 through the CLI, like every `Backend::new_capped` error. No CPU half (k = 0, every layer on the GPU) never
  reads the RAM (the CLI runs that as the plain GPU model anyway).
- **Streaming's pinned KV** (`HybridQwen2::cache_with`, `pin_verdict`): a split that streams prefill through the GPU
  page-locks the CPU layers' K and V buffers whole when the cache is made (`stream_kv`, cuMemHostRegister), not as the
  context fills: every position of the cache's max_ctx, at once. So a cache whose pinned bytes plus the tolerance
  don't fit in the RAM available doesn't stream: `note: this cache won't stream: page-locking the CPU layers' KV cache
  of N tokens takes … GiB of RAM and only … GiB is available (bound by …)`, the existing graceful path (the cache is
  an ordinary lazy one; prefill runs on the CPU's layers; bits are the non-streamed split's). The tolerance is kept
  free here (unlike the load refusal) because pinned pages can't be reclaimed and not streaming only costs speed.
- README (CPU memory limits, `AXON_STREAM_MIN`), CHANGELOG.

## Paths checked for the same gap
| path | gap? | what |
|---|---|---|
| `--gpu-layers N` split (`Backend::new_capped` → `HybridQwen2::new_with_stream` → `Qwen2::new_layers`) | yes: fixed | `check_ram_split` |
| automatic split (no `--gpu-layers`, `auto_gpu_layers`) | yes: fixed | same `build` |
| `--gpu-layers 0` with `--device cuda` (a hybrid with no GPU half) | yes: fixed | same, the whole CPU model's estimate |
| streamed prefill (`AXON_STREAM`, default on) | yes: the cache's pinned KV | `pin_verdict` per cache; serve makes one cache per slot, each checked as it's made |
| `AXON_STREAM_DECODE=1` | same cache as streaming | covered by the same check |
| MoE (`Backend::moe`) | no layer split: `--gpu-layers` < n is refused | `moe_check_ram` / `moe_check_ram_keep` (#142 and before) |
| `--gpus` pipeline | every layer on the GPUs, no CPU half | none needed |
| serve's KV RAM tier (`--cache-ram`) | off on a split (no page image) | none needed |

Follow-ups, not in this PR:
- FP8 checkpoints with `AXON_REPACK=0` still build an arena (their FP8 → F8S rows) that `dense_ram_need(repack =
  false)` doesn't count; #142's CPU check has the same blind spot. Small (the FP8 0.5B only in practice).

Follow-up candidate the cgroup demo shows (not #143's, which is the CPU half; also on main):
- The GPU half's host-side memory (the CUDA context, the upload) is taken before any check runs. Under a cap below
  it (about 0.2 GiB at these caps: the refusals' "only X available" put what's in use at the check at 0.18-0.37 GiB)
  the run is killed in CUDA init or the upload, as on main (the 0.5B at 56 and 184 MiB below; inferred from where
  the cap falls, the killed runs print nothing).

## How it was verified
Tests (all in the tier filter `model:: hybrid:: gguf::mem lm:: qwen2::stream`):
- `model::ram_dense_tests::split_need_matches_the_built_cpu_half`: for k in {0, 1, n/2, n-1, n, n+3},
  `dense_ram_need_keep` is the built `Qwen2::new_layers(m, true, k)`: arena = `r8_arena().len()` rounded to 2 MB, KV
  a position = the built cache's. The qwen3 and llama Q8_0 fixtures and the 0.5B, Qwen3-0.6B, Llama-1B, gemma-3-1b.
  k = 0 is all zeros, k >= n is `dense_ram_need`, below n no head share.
- `model::ram_dense_tests::split_refusal_wording_and_cap`: the wording (cgroup bound), the long-context note; the
  real 0.5B's half at ctx 4096 refused one byte under the tolerance, through at it and at its need; k = 0 and MoE
  never read the RAM; k = n is the whole model's estimate with the `--gpu-layers 0` wording.
- `hybrid::tests::hybrid_ram_check_on_real_splits` (GPU): the 0.5B, Qwen3-0.6B, Llama-1B at 1, n/2, n-1, n GPU
  layers. The built CPU half's arena and KV are the estimate. A half that needs more than the tolerance is refused
  one byte under it (after the GPU half; the refusal names the split) and built at the edge (need - tolerance). A
  half within it can't be refused by any RAM figure (avail + tolerance >= tolerance > need) and is built with 0
  bytes available. The largest half (one GPU layer) must be past the tolerance on each model, so the refusal is
  shown on every model whatever the CPU count, or the test fails. No CPU half never reads the RAM.
- `hybrid::tests::hybrid_ram_check_every_layer_on_the_cpu`: the same with no GPU (`--gpu-layers 0`).
- `hybrid::tests::streamed_cache_pins_its_cpu_kv_whole` (GPU, Linux, mincore): an ordinary split cache has 0 of its
  CPU KV pages resident, a streamed one all of them the moment it's made; pinned = the estimate's KV a position x
  max_ctx; one byte short of pinned + tolerance gives a non-streamed cache whose prefill doesn't stream.
- `hybrid::tests::pin_verdict_keeps_the_tolerance_free`: the pure decision and its wording.
- md5 gate (`md5.sh`, axon-podjob): greedy `gen -n 64`, main and the branch interleaved in one job. 4 models
  (qwen2.5-0.5b, Qwen3-0.6B, Llama-1B, qwen2.5-3b) x {split n/2 streamed, n/2 `AXON_STREAM=0`, n/2 `--kv q8`, n-1
  GPU layers, `--device cpu`, `--device cuda`} x 2 prompts (short; ~400 tokens, streamed) = 48 cells.
- Cgroup demo (`splitcap.sh`, axon-podjob, bella): each `gen MODEL "The capital of France is" -n 8 --device cuda
  --gpu-layers n/2` in its own `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0`. main's smallest
  working cap bisected (16 MiB steps), then main and the branch at caps around it: OOM-killed (137), refused (2),
  runs (0; same text checked). Fails on a refusal where main ran. Then the streaming case: the 3B at `--ctx 32768`,
  both binaries' smallest caps bisected.

### Round 1 (0a45942)
| gate | result | job |
|---|---|---|
| tier tests, release (bella) | PASS 122 / 0 / 3 ignored | 1104-090950-0043 |
| tier tests, debug assertions (bella) | **FAIL** 121 / 1: `hybrid_ram_check_on_real_splits` "attempt to subtract with overflow" (tests.rs:1617) | 1104-090951-f09e |
| new tests' output (release, `--nocapture`) | PASS 4 / 0 | 1104-090951-ae05 |
| md5, main e04b808 vs 0a45942, 48 cells | PASS 48 / 48 identical | 1104-090951-357d |
| cgroup demo | PASS (tables below) | 1104-090952-3d81 |
| clippy -D warnings (lib, lib tests, bin), rustfmt | clean | local |

The debug failure was the test's: it refused every CPU half at `need - tol - 1`. The 0.5B at 23/24 GPU layers
(k = 1) needs 12 MiB, the tolerance on the job's 8 CPUs is 34 MiB (32 MiB + 8 x 256 KiB): the subtraction
underflows (the last line the test printed, 12/24, comes before the failing cell: it prints after the check). In
release it wrapped to about 2^64 - 22 MiB, and `dense_verdict`'s `avail + tolerance` wrapped back to `need - 1`,
so the build was refused and the cell passed: the release pass for that cell (and Qwen3-0.6B's k = 1, 18 MiB)
showed a refusal at a RAM figure no host has, not a cap that works. No RAM figure refuses a half within the
tolerance. The other 7 CPU-half cells (0.5B k = 23, 12; Qwen3 k = 27, 14; Llama-1B k = 15, 8, 1, the last 39 MiB
against 34) were genuine refusals. The product's `avail + tolerance` can't overflow on a real figure (`mem::avail`
is None for `max` and v1's unlimited, so `avail.bytes` is MemAvailable or a cgroup's room).

### Round 2 (cbd2715 test fix + 03e6a8c merge of main 9948129; gatewatch `splitram-03e6a8c-tests`, `splitram-03e6a8c-dbg-r2`)
cbd2715 changes test code only (the refusal arm per k as above; the other `- tol - 1` sites use `checked_sub` with a
message), and main's #157 adds only tests (`src/cuda/qwen2/slots.rs`, `mod tests`) and a log. The release `axon`
binary built at 03e6a8c runs the same bytes as 0a45942's: 29 of its 30 allocated sections are byte-identical
(`.text` 4508357 B, `.rodata`, `.data.rel.ro`, `.data`, `.eh_frame`, ...); the 30th is `.note.gnu.build-id`, a hash
over the whole file. Outside them only `.strtab` differs, the symbol names (not loaded; Rust's symbol hashes take in
the source files' hashes, and model.rs's test module changed). So the md5 gate and the cgroup demo stand from round
1 and weren't rerun.

| gate | result | job |
|---|---|---|
| tier tests, release (bella) | PASS 122 / 0 / 3 ignored (582 s) | 1104-094511-cde8 |
| tier tests, debug assertions (bella) | not run to the end: bella rebooted at ~10:56 mid-run ("runner gone without an rc") | 1104-094512-39e4 |
| the same, rerun (`-r2`) | PASS 122 / 0 / 3 ignored (1276 s) | 1104-142638-daac |
| the new tests' output under debug assertions (`--nocapture`) | PASS 5 / 0 (11 s; finished 10:55:35, before the reboot) | 1104-094512-0a6e |
| clippy -D warnings (lib, lib tests, bin), rustfmt | clean | local |

0a6e's output shows the fixed arm in debug: of the 12 real-split cells, the 0.5B's and Qwen3-0.6B's one-CPU-layer
halves (12 and 18 MiB, under the 34 MiB tolerance) print "within the tolerance: built with 0 bytes available"; the 7
others with a CPU half "refused 1 byte under the tolerance, built at it"; the 3 with every layer on the GPU "no CPU half".

## Results
The estimate is the build, exactly (round 1's output; arena rounded to 2 MB pages):

| model | layers | CPU half (k = n/2) arena, of the whole | KV a position, of the whole |
|---|---|---|---|
| qwen2.5-0.5b | 24 | 122 MiB of 376 MiB | 6144 of 12288 B |
| Qwen3-0.6B | 28 | 224 of 604 | 57344 of 114688 |
| Llama-3.2-1B | 16 | 286 of 776 | 16384 of 32768 |
| gemma-3-1b | 26 | 354 of 1014 | 13312 of 26624 |

What a real split's CPU half needs at ctx 512 (arena + KV), by GPU layers: 0.5B 233 / 125 / 12 / 0 MiB at 1 / 12 /
23 / 24; Qwen3-0.6B 486 / 252 / 18 / 0 at 1 / 14 / 27 / 28; Llama-1B 547 / 294 / 39 / 0 at 1 / 8 / 15 / 16.

Cgroup demo (1104-090952-3d81), half the layers on the GPU, `-n 8`. Caps in MiB:

| model (GPU layers) | main's smallest cap | below it: main | below it: branch | at and above it |
|---|---|---|---|---|
| qwen2.5-0.5b (12/24) | 312 | OOM at 56, 184, 297 | OOM at 56, 184, 297 | both run, same text (312, 376, 624) |
| Llama-3.2-1B (8/16) | 529 | OOM at 273, 401, 514 | **refused** at 273 (0.04 GiB available), 401 (0.17); OOM at 514 | both run, same text (529, 593, 1058) |
| qwen2.5-3b (18/36) | 1009 | OOM at 753, 881, 994 | **refused** at 753 (0.56), 881 (0.68); OOM at 994 | both run, same text (1009, 1073, 2018) |
| Llama-3.2-3B (14/28) | 1211 | OOM at 955, 1083, 1196 | **refused** at 955 (0.57), 1083 (0.69); OOM at 1196 | both run, same text (1211, 1275, 2422) |

- No refusal where main ran. The OOM 15 MiB under main's smallest cap is the tolerance's band (34 MiB here): within
  it a load isn't refused (#142's rule), it may then be killed.
- The 0.5B is never refused: its CPU half is 125 MiB, and at 56 and 184 MiB the run presumably dies before the check (the
  GPU half's host memory; the follow-up above), at 297 it is in the tolerance's band.
- The refusal reads e.g. `qwen2: the CPU half of the split (18 of 36 layers) needs about 0.80 GiB of RAM (its
  repacked weights 0.79 GiB, the KV cache of 14 tokens 0.00 GiB) and only 0.56 GiB is available (bound by cgroup
  /user.slice/.../memory.max ...)`.

Streaming (the 3B at 18/36, `--ctx 32768`: the CPU layers' pinned KV is 18 x 1 KiB x 32768 = 576 MiB):

| cap (MiB) | main | branch |
|---|---|---|
| smallest working cap (bisected) | 1598 | 1118 |
| 1118, 1358, 1583 | OOM | runs, not streamed (note: pinning 0.56 GiB, only 0.11 / 0.34 / 0.56 available) |
| 1598 | runs, streamed | runs, not streamed (0.57 available: short of pinned + tolerance) |
| 1662 | runs, streamed | runs, streamed |

The branch runs a 32K-context split in 480 MiB less than main by not pinning what doesn't fit. In the tolerance's
band at 1598 MiB it gives up streaming (slower prefill, same bits) where main just managed it.
