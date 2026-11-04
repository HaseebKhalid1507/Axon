# The CPU half of a `--gpu-layers` split refused when it can't fit in the RAM available  (PR #TBD)

- Date: 2026-11-04. Branch: fix/split-ram-check. Base: main e04b808. Status: open
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

## How it was verified
(pending)

## Results
(pending)
