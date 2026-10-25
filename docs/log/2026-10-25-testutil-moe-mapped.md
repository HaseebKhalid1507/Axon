# testutil: MoE test models mapped, as axon runs them  

- Date: 2026-10-25. Branch: fix/testutil-moe-mapped. Base: main 2a6d8fd. Status: merged.
- Found by: #127's bella gate (`ds2_resident_moonlight_bits`), jobs 1024-191039-2c90 and 1025-001953-7566.

## What changed
`testutil::ds2_model` / `moe_model` load through the new `testutil::load_moe`, which uses `gguf::load_with(path,
Some(false))` (mapped) instead of `gguf::load`. New CPU-only unit test `moe_test_models_are_mapped`. Test code only.

## Why
- `gguf::load` copies a file under 60% of `MemAvailable` into anonymous memory (`copy_choice`).
- The MoE RAM check `moe::check_ram` (main code, unchanged) compares the r8 arena to `/proc/meminfo`'s
  `MemAvailable`. It is **not** cgroup-aware, and it is correct for the product.
- So a copied model counted as used RAM:
  - On bella with ~18 GB available, Moonlight (10.5 GB) was copied, the check then saw 10.0 / 10.8 GB for an
    11.3 GB need, and refused.
  - On a busier host (< 17.6 GB available) the same test maps the file and passes. That's why earlier bella runs
    and other arms passed.
- The product never copies a MoE model: `load_to_run` asks `CopyWish::MapOnly`. So `axon gen` / `chat` / `serve`
  weren't affected, and neither are cgroup-limited users.

## How it was verified
| gate | result | job |
|---|---|---|
| `testutil::load_tests` (CPU) | PASS 1 / 0 / 0 | 1025-002904-8d5e |
| `moe::` + `testutil::` suite, release | PASS 31 / 0 / 1 on avante | 1025-005852-c633 (replaces jade 1025-002904-23e1: OOM-killed by jade's memory cap, the requeue crawled; dropped. Loader-only change, so the host doesn't matter) |
| `moe::` + `testutil::` debug (CPU) | PASS 31 / 0 / 1 | 1025-002905-73a5 |
| the failing gate, via #127 0b789e3 (merges this branch) | PASS 15 / 0 / 1: Moonlight bits identical at --gpu-experts 0 / 6 / 12 / auto | 1025-003357-f54e (bella) |
