# testutil: MoE test models mapped, as axon runs them  

- Date: 2026-10-25. Branch: fix/testutil-moe-mapped. Base: main 2a6d8fd. Status: open.
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
| `testutil::load_tests` (CPU) | UNIT | 1025-002904-8d5e |
| `moe::` suite, release (jade) | MOEJ | 1025-002904-23e1 |
| `moe::` + `testutil::` debug (CPU) | DBG | 1025-002905-73a5 |
| the failing gate, via #127 (merges this branch) | BELLA | see #127 |
