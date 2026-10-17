# Determinism docs: drafting and --ctx can move an automatic split  (docs only)

- Date: 2026-10-17. Branch: docs/draft-placement-note. Base: main 85d6a08 (after #52). Status: open.
- Source: shady's NIT in the #52 re-review (r10 forum thread).

## What changed
- `docs/determinism.md`: the drafting row, the CPU/GPU-split bullet and the "left out because they never
  change bits" paragraph now say this holds for a given placement. On an automatic CPU/GPU split the split
  point follows what has to fit next to the layers (a `--draft-model`'s VRAM reserve, the KV cache's size via
  `--ctx`, `gen --ctx` included), so either can move the split, which changes the bits; the identity's `backend`
  (`gpu layers N/M`) changes with them, so the fingerprint stays truthful. Pin `--gpu-layers` for reproducible runs.
- `src/identity.rs`: the same caveat in the module comment's "not in it" list. Comment only.

## Why
The old wording said drafting and the context size never change bits. True per placement, but on a split the
automatic placement takes both as inputs. docs/design/spec-split.md (from #52) measured it on bella:
24 vs 25 GPU layers across rounds gave different chat/summary tokens; with `--gpu-layers 24` pinned, four
processes agreed on every prompt and K.

## Verification
Docs and comments only; no code paths touched. CI.
