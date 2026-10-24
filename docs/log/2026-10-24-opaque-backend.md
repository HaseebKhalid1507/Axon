# lib: opaque `Backend`, sealed `Lm`, `BackendOptions`, env errors instead of panics  (PR #124)

- Date: 2026-10-24. Branch: feat/opaque-backend (PR #124). Base: main 6e3b19c. Status: open. Round 1 code daada6f; round 2
  (Shady 8/10 APPROVE, the SHOULDs below) is the gated head.
- Pick / design: issue #118 items A and B, the v0.1.0 blockers from Shady's round-2 review of #117
  (https://github.com/HaseebKhalid1507/Axon/pull/117#issuecomment-6296699905).

## What changed
- **A. `Backend` / `BackendCache` are opaque.** `pub struct Backend<'a>(pub(crate) Inner<'a>)` and
  `pub struct BackendCache(pub(crate) CacheInner)`. `Inner` / `CacheInner` are the old enums with the same variants
  and the same payloads (`Box`es where they were). The 96 lines in the crate that named a `Backend::X` / `BackendCache::X` variant now name
  `Inner::X` / `CacheInner::X`: lm.rs 48, serve.rs 15, cli.rs 11, kvsave.rs 9, identity.rs 8, draft.rs 2, and
  three test modules 1 each.
  Each match subject went from `net` to `&net.0`, and each constructor got wrapped in `Backend(..)`. `Backend::new` and
  `Backend::ctx` keep their signatures.
- The model and cache types behind it are `pub(crate)`: `Qwen2`, `CudaQwen2`, `HybridQwen2`, `PipelineQwen2`,
  `MoeQwen3`, `KvCache`, `CudaCache`, `HybridCache`, `PipelineCache`, `MoeCache`. So are their formerly `pub` fields
  (26) and inherent methods (71), done by a script (`pub` → `pub(crate)` inside those structs and their inherent
  `impl` blocks, nothing else). Before this, they were unnameable but reachable through `Backend`'s public variants.
- `Lm` is sealed: `pub trait Lm: Sealed`, with `Sealed` a `pub` trait in a private module. The crate's 9 `Lm` impls
  (5 models, `Backend`, 3 test doubles) each add `impl Sealed`.
- `#[non_exhaustive]` on `Device`, `KvType`, `GenStats`. Constructors: `Device::Cpu` / `Device::Cuda` and the
  `KvType` variants are still usable as values, `Device::parse` / `from_env` and `KvType::parse` stay, and
  `GenStats::default` stays. Outside the crate, only `generate` makes a `GenStats` with data. No new constructor
  was needed (examples/embed.rs and tests/public_api.rs build unchanged).
- **B.** `AXON_SIMD` (unknown, or a set this CPU lacks), `AXON_DECODE_ROWS` (not a number, or not 1/2/4) and
  `AXON_PREFETCH` (not a number) are parsed once into a `Result` (`quant::simd_setting`, `DecodeCfg::setting`).
  `quant::env_check()` runs at the start of `axon::load`, of `Backend::new` (`new_capped`) and of the CLI's
  `open_model_with` (gen, logits, chat, serve), so a bad value is their error instead of a panic in a later
  forward pass. The CLI prints `axon: AXON_SIMD=bogus: expected scalar, avx2 or vnni` and exits 2. The hot-path
  getters (`simd()`, `DecodeCfg::get()`) still read one `OnceLock` of the plain value, as before. Only a caller
  that skipped `load` and `Backend::new` could still reach their panic.
- Tests: 7 `compile_fail` doc tests (below), `tests/public_api.rs` (the embedder API from outside the crate: a
  compile-time coverage function, `Device`/`KvType` parse and wildcard matches, and examples/embed.rs's path on CPU
  and CUDA, `generate` == `feed_batch` + `forward_argmax` steps), `quant::tests::decode_cfg_parse`, and
  `simd_choice_rejects_unknown` (was `#[should_panic]`, now checks the `Err`).

## Round 2 (review: Shady 8/10 APPROVE, no MUSTs; the SHOULDs, decided by the foreman)
- `Backend::new(&model, BackendOptions)`. `BackendOptions` is `#[non_exhaustive]` with private fields, `Default`
  (CPU, 4096-token context, no split, f16, no fit) and chainable setters `device`, `max_ctx`,
  `gpu_layers(impl Into<Option<usize>>)`, `kv`, `fit`. #118 item 2's settings become fields of it later without a
  break. All 19 callers moved (the crate's tests, draft.rs, lib.rs, examples/embed.rs, tests/public_api.rs).
  `new` maps it to `new_capped` field for field, so nothing it does changes.
- `#[doc(hidden)]` on `Lm::forward_cands` / `verify_cands` and on `sample::Cand` / `CandSpec`, until item 8.
- `env_check` prints nothing. `choose_simd` used to print the "AVX2 but not fma/f16c" notice while choosing.
  Now it only chooses; `simd()` prints the notice at the kernels' first use, as main did, so a pure-GPU `load`
  doesn't print it (`scalar_notice`, tested).
- `tests/public_api.rs` gains `second_backend_{cpu,cuda}`, the embedder's model swap: qwen2.5-0.5b, dropped; then
  Qwen3-0.6B (a second `Backend::new` in one process), which generates; then the 0.5B again, which must give
  the first run's tokens.
- CI checks `compile_fail` error codes. `scripts/compile_fail_check.py` finds every `compile_fail` block under
  src/, requires an error code on each, compiles it alone against `target/debug/libaxon.rlib` and checks that the
  first error has that code. It's a step after `cargo test` in ci.yml. Stable rustdoc doesn't check the code.

## Other environment parses (the B grep)
Every `std::env::var` in non-test library code was read (cli.rs's are CLI-only). Only the three above could
panic inside a library call. The rest either return an error already (`AXON_KV`, `AXON_DEVICE`, `AXON_GPUS`,
`AXON_PP_CUTS`) or treat a value they can't parse as unset or default, with no panic: `AXON_THREADS`,
`AXON_DECODE_THREADS`, `AXON_ATT_MQ_MAX`, `AXON_GEMVM_MAX`, `AXON_STREAM_RING_MB`, `AXON_STREAM_MIN`,
`AXON_UPLOAD_RING`, `AXON_PAGE_SHUFFLE` (hashed), `AXON_DEC_GU4`, `AXON_Q6K8`, `AXON_CUDA_PDL`, `AXON_MMAP` and the
0/1 switches. Making those strict too would change what main does with a typo, so it's left for #118 item 2
(an options struct).

## Why
#118 A: through the public `Backend::Cuda(c)` variant, safe code outside the crate could write
`c.vocab = 0`. `CudaQwen2`'s geometry fields size raw kernel launches and host buffers, so that write is a
safe-code path to UB. It also froze for 0.1.x every variant (a `Backend::Deepseek2` would have been a breaking
change) and every `pub` field of five model types. #118 B: the same class as #117's old `AXON_KV` panic.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| fast tier, release (2 shards) | round 2: pending | (round 1 daada6f: 1024-171219-bdcb ran; da22 dropped) |
| fast tier, debug assertions + overflow checks (2 shards) | round 2: pending | (round 1: 1024-171219-a517 PASS, 1024-171219-bb9b ran) |
| md5 of `gen`, main 6e3b19c vs candidate, bella: standard set (md5-gen.sh) + md5ext + serve A/B + public_api | round 2: pending | (round 1 1024-171216-7030 dropped before it ran) |
| `compile_fail` doc tests (rustdoc --test) | 7/7 + the crate example compile | local |
| each snippet's first error vs main | as annotated, see below | local (`snipcheck.sh`) |
| clippy -D warnings (lib, lib tests, bin, examples/embed, tests/public_api), rustfmt | clean | local |

Rustdoc on stable doesn't check a `compile_fail` block's error code, so each snippet was also compiled on its own,
against main's lib and against the candidate's:

| snippet | main 6e3b19c | candidate |
|---|---|---|
| `if let Backend::Cuda(c) = b { c.vocab = 0 }` | compiles | E0599 no associated item `Cuda` |
| `&mut b.0` (Backend) | E0609 (no field) | E0616 field is private |
| `&mut c.0` (BackendCache) | E0609 | E0616 |
| `impl axon::Lm for Mine` | compiles | E0277 `Sealed` not satisfied |
| exhaustive `match` on `Device` | compiles | E0004 `_` not covered |
| exhaustive `match` on `KvType` | compiles | E0004 |
| `GenStats { .. }` literal | compiles | E0639 non-exhaustive struct |

## Results
No behaviour change for valid settings: same code paths, same kernels, same bits (md5 gate above). No speed
claim. `Inner` has the same layout as the old enum, so there's no extra indirection: `&self.0` is the same
address.

## What didn't work / not done
- `Deref<Target = Inner>` for `Backend` (to keep `match &*net`) would need a crate-private type in a public
  impl. Explicit `.0` instead.
- `Lm::forward_cands` / `verify_cands` still name `sample::Cand` / `CandSpec` (pub types in a private module):
  outside the crate they can't be called, because `CandSpec` has no constructor there, but they show up in the
  docs. That belongs with #118 item 8 (export the sampler).
