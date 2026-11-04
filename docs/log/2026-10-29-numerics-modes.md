# Numerics modes: the scaffolding (`--numerics exact|fast`, `serve --datacenter`), no kernels (PR #173)

- Date: 2026-10-29. Branch: feat/numerics-mode. Base: main 333276d. Track WP-A0 of the beat-vLLM scope
  (`specs/beat-vllm-scope.md` §6), steps M1, M2, M3, M5 and M8 of [the ADR](../design/determinism-modes.md).
- Decision: Haseeb, 2026-10-29 ("lets try fast"): the ADR is accepted. The lead's answers to its §7: ship M1 now
  (every fingerprint changes once, with a release note); the bundle is `serve --datacenter`; M4 (CPU fast prefill)
  and M6 (exact split-K) are deferred.

Labels: **MEASURED** (job cited), **CODE**, **OPINION**.

## What changed

No kernel changed, and exact mode's output bits didn't (the md5 gate: same greedy bytes as main). What changed is
the identity text and a second mode that today runs the same kernels under its own identity.

- **`src/numerics.rs` (new): `Numerics { Exact, Fast }`** (public as `axon::Numerics`, non-exhaustive),
  `BackendOptions::numerics`, `Backend::numerics`. The mode is a `Backend`'s (its second field) and reaches the
  placement code through `lm::Placing::numerics`; the CLI sets it per process.
- **M1, identity (`src/identity.rs`):** two counters, `NUMERICS_VERSION_EXACT` and `NUMERICS_VERSION_FAST` (each
  with its bump rule), and the tuple's first line is `numerics=exact/1` or `numerics=fast/1.1` (was `numerics=1`;
  fast's tag is exact's counter and its own, below). Every fingerprint changes once (CHANGELOG.md's release note);
  the golden canonical text and both fingerprints are pinned (`fp_1e9019e52a04cc54` exact, `fp_bea740cf61d9d767`
  fast for the test tuple; hashlib agrees). `GET
  /identity`'s `numerics` is the string `"exact/1"` (was the number 1). `axon info` prints each placement's
  fingerprint in both modes.
- **M1, the flag:** `--numerics exact|fast` on `gen`, `logits`, `chat` and `serve` (default exact). With no fast
  kernel on the backend (none exists anywhere: `numerics::fast_kernels` is false), `fast` runs the exact kernels and
  prints once: `axon: --numerics fast: no fast kernels on this backend yet; running exact kernels; identity says
  fast`.
- **M2:** `serve --datacenter` implies `--numerics fast` and, today, nothing else; an explicit `--numerics` wins
  (`serve::datacenter_numerics`).
- **M3, the graded harness (`src/cuda/qwen2/tests/numerics.rs`):** `graded_vs_exact(model, mode, device)` puts the
  model on this host's device three times: the exact reference, the net under test in `mode`, and an exact-vs-exact
  floor (a second GPU of another architecture if the host has one; else the q8 KV cache vs f16; on the CPU the
  per-row kernels vs r8). It tallies PROMPTS × 16 greedy steps and 3 × 48 teacher-forced natural-text tokens, with a
  ΔNLL (mode − exact) and an SE clustered by window, and `assert_graded_vs_exact` applies `assert_graded` (the
  GPU-vs-CPU gate, unchanged) with the floor as the reference spread; while the net runs no kernel of its own it
  also requires exact's bits (KL 0, ΔNLL 0). Tests: exact vs exact and fast vs exact on the GPU and the CPU (green
  today), and `graded_catches_another_model`: 0.5B Q4_0 graded against Q4_K_M must fail the gate (the test
  asserts it does: the gate has teeth). The script hook: `scripts/numerics_dnll.py`, through `axon logits` with `AXON_LOGITS_NLL=1` (new:
  the teacher-forced next token's −ln p per position).
- **M5, cache keys:** every NVRTC module of a fast backend compiles with `-DAXON_FAST` (`cuda::FAST_DEFINE`, last
  in its options), so it is its own kernel-cache entry and `Module::fast` says so; exact's options are main's, byte
  for byte. kvsave: `cuda::numerics_tag(gpu, fast)` hashes the define in, and a fast `Backend` adds
  `numerics_mode: "fast/1.1"` to its session identity. `session_across_numerics_modes` (CPU, and CUDA where there is a
  GPU): a session saved in exact mode restores into a fast chat as tokens, `numerics_mode` named as why, and the
  other way round; within a mode it restores as bytes; the conversation continues as a fresh prefill would.
- **M8, docs:** the roadmap's "Not planned" bullet says split-KV / NCCL / atomics are not in *exact* mode and
  allowed in fast mode once a measured gap names them; the ADR's status is Accepted with §7 answered and an "As
  built" note; determinism.md links the ADR and describes both modes.

## Design calls (OPINION, each in the PR body too)

1. **`fast` is accepted, not refused** (the ADR's M1 row said refused). `--datacenter` implies fast; a refusal
   would make the bundle unusable until a kernel lands. The M3/M5 tests need a fast mode that runs. And the
   fingerprint stays honest: `fast` promises tolerance, never exact bits, and the first fast kernel bumps
   `NUMERICS_VERSION_FAST`. The notice stops anyone reading `fast` as faster today.
2. **Exact session files are unchanged.** Only fast adds `numerics_mode` to the kvsave identity, so files saved by
   main still restore as bytes, and exact module options are main's, so the NVRTC cache entries are reused.
3. **Draft models always run exact.** A drafter only guesses; the output is the main model's either way.
4. **The fast-mode χ² on the real model (ADR §4.4) is not here.** With no fast kernel, fast's verify rows are
   exact's, so identity F holds bit for bit by the same code; the χ² belongs to the first fast kernel's PR.
5. **The cross-arch floor is written but unexercised:** no Axon host has two GPU architectures in one box.

## Review round 1 (shady 7/10 CHANGES on 8a273d0): what changed

- **The graded gate never ran a batched pass** (MUST 1). Every graded position was a decode position: PROMPTS fed
  their few tokens and decoded, the teacher-forced windows were one token per step, and `numerics_dnll.py` set
  `AXON_LOGITS_ALL` to every position, so `axon logits` never called `feed_batch`. WP-A1's first fast kernels are
  prefill GEMMs (nb > 16), so a fast GEMM would have passed this gate without running once. Now the harness grades
  four arms, each on its own: *decode* (as before, 4 windows); *batched prefill* (24, 160 and, on the GPU, 520
  tokens in one `prefill_logits`: a gemvm pass, a GEMM pass ≥ 128 rows, a long pass ≥ 512; the last row, 16
  positions decoded after it on the cache it wrote, and `verify_argmax`'s pick on every row); *cached prefix* (64
  tokens cached, 96 more in a second pass); *under load* (GPU: 40 sequences stepped together in one pool, every
  row's full logits against the sequence alone). `axon logits` takes `AXON_LOGITS_PREFIX=N` (the first N tokens in
  batched passes, position N−1 from the last pass's last row), and `numerics_dnll.py --prefix N,...` (default
  0,24,160) reports ΔNLL per prefix.
- **A fast kernel must be seen running.** `CudaQwen2` counts fast-set launches (`fast_launched`, to be called at
  every fast launch site; debug builds assert it's a fast module), `Backend::fast_launches` sums them, each arm
  records its own, and the gate requires the count to move when `Backend::fast_kernels()` (now public) is true and
  to stay 0 when it's false (`fast_launch_count_is_read` checks both failures by message).
- **The fast contract promised bits** (MUST 2). determinism.md's fast row said a fixed (identity, request) still
  gets a fixed reply on a fixed placement; the ADR's §4.2 makes batching, cache reuse and drafting tolerance in
  fast mode, so the row now says no bits on any path, and that the fingerprint names the kernel set, not the bits.
  `identity_checks` takes `bits`: with a fast kernel on the backend, *every* path equality (cached, streamed, under
  load, fast == exact) is off and graded by the harness arms instead, not just fast == exact as the old comment said.
- **The counters were tied by a comment** (MUST 3). Fast's tag now carries exact's counter: `fast/1.1`
  (`fast/<exact>.<fast>`), so a change to code both modes run bumps `NUMERICS_VERSION_EXACT` once and both modes'
  fingerprints move; `NUMERICS_VERSION_FAST` counts fast-only changes. Free now: no client holds a fast fingerprint.
- **One source of truth for the mode** (SHOULD 4): `Backend::of` builds every `Backend` and, in debug builds,
  asserts each of its GPU models runs its mode's module (`Module::fast`), so a fast module never sits under an exact
  fingerprint.
- **The floor question goes to WP-A1** (SHOULD 1), written into the ADR's §7 as an open item: the floor is a
  reordering floor, and FP8 / W4A16 change precision; what they are graded against is not decided here.

## How it was verified

The gates (release and debug-assertion suites, md5 of `gen` 20 cells × {main, cand, cand `--numerics exact`, cand
`--numerics fast`}, the identity/serve/kvsave/numerics test groups, the ΔNLL hook exact-vs-fast, clippy, rustfmt)
and their job ids are in the PR body. Benchmarks: none (no kernel changed; the md5 gate is the evidence that exact
mode's bits did not move).

## What didn't work / what's left

- M4 (CPU fast prefill) and M6 (exact split-K) are deferred by decision. M7 (the first fast kernel) waits on a
  measured gap and a kill rule (ADR §4.6).
- Pod pre-warm (`axon-pods warm`) loads exact modules only, so the first `--datacenter` job on a pod pays its fast
  modules' NVRTC compiles (about a minute each) inside its own warm-ups until the warm path also runs
  `--numerics fast` (a devstack change, not in this PR; see the PR body).
- `tests/public_api.rs` gets `Numerics` in its surface; like the rest of that file it runs under cargo (CI), not in
  the rustc-built suites.
