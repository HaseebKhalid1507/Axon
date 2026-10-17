# AGENTS.md

The contract for anyone sending Axon a change: people, and AI agents working for people. Read all of
it before you touch the code. The rules are literal. If a rule and an issue disagree, ask on the issue.

## Mission

Make LLM inference cheap and fast, and run the best open models. Squeeze every drop out of the
hardware, from a laptop to a datacenter: the most tokens per GPU, per watt and per dollar, as close to
the hardware limit as the math allows. Every change serves one of those, and has numbers to show it.

Background, in this order: [README.md](README.md), [docs/START-HERE.md](docs/START-HERE.md),
[docs/methodology.md](docs/methodology.md), [docs/roadmap.md](docs/roadmap.md). Before you start
anything perf-related, read [docs/experiments-rejected.md](docs/experiments-rejected.md): it lists what
was already tried and lost, with numbers.

## Workflow

1. **Find an issue labelled `agent-ready`.** Only those have a spec with scope and acceptance gates.
   `needs-spec` means the spec isn't written yet: don't start it. Open new work with the
   **Change request** form, which a maintainer turns into a spec. Labels: [.github/labels.md](.github/labels.md).
2. **Comment `claiming`** on the issue. Add one line on your approach if it isn't obvious.
3. **Wait to be assigned.** A maintainer assigns you and adds `claimed`. An issue that is already
   assigned or `claimed` is taken. Don't start work on it.
4. **Open a PR within 48 hours of being assigned**, or the claim lapses and the issue goes back to
   `agent-ready`. An early PR with gates still pending is fine. Say which ones are pending.
5. **One PR per issue, one issue per PR.** Put `Closes #N` in the PR body. Fill in every section of
   the [PR template](.github/pull_request_template.md) ([Evidence](#evidence-every-pr-body)).
6. **Branch from current `main`** in your fork: `feat/<slug>`, `fix/<slug>` or `docs/<slug>`.
7. **Add a log entry**, `docs/log/YYYY-MM-DD-<slug>.md`, in the format of
   [docs/log/README.md](docs/log/README.md). A NO-GO with numbers is a valid result. Write it up.

Stuck, or the spec turns out to be wrong? Say so on the issue, with what you measured. Don't widen the
scope on your own.

## Hard rules

Breaking one of these gets the PR closed.

1. **Never push to `main`.** Every change is a PR from a branch. `main` is protected: a PR plus a
   passing CI check.
2. **Bit identity.** Every optimized path gives exactly the same bits as the plain path on the same
   device. Batched prefill, chunked prefill, paged KV, speculative verify, cache reuse and a request
   batched with others all produce the same logits and KV rows, bit for bit, as single-token decode
   (`forward`). SIMD equals scalar on the CPU. Float reductions run in a fixed order, with no atomics.
   The GPU is not bit-identical to the CPU, and that is expected. The full contract:
   [START-HERE](docs/START-HERE.md#the-design-contract-bit-identity), [determinism.md](docs/determinism.md).
   - That rules out split-K, stream-K, split-KV, f16 accumulation and atomics in reductions, and context
     shift ([roadmap: Not planned](docs/roadmap.md#not-planned-and-why)).
   - A feature that must change bits (a new flag such as `--kv q8` or `--rope-scaling`) changes them only
     when the flag is set. Default output stays byte-identical.
   - Changing the canonical order itself needs a maintainer's agreement on the issue *before* you build it.
3. **The md5 gate against `main`.** Greedy `axon gen` output of the candidate is compared with `main`'s,
   built from the merge base, on the same machine. Every md5 must match, or each difference must be
   intended and explained in the PR ([recipe below](#the-md5-gate)).
4. **No new crates without discussion.** The dependency list is deliberately tiny: `memmap2`,
   `crossterm`, `fancy-regex`, `unicode-width`, `serde_json`. No ggml, BLAS, binding crates or CUDA
   toolkit at build time. CUDA is loaded with `dlopen`, kernels compiled by NVRTC at run time. Want a
   crate? Ask on the issue first and say why the code can't be written in-tree.
5. **rustfmt and clippy clean.** CI runs `cargo fmt --check` and `cargo clippy --all-targets` with
   `RUSTFLAGS="-D warnings"`, on Rust 1.95.0 (MSRV 1.89). A warning fails CI.
6. **No speed claim without numbers.** See [Measuring](#measuring-honestly).
7. **Stay in scope.** Touch the files the issue names. A refactor or a drive-by fix is a separate issue.
8. **Don't edit other PRs' log entries**, except to add a dated "Update" line.
9. **Full evidence in every PR.** The body has all nine sections of [Evidence](#evidence-every-pr-body),
   enough for a maintainer to verify and rerun everything. A PR missing any of them doesn't get reviewed.

## Before you open the PR

Model setup (test models, sha256 sums, NVRTC paths) is in [CONTRIBUTING.md](CONTRIBUTING.md). Don't
skip it: without the models, the tests that matter most are skipped.

```bash
cargo fmt --check
RUSTFLAGS="-D warnings" cargo clippy --all-targets
cargo test                         # debug assertions + overflow checks (profile.test: opt-level 3)
cargo test --release               # the release suite
AXON_SIMD=scalar cargo test        # the scalar path, as CI runs it
```

- No GPU? Set `AXON_SKIP_GPU_TESTS=1` and say so in the PR. The skip is visible in the output.
- No models? `AXON_SKIP_MODEL_TESTS=1` gets CI parity, nothing more. A change to model, kernel or
  serve code needs the model tests run, by you or by a maintainer.
- Changed numerics on purpose? Also run an accuracy A/B (ΔNLL, top-1, KL) against `main`
  ([methodology: Gates](docs/methodology.md#gates)).

### The md5 gate

Build both trees on one machine, run the same greedy prompts through both, compare. A sketch:

```bash
git worktree add ../axon-base "$(git merge-base origin/main HEAD)"
(cd ../axon-base && cargo build --release)
cargo build --release
for m in models/qwen2.5-0.5b-q4_k_m.gguf models/Qwen3-0.6B-Q8_0.gguf models/Llama-3.2-1B-Instruct-Q4_K_M.gguf; do
  for p in "The capital of France is" "Write a haiku about rust." "$(cat long-prompt.txt)"; do
    for b in ../axon-base/target/release/axon target/release/axon; do
      "$b" gen "$m" "$p" -n 64 --device cpu 2>/dev/null | md5sum   # stdout only: speed goes to stderr
    done
  done
done
```

- `long-prompt.txt` is your own ~600-token prompt. Include one, so the batched prefill path is exercised.
- With a GPU, repeat with `--device cuda` and `--device cuda --kv q8`. Each device gets its own set.
  Never compare CPU md5s against GPU md5s.
- If the change touches an architecture (gemma3, MoE, ...), add that model.
- Report it as a table: set, n/N identical, base commit, machine.

## Measuring honestly

From [docs/methodology.md](docs/methodology.md#benchmark-protocol). A perf PR without this is not reviewed.

- **Base vs candidate, same machine, same GGUF files.** Base is `main` at your merge base.
- **Interleaved runs.** Alternate base and candidate, round after round, so drift hits both. Never run
  all of A and then all of B. 3 rounds minimum, 5–10 for small differences. One discarded warm-up.
- **Report medians**, with min–max or the raw values.
- **Cooldown.** GPUs throttle. Let the GPU cool between rounds and keep the machine otherwise idle. A
  number from a hot or shared machine is labelled as one, and only within-run ratios are read from it.
- **Both prefill (pp) and decode (tg)**, every time, so a prefill win can't hide a decode loss
  (`pp560`, `pp2100`, `tg128` at least; add long context if you touched attention or the cache).
- **CPU:** `scripts/bench.py -t <threads> -r 5 base=../axon-base/target/release/axon cand=target/release/axon`
  (`-t` defaults to 18, the dev box; set yours. [BENCH.md](BENCH.md)). **GPU:** `axon gen ... --device cuda` prints prompt and decode tok/s on
  stderr; interleave the runs the same way.
- **Label every number:** MEASURED (with a log), ESTIMATED (arithmetic shown), CODE (file:line) or CLAIMED
  (someone else's number, not reproduced).
- **Name the setup:** CPU, GPU, driver, CUDA, model file, commit, date.
- **State noise.** A ±2 % difference with overlapping ranges is noise, not a win. "Faster than llama.cpp"
  needs llama.cpp in the same run, at its best settings.

## Evidence: every PR body

Required, in the [PR template](.github/pull_request_template.md)'s order. Missing one means no review.
Not applicable? Write "n/a" and say why in one line (a docs-only PR has no bench).

1. **Design notes:** why this approach, how the bits are preserved, alternatives tried and dropped (with
   numbers if measured), links to the issue and spec.
2. **Hardware / environment:** CPU, RAM, GPU + VRAM, driver, CUDA/NVRTC version, OS, thread count and
   pinning, power or thermal state if it matters, and the base and candidate commit SHAs.
3. **Models:** exact GGUF file names, quant types, source (HF repo URL), sha256 of each; the context
   sizes and prompt lengths tested.
4. **Methodology:** what was measured, rounds, warmup, interleaving, cooldown, how the median was taken,
   and the exact commands or scripts, so someone else can rerun it. Attach any script that isn't in the repo.
5. **Benchmarks:** base vs candidate per model, both pp and tg, % delta and noise band. Regressions are
   shown, not left out. llama.cpp or ollama columns where relevant, with their versions and flags.
6. **Correctness:** the md5 sets vs `main` (identical, or an intended diff explained), and every suite run
   (release, debug assertions, scalar, GPU) with its device and pass/fail/ignored counts.
7. **Raw logs:** the raw bench and test output, in a collapsed `<details>` block or a gist. A summary
   table alone isn't evidence.
8. **Risks and what wasn't tested:** archs, platforms, flags or models not covered, known gaps,
   pending gates, follow-ups.
9. **AI disclosure:** which agent and model wrote the code and the body, and whether a human reviewed it.
   This is context for the reviewer, nothing more. Agent-written PRs are welcome.

### Example PR body

Adapted from [#60](https://github.com/HaseebKhalid1507/Axon/pull/60) and its
[log entry](docs/log/2026-10-17-serve-one-cache.md). The numbers are #60's. The commands, the client script and the
`<...>` placeholders show the shape an outside PR should take: a real PR fills them in full.

````markdown
Closes #123

## What changed
`axon serve` with a slot pool no longer keeps a separate KV cache for the request running alone: that path
runs in a view of a pool slot (`CudaQwen2::slot_view`, `BatchLm::slot_cache`), and `plan_slots` fits N
caches instead of N + 1. No flag; no change with one request at a time (CPU, split, pipeline).

## 1. Design notes
- Why: a pool of N slots cost N + 1 KV caches. On a 12 GB card the extra one is a whole slot (7B f16 at
  32k = 1.75 GiB), so the 7B served one request at a time at its trained context. Roadmap Scale item 2.
- Bits: a view is the slot's own device memory with its own scratch and CUDA graphs. Same kernels, same
  rows, so same bits. Tested by `slot_view_matches_a_cache` (picks and KV bytes == a plain cache).
- Rejected: no solo path once a pool exists (1-row batch steps). Measured 88.5 tok/s vs 103.5 main (−14 %,
  7B, `--ctx 8192`, C=1, 3 rounds): it loses the CUDA graph and prompt-lookup drafting.

## 2. Hardware / environment
| | |
|---|---|
| CPU, RAM | Intel Core Ultra 9 275HX (8 P + 16 E, AVX2 + AVX-VNNI), 30 GB |
| GPU, VRAM | RTX 5070 Ti Laptop (sm_120), 12 GB |
| driver, CUDA / NVRTC | 615.71.09, NVRTC 13.4 |
| OS, kernel | CachyOS, Linux 7.2 |
| threads / pinning | defaults, no pinning |
| power / thermal | AC, 140 W GPU cap; cool-gated before each server start; the desktop shares the GPU |
| base commit | `main` @ cf68e9d |
| candidate commit | 2a4eed6 |

## 3. Models
| file | quant | source | sha256 |
|---|---|---|---|
| Qwen2.5-7B-Instruct-Q4_K_M.gguf | Q4_K_M | huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF | `<full hash>` |
| qwen2.5-0.5b-q4_k_m.gguf | Q4_K_M | huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF | 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db |

Contexts: 8192, 32768. Prompts: the md5 set's 3 prompts per model (listed in the raw logs); bench: 256 generated tokens.

## 4. Methodology
C=1 serve decode, client-side streamed tok/s, 256 tokens greedy. 3 rounds; each round runs base then
candidate (order alternates), each a median of 3 requests after 1 discarded warm-up request. Cool-gated
before each server start. Same host, same file.
```bash
for r in 1 2 3; do
  for b in $( [ $((r % 2)) = 1 ] && echo "base cand" || echo "cand base" ); do
    ./$b/axon serve $M7 --device cuda --ctx 32768 --port 8080 & pid=$!
    python3 c1_stream.py --port 8080 -n 256 --requests 3 --warmup 1   # attached below
    kill $pid; wait $pid
  done
done
```

## 5. Benchmarks: base vs candidate
| model | config | metric | base | candidate | Δ % | noise band |
|---|---|---|---|---|---|---|
| 7B Q4_K_M | cuda f16 `--ctx 8192` | C=1 tg256 tok/s | 101.3 | 102.7 | +1.4 | ±2 tok/s per round |
| 7B Q4_K_M | cuda f16 `--ctx 32768` | C=1 tg256 tok/s | 101.4 | 102.8 | +1.4 | ±2 tok/s per round |
| 7B Q4_K_M | cuda f16 `--ctx 32768` | slots | 1 | **2** | | |
| 7B Q4_K_M | cuda `--kv q8 --ctx 32768` | slots | 4 | **5** | | |

Decode within noise: no speed claim. pp: n/a, prefill kernels and their inputs are unchanged; the md5 set covers it.
All MEASURED.

## 6. Correctness
fmt clean · clippy `-D warnings` clean.
| check | where | passed / failed / ignored |
|---|---|---|
| `cargo test --release` | RTX 5070 Ti (sm_120) + RTX 4050 (sm_89) | 471 / 0 / 35 |
| `serve::` tests, debug assertions | sm_120, sm_89 | 71 / 0 / 2 each |
| `scripts/serve_check.py` (serve == gen), 7B: default, `--kv q8`, `--ctx 8192` | sm_120 | ALL OK ×3 |

| md5 set vs `main` cf68e9d (greedy `gen -n 64`) | identical |
|---|---|
| 5 models × 3 prompts, cuda f16 | 15/15 |
| same, cuda `--kv q8` | 15/15 |
| same, cpu | 15/15 |

## 7. Raw logs
<details><summary>bench (3 rounds)</summary>

```text
<every request's tok/s, every round, both binaries, both configs, as the script printed it>
```
</details>
<details><summary>c1_stream.py</summary>

```python
<the full script>
```
</details>

## 8. Risks and what wasn't tested
- Not run: sm_61, sm_80, sm_90; multi-GPU pipeline (no pool there, so no change expected).
- The view's scratch (~0.4 GiB on 7B at 32k) is still separate from the pool's. Follow-up.
- Changes which slot a request alone lands in; tests that assumed one solo cache were updated.

## 9. AI disclosure
Code and body written by <agent> (<model>). A human reviewed the diff before the PR was opened.

## Log entry
`docs/log/2026-10-17-serve-one-cache.md`
````

## GPU testing

CI has no GPU and no models: it skips the GPU and model tests (`AXON_SKIP_GPU_TESTS=1`,
`AXON_SKIP_MODEL_TESTS=1`) and runs the rest, plus the scalar path and a qemu Nehalem run that catches
any AVX instruction leaking into the baseline build. Green CI does not mean the GPU paths work.

- **If you have an NVIDIA GPU:** run the full suite and the GPU md5 sets yourself, and post the results
  with the GPU, driver and CUDA version.
- **Maintainers run the GPU gates** for any PR that touches GPU, kernel, model or serve code, across
  the architectures it could affect (sm_61 to sm_120). A maintainer adds the `run-gpu` label once the diff
  has been read. Outside code never runs on maintainer machines before that, and you can't trigger it.
  Results are posted on the PR.
- A PR is not merged until its GPU gates are green, or until the PR says why they don't apply.

## What reviewers check

1. **Scope:** it does what the issue asked, no more. `Closes #N` is present.
2. **Bits:** the design says *how* the bits are preserved, and the md5 table backs it. This is checked first.
3. **Gates:** fmt, clippy, CPU suites, md5 vs `main`, GPU gates (maintainer), with results, not "should pass".
4. **Evidence:** all nine sections present and consistent. Numbers are base vs candidate, interleaved,
   medians, pp and tg, labelled, with the issue's must-not-regress rows. Raw logs back the tables.
5. **Tests:** new behaviour has a test. Bit-identity claims have a test that compares bits.
6. **Dependencies:** `Cargo.toml` and `Cargo.lock` unchanged unless agreed on the issue.
7. **Log entry** in `docs/log/`, and docs updated if a flag or default changed.
8. **Robustness:** GGUF files and HTTP requests are untrusted input. No panics, no unchecked sizes.

Review findings come back as PR comments. Fix them on the same branch and answer each one. If you
rebase onto a newer `main`, rebuild, rerun the suites and the md5 gate, and say so: a rebase that
wasn't compiled is not clean.

**Merge is maintainer-only.** CI must pass and the gates must be green. Then a maintainer merges.
Don't ask an agent, a bot or a workflow to merge for you.

## Pointers

| need | where |
|---|---|
| models, NVRTC, test environment variables | [CONTRIBUTING.md](CONTRIBUTING.md) |
| source map, reading order | [docs/START-HERE.md](docs/START-HERE.md#source-map) |
| what's on main, what's in flight | [docs/START-HERE.md](docs/START-HERE.md#what-works-on-main), [docs/roadmap.md](docs/roadmap.md) |
| gates, benchmark protocol, hardware | [docs/methodology.md](docs/methodology.md) |
| what is and isn't deterministic | [docs/determinism.md](docs/determinism.md) |
| past rounds, before/after numbers | [docs/history.md](docs/history.md), [docs/log/](docs/log/README.md) |
| flags and environment | [README.md](README.md#usage) |
