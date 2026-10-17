<!--
Read AGENTS.md first. Every section below is REQUIRED: a PR missing one is not reviewed.
If a section doesn't apply, write "n/a" and one line saying why (e.g. docs-only: no bench).
Example of a complete body: AGENTS.md, "Example PR body".
-->

Closes #

## What changed

<!-- Files and behaviour in a few lines. New flags and defaults. -->

## 1. Design notes

<!-- Why this approach. How the bits are preserved, or which flag changes them.
     Alternatives tried and dropped, with numbers if measured. Links: issue, spec, roadmap item. -->

## 2. Hardware / environment

| | |
|---|---|
| CPU, RAM | |
| GPU, VRAM | <!-- or "none" --> |
| driver, CUDA / NVRTC | |
| OS, kernel | |
| threads / pinning | |
| power / thermal | <!-- AC or battery, power cap, idle temp, shared desktop? --> |
| base commit | `main` @ |
| candidate commit | |

## 3. Models

| file | quant | source (HF repo) | sha256 |
|---|---|---|---|
| | | | |

Contexts tested: · Prompt lengths:

## 4. Methodology

<!-- What was measured; rounds; warmup; interleaving order; cooldown rule; how the median was taken.
     The exact commands or scripts, copy-pasteable. Scripts not in the repo: attach them under Raw logs. -->

```bash

```

## 5. Benchmarks: base vs candidate

<!-- Per model, both pp and tg. Medians, with the noise band (min–max or round spread). Show regressions.
     llama.cpp / ollama columns where relevant: their versions and flags. Label numbers MEASURED / ESTIMATED. -->

| model | device | metric | base | candidate | Δ % | noise band |
|---|---|---|---|---|---|---|
| | | pp | | | | |
| | | tg | | | | |

## 6. Correctness

- [ ] `cargo fmt --check` · [ ] `RUSTFLAGS="-D warnings" cargo clippy --all-targets`

| check | where (device / arch) | passed / failed / ignored |
|---|---|---|
| `cargo test --release` (release suite) | | |
| `cargo test` (debug assertions + overflow checks) | | |
| `AXON_SIMD=scalar cargo test` | | |
| GPU suite | <!-- yours, or "maintainer (run-gpu)" --> | |

| md5 set vs `main` (models × prompts × device, `-n`) | identical |
|---|---|
| | n/N |

<!-- Any md5 that differs: which, and why it is intended. -->

## 7. Raw logs

<details><summary>bench output</summary>

```text

```
</details>

<details><summary>test output (summary lines + any failure)</summary>

```text

```
</details>

<!-- Or a gist link per log. Summaries alone are not enough. -->

## 8. Risks and what wasn't tested

<!-- Archs, platforms, flags, models not covered. Known gaps. Gates still pending. Follow-ups. -->

## 9. AI disclosure

<!-- Who or what wrote the code and this body (agent + model, or "human"), and whether a human reviewed it.
     e.g. "Written by <agent> (<model>); reviewed by <name>" or "Written by <name>, no AI tools". -->

## Log entry

`docs/log/YYYY-MM-DD-<slug>.md`
