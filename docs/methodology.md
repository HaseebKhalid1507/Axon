# How Axon is developed

This page describes the process behind every change on main since 2026-10-13: how work is picked,
built, checked, measured, and then accepted or rejected. What happened, round by round, is in
[history.md](history.md). What was tried and dropped is in
[experiments-rejected.md](experiments-rejected.md). The current numbers are in [results.md](results.md).

Sources for this page: the foreman's running log (`PROGRESS.md` in the foreman's workspace, not in
the repo), `BUILDER-RULES.md`, the PR bodies (#1–#46), and the builders' `NOTES.md` files. Where those
sources describe a rule that changed over time, both versions are given with their dates.

## The loop

```
benchmark ──▶ pick ──▶ design ──▶ build ──▶ critic review ──▶ fixes ──▶ verify ──▶ A/B vs main
    ▲                                                                                  │
    │                         accept: PR, CI, merge ◀──────────────────────────────────┤
    └──────────── next round ◀── reject: logged with numbers, branch kept or deleted ◀─┘
```

1. **Benchmark.** Every round starts from a measured scoreboard: CPU (`scripts/bench.py`,
   `scripts/sweep.py`, and a CPU scoreboard script) or GPU (`scoreboard.py`, outside the repo), against
   ollama and llama.cpp on the same GGUF files. The scoreboard names the gap. Examples: G3's prefill
   ran at 40–50 % of llama.cpp (#10), and in round 9 decode lost at 32k context (r9-pick).
2. **Pick: what to improve.** A strategist agent (Fable) reads the boards, profiles and earlier specs and
   ranks candidate rounds by expected gain, risk to bit identity, size, and which other tracks they
   conflict with. Every claim carries a label (below). Written picks: r1 (forum post, 2026-10-14),
   `r6-pick.md`, `r9-pick.md`, `r10-scope.md` ([research/r10-scope.md](research/r10-scope.md)).
3. **Design: how.** A design agent (Fable) writes a spec with measurements of the current tree, the change,
   its predicted gain, a go/no-go gate, and how the bits are preserved. Examples: `r1-gemm2.md`,
   `r3-long-prefill.md`, `r4-*.md`, `r5-*.md`, `r7-hopper.md`, `r7-multigpu.md`, `r8-unpack.md`,
   `g4-offload.md`. Several rounds set a kill line in advance: r8 "NO-GO if A < +6 %" (spec), the draft
   model "kill if < +15 %" (brief), lcpre "5090 gain < 2 % of the attention kernel" (foreman's rule).
4. **Build.** A builder agent (Opus) works in its own git worktree on its own branch, with a scratch
   directory holding its scripts, binaries, logs and a `NOTES.md` that has a HANDOFF section. Builders
   get cut off (a ~3 h wall clock per run was observed on 2026-10-14), so they write checkpoints
   (`checkpoint-N.patch`) and keep HANDOFF current.
5. **Critic review.** A separate agent (Fable) with a fresh context reads the actual diff and the actual
   numbers. It is read-only. It returns a verdict (SHIP, SHIP-AFTER-FIXES, KEEP-WITH-FIXES, or a
   BLOCKER) and names the biggest remaining gap. Findings go back to the builder. Critics have caught
   real bugs, for example: the PDL race on the PTX fallback (#8), a `--repeat-last-n 0` + drafting panic
   (#17), the sm<80 attention fallback that no test reached (#19), the fused gate/up hijacking Gemma's
   GeGLU (#35), and the foreman's own rebase breaking the build (#13).
6. **Verify.** The foreman runs the gates (below) on a detached checkout of the exact tree. A verify
   was once run in a live worktree while that worktree was being rebased, and conflict markers ended
   up in the test build (2026-10-14 06:22). Since then verifies only run on detached checkouts.
7. **A/B vs main.** The candidate binary and main's binary run interleaved on the same host (protocol
   below). A change that isn't faster, or doesn't buy something else that was asked for, is rejected.
8. **Accept or reject.** Accepted work becomes a PR with an evidence table (gates, job IDs, numbers
   labelled MEASURED or ESTIMATED). CI has to pass, and then the foreman merges it. Rejected work is
   logged with its numbers (in PROGRESS.md, the builder's NOTES.md and, from now on,
   `docs/log/`). The branch is either deleted or kept with the code behind an opt-in flag
   (`feat/r8-gemm-pipeline`, `AXON_GEMM2M`, for example).
9. **Kill rules bound spend; the exactness contract bounds merge** (ruling 2026-10-26, PR #151). A
   pre-registered kill rule decides whether a line gets more builder-days and pod-hours. It is not a
   veto on code that was built within the spend the rule allowed, passes the exactness contract
   (md5 + bits gates), still passes step 7 on the term it targets (it is faster there), and regresses
   no measured cell beyond the pre-registered noise bar (2 % unless the PR set another). Such code can
   merge default-on with the miss written into the PR, the log and the roadmap, and the line closes
   (no follow-ups without a fresh pick). So a missed kill rule lowers the bar to step 7's, not to
   nothing. Code that regresses any cell past the noise bar is still rejected (M6, #148: ×0.725–×0.965
   against a 2 % bar). A change that wins some measured cells and loses others lands opt-in (#131 (c)),
   or default-on only where it wins, with its dispatch restricted to those cells. Work built after the
   rule said stop does not count as "already built": it is a foreman override. Two corollaries from the
   same case: gate a rule on the term the change touches (the decode step), not on an aggregate
   dominated by another term (the mix C = 128 cell was prefill-bound); and booking hardware past a
   missed proxy gate is an override the foreman owns and records in the PR body and `docs/log/`
   (#151's H100 session after the 4090 miss, ~$0.93).

**Who decides.** Fable (`fable-5-1`) does picks, designs, critiques and research. Opus (`opus-5-5`)
builders write the code. The foreman (Jawz) assigns tracks, runs gates and benchmarks, integrates
stacked branches and decides. Merge authority changed over time:

- 2026-10-13: every merge (#1–#9) happened on Haseeb's explicit word ("merge it", "merge 4", "merge them if u deem them fit").
- 2026-10-14 ~05:50: Haseeb granted the foreman merge rights to main for Axon. The bar: CI green, the
  foreman's verify, a clean critic review, and a post-rebase compile, suite and hashes.
- 2026-10-16 ~03:16: a standing grant for merges and `--admin` was recorded, and main was protected
  (PR plus a passing check).
- Docs and roadmap PRs that set direction (#44, for example) still say "don't merge without Haseeb's read".

## The bit-identity contract, and why it shapes every decision

The contract in its current form (docs/START-HERE.md, BUILDER-RULES.md):

- **decode == batched prefill == speculative verify**, bit for bit, at any batch size and position;
  **cache reuse == fresh prefill**. On the GPU, a request batched with others == the same request alone
  (#38).
- **Deterministic:** fixed-order float reductions, no atomics in reductions. The same input gives the
  same bytes on every run.
- **SIMD == scalar** on the CPU. One hardware property changes CPU bits: whether the CPU has FMA (#2).
- **GPU ≠ CPU** bit for bit. The two are graded against each other (top-1 agreement, KL) with bounds
  calibrated against the CPU's own spread.
- If a feature has to change bits (a RoPE scaling flag, `--kv q8`, for example), the bits change only
  when the flag is set, and default outputs stay byte-identical.

**How it got here.** Until #2 the CPU rule was stricter: no change could alter logits at all.
Speedups that changed numerics were written up as proposals, and that is why PR #2 (r8 interleaved
kernels, FMA attention, f16 KV) was a separate draft "for Haseeb's call". He merged it on
2026-10-13. Since then the CPU contract is "deterministic per build and CPU class", and
`AXON_REPACK=0` and `AXON_KV=f32` remain as opt-outs. On the GPU, G3 (#10) set batched == decode.
Then gemm2 (#12) and the tensor-core attention (#19) each **redefined the canonical order once** (GPU
md5s changed and the new references were committed). After each redefinition, decode and batched
still matched each other bit for bit.

**Why it matters for decisions:**

- It makes lossless speculative decoding, exact chat/serve cache reuse, and serve batching that equals
  solo true by construction instead of by tolerance.
- It makes gating cheap and strict: a kernel change is checked with md5s of generated text, not
  tolerances.
- It rules out whole classes of speedups. Examples: split-K GEMV, stream-K GEMM, split-KV or f16
  accumulation in prefill attention, a llama.cpp-style inexact epilogue (#10: only 10–19 % faster on the
  same tiles), and on night 1, FMA in CPU attention (until #2). A design has to say how it preserves
  the bits, and the critic checks that claim first.
- research-determinism.md estimates what the contract costs on the GPU and finds it close to free:
  decode GEMVs at 93 % of the DRAM ceiling, and prefill GEMM at llama.cpp MMQ parity. On the CPU the cost
  is measured: bit-identical r8-bi reached 1.5B pp560 458–574 tok/s, against 734 for the numerics r8
  (PROGRESS, night 1).

## Gates

Before a PR, on the exact tree that will merge:

| gate | what | since |
|---|---|---|
| release suite | full `--test-threads 2` run, with GPU tests on a GPU host and `AXON_BIG_MODELS=1` where 7B+ tests apply | night 1 |
| debug-assertion suite | same suite built with `-C debug-assertions -C overflow-checks` | 2026-10-13 03:26, after CI caught an i8 overflow the release-only verify missed (aa9be77) |
| md5 of `gen` outputs | main vs candidate, greedy, on the **same host** (and driver) | night 1 with 4 prompts. Grew to 11/17/42-gen sets across models, CPU+CUDA, short and ~610-token prompts, pool settings (#22, #24, #26, #30). Same-host rule written into BUILDER-RULES (`axon-run --same-host`) on 2026-10-16 |
| clippy | `-D warnings`, bin and tests | night 1 |
| rustfmt | `--check` | night 1 |
| CI | fmt, clippy, build, tests (model and GPU tests skipped), the scalar path, and the test binary under a qemu Nehalem CPU (no AVX may leak into the baseline build) | qemu step from the portability change (2bd8606, in #1) |

Extra gates by kind of change:

- **Numerics-changing work:** teacher-forced accuracy A/B (ΔNLL with clustered SE, top-1, KL) against the
  parent and against the CPU-vs-CPU noise floor (#2, #11, #19, #35).
- **Cross-arch:** suites on sm_120 (bella), sm_89 (avante, 4090 pods), sm_61 (jade), sm_80 (A100), sm_90
  (H100) where the change could differ by arch (#31, #33, #39).
- **Post-rebase:** after any rebase, compile and rerun before calling it clean. The foreman once reported
  a rebase as clean without compiling it, and it didn't build (#13, 2026-10-14 03:2x).
- **Integration chains:** stacked branches were integrated as tagged stages (int-1 … int-4, int2-1 …,
  night3 stages), with suites and the md5 gate at every tag (#11, #15–#17, #22).

Not every PR met every gate before merging. Where a gate was still pending at merge time, the PR body
or the foreman's log says so: #32 (bella CPU timing A/B of the exact tree), #34 (sm_120 one-device
pipeline gate, green after merge), #36 (merged ahead of its sm_89 run), #37 (merged on CPU gates, GPU
checks green after merge), #42 (the 12 GB fit check on bella). [history.md](history.md) notes
each one.

## Benchmark protocol

- **Interleaved rounds, medians.** Engines or binaries run in turn, round after round (`bench.py`; GPU
  scoreboards rotate the order), and the median is reported, usually with min–max or the raw values.
  3 rounds for boards, 5–10 for small A/Bs; one 8-round A/B decided the drafting default.
- **Same host, same files.** Base and candidate on one host; every engine on the identical GGUF (ollama
  models made with `ollama create` from the same file; pod files sha256-checked against bella's).
- **Salted prompts** so no engine gets a prompt-cache hit. ollama is unloaded after its turn (it once
  held VRAM and made llama-bench OOM).
- **Cool-gating (GPU).** The laptop GPU throttles. The rules over time:
  - Night 2: wait for ≤ 62/65 °C, 150–240 s timeouts. The desktop keeps the idle GPU at ~70 °C, so every
    gate burned its full timeout while holding the lock.
  - 2026-10-15 02:4x: ≤ 75 °C or idle + 5, 60 s cap, plus waiting up to 20 s for the GPU to be ≤ 10 % busy
    (the desktop shares it).
  - Round-5 protocol for small A/Bs: each model-round starts at ≤ 60 °C, after a `cuda_sustain` bandwidth
    gate (p50 ≥ 640 GB/s).
  - axon-run admits bella GPU jobs only at ≤ 80 °C.
  Absolute numbers from a hot or shared machine are labelled as such, and only within-round ratios are
  read (#12, #15, #20).
- **Each engine at its best.** On the hybrid CPU, llama.cpp master collapses at 18 threads, so it is
  compared at its best thread count (`sweep.py`) before any claim is made against it (2026-10-13).
- **Labels.** Every number is MEASURED (with a log or job ID), ESTIMATED (arithmetic shown), CODE (read in the
  tree, file:line) or CLAIMED (a paper's number, not reproduced).
- **Both pp and tg.** Every board and A/B reports prompt processing *and* decode, so a prefill win can't
  hide a decode loss. The cases where this mattered: r8bi (pp +13 %, tg128 −3 %), the r4 decode track
  inside night 3 (pp +8.6 %, tg −14.5 %), chunked prefill (ITL better, TTFT +21–83 %).
- **No "ahead" without numbers.** A claim names the engine, model, metric, host, commit and date.
  Known noise is stated: the 1.5B/3B GPU decode on bella is bistable from process to process (1.5B
  main ranged 343–413 tok/s, #30), and the laptop GPU sits at its 140 W power cap (sustained 7B decode
  ~9 % below burst, r5-decode).

## Hardware

| host | CPU | GPU | role / caveats |
|---|---|---|---|
| **bella** (named "Parabellum" in BENCH.md and START-HERE) | Core Ultra 9 275HX, 8 P + 16 E cores, no SMT, AVX2 + AVX-VNNI, 30 GB RAM | RTX 5070 Ti Laptop, sm_120, 46 SMs, 12 GB, ~648 GB/s measured (672 spec), 140 W cap | Haseeb's own machine and the main dev box. He games on it: `axon-run hold bella` keeps jobs off. Desktop shares the GPU. Builds run at `nice 19` on cores 22–23 |
| **avante** | i5-13420H, 4 P (SMT) + 4 E, 15 GB | RTX 4050 Laptop, sm_89, 6 GB | small models only; VRAM contention caused false test failures (fixed by #40). Lost power once mid-suite (battery, 2026-10-15 03:38) |
| **jade** ("Jaded") | i7-9750H, 6c/12t, AVX2+FMA, no VNNI, 7.5 GB | GTX 1050, sm_61, 3 GB | homelab. Runs under a memory cap and nice/ionice, with an HDD under IO load; the old-GPU and AVX2-only test host |
| **RunPod pods** | varies (EPYC, Ryzen 9950X, Xeon) | RTX 4090, RTX 5090, A100 SXM4 80 GB (sm_80), H100 SXM and PCIe (sm_90), 2×4090 (no P2P) | correctness suites, same-host md5 gates, and scoreboards. Budget-capped (below) |

## Orchestration tools (outside the repo)

- **`axon-run`** (live 2026-10-16 ~03:30): one queue for bella, avante, jade and every live pod. FIFO
  with a fair share per worktree. A job goes to the eligible host with the earliest estimated finish.
  `--timing CLASS` gives an exclusive run, `--same-host JOB` puts base and candidate on one host,
  `--shard N` splits a suite into jobs that must partition the full `--list`, and `hold bella` keeps jobs
  off bella. It can also autoscale 4090 pods, inside the budget guard. Before it there was **`axon-q`**: a
  FIFO ticket queue with `gpu` and `cpu` lanes, later a shared `test` lane with VRAM reservations, a shadow
  daemon and a phase daemon. The exclusive locks of that design kept the GPU at ~25 % busy on 2026-10-15.
- **`axon-pods`**: creates, sets up and terminates RunPod pods. A budget guard (a daemon with a 60 s tick
  and a ledger) refuses pods past the cap and kills everything at the cap or the deadline. A watchdog on
  each pod terminates it at its booked end or after 10 min idle. Budget: $20 for 2026-10-15 23:20 to
  10-16 23:20 (guard at $18), raised by $20 on 10-16 14:35 (guard at $38).
- **Heartbeat and watchdogs**: `axon-heartbeat` (a systemd user service) sends the foreman a status digest
  every 5 min, with pod and host usage flags. `axon-q-watchdog` reports queue stalls, hung lock holders,
  VRAM hogs and GPU ≥ 90 °C. Small "verify-watch" or "bench-waiter" agents block on a job and wake the foreman.
  These exist because of one failure: on 2026-10-13 the foreground loop sat dead from ~03:45 to ~09:55,
  because nothing woke it after a tool-call cap (about 6 h lost).
- **Coordination**: the project forum (claims on shared hot files, findings, critic reviews) and
  `BUILDER-RULES.md` (no cargo, only rustc build scripts on shared rlibs; nothing big in `/tmp`, which is
  RAM; every model run through `axon-run`). The `/tmp` rule comes from two OOM kills of Haseeb's desktop
  session on 2026-10-13, caused by tmpfs growth plus parallel model loads.

## How research feeds the roadmap

- **Literature.** Research agents read papers (via OpenAlex/arXiv) and map them onto Axon's code and
  contract. Each proposal says its change, gain, bit-identity risk, size and a 1-hour experiment that
  settles it. Night 1: `research-papers.md`, which put the r8 interleave (Georganas
  arXiv:2508.06753, ik_llama R8) and P/E-aware pinning (arXiv:2411.19542) on the queue. Also `research-r8-bitident.md` (bit-identical r8-bi is feasible,
  ~1.2× estimated) and `research-verify-cost.md` (why small verify batches cost 1.5–3× a forward).
  Round 10 (2026-10-16): six reports, now in [research/](research/README.md).
- **Competitor source reading.** These numbers and mechanisms come from reading llama.cpp, ollama and vLLM
  code, not from their docs: llama.cpp's CUDA op offload for batch ≥ 32 (`GGML_OP_OFFLOAD_MIN_BATCH`, which
  turned G4 "plan A" into "what llama.cpp already does" and led to #16), its MMQ kernels (the prefill
  parity target, r4-prefill-big), its SPM tokenizer (ported in #32), its server's `cache_prompt` and
  chunked prefill (r9, research-sched), and its QWEN3 and gemma3 graphs (oracles for #18 and #35). Also
  vLLM's batch-invariance mode (research-determinism), and AirLLM. AirLLM gave nothing for speed;
  research-airllm.md and a Fable review turned "stream every layer" into "stream once per prompt, never
  for decode".
- **Oracles.** Independent references are kept in `tests/fixtures/`: gguf-py dequantization, the Hugging
  Face tokenizer, and ollama's greedy path. Bigger checks use f64 reference forwards (Qwen3 in #18, Llama 1B in #39,
  Gemma 3 in #35), llama.cpp greedy runs, and template rendering by transformers.
- The roadmap ([roadmap.md](roadmap.md)) is ordered from these picks. Its Scale section cites
  `r10-scope.md`.

## Writing it down

Every PR adds an engineering-log entry, and so do NO-GO experiments: see [log/README.md](log/README.md).
