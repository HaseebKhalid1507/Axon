# Roadmap

Set 2026-11-12 at main `213465a` (code at `3014eeb`, paused since 11-06), from Haseeb's answers to the 11-07 outside
review (Fable 5.2) and its revised plan, approved 11-12 with the reviewer's changes. At most 150 lines, refreshed at
release tags ([Process v2](methodology.md#process-v2-2026-11-12)). Landed: [history.md](history.md); lost:
[experiments-rejected.md](experiments-rejected.md); numbers: [results.md](results.md); the plan before: [roadmap.md at `213465a`][old].

[old]: https://github.com/HaseebKhalid1507/Axon/blob/213465a12e8402041e39b2d0a565de192f6c38c1/docs/roadmap.md

## Mission

**Axon is for everyone: the standard for running local AI**, as Synaps should be the standard agent runtime.
Haseeb, 11-12: "its for everyone. I want it to be the standard for running local AI. kinda like how synaps should be
the standard agent runtime. so its for the masses." The means stays what it was: **make inference cheap and
fast.** The bit-identity contract stays the floor under every change ([determinism.md](determinism.md)).

## Current target

For the four weeks from 2026-11-12: **the best local backend for coding agents (opencode, codex, pi) on NVIDIA PCs,
Linux then Windows, installed with one command on a machine with only the GPU driver.**

Why this target (MEASURED unless marked):
- Agent traffic is Axon's one clear edge, on one replay: bella, Qwen2.5-7B f16, against llama-server: turn TTFT p50
  0.122 vs 1.06 s, ITL p99 119 vs 611 ms ([log](log/2026-10-27-agent-replay.md); the trace scaled to 0.15× and clipped
  at 32k, llama-server set up by Axon's side, and it re-prefilled less: 1.9 vs 10.3 %). With the RAM tier an evicted
  25k session resumes in 0.141 s vs a 2.713 s re-prefill (#201, RTX 4090); shared pages: eight subagents' TTFT 0.591× (#211).
- Single stream is parity, not a lead: 7B decode 186 vs 181 tok/s and prefill 1.13× llama.cpp on a 4090 (10-18,
  [results.md](results.md)). No single-stream board has run since `7f42c02`.
- Between that and a user: no release binary; the GPU path needs NVRTC from a CUDA toolkit
  ([models.md](models.md#nvidia-gpus)); gpt-oss-20b prefills at 0.27× llama.cpp (265 vs 995 tok/s, bella, #98); no
  constrained JSON (`response_format` is a 400, #116); paged KV, the RAM tier and sharing are off for MoE
  ([serving.md](serving.md)).

## Shelved for now

Shelved, not deleted: the code on main stays, tested; each can return once the target is met or hardware exists.
Macs wait for a Mac (none for now, Haseeb 11-12). AMD stays out of scope. So does more safetensors work.

| shelved | why, in one line | evidence |
|---|---|---|
| Datacenter tier, vLLM-class serving (`--datacenter`, multi-tenant, H100 boards) | 1.8–3.8× behind vLLM / SGLang at C = 128 on an H100; closing it needs Hopper kernels, FP8 and tensor parallel on hardware not at hand | [results](results.md#serving-runpod-h100-sxm-main-c15c90e-2026-10-19), [log](log/2026-10-19-h100-showdown.md) |
| H100 / Hopper exact GEMM | the upper-bound probe reached 224.8 TMAC/s against its 240 bar: shelved 10-26 by its own rule | [rejected](experiments-rejected.md#gpu-prefill-gemm), [log](log/2026-10-26-w4a8-probe.md) |
| FP8 fast mode; **#194 closed** | on consumer cards it loses to Q4_K_M: 7B decode 32 tok/s against a 47–55 bar, top-1 95.9 % against 97 % (bella); it would pay only on an H100. FP8 checkpoints still load, exact as Q8_0 | [rejected](experiments-rejected.md#gpu-decode), [log](log/2026-10-30-fp8-weights.md) |
| DeepSeek V3 / Kimi K2 scale | Kimi K2 has 1T parameters, ≈ 600 GB at Q4_K_M (ESTIMATED): multi-GPU pods, not a PC. `deepseek2` stays on main | [old plan][old-frontier] |
| Gemma 3 prefill | 0.49× llama.cpp at pp2100 (4B 8,782 vs 17,920 tok/s, 4090), yet that reads a 2,100-token prompt in 0.24 s (ESTIMATED); gpt-oss's 0.27× is the loss users feel | [results](results.md#known-losses-and-gaps-as-of-3014eeb) |
| Multi-GPU (pipeline stage 1b, tensor parallel) | the target is one consumer GPU; pipeline parallel stage 1 stays on main (#34) | [design](design/multi-gpu.md) |
| NEON kernels, the Metal design | no Mac to build or measure on; arm64 CI (`ubuntu-24.04-arm`, `macos-15`) will test the scalar build instead | [not planned](#not-planned-and-why) |
| Synaps on Axon | waits for a local Synaps profile with a small context budget (Haseeb, 11-12: "maybe later") | [design](design/agent-workload.md) |

[old-frontier]: https://github.com/HaseebKhalid1507/Axon/blob/213465a12e8402041e39b2d0a565de192f6c38c1/docs/roadmap.md#frontier-models-100b-to-1t

## Plan

### Week 1 (11-12 to 11-18)

1. **Process v2** ([methodology](methodology.md#process-v2-2026-11-12)). Its jobs and safeguards (builder: case):
   rules 2, 3 and 5 first, then rule 8's CI model tests and fuzzers (Windows and arm64 CI come with Distribution).
2. **Decisions** (a flip's A/B cells: Process v2 rule 4):
   - Shared prefix pages go default-on together with `--cache-ram auto`, once #218 is fixed and a soak of 4 h or more
     passes with sharing on under a small pool, plus one sharing-on, tier-off cell (the tier is off under a cgroup).
     `AXON_SHARE_KV=0` stays the kill switch. #210 and #211 passed at #216's code (jobs 1106-170837-edc9,
     1106-173522-297a); the flip's PR says the board trace shares only 4–7 %: the win is subagent fan-out.
   - `--cache-ram auto` goes default-on: the smaller of 4 GiB and ¼ of the RAM available after the model loads
     (`mem_available()`, already cgroup-capped); off when a cgroup limit is the tighter bound, and the banner says so.
     An explicit `--cache-ram N` is unchanged; `0` is the kill switch. The 0.3 s resume clause is waived (17–24× on
     bella). Today `auto` is half the RAM left after the model, up to 8 GiB, page-locked from start.
   - FP8 fast mode is parked (#194 closed 11-12); #222 (`--help` is stale) is fixed.
3. **A GPU start with only the driver:** embed the kernels as PTX, which the driver compiles itself, so no NVRTC is
   needed. Main already sends PTX to a GPU newer than its NVRTC ([models.md](models.md#nvidia-gpus)).
4. **Compatibility audit:** the 30 most-pulled ollama models, each: runs, refused, or prompt differs from llama.cpp's
   (`scripts/zoo.py`; last sweep 18 small files, [hardening.md](hardening.md#model-zoo-results-2026-10-18-measured)). It orders weeks 2–3.
5. **Harness acceptance test:** opencode, codex and pi on fixed tasks, scored on completion.
   `scripts/examples/harness-repo/make_repo.py` already builds one such task.
6. **`axon pull`:** Hugging Face files plus about ten checksummed aliases; no registry. HTTPS means `curl` or a TLS
   crate: none of the five in `Cargo.toml` speaks TLS (a dependency decision, [AGENTS.md](../AGENTS.md) hard rule 4).
7. **v0.1.0, Linux only:** the tag, the boards re-run on its commit, a Linux binary. It waits on item 3 (else the
   binary is a CPU engine), not on the two flips. The agent board's llama-server is tuned by someone arguing its side;
   stop rule: within 1.5× on turn TTFT, the edge is thinner than claimed and MoE in the pool goes first; tag either way.
   The release notes ([release-0.1.md](release-0.1.md)) say what waits: Windows, vision, embeddings, auto-update, signing.

### Weeks 2–3, and whatever the audit ranks

- **gpt-oss experts on the GPU** (prefill 0.27× today). Bar: 0.9× llama.cpp or better on a 4090, and at least 3×
  today's 265 tok/s on bella (the 12–16 GB PC); under 0.8× after 4 days, prefer Qwen3-30B-A3B.
- **Constrained JSON and tool calls** (`response_format`, #116): a sampler mask, so the bits are unaffected. Defer
  if the harness test shows under 1 % malformed calls.
- **MoE into the paged KV, the RAM tier and sharing** (all three are off for MoE today). Bar: an evicted 25k-token
  MoE session resumes in under 0.5 s on a 4090, md5 equal to solo; no page image for gpt-oss in 3 days: qwen3moe only.

### Weeks 3–4

- **Windows port (CUDA + CPU), v0.2.** Every kernel is reused; the port is host shims: `dlopen` and library names,
  libc declarations, the `/proc`, cgroup and `/sys` readers. Gate: md5 on Windows equals md5 on Linux on the same GPU.
  The CPU build runs on GitHub's `windows-latest`; booting bella into Windows stops the Linux queue, so batch it.

### Then

- **An ollama drop-in:** model swapping and `/api/pull` (today the server runs one model and refuses `/api/pull`,
  `src/serve/ollama.rs`). Vision and embeddings, later.

## Tracks

At most three at once (Process v2 rule 9), each tied to its metric; week 1's decisions and audit serve all three.

| track | holds | the metric it moves |
|---|---|---|
| **Distribution** | driver-only GPU start, `axon pull`, release binaries, the clean-machine install test, Windows, arm64 CI | one command to a first token on a machine with only the GPU driver: Linux at v0.1.0, Windows at v0.2 (pass / fail per release) |
| **gpt-oss / MoE** | gpt-oss experts on the GPU; MoE in the paged KV, the RAM tier and sharing | gpt-oss-20b prefill and decode against llama.cpp on the same card (prefill 0.27× today, bella, #98) |
| **Agent API** | constrained JSON and tool calls, the harness acceptance test | harness tasks completed by opencode, codex and pi; malformed tool calls (`axon_tool_calls_dropped_total`) |

## Open follow-ups

The live list is `gh issue list` ([issues](https://github.com/HaseebKhalid1507/Axon/issues)). At 11-12:
- **Week 1:** #222 (`--help`), #218 (blocks the sharing flip), #219 (the shared-tokens counter), #210, #211 (close with the flip).
- **Agent API:** #116 (`tool_choice: required` with several tools on gpt-oss).
- **gpt-oss / MoE:** #121 (the MoE-resident twins' hardening).
- **No track:** #188 (shared pages C4: prefill once, join many), #189 (the replay harness's provenance), #217, #220,
  #184 (RAM checks: kills still land within ~48 MiB above a refusal edge), #115, #138, #110 (gemma3 CPU drift).
- **Shelved by this direction:** #194 (closed), #174 and draft #176 (the H100 fast-GEMM probe), #165 (its rules).
- **Tooling** (gates, pods, queue): [axon-devstack's issues](https://github.com/HaseebKhalid1507/axon-devstack/issues).
- **Unscheduled ideas,** among others: batched speculative decoding v2 ([design](research/spec-gate-v2-20261024.md)),
  agent priority classes, GPU-error isolation and fault injection (one CUDA error kills every stream), the CPU
  re-board, Gemma 3 27B on CUDA, a disk KV tier ([roadmap.md at `213465a`][old]). The audit can rank them in.

## Not planned, and why

- **Non-fixed-order split-K / stream-K, split-KV, f16 accumulation, atomics in reductions:** not in exact mode: each
  breaks decode == prefill == verify ([determinism.md](determinism.md)). Fast mode may ([ADR](design/determinism-modes.md) §4.6); it's shelved with FP8.
- **Context shift** (llama.cpp's K-shift on overflow): the cache would no longer be what a fresh prefill writes.
  Harnesses compact their own history on a 400; save/restore is the exact alternative.
- **Lossy approximations sold as the model** (expert deferral, pre-gated routing, key skipping), in either mode.
- **More dense single-stream kernel rounds at 8B and under:** on bella, 7B decode GEMVs run at 93 % of the measured
  648 GB/s and 7B prefill is at parity with llama.cpp's MMQ kernels ([old roadmap][old]).
- **q4 KV cache:** paged KV solved its one 12 GB case at no quality cost. Revisit only as an experiment with a KL gate.
- **Platforms Axon can't measure on** (Metal, NEON speed work, AVX-512, ROCm) until a machine exists.

## Old section links

Links into the roadmap before 11-12 land here, and go on to that section at `213465a`:
- <a id="state-at-the-11-06-pause"></a>[State at the 11-06 pause][old-pause]: its four decisions are week 1 of the [plan](#plan).
- <a id="mission-and-positioning"></a>[Mission and positioning][old-mission]: now [Mission](#mission) and [Current target](#current-target).
- <a id="next-proposed-order-outside-review-10-19-owner-decides"></a>[Next: proposed order (10-19)][old-order]: its datacenter, Hopper and FP8 items are [shelved](#shelved-for-now); 2A–2C are built, 5 waits for the tag.
- <a id="agent-harness-workloads"></a>[Agent-harness workloads][old-agent]: the agent API [track](#tracks) and its open follow-ups.
- <a id="mixture-of-experts"></a>[Mixture-of-experts][old-moe]: the gpt-oss / MoE [track](#tracks).

[old-pause]: https://github.com/HaseebKhalid1507/Axon/blob/213465a12e8402041e39b2d0a565de192f6c38c1/docs/roadmap.md#state-at-the-11-06-pause
[old-mission]: https://github.com/HaseebKhalid1507/Axon/blob/213465a12e8402041e39b2d0a565de192f6c38c1/docs/roadmap.md#mission-and-positioning
[old-order]: https://github.com/HaseebKhalid1507/Axon/blob/213465a12e8402041e39b2d0a565de192f6c38c1/docs/roadmap.md#next-proposed-order-outside-review-10-19-owner-decides
[old-agent]: https://github.com/HaseebKhalid1507/Axon/blob/213465a12e8402041e39b2d0a565de192f6c38c1/docs/roadmap.md#agent-harness-workloads
[old-moe]: https://github.com/HaseebKhalid1507/Axon/blob/213465a12e8402041e39b2d0a565de192f6c38c1/docs/roadmap.md#mixture-of-experts
