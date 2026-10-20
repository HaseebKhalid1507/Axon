# Outside review: the big picture (2026-10-19 01:14 EDT)

> **Outside review, curated for the repo.** Written by Fable, an outside consultant model, read-only, at main
> `05fb70c` plus the PRs open then. The opinions are the reviewer's; decisions are Haseeb's. This is the review the
> determinism-modes ADR answers ([../design/determinism-modes.md](../design/determinism-modes.md) cites it as
> `fable-bigpicture-20261019` by section: §1, §2, §5 risk 1, §5 risk 4, §7, §8 recommendation 3; the section numbers
> here are the review's own). Labels and citations are the reviewer's. Curation: details about the owner and the
> owner's machines are removed or generalised (marked *[edited]*); the rest is the review's text. Many statements
> describe main as of 05fb70c and were overtaken the same day (#71, #84, #89, #98 and #102 landed; the H100 serving
> board ran): see the follow-up, [2026-10-19-outside-review-next-steps.md](2026-10-19-outside-review-next-steps.md).

Read-only. Tree: `origin/main` `05fb70c` plus open PRs #71 #84 #89 #90 #92 #94 #97 #98. Sources: repo docs, `docs/log/`,
the overnight-workspace specs (r15–r17), `sb-4090-ab-20261018.md`, the 2026-10-13 review set, and `axon-devstack`.
Labels: **MEASURED** (source cited), **CODE** (file:line), **OPINION**. Where I do arithmetic on measured numbers I
say **ESTIMATED**.

The frame for everything below: the mission is *"cheap and fast" [edited: the owner's wording, as in the
roadmap], "run the best models", "squeeze every drop at datacenters."* I score against that, not against "impressive for 9 days" (it is).

---

## 1. What's actually built

**Size and shape.** `git ls-files | xargs wc -l` = 109.6k lines: 80.2k Rust under `src/` (21.0k of it in `*tests*.rs`),
7.2k CUDA C++ (84 `__global__` kernels across 8 `.cu` files), 8.4k docs, 5.3k Python scripts (CODE). 618 `#[test]`
functions, 42 `#[ignore]`d. 5 direct crate dependencies (`Cargo.toml`). 668 occurrences of `unsafe`. 458 commits,
88 merged PRs, 7 open, 10-10 → 10-19; 137 commits landed on 10-18 alone (`git log`). Every commit has one human
author *[edited]*; agent provenance is not in git.

| area | state | evidence | thin? |
|---|---|---|---|
| Architectures | `qwen2`, `qwen3`, `llama` (3.x, Mistral/Llama2 SPM), `gemma3` 1B/4B (27B refused), `qwen3moe` (CPU, or GPU-attn+CPU-experts), `deepseek2` (CPU only on main; GPU MLA in #89), `gpt-oss` (CPU only on main; #98 for GPU attn) | README "Status"; `src/model/`, `src/moe.rs` | MoE/deepseek2/gpt-oss have **never run a production-size model**: all gates are on Moonlight-16B-A3B, a "Pruned-2B" and "4x0.6B" dev models (`docs/results.md` "New models and types") |
| Weight formats | GGUF only. CPU: F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/MXFP4. GPU: the eleven quantized types; F16/F32 matrices refused | README | **No FP8, FP4/NVFP4, AWQ, GPTQ, safetensors.** No i-quants other than IQ4_NL |
| CPU path | AVX2 / AVX-VNNI, runtime multiversioned, spinning pool, 8-row repacks | `src/quant/r8*.rs`, `src/mv.rs`, `src/pool.rs` | No AVX-512, no AMX, no NEON ("Not planned") |
| GPU path | NVIDIA only, Linux only. NVRTC compiles at first run, dlopen'd driver API, CUDA graphs for decode, int8 `mma.sync` GEMMs, wgmma path on sm_90a (opt-in `AXON_GEMMH`) | `src/cuda/mod.rs:1699` `compile`, `src/cuda/hopper.cu` | sm_80+ for tensor cores; sm_61 "correct, not fast" |
| Serve / API | Hand-written HTTP/1.1, thread-per-connection, one engine thread. `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, ollama API, tools, `/metrics`, `/slots`, `--api-key` | `src/serve.rs:1-33`, `src/serve/http.rs` | No TLS/HTTP2, no `logprobs`, no `n>1`, no JSON mode/grammar, no embeddings, no `logit_bias` (all 400) |
| Batching | Continuous batching on one GPU, `--parallel` ≤ 32 tested, chunked prefill interleaved with decode, host-side sampling per row per step | `src/serve/engine.rs:1-40` | Only when the whole model fits one GPU. CPU, split, pipeline, MoE: one request at a time (README) |
| Paged KV | Kernels read through a page table (stages 0–1, on main); allocator/pool/preemption is **PR #71, not merged** | roadmap Scale item 5; #71 body | Prefix sharing across sessions (radix/CoW) is "later: not scheduled" |
| Speculative decoding | Prompt-lookup and draft-model, single-stream only; lossless | `src/spec.rs`, `src/draft.rs`; MEASURED +37–63 % greedy tg128 | Sampled gain ≈ 0 (+0.8 %). Batched spec is #90, kill-rule pending |
| Multi-GPU | Pipeline parallel over PCIe via host memory; decode 1.00× one GPU, prefill 1.46–1.55× (MEASURED 2×4090) | `src/cuda/qwen2/pipeline.rs`; roadmap item 4 | **No tensor parallel.** P2P untested. Serve runs a pipeline one request at a time |
| Determinism | decode == batched prefill == verify == batched-with-others, bit for bit per device; fixed-order reductions; identity tuple + `system_fingerprint` | `docs/determinism.md`; tests listed there | Real and tested. CPU ≠ GPU. Streamed splits break it (documented). sm_120 vs sm_89 "not audited" |
| Sessions | `chat --session` save/restore of KV (29.8k positions in 0.41 s vs 3.16 s re-prefill, MEASURED 4090) | #48 | `serve` has no save/restore or host-RAM KV store ("no store", README) |

**Claimed but thin (OPINION, with evidence):**
- "Run the best models" is, on main, CPU-only for deepseek2 and gpt-oss, and the only DeepSeek-architecture file ever
  run is Moonlight-16B. Kimi K2 / DeepSeek V3 have never been loaded. The README's parenthetical list of what
  `deepseek2` covers reads as coverage; it is a code path validated on one 16B dev model.
- "Datacenter": the only datacenter GPU ever touched is an H100 for ~20 minutes per board, single stream, 0.5B–8B.
- Hardening: fuzzers and soaks exist (`scripts/fuzz_*.py`, `soak.py`, 223k requests, 0 failures — MEASURED, roadmap
  item 2), but "OOM / GPU-error injection, multi-day runs" are still open, and a CUDA error is handled by `exit(70)`.

---

## 2. Where it really stands vs llama.cpp, ollama, vLLM, SGLang, TensorRT-LLM

**What the benchmarks show (MEASURED, `docs/benchmarks-gpu.md`, board of 2026-10-18 on main `7f42c02`):**
- Single stream, Q4_K_M, 0.5B–7B + Qwen3-8B, RTX 4090 / 5090 / H100 SXM: ahead of llama.cpp master `ff5a233` and
  ollama 0.40.6 in 45/45 cells. Decode margin over llama.cpp: 1.03–1.19× (4090), 1.12–1.70× (5090), 1.13–1.30× (H100).
  pp2100: 1.06–2.16×. The 7B/8B 4090 decode margin is +3 %, "the least firm" (3 rounds, shared pods).
- Concurrency, one cell only: 4090, Qwen2.5-7B, ~560-in/512-out, greedy. Axon 181 / 999 / 1542 / 2240 tok/s at
  C=1/8/16/32; llama-server 133 / 613 / 667 / 976; **vLLM 0.31 (AWQ Marlin) 166 / 1149 / 1958 / 2954 / 3609 at C=64.**
  Axon trails vLLM 13–24 % at C=8–32 and has **no C=64 number**. Different weights and kernels, so not bits-equal.
- Long context, bella only, pre-#50: 7B decode at 32k **lost** to llama.cpp 72.4 vs 76.7; after #50 +3.3 % on its own
  A/B, "has not been re-run" against llama.cpp (`docs/results.md`).
- MoE: `qwen3moe` with experts on CPU trails `llama-server --cpu-moe` on prefill 1.5–2.6× (dev models). gpt-oss-20b
  prefill ~265 tok/s on bella CPU experts (roadmap). Q2_K/Q3_K CPU prefill −22 to −41 % vs llama.cpp.

**What they don't show (OPINION on gaps; all absent from the repo):**
1. **No vLLM / SGLang / TRT-LLM on an H100** at any concurrency, with FP8 weights, which is what a datacenter actually
   runs. The one vLLM comparison is a consumer card with AWQ-int4. TRT-LLM and SGLang are never named in a table.
2. **Nothing above 8B is on any scoreboard.** The 32B hybrid and 70B pipeline numbers are PR-body spot checks.
   "vLLM TP2 beats axon's single stream on 2×4090 (65–70 vs 43 tok/s)" is admitted in `results.md` "Known losses".
3. **No real MoE model.** Qwen3-30B-A3B, the model the MoE track is named for, has never been run (`results.md`).
4. **No AMD, no Apple, no ARM, no Windows/macOS.** llama.cpp and ollama run everywhere; vLLM/SGLang run ROCm.
5. **No TTFT / ITL under load vs vLLM**, no goodput curve, no prefix-cache hit-rate workload, no agent workload
   (roadmap: "no number in this section is MEASURED on this workload").
6. **Staleness.** `results.md` opens with a table of 25 merged PRs whose effect on the headline boards is "copied from
   the PR's own body". The CPU board predates several CPU merges. The sm_120 laptop board is from 10-15.

**The uncomfortable arithmetic (ESTIMATED from the measured cells).** Qwen2.5-7B Q4_K_M is ≈ 4.68 GB. On the 4090,
186 tok/s × 4.68 GB ≈ 870 GB/s = **86 % of 1008 GB/s**: genuinely near the wall. On the **H100 SXM, 311 tok/s × 4.68 GB
≈ 1.46 TB/s = 43 % of 3.35 TB/s.** llama.cpp is at 34 %. Both engines leave more than half the H100's bandwidth on the
floor at batch 1, and Axon's prefill GEMM runs at **26–30 % of int8 peak** on the 4090 (r15-4090-prefill §0; r17 finds
the MIO/L1 path, not the tensor pipe, is binding). "Beats llama.cpp" and "squeezes every drop" are different claims; the
first is measured, the second is not true yet on the hardware the mission names.

**Standing, one line each (OPINION):**
- vs **llama.cpp**: ahead on single-stream speed for small dense Q4_K GGUFs on NVIDIA; far behind on model/format/hardware
  coverage, ecosystem, and years of edge cases. Ahead on exactness guarantees.
- vs **ollama**: ahead on speed; behind on distribution, model library, pull-and-run UX, platform coverage.
- vs **vLLM / SGLang**: behind on throughput at C ≥ 8 on the one measured cell; absent on FP8, TP, prefix caching,
  structured output, logprobs, multi-LoRA, disaggregation, OpenAI API completeness, and every observability integration.
- vs **TensorRT-LLM**: not comparable yet; TRT-LLM's game (FP8/FP4 on Hopper/Blackwell at high batch) is one Axon hasn't
  entered.
- **Production readiness**: a LAN tool. No TLS, exit-on-CUDA-error, single process, no HA story, Prometheus metrics but
  no tracing, no request logs beyond stderr. Honest about it in the README.

---

## 3. Mission fit

| goal | score /10 | why |
|---|---|---|
| **Cheap and fast** *[edited]* | **5** | Fast at batch 1 on 7B/NVIDIA: yes, MEASURED and near the DRAM wall on consumer cards. Cheap means tokens per dollar at the batch sizes that make GPUs pay, and there the one measurement says −13 to −24 % vs vLLM on int4, no FP8, no C=64, and the H100 is at 43 % of bandwidth. The strict-order GEMM at 26–30 % of peak caps prefill economics. |
| **Run the best models** | **3** | Dense ≤ 8B: yes. Gemma 27B: refused. 70B: unmeasured PP. Qwen3-30B-A3B: never run. DeepSeek V3 / Kimi K2: architecture code exists, validated only on Moonlight-16B, CPU-only on main, routing groups refused. gpt-oss-120b: untested. Nothing above 16B active has produced a token on a GPU in this repo. |
| **Datacenters, squeeze every drop** | **2** | No TP, no FP8, no multi-node, no NVLink path, no high-concurrency kernels (the C-curve work is the first look at nb > 16, 10-18), one H100 session per board. The design choice that most defines Axon (fixed-order reductions) is also the one that rules out the standard datacenter tools (split-K, stream-K, ring all-reduce). |

**Biggest gap: goal 3, then goal 2 (OPINION).** The project is strongest exactly where the mission says it matters
least: single stream, small dense models, a laptop GPU. That is not a criticism of the work; it is a statement about
where the effort has gone. ~9 of the 17 GPU rounds in `history.md` are prefill/decode kernel rounds on 0.5B–8B.

---

## 4. Roadmap critique

**What's right (OPINION):**
- The "Scale" table's dependency logic (chunked prefill → one cache → paged KV → prefix sharing) is the correct order
  for multi-user on one card, and most of it has shipped in 4 days.
- Kill rules written before building, and `experiments-rejected.md`. That is rarer than it should be in this field.
- Refusing lossy MoE shortcuts (expert deferral, pre-gating) keeps the one differentiator intact.
- Stage 0 "CPU exact reference first" for new architectures is the right way to make GPU kernels gateable.

**What's missing:**
- **FP8 / NVFP4 weights and activations.** Not in the roadmap at all, except as a footnote under DeepSeek V4.1. H100/B200
  economics are FP8; every frontier model ships FP8 natively. GGUF-only means Axon can only ever run someone else's
  requantization of the best models. This is the single largest strategic hole for goals 2 and 3.
- **Tensor parallel** is "trigger: after 1b" with an admission that fixed-order all-reduce "costs speed against NCCL".
  No design note exists for how exact TP would be laid out (per-row ownership vs. deterministic tree reduce).
- **An agent/serving benchmark vs vLLM and SGLang on an H100** (TTFT, ITL p99, goodput at C=32–128, shared 20k
  prefix). The roadmap itself says nothing on this workload is measured.
- **Observability beyond Prometheus counters**: per-request trace ids, structured logs, OpenTelemetry, a slow-request
  log. Trivial to add, absent.
- **OpenAI API completeness**: `logprobs`, `n`, JSON schema / grammar, embeddings, `logit_bias`. Agent harnesses hit
  `logprobs` and structured output quickly.
- **A library crate.** Still binary-only (no `lib.rs`); the 10-13 review's top cross-cutting item, unaddressed. Nobody
  can embed Axon.

**What's mis-ordered (OPINION):**
- **Three MoE architectures in parallel at stage 1 (qwen3moe #67 merged, deepseek2 #89, gpt-oss #98), none with experts
  on the GPU.** Every one of them loses prefill to `llama.cpp --cpu-moe` because the experts stay on the CPU. One
  architecture carried through to GPU-resident experts (r16 pick 4) would prove the design; three at "attention on GPU"
  prove breadth without a win. Pick one (Qwen3-30B-A3B is the one people run), finish it, measure the real model.
- **r15–r17 GEMM micro-rounds** (+2.6–4 % pp2100 on a 4090, `sb-4090-ab-20261018.md`) while the H100 sits at 43 % of
  bandwidth at batch 1 and the concurrency curve trails vLLM. The roadmap's own thesis says "more kernel tuning won't move
  single-stream 7B much" and then keeps doing it.
- **gpt-oss before 30B-A3B on a GPU.** gpt-oss-20b is a product model for one vendor's format; 30B-A3B is the model
  class the MoE section is justified by.

**What to cut or park:**
- Pascal (sm_61) support work beyond "refuse cleanly". `jade`'s GTX 1050 is a test host, not a target.
- Prompt-lookup drafting tuning (measured ≈ 0 sampled; story text −4 %).
- `--draft` K sweeps on splits; the 32B-on-12GB story is a laptop demo, not a mission item.
- The pp560 cells on every board: the doc itself says "inflated… quote pp2100". Stop printing them as wins.

---

## 5. Architecture risks at 10× scope

1. **The bit-identity contract is a tax that compounds with batch size and GPU count (OPINION, evidence below).**
   On today's shapes it is "close to free" (research-determinism: GEMV at 93 % of DRAM). But: the prefill GEMM is 26–30 %
   of int8 peak and the strategist's own table (r16 §1) lists stream-K/split-K and the per-32 float epilogue as "changes
   the order: not worth it". TP needs a fixed-order all-reduce instead of NCCL. Flash-decoding's split-KV is forbidden.
   Every one of those is a standard tool for the top of the C-curve and for multi-GPU. The contract was chosen when the
   target was a laptop CPU; nobody has re-derived its cost for the datacenter mission. **The question isn't whether to
   keep it; it is whether "same bits as solo" should be a per-request opt-in (`seed`-style) rather than the global
   design constraint.** vLLM made the opposite default and ships batch-invariance as a mode.
2. **Everything on the GPU is a `CudaQwen2` (CODE `src/cuda/qwen2.rs:421`; `Backend` enum `src/lm.rs:317-330`).** Llama,
   Gemma 3, MoE, MLA (`mla.rs` on #89), gpt-oss (`LayerW::oss` on #98) are arms hung on one 3.1k-line struct plus a
   2.2k-line `batch.rs`; 37 arch-conditional sites in those two files today, and each new architecture adds a module
   (`gemma.cu`, `moe.cu`, `mla.cu`, `oss.cu`) with its own attention kernel. `decode.cu` + `prefill.cu` are 5.2k lines of
   hand-written kernels per quant type × tile × arch. At 10× models this becomes llama.cpp's `ggml-cuda` without ggml's
   graph abstraction. There is no operator/graph layer; the forward pass is written per backend (CPU `model.rs` 5.7k,
   GPU `qwen2.rs`, hybrid, pipeline, MoE: five `Backend` variants, five `BackendCache` variants).
3. **NVRTC at first run.** Startup compiles ~84 kernels per module per sm (cached in `~/.cache/axon/cuda`); #72 noted
   "about 9 s of one-time ptxas" per added kernel set; the devstack has an `axon-pods warm` step because compile lands
   in timing. In a fleet, every new image/GPU pair pays it, and NVRTC version skew already needed a fallback chain (#39).
   Fine for a CLI, awkward for autoscaling.
4. **Single engine thread, host-side sampling, thread-per-connection HTTP (CODE `src/serve.rs:5-13`,
   `src/serve/engine.rs:6-18`).** Each step: device→host candidates for every row, per-row `Sampler::pick` on the host,
   detokenize, stop-string match, SSE write, then the next step. At C=32 on a 7B that is inside an 8–12 ms step; at
   C=128 on a 0.5B it will be the step. No async runtime, no scheduler/executor split, so prefill/decode disaggregation
   or multi-GPU serving means rewriting the loop. The GPU handle is `!Send`: one model per process, period.
5. **MoE abstraction duplication (CODE).** `src/model/moe.rs` (CPU, 875 lines) carries `gate_fn` and `oai` side by side
   with a documented "correctness rests on never setting both" (roadmap "Cleanups owed"); `src/moe.rs` (`MoeQwen3`) is a
   third struct gluing `CudaQwen2` to `Qwen2::new_experts`; `src/cuda/qwen2/moe.rs` is the GPU half; #89 adds an MLA
   variant that is "a deepseek2 `CudaQwen2` with empty `layers`". GPU-resident experts (r16 pick 4) will be a fifth
   place. Decode via per-layer graph segments with a D2H/H2D round trip per layer is a latency floor that caps MoE decode
   regardless of kernel work.
6. **Unsafe surface.** 668 `unsafe` sites, hand-rolled dlopen FFI, a hand-rolled HTTP parser, a hand-rolled GGUF reader
   treated as untrusted (good), all in one binary with the engine. Fuzzed, but one process = one blast radius.
7. **Test cost.** The release suite is ~2200 s serial on an exclusive 4090 plus a debug-assertion suite plus md5 gates;
   GPU tests must serialize (ILLEGAL_ADDRESS on packed pods, root cause "a hypothesis", #82). At 10× tests the gate is
   hours per PR; the merge train exists because of this already.

---

## 6. The dev process

**What it does well (MEASURED by its artifacts):** written specs with kill rules; a critic with a fresh context who has
caught real bugs (methodology §5 lists five); every PR carries a MEASURED/ESTIMATED evidence table; A/B on the same host
interleaved; md5 gates across models; rejected work logged with numbers; a job queue with fair share, `--same-host`,
pod budgets and a kill switch; a merge train that gates the combined tip once. 88 PRs in 9 days with a protected main
and (as far as the docs show) no silent numerics regression is a real achievement.

**Where it is fragile (OPINION):**
- **GitHub CI is cosmetic.** `.github/workflows/ci.yml` has no GPU and skips model tests (`AXON_SKIP_GPU_TESTS=1`).
  Every real gate runs on private hardware through `axon-devstack`, which hard-codes bella's paths and a session
  id *[edited]* ("not portable yet", devstack README). A contributor cannot reproduce a gate; a reviewer cannot audit one. The
  evidence chain for every "MEASURED" is a job id in a log on one laptop.
- **One strategist voice.** Picks, designs, critiques and research all come from the same model family (methodology
  "Who decides"). The critic reviews the diff, not the premise. Nobody outside the loop has questioned whether the
  contract or the GGUF-only choice should hold at the new mission.
- **The agents write the docs that grade the agents.** `results.md`, `history.md`, the roadmap statuses, the PR bodies
  are all agent-authored. They are unusually honest (losses are bolded), but "45/45 cells" headlines sit above a results
  page that says no board has been run on the current main.
- **Pod discipline is learned by spending.** $60 cumulative; the 10-18 board wasted $0.85 of $2.79 on script bugs and
  on-pod compiles; a rule was added after. Good loop, but every lesson costs money, and the budget caps are the only
  brake.
- **Bus factor: 1, and it's not the code's author.** Every commit has one human author *[edited]* but, by the project's own account,
  "mostly AI agent builders under a foreman". The 10-13 review commissioned a "teachability" report (gojo) for this
  reason. Nothing since addresses it: there is no architecture overview a human could hold (START-HERE's source map is a
  file list), no design doc for the serve engine or the `CudaQwen2` arms, no ADRs. If the agents stop, can the owner debug
  an ILLEGAL_ADDRESS in `prefill.cu` or a slot-leak in `engine.rs`? The honest test: pick a PR and explain it cold.
- **Review blind spots:** no profiler traces in the repo (Nsight), no perf regression CI, no fuzz corpus committed,
  determinism "not audited" across sm_120/sm_89/sm_90, no soak on a GPU with the slow-reader harness ("changed, not yet
  exercised"), and the dev models (Pruned-2B, 4x0.6B) are hand-made surrogates whose relation to the real 30B is
  asserted, not shown.

---

## 7. The juicy questions

1. **Who is the first user, by name, and what do they run today?** Everything downstream (formats, API surface,
   hardware) follows from this, and the roadmap's "agent harnesses" section says nothing on that workload is measured.
2. **Six days ago the internal strategy review said "never GPU backends, own the ≤1B CPU niche". Today the mission is
   datacenters. What changed, and what will the mission be in six more days?** Strategy that pivots weekly can't be
   invested in; an investor will ask what you'd refuse to build.
3. **Is bit-identity a product or a dev tool?** If it's a product, name the customer who pays for batch-invariance and
   accepts the GEMM at 30 % of peak. If it's a dev tool, why is it a global constraint instead of a mode?
4. **GGUF-only means you run other people's quantizations of other people's models. Where is FP8 / NVFP4 and native
   safetensors, and how do you "run the best models" without it?** Frontier models ship FP8; the datacenter runs FP8.
5. **The only concurrency cell shows vLLM 13–24 % ahead on int4 and 3609 tok/s at C=64 where Axon has no number. What is
   your C=64 and C=128 number on an H100 against vLLM and SGLang with FP8, and when will it be measured?** That cell is
   the mission; everything else is a warm-up.
6. **Why three MoE architectures at stage 1 and zero at "experts on the GPU"?** Breadth at the half-finished stage costs
   the same as depth on one and proves less.
7. **What is "done" for Kimi K2?** A token on an 8×H100 pod at any speed, or ≥ X tok/s at ≥ Y concurrency within Z % of
   SGLang? Without the number, "long-term goal" means "never".
8. **Pick any kernel in `prefill.cu` and any 200 lines of `engine.rs`: can you explain them without an agent?** If the
   answer is no, the project's bus factor is the agent subscription, and that is what an interviewer will probe.
9. **What is the plan for tensor parallel with fixed-order reductions, and have you costed it against NCCL?** The
   roadmap admits it "costs speed"; nobody has said how much. If it is 2×, the contract dies at the first multi-GPU node.
10. **Which of the 45 "wins" would survive a llama.cpp build tuned the way you tune Axon?** llama.cpp ran at defaults;
    `ff5a233` is a moving target; the 7B/8B decode margin is +3 % on three rounds. What's the margin you'd bet on?
11. **How much GPU time and money has the project consumed, and what's the cost per merged PR?** $60 of pods is cheap;
    the agent-hours are not. An investor asks for burn per unit of measured progress.
12. **Why hand-written HTTP, hand-written dlopen FFI, no library crate, no async runtime?** "No dependencies" is a
    constraint you chose; say what it buys at the serving tier, where it is costing you TLS, HTTP/2, and embeddability.
13. **What happens when a CUDA error hits at C=32?** Today the process exits 70 and 32 streams die. Is that acceptable
    for a datacenter engine, and what's the recovery design?
14. **What is Axon's answer in one sentence when someone says "why not vLLM"?** If the sentence contains "llama.cpp", you
    are positioned against the wrong competitor for the stated mission.
15. **If you could only ship one thing in the next month, is it (a) Qwen3-30B-A3B GPU-resident at vLLM parity, (b) FP8 on
    H100, or (c) an exact TP design?** The answer reveals which of the three mission goals you actually hold.

---

## 8. Top 5 recommendations for the next 2 weeks

1. **Run the mission benchmark once, honestly, and publish it.** One H100 (or 2× for TP baselines), Qwen2.5-7B and
   Qwen3-30B-A3B, vLLM + SGLang with their best quant, C=1/8/32/64/128, TTFT/ITL p99/goodput, plus a shared-prefix agent
   workload. ~$30 of pod. **Payoff:** replaces "45/45 vs llama.cpp" with the number the mission is about, and re-ranks
   the roadmap in a day. Expect to lose; that's the point.
2. **Finish one MoE to the end: Qwen3-30B-A3B with GPU-resident experts (r16 pick 4), on a 24 GB card, measured against
   llama.cpp and vLLM.** Park #98's expert streaming and deepseek2 stage 2 until it lands. **Payoff:** the first "best
   model" that is actually fast on Axon, and the template for Kimi/DeepSeek.
3. **Write the determinism-cost ADR.** One page: for TP all-reduce, split-K prefill, and C ≥ 64 decode, the measured or
   estimated cost of the fixed-order rule, and a decision: global contract, per-request mode, or NUMERICS-versioned
   "fast" kernel set. **Payoff:** stops every future kernel round from re-litigating this, and tells you whether the
   datacenter mission is compatible with the engine's identity.
4. **Make the gates reproducible by someone who isn't bella.** A GPU CI runner (one 4090 pod on a schedule, or a
   self-hosted runner) that runs the fast tier and the md5 gate on every PR, with artifacts attached to the PR. Commit the
   scoreboard script. **Payoff:** the MEASURED labels become auditable; a second human can contribute; the bus factor
   moves off one laptop.
5. **Freeze kernel micro-rounds on ≤ 8B dense and spend the builder time on `lib.rs` + an FP8 weight path spike (one
   GEMM, one model, H100).** **Payoff:** +3 % pp2100 rounds are below the noise of the question that matters; an FP8
   spike answers question 4 with a measurement instead of a roadmap footnote, and a library target unlocks the
   embeddability nobody has had since 10-13's review asked for it.

---

*Fable. Read-only; nothing built or run. All numbers are from the cited repo documents and logs as of 2026-10-19 01:14 EDT.*
