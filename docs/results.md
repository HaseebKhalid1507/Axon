# Where Axon stands

The latest measured comparisons against ollama and llama.cpp, per host and model. Every number here is
MEASURED unless marked otherwise, and comes from a scoreboard or a PR. Main is `39624db` (2026-10-25 21:19 EDT).

**Newest result: the first datacenter serving board** (H100 SXM, main `c15c90e`, 2026-10-19, [below](#serving-runpod-h100-sxm-main-c15c90e-2026-10-19)).
Axon is the fastest of five engine/weight rows at one stream and 2.0–3.8× behind vLLM / SGLang at 128 concurrent
requests.

> **No single-stream board below was run on `39624db`** (nor on `764792f`, `4507c75` or `a591e17`, the mains this
> page was written for before). The newest single-stream board is RunPod at `7f42c02` (2026-10-18; see the exception at the
> end of this box); the H100 serving board ran on `c15c90e`. The newest full boards are from main `0851798` (bella GPU, 2026-10-15 13:06) and main `f8d8a86`
> (RunPod, avante, CPU, 2026-10-15/16). Merges since then that can change speed or output on some path (none
> has been re-boarded). Each effect is copied from that PR's own body: **MEASURED** where the PR measured it
> (host and A/B as stated there), **not measured** where it didn't. Nothing in this table was re-run for this page.
>
> | merged after | PR | effect on these numbers (from the PR's own A/B) |
> |---|---|---|
> | 0851798 | #24 r5 prefill | GPU pp2100 +2.3 % (7B) to +23 % (0.5B); the bella board predates it; the RunPod/avante boards include it |
> | f8d8a86 | #30 r5 decode | GPU tg128 +2–5 % on small models (bella, builder's runs), ~0 on 7B/8B |
> | f8d8a86 | #33 Hopper | sm_90a only: 7B prefill +12 % on H100 PCIe (`AXON_GEMMH=1`; the default routing was not run on an H100) |
> | f8d8a86 | #32 CPU decode pool cap | bella CPU tg +2–10 % (builder's runs; the exact merged tree wasn't timed) |
> | f8d8a86 | #41, #21 | time to first token only |
> | — | #27, #38, #34, #42 | opt-in or serve-only; `axon gen` output is unchanged (md5) |
> | a591e17 | #45 serve chunked prefill | serve batch only (a prompt admitted while others decode). **MEASURED** (RTX 4090, 5 rounds): during a 24k-token admit the decoding clients' ITL p99 2392 → 69.7 ms (7B) and 342 → 62.5 ms (0.5B); the admitted prompt's TTFT +48 % (7B) / +19 % (0.5B); a 300-token prompt arriving meanwhile 2137 → 111 ms (7B). Single user unchanged (7B tg128 184.98 vs 184.96). `gen` md5 30/30 identical |
> | a591e17 | #48 `chat --session` | opt-in. **MEASURED** (4090, 7B, 29,840 positions): restore at restart 408–451 ms (f16) / 223–234 ms (q8) vs a re-prefill of 3.16 s / 3.39 s. `gen` md5 30/30 same |
> | a591e17 | #49 YaRN | opt-in (`--rope-scaling yarn`); default output md5 30/30 equal. With the flag the output changes at every length: **MEASURED** 7B f16 perplexity +7.7 to +8.0 % at 2k–8k windows (4090). Speed: **not measured** |
> | a591e17 | #50 long-context decode | **MEASURED** (bella, tg at position 32512 vs main, two jobs): 7B f16 +3.3 % / +3.5 %, 7B q8 +17.6 % / +18.1 %, 0.5B f16 +10.0 % / +12.1 %, 0.5B q8 +25.7 % / +26.9 %, q3-8B f16 +4.3 % / +4.0 %. Short positions: 1.5B, 7B, q3-8B at or above main; 0.5B −0.7 … +0.3 % (within noise). Bits unchanged (md5 40/40 on the pre-rebase tree; the final-tree md5 was still pending in the PR body) |
> | a591e17 | #51 fingerprint | output unchanged (md5 0/15 + 0/15, twice). Startup: the model digest runs at every `gen`/`serve` start. **MEASURED** +~40 ms warm (7B, 4090 pod), +75 ms warm (avante), +~180 ms cold (Llama 3.2 1B, avante) |
> | a591e17 | #52 drafting on a CPU/GPU split | only with `--draft-model` next to a split model, where greedy now defaults to K=7. **MEASURED** (bella, 32B Q4_K_M, shared machine, not cool-gated): greedy tg128 5.8–6.1 with no draft → 11.4–21.5 at K=7 depending on the prompt; sampled 5.2 → 8.8–17.0. md5 48/48 identical on an earlier head (the final-tree md5 was pending in the PR body). With automatic placement the drafter's VRAM reserve moves the split (32B: 27 → 24 GPU layers), which changes the bits; pin `--gpu-layers` (#58) |
> | a591e17 | #55 chunk budget follow-ups | serve batch only. **MEASURED** (bella, 7B, q8 KV): idle server, two ~14.4k prompts sent together: first TTFT 4748 → 4471 ms, last 8948 → 9002 ms (+0.6 %). Mid-admit ITL p99 154.0 vs 154.6 ms. Single user 100.9 vs 101.0 tok/s. md5 0/15 + 0/15 |
> | a591e17 | #56 prefill attention launch shape | **MEASURED** (bella, `gen` prompt eval, 5 rounds): 7B pp16384 +5.5 %, pp32512 +3.4 %; q3-8B +2.8 % / +3.4 %; every pp8192, pp560 and pp2100 row −0.6 … +1.9 %. RTX 4090: q3-8B pp16384 +15.8 %, pp32512 +17.0 %, other rows within ±0.3 %. sm_80/86/87/90/100 keep main's choice by construction (**not measured**). md5 60/60 on bella, a 5090 and a 4090 |
> | a591e17 | #57 idle prefill fairness | serve batch only. **MEASURED** (bella, 7B, q8): no change (idle two 14.4k prompts, first / last TTFT 4906 / 9908 vs 4897 / 9898 ms; mid-admit ITL p99 169 vs 176 ms, a hot run). md5 0/15 + 0/15 |
> | a591e17 | #60 one KV cache per slot | serve with a pool. **MEASURED** (bella, 7B): slots at 32k f16 1 → 2, q8 4 → 5; C=1 decode 101.3 → 102.7 tok/s (`--ctx 8192`) and 101.4 → 102.8 (32k), within noise; every reply's text identical. Which slot a request reuses changed (a prefix counts if it is half the slot or more than half the prompt, else LRU), so `cached_tokens` can differ. md5 0/15 × 3 |
> | a591e17 | #61 serve ops | speed **not measured**. Opt-in flags; `/metrics` and `/slots` state is published once per engine loop. md5 `gen` + `serve` 26/26 same |
> | a591e17 | #63 MoE stages 0–1 | new path only (`qwen3moe` on the CPU); other models md5 0/18 CUDA + 0/18 CPU. Numbers: [New models and types](#new-models-and-types-since-the-boards) |
> | a591e17 | #64 paged KV stages 0–1 | **MEASURED** ratios PR / main, bella 7B: tg128 0.9995, `--kv q8` 0.9986, decode after a 30k prefix 1.003 / 0.994 (q8), pp2100 0.997 / 1.003, pp560 0.998 (7-round rerun); Gemma 3 1B pp560 1.031, pp2100 1.006, tg128 0.995, tg128@2k 1.003; jade (sm_61) 0.5B pp560 1.002, pp2100 0.9997, tg128 1.003. md5 identical |
> | a591e17 | #67 MoE stage 2 | new path only (`qwen3moe --device cuda`); dense md5 0/18 CUDA, 0/15 q8, 0/18 CPU. Numbers below |
> | a591e17 | #70 Responses + ollama APIs | speed **not measured**: the new routes translate into the chat-completions path (engine untouched). md5 `gen` + `serve` 26/26 same |
> | a591e17 | #72 Q2_K / Q3_K / IQ4_NL | new types only; other models md5 0/21 × 3 (CPU, CUDA f16, q8); SASS of all 123 existing kernels identical to main. The PR's follow-up list says the new GEMM kernels add about 9 s of one-time ptxas (cached); not labelled MEASURED there. Numbers below |
> | a591e17 | #75 `chat --forget-to` | changes `chat`'s output by default once the context fills (main == `--forget-to min`; the default 0.5 differs). **MEASURED** (4090, 40-turn scripted chat, `-n` unlimited): 7B ctx 8192 prefills 169,850 → 26,020 tokens, forgetting-turn TTFT median 474.6 → 273.6 ms; 0.5B ctx 2048 21,269 → 11,277 tokens. `gen` md5 0/15 + 0/15 |
> | a591e17 | #78 gpt-oss on the CPU | new path only (gpt-oss-20b, MXFP4); md5 36/36 on 9 other models. Numbers below |
> | — | #54, #62, #68, #76 | code with no speed claim, output unchanged (md5): session-file lock (#54), auth before any body (#62), CUDA refuses gemma3 head sizes other than 256 (#68), MoE guards and a host-only RAM check (#76: a `qwen3moe` split's RAM need, from the check's formula, Pruned-2B 3.2 → 2.7 GB, 30B 20.5 → 19.5 GB) |
> | — | #46, #47, #53, #58, #59, #65, #66, #69, #73, #74, #77, #80, #85, #87 | docs, tests and scripts only; no speed or output effect (#80's non-test release binary is byte-identical to main's) |
> | 4507c75 | #81 C curve | serve batch only (the pool step). **MEASURED** (RTX 4090): 7B step at nb 32 14.08 → 9.81 ms (−30 %), nb 64 −12 %; serve 7B C=32 1868 → 2240 tok/s (an earlier build). Same bits (the zero-filled tile columns are never stored); md5 20/20 |
> | 4507c75 | #86 C curve 2 | pool steps of ≤ 16 rows (q\|k\|v and gate\|up in one launch). **MEASURED** (RTX 4090): 7B nb 1 / 8 / 16 −6.5 / −5.1 / −6.2 %, 0.5B −16 … −19.5 %. Same bits by construction; md5 0/20 |
> | 4507c75 | #79 deepseek2 stage 0, #83 soak / zoo | new path only (`deepseek2` on the CPU) and clearer refusal messages; md5 0/51 (#79) and 0/18 + 0/18 (#83) |
> | 7f42c02 | #92 wave-sized prompt passes, #97 | `gen` / `logits` (#92), then `serve`'s single stream and `chat` (#97): multi-pass prompts split to fill the GEMM waves. **MEASURED** (RTX 4090, 7 rounds): pp2100 7B +3.03 %, Qwen3-8B +3.85 %; other lengths −0.4 … +1.8 %. Not measured on an H100 or a 5090; #97 not timed separately. Same bits (pass boundaries are outside the identity) |
> | 7f42c02 | #100 gemm2t 128 × 64 tile | prefill passes of > 64 tokens, Q4_K / Q5_K products. **MEASURED** per MAC: sm_89 +22–28 % (avante), sm_120 +21–26 % (bella); end to end 7B pp1050 +17.3 % (bella), 3B pp1050 +15.4 % (avante). RTX 4090 pp2100 ESTIMATED +12–15 %, **not run**. md5 0/35 (bella) |
> | 7f42c02 | #71 paged KV stages 2–3 | `serve` on one GPU (the pool now allocates pages; `AXON_PAGED_KV=0` = the old pool). **MEASURED** (bella exclusive): C=1 92.4 → 92.8, C=8 401.2 → 394.6 (inside the 383–404 round spread); 7B 32k, 7 × 15k agents + 600: aggregate ~+10 %. md5 0/45 |
> | 7f42c02 | #84 attn_blk_mq | one-sequence passes of ≤ 32 queries (`AXON_ATT_MQ=0` off). **MEASURED** (bella): tg128 −0.3 % (within noise); serve admit at chunk 32 TTFT −6.4 %; gen + draft at 21.7k +20.4 %. md5 22/22 |
> | 7f42c02 | #89 deepseek2 stage 1, #98 gpt-oss stage 1 | new paths only (`--device cuda` on those architectures); md5 vs main 0 differ (#89: dense, qwen3moe, gpt-oss rows; #98: 0/106). Numbers: [New models and types](#new-models-and-types-since-the-boards) |
> | 7f42c02 | #102 safetensors stage A | new path only (Hugging Face directories, CPU); GGUF `gen` md5 0/20 |
> | — | #82, #93 | tests and debug builds (#82: one GPU test at a time on every card, a debug-only launch-size check; release non-test code unchanged) and a serve metrics-ordering fix with no speed claim (#93) |
> | — | #88, #91, #96, #101 | docs only (#101 is a PROPOSED ADR: nothing built) |
> | c15c90e | #104 one MoE router kernel | MoE paths only; bits unchanged on qwen3moe, gpt-oss and deepseek2 (md5 vs 764792f). **MEASURED** (avante, Qwen3-MOE-4x0.6B, cuda): tg 100.45 → 100.21 tok/s (−0.2 %, round spread ±2.8 %: noise); gpt-oss and Moonlight **not timed** |
> | c15c90e | #117 library split | no behaviour change. **MEASURED** (avante, 7 + 15 interleaved rounds): speed-neutral, the 0.5B cuda cells 0.9999–1.0004× main in both runs (the 1.5B cells flip sign between runs, a clock state); md5 0 of 40 pairs differ |
> | c15c90e | #111 gpt-oss tool calling, #136 headerless replies | `serve` gpt-oss requests with tools only; requests without tools and every Qwen / Llama tool request byte-identical (#111: md5 56/56, serve A/B identical; #136: serve A/B 23/23 on every model). Speed **not measured** |
> | c15c90e | #127 resident deepseek2 experts | opt-in (`--gpu-experts`, default 0 = main's path). **MEASURED, TAINTED** (bella, Moonlight, `auto` vs 0): tg ×2.06, pp ×3.40 ([below](#new-models-and-types-since-the-boards)). md5 vs main all identical (dense 30/30; Moonlight main = 0 = `auto`; every local GGUF but 32B, cpu / cuda / `--kv q8`) |
> | c15c90e | #131 measured step budget | `serve` batch only: `--step-ms 60` (default) sizes the prompt chunk from the measured step cost; `--admit-order sjf` opt-in. **MEASURED** (clean bella cells, default vs main's scheduling): ITL p99 −3.8 / −0.3 / 0 %, aggregate +0.8 / −0.4 / −1.6 % (7B C=16, 1.5B C=32, showdown client C=32). Not run on a 4090 or an H100 (#139). `gen` md5 0 of 20 pairs differ; serve digests 1 per model over 12 runs. [Tables](benchmarks-gpu.md#serve-scheduler-ab-laptop-gpus-2026-10-25-131) |
> | c15c90e | #133 cgroup-aware RAM figure | load decisions (MoE RAM refusal, copy vs map, read-ahead, the CPU serve context) under a memory cgroup limit; nothing changes outside one. Speed **not measured**; md5 12/12 cpu + 12/12 cuda |
> | — | #113, #124 | `serve` metrics ordering (#113: a request leaves in-flight before its last byte; md5 0 of 20) and the opaque library API with env errors instead of panics (#124: 85 md5 pairs identical, serve A/B 138/138); no speed claim |
> | — | #105, #107, #120, #122, #128, #134 | tests and test-only kernels; no product path changed (#120 / #122: existing kernels' SASS unchanged; #134: md5 identical) |
> | — | #103, #123, #130 | docs only |
>
> **Exception:** the [RunPod board of 2026-10-18](#gpu-runpod-rtx-4090--rtx-5090--h100-sxm-main-7f42c02-2026-10-18-1411--1508)
> ran on main `7f42c02`, which contains every row above marked `4507c75` or earlier; only the `7f42c02` and `c15c90e` rows
> apply to it (for prefill, mainly #92 and #100). The [H100 serving board](#serving-runpod-h100-sxm-main-c15c90e-2026-10-19)
> ran on `c15c90e`, which contains every code row above the `c15c90e` rows; for serving at C ≥ 8, #131's step budget
> is the one that applies.
>
> Treat the tables as "at least this good on the default path", with the caveat that nobody has
> re-measured that. The exceptions are start-up costs: #51's model digest and #72's one-time kernel compile come
> after the startup rows below.

Method for all boards: [methodology.md](methodology.md#benchmark-protocol). Full raw tables:
[benchmarks-gpu.md](benchmarks-gpu.md) and [../BENCH.md](../BENCH.md).

## Serving: RunPod H100 SXM, main c15c90e, 2026-10-19

Run `h100-1019-2058`. Qwen2.5-7B-Instruct on one H100 80GB HBM3 SXM: Axon `serve --parallel 128` (Q4_K_M GGUF) vs
vLLM 0.32.0 and SGLang 0.5.22, each with AWQ int4 and with FP8. Closed loop, C = 1 … 128, two workloads (*classic*:
519-token prompt, 512 out; *mix*: chat plus 2k–7.6k-token prompts), 2 rounds, engines interleaved per C, greedy.
**The weights differ per row** (Q4_K_M vs AWQ vs FP8; Axon has no FP8 path), so this compares engines as deployed,
not kernels on equal bits. Full tables: [benchmarks-gpu.md](benchmarks-gpu.md#concurrency-h100-sxm-qwen25-7b-2026-10-19);
method, cost and what went wrong: [log](log/2026-10-19-h100-showdown.md).

| aggregate tok/s | C=1 | C=8 | C=32 | C=128 |
|---|---:|---:|---:|---:|
| mix: Axon | **287** | 795 | 1,324 | 1,868 |
| mix: best other (engine) | 284 (SGLang-AWQ) | 1,212 (SGLang-AWQ) | 3,448 (SGLang-FP8) | 7,146 (vLLM-FP8) |
| mix: vLLM-AWQ (like-for-like 4-bit) | 235 | 1,096 | 3,012 | 5,748 |
| classic: Axon | **304** | 1,278 | 2,466 | 4,485 |
| classic: best other | 296 (SGLang-AWQ) | 2,015 (SGLang-AWQ) | 6,149 (SGLang-FP8) | 14,380 (SGLang-FP8) |
| classic: vLLM-AWQ | 240 | 1,786 | 5,682 | 12,190 |

| latency at C = 128, mix | Axon | the other four |
|---|---:|---|
| TPOT p50 | 57.9 ms | 12.7–26.4 ms |
| ITL p50 / p99 | 58.0 / **77.5** ms | 5.2–10.8 / 126–320 ms |
| TTFT p50 / p99 | 2,361 / 14,513 ms | 452–1,161 / 1,920–5,266 ms |
| requests inside the SLO (TPOT ≤ 50 ms, TTFT ≤ 2 s + 0.25 s per 1k prompt tokens) | 0.256 | 0.668–0.99 |

- **Ahead:** C = 1 throughput (+20–27 % over vLLM, but only +1 % / +3 % over SGLang-AWQ: inside one pod's
  uncertainty), C = 1 mix TTFT p50 (14.6 vs 18.1–27.5 ms), ITL p99 at mix C = 128 (flat, slow steps rather than
  spikes), and cold start (1.0 s with the shipped kernel cache vs 42–71 s).
- **Behind:** throughput at every C ≥ 8, by 2.0–3.8× at C = 128 (1.8–3.2× on classic); ITL p50, TPOT and TTFT p99 at
  C ≥ 32; on classic, also ITL p99 from C = 32 up (51.7–62.0 vs 5.2–39.7 ms).
- **Why (ESTIMATED, not profiled):** the batched-decode step at 64–128 rows on sm_90 (tuned on sm_120 / sm_89; no
  decode-shaped wgmma / Marlin-class GEMM) and admission / chunked prefill at high C. On the 4090 at C = 32 Axon was
  0.76× vLLM-AWQ; on the H100 0.43×. Plan: [roadmap, proposed order](roadmap.md#next-proposed-order-outside-review-10-19-owner-decides).
- **Cost:** $3.09 for the pod (46 min, secure, $3.99/h); of the driver's $2.96, $1.46 bought timed cells.
- **Before the pod:** an exclusive avante A/B (job `1019-203211-9269`) showed the 10-19 merge train didn't regress
  serve: ecb53b7 vs c15c90e, classic C=1/2/4 333/570/1,067 vs 333/571/1,067 tok/s, mix 306/482/793 vs 306/474/791.

The earlier concurrency cell: RTX 4090, 10-18, Axon feat/c-curve vs vLLM 0.31.0 AWQ, classic shape: C=1 181 vs 166,
C=8 999 vs 1,149, C=32 2,240 vs 2,954 ([benchmarks-gpu.md](benchmarks-gpu.md#concurrency-rtx-4090-qwen25-7b-2026-10-18)).

## GPU: bella (RTX 5070 Ti Laptop, sm_120, 12 GB), main 0851798, 2026-10-15 13:06

3 interleaved rounds, medians. Each model was cooled first (39–58 °C); before each run the script waited
for the GPU to be ≤ 10 % busy. axon `gen --device cuda`; ollama `num_gpu 99`; llama.cpp 4413a19 CUDA
build via `llama-bench` (its own prompts, from an empty context). The ollama version is 0.35.0 according to
the foreman's log; BENCH.md says the version "was not recorded with this run".

| model | test | axon | ollama | llama.cpp | vs ollama | vs llama.cpp |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5 0.5B Q4_K_M | pp560 | 44,780 | 14,289 | 25,334 | 3.13x | 1.77x |
| | pp2100 | 44,681 | 42,594 | 42,531 | 1.05x | 1.05x |
| | tg128 | 825 | 583 | 693 | 1.42x | 1.19x |
| Qwen2.5 1.5B Q4_K_M | pp560 | 21,353 | 12,158 | 13,026 | 1.76x | 1.64x |
| | pp2100 | 20,168 | 17,304 | 17,320 | 1.17x | 1.16x |
| | tg128 | 420 | 370 | 406 | 1.13x | 1.03x |
| Qwen2.5 3B Q4_K_M | pp560 | 10,801 | 7,516 | 7,563 | 1.44x | 1.43x |
| | pp2100 | 10,273 | 9,416 | 9,302 | 1.09x | 1.10x |
| | tg128 | 256 | 229 | 206 | 1.12x | 1.24x |
| Qwen2.5 7B Q4_K_M | pp560 | 5,040 | 4,227 | 4,412 | 1.19x | 1.14x |
| | pp2100 | 4,898 | 4,800 | 4,105 | **1.02x** | 1.19x |
| | tg128 | 118 | 111 | 112 | 1.06x | 1.05x |
| Qwen3 0.6B Q8_0 | pp560 | 35,793 | 20,216 | 20,179 | 1.77x | 1.77x |
| | pp2100 | 31,853 | 30,778 | 30,827 | **1.03x** | **1.03x** |
| | tg128 | 646 | 472 | 542 | 1.37x | 1.19x |
| Qwen3 4B Q4_K_M | pp560 | 8,206 | 6,008 | 6,265 | 1.37x | 1.31x |
| | pp2100 | 7,781 | 7,214 | 7,091 | 1.08x | 1.10x |
| | tg128 | 205 | 173 | 163 | 1.19x | 1.25x |
| Qwen3 8B Q4_K_M | pp560 | 4,697 | 3,806 | 3,919 | 1.23x | 1.20x |
| | pp2100 | 4,512 | 4,248 | 3,791 | 1.06x | 1.19x |
| | tg128 | 110 | 104 | 106 | 1.06x | **1.04x** |

Ahead on 21 of 21 cells. The thin ones are in bold. #24 (merged after this board) measured +2.3–23 % on
pp2100 for these models. Llama 3.x and Gemma 3 are not on this board.

## GPU: RunPod RTX 4090 / RTX 5090 / H100 SXM, main 7f42c02, 2026-10-18 14:11 – 15:08

Full tables, hardware and caveats in [benchmarks-gpu.md](benchmarks-gpu.md#runpod-rtx-4090-rtx-5090-h100-sxm-2026-10-18-current).
vs llama.cpp master `ff5a233` (`llama-bench -ngl 99`) and ollama 0.40.6. Ahead of both engines on **45 of 45
cells** (0.5B–7B and Qwen3 8B × pp560/pp2100/tg128 × 3 GPUs), and of llama.cpp `4413a19` on all 15 extra 4090
cells. Ranges over llama.cpp across the five models: tg128 1.03–1.19x (4090), 1.12–1.70x (5090), 1.13–1.30x
(H100); pp2100 1.06–1.77x / 1.26–2.10x / 1.32–2.16x. The thinnest margins:

| GPU | model | pp2100 axon / llama.cpp / ollama | tg128 axon / llama.cpp / ollama |
|---|---|---|---|
| RTX 4090 | 7B | 13,632 / 12,064 / 11,655 (1.13x / 1.17x) | 186 / 181 / 167 (1.03x / 1.11x) |
| RTX 4090 | Qwen3 8B | 12,450 / 11,710 / 11,340 (1.06x / 1.10x) | 170 / 165 / 153 (1.03x / 1.11x) |
| RTX 5090 | 7B | 18,529 / 14,633 / 14,311 (1.27x / 1.29x) | 310 / 276 / 237 (1.12x / 1.30x) |
| H100 SXM | 7B | 14,413 / 9,684 / 9,742 (1.49x / 1.48x) | 314 / 264 / 218 (1.19x / 1.44x) |

The 4090's 7B and 8B decode are only +3 % over llama.cpp (7B +2.5 % before rounding). Not settled: the 5090's
axon prefill is 0.4–8.5 % below the 10-15 board, cross-pod (EPYC 7543 host, driver 580, host load up to 119;
llama.cpp also dropped on that pod). A bella A/B, job `1018-144942-8284`, is **pending**.

Cost (axon-pods guard estimate; RunPod bills per second and not for refused creates, so the real charge is at or
below it): **$2.79** in total. The three boards cost $1.94 (H100 $1.332, 5090 $0.478, 4090 $0.134). The other
$0.85 was waste: an H100 and a 4090 pod killed by a bug in the run script's model-sha check ($0.333 + $0.045), a
4090 whose on-pod llama.cpp builds never finished on an oversubscribed host ($0.373), and 7 refused creates booked
at 1 min each ($0.096). Inside the board pods, about 6 min of the H100's 19.2 min went to the on-pod llama.cpp
build and ollama install, and about 11 min of the 5090's 23.3 min to waiting on the ollama download.

New rule from this run (axon-devstack `62c3178`, BUILDER-RULES.md): **pods never compile.** Every binary a pod
needs (axon at the SHA under test, llama.cpp CUDA for the target card's architecture) is built on bella
beforehand, inside a container matching the pod image, and copied over with the pre-hashed models; a pod is
created only once binaries, models and the run script are ready. Why: on this run the H100 ($3.99/h) and a 4090
compiled llama.cpp on the pod, paid GPU time spent idle (the rule puts it at ~10 min and ~$0.80). The 5090 and
final 4090 boards already used bella-built llama.cpp binaries.

## GPU: RunPod RTX 4090 / RTX 5090 / H100 SXM, main f8d8a86, 2026-10-15 23:53 – 10-16 00:11

Full tables in [benchmarks-gpu.md](benchmarks-gpu.md#previous-board-runpod-rtx-4090-rtx-5090-h100-sxm-2026-10-15). Ahead of
both engines on **45 of 45 cells** (0.5B–7B and Qwen3 8B × pp560/pp2100/tg128 × 3 GPUs). The thinnest margins:

| GPU | model | pp2100 axon / llama.cpp / ollama | tg128 axon / llama.cpp / ollama |
|---|---|---|---|
| RTX 4090 | 7B | 13,842 / 11,937 / 12,313 (1.16x / 1.12x) | 185 / 176 / 174 (1.05x / 1.06x) |
| RTX 4090 | Qwen3 8B | 12,628 / 11,552 / 11,905 (1.09x / 1.06x) | 169 / 161 / 159 (1.05x / 1.06x) |
| RTX 5090 | 7B | 18,609 / 15,416 / 15,832 (1.21x / 1.18x) | 308 / 280 / 279 (1.10x / 1.11x) |
| H100 SXM | 7B | 12,534 / 9,691 / 9,711 (1.29x / 1.29x) | 311 / 241 / 247 (1.29x / 1.26x) |

Caveat from the board: the H100's prefill trails the 5090 and even the 4090, because this build doesn't use
wgmma. #33 added the Hopper path afterwards: on an **H100 PCIe** (not SXM), 7B prefill went 9263 → 10376
(+12.0 %) with `AXON_GEMMH=1`. There is no engine comparison on the H100 PCIe.

## GPU: avante (RTX 4050 Laptop, sm_89, 6 GB), main f8d8a86 vs ollama 0.35.0, 2026-10-16 00:23

Ahead on **23 of 24** cells: pp560 1.53–3.22x, pp2100 1.06–1.23x, tg128 1.05–1.14x. The loss is **Qwen3 0.6B Q8_0
pp2100: 12,730 vs 13,381 (0.95x)**. axon's raw values were 14463 / 12730 / 12706, so it won only round 0.
There is no llama.cpp column (avante has no CUDA llama.cpp build). Table:
[benchmarks-gpu.md](benchmarks-gpu.md#avante-rtx-4050-laptop-sm_89-6-gb).

## Long context (GPU, bella)

From round 9's pick (`r9-pick.md`), main c53f3e4, 2026-10-16, 2 interleaved rounds, cool-gated < 75 °C.
axon `gen` vs `llama-bench -fa 1`. No ollama column.

| model | depth | pp axon / llama.cpp | tg axon / llama.cpp |
|---|---|---|---|
| Qwen2.5 7B | 8192 | 3999 / 3778 (+6 %) | 94.8 / 92.4 (+3 %) |
| | 16384 | 3323 / 3353 (−1 %) | 85.8 / 86.8 (−1 %) |
| | 32512 | 2682 / 2566 (+5 %) | **72.4 / 76.7 (−6 %)** |
| Qwen2.5 0.5B | 8192 | 39185 / 35361 (+11 %) | 667 / 575 (+16 %) |
| | 32512 | 20425 / 18891 (+8 %) | **413 / 464 (−11 %)** |
| Qwen3 8B | 32512 | 2117 / 1902 (+11 %) | 50.0 / 45.0 (+11 %) |

**At 32k, decode lost to llama.cpp on the models with few KV heads** on this board. `--kv q8` didn't help (7B
70.9 at 32k). The fix, lcdec, merged as #50 (2026-10-16 23:41). Its own bella runs at position 32512, vs main: 7B
f16 +3.3 % / +3.5 % (it missed its +12 % gate), 0.5B f16 +10.0 % / +12.1 %, q8 +17.6 % (7B) and +25.7 % (0.5B).
In one of those jobs llama.cpp (`-fa 1`) ran 70.8–71.0 (7B) and 359.5–366.9 (0.5B) at that depth, with main's 7B
at 69.1; bella was slower that afternoon than for this board (main 7B 69 vs 72.4). The pod3 (RTX 5090) numbers
first quoted for lcdec were withdrawn in #50 (its rounds shared the GPU with a test suite). **This table has not
been re-run since.**

Prefill at depth: #56 (merged 10-17 00:31) measured on bella, in its own run against llama.cpp `-fa 1`, 7B ahead
at 8k / 16k / 32k by 4.7 % / 2.5 % / 4.5 % and q3-8B by 7.5 % / 3.1 % / 10 % (before it, 7B was behind at 16k:
3041 vs 3130).

## CPU: bella (Core Ultra 9 275HX, 16 threads on cores 0–15), main f8d8a86, 2026-10-15 19:02

3 rounds, rotated engine order, medians. The machine was **not quiet**: a GPU A/B ran on the other lane,
loadavg median 9.4, and pp spread was ±10–20 %. llama.cpp 4413a19 CPU build (`-fa 1`); ollama 0.35.0
(`num_gpu 0`, 16 threads).

| model | pp512 axon / llama.cpp / ollama | tg64 axon / llama.cpp / ollama |
|---|---|---|
| Qwen2.5 0.5B Q4_K_M | 2512 / 593 / 968 (4.24x / 2.60x) | 171.7 / 96.2 / 145.9 (1.78x / 1.18x) |
| Qwen2.5 1.5B Q4_K_M | 1028 / 404 / 577 (2.54x / 1.78x) | 62.7 / 47.8 / 62.6 (1.31x / **1.00x**) |
| Qwen2.5 3B Q4_K_M | 516 / 212 / 267 (2.43x / 1.94x) | 32.4 / 22.0 / 32.9 (1.47x / **0.98x**) |
| Qwen2.5 7B Q4_K_M | 233 / 77 / 111 (3.02x / 2.09x) | 14.8 / 9.5 / 14.5 (1.56x / 1.02x) |
| Qwen3 0.6B Q8_0 | 1846 / 514 / 1012 (3.59x / 1.82x) | 101.3 / 58.3 / 98.4 (1.74x / 1.03x) |
| Qwen3 1.7B Q4_K_M | 912 / 383 / 518 (2.38x / 1.76x) | 55.5 / 43.7 / 55.6 (1.27x / **1.00x**) |
| Qwen3 4B Q4_K_M | 389 / 163 / 198 (2.39x / 1.97x) | 25.6 / 19.2 / 25.3 (1.33x / 1.01x) |
| Qwen3 8B Q4_K_M | 215 / 76 / 83 (2.85x / 2.60x) | 13.5 / 10.3 / 13.2 (1.31x / 1.02x) |

Prefill leads both engines everywhere. **Decode ties ollama** on 1.5B–8B (0.98–1.03x, inside the ±5 %
decode spread). Both read weights at 61–65 GB/s per token, the DRAM wall. #32 (merged after this board) put
bella CPU decode +2–10 % in its builder's runs; that has not been re-boarded against ollama.

## CPU: jade (i7-9750H, AVX2 only, 6 threads), main f8d8a86 vs llama.cpp, 2026-10-16 01:11

A homelab box under IO load, run at nice 19 inside a 3 GB memory cap. 0.5B: pp512 505.9 vs 248.2 (2.04x),
**tg64 42.0 vs 45.1 (0.93x, a loss)**. 1.5B: pp512 158.1 vs 110.6 (1.43x), tg64 17.4 vs 16.0 (1.09x). The 3B row
is memory-cap-bound and not an engine comparison. No ollama. Board: `sb-jade-cpu.md` (foreman workspace);
summary in [benchmarks-gpu.md](benchmarks-gpu.md#jaded-cpu-only-for-context).

## Beyond single-stream throughput

| feature | measured | where / when |
|---|---|---|
| Draft model, greedy (#27) | tg128 7B 104.5 → 142.8 (+37 %), Qwen3-8B +44 %, Llama-3.1-8B +63 % at K=3; sampled +0.8 % | bella, f8d8a86 base, 10-16 |
| Serve batching (#38) | 7B C=16 355 vs 80 agg tok/s (4.4×), 0.5B C=16 2978 (4.5×); 4090 7B C=16 1017 | bella / 4090 pod, 10-16 |
| Serve TTFT (S1, builder) | 0.5B GPU stream 706 tok/s vs llama-server 560 / ollama 484; TTFT 3.69 ms vs 23.7 / 9.35 | bella, 881d7b6 base, 10-15 (builder's runs, not re-run) |
| Startup (#21, #41) | GPU 7B warm TTFT 732 → 624 ms (#21) and, in a separate run, 614 → 533 ms (#41); CPU 7B warm 5.44 s → 1.74–2.30 s (ollama cold load 3.73 s) | bella, 10-15/16 |
| Startup vs others (r4-startup scout) | GPU short-prompt TTFT axon 0.85 s (7B) vs llama-cli 1.33 s vs ollama cold 1.97 s | bella, 881d7b6, 10-15 (before #21/#41) |
| Hybrid 32B, 28/64 layers (#16) | pp560 ~390 vs llama.cpp `-ngl 28` 211; decode 4.8–5.8 vs 3.5–3.7 | bella, 10-14 |
| Multi-GPU PP (#34) | 2×4090: 7B prefill 1.46×, 32B 1.55× one GPU; decode 1.00× | 4090 pod, 10-16 |

## New models and types since the boards

Each row is the PR's own run against llama.cpp 4413a19 (#63, #67, #78, #89, #98; #72 doesn't state its llama.cpp build), except #127's, which is against Axon's own CPU-expert path. None of these models is on a scoreboard.

| path | PR | measured | host and caveats |
|---|---|---|---|
| `qwen3moe` on the CPU | #63 | 4x0.6B pp560 1542 vs 371, tg128 118.6 vs 48.3; Pruned-2B pp560 4811 vs 2209, tg128 216.8 vs 101.3 | bella CPU, each engine at its best thread count, 3 rounds. Dev models only: the real Qwen3-30B-A3B wasn't run |
| `qwen3moe`, attention on the GPU, experts on the CPU | #67 | vs `llama-server --cpu-moe`: tg128 4x0.6B 157 vs 101, Pruned-2B 674 vs 376; **pp512 4x0.6B 2080 vs 5316, Pruned-2B 8613 vs 12621 (llama.cpp ahead)** | bella, best thread count each, 3 rounds. 30B not run |
| gpt-oss-20b, attention on the GPU, MXFP4 experts on the CPU | #98 | tg128 46.6 vs 23.4 (1.99×, llama.cpp CUDA `-ot exps=CPU`, its best of 8/16/24 threads); **prefill 265 vs 995 (0.27×, llama.cpp ahead)** | bella, 3 rounds; llama.cpp decodes from an empty context, axon after a 486-token prompt. Warm only: a cold first request decoded at 29.5. Experts on the CPU for prefill too (the cause) |
| deepseek2 (Moonlight-16B-A3B Q4_K_M), MLA on the GPU, routed experts on the CPU | #89 | decode 54.0 tok/s, 1.70× llama.cpp `-ot exps=CPU`; **prefill 0.82×** | bella, 3 interleaved rounds. The only deepseek2 model run; under the spec's 55–70 tok/s ESTIMATE |
| deepseek2 (Moonlight-16B-A3B Q4_K_M), routed experts resident on the GPU (`--gpu-experts auto`: 25 of 26 layers, 8.50 GiB) | #127 | vs `--gpu-experts 0` (Axon, experts on the CPU): **tg ×2.06, pp ×3.40**; text identical in all 6 runs | bella, gen -n 128, ctx 4096, 3 interleaved rounds; **TAINTED** (foreign GPU max 31 %, mean 7.8 %), accepted with the label kept: quote the ratios, not the absolute tok/s. No llama.cpp in the run (PR D's bench of record vs `--n-cpu-moe` is next) |
| gpt-oss-20b on the CPU | #78 | pp 130.0 (739-token prompt) vs 104.8 (pp512), 1.24×; tg128 23.5 vs 11.2, 2.10× | bella, exclusive, 24 threads both, neither pinned, llama.cpp's thread count not swept, prompt lengths differ |
| Q2_K / Q3_K on the GPU | #72 | 7B Q3_K_M pp2100 3787 vs 3467 (+9 %), tg128 96.1 vs 98.9 (**−3 %**); 7B Q2_K pp560 2876 vs 3032 (**−5 %**), pp2100 +3 %, tg128 121.9 vs 111.2 (+10 %) | bella GPU vs `llama-bench -ngl 99`, 3 rounds |
| Q2_K / Q3_K on the CPU | #72 | 7B Q2_K pp2100 57.2 vs 97.0 (**−41 %**), tg128 22.2 vs 10.4; 7B Q3_K_M pp2100 87.8 vs 112.2 (**−22 %**), tg128 18.8 vs 9.0 | bella CPU, 16 threads, 3 rounds, on a binary with the merged CPU kernels (0cb1e7a); a confirmation run on the final binary was queued, not a gate |

## Known losses and gaps (as of 39624db)

- **Datacenter serving (H100 SXM, 10-19):** 2.0–3.8× behind vLLM / SGLang on throughput at C = 128 (mix 1,868 vs
  3,770–7,146 tok/s), behind from C = 8 up, TPOT p50 57.9 ms vs 12.7–26.4, 26 % of requests inside the SLO, TTFT p99
  14.5 s. Different weights per row (Q4_K_M vs AWQ / FP8). [Above](#serving-runpod-h100-sxm-main-c15c90e-2026-10-19).
- Serving on the RTX 4090 (10-18): 13–24 % behind vLLM-AWQ at C = 8–32 (since then #86 cut the ≤ 16-row step 5–6 %;
  not re-served).
- MoE prefill with the experts on the CPU: gpt-oss-20b 0.27× and Moonlight 0.82× llama.cpp (#98, #89); see also #67
  below. Moonlight with its experts resident (`--gpu-experts auto`, #127) prefills 3.40× its CPU-expert path
  (TAINTED), but hasn't been run against llama.cpp.
- `serve` runs a MoE model one request at a time (no slot pool for MoE backends).
- Shortest-first admission (`--admit-order sjf`, #131) missed its kill-rule clause on the laptop cells (short TTFT
  p50 0.56× / 0.79× of fifo where ≤ 0.5× was needed) and is opt-in; the H100 cells it was designed for are #139.
- A dense model loaded on the CPU under a memory limit is OOM-killed instead of refused (#135; #133 fixed MoE), and
  an explicit `--parallel N` can come up serving one request at a time without saying why when VRAM is short (#137).

- Long-context decode at 32k on 7B / 0.5B vs llama.cpp (above). #50 merged (+3.3–3.5 % on 7B f16, +10–12 % on 0.5B
  f16 in its own runs); nobody has re-boarded it against llama.cpp.
- Gemma 3 prefill at pp2100 vs llama.cpp on the 4090 (1B 32748 vs 40504; 4B 8782 vs 17920): no tensor-core
  attention at head dim 256 (#35).
- Qwen3 0.6B Q8_0 pp2100 on avante vs ollama (0.95x).
- CPU decode vs ollama: tied (DRAM-bound), not ahead, on 1.5B–8B. jade 0.5B tg loses to llama.cpp (0.93x).
- Sampled speculative decoding gains nothing.
- Multi-GPU: vLLM TP2 beats axon's single stream on 2×4090 (65–70 vs 43 tok/s, 32B AWQ vs Q4_K_M; different quantization).
- Serving on bella's 12 GB at 32k served 7B one request at a time by default since #42 (the N+1 cache layout);
  since #60 (no N+1 cache) it serves 2 at once (`--kv q8`: 5).
- Serve on bella, 7B: the decoders' ITL p99 during an 18.9k-token admit is 154 ms (#55), above the 100 ms target
  #45 met on a 4090. The 256-token chunk floor alone costs about 100 ms per step at that depth on the laptop GPU.
- CPU prefill on Q2_K / Q3_K 7B files trails llama.cpp by 12–41 % (#72): the new types have no repacked GEMM
  layout yet. Q2_K pp560 and Q3_K_M tg128 on the GPU trail by 5 % and 3 %.
- MoE prefill trails llama.cpp `--cpu-moe` 2.6× (4x0.6B) and 1.5× (Pruned-2B) at pp512 (#67): llama.cpp ships the
  big-batch expert products to the GPU (stage 3 here).
- MoE was measured on dev models, gpt-oss-20b and Moonlight-16B-A3B only: no real Qwen3-30B-A3B run, nothing larger.
- Batched speculative decoding in `serve` (#90) was closed by its kill rule: −3.2 % and ITL p99 +67.6 % at 16k
  depth, though +20–25 % at C = 4 on short and 4k contexts ([experiments-rejected.md](experiments-rejected.md#speculation)).
