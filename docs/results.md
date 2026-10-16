# Where Axon stands

The latest measured comparisons against ollama and llama.cpp, per host and model. Every number here is
MEASURED and comes from a scoreboard or a PR. Main is `a591e17` (2026-10-16 16:22 EDT).

> **None of the boards below was run on `a591e17`.** The newest full boards are from main `0851798`
> (bella GPU, 2026-10-15 13:06) and main `f8d8a86` (RunPod, avante, CPU, 2026-10-15/16). Merges since
> then that change speed on those paths (none has been re-boarded):
>
> | merged after | PR | effect on these numbers (from the PR's own A/B) |
> |---|---|---|
> | 0851798 | #24 r5 prefill | GPU pp2100 +2.3 % (7B) to +23 % (0.5B); the bella board predates it; the RunPod/avante boards include it |
> | f8d8a86 | #30 r5 decode | GPU tg128 +2–5 % on small models (bella, builder's runs), ~0 on 7B/8B |
> | f8d8a86 | #33 Hopper | sm_90a only: 7B prefill +12 % on H100 PCIe (`AXON_GEMMH=1`; the default routing was not run on an H100) |
> | f8d8a86 | #32 CPU decode pool cap | bella CPU tg +2–10 % (builder's runs; the exact merged tree wasn't timed) |
> | f8d8a86 | #41, #21 | time to first token only |
> | — | #27, #38, #34, #42 | opt-in or serve-only; `axon gen` output is unchanged (md5) |
>
> Treat the tables as "at least this good on the default path", with the caveat that nobody has
> re-measured that.

Method for all boards: [methodology.md](methodology.md#benchmark-protocol). Full raw tables:
[benchmarks-gpu.md](benchmarks-gpu.md) and [../BENCH.md](../BENCH.md).

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

## GPU: RunPod RTX 4090 / RTX 5090 / H100 SXM, main f8d8a86, 2026-10-15 23:53 – 10-16 00:11

Full tables in [benchmarks-gpu.md](benchmarks-gpu.md#runpod-rtx-4090-rtx-5090-h100-sxm-2026-10-15). Ahead of
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

**At 32k, decode loses to llama.cpp on the models with few KV heads.** `--kv q8` didn't help (7B 70.9 at
32k). The fix (lcdec) is in flight and not on main. Its builder measured, on an RTX 5090 pod, +12 % (7B f16)
to +30 % (0.5B) at 32k vs main; no bella confirmation yet.

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

## Known losses and gaps (as of a591e17)

- Long-context decode at 32k on 7B / 0.5B vs llama.cpp (above). The fix is in flight.
- Gemma 3 prefill at pp2100 vs llama.cpp on the 4090 (1B 32748 vs 40504; 4B 8782 vs 17920): no tensor-core
  attention at head dim 256 (#35).
- Qwen3 0.6B Q8_0 pp2100 on avante vs ollama (0.95x).
- CPU decode vs ollama: tied (DRAM-bound), not ahead, on 1.5B–8B. jade 0.5B tg loses to llama.cpp (0.93x).
- Sampled speculative decoding gains nothing.
- Multi-GPU: vLLM TP2 beats axon's single stream on 2×4090 (65–70 vs 43 tok/s, 32B AWQ vs Q4_K_M; different quantization).
- Serving on bella's 12 GB at 32k serves 7B one request at a time by default since #42 (the N+1 cache layout).
