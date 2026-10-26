# GPU benchmarks

Single-stream throughput of Axon against llama.cpp and ollama on the same GGUF files, a
[mixture-of-experts board](#mixture-of-experts-moonlight-16b-a3b-resident-experts-vs-llamacpp-parabellum-2026-10-26)
(resident experts, Moonlight on the dev laptop), and two serving
(concurrency) sections against vLLM and SGLang: [RTX 4090](#concurrency-rtx-4090-qwen25-7b-2026-10-18) and
[H100 SXM](#concurrency-h100-sxm-qwen25-7b-2026-10-19). Every number on
this page is MEASURED unless labelled ESTIMATED. Single-stream boards: tok/s, median of 3 interleaved rounds; the
concurrency sections state their own rounds. These numbers come from a scoreboard
script that isn't in the repo. For the CPU numbers, the `scripts/bench.py` method, and the full GPU
board on the dev laptop, see [BENCH.md](../BENCH.md).

- `pp560`, `pp2100`: prompt processing over ~560 / ~2100 tokens.
- `tg128`: decode over 128 tokens after a short prompt.
- Models: Qwen2.5 0.5B / 1.5B / 3B / 7B Instruct Q4_K_M and Qwen3 8B Q4_K_M (`q3-8B`). They are the
  Hugging Face files, sha256-checked against the dev laptop's copies.

## RunPod: RTX 4090, RTX 5090, H100 SXM (2026-10-18, current)

Measured 2026-10-18 EDT: H100 14:11–14:19, RTX 5090 14:43–14:59, RTX 4090 15:02–15:08. Same protocol
as the 10-15 board below. Raw logs: `sb.log` (plus `sb-axl.log` / `sb-oll.log` on the 4090) and
`hw.txt` per GPU; they are not in the repo.

### Method

- **Axon:** main at commit `7f42c02`, one release binary built on the dev laptop with the rustc build
  script (sha256 `98885da0ef7d2d0a…`), the same binary on all three GPUs. Run as `axon gen … --device cuda`
  with `AXON_THREADS=4`. NVRTC came from the PyPI wheel that matches the pod's driver.
- **llama.cpp:** upstream master `ff5a233` (2026-10-18), run as `llama-bench -ngl 99 -r 1` once per
  round, other options at their defaults. On the 4090 only, the old pin `4413a19` ran on the same pod as
  a 4th interleaved engine. Where the binaries were built differs by GPU (see caveats).
- **ollama:** 0.40.6, raw generate, `num_gpu 99`, `num_ctx 4096`, `num_thread 8` from the start on all
  three GPUs.
- 3 interleaved rounds, medians. Prompts are salted, so no prompt-cache hits. The GPU was cooled before
  each model. The five GGUFs were sha256-checked against the dev laptop's copies on every pod. Each pod
  was terminated after its board.
- **Secure cloud only:** community-cloud creates were refused.

### Hardware

From each pod's `hw.txt`; "observed under load" is the max of the 1 s GPU sampler, "host load" the
1-minute load average over the pod's run (median / max).

| GPU | $/h | driver / CUDA | power limit | max SM / mem clock | observed under load | host CPU | host load | NVRTC |
|---|---:|---|---|---|---|---|---|---|
| RTX 4090 24 GB | 0.89 | 570.195.03 / 12.8 | 450 W | 3105 / 10501 MHz | 2730 MHz, 365 W, 49 C | EPYC 75F3, 128 vCPU | 16 / 38 | 12.8 |
| RTX 5090 32 GB | 1.19 | 580.159.04 / 13.0 | 575 W | 3090 / 14001 MHz | 2932 MHz, 430 W, 54 C | EPYC 7543, 120 vCPU | 14 / 236 (119 during the axon + llama.cpp board) | 13.0 |
| H100 80GB HBM3 SXM | 3.99 | 595.91.07 / 13.2 | 700 W | 1980 / 2619 MHz | 1980 MHz, 446 W, 42 C | Xeon Platinum 8468, 160 vCPU | 21 / 35 | 13.2 |

Image: `runpod/base:1.4.0-cuda1281-ubuntu2404` on all three.

### Headline

- **Axon leads both engines in all 45 cells** (3 GPUs × 5 models × 3 metrics), and llama.cpp
  `4413a19` in all 15 extra 4090 cells.
- **Decode (tg128):** 1.03–1.19× llama.cpp on the 4090, 1.12–1.70× on the 5090, 1.13–1.30× on the H100.
  vs ollama: 1.11–1.55× / 1.30–2.08× / 1.44–1.85×.
- **Prefill pp2100:** 1.06–1.77× llama.cpp on the 4090, 1.26–2.10× on the 5090, 1.32–2.16× on the H100.
  vs ollama: 1.10–2.16× / 1.28–2.41× / 1.36–2.13×.
- **The thinnest cells are all on the 4090:** 7B tg128 186 vs 181 and Qwen3 8B tg128 170 vs 165 (both
  1.03×; the 7B is +2.5 % before rounding), Qwen3 8B pp2100 12,450 vs 11,710 (1.06×).
- **Prefill pp560:** 1.48–5.34× llama.cpp. vs ollama 1.75–12.05× without the four † cells (with them, up
  to 20.01×). Read the caveats before quoting these.
- **H100 vs 4090:** the H100 now leads the 4090 on 7B and 8B pp2100 (14,413 vs 13,632 and 12,754 vs
  12,450). It still trails it on 0.5B–3B pp2100 (0.5B: 95,522 vs 111,300) and on all five pp560 cells.

† ollama pp560 cell whose 3 samples include a cold first run (see caveats). Unreliable.

### RTX 4090

"llama.cpp" is master `ff5a233`; "llama.cpp 4413a19" is the old pin on the same pod, interleaved.

| model | test | axon | llama.cpp | ollama | axon/llama | axon/ollama | llama.cpp 4413a19 | axon/4413a19 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | pp560 | 103,781 | 26,282 | 8,613 | 3.95× | 12.05× | 25,654 | 4.05× |
| 0.5B | pp2100 | 111,300 | 62,816 | 51,584 | 1.77× | 2.16× | 62,911 | 1.77× |
| 0.5B | tg128 | 1,042 | 875 | 671 | 1.19× | 1.55× | 842 | 1.24× |
| 1.5B | pp560 | 51,326 | 18,398 | 12,435 | 2.79× | 4.13× | 18,726 | 2.74× |
| 1.5B | pp2100 | 52,695 | 35,400 | 32,395 | 1.49× | 1.63× | 34,998 | 1.51× |
| 1.5B | tg128 | 593 | 524 | 447 | 1.13× | 1.33× | 511 | 1.16× |
| 3B | pp560 | 26,508 | 12,662 | 10,960 | 2.09× | 2.42× | 12,064 | 2.20× |
| 3B | pp2100 | 27,442 | 21,593 | 20,128 | 1.27× | 1.36× | 21,297 | 1.29× |
| 3B | tg128 | 349 | 327 | 282 | 1.07× | 1.23× | 321 | 1.09× |
| 7B | pp560 | 13,849 | 9,158 | 7,898 | 1.51× | 1.75× | 9,100 | 1.52× |
| 7B | pp2100 | 13,632 | 12,064 | 11,655 | 1.13× | 1.17× | 12,020 | 1.13× |
| 7B | tg128 | 186 | 181 | 167 | 1.03× | 1.11× | 179 | 1.04× |
| q3-8B | pp560 | 12,718 | 8,584 | 7,210 | 1.48× | 1.76× | 8,565 | 1.48× |
| q3-8B | pp2100 | 12,450 | 11,710 | 11,340 | 1.06× | 1.10× | 11,568 | 1.08× |
| q3-8B | tg128 | 170 | 165 | 153 | 1.03× | 1.11× | 163 | 1.05× |

### RTX 5090

ollama ran in its own pass after axon and llama.cpp (see caveats). Cross-pod drop vs the 10-15 board: see caveats.

| model | test | axon | llama.cpp | ollama | axon/llama | axon/ollama |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | pp560 | 109,562 | 20,530 | 5,520† | 5.34× | 19.85×† |
| 0.5B | pp2100 | 130,896 | 62,389 | 54,234 | 2.10× | 2.41× |
| 0.5B | tg128 | 1,503 | 959 | 724 | 1.57× | 2.08× |
| 1.5B | pp560 | 55,884 | 16,616 | 5,622 | 3.36× | 9.94× |
| 1.5B | pp2100 | 63,607 | 32,802 | 35,067 | 1.94× | 1.81× |
| 1.5B | tg128 | 928 | 545 | 506 | 1.70× | 1.83× |
| 3B | pp560 | 33,397 | 11,339 | 6,362 | 2.95× | 5.25× |
| 3B | pp2100 | 35,110 | 22,864 | 20,995 | 1.54× | 1.67× |
| 3B | tg128 | 569 | 420 | 376 | 1.36× | 1.51× |
| 7B | pp560 | 17,982 | 9,731 | 5,243 | 1.85× | 3.43× |
| 7B | pp2100 | 18,529 | 14,633 | 14,311 | 1.27× | 1.29× |
| 7B | tg128 | 310 | 276 | 237 | 1.12× | 1.30× |
| q3-8B | pp560 | 16,801 | 8,341 | 5,034† | 2.01× | 3.34×† |
| q3-8B | pp2100 | 16,500 | 13,099 | 12,858 | 1.26× | 1.28× |
| q3-8B | tg128 | 286 | 246 | 221 | 1.16× | 1.30× |

### H100 SXM

All three engines interleaved in one board.

| model | test | axon | llama.cpp | ollama | axon/llama | axon/ollama |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | pp560 | 84,470 | 16,526 | 4,222† | 5.11× | 20.01×† |
| 0.5B | pp2100 | 95,522 | 44,186 | 44,935 | 2.16× | 2.13× |
| 0.5B | tg128 | 1,162 | 892 | 630 | 1.30× | 1.85× |
| 1.5B | pp560 | 45,480 | 12,968 | 7,433 | 3.51× | 6.12× |
| 1.5B | pp2100 | 48,824 | 26,767 | 26,443 | 1.82× | 1.85× |
| 1.5B | tg128 | 666 | 591 | 420 | 1.13× | 1.59× |
| 3B | pp560 | 23,804 | 8,564 | 4,270 | 2.78× | 5.57× |
| 3B | pp2100 | 25,853 | 15,713 | 15,300 | 1.65× | 1.69× |
| 3B | tg128 | 451 | 384 | 300 | 1.18× | 1.50× |
| 7B | pp560 | 13,628 | 6,912 | 4,783 | 1.97× | 2.85× |
| 7B | pp2100 | 14,413 | 9,684 | 9,742 | 1.49× | 1.48× |
| 7B | tg128 | 314 | 264 | 218 | 1.19× | 1.44× |
| q3-8B | pp560 | 11,537 | 7,151 | 4,740† | 1.61× | 2.43×† |
| q3-8B | pp2100 | 12,754 | 9,672 | 9,390 | 1.32× | 1.36× |
| q3-8B | tg128 | 299 | 252 | 206 | 1.19× | 1.45× |

### What changed since the 10-15 board

Board to board, each GPU on a different pod (and, for the 5090, a different host CPU) unless stated.
Percentages are medians, this board vs the 10-15 board.

| | RTX 4090 | RTX 5090 | H100 SXM |
|---|---|---|---|
| axon f8d8a86 → 7f42c02, pp560 | −2.0 to +5.1 % | −8.5 to −2.1 % | +1.7 to +12.6 % |
| axon, pp2100 | −3.1 to +0.7 % | −5.8 to −0.4 % | +2.2 to +15.0 % |
| axon, tg128 | −1.0 to +4.1 % | −0.6 to +6.1 % | +0.9 to +9.0 % |
| llama.cpp 4413a19 → master ff5a233, pp560 | +0.3 to +22.5 % | −44.2 to −24.1 % | −11.2 to +19.4 % |
| llama.cpp, pp2100 | +1.1 to +8.2 % | −21.1 to −4.3 % | −0.9 to +0.2 % |
| llama.cpp, tg128 | +1.5 to +8.7 % | −9.3 to −1.5 % | +9.3 to +17.5 % |
| ollama 0.35.0 → 0.40.6, pp560 | +29.7 to +57.6 % | −58.6 to −39.7 % | −67.2 to −29.8 % |
| ollama, pp2100 | −7.7 to −0.9 % | −25.9 to −9.6 % | −2.5 to +1.0 % |
| ollama, tg128 | −6.1 to +3.9 % | −26.4 to −12.8 % | −11.7 to −4.7 % |

- **4090, same pod, llama.cpp `4413a19` → master `ff5a233`** (the clean "llama.cpp moved" number):
  tg128 +1.5 to +3.9 %, pp2100 −0.2 to +1.4 %, pp560 −1.8 to +4.9 %. Axon is flat on the 4090 vs
  `f8d8a86`, so its narrower decode margin there comes mostly from llama.cpp's decode gain plus pod-to-pod noise.
- **H100:** axon is up on every cell, most on prefill (7B pp2100 12,534 → 14,413, +15.0 %). This board
  includes #33's default `wgmma` routing on sm_90a, but the host (Xeon 8468 vs 8470) and driver (595.91
  vs 570.195) changed too, so the gain is not attributed to #33.
- **5090:** see the caveat below. Not claimed as a regression or as noise.

### Caveats (this board)

- **Split ollama pass on the 5090 and 4090.** On those two GPUs ollama ran in its own 3-round pass on
  the same pod, after the axon + llama.cpp board. On the 5090 the ollama download (~1 MB/s) was still
  running during the axon + llama.cpp board. The H100 ran one board with all three engines interleaved.
- **Where llama.cpp was built.** The 5090 and 4090 binaries were prebuilt on the dev laptop inside the
  pod image (`runpod/base:1.4.0-cuda1281-ubuntu2404`): CUDA 12.8, `CMAKE_CUDA_ARCHITECTURES` 120 / 89,
  static, `GGML_NATIVE=OFF` with generic AVX2/FMA/F16C CPU flags. The H100's was built on the pod
  (sm_90, native CPU flags). With `-ngl 99` the CPU code path barely matters, but the builds are not
  identical.
- **ollama cold first runs (†).** On the H100 and the 5090 the round-0 ollama pp560 sample for 0.5B and
  Qwen3 8B is a cold first request after a load: 15 and 66 tok/s on the H100, 14 and 64 tok/s on the
  5090. The median of 3 is then the lower of the two warm samples. Those four cells, and their
  axon/ollama ratios, are unreliable.
- **5090 axon prefill is below the 10-15 board: unattributed.** pp560 −2.1 to −8.5 % (0.5B 119,797 →
  109,562), pp2100 −0.4 to −5.8 %; decode −0.6 to +6.1 %. This is cross-pod: a different host CPU
  (EPYC 7543 vs Ryzen 9 9950X), driver 580.159 vs 595.91, host load up to 119 during the board, and
  llama.cpp is also lower on this pod (pp2100 −4.3 to −21.1 %). The H100 and 4090 show no drop. A bella A/B
  of `f8d8a86` vs `7f42c02` (axon only, 5 interleaved rounds) is queued as job `1018-144942-8284`:
  **pending**.
- **4090:** axon is flat vs `f8d8a86` (−3.1 to +5.1 %); llama.cpp master is +1.5 to +3.9 % on decode vs
  `4413a19` on the same pod.
- **ollama changed version**, 0.35.0 → 0.40.6 (0.40.6 runs a bundled `llama-server` as its runner), so
  the ollama deltas mix the version change with the pod change.
- **pp560 ratios are inflated** the same way as on the 10-15 board (see the general caveats below);
  quote pp2100 for prefill.
- **One run per GPU**, medians of 3; small margins (the 4090's 7B/8B rows) are the least firm.

## Previous board: RunPod RTX 4090, RTX 5090, H100 SXM (2026-10-15)

Superseded by the 2026-10-18 board above; kept as it was.

### Method

- **Axon:** main at commit `f8d8a86`, a release binary built on the dev laptop (no `target-cpu`), run
  as `axon gen … --device cuda` with `AXON_THREADS=4`. NVRTC came from the PyPI wheel that matches the
  pod's driver. Main has since merged the draft model (#27, off by default); this board was not
  re-run after that merge.
- **llama.cpp:** commit `4413a19`, a CUDA build made on each pod for that GPU's architecture only
  (CUDA 12.8), run as `llama-bench -ngl 99 -r 1` once per round.
- **ollama:** 0.35.0, raw generate, `num_gpu 99`, `num_ctx 4096`, `num_thread 8` (see caveats).
- Prompts are salted, so no prompt-cache hits. The GPU was cooled before each model. Nothing was
  pinned (no taskset), and the pods have no desktop. Each pod was terminated after its board.
- **Secure cloud only:** the community cloud had no 4090 or H100 stock, and its 5090 host was broken
  (`cuInit` failed).

### Hardware

| GPU | $/h | driver / CUDA | power limit | max SM / mem clock | observed under load | host CPU | NVRTC |
|---|---:|---|---|---|---|---|---|
| RTX 4090 24 GB | 0.89 | 595.91.07 / 13.2 | 450 W | 3105 / 10501 MHz | 2820 MHz, 387 W, 53 C | EPYC 75F3, 128 vCPU (host load ~100) | 13.2.86 |
| RTX 5090 32 GB | 1.19 | 595.91.07 / 13.2 | 575 W | 3090 / 14001 MHz | 2962 MHz, 525 W, 56 C | Ryzen 9 9950X, 32 vCPU | 13.2.86 |
| H100 80GB HBM3 SXM | 3.99 | 570.195.03 / 12.8 | 700 W | 1980 / 2619 MHz | 1980 MHz, 409 W, 32 C | Xeon Platinum 8470, 208 vCPU | 12.8.93 |

Image: `runpod/base:1.4.0-cuda1281-ubuntu2404` on all three.

### Headline

- **Axon leads both engines in all 45 cells** (3 GPUs × 5 models × 3 metrics).
- **Decode (tg128):** 1.05–1.31× llama.cpp on the 4090, 1.10–1.46× on the 5090, 1.22–1.34× on the H100.
- **Prefill pp2100:** 1.09–1.90× llama.cpp on the 4090, 1.21–1.94× on the 5090, 1.19–2.10× on the H100.
  The smallest margins are the 7B/8B on the 4090: 1.16× and 1.09×.
- **Prefill pp560:** 1.4–4.7× llama.cpp and 1.6–15× ollama. Read the caveats before quoting these.

### RTX 4090

| model | test | axon | llama.cpp | ollama | axon/llama | axon/ollama |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | pp560 | 98,719 | 21,454 | 6,477 | 4.60× | 15.24× |
| 0.5B | pp2100 | 111,428 | 58,571 | 55,863 | 1.90× | 1.99× |
| 0.5B | tg128 | 1,052 | 805 | 652 | 1.31× | 1.61× |
| 1.5B | pp560 | 52,262 | 17,429 | 7,889 | 3.00× | 6.62× |
| 1.5B | pp2100 | 52,332 | 32,719 | 32,695 | 1.60× | 1.60× |
| 1.5B | tg128 | 570 | 499 | 430 | 1.14× | 1.32× |
| 3B | pp560 | 27,056 | 11,156 | 7,096 | 2.43× | 3.81× |
| 3B | pp2100 | 28,313 | 20,335 | 20,485 | 1.39× | 1.38× |
| 3B | tg128 | 344 | 322 | 301 | 1.07× | 1.14× |
| 7B | pp560 | 14,122 | 8,947 | 5,400 | 1.58× | 2.62× |
| 7B | pp2100 | 13,842 | 11,937 | 12,313 | 1.16× | 1.12× |
| 7B | tg128 | 185 | 176 | 174 | 1.05× | 1.06× |
| q3-8B | pp560 | 12,809 | 8,555 | 5,558 | 1.50× | 2.30× |
| q3-8B | pp2100 | 12,628 | 11,552 | 11,905 | 1.09× | 1.06× |
| q3-8B | tg128 | 169 | 161 | 159 | 1.05× | 1.06× |

### RTX 5090

| model | test | axon | llama.cpp | ollama | axon/llama | axon/ollama |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | pp560 | 119,797 | 36,820 | 11,250 | 3.25× | 10.65× |
| 0.5B | pp2100 | 138,908 | 71,454 | 73,151 | 1.94× | 1.90× |
| 0.5B | tg128 | 1,513 | 1,047 | 983 | 1.45× | 1.54× |
| 1.5B | pp560 | 60,612 | 26,384 | 13,586 | 2.30× | 4.46× |
| 1.5B | pp2100 | 66,256 | 41,590 | 42,797 | 1.59× | 1.55× |
| 1.5B | tg128 | 874 | 600 | 677 | 1.46× | 1.29× |
| 3B | pp560 | 35,544 | 17,281 | 11,079 | 2.06× | 3.21× |
| 3B | pp2100 | 36,123 | 25,297 | 25,520 | 1.43× | 1.42× |
| 3B | tg128 | 541 | 451 | 451 | 1.20× | 1.20× |
| 7B | pp560 | 18,363 | 12,815 | 8,774 | 1.43× | 2.09× |
| 7B | pp2100 | 18,609 | 15,416 | 15,832 | 1.21× | 1.18× |
| 7B | tg128 | 308 | 280 | 279 | 1.10× | 1.11× |
| q3-8B | pp560 | 17,223 | 11,114 | 8,353 | 1.55× | 2.06× |
| q3-8B | pp2100 | 16,805 | 13,684 | 14,320 | 1.23× | 1.17× |
| q3-8B | tg128 | 280 | 251 | 253 | 1.12× | 1.10× |

### H100 SXM

| model | test | axon | llama.cpp | ollama | axon/llama | axon/ollama |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | pp560 | 83,074 | 17,532 | 12,890 | 4.74× | 6.44× |
| 0.5B | pp2100 | 93,448 | 44,582 | 44,475 | 2.10× | 2.10× |
| 0.5B | tg128 | 1,066 | 794 | 660 | 1.34× | 1.61× |
| 1.5B | pp560 | 43,424 | 14,120 | 13,210 | 3.08× | 3.29× |
| 1.5B | pp2100 | 45,860 | 26,720 | 26,808 | 1.72× | 1.71× |
| 1.5B | tg128 | 624 | 503 | 463 | 1.24× | 1.35× |
| 3B | pp560 | 22,115 | 9,639 | 9,432 | 2.29× | 2.34× |
| 3B | pp2100 | 24,097 | 15,791 | 15,595 | 1.53× | 1.55× |
| 3B | tg128 | 429 | 351 | 340 | 1.22× | 1.26× |
| 7B | pp560 | 12,100 | 5,790 | 6,987 | 2.09× | 1.73× |
| 7B | pp2100 | 12,534 | 9,691 | 9,711 | 1.29× | 1.29× |
| 7B | tg128 | 311 | 241 | 247 | 1.29× | 1.26× |
| q3-8B | pp560 | 11,052 | 7,124 | 6,749 | 1.55× | 1.64× |
| q3-8B | pp2100 | 11,539 | 9,700 | 9,633 | 1.19× | 1.20× |
| q3-8B | tg128 | 287 | 230 | 227 | 1.25× | 1.27× |

## Concurrency: RTX 4090, Qwen2.5-7B (2026-10-18)

Closed-loop serving on one RunPod RTX 4090 (secure, driver 580.159.04). `scripts/serve_load.py`: C clients, each
sending 2 requests one after the other; ~560-token prompts, 512-token outputs, greedy; aggregate = completion tokens
/ wall. Two rounds per cell, medians. Detail, the per-step curve and methodology:
[log/2026-10-18-c-curve.md](log/2026-10-18-c-curve.md).

| engine | weights | C=1 | C=8 | C=16 | C=32 | C=64 |
|---|---|---|---|---|---|---|
| Axon (feat/c-curve), `--parallel 32` | Q4_K_M GGUF | 181 | 999 | 1542 | 2240 | – |
| llama-server 4413a19, `-np 32 -fa on` | the same GGUF | 133 | 613 | 667 | 976 | – |
| vLLM 0.31.0 (Marlin) | Qwen2.5-7B-Instruct-AWQ int4 | 166 | 1149 | 1958 | 2954 | 3609 |

vLLM runs other weights (AWQ int4, group 128) with other kernels, so it isn't a bits-equal comparison. It is the
datacenter bar on the same card. Axon leads at C=1 and is 2.3× llama-server at C=32. It trails vLLM by 13–24 % at
C=8–32 because its batched step still grows with the rows (ITL p50 7.2 → 11.7 ms from C=8 to C=32, vLLM 6.2 → 8.0).

## Concurrency: H100 SXM, Qwen2.5-7B (2026-10-19)

Closed-loop serving on one RunPod H100 80GB HBM3 SXM (secure, $3.99/h), run `h100-1019-2058`, 2026-10-19 20:58–21:36
EDT. Axon main **c15c90e** (`serve --parallel 128 --ctx 8192`, #71's paged pool: 16,384 pages × 64 = 128 full 8192-token
sequences, f16 KV; the sm_90a kernel cache compiled offline and shipped, so no compile on the pod) against **vLLM
0.32.0** and **SGLang 0.5.22**, each with Qwen2.5-7B-Instruct **AWQ** int4 g128 (`awq_marlin`) and **FP8**
(RedHatAI FP8-dynamic: W8 per-channel, A8 dynamic per-token), `--max-model-len` / `--context-length` 8192,
`--gpu-memory-utilization` / `--mem-fraction-static` 0.80. Greedy everywhere. Method, cost, what went wrong and the
pre-run A/B: [log/2026-10-19-h100-showdown.md](log/2026-10-19-h100-showdown.md). Runbook: axon-devstack
`runbooks/h100-serving-showdown.md`.

- **Weights differ per row.** Axon runs the Q4_K_M GGUF (it has no FP8 path); vLLM / SGLang run AWQ int4 (the
  like-for-like 4-bit rows) and FP8 (their best on Hopper). Engines as deployed, not kernels on equal bits; quality
  wasn't measured.
- **Workloads.** *classic*: the [4090 cell](#concurrency-rtx-4090-qwen25-7b-2026-10-18)'s shape, a 519-token prompt
  and 512 tokens out, N = 2C. *mix*: 104 short chat prompts (256 or 512 out) and 24 long-context prompts of
  1,998–7,665 tokens (256 out), N = max(16, 2C), rotated by (round, C), so small-C mix cells vary round to round for
  every engine. Each prompt starts with a unique salt: no prefix-cache hits on any engine.
- **Rounds.** 2 rounds, engines interleaved per C with the order rotated. Axon ran in both phases (int4 and FP8), so
  its cells are medians of 4, the others' of 2. 124 cells, 11,992 requests, 0 errors, 0 engine drops.
- **SLO** for goodput: TPOT ≤ 50 ms and TTFT ≤ 2 s + 0.25 s per 1k prompt tokens.
- **Cost:** pod $3.09 (46 min). The driver's split: $2.96 paid, $1.46 useful, $1.50 waste (setup 6.5 min, waiting on
  installs 5.6, cold starts 5.25, control 2.55, park / wake 2.0, warm-ups 0.7).

**Headline (MEASURED).** Axon is the fastest at C = 1 (mix 287 vs 235–284 tok/s, classic 304 vs 240–296), but only
by 1 % and 3 % over SGLang-AWQ, which one pod can't settle. From C = 8 up it trails every engine on throughput: at
C = 128, mix 1,868 vs 3,770–7,146 (2.0–3.8×) and classic 4,485 vs 8,288–14,380 (1.8–3.2×). Its ITL p99 is the lowest of
the five only at mix C = 128 (77.5 vs 126–320 ms), because its steps are uniformly slow (ITL p50 58.0 ms vs
5.2–10.8); on classic its ITL p99 is the highest from C = 32 up. Under the SLO it keeps 26 % of mix requests at C = 128
(goodput 513 vs 2,645–7,129 tok/s), with TTFT p99 14.5 s.

Why (ESTIMATED, not profiled on the pod): the batched-decode step at 64–128 rows was tuned on sm_120 / sm_89 and has
no decode-shaped wgmma / Marlin-class kernels on sm_90 (#33's wgmma GEMM, measured for prefill, covers only the Q4_K
products that fill a wave of its 128 × 64 tiles); TPOT p50 on classic grows 3.2 → 26.3 ms from C = 1 to 128 (vLLM-AWQ
4.1 → 8.8). Second, admission and chunked prefill at high C (mix TPOT is 2.2× classic's; the TTFT p99 tail). On the 4090 at C = 32
Axon was 0.76× vLLM-AWQ; here it is 0.43×: vLLM gained 1.9× from the H100, Axon 1.10×. Roadmap:
[Next: proposed order](roadmap.md#next-proposed-order-outside-review-10-19-owner-decides).

### Mix

**mix: aggregate tok/s** (median of rounds; spread = max / min − 1)

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 287 (±1 %) | 795 (±23 %) | 1,324 (±7 %) | 1,695 (±1 %) | 1,868 (±2 %) |
| vLLM-AWQ | 235 (±0 %) | 1,096 (±14 %) | 3,012 (±2 %) | 4,506 (±0 %) | 5,748 (±0 %) |
| SGLang-AWQ | 284 (±1 %) | 1,212 (±22 %) | 2,477 (±5 %) | 3,252 (±2 %) | 3,770 (±1 %) |
| vLLM-FP8 | 239 (±0 %) | 1,103 (±12 %) | 3,328 (±2 %) | 5,374 (±4 %) | 7,146 (±0 %) |
| SGLang-FP8 | 246 (±1 %) | 1,172 (±3 %) | 3,448 (±4 %) | 5,421 (±2 %) | 7,135 (±1 %) |

**mix: goodput tok/s** (tokens of requests inside the SLO / wall)

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 287 | 795 | 1,308 | 1,313 | 513 |
| vLLM-AWQ | 235 | 1,096 | 3,012 | 4,503 | 4,953 |
| SGLang-AWQ | 284 | 1,212 | 2,477 | 3,153 | 2,645 |
| vLLM-FP8 | 239 | 1,103 | 3,328 | 5,374 | 7,057 |
| SGLang-FP8 | 246 | 1,172 | 3,448 | 5,420 | 7,129 |

**mix: fraction of requests inside the SLO**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 1 | 1 | 0.961 | 0.742 | 0.256 |
| vLLM-AWQ | 1 | 1 | 1 | 0.992 | 0.853 |
| SGLang-AWQ | 1 | 1 | 1 | 0.961 | 0.668 |
| vLLM-FP8 | 1 | 1 | 1 | 1 | 0.982 |
| SGLang-FP8 | 1 | 1 | 1 | 0.996 | 0.99 |

**mix: TTFT p50, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 14.6 | 43.8 | 204 | 464 | 2,361 |
| vLLM-AWQ | 26.5 | 88.7 | 224 | 392 | 649 |
| SGLang-AWQ | 18.1 | 39.5 | 169 | 564 | 1,161 |
| vLLM-FP8 | 27.5 | 102 | 199 | 391 | 646 |
| SGLang-FP8 | 20.0 | 42.7 | 83.3 | 250 | 452 |

**mix: TTFT p99, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 556 | 780 | 3,345 | 5,867 | 14,513 |
| vLLM-AWQ | 257 | 260 | 840 | 1,494 | 3,062 |
| SGLang-AWQ | 389 | 414 | 1,473 | 2,548 | 5,266 |
| vLLM-FP8 | 188 | 217 | 612 | 1,076 | 2,114 |
| SGLang-FP8 | 170 | 181 | 553 | 984 | 1,920 |

**mix: ITL p50, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 3.2 | 5.7 | 11.5 | 15.9 | 58.0 |
| vLLM-AWQ | 4.1 | 4.2 | 5.1 | 6.1 | 8.4 |
| SGLang-AWQ | 3.3 | 3.6 | 5.3 | 7.5 | 10.8 |
| vLLM-FP8 | 4.0 | 4.2 | 5.1 | 5.6 | 8.0 |
| SGLang-FP8 | 4.0 | 4.3 | 4.9 | 5.7 | 5.2 |

**mix: ITL p99, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 3.6 | 58.2 | 68.9 | 74.0 | 77.5 |
| vLLM-AWQ | 4.5 | 10.7 | 105 | 189 | 205 |
| SGLang-AWQ | 3.7 | 7.8 | 179 | 191 | 320 |
| vLLM-FP8 | 4.5 | 9.9 | 69.1 | 123 | 140 |
| SGLang-FP8 | 4.2 | 7.6 | 24.3 | 65.8 | 126 |

**mix: TPOT p50, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 3.2 | 7.6 | 21.1 | 32.4 | 57.9 |
| vLLM-AWQ | 4.1 | 4.6 | 7.2 | 10.6 | 16.7 |
| SGLang-AWQ | 3.3 | 4.7 | 9.9 | 15.7 | 26.4 |
| vLLM-FP8 | 4.0 | 4.5 | 6.7 | 8.3 | 12.7 |
| SGLang-FP8 | 3.9 | 4.7 | 6.6 | 9.0 | 12.9 |

**mix: completion tokens per request, mean**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 278 | 271 | 266 | 276 | 278 |
| vLLM-AWQ | 278 | 270 | 256 | 273 | 273 |
| SGLang-AWQ | 272 | 270 | 258 | 270 | 271 |
| vLLM-FP8 | 285 | 274 | 262 | 278 | 278 |
| SGLang-FP8 | 283 | 276 | 265 | 278 | 279 |

**mix, control pass (restarted without the sleep / memory-saver flags; one cell each): aggregate tok/s**

| engine | C=32 | C=128 |
|---|---|---|
| vLLM-AWQ | 2,886 | 5,741 |
| SGLang-AWQ | 2,392 | 3,769 |

**mix, control pass: ITL p50, ms**

| engine | C=32 | C=128 |
|---|---|---|
| vLLM-AWQ | 5.3 | 8.4 |
| SGLang-AWQ | 5.4 | 10.8 |

### Classic

**classic: aggregate tok/s** (median of rounds; spread = max / min − 1)

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 304 (±1 %) | 1,278 (±0 %) | 2,466 (±1 %) | 3,566 (±3 %) | 4,485 (±3 %) |
| vLLM-AWQ | 240 (±0 %) | 1,786 (±0 %) | 5,682 (±0 %) | 8,699 (±0 %) | 12,190 (±0 %) |
| SGLang-AWQ | 296 (±0 %) | 2,015 (±0 %) | 4,913 (±0 %) | 6,342 (±0 %) | 8,288 (±0 %) |
| vLLM-FP8 | 242 (±0 %) | 1,798 (±1 %) | 5,856 (±0 %) | 10,047 (±1 %) | 14,205 (±1 %) |
| SGLang-FP8 | 250 (±0 %) | 1,806 (±0 %) | 6,149 (±0 %) | 9,888 (±0 %) | 14,380 (±0 %) |

**classic: goodput tok/s**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 304 | 1,278 | 2,466 | 2,981 | 3,031 |
| vLLM-AWQ | 240 | 1,786 | 5,682 | 8,699 | 12,190 |
| SGLang-AWQ | 296 | 2,015 | 4,913 | 6,342 | 5,390 |
| vLLM-FP8 | 242 | 1,798 | 5,856 | 10,047 | 14,205 |
| SGLang-FP8 | 250 | 1,806 | 6,149 | 9,888 | 14,380 |

**classic: fraction of requests inside the SLO**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 1 | 1 | 1 | 0.836 | 0.676 |
| vLLM-AWQ | 1 | 1 | 1 | 1 | 1 |
| SGLang-AWQ | 1 | 1 | 1 | 1 | 0.651 |
| vLLM-FP8 | 1 | 1 | 1 | 1 | 1 |
| SGLang-FP8 | 1 | 1 | 1 | 1 | 1 |

**classic: TTFT p50, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 48.2 | 113 | 101 | 102 | 115 |
| vLLM-AWQ | 38.8 | 129 | 306 | 440 | 875 |
| SGLang-AWQ | 35.5 | 160 | 578 | 930 | 1,525 |
| vLLM-FP8 | 47.5 | 119 | 262 | 338 | 581 |
| SGLang-FP8 | 24.7 | 67.8 | 175 | 308 | 554 |

**classic: TTFT p99, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 48.2 | 360 | 1,544 | 3,123 | 6,570 |
| vLLM-AWQ | 38.8 | 135 | 422 | 759 | 1,498 |
| SGLang-AWQ | 35.5 | 198 | 702 | 1,409 | 2,811 |
| vLLM-FP8 | 47.5 | 137 | 305 | 538 | 1,051 |
| SGLang-FP8 | 24.7 | 81.4 | 261 | 486 | 956 |

**classic: ITL p50, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 3.2 | 5.6 | 10.3 | 12.8 | 18.6 |
| vLLM-AWQ | 4.1 | 4.2 | 4.8 | 5.8 | 5.1 |
| SGLang-AWQ | 3.3 | 3.6 | 5.1 | 7.3 | 9.8 |
| vLLM-FP8 | 4.1 | 4.2 | 4.9 | 5.3 | 5.7 |
| SGLang-FP8 | 4.0 | 4.3 | 4.7 | 5.4 | 4.9 |

**classic: ITL p99, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 3.2 | 6.3 | 51.7 | 53.6 | 62.0 |
| vLLM-AWQ | 4.5 | 4.9 | 6.9 | 9.5 | 38.6 |
| SGLang-AWQ | 3.6 | 4.1 | 6.2 | 9.0 | 20.2 |
| vLLM-FP8 | 4.5 | 4.9 | 6.6 | 9.9 | 38.8 |
| SGLang-FP8 | 4.2 | 4.7 | 5.2 | 8.7 | 39.7 |

**classic: TPOT p50, ms**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 3.2 | 6.0 | 12.6 | 17.2 | 26.3 |
| vLLM-AWQ | 4.1 | 4.2 | 5.0 | 6.5 | 8.8 |
| SGLang-AWQ | 3.3 | 3.7 | 5.4 | 8.3 | 12.4 |
| vLLM-FP8 | 4.1 | 4.2 | 5.0 | 5.7 | 7.8 |
| SGLang-FP8 | 4.0 | 4.3 | 4.9 | 5.8 | 7.8 |

**classic: completion tokens per request, mean**

| engine | C=1 | C=8 | C=32 | C=64 | C=128 |
|---|---|---|---|---|---|
| Axon (Q4_K_M) | 512 | 512 | 512 | 512 | 512 |
| vLLM-AWQ | 512 | 512 | 512 | 512 | 512 |
| SGLang-AWQ | 512 | 512 | 512 | 512 | 512 |
| vLLM-FP8 | 512 | 512 | 512 | 512 | 512 |
| SGLang-FP8 | 512 | 512 | 512 | 512 | 512 |

## Mixture of experts: Moonlight-16B-A3B, resident experts vs llama.cpp (Parabellum, 2026-10-26)

The bench of record for `--gpu-experts` (MoE-resident PR D, roadmap item 3).
- Axon main `39624db` runs at `--gpu-experts 0 / 6 / 12 / auto`. llama.cpp CUDA `4413a19` runs at the same
  placement (`--n-cpu-moe 26 − N`).
- Moonlight-16B-A3B-Instruct Q4_K_M, ctx 4096.
- Medians of 3 interleaved rounds. A † marks a run that is **TAINTED** (foreign GPU load during that run: mean
  > 5 % or max > 25 %). The job as a whole is TAINTED (max 46 %, mean 3.1 %), with 17 of 72 runs tainted.
- MEASURED, job `1025-225107-e28c`; raw results, job log and script are in the [gist](https://gist.github.com/HaseebKhalid1507/7c37456c09d035b98d2f2b7d7b21bf51).
  Log: [2026-10-26-moe-resident-bench](log/2026-10-26-moe-resident-bench.md).

**Head-to-head at full residency, up front: llama.cpp is ahead.** At `auto` (25 or 26 of 26 expert layers on the
GPU), Axon decodes at **0.71×** llama.cpp (121.7 vs 171.0 tok/s). Prefill is **0.45×** at pp512 and **0.34×** at
pp2100. All of llama.cpp's auto runs are clean.

Axon leads decode only while most experts are on the CPU: 1.72× at N = 0, 1.62× at N = 6, 1.44× at N = 12. Those
rows are biased in Axon's favour by thread count. llama-bench ran at its default `-t 4`; Axon uses 8 threads for
decode and 24 for prefill. See the caveats.

**Accept rule (spec): PASS.** It asks for `auto` vs N = 0 of at least +50 % on decode and at least +100 % on pp512.
Axon gets tg128 ×2.23 (54.5 → 121.7) and pp512 ×4.02 (594 → 2,386). Residency does what it promised against
Axon's own CPU-expert split. Against llama.cpp at the same placement it doesn't win.

| N (resident) | metric | axon median (r1 / r2 / r3) | llama.cpp median (r1 / r2 / r3) | axon / llama | tainted rows (axon, llama) |
|---|---|---|---|---:|---|
| 0 | tg128 | **54.5** (50.4 / 54.5 / 54.9) | **31.6** (33.6† / 27.5† / 31.6) | 1.72× | 0, 2 |
| 0 | pp512 | **594.1** (641.7 / 412.6 / 594.1) | **795.6** (727.3† / 795.6† / 827.5) | 0.75× | 0, 2 |
| 0 | pp2100 | **589.7** (589.7† / 583.1 / 595.7†) | **674.1** (639.7† / 674.1† / 687.5) | 0.87× | 2, 2 |
| 6 | tg128 | **66.6** (66.6 / 63.3 / 67.6) | **41.2** (41.2 / 41.2 / 40.9†) | 1.62× | 0, 1 |
| 6 | pp512 | **703.1** (703.1 / 625.3 / 715.8) | **997.3** (1061.2 / 997.3 / 996.9†) | 0.70× | 0, 1 |
| 6 | pp2100 | **699.2** (699.2 / 703.0 / 696.2†) | **891.9** (891.9 / 889.1 / 906.5†) | 0.78× | 1, 1 |
| 12 | tg128 | **74.0** (74.5 / 74.0 / 73.9) | **51.4** (51.4 / 49.8 / 52.7) | 1.44× | 0, 0 |
| 12 | pp512 | **1014.1** (1028.3 / 1014.1 / 930.6†) | **1382.0** (1382.0 / 1240.0 / 1415.4) | 0.73× | 1, 0 |
| 12 | pp2100 | **838.9** (840.5 / 834.0 / 838.9†) | **1203.3** (1201.8 / 1203.3 / 1260.5) | 0.70× | 1, 0 |
| auto (25/25/26) | tg128 | **121.7** (121.0 / 121.7 / 127.5) | **171.0** (170.6 / 177.8 / 171.0) | 0.71× | 0, 0 |
| auto (25/25/26) | pp512 | **2386.1** (2356.5 / 2386.1† / 2652.6†) | **5321.1** (5333.8 / 5260.3 / 5321.1) | 0.45× | 2, 0 |
| auto (25/25/26) | pp2100 | **1588.0** (1580.5† / 1588.0 / 1696.4) | **4680.5** (4680.5 / 4695.6 / 4670.6) | 0.34× | 1, 0 |

"auto (25/25/26)": the resident count in rounds 1 / 2 / 3. `auto` places as many layers as fit after a ctx-4096
cache in the VRAM free at load, and the desktop's share of the 12 GB varies. llama.cpp ran each round at that
round's count (`--n-cpu-moe` 1 / 1 / 0).

**Thread-matched rerun (MEASURED, job `1026-011428-edef`, N = 0 / 6 / 12, 3 interleaved rounds).** llama-bench at
`-t 8` (Axon's decode threads, the P cores) and `-t 24` (Axon's prefill threads), interleaved with Axon. The job is
TAINTED overall (Xorg, max 53 %). Its FOREIGN_GPU samples put two spikes at 02:26:17–02:26:51 and 02:27:25–02:27:59.
The six runs that overlapped them are marked * and left out of the medians. Every other run is a clean window. The
* runs are being rerun (job `1026-023845-8bc8`).

| N | metric | axon | llama.cpp `-t 8` | llama.cpp `-t 24` | axon / best llama |
|---|---|---|---|---|---:|
| 0 | tg128 | 56.6 (53.4 / 56.6 / 62.5) | 34.3 (35.4* / 33.6 / 35.0) | 9.0 (9.5* / 8.8 / 9.2) | 1.65× |
| 0 | pp512 | 700.1 (700.1 / 705.5 / 641.5) | 780.4 (812.0* / 719.1 / 841.6) | 749.4 (704.1* / 795.6 / 703.2) | 0.90× |
| 0 | pp2100 | 611.2 (600.9 / 612.9 / 611.2) | 695.8 (663.5* / 645.5 / 746.0) | 690.1 (647.0* / 699.1 / 681.1) | 0.88× |
| 6 | tg128 | 68.0 (66.4 / 68.0 / 68.9) | 42.5 (46.6* / 38.4 / 46.5) | 11.7 (11.2* / 11.4 / 12.0) | 1.60× |
| 6 | pp512 | 836.8 (838.0 / 836.8 / 815.2) | 1054.9 (947.7* / 1064.5 / 1045.2) | 1026.9 (914.1* / 1046.1 / 1007.7) | 0.79× |
| 6 | pp2100 | 705.3 (706.6 / 696.2 / 705.3) | 932.7 (902.8* / 930.2 / 935.1) | 925.6 (876.8* / 924.1 / 927.0) | 0.76× |
| 12 | tg128 | 75.2 (78.2* / 76.4 / 74.0) | 52.5 (52.5 / 55.8 / 51.7) | 16.1 (16.1 / 16.0 / 17.5) | 1.43× |
| 12 | pp512 | 1045.9 (982.3* / 1047.7 / 1044.1) | 1324.0 (1324.0 / 1337.1 / 1304.2) | 1244.3 (1350.3 / 1244.3 / 1079.3) | 0.79× |
| 12 | pp2100 | 849.1 (846.9 / 852.5 / 849.1) | 1179.3 (1179.3 / 1230.8 / 1078.5) | 1200.4 (1201.6 / 1200.4 / 1062.6) | 0.71× |

Matching threads doesn't change the picture.
- llama.cpp decodes best at `-t 8`. At `-t 24` its decode collapses (9–17 tok/s), presumably because the hybrid CPU's
  E cores stall each step's barrier (not measured).
- At `-t 8`, Axon still leads decode by 1.43–1.65×. llama.cpp's prefill is ahead by 1.11–1.41× at either thread
  count.
- So llama-bench's default `-t 4` understated llama.cpp's decode by up to 9 % at these placements (N = 0: 31.6 vs
  34.3), not enough to flip any row.

**Method.** One `--timing bella` job (`benchd.sh`, in the gist). Each round runs N = 0, 6, 12, auto; within a
round, the engine order alternates (odd rounds Axon first). Every run is preceded by a 5 s pause.
- **Axon:** `axon gen <model> "<prompt>" --device cuda --ctx 4096 --gpu-experts N`, with default threads (decode
  on the 8 P cores, prefill on all 24).
  - tg128: 128 tokens after an 8-token prompt.
  - pp512 / pp2100: salted copies of `scripts/bench.py`'s sentence, 510 and 2,106 tokens, `-n 1`.
  - The rates are `gen`'s own `prompt eval` / `decode` lines.
- **llama.cpp:** `llama-bench -m <model> -ngl 99 -ncmoe (26−N) -p 512,2100 -n 128 -r 1 -o jsonl`, at its
  defaults otherwise: `-t 4`, flash attention `auto` (`fa=-1` in its output), warm-up on. Build: master
  `4413a19`, Release, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120`.
- **Taint:** `nvidia-smi pmon` every second during each run; any process other than axon or llama-bench counts as
  foreign.

**Hardware.** Parabellum (bella): Intel Core Ultra 9 275HX (8 P + 16 E), 30 GB DDR5, RTX 5070 Ti Laptop
(sm_120, 12 GB) shared with the desktop. Driver 615.78.08, CUDA / NVRTC 13, CachyOS (Linux 7.2.11).

**Model.** `Moonlight-16B-A3B-Instruct-Q4_K_M.gguf`, sha256
`2b113655999f11730a6dc31b7ba93463e6523a323f46074b01c55d96101df360`.
- 26 MoE layers of 64 experts, top-6, plus 2 shared experts.
- gate/up Q4_K; down Q8_0 in 12 layers and Q5_0 in 14.

**Caveats.**
- **Threads.** llama-bench's default is `-t 4`, against Axon's 8 decode / 24 prefill threads. At N < 26
  llama.cpp's CPU expert layers run on 4 threads, so its N = 0 / 6 / 12 rows are understated and Axon's decode
  lead there is partly thread count. The `auto` rows hardly depend on it: one layer on the CPU in rounds 1–2,
  none in round 3. The thread-matched rerun above settles it: llama.cpp's best is `-t 8`, and Axon still
  leads decode at N = 0 / 6 / 12.
- **Attention.** llama.cpp chooses flash attention itself (`auto`); Axon runs its own MLA kernels. Both are each
  engine's default.
- **Prompts.** Axon's prompts are real text: 510 / 2,106 tokens, and its tg128 starts after an 8-token prompt.
  llama-bench uses synthetic prompts of exactly 512 / 2,100 tokens, and its tg128 starts from an empty context.
  The pp figures are rates, so a 2-token difference doesn't matter.
- **Bits.** Axon's resident experts are bit-identical to its CPU experts (the twin kernels; PR #127's gates).
  llama.cpp's GPU and CPU expert paths are not bit-identical to each other. The speed gap is the subject of the
  diagnosis in the log entry.

## Laptops

### Parabellum: RTX 5070 Ti Laptop (sm_120, 12 GB)

The dev laptop's board (main `0851798`, 2026-10-15, 7 models, vs ollama 0.35.0 and llama.cpp
`4413a19`) is in [BENCH.md](../BENCH.md#2026-10-15-gpu-main--0851798) and the README. Axon led in all
21 cells. The thinnest margins were 7B pp2100 (+2% vs ollama) and 7B tg128 (+5%). That board predates
the round-5 prefill changes now on main. For scale, Axon 7B there: pp560 5,040 / tg128 118. On the
RunPod 4090 it was 14,122 / 185.

### Avante: RTX 4050 Laptop (sm_89, 6 GB)

Avante is a laptop with an i5-13420H and an RTX 4050 Laptop GPU (6141 MiB, driver 610.43.02 / CUDA
13.3, NVRTC 13.3.33, 54–68 W under load). Measured 2026-10-16, main `f8d8a86` vs ollama 0.35.0 (CUDA
backend, every layer offloaded). 3 rounds, engine order rotated each round, salted prompts. ollama
used temperature 0, top_k 1, `num_ctx 4096`, and unloaded between engines. There is no llama.cpp CUDA
build on this machine.

| model | pp560 axon | ollama | ratio | pp2100 axon | ollama | ratio | tg128 axon | ollama | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5 0.5B Q4_K_M | 18,014 | 5,596 | 3.22× | 18,363 | 17,329 | 1.06× | 322.2 | 282.5 | 1.14× |
| Qwen2.5 1.5B Q4_K_M | 9,332 | 4,754 | 1.96× | 8,414 | 6,842 | 1.23× | 156.5 | 143.4 | 1.09× |
| Qwen2.5 3B Q4_K_M | 4,623 | 2,804 | 1.65× | 4,269 | 3,732 | 1.14× | 86.9 | 82.6 | 1.05× |
| Qwen3 0.6B Q8_0 | 13,533 | 8,849 | 1.53× | 12,730 | 13,381 | **0.95×** | 233.0 | 204.2 | 1.14× |
| Qwen3 1.7B Q4_K_M | 9,058 | 4,438 | 2.04× | 7,606 | 6,568 | 1.16× | 146.9 | 135.0 | 1.09× |
| Qwen3 4B Q4_K_M | 3,622 | 2,270 | 1.60× | 3,153 | 2,811 | 1.12× | 68.0 | 63.3 | 1.07× |
| Llama 3.2 1B Q4_K_M | 11,806 | 5,717 | 2.07× | 11,234 | 9,781 | 1.15× | 206.7 | 191.2 | 1.08× |
| Llama 3.2 3B Q4_K_M | 4,854 | 2,858 | 1.70× | 4,309 | 3,800 | 1.13× | 85.2 | 80.5 | 1.06× |

Axon leads in 23 of 24 cells. The loss is real: **Qwen3 0.6B Q8_0 pp2100 at 0.95×**. Axon won only
its first round there (raw 14,463 / 12,730 / 12,706 vs ollama's 13,404 / 13,376 / 13,381). On
Parabellum the same cell is a narrow win (1.03×). Q8_0 long prefill on sm_89 is an open item. The
decode lead is smaller than on Parabellum: 3B tg128 is 1.05× here and 1.12× there.

### Jaded: CPU only (for context)

Jaded is a homelab box with an AVX2-only i7-9750H (6 cores, no AVX-VNNI) and a GTX 1050 (sm_61). Main
has no tested sm_61 path, so this board is CPU only: main `f8d8a86` vs llama.cpp `4413a19` (CPU build),
6 threads pinned to the physical cores, 2026-10-16. The box was serving disk-heavy homelab jobs (IO
pressure 78–91%), and the benchmark ran at nice 19 / ionice idle under a 3 GB memory cap.

| model | pp512 axon | llama.cpp | ratio | tg64 axon | llama.cpp | ratio |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 0.5B Q4_K_M | 505.9 | 248.2 | 2.04× | 42.0 | 45.1 | 0.93× |
| Qwen2.5 1.5B Q4_K_M | 158.1 | 110.6 | 1.43× | 17.4 | 16.0 | 1.09× |

Prefill leads, and decode is roughly a tie: both engines are memory-bound on dual-channel DDR4.
Axon's 0.5B decode raw values spread 36.6–47.0, while llama.cpp's spread 44.4–45.2. The 3B row was
left out: the model didn't fit the memory cap and was re-read from disk, so its numbers rank the
disk, not the engines.

## Caveats

Written for the 10-15 RunPod board and the laptop boards; the 10-18 board's own caveats are in its section.

- **pp560 ratios are inflated.** The other engines' short-prompt rate is far below their long-prompt
  rate. For example, ollama on the 4090 0.5B ran 6,477 at pp560 and 55,863 at pp2100; Axon's pp560 is
  close to its pp2100. That points to a fixed per-request cost in llama-bench and ollama that dominates
  a 560-token prompt, which the source logs didn't break down. Each engine is also timed differently:
  Axon's own prompt timing from `gen`, llama-bench's synthetic prompt from an empty context, and
  ollama's API timings. **Quote pp2100 for prefill.**
- **H100 prefill is weak.** The H100 beats the 5090 only on 7B/8B decode (HBM3: 7B tg128 311 vs 308).
  Its prefill trails the 5090 and even the 4090 (7B pp2100: 12,534 vs 18,609 and 13,842). Axon's
  int8 GEMM uses `mma.sync`, not Hopper's `wgmma`, and the H100's clock is 1980 MHz. Analysis and plan:
  [design/hopper.md](design/hopper.md).
- **ollama thread count.** On the 4090 host (128 shared vCPUs, load average ~100), ollama's default
  thread count stalled it: 0.5B pp560 1,383 / tg128 245. With `num_thread 8` it reached
  6,477 / 652. The 4090 and 5090 ollama columns come from an ollama-only re-run on the same pod right
  after the main board. On the quiet 5090 host the two runs agree within 1–3%. The H100 ran with 8
  threads from the start.
- **ollama outliers on Avante.** Its first measurement after a load was sometimes very low (0.5B pp560
  raw 23 tok/s). Medians absorb these. Best vs best, the 0.5B pp560 ratio is 2.95×, not 3.22×.
- **Medians of 3** rounds per cell; raw values are in the source logs. Three rounds catch a single
  outlier but say little about run-to-run spread, so small margins (the 7B/8B rows) are the least firm.
- **Shared, unpinned pods.** Secure-cloud pods share the host CPU with other tenants. Nothing was
  pinned. Laptop GPUs are power-capped (the RTX 4050 ran at 54–68 W, with the SM clock varying
  between 2280 and 2520 MHz).
- **One run per GPU.** Each pod board was taken once, on one host.
