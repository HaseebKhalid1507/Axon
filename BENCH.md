# Benchmarks

Measured with `scripts/bench.py`, which runs every binary (and optionally the local
ollama server) in turn, round after round, and reports the median (min–max).
Compare binaries within one run; numbers across runs depend on what else the
machine was doing.

- `pp560`, `pp2100`: prompt evaluation, tok/s, ~560 / ~2100-token prompt
- `tg128`: decode tok/s over 128 tokens; `tg128@1`: the same on one thread
- `--load N`: N busy-loop processes running alongside (a machine that isn't idle)
- `--warmup N` (default 1): discarded rounds first, so model loads never land in a sample
- `--ollama NAME`, `--llama-server PATH`: reference columns (llama-server: one
  server per thread count, `-c 4096 -np 1 -ngl 0`, `/completion` with `cache_prompt: false`)
- `LABEL=PATH[,ENV=VAL...]` as a binary sets its column label and environment,
  e.g. `att-t1=target/release/axon,AXON_ATT_TILE=1`
- `--json PATH` saves raw samples, medians and machine metadata; `--dry-run`
  prints the commands/requests without running anything
- `scripts/sweep.py` runs pp560/tg128 for every engine at each of `--threads-list`
  and reports each engine at its own best thread count

Binaries are the default build: baseline x86-64 with runtime AVX2/VNNI dispatch
(no `target-cpu`; before the portability change the default was x86-64-v3).

Model: Qwen2.5-0.5B-Instruct Q4_K_M (see CONTRIBUTING.md). Machine "bella":
Intel Core Ultra 9 275HX (8 P + 16 E cores, AVX2 + AVX-VNNI, no AVX-512), DDR5-5600,
18 threads unless noted.

## 2026-10-13, main @ 74fe85e (M3 step 6)

Quiet machine, 3 rounds, vs ollama 0.35.0 (its bundled llama-server, llama.cpp 631109b34):

| engine | pp560 | pp2100 | tg128 | tg128@1 |
|---|---|---|---|---|
| axon 74fe85e | 1294.7 (1257–1315) | 1068.8 (978–1087) | 163.8 (163–176) | 29.9 (30–30) |
| ollama 0.35.0 | 1027.3 (990–1028) | 961.2 (952–970) | 161.6 (152–164) | 34.4 (34–35) |

With 6 busy-loop processes alongside (`--load 6`):

| engine | pp560 | tg128 | tg128@1 |
|---|---|---|---|
| axon 74fe85e | 370.1 (317–456) | 31.9 (30–33) | 28.8 (29–29) |

Decode at 18 threads reads ~392 MB of weights per token; 164 tok/s is ~64 GB/s,
about the practical DRAM bandwidth measured on this machine (63–69 GB/s).

## 2026-10-13, overnight/2026-10-13 @ aa9be77

Each engine at its own best thread count (`scripts/sweep.py`, threads 6/8/12/18/24,
2 rounds, quiet machine). Same GGUF file for every engine; ollama models are
created from it (`ollama create` with `FROM <gguf>`). llama.cpp is master
4413a19 (Release, `GGML_NATIVE=ON`, OpenMP).

| model | engine | pp560 (threads) | tg128 (threads) |
|---|---|---|---|
| 0.5B Q4_K_M | axon | **1345.4** (18) | **192.1** (12) |
| | ollama 0.35.0 | 1037.2 (18) | 162.6 (18) |
| | llama.cpp master | 597.0 (24) | 64.6 (8) |
| 1.5B Q4_K_M | axon | 382.2 (18) | **71.8** (18) |
| | ollama 0.35.0 | **396.9** (18) | 67.7 (18) |
| | llama.cpp master | 337.9 (24) | 34.0 (24) |

llama.cpp master's own `llama-bench` agrees (0.5B, 8 threads pinned to the P-cores:
pp512 581, tg128 85; unpinned: 380 / 75): on this hybrid CPU its OpenMP build
is well behind ollama's bundled engine.

Fixed 18 threads, 3 rounds (`bench.py`, drafting off), axon / ollama:

| model | pp560 | pp2100 | tg128 | tg128@1 |
|---|---|---|---|---|
| 0.5B | 1014 / 729 | 764 / 649 | 164 / 151 | 54.6 / 30.3 |
| 1.5B | 354 / 387 | 309 / 349 | 72.0 / 66.4 | 23.7 / 12.4 |
| 3B | 271 / 276 | | 39.8 / 37.3 | 12.8 / 8.1 |
| 7B | 131 / 90 | | 17.9 / 16.9 | |

Prompt-lookup drafting (`--draft 7`, 0.5B, 8 rounds, decode tok/s, off / on):
prose 196 / 233, story 194 / 186, code rewrite 176 / 420, summarize 180 / 313,
random list 192 / 401. It loses on low-repetition text, so it stays off by default.

Small verify batches (2-9 tokens) are compute-bound on the E-cores while the
P-cores wait, so they now use 16 pool chunks per thread instead of 4 (output
unchanged). Decode tok/s, 18 threads, no drafting / `--draft 7` before / after:

| prompt | 0.5B (5 rounds) | 1.5B (4 rounds) |
|---|---|---|
| prose | 183 / 216 / 226 | 71.2 / 73.9 / 75.6 |
| story | 182 / 173 / 180 | 67.8 / 65.6 / 66.7 |
| code rewrite | 168 / 418 / 403 | 64.9 / 219 / 229 |
| summarize | 169 / 319 / 332 | 65.3 / 71.5 / 75.0 |
| random list | 180 / 389 / 386 | 67.4 / 272 / 285 |

The worst case (story: -1.1% on 0.5B, -1.6% on 1.5B) is still a small loss, so
drafting stays opt-in (`--draft 7` / `AXON_DRAFT=7`).
