# Benchmarks

Measured with `scripts/bench.py`, which runs every binary (and optionally the local
ollama server) in turn, round after round, and reports the median (min–max).
Compare binaries within one run; numbers across runs depend on what else the
machine was doing.

- `pp560`, `pp2100`: prompt evaluation, tok/s, ~560 / ~2100-token prompt
- `tg128`: decode tok/s over 128 tokens; `tg128@1`: the same on one thread
- `--load N`: N busy-loop processes running alongside (a machine that isn't idle)

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
