# Draft-model speculative decoding on a CPU/GPU split

A model too big for the GPU runs split (`src/hybrid.rs`): the CPU runs the embedding and
the first layers, the GPU the rest and the head. On bella (RTX 5070 Ti Laptop 12 GB,
30 GB DDR5) Qwen2.5-32B Q4_K_M puts 27 of its 64 layers on the GPU; the other 37 are
~11 GB of weights the CPU reads from RAM for every token. That read is the step:
**~185 ms per token, 5.2–5.4 tok/s** greedy.

A verify pass (`Lm::verify_argmax`, K draft tokens + 1) reads those weights **once for all
K + 1 rows** (the CPU's batched r8 GEMM, `Qwen2::hidden_batch`), then the GPU half runs
the rows through its layers and returns one argmax per row. As long as the CPU's compute
for K + 1 rows hides under the RAM read, a pass costs about what a single token costs, and
a draft model on the GPU (Qwen2.5-0.5B: ~1 ms per guess) is almost free next to it.

Verify is bit-identical to decode per token (CPU rows for CPU layers, GPU rows for GPU
layers: the hybrid's existing contract), so the output is exactly the non-draft greedy
output. The bench checks this for every run (below: SAME).

## Measured cost of a verify pass by rows (nb = K + 1)

`spec_bench` microbench, median of 5, ms per pass at one position after a 230-token prompt:

| host | split | nb1 | nb2 | nb4 | nb6 | nb8 | nb9 | nb10 | nb12 |
|---|---|---|---|---|---|---|---|---|---|
| bella (Core Ultra 9 275HX, DDR5) | 32B, 27/64 GPU | 182 | 207 | 203 | 209 | 231 | 253 | 283 | 299 |
| pod (Ryzen 9 7950X, 4090) | 14B, 20/48 GPU | 138 | 124 | 126 | 128 | 123 | 120 | 129 | 134 |

On bella a pass of up to 8 rows costs 1.1–1.27× a token; past 8 the CPU's compute shows
(≈ +17 ms per row). The pod's Zen 4 cores keep 12 rows under the RAM read.

## Choosing K

BELLA_TABLE

Default: **K = 7** for a draft model next to a split model (gen and chat; `--draft K` still
wins). nb = 8 is the last row count bella's CPU hides under the weight read, and longer
drafts gain little: acceptance per guess falls with depth (chat 42 % at K=3, 23 % at K=7).
Not split: K = 3 as before (an all-GPU verify costs 1.45× a token at nb 8).

## The drafter's VRAM

The drafter's memory is set aside before the main model is placed (`draft::load_pair`),
so the split moves layers to the CPU: VRAM_LINE

## `--draft-model auto` (opt-in)

gen and chat take `--draft-model auto`: the smallest GGUF in the model's directory, at most
1/8 of its size, whose vocabulary and end tokens match (`draft::check_vocab`). Nothing
turns drafting on by itself; a product default (drafting on whenever such a file is there)
is proposed, not implemented: with a draft model, greedy output is unchanged and greedy
decode is 2–5× faster on a split, but sampled decoding gains ~nothing and the drafter costs
VRAM and a load.

## Reproduce

`axon gen <32B> "<prompt>" -n 128 --device cuda --draft-model <0.5B>` prints the pass
counts, acceptance and the per-nb verify times (`draft:` line). `axon gen --ctx N` places
the model as for an N-token context (a long context can split a model that otherwise fits).
