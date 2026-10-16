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

Median of 5 (a local bench, not in the tree: one model load, then `verify_argmax` timed per nb), ms per pass at one position after a 230-token prompt:

| host | split | nb1 | nb2 | nb4 | nb6 | nb8 | nb9 | nb10 | nb12 |
|---|---|---|---|---|---|---|---|---|---|
| bella (Core Ultra 9 275HX, DDR5), main's GEMM | 32B, 27/64 GPU | 182 | 207 | 203 | 209 | 231 | 253 | 283 | 299 |
| bella, GEMM prefetch (this PR) | 32B, 27/64 GPU | 155–167 | 163–183 | 170–187 | 170–195 | 175–198 | 190–203 | 224–254 | 217–233 |
| pod (Ryzen 9 7950X, 4090) | 14B, 20/48 GPU | 138 | 124 | 126 | 128 | 123 | 120 | 129 | 134 |

On bella a pass of up to 8 rows costs 1.1–1.27× a token (1.05–1.2× with the prefetch below);
past 8–9 the CPU's compute shows. Different runs: the first row is main's binary on a quiet
machine, the second this PR's under load. The pod's Zen 4 cores keep 12 rows under the RAM read.

## Choosing K

Pod (Ryzen 9 7950X 16 threads + 4090; POD numbers, not bella's): Qwen2.5-14B forced to a split
with `--gpu-layers 20` (of 48), 0.5B drafter, two rounds, all SAME: no draft 8.2–8.5 tok/s;
K=7 code 40.5–41.3, chat 19.2–19.4, summary 23.5–24.3; K=8 43.3–43.5 / 18.7–19.3 / 23.5–23.6.
The 1.5B drafter at equal layers: K=8 45.0 / 20.2 / 23.3.

Bella, Qwen2.5-32B Q4_K_M, greedy, 128 tokens, three chat-template prompts (code rewrite,
travel chat, summary). MEASURED, two interleaved rounds (low–high), the same local bench (one load per
configuration; bella was shared with other builders' CPU work, load average 6–16, so the
absolute numbers are on the low side). Baseline: no drafter, 27/64 layers on the GPU.
Every drafted run's output was token-for-token the non-draft output (SAME, 48/48).

| prompt | no draft | 0.5B K=3 | K=5 | **K=7** | K=8 | K=10 | 1.5B K=7 | accepted @ K=7 |
|---|---|---|---|---|---|---|---|---|
| code | 5.90–6.13 | 13.8–14.8 | 19.4–19.9 | **20.8–21.5** | 21.6–23.1 | 19.5–22.2 | 19.8–20.9 | 61 % |
| chat | 5.87–6.12 | 10.2–11.1 | 11.0–12.0 | **11.4–12.9** | 11.2–12.7 | 10.3–10.6 | 11.4–12.5 | 28–29 % |
| summary | 5.82–5.88 | 10.9–12.1 | 14.3–15.3 | **14.2–15.8** | 14.9–16.3 | 13.3–14.8 | 13.2–14.9 | 38–40 % |

tok/s; ×3.5 code, ×2.0 chat, ×2.5 summary at K=7 over the no-draft baseline. K=7 and
K=8 are within noise of each other; K=10 loses (nb 11 passes cost 1.3–1.4×). The 1.5B
drafter accepts a little more but costs two more GPU layers: a wash or slightly worse.

Default: **K = 7** for a draft model next to a split model (gen and chat; `--draft K` still
wins). nb = 8 is the last row count bella's CPU hides under the weight read, and longer
drafts gain little: acceptance per guess falls with depth (chat 49 % at K=3, 29 % at K=7).
serve keeps K=3 (its split path runs one request at a time; not changed here).
Not split: K = 3 as before (an all-GPU verify costs 1.45× a token at nb 8).

14B at a 32k context on bella (`gen --ctx 32768` placement) splits as well: 28/48 GPU layers
with no drafter, 23/48 with the 0.5B drafter, whose 32k KV comes out of the GPU's share. Greedy
tg128, 2 rounds, 18/18 SAME: no draft 13.8–15.4 tok/s; K=7 code 51.9–52.5, chat 27.2–29.1,
summary 28.4–28.8 (K=4 is equal on chat and summary, K=10 is slower).

## CPU GEMM prefetch

The CPU's batched r8 GEMM had no software prefetch (decode's GEMV has had it since r8bi). A
verify pass of a few rows is still a stream over the weights, so the GEMM now prefetches as the
GEMV does for batches of up to 16 tokens (loads only: same bits; `AXON_GEMM_PREFETCH=0` off).
Bella, 32B, same process layout, two rounds: nb 4 187–170 ms vs 194–200 off, nb 8 175–198 vs
213–223, nb 12 217–233 vs 259–271 (−10–15 % per verify pass; nb 1, which doesn't use the GEMM,
moved 4–7 % between the two runs, so read it as ~−8 % net). The Ryzen pod showed no difference
(its verify pass was already flat).

## The drafter's VRAM

The drafter's memory is set aside before the main model is placed (`draft::load_pair`),
so the split moves layers to the CPU: with the 0.5B drafter (K=0) the 32B gets 24 of 64 layers on the GPU instead of 27, and
plain decode drops from 5.8–6.1 to 4.6–5.3 tok/s (about −13 %: three more layers' weights read
from RAM per pass). The 1.5B drafter leaves 22/64. Drafting more than repays it (table above);
trimming the drafter's reserve on a split (it keeps its own 256 MiB headroom on top of the main
model's) would win back about a layer and is left as a follow-up.

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
