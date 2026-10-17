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
| pod (Ryzen 9 7950X, 4090) | 14B, 20/48 GPU | 138 | 124 | 126 | 128 | 123 | 120 | 129 | 134 |

On bella a pass of up to 8 rows costs 1.1–1.27× a token;
past 8–9 the CPU's compute shows. The pod's Zen 4 cores keep 12 rows under the RAM read.

## Choosing K

Pod (Ryzen 9 7950X 16 threads + 4090; POD numbers, not bella's): Qwen2.5-14B forced to a split
with `--gpu-layers 20` (of 48), 0.5B drafter, two rounds, all SAME: no draft 8.2–8.5 tok/s;
K=7 code 40.5–41.3, chat 19.2–19.4, summary 23.5–24.3; K=8 43.3–43.5 / 18.7–19.3 / 23.5–23.6.
The 1.5B drafter at equal layers: K=8 45.0 / 20.2 / 23.3.

Bella, Qwen2.5-32B Q4_K_M, greedy, 128 tokens, three chat-template prompts (code rewrite,
travel chat, summary). MEASURED with a local bench (one load per configuration; bella was
shared with other builders' CPU work, load average 6–16, not cool-gated, so absolute numbers
are on the low side). The split was automatic, and the free VRAM differed between the two
rounds, so the placement did too (round 1 / round 2 below); each cell is one run.

| prompt | no draft (27 / 28 GPU layers) | 0.5B K=3 (24 / 25) | K=5 | **K=7** | K=8 | K=10 | 1.5B K=7 (22 / 23) | accepted @ K=7 |
|---|---|---|---|---|---|---|---|---|
| code | 6.13 / 5.90 | 14.8 / 13.8 | 19.9 / 19.4 | **21.5 / 20.8** | 23.1 / 21.6 | 22.2 / 19.5 | 20.9 / 19.8 | 61 / 61 % |
| chat | 6.12 / 5.87 | 11.1 / 10.2 | 12.0 / 11.0 | **12.9 / 11.4** | 12.7 / 11.2 | 10.6 / 10.3 | 12.5 / 11.4 | 29 / 28 % |
| summary | 5.88 / 5.82 | 12.1 / 10.9 | 15.3 / 14.3 | **15.8 / 14.2** | 16.3 / 14.9 | 14.8 / 13.3 | 14.9 / 13.2 | 39 / 40 % |

tok/s; at K=7, ×3.5 code, ×2.0 chat, ×2.5 summary over the no-draft run of the same round.
K=7 and K=8 are within noise of each other (K=8 ahead on point estimates, n = 2); K=10 loses
(nb 11 passes cost 1.3–1.4×). The 1.5B drafter accepts a little more but costs two more GPU
layers: a wash or slightly worse. Within each process every drafted run's tokens equal the
no-draft run's (48/48). Across rounds the chat/summary tokens differ, because the placement
did (24 vs 25 layers: a different CPU/GPU cut is different numerics, as the hybrid documents);
code's matched. With the placement pinned (`--gpu-layers 24`, drafter loaded), four bella
processes (two with PDL on, two with `AXON_CUDA_PDL=0`) gave identical tokens for every prompt
and K, and identical drafted counts: the output is deterministic for a given placement. The
automatic placement depends on the VRAM other programs hold at load time, so pin
`--gpu-layers` where cross-run reproducibility matters.

Default: **K = 7** for a draft model next to a split model when decoding greedily (gen, and
chat with `--temp 0`; `--draft K` still wins). nb = 8 is the last row count bella's CPU hides under the weight read, and longer
drafts gain little: acceptance per guess falls with depth (chat 49 % at K=3, 29 % at K=7).
serve keeps K=3 (its split path runs one request at a time; not changed here).
Not split: K = 3 as before (an all-GPU verify costs 1.45× a token at nb 8).

14B at a 32k context on bella (placed for a 32768-token cache, the drafter's too, as
`chat --ctx 32768` and `gen --ctx 32768` place it) splits as well: 28/48 GPU layers
with no drafter, 23/48 with the 0.5B drafter, whose 32k KV comes out of the GPU's share. Greedy
tg128, 2 rounds, 18/18 SAME: no draft 13.8–15.4 tok/s; K=7 code 51.9–52.5, chat 27.2–29.1,
summary 28.4–28.8 (K=4 is equal on chat and summary, K=10 is slower).

## Sampled decoding

Sampled drafting picks each row's token from that row's candidates and keeps a guess while it
equals the pick, so it is as lossless as greedy (same output for the same seed), but a
sampled pick rarely equals the drafter's greedy guess. On a split, `Lm::verify_cands` used to
fall back to K + 1 forwards one after another (`HybridQwen2` didn't implement it): a sampled
draft pass cost K + 1 tokens' weight reads. It is now batched like the greedy verify (the CPU
half's `hidden_batch`, then `CudaQwen2::verify_cands_hidden`: batched GPU passes with the
candidates selected on the device), bit for bit `forward_cands` per row
(`hybrid_verify_cands_matches_forward_cands`).

Bella, 32B, temp 0.8 (top-k 40, top-p 0.9, repeat penalty 1.1), seed 42, 128 tokens, one
run each, MEASURED (load average 10–15): no draft at 27/64 layers 5.16–5.21 tok/s; with the 0.5B
drafter at 24/64 (K=0: 4.4–4.9):

| prompt | no draft | K=3 | K=7 | accepted K=3 / K=7 |
|---|---|---|---|---|
| code | 5.17 | 12.8 | 17.0 | 69 / 48 % |
| chat | 5.21 | 8.6 | 8.8 | 36 / 19 % |
| summary | 5.16 | 11.4 | 12.9 | 62 / 37 % |

Every sampled drafted run gave the no-draft run's tokens (same seed). A sampled draft pays on a
split (×1.7 chat, ×2.5–3.3 code and summary); K=7 is no better than K=3 on chat.

The split's K = 7 default applies to greedy decoding only; sampled keeps the draft model's
K = 3 (chat samples by default, temp 0.8; `--draft 7` for code-heavy sampled sessions). The
"little or no gain when sampling" note is no longer printed next to a split model.

## CPU GEMM prefetch (tried, dropped)

A software prefetch in the batched r8 GEMM for small batches (as decode's GEMV has) measured
from −15 % to no difference per verify pass on a loaded bella, and nothing on the Ryzen pod.
Unproven, so not in this change; it needs a quiet A/B as its own PR. The 32B tables above
(bench jobs 0d19 and 1cf9) were taken with it on and the 14B ones without; it changes no bits,
so only those tok/s figures carry its effect, at most the range above.

## The drafter's VRAM

The drafter's memory is set aside before the main model is placed (`draft::load_pair`),
so the split moves layers to the CPU: with the 0.5B drafter (K=0) the 32B gets 24 of 64 layers on the GPU instead of 27, and
plain decode drops from 5.8–6.1 to 4.6–5.3 tok/s (about −13 %: three more layers' weights read
from RAM per pass). The 1.5B drafter leaves 22/64. Drafting more than repays it (table above);
trimming the drafter's reserve on a split (it keeps its own 256 MiB headroom on top of the main
model's) would win back about a layer and is left as a follow-up. The drafter is loaded without
the spare-VRAM Q6_K prefill copy (same bits), and `load_pair` no longer holds the rest of the
GPU's free memory while it loads. One spare-VRAM extra is left: `CudaQwen2::build` widens the
drafter's batch scratch from 1024 to 1152 tokens if free VRAM shows room when it loads, and
that free VRAM still includes the main model's planned but not yet allocated KV cache. The
extra is the scratch for 128 more tokens (not measured); pinning the drafter's batch would
close it.

## `--draft-model auto` (opt-in)

gen and chat take `--draft-model auto`: the smallest GGUF in the model's directory, at most
1/8 of its size (the file's size, symlinks followed: a Hugging Face cache's snapshot entries
are links), whose vocabulary and end tokens match (`draft::check_vocab`). It picks the smallest
file, so a Q4_0 over a Q4_K_M of the same model when both are there. Nothing
turns drafting on by itself; a product default (drafting on whenever such a file is there)
is proposed, not implemented: with a draft model, greedy output is unchanged and greedy
decode is 2–3.5× faster on a split, but sampled decoding gains ~nothing and the drafter costs
VRAM and a load.

## Reproduce

`axon gen <32B> "<prompt>" -n 128 --device cuda --draft-model <0.5B>` prints the pass
counts, acceptance and the per-nb verify times (`draft:` line). `axon gen --ctx N` places
the model, and sizes the drafter's cache, for an N-token context as chat does (a long context
can split a model that otherwise fits); an N below the prompt and `-n` is noted and ignored.
