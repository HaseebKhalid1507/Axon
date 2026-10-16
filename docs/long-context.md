# Long context: YaRN (`--rope-scaling yarn`)

Qwen2.5 (7B/14B/32B) is trained to 32768 tokens and Qwen3 to 32768 (its GGUF says 40960:
32k plus an output reserve). Past that, RoPE turns positions the model never saw and the
output degrades. The model cards' answer is YaRN, a RoPE rescaling done at inference time,
with no extra weights: `rope_scaling {type: yarn, factor: 4, original_max_position_embeddings:
32768}` takes them to 131072. None of the GGUF files carry it (llama.cpp's converter only
writes `rope.scaling.*` keys for a config that already has `rope_type: yarn`), so in
llama.cpp, vLLM and here it is a flag:

```
axon serve models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --device cuda --kv q8 --ctx 131072 \
    --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768
```

## When to use it

Only when your prompts actually run past the trained context. It is **off by default** and
should stay off otherwise: static YaRN changes the output at every length, short prompts
included (the measurements below), and Qwen says the same ("if the average context length
does not exceed 32,768 tokens, we do not recommend enabling YaRN"). If your sessions top out
around 64k, use `--rope-scale 2` (Qwen3's card advice): the smaller the factor, the
smaller the change at short lengths.

The flag is global and fixed for the process: every request of a `serve`, every turn of a
`chat`, is computed with it. That keeps Axon's contract (decode == batched prefill ==
verify, bit for bit; a cache is a function of the prefix and the flags). Dynamic YaRN (the
factor following the current length) is not offered: every cached key would depend on the
length the session has reached.

## Flags (llama.cpp's names)

| flag | meaning | default |
|---|---|---|
| `--rope-scaling yarn` | turn YaRN on (`none`: drop a GGUF's own `rope.scaling.*` keys) | off |
| `--rope-scale F` | the factor; the context becomes orig x F | required (or the GGUF's) |
| `--yarn-orig-ctx N` | the trained context the ramp is computed from | the GGUF's `context_length` (pass 32768 for Qwen3, as its card does) |
| `--yarn-beta-fast B` | pairs turning at least B times in N keep their frequency | 32 |
| `--yarn-beta-slow B` | pairs turning at most B times in N are divided by F | 1 |
| `--yarn-attn-factor M` | the scale on q and k (logits x M^2) | 0.1 ln F + 1 (1.1386 at F = 4) |

They work on `gen`, `logits`, `chat` and `serve`, and print what they resolved to on
stderr, e.g. `rope scaling: yarn, factor 4, original context 32768, context 131072, beta
32/1, mscale 1.1386`. `--ctx` (chat, serve) may then go up to orig x F; `/v1/models`
reports the context served. `serve` without `--ctx` takes the model's context where it
fits, so with the flag that is now orig x F (131072): pass `--kv q8` (and `--ctx` if you
want less) to keep the KV cache in budget. A GGUF that does carry `rope.scaling.type = yarn` (with
`factor`, `original_context_length`, and optionally `attn_factor`, `yarn_beta_fast`,
`yarn_beta_slow`, `yarn_ext_factor`) gets YaRN without flags, as in llama.cpp. Not on gemma3.

## What it computes

As llama.cpp (`ggml_rope_yarn_corr_dims`, `rope_yarn`), vLLM and HF transformers, static,
`truncate` on. For the Qwen2.5-7B recipe (head size 128, base 1e6, orig 32768, F 4,
beta 32/1):

- pairs 0..23 (wavelength up to ~900 positions) keep their frequency;
- pairs 40..63 (less than one turn in 32768) are divided by 4;
- pairs 24..39 blend linearly between the two;
- every (sin, cos) of the RoPE table is multiplied by 0.1 ln 4 + 1 = 1.1386, so q and k
  both are, and the attention logits by 1.296.

In Axon that is the frequency vector (`model::rope_freqs`, after Llama 3's `rope_freqs`
factors, as llama.cpp's theta / ff) and one multiply in the three table builders (CPU
`rope_table_of`, CUDA `rope_table` and `rope_table_b`, f64 before the f32 rounding). Every
kernel reads the tables unchanged: no per-token cost. Without the flag the tables are
bit-identical to before (the multiply by 1 is exact).

## Measurements

Qwen2.5-7B-Instruct Q4_K_M, factor 4, orig 32768.

**Retrieval** (`cuda::qwen2::yarn_eval::yarn_needle`): a five-digit pass key, stated twice
in one sentence, at a depth of a Sherlock Holmes excerpt (Project Gutenberg), `--kv q8`,
greedy answer checked for the key. RTX 5090. MEASURED:

| prompt tokens | depth | yarn | no flag |
|---:|---:|---|---|
| 40000 | 0.1 / 0.5 / 0.9 | pass / pass / pass | pass / pass / pass |
| 64000 | 0.1 / 0.5 / 0.9 | pass / pass / pass | pass / **fail** ("25983") / pass |
| 100000 | 0.1 / 0.5 / 0.9 | pass / pass / pass | **fail** ("5") / pass / pass |
| 64000 | 0.02 / 0.25 / 0.35 | pass / pass / pass | pass / **fail** ("25987") / pass |
| 100000 | 0.02 / 0.25 / 0.35 | pass / pass / pass | **fail** ("5") / **fail** ("5") / **fail** ("59") |
| 128000 | 0.02 / 0.25 / 0.35 | pass / pass / pass | **fail** ("400") / **fail** ("4000") / **fail** ("4000") |

Total: 18/18 with the flag, 9/18 without; every 100k-128k case with the key more than
~65k tokens back fails without it.

Prefill: 23 s for 100k tokens, 35-37 s for 128k on the 5090 (q8 cache). Without the flag the model still
finds a key that sits near the end of the prompt (the distance back to it is under 32k: RoPE
only sees relative positions) and sometimes one further back; with it every case passes.
A pass key repeated in one sentence is an easy test (RULER calls single-needle retrieval
too easy to gate on): it shows the flag works, not that 128k is as good as 32k. Qwen's own
RULER numbers for Qwen2.5-7B-Instruct with their DCA + YaRN serving (not the static recipe
here) are 82.3 at 64k and 55.1 at 128k, against 93.7-89.4 at 16k-32k.

**Short-context cost** (`yarn_short_ctx`): every position of 2 windows each of Pride and
Prejudice and Sherlock Holmes, f16 cache, next-token perplexity with and without the flag,
KL(plain || yarn) per position and top-1 agreement. RTX 4090. MEASURED:

| window | positions | perplexity, no flag | with yarn | change | mean KL(plain ‖ yarn) | top-1 agreement |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 8188 | 5.956 | 6.433 | +8.0 % | 0.106 nats | 86.6 % |
| 4096 | 16380 | 6.457 | 6.956 | +7.7 % | 0.101 nats | 86.4 % |
| 8192 | 32764 | 6.511 | 7.013 | +7.7 % | 0.097 nats | 85.7 % |

llama.cpp's own `llama-perplexity` (CPU build, `-c 2048 --chunks 4`, Sherlock Holmes)
agrees on the size: 7.094 without the flags, 7.657 with `--rope-scaling yarn --rope-scale 4
--yarn-orig-ctx 32768` (+7.9 %). For comparison the YaRN paper's non-fine-tuned LLaMA-7B,
static factor 4, lost +3.5 % at 2k (4.05 to 4.19).

That is a real cost for prompts that never needed it: about +8 % perplexity at 2k-8k and
a different top-1 token at one position in seven. Keep it off unless the prompts are long.

**Agreement with llama.cpp**: Qwen2.5-7B-Instruct Q4_K_M, llama.cpp `4413a19` CUDA
(`llama-server --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`) greedy 16 tokens
after a Sherlock Holmes prompt of N tokens, top-5 log-probabilities; Axon (CUDA, f16 cache,
the same flags) teacher-forced on the same tokens. RTX 5070 Ti Laptop (bella). MEASURED:

| | prompt | top-1 agreement | max abs(Δ logprob) of llama.cpp's top token |
|---|---:|---:|---:|
| no flag (baseline spread) | 8000 / 30000 | 16/16, 16/16 | 0.263, 0.144 |
| yarn, both | 8000 / 30000 / 40000 / 60000 | 16/16, 14/16, 16/16, 15/16 | 0.224, 0.165, 0.176, 0.211 |
| control: Axon without the flag vs llama.cpp with it | 8000 / 30000 / 40000 / 60000 | 10/16, 16/16, 14/16, 11/16 | 0.750, 0.701, 0.751, 0.804 |

With YaRN on both sides the difference stays within the no-flag spread of the two engines
(f32 vs f64 tables, different kernels); a wrong RoPE (the control) is 3-4x further off.
The three yarn misses are near-ties in llama.cpp's own distribution (its top two 0.001,
0.28 and 0.21 nats apart).
