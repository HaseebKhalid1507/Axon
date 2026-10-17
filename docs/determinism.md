# Determinism: what Axon guarantees, and how to check it

Axon's output is a function of a small, named set of inputs. Fix them and you get the
same tokens, bit for bit, every time. Every response says which set produced it.

## The contract

For one **identity** (below), the same prompt tokens, the same sampling parameters and the
same `seed` give the same tokens, and the same logits behind them. That holds no matter
how the request was run:

| Run it...                                                        | Same bits? |
|------------------------------------------------------------------|------------|
| one token at a time (decode) or the whole prompt at once (batched prefill) | yes, except on a streamed CPU/GPU split (below) |
| through speculative drafting (prompt lookup or a draft model): drafts are verified with the same kernels, and sampled drafting takes the same random numbers | yes |
| with the prompt prefilled in chunks (the batch engine's chunked prefill) | yes |
| with its prefix taken from the cache (a chat follow-up, a slot reused) or prefilled fresh | yes, except on a streamed CPU/GPU split (below) |
| alone, or in a batch with up to `--parallel` other requests, in any slot, at any position in the batch, joining or leaving mid-generation | yes |
| again after a restart, on another day, on another machine with the same identity | yes (a split that streams says so in its identity, so a restart that can't stream has another one) |
| at temperature > 0 with the same `seed` (and params)              | yes: the sampler is seeded xoshiro256**, one draw per decided token |
| on the CPU with any thread count, SIMD or scalar paths, AVX2 or AVX-VNNI | yes (the md5 gate runs the CPU too; an AVX2 and an AVX-VNNI host agree 20/20) |

How: every matrix product is an exact integer block sum followed by one fixed float chain
over the blocks in order, the same in every kernel (decode GEMV, batched GEMM, verify);
attention folds 64-position blocks in ascending order in prefill and decode alike; nothing
reduces with atomics. So the order of every float operation depends on the model's shapes
and the token's position, never on the batch. The tests that hold this in place:
`feed_batch_matches_feed*`, `gpu_prefill_batchings_bit_identical_*`,
`gpu_products_match_reference_bits`, `verify_batch_matches_forward`,
`turns_match_fresh_prefill{,_cuda,_hybrid}`, `drafting_matches_*`,
`pool_steps_match_solo_in_any_order`, `serve_batched_matches_solo`, `serve_identity`,
`serve_identity_cpu`, and `drafting_matches_plain_sampling` (sampled drafting over
300 seeds), plus the gen md5 gate every PR runs against main.

## What is NOT guaranteed

- **CPU vs GPU.** Same model, same answer to within float tolerance (the tests check
  logits agree closely), but not the same bits. They are different identities. A
  CPU/GPU split (`--gpu-layers`, or placed automatically) is its own identity too.
- **A split that streams its CPU layers for prefill** (on by default when there's VRAM for
  it; `AXON_STREAM=0` turns it off). Prefill batches of at least 32 tokens
  (`AXON_STREAM_MIN`) run the CPU's layers on the GPU, with GPU numerics; decode and shorter
  batches run them on the CPU. So on such a split a 31-token and a 32-token prompt take
  different paths, a cached prefix (whose generated tokens' KV the CPU wrote) differs from a
  fresh prefill, and decode ≠ batched prefill. Same identity, same request, same schedule
  still gives the same bits. Whether it streams depends on free VRAM at load, so the
  identity's `backend` says `prefill streams >=32` when it does. A split without streaming
  keeps the whole contract (its CPU layers always use CPU numerics).
- **f16 vs q8 KV cache.** q8 rounds the cache: a different identity.
- **Across GPU architectures.** What was measured (gen md5 and prefill top-20 logits, 4
  models, 20 cells per cache type, main a591e17):
  - an RTX 4050 and an RTX 4090 (both sm_89, NVRTC 13.3 vs 12.8, different drivers):
    identical, 20/20 with f16 and 20/20 with q8. The toolkit and the driver don't change bits.
  - sm_61 (GTX 1050) vs sm_89: different from the prefill logits on. Below sm_80 attention
    takes another algorithm (split f32 dots, `exp2f`); from sm_80 it runs on tensor cores
    (f16 `mma.sync`, `ex2.approx`).
  - The split (non-tensor-core) attention, used below sm_80 and for head size 256 (Gemma 3)
    on every GPU, divides long contexts into `max_split = 8 × SMs / kv heads` parts (at most
    256), so its order depends on the card's SM count, not just the architecture: a 4050
    (20 SMs) and a 4090 (128) can differ past ~32 × max_split positions. On that path the
    identity's `gpu` carries the count (`sm_89, split attention 64`). Not audited across
    cards.
  - The code loaded: the device's own cubin by default. When NVRTC can't build that (an
    older toolkit: an older cubin of the major, or PTX the driver compiles) or on sm_90 with
    `AXON_HOPPER` (sm_90a code), the `gpu` field names it (`sm_120 [code compute_90 PTX,
    driver 12.8]`). Only same-target builds were audited.
  - sm_120 vs sm_89, and sm_80/sm_90: not audited yet. The GEMMs (integer sums, IEEE
    fma/mul/div) should be architecture-exact, but f16 `mma.sync` internal accumulation is
    architecture-specific (Hawkeye, arXiv:2603.20421).

  So the identity keys on the exact `sm_XY` (plus the code, and the split count where it
  matters): on the tensor-core path the same architecture and code share an identity
  whatever the card, toolkit or driver.
- **Across numerics versions.** When a change to Axon's kernels or sampler changes bits,
  `NUMERICS_VERSION` goes up and every fingerprint changes with it.
- **Across models**, obviously, including another quantization of the same weights.

Not guaranteed by design: a different seed, different sampling params, or a different
prompt rendering (a client that sends other text). Server settings a request inherits when
it leaves them out (`--no-think`, `--think-inline`, the server's sampling flags) are in the
identity's `serve` field.

## The identity tuple and `system_fingerprint`

The identity (`src/identity.rs`) is everything that decides the bits:

| field      | what                                                                  |
|------------|-----------------------------------------------------------------------|
| `numerics` | `NUMERICS_VERSION`: Axon's kernels and sampler (bumped by rule, below) |
| `model`    | SHA-256 over the GGUF file's length, its whole header (all metadata, tensor names, shapes, quant types) and samples of the tensor data |
| `types`    | the quant types and their tensor counts (readable; the header hash covers them) |
| `backend`  | `cpu`, `cuda`, `cuda+cpu` with the layer split (and `prefill streams >=N` when it streams), or a pipeline with its cuts |
| `gpu`      | per GPU: its exact architecture (`sm_120`); the code when it isn't the device's own cubin; the split count without tensor-core attention |
| `cpu`      | when any layer runs on the CPU: arch, FMA or not, r8 repack or per-row (`AXON_REPACK=0` changes the low bits); not the SIMD path, which matches scalar bit for bit |
| `kv`       | `f16`, `q8`, `q8/32`; `f32` on the CPU with `AXON_KV=f32`             |
| `rope`     | when not plain RoPE: the effective scaling as run (YaRN's factor, original context, betas, ext factor and (sin, cos) multiplier, from the GGUF or `--rope-scaling`; gemma3's linear scale) and `freq_base`. Read from the model's metadata after the flags rewrite it: the model digest hashes the file, which the flags don't change. `--rope-scaling none` on a plain model adds nothing |
| `serve`    | `axon serve` only, when not the defaults: `no-think`, `think-inline`, the server's sampling flags |

Left out because they never change bits: drafting, `--parallel`/slots, chunk sizes,
threads, CUDA graphs, PDL, and the context size (`--ctx`, or `serve`'s default, the trained
length where it fits). Attention covers only the positions filled, in 64-position blocks,
so a bigger cache lets a request run longer but never changes a token it would have
produced under the smaller limit; a request that hits the limit just stops (or is refused)
sooner. So set `max_tokens` when replaying: without it a reply runs to the server's context
or `--max-tokens`, which bound its length (not its tokens). Sampling params and the seed
belong to the request. A part that can't be read (the model file, the GPU) makes the
fingerprint `fp_unknown`: no identity is claimed. The
fingerprint is conservative: two identities can differ and still give the same bits
(sm_120 and sm_89 may well agree; unaudited); the promise is only that the same identity
gives the same bits.

The fingerprint is `fp_` plus 16 hex digits of the SHA-256 of the tuple's canonical text
(`axon-identity/1` then `name=value` lines). Where to see it:

- every successful `/v1/chat/completions` and `/v1/completions` response, whole or
  streamed (every chunk), as OpenAI's `system_fingerprint` (error bodies don't carry it);
- `GET /identity`: the whole tuple as JSON; also under `meta.identity` in `GET /v1/models`;
- `axon serve` logs it at start; `axon gen` prints `identity: fp_...` on stderr
  (`--verbose`: the whole tuple); `axon info` prints the model digest and the fingerprints
  for the CPU and, if there's a GPU the weights fit, for CUDA with each cache type (a model
  that would be split gets none: its identity depends on the placement).

A client that caches or replays by `(system_fingerprint, prompt, params, seed)` gets
exact hits. A changed fingerprint means stored outputs may not reproduce.

**Model digest: why sampled.** Hashing all of a 5 GB file costs seconds on every start
(software SHA-256 runs at a few hundred MB/s). The digest instead hashes the file length
and the union, in file order, of: the header in full, 256 bytes at each end of every
tensor, and 1024 windows of 4 KiB evenly spaced over the data (`axon-model-digest/2`;
pinned by `model_digest_golden`, checked against an independent Python model,
scripts/identity_digest_ref.py). That's a few MB in ~1-2k sorted reads with readahead off:
over what `axon info` used to read (the header alone): +40 ms warm for a 7B Q4_K_M on a
4090 pod, +75 ms warm on a laptop i5 (1.5B), and +180 ms cold (the file evicted with
`posix_fadvise(DONTNEED)`, 0 bytes resident by `fincore`; Llama 3.2 1B on a laptop NVMe,
~10 MB read). A slow disk or network filesystem pays ~2k random reads. Another model, a
requantization, a fine-tune or a merge changes the header or almost every block, so it
changes the digest. An edit made on purpose to bytes between the samples would not: the
digest names a file, it isn't a tamper seal. It reads the file itself, not the loaded copy
(which may have been repacked or released), so it depends on nothing but the file's bytes.

**The numerics rule.** `NUMERICS_VERSION` (src/identity.rs) is bumped in the same commit
as any change that can change an output bit for some model on some backend: a kernel's
arithmetic or reduction order (CPU or CUDA), the attention fold, RoPE, norms, dequant, the
sampler's order or RNG; and what turns a request into tokens or text: the tokenizer,
chat templates and prompt rendering, stop tokens and stop strings, think handling. A
change that keeps every bit (proven by the md5 gate) doesn't bump it. A new flag that
changes bits adds a field to the tuple, set only when the flag is, so default fingerprints
stay put (as `--rope-scaling` adds `rope`).

## How this compares

Most engines don't promise this. llama.cpp's server documents that with `cache_prompt`
"logits are **not** guaranteed to be bit-for-bit identical for different batch sizes
(prompt processing vs. token generation)": it switches kernels by token count (MMVQ for
≤ 8 tokens, MMQ above) and its MMQ and flash attention use stream-K, whose split depends
on the token count and the SM count. Thinking Machines ("Defeating Nondeterminism in LLM
Inference", 2025) showed batch invariance is the property that matters and gave the
recipe: one reduction order for every batch size, and attention split at a fixed size, not
a fixed count. That is what Axon's kernels do. vLLM (`VLLM_BATCH_INVARIANT=1`, beta) and
SGLang (`--enable-deterministic-inference`) offer it as a mode: SGLang measures 24–55 %
slower (34 % on average, Qwen3-8B on H100), Thinking Machines measured +62 % on vLLM, and
Vosti (arXiv:2609.38981) finds the modes still fail some of its equivalence tests (vLLM's
batch-invariant mode on Gemma3-4B passed 0 of 90 chunking and 5 of 384 prefill/decode
tests; SGLang's Triton backend 3 of 384 prefill/decode). Those are their numbers on their
hardware, not ours. Axon's contract is the default and costs little on the GPU (decode
GEMVs at ~93 % of bandwidth, prefill GEMM at llama.cpp MMQ parity, measured on an RTX 5070 Ti
Laptop, sm_120). Where it does cost something is CPU prefill, where a free float order would
be faster.

Cross-GPU bit identity is a separate, harder problem (it needs the reduction order to be
independent of the architecture's float units too). Axon claims it within one
architecture (measured on sm_89) and not across: hence `gpu` in the tuple.
