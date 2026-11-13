# Models and hardware

What runs on main, in detail: the architectures, tensor types and model files, then the CPU and GPU
requirements. Commands and flags: [usage.md](usage.md). Serving: [serving.md](serving.md).

## Architectures

Architectures: `qwen2`, `qwen3` (dense) and `llama` (Llama 3.x). The tests use
Qwen2.5-0.5B-Instruct (Q4_K_M), Qwen3 0.6B (Q8_0) and Llama 3.2 1B Instruct, and the
GPU tests also the Qwen2.5 1.5B, 3B and 7B Instruct, Qwen3 4B and 8B, and Llama 3.2 3B
and Llama 3.1 8B Instruct Q4_K_M builds; Qwen3 1.7B (Q4_K_M) runs too. Llama 3's RoPE
frequency factors and the llama-bpe tokenizer with its BOS token are handled. `llama`
models with a SentencePiece tokenizer (Mistral 7B Instruct v0.3, Llama 2, TinyLlama) work
too; Mistral's `[INST]` chat format is built in.

`gemma3` (Gemma 3 1B and 4B instruct;
head size 256, sliding-window layers, GeGLU) runs on the CPU and the GPU with its
`<start_of_turn>` chat format; on the GPU the whole model must fit (no CPU/GPU split, no
`--kv q8` yet), and its local layers keep a full-length KV cache. 27B is refused.

`qwen3moe` (Qwen3-30B-A3B and its relatives: softmax top-k routing, no shared expert)
runs on the CPU (`--device cpu`: every expert repacked for the r8 kernels, stage 1, see
[the log](log/2026-10-17-moe-stage1.md)) or, with `--device cuda`, with its attention,
norms, routers and head on the GPU and its experts on the CPU (llama.cpp's `--cpu-moe`
layout; stage 2, [log](log/2026-10-17-moe-stage2.md)). Both: decode == batched
prefill == verify bit for bit, logits within llama.cpp's own CPU/CUDA spread; no layer
split (`--gpu-layers`), multi-GPU or MoE draft model. `axon info` prints its experts and
bytes per token. `--gpu-experts N|auto` keeps the experts of N MoE layers on the GPU too,
with the same bits (below; [roadmap](roadmap.md#tracks)).

`deepseek2` (DeepSeek V2/V3/V3.1, Kimi K2,
Moonlight: MLA attention in the absorbed form with a K-only 576-wide cache, sigmoid routing
with `exp_probs_b` and shared experts, YaRN's DeepSeek attention scale; the `kimi-k2` and
`deepseek-v3` tokenizers and the Kimi and DeepSeek V3 chat formats) runs on the CPU
(`--device cpu`, [log](log/2026-10-18-deepseek2-s0.md)) or, with `--device cuda`, as
qwen3moe does: MLA attention (its own kernels, a 576-wide f16 latent cache), router, shared
experts and the dense lead layer on the GPU, the routed experts on the CPU (stage 1,
[log](log/2026-10-18-deepseek2-s1.md); f16 KV only for now). `--gpu-experts N|auto`
keeps the routed experts of N MoE layers (auto: as many as fit) on the GPU as well, with
the same bits; opt-in, default 0. Moonlight on a 12 GB laptop GPU, `auto` (25 of 26 layers)
vs 0: decode ×2.06, prompt ×3.40 (MEASURED, TAINTED: other GPU load up to 31 %;
[log](log/2026-10-24-moe-resident-c.md)). Both: decode == batched
prefill == verify bit for bit; checked on Moonlight-16B-A3B. Legacy non-MLA deepseek2
GGUFs (an unsplit `attn_kv_b`, e.g. the published DeepSeek-V2-Lite files) are refused with a
re-convert message.

`gpt-oss` (gpt-oss-20b, ggml-org's MXFP4
GGUF) runs on the CPU (stage 0, [log](log/2026-10-18-gpt-oss-s0.md)) or, with
`--device cuda`, with its attention, norms, router and head on the GPU and its MXFP4 experts on
the CPU (stage 1, [log](log/2026-10-18-gpt-oss-s1.md): f16 KV only): MXFP4
experts read in place, attention sinks, the 128-token sliding window on alternate layers,
YaRN, the o200k tokenizer and the harmony chat format (`axon chat` shows the `final`
channel; `axon serve` returns the `analysis` channel as `reasoning_content`, with tool
calling: [Tool calling](serving.md#tool-calling)). Other architectures, other
mixture-of-experts models and RoPE scaling (a factor other than 1; Gemma 3's linear
scaling and YaRN ([usage.md](usage.md#long-context)) excepted) are refused with an error.

## Tensor types

Tensor types: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and
IQ4_NL on the CPU, so every Q2_K and Q3_K_S/M/L GGUF runs (IQ4_NL is llama.cpp's fallback
for their rows that aren't a multiple of 256, e.g. the 0.5B's 896), and MXFP4 for
gpt-oss's experts (CPU only: the GPU gets gpt-oss's other tensors). On the GPU every weight matrix must be one of the eleven
quantized types (F16/F32 matrices are refused there). Others load for `info` but can't run.

## Model files: GGUF and safetensors

Model files: GGUF, or a Hugging Face checkpoint directory (`config.json`, `tokenizer.json`,
`tokenizer_config.json`, and `model.safetensors` or the shards of `model.safetensors.index.json`)
of a `qwen2`, `qwen3` or `llama` model with BF16, F16 or F32 weights, wherever a `.gguf` path is
taken (`axon info <dir>` lists each tensor's Hugging Face name, GGUF name and type). A BF16 or F16
checkpoint loads as the F16 GGUF `convert_hf_to_gguf.py --outtype f16` makes of the same weights:
BF16 matrices converted to F16 and vectors to F32 at load (numpy's rounding, checked for all 65536
values), the same metadata and tokenizer, and the same logits and greedy output bit for bit
(checked on Qwen2.5-0.5B-Instruct and Llama-3.2-1B-Instruct,
[log](log/2026-10-19-safetensors-loader.md)). F32 matrices stay F32 (lossless, where
`--outtype f16` would round them). CPU only for now
(the GPU has no F16 matrix kernels). **FP8 checkpoints** (compressed-tensors per-channel e4m3 with
dynamic per-token activations, RedHatAI's `*-FP8-dynamic`) run everywhere: in exact mode (the
default) as Q8_0 converted at load (today's kernels and bit contract; the 7B passes both accuracy bars against the CPU FP8 reference (KL 1.2e-3, top-1 98.1 %), while the 0.5B scores 96.7 % top-1 against HF F32, below the 97 % bar), on the CPU as the exact dequant (slow; the accuracy reference), and on the
GPU natively with `--numerics fast` (sm_89+; the prefill GEMM's tensor cores multiply e4m3 on sm_120+ and f16 below,
#190; [fp8-weights.md](design/fp8-weights.md)).
Fast FP8 **fails its bars on bella** (sm_120): 7B tg128 32 tok/s vs Q4_K_M's 85–101 (decode runs
without CUDA graphs: the follow-up), top-1 95.9 % vs the 97 % bar; it isn't measured on an H100. Block-scaled FP8, NVFP4, MoE and the other
architectures are refused with a message (roadmap). The pre-tokenizer must be one
Axon knows (Qwen2's or Llama 3's regex: recognized, never executed); SentencePiece-style
`tokenizer.json` files are refused (use the GGUF). Like llama.cpp, the NFC normalizer isn't applied.

### Getting models: `axon pull`

`axon pull NAME` downloads a GGUF from Hugging Face into the models directory: by a short name
(`axon pull --list`: about ten models for local coding agents, from qwen2.5-coder and qwen3 4B/8B/14B
to llama3.1-8b, gemma3-4b, gpt-oss-20b and the 30B-A3B mixtures of experts), each a fixed file at a
pinned commit with its SHA-256, or any single-file GGUF as `OWNER/REPO/FILE.gguf` (or the file's
`https://huggingface.co/OWNER/REPO/resolve/main/FILE` URL), checked against the SHA-256 Hugging Face
publishes for it (the Git LFS oid). No registry. The models directory is `$AXON_MODELS_DIR`, else
`$XDG_DATA_HOME/axon/models`, else `~/.local/share/axon/models`; a file lands in
`OWNER/REPO/FILE` under it (`OWNER/REPO/DIR/FILE` for a file in a directory of the repo). The bytes come
through `curl` (HTTPS only, redirects included; `-q`, so `~/.curlrc` is not read), are hashed as they
arrive into `FILE.part`, and the file is renamed into place only once its size and SHA-256 match (a
mismatch deletes the `.part`). An interrupted pull resumes where its `.part` ends, if `FILE.part.want`
says it was a download of the same file (SHA-256 and size; a `.part` of an older version starts over).
Gated repos need `HF_TOKEN` (an access token; given to curl on its stdin, not its command line; curl
7.58 or newer, which doesn't send it on to another host). `gen`, `chat`, `serve` and `info` take a
pulled model by its name (`axon chat qwen3-8b`), its `OWNER/REPO/FILE.gguf` or its file name, wherever
they take a path; a name not pulled yet says the command that pulls it. `axon serve qwen3-8b` serves
the model as `qwen3-8b` (`/v1/models`, a request's `model`).
Split GGUFs (`-00001-of-0000N.gguf`) are refused: axon loads single files.

## Hardware

### CPU

CPU: any x86-64 (others build too, scalar only). The default build is baseline x86-64 and
picks its AVX2 / AVX-VNNI code at runtime, with the same output bits as a `-C target-cpu`
build (see `src/mv.rs`). Attention and SwiGLU use fused multiply-adds when the CPU has
FMA and a separate multiply and add when it doesn't (`src/vmath.rs`), so logits depend on
that one CPU property (on x86: whether the CPU has FMA; other targets fuse only when
built with `fma` enabled), and are identical across paths (`AXON_SIMD`), batch sizes and
thread counts for a given answer to it.

### NVIDIA GPUs

GPU (`--device cuda` or `AXON_DEVICE=cuda`): NVIDIA, Linux only. Nothing CUDA is needed
to build. At run time axon loads the driver (`libcuda.so.1`) and NVRTC (`libnvrtc.so`,
`.so.13` or `.so.12`, from the library path or the toolkit's `lib64` under `$CUDA_PATH`,
`/usr/local/cuda` or `/opt/cuda`), compiles its kernels for the GPU on first use and
caches them in `~/.cache/axon/cuda` (`$XDG_CACHE_HOME/axon/cuda`). The driver must
support CUDA 12 or newer, and at least NVRTC's CUDA major version. A GPU newer than
NVRTC gets PTX for the driver to compile. If the first NVRTC found can't target the GPU
(NVRTC 13 dropped Pascal and Volta), the next one installed is tried, so a GTX 1050
(sm_61) runs on NVRTC 12.x beside 13; a GPU no installed NVRTC can target is refused.
The int8 tensor-core GEMMs (batched prefill, speculative verify) and the tensor-core
attention need sm_80 or newer; older GPUs run batches through the decode kernels
(dp4a): correct, not fast. Without a usable GPU, `--device cuda` stops with the reason;
it never falls back to the CPU silently.

### CPU + GPU

CPU + GPU: a model too big for the GPU is split (the first layers on the CPU, the rest
and the output head on the GPU; `--gpu-layers N` to choose). For prompts (batches of
32 tokens or more) the CPU layers' weights are streamed through the GPU, chunk by chunk,
instead of running those layers on the CPU (`AXON_STREAM=0`: off).

Several GPUs (`--gpus 0,1`, pipeline parallel): [usage.md](usage.md#flags-and-defaults).

## Not on main yet

The list is in [START-HERE](START-HERE.md#what-works-on-main), under the table.
