# Compatibility audit: ollama's 30 most-pulled models (2026-11-13, MEASURED)

**Does Axon run what ollama users pull, and does it run it like llama.cpp?** 30 models, one Q4_K_M GGUF each (or the
quant ollama ships), on one RTX 4090 pod. Axon `3014eeb` (= main 213465a's `src/`) on CUDA, against llama.cpp b11580
(`4413a19`, the CPU release build) as the reference.

- **17 run, 13 are refused.** Every refusal is clean: exit 2, one line naming the architecture, no panic.
- **Of the 17 that run:**
  - **8** match llama.cpp end to end: prompt ids equal (Mistral differs by design), greedy within zoo's match rule.
  - **4** run but build a different chat prompt from the GGUF's own template. These are product bugs, filed as #238, #239, #240 and #241.
  - **4** run but refuse `/v1/chat/completions`: plain-text turn markers, #242.
  - **1** (Qwen2.5-7B) has one greedy flip outside a near-tie.
- **Greedy:** no model's logits are far off. The worst is DeepSeek-R1-0528 (YaRN) at KL 0.021, against at most 0.007 for every model that matches. Its triage is staged but not run (blocked, see the end).
- **Refusals,** ranked by what each fix unlocks, are in roadmap issue #243.

Job `1113-010532-8176` (gatewatch `eval-09efdf0-compat-audit`, rc 0, 49 min). Pod4 was a secure 4090 at $0.89/h, up
58 min, **$0.883** (cap $5). Axon ran it alone: the pod was held for the run, then taken down and released by
`~/.cache/axon-eval/pod4-guard.sh`. Draft PR #235.

## Method

- **The list.** ollama's library sorted by pulls (2026-11-12), top 30. For each: the default tag's size and quant,
  read from ollama's registry manifest, and a matching HF GGUF (official, bartowski, unsloth or ggml-org). Repo,
  file, size and sha256 are in `scripts/examples/compat-audit/ollama-top30.json`. These are HF files, not ollama's own
  blobs: the quant mix can differ. The unsloth "UD" files for Qwen3.6/3.8, for instance, contain i-quants that
  ollama's blobs don't.
- **The run.** `scripts/compat_audit.sh` drives `scripts/zoo.py run --chat --fetch-each --delete-after --sparse-first`,
  one model at a time: download (sha256-checked), run, delete. A refused file is first tried as a sparse header-only
  copy, so it is never downloaded in full. Per file:
  - `axon info`, plus `axon gen -n 64` greedy.
  - Top-20 next-token logprobs from `axon logits` (CUDA) and from llama-server (CPU, given **Axon's** token ids) on
    16 prompts: top-1 agreement and KL. The verdict is "match" if mean KL ≤ 0.02 and every top-1 disagreement is a
    near-tie (llama.cpp's own top-2 gap < 0.1 nats).
  - Tokenization of 17 prompts against llama.cpp's.
  - Two chat conversations, one with a system message and one 3-turn without, through `axon serve` and through
    llama-server `--jinja` (the GGUF's own template).
- **Chat prompts at the id level.** The run can only compare prompt token *counts*: axon 3014eeb can't print the ids
  it serves (#229 adds `AXON_LOGITS_CHAT`). Equal counts can hide different ids, and DeepSeek-R1's first conversation
  is exactly that case (14 = 14). So `scripts/examples/compat-audit/prompt_check.py` rebuilds Axon's ids offline from
  the code paths at 3014eeb (`chat::Framing`, `serve.rs` `render()`) with each model's HF tokenizer, encoding piece by
  piece as Axon does. It requires every rebuilt count to equal the measured one. **24 of 24 do**, so the rebuilt ids
  are the served ids. It then compares them with llama.cpp's rendering. Run on 12 models: all that serve chat, except
  gpt-oss, whose harmony prompt golden.py checks nightly.

## Results

Pulls are ollama's whole-family counts, in millions. "top-1" and "KL" are Axon against llama.cpp on 16 prompts
(llama.cpp given Axon's ids). "near-tie" means llama.cpp's own top-2 gap is under 0.1 nats.

| # | ollama (pulls M) | GGUF | arch / quant | verdict | top-1, KL mean | chat prompt vs the GGUF's template | note |
|---:|---|---|---|---|---|---|---|
| 1 | llama3.1 8b (121) | bartowski Meta-Llama-3.1-8B-Instruct | llama / Q4_K_M | **runs, matches** | 15/16 (near-tie 0.055), 6.1e-4 | ids equal | |
| 2 | nomic-embed-text (97.5) | nomic-ai nomic-embed-text-v1.5 | nomic-bert / Q4_K_M | **refused** | | | architecture 'nomic-bert' is not supported (embedding model) |
| 3 | deepseek-r1 8b (95.0) | unsloth DeepSeek-R1-0528-Qwen3-8B | qwen3 / Q4_K_M | **runs, differs** | 14/16 (gaps 0.205, 0.199), **2.1e-2** | **differs**: no BOS, `</think>` forced, earlier reply without `<｜Assistant｜>` | #238, #240; greedy drift, triage pending (YaRN) |
| 4 | llama3.2 3b (87.6) | bartowski Llama-3.2-3B-Instruct | llama / Q4_K_M | **runs, matches** | 16/16, 1.4e-3 | ids equal | |
| 5 | qwen2.5 7b (45.6) | bartowski Qwen2.5-7B-Instruct | qwen2 / Q4_K_M | **runs, differs** (greedy) | 15/16 (gap 0.249), 5.7e-3 | ids equal | one flip outside a near-tie; triage pending |
| 6 | qwen3 8b (43.3) | Qwen Qwen3-8B | qwen3 / Q4_K_M | **runs, matches** | 16/16, 3.6e-3 | ids equal | |
| 7 | gemma3 4b (41.8) | bartowski gemma-3-4b-it | gemma3 / Q4_K_M | **runs, matches** | 16/16, 5.9e-3 | ids equal | text only (ollama ships a vision projector) |
| 8 | gemma2 9b (37.1) | bartowski gemma-2-9b-it | gemma2 / Q4_K_M | **refused** | | | architecture 'gemma2' is not supported |
| 9 | mistral 7b (34.2) | bartowski Mistral-7B-Instruct-v0.3 | llama / Q4_K_M | **runs, matches** | 16/16, 1.2e-3 | differs **by design** (chat.rs: mistral_common's tokens) | see below |
| 10 | gemma4 e4b (29.1) | unsloth gemma-4-E4B-it | gemma4 / Q4_K_M | **refused** | | | architecture 'gemma4' is not supported |
| 11 | llama3 8b (26.0) | bartowski Meta-Llama-3-8B-Instruct | llama / Q4_K_M | **runs, differs** | 15/16 (near-tie 0.065), 1.3e-3 | **differs**: 3.1's date header + a forced system block (46 vs 26, 65 vs 40 tokens) | #241 |
| 12 | qwen3.5 9b (25.3) | unsloth Qwen3.5-9B | qwen35 / Q4_K_M | **refused** | | | architecture 'qwen35' is not supported |
| 13 | qwen2.5-coder 7b (23.7) | bartowski Qwen2.5-Coder-7B-Instruct | qwen2 / Q4_K_M | **runs, matches** | 16/16, 1.1e-3 | ids equal | |
| 14 | phi3 3.8b (18.4) | bartowski Phi-3-mini-4k-instruct | phi3 / Q4_K_M | **refused** | | | architecture 'phi3' is not supported |
| 15 | mxbai-embed-large (16.9) | ChristianAzinn mxbai-embed-large-v1 | bert / Q4_K_M | **refused** | | | architecture 'bert' is not supported (embedding model) |
| 16 | llava 7b (15.4) | cjpais llava-v1.6-mistral-7b | llama / Q4_K_M | **runs, chat refused** | 16/16, 3.4e-4 | 400: "this model has no chat tokens" | #242; text model only, no projector |
| 17 | gpt-oss 20b (14.9) | ggml-org gpt-oss-20b | gpt-oss / MXFP4 | **runs, matches** | 15/16 (near-tie 0.004), 6.9e-4 | counts equal (harmony; ids: golden.py nightly) | |
| 18 | qwen3-coder 30b (10.9) | unsloth Qwen3-Coder-30B-A3B-Instruct | qwen3moe / Q4_K_M | **runs, matches** | 16/16, 7.0e-3 | ids equal | |
| 19 | glm-ocr (9.3) | ggml-org GLM-OCR | glm4 / Q8_0 | **refused** | | | architecture 'glm4' is not supported (an OCR model) |
| 20 | gemma 7b (8.4) | bartowski gemma-1.1-7b-it | gemma / Q4_K_M | **refused** | | | architecture 'gemma' is not supported |
| 21 | bge-m3 (8.4) | gpustack bge-m3 | bert / Q4_K_M | **refused** | | | architecture 'bert' is not supported (embedding model) |
| 22 | qwen 4b, Qwen1.5 (8.3) | Qwen qwen1_5-4b-chat | qwen2 / Q4_K_M | **runs, differs** | 16/16, 1.6e-4 | **differs**: Qwen2.5's default system prompt (59 vs 38 tokens) | #239; tokenizer: Axon is right, see below |
| 23 | codellama 7b (8.2) | TheBloke CodeLlama-7B-Instruct | llama / Q4_K_M | **runs, chat refused** | 16/16, 1.7e-3 | 400: "this model has no chat tokens" | #242 |
| 24 | phi4 14b (7.8) | bartowski phi-4 | phi3 / Q4_K_M | **refused** | | | architecture 'phi3' is not supported (also: pre-tokenizer `dbrx`) |
| 25 | llama2 7b (7.6) | TheBloke Llama-2-7B-Chat | llama / Q4_K_M | **runs, chat refused** | 16/16, 6.1e-4 | 400: "this model has no chat tokens" | #242 |
| 26 | qwen3.6 35b-a3b (7.6) | unsloth Qwen3.6-35B-A3B (UD) | qwen35moe / Q4_K_M | **refused** | | | architecture 'qwen35moe' is not supported |
| 27 | qwen3-vl 8b (7.2) | Qwen Qwen3-VL-8B-Thinking | qwen3vl / Q4_K_M | **refused** | | | architecture 'qwen3vl' is not supported |
| 28 | tinyllama 1.1b (6.8) | TheBloke TinyLlama-1.1B-Chat-v1.0 | llama / Q4_K_M | **runs, chat refused** | 16/16, 3.5e-3 | 400: "this model has no chat tokens" | #242 |
| 29 | qwen3.8 27b (6.4) | unsloth Qwen3.8-27B (UD) | qwen35 / Q4_K_M | **refused** | | | architecture 'qwen35' is not supported (the UD file also has 142 IQ4_XS + 2 IQ3_S) |
| 30 | qwen2 7b (6.4) | Qwen qwen2-7b-instruct | qwen2 / Q4_K_M | **runs, differs** | 16/16, 3.5e-3 | **differs**: Qwen2.5's default system prompt (59 vs 49 tokens) | #239 |

By pulls (876M across the 30):
- **Runs and matches:** 377M (43 %).
- **Runs, prompt differs (bug):** 136M.
- **Runs, chat refused:** 38M.
- **Runs, one greedy flip:** 46M.
- **Refused:** 279M.

zoo's `gen` column (greedy 64 tokens: characters in common with llama.cpp from the start, 1 to 347) isn't a verdict.
Two paraphrases split at the first near-tie: Qwen3-8B's " with" vs " in" at token 1, and Qwen2.5-Coder's " with
Johannes" vs " with the" at token 2. Both read correctly.

## What Axon refuses that users would expect to work

Ranked by pulls (details, tensor layouts and sizes in #243):

1. **Embeddings: nomic-bert, bert.** nomic-embed-text, mxbai-embed-large, bge-m3: **122.8M**. A new surface (an
   encoder plus `/v1/embeddings`), not an arch tweak. RAG and codebase-index setups pull it next to the chat model;
   Continue's default ollama embedder is nomic-embed-text.
2. **gemma2 + gemma:** **45.5M.** Small to medium: gemma3's path without q/k norm, plus logit soft-capping and
   alternating sliding windows.
3. **qwen35 / qwen35moe:** Qwen3.5, 3.6, 3.8: **39.3M**, Qwen's current family. Large: 3 Gated-DeltaNet layers per
   full-attention layer, which needs a recurrent state next to the KV cache.
4. **gemma4:** **29.1M.** Large: Gemma-3n-style per-layer embeddings, KV shared across layers, a new tokenizer model.
5. **phi3:** Phi-3, Phi-4: **26.2M.** Small: fused qkv and gate+up tensors, split at load. Phi-4 also needs the
   `dbrx` pre-tokenizer.
6. **qwen3vl** (7.2M) and **glm4** (GLM-OCR, 9.3M): vision models. The qwen3vl text decoder is tensor-for-tensor
   qwen3, so text-only support is nearly free, but users want the vision part.

By pulls per unit of work, gemma2 and phi3 come first: 71.7M pulls, both small.

## Runs, but differs: the product bugs

All four were measured as prompt-token counts and confirmed at the id level by `prompt_check.py`. "GGUF" below means
llama.cpp's `--jinja` rendering of the file's own template.

- **#238 DeepSeek-R1** (ollama #3, 95M). Axon renders R1 as DeepSeek V3.1's non-thinking mode:
  - `</think>` comes right after `<｜Assistant｜>`, so R1's reasoning is switched off. Single turn: "\nRed, green,
    blue." In the next conversation, plain-text reasoning leaked into `content`.
  - BOS is dropped: the GGUF says `add_bos_token = false`, but its template writes `{{ bos_token }}`.
  ```
  axon: You are a terse assistant.<｜User｜>Name three primary colors.<｜Assistant｜></think>
  GGUF: <｜begin▁of▁sentence｜>You are a terse assistant.<｜User｜>Name three primary colors.<｜Assistant｜>
  ```
- **#240 multi-turn DeepSeek prompts** (all DeepSeek-format models, every agent conversation). `serve.rs` `render()`
  frames an earlier reply as `{text}<｜end▁of▁sentence｜>` without its `<｜Assistant｜>` opening, so the reply is glued
  onto the user message: `<｜User｜>Write a haiku about autumn.Leaves fall, red and gold.<｜end▁of▁sentence｜>…`
  (25 vs 26 tokens). `axon chat` is fine.
- **#241 Llama 3** (ollama #11, 26M). `Format::Llama3` always writes 3.1/3.2's "Cutting Knowledge Date… Today Date…"
  system block, even with no system message. The original Llama 3 template has neither: 46 vs 26 tokens, 65 vs 40.
  Llama 3.1 and 3.2: ids equal.
- **#239 the qwen2 architecture's default system prompt.** `serve.rs:218` gives every `general.architecture = qwen2`
  file Qwen2.5's "You are Qwen, created by Alibaba Cloud…":
  - Qwen2's template says "You are a helpful assistant." (59 vs 49 tokens).
  - Qwen1.5's has no default (59 vs 38).
  - By the same path (not measured), the qwen2-arch R1 distills get that line bare in front of `<｜User｜>`.

  With a system message, ids are equal.

**The greedy drifts are not filed yet.** Both need llama.cpp's own CUDA-vs-CPU spread as a yardstick before they can
be called bugs.
- **DeepSeek-R1-0528:**
  - KL mean 0.0213 (max 0.077), top-1 14/16. Flips where llama.cpp's top-2 gap is 0.205 ("Dear Sir or Madam,\n\nI
    am writing to") and 0.199 ("Photosynthesis converts").
  - The control has the same arch, size and quant mix (Qwen3-8B, F32:145 Q4_K:217 Q6_K:37): KL 0.0036, 16/16.
  - What differs is that R1 runs YaRN (factor 4, original context 32768, mscale 1.1386), and Axon applies it. So the
    first suspect is the YaRN path on CUDA.
- **Qwen2.5-7B:** KL 0.0057 (max 0.0165), within the matching models' band. One flip at a 0.249-nat gap after
  `import numpy as np\nimport matplotlib.pyplot as plt\n\n`: Axon "def", llama.cpp "#".
- **The triage:** `scripts/compat_triage.sh` runs it in one bella job:
  - Axon CUDA vs llama.cpp CPU on the same machine.
  - R1 with `--rope-scaling none` on both sides.
  - R1 with Axon on the CPU.
  - golden.py (#229) on R1, Qwen2.5-7B and Qwen3-8B, with `--llama-server-ref` (llama.cpp CUDA) for the spread.

**Not bugs:**
- **Mistral v0.3: differs by design** (chat.rs module doc): Axon emits mistral_common's tokens. This GGUF's older
  template raises on a system role; llama.cpp's polyfill puts the system message in the first turn after a single
  "\n" and writes ` [/INST]` with a space. Replies: identical on the first conversation. On the second, Axon's
  stopped after one haiku, while llama.cpp's ran on into spring and summer.
- **Qwen1.5's tokenizer: Axon is right.** The GGUF has no `tokenizer.ggml.pre`. llama.cpp falls back to its
  "default" pre-tokenizer, while Axon uses Qwen2's regex. On every recorded differing prompt, the HF tokenizer agrees
  with Axon (`def fibonacci(n):` → `(n` is one token, 1445), not with llama.cpp.

## The nightly golden set (top 10)

The 10 most-pulled models Axon runs (`"golden": true` in the list; `scripts/zoo.py run --set golden`):
llama3.1 8b, deepseek-r1 8b, llama3.2 3b, qwen2.5 7b, qwen3 8b, gemma3 4b, mistral 7b v0.3, llama3 8b,
qwen2.5-coder 7b, gpt-oss 20b.

llava was left out: Axon runs only its text model. qwen3-coder 30B-A3B (#18) is the only qwen3moe path and the
coding-agent model, so it's the first alternate, though it needs experts on the CPU on bella. For golden.py (#229),
three of these will show a chat (b) DIFF from day one:
- deepseek-r1 until #238 and #240 are fixed;
- llama3 until #241 is fixed;
- mistral permanently, by design. It needs an expected-diff entry, or zoo-only.

Of the ten, bella lacks DeepSeek-R1-0528-Qwen3-8B, Meta-Llama-3-8B-Instruct and Qwen2.5-Coder-7B-Instruct (14.6 GB);
`zoo.py fetch --set golden` gets them sha256-checked.

## Reproduce

```
axon-podjob --on POD --timing POD --models none --max-min 110 --name eval-<sha7>-compat-audit --expect '^ZOO [0-9]+ files' -- \
  scripts/compat_audit.sh scripts/zoo.py scripts/oracle/llama_logits.py scripts/examples/compat-audit/ollama-top30.json \
  AXON llama-b11580-bin-ubuntu-x64.tar.gz --device cuda --chat --fetch-each --delete-after --sparse-first --prompts 16
python3 scripts/zoo.py audit-table LOG                                  # the table, and the fixes ranked by pulls
python3 scripts/examples/compat-audit/prompt_check.py LOG               # chat prompts at the id level (pip install tokenizers)
```
