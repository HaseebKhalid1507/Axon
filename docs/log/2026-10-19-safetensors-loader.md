# safetensors stage A: native safetensors loader, BF16 run as the GGUF F16  (PR #102)

- Date: 2026-10-19. Branch: feat/safetensors-loader. Base: main 05fb70c. Status: open (draft)
- Pick / design: the FP8 / NVFP4 plan (`specs/fp8-safetensors.md`, Zero, 2026-10-19), §2 and §5 stage A.

## What changed

- **`src/safetensors.rs`** (new): a Hugging Face checkpoint directory (`config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `model.safetensors` or the shards of `model.safetensors.index.json`) loads into a
  `gguf::Model` the rest of Axon can't tell from the GGUF `convert_hf_to_gguf.py --outtype f16` makes of it:
  - header parse with every check before anything is sized from the file: `header_len` ≤ 100 MB and inside
    the file, a JSON object of `{dtype, shape, data_offsets}` objects, ≤ 4 dims, no overflow, every range
    inside the data and exactly shape × dtype bytes; overlapping ranges accepted (legal, nothing writes
    through them). All 17 safetensors dtypes are named and sized (F8_*, F4, U* get Axon-private type ids
    ≥ 1100 that `check_supported` refuses by name).
  - the index: plain file names only (no `/`, `..`), every named file present, every tensor in the shard
    the index names, no tensor in two shards. Shards are opened in name order (= layer order).
  - `config.json` → the GGUF keys the converter writes (one table, `metadata`): `block_count`,
    `context_length`, `embedding_length`, `feed_forward_length`, `head_count(_kv)`, `rope.freq_base` (F32),
    `layer_norm_rms_epsilon` (F32), `key_length`/`value_length` from `head_dim`; llama also `vocab_size`,
    `rope.dimension_count`; RoPE scaling linear / yarn keys, and for `llama3` the `rope_freqs.weight`
    tensor computed in f32 as the converter (torch) does. Missing optional keys stay missing, so
    `check_supported` sees what it would see in the GGUF.
  - `tokenizer.json` → `tokenizer.ggml.*` (`tokenizer_metadata`), the converter's rules read from its source
    (llama.cpp 4413a19 `get_vocab_base`, gguf-py `SpecialVocab`): ids below config `vocab_size` without a token
    become `[PAD<i>]` (UNUSED); added tokens CONTROL if special or `<|…|>`-looking, else USER_DEFINED; merges
    normalized to `"a b"`; bos/eos/pad/unk from tokenizer_config's token strings, else config.json's ids;
    `add_bos_token` from the post-processor, else tokenizer_config; the chat template. The pre-tokenizer's
    `Split` regex is compared byte for byte with Qwen2's and Llama 3's (→ `qwen2`, `llama-bpe`), never run;
    anything else is refused with the pattern shown. `tokenizer.rs` is untouched.
  - HF names → GGUF names (`gguf_name`, qwen2 / qwen3 / llama), shapes reversed. No RoPE row permutation
    for llama: HF rows are already in Axon's rotate-half order (the GGUF converter permutes them, Axon
    un-permutes GGUF llama; a safetensors model never goes through that plan).
  - conversion as the converter makes it: vectors, `*_norm.weight` and routers → F32; other BF16/F16
    matrices → F16; F32 matrices stay F32 (lossless; `--outtype f16` would round them). Converted tensors go
    into one anonymous huge-page arena (`Model::aux`, 64-byte slots), filled on every core with pread +
    a 65536-entry BF16→F16 table; F16 / F32 tensors already in their final type and 32-byte aligned in the
    file are read in place from the mapped shard.
- **`src/gguf.rs`**: `Model` holds N mappings: `data` (the GGUF, or shard 0), `shards` (1..N) and `aux`;
  `Tensor` gains `shard` (0, i, or `AUX`) and `hf` (the HF name and dtype). `tensor_data`, `tensor_file`,
  `release`, `prefault`, `unowned` dispatch per mapping (the `released` list is keyed by mapping);
  `resident_fraction` and read-ahead stay GGUF-only. `load_with` / `load_to_run` take a directory or a
  `.safetensors` file; `load_header` (for `axon info`) skips the conversion. GGUF loading is unchanged.
  `Source::{Gguf, Safetensors{dir, files}}`.
- **CLI**: `gen`, `logits`, `chat`, `serve`, `info` take a checkpoint directory. `axon info <dir>` lists every
  tensor's HF name → GGUF name, type ("F16 (from BF16)"), dims. `--device cuda` is refused ("runs on the CPU
  only for now … comes with the FP8 stage B") in `Backend::new_capped`.
- **identity / kvsave**: the model digest of a checkpoint is SHA-256 over each file's sampled digest (header +
  tensor edges + windows for shards, whole for the JSON files); `kvsave::model_id` of a directory hashes its
  files' ids. Both were file-only and would have failed on a directory.
- **`scripts/fuzz_gguf.py --safetensors`**: 309 mutant checkpoint directories (truncated header, 2 GB and
  2^64 − 1 header lengths, non-object / invalid JSON, bad / reversed / huge / overlapping offsets, shape and
  dtype edits, a missing shard, index path traversal, 42 config and 19 tokenizer-file edits, 192 single-byte header
  flips), each run through `info` and `gen -n 1`.
- Not done (stage B and later, per the spec): `Tensor.quant` / `Scheme` (FP8 scales), MoE per-expert virtual
  tensors, deepseek2 / gemma3 / gpt-oss name maps, any GPU path. The spec's `F16X` private type id is not used:
  converted matrices are `F16` (id 1) because their bytes *are* the GGUF F16 bytes; provenance is in `Tensor.hf`.

## Why

The spec's §0: every frontier checkpoint ships as safetensors + config.json + tokenizer.json; GGUF-only means
running someone else's conversion. Stage A is the loader with a bit-exact reference: the GGUF F16 of the same
weights, run by Axon itself.

**The rounding was checked, not assumed.** The converter does `torch bf16 → f32` (exact) then
`numpy.astype(float16)`. numpy keeps a signalling NaN signalling (`0x7f81 → 0x7c08`), where Axon's
`quant::f32_to_f16` quiets it (`0x7e08`, as `vcvtps2ph`). `safetensors::bf16_to_f16` matches numpy for all
65536 patterns (FNV-1a of the table pinned in `bf16_table_matches_numpy`; torch's path gives the same table).
No real weight is a NaN, but now it can't differ.

## How it was verified

Models: `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (model.safetensors sha256
`fdf756fa…fb7fe` = the HF LFS oid), and `models/Qwen2.5-0.5B-Instruct-F16.gguf` made from it locally with
llama.cpp 4413a19 `convert_hf_to_gguf.py --outtype f16` (venv: torch 2.11 CPU, transformers 4.57.6). A
synthetic 2-shard split of the same checkpoint (`split2.py`, tensors in data order, cut at half the bytes).
Metadata / tokenizer only (no weights): `Qwen/Qwen3-0.6B` and `unsloth/Llama-3.2-1B-Instruct` config and
tokenizer files, against `Qwen3-0.6B-Q8_0.gguf` and `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.

| gate | result | job ID / log |
|---|---|---|
| (1) CPU `logits` (all 151936, 9 decimals) + greedy `gen -n 128` md5, dir == 2-shard dir == GGUF F16, 50 prompts | pending | |
| (1) every converted tensor == the GGUF F16's (name, dims, type, bytes), 290/290 | pending | |
| (2) tokenizer parity, 135 texts (50 gate prompts, tokens.json, natural_text, edge cases incl. decomposed accents), `encode_prompt` and `encode_special`, Qwen2.5 / Qwen3 / Llama 3.2 | pending | |
| (2) metadata parity: every key Axon reads, same value (Qwen2.5 all; Qwen3 / Llama: chat template excluded, different file revisions) | pending | |
| llama3 `rope_freqs.weight` computed == the GGUF's, bit for bit | pending | |
| (3) `check_supported` refusal parity: 8 config edits, same message as the GGUF built from the same keys | pass (synthetic, in the suite) | `check_supported_refuses_as_for_the_gguf` |
| (3) new refusals: 23 loader cases | pass (synthetic, in the suite) | `bad_checkpoints_are_clean_errors` |
| (4) fuzz, safetensors mode (309 mutants × info + gen) and GGUF mode | pending | |
| (5) load time vs GGUF warm | pending | |
| tiny synthetic checkpoint (1 and 3 shards) == its independently built GGUF F16: tensors and CPU logits bit for bit | pass | `tiny_checkpoint_is_its_gguf_f16` |
| md5 of `gen`, main vs candidate, GGUF models, CPU | pending | |
| release suite (fast tier) | pending | |
| debug-assertion suite (fast tier, CPU) | pending | |
| clippy -D warnings (bin + tests), rustfmt | pass | |

## Results

Pending.

## What didn't work / notes

- First draft quieted signalling NaNs (`quant::f32_to_f16`); numpy doesn't. Fixed before any gate (above).
- `encode_prompt` doesn't split special-token text; the parity test checks `encode_special` too.
