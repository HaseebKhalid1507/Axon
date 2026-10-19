# safetensors stage A: native safetensors loader, BF16 run as the GGUF F16  (PR #102)

- Date: 2026-10-19. Branch: feat/safetensors-loader. Base: main 05fb70c, main ecb53b7 (#89 + #92) merged in at 95683e4.
  Status: open (draft). Review: shady 7/10 SHIP-AFTER-FIXES (fixes in 14ea340, below).
- Pick / design: the FP8 / NVFP4 plan (`specs/fp8-safetensors.md`, Zero, 2026-10-19), §2 and §5 stage A.

## What changed

- **`src/safetensors.rs`** (new): a Hugging Face checkpoint directory (`config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `model.safetensors` or the shards of `model.safetensors.index.json`) loads into a
  `gguf::Model` the rest of Axon can't tell from the GGUF `convert_hf_to_gguf.py --outtype f16` makes of it:
  - header parse with every check before anything is sized from the file: `header_len` ≤ 100 MB and inside
    the file, a JSON object of `{dtype, shape, data_offsets}` objects, ≤ 4 dims, no overflow, every range
    inside the data and exactly shape × dtype bytes, and the ranges tile the data exactly (sorted, each
    begins where the previous ends, 0 to the end of the file: the reference reader's `Metadata::validate`).
    The JSON side files are capped at 256 MB. All 17 safetensors dtypes are named and sized (F8_*, F4, U* get Axon-private type ids
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
  `.safetensors` file; `load_header` (for `axon info`) skips the conversion (computed tensors such as
  `rope_freqs.weight` are there; converted ones can't be read). GGUF loading is unchanged. Read-ahead
  (`read_ahead`) stays GGUF-only: deferred, since a BF16 checkpoint is read whole by the conversion's
  pread; an F16 checkpoint's in-place matrices fault in page by page from the shard mapping (not measured).
  `Source::{Gguf, Safetensors{dir, files}}`.
- **CLI**: `gen`, `logits`, `chat`, `serve`, `info` take a checkpoint directory. `axon info <dir>` lists every
  tensor's HF name → GGUF name, type ("F16 (from BF16)"), dims. `--device cuda` is refused ("runs on the CPU
  only for now … comes with the FP8 stage B") in `Backend::new_capped`.
- **identity / kvsave**: the model digest of a checkpoint is SHA-256 over each file's sampled digest (header +
  tensor edges + windows for shards, whole for the JSON files); `kvsave::model_id` of a checkpoint, named by
  its directory or by a shard in it, hashes every file's id. Both were file-only and would have failed on a
  directory.
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
`numpy.astype(float16)`. numpy's portable software path keeps a signalling NaN signalling
(`0x7f81 → 0x7c08`), where Axon's `quant::f32_to_f16` quiets it (`0x7e08`, as `vcvtps2ph`).
`safetensors::bf16_to_f16` matches numpy's software path for all 65536 patterns (FNV-1a of the table pinned in
`bf16_table_matches_numpy`, computed on x86-64; torch's path gives the same table). A numpy that uses F16C or
aarch64's `__fp16` quiets sNaN instead, so NaN payloads depend on the converter's platform; every non-NaN value
is the same everywhere, and no real weight is a NaN. The llama3 RoPE factors (Rust `powf` vs torch's pow) are
verified bit for bit on Llama-3.2 (base 500000, head 64) only.

## How it was verified

Models: `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (model.safetensors sha256
`fdf756fa…fb7fe` = the HF LFS oid), and `models/Qwen2.5-0.5B-Instruct-F16.gguf` made from it locally with
llama.cpp 4413a19 `convert_hf_to_gguf.py --outtype f16` (venv: torch 2.11 CPU, transformers 4.57.6). A
synthetic 2-shard split of the same checkpoint (`split2.py`, tensors in data order, cut at half the bytes).
Metadata / tokenizer only (no weights): `Qwen/Qwen3-0.6B` and `unsloth/Llama-3.2-1B-Instruct` config and
tokenizer files, against `Qwen3-0.6B-Q8_0.gguf` and `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.

Round 1 (code 82f049b / fe4c3ef, base main 05fb70c), all on bella's CPU unless named:

| gate | result | job ID | label |
|---|---|---|---|
| (1) 50 prompts: `logits` (all 151936, 9 decimals) + greedy `gen -n 128`, md5, checkpoint dir == 2-shard dir == GGUF F16 | 50/50 equal, all three | 1019-023221-03ce (B) | MEASURED |
| (1) 290/290 converted tensors == the GGUF F16's (name, dims, type, bytes) | pass | 1019-023221-03ce (A) | MEASURED |
| (2) Qwen2.5 metadata parity + tokenizer parity, 193 texts × `encode_prompt` / `encode_special` | pass | 1019-023221-03ce (A) | MEASURED |
| (2) Qwen3-0.6B metadata + tokenizer parity (files only) | pass | 1019-023221-03ce (A) | MEASURED |
| (2) Llama-3.2-1B metadata + tokenizer parity, llama3 `rope_freqs.weight` bits (files only) | pass 4/4 at fe4c3ef (failed at 82f049b on `add_bos_token` only, below) | 1019-030337-597f | MEASURED |
| (3) refusal parity: 8 config edits, same `check_supported` message as the GGUF; 23 new loader refusals | pass | in the suite | MEASURED |
| (4) fuzz at 82f049b: 309 safetensors mutants: 1 finding (below); GGUF 1762 mutants: 0 | FAIL → fixed | 1019-023221-b0fe | MEASURED |
| (4) fuzz at fe4c3ef: 309 safetensors + 1762 GGUF mutants, 0 findings | pass | 1019-030322-14bc | MEASURED |
| (5) load time, `gen Hi -n 1` wall, medians of 10: dir 0.365 s, GGUF 0.3545 s (+3.0 %), 2-shard dir 0.433 s (+22 %) | dir passes; 2-shard: see below | 1019-023221-03ce (D) | MEASURED, non-exclusive host |
| (5) re-measured with every variant warmed, 15 reps: dir 0.573, 2-shard 0.564, GGUF 0.902 s | 2-shard == dir; host noisy | 1019-035502-711e | MEASURED, non-exclusive host |
| md5 of `gen -n 64`, main 05fb70c vs candidate, 4 GGUFs × 5 prompts, CPU | 0/20 differ | 1019-023221-03ce (C) | MEASURED |
| GPU refusal message | pass | 1019-023221-03ce (E) | MEASURED |
| release suite, fast tier (82f049b) | 571 passed, 0 failed | 1019-023225-89e5 | MEASURED |
| debug-assertion suite, fast tier, CPU (82f049b) | 589 passed, 0 failed, 42 ignored | 1019-023225-24de | MEASURED |
| clippy -D warnings (bin + tests), rustfmt | pass | local | MEASURED |

**Load time, 2-shard (gate 5).** The +22 % of the first run was the gate script's: it warmed the one-file
directory and the GGUF twice each and never the 2-shard copy, so its first reps read 1 GB of cold page cache.
With every variant warmed (1019-035502-711e) the 2-shard median is the one-file one's (0.564 vs 0.573 s). On that
run the GGUF was slower (0.902 s): bella was busy and these numbers move ±40 % between runs, so the honest claim
is "no sharding cost measurable on this host", not a percentage. An exclusive timing was not run (bella's timing
queue is reserved for the merge train); the final round's gate D uses the warmed, 15-rep script.

Final round (merge head 14ea340 = main ecb53b7 + this branch + the review fixes; base binary main ecb53b7):

| gate | job id | result | label |
|---|---|---|---|
| gates.sh A: unit tests on real checkpoints (Qwen2.5 tensors + metadata + tokenizer, Qwen3, Llama 3.2 metadata/tokenizer/rope_freqs, Llama-3.2-1B 147/147 tensors byte-equal, q/k included, rope_freqs included) | 1019-040921-9ab5 | 5 passed / 0 failed | MEASURED |
| gates.sh B: 50 prompts, all-vocab logits + gen 128 md5, dir == 2-shard == GGUF F16 (Qwen2.5-0.5B) | 1019-040921-9ab5 | 50/50 equal ×3, 0 nonzero rc | MEASURED |
| gates.sh C: md5 vs main ecb53b7, GGUF gen -n 64, 4 models × 5 prompts, CPU | 1019-040921-9ab5 | 0/20 differ | MEASURED |
| gates.sh D: load, `gen Hi -n 1`, all warmed, 15 interleaved reps: dir 0.458 s, 2-shard 0.477 s, GGUF 0.578 s | 1019-040921-9ab5 | both checkpoint forms faster than the GGUF (copy load) on this run | MEASURED, non-exclusive host |
| gates.sh E: `--device cuda` refusal names stage B | 1019-040921-9ab5 | pass | MEASURED |
| gate B on Llama-3.2-1B-Instruct weights: 50 prompts, all-vocab (128256) logits + gen 128, dir == GGUF F16 | 1019-040921-690d | 50/50 equal, 0 nonzero rc | MEASURED |
| fuzz: 311 safetensors mutants (incl. `st-overlap-bomb`, `st-trailing`) × info + gen; GGUF 1762 mutants | 1019-040921-5f19 | 0 findings (st: 46 ok, 576 refused; gguf: 389 ok, 3135 refused) | MEASURED |
| md5 vs main ecb53b7, GPU (bella), 6 GGUFs × 5 prompts, -n 64 | 1019-040922-90d2 | 0/30 differ | MEASURED |
| full tier release, shard 1/3, 2/3, 3/3 | 1019-040926-9562 / -bfcc / -2620 | 188/0 (7 ign), 298/0 (9 ign), 118/0 (26 ign) | MEASURED |
| full tier debug, shard 1/3, 3/3 | 1019-040926-6f97 / -7a09 | 188/0 (7 ign), 118/0 (26 ign) | MEASURED |
| full tier debug, shard 2/3 | ~~1019-040926-5dcd~~ (rc 125: avante lost power mid-run, no test failed) superseded by 1019-112219-becb | 298/0 (9 ign) | MEASURED |
| clippy -D warnings (bin + tests), rustfmt | local | pass | MEASURED |

### Merge head b9d8364 (main 87ac27d, #97, merged in; textually clean)
#102's only `src/lm.rs` change is the safetensors GPU refusal in `Backend::new_capped`; it adds no backend and no `Lm` impl, so `prompt_chunks` composes (a checkpoint runs as `Backend::Cpu` → `Qwen2`, the trait default). clippy -D warnings bin + tests, rustfmt: pass.

| gate | job id | result |
|---|---|---|
| gates.sh A (real-checkpoint unit tests) | 1019-130620-30fd | pending |
| fast tier release | 1019-130620-f273 | pending |
| fast tier debug (CPU) | 1019-130620-dca0 | pending |

## Results

Bit-identical to the converter's GGUF F16 on Qwen2.5-0.5B-Instruct (one file and two shards) and on
Llama-3.2-1B-Instruct, 50/50 prompts each, all-vocab logits and 128 greedy tokens. Load time: the checkpoint
loads no slower than the GGUF on a shared host (final round: dir 0.458 s, 2-shard 0.477 s, GGUF 0.578 s,
medians of 15, warmed). GGUF outputs unchanged (md5 vs main: CPU 0/20, GPU 0/30). Fuzz: 0 findings in 311
safetensors and 1762 GGUF mutants.

## What didn't work / notes

- First draft quieted signalling NaNs (`quant::f32_to_f16`); numpy doesn't. Fixed before any gate (above).
- Fuzz finding (1019-023221-b0fe, at 82f049b): a qwen2 config with `rope_scaling {rope_type: llama3}` made
  `gen` panic. The computed `rope_freqs.weight` lived in the aux arena, which was only created after the
  `check` callback, and `check_supported` reads it (`rope_freqs`). Fix (fe4c3ef): computed tensors sit first
  in the arena and are mapped before `check` (and in a header-only load); the conversion fills the rest
  after. Unit test `llama3_rope_on_a_checkpoint_is_readable_before_conversion`. GGUF mode: 1762 mutants, clean.
- Llama 3.2 metadata parity failed on `add_bos_token` only: bartowski's Q4_K_M predates the converter writing
  that key; Axon's llama-bpe default is the same `true` (the tokenizer test compares the BOS both add).
- Review (shady, 7/10): the parser accepted overlapping ranges (the spec called them legal), so N tensors
  could share one range and each get its own converted copy: a 1 GB file could ask for N × its size of
  anonymous memory and be OOM-killed instead of refused. The reference reader refuses any layout that doesn't
  tile the data; now so does this one (`shared_ranges_cant_inflate_the_arena`, fuzz `st-overlap-bomb`,
  `st-trailing`; the spec sentence is corrected). Also from the review: JSON side files capped at 256 MB;
  `model_id` of `dir/model.safetensors` hashes the whole checkpoint like `dir` does (it hashed one shard);
  layer numbers must be canonical (`007` refused, it aliased layer 7); the Llama `add_bos_token` is asserted
  `true`, not skipped; `rope_scaling linear` without a factor says so.
- Qwen3-0.6B's weights were not run (only its files): the download budget went to Llama-3.2-1B, whose q/k
  rows are the case the "no permutation for a checkpoint" claim rests on.
- `encode_prompt` doesn't split special-token text; the parity test checks `encode_special` too.
