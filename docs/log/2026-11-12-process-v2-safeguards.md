# Process v2 safeguards: the golden check, model tests in CI, nightly fuzzing  (PR #N)

Process v2 ([methodology.md](../methodology.md#process-v2-2026-11-12), #224) rules 2 and 8 asked for safeguards
because no human reads the code. This PR is the Axon side; the schedulers live in axon-devstack (`axon-nightly`,
`axon-weekly`, the second-family review trigger).

## What landed
- **`scripts/golden.py`**, a sibling of `scripts/zoo.py`. zoo.py hands llama.cpp Axon's own token ids, so a
  tokenizer or chat-template bug passes it. golden.py starts from text on both sides and compares, per model, with
  llama-server (CPU, `--jinja`, the GGUF's own template):
  - (a) tokenization of five plain prompts (whitespace, Unicode, code, special-token text that must stay text):
    `axon logits`' prompt ids vs `/tokenize` with add_special, exactly;
  - (b) the rendered chat prompt of a system message and two user turns: Axon's ids vs `/apply-template` then
    `/tokenize` (add_special + special parsing, as llama-server's chat endpoint runs it), exactly. Axon pins two
    template dates (chat.rs: Llama 3's "26 Jul 2024", harmony's `HARMONY_DATE`); golden.py passes them to the
    template as `chat_template_kwargs`, so today's date is not a difference;
  - (c) greedy top-1 agreement: llama.cpp's greedy 64 tokens after (b)'s prompt and after a text prompt, Axon
    teacher-forced on the same ids (`AXON_LOGITS_IDS` + `AXON_LOGITS_PREFIX`). A disagreement where llama.cpp's own
    top-2 are within 0.1 nats is a near-tie; a clear one fails the model, as does agreement under 90 % or a logprob
    gap of llama.cpp's token over 1.0 nats.
- **`AXON_LOGITS_CHAT=FILE`** for `axon logits`: the ids `/v1/chat/completions` would run for a request body
  (`serve::chat_prompt`, rendered exactly as `completion` renders it), printed as `chat prompt ids (N): [...]`,
  then the logits after them. An env-gated debug path of `logits`; gen, chat and serve are unchanged.
- **CI, per PR** (`model-oracles`, next to `check`, so the wall time stays check's ~12 min): the test model, cached
  across runs and sha256-checked, and the tests that hold Axon against independent references:
  `tokenizer::tests::oracle` (Hugging Face's tokenizer), `model::tests::matches_ollama` (ollama's recorded greedy
  path), the golden dequant rows (gguf-py). The step fails if the model tests skipped.
- **CI, nightly** (`.github/workflows/nightly.yml`, 07:23 UTC; also by hand and on PRs that touch it or a fuzzer):
  `fuzz_gguf.py` on the GGUF (30 min cap) and on the FP8 checkpoint with `--safetensors` (20 min), on an XFS image
  with reflink (the runner's ext4 can't share the mutants' weights); `fuzz_serve.py --slow`, `--ops`, `--apis`
  (10 min each); then the whole suite on the CPU with the test model and the FP8 checkpoint present. Session files
  have no fuzz harness yet (not cheap: `axon chat --session` has no non-interactive load path to drive).

## How it runs nightly
axon-devstack's `axon-nightly` (bella, 03:30): `axon-postmerge` when main moved, then one `--low` axon-run job
running main's own `scripts/golden.py` with main's release binary on the golden model list (11 models today:
Qwen2.5 0.5B/1.5B/7B, Qwen3 4B/8B, Llama 3.2 1B/3B, Llama 3.1 8B, Gemma 3 1B/4B, gpt-oss-20b; the compatibility
audit's top 10 replaces it). A check that passed the previous night and fails now is bisected over main's merges
and the foreman is notified with the first bad sha.

## First results
(filled from the smoke runs below)

## Gates
(job ids in the PR body)
