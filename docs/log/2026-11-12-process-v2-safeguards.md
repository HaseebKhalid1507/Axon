# Process v2 safeguards: the golden check, model tests in CI, nightly fuzzing  (PR #229)

Process v2 ([methodology.md](../methodology.md#process-v2-2026-11-12), #224) rules 2 and 8 asked for safeguards
because no human reads the code. This PR is the Axon side; the schedulers live in axon-devstack (`axon-nightly`,
`axon-weekly`, the second-family review trigger).

## What landed
- **`scripts/golden.py`**, a sibling of `scripts/zoo.py`. zoo.py hands llama.cpp Axon's own token ids, so a
  tokenizer or chat-template bug passes it. golden.py starts from text on both sides and compares, per model, with
  llama-server (CPU, `--jinja`, the GGUF's own template):
  - (a) tokenization of five plain prompts (whitespace, Unicode, code, special-token text that must stay text):
    `axon logits`' prompt ids vs `/tokenize` with add_special, exactly. One recorded exception: llama.cpp turns
    some user-defined tokens into control tokens at load (it logs "control-looking token ... overridden"; gemma 3's
    `</s>`, id 212) and then spells their text out. Axon keeps them whole, as the GGUF's token type and Hugging
    Face's tokenizer do (HF: `[..., 236743, 212]`), so the prompt counts as equal when Axon matches llama.cpp's
    tokenization with exactly those tokens kept whole;
  - (b) the rendered chat prompt of a system message and two user turns: Axon's ids vs `/apply-template` then
    `/tokenize` (add_special + special parsing, as llama-server's chat endpoint runs it), exactly. Axon pins two
    template dates (chat.rs: Llama 3's "26 Jul 2024", harmony's `HARMONY_DATE`); golden.py passes them to the
    template as `chat_template_kwargs`. gpt-oss's template ignores them (`strftime_now`), so its "Current date:"
    in llama.cpp's rendering is replaced by Axon's before tokenizing: today's date is never a difference;
  - (c) the distributions along llama.cpp's greedy path: llama.cpp's greedy 64 tokens after (b)'s prompt and after
    a text prompt, Axon teacher-forced on the same ids (`AXON_LOGITS_IDS` + `AXON_LOGITS_PREFIX`: position N-1
    batched, every later one a decode step), top 20 on both sides. Per step: top-1 agreement, KL(llama.cpp || Axon)
    over llama.cpp's top 20, |dlogprob| of llama.cpp's token, and for a flip its swing (llama.cpp's margin for its
    token plus Axon's for its own). The yardstick is llama.cpp's own spread: its CUDA build runs the same steps
    (`--llama-server-ref`), and Axon may differ from llama.cpp's CPU by 2.5x that spread, never less than the floors
    (mean KL 0.006, |dlogprob| 0.3 nats); a flip may swing twice the |dlogprob| limit. It also fails on top-1 under
    90 %, llama.cpp's token outside Axon's top 20, or a non-finite logit. "Thresholds" below has the numbers.
  - Verdicts: pass; FAIL, a real difference (exit 1); ERROR when something wasn't judged (axon crashed or timed out,
    chat n/a, the reference backend didn't start), missing, llama-error (exit 2). axon-nightly bisects a FAIL only.
  - **By design** (`BY_DESIGN`, from shady's #246 review): a (model file pattern, check, reason) list of places where
    Axon departs from llama.cpp on purpose. The check still runs and keeps its difference in the record, but reports
    BY-DESIGN instead of DIFF, and the model's other checks grade as usual, so it can't turn into a permanent red
    that teaches everyone to ignore red. One entry: Mistral v0.3's chat prompt, where Axon follows mistral_common's
    tokens (`src/chat.rs`'s module doc, "Mistral"; llama.cpp tokenizes the rendered text, one `▁` more after `[INST]`).
    A listed check that comes out equal is flagged stale.
  - `scripts/golden_selftest.py` (CI, beside the agent replay self-test, 0.05 s): the parsers on canned
    `axon logits` output, the metrics and limits, the verdicts and exit codes, `BY_DESIGN`, the date fixes.
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
- **Fuzzers, first CI run** (PR run, ubuntu-latest, 4 jobs): `fuzz_gguf.py` 1762 mutants / 3524 runs (ok 389,
  refused 3135), `--safetensors` 311 mutants / 622 runs (ok 47, refused 575), `fuzz_serve.py` --slow / --ops /
  --apis 302 + 349 + 117 checks: **0 findings**, 7 min 49 s in all. The run before it caught the harness's own bug
  (copy_file_range across filesystems, EXDEV): the models now sit on the XFS image with the mutants.
- **Per-PR wall time**: `check` 12 min, `model-oracles` 2 min in parallel. The nightly whole suite with the test
  models (`model-suite`): 12 min.
- **golden.py, first runs** (bella 11-12/13, then the compatibility audit's top 10 on a 4090): see "Thresholds".

## Thresholds (11-13)
The first golden.py had round-number limits: a top-1 flip over a 0.1-nat llama.cpp gap failed, and |dlogprob| up to
1.0 nat passed. shady's review called them strict where the noise is and loose where regressions are. The first
bella smokes (`1112-230701-3acb`, `1112-232326-f651`: Gemma 3 1B, Llama 3.2 1B, Qwen3 4B) agreed:
- **The noise is per model.** An independent float64 forward on each GGUF's own dequantized weights
  (a numpy script in the builder's scratch, `golden-diag/ref.py`, not committed) differs from llama.cpp's CPU by up to 0.40 nats on Gemma 3 1B (two flips, at 0.20
  and 0.31 nats of llama.cpp margin) and 0.12 on Llama 3.2 1B. The old 0.1-nat rule failed the exact answer, and
  any one fixed limit is wrong for one of the two. Rounding that reference's activations the way llama.cpp's CPU
  backend does (Q8_K for K-quant weights, Q8_0 for the rest) moves it by mean KL 0.0056 and up to 0.29 nats on
  Gemma 3 1B, nearly the whole gap (0.0072, 0.40): the gap is ordinary rounding noise there, not a bug in either
  engine. Errors correlate within an engine family (Axon CPU/CUDA 0.72, llama.cpp CPU/CUDA 0.58).
- **llama.cpp's own CUDA-vs-CPU spread tracks it.** So the limit is a multiple of that spread, measured on the same
  steps in the same run, with floors for models where llama.cpp's two backends nearly agree.

Clean Axon against llama.cpp's CPU (golden.py's (c), 64 tokens chat + text, top 20). Ratio = value / limit,
limit = max(floor, 2.5 x spread), the final constants; a ratio over 1 fails.

| model | host | steps | top-1 | mean KL | max dlp | worst swing | spread KL / dlp | KL ratio | dlp ratio | swing ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | bella | 98 | 1.0 | 0.00308 | 0.173 | 0 | 0.0016 / 0.183 | 0.51 | 0.38 | 0 |
| Gemma 3 1B | bella | 120 | 0.958 | 0.0121 | 0.841 | 1.37 | 0.0190 / 0.848 | 0.26 | 0.40 | 0.32 |
| Qwen3 4B | bella | 128 | 0.977 | 0.00229 | 0.258 | 0.381 | 0.0018 / 0.275 | 0.38 | 0.38 | 0.28 |
| R1-0528 Qwen3 8B | 4090 | 128 | 0.953 | 0.0114 | 0.622 | 1.03 | 0.0102 / 0.688 | 0.45 | 0.36 | 0.30 |
| Llama 3 8B | 4090 | 110 | 1.0 | 0.00098 | 0.090 | 0 | 0.0010 / 0.102 | 0.16 | 0.30 | 0 |
| Qwen3 8B | 4090 | 128 | 0.969 | 0.00167 | 0.147 | 0.335 | 0.0017 / 0.200 | 0.28 | 0.29 | 0.34 |
| Llama 3.1 8B | 4090 | 115 | 0.991 | 0.00125 | 0.124 | 0.220 | 0.0011 / 0.167 | 0.21 | 0.30 | 0.26 |
| Llama 3.2 3B | 4090 | 123 | 0.976 | 0.00087 | 0.135 | 0.262 | 0.0010 / 0.162 | 0.14 | 0.33 | 0.32 |
| Qwen2.5 7B | 4090 | 104 | 0.981 | 0.00336 | 0.247 | 0.273 | 0.0050 / 0.275 | 0.27 | 0.36 | 0.20 |
| Gemma 3 4B | 4090 | 119 | 0.975 | 0.0219 | 1.682 | 2.96 | 0.0237 / 1.014 | 0.37 | 0.66 | 0.58 |
| Mistral 7B v0.3 | 4090 | 102 | 0.990 | 0.00094 | 0.170 | 0.093 | 0.0019 / 0.138 | 0.16 | 0.49 | 0.13 |
| Qwen2.5 Coder 7B | 4090 | 106 | 0.991 | 0.00118 | 0.110 | 0.257 | 0.0009 / 0.103 | 0.20 | 0.37 | 0.43 |
| gpt-oss 20B | 4090 | 128 | 0.992 | 0.00075 | 0.086 | 0.193 | 0.0006 / 0.067 | 0.12 | 0.29 | 0.32 |

- **bella** (sm_120): `golden-diag/judge.py` (scratch), which runs golden.py's own compare / summary / judge on the
  recorded steps. Axon 292018f's CUDA build; llama.cpp 4413a19, CPU and CUDA builds.
- **4090**: job `1113-041027-d4fb`, #246's golden gate (golden.py at fddac98; Axon 75a33ef, i.e. #246 with this PR's
  hook), re-judged with the final constants. llama.cpp b11580 (= 4413a19): the CPU release build, and the CUDA build
  for sm_89.
- Axon's CPU path on the bella models is closer still (mean KL 0.00204 on Llama 3.2 1B, 0.0063 on Gemma 3 1B).

**The constants** (`scripts/golden.py`):
- **SPREAD_FACTOR 2.5.** Clean Axon reaches 1.66x llama.cpp's spread on max |dlogprob| (Gemma 3 4B), so 1.0 would
  be red on a good day. 2.0 left that model at 0.83 of its limit, measured on a 4090, while the nightly runs on
  bella's sm_120. 2.5 keeps every clean ratio at or under 0.66 and catches every planted bug 2.0 caught.
- **KL_MEAN_FLOOR 0.006**, twice the worst clean mean KL among the models where the floor applies (0.00308, Llama 3.2
  1B CUDA). fddac98 had 0.01, which let RMSNorm eps x10 through on KL.
- **DLP_FLOOR 0.3** nats. Where it applies (Llama 3 8B, Qwen2.5 Coder, gpt-oss), clean Axon stays under 0.11.
- **AGREE_MIN 0.9**; the worst clean top-1 is 0.953.

**What it catches.** Bugs planted in the float64 reference, judged the same way on Llama 3.2 1B (limits from
llama.cpp's spread: mean KL 0.006, |dlogprob| 0.456):

| planted | mean KL | max dlp | verdict |
|---|---:|---:|---|
| RMSNorm eps x10 | 0.0097 (1.6x) | 0.765 (1.7x) | FAIL |
| RMSNorm eps x3 | 0.0019 | 0.206 | pass (inside the noise) |
| Q3_K_L weights for Q4_K_M | 0.089 (15x) | 2.00 (4.4x) | FAIL |
| Q6_K tensors' scale x1.10 | <0 | 0.657 (1.4x) | FAIL |
| Q6_K scale x1.05 / x1.03 / x1.01 | <0 | 0.305 / 0.195 / 0.147 | pass |

- **Gemma 3 1B is coarser.** llama.cpp's own spread there is ten times wider (mean KL 0.019, |dlogprob| 0.85), and
  so are its limits. eps x1000 fails (KL 2.6x, top-1 0.87). eps x100, eps x10, exact GELU for the tanh form, and the
  embedding scale sqrt(d) rounded to bf16 all pass. On such a model (c) is coarse by necessity; (a) and (b) are
  exact on every model.
- **A negative mean KL is real.** The KL is over llama.cpp's top 20 only, so a scale error that sharpens Axon's
  distribution can come out below zero. |dlogprob| still sees it.

## Gates
fmt and clippy `-D warnings` clean. First round (98a0d8d): fast tier (rel on a 4090 pod, dbg on the CPU) and md5
against main. Merge-ready round (main 8e11543 merged in): the full tier (every test of the release and the
debug-assertion builds on GPU hosts, sharded), `gpu_fp8_graded_vs_fp8_ref` as its own job (the interim split while
it holds the GPU test lock through its CPU reference, #232), md5 against main 8e11543 (6 models x 5 prompts, cuda),
and golden.py itself at the head on the compatibility audit's top 10 (a 4090, llama.cpp b11580 CPU, its CUDA build
for the spread). `scripts/golden_selftest.py` in CI. Job ids and verdicts in the PR body.
