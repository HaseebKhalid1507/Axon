# Constrained decoding, PR A: Hermes tool calls held to JSON behind `AXON_CONSTRAIN=1` (PR #264)

- Date: 2026-11-14. Branch: feat/constrain-a. Base: main 0497708. Status: open (PR #264).
- Pick / design: specs/constrained-decoding.md (overnight workspace) §5 row A, for #259 (malformed Hermes calls from
  Qwen2.5-7B under codex) and #116. The maintainer's rulings on the spec's §6 are built in where PR A touches them:
  (1) no global `NUMERICS_VERSION_EXACT` bump, the constraint goes in the request's `serve` identity field;
  (2) lazy re-arm after `</tool_call>`; (5) the Parser is to read `</tool_call>` inside a JSON string as text (PR B),
  and PR A exposes the state it needs (`Constraint::in_string`). (3) and (4) are PR C's.

## What changed
- **`src/constrain/`** (new).
  - `pda.rs`: a byte-level pushdown automaton. Nodes have byte-set edges, special-token edges and call edges (push a
    return node), and a node marked `ret` pops when no edge takes the symbol. `Builder::finish` checks that it is
    deterministic (alternatives start disjoint, through calls), has no left recursion, never calls a nullable node,
    and can finish from every node. The JSON library is RFC 8259, the same language as `tools::parse_json`
    (checked on 10k values): strict numbers, the 8 escapes and `\uXXXX`, raw bytes < 0x20 refused in strings, bytes
    >= 0x80 taken as `parse_json`'s lossy callers read them. Containers nest at most 64 frames (`MAX_DEPTH`).
  - `vocab.rs`: each token's bytes (`decode_bytes`). Special tokens (control, user-defined, unknown) are one
    symbol each, never bytes, so none of them (EOS, `<|im_end|>`, `<tool_call>`) can be picked inside a string. The
    other tokens go in a byte trie stored in preorder, so a walk skips a refused prefix's whole subtree.
  - `mask.rs`: masks cached per (node, stack top, room to the depth cap). A walk from the cached state knows the
    top frame only. A token whose bytes pop below it (`"}}`) is "deep" and is checked against the live stack when
    the mask is used. `TokenMask` = the cached bits (shared `Arc`) + the deep tokens that pass. 64 MiB cache cap
    per grammar.
  - `mod.rs`: `Grammar::hermes` (`ws* {ws* "name" ws* : ws* STRING ws* , ws* "arguments" ws* : ws* OBJECT ws* }
    ws* </tool_call>`, the close as its token or spelled out), and `Constraint`: free until `<tool_call>`, engaged
    until the close, then free and armed again (ruling 2). A grammar needs a byte-complete vocabulary (every byte a
    JSON text can hold is a token by itself, as in byte-level BPE), else it isn't built. Then no mask is ever empty.
- **Sampling** (`sample.rs`): `cands_masked` is the reference: the repeat penalty as `cands` applies it, then the
  `k` best allowed tokens in `top_k_ref`'s order. `masked_prefix` keeps the allowed tokens of a backend's CAND_MAX
  best when there are enough of them (`min(k, allowed)`), and says so. `generate_constrained` is `generate`'s loop
  with a constraint, greedy or sampled. A free position takes the unconstrained head (`forward_argmax` /
  `forward_cands`), so a constrained request's tokens equal an unconstrained one's up to its trigger. An engaged
  position takes `forward_cands_masked`; greedy uses `GREEDY_SPEC` (k 1, no penalty). Drafting: the guesses
  advance a clone of the constraint, which gives each verify row its mask, and the draft is cut at the first guess
  that row's mask forbids (it could never be accepted).
- **Backends** (`lm.rs`): `Lm::forward_cands_masked` / `verify_cands_masked`. The defaults select on the host
  from the logits. The CPU Qwen2 verifies in one batched pass. `CudaQwen2` overrides both: sample.cu selects the
  CAND_MAX best, and the allowed ones are kept. When too few are allowed, the step's logits row comes back from the
  device and the host selects (`MASK_FALLBACKS` counts it). The spec's open point: the solo `cands` path's row
  stays readable in the cache's `logits`, because the sampling step's head GEMV writes it there and `samp_topk`
  reads it as `const float*` ("the logits buffer is left alone"). The batched masked verify takes the device path
  when its rows fit one head launch (<= `HEAD_SUB` = 16 rows, all still in the scratch), else row by row.
  `RowHead::Sample` gains `mask`. A pool step asks CAND_MAX for masked rows and falls back to `step_row_logits`.
- **Serve**: `Opts::constrain` / `AXON_CONSTRAIN=1`. The grammar is built once at load, for a Qwen2.5-family
  template. A request with tools and `tool_choice` auto gets a `Constraint` (`Job.constraint`). It goes along to
  the solo path (`generate`), into the batch on handover, through preemption and refeed. A constrained reply's
  `system_fingerprint` adds `constrain hermes-json <16 hex>` to its `serve` field (`identity::constrain_field`,
  `Identity::with_serve_extra`). Unconstrained replies keep the server's fingerprint. A request whose grammar
  allowed nothing would fail alone (`End::Failed`). No `.cu` edits; `serve/tools.rs` untouched.

## Why
#259: in the codex × Qwen2.5-7B-Instruct Q4_K_M acceptance run (job 1113-010041-0dce, axon-3014eeb), 8 `<tool_call>`
blocks came back as text because they weren't JSON: 4 unescaped quotes inside a string, 2 invalid escapes (`\(`,
`\'`), 1 raw newline in a string, 1 second `<tool_call>` inside an unclosed call. They're in
`tests/fixtures/constrain_259_malformed.json` with the byte where each must be refused. Under the grammar each
is refused at that byte, and the mask there doesn't hold the token (`malformed_259_impossible`).

## How it was verified
Code at 4af6874 (= 20849a8 + CHANGELOG). 977972c only fixes the ignored bench `constrained_step_cost_cuda`. Jobs
through axon-run. "pod1"/"pod2" are RTX 4090 (sm_89) pods; bella is an RTX 5070 Ti Laptop (sm_120).

| gate | result | job ID / log |
|---|---|---|
| clippy -D warnings (lib, tests, bin; debug and release cfg), rustfmt | clean | axon-gate `constrain-4af6874-fast-clippy.log`; 977972c re-run clean |
| fast tier, release suite (minus test-fast-skip.txt) | 879 passed, 0 failed, 65 ignored | 1114-074845-c0fd (pod1) |
| fast tier, debug-assertion suite (CPU queue) | 897 passed, 0 failed, 65 ignored | 1114-074845-6d78 (pod2) |
| full release suite, every test incl. big models, 3 shards | PASS: 962 tests, 897 passed, 0 failed, 65 ignored (shards 365 + 285 + 247 passed) | 1114-083143-9e1b: 34d4 (pod1), 8b95 (bella), 57fa (pod2) |
| md5 of `gen` -n 64, base 0497708 vs candidate vs candidate with `AXON_CONSTRAIN=1`, same host | 50/50 equal (CPU 15, CUDA f16 20, CUDA q8 15: 0.5B, Qwen3-0.6B, Llama-3.2-1B on each, Qwen2.5-7B on CUDA f16; md5-gen.sh's 5 prompts, two long), 0 nonzero rc | 1114-074613-1847 (bella) |
| model-gated contract, 0.5B (unit tests too) | 11 passed | 1114-073333-32e2 (pod1) |
| model-gated contract, Qwen2.5-7B-Instruct Q4_K_M | 3 passed | 1114-073601-8253 (pod1) |
| serve end to end + identity tests | 12 passed | 1114-074910-f297 (pod1) |

## Results
MEASURED. Host and job named per line.

**Mask cost per token** (`Constraint::mask` + `accept`, cache warm; the spec's bar is median <= 20 us and p99 <=
100 us at a 152k vocabulary):

Exclusive CPU-only job on a 4090 pod's node (pod1), 977972c (the mask code is 4af6874's), job 1114-085305-aa2f:

| vocabulary | tokens timed | median | p99 | max | cold first pass | string-body compile | cache entries |
|---|---|---|---|---|---|---|---|
| Qwen2.5 real, 151,936 logits, 333,588 trie nodes; the call as the tokenizer writes it | 75 x 20 | 0.08 us | 4.86 us | 8.4 us | 11.8 ms | 2.79 ms (147,580 allowed, 0 deep) | 29 |
| the same, a token per byte (the worst case for states visited) | 181 x 20 | 0.06 us | 4.41 us | 7.2 us | 12.1 ms | 2.83 ms | 64 |
| synthetic 151,717: every byte + 151k random pieces over `{}[]":, \n\\abcxyz019.-+eEtrunl/<>` (many tokens pop several frames: deep, checked live) | 181 x 20 | 0.37 us | 85.61 us | 181.5 us | 11.4 ms | 1.84 ms | 64 |

All three meet the bar. On the real vocabulary, a warm mask costs a hash lookup plus a handful of live checks. The
synthetic vocabulary is adversarial: its p99 comes from states where tens of thousands of random pieces are deep.
The cost that remains is the first visit to each state: about 12 ms for a whole call's states, once per grammar per
process, then cached (64 MiB cap). `Vocab::new` at load: 41.8 ms.

**Decode step, end to end** (`constrained_step_cost_cuda`, exclusive 4090 on pod2, 977972c). One constrained greedy
reply, each position run as a greedy step, a sampled k=40 step and a constrained step on three caches fed the same
tokens, medians over 10 rounds:

| model | engaged tokens | greedy | sampled k40 | constrained | vs greedy | fallbacks | job |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M | 20 x 10 | 0.921 ms | 0.926 ms | 0.959 ms | +4.1% | 0 of 200 | 1114-083913-79fa |
| Qwen2.5-7B-Instruct Q4_K_M | 47 x 10 | 5.385 ms | 5.388 ms | 5.422 ms | +0.7% | 0 of 470 | 1114-083914-ab77 |

The 7B's greedy constrained call was
`{"name": "exec_command", "arguments": {"cmd": "sed -i 's/raise KeyError.*/raise ValueError(\"unknown \\\"account\\\"\")/g' ledger/cli.py && echo 'done'"}}`:
the nested escapes that broke #259's calls, written correctly.

**sample.cu at k = 40 vs 256** (`samp_kernel_profile`, pod2 4090, 0.5B row, 1114-083126-6c23). A masked row asks the
device for CAND_MAX = 256, which lifts the step's kmax for every sampled row. `samp_topk` takes 21.5 us at k=40 and
47.1 us at k=256 (argmax: 11.0 us), against a 0.9 ms 0.5B step and a 5.4 ms 7B step. So +26 us per step that
has a masked row: 2.8% of the 0.5B step, 0.5% of the 7B's (the spec ESTIMATED "tens of us").

**Fallbacks**: 0 in 670 engaged tokens of real calls (above). The model tests force them with masks that allow 1-4
random tokens, and assert the counter grew.

## What didn't work / notes
- **A toy vocabulary without every byte got stuck.** With `{"` + `n` + `a` and no `m` token, the literal `"name"`
  couldn't be finished, and the mask came up empty. Real byte-level vocabularies have every byte, so a grammar now
  requires `Vocab::byte_complete`, and the toy vocabularies got the bytes.
- **The first oracle for "every call parses" scanned the text for `</tool_call>`.** That split calls whose string
  values spell `</tool_call>`: 4 of 88 on the toy model. The automaton reads them as string content, so the test now
  takes the call's end from a shadow `Constraint`. That is ruling 5's case, for PR B's Parser.
- **Observed, not fixed here:** a 0.5B call sampled at temperature 1.2 with top_k off ran into spaces inside a
  string until `max_tokens` (finish_reason `length`). The grammar guarantees a call is valid JSON, not that it
  ends. #260's runaway bound is the place for that.
- The step bench's first version took its reference tokens from a penalized decode, while its timed step is
  greedy. On the 7B they parted (job 1114-083127-af8e failed on that test-only mismatch); 977972c fixes the bench.
