# Chat prompts follow each GGUF's own template, token for token (PR #246)

- Date: 2026-11-13. Branch: fix/chat-templates. Base: main 4f347a3. Status: open (PR #246).
- Fixes #240, #238, #241, #239, all filed from the compatibility audit (#235: four popular families framed
  differently from llama.cpp). Also fixes #249 (the DeepSeek-R1 Qwen2.5 distills were refused at load). Shady's
  review (7/10, SHIP-AFTER-FIXES) was applied in the second round, below.

## What changed
- **`chat::Framing::from_model`** keeps choosing the format from the vocabulary's special tokens. It then reads the
  variant from fixed substrings of the file's `tokenizer.chat_template` (no Jinja). Without a template, the
  defaults are as before (Llama 3.1, DeepSeek V3.1):
  - Llama 3 is dated iff the template contains `Cutting Knowledge Date` (#241: Llama 3.0 has no date lines, and a
    system block only for a given system message).
  - DeepSeek is V3.1 iff it contains `<｜Assistant｜></think>` (#238: R1 and V3 open a reply with `<｜Assistant｜>`
    alone, so R1 reasons). BOS goes in iff the template writes `bos_token`, even when `add_bos_token` is false
    (R1-0528's).
  - An earlier reply of a thinking model is cut by `ThinkStrip`. It keeps the text after the last `</think>`, or
    after the first for V3.1 (`split('</think>')[1:] | join('</think>')`), and drops the newlines after it iff the
    template says `lstrip(` (Qwen3 does; R1 keeps them). A template that never mentions `</think>` keeps the whole
    reply (Qwen3-Coder).
- **#240:** `Framing::frame("assistant")` is self-contained for DeepSeek: `<｜Assistant｜>[</think>]{text}<eos>`. Serve
  used to glue an earlier reply onto the user turn.
- **#239, `chat::default_system`:** the template's own default system message, if it is a fixed text (Qwen2.5's,
  Qwen2's, Qwen2.5-Math's, Kimi's). Qwen1.5, Qwen3 and the R1 distills have none. Qwen2.5's text applies to `qwen2`
  only when the file has no template. `template_default_system` is a lexical scan of the template:
  - The text must be one string literal's whole value (escapes resolved), or template text outside the tags with no
    Jinja in it.
  - The innermost `if` / `elif` / `else` around it must test a role against `'system'`. So a concatenation
    (`'...system\n' + x + '<|im_end|>'`) and a tools-only block don't count.
- **`axon chat` and serve share all of it.** `axon chat` drops an earlier reply's reasoning by the same `ThinkStrip`,
  on the reply's tokens. Before the review, chat kept R1's whole reasoning in the conversation (shady's MUST 1).
- **Tools** need a template that renders them: `<tool_call>` in a ChatML template, or `ipython` in a Llama 3 one. For
  Qwen2, Qwen1.5 and Llama 3.0, a request with tools or with a tool turn is now a 400. It used to render in Qwen2.5's
  or Llama 3.1's format. Qwen2.5's tool block takes the template's default system message.
- **Metadata ids are range-checked.** A `bos_token_id` past the vocabulary is none, as the tokenizer reads it. A
  framing whose end-of-turn marker is an EOS past the vocabulary (DeepSeek, Mistral) is no framing (MUST 2): the id
  would have gone into every prompt, and the model's embedding panics on it.
- **`tokenizer::pre_of`** accepts pre-tokenizer `deepseek-r1-qwen` as Qwen2's, as llama.cpp 4413a19 maps it (#249).

## The reference: Hugging Face, token for token
`scripts/chat_template_fixtures.py` writes `tests/fixtures/chat_templates.json` (537 KB). For each family:
- It range-reads the GGUF's header from Hugging Face at a pinned commit, giving the template and the vocabulary.
- It renders each conversation with transformers 5.22.2's `apply_chat_template(chat_template=<the GGUF's>,
  add_generation_prompt=True)`.
- It tokenizes the result with the HF repo's `tokenizer.json` (tokenizers 0.23.3, pinned commit).
- It keeps a sparse vocabulary: every special and byte token, plus every token and merge that is a substring of a
  rendered piece. For those pieces, that gives the full vocabulary's ids.

The test builds a header-only GGUF per family, so it needs no model and no network.

**16 families, 148 conversations:** qwen2.5, qwen2.5-coder, qwen2, qwen1.5, qwen3, qwen3-coder, llama3, llama3.1,
llama3.2, gemma3, gpt-oss, deepseek-r1-0528-qwen3, deepseek-v3.1, deepseek-r1-distill-qwen,
deepseek-r1-distill-llama, deepseek-v3. The cases:
- one turn and multi-turn, with and without a system message;
- whitespace around every message;
- an empty system message, alone and multi-turn;
- an earlier reply with its reasoning, and one with two `</think>` (the thinking families);
- Qwen3 with thinking off;
- tool turns where the template has them.

Mistral v0.3 is left out by design: Axon follows mistral_common's tokens, and the GGUF's older template disagrees with
them (pinned by chat.rs's mistral tests and `tokens_mistral.json`).

`serve::template_tests`:
- `chat_templates_match_hf`:
  - checks that the sparse vocabulary reproduces HF's ids;
  - renders every case through serve (`Template::render`, or `tools::render` from the client's exact request bytes);
  - feeds every non-tool case through `axon chat` turn by turn (`Chat` on a model that replies from a script);
  - expects HF's ids from both: 148 serve renders, 124 chat runs.
  - Chat is skipped where an earlier reply has whitespace around it, because chat keeps the model's own reply tokens
    while Llama's and Gemma's templates trim them.
  - V3.1's two-`</think>` case is compared as text: its template keeps the second `</think>` as text, which HF's
    tokenizer then parses as the special token, while Axon never turns message text into special tokens.
- `message_text_never_forges_turns`: in every family, through serve (thinking on and off), chat and the tools path,
  message text that spells the vocabulary's control tokens (and `<think>`, `</think>` and the tool markers) yields
  exactly the special ids of the same conversation in plain words. HF can't be the reference for this, because it
  parses special tokens in text.
- `broken_special_ids_never_reach_a_prompt`: R1-0528 with `bos_token_id` = 2^31 renders without BOS, through serve
  and chat; the scripted model asserts every id is in the vocabulary. With a broken `eos_token_id`, the file has no
  chat framing and nothing panics.
- `tools_need_the_template`: Qwen2, Qwen1.5 and Llama 3.0 refuse tools and tool turns; a file without a template
  keeps the vocabulary's tool family.
- `default_system_from_the_template`:
  - the families' defaults;
  - three concatenation shapes, which give none;
  - a tools-only block, alone and right after a system test, which gives none;
  - Qwen2.5-Math's literal with braces and `\\boxed`;
  - Moonlight's and Kimi K2's template-text defaults.

**The tests fail on the bugs.** A build with chat's strip off for DeepSeek and the BOS filter removed (job
1113-053745-b3b5, jade) fails on two counts:
- `chat_templates_match_hf` reports exactly the 8 chat cases of the R1/V3.1 reasoning conversations. R1 had fed back
  `<think>\nSimple sum.\n</think>\n\n4.`, where the template has `\n\n4.`.
- `broken_special_ids_never_reach_a_prompt` fails too.

The fixed build passes all 5 (1113-053516-f767).

## Results
On main 4f347a3 the fixture's first round (68 cases) differed in 16 of the 55 non-tool cases that load: qwen2 2,
qwen1.5 2, llama3 4, deepseek-r1-0528 5, deepseek-v3.1 3. The R1 distill was refused at load. With this PR all 148
cases match HF, and the 124 chat runs do too.

The first round's gates, at 5c00ed9 on an RTX 4090 pod against main's binary (`axon-3014eeb`):
- fast tier, release: 826 / 0 (1113-042851-16ef, with the interim `fp8_graded` skip of #232/#247);
- fast tier, debug + CPU: 845 / 0 (1113-033053-cdfc);
- md5 of `axon gen` vs main: 0 differ (1113-033104-d671);
- serve md5, 7 models × 4 conversations: 0 / 28 differ (1113-035003-ec85);
- golden.py (#229) on the nightly top 10, llama.cpp b11580 CPU as the reference (1113-041027-d4fb): 9 / 10 pass with
  the chat prompt equal id for id; deepseek-r1 and llama3 were the audit's DIFFs. mistral's logits match (KL
  0.00094) and only its prompt differs, by design.

The second round's regate (fast tier, serve md5, the template test, golden on deepseek-r1, llama3, two controls and
the 7B R1 distill) is in the PR.

## Not done (follow-ups)
- #250: `no_think` on DeepSeek. V3.1's prompt gets Qwen3's empty block after its own `</think>`, and `axon chat
  --no-think` on R1 says it isn't a thinking model. Both are already on main.
- #251: templates whose generation prompt opens the reasoning block: QwQ (`<|im_start|>assistant\n<think>\n`) and
  R1's Feb-2025 HF template. Not in the fixture for that reason.
- #248: Qwen's official Qwen2.5 GGUFs carry the pre-2024-09-19 template (doubled braces in the tool instruction).
  Axon renders single braces, as the fixed HF template, ollama and the re-quants do.
- #242: text-marker formats (llama2, codellama, tinyllama, llava-mistral). A small sandboxed Jinja subset is proposed
  in the PR instead of more substrings.
