# gpt-oss with tools: a reply with no header streams as the answer  (PR #136)

- Date: 2026-10-25. Branch: fix/harmony-unheaded. Base: main b2068e7. Status: open
- Pick / design: issue #114 (from #111's round-2 review, SHOULD 4) and its comment (#111 round 3, SHOULD 6).

## What changed
`src/chat.rs` (`Channels`, with tools only), `src/serve.rs` (`TextOut::harmony`), tests in `src/serve/tests.rs`
(`harmony_calls_from_channels`), a README line and the `serve/tools.rs` module doc. No flag.

- The reply's first role section (what follows the prompt's `<|start|>assistant`) is held only while it may still
  be a header (`maybe_header`): blank, then (a prefix of) ` to=` or `<|constrain|>`, at most 512 bytes. The first
  byte that rules that out releases the held text as the answer (`Channels::take_released`), and the rest of that
  message streams as `Answer` through the stop matcher, as it does without tools.
- `<|end|>` / `<|start|>` that cut held text release it as the answer (they used to drop it).
- A role-section recipient counts only right after the role (`assistant to=functions.X`, the template's form), not
  anywhere in the role section. A channel header (`<|channel|>commentary to=functions.X ...<|message|>`) names a
  call as before, also after released text.

## Why
With tools, gpt-oss's splitter held every byte after `<|start|>assistant` in the role buffer until a marker, so a
reply that never opened a header (1) was dropped if `<|end|>` or `<|start|>` came next, (2) didn't stream (it came
out whole in `finish`), and (3) ignored stop strings (the hit was discarded in `finish`: the reply ran to the token
limit and reported `length`). Without tools the same text streams as the answer from its first byte. gpt-oss
opens every reply with a header in practice, so this is rare, but the path claimed to match the one without tools.
SHOULD 6: `call_recipient` searched the whole role section for `to=functions.X`, so headerless prose naming a
function, followed by a real `<|message|>` token, made a call.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| unit (`harmony serve::tools::`), release, CPU | 17 / 0 / 0 | 1025-054004-c566 |
| the new assertions on main's code (fix stashed, tests kept) | FAIL at "streamed" (the reply came out as 1 piece) | 1025-054247-57c3 |
| `serve::` + `chat::tests::harmony`, release, bella | pending | 1025-055425-82ff §1 |
| `serve::` + `chat::tests::harmony`, debug assertions, bella | pending | 1025-055425-82ff §2 |
| md5 of `gen`, main b2068e7 vs candidate, same host (cuda f16, cuda q8, cpu) | pending | 1025-055425-82ff §3 |
| serve A/B, main vs candidate: 17 cases × 6 models incl. gpt-oss-20b with tools, + 6 concurrent | pending | 1025-055425-82ff §4 |
| clippy -D warnings (lib, lib tests, bin, example, integration test), rustfmt | clean | local |

## Results
No performance change is expected or claimed: the change is in text splitting after sampling. Bits: the model's
tokens are untouched; requests without tools never reach the new code (`Channels::with_tools` gates every branch);
replies that open a header first take the same path as before (the serve A/B is the evidence).

## What didn't work
- Deciding only at the end (`take_unheaded` alone, #111's fix) is what left (1)-(3). A fixed byte count alone
  (release after N bytes with no header) would delay every short answer by N bytes, so the decision is made on
  content first, with the count only as a cap for a role section that stays ` to=...`-shaped.
