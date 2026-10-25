# gpt-oss with tools: a reply with no header streams as the answer  (PR #136)

- Date: 2026-10-25. Branch: fix/harmony-unheaded. Base: main b2068e7. Status: open
- Pick / design: issue #114 (from #111's round-2 review, SHOULD 4) and its comment (#111 round 3, SHOULD 6).

## What changed
`src/chat.rs` (`Channels`, with tools only), `src/serve.rs` (`TextOut::harmony`), tests in `src/serve/tests.rs`
(`harmony_calls_from_channels`), a README line and the `serve/tools.rs` module doc. No flag.

- The reply's first role section (what follows the prompt's `<|start|>assistant`) is held only while it may still
  be a header (`maybe_header`): blank, then (a prefix of) `to=RECIPIENT` followed only by blank and (a prefix of)
  `<|constrain|>json` / `json`, or blank then `<|constrain|>json`; at most 512 bytes. The first
  byte that rules that out releases the held text as the answer (`Channels::take_released`), and the rest of that
  message streams as `Answer` through the stop matcher, as it does without tools.
- `<|end|>` / `<|start|>` that cut held text release it as the answer (they used to drop it).
- A role-section recipient counts only right after the role (`assistant to=functions.X`, the template's form), not
  anywhere in the role section, and the role must be exactly `assistant`. A channel header (`<|channel|>commentary to=functions.X ...<|message|>`) names a
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
| gate | code | result | job ID / log |
|---|---|---|---|
| unit (`harmony serve::tools::`), release, CPU | 1fccb6f | 17 / 0 / 0 | 1025-054004-c566 |
| the new assertions on main's code (fix stashed, tests kept) | main | FAIL at "streamed" (the reply came out as 1 piece) | 1025-054247-57c3 |
| `serve::` + `chat::tests::harmony`, release, bella sm_120 | 1fccb6f | 122 / 0 / 2 | 1025-055425-82ff §1 |
| `serve::` + `chat::tests::harmony`, debug assertions, bella sm_120 | 1fccb6f | 122 / 0 / 2 | 1025-055425-82ff §2 |
| md5 of `gen`, main b2068e7 vs candidate, same host (cuda f16 32, cuda q8 12, cpu 12) | 1fccb6f | 56/56 equal | 1025-055425-82ff §3 |
| serve A/B main vs candidate: 17 cases + 6 concurrent × 6 models incl. gpt-oss-20b with tools | 1fccb6f | 23/23 identical on each model | 1025-055425-82ff §4 |
| `harmony serve::tools:: serve_gptoss`, release + debug, bella | 3b4f95f | 18 / 0 / 0 release, 18 / 0 / 0 debug | 1025-063406-aa1a §1-2 |
| serve A/B main vs candidate, gpt-oss-20b, 17 + 6 cases | 3b4f95f | **23/23 identical** (22 × 200 + the same #116 400 on both; fingerprints and concurrent cached_tokens equal) | 1025-063406-aa1a §3 |
| clippy -D warnings (lib, lib tests, bin, example, integration test), rustfmt | 3b4f95f | clean | local |

Raw logs: [82ff](https://gist.github.com/HaseebKhalid1507/167f3b86d00baf6ac53505ff502398fe); round 2: [aa1a](https://gist.github.com/HaseebKhalid1507/4b2fe80e4a3424f9621eb3675d4cb484). Review: shady 8/10 APPROVE (round 1); round 2 (3b4f95f) applies
its SHOULD 1 (below) and the `assistant`-exact NIT.

## Results
No performance change is expected or claimed: the change is in text splitting after sampling. Bits: the model's
tokens are untouched; requests without tools never reach the new code (`Channels::with_tools` gates every branch);
replies that open a header first take the same path as before. Evidence: the serve A/B, 23/23 identical bodies on
each of 6 models (gpt-oss-20b with tools included), and 56/56 `gen` md5s (job 1025-055425-82ff). Visible change: a
gpt-oss reply with prose and then a real header call now returns both `content` and `tool_calls` (main dropped the
prose).

## What didn't work
- Deciding only at the end (`take_unheaded` alone, #111's fix) is what left (1)-(3). A fixed byte count alone
  (release after N bytes with no header) would delay every short answer by N bytes, so the decision is made on
  content first, with the count only as a cap.
- Round 1 held anything after ` to=` up to the cap, so ` to=functions.x but ...` streamed nothing for 512 bytes
  (review SHOULD 1). Round 2 releases as soon as what follows the recipient's word can't be ` <|constrain|>json` /
  ` json`.
