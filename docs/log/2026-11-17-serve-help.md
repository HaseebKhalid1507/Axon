# serve --help exits 0, and serve's help states every flag it takes  (PR #280)

- Date: 2026-11-17. Branch: fix/serve-help. Base: main 1a1d751, main 29fd8f8 merged in at 761863e. Status: open
- Pick / design: issue #252 (shady's SHOULDs on #225, the `--help` refresh); the foreman's 11-17 ruling: bug-fix scope
  only (serve --help exit 0, SERVE_USAGE current, missing or mislabelled flags and caveats, `--ctx` included), with a test
  that pins `serve --help`'s exit 0 and that every flag the parser accepts is in the help.

## What changed
- **`axon serve -h | --help` exits 0** and prints `axon --help`'s text on stdout (`cli::help()`, shared by both). The
  check comes first in `serve::run`, before the model argument is resolved. That matters because a name that isn't
  pulled yet is a usage error there, so `axon serve qwen3-8b --help` must not get that far. `-h` or `--help` anywhere
  among serve's arguments counts. A model path or name is not required. Before, `serve --help` was a usage error
  (`fail(SERVE_USAGE)`, exit 2), and `serve MODEL --help` was "unknown argument '--help' for serve" (exit 2), or the
  not-pulled error.
- **`SERVE_USAGE`**, the line a serve usage error prints:
  - drops `--prefill-ms`, the old name of `--step-ms`, which is still accepted and said in `--step-ms`'s entry;
  - adds `--step-ms`, `--step-tokens`, `--admit-order sjf|fifo`, `--admit-age-ms`, `--prefill-lanes` and `[rope flags]`;
  - points to `axon serve --help`.
- **`--help`, issue item 2:** entries under `flags:` for `--host`, `--port`, `--alias`, `--max-tokens`, `--cors`,
  `--max-conns`, `--max-queue`, `--think-inline` and `--quiet`, with their defaults and behaviour from `serve.rs` and
  docs/serving.md. Under `environment:`:
  - `AXON_SHARE_KV` (#267's kill switch) and `AXON_KV_RAM_ALLOC`, worded from docs/environment.md;
  - a header pointer: "these and the rest, serve's among them: docs/environment.md; serve's flags in full:
    docs/serving.md".
- **Item 5, `--ctx`:** it now opens "chat, serve, gen: the context in tokens, which means something different in each",
  then one clause each:
  - chat: the conversation's;
  - serve: each request's, the most one slot holds. With paged KV (the GPU's default) it is a slot's cap, not memory
    set aside, and the slots share one pool of pages;
  - gen: as before.
- **Items 3 and 4, the footer:** deepseek2 "MLA GGUFs" (legacy non-MLA V2 files are refused: docs/models.md). Native FP8
  with `--numerics fast` "experimental, parked: docs/roadmap.md" (roadmap.md says parked; models.md says it fails its
  bars on bella).
- `--quiet`'s entry and docs/serving.md say what it hides: the startup banner, its notes (api key, `--datacenter`) and
  the per-request lines. Warnings (no paged pool, one request at a time, constrained tool calls off) still print.
- The serve synopsis in `--help` gains `[rope flags]`. serve takes them; the rope entry already says so.
- **For the test, no behaviour change:**
  - `serve::run`'s parser reads its switches and value flags from two consts, `SWITCHES` and `VALUE_FLAGS`: the same 4
    and 25 flags, in the same order, as match guards instead of literal arms.
  - `SampleArgs::is_flag` reads `sample::FLAGS` (the same 7).
  - `OLD_FLAGS` (`--prefill-ms`) is `cfg(test)`.
- **The value match has no default arm** (shady's SHOULD): `--draft` has its own arm, and a flag in `VALUE_FLAGS`
  with no arm is `unreachable!`. On main the last arm was `_ => --draft`, so a value flag added to the list without an
  arm would silently set the draft length. No test drives each value flag through `run` (it loads a model), so the
  panic is the guard; with the help test, a flag added to the list needs an entry and an arm.
- `serve_help_lists_every_flag` checks each entry's label, the text before its first `:` ("gen, chat, serve"), for
  the word `serve`, where it checked the whole entry ("server" matched).
- CHANGELOG entry.

## Why
#252, from shady's review of #225. A help that exits 2 is a usage error to every script and wrapper that probes it. A
usage line that names a renamed flag and misses five current ones sends people to the wrong flags. Nine flags had no
entry at all. `--ctx`'s serve meaning was buried under a `chat:` label, and the footer offered native FP8 and deepseek2
without the caveats docs/models.md has.

## How it was verified
**At the merged head 761863e** (77ae344 with main 29fd8f8 merged in, #274's constrained tool calls among it):

| gate | result | job ID / log |
|---|---|---|
| build: release bin, release and debug-assertion test bins (rustc, lib layout) | OK | bella `~/.cache/axon-eval/logs/buildall-761863e.log` |
| clippy `-D warnings` (lib, tests, bin), rustfmt | clean | same log |
| release suite, 3 shards (jade, CPU; `AXON_SKIP_FP8_MODEL_TESTS=1`) | PENDING | 1117-190853-0efb, -44f0, -d833 (group 9523) |
| debug-assertion suite, 3 shards (jade, CPU; `AXON_SKIP_FP8_MODEL_TESTS=1`) | PENDING | 1117-190853-5523, -33c1, -e08e (group 81b3) |
| mutants H1–H7, the candidate as control (`mutants2.sh`, jade, CPU) | PENDING | 1117-191442-ef07 |
| md5 of `gen` | not run: no generation code changed (`sample::FLAGS` is the same 7 names `is_flag` matched) | |

The targeted run isn't repeated: the debug-assertion suite runs `cli::tests::` and `sample::tests::` too. Before
queueing, the three help tests and `mutants2.sh` ran on bella (fixture-only, no model): the candidate passed and each
of H1–H7 failed its test.

**The merge.** Four files conflicted. `--no-constrain` (#274's switch) is now in `SWITCHES`, `SERVE_USAGE` and the
synopsis; its `--help` entry, labelled serve, came from main. docs/serving.md keeps main's `--no-constrain` sentence
and this PR's `--quiet` text, and CHANGELOG keeps both entries. #274 added no other serve flag. Its environment
variables, `AXON_CONSTRAIN` and `AXON_CONSTRAIN_STALL`, are in `--no-constrain`'s entry.

**At 77ae344** (before the merge; all PASS):

| gate | result | job ID / log |
|---|---|---|
| build: release bin, release and debug-assertion test bins (rustc, lib layout) | OK | bella `~/.cache/axon-eval/logs/buildall-77ae344.log` |
| clippy `-D warnings` (lib, tests, bin), rustfmt | clean | same log |
| targeted, debug-assertion build: `cli::tests::` (the help tests among them), `sample::tests::` (`is_flag` against `set`) (jade, CPU) | PASS, 30 passed | 1117-104814-0039 |
| release suite, 3 shards (jade, CPU; `AXON_SKIP_FP8_MODEL_TESTS=1`) | PASS, 377 + 295 + 246 passed | 1117-135258-faf9, 1117-135258-a6f0, 1117-135259-271f |
| debug-assertion suite, 3 shards (jade, CPU; `AXON_SKIP_FP8_MODEL_TESTS=1`) | PASS, 377 + 295 + 246 passed | 1117-135259-9883, 1117-135300-a625, 1117-135301-a2ee |
| mutants H1–H7, the candidate as control (`mutants2.sh`, jade, CPU) | MUTANTS PASS: each of H1–H7 failed its test | 1117-135952-83eb |

These replace 1117-104830-f6b3/-ae0d/-974c and -a7e6/-a0f2/-0648, dropped while queued. jade has no FP8
checkpoint, so the reruns set `AXON_SKIP_FP8_MODEL_TESTS=1`; nothing here touches FP8, and the FP8 tests run on bella
in the nightly and postmerge runs. They also replace mutants 1117-105835-0eed, dropped while queued. axon-run appends
`--test-threads 1` to a jade CPU job's argv, scripts included, and `mutants2.sh` had read those two words as binaries.
It now stops at the first `-` argument.

**`serve_help_lists_every_flag`** (cli.rs). The flags serve takes are `SWITCHES`, `VALUE_FLAGS` and `--cors`, minus
`OLD_FLAGS`, plus the sampling flags (`sample::FLAGS`) and the rope flags (`rope::FLAGS`). The test checks:
- serve's synopsis in `--help` lists **exactly** those flags: none missing, and no stale or old name, so `--prefill-ms`
  in the synopsis fails;
- `SERVE_USAGE` lists exactly those flags, plus its pointer to `--help`;
- each of them has an entry under `flags:` whose label, the text before its first `:`, names serve;
- each old name is said in its new name's entry ("--prefill-ms is its old name");
- each sampling flag has an entry in the "sampling flags (gen, chat, serve; ..." section;
- each rope flag is in the `--rope-scaling` entry, which says serve.

A flag added to the parser's lists without help fails the test. A new literal match arm (as `--cors` is) wouldn't,
which is why the test adds `--cors` by hand.

**`serve_help_exits_0`** (cli.rs). The test binary runs itself as a child, running the ignored `serve_help_child`, which
calls `serve::run` on given arguments and requires `Ok` (then `main` exits 0). The parent checks the child's exit
status and that the help is on its stdout. A usage error exits the process, so it can't be tested in-process. Five
argument lists: `--help`, `-h`, `no-such-model-name --help` (a name not pulled), `/no/such/file.gguf --port 1 -h`, and
`/no/such/file.gguf --bogus --help`.

**Mutants** (sources at bella `~/.cache/axon-eval/mut2/H*/src/...`, each a copy of 77ae344's file with one change; release
test builds, `cli::tests::serve_help`):

| mutant | the one change | must fail |
|---|---|---|
| H1-no-help-check | the help check never fires | `serve_help_exits_0` |
| H2-help-first-only | help only as serve's first argument | `serve_help_exits_0` |
| H3-help-on-stderr | the help on stderr | `serve_help_exits_0` |
| H4-stale-prefill-ms | `SERVE_USAGE` says `--prefill-ms` again | `serve_help_lists_every_flag` |
| H5-no-host-entry | `--host`'s entry deleted | `serve_help_lists_every_flag` |
| H6-new-flag-no-help | the parser takes a new `--new-flag` | `serve_help_lists_every_flag` |
| H7-cors-mislabelled | `--cors`'s entry labelled `chat:`; its text still says "server", which the old `contains("serve")` passed | `serve_help_lists_every_flag` |

H1–H6 are 0e95bda's, carried to 77ae344 by their diffs; H7 is new with the label check. At 761863e all seven are
77ae344's one-line diffs applied to the merged tree (bella `~/.cache/axon-eval/mut2/patches/`). No mutant removes the
`--draft` arm: no test drives a value flag through `run` (it loads a model and serves), so nothing would catch it but
the panic at the first `--draft`, which is the point of the `unreachable!`.

## Results
77ae344: every gate PASS. 761863e, the merged head: PENDING (gatewatch `eval-761863e-gates`). 0e95bda's (55f0,
770e/df4a/c812, a4cb/4560/2186, 13e1) were dropped while queued: shady's round changed `src/`. Help text and a parser refactor: no performance numbers apply.

## Not done
- `axon chat --help`, `axon gen --help` and `axon info --help` are still usage errors (exit 2). #252 is about serve; the
  same one-line check would fix them, in another PR if wanted.
- `-h` or `--help` given as another flag's value (`--alias --help`) is taken as the help. No real value is spelled that
  way.
- `axon serve help` (the bare word the top-level command takes) is still read as a model name, the not-pulled error.
  Left as it is (shady's NIT 7, the foreman's call).

## Review
shady 9/10 SHIP on cd2c535 ([comment 6952843853](https://github.com/HaseebKhalid1507/Axon/pull/280#issuecomment-6952843853)).
Fixed at 77ae344: the SHOULD (`--draft`'s own arm, `unreachable!` fallback), NIT 1 (the PR body's claim now says "the
parser's lists", as this log did), NIT 2 (label check, H7), NIT 3 (`--quiet`), NIT 4 (the GPU's default), NIT 5
(roadmap.md), NIT 6 (CHANGELOG). NIT 7: above.
