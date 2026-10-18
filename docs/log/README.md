# Engineering log

One file per PR and per NO-GO experiment, written by whoever did the work. The long-form history up to
2026-10-16 is in [../history.md](../history.md) (its last two phases summarize the merges to 2026-10-18); rejected experiments up to then are in
[../experiments-rejected.md](../experiments-rejected.md). From now on, new entries go here, so parallel PRs
never edit the same file.

## Convention

- **One new file per PR:** `docs/log/YYYY-MM-DD-<slug>.md`. The date is the day the PR is opened (EDT); the slug
  is short and unique, usually the branch name without its prefix (`2026-10-17-chunked-prefill.md`).
  Never edit another PR's entry, except to append a dated "Update" line (merged, reverted, superseded).
- **NO-GO experiments get an entry too**, in a docs-only PR or bundled with the next PR from the same
  track. A rejected idea with numbers is worth more than a merged one without them.
- Entries don't replace the PR body. They are the durable, greppable version of it.

## Template

```markdown
# <title>  (PR #N | NO-GO)

- Date: YYYY-MM-DD. Branch: feat/<x>. Base: main <sha>. Status: open | merged <sha> | rejected | superseded by #M
- Pick / design: <which round, spec or issue asked for this>

## What changed
Files and behaviour, in a few lines. Flags and defaults.

## Why
The measurement or problem behind it, with numbers and the host.

## How it was verified
| gate | result | job ID / log |
|---|---|---|
| release suite | passed/failed/ignored | |
| debug-assertion suite | | |
| md5 of `gen`, main vs candidate, same host | n/N equal | |
| clippy -D warnings, rustfmt | | |
| extra (accuracy A/B, cross-arch, ...) | | |

## Results
Before → after, host named, rounds, medians, labelled MEASURED or ESTIMATED. Always both pp and tg
(or say why one doesn't apply). Thin or noisy results are stated as such.

## What didn't work
Variants tried and dropped, with numbers. Leads not taken, and what would settle them.
```

## Rules of thumb

- Numbers without a host, a commit and a round count don't go in.
- "Ahead" or "faster" needs the comparison in the same entry.
- If a gate was still pending when the PR was opened, say so, and add an "Update" line when it lands.
