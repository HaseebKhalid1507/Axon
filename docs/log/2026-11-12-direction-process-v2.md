# Direction and Process v2: a lean roadmap, the four-week target, methodology's v2  (docs PR)

- Date: 2026-11-12. Branch: docs/direction-process-1112. Base: main 213465a. Status: open
- Pick / design: Haseeb's answers to the 11-07 outside review (Fable 5.2) and its revised plan of 11-12, approved
  with the reviewer's changes; the foreman's brief for this PR.

## What changed
- [roadmap.md](../roadmap.md) rewritten from 653 lines to 150: the mission in Haseeb's words, the four-week target,
  what is shelved (each with one line of why and a link to the evidence), the week-by-week plan, three tracks with
  their metrics, the open issues by track, and what is not planned. History, decided material and numbers are linked,
  not repeated; the old text stays at [`213465a`](https://github.com/HaseebKhalid1507/Axon/blob/213465a12e8402041e39b2d0a565de192f6c38c1/docs/roadmap.md).
- [methodology.md](../methodology.md): "Process v2 (2026-11-12)" at the top, nine rules and a table of what each
  replaces; the old sections sit under "Process v1" (one heading level down, same anchors) with *Superseded* notes.
  [testing.md](../testing.md)'s tier policy carries the same note.
- Cross-references: the mission lines of README, START-HERE and AGENTS.md; START-HERE's north star and "In flight";
  README's Synaps line; models.md and serving.md's roadmap links; the change-request form's example.

## Why
The old roadmap described the 11-06 pause and a datacenter-first order that the 11-12 direction shelves, and it
was four times the 150-line cap the review asked for.

## How it was verified
| check | result |
|---|---|
| relative links and anchors in every tracked `.md`, plus anchors of permalinks into this repo (a script outside the repo) | 0 broken at 213465a, 0 broken on this branch |
| old roadmap anchors linked from history, results, experiments-rejected, logs, design, reviews, benchmarks-gpu | resolve through `<a id>` anchors in the roadmap's "Old section links", which go on to the `213465a` text; those files are not edited (logs are immutable, results and history refresh at tags) |
| roadmap length | 150 lines |
| numbers | each copied from a doc in the repo, linked where it is quoted; ESTIMATED where it is arithmetic |

## What didn't work / left open
- The 11-07 review and its 11-12 addendum are not in [reviews/](../reviews/README.md); a curated copy is the
  foreman's call (it holds personal and budget specifics).
- `--cache-ram auto`'s "4 GiB or ¼ of free RAM" is written as given; the flip's PR should say which of the two wins.
