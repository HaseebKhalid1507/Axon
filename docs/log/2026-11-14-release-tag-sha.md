# Release workflow: the draft only while the tag names the built commit (#261, #258's must-fix)

- Date: 2026-11-14. Branch: fix/release-tag-sha. Base: main 0497708. Status: open
- Pick: the must-fix in #258 (foreman comment 6874945784), from shady's #257 review (8/10 SHIP, comment
  6874937092). Must land before v0.1.0 is tagged.

## The hole
#257 made "one release per tag" the rule, so the first run to finish wins. Push `v0.1.0` at A (run R1, about
35 minutes cold), move the tag to B (run R2). If R1 finishes first, `--verify-tag` passes, because a tag by that
name exists. R1 makes the draft for `v0.1.0`, which now names B, but the draft holds A's tarball, and R2 is
refused. The hand checks might not catch it: the version string is the same, and the md5 gate passes if the
kernels didn't change between A and B.

## What changed
Workflow, docs, and one new test script; no Rust.
- **pack job:** `commit=$(git rev-parse HEAD)`, a job output, next to `epoch`.
- **build job:** a step that fails unless its own checkout is that commit, so the binary and the pack in the
  tarball come from one commit. It uses `git -c safe.directory=$PWD`, because in the manylinux container the
  checkout may belong to another user.
- **release job:** `needs: [pack, build]`, `BUILT` = the pack's commit. Before the listing:
  - `BUILT` must be 40 hex digits, and the tag must match `^v[0-9][0-9A-Za-z.+-]*$`.
  - `gh api repos/$REPO/commits/refs/tags/$tag --jq .sha` must equal `BUILT`; otherwise exit 1, nothing uploaded.
  - The same check runs again after `gh release create`, because the tag can move during the upload. If it
    fails there, the job goes red and says to delete the draft.
  - The draft's notes end with "Built from commit <sha>".
  - Not GITHUB_SHA: what it holds for an annotated tag isn't documented.
- **releasing.md:** step 6 describes both checks and says a failed upload leaves a half-made draft
  (`gh release create --draft` doesn't remove it): delete it, the tag stays, then "Re-run failed jobs".
  "Before publishing" adds a first check: the tag still names the commit in the draft's notes.
- **`scripts/release/test_draft_guards.sh`:** takes the step's `run:` block out of release.yml (an indentation
  parser, because PyYAML may be missing on the runner; it matches PyYAML's parse byte for byte) and runs it under
  `bash --noprofile --norc -eo pipefail` with a stub `gh` that applies the step's own `--jq`. The pack job runs it.

## How it was checked
No GPU gates: nothing here runs on a GPU or changes a binary. Stop rule in the PR body, written before the
harness existed.
- **test_draft_guards.sh: 15 offline cases.**
  - Match: creates, and the tag is asked about twice.
  - Moved before the upload, API error (422), `null`, an empty SHA: refused, nothing created.
  - `BUILT` empty, short or with a newline, and a tag containing `#` or `/`: refused before any API call.
  - Moved during the upload: created, then fails and says to delete the draft.
  - #257's draft on page 2, and a failed listing: refused.
  - The upload fails: the step fails.
- **Mutations:** six edits that break the step: no pre-check, the comparison always true, no `BUILT` check, no
  post-check, no commit in the notes, an API error ignored. Each turns 1 to 10 cases red. Without the `BUILT`
  check, an empty `BUILT` is still refused by the comparison; only the call count and the message change.
- **`--live`, the real endpoint:**
  - git/git v2.40.0, an annotated tag (object d4ca2e31, commit 73876f48): `BUILT` = the commit creates;
    `BUILT` = the tag object is refused.
  - cli/cli v2.40.0, a lightweight tag: the commit creates; another commit is refused.
  - A missing tag (HTTP 422): refused.
  - Cross-checked by hand against `git ls-remote` `refs/tags/T^{}`.
- **actionlint 1.7.12** (with shellcheck): the same two `SC2012` infos as main 0497708, nothing new. shellcheck
  is clean on the test script at style level.
- **The PR's Release run:** its result is recorded on the PR once the run finishes, not here.
- **Not exercised:** a real tag push (none may be pushed before v0.1.0), or GitHub moving a tag mid-run.
