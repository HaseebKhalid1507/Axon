# Release workflow: the tag check right before Publish; DO NOT PUBLISH on a failed after-check (#265)

- Date: 2026-11-14. Branch: fix/release-publish-check. Base: main be37005. Status: open
- Pick: shady's review of #261 (9/10 SHIP, comment 6877091068): its SHOULD (docs, before v0.1.0) and its nits.
  The last release follow-up before v0.1.0 is tagged.

## The SHOULD: the hand check sat at the wrong end
releasing.md ran "the tag still names the draft's commit" first, then checks that take hours (bella's reference
pack alone is about an hour). A tag moved during those hours is the case the check exists for. Worse, a deleted
tag: the draft has no `--target`, so its target is main, and publishing a draft whose tag is gone creates the
tag at main's tip, holding the old commit's tarball. Now the check runs first and again right before Publish,
and a 422 means don't publish (delete the draft).

Not `--target "$BUILT"`: GitHub's REST docs ("Create a release", and "Update a release" for the existing
target) say the call returns 404 when the target commit adds or modifies files under `.github/workflows/`
relative to the default branch, unless the token can modify workflows, which GITHUB_TOKEN can't. A tag on an
older commit, with main's workflows changed since, would break the release. The workflow says so next to the
create.

## The nits
- **A message that misled.** On a transient API error the pre-check said "the run for the commit the tag names
  makes the draft", but then this run is that run. `tag_names_built` now returns 1 (another commit) or 2 (no
  commit: an error, or an answer that isn't 40 hex digits), and the 2 case says "if the tag still names
  <commit> (a transient error), re-run this job". releasing.md step 6 says it too.
- **The tag deleted during the upload:** a test case now (`"$A FAIL"`).
- **`concurrency` on the release job**, keyed by `github.ref`, `cancel-in-progress: false`. Two runs of one
  commit both pass the tag check, and the one-release check is check-then-act. GitHub keeps one waiting job
  per group, so a third cancels the second; a cancelled release job made nothing (re-run it).
- **DO NOT PUBLISH.** When the after-upload check fails, the job runs `gh release edit <tag> --title "DO NOT
  PUBLISH: <tag> moved during the upload"` (or "unchecked after the upload"), then fails. If the retitle fails
  too, it says so and still fails. gh's source (v2.102.1): `release edit` finds a draft by its pending tag
  name (GraphQL `release(tagName:)`), so it works with the tag gone, and `--title` sends only `name` (plus
  `tag_name`, unchanged). The PATCH's target is main's tip, so the workflows rule above doesn't bite.
- **"(on a tag)":** the build job's comment now says the pack is compiled from the same commit on a tag, and
  may come from the cache, from an earlier commit with the same kernel sources, elsewhere.

## Tests
`scripts/release/test_draft_guards.sh`: 18 step cases plus a wiring check (was 15). Each case also checks
whether the draft was retitled and to what, and stderr checks can be several, or negative (`!text`): a moved
tag must not say "re-run this job". The wiring check (PyYAML) needs the release job's concurrency group keyed
by `github.ref` with `cancel-in-progress: false`, `pack` in its `needs`, and `BUILT` = the pack job's commit.
18 mutations of release.yml (the stop rule's, #261's six and three of shady's) each turn it red.

Not tested: the concurrency group on GitHub, and `gh release edit` on a real draft. Either needs a pushed `v*`
tag or a real release, and this PR makes neither. The first is checked by actionlint and the wiring check, the
second against gh's source.
