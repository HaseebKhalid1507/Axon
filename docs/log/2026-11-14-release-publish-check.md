# Release workflow: the tag check right before Publish; publish.sh, tied to the checked tarball; a draft title that fails closed (#265)

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
- **DO NOT PUBLISH** (round 1; round 2 below replaced it). When the after-upload check fails, the job runs `gh release edit <tag> --title "DO NOT
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
18 mutations of release.yml each turn it red: the stop rule's seven (07–13), two more of mine (14: no SHA-shape
check; 15: an API error taken for a moved tag), #261's six (01–06) and three of shady's from the #261 review
(16–18). The first version of this line said "the stop rule's, #261's six and three of shady's", which adds
up to 16 (shady's #265 nit 4).

Not tested: the concurrency group on GitHub, and `gh release edit` on a real draft. Either needs a pushed `v*`
tag or a real release, and this PR makes neither. The first is checked by actionlint and the wiring check, the
second against gh's source.

## Round 2: shady's #265 review (9/10 SHIP, comment 6880348449)
Stop rule: PR comment 6880396712, written before any edit.

- **SHOULD 1, the harness pinned what the step decides, not what it asks GitHub for.** Six mutants of shady's
  stayed 19/19 green: drop `--draft` (the tag push would publish at once, skipping every GPU hand check), drop
  `--verify-tag`, another title, drop `--repo` from the create or the edit (the release job has no checkout, so
  every retitle would fail on the runner), `gh release edit "$BUILT"`. The stub now appends each call's
  arguments, and every case checks them:
  - every create: the tag, then exactly `--repo`, `--draft`, `--verify-tag`, `--title` and `--notes`, then
    exactly the three files;
  - every edit: exactly `release edit <tag> --repo <repo> --title "Axon <tag>"`.

  A second call shows up as a mismatch.
- **Nit 1, the title failed open.** The draft started as `Axon vX.Y.Z` and was demoted only when a failure was
  detected, so a cancel, a lost runner or a failed edit left a draft titled the way the publish rule trusts.
  Now the create says "DO NOT PUBLISH: Axon vX.Y.Z (unchecked)", and the job retitles it `Axon vX.Y.Z` only
  after the tag check after the upload passes. The failure branch makes no API call, and "couldn't retitle
  either" is gone. If the final retitle fails, the draft is right but still says DO NOT PUBLISH, and the job
  fails, saying how to retitle it.
- **SHOULD 2, the last check sat 35 lines above the Publish click.** `scripts/release/publish.sh vX.Y.Z` is now
  releasing.md's numbered step 5. It checks the listing: exactly one release for the tag, a draft, titled
  exactly `Axon vX.Y.Z`, one commit in its notes, exactly the three assets, all uploaded. Then the tag, last.
  Then it runs `gh release edit vX.Y.Z --tag vX.Y.Z --verify-tag --draft=false --prerelease=false`. Shady's trap,
  in the script and the doc: in gh 2.102.1, `--verify-tag` checks only with `--tag` (`if opts.VerifyTag &&
  opts.TagName != ""`). `--check` runs every check and publishes nothing; releasing.md runs it first. It uses
  the listing, not `gh release view`, because the listing shows every release for the tag, and GitHub allows
  several drafts for one.
- **Nits 2–5.** After deleting a moved tag's draft: the run for the new commit makes the draft; re-run its
  release job if it already refused; the checks start over. The no-commit-after-upload message is checked not
  to say "re-run this job" (shady's mutant, "Or re-run this job", is red now). The mutant count above. A tag
  pushed again at the same commit gives one expected red release job (releasing.md step 6).

### Tests, round 2
`test_draft_guards.sh`: 18 step cases, the wiring check and 21 publish.sh cases (40). `--live` adds the 5
endpoint cases and 2 read-only publish.sh `--check` runs against real listings: no release for `v9.9.9` here,
and rhysd/actionlint's published v1.7.12. Both refuse before the tag check and the edit. 40 mutants, all red:
- shady's 7 survivors, on the new step;
- 6 for the fail-closed title: created titled `Axon`, no final retitle, the retitle before the check, a retitle
  in the failure branch, the retitle's failure swallowed, the final edit with `--draft=false`;
- 12 in publish.sh: no commit compare, no title compare, no `--tag`, no `--verify-tag`, no exactly-one check,
  no draft check, no asset check, `--check` that publishes, `--prerelease`, no SHA-shape check, the API error
  ignored, `--check` skipping the tag check;
- the 18 above where their code still exists. 07–09 aimed at the failure-branch retitle, which is gone; n2, n4
  and n5 cover it.

Not tested: `gh release edit --draft=false` on a real draft (publish.sh's edit never runs live), the
concurrency group on GitHub, a cancelled job leaving its "unchecked" draft (only the title it is created with
is tested).

## Round 3: shady's round-2 review (comment 6880776856, 9/10 SHIP)

Stop rule: PR comment 6880937124 (11:47 EST), written before any edit.

- **SHOULD 1: publish.sh proved the draft was current, not that it was the one checked.** The hole: checks
  start on A's tarball, the tag moves to B, the old draft is deleted as told, B's run makes a new draft, and
  publish.sh, run when A's checks finish, passes everything and publishes B's tarball. Now it is `publish.sh
  vX.Y.Z SHA256` (required to publish; an empty or malformed one is refused before any gh call). It refuses
  unless SHA256 equals the notes' "SHA-256 of the tarball" (exactly one, 64 hex) and the tarball asset's
  `digest`, which GitHub computes at upload ("sha256:…"; null or malformed refused; checked on real listings:
  the listing endpoint carries it). The notes already had the line; the harness now pins it and "Built from
  commit" exactly once each in every create. `--check vX.Y.Z [SHA256]` prints the SHA-256 to write down.
- **SHOULD 2: the 1–2 s between the last tag check and the publish.** gh's own lookups and `remoteTagExists` sit
  between them, and GitHub has no "publish only if the tag names X". Now, after the edit, publish.sh asks for
  the tag's commit again and re-reads the release by its id (published, its tag, its title). On a miss, or an
  answer it can't read, it PATCHes the release back to a draft titled "DO NOT PUBLISH: Axon vX.Y.Z
  (unpublished by publish.sh)" and fails, saying what to do; if that PATCH fails, it says to run it by hand,
  with the command. A failed edit that left a draft is "nothing published", with no revert. releasing.md
  documents the residual window and that a tag ruleset on `refs/tags/v*` (blocking updates and deletions)
  closes it and the hours-long one; the maintainer's call, not set up.
- **Nit 1: the final title in the publish itself.** The job now titles the checked draft "Axon vX.Y.Z (checked;
  publish with scripts/release/publish.sh)"; publish.sh refuses any other title, including `Axon vX.Y.Z`, and
  sends `--title "Axon vX.Y.Z"` with `--draft=false` in one edit. The web page's Publish button would publish a
  visibly mis-titled release.
- **Nit 2:** releasing.md: keep the "SHA-256 of the tarball" and "Built from commit" lines, each exactly once,
  and the title, when putting the release notes into the draft; publish.sh's refusals say so too.
- **Nit 3:** 0 releases: publish.sh asks for `.permissions.push` and says when the account lacks push access
  (drafts hidden), or names both causes when it can't tell. jq is checked up front ("needs jq").

### Tests, round 3
`test_draft_guards.sh`: 18 step cases, the wiring check and 48 publish.sh cases (67). The stub answers the
re-read as the recorded edit left the release, the way GitHub would, and records the revert's PATCH. `--live`
adds the 5 endpoint cases and 3 read-only `--check` runs: no release for `v9.9.9` here (push access: "none
made"), none on rhysd/actionlint (no push access: says so), and its published v1.7.12. 62 mutants, all red, the
runner checking it ran every one it generated:
- 29 for round 3: 21 in publish.sh (no SHA256 compare, no digest compare, a null digest accepted, SHA256
  optional, an empty one accepted, the first of two notes SHAs taken, no tag check after the publish, a miss
  that doesn't revert, the revert's failure ignored, the re-read's draft field ignored, the publish without
  `--title`, `Axon vX.Y.Z` accepted, the push-access hint dropped, the jq check dropped, a still-draft release
  reported published, no lowercasing, the re-read's title or tag unchecked, the revert keeping the title, and
  ports of round 2's p2 and p9); 8 in release.yml (the final title back to `Axon vX.Y.Z`, the notes without or
  with two tarball SHAs, and ports of round 2's s3, n2, n3, n5, n6);
- round 2's 40 where their anchor still exists: 18; the 7 whose anchor this round rewrote are ported above;
- the original 18 where their code still exists: 15 (07–09 aimed at code gone since round 2).

Not tested: a live publish or revert (publish.sh's edit and PATCH never run against GitHub); that a draft's
tarball carries a `digest` (GitHub's changelog says digests are computed at upload for all assets; checked
only on published releases); the concurrency group on GitHub; a cancelled job leaving its "unchecked" draft.

