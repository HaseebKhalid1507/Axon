# Release workflow: only the tag's push makes a draft (#257, follows #230)

- Date: 2026-11-14. Branch: fix/release-dispatch-gate. Base: main 6273ac5. Status: open
- Pick: shady's SHOULD on #230 (review 9/10 SHIP at 439f3b6, comment 6872717480), plus his one-line nits.

## What changed
Workflow, docs and one string; no Rust.
- **`release` job:** `if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')`. A
  `workflow_dispatch` on the tag (releasing.md step 4) has `github.ref` = `refs/tags/vX.Y.Z` too, so the ref-only
  condition ran the job again and `gh release create --draft` added a second draft for the same tag (GitHub
  allows several), holding the dispatch run's tarball: the very build whose sameness step 4 is checking.
- **One release per tag:** before `gh release create`, the job lists the repo's releases (drafts included: the
  job's token can push) and exits 1, uploading nothing, if one has the tag. That covers what the event gate
  can't: "Re-run all jobs" on the tag's own run is still a `push`.
- **Retention:** the `kernels-pack` artifact is kept 30 days (was 7). releasing.md step 1 needs it, plus about an
  hour on bella.
- **Nits:** CHANGELOG says "built to be reproducible" (no two builds of one commit have been compared yet);
  releasing.md says the dispatch only builds, how to fetch its tarball, and to publish a full release (install.sh
  follows `latest`, which skips pre-releases); THIRD_PARTY_LICENSES.txt's std paragraph no longer says the tarball
  ships `RUST_STD_COPYRIGHT.html` "when the toolchain has it" (the build fails without it).

## How it was checked
No GPU gates: nothing here runs on a GPU or changes a binary.
- actionlint 1.7.12 (with shellcheck): the same two `SC2012` infos as main 6273ac5, nothing new.
- The guard's `run:` block, taken from the YAML, run under `bash -eo pipefail` with a stub `gh`: the real API for
  `v0.1.0` (no releases exist) proceeds to create; a fixture with a draft for the tag on the second page refuses
  (rc 1, nothing created); a tag containing `"` neither matches its prefix nor breaks jq (`--arg`, no string
  splicing); a failing `gh api` fails the step (rc 5).
- The PR's own Release run (the workflow file changed, so it runs) builds and tests the tarball, and its
  `draft release` job is skipped. CI is green.
- Not exercised: a real tag push or tag dispatch. No tag may be pushed before v0.1.0. The condition's two
  inputs are GitHub's documented `github.event_name` (`push`, `workflow_dispatch`) and `github.ref`.
