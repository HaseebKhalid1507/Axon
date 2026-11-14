# Releasing

How a release gets from a tag to `curl … | sh`, and the checks a person runs before publishing, because the tag
build can't run them itself.

## What the tag build does

Pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml`:

1. **Kernel pack.** NVRTC 13.4.92, from NVIDIA's apt repo on a runner with no GPU, compiles the kernel pack
   for `sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_120` plus `compute_120` PTX. A tag build never restores it
   from a cache: it is compiled from the tagged source.
2. **Binary.** `axon` is built with the pack embedded, in `manylinux_2_28` (glibc 2.28), pinned by digest.
   The build fails if:
   - the binary's version isn't the tag's;
   - the pack lacks a module this build loads;
   - the binary needs a glibc newer than 2.28;
   - a linked crate's license is off the allowed list (THIRD_PARTY_NOTICES.md).
3. **Tarball.** `axon`, `LICENSE`, `README.md`, `THIRD_PARTY_NOTICES.md`, `THIRD_PARTY_LICENSES.txt` and
   `RUST_STD_COPYRIGHT.html` are packed to be reproducible: fixed order, owners, modes and the commit's time.
4. **install.sh.** It installs the tarball twice; the second run must change nothing. A tampered tarball must
   be refused.
5. **No-GPU smoke test.** `axon pull qwen2.5-0.5b`, then `axon gen` on the CPU. `--device cuda` must be
   refused with exit code 2.
6. **Draft release.** The workflow uploads the tarball, its `.sha256` and `install.sh` to a **draft** release
   for the tag. Nothing is public until a maintainer publishes the draft. Only the tag's push uploads: a
   `workflow_dispatch` run, on the tag or not, only builds. The job uploads nothing if:
   - the tag no longer names the commit the run built: it was moved while the run was building, and the run
     for the commit it names makes the draft;
   - GitHub gave no commit for the tag: it was deleted, or the API failed. If the tag still names the run's
     commit (a transient error), re-run the release job ("Re-run failed jobs");
   - a release for the tag (draft or published) exists already, so "Re-run all jobs" on the tag's run can't
     add a second draft.

   One release job per tag runs at a time (a `concurrency` group). A waiting one isn't cancelled, but a third
   cancels the second, which then made nothing: re-run it if its run built the commit the tag names. A tag
   deleted and pushed again at the same commit gives one expected red run: the second release job waits for
   the first, finds its draft and refuses. Check the first run's draft.

   The draft is created titled "DO NOT PUBLISH: Axon vX.Y.Z (unchecked)". After the upload the job checks the
   tag again, and only if it still names the run's commit does the job retitle the draft `Axon vX.Y.Z`. A job
   that fails, is cancelled or dies after the create leaves a draft whose title says not to publish it, and
   `publish.sh` refuses any other title. If the check fails (the tag moved, or GitHub gave no commit), the
   job fails and its log says whether to delete the draft or, after a transient error with the tag unmoved, to
   retitle it. The draft's notes name the commit it was built from. If the upload fails partway, the half-made
   draft stays (`gh release create --draft` doesn't remove it): delete it (`gh release delete vX.Y.Z --yes`;
   the tag stays), then "Re-run failed jobs".

The workflow never checks the GPU side. No runner has a GPU, so the draft's kernels have run nowhere yet.

## Before the tag

- `Cargo.toml`'s `version` is the tag's version without the `v`. The build refuses a mismatch.
- CHANGELOG: the Unreleased entries move under `## X.Y.Z (date)`. Release notes: `docs/release-0.1.md` for 0.1.
- main is green (CI and the post-merge suites) at the commit you tag.

## Checking and publishing the draft

Run every check on **the draft's own tarball**: `gh release download vX.Y.Z -p 'axon-x86_64-linux.tar.gz*'`,
then `sha256sum -c axon-x86_64-linux.tar.gz.sha256`. Write its SHA-256 next to each result.

**First, `scripts/release/publish.sh --check vX.Y.Z`**, so you don't spend hours on a stale draft. It runs the
checks of the Publish step (5, below) and publishes nothing. If it refuses, it says why:
- the tag names another commit: the tag moved, and the draft holds the old commit's tarball. Delete the draft
  (`gh release delete vX.Y.Z --yes`; the tag stays). The run that built the commit the tag names now makes the
  draft. If its release job already refused, because this draft existed, re-run it ("Re-run failed jobs").
  The checks below start over, on the new draft's tarball.
- GitHub gave no commit (`No commit found for SHA: refs/tags/vX.Y.Z (HTTP 422)`): the tag is gone. **Don't
  publish.** The draft's target is main, so publishing would create `vX.Y.Z` at main's tip, holding the old
  commit's tarball. Delete the draft (`gh release delete vX.Y.Z --yes` finds a draft by its tag name even with
  the tag gone).
- the title isn't `Axon vX.Y.Z` (it starts "DO NOT PUBLISH"): the release job didn't confirm the tag after the
  upload, or didn't finish. Its log says what to do.

1. **The pack is NVRTC's output, byte for byte.** Download the tag run's `kernels-pack` artifact (kept 30
   days): the draft's binary embeds exactly this pack, because the build job takes it from the same run. On
   bella, build the reference with the same NVRTC (13.4.92): `axon kernels build ref.pack --targets <the
   workflow's PACK_TARGETS> -j N`. That takes about an hour for all eight targets. Then run
   `scripts/release/packcmp.py kernels.pack ref.pack`. It must say PASS, every entry the same raw bytes.
2. **md5 gate.** On bella (sm_120) and on an RTX 4090 (sm_89), run the md5 gate (docs/testing.md) with the
   draft's `axon` as the candidate under `AXON_KERNEL_PACK=only`, against main's NVRTC-compiled run of the same
   commit. 0 of 42 may differ.
3. **Clean machine.** Use a GPU with only the driver: no CUDA toolkit, and `ldconfig -p | grep -i nvrtc` and
   `find / -name 'libnvrtc*'` both empty. As a fresh user:
   - `AXON_INSTALL_URL=file://<dir with the tarball and .sha256> sh install.sh`;
   - `axon kernels list` exits 0 and says "precompiled sm_XX";
   - `axon pull qwen2.5-0.5b`;
   - the md5 gens with `--device cuda` match the base's on the same GPU.

   The procedure as run for #227 is in that PR's "Clean-machine proof".
4. **Optional, reproducibility.** Run the workflow by hand on the tag: `gh workflow run release.yml --ref vX.Y.Z`
   (or Actions → Release → Run workflow, from the tag). It only builds; it can't touch the draft. Its tarball
   (`gh run download RUN -n axon-x86_64-linux`) should have the draft's SHA-256. Not "Re-run all jobs" on the
   tag's own run: its release job would refuse, because the draft exists.

If any check fails, delete the draft and the tag, fix the problem on main, and tag again.

5. **Publish: `scripts/release/publish.sh vX.Y.Z`.** Not the web page's Publish button, and not a bare
   `gh release edit --draft=false`: neither checks anything. Right before it publishes, the script checks,
   and refuses on any miss (the list above says what to do):
   - there is exactly one release for the tag, a draft, titled exactly `Axon vX.Y.Z`;
   - its notes name one commit, and its assets are exactly the tarball, its `.sha256` and `install.sh`, all
     uploaded;
   - the tag names that commit now. This runs last, so a tag moved during the hours of checks above is caught.

   Then it runs `gh release edit vX.Y.Z --tag vX.Y.Z --verify-tag --draft=false --prerelease=false`. That makes a
   full release, not a pre-release, because install.sh installs GitHub's `latest`, which never points at a
   pre-release. If you ever run that line by hand, keep `--tag`: in gh 2.102.1, `--verify-tag` checks the tag
   only when `--tag` is given too.

## After publishing

- On a machine without axon, run the real thing: `curl -fsSL
  https://raw.githubusercontent.com/HaseebKhalid1507/Axon/main/install.sh | sh`. It must install vX.Y.Z, and
  `axon --version` must say so. Until then, install.sh's GitHub path has only run as far as the 404 of a
  missing release. The `latest` redirect to the asset, with `--proto-redir`, only runs once a release is
  published.
- First release only: delete the "Until v0.1.0 is published, build from source" lines from README.md and
  docs/usage.md.
