# 2026-11-12: release workflow and install.sh (dist track)

**What:** `.github/workflows/release.yml` builds the Linux x86-64 tarball on a tag `v[0-9]*` and uploads it to a
*draft* release (a maintainer publishes it); a pull request that touches the release files or `workflow_dispatch`
builds and tests it without a release. Job 1 compiles the kernel pack with NVRTC 13.4.92 from NVIDIA's apt repo on
a GPU-less runner (cached by the kernel sources' hash); job 2 builds `axon` with the pack embedded in
`manylinux_2_28` (glibc 2.28: RHEL 8+, Debian 10+, Ubuntu 20.04+), checks it carries every module it loads,
packages `axon`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `README.md` and `THIRD_PARTY_LICENSES.txt` (every linked
crate's license files, `scripts/release/rust_licenses.py`), and runs `install.sh` against it (twice: the second
changes nothing; then a tampered tarball must be refused).

**install.sh:** POSIX sh, `curl | sh`; Linux x86-64 only; downloads the tarball and its `.sha256`, refuses a
mismatch before extracting anything into place, installs `~/.local/bin/axon` by rename (a running axon keeps its
file), idempotent (same bytes: nothing replaced).

**Why build the pack in CI:** NVRTC needs no GPU, and the same NVRTC version gives the same bytes; the CI pack is
checked against bella's NVRTC entry by entry (`gpu_release_pack_is_nvrtcs`) and by the md5 gate: results in the PR.

**README:** the install section leads with `install.sh` (driver R580+, the one-line refusal on older drivers and
`--device cpu`), then the source build; the quick start uses `axon pull` and model names.

**Review round (shady, 7/10 SHIP-AFTER-FIXES, 11-13):**
- **MUSTs.**
  - No release exists yet, so the README says "until v0.1.0 is published, build from source". On a 404 from
    `latest`, install.sh says "no published release yet; build from source". That message was checked against
    the real GitHub.
  - Every workflow step now runs under `bash -eo pipefail` (`defaults.run.shell: bash`). The smoke step fails
    when `axon gen` exits non-zero or generates nothing. `--device cuda` without a GPU must exit with code 2. The
    second install run's exit status counts.
- **Supply chain.**
  - Actions are pinned to commit SHAs and `manylinux_2_28` to `2026.11.13-1@sha256:15b2f702…` (the image of the
    green run 43502899487).
  - A tag build restores no cache, neither the pack nor rust-cache.
  - Timeouts: 90/60/10 min.
- **Tags.** The trigger is `v[0-9]*`, and the build fails unless `axon --version` is the tag's version.
- **Reproducibility.** The tarball is built with `tar --sort=name --mtime=@<commit time> --owner=0 --group=0
  --numeric-owner` and `gzip -n`, with fixed modes. The release build gets `--remap-path-prefix` for
  `~/.cargo` and the checkout.
- **Licenses.**
  - `rust_licenses.py` parses SPDX: OR needs one allowed term, AND needs all, `/` is OR, and an unparseable
    expression fails. `--self-test` covers 19 expressions and 7 malformed ones.
  - Its allowed list must equal THIRD_PARTY_NOTICES.md's "Allowed licenses" line, which the script reads.
  - The output ends with the Rust standard library's paragraph, and the tarball ships the toolchain's
    `COPYRIGHT-library.html` as `RUST_STD_COPYRIGHT.html`.
  - ci.yml's `check` runs the script on every PR, and `src/main.rs` joins the release workflow's paths.
- **install.sh.**
  - The whole body is in `main()`.
  - The binary is staged and first run in the install dir, so a noexec `/tmp` doesn't matter. A noexec install
    dir is named as such, found with a one-line probe script.
  - It warns when another `axon` comes first on PATH, and says which version it replaced.
  - It shows `--progress-bar` on a terminal, cleans up the staged file on any exit, and turns `AXON_VERSION=0.1.0`
    into `v0.1.0`.
  - Tested under busybox ash (noexec `/tmp` and install dir as tmpfs mounts) and dash (Debian): first install,
    second, upgrade, tampered, broken binary, missing release, shadowed, `AXON_VERSION` cases.
- **docs/releasing.md.** The publishing checklist, packcmp against bella, the md5 gate and the clean machine on
  the draft's own tarball, then one real `curl | sh` after publishing. `packcmp.py` moved to
  `scripts/release/`. A CHANGELOG entry.
- **Left for later.** Build provenance attestations, and a check that the tag's commit is on main.
