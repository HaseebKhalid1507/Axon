# 2026-11-12: release workflow and install.sh (dist track)

**What:** `.github/workflows/release.yml` builds the Linux x86-64 tarball on a tag `v*` and uploads it to a
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
