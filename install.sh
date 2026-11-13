#!/bin/sh
# Axon installer.
#
#   curl -fsSL https://raw.githubusercontent.com/HaseebKhalid1507/Axon/main/install.sh | sh
#
# Puts `axon` in ~/.local/bin from the latest GitHub release, after checking the tarball's SHA-256 against the
# release's .sha256; the license notices go to ~/.local/share/doc/axon. Linux x86-64; a release binary needs only the
# NVIDIA driver for --device cuda (its GPU kernels are built in) and nothing for the CPU. Run it again any time:
# with nothing new it changes nothing.
#
# Environment: AXON_VERSION (a release tag, e.g. v0.1.0; default the latest), AXON_INSTALL_DIR (default
# ~/.local/bin), AXON_INSTALL_URL (a directory URL holding the tarball and its .sha256 instead of GitHub's: tests).
set -eu

repo="HaseebKhalid1507/Axon"
asset="axon-x86_64-linux.tar.gz"
bin_dir="${AXON_INSTALL_DIR:-$HOME/.local/bin}"
doc_dir="${XDG_DATA_HOME:-$HOME/.local/share}/doc/axon"
version="${AXON_VERSION:-latest}"
proto="=https"
if [ -n "${AXON_INSTALL_URL:-}" ]; then
    base="${AXON_INSTALL_URL%/}"
    proto="=https,file"
elif [ "$version" = latest ]; then
    base="https://github.com/$repo/releases/latest/download"
else
    base="https://github.com/$repo/releases/download/$version"
fi

say() { printf 'axon install: %s\n' "$*" >&2; }
die() {
    say "$*"
    exit 1
}

case "$(uname -s)/$(uname -m)" in
Linux/x86_64 | Linux/amd64) ;;
*) die "no release binary for $(uname -s) $(uname -m) yet (Linux x86-64 only); build from source: https://github.com/$repo#quick-start" ;;
esac
command -v curl >/dev/null 2>&1 || die "needs curl"
command -v tar >/dev/null 2>&1 || die "needs tar"
if command -v sha256sum >/dev/null 2>&1; then
    sha() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    die "needs sha256sum or shasum"
fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/axon-install.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
trap 'exit 130' INT TERM

fetch() { curl -fsSL --proto "$proto" --proto-redir =https --retry 3 -o "$2" "$1"; }
say "downloading $base/$asset"
fetch "$base/$asset.sha256" "$tmp/sum" || die "can't download $base/$asset.sha256"
fetch "$base/$asset" "$tmp/$asset" || die "can't download $base/$asset"
want=$(cut -d' ' -f1 <"$tmp/sum")
case "$want" in
*[!0-9a-f]* | "") die "the .sha256 file isn't a SHA-256" ;;
esac
[ "${#want}" -eq 64 ] || die "the .sha256 file isn't a SHA-256"
got=$(sha "$tmp/$asset")
[ "$got" = "$want" ] || die "checksum mismatch: the tarball's SHA-256 is $got, the release says $want; nothing installed"

mkdir "$tmp/x"
tar -xzf "$tmp/$asset" -C "$tmp/x"
src=$(find "$tmp/x" -mindepth 2 -maxdepth 2 -type f -name axon | head -n 1)
[ -n "$src" ] && [ -x "$src" ] || die "no axon binary in the tarball"
new=$("$src" --version 2>/dev/null) || die "the downloaded axon doesn't run here: $("$src" --version 2>&1 | head -n 1)"

mkdir -p "$bin_dir" "$doc_dir" || die "can't create $bin_dir and $doc_dir"
dest="$bin_dir/axon"
if [ -f "$dest" ] && [ "$(sha "$dest")" = "$(sha "$src")" ]; then
    say "$dest is already $new"
else
    # A temp name in the same directory, then a rename: a running axon keeps its file, and a failed copy leaves the
    # old one.
    cp "$src" "$dest.tmp.$$"
    chmod 755 "$dest.tmp.$$"
    mv -f "$dest.tmp.$$" "$dest"
    say "installed $dest ($new)"
fi
for f in LICENSE THIRD_PARTY_NOTICES.md THIRD_PARTY_LICENSES.txt; do
    if [ -f "$(dirname "$src")/$f" ]; then cp "$(dirname "$src")/$f" "$doc_dir/$f"; fi
done
case ":$PATH:" in
*":$bin_dir:"*) ;;
*) say "$bin_dir isn't on your PATH: add  export PATH=\"$bin_dir:\$PATH\"  to your shell's profile" ;;
esac
say "next: axon pull qwen2.5-0.5b, then axon chat qwen2.5-0.5b (add --device cuda for an NVIDIA GPU)"
