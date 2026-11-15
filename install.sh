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
#
# Everything is in main(), called on the last line: a download cut short runs nothing.
set -eu

repo="HaseebKhalid1507/Axon"
asset="axon-x86_64-linux.tar.gz"
source_url="https://github.com/$repo#install"

say() { printf 'axon install: %s\n' "$*" >&2; }
die() {
    say "$*"
    exit 1
}

# fetch URL OUT [CURL-FLAG]: download URL to OUT over HTTPS (redirects too); on failure, the HTTP status in $code.
fetch() {
    code=$(curl -fL "${3:--sS}" --proto "$proto" --proto-redir =https --retry 3 -w '%{http_code}' -o "$2" "$1") && return 0
    return 1
}

main() {
    bin_dir="${AXON_INSTALL_DIR:-$HOME/.local/bin}"
    doc_dir="${XDG_DATA_HOME:-$HOME/.local/share}/doc/axon"
    version="${AXON_VERSION:-latest}"
    case "$version" in
    latest | v*) ;;
    [0-9]*)
        say "AXON_VERSION=$version: release tags start with v; installing v$version"
        version="v$version"
        ;;
    *) die "AXON_VERSION=$version isn't a release tag (vX.Y.Z, or latest): https://github.com/$repo/releases" ;;
    esac
    proto="=https"
    if [ -n "${AXON_INSTALL_URL:-}" ]; then
        base="${AXON_INSTALL_URL%/}"
        proto="=https,file"
    elif [ "$version" = latest ]; then
        base="https://github.com/$repo/releases/latest/download"
    else
        base="https://github.com/$repo/releases/download/$version"
    fi

    case "$(uname -s)/$(uname -m)" in
    Linux/x86_64 | Linux/amd64) ;;
    *) die "no release binary for $(uname -s) $(uname -m) yet (Linux x86-64 only); build from source: $source_url" ;;
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

    # The download, the check and the unpacking happen in $tmp; the binary is first run from $bin_dir, where it
    # will live (a /tmp mounted noexec can't stop it). Whatever is left on any exit is removed.
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/axon-install.XXXXXX") || die "can't create a temporary directory in ${TMPDIR:-/tmp}"
    stage=""
    probe=""
    trap 'rm -rf "$tmp"; if [ -n "$stage$probe" ]; then rm -f "$stage" "$probe"; fi' EXIT
    trap 'exit 130' INT TERM

    say "downloading $base/$asset"
    code=""
    if ! fetch "$base/$asset.sha256" "$tmp/sum" -s; then
        if [ "$code" = 404 ] && [ "$version" = latest ]; then
            die "no published release yet (GitHub has no latest $asset); build from source: $source_url"
        elif [ "$code" = 404 ]; then
            die "no release $version with $asset: https://github.com/$repo/releases"
        fi
        case "$code" in "" | 000) code="no HTTP answer" ;; *) code="HTTP $code" ;; esac
        die "can't download $base/$asset.sha256 ($code)"
    fi
    if [ -t 2 ]; then progress=--progress-bar; else progress=-sS; fi
    fetch "$base/$asset" "$tmp/$asset" "$progress" || die "can't download $base/$asset"
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
    if [ -z "$src" ] || [ ! -f "$src" ]; then die "no axon binary in the tarball"; fi

    mkdir -p "$bin_dir" "$doc_dir" || die "can't create $bin_dir and $doc_dir"
    dest="$bin_dir/axon"
    if [ -f "$dest" ] && [ "$(sha "$dest")" = "$(sha "$src")" ]; then
        say "$dest is already $("$dest" --version 2>/dev/null || echo 'this release')"
    else
        # A temp name in the same directory, run there, then a rename: a running axon keeps its file, and a failed
        # copy or a binary that doesn't run leaves the old one.
        stage="$dest.tmp.$$"
        cp "$src" "$stage"
        chmod 755 "$stage"
        if ! new=$("$stage" --version 2>"$tmp/err"); then
            # Can anything run from $bin_dir? A one-line script there tells a noexec mount from a bad binary.
            probe="$bin_dir/.axon-probe.$$"
            printf '#!/bin/sh\nexit 0\n' >"$probe" && chmod 755 "$probe"
            if ! "$probe" 2>/dev/null; then
                rm -f "$probe"
                die "programs can't run from $bin_dir (mounted noexec?): set AXON_INSTALL_DIR to a directory where they can"
            fi
            rm -f "$probe"
            die "the downloaded axon doesn't run here: $(head -n 1 "$tmp/err")"
        fi
        old=""
        if [ -f "$dest" ]; then old=$("$dest" --version 2>/dev/null || echo "an axon that doesn't run"); fi
        mv -f "$stage" "$dest"
        stage=""
        say "installed $dest ($new)${old:+, replacing $old}"
    fi
    for f in LICENSE THIRD_PARTY_NOTICES.md THIRD_PARTY_LICENSES.txt RUST_STD_COPYRIGHT.html; do
        if [ -f "$(dirname "$src")/$f" ]; then cp "$(dirname "$src")/$f" "$doc_dir/$f"; fi
    done
    case ":$PATH:" in
    *":$bin_dir:"*)
        # Another axon earlier on PATH (cargo install's ~/.cargo/bin/axon, say) would run instead of this one.
        first=$(command -v axon 2>/dev/null || true)
        if [ -n "$first" ] && [ -f "$first" ] && [ "$(sha "$first")" != "$(sha "$dest")" ]; then
            say "warning: $first comes first on your PATH ($("$first" --version 2>/dev/null || echo 'another axon')), so 'axon' runs it, not $dest: remove it, or put $bin_dir earlier in PATH"
        fi
        ;;
    *) say "$bin_dir isn't on your PATH: add  export PATH=\"$bin_dir:\$PATH\"  to your shell's profile" ;;
    esac
    say "next: axon pull qwen2.5-0.5b, then axon chat qwen2.5-0.5b (add --device cuda for an NVIDIA GPU)"
}

main "$@"
