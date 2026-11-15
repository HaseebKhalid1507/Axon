#!/usr/bin/env python3
"""The license texts of every crate linked into the Linux `axon` binary, for a
release tarball's THIRD_PARTY_LICENSES.txt (THIRD_PARTY_NOTICES.md: "a binary
distribution should include them").

Reads `cargo metadata` for x86_64-unknown-linux-gnu (so Windows-only crates
are out), walks the normal dependencies from the `axon` package (no dev or
build deps), and prints each crate's name, version, license expression and
the license/notice files its package ships (LICENSE*, COPYING*, NOTICE*,
UNLICENSE*). A crate without such a file is listed with its expression and
crates.io link. Then a paragraph on Rust's standard library, which every Rust
binary links statically.

Exit 1 if a crate's license expression isn't satisfied by the allowed list
(SPDX: `A OR B` needs one of them allowed, `A AND B` both; `A/B` is the old
spelling of `A OR B`; `X WITH exception` is one term), or if the list in
THIRD_PARTY_NOTICES.md ("Allowed licenses (scripts/release/rust_licenses.py):
...") isn't exactly ALLOWED. CI runs it on every pull request (ci.yml), so a
new dependency off the list fails there, not at tag time.

usage: scripts/release/rust_licenses.py [--manifest-path Cargo.toml] > THIRD_PARTY_LICENSES.txt
       scripts/release/rust_licenses.py --self-test
"""
import json, os, re, subprocess, sys

# Permissive only. Keep it equal to THIRD_PARTY_NOTICES.md's list (checked below).
ALLOWED = {"MIT", "Apache-2.0", "Apache-2.0 WITH LLVM-exception", "Unlicense", "Zlib", "BSL-1.0", "Unicode-3.0"}
NOTICE_LINE = re.compile(r"^Allowed licenses \(`scripts/release/rust_licenses\.py`\): (.+)\.$")

STD = """==============================================================================
The Rust standard library (std, core, alloc, and what it builds in: compiler_builtins, hashbrown, addr2line, gimli,
miniz_oxide, rustc-demangle, ...)  (MIT OR Apache-2.0)
==============================================================================
Linked statically into every Rust binary. Its copyright notices, including those of the crates it builds in:
https://github.com/rust-lang/rust/blob/master/COPYRIGHT, and the Rust toolchain's
share/doc/rust/COPYRIGHT-library.html (every release tarball ships it as RUST_STD_COPYRIGHT.html; the release
build fails without it). The MIT and Apache-2.0 texts are in this file above (for example under serde_json).
"""


def tokens(expr):
    return re.findall(r"\(|\)|[^\s()/]+|/", expr)


def allowed(expr, ok=ALLOWED):
    """Whether the SPDX license expression `expr` is satisfied by the ids in `ok`. Raises ValueError if it doesn't
    parse (a crate with an unparseable expression fails the check, as one off the list does)."""
    toks = tokens(expr)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def atom():
        t = take() if peek() is not None else None
        if t == "(":
            v = disj()
            if peek() != ")":
                raise ValueError("unbalanced parentheses")
            take()
            return v
        if t is None or t in (")", "/") or t.upper() in ("OR", "AND", "WITH"):
            raise ValueError(f"expected a license id, got {t!r}")
        if peek() is not None and peek().upper() == "WITH":
            take()
            if peek() is None:
                raise ValueError("WITH needs an exception")
            t = f"{t} WITH {take()}"
        return not t.endswith("+") and t in ok  # "GPL-2.0+": an or-later id is never on the list

    def conj():
        v = atom()
        while peek() is not None and peek().upper() == "AND":
            take()
            v = atom() and v
        return v

    def disj():
        v = conj()
        while peek() is not None and (peek().upper() == "OR" or peek() == "/"):
            take()
            v = conj() or v
        return v

    v = disj()
    if pos != len(toks):
        raise ValueError(f"trailing {toks[pos]!r}")
    return v


def self_test():
    cases = [("MIT", True), ("MIT OR Apache-2.0", True), ("Apache-2.0/MIT", True), ("MIT/GPL-3.0", True),
             ("Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT", True), ("Unlicense OR MIT", True),
             ("(MIT OR Apache-2.0) AND Unicode-3.0", True), ("MIT AND GPL-3.0-only", False),
             ("GPL-3.0-only AND MIT", False), ("GPL-3.0-only", False), ("GPL-3.0-only OR LGPL-2.1", False),
             ("MIT AND (GPL-2.0 OR Apache-2.0)", True), ("MIT AND (GPL-2.0 OR LGPL-2.1)", False),
             ("Apache-2.0 WITH LLVM-exception", True), ("GPL-2.0 WITH Classpath-exception-2.0", False),
             ("MIT WITH Foo-exception", False), ("GPL-2.0+", False), ("MIT+", False), ("mit", False)]
    for e, want in cases:
        assert allowed(e) is want, (e, want)
    for e in ["", "MIT OR", "(MIT", "MIT)", "AND MIT", "MIT WITH", "MIT Apache-2.0"]:
        try:
            allowed(e)
        except ValueError:
            continue
        raise AssertionError(f"{e!r} parsed")
    print(f"rust_licenses.py self-test: {len(cases)} expressions, 7 malformed: ok", file=sys.stderr)


def notice_list(root):
    path = os.path.join(root, "THIRD_PARTY_NOTICES.md")
    with open(path, encoding="utf-8") as f:
        hits = [m.group(1) for m in map(NOTICE_LINE.match, f.read().splitlines()) if m]
    if len(hits) != 1:
        sys.exit(f"{path}: want exactly one line 'Allowed licenses (`scripts/release/rust_licenses.py`): A, B.'")
    return {t.strip().strip("`") for t in hits[0].replace(" and ", ", ").split(",") if t.strip()}


def main():
    args = sys.argv[1:]
    if args == ["--self-test"]:
        return self_test()
    cmd = ["cargo", "metadata", "--format-version", "1", "--locked",
           "--filter-platform", "x86_64-unknown-linux-gnu"] + args
    meta = json.loads(subprocess.check_output(cmd))
    listed = notice_list(meta["workspace_root"])
    if listed != ALLOWED:
        sys.exit("THIRD_PARTY_NOTICES.md's allowed licenses and rust_licenses.py's ALLOWED differ: only in the "
                 f"notices {sorted(listed - ALLOWED)}, only in the script {sorted(ALLOWED - listed)}")
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = next(i for i, p in pkgs.items() if p["name"] == "axon")
    seen, todo = set(), [root]
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        for d in nodes[i]["deps"]:
            if any(k.get("kind") in (None, "normal") for k in d["dep_kinds"]):
                todo.append(d["pkg"])
    seen.discard(root)
    bad = []
    out = ["Third-party licenses of the Rust crates linked into axon (x86_64 Linux).",
           "Generated by scripts/release/rust_licenses.py from Cargo.lock; see also THIRD_PARTY_NOTICES.md.", ""]
    for p in sorted((pkgs[i] for i in seen), key=lambda p: (p["name"], p["version"])):
        lic = p.get("license") or "(none given)"
        try:
            ok = allowed(lic)
        except ValueError as e:
            ok, lic = False, f"{lic} [unparseable: {e}]"
        if not ok:
            bad.append(f"{p['name']} {p['version']}: {lic}")
        d = os.path.dirname(p["manifest_path"])
        files = sorted(f for f in os.listdir(d)
                       if re.match(r"(?i)^(licen[cs]e|copying|notice|unlicense)", f) and os.path.isfile(os.path.join(d, f)))
        out.append("=" * 78)
        out.append(f"{p['name']} {p['version']}  ({lic})  https://crates.io/crates/{p['name']}")
        out.append("=" * 78)
        if not files:
            out.append(f"(the package ships no license file; its license is {lic})")
        for f in files:
            out.append(f"--- {f}")
            with open(os.path.join(d, f), encoding="utf-8", errors="replace") as fh:
                out.append(fh.read().rstrip())
        out.append("")
    out.append(STD)
    sys.stdout.write("\n".join(out))
    if bad:
        print("licenses the allowed list doesn't satisfy (THIRD_PARTY_NOTICES.md, Rust dependencies):\n  "
              + "\n  ".join(bad), file=sys.stderr)
        sys.exit(1)
    print(f"{len(seen)} crates, every license allowed", file=sys.stderr)


if __name__ == "__main__":
    main()
