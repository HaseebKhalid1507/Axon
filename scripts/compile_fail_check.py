#!/usr/bin/env python3
"""Check that every `compile_fail` doc test fails for the reason it names.

Stable rustdoc runs a ```compile_fail,E0599 block and passes it on *any*
compile error: the code after the comma isn't checked. So a rename that
breaks a snippet for another reason (a typo, a moved item) keeps it green.
This script finds every compile_fail block in src/, requires an error code
on each, compiles it on its own against the library (metadata only) and
checks that the first error has that code. CI runs it after `cargo build
--lib` (.github/workflows/ci.yml); locally:

    scripts/compile_fail_check.py target/debug/libaxon.rlib target/debug/deps

usage: compile_fail_check.py LIBAXON(.rlib|.rmeta) [DEPDIR...]
Env: RUSTC (default rustc). Exit 1 on any mismatch or missing code.
"""
import os, re, subprocess, sys, tempfile

def blocks(root):
    """(file, line, code, source) of each compile_fail doc block under root."""
    for dp, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if not f.endswith(".rs"):
                continue
            path = os.path.join(dp, f)
            lines = open(path, encoding="utf-8").read().split("\n")
            i = 0
            while i < len(lines):
                m = re.match(r"^\s*//[/!] ?```compile_fail(.*)$", lines[i])
                if not m:
                    i += 1
                    continue
                attrs = [a.strip() for a in m.group(1).split(",") if a.strip()]
                code = next((a for a in attrs if re.fullmatch(r"E\d{4}", a)), None)
                body, j = [], i + 1
                while not re.match(r"^\s*//[/!] ?```\s*$", lines[j]):
                    t = re.sub(r"^\s*//[/!] ?", "", lines[j])
                    # rustdoc's hidden lines: "# x" is compiled as "x".
                    body.append(t[2:] if t.startswith("# ") else ("" if t == "#" else t))
                    j += 1
                yield path, i + 1, code, "\n".join(body)
                i = j + 1

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    lib, deps = sys.argv[1], sys.argv[2:]
    rustc = os.environ.get("RUSTC", "rustc")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    bad = n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path, line, code, src in blocks(root):
            n += 1
            where = f"{os.path.relpath(path, os.path.join(root, '..'))}:{line}"
            if code is None:
                print(f"FAIL {where}: compile_fail without an error code (```compile_fail,E0599)")
                bad += 1
                continue
            if "fn main" not in src:
                src = "#![allow(unused)]\nfn main() {\n" + src + "\n}\n"
            f = os.path.join(tmp, f"snip{n}.rs")
            open(f, "w", encoding="utf-8").write(src)
            cmd = [rustc, "--edition", "2024", "--crate-type", "bin", "--emit=metadata",
                   "--out-dir", tmp, "--crate-name", f"snip{n}", f, "--extern", f"axon={lib}"]
            for d in [os.path.dirname(os.path.abspath(lib))] + deps:
                cmd += ["-L", f"dependency={d}"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            first = next((l for l in r.stderr.split("\n") if l.startswith("error")), "")
            got = re.match(r"error\[(E\d{4})\]", first)
            if r.returncode != 0 and got and got.group(1) == code:
                print(f"ok   {where}: {code}  {first}")
            else:
                bad += 1
                print(f"FAIL {where}: wants {code}, got rc={r.returncode} {first or '(compiles)'}")
                if r.returncode != 0 and not got:
                    print(r.stderr)
    print(f"{n - bad} of {n} compile_fail blocks fail as annotated")
    sys.exit(1 if bad or n == 0 else 0)

if __name__ == "__main__":
    main()
