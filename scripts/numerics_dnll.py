#!/usr/bin/env python3
"""Teacher-forced ΔNLL between two axon runs: the numerics gate's script hook (docs/design/determinism-modes.md §4.2,
methodology.md "Numerics-changing work"). By default one binary in exact vs fast mode; or two binaries (a kernel's
parent vs the candidate) with their own flags.

    numerics_dnll.py --model M --base BIN [--cand BIN] [--base-flags '...'] [--cand-flags '...']
                     [--texts FILE] [--windows K] [--chars C] [--expect-zero] [--json OUT]

Each of K windows of C characters, evenly spaced over the corpus (default tests/fixtures/natural_text.txt), is one
`axon logits` run per side, with AXON_LOGITS_ALL (every position) and AXON_LOGITS_NLL=1: the -ln p of each position's
next prompt token (teacher forcing: both sides read the same text, tokenized by the binary itself; the two sides must
agree on the ids or the run stops). Per position: d = nll(cand) - nll(base). Reported: mean d (nats per position) with
its standard error clustered by window (Liang-Zeger, CR1), each side's mean NLL, and top-1 agreement.

Defaults: --cand = --base, --base-flags '--numerics exact', --cand-flags '--numerics fast'. Extra axon flags
(e.g. '--device cuda --kv q8') go in both. Exit 1 with --expect-zero if any position's NLL or top-1 differs (the
sanity check while no fast kernel exists: fast runs the exact kernels), or with --max-dnll X if |mean d| > X.
"""
import argparse, json, math, os, re, shlex, subprocess, sys


def windows(text, k, chars):
    """k windows of `chars` characters, evenly spaced, cut at whitespace."""
    if len(text) <= chars:
        return [text]
    step = (len(text) - chars) / max(k - 1, 1)
    out = []
    for i in range(k):
        a = int(i * step)
        w = text[a:a + chars]
        # Start and end at whitespace, so no window opens mid-word.
        s = w.find(" ")
        e = w.rfind(" ")
        if 0 <= s < e:
            w = w[s + 1:e]
        out.append(w)
    return out


def run(binary, model, prompt, flags):
    env = dict(os.environ, AXON_LOGITS_ALL="1000000", AXON_LOGITS_NLL="1", AXON_LOGITS_DIGITS="6")
    r = subprocess.run([binary, "logits", model, prompt, "-k", "1"] + flags, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"{binary} logits exited {r.returncode}: {r.stderr[-2000:]}")
    ids, nll, top1, pos = None, {}, {}, None
    for line in r.stdout.splitlines():
        m = re.match(r"prompt ids \((\d+)\): \[(.*)\]", line)
        if m:
            ids = [int(x) for x in m.group(2).split(",") if x.strip()]
            continue
        m = re.match(r"position (\d+):", line)
        if m:
            pos = int(m.group(1))
            continue
        m = re.match(r"nll (\d+) (\d+) (\S+)", line)
        if m:
            nll[int(m.group(1))] = float(m.group(3))
            continue
        m = re.match(r"\s*(\d+)\s+\S+\s+\S+\s+\"", line)
        if m and pos is not None and pos not in top1:
            top1[pos] = int(m.group(1))
    if ids is None or not nll:
        sys.exit(f"{binary}: no positions in the output (AXON_LOGITS_NLL unsupported? an older binary)")
    return ids, nll, top1, r.stderr


def clustered(d):
    """(mean, SE clustered by key) of [(cluster, x)]."""
    n = len(d)
    if n == 0:
        return 0.0, 0.0
    mean = sum(x for _, x in d) / n
    sums = {}
    for c, x in d:
        sums[c] = sums.get(c, 0.0) + (x - mean)
    k = len(sums)
    if k < 2:
        return mean, float("nan")
    v = sum(s * s for s in sums.values()) / (n * n) * k / (k - 1)
    return mean, math.sqrt(v)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True, help="axon binary (the reference side)")
    ap.add_argument("--cand", help="axon binary under test (default: --base)")
    ap.add_argument("--base-flags", default="--numerics exact")
    ap.add_argument("--cand-flags", default="--numerics fast")
    ap.add_argument("--flags", default="", help="axon flags for both sides")
    ap.add_argument("--texts", default=os.path.join(here, "..", "tests", "fixtures", "natural_text.txt"))
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--chars", type=int, default=400)
    ap.add_argument("--expect-zero", action="store_true")
    ap.add_argument("--max-dnll", type=float)
    ap.add_argument("--json")
    a = ap.parse_args()
    cand = a.cand or a.base
    both = shlex.split(a.flags)
    bf, cf = shlex.split(a.base_flags) + both, shlex.split(a.cand_flags) + both
    text = open(a.texts, encoding="utf-8").read()
    d, agree, n_pos, nb, nc, diffs = [], 0, 0, 0.0, 0.0, 0
    for w, prompt in enumerate(windows(text, a.windows, a.chars)):
        ib, lb, tb, _ = run(a.base, a.model, prompt, bf)
        ic, lc, tc, _ = run(cand, a.model, prompt, cf)
        if ib != ic:
            sys.exit(f"window {w}: the two sides tokenized differently ({len(ib)} vs {len(ic)} ids)")
        for p in sorted(lb):
            if p not in lc:
                sys.exit(f"window {w}: position {p} missing on the candidate side")
            d.append((w, lc[p] - lb[p]))
            nb += lb[p]
            nc += lc[p]
            n_pos += 1
            agree += tb.get(p) == tc.get(p)
            diffs += lb[p] != lc[p] or tb.get(p) != tc.get(p)
    mean, se = clustered(d)
    res = {
        "model": a.model, "base": [a.base] + bf, "cand": [cand] + cf, "windows": a.windows,
        "positions": n_pos, "dnll": mean, "se": se, "nll_base": nb / max(n_pos, 1), "nll_cand": nc / max(n_pos, 1),
        "top1": agree, "positions_differing": diffs,
    }
    print(f"ΔNLL (cand - base) {mean:+.3e} ± {se:.1e} nats/position (SE clustered by window, {a.windows} windows, "
          f"{n_pos} positions); NLL base {res['nll_base']:.4f}, cand {res['nll_cand']:.4f}; top-1 {agree}/{n_pos}; "
          f"positions differing at all: {diffs}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    if a.expect_zero and diffs:
        print("EXPECT-ZERO FAILED: some position's NLL or top-1 differs")
        sys.exit(1)
    if a.max_dnll is not None and abs(mean) > a.max_dnll:
        print(f"|ΔNLL| {abs(mean):.3e} > --max-dnll {a.max_dnll}")
        sys.exit(1)


if __name__ == "__main__":
    main()
