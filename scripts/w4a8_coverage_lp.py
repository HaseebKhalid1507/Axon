#!/usr/bin/env python3
"""W4A8 prefill coverage / gain estimate for docs/design/w4a8-prefill.md section 4 (Checkpoint 0 pre-flight and the
unconditional post-P1 re-run). Analysis only: reads logs, runs nothing on a GPU. Needs numpy + scipy.

  w4a8_coverage_lp.py bound [--log FILE] [--T 128,160,192,256,320] [--slack 10,40,80] [--s-gt 1.35,1.25,1.15]
  w4a8_coverage_lp.py p1 --log FILE --p1 P1_LOG [--gemm-share 0.726]
  w4a8_coverage_lp.py selftest [--log FILE]

INPUT (pinned). The H100 feed-anatomy job 1027-030631-09ae (feat/feed-anatomy, binaries 86953be = main 0c23a87 +
test-only harness, H100 SXM, Qwen2.5-7B-Instruct-Q4_K_M). Public copy: branch gate-logs, commit
4cbe7f3ff482b5accf87789f0dfd195f1669f5f3, path 2026-10/feedanat-86953be-h100/1027-030631-09ae.log. The script
refuses any other bytes (sha256 below). Get it with
  git show 4cbe7f3ff482b5accf87789f0dfd195f1669f5f3:2026-10/feedanat-86953be-h100/1027-030631-09ae.log > 09ae.log

From that log:
  - the anatomy grid ("| depth | n | wall | enqueue | graph | GEMM % |..." rows): g(n) = graph ms x GEMM % / 100
    (anat.py's units), median over the 5 depths (GEMM ms is depth-independent to ~1 %), n in 16..1024;
  - ANATSEG lines: per-class event ms per (depth, n), for the per-product split in `p1` mode;
  - the step-trace table: per arm-round feed passes N, prompt tokens, p50 tokens per pass (anat.py: over steps,
    fed / feeds), feeds s, wall s;
  - fifo60's grid GEMM total over its real passes: 0.784 x 41.32 s / 2 cells = 16.20 s (the "Weighted feed anatomy"
    and "calibration" lines).

g between grid points: linear in n; below 16: g(16); above 1024: g(1024) x n / 1024 (as anat.py). Pass sizes 1..1152
(the serve's pass max).

MODE bound (ESTIMATED, a bound under assumptions; section 4 lists them). For each arm in {fifo60, sjf80} x C=128 x
round, the min over every pass-size histogram x_s >= 0 consistent with
  sum x = N, sum s x = tokens, #(s < p50) <= N/2 + slack, #(s > p50) <= N/2 + slack, and for fifo60 only
  sum g x within +-3 % of 16.20 s,
of the GEMM-time fraction saved
  r = sum_{s >= T} x_s g(s) (1 - 1/S(s)) / sum_s x_s g(s),   S(s) = S_le for s <= 512, S_gt for s > 512,
as a true minimum-ratio LP (Charnes-Cooper: y = t x, t = 1 / sum g x). The aggregate gain is then
  agg = 1 / (1 - feed_share x gemm_share x r) - 1,
with feed_share = feeds / wall of the arm's two rounds and gemm_share = 0.784 (the anatomy's, partly ESTIMATED).
S is the GEMM-time-weighted speedup over all GEMM time, k/v at 1 (the note's S_w). The stop rule reads the LOWER of
the two arms (lead's ruling M9).

MODE p1 (the unconditional post-P1 re-run, pre-registered in section 4). It reads P1's log, which must print:
  W4A8 S <product> nb <nb> <S>       per product in {gate_up, down4, down6, q/wo}, at every Checkpoint-1-quality nb
                                     (at least 384, 512, 768, 1024) and every swept nb (bits-checked; S = today's
                                     kernel ms / production W4A8 ms, median over 7 interleaved rounds)
  W4A8 T <product> <T>|none          the threshold P1's sweep picked, per product; "none" = the product never stays
                                     >= 1.05x from some nb up, so it is not routed and counts at S = 1 (a valid
                                     input, not a missing one; M11)
  PASSHIST <arm> <fed>x<count> ...   single-feed-pass steps of the arm's C=128 serve cell (arm in {fifo60, sjf80})
  PASSMULTI <arm> <fed>/<feeds> ...  steps with >= 2 feed passes (fed tokens over feeds passes)
  FEEDSHARE <arm> <feed_ms> <wall_ms>
Per product p, g_p(n) comes from the ANATSEG classes (gate_up = gate + gate_up; down4; down6; q/wo = q + wo; the
rest, k / v4 / v6 / qkv / down, is uncovered, S = 1), median over depths, interpolated as g. S_p(n): linear between
P1's measured nb, S_p(n) = 1 for n < T_p, S_p(1024) for n > 1024. A multi-pass step is split worst case (its first
feeds - 1 passes at min(T_min - 1, fed / feeds) tokens, the rest in one pass). Decode steps are not included: the
W4A8 path is routed for feed passes only (section 1). Then
  r = sum_passes sum_p g_p(n) (1 - 1/S_p(n)) / sum_passes g_all(n),  agg = 1 / (1 - feed_share x gemm_share x r) - 1,
with gemm_share 0.726 for the stop (the anatomy's 78.4 % less its 1.08 grid-calibration gap, section 4), and 0.784
reported. Prints "W4A8 P1 LOW END <x> %" (the lower arm) and "W4A8 P1 STOP RULE: PROCEED|STOP".
"""
import argparse, hashlib, re, statistics, sys

PIN_SHA256 = "55efde1e43d84ba2dce36a40d05af6a481e157efe769cd7809f98d5e745ae303"
PIN_WHERE = "gate-logs 4cbe7f3ff482b5accf87789f0dfd195f1669f5f3:2026-10/feedanat-86953be-h100/1027-030631-09ae.log"
SMAX = 1152
G_FIFO60 = 0.784 * 41.32 / 2 * 1000.0   # ms, grid GEMM over one fifo60 C=128 cell's passes
BAR = 0.08
PRODUCTS = {"gate_up": ("gate", "gate_up"), "down4": ("down4",), "down6": ("down6",), "q/wo": ("q", "wo")}
GEMM_CLASSES = {"q", "k", "v4", "v6", "qkv", "wo", "gate", "gate_up", "down", "down4", "down6"}


def read_pinned(path):
    b = open(path, "rb").read()
    h = hashlib.sha256(b).hexdigest()
    if h != PIN_SHA256:
        sys.exit(f"{path}: sha256 {h} is not the pinned input {PIN_SHA256} ({PIN_WHERE})")
    return b.decode()


def interp(ns, vs, n):
    if n <= ns[0]:
        return vs[0]
    if n >= ns[-1]:
        return vs[-1] * n / ns[-1]
    for a, b, va, vb in zip(ns, ns[1:], vs, vs[1:]):
        if a <= n <= b:
            return va + (vb - va) * (n - a) / (b - a)


def parse(text):
    grid, seg, cells = {}, {}, {}
    for l in text.splitlines():
        m = re.match(r"\| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \|", l)
        if m:
            n, graph, gp = int(m[2]), float(m[5]), int(m[6])
            grid.setdefault(n, []).append(graph * gp / 100)
        m = re.match(r"ANATSEG depth (\d+) n (\d+): (.*)", l)
        if m:
            d = {k: float(v) for k, v in re.findall(r"([a-z_0-9]+) ([0-9.]+)", m[3])}
            seg.setdefault(int(m[2]), []).append(d)
        m = re.match(r"\| (\w+) \| (\d) \| (\d+) \| (\d+) \| ([\d.]+) \| (\d+) \| ([\d,]+) \| ([\d.]+) \| ([\d.]+) \| "
                     r"([\d.]+) \| (\d+) \| (\d+) \|", l)
        if m and m[3] == "128":
            cells[(m[1], int(m[2]))] = dict(N=int(m[6]), tok=int(m[7].replace(",", "")), feeds=float(m[8]),
                                           wall=float(m[5]), p50=int(m[12]))
    ns = sorted(grid)
    g = [statistics.median(grid[n]) for n in ns]
    return ns, g, seg, cells


def feed_share(cells, arm):
    rs = [v for (a, _), v in cells.items() if a == arm]
    return sum(v["feeds"] for v in rs) / sum(v["wall"] for v in rs)


def min_ratio(num, den, A_ub, b_ub, A_eq, b_eq):
    """min num.x / den.x over A_ub x <= b_ub, A_eq x = b_eq, x >= 0 (den.x > 0): Charnes-Cooper."""
    import numpy as np
    from scipy.optimize import linprog
    k = len(num)
    c = np.append(num, 0.0)
    Au = np.hstack([A_ub, -b_ub.reshape(-1, 1)])
    Ae = np.vstack([np.hstack([A_eq, -b_eq.reshape(-1, 1)]), np.append(den, 0.0)])
    be = np.append(np.zeros(len(b_eq)), 1.0)
    r = linprog(c, A_ub=Au, b_ub=np.zeros(len(b_ub)), A_eq=Ae, b_eq=be, bounds=(0, None), method="highs")
    if r.status != 0:
        return None
    return float(r.fun)


def bound_r(ns, gv, cell, arm, T, slack, s_le, s_gt):
    import numpy as np
    S = np.arange(1, SMAX + 1)
    G = np.array([interp(ns, gv, s) for s in S])
    Sp = np.where(S <= 512, s_le, s_gt)
    num = np.where(S >= T, G * (1 - 1 / Sp), 0.0)
    N, tok, p = cell["N"], cell["tok"], cell["p50"]
    A_ub = [np.where(S < p, 1.0, 0.0), np.where(S > p, 1.0, 0.0)]
    b_ub = [N / 2 + slack, N / 2 + slack]
    if arm == "fifo60":
        A_ub += [G, -G]
        b_ub += [G_FIFO60 * 1.03, -G_FIFO60 * 0.97]
    return min_ratio(num, G, np.array(A_ub), np.array(b_ub),
                     np.array([np.ones_like(G), S.astype(float)]), np.array([N, tok], dtype=float))


def agg(fs, share, r):
    return 1 / (1 - fs * share * r) - 1


def cmd_bound(a, text):
    ns, gv, _, cells = parse(text)
    print(f"input {PIN_WHERE} (sha256 ok)")
    print("g(n) ms (graph x GEMM %, median over depths):", {n: round(v, 2) for n, v in zip(ns, gv)})
    fs = {arm: feed_share(cells, arm) for arm in ("fifo60", "sjf80")}
    print("feed share of wall (C=128, 2 rounds):", {k: round(v, 4) for k, v in fs.items()})
    Ts = [int(x) for x in a.T.split(",")]
    for slack in [int(x) for x in a.slack.split(",")]:
        for s_gt in [float(x) for x in a.s_gt.split(",")]:
            print(f"\n## slack {slack}, S {a.s_le} for n <= 512, S {s_gt} for n > 512, GEMM share {a.gemm_share}")
            print("| T | fifo60 r >= | fifo60 low end | sjf80 r >= | sjf80 low end | lower arm | vs +8 % |")
            print("|---|---|---|---|---|---|---|")
            for T in Ts:
                row, lows = [], []
                for arm in ("fifo60", "sjf80"):
                    r = min(bound_r(ns, gv, cells[(arm, rd)], arm, T, slack, a.s_le, s_gt) for rd in (1, 2))
                    lo = agg(fs[arm], a.gemm_share, r)
                    row += [f"{r:.4f}", f"{lo * 100:+.2f} %"]
                    lows.append(lo)
                lo = min(lows)
                print(f"| {T} | " + " | ".join(row) + f" | {lo * 100:+.2f} % | {'clears' if lo >= BAR else 'MISSES'} |")


def seg_g(seg):
    ns = sorted(seg)
    per = {p: [statistics.median(sum(d.get(c, 0.0) for c in cls) for d in seg[n]) for n in ns]
           for p, cls in PRODUCTS.items()}
    allg = [statistics.median(sum(v for k, v in d.items() if k in GEMM_CLASSES) for d in seg[n]) for n in ns]
    return ns, per, allg


def cmd_p1(a, text):
    _, _, seg, _ = parse(text)
    ns, per, allg = seg_g(seg)
    S, T, hist, multi, fsh = {}, {}, {}, {}, {}
    for l in open(a.p1):
        m = re.match(r"W4A8 S (\S+) nb (\d+) ([\d.]+)", l)
        if m:
            S.setdefault(m[1], {})[int(m[2])] = float(m[3])
        m = re.match(r"W4A8 T (\S+) (\d+|none)\b", l)
        if m:
            T[m[1]] = None if m[2] == "none" else int(m[2])
        m = re.match(r"PASSHIST (\w+) (.*)", l)
        if m:
            for f, c in re.findall(r"(\d+)x(\d+)", m[2]):
                hist.setdefault(m[1], []).append((int(f), int(c)))
        m = re.match(r"PASSMULTI (\w+) (.*)", l)
        if m:
            multi.setdefault(m[1], []).extend((int(f), int(k)) for f, k in re.findall(r"(\d+)/(\d+)", m[2]))
        m = re.match(r"FEEDSHARE (\w+) ([\d.]+) ([\d.]+)", l)
        if m:
            fsh[m[1]] = float(m[2]) / float(m[3])
    # Every product needs its T line (a number or "none"); routed products need S at the Checkpoint-1 nb.
    need = [p for p in PRODUCTS if p not in T
            or (T[p] is not None and not {384, 512, 768, 1024} <= set(S.get(p, {})))]
    arms = [x for x in ("fifo60", "sjf80") if x in hist and x in fsh]
    if need or len(arms) < 2:
        print(f"W4A8 P1 STOP RULE: STOP (inputs missing: products {need}, arms {arms})")
        return 1

    def Sp(p, n):
        if T[p] is None or n < T[p]:
            return 1.0
        xs = sorted(S[p])
        if n >= xs[-1]:
            return S[p][xs[-1]]
        if n <= xs[0]:
            return S[p][xs[0]]
        for lo, hi in zip(xs, xs[1:]):
            if lo <= n <= hi:
                return S[p][lo] + (S[p][hi] - S[p][lo]) * (n - lo) / (hi - lo)

    routed = [t for t in T.values() if t is not None]
    print("W4A8 P1 T:", {p: ("none" if t is None else t) for p, t in T.items()})
    tmin = min(routed) if routed else SMAX + 1
    lows = {}
    for arm in arms:
        passes = list(hist[arm])
        for fed, k in multi.get(arm, []):
            small = min(tmin - 1, fed // k)
            passes += [(small, k - 1), (fed - small * (k - 1), 1)]
        num = den = 0.0
        for n, c in passes:
            if n <= 0:
                continue
            den += c * interp(ns, allg, n)
            num += c * sum(interp(ns, per[p], n) * (1 - 1 / Sp(p, n)) for p in PRODUCTS)
        r = num / den
        steps2 = len(multi.get(arm, []))
        steps1 = sum(c for _, c in hist[arm])
        lows[arm] = agg(fsh[arm], a.gemm_share, r)
        print(f"W4A8 P1 {arm}: GEMM fraction saved r {r:.4f}, feed share {fsh[arm]:.4f}, two-feed-pass steps "
              f"{steps2} of {steps1 + steps2} ({100 * steps2 / max(1, steps1 + steps2):.2f} %), agg at GEMM share "
              f"{a.gemm_share} {lows[arm] * 100:+.2f} %, at 0.784 {agg(fsh[arm], 0.784, r) * 100:+.2f} %")
    lo = min(lows.values())
    print(f"W4A8 P1 LOW END {lo * 100:+.2f} % (lower arm, GEMM share {a.gemm_share})")
    print(f"W4A8 P1 STOP RULE: {'PROCEED' if lo >= BAR else 'STOP'}")
    return 0


def cmd_selftest(a, text):
    """The ratio LP reproduces the numerator-minimising LP of 300e6c3 at T <= 192 (Shady's re-solve agreed within
    0.001) and the registered low ends +9.74 / +9.20 / +8.66 % (fifo60) at slack 10, S 1.35."""
    ns, gv, _, cells = parse(text)
    fs = feed_share(cells, "fifo60")
    want = {128: 9.74, 160: 9.20, 192: 8.66}
    for T, w in want.items():
        r = min(bound_r(ns, gv, cells[("fifo60", rd)], "fifo60", T, 10, 1.35, 1.35) for rd in (1, 2))
        got = agg(fs, 0.784, r) * 100
        assert abs(got - w) < 0.02, (T, got, w)
        print(f"selftest T {T}: fifo60 low end {got:+.2f} % (registered {w:+.2f} %) ok")
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["bound", "p1", "selftest"])
    ap.add_argument("--log", default="09ae.log", help="the pinned input (sha256-checked)")
    ap.add_argument("--T", default="128,160,192,256,320")
    ap.add_argument("--slack", default="10,40,80")
    ap.add_argument("--s-le", type=float, default=1.35)
    ap.add_argument("--s-gt", default="1.35,1.25,1.15")
    ap.add_argument("--gemm-share", type=float, default=None)
    ap.add_argument("--p1", help="P1's job log (mode p1)")
    a = ap.parse_args()
    text = read_pinned(a.log)
    if a.mode == "bound":
        a.gemm_share = a.gemm_share or 0.784
        return cmd_bound(a, text)
    if a.mode == "p1":
        a.gemm_share = a.gemm_share or 0.726
        return cmd_p1(a, text)
    return cmd_selftest(a, text)


if __name__ == "__main__":
    sys.exit(main() or 0)
