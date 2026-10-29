#!/usr/bin/env python3
"""a1_verdict.py LOG: WP-A1's table, validity and the pre-registered rule lines from a cuda_a1_session log.

    python3 scripts/a1/a1_verdict.py SESSION.log

Reads the session's `A1 CELL`, `A1 PEAK` and `A1 BENCH device` lines (src/cuda/qwen2/tests.rs cuda_a1_bench /
cuda_a1_peak) and prints:
- `A1 TABLE`: median TMAC/s of real MACs per cell (9 interleaved rounds), x = / gemmh2 of the same bench run;
- `A1 VALIDITY`: VALID or VOID (the pre-registered conditions below);
- `A1 RULE 1..3` and `A1 VERDICT`: the rules of the GitHub issue "A1: fast GEMM probe, pre-registered rules",
  applied verbatim.
A missing cell is "not measured", never a crash: a bella rehearsal (no sm_90a) has baselines only.

Rules (thresholds verbatim from the WP-A1 plan, beat-vllm-scope.md section 6):
1. FP8 gate 3584x18944 nb 1024 < 350 TMAC/s -> Track A closed ("not with this fleet"); Track C gets the budget.
2. W4A16 gate nb 1024 < 230 -> W4A16 is not Marlin-parity; drop (b).
3. down 18944x3584 nb 128 with split-K < 180 for both candidates -> classic M1 is out (record; continue mix-only
   if FP8 passed).
Measurement definitions (pre-registered with the rules):
- A candidate is a family of two CTA tiles with the same instructions: FP8 = probe_fp8_128x64 and
  probe_fp8_128x128 (SS wgmma.m64n64k32.f32.e4m3.e4m3); W4A16 = probe_w4a16_128x64 and probe_w4a16_256x64
  (RS wgmma.m64n64k16.f32.f16.f16). The second tile of each cuts the operands' L2 -> SM bytes per MAC (at T3's
  128 x 64, FP8 at 350 TMAC/s would need 8.2 TB/s and W4A16 at 230 5.8 TB/s).
- Rule 1 / 2 cell: the family's better data-parallel variant (ksplit 1) at gate nb 1024; rule 3: the family's
  best split-K or stream-K variant (either tile) at down nb 128.
- A number counts only if its cell's tolerance check passed in the same job (`tol PASS`).
- If the main bench died and a fallback bench ran, a cell's numbers come from the first run that has it, and its
  x columns from the baselines of that same run.
Validity (VOID = no verdict; a rerun needs the lead's ruling):
- V1 device: an H100 SXM (name has "H100", 132 SMs), the part the rules were derived for;
- V2 spread: (max - min) / median < 5 % in every rule cell that was measured;
- V3 scaling: for each candidate, t(gate nb 1024) / t(q/wo nb 128) >= 10 (the MACs differ 42.3x: a host-bound
  harness, P1's defect, reads near 1);
- V4 baseline: the in-session gemmh2_q4_k at gate nb 1024 within +-15 % of T3's 116.5 TMAC/s (job
  1026-222543-3f73, same harness method).
Exit: 0 (a verdict, any verdict), 2 if the log has no A1 BENCH line at all.
"""
import re
import sys

CELL = re.compile(
    r"A1 CELL (\S+) (\d+)x(\d+) nb (\d+) (\S+): median ([\d.]+) TMAC/s \(min ([\d.]+) max ([\d.]+); "
    r"(\d+) rounds; ([\d.]+) ms\)(?: tol (PASS|FAIL) rel-rms (\S+) rel-max (\S+))?"
)
PEAK = re.compile(r"A1 PEAK (\S+) (\d+) WG/SM: median ([\d.]+) TMAC/s")
DEV = re.compile(r"A1 BENCH device (.*) sm_(\d+) (\d+) SMs;")
BARS = {"fp8": 350.0, "w4a16": 230.0, "down": 180.0}
FAMS = {"fp8": ["fp8_128x64", "fp8_128x128"], "w4a16": ["w4a16_128x64", "w4a16_256x64"]}
VARS = FAMS["fp8"] + FAMS["w4a16"]
T3_GEMMH2 = 116.5


def parse(lines):
    runs, cells, peaks = [], {}, {}
    for ln in lines:
        m = DEV.search(ln)
        if m:
            runs.append({"name": m.group(1), "sm": m.group(2), "sms": int(m.group(3))})
            continue
        m = PEAK.search(ln)
        if m:
            peaks.setdefault(m.group(1), float(m.group(3)))
            continue
        m = CELL.search(ln)
        if m and runs:
            shape, k, rows, nb, kern = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
            key = (shape, nb, kern)
            if key in cells:   # the first run that has it (the main bench before any fallback)
                continue
            cells[key] = {
                "run": len(runs) - 1, "k": k, "rows": rows, "med": float(m.group(6)), "lo": float(m.group(7)),
                "hi": float(m.group(8)), "rounds": int(m.group(9)), "ms": float(m.group(10)),
                "tol": m.group(11), "rr": m.group(12), "rm": m.group(13),
            }
    return runs, cells, peaks


def base_of(cells, shape, nb, run, kern):
    """The baseline `kern` of the same bench run as a candidate cell, or None."""
    for (s, n, k), c in cells.items():
        if s == shape and n == nb and k == kern and c["run"] == run:
            return c
    return None


def variant(cells, shape, nb, cand, kind):
    """The candidate's split-K (`splitk`) or stream-K (`streamk`) cell at (shape, nb): (name, cell) or (None, None)."""
    for (s, n, k), c in cells.items():
        if s == shape and n == nb and k.startswith(cand + "_" + kind):
            return k, c
    return None, None


def fmt(c, x=None):
    if c is None:
        return "–"
    t = f"{c['med']:.1f}"
    if x is not None and x["med"] > 0:
        t += f" ({c['med'] / x['med']:.2f}x)"
    if c["tol"] == "FAIL":
        t += " tol FAIL"
    return t


def counts(c):
    return c is not None and c["tol"] == "PASS"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    lines = open(sys.argv[1], errors="replace").read().splitlines()
    runs, cells, peaks = parse(lines)
    if not runs:
        print("A1 VERDICT: NO DATA (no `A1 BENCH device` line in the log: the bench never started)")
        return 2
    print(f"A1 RUNS {len(runs)} bench run(s); device {runs[0]['name']} sm_{runs[0]['sm']} {runs[0]['sms']} SMs")
    for k, v in peaks.items():
        print(f"A1 PEAK {k}: {v:.1f} TMAC/s raw")

    # ---- the table ----
    shapes = [("gate", "3584x18944", [128, 512, 1024]), ("down", "18944x3584", [128, 512, 1024]),
              ("q/wo", "3584x3584", [128, 1024]), ("kv", "3584x512", [128, 1024])]
    print("A1 TABLE median TMAC/s of real MACs (rows K nb / time), interleaved rounds; (x) = / gemmh2 of the same run;"
          " tol FAIL = doesn't count")
    print("| shape (K x rows) | nb | gemm2s | gemmh2 | " + " | ".join(VARS) + " |")
    print("|---|---|---|---|" + "---|" * len(VARS))
    for shape, kr, nbs in shapes:
        for nb in nbs:
            row = [f"{shape} {kr}", str(nb)]
            row += [fmt(cells.get((shape, nb, "gemm2s_q4_k"))), fmt(cells.get((shape, nb, "gemmh2_q4_k")))]
            for v in VARS:
                c = cells.get((shape, nb, v))
                row.append(fmt(c, c and base_of(cells, shape, nb, c["run"], "gemmh2_q4_k")))
            print("| " + " | ".join(row) + " |")
    print("A1 TABLE (c) nb 128: split-K [S] / stream-K per variant, TMAC/s; (x) = / gemmh2 of the same run")
    print("| shape (K x rows) | " + " | ".join(f"{v} split-K | {v} stream-K" for v in VARS) + " |")
    print("|---|" + "---|" * (2 * len(VARS)))
    for shape, kr, _ in shapes:
        row = [f"{shape} {kr}"]
        for v in VARS:
            for kind in ["splitk", "streamk"]:
                name, c = variant(cells, shape, 128, v, kind)
                t = fmt(c, c and base_of(cells, shape, 128, c["run"], "gemmh2_q4_k"))
                if name and kind == "splitk":
                    t += f" [{name.split('_')[-1]}]"
                row.append(t)
        print("| " + " | ".join(row) + " |")
    for (s, nb, k), c in cells.items():
        if c["tol"]:
            print(f"A1 TOL {s} nb {nb} {k}: {c['tol']} rel-rms {c['rr']} rel-max {c['rm']}")

    # ---- the rule cells ----
    def best_tiles(fam):
        """The family's best data-parallel cell at gate nb 1024 that counts, else its best that doesn't, else
        None: (variant, cell)."""
        cs = [(v, cells.get(("gate", 1024, v))) for v in FAMS[fam]]
        cs = [x for x in cs if x[1] is not None]
        ok = [x for x in cs if counts(x[1])]
        pool = ok or cs
        return max(pool, key=lambda x: x[1]["med"]) if pool else (None, None)

    (r1v, r1), (r2v, r2) = best_tiles("fp8"), best_tiles("w4a16")
    r3 = {}
    for fam in ["fp8", "w4a16"]:
        best = None
        for v in FAMS[fam]:
            for kind in ["splitk", "streamk"]:
                name, c = variant(cells, "down", 128, v, kind)
                if counts(c) and (best is None or c["med"] > best[1]["med"]):
                    best = (name, c)
        r3[fam] = best

    # ---- validity ----
    void = []
    d = runs[0]
    if "H100" not in d["name"] or d["sms"] != 132:
        void.append(f"V1 device {d['name']} with {d['sms']} SMs is not an H100 SXM (132 SMs)")
    rule_cells = [(f"{r1v} gate nb 1024", r1), (f"{r2v} gate nb 1024", r2)]
    rule_cells += [(f"{cand} down nb 128 {b[0]}", b[1]) for cand, b in r3.items() if b]
    for what, c in rule_cells:
        if c is not None:
            sp = (c["hi"] - c["lo"]) / c["med"] * 100 if c["med"] > 0 else float("inf")
            print(f"A1 SPREAD {what}: {sp:.2f} %")
            if sp >= 5.0:
                void.append(f"V2 spread {what} {sp:.2f} % >= 5 %")
    for v in VARS:
        a, b = cells.get(("gate", 1024, v)), cells.get(("q/wo", 128, v))
        if a and b:
            ratio = a["ms"] / b["ms"] if b["ms"] > 0 else float("inf")
            print(f"A1 SCALING {v}: t(gate nb 1024) / t(q/wo nb 128) = {ratio:.1f} (bar >= 10; MACs 42.3x)")
            if ratio < 10:
                void.append(f"V3 scaling {v} {ratio:.1f} < 10")
    gh = cells.get(("gate", 1024, "gemmh2_q4_k"))
    if gh is None:
        void.append("V4 no in-session gemmh2_q4_k at gate nb 1024")
    else:
        dev = (gh["med"] / T3_GEMMH2 - 1) * 100
        print(f"A1 BASELINE gemmh2_q4_k gate nb 1024: {gh['med']:.1f} TMAC/s, {dev:+.1f} % vs T3's {T3_GEMMH2}")
        if abs(dev) > 15:
            void.append(f"V4 gemmh2 gate nb 1024 {gh['med']:.1f} is {dev:+.1f} % off T3's {T3_GEMMH2} (bar +-15 %)")
    valid = not void
    print("A1 VALIDITY: " + ("VALID" if valid else "VOID (" + "; ".join(void) + ")"))
    tag = "" if valid else " [VOID run: no verdict]"

    # ---- the rules ----
    def rule_line(n, what, c, bar, ok_txt, bad_txt):
        if c is None:
            print(f"A1 RULE {n} {what}: NOT MEASURED{tag}")
            return None
        if c["tol"] != "PASS":
            print(f"A1 RULE {n} {what}: median {c['med']:.1f} TMAC/s but tol {c['tol']}: NOT MEASURED "
                  f"(a number counts only with its tolerance check passed){tag}")
            return None
        ok = c["med"] >= bar
        print(f"A1 RULE {n} {what}: median {c['med']:.1f} TMAC/s (tol PASS) vs {bar:.0f}: "
              f"{ok_txt if ok else bad_txt}{tag}")
        return ok

    p1 = rule_line(1, f"FP8 gate 3584x18944 nb 1024 (best tile: {r1v})", r1, BARS["fp8"],
                   "PASS (FP8 can carry M1'; WP-A2 next)",
                   "KILL (FP8 cannot carry M1': Track A closed as \"not with this fleet\"; Track C gets the budget)")
    p2 = rule_line(2, f"W4A16 gate 3584x18944 nb 1024 (best tile: {r2v})", r2, BARS["w4a16"],
                   "PASS (Marlin-parity class: keep (b))",
                   "DROP W4A16 (not Marlin-parity: drop (b))")
    have = {c: b for c, b in r3.items() if b}
    parts = [f"{c} {b[1]['med']:.1f} ({b[0]})" if b else f"{c} not measured" for c, b in r3.items()]
    if not have:
        p3 = None
        print(f"A1 RULE 3 down 18944x3584 nb 128 with split-K: {', '.join(parts)}: NOT MEASURED{tag}")
    elif any(b[1]["med"] >= BARS["down"] for b in have.values()):
        p3 = True
        print(f"A1 RULE 3 down 18944x3584 nb 128 with split-K: {', '.join(parts)} vs 180: PASS "
              f"(at least one candidate >= 180){tag}")
    elif len(have) == 2:
        p3 = False
        print(f"A1 RULE 3 down 18944x3584 nb 128 with split-K: {', '.join(parts)} vs 180: CLASSIC M1 OUT "
              f"(both < 180: the decode step can't reach <= 5 ms on classic; record, continue mix-only if FP8 "
              f"passed){tag}")
    else:
        p3 = None
        print(f"A1 RULE 3 down 18944x3584 nb 128 with split-K: {', '.join(parts)} vs 180: INCOMPLETE (one "
              f"candidate < 180, the other not measured: the rule needs both){tag}")

    # ---- the verdict ----
    if not valid:
        print("A1 VERDICT: VOID (" + "; ".join(void) + "): no verdict; a rerun needs the lead's ruling")
        return 0
    missing = [n for n, p in [("rule 1", p1), ("rule 2", p2), ("rule 3", p3)] if p is None]
    v = []
    v.append("Track A OPEN (FP8 passed: WP-A2 next)" if p1 else
             "Track A CLOSED (FP8 < 350)" if p1 is False else "Track A UNDECIDED (rule 1 not measured)")
    moot = " (recorded; moot: Track A closed)" if p1 is False else ""
    v.append(("W4A16 KEEP" if p2 else "W4A16 DROP" if p2 is False else "W4A16 UNDECIDED") + moot)
    v.append(("classic M1 IN" if p3 else "classic M1 OUT" if p3 is False else "classic M1 UNDECIDED") + moot)
    print("A1 VERDICT: " + "; ".join(v) + (f" (not measured: {', '.join(missing)})" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
