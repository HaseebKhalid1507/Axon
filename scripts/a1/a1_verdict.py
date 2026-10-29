#!/usr/bin/env python3
"""a1_verdict.py LOG [SMI_TRACE.csv]: WP-A1's table, validity and the pre-registered rule lines from a
cuda_a1_session log.

    python3 scripts/a1/a1_verdict.py SESSION.log [smi-trace.csv]

Reads the session's `A1 CELL`, `A1 PEAK`, `A1 BENCH device` and `A1 STAGE` lines (src/cuda/qwen2/tests.rs
cuda_a1_bench / cuda_a1_peak / cuda_a1_session), the `A1 SMI` line scripts/a1/a1_h100.sh appends, and optionally
the job's nvidia-smi trace (csv with a header: timestamp, clocks.sm, power.draw, ...). Prints:
- `A1 TABLE`: median TMAC/s of real MACs per cell (9 interleaved rounds), x = / gemmh2 of the same bench run;
  a cell whose median time per launch is under 10 us, and every kv nb 128 cell, is marked `~` (host-launch-
  sensitive: the 3 timed fires partly measure host enqueue; no rule reads such a cell);
- `A1 CLOCKS`: SM clock / power / temperature over the bench window(s), when a trace is given (recorded, not a
  rule input);
- `A1 VALIDITY`: VALID or VOID (the pre-registered conditions below);
- `A1 RULE 1..3` and `A1 VERDICT`: the rules of GitHub issue #174 "A1: fast GEMM probe, pre-registered rules"
  (with its amendment of 2026-10-29), applied verbatim.
A missing cell is "not measured", never a crash: a bella rehearsal (no sm_90a) has baselines only.

Rules (thresholds verbatim from the WP-A1 plan, beat-vllm-scope.md section 6):
1. FP8 gate 3584x18944 nb 1024 < 350 TMAC/s -> Track A closed ("not with this fleet"); Track C gets the budget.
2. W4A16 gate nb 1024 < 230 -> W4A16 is not Marlin-parity; drop (b).
3. down 18944x3584 nb 128 with split-K < 180 for both candidates -> classic M1 is out (record; continue mix-only
   if FP8 passed).
Measurement definitions (pre-registered with the rules, #174 and its amendment):
- A candidate is a family of CTA tiles: FP8 = probe_fp8_128x64 and probe_fp8_128x128 (SS wgmma.m64n64k32
  .f32.e4m3.e4m3) and probe_fp8n128_128x128 (the 128 x 128 tile on SS wgmma.m64n128k32); W4A16 =
  probe_w4a16_128x64 and probe_w4a16_256x64 (RS wgmma.m64n64k16.f32.f16.f16).
- A cell COUNTS only if its tolerance check passed in the same job (`tol PASS`). Only counting cells are rule
  cells, and only counting cells enter V2 and V3: a variant that fails tolerance can't void another's rule.
- Rule 1 / 2 cell: the family's best COUNTING data-parallel variant (ksplit 1) at gate nb 1024; rule 3: the
  family's best COUNTING split-K or stream-K variant (any tile) at down nb 128.
- Rule 1 KILL scope: a KILL closes Track A only if probe_fp8n128_128x128's gate nb 1024 cell counted. If only the
  n64 tiles counted, the KILL is "FP8 at the n64 instruction" and Track A stays open pending a wider-N result.
- If the main bench died and a fallback bench ran, a cell's numbers come from the first run that has it, and its
  x columns from the baselines of that same run.
Validity (VOID = no verdict; a rerun needs the lead's ruling):
- V1 device: an H100 SXM: the bench's CUDA device name has "H100" and neither "NVL" nor "PCIe", 132 SMs, and the
  `A1 SMI` power.limit >= 690 W (SXM5: 700 W; NVL 350-400 W, PCIe 350 W). If the power limit is unreadable or
  the SMI line is missing, the name must be exactly "NVIDIA H100 80GB HBM3" (the SXM5 part; T3's and P1's pods).
- V2 spread: (max - min) / median < 5 % in every rule cell (counting cells only);
- V3 scaling, per variant whose gate nb 1024 and q/wo nb 128 cells both count: t(gate nb 1024) / t(q/wo nb 128)
  >= 0.5 x W, W = waves(gate nb 1024) / waves(q/wo nb 128), waves = ceil(tiles / SMs) (the kernels are
  persistent: a CTA's time scales with its tiles, not with the MACs). W = 18 for the 128 x 64 tiles, 9 for
  128 x 128 and 256 x 64, so the bars are 9 and 4.5. T3 measured 15.5 against W 17.9 on this skeleton (0.87 W);
  P1's host-bound defect read about 1. If a rule reads a number but no variant is checkable, VOID.
- V4 baseline: the in-session gemmh2_q4_k at gate nb 1024 within +-15 % of T3's 116.5 TMAC/s (job
  1026-222543-3f73, same harness method).
Exit: 0 (a verdict, any verdict), 2 if the log has no A1 BENCH line at all.
"""
import math
import re
import statistics
import sys
import time

CELL = re.compile(
    r"A1 CELL (\S+) (\d+)x(\d+) nb (\d+) (\S+): median ([\d.]+) TMAC/s \(min ([\d.]+) max ([\d.]+); "
    r"(\d+) rounds; ([\d.]+) ms\)(?: tol (PASS|FAIL) rel-rms (\S+) rel-max (\S+))?"
)
PEAK = re.compile(r"A1 PEAK (\S+) (\d+) WG/SM: median ([\d.]+) TMAC/s")
DEV = re.compile(r"A1 BENCH device (.*) sm_(\d+) (\d+) SMs;")
SMI = re.compile(r"^A1 SMI (.*)$")
STAGE_START = re.compile(r"A1 STAGE cuda_a1_bench: start at unix ([\d.]+)")
STAGE_END = re.compile(r"A1 STAGE cuda_a1_bench: rc \S+ after [\d.]+ s at unix ([\d.]+)")
BARS = {"fp8": 350.0, "w4a16": 230.0, "down": 180.0}
FAMS = {"fp8": ["fp8_128x64", "fp8_128x128", "fp8n128_128x128"], "w4a16": ["w4a16_128x64", "w4a16_256x64"]}
VARS = FAMS["fp8"] + FAMS["w4a16"]
TILE = {"fp8_128x64": (128, 64), "fp8_128x128": (128, 128), "fp8n128_128x128": (128, 128),
        "w4a16_128x64": (128, 64), "w4a16_256x64": (256, 64)}
FP8_WIDE = "fp8n128_128x128"
T3_GEMMH2 = 116.5
SXM_NAME = "NVIDIA H100 80GB HBM3"
HOST_US = 10.0


def parse(lines):
    runs, cells, peaks, smi, windows = [], {}, {}, None, []
    start = None
    for ln in lines:
        m = SMI.search(ln)
        if m:
            smi = [f.strip() for f in m.group(1).split(",")]
            continue
        m = STAGE_START.search(ln)
        if m:
            start = float(m.group(1))
            continue
        m = STAGE_END.search(ln)
        if m and start is not None:
            windows.append((start, float(m.group(1))))
            start = None
            continue
        m = DEV.search(ln)
        if m:
            runs.append({"name": m.group(1).strip(), "sm": m.group(2), "sms": int(m.group(3))})
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
                "run": len(runs) - 1, "k": k, "rows": rows, "nb": nb, "kern": kern, "med": float(m.group(6)),
                "lo": float(m.group(7)), "hi": float(m.group(8)), "rounds": int(m.group(9)),
                "ms": float(m.group(10)), "tol": m.group(11), "rr": m.group(12), "rm": m.group(13),
            }
    return runs, cells, peaks, smi, windows


def num(text):
    m = re.search(r"[-\d.]+", text or "")
    try:
        return float(m.group(0)) if m else None
    except ValueError:
        return None


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


def launches(kern):
    return 2 if ("_splitk" in kern or "_streamk" in kern) else 1


def host_sensitive(shape, nb, c):
    return (shape == "kv" and nb == 128) or c["ms"] * 1000.0 / launches(c["kern"]) < HOST_US


def fmt(c, shape, nb, x=None):
    if c is None:
        return "–"
    t = f"{c['med']:.2f}"
    if x is not None and x["med"] > 0:
        t += f" ({c['med'] / x['med']:.2f}x)"
    if c["tol"] == "FAIL":
        t += " tol FAIL"
    if host_sensitive(shape, nb, c):
        t += " ~"
    return t


def counts(c):
    return c is not None and c["tol"] == "PASS"


def waves(var, rows, nb, sms):
    r, t = TILE[var]
    return math.ceil(math.ceil(rows / r) * math.ceil(nb / t) / sms)


def smi_trace(path, windows):
    """A1 CLOCKS lines: SM clock, power, temperature over the bench windows (all samples if there are none)."""
    try:
        rows = [ln.rstrip("\n") for ln in open(path, errors="replace") if ln.strip()]
    except OSError as e:
        print(f"A1 CLOCKS no trace ({e})")
        return
    if not rows:
        print("A1 CLOCKS no trace (empty)")
        return
    head = [h.strip() for h in rows[0].split(",")]
    col = {h.split(" ")[0]: i for i, h in enumerate(head)}
    samples = []
    for ln in rows[1:]:
        f = [x.strip() for x in ln.split(",")]
        if len(f) != len(head) or f[0] == head[0]:
            continue
        try:
            ts, frac = f[col["timestamp"]].split(".") if "." in f[col["timestamp"]] else (f[col["timestamp"]], "0")
            t = time.mktime(time.strptime(ts, "%Y/%m/%d %H:%M:%S")) + float("0." + frac)
        except (KeyError, ValueError):
            continue
        samples.append((t, f))
    inw = [s for s in samples if any(a <= s[0] <= b for a, b in windows)] if windows else samples
    what = f"{len(windows)} bench window(s)" if windows else "the whole session (no bench window in the log)"
    if not inw:
        print(f"A1 CLOCKS no trace samples in {what} ({len(samples)} samples in the file)")
        return

    def stat(key):
        if key not in col:
            return None
        v = [num(f[col[key]]) for _, f in inw]
        v = [x for x in v if x is not None]
        return (min(v), statistics.median(v), max(v)) if v else None

    parts = []
    for key, unit in [("clocks.sm", "MHz"), ("clocks.mem", "MHz"), ("power.draw", "W"), ("temperature.gpu", "C")]:
        s = stat(key)
        if s:
            parts.append(f"{key} min {s[0]:.0f} median {s[1]:.0f} max {s[2]:.0f} {unit}")
    rk = col.get("clocks_event_reasons.active", col.get("clocks_throttle_reasons.active"))
    if rk is not None:
        rs = sorted({f[rk] for _, f in inw})
        parts.append("event reasons " + "/".join(rs))
    print(f"A1 CLOCKS over {what}, {len(inw)} samples: " + "; ".join(parts) + " (recorded, not a rule input)")


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    lines = open(sys.argv[1], errors="replace").read().splitlines()
    runs, cells, peaks, smi, windows = parse(lines)
    if not runs:
        print("A1 VERDICT: NO DATA (no `A1 BENCH device` line in the log: the bench never started)")
        return 2
    d = runs[0]
    print(f"A1 RUNS {len(runs)} bench run(s); device {d['name']} sm_{d['sm']} {d['sms']} SMs")
    if smi:
        print("A1 SMI " + ", ".join(smi))
    for k, v in peaks.items():
        print(f"A1 PEAK {k}: {v:.2f} TMAC/s raw")
    if len(sys.argv) == 3:
        smi_trace(sys.argv[2], windows)

    # ---- the table ----
    shapes = [("gate", "3584x18944", [128, 512, 1024]), ("down", "18944x3584", [128, 512, 1024]),
              ("q/wo", "3584x3584", [128, 1024]), ("kv", "3584x512", [128, 1024])]
    print("A1 TABLE median TMAC/s of real MACs (rows K nb / time), interleaved rounds; (x) = / gemmh2 of the same run;"
          " tol FAIL = doesn't count; ~ = host-launch-sensitive (< 10 us per launch, or kv nb 128)")
    print("| shape (K x rows) | nb | gemm2s | gemmh2 | " + " | ".join(VARS) + " |")
    print("|---|---|---|---|" + "---|" * len(VARS))
    for shape, kr, nbs in shapes:
        for nb in nbs:
            row = [f"{shape} {kr}", str(nb)]
            row += [fmt(cells.get((shape, nb, "gemm2s_q4_k")), shape, nb),
                    fmt(cells.get((shape, nb, "gemmh2_q4_k")), shape, nb)]
            for v in VARS:
                c = cells.get((shape, nb, v))
                row.append(fmt(c, shape, nb, c and base_of(cells, shape, nb, c["run"], "gemmh2_q4_k")))
            print("| " + " | ".join(row) + " |")
    print("A1 TABLE (c) nb 128: split-K [S] / stream-K per variant, TMAC/s; (x) = / gemmh2 of the same run")
    print("| shape (K x rows) | " + " | ".join(f"{v} split-K | {v} stream-K" for v in VARS) + " |")
    print("|---|" + "---|" * (2 * len(VARS)))
    for shape, kr, _ in shapes:
        row = [f"{shape} {kr}"]
        for v in VARS:
            for kind in ["splitk", "streamk"]:
                name, c = variant(cells, shape, 128, v, kind)
                t = fmt(c, shape, 128, c and base_of(cells, shape, 128, c["run"], "gemmh2_q4_k"))
                if name and kind == "splitk":
                    t += f" [{name.split('_')[-1]}]"
                row.append(t)
        print("| " + " | ".join(row) + " |")
    for (s, nb, k), c in cells.items():
        if c["tol"]:
            print(f"A1 TOL {s} nb {nb} {k}: {c['tol']} rel-rms {c['rr']} rel-max {c['rm']}")

    # ---- the rule cells: counting cells only ----
    def best_tiles(fam):
        """(variant, cell) of the family's best COUNTING data-parallel cell at gate nb 1024, else (None, None);
        and its best non-counting one (for the NOT MEASURED line only)."""
        cs = [(v, cells.get(("gate", 1024, v))) for v in FAMS[fam]]
        cs = [x for x in cs if x[1] is not None]
        ok = [x for x in cs if counts(x[1])]
        bad = [x for x in cs if not counts(x[1])]
        best = max(ok, key=lambda x: x[1]["med"]) if ok else (None, None)
        return best, (max(bad, key=lambda x: x[1]["med"]) if bad else (None, None))

    ((r1v, r1), r1bad), ((r2v, r2), r2bad) = best_tiles("fp8"), best_tiles("w4a16")
    wide_counted = counts(cells.get(("gate", 1024, FP8_WIDE)))
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
    name = d["name"]
    plim = num(smi[1]) if smi and len(smi) > 1 else None
    if "H100" not in name or "NVL" in name or "PCIe" in name or d["sms"] != 132:
        void.append(f"V1 device {name} with {d['sms']} SMs is not an H100 SXM (132 SMs, not NVL / PCIe)")
    elif plim is not None and plim < 690:
        void.append(f"V1 power.limit {plim:.0f} W < 690 W: not an H100 SXM at its rated power")
    elif plim is None and name != SXM_NAME:
        void.append(f"V1 power limit unreadable and the name {name!r} is not {SXM_NAME!r}")
    else:
        print(f"A1 V1 device {name}, {d['sms']} SMs, power.limit "
              + (f"{plim:.0f} W" if plim is not None else "unreadable (name is the SXM5 part's)") + ": ok")
    rule_cells = [(f"{r1v} gate nb 1024", r1), (f"{r2v} gate nb 1024", r2)]
    rule_cells += [(f"{cand} down nb 128 {b[0]}", b[1]) for cand, b in r3.items() if b]
    for what, c in rule_cells:
        if c is not None:
            sp = (c["hi"] - c["lo"]) / c["med"] * 100 if c["med"] > 0 else float("inf")
            print(f"A1 SPREAD {what}: {sp:.2f} %")
            if sp >= 5.0:
                void.append(f"V2 spread {what} {sp:.2f} % >= 5 %")
    checked = 0
    for v in VARS:
        a, b = cells.get(("gate", 1024, v)), cells.get(("q/wo", 128, v))
        if a is None or b is None:
            continue
        if not (counts(a) and counts(b)):
            print(f"A1 SCALING {v}: not checked (a cell doesn't count: gate nb 1024 tol {a['tol']}, "
                  f"q/wo nb 128 tol {b['tol']})")
            continue
        sms = runs[a["run"]]["sms"]
        w = waves(v, 18944, 1024, sms) / waves(v, 3584, 128, sms)
        bar = 0.5 * w
        ratio = a["ms"] / b["ms"] if b["ms"] > 0 else float("inf")
        checked += 1
        print(f"A1 SCALING {v}: t(gate nb 1024) / t(q/wo nb 128) = {ratio:.2f}, waves ratio {w:.0f}, "
              f"bar >= {bar:.2f}: {'ok' if ratio >= bar else 'FAIL'}")
        if ratio < bar:
            void.append(f"V3 scaling {v} {ratio:.2f} < {bar:.2f} (0.5 x waves ratio {w:.0f})")
    if checked == 0 and (r1 or r2 or any(r3.values())):
        void.append("V3 no variant has both gate nb 1024 and q/wo nb 128 counting: the scaling check can't run")
    gh = cells.get(("gate", 1024, "gemmh2_q4_k"))
    if gh is None:
        void.append("V4 no in-session gemmh2_q4_k at gate nb 1024")
    else:
        dev = (gh["med"] / T3_GEMMH2 - 1) * 100
        print(f"A1 BASELINE gemmh2_q4_k gate nb 1024: {gh['med']:.2f} TMAC/s, {dev:+.1f} % vs T3's {T3_GEMMH2}")
        if abs(dev) > 15:
            void.append(f"V4 gemmh2 gate nb 1024 {gh['med']:.2f} is {dev:+.1f} % off T3's {T3_GEMMH2} (bar +-15 %)")
    valid = not void
    print("A1 VALIDITY: " + ("VALID" if valid else "VOID (" + "; ".join(void) + ")"))
    tag = "" if valid else " [VOID run: no verdict]"

    # ---- the rules ----
    def rule_line(n, what, c, bad, bar, ok_txt, bad_txt):
        if c is None:
            extra = ""
            if bad[1] is not None:
                extra = (f" ({bad[0]} ran at {bad[1]['med']:.2f} TMAC/s but tol {bad[1]['tol']}: a number counts "
                         f"only with its tolerance check passed)")
            print(f"A1 RULE {n} {what}: NOT MEASURED{extra}{tag}")
            return None
        ok = c["med"] >= bar
        print(f"A1 RULE {n} {what}: median {c['med']:.2f} TMAC/s (tol PASS) vs {bar:.0f}: "
              f"{ok_txt if ok else bad_txt}{tag}")
        return ok

    kill1 = ("KILL (FP8 cannot carry M1': Track A closed as \"not with this fleet\"; Track C gets the budget)"
             if wide_counted else
             f"KILL AT n64 ONLY (FP8 on m64n64k32 < 350; {FP8_WIDE} did not count: Track A stays open pending a "
             f"wider-N result)")
    p1 = rule_line(1, f"FP8 gate 3584x18944 nb 1024 (best counting tile: {r1v})", r1, r1bad, BARS["fp8"],
                   "PASS (FP8 can carry M1'; WP-A2 next)", kill1)
    p2 = rule_line(2, f"W4A16 gate 3584x18944 nb 1024 (best counting tile: {r2v})", r2, r2bad, BARS["w4a16"],
                   "PASS (Marlin-parity class: keep (b))",
                   "DROP W4A16 (not Marlin-parity: drop (b))")
    have = {c: b for c, b in r3.items() if b}
    parts = [f"{c} {b[1]['med']:.2f} ({b[0]})" if b else f"{c} not measured" for c, b in r3.items()]
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
    closed = p1 is False and wide_counted
    missing = [n for n, p in [("rule 1", p1), ("rule 2", p2), ("rule 3", p3)] if p is None]
    v = []
    if p1:
        v.append("Track A OPEN (FP8 passed: WP-A2 next)")
    elif closed:
        v.append("Track A CLOSED (FP8 < 350 at n64 and n128)")
    elif p1 is False:
        v.append(f"Track A OPEN PENDING WIDER-N (FP8 < 350 at n64; {FP8_WIDE} not measured)")
    else:
        v.append("Track A UNDECIDED (rule 1 not measured)")
    moot = " (recorded; moot: Track A closed)" if closed else ""
    v.append(("W4A16 KEEP" if p2 else "W4A16 DROP" if p2 is False else "W4A16 UNDECIDED") + moot)
    v.append(("classic M1 IN" if p3 else "classic M1 OUT" if p3 is False else "classic M1 UNDECIDED") + moot)
    print("A1 VERDICT: " + "; ".join(v) + (f" (not measured: {', '.join(missing)})" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
