#!/usr/bin/env python3
"""a1_verdict.py LOG [SMI_TRACE.csv]: WP-A1's table, validity and the pre-registered rule lines from a
cuda_a1_session log.

    python3 scripts/a1/a1_verdict.py SESSION.log [smi-trace.csv]

Reads the session's `A1 CELL`, `A1 PEAK`, `A1 BENCH device`, `A1 STAGE`, `A1 COMPILE` and `A1 MODULE` lines
(src/cuda/qwen2/tests.rs cuda_a1_bench / cuda_a1_peak / cuda_a1_session / probe_fast_module), the `A1 SMI` and
`A1 CUBIN` lines scripts/a1/a1_h100.sh appends, and optionally the job's nvidia-smi trace (csv with a header:
timestamp, clocks.sm, power.draw, ...). Prints:
- `A1 TABLE`: median TMAC/s of real MACs per cell (9 interleaved rounds), x = / gemmh2 of the same bench run;
  a cell whose median time per launch is under 10 us, and every kv nb 128 cell, is marked `~` (host-launch-
  sensitive: the 3 timed fires partly measure host enqueue; no rule reads such a cell);
- `A1 CLOCKS`: SM clock / power / temperature over the bench window(s), when a trace is given (recorded, not a
  rule input);
- `A1 SCALING`, `A1 WIDE-N`, `A1 V5`: the per-variant checks below;
- `A1 VALIDITY`: VALID or VOID (the pre-registered conditions below);
- `A1 RULE 1..3` and `A1 VERDICT`: the rules of GitHub issue #174 "A1: fast GEMM probe, pre-registered rules"
  (with its Amendments 1 and 2 of 2026-10-29), applied verbatim.
A missing cell is "not measured", never a crash: a bella rehearsal (no sm_90a) has baselines only.

Rules (thresholds verbatim from the WP-A1 plan, beat-vllm-scope.md section 6):
1. FP8 gate 3584x18944 nb 1024 < 350 TMAC/s -> Track A closed ("not with this fleet"); Track C gets the budget.
2. W4A16 gate nb 1024 < 230 -> W4A16 is not Marlin-parity; drop (b).
3. down 18944x3584 nb 128 with split-K < 180 for both candidates -> classic M1 is out (record; continue mix-only
   if FP8 passed).
Measurement definitions (pre-registered with the rules, #174 and its amendments):
- A candidate is a family of CTA tiles: FP8 = probe_fp8_128x64 and probe_fp8_128x128 (SS wgmma.m64n64k32
  .f32.e4m3.e4m3) and probe_fp8n128_128x128 (the 128 x 128 tile on SS wgmma.m64n128k32); W4A16 =
  probe_w4a16_128x64 and probe_w4a16_256x64 (RS wgmma.m64n64k16.f32.f16.f16).
- A cell COUNTS only if its tolerance check passed in the same job (`tol PASS`).
- A variant is READABLE only if its own V3 scaling check ran (its gate nb 1024 and q/wo nb 128 cells both count)
  and passed. Rules read readable variants only; a variant that fails tolerance or V3 drops out of the rules
  alone (it can't void the other family's rule).
- Rule 1 / 2 cell: the family's best readable data-parallel variant (ksplit 1) at gate nb 1024; rule 3: the
  family's best counting split-K or stream-K cell (any tile) at down nb 128 of a readable variant.
- No KILL past an unread number: if a counting but unreadable cell of the family is at or above the bar, the
  rule reads UNRESOLVED (no PASS, no KILL / DROP / OUT).
- Bars are compared on the raw median (`raw median`, 6 decimals), never the 2-decimal print.
- Rule 1 KILL scope: a KILL closes Track A only if probe_fp8n128_128x128 is CLEAN: its gate nb 1024 cell counts
  with spread < 5 %, its V3 ran and passed, none of its cells is out of tolerance, its tiny and ragged stages
  both read `ok` and its memcheck stage (if run) didn't fault. Otherwise the KILL reads "KILL AT n64 ONLY" and
  Track A stays OPEN PENDING WIDER-N, whose next step is pre-registered in #174 Amendment 2.
- If the main bench died and a fallback bench ran, a cell's numbers come from the first run that has it, and its
  x columns from the baselines of that same run.
Validity (VOID = no verdict; a rerun needs the lead's ruling):
- V1 device: an H100 SXM: the bench's CUDA device name has "H100" and neither "NVL" nor "PCIe", 132 SMs, and the
  `A1 SMI` power.limit >= 690 W (SXM5: 700 W; NVL 350-400 W, PCIe 350 W). If the power limit is unreadable or
  the SMI line is missing, the name must be exactly "NVIDIA H100 80GB HBM3" (the SXM5 part; T3's and P1's pods).
- V2 spread: (max - min) / median < 5 % in every rule cell the rules read;
- V3 scaling, per variant whose gate nb 1024 and q/wo nb 128 cells both count: t(gate nb 1024) / t(q/wo nb 128)
  >= 0.5 x W, W = waves(gate nb 1024) / waves(q/wo nb 128), waves = ceil(tiles / SMs) (the kernels are
  persistent: a CTA's time scales with its tiles, not with the MACs). W = 18 for the 128 x 64 tiles, 9 for
  128 x 128 and 256 x 64, so the bars are 9 and 4.5. T3 measured 15.5 against W 17.9 on this skeleton (0.87 W);
  P1's host-bound defect read about 1. A variant failing V3 is unreadable; VOID if every checked variant fails
  (the timing harness is shared: that's its signal), or if no variant is checkable while a candidate cell counts.
- V4 baseline: the in-session gemmh2_q4_k at gate nb 1024 within +-15 % of T3's 116.5 TMAC/s (job
  1026-222543-3f73, same harness method).
- V5 compile: if this job compiled a kernel module itself rather than loading the kit's cubins (an `A1 MODULE
  probe-fast` line other than `cache Hit`, new kernel-cache entries in `A1 CUBIN`, or either line missing), the
  compile check (`A1 STAGE compile: ok`) must have passed, on the same NVRTC as a probe module built here.
Exit: 0 (a verdict, any verdict), 2 if the log has no A1 BENCH line at all.
"""
import math
import re
import statistics
import sys
import time

CELL = re.compile(
    r"A1 CELL (\S+) (\d+)x(\d+) nb (\d+) (\S+): median ([\d.]+) TMAC/s \(min ([\d.]+) max ([\d.]+); "
    r"(\d+) rounds; ([\d.]+) ms(?:; raw median ([\d.]+))?\)(?: tol (PASS|FAIL) rel-rms (\S+) rel-max (\S+))?"
)
PEAK = re.compile(r"A1 PEAK (\S+) (\d+) WG/SM: median ([\d.]+) TMAC/s")
DEV = re.compile(r"A1 BENCH device (.*) sm_(\d+) (\d+) SMs;")
SMI = re.compile(r"^A1 SMI (.*)$")
STAGE_START = re.compile(r"A1 STAGE cuda_a1_bench: start at unix ([\d.]+)")
STAGE_END = re.compile(r"A1 STAGE cuda_a1_bench: rc \S+ after [\d.]+ s at unix ([\d.]+)")
STAGE_OUT = re.compile(r"A1 STAGE (sanitizer )?([a-z0-9_]+) (tiny|ragged): (.+)$")
COMPILE = re.compile(r"A1 STAGE compile: (.+)$")
COMPILE_NV = re.compile(r"A1 COMPILE sm_90a: \d+ bytes of cubin \(NVRTC (\d+\.\d+)\)")
MODULE = re.compile(r"A1 MODULE probe-fast: cache (\w+), NVRTC (\d+\.\d+|unknown)")
CUBIN = re.compile(r"^A1 CUBIN new cache entries during the run: (none|\d+)")
END = re.compile(r"A1 END (.+)$")
BARS = {"fp8": 350.0, "w4a16": 230.0, "down": 180.0}
FAMS = {"fp8": ["fp8_128x64", "fp8_128x128", "fp8n128_128x128"], "w4a16": ["w4a16_128x64", "w4a16_256x64"]}
VARS = FAMS["fp8"] + FAMS["w4a16"]
TILE = {"fp8_128x64": (128, 64), "fp8_128x128": (128, 128), "fp8n128_128x128": (128, 128),
        "w4a16_128x64": (128, 64), "w4a16_256x64": (256, 64)}
FP8_WIDE = "fp8n128_128x128"
T3_GEMMH2 = 116.5
SXM_NAME = "NVIDIA H100 80GB HBM3"
HOST_US = 10.0
WIDER_N_NEXT = ("next step pre-registered in #174 Amendment 2: no spend until the reason n128 wasn't clean is found "
                "on bella, then one FP8-only rerun of at most $1")


def parse(lines):
    runs, cells, peaks, smi, windows = [], {}, {}, None, []
    meta = {"stages": {}, "compile": None, "compile_nv": None, "modules": [], "cubin": None, "end": None}
    start = None
    for ln in lines:
        m = SMI.search(ln)
        if m:
            smi = [f.strip() for f in m.group(1).split(",")]
            continue
        m = CUBIN.search(ln)
        if m:
            meta["cubin"] = 0 if m.group(1) == "none" else int(m.group(1))
            continue
        m = MODULE.search(ln)
        if m:
            meta["modules"].append((m.group(1), m.group(2)))
        m = COMPILE_NV.search(ln)
        if m:
            meta["compile_nv"] = m.group(1)
        m = COMPILE.search(ln)
        if m:
            meta["compile"] = m.group(1).strip()
            continue
        m = END.search(ln)
        if m:
            meta["end"] = m.group(1).strip()
            continue
        m = STAGE_OUT.search(ln)
        if m:
            meta["stages"][("sanitizer" if m.group(1) else "plain", m.group(2), m.group(3))] = m.group(4).strip()
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
            med = float(m.group(6))
            cells[key] = {
                "run": len(runs) - 1, "k": k, "rows": rows, "nb": nb, "kern": kern, "med": med,
                "raw": float(m.group(11)) if m.group(11) else med,
                "lo": float(m.group(7)), "hi": float(m.group(8)), "rounds": int(m.group(9)),
                "ms": float(m.group(10)), "tol": m.group(12), "rr": m.group(13), "rm": m.group(14),
            }
    return runs, cells, peaks, smi, windows, meta


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


def var_of(kern):
    """The candidate variant a kernel name belongs to (itself, or its _splitkS / _streamk schedule), or None."""
    for v in sorted(VARS, key=len, reverse=True):
        if kern == v or kern.startswith(v + "_"):
            return v
    return None


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


def spread(c):
    return (c["hi"] - c["lo"]) / c["med"] * 100 if c["med"] > 0 else float("inf")


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
    runs, cells, peaks, smi, windows, meta = parse(lines)
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

    void = []
    # ---- V1 device ----
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

    # ---- V3 scaling, per variant: who is readable ----
    v3 = {}
    for v in VARS:
        a, b = cells.get(("gate", 1024, v)), cells.get(("q/wo", 128, v))
        if a is None and b is None:
            v3[v] = "absent"
            continue
        if not (counts(a) and counts(b)):
            v3[v] = "unchecked"
            print(f"A1 SCALING {v}: not checked (gate nb 1024 {'tol ' + str(a['tol']) if a else 'missing'}, "
                  f"q/wo nb 128 {'tol ' + str(b['tol']) if b else 'missing'}): {v} is not readable")
            continue
        sms = runs[a["run"]]["sms"]
        w = waves(v, 18944, 1024, sms) / waves(v, 3584, 128, sms)
        bar = 0.5 * w
        ratio = a["ms"] / b["ms"] if b["ms"] > 0 else float("inf")
        v3[v] = "ok" if ratio >= bar else "FAIL"
        print(f"A1 SCALING {v}: t(gate nb 1024) / t(q/wo nb 128) = {ratio:.2f}, waves ratio {w:.0f}, "
              f"bar >= {bar:.2f}: " + ("ok" if ratio >= bar else f"FAIL ({v} is not readable)"))
    checked = [v for v in VARS if v3[v] in ("ok", "FAIL")]
    any_count = any(counts(c) and var_of(k) for (_, _, k), c in cells.items())
    if checked and all(v3[v] == "FAIL" for v in checked):
        void.append("V3 every checked variant fails its scaling check (" + ", ".join(checked)
                    + "): the shared timing harness is suspect")
    elif not checked and any_count:
        void.append("V3 no variant has both gate nb 1024 and q/wo nb 128 counting: the scaling check can't run")

    # ---- the rule cells: readable variants only ----
    def pick(fam, items, bar):
        """items: [(label, variant, cell)]. -> best readable (label, cell) or None; counting-but-unreadable ones at
        or above the bar; counting-but-unreadable ones; non-counting ones."""
        cnt = [x for x in items if counts(x[2])]
        ok = [x for x in cnt if v3.get(x[1]) == "ok"]
        unread = [x for x in cnt if v3.get(x[1]) != "ok"]
        best = max(ok, key=lambda x: x[2]["raw"]) if ok else None
        hi = [x for x in unread if x[2]["raw"] >= bar]
        bad = [x for x in items if x[2] is not None and not counts(x[2])]
        return best, hi, unread, bad

    gate_items = {f: [(v, v, cells.get(("gate", 1024, v))) for v in FAMS[f]] for f in FAMS}
    g = {f: pick(f, gate_items[f], BARS[f]) for f in FAMS}
    down_items = {f: [(k, var_of(k), c) for (s, nb, k), c in cells.items()
                      if s == "down" and nb == 128 and var_of(k) in FAMS[f] and ("_splitk" in k or "_streamk" in k)]
                  for f in FAMS}
    r3 = {f: pick(f, down_items[f], BARS["down"]) for f in FAMS}

    # ---- V2 spread on the cells the rules read ----
    rule_cells = [(f"{g[f][0][0]} gate nb 1024", g[f][0][2]) for f in FAMS if g[f][0]]
    rule_cells += [(f"{f} down nb 128 {r3[f][0][0]}", r3[f][0][2]) for f in FAMS if r3[f][0]]
    for what, c in rule_cells:
        sp = spread(c)
        print(f"A1 SPREAD {what}: {sp:.2f} %")
        if sp >= 5.0:
            void.append(f"V2 spread {what} {sp:.2f} % >= 5 %")

    # ---- V4 baseline ----
    gh = cells.get(("gate", 1024, "gemmh2_q4_k"))
    if gh is None:
        void.append("V4 no in-session gemmh2_q4_k at gate nb 1024")
    else:
        dev = (gh["raw"] / T3_GEMMH2 - 1) * 100
        print(f"A1 BASELINE gemmh2_q4_k gate nb 1024: {gh['med']:.2f} TMAC/s, {dev:+.1f} % vs T3's {T3_GEMMH2}")
        if abs(dev) > 15:
            void.append(f"V4 gemmh2 gate nb 1024 {gh['med']:.2f} is {dev:+.1f} % off T3's {T3_GEMMH2} (bar +-15 %)")

    # ---- V5 compile: binding when this job compiled a module itself ----
    comp, cnv, mods, cub = meta["compile"], meta["compile_nv"], meta["modules"], meta["cubin"]
    comp_ok = comp is not None and comp.startswith("ok")
    here = [m for m in mods if m[0] != "Hit"]
    cand_cells = any(var_of(k) for (_, _, k) in cells)
    why = []
    if here:
        why.append(f"the probe module was compiled here {len(here)}x (cache "
                   + "/".join(sorted({m[0] for m in here})) + ", NVRTC " + "/".join(sorted({m[1] for m in here})) + ")")
    if not mods and cand_cells:
        why.append("no `A1 MODULE probe-fast` line: how the probe module was loaded is unknown")
    if cub is None:
        why.append("no `A1 CUBIN` line: whether this job compiled kernels is unknown")
    elif cub > 0:
        why.append(f"{cub} new kernel-cache entries during the run")
    comp_txt = (f"compile check: {comp}" if comp else "no `A1 STAGE compile` line")
    if why and not comp_ok:
        void.append("V5 kernels were (or may have been) compiled in this job (" + "; ".join(why)
                    + ") and the compile check didn't pass (" + comp_txt + ")")
    elif here and comp_ok and any(m[1] != cnv for m in here):
        void.append("V5 the probe module was compiled here with NVRTC " + "/".join(sorted({m[1] for m in here}))
                    + f" but the compile check ran NVRTC {cnv}: the check doesn't cover the code that ran")
    elif comp_ok:
        print(f"A1 V5 compile check ok (NVRTC {cnv}); compiled in this job: "
              + ("; ".join(why) if why else "nothing (every module from the kit's cubins / the cache)"))
    else:
        print(f"A1 V5 {comp_txt}, but every module came from the kit's cubins (probe module cache Hit x{len(mods)}, "
              f"no new kernel-cache entries): recorded, not binding")

    # ---- the n128 KILL licence ----
    def wide_clean():
        why = []
        gw = cells.get(("gate", 1024, FP8_WIDE))
        if not counts(gw):
            why.append("gate nb 1024 " + ("missing" if gw is None else f"tol {gw['tol']}"))
        elif spread(gw) >= 5.0:
            why.append(f"gate nb 1024 spread {spread(gw):.2f} % >= 5 %")
        if v3[FP8_WIDE] != "ok":
            why.append(f"scaling check {v3[FP8_WIDE]}")
        bad = [f"{s} nb {nb} {k}" for (s, nb, k), c in cells.items() if var_of(k) == FP8_WIDE and c["tol"] != "PASS"]
        if bad:
            why.append(f"{len(bad)} cell(s) not tol PASS ({', '.join(bad[:3])}{', ...' if len(bad) > 3 else ''})")
        for part in ("tiny", "ragged"):
            o = meta["stages"].get(("plain", FP8_WIDE, part))
            if o != "ok":
                why.append(f"{part} stage {o or 'missing'}")
        o = meta["stages"].get(("sanitizer", FP8_WIDE, "tiny"))
        if o is not None and not (o == "ok" or o.startswith("skipped")):
            why.append(f"memcheck tiny {o}")
        return why

    wide_why = wide_clean()
    print(f"A1 WIDE-N {FP8_WIDE}: " + ("CLEAN (gate nb 1024 counts, spread < 5 %, scaling ok, every cell tol PASS, "
                                       "tiny and ragged ok): a rule-1 KILL may close Track A" if not wide_why else
                                       "NOT CLEAN (" + "; ".join(wide_why) + "): a rule-1 KILL is n64 only"))
    valid = not void
    print("A1 VALIDITY: " + ("VALID" if valid else "VOID (" + "; ".join(void) + ")"))
    tag = "" if valid else " [VOID run: no verdict]"

    # ---- the rules ----
    def unread_txt(xs):
        return ", ".join(f"{lab} {c['med']:.2f} (scaling {v3.get(v)})" for lab, v, c in xs)

    def rule_line(n, what, fam, bar, ok_txt, bad_txt):
        best, hi, unread, bad = g[fam]
        if best and best[2]["raw"] >= bar:
            c = best[2]
            print(f"A1 RULE {n} {what} (best readable tile: {best[0]}): median {c['med']:.2f} TMAC/s (raw "
                  f"{c['raw']:.4f}, tol PASS) vs {bar:.0f}: {ok_txt}{tag}")
            return True
        if hi:
            print(f"A1 RULE {n} {what}: UNRESOLVED ({unread_txt(hi)} >= {bar:.0f} but not readable: no PASS on it, "
                  f"and no KILL past it" + (f"; best readable {best[0]} {best[2]['med']:.2f}" if best else "")
                  + f"){tag}")
            return None
        if best:
            c = best[2]
            print(f"A1 RULE {n} {what} (best readable tile: {best[0]}): median {c['med']:.2f} TMAC/s (raw "
                  f"{c['raw']:.4f}, tol PASS) vs {bar:.0f}: {bad_txt}{tag}")
            return False
        extra = []
        if unread:
            extra.append(f"counting but not readable: {unread_txt(unread)}")
        if bad:
            extra.append("not counting: " + ", ".join(f"{lab} {c['med']:.2f} tol {c['tol']}" for lab, _, c in bad))
        print(f"A1 RULE {n} {what}: NOT MEASURED" + (f" ({'; '.join(extra)})" if extra else "") + tag)
        return None

    kill1 = ("KILL (FP8 cannot carry M1': Track A closed as \"not with this fleet\"; Track C gets the budget)"
             if not wide_why else
             f"KILL AT n64 ONLY (FP8 < 350 on the readable tiles; {FP8_WIDE} not clean: Track A stays open pending a "
             f"wider-N result)")
    p1 = rule_line(1, "FP8 gate 3584x18944 nb 1024", "fp8", BARS["fp8"], "PASS (FP8 can carry M1'; WP-A2 next)", kill1)
    p2 = rule_line(2, "W4A16 gate 3584x18944 nb 1024", "w4a16", BARS["w4a16"], "PASS (Marlin-parity class: keep (b))",
                   "DROP W4A16 (not Marlin-parity: drop (b))")
    parts = [f"{f} {r3[f][0][2]['med']:.2f} ({r3[f][0][0]})" if r3[f][0] else f"{f} not measured" for f in FAMS]
    head = "A1 RULE 3 down 18944x3584 nb 128 with split-K: " + ", ".join(parts)
    bests = {f: r3[f][0] for f in FAMS if r3[f][0]}
    his = [x for f in FAMS for x in r3[f][1]]
    if any(b[2]["raw"] >= BARS["down"] for b in bests.values()):
        p3 = True
        print(f"{head} vs 180: PASS (at least one candidate >= 180){tag}")
    elif his:
        p3 = None
        print(f"{head} vs 180: UNRESOLVED ({unread_txt(his)} >= 180 but not readable){tag}")
    elif len(bests) == 2:
        p3 = False
        print(f"{head} vs 180: CLASSIC M1 OUT (both < 180: the decode step can't reach <= 5 ms on classic; record, "
              f"continue mix-only if FP8 passed){tag}")
    elif bests:
        p3 = None
        print(f"{head} vs 180: INCOMPLETE (one candidate < 180, the other not measured: the rule needs both){tag}")
    else:
        p3 = None
        print(f"{head}: NOT MEASURED{tag}")

    # ---- the verdict ----
    if not valid:
        print("A1 VERDICT: VOID (" + "; ".join(void) + "): no verdict; a rerun needs the lead's ruling")
        return 0
    closed = p1 is False and not wide_why
    missing = [n for n, p in [("rule 1", p1), ("rule 2", p2), ("rule 3", p3)] if p is None]
    v = []
    if p1:
        v.append("Track A OPEN (FP8 passed: WP-A2 next)")
    elif closed:
        v.append("Track A CLOSED (FP8 < 350 at n64 and n128)")
    elif p1 is False:
        v.append(f"Track A OPEN PENDING WIDER-N (FP8 < 350 at n64; {FP8_WIDE} not clean; {WIDER_N_NEXT})")
    else:
        v.append("Track A UNDECIDED (rule 1 not measured or unresolved)")
    moot = " (recorded; moot: Track A closed)" if closed else ""
    v.append(("W4A16 KEEP" if p2 else "W4A16 DROP" if p2 is False else "W4A16 UNDECIDED") + moot)
    v.append(("classic M1 IN" if p3 else "classic M1 OUT" if p3 is False else "classic M1 UNDECIDED") + moot)
    print("A1 VERDICT: " + "; ".join(v) + (f" (not decided: {', '.join(missing)})" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
