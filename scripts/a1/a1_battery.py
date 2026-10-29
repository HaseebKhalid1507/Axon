#!/usr/bin/env python3
"""a1_battery.py [VERDICT]: a1_verdict.py on synthetic cuda_a1_session logs, each with its expected lines.

    python3 scripts/a1/a1_battery.py            # uses scripts/a1/a1_verdict.py next to this file
    python3 scripts/a1/a1_battery.py -v         # also print every case's full verdict output

No GPU. The bella rehearsal only reaches the baseline paths of the verdict (sm_120: no candidate cell), so the
candidate paths (validity V1-V4, the rule lines, the fallback merge) are proven here, before any H100 data exists.
The logs are generated in the exact line formats tests.rs prints: the FORMAT case's cell line is a1_cell_line's
output verbatim, and the Rust test a1_cell_line_pinned asserts this file contains it.

Scaling model (the healthy cases): the probe kernels are persistent, so a cell's time scales with waves =
ceil(tiles / 132). T3 (job 1026-222543-3f73, this skeleton, 128 x 64 tiles) measured t(gate nb 1024) /
t(q/wo nb 128) = 15.5 against a waves ratio of 17.9, i.e. 0.87 of it (the single-wave cell carries fixed fill
and launch time); the healthy cases use 0.87 x the waves ratio for every variant (7.8 for the 9-wave tiles:
shady's review case, which the pre-amendment V3 bar of 10 voided).
Exit 0 if every case printed what it must (and nothing it mustn't), else 1.
"""
import math
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VERDICT = os.path.join(HERE, "a1_verdict.py")
FAMS = {"fp8": ["fp8_128x64", "fp8_128x128", "fp8n128_128x128"], "w4a16": ["w4a16_128x64", "w4a16_256x64"]}
VARS = FAMS["fp8"] + FAMS["w4a16"]
TILE = {"fp8_128x64": (128, 64), "fp8_128x128": (128, 128), "fp8n128_128x128": (128, 128),
        "w4a16_128x64": (128, 64), "w4a16_256x64": (256, 64)}
CELLS = [("gate", 3584, 18944, [128, 512, 1024]), ("down", 18944, 3584, [128, 512, 1024]),
         ("q/wo", 3584, 3584, [128, 1024]), ("kv", 3584, 512, [128, 1024])]
SXM = "NVIDIA H100 80GB HBM3"
# a1_cell_line's output for fixed inputs (tests.rs a1_cell_line_pinned asserts this file has it verbatim).
FORMAT_LINE = ("A1 CELL gate 3584x18944 nb 1024 fp8n128_128x128: median 380.00 TMAC/s (min 377.50 max 383.25; "
               "9 rounds; 0.1829 ms) tol PASS rel-rms 1.23e-2 rel-max 5.00e-2 (bounds 0.1 / 0.5)")
# Healthy gate nb 1024 TMAC/s per variant (shady's case: fp8_128x128 380, w4a16_256x64 240).
HEALTHY = {"fp8_128x64": 300.0, "fp8_128x128": 380.0, "fp8n128_128x128": 420.0,
           "w4a16_128x64": 200.0, "w4a16_256x64": 240.0}
# Other cells as a fraction of the variant's gate nb 1024 rate (any plausible shape will do: only the rule
# cells, gate nb 1024 / q/wo nb 128 (V3) and down nb 128 split-K, are read by the verdict).
FRAC = {("gate", 128): 0.55, ("gate", 512): 0.92, ("down", 128): 0.40, ("down", 512): 0.85, ("down", 1024): 0.95,
        ("q/wo", 1024): 0.80, ("kv", 128): 0.08, ("kv", 1024): 0.35}
BASE = {"gemm2s_q4_k": 62.0, "gemmh2_q4_k": 116.5}


def tiles(v, rows, nb):
    r, t = TILE[v]
    return math.ceil(rows / r) * math.ceil(nb / t)


def waves(v, rows, nb, sms=132):
    return math.ceil(tiles(v, rows, nb) / sms)


def split(v, rows, cols, nb, sms=132):
    return max(2, min(sms // tiles(v, rows, nb), cols // 256))


def tol(fp8, ok=True):
    b = "0.1 / 0.5" if fp8 else "0.005 / 0.03"
    return (f" tol PASS rel-rms {'1.10e-2' if fp8 else '4.00e-4'} rel-max {'6.00e-2' if fp8 else '2.00e-3'} "
            f"(bounds {b})" if ok else f" tol FAIL rel-rms 1.00e0 rel-max 4.10e0 (bounds {b})")


def line(shape, cols, rows, nb, name, med, ms, tl="", spread=0.02):
    lo, hi = med * (1 - spread / 2), med * (1 + spread / 2)
    return (f"A1 CELL {shape} {cols}x{rows} nb {nb} {name}: median {med:.2f} TMAC/s (min {lo:.2f} max {hi:.2f}; "
            f"9 rounds; {ms:.4f} ms){tl}")


def cell(shape, cols, rows, nb, name, med, tl="", spread=0.02):
    ms = rows * cols * nb / (med * 1e12) * 1e3
    return line(shape, cols, rows, nb, name, med, ms, tl, spread)


def bench(dev="NVIDIA H100 80GB HBM3", sms=132, cand=VARS, rates=None, frac_t3=0.87, host_floor_ms=0.0,
          bad=(), shapes=None, rule3=None, base=None, spread=0.02, cc="90"):
    """One cuda_a1_bench run's lines. rates: gate nb 1024 TMAC/s per variant; bad: variants whose every cell is
    tol FAIL (a layout bug: also noisy and flat); rule3: down nb 128 split-K TMAC/s per family; host_floor_ms:
    every time is at least this (a host-bound harness)."""
    rates = dict(HEALTHY, **(rates or {}))
    base = dict(BASE, **(base or {}))
    out = [f"A1 BENCH device {dev} sm_{cc} {sms} SMs; hopper module {cc == '90'}, probe module {cc == '90'}; "
           f"candidates [{','.join(cand)}]; 9 rounds"]
    for shape, cols, rows, nbs in CELLS:
        if shapes and shape not in shapes:
            continue
        for nb in nbs:
            macs = rows * cols * nb
            for b, r in base.items():
                if b == "gemmh2_q4_k" and cc != "90":
                    continue
                f = 1.0 if (shape, nb) == ("gate", 1024) else FRAC.get((shape, nb), 0.9)
                out.append(cell(shape, cols, rows, nb, b, r * f))
            for v in cand:
                fp8 = v.startswith("fp8")
                g = rates[v]
                t_gate = 18944 * 3584 * 1024 / (g * 1e12) * 1e3
                if (shape, nb) == ("gate", 1024):
                    ms = t_gate
                elif (shape, nb) == ("q/wo", 128):
                    w = waves(v, 18944, 1024, sms) / waves(v, 3584, 128, sms)
                    ms = t_gate / (frac_t3 * w)
                else:
                    ms = macs / (g * FRAC.get((shape, nb), 0.9) * 1e12) * 1e3
                ms = max(ms, host_floor_ms)
                if v in bad:   # layout bug: flat, noisy, wrong
                    ms = max(ms, 0.25)
                med = macs / (ms * 1e-3) / 1e12
                sp = 0.30 if v in bad else spread
                out.append(line(shape, cols, rows, nb, v, med, ms, tol(fp8, v not in bad), sp))
                if nb == 128:
                    ks = ([split(v, rows, cols, nb, sms)] if shape != "gate" else []) + [0]
                    for k in ks:
                        nm = f"{v}_splitk{k}" if k else f"{v}_streamk"
                        fam = "fp8" if fp8 else "w4a16"
                        r = (rule3 or {}).get(fam, 190.0 if fp8 else 150.0) if shape == "down" else med * 1.3
                        if k == 0 and shape == "down":
                            r *= 0.97
                        mk = max(macs / (r * 1e12) * 1e3, host_floor_ms)
                        out.append(line(shape, cols, rows, nb, nm, macs / (mk * 1e-3) / 1e12, mk,
                                        tol(fp8, v not in bad), 0.30 if v in bad else spread))
    return out


def session(benches, smi="NVIDIA H100 80GB HBM3, 700.00 W, 700.00 W, 1980 MHz", t0=None, stages=True):
    """A session log: stage lines around each bench run (unix windows), then the job's A1 SMI line."""
    t0 = t0 or time.time() - 3600
    out = ["A1 STAGE compile: ok", "A1 PEAK a1_peak_fp8_ss_n64 2 WG/SM: median 610.00 TMAC/s raw (min 600.00 max 612.00; 9 rounds)",
           "A1 PEAK a1_peak_fp8_ss_n128 2 WG/SM: median 640.00 TMAC/s raw (min 630.00 max 641.00; 9 rounds)",
           "A1 PEAK a1_peak_f16_rs_n64 2 WG/SM: median 330.00 TMAC/s raw (min 329.00 max 331.00; 9 rounds)"]
    t = t0
    for i, b in enumerate(benches):
        if stages:
            out.append(f"A1 STAGE cuda_a1_bench: start at unix {t:.1f}, limit 720 s (./axon-test ...)")
        out += b
        t += 60
        if stages:
            out.append(f"A1 STAGE cuda_a1_bench: rc {0 if i == len(benches) - 1 else 101} after 60.0 s at unix {t:.1f}")
        t += 5
    out.append("A1 END ok")
    if smi is not None:
        out.append(f"A1 SMI {smi}")
    return out, t0


def trace(t0, n=400, step=0.5):
    """An nvidia-smi --format=csv trace (header + samples every 0.5 s from t0)."""
    rows = ["timestamp, clocks.sm [MHz], clocks.mem [MHz], power.draw [W], temperature.gpu, clocks_event_reasons.active"]
    for i in range(n):
        t = t0 + i * step
        ts = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"
        rows.append(f"{ts}, {1980 - (i % 7) * 15} MHz, 2619 MHz, {640 + (i % 11) * 5:.2f} W, {55 + i % 9}, 0x0000000000000000")
    return rows


def cases():
    c = []
    # FORMAT: the pinned Rust line parses (a minimal log around it).
    c.append(("FORMAT (a1_cell_line's pinned line parses)",
              session([[f"A1 BENCH device {SXM} sm_90 132 SMs; hopper module true, probe module true; "
                        f"candidates [fp8n128_128x128]; 9 rounds", FORMAT_LINE]])[0], None,
              ["A1 TOL gate nb 1024 fp8n128_128x128: PASS rel-rms 1.23e-2 rel-max 5.00e-2",
               "| 380.00 |", "A1 RULE 1 FP8 gate 3584x18944 nb 1024 (best counting tile: fp8n128_128x128): "
               "median 380.00 TMAC/s (tol PASS) vs 350: PASS"], [], 0))
    # HEALTHY: T3-like scaling (0.87 x waves) at shady's numbers: VALID (the old bar of 10 voided this).
    log, t0 = session([bench()])
    c.append(("HEALTHY (T3-like scaling, shady's 380 / 240 case)", log, trace(t0),
              [r"re:A1 SCALING fp8_128x128: t\(gate nb 1024\) / t\(q/wo nb 128\) = 7\.8\d, waves ratio 9, bar >= 4\.50: ok",
               r"re:A1 SCALING fp8n128_128x128: .* = 7\.8\d, waves ratio 9, bar >= 4\.50: ok",
               r"re:A1 SCALING w4a16_256x64: .* = 7\.8\d, waves ratio 9, bar >= 4\.50: ok",
               r"re:A1 SCALING fp8_128x64: .* = 15\.6\d, waves ratio 18, bar >= 9\.00: ok",
               r"re:A1 SCALING w4a16_128x64: .* = 15\.6\d, waves ratio 18, bar >= 9\.00: ok",
               "A1 V1 device NVIDIA H100 80GB HBM3, 132 SMs, power.limit 700 W: ok", "A1 VALIDITY: VALID",
               "A1 CLOCKS over 1 bench window(s)",
               "A1 RULE 1 FP8 gate 3584x18944 nb 1024 (best counting tile: fp8n128_128x128): median 420.00",
               "A1 RULE 2 W4A16 gate 3584x18944 nb 1024 (best counting tile: w4a16_256x64): median 240.00",
               "A1 VERDICT: Track A OPEN (FP8 passed: WP-A2 next); W4A16 KEEP; classic M1 IN"],
              ["VOID"], 0))
    # HOST-BOUND: every time floored at 0.3 ms (P1's defect class): flat, VOID by V3.
    c.append(("HOST-BOUND (every launch >= 0.3 ms: P1's defect class)",
              session([bench(host_floor_ms=0.3)])[0], None,
              ["V3 scaling fp8_128x128", "V3 scaling fp8_128x64", "A1 VALIDITY: VOID", "A1 VERDICT: VOID"],
              ["A1 VALIDITY: VALID"], 0))
    # TOL-FAIL in one family: W4A16 layout bug (all its cells tol FAIL, flat and noisy): FP8's verdict stands.
    c.append(("TOL-FAIL W4A16 (layout bug: flat, 30 % spread, tol FAIL)",
              session([bench(bad=FAMS["w4a16"])])[0], None,
              ["A1 SCALING w4a16_256x64: not checked", "A1 VALIDITY: VALID",
               "A1 RULE 2 W4A16 gate 3584x18944 nb 1024 (best counting tile: None): NOT MEASURED (w4a16_",
               "A1 RULE 1 FP8 gate 3584x18944 nb 1024 (best counting tile: fp8n128_128x128): median 420.00",
               "A1 VERDICT: Track A OPEN (FP8 passed: WP-A2 next); W4A16 UNDECIDED; classic M1 IN"],
              ["V2 spread w4a16", "V3 scaling w4a16", "A1 VALIDITY: VOID"], 0))
    # FP8 KILL with n128 counted: Track A closed; the rest moot.
    c.append(("FP8 KILL (all three FP8 tiles < 350, n128 counted)",
              session([bench(rates={"fp8_128x64": 250.0, "fp8_128x128": 300.0, "fp8n128_128x128": 330.0},
                             rule3={"fp8": 160.0, "w4a16": 150.0})])[0], None,
              ["A1 VALIDITY: VALID", "vs 350: KILL (FP8 cannot carry M1': Track A closed",
               "A1 VERDICT: Track A CLOSED (FP8 < 350 at n64 and n128); W4A16 KEEP (recorded; moot: Track A closed); "
               "classic M1 OUT (recorded; moot: Track A closed)"], [], 0))
    # FP8 n64 KILL with n128 tol FAIL: the KILL is scoped to n64, Track A stays open.
    c.append(("FP8 n64 KILL, n128 tol FAIL (KILL scoped to the n64 instruction)",
              session([bench(rates={"fp8_128x64": 250.0, "fp8_128x128": 300.0}, bad=["fp8n128_128x128"])])[0], None,
              ["A1 VALIDITY: VALID", "KILL AT n64 ONLY (FP8 on m64n64k32 < 350; fp8n128_128x128 did not count",
               "A1 VERDICT: Track A OPEN PENDING WIDER-N (FP8 < 350 at n64; fp8n128_128x128 not measured); W4A16 KEEP"],
              ["Track A CLOSED", "A1 VALIDITY: VOID"], 0))
    # FALLBACK: the main bench died after its gate cells; per-candidate fallbacks + baselines-only run.
    main = bench(shapes=["gate"])
    fb = [bench(cand=[v]) for v in VARS] + [bench(cand=[])]
    c.append(("FALLBACK (main bench died after gate; per-candidate reruns; first run that has a cell wins)",
              session([main] + fb)[0], None,
              ["A1 RUNS 7 bench run(s)", "A1 VALIDITY: VALID",
               "A1 VERDICT: Track A OPEN (FP8 passed: WP-A2 next); W4A16 KEEP; classic M1 IN"], ["Traceback"], 0))
    # V1: NVL (132 SMs, 400 W), PCIe (114 SMs), a capped SXM, no SMI line with the SXM name (ok).
    c.append(("V1 H100 NVL (132 SMs, 400 W)",
              session([bench(dev="NVIDIA H100 NVL")], smi="NVIDIA H100 NVL, 400.00 W, 400.00 W, 1785 MHz")[0], None,
              ["V1 device NVIDIA H100 NVL with 132 SMs is not an H100 SXM", "A1 VERDICT: VOID"], [], 0))
    c.append(("V1 H100 PCIe (114 SMs)",
              session([bench(dev="NVIDIA H100 PCIe", sms=114)], smi="NVIDIA H100 PCIe, 350.00 W, 350.00 W, 1755 MHz")[0],
              None, ["V1 device NVIDIA H100 PCIe with 114 SMs is not an H100 SXM", "A1 VERDICT: VOID"], [], 0))
    c.append(("V1 SXM capped at 500 W",
              session([bench()], smi="NVIDIA H100 80GB HBM3, 500.00 W, 700.00 W, 1980 MHz")[0], None,
              ["V1 power.limit 500 W < 690 W", "A1 VERDICT: VOID"], [], 0))
    c.append(("V1 no SMI line, SXM5 name (accepted by name)", session([bench()], smi=None)[0], None,
              ["power.limit unreadable (name is the SXM5 part's): ok", "A1 VALIDITY: VALID"], [], 0))
    # REHEARSAL-like: bella sm_120, baselines only (no gemmh2, no candidates).
    c.append(("REHEARSAL-like (sm_120, gemm2s only)",
              session([bench(dev="NVIDIA GeForce RTX 5070", sms=48, cand=[], cc="120")], smi="NVIDIA GeForce RTX 5070, 250.00 W, 250.00 W, 3090 MHz")[0],
              None, ["A1 VALIDITY: VOID (V1", "V4 no in-session gemmh2_q4_k", "A1 RULE 1 FP8 gate 3584x18944 nb 1024 (best counting tile: None): NOT MEASURED",
                     "A1 VERDICT: VOID"], ["Traceback"], 0))
    # GARBAGE: no bench line at all (NO DATA, rc 2); a bench line with mangled cells (no crash).
    c.append(("GARBAGE (binary noise, no bench line)",
              ["\x00\x01garbage", "A1 CELL gate nb ???", "Segmentation fault (core dumped)", "A1 BENCH devic"], None,
              ["A1 VERDICT: NO DATA"], ["Traceback"], 2))
    c.append(("GARBAGE (a bench line, then truncated / mangled cells)",
              [f"A1 BENCH device {SXM} sm_90 132 SMs; hopper module true, probe module true; candidates [x]; 9 rounds",
               "A1 CELL gate 3584x18944 nb 1024 fp8_128x64: median nan TMAC/s (min nan max nan; 9 rounds; nan ms)",
               "A1 CELL gate 3584x18944 nb 1024 gemmh2_q4_k: median 116.50 TMAC/s (min 115",
               "A1 CELL down 18944x3584 nb 128 fp8_128x64_splitk2: median 0.00 TMAC/s (min 0.00 max 0.00; 9 rounds; "
               "0.0000 ms) tol PASS rel-rms 1e-2 rel-max 1e-2 (bounds 0.1 / 0.5)",
               "A1 SMI garbage"], None,
              ["A1 VALIDITY: VOID", "A1 VERDICT: VOID"], ["Traceback"], 0))
    return c


def main():
    verbose = "-v" in sys.argv[1:]
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for i, (name, log, tr, must, mustnt, rc_want) in enumerate(cases()):
            lp = os.path.join(td, f"case{i}.log")
            open(lp, "w").write("\n".join(log) + "\n")
            args = [sys.executable, VERDICT, lp]
            if tr is not None:
                tp = os.path.join(td, f"case{i}.csv")
                open(tp, "w").write("\n".join(tr) + "\n")
                args.append(tp)
            p = subprocess.run(args, capture_output=True, text=True)
            out = p.stdout + p.stderr
            def has(m):
                return re.search(m[3:], out) is not None if m.startswith("re:") else m in out
            miss = [m for m in must if not has(m)]
            extra = [m for m in mustnt if has(m)]
            ok = not miss and not extra and p.returncode == rc_want
            bad += not ok
            print(f"=== BATTERY {'PASS' if ok else 'FAIL'} {name} (rc {p.returncode}, want {rc_want})")
            keys = ("A1 V1", "A1 SCALING", "A1 SPREAD", "A1 VALIDITY", "A1 RULE", "A1 VERDICT", "A1 CLOCKS", "A1 RUNS",
                    "Traceback", "Error")
            for ln in out.splitlines():
                if verbose or ln.startswith(keys) or "Error" in ln:
                    print("    " + ln)
            for m in miss:
                print(f"    MISSING: {m}")
            for m in extra:
                print(f"    UNEXPECTED: {m}")
    print(f"A1 BATTERY {'ok' if bad == 0 else 'FAILED'}: {len(cases()) - bad}/{len(cases())} cases as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
