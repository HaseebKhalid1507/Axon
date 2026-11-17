#!/usr/bin/env python3
"""cbbar.py LOG|results.jsonl...: PR B's harness bar (#259) as the lead ruled it on Fable's review of #274 (11-16).

Per harness (codex, pi, opencode) and engine (axon-prb: the candidate's default, constrained; axon-off: the same
binary with AXON_CONSTRAIN=0, main's paths), over every task run (RESULT lines of harness_accept.py's jobs):
  replies    tool-call replies (the proxy's view: a parsed call, or a call left as text)
  malformed  replies with a call left as text or whose arguments don't parse (harness_accept.py's malf_req)
  stalls     calls the stall rule closed (axon_constraint_stalls_total, per task: axon_constraint_d; a stalled
             call ends its reply, so one a reply at most)
  headline   (malformed + stalls) / replies: a stall-closed call is well-formed by construction but cut, so it
             counts against the bar as a malformed one would
  malf-excl  malformed / replies (stalls left out)
  fallback   axon_constraint_fallback_total / axon_constraint_tokens_total over the tasks
  tasks passed, HTTP errors, crashes, timeouts, length finishes
PASS per harness: headline(axon-prb) <= headline(axon-off) AND malf-excl(axon-prb) < 1 %. Also reported (the spec's
bar, not the lead's pass rule): tasks passed vs axon-off; no new HTTP errors, crashes or length finishes.

--stalls-malformed (runs at 3c205cc or later, the lead's ruling of 11-17): a stalled call arrives cut off, its
arguments not JSON, so harness_accept.py already counts it in malformed. Then headline = malformed / replies (the
same bar: stalls aren't added twice) and malf-excl = (malformed - stalls) / replies. Runs at 78790e6 (the binary
of codex 599c, pi 96a7/3c08, opencode 6ccd/df2a): without the flag."""
import json, sys

STALLS_MALFORMED = "--stalls-malformed" in sys.argv[1:]
rows, seen = [], set()
for p in [x for x in sys.argv[1:] if x != "--stalls-malformed"]:
    for ln in open(p, errors="replace"):
        ln = ln.strip()
        if ln.startswith("RESULT "):
            ln = ln[7:]
        if not ln.startswith("{"):
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if "task" not in r or "engine" not in r:
            continue
        k = (r["model"], r["engine"], r["harness"], r["task"], r.get("round"), r.get("engine_sha"))
        if k not in seen:
            seen.add(k)
            rows.append(r)
cells = {}
for r in rows:
    cells.setdefault((r["harness"], r["engine"]), []).append(r)


def agg(rs):
    d = lambda r, k: (r.get("axon_constraint_d") or {}).get(k, 0)
    a = dict(runs=len(rs), passed=sum(bool(r["ok"]) for r in rs),
             replies=sum(r["tc_req"] + (r.get("tc_text") or 0) for r in rs),
             malformed=sum(r["malf_req"] for r in rs),
             stalls=int(sum(d(r, "axon_constraint_stalls_total") for r in rs)),
             ctok=int(sum(d(r, "axon_constraint_tokens_total") for r in rs)),
             fb=int(sum(d(r, "axon_constraint_fallback_total") for r in rs)),
             http=sum(r.get("http_err") or 0 for r in rs), crash=sum(bool(r.get("crash")) for r in rs),
             tmo=sum(bool(r.get("timed_out")) for r in rs), length=sum(r.get("length") or 0 for r in rs),
             no_delta=sum("axon_constraint_d" not in r for r in rs))
    n = a["replies"] or 1
    if STALLS_MALFORMED:
        a["headline"] = a["malformed"] / n
        a["malf_excl"] = (a["malformed"] - a["stalls"]) / n
    else:
        a["headline"] = (a["malformed"] + a["stalls"]) / n
        a["malf_excl"] = a["malformed"] / n
    a["fb_rate"] = a["fb"] / a["ctok"] if a["ctok"] else None
    return a


pct = lambda x: "-" if x is None else "%.2f %%" % (100 * x)
print("%-9s %-9s %6s %6s %5s %5s %6s %10s %10s %10s %6s %4s %5s %4s %6s" % (
    "harness", "engine", "pass", "reply", "malf", "stall", "", "headline", "malf-excl", "fallback", "http", "crash",
    "tmo", "len", "nodelta"))
verdicts = []
for h in ("codex", "pi", "opencode"):
    a = {e: agg(cells.get((h, e), [])) for e in ("axon-prb", "axon-off")}
    for e in ("axon-prb", "axon-off"):
        x = a[e]
        print("%-9s %-9s %3d/%-2d %6d %5d %5d %6s %10s %10s %10s %6d %4d %5d %4d %6d" % (
            h, e, x["passed"], x["runs"], x["replies"], x["malformed"], x["stalls"], "", pct(x["headline"]),
            pct(x["malf_excl"]), pct(x["fb_rate"]) + ("" if x["fb_rate"] is None else " of %d" % x["ctok"]),
            x["http"], x["crash"], x["tmo"], x["length"], x["no_delta"]))
    p, o = a["axon-prb"], a["axon-off"]
    if not p["runs"] or not o["runs"]:
        verdicts.append("%s: MISSING (prb %d runs, off %d)" % (h, p["runs"], o["runs"]))
        continue
    ok1, ok2 = p["headline"] <= o["headline"], p["malf_excl"] < 0.01
    also = []
    if p["passed"] < o["passed"]:
        also.append("tasks passed %d < off's %d" % (p["passed"], o["passed"]))
    for k in ("http", "crash", "length"):
        if p[k] > o[k]:
            also.append("%s %d > off's %d" % (k, p[k], o[k]))
    verdicts.append("%s: %s (headline %s vs off %s: %s; malformed excl. stalls %s < 1 %%: %s; stalls %d; fallback "
                    "%s)%s" % (h, "PASS" if ok1 and ok2 else "FAIL", pct(p["headline"]), pct(o["headline"]),
                               "ok" if ok1 else "WORSE", pct(p["malf_excl"]), "ok" if ok2 else "NO", p["stalls"],
                               pct(p["fb_rate"]), ("; also: " + ", ".join(also)) if also else ""))
print("\n".join(verdicts))
print("BAR %s" % ("PASS" if all(": PASS" in v for v in verdicts) else "FAIL"))
