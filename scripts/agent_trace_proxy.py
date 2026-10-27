#!/usr/bin/env python3
"""Convert an agent_proxy.py capture into a #160 agent trace: NUMBERS ONLY (stdlib only).

    agent_trace_proxy.py CAPTURE.jsonl [--out FILE] [--anonymize-time] [--keep-raw]

Output: the schema of `agent_trace.py`'s traces (version 1, `requests[]` with t_ms, session, parent, depth, prompt,
shared[{with, tokens}], new, output, gap_prev_ms, fanout, fork), plus three optional fields the replay ignores:
`kind` ("main" / "side" / "compact"), `harness` (enum) and `lat_ms` (the server's latency as captured). Default
FILE: ~/.cache/axon-agent/traces/harness-<capture name>.json. The trace passes `agent_trace.py --check` (run on it
before writing; refused otherwise). The capture's message digests are used here only to find prefix relations
and are not written. The CAPTURE file is deleted after a successful conversion unless --keep-raw.

How structure is recovered (spec section 3), per request in arrival order:
- **Continuation:** an open session whose last request's digest list is a proper prefix of this one's. Shared =
  "session", tokens = this prompt x the prefix's share of the message bytes (ESTIMATED: apportioned by bytes).
- **Compaction:** no session extends, but one on the same connection, active within 60 s, shares a leading prefix
  (system / tools) and this list is shorter than its last. Same session, kind "compact", shared = that prefix.
- **New session sharing a pool prefix:** the longest common leading digest prefix with any earlier request
  (>= 1 message). shared = "sys". If another session was active within 60 s on another request, it's a subagent:
  depth 1, parent = the most recent active session. Bursts <= 5 s apart give `fanout`.
- **Side request:** no prefix shared with anything, no tools offered, <= 3 messages. kind "side", its own session.
- Anything else: a new top-level session.
"""
import argparse, json, os, subprocess, sys

ACTIVE_MS, BURST_MS = 60_000, 5_000


def in_git_tree(path):
    d = os.path.dirname(os.path.abspath(path)) or "."
    while not os.path.isdir(d):
        d = os.path.dirname(d)
    r = subprocess.run(["git", "-C", d, "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def share(rec, k):
    """Tokens of the first k messages: prompt_tokens apportioned by message bytes (ESTIMATED)."""
    p = rec.get("prompt_tokens") or 0
    tot = sum(rec["b"]) or 1
    return int(round(p * sum(rec["b"][:k]) / tot))


def convert(recs):
    recs = sorted((r for r in recs if r.get("status") == 200 and r.get("route")), key=lambda r: r["t_ms"])
    sess = []          # per session: dict(last=rec, conn, last_t, depth)
    out = []
    for r in recs:
        h, t = r["h"], r["t_ms"]
        kind, shared, parent, depth, sid, fork = "main", [], None, 0, None, None
        # Continuation: the longest last-list that is a proper prefix of this one.
        best = None
        for i, s in enumerate(sess):
            L = s["last"]["h"]
            if len(L) < len(h) and common(L, h) == len(L) and (best is None or len(L) > len(sess[best]["last"]["h"])):
                best = i
        if best is not None:
            sid = best
            k = len(sess[best]["last"]["h"])
            shared = [{"with": "session", "tokens": share(r, k)}]
            depth, parent = sess[best]["depth"], sess[best]["parent"]
        else:
            # Compaction on the same connection.
            for i, s in reversed(list(enumerate(sess))):
                k = common(s["last"]["h"], h)
                if s["conn"] == r["conn"] and t - s["last_t"] <= ACTIVE_MS and k >= 1 and len(h) < len(s["last"]["h"]):
                    sid, kind = i, "compact"
                    shared = [{"with": "session", "tokens": share(r, k)}]
                    depth, parent = s["depth"], s["parent"]
                    break
        if sid is None:
            k_sys = max((common(x["h"], h) for x in recs if x["t_ms"] < t), default=0)
            if k_sys >= 1:
                shared = [{"with": "sys", "tokens": share(r, k_sys)}]
                # A side request (title, summary) never spawns a subagent: not a parent candidate.
                act = [i for i, s in enumerate(sess) if t - s["last_t"] <= ACTIVE_MS and s["kind"] != "side"]
                if act:
                    parent = max(act, key=lambda i: sess[i]["last_t"])
                    depth = sess[parent]["depth"] + 1
            elif r.get("n_tools", 0) == 0 and r.get("n_msgs", 0) <= 3 and sess:
                kind = "side"
            sess.append(dict(last=r, conn=r["conn"], last_t=t, depth=depth, parent=parent, kind=kind))
            sid = len(sess) - 1
        prev_t = sess[sid]["last_t"] if sess[sid]["last"] is not r else None
        if sess[sid]["last"] is not r:
            sess[sid].update(last=r, last_t=t)
        p = r.get("prompt_tokens") or 0
        out.append(dict(t_ms=t, session=sid, parent=parent, depth=depth, prompt=p, shared=shared,
                        new=max(0, p - sum(x["tokens"] for x in shared)), output=r.get("completion_tokens") or 0,
                        gap_prev_ms=None if prev_t is None else t - prev_t, fanout=0, fork=fork, kind=kind,
                        harness=r.get("harness", "other"), lat_ms=r.get("total_ms")))
    # Fan-out: children of one parent starting within BURST_MS of each other.
    first = {}
    for x in out:
        first.setdefault(x["session"], x)
    kids = sorted((x for x in first.values() if x["depth"] >= 1), key=lambda x: x["t_ms"])
    i = 0
    while i < len(kids):
        j = i
        while j + 1 < len(kids) and kids[j + 1]["parent"] == kids[i]["parent"] and \
                kids[j + 1]["t_ms"] - kids[j]["t_ms"] <= BURST_MS:
            j += 1
        for x in kids[i:j + 1]:
            x["fanout"] = j - i + 1
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--out")
    ap.add_argument("--anonymize-time", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.capture) if l.strip()]
    reqs = convert(recs)
    if a.anonymize_time and reqs:
        t0 = reqs[0]["t_ms"]
        for x in reqs:
            x["t_ms"] -= t0
    kinds = {k: sum(1 for x in reqs if x["kind"] == k) for k in ("main", "side", "compact")}
    trace = dict(version=1, days=1, tokenizer="axon", synthetic=False,
                 summary=dict(requests=len(reqs), sessions=len({x["session"] for x in reqs}),
                              subagent_sessions=len({x["session"] for x in reqs if x["depth"] >= 1}), **kinds,
                              dropped_non200=sum(1 for r in recs if r.get("route") and r.get("status") != 200),
                              injected_usage=sum(1 for r in recs if r.get("inj")),
                              missing_usage=sum(1 for r in recs if r.get("route") and r.get("prompt_tokens") is None)),
                 requests=reqs)
    out = a.out or os.path.expanduser(
        f"~/.cache/axon-agent/traces/harness-{os.path.splitext(os.path.basename(a.capture))[0]}.json")
    if in_git_tree(out):
        sys.exit(f"agent_trace_proxy: {out} is inside a git tree: traces from real runs never go into a repo")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(trace, f)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import agent_trace
    if not agent_trace.check(tmp):
        os.remove(tmp)
        sys.exit("agent_trace_proxy: the privacy check failed: nothing written")
    os.replace(tmp, out)
    print(f"agent_trace_proxy: {out}: {json.dumps(trace['summary'])}")
    if not a.keep_raw:
        os.remove(a.capture)
        print(f"agent_trace_proxy: removed the raw capture {a.capture} (--keep-raw keeps it)")


if __name__ == "__main__":
    main()
