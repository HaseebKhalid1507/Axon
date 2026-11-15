#!/usr/bin/env python3
"""Constrained tool calls on a real model, end to end (stdlib only): #259's PR B gates.

    scripts/constrain_check.py run --axon BIN --model M.gguf --bodies DIR --out OUT.json
                                   [--tasks harness_accept.py] [--stress N] [--pins PINS.json [--record]]
                                   [--expect-constraint on|off|any] [--port P] [-- extra serve flags]
    scripts/constrain_check.py diff A.json B.json     the cells' digests of two runs: equal or not
    scripts/constrain_check.py replay --axon BIN --model M.gguf --replays DIR --out OUT.json [--seeds N]
                                      [--expect-constraint on|off|any]
                                      [--port P] [-- extra serve flags]

Starts `axon serve M` with the acceptance test's flags (scripts/harness_accept.py on feat/eval: --device cuda,
--ctx 32768, --kv q8, --parallel 4, Qwen2.5's card sampler: temp 0.7, top-k 20, top-p 0.8, min-p 0, repeat
penalty off), then:

  cells   8 fixed requests, each the first turn of a coding agent as it sends it (DIR holds the bodies
          `harness_accept.py capture` records: codex-*.json on /v1/responses, opencode-*.json and pi-*.json on
          /v1/chat/completions), greedy and seeded, plus codex's tools with #259's shell-quoting ask and with
          tool_choice required. Each runs alone, then all 8 at once (the batch): the reply (content, reasoning,
          calls' names and arguments, finish, token count) must be the same. Its md5 is the cell's digest.
  pins    --pins: the digests per server fingerprint (`/identity`). --record writes this run's; else a run whose
          fingerprint has pins must match them (another GPU or build has another fingerprint: not judged).
  stress  --stress N: N sampled replies (seeds 1..N) with codex's tools, the first prompt of each of the acceptance
          test's tasks in turn (from --tasks, harness_accept.py's TASKS), 4 at a time.

Every call of every reply is judged: its name one of the request's tools and its arguments a JSON object (a call
the token limit cut is reported apart, `cut`), and no `<tool_call>` left in the content. Exit 0 if every call is
valid and the cells agree (and match their pins), 1 otherwise. The JSON (--out) has every reply and /metrics'
constraint counters (before and after).

replay (shady's review of #274: the stall rule on the real model): DIR's cells (`{"kind", "what", "model",
"route", "body"}`; the ones whose "model" isn't M's file name are skipped), each greedy and with seeds 1..N, one at a
time (each reply's stalls are /metrics' delta), on the same server flags:
  runaway   a request after which an acceptance run's call ran on: the reply must end before its token limit, with
            every call valid (a stall closes it, or the call ends by itself);
  legit     a request for data that repeats one value (an empty board, a table of zeros): no stall, every call
            valid;
  known-fp  one the rule is known to cut (reported, not judged).

Integrity (11-15: a gate passed on 404s). The captured bodies name the model by the harness's alias
("qwen2.5-7b"), which a server started on the .gguf doesn't answer (404 model_not_found): every cell, replay and
batched copy got the same error, the digests of two 404s matched, and "kill switch == main" passed on nothing. So:
  - a body's "model" is set to the id the server reports (/v1/models), as a harness's --alias would make it;
  - every request's HTTP status is counted, in every arm (the kill switch and main too): any non-200 is
    `INTEGRITY FAIL` and exit 2, whatever the verdict says;
  - --expect-constraint on: the constraint must have run (axon_constraint_tokens_total moved); off: it must not
    have (the kill switch, main). replay defaults to on;
  - `diff` refuses runs with a non-200 cell, an integrity failure or different cells (`CELLS-INVALID`, exit 2)
    instead of calling two errors equal.
"""

import argparse
import hashlib
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

SAMPLER = ["--temp", "0.7", "--top-k", "20", "--top-p", "0.8", "--min-p", "0", "--repeat-penalty", "1.0"]
ASK_259 = ("In ledger/cli.py replace raise KeyError(f'No such account: {name}') with raise "
           "ValueError(\"unknown \\\"account\\\" {name}\"), then append the line "
           "{\"order_id\": \"o1\", \"total\": 3} to orders.jsonl with echo and print \"done\".")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def req(port, method, path, body=None, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, data, {"Content-Type": "application/json"} if data else {})
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, out


def load(d, prefix):
    for f in sorted(os.listdir(d)):
        if f.startswith(prefix) and f.endswith(".json"):
            return json.load(open(os.path.join(d, f)))
    sys.exit("constrain_check: no %s*.json in %s" % (prefix, d))


def chat_tools(resp_tools):
    """Responses' function tools as chat completions' (others dropped, as axon drops them)."""
    return [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
            for t in resp_tools if t.get("type") == "function"]


def offered(body):
    names = set()
    for t in body.get("tools") or []:
        if t.get("type") == "function":
            names.add(t.get("name") or (t.get("function") or {}).get("name"))
    return names


def served_id(port):
    """The model id the server answers to (/v1/models' first)."""
    st, raw = req(port, "GET", "/v1/models")
    if st != 200:
        sys.exit("constrain_check: /v1/models answered %d: %s" % (st, raw[:300]))
    return json.loads(raw)["data"][0]["id"]


def aim(body, mid):
    """BODY for this server: the harnesses name the model by their alias ("qwen2.5-7b" in the captured bodies), which
    a server started on the file refuses (404 model_not_found); the harness's own server has --alias for it."""
    b = dict(body)
    if "model" in b:
        b["model"] = mid
    return b


class Http:
    """Every request's status (thread-safe), for the integrity check."""

    def __init__(self):
        self.lock, self.n, self.bad = threading.Lock(), 0, []

    def see(self, what, st, raw):
        with self.lock:
            self.n += 1
            if st != 200:
                self.bad.append([what, st, raw[:200].decode(errors="replace")])


def integrity(res, http, expect):
    """The run's transport and switch checks, judged in every arm: (its problems, the lines to print)."""
    probs = []
    if http.n == 0:
        probs.append("no request made")
    if http.bad:
        by = {}
        for _w, st, _b in http.bad:
            by[st] = by.get(st, 0) + 1
        probs.append("%d of %d requests not 200 %s (first: %s: %s)" % (len(http.bad), http.n, by, http.bad[0][0],
                                                                      http.bad[0][2]))
    k = "axon_constraint_tokens_total"
    tok = res.get("metrics_after", {}).get(k, 0) - res.get("metrics_before", {}).get(k, 0)
    if expect == "on" and tok <= 0:
        probs.append("the constraint was to run, but %s +0" % k)
    if expect == "off" and tok != 0:
        probs.append("the constraint was to be off, but %s +%d" % (k, tok))
    res["http"] = {"requests": http.n, "non200": http.bad}
    res["constraint_tokens"] = tok
    res["integrity"] = probs
    lines = ["HTTP %s" % ("all 200 (%d requests)" % http.n if not http.bad else
                          "NON-200: %d of %d" % (len(http.bad), http.n)),
             "CONSTRAINT-TOKENS %d (expected %s)" % (tok, expect),
             "INTEGRITY %s" % ("PASS" if not probs else "FAIL: " + "; ".join(probs))]
    return probs, lines


def cells(bodies):
    codex, oc, pi = load(bodies, "codex-"), load(bodies, "opencode-"), load(bodies, "pi-")
    for b in (codex, oc, pi):
        b.pop("stream", None)
        b.pop("stream_options", None)
    codex_chat = dict(messages=[{"role": "system", "content": codex["instructions"]}], tools=chat_tools(codex["tools"]))

    def chat_ask(ask, **kw):
        b = json.loads(json.dumps(codex_chat))
        b["messages"].append({"role": "user", "content": ask})
        b.update(kw)
        return b

    first_codex_ask = next(c["text"] for it in reversed(codex["input"]) if it.get("role") == "user"
                           for c in it["content"] if c.get("type") == "input_text")
    out = [
        ("codex-greedy", "/v1/responses", dict(codex, temperature=0, max_output_tokens=512)),
        ("codex-seed1", "/v1/responses", dict(codex, seed=1, max_output_tokens=512)),
        ("opencode-greedy", "/v1/chat/completions", dict(oc, temperature=0, max_completion_tokens=512)),
        ("opencode-seed2", "/v1/chat/completions", dict(oc, seed=2, max_completion_tokens=512)),
        ("pi-greedy", "/v1/chat/completions", dict(pi, temperature=0, max_completion_tokens=512)),
        ("pi-seed3", "/v1/chat/completions", dict(pi, seed=3, max_completion_tokens=512)),
        ("codex-259-seed4", "/v1/chat/completions", chat_ask(ASK_259, seed=4, max_tokens=512)),
        ("codex-required-seed5", "/v1/chat/completions",
         chat_ask(first_codex_ask, seed=5, max_tokens=512, tool_choice="required")),
    ]
    return out, codex_chat


def reply_of(route, st, raw, names):
    """(the reply as compared, its calls judged)."""
    if st != 200:
        return {"status": st, "body": raw[:300].decode(errors="replace")}, [("http", st)]
    v = json.loads(raw)
    calls, content, reasoning, finish, ntok = [], "", None, None, None
    if route == "/v1/responses":
        for it in v.get("output", []):
            if it.get("type") == "function_call":
                calls.append((it.get("name"), it.get("arguments"), it.get("status")))
            elif it.get("type") == "message":
                content += "".join(c.get("text", "") for c in it.get("content", []))
            elif it.get("type") == "reasoning":
                reasoning = json.dumps(it.get("content") or it.get("summary"))
        finish = v.get("status")
        ntok = (v.get("usage") or {}).get("output_tokens")
        cut = finish == "incomplete"
    else:
        ch = v["choices"][0]
        m = ch["message"]
        for c in m.get("tool_calls") or []:
            calls.append((c["function"]["name"], c["function"]["arguments"], None))
        content, reasoning, finish = m.get("content") or "", m.get("reasoning_content"), ch.get("finish_reason")
        ntok = (v.get("usage") or {}).get("completion_tokens")
        cut = finish == "length"
    judged = []
    for i, (name, args, _st) in enumerate(calls):
        last_cut = cut and i + 1 == len(calls)
        try:
            ok_args = isinstance(json.loads(args), dict)
        except (ValueError, TypeError):
            ok_args = False
        if name not in names:
            judged.append(("unoffered", name))
        elif not ok_args:
            judged.append(("cut" if last_cut else "badjson", name))
        else:
            judged.append(("ok", name))
    if "<tool_call>" in content and not cut:
        judged.append(("text", None))
    rep = {"content": content, "reasoning": reasoning, "calls": [[n, a] for n, a, _ in calls], "finish": finish,
           "tokens": ntok, "fp": v.get("system_fingerprint")}
    return rep, judged


def digest(rep):
    r = {k: rep.get(k) for k in ("content", "reasoning", "calls", "finish", "tokens", "status", "body")}
    return hashlib.md5(json.dumps(r, sort_keys=True).encode()).hexdigest()


def metrics(port):
    st, raw = req(port, "GET", "/metrics")
    out = {}
    for ln in raw.decode().splitlines():
        if ln.startswith(("axon_constraint", "axon_tool_calls")) and " " in ln:
            k, v = ln.rsplit(" ", 1)
            out[k] = float(v)
    return out


def start(a):
    """`axon serve` with the acceptance test's flags, up: (process, port, its log's path, the result's start)."""
    port = a.port or free_port()
    argv = [a.axon, "serve", a.model, "--device", "cuda", "--host", "127.0.0.1", "--port", str(port), "--ctx",
            "32768", "--kv", "q8", "--parallel", "4"] + SAMPLER + a.serve
    logf = a.out + ".server.log"
    lf = open(logf, "w")
    proc = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    res = {"argv": argv, "env": {k: v for k, v in os.environ.items() if k.startswith("AXON_")}}
    for _ in range(1200):
        if proc.poll() is not None:
            sys.exit("constrain_check: the server exited %s: %s" % (proc.poll(), open(logf).read()[-800:]))
        try:
            if req(port, "GET", "/health", timeout=5)[0] == 200:
                break
        except OSError:
            pass
        time.sleep(0.5)
    return proc, port, logf, res


def stop(proc):
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()


def run(a):
    proc, port, logf, res = start(a)
    bad, http = [], Http()
    try:
        ident = json.loads(req(port, "GET", "/identity")[1])
        res["identity"] = ident
        res["banner"] = [ln.strip() for ln in open(logf, errors="replace") if "constrained tool calls" in ln]
        res["metrics_before"] = metrics(port)
        cs, codex_chat = cells(a.bodies)
        res["model_id"] = mid = served_id(port)
        cs = [(name, route, aim(body, mid)) for name, route, body in cs]
        alone, together = {}, {}
        res["cells"] = {}
        for name, route, body in cs:
            st, raw = req(port, "POST", route, body)
            http.see(name, st, raw)
            rep, judged = reply_of(route, st, raw, offered(body))
            alone[name] = digest(rep)
            res["cells"][name] = dict(digest=alone[name], judged=judged, reply=rep)
            bad += ["%s: %s %s" % (name, j, n) for j, n in judged if j not in ("ok", "cut")]
        th, outs = [], {}

        def one(name, route, body):
            st, raw = req(port, "POST", route, body)
            http.see(name + " (batched)", st, raw)
            outs[name] = digest(reply_of(route, st, raw, offered(body))[0])
        for name, route, body in cs:
            t = threading.Thread(target=one, args=(name, route, body))
            t.start()
            th.append(t)
        for t in th:
            t.join()
        together = outs
        res["batched_equal"] = {n: together.get(n) == alone[n] for n in alone}
        bad += ["%s: batched %s != alone %s" % (n, together.get(n), alone[n]) for n in alone
                if together.get(n) != alone[n]]
        fp = ident["system_fingerprint"]
        if a.pins:
            pins = json.load(open(a.pins)) if os.path.exists(a.pins) else {}
            if a.record:
                pins[fp] = {"model": os.path.basename(a.model), "cells": alone,
                            "recorded": time.strftime("%Y-%m-%d"), "numerics": ident.get("numerics")}
                json.dump(pins, open(a.pins, "w"), indent=1, sort_keys=True)
                res["pins"] = "recorded"
            elif fp in pins:
                diff = [n for n in alone if pins[fp]["cells"].get(n) != alone[n]]
                res["pins"] = "match" if not diff else "DIFF " + ",".join(diff)
                bad += ["pin %s" % n for n in diff]
            else:
                res["pins"] = "no pins for %s" % fp
        if a.stress:
            tasks = []
            if a.tasks:
                sys.path.insert(0, os.path.dirname(os.path.abspath(a.tasks)))
                ha = __import__(os.path.basename(a.tasks)[:-3])
                tasks = [t["prompts"][0] for t in ha.TASKS]
            tasks = tasks or [ASK_259]
            names = {t["function"]["name"] for t in codex_chat["tools"]}
            counts, lock, reps = {}, threading.Lock(), []
            seeds = list(range(1, a.stress + 1))

            def worker():
                while True:
                    with lock:
                        if not seeds:
                            return
                        s = seeds.pop(0)
                    b = json.loads(json.dumps(codex_chat))
                    b["messages"].append({"role": "user", "content": tasks[(s - 1) % len(tasks)]})
                    b.update(seed=s, max_tokens=1024)
                    st, raw = req(port, "POST", "/v1/chat/completions", b)
                    http.see("stress seed %d" % s, st, raw)
                    rep, judged = reply_of("/v1/chat/completions", st, raw, names)
                    with lock:
                        reps.append(dict(seed=s, judged=judged, finish=rep.get("finish"), calls=rep.get("calls")))
                        for j, _n in judged:
                            counts[j] = counts.get(j, 0) + 1
                        counts["replies"] = counts.get("replies", 0) + 1
                        counts["replies_with_call"] = counts.get("replies_with_call", 0) + bool(rep.get("calls"))
                        counts["finish_" + str(rep.get("finish"))] = counts.get("finish_" + str(rep.get("finish")), 0) + 1
            ws = [threading.Thread(target=worker) for _ in range(4)]
            for w in ws:
                w.start()
            for w in ws:
                w.join()
            res["stress"] = dict(counts=counts, replies=sorted(reps, key=lambda r: r["seed"]))
            bad += ["stress: %d %s" % (counts[k], k) for k in ("badjson", "unoffered", "text", "http") if counts.get(k)]
        res["metrics_after"] = metrics(port)
    finally:
        stop(proc)
    probs, ilines = integrity(res, http, a.expect_constraint)
    bad += ["integrity: " + p for p in probs]
    res["bad"] = bad
    json.dump(res, open(a.out, "w"), indent=1)
    print("fingerprint %s numerics %s model %s banner %s" % (res["identity"]["system_fingerprint"],
                                                            res["identity"].get("numerics"), res.get("model_id"),
                                                            res["banner"]))
    for n, c in res["cells"].items():
        print("CELL %-22s %s %s %s" % (n, c["digest"], c["reply"].get("finish"),
                                       ",".join("%s:%s" % j for j in c["judged"]) or "no call"))
    print("batched == alone: %s; pins: %s" % (all(res["batched_equal"].values()), res.get("pins")))
    if "stress" in res:
        print("STRESS %s" % json.dumps(res["stress"]["counts"], sort_keys=True))
    print("METRICS %s" % json.dumps({k: res["metrics_after"].get(k, 0) - res["metrics_before"].get(k, 0)
                                     for k in res["metrics_after"]}, sort_keys=True))
    print("\n".join(ilines))
    print("VERDICT %s" % ("PASS" if not bad else "FAIL: " + "; ".join(bad[:20])))
    return 2 if probs else 0 if not bad else 1


def replay(a):
    m = os.path.basename(a.model)[:-len(".gguf")] if a.model.endswith(".gguf") else os.path.basename(a.model)
    cs = []
    for f in sorted(os.listdir(a.replays)):
        if f.endswith(".json"):
            c = json.load(open(os.path.join(a.replays, f)))
            if c.get("model") in (None, m):
                cs.append((f[:-5], c))
    if not cs:
        sys.exit("constrain_check: no replay in %s for %s" % (a.replays, m))
    proc, port, logf, res = start(a)
    bad, rows, http = [], [], Http()
    try:
        res["identity"] = json.loads(req(port, "GET", "/identity")[1])
        res["model_id"] = mid = served_id(port)
        res["metrics_before"] = metrics(port)
        for name, c in cs:
            for mode in ["greedy"] + ["seed%d" % s for s in range(1, a.seeds + 1)]:
                body = aim(c["body"], mid)
                if mode == "greedy":
                    body["temperature"] = 0
                else:
                    body["seed"] = int(mode[4:])
                cap = body.get("max_output_tokens") or body.get("max_completion_tokens") or body.get("max_tokens")
                if not cap:
                    cap = body["max_output_tokens"] = 8192
                m0 = metrics(port)
                t0 = time.time()
                st, raw = req(port, "POST", c["route"], body, timeout=3600)
                http.see("%s %s" % (name, mode), st, raw)
                m1 = metrics(port)
                rep, judged = reply_of(c["route"], st, raw, offered(body))
                stalls = m1.get("axon_constraint_stalls_total", 0) - m0.get("axon_constraint_stalls_total", 0)
                ran_on = rep.get("finish") in ("length", "incomplete")
                badj = [j for j, _n in judged if j != "ok"]
                if c["kind"] == "runaway":
                    ok = not ran_on and not badj
                elif c["kind"] == "legit":
                    ok = not stalls and not badj
                else:
                    ok = None
                row = dict(cell=name, kind=c["kind"], what=c["what"], mode=mode, ok=ok, stalls=stalls,
                           tokens=rep.get("tokens"), cap=cap, finish=rep.get("finish"), judged=judged,
                           args_len=[len(x[1] or "") for x in rep.get("calls") or []], s=round(time.time() - t0, 1),
                           calls=[[n, (x or "")[-160:]] for n, x in rep.get("calls") or []])
                if ok is not True:  # a failure or a known false positive: the whole reply, to see why
                    row["reply"] = {"content": rep.get("content"), "calls": rep.get("calls")}
                rows.append(row)
                print("REPLAY %-34s %-8s %-6s %-4s stalls %d tokens %s/%s finish %s calls %s" % (
                    name, c["kind"], mode, {True: "ok", False: "FAIL", None: "-"}[ok], stalls, rep.get("tokens"),
                    cap, rep.get("finish"), ",".join("%s:%s" % j for j in judged) or "none"), flush=True)
                if ok is False:
                    bad.append("%s %s: %s" % (name, mode, "ran on" if ran_on else (badj or "stalled")))
        res["metrics_after"] = metrics(port)
    finally:
        stop(proc)
    probs, ilines = integrity(res, http, a.expect_constraint)
    bad += ["integrity: " + p for p in probs]
    res["rows"], res["bad"] = rows, bad
    res["stall_lines"] = [ln.strip() for ln in open(logf, errors="replace") if "stalled after" in ln]
    json.dump(res, open(a.out, "w"), indent=1)
    print("STALL-LINES %d" % len(res["stall_lines"]))
    print("METRICS %s" % json.dumps({k: res["metrics_after"].get(k, 0) - res["metrics_before"].get(k, 0)
                                     for k in res["metrics_after"]}, sort_keys=True))
    print("\n".join(ilines))
    print("VERDICT %s" % ("PASS" if not bad else "FAIL: " + "; ".join(bad[:20])))
    return 2 if probs else 0 if not bad else 1


def diff(a, b):
    x, y = json.load(open(a)), json.load(open(b))
    why = []
    for f, d in ((a, x), (b, y)):
        non200 = [n for n, c in d["cells"].items() if c["reply"].get("status") not in (None, 200)]
        if non200:
            why.append("%s: cells not 200: %s" % (f, ",".join(non200)))
        if d.get("integrity", ["no integrity record (a run before 11-15's check)"]):
            why.append("%s: integrity %s" % (f, d.get("integrity", "missing")))
    if set(x["cells"]) != set(y["cells"]):
        why.append("different cells: %s vs %s" % (sorted(x["cells"]), sorted(y["cells"])))
    if why:
        print("CELLS-INVALID: " + "; ".join(why))
        return 2
    same = [n for n in x["cells"] if y["cells"].get(n, {}).get("digest") == x["cells"][n]["digest"]]
    differ = [n for n in x["cells"] if n not in same]
    print("same %d: %s" % (len(same), ",".join(same)))
    print("DIFFER %d: %s" % (len(differ), ",".join(differ)))
    print("CELLS-%s" % ("EQUAL" if not differ else "DIFFER"))
    return 0 if not differ else 1


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--axon", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--bodies", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tasks")
    r.add_argument("--stress", type=int, default=0)
    r.add_argument("--pins")
    r.add_argument("--record", action="store_true")
    r.add_argument("--port", type=int, default=0)
    r.add_argument("--expect-constraint", choices=("on", "off", "any"), default="any")
    r.add_argument("serve", nargs="*")
    d = sp.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    rp = sp.add_parser("replay")
    rp.add_argument("--axon", required=True)
    rp.add_argument("--model", required=True)
    rp.add_argument("--replays", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--seeds", type=int, default=2)
    rp.add_argument("--port", type=int, default=0)
    rp.add_argument("--expect-constraint", choices=("on", "off", "any"), default="on")
    rp.add_argument("serve", nargs="*")
    a = p.parse_args()
    sys.exit({"run": run, "replay": replay}[a.cmd](a) if a.cmd != "diff" else diff(a.a, a.b))


if __name__ == "__main__":
    main()
