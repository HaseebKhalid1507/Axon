#!/usr/bin/env python3
"""Constrained tool calls on a real model, end to end (stdlib only): #259's PR B gates.

    scripts/constrain_check.py run --axon BIN --model M.gguf --bodies DIR --out OUT.json
                                   [--tasks harness_accept.py] [--stress N] [--pins PINS.json [--record]]
                                   [--port P] [-- extra serve flags]
    scripts/constrain_check.py diff A.json B.json     the cells' digests of two runs: equal or not

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


def run(a):
    port = a.port or free_port()
    argv = [a.axon, "serve", a.model, "--device", "cuda", "--host", "127.0.0.1", "--port", str(port), "--ctx",
            "32768", "--kv", "q8", "--parallel", "4"] + SAMPLER + a.serve
    logf = a.out + ".server.log"
    lf = open(logf, "w")
    proc = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    res = {"argv": argv, "env": {k: v for k, v in os.environ.items() if k.startswith("AXON_")}}
    bad = []
    try:
        for _ in range(1200):
            if proc.poll() is not None:
                sys.exit("constrain_check: the server exited %s: %s" % (proc.poll(), open(logf).read()[-800:]))
            try:
                if req(port, "GET", "/health", timeout=5)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        ident = json.loads(req(port, "GET", "/identity")[1])
        res["identity"] = ident
        res["banner"] = [ln.strip() for ln in open(logf, errors="replace") if "constrained tool calls" in ln]
        res["metrics_before"] = metrics(port)
        cs, codex_chat = cells(a.bodies)
        alone, together = {}, {}
        res["cells"] = {}
        for name, route, body in cs:
            st, raw = req(port, "POST", route, body)
            rep, judged = reply_of(route, st, raw, offered(body))
            alone[name] = digest(rep)
            res["cells"][name] = dict(digest=alone[name], judged=judged, reply=rep)
            bad += ["%s: %s %s" % (name, j, n) for j, n in judged if j not in ("ok", "cut")]
        th, outs = [], {}

        def one(name, route, body):
            st, raw = req(port, "POST", route, body)
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
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    res["bad"] = bad
    json.dump(res, open(a.out, "w"), indent=1)
    print("fingerprint %s numerics %s banner %s" % (res["identity"]["system_fingerprint"],
                                                   res["identity"].get("numerics"), res["banner"]))
    for n, c in res["cells"].items():
        print("CELL %-22s %s %s %s" % (n, c["digest"], c["reply"].get("finish"),
                                       ",".join("%s:%s" % j for j in c["judged"]) or "no call"))
    print("batched == alone: %s; pins: %s" % (all(res["batched_equal"].values()), res.get("pins")))
    if "stress" in res:
        print("STRESS %s" % json.dumps(res["stress"]["counts"], sort_keys=True))
    print("METRICS %s" % json.dumps({k: res["metrics_after"].get(k, 0) - res["metrics_before"].get(k, 0)
                                     for k in res["metrics_after"]}, sort_keys=True))
    print("VERDICT %s" % ("PASS" if not bad else "FAIL: " + "; ".join(bad[:20])))
    return 0 if not bad else 1


def diff(a, b):
    x, y = json.load(open(a)), json.load(open(b))
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
    r.add_argument("serve", nargs="*")
    d = sp.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = p.parse_args()
    sys.exit(run(a) if a.cmd == "run" else diff(a.a, a.b))


if __name__ == "__main__":
    main()
