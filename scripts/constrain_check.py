#!/usr/bin/env python3
"""Constrained tool calls on a real model, end to end (stdlib only): #259's PR B gates.

    scripts/constrain_check.py run --axon BIN --model M.gguf --bodies DIR --out OUT.json
                                   [--tasks harness_accept.py] [--stress N] [--pins PINS.json [--record]]
                                   [--expect-constraint on|off|any] [--port P] [-- extra serve flags]
    scripts/constrain_check.py diff A.json B.json     the cells' digests of two runs: equal or not
    scripts/constrain_check.py replay --axon BIN --model M.gguf --replays DIR --out OUT.json [--seeds N]
                                      [--expect-constraint on|off|any] [--known-miss CELL:MODE,...]
                                      [--port P] [-- extra serve flags]
    scripts/constrain_check.py cost --axon BIN --model M.gguf --bodies DIR --tasks harness_accept.py
                                    --out OUT.json [--rounds 5] [--n 28] [--conc 1,4] [--max-tokens 512]
                                    [-- extra serve flags]
    (run, replay, cost: --parallel N, default 4: the acceptance test's; 1 for a model that serves one request at
    a time, a mixture of experts)

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
"route", "body"}`, and a legit cell's `"want"`; the ones whose "model" isn't M's file name are skipped), each greedy and with seeds 1..N, one at a
time (each reply's stalls are /metrics' delta), on the same server flags:
  runaway   a request after which an acceptance run's call ran on: the reply must end before its token limit, every
            call valid but a stalled one (the call ends by itself, or it stalls: the reply ends there and the call,
            the last, arrives cut off, its arguments not JSON: judged `stalled`, never `ok`, the lead's ruling of
            11-17), and a stalled reply must say it was cut: finish `length` (Responses: `incomplete`; the lead's
            ruling of 11-17 on Fable's review: pi runs a repaired call otherwise). A reply cut without a stall ran
            on. A runaway cell that no mode loops on is printed NO-LOOP (it tests nothing at that head; shady's
            review of 3c205cc). --known-miss names (cell, mode) pairs
            the stall rule is known to miss (11-16: 11-pi-edit seeds 1 and 2, a loop that breaks off and starts
            over; docs/serving.md): such a reply may run on to its limit, its call cut there and nothing else wrong,
            no stall (reported as a miss, not judged); a listed reply that ends is fine. Any other runaway that
            runs on still fails, so a stall rule that stopped closing calls can't pass;
  legit     a request for data that repeats one value (an empty board, a table of zeros), of a size it names (the
            cell's `want`: {"unit": a regex, "n": N}, e.g. 256 zeros): every call valid, and no stall while the
            call holds the size asked for. A stall after the call held more than N units (counted in its arguments,
            JSON escapes undone) ran past the request: the rule caught a real runaway, not legit data. That row
            passes, printed `overran U/N` (11-17, #274's 7f0e and 899c: on sm_89, 14-c's greedy reply writes 683
            zeros on one line for the 256 asked and stalls; with --no-constrain the same text runs on to
            max_tokens, so the mask changed nothing). A stall at N or fewer cut the data asked for: a failure. A
            stall that can't be counted (no `want`) fails. A reply that runs on to its limit fails, as before;
  known-fp  one the rule is known to cut (reported, not judged). No cell now: 16-csv was one under the few-ids rule
            (1,024 tokens of at most 8 ids), dropped in Fable's review (0932813); under the 2,048-token rule it
            stalls at 120 rows of the 80 asked (sm_89 and sm_120), so it is `legit` with a `want` since 11-17.

cost (Fable's review of #274, the lead's ruling of 11-16: the constraint's end-to-end cost with the harnesses'
sampler, against the spec's bars): the candidate's server with constrained tool calls (`on`, its default) and with
--no-constrain (`off`, main's paths), a fresh server each, ROUNDS rounds, the arms' order alternating (on off, off
on, ...). Each server: 4 untimed requests, then at each concurrency C the same N streamed chat requests, the
coding agents' tools (codex's, opencode's and pi's in turn: DIR's captured bodies, system prompt and tools) with the
acceptance test's task prompts (--tasks), seeds 1..N, max_tokens --max-tokens, Qwen2.5's card sampler (the server's
flags, as `run`'s). Per (arm, C, round): decode tok/s, the median of the requests' server `timings`
(predicted_per_second); TTFT, the median of the client's time to the first token (content, reasoning or a call);
ITL p99 over every gap between token chunks of every request; the constraint's tokens and fallbacks (/metrics).
Per (arm, C): the median over rounds. Bars (the spec's, measured sampled): on/off decode tok/s >= 0.98, TTFT
p50 <= 1.02, ITL p99 <= 1.03, at every C. Prints COST PASS or COST FAIL; exit 0 / 1.

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
import re
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
            "32768", "--kv", "q8", "--parallel", str(a.parallel)] + SAMPLER + a.serve
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
    bad, calls_bad, http = [], [], Http()
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
            calls_bad += ["%s: %s %s" % (name, j, n) for j, n in judged if j not in ("ok", "cut")]
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
            calls_bad += ["stress: %d %s" % (counts[k], k) for k in ("badjson", "unoffered", "text", "http")
                          if counts.get(k)]
        res["metrics_after"] = metrics(port)
    finally:
        stop(proc)
    probs, ilines = integrity(res, http, a.expect_constraint)
    # The calls are judged where the constraint is on; an unconstrained arm (--no-constrain, AXON_CONSTRAIN=0, main:
    # --expect-constraint off) is expected to write malformed calls, so they are reported, not judged (shady's
    # review of 3c205cc, SHOULD 4: its VERDICT said FAIL on a run that passed).
    judge_calls = a.expect_constraint != "off"
    bad = (calls_bad if judge_calls else []) + bad + ["integrity: " + p for p in probs]
    res["bad"], res["calls_not_judged"] = bad, None if judge_calls else calls_bad
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
    if not judge_calls:
        print("CALLS NOT JUDGED (an unconstrained arm, malformed calls expected): %s"
              % ("; ".join(calls_bad[:20]) or "none malformed"))
    print("VERDICT %s" % ("PASS" if not bad else "FAIL: " + "; ".join(bad[:20])))
    return 2 if probs else 0 if not bad else 1


def unescape(s):
    """A call's arguments (JSON text, maybe cut off mid-string) with JSON's escapes undone, best effort: for counting
    a legit cell's units in the file it writes (a `\\n0` is a line break and a zero)."""
    return re.sub(r'\\(u[0-9a-fA-F]{4}|.)', lambda m: chr(int(m.group(1)[1:], 16)) if len(m.group(1)) == 5 else
                  {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}.get(m.group(1), m.group(1)), s)


def overran(c, rep):
    """A legit cell's reply: (units of `want` in its last call's arguments, N asked), or None (no `want`, no call)."""
    w = c.get("want")
    calls = rep.get("calls") or []
    if not w or not calls:
        return None
    return len(re.findall(w["unit"], unescape(calls[-1][1] or ""))), w["n"]


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
    a.known_miss = {x for x in ",".join(a.known_miss).split(",") if x}
    modes = ["greedy"] + ["seed%d" % s for s in range(1, a.seeds + 1)]
    for x in sorted(a.known_miss):  # a typo must not pass for a miss that never runs
        cell, _, mode = x.partition(":")
        if not any(n == cell and c["kind"] == "runaway" for n, c in cs) or mode not in modes:
            sys.exit("constrain_check: --known-miss %s: no runaway cell %r run as %r here" % (x, cell, mode))
    proc, port, logf, res = start(a)
    bad, rows, http = [], [], Http()
    try:
        res["identity"] = json.loads(req(port, "GET", "/identity")[1])
        res["model_id"] = mid = served_id(port)
        res["metrics_before"] = metrics(port)
        for name, c in cs:
            for mode in modes:
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
                cut = rep.get("finish") in ("length", "incomplete")
                # A stall ends the reply at the call, cut off, and the reply says it was cut as the token limit
                # says it (`length`; Responses `incomplete`: the lead's ruling of 11-17 on Fable's review, pi runs
                # a repaired call otherwise). Its last call, arguments not JSON, reply_of judges `cut`; that one is
                # `stalled`. A stalled call judged `ok` (closed, usable), or a stalled reply that doesn't say it
                # was cut, is a failure below. A reply cut without a stall ran on to its limit.
                ran_on = cut and not stalls
                if stalls == 1 and cut and judged and judged[-1][0] == "cut":
                    judged[-1] = ("stalled", judged[-1][1])
                badj = [j for j, _n in judged if j not in ("ok", "stalled")]
                usable_stall = stalls > 0 and not any(j == "stalled" for j, _n in judged)
                uncut_stall = stalls > 0 and not cut
                miss = None
                if c["kind"] == "runaway":
                    ok = not ran_on and not badj and not usable_stall and not uncut_stall
                    if not ok and "%s:%s" % (name, mode) in a.known_miss and ran_on and not stalls \
                            and badj and all(j == "cut" for j in badj):
                        ok, miss = None, "known-miss"
                elif c["kind"] == "legit":
                    # A stall past the size asked for (`want`) caught a real runaway: passes, `overran`. One at or
                    # under it cut legit data (11-17: 14-c overruns on sm_89, the same text unconstrained).
                    un = overran(c, rep) if stalls else None
                    over = bool(un and un[0] > un[1] and stalls == 1 and cut and judged and judged[-1][0] == "stalled")
                    ok = (not stalls or over) and not badj and not usable_stall and not uncut_stall
                    if over:
                        miss = "overran %d/%d" % un
                else:
                    ok = None
                row = dict(cell=name, kind=c["kind"], what=c["what"], mode=mode, ok=ok, miss=miss, stalls=stalls,
                           tokens=rep.get("tokens"), cap=cap, finish=rep.get("finish"), judged=judged,
                           args_len=[len(x[1] or "") for x in rep.get("calls") or []], s=round(time.time() - t0, 1),
                           calls=[[n, (x or "")[-160:]] for n, x in rep.get("calls") or []])
                if ok is not True or stalls:  # a failure, a known false positive or a stall: the whole reply
                    row["reply"] = {"content": rep.get("content"), "calls": rep.get("calls")}
                rows.append(row)
                verdict = "miss" if miss == "known-miss" else {True: "ok", False: "FAIL", None: "-"}[ok]
                un = overran(c, rep) if c["kind"] == "legit" and stalls else None
                print("REPLAY %-34s %-8s %-6s %-4s stalls %d tokens %s/%s finish %s calls %s%s" % (
                    name, c["kind"], mode, verdict, stalls, rep.get("tokens"), cap, rep.get("finish"),
                    ",".join("%s:%s" % j for j in judged) or "none",
                    " %s" % miss if miss and miss != "known-miss" else
                    " units %d/%d" % un if un else ""), flush=True)
                if ok is False:
                    bad.append("%s %s: %s" % (name, mode, "ran on" if ran_on else badj or (
                        "a stalled call delivered as a usable one" if usable_stall else
                        "a stalled reply that finished %s, not length/incomplete" % rep.get("finish")
                        if uncut_stall else "stalled at %d of the %d units asked (cut legit data)" % un if un else
                        "stalled")))
        res["metrics_after"] = metrics(port)
    finally:
        stop(proc)
    probs, ilines = integrity(res, http, a.expect_constraint)
    bad += ["integrity: " + p for p in probs]
    res["rows"], res["bad"] = rows, bad
    res["stall_lines"] = [ln.strip() for ln in open(logf, errors="replace") if "stalled after" in ln]
    json.dump(res, open(a.out, "w"), indent=1)
    print("STALL-LINES %d" % len(res["stall_lines"]))
    # A runaway cell no mode looped on (no stall, none ran on) tests nothing at this head: said, so a gate that
    # passes on such cells alone isn't read as coverage (shady's review of 3c205cc, SHOULD 3).
    res["no_loop"] = sorted({r["cell"] for r in rows if r["kind"] == "runaway"} -
                            {r["cell"] for r in rows if r["kind"] == "runaway" and (r["stalls"] or
                                                                                   r["finish"] in ("length", "incomplete"))})
    for cell in res["no_loop"]:
        print("NO-LOOP %s: no mode stalled or ran on (%s): tests nothing at this head" % (cell, ", ".join(modes)))
    for x in sorted(a.known_miss):
        r = [r for r in rows if "%s:%s" % (r["cell"], r["mode"]) == x]
        print("KNOWN-MISS %s: %s" % (x, "ran on (to %s/%s tokens), not judged" % (r[0]["tokens"], r[0]["cap"])
                                      if r and r[0]["miss"] else "ended (%s)" % (r[0]["finish"] if r else "not run")))
    print("METRICS %s" % json.dumps({k: res["metrics_after"].get(k, 0) - res["metrics_before"].get(k, 0)
                                     for k in res["metrics_after"]}, sort_keys=True))
    print("\n".join(ilines))
    print("VERDICT %s" % ("PASS" if not bad else "FAIL: " + "; ".join(bad[:20])))
    return 2 if probs else 0 if not bad else 1


def cost_bodies(a):
    """The cost workload: N chat requests, the agents' system prompts and tools in turn, the tasks' prompts."""
    _cells, codex_chat = cells(a.bodies)
    bases = [codex_chat]
    for prefix in ("opencode-", "pi-"):
        b = load(a.bodies, prefix)
        bases.append(dict(messages=[m for m in b["messages"] if m.get("role") == "system"], tools=b["tools"]))
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.tasks)))
    ha = __import__(os.path.basename(a.tasks)[:-3])
    tasks = [t["prompts"][0] for t in ha.TASKS]
    out = []
    for i in range(a.n):
        b = json.loads(json.dumps(bases[i % len(bases)]))
        b["messages"].append({"role": "user", "content": tasks[i % len(tasks)]})
        b.update(seed=i + 1, max_tokens=a.max_tokens)
        out.append(b)
    return out


def streamed(port, body, timeout=900):
    """One streamed chat request: (status, time to the first token chunk, the gaps between token chunks, the final
    usage chunk's usage and timings, finish)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = dict(body, stream=True, stream_options={"include_usage": True})
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raw = r.read()
        c.close()
        return r.status, None, [], None, None, raw[:200]
    times, usage, timings, finish, buf = [], None, None, None, b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        now = time.perf_counter()
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                    continue
                d = json.loads(line[6:])
                if d.get("usage"):
                    usage, timings = d["usage"], d.get("timings")
                for ch in d.get("choices") or []:
                    dl = ch.get("delta") or {}
                    if dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls"):
                        times.append(now)
                    finish = ch.get("finish_reason") or finish
    c.close()
    ttft = (times[0] - t0) * 1e3 if times else None
    gaps = [(y - x) * 1e3 for x, y in zip(times, times[1:])]
    return 200, ttft, gaps, usage, timings, finish


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def med(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def cost(a):
    bodies = cost_bodies(a)
    concs = [int(x) for x in a.conc.split(",")]
    res = {"rounds": [], "args": vars(a)}
    http_ = Http()
    serve0 = list(a.serve)
    for rnd in range(a.rounds):
        order = ("on", "off") if rnd % 2 == 0 else ("off", "on")
        for arm in order:
            a.serve = serve0 + (["--no-constrain"] if arm == "off" else [])
            a.out = "%s.r%d-%s" % (a.out0, rnd, arm)
            proc, port, logf, info = start(a)
            try:
                mid = served_id(port)
                fp = json.loads(req(port, "GET", "/identity")[1]).get("system_fingerprint")
                for b in bodies[:4]:
                    st, *_ = streamed(port, aim(dict(b, seed=1000 + b["seed"]), mid))
                    http_.see("warm %s r%d" % (arm, rnd), st, b"")
                for conc in concs:
                    m0 = metrics(port)
                    rows, lock, queue = [], threading.Lock(), list(range(len(bodies)))
                    t0 = time.perf_counter()

                    def worker():
                        while True:
                            with lock:
                                if not queue:
                                    return
                                i = queue.pop(0)
                            st, ttft, gaps, usage, timings, finish = streamed(port, aim(bodies[i], mid))
                            http_.see("%s c%d r%d #%d" % (arm, conc, rnd, i), st, b"" if st == 200 else finish)
                            with lock:
                                rows.append(dict(i=i, st=st, ttft=ttft, gaps=gaps, finish=finish,
                                                 tokens=(usage or {}).get("completion_tokens"),
                                                 tps=(timings or {}).get("predicted_per_second"),
                                                 prompt_ms=(timings or {}).get("prompt_ms")))
                    ws = [threading.Thread(target=worker) for _ in range(conc)]
                    for w in ws:
                        w.start()
                    for w in ws:
                        w.join()
                    wall = time.perf_counter() - t0
                    m1 = metrics(port)
                    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
                    cell = dict(round=rnd, arm=arm, conc=conc, fp=fp, wall_s=round(wall, 2),
                                decode_tps=med([r["tps"] for r in rows]),
                                ttft_ms=med([r["ttft"] for r in rows]),
                                ttft_srv_ms=med([r["prompt_ms"] for r in rows]),
                                itl_p99_ms=pct([g for r in rows for g in r["gaps"]], 0.99),
                                itl_p50_ms=pct([g for r in rows for g in r["gaps"]], 0.5),
                                tokens=sum(r["tokens"] or 0 for r in rows),
                                finish={f: sum(1 for r in rows if r["finish"] == f) for f in {r["finish"] for r in rows}},
                                constraint_tokens=d.get("axon_constraint_tokens_total", 0),
                                fallbacks=d.get("axon_constraint_fallback_total", 0),
                                stalls=d.get("axon_constraint_stalls_total", 0))
                    cell["agg_tps"] = round(cell["tokens"] / wall, 2)
                    res["rounds"].append(cell)
                    print("COST-CELL " + json.dumps(cell, sort_keys=True), flush=True)
            finally:
                stop(proc)
    a.serve = serve0
    bad, lines = [], []
    bars = (("decode_tps", 0.98, ">="), ("ttft_ms", 1.02, "<="), ("itl_p99_ms", 1.03, "<="))
    for conc in concs:
        agg = {}
        for arm in ("on", "off"):
            cs = [c for c in res["rounds"] if c["arm"] == arm and c["conc"] == conc]
            agg[arm] = {k: med([c[k] for c in cs]) for k in ("decode_tps", "ttft_ms", "ttft_srv_ms", "itl_p99_ms",
                                                              "itl_p50_ms", "agg_tps")}
            agg[arm]["constraint_tokens"] = sum(c["constraint_tokens"] for c in cs)
            agg[arm]["fallbacks"] = sum(c["fallbacks"] for c in cs)
            agg[arm]["per_round"] = {k: [c[k] for c in cs] for k in ("decode_tps", "ttft_ms", "itl_p99_ms")}
        on, off = agg["on"], agg["off"]
        fb = on["fallbacks"] / on["constraint_tokens"] if on["constraint_tokens"] else None
        lines.append("C=%d fallbacks %d of %d constrained tokens (%s)" % (
            conc, on["fallbacks"], on["constraint_tokens"], "-" if fb is None else "%.1f %%" % (100 * fb)))
        for k, bar, way in bars:
            r = on[k] / off[k] if on[k] and off[k] else None
            ok = r is not None and (r >= bar if way == ">=" else r <= bar)
            lines.append("C=%d %-11s on %9.2f off %9.2f on/off %s (bar %s %.2f) %s   per round on %s off %s" % (
                conc, k, on[k] or 0, off[k] or 0, "-" if r is None else "%.4f" % r, way, bar,
                "PASS" if ok else "FAIL", on["per_round"][k], off["per_round"][k]))
            if not ok:
                bad.append("C=%d %s %s" % (conc, k, "-" if r is None else "%.4f" % r))
        res.setdefault("summary", {})[str(conc)] = agg
    if http_.bad:
        bad.append("non-200: %s" % http_.bad[:5])
    # The switch did what each arm says: the constraint ran in every `on` cell and in no `off` one.
    wrong = ["r%d %s C=%d: %d constrained tokens" % (c["round"], c["arm"], c["conc"], c["constraint_tokens"])
             for c in res["rounds"] if (c["arm"] == "on") != (c["constraint_tokens"] > 0)]
    if wrong:
        bad.append("integrity: " + "; ".join(wrong[:5]))
    res["bad"] = bad
    json.dump(res, open(a.out0, "w"), indent=1)
    print("\n".join(lines))
    print("COST %s" % ("PASS" if not bad else "FAIL: " + "; ".join(bad)))
    return 0 if not bad else 1


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
    r.add_argument("--parallel", type=int, default=4)
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
    rp.add_argument("--parallel", type=int, default=4)
    rp.add_argument("--known-miss", action="append", default=[],
                    help="CELL:MODE[,...]: a runaway reply the stall rule is known to miss (may run on, cut)")
    rp.add_argument("serve", nargs="*")
    c = sp.add_parser("cost")
    c.add_argument("--axon", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--bodies", required=True)
    c.add_argument("--tasks", required=True)
    c.add_argument("--out", required=True, dest="out0")
    c.add_argument("--rounds", type=int, default=5)
    c.add_argument("--n", type=int, default=28)
    c.add_argument("--conc", default="1,4")
    c.add_argument("--max-tokens", type=int, default=512)
    c.add_argument("--port", type=int, default=0)
    c.add_argument("--parallel", type=int, default=4)
    c.add_argument("serve", nargs="*")
    a = p.parse_args()
    sys.exit({"run": run, "replay": replay, "cost": cost}[a.cmd](a) if a.cmd != "diff" else diff(a.a, a.b))


if __name__ == "__main__":
    main()
