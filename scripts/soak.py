#!/usr/bin/env python3
"""Multi-hour soak of `axon serve` (stdlib only).

    scripts/soak.py <axon binary> <model.gguf> --hours H [--clients N] [--out DIR]
                    [--port P] [--sample-s 30] [--warmup-min M] [--seed S] [-- serve flags]
    scripts/soak.py report DIR            (re-read DIR/summary.json, print the verdict)
    scripts/soak.py summarise FILE [MIN]  (a markdown table of samples.csv or a job log, MIN-minute buckets)

Starts `axon serve` itself (so it knows the pid), then N client threads run a
realistic mix until the deadline:

  chat-stream / chat-whole   short prompts, streamed or not, greedy or sampled
  completion                 /v1/completions, streamed or not
  tool                       a tool-call turn, then the tool result turn
  long                       a prompt of ~40-60% of the per-slot context
  agent                      a long shared system prefix + a short varying turn (prefix cache)
  disconnect                 a stream the client closes after a few chunks
  slow-reader                a stream read a few bytes at a time with pauses (tiny SO_RCVBUF)
  oversize                   a prompt past the context: must be a 4xx, never a 5xx
  invalid                    malformed JSON / bad params / wrong method: must be a 4xx
  burst                      every ~2 min one client fires 3x the slot count at once (503 allowed)

Every --sample-s seconds one CSV row (DIR/samples.csv, also printed as `CSV,...`
on stdout so a remote log keeps it) records: server RSS, VRAM (nvidia-smi
compute apps for the server pid; the whole card's used memory if the pid isn't
listed), open fds, threads, requests and latency p50/p99 (whole request, ok
ones) and TTFT p50/p99 of the window, and counts by class.

Identity: a fixed set of greedy requests runs before the load, every
--identity-min minutes under load, and after it; every run must give the
same text and token counts as the first (batch invariance + no state leak).

Verdict (summary.json, exit 0 = PASS):
  - the server is alive at the end and its log has no `panicked`
  - no 5xx other than 503, no unexpected statuses (a valid request that gets a
    4xx, an invalid one that gets a 200), no dropped connections on valid requests
  - every identity run matches
  - memory is flat after warm-up: a least-squares line through RSS (and VRAM)
    against time, its slope projected over 24 h, must be under the bound
    max(--rss-floor-mib, --rss-frac * median RSS) (VRAM: --vram-floor-mib). The
    bound is set from what a leak would have to be to matter for a server left
    up for a day, well above allocator noise (see docs/hardening.md).
  - after the load stops and the server idles, fds and threads are back at
    the pre-load baseline (+2 fds of slack for a lingering keep-alive).
"""

import argparse
import csv
import http.client
import json
import math
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

HOST = "127.0.0.1"
MODEL = ["x"]  # the server's model id (/v1/models), set at start
WORDS = ("the river model token server memory signal garden window copper winter yellow harbor "
         "engine lattice quiet bridge orbit marble paper silver forest cable planet market "
         "violet anchor thunder needle canvas meadow").split()


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


# ---------------------------------------------------------------- server

class Server:
    def __init__(self, axon, model, port, extra, out):
        self.port = port
        self.log_path = os.path.join(out, "serve.log")
        self.log = open(self.log_path, "wb")
        cmd = [axon, "serve", model, "--port", str(port)] + extra
        print("serve:", " ".join(cmd), flush=True)
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self.pid = self.p.pid
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"server exited rc={self.p.returncode} while loading: {self.log_path}")
            try:
                st, _, _ = request(port, "GET", "/health", None, timeout=5)
                if st == 200:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            raise SystemExit("server not healthy after 900 s")
        self.load_s = time.time() - t0

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        if self.alive():
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.log.close()


def request(port, method, path, body, timeout=600, raw=None):
    """(status, headers dict, body bytes). raw: bytes sent as the body as is."""
    c = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        c.request(method, path, data, {"Content-Type": "application/json"} if data is not None else {})
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        c.close()


def sse_events(r, rd=None):
    """Yield decoded SSE data payloads (dict, or the string '[DONE]')."""
    buf = b""
    while True:
        chunk = rd() if rd else r.read1(65536)
        if not chunk:
            return
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if line.startswith(b"data: "):
                    d = line[6:]
                    yield "[DONE]" if d == b"[DONE]" else json.loads(d)


# ---------------------------------------------------------------- results

class Outcome(Exception):
    """A request's class when it isn't 'ok'."""

    def __init__(self, cls, detail=""):
        super().__init__(f"{cls}: {detail}")
        self.cls, self.detail = cls, detail


class Stats:
    """Per-window counters and latencies (reset at every sample) + run totals."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()
        self.total = {}
        self.findings = []
        self.requests = 0
        self.lat_max = {}

    def reset(self):
        self.win = {}
        self.lat = []
        self.ttft = []

    def add(self, kind, cls, lat=None, ttft=None, detail=""):
        with self.lock:
            self.win[cls] = self.win.get(cls, 0) + 1
            self.total[f"{kind}:{cls}"] = self.total.get(f"{kind}:{cls}", 0) + 1
            self.requests += 1
            if lat is not None and cls == "ok":
                self.lat.append(lat)
                m = self.lat_max.get(kind, 0.0)
                self.lat_max[kind] = max(m, round(lat, 2))
            if ttft is not None:
                self.ttft.append(ttft)
            if cls in BAD and len(self.findings) < 200:
                self.findings.append({"t": time.time(), "kind": kind, "class": cls, "detail": detail[:300]})

    def take(self):
        with self.lock:
            w, l, t = self.win, self.lat, self.ttft
            self.reset()
            return w, l, t


# Classes that fail the soak.
BAD = {"5xx", "unexpected-4xx", "unexpected-200", "dropped", "bad-body", "timeout", "other-status"}
CLASSES = ["ok", "4xx-expected", "503", "cancelled"] + sorted(BAD)


def classify_status(st, body, expect_ok):
    """None if st is the expected success; else raise the right Outcome."""
    if st == 503:
        raise Outcome("503", body[:120].decode("utf-8", "replace"))
    if st >= 500:
        raise Outcome("5xx", f"{st} {body[:200]!r}")
    if 400 <= st < 500:
        try:
            json.loads(body)["error"]["message"]
        except Exception:
            raise Outcome("bad-body", f"{st} error without a JSON message: {body[:200]!r}")
        if expect_ok:
            raise Outcome("unexpected-4xx", f"{st} {body[:200]!r}")
        raise Outcome("4xx-expected")
    if st == 200 and not expect_ok:
        raise Outcome("unexpected-200", body[:200].decode("utf-8", "replace"))
    if st != 200:
        raise Outcome("other-status", f"{st}")


# ---------------------------------------------------------------- workload

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"]}}}]


class Client:
    def __init__(self, i, cfg, stats, stop_at):
        self.i, self.cfg, self.stats, self.stop_at = i, cfg, stats, stop_at
        self.rng = random.Random(cfg.seed * 1000 + i)
        self.port = cfg.port
        # ~1.3 tokens per WORDS word (measured on qwen tokenizers: mostly 1).
        self.ctx_words = cfg.slot_ctx

    def chat_body(self, msgs, **kw):
        b = {"model": MODEL[0], "messages": msgs, "max_tokens": kw.pop("max_tokens", self.rng.choice([8, 16, 32, 64, 128]))}
        if self.rng.random() < 0.5:
            b["temperature"] = 0
        else:
            b.update(temperature=round(self.rng.uniform(0.3, 1.0), 2), top_p=0.95, seed=self.rng.randrange(1 << 30))
        b.update(kw)
        return b

    def stream(self, path, body, kind, close_after=None, slow=False):
        t0 = time.perf_counter()
        s = None
        if slow:
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        c = http.client.HTTPConnection(HOST, self.port, timeout=self.cfg.req_timeout)
        if s is not None:
            s.settimeout(self.cfg.req_timeout)
            s.connect((HOST, self.port))
            c.sock = s
        ttft = None
        try:
            body = dict(body, stream=True, stream_options={"include_usage": True})
            c.request("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                classify_status(r.status, r.read(), True)
            n, done, finish, usage = 0, False, None, None
            rd = (lambda: (time.sleep(self.rng.uniform(0.02, 0.1)), r.read1(64))[1]) if slow else None
            for ev in sse_events(r, rd):
                if ev == "[DONE]":
                    done = True
                    continue
                if "error" in ev:
                    raise Outcome("5xx", f"stream error event {ev['error']}")
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    dl = ch.get("delta") or {}
                    if ttft is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls")
                                         or ch.get("text")):
                        ttft = time.perf_counter() - t0
                    n += 1
                    finish = ch.get("finish_reason") or finish
                if close_after is not None and n >= close_after:
                    raise Outcome("cancelled")
            if not done or finish is None or usage is None:
                raise Outcome("bad-body", f"stream ended early: done={done} finish={finish} usage={usage}")
            self.stats.add(kind, "ok", time.perf_counter() - t0, ttft)
        finally:
            c.close()

    def whole(self, path, body, kind, expect_ok=True, raw=None, method="POST"):
        t0 = time.perf_counter()
        st, _, rb = request(self.port, method, path, body, timeout=self.cfg.req_timeout, raw=raw)
        classify_status(st, rb, expect_ok)
        if expect_ok:
            j = json.loads(rb)
            ch = j["choices"][0]
            if ch.get("finish_reason") is None or "usage" not in j:
                raise Outcome("bad-body", rb[:200].decode("utf-8", "replace"))
            self.stats.add(kind, "ok", time.perf_counter() - t0)
            return j
        return None

    # --- the mix
    def chat(self):
        msgs = [{"role": "user", "content": self.rng.choice([
            "Write one sentence about ", "Summarise in a line: ", "List three facts about ",
            "Explain briefly: "]) + words(self.rng, self.rng.randint(3, 40))}]
        if self.rng.random() < 0.6:
            self.stream("/v1/chat/completions", self.chat_body(msgs), "chat-stream")
        else:
            self.whole("/v1/chat/completions", self.chat_body(msgs), "chat-whole")

    def completion(self):
        b = {"model": MODEL[0], "prompt": "Once upon a time " + words(self.rng, self.rng.randint(1, 30)),
             "max_tokens": self.rng.choice([8, 32, 64]), "temperature": 0}
        if self.rng.random() < 0.5:
            self.stream("/v1/completions", b, "completion")
        else:
            self.whole("/v1/completions", b, "completion")

    def tool(self):
        city = self.rng.choice(["Paris", "Tokyo", "Lagos", "Lima", "Oslo"])
        msgs = [{"role": "user", "content": f"What's the weather in {city}?"}]
        choice = self.rng.choice(["auto", "required", {"type": "function", "function": {"name": "get_weather"}}])
        j = self.whole("/v1/chat/completions",
                       self.chat_body(msgs, tools=TOOLS, tool_choice=choice, max_tokens=96, temperature=0), "tool")
        m = j["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        if choice != "auto" and not calls:
            raise Outcome("bad-body", f"tool_choice={choice} gave no tool call: {m}")
        if calls:
            msgs += [{"role": "assistant", "content": m.get("content"), "tool_calls": calls},
                     {"role": "tool", "tool_call_id": calls[0].get("id", "c0"),
                      "content": json.dumps({"temp_c": self.rng.randint(-5, 35), "sky": "clear"})}]
            self.stream("/v1/chat/completions", self.chat_body(msgs, tools=TOOLS, max_tokens=48), "tool")

    def long(self):
        n = int(self.ctx_words * self.rng.uniform(0.35, 0.55))
        msgs = [{"role": "user", "content": "Read this and give the most common word: " + words(self.rng, n)}]
        self.stream("/v1/chat/completions", self.chat_body(msgs, max_tokens=self.rng.choice([8, 32])), "long")

    def agent(self):
        # The shared prefix: the same for every client (prefix cache across slots).
        sysmsg = self.cfg.agent_prefix
        turns = [{"role": "system", "content": sysmsg}]
        for _ in range(self.rng.randint(1, 3)):
            turns.append({"role": "user", "content": "Task: " + words(self.rng, self.rng.randint(5, 20))})
            j = self.whole("/v1/chat/completions", self.chat_body(turns, max_tokens=32), "agent")
            turns.append({"role": "assistant", "content": j["choices"][0]["message"].get("content") or ""})

    def disconnect(self):
        msgs = [{"role": "user", "content": "Count slowly from one to two hundred."}]
        try:
            self.stream("/v1/chat/completions", self.chat_body(msgs, max_tokens=400),
                        "disconnect", close_after=self.rng.randint(1, 20))
        except Outcome as o:
            if o.cls != "cancelled":
                raise
            self.stats.add("disconnect", "cancelled")

    def slow(self):
        msgs = [{"role": "user", "content": "Write a short poem about " + words(self.rng, 3)}]
        self.stream("/v1/chat/completions", self.chat_body(msgs, max_tokens=96), "slow-reader", slow=True)

    def oversize(self):
        n = int(self.cfg.ctx * 1.3) + 64
        b = {"model": MODEL[0], "messages": [{"role": "user", "content": words(self.rng, n)}], "max_tokens": 8}
        self.whole("/v1/chat/completions", b, "oversize", expect_ok=False)

    def invalid(self):
        case = self.rng.randrange(8)
        P = "/v1/chat/completions"
        if case == 0:
            self.whole(P, None, "invalid", False, raw=b'{"messages": [{"role": "user", "content": "x"')
        elif case == 1:
            self.whole(P, {"model": MODEL[0], "messages": [{"role": "user", "content": "x"}], "temperature": -1}, "invalid", False)
        elif case == 2:
            self.whole(P, {"model": MODEL[0], "messages": "nope"}, "invalid", False)
        elif case == 3:
            self.whole(P, {"model": MODEL[0], "messages": [{"role": "user", "content": "x"}], "n": 3}, "invalid", False)
        elif case == 4:
            self.whole(P, None, "invalid", False, raw=b"\xff\xfe not utf8 \xc3")
        elif case == 5:
            self.whole("/v1/completions", {"model": MODEL[0], "prompt": "x", "logit_bias": {"1": 5}}, "invalid", False)
        elif case == 6:
            self.whole(P, {"model": MODEL[0], "messages": [{"role": "user", "content": "x"}], "max_tokens": -5}, "invalid", False)
        else:
            self.whole("/v1/nope", {"x": 1}, "invalid", False)

    def burst(self):
        n = self.cfg.slots * 3
        errs = []

        def one(k):
            try:
                self.stream("/v1/chat/completions", self.chat_body(
                    [{"role": "user", "content": f"burst {k}: " + words(self.rng, 8)}], max_tokens=16), "burst")
            except Outcome as o:
                self.stats.add("burst", o.cls, detail=o.detail)
            except Exception as e:  # noqa: BLE001
                errs.append(e)
                self.stats.add("burst", "dropped", detail=repr(e))
        ts = [threading.Thread(target=one, args=(k,)) for k in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    MIX = [("chat", 30), ("completion", 8), ("tool", 10), ("long", 6), ("agent", 14),
           ("disconnect", 8), ("slow", 4), ("oversize", 3), ("invalid", 7)]

    def run(self):
        names = [m for m, _ in self.MIX]
        weights = [w for _, w in self.MIX]
        last_burst = time.time() + self.rng.uniform(30, 120)
        while time.time() < self.stop_at:
            if self.i == 0 and time.time() - last_burst > 120:
                kind, last_burst = "burst", time.time()
            else:
                kind = self.rng.choices(names, weights)[0]
            try:
                getattr(self, kind)()
            except Outcome as o:
                self.stats.add(kind, o.cls, detail=o.detail)
                if o.cls == "503":  # back off, as a real client would
                    time.sleep(self.rng.uniform(1, 3))
            except (socket.timeout, TimeoutError) as e:
                self.stats.add(kind, "timeout", detail=repr(e))
            except (ConnectionError, http.client.HTTPException, OSError) as e:
                self.stats.add(kind, "dropped", detail=repr(e))
            except Exception as e:  # noqa: BLE001
                self.stats.add(kind, "bad-body", detail=repr(e))
            time.sleep(self.rng.uniform(0, self.cfg.think_s))


# ---------------------------------------------------------------- identity

IDENT = [
    ("chat", [{"role": "user", "content": "Write a haiku about the sea."}], {}),
    ("chat", [{"role": "system", "content": "You are terse."},
              {"role": "user", "content": "Explain what a hash map is."}], {}),
    ("tool", [{"role": "user", "content": "What's the weather in Paris in celsius?"}],
     {"tools": TOOLS, "tool_choice": "required"}),
    ("long", None, {}),
]


def identity_run(cfg):
    out = []
    for kind, msgs, extra in IDENT:
        if msgs is None:
            rng = random.Random(7)
            msgs = [{"role": "user", "content": "Repeat the last word: " + words(rng, cfg.slot_ctx // 3)}]
        b = dict({"model": MODEL[0], "messages": msgs, "max_tokens": 64, "temperature": 0}, **extra)
        # A 503 is back-pressure (a full queue under load), not an answer: retry it.
        t0 = time.time()
        while True:
            st, _, rb = request(cfg.port, "POST", "/v1/chat/completions", b, timeout=cfg.req_timeout)
            if st != 503 or time.time() - t0 > 600:
                break
            time.sleep(2)
        if st != 200:
            out.append({"kind": kind, "status": st})
            continue
        j = json.loads(rb)
        m = j["choices"][0]["message"]
        out.append({"kind": kind, "content": m.get("content"), "tool_calls": m.get("tool_calls"),
                    "reasoning": m.get("reasoning_content"),
                    "tokens": j["usage"]["completion_tokens"], "finish": j["choices"][0]["finish_reason"]})
    for o in out:  # tool call ids are random per response
        for c in o.get("tool_calls") or []:
            c.pop("id", None)
    return out


# ---------------------------------------------------------------- sampling

def proc_stats(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k, _, v = line.partition(":")
            if k in ("VmRSS", "Threads", "VmHWM"):
                d[k] = int(v.split()[0])
        d["fds"] = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        pass
    return d.get("VmRSS", -1) / 1024, d.get("Threads", -1), d.get("fds", -1), d.get("VmHWM", -1) / 1024


def vram_mib(pid):
    """(MiB used by pid, 'pid'), or the card's used total (,'card'); (-1, 'none') without nvidia-smi."""
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout
        for line in o.splitlines():
            p, _, m = line.partition(",")
            if p.strip() == str(pid):
                return float(m), "pid"
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout
        return float(o.split()[0]), "card"
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return -1.0, "none"


def pct(xs, q):
    if not xs:
        return ""
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))] * 1000, 1)


FIELDS = ["t_s", "rss_mib", "hwm_mib", "vram_mib", "vram_src", "fds", "threads", "requests",
          "lat_p50_ms", "lat_p99_ms", "ttft_p50_ms", "ttft_p99_ms"] + CLASSES + ["alive"]


def slope_per_h(ts, ys):
    """Least-squares slope of ys over ts (seconds), per hour."""
    n = len(ts)
    if n < 3:
        return 0.0
    mt, my = sum(ts) / n, sum(ys) / n
    den = sum((t - mt) ** 2 for t in ts)
    return 0.0 if den == 0 else sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / den * 3600


# ---------------------------------------------------------------- main

def verdict(rows, summary, cfg):
    """Fill summary['checks'] and summary['pass'] from the samples."""
    warm = cfg.warmup_min * 60
    post = [r for r in rows if float(r["t_s"]) >= warm and r["alive"] == "1"]
    checks = {}
    ts = [float(r["t_s"]) for r in post]
    rss = [float(r["rss_mib"]) for r in post]
    if len(post) >= 3:
        med = statistics.median(rss)
        s = slope_per_h(ts, rss)
        bound = max(cfg.rss_floor_mib, cfg.rss_frac * med)
        q = max(1, len(rss) // 4)
        checks["rss"] = {"slope_mib_per_h": round(s, 3), "growth_24h_mib": round(s * 24, 1),
                         # informational: a one-time step (an arena, a lazily captured graph) vs steady growth
                         "last_vs_first_quarter_mib": round(statistics.median(rss[-q:]) - statistics.median(rss[:q]), 1),
                         "bound_24h_mib": round(bound, 1), "median_mib": round(med, 1),
                         "min_mib": min(rss), "max_mib": max(rss), "pass": s * 24 < bound}
        vp = [r for r in post if float(r["vram_mib"]) >= 0 and (r["vram_src"] == "pid" or cfg.vram_card)]
        vr = [float(r["vram_mib"]) for r in vp]
        if len(vr) >= 3:
            vt = [float(r["t_s"]) for r in vp]
            vs = slope_per_h(vt, vr)
            q = max(1, len(vr) // 4)
            checks["vram"] = {"slope_mib_per_h": round(vs, 3), "growth_24h_mib": round(vs * 24, 1),
                              "last_vs_first_quarter_mib": round(statistics.median(vr[-q:]) - statistics.median(vr[:q]), 1),
                              "bound_24h_mib": cfg.vram_floor_mib, "min_mib": min(vr), "max_mib": max(vr),
                              "src": post[-1]["vram_src"], "pass": vs * 24 < cfg.vram_floor_mib}
    else:
        checks["rss"] = {"pass": False, "why": "fewer than 3 samples after warm-up"}
    bad = {k: v for k, v in summary["totals"].items() if k.split(":")[1] in BAD}
    checks["errors"] = {"bad": bad, "pass": not bad}
    checks["alive"] = {"pass": summary["alive_at_end"]}
    checks["panic"] = {"lines": summary["panic_lines"], "pass": not summary["panic_lines"]}
    checks["identity"] = {"runs": summary["identity_runs"], "mismatches": summary["identity_mismatches"],
                          "pass": summary["identity_runs"] >= 2 and not summary["identity_mismatches"]}
    b, e = summary["idle_before"], summary["idle_after"]
    checks["fds_threads"] = {"before": b, "after": e,
                             "pass": e["fds"] <= b["fds"] + 2 and e["threads"] <= b["threads"]}
    summary["checks"] = checks
    summary["pass"] = all(c["pass"] for c in checks.values())


def summarise(path, bucket_min=30):
    """A markdown table of a samples.csv (or a job log with CSV, lines) in bucket_min buckets."""
    lines = [ln[4:] if ln.startswith("CSV,") else ln for ln in open(path, errors="replace")]
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("t_s,"))
    rows = list(csv.DictReader([ln for ln in lines[hdr:] if ln[:1].isdigit() or ln.startswith("t_s,")]))
    print("| minutes | requests | ok | 503 | 4xx exp. | cancelled | failing | RSS MiB | VRAM MiB | fds | threads "
          "| lat p50 ms | lat p99 ms | TTFT p50 ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    b = {}
    for r in rows:
        b.setdefault(int(float(r["t_s"]) // (bucket_min * 60)), []).append(r)
    for k in sorted(b):
        rs = b[k]
        n = lambda c: sum(int(r[c]) for r in rs)  # noqa: E731
        med = lambda c: statistics.median([float(r[c]) for r in rs if r[c] != ""] or [0])  # noqa: E731
        print(f"| {k * bucket_min}-{(k + 1) * bucket_min} | {n('requests')} | {n('ok')} | {n('503')} | "
              f"{n('4xx-expected')} | {n('cancelled')} | {sum(n(c) for c in sorted(BAD))} | "
              f"{rs[-1]['rss_mib']} | {rs[-1]['vram_mib'] if rs[-1]['vram_src'] == 'pid' else '-'} | {max(int(r['fds']) for r in rs)} | "
              f"{max(int(r['threads']) for r in rs)} | {med('lat_p50_ms'):.0f} | {med('lat_p99_ms'):.0f} | "
              f"{med('ttft_p50_ms'):.0f} |")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summarise":
        summarise(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        d = sys.argv[2]
        s = json.load(open(os.path.join(d, "summary.json")))
        print(json.dumps(s["checks"], indent=1))
        print("PASS" if s["pass"] else "FAIL")
        return 0 if s["pass"] else 1
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("axon")
    ap.add_argument("model")
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--clients", type=int, default=12)
    ap.add_argument("--out", default="soak-out")
    ap.add_argument("--port", type=int, default=18400)
    ap.add_argument("--ctx", type=int, default=4096, help="serve --ctx (per slot)")
    ap.add_argument("--slots", type=int, default=4, help="serve --parallel (1 on CPU)")
    ap.add_argument("--sample-s", type=float, default=30)
    ap.add_argument("--warmup-min", type=float, default=20)
    ap.add_argument("--identity-min", type=float, default=15)
    ap.add_argument("--think-s", type=float, default=1.0, help="most idle time between a client's requests")
    ap.add_argument("--req-timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rss-floor-mib", type=float, default=64)
    ap.add_argument("--rss-frac", type=float, default=0.02)
    ap.add_argument("--vram-floor-mib", type=float, default=16)
    ap.add_argument("--vram-card", action="store_true",
                    help="judge VRAM on the card's total when nvidia-smi doesn't list the server pid "
                         "(only on a GPU nothing else uses)")
    cfg = ap.parse_args(argv)
    cfg.slot_ctx = cfg.ctx
    os.makedirs(cfg.out, exist_ok=True)
    rng = random.Random(cfg.seed)
    cfg.agent_prefix = ("You are an agent with these rules. " + words(rng, cfg.ctx // 4))
    sargs = ["--ctx", str(cfg.ctx)] + (["--parallel", str(cfg.slots)] if cfg.slots > 1 else [])
    srv = Server(cfg.axon, cfg.model, cfg.port, sargs + extra, cfg.out)
    print(f"loaded in {srv.load_s:.1f}s, pid {srv.pid}", flush=True)
    MODEL[0] = json.loads(request(cfg.port, "GET", "/v1/models", None)[2])["data"][0]["id"]
    summary = {"args": {k: v for k, v in vars(cfg).items() if k != "agent_prefix"}, "extra": extra,
               "load_s": round(srv.load_s, 1), "identity_runs": 0, "identity_mismatches": []}
    t_start = time.time()
    rows = []
    try:
        ident0 = identity_run(cfg)
        summary["identity_ref"] = ident0
        summary["identity_runs"] = 1
        time.sleep(2)
        r0, th0, fd0, _ = proc_stats(srv.pid)
        summary["idle_before"] = {"rss_mib": r0, "threads": th0, "fds": fd0}
        stats = Stats()
        stop_at = t_start + cfg.hours * 3600
        clients = [Client(i, cfg, stats, stop_at) for i in range(cfg.clients)]
        ths = [threading.Thread(target=c.run, daemon=True) for c in clients]
        for t in ths:
            t.start()
        f = open(os.path.join(cfg.out, "samples.csv"), "w", newline="")
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        print("CSV," + ",".join(FIELDS), flush=True)
        next_s, next_id = time.time() + cfg.sample_s, time.time() + cfg.identity_min * 60
        idt = None

        def ident_check():
            """One identity run under load (a thread: it takes a while)."""
            try:
                got = identity_run(cfg)
            except Exception as e:  # noqa: BLE001
                got = [{"error": repr(e)}]
            summary["identity_runs"] += 1
            if got != ident0:
                summary["identity_mismatches"].append({"t_s": round(time.time() - t_start), "got": got})
                print(f"IDENTITY MISMATCH at {time.time() - t_start:.0f}s: {got}", flush=True)
        while time.time() < stop_at and srv.alive():
            time.sleep(min(1.0, max(0.0, next_s - time.time())))
            if time.time() >= next_id and not (idt and idt.is_alive()):
                next_id = time.time() + cfg.identity_min * 60
                idt = threading.Thread(target=ident_check, daemon=True)
                idt.start()
            if time.time() < next_s:
                continue
            next_s += cfg.sample_s
            win, lat, ttft = stats.take()
            rss, th, fds, hwm = proc_stats(srv.pid)
            vr, vsrc = vram_mib(srv.pid)
            row = {"t_s": round(time.time() - t_start, 1), "rss_mib": round(rss, 1), "hwm_mib": round(hwm, 1),
                   "vram_mib": vr, "vram_src": vsrc, "fds": fds, "threads": th,
                   "requests": sum(win.values()), "lat_p50_ms": pct(lat, 0.5), "lat_p99_ms": pct(lat, 0.99),
                   "ttft_p50_ms": pct(ttft, 0.5), "ttft_p99_ms": pct(ttft, 0.99),
                   "alive": int(srv.alive())}
            row.update({c: win.get(c, 0) for c in CLASSES})
            rows.append({k: str(v) for k, v in row.items()})
            w.writerow(row)
            f.flush()
            print("CSV," + ",".join(str(row[k]) for k in FIELDS), flush=True)
        for t in ths + ([idt] if idt else []):
            t.join(timeout=cfg.req_timeout + 30)
        summary["alive_at_end"] = srv.alive()
        if srv.alive():
            try:
                got = identity_run(cfg)
                summary["identity_runs"] += 1
                if got != ident0:
                    summary["identity_mismatches"].append({"t_s": "end", "got": got})
            except Exception as e:  # noqa: BLE001
                summary["identity_mismatches"].append({"t_s": "end", "error": repr(e)})
        time.sleep(35)  # idle keep-alives (30 s) close
        r1, th1, fd1, hwm1 = proc_stats(srv.pid)
        summary["idle_after"] = {"rss_mib": r1, "threads": th1, "fds": fd1, "hwm_mib": hwm1}
        summary["totals"] = dict(sorted(stats.total.items()))
        summary["requests"] = stats.requests
        summary["lat_max_s"] = stats.lat_max
        summary["findings"] = stats.findings
        summary["wall_h"] = round((time.time() - t_start) / 3600, 3)
    finally:
        alive = srv.alive()
        srv.stop()
        summary.setdefault("alive_at_end", alive)
    log = open(srv.log_path, errors="replace").read()
    summary["panic_lines"] = [ln for ln in log.splitlines() if re.search(r"panicked|RUST_BACKTRACE", ln)][:20]
    summary.setdefault("totals", {})
    summary.setdefault("idle_before", {"fds": 0, "threads": 0})
    summary.setdefault("idle_after", {"fds": 1 << 30, "threads": 1 << 30})
    verdict(rows, summary, cfg)
    json.dump(summary, open(os.path.join(cfg.out, "summary.json"), "w"), indent=1)
    print("SUMMARY " + json.dumps({k: summary[k] for k in ("checks", "requests", "totals", "wall_h") if k in summary}))
    for fd in summary.get("findings", [])[:30]:
        print("FINDING", json.dumps(fd))
    print("VERDICT", "PASS" if summary["pass"] else "FAIL", flush=True)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
