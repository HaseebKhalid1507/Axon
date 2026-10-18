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
        self.probes = []

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
BAD = {"5xx", "unexpected-4xx", "unexpected-200", "dropped", "bad-body", "timeout", "other-status",
       "503-undocumented", "slow-blocks-others"}
# accepted-unsupported: a request for a feature serve refuses today (n > 1, logit_bias) got a 200:
# the feature shipped and the harness should move the case; reported, not failing.
CLASSES = ["ok", "4xx-expected", "503", "cancelled", "accepted-unsupported"] + sorted(BAD)
# The documented back-pressure 503s (serve.rs): anything else is 503-undocumented.
BACKPRESSURE = (b"too many queued requests", b"too many connections")


def classify_status(st, body, expect_ok, unsupported_today=False):
    """None if st is the expected success; else raise the right Outcome."""
    if st == 503:
        if not any(m in body for m in BACKPRESSURE):
            raise Outcome("503-undocumented", body[:200].decode("utf-8", "replace"))
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
        if unsupported_today:
            raise Outcome("accepted-unsupported", body[:200].decode("utf-8", "replace"))
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

    def whole(self, path, body, kind, expect_ok=True, raw=None, method="POST", unsupported_today=False):
        t0 = time.perf_counter()
        st, _, rb = request(self.port, method, path, body, timeout=self.cfg.req_timeout, raw=raw)
        classify_status(st, rb, expect_ok, unsupported_today)
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
        """A reader slow enough to block the server's writes: ~1500 tokens of SSE (~300 KB, far past the
        socket buffers) read 64 bytes at a time through a 4 KiB SO_RCVBUF. While it is blocked, a probe
        (greedy, 4 tokens) must still get its first token within --probe-ttft-s, or the slow client is
        holding up others (slow-blocks-others). Only with several slots: one slot is held by design."""
        msgs = [{"role": "user", "content": "Count from 1 to 2000, one number per line, no other text."}]
        probe = None
        if self.cfg.slots > 1:
            probe = threading.Timer(self.rng.uniform(3, 8), self.probe)
            probe.daemon = True
            probe.start()
        try:
            self.stream("/v1/chat/completions", self.chat_body(msgs, max_tokens=1500, temperature=0),
                        "slow-reader", slow=True)
        finally:
            if probe:
                probe.cancel()

    def probe(self):
        t0 = time.perf_counter()
        b = {"model": MODEL[0], "max_tokens": 4, "temperature": 0, "stream": True,
             "messages": [{"role": "user", "content": "Say hi."}]}
        c = http.client.HTTPConnection(HOST, self.port, timeout=self.cfg.req_timeout)
        try:
            c.request("POST", "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                r.read()
                return  # a 503 under load: not this check's business
            r.read1(65536)
            dt = time.perf_counter() - t0
            with self.stats.lock:
                self.stats.probes.append(round(dt, 3))
            if dt > self.cfg.probe_ttft_s:
                self.stats.add("slow-reader-probe", "slow-blocks-others",
                               detail=f"probe first bytes after {dt:.1f}s while a slow reader was blocked")
        except OSError:
            pass
        finally:
            c.close()

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
            # Unsupported today (api.rs: n is 1 only), not invalid: a 200 is accepted-unsupported.
            self.whole(P, {"model": MODEL[0], "messages": [{"role": "user", "content": "x"}], "n": 3}, "invalid", False,
                       unsupported_today=True)
        elif case == 4:
            self.whole(P, None, "invalid", False, raw=b"\xff\xfe not utf8 \xc3")
        elif case == 5:
            # Unsupported today (api.rs refuses logit_bias), not invalid.
            self.whole("/v1/completions", {"model": MODEL[0], "prompt": "x", "logit_bias": {"1": 5}}, "invalid", False,
                       unsupported_today=True)
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


# ---------------------------------------------------------------- priming

def prime(cfg):
    """Before the measured window: run every batch mix the slots can form, so
    the server's one-time allocations are made up front. On CUDA the batched
    decode step is captured as a graph per (rows, greedy rows, sampled rows)
    on its first use (cuda/qwen2/slots.rs, at most MAX_GRAPHS = 64 kept), and
    each instantiated graph takes driver memory: a mix first seen hours into
    the soak shows as a VRAM step that isn't a leak. Each mix: nb streams sent
    together, ng of them greedy, twice."""
    n = 0
    for nb in range(1, cfg.slots + 1):
        for ng in range(nb + 1):
            for rep in range(2):
                def one(k):
                    b = {"model": MODEL[0], "max_tokens": 24, "stream": True,
                         "messages": [{"role": "user", "content": f"prime {nb} {ng} {rep} {k}: count to ten"}]}
                    b.update(temperature=0) if k < ng else b.update(temperature=0.7, seed=k)
                    try:
                        request(cfg.port, "POST", "/v1/chat/completions", b, timeout=cfg.req_timeout)
                    except OSError:
                        pass
                ts = [threading.Thread(target=one, args=(k,)) for k in range(nb)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                n += nb
    return n


def slots_idle(cfg):
    """/slots after the load: how many slots (and the solo path) are still busy, and the queue depth."""
    try:
        st, _, rb = request(cfg.port, "GET", "/slots", None, timeout=10)
        j = json.loads(rb)
        busy = sum(1 for x in j.get("slots", []) if x.get("busy")) + int(bool((j.get("solo") or {}).get("busy")))
        return {"busy": busy, "slots": len(j.get("slots", [])), "queue_depth": j.get("queue_depth", 0)}
    except (OSError, ValueError) as e:
        return {"error": repr(e)}


# ---------------------------------------------------------------- sampling

def proc_stats(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k, _, v = line.partition(":")
            if k in ("VmRSS", "Threads", "VmHWM", "RssAnon", "RssFile"):
                d[k] = int(v.split()[0])
        d["fds"] = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        pass
    return d


def mib(d, k):
    return round(d.get(k, -1024) / 1024, 1)


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


FIELDS = ["t_s", "rss_mib", "anon_mib", "file_mib", "hwm_mib", "vram_mib", "vram_src", "fds", "threads", "requests",
          "lat_p50_ms", "lat_p99_ms", "ttft_p50_ms", "ttft_p99_ms"] + CLASSES + ["alive"]


def fit(ts, ys):
    """Least-squares (slope, standard error of the slope), both per hour, of ys over ts (seconds).
    The SE comes from the residuals and assumes them independent. A step or slow drift makes them
    autocorrelated, so the SE is if anything optimistic, never generous."""
    n = len(ts)
    if n < 3:
        return 0.0, float("inf")
    mt, my = sum(ts) / n, sum(ys) / n
    sxx = sum((t - mt) ** 2 for t in ts)
    if sxx == 0:
        return 0.0, float("inf")
    b = sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / sxx
    res = [y - (my + b * (t - mt)) for t, y in zip(ts, ys)]
    se = math.sqrt(sum(r * r for r in res) / (n - 2) / sxx)
    return b * 3600, se * 3600


def slope_per_h(ts, ys):
    return fit(ts, ys)[0]


def memory_check(ts, ys, bound_24h, min_window_h):
    """The memory verdict for one series after warm-up.
    PASS: growth_24h + 2 SE < bound. FAIL: growth_24h - 2 SE > bound. Otherwise INCONCLUSIVE.
    Judged only over at least min_window_h hours after warm-up; shorter windows are reported
    with status NOT-JUDGED (they don't decide the verdict)."""
    s, se = fit(ts, ys)
    g, e = s * 24, se * 24
    window_h = (ts[-1] - ts[0]) / 3600 if len(ts) > 1 else 0.0
    q = max(1, len(ys) // 4)
    out = {"slope_mib_per_h": round(s, 3), "se_mib_per_h": round(se, 3), "growth_24h_mib": round(g, 1),
           "growth_24h_2se_mib": [round(g - 2 * e, 1), round(g + 2 * e, 1)], "bound_24h_mib": round(bound_24h, 1),
           "window_h": round(window_h, 2), "n": len(ys), "min_mib": min(ys), "max_mib": max(ys),
           "median_mib": round(statistics.median(ys), 1),
           "last_vs_first_quarter_mib": round(statistics.median(ys[-q:]) - statistics.median(ys[:q]), 1)}
    if g + 2 * e < bound_24h:
        st = "PASS"
    elif g - 2 * e > bound_24h:
        st = "FAIL"
    else:
        st = "INCONCLUSIVE"
    out["status"] = st if window_h >= min_window_h else "NOT-JUDGED"
    out["would_be"] = st
    return out


def memory_checks(rows, cfg):
    """{name: memory_check} for RSS (RssAnon when sampled, else VmRSS) and per-pid VRAM."""
    warm = cfg.warmup_min * 60
    post = [r for r in rows if float(r["t_s"]) >= warm and r.get("alive", "1") == "1"]
    out = {}
    if len(post) < 3:
        return {"rss": {"status": "NOT-JUDGED", "why": "fewer than 3 samples after warm-up"}}
    ts = [float(r["t_s"]) for r in post]
    anon = [float(r["anon_mib"]) for r in post if "anon_mib" in r and float(r["anon_mib"]) >= 0]
    key = "anon_mib" if len(anon) == len(post) else "rss_mib"
    ys = [float(r[key]) for r in post]
    bound = max(cfg.rss_floor_mib, cfg.rss_frac * statistics.median(ys))
    out["rss"] = dict(memory_check(ts, ys, bound, cfg.min_judge_h), series="RssAnon" if key == "anon_mib" else "VmRSS")
    if key == "anon_mib":
        out["rss"]["vmrss_slope_mib_per_h"] = round(slope_per_h(ts, [float(r["rss_mib"]) for r in post]), 3)
    vp = [r for r in post if float(r["vram_mib"]) >= 0 and (r["vram_src"] == "pid" or cfg.vram_card)]
    if len(vp) >= 3:
        out["vram"] = dict(memory_check([float(r["t_s"]) for r in vp], [float(r["vram_mib"]) for r in vp],
                                        cfg.vram_floor_mib, cfg.min_judge_h), src=vp[-1]["vram_src"])
    return out


def verdict(rows, summary, cfg):
    """Fill summary['checks'] and summary['verdict'] (PASS / FAIL / INCONCLUSIVE) from the samples."""
    checks = {k: dict(v, pass_=v["status"]) for k, v in memory_checks(rows, cfg).items()}
    tot = summary["totals"]
    bad = {k: v for k, v in tot.items() if k.split(":")[1] in BAD}
    st = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    checks["errors"] = {"bad": bad, "pass_": st(not bad)}
    n503 = sum(v for k, v in tot.items() if k.endswith(":503"))
    nreq = sum(tot.values())
    checks["503"] = {"count": n503, "fraction": round(n503 / max(1, nreq), 4),
                     "ok": sum(v for k, v in tot.items() if k.endswith(":ok")), "requests": nreq,
                     "note": "every 503 body is checked against the documented back-pressure messages "
                             "(otherwise class 503-undocumented, failing); the fraction is reported, not capped",
                     "pass_": "PASS"}
    checks["alive"] = {"pass_": st(summary["alive_at_end"])}
    checks["panic"] = {"lines": summary["panic_lines"], "pass_": st(not summary["panic_lines"])}
    checks["identity"] = {"runs": summary["identity_runs"], "mismatches": summary["identity_mismatches"],
                          "pass_": st(summary["identity_runs"] >= 2 and not summary["identity_mismatches"])}
    b, e = summary["idle_before"], summary["idle_after"]
    checks["fds_threads"] = {"before": b, "after": e,
                             "pass_": st(e["fds"] <= b["fds"] + 2 and e["threads"] <= b["threads"])}
    sl = summary.get("slots_after")
    if sl is not None:
        checks["slots_idle"] = dict(sl, pass_=st(sl.get("busy", 1) == 0 and sl.get("queue_depth", 1) == 0))
    c = summary.get("cancels")
    if c is not None:
        checks["cancels"] = dict(c, pass_="REPORTED")
    for v in checks.values():
        v["status"] = v.pop("pass_")
    sts = [v["status"] for v in checks.values()]
    summary["checks"] = checks
    summary["verdict"] = "FAIL" if "FAIL" in sts else "INCONCLUSIVE" if "INCONCLUSIVE" in sts else "PASS"
    notj = [k for k, v in checks.items() if v["status"] == "NOT-JUDGED"]
    summary["verdict_note"] = (f"{', '.join(notj)} not judged: window under {cfg.min_judge_h} h after warm-up"
                               if notj else "")
    summary["pass"] = summary["verdict"] == "PASS"


def rows_from(path):
    """Sample rows from a samples.csv or a job log with CSV, lines."""
    lines = [ln[4:] if ln.startswith("CSV,") else ln for ln in open(path, errors="replace")]
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("t_s,"))
    return list(csv.DictReader([ln for ln in lines[hdr:] if ln[:1].isdigit() or ln.startswith("t_s,")]))


def rejudge(log, warmup_min=20, min_judge_h=4.0):
    """The memory checks of a finished soak, from its CSV rows, with the current logic."""
    class C:
        pass
    cfg = C()
    cfg.warmup_min, cfg.min_judge_h, cfg.rss_floor_mib, cfg.rss_frac = warmup_min, min_judge_h, 64.0, 0.02
    cfg.vram_floor_mib, cfg.vram_card = 16.0, False
    return memory_checks(rows_from(log), cfg)


def summarise(path, bucket_min=30):
    """A markdown table of a samples.csv (or a job log with CSV, lines) in bucket_min buckets."""
    rows = rows_from(path)
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
        v = s.get("verdict", "PASS" if s["pass"] else "FAIL")
        print(v, s.get("verdict_note", ""))
        return {"PASS": 0, "FAIL": 1}.get(v, 2)
    if len(sys.argv) > 1 and sys.argv[1] == "rejudge":
        a = sys.argv[2:]
        w = float(a[1]) if len(a) > 1 else 20
        print(json.dumps(rejudge(a[0], w), indent=1))
        return 0
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
    ap.add_argument("--probe-ttft-s", type=float, default=30.0,
                    help="most seconds a probe may wait for its first bytes while a slow reader is blocked")
    ap.add_argument("--think-s", type=float, default=1.0, help="most idle time between a client's requests")
    ap.add_argument("--req-timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rss-floor-mib", type=float, default=64)
    ap.add_argument("--min-judge-h", type=float, default=4.0,
                    help="memory is judged (PASS/FAIL/INCONCLUSIVE) only over at least this many hours after "
                         "warm-up; a shorter window is reported, NOT-JUDGED")
    ap.add_argument("--rss-frac", type=float, default=0.02)
    ap.add_argument("--vram-floor-mib", type=float, default=16)
    ap.add_argument("--no-prime", dest="prime", action="store_false",
                    help="skip priming every batch mix before the load (see prime())")
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
        if cfg.prime:
            t0 = time.time()
            summary["primed_requests"] = prime(cfg)
            print(f"primed {summary['primed_requests']} requests in {time.time() - t0:.0f}s", flush=True)
        ident0 = identity_run(cfg)
        summary["identity_ref"] = ident0
        summary["identity_runs"] = 1
        time.sleep(2)
        d0 = proc_stats(srv.pid)
        summary["idle_before"] = {"rss_mib": mib(d0, "VmRSS"), "anon_mib": mib(d0, "RssAnon"),
                                  "threads": d0.get("Threads", -1), "fds": d0.get("fds", -1)}
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
            d = proc_stats(srv.pid)
            vr, vsrc = vram_mib(srv.pid)
            row = {"t_s": round(time.time() - t_start, 1), "rss_mib": mib(d, "VmRSS"),
                   "anon_mib": mib(d, "RssAnon"), "file_mib": mib(d, "RssFile"), "hwm_mib": mib(d, "VmHWM"),
                   "vram_mib": vr, "vram_src": vsrc, "fds": d.get("fds", -1), "threads": d.get("Threads", -1),
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
        d1 = proc_stats(srv.pid)
        summary["idle_after"] = {"rss_mib": mib(d1, "VmRSS"), "anon_mib": mib(d1, "RssAnon"),
                                 "threads": d1.get("Threads", -1), "fds": d1.get("fds", -1),
                                 "hwm_mib": mib(d1, "VmHWM")}
        summary["slots_after"] = slots_idle(cfg)
        summary["totals"] = dict(sorted(stats.total.items()))
        summary["requests"] = stats.requests
        summary["lat_max_s"] = stats.lat_max
        pr = sorted(stats.probes)
        summary["slow_reader_probes"] = {"n": len(pr), "p50_s": pr[len(pr) // 2] if pr else None,
                                         "max_s": pr[-1] if pr else None}
        summary["findings"] = stats.findings
        summary["wall_h"] = round((time.time() - t_start) / 3600, 3)
    finally:
        alive = srv.alive()
        srv.stop()
        summary.setdefault("alive_at_end", alive)
    log = open(srv.log_path, errors="replace").read()
    summary["panic_lines"] = [ln for ln in log.splitlines() if re.search(r"panicked|RUST_BACKTRACE", ln)][:20]
    # Server-side cancellation: the engine logs "#N cancelled after K tokens" when it reaps a request whose
    # client went away. A client disconnect after the generation already ended has nothing to cancel, so
    # the ratio is reported, not judged.
    nc = len(re.findall(r"cancelled after \d+ tokens", log))
    nd = summary.get("totals", {}).get("disconnect:cancelled", 0)
    summary["cancels"] = {"client_disconnects": nd, "server_cancelled": nc,
                          "ratio": round(nc / nd, 3) if nd else None}
    summary.setdefault("totals", {})
    summary.setdefault("idle_before", {"fds": 0, "threads": 0})
    summary.setdefault("idle_after", {"fds": 1 << 30, "threads": 1 << 30})
    verdict(rows, summary, cfg)
    json.dump(summary, open(os.path.join(cfg.out, "summary.json"), "w"), indent=1)
    print("SUMMARY " + json.dumps({k: summary[k] for k in ("checks", "requests", "totals", "wall_h") if k in summary}))
    for fd in summary.get("findings", [])[:30]:
        print("FINDING", json.dumps(fd))
    print("VERDICT", summary["verdict"], summary["verdict_note"], flush=True)
    return {"PASS": 0, "FAIL": 1}.get(summary["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
