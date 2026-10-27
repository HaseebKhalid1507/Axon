#!/usr/bin/env python3
"""Recording reverse proxy for agent-harness traces: NUMBERS ONLY (stdlib only).

    agent_proxy.py --listen PORT --upstream http://127.0.0.1:UPORT --out FILE [--harness NAME]

Put it between a harness (opencode, pi, codex, crush) and `axon serve` (or any OpenAI / Ollama-compatible server).
It forwards every request and response byte for byte, streaming included. The one exception: an OpenAI chat request
with `stream: true` that lacks `stream_options.include_usage` gets it added, because the token counts come from the
response's `usage`. The record says so (`inj`).

Per request it appends ONE JSON line to FILE (spec: specs/agent-harness-traces.md section 2):
  t_ms, route (enum), stream, status, ttft_ms, total_ms, prompt_tokens, completion_tokens, cached_tokens,
  n_msgs, roles {system,user,assistant,tool,other}, n_tools, n_tool_calls_req, n_tool_calls_resp, max_tokens,
  has_prev, store, keep_alive_s, conn, harness (enum), inj, h (list of 16-hex message digests), b (list of message
  byte lengths), prev (16-hex digest of previous_response_id or null), resp (16-hex digest of the response id).

PRIVACY (the #160 rule, docs/log/2026-10-27-agent-replay.md; this file adds no exception):
- No request or response text, tool names or arguments, ids, model names or headers are written.
- Message structure is kept only as HMAC-SHA256(capture key, role | canonical message JSON), first 8 bytes. The key
  is random per run, held in memory only, never written, so the digests can't be reversed or linked across
  captures. `scripts/agent_trace_proxy.py` reads them to find prefix relations and drops them; its output passes
  `agent_trace.py --check`.
- The capture file must be outside any git tree (refused otherwise). Default directory: ~/.cache/axon-agent/raw/.
"""
import argparse, hashlib, hmac, http.client, http.server, json, os, re, secrets, socketserver, subprocess, sys, threading, time, urllib.parse

ROUTES = {
    "/v1/chat/completions": "chat", "/v1/completions": "compl", "/v1/responses": "resp",
    "/api/chat": "ochat", "/api/generate": "ogen",
}
HARNESSES = ("opencode", "pi", "codex", "crush", "other")
TASKS = ("bugfix", "feature", "explain", "resume", "compact", "subagent", "other")
# Tool-result text is matched in memory to COUNT errors; nothing of it is kept. "harness" errors are the harness
# rejecting a call (unknown tool, bad arguments); "any" also counts ordinary command failures (tests failing etc.).
ERR_HARNESS = re.compile(r"(?i)(unknown tool|no such tool|tool .{0,40}not found|invalid (tool|argument|input|param)|"
                         r"failed to parse|could not parse|missing required|validation error|schema)")
ERR_ANY = re.compile(r"(Traceback|Error\b|error:|FAILED|command not found|No such file|exit code [1-9])")
TEXT_TC = re.compile(r'(<tool_call>|"name"\s*:\s*"[A-Za-z_]+"\s*,\s*"(arguments|parameters)")')
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade", "content-length",
       "host"}


def in_git_tree(path):
    d = os.path.dirname(os.path.abspath(path)) or "."
    while not os.path.isdir(d):
        d = os.path.dirname(d)
    r = subprocess.run(["git", "-C", d, "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


class Capture:
    def __init__(self, out, harness):
        self.key = secrets.token_bytes(32)  # never written
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.f = open(out, "a")
        self.harness = harness if harness in HARNESSES else "other"
        self.conns = {}
        self.seq = 0

    def next_seq(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def dig(self, *parts):
        m = hmac.new(self.key, digestmod=hashlib.sha256)
        for p in parts:
            m.update(p if isinstance(p, bytes) else str(p).encode())
            m.update(b"\x00")
        return m.hexdigest()[:16]

    def conn_id(self, addr):
        with self.lock:
            return self.conns.setdefault(addr, len(self.conns))

    def write(self, rec):
        with self.lock:
            self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.f.flush()


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def shape(cap, route, body):
    """Counts and digests of a request body (no text kept)."""
    r = dict(n_msgs=0, roles=dict(system=0, user=0, assistant=0, tool=0, other=0), n_tools=0, n_tool_calls_req=0,
             max_tokens=None, has_prev=False, store=None, keep_alive_s=None, h=[], b=[], prev=None, stream=False,
             n_tool_res=0, n_tool_err_harness=0, n_tool_err_any=0)
    if not isinstance(body, dict):
        return r
    r["stream"] = bool(body.get("stream", route in ("ochat", "ogen")))  # Ollama streams by default
    r["n_tools"] = len(body.get("tools") or [])
    mt = body.get("max_tokens", body.get("max_completion_tokens", body.get("max_output_tokens")))
    if mt is None and isinstance(body.get("options"), dict):
        mt = body["options"].get("num_predict")
    r["max_tokens"] = mt if isinstance(mt, int) else None
    if route == "resp":
        items = []
        if body.get("instructions"):
            items.append({"role": "system", "content": body["instructions"]})
        inp = body.get("input")
        items += [{"role": "user", "content": inp}] if isinstance(inp, str) else list(inp or [])
        r["has_prev"] = bool(body.get("previous_response_id"))
        r["prev"] = cap.dig("rid", body["previous_response_id"]) if r["has_prev"] else None
        r["store"] = body.get("store", True)
    elif route in ("ogen", "compl"):
        p = body.get("prompt", "")
        items = ([{"role": "system", "content": body["system"]}] if body.get("system") else []) + \
            [{"role": "user", "content": p if isinstance(p, str) else canon(p).decode()}]
    else:
        items = list(body.get("messages") or [])
    if route in ("ochat", "ogen") and "keep_alive" in body:
        ka = body["keep_alive"]
        if isinstance(ka, (int, float)):
            r["keep_alive_s"] = ka
        elif isinstance(ka, str):
            units = {"s": 1, "m": 60, "h": 3600}
            try:
                r["keep_alive_s"] = float(ka[:-1]) * units[ka[-1]] if ka[-1] in units else float(ka)
            except (ValueError, IndexError):
                r["keep_alive_s"] = -1
    for m in items:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("type") or "other"
        role = role if role in ("system", "user", "assistant", "tool") else ("tool" if "output" in str(role) else "other")
        r["roles"][role] += 1
        r["n_tool_calls_req"] += len(m.get("tool_calls") or []) + (1 if m.get("type") == "function_call" else 0)
        c = canon(m)
        r["h"].append(cap.dig("msg", c))
        r["b"].append(len(c))
    # New tool results only: those after the last assistant turn (the history re-sends older ones every request,
    # which must not be counted again).
    tail = []
    for m in items:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant" or m.get("type") in ("function_call", "message"):
            tail = []
        elif m.get("role") == "tool" or m.get("type") == "function_call_output":
            tail.append(m)
    for m in tail:
        txt = m.get("content", m.get("output", ""))
        txt = txt if isinstance(txt, str) else canon(txt).decode("utf-8", "replace")
        r["n_tool_res"] += 1
        r["n_tool_err_harness"] += bool(ERR_HARNESS.search(txt))
        r["n_tool_err_any"] += bool(ERR_ANY.search(txt))
    r["n_msgs"] = len(r["h"])
    return r


def usage_of(route, o, rec):
    """Fill token counts / tool calls / response id from one parsed JSON object of the response."""
    if not isinstance(o, dict):
        return
    if route == "resp":
        resp = o.get("response") if isinstance(o.get("response"), dict) else o
        u = resp.get("usage")
        if isinstance(u, dict):
            rec["prompt_tokens"] = u.get("input_tokens")
            rec["completion_tokens"] = u.get("output_tokens")
            rec["cached_tokens"] = (u.get("input_tokens_details") or {}).get("cached_tokens")
        if resp.get("id") and resp.get("object") == "response":
            rec["resp"] = rec["_cap"].dig("rid", resp["id"])
        for it in resp.get("output") or []:
            if isinstance(it, dict) and it.get("type") == "function_call":
                k = it.get("call_id") or len(rec["_tc"])
                rec["_tc"].add(k)
                if isinstance(it.get("arguments"), str):
                    rec["_args"][k] = it["arguments"]
            elif isinstance(it, dict) and it.get("type") == "message":
                for c in it.get("content") or []:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        rec["_text"].append(c["text"])
        return
    if route in ("ochat", "ogen"):
        if o.get("done"):
            rec["prompt_tokens"] = o.get("prompt_eval_count")
            rec["completion_tokens"] = o.get("eval_count")
        msg = o.get("message") or {}
        for i, _ in enumerate(msg.get("tool_calls") or []):
            rec["_tc"].add(("o", len(rec["_tc"])))
        return
    u = o.get("usage")
    if isinstance(u, dict):
        rec["prompt_tokens"] = u.get("prompt_tokens")
        rec["completion_tokens"] = u.get("completion_tokens")
        rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    for ch in o.get("choices") or []:
        msg = ch.get("message") or ch.get("delta") or {}
        if isinstance(msg.get("content"), str):
            rec["_text"].append(msg["content"])
        for tc in msg.get("tool_calls") or []:
            k = tc.get("index", tc.get("id", len(rec["_tc"])))
            rec["_tc"].add(k)
            a = (tc.get("function") or {}).get("arguments")
            if isinstance(a, str):
                rec["_args"][k] = rec["_args"].get(k, "") + a


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    cap = None
    up = None

    def log_message(self, *a):  # no access log (it would carry paths and agents)
        pass

    def _body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            out = b""
            while True:
                n = int(self.rfile.readline().strip().split(b";")[0], 16)
                if n == 0:
                    self.rfile.readline()
                    return out
                out += self.rfile.read(n)
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _proxy(self):
        t_arr = time.time()
        path = self.path.split("?")[0]
        if path == "/_mark":
            # The capture script's task boundaries: /_mark?task=<enum>&round=<int> (enums only).
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            task = (q.get("task") or ["other"])[0]
            rnd = int((q.get("round") or ["0"])[0])
            self.cap.write(dict(mark=task if task in TASKS else "other", round=rnd,
                                t_ms=int((t_arr - self.cap.t0) * 1000), seq=self.cap.next_seq()))
            self._body()
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        route = ROUTES.get(path)
        raw = self._body()
        body = None
        inj = False
        if route and raw:
            try:
                body = json.loads(raw)
            except ValueError:
                body = None
        rec = None
        if route:
            rec = dict(t_ms=int((t_arr - self.cap.t0) * 1000), seq=self.cap.next_seq(), route=route, conn=self.cap.conn_id(self.client_address),
                       harness=self.cap.harness, prompt_tokens=None, completion_tokens=None, cached_tokens=None,
                       resp=None, _cap=self.cap, _tc=set(), _args={}, _text=[])
            rec.update(shape(self.cap, route, body))
            if route == "chat" and isinstance(body, dict) and body.get("stream") and \
                    not (body.get("stream_options") or {}).get("include_usage"):
                body.setdefault("stream_options", {})["include_usage"] = True
                raw = json.dumps(body).encode()
                inj = True
        u = urllib.parse.urlsplit(self.up)
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        hdrs["Content-Length"] = str(len(raw))
        try:
            c.request(self.command, self.path, body=raw if raw or self.command in ("POST", "PUT") else None, headers=hdrs)
            r = c.getresponse()
        except OSError as e:
            self.send_error(502, "upstream: %s" % type(e).__name__)
            return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        t_first = None
        buf = b""
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            if t_first is None:
                t_first = time.time()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
            if rec is not None:
                buf += chunk
                # Parse complete lines as they come (SSE `data: {...}` or NDJSON), keep only the tail.
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    ln = ln.strip()
                    if ln.startswith(b"data:"):
                        ln = ln[5:].strip()
                    if ln.startswith(b"{"):
                        try:
                            usage_of(route, json.loads(ln), rec)
                        except ValueError:
                            pass
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        c.close()
        if rec is None:
            return
        tail = buf.strip()
        if tail.startswith(b"data:"):
            tail = tail[5:].strip()
        if tail.startswith(b"{"):
            try:
                usage_of(route, json.loads(tail), rec)
            except ValueError:
                pass
        t_end = time.time()
        args, text = rec.pop("_args"), "".join(rec.pop("_text"))
        bad = 0
        for a in args.values():
            try:
                json.loads(a or "{}")
            except ValueError:
                bad += 1
        ntc = len(rec["_tc"])
        rec.update(tc_badjson=bad, tc_text=int(ntc == 0 and bool(TEXT_TC.search(text))))
        rec.update(status=r.status, inj=inj, n_tool_calls_resp=len(rec.pop("_tc")),
                   ttft_ms=None if t_first is None else int((t_first - t_arr) * 1000),
                   total_ms=int((t_end - t_arr) * 1000))
        rec.pop("_cap")
        self.cap.write(rec)

    do_POST = do_GET = do_PUT = do_DELETE = _proxy


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--harness", default="other", choices=HARNESSES)
    a = ap.parse_args()
    if in_git_tree(a.out):
        sys.exit(f"agent_proxy: {a.out} is inside a git tree: captures never go into a repo")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Handler.cap, Handler.up = Capture(a.out, a.harness), a.upstream.rstrip("/")
    srv = Server(("127.0.0.1", a.listen), Handler)
    print(f"agent_proxy: 127.0.0.1:{a.listen} -> {a.upstream}, capture {a.out} ({a.harness})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
