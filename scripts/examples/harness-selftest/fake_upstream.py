# A fake OpenAI-compatible server for the self-tests: /v1/models; /v1/chat/completions (JSON or SSE with usage when
# asked; prompt_tokens = 10 x messages); /v1/responses (SSE: created, a text delta, completed with usage). Synthetic.
import http.server, json, socketserver, sys


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send(200, json.dumps({"object": "list", "data": [{"id": "axon-7b", "object": "model"}]}).encode())
        self._send(200, b'{"status":"ok"}')

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path.startswith("/v1/responses"):
            u = {"input_tokens": 50, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 0}}
            done = {"id": "resp_1", "object": "response", "status": "completed", "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}], "usage": u}
            ev = [("response.created", {"type": "response.created", "response": dict(done, status="in_progress", output=[])}),
                  ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "done"}),
                  ("response.completed", {"type": "response.completed", "response": done})]
            body = b"".join(f"event: {e}\ndata: {json.dumps(d)}\n\n".encode() for e, d in ev)
            return self._send(200, body, "text/event-stream")
        n = 10 * len(b.get("messages", []))
        u = {"prompt_tokens": n, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": max(0, n - 10)}}
        if b.get("stream"):
            parts = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello secret text"}}]},
                     {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                                       "function": {"name": "bash", "arguments": "{}"}}]}}]}]
            if b.get("tools") is None:
                parts = parts[:1]
            parts.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            if (b.get("stream_options") or {}).get("include_usage"):
                parts.append({"choices": [], "usage": u})
            body = b"".join(b"data: " + json.dumps(p).encode() + b"\n\n" for p in parts) + b"data: [DONE]\n\n"
            return self._send(200, body, "text/event-stream")
        o = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "secret answer"}, "finish_reason": "stop"}],
             "usage": u}
        self._send(200, json.dumps(o).encode())


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


S(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
