# A fake OpenAI-compatible server for the self-tests: /v1/models; /v1/chat/completions (JSON or SSE; with
# include_usage, "usage": null on every chunk and a usage-only last chunk, as OpenAI does; prompt_tokens = 10 x messages); /v1/responses (SSE: created, a text delta, completed with usage). Synthetic.
#   fake_upstream.py PORT [--text-only]
# --text-only: a streamed reply never calls a tool, so a real harness pointed at it (capture_dry.sh) gets a plain
# answer and finishes its task instead of looping on tool calls.
import http.server, json, socketserver, sys

TEXT_ONLY = "--text-only" in sys.argv[2:]


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
            # The full event sequence of one text message (codex 0.166 refuses a delta without its item, and a
            # completed response without usage.total_tokens).
            u = {"input_tokens": 50, "output_tokens": 3, "total_tokens": 53, "input_tokens_details": {"cached_tokens": 0},
                 "output_tokens_details": {"reasoning_tokens": 0}}
            part = {"type": "output_text", "text": "done", "annotations": []}
            item = {"id": "msg_1", "type": "message", "status": "completed", "role": "assistant", "content": [part]}
            done = {"id": "resp_1", "object": "response", "status": "completed", "output": [item], "usage": u}
            at = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
            ev = [("response.created", {"response": dict(done, status="in_progress", output=[], usage=None)}),
                  ("response.output_item.added", {"output_index": 0, "item": dict(item, status="in_progress", content=[])}),
                  ("response.content_part.added", dict(at, part=dict(part, text=""))),
                  ("response.output_text.delta", dict(at, delta="done")),
                  ("response.output_text.done", dict(at, text="done")),
                  ("response.content_part.done", dict(at, part=part)),
                  ("response.output_item.done", {"output_index": 0, "item": item}),
                  ("response.completed", {"response": done})]
            ev = [(e, dict(d, type=e, sequence_number=i)) for i, (e, d) in enumerate(ev)]
            body = b"".join(f"event: {e}\ndata: {json.dumps(d)}\n\n".encode() for e, d in ev)
            return self._send(200, body, "text/event-stream")
        n = 10 * len(b.get("messages", []))
        u = {"prompt_tokens": n, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": max(0, n - 10)}}
        if b.get("stream"):
            parts = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello secret text"}}]},
                     {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                                       "function": {"name": "bash", "arguments": "{}"}}]}}]}]
            if b.get("tools") is None or TEXT_ONLY:
                parts = parts[:1]
            parts.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            if (b.get("stream_options") or {}).get("include_usage"):
                # as OpenAI does: every chunk carries "usage": null, then one usage-only chunk
                parts = [dict(x, usage=None) for x in parts] + [{"choices": [], "usage": u}]
            body = b"".join(b"data: " + json.dumps(p).encode() + b"\n\n" for p in parts) + b"data: [DONE]\n\n"
            return self._send(200, body, "text/event-stream")
        o = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "secret answer"}, "finish_reason": "stop"}],
             "usage": u}
        self._send(200, json.dumps(o).encode())


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


S(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
