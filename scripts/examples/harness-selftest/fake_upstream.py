# A fake OpenAI chat server for the proxy self-test: non-stream JSON and SSE with usage; prompt_tokens = 10 x messages.
import http.server, json, socketserver, sys
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = 10 * len(b.get("messages", []))
        u = {"prompt_tokens": n, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": max(0, n - 10)}}
        if b.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Content-Length", "-1"); self.end_headers()
            parts = [{"choices": [{"delta": {"content": "hello secret text"}}]},
                     {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "bash", "arguments": "{}"}}]}}]}]
            if (b.get("stream_options") or {}).get("include_usage"): parts.append({"choices": [], "usage": u})
            body = b"".join(b"data: " + json.dumps(p).encode() + b"\n\n" for p in parts) + b"data: [DONE]\n\n"
            self.wfile.write(body); self.wfile.flush(); self.close_connection = True
        else:
            o = json.dumps({"choices": [{"message": {"role": "assistant", "content": "secret answer"}}], "usage": u}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(o))); self.end_headers(); self.wfile.write(o)
class S(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads = True
S(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
