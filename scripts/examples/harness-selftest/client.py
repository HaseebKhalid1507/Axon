import json, sys, urllib.request
def post(port, body):
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}))
    return r.read()
P = int(sys.argv[1]); sysm = {"role": "system", "content": "You are a SECRET coding agent with tools " * 20}
tools = [{"type": "function", "function": {"name": "bash"}}]
m = [sysm, {"role": "user", "content": "fix the SECRET bug"}]
for turn in range(3):                       # one conversation, growing
    out = post(P, {"messages": m, "tools": tools, "stream": turn != 1})
    assert b"secret" in out, out[:100]
    m = m + [{"role": "assistant", "content": f"step {turn}"}, {"role": "tool", "content": "SECRET output " * 30}]
post(P, {"messages": [{"role": "user", "content": "title for: SECRET"}], "max_tokens": 20})   # side request
post(P, {"messages": [sysm, {"role": "user", "content": "SECRET subtask"}], "tools": tools})   # shares the system prompt
print("client ok")
