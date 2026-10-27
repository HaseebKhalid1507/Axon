#!/bin/bash
# Self-test of agent_proxy.py + agent_trace_proxy.py on SYNTHETIC traffic (no model, no real data): a fake
# OpenAI chat server, the proxy in front, a client playing a 3-turn conversation (stream / non-stream, tool calls),
# a side request and a subagent sharing the system prompt. Checks: the capture holds none of the request text;
# the converted trace passes agent_trace.py --check and recovers the chain, the side request and the subagent.
set -eu
H=$(cd "$(dirname "$0")" && pwd); S=$H/../..; OUT=${HOME}/.cache/axon-agent/raw/selftest-$$.jsonl; TR=${HOME}/.cache/axon-agent/traces/harness-selftest-$$.json
python3 $H/fake_upstream.py 19921 & U=$!; python3 $S/agent_proxy.py --listen 19922 --upstream http://127.0.0.1:19921 --out $OUT --harness pi & P=$!
trap 'kill $U $P 2>/dev/null' EXIT; sleep 1
python3 $H/client.py 19922
sleep 0.3
if grep -qi "secret\|bash\|coding agent\|frobnicate\|traceback" $OUT; then echo "SELFTEST FAIL: request text in the capture"; exit 1; fi
python3 $S/agent_trace_proxy.py $OUT --out $TR
python3 - "$TR" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); r = d["requests"]
assert [x["session"] for x in r] == [0, 0, 0, 1, 2], r
assert [x["kind"] for x in r] == ["main", "main", "main", "side", "main"], r
assert r[1]["shared"][0]["with"] == "session" and r[4]["shared"][0]["with"] == "sys", r
assert (r[4]["parent"], r[4]["depth"]) == (0, 1), r[4]
sm = d["summary"]
assert set(sm["tasks"]) == {"bugfix", "explain"}, sm["tasks"]
assert sm["tasks"]["bugfix"]["requests"] == 3 and sm["tasks"]["explain"]["requests"] == 2, sm["tasks"]
assert sm["tool_err_harness"] == 1 and sm["tool_err_any"] == 1 and sm["tool_results"] == 2, sm  # new results only
assert sm["tool_calls"] >= 1 and sm["tc_badjson"] == 0 and sm["reuse_pct"] is not None, sm
print("SELFTEST OK")
PY
rm -f $TR
