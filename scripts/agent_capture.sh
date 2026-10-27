#!/bin/bash
# agent_capture.sh HARNESS [ROUNDS]: capture one agent harness's traffic against axon serve (specs/agent-harness-
# traces.md section 4), ONE job: axon serve (Qwen2.5-7B-Instruct Q4_K_M, --ctx 32768 --kv q8 --parallel 4; the
# harness's titles / summaries go to the same model) behind scripts/agent_proxy.py; per round, each task in a fresh
# copy of the synthetic ledger repo (examples/harness-repo/make_repo.py):
#   bugfix   the planted bug; tests run after
#   feature  a --format csv option across cli / reports / formats + a test; tests run after
#   resume   the harness's continue on the feature session: add json + README; tests run after
#   explain  read-only: trace an order through the code
#   compact  read every file into NOTES.md with a SMALL context window in the harness config (forces compaction)
#   subagent parallel review via subagents (opencode, crush; skipped where the harness has none)
# Each harness runs with a scratch HOME / XDG dirs / XDG_RUNTIME_DIR and its own config env, dummy keys: no real
# config is read or written. Then agent_trace_proxy.py converts the capture (privacy check inside) and prints the
# summary. Output lines: TASK <harness> <task> r<round> rc <rc> <secs>s tests <OK|FAIL|-> ; CAPTURE <harness> DONE rc.
set -u
HN=${1:?harness: pi | opencode | codex | crush}; ROUNDS=${2:-3}
WT=$(cd "$(dirname "$0")/.." && pwd); S=$WT/scripts
SMOKE=${AXON_CAPTURE_SMOKE:-}   # a URL: no serve, the proxy in front of that upstream, bugfix r1 only (config smoke)
[ -n "$SMOKE" ] || A=${AXON_CAPTURE_BIN:?AXON_CAPTURE_BIN: an axon binary}
MD=${AXON_MODELS:-$HOME/Projects/axon/models}
M=$MD/Qwen2.5-7B-Instruct-Q4_K_M.gguf
TS=$(date +%m%d-%H%M%S); C=$HOME/.cache/axon-harness/cap/$HN-$TS; mkdir -p $C
RAW=$HOME/.cache/axon-agent/raw/$HN-$TS.jsonl
SP=$((19100 + RANDOM % 300)); PP=$((SP + 400)); PX=http://127.0.0.1:$PP
log() { echo "[cap $HN +${SECONDS}s] $*"; }
mark() { curl -s -X POST "$PX/_mark?task=$1&round=$2" > /dev/null; }
TO=${AXON_CAPTURE_TASK_S:-600}

if [ -n "$SMOKE" ]; then
  UP=$SMOKE; SPID=; ROUNDS=1
else
  ( exec $A serve $M --device cuda --host 127.0.0.1 --port $SP --parallel 4 --ctx 32768 --kv q8 --alias axon-7b --quiet ) \
    > $C/serve.log 2>&1 &
  SPID=$!
  for i in $(seq 180); do curl -sf http://127.0.0.1:$SP/health > /dev/null 2>&1 && break; sleep 1; done
  curl -sf http://127.0.0.1:$SP/health > /dev/null || { log "FATAL serve never came up"; tail -5 $C/serve.log; kill $SPID; exit 1; }
  UP=http://127.0.0.1:$SP
fi
python3 $S/agent_proxy.py --listen $PP --upstream $UP --out $RAW --harness $HN > $C/proxy.log 2>&1 &
XPID=$!; sleep 1
trap 'kill $XPID $SPID 2>/dev/null' EXIT
log "serve :$SP, proxy :$PP, capture $RAW, rounds $ROUNDS"

# --- per-harness configuration (two variants: normal = 32k context, small = 12k context for the compact task) ---
H=$C/home; mkdir -p $H
BASEENV=(HOME=$H XDG_CONFIG_HOME=$H/.config XDG_DATA_HOME=$H/.local/share XDG_STATE_HOME=$H/.local/state
         XDG_CACHE_HOME=$H/.cache XDG_RUNTIME_DIR=$H/run)
mkdir -p $H/.config $H/.local/share $H/.local/state $H/.cache $H/run; chmod 700 $H/run
cfg() {  # cfg VARIANT CTX: write the harness config for a context size into $C/cfg-VARIANT
  local v=$1 ctx=$2 d=$C/cfg-$1; mkdir -p $d
  case $HN in
    pi)
      cat > $d/models.json <<J
{"providers": {"axon": {"baseUrl": "$PX/v1", "api": "openai-completions", "apiKey": "dummy",
  "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": false},
  "models": [{"id": "axon-7b", "name": "Axon Qwen2.5-7B", "contextWindow": $ctx, "maxTokens": 4096}]}}}
J
      [ $v = small ] && echo '{"compaction": {"enabled": true, "reserveTokens": 4096, "keepRecentTokens": 4000}}' > $d/settings.json ;;
    opencode)
      mkdir -p $d/opencode
      cat > $d/opencode/opencode.json <<J
{"\$schema": "https://opencode.ai/config.json", "model": "axon/axon-7b", "small_model": "axon/axon-7b",
 "share": "disabled", "autoupdate": false,
 "provider": {"axon": {"npm": "@ai-sdk/openai-compatible", "name": "Axon",
   "options": {"baseURL": "$PX/v1", "apiKey": "dummy"},
   "models": {"axon-7b": {"name": "Axon Qwen2.5-7B", "limit": {"context": $ctx, "output": 4096}}}}}}
J
      ;;
    codex)
      cat > $d/config.toml <<T
model = "axon-7b"
model_provider = "axon"
approval_policy = "never"
sandbox_mode = "workspace-write"
model_context_window = $ctx
$( [ $v = small ] && echo "model_auto_compact_token_limit = 8192" )

[model_providers.axon]
name = "Axon"
base_url = "$PX/v1"
wire_api = "responses"
env_key = "AXON_DUMMY_KEY"
T
      ;;
    crush)
      mkdir -p $d/crush
      cat > $d/crush/crushrc <<R
provider add axon --type openai-compat --base-url "$PX/v1" --api-key dummy
model add axon/axon-7b --name "Axon Qwen2.5-7B" --context-window $ctx --default-max-tokens 4096
option provider-auto-update false
permissions allow view ls grep glob edit multiedit write bash agent
permissions deny fetch download sourcegraph
R
      ;;
  esac
}
cfg normal 32768; cfg small 12288

run_h() {  # run_h VARIANT WORKDIR CONTINUE(0|1) PROMPT -> harness rc
  local v=$1 w=$2 cont=$3 p=$4 d=$C/cfg-$1
  case $HN in
    pi)
      (cd $w && env "${BASEENV[@]}" PI_CODING_AGENT_DIR=$d PI_OFFLINE=1 PI_TELEMETRY=0 PI_SKIP_VERSION_CHECK=1 \
        timeout $TO pi --offline --provider axon --model axon-7b --api-key dummy --session-dir $w/.pi-sessions \
        $( [ $cont = 1 ] && echo -c ) -p "$p") ;;
    opencode)
      (cd $w && env "${BASEENV[@]}" XDG_CONFIG_HOME=$d OPENCODE_DISABLE_AUTOUPDATE=1 \
        timeout $TO opencode run --standalone --auto -m axon/axon-7b $( [ $cont = 1 ] && echo -c ) "$p") ;;
    codex)
      mkdir -p $C/codex-home-$v; cp $d/config.toml $C/codex-home-$v/config.toml
      if [ $cont = 1 ]; then
        (cd $w && env "${BASEENV[@]}" CODEX_HOME=$C/codex-home-$v AXON_DUMMY_KEY=dummy \
          timeout $TO codex exec resume --last --skip-git-repo-check "$p")
      else
        (cd $w && env "${BASEENV[@]}" CODEX_HOME=$C/codex-home-$v AXON_DUMMY_KEY=dummy \
          timeout $TO codex exec --skip-git-repo-check -s workspace-write "$p")
      fi ;;
    crush)
      (cd $w && env "${BASEENV[@]}" XDG_CONFIG_HOME=$d CRUSH_DISABLE_METRICS=1 CRUSH_DISABLE_PROVIDER_AUTO_UPDATE=1 \
        timeout $TO crush run -q -m axon/axon-7b --small-model axon/axon-7b -c $w -D $w/.crush-data \
        $( [ $cont = 1 ] && echo -C ) "$p") ;;
  esac
}

P_BUGFIX="The test suite fails. Run it with: python3 -m unittest discover -s tests -t .   Find the bug in the ledger package, fix it with a minimal change, and rerun the tests until they all pass."
P_FEATURE="Add a CSV output format to the reports: a --format option on the 'report' command (choices: text, csv) wired through ledger/cli.py, ledger/reports.py (report) and ledger/formats.py (render), with the CSV writer implemented in formats.py using the csv module. Add a test for the CSV output in tests/test_reports.py and run the full test suite (python3 -m unittest discover -s tests -t .)."
P_RESUME="Now also add a json format the same way, document the --format option in README.md, and run the full test suite again."
P_EXPLAIN="Do not modify any file. Explain how an order flows from 'ledger add' on the command line into the JSON-lines file and back out into a customers report: name every function involved, in order, with its file, and list every validation that can reject input and where it lives."
P_COMPACT="Go through every Python file in this repository one at a time: read each file completely, then append a detailed paragraph about it to NOTES.md describing every function and how it is tested. After all files are done, propose three refactorings with code sketches at the end of NOTES.md."
P_SUBAGENT="Use subagents (your task / agent tool) to review ledger/pricing.py, ledger/storage.py and ledger/reports.py in parallel for bugs and missing tests, one subagent per file. Then write their combined findings to REVIEW.md."

tests() { (cd $1 && timeout 60 python3 -m unittest discover -s tests -t . > /dev/null 2>&1) && echo OK || echo FAIL; }
task() {  # task NAME ROUND VARIANT WORKDIR CONTINUE PROMPT CHECKTESTS
  local n=$1 r=$2 t0=$SECONDS rc; mark $n $r
  run_h $3 $4 $5 "$6" < /dev/null > $C/$n-r$r.out 2>&1; rc=$?
  echo "TASK $HN $n r$r rc $rc $((SECONDS - t0))s tests $( [ $7 = 1 ] && tests $4 || echo - )"
}
newrepo() { local w=$C/work/$1; mkdir -p $w; python3 $S/examples/harness-repo/make_repo.py $w; echo $w; }

for r in $(seq $ROUNDS); do
  task bugfix $r normal "$(newrepo bugfix-r$r)" 0 "$P_BUGFIX" 1
  if [ $r = 1 ] && [ "$(grep -c '"route"' $RAW 2>/dev/null)" = 0 ]; then
    log "FATAL: the harness made no request through the proxy (config?): see $C/bugfix-r1.out"; tail -15 $C/bugfix-r1.out
    echo "CAPTURE $HN DONE rc 3 (harness never reached the proxy)"; exit 3
  fi
  if [ -n "$SMOKE" ]; then
    echo "SMOKE $HN: $(grep -c '"route"' $RAW) request(s) reached the proxy; routes: $(grep -o '"route":"[a-z]*"' $RAW | sort | uniq -c | tr '\n' ' ')"
    exit 0
  fi
  wf=$(newrepo feature-r$r)
  task feature $r normal $wf 0 "$P_FEATURE" 1
  task resume $r normal $wf 1 "$P_RESUME" 1
  task explain $r normal "$(newrepo explain-r$r)" 0 "$P_EXPLAIN" 0
  task compact $r small "$(newrepo compact-r$r)" 0 "$P_COMPACT" 0
  case $HN in
    opencode|crush) task subagent $r normal "$(newrepo subagent-r$r)" 0 "$P_SUBAGENT" 0 ;;
    *) echo "TASK $HN subagent r$r skipped (no subagents in $HN)" ;;
  esac
done
kill $XPID $SPID 2>/dev/null; wait 2>/dev/null; trap - EXIT
log "capture: $(grep -c '"route"' $RAW) requests"
python3 $S/agent_trace_proxy.py $RAW --keep-raw --out $HOME/.cache/axon-agent/traces/harness-$HN-$TS.json; rc=$?
[ $rc = 0 ] && python3 - $HOME/.cache/axon-agent/traces/harness-$HN-$TS.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d["summary"]
print("SUMMARY", json.dumps({k: v for k, v in s.items() if k != "tasks"}))
for t, v in s["tasks"].items():
    print("SUMMARY-TASK", t, json.dumps(v))
PY
echo "CAPTURE $HN DONE rc $rc"
exit $rc
