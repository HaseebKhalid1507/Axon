#!/bin/bash
# agent_board_kill_smoke.sh : no engine outlives a killed board (10-27: a llama-server in its own session survived
# axon-run's killpg and held 10 GB of VRAM for 37 min). CPU only (0.5B), runs anywhere axon-run --cpu puts it:
#   axon-run --cpu --ram 6000 --max-min 20 --expect "KILL SMOKE PASS" -- scripts/agent_board_kill_smoke.sh
# For each case it starts `agent_replay.py board --cpu` (llama-server, then ollama, then axon) in a session of its
# own, waits until an engine is up and serving, sends the signal to the board's python pid ONLY (not the group:
# the worst case), and then asserts that no process carrying the board's environment tag survives (the tag is
# inherited by every engine and every engine child, e.g. ollama's runner).
#   term-llama   SIGTERM while llama-server serves (the handlers)
#   term-ollama  SIGTERM while ollama + its runner serve
#   kill-llama   SIGKILL while llama-server serves (PR_SET_PDEATHSIG)
#   group-axon   SIGTERM to the whole process group while axon serves (what axon-run does)
set -u
W=$(cd "$(dirname "$0")/.." && pwd)
D=${AXON_AGENT_DIR:-$HOME/.cache/axon-agent}
M=${AXON_MODELS:-$W/models}
AX=${AXON_AGENT_AXON:-$(ls -t "$D"/bin/axon-* 2>/dev/null | head -1)}
LS=${LLAMA_SERVER_CPU:-$HOME/Projects/llama.cpp/build/bin/llama-server}
fail=0
tagged() {  # pids (our uid) whose environment carries AXON_KILL_SMOKE_TAG=$1
  for e in /proc/[0-9]*/environ; do
    p=${e#/proc/}; p=${p%/environ}
    [ "$p" = $$ ] && continue
    tr '\0' '\n' < "$e" 2>/dev/null | grep -qx "AXON_KILL_SMOKE_TAG=$1" && echo "$p"
  done
}
case_run() {  # name signal target engines wait_kind
  local name=$1 sig=$2 target=$3 engines=$4 kind=$5 tag="ks-$$-$1-$RANDOM" out
  out=$(mktemp -d "$D/killsmoke-$name-XXXX")
  AXON_KILL_SMOKE_TAG=$tag setsid python3 -u "$W/scripts/agent_replay.py" board --cpu --engines "$engines" --kv f16 \
    --rounds 1 --speeds 4 --model "$M/qwen2.5-0.5b-q4_k_m.gguf" --ollama-model axon-oracle --axon "$AX" \
    --llama-server "$LS" --trace "$W/scripts/examples/agent-trace-synthetic.json" --window-s 300 --scale 0.05 \
    --clip 4096 --kv-tokens 8192 --port $((19000 + RANDOM % 500 * 10)) --out "$out" > "$out/board.log" 2>&1 &
  local bp=$! t=0
  until grep -q "\"kind\": \"$kind\", \"port\": [0-9]*, \"state\": \"ready\"" "$out/engines.jsonl" 2>/dev/null; do
    sleep 1; t=$((t + 1))
    if [ $t -ge 240 ] || ! kill -0 $bp 2>/dev/null; then
      echo "KILL SMOKE $name: SETUP FAIL ($kind never ready; board log tail:)"; tail -5 "$out/board.log"
      kill -KILL -- -$bp 2>/dev/null; fail=1; return
    fi
  done
  local before; before=$(tagged "$tag" | wc -l)
  if [ "$target" = group ]; then kill -$sig -- -$bp; else kill -$sig $bp; fi
  sleep 20
  local left; left=$(tagged "$tag")
  if [ -n "$left" ]; then
    echo "KILL SMOKE $name: FAIL ($before tagged processes before, survivors: $(echo $left | tr '\n' ' '))"
    for p in $left; do tr '\0' ' ' < /proc/$p/cmdline; echo; kill -KILL $p 2>/dev/null; done
    fail=1
  else
    echo "KILL SMOKE $name: ok ($before tagged processes before SIG$sig to the $target, none after 20 s)"
  fi
}
case_run term-llama TERM pid llama,ollama,axon llama
case_run term-ollama TERM pid ollama,llama,axon ollama
case_run kill-llama KILL pid llama,ollama,axon llama
case_run group-axon TERM group axon,llama,ollama axon
[ $fail = 0 ] && echo "KILL SMOKE PASS" || echo "KILL SMOKE FAIL"
exit $fail
