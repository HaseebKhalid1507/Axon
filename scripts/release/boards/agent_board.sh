#!/bin/bash
# agent_board.sh: release 0.1's agent board, part 2 of 2 (bella, --timing bella), after agent_sweep.sh.
# axon (the release binary) vs llama-server in the sweep's winning config (sweep/WINNER.env), cell agent-32k,
# f16 KV, speeds 1 and 4, 3 interleaved rounds (scripts/agent_board.sh's protocol; no ollama), the engine order
# reversed on odd rounds (--rotate-engines), llama-server on the sweep's KV budget (sweep/POOL: the budget its
# winner won on). read_board.py reads the stop rule (speed 1's TTFT-turn p50 over rounds where both engines are
# clean; R and the verdicts are defined there and in docs/log/2026-11-14-release-0.1-prep.md).
set -u
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"
need_smoke agent
need_axon
[ -f "$D/sweep/WINNER.env" ] || { echo "STOP: no sweep winner ($D/sweep/WINNER.env)"; exit 1; }
POOL=$(cat "$D/sweep/POOL") && [ "$POOL" -gt 0 ] || { echo "STOP: no sweep budget ($D/sweep/POOL)"; exit 1; }
# shellcheck source=/dev/null  # written by pick.py
. "$D/sweep/WINNER.env"
OUT=$D/board; rm -rf "$OUT"; mkdir -p "$OUT"; hw "$OUT/hw.txt"; cat "$OUT/hw.txt"
echo "llama-server: arm $ARM ($WHICH build, -np ${PAR:-default}, args '${LARGS}'); KV budget $POOL tokens (the sweep's)"
bin=$NEW/llama-server lenv=(--llama-env "LD_LIBRARY_PATH=$NEW")
[ "$WHICH" = old ] && bin=$OLD/llama-server lenv=()
extra=(); [ "$PAR" != 0 ] && extra+=(--llama-parallel "$PAR"); [ -n "$LARGS" ] && extra+=("--llama-args=$LARGS")
python3 -u "$W/scripts/agent_replay.py" board --engines axon,llama --axon "$AX" --llama-server "$bin" "${lenv[@]}" \
  "${extra[@]}" --kv-tokens "$POOL" --rotate-engines --kv f16 --rounds "${AXON_REL_ROUNDS:-3}" \
  --speeds "${AXON_REL_SPEEDS:-1,4}" --model "$M" --trace "$TR" --window-s "$WIN" --scale 0.15 --clip 32768 \
  --port 18900 --out "$OUT" || { echo "STOP: board rc $?"; exit 1; }
python3 "$W/scripts/agent_replay.py" table --clean-only "$OUT" > "$OUT/table-clean.md" 2>&1
python3 "$S/read_board.py" "$OUT" --pool "$POOL" > "$OUT/READING.txt" 2>&1; cat "$OUT/READING.txt"
smoke_passed agent
echo "BOARD END ok"
