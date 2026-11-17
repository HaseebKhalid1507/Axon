#!/bin/bash
# agent_board.sh ROUND: release 0.1's agent board, part 2 of 2 (bella, --timing bella), after agent_sweep.sh: ONE round
# per job (queue.py runs the rounds, each its own job; a round is ~1 h: docs/log/2026-11-14-release-0.1-prep.md).
# axon (the release binary) vs llama-server in the sweep's winning config (sweep/WINNER.env), cell agent-32k, f16 KV,
# speeds 1 and 4 (scripts/agent_board.sh's protocol; no ollama), the engine order reversed on odd rounds
# (--rotate-engines: by the round number), llama-server on the sweep's KV budget (sweep/POOL: the budget its winner
# won on). Rounds 0, 1, 2, ...: queue.py runs them until 3 have run that weren't lost to taint alone (at most 8); a
# round lost to taint alone is replaced by the next round number. read_board.py reads the stop rule over every round in
# board/ (speed 1's TTFT-turn p50 over rounds where both engines are clean; R and the verdicts are defined there and in
# the log). Never overwrites a round.
set -u
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"
need_smoke agent
need_axon
R=${1:?usage: agent_board.sh ROUND}
[[ "$R" =~ ^[0-9]$ ]] || { echo "STOP: ROUND '$R' isn't 0-9"; exit 1; }
[ -f "$D/sweep/WINNER.env" ] || { echo "STOP: no sweep winner ($D/sweep/WINNER.env)"; exit 1; }
POOL=$(cat "$D/sweep/POOL") && [ "$POOL" -gt 0 ] || { echo "STOP: no sweep budget ($D/sweep/POOL)"; exit 1; }
# shellcheck source=/dev/null  # written by pick.py
. "$D/sweep/WINNER.env"
OUT=$D/board; mkdir -p "$OUT"
compgen -G "$OUT/run-*-r$R.json" > /dev/null && { echo "STOP: round $R exists in $OUT (never overwritten)"; exit 1; }
hw "$OUT/hw-r$R.txt"; cat "$OUT/hw-r$R.txt"
echo "round $R: llama-server: arm $ARM ($WHICH build, -np ${PAR:-default}, args '${LARGS}'); KV budget $POOL tokens (the sweep's)"
bin=$NEW/llama-server lenv=(--llama-env "LD_LIBRARY_PATH=$NEW")
[ "$WHICH" = old ] && bin=$OLD/llama-server lenv=()
extra=(); [ "$PAR" != 0 ] && extra+=(--llama-parallel "$PAR"); [ -n "$LARGS" ] && extra+=("--llama-args=$LARGS")
python3 -u "$W/scripts/agent_replay.py" board --engines axon,llama --axon "$AX" --llama-server "$bin" "${lenv[@]}" \
  "${extra[@]}" --kv-tokens "$POOL" --rotate-engines --kv f16 --round-offset "$R" --rounds 1 \
  --speeds "${AXON_REL_SPEEDS:-1,4}" --model "$M" --trace "$TR" --window-s "$WIN" --scale 0.15 --clip 32768 \
  --port $(( 18900 + 10 * R )) --out "$OUT" || { echo "STOP: round $R: board rc $?"; exit 1; }
cp "$OUT/meta.json" "$OUT/meta-r$R.json"   # each round's own (agent_replay rewrites meta.json per run)
python3 "$S/read_board.py" "$OUT" --pool "$POOL" 2>&1 | tail -4   # the reading so far (queue.py writes READING.txt)
smoke_passed agent
echo "ROUND END ok"
