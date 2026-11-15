#!/bin/bash
# agent_board.sh: release 0.1's agent board, part 2 of 2 (bella, --timing bella), after agent_sweep.sh.
# axon (the release binary) vs llama-server in the sweep's winning config (sweep/WINNER.env), cell agent-32k,
# f16 KV, speeds 1 and 4, 3 interleaved rounds (scripts/agent_board.sh's protocol; no ollama). The stop rule reads
# speed 1's TTFT-turn p50, the median over clean rounds (docs/log/2026-11-14-release-0.1-prep.md).
set -u
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"
need_axon
[ -f "$D/sweep/WINNER.env" ] || { echo "STOP: no sweep winner ($D/sweep/WINNER.env)"; exit 1; }
# shellcheck source=/dev/null  # written by pick.py
. "$D/sweep/WINNER.env"
OUT=$D/board; rm -rf "$OUT"; mkdir -p "$OUT"; hw "$OUT/hw.txt"; cat "$OUT/hw.txt"
echo "llama-server: arm $ARM ($WHICH build, -np ${PAR:-default}, args '${LARGS}')"
bin=$NEW/llama-server env=(env LD_LIBRARY_PATH="$NEW")
[ "$WHICH" = old ] && bin=$OLD/llama-server env=(env -u LD_LIBRARY_PATH)
extra=(); [ "$PAR" != 0 ] && extra+=(--llama-parallel "$PAR"); [ -n "$LARGS" ] && extra+=("--llama-args=$LARGS")
"${env[@]}" python3 -u "$W/scripts/agent_replay.py" board --engines axon,llama --axon "$AX" --llama-server "$bin" \
  "${extra[@]}" --kv f16 --rounds "${AXON_REL_ROUNDS:-3}" --speeds "${AXON_REL_SPEEDS:-1,4}" --model "$M" --trace "$TR" \
  --window-s "$WIN" --scale 0.15 --clip 32768 --port 18900 --out "$OUT" || { echo "STOP: board rc $?"; exit 1; }
python3 "$W/scripts/agent_replay.py" table --clean-only "$OUT" > "$OUT/table-clean.md" 2>&1
echo "BOARD END ok"
