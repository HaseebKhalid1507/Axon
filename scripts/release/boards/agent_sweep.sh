#!/bin/bash
# agent_sweep.sh: release 0.1's agent board, part 1 of 2 (bella, --timing bella). Pre-registered in
# docs/log/2026-11-14-release-0.1-prep.md before it ran. Cell agent-32k (scripts/agent_board.sh's), f16 KV, speed 1, one round per arm:
#   axon (the release binary): its paged pool is the KV budget every llama-server arm gets (--kv-tokens);
#   llama-server arms, each llama.cpp's own tuning, on that budget:
#     A0  4413a19 (bella's build), the board's defaults: -np budget//32768 -c np*32768 -kvu --cache-reuse 256
#         (the 10-27 board and the weekly's config; the reference)
#     A1  b12387 (upstream's official build of today's master), the same defaults
#     A2  A1 + --no-cache-idle-slots   (by default llama-server saves AND clears every idle slot on each new task
#         under unified KV, so a resumed turn reloads its whole context from RAM)
#     A3  A2 + -ub 2048                (a turn's ~1.1k new tokens in one micro-batch)
#     A4  A1 with twice the slots on the same pool (-np 2*np -c budget, idle slots cleared: more concurrency)
#     A5  A2 + -cram 16384             (twice the RAM prompt cache: 20 sessions don't fit in 8 GiB)
# Then pick.py names the winner (its rule: pick.py and the log); agent_board.sh runs it against axon.
set -u
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"
OUT=$D/sweep
rm -rf "$OUT"; mkdir -p "$OUT"; hw "$OUT/hw.txt"; cat "$OUT/hw.txt"
CELL=(--kv f16 --rounds 1 --speeds 1 --model "$M" --trace "$TR" --window-s "$WIN" --scale 0.15 --clip 32768)
python3 -u "$W/scripts/agent_replay.py" board --engines axon --axon "$AX" "${CELL[@]}" --port 18700 --out "$OUT/axon" \
  || { echo "STOP: axon's reference run failed"; exit 1; }
POOL=$(python3 -c "import json,glob,sys; print(json.load(open(glob.glob(sys.argv[1]+'/run-axon-*.json')[0]))['kv_tokens'])" "$OUT/axon") \
  || { echo "STOP: no axon run file"; exit 1; }
NP=$(( POOL / 32768 )); [ "$NP" -ge 1 ] || NP=1
echo "KV budget $POOL tokens (axon's paged pool): llama-server -np $NP by default"
echo "$POOL" > "$OUT/POOL"
arm() {   # arm NAME old|new PORT [agent_replay args...]
  local name=$1 which=$2 port=$3; shift 3
  local bin=$NEW/llama-server env=(env LD_LIBRARY_PATH="$NEW")
  [ "$which" = old ] && bin=$OLD/llama-server env=(env -u LD_LIBRARY_PATH)
  echo "== arm $name: $bin $*"
  "${env[@]}" python3 -u "$W/scripts/agent_replay.py" board --engines llama --llama-server "$bin" --kv-tokens "$POOL" \
    "${CELL[@]}" --port "$port" --out "$OUT/$name" "$@" || echo "arm $name: rc $?"
}
arm A0-4413a19-defaults old 18710
arm A1-b12387-defaults new 18720
arm A2-noidle new 18730 --llama-args=--no-cache-idle-slots
arm A3-noidle-ub2048 new 18740 "--llama-args=--no-cache-idle-slots -ub 2048"
arm A4-2x-slots new 18750 --llama-parallel $(( 2 * NP ))
arm A5-noidle-cram16g new 18760 "--llama-args=--no-cache-idle-slots -cram 16384"
# A tainted arm (foreign GPU load) is run once more at the end; pick.py reads the last clean run of each arm.
for t in $(python3 "$S/pick.py" "$OUT" --tainted); do
  case $t in
    A0-*) arm "$t-r2" old 18810 ;; A1-*) arm "$t-r2" new 18820 ;;
    A2-*) arm "$t-r2" new 18830 --llama-args=--no-cache-idle-slots ;;
    A3-*) arm "$t-r2" new 18840 "--llama-args=--no-cache-idle-slots -ub 2048" ;;
    A4-*) arm "$t-r2" new 18850 --llama-parallel $(( 2 * NP )) ;;
    A5-*) arm "$t-r2" new 18860 "--llama-args=--no-cache-idle-slots -cram 16384" ;;
  esac
done
python3 "$S/pick.py" "$OUT" > "$OUT/WINNER.txt"; rc=$?
cat "$OUT/WINNER.txt"
[ $rc = 0 ] || { echo "STOP: no llama-server arm qualified (pick.py rc $rc)"; exit 1; }
echo "SWEEP END ok"
