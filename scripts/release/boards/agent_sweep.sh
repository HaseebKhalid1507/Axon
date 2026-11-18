#!/bin/bash
# agent_sweep.sh CELL [ATTEMPT]: release 0.1's agent board, part 1 of 2: ONE cell of the sweep per job, on a 4090 pod
# (--timing 4090, through pod.sh; amendment 3, 11-17: bella's desktop tainted its runs) or on bella (--timing bella). queue.py runs the cells, each its own job (the 11-15 sweep ran all its cells in one job and hit its bound:
# docs/log/2026-11-14-release-0.1-prep.md). Pre-registered in that log before it ran. Cell agent-32k
# (scripts/agent_board.sh's), f16 KV, speed 1, one round per run:
#   axon (the release binary): its paged pool is the KV budget every llama-server arm gets (--kv-tokens);
#   llama-server arms on that budget: seven configurations chosen to favor it, from llama.cpp's source (A6 from the
#   reviewer arguing its side, shady on #270):
#     A0  4413a19 (on a pod axon-weekly's sm_89 build, on bella bella's), the board's defaults: -np budget//32768 -c np*32768 -kvu --cache-reuse 256
#         (the 10-27 board and the weekly's config; the reference)
#     A1  b12387 (upstream's official build of today's master), the same defaults
#     A2  A1 + --no-cache-idle-slots   (by default llama-server saves AND clears every idle slot on each new task
#         under unified KV, so a resumed turn reloads its whole context from RAM)
#     A3  A2 + -ub 2048                (a turn's ~1.1k new tokens in one micro-batch)
#     A4  A1 with twice the slots on the same pool (-np 2*np -c budget, idle slots cleared: more concurrency)
#     A5  A2 + -cram 16384             (twice the RAM prompt cache: 20 sessions don't fit in 8 GiB)
#     A6  A1 + -cram 16384             (idle slots still saved and cleared, and a RAM cache that holds every session:
#                                       the save-and-restore path llama.cpp built for unified KV; A5 drops it)
# CELL: axon (first: it writes sweep/POOL) | A0 .. A6. ATTEMPT (default 1): an arm's run k goes to NAME (k = 1) or
# NAME-rk; queue.py runs it again only when the run's one fault is taint (pick.taint_only), at most 3 runs. pick.py
# then names the winner (the lowest TTFT-turn p50 of the qualifying arms; an arm's first untainted run counts).
# llama-server b12387 gets its library path through --llama-env, axon never runs under it. Never overwrites a run.
set -u
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"
need_smoke agent
need_axon
C=${1:?usage: agent_sweep.sh axon|A0..A6 [ATTEMPT]} K=${2:-1}
[[ "$K" =~ ^[1-9]$ ]] || { echo "STOP: ATTEMPT '$K' isn't 1-9"; exit 1; }
OUT=$D/sweep; mkdir -p "$OUT"
CELL=(--kv f16 --rounds 1 --speeds 1 --model "$M" --trace "$TR" --window-s "$WIN" --scale 0.15 --clip 32768)
if [ "$C" = axon ]; then
  [ "$K" = 1 ] || { echo "STOP: axon's reference cell runs once (its taint doesn't matter: it sets the budget)"; exit 1; }
  [ -e "$OUT/axon" ] && { echo "STOP: $OUT/axon exists (queue.py starts a sweep in an empty dir)"; exit 1; }
  mkdir -p "$OUT/axon"; hw "$OUT/axon/hw.txt"; cat "$OUT/axon/hw.txt"
  python3 -u "$W/scripts/agent_replay.py" board --engines axon --axon "$AX" "${CELL[@]}" --port 18700 --out "$OUT/axon" \
    || { echo "STOP: axon's reference run failed"; exit 1; }
  POOL=$(python3 -c "import json,glob,sys; print(json.load(open(glob.glob(sys.argv[1]+'/run-axon-*.json')[0]))['kv_tokens'])" \
         "$OUT/axon") || { echo "STOP: no axon run file"; exit 1; }
  echo "KV budget $POOL tokens (axon's paged pool): llama-server -np $(( POOL / 32768 > 0 ? POOL / 32768 : 1 )) by default"
  echo "$POOL" > "$OUT/POOL"
  echo "CELL END ok"; exit 0
fi
POOL=$(cat "$OUT/POOL" 2>/dev/null) && [ "$POOL" -gt 0 ] || { echo "STOP: no KV budget ($OUT/POOL: axon's cell first)"; exit 1; }
NP=$(( POOL / 32768 )); [ "$NP" -ge 1 ] || NP=1
case $C in   # name, build, port (+100 per attempt), agent_replay args
  A0) name=A0-4413a19-defaults which=old port=18710 args=() ;;
  A1) name=A1-b12387-defaults which=new port=18720 args=() ;;
  A2) name=A2-noidle which=new port=18730 args=(--llama-args=--no-cache-idle-slots) ;;
  A3) name=A3-noidle-ub2048 which=new port=18740 args=("--llama-args=--no-cache-idle-slots -ub 2048") ;;
  A4) name=A4-2x-slots which=new port=18750 args=(--llama-parallel $(( 2 * NP ))) ;;
  A5) name=A5-noidle-cram16g which=new port=18760 args=("--llama-args=--no-cache-idle-slots -cram 16384") ;;
  A6) name=A6-cram16g which=new port=18770 args=("--llama-args=-cram 16384") ;;
  *) echo "STOP: no cell '$C' (axon, A0 .. A6)"; exit 1 ;;
esac
dir=$name; [ "$K" -gt 1 ] && dir=$name-r$K
[ -e "$OUT/$dir" ] && { echo "STOP: $OUT/$dir exists (a run is never overwritten)"; exit 1; }
bin=$NEW/llama-server; [ "$which" = old ] && bin=$OLD/llama-server
read -ra lenv <<< "$(llama_env "$which")"
mkdir -p "$OUT/$dir"; hw "$OUT/$dir/hw.txt"; cat "$OUT/$dir/hw.txt"
echo "== arm $dir (run $K): $bin ${args[*]}; KV budget $POOL tokens"
python3 -u "$W/scripts/agent_replay.py" board --engines llama --llama-server "$bin" "${lenv[@]}" --kv-tokens "$POOL" \
  "${CELL[@]}" --port $(( port + 100 * (K - 1) )) --out "$OUT/$dir" "${args[@]}" \
  || echo "arm $dir: agent_replay rc $? (no run file: the arm doesn't qualify; pick.py says why)"
echo "CELL END ok"
