#!/bin/bash
# agent_board.sh f16|q8 : PR A's baseline board on bella (agent_replay.py board), one KV type per job.
#   Qwen2.5-7B-Instruct Q4_K_M, agent-32k (scale 0.15, clip 32768, the 10-min trace window with the most sessions),
#   speeds 1 and 4, 3 rounds interleaved axon -> llama-server -> ollama; the baselines get Axon's KV token budget.
#
# DURATION (MEASURED, f16 board 1027-014629-7829): ~72 min per round, ~220-240 min for 3 rounds.
#   axon s1 640 s + s4 630 s; llama-server s1 655 s + s4 1,460-1,515 s (speed 4 overloads it); ollama s1 642 s +
#   s4 265 s; plus engine starts. Submit with --max-min >= 270 (200 killed the f16 board at 15 of 18 runs):
#     axon-run --detach --timing bella --vram 11000 --ram 14000 --max-min 270 --name agent-a-<sha7>-board-f16 \
#       --expect "BOARD END ok" -- scripts/agent_board.sh f16
#
# Inputs (env): AXON_AGENT_DIR (default ~/.cache/axon-agent: bin/axon-<sha>, traces/), AXON_AGENT_AXON (the axon
# binary), AXON_AGENT_TRACE (an --anonymize-time trace; never a file in this repo), LLAMA_SERVER, AXON_MODELS.
set -u
KV=${1:?usage: agent_board.sh f16|q8}
D=${AXON_AGENT_DIR:-$HOME/.cache/axon-agent}
W=$(cd "$(dirname "$0")/.." && pwd)
M=${AXON_MODELS:-$W/models}
OUT=$D/board-$KV-$(date +%m%d-%H%M%S); mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader > "$OUT/gpu.txt"
python3 -u "$W/scripts/agent_replay.py" board --engines axon,llama,ollama --kv "$KV" --rounds 3 --speeds 1,4 \
  --model "$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf" --ollama-model axon-q7 \
  --axon "${AXON_AGENT_AXON:?set AXON_AGENT_AXON to the axon binary under test}" \
  --llama-server "${LLAMA_SERVER:-$HOME/Projects/llama.cpp/build-cuda/bin/llama-server}" \
  --trace "${AXON_AGENT_TRACE:-$D/traces/board-trace-v1.json}" --window-s 600 --scale 0.15 --clip 32768 --out "$OUT"
echo "BOARD DIR $OUT"
