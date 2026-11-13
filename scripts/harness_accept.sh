#!/bin/bash
# harness_accept.sh PY MAKE_REPO AXON LLAMA MODEL HARNESS [harness_accept.py run args...]
# One coding-agent acceptance job in axon-podjob's form (docs/log/2026-11-13-harness-accept.md): axon-podjob stages
# this script, harness_accept.py, make_repo.py and both engine binaries as ./name and runs it in the staged dir.
#   axon-podjob --on bella --vram 10500 --ram 12000 --name eval-<sha7>-accept-<model>-<harness> --expect '^SUMMARY' \
#     -- scripts/harness_accept.sh scripts/harness_accept.py scripts/examples/harness-repo/make_repo.py \
#        AXON_BIN LLAMA_SERVER_BIN qwen2.5-7b pi [--tasks all] [--engines axon,llama]
# Results: RESULT / ENGINE / SUMMARY lines on stdout; the harness outputs, engine logs and workspaces under
# ${AXON_ACCEPT_OUT:-~/.cache/axon-eval/accept}/<model>-<harness>-<time>/ on the host that ran it.
set -u
[ $# -ge 6 ] || { echo "usage: $0 PY MAKE_REPO AXON LLAMA MODEL HARNESS [args]"; exit 2; }
PY=$1; MR=$2; AX=$3; LL=$4; M=$5; H=$6; shift 6
D=$(cd "$(dirname "$PY")" && pwd)
[ -e "$D/make_repo.py" ] || cp "$MR" "$D/make_repo.py"
OUT=${AXON_ACCEPT_OUT:-$HOME/.cache/axon-eval/accept}/$M-$H-$(date +%m%d-%H%M%S)
mkdir -p "$OUT"
echo "harness_accept.sh: host $(hostname) out $OUT gpu $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null)"
exec python3 "$PY" run --model "$M" --harness "$H" --axon-bin "$AX" --llama-bin "$LL" --out "$OUT" "$@"
