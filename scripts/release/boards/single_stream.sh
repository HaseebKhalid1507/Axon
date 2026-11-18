#!/bin/bash
# single_stream.sh [ATTEMPT]: release 0.1's single-stream board, one job, on a 4090 pod (--timing 4090, through
# pod.sh; amendment 3, 11-17) or on bella (--timing bella); queue.py runs it again while axon-run marks the job tainted
# (at most 4 jobs, sb/tN for ATTEMPT N; the first untainted one is read). axon (the release binary) vs llama-bench,
# upstream b12387's official build and 4413a19 (a pod: axon-weekly's sm_89 build; bella: bella's), with the weekly pod
# board's protocol: sb_pod.py here is
# axon-devstack's pod/board/sb_pod.py (954985e6) with the run order rotated by round, the GPU state per run, sb.json
# and a temperature-parsing fix (its docstring). Per model the GPU cooled, 3 rounds, salted prompts, pp560 / pp2100 /
# tg128, llama-bench -fa 1 -b 4096 -ub 512,4096 -r 3 (its best ubatch, and ub 512). sb_read.py reads it: each cell
# against the faster of the two llama.cpp builds. The pod paths come from a node dir here: models linked, NVRTC
# refused (AXON_KERNEL_PACK=only, common.sh: the release binary's own kernels), an empty kernel cache, a no-op
# axon-node hold, and each llama-bench through a wrapper written here (its library path; hashed).
set -u
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"
need_smoke sb
need_axon
K=${1:-1}; [[ "$K" =~ ^[1-9]$ ]] || { echo "STOP: ATTEMPT '$K' isn't 1-9"; exit 1; }
MODELS=${AXON_REL_SB_MODELS:-0.5B,1.5B,3B,7B,q3-8B}; ROUNDS=${AXON_REL_SB_ROUNDS:-3}
N=$D/sbnode; rm -rf "$N"; mkdir -p "$N/models" "$N/lib/nvrtc-none" "$N/cache" "$N/bin"
for f in qwen2.5-0.5b-q4_k_m.gguf qwen2.5-1.5b-instruct-q4_k_m.gguf qwen2.5-3b-instruct-q4_k_m.gguf \
         Qwen2.5-7B-Instruct-Q4_K_M.gguf Qwen3-8B-Q4_K_M.gguf; do ln -s "$MD/$f" "$N/models/$f"; done
echo pass > "$N/bin/axon-node"
LB=$N/bin/llama-bench-b12387   # the official build with its own cudart / cuBLAS 13.5 (RUNPATH $ORIGIN too)
printf '#!/bin/sh\nLD_LIBRARY_PATH=%s exec %s/llama-bench "$@"\n' "$NEW" "$NEW" > "$LB"; chmod +x "$LB"
LBO=$OLD/llama-bench           # bella: its own build, as is; a pod: the sm_89 build on the CUDA 12.8.1 runtime it links
if [ -n "$OLD_LIB" ]; then
  LBO=$N/bin/llama-bench-4413a19
  printf '#!/bin/sh\nLD_LIBRARY_PATH=%s exec %s/llama-bench "$@"\n' "$OLD_LIB" "$OLD" > "$LBO"; chmod +x "$LBO"
fi
OUT=$D/sb/t$K; [ -e "$OUT" ] && { echo "STOP: $OUT exists (never overwritten)"; exit 1; }
mkdir -p "$OUT"; cd "$OUT" || exit 1
cp "$S/sb_pod.py" "$S/sb_read.py" .
hw hw.txt; { sha256sum sb_pod.py sb_read.py "$LB" "$LBO"; cat "$LB"; [ "$LBO" = "$OLD/llama-bench" ] || cat "$LBO"
  up=~/.local/share/axon-devstack/current/pod/board/sb_pod.py; [ -e "$up" ] && sha256sum "$up"   # bella: upstream's
  (cd "$N/models" && sha256sum -- *.gguf); } >> hw.txt; cat hw.txt
SB_NODE=$N python3 -u sb_pod.py "$ROUNDS" "$MODELS" "$AX" --llama=llama:"$LB" --llama=llama_4413a19:"$LBO" \
  2>&1 | tee sb.log
grep -q SCOREBOARD_DONE sb.log || { echo "STOP: no SCOREBOARD_DONE"; exit 1; }
python3 sb_read.py sb.json > READING.txt 2>&1; cat READING.txt
smoke_passed sb
echo "SB END ok"
