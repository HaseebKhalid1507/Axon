#!/bin/bash
# single_stream.sh: release 0.1's single-stream board on bella (--timing bella): axon (the release binary) vs llama-bench,
# upstream b12387's official build and bella's 4413a19, with the weekly pod board's script and protocol unchanged
# (axon-devstack pod/board/sb_pod.py, copied and hashed): per model the GPU cooled, 3 interleaved rounds, salted
# prompts, pp560 / pp2100 / tg128, llama-bench -fa 1 -b 4096 -ub 512,4096 -r 3 (its best ubatch, and ub 512).
# The pod paths come from a node dir here: models linked, no NVRTC (the release binary carries its kernels), an
# empty kernel cache, a no-op axon-node hold.
set -u
. "$(dirname "$0")/common.sh"
MODELS=${REL_SB_MODELS:-0.5B,1.5B,3B,7B,q3-8B}; ROUNDS=${REL_SB_ROUNDS:-3}
N=$D/sbnode; rm -rf "$N"; mkdir -p "$N/models" "$N/lib/nvrtc-none" "$N/cache" "$N/bin"
for f in qwen2.5-0.5b-q4_k_m.gguf qwen2.5-1.5b-instruct-q4_k_m.gguf qwen2.5-3b-instruct-q4_k_m.gguf \
         Qwen2.5-7B-Instruct-Q4_K_M.gguf Qwen3-4B-Q4_K_M.gguf Qwen3-8B-Q4_K_M.gguf; do ln -s "$MD/$f" "$N/models/$f"; done
echo pass > "$N/bin/axon-node"
OUT=$D/sb${SMOKE:+-smoke}; rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT" || exit 1
cp ~/.local/share/axon-devstack/current/pod/board/sb_pod.py .
hw hw.txt; { sha256sum sb_pod.py; (cd "$N/models" && sha256sum -- *.gguf); } >> hw.txt; cat hw.txt
"$AX" kernels list 2>&1 | tail -4
SB_NODE=$N python3 -u sb_pod.py "$ROUNDS" "$MODELS" "$AX" --llama=llama:"$HOME/.cache/axon-dist/llama-b12387/llama-bench" \
  --llama=llama_4413a19:"$OLD/llama-bench" 2>&1 | tee sb.log
grep -q SCOREBOARD_DONE sb.log && echo "SB END ok"
