#!/bin/bash
# prompt_diff.sh PROMPT_DIFF_PY HARNESS_ACCEPT_PY MAKE_REPO AXON LLAMA_TGZ MODEL [BODY.json...]
# The prompt each engine builds from the same harness request, as one axon-podjob job (CPU, no GPU):
# captures codex's first request on this node (`harness_accept.py capture`, if codex is installed), then runs
# prompt_diff.py over the BODY files given plus codex's. MODEL is a file in ./models; a small model with the run
# model's chat template is enough (the prompt doesn't depend on the weights). Qwen's own qwen2.5-0.5b-q4_k_m.gguf
# (on jade) has the run model's template but for #248's doubled braces in the tool instruction, which then shows as
# one more differing line on llama.cpp's side. LLAMA_TGZ is llama.cpp's prebuilt CPU
# llama-server, llama-b11580-bin-ubuntu-x64.tar.gz (= 4413a19, the acceptance test's llama.cpp).
#   axon-podjob --on jade --cpu --ram 3000 --models qwen2.5-0.5b-q4_k_m.gguf --name eval-<sha7>-prompt-diff \
#     --expect '^PROMPT-DIFF DONE' -- scripts/examples/harness-repo/prompt_diff.sh \
#     scripts/examples/harness-repo/prompt_diff.py scripts/harness_accept.py scripts/examples/harness-repo/make_repo.py \
#     AXON_BIN llama-b11580-bin-ubuntu-x64.tar.gz qwen2.5-0.5b-q4_k_m.gguf pi.json opencode.json
set -u
[ $# -ge 6 ] || { echo "usage: $0 PROMPT_DIFF_PY HARNESS_ACCEPT_PY MAKE_REPO AXON LLAMA_TGZ MODEL [BODY...]"; exit 2; }
PD=$1; HA=$2; MR=$3; AX=$4; TGZ=$5; MF=$6; shift 6
# axon-run appends `--test-threads N` on CPU nodes: not ours, drop it.
A=()
while [ $# -gt 0 ]; do
  case $1 in
    --test-threads) shift; [ $# -gt 0 ] && shift ;;
    --test-threads=*) shift ;;
    *) A+=("$1"); shift ;;
  esac
done
set -- ${A[@]+"${A[@]}"}
MD=${AXON_MODELS:-models}
D=$(cd "$(dirname "$HA")" && pwd)
[ -e "$D/make_repo.py" ] || cp "$MR" "$D/make_repo.py"
echo "$(sha256sum "$TGZ" | cut -c1-64)" | grep -q '^de67caf142984e7ba2fb2b5396d6ea40f2946a512650d1f7e9d38ea1a6e5fac7$' ||
  { echo "prompt_diff: $TGZ is not llama-b11580-bin-ubuntu-x64.tar.gz (sha256)"; exit 2; }
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
tar --no-same-owner --no-same-permissions -xzf "$TGZ" -C "$W" || exit 2
LS=$(find "$W" -name llama-server -type f | head -1)
echo "prompt_diff: host $(hostname) cpus $(nproc); model $MF"
B=("$@")
if command -v codex >/dev/null; then
  python3 "$HA" capture --harness codex --out "$W/cap" --timeout 120
  for f in "$W"/cap/codex-*.json; do [ -e "$f" ] && B+=("$f"); done
fi
for f in ${B[@]+"${B[@]}"}; do
  python3 -c "import json,sys; b=json.load(open(sys.argv[1])); print('body', sys.argv[1].split('/')[-1], 'tools', len(b.get('tools') or []))" "$f"
done
python3 "$PD" --axon "$(cd "$(dirname "$AX")" && pwd)/$(basename "$AX")" --llama-server "$LS" --model "$MD/$MF" \
  ${B[@]+"${B[@]}"}
