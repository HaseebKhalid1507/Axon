#!/bin/bash
# compat_audit.sh ZOO_PY LLAMA_LOGITS_PY LIST AXON LLAMA_TGZ [zoo.py run args...]
# The compatibility audit as one axon-podjob job (docs/log/2026-11-13-compat-audit.md): scripts/zoo.py over LIST
# (scripts/examples/compat-audit/ollama-top30.json), downloading each model from Hugging Face just before it runs
# (sha256 checked) and deleting it after; `expect: refuse` entries are tried on their header first (a sparse file),
# so a clean refusal costs no download. The reference is llama.cpp's prebuilt CPU llama-server from LLAMA_TGZ
# (llama-b11580-bin-ubuntu-x64.tar.gz, sha256 de67caf1...: build 11580 = commit 4413a19, the 10-18 zoo's version).
#   axon-podjob --on POD --models none --max-min 240 --name eval-<sha7>-compat-audit --expect '^ZOO [0-9]+ files' -- \
#     scripts/compat_audit.sh scripts/zoo.py scripts/oracle/llama_logits.py scripts/examples/compat-audit/ollama-top30.json \
#     AXON_BIN llama-b11580-bin-ubuntu-x64.tar.gz --device cuda --chat --fetch-each --delete-after --sparse-first
# Results: ZOO-RESULT lines on stdout (`zoo.py audit-table LOG` makes the table); work files under
# ${AXON_AUDIT_DIR:-$HOME/.cache/axon-eval/audit-run}/<time>/ (models are deleted as they finish).
set -u
[ $# -ge 5 ] || { echo "usage: $0 ZOO_PY LLAMA_LOGITS_PY LIST AXON LLAMA_TGZ [zoo run args]"; exit 2; }
ZOO=$1; LLP=$2; LIST=$3; AX=$4; TGZ=$5; shift 5
D=$(cd "$(dirname "$ZOO")" && pwd)
mkdir -p "$D/oracle" && cp "$LLP" "$D/oracle/llama_logits.py"
W=${AXON_AUDIT_DIR:-$HOME/.cache/axon-eval/audit-run}/$(date +%m%d-%H%M%S)
mkdir -p "$W/models" "$W/llama"
echo "$(sha256sum "$TGZ" | cut -c1-64)" | grep -q '^de67caf142984e7ba2fb2b5396d6ea40f2946a512650d1f7e9d38ea1a6e5fac7$' ||
  { echo "compat_audit: $TGZ is not llama-b11580-bin-ubuntu-x64.tar.gz (sha256)"; exit 2; }
# No owner/mode restore: as root (pods) tar would chown to the archive's uid 1001, which fails on filesystems
# without chown (jade's job dir, 1113-004918-e28e).
tar --no-same-owner --no-same-permissions -xzf "$TGZ" -C "$W/llama" || exit 2
LS=$(find "$W/llama" -name llama-server -type f | head -1)
echo "compat_audit: host $(hostname) work $W cpus $(nproc) ram $(free -g | awk '/Mem:/{print $2}')G disk $(df -h "$W" | awk 'NR==2{print $4}') free"
echo "compat_audit: gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null); axon $(sha256sum "$AX" | cut -c1-12); $("$LS" --version 2>&1 | head -1)"
python3 "$D/zoo.py" run --axon "$(cd "$(dirname "$AX")" && pwd)/$(basename "$AX")" --llama-server "$LS" --list "$LIST" \
  --dir "$W/models" --out "$W/zoo.json" --threads "$(nproc)" "$@"
rc=$?
rm -rf "$W/models"
exit $rc
