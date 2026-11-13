#!/bin/bash
# compat_triage.sh GOLDEN_PY ZOO_PY LLAMA_LOGITS_PY LIST_JSON AXON LLAMA_CPU LLAMA_CUDA [MODELS_DIR]
#
# The 11-13 compatibility audit's two greedy drifts, bracketed by llama.cpp's own spread (docs/compat-audit-2026-11-13.md):
# DeepSeek-R1-0528-Qwen3-8B (YaRN factor 4: KL 0.021, 2 flips at 0.20 nats) and Qwen2.5-7B (1 flip at 0.25 nats), with
# Qwen3-8B as the control (R1's architecture and size, no YaRN). Runs on bella through axon-podjob:
#   1. zoo.py, Axon CUDA vs llama.cpp CPU as in the audit, the three models (same-machine baseline);
#   2. zoo.py on R1 with YaRN off on both sides (--rope-scaling none): does the gap go with YaRN?
#   3. zoo.py on R1, Axon CPU vs llama.cpp CPU: is it Axon's CUDA path?
#   4. golden.py (PR #229) on the three models: text in on both sides, llama.cpp's greedy 64 tokens teacher-forced
#      into Axon, limits from llama.cpp's own CUDA-vs-CPU spread (--llama-server-ref).
# AXON must have AXON_LOGITS_CHAT (golden.py's chat check: #229's build). R1 is downloaded (sha256 from LIST_JSON)
# into the job's work dir and deleted at the end; the other two are read from MODELS_DIR after a sha256 check.
set -euo pipefail
GOLDEN=$(readlink -f "$1"); ZOO=$(readlink -f "$2"); LLPY=$(readlink -f "$3"); LIST=$(readlink -f "$4")
AX=$(readlink -f "$5"); LCPU=$(readlink -f "$6"); LCUDA=$(readlink -f "$7"); MD=${8:-$HOME/Projects/axon/models}
W=$HOME/.cache/axon-eval/triage-$(date +%m%d-%H%M%S); mkdir -p "$W/models" "$W/oracle"
trap 'rm -rf "$W/models"' EXIT
cp "$ZOO" "$W/zoo.py"; cp "$LLPY" "$W/oracle/llama_logits.py"; cp "$GOLDEN" "$W/golden.py"
R1=DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf; Q25=Qwen2.5-7B-Instruct-Q4_K_M.gguf; Q3=Qwen3-8B-Q4_K_M.gguf
echo "triage: host $(hostname) work $W; axon $(sha256sum "$AX" | cut -c1-12); llama $("$LCPU" --version 2>&1 | grep -m1 version)"
for f in $Q25 $Q3; do
  want=$(python3 -c "import json,sys; print(next(e['sha256'] for e in json.load(open('$LIST')) if e['file'] == '$f'))")
  got=$( [ -f "$MD/$f.sha256" ] && cut -d' ' -f1 "$MD/$f.sha256" || sha256sum "$MD/$f" | cut -d' ' -f1)
  echo "triage: $f sha256 $got (audit's $want): $([ "$got" = "$want" ] && echo same file || echo DIFFERENT FILE)"
  ln -s "$MD/$f" "$W/models/$f"
done
cd "$W"
set +e   # from here a step that fails is reported and the next one runs (zoo.py exits 1 on a BUG verdict)
z() { python3 zoo.py run --list "$LIST" --axon "$AX" --llama-server "$LCPU" --dir "$W/models" --prompts 16 --threads 8 \
        --fetch-each "$@"; }
echo "== 1. zoo: axon cuda vs llama.cpp cpu (as the audit)"
z --device cuda --only DeepSeek-R1-0528 --out z1-r1.json | grep -E "^\{|^ZOO [0-9]|Traceback|Error"
z --device cuda --only Qwen2.5-7B-Instruct --out z1-q25.json | grep -E "^\{|^ZOO [0-9]|Traceback|Error"
z --device cuda --only Qwen3-8B --out z1-q3.json | grep -E "^\{|^ZOO [0-9]|Traceback|Error"
echo "== 2. zoo: R1 with YaRN off on both sides"
z --device cuda --only DeepSeek-R1-0528 --axon-args "--rope-scaling none" --llama-args "--rope-scaling none" \
  --out z2-r1-noyarn.json | grep -E "^\{|^ZOO [0-9]|Traceback|Error"
echo "== 3. zoo: R1, axon cpu vs llama.cpp cpu"
z --device cpu --only DeepSeek-R1-0528 --out z3-r1-cpu.json | grep -E "^\{|^ZOO [0-9]|Traceback|Error"
echo "== 4. golden.py (#229): llama.cpp's spread as the yardstick"
python3 golden.py run --axon "$AX" --llama-server "$LCPU" --llama-server-ref "$LCUDA" --llama-ref-ngl 99 \
  --models-dir "$W/models" --models "$R1,$Q25,$Q3" --device cuda --n 64 --threads 8 --out golden.json
echo "golden rc $?"
python3 golden.py table golden.json
for f in z1-r1 z1-q25 z1-q3 z2-r1-noyarn z3-r1-cpu; do [ -f $f.json ] && echo "TRIAGE-JSON $f $(python3 -c "import json; print(json.dumps(json.load(open('$f.json'))))")"; done
[ -f golden.json ] && echo "TRIAGE-JSON golden $(python3 -c "import json; print(json.dumps(json.load(open('golden.json'))))")"
echo "TRIAGE done"
