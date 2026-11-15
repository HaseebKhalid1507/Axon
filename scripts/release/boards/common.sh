# shellcheck shell=bash disable=SC2034  # sourced: the variables are the callers'
# common.sh (sourced): release 0.1's boards on bella (docs/log/2026-11-14-release-0.1-prep.md). The paths are
# bella's: llama.cpp builds, models, the agent trace (never in this repo). Outputs go to $AXON_REL_OUT. The axon under
# test is the release dry run's own binary (the workflow's tarball), AXON_REL_AXON. Settings are AXON_REL_* because
# axon-run forwards only AXON_* variables to a job.
# AXON_REL_SMOKE=1: the mechanics only, next to other GPU jobs: the 0.5B, a 60-s window, a 512-page pool
# (AXON_KV_PAGES), outputs under rel01-smoke. A real run starts only after a smoke of the same scripts passed
# (need_smoke) and takes the pre-registered protocol: no AXON_REL_* setting may change it.
S=$(cd "$(dirname "$0")" && pwd)
W=$(cd "$S/../../.." && pwd)
smoke=; [ "${AXON_REL_SMOKE:-0}" = 1 ] && smoke=-smoke
# No NVRTC: the release binary's own kernel pack or a failure (a broken pack must fail the board, not hide behind
# bella's NVRTC 13.4 in /opt/cuda). No inherited library path: llama-server b12387 gets its own (--llama-env, the
# llama-bench wrapper), nothing else runs under it.
export AXON_KERNEL_PACK=only
unset LD_LIBRARY_PATH
if [ -z "$smoke" ]; then
  for v in AXON_REL_ROUNDS AXON_REL_SPEEDS AXON_REL_SB_MODELS AXON_REL_SB_ROUNDS AXON_REL_MODEL AXON_REL_WINDOW \
           AXON_REL_LLAMA_NEW AXON_REL_LLAMA_OLD; do
    [ -z "${!v:-}" ] || { echo "STOP: $v is set: a real run takes the pre-registered protocol"; exit 1; }
  done
fi
D=${AXON_REL_OUT:-$HOME/.cache/axon-dist/rel01$smoke}
AX=${AXON_REL_AXON:-$D/rc/axon}
NEW=${AXON_REL_LLAMA_NEW:-$HOME/.cache/axon-dist/llama-b12387/x/llama-b12387}    # upstream b12387 (a956a852c), official cuda-13.5 build
OLD=${AXON_REL_LLAMA_OLD:-$HOME/Projects/llama.cpp/build-cuda/bin}                # bella's 4413a19 build (the 10-27 board, the weekly)
MD=$HOME/Projects/axon/models
M=${AXON_REL_MODEL:-$MD/Qwen2.5-7B-Instruct-Q4_K_M.gguf}
TR=$HOME/.cache/axon-agent/traces/board-trace-v1.json
WIN=${AXON_REL_WINDOW:-600}
if [ -n "$smoke" ]; then M=$MD/qwen2.5-0.5b-q4_k_m.gguf WIN=60; export AXON_KV_PAGES=512; fi
SMOKE_D=$HOME/.cache/axon-dist/rel01-smoke
harness_id() {   # the scripts tree this run uses, and anything uncommitted in it
  echo "scripts tree $(git -C "$W" rev-parse HEAD:scripts) dirty: $(git -C "$W" status --porcelain -- scripts | tr '\n' ' ')"
}
smoke_passed() {   # smoke mode: record that smoke $1 (agent | sb) passed on these scripts
  [ -n "$smoke" ] && harness_id > "$SMOKE_D/PASSED-$1"
  return 0
}
need_smoke() {   # a real run: only on committed scripts, after smoke $1 passed on the same scripts (shady, #270)
  [ -n "$smoke" ] && return 0
  [ -z "$(git -C "$W" status --porcelain -- scripts)" ] || { echo "STOP: uncommitted changes under scripts/"; exit 1; }
  local m=$SMOKE_D/PASSED-$1
  { [ -f "$m" ] && [ "$(cat "$m")" = "$(harness_id)" ]; } \
    || { echo "STOP: no smoke ($1) passed on these scripts ($(harness_id)); $m: $(cat "$m" 2>&1)"; exit 1; }
}
need_axon() {   # a real run measures only the staged release binary: rc/axon.sha256 is written from the tarball
  # the workflow checksummed (docs/log/2026-11-14-release-0.1-prep.md, "Release dry run"); a smoke takes any axon
  [ -n "$smoke" ] && return 0
  if [ "$AX" != "$D/rc/axon" ] || ! (cd "$D/rc" && sha256sum --quiet -c axon.sha256); then
    echo "STOP: $AX isn't the staged release binary (rc/axon.sha256)"; exit 1
  fi
}
hw() {   # the host, the GPU and every input's hash, into $1
  { date '+%F %T %Z'; nvidia-smi --query-gpu=name,driver_version,vbios_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv
    nproc; grep -m1 'model name' /proc/cpuinfo; uptime; free -m | head -2
    echo "harness $(git -C "$W" rev-parse HEAD), $(harness_id)"
    echo "env: $(env | grep -E '^(AXON_|LD_LIBRARY_PATH=)' | grep -v '^AXON_AGENT_TRAILER=' | sort | tr '\n' ' ')"
    "$AX" --version 2>&1 | head -1
    "$AX" kernels list 2>&1 | tail -3
    sha256sum "$AX" "$NEW/llama-server" "$NEW/libllama-server-impl.so" "$NEW/llama-bench" "$NEW/libggml-cuda.so" \
      "$OLD/llama-server" "$OLD/libllama-server-impl.so" "$OLD/llama-bench" "$OLD/libggml-cuda.so.0" "$M" "$TR"
    co=$(command -v cuobjdump || echo /opt/cuda/bin/cuobjdump)   # is sm_120 native in each build, or JIT'd PTX?
    for lib in "$NEW/libggml-cuda.so" "$OLD/libggml-cuda.so.0"; do
      echo "$lib: elf $("$co" --list-elf "$lib" 2>&1 | grep -o 'sm_[0-9]*[af]\?' | sort | uniq -c | tr -s ' \n' ' ')" \
           "ptx $("$co" --list-ptx "$lib" 2>&1 | grep -o 'sm_[0-9]*[af]\?' | sort | uniq -c | tr -s ' \n' ' ')"
    done
  } > "$1" 2>&1
}
