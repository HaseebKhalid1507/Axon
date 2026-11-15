# shellcheck shell=bash disable=SC2034  # sourced: the variables are the callers'
# common.sh (sourced): release 0.1's boards on bella (docs/log/2026-11-14-release-0.1-prep.md). The paths are
# bella's: llama.cpp builds, models, the agent trace (never in this repo). Outputs go to $AXON_REL_OUT. The axon under
# test is the release dry run's own binary (the workflow's tarball), AXON_REL_AXON. Settings are AXON_REL_* because
# axon-run forwards only AXON_* variables to a job.
# AXON_REL_SMOKE=1: the mechanics only, next to other GPU jobs: the 0.5B, a 60-s window, a 512-page pool
# (AXON_KV_PAGES), outputs under rel01-smoke.
S=$(cd "$(dirname "$0")" && pwd)
W=$(cd "$S/../../.." && pwd)
smoke=; [ "${AXON_REL_SMOKE:-0}" = 1 ] && smoke=-smoke
D=${AXON_REL_OUT:-$HOME/.cache/axon-dist/rel01$smoke}
AX=${AXON_REL_AXON:-$D/rc/axon}
NEW=${AXON_REL_LLAMA_NEW:-$HOME/.cache/axon-dist/llama-b12387/x/llama-b12387}    # upstream b12387 (a956a852c), official cuda-13.5 build
OLD=${AXON_REL_LLAMA_OLD:-$HOME/Projects/llama.cpp/build-cuda/bin}                # bella's 4413a19 build (the 10-27 board, the weekly)
MD=$HOME/Projects/axon/models
M=${AXON_REL_MODEL:-$MD/Qwen2.5-7B-Instruct-Q4_K_M.gguf}
TR=$HOME/.cache/axon-agent/traces/board-trace-v1.json
WIN=${AXON_REL_WINDOW:-600}
if [ -n "$smoke" ]; then M=$MD/qwen2.5-0.5b-q4_k_m.gguf WIN=60; export AXON_KV_PAGES=512; fi
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
    echo "harness $(git -C "$W" rev-parse HEAD) dirty: $(git -C "$W" status --porcelain -- scripts | tr '\n' ' ')"
    "$AX" --version 2>&1 | head -1
    sha256sum "$AX" "$NEW/llama-server" "$NEW/libllama-server-impl.so" "$NEW/llama-bench" "$NEW/libggml-cuda.so" \
      "$OLD/llama-server" "$OLD/libllama-server-impl.so" "$OLD/llama-bench" "$OLD/libggml-cuda.so.0" "$M" "$TR"
  } > "$1" 2>&1
}
