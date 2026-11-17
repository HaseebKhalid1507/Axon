# shellcheck shell=bash disable=SC2034  # sourced: the variables are the callers'
# common.sh (sourced): release 0.1's boards (docs/log/2026-11-14-release-0.1-prep.md). Two host profiles:
#   pod (AXON_REL_POD=1, set by pod.sh; amendment 3, 11-17): a RunPod 4090 (axon-run --timing 4090, through
#     axon-podjob). Every input is staged next to the job or fetched by pod.sh, sha256-checked: the release binary
#     ./axon, ./board-trace-v1.json, the 4413a19 sm_89 build and the CUDA 12.8.1 runtime it links, upstream
#     b12387's official build; the models are the pod's own (./models, models.tsv rows, sha256 = HF's). Outputs go
#     to $AXON_REL_OUT in the job's run dir and come back to bella in the job's log (pod.sh, queue.py).
#   bella (default): the 11-14/11-15 runs. The paths are bella's: llama.cpp builds, models, the agent trace (never in
#     this repo). Outputs go to $AXON_REL_OUT.
# The axon under test is the release dry run's own binary (the workflow's tarball): REL_AXON_SHA256 below. Settings
# are AXON_REL_* because axon-run forwards only AXON_* variables to a job.
# AXON_REL_SMOKE=1: the mechanics only: the 0.5B, a 60-s window, a 512-page pool (AXON_KV_PAGES). A real run starts
# only after a smoke of the same scripts passed (need_smoke) and takes the pre-registered protocol: no AXON_REL_*
# setting may change it.
S=$(cd "$(dirname "$0")" && pwd)
W=$(cd "$S/../../.." && pwd)
smoke=; [ "${AXON_REL_SMOKE:-0}" = 1 ] && smoke=-smoke
POD=; [ "${AXON_REL_POD:-0}" = 1 ] && POD=1
# The release candidate: the `axon` in main 9f3630c's release dry run (run 43657907368; main 1a1d751's, run
# 44104636797, is the same bytes). need_axon refuses any other binary in a real run.
REL_AXON_SHA256=2c19415bf329b30fa2e2805e9a0aa3d17321449b154010fa9241d67f7f31d245
# No NVRTC: the release binary's own kernel pack or a failure (a broken pack must fail the board, not hide behind a
# host's NVRTC). No inherited library path: each llama.cpp build gets its own (--llama-env, the llama-bench wrappers),
# nothing else runs under it.
export AXON_KERNEL_PACK=only
unset LD_LIBRARY_PATH
if [ -z "$smoke" ]; then
  for v in AXON_REL_ROUNDS AXON_REL_SPEEDS AXON_REL_SB_MODELS AXON_REL_SB_ROUNDS AXON_REL_MODEL AXON_REL_WINDOW \
           AXON_REL_LLAMA_NEW AXON_REL_LLAMA_OLD; do
    [ -z "${!v:-}" ] || { echo "STOP: $v is set: a real run takes the pre-registered protocol"; exit 1; }
  done
fi
if [ -n "$POD" ]; then
  # pod.sh sets these: the job's run dir (staged files, models/ = the pod's), its cache (fetched builds), the harness
  # tarball's sha256 (this tree), and the smoke that passed on it (queue.py: the real queue's jobs only)
  RUN=${AXON_REL_RUN:?pod.sh sets AXON_REL_RUN}
  PC=${AXON_REL_CACHE:?pod.sh sets AXON_REL_CACHE}
  D=${AXON_REL_OUT:-$RUN/out}
  AX=$RUN/axon
  NEW=$PC/llama-b12387/llama-b12387           # upstream b12387 (a956a852c), official cuda-13.5 build + its cudart
  OLD=$PC/llama-4413a19-sm89                  # 4413a19 built for sm_89 (axon-weekly's build, the weekly pod board's)
  OLD_LIB=$PC/cuda-12.8.1/lib                 # the CUDA 12.8.1 runtime that build links (cudart, cuBLAS)
  MD=$RUN/models
  TR=$RUN/board-trace-v1.json
  SMOKE_D=$D
else
  D=${AXON_REL_OUT:-$HOME/.cache/axon-dist/rel01$smoke}
  AX=${AXON_REL_AXON:-$D/rc/axon}
  NEW=${AXON_REL_LLAMA_NEW:-$HOME/.cache/axon-dist/llama-b12387/x/llama-b12387}    # upstream b12387 (a956a852c), official cuda-13.5 build
  OLD=${AXON_REL_LLAMA_OLD:-$HOME/Projects/llama.cpp/build-cuda/bin}                # bella's 4413a19 build (the 10-27 board, the weekly)
  OLD_LIB=
  MD=$HOME/Projects/axon/models
  TR=$HOME/.cache/axon-agent/traces/board-trace-v1.json
  SMOKE_D=$HOME/.cache/axon-dist/rel01-smoke
fi
M=${AXON_REL_MODEL:-$MD/Qwen2.5-7B-Instruct-Q4_K_M.gguf}
WIN=${AXON_REL_WINDOW:-600}
if [ -n "$smoke" ]; then M=$MD/qwen2.5-0.5b-q4_k_m.gguf WIN=60; export AXON_KV_PAGES=512; fi
# llama-server's environment for a build (agent_replay --llama-env): its own library dir, nothing inherited
llama_env() {   # new | old -> the --llama-env args
  if [ "$1" = new ]; then echo "--llama-env LD_LIBRARY_PATH=$NEW"
  elif [ -n "$OLD_LIB" ]; then echo "--llama-env LD_LIBRARY_PATH=$OLD_LIB"; fi
}
harness_id() {   # the scripts this run uses, and anything uncommitted in them
  if [ -n "$POD" ]; then echo "harness tar sha256 ${AXON_REL_HARNESS:?pod.sh sets AXON_REL_HARNESS} (commit ${AXON_REL_HARNESS_COMMIT:-?})"
  else echo "scripts tree $(git -C "$W" rev-parse HEAD:scripts) dirty: $(git -C "$W" status --porcelain -- scripts | tr '\n' ' ')"; fi
}
smoke_passed() {   # smoke mode: record that smoke $1 (agent | sb) passed on these scripts
  [ -n "$smoke" ] || return 0
  if [ -n "$POD" ]; then echo "SMOKE PASSED $1 $(harness_id)"   # queue.py reads it from the job's log
  else harness_id > "$SMOKE_D/PASSED-$1"; fi
  return 0
}
need_smoke() {   # a real run: only on committed scripts, after smoke $1 passed on the same scripts (shady, #270)
  [ -n "$smoke" ] && return 0
  if [ -n "$POD" ]; then   # queue.py passes the harness its smoke passed on; the tarball is `git archive` of a commit
    [ "${AXON_REL_SMOKE_OK:-}" = "$AXON_REL_HARNESS" ] \
      || { echo "STOP: no smoke passed on this harness ($(harness_id); smoke passed on '${AXON_REL_SMOKE_OK:-none}')"; exit 1; }
    return 0
  fi
  [ -z "$(git -C "$W" status --porcelain -- scripts)" ] || { echo "STOP: uncommitted changes under scripts/"; exit 1; }
  local m=$SMOKE_D/PASSED-$1
  { [ -f "$m" ] && [ "$(cat "$m")" = "$(harness_id)" ]; } \
    || { echo "STOP: no smoke ($1) passed on these scripts ($(harness_id)); $m: $(cat "$m" 2>&1)"; exit 1; }
}
need_axon() {   # a real run measures only the release binary (the workflow's checksummed tarball); a smoke takes any axon
  [ -n "$smoke" ] && return 0
  if [ -n "$POD" ]; then
    [ "$(sha256sum < "$AX" | cut -d' ' -f1)" = "$REL_AXON_SHA256" ] || { echo "STOP: $AX isn't the release binary"; exit 1; }
  elif [ "$AX" != "$D/rc/axon" ] || ! (cd "$D/rc" && sha256sum --quiet -c axon.sha256); then
    echo "STOP: $AX isn't the staged release binary (rc/axon.sha256)"; exit 1
  fi
}
hw() {   # the host, the GPU and every input's hash, into $1
  { date '+%F %T %Z'; nvidia-smi --query-gpu=name,driver_version,vbios_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv
    echo "host $(hostname) pod ${RUNPOD_POD_ID:-none}"
    nproc; grep -m1 'model name' /proc/cpuinfo; uptime; free -m | head -2
    if [ -n "$POD" ]; then echo "harness: $(harness_id)"; else echo "harness $(git -C "$W" rev-parse HEAD), $(harness_id)"; fi
    echo "env: $(env | grep -E '^(AXON_|LD_LIBRARY_PATH=)' | grep -v '^AXON_AGENT_TRAILER=' | sort | tr '\n' ' ')"
    "$AX" --version 2>&1 | head -1
    "$AX" kernels list 2>&1 | tail -3
    for f in "$AX" "$NEW/llama-server" "$NEW/libllama-server-impl.so" "$NEW/llama-bench" "$NEW/libggml-cuda.so" \
             "$OLD/llama-server" "$OLD/libllama-server-impl.so" "$OLD/llama-bench" "$OLD/libggml-cuda.so.0" "$M" "$TR"; do
      [ -e "$f" ] && sha256sum "$f"
    done
    if [ -n "$OLD_LIB" ]; then   # what each llama.cpp build loads, under the library path it gets
      echo "b12387: $(LD_LIBRARY_PATH=$NEW ldd "$NEW/llama-server" | grep -E 'cuda|cublas|not found' | tr -s ' \t\n' ' ')"
      echo "4413a19: $(LD_LIBRARY_PATH=$OLD_LIB ldd "$OLD/llama-server" | grep -E 'cuda|cublas|not found' | tr -s ' \t\n' ' ')"
      (cd "$PC" && sha256sum cuda-12.8.1/lib/* 2>&1)
    fi
    co=$(command -v cuobjdump || echo /opt/cuda/bin/cuobjdump)   # is the GPU's arch native in each build, or JIT'd PTX?
    if [ -x "$co" ]; then
      for lib in "$NEW/libggml-cuda.so" "$OLD/libggml-cuda.so.0"; do
        [ -e "$lib" ] || continue
        echo "$lib: elf $("$co" --list-elf "$lib" 2>&1 | grep -o 'sm_[0-9]*[af]\?' | sort | uniq -c | tr -s ' \n' ' ')" \
             "ptx $("$co" --list-ptx "$lib" 2>&1 | grep -o 'sm_[0-9]*[af]\?' | sort | uniq -c | tr -s ' \n' ' ')"
      done
    else echo "cuobjdump: none here (b12387's libggml-cuda.so: native sm_89, cuobjdump on bella 11-14; 4413a19: built for sm_89)"; fi
  } > "$1" 2>&1
}
