#!/bin/bash
# pod.sh: release 0.1's boards on a RunPod 4090, one cell per job (amendment 3, 11-17: bella's desktop tainted its runs;
# docs/log/2026-11-14-release-0.1-prep.md). queue.py submits it through axon-podjob (--timing 4090):
#
#   pod.sh ./axon ./board-trace-v1.json ./harness.tar ./llama-4413a19-sm89.tar ./SHA256SUMS CELL_SCRIPT ARGS...
#
# The five files are staged on bella by queue.py (the same five for every job, so the pod gets them once):
#   axon                    the release candidate (common.sh REL_AXON_SHA256; need_axon checks it)
#   board-trace-v1.json     the agent trace (sha256 below)
#   harness.tar             `git archive` of the commit queue.py runs: scripts/agent_replay.py, scripts/release/boards/
#   llama-4413a19-sm89.tar  axon-weekly's 4413a19 build for sm_89 (the weekly pod board's; sha256s below)
#   SHA256SUMS              of the four
# Fetched here once per pod, sha256-pinned below, into $AXON_REL_CACHE (default ~/axon-node/cache/axon-rel01):
#   upstream llama.cpp b12387's official ubuntu-cuda-13.5-x64 build and its cudart/cuBLAS tarball (GitHub release);
#   the CUDA 12.8.1 runtime the 4413a19 build links (libcudart 12.8.90, libcublas 12.8.4.1: NVIDIA's redist).
# The models are the pod's own (./models: models.tsv rows, sha256 = Hugging Face's; axon-run --models). It runs
# CELL_SCRIPT (single_stream.sh | agent_sweep.sh | agent_board.sh) from the harness with common.sh's pod profile, then
# prints out/ through payload.py (queue.py takes it from the job's log) and exits with the cell's rc.
# Inputs from earlier cells come in the environment: AXON_REL_IN_POOL (sweep/POOL), AXON_REL_IN_WINNER (sweep/
# WINNER.env, base64). AXON_REL_SMOKE_OK: the harness sha256 the smoke passed on (common.sh need_smoke).
set -u
RUN=$PWD
die() { echo "STOP: $*"; exit 1; }
want=(./axon ./board-trace-v1.json ./harness.tar ./llama-4413a19-sm89.tar ./SHA256SUMS)
[ $# -ge 6 ] || die "usage: pod.sh ${want[*]} CELL_SCRIPT ARGS..."
for w in "${want[@]}"; do [ "$1" = "$w" ] || die "argument '$1': want $w"; shift; done
CELL=$1; shift
case $CELL in single_stream.sh|agent_sweep.sh|agent_board.sh) ;; *) die "no cell script '$CELL'";; esac
TRACE_SHA256=713285ba54487e17035d6757b5e76391b568b45dce776925d699ede3ae3cccdc
OLD_SERVER_SHA256=56e3da531db339d948a6ad78bdbb157af4d8fc2f61e1748945ffd572e8a7d1a2
OLD_BENCH_SHA256=b5ad496b9c6ee4d7edcc5a9511754fa582130e26c61d7d0882016e8101fb19b8
B=https://github.com/ggml-org/llama.cpp/releases/download/b12387
NEW_TGZ="llama-b12387-bin-ubuntu-cuda-13.5-x64.tar.gz a2a1df59bcbec3239e052b1b22c4ccb8877de308d891f9ba284c5e19c8824f5e"
NEW_RT="cudart-llama-b12387-bin-ubuntu-cuda-13.5-x64.tar.gz ed84c96b7a0f885365d5fa9e0496404b6f3b0317644b81adcf8081f95ee1fdb6"
R=https://developer.download.nvidia.com/compute/cuda/redist
CU_RT="cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz 8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68"
CU_BL="libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.4.1-archive.tar.xz 21718957c2cf000bacd69d36c95708a2319199e39e056f8b4f0f68e3b9f323bb"

echo "POD JOB $CELL $* on $(hostname) pod ${RUNPOD_POD_ID:-none} $(date '+%F %T %Z')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
sha256sum --quiet -c SHA256SUMS || die "a staged input differs from SHA256SUMS"
[ "$(sha256sum < board-trace-v1.json | cut -d' ' -f1)" = $TRACE_SHA256 ] || die "the trace isn't board-trace-v1.json"
HS=$(sha256sum < harness.tar | cut -d' ' -f1)
if ! mkdir h || ! tar -xf harness.tar -C h; then die "can't unpack harness.tar"; fi
cmp -s "$0" h/scripts/release/boards/pod.sh || die "this pod.sh isn't the harness's own"
echo "harness.tar sha256 $HS (commit ${AXON_REL_HARNESS_COMMIT:-?})"

PC=${AXON_REL_CACHE:-$HOME/axon-node/cache/axon-rel01}
mkdir -p "$PC/dl" || die "can't make $PC"
get() {   # URL FILE SHA256: into $PC/dl, resumed, checked
  local t
  # already here and checked (a run that stopped after its fetch): no second download
  [ -f "$PC/dl/$2" ] && [ "$(sha256sum < "$PC/dl/$2" | cut -d' ' -f1)" = "$3" ] && return 0
  for t in 1 2 3; do
    curl -fsSL --retry 5 --connect-timeout 30 -C - -o "$PC/dl/$2" "$1" \
      && [ "$(sha256sum < "$PC/dl/$2" | cut -d' ' -f1)" = "$3" ] && return 0
    echo "fetch $2: try $t failed"; rm -f "$PC/dl/$2"
  done
  return 1
}
setup() {   # once per pod (a marker per input set); under flock: two jobs never set up at once
  local avail; avail=$(df -Pk "$PC" | awk 'NR == 2 {print int($4 / 1048576)}')
  if [ ! -f "$PC/llama-b12387/OK" ]; then
    [ "$avail" -ge 4 ] || die "$avail GB free under $PC: 4 needed"
    rm -rf "$PC/llama-b12387" "$PC/llama-b12387.tmp"; mkdir -p "$PC/llama-b12387.tmp"
    for x in "$NEW_TGZ" "$NEW_RT"; do
      read -r f h <<< "$x"; get "$B/$f" "$f" "$h" || die "fetch $f"
      tar -xzf "$PC/dl/$f" -C "$PC/llama-b12387.tmp" || die "unpack $f"
      rm -f "$PC/dl/$f"
    done
    if ! mv "$PC"/llama-b12387.tmp/cudart-llama-b12387-bin-ubuntu-cuda-13.5-x64/* "$PC/llama-b12387.tmp/llama-b12387/" \
       || ! rmdir "$PC/llama-b12387.tmp/cudart-llama-b12387-bin-ubuntu-cuda-13.5-x64"; then die "b12387 layout"; fi
    [ -x "$PC/llama-b12387.tmp/llama-b12387/llama-server" ] && [ -e "$PC/llama-b12387.tmp/llama-b12387/libcudart.so.13" ] \
      || die "b12387: no llama-server or libcudart.so.13"
    echo "$NEW_TGZ $NEW_RT" > "$PC/llama-b12387.tmp/OK"; mv "$PC/llama-b12387.tmp" "$PC/llama-b12387"
    echo "b12387: fetched and checked"
  fi
  if [ ! -f "$PC/cuda-12.8.1/OK" ]; then
    rm -rf "$PC/cuda-12.8.1"; mkdir -p "$PC/cuda-12.8.1/lib"
    for x in "$CU_RT" "$CU_BL"; do
      read -r f h <<< "$x"; get "$R/$f" "${f##*/}" "$h" || die "fetch ${f##*/}"
      # only the runtime libraries, flat; regular files, and symlinks to a runtime library in the same directory
      # (relative; NVIDIA's archives write libcublas.so.12 -> ./libcublas.so.12.8.4.1, cudart's without the ./);
      # absolute or escaping links refused (tarfile, no xz binary needed)
      python3 - "$PC/dl/${f##*/}" "$PC/cuda-12.8.1/lib" <<'PY' || die "unpack ${f##*/}"
import os, re, sys, tarfile
want = re.compile(r'^lib(cudart|cublas|cublasLt)\.so\.12(\.[0-9.]+)?$')


def same_dir(link):
    """The link's target if it names a runtime library in the link's own directory, else None."""
    if not link or os.path.isabs(link):
        return None
    n = os.path.normpath(link)
    return n if '/' not in n and n not in ('.', '..') and want.match(n) else None


n = 0
with tarfile.open(sys.argv[1]) as t:
    for m in t.getmembers():
        b = os.path.basename(m.name)
        if '/lib/' not in m.name or not want.match(b):
            continue
        if m.issym() and same_dir(m.linkname):
            m.linkname = same_dir(m.linkname)
        elif not m.isfile():
            sys.exit(f'refused: {m.name} (type {m.type!r}, link {m.linkname!r})')
        m.name = b
        t.extract(m, sys.argv[2], filter='data')
        n += 1
print(f'{sys.argv[1]}: {n} runtime files')
PY
      rm -f "$PC/dl/${f##*/}"
    done
    for f in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do [ -e "$PC/cuda-12.8.1/lib/$f" ] || die "no $f"; done
    echo "$CU_RT $CU_BL" > "$PC/cuda-12.8.1/OK"
  fi
  local ot; ot=$(sha256sum < llama-4413a19-sm89.tar | cut -d' ' -f1)
  if [ "$(cat "$PC/llama-4413a19-sm89/OK" 2>/dev/null)" != "$ot" ]; then
    rm -rf "$PC/llama-4413a19-sm89"; tar -xf llama-4413a19-sm89.tar -C "$PC" || die "unpack llama-4413a19-sm89.tar"
    [ "$(sha256sum < "$PC/llama-4413a19-sm89/llama-server" | cut -d' ' -f1)" = $OLD_SERVER_SHA256 ] \
      && [ "$(sha256sum < "$PC/llama-4413a19-sm89/llama-bench" | cut -d' ' -f1)" = $OLD_BENCH_SHA256 ] \
      || die "the 4413a19 sm_89 build isn't the pinned one"
    echo "$ot" > "$PC/llama-4413a19-sm89/OK"
  fi
}
exec 9> "$PC/.lock"; flock 9; ( setup ) || exit 1; flock -u 9

case $CELL in   # the pod's models (axon-run --models): a missing one is a STOP, not a skipped cell
  single_stream.sh) need="qwen2.5-0.5b-q4_k_m.gguf qwen2.5-1.5b-instruct-q4_k_m.gguf qwen2.5-3b-instruct-q4_k_m.gguf
                          Qwen2.5-7B-Instruct-Q4_K_M.gguf Qwen3-8B-Q4_K_M.gguf" ;;
  *) need=Qwen2.5-7B-Instruct-Q4_K_M.gguf; [ "${AXON_REL_SMOKE:-0}" = 1 ] && need=qwen2.5-0.5b-q4_k_m.gguf ;;
esac
for f in $need; do [ -s "models/$f" ] || die "this pod has no models/$f"; done

O=$RUN/out
mkdir -p "$O/sweep"
[ -n "${AXON_REL_IN_POOL:-}" ] && echo "$AXON_REL_IN_POOL" > "$O/sweep/POOL"
[ -n "${AXON_REL_IN_WINNER:-}" ] && { echo "$AXON_REL_IN_WINNER" | base64 -d > "$O/sweep/WINNER.env" || die "AXON_REL_IN_WINNER"; }
AXON_REL_POD=1 AXON_REL_RUN=$RUN AXON_REL_CACHE=$PC AXON_REL_OUT=$O AXON_REL_HARNESS=$HS \
  bash "h/scripts/release/boards/$CELL" "$@"
rc=$?
rm -rf "$O/sbnode"
tar -czf result.tgz out || die "can't pack out/"
if [ "$(stat -c %s result.tgz)" -gt 30000000 ]; then   # never expected (a board round is ~1 MB): drop the engine logs
  echo "out/ packs to $(stat -c %s result.tgz) bytes: sending it without *.log"
  tar -czf result.tgz --exclude='*.log' out || die "can't pack out/"
fi
python3 h/scripts/release/boards/payload.py emit result.tgz "$CELL:$(IFS=,; echo "$*")" || die "payload"
echo "POD END rc $rc"
exit $rc
