#!/bin/bash
# a1_h100.sh: WP-A1's ONE H100 job, the fast-GEMM probe (FP8 SS wgmma and W4A16 RS wgmma on the T3 skeleton,
# src/cuda/probe_fast.cu). Rules pre-registered in the GitHub issue "A1: fast GEMM probe, pre-registered rules".
# Submitted in the staged --src form (axon-podjob), so bella or a pod can run it:
#   axon-podjob --detach --timing h100 --models none --vram 8000 --ram 8000 --max-min 45 \
#     --name a1probe-<sha7>-h100 --expect 'A1 H100 DONE rc 0' \
#     -- scripts/a1/a1_h100.sh ~/.cache/axon-a1probe/axon-test-<sha7> scripts/a1/a1_verdict.py \
#        ~/.cache/axon-a1probe/a1-kit-<sha7>.tar
# Args (staged as ./name): the release test binary, the analysis script, the kit tarball:
#   san/            compute-sanitizer 13.2.76 (NVIDIA redist; the PyPI wheel can't start a target)
#   kcache-sm90a/   NVRTC 13.2 / 13.3 / 13.4 sm_90a cubins of the binary's KERNELS (plain, ext) and probe module,
#                   installed into the kernel cache on a cc 9.0 device only (the pod never compiles them)
#   BUILT_FOR       the binary the kit was built for (name, sha256, commit): another binary stops the job
#   SHA256SUMS      of everything in the kit (checked first)
# Stages: cuda_a1_session (compile check, sanitizer self-check, memcheck on each candidate's tiny shapes, tiny,
# ragged, peaks, the bench, the fallbacks; tests.rs), then a1_verdict.py on its log (table, validity, rule lines).
# Env (axon-run forwards AXON_* only): AXON_A1_BUDGET_S (the session's budget, default 2400 s),
# AXON_A1_REHEARSE_FALLBACK=1 (take the fallback path anyway: the bella rehearsal).
# Output: stdout / stderr only (results stay on the pod). Last line: `A1 H100 DONE rc 0 ...` (the --expect) or
# `A1 H100 DONE rc 1 FAILED (...)`.
set -u
T=${1:?test binary}; V=${2:?a1_verdict.py}; KIT=${3:?kit tarball}
T0=$(date +%s); log() { echo "[a1 +$(( $(date +%s) - T0 ))s] $*"; }
O=$PWD/a1-out-$(date +%m%d-%H%M%S)-$$; mkdir -p "$O"
# The staged dir (cwd) is content-addressed on the script and files, not on axon-run's options: a later submit of
# the same kit reuses it and ships whatever is left here. Keep only the logs: drop the extracted kit on any exit.
trap 'rm -rf "$O/kit"' EXIT
fail=""
done_line() {
  if [ -z "$fail" ]; then log "A1 H100 DONE rc 0 ($*)"; exit 0; fi
  log "A1 H100 DONE rc 1 FAILED ($fail)"; exit 1
}
log "kit $KIT, binary $T, out $O"
tar -xf "$KIT" -C "$O" || { fail="kit tarball unreadable"; done_line; }
K=$O/kit
( cd "$K" && sha256sum -c --quiet SHA256SUMS ) || { fail="kit SHA256SUMS mismatch"; done_line; }
want=$(awk '{print $2}' "$K/BUILT_FOR"); have=$(sha256sum "$T" | cut -c1-64)
if [ "$want" != "$have" ]; then fail="the kit was built for binary ${want:0:16}, this is ${have:0:16}"; done_line; fi
log "binary sha256 $have = the kit's BUILT_FOR: $(cat "$K/BUILT_FOR")"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv,noheader
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
CD=${XDG_CACHE_HOME:-$HOME/.cache}/axon/cuda; mkdir -p "$CD"
if [ "$cc" = "9.0" ]; then
  n0=$(ls "$CD" | wc -l); cp -n "$K"/kcache-sm90a/*.cubin "$CD/" 2>/dev/null
  log "kernel cache: $n0 -> $(ls "$CD" | wc -l) entries ($CD; the kit's sm_90a cubins installed)"
else
  log "compute capability $cc: not sm_90a, the kit's cubins are not installed (rehearsal: the sm_90a stages skip)"
fi
touch "$O/.t0"
SAN=$K/san/compute-sanitizer; [ -x "$SAN" ] || { log "no sanitizer in the kit: that stage is skipped"; SAN=; }
B=${AXON_A1_BUDGET_S:-2400}
log "session: budget $B s"
env ${SAN:+AXON_A1_SANITIZER=$SAN} AXON_A1_BUDGET_S="$B" timeout -k 10 $((B + 300)) "$T" \
  cuda::qwen2::tests::cuda_a1_session --exact --ignored --nocapture --test-threads 1 > "$O/session.log" 2>&1 &
pid=$!
tail -n +1 --pid=$pid -f "$O/session.log"
wait $pid; rc=$?
log "session rc $rc"
[ $rc = 0 ] || fail="session rc $rc"
grep -q '^A1 END ok' "$O/session.log" || fail="${fail:+$fail, }no 'A1 END ok'"
new=$(find "$CD" -maxdepth 1 -name '*.cubin' -newer "$O/.t0" 2>/dev/null)
if [ -z "$new" ]; then echo "A1 CUBIN new cache entries during the run: none"
else echo "A1 CUBIN new cache entries during the run (compiled here, not from the kit):"; for x in $new; do ls -l "$x"; done; fi
log "analysis: $V"
python3 "$V" "$O/session.log"; vrc=$?
[ $vrc = 0 ] || fail="${fail:+$fail, }a1_verdict.py rc $vrc"
done_line "session ok, analysis ok"
