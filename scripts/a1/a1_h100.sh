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
# ragged, peaks, the bench, the fallbacks; tests.rs; every stage a child process), then a1_verdict.py on its log
# and the nvidia-smi trace (table, clocks, validity, rule lines). An nvidia-smi trace (SM / memory clock, power,
# temperature, clock-event reasons every 500 ms) runs over the whole session; the verdict summarises the bench
# window(s). The device line (name, power limits) is appended to the session log as `A1 SMI` (V1 reads it).
# Env (axon-run forwards AXON_* only): AXON_A1_BUDGET_S (the session's budget, default 2400 s),
# AXON_A1_REHEARSE_FALLBACK=1 (take the fallback path anyway: the bella rehearsal).
# Output: stdout / stderr only (results stay on the pod). THE RESULT IS THE `A1 VERDICT` LINE. The last line is
# `A1 H100 DONE rc 0 (verdict printed; stage failures: none | <list>)` whenever a1_verdict.py printed a verdict
# (any verdict, VOID included: stage failures, e.g. a candidate's tolerance miss, are listed, not fatal), or
# `A1 H100 DONE rc 1 FAILED (no verdict: ...)` when there is none (kit or binary mismatch, no bench data,
# the analysis crashed). --expect 'A1 H100 DONE rc 0' = a verdict exists.
set -u
T=${1:?test binary}; V=${2:?a1_verdict.py}; KIT=${3:?kit tarball}
T0=$(date +%s); log() { echo "[a1 +$(( $(date +%s) - T0 ))s] $*"; }
O=$PWD/a1-out-$(date +%m%d-%H%M%S)-$$; mkdir -p "$O"
# The staged dir (cwd) is content-addressed on the script and files, not on axon-run's options: a later submit of
# the same kit reuses it and ships whatever is left here. Keep only the logs: drop the extracted kit on any exit.
trap 'rm -rf "$O/kit"; [ -n "${smipid:-}" ] && kill "$smipid" 2>/dev/null' EXIT
fail=""      # no verdict possible: rc 1
stagefail="" # the session's own failures: listed, the verdict still stands
done_line() {
  if [ -z "$fail" ]; then log "A1 H100 DONE rc 0 (verdict printed; stage failures: ${stagefail:-none})"; exit 0; fi
  log "A1 H100 DONE rc 1 FAILED (no verdict: $fail${stagefail:+; stage failures: $stagefail})"; exit 1
}
log "kit $KIT, binary $T, out $O"
tar -xf "$KIT" -C "$O" || { fail="kit tarball unreadable"; done_line; }
K=$O/kit
( cd "$K" && sha256sum -c --quiet SHA256SUMS ) || { fail="kit SHA256SUMS mismatch"; done_line; }
want=$(awk '{print $2}' "$K/BUILT_FOR"); have=$(sha256sum "$T" | cut -c1-64)
if [ "$want" != "$have" ]; then fail="the kit was built for binary ${want:0:16}, this is ${have:0:16}"; done_line; fi
log "binary sha256 $have = the kit's BUILT_FOR: $(cat "$K/BUILT_FOR")"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,power.max_limit --format=csv
SMI="A1 SMI $(nvidia-smi --query-gpu=name,power.limit,power.max_limit,clocks.max.sm --format=csv,noheader 2>&1 | head -1)"
echo "$SMI"
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
Q=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active
nvidia-smi --query-gpu=$Q --format=csv > /dev/null 2>&1 || Q=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu
nvidia-smi --query-gpu=$Q --format=csv -lms 500 > "$O/smi-trace.csv" 2>&1 &
smipid=$!
log "session: budget $B s (nvidia-smi trace pid $smipid: $Q)"
env ${SAN:+AXON_A1_SANITIZER=$SAN} AXON_A1_BUDGET_S="$B" timeout -k 10 $((B + 300)) "$T" \
  cuda::qwen2::tests::cuda_a1_session --exact --ignored --nocapture --test-threads 1 > "$O/session.log" 2>&1 &
pid=$!
tail -n +1 --pid=$pid -f "$O/session.log"
wait $pid; rc=$?
kill $smipid 2>/dev/null; wait $smipid 2>/dev/null
log "session rc $rc; nvidia-smi trace $(($(wc -l < "$O/smi-trace.csv") - 1)) samples"
echo "$SMI" >> "$O/session.log"
[ $rc = 0 ] || stagefail="session rc $rc"
grep -q '^A1 END ok' "$O/session.log" || stagefail="${stagefail:+$stagefail, }$(grep -m1 '^A1 END' "$O/session.log" || echo "no 'A1 END' line")"
new=$(find "$CD" -maxdepth 1 -name '*.cubin' -newer "$O/.t0" 2>/dev/null)
if [ -z "$new" ]; then echo "A1 CUBIN new cache entries during the run: none"
else echo "A1 CUBIN new cache entries during the run (compiled here, not from the kit):"; for x in $new; do ls -l "$x"; done; fi
log "analysis: $V"
python3 "$V" "$O/session.log" "$O/smi-trace.csv" > "$O/verdict.txt" 2>&1; vrc=$?
cat "$O/verdict.txt"
[ $vrc = 0 ] || fail="a1_verdict.py rc $vrc"
grep -q '^A1 VERDICT: ' "$O/verdict.txt" || fail="${fail:+$fail, }no 'A1 VERDICT' line"
grep '^A1 VERDICT: ' "$O/verdict.txt" | sed 's/^/[a1] RESULT: /'
done_line
