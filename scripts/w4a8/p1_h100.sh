#!/bin/bash
# p1_h100.sh: session P1 of docs/design/w4a8-prefill.md (branch feat/w4a8-prefill), ONE axon-run job:
#   axon-run --timing h100 --src <this kit> --models Qwen2.5-7B-Instruct-Q4_K_M.gguf --max-min 22 -- /usr/bin/bash p1_h100.sh
# Kit (cwd): axon-test (the candidate's test binary; sha in SHA256SUMS), axon-main (main's `axon` at the merge base,
# for the serve cells), kcache-sm90a/ (NVRTC 13.2/13.3/13.4 sm_90a: both trees' KERNELS), san/ (compute-sanitizer
# 13.2.76 redist), sd/ (sd_client.py + prompts-mix-v2.jsonl, #165's kit), passhist.py.
# 1. cuda_w4a8_p1_session (candidate): compile + ptxas, memcheck on the tiny launches, bits, Checkpoint 1 (the sweep,
#    every cell bit-checked; prints W4A8 S / W4A8 T / W4A8 CHECKPOINT1 ... VERDICT / W4A8 Q6 COPY).
# 2. Two serve cells on MAIN's binary, explicit flags (section 4, M4): fifo60 and sjf80, C=128, one round each, the
#    #165 warm-ups; their traces -> PASSHIST / PASSMULTI / FEEDSHARE (passhist.py, the measured cell only).
# Then `scripts/w4a8_coverage_lp.py p1` (local, on the job log) gives the post-P1 PROCEED / STOP.
# The serve cells use #165's exact config (devstack feat/h100-r11 runbooks/h100-r11/r11.sh serve_arm), pinned here with
# no override: Qwen2.5-7B-Instruct-Q4_K_M, `serve --parallel 128 --ctx 8192 --max-queue 256 --max-conns 256` (no -n),
# warm-ups `--c 8 --n 16 --long 4` and `--c 128 --n 128`, the cell `--workload mix --c 128 --ctx 8192` with r11's SLO
# flags, plus the explicit scheduler flags per arm. The bella rehearsal runs the same (r11's own bella rehearsal did).
# Env (axon-run forwards AXON_* only): AXON_P1_SKIP ("session" / "serve"), AXON_W4A8_BUDGET_S (session stages, 720).
# Success line (the job's --expect): `P1 H100 DONE rc 0 cells fifo60,sjf80 ok`; any failure prints rc 1 and FAILED.
set -u
KIT=$(cd "$(dirname "$0")" && pwd); cd "$KIT"
T=$KIT/axon-test; A=$KIT/axon-main
MD=/home/haseeb/Projects/axon/models; [ -d /root/axon-node/models ] && MD=/root/axon-node/models
M=$MD/Qwen2.5-7B-Instruct-Q4_K_M.gguf
PAR=128; CTX=8192; C=128; SKIP=${AXON_P1_SKIP:-}   # r11.sh's serve / cell config, pinned
PORT=$((19500 + RANDOM % 400))
O=$KIT/out-$(date +%m%d-%H%M%S); mkdir -p $O/trace
T0=$(date +%s); log() { echo "[p1 +$(( $(date +%s) - T0 ))s] $*"; }
unset AXON_W4A8 AXON_W4A8_T AXON_ADMIT AXON_STEP_COST AXON_SERVE_TRACE
CD=${XDG_CACHE_HOME:-$HOME/.cache}/axon/cuda; mkdir -p "$CD"
n0=$(ls "$CD" | wc -l); cp -n kcache-sm90a/*.cubin "$CD/" 2>/dev/null; log "kernel cache: $n0 -> $(ls "$CD" | wc -l) entries ($CD)"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm --format=csv,noheader
sha256sum axon-test axon-main sd/* passhist.py | sed 's|  .*/|  |'; grep '^src ' SHA256SUMS
log "model $M ($(sha256sum "$M" 2>/dev/null | cut -c1-16)); serve --parallel $PAR --ctx $CTX, cells C=$C"
rc=0
if [[ " $SKIP " != *" session "* ]]; then
  SAN=$KIT/san/compute-sanitizer; [ -x "$SAN" ] || { log "no sanitizer in the kit: stage skipped"; SAN=; }
  env ${SAN:+AXON_PROBE_SANITIZER=$SAN} timeout -k 10 780 ./axon-test cuda::qwen2::tests::cuda_w4a8_p1_session \
    --exact --ignored --nocapture --test-threads 1 || rc=1
  log "session rc $rc"
fi
serve_cell() {  # serve_cell ARM FLAGS: one main server (r11.sh's flags + the arm's), r11's warm-ups, one C=128 cell,
  # checked (every request sent and ok), then its trace -> passhist lines. rc 0 only if all of that held.
  local arm=$1 f=$2 port=$((PORT + ${#arm})); local url=http://127.0.0.1:$port tr=$O/trace/$arm.jsonl
  ( export AXON_SERVE_TRACE=$tr; exec "$A" serve "$M" --device cuda --host 127.0.0.1 --port "$port" --parallel 128 \
      --ctx 8192 --max-queue 256 --max-conns 256 --alias qwen2.5-7b-q4km --quiet $f ) > "$O/serve-$arm.log" 2>&1 &
  local pid=$! i up=0
  for i in $(seq 300); do
    kill -0 $pid 2>/dev/null || break
    curl -sf "$url/health" > /dev/null 2>&1 && { up=1; break; }
    sleep 1
  done
  if [ $up = 0 ]; then log "FATAL serve $arm never came up"; tail -5 "$O/serve-$arm.log"; kill $pid 2>/dev/null; return 1; fi
  log "serve $arm up after ${i}s ($f)"
  local r2=0 out=$O/cell-$arm.out
  python3 sd/sd_client.py warm "$url" --c 8 --n 16 --long 4 --ctx 8192 > /dev/null || { log "FATAL warm-up 1 failed ($arm)"; r2=1; }
  python3 sd/sd_client.py warm "$url" --c 128 --n 128 --ctx 8192 > /dev/null || { log "FATAL warm-up 2 failed ($arm)"; r2=1; }
  echo "{\"mark\":\"cell\",\"arm\":\"$arm\",\"round\":1,\"c\":128}" >> $tr
  SD_SALT="p1 c128 fixed" python3 sd/sd_client.py cell "$url" --workload mix --c 128 --round 1 --ctx 8192 \
    --out "$O" --tag "$arm" --slo-itl-ms 50 --slo-ttft-ms 2000 --slo-ttft-ms-per-ktok 250 > "$out" 2>&1
  local crc=$?
  echo "{\"mark\":\"end\"}" >> $tr
  kill $pid; wait $pid 2>/dev/null
  tail -1 "$out" | cut -c1-300 | sed "s/^/CELL $arm c128 /"
  # The cell counts only if sd_client exited 0 and every request was sent and came back ok.
  local sent
  sent=$(tail -1 "$out" | python3 -c "import json,sys
try:
    d = json.loads(sys.stdin.read())
    n, ok = int(d.get('requests') or 0), int(d.get('ok') or 0)
    print('ok' if n > 0 and ok == n else f'bad requests {n} ok {ok}')
except Exception as e:
    print(f'bad: no cell JSON ({type(e).__name__})')")
  if [ $crc != 0 ] || [ "$sent" != ok ]; then
    log "FATAL cell $arm sent nothing usable (sd_client rc $crc, $sent):"; tail -3 "$out"; return 1
  fi
  python3 passhist.py "$tr" "$arm" 128 || { log "FATAL passhist $arm"; return 1; }
  return $r2
}
if [[ " $SKIP " != *" serve "* ]]; then
  cells=""
  for arm in fifo60 sjf80; do
    case $arm in fifo60) f="--admit-order fifo --step-ms 60";; sjf80) f="--admit-order sjf --step-ms 80";; esac
    if timeout -k 10 420 bash -c "$(declare -f serve_cell log); A=$A M=$M PORT=$PORT O=$O T0=$T0 serve_cell $arm '$f'"; then
      cells="$cells${cells:+,}$arm"
    else
      log "serve cell $arm FAILED"; rc=1
    fi
  done
  [ "$cells" = fifo60,sjf80 ] || rc=1
fi
if [ $rc = 0 ]; then log "P1 H100 DONE rc 0 cells ${cells:-skipped} ok"; else log "P1 H100 DONE rc 1 FAILED (cells ok: ${cells:-none})"; fi
exit $rc
