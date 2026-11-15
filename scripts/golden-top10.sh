#!/bin/bash
# scripts/golden-top10.sh GOLDEN_PY LIST_JSON AXON LLAMA_TGZ LLAMA_CUDA [ONLY,ONLY]
#
# golden.py on the compatibility audit's top 10 (scripts/golden-top10.json: Hugging Face repo, pinned revision, file,
# sha256), one model at a time, on a pod or node through axon-podjob. Each model: ./models' copy if its sha256 matches,
# else a download of the pinned revision (sha256-checked, deleted after its run). llama.cpp's CPU build (LLAMA_TGZ, the
# prebuilt b11580 = 4413a19) is the reference, its CUDA build (LLAMA_CUDA: a llama-server or its dir) the spread
# (--llama-server-ref). AXON needs AXON_LOGITS_CHAT (#229) for the chat check. ONLY (comma-separated ollama names)
# picks the models AND their order (a few models: `... mistral,gpt-oss`).
#
# Exit: the worst per-model status, as golden.py's own: 1 if any model FAILs (a real difference), else 2 if any
# model was not judged, else 0. A model is FAIL only when its golden.py JSON is complete and says exit 1; a failed
# download, a sha256 mismatch, a model past PER_MODEL_S, a golden.py crash or a cut-off JSON count as not judged (2),
# never as a pass (shady, #229 round 2: job 7080 printed "done" over 2 FAILs and a failed download, EXPECT ok, green).
# A CPU reference that won't start stops the run (2). A CUDA spread backend that won't start or load leaves golden
# without --llama-server-ref: its mean KL / max dlogprob / swing are then not judged (ERROR, 2).
# Last lines: "GOLDEN-TOP10 SUMMARY pass P, FAIL F, not judged E of N ref=yes|no", then
# "GOLDEN-TOP10 done rc=R". Gate with axon-podjob --expect '^GOLDEN-TOP10 done rc=0$' (and the job's exit code).
#
# Guards (after job 1113-033105-580d, 11-13: llama-server -t 128 under a 13.6-CPU cgroup v1 quota crawled; one
# request ran past golden.py's 900 s, 16 min per model, every model llama-error):
#  - threads: AXON_THREADS (axon-node's share), else the cgroup v2 cpu.max or v1 cfs quota, else nproc; at most 16;
#  - a startup check of both llama-servers on the first model, stdout+stderr to a log: /health within 180 s, then 16
#    tokens within 120 s (prints load s and tok/s);
#  - a per-model wall clock (PER_MODEL_S, default 900 s; timeout -k 30), and on any verdict but pass, the tails of that
#    model's llama logs in the job output.
# Env: PER_MODEL_S, AXON_THREADS, AXON_MODELS (the models dir, default ./models), GOLDEN_NO_PREFLIGHT=1 (the
# self-test: no llama-server startup check).
set -uo pipefail
[ $# -ge 5 ] || { sed -n '2,3p' "$0" >&2; exit 2; }
G=$(readlink -f "$1"); L=$(readlink -f "$2"); AX=$(readlink -f "$3"); TGZ=$(readlink -f "$4"); LC=$(readlink -f "$5")
ONLY=${6:-}
[ -d "$LC" ] && LC=$LC/llama-server
MD=$(readlink -f "${AXON_MODELS:-models}" 2>/dev/null || true); [ -d "$MD" ] || MD=/root/axon-node/models
W=$PWD/gw; mkdir -p "$W/models" "$W/llama"
tar --no-same-owner --no-same-permissions -xzf "$TGZ" -C "$W/llama" || { echo "GOLDEN-TOP10 done rc=2 (no llama.cpp: $TGZ)"; exit 2; }
LS=$(find "$W/llama" -name llama-server -type f | head -1)
# Threads for llama.cpp's CPU reference: axon-node's share for this job (AXON_THREADS), else nproc capped by the cgroup's
# CPU quota; at most 16 either way. Job 1113-033105-580d (11-13): on a pod, cpu.max said "max" and nproc 128 while the pod
# had ~14 vCPU, so llama-server ran -t 128 and crawled (a request over golden.py's 900 s: every model llama-error).
TS=AXON_THREADS; T=${AXON_THREADS:-}
if [ -z "$T" ]; then TS=nproc; T=$(nproc); q=
  if [ -r /sys/fs/cgroup/cpu.max ]; then read -r q p < /sys/fs/cgroup/cpu.max; [ "$q" = max ] && q=; TS2=cpu.max
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then   # cgroup v1 (580d's pod: 1360000/100000 = 13.6 CPUs)
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); [ "$q" -gt 0 ] || q=; TS2=cfs_quota
  fi
  [ -n "$q" ] && TS=$TS2 && T=$(( (q + p - 1) / p < T ? (q + p - 1) / p : T ))
fi
[ "$T" -gt 16 ] && T=16 && TS="$TS, capped"
# The CUDA llama-server links cudart/cuBLAS 12: from the image if it has them, else NVIDIA's wheels.
NV=$(ls -d /root/axon-node/lib/nvrtc-* 2>/dev/null | tail -1)
export LD_LIBRARY_PATH=${NV:+$NV:}${LD_LIBRARY_PATH:-}
REF=(--llama-server-ref "$LC" --llama-ref-ngl 99)
if ldd "$LC" 2>/dev/null | grep -v libcuda.so | grep -q "not found"; then
  echo "golden-top10: $(ldd "$LC" | grep -v libcuda.so | grep "not found" | awk '{print $1}' | tr '\n' ' ')missing: pip wheels"
  python3 -m pip install -q --target "$W/cu" nvidia-cuda-runtime-cu12==12.8.* nvidia-cublas-cu12==12.8.* 2>&1 | tail -2
  for d in "$W"/cu/nvidia/*/lib; do LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH; done; export LD_LIBRARY_PATH
  if ldd "$LC" | grep -v libcuda.so | grep -q "not found"; then
    echo "golden-top10: CUDA llama-server still can't load: NO --llama-server-ref (golden's scaled clauses not judged)"; REF=()
  fi
fi
echo "golden-top10: host $(hostname) threads $T ($TS) axon $(sha256sum "$AX" | cut -c1-12) llama $("$LS" --version 2>&1 | grep -m1 version) ref ${REF[1]:-none}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
# The list: golden-top10.json's entries (every entry), or an audit list's entries marked "golden": true.
python3 - "$L" > "$W/list.tsv" <<'PY' || { echo "GOLDEN-TOP10 done rc=2 (unreadable list $L)"; exit 2; }
import json, sys
for e in json.load(open(sys.argv[1])):
    if e.get("golden", True):
        print("\t".join([e["ollama"], e["repo"], e["file"], e["sha256"], e.get("revision") or "main"]))
PY
if [ -n "$ONLY" ]; then   # ONLY's order
  for n in ${ONLY//,/ }; do awk -F'\t' -v n="$n" '$1 == n' "$W/list.tsv"; done > "$W/run.tsv"
  [ "$(wc -l < "$W/run.tsv")" = "$(tr ',' '\n' <<< "$ONLY" | grep -c .)" ] || { echo "golden-top10: ONLY names not all in the list: $ONLY"; echo "GOLDEN-TOP10 done rc=2"; exit 2; }
else cp "$W/list.tsv" "$W/run.tsv"; fi
N=$(wc -l < "$W/run.tsv")
echo "golden-top10: models $(cut -f1 "$W/run.tsv" | tr '\n' ' ')(per-model limit ${PER_MODEL_S:-900} s)"
tails() { # tails FILE: the model's llama logs, last lines, into the job output
  for x in "$W/$1.llama.log" "$W/$1.llama-ref.log" "$W/$1.preflight-cpu.log" "$W/$1.preflight-cuda.log"; do
    [ -f "$x" ] && { echo "  -- tail $(basename "$x"):"; tail -n 12 "$x" | cut -c1-200 | sed 's/^/  | /'; }; done; }
preflight() { # preflight BIN NGL MODEL TAG: start it, /health within 180 s, 16 tokens within 120 s; rc 0 if fine
  [ "${GOLDEN_NO_PREFLIGHT:-}" = 1 ] && return 0
  python3 - "$1" "$2" "$3" "$T" "$W/$(basename "$3").preflight-$4.log" "$4" <<'PY'
import json, socket, subprocess, sys, time, urllib.request
binary, ngl, model, threads, log, tag = sys.argv[1:]
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
t0 = time.time()
with open(log, "w") as lf:
    p = subprocess.Popen([binary, "-m", model, "--port", str(port), "-c", "1024", "-np", "1", "--no-warmup",
                          "-ngl", ngl, "-t", threads], stdout=lf, stderr=subprocess.STDOUT)
try:
    while True:
        if p.poll() is not None:
            print(f"GOLDEN-PREFLIGHT {tag} FAIL: llama-server exited {p.returncode} after {time.time() - t0:.0f} s"); sys.exit(1)
        if time.time() - t0 > 180:
            print(f"GOLDEN-PREFLIGHT {tag} FAIL: not healthy in 180 s"); sys.exit(1)
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(0.5)
    load = time.time() - t0; t1 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(
        {"prompt": "The capital of France is", "n_predict": 16, "temperature": 0, "cache_prompt": False}).encode(),
        {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=120))
    except Exception as e:
        print(f"GOLDEN-PREFLIGHT {tag} FAIL: 16 tokens not back in 120 s ({e})"); sys.exit(1)
    tm = r.get("timings", {})
    print(f"GOLDEN-PREFLIGHT {tag} ok: -ngl {ngl} -t {threads}: healthy in {load:.0f} s, 16 tokens in "
          f"{time.time() - t1:.1f} s (prompt {tm.get('prompt_per_second', 0):.1f} tok/s, gen {tm.get('predicted_per_second', 0):.1f} tok/s)")
finally:
    p.terminate()
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()
PY
}
# status JSON RC: golden.py's per-model status. 0 or 1 only when its JSON is complete and agrees with its exit code;
# anything else (no JSON, cut off, a crash's rc) is 2, not judged.
status() {
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print(2); sys.exit()
rc = int(sys.argv[2])
print(rc if rc in (0, 1, 2) and d.get("complete") is True and d.get("exit") == rc else 2)
PY
}
first=1; NP=0; NF=0; NE=0; WORST=0
note() { # note NAME STATUS: count it, keep the worst (FAIL 1 beats not judged 2)
  case $2 in 0) NP=$((NP + 1));; 1) NF=$((NF + 1)); WORST=1;; *) NE=$((NE + 1)); [ $WORST = 0 ] && WORST=2;; esac; }
finish() { # finish [WHY]: the summary, the done line, the exit code
  echo "GOLDEN-TOP10 SUMMARY pass $NP, FAIL $NF, not judged $NE of $N ref=$([ ${#REF[@]} -gt 0 ] && echo yes || echo no)${1:+ ($1)}"
  rm -rf "$W/models" "$W/cu"
  echo "GOLDEN-TOP10 done rc=$WORST"
  exit $WORST; }
i=0
while IFS=$'\t' read -r name repo f sha rev <&3; do
  i=$((i + 1)); got=; dl=0
  if [ -f "$MD/$f" ]; then got=$( [ -f "$MD/$f.sha256" ] && cut -d' ' -f1 "$MD/$f.sha256" || sha256sum "$MD/$f" | cut -d' ' -f1); fi
  if [ "$got" = "$sha" ]; then ln -sf "$MD/$f" "$W/models/$f"
  else
    dl=1; t0=$(date +%s)
    [ "$rev" = main ] && echo "golden-top10: WARNING $name has no pinned revision: resolve/main (sha256 still checked)"
    if ! curl -sSL --retry 5 -o "$W/models/$f" "https://huggingface.co/$repo/resolve/$rev/$f"; then
      echo "GOLDEN-MODEL $name $f rc=2 NOT-JUDGED fetch-failed"; rm -f "$W/models/$f"; note "$name" 2; continue
    fi
    got=$(sha256sum "$W/models/$f" | cut -d' ' -f1)
    if [ "$got" != "$sha" ]; then
      echo "GOLDEN-MODEL $name $f rc=2 NOT-JUDGED sha256-MISMATCH $got"; rm -f "$W/models/$f"; note "$name" 2; continue
    fi
    echo "golden-top10: $f downloaded in $(( $(date +%s) - t0 )) s"
  fi
  if [ $first = 1 ]; then first=0
    if ! preflight "$LS" 0 "$W/models/$f" cpu; then
      tails "$f"; echo "golden-top10: llama.cpp's CPU reference won't run: stopping"
      NE=$((NE + N - i + 1)); [ $WORST = 0 ] && WORST=2; finish "the CPU reference didn't start"
    fi
    if [ ${#REF[@]} -gt 0 ]; then
      preflight "$LC" 99 "$W/models/$f" cuda || { tails "$f"; echo "golden-top10: CUDA spread backend won't run: golden without --llama-server-ref (scaled clauses not judged)"; REF=(); }
    fi
  fi
  t0=$(date +%s)
  timeout -k 30 "${PER_MODEL_S:-900}" python3 "$G" run --axon "$AX" --llama-server "$LS" "${REF[@]}" --models-dir "$W/models" --models "$f" \
    --device cuda --n 64 --threads "$T" --out "$W/golden-$name.json" > "$W/golden-$name.out" 2>&1 < /dev/null
  rc=$?
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "GOLDEN-MODEL $name $f TIMEOUT: past ${PER_MODEL_S:-900} s (rc $rc)"
  st=$(status "$W/golden-$name.json" $rc)
  grep -E "^GOLDEN" "$W/golden-$name.out" | sed "s/^/  /"
  [ $st = 2 ] && [ $rc != 2 ] && { echo "  -- golden.py rc $rc without a complete JSON (crash or cut off): tail"; tail -n 8 "$W/golden-$name.out" | cut -c1-200 | sed 's/^/  | /'; }
  v=$(python3 -c "import json; r=json.load(open('$W/golden-$name.json'))['models'][0]; c=r['checks'].get('chat',{}); print(r['verdict'], 'chat='+str(c.get('status')), 'at='+str(c.get('at','')), 'axon_len='+str(c.get('axon_len','')), 'llama_len='+str(c.get('llama_len','')))" 2>/dev/null | tail -1)
  echo "GOLDEN-MODEL $name $f rc=$st ${v:-no-result} ($(( $(date +%s) - t0 )) s)"
  python3 "$G" table "$W/golden-$name.json" 2>/dev/null | tail -1
  [ $st = 0 ] || tails "$f"
  echo "GOLDEN-JSON $name $(python3 -c "import json; print(json.dumps(json.load(open('$W/golden-$name.json'))))" 2>/dev/null)"
  note "$name" "$st"
  [ $dl = 1 ] && rm -f "$W/models/$f"
done 3< "$W/run.tsv"
finish
