#!/bin/bash
# cbcheck.sh BASE CAND PY SERVE_MD5_PY HARNESS_PY BODIES OUT REPLAYS: #274's (PR B of #259, constrained tool calls by
# default) bit and validity gates, one GPU job (24 GB: Qwen2.5-7B and Qwen3-8B at ctx 32768). BASE: main at the PR's
# merge base; CAND: the PR's head; PY: scripts/constrain_check.py; SERVE_MD5_PY: scripts/constrain/serve_md5_share.py;
# HARNESS_PY: the harness PR's scripts/harness_accept.py (its 14 tasks' first prompts, for the stress); BODIES:
# scripts/constrain/bodies; REPLAYS: scripts/constrain/replays; models in $AXON_MODELS (default ./models).
# Run on bella as: axon-podjob --big-models --vram 10500 --ram 12000 --expect '^CBCHECK PASS' -- cbcheck.sh ...
# scripts/constrain_check.py runs (each a fresh server with the
# acceptance test's flags: cuda, ctx 32768, q8 KV, --parallel 4, Qwen2.5's card sampler):
#  Qwen2.5-7B-Instruct Q4_K_M and Qwen3-8B Q4_K_M, each:
#   (1) cand default: the 8 cells alone and batched (equal), pins recorded (OUT/pins.json); Qwen2.5-7B: also the
#       200-reply stress (codex's tools, the 14 tasks' first prompts, seeds 1..200);
#   (2) cand default again, a new server: the cells match (1)'s pins;
#   (3) cand --no-constrain and (4) BASE (main 9f3630c): their cells' replies equal (the kill switch runs main's
#       paths: tool traffic identical), (5) cand AXON_CONSTRAIN=0 too.
#  serve md5 of tool-free traffic (serve_md5_share-v3.py: 7 sessions x 3 turns, greedy and seeded): BASE vs cand
#  default on Qwen2.5-7B cuda q8 and qwen2.5-0.5b cpu.
#  replay (shady's SHOULD 5 on #274): constrain_check.py replay of REPLAYS (the 13 acceptance calls that tripped the
#  first stall windows, as the requests that led to them, each cell's `orig` naming the call; plus data that repeats
#  one value)
#  on the cand default, greedy and seeds 1, 2: a runaway's reply ends before its token limit with every call valid; a
#  legit cell doesn't stall; the known false positive is reported. Qwen2.5-7B Q4_K_M, and Qwen2.5-0.5B-Instruct
#  Q4_0 for the one 0.5B runaway.
# Integrity (11-15, after a8e1 passed "kill switch == main" on 404s: the captured bodies' model "qwen2.5-7b" isn't a
# name the server answers): constrain_check.py now aims every body at the server's model id, and every run, every arm
# (cand, --no-constrain, AXON_CONSTRAIN=0, main, replay), must print INTEGRITY PASS: every request 200, and the
# constraint ran (cand, replay: --expect-constraint on) or didn't (the kill switch, main: off). A diff of runs with a
# non-200 cell is CELLS-INVALID, never CELLS-EQUAL.
# Prints CBCHECK PASS when every verdict passed.
set -u
B=$1; C=$2; PY=$3; SP=$4; HP=$5; BD=$6; O=$7; RP=$8
MD=${AXON_MODELS:-models}; [ -d "$MD" ] || { echo "cbcheck: no models dir $MD (set AXON_MODELS)"; exit 2; }
case $B in */*) ;; *) B=./$B;; esac; case $C in */*) ;; *) C=./$C;; esac
mkdir -p "$O"; export XDG_CACHE_HOME="$O/cache"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | tee "$O/gpu.txt"
fails=()
v() { echo "== $1: $2"; [ "$2" = PASS ] || fails+=("$1"); }
integ() { local f; for f in "$@"; do grep -q '^INTEGRITY PASS' "$f" || { echo FAIL; return; }; done; echo PASS; }
for m in Qwen2.5-7B-Instruct-Q4_K_M Qwen3-8B-Q4_K_M; do
  M=$MD/$m.gguf; st=(); [ $m = Qwen2.5-7B-Instruct-Q4_K_M ] && st=(--stress 200 --tasks "$HP")
  python3 "$PY" run --axon "$C" --model "$M" --bodies "$BD" --out "$O/$m-cand1.json" --pins "$O/pins.json" --record \
    --expect-constraint on "${st[@]}" | tee "$O/$m-cand1.txt"
  v "$m cand default (cells, batched, stress)" "$(grep -q '^VERDICT PASS' "$O/$m-cand1.txt" && echo PASS || echo FAIL)"
  python3 "$PY" run --axon "$C" --model "$M" --bodies "$BD" --out "$O/$m-cand2.json" --pins "$O/pins.json" \
    --expect-constraint on | tee "$O/$m-cand2.txt"
  v "$m cand default again (pins)" "$(grep -q '^batched == alone: True; pins: match' "$O/$m-cand2.txt" \
    && grep -q '^VERDICT PASS' "$O/$m-cand2.txt" && echo PASS || echo FAIL)"
  python3 "$PY" run --axon "$C" --model "$M" --bodies "$BD" --out "$O/$m-off.json" --expect-constraint off \
    -- --no-constrain | tee "$O/$m-off.txt"
  AXON_CONSTRAIN=0 python3 "$PY" run --axon "$C" --model "$M" --bodies "$BD" --out "$O/$m-env0.json" \
    --expect-constraint off | tee "$O/$m-env0.txt"
  python3 "$PY" run --axon "$B" --model "$M" --bodies "$BD" --out "$O/$m-base.json" --expect-constraint off \
    | tee "$O/$m-base.txt"
  v "$m integrity, every arm (all 200; constraint on in cand, off in --no-constrain, AXON_CONSTRAIN=0, main)" \
    "$(integ "$O/$m-cand1.txt" "$O/$m-cand2.txt" "$O/$m-off.txt" "$O/$m-env0.txt" "$O/$m-base.txt")"
  python3 "$PY" diff "$O/$m-base.json" "$O/$m-off.json" | tee "$O/$m-diff-off.txt"
  v "$m --no-constrain == main" "$(grep -q '^CELLS-EQUAL' "$O/$m-diff-off.txt" && echo PASS || echo FAIL)"
  python3 "$PY" diff "$O/$m-base.json" "$O/$m-env0.json" | tee "$O/$m-diff-env0.txt"
  v "$m AXON_CONSTRAIN=0 == main" "$(grep -q '^CELLS-EQUAL' "$O/$m-diff-env0.txt" && echo PASS || echo FAIL)"
  grep -h '^fingerprint' "$O/$m-cand1.txt" "$O/$m-off.txt" "$O/$m-base.txt"
done
for set in "cuda q8 Qwen2.5-7B-Instruct-Q4_K_M" "cpu f16 qwen2.5-0.5b-q4_k_m"; do
  set -- $set
  python3 "$SP" --model "$MD/$3.gguf" --device $1 --kv $2 --out "$O/serve-$1-$2-$3" "base=$B" "cand=$C" \
    > "$O/serve-$1-$3.txt" 2>&1; r=$?
  tail -3 "$O/serve-$1-$3.txt"
  v "serve md5 (no tools) $1 $3" "$([ $r = 0 ] && echo PASS || echo FAIL)"
done
# 11-16 (7b6f at 78790e6): 11-pi-edit seeds 1 and 2 are the stall rule's known miss (a loop that starts over; the
# PR's serving.md): reported, not judged. Any other runaway that runs on still fails.
KM=11-pi-edit:seed1,11-pi-edit:seed2
for m in Qwen2.5-7B-Instruct-Q4_K_M Qwen2.5-0.5B-Instruct-Q4_0; do
  km=(); [ $m = Qwen2.5-7B-Instruct-Q4_K_M ] && km=(--known-miss $KM)
  python3 "$PY" replay --axon "$C" --model "$MD/$m.gguf" --replays "$RP" --out "$O/replay-$m.json" --seeds 2 \
    "${km[@]}" | tee "$O/replay-$m.txt"
  v "replay $m (runaways end, legit data doesn't stall; all 200, constraint on)" \
    "$(grep -q '^VERDICT PASS' "$O/replay-$m.txt" && integ "$O/replay-$m.txt")"
  # 11-17 (3c205cc, the lead's ruling): a call the stall rule stops is not delivered as a call. Every row that
  # stalled shows its call as `stalled:` (cut off, arguments not JSON), none as a usable `ok:` call after a stall;
  # and (the lead's ruling of 11-17 on Fable's review) every such reply finishes `length` (Responses: `incomplete`),
  # as a call max_tokens cut, which pi refuses (pi runs a repaired call on any other finish).
  ns=$(grep -c '^REPLAY .* stalls [1-9]' "$O/replay-$m.txt")
  nd=$(grep '^REPLAY .* stalls [1-9]' "$O/replay-$m.txt" | grep -c 'stalled:')
  nf=$(grep '^REPLAY .* stalls [1-9]' "$O/replay-$m.txt" | grep -c -E ' finish (length|incomplete) ')
  echo "STALLED-DELIVERY $m: $nd of $ns rows that stalled show the call as stalled:, $nf of $ns finish length/incomplete"
  ok=$([ $nd = $ns ] && [ $nf = $ns ] && echo PASS || echo FAIL)
  if [ $m = Qwen2.5-7B-Instruct-Q4_K_M ]; then
    v "replay $m: every stalled call delivered cut off, the reply finishing length ($nd, $nf of $ns), at least one" \
      "$([ $ns -gt 0 ] && echo $ok || echo FAIL)"
  else
    v "replay $m: every stalled call delivered cut off, the reply finishing length ($nd, $nf of $ns)" "$ok"
  fi
done
if [ ${#fails[@]} = 0 ]; then echo "CBCHECK PASS"; else echo "CBCHECK FAIL: ${fails[*]}"; exit 1; fi
