#!/bin/bash
# fp8cap.sh MODE MODEL GL TOP STEP BIN... CAND: the RAM pre-load checks under a real memory cap (#206, #208, #209).
# Each run is `BIN gen MODEL "The capital of France is" -n 8 --device cuda --gpu-layers GL --numerics exact` (an FP8
# checkpoint's Q8_0 path: natively, FP8 runs only the whole model) in its own
# `systemd-run --user --scope -p MemoryMax=…M -p MemorySwapMax=0`. Needs Linux with systemd's user manager and a GPU
# (bella); run it through axon-podjob (it loads a model). The binaries are axon builds, the last the candidate.
# (1) Each binary's smallest cap that runs (rc 0), bisected under TOP MiB (a cap they all run under) to STEP/2 MiB.
# (2) Every binary at every STEP MiB cap from below the lowest of those (20 steps for floor, 4 for steps) to 2 steps
#     above the highest: runs / refused (conv: the loader's conversions, #208; split: the split's CPU half,
#     #184/#209) / oom (rc 137) / errN (another failure: its last line is shown).
# PASS, both modes: every run's text is the one at TOP; CAND fails other than by a refusal only below every cap it
# refused at (no check has run yet: the process's own floor, CUDA's context) or within 48 MiB above one (a check's
# tolerance, 32 MiB + 256 KiB a CPU, plus a 16 MiB step). MODE floor (#208): at some cap the first binary is
# OOM-killed and CAND refuses. MODE steps (#209): at some cap another binary refuses and CAND runs.
# The FP8 checkpoints carry their own lm_head (Qwen2.5-0.5B's config ties it; the file has one anyway), so a split's
# GPU half converts it to Q8_0 on the host: 138 MiB on the 0.5B, 552 MiB on the 7B (untied).
set -u
MODE=$1; F=$2; GL=$3; TOP=$4; STEP=$5; shift 5
case $MODE in floor|steps) ;; *) echo "MODE: floor or steps"; exit 2;; esac
[ $# -ge 2 ] || { echo "two binaries at least"; exit 2; }
BINS=(); for b in "$@"; do case $b in */*) BINS+=("$b");; *) BINS+=("./$b");; esac; done
NB=${#BINS[@]}; C=${BINS[$((NB - 1))]}
name() { basename "$1"; }
[ -f "$F/config.json" ] || [ -f "$F" ] || { echo "MISSING $F"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
free -m | head -2
try() { # bin cap -> rc; stdout in out.txt, stderr in err.txt
  systemd-run --user --scope --quiet --unit=fp8cap-$$-$RANDOM -p MemoryMax=$2M -p MemorySwapMax=0 -- \
    "$1" gen "$F" "The capital of France is" -n 8 --device cuda --gpu-layers $GL --numerics exact > out.txt 2> err.txt
  echo $?; }
kind() { # rc -> runs | conv | split | oom | errN
  if [ $1 = 0 ]; then echo runs
  elif grep -q "converting the checkpoint's tensors at load" err.txt; then echo conv
  elif grep -q "the CPU half of the split .* needs about" err.txt; then echo split
  elif [ $1 = 137 ]; then echo oom
  else echo "err$1"; fi; }
bisect() { # bin -> smallest cap (MiB) with rc 0, to STEP/2 MiB
  local lo=64 hi=$TOP mid
  [ "$(try "$1" $hi)" = 0 ] || { echo FAIL; return; }
  while [ $((hi - lo)) -gt $((STEP / 2)) ]; do
    mid=$(( (lo + hi) / 2 ))
    if [ "$(try "$1" $mid)" = 0 ]; then hi=$mid; else lo=$mid; fi
  done
  echo $hi; }
# The text at TOP, every capped run's reference.
[ "$(try "$C" $TOP)" = 0 ] || { echo "CAND fails at $TOP MiB: $(tail -3 err.txt)"; echo "FP8CAP FAIL"; exit 1; }
cp out.txt ref.txt; grep -i "split:\|gpu layers" err.txt | head -2; cat ref.txt; echo
lo=$TOP; hi=0; floors=
for b in "${BINS[@]}"; do
  h=$(bisect "$b"); floors="$floors $(name "$b") $h MiB,"
  [ "$h" = FAIL ] && { echo "$(name "$b") fails even at $TOP MiB: $(tail -3 err.txt)"; echo "FP8CAP FAIL"; exit 1; }
  [ $h -lt $lo ] && lo=$h; [ $h -gt $hi ] && hi=$h
  rc=$(try "$b" $((h - STEP / 2)))
  echo "$(name "$b"): runs from $h MiB; at $((h - STEP / 2)): $(kind $rc) $(grep -o 'needs about.*' err.txt | head -1 | cut -c1-400)"
done
echo "FP8CAP smallest cap that runs:${floors%,}"
under=4; [ $MODE = floor ] && under=20
lo=$((lo - under * STEP)); [ $lo -lt 256 ] && lo=256; hi=$((hi + 2 * STEP))
fail=0; refused=; below=0; above=0
declare -A a s
for cap in $(seq $lo $STEP $hi); do
  line="  ${cap}M:"; ks=()
  for b in "${BINS[@]}"; do
    rc=$(try "$b" $cap); k=$(kind $rc)
    [ "$k" = runs ] && ! cmp -s out.txt ref.txt && { k="runs(TEXT DIFFERS)"; fail=1; }
    case $k in err*) k="$k ($(tail -1 err.txt | cut -c1-120))";; esac
    ks+=("$k"); line="$line $(name "$b") $k |"
  done
  echo "$line"
  kc=${ks[$((NB - 1))]}
  case $kc in
    conv|split) refused=$cap;;
    runs) ;;
    *) if [ -z "$refused" ]; then below=$((below + 1))
       elif [ $cap -le $((refused + 48)) ]; then above=$((above + 1))
       else echo "  CAND $kc $((cap - refused)) MiB above its last refusal"; fail=1; fi;;
  esac
  for i in $(seq 0 $((NB - 2))); do
    case "${ks[$i]}:$kc" in oom:conv|oom:split) a[$i]=$(( ${a[$i]:-0} + 1 ));; conv:runs|split:runs) s[$i]=$(( ${s[$i]:-0} + 1 ));; esac
  done
done
for i in $(seq 0 $((NB - 2))); do
  echo "FP8CAP $(name "${BINS[$i]}") vs $(name "$C"): caps where it's OOM-killed and cand refuses: ${a[$i]:-0};" \
    "where it refuses and cand runs: ${s[$i]:-0}"
done
echo "FP8CAP cand failed below every refusal: $below; within 48 MiB above one: $above"
[ $MODE = floor ] && [ ${a[0]:-0} = 0 ] && fail=1
if [ $MODE = steps ]; then n=0; for i in $(seq 0 $((NB - 2))); do n=$((n + ${s[$i]:-0})); done; [ $n = 0 ] && fail=1; fi
[ $fail = 0 ] && echo "FP8CAP PASS" || echo "FP8CAP FAIL"
exit $fail
