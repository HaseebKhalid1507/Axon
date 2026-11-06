#!/bin/bash
# fp8cap.sh MODE BASE CAND MODEL GL TOP [STEP]: the RAM pre-load checks under a real memory cap (#206, #208, #209).
# Each run is `CAND|BASE gen MODEL "The capital of France is" -n 8 --device cuda --gpu-layers GL --numerics exact`
# (an FP8 checkpoint's Q8_0 path: natively, FP8 runs only the whole model) in its own
# `systemd-run --user --scope -p MemoryMax=…M -p MemorySwapMax=0`. Linux with systemd's user manager and a GPU
# (bella); run it through axon-podjob (it loads a model). BASE and CAND are axon binaries, MODEL a checkpoint dir.
# (1) Each binary's smallest cap that runs (rc 0), bisected under TOP MiB (a cap both run under) in STEP/2 MiB
#     steps (STEP: 16 by default).
# (2) Both at every STEP MiB cap from below the lower of the two (20 steps for floor, 4 for steps) to 2 steps above
#     the higher: runs / refused (conv: the loader's conversions, #208; split: the split's CPU half, #184/#209) /
#     OOM (rc 137) / errN (another failure: its last line is shown).
# PASS, both modes: every run's text is the uncapped one; CAND fails other than by a refusal only below every cap it
# refused at (no check has run yet: the process's own floor, CUDA's context) or within 48 MiB above one (a check's
# tolerance, 32 MiB + 256 KiB a CPU, plus a 16 MiB step). MODE floor (#208): at some cap BASE is OOM-killed and CAND
# refuses. MODE steps (#209): at some cap BASE refuses and CAND runs.
# The FP8 checkpoints carry their own lm_head (Qwen2.5-0.5B's config ties it; the file has one anyway), so a split's
# GPU half converts it to Q8_0 on the host: 138 MiB on the 0.5B, 552 MiB on the 7B (untied).
set -u
MODE=$1; B=$2; C=$3; F=$4; GL=$5; TOP=$6; STEP=${7:-16}
for v in B C; do eval "b=\$$v"; case $b in */*) ;; *) eval "$v=./$b";; esac; done
case $MODE in floor|steps) ;; *) echo "MODE: floor or steps"; exit 2;; esac
[ -f "$F/config.json" ] || { echo "MISSING $F"; exit 1; }
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
bisect() { # bin -> smallest cap (MiB) with rc 0, STEP/2 MiB steps
  local lo=64 hi=$TOP mid
  [ "$(try "$1" $hi)" = 0 ] || { echo FAIL; return; }
  while [ $((hi - lo)) -gt $((STEP / 2)) ]; do
    mid=$(( (lo + hi) / 2 ))
    if [ "$(try "$1" $mid)" = 0 ]; then hi=$mid; else lo=$mid; fi
  done
  echo $hi; }
# The text under TOP, every capped run's reference.
[ "$(try "$C" $TOP)" = 0 ] || { echo "CAND fails at $TOP MiB: $(tail -3 err.txt)"; echo "FP8CAP FAIL"; exit 1; }
cp out.txt ref.txt; grep -i "split:\|gpu layers" err.txt | head -2; cat ref.txt; echo
hb=$(bisect "$B"); hc=$(bisect "$C")
echo "FP8CAP smallest cap that runs: base $hb MiB, cand $hc MiB"
case "$hb$hc" in *FAIL*) echo "a binary fails even at $TOP MiB: $(tail -3 err.txt)"; echo "FP8CAP FAIL"; exit 1;; esac
for who in base cand; do
  case $who in base) b=$B; h=$hb;; cand) b=$C; h=$hc;; esac
  rc=$(try "$b" $((h - STEP / 2))); echo "$who at $((h - STEP / 2)) MiB: $(kind $rc) $(grep -o 'needs about.*' err.txt | head -1)"
done
under=4; [ $MODE = floor ] && under=20
lo=$(( (hc < hb ? hc : hb) - under * STEP )); [ $lo -lt 256 ] && lo=256
hi=$(( (hc > hb ? hc : hb) + 2 * STEP ))
fail=0; a=0; s=0; refused=; below=0; above=0
for cap in $(seq $lo $STEP $hi); do
  line="  ${cap}M:"
  for who in base cand; do
    case $who in base) b=$B;; cand) b=$C;; esac
    rc=$(try "$b" $cap); k=$(kind $rc)
    [ "$k" = runs ] && ! cmp -s out.txt ref.txt && { k="runs(TEXT DIFFERS)"; fail=1; }
    eval "k_$who=\$k"; line="$line $who $k |"
    [ $who = cand ] && case $k in conv|split) refused=$cap;; esac
    case $k in err*) line="$line $(tail -1 err.txt | cut -c1-160) |";; esac
  done
  echo "$line"
  case $k_cand in
    runs|conv|split) ;;
    *) if [ -z "$refused" ]; then below=$((below + 1))
       elif [ $cap -le $((refused + 48)) ]; then above=$((above + 1))
       else echo "  CAND $k_cand $((cap - refused)) MiB above its last refusal"; fail=1; fi;;
  esac
  case "$k_base:$k_cand" in oom:conv|oom:split) a=$((a + 1));; split:runs|conv:runs) s=$((s + 1));; esac
done
echo "FP8CAP caps where base is OOM-killed and cand refuses: $a; where base refuses and cand runs: $s;" \
  "cand failed below every refusal: $below, within 48 MiB above one: $above"
[ $MODE = floor ] && [ $a = 0 ] && fail=1
[ $MODE = steps ] && [ $s = 0 ] && fail=1
[ $fail = 0 ] && echo "FP8CAP PASS" || echo "FP8CAP FAIL"
exit $fail
