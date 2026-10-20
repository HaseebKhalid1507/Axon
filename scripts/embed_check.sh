#!/bin/bash
# embed_check.sh AXON EMBED MODEL [cpu|cuda|both]: examples/embed (the library API) must print exactly what
# `axon gen` prints under the same environment: greedy, 32 tokens, 2 prompts, for each environment below.
# Also: a bad AXON_KV is an error from both (embed: rc 1, a message naming AXON_KV, no panic), and AXON_KV=f32
# changes the output on the CPU (else the setting was dropped somewhere). Run through axon-run (it loads a model).
# Each SAME line prints its md5, so which environments change the output is visible. Exit 1 on any mismatch. Written after the md5 gate caught embed ignoring AXON_KV=f32 (PR #117, round 3).
set -u
AX=$1; EM=$2; M=$3; DEVS=${4:-both}
# P2 discriminates f16 from f32 on the 0.5B (round 3 md5 gate: different md5s); "def fibonacci(n):" did not.
P1="The capital of France is"; P2="Write a short story about a robot who learns to paint. Once upon a time,"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fails=0; n=0
pair() { # pair LABEL ENV...: gen vs embed stdout under ENV, both prompts
  local label=$1; shift
  for p in "$P1" "$P2"; do
    n=$((n + 1))
    env "$@" "$AX" gen "$M" "$p" -n 32 > "$T/g" 2> "$T/ge"; local rg=$?
    env "$@" "$EM" "$M" "$p" 32 > "$T/e" 2> "$T/ee"; local re=$?
    if [ $rg = 0 ] && [ $re = 0 ] && cmp -s "$T/g" "$T/e"; then
      echo "SAME   $label  [$p]  $(md5sum < "$T/g" | cut -c1-12)"
    else
      echo "DIFF   $label  [$p]  gen rc=$rg $(md5sum < "$T/g" | cut -c1-12)  embed rc=$re $(md5sum < "$T/e" | cut -c1-12)"
      tail -2 "$T/ee"; fails=$((fails + 1))
    fi
  done
}
cpu=0; cuda=0
[ "$DEVS" = cpu ] || [ "$DEVS" = both ] && cpu=1
[ "$DEVS" = cuda ] || [ "$DEVS" = both ] && cuda=1
if [ $cpu = 1 ]; then
  pair "cpu default" AXON_DEVICE=cpu
  pair "cpu AXON_KV=f16" AXON_DEVICE=cpu AXON_KV=f16
  pair "cpu AXON_KV=f32" AXON_DEVICE=cpu AXON_KV=f32
  env AXON_DEVICE=cpu "$AX" gen "$M" "$P1" -n 32 > "$T/f16" 2> /dev/null
  env AXON_DEVICE=cpu AXON_KV=f32 "$EM" "$M" "$P1" 32 > "$T/f32" 2> /dev/null
  if cmp -s "$T/f16" "$T/f32"; then echo "DIFF   cpu AXON_KV=f32 gives the f16 output (dropped)"; fails=$((fails + 1))
  else echo "OK     cpu AXON_KV=f32 output differs from f16"; fi
fi
if [ $cuda = 1 ]; then
  pair "cuda default" AXON_DEVICE=cuda
  pair "cuda AXON_KV=q8" AXON_DEVICE=cuda AXON_KV=q8
  pair "cuda AXON_GPU_LAYERS=0 (= cpu)" AXON_DEVICE=cuda AXON_GPU_LAYERS=0
  pair "cuda AXON_GPU_LAYERS=12 (split)" AXON_DEVICE=cuda AXON_GPU_LAYERS=12
  pair "cuda AXON_KV=f32 split (CPU half f32)" AXON_DEVICE=cuda AXON_KV=f32 AXON_GPU_LAYERS=12
fi
env AXON_DEVICE=cpu AXON_KV=bogus "$EM" "$M" "$P1" 8 > /dev/null 2> "$T/bad"; rb=$?
env AXON_DEVICE=cpu AXON_KV=bogus "$AX" gen "$M" "$P1" -n 8 > /dev/null 2> "$T/badg"; rg=$?
if [ $rb = 1 ] && grep -q AXON_KV "$T/bad" && ! grep -q panicked "$T/bad" && [ $rg != 0 ]; then
  echo "OK     bad AXON_KV: embed rc 1 ($(head -c 120 "$T/bad" | tr '\n' ' ')), gen rc $rg"
else
  echo "FAIL   bad AXON_KV: embed rc $rb, gen rc $rg: $(head -c 300 "$T/bad")"; fails=$((fails + 1))
fi
echo "embed_check: $n pairs, $fails failures"
[ $fails = 0 ]
