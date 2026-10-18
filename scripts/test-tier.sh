#!/bin/bash
# Tiered test gate (docs/testing.md). Submits through axon-run; run it from your worktree.
#
#   scripts/test-tier.sh fast REL_BIN [DBG_BIN]   intermediate review rounds
#   scripts/test-tier.sh full REL_BIN DBG_BIN     the final round before merge (plus the md5 gate)
#
# fast: the release suite minus scripts/test-fast-skip.txt (bigger-model repeats of contracts the fast tier
#       checks on a small model; the NVRTC cross-arch builds stay in when the branch or the working tree
#       touches src/cuda/*.cu, src/cuda/mod.rs or src/cuda/ffi.rs), and, with DBG_BIN, the debug-assertion
#       suite on the CPU queue (axon-run --cpu: GPU tests skip themselves), so it takes no GPU slot.
# full: every test, release and debug-assertion builds, on GPU hosts.
#
# Env: TIER_SHARD=N (axon-run --shard N per suite, default: one job per suite), TIER_DETACH=1 (submit and
# return), TIER_NAME (job name prefix, default the worktree's name), TIER_BASE (ref for the kernel-change
# check, default origin/main), TIER_RUN_OPTS (extra axon-run options, e.g. "--big-models --on pods").
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
tier=${1:-}
rel=${2:-}
dbg=${3:-}
usage() {
    sed -n '2,6p' "$0" >&2
    exit 2
}
[ "$tier" = fast ] || [ "$tier" = full ] || usage
[ -n "$rel" ] || usage
[ "$tier" = fast ] || [ -n "$dbg" ] || usage
for b in "$rel" $dbg; do
    [ -x "$b" ] || { echo "test-tier: $b is not an executable" >&2; exit 2; }
done
name=${TIER_NAME:-$(basename "$root")}
if [ -n "${TIER_SHARD:-}" ] && [[ " ${TIER_RUN_OPTS:-} " == *" --dry-run "* ]]; then
    # axon-run --shard submits for real even with --dry-run (seen 2026-10-18).
    echo "test-tier: --dry-run doesn't work with TIER_SHARD; unset one" >&2
    exit 2
fi
run_opts=(${TIER_RUN_OPTS:-})
[ "${TIER_DETACH:-0}" = 1 ] && run_opts+=(--detach)

# Every test of the release binary.
list=$(nice -n 19 "$rel" --list 2>/dev/null | sed -n 's/: test$//p' || true)
if [ -z "$list" ]; then
    echo "test-tier: '$rel --list' listed no tests (not a libtest binary, or it failed to start)" >&2
    exit 2
fi
total=$(grep -c . <<<"$list")

skip=()
if [ "$tier" = fast ]; then
    # Kernel sources changed since the merge base with TIER_BASE, committed or not (the working tree,
    # untracked files included): then the NVRTC cross-arch builds stay in.
    kernels=0
    base=$(git -C "$root" merge-base "${TIER_BASE:-origin/main}" HEAD) || {
        echo "test-tier: no merge base with ${TIER_BASE:-origin/main}" >&2
        exit 2
    }
    if { git -C "$root" diff --name-only "$base" && git -C "$root" ls-files --others --exclude-standard; } |
        grep -qE '^src/cuda/([^/]+\.cu|mod\.rs|ffi\.rs)$'; then
        kernels=1
    fi
    while read -r t; do
        t=${t%%#*}
        t=${t//[[:space:]]/}
        [ -n "$t" ] || continue
        # libtest's --skip matches substrings: each entry must name exactly one test.
        n=$(grep -cF -- "$t" <<<"$list" || true)
        if [ "$n" != 1 ]; then
            echo "test-tier: $t matches $n tests (want exactly 1); fix scripts/test-fast-skip.txt" >&2
            exit 2
        fi
        if [ $kernels = 1 ] && [[ $t == cuda::tests::nvrtc_kernels_compile_for_* ]]; then
            continue
        fi
        skip+=(--skip "$t")
    done <"$here/test-fast-skip.txt"
fi
echo "test-tier $tier: $total tests, $((${#skip[@]} / 2)) skipped in the release suite"

submit() { # submit NAME [axon-run opts] -- BIN [args]
    local n=$1
    shift
    if [ -n "${TIER_SHARD:-}" ]; then
        axon-run --shard "$TIER_SHARD" --name "$n" "${run_opts[@]}" "$@"
    else
        axon-run --name "$n" "${run_opts[@]}" "$@"
    fi
}

# Both suites at once (each axon-run follows its job unless TIER_DETACH=1); exit = the worst rc.
pids=()
if [ "$tier" = fast ]; then
    submit "$name-fast-rel" -- "$rel" "${skip[@]}" &
    pids+=($!)
    if [ -n "$dbg" ]; then
        submit "$name-fast-dbg-cpu" --cpu -- "$dbg" &
        pids+=($!)
    fi
else
    submit "$name-full-rel" -- "$rel" &
    pids+=($!)
    submit "$name-full-dbg" -- "$dbg" &
    pids+=($!)
    echo "test-tier full: also run the md5 gate (gen outputs, main vs candidate, axon-run --same-host)." >&2
fi
rc=0
for p in "${pids[@]}"; do
    r=0
    wait "$p" || r=$?
    [ $r -gt $rc ] && rc=$r
done
exit $rc
