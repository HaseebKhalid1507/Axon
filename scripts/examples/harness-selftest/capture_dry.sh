#!/bin/bash
# capture_dry.sh [HARNESS...]: the capture pipeline end to end with no model. First run.sh: the proxy and the
# converter on synthetic traffic, no harness binary. Then, for each harness (default: pi opencode codex crush)
# installed here, agent_capture.sh in its config-smoke mode (AXON_CAPTURE_SMOKE) against fake_upstream.py
# --text-only:
#   - the harness's real binary, with the driver's scratch HOME / XDG dirs, config and dummy key;
#   - one bugfix task in a fresh synthetic ledger repo;
#   - every request through the recording proxy;
#   - the capture converted, with agent_trace.py --check inside.
# Pass per harness: it finishes the task (rc 0), and its capture reaches the proxy and converts. A harness that
# isn't installed is SKIPPED and named; the dry run FAILS if no harness was driven at all (#186 review: it passed
# with 0). The dry run reads and writes no real harness config and
# runs no model. The fake model never fixes the bug, so "tests FAIL" on the TASK line is expected.
#   axon-run --detach --cpu --ram 3000 --max-min 30 --on bella --expect "CAPTURE DRY PASS" \
#     -- scripts/examples/harness-selftest/capture_dry.sh
set -u
H=$(cd "$(dirname "$0")" && pwd); S=$(cd "$H/../.." && pwd)   # S = scripts/
fail=0; ran=0
echo "capture_dry: scripts at $(git -C "$S" rev-parse HEAD 2>/dev/null) ($(git -C "$S" status --porcelain \
  --untracked-files=no 2>/dev/null | wc -l) tracked files modified)"
if bash "$H/run.sh"; then echo "CAPTURE DRY selftest: ok"; else echo "CAPTURE DRY selftest: FAIL"; fail=1; fi
for hn in ${*:-pi opencode codex crush}; do
  b=$(command -v "$hn") || { echo "CAPTURE DRY $hn: SKIPPED (not installed)"; continue; }
  ver=$(pacman -Qqo "$b" 2>/dev/null | head -1); ver=${ver:+$(pacman -Q "$ver" 2>/dev/null)}
  port=$((19600 + RANDOM % 300))
  python3 "$H/fake_upstream.py" $port --text-only & U=$!
  sleep 0.5
  out=$(AXON_CAPTURE_SMOKE=http://127.0.0.1:$port AXON_CAPTURE_TASK_S=${AXON_CAPTURE_TASK_S:-180} \
        bash "$S/agent_capture.sh" "$hn" 2>&1); rc=$?
  kill $U 2>/dev/null; wait $U 2>/dev/null
  echo "$out" | grep -E "^(SMOKE|TASK|CAPTURE)"
  # pass: the harness finished its task (rc 0, not a timeout), and its capture reached the proxy and converted
  if [ $rc = 0 ] && echo "$out" | grep -q "^TASK $hn bugfix r1 rc 0 " && echo "$out" | grep -q "^SMOKE $hn converted: "; then
    ran=$((ran + 1)); echo "CAPTURE DRY $hn: ok (${ver:-version unknown}, $b)"
  else
    echo "CAPTURE DRY $hn: FAIL rc $rc (${ver:-version unknown}, $b); output tail:"; echo "$out" | tail -15; fail=1
  fi
done
[ $ran -ge 1 ] || { echo "CAPTURE DRY: no harness driven (none of: ${*:-pi opencode codex crush} installed)"; fail=1; }
if [ $fail = 0 ]; then echo "CAPTURE DRY PASS ($ran harnesses driven)"; else echo "CAPTURE DRY FAIL"; fi
exit $fail
