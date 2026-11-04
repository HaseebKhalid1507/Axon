#!/bin/bash
# agent_replay_smoke.sh AXON_BIN [ENGINES]: the harness end to end in ~1-2 min, extract -> replay -> report, the gate
# this PR ran at 060c555 and after (11-04). ENGINES: axon (default) or axon,ollama (covers the board's own ollama
# store, ollama_check and ollama's raw path). 1.5B Q4_K_M, one short cell (speed 4, a 120-s trace window = 30 s of
# replay, scale 0.05), GPU. NOT a board, NOT timing evidence. Then the --sessions ladder on Axon alone: the same
# cell with S/2 sessions and with 2S (every session plus one fresh copy). Also the harness's own checks: CI's
# --check-committed and agent_replay_selftest.py, and --synthetic must reproduce the committed example byte for
# byte. The trace is extracted fresh from the Synaps logs (AXON_SMOKE_FROM..AXON_SMOKE_TO, 3 complete days) into
# the out dir, never into this repo.
#   axon-run --detach --vram 10000 --ram 8000 --max-min 20 --name agent-a-<sha7>-smoke --expect "SMOKE END ok" \
#     -- scripts/agent_replay_smoke.sh ~/.cache/axon-agent/bin/axon-<main sha7> axon,ollama
set -u
D=${AXON_AGENT_DIR:-$HOME/.cache/axon-agent}
W=$(cd "$(dirname "$0")/.." && pwd)
M=${AXON_MODELS:-$W/models}
AX=${1:?usage: agent_replay_smoke.sh AXON_BIN [ENGINES]}
ENG=${2:-axon}
OUT=$D/smoke-merge-$(date +%m%d-%H%M%S); mkdir -p "$OUT"
fail() { echo "SMOKE FAIL: $*"; exit 1; }
echo "harness $(git -C "$W" rev-parse HEAD) ($(git -C "$W" status --porcelain --untracked-files=no | wc -l) tracked files" \
  "modified)  axon $AX sha256 $(sha256sum "$AX" | cut -c1-16)"

# 0. the harness's own checks
(cd "$W" && python3 scripts/agent_trace.py --check-committed) || fail "--check-committed"
python3 "$W/scripts/agent_replay_selftest.py" || fail "agent_replay_selftest.py"
python3 "$W/scripts/agent_trace.py" --synthetic "$OUT/synthetic.json" > /dev/null || fail "--synthetic"
cmp "$OUT/synthetic.json" "$W/scripts/examples/agent-trace-synthetic.json" || fail "synthetic example not reproduced"
echo "CHECK --synthetic reproduces scripts/examples/agent-trace-synthetic.json byte for byte"

# 1. extract: a fresh numbers-only trace from the Synaps logs (3 complete days), times anonymized, outside the repo
python3 "$W/scripts/agent_trace.py" --from "${AXON_SMOKE_FROM:-2026-11-01}" --to "${AXON_SMOKE_TO:-2026-11-03}" \
  --anonymize-time --summary \
  --out "$OUT/trace-anon.json" || fail "extract"
python3 "$W/scripts/agent_trace.py" --check "$OUT/trace-anon.json" || fail "privacy check"

# 2. replay: $ENG, one round, one short cell (the board's code path: start engine, tokcheck, prime, replay)
python3 -u "$W/scripts/agent_replay.py" board --engines "$ENG" --kv f16 --rounds 1 --speeds 4 \
  --model "$M/qwen2.5-1.5b-instruct-q4_k_m.gguf" --axon "$AX" --ollama-model axon-q15 --kv-tokens 65536 \
  --trace "$OUT/trace-anon.json" --window-s 120 --scale 0.05 --out "$OUT" | tee "$OUT/board.log"
grep -q '^BOARD END ok' "$OUT/board.log" || fail "board did not end ok"
python3 - "$OUT" "$ENG" <<'EOF' || fail "per-run assertions"
import glob, json, sys
d, engines = sys.argv[1], sys.argv[2].split(',')
for e in engines:
    runs = sorted(glob.glob(f'{d}/run-{e}-*.json'))
    assert len(runs) == 1, (e, runs)
    r = json.load(open(runs[0]))
    # axon: no error at all. ollama: only its own repeat-limit aborts are allowed (counted, reported).
    assert r['errors'] == (r.get('aborted') or 0 if e == 'ollama' else 0), (e, r['errors'], r.get('first_error'))
    assert not r.get('prompt_mismatch') and not r.get('out_tokens_short'), (e, r.get('prompt_mismatch'))
    assert r['requests'] > 0 and r['prompt_checked'] == r['requests'] - (r.get('aborted') or 0), (e, r['prompt_checked'])
    print(f"CHECK {e}: {r['requests']} requests, {r['prompt_checked']} prompt counts checked, 0 mismatches, "
          f"{r['errors']} errors ({r.get('aborted') or 0} engine aborts), md5 {r['md5']}; {r['config']}")
m = json.load(open(f'{d}/meta.json'))
assert len(m['model_sha256']) == 64 and m['binaries']['axon']['sha256'], m
h = m['harness']
assert m['state'] == 'ended' and len(h['commit']) == 40 and isinstance(h['dirty'], list) and h['sha256'], m
print(f"CHECK meta.json: model sha256 {m['model_sha256']}, axon binary sha256 {m['binaries']['axon']['sha256'][:16]}, "
      f"harness {h['commit']} dirty {h['dirty']} agent_replay.py sha256 {h['sha256'][:16]}")
if 'ollama' in engines:
    o = m['ollama_model']
    assert m['engine_versions']['ollama'] and o['weights'] == 'sha256:' + m['model_sha256'], m
    assert o['layers'][0]['type'] == 'model' and o['blobs'] == len(o['layers']) + 1 and o['params'] is not None, o
    print(f"CHECK ollama {m['engine_versions']['ollama']} on {m['ollama_models']}: {o['name']} -> {o['manifest']}, "
          f"weights {o['weights'][:19]} = sha256(--model), layers {[x['type'] for x in o['layers']]}, "
          f"all {o['blobs']} blobs hashed, params {o['params']}")
EOF

# 2b. the --sessions ladder (Axon): S/2 sessions (a subset) and 2S (every session plus one fresh copy)
S=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cells']['4.0']['n_sessions'])" "$OUT/meta.json") \
  || fail "cell sessions"
N1=$(( S / 2 > 0 ? S / 2 : 1 )); N2=$(( 2 * S ))
python3 -u "$W/scripts/agent_replay.py" board --engines axon --kv f16 --rounds 1 --speeds 4 --sessions $N1,$N2 \
  --model "$M/qwen2.5-1.5b-instruct-q4_k_m.gguf" --axon "$AX" --kv-tokens 65536 \
  --trace "$OUT/trace-anon.json" --window-s 120 --scale 0.05 --out "$OUT/ladder" | tee "$OUT/ladder.log"
grep -q '^BOARD END ok' "$OUT/ladder.log" || fail "ladder board did not end ok"
grep -q '^- speed 4, axon f16: N_warm = ' "$OUT/ladder.log" || fail "no N_warm line"
python3 - "$OUT" $S $N1 $N2 <<'EOF' || fail "ladder assertions"
import json, sys
d, S, N1, N2 = sys.argv[1], *map(int, sys.argv[2:])
base = json.load(open(f'{d}/run-axon-f16-s4-r0.json'))
m = json.load(open(f'{d}/ladder/meta.json'))
for n in (N1, N2):
    r = json.load(open(f'{d}/ladder/run-axon-f16-s4-n{n}-r0.json'))
    c = m['cells'][f'4.0/n{n}']
    assert r['sessions'] == n and c['n_sessions'] == n and c['sessions'] == n, (n, c)
    assert r['errors'] == 0 and not r['prompt_mismatch'] and r['prompt_checked'] == r['requests'], (n, r['first_error'])
    assert len({x['chain'] for x in r['per_request']}) == n, n
    print(f"CHECK ladder n={n}: {r['requests']} requests in {n} sessions ({c['n_primes']} primed), "
          f"{r['prompt_checked']} prompt counts checked, 0 mismatches, 0 errors; turn p50 {r['ttft_turn_p50']} s, "
          f"re-prefill {r['reprefill_share']}")
    if n == N2:
        assert r['requests'] == 2 * base['requests'], (r['requests'], base['requests'])
EOF

# 3. report
python3 "$W/scripts/agent_replay.py" table "$OUT" || fail "table"
echo "SMOKE END ok $OUT"
