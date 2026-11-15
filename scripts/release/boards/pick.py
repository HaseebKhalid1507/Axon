#!/usr/bin/env python3
"""pick.py SWEEP_DIR [--tainted]: llama-server's winning arm in agent_sweep.sh's sweep (release 0.1's agent board).

The rule (pre-registered in docs/log/2026-11-14-release-0.1-prep.md before the sweep ran):
  1. An arm qualifies if its run (speed 1, one round) has 0 errors of any kind, 0 prompt-count mismatches, is not
     tainted (foreign GPU load), and measured every turn axon's reference run measured (n turn equal).
     A tainted arm's rerun (NAME-r2) counts instead of it if the rerun is clean.
  2. The winner is the qualifying arm with the lowest TTFT-turn p50 (the stop rule's metric).
  3. Arms within 5 % of that TTFT-turn p50 tie; among them the lowest ITL p99 wins, then the lowest TTFT-new p50.
Prints a table of every arm, then WINNER; writes SWEEP_DIR/WINNER.env (sourced by agent_board.sh).
Exit 1 if no arm qualifies. --tainted: print the arms whose run is tainted (agent_sweep.sh reruns them once).
--smoke (AXON_REL_SMOKE=1, next to other GPU jobs): taint doesn't disqualify; the mechanics only."""
import glob, json, os, sys

ARMS = {  # prefix -> (llama.cpp build, -np as a multiple of budget//32768 (0 = the board's default), --llama-args)
    'A0': ('old', 0, ''), 'A1': ('new', 0, ''), 'A2': ('new', 0, '--no-cache-idle-slots'),
    'A3': ('new', 0, '--no-cache-idle-slots -ub 2048'), 'A4': ('new', 2, ''),
    'A5': ('new', 0, '--no-cache-idle-slots -cram 16384'),
}
out = sys.argv[1]


def run(d):
    fs = glob.glob(os.path.join(d, 'run-*-s1-r0.json'))
    return json.load(open(fs[0])) if fs else None


ax = run(os.path.join(out, 'axon'))
arms = {}
for d in sorted(glob.glob(os.path.join(out, 'A[0-9]-*'))):
    name = os.path.basename(d)
    base = name[:-3] if name.endswith('-r2') else name
    r = run(d)
    if name.endswith('-r2'):
        if r and not r.get('tainted'):
            arms[base] = (r, name)
    else:
        arms.setdefault(base, (r, name))
if '--tainted' in sys.argv:
    print(' '.join(b for b, (r, _) in arms.items() if r and r.get('tainted')))
    sys.exit(0)


def why_not(r):
    if r is None:
        return 'no run (the engine failed: see its log)'
    bad = []
    if r.get('errors'):
        bad.append(f"{r['errors']} errors")
    if r.get('prompt_mismatch'):
        bad.append(f"{r['prompt_mismatch']} prompt-count mismatches")
    if r.get('tainted') and '--smoke' not in sys.argv:
        bad.append('tainted')
    if ax and r.get('n_turn') != ax.get('n_turn'):
        bad.append(f"n turn {r.get('n_turn')} vs axon's {ax.get('n_turn')}")
    return ', '.join(bad)


f = lambda v, n=3: 'n/a' if v is None else f'{v:.{n}g}'  # noqa: E731
cols = ['ttft_turn_p50', 'ttft_new_p50', 'ttft_resume_p50', 'itl_p99_ms', 'goodput_tok_s', 'reprefill_share']
print('| arm | ' + ' | '.join(cols) + ' | kv_tokens | qualifies |')
print('|---' * (len(cols) + 3) + '|')
rows = [('axon (reference)', ax, 'axon')] + [(n, r, src) for n, (r, src) in arms.items()]
for n, r, src in rows:
    w = '' if n.startswith('axon') else (why_not(r) or 'yes')
    print(f'| {n}{"" if src in (n, "axon") else " (" + src + ")"} | '
          + ' | '.join(f((r or {}).get(c)) for c in cols) + f' | {(r or {}).get("kv_tokens", "n/a")} | {w} |')
q = {n: r for n, (r, _) in arms.items() if r and not why_not(r)}
if not q:
    print('WINNER none: no arm qualifies')
    sys.exit(1)
best = min(r['ttft_turn_p50'] for r in q.values())
tie = [n for n, r in q.items() if r['ttft_turn_p50'] <= best * 1.05]
win = min(tie, key=lambda n: (q[n]['itl_p99_ms'], q[n]['ttft_new_p50']))
which, mult, largs = ARMS[win[:2]]
print(f'WINNER {win}: TTFT-turn p50 {q[win]["ttft_turn_p50"]:.3g} s (best {best:.3g}; tied within 5 %: {", ".join(tie)})')
if ax:
    print(f'axon reference TTFT-turn p50 {ax["ttft_turn_p50"]:.3g} s: llama-server / axon = '
          f'{q[win]["ttft_turn_p50"] / ax["ttft_turn_p50"]:.3g} on this one-round sweep (the board decides, not this)')
pool = int(open(os.path.join(out, 'POOL')).read())
par = mult * max(1, pool // 32768) if mult else 0
with open(os.path.join(out, 'WINNER.env'), 'w') as fh:
    fh.write(f"ARM='{win}'\nWHICH='{which}'\nPAR={par}\nLARGS='{largs}'\n")
