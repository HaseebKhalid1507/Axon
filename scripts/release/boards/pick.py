#!/usr/bin/env python3
"""pick.py SWEEP_DIR [--tainted] [--smoke] | pick.py --selftest: llama-server's winning arm in agent_sweep.sh's sweep
(release 0.1's agent board).

The rule (pre-registered in docs/log/2026-11-14-release-0.1-prep.md before the sweep ran; amended before any board
ran, after shady's #270 review):
  1. An arm qualifies if its run (speed 1, one round) has 0 errors of any kind (aborted streams included), 0
     prompt-count mismatches, is not tainted (foreign GPU load), and was sent and measured every request of the plan:
     requests = the plan's requests, n turn = the plan's turns (agent_replay.py's plan_counts). The counts are the
     plan's, never axon's: axon's reference run only sets the KV budget, and its failures don't veto an arm.
     An arm's runs are NAME, NAME-r2, NAME-r3 (queue.py reruns an arm whose run's one fault is taint, at most 3
     runs): its first untainted run counts; if every run is tainted, the arm doesn't qualify.
  2. The winner is the qualifying arm with the lowest TTFT-turn p50, the stop rule's own metric, full stop. No
     tie band: an arm that is slower on that metric never wins. (An exact tie, equal to the millisecond: the lower
     TTFT-new p50, then the arm's name.) ITL p99 and the rest are reported for every arm, not used.
Prints a table of every arm, then WINNER; writes SWEEP_DIR/WINNER.env (sourced by agent_board.sh).
Exit 1 if no arm qualifies. --tainted: print the arms whose counted run is tainted.
--smoke (AXON_REL_SMOKE=1, next to other GPU jobs): taint doesn't disqualify; the mechanics only.
--selftest: the rule on fake sweeps (no GPU, no engine)."""
import glob, json, os, re, shutil, sys, tempfile

ARMS = {  # prefix -> (llama.cpp build, -np as a multiple of budget//32768 (0 = the board's default), --llama-args)
    'A0': ('old', 0, ''), 'A1': ('new', 0, ''), 'A2': ('new', 0, '--no-cache-idle-slots'),
    'A3': ('new', 0, '--no-cache-idle-slots -ub 2048'), 'A4': ('new', 2, ''),
    'A5': ('new', 0, '--no-cache-idle-slots -cram 16384'), 'A6': ('new', 0, '-cram 16384'),
}
COLS = ['ttft_turn_p50', 'ttft_new_p50', 'ttft_resume_p50', 'itl_p99_ms', 'goodput_tok_s', 'reprefill_share']


def run(d):
    fs = glob.glob(os.path.join(d, 'run-*-s1-r0.json'))
    return json.load(open(fs[0])) if fs else None


def attempt(name):
    """'A2-noidle-r3' -> ('A2-noidle', 3); 'A2-noidle' -> ('A2-noidle', 1)."""
    m = re.match(r'(.*)-r(\d+)$', name)
    return (m.group(1), int(m.group(2))) if m else (name, 1)


def load(out):
    """(axon's reference run, {arm: (run, the dir it came from)}): an arm's first untainted run (NAME, NAME-r2, ...),
    else its first run (tainted: shown, doesn't qualify)."""
    ax = run(os.path.join(out, 'axon'))
    tries = {}
    for d in glob.glob(os.path.join(out, 'A[0-9]-*')):
        if os.path.isdir(d):
            base, k = attempt(os.path.basename(d))
            tries.setdefault(base, []).append((k, os.path.basename(d), run(d)))
    arms = {}
    for base in sorted(tries):
        ts = sorted(tries[base])
        good = [(r, name) for _, name, r in ts if r and not r.get('tainted')]
        arms[base] = good[0] if good else (ts[0][2], ts[0][1])
    return ax, arms


def taint_only(r):
    """True if the run's one fault is foreign GPU load: queue.py reruns it. Any other fault is the run's result."""
    return bool(r) and bool(r.get('tainted')) and why_not(dict(r, tainted=False)) == ''


def why_not(r, smoke=False):
    """'' if the run qualifies, else why not. The same checks for every engine (read_board.py's clean())."""
    if r is None:
        return 'no run (the engine failed: see its log)'
    bad = []
    if r.get('errors'):
        bad.append(f"{r['errors']} errors")
    if r.get('prompt_mismatch'):
        bad.append(f"{r['prompt_mismatch']} prompt-count mismatches")
    if r.get('tainted') and not smoke:
        bad.append('tainted')
    if r.get('plan_requests') is None or r.get('plan_turn') is None:
        bad.append('no plan counts (an agent_replay.py older than the rule)')
    else:
        if r.get('requests') != r['plan_requests']:
            bad.append(f"{r.get('requests')} of the plan's {r['plan_requests']} requests")
        if r.get('n_turn') != r['plan_turn']:
            bad.append(f"n turn {r.get('n_turn')} vs the plan's {r['plan_turn']}")
    return ', '.join(bad)


def pick(out, smoke=False, say=print):
    """The winner's name (WINNER.env written), or None. Prints the table through `say`."""
    ax, arms = load(out)
    f = lambda v, n=3: 'n/a' if v is None else f'{v:.{n}g}'  # noqa: E731
    say('| arm | ' + ' | '.join(COLS) + ' | errors | mismatches | requests / plan | n turn / plan | kv_tokens | qualifies |')
    say('|---' * (len(COLS) + 7) + '|')
    rows = [('axon (reference: sets the budget, not judged)', ax, 'axon')] + [(n, r, src) for n, (r, src) in arms.items()]
    for n, r, src in rows:
        r0 = r or {}
        w = ('(not judged) ' + (why_not(r, smoke) or 'clean')) if src == 'axon' else (why_not(r, smoke) or 'yes')
        say(f'| {n}{"" if src in (n, "axon") else " (" + src + ")"} | ' + ' | '.join(f(r0.get(c)) for c in COLS)
            + f' | {r0.get("errors", "n/a")} | {r0.get("prompt_mismatch", "n/a")}'
            + f' | {r0.get("requests", "n/a")} / {r0.get("plan_requests", "n/a")}'
            + f' | {r0.get("n_turn", "n/a")} / {r0.get("plan_turn", "n/a")} | {r0.get("kv_tokens", "n/a")} | {w} |')
    q = {n: r for n, (r, _) in arms.items() if r and not why_not(r, smoke) and r.get('ttft_turn_p50') is not None}
    if not q:
        say('WINNER none: no arm qualifies')
        return None
    win = min(q, key=lambda n: (q[n]['ttft_turn_p50'], q[n].get('ttft_new_p50') or 0, n))
    which, mult, largs = ARMS[win[:2]]
    say(f'WINNER {win}: TTFT-turn p50 {q[win]["ttft_turn_p50"]:.3g} s, the lowest of {len(q)} qualifying arm(s) '
        f'({", ".join(f"{n} {q[n]["ttft_turn_p50"]:.3g}" for n in sorted(q, key=lambda n: q[n]["ttft_turn_p50"]))})')
    if ax and ax.get('ttft_turn_p50'):
        say(f'axon reference TTFT-turn p50 {ax["ttft_turn_p50"]:.3g} s: llama-server / axon = '
            f'{q[win]["ttft_turn_p50"] / ax["ttft_turn_p50"]:.3g} on this one-round sweep (the board decides, not this)')
    pool = int(open(os.path.join(out, 'POOL')).read())
    par = mult * max(1, pool // 32768) if mult else 0
    with open(os.path.join(out, 'WINNER.env'), 'w') as fh:
        fh.write(f"ARM='{win}'\nWHICH='{which}'\nPAR={par}\nLARGS='{largs}'\n")
    return win


def selftest():
    def fake(out, name, **kw):
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        r = dict(errors=0, prompt_mismatch=0, tainted=False, requests=120, plan_requests=120, n_turn=100,
                 plan_turn=100, ttft_turn_p50=1.0, ttft_new_p50=2.0, itl_p99_ms=100.0, kv_tokens=65536)
        r.update(kw)
        json.dump(r, open(os.path.join(d, f'run-{"axon" if name == "axon" else "llama"}-f16-s1-r0.json'), 'w'))

    quiet = lambda *_: None  # noqa: E731
    t = tempfile.mkdtemp(prefix='pick-selftest-')
    try:
        out = os.path.join(t, 's1')
        os.makedirs(out)
        open(os.path.join(out, 'POOL'), 'w').write('65536\n')
        # axon's own run failed 10 turns: under the old rule (n turn = axon's) every full arm was out
        fake(out, 'axon', errors=10, n_turn=90, ttft_turn_p50=0.12)
        fake(out, 'A0-4413a19-defaults', ttft_turn_p50=1.00, itl_p99_ms=100)
        fake(out, 'A1-b12387-defaults', ttft_turn_p50=0.97, itl_p99_ms=150)
        fake(out, 'A2-noidle', ttft_turn_p50=0.90, errors=1)                    # one error: out
        fake(out, 'A3-noidle-ub2048', ttft_turn_p50=0.95, itl_p99_ms=600)       # best TTFT-turn, worst ITL p99
        fake(out, 'A4-2x-slots', ttft_turn_p50=0.85, n_turn=99)                 # a turn short of the plan: out
        fake(out, 'A5-noidle-cram16g', ttft_turn_p50=0.80, tainted=True)        # tainted; its rerun counts
        fake(out, 'A5-noidle-cram16g-r2', ttft_turn_p50=0.99)
        fake(out, 'A6-cram16g', ttft_turn_p50=0.70, requests=119)               # a request never sent: out
        os.makedirs(os.path.join(out, 'A7-none'))                               # no run file: out (no such arm)
        assert pick(out, say=quiet) == 'A3-noidle-ub2048', 'the lowest TTFT-turn p50 wins, whatever its ITL p99'
        env = open(os.path.join(out, 'WINNER.env')).read()
        assert env == "ARM='A3-noidle-ub2048'\nWHICH='new'\nPAR=0\nLARGS='--no-cache-idle-slots -ub 2048'\n", env
        _, arms = load(out)
        assert arms['A5-noidle-cram16g'][1] == 'A5-noidle-cram16g-r2' and why_not(arms['A5-noidle-cram16g'][0]) == ''
        assert why_not(arms['A4-2x-slots'][0]) == "n turn 99 vs the plan's 100"
        assert why_not(arms['A6-cram16g'][0]) == "119 of the plan's 120 requests"
        assert why_not(arms['A2-noidle'][0]) == '1 errors' and why_not(None).startswith('no run')
        assert why_not(dict(errors=0, tainted=True, requests=1, plan_requests=1, n_turn=1, plan_turn=1), smoke=True) == ''
        assert 'no plan counts' in why_not(dict(errors=0, requests=1, n_turn=1))
        # an exact tie: the lower TTFT-new p50, then the name
        fake(out, 'A0-4413a19-defaults', ttft_turn_p50=0.95, ttft_new_p50=1.5)
        assert pick(out, say=quiet) == 'A0-4413a19-defaults'
        # A4 (twice the slots) wins: -np = 2 x budget//32768
        fake(out, 'A4-2x-slots', ttft_turn_p50=0.50)
        assert pick(out, say=quiet) == 'A4-2x-slots' and 'PAR=4\n' in open(os.path.join(out, 'WINNER.env')).read()
        # nothing qualifies: no winner (exit 1 in main)
        out2 = os.path.join(t, 's2')
        os.makedirs(out2)
        open(os.path.join(out2, 'POOL'), 'w').write('65536\n')
        fake(out2, 'axon')
        fake(out2, 'A1-b12387-defaults', errors=2)
        fake(out2, 'A2-noidle', tainted=True)
        fake(out2, 'A3-noidle-ub2048', ttft_turn_p50=0.1, tainted=True)          # tainted 3 times: out, however fast
        fake(out2, 'A3-noidle-ub2048-r2', ttft_turn_p50=0.1, tainted=True)
        fake(out2, 'A3-noidle-ub2048-r3', ttft_turn_p50=0.1, tainted=True)
        fake(out2, 'A4-2x-slots', tainted=True)                                 # clean on its 3rd run: that run counts
        fake(out2, 'A4-2x-slots-r2', tainted=True)
        fake(out2, 'A4-2x-slots-r3', ttft_turn_p50=3.0)
        fake(out2, 'A5-noidle-cram16g', tainted=True, errors=1)                  # tainted AND an error: no rerun
        _, arms2 = load(out2)
        assert arms2['A3-noidle-ub2048'][1] == 'A3-noidle-ub2048' and why_not(arms2['A3-noidle-ub2048'][0]) == 'tainted'
        assert arms2['A4-2x-slots'][1] == 'A4-2x-slots-r3' and why_not(arms2['A4-2x-slots'][0]) == ''
        assert taint_only(arms2['A3-noidle-ub2048'][0]) and not taint_only(arms2['A5-noidle-cram16g'][0])
        assert not taint_only(arms2['A4-2x-slots'][0]) and not taint_only(None)
        assert attempt('A4-2x-slots-r3') == ('A4-2x-slots', 3) and attempt('A4-2x-slots') == ('A4-2x-slots', 1)
        assert pick(out2, say=quiet) == 'A4-2x-slots', 'a clean 3rd run qualifies'
        for n in ('A4-2x-slots', 'A4-2x-slots-r2', 'A4-2x-slots-r3'):
            shutil.rmtree(os.path.join(out2, n))
        os.remove(os.path.join(out2, 'WINNER.env'))
        assert pick(out2, say=quiet) is None and not os.path.exists(os.path.join(out2, 'WINNER.env'))
        assert pick(out2, smoke=True, say=quiet) == 'A3-noidle-ub2048', 'smoke mode: taint does not disqualify'
    finally:
        shutil.rmtree(t)
    print('SELFTEST pick OK: lowest TTFT-turn p50 wins over a lower ITL p99; axon\'s failed turns veto nothing; an '
          'error, a short turn count, an unsent request, no run, no plan counts disqualify; an arm\'s first untainted run '
          '(of NAME, -r2, -r3) counts; tainted on every run: out; taint_only; exact ties; -np for A4; no winner -> none')


def main():
    if '--selftest' in sys.argv:
        selftest()
        return
    out = sys.argv[1]
    smoke = '--smoke' in sys.argv
    if '--tainted' in sys.argv:
        print(' '.join(b for b, (r, _) in load(out)[1].items() if r and r.get('tainted')))
        return
    sys.exit(0 if pick(out, smoke) else 1)


if __name__ == '__main__':
    main()
