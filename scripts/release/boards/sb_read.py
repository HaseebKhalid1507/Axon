#!/usr/bin/env python3
"""sb_read.py SB_JSON | sb_read.py --selftest: the reading of release 0.1's single-stream board (sb_pod.py's sb.json),
as pre-registered in docs/log/2026-11-14-release-0.1-prep.md (amended after shady's #270 review, before any board ran).

Per cell (model x pp560 / pp2100 / tg128, tok/s, higher is better):
  - axon: the median of its rounds. The reference: the faster of the two llama.cpp builds (b12387's official build,
    `llama`, and bella's 4413a19, `llama_4413a19`), each at its best ubatch per round (sb_pod.py's `llama*` columns),
    by median. The margin m = axon / reference - 1.
  - AHEAD: m > +2 % and every axon round beats every round of the reference build (the round ranges don't overlap).
    BEHIND: m < -2 % and every axon round is below every reference round. Else PARITY. A missing engine or round:
    NO_VERDICT for that cell.
  - Reported: every cell with both builds' medians (and their -ub 512 runs, the 10-18 board's protocol), the counts,
    the thinnest AHEAD margins, the worst losses (the most negative margins, whatever their verdict) and every
    BEHIND cell. Nothing PARITY is called a lead."""
import json, shutil, statistics, sys, tempfile, os

BUILDS = ('llama', 'llama_4413a19')
BAND = 0.02


def cell(c):
    """(verdict, margin, reference build, axon median, {build: median}) for one cell's samples."""
    ax = c.get('axon0') or []
    meds = {b: statistics.median(c[b]) for b in BUILDS if c.get(b)}
    if not ax or len(meds) < len(BUILDS):
        return 'NO_VERDICT', None, None, (statistics.median(ax) if ax else None), meds
    ref = max(meds, key=meds.get)
    a = statistics.median(ax)
    m = a / meds[ref] - 1
    if m > BAND and min(ax) > max(c[ref]):
        v = 'AHEAD'
    elif m < -BAND and max(ax) < min(c[ref]):
        v = 'BEHIND'
    else:
        v = 'PARITY'
    return v, m, ref, a, meds


def read(sb):
    rows = []
    for tag, jobs in sb['cells'].items():
        for job, c in jobs.items():
            rows.append((tag, job, c) + cell(c))
    L = ['| model | cell | axon median [min-max] | b12387 best ub | 4413a19 best ub | b12387 ub512 | 4413a19 ub512 '
         '| reference | margin | verdict |', '|---' * 10 + '|']
    f = lambda v: 'n/a' if v is None else f'{v:.1f}'  # noqa: E731
    for tag, job, c, v, m, ref, a, meds in rows:
        ax = c.get('axon0') or []
        u = {b: (statistics.median(c[b + '_ub512']) if c.get(b + '_ub512') else None) for b in BUILDS}
        L.append(f'| {tag} | {job} | {f(a)} [{f(min(ax) if ax else None)}-{f(max(ax) if ax else None)}] '
                 f'| {f(meds.get("llama"))} | {f(meds.get("llama_4413a19"))} | {f(u["llama"])} | {f(u["llama_4413a19"])} '
                 f'| {ref or "n/a"} | {"n/a" if m is None else f"{m:+.1%}"} | {v} |')
    n = {k: sum(1 for r in rows if r[3] == k) for k in ('AHEAD', 'PARITY', 'BEHIND', 'NO_VERDICT')}
    L.append(f'COUNTS {len(rows)} cells: ' + ', '.join(f'{k} {v}' for k, v in n.items()))
    ahead = sorted((r for r in rows if r[3] == 'AHEAD'), key=lambda r: r[4])
    L.append('THINNEST AHEAD: ' + (', '.join(f'{r[0]} {r[1]} {r[4]:+.1%}' for r in ahead[:3]) or 'none'))
    scored = sorted((r for r in rows if r[4] is not None), key=lambda r: r[4])
    L.append('WORST MARGINS: ' + (', '.join(f'{r[0]} {r[1]} {r[4]:+.1%} ({r[3]})' for r in scored[:3]) or 'none'))
    L.append('BEHIND: ' + (', '.join(f'{r[0]} {r[1]} {r[4]:+.1%} vs {r[5]}' for r in rows if r[3] == 'BEHIND') or 'none'))
    return n, rows, L


def selftest():
    sb = {'cells': {
        # axon 3 % ahead of the faster build, ranges apart: AHEAD
        '7B': {'tg128': {'axon0': [103, 104, 103.5], 'llama': [99, 100, 100.5], 'llama_4413a19': [90, 91, 92],
                         'llama_ub512': [99, 100, 100], 'llama_4413a19_ub512': [90, 90, 90]},
               # 3 % ahead of b12387 but 4413a19 is faster: judged against 4413a19 -> PARITY
               'pp560': {'axon0': [1030, 1031, 1029], 'llama': [1000, 1001, 999], 'llama_4413a19': [1020, 1021, 1019]},
               # 4 % ahead by median but the ranges overlap: PARITY
               'pp2100': {'axon0': [1040, 1100, 990], 'llama': [1000, 1001, 999], 'llama_4413a19': [900, 901, 899]}},
        '0.5B': {'tg128': {'axon0': [90, 91, 90], 'llama': [99, 100, 100], 'llama_4413a19': [95, 96, 97]},   # BEHIND
                 'pp560': {'axon0': [100, 100, 100], 'llama': [100, 101, 99]},                            # one build missing
                 'pp2100': {'axon0': [1010, 1011, 1009], 'llama': [1000, 1001, 999], 'llama_4413a19': [1, 1, 1]}}}}
    n, rows, L = read(sb)
    v = {(r[0], r[1]): (r[3], r[5]) for r in rows}
    assert v[('7B', 'tg128')] == ('AHEAD', 'llama'), v
    assert v[('7B', 'pp560')] == ('PARITY', 'llama_4413a19'), 'the faster build is the reference'
    assert v[('7B', 'pp2100')] == ('PARITY', 'llama'), 'overlapping rounds are never AHEAD'
    assert v[('0.5B', 'tg128')] == ('BEHIND', 'llama')
    assert v[('0.5B', 'pp560')] == ('NO_VERDICT', None)
    assert v[('0.5B', 'pp2100')] == ('PARITY', 'llama'), '+1 % is inside the band'
    assert n == {'AHEAD': 1, 'PARITY': 3, 'BEHIND': 1, 'NO_VERDICT': 1}, n
    assert L[-3].startswith('THINNEST AHEAD: 7B tg128 +3.')
    assert L[-2].startswith('WORST MARGINS: 0.5B tg128 -10.0% (BEHIND)') and L[-1] == 'BEHIND: 0.5B tg128 -10.0% vs llama', L[-3:]
    t = tempfile.mkdtemp(prefix='sb-read-selftest-')
    try:   # sb_pod.py's file round-trips
        p = os.path.join(t, 'sb.json')
        json.dump(sb, open(p, 'w'))
        assert read(json.load(open(p)))[0] == n
    finally:
        shutil.rmtree(t)
    print('SELFTEST sb_read OK: the faster llama.cpp build per cell is the reference; AHEAD / BEHIND need > 2 % and '
          'round ranges apart; a missing build -> NO_VERDICT; counts, thinnest leads, worst margins, BEHIND listed')


def main():
    if '--selftest' in sys.argv:
        selftest()
        return
    print('\n'.join(read(json.load(open(sys.argv[1])))[2]))


if __name__ == '__main__':
    main()
