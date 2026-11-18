#!/usr/bin/env python3
"""read_board.py BOARD_DIR [--pool N] | read_board.py --selftest: the stop rule's reading of release 0.1's agent board
(agent_board.sh's output), as pre-registered in docs/log/2026-11-14-release-0.1-prep.md (amended after shady's #270
review, before any board ran).

  - A run is clean if it has 0 errors of any kind, 0 prompt-count mismatches, is not tainted, and was sent and
    measured every request and turn of the plan (pick.py's why_not: the same checks for axon and llama-server).
  - A round counts only if BOTH engines' speed-1 runs in it are clean (paired rounds). Fewer than 2: NO VERDICT
    (rerun the rounds; nothing is read).
  - R = median over the counted rounds of llama-server's TTFT-turn p50, divided by the median over the same rounds
    of axon's TTFT-turn p50 (a ratio of medians; R > 1: axon is faster). The round spread: the per-round ratios
    llama_i / axon_i over the counted rounds, their min and max.
  - Verdict, the first that applies:
      LLAMA_LEADS  every counted round's ratio < 1: "llama-server leads on turn TTFT, by 1/R"; in the notes' one-sentence
                   summary.
      LEVEL        the round spread includes 1 (min <= 1 <= max): "level on turn TTFT (R = .., rounds ..-..)"; in the
                   one-sentence summary.
      THIN         R <= 1.5: "axon leads by R, thinner than the 10-27 8.7x"; the notes say so with the numbers.
      LEAD         R > 1.5: the notes state R as measured, ITL p99 beside it.
    R <= 1.5 (LLAMA_LEADS, LEVEL, THIN): MoE in the paged pool goes first in the roadmap (item 7).
  - Speed 4 is printed the same way and marked "reported, not read".
--pool N: the sweep's KV budget; a run whose kv_tokens differ is flagged (llama-server's start shrink), not dropped."""
import glob, json, os, re, shutil, statistics, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pick import why_not  # noqa: E402

ENG = ('axon', 'llama')


def runs(out, speed):
    """{round: {engine: run}} for one speed."""
    got = {}
    for f in glob.glob(os.path.join(out, f'run-*-f16-s{speed}-r*.json')):
        m = re.match(r'run-(axon|llama)-f16-s[\d.]+-r(\d+)\.json$', os.path.basename(f))
        if m:
            got.setdefault(int(m.group(2)), {})[m.group(1)] = json.load(open(f))
    return got


def reading(out, speed='1', pool=None, read=True):
    """(verdict, R, lines): the table and the verdict for one speed."""
    rs = runs(out, speed)
    L = [f'### speed {speed}' + ('' if read else ' (reported, not read)'),
         '| round | engine | TTFT-turn p50 s | ITL p99 ms | errors | mismatches | requests / plan | n turn / plan '
         '| tainted | kv_tokens | GPU before | clean |', '|---' * 12 + '|']
    counted = []
    for rnd in sorted(rs):
        ok = True
        for e in ENG:
            r = rs[rnd].get(e)
            w = why_not(r)
            ok = ok and not w
            r0 = r or {}
            kv = r0.get('kv_tokens', 'n/a')
            if pool and r and kv != pool:
                kv = f'{kv} (budget {pool}: differs)'
            L.append(f'| {rnd} | {e} | {r0.get("ttft_turn_p50", "n/a")} | {r0.get("itl_p99_ms", "n/a")} '
                     f'| {r0.get("errors", "n/a")} | {r0.get("prompt_mismatch", "n/a")} '
                     f'| {r0.get("requests", "n/a")} / {r0.get("plan_requests", "n/a")} '
                     f'| {r0.get("n_turn", "n/a")} / {r0.get("plan_turn", "n/a")} | {r0.get("tainted", "n/a")} '
                     f'| {kv} | {r0.get("gpu_before", "n/a")} | {w or "yes"} |')
        if ok:
            counted.append(rnd)
    if len(counted) < 2:
        L.append(f'VERDICT NO_VERDICT speed {speed}: {len(counted)} round(s) with both engines clean (rounds '
                 f'{counted}); at least 2 are needed: rerun the rounds, read nothing')
        return 'NO_VERDICT', None, L
    lt = [rs[i]['llama']['ttft_turn_p50'] for i in counted]
    at = [rs[i]['axon']['ttft_turn_p50'] for i in counted]
    R = statistics.median(lt) / statistics.median(at)
    ratios = [x / y for x, y in zip(lt, at)]
    lo, hi = min(ratios), max(ratios)
    itl = {e: statistics.median(rs[i][e]['itl_p99_ms'] for i in counted) for e in ENG}
    spread = f'rounds {counted}: per-round ratios {", ".join(f"{x:.3g}" for x in ratios)} (spread {lo:.3g}-{hi:.3g})'
    base = (f'TTFT-turn p50 median: llama-server {statistics.median(lt):.3g} s, axon {statistics.median(at):.3g} s; '
            f'R = {R:.3g}; {spread}; ITL p99 median: axon {itl["axon"]:.4g} ms, llama-server {itl["llama"]:.4g} ms')
    if hi < 1:
        v, say = 'LLAMA_LEADS', f'llama-server leads on turn TTFT, by 1/R = {1 / R:.3g}x (every counted round)'
    elif lo <= 1 <= hi:
        v, say = 'LEVEL', f'level on turn TTFT: R = {R:.3g}, but the round spread {lo:.3g}-{hi:.3g} includes 1'
    elif R <= 1.5:
        v, say = 'THIN', f'axon leads on turn TTFT by R = {R:.3g}x, thinner than the 8.7x claimed from 10-27'
    else:
        v, say = 'LEAD', f'axon leads on turn TTFT by R = {R:.3g}x'
    tail = '' if not read else (' -> one-sentence summary; MoE in the paged pool first (roadmap item 7)'
                                if v in ('LLAMA_LEADS', 'LEVEL') else
                                ' -> MoE in the paged pool first (roadmap item 7)' if v == 'THIN' else '')
    L.append(base)
    L.append(f'VERDICT {v if read else "REPORTED_" + v} speed {speed}: {say}{tail}')
    return v, R, L


def selftest():
    t = tempfile.mkdtemp(prefix='read-board-selftest-')

    def board(name, rounds, speed='1'):
        d = os.path.join(t, name)
        os.makedirs(d, exist_ok=True)
        for rnd, (ax, ll) in enumerate(rounds):
            for e, kw in (('axon', ax), ('llama', ll)):
                r = dict(errors=0, prompt_mismatch=0, tainted=False, requests=120, plan_requests=120, n_turn=100,
                         plan_turn=100, ttft_turn_p50=1.0, itl_p99_ms=100.0, kv_tokens=65536)
                r.update(kw if isinstance(kw, dict) else dict(ttft_turn_p50=kw))
                json.dump(r, open(os.path.join(d, f'run-{e}-f16-s{speed}-r{rnd}.json'), 'w'))
        return d
    try:
        v, R, _ = reading(board('lead', [(0.5, 1.0), (0.5, 1.1), (0.6, 0.9)]))
        assert v == 'LEAD' and abs(R - 1.0 / 0.5) < 1e-9, (v, R)   # ratio of medians, not median of ratios (2.0)
        v, R, L = reading(board('thin', [(1.0, 1.3), (1.0, 1.2), (1.0, 1.4)]))
        assert v == 'THIN' and abs(R - 1.3) < 1e-9 and 'roadmap item 7' in L[-1], (v, R)
        v, R, L = reading(board('level', [(1.0, 1.3), (1.0, 0.95), (1.0, 1.2)]))
        assert v == 'LEVEL' and 'one-sentence' in L[-1], (v, R)
        v, R, L = reading(board('llama', [(1.0, 0.5), (1.0, 0.6), (1.0, 0.4)]))
        assert v == 'LLAMA_LEADS' and abs(1 / R - 2.0) < 1e-9 and 'by 1/R = 2x' in L[-1], (v, R, L[-1])
        # axon's errors drop the round exactly as llama-server's do (and a short turn count, an unsent request)
        v, R, L = reading(board('axerr', [(0.5, 1.0), (dict(ttft_turn_p50=0.1, errors=3), 1.0), (0.5, 1.2)]))
        assert v == 'LEAD' and abs(R - 1.1 / 0.5) < 1e-9 and 'rounds [0, 2]' in L[-2], (v, R, L[-2])
        v, R, _ = reading(board('axshort', [(dict(ttft_turn_p50=0.1, n_turn=90), 1.0), (0.5, 1.0), (0.5, 1.0)]))
        assert v == 'LEAD' and abs(R - 2.0) < 1e-9
        v, R, _ = reading(board('llsent', [(0.5, dict(ttft_turn_p50=0.1, requests=119)), (0.5, 1.0), (0.5, 1.0)]))
        assert abs(R - 2.0) < 1e-9
        # fewer than 2 paired clean rounds: no verdict, no R
        v, R, L = reading(board('none', [(0.5, dict(errors=1)), (dict(tainted=True), 1.0), (0.5, 1.0)]))
        assert v == 'NO_VERDICT' and R is None and 'read nothing' in L[-1], (v, L[-1])
        # speed 4 is reported, never read; the budget check flags, doesn't drop
        v, R, L = reading(board('s4', [(0.5, 1.0), (0.5, 1.0)], speed='4'), speed='4', read=False)
        assert L[-1].startswith('VERDICT REPORTED_LEAD speed 4') and '(reported, not read)' in L[0]
        v, R, L = reading(board('pool', [(0.5, dict(ttft_turn_p50=1.0, kv_tokens=55552)), (0.5, 1.0)]), pool=65536)
        assert v == 'LEAD' and any('55552 (budget 65536: differs)' in x for x in L)
    finally:
        shutil.rmtree(t)
    print('SELFTEST read_board OK: R = ratio of the medians over rounds where both engines are clean; LEAD, THIN, '
          'LEVEL (spread includes 1), LLAMA_LEADS (1/R); axon\'s errors / short turns drop a round like llama-server\'s; '
          '< 2 rounds -> NO_VERDICT; speed 4 reported only; a budget mismatch flagged')


def main():
    if '--selftest' in sys.argv:
        selftest()
        return
    out = sys.argv[1]
    pool = int(sys.argv[sys.argv.index('--pool') + 1]) if '--pool' in sys.argv else None
    for sp, rd in (('1', True), ('4', False)):
        if runs(out, sp):
            print('\n'.join(reading(out, sp, pool, rd)[2]) + '\n')


if __name__ == '__main__':
    main()
