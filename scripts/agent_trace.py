#!/usr/bin/env python3
"""Synaps agent-workload trace extractor: NUMBERS ONLY (stdlib only).

    agent_trace.py [--logs DIR] [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--out FILE] [--summary]
    agent_trace.py --check FILE          (the privacy check: exit 1 if FILE has any string field > 16 bytes)

Reads ~/.synaps-cli (default --logs) and writes one JSON trace (default
~/.cache/axon-agent/traces/synaps-<from>-<to>.json; traces never go into the repo).

PRIVACY (hard rule, docs/log/2026-10-27-agent-replay.md):
- From `synaps.log.*` it parses only four line kinds, by regex, and only their numeric fields:
  `Starting API request` (a timestamp), `Token Usage: I input | O output | R cache_read | C cache_create`,
  `cache keep-warm ping ... cache_read=.. input=.. output=..` (to drop the pings' own usage lines) and
  `subagent_start: dispatching` / `subagent_resume: dispatching` (a timestamp). Every other line is skipped
  unread. The names and ids those lines carry are never stored.
- From `subagent-history/<parent>/sa_*.json` it reads the directory layout (files per parent) and, per file,
  only the number of history entries. Message text and system prompts are not opened as strings by this
  script beyond the JSON parser itself; nothing derived from text other than a count is kept.
- The output has integers, `null` and the enum strings "sys" / "session". `--check` enforces it.
- The Axon repo is PUBLIC: traces from real logs are NEVER committed, not even these numbers-only ones
  (per-request timestamps would publish when the owner works). They live in ~/.cache/axon-agent/traces; the
  script refuses to write one inside a git tree. The repo carries only `--synthetic` example traces. Anything
  derived that is shared (a PR, a gist) is made with `--anonymize-time`: dates and times of day stripped, each
  day shifted to start at t = 0.
- A record's `fork` is the opaque id of the chain whose cache boundary its first cached prefix ends on (a
  resumed or re-branched conversation); `shared[].with` is then "session". A new chain's cached prefix that
  ends on no known boundary is a shared pool ("sys": system prompt + tool catalog); equal token counts are
  taken as the same pool.

How the structure is recovered (the log has no session id on API requests):
- **Sessions = cache chains.** Anthropic's prompt cache makes request n+1 of a conversation read exactly what
  request n read plus what it wrote: cache_read(n+1) == cache_read(n) + cache_create(n) (MEASURED: holds for
  the large majority of requests; `--summary` prints the share). A usage line whose cache_read matches an open
  chain's expected value continues that chain; anything else starts a new chain. A new chain whose first
  cache_read is > 0 read a prefix another conversation wrote (system prompt + tool catalog, or a resumed
  subagent): that is the cross-session shared prefix ("sys").
- **Arrival** = the `Starting API request` line paired with the usage line: the latest unpaired start on the
  same log thread, else the latest unpaired start before it. ESTIMATED to within the request's duration.
- **Subagents**: a new chain starting within 60 s after a `subagent_start` / `subagent_resume` dispatch line is
  depth 1; its parent is the chain with the latest arrival before that dispatch; dispatch lines <= 5 s apart
  are one burst, and `fanout` is the burst's size.
Token counts are Claude's tokenizer; the replay re-tokenizes synthetic filler for the target model.
"""
import argparse, bisect, datetime as dt, glob, json, os, re, statistics, sys

TS = r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z\s+\w+\s+ThreadId\((\d+)\)\s+'
RE_START = re.compile(TS + r'Starting API request\b')
RE_USAGE = re.compile(TS + r'Token Usage: (\d+) input \| (\d+) output \| (\d+) cache_read \| (\d+) cache_create')
RE_PING = re.compile(TS + r'cache keep-warm ping\b.*?\bcache_read=(\d+) cache_write=(\d+) input=(\d+) output=(\d+)')
RE_DISPATCH = re.compile(TS + r'subagent_(start|resume): dispatching\b')
CHAIN_WINDOW_MS = 60 * 60 * 1000   # a chain not continued within an hour is closed (the cache TTL is <= 1 h)


def ms(ts):
    return int(dt.datetime.fromisoformat(ts).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def read_logs(logs, d0, d1):
    starts, usages, pings, dispatches = [], [], [], []
    files = sorted(glob.glob(os.path.join(logs, 'synaps.log.*')))
    used = []
    for f in files:
        day = f.rsplit('.', 1)[-1]
        if not re.fullmatch(r'\d{4}-\d\d-\d\d', day) or day < d0 or day > d1:
            continue
        used.append(day)
        with open(f, 'rb') as fh:
            for raw in fh:
                # Cheap prefilter on bytes; only the four line kinds are decoded at all.
                if b'Starting API request' in raw:
                    m = RE_START.match(raw.decode('utf-8', 'replace'))
                    if m:
                        starts.append((ms(m.group(1)), int(m.group(2))))
                elif b'Token Usage:' in raw:
                    m = RE_USAGE.match(raw.decode('utf-8', 'replace'))
                    if m:
                        usages.append((ms(m.group(1)), int(m.group(2))) + tuple(int(m.group(i)) for i in range(3, 7)))
                elif b'keep-warm ping' in raw:
                    m = RE_PING.match(raw.decode('utf-8', 'replace'))
                    if m:
                        pings.append((ms(m.group(1)),) + tuple(int(m.group(i)) for i in range(3, 7)))
                elif b'dispatching' in raw:
                    m = RE_DISPATCH.match(raw.decode('utf-8', 'replace'))
                    if m:
                        dispatches.append(ms(m.group(1)))
    return used, starts, usages, pings, dispatches


def history_counts(logs):
    """subagent-history layout: per parent dir, the number of sa_*.json files; per file, len(history)."""
    root = os.path.join(logs, 'subagent-history')
    per_parent, msgs = [], []
    if not os.path.isdir(root):
        return per_parent, msgs
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        fs = [f for f in os.listdir(p) if re.fullmatch(r'sa_\d+\.json', f)]
        per_parent.append(len(fs))
        for f in fs:
            try:
                with open(os.path.join(p, f), 'rb') as fh:
                    h = json.load(fh).get('history')
                msgs.append(len(h) if isinstance(h, list) else 0)
            except (OSError, ValueError, AttributeError):
                continue
    return per_parent, msgs


def build(starts, usages, pings, dispatches, anonymize=False):
    # Drop the keep-warm pings' own usage lines (same numbers, logged just before the ping line).
    ping_keys = {}
    for t, r, w, i, o in pings:
        ping_keys.setdefault((r, w, i, o), []).append(t)
    real = []
    for (t, th, i, o, r, c) in usages:
        ts = ping_keys.get((r, c, i, o))
        if ts and any(0 <= pt - t <= 2000 for pt in ts):
            continue
        real.append((t, th, i, o, r, c))
    # Pair each usage line with its request start.
    starts = sorted(starts)
    used = [False] * len(starts)
    by_thread = {}
    for k, (t, th) in enumerate(starts):
        by_thread.setdefault(th, []).append(k)
    reqs = []
    for (t, th, i, o, r, c) in sorted(real):
        k = None
        cand = by_thread.get(th, [])
        j = bisect.bisect_right([starts[x][0] for x in cand], t) - 1
        while j >= 0 and used[cand[j]]:
            j -= 1
        if j >= 0:
            k = cand[j]
        else:
            j = bisect.bisect_right(starts, (t, 1 << 62)) - 1
            while j >= 0 and used[j]:
                j -= 1
            k = j if j >= 0 else None
        if k is not None:
            used[k] = True
            arr = starts[k][0]
        else:
            arr = t
        reqs.append(dict(arr=arr, done=t, i=i, o=o, r=r, c=c))
    reqs.sort(key=lambda x: (x['arr'], x['done']))
    # Chains: expected next cache_read -> chain id, in completion order.
    order = sorted(range(len(reqs)), key=lambda k: reqs[k]['done'])
    open_ = {}   # expected cache_read -> (chain, last done)
    nchain = 0
    chained = 0
    bound = {}   # every cache boundary seen (r, r + c) -> the chain that had it
    forks = 0
    for k in order:
        q = reqs[k]
        hit = open_.pop(q['r'], None) if q['r'] > 0 else None
        q['fork'] = None
        if hit is not None and q['done'] - hit[1] <= CHAIN_WINDOW_MS:
            q['session'], q['cont'] = hit[0], True
            chained += 1
        else:
            q['session'], q['cont'] = nchain, False
            nchain += 1
            # A new chain whose cached prefix ends on another chain's boundary forked from it (a resume,
            # a compaction replay); otherwise its cached prefix is a shared pool (system prompt + tools).
            if q['r'] > 0 and q['r'] in bound:
                q['fork'] = bound[q['r']]
                forks += 1
        bound.setdefault(q['r'], q['session'])
        bound.setdefault(q['r'] + q['c'], q['session'])
        open_[q['r'] + q['c']] = (q['session'], q['done'])
    # Subagents: a new chain soon after a dispatch line.
    dispatches.sort()
    bursts = []   # (first t, size)
    for t in dispatches:
        if bursts and t - bursts[-1][2] <= 5000:
            bursts[-1][1] += 1
            bursts[-1][2] = t
        else:
            bursts.append([t, 1, t])
    first = {}
    for k, q in enumerate(reqs):
        first.setdefault(q['session'], k)
    disp_free = list(dispatches)
    arrivals = [q['arr'] for q in reqs]
    depth = {}
    parent = {}
    fan = {}
    for s, k in sorted(first.items(), key=lambda x: reqs[x[1]]['arr']):
        q = reqs[k]
        j = bisect.bisect_right(disp_free, q['arr']) - 1
        if j >= 0 and q['arr'] - disp_free[j] <= 60000:
            td = disp_free.pop(j)
            depth[s] = 1
            b = max((x for x in bursts if x[0] <= td), key=lambda x: x[0])
            fan[s] = b[1]
            pk = bisect.bisect_left(arrivals, td) - 1
            while pk >= 0 and reqs[pk]['session'] == s:
                pk -= 1
            parent[s] = reqs[pk]['session'] if pk >= 0 else None
        else:
            depth[s] = 0
    t0 = reqs[0]['arr'] if reqs else 0
    if anonymize:
        # Each local calendar day starts at t = 0 (its first request), days laid end to end 24 h apart:
        # no date and no time of day survives; gaps are recomputed from the shifted times.
        day0 = {}
        for q in reqs:
            d = dt.datetime.fromtimestamp(q['arr'] / 1000).date()
            day0.setdefault(d, q['arr'])
        order_days = {d: i for i, d in enumerate(sorted(day0))}
        for q in reqs:
            d = dt.datetime.fromtimestamp(q['arr'] / 1000).date()
            sh = order_days[d] * 86_400_000 + (q['arr'] - day0[d])
            q['done'] = sh + (q['done'] - q['arr'])
            q['arr'] = sh
        reqs.sort(key=lambda x: (x['arr'], x['done']))
        t0 = 0
    last_arr = {}
    out = []
    for q in reqs:
        s = q['session']
        shared = []
        if q['r'] > 0:
            shared.append({'with': 'session' if q['cont'] or q['fork'] is not None else 'sys',
                           'tokens': q['r']})
        out.append({
            't_ms': q['arr'] - t0,
            'session': s,
            'parent': parent.get(s),
            'depth': depth.get(s, 0),
            'prompt': q['i'] + q['r'] + q['c'],
            'shared': shared,
            'new': q['i'] + q['c'],
            'output': q['o'],
            'gap_prev_ms': (q['arr'] - last_arr[s]) if s in last_arr else None,
            'fanout': fan.get(s, 0),
            'fork': q['fork'],
            'claude_ms': q['done'] - q['arr'],
        })
        last_arr[s] = q['arr']
    stats = dict(usage_lines=len(usages), pings_dropped=len(usages) - len(real), requests=len(reqs),
                 chained=chained, chains=nchain, forks=forks, dispatches=len(dispatches), bursts=len(bursts),
                 starts_unpaired=used.count(False))
    return out, stats


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100 * len(v)))] if v else None


def summary(trace, stats, per_parent, msgs):
    P = [x['prompt'] for x in trace]
    rd = sum(s['tokens'] for x in trace for s in x['shared'])
    starts = [x for x in trace if x['gap_prev_ms'] is None]
    sys_hit = [x['shared'][0]['tokens'] if x['shared'] else 0 for x in starts]
    # The cross-session shared prefix a new subagent reads (system prompt + tools): its first cache_read.
    sub_sys = [x['shared'][0]['tokens'] if x['shared'] else 0 for x in starts if x['depth'] == 1]
    gaps = [x['gap_prev_ms'] for x in trace if x['gap_prev_ms'] is not None]
    # in flight: arrivals vs completions
    ev = sorted([(x['t_ms'], 1) for x in trace] + [(x['t_ms'] + x['claude_ms'], -1) for x in trace])
    cur, infl = 0, []
    for _, d in ev:
        cur += d
        if d > 0:
            infl.append(cur)
    lines = {
        **stats,
        'chained_share': round(stats['chained'] / max(1, stats['requests']), 4),
        'prompt_p50_p90_p99': [pct(P, 50), pct(P, 90), pct(P, 99)],
        'cached_share_of_prompt_tokens': round(rd / max(1, sum(P)), 4),
        'new_p50_p90_p99': [pct([x['new'] for x in trace], p) for p in (50, 90, 99)],
        'output_p50_p90_p99_mean': [pct([x['output'] for x in trace], p) for p in (50, 90, 99)] +
                                   [round(statistics.mean([x['output'] for x in trace]), 1) if trace else None],
        'gap_ms_p50_p90_p99': [pct(gaps, p) for p in (50, 90, 99)],
        'in_flight_p50_p90_p99_max': [pct(infl, 50), pct(infl, 90), pct(infl, 99), max(infl) if infl else None],
        'chain_starts': len(starts),
        'chain_start_sys_prefix_p10_p50_p90': [pct(sys_hit, p) for p in (10, 50, 90)],
        'chain_starts_with_sys_hit': sum(1 for v in sys_hit if v > 0),
        'subagent_start_sys_prefix_p25_p50_p75': [pct(sub_sys, p) for p in (25, 50, 75)],
        'subagent_chains': sum(1 for x in starts if x['depth'] == 1),
        'requests_per_chain_p50_p90': [pct(list(_per_chain(trace).values()), p) for p in (50, 90)],
        'history_parents': len(per_parent),
        'history_subagents': len(msgs),
        'history_fanout_p50_p90_max': [pct(per_parent, 50), pct(per_parent, 90), max(per_parent) if per_parent else None],
        'history_msgs_p50_p90': [pct(msgs, 50), pct(msgs, 90)],
    }
    return lines


def _per_chain(trace):
    c = {}
    for x in trace:
        c[x['session']] = c.get(x['session'], 0) + 1
    return c


def check(path):
    """Privacy check: every string VALUE anywhere in the file is <= 16 bytes, and `with` is only the enum."""
    bad = []

    def walk(o, where):
        if isinstance(o, dict):
            for k, v in o.items():   # keys are this script's own schema names
                walk(v, where + '.' + k)
        elif isinstance(o, list):
            for i, v in enumerate(o[:1000000]):
                walk(v, where)
        elif isinstance(o, str):
            if len(o.encode()) > 16:
                bad.append(f'{where}: string of {len(o.encode())} bytes')
            elif where.endswith('.with') and o not in ('sys', 'session', 'tools'):
                bad.append(f'{where}: unexpected value')
    with open(path) as f:
        walk(json.load(f), '$')
    for b in bad[:10]:
        print('PRIVACY CHECK FAIL', b)
    print(f'privacy check {path}: {"FAIL" if bad else "PASS"} ({len(bad)} findings)')
    return not bad


def synthetic(n_sessions=6, seed=7):
    """A small SYNTHETIC trace in the same schema (no log is read): a parent session that grows ~1k tokens a
    turn, a burst of subagents sharing a 6k-token pool, gaps from 2 s to 3 min. For tests and the repo example."""
    import random
    r = random.Random(seed)
    reqs = []
    t = 0
    pool = 6000
    sessions = []
    for s in range(n_sessions):
        sessions.append(dict(id=s, parent=None if s == 0 else 0, depth=0 if s == 0 else 1,
                             start=0 if s == 0 else 60_000 + 2_000 * s, prompt=pool + r.randrange(2_000, 9_000),
                             fanout=0 if s == 0 else n_sessions - 1))
    for sd in sessions:
        t = sd['start']
        prev = None
        for k in range(r.randrange(6, 14)):
            new = r.randrange(300, 3_000)
            out = r.randrange(50, 1_200)
            if prev is None:
                shared = [{'with': 'sys', 'tokens': pool}] if sd['depth'] == 1 else []
                prompt = sd['prompt']
            else:
                shared = [{'with': 'session', 'tokens': prev}]
                prompt = prev + new
            reqs.append({'t_ms': t, 'session': sd['id'], 'parent': sd['parent'], 'depth': sd['depth'],
                         'prompt': prompt, 'shared': shared, 'new': prompt - (shared[0]['tokens'] if shared else 0),
                         'output': out, 'gap_prev_ms': None if prev is None else gap, 'fanout': sd['fanout'],
                         'fork': None, 'claude_ms': r.randrange(2_000, 20_000)})
            prev = prompt
            gap = r.choice([2_000, 4_000, 8_000, 15_000, 45_000, 180_000])
            t += gap
    reqs.sort(key=lambda x: x['t_ms'])
    return {'version': 1, 'days': 0, 'tokenizer': 'synthetic', 'synthetic': True, 'summary': {},
            'requests': reqs}


def inside_git(path):
    d = os.path.dirname(os.path.abspath(path))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, '.git')):
            return d
        d = os.path.dirname(d)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', default=os.path.expanduser('~/.synaps-cli'))
    ap.add_argument('--from', dest='d0', default='2026-10-20')
    ap.add_argument('--to', dest='d1', default='2026-10-27')
    ap.add_argument('--out')
    ap.add_argument('--summary', action='store_true', help='print the structure summary (numbers only)')
    ap.add_argument('--check', metavar='FILE')
    ap.add_argument('--anonymize-time', action='store_true',
                    help='strip dates and times of day: each day starts at t = 0 (for anything shared)')
    ap.add_argument('--synthetic', metavar='FILE', help='write a small SYNTHETIC example trace (reads no logs)')
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a.check) else 1)
    if a.synthetic:
        with open(a.synthetic, 'w') as f:
            json.dump(synthetic(), f, indent=0, separators=(',', ':'))
        sys.exit(0 if check(a.synthetic) else 1)
    days, starts, usages, pings, dispatches = read_logs(a.logs, a.d0, a.d1)
    trace, stats = build(starts, usages, pings, dispatches, a.anonymize_time)
    per_parent, msgs = history_counts(a.logs)
    summ = summary(trace, stats, per_parent, msgs)
    tag = 'anon' if a.anonymize_time else f'{a.d0}-{a.d1}'
    out = a.out or os.path.expanduser(f'~/.cache/axon-agent/traces/synaps-{tag}.json')
    if inside_git(out):
        # The Axon repo is public: a trace from real logs never goes into a git tree (only --synthetic does).
        sys.exit(f'refusing to write a trace from real logs inside a git tree ({inside_git(out)}): '
                 'traces live in ~/.cache/axon-agent/traces')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = {'version': 1, 'days': len(days), 'tokenizer': 'claude', 'summary': summ, 'requests': trace}
    with open(out, 'w') as f:
        json.dump(doc, f, separators=(',', ':'))
    print(f'wrote {out}: {len(trace)} requests from {len(days)} log days')
    if a.summary:
        for k, v in summ.items():
            print(f'  {k}: {v}')
    if not check(out):
        os.unlink(out)
        sys.exit('privacy check failed: trace deleted')


if __name__ == '__main__':
    main()
