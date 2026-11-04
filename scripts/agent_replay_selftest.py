#!/usr/bin/env python3
"""Offline self-test of agent_replay.py: stdlib only, no engine, no model, no network, about a second (CI runs it).

- The --sessions ladder (plan / replicate), on a small synthetic trace with a pool, a fork and a primed session:
  N = S is the canonical cell byte for byte, each cell rendered with its OWN prompt closure; the session order
  interleaves primed and new sessions in proportion; N < S keeps the first N sessions of it (and only their primes);
  copy j of a session has its schedule and lengths, shares the pool prefix with everyone and a fork's parent prefix
  with the parent's copy j only, and its own text is new.
- board (cmd_board with fake engines): every cell runs on a process of its own, for every engine, and the cell order
  rotates by round; the restart cost is recorded.
- ollama_check on a synthetic store: a good manifest passes and records its config, layers and params; an adapter
  layer, a junk blob under the right name, a wrong size, a missing blob, other weights, a v2 manifest that disagrees
  with the v1 one, and a good v2 next to a v1 with an adapter or a digest-less layer are each refused.
- table's N_warm reading on synthetic run results (3 rounds: median not mean, <= at both thresholds, tainted runs
  left out, n/a without a re-prefill number, NON-MONOTONE), and legacy (no --sessions) runs keep their old heading.
- harness_provenance on a scratch git repo: clean, an edited file, an edit hidden by assume-unchanged (of another
  file and of the script), an untracked copy and a copy in a repo that tracks other bytes there; and on this tree.
Any failure is an assertion (rc 1); success prints SELFTEST agent_replay OK.
"""
import contextlib, hashlib, importlib.util, io, json, os, shutil, subprocess, sys, tempfile, time, types

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('agent_replay', os.path.join(HERE, 'agent_replay.py'))
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def refused(f, what):
    try:
        f()
    except SystemExit as e:
        assert what in str(e), (what, str(e))
        return str(e)
    raise AssertionError(f'not refused: {what}')


# ------------------------------------------------------------------------------------------------ --sessions
def R(t, s, prompt, out=40, pool=0, fork=None, gap=None, depth=0):
    return dict(t_ms=t, session=s, parent=None, depth=depth, prompt=prompt, new=prompt, output=out, gap_prev_ms=gap,
                fanout=None, fork=fork, shared=[dict(tokens=pool)] if pool else [])


trace = dict(version=1, requests=[
    R(0, 0, 1000, pool=400),                        # s0 starts before the window: primed
    R(10_000, 0, 1200, gap=10_000),
    R(100_000, 1, 900, pool=500, fork=0, depth=1),  # window start: s1 forks s0's first 500 tokens
    R(105_000, 0, 1500, gap=95_000),
    R(110_000, 2, 300),                             # no shared prefix
    R(120_000, 1, 1100, gap=20_000),
    R(130_000, 2, 400, gap=20_000),
    R(140_000, 3, 800, pool=400),                   # the same pool as s0
])
a = types.SimpleNamespace(scale=1.0, clip=32768, window_s=60, start_ms=100_000, seed=7, speed=2.0, nwords=500)
c0 = ar.plan(trace, a)
P = c0['prompt']
SCHED = ('at', 'gap', 'new', 'length', 'max_tokens', 'epoch', 'depth')
assert list(dict.fromkeys(r['chain'] for r in c0['requests'])) == [1, 0, 2, 3], 'first arrivals'
assert [c for c, _, _ in c0['primes']] == [0], c0['primes']
order = ar.session_order(c0)
assert order == [0, 1, 2, 3], order          # the one primed session first: 1 of 4, then 1 of 1 primed taken
fake = lambda P, Q: dict(requests=[dict(chain=c) for c in range(P + Q)], primes=[(c, 0, 1) for c in range(P)])
o = ar.session_order(fake(7, 13))                 # agent-32k's mix: 7 primed of 20
assert sorted(o) == list(range(20)) and [sum(c < 7 for c in o[:n]) for n in (1, 4, 8, 16, 20)] == [1, 2, 3, 6, 7], o
assert ar.session_order(fake(0, 3)) == [0, 1, 2] and ar.session_order(fake(3, 0)) == [0, 1, 2]

cS = ar.plan(trace, a, len(order))
strip = lambda c: [(r, c['prompt'](r['chain'], r['epoch'], r['length'])) for r in c['requests']]   # its OWN text
assert strip(cS) == strip(c0) and cS['primes'] == c0['primes'], 'N = S must be the canonical cell'
assert [cS['prompt'](*p) for p in cS['primes']] == [P(*p) for p in c0['primes']], 'N = S primes'

c2 = ar.plan(trace, a, 3)
assert {r['chain'] for r in c2['requests']} == {0, 1, 2} and [c for c, _, _ in c2['primes']] == [0], c2['primes']
c1 = ar.plan(trace, a, 2)
assert {r['chain'] for r in c1['requests']} == {0, 1} and [c for c, _, _ in c1['primes']] == [0]
cx = dict(c0, primes=[], requests=[r for r in c0['requests'] if r['chain'] != 0])   # no primed session at all
assert {r['chain'] for r in ar.replicate(cx, 1, 4, 8)['requests']} == {1}
assert ar.replicate(cx, 1, 4, 8)['primes'] == [] and len(ar.replicate(c0, 1, 4, 8)['primes']) == 1, 'primes follow'

N = 10
cN = ar.plan(trace, a, N)
PN = cN['prompt']
M = 1 + max(x['session'] for x in trace['requests'])
chains = {r['chain'] for r in cN['requests']}
assert len(chains) == N and len({r['k'] for r in cN['requests']}) == len(cN['requests']), 'ids must be unique'
copies = {}
for i in range(N):
    copies.setdefault(order[i % len(order)], []).append(i // len(order))
assert len(cN['requests']) == sum(len(copies[c]) * sum(r['chain'] == c for r in c0['requests']) for c in copies)
assert sorted(c for c, _, _ in cN['primes']) == [0 + j * M for j in copies[0]], cN['primes']
orig = {r['k']: r for r in c0['requests']}
for r in cN['requests']:
    c, j = r['chain'] % M, r['chain'] // M
    o = orig[r['k'] - j * len(trace['requests'])]
    assert o['chain'] == c and all(r[f] == o[f] for f in SCHED), (r, o)
    t, to = PN(r['chain'], r['epoch'], r['length']), P(o['chain'], o['epoch'], o['length'])
    assert len(t) == len(to) == r['length']
    shared = {0: 400, 3: 400, 1: 500, 2: 0}[c]
    if j == 0:
        assert t == to
    else:
        assert t[:min(shared, 400)] == to[:min(shared, 400)], 'the pool is shared with the original'
        assert t[shared:] != to[shared:], 'a copy has its own text'
        if c == 1:   # the fork: its parent's copy j, not the original parent
            assert t[:500] == PN(0 + j * M, 0, 500) and t[400:500] != P(0, 0, 500)[400:500]
assert PN(3 + M, 0, 400) == P(0, 0, 400) == PN(0 + 2 * M, 0, 400), 'one pool for all sessions and copies'
d = ar.cell_desc(a, cN)
assert d['n_sessions'] == N and d['sessions'] == N and d['speed'] == 2.0, d
print(f'SELFTEST sessions OK: S = {len(order)}; N = S canonical; stratified order (7:13 mix kept); subsets; N = {N}: '
      f'{len(cN["requests"])} requests, {len(cN["primes"])} primes, schedule kept, pool / fork / own text checked')

# ------------------------------------------------------------------------------------------------ board: an engine per cell
tb = tempfile.mkdtemp(prefix='agent-replay-selftest-board-')
saved = {k: getattr(ar, k) for k in ('start_engine', 'run_cell', 'stop_engine', 'word_list', 'kv_bytes_per_token',
                                     'gpu_name', 'install_guards')}
try:
    tf, fm = os.path.join(tb, 'trace.json'), os.path.join(tb, 'm.gguf')
    json.dump(trace, open(tf, 'w'))
    open(fm, 'wb').write(b'fake gguf')
    events, pids = [], iter(range(1000, 2000))

    class Proc:
        def __init__(self):
            self.pid = next(pids)

    def fake_start(kind, a_, kv, port, logdir, tokens):
        p = Proc()
        events.append(('start', kind, p.pid))
        return p, (kind, p.pid), 4096, f'fake {kind}'

    def fake_run(eng, words, cell, a_, kv_gb):
        events.append(('run', eng[0], eng[1], cell['speed'], cell['sessions']))
        return dict(requests=len(cell['requests']), errors=0, md5='m', tainted=False, ttft_turn_p50=0.1,
                    reprefill_share=0.01)
    ar.start_engine, ar.run_cell = fake_start, fake_run
    ar.stop_engine = lambda p, grace=30.0: events.append(('stop', None, p.pid))
    ar.word_list, ar.kv_bytes_per_token = lambda m, n=2048: [f'w{i}' for i in range(500)], lambda m: 1
    ar.gpu_name, ar.install_guards = lambda: None, lambda: None
    ba = types.SimpleNamespace(trace=tf, model=fm, out=os.path.join(tb, 'out'), speeds='1,4', sessions='2,4',
                               engines='axon,llama', kv='f16', rounds=3, round_offset=0, axon=fm, llama_server=fm,
                               ollama_models=None, kv_tokens=None, port=18600, scale=1.0, clip=32768, window_s=60,
                               start_ms=100_000, seed=7)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        ar.cmd_board(ba)
    log = log.getvalue()
    assert 'BOARD END ok' in log and 'ENGINE RESTARTS (one per cell per engine): axon 12 x start' in log, log[-800:]
    # start -> its one cell -> stop, every time: no process runs a second cell, no two are alive at once
    assert len(events) == 3 * 24 and all(
        events[i][0] == 'start' and events[i + 1][:3] == ('run', events[i][1], events[i][2])
        and events[i + 2] == ('stop', None, events[i][2]) for i in range(0, len(events), 3)), events
    runs = [e for e in events if e[0] == 'run']
    assert len({e[2] for e in runs}) == 24, 'a fresh process per cell'
    base = [(1.0, 2), (1.0, 4), (4.0, 2), (4.0, 4)]
    for rnd in range(3):   # per round: the cells rotated by the round, each on axon then llama
        got = [(e[1], e[3], e[4]) for e in runs[rnd * 8:(rnd + 1) * 8]]
        assert got == [(k, sp, n) for sp, n in base[rnd:] + base[:rnd] for k in ('axon', 'llama')], (rnd, got)
    m = json.load(open(os.path.join(ba.out, 'meta.json')))
    assert len(m['starts']) == 24 and all(x['start_s'] >= 0 and x['stop_s'] >= 0 for x in m['starts']), m['starts']
    rj = [json.load(open(os.path.join(ba.out, f))) for f in os.listdir(ba.out) if f.startswith('run-')]
    assert len(rj) == 24 and len({r['engine_pid'] for r in rj}) == 24 and all('engine_start_s' in r for r in rj)
finally:
    for k, v in saved.items():
        setattr(ar, k, v)
    shutil.rmtree(tb)
print('SELFTEST board OK: 2 speeds x 2 N x 2 engines x 3 rounds = 24 cells on 24 processes (start, one cell, stop), '
      'cell order rotated per round, restart cost recorded')

# ------------------------------------------------------------------------------------------------ ollama_check
tmp = tempfile.mkdtemp(prefix='agent-replay-selftest-')
try:
    def blob(store, data):
        h = 'sha256:' + hashlib.sha256(data).hexdigest()
        os.makedirs(os.path.join(store, 'blobs'), exist_ok=True)
        open(os.path.join(store, 'blobs', h.replace(':', '-')), 'wb').write(data)
        return dict(digest=h, size=len(data))

    model = os.path.join(tmp, 'm.gguf')
    open(model, 'wb').write(b'GGUF synthetic weights ' * 64)
    msha = ar.file_sha256(model)
    T = 'application/vnd.ollama.image.'

    def store(name, extra=(), v2=None):
        s = os.path.join(tmp, name)
        layers = [dict(blob(s, open(model, 'rb').read()), mediaType=T + 'model'),
                  dict(blob(s, b'{{ .Prompt }}'), mediaType=T + 'template'),
                  dict(blob(s, b'{"temperature":0}'), mediaType=T + 'params')] + list(extra)
        man = dict(schemaVersion=2, config=dict(blob(s, b'{"model_format":"gguf"}'), mediaType='config'), layers=layers)
        p = os.path.join(s, 'manifests', 'registry.ollama.ai', 'library', 'tm', 'latest')
        os.makedirs(os.path.dirname(p))
        json.dump(man, open(p, 'w'))
        if v2 is not None:
            p2 = os.path.join(s, 'manifests-v2', 'ollama.com', 'library', 'tm', 'latest')
            os.makedirs(os.path.dirname(p2))
            json.dump(v2(man), open(p2, 'w'))
        return s, man, p

    s, man, _ = store('good')
    ok = ar.ollama_check(s, 'tm', msha)
    assert [x['type'] for x in ok['layers']] == ['model', 'template', 'params'] and ok['params'] == {'temperature': 0}
    assert ok['weights'] == 'sha256:' + msha and ok['blobs'] == 4 and ok['config'] == {
        k: man['config'][k] for k in ('digest', 'size')} and ok['others'] == [], ok
    lic = store('license', [dict(blob(os.path.join(tmp, 'license'), b'MIT'), mediaType=T + 'license')])[0]
    assert [x['type'] for x in ar.ollama_check(lic, 'tm', msha)['layers']][-1] == 'license'
    adapter = store('adapter', [dict(blob(os.path.join(tmp, 'adapter'), b'lora'), mediaType=T + 'adapter')])[0]
    refused(lambda: ar.ollama_check(adapter, 'tm', msha), 'not run --model alone')
    s, man, _ = store('junk')
    tpl = os.path.join(s, 'blobs', man['layers'][1]['digest'].replace(':', '-'))
    open(tpl, 'wb').write(b'X' * man['layers'][1]['size'])
    refused(lambda: ar.ollama_check(s, 'tm', msha), 'hashes to')
    s, man, p = store('size')
    man['layers'][2]['size'] += 1
    json.dump(man, open(p, 'w'))
    refused(lambda: ar.ollama_check(s, 'tm', msha), 'its manifest says')
    s, man, _ = store('missing')
    os.remove(os.path.join(s, 'blobs', man['layers'][2]['digest'].replace(':', '-')))
    refused(lambda: ar.ollama_check(s, 'tm', msha), 'blobs missing')
    refused(lambda: ar.ollama_check(store('other')[0], 'tm', '0' * 64), 'not --model')
    other = 'sha256:' + '1' * 64
    v2 = store('v2', v2=lambda m: dict(m, layers=[dict(m['layers'][0], digest=other)] + m['layers'][1:]))[0]
    refused(lambda: ar.ollama_check(v2, 'tm', msha), 'names')
    refused(lambda: ar.ollama_check(s, 'nosuch', msha), 'no manifest')
    # every manifest the name resolves to is checked in full, not just the first (v2) one
    first3 = lambda m: dict(m, layers=m['layers'][:3])
    both = store('both', v2=first3)[0]
    assert ar.ollama_check(both, 'tm', msha)['others'] == ['manifests/registry.ollama.ai/library/tm/latest']
    sec = store('sec', [dict(blob(os.path.join(tmp, 'sec'), b'lora'), mediaType=T + 'adapter')], v2=first3)[0]
    refused(lambda: ar.ollama_check(sec, 'tm', msha), 'not run --model alone')
    nod = store('nodigest', [dict(mediaType=T + 'license', size=3)], v2=first3)[0]
    refused(lambda: ar.ollama_check(nod, 'tm', msha), 'no sha256 digest')
    print('SELFTEST ollama_check OK: good + license pass (config, layers, params recorded); adapter, junk blob, size, '
          'missing blob, other weights, v1/v2 disagreement, no manifest refused; a good v2 next to a v1 with an '
          'adapter or a digest-less layer refused')

    # -------------------------------------------------------------------------------------------- table / N_warm
    bd = os.path.join(tmp, 'board')
    os.makedirs(bd)

    def run(eng, n, rnd, turn, rp, tainted=False):
        json.dump(dict(engine=eng, kv='q8', speed=1.0, sessions=n, round=rnd, md5='x', tainted=tainted,
                       ttft_turn_p50=turn, reprefill_share=rp, prompt_checked=1),
                  open(os.path.join(bd, f'run-{eng}-q8-s1-n{n}-r{rnd}.json'), 'w'))
    # axon: N = 16 sits exactly on both thresholds as a median of 3 rounds (the means miss: 1.3 s, 25 %); N = 24
    # misses on its clean rounds but would pass with its 4 tainted ones; N = 32 misses.
    for rnd, (t16, r16) in enumerate([(0.40, 0.05), (0.50, 0.10), (3.0, 0.60)]):
        run('axon', 8, rnd, 0.20 + rnd * 0.01, 0.02)
        run('axon', 16, rnd, t16, r16)
        run('axon', 24, rnd, 0.90 + rnd * 0.02, 0.31)
        run('axon', 32, rnd, 3.1, 0.55)
    for rnd in (3, 4, 5, 6):
        run('axon', 24, rnd, 0.1, 0.01, tainted=True)
    for rnd in (0, 1):
        run('llama', 8, rnd, 0.7, 0.02)    # misses below a passing N: NON-MONOTONE
        run('llama', 16, rnd, 0.3, 0.05)
        run('ollama', 8, rnd, 0.3, 0.9)    # table() drops ollama's re-prefill: n/a
    json.dump(dict(engine='axon', kv='q8', speed=1.0, round=0, md5='y', tainted=False, ttft_turn_p50=0.1,
                   reprefill_share=0.01), open(os.path.join(bd, 'run-axon-q8-s1-r0.json'), 'w'))
    t = ar.table([bd])
    assert '**speed 1** (median' in t and '**speed 1, 16 sessions** (median' in t, t
    ln = {e: [x for x in t.splitlines() if x.startswith(f'- speed 1, {e} q8: N_warm = ')] for e in ('axon', 'llama', 'ollama')}
    assert all(len(v) == 1 for v in ln.values()), ln
    ax, ll, ol = ln['axon'][0], ln['llama'][0], ln['ollama'][0]
    assert ax.startswith('- speed 1, axon q8: N_warm = 16 (8: turn 0.21 s, re-prefill 2.0% ok (n=3); '), ax
    assert '16: turn 0.5 s, re-prefill 10.0% ok (n=3)' in ax and '24: turn 0.92 s, re-prefill 31.0% MISS (n=3)' in ax, ax
    assert ax.endswith('[4 tainted run(s) left out]') and 'NON-MONOTONE' not in ax, ax
    assert ll.startswith('- speed 1, llama q8: N_warm = 16 NON-MONOTONE (8 miss) ('), ll
    assert ol.startswith('- speed 1, ollama q8: N_warm = n/a (no re-prefill number) (8: turn 0.3 s, re-prefill n/a (n=2))'), ol
    assert 'none' not in ol and 'MISS' not in ol, ol
    print('SELFTEST table OK: ladder headings; N_warm = 16 on the synthetic ladder (median of 3 at <= on both '
          'thresholds, 4 tainted runs left out), NON-MONOTONE flagged, ollama n/a; legacy heading kept')
finally:
    shutil.rmtree(tmp)

# ------------------------------------------------------------------------------------------------ provenance
SRC = os.path.join(HERE, 'agent_replay.py')
if shutil.which('git'):
    tg = tempfile.mkdtemp(prefix='agent-replay-selftest-git-')
    ceiling = os.environ.get('GIT_CEILING_DIRECTORIES')
    os.environ['GIT_CEILING_DIRECTORIES'] = tg    # never find a repo above the scratch dir
    try:
        def git(repo, *x):
            return subprocess.run(['git', '-C', repo, '-c', 'user.name=selftest', '-c', 'user.email=selftest@invalid',
                                   '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', *x],
                                  check=True, capture_output=True, text=True).stdout.strip()

        def repo(name, files):
            r = os.path.join(tg, name)
            os.makedirs(os.path.join(r, 'scripts'))
            git(r, 'init', '-q')
            for f, data in files.items():
                open(os.path.join(r, f), 'wb').write(data)
                os.utime(os.path.join(r, f), (time.time() - 3600,) * 2)   # not racily clean: git trusts its stat
            git(r, 'add', '-A')
            git(r, 'commit', '-q', '-m', 'selftest')
            return r, git(r, 'rev-parse', 'HEAD')
        code = open(SRC, 'rb').read()
        r, head = repo('a', {'scripts/agent_replay.py': code, 'other.txt': b'one\n'})
        mine = os.path.join(r, 'scripts', 'agent_replay.py')
        prov = lambda p=mine: (lambda h: (h['commit'], h['dirty']))(ar.harness_provenance(p))
        assert prov() == (head, []), prov()
        open(os.path.join(r, 'other.txt'), 'wb').write(b'two\n')
        assert prov() == (head, ['other.txt']), ('an edited file', prov())
        git(r, 'update-index', '--assume-unchanged', 'other.txt')
        assert git(r, 'status', '--porcelain', '--untracked-files=no') == '', 'git hides it'
        assert prov() == (head, ['other.txt']), ('an edit hidden by assume-unchanged', prov())
        git(r, 'update-index', '--no-assume-unchanged', 'other.txt')
        git(r, 'checkout', '-q', '--', 'other.txt')
        assert prov() == (head, [])
        # the script itself, same size, same mtime, ctime not trusted: git's stat check can't see it
        git(r, 'config', 'core.trustctime', 'false')
        st = os.stat(mine)
        with open(mine, 'r+b') as f:
            f.seek(len(code) - 2)
            f.write(b'#' if code[-2:-1] != b'#' else b' ')
        os.utime(mine, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert git(r, 'status', '--porcelain', '--untracked-files=no') == '', 'git misses it'
        assert prov() == (head, ['scripts/agent_replay.py']), ('the script differs from HEAD', prov())
        git(r, 'update-index', '--assume-unchanged', 'scripts/agent_replay.py')
        assert prov() == (head, ['scripts/agent_replay.py']), ('and is marked assume-unchanged', prov())
        cp = os.path.join(r, 'scripts', 'copy.py')   # an untracked copy in a tree: no commit to name
        shutil.copy(SRC, cp)
        assert prov(cp) == (None, None), prov(cp)
        r2, head2 = repo('b', {'scripts/agent_replay.py': b'print(1)\n'})   # a repo that tracks other bytes there
        shutil.copy(SRC, os.path.join(r2, 'scripts', 'agent_replay.py'))
        assert prov(os.path.join(r2, 'scripts', 'agent_replay.py')) == (head2, ['scripts/agent_replay.py'])
        plain = os.path.join(tg, 'plain')   # no repo at all
        os.makedirs(plain)
        shutil.copy(SRC, plain)
        assert prov(os.path.join(plain, 'agent_replay.py')) == (None, None)
        h = ar.harness_provenance(cp)
        assert h['sha256'] == ar.file_sha256(SRC) and h['path'] == os.path.realpath(cp), h
    finally:
        if ceiling is None:
            os.environ.pop('GIT_CEILING_DIRECTORIES', None)
        else:
            os.environ['GIT_CEILING_DIRECTORIES'] = ceiling
        shutil.rmtree(tg)
    print('SELFTEST provenance OK (scratch repo): clean; edited file; assume-unchanged edit; script edited behind '
          "git's stat check; untracked copy -> None; copy in a repo tracking other bytes -> dirty; no repo -> None")
else:
    print('SELFTEST provenance: scratch-repo cases SKIPPED (no git)')
h = ar.harness_provenance()
assert h['sha256'] == ar.file_sha256(SRC), h
if shutil.which('git') and os.path.exists(os.path.join(HERE, '..', '.git')):
    assert h['commit'] and len(h['commit']) == 40 and isinstance(h['dirty'], list), h
print(f'SELFTEST provenance (this tree): commit {h["commit"]}, dirty {h["dirty"]}, sha256 {h["sha256"][:16]}')
print('SELFTEST agent_replay OK')
