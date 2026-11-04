#!/usr/bin/env python3
"""Offline self-test of agent_replay.py: stdlib only, no engine, no model, no network, about a second (CI runs it).

- The --sessions ladder (plan / replicate), on a small synthetic trace with a pool, a fork and a primed session:
  N = S is the canonical cell byte for byte; the session order interleaves primed and new sessions in proportion;
  N < S keeps the first N sessions of it (and only their primes); copy j of a
  session has its schedule and lengths, shares the pool prefix with everyone and a fork's parent prefix with the
  parent's copy j only, and its own text is new.
- ollama_check on a synthetic store: a good manifest passes and records its layers and params; an adapter layer, a
  junk blob under the right name, a wrong size, a missing blob, other weights and a v2 manifest that disagrees with
  the v1 one are each refused.
- table's N_warm reading on synthetic run results, and legacy (no --sessions) runs keep their old heading.
- harness_provenance names this tree's commit, its dirty files and this copy's sha256.
Any failure is an assertion (rc 1); success prints SELFTEST agent_replay OK.
"""
import hashlib, importlib.util, json, os, shutil, sys, tempfile, types

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
strip = lambda c: [(r, P(r['chain'], r['epoch'], r['length'])) for r in c['requests']]
assert strip(cS) == strip(c0) and cS['primes'] == c0['primes'], 'N = S must be the canonical cell'
assert [P(*p) for p in cS['primes']] == [P(*p) for p in c0['primes']]

c2 = ar.plan(trace, a, 3)
assert {r['chain'] for r in c2['requests']} == {0, 1, 2} and [c for c, _, _ in c2['primes']] == [0], c2['primes']
c1 = ar.plan(trace, a, 2)
assert {r['chain'] for r in c1['requests']} == {0, 1} and [c for c, _, _ in c1['primes']] == [0]
cx = dict(c0, primes=[], requests=[r for r in c0['requests'] if r['chain'] != 0])   # no primed session at all
assert {r['chain'] for r in ar.replicate(cx, 1, 4, 8)['requests']} == {1}
assert ar.replicate(cx, 1, 4, 8)['primes'] == [] and len(ar.replicate(c0, 1, 4, 8)['primes']) == 1, 'primes follow'

N = 10
cN = ar.plan(trace, a, N)
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
    t, to = P(r['chain'], r['epoch'], r['length']), P(o['chain'], o['epoch'], o['length'])
    assert len(t) == len(to) == r['length']
    shared = {0: 400, 3: 400, 1: 500, 2: 0}[c]
    if j == 0:
        assert t == to
    else:
        assert t[:min(shared, 400)] == to[:min(shared, 400)], 'the pool is shared with the original'
        assert t[shared:] != to[shared:], 'a copy has its own text'
        if c == 1:   # the fork: its parent's copy j, not the original parent
            assert t[:500] == P(0 + j * M, 0, 500) and t[400:500] != P(0, 0, 500)[400:500]
assert P(3 + M, 0, 400) == P(0, 0, 400) == P(0 + 2 * M, 0, 400), 'one pool for all sessions and copies'
d = ar.cell_desc(a, cN)
assert d['n_sessions'] == N and d['sessions'] == N and d['speed'] == 2.0, d
print(f'SELFTEST sessions OK: S = {len(order)}; N = S canonical; stratified order (7:13 mix kept); subsets; N = {N}: '
      f'{len(cN["requests"])} requests, {len(cN["primes"])} primes, schedule kept, pool / fork / own text checked')

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
    assert ok['weights'] == 'sha256:' + msha and ok['blobs'] == 4, ok
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
    print('SELFTEST ollama_check OK: good + license pass (layers, params recorded); adapter, junk blob, size, '
          'missing blob, other weights, v1/v2 disagreement, no manifest refused')

    # -------------------------------------------------------------------------------------------- table / N_warm
    bd = os.path.join(tmp, 'board')
    os.makedirs(bd)
    runs = {8: (0.21, 0.02), 16: (0.44, 0.08), 24: (0.92, 0.31), 32: (3.1, 0.55)}
    for n, (turn, rp) in runs.items():
        for rnd in (0, 1):
            json.dump(dict(engine='axon', kv='q8', speed=1.0, sessions=n, round=rnd, md5='x', tainted=False,
                           ttft_turn_p50=turn + rnd * 0.01, reprefill_share=rp, prompt_checked=1),
                      open(os.path.join(bd, f'run-axon-q8-s1-n{n}-r{rnd}.json'), 'w'))
    json.dump(dict(engine='axon', kv='q8', speed=1.0, round=0, md5='y', tainted=False, ttft_turn_p50=0.1,
                   reprefill_share=0.01), open(os.path.join(bd, 'run-axon-q8-s1-r0.json'), 'w'))
    t = ar.table([bd])
    assert '**speed 1** (median' in t and '**speed 1, 16 sessions** (median' in t, t
    assert '- speed 1, axon q8: N_warm = 16 (8: turn 0.215 s, re-prefill 2.0% ok;' in t, t
    assert '24: turn 0.925 s, re-prefill 31.0% MISS' in t, t
    print('SELFTEST table OK: ladder headings, N_warm = 16 on the synthetic ladder, legacy heading kept')
finally:
    shutil.rmtree(tmp)

# ------------------------------------------------------------------------------------------------ provenance
h = ar.harness_provenance()
assert h['sha256'] == ar.file_sha256(os.path.join(HERE, 'agent_replay.py')), h
if shutil.which('git') and os.path.exists(os.path.join(HERE, '..', '.git')):
    assert h['commit'] and len(h['commit']) == 40 and isinstance(h['dirty'], list), h
print(f'SELFTEST provenance OK: commit {h["commit"]}, dirty {h["dirty"]}, sha256 {h["sha256"][:16]}')
print('SELFTEST agent_replay OK')
