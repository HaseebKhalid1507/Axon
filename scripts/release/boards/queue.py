#!/usr/bin/env python3
"""queue.py [--with-smoke] [--resume] | queue.py --selftest: release 0.1's boards on bella, one cell per axon-run job.

Why (11-15, jobs 1115-032610-223e, -bbd2, 1115-032613-4ff7; docs/log/2026-11-14-release-0.1-prep.md): the sweep ran its
8 cells (~650 s each: a 600-s window, ~40 s of priming, engine start and stop) and the reruns of its tainted arms in
ONE --timing job bounded at 120 min. 8 cells are 87 min; 6 of the 7 arms were tainted, so 14 cells (152 min) were due
and the bound stopped it in its 12th. bella is a desktop: its foreign GPU load passes the start check (<= 10 % for
120 s) and still tainted 8 of 11 runs (foreign max 28-51 % > 25 %, foreign mean 1.4-3.9 %: Xorg, every 10-20 min).
So every cell is its own --timing job (it waits for a quiet desktop, under a bound that fits one cell), and a run whose
ONE fault is taint is run again as a new job, a bounded number of times:

  1. Single stream (single_stream.sh K): one job; run again while axon-run marks the job tainted, at most 4 jobs.
     The first untainted job's sb_read.py reading is the board's (sb/READING.txt says which).
  2. Sweep (agent_sweep.sh CELL K): axon's reference cell (it sets the KV budget; its taint doesn't matter), A0 .. A6,
     then a pass over the arms whose run's one fault is taint (pick.taint_only), then a third: at most 3 runs per arm.
  3. pick.py: the winner (an arm's first untainted run counts). No winner: STOP, the board isn't run.
  4. Board (agent_board.sh R): rounds 0, 1, 2, ..., one job each, until 3 rounds have run that weren't lost to taint
     alone, at most 8 rounds. A round lost to taint alone (each speed-1 run that isn't clean has taint as its one
     fault) doesn't count toward the 3; the next round number replaces it (the engine order alternates by round
     number). read_board.py reads every round (R over the rounds where both engines are clean).
A rerun depends on taint only: never on a run's numbers, its errors or its verdict (a run with any other fault keeps
it). A job that fails (rc != 0: a script's STOP, the --max-min bound, an OOM, a kill) stops the queue; nothing is
retried blind. Order: single stream first (one job, the release notes' table), then the sweep and the board.

--with-smoke: first the smoke (AXON_REL_SMOKE=1: the 0.5B, 60-s windows, 1 round, speed 1, next to other GPU jobs, no
reruns, outputs under rel01-smoke; it writes the PASSED markers the real cells check), then the real run.
--resume: continue after this process died: the jobs already submitted (DIR/queue-jobs.jsonl) are waited for, never
submitted again. At the end (done or STOP) it writes DIR/QUEUE-RESULT.txt, submits a CPU job that prints it
(rc 1 on STOP) and arms `axon-gatewatch dist-9f3630c-boards` on that job: one event for the whole queue.
Run on bella, detached: setsid nohup scripts/release/boards/queue.py --with-smoke > ~/.cache/axon-dist/queue.log 2>&1 &
--selftest: the queue's decisions on a fake axon-run (no GPU, no engine, no jobs)."""
import glob, json, os, re, shutil, subprocess, sys, tempfile, time

S = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(os.path.dirname(S))
sys.path.insert(0, S)
from pick import ARMS, attempt, pick, run as sweep_run, taint_only, why_not  # noqa: E402,F401
import read_board  # noqa: E402

K_SB, K_ARM, ROUNDS, MAX_ROUNDS = 4, 3, 3, 8
CELLS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
NAMES = {'A0': 'A0-4413a19-defaults', 'A1': 'A1-b12387-defaults', 'A2': 'A2-noidle', 'A3': 'A3-noidle-ub2048',
         'A4': 'A4-2x-slots', 'A5': 'A5-noidle-cram16g', 'A6': 'A6-cram16g'}
SMOKE_ENV = {'AXON_REL_SMOKE': '1', 'AXON_REL_SPEEDS': '1', 'AXON_REL_SB_MODELS': '0.5B', 'AXON_REL_SB_ROUNDS': '1'}
TAG = 'dist-9f3630c-rel01'
LABEL = 'dist-9f3630c-boards'


class Stop(Exception):
    pass


def out_dir(smoke):
    return os.path.expanduser('~/.cache/axon-dist/rel01' + ('-smoke' if smoke else ''))


def spec(smoke, phase, cell, k):
    """One job: (key, name, argv, timing, vram, ram, max_min, expect)."""
    sm = '-smoke' if smoke else ''
    b = os.path.join('scripts', 'release', 'boards')
    if phase == 'sb':
        return dict(key=f'sb-t{k}', name=f'{TAG}{sm}-sb-t{k}', argv=[os.path.join(b, 'single_stream.sh'), str(k)],
                    timing=not smoke, vram=2500 if smoke else 11000, ram=5000 if smoke else 8000,
                    max_min=15 if smoke else 20, expect='SB END ok')
    if phase == 'sweep':
        return dict(key=f'sweep-{cell}-t{k}', name=f'{TAG}{sm}-sweep-{cell}-t{k}',
                    argv=[os.path.join(b, 'agent_sweep.sh'), cell, str(k)], timing=not smoke,
                    vram=3000 if smoke else 11000, ram=6000 if smoke else 20000, max_min=10 if smoke else 20,
                    expect='CELL END ok')
    return dict(key=f'board-r{cell}', name=f'{TAG}{sm}-board-r{cell}', argv=[os.path.join(b, 'agent_board.sh'), str(cell)],
                timing=not smoke, vram=3000 if smoke else 11000, ram=6000 if smoke else 20000,
                max_min=15 if smoke else 90, expect='ROUND END ok')


def round_state(board, r, smoke=False):
    """'counted' | 'taint' (lost to taint alone: replaced) | 'lost: why' (counts toward the 3, not replaced)."""
    rs = read_board.runs(board, '1').get(r, {})
    bad = {e: why_not(rs.get(e), smoke) for e in read_board.ENG}
    bad = {e: w for e, w in bad.items() if w}
    if not bad:
        return 'counted'
    if all(taint_only(rs.get(e)) for e in bad):
        return 'taint'
    return 'lost: ' + '; '.join(f'{e}: {w}' for e, w in bad.items())


def queue(R, D, smoke, say):
    """One queue (smoke or real) into D; returns the summary lines. Raises Stop."""
    lines = [f'== {"smoke" if smoke else "real"} run, {D}']
    def job(phase, cell, k):
        sp = spec(smoke, phase, cell, k)
        j = R.run(sp)
        say(f'{sp["name"]}: job {j["id"]} rc {j["rc"]}' + (f' TAINTED ({j["tainted"]})' if j.get('tainted') else ''))
        if j['rc'] != 0:
            raise Stop(f'{sp["name"]}: job {j["id"]} rc {j["rc"]} (see its log); the queue stops here')
        return j
    # 1. single stream
    sb = None
    tries = []
    for k in range(1, (1 if smoke else K_SB) + 1):
        j = job('sb', None, k)
        tries.append(f't{k} job {j["id"]}' + (' tainted' if j.get('tainted') else ''))
        if smoke or not j.get('tainted'):
            sb = k
            break
    os.makedirs(os.path.join(D, 'sb'), exist_ok=True)
    rd = os.path.join(D, 'sb', 'READING.txt')
    if sb:
        src = os.path.join(D, 'sb', f't{sb}', 'READING.txt')
        body = open(src).read() if os.path.exists(src) else '(no READING.txt)\n'
        open(rd, 'w').write(f'single stream: the run read is t{sb} (jobs: {", ".join(tries)})\n' + body)
        counts = [x for x in body.splitlines() if x.startswith('COUNTS')]
        lines.append(f'single stream: t{sb} read ({", ".join(tries)}); {counts[0] if counts else "no COUNTS line"}')
    else:
        open(rd, 'w').write(f'single stream: NO UNTAINTED RUN in {K_SB} jobs ({", ".join(tries)}): nothing is read\n')
        lines.append(f'single stream: NO UNTAINTED RUN in {K_SB} jobs ({", ".join(tries)})')
    # 2. sweep
    sw = os.path.join(D, 'sweep')
    job('sweep', 'axon', 1)
    pending, log = list(CELLS), {c: [] for c in CELLS}
    for k in range(1, (1 if smoke else K_ARM) + 1):
        for c in pending:
            job('sweep', c, k)
            r = sweep_run(os.path.join(sw, NAMES[c] + ('' if k == 1 else f'-r{k}')))
            log[c].append('tainted' if r and r.get('tainted') else 'untainted' if r else 'no run')
        pending = [] if smoke else [c for c in pending
                                    if taint_only(sweep_run(os.path.join(sw, NAMES[c] + ('' if k == 1 else f'-r{k}'))))]
        if pending:
            say(f'sweep pass {k}: taint alone on {", ".join(pending)}: run again')
        if not pending:
            break
    lines.append('sweep runs: ' + '; '.join(f'{c} {"/".join(v)}' for c, v in log.items()))
    table = []
    win = pick(sw, smoke, say=table.append)
    open(os.path.join(sw, 'WINNER.txt'), 'w').write('\n'.join(table) + '\n')
    lines.append([x for x in table if x.startswith('WINNER')][0] if win else 'WINNER none')
    if not win:
        raise Stop('no sweep arm qualifies (sweep/WINNER.txt): the board is not run')
    # 3. board
    bd = os.path.join(D, 'board')
    states, r, done = [], 0, 0
    want, cap = (1, 1) if smoke else (ROUNDS, MAX_ROUNDS)
    while done < want and r < cap:
        job('board', r, 1)
        st = round_state(bd, r, smoke)
        states.append(f'r{r} {st}')
        say(f'board round {r}: {st}')
        if st != 'taint':
            done += 1
        r += 1
    pool = int(open(os.path.join(sw, 'POOL')).read())
    reading = []
    for sp_, rd_ in (('1', True), ('4', False)):
        if read_board.runs(bd, sp_):
            reading += read_board.reading(bd, sp_, pool, rd_)[2] + ['']
    open(os.path.join(bd, 'READING.txt'), 'w').write('\n'.join(reading) + '\n')
    lines.append(f'board: {r} round(s) ({", ".join(states)}); {done} not lost to taint alone (want {want}, cap {cap})')
    lines += [x for x in reading if x.startswith('VERDICT')]
    return lines


def archive(D, say):
    """Move the last run's outputs aside (never deleted): DIR/old-<time>/."""
    old = [n for n in ('sweep', 'board', 'sb', 'sbnode', 'queue-jobs.jsonl', 'QUEUE-RESULT.txt', 'QUEUE-SMOKE-OK')
           + tuple(os.path.basename(p) for p in glob.glob(os.path.join(D, 'PASSED-*'))) if os.path.exists(os.path.join(D, n))]
    if old:
        a = os.path.join(D, 'old-' + time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(a)
        for n in old:
            shutil.move(os.path.join(D, n), os.path.join(a, n))
        say(f'archived {", ".join(old)} -> {a}')


class AxonRun:
    """Submits a job with axon-run and waits for it; a job already in DIR/queue-jobs.jsonl is never submitted again."""
    WAITING = ('queued', 'running', 'paused', 'starting', 'placed')

    def __init__(self, D, env, say, poll=60):
        self.D, self.env, self.say, self.poll = D, env, say, poll
        self.bin = shutil.which('axon-run') or os.path.expanduser('~/.local/bin/axon-run')
        self.state_f = os.path.join(D, 'queue-jobs.jsonl')
        self.state = {}
        if os.path.exists(self.state_f):
            for ln in open(self.state_f):
                x = json.loads(ln)
                self.state[x['key']] = x['id']

    def run(self, sp):
        jid = self.state.get(sp['key']) or self.submit(sp)
        return self.wait(jid)

    def submit(self, sp):
        cmd = [self.bin, '--detach'] + (['--timing', 'bella'] if sp['timing'] else []) + [
            '--vram', str(sp['vram']), '--ram', str(sp['ram']), '--max-min', str(sp['max_min']),
            '--name', sp['name'], '--expect', sp['expect'], '--'] + sp['argv']
        if sp.get('cpu'):
            cmd = [self.bin, '--detach', '--cpu', '--ram', str(sp['ram']), '--max-min', str(sp['max_min']),
                   '--name', sp['name'], '--expect', sp['expect'], '--'] + sp['argv']
        p = subprocess.run(cmd, cwd=W, env=self.env, capture_output=True, text=True, timeout=300)
        m = re.search(r'job (\d{4}-\d{6}-[0-9a-f]{4})', p.stdout + p.stderr)
        if not m:
            raise Stop(f'submit {sp["name"]}: rc {p.returncode}: {(p.stdout + p.stderr)[-400:]}')
        jid = m.group(1)
        with open(self.state_f, 'a') as f:
            f.write(json.dumps(dict(key=sp['key'], id=jid, name=sp['name'], at=time.strftime('%F %T'))) + '\n')
        self.state[sp['key']] = jid
        self.say(f'submitted {sp["name"]}: {jid}')
        return jid

    def wait(self, jid):
        bad_since = None
        while True:
            d = None
            try:
                p = subprocess.run([self.bin, 'status', jid], capture_output=True, text=True, timeout=180)
                t = p.stdout
                i = t.find('--- log tail')
                d = json.loads(t[:i] if i >= 0 else t)
            except (subprocess.SubprocessError, ValueError, OSError):
                d = None
            if d is None:
                bad_since = bad_since or time.time()
                if time.time() - bad_since > 3600:
                    raise Stop(f'axon-run status {jid}: unreadable for an hour')
            else:
                bad_since = None
                if d.get('rc') is not None and d.get('state') not in self.WAITING:
                    return dict(id=jid, rc=d['rc'], tainted=d.get('tainted'), state=d.get('state'))
            time.sleep(self.poll)


def say_to(f):
    def say(msg):
        print(f'{time.strftime("%F %T")} {msg}', flush=True)
        if f:
            with open(f, 'a') as fh:
                fh.write(f'{time.strftime("%F %T")} {msg}\n')
    return say


def main():
    if '--selftest' in sys.argv:
        selftest()
        return
    resume, with_smoke = '--resume' in sys.argv, '--with-smoke' in sys.argv
    base = {k: v for k, v in os.environ.items() if not k.startswith('AXON_REL_')}
    base.setdefault('AXON_FOREMAN', 'axon')
    lines, ok, last_D = [], False, out_dir(False)
    say = say_to(None)
    try:
        for smoke in ([True, False] if with_smoke else [False]):
            D = out_dir(smoke)
            os.makedirs(D, exist_ok=True)
            say = say_to(os.path.join(D, 'queue.log'))
            last_D = D
            if not (resume and os.path.exists(os.path.join(D, 'queue-jobs.jsonl'))):
                archive(D, say)
            env = dict(base, **SMOKE_ENV, AXON_REL_AXON=os.path.join(out_dir(False), 'rc', 'axon')) if smoke else base
            if smoke and resume and os.path.exists(os.path.join(D, 'QUEUE-SMOKE-OK')):
                say('smoke already passed in this queue (--resume): skipped')
                continue
            lines += queue(AxonRun(D, env, say), D, smoke, say)
            if smoke:
                open(os.path.join(D, 'QUEUE-SMOKE-OK'), 'w').write(time.strftime('%F %T') + '\n')
        ok = True
    except Stop as e:
        lines.append(f'QUEUE STOP: {e}')
    except Exception as e:  # noqa: BLE001  (a bug in the queue itself: say so, still wake the foreman)
        lines.append(f'QUEUE STOP: queue.py error: {type(e).__name__}: {e}')
    if ok:
        lines.append('QUEUE END ok')
    res = os.path.join(last_D, 'QUEUE-RESULT.txt')
    open(res, 'w').write('\n'.join(lines) + '\n')
    say('\n'.join(lines))
    sp = dict(key='result', name=f'{TAG}-result', cpu=True, ram=200, max_min=5, expect='QUEUE END ok',
              argv=['bash', '-c', f'cat {res}; grep -q "^QUEUE END ok" {res}'], timing=False, vram=0)
    try:
        jid = AxonRun(last_D, base, say).submit(sp)
        gw = shutil.which('axon-gatewatch') or os.path.expanduser('~/.local/bin/axon-gatewatch')
        subprocess.Popen(['setsid', 'nohup', gw, LABEL, jid], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        say(f'gatewatch {LABEL} armed on {jid}')
    except Exception as e:  # noqa: BLE001
        say(f'could not submit the result job / arm the gatewatch: {e}')
    sys.exit(0 if ok else 1)


# ---- selftest ------------------------------------------------------------------------------------------------------
class FakeRun:
    """Plays axon-run: each job 'runs' by writing the run files its cell script would, from a script of outcomes."""
    def __init__(self, D, plan):
        self.D, self.plan, self.keys, self.n = D, plan, [], 0

    def run(self, sp):
        self.keys.append(sp['key'])
        self.n += 1
        o = self.plan.get(sp['key'], {})
        jid = f'0000-000000-{self.n:04x}'
        if o.get('rc', 0):
            return dict(id=jid, rc=o['rc'], tainted=None)
        a = sp['argv']
        good = dict(errors=0, prompt_mismatch=0, tainted=False, requests=120, plan_requests=120, n_turn=100,
                    plan_turn=100, ttft_turn_p50=1.0, ttft_new_p50=2.0, itl_p99_ms=100.0, kv_tokens=65536)
        if a[0].endswith('single_stream.sh'):
            d = os.path.join(self.D, 'sb', 't' + a[1])
            os.makedirs(d)
            open(os.path.join(d, 'READING.txt'), 'w').write('COUNTS 15 cells: AHEAD 1\n')
            return dict(id=jid, rc=0, tainted='FOREIGN_GPU max=55% mean=13.3%' if o.get('tainted') else None)
        if a[0].endswith('agent_sweep.sh'):
            sw = os.path.join(self.D, 'sweep')
            c, k = a[1], int(a[2])
            if c == 'axon':
                os.makedirs(os.path.join(sw, 'axon'))
                json.dump(dict(good, ttft_turn_p50=0.13), open(os.path.join(sw, 'axon', 'run-axon-f16-s1-r0.json'), 'w'))
                open(os.path.join(sw, 'POOL'), 'w').write('101248\n')
            else:
                d = os.path.join(sw, NAMES[c] + ('' if k == 1 else f'-r{k}'))
                os.makedirs(d)
                if not o.get('norun'):
                    json.dump(dict(good, **o.get('run', {})), open(os.path.join(d, 'run-llama-f16-s1-r0.json'), 'w'))
            return dict(id=jid, rc=0, tainted=None)
        r = int(a[1])
        bd = os.path.join(self.D, 'board')
        os.makedirs(bd, exist_ok=True)
        assert not glob.glob(os.path.join(bd, f'run-*-r{r}.json')), 'a round is never run twice'
        for e in ('axon', 'llama'):
            for sp_ in ('1', '4'):
                json.dump(dict(good, ttft_turn_p50=0.13 if e == 'axon' else 0.3, **o.get(e, {}) if sp_ == '1' else {}),
                          open(os.path.join(bd, f'run-{e}-f16-s{sp_}-r{r}.json'), 'w'))
        return dict(id=jid, rc=0, tainted=None)


def selftest():
    T = {'tainted': True}
    t = tempfile.mkdtemp(prefix='queue-selftest-')
    quiet = lambda *_: None  # noqa: E731
    try:
        # 1. the full queue: sb tainted once; A1 clean on its 2nd run; A2 tainted 3 times (out); A5 tainted AND an
        #    error (no rerun); A6 no run file (no rerun); board: r0 counted, r1 lost to taint alone (replaced), r2 lost
        #    to an axon error (counts toward the 3, not replaced), r3 counted -> 4 rounds, done.
        D = os.path.join(t, 'q1')
        plan = {'sb-t1': T, 'sweep-A1-t1': {'run': T}, 'sweep-A2-t1': {'run': T}, 'sweep-A2-t2': {'run': T},
                'sweep-A2-t3': {'run': T}, 'sweep-A5-t1': {'run': dict(tainted=True, errors=2)},
                'sweep-A6-t1': {'norun': True}, 'sweep-A3-t1': {'run': dict(ttft_turn_p50=0.4)},
                'board-r1': {'llama': T}, 'board-r2': {'axon': dict(errors=1)}}
        R = FakeRun(D, plan)
        L = queue(R, D, False, quiet)
        assert R.keys == ['sb-t1', 'sb-t2', 'sweep-axon-t1'] + [f'sweep-{c}-t1' for c in CELLS] + [
            'sweep-A1-t2', 'sweep-A2-t2', 'sweep-A2-t3', 'board-r0', 'board-r1', 'board-r2', 'board-r3'], R.keys
        assert open(os.path.join(D, 'sb', 'READING.txt')).read().startswith('single stream: the run read is t2')
        _, arms = __import__('pick').load(os.path.join(D, 'sweep'))
        assert arms['A1-b12387-defaults'][1] == 'A1-b12387-defaults-r2' and arms['A2-noidle'][1] == 'A2-noidle'
        assert 'WINNER A3-noidle-ub2048' in L[3], L
        assert 'r0 counted, r1 taint, r2 lost: axon: 1 errors, r3 counted' in L[4], L[4]
        assert [x for x in L if x.startswith('VERDICT LEAD speed 1')] and L[-1].startswith('VERDICT REPORTED_'), L
        # 2. every round lost to taint alone: 8 rounds, then the reading says NO_VERDICT
        D = os.path.join(t, 'q2')
        plan = {f'board-r{i}': {'axon': T} for i in range(10)}
        R = FakeRun(D, plan)
        L = queue(R, D, False, quiet)
        assert [k for k in R.keys if k.startswith('board')] == [f'board-r{i}' for i in range(8)], R.keys
        assert [x for x in L if x.startswith('VERDICT NO_VERDICT speed 1')], L
        # 3. a job fails: STOP there (nothing after it runs)
        D = os.path.join(t, 'q3')
        R = FakeRun(D, {'sweep-A4-t1': {'rc': 143}})
        try:
            queue(R, D, False, quiet)
            raise AssertionError('no Stop')
        except Stop as e:
            assert 'sweep-A4-t1: job' in str(e) and 'rc 143' in str(e)
        assert R.keys[-1] == 'sweep-A4-t1'
        # 4. no arm qualifies: STOP before the board
        D = os.path.join(t, 'q4')
        R = FakeRun(D, {f'sweep-{c}-t{k}': {'run': T} for c in CELLS for k in (1, 2, 3)})
        try:
            queue(R, D, False, quiet)
            raise AssertionError('no Stop')
        except Stop as e:
            assert 'no sweep arm qualifies' in str(e)
        assert not [k for k in R.keys if k.startswith('board')] and len(R.keys) == 1 + 1 + 21, R.keys
        # 5. smoke: taint ignored, no reruns, one round, non-timing jobs
        D = os.path.join(t, 'q5')
        R = FakeRun(D, {'sb-t1': T, 'sweep-A1-t1': {'run': T}, 'board-r0': {'llama': T}})
        L = queue(R, D, True, quiet)
        assert R.keys == ['sb-t1', 'sweep-axon-t1'] + [f'sweep-{c}-t1' for c in CELLS] + ['board-r0'], R.keys
        assert 'r0 counted' in L[4] and not spec(True, 'board', 0, 1)['timing'] and spec(False, 'board', 0, 1)['timing']
        # 6. archive moves a run aside, never deletes; resume keeps the state
        D = os.path.join(t, 'q6')
        os.makedirs(os.path.join(D, 'sweep'))
        open(os.path.join(D, 'PASSED-agent'), 'w').write('x')
        open(os.path.join(D, 'QUEUE-SMOKE-OK'), 'w').write('x')   # a fresh queue must smoke again
        archive(D, quiet)
        assert not os.path.exists(os.path.join(D, 'QUEUE-SMOKE-OK'))
        assert not os.path.exists(os.path.join(D, 'sweep')) and len(glob.glob(os.path.join(D, 'old-*', 'sweep'))) == 1
        assert glob.glob(os.path.join(D, 'old-*', 'PASSED-agent'))
        open(os.path.join(D, 'queue-jobs.jsonl'), 'w').write(json.dumps(dict(key='sb-t1', id='1117-000000-abcd')) + '\n')
        assert AxonRun(D, {}, quiet).state == {'sb-t1': '1117-000000-abcd'}
    finally:
        shutil.rmtree(t)
    print('SELFTEST queue OK: reruns on taint alone (sb <= 4 jobs, an arm <= 3 runs, a round replaced, <= 8 rounds); '
          'errors and missing runs are kept, never rerun; a failed job stops the queue; no winner -> no board; smoke: '
          'no reruns, one round; archive moves, resume reuses the submitted jobs')


if __name__ == '__main__':
    main()
