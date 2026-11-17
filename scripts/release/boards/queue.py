#!/usr/bin/env python3
"""queue.py [--with-smoke] [--resume] | queue.py --selftest: release 0.1's boards on a 4090 pod, one cell per job.

Hardware (amendment 3, 11-17, Haseeb's decision): a RunPod RTX 4090 (axon-run --timing 4090, secure), no longer bella.
On bella no board produced a valid result: the 11-15 sweep hit its bound and 8 of its 11 runs were tainted by the
desktop's GPU load; then bella was held (paused) for its owner. The change was made before any board ran clean, so no
result informed it. The protocol and its rules (pick.py, read_board.py, sb_read.py, the reruns below) are unchanged.

Why one cell per job (11-15, jobs 1115-032610-223e, -bbd2, 1115-032613-4ff7; docs/log/2026-11-14-release-0.1-prep.md):
the sweep ran its 8 cells (~650 s each) and the reruns of its tainted arms in ONE --timing job bounded at 120 min and
the bound stopped it. So every cell is its own --timing job, and a run whose ONE fault is taint is run again as a new
job, a bounded number of times:

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

How a cell reaches the pod: axon-podjob (axon-run sends scripts to pods only that way) with pod.sh and five files staged
once on bella (stage(): the release binary, the agent trace, `git archive` of this commit's harness, axon-weekly's
4413a19 sm_89 build, their SHA256SUMS; the same five for every job, so a pod gets them once). pod.sh fetches the rest
(sha256-pinned) and runs the cell with common.sh's pod profile. Every job declares the models.tsv files its cell reads
(--models; the first job, single stream, all five: the pod booked for the queue holds them all). The cell's outputs come back in the job's
log (payload.py); queue.py unpacks each into DIR/jobs/KEY/ and merges it into DIR, where the readers find it as before.
Inputs from earlier cells go to the next job in its environment (AXON_REL_IN_POOL, AXON_REL_IN_WINNER).

--with-smoke: first the smoke (AXON_REL_SMOKE=1: the 0.5B, 60-s windows, 1 round, speed 1, no reruns, outputs under
rel01-4090-smoke), on the pod, then the real run. A real job runs only on the harness a smoke passed on: the smoke's
jobs print "SMOKE PASSED sb|agent <harness sha256>", queue.py records it (QUEUE-SMOKE-OK), and passes it to the real
jobs (AXON_REL_SMOKE_OK; common.sh need_smoke checks it against the harness the pod unpacked).
--resume: continue after this process died: the jobs already submitted (DIR/queue-jobs.jsonl) are waited for, never
submitted again. At the end (done or STOP) it writes DIR/QUEUE-RESULT.txt and sends ONE event to the foreman
(axon-notify, label dist-9f3630c-boards).
Run on bella, detached: setsid nohup scripts/release/boards/queue.py --with-smoke > ~/.cache/axon-dist/queue.log 2>&1 &
--selftest: the queue's decisions on a fake axon-run, the payload path, the staged harness (no GPU, no engine, no jobs)."""
import base64, glob, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time

S = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(os.path.dirname(os.path.dirname(S)))   # the repo: scripts/release/boards -> ../../..
sys.path.insert(0, S)
from pick import ARMS, attempt, pick, run as sweep_run, taint_only, why_not  # noqa: E402,F401
import payload  # noqa: E402
import read_board  # noqa: E402

K_SB, K_ARM, ROUNDS, MAX_ROUNDS = 4, 3, 3, 8
CELLS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
NAMES = {'A0': 'A0-4413a19-defaults', 'A1': 'A1-b12387-defaults', 'A2': 'A2-noidle', 'A3': 'A3-noidle-ub2048',
         'A4': 'A4-2x-slots', 'A5': 'A5-noidle-cram16g', 'A6': 'A6-cram16g'}
SMOKE_ENV = {'AXON_REL_SMOKE': '1', 'AXON_REL_SPEEDS': '1', 'AXON_REL_SB_MODELS': '0.5B', 'AXON_REL_SB_ROUNDS': '1'}
TAG = 'dist-9f3630c-rel01-4090'
LABEL = 'dist-9f3630c-boards'
TIMING = '4090'
# the models.tsv files a cell reads, declared per job (--models): single stream all five (pod.sh checks all five, the
# smoke too), an agent cell the 7B (the smoke: the 0.5B). A pod booked for this queue is booked for its first job, the
# single-stream smoke: it gets all five, and every later cell's set is a subset (one pod, no re-booking). An agent cell
# can also run on a pod booked for other 4090 timing work that has the 7B.
SB_MODELS = ['qwen2.5-0.5b-q4_k_m.gguf', 'qwen2.5-1.5b-instruct-q4_k_m.gguf', 'qwen2.5-3b-instruct-q4_k_m.gguf',
             'Qwen2.5-7B-Instruct-Q4_K_M.gguf', 'Qwen3-8B-Q4_K_M.gguf']
AGENT_MODEL = {True: 'qwen2.5-0.5b-q4_k_m.gguf', False: 'Qwen2.5-7B-Instruct-Q4_K_M.gguf'}
WS_MB = 5200   # the model working set: one model mapped at a time (the 8B, 4.7 GB)
STAGED = ['axon', 'board-trace-v1.json', 'harness.tar', 'llama-4413a19-sm89.tar', 'SHA256SUMS']
HARNESS = ['scripts/agent_replay.py', 'scripts/release/boards']
RC_AXON = os.path.expanduser('~/.cache/axon-dist/rel01/rc/axon')       # the dry run's binary (rc/axon.sha256)
TRACE = os.path.expanduser('~/.cache/axon-agent/traces/board-trace-v1.json')
OLD_BUILD = os.path.expanduser('~/.cache/axon-weekly/llama-4413a19-sm89')   # axon-weekly's 4413a19 sm_89 build
RUN_LOGS = os.path.expanduser('~/.cache/axon-run/logs')


class Stop(Exception):
    pass


def out_dir(smoke):
    return os.path.expanduser('~/.cache/axon-dist/rel01-4090' + ('-smoke' if smoke else ''))


def spec(smoke, phase, cell, k):
    """One job: (key, name, argv = the cell script in the harness + its args, timing, vram, ram, max_min, expect).
    Bounds (--max-min) are the cell's time plus the pod's first-job setup (the upload, b12387 and CUDA 12.8 fetched)."""
    sm = '-smoke' if smoke else ''
    if phase == 'sb':
        return dict(key=f'sb-t{k}', name=f'{TAG}{sm}-sb-t{k}', argv=['single_stream.sh', str(k)], timing=TIMING,
                    models=SB_MODELS, vram=20000, ram=5000 if smoke else 8000, max_min=25 if smoke else 45,
                    expect='SB END ok')
    if phase == 'sweep':
        return dict(key=f'sweep-{cell}-t{k}', name=f'{TAG}{sm}-sweep-{cell}-t{k}',
                    argv=['agent_sweep.sh', cell, str(k)], timing=TIMING, models=[AGENT_MODEL[smoke]],
                    vram=20000, ram=6000 if smoke else 20000, max_min=20 if smoke else 30, expect='CELL END ok')
    return dict(key=f'board-r{cell}', name=f'{TAG}{sm}-board-r{cell}', argv=['agent_board.sh', str(cell)], timing=TIMING,
                models=[AGENT_MODEL[smoke]], vram=20000, ram=6000 if smoke else 20000, max_min=20 if smoke else 75, expect='ROUND END ok')


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
    old = [n for n in ('sweep', 'board', 'sb', 'jobs', 'queue-jobs.jsonl', 'QUEUE-RESULT.txt', 'QUEUE-SMOKE-OK')
           if os.path.exists(os.path.join(D, n))]
    if old:
        a = os.path.join(D, 'old-' + time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(a)
        for n in old:
            shutil.move(os.path.join(D, n), os.path.join(a, n))
        say(f'archived {", ".join(old)} -> {a}')


def harness_tar(rev='HEAD', clean=True):
    """(tar bytes, sha256): `git archive` of the harness at rev (byte-stable for a commit); refuses a dirty tree."""
    st = subprocess.run(['git', '-C', W, 'status', '--porcelain', '--'] + HARNESS, capture_output=True, text=True)
    if clean and (st.returncode or st.stdout.strip()):
        raise Stop(f'uncommitted changes in the harness ({" ".join(HARNESS)}): {st.stdout.strip()[:200]}')
    t = subprocess.run(['git', '-C', W, 'archive', '--format=tar', rev, '--'] + HARNESS, capture_output=True)
    if t.returncode:
        raise Stop(f'git archive: {t.stderr.decode()[:200]}')
    return t.stdout, hashlib.sha256(t.stdout).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pinned(name, path):
    """A sha256 constant from a harness file (one source: common.sh REL_AXON_SHA256, pod.sh TRACE_SHA256)."""
    m = re.search(rf'^{name}=([0-9a-f]{{64}})$', open(os.path.join(S, path)).read(), re.M)
    if not m:
        raise Stop(f'no {name} in {path}')
    return m.group(1)


def stage(say):
    """The five files every pod job gets, staged once per harness commit: dict(dir, harness, commit)."""
    tar, hs = harness_tar()
    commit = subprocess.run(['git', '-C', W, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    sd = os.path.expanduser(f'~/.cache/axon-dist/rel01-pod/stage-{hs[:12]}')
    if not os.path.exists(os.path.join(sd, '.done')):
        if sha256_file(RC_AXON) != pinned('REL_AXON_SHA256', 'common.sh'):
            raise Stop(f'{RC_AXON} is not the release candidate')
        if sha256_file(TRACE) != pinned('TRACE_SHA256', 'pod.sh'):
            raise Stop(f'{TRACE} is not board-trace-v1.json')
        tmp = sd + f'.tmp{os.getpid()}'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        open(os.path.join(tmp, 'harness.tar'), 'wb').write(tar)
        pod = subprocess.run(['git', '-C', W, 'show', 'HEAD:scripts/release/boards/pod.sh'], capture_output=True)
        open(os.path.join(tmp, 'pod.sh'), 'wb').write(pod.stdout)
        os.chmod(os.path.join(tmp, 'pod.sh'), 0o755)
        for src, n in ((RC_AXON, 'axon'), (TRACE, 'board-trace-v1.json')):
            subprocess.run(['cp', '--reflink=auto', '-p', src, os.path.join(tmp, n)], check=True)
        subprocess.run(['tar', '-cf', os.path.join(tmp, 'llama-4413a19-sm89.tar'), '--sort=name', '--mtime=@0',
                        '--owner=0', '--group=0', '--numeric-owner', '-C', os.path.dirname(OLD_BUILD)]
                       + [f'llama-4413a19-sm89/{f}' for f in ('BUILD.txt', 'llama-bench', 'llama-server')], check=True)
        with open(os.path.join(tmp, 'SHA256SUMS'), 'w') as f:
            for n in STAGED[:-1]:
                f.write(f'{sha256_file(os.path.join(tmp, n))}  {n}\n')
        open(os.path.join(tmp, '.done'), 'w').write(f'{commit}\n')
        os.rename(tmp, sd)
        say(f'staged {sd} (harness {hs[:16]}, commit {commit[:12]})')
    return dict(dir=sd, harness=hs, commit=commit)


class AxonRun:
    """Submits a cell to a 4090 pod (axon-podjob -> axon-run --timing 4090), waits, and brings its outputs back from the
    job's log into DIR (payload.py). A job already in DIR/queue-jobs.jsonl is never submitted again."""
    WAITING = ('queued', 'running', 'paused', 'starting', 'placed')

    def __init__(self, D, env, say, st=None, poll=20):
        self.D, self.env, self.say, self.st, self.poll = D, env, say, st, poll
        self.bin = shutil.which('axon-run') or os.path.expanduser('~/.local/bin/axon-run')
        self.podjob = shutil.which('axon-podjob') or os.path.expanduser('~/.local/bin/axon-podjob')
        self.state_f = os.path.join(D, 'queue-jobs.jsonl')
        self.state, self.marks, self.logs = {}, set(), RUN_LOGS
        if os.path.exists(self.state_f):
            for ln in open(self.state_f):
                x = json.loads(ln)
                self.state[x['key']] = x['id']

    def run(self, sp):
        jid = self.state.get(sp['key']) or self.submit(sp)
        j = self.wait(jid)
        self.collect(sp, j)
        return j

    def inputs(self, sp):
        """The earlier cells' outputs this cell reads: sweep/POOL (an arm, a round), sweep/WINNER.env (a round)."""
        a, env, sw = sp['argv'], {}, os.path.join(self.D, 'sweep')
        if (a[0] == 'agent_sweep.sh' and a[1] != 'axon') or a[0] == 'agent_board.sh':
            if os.path.exists(os.path.join(sw, 'POOL')):
                env['AXON_REL_IN_POOL'] = open(os.path.join(sw, 'POOL')).read().strip()
        if a[0] == 'agent_board.sh' and os.path.exists(os.path.join(sw, 'WINNER.env')):
            env['AXON_REL_IN_WINNER'] = base64.b64encode(open(os.path.join(sw, 'WINNER.env'), 'rb').read()).decode()
        return env

    def submit(self, sp):
        sd = self.st['dir']
        clash = [a for a in sp['argv'] if os.path.exists(os.path.join(W, a))]
        if clash:   # axon-podjob stages an argument that names a file in its cwd and rewrites it to ./NAME
            raise Stop(f'submit {sp["name"]}: argument(s) {clash} name files in {W}: axon-podjob would stage them')
        cmd = [self.podjob, '--detach', '--timing', sp['timing'], '--vram', str(sp['vram']), '--ram', str(sp['ram']),
               '--ws', str(WS_MB), '--models', ','.join(sp['models']), '--max-min', str(sp['max_min']), '--name', sp['name'],
               '--expect', sp['expect'], '--', os.path.join(sd, 'pod.sh')] + [os.path.join(sd, n) for n in STAGED] \
            + sp['argv']
        env = dict(self.env, **self.inputs(sp))
        p = subprocess.run(cmd, cwd=W, env=env, capture_output=True, text=True, timeout=300)
        m = re.search(r'job (\d{4}-\d{6}-[0-9a-f]{4})', p.stdout + p.stderr)
        if not m:
            raise Stop(f'submit {sp["name"]}: rc {p.returncode}: {(p.stdout + p.stderr)[-400:]}')
        jid = m.group(1)
        with open(self.state_f, 'a') as f:
            f.write(json.dumps(dict(key=sp['key'], id=jid, name=sp['name'], harness=(self.st or {}).get('harness'),
                                    at=time.strftime('%F %T'))) + '\n')
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
                    return dict(id=jid, rc=d['rc'], tainted=d.get('tainted'), state=d.get('state'),
                                host=d.get('host'), pod=d.get('pod'))
            time.sleep(self.poll)

    def collect(self, sp, j):
        """The job's payload -> DIR/jobs/KEY/out -> merged into DIR (once: DIR/jobs/KEY/.merged); its smoke markers."""
        lp = os.path.join(self.logs, j['id'] + '.log')
        text = open(lp, errors='replace').read() if os.path.exists(lp) else ''
        self.marks |= set(re.findall(r'^SMOKE PASSED (agent|sb) harness tar sha256 ([0-9a-f]{64})', text, re.M))
        jd = os.path.join(self.D, 'jobs', sp['key'])
        if os.path.exists(os.path.join(jd, '.merged')):
            return
        try:
            data, _ = payload.extract(text)
        except ValueError as e:
            if j['rc'] == 0:
                raise Stop(f'{sp["name"]}: job {j["id"]} rc 0, but its log ({lp}) carries no outputs: {e}')
            self.say(f'{sp["name"]}: job {j["id"]} rc {j["rc"]}: no outputs in its log ({e})')
            return
        shutil.rmtree(jd, ignore_errors=True)
        payload.unpack(data, jd)
        try:
            payload.merge(os.path.join(jd, 'out'), self.D)
        except ValueError as e:
            raise Stop(f'{sp["name"]}: job {j["id"]}: {e}')
        open(os.path.join(jd, '.merged'), 'w').write(json.dumps(j) + '\n')
        self.say(f'{sp["name"]}: job {j["id"]} on {j.get("host")} (pod {j.get("pod")}): outputs merged')


def notify(lines, ok, say):
    """ONE event to the foreman for the whole queue (the gatewatch's channel: axon-notify, source axon-gates)."""
    n = shutil.which('axon-notify') or os.path.expanduser('~/.local/bin/axon-notify')
    txt = f'[axon-gates] {LABEL} queue {"done" if ok else "STOPPED"}: ' + ' | '.join(lines)
    try:
        p = subprocess.run([n, '--session', os.environ.get('AXON_FOREMAN') or 'axon', '--source', 'axon-gates',
                            '--severity', 'high', txt[:3800]], capture_output=True, text=True, timeout=180)
        say(f'notify rc {p.returncode} {(p.stdout + p.stderr).strip()[-200:]}')
    except (subprocess.SubprocessError, OSError) as e:
        say(f'notify failed: {e}')


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
        st = stage(say)
        base['AXON_REL_HARNESS_COMMIT'] = st['commit']
        lines.append(f'harness {st["harness"][:16]} (commit {st["commit"][:12]}), staged {st["dir"]}')
        for smoke in ([True, False] if with_smoke else [False]):
            D = out_dir(smoke)
            os.makedirs(D, exist_ok=True)
            say = say_to(os.path.join(D, 'queue.log'))
            last_D = D
            okf = os.path.join(D, 'QUEUE-SMOKE-OK')
            if smoke and resume and os.path.exists(okf) and open(okf).read().split()[0] == st['harness']:
                say('smoke already passed on this harness (--resume): skipped')
                continue
            jf = os.path.join(D, 'queue-jobs.jsonl')
            if resume and os.path.exists(jf) and {json.loads(x).get('harness') for x in open(jf)} != {st['harness']}:
                say(f'--resume: {jf} has jobs of another harness: not resumed')
                resume_d = False
            else:
                resume_d = resume and os.path.exists(jf)
            if not resume_d:
                archive(D, say)
            if smoke:
                env = dict(base, **SMOKE_ENV)
            else:
                sok = os.path.join(out_dir(True), 'QUEUE-SMOKE-OK')
                h = open(sok).read().split()[0] if os.path.exists(sok) else 'none'
                if h != st['harness']:
                    raise Stop(f'no smoke passed on this harness ({st["harness"][:16]}; {sok}: {h[:16]}): run --with-smoke')
                env = dict(base, AXON_REL_SMOKE_OK=st['harness'])
            R = AxonRun(D, env, say, st)
            lines += queue(R, D, smoke, say)
            if smoke:
                need = {('sb', st['harness']), ('agent', st['harness'])}
                if not need <= R.marks:
                    raise Stop(f'smoke: the jobs ran but printed no SMOKE PASSED for {sorted(x for x, _ in need - R.marks)}')
                open(okf, 'w').write(f'{st["harness"]} {time.strftime("%F %T")}\n')
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
    notify(lines + [f'details: bella {res}'], ok, say)
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
    assert os.path.isdir(os.path.join(W, '.git')) or os.path.isfile(os.path.join(W, '.git')), f'W {W} is not the repo'
    for smoke in (True, False):   # pod.sh runs the cell script from the harness: it exists, executable; all on pods
        for sp in (spec(smoke, 'sb', None, 1), spec(smoke, 'sweep', 'A0', 1), spec(smoke, 'board', 0, 1)):
            assert os.access(os.path.join(S, sp['argv'][0]), os.X_OK) and sp['timing'] == '4090', sp
            pm = open(os.path.join(S, 'pod.sh')).read()   # what pod.sh checks is what the job declares
            assert all(m in pm for m in sp['models']) and set(sp['models']) <= set(SB_MODELS), sp
            assert sp['argv'][0] in open(os.path.join(S, 'pod.sh')).read()
    assert os.access(os.path.join(S, 'pod.sh'), os.X_OK)
    assert pinned('REL_AXON_SHA256', 'common.sh').startswith('2c19415b') and pinned('TRACE_SHA256', 'pod.sh')
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
        assert 'r0 counted' in L[4]
        # 6. archive moves a run aside, never deletes; resume keeps the state
        D = os.path.join(t, 'q6')
        os.makedirs(os.path.join(D, 'sweep'))
        open(os.path.join(D, 'QUEUE-SMOKE-OK'), 'w').write('x')   # a fresh queue must smoke again
        archive(D, quiet)
        assert not os.path.exists(os.path.join(D, 'QUEUE-SMOKE-OK'))
        assert not os.path.exists(os.path.join(D, 'sweep')) and len(glob.glob(os.path.join(D, 'old-*', 'sweep'))) == 1
        open(os.path.join(D, 'queue-jobs.jsonl'), 'w').write(json.dumps(dict(key='sb-t1', id='1117-000000-abcd')) + '\n')
        assert AxonRun(D, {}, quiet).state == {'sb-t1': '1117-000000-abcd'}
        # 7. the pod path: each cell's inputs; a job's outputs from its log, merged once; smoke markers; no payload
        D = os.path.join(t, 'q7')
        R = AxonRun(D, {}, quiet)
        R.logs = os.path.join(t, 'logs')
        os.makedirs(R.logs)
        assert R.inputs(spec(False, 'sweep', 'A3', 1)) == {}   # nothing yet
        os.makedirs(os.path.join(D, 'sweep'))
        open(os.path.join(D, 'sweep', 'POOL'), 'w').write('101248\n')
        open(os.path.join(D, 'sweep', 'WINNER.env'), 'w').write("ARM=A3\nWHICH=new\nPAR=0\nLARGS='-ub 2048'\n")
        assert R.inputs(spec(False, 'sweep', 'axon', 1)) == {} and R.inputs(spec(False, 'sb', None, 1)) == {}
        assert R.inputs(spec(False, 'sweep', 'A3', 2)) == {'AXON_REL_IN_POOL': '101248'}
        b = R.inputs(spec(False, 'board', 4, 1))
        assert b['AXON_REL_IN_POOL'] == '101248' and base64.b64decode(b['AXON_REL_IN_WINNER']).startswith(b'ARM=A3')
        src = os.path.join(t, 'pod', 'out')
        os.makedirs(os.path.join(src, 'sweep', 'A3-noidle-ub2048'))
        open(os.path.join(src, 'sweep', 'POOL'), 'w').write('101248\n')   # what pod.sh wrote: the same bytes
        open(os.path.join(src, 'sweep', 'A3-noidle-ub2048', 'run-llama-f16-s1-r0.json'), 'w').write('{"errors": 0}')
        tgz = os.path.join(t, 'r.tgz')
        subprocess.run(['tar', '-czf', tgz, '-C', os.path.dirname(src), 'out'], check=True)
        import io
        buf = io.StringIO()
        payload.emit(open(tgz, 'rb').read(), 'agent_sweep.sh:A3,1', buf)
        hs = 'ab' * 32
        open(os.path.join(R.logs, '1117-000000-0007.log'), 'w').write(
            f'[axon-run 11:00:00] submitted\nSMOKE PASSED sb harness tar sha256 {hs} (commit c)\nCELL END ok\n'
            + buf.getvalue() + 'POD END rc 0\n')
        j = dict(id='1117-000000-0007', rc=0, host='pod1', pod='p')
        sp7 = spec(False, 'sweep', 'A3', 1)
        R.collect(sp7, j)
        R.collect(sp7, j)   # once
        assert json.load(open(os.path.join(D, 'sweep', 'A3-noidle-ub2048', 'run-llama-f16-s1-r0.json'))) == {'errors': 0}
        assert os.path.exists(os.path.join(D, 'jobs', sp7['key'], '.merged')) and R.marks == {('sb', hs)}
        open(os.path.join(R.logs, '1117-000000-0008.log'), 'w').write('CELL END ok\n')
        try:
            R.collect(spec(False, 'sweep', 'A4', 1), dict(id='1117-000000-0008', rc=0))
            raise AssertionError('no Stop')
        except Stop as e:
            assert 'carries no outputs' in str(e)
        R.collect(spec(False, 'sweep', 'A5', 1), dict(id='1117-000000-0008', rc=1))   # failed: queue() stops on its rc
        try:   # a cell argument that names a file in the repo root: refused before axon-podjob could stage it
            R.st = dict(dir=t)
            R.submit(dict(spec(False, 'sweep', 'A0', 1), argv=['agent_sweep.sh', 'README.md', '1']))
            raise AssertionError('no Stop')
        except Stop as e:
            assert "['README.md']" in str(e)
        # 8. the staged harness: git archive of the cell scripts, pod.sh, payload.py, agent_replay.py; byte-stable
        tar1, h1 = harness_tar(clean=False)
        assert h1 == harness_tar(clean=False)[1]
        names = subprocess.run(['tar', '-tf', '-'], input=tar1, capture_output=True).stdout.decode().split()
        for n in ('scripts/agent_replay.py', 'scripts/release/boards/pod.sh', 'scripts/release/boards/payload.py',
                  'scripts/release/boards/common.sh', 'scripts/release/boards/agent_board.sh'):
            assert n in names, n
    finally:
        shutil.rmtree(t)
    print('SELFTEST queue OK: reruns on taint alone (sb <= 4 jobs, an arm <= 3 runs, a round replaced, <= 8 rounds); '
          'errors and missing runs are kept, never rerun; a failed job stops the queue; no winner -> no board; smoke: '
          'no reruns, one round; every cell on a 4090 pod; archive moves, resume reuses the submitted jobs; a cell '
          'gets its inputs, its outputs come back from its log and merge once; smoke markers; the staged harness')


if __name__ == '__main__':
    main()
