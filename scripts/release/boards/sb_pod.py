#!/usr/bin/env python3
"""Release 0.1's copy of axon-devstack pod/board/sb_pod.py (sha256 954985e6...; docs/log/2026-11-14-release-0.1-prep.md),
changed in four places, each marked "release 0.1": three from shady's #270 review (the run order rotates by round:
round r starts at unit r, axon then each llama.cpp build, a Latin square, so no engine always runs first after the
cool or after another's long bench; the GPU's state is recorded before every run; every sample, full precision, goes
to sb.json, read by sb_read.py) and a fix: cool() parses a fractional temperature (driver 620.70.06 on bella and avante
reports 86.59; the original's int() raised before the first run). The protocol is otherwise the weekly board's.

Single-stream board on a pod (axon-weekly; from the 10-18 RunPod board's sb_pod7.py, same protocol): per model,
cool the GPU, warm each engine once, then ROUNDS interleaved rounds of axon gen (pp560, pp2100 with -n 2; tg128) and
llama-bench (LB_ARGS: -fa 1 -b 4096, -ub 512 and 4096, -t 4, -r 3), salted prompts; medians and best printed per
cell, raw samples kept: llama-bench's JSON in lbjson/, axon's stderr in axraw/. Before every run the GPU must be
<= QUIET_PCT busy (waited for, up to QUIET_S; the busy % at the start is printed per round).
llama.cpp is reported twice (11-13, devstack #61): `llama` = its best ubatch per cell (named in the raw part), and
`llama_ub512` = its -ub 512 default.
Keeps the pod's axon-node hold fresh (the idle guard and the on-pod watchdog leave it alone).
usage: sb_pod.py ROUNDS MODELS AXON_BIN --llama=NAME:PATH [--llama=...]   (run from the board dir; models from
       /root/axon-node/models; MODELS e.g. 0.5B,1.5B,3B,7B,q3-8B). Ends with SCOREBOARD_DONE."""
import glob, itertools, json, os, re, statistics, subprocess, sys, time
args = [a for a in sys.argv[1:] if not a.startswith('--')]
flags = [a for a in sys.argv[1:] if a.startswith('--')]
ROUNDS, WANT, BINS = int(args[0]), args[1].split(','), args[2:]
NODE = os.environ.get('SB_NODE', '/root/axon-node')   # tests: a fake node dir
M = NODE + '/models/'
LBS = dict(f[8:].split(':', 1) for f in flags if f.startswith('--llama='))
os.makedirs('lbjson', exist_ok=True)
os.makedirs('axraw', exist_ok=True)
LB_ARGS = ['-p', '560,2100', '-n', '128', '-ngl', '99', '-fa', '1', '-b', '4096', '-ub', '512,4096', '-t', '4',
           '-r', '3', '-o', 'json']
QUIET_PCT, QUIET_S = 10, 120
NV = sorted(glob.glob(NODE + '/lib/nvrtc-*'))[-1]
AXENV = dict(os.environ, AXON_THREADS='4', LD_LIBRARY_PATH=NV, XDG_CACHE_HOME=NODE + '/cache')
SENT = "The quick brown fox jumps over the lazy dog near the river bank."
PROMPTS = {"pp560": " ".join([SENT] * 40), "pp2100": " ".join([SENT] * 150)}
DECODE = "The history of the Roman Empire"
MODELS = {'0.5B': 'qwen2.5-0.5b-q4_k_m.gguf', '1.5B': 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
          '3B': 'qwen2.5-3b-instruct-q4_k_m.gguf', '7B': 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
          'q3-4B': 'Qwen3-4B-Q4_K_M.gguf', 'q3-8B': 'Qwen3-8B-Q4_K_M.gguf'}
COOL = 75
salt = itertools.count(int(time.time()) % 100000)
RATE = re.compile(r'\(([\d.]+) tok/s\)')


def hold():
    subprocess.run(['python3', NODE + '/bin/axon-node', 'hold'], capture_output=True)


def gpu():
    return subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem',
                           '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()


def cool():
    t0, t = time.time(), 0
    while time.time() - t0 < 60:
        t = int(float(gpu().split(',')[0]))   # release 0.1: driver 620 reports 86.59, and int() raised
        if t < COOL:
            return t
        time.sleep(10)
    return t


def axon(b, f, prompt, n, save=None):
    p = subprocess.run(['timeout', '600', b, 'gen', M + f, prompt, '-n', str(n), '--device', 'cuda'],
                       capture_output=True, text=True, env=AXENV)
    if save:   # token counts, device lines: what the rate was measured on
        open(f'axraw/{save}.txt', 'w').write(f'rc {p.returncode}\n' + p.stderr[-20000:])
    r = {}
    for line in p.stderr.splitlines():
        m = RATE.search(line)
        if m and line.startswith('prompt eval'):
            r['pp'] = float(m.group(1))
        if m and line.startswith('decode'):
            r['tg'] = float(m.group(1))
    if not r:
        raise RuntimeError(p.stderr[-400:])
    return r


def llama(f, e, tag, rnd):
    """{(metric, n_ubatch): avg tok/s} from one llama-bench run (every LB_ARGS combination)."""
    p = subprocess.run([LBS[e], '-m', M + f, *LB_ARGS], capture_output=True, text=True, timeout=900)
    open(f'lbjson/{e}-{tag}-r{rnd}.json', 'w').write(p.stdout)
    if p.stderr:
        open(f'lbjson/{e}-{tag}-r{rnd}.err', 'w').write(p.stderr[-4000:])
    out = {}
    for t in json.loads(p.stdout):
        out[('tg128' if t['n_gen'] else f"pp{t['n_prompt']}", t['n_ubatch'])] = t['avg_ts']
    return out


def busy(n=2):
    xs = []
    for _ in range(n):
        o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True).stdout.strip()
        xs.append(int(o or 0))
        time.sleep(0.1)
    return max(xs)


CONT = []


def quiet():
    """Wait (up to QUIET_S) until the GPU is <= QUIET_PCT busy; record the busy % the run starts at."""
    t0 = time.time()
    b = busy()
    while b > QUIET_PCT and time.time() - t0 < QUIET_S:
        time.sleep(1)
        b = busy()
    CONT.append(b)
    hold()


med = lambda xs: statistics.median(xs) if xs else float('nan')  # noqa: E731
engines = [f'axon{i}' for i in range(len(BINS))] + [x for e in LBS for x in (e, e + '_ub512')]
units = [e for e in engines if not e.endswith('_ub512')]   # release 0.1: what one run slot is (rotated by round)
OUT = dict(rounds=ROUNDS, bins=BINS, llama=LBS, engines=engines, cells={}, best_ub={}, runs=[])   # release 0.1: sb.json
print(f"# scoreboard {time.strftime('%Y-%m-%d %H:%M')} rounds={ROUNDS} bins={BINS} flags={flags}", flush=True)
for tag in WANT:
    f = MODELS[tag]
    hold()
    t = cool()
    print(f"== {tag} cooled to {t}C | {gpu()}", flush=True)
    for b in BINS:
        axon(b, f, "Hello", 2)
    res, ubs = {}, {}
    for rnd in range(ROUNDS):
        k = rnd % len(units)   # release 0.1: round r starts at unit r
        for e in units[k:] + units[:k]:
            if e.endswith('_ub512'):
                continue   # filled with its llama-bench run below
            OUT['runs'].append(dict(model=tag, round=rnd, engine=e, gpu=gpu()))   # release 0.1: state before the run
            if e in LBS:
                quiet()
                got = llama(f, e, tag, rnd)
                for k in sorted({k for k, _ in got}):
                    ub, v = max(((u, x) for (kk, u), x in got.items() if kk == k), key=lambda t: t[1])
                    res.setdefault((k, e), []).append(v)
                    ubs.setdefault((k, e), []).append(ub)
                    if (k, 512) in got:
                        res.setdefault((k, e + '_ub512'), []).append(got[(k, 512)])
                continue
            for job, prompt, n in [('pp560', PROMPTS['pp560'], 2), ('pp2100', PROMPTS['pp2100'], 2),
                                   ('tg128', DECODE, 128)]:
                quiet()
                r = axon(BINS[int(e[4:])], f, f"Note {next(salt)}. {prompt}", n, save=f"{e}-{tag}-{job}-r{rnd}")
                res.setdefault((job, e), []).append(r['tg' if job == 'tg128' else 'pp'])
        print(f"  round {rnd} | {gpu()} | pre-run GPU busy% {CONT[-9:]}", flush=True)
    for job in ['pp560', 'pp2100', 'tg128']:
        cells = '  '.join(f"{e}={med(res.get((job, e), [])):9.1f}/{max(res.get((job, e), [float('nan')])):.0f}"
                          for e in engines)
        raw = ' '.join(f"{e}:{[round(x) for x in res.get((job, e), [])]}" for e in engines)
        raw += ''.join(f" {e}_best_ub:{ubs[(job, e)]}" for e in LBS if (job, e) in ubs)
        print(f"  {tag:7} {job:6} {cells}   | {raw}", flush=True)
        OUT['cells'].setdefault(tag, {})[job] = {e: res.get((job, e), []) for e in engines}   # release 0.1
        OUT['best_ub'].setdefault(tag, {})[job] = {e: ubs.get((job, e), []) for e in LBS}
    with open('sb.json.tmp', 'w') as fh:   # release 0.1: after every model, so a cut-short board keeps its cells
        json.dump(OUT, fh, indent=1)
    os.replace('sb.json.tmp', 'sb.json')
print("SCOREBOARD_DONE", flush=True)
