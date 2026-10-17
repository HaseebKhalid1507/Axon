#!/usr/bin/env python3
"""`axon serve` under load: a long prompt admitted while other clients decode (stdlib only).

    bench_serve_admit.py admit  ROUNDS MODEL_TAG BIN... [--decoders=4] [--long=20000] [--flags-I=...]
    bench_serve_admit.py single ROUNDS MODEL_TAG BIN... [--n=128] [--parallel=8] [--idle-long=N]
    common: [--models=DIR] (default bella's ~/Projects/axon/models) [--kv=f16]

Run through the GPU queue (it loads models on the GPU). Binaries interleaved per round, cool-gated, GPU logged.
  * admit: `--decoders` streamed greedy /v1/completions clients (salted prompts, long max_tokens); once each has
    tokens, one streamed request with a ~`--long`-token prompt (salted: no prefix reuse) and max_tokens 16.
    Per client token event the arrival time; ITL = gaps between a client's events. Reported: the decoding
    clients' ITL p50/p99/max before the admit and during it (gaps overlapping its send .. its first token),
    the admitted request's TTFT, tokens the others made meanwhile; with --short=N, a ~N-token prompt sent 0.3 s
    after the long one, its TTFT. Every binary gets the same prompts in a
    round: same_text=1 when its greedy outputs equal the first binary's (the batching contract).
  * single: one server, one client: a streamed greedy request (max_tokens N, salted): TTFT and decode tok/s;
    then a 560-token prompt with max_tokens 1: TTFT; with --idle-long=N a ~N-token prompt alone: TTFT.
Extra server flags per binary: --flags-1="--prefill-chunk 256" (index into BIN...). --models=DIR: the model files.
"""
import http.client, json, statistics, subprocess, sys, threading, time, itertools, urllib.request

args = [a for a in sys.argv[1:] if not a.startswith('--')]
opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
MODE, ROUNDS, TAG, BINS = args[0], int(args[1]), args[2], args[3:]
N = int(opts.get('n', '128'))
DEC = int(opts.get('decoders', '4'))
LONG = int(opts.get('long', '20000'))
SHORT = int(opts.get('short', '0'))
IDLE_LONG = int(opts.get('idle-long', '0'))
KV = opts.get('kv', 'f16')
M = opts.get('models', '/home/haseeb/Projects/axon/models').rstrip('/') + '/'
MODELS = {'0.5B': 'qwen2.5-0.5b-q4_k_m.gguf', '7B': 'Qwen2.5-7B-Instruct-Q4_K_M.gguf'}
FILE = MODELS[TAG]
SENT = "The quick brown fox jumps over the lazy dog near the river bank."
PP560 = " ".join([SENT] * 40)
DECODE = "The history of the Roman Empire"
COOL = 70
PORT = 18531
salt = itertools.count(int(time.time()) % 100000)


def gpu():
    return subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm',
                           '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()


def cool():
    t0 = time.time()
    while time.time() - t0 < 120:
        t = int(gpu().split(',')[0])
        if t <= COOL:
            return t
        time.sleep(5)
    return t


def stream(path, body, times=None, out=None):
    """Streamed request: (ttft ms, decode tok/s, tokens, usage, t0). `times`: append each event's arrival;
    `out`: append the text."""
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=900)
    t0 = time.perf_counter()
    c.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
    r = c.getresponse()
    assert r.status == 200, (r.status, r.read()[:300])
    first = last = None
    events, usage = 0, None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith('data: '):
            continue
        d = line[6:]
        if d == '[DONE]':
            break
        v = json.loads(d)
        if v.get('usage'):
            usage = v['usage']
        for ch in v.get('choices') or []:
            if ch.get('text'):
                if out is not None:
                    out.append(ch['text'])
                now = time.perf_counter()
                first = first or now
                last = now
                events += 1
                if times is not None:
                    times.append(now)
    c.close()
    n = usage['completion_tokens'] if usage else events
    rate = (n - 1) / (last - first) if first and last > first else 0.0
    return ((first or time.perf_counter()) - t0) * 1e3, rate, n, usage, t0


def serve(b, i, ctx, parallel):
    cmd = [b, 'serve', M + FILE, '--device', 'cuda', '--port', str(PORT), '--ctx', str(ctx), '--kv', KV,
           '--parallel', str(parallel)] + opts.get(f'flags-{i}', '').split()
    log = open(f'/dev/shm/bsa-{i}.log', 'w')
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log)
    t0 = time.time()
    while time.time() - t0 < 300:
        if p.poll() is not None:
            raise RuntimeError('server exited: ' + open(f'/dev/shm/bsa-{i}.log').read()[-600:])
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                if r.status == 200:
                    return p
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError('server did not come up')


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] if v else float('nan')


def long_prompt(n, s):
    # ~16 tokens per numbered sentence, numbered so the text isn't one repeated token pattern
    k = n // 16
    return f"{s} " + " ".join(f"{j}. {SENT}" for j in range(k)) + " Summary:"


def admit_once(n_long, dec_tokens, s):
    """DEC decoders, then one long admit; prompts salted from `s` (the same for every binary in a round,
    so their greedy texts must match); returns the stats dict."""
    times = [[] for _ in range(DEC)]
    texts = [[] for _ in range(DEC + 2)]  # decoders, the long prompt, the short one
    res = [None] * DEC
    base = {'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}

    def dec(k):
        res[k] = stream('/v1/completions', {**base, 'prompt': f'{s + k + 1} {DECODE}', 'max_tokens': dec_tokens},
                        times[k], texts[k])
    th = [threading.Thread(target=dec, args=(k,)) for k in range(DEC)]
    for t in th:
        t.start()
    t0 = time.time()
    while min(len(t) for t in times) < 64 and time.time() - t0 < 60:
        time.sleep(0.01)
    short = {}

    def late():
        # a short prompt arriving while the long one prefills: its TTFT (shortest-first)
        time.sleep(0.3)
        short['ttft'] = stream('/v1/completions', {**base, 'prompt': long_prompt(SHORT, s + 99), 'max_tokens': 8},
                               None, texts[DEC + 1])[0]
    ts = threading.Thread(target=late)
    if SHORT:
        ts.start()
    ttft, _, _, usage, ta = stream('/v1/completions', {**base, 'prompt': long_prompt(n_long, s), 'max_tokens': 16},
                                   None, texts[DEC])
    tf = ta + ttft / 1e3
    for t in th:
        t.join()
    if SHORT:
        ts.join()
    before, during, after = [], [], []
    made = 0
    for ts in times:
        for a, b in zip(ts, ts[1:]):
            g = (b - a) * 1e3
            # during: the gap overlaps the admit (its send .. its first token)
            (before if b <= ta else after if a >= tf else during).append(g)
        made += sum(1 for x in ts if ta < x <= tf)
    still = sum(1 for ts in times if ts and ts[-1] > tf)
    return {'ttft': ttft, 'prompt': usage['prompt_tokens'] if usage else 0,
            'cached': (usage or {}).get('prompt_tokens_details', {}).get('cached_tokens', 0),
            'p50_before': pct(before, 50), 'p50_during': pct(during, 50), 'p99_during': pct(during, 99),
            'max_during': max(during, default=float('nan')), 'n_during': len(during), 'made_during': made,
            'still_decoding': still, 'p50_all': pct(before + during + after, 50),
            'p99_all': pct(before + during + after, 99), **({'ttft_short': short['ttft']} if SHORT else {}),
            'texts': [''.join(t) for t in texts]}


def main():
    res = {i: [] for i in range(len(BINS))}
    print(f"== {MODE} {TAG} ({FILE}) kv {KV}, rounds {ROUNDS}, bins {BINS} {opts}", flush=True)
    for rnd in range(ROUNDS):
        s0 = next(salt) * 100  # this round's prompts, the same for every binary
        ref = None
        for i, b in enumerate(BINS):
            t = cool()
            g0 = gpu()
            if MODE == 'admit':
                ctx = LONG * 5 // 4 + 1024
                p = serve(b, i, ctx, DEC + 1 + (1 if SHORT else 0))
                try:
                    admit_once(2000, 200, s0 + 50)  # warm-up: graphs for every row count, the long path
                    t = cool()
                    g0 = gpu()
                    dec_tokens = 1500 if TAG == '7B' else 4000
                    r = admit_once(LONG, dec_tokens, s0)
                    texts = r.pop('texts')
                    ref = ref or texts
                    r['same_text'] = int(texts == ref)  # vs bin0 (greedy: must be 1)
                finally:
                    p.kill()
                    p.wait()
            else:
                p = serve(b, i, max(4096, IDLE_LONG * 5 // 4 + 1024), int(opts.get('parallel', '8')))
                try:
                    base = {'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}
                    stream('/v1/completions', {**base, 'prompt': 'Hello', 'max_tokens': 4})
                    out = []
                    ttft, tg, n, _, _ = stream('/v1/completions',
                                               {**base, 'prompt': f'{s0} {DECODE}', 'max_tokens': N}, None, out)
                    pp, _, _, _, _ = stream('/v1/completions',
                                            {**base, 'prompt': f'{s0 + 1} {PP560}', 'max_tokens': 1})
                    r = {'ttft_dec': ttft, 'tg': tg, 'n': n, 'pp560_ttft': pp}
                    if IDLE_LONG:
                        # a long prompt sent to the idle server, alone
                        out.append('|')
                        r['idle_long_ttft'] = stream('/v1/completions', {**base, 'prompt': long_prompt(IDLE_LONG, s0 + 2),
                                                                         'max_tokens': 4}, None, out)[0]
                    ref = ref or out
                    r['same_text'] = int(out == ref)
                finally:
                    p.kill()
                    p.wait()
            r['gpu'] = f'{g0} -> {gpu()}'
            res[i].append(r)
            print(f"  r{rnd} bin{i} start {t}C " + " ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    print("medians [min..max]:")
    for i, b in enumerate(BINS):
        print(f"  bin{i} {b} {opts.get(f'flags-{i}', '')}")
        for k in res[i][0]:
            if k == 'gpu':
                continue
            v = [r[k] for r in res[i]]
            print(f"    {k:15} {statistics.median(v):10.2f}  [{min(v):.2f}..{max(v):.2f}]", flush=True)


main()
