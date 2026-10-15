#!/usr/bin/env python3
"""Single-stream serving benchmark: `axon gen` vs `axon serve` vs llama-server vs ollama (stdlib only).

    bench_serve.py ROUNDS MODEL_TAG AXON_BIN [--engines gen,axon,llama,ollama] [--n 128]

Run through the GPU queue: axon-q gpu timeout 2400 ... (it loads models on the GPU).
Per round, per engine (interleaved; medians reported):
  * decode: streamed greedy request, max_tokens N, the scoreboard's decode prompt, salted.
    Client-side: TTFB (response head), TTFT (first content), decode tok/s = (tokens - 1) / (last - first content).
    `gen`: axon gen's own prompt-eval and decode numbers (same prompt, -n N).
  * pp560: the scoreboard's 560-token prompt (salted), max_tokens 1: TTFT (gen: prompt eval time).
  * chat: /v1/chat/completions, an essay request, max_tokens N, streamed: TTFT, decode tok/s.
"""
import http.client, json, os, re, statistics, subprocess, sys, time, itertools, urllib.request

args = [a for a in sys.argv[1:] if not a.startswith('--')]
opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
ROUNDS, TAG, AXON = int(args[0]), args[1], args[2]
ENGINES = opts.get('engines', 'gen,axon,llama,ollama').split(',')
N = int(opts.get('n', '128'))
M = '/home/haseeb/Projects/axon/models/'
LS = '/home/haseeb/Projects/llama.cpp/build-cuda/bin/llama-server'
MODELS = {'0.5B': ('qwen2.5-0.5b-q4_k_m.gguf', 'axon-oracle'), '7B': ('Qwen2.5-7B-Instruct-Q4_K_M.gguf', 'axon-q7'),
          '1.5B': ('qwen2.5-1.5b-instruct-q4_k_m.gguf', 'axon-q15'), 'q3-8B': ('Qwen3-8B-Q4_K_M.gguf', 'axon-qwen3-8b')}
FILE, OLLAMA = MODELS[TAG]
SENT = "The quick brown fox jumps over the lazy dog near the river bank."
PP560 = " ".join([SENT] * 40)
DECODE = "The history of the Roman Empire"
CHAT = "Write a long, detailed essay about the history of the Roman Empire."
COOL = 75
PORT = 18431
salt = itertools.count(int(time.time()) % 100000)
RATE = re.compile(r'\(([\d.]+) tok/s\)')


def gpu():
    return subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm',
                           '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()


def cool():
    t0 = time.time()
    while time.time() - t0 < 60:
        t = int(gpu().split(',')[0])
        if t <= COOL:
            return t
        time.sleep(10)
    return t


def stream(host, port, path, body):
    """Streamed request: (ttfb, ttft, decode tok/s, tokens, text, the server's own decode tok/s or 0)."""
    c = http.client.HTTPConnection(host, port, timeout=600)
    data = json.dumps(body)
    t0 = time.perf_counter()
    c.request('POST', path, data, {'Content-Type': 'application/json'})
    r = c.getresponse()
    ttfb = time.perf_counter() - t0
    assert r.status == 200, (r.status, r.read()[:300])
    first = last = None
    chunks, usage, text, srv = 0, None, '', 0.0
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
        if v.get('timings'):
            srv = v['timings'].get('predicted_per_second', 0.0)
        for ch in v.get('choices') or []:
            t = ch.get('text') if 'text' in ch else (ch.get('delta') or {}).get('content')
            if t:
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
                text += t
    c.close()
    n = usage['completion_tokens'] if usage else chunks
    rate = (n - 1) / (last - first) if first and last > first else 0.0
    return ttfb * 1e3, ((first or time.perf_counter()) - t0) * 1e3, rate, n, text, srv


def run_gen(prompt, n):
    p = subprocess.run(['timeout', '600', AXON, 'gen', M + FILE, prompt, '-n', str(n), '--device', 'cuda'],
                       capture_output=True, text=True)
    r = {}
    for line in p.stderr.splitlines():
        m = RATE.search(line)
        if m and line.startswith('prompt eval'):
            # tokens / rate: the printed seconds have only 10 ms resolution
            r['pp_ms'] = int(line.split()[2]) / float(m.group(1)) * 1e3
        if m and line.startswith('decode'):
            r['tg'] = float(m.group(1))
            r['n'] = int(line.split()[1])
    if not r:
        raise RuntimeError(p.stderr[-400:])
    return r


def wait_up(url, proc, secs=180):
    t0 = time.time()
    while time.time() - t0 < secs:
        if proc.poll() is not None:
            raise RuntimeError('server exited')
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError('server did not come up')


def ollama_unload():
    body = json.dumps({"model": OLLAMA, "prompt": "", "keep_alive": 0}).encode()
    urllib.request.urlopen(urllib.request.Request('http://localhost:11434/api/generate', body), timeout=120).read()


def bench_server(engine):
    """(results dict) for one server engine: start it, warm up, measure, stop."""
    proc = None
    if engine == 'axon':
        host, port, model = '127.0.0.1', PORT, FILE.rsplit('.', 1)[0]
        proc = subprocess.Popen([AXON, 'serve', M + FILE, '--device', 'cuda', '--port', str(port), '--ctx', '4096',
                                 '--quiet'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_up(f'http://{host}:{port}/health', proc)
    elif engine == 'llama':
        host, port, model = '127.0.0.1', PORT + 1, 'x'
        proc = subprocess.Popen([LS, '-m', M + FILE, '-ngl', '99', '-fa', 'on', '-c', '4096', '--port', str(port),
                                 '-np', '1'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_up(f'http://{host}:{port}/health', proc)
    else:
        host, port, model = 'localhost', 11434, OLLAMA
    try:
        base = {'model': model, 'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}
        # warm-up (ollama: loads the model)
        stream(host, port, '/v1/completions', {**base, 'prompt': 'Hello', 'max_tokens': 4})
        r = {}
        s = next(salt)
        ttfb, ttft, tg, n, _, srv = stream(host, port, '/v1/completions',
                                           {**base, 'prompt': f'{s} {DECODE}', 'max_tokens': N})
        r.update(ttfb=ttfb, ttft_dec=ttft, tg=tg, n=n, srv_tg=srv)
        _, ttft, _, _, _, _ = stream(host, port, '/v1/completions',
                                  {**base, 'prompt': f'{next(salt)} {PP560}', 'max_tokens': 1})
        r['pp_ms'] = ttft
        _, ttft, tg, n, _, _ = stream(host, port, '/v1/chat/completions',
                                   {**base, 'messages': [{'role': 'user', 'content': f'({next(salt)}) {CHAT}'}],
                                    'max_tokens': N})
        r.update(chat_ttft=ttft, chat_tg=tg, chat_n=n)
        return r
    finally:
        if proc:
            proc.kill()
            proc.wait()
        if engine == 'ollama':
            ollama_unload()


def main():
    res = {e: [] for e in ENGINES}
    print(f"== {TAG} ({FILE}), N={N}, rounds {ROUNDS}", flush=True)
    for rnd in range(ROUNDS):
        for e in ENGINES:
            t = cool()
            before = gpu()
            if e == 'gen':
                g = run_gen(f'{next(salt)} {DECODE}', N)
                p = run_gen(f'{next(salt)} {PP560}', 1)
                r = {'tg': g['tg'], 'n': g['n'], 'ttft_dec': g['pp_ms'], 'pp_ms': p['pp_ms']}
            else:
                r = bench_server(e)
            r['gpu'] = f'{before} -> {gpu()}'
            res[e].append(r)
            print(f"  r{rnd} {e:6} start {t}C " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in r.items()), flush=True)
    print("medians:")
    keys = ['tg', 'srv_tg', 'n', 'ttfb', 'ttft_dec', 'pp_ms', 'chat_tg', 'chat_ttft', 'chat_n']
    print(f"  {'engine':7} " + " ".join(f"{k:>10}" for k in keys))
    for e in ENGINES:
        row = []
        for k in keys:
            v = [r[k] for r in res[e] if k in r]
            row.append(f"{statistics.median(v):10.2f}" if v else f"{'-':>10}")
        print(f"  {e:7} " + " ".join(row), flush=True)


main()
