#!/usr/bin/env python3
"""Concurrent load on an OpenAI-compatible server (axon serve), stdlib only.

    serve_load.py load URL --c 8 [--rounds-per-client 1] [--max-tokens 128] [--prompt-words 400]
                           [--temp 0] [--seed S] [--tag T]
    serve_load.py texts URL --c C [--max-tokens N]     (identity: each request's text and token count)

load: C client threads, each sending its requests one after another to /v1/chat/completions
(streamed, include_usage). Every prompt is salted (a unique id first), so no request reuses
another's cached prefix. Prints one JSON line:
  c, requests, tokens, wall_s, agg_tok_s (all completion tokens / wall time from the first send
  to the last byte), ttft_ms p50/max, itl_ms p50/p99 (gaps between content chunks, all streams),
  per_stream_tok_s p50 ((tokens - 1) / (last - first content) per request).

texts: the identity check's workload: a fixed set of 16 requests (greedy and seeded sampling,
different lengths and penalties), sent C at a time (C=1: one after another). Prints one JSON line
per request (index, finish_reason, completion_tokens, sha256 of the text) then a digest line.
Run it with --c 1 and --c 8 against the same server, and --c 1 against main's: all must agree.
"""
import hashlib, http.client, json, statistics, sys, threading, time, urllib.parse

SENT = "The quick brown fox jumps over the lazy dog near the river bank."


def opts(argv):
    o, pos, i = {}, [], 0
    while i < len(argv):
        if argv[i].startswith('--'):
            o[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    return pos, o


def stream(url, body, timeout=900):
    """One streamed chat request: (t_send, [t of each content chunk], text, usage, finish)."""
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    body = dict(body, stream=True, stream_options={'include_usage': True})
    t0 = time.perf_counter()
    c.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f'HTTP {r.status}: {r.read()[:300]!r}')
    times, text, usage, finish = [], [], None, None
    buf = b''
    while True:
        chunk = r.read1(65536) if hasattr(r, 'read1') else r.read(1)
        if not chunk:
            break
        buf += chunk
        while b'\n\n' in buf:
            ev, buf = buf.split(b'\n\n', 1)
            for line in ev.split(b'\n'):
                if not line.startswith(b'data: '):
                    continue
                d = line[6:]
                if d == b'[DONE]':
                    continue
                j = json.loads(d)
                if j.get('usage'):
                    usage = j['usage']
                for ch in j.get('choices', []):
                    dl = ch.get('delta', {})
                    # Reasoning (qwen3's <think>) counts as text too, marked.
                    piece = (dl.get('content') or '') + ''.join(
                        '\x00' + r for r in [dl.get('reasoning_content')] if r)
                    if piece:
                        times.append(time.perf_counter())
                        text.append(piece)
                    if ch.get('finish_reason'):
                        finish = ch['finish_reason']
    c.close()
    return t0, times, ''.join(text), usage, finish


def model_id(url):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=30)
    c.request('GET', '/v1/models')
    return json.loads(c.getresponse().read())['data'][0]['id']


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def load(url, o):
    C = int(o.get('c', 8))
    per = int(o.get('rounds-per-client', 1))
    N = int(o.get('max-tokens', 128))
    words = int(o.get('prompt-words', 400))
    temp = float(o.get('temp', 0))
    tag = o.get('tag', '')
    base = ' '.join([SENT] * (words // 12 + 1))
    results, errors = [], []
    lock = threading.Lock()
    salt0 = int(time.time() * 1000) % 10**9
    mid = model_id(url)

    def client(k):
        for r in range(per):
            body = {
                'model': mid,
                'messages': [{'role': 'user', 'content':
                              f'[request {salt0}-{k}-{r}] {base}\nWrite a long, detailed essay about the history of the Roman Empire.'}],
                'max_tokens': N,
                'temperature': temp,
            }
            if 'seed' in o:
                body['seed'] = int(o['seed']) + k
            try:
                res = stream(url, body)
            except Exception as e:  # noqa: BLE001 (reported, not raised)
                with lock:
                    errors.append(repr(e))
                return
            with lock:
                results.append(res)

    ts = [threading.Thread(target=client, args=(k,)) for k in range(C)]
    t_start = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    t_end = max((r[1][-1] for r in results if r[1]), default=time.perf_counter())
    tokens = sum((r[3] or {}).get('completion_tokens', len(r[1])) for r in results)
    ttft = [(r[1][0] - r[0]) * 1e3 for r in results if r[1]]
    itl = [(b - a) * 1e3 for r in results for a, b in zip(r[1], r[1][1:])]
    pst = [(len(r[1]) - 1) / (r[1][-1] - r[1][0]) for r in results if len(r[1]) > 2]
    wall = t_end - t_start
    out = {
        'tag': tag, 'c': C, 'requests': len(results), 'errors': errors[:3], 'tokens': tokens,
        'wall_s': round(wall, 3), 'agg_tok_s': round(tokens / wall, 1) if wall > 0 else None,
        'ttft_ms_p50': round(pct(ttft, 50), 1) if ttft else None,
        'ttft_ms_max': round(max(ttft), 1) if ttft else None,
        'itl_ms_p50': round(pct(itl, 50), 2) if itl else None,
        'itl_ms_p99': round(pct(itl, 99), 2) if itl else None,
        'per_stream_tok_s_p50': round(statistics.median(pst), 1) if pst else None,
    }
    print(json.dumps(out), flush=True)


IDENT = [
    # (prompt, max_tokens, sampling fields)
    ('Write a short poem about the sea.', 64, {'temperature': 0}),
    ('Explain how a CPU cache works.', 96, {'temperature': 0.8, 'seed': 1}),
    ('List ten fruits, one per line.', 80, {'temperature': 0.7, 'seed': 2, 'top_k': 20}),
    ('What is 17 * 23? Answer briefly.', 32, {'temperature': 0}),
    ('Tell me a story about a dragon and a knight.', 128, {'temperature': 1.0, 'seed': 3, 'top_p': 0.9}),
    ('Summarize the plot of Hamlet.', 100, {'temperature': 0.8, 'seed': 4, 'repetition_penalty': 1.15}),
    ('Write Python code that reverses a linked list.', 120, {'temperature': 0}),
    ('Give three tips for learning a language.', 70, {'temperature': 0.9, 'seed': 5, 'min_p': 0.05}),
    ('Why is the sky blue?', 60, {'temperature': 0.6, 'seed': 6, 'top_k': 0}),
    ('Translate "good morning" into five languages.', 50, {'temperature': 0}),
    ('Describe a sunset in one paragraph.', 90, {'temperature': 1.2, 'seed': 7, 'top_k': 100}),
    ('What are prime numbers? Give examples.', 75, {'temperature': 0, 'repetition_penalty': 1.1}),
    ('Write a haiku about winter.', 40, {'temperature': 0.8, 'seed': 8}),
    ('Explain recursion to a child.', 110, {'temperature': 0.5, 'seed': 9, 'top_p': 0.8}),
    ('Name the planets of the solar system.', 48, {'temperature': 0}),
    ('Write a limerick about a cat.', 64, {'temperature': 0.9, 'seed': 10, 'repetition_penalty': 1.3}),
]


def texts(url, o):
    C = int(o.get('c', 1))
    scale = float(o.get('scale', 1))
    out = [None] * len(IDENT)
    mid = model_id(url)

    def run(i):
        p, n, s = IDENT[i]
        body = dict({'model': mid, 'messages': [{'role': 'user', 'content': p}], 'max_tokens': int(n * scale)}, **s)
        _, _, text, usage, finish = stream(url, body)
        out[i] = {'i': i, 'finish': finish, 'tokens': (usage or {}).get('completion_tokens'),
                  'sha': hashlib.sha256(text.encode()).hexdigest()[:16]}

    for g in range(0, len(IDENT), C):
        ts = [threading.Thread(target=run, args=(i,)) for i in range(g, min(g + C, len(IDENT)))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    if None in out:
        sys.exit(f'texts: {out.count(None)} requests failed')
    for r in out:
        print(json.dumps(r))
    d = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16]
    print(json.dumps({'digest': d, 'c': C}), flush=True)


if __name__ == '__main__':
    pos, o = opts(sys.argv[1:])
    if len(pos) != 2 or pos[0] not in ('load', 'texts'):
        sys.exit(__doc__)
    {'load': load, 'texts': texts}[pos[0]](pos[1], o)
