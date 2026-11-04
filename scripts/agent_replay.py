#!/usr/bin/env python3
"""Replay a Synaps agent trace (agent_trace.py) against an OpenAI-API server (stdlib only).

    agent_replay.py run   --url http://127.0.0.1:PORT --model GGUF --trace FILE [cell options] --out FILE.json
    agent_replay.py board --engines axon,llama,ollama --kv f16 --model GGUF --rounds 3 --speeds 1,4
                          --axon BIN --llama-server BIN [--ollama-model NAME] --trace FILE --out DIR
                          [--sessions 8,16,24,32]   (the N_warm ladder: one cell per speed x N)
                          (bella, 7B, 3 rounds x speeds 1,4 x 3 engines: ~220-240 min MEASURED; --max-min >= 270;
                          scripts/agent_board.sh wraps it)
                          Every cell runs on a fresh process of each engine (--engines order, back to back): cells
                          share text, so an engine that ran one would start the next with its prompts cached. The
                          cell order rotates by round. MEASURED start to ready on bella, 7B (q8 board 1027-122343's
                          engines.jsonl): 2.1-3.2 s (axon, llama-server), 6.0-6.9 s (ollama). Those logs have no stop
                          time; the board measures both and prints them (ENGINE RESTARTS).
    agent_replay.py fanout --url URL --model GGUF [--busy 4] --out FILE.json   (row C's fan-out cell)
    agent_replay.py table DIR...       (markdown board from board result dirs)

Cell options (defaults = the canonical agent-32k cell, docs/log/2026-10-27-agent-replay.md):
  --scale 0.15        multiply every span (prompt, new, output) by s
  --clip 32768        context: a session whose scaled prompt would exceed clip - output - 256 is compacted once:
                      its own history drops to half the budget (the system/shared prefix is kept), then grows again
  --window-s 600      replay this many seconds of the trace: the window with the most distinct sessions
                      (ties: most requests), or the one starting at --start-ms
  --speed 1           gaps compressed by k (1 = real time; 4 = the stress cell)
  --seed 1234
  --sessions N        the cell with N sessions instead of the window's S (the N_warm ladder; board: a list, one cell
                      each). Sessions in a fixed order: primed (started before the window) and new ones interleaved in
                      proportion, each class by first request in the window, so every N keeps the cell's primed share
                      (by first arrival alone, N = 8 of agent-32k would be 6 primed of 8, against 7 of 20). N <= S keeps
                      the first N; N > S adds copies in the same order. Copy j of a session has its arrivals, lengths, compactions and max_tokens, and its
                      own fresh filler: it shares with the original only what two real sessions share (the
                      system/tool pool; a forked session's copy j forks from copy j of its parent). Copy 0 is the
                      session itself, so N = S is the canonical cell byte for byte. N_warm (plan section 4) = the
                      largest N whose median TTFT-turn p50 <= 0.5 s and re-prefill (clamped) <= 10 %, over clean
                      rounds; `table` prints it. The cells are nested (N = 8's sessions are N = 16's first 8, same
                      text): that is why a board starts a fresh engine per cell. ESTIMATED cost: each N at speed 1
                      is the 600-s window per engine per round plus its primes and one engine restart.

Text: every token span is synthetic filler, words that are single tokens in the target model's vocabulary (read
from the GGUF), one token each, so the replay is in the target model's tokens. Spans shared in the trace are
byte-identical in the replay: a session's prompt is always a prefix of its own text stream, a new session that
read a shared pool (system prompt + tools) starts with that pool's text (equal token counts = the same pool), a
forked one with its parent's first tokens. The engine's own replies are discarded (prompts are engine-independent,
so the same trace gives the same prompts on every engine and every run; reply md5s are recorded for bit checks).

Timing: open loop. Each request is sent at its trace arrival (relative to the window, / speed), or when the
session's previous reply has finished if that is later (a session never has two requests in flight; `late_ms`
records the wait). Before the window, every session that already has history is primed (its previous prompt,
max_tokens 1, oldest first, unmeasured), so each engine starts from the trace's live state.

Metrics (per run; the board takes medians over rounds):
  TTFT-new      first token of a session's first request (a new session / subagent): p50, p99
  TTFT-resume   first token after a session gap > 30 s (the replayed gap, i.e. trace gap / speed): p50, p99, and
                p50 per bucket 30 s-2 min, 2-10 min, > 10 min
  TTFT-turn     first token of an ordinary next turn (gap <= 30 s): p50, p99
  goodput       output tokens/s over requests with TTFT <= 2 s and their own ITL p99 <= 100 ms
  ITL p99       over all streamed chunks (one token per chunk on all three engines)
  warm/GB       sessions whose turns have median TTFT-turn <= 0.5 s, per GB of KV the engine holds (its config)
  re-prefill    prompt tokens the engine computed beyond the ideal (the tokens not in any prefix sent before):
                Axon usage.prompt_tokens_details.cached_tokens, llama-server timings.prompt_n; ollama: n/a (its
                prompt_eval_count is the whole prompt, cached or not: used as the prompt-count check instead)
"""
import argparse, hashlib, http.client, json, math, os, random, re, shutil, signal, statistics, struct, subprocess, sys
import threading, time, urllib.parse

# --------------------------------------------------------------------------------------------- vocabulary


def gguf_tokens(path):
    """tokenizer.ggml.tokens from a GGUF file (reads the metadata header only)."""
    f = open(path, 'rb')
    if f.read(4) != b'GGUF':
        raise SystemExit(f'{path}: not a GGUF file')
    ver, = struct.unpack('<I', f.read(4))
    _nt, nkv = struct.unpack('<QQ', f.read(16))
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def rstr():
        n, = struct.unpack('<Q', f.read(8))
        return f.read(n)

    def skip(t):
        if t in SZ:
            f.seek(SZ[t], 1)
        elif t == 8:
            rstr()
        elif t == 9:
            et, n = struct.unpack('<IQ', f.read(12))
            if et in SZ:
                f.seek(SZ[et] * n, 1)
            else:
                for _ in range(n):
                    skip(et)
    for _ in range(nkv):
        key = rstr()
        t, = struct.unpack('<I', f.read(4))
        if key == b'tokenizer.ggml.tokens' and t == 9:
            et, n = struct.unpack('<IQ', f.read(12))
            assert et == 8
            return [rstr() for _ in range(n)]
        skip(t)
    raise SystemExit(f'{path}: no tokenizer.ggml.tokens')


def word_list(model, n=2048):
    """Common lowercase words that are one token with a leading space ("Ġword") in the model's vocab."""
    toks = gguf_tokens(model)
    words = []
    for i, t in enumerate(toks[:40000]):
        if t.startswith(b'\xc4\xa0') and re.fullmatch(rb'[a-z]{3,9}', t[2:]):
            words.append(t[2:].decode())
        if len(words) >= n:
            break
    if len(words) < 256:
        raise SystemExit(f'{model}: only {len(words)} single-token words (not a GPT-2-style BPE vocab?)')
    return words


# --------------------------------------------------------------------------------------------- the cell


class Texts:
    """Deterministic token streams (word indices): one per (kind, key), extended on demand."""

    def __init__(self, nwords, seed):
        self.n, self.seed, self.s = nwords, seed, {}

    def get(self, key, length):
        a = self.s.get(key)
        if a is None:
            a = self.s[key] = []
        if len(a) < length:
            r = random.Random(f'{self.seed}:{key}:{len(a)}')
            a.extend(r.randrange(self.n) for _ in range(length - len(a)))
        return a[:length]


def pick_window(reqs, window_s, start_ms=None):
    T = [x['t_ms'] for x in reqs]
    import bisect
    if start_ms is not None:
        i = bisect.bisect_left(T, start_ms)
        return i, bisect.bisect_left(T, start_ms + window_s * 1000)
    best = None
    for i in range(len(reqs)):
        j = bisect.bisect_left(T, T[i] + window_s * 1000)
        k = (len({x['session'] for x in reqs[i:j]}), j - i, -i)
        if best is None or k > best[0]:
            best = (k, i, j)
    return best[1], best[2]


def plan(trace, a, sessions=None):
    """The cell: per request its prompt (word indices), max_tokens, schedule, class; and the primes. `sessions`:
    see --sessions (replicate())."""
    reqs = trace['requests']
    i0, i1 = pick_window(reqs, a.window_s, a.start_ms)
    win = reqs[i0:i1]
    t0 = win[0]['t_ms']
    texts = Texts(a.nwords, a.seed)
    s = a.scale
    sc = lambda v: max(0, int(round(v * s)))
    # Per chain: its base (pool or fork) and compaction state, built in trace order over the whole trace so
    # forks of chains outside the window still get their parent's text.
    base = {}      # chain -> ('pool', n) | ('fork', parent, n) | None
    comp = {}      # chain -> (epoch, drop)
    first_seen = set()

    def chain_text(c, e, length, j=0):
        """Text of chain c (copy j: see replicate) at compaction epoch e, first `length` tokens."""
        b = base.get(c)
        bl = 0
        pre = []
        if b is not None:
            bl = b[-1] if e == 0 else min(b[-1], a.clip // 4)
            if b[0] == 'pool':
                pre = texts.get(('pool', b[1]), bl)
            else:
                pe = 0
                pre = chain_text(b[1], pe, bl, j)
        if length <= len(pre):
            return pre[:length]
        return pre + texts.get(('chain', c, e) if j == 0 else ('chain', c, e, j), length - len(pre))

    out = []
    limit = lambda o: a.clip - o - 256
    prev_prompt = {}   # chain -> (epoch, length) of its latest request so far
    for k, x in enumerate(reqs):
        c = x['session']
        if c not in first_seen:
            first_seen.add(c)
            sh = x['shared'][0]['tokens'] if x['shared'] else 0
            if sh > 0 and x.get('fork') is not None and x['fork'] in first_seen:
                base[c] = ('fork', x['fork'], sc(sh))
            elif sh > 0:
                base[c] = ('pool', sc(sh), sc(sh))
            else:
                base[c] = None
            comp[c] = (0, 0)
        o = max(1, sc(x['output']))
        e, drop = comp[c]
        L = sc(x['prompt']) - drop
        if L > limit(o):
            # compaction: the session's own history drops to half the budget
            e += 1
            drop = sc(x['prompt']) - limit(o) // 2
            L = sc(x['prompt']) - drop
            comp[c] = (e, drop)
        L = max(1, L)
        if i0 <= k < i1:
            out.append(dict(k=k, chain=c, epoch=e, length=L, max_tokens=o,
                            at=(x['t_ms'] - t0) / 1000 / a.speed,
                            gap=None if x['gap_prev_ms'] is None else x['gap_prev_ms'] / 1000 / a.speed,
                            new=x['gap_prev_ms'] is None, depth=x['depth']))
        elif k < i0:
            prev_prompt[c] = (e, L)
        if k >= i1:
            break
    # primes: chains in the window with history before it, oldest last request first
    inwin = {}
    for r in out:
        inwin.setdefault(r['chain'], r)
    primes = []
    for c, r in inwin.items():
        if not r['new'] and c in prev_prompt:
            last_k = max(k for k in range(i0) if reqs[k]['session'] == c)
            primes.append((last_k, c, prev_prompt[c]))
    primes.sort()

    # Copy j of chain c is chain id c + j * M (ints, like the trace's), its requests k + j * K: unique, and copy 0
    # keeps the trace's ids.
    M = 1 + max(x['session'] for x in reqs)
    K = len(reqs)

    def prompt(chain, epoch, length):
        return chain_text(chain % M, epoch, length, chain // M)
    cell = dict(window=(i0, i1, t0), requests=out, primes=[(c, e, L) for _, c, (e, L) in primes], prompt=prompt,
                speed=a.speed, sessions=None)
    return cell if sessions is None else replicate(cell, sessions, M, K)


def session_order(cell):
    """The cell's sessions, primed and new interleaved in proportion (each by first request in the window): the next
    one is primed while the primed share taken so far is at most the new share taken (p / P <= q / Q)."""
    first = list(dict.fromkeys(r['chain'] for r in cell['requests']))
    primed = {c for c, _, _ in cell['primes']}
    pr, nw = [c for c in first if c in primed], [c for c in first if c not in primed]
    order, p, q = [], 0, 0
    while p + q < len(first):
        if p < len(pr) and (q == len(nw) or p * len(nw) <= q * len(pr)):
            order.append(pr[p]); p += 1
        else:
            order.append(nw[q]); q += 1
    return order


def replicate(cell, n, M, K):
    """--sessions N: the cell with N sessions (copy j of session c: chain c + j * M, requests k + j * K)."""
    if n < 1:
        raise SystemExit(f'--sessions {n}: need at least 1')
    order = session_order(cell)
    copies = {}
    for i in range(n):
        copies.setdefault(order[i % len(order)], []).append(i // len(order))
    reqs = [dict(r, chain=r['chain'] + j * M, k=r['k'] + j * K)
            for r in cell['requests'] for j in copies.get(r['chain'], [])]
    primes = [(c + j * M, e, L) for c, e, L in cell['primes'] for j in copies.get(c, [])]
    return dict(cell, requests=reqs, primes=primes, sessions=n)


# --------------------------------------------------------------------------------------------- client


class Engine:
    def __init__(self, url, model, kind):
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.model, self.kind = u.hostname, u.port, model, kind

    def complete(self, text, max_tokens, seed):
        if self.kind == 'ollama':
            return self.complete_ollama(text, max_tokens, seed)
        body = {'model': self.model, 'prompt': text, 'max_tokens': max_tokens, 'temperature': 0, 'seed': seed,
                'stream': True, 'stream_options': {'include_usage': True}}
        if self.kind == 'llama':
            body['cache_prompt'] = True
        data = json.dumps(body).encode()
        c = http.client.HTTPConnection(self.host, self.port, timeout=1800)
        t_send = time.perf_counter()
        c.request('POST', '/v1/completions', data, {'Content-Type': 'application/json'})
        r = c.getresponse()
        if r.status != 200:
            msg = r.read()[:300]
            c.close()
            return dict(error=f'HTTP {r.status}: {msg!r}', t_send=t_send)
        times, parts, usage, timings = [], [], None, None
        buf = b''
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b'data:'):
                continue
            p = line[5:].strip()
            if p == b'[DONE]':
                break
            try:
                j = json.loads(p)
            except ValueError:
                continue
            ch = j.get('choices') or []
            if ch:
                t = ch[0].get('text') or ''
                if t or ch[0].get('finish_reason') is None:
                    if t:
                        times.append(time.perf_counter())
                        parts.append(t)
            if j.get('usage'):
                usage = j['usage']
            if j.get('timings'):
                timings = j['timings']
        t_done = time.perf_counter()
        c.close()
        text_out = ''.join(parts)
        return dict(t_send=t_send, t_first=times[0] if times else None, t_done=t_done, chunk_t=times,
                    md5=hashlib.md5(text_out.encode()).hexdigest(), usage=usage, timings=timings)


    def complete_ollama(self, text, max_tokens, seed):
        """ollama's own /api/generate with raw: true: the prompt goes to the model exactly as sent. (Its
        /v1/completions applies the model's chat template: +29 tokens on a 24.5k prompt, and the model then
        answers a chat turn and stops at <|im_end|>; shady's #160 review, MUST 1.)"""
        body = {'model': self.model, 'prompt': text, 'raw': True, 'stream': True, 'keep_alive': -1,
                'options': {'num_predict': max_tokens, 'temperature': 0, 'seed': seed}}
        c = http.client.HTTPConnection(self.host, self.port, timeout=1800)
        t_send = time.perf_counter()
        c.request('POST', '/api/generate', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        r = c.getresponse()
        if r.status != 200:
            msg = r.read()[:300]
            c.close()
            return dict(error=f'HTTP {r.status}: {msg!r}', t_send=t_send)
        times, parts, usage = [], [], None
        while True:
            line = r.readline()
            if not line:
                break
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if j.get('error'):
                # A mid-stream abort (e.g. ollama's "token repeat limit reached"): an engine-side failure. Keep
                # its timing so its TTFT / ITL still count; it never counts toward goodput.
                c.close()
                return dict(error=f'ollama: {j["error"]}'[:300], aborted=True, t_send=t_send,
                            t_first=times[0] if times else None, t_done=time.perf_counter(), chunk_t=times,
                            md5=hashlib.md5(''.join(parts).encode()).hexdigest(), usage=None, timings=None)
            t = j.get('response') or ''
            if t:
                times.append(time.perf_counter())
                parts.append(t)
            if j.get('done'):
                usage = {'prompt_eval_count': j.get('prompt_eval_count', 0),
                         'completion_tokens': j.get('eval_count'), 'done_reason': j.get('done_reason')}
                break
        t_done = time.perf_counter()
        c.close()
        text_out = ''.join(parts)
        return dict(t_send=t_send, t_first=times[0] if times else None, t_done=t_done, chunk_t=times,
                    md5=hashlib.md5(text_out.encode()).hexdigest(), usage=usage, timings=None)


def engine_prompt_tokens(kind, res):
    """The prompt length the engine says it got."""
    u, tm = res.get('usage') or {}, res.get('timings') or {}
    if kind == 'axon' and 'prompt_tokens' in u:
        return u['prompt_tokens']
    if kind == 'llama':
        if 'prompt_n' in tm and 'cache_n' in tm:
            return tm['prompt_n'] + tm['cache_n']
        if 'prompt_tokens' in u:
            return u['prompt_tokens']
    if kind == 'ollama' and 'prompt_eval_count' in u:
        # MEASURED (q8 board): prompt_eval_count == the whole prompt on every request, cached or not, so it checks
        # the prompt but says nothing about what was recomputed.
        return u['prompt_eval_count']
    return None


def computed_tokens(kind, res):
    u, tm = res.get('usage') or {}, res.get('timings') or {}
    if kind == 'axon' and u:
        d = u.get('prompt_tokens_details') or {}
        if 'cached_tokens' in d:
            return u['prompt_tokens'] - d['cached_tokens']
    if kind == 'llama' and 'prompt_n' in tm:
        return tm['prompt_n']
    return None   # ollama: not reported (prompt_eval_count counts cached tokens too)


class Prefixes:
    """Prefixes sent so far (16-token blocks): the ideal cache's hit for a new prompt."""
    B = 16

    def __init__(self):
        self.seen = set()

    def hit_and_add(self, toks):
        h, hit, miss = 0, 0, False
        for i in range(0, len(toks) - len(toks) % self.B, self.B):
            h = hash((h, tuple(toks[i:i + self.B])))
            if not miss and (i, h) in self.seen:
                hit = i + self.B
            else:
                miss = True
            self.seen.add((i, h))
        return hit


def foreign_sampler(stop, out, own_pids):
    """nvidia-smi pmon every ~5 s: summed SM % of GPU processes outside this process tree."""
    while not stop.is_set():
        try:
            r = subprocess.run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'u'], capture_output=True, text=True,
                               timeout=20)
            mine = own_pids()
            tot = 0
            for l in r.stdout.splitlines():
                f = l.split()
                if len(f) > 3 and f[0].isdigit() and f[1].isdigit():
                    if int(f[1]) not in mine and f[3].isdigit():
                        tot += int(f[3])
            out.append(tot)
        except (OSError, subprocess.TimeoutExpired):
            pass
        stop.wait(5)


def tree_pids():
    me = os.getpid()
    kids = {}
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                with open(f'/proc/{p}/stat') as f:
                    pp = int(f.read().rsplit(')', 1)[1].split()[1])
                kids.setdefault(pp, []).append(int(p))
            except (OSError, ValueError, IndexError):
                pass
    out, st = set(), [me]
    while st:
        p = st.pop()
        if p not in out:
            out.add(p)
            st.extend(kids.get(p, []))
    return out


def slots_counters(eng):
    """Axon's cumulative /slots counters (preempted requests, prefills that gave their rows up); None elsewhere."""
    if eng.kind != 'axon':
        return None
    try:
        c = http.client.HTTPConnection(eng.host, eng.port, timeout=30)
        c.request('GET', '/slots')
        j = json.loads(c.getresponse().read())
        c.close()
        return j.get('preempted', 0), j.get('prefills_unfed', 0)
    except (OSError, ValueError):
        return None


def run_cell(eng, words, cell, a, kv_gb):
    c0 = slots_counters(eng)
    s = _run_cell(eng, words, cell, a, kv_gb)
    c1 = slots_counters(eng)
    if c0 and c1:
        s['preempted'], s['prefills_unfed'] = c1[0] - c0[0], c1[1] - c0[1]
    return s


def _run_cell(eng, words, cell, a, kv_gb):
    P = cell['prompt']
    to_text = lambda toks: ''.join(' ' + words[i] for i in toks)
    pre = Prefixes()
    # prime
    t_p = time.perf_counter()
    for c, e, L in cell['primes']:
        toks = P(c, e, L)
        pre.hit_and_add(toks)
        r = eng.complete(to_text(toks), 1, a.seed)
        if 'error' in r:
            raise SystemExit(f'prime failed: {r["error"]}')
    prime_s = time.perf_counter() - t_p
    # replay
    by_chain = {}
    for r in cell['requests']:
        by_chain.setdefault(r['chain'], []).append(r)
    lock = threading.Lock()
    results = []
    stop = threading.Event()
    foreign = []
    smp = threading.Thread(target=foreign_sampler, args=(stop, foreign, tree_pids), daemon=True)
    smp.start()
    T0 = time.perf_counter() + 1.0

    def session(reqs):
        for r in reqs:
            toks = P(r['chain'], r['epoch'], r['length'])
            text = to_text(toks)
            wait = T0 + r['at'] - time.perf_counter()
            late = max(0.0, -wait)
            if wait > 0:
                time.sleep(wait)
            with lock:   # the ideal cache sees prompts in send order
                hit = pre.hit_and_add(toks)
            res = eng.complete(text, r['max_tokens'], a.seed)
            res.update(k=r['k'], chain=r['chain'], new=r['new'], gap=r['gap'], prompt=len(toks),
                       ideal=len(toks) - hit, max_tokens=r['max_tokens'], late_ms=round(late * 1000))
            with lock:
                results.append(res)
    th = [threading.Thread(target=session, args=(v,)) for v in by_chain.values()]
    for t in th:
        t.start()
    for t in th:
        t.join()
    stop.set()
    wall = max((x['t_done'] for x in results if 't_done' in x), default=T0) - T0
    return summarize(results, wall, prime_s, kv_gb, eng.kind, foreign)


def pct(v, p):
    v = sorted(v)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, int(math.ceil(p / 100 * len(v))) - 1))]


def summarize(results, wall, prime_s, kv_gb, kind, foreign):
    ok = [x for x in results if 'error' not in x and x.get('t_first')]
    # Requests the engine aborted after streaming started: their latency counts (dropping them would flatter the
    # engine), their tokens don't count toward goodput.
    aborted = [x for x in results if x.get('aborted') and x.get('t_first')]
    lat = ok + aborted
    ttft = lambda xs: [x['t_first'] - x['t_send'] for x in xs]
    new = [x for x in lat if x['new']]
    resume = [x for x in lat if not x['new'] and x['gap'] is not None and x['gap'] > 30]
    turn = [x for x in lat if not x['new'] and (x['gap'] is None or x['gap'] <= 30)]
    itl_all, good_tok = [], 0
    for x in aborted:
        itl_all += [b - a_ for a_, b in zip(x['chunk_t'], x['chunk_t'][1:])]
        x['n_out'] = len(x['chunk_t'])
    for x in ok:
        it = [b - a_ for a_, b in zip(x['chunk_t'], x['chunk_t'][1:])]
        itl_all += it
        n_out = (x.get('usage') or {}).get('completion_tokens') or len(x['chunk_t'])
        x['n_out'] = n_out
        if x['t_first'] - x['t_send'] <= 2.0 and (not it or pct(it, 99) <= 0.100):
            good_tok += n_out
    buckets = {'30s-2m': (30, 120), '2-10m': (120, 600), '>10m': (600, 1e18)}
    per_chain = {}
    for x in turn:
        per_chain.setdefault(x['chain'], []).append(x['t_first'] - x['t_send'])
    warm = sum(1 for v in per_chain.values() if statistics.median(v) <= 0.5)
    comp = [(computed_tokens(kind, x), x['ideal'], x['prompt']) for x in ok]
    have = [c for c in comp if c[0] is not None]
    full = bool(have) and len(have) == len(comp)
    reprefill = sum(max(0, c - i) for c, i, _ in have) if full else None
    reprefill_net = sum(c - i for c, i, _ in have) if full else None
    # The engine must have seen exactly our prompt (axon, llama: their own count; ollama: what it evaluated can't
    # exceed the prompt).
    eng_p = [(engine_prompt_tokens(kind, x), x['prompt'], computed_tokens(kind, x)) for x in ok]
    mismatch = sum(1 for e, p, c in eng_p if (e is not None and e != p) or (e is None and c is not None and c > p))
    errs = [x.get('error') for x in results if 'error' in x]
    s = {
        'requests': len(results), 'errors': len(results) - len(ok), 'aborted': len(aborted),
        'errors_without_first_token': len(results) - len(ok) - len(aborted), 'wall_s': round(wall, 1),
        'first_error': errs[0][:200] if errs else None,
        'prime_s': round(prime_s, 1),
        'ttft_new_p50': pct(ttft(new), 50), 'ttft_new_p99': pct(ttft(new), 99), 'n_new': len(new),
        'ttft_resume_p50': pct(ttft(resume), 50), 'ttft_resume_p99': pct(ttft(resume), 99), 'n_resume': len(resume),
        'ttft_turn_p50': pct(ttft(turn), 50), 'ttft_turn_p99': pct(ttft(turn), 99), 'n_turn': len(turn),
        'goodput_tok_s': round(good_tok / wall, 2) if wall > 0 else None,
        'output_tok_s': round(sum(x['n_out'] for x in ok) / wall, 2) if wall > 0 else None,
        'itl_p99_ms': round(pct(itl_all, 99) * 1000, 2) if itl_all else None,
        'warm_sessions': warm, 'kv_gb': kv_gb, 'warm_per_gb': round(warm / kv_gb, 3) if kv_gb else None,
        'prompt_tokens': sum(x['prompt'] for x in ok), 'ideal_tokens': sum(x['ideal'] for x in ok),
        'reprefill_tokens': reprefill, 'reprefill_net_tokens': reprefill_net,
        'reprefill_net_share': round(reprefill_net / max(1, sum(x['prompt'] for x in ok)), 4) if full else None,
        'prompt_mismatch': mismatch,
        'prompt_checked': sum(1 for e, _, _ in eng_p if e is not None),
        'reprefill_share': round(reprefill / max(1, sum(x['prompt'] for x in ok)), 4) if reprefill is not None else None,
        'late_p50_ms': pct([x['late_ms'] for x in ok], 50), 'late_max_ms': max([x['late_ms'] for x in ok], default=None),
        'md5': hashlib.md5(''.join(x['md5'] for x in sorted(ok, key=lambda x: x['k'])).encode()).hexdigest(),
        'out_tokens_short': sum(1 for x in ok if x['n_out'] < x['max_tokens']),
        'foreign_sm_mean': round(statistics.mean(foreign), 1) if foreign else None,
        'foreign_sm_max': max(foreign) if foreign else None,
    }
    for b, (lo, hi) in buckets.items():
        v = ttft([x for x in resume if lo < x['gap'] <= hi])
        s[f'ttft_resume_{b}_p50'] = pct(v, 50)
        s[f'n_resume_{b}'] = len(v)
    s['tainted'] = bool(foreign) and (statistics.mean(foreign) > 5 or max(foreign) > 25)
    for k, v in list(s.items()):
        if isinstance(v, float) and k.startswith('ttft'):
            s[k] = round(v, 3)
    s['per_request'] = [dict(k=x['k'], chain=x['chain'], new=x['new'], gap=x['gap'], prompt=x['prompt'],
                             ideal=x.get('ideal'), computed=computed_tokens(kind, x), max_tokens=x.get('max_tokens'),
                             engine_prompt=engine_prompt_tokens(kind, x),
                             n_out=x.get('n_out'), late_ms=x.get('late_ms'), md5=x.get('md5'),
                             ttft=round(x['t_first'] - x['t_send'], 4) if x.get('t_first') else None,
                             dur=round(x['t_done'] - x['t_send'], 3) if x.get('t_done') else None,
                             err=x.get('error'), aborted=bool(x.get('aborted')))
                        for x in sorted(results, key=lambda x: x['k'])]
    return s


# --------------------------------------------------------------------------------------------- engines


KV_BYTES = {'f16': 2.0, 'q8': 34 / 32}   # bytes per element (q8_0: 34 bytes per 32)


def kv_bytes_per_token(model):
    """K + V bytes per token at f16 from the GGUF's shape (block_count, head_count_kv, key/value length)."""
    meta = gguf_meta(model, (b'block_count', b'head_count_kv', b'head_count', b'embedding_length'))
    L, hkv, h, d = meta['block_count'], meta['head_count_kv'], meta['head_count'], meta['embedding_length']
    return L * hkv * (d // h) * 2   # elements per token (K and V)


def gguf_meta(path, want):
    f = open(path, 'rb')
    f.read(4)
    struct.unpack('<I', f.read(4))
    _nt, nkv = struct.unpack('<QQ', f.read(16))
    SZ = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
    out = {}

    def rstr():
        n, = struct.unpack('<Q', f.read(8))
        return f.read(n)

    def val(t):
        if t in SZ:
            fmt = SZ[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return rstr()
        et, n = struct.unpack('<IQ', f.read(12))
        if et in SZ:
            f.seek(struct.calcsize(SZ[et]) * n, 1)
        else:
            for _ in range(n):
                val(et)
        return None
    for _ in range(nkv):
        key = rstr()
        t, = struct.unpack('<I', f.read(4))
        v = val(t)
        for w in want:
            if key.endswith(b'.' + w):
                out[w.decode()] = v
    return out


def wait_http(port, path, proc, secs, log):
    t0 = time.time()
    while time.time() - t0 < secs:
        if proc.poll() is not None:
            raise RuntimeError(f'engine exited {proc.returncode}; log {log}')
        try:
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            c.request('GET', path)
            r = c.getresponse()
            body = r.read()
            c.close()
            if r.status == 200:
                return body
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f'engine not up after {secs} s; log {log}')


def start_engine(kind, a, kv, port, logdir, axon_tokens):
    """_start_engine, and no engine outlives a failed start."""
    before = list(ENGINES)
    try:
        return _start_engine(kind, a, kv, port, logdir, axon_tokens)
    except BaseException:
        for p in [x for x in ENGINES if x not in before]:
            stop_engine(p, 10)
        raise


def _start_engine(kind, a, kv, port, logdir, axon_tokens):
    """Start one engine; return (proc, Engine, kv_tokens, config string)."""
    log = os.path.join(logdir, f'{kind}-{kv}-{port}.log')
    lf = open(log, 'w')
    env = dict(os.environ)
    if kind == 'axon':
        cmd = [a.axon, 'serve', a.model, '--device', 'cpu' if a.cpu else 'cuda', '--port', str(port), '--ctx', str(a.clip),
               '--parallel', str(1 if a.cpu else a.axon_parallel), '--kv', kv]
    elif kind == 'llama':
        t = {'f16': 'f16', 'q8': 'q8_0'}[kv]
        # slots = budget // clip, the pool = slots x clip (<= the budget): with -np 8 on the same budget the
        # unified pool overflowed under concurrent long prompts ("Context size has been exceeded", smoke
        # 1027-002031-baea), so no more slots than full-context sequences fit.
        npl = a.llama_parallel or max(1, axon_tokens // a.clip)
        ctx = npl * a.clip if not a.llama_parallel else axon_tokens
        cmd = [a.llama_server, '-m', a.model, '-ngl', '0' if a.cpu else '99', '-fa', 'on', '-c', str(ctx),
               '-np', str(npl), '-kvu', '--cache-reuse', '256', '-ctk', t, '-ctv', t,
               '--port', str(port), '--host', '127.0.0.1', '--metrics']
    elif kind == 'ollama':
        npar = max(1, axon_tokens // a.clip)
        env.update(OLLAMA_HOST=f'127.0.0.1:{port}', OLLAMA_MODELS=a.ollama_models, OLLAMA_NUM_PARALLEL=str(npar),
                   OLLAMA_CONTEXT_LENGTH=str(a.clip), OLLAMA_FLASH_ATTENTION='1',
                   OLLAMA_KV_CACHE_TYPE={'f16': 'f16', 'q8': 'q8_0'}[kv], OLLAMA_KEEP_ALIVE='-1',
                   OLLAMA_NOPRUNE='1', OLLAMA_MAX_LOADED_MODELS='1')
        cmd = [a.ollama, 'serve']
    else:
        raise SystemExit(f'unknown engine {kind}')
    print(f'start {kind} kv {kv}: {" ".join(cmd)}', flush=True)
    marker = f'OLLAMA_HOST=127.0.0.1:{port}' if kind == 'ollama' else None
    kill_leftover(port, cmd, marker)
    proc = launch(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    record_engine(logdir, proc, kind, port, 'started')
    if kind == 'axon':
        wait_http(port, '/health', proc, 600, log)
        sl = json.loads(wait_http(port, '/slots', proc, 60, log))
        tokens = sl['pages_total'] * sl['page_tokens'] if 'pages_total' in sl else a.axon_parallel * a.clip
        cfg = f'--ctx {a.clip} --parallel {a.axon_parallel} --kv {kv}; paged pool {tokens} tokens'
        model = 'axon'
    elif kind == 'llama':
        wait_http(port, '/health', proc, 600, log)
        tokens = ctx
        cfg = (f'-c {ctx} -np {npl} -kvu --cache-reuse 256 -ctk/-ctv {t} -fa on '
               f'(--cache-ram default 8192 MiB, --cache-idle-slots default on, cache_prompt true)')
        model = 'llama'
    else:
        try:
            a.ollama_version = json.loads(wait_http(port, '/api/version', proc, 120, log))['version']
        except (KeyError, ValueError):
            a.ollama_version = 'unknown'
        tokens = npar * a.clip
        cfg = (f'ollama {a.ollama_version}; OLLAMA_NUM_PARALLEL={npar} OLLAMA_CONTEXT_LENGTH={a.clip} KV {kv} '
               f'flash attention, keep_alive -1 ({tokens} tokens)')
        model = a.ollama_model
    if kind in ('axon', 'llama'):
        # the server's own model id (axon serve refuses any other name)
        try:
            model = json.loads(wait_http(port, '/v1/models', proc, 60, log))['data'][0]['id']
        except (KeyError, IndexError, ValueError):
            pass
    eng = Engine(f'http://127.0.0.1:{port}', model, kind)
    record_engine(logdir, proc, kind, port, 'up')
    # load / warm the model with a tiny request (not measured)
    r = eng.complete(' hello', 2, 1)
    if 'error' in r:
        stop_engine(proc)
        raise RuntimeError(f'{kind}: warm-up request failed: {r["error"]}; log {log}')
    if kind == 'ollama' and getattr(a, 'ollama_expect', None):
        # ollama 0.40.13 migrates its store on first use (manifests -> manifests-v2), after the board's check: check
        # again what it can load now. Blobs that didn't change aren't hashed again (file_sha256).
        try:
            again = ollama_check(a.ollama_models, a.ollama_model, a.model_sha256)
        except SystemExit as e:
            raise RuntimeError(f'ollama: check after start failed: {e}; log {log}')
        same = lambda x: (x['weights'], x['config'], x['layers'], x['params'])
        if same(again) != same(a.ollama_expect):
            raise RuntimeError(f'ollama: after start {a.ollama_model} resolves to other blobs ({again["manifest"]}) '
                               f'than the board checked ({a.ollama_expect["manifest"]}); log {log}')
        cfg += (f'; ollama_check after start: {again["manifest"]} + {len(again["others"])} other manifest(s), '
                f'same blobs')
    # Tokenization check: a fresh (never sent) prompt of N filler words must be exactly N tokens to the engine.
    n = 512
    probe = Texts(len(a.words), a.seed).get(('tokcheck', kind, port, time.time()), n)
    r = eng.complete(''.join(' ' + a.words[i] for i in probe), 1, a.seed)
    got = engine_prompt_tokens(kind, r)
    if 'error' in r or got != n:
        raise RuntimeError(f'{kind}: tokenization check failed: {n} filler words -> {got} prompt tokens '
                           f'({r.get("error")}); log {log}')
    cfg += f'; tokenization check {n} words = {got} tokens'
    print(f'{kind} ready: {cfg}', flush=True)
    record_engine(logdir, proc, kind, port, 'ready')
    return proc, eng, tokens, cfg


def ollama_mirror(src, dst):
    """A user-owned OLLAMA_MODELS for the board's own `ollama serve`, built inside the board's out dir (11-04).
    The system store `src` (owned by the ollama user, mode 755) is read-only to us, and ollama 0.40.13 (bella,
    upgraded from 0.40.6 on 10-31) migrates a store on first use by creating `manifests-v2` in it; against
    /var/lib/ollama that failed with EACCES (kill smoke 1104-090227-04aa). The mirror: `blobs/` is a directory of
    per-file symlinks into src/blobs (no copy of the GGUFs; anything ollama writes there, and any prune, stays in
    the mirror), and src's manifest dirs are copied. One per board: it is never stale, two boards never share it,
    and it records which manifests the board ran with. `src` is only read."""
    if os.path.lexists(dst):
        shutil.rmtree(dst)
    sb = os.path.join(src, 'blobs')
    os.makedirs(os.path.join(dst, 'blobs'))
    for f in sorted(os.listdir(sb)):
        os.symlink(os.path.join(sb, f), os.path.join(dst, 'blobs', f))
    copied = [d for d in ('manifests', 'manifests-v2') if os.path.isdir(os.path.join(src, d))]
    if not copied:
        raise SystemExit(f'ollama store {src}: no manifests dir')
    for d in copied:
        shutil.copytree(os.path.join(src, d), os.path.join(dst, d), symlinks=True)
    return f'{dst} ({len(os.listdir(sb))} blobs linked from {src}, {"+".join(copied)} copied)'


OLLAMA_PUBLIC_HOSTS = ('registry.ollama.ai', 'ollama.com')
OLLAMA_WEIGHTS = 'application/vnd.ollama.image.model'
# The layer types a board's ollama model may have (#182): weights, chat template (unused: prompts go raw), sampling
# params, license, system prompt (unused: raw). An adapter (LoRA), projector or anything newer would make ollama run
# something other than --model, so ollama_check refuses it.
OLLAMA_LAYERS = {f'application/vnd.ollama.image.{t}': t for t in ('model', 'template', 'params', 'license', 'system')}


def ollama_manifest_paths(store, model):
    """The manifest paths ollama tries for `model` under OLLAMA_MODELS=store, in its order (11-04: ollama
    manifest/paths.go resolveManifestPath, identical at tag v0.40.13 and on main): manifests-v2/<host>/<ns>/<model>/
    <tag> with the public host spelled ollama.com (must exist), then manifests/<host>/... for the host as given and
    then the other public spelling (the first that is present, even as a dangling link, wins). Default host
    registry.ollama.ai, namespace library, tag latest. Returns [(path, picked_by_existence_test)]."""
    name, tag = model, 'latest'
    if ':' in model.rsplit('/', 1)[-1]:
        name, tag = model.rsplit(':', 1)
    parts = name.split('/')
    if not tag or not all(parts) or len(parts) > 3:
        raise SystemExit(f'ollama model {model!r}: not a model name this check can resolve')
    if len(parts) == 1:
        host, ns, mdl = OLLAMA_PUBLIC_HOSTS[0], 'library', parts[0]
    elif len(parts) == 2:
        host, ns, mdl = OLLAMA_PUBLIC_HOSTS[0], parts[0], parts[1]
    else:
        host, ns, mdl = parts
    public = host.lower() in OLLAMA_PUBLIC_HOSTS
    v2 = os.path.join(store, 'manifests-v2', ('ollama.com' if public else host).replace(':', '%3A'), ns, mdl, tag)
    hosts = [host] + ([OLLAMA_PUBLIC_HOSTS[1] if host.lower() == OLLAMA_PUBLIC_HOSTS[0] else OLLAMA_PUBLIC_HOSTS[0]]
                      if public else [])
    return [(v2, os.path.exists)] + [(os.path.join(store, 'manifests', h, ns, mdl, tag), os.path.lexists) for h in hosts]


def ollama_check(store, model, model_sha):
    """Refuse to start the board unless ollama provably runs --model (11-04, shady's review of 060c555; #182; every
    manifest after shady's #186 review). EVERY manifest ollama could resolve for the name (ollama_manifest_paths: a v2
    entry and a v1 one can both be present, and ollama migrates v1 to v2 on first use) must pass, not just the first:
    one weights layer, the same in all of them, = sha256(--model) (axon and llama-server get --model itself); no layer
    outside OLLAMA_LAYERS; every blob it names (config and layers) a sha256 digest, present in store/blobs with the
    manifest's size and hashing to its own digest (a file under the right name is not enough; ~3 s for the 7B's
    weights, each blob hashed once). The board runs it again once its ollama is up (_start_engine). Returns what
    meta.json records: the manifest ollama loads first, the others, the config blob and every layer (type, digest,
    size), and the params layer's contents (sampling settings ollama applies on top of the request)."""
    cands = [p for p, present in ollama_manifest_paths(store, model) if present(p)]
    if not cands:
        raise SystemExit(f'ollama model {model}: no manifest under {store} (tried: '
                         f'{", ".join(os.path.relpath(p, store) for p, _ in ollama_manifest_paths(store, model))})')
    rel = lambda p: os.path.relpath(p, store)

    def weights(path):
        try:
            j = json.load(open(path))
        except (OSError, ValueError) as e:
            raise SystemExit(f'ollama model {model}: manifest {path} unreadable ({e})')
        w = [x.get('digest') for x in j.get('layers') or [] if x.get('mediaType') == OLLAMA_WEIGHTS]
        if len(w) != 1:
            raise SystemExit(f'ollama model {model}: {len(w)} weights layers in {path}')
        return j, w[0]
    mans = [weights(p) for p in cands]
    w = mans[0][1]
    for p, (_, wo) in zip(cands[1:], mans[1:]):
        if wo != w:
            raise SystemExit(f'ollama model {model}: {rel(cands[0])} (the one ollama loads) names {w}, {rel(p)} '
                             f'names {wo}')
    if w != f'sha256:{model_sha}':
        raise SystemExit(f'ollama model {model} runs {w}, not --model (sha256:{model_sha}): the engines would not '
                         f'run the same file')
    out = [_ollama_manifest(store, model, rel(p), j) for p, (j, _) in zip(cands, mans)]
    return dict(out[0], name=model, manifest=rel(cands[0]), manifests_found=len(cands), weights=w,
                others=[rel(p) for p in cands[1:]])


def _ollama_manifest(store, model, where, j):
    """ollama_check for one parsed manifest `j` (at `where`): layer types, digests, presence, sizes, hashes."""
    layers = j.get('layers') or []
    other = [x.get('mediaType') for x in layers if x.get('mediaType') not in OLLAMA_LAYERS]
    if other:
        raise SystemExit(f'ollama model {model}: layers {other} in {where}: ollama would not run --model alone '
                         f'(allowed: {", ".join(OLLAMA_LAYERS.values())})')
    blobs = [('config', j.get('config') or {})] + [(OLLAMA_LAYERS[x['mediaType']], x) for x in layers]
    bad = [t for t, x in blobs if not re.fullmatch(r'sha256:[0-9a-f]{64}', str(x.get('digest')))]
    if bad:
        raise SystemExit(f'ollama model {model}: no sha256 digest on its {bad} blob(s) in {where}')
    path = lambda x: os.path.join(store, 'blobs', x['digest'].replace(':', '-'))
    missing = [x['digest'] for _, x in blobs if not os.path.exists(path(x))]
    if missing:
        raise SystemExit(f'ollama model {model}: blobs missing from {store}/blobs: {missing} ({where})')
    for t, x in blobs:   # sizes first (free), then the bytes
        if x.get('size') is not None and os.path.getsize(path(x)) != x['size']:
            raise SystemExit(f'ollama model {model}: {t} blob {x["digest"]} is {os.path.getsize(path(x))} bytes, '
                             f'its manifest says {x["size"]} ({where})')
    for t, x in blobs:
        got = 'sha256:' + file_sha256(path(x))
        if got != x['digest']:
            raise SystemExit(f'ollama model {model}: {t} blob {path(x)} hashes to {got}, not its name {x["digest"]}')
    params = None
    for t, x in blobs:
        if t == 'params':
            try:
                params = json.load(open(path(x)))
            except ValueError as e:
                raise SystemExit(f'ollama model {model}: params blob {x["digest"]} is not JSON ({e})')
    return dict(blobs=len(blobs), config=dict(digest=blobs[0][1]['digest'], size=blobs[0][1].get('size')),
                layers=[dict(type=t, digest=x['digest'], size=x.get('size')) for t, x in blobs[1:]], params=params)


_SHA = {}   # (dev, inode, size, mtime, ctime) -> sha256: a file is hashed again if any of them moved. ctime too:
#             a write that keeps size and mtime (utime) still moves it, and nothing short of the clock sets it back


def file_sha256(path):
    st = os.stat(path)
    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if key not in _SHA:
        with open(path, 'rb') as f:
            _SHA[key] = hashlib.file_digest(f, 'sha256').hexdigest()
    return _SHA[key]


def harness_provenance(here=None):
    """This script's provenance for meta.json and `run` results (#182; hardened after shady's #186 review): the commit
    of the git tree it runs from, the tracked files that differ from that commit, and this file's own sha256.
    - commit and dirty are None unless this file is tracked at HEAD in that tree: an untracked copy, or a copy
      dropped into some other repo, has no commit to name.
    - dirty: what `git status` lists (staged or not; untracked files aren't listed); plus the files git is told not
      to look at (assume-unchanged / skip-worktree, `git ls-files -v`) whose bytes differ from HEAD's; plus this file
      itself whenever its bytes differ from HEAD's blob, whatever git's index says."""
    here = os.path.realpath(here or __file__)
    sha = file_sha256(here)
    commit = dirty = None
    try:
        def git(*x, cwd=os.path.dirname(here)):
            return subprocess.run(['git', '-c', 'core.fsmonitor=false', '-C', cwd, *x], capture_output=True,
                                  timeout=60)
        top, head = git('rev-parse', '--show-toplevel'), git('rev-parse', '--verify', '-q', 'HEAD')
        if top.returncode == 0 and head.returncode == 0:
            top = top.stdout.decode().strip()
            mine = os.path.relpath(here, os.path.realpath(top))
            at_head = lambda f: git('cat-file', 'blob', f'HEAD:{f}', cwd=top)
            blob = at_head(mine)
            if blob.returncode == 0:
                commit, dirty = head.stdout.decode().strip(), set()
                st = git('status', '--porcelain', '-z', '--untracked-files=no', cwd=top)
                ent = st.stdout.decode(errors='replace').split('\0')
                i = 0
                while i < len(ent):
                    e = ent[i]
                    if len(e) > 3:
                        dirty.add(e[3:])
                        i += 2 if set(e[:2]) & {'R', 'C'} else 1   # a rename / copy: then its source path
                    else:
                        i += 1
                lv = git('ls-files', '-v', '-z', cwd=top).stdout.decode(errors='replace').split('\0')
                for e in lv:
                    if len(e) > 2 and (e[0].islower() or e[0] == 'S'):
                        f, p = e[2:], os.path.join(top, e[2:])
                        if os.path.isfile(p):
                            b = at_head(f)
                            if b.returncode != 0 or hashlib.sha256(b.stdout).hexdigest() != file_sha256(p):
                                dirty.add(f)
                if hashlib.sha256(blob.stdout).hexdigest() != sha:
                    dirty.add(mine)
                dirty = sorted(dirty)
    except (OSError, subprocess.SubprocessError):
        commit = dirty = None
    return dict(commit=commit, dirty=dirty, sha256=sha, path=here)

def ollama_offload(logdir, port):
    """ollama's own layer placement from its log (a partial offload makes it slow, and says so)."""
    txt = ''
    for f in os.listdir(logdir):
        if f.startswith('ollama-') and f.endswith(f'-{port}.log'):
            txt = open(os.path.join(logdir, f), errors='replace').read()
    m = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', txt)
    return f'; offloaded {m[-1][0]}/{m[-1][1]} layers to GPU' if m else '; offload: not in log'


# Engine lifetime (10-27: a llama-server started in its own session survived axon-run's killpg of the board
# job and held 10 GB of bella's VRAM for 37 min). Engines now (1) stay in the board's process group, so the
# job's killpg reaches them; (2) get PR_SET_PDEATHSIG = SIGKILL, so they die with this process even if it is
# SIGKILLed; (3) are stopped, with their whole process tree (ollama's runner), by SIGTERM / SIGINT / SIGHUP
# handlers and atexit. At start, a leftover engine from a crashed run on the same port is killed only if it is
# ours (same uid) with exactly the command line we are about to run (ollama: and our OLLAMA_HOST); a foreign
# listener makes the start fail instead.
ENGINES = []   # live Popen objects this process launched
_LIBC = None


def _pdeathsig():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except OSError:
        pass


def launch(cmd, **kw):
    proc = subprocess.Popen(cmd, preexec_fn=_pdeathsig, **kw)
    ENGINES.append(proc)
    return proc


def _children():
    kids = {}
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                with open(f'/proc/{p}/stat') as f:
                    pp = int(f.read().rsplit(')', 1)[1].split()[1])
                kids.setdefault(pp, []).append(int(p))
            except (OSError, ValueError, IndexError):
                pass
    return kids


def _tree(pid):
    kids, out, st = _children(), [], [pid]
    while st:
        p = st.pop()
        out.append(p)
        st.extend(kids.get(p, []))
    return out


def _alive(pid):
    try:
        with open(f'/proc/{pid}/stat') as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def kill_tree(pid, grace=30.0):
    """SIGTERM pid and every descendant, wait up to `grace` s, then SIGKILL whatever is left."""
    pids = _tree(pid)
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    t0 = time.time()
    while time.time() - t0 < grace and any(_alive(p) for p in pids):
        for p in pids:   # reap our own child
            try:
                os.waitpid(p, os.WNOHANG)
            except OSError:
                pass
        time.sleep(0.2)
    for p in pids + _tree(pid):
        if _alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass


def stop_engine(proc, grace=30.0):
    kill_tree(proc.pid, grace)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        pass
    if proc in ENGINES:
        ENGINES.remove(proc)
    time.sleep(3)


def stop_all(grace=10.0):
    for proc in list(ENGINES):
        kill_tree(proc.pid, grace)
        if proc in ENGINES:
            ENGINES.remove(proc)


def _on_signal(sig, _frame):
    print(f'agent_replay: signal {sig}: stopping {len(ENGINES)} engine(s)', flush=True)
    stop_all()
    os._exit(128 + sig)


def install_guards():
    import atexit
    atexit.register(stop_all)
    for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sg, _on_signal)


def listeners(port):
    """PIDs with a TCP socket listening on `port` (IPv4/IPv6), from /proc."""
    inodes = set()
    for f in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            for l in open(f).readlines()[1:]:
                x = l.split()
                if x[3] == '0A' and int(x[1].rsplit(':', 1)[1], 16) == port:
                    inodes.add(x[9])
        except OSError:
            pass
    out = set()
    if not inodes:
        return out
    for p in os.listdir('/proc'):
        if not p.isdigit():
            continue
        try:
            for fd in os.listdir(f'/proc/{p}/fd'):
                t = os.readlink(f'/proc/{p}/fd/{fd}')
                if t.startswith('socket:[') and t[8:-1] in inodes:
                    out.add(int(p))
                    break
        except OSError:
            pass
    return out


def kill_leftover(port, cmd, env_marker=None):
    for pid in listeners(port):
        try:
            mine = os.stat(f'/proc/{pid}').st_uid == os.getuid()
            argv = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')[:-1]
            env = open(f'/proc/{pid}/environ', 'rb').read().split(b'\0') if mine else []
        except OSError:
            continue
        same = mine and [x.decode(errors='replace') for x in argv] == cmd and \
            (env_marker is None or env_marker.encode() in env)
        if not same:
            raise RuntimeError(f'port {port} is held by pid {pid}, not a leftover of this command: not touching it')
        print(f'killing leftover engine pid {pid} on port {port} (our uid, same command line)', flush=True)
        kill_tree(pid, 15)


def record_engine(logdir, proc, kind, port, state):
    with open(os.path.join(logdir, 'engines.jsonl'), 'a') as f:
        f.write(json.dumps({'pid': proc.pid, 'kind': kind, 'port': port, 'state': state, 't': time.time()}) + '\n')


# --------------------------------------------------------------------------------------------- fan-out cell


def bg_stream(eng, text, max_tokens, seed, first, stop):
    """A background decoding session: stream until `stop`, then drop the connection."""
    body = json.dumps({'model': eng.model, 'prompt': text, 'max_tokens': max_tokens, 'temperature': 0,
                       'seed': seed, 'stream': True}).encode()
    c = http.client.HTTPConnection(eng.host, eng.port, timeout=1800)
    try:
        c.request('POST', '/v1/completions', body, {'Content-Type': 'application/json'})
        r = c.getresponse()
        while not stop.is_set():
            line = r.readline()
            if not line:
                break
            if line.startswith(b'data:') and b'"text"' in line:
                first.set()
    except OSError:
        pass
    finally:
        first.set()
        c.close()


def run_fanout(eng, words, a, rnd):
    """One fan-out burst (row C's cell, re-registered 10-27): `siblings` requests sent `spread_ms` apart in
    total, each = the SAME shared span (S tokens) + its own private delta (d tokens), `reply` tokens each.
    busy > 0: the burst arrives while that many unrelated sessions are decoding (so the siblings are admitted
    into a running batch and in-flight dedupe can fire). Every round uses fresh text, so the server holds none
    of it when the burst starts."""
    tx = Texts(len(words), a.seed)
    to_text = lambda toks: ''.join(' ' + words[i] for i in toks)
    shared = tx.get(('fan-shared', rnd), a.shared)
    prompts = [shared + tx.get(('fan-delta', rnd, i), a.delta) for i in range(a.siblings)]
    stop, bg = threading.Event(), []
    for j in range(a.busy):
        first = threading.Event()
        t = threading.Thread(target=bg_stream, daemon=True,
                             args=(eng, to_text(tx.get(('fan-bg', rnd, j), a.busy_prompt)), a.busy_reply,
                                   a.seed, first, stop))
        t.start()
        bg.append((t, first))
    for t, first in bg:
        first.wait(600)
    if bg:
        time.sleep(2.0)   # all background sessions are decoding
    res = [None] * a.siblings
    T0 = time.perf_counter() + 0.2
    step = a.spread_ms / 1000 / max(1, a.siblings - 1)

    def one(i):
        w = T0 + i * step - time.perf_counter()
        if w > 0:
            time.sleep(w)
        res[i] = eng.complete(to_text(prompts[i]), a.reply, a.seed)
    th = [threading.Thread(target=one, args=(i,)) for i in range(a.siblings)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    stop.set()
    errs = [r.get('error') for r in res if 'error' in r]
    if errs:
        return dict(round=rnd, busy=a.busy, errors=len(errs), first_error=errs[0][:200])
    ttft = [r['t_first'] - r['t_send'] for r in res]
    # the leader is sibling 1 (sent first); each sibling's increment over it, both from their own send
    inc = [ttft[i] - ttft[0] for i in range(1, a.siblings)]
    comp = [computed_tokens(eng.kind, r) for r in res]
    shared_prefilled = [None if c is None else max(0, c - a.delta) for c in comp[1:]]
    return dict(round=rnd, busy=a.busy, errors=0, shared=a.shared, delta=a.delta, siblings=a.siblings,
                leader_ttft=round(ttft[0], 4), ttft=[round(x, 4) for x in ttft],
                sib_ttft_p50=round(statistics.median(ttft[1:]), 4),
                sib_inc_ttft_p50=round(statistics.median(inc), 4),
                computed=comp, sib_shared_prefilled=shared_prefilled,
                md5=hashlib.md5(''.join(r['md5'] for r in res).encode()).hexdigest())


def cmd_fanout(a):
    words = word_list(a.model)
    eng = Engine(a.url, a.engine_model, a.kind)
    if a.kind in ('axon', 'llama'):
        try:
            u = urllib.parse.urlparse(a.url)
            c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
            c.request('GET', '/v1/models')
            eng.model = json.loads(c.getresponse().read())['data'][0]['id']
        except (OSError, KeyError, IndexError, ValueError):
            pass
    out = []
    for rnd in range(a.rounds):
        r = run_fanout(eng, words, a, rnd)
        out.append(r)
        print(f'FANOUT busy {a.busy} round {rnd}: ' + ' '.join(f'{k}={r.get(k)}' for k in
              ('errors', 'leader_ttft', 'sib_ttft_p50', 'sib_inc_ttft_p50', 'sib_shared_prefilled', 'md5')), flush=True)
        time.sleep(a.cool_s)
    ok = [r for r in out if not r['errors']]
    summ = dict(variant='busy' if a.busy else 'idle', busy=a.busy, shared=a.shared, delta=a.delta,
                siblings=a.siblings, reply=a.reply, spread_ms=a.spread_ms, rounds=a.rounds,
                sib_inc_ttft_p50=statistics.median([r['sib_inc_ttft_p50'] for r in ok]) if ok else None,
                sib_ttft_p50=statistics.median([r['sib_ttft_p50'] for r in ok]) if ok else None,
                leader_ttft_p50=statistics.median([r['leader_ttft'] for r in ok]) if ok else None,
                md5s=sorted({r['md5'] for r in ok}), runs=out)
    json.dump(summ, open(a.out, 'w'), indent=1)
    print('FANOUT ' + ' '.join(f'{k}={summ[k]}' for k in ('variant', 'sib_inc_ttft_p50', 'sib_ttft_p50',
                                                          'leader_ttft_p50', 'md5s')))
    print('FANOUT END')


# --------------------------------------------------------------------------------------------- commands


def cmd_run(a):
    trace = json.load(open(a.trace))
    words = word_list(a.model)
    a.nwords = len(words)
    cell = plan(trace, a, a.sessions)
    eng = Engine(a.url, a.engine_model, a.kind)
    s = run_cell(eng, words, cell, a, a.kv_gb)
    s['cell'] = cell_desc(a, cell)
    s['sessions'] = a.sessions
    s['harness'] = harness_provenance()
    json.dump(s, open(a.out, 'w'), indent=1)
    print(brief(s))


def cell_desc(a, cell):
    i0, i1, t0 = cell['window']
    return dict(scale=a.scale, clip=a.clip, window_s=a.window_s, speed=cell['speed'], seed=a.seed, trace_start_ms=t0,
                sessions=cell['sessions'],
                n_requests=len(cell['requests']), n_sessions=len({r['chain'] for r in cell['requests']}),
                n_primes=len(cell['primes']), prime_tokens=sum(L for _, _, L in cell['primes']),
                prompt_p50=pct([r['length'] for r in cell['requests']], 50))


def brief(s):
    keys = ['requests', 'errors', 'first_error', 'wall_s', 'prime_s', 'ttft_new_p50', 'ttft_resume_p50', 'ttft_turn_p50',
            'goodput_tok_s', 'itl_p99_ms', 'warm_per_gb', 'reprefill_share', 'reprefill_net_share', 'out_tokens_short',
            'prompt_mismatch', 'prompt_checked', 'preempted', 'prefills_unfed', 'tainted']
    return ' '.join(f'{k}={s.get(k)}' for k in keys)


def cmd_board(a):
    install_guards()
    t_board = time.time()
    trace = json.load(open(a.trace))
    words = a.words = word_list(a.model)
    a.nwords = len(words)
    os.makedirs(a.out, exist_ok=True)
    bpt = kv_bytes_per_token(a.model)
    speeds = [float(x) for x in a.speeds.split(',')]
    ladder = [int(x) for x in a.sessions.split(',')] if a.sessions else [None]
    engines = a.engines.split(',')
    cells = {}   # (speed, sessions) -> cell; sessions None = the window's own sessions
    for sp in speeds:
        for n in ladder:
            a.speed = sp
            cells[sp, n] = plan(trace, a, n)
            print(f'cell speed {sp}' + (f' sessions {n}' if n else '') + f': {cell_desc(a, cells[sp, n])}', flush=True)
    meta = dict(model=os.path.basename(a.model), kv=a.kv, engines=engines, rounds=a.rounds, speeds=speeds,
                sessions=ladder if a.sessions else None, trace=os.path.basename(a.trace), kv_elems_per_token=bpt,
                cells={(str(sp) if n is None else f'{sp}/n{n}'): cell_desc(a, v) for (sp, n), v in cells.items()},
                started=time.strftime('%Y-%m-%d %H:%M:%S'), gpu=gpu_name(), harness=harness_provenance(),
                state='checking')
    meta_path = os.path.join(a.out, 'meta.json')

    def save_meta():
        # Written at start and after every run, so a capped or killed board keeps its hashes and config (#182).
        json.dump(meta, open(meta_path + '.tmp', 'w'), indent=1)
        os.replace(meta_path + '.tmp', meta_path)
    save_meta()
    h = meta['harness']
    print(f'harness {h["commit"]}{" DIRTY " + ",".join(h["dirty"]) if h["dirty"] else ""} '
          f'agent_replay.py sha256 {h["sha256"][:16]}', flush=True)
    # Every engine must run the same file: axon and llama-server get --model, ollama the blob its manifest names.
    meta['model_sha256'] = file_sha256(a.model)
    print(f'model {a.model} sha256 {meta["model_sha256"]}', flush=True)
    for kind, b in (('axon', a.axon), ('llama', a.llama_server)):
        if kind in engines and b:
            meta.setdefault('binaries', {})[kind] = dict(path=os.path.abspath(b), sha256=file_sha256(b))
    if 'ollama' in engines:
        if a.ollama_models:
            a.ollama_models = os.path.abspath(a.ollama_models)
            meta['ollama_models'] = f'{a.ollama_models} (as given)'
        else:
            a.ollama_models = os.path.join(os.path.abspath(a.out), 'ollama-models')
            meta['ollama_models'] = ollama_mirror(os.path.abspath(a.ollama_source), a.ollama_models)
        meta['ollama_model'] = ollama_check(a.ollama_models, a.ollama_model, meta['model_sha256'])
        om = meta['ollama_model']
        print(f'ollama store: {meta["ollama_models"]}; {a.ollama_model} -> {om["manifest"]}, weights {om["weights"]} '
              f'= sha256(--model); layers {", ".join(x["type"] for x in om["layers"])}, all {om["blobs"]} '
              f'blobs hashed = their digests; {om["manifests_found"]} manifest(s) checked '
              f'({", ".join([om["manifest"]] + om["others"])}); params {json.dumps(om["params"])}', flush=True)
    meta['state'] = 'running'
    save_meta()
    a.model_sha256 = meta['model_sha256']
    a.ollama_expect = meta.get('ollama_model')
    axon_tokens = a.kv_tokens
    port = a.port
    shrink = {}    # kind -> budget cuts a start needed (llama-server): later starts begin there
    starts = meta['starts'] = []
    order = list(cells)
    for rnd in range(a.round_offset, a.round_offset + a.rounds):
        # A fresh engine process per cell (shady, #186 review): the cells share text (the ladder's N are nested; the
        # speeds replay one window), so a process that ran one cell would start the next with its prompts cached.
        # MEASURED on #160's q8 board, where speed 4 ran on speed 1's process: llama-server served 9 of 13 new
        # sessions' first prompts at speed 4 from speed 1's leftovers (138,239 tokens, 69 % of the cell's ideal
        # prefill). The order rotates by round (the absolute round: a --round-offset rerun repeats it).
        k = rnd % len(order)
        for sp, n in order[k:] + order[:k]:
            cell = cells[sp, n]
            ns = '' if n is None else f' sessions {n}'
            for kind in engines:
                port += 1
                proc = None
                st = dict(kind=kind, round=rnd, speed=sp, sessions=n)
                try:
                    budget = axon_tokens
                    for _ in range(shrink.get(kind, 0)):
                        budget = int(budget * 0.85) // 256 * 256
                    for attempt in range(3):
                        try:
                            t0 = time.time()
                            proc, eng, tokens, cfg = start_engine(kind, a, a.kv, port, a.out, budget)
                            st.update(pid=proc.pid, start_s=round(time.time() - t0, 2))
                            break
                        except RuntimeError as e:
                            # llama-server's own buffers may not fit next to Axon's KV budget: shrink it, say so
                            if kind != 'llama' or attempt == 2:
                                raise
                            budget = int(budget * 0.85) // 256 * 256
                            shrink[kind] = shrink.get(kind, 0) + 1
                            port += 1
                            print(f'{kind}: start failed ({e}); retrying with {budget} KV tokens', flush=True)
                    if kind == 'ollama':
                        cfg += ollama_offload(a.out, port)
                        meta.setdefault('engine_versions', {})['ollama'] = a.ollama_version
                    if kind == 'axon' and not a.kv_tokens:
                        axon_tokens = tokens
                    meta.setdefault('config', {})[kind] = cfg
                    kv_gb = tokens * bpt * KV_BYTES[a.kv] / 1e9
                    save_meta()
                    a.speed = sp
                    t = time.time()
                    s = run_cell(eng, words, cell, a, kv_gb)
                    s.update(engine=kind, kv=a.kv, round=rnd, speed=sp, config=cfg, kv_tokens=tokens,
                             engine_pid=proc.pid, engine_start_s=st['start_s'])
                    if n is not None:
                        s['sessions'] = n
                    if s.get('prompt_mismatch'):
                        meta.setdefault('failures', []).append(
                            f'{kind} speed {sp:g}{ns} round {rnd}: {s["prompt_mismatch"]} prompts the engine counted differently')
                    f = os.path.join(a.out, f'run-{kind}-{a.kv}-s{sp:g}' + (f'-n{n}' if n else '') + f'-r{rnd}.json')
                    json.dump(s, open(f, 'w'), indent=1)
                    print(f'RUN {kind} kv {a.kv} speed {sp:g}{ns} round {rnd}: {brief(s)} ({time.time() - t:.0f} s; '
                          f'engine pid {proc.pid}, started in {st["start_s"]:.1f} s)', flush=True)
                    meta.setdefault('runs', []).append(os.path.basename(f))
                except Exception as e:  # one engine failing must not lose the others
                    print(f'RUN {kind} kv {a.kv} speed {sp:g}{ns} round {rnd}: FAILED {e}', flush=True)
                    meta.setdefault('failures', []).append(f'{kind} speed {sp:g}{ns} round {rnd}: {e}')
                finally:
                    if proc:
                        t0 = time.time()
                        stop_engine(proc)
                        st['stop_s'] = round(time.time() - t0, 2)
                        starts.append(st)
                    save_meta()
                if kind == 'axon' and not axon_tokens:
                    print('axon failed and no --kv-tokens: the baselines need the KV budget', flush=True)
    print(restart_cost(starts, time.time() - t_board), flush=True)
    meta['ended'] = time.strftime('%Y-%m-%d %H:%M:%S')
    meta['state'] = 'ended'
    save_meta()
    print(table([a.out]))
    errs = sum(json.load(open(os.path.join(a.out, f)))['errors'] for f in os.listdir(a.out) if f.startswith('run-'))
    if meta.get('failures'):
        print(f'BOARD END WITH FAILURES: {meta["failures"]}')
    else:
        print(f'BOARD END ok ({errs} request errors)')


def restart_cost(starts, wall):
    """What a fresh engine per cell costs: per engine, its starts (launch to ready: load, warm-up request and
    tokenization check) and stops, and their sum against the board's wall time."""
    if not starts:
        return 'ENGINE RESTARTS: none'
    rng = lambda v: f'{statistics.median(v):.1f} s [{min(v):.1f}-{max(v):.1f}]'
    out, tot = [], 0.0
    for kind in dict.fromkeys(x['kind'] for x in starts):
        xs = [x for x in starts if x['kind'] == kind]
        tot += sum(x['start_s'] + x['stop_s'] for x in xs)
        out.append(f'{kind} {len(xs)} x start {rng([x["start_s"] for x in xs])} + stop {rng([x["stop_s"] for x in xs])}')
    return (f'ENGINE RESTARTS (one per cell per engine): {"; ".join(out)}; total {tot:.0f} s of {wall:.0f} s board wall '
            f'({tot / max(wall, 1e-9):.1%})')


def gpu_name():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except OSError:
        return None


METRICS = [('TTFT-new p50 s', 'ttft_new_p50'), ('TTFT-new p99 s (max when n < 100)', 'ttft_new_p99'),
           ('  n new', 'n_new'),
           ('TTFT-resume p50 s', 'ttft_resume_p50'), ('TTFT-resume p99 s (max when n < 100)', 'ttft_resume_p99'),
           ('  n resume', 'n_resume'),
           ('resume 30s-2m p50', 'ttft_resume_30s-2m_p50'), ('  n 30s-2m', 'n_resume_30s-2m'),
           ('resume 2-10m p50', 'ttft_resume_2-10m_p50'), ('  n 2-10m', 'n_resume_2-10m'),
           ('resume >10m p50', 'ttft_resume_>10m_p50'), ('  n >10m', 'n_resume_>10m'),
           ('TTFT-turn p50 s', 'ttft_turn_p50'), ('  n turn', 'n_turn'),
           ('goodput tok/s', 'goodput_tok_s'), ('ITL p99 ms', 'itl_p99_ms'),
           ('warm/GB (GPU KV budget)', 'warm_per_gb'), ('re-prefill share (clamped)', 'reprefill_share'),
           ('re-prefill share (net)', 'reprefill_net_share'), ('preempted (axon /slots)', 'preempted'),
           ('output tok/s', 'output_tok_s'), ('replies short of max_tokens', 'out_tokens_short'),
           ('prompt counts checked (engine-reported)', 'prompt_checked'), ('prompt-count mismatches', 'prompt_mismatch'),
           ('wall s', 'wall_s'), ('errors (all kinds)', 'errors'),
           ('  engine aborts mid-stream (latency counted, no goodput)', 'aborted'),
           ('tainted runs', 'tainted')]


def comparable(group):
    """Runs from before the raw-prompt fix (no prompt_checked field) sent ollama through its chat template."""
    return not any(r['engine'] == 'ollama' and 'prompt_checked' not in r for r in group)


def table(dirs, clean_only=False):
    runs = []
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.startswith('run-') and f.endswith('.json'):
                r = json.load(open(os.path.join(d, f)))
                if r['engine'] == 'ollama':
                    # runs before d7f5fd3+1 took prompt_eval_count as recomputed tokens; it is the whole prompt
                    r['reprefill_share'] = r['reprefill_net_share'] = None
                if clean_only and r.get('tainted'):
                    continue   # the methodology's taint rule: a tainted run is a missing round
                runs.append(r)
    groups = {}
    for r in runs:
        groups.setdefault((r['speed'], r.get('sessions'), r['kv'], r['engine']), []).append(r)
    lines = []
    for sp, n in sorted({k[:2] for k in groups}, key=lambda x: (x[0], x[1] or 0)):
        cols = sorted([k for k in groups if k[:2] == (sp, n)],
                      key=lambda k: (k[2], ['axon', 'llama', 'ollama'].index(k[3])))
        lines.append(f'\n**speed {sp:g}' + (f', {n} sessions' if n else '') + '** (median of rounds; [min–max])\n')
        lines.append('| metric | ' + ' | '.join(
            f"{k[3]} {k[2]} (n={len(groups[k])}: r{','.join(str(r['round']) for r in sorted(groups[k], key=lambda r: r['round']))})"
            + ('' if comparable(groups[k]) else ' NON-COMPARABLE: chat template') for k in cols) + ' |')
        lines.append('|---|' + '---|' * len(cols))
        for name, key in METRICS:
            row = []
            for k in cols:
                v = [r.get(key) for r in groups[k]]
                if key == 'tainted':
                    rs = sorted(groups[k], key=lambda r: r['round'])
                    bad = [f"r{r['round']}" for r in rs if r.get('tainted')]
                    row.append(f"{len(bad)}/{len(rs)}" + (f" ({', '.join(bad)})" if bad else ''))
                    continue
                v = [x for x in v if x is not None]
                if not v:
                    row.append('n/a')
                elif len(v) == 1:
                    row.append(f'{v[0]:.3g}')
                else:
                    row.append(f'{statistics.median(v):.3g} [{min(v):.3g}–{max(v):.3g}]')
            lines.append(f'| {name} | ' + ' | '.join(row) + ' |')
        md = {k: {r['md5'] for r in groups[k]} for k in cols}
        lines.append('| reply md5 stable over rounds | ' + ' | '.join('yes' if len(md[k]) == 1 else f'no ({len(md[k])})'
                                                                   for k in cols) + ' |')
    lines += n_warm(groups)
    return '\n'.join(lines)


def n_warm(groups):
    """The N_warm ladder's reading (plan section 4), per speed / KV / engine over its --sessions cells: N_warm = the
    largest N whose median TTFT-turn p50 <= 0.5 s and median re-prefill share (clamped) <= 10 %, over CLEAN rounds
    only (the plan's taint rule: a tainted run is a missing round), whatever the table above includes. n/a for an
    engine that reports no re-prefill (ollama). Flags: NON-MONOTONE (a smaller N misses), and a cell with no clean
    round (N_warm then only reads the cells that have one)."""
    ladder = {}
    for (sp, n, kv, eng), rs in groups.items():
        if n is not None:
            ladder.setdefault((sp, kv, eng), {})[n] = rs
    lines = []
    for (sp, kv, eng), cells in sorted(ladder.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        marks, verdict, dropped = [], {}, 0
        for n in sorted(cells):
            clean = [r for r in cells[n] if not r.get('tainted')]
            dropped += len(cells[n]) - len(clean)

            def med(key):
                v = [r[key] for r in clean if r.get(key) is not None]
                return statistics.median(v) if v else None
            turn, rp = med('ttft_turn_p50'), med('reprefill_share')
            if not clean:
                verdict[n] = None
                marks.append(f'{n}: no clean round')
                continue
            verdict[n] = None if turn is None or rp is None else turn <= 0.5 and rp <= 0.10
            marks.append(f'{n}: turn {"n/a" if turn is None else f"{turn:.3g} s"}, re-prefill '
                         f'{"n/a" if rp is None else f"{rp:.1%}"}'
                         + ('' if verdict[n] is None else ' ok' if verdict[n] else ' MISS') + f' (n={len(clean)})')
        judged = [n for n, v in verdict.items() if v is not None]
        if not judged:
            nw = 'n/a (no clean round)' if all(not [r for r in cells[n] if not r.get('tainted')] for n in cells) \
                else 'n/a (no re-prefill number)'
        else:
            ok = [n for n in judged if verdict[n]]
            best = max(ok) if ok else None
            nw = f'{best}' if best is not None else f'< {min(judged)} (every N misses)'
            below = [n for n in judged if best is not None and n < best and not verdict[n]]
            if below:
                nw += f' NON-MONOTONE ({", ".join(map(str, below))} miss)'
            gaps = [n for n, v in verdict.items() if v is None]
            if gaps:
                nw += f' INCOMPLETE (no reading at {", ".join(map(str, gaps))})'
        if not lines:
            lines.append('\n**N_warm** (largest N with median TTFT-turn p50 <= 0.5 s and re-prefill <= 10 %, clean '
                         'rounds only)\n')
        lines.append(f'- speed {sp:g}, {eng} {kv}: N_warm = {nw} ({"; ".join(marks)})'
                     + (f' [{dropped} tainted run(s) left out]' if dropped else ''))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'board'):
        p = sub.add_parser(name)
        p.add_argument('--trace', required=True)
        p.add_argument('--model', required=True, help='the GGUF (its vocabulary builds the filler)')
        p.add_argument('--scale', type=float, default=0.15)
        p.add_argument('--clip', type=int, default=32768)
        p.add_argument('--window-s', type=float, default=600)
        p.add_argument('--start-ms', type=int)
        p.add_argument('--seed', type=int, default=1234)
        if name == 'run':
            p.add_argument('--url', required=True)
            p.add_argument('--kind', default='axon', choices=['axon', 'llama', 'ollama'])
            p.add_argument('--engine-model', default='axon')
            p.add_argument('--speed', type=float, default=1)
            p.add_argument('--sessions', type=int, help='the cell with N sessions (see Cell options)')
            p.add_argument('--kv-gb', type=float)
            p.add_argument('--out', required=True)
        else:
            p.add_argument('--engines', default='axon,llama,ollama')
            p.add_argument('--kv', default='f16', choices=['f16', 'q8'])
            p.add_argument('--rounds', type=int, default=3)
            p.add_argument('--round-offset', type=int, default=0, help='number rounds from here (targeted reruns)')
            p.add_argument('--speeds', default='1,4')
            p.add_argument('--sessions', help='the N_warm ladder: N,N,... (one cell per speed x N; see Cell options)')
            p.add_argument('--axon')
            p.add_argument('--axon-parallel', type=int, default=16)
            p.add_argument('--llama-server')
            p.add_argument('--llama-parallel', type=int, default=0, help='default: KV budget // clip')
            p.add_argument('--ollama', default='ollama')
            p.add_argument('--ollama-model', default='axon-q7')
            p.add_argument('--ollama-models', help='OLLAMA_MODELS for the board\'s ollama, used as is (default: a '
                           'user-owned mirror of --ollama-source in OUT/ollama-models; see ollama_mirror). Either '
                           'way ollama_check must pass')
            p.add_argument('--ollama-source', default='/var/lib/ollama', help='the system store the mirror links '
                           'to (only read)')
            p.add_argument('--kv-tokens', type=int, help='KV budget for the baselines (default: Axon\'s pool)')
            p.add_argument('--port', type=int, default=18600)
            p.add_argument('--cpu', action='store_true', help='smoke only: engines on the CPU')
            p.add_argument('--out', required=True)
    p = sub.add_parser('fanout', help="row C's fan-out cell (idle and busy variants), against a running server")
    p.add_argument('--url', required=True)
    p.add_argument('--kind', default='axon', choices=['axon', 'llama', 'ollama'])
    p.add_argument('--engine-model', default='axon')
    p.add_argument('--model', required=True, help='the GGUF (its vocabulary builds the filler)')
    p.add_argument('--siblings', type=int, default=8)
    p.add_argument('--shared', type=int, default=6006, help="shared span S (A's measured p50, unscaled)")
    p.add_argument('--delta', type=int, default=1110, help="private delta d (A's measured p50 new, unscaled)")
    p.add_argument('--reply', type=int, default=600)
    p.add_argument('--spread-ms', type=float, default=100)
    p.add_argument('--busy', type=int, default=0, help='background decoding sessions (0 = the idle variant)')
    p.add_argument('--busy-prompt', type=int, default=2048)
    p.add_argument('--busy-reply', type=int, default=4000)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--cool-s', type=float, default=3)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--out', required=True)
    p = sub.add_parser('table')
    p.add_argument('dirs', nargs='+')
    p.add_argument('--clean-only', action='store_true', help='drop tainted runs (they count as missing rounds)')
    a = ap.parse_args()
    if a.cmd == 'run':
        cmd_run(a)
    elif a.cmd == 'board':
        cmd_board(a)
    elif a.cmd == 'fanout':
        cmd_fanout(a)
    else:
        print(table(a.dirs, a.clean_only))


if __name__ == '__main__':
    main()
