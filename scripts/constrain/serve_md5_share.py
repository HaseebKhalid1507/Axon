#!/usr/bin/env python3
"""serve md5 for shared prefix KV pages (#187), from #201's serve_md5.py:
one multi-session, multi-turn conversation load (more sessions than slots,
so sessions lose their slot between turns, and on the CPU the one cache),
every session's prompt starting with the same system text (--system words:
the prefix sessions share), run through each server binary; every reply's
md5 must be the same across binaries.

usage: serve_md5_share.py --model M --device cuda|cpu --kv f16|q8 --out PREFIX NAME=BIN[:ARGS[:ENV]] ...
ARGS: extra `axon serve` flags, comma-separated. ENV: K=V,K=V for that server.
"""
import argparse
import hashlib
import http.client
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

WORDS = "river stone lamp gate mill field north copper glass harbor market tower garden candle anchor".split()


def text(seed, n):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + ("." if i % 11 == 10 else "") for i in range(n)) + " "


def complete(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    v = json.loads(r.read())
    c.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {v}")
    return v["choices"][0]["text"], v["usage"]


def metric(port, name):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/metrics")
    for line in c.getresponse().read().decode().splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    return None


def wait_ready(port, proc, name, limit=900):
    """Until both /health and /v1/models answer 200 (a loading server
    answers 503), sleeping between polls; an error past `limit` seconds or
    if the server exits."""
    t0 = time.time()
    last = None
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"{name}: server exited {proc.returncode}")
        try:
            ok = True
            for path in ("/health", "/v1/models"):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                c.request("GET", path)
                r = c.getresponse()
                r.read()
                c.close()
                last = f"{path} {r.status}"
                ok = ok and r.status == 200
            if ok:
                print(f"{name}: ready after {time.time() - t0:.1f} s", flush=True)
                return
        except OSError as e:
            last = str(e)
        time.sleep(1)
    raise RuntimeError(f"{name}: not ready after {limit} s (last: {last})")


def slots(port):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", "/slots")
        v = json.loads(c.getresponse().read())
        c.close()
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def run(name, binp, extra, env, a, port):
    cmd = [binp, "serve", a.model, "--device", a.device, "--ctx", "8192", "--port", str(port)]
    if a.device == "cuda":
        cmd += ["--kv", a.kv, "--parallel", str(a.parallel)]
    cmd += extra
    log = open(f"{a.out}.{name}.log", "w")
    print(f"{name}: {' '.join(f'{k}={v}' for k, v in env.items())} {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    try:
        wait_ready(port, p, name)
        sessions = a.sessions
        system = text(7, a.system) if a.system else ""
        prompts = [system + text(10 + s, 400 + 150 * s) for s in range(sessions)]
        out = {}
        cached = 0
        gen = []
        for turn in range(a.turns):
            def one(s):
                body = {"prompt": prompts[s], "max_tokens": a.max_tokens, "temperature": 0}
                if s % 3 == 2:  # a sampled session: seeded, so deterministic too
                    body.update(temperature=0.7, seed=s, top_k=40)
                return s, complete(port, body)
            with ThreadPoolExecutor(a.threads) as ex:
                for s, (t, u) in ex.map(one, range(sessions)):
                    out[(s, turn)] = hashlib.md5(t.encode()).hexdigest()[:12]
                    cached += u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    gen.append(u.get("completion_tokens", 0))
                    prompts[s] += t + "\n" + text(1000 * turn + s, 60)
        restored = metric(port, "axon_kv_ram_restored_tokens_total")
        shared = metric(port, "axon_prompt_shared_tokens_total")
        evicted = metric(port, "axon_pages_evicted_total")
        sl = slots(port)
    finally:
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
    print(f"SERVE {name}: cached {cached}, restored {restored}, shared {shared}, evicted {evicted}, "
          f"preempted {sl.get('preempted')}, unfed {sl.get('prefills_unfed')}, pages {sl.get('pages_total')}, "
          f"generated {sum(gen)} (max {max(gen)}, >1024: {sum(g > 1024 for g in gen)}), digest "
          f"{hashlib.md5(json.dumps(sorted(out.items())).encode()).hexdigest()[:12]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kv", default="f16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=7)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--port", type=int, default=18611)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--system", type=int, default=600, help="words of system text every session starts with")
    ap.add_argument("--max-tokens", type=int, default=40, help="each reply's max_tokens (v3; > 1024 decodes past admission's page reservation: preemption under a small pool)")
    ap.add_argument("--threads", type=int, default=3, help="requests in flight at once (v3)")
    ap.add_argument("bins", nargs="+")
    a = ap.parse_args()
    res = {}
    for b in a.bins:
        name, rest = b.split("=", 1)
        path, _, rest = rest.partition(":")
        args, _, envs = rest.partition(":")
        env = dict(x.split("=", 1) for x in envs.split(",") if x)
        res[name] = run(name, path, [x for x in args.split(",") if x], env, a, a.port)
    names = list(res)
    ref = res[names[0]]
    bad = 0
    for n in names[1:]:
        for k, v in ref.items():
            if res[n].get(k) != v:
                print(f"DIFF {n} session {k[0]} turn {k[1]}: {res[n].get(k)} vs {v}")
                bad += 1
    what = f"{a.device} {a.kv} {a.model.rsplit('/', 1)[-1]}"
    print(f"SERVE MD5 {what}: {len(ref) * (len(names) - 1) - bad}/{len(ref) * (len(names) - 1)} same", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
