#!/usr/bin/env python3
"""TTFT-resume micro-bench for `axon serve --cache-ram` (agent-workload PR B).

The cell of docs/design/agent-workload.md §6 row B, minimal form (until the
replay client of PR A runs the whole trace): one session of ~25k tokens is
pushed out of the GPU pool by `--others` sessions of the same size, the
server idles for `--gap` seconds (2-10 min bucket), then the session's next
turn (+`--new` tokens) is sent and its time to first token measured. Then
the write-behind's cost: `--itl-streams` short requests decode while a fresh
25k prompt prefills, and their inter-token latency (p50 / p99) is recorded.

Every server binary runs `--rounds` times, interleaved (A B / B A / ...).
Raw prompts (/v1/completions), greedy, so the replies are deterministic and
their md5s are printed for the base-vs-candidate check.

Eviction is sized and checked, per round (since the bella runs 1030-195012-05d7
and 1031-045801-3742): a server sizes its paged pool from the VRAM free at its
start, so on a desktop GPU the pool moves between starts (bella 7B f16, default
--parallel: 1570-1698 pages of 64). The session and its 3 others need 1656
pages; above that nothing pushed it out, and "resume" timed a prefix-cache hit
(main 0.1-0.3 s with no tier). So:
- Pressure: after the `--others` sessions (hashed, as before), unhashed
  pressure sessions (seeds 200+) are sent until the floor page counts of the
  session and everything after it exceed the pool the server logged
  ("KV: N pages x P"), so the LRU must have pushed the session out.
- Check: the GPU's own prefix-cache hit on the resume, `resume_cached` minus
  the tier's restored tokens over that request, must be 0, and a resume with
  nothing cached must take at least half the first turn's time (a cold
  prefill). A round that fails is VOID: printed with its reason, kept in the
  JSON (`void`), left out of every median, and the run exits 3. A tier that
  restored nothing after a real eviction is not void: that's a result.

usage: kv_resume_bench.py --model M --bin NAME=PATH[:ARGS] ... [--rounds 3]
       [--gap 120] [--tokens 25000] [--others 3] [--new 157] [--out FILE]
ARGS are extra `axon serve` flags, comma-separated (e.g.
cand=./axon:--cache-ram,6G).
"""

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time

WORDS = (
    "river stone lamp gate north south east west mill bridge field wheat "
    "copper glass lantern harbor market tower garden winter summer autumn "
    "spring candle ledger anchor rope sail cart horse wagon road valley hill "
    "forest cedar maple willow oak birch meadow brook pond island coast cliff"
).split()


def text(seed, n_words):
    r = random.Random(seed)
    out = []
    for i in range(n_words):
        out.append(r.choice(WORDS))
        if i % 12 == 11:
            out.append(f"{i}.")
    return " ".join(out) + " "


def post(port, body, stream=False, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    return c, c.getresponse()


def complete(port, prompt, n):
    """Whole reply: (text, usage)."""
    c, r = post(port, {"prompt": prompt, "max_tokens": n, "temperature": 0})
    v = json.loads(r.read())
    c.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {v}")
    return v["choices"][0]["text"], v["usage"]


def stream(port, prompt, n):
    """Streamed reply: (ttft s, token times, text, usage)."""
    t0 = time.perf_counter()
    c, r = post(port, {"prompt": prompt, "max_tokens": n, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}})
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]}")
    times, parts, usage = [], [], None
    for line in r:
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        d = line[6:]
        if d == b"[DONE]":
            break
        v = json.loads(d)
        if v.get("usage"):
            usage = v["usage"]
        for ch in v.get("choices", []):
            t = ch.get("text") or ""
            if t:
                times.append(time.perf_counter())
                parts.append(t)
    c.close()
    ttft = times[0] - t0 if times else float("nan")
    return ttft, times, "".join(parts), usage


def wait_up(port, proc, limit=900):
    """Until /health and /v1/models both answer 200 (a loading server
    answers 503), a poll a second."""
    t0 = time.time()
    last = None
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
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
                return
        except OSError as e:
            last = str(e)
        time.sleep(1)
    raise RuntimeError(f"server not ready after {limit} s (last: {last})")


def calibrate(port, tokens):
    """Words for about `tokens` prompt tokens (the model's tokenizer)."""
    probe = 2000
    _, u = complete(port, text(999, probe), 1)
    return int(probe * tokens / u["prompt_tokens"])


def metric(port, name):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/metrics")
    body = c.getresponse().read().decode()
    for line in body.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    return None


def itl_cell(port, streams, seed0, during=None):
    """`streams` short requests decode 256 tokens each (while `during()`
    runs, if given, started a second in): their inter-token gaps in ms,
    sorted, and their texts."""
    itl, outs = [], [None] * streams
    lock = threading.Lock()

    def short(i):
        _, ts, txt, _ = stream(port, text(seed0 + i, 40), 256)
        outs[i] = txt
        with lock:
            itl.extend((y - x) * 1e3 for x, y in zip(ts, ts[1:]))

    ths = [threading.Thread(target=short, args=(i,)) for i in range(streams)]
    for th in ths:
        th.start()
    extra = None
    if during:
        time.sleep(1.0)
        extra = during()
    for th in ths:
        th.join()
    itl.sort()
    return itl, outs, extra


def pct(xs, q):
    return xs[int(q * (len(xs) - 1))] if xs else None


def pool_pages(logp, limit=10):
    """(pages, tokens per page) of the server's paged pool, from its banner
    ("KV: N pages x P = T tokens"); None if it never says."""
    for _ in range(limit):
        m = re.search(r"KV: (\d+) pages x (\d+) = \d+ tokens", open(logp, errors="replace").read())
        if m:
            return int(m.group(1)), int(m.group(2))
        time.sleep(1)
    return None


def kept_pages(usage, page):
    """Pages a finished request's KV keeps in the pool, rounded down (an
    under-count, so pressure sized from it can only be more than needed)."""
    return (usage["prompt_tokens"] + usage.get("completion_tokens", 0) - 1) // page


def void_reason(res):
    """Why a round's resume wasn't from an evicted session, or None. The GPU's
    own prefix cache must hold none of it (tokens cached on the resume beyond
    what the RAM tier restored during it), and a resume with nothing cached
    must be a cold prefill (at least half the first turn's time)."""
    cached = res.get("resume_cached") or 0
    restored = res.get("resume_restored") or 0
    gpu = cached - restored
    if gpu > 0:
        return (f"not evicted: {gpu} of {res.get('resume_prompt')} prompt tokens still in the GPU pool "
                f"(cached {cached}, tier restored {restored}; pool {res.get('pool_pages')} pages)")
    if cached == 0 and res["ttft_resume"] < 0.5 * res["a1_s"]:
        return f"no cold prefill: ttft {res['ttft_resume']:.3f} s with nothing cached vs first turn {res['a1_s']:.3f} s"
    return None


def trace_stats(path):
    """AXON_SERVE_TRACE lines: the host time between consecutive steps while
    requests run (a step with requests running, then the next step: what the
    engine did outside the step's own passes: write-behind, allocation,
    admission), and the steps' own times."""
    try:
        lines = [json.loads(x) for x in open(path) if x.strip()]
    except OSError:
        return {}
    gaps, steps = [], []
    for x, y in zip(lines, lines[1:]):
        steps.append(x["ms"])
        if x.get("running", 0) > 0 and y.get("running", 0) > 0:
            gaps.append(y["t_ms"] - x["t_ms"] - x["ms"])
    gaps.sort()
    steps.sort()
    return {"trace_steps": len(lines), "trace_gap_max_ms": round(gaps[-1], 2) if gaps else None,
            "trace_gaps_over_20ms": sum(g > 20 for g in gaps),
            "trace_gap_top5_ms": [round(g, 1) for g in gaps[-5:]],
            "trace_step_p99_ms": round(pct(steps, 0.99), 2) if steps else None,
            "trace_step_max_ms": round(steps[-1], 2) if steps else None}


def one_round(name, path, extra, env, a, port, rnd):
    cmd = [path, "serve", a.model, "--device", "cuda", "--ctx", str(a.ctx), "--port", str(port)] + extra
    print(f"## {name}: {' '.join(cmd)} {env}", flush=True)
    logp = f"{a.out}.{name}.r{rnd}.server.log"
    tr = f"{a.out}.{name}.r{rnd}.trace.jsonl"
    log = open(logp, "w")
    e = dict(os.environ)
    e.update(env)
    if a.trace:
        e["AXON_SERVE_TRACE"] = tr
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=e)
    res = {"bin": name}
    try:
        wait_up(port, proc)
        res["startup_s"] = round(time.perf_counter() - t0, 3)
        md5 = hashlib.md5()
        # Right after start: decode while a tier pinned by a thread may
        # still be pinning (Alloc::Thread).
        itl0, outs0, _ = itl_cell(port, a.itl_streams, 300)
        res["itl0_p50_ms"], res["itl0_p99_ms"], res["itl0_max_ms"] = pct(itl0, 0.5), pct(itl0, 0.99), itl0[-1] if itl0 else None
        for o in outs0:
            md5.update((o or "").encode())
        if not a.startup_only:
            pool = pool_pages(logp)
            res["pool_pages"] = pool and pool[0]
            nw = calibrate(port, a.tokens)
            sessions = [text(100 + i, nw) for i in range(a.others + 1)]
            t = time.perf_counter()
            a1, ua = complete(port, sessions[0], 64)
            res["a1_s"] = time.perf_counter() - t
            res["a_tokens"] = ua["prompt_tokens"]
            md5.update(a1.encode())
            page = pool[1] if pool else 64
            kept = kept_pages(ua, page)
            for s_ in sessions[1:]:
                r, u = complete(port, s_, 64)
                kept += kept_pages(u, page)
                md5.update(r.encode())
            # Pressure: more sessions (not hashed: their count follows the
            # pool) until the session and what came after it can't all fit.
            extra = 0
            while pool and kept <= pool[0] and extra < 8:
                _, u = complete(port, text(200 + extra, nw), 64)
                kept += kept_pages(u, page)
                extra += 1
            res["pressure_extra"], res["pressure_pages"] = extra, kept
            time.sleep(a.gap)
            new = text(7, int(nw * a.new / a.tokens) + 1)
            restored0 = metric(port, "axon_kv_ram_restored_tokens_total")
            ttft, times, a2, u2 = stream(port, sessions[0] + a1 + "\n" + new, 64)
            restored1 = metric(port, "axon_kv_ram_restored_tokens_total")
            md5.update(a2.encode())
            res.update(ttft_resume=ttft, resume_prompt=u2 and u2["prompt_tokens"],
                       resume_cached=u2 and u2.get("prompt_tokens_details", {}).get("cached_tokens"),
                       resume_restored=None if restored1 is None else restored1 - (restored0 or 0))
            res["void"] = void_reason(res)
            # The write-behind's cost: short streams decode while a fresh long
            # prompt prefills (its pages written behind with the tier on).
            itl, outs, fresh = itl_cell(port, a.itl_streams, 500,
                                        during=lambda: complete(port, text(900, nw), 16)[0])
            for o in outs:
                md5.update((o or "").encode())
            md5.update(fresh.encode())
            res["itl_p50_ms"], res["itl_p99_ms"], res["itl_n"] = pct(itl, 0.5), pct(itl, 0.99), len(itl)
            res["itl_top8_ms"] = [round(x, 1) for x in itl[-8:]]
        res["restored_tokens"] = metric(port, "axon_kv_ram_restored_tokens_total")
        res["pinned_bytes"] = metric(port, "axon_kv_ram_pinned_bytes")
        res["md5"] = md5.hexdigest()
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if a.trace:
        res.update(trace_stats(tr))
    res["tier_lines"] = [x.strip() for x in open(logp) if "KV RAM tier" in x][:3]
    print("RESULT " + json.dumps(res), flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--bin", action="append", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--gap", type=float, default=120)
    p.add_argument("--tokens", type=int, default=25000)
    p.add_argument("--others", type=int, default=3)
    p.add_argument("--new", type=int, default=157)
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--itl-streams", type=int, default=4)
    p.add_argument("--port", type=int, default=18571)
    p.add_argument("--out", default="kv_resume")
    p.add_argument("--trace", action="store_true", help="AXON_SERVE_TRACE per server: inter-step gaps")
    p.add_argument("--startup-only", action="store_true", help="only the start-up time and the ITL cell after it")
    a = p.parse_args()
    bins = []
    for b in a.bin:
        # NAME=PATH[:ARG,ARG...][|ENV=V;ENV=V]
        spec, _, envs = b.partition("|")
        name, rest = spec.split("=", 1)
        path, _, args = rest.partition(":")
        env = dict(x.split("=", 1) for x in envs.split(";") if x)
        bins.append((name, path, [x for x in args.split(",") if x], env))
    rows = []
    for r in range(a.rounds):
        order = bins if r % 2 == 0 else bins[::-1]
        for name, path, extra, env in order:
            row = one_round(name, path, extra, env, a, a.port, r)
            row["round"] = r
            rows.append(row)
    voids = [x for x in rows if x.get("void")]
    print("\n## eviction check (a VOID round is left out of every median below)")
    for x in rows:
        print(f"{'VOID' if x.get('void') else 'OK  '} {x['bin']} r{x['round']}: pool {x.get('pool_pages')} pages, "
              f"pressure +{x.get('pressure_extra')} ({x.get('pressure_pages')} pages kept), "
              f"cached {x.get('resume_cached')}, restored {x.get('resume_restored')}, "
              f"ttft {x.get('ttft_resume')}" + (f": {x['void']}" if x.get("void") else ""))
    print("\n## summary (medians over valid rounds, raw per valid round)")
    keys = ["startup_s", "itl0_p50_ms", "itl0_p99_ms", "itl0_max_ms", "ttft_resume", "resume_cached",
            "resume_restored", "itl_p50_ms", "itl_p99_ms", "trace_gap_max_ms", "trace_gaps_over_20ms",
            "trace_step_p99_ms", "pinned_bytes", "pool_pages", "pressure_extra"]
    for name, _, _, _ in bins:
        rs = [x for x in rows if x["bin"] == name and not x.get("void")]
        print(f"VALID {name}: {len(rs)} of {sum(x['bin'] == name for x in rows)} rounds")
        parts = []
        for k in keys:
            vals = [x.get(k) for x in rs]
            got = [v for v in vals if v is not None]
            if got:
                r = lambda v: round(v, 3) if isinstance(v, float) else v
                parts.append(f"{k} {r(statistics.median(got))} {[r(v) for v in vals]}")
        print(f"SUMMARY {name}: " + "; ".join(parts) + f"; md5 {sorted(set(x['md5'] for x in rs))}")
    with open(a.out + ".json", "w") as f:
        json.dump(rows, f, indent=1)
    md5s = {x["md5"] for x in rows}
    print("MD5 " + ("SAME" if len(md5s) == 1 else f"DIFFER {sorted(md5s)}"))
    if voids:
        print(f"EVICTION VOID {len(voids)} of {len(rows)} rounds: the cell is short of its rounds")
    return 1 if len(md5s) != 1 else 3 if voids else 0


if __name__ == "__main__":
    sys.exit(main())
