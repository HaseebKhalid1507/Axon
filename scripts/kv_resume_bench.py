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

usage: kv_resume_bench.py --model M --bin NAME=PATH[:ARGS] ... [--rounds 3]
       [--gap 120] [--tokens 25000] [--others 3] [--new 157] [--out FILE]
ARGS are extra `axon serve` flags, comma-separated (e.g.
cand=./axon:--cache-ram,6G).
"""

import argparse
import hashlib
import http.client
import json
import random
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


def wait_up(port, proc, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("server did not come up")


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


def one_round(name, path, extra, a, port):
    cmd = [path, "serve", a.model, "--device", "cuda", "--ctx", str(a.ctx), "--port", str(port)] + extra
    print(f"## {name}: {' '.join(cmd)}", flush=True)
    log = open(f"{a.out}.{name}.server.log", "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    res = {"bin": name}
    try:
        wait_up(port, proc)
        nw = calibrate(port, a.tokens)
        sessions = [text(100 + i, nw) for i in range(a.others + 1)]
        md5 = hashlib.md5()
        t = time.perf_counter()
        a1, ua = complete(port, sessions[0], 64)
        res["a1_s"] = time.perf_counter() - t
        res["a_tokens"] = ua["prompt_tokens"]
        md5.update(a1.encode())
        for s in sessions[1:]:
            r, _ = complete(port, s, 64)
            md5.update(r.encode())
        time.sleep(a.gap)
        new = text(7, int(nw * a.new / a.tokens) + 1)
        ttft, times, a2, u2 = stream(port, sessions[0] + a1 + "\n" + new, 64)
        md5.update(a2.encode())
        res.update(ttft_resume=ttft, resume_prompt=u2 and u2["prompt_tokens"],
                   resume_cached=u2 and u2.get("prompt_tokens_details", {}).get("cached_tokens"))
        # The write-behind's cost: short streams decode while a fresh long
        # prompt prefills (its pages written behind with the tier on).
        itl, outs = [], [None] * a.itl_streams

        def short(i):
            _, ts, txt, _ = stream(port, text(500 + i, 40), 256)
            outs[i] = txt
            itl.extend((y - x) * 1e3 for x, y in zip(ts, ts[1:]))

        ths = [threading.Thread(target=short, args=(i,)) for i in range(a.itl_streams)]
        for th in ths:
            th.start()
        time.sleep(1.0)
        fresh, _ = complete(port, text(900, nw), 16)
        for th in ths:
            th.join()
        for o in outs:
            md5.update((o or "").encode())
        md5.update(fresh.encode())
        itl.sort()
        res["itl_p50_ms"] = statistics.median(itl) if itl else None
        res["itl_p99_ms"] = itl[int(0.99 * (len(itl) - 1))] if itl else None
        res["itl_n"] = len(itl)
        res["restored_tokens"] = metric(port, "axon_kv_ram_restored_tokens_total")
        res["pinned_bytes"] = metric(port, "axon_kv_ram_pinned_bytes")
        res["md5"] = md5.hexdigest()
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
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
    a = p.parse_args()
    bins = []
    for b in a.bin:
        name, rest = b.split("=", 1)
        path, _, args = rest.partition(":")
        bins.append((name, path, [x for x in args.split(",") if x]))
    rows = []
    for r in range(a.rounds):
        order = bins if r % 2 == 0 else bins[::-1]
        for name, path, extra in order:
            row = one_round(name, path, extra, a, a.port)
            row["round"] = r
            rows.append(row)
    print("\n## summary (medians over rounds)")
    for name, _, _ in bins:
        rs = [x for x in rows if x["bin"] == name]
        med = lambda k: statistics.median(x[k] for x in rs if x.get(k) is not None)
        print(f"SUMMARY {name}: ttft_resume p50 {med('ttft_resume'):.3f} s "
              f"(raw {[round(x['ttft_resume'], 3) for x in rs]}), cached {[x['resume_cached'] for x in rs]} "
              f"of {[x['resume_prompt'] for x in rs]}, itl p50 {med('itl_p50_ms'):.2f} ms "
              f"p99 {med('itl_p99_ms'):.2f} ms (raw p99 {[round(x['itl_p99_ms'], 2) for x in rs]}), "
              f"md5 {sorted(set(x['md5'] for x in rs))}")
    with open(a.out + ".json", "w") as f:
        json.dump(rows, f, indent=1)
    md5s = {x["md5"] for x in rows}
    print("MD5 " + ("SAME" if len(md5s) == 1 else f"DIFFER {sorted(md5s)}"))
    return 0 if len(md5s) == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
