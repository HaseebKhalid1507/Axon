#!/usr/bin/env python3
"""Interleaved throughput benchmark for axon binaries (and optionally ollama).

Runs each binary in turn, round after round, so drift in machine load hits all
of them equally, and reports the median of each metric:

  pp560   prompt eval tok/s, ~560-token prompt, T threads
  pp2100  prompt eval tok/s, ~2100-token prompt, T threads
  tg128   decode tok/s over 128 tokens, T threads
  tg128@1 decode tok/s over 128 tokens, 1 thread

Usage:
  scripts/bench.py [-m MODEL] [-t THREADS] [-r ROUNDS] [--load N] [--ollama MODEL] BIN [BIN ...]

--load N runs N busy-loop processes during the measurements, to see how each
binary behaves on a machine that isn't idle. --ollama NAME adds the local ollama
server (same prompts, same thread count) as a reference column.

Throughput depends on everything else running on the machine: compare
binaries within one run, not numbers across runs.
"""

import argparse
import json
import multiprocessing
import os
import re
import statistics
import subprocess
import sys
import urllib.request

SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank."
PROMPTS = {"pp560": " ".join([SENTENCE] * 40), "pp2100": " ".join([SENTENCE] * 150)}
DECODE_PROMPT = "The history of the Roman Empire"
RATE = re.compile(r"\(([0-9.]+) tok/s\)")


def run_axon(binary, model, prompt, n, threads):
    env = dict(os.environ, AXON_THREADS=str(threads))
    p = subprocess.run(
        [binary, "gen", model, prompt, "-n", str(n)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if p.returncode != 0:
        raise RuntimeError(f"{binary} exited {p.returncode}: {p.stderr[-300:]}")
    rates = {}
    for line in p.stderr.splitlines():
        m = RATE.search(line)
        if m and line.startswith("prompt eval"):
            rates["pp"] = float(m.group(1))
        elif m and line.startswith("decode"):
            rates["tg"] = float(m.group(1))
    return rates


def run_ollama(name, prompt, n, threads, salt):
    body = json.dumps({
        "model": name, "prompt": f"Note {salt}. {prompt}", "raw": True, "stream": False,
        "options": {"num_predict": n, "num_thread": threads, "num_gpu": 0,
                    "temperature": 0, "num_ctx": 4096},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    rates = {"pp": r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)}
    if r.get("eval_count", 0) > 1:
        rates["tg"] = r["eval_count"] / (r["eval_duration"] / 1e9)
    return rates


def burn():
    while True:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bins", nargs="+")
    ap.add_argument("-m", "--model", default="models/qwen2.5-0.5b-q4_k_m.gguf")
    ap.add_argument("-t", "--threads", type=int, default=18)
    ap.add_argument("-r", "--rounds", type=int, default=3)
    ap.add_argument("--load", type=int, default=0, help="busy-loop processes during the run")
    ap.add_argument("--ollama", default=None, help="ollama model name to include")
    ap.add_argument("--skip", default="", help="comma list of metrics to skip")
    a = ap.parse_args()

    skip = set(filter(None, a.skip.split(",")))
    metrics = [m for m in ["pp560", "pp2100", "tg128", "tg128@1"] if m not in skip]
    names = list(a.bins) + ([f"ollama:{a.ollama}"] if a.ollama else [])
    results = {n: {m: [] for m in metrics} for n in names}

    hogs = [multiprocessing.Process(target=burn, daemon=True) for _ in range(a.load)]
    for h in hogs:
        h.start()
    try:
        salt = 0
        for rnd in range(a.rounds):
            for metric in metrics:
                for name in names:
                    salt += 1
                    threads = 1 if metric.endswith("@1") else a.threads
                    if metric.startswith("pp"):
                        prompt, n, key = PROMPTS[metric], 2, "pp"
                    else:
                        prompt, n, key = DECODE_PROMPT, 128, "tg"
                    if name.startswith("ollama:"):
                        r = run_ollama(a.ollama, prompt, n, threads, salt)
                    else:
                        r = run_axon(name, a.model, prompt, n, threads)
                    if key in r:
                        results[name][metric].append(r[key])
            print(f"round {rnd + 1}/{a.rounds} done", file=sys.stderr)
    finally:
        for h in hogs:
            h.kill()

    print(f"threads={a.threads} rounds={a.rounds} load={a.load} model={a.model}")
    print("| binary | " + " | ".join(metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    for name in names:
        cells = []
        for m in metrics:
            v = results[name][m]
            cells.append(f"{statistics.median(v):.1f} ({min(v):.0f}–{max(v):.0f})" if v else "-")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
