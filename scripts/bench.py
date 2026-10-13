#!/usr/bin/env python3
"""Interleaved throughput benchmark for axon binaries (and optionally ollama / llama-server).

Runs each engine in turn, round after round, so drift in machine load hits all
of them equally, and reports the median (min-max) of each metric:

  pp560   prompt eval tok/s, ~560-token prompt, T threads
  pp2100  prompt eval tok/s, ~2100-token prompt, T threads
  tg128   decode tok/s over 128 tokens, T threads
  tg128@1 decode tok/s over 128 tokens, 1 thread

Usage:
  scripts/bench.py [-m MODEL] [-t THREADS] [-r ROUNDS] [--warmup N] [--load N]
                   [--ollama NAME] [--llama-server PATH] [--skip METRICS]
                   [--json PATH] [--dry-run] BIN [BIN ...]

BIN is a path to an axon binary (column label = basename) or
LABEL=PATH[,ENV=VAL...] to set the label and extra environment for that binary,
e.g. att-t1=target/release/axon,AXON_ATT_TILE=1 (no wrapper scripts needed).

--warmup N     discarded rounds before the timed ones (default 1), so model load,
               page cache and ollama/llama-server model load never land in a sample
--load N       N busy-loop processes during the measurements (a machine that isn't idle)
--ollama NAME  local ollama server (/api/generate, raw, num_gpu 0, same prompts and
               thread count, kept loaded via keep_alive) as a reference column
--llama-server PATH
               llama.cpp llama-server as a reference column; one server per thread
               count (T, and 1 for tg128@1), started lazily on a free localhost port,
               killed on exit (also on Ctrl-C)
--json PATH    write raw samples, medians and machine/run metadata
--dry-run      print the exact commands/requests instead of running anything

Prompts sent to servers are salted per request so no prompt cache can help.
Throughput depends on everything else running on the machine: compare
engines within one run, not numbers across runs. See scripts/sweep.py for a
per-engine thread-count sweep.
"""

import argparse
import datetime
import itertools
import json
import multiprocessing
import os
import platform
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank."
PROMPTS = {"pp560": " ".join([SENTENCE] * 40), "pp2100": " ".join([SENTENCE] * 150)}
DECODE_PROMPT = "The history of the Roman Empire"
RATE = re.compile(r"\(([0-9.]+) tok/s\)")
ALL_METRICS = ["pp560", "pp2100", "tg128", "tg128@1", "tg128@2k"]
# tg128@2k: decode 128 tokens after the ~2100-token prompt (attention over a long context).
DEFAULT_SKIP = {"tg128@2k"}
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
_SALT = itertools.count(int(time.time() * 1000) % 10**9)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def metric_job(metric, threads):
    """(prompt, n_predict, rate key, threads) for a metric name."""
    if metric.endswith("@1"):
        threads, metric = 1, metric[:-2]
    if metric.endswith("@2k"):
        return PROMPTS["pp2100"], int(metric[2:-3]), "tg", threads
    if metric.startswith("pp"):
        return PROMPTS[metric], 2, "pp", threads
    return DECODE_PROMPT, int(metric[2:]), "tg", threads


def post_json(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


class Axon:
    kind = "axon"

    def __init__(self, spec, model, dry_run=False):
        self.name, self.path, self.env = parse_bin(spec)
        self.model, self.dry_run = model, dry_run

    def run(self, prompt, n, threads, salt):
        env = dict(self.env, AXON_THREADS=str(threads))
        argv = [self.path, "gen", self.model, prompt, "-n", str(n)]
        if self.dry_run:
            print("  " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
                  + " " + shlex.join(argv))
            return {}
        p = subprocess.run(argv, capture_output=True, text=True,
                           env=dict(os.environ, **env), timeout=600)
        if p.returncode != 0:
            raise RuntimeError(f"{self.path} exited {p.returncode}: {p.stderr[-300:]}")
        rates = {}
        for line in p.stderr.splitlines():
            m = RATE.search(line)
            if m and line.startswith("prompt eval"):
                rates["pp"] = float(m.group(1))
            elif m and line.startswith("decode"):
                rates["tg"] = float(m.group(1))
        return rates

    def describe(self):
        return {"kind": self.kind, "path": os.path.abspath(self.path), "env": self.env}

    def close(self):
        pass


class Ollama:
    kind = "ollama"

    def __init__(self, model_name, dry_run=False, keep_alive="30m"):
        self.model_name, self.dry_run, self.keep_alive = model_name, dry_run, keep_alive
        self.name = f"ollama:{model_name}"

    def run(self, prompt, n, threads, salt):
        body = {
            "model": self.model_name, "prompt": f"Note {salt}. {prompt}", "raw": True,
            "stream": False, "keep_alive": self.keep_alive,
            "options": {"num_predict": n, "num_thread": threads, "num_gpu": 0,
                        "temperature": 0, "num_ctx": 4096},
        }
        if self.dry_run:
            print(f"  POST {OLLAMA_URL} {json.dumps(body)}")
            return {}
        r = post_json(OLLAMA_URL, body)
        rates = {"pp": r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)}
        if r.get("eval_count", 0) > 1:
            rates["tg"] = r["eval_count"] / (r["eval_duration"] / 1e9)
            if r["eval_count"] < n:
                log(f"warning: {self.name} stopped after {r['eval_count']}/{n} tokens")
        return rates

    def describe(self):
        return {"kind": self.kind, "model": self.model_name, "keep_alive": self.keep_alive}

    def close(self):
        pass


def die_with_parent():
    """preexec_fn: SIGKILL the child if this script dies, even by SIGKILL (Linux)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def llama_commit(path):
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
        m = re.search(r"commit ([0-9a-f]{7,})|\(([0-9a-f]{7,})\)", out.stdout + out.stderr)
        if m:
            return (m.group(1) or m.group(2))[:7]
    except (OSError, subprocess.SubprocessError):
        pass
    return "server"


class LlamaServer:
    """llama-server reference: one server per thread count, started lazily."""
    kind = "llama-server"

    def __init__(self, path, model, dry_run=False):
        self.path, self.model, self.dry_run = os.path.expanduser(path), model, dry_run
        self.commit = llama_commit(self.path)
        self.name = f"llama.cpp:{self.commit}"
        self.servers = {}  # threads -> (proc, port, logfile)

    def argv(self, threads, port):
        return [self.path, "-m", self.model, "-t", str(threads), "-tb", str(threads),
                "-c", "4096", "-np", "1", "-ngl", "0", "--no-webui",
                "--host", "127.0.0.1", "--port", str(port)]

    def server(self, threads):
        if threads in self.servers:
            return self.servers[threads][1]
        port = free_port()
        argv = self.argv(threads, port)
        if self.dry_run:
            print(f"  start: {shlex.join(argv)}  (wait for GET http://127.0.0.1:{port}/health)")
            self.servers[threads] = (None, port, None)
            return port
        logf = tempfile.NamedTemporaryFile("w+", prefix=f"llama-server-t{threads}-",
                                           suffix=".log", delete=False)
        proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True,
                                preexec_fn=die_with_parent)
        self.servers[threads] = (proc, port, logf)
        deadline = time.time() + 300
        while True:
            if proc.poll() is not None:
                logf.seek(0)
                raise RuntimeError(f"llama-server exited {proc.returncode}: {logf.read()[-500:]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200 and json.load(r).get("status") == "ok":
                        break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.time() > deadline:
                raise RuntimeError(f"llama-server t={threads} not healthy after 300 s (log {logf.name})")
            time.sleep(0.2)
        log(f"llama-server t={threads} up on :{port} (log {logf.name})")
        return port

    def run(self, prompt, n, threads, salt):
        port = self.server(threads)
        body = {"prompt": f"Note {salt}. {prompt}", "n_predict": n, "cache_prompt": False,
                "temperature": 0, "top_k": 1, "seed": 1}
        url = f"http://127.0.0.1:{port}/completion"
        if self.dry_run:
            print(f"  POST {url} {json.dumps(body)}")
            return {}
        t = post_json(url, body)["timings"]
        if t.get("prompt_n", 0) <= 0:
            raise RuntimeError(f"llama-server evaluated no prompt tokens: {t}")
        rates = {"pp": t["prompt_per_second"]}
        if t.get("predicted_n", 0) > 1:
            rates["tg"] = t["predicted_per_second"]
            if t["predicted_n"] < n:
                log(f"warning: {self.name} stopped after {t['predicted_n']}/{n} tokens")
        return rates

    def describe(self):
        return {"kind": self.kind, "path": self.path, "commit": self.commit,
                "argv": self.argv("N", "PORT")}

    def close(self, keep=()):
        for threads in [t for t in self.servers if t not in keep]:
            proc, _, logf = self.servers.pop(threads)
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            if logf is not None:
                logf.close()


def parse_bin(spec):
    """'PATH' or 'LABEL=PATH[,ENV=VAL...]' -> (label, path, env)."""
    head, *rest = spec.split(",")
    label, sep, path = head.partition("=")
    if not sep or "/" in label:
        label, path = os.path.basename(spec.rstrip("/")), spec
        rest = []
    env = {}
    for kv in rest:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise SystemExit(f"bad env assignment {kv!r} in {spec!r} (want ENV=VAL)")
        env[k] = v
    return label, path, env


def make_engines(a):
    engines = [Axon(spec, a.model, a.dry_run) for spec in a.bins]
    if a.ollama:
        engines.append(Ollama(a.ollama, a.dry_run))
    if a.llama_server:
        engines.append(LlamaServer(a.llama_server, a.model, a.dry_run))
    names = [e.name for e in engines]
    dups = {n for n in names if names.count(n) > 1}
    if dups:
        raise SystemExit(f"duplicate column labels {sorted(dups)}: use LABEL=PATH")
    return engines


def burn():
    while True:
        pass


class Hogs:
    def __init__(self, n):
        self.procs = [multiprocessing.Process(target=burn, daemon=True) for _ in range(n)]

    def __enter__(self):
        for h in self.procs:
            h.start()
        return self

    def __exit__(self, *exc):
        for h in self.procs:
            h.kill()


def sigterm_as_interrupt():
    """Make SIGTERM unwind like Ctrl-C so finally blocks kill the servers."""
    def handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handler)


def run_rounds(engines, jobs, rounds, warmup=1, dry_run=False, tag=""):
    """jobs: [(label, metric, threads)]. Interleaves engines within each job, jobs
    within each round. Returns {engine name: {label: [samples]}}; warmup rounds are
    run first and discarded."""
    results = {e.name: {label: [] for label, _, _ in jobs} for e in engines}
    for rnd in range(warmup + rounds):
        timed = rnd >= warmup
        what = f"round {rnd - warmup + 1}/{rounds}" if timed else f"warmup {rnd + 1}/{warmup}"
        if dry_run:
            print(f"# {tag}{what}")
        for label, metric, threads in jobs:
            prompt, n, key, threads = metric_job(metric, threads)
            for e in engines:
                r = e.run(prompt, n, threads, next(_SALT))
                if timed and key in r:
                    results[e.name][label].append(r[key])
        if not dry_run:
            log(f"{tag}{what} done")
    return results


def cell(v):
    return f"{statistics.median(v):.1f} ({min(v):.0f}–{max(v):.0f})" if v else "-"


def medians(results):
    return {name: {k: (statistics.median(v) if v else None) for k, v in by.items()}
            for name, by in results.items()}


def git_sha(path=REPO):
    try:
        sha = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty and sha else "") or None
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def metadata(a, engines, **extra):
    model = os.path.abspath(a.model)
    return {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "axon_git_sha": git_sha(),
        "threads": getattr(a, "threads", None),
        "rounds": a.rounds, "warmup": a.warmup, "load": a.load,
        "model": model,
        "model_size": os.path.getsize(model) if os.path.exists(model) else None,
        "uname": " ".join(platform.uname()),
        "cpu": cpu_model(), "nproc": os.cpu_count(),
        "argv": sys.argv,
        "engines": {e.name: e.describe() for e in engines},
        **extra,
    }


def write_json(path, meta, results, dry_run=False):
    doc = {"meta": meta, "samples": results, "medians": medians(results)}
    if dry_run:
        print(f"# would write {path}: keys meta, samples, medians")
        return
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    log(f"wrote {path}")


def add_common_args(ap):
    ap.add_argument("bins", nargs="*", help="PATH or LABEL=PATH[,ENV=VAL...]")
    ap.add_argument("-m", "--model", default="models/qwen2.5-0.5b-q4_k_m.gguf")
    ap.add_argument("-r", "--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="discarded rounds before timing")
    ap.add_argument("--load", type=int, default=0, help="busy-loop processes during the run")
    ap.add_argument("--ollama", default=None, help="ollama model name to include")
    ap.add_argument("--llama-server", default=None, metavar="PATH",
                    help="llama-server binary to include as a reference")
    ap.add_argument("--json", default=None, metavar="PATH", help="write raw results here")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap)
    ap.add_argument("-t", "--threads", type=int, default=18)
    ap.add_argument("--skip", default="", help="comma list of metrics to skip")
    ap.add_argument("--only", default="", help="comma list of metrics to run (overrides --skip)")
    a = ap.parse_args()

    skip = set(filter(None, a.skip.split(","))) | DEFAULT_SKIP
    only = [m for m in filter(None, a.only.split(","))]
    for m in only:
        if m not in ALL_METRICS:
            ap.error(f"unknown metric {m}; known: {', '.join(ALL_METRICS)}")
    metrics = only or [m for m in ALL_METRICS if m not in skip]
    engines = make_engines(a)
    if not engines:
        ap.error("nothing to benchmark: give BINs, --ollama or --llama-server")
    jobs = [(m, m, a.threads) for m in metrics]

    sigterm_as_interrupt()
    try:
        with Hogs(0 if a.dry_run else a.load):
            results = run_rounds(engines, jobs, a.rounds, a.warmup, a.dry_run)
    finally:
        for e in engines:
            e.close()

    print(f"threads={a.threads} rounds={a.rounds} warmup={a.warmup} load={a.load} model={a.model}")
    print("| binary | " + " | ".join(metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    for e in engines:
        print(f"| {e.name} | " + " | ".join(cell(results[e.name][m]) for m in metrics) + " |")
    if a.json:
        write_json(a.json, metadata(a, engines, metrics=metrics), results, a.dry_run)


if __name__ == "__main__":
    main()
