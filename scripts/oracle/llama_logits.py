#!/usr/bin/env python3
"""Next-token top-K logprobs from llama.cpp (llama-server) and from axon, and
their agreement: the "within llama.cpp's own CPU/CUDA spread" gate.

    llama_logits.py dump-axon  --axon BIN --model M [-k 20] --out A.json [-- axon gen flags]
    llama_logits.py dump-llama --server BIN --model M [-k 20] --out L.json [-- server flags]
    llama_logits.py compare A.json B.json [C.json ...]   (every pair against the first)

Both sides see the same token ids: axon's (`axon logits` prints them), given to
llama-server's /completion as a token list (cache_prompt off, n_predict 1,
n_probs K: the softmax of the raw logits). The prompts are PROMPTS below (50).

compare: top-1 agreement and KL(P_a || P_b) over a's top K (b's logprob for a
token outside its own top K is floored at its K-th: an underestimate of the
gap, the same for every pair).
"""
import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "The three laws of thermodynamics are",
    "import numpy as np\nimport matplotlib.pyplot as plt\n\n",
    "Q: What is 17 times 23?\nA:",
    "The quick brown fox",
    "In 1969, Neil Armstrong",
    "To make a cup of tea, first",
    "SELECT name, age FROM users WHERE",
    "The mitochondria is",
    "Dear Sir or Madam,\n\nI am writing to",
    "The best way to learn a new language is",
    "fn main() {\n    let v: Vec<i32> =",
    "Photosynthesis converts",
    "The stock market crashed in",
    "Water boils at",
    "Shakespeare wrote",
    "The largest planet in our solar system is",
    "<|im_start|>user\nWhat is the speed of light?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a haiku about autumn.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nTranslate 'good morning' into German.<|im_end|>\n<|im_start|>assistant\n",
    "Le chat est assis sur",
    "Der schnellste Weg nach Berlin ist",
    "東京は日本の",
    "A neural network consists of",
    "The recipe calls for two cups of flour,",
    "#include <stdio.h>\n\nint main(void) {\n",
    "Monday, Tuesday, Wednesday,",
    "1, 1, 2, 3, 5, 8, 13,",
    "The opposite of hot is",
    "My favorite color is blue because",
    "The French Revolution began in",
    "In machine learning, overfitting means",
    "She opened the door and saw",
    "The theory of relativity states that",
    "Breaking news: scientists have discovered",
    "To install the package, run",
    "Roses are red, violets are",
    "The function returns None if",
    "Mount Everest is located in",
    "A haiku has three lines with",
    "The integral of x squared is",
    "Once the server starts, it listens on port",
    "The primary colors are red, yellow, and",
    "He was born in 1879 in Ulm, Germany, and",
    "git commit -m \"",
    "The patient presented with a fever and",
    "Chapter 1\n\nIt was a bright cold day in April, and",
    "The square root of 144 is",
]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def axon_ids_and_top(axon, model, prompt, k, extra):
    out = subprocess.run(
        [axon, "logits", model, prompt, "-k", str(k)] + extra,
        capture_output=True, text=True, check=True,
        env=dict(os.environ, AXON_LOGITS_DIGITS="6"),
    ).stdout
    ids = json.loads(re.search(r"prompt ids \(\d+\): (\[.*\])", out).group(1))
    top = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(-?[\d.]+|-?inf|NaN)\s+(-?[\d.]+|-?inf)\s+\"", line)
        if m:
            top.append([int(m.group(1)), float(m.group(3))])
    return ids, top[:k]


def dump_axon(a):
    res = []
    for p in PROMPTS:
        ids, top = axon_ids_and_top(a.axon, a.model, p, a.k, a.extra)
        res.append({"prompt": p, "ids": ids, "top": top})
        print(f"{len(res):2d} {top[0]}", file=sys.stderr)
    json.dump(res, open(a.out, "w"))


def dump_llama(a):
    # The ids: axon's tokenization (the dump-axon file), so both see the same input.
    ref = json.load(open(a.ids))
    port = free_port()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.dirname(a.server) + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [a.server, "-m", a.model, "--port", str(port), "-c", "2048", "-np", "1",
           "--no-warmup"] + a.extra
    log = open(a.out + ".server.log", "w")
    srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(600):
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if srv.poll() is not None:
                sys.exit(f"llama-server exited: see {a.out}.server.log")
            time.sleep(0.5)
        res = []
        for e in ref:
            body = json.dumps({"prompt": e["ids"], "n_predict": 1, "n_probs": a.k,
                               "temperature": 0, "cache_prompt": False}).encode()
            req = urllib.request.Request(url + "/completion", body,
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=600))
            cp = r["completion_probabilities"][0]
            top = [[t["id"], t["logprob"]] for t in cp["top_logprobs"]][: a.k]
            res.append({"prompt": e["prompt"], "ids": e["ids"], "top": top})
            print(f"{len(res):2d} {top[0]}", file=sys.stderr)
        json.dump(res, open(a.out, "w"))
    finally:
        srv.terminate()
        srv.wait()


def kl_top(pa, pb):
    """KL(a || b) over a's top K, b floored at its K-th logprob."""
    db = dict((i, lp) for i, lp in pb)
    floor = min(lp for _, lp in pb)
    kl = 0.0
    for i, lp in pa:
        p = math.exp(lp)
        kl += p * (lp - db.get(i, floor))
    return kl


def compare(files):
    runs = [(f, json.load(open(f))) for f in files]
    base_name, base = runs[0]
    print(f"{'pair':<70} {'top1':>7} {'KL mean':>10} {'KL p50':>10} {'KL max':>10}")
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            (fa, a), (fb, b) = runs[i], runs[j]
            assert [e["ids"] for e in a] == [e["ids"] for e in b], "different inputs"
            agree = sum(x["top"][0][0] == y["top"][0][0] for x, y in zip(a, b))
            kls = sorted(kl_top(x["top"], y["top"]) for x, y in zip(a, b))
            name = f"{os.path.basename(fa)} vs {os.path.basename(fb)}"
            print(f"{name:<70} {agree:>3}/{len(a):<3} {sum(kls) / len(kls):>10.2e} "
                  f"{kls[len(kls) // 2]:>10.2e} {kls[-1]:>10.2e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump-axon")
    d.add_argument("--axon", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("-k", type=int, default=20)
    d.add_argument("--out", required=True)
    d.add_argument("extra", nargs="*")
    l = sub.add_parser("dump-llama")
    l.add_argument("--server", required=True)
    l.add_argument("--model", required=True)
    l.add_argument("--ids", required=True, help="a dump-axon file: its token ids are used")
    l.add_argument("-k", type=int, default=20)
    l.add_argument("--out", required=True)
    l.add_argument("extra", nargs="*")
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"dump-axon": dump_axon, "dump-llama": dump_llama}.get(a.cmd, lambda a: compare(a.files))(a)


if __name__ == "__main__":
    main()
