#!/usr/bin/env python3
"""Golden check: Axon against llama.cpp on whole models, from text in (stdlib only).

    scripts/golden.py run --axon BIN --llama-server BIN --models-dir D --models a.gguf,b.gguf
                          [--device cuda] [--n 64] [--threads 8] [--llama-ngl 0] --out golden.json
    scripts/golden.py table golden.json      a markdown table of a run

scripts/zoo.py feeds llama.cpp Axon's own token ids, so a tokenizer or chat-template bug passes it. This one starts
from text on both sides and compares, per model, against llama.cpp's llama-server (CPU unless --llama-ngl):
  (a) tokenize  TOK_PROMPTS as a plain prompt: `axon logits`' prompt ids vs /tokenize (add_special, no special
                parsing), exactly.
  (b) chat      CHAT (a system message and two user turns) as /v1/chat/completions would run it: Axon's rendering
                (`axon logits` with AXON_LOGITS_CHAT) vs llama.cpp's (/apply-template of the GGUF's own template,
                then /tokenize with add_special and special parsing, as llama-server's chat endpoint does),
                exactly. The dates Axon pins (chat.rs: Llama 3's "26 Jul 2024", harmony's HARMONY_DATE) go to the
                template as chat_template_kwargs, so today's date isn't a difference. An axon without
                AXON_LOGITS_CHAT says "n/a" here.
  (c) greedy    llama.cpp's greedy N tokens after (b)'s llama.cpp ids and after GEN_PROMPT, then Axon teacher-
                forced on the same ids (AXON_LOGITS_IDS + AXON_LOGITS_PREFIX): at each step, is Axon's top-1
                llama.cpp's token? A disagreement where llama.cpp's own top-2 are within TIE nats is a near-tie
                (quantization noise flips those); a clear one counts. Also the largest |logprob| gap of llama.cpp's
                token between the engines.
A model passes when (a) and (b) are equal (or (b) is n/a), (c) has no clear disagreement, top-1 agreement is at least
AGREE_MIN, and no step's logprob gap exceeds DLP_MAX. Exit 0 all pass, 1 any FAIL, 2 usage / setup error.
`--only SUBSTR` runs the matching models only (axon-nightly's bisect).
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

TOK_PROMPTS = [
    "Hello, world!",
    "  leading spaces,\ttabs\n\nand newlines\r\n",
    "naïve café 東京 🚀 — ñ ü ß",
    "def f(x):\n    return x ** 2  # 1234567890 3.14159",
    "<|im_start|>user [INST] <s> not specials when plain </s>",
]
SYSTEM = "You are a concise assistant. Answer in at most three sentences."
CHAT = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "And of Japan? Say why it became the capital."},
]
GEN_PROMPT = "The history of the printing press begins"
# Axon's pinned template dates (src/chat.rs: LLAMA3_SYSTEM_HEAD, HARMONY_DATE); llama.cpp's templates read them
# from these variables when set, else from the clock.
TEMPLATE_KWARGS = {"date_string": "26 Jul 2024", "current_date": "2025-08-05"}
TIE, AGREE_MIN, DLP_MAX = 0.1, 0.9, 1.0


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Llama:
    """A llama-server on a free port; stopped on exit."""

    def __init__(self, a, model, log):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.path.dirname(a.llama_server) + ":" + env.get("LD_LIBRARY_PATH", "")
        self.log = open(log, "w")
        self.p = subprocess.Popen([a.llama_server, "-m", model, "--port", str(self.port), "-c", "4096", "-np", "1",
                                   "--jinja", "--no-warmup", "-ngl", str(a.llama_ngl), "-t", str(a.threads)],
                                  env=env, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def wait(self, limit=600):
        t0 = time.time()
        while time.time() - t0 < limit:
            if self.p.poll() is not None:
                raise RuntimeError(f"llama-server exited {self.p.returncode}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001 (not up yet)
                pass
            time.sleep(0.5)
        raise RuntimeError("llama-server not healthy after %d s" % limit)

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)

    def tokenize(self, text, add_special, parse_special):
        return self.post("/tokenize", {"content": text, "add_special": add_special,
                                       "parse_special": parse_special})["tokens"]

    def greedy(self, ids, n):
        """llama.cpp's greedy continuation of ids: [(id, logprob, [[id, logprob] top-5])]."""
        r = self.post("/completion", {"prompt": ids, "n_predict": n, "temperature": 0, "n_probs": 5,
                                      "cache_prompt": False})
        out = []
        for cp in r.get("completion_probabilities", []):
            top = [[t["id"], t["logprob"]] for t in cp["top_logprobs"]]
            out.append((cp["id"], cp["logprob"], top))
        return out

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.log.close()


def axon(a, model, prompt, env, k=1, timeout=900):
    """`axon logits` stdout; raises with the tail of stderr on failure."""
    p = subprocess.run([a.axon, "logits", model, prompt, "-k", str(k), "--device", a.device], capture_output=True,
                       text=True, errors="replace", timeout=timeout, env=dict(os.environ, **env))
    if p.returncode != 0:
        raise RuntimeError(f"axon logits rc {p.returncode}: {p.stderr.strip()[-300:]}")
    return p.stdout


def ids_line(out, label):
    m = re.search(re.escape(label) + r" \(\d+\): (\[.*\])", out)
    return json.loads(m.group(1)) if m else None


def positions(out):
    """{position: [[id, logprob], ...]} from AXON_LOGITS_PREFIX output."""
    res, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"position (\d+):", line)
        if m:
            cur = int(m.group(1))
            res[cur] = []
            continue
        m = re.match(r"\s*(\d+)\s+(-?[\d.]+|-?inf|NaN)\s+(-?[\d.]+|-?inf|NaN)\s+\"", line)
        if m and cur is not None:
            res[cur].append([int(m.group(1)), float(m.group(3))])
    return res


def first_diff(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def strip_bos(ids, bos):
    i = 0
    while bos is not None and i < len(ids) and ids[i] == bos:
        i += 1
    return ids[i:], i


def one(a, name):
    path = os.path.join(a.models_dir, name)
    r = {"model": name, "checks": {}, "problems": []}
    if not os.path.exists(path):
        r["verdict"] = "missing"
        return r
    r["size_mb"] = round(os.path.getsize(path) / 1e6)
    work = tempfile.mkdtemp(prefix="golden-", dir=a.work)
    ll = Llama(a, path, os.path.join(a.work, name + ".llama.log"))
    try:
        ll.wait()
        l_tok = [ll.tokenize(t, True, False) for t in TOK_PROMPTS]
        # llama-server's chat endpoint: the template (BOS stripped when the vocab adds one), then add_special + parse
        rendered = ll.post("/apply-template", {"messages": CHAT, "chat_template_kwargs": TEMPLATE_KWARGS})["prompt"]
        l_chat = ll.tokenize(rendered, True, True)
        l_gen_ids = ll.tokenize(GEN_PROMPT, True, False)
        greedy = {"chat": ll.greedy(l_chat, a.n), "text": ll.greedy(l_gen_ids, a.n)}
        bos_probe = ll.tokenize("", True, False)
    except Exception as e:  # noqa: BLE001 (any llama.cpp failure: recorded, the model is not judged)
        r["verdict"] = "llama-error"
        r["problems"].append(f"llama.cpp: {e}")
        return r
    finally:
        ll.stop()
    bos = bos_probe[0] if bos_probe else None
    r["llama_rendered"] = rendered[:2000]
    # (a) tokenization
    tok = {"prompts": len(TOK_PROMPTS), "equal": 0, "diffs": []}
    try:
        for text, want in zip(TOK_PROMPTS, l_tok):
            got = ids_line(axon(a, path, text, {}), "prompt ids")
            if got == want:
                tok["equal"] += 1
            else:
                i = first_diff(got or [], want)
                tok["diffs"].append({"text": text, "at": i, "axon": (got or [])[max(0, (i or 0) - 2):(i or 0) + 4],
                                     "llama": want[max(0, (i or 0) - 2):(i or 0) + 4],
                                     "axon_len": len(got or []), "llama_len": len(want)})
        tok["pass"] = tok["equal"] == tok["prompts"]
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        tok.update(error=str(e)[-300:])
        tok["pass"] = False
    r["checks"]["tokenize"] = tok
    # (b) chat rendering
    chat = {}
    body = os.path.join(work, "chat.json")
    json.dump({"messages": CHAT}, open(body, "w"))
    try:
        out = axon(a, path, "x", {"AXON_LOGITS_CHAT": body})
        got = ids_line(out, "chat prompt ids")
        if got is None:
            chat.update(status="n/a", why="this axon has no AXON_LOGITS_CHAT")
        else:
            g, gb = strip_bos(got, bos)
            w, wb = strip_bos(l_chat, bos)
            i = first_diff(g, w)
            chat.update(status="equal" if i is None and gb == wb else "DIFF", axon_len=len(got), llama_len=len(l_chat),
                        axon_bos=gb, llama_bos=wb)
            if i is not None:
                chat.update(at=i, axon=g[max(0, i - 3):i + 6], llama=w[max(0, i - 3):i + 6])
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        chat.update(status="ERROR", why=str(e)[-300:])
    chat["pass"] = chat["status"] in ("equal", "n/a")
    r["checks"]["chat"] = chat
    # (c) greedy top-1, teacher-forced on llama.cpp's path
    g = {"n": a.n, "steps": 0, "agree": 0, "near_tie": 0, "clear": [], "max_dlp": 0.0}
    try:
        for kind, prompt_ids in (("chat", l_chat), ("text", l_gen_ids)):
            path_ = greedy[kind]
            if not path_:
                continue
            seq = prompt_ids + [t for t, _, _ in path_[:-1]]
            out = axon(a, path, "x", {"AXON_LOGITS_IDS": ",".join(map(str, seq)),
                                      "AXON_LOGITS_PREFIX": str(len(prompt_ids)), "AXON_LOGITS_DIGITS": "6"}, k=5)
            pos = positions(out)
            for i, (tid, lp, top) in enumerate(path_):
                at = pos.get(len(prompt_ids) - 1 + i)
                if not at:
                    raise RuntimeError(f"{kind}: no axon logits at position {len(prompt_ids) - 1 + i}")
                g["steps"] += 1
                mine = dict((t, p) for t, p in at)
                if tid in mine:
                    g["max_dlp"] = max(g["max_dlp"], abs(mine[tid] - lp))
                if at[0][0] == tid:
                    g["agree"] += 1
                    continue
                gap = top[0][1] - dict((t, p) for t, p in top).get(at[0][0], top[-1][1])
                if gap <= TIE:
                    g["near_tie"] += 1
                else:
                    g["clear"].append({"kind": kind, "step": i, "llama": tid, "axon": at[0][0],
                                       "llama_gap": round(gap, 3)})
        g["agree_frac"] = round(g["agree"] / g["steps"], 4) if g["steps"] else None
        g["max_dlp"] = round(g["max_dlp"], 4)
        g["pass"] = bool(g["steps"]) and not g["clear"] and g["agree_frac"] >= AGREE_MIN and g["max_dlp"] <= DLP_MAX
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        g.update(error=str(e)[-300:])
        g["pass"] = False
    r["checks"]["greedy"] = g
    r["verdict"] = "pass" if all(c["pass"] for c in r["checks"].values()) else "FAIL"
    return r


def llama_version(a):
    try:
        p = subprocess.run([a.llama_server, "--version"], capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(a.llama_server)))
        return (p.stdout + p.stderr).strip().splitlines()[-2:]
    except Exception as e:  # noqa: BLE001
        return [f"unknown: {e}"]


def run(a):
    a.work = os.path.dirname(os.path.abspath(a.out)) or "."
    models = [m for m in a.models.split(",") if m and (not a.only or any(s in m for s in a.only))]
    if not models:
        print("golden: no models", file=sys.stderr)
        return 2
    ver = subprocess.run([a.axon, "--version"], capture_output=True, text=True).stdout.strip()
    doc = {"axon": a.axon, "axon_version": ver, "llama_server": a.llama_server, "llama_version": llama_version(a),
           "device": a.device, "n": a.n, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "models": []}
    for m in models:
        t0 = time.time()
        r = one(a, m)
        r["secs"] = round(time.time() - t0, 1)
        doc["models"].append(r)
        c = r["checks"]
        print(json.dumps({"model": m, "verdict": r["verdict"],
                          "tokenize": c.get("tokenize", {}).get("equal"),
                          "chat": c.get("chat", {}).get("status"),
                          "agree": c.get("greedy", {}).get("agree_frac"),
                          "clear": len(c.get("greedy", {}).get("clear", [])),
                          "max_dlp": c.get("greedy", {}).get("max_dlp"), "secs": r["secs"],
                          "problems": r["problems"]}), flush=True)
        json.dump(doc, open(a.out, "w"), indent=1)
    v = [r["verdict"] for r in doc["models"]]
    print(f"GOLDEN {len(v)} models: pass {v.count('pass')}, FAIL {v.count('FAIL')}, "
          f"missing {v.count('missing')}, llama-error {v.count('llama-error')}", flush=True)
    return 1 if "FAIL" in v else 0


def table(a):
    doc = json.load(open(a.file))
    print(f"axon {doc['axon_version']} vs llama.cpp {' '.join(doc['llama_version'])}, {doc['device']}, "
          f"greedy n={doc['n']}\n")
    print("| model | verdict | tokenize | chat | greedy top-1 | clear / near-tie | max dlogprob |")
    print("|---|---|---|---|---:|---:|---:|")
    for r in doc["models"]:
        c = r["checks"]
        t, ch, g = c.get("tokenize", {}), c.get("chat", {}), c.get("greedy", {})
        print(f"| {r['model']} | {r['verdict']} | {t.get('equal', '')}/{t.get('prompts', '')} | "
              f"{ch.get('status', '')} | {g.get('agree', '')}/{g.get('steps', '')} | "
              f"{len(g.get('clear', []))} / {g.get('near_tie', '')} | {g.get('max_dlp', '')} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--axon", required=True)
    r.add_argument("--llama-server", required=True)
    r.add_argument("--models-dir", required=True)
    r.add_argument("--models", required=True, help="comma-separated GGUF file names in --models-dir")
    r.add_argument("--only", action="append", help="run the models whose name contains this (repeatable)")
    r.add_argument("--device", default="cuda")
    r.add_argument("--n", type=int, default=64)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--llama-ngl", type=int, default=0)
    r.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("file")
    a = ap.parse_args()
    if a.cmd == "table":
        table(a)
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
