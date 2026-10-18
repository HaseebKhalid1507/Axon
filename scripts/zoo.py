#!/usr/bin/env python3
"""Model-zoo sweep: GGUFs from the wild either match llama.cpp or are refused clearly (stdlib only).

    scripts/zoo.py fetch --dir D                 download ZOO into D (resumable), write D/<file>.sha256
    scripts/zoo.py run --axon BIN --llama-server BIN --dir D [--models-dir M] [--out zoo.json]
                       [--device cpu] [--only SUBSTR] [--prompts N]
    scripts/zoo.py table zoo.json                the markdown table for docs/hardening.md

For every file:
  1. `axon info`: rc and the first lines.
  2. `axon gen -n 64` greedy on a fixed prompt: rc, a panic, garbage (mostly
     unprintable or replacement characters, or one token repeated end to end).
  3. Next-token top-20 logprobs from axon (`axon logits`) and from llama.cpp's
     CPU llama-server, given axon's token ids, on --prompts prompts of
     scripts/oracle/llama_logits.py: top-1 agreement and KL(axon || llama).
     Plus llama.cpp's greedy 64 tokens from the same ids, against axon's text
     (characters in common from the start).
A file Axon refuses must exit non-zero with one clear line naming what is
unsupported (no panic, no backtrace, under 400 characters), in info, gen and
logits alike. Verdicts:
  match    runs, mean KL <= 0.02 against llama.cpp and every top-1 disagreement is
           a near-tie (llama.cpp's own top-2 within 0.1 nats)
  drift    runs, but outside that (look at it)
  refused  a clean refusal, as expected (`expect` in ZOO says which)
  BUG      a panic, garbage, an unclear refusal, a refusal where `run` was
           expected, or a run where `refuse` was
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "oracle"))
import llama_logits as LL  # noqa: E402

# (repo, file, expect, why).  "local:" repos are files already in models/.
ZOO = [
    ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "local:qwen2.5-0.5b-q4_k_m.gguf", "run", "Qwen official, Q4_K_M"),
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-UD-Q4_K_XL.gguf", "refuse",
     "unsloth dynamic 2.0 mix (IQ4_XS in some layers)"),
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_1.gguf", "run", "Q4_1"),
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-IQ4_NL.gguf", "run", "IQ4_NL (supported since #72)"),
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-BF16.gguf", "refuse", "BF16 weights"),
    ("lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF", "Qwen2.5-0.5B-Instruct-Q6_K.gguf", "run", "Q6_K"),
    ("mradermacher/Qwen2.5-0.5B-Instruct-i1-GGUF", "Qwen2.5-0.5B-Instruct.i1-Q4_K_M.gguf", "run",
     "imatrix Q4_K_M"),
    ("mradermacher/Qwen2.5-0.5B-Instruct-i1-GGUF", "Qwen2.5-0.5B-Instruct.i1-IQ2_M.gguf", "refuse", "IQ2_M"),
    ("mradermacher/Qwen2.5-0.5B-Instruct-i1-GGUF", "Qwen2.5-0.5B-Instruct.i1-Q4_0_4_4.gguf", "refuse",
     "Q4_0_4_4 (old ARM repack type, dropped by llama.cpp)"),
    ("bartowski/SmolLM2-360M-Instruct-GGUF", "SmolLM2-360M-Instruct-Q8_0.gguf", "refuse",
     "llama arch, but the smollm pre-tokenizer"),
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "run",
     "2023 converter (TheBloke)"),
    ("QuantFactory/gpt2-GGUF", "gpt2.Q8_0.gguf", "refuse", "gpt2 architecture"),
    ("ggml-org/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf", "run", "ggml-org gemma3"),
    ("bartowski/google_gemma-3-1b-it-GGUF", "local:google_gemma-3-1b-it-Q4_K_M.gguf", "run", "bartowski gemma3"),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "local:Llama-3.2-1B-Instruct-Q4_K_M.gguf", "run", "llama3.2"),
    ("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "local:Qwen2.5-0.5B-Instruct-Q2_K.gguf", "run",
     "Q2_K file: Q3_K + IQ4_NL fallback tensors (896 columns aren't whole 256-blocks); since #72"),
    ("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "local:Qwen2.5-0.5B-Instruct-Q3_K_M.gguf", "run",
     "Q3_K_M (+ IQ4_NL fallback tensors); since #72"),
    ("Qwen/Qwen3-0.6B-GGUF", "local:Qwen3-0.6B-Q8_0.gguf", "run", "Qwen official Qwen3 Q8_0"),
]

GEN_PROMPT = "The history of the printing press begins"
KL_MAX, TIE = 0.02, 0.1


def path_of(e, a):
    f = e[1]
    return os.path.join(a.models_dir, f[6:]) if f.startswith("local:") else os.path.join(a.dir, f)


def fetch(a):
    os.makedirs(a.dir, exist_ok=True)
    for repo, f, _, _ in ZOO:
        if f.startswith("local:"):
            continue
        p = os.path.join(a.dir, f)
        if os.path.exists(p + ".sha256"):
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{f}"
        print("fetch", url, flush=True)
        subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "5", "-o", p, url], check=True)
        h = sha256(p)
        open(p + ".sha256", "w").write(f"{h}  {f}\n")


def sha256(p):
    side = p + ".sha256"
    if os.path.exists(side):
        return open(side).read().split()[0]
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def gguf_meta(p, want=("general.quantization_version", "general.file_type", "tokenizer.ggml.pre",
                        "general.name")):
    """GGUF version and a few metadata values, read from the header (arrays skipped)."""
    import struct
    out = {}
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(p, "rb") as f:
        if f.read(4) != b"GGUF":
            return {"gguf": "not GGUF"}
        ver, = struct.unpack("<I", f.read(4))
        out["gguf"] = ver
        _, nkv = struct.unpack("<QQ", f.read(16))

        def s():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def val(t):
            if t == 8:
                return s()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et == 8:
                    for _ in range(n):
                        s()
                else:
                    f.seek(sizes[et] * n, 1)
                return None
            return struct.unpack("<" + fmt[t], f.read(sizes[t]))[0]
        for _ in range(min(nkv, 10000)):
            k = s()
            t, = struct.unpack("<I", f.read(4))
            v = val(t)
            if k in want:
                out[k.split(".")[-1]] = v
    return out


def run(cmd, timeout=900, env=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout,
                           env=dict(os.environ, **(env or {})))
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return "timeout", (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (
            e.stdout or ""), "timeout"


def panicked(s):
    return bool(re.search(r"panicked at|RUST_BACKTRACE|stack backtrace|SIGSEGV|core dumped", s))


def refusal_line(rc, out, err):
    """The refusal message if it's clean (one short line naming the problem), else None."""
    if rc in (0, "timeout") or panicked(out + err):
        return None
    lines = [ln.strip() for ln in err.strip().splitlines() if ln.startswith("axon: ")]
    if len(lines) != 1:
        return None
    return lines[0] if len(lines[0]) < 400 else None


def garbage(text):
    """Why text looks broken, or None."""
    t = text.strip()
    if not t:
        return "empty output"
    bad = sum(1 for c in t if c == "\ufffd" or (ord(c) < 32 and c not in "\n\t\r"))
    if bad > len(t) * 0.1:
        return f"{bad}/{len(t)} unprintable or replacement chars"
    w = t.split()
    if len(w) >= 16 and len(set(w)) <= 2:
        return f"degenerate repetition: {w[:4]}"
    return None


def gen_text(out):
    """axon gen's stdout is the prompt then the completion."""
    return out[len(GEN_PROMPT):] if out.startswith(GEN_PROMPT) else out


def llama_side(a, model, ids_list):
    """llama-server (CPU): top-K after each prompt + greedy 64 tokens after GEN ids. None if it can't load."""
    import socket
    import time
    port = LL.free_port()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.dirname(a.llama_server) + ":" + env.get("LD_LIBRARY_PATH", "")
    log = open(os.path.join(a.work, os.path.basename(model) + ".llama.log"), "w")
    srv = subprocess.Popen([a.llama_server, "-m", model, "--port", str(port), "-c", "2048", "-np", "1",
                            "--no-warmup", "-ngl", "0", "-t", str(a.threads)],
                           env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):
            if srv.poll() is not None:
                return None, f"llama-server exited {srv.returncode}"
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            return None, "llama-server not healthy"

        def post(body):
            req = urllib.request.Request(url + "/completion", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=600))
        tops = []
        for ids in ids_list[:-1]:
            r = post({"prompt": ids, "n_predict": 1, "n_probs": 20, "temperature": 0, "cache_prompt": False})
            cp = r["completion_probabilities"][0]
            tops.append([[t["id"], t["logprob"]] for t in cp["top_logprobs"]][:20])
        g = post({"prompt": ids_list[-1], "n_predict": 64, "temperature": 0, "top_k": 1,
                  "cache_prompt": False, "ignore_eos": False})
        return {"tops": tops, "gen": g.get("content", "")}, None
    finally:
        srv.terminate()
        srv.wait()
        _ = socket


def common_prefix(x, y):
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def one(a, e):
    repo, f, expect, why = e
    p = path_of(e, a)
    r = {"repo": repo, "file": os.path.basename(p), "expect": expect, "why": why}
    if not os.path.exists(p):
        r.update(verdict="missing")
        return r
    r["size_mb"] = round(os.path.getsize(p) / 1e6)
    r["sha256"] = sha256(p)
    try:
        r["meta"] = gguf_meta(p)
    except Exception as ex:  # noqa: BLE001
        r["meta"] = {"error": repr(ex)}
    env = {"AXON_DEVICE": a.device} if a.device else {}
    rc, out, err = run([a.axon, "info", p], env=env)
    r["info_rc"] = rc
    r["info"] = (out + err).strip().splitlines()[:6]
    m = re.search(r"types? [^\n]*", out + err)
    r["types"] = m.group(0)[:200] if m else ""
    problems = []
    if panicked(out + err):
        problems.append("info panicked")
    grc, gout, gerr = run([a.axon, "gen", p, GEN_PROMPT, "-n", "64", "--device", a.device or "cpu"], env=env)
    r["gen_rc"] = grc
    if panicked(gout + gerr):
        problems.append("gen panicked")
    if grc != 0:
        msg = refusal_line(grc, gout, gerr)
        r["refusal"] = msg
        r["stderr"] = gerr.strip()[-600:]
        lrc, lout, lerr = run([a.axon, "logits", p, "Hello", "-k", "5"], env=env)
        lmsg = refusal_line(lrc, lout, lerr)
        if msg is None:
            problems.append(f"unclear refusal (rc {grc}): {gerr.strip()[-300:]!r}")
        elif lmsg is None:
            problems.append(f"logits: no clean refusal (rc {lrc}): {(lout + lerr).strip()[-300:]!r}")
        if expect == "run":
            problems.append("refused, expected to run")
        if rc == 0 and expect == "refuse":
            r["note"] = "info prints the file (fine: info only reads the header)"
        r["verdict"] = "BUG" if problems else "refused"
        r["problems"] = problems
        return r
    text = gen_text(gout)
    r["gen"] = text[:300]
    g = garbage(text)
    if g:
        problems.append("garbage: " + g)
    if expect == "refuse":
        problems.append("ran, expected a refusal")
    # Logits vs llama.cpp.
    prompts = LL.PROMPTS[: a.prompts]
    ids_list, tops = [], []
    for q in prompts + [GEN_PROMPT]:
        try:
            ids, top = LL.axon_ids_and_top(a.axon, p, q, 20, ["--device", a.device or "cpu"])
        except subprocess.CalledProcessError as ex:
            problems.append(f"logits failed rc {ex.returncode}: {(ex.stderr or '')[-200:]!r}")
            break
        ids_list.append(ids)
        tops.append(top)
    if len(ids_list) == len(prompts) + 1:
        ll, why_not = llama_side(a, p, ids_list)
        if ll is None:
            r["llama"] = why_not
        else:
            n = len(prompts)
            agree = sum(tops[i][0][0] == ll["tops"][i][0][0] for i in range(n))
            kls = [LL.kl_top(tops[i], ll["tops"][i]) for i in range(n)]
            r["top1"] = f"{agree}/{n}"
            # A disagreement where llama.cpp's top-2 are within TIE nats is a near-tie
            # (quantization noise flips it); a clear one counts against the file.
            clear = []
            for i in range(n):
                lt, at = ll["tops"][i], tops[i]
                if at[0][0] != lt[0][0]:
                    d = dict((t, lp) for t, lp in lt)
                    gap = lt[0][1] - d.get(at[0][0], lt[-1][1])
                    r.setdefault("disagree", []).append({"prompt": prompts[i], "axon": at[0][0],
                                                         "llama": lt[0][0], "llama_gap": round(gap, 3)})
                    if gap > TIE:
                        clear.append(i)
            r["clear_disagree"] = len(clear)
            r["kl_mean"] = sum(kls) / n
            r["kl_max"] = max(kls)
            r["llama_gen"] = ll["gen"][:300]
            r["gen_common_chars"] = common_prefix(text, ll["gen"])
            if clear or r["kl_mean"] > KL_MAX:
                r["drift"] = True
    r["problems"] = problems
    r["verdict"] = "BUG" if problems else ("drift" if r.get("drift") else ("match" if "top1" in r else "runs"))
    return r


def do_run(a):
    a.work = os.path.dirname(os.path.abspath(a.out)) or "."
    res = []
    if os.path.exists(a.out) and a.resume:
        res = json.load(open(a.out))
    done = {x["file"] for x in res}
    for e in ZOO:
        if a.only and a.only not in e[1]:
            continue
        if os.path.basename(e[1].replace("local:", "")) in done:
            continue
        r = one(a, e)
        res.append(r)
        print(json.dumps({k: r.get(k) for k in ("file", "verdict", "top1", "kl_mean", "gen_common_chars",
                                                 "refusal", "problems")}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    bugs = [x for x in res if x["verdict"] in ("BUG",)]
    print(f"ZOO {len(res)} files: " + ", ".join(f"{v} {sum(x['verdict'] == v for x in res)}"
                                              for v in ("match", "drift", "runs", "refused", "BUG", "missing")))
    return 1 if bugs else 0


def table(a):
    res = json.load(open(a.file))
    print("| file | source | MB | GGUF v / quant v / pre | tensor types | expect | verdict | top-1 | KL mean "
          "| gen chars = llama.cpp | refusal / note |")
    print("|---|---|---:|---|---|---|---|---:|---:|---:|---|")
    for r in res:
        kl = f"{r['kl_mean']:.1e}" if "kl_mean" in r else ""
        note = r.get("refusal") or "; ".join(r.get("problems", [])) or r.get("llama", "") or r.get("why", "")
        note = re.sub(r"^axon: [^:]*\.gguf: ", "", note)
        if r.get("refusal"):
            note = re.sub(r" \(supported: [^)]*\)", "", note)
        m = r.get("meta", {})
        mv = f"{m.get('gguf', '')} / {m.get('quantization_version', '-')} / {m.get('pre', '-')}"
        ty = re.sub(r"^types? ", "", r.get("types", "")).split(", backend")[0]
        print(f"| {r['file']} | {r['repo'].split('/')[0]} | {r.get('size_mb', '')} | {mv} | {ty} | {r['expect']} | "
              f"{r['verdict']} | {r.get('top1', '')} | {kl} | {r.get('gen_common_chars', '')} | "
              f"{note.replace('|', '/')[:220]} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dir", required=True)
    r = sub.add_parser("run")
    r.add_argument("--axon", required=True)
    r.add_argument("--llama-server", required=True)
    r.add_argument("--dir", required=True)
    r.add_argument("--models-dir", default=os.path.join(HERE, "..", "models"))
    r.add_argument("--out", default="zoo.json")
    r.add_argument("--device", default="cpu")
    r.add_argument("--only")
    r.add_argument("--prompts", type=int, default=16)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--resume", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("file")
    a = ap.parse_args()
    if a.cmd == "fetch":
        return fetch(a)
    if a.cmd == "table":
        return table(a)
    return do_run(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
