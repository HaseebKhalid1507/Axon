#!/usr/bin/env python3
"""Model-zoo sweep: GGUFs from the wild either match llama.cpp or are refused clearly (stdlib only).

    scripts/zoo.py fetch --dir D                 download ZOO into D (resumable), write D/<file>.sha256
    scripts/zoo.py run --axon BIN --llama-server BIN --dir D [--models-dir M] [--out zoo.json]
                       [--device cpu] [--only SUBSTR] [--prompts N]
    scripts/zoo.py table zoo.json                the markdown table for docs/hardening.md
    scripts/zoo.py audit-table LOG|zoo.json...   the compatibility-audit table (docs/compat-audit-2026-11-13.md)

  --list FILE (fetch, run): a JSON list of {repo, file, sha256, expect, why, ...} instead of ZOO (the audit's
  scripts/examples/compat-audit/ollama-top30.json); --set golden: the nightly golden-model set (GOLDEN below).
  run --fetch-each: download each missing file just before it runs (sha256 checked), --delete-after removes it
  after, and the next file downloads while one runs. --sparse-first: for an `expect: refuse` entry, first a sparse
  file (the real GGUF header from an HTTP range request, holes to the full size): Axon's support check reads metadata
  only, so a clean refusal needs no download; anything else (it runs, an unclear error) falls back to the full file
  (--no-full-fetch: record `needs-full` instead). --chat adds, per runnable file: each engine's own chat prompt for 2
  conversations (Axon's built-in format vs llama.cpp's --jinja rendering of the GGUF template): prompt token counts
  and the greedy reply's characters in common; and the plain prompts' token ids (Axon's vs llama.cpp's tokenizer).
  Every result is also printed as one `ZOO-RESULT {json}` line (a pod job's results come back through its log).

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
import shlex
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

ENTRIES = ZOO
EXTRA = {}   # file -> the --list entry (sha256, bytes, rank, ollama, ...)


def load_entries(a):
    """--list FILE / --set golden: the entries to fetch and run instead of ZOO."""
    global ENTRIES
    src = getattr(a, "list", None)
    if getattr(a, "set", None) == "golden":
        src = src or os.path.join(HERE, "examples", "compat-audit", "ollama-top30.json")
    if not src:
        return
    rows = json.load(open(src))
    if getattr(a, "set", None) == "golden":
        rows = [x for x in rows if x.get("golden")]
    ENTRIES = []
    for x in rows:
        if not x.get("file"):
            continue
        ENTRIES.append((x["repo"], x["file"], x.get("expect", "run"), x.get("why", x.get("ollama", ""))))
        EXTRA[x["file"]] = x


GEN_PROMPT = "The history of the printing press begins"
KL_MAX, TIE = 0.02, 0.1


def path_of(e, a):
    f = e[1]
    return os.path.join(a.models_dir, f[6:]) if f.startswith("local:") else os.path.join(a.dir, f)


def fetch(a):
    os.makedirs(a.dir, exist_ok=True)
    if EXTRA:
        bad = [f"{e[1]}: {why}" for e in ENTRIES if not e[1].startswith("local:") for why in [fetch_one(a, e)] if why]
        print("\n".join(bad) or "fetched")
        return 1 if bad else 0
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
                        "general.name", "general.architecture", "tokenizer.ggml.model")):
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


CHATS = [
    [{"role": "system", "content": "You are a terse assistant."},
     {"role": "user", "content": "Name three primary colors."}],
    [{"role": "user", "content": "Write a haiku about autumn."},
     {"role": "assistant", "content": "Leaves fall, red and gold."},
     {"role": "user", "content": "Now one about winter."}],
]


def chat_text(j):
    m = ((j.get("choices") or [{}])[0]).get("message") or {}
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


def llama_side(a, model, ids_list, texts=None):
    """llama-server (CPU): top-K after each prompt + greedy 64 tokens after GEN ids. None if it can't load.
    texts (the prompts' text, as ids_list): llama.cpp's own token ids for them; with --chat its chat prompt (the
    GGUF template, --jinja) and greedy reply for each of CHATS."""
    import socket
    import time
    port = LL.free_port()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.dirname(a.llama_server) + ":" + env.get("LD_LIBRARY_PATH", "")
    log = open(os.path.join(a.work, os.path.basename(model) + ".llama.log"), "w")
    srv = subprocess.Popen([a.llama_server, "-m", model, "--port", str(port), "-c", "4096", "-np", "1",
                            "--no-warmup", "-ngl", str(getattr(a, "llama_ngl", 0)), "-t", str(a.threads), "--jinja"]
                           + shlex.split(getattr(a, "llama_args", "") or ""),
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

        def post(body, path="/completion"):
            req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=600))
        tops = []
        for ids in ids_list[:-1]:
            r = post({"prompt": ids, "n_predict": 1, "n_probs": 20, "temperature": 0, "cache_prompt": False})
            cp = r["completion_probabilities"][0]
            tops.append([[t["id"], t["logprob"]] for t in cp["top_logprobs"]][:20])
        g = post({"prompt": ids_list[-1], "n_predict": 64, "temperature": 0, "top_k": 1,
                  "cache_prompt": False, "ignore_eos": False})
        res = {"tops": tops, "gen": g.get("content", "")}
        if texts:
            res["ids"] = [post({"content": t, "add_special": True}, "/tokenize").get("tokens") for t in texts]
        if getattr(a, "chat", False):
            res["chat"] = []
            for conv in CHATS:
                c = {}
                try:
                    c["template"] = post({"messages": conv}, "/apply-template").get("prompt", "")
                    j = post({"messages": conv, "temperature": 0, "max_tokens": 48, "seed": 0}, "/v1/chat/completions")
                    c.update(prompt_tokens=(j.get("usage") or {}).get("prompt_tokens"), reply=chat_text(j))
                except Exception as ex:  # noqa: BLE001
                    c["error"] = repr(ex)[:200]
                res["chat"].append(c)
        return res, None
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
    xa = shlex.split(getattr(a, "axon_args", "") or "")
    grc, gout, gerr = run([a.axon, "gen", p, GEN_PROMPT, "-n", "64", "--device", a.device or "cpu"] + xa, env=env)
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
            ids, top = LL.axon_ids_and_top(a.axon, p, q, 20, ["--device", a.device or "cpu"] + xa)
        except subprocess.CalledProcessError as ex:
            problems.append(f"logits failed rc {ex.returncode}: {(ex.stderr or '')[-200:]!r}")
            break
        ids_list.append(ids)
        tops.append(top)
    if len(ids_list) == len(prompts) + 1:
        ll, why_not = llama_side(a, p, ids_list, prompts + [GEN_PROMPT])
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
            if ll.get("ids"):
                same = [i for i, x in enumerate(ll["ids"]) if x == ids_list[i]]
                r["tok_agree"] = f"{len(same)}/{len(ids_list)}"
                r["tok_differ"] = [{"prompt": (prompts + [GEN_PROMPT])[i][:40], "axon": ids_list[i][:12],
                                    "llama": (ll["ids"][i] or [])[:12]} for i in range(len(ids_list)) if i not in same][:3]
            if "chat" in ll:
                chat_compare(a, p, r, ll["chat"])
    r["problems"] = problems
    r["verdict"] = "BUG" if problems else ("drift" if r.get("drift") else ("match" if "top1" in r else "runs"))
    return r


def axon_chat(a, p, ctx=4096):
    """axon serve on p: usage.prompt_tokens and the greedy reply for each of CHATS (None, why if it won't serve).
    ctx: at most the model's training context (Axon refuses a longer --ctx without --rope-scaling; llama.cpp caps)."""
    import time
    port = LL.free_port()
    log = open(os.path.join(a.work, os.path.basename(p) + ".serve.log"), "w")
    srv = subprocess.Popen([a.axon, "serve", p, "--device", a.device or "cpu", "--port", str(port), "--ctx", str(ctx)],
                           stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(1200):
            if srv.poll() is not None:
                return None, f"axon serve exited {srv.returncode}"
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            return None, "axon serve not healthy"
        out = []
        for conv in CHATS:
            body = {"messages": conv, "temperature": 0, "max_tokens": 48, "seed": 0}
            req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            try:
                j = json.load(urllib.request.urlopen(req, timeout=600))
                out.append({"prompt_tokens": (j.get("usage") or {}).get("prompt_tokens"), "reply": chat_text(j)})
            except urllib.error.HTTPError as ex:
                out.append({"error": f"{ex.code} {ex.read()[:200]!r}"})
        return out, None
    finally:
        srv.terminate()
        srv.wait()


def chat_compare(a, p, r, lchat):
    """Each engine's own chat prompt for CHATS: token counts and the greedy replies' characters in common."""
    m = re.search(r"\| ctx (\d+)", "\n".join(r.get("info") or []))
    ax, why = axon_chat(a, p, min(4096, int(m.group(1))) if m else 2048)
    if ax is None:
        r.setdefault("problems_chat", []).append(why)
        r["chat"] = why
        return
    if ax and all("error" in x for x in ax):
        # Axon serves the model but refuses chat (no chat format for its vocabulary): not a prompt difference
        m = re.search(r'"message":\s*"([^"]+)', ax[0]["error"])
        r["chat"] = "axon refuses chat: " + (m.group(1) if m else ax[0]["error"])[:160]
        r["chat_prompt_eq"] = "refused"
        return
    rows = []
    for i, (x, y) in enumerate(zip(ax, lchat)):
        rows.append({"axon_tokens": x.get("prompt_tokens"), "llama_tokens": y.get("prompt_tokens"),
                     "common_chars": common_prefix(x.get("reply", ""), y.get("reply", "")),
                     "axon_reply": x.get("reply", x.get("error", ""))[:120], "llama_reply": y.get("reply", y.get("error", ""))[:120],
                     "llama_template": (y.get("template") or "")[:400]})
    r["chat_rows"] = rows
    r["chat_prompt_eq"] = f"{sum(1 for c in rows if c['axon_tokens'] is not None and c['axon_tokens'] == c['llama_tokens'])}/{len(rows)}"
    r["chat_common_chars"] = [c["common_chars"] for c in rows]


def hf_url(e):
    return f"https://huggingface.co/{e[0]}/resolve/main/{e[1]}"


def header_bytes(url, start=8 << 20):
    """The GGUF header (metadata + tensor infos, up to the data's start) by HTTP range requests."""
    import struct
    n = start
    while True:
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}", "User-Agent": "axon-zoo"})
        with urllib.request.urlopen(req, timeout=120) as r:
            buf = r.read()
        try:
            return buf[:gguf_data_start(buf)]
        except (struct.error, IndexError, ValueError):
            if len(buf) < n or n >= 256 << 20:
                raise
            n *= 4


def gguf_data_start(b):
    """Offset of the tensor data in a GGUF whose first bytes are b (ValueError/struct.error if b is too short)."""
    import struct
    if b[:4] != b"GGUF":
        raise ValueError("not GGUF")
    pos = 8
    nt, nkv = struct.unpack_from("<QQ", b, pos)
    pos += 16
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    align = 32

    def s(pos):
        n, = struct.unpack_from("<Q", b, pos)
        if pos + 8 + n > len(b):
            raise ValueError("short")
        return b[pos + 8:pos + 8 + n].decode("utf-8", "replace"), pos + 8 + n

    def val(t, pos):
        if t == 8:
            return s(pos)
        if t == 9:
            et, n = struct.unpack_from("<IQ", b, pos)
            pos += 12
            if et == 8:
                for _ in range(n):
                    _, pos = s(pos)
                return None, pos
            return None, pos + sizes[et] * n
        v, = struct.unpack_from("<" + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q",
                                       11: "q", 12: "d"}[t], b, pos)
        return v, pos + sizes[t]
    for _ in range(nkv):
        k, pos = s(pos)
        t, = struct.unpack_from("<I", b, pos)
        v, pos = val(t, pos + 4)
        if k == "general.alignment":
            align = v
    for _ in range(nt):
        _, pos = s(pos)
        nd, = struct.unpack_from("<I", b, pos)
        pos += 4 + 8 * nd + 4 + 8
    if pos > len(b):
        raise ValueError("short")
    return (pos + align - 1) // align * align


def sparse_check(a, e):
    """A sparse copy of e (real header, holes to the full size) through axon gen: the clean refusal, or None."""
    ex = EXTRA.get(e[1], {})
    p = os.path.join(a.dir, "sparse-" + os.path.basename(e[1]))
    r = {"repo": e[0], "file": os.path.basename(e[1]), "expect": e[2], "why": e[3], "sparse": True}
    r.update({k: ex[k] for k in ("rank", "ollama", "size", "ollama_arch", "ollama_quant", "ollama_params") if k in ex})
    try:
        h = header_bytes(hf_url(e))
    except Exception as x:  # noqa: BLE001
        r.update(verdict="fetch-failed", problems=[f"header: {x!r}"[:200]])
        return r, None
    with open(p, "wb") as f:
        f.write(h)
    os.truncate(p, ex.get("bytes") or len(h))
    try:
        r["meta"] = gguf_meta(p)
        rc, out, err = run([a.axon, "info", p])
        m = re.search(r"types? [^\n]*", out + err)
        r["types"] = m.group(0)[:200] if m else ""
        grc, gout, gerr = run([a.axon, "gen", p, "Hi", "-n", "1", "--device", "cpu"], timeout=a.sparse_timeout)
        msg = refusal_line(grc, gout, gerr)
        r["sparse_rc"] = grc
        if msg:
            r.update(verdict="refused", refusal=msg, problems=[], size_mb=round((ex.get("bytes") or 0) / 1e6),
                     sha256=ex.get("sha256"))
            return r, msg
        r["sparse_out"] = (gerr or gout).strip()[-300:]
        return r, None
    finally:
        os.remove(p)


def fetch_one(a, e):
    """Download e to its path (sha256 checked against the list). -> None, or why it failed."""
    p = path_of(e, a)
    if os.path.exists(p):
        return None
    ex = EXTRA.get(e[1], {})
    tmp = p + ".part"
    rc = subprocess.run(["curl", "-sSfL", "-C", "-", "--retry", "5", "-o", tmp, hf_url(e)]).returncode
    if rc != 0:
        return f"curl rc {rc}"
    if ex.get("sha256"):
        h = hashlib.sha256()
        with open(tmp, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() != ex["sha256"]:
            os.remove(tmp)
            return f"sha256 {h.hexdigest()[:12]} != {ex['sha256'][:12]}"
        open(p + ".sha256", "w").write(f"{ex['sha256']}  {os.path.basename(p)}\n")
    os.rename(tmp, p)
    return None


def emit(r):
    print("ZOO-RESULT " + json.dumps(r), flush=True)


def do_run(a):
    a.work = os.path.dirname(os.path.abspath(a.out)) or "."
    res = []
    if os.path.exists(a.out) and a.resume:
        res = json.load(open(a.out))
    done = {x["file"] for x in res}
    todo = [e for e in ENTRIES if not (a.only and a.only not in e[1])
            and os.path.basename(e[1].replace("local:", "")) not in done]
    os.makedirs(a.dir, exist_ok=True)
    pool = None
    if a.fetch_each:
        import concurrent.futures
        pool = concurrent.futures.ThreadPoolExecutor(1)
    futs = {}

    def full(e):
        return not (a.sparse_first and e[2] == "refuse") and not e[1].startswith("local:")

    def prefetch(i):
        for e in todo[i:]:
            if full(e) and e[1] not in futs and not os.path.exists(path_of(e, a)):
                futs[e[1]] = pool.submit(fetch_one, a, e)
                return
    for i, e in enumerate(todo):
        r = None
        if a.fetch_each and a.sparse_first and e[2] == "refuse" and not os.path.exists(path_of(e, a)):
            r, msg = sparse_check(a, e)
            if msg is None and r.get("verdict") != "fetch-failed":
                if a.no_full_fetch:
                    r.update(verdict="needs-full", problems=["not refused on its header"])
                else:
                    r = None  # not a clean refusal on its header: test the real file
        if r is None:
            if a.fetch_each and not os.path.exists(path_of(e, a)):
                f = futs.pop(e[1], None) or pool.submit(fetch_one, a, e)
                prefetch(i + 1)
                why = f.result()
                if why:
                    r = {"repo": e[0], "file": os.path.basename(e[1]), "expect": e[2], "why": e[3],
                         "verdict": "fetch-failed", "problems": [why]}
            elif a.fetch_each:
                prefetch(i + 1)
            if r is None:
                r = one(a, e)
                ex = EXTRA.get(e[1], {})
                r.update({k: ex[k] for k in ("rank", "ollama", "size", "ollama_arch", "ollama_quant", "ollama_params")
                          if k in ex})
                if a.delete_after and not e[1].startswith("local:"):
                    for x in (path_of(e, a), path_of(e, a) + ".sha256"):
                        if os.path.exists(x):
                            os.remove(x)
        res.append(r)
        emit(r)
        print(json.dumps({k: r.get(k) for k in ("file", "verdict", "top1", "kl_mean", "gen_common_chars",
                                                 "tok_agree", "chat_prompt_eq", "refusal", "problems")}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    if pool:
        pool.shutdown()
    bugs = [x for x in res if x["verdict"] in ("BUG",)]
    print(f"ZOO {len(res)} files: " + ", ".join(f"{v} {sum(x['verdict'] == v for x in res)}"
                                              for v in ("match", "drift", "runs", "refused", "BUG", "missing",
                                                        "needs-full", "fetch-failed")))
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


FTYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
         12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16",
         38: "MXFP4"}


def load_results(paths):
    """ZOO-RESULT lines from job logs, or zoo.json lists; the last record per file wins."""
    out = {}
    for p in paths:
        txt = open(p, errors="replace").read()
        try:   # a zoo.json list; a job log also opens with '[' ("[axon-run ...]"), so parse, don't sniff
            rows = json.loads(txt)
        except ValueError:
            rows = None
        if isinstance(rows, list):
            for r in rows:
                out[r["file"]] = r
            continue
        for ln in txt.splitlines():
            if ln.startswith("ZOO-RESULT "):
                r = json.loads(ln[11:])
                out[r["file"]] = r
    return sorted(out.values(), key=lambda r: (r.get("rank") or 999, r["file"]))


def fix_key(r):
    """The fix a refusal asks for, normalized: what one change would unlock."""
    m = r.get("refusal") or ""
    m = re.sub(r"^axon: [^:]*\.gguf: ", "", m)
    for rx in (r"architecture '[^']+'", r"pre-tokenizer \S+", r"type \S+ \(\d+\)", r"RoPE scaling \S+"):
        x = re.search(rx, m)
        if x:
            return x.group(0)
    return m[:80]


def audit_table(a):
    rs = load_results(a.paths)
    lst = {x["file"]: x for x in json.load(open(os.path.join(HERE, "examples", "compat-audit", "ollama-top30.json")))}
    for r in rs:   # the list's pull counts (ZOO-RESULT records don't carry them)
        r.setdefault("pulls_m", (lst.get(r["file"]) or {}).get("pulls_m"))
    print("| # | ollama model | HF file | arch | quant | Axon | prompt vs llama.cpp | greedy vs llama.cpp | reason / note |")
    print("|---:|---|---|---|---|---|---|---|---|")
    for r in rs:
        m = r.get("meta") or {}
        arch = m.get("architecture") or r.get("ollama_arch") or "?"
        q = FTYPE.get(m.get("file_type"), str(m.get("file_type", "?")))
        v = r.get("verdict")
        axon = {"match": "runs", "drift": "runs", "runs": "runs"}.get(v, v)
        prompt = greedy = ""
        note = ""
        if axon == "runs":
            eq = r.get("chat_prompt_eq")
            if eq == "refused":
                prompt = "chat refused: " + str(r.get("chat", "")).replace("axon refuses chat: ", "")[:60]
            elif eq:
                rows = r.get("chat_rows") or []
                prompt = "same" if eq == f"{len(rows)}/{len(rows)}" else "differs (" + ", ".join(
                    f"{c['axon_tokens']} vs {c['llama_tokens']}" for c in rows if c["axon_tokens"] != c["llama_tokens"]) + " tokens)"
            elif r.get("chat"):
                prompt = "n/a: " + str(r["chat"])[:40]
            if "top1" in r:
                greedy = ("agrees" if v == "match" else "drifts") + f" (top-1 {r['top1']}, KL {r['kl_mean']:.1e}, " \
                    f"{r.get('gen_common_chars', '?')} chars)"
            elif r.get("llama"):
                greedy = "n/a: " + r["llama"][:40]
            if r.get("tok_agree") and not r["tok_agree"].startswith(str(r["tok_agree"].split("/")[1]) + "/"):
                note += f"tokenizer: {r['tok_agree']} prompts agree with llama.cpp; "
        elif axon == "refused":
            note = re.sub(r" \(supported: [^)]*\)", "", re.sub(r"^axon: [^:]*\.gguf: ", "", r.get("refusal") or ""))
        note += "; ".join(r.get("problems") or [])
        name = f"{r.get('ollama', '')} {r.get('size', '')}".strip()
        print(f"| {r.get('rank', '')} | {name} | {r['repo']}/{r['file']} | {arch} | {q} | {axon} | {prompt} | {greedy} | "
              f"{note.replace('|', '/')[:200]} |")
    fixes = {}
    for r in rs:
        if r.get("verdict") == "refused":
            fixes.setdefault(fix_key(r), []).append(r)
    if fixes:
        print("\n| fix (refusal) | models it unlocks | ollama pulls (M) | models |")
        print("|---|---:|---:|---|")
        for k, v in sorted(fixes.items(), key=lambda kv: (-len(kv[1]), -sum(x.get("pulls_m") or 0 for x in kv[1]))):
            print(f"| {k} | {len(v)} | {sum(x.get('pulls_m') or 0 for x in v):.1f} | "
                  f"{', '.join(x.get('ollama') or x['file'] for x in v)} |")
    tally = {}
    for r in rs:
        tally[r.get("verdict")] = tally.get(r.get("verdict"), 0) + 1
    print("\nverdicts: " + ", ".join(f"{k} {n}" for k, n in sorted(tally.items(), key=lambda kv: str(kv[0]))))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dir", required=True)
    f.add_argument("--list")
    f.add_argument("--set", choices=["golden"])
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
    r.add_argument("--list")
    r.add_argument("--set", choices=["golden"])
    r.add_argument("--llama-ngl", type=int, default=0)
    r.add_argument("--axon-args", default="", help="extra flags for every axon gen/logits (e.g. '--rope-scaling none')")
    r.add_argument("--llama-args", default="", help="extra flags for llama-server (e.g. '--rope-scaling none')")
    r.add_argument("--chat", action="store_true")
    r.add_argument("--fetch-each", action="store_true")
    r.add_argument("--delete-after", action="store_true")
    r.add_argument("--sparse-first", action="store_true")
    r.add_argument("--no-full-fetch", action="store_true")
    r.add_argument("--sparse-timeout", type=int, default=60)
    t = sub.add_parser("table")
    t.add_argument("file")
    at = sub.add_parser("audit-table")
    at.add_argument("paths", nargs="+")
    a = ap.parse_args()
    if a.cmd == "audit-table":
        return audit_table(a)
    load_entries(a)
    if a.cmd == "fetch":
        return fetch(a)
    if a.cmd == "table":
        return table(a)
    return do_run(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
