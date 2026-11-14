#!/usr/bin/env python3
"""Golden check: Axon against llama.cpp on whole models, from text in (stdlib only).

    scripts/golden.py run --axon BIN --llama-server BIN --models-dir D --models a.gguf,b.gguf
                          [--llama-server-ref CUDA_BIN [--llama-ref-ngl 99] [--llama-ref-extra SUBSTR=ARGS]]
                          [--device cuda] [--n 64] [--threads 8] [--llama-ngl 0] --out golden.json
    scripts/golden.py table golden.json      a markdown table of a run

scripts/zoo.py feeds llama.cpp Axon's own token ids, so a tokenizer or chat-template bug passes it. This one starts
from text on both sides and compares, per model, against llama.cpp's llama-server (CPU unless --llama-ngl):
  (a) tokenize  TOK_PROMPTS as a plain prompt: `axon logits`' prompt ids vs /tokenize (add_special, no special
                parsing), exactly. One exception, recorded: llama.cpp turns some user-defined tokens into control
                tokens at load (it logs "control-looking token ... overridden"; gemma 3's "</s>") and then spells
                their text out; Axon keeps them whole, as the GGUF and Hugging Face do. A prompt is equal if Axon
                matches llama.cpp's tokenization with exactly those tokens kept whole (tokenize_keeping).
  (b) chat      CHAT (a system message and two user turns) as /v1/chat/completions would run it: Axon's rendering
                (`axon logits` with AXON_LOGITS_CHAT) vs llama.cpp's (/apply-template of the GGUF's own template,
                then /tokenize with add_special and special parsing, as llama-server's chat endpoint does),
                exactly. The dates Axon pins (chat.rs: Llama 3's "26 Jul 2024", harmony's HARMONY_DATE) go to the
                template as chat_template_kwargs; a template that reads the clock instead (gpt-oss: strftime_now)
                has its date replaced in llama.cpp's rendering (DATE_FIXES), so today's date is never a difference.
  (c) greedy    llama.cpp's greedy N tokens after (b)'s llama.cpp ids and after GEN_PROMPT, then Axon teacher-
                forced on the same ids (AXON_LOGITS_IDS + AXON_LOGITS_PREFIX), top K on both sides. Per step: top-1
                agreement, KL(llama || axon) over llama.cpp's top K, |dlogprob| of llama.cpp's token, and for a flip
                its swing (llama.cpp's margin for its token + Axon's for its own). KL has a tail bucket on each side
                (the mass outside the top K), so it is a KL over K+1 categories: never negative.
                Hard clauses, always judged: top-1 at least AGREE_MIN, llama.cpp's token inside Axon's top K at every
                step, every logit finite. Scaled clauses: with --llama-server-ref, llama.cpp's other backend runs the
                same steps, and Axon may differ from llama.cpp's CPU by SPREAD_FACTOR x that spread (never less than
                the floors): mean KL, max |dlogprob|, and a swing of twice the |dlogprob| limit (two tokens each off
                by the limit). Gemma 3 1B differs from a float64 reference by up to 0.4 nats in BOTH engines while
                Llama 3.2 1B stays under 0.13, so one fixed limit is wrong for one of them, and the floors alone fail
                clean Gemma 3 models. So without a usable spread (no --llama-server-ref, a reference that didn't
                start, or one that fails the hard clauses itself against llama.cpp's CPU: a garbage reference must not
                widen the limits) the scaled clauses are NOT judged: the model is ERROR unless a hard clause FAILs.
                --llama-ref-extra SUBSTR=ARGS adds ARGS to the reference's llama-server for the models whose file name
                contains SUBSTR (gpt-oss-20b on a 12 GB card: `gpt-oss=--n-cpu-moe 12`).
A model passes when (a) and (b) are equal and (c) passes. Verdicts: pass, FAIL (a real difference), ERROR (not
judged: axon crashed or timed out, chat n/a, no usable reference spread, golden.py itself crashed on the model),
missing, llama-error. Exit 0 all pass, 1 any FAIL, 2 anything not judged (also usage and a crash outside a model).
The JSON (--out) is rewritten after every model: "planned" (the models of this run, written before the first),
"complete" (false until the last model is done) and "exit". A doc that isn't complete is a run that was cut off.
BY_DESIGN lists the checks where Axon departs from llama.cpp on purpose (a model file pattern, the check, the reason
with the code or doc that says so, and Axon's exact ids for that check). Such a check still runs and keeps its
difference in the record; if Axon's ids are exactly the listed ones it reports BY-DESIGN instead of DIFF and doesn't
fail the model, any other difference stays DIFF (FAIL). The model's other checks are graded as usual. A listed check
that comes out equal is flagged stale (the entry can go).
`--only SUBSTR` runs the matching models only (axon-nightly's bisect).
`scripts/golden_selftest.py` checks the parsing and the verdict logic offline (CI runs it).
"""

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
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
K = 20   # top-K logprobs from both engines (KL over llama.cpp's top K, as scripts/oracle/llama_logits.py)
# (c)'s limits: Axon vs llama.cpp's CPU may differ by SPREAD_FACTOR x llama.cpp's own CUDA-vs-CPU spread on the same
# steps (--llama-server-ref), never less than the floors. From the 11-13 measurements (13 models, bella sm_120 and a
# 4090) in docs/log/2026-11-12-process-v2-safeguards.md, "Thresholds": clean Axon reaches 1.66x llama.cpp's spread
# on max |dlogprob| (Gemma 3 4B), so 2.5 keeps every clean model at or under 2/3 of each limit; KL_MEAN_FLOOR is
# twice the worst clean mean KL where the floor applies (0.00308, Llama 3.2 1B CUDA). Planted on Llama 3.2 1B: rms
# eps x10, a Q6_K scale x1.10 and Q3_K_L weights fail; eps x3 and a Q6_K scale x1.05 pass (inside the noise).
SPREAD_FACTOR, KL_MEAN_FLOOR, DLP_FLOOR, AGREE_MIN = 2.5, 0.006, 0.3, 0.9
# Differences by design (shady on #246): (model file regex, check, why: the code or doc that decides it).
# "axon_ids": Axon's whole rendering for that check, exactly; only that difference is by design (shady #229 round 2:
# a Mistral serve-prompt regression must not pass as BY-DESIGN).
BY_DESIGN = [
    {"model": re.compile(r"(?i)mistral-7b-instruct-v0\.3"), "check": "chat",
     "why": "Axon frames Mistral v0.3 with mistral_common's tokens, each piece between control tokens encoded on its "
            "own, in llama.cpp's mistral-v3 placement (system message in the first user turn) (src/chat.rs module "
            "doc, \"Mistral\"; pinned by chat.rs's mistral tests); llama.cpp tokenizes the rendered text, a "
            "'\u2581' more after [INST], and the GGUF embeds an older template",
     # CHAT as <s>[INST] {system}\n\n{user}[/INST] {reply}</s>[INST] {user}[/INST], each text piece tokenized on its
     # own by llama.cpp's SentencePiece (/tokenize, no specials: the piece's dummy-prefix '\u2581'), 11-13, bartowski's
     # Mistral-7B-Instruct-v0.3-Q4_K_M.gguf (sha256 1270d22c...); Axon at bc11938 renders the same 47 (golden 7080).
     "axon_ids": [1, 3, 1763, 1228, 1032, 3846, 1632, 14660, 29491, 27075, 1065, 1206, 1848, 2480, 24516, 29491, 781,
                  781, 3963, 1117, 1040, 6333, 1070, 5611, 29572, 4, 1183, 6333, 1070, 5611, 1117, 6233, 29491, 2, 3,
                  1783, 1070, 5488, 29572, 16521, 2847, 1146, 4014, 1040, 6333, 29491, 4]},
]
# Templates that print today's date ignore chat_template_kwargs (gpt-oss: strftime_now): llama.cpp's rendering is
# normalized to Axon's pinned dates before it is tokenized.
DATE_FIXES = [(re.compile(r"Current date: \d{4}-\d{2}-\d{2}"), "Current date: {current_date}"),
              (re.compile(r"Today Date: \d{1,2} [A-Z][a-z]{2} \d{4}"), "Today Date: {date_string}")]
# llama.cpp turns a user-defined token whose text is on its end-of-generation list (e.g. "</s>") into a control token
# at load, and logs it; with parse_special off it then spells that text out, where the GGUF's type, SentencePiece and
# Hugging Face's tokenizer keep it one token (gemma 3's </s>, id 212: HF encodes "... plain </s>" as [..., 236743, 212],
# 2026-11-13). See override_ids / tokenize_keeping.
OVERRIDE_RE = re.compile(r"control-looking token:\s+(\d+) '(.*)' was not control-type")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Llama:
    """A llama-server on a free port; stopped on exit."""

    def __init__(self, a, model, log, extra=()):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.path.dirname(a.llama_server) + ":" + env.get("LD_LIBRARY_PATH", "")
        self.log = open(log, "w")
        self.p = subprocess.Popen([a.llama_server, "-m", model, "--port", str(self.port), "-c", "4096", "-np", "1",
                                   "--jinja", "--no-warmup", "-ngl", str(a.llama_ngl), "-t", str(a.threads),
                                   *extra],
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
        r = self.post("/completion", {"prompt": ids, "n_predict": n, "temperature": 0, "n_probs": K,
                                      "cache_prompt": False})
        out = []
        for cp in r.get("completion_probabilities", []):
            top = [[t["id"], t["logprob"]] for t in cp["top_logprobs"]]
            out.append((cp["id"], cp["logprob"], top))
        return out

    def forced(self, ids, path):
        """this server's top-K at each step of another run's path (one /completion per step, n_predict 1)."""
        out = []
        for i in range(len(path)):
            r = self.post("/completion", {"prompt": ids + [t for t, _, _ in path[:i]], "n_predict": 1,
                                          "temperature": 0, "n_probs": K, "cache_prompt": True})
            out.append([[t["id"], t["logprob"]] for t in r["completion_probabilities"][0]["top_logprobs"]])
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


def override_ids(log_path):
    """{id: text} of the user-defined tokens llama-server says it overrode to control (its load log)."""
    try:
        return {int(m.group(1)): m.group(2) for m in OVERRIDE_RE.finditer(open(log_path, errors="replace").read())}
    except OSError:
        return {}


def tokenize_keeping(ll, text, keep):
    """llama.cpp's plain tokenization of text with the overridden tokens in keep ({id: text}) kept whole, as the
    GGUF's user-defined type says: the text between them tokenized by llama.cpp itself, the ids put back between.
    None if no kept token's text occurs in text."""
    pieces = [(text, None)]
    for tid, t in keep.items():
        if not t:
            continue
        out = []
        for seg, sid in pieces:
            if sid is not None:
                out.append((seg, sid))
                continue
            parts = seg.split(t)
            for j, part in enumerate(parts):
                out.append((part, None))
                if j < len(parts) - 1:
                    out.append((t, tid))
        pieces = out
    if all(sid is None for _, sid in pieces):
        return None
    ids, first = [], True
    for seg, sid in pieces:
        if sid is not None:
            if first:
                ids += ll.tokenize("", True, False)   # the BOS a plain tokenization would start with
                first = False
            ids.append(sid)
        elif seg:
            ids += ll.tokenize(seg, first, False)
            first = False
    return ids


class NotJudged(Exception):
    """the harness couldn't measure (axon crashed or timed out, a server didn't start): ERROR, not FAIL."""


def kl_top(pa, pb):
    """KL(a || b) over a's top K plus one tail bucket (the mass outside those K) on each side: a KL between two
    distributions over the same K+1 categories, so never negative (Gibbs), and at most the full KL. Over the top K
    alone it went negative when b is sharper than a (a logit-scale bug: Q6_K x1.10, the log's planted bugs). b's
    logprob of a token outside its own top K is floored at its K-th (an upper bound); each side is renormalized if
    rounding or that floor puts its K masses over 1."""
    db = dict((i, lp) for i, lp in pb)
    floor = min(lp for _, lp in pb)
    p = [math.exp(lp) for _, lp in pa]
    q = [math.exp(db.get(i, floor)) for i, _ in pa]
    pt, qt = max(0.0, 1.0 - sum(p)), max(0.0, 1.0 - sum(q))
    zp, zq = sum(p) + pt, sum(q) + qt
    kl = sum(x / zp * math.log((x / zp) / (y / zq)) for x, y in zip(p, q) if x > 0)
    if pt > 0:
        kl += pt / zp * math.log((pt / zp) / max(qt / zq, 1e-12))
    return max(0.0, kl)   # >= 0 in exact arithmetic; clamp the last-ulp rounding


def compare(path_, tops, kind, m):
    """llama.cpp's greedy path [(id, logprob, top)] vs another engine's top-K at the same steps, into m."""
    for i, ((tid, lp, top), at) in enumerate(zip(path_, tops)):
        if not at:
            raise NotJudged(f"{kind}: no logits at step {i}")
        if not all(math.isfinite(v) for _, v in top) or not math.isfinite(lp):
            raise NotJudged(f"{kind}: llama.cpp gave a non-finite logprob at step {i}")
        if not all(math.isfinite(v) for _, v in at):
            m["nonfinite"].append({"kind": kind, "step": i})
            continue
        m["steps"] += 1
        mine, theirs = dict((t, v) for t, v in at), dict((t, v) for t, v in top)
        m["kl"].append(kl_top(top, at))
        if tid in mine:
            m["dlp"].append(abs(mine[tid] - lp))
        else:
            m["missing"].append({"kind": kind, "step": i, "llama": tid})
        if at[0][0] == tid:
            m["agree"] += 1
            continue
        lg = lp - theirs.get(at[0][0], min(theirs.values()))   # llama.cpp's margin for its token (>= if floored)
        og = at[0][1] - mine.get(tid, min(mine.values()))      # the other engine's margin for its own
        m["flips"].append({"kind": kind, "step": i, "llama": tid, "other": at[0][0], "llama_gap": round(lg, 4),
                           "other_gap": round(og, 4), "swing": round(lg + og, 4)})


def summary(m):
    n = m["steps"]
    return {"steps": n, "agree": m["agree"], "agree_frac": round(m["agree"] / n, 4) if n else None,
            "kl_mean": round(sum(m["kl"]) / n, 5) if n else None, "kl_max": round(max(m["kl"] or [0]), 4),
            "max_dlp": round(max(m["dlp"] or [0]), 4), "missing": m["missing"], "nonfinite": m["nonfinite"],
            "flips": m["flips"]}


def ref_problems(spread):
    """Why llama.cpp's other backend can't set (c)'s limits: summary() of it vs llama.cpp's CPU on the same steps must
    pass golden's hard clauses itself (finite logits, the CPU's token inside its top K, top-1 >= AGREE_MIN). The limits
    scale with the spread and have no ceiling, so a garbage reference would otherwise widen them (shady, #229 round 2:
    a synthetic one gave KL 13, dlp 17). [] when it is sane."""
    if not spread or not spread.get("steps"):
        return ["no finite steps"]
    why = []
    if spread["nonfinite"]:
        why.append(f"non-finite logits at {len(spread['nonfinite'])} step(s)")
    if spread["missing"]:
        why.append(f"llama.cpp CPU's token outside its top {K} at {len(spread['missing'])} step(s)")
    if spread["agree_frac"] < AGREE_MIN:
        why.append(f"top-1 {spread['agree_frac']} < {AGREE_MIN}")
    return why


def judge(g, spread=None, no_spread="no --llama-server-ref"):
    """(c)'s verdict on summary() g. Hard clauses, always: steps, finite logits, llama.cpp's token inside Axon's top K,
    top-1 >= AGREE_MIN. Scaled clauses (mean KL, max |dlogprob|, the flips' swing), only against a sane spread
    (summary() of llama.cpp's other backend vs its CPU on the same steps, see ref_problems): SPREAD_FACTOR x it, never
    under the floors. Returns (limits, clear flips, why, unjudged): a non-empty why FAILs the check; else a non-empty
    unjudged says why the scaled clauses weren't judged (the check is ERROR); else it passes. no_spread: why there is
    no spread (the reference didn't start, ...)."""
    why = []
    if not g["steps"]:
        why.append("no steps")
    if g["nonfinite"]:
        why.append(f"{len(g['nonfinite'])} step(s) with non-finite logits")
    if g["missing"]:
        why.append(f"llama.cpp's token outside Axon's top {K} at {len(g['missing'])} step(s)")
    if g["steps"] and g["agree_frac"] < AGREE_MIN:
        why.append(f"top-1 {g['agree_frac']} < {AGREE_MIN}")
    bad = ref_problems(spread) if spread is not None else None
    if spread is None or bad:
        unj = ("mean KL, max dlogprob and swing not judged: "
               + (f"llama.cpp's reference spread is not sane ({'; '.join(bad)})" if bad else no_spread))
        return {"kl_mean": None, "max_dlp": None, "swing": None, "agree_frac": AGREE_MIN, "from": unj}, [], why, unj
    lim = {"kl_mean": round(max(KL_MEAN_FLOOR, SPREAD_FACTOR * spread["kl_mean"]), 5),
           "max_dlp": round(max(DLP_FLOOR, SPREAD_FACTOR * spread["max_dlp"]), 4), "agree_frac": AGREE_MIN,
           "from": "llama.cpp's spread"}
    lim["swing"] = round(2 * lim["max_dlp"], 4)
    clear = [f for f in g["flips"] if f["swing"] > lim["swing"]]
    if clear:
        why.append(f"{len(clear)} flip(s) with a swing over {lim['swing']}")
    if g["steps"] and g["kl_mean"] > lim["kl_mean"]:
        why.append(f"KL mean {g['kl_mean']} > {lim['kl_mean']}")
    if g["max_dlp"] > lim["max_dlp"]:
        why.append(f"max dlogprob {g['max_dlp']} > {lim['max_dlp']}")
    return lim, clear, why, None


def by_design(model, check):
    """BY_DESIGN's entry for this model file's check, or None."""
    for e in BY_DESIGN:
        if e["check"] == check and e["model"].search(model):
            return e
    return None


def apply_by_design(r):
    """A judged check that differs where BY_DESIGN says Axon departs on purpose, with Axon's ids exactly the entry's:
    BY-DESIGN (pass, with the reason and the difference kept). Any other difference stays DIFF, with by_design_mismatch
    saying where Axon left the known rendering. One that came out equal anyway: by_design_stale. Not-judged checks stay
    not judged."""
    r["by_design"] = []
    for k, c in r["checks"].items():
        e = by_design(r["model"], k)
        if not e or c.get("not_judged"):
            continue
        if c.get("pass"):
            c["by_design_stale"] = f"listed as by design but equal: drop its BY_DESIGN entry ({e['why'][:60]}...)"
            r["problems"].append(f"{k}: {c['by_design_stale']}")
            continue
        got, want = c.get("axon_ids"), e["axon_ids"]
        if got != want:
            i = first_diff(got or [], want)
            c["by_design_mismatch"] = (f"not the known by-design difference: Axon's ids leave the listed rendering at "
                                       f"{i} (axon {(got or [])[max(0, (i or 0) - 2):(i or 0) + 4]} vs listed "
                                       f"{want[max(0, (i or 0) - 2):(i or 0) + 4]}, {len(got or [])} vs {len(want)} ids)")
            r["problems"].append(f"{k}: {c['by_design_mismatch']}")
            continue
        c.update(by_design=e["why"], was=c.get("status", "DIFF"), status="BY-DESIGN")
        c["pass"] = True   # graded as by design; the difference itself stays in the record (at / axon / llama)
        r["by_design"].append(k)


def verdict(r):
    """The model's verdict from its checks: FAIL if a judged check failed, else ERROR if one wasn't judged, else pass."""
    judged = [c for c in r["checks"].values() if not c.get("not_judged")]
    if any(not c["pass"] for c in judged):
        return "FAIL"
    if len(judged) < len(r["checks"]):
        r["problems"] += [f"{k}: {c.get('error') or c.get('why')}" for k, c in r["checks"].items() if c.get("not_judged")]
        return "ERROR"
    return "pass"


def exit_code(verdicts):
    """1: a real difference; 2: something wasn't judged (a missing model, llama.cpp or axon not running, chat n/a)."""
    return 1 if "FAIL" in verdicts else (2 if any(x != "pass" for x in verdicts) else 0)


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


def normalize_dates(rendered):
    """llama.cpp's rendering with DATE_FIXES applied: (text, the patterns that changed something)."""
    fixed = []
    for rx, sub in DATE_FIXES:
        new = rx.sub(sub.format(**TEMPLATE_KWARGS), rendered)
        if new != rendered:
            fixed.append(rx.pattern)
        rendered = new
    return rendered, fixed


def ref_extra(a, name):
    """--llama-ref-extra's args for this model file (every SUBSTR=ARGS whose SUBSTR is in the name)."""
    out = []
    for x in getattr(a, "llama_ref_extra", None) or []:
        sub, _, args = x.partition("=")
        if sub and sub in name:
            out += args.split()
    return out


def new_metrics():
    return {"steps": 0, "agree": 0, "kl": [], "dlp": [], "missing": [], "nonfinite": [], "flips": []}


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
        keep = override_ids(os.path.join(a.work, name + ".llama.log"))
        l_keep = [tokenize_keeping(ll, t, keep) if keep else None for t in TOK_PROMPTS]
        # llama-server's chat endpoint: the template (BOS stripped when the vocab adds one), then add_special + parse
        rendered = ll.post("/apply-template", {"messages": CHAT, "chat_template_kwargs": TEMPLATE_KWARGS})["prompt"]
        rendered, fixed = normalize_dates(rendered)
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
    # llama.cpp's other backend on the same steps: the spread that sets (c)'s scaled limits. One that can't start or
    # answer leaves them unjudged (ERROR); (a), (b) and (c)'s hard clauses still run.
    spread, no_spread = None, "no --llama-server-ref"
    if a.llama_server_ref:
        b = argparse.Namespace(llama_server=a.llama_server_ref, threads=a.threads, llama_ngl=a.llama_ref_ngl)
        extra = ref_extra(a, name)
        if extra:
            r["ref_args"] = extra
        lr = Llama(b, path, os.path.join(a.work, name + ".llama-ref.log"), extra)
        try:
            lr.wait()
            spread = {k: lr.forced(ids, greedy[k]) for k, ids in (("chat", l_chat), ("text", l_gen_ids))}
        except Exception as e:  # noqa: BLE001 (no yardstick: the scaled clauses are not judged)
            no_spread = f"llama.cpp reference backend ({a.llama_server_ref}, -ngl {a.llama_ref_ngl}): {e}"
            r["ref_error"] = no_spread
        finally:
            lr.stop()
    bos = bos_probe[0] if bos_probe else None
    r["llama_rendered"] = rendered[:2000]
    r["date_normalized"] = fixed
    # (a) tokenization
    tok = {"prompts": len(TOK_PROMPTS), "equal": 0, "diffs": [], "llama_override": []}
    try:
        for text, want, alt in zip(TOK_PROMPTS, l_tok, l_keep):
            got = ids_line(axon(a, path, text, {}), "prompt ids")
            if got == want:
                tok["equal"] += 1
            elif alt is not None and got == alt:
                # equal once llama.cpp's own override is undone (OVERRIDE_RE): counted, and named in the record
                tok["equal"] += 1
                tok["llama_override"].append({"text": text, "tokens": {str(k): v for k, v in keep.items() if v in text},
                                              "llama_plain_len": len(want), "axon_len": len(got)})
            else:
                i = first_diff(got or [], want)
                tok["diffs"].append({"text": text, "at": i, "axon": (got or [])[max(0, (i or 0) - 2):(i or 0) + 4],
                                     "llama": want[max(0, (i or 0) - 2):(i or 0) + 4],
                                     "axon_len": len(got or []), "llama_len": len(want)})
        tok["pass"] = tok["equal"] == tok["prompts"]
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        tok.update(error=str(e)[-300:])
        tok["pass"] = False
        tok["not_judged"] = True
    r["checks"]["tokenize"] = tok
    # (b) chat rendering
    chat = {}
    body = os.path.join(work, "chat.json")
    json.dump({"messages": CHAT}, open(body, "w"))
    try:
        out = axon(a, path, "x", {"AXON_LOGITS_CHAT": body})
        got = ids_line(out, "chat prompt ids")
        if got is None:
            chat.update(status="n/a", why="this axon has no AXON_LOGITS_CHAT", not_judged=True)
        else:
            g, gb = strip_bos(got, bos)
            w, wb = strip_bos(l_chat, bos)
            i = first_diff(g, w)
            chat.update(status="equal" if i is None and gb == wb else "DIFF", axon_len=len(got), llama_len=len(l_chat),
                        axon_bos=gb, llama_bos=wb)
            if i is not None:
                chat.update(at=i, axon=g[max(0, i - 3):i + 6], llama=w[max(0, i - 3):i + 6])
            if chat["status"] == "DIFF":   # both whole (BY_DESIGN compares Axon's with its listed ids)
                chat.update(axon_ids=got, llama_ids=l_chat)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        chat.update(status="ERROR", why=str(e)[-300:], not_judged=True)
    chat["pass"] = chat["status"] == "equal"
    r["checks"]["chat"] = chat
    # (c) greedy top-1 and the distributions, teacher-forced on llama.cpp's path; limits from llama.cpp's own spread
    g = {"n": a.n, "k": K}
    ax, sp = new_metrics(), new_metrics()
    try:
        for kind, prompt_ids in (("chat", l_chat), ("text", l_gen_ids)):
            path_ = greedy[kind]
            if not path_:
                raise NotJudged(f"{kind}: llama.cpp generated nothing")
            seq = prompt_ids + [t for t, _, _ in path_[:-1]]
            out = axon(a, path, "x", {"AXON_LOGITS_IDS": ",".join(map(str, seq)),
                                      "AXON_LOGITS_PREFIX": str(len(prompt_ids)), "AXON_LOGITS_DIGITS": "6"}, k=K)
            pos = positions(out)
            compare(path_, [pos.get(len(prompt_ids) - 1 + i) for i in range(len(path_))], kind, ax)
        g.update(summary(ax))
        if spread is not None:
            try:
                for kind in ("chat", "text"):
                    compare(greedy[kind], spread[kind], kind, sp)
                g["spread"] = summary(sp)
            except NotJudged as e:   # the reference gave no logits at a step
                no_spread = f"llama.cpp's reference: {e}"
        g["limits"], g["clear"], g["why"], unjudged = judge(g, g.get("spread"), no_spread)
        g["pass"] = not g["why"]
        if g["pass"] and unjudged:   # the hard clauses held; the scaled ones had no yardstick: ERROR, never a pass
            g.update(not_judged=True, error=unjudged)
            g["pass"] = False
    except NotJudged as e:
        g.update(error=str(e)[-300:], not_judged=True)
        g["pass"] = False
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        g.update(error=str(e)[-300:], not_judged=True)
        g["pass"] = False
    r["checks"]["greedy"] = g
    apply_by_design(r)
    r["verdict"] = verdict(r)
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
    try:
        ver = subprocess.run([a.axon, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        ver = f"unknown: {e}"
    doc = {"axon": a.axon, "axon_version": ver, "llama_server": a.llama_server, "llama_version": llama_version(a),
           "device": a.device, "n": a.n, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "planned": models, "complete": False, "models": []}
    save(doc, a.out)   # a run cut off before its last model leaves complete: false (axon-nightly checks)
    for m in models:
        t0 = time.time()
        try:
            r = one(a, m)
        except Exception as e:  # noqa: BLE001 (a bug in golden.py itself: this model is not judged, the next one runs)
            r = {"model": m, "checks": {}, "verdict": "ERROR",
                 "problems": [f"golden.py crashed on this model: {type(e).__name__}: {e}"[:400]],
                 "traceback": traceback.format_exc()[-3000:]}
            print(r["traceback"], file=sys.stderr, flush=True)
        r["secs"] = round(time.time() - t0, 1)
        doc["models"].append(r)
        c = r["checks"]
        print(json.dumps({"model": m, "verdict": r["verdict"],
                          "tokenize": c.get("tokenize", {}).get("equal"),
                          "chat": c.get("chat", {}).get("status"),
                          "by_design": r.get("by_design") or None,
                          "agree": c.get("greedy", {}).get("agree_frac"),
                          "clear": len(c.get("greedy", {}).get("clear", [])),
                          "max_dlp": c.get("greedy", {}).get("max_dlp"),
                          "kl_mean": c.get("greedy", {}).get("kl_mean"),
                          "limits": {k: v for k, v in c.get("greedy", {}).get("limits", {}).items() if k != "from"},
                          "secs": r["secs"],
                          "problems": r["problems"]}), flush=True)
        save(doc, a.out)
    v = [r["verdict"] for r in doc["models"]]
    bd = sum(1 for r in doc["models"] if r.get("by_design"))
    print(f"GOLDEN {len(v)} models: pass {v.count('pass')}, FAIL {v.count('FAIL')}, ERROR {v.count('ERROR')}, "
          f"missing {v.count('missing')}, llama-error {v.count('llama-error')}"
          + (f" (by design: {bd} model(s) with a listed difference)" if bd else ""), flush=True)
    rc = exit_code(v)
    doc.update(complete=True, exit=rc, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(doc, a.out)
    return rc


def save(doc, out):
    """The JSON, replaced atomically (a reader never sees half a file)."""
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, out)


def table(a):
    doc = json.load(open(a.file))
    print(f"axon {doc['axon_version']} vs llama.cpp {' '.join(doc['llama_version'])}, {doc['device']}, "
          f"greedy n={doc['n']}\n")
    print("| model | verdict | tokenize | chat | greedy top-1 | flips (over the swing limit) | KL mean (limit) | "
          "max dlogprob (limit) |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    for r in doc["models"]:
        c = r["checks"]
        t, ch, g = c.get("tokenize", {}), c.get("chat", {}), c.get("greedy", {})
        ov = f" ({len(t['llama_override'])} via llama.cpp's override)" if t.get("llama_override") else ""
        ov += " BY-DESIGN" if t.get("by_design") else ""
        print(f"| {r['model']} | {r['verdict']} | {t.get('equal', '')}/{t.get('prompts', '')}{ov} | "
              f"{ch.get('status', '')} | {g.get('agree', '')}/{g.get('steps', '')} | "
              f"{len(g.get('flips', []))} ({len(g.get('clear', []))}) | {g.get('kl_mean', '')} "
              f"({g.get('limits', {}).get('kl_mean', '')}) | {g.get('max_dlp', '')} "
              f"({g.get('limits', {}).get('max_dlp', '')}) |")


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
    r.add_argument("--llama-server-ref", help="llama.cpp's other backend (a CUDA build): its spread from --llama-server "
                                              "on the same steps sets (c)'s limits")
    r.add_argument("--llama-ref-ngl", type=int, default=99)
    r.add_argument("--llama-ref-extra", action="append", metavar="SUBSTR=ARGS",
                   help="extra llama-server args for the reference on models whose file name contains SUBSTR "
                        "(repeatable; e.g. 'gpt-oss=--n-cpu-moe 12' to fit a 12 GB card)")
    r.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("file")
    a = ap.parse_args()
    if a.cmd == "table":
        table(a)
        return 0
    return run(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001 (a crash outside a model is "not judged", never FAIL's exit 1)
        traceback.print_exc()
        print("golden: crashed (exit 2: not judged)", file=sys.stderr)
        sys.exit(2)
