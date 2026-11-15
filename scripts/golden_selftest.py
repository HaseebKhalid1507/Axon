#!/usr/bin/env python3
"""Offline self-test of golden.py and golden-top10.sh: stdlib only, no engine, no model, no llama.cpp, no network,
about a second (CI runs it beside agent_replay_selftest.py).

- The parsers on canned `axon logits` output: positions() (AXON_LOGITS_PREFIX's per-position top-K, inf / NaN too)
  and ids_line() ("prompt ids" and "chat prompt ids"; a missing line is None, which golden reports as chat n/a).
- compare() / summary() / judge() on synthetic steps: agreement, a flip and its swing, llama.cpp's token outside
  Axon's top K, a non-finite logit. The hard clauses FAIL with or without a reference; the scaled ones (mean KL, max
  dlogprob, swing) are judged only against a sane reference spread: none, or a garbage one, is ERROR, never FAIL
  and never a widened pass. kl_top()'s tail bucket keeps KL >= 0 and <= the full KL (a logit-scale bug).
- The verdicts: FAIL beats ERROR, a not-judged check is ERROR, exit codes 0 / 1 / 2.
- BY_DESIGN: Mistral v0.3's chat reports BY-DESIGN only when Axon's ids are exactly the listed ones; any other
  Mistral difference FAILs; its tokenize and greedy checks still grade; a listed check that came out equal is flagged
  stale; one that wasn't judged stays ERROR.
- one() itself, driven by a fake llama-server and a stub axon: a clean model passes; chat n/a and an axon crash in
  (a) are ERROR, not pass; DATE_FIXES are applied to llama.cpp's rendering (a clock date is no difference); no
  reference, a reference that doesn't start (its (a) and (b) still run), one that is non-finite at every step (round
  2's TypeError) or one that is garbage: ERROR; a hard clause still FAILs without a reference; --llama-ref-extra
  reaches only the matching model's reference.
- run(): a golden.py crash on one model is that model's ERROR and the next model runs; the JSON carries "planned" up
  front and "complete" / "exit" at the end; a crash outside any model exits 2, not FAIL's 1.
- golden-top10.sh with a fake golden.py, llama.cpp and curl: the worst per-model status is its exit code and its
  "done rc=" line; a failed download and a cut-off JSON are not judged; a FAIL beats them.
Any failure is an assertion (rc 1); success prints SELFTEST golden OK.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("golden", os.path.join(HERE, "golden.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
TMP = tempfile.mkdtemp(prefix="golden-selftest-")
m0 = g.new_metrics

# ---- parsers, on output in print_top's format (src/cli.rs: "{:>7}  {:>10.dl$}  {:>8.dp$}  {:?}")
OUT = """prompt ids (4): [1, 450, 3303, 310]
position 3:
top-3 of 32000 (id, logit, logprob, text):
    310   17.250000  -0.120000  "the"
   1023   15.000000  -2.370000  " a"
      7        -inf       -inf  "\\n"
position 4:
top-3 of 32000 (id, logit, logprob, text):
     42         NaN        NaN  "x"
     43    1.000000  -9.500000  "\\"q\\""
"""
pos = g.positions(OUT)
assert sorted(pos) == [3, 4], pos
assert pos[3][:2] == [[310, -0.12], [1023, -2.37]], pos[3]
assert pos[3][2][0] == 7 and pos[3][2][1] == -math.inf, pos[3]
assert pos[4][0][0] == 42 and math.isnan(pos[4][0][1]) and pos[4][1] == [43, -9.5], pos[4]
assert g.ids_line(OUT, "prompt ids") == [1, 450, 3303, 310]
assert g.ids_line("chat prompt ids (3): [1, 2, 3]\n" + OUT, "chat prompt ids") == [1, 2, 3]
assert g.ids_line(OUT, "chat prompt ids") is None   # an axon without AXON_LOGITS_CHAT: golden's chat n/a (ERROR)

# ---- compare / summary
path = [(10, -0.1, [[10, -0.1], [11, -2.5], [12, -4.0]]),    # agree
        (20, -0.6, [[20, -0.6], [21, -0.9], [22, -5.0]]),    # flip: Axon prefers 21
        (30, -0.2, [[30, -0.2], [31, -3.0], [32, -6.0]])]    # llama.cpp's 30 outside Axon's top K
ax = [[[10, -0.12], [11, -2.4], [12, -4.1]],
      [[21, -0.5], [20, -0.95], [22, -5.0]],
      [[31, -0.3], [32, -2.0], [33, -3.0]]]
m = m0()
g.compare(path, ax, "text", m)
s = g.summary(m)
assert s["steps"] == 3 and s["agree"] == 1 and s["agree_frac"] == round(1 / 3, 4), s
assert len(s["flips"]) == 2 and s["missing"] == [{"kind": "text", "step": 2, "llama": 30}], s
f = s["flips"][0]
# llama.cpp's margin for 20 over Axon's 21 (-0.6 - -0.9 = 0.3) + Axon's margin for 21 over 20 (-0.5 - -0.95 = 0.45)
assert (f["llama_gap"], f["other_gap"], f["swing"]) == (0.3, 0.45, 0.75), f
assert s["max_dlp"] == round(0.35, 4), s   # step 1: |(-0.95) - (-0.6)|
# a non-finite Axon logit: counted, never silently dropped (max(0, nan) would be 0)
m = m0()
g.compare([path[0]], [[[10, float("nan")], [11, -2.0]]], "chat", m)
assert m["nonfinite"] == [{"kind": "chat", "step": 0}] and m["steps"] == 0
lim, clear, why, unj = g.judge(g.summary(m))
assert any("non-finite" in w for w in why) and any("no steps" in w for w in why), why
# llama.cpp's own non-finite logprob or a step with no Axon logits: not judged
for bad_path, bad_ax in (([(10, float("nan"), [[10, -0.1]])], [[[10, -0.1]]]), ([path[0]], [None])):
    try:
        g.compare(bad_path, bad_ax, "text", m0())
        raise AssertionError("compare accepted an unjudgeable step")
    except g.NotJudged:
        pass


# ---- kl_top: a KL over the top K plus a tail bucket per side
def softmax_lp(x):
    mx = max(x)
    z = sum(math.exp(v - mx) for v in x)
    return [v - mx - math.log(z) for v in x]


def topk(lp, k=20):
    return [[i, v] for i, v in sorted(enumerate(lp), key=lambda t: -t[1])[:k]]


assert g.kl_top(path[0][2], path[0][2]) == 0.0
la = softmax_lp([-0.05 * i for i in range(200)])
lb = softmax_lp([-0.05 * 1.1 * i for i in range(200)])   # a logit-scale bug: b sharper than a
full = sum(math.exp(x) * (x - y) for x, y in zip(la, lb))
ta, tb = topk(la), topk(lb)
dbb = dict((i, v) for i, v in tb)
old = sum(math.exp(v) * (v - dbb.get(i, tb[-1][1])) for i, v in ta)   # the top-K-only sum golden used before
kl = g.kl_top(ta, tb)
assert old < 0 < kl <= full + 1e-12, (old, kl, full)
# a token of a's top K outside b's: floored at b's K-th; still >= 0
assert g.kl_top([[1, -0.1], [2, -2.5]], [[1, -0.2], [3, -1.9]]) > 0

# ---- judge: hard clauses always, scaled clauses only against a sane spread
clean = {"steps": 100, "agree": 98, "agree_frac": 0.98, "kl_mean": 0.002, "kl_max": 0.05, "max_dlp": 0.2,
         "missing": [], "nonfinite": [], "flips": [{"swing": 0.3}]}
sane = dict(clean, agree=99, agree_frac=0.99, kl_mean=0.001, max_dlp=0.05, flips=[])
wide = dict(sane, kl_mean=0.02, max_dlp=0.8)
lim, clear, why, unj = g.judge(clean, sane)
assert why == [] and clear == [] and unj is None, (why, unj)
assert (lim["kl_mean"], lim["max_dlp"], lim["swing"]) == (g.KL_MEAN_FLOOR, g.DLP_FLOOR, 2 * g.DLP_FLOOR), lim
lim, clear, why, unj = g.judge(clean, wide)
assert (lim["kl_mean"], lim["max_dlp"]) == (round(g.SPREAD_FACTOR * 0.02, 5), round(g.SPREAD_FACTOR * 0.8, 4)), lim
assert lim["from"] == "llama.cpp's spread" and why == [] and unj is None
for bad, word in ((dict(clean, kl_mean=g.KL_MEAN_FLOOR * 1.5), "KL mean"),
                  (dict(clean, max_dlp=g.DLP_FLOOR * 1.5), "max dlogprob"),
                  (dict(clean, flips=[{"swing": 2 * g.DLP_FLOOR + 0.01}]), "swing")):
    lim, clear, why, unj = g.judge(bad, sane)
    assert len(why) == 1 and word in why[0] and unj is None, (word, why)
    # the same numbers without a reference: not judged (ERROR), never FAIL (round 2: floors-only failed clean Gemma)
    lim, clear, why, unj = g.judge(bad)
    assert why == [] and unj and "no --llama-server-ref" in unj and lim["kl_mean"] is None, (word, why, unj)
assert g.judge(dict(clean, kl_mean=0.01), wide)[2] == []   # the KL that fails on the floors passes inside a wide spread
# round 2's clean Gemma 3 4B without a reference: ERROR, not FAIL
gemma = dict(clean, kl_mean=0.0219, max_dlp=1.682, flips=[{"swing": 2.96}])
assert g.judge(gemma)[2] == [] and g.judge(gemma)[3]
# the hard clauses FAIL with or without a reference
for bad, word in ((dict(clean, agree_frac=g.AGREE_MIN - 0.01), "top-1"),
                  (dict(clean, missing=[{"kind": "text", "step": 3, "llama": 9}]), "outside Axon's top"),
                  (dict(clean, nonfinite=[{"kind": "text", "step": 3}]), "non-finite")):
    for sp in (None, sane):
        lim, clear, why, unj = g.judge(bad, sp)
        assert len(why) == 1 and word in why[0], (word, why)
# a reference that isn't sane sets no limits: no finite steps (round 2's TypeError: kl_mean None), its own top-1,
# llama.cpp's token outside its top K
for bad_ref, word in ((dict(sane, steps=0, agree_frac=None, kl_mean=None), "no finite steps"),
                      (dict(wide, agree_frac=0.5), "top-1"),
                      (dict(wide, missing=[{"kind": "chat", "step": 0, "llama": 5}]), "outside its top")):
    lim, clear, why, unj = g.judge(dict(clean, max_dlp=1.5), bad_ref)
    assert why == [] and unj and "not sane" in unj and word in unj and lim["max_dlp"] is None, (word, why, unj)
assert g.ref_problems(sane) == [] and g.ref_problems(None) == ["no finite steps"]


# ---- verdicts and exit codes
def rec(model, **checks):
    return {"model": model, "checks": checks, "problems": []}


ok = {"pass": True}
diff = {"status": "DIFF", "pass": False, "at": 1, "axon": [3, 1763], "llama": [3, 29473]}
r = rec("x.gguf", tokenize=dict(ok), chat={"status": "equal", "pass": True}, greedy=dict(ok))
g.apply_by_design(r)
assert g.verdict(r) == "pass" and r["by_design"] == []
r = rec("x.gguf", tokenize=dict(ok), chat=dict(diff), greedy={"pass": False, "not_judged": True, "error": "axon rc 134"})
assert g.verdict(r) == "FAIL"   # a real difference outranks a check that couldn't run
r = rec("x.gguf", tokenize=dict(ok), chat={"status": "n/a", "pass": False, "not_judged": True, "why": "no hook"},
        greedy=dict(ok))
assert g.verdict(r) == "ERROR" and any("no hook" in p for p in r["problems"]), r
assert g.exit_code(["pass", "pass"]) == 0 and g.exit_code(["pass", "FAIL", "ERROR"]) == 1
for v in ("ERROR", "missing", "llama-error"):
    assert g.exit_code(["pass", v]) == 2, v

# ---- BY_DESIGN: only the listed ids
MI = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
E = g.by_design(MI, "chat")
assert E and "mistral_common" in E["why"] and "src/chat.rs" in E["why"]
assert g.by_design(MI, "greedy") is None and g.by_design(MI, "tokenize") is None
assert g.by_design("Qwen2.5-7B-Instruct-Q4_K_M.gguf", "chat") is None
for e in g.BY_DESIGN:   # every entry names a check golden has, says why with a pointer, and pins Axon's ids
    assert e["check"] in ("tokenize", "chat", "greedy") and len(e["why"]) > 40 and ("src/" in e["why"] or "docs/" in e["why"])
    assert e["axon_ids"] and all(isinstance(x, int) for x in e["axon_ids"]), e
KNOWN = list(E["axon_ids"])
mdiff = dict(diff, axon_ids=KNOWN, llama_ids=[1, 3, 29473] + KNOWN[3:])
r = rec(MI, tokenize=dict(ok), chat=dict(mdiff), greedy=dict(ok))
g.apply_by_design(r)
c = r["checks"]["chat"]
assert g.verdict(r) == "pass" and r["by_design"] == ["chat"], r
assert c["status"] == "BY-DESIGN" and c["was"] == "DIFF" and c["at"] == 1 and c["axon"] == [3, 1763], c
# another Mistral difference: the system message dropped, turns glued, a doubled BOS, no ids at all -> DIFF, FAIL
sys_end = KNOWN.index(781)   # the "\n\n" after the system message
for other in (KNOWN[:2] + KNOWN[sys_end + 2:], KNOWN[:25] + KNOWN[26:], [1] + KNOWN, None):
    r = rec(MI, tokenize=dict(ok), chat=dict(diff, axon_ids=other), greedy=dict(ok))
    g.apply_by_design(r)
    assert g.verdict(r) == "FAIL" and r["checks"]["chat"]["status"] == "DIFF", r
    assert r["checks"]["chat"]["by_design_mismatch"] and r["by_design"] == [] and r["problems"], r
r = rec(MI, tokenize=dict(ok), chat=dict(mdiff), greedy={"pass": False, "why": ["KL mean 0.02 > 0.006"]})
g.apply_by_design(r)
assert g.verdict(r) == "FAIL" and r["checks"]["chat"]["status"] == "BY-DESIGN"   # its numbers still grade
r = rec(MI, tokenize={"pass": False, "equal": 4, "prompts": 5}, chat=dict(mdiff), greedy=dict(ok))
g.apply_by_design(r)
assert g.verdict(r) == "FAIL"   # and its tokenizer
r = rec("Qwen3-8B-Q4_K_M.gguf", tokenize=dict(ok), chat=dict(mdiff), greedy=dict(ok))
g.apply_by_design(r)
assert g.verdict(r) == "FAIL" and r["checks"]["chat"]["status"] == "DIFF"   # the same ids on another model
r = rec(MI, tokenize=dict(ok), chat={"status": "equal", "pass": True}, greedy=dict(ok))
g.apply_by_design(r)
assert g.verdict(r) == "pass" and r["checks"]["chat"].get("by_design_stale") and r["problems"], r
r = rec(MI, tokenize=dict(ok), chat={"status": "ERROR", "pass": False, "not_judged": True, "why": "rc 1"},
        greedy=dict(ok))
g.apply_by_design(r)
assert g.verdict(r) == "ERROR" and r["checks"]["chat"]["status"] == "ERROR"

# ---- DATE_FIXES (normalize_dates, the function one() calls) and tokenize_keeping
kw = g.TEMPLATE_KWARGS
assert g.normalize_dates("<|start|>system<|message|>Knowledge cutoff: 2024-06\nCurrent date: 2026-11-13\n") == \
    (f"<|start|>system<|message|>Knowledge cutoff: 2024-06\nCurrent date: {kw['current_date']}\n",
     [g.DATE_FIXES[0][0].pattern])
assert g.normalize_dates("Cutting Knowledge Date: December 2023\nToday Date: 13 Nov 2026\n\n")[0] == \
    f"Cutting Knowledge Date: December 2023\nToday Date: {kw['date_string']}\n\n"
assert g.normalize_dates("no date here, 2026-11-13") == ("no date here, 2026-11-13", [])


class CharTok:
    def tokenize(self, text, add_special, parse_special):
        return ([1] if add_special else []) + [100 + ord(ch) for ch in text]


assert g.tokenize_keeping(CharTok(), "a </s> b", {212: "</s>"}) == [1, 100 + ord("a"), 100 + ord(" "), 212,
                                                                    100 + ord(" "), 100 + ord("b")]
assert g.tokenize_keeping(CharTok(), "nothing kept", {212: "</s>"}) is None
assert g.OVERRIDE_RE.search("load: control-looking token:    212 '</s>' was not control-type; this is probably a bug")

# ---- one(), with a fake llama-server and a stub axon. Each model file's name picks its scenario: SCEN[name].
# Tokens are characters (100 + ord). llama.cpp's template prints today's date (strftime_now); Axon pins it.
RENDER = "S:{s}|U:{u}|A:{a}|U:{u2}|Current date: {d}|"
CHAT_ARGS = dict(s=g.CHAT[0]["content"], u=g.CHAT[1]["content"], a=g.CHAT[2]["content"], u2=g.CHAT[3]["content"])
LLAMA_TOP = [-0.2, -2.5, -4.0]
SCEN = {}
REFS = []   # (model, extra args) of every reference llama-server started


def top_at(i, lps, ids=(500, 600, 700)):
    return [[t + i, lp] for t, lp in zip(ids, lps)]


class FakeLlama(CharTok):
    def __init__(self, a, model, log, extra=()):
        self.name, self.ref = os.path.basename(model), a.llama_server == "llama-cuda"
        self.sc = SCEN[self.name]
        open(log, "w").close()
        if self.ref:
            REFS.append((self.name, list(extra)))

    def wait(self):
        if self.ref and self.sc.get("ref") == "crash":
            raise RuntimeError("llama-server exited 1")

    def post(self, p, body):
        assert p == "/apply-template" and body["chat_template_kwargs"] == kw
        return {"prompt": RENDER.format(d="2026-11-13", **CHAT_ARGS)}

    def greedy(self, ids, n):
        return [(500 + i, LLAMA_TOP[0], top_at(i, LLAMA_TOP)) for i in range(n)]

    def forced(self, ids, path_):
        ref = self.sc.get("ref", "same")
        out = []
        for i in range(len(path_)):
            if ref == "same":
                out.append(top_at(i, [-0.205, -2.48, -4.02]))
            elif ref == "nan":   # non-finite at every step: round 2's TypeError (kl_mean None)
                out.append(top_at(i, [float("nan"), -2.5, -4.0]))
            elif ref == "garbage":   # top-1 flipped at every step, wildly off: would widen the limits
                out.append(top_at(i, [-0.1, -3.0, -4.0], ids=(600, 500, 700)))
        return out

    def stop(self):
        pass


def stub_axon(a, model, prompt, env, k=1, timeout=900):
    sc = SCEN[os.path.basename(model)]
    mode = sc.get("axon", "clean")
    if "AXON_LOGITS_CHAT" in env:
        assert json.load(open(env["AXON_LOGITS_CHAT"]))["messages"] == g.CHAT
        if mode == "nochat":   # an axon that ignores AXON_LOGITS_CHAT: no "chat prompt ids" line
            return "prompt ids (2): [1, 120]\n"
        ids = [1] + [100 + ord(ch) for ch in RENDER.format(d=kw["current_date"], **CHAT_ARGS)]
        return f"chat prompt ids ({len(ids)}): {json.dumps(ids)}\n"
    if "AXON_LOGITS_IDS" in env:
        seq, pre = env["AXON_LOGITS_IDS"].split(","), int(env["AXON_LOGITS_PREFIX"])
        lps = {"clean": [-0.21, -2.45, -4.05], "far": [-0.6, -1.0, -4.0]}.get(mode, [-0.21, -2.45, -4.05])
        out = []
        for p in range(pre - 1, len(seq)):
            i = p - (pre - 1)
            top = top_at(i, [-0.3, -1.0, -4.0], ids=(600, 601, 700)) if mode == "missing" else top_at(i, lps)
            out.append(f"position {p}:\ntop-3 of 1000 (id, logit, logprob, text):")
            out += [f"{t:>7}  {lp:>10.6f}  {lp:>8.6f}  \"x\"" for t, lp in top]
        return "\n".join(out) + "\n"
    if mode == "crash_a":
        raise RuntimeError("axon logits rc 134: boom")
    if mode == "explode":
        raise ZeroDivisionError("a bug in golden itself")
    ids = [1] + [100 + ord(ch) for ch in prompt]
    return f"prompt ids ({len(ids)}): {json.dumps(ids)}\n"


g.Llama, g.axon = FakeLlama, stub_axon
MD = os.path.join(TMP, "models")
os.makedirs(MD)


def ns(ref=True, extra=None, out=None):
    return argparse.Namespace(axon="axon-stub", llama_server="llama-cpu", llama_server_ref="llama-cuda" if ref else None,
                              llama_ref_ngl=99, llama_ref_extra=extra, models_dir=MD, device="cuda", n=6, threads=1,
                              llama_ngl=0, work=TMP, only=None, out=out or os.path.join(TMP, "golden.json"))


def model(name, **sc):
    SCEN[name] = sc
    open(os.path.join(MD, name), "w").close()
    return name


def one(name, **kw_):
    return g.one(ns(**kw_), name)


r = one(model("clean.gguf"))
c = r["checks"]
assert r["verdict"] == "pass", r
assert c["tokenize"]["equal"] == len(g.TOK_PROMPTS) and c["chat"]["status"] == "equal", c
assert r["date_normalized"] == [g.DATE_FIXES[0][0].pattern]   # the clock date was no difference
assert c["greedy"]["agree"] == 12 and c["greedy"]["limits"]["from"] == "llama.cpp's spread", c["greedy"]
assert c["greedy"]["kl_mean"] >= 0 and "axon_ids" not in c["chat"]
r = one("clean.gguf", ref=False)   # no reference: the scaled clauses are not judged
assert r["verdict"] == "ERROR" and r["checks"]["greedy"]["not_judged"] and not r["checks"]["greedy"]["pass"], r
assert r["checks"]["tokenize"]["pass"] and r["checks"]["chat"]["pass"]
assert any("no --llama-server-ref" in p for p in r["problems"]), r["problems"]
r = one(model("far.gguf", axon="far"))   # Axon 0.4 nats off at every step: over the floor-level limits
assert r["verdict"] == "FAIL" and any("max dlogprob" in w for w in r["checks"]["greedy"]["why"]), r
r = one("far.gguf", ref=False)   # ... the same without a reference: ERROR, never FAIL
assert r["verdict"] == "ERROR", r
r = one(model("garbage.gguf", axon="far", ref="garbage"))   # a garbage reference must not widen the limits to a pass
assert r["verdict"] == "ERROR" and r["checks"]["greedy"]["limits"]["max_dlp"] is None, r
assert "not sane" in r["checks"]["greedy"]["error"], r["checks"]["greedy"]
r = one(model("nanref.gguf", ref="nan"))   # round 2's TypeError repro: ERROR, no exception
assert r["verdict"] == "ERROR" and "no finite steps" in r["checks"]["greedy"]["error"], r
r = one(model("refcrash.gguf", ref="crash"))   # the reference doesn't start: (a) and (b) still run
assert r["verdict"] == "ERROR" and "llama-server exited 1" in r["ref_error"], r
assert r["checks"]["tokenize"]["pass"] and r["checks"]["chat"]["status"] == "equal", r["checks"]
r = one(model("missing.gguf", axon="missing"), ref=False)   # a hard clause FAILs even without a reference
assert r["verdict"] == "FAIL" and any("outside Axon's top" in w for w in r["checks"]["greedy"]["why"]), r
r = one(model("nochat.gguf", axon="nochat"))   # chat n/a: ERROR, never "equal"
assert r["checks"]["chat"]["status"] == "n/a" and r["checks"]["chat"]["not_judged"] and r["verdict"] == "ERROR", r
r = one(model("crash_a.gguf", axon="crash_a"))   # axon crashes in (a): not judged, never a pass
t = r["checks"]["tokenize"]
assert t["not_judged"] and t["pass"] is False and "rc 134" in t["error"] and r["verdict"] == "ERROR", r
REFS.clear()   # --llama-ref-extra: only the matching model's reference gets the args
r = one(model("gpt-oss-20b-MXFP4.gguf"), extra=["gpt-oss=--n-cpu-moe 12", "nomatch=--x"])
r2 = one("clean.gguf", extra=["gpt-oss=--n-cpu-moe 12"])
assert REFS == [("gpt-oss-20b-MXFP4.gguf", ["--n-cpu-moe", "12"]), ("clean.gguf", [])], REFS
assert r["ref_args"] == ["--n-cpu-moe", "12"] and "ref_args" not in r2


# a chat difference keeps both whole id lists (BY_DESIGN compares Axon's)
def mistral_axon(a, model_, prompt, env, k=1, timeout=900):
    if "AXON_LOGITS_CHAT" in env:
        return f"chat prompt ids ({len(KNOWN)}): {json.dumps(KNOWN)}\n"
    return stub_axon(a, model_, prompt, env, k, timeout)


g.axon = mistral_axon
r = one(model(MI))
assert r["checks"]["chat"]["was"] == "DIFF" and r["checks"]["chat"]["status"] == "BY-DESIGN", r["checks"]["chat"]
assert r["checks"]["chat"]["axon_ids"] == KNOWN and r["verdict"] == "pass", r
g.axon = stub_axon

# ---- run(): planned / complete / exit; a crash on one model is that model's ERROR, the next one runs
model("explode.gguf", axon="explode")
seen, orig_one = [], g.one


def spy(a, name):
    seen.append(json.load(open(a.out)))
    return orig_one(a, name)


g.one = spy
a = ns(out=os.path.join(TMP, "run.json"))
a.models = "explode.gguf,clean.gguf,absent.gguf"
with contextlib.redirect_stdout(io.StringIO()) as so, contextlib.redirect_stderr(io.StringIO()):
    rc = g.run(a)
g.one = orig_one
doc = json.load(open(a.out))
assert rc == 2 and doc["exit"] == 2 and doc["complete"] is True, (rc, doc.get("exit"), doc.get("complete"))
assert doc["planned"] == ["explode.gguf", "clean.gguf", "absent.gguf"]
assert [x["verdict"] for x in doc["models"]] == ["ERROR", "pass", "missing"], [x["verdict"] for x in doc["models"]]
assert "ZeroDivisionError" in doc["models"][0]["problems"][0] and "golden.py crashed" in doc["models"][0]["problems"][0]
assert [len(d["models"]) for d in seen] == [0, 1, 2] and all(d["complete"] is False for d in seen), seen
assert seen[0]["planned"] == doc["planned"] and "exit" not in seen[1]
assert "GOLDEN 3 models: pass 1, FAIL 0, ERROR 1, missing 1" in so.getvalue()
# a crash outside any model (here: --out in a directory that doesn't exist): exit 2, not FAIL's 1
p = subprocess.run([sys.executable, os.path.join(HERE, "golden.py"), "run", "--axon", "/nonexistent/axon",
                    "--llama-server", "/nonexistent/llama-server", "--models-dir", MD, "--models", "clean.gguf",
                    "--out", os.path.join(TMP, "no-such-dir", "x.json")], capture_output=True, text=True, timeout=60)
assert p.returncode == 2 and "crashed (exit 2" in p.stderr, (p.returncode, p.stderr[-300:])

# ---- golden-top10.sh: its exit code is the worst per-model status
W = os.path.join(TMP, "top10")
os.makedirs(os.path.join(W, "models"))
bins = os.path.join(TMP, "bin")
os.makedirs(bins)


def script(path, body):
    with open(path, "w") as fh:
        fh.write(body)
    os.chmod(path, 0o755)


# fake golden.py: the model file's name says what it does (pass / FAIL / ERROR / a crash with no JSON / cut off)
script(os.path.join(TMP, "fake_golden.py"), """#!/usr/bin/env python3
import json, sys
a = sys.argv
if a[1] == "table":
    sys.exit(0)
m, out = a[a.index("--models") + 1], a[a.index("--out") + 1]
kind = m.split("-")[0]
if kind == "crash":
    sys.exit(1)   # a traceback's exit 1, no JSON: must not count as FAIL
rc = {"pass": 0, "fail": 1, "error": 2, "cut": 1}[kind]
v = {0: "pass", 1: "FAIL", 2: "ERROR"}[rc]
doc = {"planned": [m], "complete": kind != "cut", "models": [{"model": m, "verdict": v, "checks": {}}]}
if kind != "cut":
    doc["exit"] = rc
json.dump(doc, open(out, "w"))
print("GOLDEN 1 models: " + v)
sys.exit(rc)
""")
CURL_LOG = os.path.join(TMP, "curl.log")
script(os.path.join(bins, "curl"), f'#!/bin/sh\necho "$@" >> {CURL_LOG}\nexit 22\n')   # every download fails (no network)
llama_dir = os.path.join(TMP, "llama-b0")
os.makedirs(llama_dir)
script(os.path.join(llama_dir, "llama-server"), "#!/bin/sh\necho 'version: 0 (fake)'\n")
tgz = os.path.join(TMP, "llama.tar.gz")
with tarfile.open(tgz, "w:gz") as tf:
    tf.add(llama_dir, arcname="llama-b0")
script(os.path.join(TMP, "axon"), "#!/bin/sh\n")


def top10(entries, only=""):
    shutil.rmtree(os.path.join(W, "gw"), ignore_errors=True)
    lst = []
    for name, kind in entries:
        f = f"{kind}-{name}.gguf"
        lst.append({"ollama": name, "repo": "o/r", "revision": "0" * 40, "file": f, "sha256": f"sha-{name}"})
        if kind != "nofile":
            open(os.path.join(W, "models", f), "w").close()
            with open(os.path.join(W, "models", f + ".sha256"), "w") as fh:
                fh.write(f"sha-{name}  {f}\n")
    json.dump(lst, open(os.path.join(TMP, "list.json"), "w"))
    env = dict(os.environ, PATH=bins + os.pathsep + os.environ["PATH"], GOLDEN_NO_PREFLIGHT="1", AXON_THREADS="1",
               AXON_MODELS=os.path.join(W, "models"))
    p = subprocess.run(["bash", os.path.join(HERE, "golden-top10.sh"), os.path.join(TMP, "fake_golden.py"),
                        os.path.join(TMP, "list.json"), os.path.join(TMP, "axon"), tgz,
                        os.path.join(llama_dir, "llama-server")] + ([only] if only else []),
                       cwd=W, env=env, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


for entries, want in (([("a", "pass"), ("b", "pass")], 0),
                      ([("a", "pass"), ("b", "error")], 2),
                      ([("a", "pass"), ("b", "nofile")], 2),     # a failed download: not judged, never "done" green
                      ([("a", "pass"), ("b", "crash")], 2),      # golden.py's exit 1 without a JSON: not a FAIL
                      ([("a", "pass"), ("b", "cut")], 2),        # a cut-off JSON (complete false)
                      ([("a", "fail"), ("b", "nofile"), ("c", "error")], 1),    # a FAIL beats not judged,
                      ([("a", "error"), ("b", "pass"), ("c", "fail")], 1)):     # whichever comes first
    rc, out = top10(entries)
    assert rc == want and f"GOLDEN-TOP10 done rc={want}\n" in out, (entries, rc, out[-1500:])
    np_ = sum(1 for _, k in entries if k == "pass")
    assert f"SUMMARY pass {np_}," in out, out[-800:]
rc, out = top10([("a", "pass"), ("b", "fail"), ("c", "pass")], only="c,a")   # ONLY picks the models and their order
assert rc == 0 and out.index("GOLDEN-MODEL c ") < out.index("GOLDEN-MODEL a ") and "GOLDEN-MODEL b " not in out, out
open(CURL_LOG, "w").close()
rc, out = top10([("a", "nofile")])   # the download asks for the pinned revision, never resolve/main
assert rc == 2 and "fetch-failed" in out and "SUMMARY pass 0, FAIL 0, not judged 1 of 1" in out, out[-800:]
assert "https://huggingface.co/o/r/resolve/" + "0" * 40 + "/nofile-a.gguf" in open(CURL_LOG).read()

shutil.rmtree(TMP)
print("SELFTEST golden OK")
