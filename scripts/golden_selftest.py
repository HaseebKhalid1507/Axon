#!/usr/bin/env python3
"""Offline self-test of golden.py: stdlib only, no engine, no model, no llama.cpp, no network, well under a second
(CI runs it beside agent_replay_selftest.py).

- The parsers on canned `axon logits` output: positions() (AXON_LOGITS_PREFIX's per-position top-K, inf / NaN too)
  and ids_line() ("prompt ids" and "chat prompt ids"; a missing line is None, which golden reports as chat n/a).
- compare() / summary() / judge() on synthetic steps: agreement, a flip and its swing, llama.cpp's token outside
  Axon's top K, a non-finite logit, KL over llama.cpp's top K; limits from llama.cpp's spread vs the floors.
- The verdicts: FAIL beats ERROR, a not-judged check is ERROR, exit codes 0 / 1 / 2.
- BY_DESIGN: Mistral v0.3's chat DIFF reports BY-DESIGN and the model passes, while its tokenize and greedy checks
  still grade (a greedy failure still FAILs it); the same DIFF on another model FAILs; a listed check that came out
  equal is flagged stale; a listed check that wasn't judged stays ERROR.
- DATE_FIXES (gpt-oss's strftime_now date, Llama 3's "Today Date") and tokenize_keeping() with a fake tokenizer.
Any failure is an assertion (rc 1); success prints SELFTEST golden OK.
"""
import importlib.util
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("golden", os.path.join(HERE, "golden.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def m0():
    return {"steps": 0, "agree": 0, "kl": [], "dlp": [], "missing": [], "nonfinite": [], "flips": []}


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

# ---- compare / summary / judge
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
kl0 = math.exp(-0.1) * (-0.1 + 0.12) + math.exp(-2.5) * (-2.5 + 2.4) + math.exp(-4.0) * (-4.0 + 4.1)
assert abs(g.kl_top(path[0][2], ax[0]) - kl0) < 1e-12
# a non-finite Axon logit: counted, never silently dropped (max(0, nan) would be 0)
m = m0()
g.compare([path[0]], [[[10, float("nan")], [11, -2.0]]], "chat", m)
assert m["nonfinite"] == [{"kind": "chat", "step": 0}] and m["steps"] == 0
lim, clear, why = g.judge(g.summary(m))
assert any("non-finite" in w for w in why) and any("no steps" in w for w in why), why
# llama.cpp's own non-finite logprob or a step with no Axon logits: not judged
for bad_path, bad_ax in (([(10, float("nan"), [[10, -0.1]])], [[[10, -0.1]]]), ([path[0]], [None])):
    try:
        g.compare(bad_path, bad_ax, "text", m0())
        raise AssertionError("compare accepted an unjudgeable step")
    except g.NotJudged:
        pass

# judge: floors without a spread, SPREAD_FACTOR x the spread above them; each limit fails on its own
clean = {"steps": 100, "agree": 98, "agree_frac": 0.98, "kl_mean": 0.002, "kl_max": 0.05, "max_dlp": 0.2,
         "missing": [], "nonfinite": [], "flips": [{"swing": 0.3}]}
lim, clear, why = g.judge(clean)
assert why == [] and clear == [], why
assert (lim["kl_mean"], lim["max_dlp"], lim["swing"]) == (g.KL_MEAN_FLOOR, g.DLP_FLOOR, 2 * g.DLP_FLOOR), lim
wide = dict(clean, kl_mean=0.02, max_dlp=0.8)
lim, clear, why = g.judge(clean, wide)
assert (lim["kl_mean"], lim["max_dlp"]) == (round(g.SPREAD_FACTOR * 0.02, 5), round(g.SPREAD_FACTOR * 0.8, 4)), lim
assert lim["from"] == "llama.cpp's spread" and why == []
for bad, word in ((dict(clean, kl_mean=g.KL_MEAN_FLOOR * 1.5), "KL mean"),
                  (dict(clean, max_dlp=g.DLP_FLOOR * 1.5), "max dlogprob"),
                  (dict(clean, agree_frac=g.AGREE_MIN - 0.01), "top-1"),
                  (dict(clean, flips=[{"swing": 2 * g.DLP_FLOOR + 0.01}]), "swing"),
                  (dict(clean, missing=[{"kind": "text", "step": 3, "llama": 9}]), "outside Axon's top")):
    lim, clear, why = g.judge(bad)
    assert len(why) == 1 and word in why[0], (word, why)
# the same KL that fails on the floors passes inside a wide spread
assert g.judge(dict(clean, kl_mean=0.01), wide)[2] == []


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

# ---- BY_DESIGN
MI = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
assert g.by_design(MI, "chat") and "mistral_common" in g.by_design(MI, "chat") and "src/chat.rs" in g.by_design(MI, "chat")
assert g.by_design(MI, "greedy") is None and g.by_design(MI, "tokenize") is None
assert g.by_design("Qwen2.5-7B-Instruct-Q4_K_M.gguf", "chat") is None
for rx, check, why in g.BY_DESIGN:   # every entry names a check golden has, and says why with a pointer
    assert check in ("tokenize", "chat", "greedy") and len(why) > 40 and ("src/" in why or "docs/" in why), why
r = rec(MI, tokenize=dict(ok), chat=dict(diff), greedy=dict(ok))
g.apply_by_design(r)
c = r["checks"]["chat"]
assert g.verdict(r) == "pass" and r["by_design"] == ["chat"], r
assert c["status"] == "BY-DESIGN" and c["was"] == "DIFF" and c["at"] == 1 and c["axon"] == [3, 1763], c
r = rec(MI, tokenize=dict(ok), chat=dict(diff), greedy={"pass": False, "why": ["KL mean 0.02 > 0.006"]})
g.apply_by_design(r)
assert g.verdict(r) == "FAIL" and r["checks"]["chat"]["status"] == "BY-DESIGN"   # its numbers still grade
r = rec(MI, tokenize={"pass": False, "equal": 4, "prompts": 5}, chat=dict(diff), greedy=dict(ok))
g.apply_by_design(r)
assert g.verdict(r) == "FAIL"   # and its tokenizer
r = rec("Qwen3-8B-Q4_K_M.gguf", tokenize=dict(ok), chat=dict(diff), greedy=dict(ok))
g.apply_by_design(r)
assert g.verdict(r) == "FAIL" and r["checks"]["chat"]["status"] == "DIFF"
r = rec(MI, tokenize=dict(ok), chat={"status": "equal", "pass": True}, greedy=dict(ok))
g.apply_by_design(r)
assert g.verdict(r) == "pass" and r["checks"]["chat"].get("by_design_stale") and r["problems"], r
r = rec(MI, tokenize=dict(ok), chat={"status": "ERROR", "pass": False, "not_judged": True, "why": "rc 1"},
        greedy=dict(ok))
g.apply_by_design(r)
assert g.verdict(r) == "ERROR" and r["checks"]["chat"]["status"] == "ERROR"

# ---- DATE_FIXES and tokenize_keeping
kw = g.TEMPLATE_KWARGS


def fix(text):
    for rx, sub in g.DATE_FIXES:
        text = rx.sub(sub.format(**kw), text)
    return text


assert fix("<|start|>system<|message|>Knowledge cutoff: 2024-06\nCurrent date: 2026-11-13\n") == \
    f"<|start|>system<|message|>Knowledge cutoff: 2024-06\nCurrent date: {kw['current_date']}\n"
assert fix("Cutting Knowledge Date: December 2023\nToday Date: 13 Nov 2026\n\n") == \
    f"Cutting Knowledge Date: December 2023\nToday Date: {kw['date_string']}\n\n"
assert fix("no date here, 2026-11-13") == "no date here, 2026-11-13"


class FakeLlama:
    def tokenize(self, text, add_special, parse_special):
        return ([1] if add_special else []) + [100 + ord(ch) for ch in text]


assert g.tokenize_keeping(FakeLlama(), "a </s> b", {212: "</s>"}) == [1, 100 + ord("a"), 100 + ord(" "), 212,
                                                                      100 + ord(" "), 100 + ord("b")]
assert g.tokenize_keeping(FakeLlama(), "nothing kept", {212: "</s>"}) is None
assert g.OVERRIDE_RE.search("load: control-looking token:    212 '</s>' was not control-type; this is probably a bug")

print("SELFTEST golden OK")
