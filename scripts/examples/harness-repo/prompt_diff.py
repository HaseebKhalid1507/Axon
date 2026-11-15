#!/usr/bin/env python3
"""prompt_diff.py: the prompt each engine builds from the same harness request (docs/log/2026-11-13-harness-accept.md).

    prompt_diff.py --axon AXON --llama-server LLAMA_SERVER --model GGUF [--threads N] BODY.json...

BODY.json is a request a harness sent, as `harness_accept.py capture` saves it. For each one:
  1. Both servers, on the CPU, are sent the body with one output token (axon serve; llama-server --jinja, as
     harness_accept.py runs it): the prompt token count each engine reports.
  2. A chat-completions body is also rendered without serving it: Axon's ids from `axon logits` with
     AXON_LOGITS_CHAT=BODY (the prompt serve builds), llama.cpp's from /apply-template, then /tokenize. Both are
     decoded with llama-server's /detokenize and diffed line by line; the counts must equal step 1's.
A Responses API body (codex) has no rendering endpoint on either engine: step 1 only.
Stdlib only. Exit 0 if every body was measured, 1 otherwise.
"""
import argparse, difflib, json, os, re, socket, subprocess, sys, tempfile, time, urllib.error, urllib.request


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(port, path, obj, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:400].decode("utf-8", "replace")}


def up(argv, port, log, health="/health", secs=300):
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    end = time.time() + secs
    while time.time() < end:
        if p.poll() is not None:
            raise RuntimeError("%s exited %s" % (argv[0], p.returncode))
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, health), timeout=2) as r:
                if r.status == 200:
                    return p
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    p.kill()
    raise RuntimeError("%s not healthy in %d s" % (argv[0], secs))


def served_count(port, path, body):
    """The prompt tokens the server reports for BODY with one output token, not streamed."""
    b = dict(body, stream=False)
    for k in ("stream_options", "max_tokens", "max_completion_tokens", "max_output_tokens"):
        b.pop(k, None)
    b["max_output_tokens" if path.endswith("responses") else "max_tokens"] = 1
    st, r = post(port, path, b)
    u = r.get("usage") or {}
    n = u.get("input_tokens", u.get("prompt_tokens"))
    return n, (None if st == 200 else "%s %s" % (st, str(r.get("error"))[:300]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axon", required=True)
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("bodies", nargs="+")
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="prompt-diff-")
    bodies = [(os.path.basename(p), json.load(open(p))) for p in a.bodies]
    route = lambda b: "/v1/responses" if "input" in b and "messages" not in b else "/v1/chat/completions"
    bad = 0

    # llama.cpp: the server harness_accept.py runs (--jinja), on the CPU; renders, tokenizes, serves
    lp = free_port()
    llog = open(os.path.join(work, "llama.log"), "w")
    ls = up([a.llama_server, "-m", a.model, "--host", "127.0.0.1", "--port", str(lp), "--jinja", "-c", str(a.ctx),
             "-np", "1", "-ngl", "0", "-t", str(a.threads), "--no-warmup"], lp, llog)
    ver = subprocess.run([a.llama_server, "--version"], capture_output=True, text=True).stderr.strip().splitlines()
    print("PROMPT-DIFF llama-server %s; axon %s; model %s" % (
        " ".join(ver[-2:]) if ver else "?", os.path.basename(a.axon), os.path.basename(a.model)), flush=True)
    rows = {}
    for name, b in bodies:
        r = route(b)
        n, err = served_count(lp, r, b)
        rows[name] = dict(route=r, ntools=len(b.get("tools") or []), llama_served=n, llama_err=err)
        if r.endswith("completions"):
            st, t = post(lp, "/apply-template", {k: b[k] for k in ("messages", "tools") if k in b})
            text = t.get("prompt", "")
            st, tk = post(lp, "/tokenize", {"content": text, "add_special": True, "parse_special": True})
            rows[name].update(llama_text=text, llama_ids=tk.get("tokens") or [])

    # Axon: the ids serve renders (AXON_LOGITS_CHAT: printed before any compute; killed then), and serve's count
    for name, b in bodies:
        if not rows[name]["route"].endswith("completions"):
            continue
        f = os.path.join(work, name)
        json.dump(b, open(f, "w"))
        env = dict(os.environ, AXON_LOGITS_CHAT=f)
        p = subprocess.Popen([a.axon, "logits", a.model, "x", "--device", "cpu"], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        ids, tail = None, []
        for ln in p.stdout:
            m = re.match(r"chat prompt ids \((\d+)\): \[(.*)\]", ln.strip())
            if m:
                ids = [int(x) for x in m.group(2).split(",") if x.strip()]
                break
            tail.append(ln.rstrip()[:200])
        p.kill()
        p.wait()
        rows[name]["axon_ids"] = ids
        if ids is None:
            rows[name]["axon_err"] = " | ".join(tail[-3:])
        else:  # decoded with llama.cpp's detokenizer (the same vocabulary): text to diff
            st, d = post(lp, "/detokenize", {"tokens": ids})
            rows[name]["axon_text"] = d.get("content", "")
    ap_ = free_port()
    alog = open(os.path.join(work, "axon.log"), "w")
    try:
        ax = up([a.axon, "serve", a.model, "--host", "127.0.0.1", "--port", str(ap_), "--device", "cpu",
                 "--ctx", str(a.ctx)], ap_, alog)
        try:
            for name, b in bodies:
                n, err = served_count(ap_, rows[name]["route"], b)
                rows[name].update(axon_served=n, axon_err_served=err)
        finally:
            ax.kill()
            ax.wait()
    except RuntimeError as e:
        print("PROMPT-DIFF axon serve: %s" % e, flush=True)
    ls.kill()
    ls.wait()

    for name, _ in bodies:
        r = rows[name]
        ai, li = r.get("axon_ids"), r.get("llama_ids")
        print("PROMPT-DIFF %-18s %-22s tools %2d | served prompt tokens: axon %s, llama %s (gap %s)%s%s" % (
            name, r["route"], r["ntools"], r.get("axon_served"), r.get("llama_served"),
            (r["axon_served"] - r["llama_served"]) if isinstance(r.get("axon_served"), int)
            and isinstance(r.get("llama_served"), int) else "?",
            "" if not r.get("llama_err") else " llama error: " + r["llama_err"],
            "" if not r.get("axon_err_served") else " axon error: " + r["axon_err_served"]), flush=True)
        if r.get("axon_served") is None or r.get("llama_served") is None:
            bad += 1
        if not r["route"].endswith("completions"):
            continue
        if ai is None or li is None:
            print("    rendering: axon %s" % r.get("axon_err"), flush=True)
            bad += 1
            continue
        same = ai == li
        print("    rendered: axon %d ids (= served: %s), llama.cpp %d ids (= served: %s), ids %s" % (
            len(ai), len(ai) == r.get("axon_served"), len(li), len(li) == r.get("llama_served"),
            "EQUAL" if same else "DIFFER"), flush=True)
        if not same:
            at, lt = r.get("axon_text", ""), r.get("llama_text", "")
            # one tool per line in both templates: a line diff shows the per-tool difference
            d = list(difflib.unified_diff(lt.splitlines(), at.splitlines(), "llama.cpp", "axon", n=0, lineterm=""))
            print("    text diff (- llama.cpp, + axon): %d lines differ" % sum(
                1 for x in d if x[:1] in "+-" and x[:3] not in ("+++", "---")), flush=True)
            for x in d[:60]:
                print("      " + x[:400], flush=True)
            # the same line, character by character, for the first differing pair
            for x, y in zip([x for x in d if x.startswith("-") and not x.startswith("---")],
                            [y for y in d if y.startswith("+") and not y.startswith("+++")]):
                sm = difflib.SequenceMatcher(None, x[1:], y[1:], autojunk=False)
                ops = [(op, x[1:][i1:i2], y[1:][j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
                print("    first changed line, edits (llama.cpp -> axon): %s" % ops[:8], flush=True)
                break
    print("PROMPT-DIFF %s (%d bodies, %d not measured)" % ("DONE" if not bad else "INCOMPLETE", len(bodies), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
