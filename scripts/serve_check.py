#!/usr/bin/env python3
"""End-to-end check of `axon serve` against `axon gen` (stdlib only; the
official `openai` client too if it imports).

    scripts/serve_check.py <axon binary> <model.gguf> [--device cuda]

Checks: greedy /v1/completions text == `axon gen` text, byte for byte,
whole and streamed; streamed chat == whole chat; with the `openai` package:
models.list, completions and chat completions (whole and streamed) give the
same text. Exits non-zero on a mismatch.
"""

import json
import re
import subprocess
import sys
import urllib.request

PROMPT = "The capital of France is"
N = 32


def post(url, path, body):
    req = urllib.request.Request(
        url + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read().decode()


def sse_text(raw, chat):
    out, done = "", False
    for ev in raw.split("\n\n"):
        if not ev:
            continue
        assert ev.startswith("data: "), ev
        d = ev[6:]
        if d == "[DONE]":
            done = True
            continue
        v = json.loads(d)
        for c in v.get("choices", []):
            out += (c["delta"].get("content") or "") if chat else c["text"]
    assert done, "no [DONE]"
    return out


def main():
    axon, model = sys.argv[1], sys.argv[2]
    extra = sys.argv[3:]
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}" + ("" if good else f": {got!r} != {want!r}"))

    g = subprocess.run(
        [axon, "gen", model, PROMPT, "-n", str(N), *extra],
        capture_output=True,
        check=True,
    )
    out = g.stdout.decode()
    assert out.startswith(PROMPT) and out.endswith("\n"), out
    gen_text = out[len(PROMPT) : -1]
    print(f"axon gen: {gen_text!r}")

    srv = subprocess.Popen(
        [axon, "serve", model, "--port", "0", *extra],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        url = None
        for line in srv.stderr:
            sys.stderr.write(line)
            m = re.search(r"on (http://\S+/v1)", line)
            if m:
                url = m.group(1)
                break
        assert url, "the server did not start"
        base = url[: -len("/v1")]

        req = {"prompt": PROMPT, "max_tokens": N, "temperature": 0}
        v = json.loads(post(base, "/v1/completions", req))
        check("completions == axon gen", v["choices"][0]["text"], gen_text)
        raw = post(base, "/v1/completions", {**req, "stream": True})
        check("streamed completions == axon gen", sse_text(raw, False), gen_text)

        creq = {
            "messages": [{"role": "user", "content": "Name three rivers."}],
            "max_tokens": 40,
            "temperature": 0,
        }
        whole = json.loads(post(base, "/v1/chat/completions", creq))["choices"][0]["message"]["content"]
        streamed = sse_text(post(base, "/v1/chat/completions", {**creq, "stream": True}), True)
        check("streamed chat == whole chat", streamed, whole)

        try:
            import openai
        except ImportError:
            print("skip openai client checks: `import openai` fails")
        else:
            c = openai.OpenAI(base_url=url, api_key="none")
            mid = c.models.list().data[0].id
            r = c.completions.create(model=mid, prompt=PROMPT, max_tokens=N, temperature=0)
            check("openai completions == axon gen", r.choices[0].text, gen_text)
            s = c.completions.create(
                model=mid, prompt=PROMPT, max_tokens=N, temperature=0, stream=True
            )
            check("openai streamed completions == axon gen", "".join(x.choices[0].text for x in s), gen_text)
            r = c.chat.completions.create(model=mid, **creq)
            check("openai chat == whole chat", r.choices[0].message.content, whole)
            check("openai usage", r.usage.completion_tokens > 0, True)
            s = c.chat.completions.create(
                model=mid, stream=True, stream_options={"include_usage": True}, **creq
            )
            text, usage = "", None
            for x in s:
                if x.choices and x.choices[0].delta.content:
                    text += x.choices[0].delta.content
                if x.usage:
                    usage = x.usage
            check("openai streamed chat == whole chat", text, whole)
            check("openai streamed usage", usage is not None and usage.completion_tokens > 0, True)
            try:
                c.chat.completions.create(model="nope", **creq)
                check("openai unknown model -> NotFoundError", False, True)
            except openai.NotFoundError:
                check("openai unknown model -> NotFoundError", True, True)
    finally:
        srv.kill()
        srv.wait()
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
