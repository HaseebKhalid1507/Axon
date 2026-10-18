#!/usr/bin/env python3
"""Per-position top-K logprobs from axon and from llama.cpp (llama-server) on the
same token ids, and their agreement: a teacher-forced version of llama_logits.py
(every one of the last N positions of each sequence, not just the next token).

    positions.py seqs     --hf TOKENIZER_DIR --out S.json [--texts F] [-n 64]
    positions.py dump-axon  --axon BIN --model M --seqs S.json [-k 20] [-n 64] --out A.json [-- axon flags]
    positions.py dump-axon  --log LOG --model M --seqs S.json ... (the output of a pod job, see `env`)
    positions.py env --model M --seqs S.json [-n 64] --out E.sh   (AXON_LOGITS_IDS/ALL for such a job)
    positions.py dump-llama --server BIN --model M --seqs S.json [-k 20] [-n 64] --out L.json [-- server flags]
    positions.py compare A.json B.json [C.json ...]

seqs: the token ids of each text by Hugging Face's tokenizer (axon's kimi-k2 is
checked against it in the tests), cut to their first `-n` + 32 tokens. Both sides then see exactly these ids: axon through
AXON_LOGITS_IDS / AXON_LOGITS_ALL, llama-server through token-list prompts
(cache_prompt on: each position extends the last).
compare: top-1 agreement over all positions, KL(P_a || P_b) over a's top K (b
floored at its K-th logprob), max |dlogprob| of a's top-1.
"""
import argparse, json, math, os, re, socket, subprocess, sys, time, urllib.request

TEXTS = [
    "The history of the Roman Empire spans more than a thousand years, from the founding of the city to the fall of Constantinople. Historians often divide it into the Republic, the Principate and the Dominate, each with its own institutions and conflicts.",
    "def merge_sort(xs):\n    if len(xs) <= 1:\n        return xs\n    mid = len(xs) // 2\n    left = merge_sort(xs[:mid])\n    right = merge_sort(xs[mid:])\n    out = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    return out + left[i:] + right[j:]\n",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
    "Q: A train leaves the station at 3 pm travelling at 60 km/h. A second train leaves at 4 pm at 90 km/h on the same track. When does the second train catch up?\nA: Let t be the hours after 3 pm. The first train covers 60t, the second 90(t - 1).",
    "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|><|im_user|>user<|im_middle|>Explain what a hash table is, in two short paragraphs.<|im_end|><|im_assistant|>assistant<|im_middle|>A hash table is a data structure that maps keys to values. It uses a hash function to compute an index into an array of buckets.",
    "東京は日本の首都であり、世界最大級の都市圏を形成している。江戸時代には江戸と呼ばれ、徳川幕府の所在地として栄えた。明治維新の後、東京と改称された。",
    "Le français est une langue indo-européenne de la famille des langues romanes. Il est parlé en France, en Belgique, en Suisse, au Canada et dans de nombreux pays d'Afrique, où il sert souvent de langue administrative.",
    "SELECT customers.name, COUNT(orders.id) AS n_orders\nFROM customers\nLEFT JOIN orders ON orders.customer_id = customers.id\nWHERE customers.created_at > '2023-01-01'\nGROUP BY customers.name\nHAVING COUNT(orders.id) > 5\nORDER BY n_orders DESC;",
    "Once upon a time, in a kingdom by the sea, there lived a young princess who loved to explore the cliffs above the harbour. Every morning she would slip away from the castle before the guards awoke and climb down to the rocks.",
    "The derivative of a function measures how its output changes as its input changes. For f(x) = x^3 - 2x + 1, the derivative is f'(x) = 3x^2 - 2, which is zero at x = ±sqrt(2/3).",
    "fn main() {\n    let mut counts = std::collections::HashMap::new();\n    for word in std::io::stdin().lines().flatten().flat_map(|l| l.split_whitespace().map(str::to_owned).collect::<Vec<_>>()) {\n        *counts.entry(word).or_insert(0usize) += 1;\n    }\n    println!(\"{}\", counts.len());\n}\n",
    "Machine learning models are trained by minimizing a loss function over a dataset. Gradient descent updates the parameters in the direction that reduces the loss, with a step size called the learning rate.",
    "Der Rhein ist einer der längsten und wichtigsten Flüsse Europas. Er entspringt in den Schweizer Alpen, fließt durch den Bodensee und mündet schließlich in den Niederlanden in die Nordsee.",
    "1. Preheat the oven to 180 degrees Celsius.\n2. Mix the flour, sugar and baking powder in a large bowl.\n3. Add the eggs, milk and melted butter, and stir until smooth.\n4. Pour the batter into a greased tin and bake for 35 minutes.",
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms. Mitochondria generate most of the cell's supply of adenosine triphosphate, used as a source of chemical energy.",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. Neil Armstrong and Buzz Aldrin spent about two and a quarter hours outside the lunar module, while Michael Collins orbited above in the command module.",
]


def axon_run(axon, model, prompt, k, extra, env):
    out = subprocess.run([axon, "logits", model, prompt, "-k", str(k)] + extra,
                         capture_output=True, text=True, check=True,
                         env=dict(os.environ, AXON_LOGITS_DIGITS="6", **env)).stdout
    return out


def parse_top(lines, k):
    top = []
    for line in lines:
        m = re.match(r"\s*(\d+)\s+(-?[\d.]+|-?inf|NaN)\s+(-?[\d.]+|-?inf)\s+\"", line)
        if m:
            top.append([int(m.group(1)), float(m.group(3))])
    return top[:k]


def seqs(a):
    # Hugging Face's tokenizer (the reference axon's is tested against;
    # special-token text parsed, as HF does by default).
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.hf, trust_remote_code=True)
    texts = json.load(open(a.texts)) if a.texts else TEXTS
    res = []
    for i, t in enumerate(texts):
        # Short texts run on into the next ones (n + 32 tokens each).
        j = i
        while len(tk.encode(t, add_special_tokens=False)) < a.n + 32:
            j += 1
            t += "\n\n" + texts[j % len(texts)]
        ids = tk.encode(t, add_special_tokens=False)
        res.append({"text": t, "ids": ids[: a.n + 32]})
        print(len(res), len(ids), file=sys.stderr)
    json.dump(res, open(a.out, "w"))


def parse_axon(out, seqs, n, k):
    """`axon logits` output with AXON_LOGITS_ALL over several sequences."""
    parts = re.split(r"^prompt ids \(\d+\): .*$", out, flags=re.M)[1:]
    assert len(parts) == len(seqs), (len(parts), len(seqs))
    res = []
    for s, part in zip(seqs, parts):
        ids = s["ids"]
        m = min(n, len(ids))
        blocks = re.split(r"^position (\d+):$", part, flags=re.M)
        tops = [parse_top(blocks[i + 1].splitlines(), k) for i in range(1, len(blocks), 2)]
        assert len(tops) == m, (len(tops), m)
        res.append({"ids": ids, "first": len(ids) - m, "tops": tops})
    return res


def axon_env(seqs, n):
    return {"AXON_LOGITS_IDS": ";".join(",".join(map(str, s["ids"])) for s in seqs),
            "AXON_LOGITS_ALL": str(n)}


def dump_axon(a):
    seqs = json.load(open(a.seqs))
    if a.log:
        out = open(a.log).read()
    else:
        out = axon_run(a.axon, a.model, "x", a.k, a.extra, axon_env(seqs, a.n))
    json.dump(parse_axon(out, seqs, a.n, a.k), open(a.out, "w"))


def env_for(a):
    # The env a job running `axon logits` by itself needs (pods run binaries only).
    e = axon_env(json.load(open(a.seqs)), a.n)
    with open(a.out, "w") as f:
        for k, v in e.items():
            f.write(f"export {k}='{v}'\n")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def dump_llama(a):
    port = free_port()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.dirname(a.server) + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [a.server, "-m", a.model, "--port", str(port), "-c", "1024", "-np", "1", "--no-warmup"] + a.extra
    log = open(a.out + ".server.log", "w")
    srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(1200):
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
        for s in json.load(open(a.seqs)):
            ids = s["ids"]
            n = min(a.n, len(ids))
            tops = []
            for p in range(len(ids) - n, len(ids)):
                body = json.dumps({"prompt": ids[: p + 1], "n_predict": 1, "n_probs": a.k,
                                   "temperature": 0, "cache_prompt": True,
                                   "ignore_eos": True}).encode()
                req = urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=600))
                cp = r["completion_probabilities"][0]
                tops.append([[t["id"], t["logprob"]] for t in cp["top_logprobs"]][: a.k])
            res.append({"ids": ids, "first": len(ids) - n, "tops": tops})
            print(len(res), tops[-1][0], file=sys.stderr)
        json.dump(res, open(a.out, "w"))
    finally:
        srv.terminate(); srv.wait()


def kl_top(pa, pb):
    db = dict((i, lp) for i, lp in pb)
    floor = min(lp for _, lp in pb)
    return sum(math.exp(lp) * (lp - db.get(i, floor)) for i, lp in pa)


def compare(files):
    runs = [(f, json.load(open(f))) for f in files]
    print(f"{'pair':<66} {'top1':>11} {'KL mean':>9} {'KL p99':>9} {'KL max':>9} {'max|dlp1|':>9}")
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            (fa, a), (fb, b) = runs[i], runs[j]
            assert [e["ids"] for e in a] == [e["ids"] for e in b], "different inputs"
            agree = tot = 0
            kls, dl = [], 0.0
            for x, y in zip(a, b):
                for ta, tb in zip(x["tops"], y["tops"]):
                    tot += 1
                    agree += ta[0][0] == tb[0][0]
                    kls.append(kl_top(ta, tb))
                    db = dict((t, lp) for t, lp in tb)
                    if ta[0][0] in db:
                        dl = max(dl, abs(ta[0][1] - db[ta[0][0]]))
            kls.sort()
            name = f"{os.path.basename(fa)} vs {os.path.basename(fb)}"
            print(f"{name:<66} {agree:>5}/{tot:<5} {sum(kls) / len(kls):>9.2e} "
                  f"{kls[int(0.99 * (len(kls) - 1))]:>9.2e} {kls[-1]:>9.2e} {dl:>9.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seqs")
    s.add_argument("--hf", required=True, help="a Hugging Face tokenizer directory")
    s.add_argument("--out", required=True); s.add_argument("--texts"); s.add_argument("-n", type=int, default=64)
    for name in ("dump-axon", "dump-llama", "env"):
        d = sub.add_parser(name)
        d.add_argument("--axon" if name != "dump-llama" else "--server", required=name == "dump-llama")
        d.add_argument("--log", help="dump-axon: parse this `axon logits` output instead of running it")
        d.add_argument("--model", required=True); d.add_argument("--seqs", required=True)
        d.add_argument("-k", type=int, default=20); d.add_argument("-n", type=int, default=64)
        d.add_argument("--out", required=True); d.add_argument("extra", nargs="*")
    c = sub.add_parser("compare"); c.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"seqs": seqs, "dump-axon": dump_axon, "dump-llama": dump_llama, "env": env_for}.get(a.cmd, lambda a: compare(a.files))(a)


if __name__ == "__main__":
    main()
