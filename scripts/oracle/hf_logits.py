#!/usr/bin/env python3
"""FP8 checkpoints against Hugging Face: the teacher-forced accuracy gate of docs/design/fp8-weights.md §6 A.

    hf_logits.py prep  --model DIR [--texts F] [--windows 32] [--chars 900] --out W.json
    hf_logits.py axon  --axon BIN --model M --windows W.json --arm decode|batched [-k 64] --out A.json [-- flags]
    hf_logits.py score --model DIR --windows W.json --dtype f32|bf16 [--json OUT] A.json [B.json ...]

prep: windows of natural text (numerics_dnll.py's windowing), tokenized by the checkpoint's tokenizer.json (no
special tokens); every side is given these ids (axon via AXON_LOGITS_IDS), so no side tokenizes.

axon: one `axon logits` run over all windows (';'-separated sequences), top-k per position. Arms: decode
(AXON_LOGITS_PREFIX=1: every position one decode step) and batched (AXON_LOGITS_PREFIX=17 + AXON_LOGITS_EACH=1:
every position from 16 on the last row of its own batched pass of >= 17 rows, the tensor-core GEMM).

score: HF transformers on the checkpoint's weights dequantized exactly in f32 (w = e4m3 * weight_scale), as
--dtype f32, or that cast to bf16 (= compressed-tensors' decompress to bf16), one forward per window. Per
position p (the logits after token p): KL(P_hf || P_axon) on the partition {axon's top-k, the rest} (a lower bound
of the full KL; both distributions' rest masses reported), top-1 agreement (raw: near ties are counted, and the
positions where HF's top-2 margin < 0.1 nats are listed), d = nll_axon - nll_hf of token p+1. Means with SE
clustered by window (CR1, numerics_dnll.py's). Bars (pre-registered, §6 A): KL mean <= 0.01, top-1 >= 97 %;
KL mean > 0.02 is the stop line.
"""
import argparse, json, math, os, re, subprocess, sys

TIE = 0.1


def windows(text, k, chars):
    """numerics_dnll.py's: k windows of `chars` characters, evenly spaced, cut at whitespace."""
    if len(text) <= chars:
        return [text]
    step = (len(text) - chars) / max(k - 1, 1)
    out = []
    for i in range(k):
        a = int(i * step)
        w = text[a:a + chars]
        s, e = w.find(" "), w.rfind(" ")
        if 0 <= s < e:
            w = w[s + 1:e]
        out.append(w)
    return out


def cmd_prep(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    if a.prompts:
        # The 50 prompts of llama_logits.py (the diagnostic greedy picks:
        # `score --last`).
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import llama_logits
        texts = llama_logits.PROMPTS
    else:
        texts = windows(open(a.texts).read(), a.windows, a.chars)
    ws = []
    for w in texts:
        ids = tok.encode(w, add_special_tokens=False).ids
        ws.append({"text": w, "ids": ids})
    json.dump({"windows": ws}, open(a.out, "w"))
    n = [len(w["ids"]) for w in ws]
    print(f"{len(ws)} windows, {sum(n)} tokens ({min(n)}..{max(n)} each)")


def cmd_axon(a):
    ws = json.load(open(a.windows))["windows"]
    seqs = ";".join(",".join(map(str, w["ids"])) for w in ws)
    env = dict(os.environ, AXON_LOGITS_IDS=seqs, AXON_LOGITS_DIGITS="6")
    if a.arm == "decode":
        env["AXON_LOGITS_PREFIX"] = "1"
    else:
        env["AXON_LOGITS_PREFIX"] = "17"
        env["AXON_LOGITS_EACH"] = "1"
    r = subprocess.run([a.axon, "logits", a.model, "x", "-k", str(a.k)] + a.flags, capture_output=True, text=True,
                       env=env)
    if r.returncode != 0:
        sys.exit(f"{a.axon} logits exited {r.returncode}: {r.stderr[-3000:]}")
    out, cur, pos = [], None, None
    for line in r.stdout.splitlines():
        m = re.match(r"prompt ids \((\d+)\):", line)
        if m:
            cur = {}
            out.append(cur)
            continue
        m = re.match(r"position (\d+):", line)
        if m:
            pos = int(m.group(1))
            cur[pos] = []
            continue
        m = re.match(r"\s*(\d+)\s+(\S+)\s+(\S+)\s+\"", line)
        if m and pos is not None and cur is not None:
            cur[pos].append((int(m.group(1)), float(m.group(3))))
    if len(out) != len(ws):
        sys.exit(f"axon printed {len(out)} sequences for {len(ws)} windows")
    json.dump({"arm": a.arm, "flags": a.flags, "model": a.model, "k": a.k, "stderr": r.stderr[-2000:],
               "windows": [{str(p): v for p, v in w.items()} for w in out]}, open(a.out, "w"))
    print(f"{a.out}: {sum(len(w) for w in out)} positions")


def hf_model(d, dtype):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(d)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    sd = {}
    files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    for f in files:
        sd.update(load_file(os.path.join(d, f)))
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_scale"):
            continue
        if v.dtype == torch.float8_e4m3fn:
            s = sd[k + "_scale"].to(torch.float32)
            assert s.shape == (v.shape[0], 1), (k, s.shape)
            v = v.to(torch.float32) * s
        out[k] = v.to(torch.float32)
    if getattr(cfg, "tie_word_embeddings", False) and "lm_head.weight" in out:
        assert torch.equal(out["lm_head.weight"], out["model.embed_tokens.weight"]), "tied, but lm_head differs"
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32, attn_implementation="eager")
    missing, unexpected = m.load_state_dict(out, strict=False)
    missing = [k for k in missing if k != "lm_head.weight" or not cfg.tie_word_embeddings]
    assert not missing and not unexpected, (missing, unexpected)
    if dtype == "bf16":
        m = m.to(torch.bfloat16)
    return m.eval()


def clustered(d):
    n = len(d)
    if n == 0:
        return 0.0, 0.0
    mean = sum(x for _, x in d) / n
    sums = {}
    for c, x in d:
        sums[c] = sums.get(c, 0.0) + (x - mean)
    k = len(sums)
    if k < 2:
        return mean, float("nan")
    return mean, math.sqrt(sum(s * s for s in sums.values()) / (n * n) * k / (k - 1))


def cmd_score(a):
    import torch
    torch.set_num_threads(int(os.environ.get("AXON_HF_THREADS", "8")))
    ws = json.load(open(a.windows))["windows"]
    dumps = [json.load(open(f)) for f in a.dumps]
    m = hf_model(a.model, a.dtype)
    acc = [{"kl": [], "top1": [], "dnll": [], "rest_hf": [], "rest_ax": [], "ties": [], "flips": []} for _ in dumps]
    with torch.no_grad():
        for wi, w in enumerate(ws):
            ids = torch.tensor([w["ids"]])
            lp = torch.log_softmax(m(ids).logits[0].double(), dim=-1)
            for di, d in enumerate(dumps):
                r = acc[di]
                for ps, top in d["windows"][wi].items():
                    p = int(ps)
                    if a.last and p != len(w["ids"]) - 1:
                        continue
                    if not a.last and p + 1 >= len(w["ids"]):
                        continue
                    hp = lp[p]
                    ax = {t: v for t, v in top}
                    tids = torch.tensor(list(ax))
                    ph = hp[tids].exp()
                    pa = torch.tensor([ax[t] for t in ax], dtype=torch.float64).exp()
                    rh, ra = max(1.0 - ph.sum().item(), 0.0), max(1.0 - pa.sum().item(), 0.0)
                    kl = (ph * (hp[tids] - pa.log())).sum().item()
                    if rh > 0:
                        kl += rh * (math.log(rh) - math.log(max(ra, 1e-300)))
                    r["kl"].append((wi, kl))
                    h2 = torch.topk(hp, 2)
                    h1 = h2.indices[0].item()
                    a1 = top[0][0]
                    margin = (h2.values[0] - h2.values[1]).item()
                    r["top1"].append((wi, 1.0 if a1 == h1 else 0.0))
                    if margin < TIE:
                        r["ties"].append((wi, p))
                    if a1 != h1:
                        r["flips"].append({"window": wi, "pos": p, "hf": h1, "axon": a1, "hf_margin": margin})
                    nxt = w["ids"][p + 1] if p + 1 < len(w["ids"]) else None
                    if nxt in ax:
                        r["dnll"].append((wi, -ax[nxt] + hp[nxt].item()))
                    r["rest_hf"].append(rh)
                    r["rest_ax"].append(ra)
    res = []
    for f, d, r in zip(a.dumps, dumps, acc):
        kl, kse = clustered(r["kl"])
        t1, tse = clustered(r["top1"])
        dn, dse = clustered(r["dnll"])
        n = len(r["kl"])
        e = {"dump": f, "arm": d["arm"], "flags": d["flags"], "model": d["model"], "ref": f"hf-{a.dtype}",
             "positions": n, "windows": len({c for c, _ in r["kl"]}), "kl_mean": kl, "kl_se": kse,
             "kl_max": max((x for _, x in r["kl"]), default=0.0), "top1": t1, "top1_se": tse,
             "dnll": dn, "dnll_se": dse, "dnll_n": len(r["dnll"]),
             "rest_hf_mean": sum(r["rest_hf"]) / max(n, 1), "rest_ax_mean": sum(r["rest_ax"]) / max(n, 1),
             "near_ties": len(r["ties"]), "flips": len(r["flips"]),
             "flips_at_near_ties": sum(1 for x in r["flips"] if x["hf_margin"] < TIE),
             "pass": kl <= 0.01 and t1 >= 0.97, "stop": kl > 0.02, "flip_list": r["flips"][:50]}
        res.append(e)
        print(f"{os.path.basename(f):40s} {d['arm']:8s} vs hf-{a.dtype}: n {n} ({e['windows']} windows)  "
              f"KL {kl:.3e} ±{kse:.1e} (max {e['kl_max']:.3f})  top-1 {100 * t1:.2f}% ±{100 * tse:.2f}  "
              f"dNLL {dn:+.4f} ±{dse:.4f}  flips {e['flips']} ({e['flips_at_near_ties']} at HF margin < {TIE})  "
              f"rest {e['rest_hf_mean']:.1e}/{e['rest_ax_mean']:.1e}  {'PASS' if e['pass'] else 'FAIL'}"
              f"{'  STOP' if e['stop'] else ''}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--model", required=True)
    p.add_argument("--texts", default=os.path.join(here, "..", "..", "tests", "fixtures", "natural_text.txt"))
    p.add_argument("--windows", type=int, default=32)
    p.add_argument("--chars", type=int, default=900)
    p.add_argument("--prompts", action="store_true", help="llama_logits.py's 50 prompts instead of windows")
    p.add_argument("--out", required=True)
    p = sub.add_parser("axon")
    p.add_argument("--axon", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument("--arm", choices=["decode", "batched"], required=True)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("--out", required=True)
    p.add_argument("flags", nargs="*")
    p = sub.add_parser("score")
    p.add_argument("--model", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument("--dtype", choices=["f32", "bf16"], required=True)
    p.add_argument("--json")
    p.add_argument("--last", action="store_true", help="each window's last position only (--prompts)")
    p.add_argument("dumps", nargs="+")
    a = ap.parse_args()
    {"prep": cmd_prep, "axon": cmd_axon, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
