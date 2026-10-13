#!/usr/bin/env python3
"""Regenerate the reference data in tests/fixtures/ that axon's tests check against.

Three independent oracles, so a bug in axon can't hide by agreeing with itself:

  golden/   gguf-py's dequantizer: raw quantized row bytes (.raw) + expected f32 values (.f32)
  tokens.json   Hugging Face `tokenizers` with Qwen2.5's tokenizer.json: text -> token ids
  gen/      Ollama (llama.cpp) on CPU, greedy: the token it picks at each step and its logprob

Run from the repo root:

    python3 -m venv /tmp/axon-venv
    /tmp/axon-venv/bin/pip install gguf numpy tokenizers huggingface_hub requests
    /tmp/axon-venv/bin/python scripts/oracle/make_fixtures.py

Needs models/qwen2.5-0.5b-q4_k_m.gguf, network for the tokenizer download (or
--tokenizer-json PATH), and a running `ollama serve` for the gen fixtures.
Pass --skip-gen to leave gen/ alone.
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "qwen2.5-0.5b-q4_k_m.gguf"
FIX = ROOT / "tests" / "fixtures"
OLLAMA = "http://localhost:11434"
OLLAMA_MODEL = "axon-oracle"

TOKEN_CASES = [
    "Hello, world!",
    "The capital of France is",
    "  leading spaces\n\nnewlines",
    "naïve café 東京 🚀",
    "<|im_start|>user\nhi<|im_end|>",
    'fn main() {\n    println!("{}", 1234567);\n}',
    "I'm   don't  you'll",
    "Ünïcödé\t\ttabs   \n  ",
]

# (fixture name, prompt, tokens). The test feeds Ollama's token back in at every step
# and checks axon's logprob for it. Steps where Ollama's top two choices are within
# NEAR_TIE of each other are marked, and the test doesn't require axon to pick the
# same token there, because quantization noise can legitimately flip a near-tie.
GEN_CASES = [
    ("paris", "The capital of France is", 16),
    ("arith", "Question: What is 12 + 30?\nAnswer:", 12),
    ("rust", "// Return the nth Fibonacci number.\nfn fib(n: u64) -> u64 {", 24),
]
NEAR_TIE = 0.25


def golden():
    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize

    out = FIX / "golden"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        f.unlink()
    r = GGUFReader(str(MODEL))
    # One tensor per quant type in this model, first and last rows.
    by_type = {}
    for t in r.tensors:
        by_type.setdefault(t.tensor_type.name, t)
    # ffn_down alternates Q4_K / Q6_K between layers; make sure both are covered.
    for t in r.tensors:
        if t.name.endswith("ffn_down.weight"):
            by_type.setdefault(t.tensor_type.name, t)
    n = 0
    for qname, t in sorted(by_type.items()):
        ne0 = int(t.shape[0])
        raw = np.asarray(t.data)
        n_rows = int(np.prod(t.shape)) // ne0
        rows_raw = raw.reshape(n_rows, -1)
        full = dequantize(raw, t.tensor_type).reshape(n_rows, ne0)
        for row in sorted({0, n_rows - 1}):
            stem = f"{qname}_{t.name.replace('.', '_')}_r{row}"
            (out / f"{stem}.raw").write_bytes(rows_raw[row].tobytes())
            (out / f"{stem}.f32").write_bytes(full[row].astype("<f4").tobytes())
            meta = {"tensor": t.name, "qtype": int(t.tensor_type), "qtype_name": qname, "row": row, "row_len": ne0}
            (out / f"{stem}.json").write_text(json.dumps(meta) + "\n")
            n += 1
    print(f"golden: {n} rows covering {', '.join(sorted(by_type))}")


def load_tokenizer(tokenizer_json):
    from tokenizers import Tokenizer

    if tokenizer_json is None:
        from huggingface_hub import hf_hub_download

        tokenizer_json = hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "tokenizer.json")
    return Tokenizer.from_file(str(tokenizer_json))


def byte_level(bs):
    """GPT-2 byte-level BPE alphabet: raw bytes -> the vocab string for them."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    table, extra = {}, 0
    for b in range(256):
        if b in keep:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + extra)
            extra += 1
    return "".join(table[b] for b in bs)


def tokens(tk):
    cases = [{"text": s, "ids": tk.encode(s, add_special_tokens=False).ids} for s in TOKEN_CASES]
    (FIX / "tokens.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")
    print(f"tokens: {len(cases)} cases")


def gen(tk):
    import requests

    have = requests.get(f"{OLLAMA}/api/tags", timeout=10).json()
    if not any(m["name"].split(":")[0] == OLLAMA_MODEL for m in have.get("models", [])):
        modelfile = FIX / "Modelfile.tmp"
        modelfile.write_text(
            f'FROM {MODEL}\nTEMPLATE """{{{{ .Prompt }}}}"""\nPARAMETER temperature 0\nPARAMETER num_ctx 2048\n'
        )
        subprocess.run(["ollama", "create", OLLAMA_MODEL, "-f", str(modelfile)], check=True)
        modelfile.unlink()
    out = FIX / "gen"
    out.mkdir(parents=True, exist_ok=True)
    for name, prompt, n in GEN_CASES:
        resp = requests.post(
            f"{OLLAMA}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "logprobs": True,
                "top_logprobs": 2,
                # CPU only: the GPU kernels round differently and drift off the CPU path.
                "options": {"temperature": 0, "num_predict": n, "num_gpu": 0, "seed": 0},
            },
            timeout=300,
        ).json()
        steps = []
        for lp in resp["logprobs"]:
            top = lp["top_logprobs"]
            tid = tk.token_to_id(byte_level(lp["bytes"]))
            if tid is None:
                sys.exit(f"gen/{name}: no single token id for {lp['token']!r}")
            near_tie = len(top) > 1 and top[0]["logprob"] - top[1]["logprob"] < NEAR_TIE
            steps.append({"token": lp["token"], "id": tid, "logprob": lp["logprob"], "near_tie": near_tie})
        fx = {"prompt": prompt, "oracle": f"ollama {requests.get(f'{OLLAMA}/api/version').json()['version']} cpu", "steps": steps}
        (out / f"{name}.json").write_text(json.dumps(fx, ensure_ascii=False, indent=1) + "\n")
        text = "".join(s["token"] for s in steps)
        ties = sum(s["near_tie"] for s in steps)
        print(f"gen/{name}: {len(steps)} steps, {ties} near-ties: {prompt!r} -> {text!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer-json", help="local Qwen2.5 tokenizer.json instead of downloading it")
    ap.add_argument("--skip-gen", action="store_true", help="don't touch gen/ (no Ollama needed)")
    a = ap.parse_args()
    if not MODEL.exists():
        sys.exit(f"missing {MODEL}")
    FIX.mkdir(parents=True, exist_ok=True)
    golden()
    tk = load_tokenizer(a.tokenizer_json)
    tokens(tk)
    if not a.skip_gen:
        gen(tk)


if __name__ == "__main__":
    main()
