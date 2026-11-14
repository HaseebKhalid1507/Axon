#!/usr/bin/env python3
"""The compatibility audit's chat-prompt check at the token-id level (docs/compat-audit-2026-11-13.md).

    pip install tokenizers   (the one non-stdlib dependency)
    python3 scripts/examples/compat-audit/prompt_check.py JOB_LOG [--cache DIR]

zoo.py --chat compares *counts* (`usage.prompt_tokens` from axon serve and from llama-server --jinja): axon 3014eeb has
no way to print the ids it serves (#229 adds AXON_LOGITS_CHAT). Equal counts can hide different ids: DeepSeek-R1's
first conversation is 14 = 14 with two offsetting differences. This rebuilds Axon's ids from the code paths at 3014eeb
(src/chat.rs `Framing`, serve.rs `render()`), each piece encoded on its own with the model's Hugging Face tokenizer as
Axon encodes them, and llama.cpp's ids from its rendering of the GGUF's own template (recorded in the log), then
requires both counts to equal the counts the audit measured. If they do, the rebuilt ids are what was served, and the
two can be compared token for token. Llama 3.1/3.2's "Today Date" is set to Axon's pinned date first (as golden.py).
Mistral: the HF SentencePiece tokenizer doesn't add llama.cpp's extra U+2581 after a control token (chat.rs's module
doc), so llama.cpp's rebuilt count is 1 short per [INST]; Axon's is checked.
"""
import argparse
import json
import os
import re
import urllib.request

from tokenizers import Tokenizer

CHATS = [[("system", "You are a terse assistant."), ("user", "Name three primary colors.")],
         [("user", "Write a haiku about autumn."), ("assistant", "Leaves fall, red and gold."),
          ("user", "Now one about winter.")]]
HEAD = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"   # chat.rs LLAMA3_SYSTEM_HEAD
QWEN25 = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."  # chat.rs DEFAULT_SYSTEM
# ollama name: (HF repo with the model's tokenizer.json, Axon's format at 3014eeb, the BOS llama-server adds or None)
MODELS = {
    "llama3.1": ("unsloth/Meta-Llama-3.1-8B-Instruct", "llama3", 128000),
    "llama3.2": ("unsloth/Llama-3.2-3B-Instruct", "llama3", 128000),
    "llama3": ("NousResearch/Meta-Llama-3-8B-Instruct", "llama3", 128000),
    "qwen2.5": ("Qwen/Qwen2.5-7B-Instruct", "chatml-qwen2arch", None),
    "qwen2.5-coder": ("Qwen/Qwen2.5-Coder-7B-Instruct", "chatml-qwen2arch", None),
    "qwen2": ("Qwen/Qwen2-7B-Instruct", "chatml-qwen2arch", None),
    "qwen": ("Qwen/Qwen1.5-4B-Chat", "chatml-qwen2arch", None),
    "qwen3": ("Qwen/Qwen3-8B", "chatml", None),
    "qwen3-coder": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "chatml", None),
    "gemma3": ("unsloth/gemma-3-4b-it", "gemma", 2),
    "deepseek-r1": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "deepseek3", None),
    "mistral": ("unsloth/mistral-7b-instruct-v0.3", "mistral", 1),
}


def axon_ids(fmt, T, conv):
    sp = T.token_to_id
    enc = lambda s: T.encode(s, add_special_tokens=False).ids  # noqa: E731
    sysmsg = conv[0][1] if conv[0][0] == "system" else None
    rest = conv[1:] if sysmsg is not None else conv
    out = []
    if fmt == "llama3":   # Framing::system: always the block with the date lines, BOS first
        out += [sp("<|begin_of_text|>"), sp("<|start_header_id|>")] + enc("system") + [sp("<|end_header_id|>")]
        out += enc("\n\n" + HEAD + (sysmsg or "").strip()) + [sp("<|eot_id|>")]
        for role, text in rest:
            out += [sp("<|start_header_id|>")] + enc(role) + [sp("<|end_header_id|>")] + enc("\n\n" + text.strip())
            out += [sp("<|eot_id|>")]
        out += [sp("<|start_header_id|>")] + enc("assistant") + [sp("<|end_header_id|>")] + enc("\n\n")
    elif fmt in ("chatml", "chatml-qwen2arch"):   # serve.rs: default_system = Qwen2.5's for arch qwen2
        s = sysmsg if sysmsg is not None else (QWEN25 if fmt == "chatml-qwen2arch" else None)
        for role, text in ([("system", s)] if s is not None else []) + rest:
            out += [sp("<|im_start|>")] + enc(f"{role}\n{text}") + [sp("<|im_end|>")] + enc("\n")
        out += [sp("<|im_start|>")] + enc("assistant\n")
    elif fmt == "gemma":   # the system message in the first user turn, messages trimmed, role "model"
        out += [sp("<bos>")]
        for i, (role, text) in enumerate(rest):
            t = f"{sysmsg}\n\n{text.strip()}" if (i == 0 and sysmsg is not None) else text.strip()
            out += [sp("<start_of_turn>")] + enc(f"{'model' if role == 'assistant' else role}\n{t}")
            out += [sp("<end_of_turn>")] + enc("\n")
        out += [sp("<start_of_turn>")] + enc("model\n")
    elif fmt == "deepseek3":   # BOS only if add_bos_token (false in this GGUF); replies framed without an opening
        out += enc(sysmsg) if sysmsg else []
        for role, text in rest:
            out += enc(text) + [sp("<｜end▁of▁sentence｜>")] if role == "assistant" else [sp("<｜User｜>")] + enc(text)
        out += [sp("<｜Assistant｜>"), sp("</think>")]
    elif fmt == "mistral":   # mistral_common's tokens, the system message in the first user turn after \n\n
        out += [sp("<s>")]
        for i, (role, text) in enumerate(rest):
            if role == "assistant":
                out += enc(text.strip()) + [sp("</s>")]
            else:
                t = f"{sysmsg}\n\n{text}" if (i == 0 and sysmsg) else text
                out += [sp("[INST]")] + enc(t) + [sp("[/INST]")]
    return out


def results(path):
    out = {}
    for ln in open(path, errors="replace"):
        if ln.startswith("ZOO-RESULT "):
            r = json.loads(ln[11:])
            out[r["ollama"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/axon-eval/tokenizers"))
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    R = results(a.log)
    bad = 0
    for name, (repo, fmt, bos) in MODELS.items():
        r = R.get(name)
        if not r or not r.get("chat_rows"):
            print(f"{name}: no chat rows in the log")
            continue
        p = os.path.join(a.cache, repo.replace("/", "__") + ".json")
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/tokenizer.json", p)
        T = Tokenizer.from_file(p)
        for i, (conv, row) in enumerate(zip(CHATS, r["chat_rows"])):
            ax = axon_ids(fmt, T, conv)
            tmpl = re.sub(r"Today Date: \d{1,2} [A-Z][a-z]{2} \d{4}", "Today Date: 26 Jul 2024", row["llama_template"])
            lead = T.decode(T.encode(tmpl, add_special_tokens=False).ids[:1], skip_special_tokens=False)
            ll = ([bos] if bos is not None and lead not in ("<|begin_of_text|>", "<bos>", "<s>") else [])
            ll += T.encode(tmpl, add_special_tokens=False).ids
            ok = len(ax) == row["axon_tokens"]
            bad += not ok
            print(f"{name:14s} conv{i + 1}: axon rebuilt {len(ax)} (measured {row['axon_tokens']}) "
                  f"{'ok' if ok else 'MISMATCH'} | llama.cpp {len(ll)} (measured {row['llama_tokens']}) | "
                  f"ids {'EQUAL' if ax == ll else 'DIFFER'}")
            if ax != ll:
                print("    axon     :", repr(T.decode(ax, skip_special_tokens=False)))
                print("    llama.cpp:", repr(T.decode(ll, skip_special_tokens=False)))
    print("REBUILD", "OK" if not bad else f"{bad} MISMATCH(ES)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
