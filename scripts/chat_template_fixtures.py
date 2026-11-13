#!/usr/bin/env python3
"""Regenerates tests/fixtures/chat_templates.json: each chat family's prompt ids from Hugging Face, for
src/serve/template_tests.rs (token-exact: Axon's ids == HF's for the same messages). Network once; the test needs none.

    python3 -m venv V && V/bin/pip install transformers==5.22.2 tokenizers==0.23.3 jinja2 huggingface_hub
    V/bin/python scripts/chat_template_fixtures.py [--only NAME] [--out tests/fixtures/chat_templates.json]

Every file comes from a pinned Hugging Face commit (REV: the GGUF repo's and the tokenizer repo's), recorded in the
JSON: templates have changed in place before (Qwen2.5's on 2024-09-19, DeepSeek R1's in Feb 2025). To move a family
to a newer revision, change REV and regenerate it (--only NAME); the test then shows what changed.

Per family (FAMILIES): the GGUF's header is range-read from Hugging Face (its embedded tokenizer.chat_template and its
vocabulary: Axon follows the file, so the file's template is the reference, not the HF repo's). Each case is rendered
by transformers' `apply_chat_template(messages, chat_template=<the GGUF's>, add_generation_prompt=True,
tokenize=False, tools=...)` and tokenized by the model's HF tokenizer.json (`tokenizers`, add_special_tokens=False:
specials in the text parsed, as apply_chat_template(tokenize=True) does). Axon's pinned dates are passed or
substituted (Llama 3.x `date_string`; gpt-oss `strftime_now`), as golden.py does.

The fixture carries a sparse vocabulary, not the GGUF: every named special and byte token (not padding, not
`<unused…>`/reserved controls), and every token (merge, for BPE)
whose text is a substring of a piece of the rendered prompts (the text between special tokens). Any merge
candidate while encoding a piece is a substring of it, so for those pieces the sparse vocabulary has exactly the
full one's candidates: Axon's tokenizer gives the same ids with either. Other ids get unmatchable placeholders in
the test. The test checks that first (`encode_special(text) == ids`), then the framing.
"""
import argparse
import copy
import json
import re
import struct
import sys
import urllib.request

HARMONY_DATE = "2025-08-05"          # src/chat.rs HARMONY_DATE
LLAMA_DATE = "26 Jul 2024"           # src/chat.rs LLAMA3_SYSTEM_HEAD
HF = "https://huggingface.co/"

# name: (GGUF repo/file, HF repo with tokenizer.json, case groups). Every family also gets BASE's groups.
FAMILIES = {
    "qwen2.5": ("bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf", "Qwen/Qwen2.5-7B-Instruct", ["tools_obj"]),
    "qwen2.5-coder": ("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", "Qwen/Qwen2.5-Coder-7B-Instruct", []),
    "qwen2": ("Qwen/Qwen2-7B-Instruct-GGUF/qwen2-7b-instruct-q4_k_m.gguf", "Qwen/Qwen2-7B-Instruct", []),
    "qwen1.5": ("Qwen/Qwen1.5-4B-Chat-GGUF/qwen1_5-4b-chat-q4_k_m.gguf", "Qwen/Qwen1.5-4B-Chat", []),
    "qwen3": ("Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", "Qwen/Qwen3-8B", ["think", "think2", "nothink", "tools_str"]),
    "qwen3-coder": ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", "Qwen/Qwen3-Coder-30B-A3B-Instruct", []),
    "llama3": ("bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", "NousResearch/Meta-Llama-3-8B-Instruct", []),
    "llama3.1": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "unsloth/Meta-Llama-3.1-8B-Instruct", ["tools_obj1", "tools_nosys"]),
    "llama3.2": ("bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf", "unsloth/Llama-3.2-3B-Instruct", ["tools_obj1", "tools_nosys"]),
    "gemma3": ("bartowski/google_gemma-3-4b-it-GGUF/google_gemma-3-4b-it-Q4_K_M.gguf", "unsloth/gemma-3-4b-it", []),
    # Not Mistral v0.3: Axon frames it with mistral_common's tokens by design (src/chat.rs module doc), pinned by
    # chat.rs's mistral tests and tests/fixtures/tokens_mistral.json; its GGUF's older template disagrees with
    # mistral_common (` [/INST]`, no system role).
    "gpt-oss": ("ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf", "openai/gpt-oss-20b", []),
    "deepseek-r1-0528-qwen3": ("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", ["think", "think2"]),
    # V3.1 (non-thinking default): the framing Axon falls back to for this vocabulary without a template.
    "deepseek-v3.1": ("unsloth/DeepSeek-V3.1-GGUF/Q4_K_M/DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf", "deepseek-ai/DeepSeek-V3.1", ["think", "think2"]),
    "deepseek-r1-distill-qwen": ("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", ["think", "think2"]),
    # Llama 3's vocabulary with DeepSeek's markers: DeepSeek is picked before Llama 3; its BOS is renamed.
    "deepseek-r1-distill-llama": ("bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", ["think", "think2"]),
    # V3 (0324): R1's framing without reasoning (no </think> in its template).
    "deepseek-v3": ("unsloth/DeepSeek-V3-0324-GGUF/Q4_K_M/DeepSeek-V3-0324-Q4_K_M-00001-of-00009.gguf", "deepseek-ai/DeepSeek-V3-0324", []),
}
BASE = ["chat", "ws", "empty_sys"]
# The Hugging Face commits each family's files come from: (GGUF repo, tokenizer repo). 11-13.
REV = {
    "qwen2.5": ("8911e8a47f92bac19d6f5c64a2e2095bd2f7d031", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "qwen2.5-coder": ("1f629da0c8bed16b9e50cee91c70693650e66c35", "c03e6d358207e414f1eca0bb1891e29f1db0e242"),
    "qwen2": ("c3024c6fff0a02d52119ecee024bbb93d4b4b8e4", "f2826a00ceef68f0f2b946d945ecc0477ce4450c"),
    "qwen1.5": ("ff45e22352f7ff8d2be76ba26497fbf4ad0343e4", "a7a4d4945d28bac955554c9abd2f74a71ebbf22f"),
    "qwen3": ("7c41481f57cb95916b40956ab2f0b139b296d974", "b968826d9c46dd6066d109eabc6255188de91218"),
    "qwen3-coder": ("b17cb02dd882d5b6ab62fc777ad2995f19668350", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"),
    "llama3": ("4ebc4aa83d60a5d6f9e1e1e9272a4d6306d770c1", "53346005fb0ef11d3b6a83b12c895cca40156b6c"),
    "llama3.1": ("bf5b95e96dac0462e2a09145ec66cae9a3f12067", "a2856192dd7c25b842431f39c179a6c2c2f627d1"),
    "llama3.2": ("5ab33fa94d1d04e903623ae72c95d1696f09f9e8", "006f5dcd1393c3add266de40994ba96225e9689d"),
    "gemma3": ("71506238f970075ca85125cd749c28b1b0eee84e", "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"),
    "gpt-oss": ("41e8a94d15d62070b36cf4553a0250717ecf46b8", "6cee5e81ee83917806bbde320786a8fb61efebee"),
    "deepseek-r1-0528-qwen3": ("eb48357c179d34dbf515983f798dfb8752a0f261", "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"),
    "deepseek-v3.1": ("feb73effeee56ec0e5733a38c825d8fec1d76314", "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db"),
    "deepseek-r1-distill-qwen": ("361004151d4f4f6b446dc5e6d46fbf4422a80d5f", "916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "deepseek-r1-distill-llama": ("935d400c2ed1a29cc2a1045df25ec7998fb4dfe4", "6a6f4aa4197940add57724a7707d069478df56b1"),
    "deepseek-v3": ("522a005de381512eeb3e22aff4232335431f7303", "e9b33add76883f293d6bf61f6bd89b497e80e335"),
}

SYS = "You are a terse assistant."
CHAT = {
    "one": [{"role": "user", "content": "Name three primary colors."}],
    "one_sys": [{"role": "system", "content": SYS}, {"role": "user", "content": "Name three primary colors."}],
    "multi": [{"role": "user", "content": "Write a haiku about autumn."},
              {"role": "assistant", "content": "Leaves fall, red and gold.\nCrisp air, café warmth."},
              {"role": "user", "content": "Now one about winter."}],
    "multi_sys": [{"role": "system", "content": SYS},
                  {"role": "user", "content": "Write a haiku about autumn."},
                  {"role": "assistant", "content": "Leaves fall, red and gold.\nCrisp air, café warmth."},
                  {"role": "user", "content": "Now one about winter."}],
}
# An earlier reply with its reasoning (thinking models): templates keep what follows </think>.
THINK = {"multi_think": [{"role": "user", "content": "What is 2+2?"},
                         {"role": "assistant", "content": "<think>\nSimple sum.\n</think>\n\n4."},
                         {"role": "user", "content": "And 3+3?"}]}
# Two </think> in one reply: Qwen3's and R1's templates keep what follows the last, V3.1's the first (the second
# stays in the text: `specials_in_text`, HF's tokenizer parses it, Axon doesn't).
THINK2 = {"multi_think2": [{"role": "user", "content": "What is 2+2?"},
                           {"role": "assistant", "content": "<think>\nFirst.\n</think>\nSecond.\n</think>\n\n4."},
                           {"role": "user", "content": "And 3+3?"}]}
# Whitespace around every message (Llama and Gemma trim it, ChatML and DeepSeek keep it).
WS = {"ws": [{"role": "system", "content": "  You are terse.\n"},
             {"role": "user", "content": "  Write a haiku about autumn. \n"},
             {"role": "assistant", "content": "\n Leaves fall, red and gold. \n"},
             {"role": "user", "content": " Now one about winter.\n\n"}],
      "ws_nosys": [{"role": "user", "content": "\n\nName three primary colors.  "}]}
# An empty system message: it is given, so no default replaces it.
EMPTY = {"empty_sys": [{"role": "system", "content": ""}, {"role": "user", "content": "Name three primary colors."}],
         "empty_sys_multi": [{"role": "system", "content": ""}] + CHAT["multi"]}
# Qwen3 with thinking off (`enable_thinking=False`; Axon's no_think).
NOTHINK = {"one_nothink": CHAT["one"], "multi_nothink": CHAT["multi"]}
# Tool turns: tool_template_ref.py's conversations (client bodies send arguments as a JSON string; Qwen2.5's and
# Llama's templates tojson the object, as llama.cpp/vLLM pass it; Qwen3's prints a string verbatim).
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "now", "description": "Time ünï \"q\"\n", "parameters": {"type": "object", "properties": {}}}},
]
T_ONE = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"},
]
# No system message: Llama 3.x's header with tools then has no system text (serve-md5's tools shape, 11-13).
T_NOSYS = T_ONE[1:]
T_MULTI = [
    {"role": "user", "content": "Paris and Rome?"},
    {"role": "assistant", "content": "Checking both.", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "18C"},
    {"role": "tool", "tool_call_id": "c2", "content": "25C"},
    {"role": "assistant", "content": "Paris 18C, Rome 25C."},
    {"role": "user", "content": "Thanks"},
]


def body_of(msgs, tools):
    """The request a client sends: assistant content null with calls, arguments a JSON string."""
    out = []
    for m in copy.deepcopy(msgs):
        if m.get("tool_calls"):
            if not m["content"]:
                m["content"] = None
            for c in m["tool_calls"]:
                a = c["function"]["arguments"]
                if not isinstance(a, str):
                    c["function"]["arguments"] = json.dumps(a, separators=(",", ":"))
        out.append(m)
    return {"tools": tools, "messages": out}


def stringify(msgs):
    """Qwen3's cases: the arguments as the client's string (its template prints a string verbatim)."""
    m = copy.deepcopy(msgs)
    for x in m:
        for c in x.get("tool_calls") or []:
            c["function"]["arguments"] = json.dumps(c["function"]["arguments"], separators=(",", ":"))
    return m


def cases_for(groups):
    out = []
    for g in BASE + groups:
        if g in ("chat", "nosys"):
            for k, v in CHAT.items():
                if g == "nosys" and k.endswith("_sys"):
                    continue
                out.append((k, v, None))
        elif g == "think":
            out += [(k, v, None) for k, v in THINK.items()]
        elif g == "think2":
            out += [(k, v, None) for k, v in THINK2.items()]
        elif g == "ws":
            out += [(k, v, None) for k, v in WS.items()]
        elif g == "empty_sys":
            out += [(k, v, None) for k, v in EMPTY.items()]
        elif g == "nothink":
            out += [(k, v, None) for k, v in NOTHINK.items()]
        elif g == "tools_obj":
            out += [("tools_one", T_ONE, TOOLS), ("tools_multi", T_MULTI, TOOLS)]
        elif g == "tools_obj1":
            out += [("tools_one", T_ONE, TOOLS)]
        elif g == "tools_nosys":
            out += [("tools_nosys", T_NOSYS, TOOLS)]
        elif g == "tools_str":
            out += [("tools_one", stringify(T_ONE), TOOLS), ("tools_multi", stringify(T_MULTI), TOOLS)]
    return out


# ------------------------------------------------------------------ the GGUF header (KV section, arrays included)

def fetch_header(path, rev):
    url = HF + "/".join(path.split("/")[:2]) + f"/resolve/{rev}/" + "/".join(path.split("/")[2:])
    for n in (16 << 20, 48 << 20, 128 << 20):
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"})
        buf = urllib.request.urlopen(req, timeout=300).read()
        try:
            return parse_kv(buf)
        except (struct.error, IndexError):
            continue
    raise SystemExit(f"{path}: header larger than 128 MB?")


def parse_kv(buf):
    o = [8]
    F = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rd(f):
        v = struct.unpack_from("<" + f, buf, o[0])
        o[0] += struct.calcsize("<" + f)
        return v[0]

    def s():
        n = rd("Q")
        if o[0] + n > len(buf):
            raise IndexError
        v = buf[o[0]:o[0] + n].decode("utf-8", "surrogateescape")
        o[0] += n
        return v

    def val(t):
        if t == 8:
            return s()
        if t == 9:
            et, n = rd("I"), rd("Q")
            if et == 8:
                return [s() for _ in range(n)]
            sz = struct.calcsize("<" + F[et])
            if o[0] + n * sz > len(buf):
                raise IndexError
            v = list(struct.unpack_from("<%d%s" % (n, F[et]), buf, o[0]))
            o[0] += n * sz
            return v
        return rd(F[t])
    assert buf[:4] == b"GGUF"
    rd("Q")
    kv = {}
    for _ in range(rd("Q")):
        k = s()
        kv[k] = val(rd("I"))
    return kv


# ------------------------------------------------------------------ the sparse vocabulary

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def substrings(s, cap=64):
    out = set()
    for i in range(len(s)):
        for j in range(i + 1, min(len(s), i + cap) + 1):
            out.add(s[i:j])
    return out


RESERVED = re.compile(r"^(<unused\d+>|<\|reserved_special_token_\d+\|>|<\|reserved_\d+\|>|\[control_\d+\])$")


def sparse_vocab(kv, texts):
    toks = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(toks)
    scores = kv.get("tokenizer.ggml.scores")
    model = kv["tokenizer.ggml.model"]
    specials = {t for t, ty in zip(toks, types) if ty in (3, 4)}
    # The pieces: each rendered text cut at its special tokens (longest first, as tokenizers matches them).
    alt = "|".join(re.escape(t) for t in sorted(specials, key=len, reverse=True) if t)
    pieces = set()
    for text in texts:
        pieces.update(p for p in (re.split(alt, text) if alt else [text]) if p)
    if model == "gpt2":
        b2u = bytes_to_unicode()
        conv = {"".join(b2u[b] for b in p.encode("utf-8")) for p in pieces}
        base = set(b2u.values())
    else:   # SentencePiece: spaces as U+2581, with and without the dummy prefix
        conv = set()
        for p in pieces:
            q = p.replace(" ", "\u2581")
            conv |= {q, "\u2581" + q}
        base = set()
    subs = set()
    for p in conv:
        subs |= substrings(p)
    keep = {}
    for i, (t, ty) in enumerate(zip(toks, types)):
        if ty == 5 or (ty == 3 and RESERVED.match(t)):
            continue    # padding and reserved/unused controls: never in these texts, never looked up by name
        if ty != 1 or t in subs or t in base:
            keep[i] = t
    vocab = [[i, keep[i], types[i]] + ([scores[i]] if scores else []) for i in sorted(keep)]
    merges = []
    for m in kv.get("tokenizer.ggml.merges") or []:
        a, b = m.split(" ", 1)
        if a + b in subs:
            merges.append(m)
    return {"n": len(toks), "tokens": vocab, "merges": merges}


META_KEYS = ("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
             "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id", "tokenizer.ggml.padding_token_id",
             "tokenizer.ggml.unknown_token_id", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
             "tokenizer.ggml.add_space_prefix", "tokenizer.chat_template")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append")
    ap.add_argument("--out", default="tests/fixtures/chat_templates.json")
    a = ap.parse_args()
    import tokenizers
    import transformers
    from huggingface_hub import hf_hub_download
    try:
        old = json.load(open(a.out))
    except (OSError, ValueError):
        old = {"families": []}
    fams = {f["name"]: f for f in old.get("families", [])}
    for name, (gguf, repo, groups) in FAMILIES.items():
        if a.only and name not in a.only:
            continue
        grev, hrev = REV[name]
        print(f"== {name}: {gguf} @ {grev[:10]}", file=sys.stderr)
        kv = fetch_header(gguf, grev)
        tmpl = kv["tokenizer.chat_template"]
        hft = transformers.AutoTokenizer.from_pretrained(repo, revision=hrev)
        T = tokenizers.Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=hrev))
        cases, texts = [], []
        for cname, msgs, tools in cases_for(groups):
            kw = {"date_string": LLAMA_DATE}
            if cname.endswith("_nothink"):
                kw["enable_thinking"] = False
            text = hft.apply_chat_template(msgs, chat_template=tmpl, tools=tools, add_generation_prompt=True,
                                           tokenize=False, **kw)
            text = re.sub(r"Current date: \d{4}-\d{2}-\d{2}", "Current date: " + HARMONY_DATE, text)
            ids = T.encode(text, add_special_tokens=False).ids
            c = {"name": cname, "text": text, "ids": ids}
            if cname.endswith("_nothink"):
                c["no_think"] = True
            if cname == "multi_think2" and "[1:] | join('</think>')" in tmpl:
                c["specials_in_text"] = True
            if tools:   # the request's bytes, as a client sends them (key order kept: tools are dumped as given)
                c["body"] = json.dumps(body_of(msgs, tools), ensure_ascii=False)
            else:
                c["messages"] = msgs
            cases.append(c)
            texts.append(text)
        fams[name] = {
            "name": name, "gguf": gguf, "hf_tokenizer": repo, "revisions": {"gguf": grev, "hf_tokenizer": hrev},
            "arch": kv["general.architecture"],
            "meta": {k: kv[k] for k in META_KEYS if k in kv},
            "vocab": sparse_vocab(kv, texts), "cases": cases,
        }
        v = fams[name]["vocab"]
        print(f"   {len(cases)} cases, sparse vocab {len(v['tokens'])} of {v['n']} tokens, {len(v['merges'])} merges",
              file=sys.stderr)
    out = {"source": (f"scripts/chat_template_fixtures.py: transformers {transformers.__version__} apply_chat_template "
                      f"(the GGUF's own template, add_generation_prompt), tokenizers {tokenizers.__version__} "
                      "(the HF repo's tokenizer.json, add_special_tokens=False)"),
           "families": [fams[k] for k in FAMILIES if k in fams]}
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
