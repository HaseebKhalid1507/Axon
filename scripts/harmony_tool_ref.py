#!/usr/bin/env python3
# Regenerates tests/fixtures/tool_templates_harmony.json: the expected renders
# of gpt-oss's GGUF chat template (tokenizer.chat_template of
# gpt-oss-20b-MXFP4.gguf) with tools, for src/serve/tools/tests.rs.
# transformers' jinja setup (as scripts/tool_template_ref.py), the date fixed
# to chat.rs's HARMONY_DATE.
# Usage: scripts/harmony_tool_ref.py models/gpt-oss-20b-MXFP4.gguf  (or a .jinja file)
import json, struct, sys, jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
def raise_exception(m): raise jinja2.exceptions.TemplateError(m)
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters['tojson'] = tojson
env.globals['raise_exception'] = raise_exception
env.globals['strftime_now'] = lambda fmt: '2025-08-05'
def gguf_template(path):
    f = open(path, 'rb')
    rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
    rstr = lambda: f.read(rd('Q')).decode('utf-8')
    size = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    assert f.read(4) == b'GGUF'
    rd('I'); rd('Q'); n = rd('Q')
    for _ in range(n):
        k = rstr(); t = rd('I')
        if t == 8:
            v = rstr()
            if k == 'tokenizer.chat_template':
                return v
        elif t == 9:
            et = rd('I'); m = rd('Q')
            if et == 8:
                for _ in range(m): rstr()
            else:
                f.seek(m * struct.calcsize(size[et]), 1)
        else:
            rd(size[t])
    raise SystemExit('no tokenizer.chat_template')
src = sys.argv[1]
tpl = env.from_string(gguf_template(src) if src.endswith('.gguf') else open(src).read())
def render(messages, tools=None):
    return tpl.render(messages=messages, tools=tools, add_generation_prompt=True)
def fn(name, desc, params=None):
    f = {"name": name, "description": desc}
    if params is not None:
        f["parameters"] = params
    return {"type": "function", "function": f}

weather = fn("get_weather", "Get the current weather in a city", {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name, e.g. Paris"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
    },
    "required": ["city"]})
shell = fn("bash", "Run a shell command and return its output.", {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "The command to run"},
        "timeout": {"type": "integer", "description": "Seconds", "default": 30},
    },
    "required": ["command"]})
rich = fn("search", "Search the index.\nReturns ranked hits.", {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to look for"},
        "limit": {"type": "integer", "default": 10},
        "ratio": {"type": "number", "default": 0.5},
        "exact": {"type": "boolean", "default": False},
        "tags": {"type": "array", "items": {"type": "string"}},
        "scores": {"type": "array", "items": {"type": "number"}, "nullable": True},
        "ids": {"type": "array", "items": {"type": "integer"}},
        "flags": {"type": "array", "items": {"type": "boolean"}},
        "grid": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "modes": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
        "anything": {"type": "array"},
        "filters": {"type": "object", "description": "Narrow it down", "properties": {
            "lang": {"type": "string", "enum": ["en", "fr"]},
            "since": {"type": "string", "nullable": True},
            "deep": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
        }, "required": ["lang"]},
        "docs": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "body": {"type": "string"}, "rank": {"type": "number"}}}},
        "pair": {"type": "array", "items": {"type": "object", "properties": {"k": {"type": "string"}}}},
        "blob": {"type": "object"},
        "kind": {"type": ["string", "null"]},
        "one": {"type": ["number"]},
        "either": {"oneOf": [{"type": "string"}, {"type": "number"}]},
        "objs": {"oneOf": [{"type": "object"}, {"type": "string"}]},
        "fancy": {"oneOf": [{"type": "string", "description": "a name"}, {"type": "integer", "default": 3}], "default": "x"},
        "any_of": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        "untyped": {"description": "no type at all"},
        "level": {"type": "string", "enum": ["low", "high"], "default": "low", "description": "How hard"},
        "maybe": {"type": "string", "default": None},
        "unicode": {"type": "string", "description": "ünï \"q\" 中文"},
    },
    "required": ["query", "filters"]})
noparams = fn("now", "The time", {"type": "object", "properties": {}})
bare = fn("ping", "Ping")

def call(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}]}
def tool(content):
    return {"role": "tool", "tool_call_id": "c1", "content": content}
sysmsg = {"role": "system", "content": "You are helpful."}
user = lambda c: {"role": "user", "content": c}

import copy
def request(messages, tools):
    """The same conversation as an OpenAI client sends it: `reasoning_content`
    for the template's `thinking`, compact JSON strings for `arguments`, a
    null content beside tool calls."""
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if "thinking" in m:
            m["reasoning_content"] = m.pop("thinking")
        for c in m.get("tool_calls", []):
            c["function"]["arguments"] = json.dumps(c["function"]["arguments"], separators=(',', ':'), ensure_ascii=False)
        if m.get("tool_calls") and m["content"] == "":
            m["content"] = None
    r = {"messages": msgs}
    if tools is not None:
        r["tools"] = tools
    return r
out = {}
def case(key, messages, tools):
    out[key] = {"request": request(messages, tools), "prompt": render(messages, tools)}
case('tools_weather', [user("Weather in Paris?")], [weather])
case('tools_rich', [sysmsg, user("Find it")], [rich, noparams, bare, shell])
case('one', [sysmsg, user("Weather in Paris?"), call("get_weather", {"city": "Paris"}), tool("18C sunny")], [weather])
thinking = call("get_weather", {"city": "Paris"})
thinking["thinking"] = "Need the weather tool."
case('thinking_kept', [user("Weather in Paris?"), thinking, tool("{\"temp\": 18}")], [weather])
content_cot = call("get_weather", {"city": "Paris", "unit": "celsius"})
content_cot["content"] = "Checking."
case('content_kept', [user("Weather in Paris?"), content_cot, tool("18C")], [weather])
case('dropped_after_final', [user("Weather in Paris?"), thinking, tool("18C"),
                             {"role": "assistant", "content": "It is 18C in Paris."}, user("Thanks")], [weather])
case('notools_history', [sysmsg, user("Weather in Paris?"), call("get_weather", {"city": "Paris"}),
                         tool("18C\nsunny \"warm\"")], None)
case('two_rounds', [user("Paris then Rome"), call("get_weather", {"city": "Paris"}), tool("18C"),
                    call("get_weather", {"city": "Rome"}), tool("25C")], [weather, shell])
json.dump(out, open('tests/fixtures/tool_templates_harmony.json', 'w'), ensure_ascii=False, indent=1)
print(len(out), 'cases')
