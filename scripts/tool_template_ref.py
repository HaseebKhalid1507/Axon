#!/usr/bin/env python3
# Regenerates tests/fixtures/tool_templates.json (the expected renders for
# src/serve/tools/tests.rs) with transformers' jinja setup.
# Usage: scripts/tool_template_ref.py TEMPLATE_DIR
# Reference renders of the GGUF chat templates with HF transformers' jinja setup.
import json, sys, jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
def raise_exception(m): raise jinja2.exceptions.TemplateError(m)
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters['tojson'] = tojson
env.globals['raise_exception'] = raise_exception
T = sys.argv[1] + '/'  # a dir of <model>.jinja files: each GGUF's tokenizer.chat_template
def render(name, messages, tools=None, **kw):
    t = env.from_string(open(T+name+'.jinja').read())
    return t.render(messages=messages, tools=tools, add_generation_prompt=True, bos_token='<|begin_of_text|>', **kw)

tools = [
 {"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}},
 {"type": "function", "function": {"name": "now", "description": "Time ünï \"q\"\n", "parameters": {"type": "object", "properties": {}}}},
]
one = [
 {"role": "system", "content": "You are helpful."},
 {"role": "user", "content": "Weather in Paris?"},
 {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
 {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"},
]
multi = [
 {"role": "user", "content": "Paris and Rome?"},
 {"role": "assistant", "content": "Checking both.", "tool_calls": [
   {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
   {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
 {"role": "tool", "tool_call_id": "c1", "content": "18C"},
 {"role": "tool", "tool_call_id": "c2", "content": "25C"},
 {"role": "assistant", "content": "Paris 18C, Rome 25C."},
 {"role": "user", "content": "Thanks"},
]
# Qwen3's template prints string `arguments` verbatim, so its cases use the
# strings the tests send (Qwen2.5's and Llama's tojson the value: given the
# object, as llama.cpp/vLLM pass it).
import copy
one_s = copy.deepcopy(one)
one_s[2]["tool_calls"][0]["function"]["arguments"] = '{"city":"Paris"}'
multi_s = copy.deepcopy(multi)
multi_s[1]["tool_calls"][0]["function"]["arguments"] = '{"city": "Paris"}'  # second stays an object
reasoning = [
 {"role": "user", "content": "q"},
 {"role": "assistant", "content": "ok", "reasoning_content": "plan",
  "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1, "b": [ ]}'}}]},
 {"role": "tool", "tool_call_id": "c1", "content": "r"},
]
out = {}
q3 = 'Qwen3-8B-Q4_K_M'
out[q3 + '/one'] = render(q3, one_s, tools)
out[q3 + '/multi'] = render(q3, multi_s, tools)
out[q3 + '/notools_history'] = render(q3, one_s, None)
out[q3 + '/one_nothink'] = render(q3, one_s, tools, enable_thinking=False)
out[q3 + '/reasoning'] = render(q3, reasoning, tools)
for fam in ['Qwen2.5-7B-Instruct-Q4_K_M']:
    out[fam+'/one'] = render(fam, one, tools)
    out[fam+'/multi'] = render(fam, multi, tools)
    out[fam+'/notools_history'] = render(fam, one, None)
out['llama31/one'] = render('Meta-Llama-3.1-8B-Instruct-Q4_K_M', one, tools)
json.dump(out, open('tests/fixtures/tool_templates.json', 'w'), ensure_ascii=False, indent=1)
