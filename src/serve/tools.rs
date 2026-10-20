//! Tool calling for `axon serve` (OpenAI `tools` / `tool_calls`).
//!
//! **Prompt.** A request with tools, or with tool turns in its history
//! (`role: "tool"`, assistant `tool_calls`), is rendered as the model's own
//! chat template renders it (the GGUF's `tokenizer.chat_template`, with
//! transformers' `tojson`):
//!
//! - Qwen3 and Qwen2.5 (ChatML): the Hermes format. A `# Tools` block with
//!   the tools' JSON inside `<tools></tools>` is appended to the system
//!   message (Qwen2.5 brings its default system message). Calls are
//!   `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`, and a run
//!   of results is one user turn of `<tool_response>` blocks.
//! - Llama 3.1/3.2: tools go into the first user message (the template's
//!   default `tools_in_user_message`), with `Environment: ipython` in the
//!   system block. A call is a bare `{"name": ..., "parameters": ...}` reply,
//!   and a result is an `ipython` turn whose content is JSON-encoded, as the
//!   template's `tojson` does with a string. The template allows one call
//!   per assistant message.
//! - gpt-oss (harmony): the tools as TypeScript (`# Tools` / `## functions`
//!   / `namespace functions { ... }`, the template's `render_typescript_type`
//!   with its whitespace) in the developer message, and a line about them
//!   in the system message. A call is `<|start|>assistant to=functions.NAME
//!   <|channel|>commentary json<|message|>{args}<|call|>`, a result
//!   `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>`
//!   `{content as a JSON string}<|end|>`; the template's rules on which
//!   reasoning stays are in `harmony`.
//!
//! The template's own markers (`<|im_start|>`, `<tool_call>`, ...) become
//! their special tokens. Message text, tool schemas and arguments are
//! encoded as plain text, never scanned for markers, so text can't forge a
//! turn (as everywhere else in axon). Tool JSON keeps the request's key
//! order, using an order-preserving parser here because serde_json's maps
//! sort keys. History `arguments` follow each template: Qwen3's prints a
//! string verbatim (`{%- if tool_call.arguments is string %}`). Qwen2.5's
//! and Llama 3.1's apply `tojson` to the value: a string that is valid JSON
//! is rendered as that value re-serialized (as llama.cpp and vLLM pass the
//! object to these templates), and any other string as-is.
//!
//! A request with neither tools nor tool turns never reaches this module:
//! its prompt is the one main renders, byte for byte.
//!
//! **Output.** `Parser` turns the reply's text into content and calls,
//! incrementally for streaming:
//!
//! - Hermes: text outside `<tool_call>...</tool_call>` is content. Anything
//!   that may be the start of `<tool_call>` is held back until decided, so
//!   no part of a block leaks into `content`. Whitespace that only separates
//!   content from calls is dropped.
//! - Llama: a reply whose first non-blank character is `{` is held whole and
//!   decided at the end. Any other reply streams as content.
//! - Harmony: the engine splits the reply by channel (`chat::Channels`, told
//!   to look for calls) on the special tokens: analysis goes to reasoning,
//!   final to content, and an assistant message addressed to `functions.X`
//!   (in its role section or its channel header) is a call. `TextOut` hands
//!   the parser the content escaped and each call framed out of band
//!   (`harmony_text`), so no text the model writes, the template's call
//!   syntax included, can become a call. `<|call|>` ends the reply: one call
//!   per reply. Arguments that aren't JSON come back as their text, content.
//!
//! **Invalid JSON policy.** A block is a call only if it parses as a JSON
//! object with a string `"name"`. Otherwise the whole block, tags included,
//! is returned as content, verbatim. So `tool_calls[].function.arguments`
//! is always valid JSON (an object, or whatever JSON value the model gave;
//! a string value is used as the arguments if it is itself JSON, else it is
//! returned as its JSON string literal).
//!
//! Known false positives (the parser does not understand prose or JSON
//! strings): `<tool_call>` written in prose opens a block (which then turns
//! back into content if it isn't a call), and a `</tool_call>` inside a JSON
//! string value closes the block early (the call then fails to parse and is
//! returned as content). Llama: a `;`-separated list is tried only when the
//! whole reply isn't one call.
//! A block left open when generation ends is a call if its text up to then
//! parses (the model stopped right after the JSON), else content.
//!
//! **tool_choice.** `auto` (the default) renders the tools. `none` doesn't
//! render them (tool history still is), and the output is plain content.
//! `required` and a named function prefill the reply with the opening of a
//! call: `<tool_call>\n{"name": "` (named: through `"arguments": `). The model
//! then has to call, and for Qwen3 the reply skips reasoning (the empty
//! `<think>` block comes first). A named choice renders only that tool.
//! gpt-oss: the reply opens as `<|start|>assistant to=functions` (required:
//! the model writes `.NAME`, the pretoken split it learned) or `... to=
//! functions.NAME<|channel|>commentary json<|message|>` (named), skipping
//! the analysis; a named choice still renders every tool, so the prompt's
//! prefix stays the same as with auto.

use super::Template;
use super::api::{ApiError, Usage, quote};
use crate::chat::{self, Format};
use crate::tokenizer::Tokenizer;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Order-preserving JSON, printed as Python's json.dumps(ensure_ascii=False).

/// A JSON value that keeps its objects' key order and its numbers' text.
#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, k: &str) -> Option<&J> {
        match self {
            // The last of duplicate keys wins, as in serde_json.
            J::Obj(m) => m.iter().rev().find(|(key, _)| key == k).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }

    fn is_null(&self) -> bool {
        matches!(self, J::Null)
    }

    /// `json.dumps(self, ensure_ascii=False)`: `", "` and `": "` separators.
    pub fn dumps(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, None, 0);
        s
    }

    /// `json.dumps(self, ensure_ascii=False, indent=n)`.
    pub fn dumps_indent(&self, n: usize) -> String {
        let mut s = String::new();
        self.write(&mut s, Some(n), 0);
        s
    }

    fn write(&self, s: &mut String, indent: Option<usize>, depth: usize) {
        let nl = |s: &mut String, d: usize| {
            if let Some(n) = indent {
                s.push('\n');
                s.extend(std::iter::repeat_n(' ', n * d));
            }
        };
        let sep = if indent.is_some() { "," } else { ", " };
        match self {
            J::Null => s.push_str("null"),
            J::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            J::Num(n) => s.push_str(n),
            J::Str(t) => py_str(s, t),
            J::Arr(a) if a.is_empty() => s.push_str("[]"),
            J::Obj(m) if m.is_empty() => s.push_str("{}"),
            J::Arr(a) => {
                s.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        s.push_str(sep);
                    }
                    nl(s, depth + 1);
                    v.write(s, indent, depth + 1);
                }
                nl(s, depth);
                s.push(']');
            }
            J::Obj(m) => {
                s.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        s.push_str(sep);
                    }
                    nl(s, depth + 1);
                    py_str(s, k);
                    s.push_str(": ");
                    v.write(s, indent, depth + 1);
                }
                nl(s, depth);
                s.push('}');
            }
        }
    }
}

/// A string as Python's json.dumps(ensure_ascii=False) writes it.
fn py_str(s: &mut String, t: &str) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            '\u{8}' => s.push_str("\\b"),
            '\u{c}' => s.push_str("\\f"),
            c if (c as u32) < 0x20 => s.push_str(&format!("\\u{:04x}", c as u32)),
            c => s.push(c),
        }
    }
    s.push('"');
}

struct Reader<'a> {
    s: &'a [u8],
    i: usize,
}

/// Nesting deeper than this is refused (the parser recurses).
const MAX_DEPTH: usize = 128;

impl Reader<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> Option<()> {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            Some(())
        } else {
            None
        }
    }

    fn value(&mut self, depth: usize) -> Option<J> {
        if depth > MAX_DEPTH {
            return None;
        }
        self.ws();
        match *self.s.get(self.i)? {
            b'n' => self.eat("null").map(|_| J::Null),
            b't' => self.eat("true").map(|_| J::Bool(true)),
            b'f' => self.eat("false").map(|_| J::Bool(false)),
            b'"' => self.string().map(J::Str),
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.eat("]").is_some() {
                    return Some(J::Arr(a));
                }
                loop {
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match *self.s.get(self.i)? {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            return Some(J::Arr(a));
                        }
                        _ => return None,
                    }
                }
            }
            b'{' => self.object(depth, &mut |_, _, _| {}).map(J::Obj),
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    /// An object; `span` sees each member's key and its value's byte range.
    fn object(
        &mut self,
        depth: usize,
        span: &mut dyn FnMut(&str, usize, usize),
    ) -> Option<Vec<(String, J)>> {
        self.i += 1;
        let mut m = Vec::new();
        self.ws();
        if self.eat("}").is_some() {
            return Some(m);
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return None;
            }
            let k = self.string()?;
            self.ws();
            self.eat(":")?;
            self.ws();
            let at = self.i;
            let v = self.value(depth + 1)?;
            span(&k, at, self.i);
            m.push((k, v));
            self.ws();
            match *self.s.get(self.i)? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Some(m);
                }
                _ => return None,
            }
        }
    }

    fn number(&mut self) -> Option<J> {
        let at = self.i;
        self.eat("-");
        let digits = |r: &mut Self| {
            let d = r.i;
            while r.i < r.s.len() && r.s[r.i].is_ascii_digit() {
                r.i += 1;
            }
            r.i > d
        };
        if self.eat("0").is_none() && !digits(self) {
            return None;
        }
        if self.eat(".").is_some() && !digits(self) {
            return None;
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !digits(self) {
                return None;
            }
        }
        Some(J::Num(
            std::str::from_utf8(&self.s[at..self.i]).ok()?.to_string(),
        ))
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = self.s.get(self.i..self.i + 4)?;
        if !h.iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        let h = std::str::from_utf8(h).ok()?;
        self.i += 4;
        u32::from_str_radix(h, 16).ok()
    }

    fn string(&mut self) -> Option<String> {
        self.i += 1;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).ok(),
                b'\\' => {
                    let e = *self.s.get(self.i)?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let u = self.hex4()?;
                            let u = if (0xD800..0xDC00).contains(&u)
                                && self.s[self.i..].starts_with(b"\\u")
                            {
                                let save = self.i;
                                self.i += 2;
                                match self.hex4()? {
                                    lo @ 0xDC00..0xE000 => {
                                        0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00)
                                    }
                                    _ => {
                                        self.i = save;
                                        u
                                    }
                                }
                            } else {
                                u
                            };
                            char::from_u32(u).unwrap_or('\u{FFFD}')
                        }
                        _ => return None,
                    };
                    let mut b = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                }
                0..0x20 => return None,
                c => out.push(c),
            }
        }
    }
}

/// Parse `s` as one JSON value (surrounding whitespace allowed).
pub fn parse_json(s: &str) -> Option<J> {
    let mut r = Reader {
        s: s.as_bytes(),
        i: 0,
    };
    let v = r.value(0)?;
    r.ws();
    (r.i == s.len()).then_some(v)
}

/// Parse `s` as one JSON object: its members, and each member's value as
/// the text it was written as.
fn parse_object_spans(s: &str) -> Option<Vec<(String, J, &str)>> {
    let mut r = Reader {
        s: s.as_bytes(),
        i: 0,
    };
    r.ws();
    if r.s.get(r.i) != Some(&b'{') {
        return None;
    }
    let mut spans = Vec::new();
    let m = r.object(0, &mut |_, a, b| spans.push((a, b)))?;
    r.ws();
    if r.i != s.len() {
        return None;
    }
    Some(
        m.into_iter()
            .zip(spans)
            .map(|((k, v), (a, b))| (k, v, &s[a..b]))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// The request.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A call in an assistant message of the history.
#[derive(Clone, Debug, PartialEq)]
pub struct HistCall {
    pub name: String,
    /// The `arguments` as the template gets them: the object if the
    /// string is JSON (or it was given as an object), else the raw string.
    pub args: Result<J, String>,
    /// The `arguments` string as given (None: given as an object).
    pub raw: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TMsg {
    pub role: TRole,
    pub content: String,
    /// `reasoning_content` (an earlier reply's thinking), if given.
    pub reasoning: Option<String>,
    pub calls: Vec<HistCall>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Choice {
    Auto,
    None,
    Required,
    Named(String),
}

/// The tool side of a chat request.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolReq {
    /// The `tools` entries, whole, in request order (a named choice: that
    /// one only; none: none).
    pub tools: Vec<J>,
    /// Every `tools` entry, whatever the choice (harmony renders them all
    /// for a named choice, so switching between it and auto keeps the
    /// prompt's prefix).
    pub offered: Vec<J>,
    pub choice: Choice,
    pub msgs: Vec<TMsg>,
}

impl ToolReq {
    /// Bytes of text in the prompt besides the messages' content (for the
    /// up-front size check).
    pub fn extra_bytes(&self) -> usize {
        let t: usize = self.tools.iter().map(|t| t.dumps().len()).sum();
        let c: usize = self
            .msgs
            .iter()
            .flat_map(|m| &m.calls)
            .map(|c| {
                c.name.len()
                    + c.args
                        .as_ref()
                        .map_or_else(|s| s.len(), |j| j.dumps().len())
            })
            .sum();
        t + c
    }
}

fn bad(param: &'static str, msg: impl Into<String>) -> ApiError {
    ApiError::bad(param, msg)
}

/// A message's text: a string, null/absent (empty), or text parts joined
/// with newlines (as api.rs's `content`).
fn text_of(v: Option<&J>) -> Result<String, ApiError> {
    match v {
        None | Some(J::Null) => Ok(String::new()),
        Some(J::Str(s)) => Ok(s.clone()),
        Some(J::Arr(parts)) => {
            let mut texts = Vec::new();
            for p in parts {
                match (
                    p.get("type").and_then(J::str),
                    p.get("text").and_then(J::str),
                ) {
                    (Some("text"), Some(t)) => texts.push(t),
                    _ => {
                        return Err(bad("messages", "only text content parts are supported"));
                    }
                }
            }
            Ok(texts.join("\n"))
        }
        Some(_) => Err(bad(
            "messages",
            "a message's content must be a string or an array of text parts",
        )),
    }
}

/// The tool side of a /v1/chat/completions body (already valid JSON with a
/// `messages` array: api.rs checked). None if it has no tools and no tool
/// turns: then it is an ordinary chat request.
pub fn parse(body: &[u8]) -> Result<Option<ToolReq>, ApiError> {
    let Some(J::Obj(o)) = std::str::from_utf8(body).ok().and_then(parse_json) else {
        return Ok(None);
    };
    let get = |k: &str| {
        o.iter()
            .rev()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v)
            .filter(|v| !v.is_null())
    };
    let tools = match get("tools") {
        None => Vec::new(),
        Some(J::Arr(a)) => a.clone(),
        // api.rs refuses it, as before.
        Some(_) => return Ok(None),
    };
    let Some(J::Arr(msgs)) = get("messages") else {
        return Ok(None);
    };
    let has_calls = |m: &J| matches!(m.get("tool_calls"), Some(J::Arr(a)) if !a.is_empty());
    let tool_turns = msgs.iter().any(|m| {
        matches!(m.get("role").and_then(J::str), Some("tool" | "ipython")) || has_calls(m)
    });
    if tools.is_empty() && !tool_turns {
        // A call can't be forced without tools (OpenAI: 400). Anything else
        // here is an ordinary chat request.
        let forced = match get("tool_choice") {
            Some(J::Str(s)) => s == "required",
            Some(J::Obj(_)) => true,
            _ => false,
        };
        if forced {
            return Err(bad("tool_choice", "'tool_choice' needs 'tools'"));
        }
        return Ok(None);
    }
    let mut names = Vec::new();
    for t in &tools {
        let name = match (t.get("type").and_then(J::str), t.get("function")) {
            (Some("function"), Some(f)) => f.get("name").and_then(J::str),
            _ => None,
        };
        let Some(name) = name.filter(|n| !n.is_empty()) else {
            return Err(bad(
                "tools",
                "each tool must be {\"type\": \"function\", \"function\": {\"name\": ...}}",
            ));
        };
        names.push(name.to_string());
    }
    let choice = match get("tool_choice") {
        None => Choice::Auto,
        Some(J::Str(s)) => match s.as_str() {
            "auto" => Choice::Auto,
            "none" => Choice::None,
            "required" => Choice::Required,
            _ => {
                return Err(bad(
                    "tool_choice",
                    "'tool_choice' must be auto, none, required or a function",
                ));
            }
        },
        Some(c) => match c
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(J::str)
        {
            Some(n) if names.iter().any(|m| m == n) => Choice::Named(n.to_string()),
            Some(n) => {
                return Err(bad(
                    "tool_choice",
                    format!("no tool named '{n}' in 'tools'"),
                ));
            }
            None => {
                return Err(bad(
                    "tool_choice",
                    "'tool_choice' must be auto, none, required or a function",
                ));
            }
        },
    };
    if matches!(choice, Choice::Required | Choice::Named(_)) && tools.is_empty() {
        return Err(bad("tool_choice", "'tool_choice' needs 'tools'"));
    }
    let mut out = Vec::with_capacity(msgs.len());
    for m in msgs {
        let role = match m.get("role").and_then(J::str) {
            Some("system" | "developer") => TRole::System,
            Some("user") => TRole::User,
            Some("assistant") => TRole::Assistant,
            Some("tool" | "ipython") => TRole::Tool,
            Some(r) => {
                return Err(bad(
                    "messages",
                    format!("role '{r}' is not supported (system, user, assistant, tool)"),
                ));
            }
            None => return Err(bad("messages", "a message has no 'role'")),
        };
        let mut calls = Vec::new();
        if let Some(J::Arr(a)) = m.get("tool_calls") {
            if role != TRole::Assistant {
                return Err(bad("messages", "only assistant messages have tool_calls"));
            }
            for c in a {
                let f = c.get("function");
                let Some(name) = f.and_then(|f| f.get("name")).and_then(J::str) else {
                    return Err(bad("messages", "a tool call has no function name"));
                };
                let args = match f.and_then(|f| f.get("arguments")) {
                    None | Some(J::Null) => Ok(J::Obj(Vec::new())),
                    Some(J::Str(s)) => parse_json(s).ok_or_else(|| s.clone()),
                    Some(v) => Ok(v.clone()),
                };
                let raw = match f.and_then(|f| f.get("arguments")) {
                    Some(J::Str(s)) => Some(s.clone()),
                    _ => None,
                };
                calls.push(HistCall {
                    name: name.to_string(),
                    args,
                    raw,
                });
            }
        }
        let reasoning = match m.get("reasoning_content") {
            Some(J::Str(s)) => Some(s.clone()),
            _ => None,
        };
        out.push(TMsg {
            role,
            content: text_of(m.get("content"))?,
            reasoning,
            calls,
        });
    }
    let offered = tools.clone();
    let tools = match &choice {
        Choice::None => Vec::new(),
        Choice::Named(n) => tools
            .into_iter()
            .zip(&names)
            .filter(|(_, m)| *m == n)
            .map(|(t, _)| t)
            .collect(),
        _ => tools,
    };
    Ok(Some(ToolReq {
        tools,
        offered,
        choice,
        msgs: out,
    }))
}

// ---------------------------------------------------------------------------
// Rendering.

/// Tokens of a rendered template: template text (`lit`, whose markers
/// become special tokens) and message text (`text`, plain), with the text
/// between two markers encoded as one piece, as a tokenizer splitting the
/// rendered string at special tokens does.
struct Toks<'t> {
    tok: &'t Tokenizer,
    markers: Vec<(&'static str, u32)>,
    ids: Vec<u32>,
    pending: String,
}

const MARKERS: [&str; 14] = [
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|python_tag|>",
];

/// Harmony's markers (gpt-oss).
const HARMONY_MARKERS: [&str; 7] = [
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|call|>",
    "<|constrain|>",
    "<|return|>",
];

impl<'t> Toks<'t> {
    fn new(tok: &'t Tokenizer, markers: &[&'static str]) -> Toks<'t> {
        let markers = markers
            .iter()
            .filter_map(|&m| chat::special(tok, m).map(|id| (m, id)))
            .collect();
        Toks {
            tok,
            markers,
            ids: Vec::new(),
            pending: String::new(),
        }
    }

    fn flush(&mut self) {
        if !self.pending.is_empty() {
            self.ids.extend(self.tok.encode(&self.pending));
            self.pending.clear();
        }
    }

    fn id(&mut self, id: u32) {
        self.flush();
        self.ids.push(id);
    }

    fn text(&mut self, s: &str) {
        self.pending.push_str(s);
    }

    fn lit(&mut self, mut s: &str) {
        while !s.is_empty() {
            let next = self
                .markers
                .iter()
                .filter_map(|&(m, id)| s.find(m).map(|at| (at, m.len(), id)))
                .min();
            match next {
                Some((at, len, id)) => {
                    self.pending.push_str(&s[..at]);
                    self.id(id);
                    s = &s[at + len..];
                }
                None => {
                    self.pending.push_str(s);
                    s = "";
                }
            }
        }
    }

    fn done(mut self) -> Vec<u32> {
        self.flush();
        self.ids
    }
}

/// Which template family a model's tools are rendered and parsed in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Family {
    Qwen3,
    Qwen25,
    Llama31,
    /// gpt-oss.
    Harmony,
}

pub fn family(t: &Template) -> Option<Family> {
    if t.fr.format == Format::ChatMl {
        Some(if t.think.is_some() {
            Family::Qwen3
        } else {
            Family::Qwen25
        })
    } else if matches!(t.fr.format, Format::Llama3 { .. }) {
        Some(Family::Llama31)
    } else if matches!(t.fr.format, Format::Harmony { .. }) {
        Some(Family::Harmony)
    } else {
        None
    }
}

const HERMES_HEAD: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const HERMES_TAIL: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
const LLAMA_TOOLS_HEAD: &str = "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n";

/// A history call's arguments as Qwen2.5's and Llama 3.1's templates print
/// them (`arguments | tojson`): the value, re-serialized; a string that
/// isn't JSON, as is.
fn args_tojson(c: &HistCall) -> String {
    match &c.args {
        Ok(j) => j.dumps(),
        Err(s) => s.clone(),
    }
}

/// Qwen3's template: a string is printed verbatim, anything else tojson.
fn args_qwen3(c: &HistCall) -> String {
    c.raw.clone().unwrap_or_else(|| args_tojson(c))
}

/// A rendered tool prompt.
pub struct Rendered {
    pub ids: Vec<u32>,
    /// How to read the reply.
    pub parser: Option<Parser>,
    /// The reply was prefilled past where a Qwen3 reply would think.
    pub no_reasoning: bool,
    /// Harmony: the reply's channel splitter, telling calls apart, in the
    /// state the prompt leaves it (a call it opened, for tool_choice).
    pub channels: Option<chat::Channels>,
}

/// The prompt for a tool request, as the model's template renders it.
pub fn render(
    t: &Template,
    tok: &Tokenizer,
    r: &ToolReq,
    no_think: bool,
) -> Result<Rendered, ApiError> {
    let Some(fam) = family(t) else {
        return Err(bad(
            "tools",
            "tool calling is not supported for this model's chat format",
        ));
    };
    if fam == Family::Harmony {
        return harmony(t, tok, r);
    }
    let forced = match &r.choice {
        Choice::Required => Some("{\"name\": \"".to_string()),
        Choice::Named(n) => Some(match fam {
            Family::Llama31 => format!("{{\"name\": \"{n}\", \"parameters\": "),
            _ => format!("{{\"name\": \"{n}\", \"arguments\": "),
        }),
        _ => None,
    };
    let mut k = Toks::new(tok, &MARKERS);
    match fam {
        Family::Qwen3 | Family::Qwen25 => hermes(&mut k, fam, r)?,
        Family::Llama31 => llama(&mut k, t, r)?,
        Family::Harmony => unreachable!(),
    }
    let thinks = fam == Family::Qwen3;
    if thinks && (no_think || forced.is_some()) {
        k.lit("<think>\n\n</think>\n\n");
    }
    let kind = if fam == Family::Llama31 {
        Kind::Json
    } else {
        Kind::Hermes
    };
    if let Some(seed) = &forced {
        if kind == Kind::Hermes {
            k.lit("<tool_call>\n");
        }
        k.text(seed);
    }
    let no_reasoning = thinks && forced.is_some();
    let parser = match (&r.choice, forced) {
        (Choice::None, _) => None,
        (_, Some(seed)) => Some(Parser::forced(kind, &seed)),
        _ => Some(Parser::new(kind)),
    };
    Ok(Rendered {
        ids: k.done(),
        parser,
        no_reasoning,
        channels: None,
    })
}

/// Qwen3's and Qwen2.5's templates.
fn hermes(k: &mut Toks, fam: Family, r: &ToolReq) -> Result<(), ApiError> {
    let m = &r.msgs;
    let first_sys = m.first().filter(|x| x.role == TRole::System);
    let default_sys = (fam == Family::Qwen25).then_some(chat::DEFAULT_SYSTEM);
    if !r.tools.is_empty() {
        k.lit("<|im_start|>system\n");
        match (fam, first_sys) {
            (Family::Qwen3, Some(s)) => {
                k.text(&s.content);
                k.lit("\n\n");
            }
            (Family::Qwen3, None) => {}
            (_, s) => {
                k.text(s.map_or(chat::DEFAULT_SYSTEM, |s| &s.content));
                k.lit("\n\n");
            }
        }
        k.lit(HERMES_HEAD);
        for t in &r.tools {
            k.lit("\n");
            k.text(&t.dumps());
        }
        k.lit(HERMES_TAIL);
    } else if let Some(s) = first_sys.map(|s| s.content.as_str()).or(default_sys) {
        k.lit("<|im_start|>system\n");
        k.text(s);
        k.lit("<|im_end|>\n");
    }
    // Qwen3: the last user message that isn't only tool responses; replies
    // after it keep their reasoning.
    let mut last_query = m.len().saturating_sub(1);
    for (i, x) in m.iter().enumerate().rev() {
        if x.role == TRole::User
            && !(x.content.contains("<tool_response>") && x.content.contains("</tool_response>"))
        {
            last_query = i;
            break;
        }
    }
    for (i, x) in m.iter().enumerate() {
        let last = i + 1 == m.len();
        match x.role {
            TRole::System if i == 0 => {}
            TRole::System | TRole::User => {
                k.lit(if x.role == TRole::User {
                    "<|im_start|>user\n"
                } else {
                    "<|im_start|>system\n"
                });
                k.text(&x.content);
                k.lit("<|im_end|>\n");
            }
            TRole::Assistant if fam == Family::Qwen25 => {
                if x.calls.is_empty() {
                    k.lit("<|im_start|>assistant\n");
                    k.text(&x.content);
                    k.lit("<|im_end|>\n");
                    continue;
                }
                k.lit("<|im_start|>assistant");
                if !x.content.is_empty() {
                    k.lit("\n");
                    k.text(&x.content);
                }
                for c in &x.calls {
                    k.lit("\n<tool_call>\n{\"name\": \"");
                    k.text(&c.name);
                    k.lit("\", \"arguments\": ");
                    k.text(&args_tojson(c));
                    k.lit("}\n</tool_call>");
                }
                k.lit("<|im_end|>\n");
            }
            TRole::Assistant => {
                let (mut content, mut reasoning) = (x.content.as_str(), "");
                match &x.reasoning {
                    Some(rc) => reasoning = rc,
                    None => {
                        if let Some(at) = x.content.find("</think>") {
                            let (_, after) = x.content.rsplit_once("</think>").unwrap_or_default();
                            content = after.trim_start_matches('\n');
                            let before = x.content[..at].trim_end_matches('\n');
                            reasoning = before
                                .rsplit_once("<think>")
                                .map_or(before, |(_, r)| r)
                                .trim_start_matches('\n');
                        }
                    }
                }
                if i > last_query && (last || !reasoning.is_empty()) {
                    k.lit("<|im_start|>assistant\n<think>\n");
                    k.text(reasoning.trim_matches('\n'));
                    k.lit("\n</think>\n\n");
                    k.text(content.trim_start_matches('\n'));
                } else {
                    k.lit("<|im_start|>assistant\n");
                    k.text(content);
                }
                for (j, c) in x.calls.iter().enumerate() {
                    if j > 0 || !content.is_empty() {
                        k.lit("\n");
                    }
                    k.lit("<tool_call>\n{\"name\": \"");
                    k.text(&c.name);
                    k.lit("\", \"arguments\": ");
                    k.text(&args_qwen3(c));
                    k.lit("}\n</tool_call>");
                }
                k.lit("<|im_end|>\n");
            }
            TRole::Tool => {
                if i == 0 || m[i - 1].role != TRole::Tool {
                    k.lit("<|im_start|>user");
                }
                k.lit("\n<tool_response>\n");
                k.text(&x.content);
                k.lit("\n</tool_response>");
                if last || m[i + 1].role != TRole::Tool {
                    k.lit("<|im_end|>\n");
                }
            }
        }
    }
    k.lit("<|im_start|>assistant\n");
    Ok(())
}

/// Llama 3.1's template (tools in the first user message).
fn llama(k: &mut Toks, t: &Template, r: &ToolReq) -> Result<(), ApiError> {
    if let Format::Llama3 { bos: Some(bos), .. } = t.fr.format {
        k.id(bos);
    }
    let (system, mut rest) = match r.msgs.split_first() {
        Some((s, rest)) if s.role == TRole::System => (s.content.trim(), rest),
        _ => ("", &r.msgs[..]),
    };
    let tools = !r.tools.is_empty();
    k.lit("<|start_header_id|>system<|end_header_id|>\n\n");
    if tools {
        k.lit("Environment: ipython\n");
    }
    k.lit(chat::LLAMA3_SYSTEM_HEAD);
    k.text(system);
    k.lit("<|eot_id|>");
    if tools {
        let Some((first, more)) = rest.split_first() else {
            return Err(bad(
                "messages",
                "Llama 3.1 puts the tools in the first user message: there is none",
            ));
        };
        rest = more;
        k.lit("<|start_header_id|>user<|end_header_id|>\n\n");
        k.lit(LLAMA_TOOLS_HEAD);
        for t in &r.tools {
            k.text(&t.dumps_indent(4));
            k.lit("\n\n");
        }
        k.text(first.content.trim());
        k.lit("<|eot_id|>");
    }
    for x in rest {
        match x.role {
            TRole::Tool => {
                k.lit("<|start_header_id|>ipython<|end_header_id|>\n\n");
                // The template's `content is iterable` holds for a string: tojson.
                k.text(&J::Str(x.content.clone()).dumps());
                k.lit("<|eot_id|>");
            }
            TRole::Assistant if !x.calls.is_empty() => {
                let [c] = &x.calls[..] else {
                    return Err(bad(
                        "messages",
                        "Llama 3.1 takes one tool call per assistant message",
                    ));
                };
                k.lit("<|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"");
                k.text(&c.name);
                k.lit("\", \"parameters\": ");
                k.text(&args_tojson(c));
                k.lit("}<|eot_id|>");
            }
            _ => {
                let role = match x.role {
                    TRole::System => "system",
                    TRole::User => "user",
                    _ => "assistant",
                };
                k.lit(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n"));
                k.text(x.content.trim());
                k.lit("<|eot_id|>");
            }
        }
    }
    k.lit("<|start_header_id|>assistant<|end_header_id|>\n\n");
    Ok(())
}

// ---------------------------------------------------------------------------
// Harmony (gpt-oss).

/// How `TextOut` hands a harmony reply with tools to the parser: one
/// string, its text escaped (`harmony_text`: every ESC becomes ESC '0') and
/// each call framed out of band, `ESC '1' {recipient} ESC '2' {args} ESC
/// '3'`, from the special tokens that delimit it. Text the model writes can
/// never frame a call, whatever it spells (the template's call syntax
/// included), and a `<|call|>` in the arguments' text doesn't end one.
const ESC: char = '\u{1}';

/// Text (content, a recipient or arguments) as `TextOut` passes it on.
pub fn harmony_text(s: &str) -> String {
    s.replace(ESC, "\u{1}0")
}

/// The opening of a call to `to`.
pub fn harmony_call_open(to: &str) -> String {
    format!("{ESC}1{}{ESC}2", harmony_text(to))
}

/// A call's end, at its closing marker (`<|call|>` or `<|end|>`).
pub const HARMONY_CALL_END: &str = "\u{1}3";

/// A call cut off before its closing marker (the token limit): not a call.
pub const HARMONY_CALL_CUT: &str = "\u{1}4";

/// gpt-oss's template: tools as TypeScript in the developer message, a call
/// as `<|start|>assistant to=functions.NAME<|channel|>commentary json
/// <|message|>{args}<|call|>`, its result as `<|start|>functions.NAME
/// to=assistant<|channel|>commentary<|message|>{content|tojson}<|end|>`.
/// The template's rules: only a message's first call is rendered (one call
/// per message), and a result is named after the last call; a call's
/// reasoning (`reasoning_content`, else its content) is kept as an analysis
/// message unless a later assistant message has no calls (a final answer);
/// other reasoning is dropped; a system message other than the first isn't
/// rendered. Where the template raises, this is lenient: given both content
/// and reasoning, the reasoning is kept; it does refuse a tool message with
/// no call before it.
fn harmony(t: &Template, tok: &Tokenizer, r: &ToolReq) -> Result<Rendered, ApiError> {
    let mut k = Toks::new(tok, &HARMONY_MARKERS);
    // A named choice is forced by the prefill below, not by the tool list.
    let listed = match r.choice {
        Choice::Named(_) => &r.offered,
        _ => &r.tools,
    };
    let tools = !listed.is_empty();
    k.lit("<|start|>system<|message|>");
    k.text(&chat::harmony_system());
    if tools {
        k.text("\nCalls to these tools must go to the commentary channel: 'functions'.");
    }
    k.lit("<|end|>");
    let (dev, msgs) = match r.msgs.split_first() {
        Some((s, rest)) if s.role == TRole::System => (s.content.as_str(), rest),
        _ => ("", &r.msgs[..]),
    };
    if !dev.is_empty() || tools {
        k.lit("<|start|>developer<|message|>");
        if !dev.is_empty() {
            k.text("# Instructions\n\n");
            k.text(dev);
            k.text("\n\n");
        }
        if tools {
            k.text("# Tools\n\n");
            k.text(&ts_namespace(listed));
        }
        k.lit("<|end|>");
    }
    let mut last: Option<&str> = None;
    for (i, m) in msgs.iter().enumerate() {
        match m.role {
            TRole::Assistant if !m.calls.is_empty() => {
                let answered = msgs[i + 1..]
                    .iter()
                    .any(|x| x.role == TRole::Assistant && x.calls.is_empty());
                let cot = match m.reasoning.as_deref() {
                    Some(s) if !s.is_empty() => s,
                    _ => &m.content,
                };
                if !answered && !cot.is_empty() {
                    k.lit("<|start|>assistant<|channel|>analysis<|message|>");
                    k.text(cot);
                    k.lit("<|end|>");
                }
                let c = &m.calls[0];
                k.lit("<|start|>assistant to=functions.");
                k.text(&c.name);
                k.lit("<|channel|>commentary json<|message|>");
                k.text(&args_tojson(c));
                k.lit("<|call|>");
                last = Some(&c.name);
            }
            TRole::Assistant => {
                k.lit("<|start|>assistant<|channel|>final<|message|>");
                k.text(&m.content);
                k.lit("<|end|>");
                last = None;
            }
            TRole::Tool => {
                let Some(name) = last else {
                    return Err(bad(
                        "messages",
                        "a tool message must follow an assistant message with a tool call",
                    ));
                };
                k.lit("<|start|>functions.");
                k.text(name);
                k.lit(" to=assistant<|channel|>commentary<|message|>");
                k.text(&J::Str(m.content.clone()).dumps());
                k.lit("<|end|>");
            }
            TRole::User => {
                k.lit("<|start|>user<|message|>");
                k.text(&m.content);
                k.lit("<|end|>");
            }
            TRole::System => {}
        }
    }
    k.lit("<|start|>assistant");
    // tool_choice required / named: the reply opens a call (skipping the
    // analysis), in the history form. Required stops before the `.`: o200k
    // splits `to=functions.get_weather` as ` to`, `=functions`, `.get`,
    // `_weather`, so the model writes `.name` as it learned to.
    match &r.choice {
        Choice::Required => k.text(" to=functions"),
        Choice::Named(n) => {
            k.text(" to=functions.");
            k.text(n);
            k.lit("<|channel|>commentary json<|message|>");
        }
        _ => {}
    }
    let ids = k.done();
    let channels = match r.choice {
        Choice::None => None,
        _ => {
            let call = chat::special(tok, "<|call|>");
            let mut ch = chat::Channels::new(&t.fr).map(|c| c.with_tools(call));
            if let Some(ch) = &mut ch {
                let at = ids.iter().rposition(|&i| i == t.fr.im_start).unwrap_or(0);
                for &id in &ids[at..] {
                    ch.push(id, &tok.decode_bytes(&[id]));
                }
                ch.primed();
            }
            ch
        }
    };
    Ok(Rendered {
        ids,
        parser: channels.as_ref().map(|_| Parser::new_harmony()),
        no_reasoning: false,
        channels,
    })
}

/// Jinja's truth of a value (absent: false).
fn truthy(v: Option<&J>) -> bool {
    match v {
        None | Some(J::Null) => false,
        Some(J::Bool(b)) => *b,
        Some(J::Num(n)) => n.parse::<f64>() != Ok(0.0),
        Some(J::Str(s)) => !s.is_empty(),
        Some(J::Arr(a)) => !a.is_empty(),
        Some(J::Obj(o)) => !o.is_empty(),
    }
}

/// A value as jinja prints it (Python's `str`); containers as JSON.
fn py(v: &J) -> String {
    match v {
        J::Str(s) => s.clone(),
        J::Bool(true) => "True".into(),
        J::Bool(false) => "False".into(),
        J::Null => "None".into(),
        J::Num(n) => n.clone(),
        _ => v.dumps(),
    }
}

/// `name in (required or [])`, Python's `in`.
fn listed(required: Option<&J>, name: &str) -> bool {
    match required {
        Some(J::Arr(a)) => a.iter().any(|x| x.str() == Some(name)),
        Some(J::Str(s)) => s.contains(name),
        Some(J::Obj(o)) => o.iter().any(|(k, _)| k == name),
        _ => false,
    }
}

/// The template's `render_typescript_type`: a JSON Schema as a TypeScript
/// type, whitespace and quirks included (a nested object's members on
/// separate lines, a oneOf's variants joined with " | \n"; its "any" for
/// object variants never applies: the flag is set inside a loop, and
/// jinja's scoping drops it).
fn ts_type(p: &J) -> String {
    let ty = p.get("type");
    let is = |s: &str| ty.and_then(J::str) == Some(s);
    let nullable = if truthy(p.get("nullable")) {
        " | null"
    } else {
        ""
    };
    if is("array") {
        let items = p.get("items").filter(|i| truthy(Some(i)));
        let Some(it) = items else {
            return format!("any[]{nullable}");
        };
        let t = match it.get("type").and_then(J::str) {
            Some("string") => "string[]".to_string(),
            Some("number" | "integer") => "number[]".to_string(),
            Some("boolean") => "boolean[]".to_string(),
            _ => {
                let inner = ts_type(it);
                if inner == "object | object" || inner.chars().count() > 50 {
                    "any[]".to_string()
                } else {
                    inner + "[]"
                }
            }
        };
        return t + nullable;
    }
    if let Some(J::Arr(a)) = ty
        && !a.is_empty()
    {
        return a.iter().map(py).collect::<Vec<_>>().join(" | ");
    }
    if truthy(p.get("oneOf")) {
        let Some(J::Arr(vs)) = p.get("oneOf") else {
            return "any".into();
        };
        let mut s = String::new();
        for (i, v) in vs.iter().enumerate() {
            s.push_str(&ts_type(v));
            if let Some(d) = v.get("description").filter(|d| truthy(Some(d))) {
                s.push_str("// ");
                s.push_str(&py(d));
            }
            if let Some(d) = v.get("default") {
                s.push_str("                    // default: ");
                s.push_str(&d.dumps());
            }
            if i + 1 < vs.len() {
                s.push_str(" | \n");
            }
        }
        return s;
    }
    if is("string") {
        return match p.get("enum") {
            Some(J::Arr(e)) if !e.is_empty() => {
                let e: Vec<String> = e.iter().map(py).collect();
                format!("\"{}\"", e.join("\" | \""))
            }
            Some(e) if truthy(Some(e)) => format!("\"{}\"", py(e)),
            _ => format!("string{nullable}"),
        };
    }
    if is("number") || is("integer") {
        return "number".into();
    }
    if is("boolean") {
        return "boolean".into();
    }
    if is("object") {
        return match p.get("properties") {
            Some(J::Obj(props)) if !props.is_empty() => {
                let req = p.get("required");
                let mut s = "{\n".to_string();
                for (i, (name, spec)) in props.iter().enumerate() {
                    s.push_str(name);
                    if !listed(req, name) {
                        s.push('?');
                    }
                    s.push_str(": \n                ");
                    s.push_str(&ts_type(spec));
                    if i + 1 < props.len() {
                        s.push_str(", ");
                    }
                }
                s + "}"
            }
            _ => "object".into(),
        };
    }
    "any".into()
}

/// The template's `render_tool_namespace("functions", tools)`.
fn ts_namespace(tools: &[J]) -> String {
    let mut s = "## functions\n\nnamespace functions {\n\n".to_string();
    let none = J::Null;
    for t in tools {
        let f = t.get("function").unwrap_or(&none);
        s.push_str("// ");
        s.push_str(&f.get("description").map_or(String::new(), py));
        s.push('\n');
        s.push_str("type ");
        s.push_str(&f.get("name").map_or(String::new(), py));
        s.push_str(" = ");
        let params = f.get("parameters").filter(|p| truthy(Some(p)));
        match params.and_then(|p| p.get("properties")) {
            Some(J::Obj(props)) if !props.is_empty() => {
                let req = params.and_then(|p| p.get("required"));
                s.push_str("(_: {\n");
                for (name, spec) in props {
                    if let Some(d) = spec.get("description").filter(|d| truthy(Some(d))) {
                        s.push_str("// ");
                        s.push_str(&py(d));
                        s.push('\n');
                    }
                    s.push_str(name);
                    if !listed(req, name) {
                        s.push('?');
                    }
                    s.push_str(": ");
                    s.push_str(&ts_type(spec));
                    if let Some(d) = spec.get("default") {
                        if truthy(spec.get("enum")) {
                            s.push_str(", // default: ");
                            s.push_str(&py(d));
                        } else if truthy(spec.get("oneOf")) {
                            s.push_str("// default: ");
                            s.push_str(&py(d));
                        } else {
                            s.push_str(", // default: ");
                            s.push_str(&d.dumps());
                        }
                    }
                    s.push_str(",\n");
                }
                s.push_str("}) => any;\n\n");
            }
            _ => s.push_str("() => any;\n\n"),
        }
    }
    s + "} // namespace functions"
}

// ---------------------------------------------------------------------------
// The reply.

/// A call the model made.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub id: String,
    pub name: String,
    /// The arguments' JSON text, as the model wrote it.
    pub args: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Text(String),
    Call(Call),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// `<tool_call>{...}</tool_call>` blocks among text.
    Hermes,
    /// A whole reply of `{"name": ..., "parameters": ...}` (Llama 3.1).
    Json,
    /// Harmony: calls in their history form among the final text (as
    /// `TextOut` writes them from the channels).
    Harmony,
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// A fresh call id: `call_` and 24 hex digits, unique in this process.
fn call_id() -> String {
    static SEED: AtomicU64 = AtomicU64::new(0);
    static NEXT: AtomicU64 = AtomicU64::new(0);
    if SEED.load(Ordering::Relaxed) == 0 {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |d| d.as_nanos() as u64)
            | 1;
        let _ = SEED.compare_exchange(0, t, Ordering::Relaxed, Ordering::Relaxed);
    }
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    // splitmix64 of seed + n: distinct for distinct n.
    let mut z = SEED
        .load(Ordering::Relaxed)
        .wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    format!("call_{z:016x}{:08x}", n as u32)
}

/// A call from a block's JSON: an object with a string "name"; arguments
/// from "arguments" (Llama: "parameters"), as written, `{}` if absent. A
/// string value is the arguments' text if it is JSON (a double-encoded
/// object), else the string's own JSON literal: arguments are always JSON.
fn call_of(body: &str) -> Option<Call> {
    let m = parse_object_spans(body.trim())?;
    let name = m.iter().find(|(k, _, _)| k == "name")?.1.str()?.to_string();
    let args = match m
        .iter()
        .find(|(k, _, _)| k == "arguments" || k == "parameters")
    {
        None => "{}".to_string(),
        Some((_, J::Str(s), raw)) => match parse_json(s) {
            Some(_) => s.clone(),
            None => raw.to_string(),
        },
        Some((_, _, raw)) => raw.to_string(),
    };
    Some(Call {
        id: call_id(),
        name,
        args,
    })
}

/// Reads a reply into content and calls as its text arrives.
pub struct Parser {
    kind: Kind,
    /// Hermes: inside a block. Json: holding the reply.
    inside: bool,
    /// Json: the reply's first non-blank character is still to come.
    undecided: bool,
    /// Text not yet decided (a block's body; a possible `<tool_call>` start).
    buf: String,
    /// Hermes, inside a block: `buf` before this has no `</tool_call>`
    /// (a char boundary), so each search covers only the new text.
    scan: usize,
    /// Whitespace after the last content, held: dropped if a call follows.
    ws: String,
    /// Calls made so far.
    calls: usize,
    /// No content since the last call: leading blanks drop.
    after_call: bool,
    /// Harmony, inside a call: its recipient and arguments so far.
    hname: String,
    hargs: String,
}

impl Parser {
    fn new(kind: Kind) -> Parser {
        Parser {
            kind,
            inside: false,
            undecided: kind == Kind::Json,
            buf: String::new(),
            scan: 0,
            ws: String::new(),
            calls: 0,
            after_call: false,
            hname: String::new(),
            hargs: String::new(),
        }
    }

    /// A harmony reply's parser (`TextOut` writes the calls in its content).
    pub fn new_harmony() -> Parser {
        Parser::new(Kind::Harmony)
    }

    /// A reply prefilled with the start of a call (`seed`: the call's text
    /// after `<tool_call>`).
    fn forced(kind: Kind, seed: &str) -> Parser {
        let mut p = Parser::new(kind);
        p.inside = true;
        p.undecided = false;
        p.buf = if kind == Kind::Hermes {
            format!("\n{seed}")
        } else {
            seed.to_string()
        };
        p
    }

    /// Calls made so far.
    pub fn calls(&self) -> usize {
        self.calls
    }

    fn content(&mut self, t: &str, out: &mut Vec<Piece>) {
        let body = t.trim_end();
        if body.is_empty() {
            self.ws.push_str(t);
            return;
        }
        let mut s = std::mem::take(&mut self.ws);
        if std::mem::replace(&mut self.after_call, false) {
            // Blank lines after a call only separate it from what follows.
            s.clear();
            s.push_str(body.trim_start());
        } else {
            s.push_str(body);
        }
        out.push(Piece::Text(s));
        self.ws = t[body.len()..].to_string();
    }

    fn call(&mut self, c: Call, out: &mut Vec<Piece>) {
        self.ws.clear();
        self.calls += 1;
        self.after_call = true;
        out.push(Piece::Call(c));
    }

    /// More of the reply's text: what is decided now.
    pub fn push(&mut self, s: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        self.buf.push_str(s);
        match self.kind {
            Kind::Hermes => self.hermes(&mut out),
            Kind::Harmony => self.harmony(&mut out),
            Kind::Json => {
                if self.undecided {
                    let t = self.buf.trim_start();
                    let t = t.strip_prefix("<|python_tag|>").unwrap_or(t).trim_start();
                    if t.is_empty() || "<|python_tag|>".starts_with(t) {
                        return out;
                    }
                    self.undecided = false;
                    self.inside = t.starts_with('{');
                }
                if !self.inside {
                    let b = std::mem::take(&mut self.buf);
                    if !b.is_empty() {
                        out.push(Piece::Text(b));
                    }
                }
            }
        }
        out
    }

    fn hermes(&mut self, out: &mut Vec<Piece>) {
        loop {
            if self.inside {
                let Some(at) = self.buf[self.scan..].find(CLOSE).map(|a| a + self.scan) else {
                    // The next search starts where a `</tool_call>` could still begin.
                    let mut w = self.buf.len().saturating_sub(CLOSE.len() - 1);
                    while !self.buf.is_char_boundary(w) {
                        w -= 1;
                    }
                    self.scan = w;
                    return;
                };
                let body = self.buf[..at].to_string();
                self.buf.drain(..at + CLOSE.len());
                self.inside = false;
                self.scan = 0;
                self.block(&body, true, out);
            } else if let Some(at) = self.buf.find(OPEN) {
                let text = self.buf[..at].to_string();
                self.buf.drain(..at + OPEN.len());
                self.content(&text, out);
                self.inside = true;
            } else {
                // Hold back a tail that may still become `<tool_call>`.
                let keep = (1..OPEN.len())
                    .rev()
                    .find(|&n| self.buf.ends_with(&OPEN[..n]))
                    .unwrap_or(0);
                let cut = self.buf.len() - keep;
                let text: String = self.buf.drain(..cut).collect();
                if !text.is_empty() {
                    self.content(&text, out);
                }
                return;
            }
        }
    }

    /// A finished block (`closed`: by `</tool_call>`): a call, or content.
    fn block(&mut self, body: &str, closed: bool, out: &mut Vec<Piece>) {
        match call_of(body) {
            Some(c) => self.call(c, out),
            None => {
                let raw = format!("{OPEN}{body}{}", if closed { CLOSE } else { "" });
                let ws = std::mem::take(&mut self.ws);
                self.after_call = false;
                out.push(Piece::Text(ws + &raw));
            }
        }
    }

    /// Harmony: decode `TextOut`'s escaped stream (`harmony_text`,
    /// `harmony_call_open`): text is content, a framed call is a call.
    fn harmony(&mut self, out: &mut Vec<Piece>) {
        loop {
            let Some(at) = self.buf.find(ESC) else {
                let t = std::mem::take(&mut self.buf);
                self.harmony_part(&t, out);
                return;
            };
            let code = at + ESC.len_utf8();
            let Some(c) = self.buf[code..].chars().next() else {
                // The escape's code is still to come.
                let t: String = self.buf.drain(..at).collect();
                self.harmony_part(&t, out);
                return;
            };
            let t: String = self.buf.drain(..at).collect();
            self.harmony_part(&t, out);
            self.buf.drain(..ESC.len_utf8() + c.len_utf8());
            match c {
                '0' => self.harmony_part(&ESC.to_string(), out),
                '1' => {
                    debug_assert!(!self.inside, "TextOut opens one call at a time");
                    self.inside = true;
                    self.undecided = true;
                    self.hname.clear();
                    self.hargs.clear();
                }
                '2' => self.undecided = false,
                '3' if self.inside => self.harmony_end(true, out),
                '4' if self.inside => self.harmony_end(false, out),
                _ => {}
            }
        }
    }

    /// Text of the current place: content, a call's name (`undecided`) or
    /// its arguments.
    fn harmony_part(&mut self, t: &str, out: &mut Vec<Piece>) {
        if t.is_empty() {
            return;
        }
        match (self.inside, self.undecided) {
            (false, _) => self.content(t, out),
            (true, true) => self.hname.push_str(t),
            (true, false) => self.hargs.push_str(t),
        }
    }

    /// A call's end. A call iff it was `closed` by its marker and its
    /// arguments are a JSON object (as OpenAI clients parse them); else its
    /// arguments' text is content (never axon's framing), and blank
    /// arguments are nothing (never `{}`).
    fn harmony_end(&mut self, closed: bool, out: &mut Vec<Piece>) {
        self.inside = false;
        self.undecided = false;
        let name = std::mem::take(&mut self.hname);
        let args = std::mem::take(&mut self.hargs);
        let name = name.strip_prefix("functions.").unwrap_or(&name);
        let a = args.trim();
        if closed && !name.is_empty() && matches!(parse_json(a), Some(J::Obj(_))) {
            let c = Call {
                id: call_id(),
                name: name.to_string(),
                args: a.to_string(),
            };
            self.call(c, out);
        } else if !a.is_empty() {
            let ws = std::mem::take(&mut self.ws);
            self.after_call = false;
            out.push(Piece::Text(ws + &args));
        }
    }

    /// The end of the reply: the rest.
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        let buf = std::mem::take(&mut self.buf);
        match self.kind {
            Kind::Harmony => {
                self.buf = buf;
                self.harmony(&mut out);
                if self.inside {
                    // No end frame: cut off.
                    self.harmony_end(false, &mut out);
                }
                // A lone escape left at the end can't come from TextOut.
                let rest = std::mem::take(&mut self.buf);
                if !rest.is_empty() {
                    self.content(&rest, &mut out);
                }
            }
            Kind::Hermes if self.inside => {
                self.inside = false;
                self.block(&buf, false, &mut out);
            }
            Kind::Hermes => {
                if !buf.is_empty() {
                    self.content(&buf, &mut out);
                }
            }
            Kind::Json => {
                if self.inside {
                    let t = buf.trim_start();
                    let t = t.strip_prefix("<|python_tag|>").unwrap_or(t);
                    // One object (whose strings may hold `;`), else several
                    // separated by `;`.
                    let calls: Option<Vec<Call>> = match call_of(t) {
                        Some(c) => Some(vec![c]),
                        None => t
                            .split(';')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(call_of)
                            .collect(),
                    };
                    match calls {
                        Some(cs) if !cs.is_empty() => {
                            for c in cs {
                                self.call(c, &mut out);
                            }
                        }
                        _ => out.push(Piece::Text(buf)),
                    }
                } else if !buf.is_empty() {
                    out.push(Piece::Text(buf));
                }
            }
        }
        let ws = std::mem::take(&mut self.ws);
        if self.calls == 0 && !ws.is_empty() {
            out.push(Piece::Text(ws));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Response JSON.

fn call_json(c: &Call) -> String {
    format!(
        "{{\"id\":{},\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":{}}}}}",
        quote(&c.id),
        quote(&c.name),
        quote(&c.args)
    )
}

/// The two stream chunks of call `index`: its id and name (empty
/// arguments), then its arguments.
pub fn call_chunks(head: &str, index: usize, c: &Call) -> [String; 2] {
    let wrap = |tc: String| {
        format!(
            "{{{head},\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{tc}]}},\"logprobs\":null,\"finish_reason\":null}}]}}"
        )
    };
    [
        wrap(format!(
            "{{\"index\":{index},\"id\":{},\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":\"\"}}}}",
            quote(&c.id),
            quote(&c.name)
        )),
        wrap(format!(
            "{{\"index\":{index},\"function\":{{\"arguments\":{}}}}}",
            quote(&c.args)
        )),
    ]
}

/// A whole chat response with tool calls (content null if there is none).
pub fn chat_response(
    head: &str,
    content: &str,
    reasoning: &str,
    calls: &[Call],
    finish: &str,
    u: &Usage,
) -> String {
    let content = if content.is_empty() {
        "null".to_string()
    } else {
        quote(content)
    };
    let reasoning = if reasoning.is_empty() {
        String::new()
    } else {
        format!(",\"reasoning_content\":{}", quote(reasoning))
    };
    let calls: Vec<String> = calls.iter().map(call_json).collect();
    format!(
        "{{{head},\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{content}{reasoning},\"tool_calls\":[{}]}},\"logprobs\":null,\"finish_reason\":{}}}],\"usage\":{},\"timings\":{}}}",
        calls.join(","),
        quote(finish),
        u.json(),
        u.timings()
    )
}

#[cfg(test)]
mod tests;
