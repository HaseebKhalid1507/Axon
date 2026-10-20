//! OpenAI's Responses API (`POST /v1/responses`, what codex speaks) over the
//! chat-completions path: the body is translated into a
//! /v1/chat/completions body (with tools.rs's order-preserving JSON, so tool
//! schemas render byte for byte as they do there), parsed by api.rs, and run
//! by the same `run_request` (auth, size limits, queue, engine). Only the
//! answer's shape is this module's. docs/serve-apis.md lists the fields.
//!
//! Input items to chat messages:
//! - `instructions` is a leading system message (not carried over by
//!   `previous_response_id`, as OpenAI's).
//! - `message` items (or bare `{role, content}`): user, system, developer
//!   (a system message), assistant. Content is a string or text parts
//!   (`input_text`, `output_text`, `text`, `refusal`), joined with newlines
//!   as chat content parts are. Images, files and audio are refused.
//! - An assistant turn is the run of `reasoning`, assistant `message` and
//!   `function_call` items between other items: one assistant message
//!   (`reasoning_content`, `content`, `tool_calls`), which is what this
//!   module answers with, in that order, so a reply sent back as input
//!   renders as the chat-completions history would.
//! - `function_call_output` is a `tool` message.
//!
//! Tools: `function` tools become chat tools (`{"type": "function",
//! "function": {name, description, parameters, strict}}`). Other tool types
//! (`web_search`, codex's `namespace`, `custom`, `local_shell`, ...) are
//! hosted or need grammar-constrained output, and are dropped, so a client
//! that always sends them still works. `tool_choice` maps one to one; a
//! choice of a hosted tool or `allowed_tools` is refused.
//!
//! `previous_response_id`: responses with `store` (OpenAI's default true)
//! keep their conversation in memory (the last `STORE_MAX` of them, at most
//! `STORE_BYTES`), and a request naming one continues it. An unknown or
//! evicted id is a 400.
//!
//! Output: `[reasoning?, message?, function_call...]`. Streaming sends
//! OpenAI's events (`response.created`, `response.in_progress`,
//! `response.output_item.added`, `response.content_part.added`,
//! `response.output_text.delta`, `response.reasoning_text.delta`,
//! `response.function_call_arguments.delta`/`.done`,
//! `response.output_item.done`, `response.completed`, or
//! `response.incomplete` when cut off by `max_output_tokens` or the
//! context, `error` + `response.failed` on an engine error), each with a
//! `sequence_number`. Function calls are sent when the reply ends, after
//! the message item (the parser decides a call whole anyway).

use super::api::{self, ApiError, GenReq, Usage};
use super::tools::{self, Call, J, Piece};
use super::{Conn, Event, Events, Request, Shared, json_reply};
use crate::tokenizer::Tokenizer;
use serde_json::{Map, Value, json};
use std::collections::VecDeque;
use std::io;
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::Instant;

/// Most stored responses (`previous_response_id`).
pub const STORE_MAX: usize = 256;
/// Most bytes of stored conversations (their JSON).
pub const STORE_BYTES: usize = 64 << 20;

/// The Responses side of a request: what the response object echoes, and
/// the conversation to store with the reply.
pub struct Req {
    echo: Map<String, Value>,
    store: bool,
    /// The conversation so far as chat messages (without `instructions`).
    history: Vec<J>,
}

fn bad(param: &'static str, msg: impl Into<String>) -> ApiError {
    ApiError::bad(param, msg)
}

fn unsupported(param: &'static str, what: &str) -> ApiError {
    bad(param, format!("'{param}'{what} is not supported by axon"))
}

fn s(t: &str) -> J {
    J::Str(t.to_string())
}

fn obj(m: Vec<(&str, J)>) -> J {
    J::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// A content value's text: a string, or text parts joined with newlines.
fn text_of(v: Option<&J>, param: &'static str) -> Result<String, ApiError> {
    match v {
        None | Some(J::Null) => Ok(String::new()),
        Some(J::Str(t)) => Ok(t.clone()),
        Some(J::Arr(parts)) => {
            let mut out = Vec::new();
            for p in parts {
                let ty = p.get("type").and_then(J::str);
                let key = match ty {
                    Some(
                        "input_text" | "output_text" | "text" | "summary_text" | "reasoning_text",
                    ) => "text",
                    Some("refusal") => "refusal",
                    Some(t) => {
                        return Err(bad(
                            param,
                            format!("content parts of type '{t}' are not supported (text only)"),
                        ));
                    }
                    None => return Err(bad(param, "a content part has no 'type'")),
                };
                match p.get(key).and_then(J::str) {
                    Some(t) => out.push(t),
                    None => {
                        return Err(bad(param, format!("a content part has no '{key}' string")));
                    }
                }
            }
            Ok(out.join("\n"))
        }
        Some(_) => Err(bad(
            param,
            "content must be a string or an array of content parts",
        )),
    }
}

/// An assistant turn being gathered from consecutive items.
#[derive(Default)]
struct Turn {
    reasoning: String,
    content: String,
    calls: Vec<J>,
}

impl Turn {
    fn msg(self) -> J {
        let mut m = vec![("role", s("assistant")), ("content", J::Str(self.content))];
        if !self.reasoning.is_empty() {
            m.push(("reasoning_content", J::Str(self.reasoning)));
        }
        if !self.calls.is_empty() {
            m.push(("tool_calls", J::Arr(self.calls)));
        }
        obj(m)
    }
}

/// `input` items as chat messages, appended to `out`.
fn items(input: &[J], out: &mut Vec<J>) -> Result<(), ApiError> {
    let mut turn: Option<Turn> = None;
    let flush = |turn: &mut Option<Turn>, out: &mut Vec<J>| {
        if let Some(t) = turn.take() {
            out.push(t.msg());
        }
    };
    for it in input {
        if !matches!(it, J::Obj(_)) {
            return Err(bad("input", "each input item must be an object"));
        }
        let ty = it.get("type").and_then(J::str);
        let role = it.get("role").and_then(J::str);
        match (ty, role) {
            (Some("message") | None, Some(r)) => {
                let text = text_of(it.get("content"), "input")?;
                match r {
                    "assistant" => {
                        if turn
                            .as_ref()
                            .is_some_and(|t| !t.content.is_empty() || !t.calls.is_empty())
                        {
                            flush(&mut turn, out);
                        }
                        turn.get_or_insert_default().content.push_str(&text);
                    }
                    "user" | "system" | "developer" => {
                        flush(&mut turn, out);
                        out.push(obj(vec![("role", s(r)), ("content", J::Str(text))]));
                    }
                    r => {
                        return Err(bad(
                            "input",
                            format!(
                                "role '{r}' is not supported (user, system, developer, assistant)"
                            ),
                        ));
                    }
                }
            }
            (Some("message"), None) => return Err(bad("input", "a message item has no 'role'")),
            (Some("function_call"), _) => {
                let name = it.get("name").and_then(J::str);
                let id = it.get("call_id").and_then(J::str);
                let (Some(name), Some(id)) = (name, id) else {
                    return Err(bad(
                        "input",
                        "a function_call item needs 'name' and 'call_id' strings",
                    ));
                };
                let args = match it.get("arguments") {
                    None | Some(J::Null) => s("{}"),
                    Some(a @ J::Str(_)) => a.clone(),
                    Some(_) => {
                        return Err(bad(
                            "input",
                            "a function_call item's 'arguments' must be a string",
                        ));
                    }
                };
                let call = obj(vec![
                    ("id", s(id)),
                    ("type", s("function")),
                    (
                        "function",
                        obj(vec![("name", s(name)), ("arguments", args)]),
                    ),
                ]);
                turn.get_or_insert_default().calls.push(call);
            }
            (Some("function_call_output"), _) => {
                flush(&mut turn, out);
                let Some(id) = it.get("call_id").and_then(J::str) else {
                    return Err(bad(
                        "input",
                        "a function_call_output item needs a 'call_id' string",
                    ));
                };
                let text = text_of(it.get("output"), "input")?;
                out.push(obj(vec![
                    ("role", s("tool")),
                    ("tool_call_id", s(id)),
                    ("content", J::Str(text)),
                ]));
            }
            (Some("reasoning"), _) => {
                // Its text (the raw reasoning, else the summary); an
                // encrypted-only item (another server's) has none.
                let mut text = text_of(it.get("content"), "input")?;
                if text.is_empty() {
                    text = text_of(it.get("summary"), "input")?;
                }
                if turn
                    .as_ref()
                    .is_some_and(|t| !t.content.is_empty() || !t.calls.is_empty())
                {
                    flush(&mut turn, out);
                }
                turn.get_or_insert_default().reasoning.push_str(&text);
            }
            (Some(t), _) => {
                return Err(bad(
                    "input",
                    format!(
                        "input items of type '{t}' are not supported (message, function_call, \
                         function_call_output, reasoning)"
                    ),
                ));
            }
            (None, None) => return Err(bad("input", "an input item needs a 'type' or a 'role'")),
        }
    }
    flush(&mut turn, out);
    Ok(())
}

/// A response id: `resp-`, 128 random bits (/dev/urandom) and a
/// process-wide sequence number (hex). The id is the only key to a stored
/// conversation (`previous_response_id`), so it must not be guessable; the
/// sequence number keeps it unique in the process (two servers in one
/// process, the tests) whatever the random bits.
pub(super) fn fresh_id() -> String {
    use std::io::Read;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut b = [0u8; 16];
    let ok = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b));
    if ok.is_err() {
        // No /dev/urandom: the clock, the counter and an address, mixed.
        // Not unguessable; say so once.
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!("axon serve: /dev/urandom unreadable: response ids are guessable")
        });
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let mut z = t ^ n.rotate_left(32) ^ (&b as *const _ as u64);
        for c in b.chunks_mut(8) {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            c.copy_from_slice(&(x ^ (x >> 31)).to_le_bytes());
        }
    }
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!("resp-{hex}{n:x}")
}

/// The stored conversations, oldest first: (response id, messages, bytes).
static STORE: Mutex<VecDeque<(String, Vec<J>, usize)>> = Mutex::new(VecDeque::new());

fn store_get(id: &str) -> Option<Vec<J>> {
    let st = STORE.lock().unwrap_or_else(|e| e.into_inner());
    st.iter()
        .find(|(i, _, _)| i == id)
        .map(|(_, m, _)| m.clone())
}

/// About the JSON size of `j` (its strings and numbers plus a little per
/// node), without printing it.
fn approx_bytes(j: &J) -> usize {
    8 + match j {
        J::Str(t) | J::Num(t) => t.len(),
        J::Arr(a) => a.iter().map(approx_bytes).sum(),
        J::Obj(m) => m.iter().map(|(k, v)| k.len() + approx_bytes(v)).sum(),
        J::Null | J::Bool(_) => 0,
    }
}

fn store_put(id: String, msgs: Vec<J>) {
    let bytes: usize = msgs.iter().map(approx_bytes).sum();
    if bytes > STORE_BYTES {
        return;
    }
    let mut st = STORE.lock().unwrap_or_else(|e| e.into_inner());
    let mut total: usize = st.iter().map(|(_, _, b)| b).sum::<usize>() + bytes;
    while st.len() >= STORE_MAX || total > STORE_BYTES {
        let Some((_, _, b)) = st.pop_front() else {
            break;
        };
        total -= b;
    }
    st.push_back((id, msgs, bytes));
}

/// A Responses `tools` entry as a chat tool (None: not a function tool).
fn chat_tool(t: &J) -> Result<Option<J>, ApiError> {
    let J::Obj(m) = t else {
        return Err(bad("tools", "each tool must be an object"));
    };
    match t.get("type").and_then(J::str) {
        Some("function") => {}
        Some(_) => return Ok(None),
        None => return Err(bad("tools", "a tool has no 'type'")),
    }
    if t.get("name").and_then(J::str).is_none_or(str::is_empty) {
        return Err(bad("tools", "a function tool needs a 'name'"));
    }
    let f: Vec<(String, J)> = m.iter().filter(|(k, _)| k != "type").cloned().collect();
    Ok(Some(obj(vec![
        ("type", s("function")),
        ("function", J::Obj(f)),
    ])))
}

/// Parse a /v1/responses body: the chat request it runs as, and what the
/// answer needs.
pub fn parse(body: &[u8]) -> Result<(GenReq, Req), ApiError> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::new(400, "invalid_request_error", format!("invalid JSON: {e}")))?;
    let Value::Object(o) = v else {
        return Err(ApiError::new(
            400,
            "invalid_request_error",
            "the body must be a JSON object",
        ));
    };
    let Some(J::Obj(jo)) = std::str::from_utf8(body).ok().and_then(tools::parse_json) else {
        return Err(ApiError::new(
            400,
            "invalid_request_error",
            "the body is nested too deeply",
        ));
    };
    let jget = |k: &str| {
        jo.iter()
            .rev()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v)
            .filter(|v| !matches!(v, J::Null))
    };
    let get = |k: &str| o.get(k).filter(|v| !v.is_null());

    if get("background").is_some_and(|b| b != &Value::Bool(false)) {
        return Err(unsupported("background", " (true)"));
    }
    for k in ["conversation", "prompt"] {
        if get(k).is_some() {
            return Err(unsupported(
                if k == "prompt" {
                    "prompt"
                } else {
                    "conversation"
                },
                "",
            ));
        }
    }
    if let Some(t) = get("text") {
        match t.get("format").filter(|f| !f.is_null()) {
            None => {}
            Some(f) if f.get("type").and_then(Value::as_str) == Some("text") => {}
            Some(_) => return Err(unsupported("text", ".format other than text")),
        }
    }
    if get("top_logprobs").is_some_and(|n| n.as_u64() != Some(0)) {
        return Err(unsupported("top_logprobs", " above 0"));
    }
    let store = match get("store") {
        None => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad("store", "'store' must be true or false")),
    };

    // The conversation: a stored one, then this input.
    let mut history = Vec::new();
    if let Some(p) = get("previous_response_id") {
        let Some(id) = p.as_str() else {
            return Err(bad(
                "previous_response_id",
                "'previous_response_id' must be a string",
            ));
        };
        match store_get(id) {
            Some(m) => history = m,
            None => {
                let mut e = bad(
                    "previous_response_id",
                    format!(
                        "previous response '{id}' not found: axon keeps the last {STORE_MAX} \
                         responses made with store true, in memory, until it restarts"
                    ),
                );
                e.code = Some("previous_response_not_found");
                return Err(e);
            }
        }
    }
    match jget("input") {
        None => {}
        Some(J::Str(t)) => history.push(obj(vec![("role", s("user")), ("content", s(t))])),
        Some(J::Arr(a)) => items(a, &mut history)?,
        Some(_) => {
            return Err(bad(
                "input",
                "'input' must be a string or an array of input items",
            ));
        }
    }
    let mut msgs = Vec::with_capacity(history.len() + 1);
    match jget("instructions") {
        None => {}
        Some(J::Str(t)) if t.is_empty() => {}
        Some(J::Str(t)) => msgs.push(obj(vec![("role", s("system")), ("content", s(t))])),
        Some(_) => return Err(bad("instructions", "'instructions' must be a string")),
    }
    if history.is_empty() {
        return Err(bad("input", "'input' is empty"));
    }
    msgs.extend(history.iter().cloned());

    // The chat-completions body.
    let mut chat: Vec<(&str, J)> = vec![("messages", J::Arr(msgs))];
    let mut echo_tools = Vec::new();
    match jget("tools") {
        None => {}
        Some(J::Arr(a)) => {
            let mut ts = Vec::new();
            for (t, raw) in a
                .iter()
                .zip(get("tools").and_then(Value::as_array).into_iter().flatten())
            {
                if let Some(c) = chat_tool(t)? {
                    ts.push(c);
                    echo_tools.push(raw.clone());
                }
            }
            if !ts.is_empty() {
                chat.push(("tools", J::Arr(ts)));
            }
        }
        Some(_) => return Err(bad("tools", "'tools' must be an array")),
    }
    match jget("tool_choice") {
        None => {}
        Some(c @ J::Str(_)) => chat.push(("tool_choice", c.clone())),
        Some(c @ J::Obj(_)) => match (c.get("type").and_then(J::str), c.get("name")) {
            (Some("function"), Some(n @ J::Str(_))) => chat.push((
                "tool_choice",
                obj(vec![
                    ("type", s("function")),
                    ("function", obj(vec![("name", n.clone())])),
                ]),
            )),
            (Some("function"), _) => {
                return Err(bad("tool_choice", "a function tool_choice needs a 'name'"));
            }
            (t, _) => {
                return Err(unsupported(
                    "tool_choice",
                    &format!(" of type '{}'", t.unwrap_or("?")),
                ));
            }
        },
        Some(_) => {
            return Err(bad(
                "tool_choice",
                "'tool_choice' must be auto, none, required or a function",
            ));
        }
    }
    if let Some(Value::Object(r)) = get("reasoning")
        && r.get("effort").and_then(Value::as_str) == Some("none")
    {
        chat.push((
            "chat_template_kwargs",
            obj(vec![("enable_thinking", J::Bool(false))]),
        ));
    }
    if let Some(n) = jget("max_output_tokens") {
        chat.push(("max_completion_tokens", n.clone()));
    }
    // As given (api.rs checks them); the extensions as on chat completions.
    for k in [
        "model",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "repetition_penalty",
        "repeat_penalty",
        "repeat_last_n",
        "presence_penalty",
        "frequency_penalty",
        "stop",
    ] {
        if let Some(v) = jget(k) {
            chat.push((k, v.clone()));
        }
    }
    let g = api::parse(obj(chat).dumps().as_bytes(), true).map_err(|mut e| {
        for (from, to) in [
            ("messages", "input"),
            ("max_completion_tokens", "max_output_tokens"),
        ] {
            if e.param == Some(from) {
                e.param = Some(to);
                e.msg = e.msg.replace(&format!("'{from}'"), &format!("'{to}'"));
            }
        }
        e
    })?;

    let mut echo = Map::new();
    for k in [
        "instructions",
        "max_output_tokens",
        "metadata",
        "parallel_tool_calls",
        "previous_response_id",
        "temperature",
        "top_p",
        "tool_choice",
        "user",
    ] {
        echo.insert(k.into(), get(k).cloned().unwrap_or(Value::Null));
    }
    if echo["metadata"].is_null() {
        echo["metadata"] = json!({});
    }
    if echo["tool_choice"].is_null() {
        echo["tool_choice"] = json!("auto");
    }
    if echo["parallel_tool_calls"].is_null() {
        echo["parallel_tool_calls"] = json!(true);
    }
    echo.insert("tools".into(), Value::Array(echo_tools));
    echo.insert("store".into(), Value::Bool(store));
    echo.insert(
        "reasoning".into(),
        json!({"effort": get("reasoning").and_then(|r| r.get("effort")).cloned(), "summary": null}),
    );
    echo.insert("text".into(), json!({"format": {"type": "text"}}));
    echo.insert("truncation".into(), json!("disabled"));
    Ok((
        g,
        Req {
            echo,
            store,
            history,
        },
    ))
}

/// POST /v1/responses.
pub(super) fn handle(c: &mut Conn<TcpStream>, req: &Request, sh: &Shared) -> io::Result<bool> {
    let arrived = Instant::now();
    let out = |x| super::Out::Responses(Box::new(x));
    if sh.model.get().is_none() {
        return super::api_err(c, super::loading(), req.keep_alive);
    }
    match parse(&req.body) {
        Ok((g, x)) => super::run_request(c, req, sh, g, out(x), arrived),
        Err(e) => super::api_err(c, e, req.keep_alive),
    }
}

/// Of a reply's tokens, its reasoning's: the reasoning text encoded again,
/// plus `<think>` and `</think>` (at most the reply's count). The engine
/// reports text, not tokens per channel, so this is the text's tokens: the
/// count generated unless the model wrote a split the tokenizer wouldn't.
fn reasoning_tokens(tok: &Tokenizer, reasoning: &str, completion: usize) -> usize {
    if reasoning.is_empty() {
        return 0;
    }
    (tok.encode(reasoning).len() + 2).min(completion)
}

fn usage_json((u, reasoning): (&Usage, usize)) -> Value {
    json!({
        "input_tokens": u.prompt_tokens,
        "input_tokens_details": {"cached_tokens": u.cached_tokens},
        "output_tokens": u.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": u.prompt_tokens + u.completion_tokens,
    })
}

/// The response object.
#[allow(clippy::too_many_arguments)]
fn response(
    x: &Req,
    id: &str,
    model: &(&str, String),
    created: u64,
    status: &str,
    output: &[Value],
    // The usage and its reasoning tokens.
    usage: Option<(&Usage, usize)>,
    finish: Option<&str>,
) -> Value {
    let mut r = x.echo.clone();
    r.insert("id".into(), json!(id));
    r.insert("object".into(), json!("response"));
    r.insert("created_at".into(), json!(created));
    r.insert("status".into(), json!(status));
    r.insert("model".into(), json!(model.0));
    r.insert("system_fingerprint".into(), json!(model.1));
    r.insert("output".into(), Value::Array(output.to_vec()));
    r.insert("error".into(), Value::Null);
    r.insert(
        "incomplete_details".into(),
        match finish {
            Some("length") => json!({"reason": "max_output_tokens"}),
            _ => Value::Null,
        },
    );
    r.insert("usage".into(), usage.map_or(Value::Null, usage_json));
    Value::Object(r)
}

/// "completed", or "incomplete" for a reply cut off.
fn status_of(finish: &str) -> &'static str {
    if finish == "length" {
        "incomplete"
    } else {
        "completed"
    }
}

fn item_id(kind: &str, id: &str, n: usize) -> String {
    format!("{kind}_{}{n}", id.trim_start_matches("resp-"))
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({"type": "reasoning", "id": id, "summary": [],
           "content": [{"type": "reasoning_text", "text": text}]})
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({"type": "message", "id": id, "status": status, "role": "assistant",
           "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]})
}

fn call_item(c: &Call, status: &str, args: &str) -> Value {
    json!({"type": "function_call", "id": format!("fc_{}", c.id.trim_start_matches("call_")),
           "call_id": c.id, "name": c.name, "arguments": args, "status": status})
}

/// A reply's text through the tool parser (if any): content, calls.
pub(super) fn split(parser: Option<tools::Parser>, content: String) -> (String, Vec<Call>) {
    let Some(mut p) = parser else {
        return (content, Vec::new());
    };
    let (mut text, mut calls) = (String::new(), Vec::new());
    for piece in p.push(&content).into_iter().chain(p.finish()) {
        match piece {
            Piece::Text(t) => text.push_str(&t),
            Piece::Call(c) => calls.push(c),
        }
    }
    (text, calls)
}

/// Keep the conversation with its reply (`store`).
fn keep(x: &Req, id: &str, content: &str, reasoning: &str, calls: &[Call]) {
    if !x.store {
        return;
    }
    let mut msgs = x.history.clone();
    msgs.push(
        Turn {
            reasoning: reasoning.to_string(),
            content: content.to_string(),
            calls: calls
                .iter()
                .map(|c| {
                    obj(vec![
                        ("id", s(&c.id)),
                        ("type", s("function")),
                        (
                            "function",
                            obj(vec![("name", s(&c.name)), ("arguments", s(&c.args))]),
                        ),
                    ])
                })
                .collect(),
        }
        .msg(),
    );
    store_put(id.to_string(), msgs);
}

fn error_body(m: &str) -> Value {
    json!({"code": "server_error", "message": m})
}

/// A whole (non-streamed) response.
#[allow(clippy::too_many_arguments)]
pub(super) fn whole(
    c: &mut Conn<TcpStream>,
    keep_alive: bool,
    x: &Req,
    tok: &Tokenizer,
    ev: &mut Events,
    id: &str,
    model: (&str, String),
    created: u64,
    parser: Option<tools::Parser>,
) -> io::Result<bool> {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut checked = Instant::now();
    loop {
        match ev.next(c, &mut checked) {
            None => return Ok(false),
            Some(Event::Text {
                content: t,
                reasoning: r,
            }) => {
                content.push_str(&t);
                reasoning.push_str(&r);
            }
            Some(Event::Error(m)) => {
                return super::api_err(c, ApiError::new(500, "server_error", m), keep_alive);
            }
            Some(Event::Done { usage, finish }) => {
                let (text, calls) = split(parser, content);
                let mut out = Vec::new();
                if !reasoning.is_empty() {
                    out.push(reasoning_item(&item_id("rs", id, 0), &reasoning));
                }
                if !text.is_empty() || calls.is_empty() {
                    let st = status_of(finish);
                    out.push(message_item(&item_id("msg", id, 0), &text, st));
                }
                for call in &calls {
                    out.push(call_item(call, "completed", &call.args));
                }
                keep(x, id, &text, &reasoning, &calls);
                let r = response(
                    x,
                    id,
                    &model,
                    created,
                    status_of(finish),
                    &out,
                    Some((
                        &usage,
                        reasoning_tokens(tok, &reasoning, usage.completion_tokens),
                    )),
                    Some(finish),
                );
                return json_reply(c, 200, &r.to_string(), keep_alive);
            }
        }
    }
}

/// Server-sent events with sequence numbers.
struct Sse {
    buf: String,
    seq: u64,
}

impl Sse {
    fn ev(&mut self, ty: &str, v: Value) {
        let Value::Object(mut m) = v else {
            unreachable!("events are objects")
        };
        m.insert("type".into(), json!(ty));
        m.insert("sequence_number".into(), json!(self.seq));
        self.seq += 1;
        self.buf
            .push_str(&format!("event: {ty}\ndata: {}\n\n", Value::Object(m)));
    }
}

/// The item being streamed.
enum Open {
    None,
    Reasoning { id: String, text: String },
    Message { id: String, text: String },
}

/// A streamed response.
#[allow(clippy::too_many_arguments)]
pub(super) fn stream(
    c: &mut Conn<TcpStream>,
    req: &Request,
    x: &Req,
    tok: &Tokenizer,
    ev: &mut Events,
    id: &str,
    model: (&str, String),
    created: u64,
    mut parser: Option<tools::Parser>,
) -> io::Result<bool> {
    let keep_alive = req.keep_alive && req.minor == 1;
    c.start_stream(200, "text/event-stream", req.minor, keep_alive)?;
    let mut sse = Sse {
        buf: String::new(),
        seq: 0,
    };
    let start = response(x, id, &model, created, "in_progress", &[], None, None);
    sse.ev("response.created", json!({"response": start}));
    sse.ev("response.in_progress", json!({"response": start}));
    c.send(sse.buf.as_bytes())?;
    sse.buf.clear();

    let mut out: Vec<Value> = Vec::new();
    let mut open = Open::None;
    let mut calls: Vec<Call> = Vec::new();
    let (mut all_text, mut all_reasoning) = (String::new(), String::new());
    // `status`: the message item's (the response's, at its end).
    let close = |sse: &mut Sse, open: &mut Open, out: &mut Vec<Value>, status: &str| {
        let n = out.len();
        match std::mem::replace(open, Open::None) {
            Open::None => {}
            Open::Reasoning { id: iid, text } => {
                sse.ev(
                    "response.reasoning_text.done",
                    json!({"item_id": iid, "output_index": n, "content_index": 0, "text": text}),
                );
                let item = reasoning_item(&iid, &text);
                sse.ev(
                    "response.output_item.done",
                    json!({"output_index": n, "item": item}),
                );
                out.push(item);
            }
            Open::Message { id: iid, text } => {
                let part =
                    json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []});
                sse.ev(
                    "response.output_text.done",
                    json!({"item_id": iid, "output_index": n, "content_index": 0, "text": text, "logprobs": []}),
                );
                sse.ev(
                    "response.content_part.done",
                    json!({"item_id": iid, "output_index": n, "content_index": 0, "part": part}),
                );
                let item = message_item(&iid, &text, status);
                sse.ev(
                    "response.output_item.done",
                    json!({"output_index": n, "item": item}),
                );
                out.push(item);
            }
        }
    };
    let reasoning_delta = |sse: &mut Sse, open: &mut Open, out: &mut Vec<Value>, r: &str| {
        if !matches!(open, Open::Reasoning { .. }) {
            close(sse, open, out, "completed");
            let iid = item_id("rs", id, out.len());
            sse.ev(
                "response.output_item.added",
                json!({"output_index": out.len(),
                       "item": {"type": "reasoning", "id": iid, "summary": [], "content": []}}),
            );
            *open = Open::Reasoning {
                id: iid,
                text: String::new(),
            };
        }
        if let Open::Reasoning { id: iid, text } = open {
            text.push_str(r);
            sse.ev(
                "response.reasoning_text.delta",
                json!({"item_id": iid, "output_index": out.len(), "content_index": 0, "delta": r}),
            );
        }
    };
    let text_delta = |sse: &mut Sse, open: &mut Open, out: &mut Vec<Value>, t: &str| {
        if !matches!(open, Open::Message { .. }) {
            close(sse, open, out, "completed");
            let iid = item_id("msg", id, out.len());
            let n = out.len();
            sse.ev(
                "response.output_item.added",
                json!({"output_index": n, "item": {"type": "message", "id": iid,
                       "status": "in_progress", "role": "assistant", "content": []}}),
            );
            sse.ev(
                "response.content_part.added",
                json!({"item_id": iid, "output_index": n, "content_index": 0,
                       "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
            );
            *open = Open::Message {
                id: iid,
                text: String::new(),
            };
        }
        if let Open::Message { id: iid, text } = open {
            text.push_str(t);
            sse.ev(
                "response.output_text.delta",
                json!({"item_id": iid, "output_index": out.len(), "content_index": 0,
                       "delta": t, "logprobs": []}),
            );
        }
    };
    let mut checked = Instant::now();
    let mut e = ev.next(c, &mut checked);
    loop {
        let mut done = false;
        while let Some(event) = e {
            match event {
                Event::Text { content, reasoning } => {
                    if !reasoning.is_empty() {
                        all_reasoning.push_str(&reasoning);
                        reasoning_delta(&mut sse, &mut open, &mut out, &reasoning);
                    }
                    let pieces = match parser.as_mut() {
                        Some(p) => p.push(&content),
                        None if content.is_empty() => Vec::new(),
                        None => vec![Piece::Text(content)],
                    };
                    for p in pieces {
                        match p {
                            Piece::Text(t) => {
                                all_text.push_str(&t);
                                text_delta(&mut sse, &mut open, &mut out, &t);
                            }
                            Piece::Call(call) => calls.push(call),
                        }
                    }
                }
                Event::Error(m) => {
                    close(&mut sse, &mut open, &mut out, "incomplete");
                    sse.ev(
                        "error",
                        json!({"code": "server_error", "message": m, "param": null}),
                    );
                    let mut r = response(x, id, &model, created, "failed", &out, None, None);
                    r["error"] = error_body(&m);
                    sse.ev("response.failed", json!({"response": r}));
                    done = true;
                }
                Event::Done { usage, finish } => {
                    for p in parser.as_mut().map(|p| p.finish()).unwrap_or_default() {
                        match p {
                            Piece::Text(t) => {
                                all_text.push_str(&t);
                                text_delta(&mut sse, &mut open, &mut out, &t);
                            }
                            Piece::Call(call) => calls.push(call),
                        }
                    }
                    if calls.is_empty()
                        && !matches!(open, Open::Message { .. })
                        && all_text.is_empty()
                    {
                        // An empty reply is still one message item.
                        text_delta(&mut sse, &mut open, &mut out, "");
                    }
                    close(&mut sse, &mut open, &mut out, status_of(finish));
                    for call in &calls {
                        let n = out.len();
                        let fid = format!("fc_{}", call.id.trim_start_matches("call_"));
                        sse.ev(
                            "response.output_item.added",
                            json!({"output_index": n, "item": call_item(call, "in_progress", "")}),
                        );
                        sse.ev(
                            "response.function_call_arguments.delta",
                            json!({"item_id": fid, "output_index": n, "delta": call.args}),
                        );
                        sse.ev(
                            "response.function_call_arguments.done",
                            json!({"item_id": fid, "output_index": n, "name": call.name,
                                   "arguments": call.args}),
                        );
                        let item = call_item(call, "completed", &call.args);
                        sse.ev(
                            "response.output_item.done",
                            json!({"output_index": n, "item": item}),
                        );
                        out.push(item);
                    }
                    keep(x, id, &all_text, &all_reasoning, &calls);
                    let status = status_of(finish);
                    let r = response(
                        x,
                        id,
                        &model,
                        created,
                        status,
                        &out,
                        Some((
                            &usage,
                            reasoning_tokens(tok, &all_reasoning, usage.completion_tokens),
                        )),
                        Some(finish),
                    );
                    sse.ev(&format!("response.{status}"), json!({"response": r}));
                    done = true;
                }
            }
            if done {
                break;
            }
            e = ev.try_next();
        }
        c.send(sse.buf.as_bytes())?;
        sse.buf.clear();
        if done {
            c.end_stream()?;
            return Ok(keep_alive);
        }
        e = ev.next(c, &mut checked);
        if e.is_none() {
            return Ok(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(v: Value) -> Result<(GenReq, Req), ApiError> {
        parse(v.to_string().as_bytes())
    }

    fn err(v: Value) -> ApiError {
        match p(v) {
            Err(e) => e,
            Ok(_) => panic!("accepted"),
        }
    }

    /// The chat-completions body a Responses body becomes, parsed: equal to
    /// parsing `chat` directly (prompt, tools, sampling).
    fn same(resp: Value, chat: Value) {
        let (g, _) = p(resp).unwrap();
        let c = api::parse(chat.to_string().as_bytes(), true).unwrap();
        assert_eq!(g.prompt, c.prompt);
        // json! sorts keys (tool_key_order_survives checks the order).
        fn sorted(j: &J) -> J {
            match j {
                J::Obj(m) => {
                    let mut m: Vec<_> = m.iter().map(|(k, v)| (k.clone(), sorted(v))).collect();
                    m.sort_by(|a, b| a.0.cmp(&b.0));
                    J::Obj(m)
                }
                J::Arr(a) => J::Arr(a.iter().map(sorted).collect()),
                j => j.clone(),
            }
        }
        let norm = |t: Option<tools::ToolReq>| {
            t.map(|mut t| {
                t.tools = t.tools.iter().map(sorted).collect();
                t.offered = t.offered.iter().map(sorted).collect();
                t
            })
        };
        assert_eq!(norm(g.tools), norm(c.tools));
        assert_eq!(g.max_tokens, c.max_tokens);
        assert_eq!(g.stream, c.stream);
        assert_eq!(format!("{:?}", g.sample), format!("{:?}", c.sample));
    }

    #[test]
    fn translates_to_chat() {
        same(
            json!({"model": "m", "instructions": "Be brief.", "input": "Hi", "temperature": 0,
                   "max_output_tokens": 7, "stream": true}),
            json!({"model": "m", "messages": [{"role": "system", "content": "Be brief."},
                   {"role": "user", "content": "Hi"}], "temperature": 0,
                   "max_completion_tokens": 7, "stream": true}),
        );
        // Tools (key order kept), a call, its output, parts, developer.
        let params = json!({"type": "object", "properties": {"z": {"type": "string"},
                            "a": {"type": "integer"}}, "required": ["z"]});
        same(
            json!({"input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "d"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "a"},
                                             {"type": "input_text", "text": "b"}]},
                {"type": "reasoning", "id": "rs_1", "summary": [],
                 "content": [{"type": "reasoning_text", "text": "think"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
                {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{\"z\":\"q\"}"},
                {"type": "function_call", "call_id": "call_2", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "r1"},
                {"type": "function_call_output", "call_id": "call_2",
                 "output": [{"type": "input_text", "text": "r2"}]}],
                   "tools": [{"type": "function", "name": "f", "description": "F",
                              "parameters": params, "strict": false},
                             {"type": "web_search"},
                             {"type": "namespace", "name": "ns", "tools": []}],
                   "tool_choice": {"type": "function", "name": "f"}}),
            json!({"messages": [
                {"role": "developer", "content": "d"},
                {"role": "user", "content": "a\nb"},
                {"role": "assistant", "content": "ok", "reasoning_content": "think",
                 "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": "f", "arguments": "{\"z\":\"q\"}"}},
                                {"id": "call_2", "type": "function",
                                 "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "r1"},
                {"role": "tool", "tool_call_id": "call_2", "content": "r2"}],
                   "tools": [{"type": "function", "function": {"name": "f", "description": "F",
                              "parameters": params, "strict": false}}],
                   "tool_choice": {"type": "function", "function": {"name": "f"}}}),
        );
        // reasoning effort none: no thinking.
        same(
            json!({"input": "x", "reasoning": {"effort": "none"}}),
            json!({"messages": [{"role": "user", "content": "x"}],
                   "chat_template_kwargs": {"enable_thinking": false}}),
        );
    }

    #[test]
    fn tool_key_order_survives() {
        // serde_json would sort these; the prompt must keep the request's order.
        let body = r#"{"input":"x","tools":[{"type":"function","name":"f","parameters":{"type":"object","properties":{"zz":{"type":"string"},"aa":{"type":"string"}}}}]}"#;
        let (g, _) = parse(body.as_bytes()).unwrap();
        let t = g.tools.unwrap().tools[0].dumps();
        assert!(t.find("zz").unwrap() < t.find("aa").unwrap(), "{t}");
    }

    #[test]
    fn bad_shapes_are_400() {
        for (v, param) in [
            (json!([]), None),
            (json!({}), Some("input")),
            (json!({"input": []}), Some("input")),
            (json!({"input": 5}), Some("input")),
            (json!({"input": [5]}), Some("input")),
            (json!({"input": [{"foo": 1}]}), Some("input")),
            (json!({"input": [{"type": "message"}]}), Some("input")),
            (
                json!({"input": [{"role": "wizard", "content": "x"}]}),
                Some("input"),
            ),
            (
                json!({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]}),
                Some("input"),
            ),
            (
                json!({"input": [{"role": "user", "content": [{"type": "input_text"}]}]}),
                Some("input"),
            ),
            (
                json!({"input": [{"role": "user", "content": 7}]}),
                Some("input"),
            ),
            (
                json!({"input": [{"type": "function_call", "name": "f"}]}),
                Some("input"),
            ),
            (
                json!({"input": [{"type": "function_call", "name": "f", "call_id": "c", "arguments": {}}]}),
                Some("input"),
            ),
            (
                json!({"input": [{"type": "function_call_output", "output": "x"}]}),
                Some("input"),
            ),
            (
                json!({"input": [{"type": "item_reference", "id": "x"}]}),
                Some("input"),
            ),
            (
                json!({"input": "x", "instructions": 3}),
                Some("instructions"),
            ),
            (json!({"input": "x", "tools": {}}), Some("tools")),
            (json!({"input": "x", "tools": [3]}), Some("tools")),
            (
                json!({"input": "x", "tools": [{"name": "f"}]}),
                Some("tools"),
            ),
            (
                json!({"input": "x", "tools": [{"type": "function"}]}),
                Some("tools"),
            ),
            (json!({"input": "x", "tool_choice": 3}), Some("tool_choice")),
            (
                json!({"input": "x", "tool_choice": {"type": "web_search"}}),
                Some("tool_choice"),
            ),
            (
                json!({"input": "x", "tool_choice": {"type": "function"}}),
                Some("tool_choice"),
            ),
            (
                json!({"input": "x", "tool_choice": "required"}),
                Some("tool_choice"),
            ),
            (
                json!({"input": "x", "tools": [{"type": "function", "name": "f"}],
                    "tool_choice": {"type": "function", "name": "g"}}),
                Some("tool_choice"),
            ),
            (
                json!({"input": "x", "max_output_tokens": 0}),
                Some("max_output_tokens"),
            ),
            (
                json!({"input": "x", "max_output_tokens": "a"}),
                Some("max_output_tokens"),
            ),
            (
                json!({"input": "x", "temperature": -1}),
                Some("temperature"),
            ),
            (
                json!({"input": "x", "text": {"format": {"type": "json_schema"}}}),
                Some("text"),
            ),
            (
                json!({"input": "x", "background": true}),
                Some("background"),
            ),
            (
                json!({"input": "x", "conversation": "c"}),
                Some("conversation"),
            ),
            (
                json!({"input": "x", "top_logprobs": 2}),
                Some("top_logprobs"),
            ),
            (json!({"input": "x", "store": "yes"}), Some("store")),
            (
                json!({"input": "x", "previous_response_id": 1}),
                Some("previous_response_id"),
            ),
            (
                json!({"input": "x", "previous_response_id": "resp-nope"}),
                Some("previous_response_id"),
            ),
            (json!({"input": "x", "stream": "yes"}), Some("stream")),
        ] {
            let e = err(v.clone());
            assert_eq!(e.status, 400, "{v}");
            assert_eq!(e.param, param, "{v}: {}", e.msg);
            assert!(!e.msg.contains("max_completion_tokens"), "{}", e.msg);
        }
        for b in [&b"{"[..], b"", b"\xff", b"nul"] {
            assert_eq!(parse(b).err().unwrap().status, 400);
        }
        // Deep nesting: 400, never a stack overflow.
        let deep = format!(
            "{{\"input\":\"x\",\"metadata\":{}{}}}",
            "[".repeat(100_000),
            "]".repeat(100_000)
        );
        assert_eq!(parse(deep.as_bytes()).err().unwrap().status, 400);
    }

    #[test]
    fn previous_response_id_continues() {
        let (_, x) = p(json!({"input": "first", "instructions": "sys"})).unwrap();
        let call = Call {
            id: "call_9".into(),
            name: "f".into(),
            args: "{}".into(),
        };
        keep(
            &x,
            "resp-test-prev",
            "hello",
            "",
            std::slice::from_ref(&call),
        );
        let (g, _) = p(json!({"previous_response_id": "resp-test-prev",
            "input": [{"type": "function_call_output", "call_id": "call_9", "output": "r"}],
            "tools": [{"type": "function", "name": "f"}]}))
        .unwrap();
        let tr = g.tools.unwrap();
        // Instructions aren't carried over; the reply and its call are.
        let roles: Vec<_> = tr.msgs.iter().map(|m| m.role).collect();
        use super::super::tools::TRole;
        assert_eq!(roles, [TRole::User, TRole::Assistant, TRole::Tool]);
        assert_eq!(tr.msgs[1].content, "hello");
        assert_eq!(tr.msgs[1].calls[0].name, "f");
        // store false: not kept.
        let (_, x) = p(json!({"input": "x", "store": false})).unwrap();
        keep(&x, "resp-test-nostore", "a", "", &[]);
        assert!(store_get("resp-test-nostore").is_none());
    }

    #[test]
    fn ids_are_unique_per_process() {
        // Two servers started in the same second share `created`.
        let (a, b) = (fresh_id(), fresh_id());
        assert_ne!(a, b);
        assert!(a.starts_with("resp-"));
        // 128 random bits (32 hex digits) before the sequence number.
        assert!(a.len() > 5 + 32, "{a}");
        assert_ne!(a[5..37], b[5..37], "random parts differ");
    }

    #[test]
    fn store_is_bounded() {
        for i in 0..STORE_MAX + 10 {
            store_put(format!("resp-bound-{i}"), vec![s("x")]);
        }
        assert!(STORE.lock().unwrap().len() <= STORE_MAX);
        assert!(store_get("resp-bound-0").is_none());
        assert!(store_get(&format!("resp-bound-{}", STORE_MAX + 9)).is_some());
    }

    #[test]
    fn response_shapes() {
        let (_, x) =
            p(json!({"input": "x", "max_output_tokens": 5, "metadata": {"a": "b"}})).unwrap();
        let u = Usage {
            prompt_tokens: 10,
            cached_tokens: 4,
            completion_tokens: 5,
            ..Default::default()
        };
        let m = ("m", "fp_x".to_string());
        let r = response(
            &x,
            "resp-1",
            &m,
            7,
            status_of("length"),
            &[],
            Some((&u, 3)),
            Some("length"),
        );
        assert_eq!(r["usage"]["output_tokens_details"]["reasoning_tokens"], 3);
        assert_eq!(r["status"], "incomplete");
        assert_eq!(r["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(r["usage"]["total_tokens"], 15);
        assert_eq!(r["usage"]["input_tokens_details"]["cached_tokens"], 4);
        assert_eq!(r["max_output_tokens"], 5);
        assert_eq!(r["metadata"]["a"], "b");
        assert_eq!(r["object"], "response");
        let r = response(
            &x,
            "resp-1",
            &m,
            7,
            status_of("stop"),
            &[],
            Some((&u, 0)),
            Some("stop"),
        );
        assert_eq!(r["status"], "completed");
        assert!(r["incomplete_details"].is_null());
        let c = Call {
            id: "call_ab".into(),
            name: "f".into(),
            args: "{\"a\":1}".into(),
        };
        let it = call_item(&c, "completed", &c.args);
        assert_eq!(it["id"], "fc_ab");
        assert_eq!(it["call_id"], "call_ab");
        assert_eq!(it["arguments"], "{\"a\":1}");
        let mut sse = Sse {
            buf: String::new(),
            seq: 0,
        };
        sse.ev("response.output_text.delta", json!({"delta": "x"}));
        sse.ev("response.completed", json!({}));
        assert!(
            sse.buf
                .starts_with("event: response.output_text.delta\ndata: {")
        );
        let d: Vec<Value> = sse
            .buf
            .split("\n\n")
            .filter(|e| !e.is_empty())
            .map(|e| serde_json::from_str(e.split_once("data: ").unwrap().1).unwrap())
            .collect();
        assert_eq!(d[1]["sequence_number"], 1);
        assert_eq!(d[1]["type"], "response.completed");
    }
}
