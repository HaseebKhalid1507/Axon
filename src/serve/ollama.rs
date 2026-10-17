//! Ollama's native API over the chat-completions path, so ollama clients
//! (the `ollama` CLI and libraries, Open WebUI, opencode/pi/crush with an
//! ollama provider, ...) work against `axon serve` unchanged. As with
//! responses.rs, a request is translated into a /v1/chat/completions (or,
//! for raw /api/generate, /v1/completions) body, parsed by api.rs, and run
//! by the same `run_request`; only the answer's shape is this module's.
//! docs/serve-apis.md lists the routes and fields.
//!
//! - `POST /api/chat`: `messages` (`role`, `content`, `thinking`,
//!   `tool_calls` with object arguments, `tool_name`), `tools`, `stream`
//!   (default true: NDJSON lines), `think`, `options` (`temperature`,
//!   `top_p`, `top_k`, `min_p`, `seed`, `num_predict`, `stop`,
//!   `repeat_penalty`, `repeat_last_n`, `presence_penalty`,
//!   `frequency_penalty`; `num_ctx` and the runtime options are accepted
//!   and ignored: the server's --ctx rules). `format` (structured output)
//!   and `images` are refused.
//! - `POST /api/generate`: `prompt` and `system` as a chat (or `raw`: the
//!   text as /v1/completions takes it). `suffix`, `template`, `context`,
//!   `images` and `format` are refused.
//! - An empty /api/chat or /api/generate (no messages, no prompt) is
//!   ollama's "load the model" request: answered at once, `done_reason`
//!   "load".
//! - `GET /api/tags`, `GET /api/ps`, `POST /api/show`, `GET /api/version`,
//!   and `GET /` ("Ollama is running", the CLI's heartbeat) describe the
//!   one model this server runs, under its file name and `--alias`, with
//!   or without `:latest`.
//! - `/api/embed`, `/api/embeddings` and the model-management routes
//!   (`pull`, `push`, `create`, `copy`, `delete`) answer 501 with a reason.
//!
//! Errors are ollama's `{"error": "..."}`.

use super::api::{self, ApiError, GenReq, Usage};
use super::tools::{self, Call, J, Piece};
use super::{Conn, Event, Events, Out, Request, Shared, json_reply};
use serde_json::{Value, json};
use std::io;
use std::net::TcpStream;
use std::time::Instant;

/// The ollama version /api/version reports: clients check for a minimum.
pub const VERSION: &str = "0.14.1";

pub fn error_json(msg: &str) -> String {
    json!({ "error": msg }).to_string()
}

/// The Ollama side of a request: what the answer needs.
pub struct Req {
    /// /api/chat (`message`) or /api/generate (`response`).
    chat: bool,
    /// The model name as asked for (echoed).
    model: Option<String>,
}

fn bad(param: &'static str, msg: impl Into<String>) -> ApiError {
    ApiError::bad(param, msg)
}

fn s(t: &str) -> J {
    J::Str(t.to_string())
}

fn obj(m: Vec<(&str, J)>) -> J {
    J::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// `name` as this server's name for it: ollama's default tag dropped.
fn base_name(name: &str) -> &str {
    name.strip_suffix(":latest").unwrap_or(name)
}

/// The routes this module answers, with their methods.
pub fn methods(path: &str) -> Option<&'static str> {
    Some(match path {
        "/" | "/api/tags" | "/api/ps" | "/api/version" => "GET, HEAD, OPTIONS",
        "/api/chat" | "/api/generate" | "/api/show" | "/api/embed" | "/api/embeddings"
        | "/api/pull" | "/api/push" | "/api/create" | "/api/copy" => "POST, OPTIONS",
        "/api/delete" => "DELETE, OPTIONS",
        _ => return None,
    })
}

/// Whether `method` on `path` is one of this module's routes.
pub fn serves(method: &str, path: &str) -> bool {
    methods(path).is_some_and(|m| {
        let has = |x: &str| m.split(", ").any(|y| y == x);
        method != "OPTIONS" && (has(method) || (method == "HEAD" && has("GET")))
    })
}

fn not_supported(what: &str) -> ApiError {
    ApiError::new(
        501,
        "not_implemented",
        format!("{what} not supported by axon serve"),
    )
}

/// Answer one of `methods`' routes.
pub(super) fn route(c: &mut Conn<TcpStream>, req: &Request, sh: &Shared) -> io::Result<bool> {
    let keep = req.keep_alive;
    let reply =
        |c: &mut Conn<TcpStream>, e: ApiError| json_reply(c, e.status, &error_json(&e.msg), keep);
    match req.path.as_str() {
        "/" => {
            c.respond(
                200,
                "text/plain; charset=utf-8",
                b"Ollama is running",
                keep,
                &[],
            )?;
            return Ok(keep);
        }
        "/api/version" => {
            return json_reply(c, 200, &json!({"version": VERSION}).to_string(), keep);
        }
        "/api/embed" | "/api/embeddings" => {
            return reply(c, not_supported("embeddings are"));
        }
        "/api/pull" | "/api/push" | "/api/create" | "/api/copy" | "/api/delete" => {
            return reply(
                c,
                not_supported(
                    "managing models is: this server runs the one model it started with; it is",
                ),
            );
        }
        _ => {}
    }
    // While the model loads, the four GET routes still answer (`/` and
    // /api/version as ever; /api/tags lists the model from its file;
    // /api/ps lists nothing: no model is loaded yet), as ollama does with
    // a model on disk and none in memory. The rest are 503.
    let loaded = sh.model.get();
    match (req.path.as_str(), loaded) {
        ("/api/tags", _) => {
            let i = Info::of(sh, loaded);
            let models: Vec<Value> = sh.names().iter().map(|n| i.tag(n)).collect();
            return json_reply(c, 200, &json!({ "models": models }).to_string(), keep);
        }
        ("/api/ps", None) => {
            return json_reply(c, 200, &json!({ "models": [] }).to_string(), keep);
        }
        _ => {}
    }
    let arrived = Instant::now();
    let Some(m) = loaded else {
        return reply(c, super::loading());
    };
    let info = || Info::of(sh, Some(m));
    match req.path.as_str() {
        "/api/ps" => {
            let i = info();
            let models: Vec<Value> = sh
                .names()
                .iter()
                .map(|n| {
                    let mut t = i.tag(n);
                    t["expires_at"] = json!("2318-01-01T00:00:00Z");
                    t["size_vram"] = json!(if i.gpu { i.size } else { 0 });
                    t["context_length"] = json!(m.ctx);
                    t
                })
                .collect();
            json_reply(c, 200, &json!({ "models": models }).to_string(), keep)
        }
        "/api/show" => {
            let v: Value = match serde_json::from_slice(&req.body) {
                Ok(Value::Object(o)) => Value::Object(o),
                _ => return reply(c, bad("model", "the body must be a JSON object")),
            };
            // The CLI sends `name` with an empty `model`.
            let name = ["model", "name"]
                .iter()
                .filter_map(|k| v.get(*k).and_then(Value::as_str))
                .find(|n| !n.is_empty());
            let Some(name) = name else {
                return reply(c, bad("model", "'model' is required"));
            };
            if !sh.serves(base_name(name)) {
                return reply(c, not_found(name, sh));
            }
            let thinks = m.template.as_ref().is_some_and(super::Template::thinks);
            json_reply(
                c,
                200,
                &info().show(m.ctx, thinks, m.template.is_some()).to_string(),
                keep,
            )
        }
        "/api/chat" | "/api/generate" => {
            let chat = req.path == "/api/chat";
            let (g, x) = match parse(&req.body, chat) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    // ollama's "load the model": nothing to generate.
                    let name = model_name(&req.body, sh);
                    if !sh.serves(base_name(&name)) {
                        return reply(c, not_found(&name, sh));
                    }
                    let mut r = json!({"model": name, "created_at": rfc3339_now(),
                                       "done": true, "done_reason": "load"});
                    if chat {
                        r["message"] = json!({"role": "assistant", "content": ""});
                    } else {
                        r["response"] = json!("");
                    }
                    return json_reply(c, 200, &r.to_string(), keep);
                }
                Err(e) => return reply(c, e),
            };
            super::run_request(c, req, sh, g, Out::Ollama(Box::new(x)), arrived)
        }
        _ => reply(c, ApiError::new(404, "invalid_request_error", "no route")),
    }
}

fn not_found(name: &str, sh: &Shared) -> ApiError {
    ApiError::new(
        404,
        "invalid_request_error",
        format!(
            "model '{name}' not found; this server has '{}'",
            sh.model_id
        ),
    )
}

fn model_name(body: &[u8], sh: &Shared) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| sh.model_id.clone())
}

/// What /api/tags, /api/ps and /api/show say about the model.
struct Info {
    size: u64,
    digest: String,
    arch: String,
    params: u64,
    quant: String,
    gpu: bool,
    modified: String,
}

impl Info {
    /// `m`: None while the model loads (then only the file's facts).
    fn of(sh: &Shared, m: Option<&super::Loaded>) -> Info {
        let meta = std::fs::metadata(&sh.path).ok();
        let modified = meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or_else(rfc3339_now, |d| rfc3339(d.as_secs(), 0));
        let field = |k: &str| {
            m.and_then(|m| m.identity.fields.iter().find(|(n, _)| *n == k))
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        };
        // The most common quant type ("Q4_K:197 Q6_K:...": the first is the most).
        let types = field("types");
        let quant = types
            .split([' ', ','])
            .filter_map(|t| t.split_once(':'))
            .max_by_key(|(_, n)| n.parse::<u64>().unwrap_or(0))
            .map_or(String::new(), |(t, _)| t.to_string());
        Info {
            size: meta.map_or(0, |m| m.len()),
            digest: field("model"),
            arch: m.map_or(String::new(), |m| m.arch.clone()),
            params: m.map_or(0, |m| m.params),
            quant,
            gpu: field("backend").contains("cuda"),
            modified,
        }
    }

    fn details(&self) -> Value {
        let p = self.params as f64;
        let size = if p >= 1e9 {
            format!("{:.1}B", p / 1e9)
        } else if p > 0.0 {
            format!("{:.0}M", p / 1e6)
        } else {
            // Still loading: not known yet.
            String::new()
        };
        let families: Vec<&str> = Some(self.arch.as_str())
            .filter(|a| !a.is_empty())
            .into_iter()
            .collect();
        json!({"parent_model": "", "format": "gguf", "family": self.arch,
               "families": families, "parameter_size": size,
               "quantization_level": self.quant})
    }

    fn tag(&self, name: &str) -> Value {
        json!({"name": name, "model": name, "modified_at": self.modified, "size": self.size,
               "digest": self.digest, "details": self.details()})
    }

    fn show(&self, ctx: usize, thinks: bool, chat: bool) -> Value {
        let mut caps = vec!["completion"];
        if chat {
            caps.push("tools");
        }
        if thinks {
            caps.push("thinking");
        }
        let mut info = serde_json::Map::new();
        info.insert("general.architecture".into(), json!(self.arch));
        info.insert("general.parameter_count".into(), json!(self.params));
        // The context this server runs (--ctx), not the trained one.
        info.insert(format!("{}.context_length", self.arch), json!(ctx));
        json!({"license": "", "modelfile": "# served by axon serve\n", "parameters": "",
               "template": "", "details": self.details(), "model_info": info,
               "capabilities": caps, "modified_at": self.modified})
    }
}

/// RFC 3339 (UTC) of a Unix time.
pub fn rfc3339(secs: u64, nanos: u32) -> String {
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{nanos:09}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

fn rfc3339_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339(d.as_secs(), d.subsec_nanos())
}

/// A message's content: a string (or null).
fn content(m: &J, param: &'static str) -> Result<J, ApiError> {
    match m.get("content") {
        None | Some(J::Null) => Ok(s("")),
        Some(c @ J::Str(_)) => Ok(c.clone()),
        Some(_) => Err(bad(param, "a message's 'content' must be a string")),
    }
}

fn has_images(m: &J) -> bool {
    matches!(m.get("images"), Some(J::Arr(a)) if !a.is_empty())
}

/// An ollama message as a chat message.
fn message(m: &J) -> Result<J, ApiError> {
    if !matches!(m, J::Obj(_)) {
        return Err(bad("messages", "each message must be an object"));
    }
    if has_images(m) {
        return Err(bad(
            "messages",
            "images are not supported by axon (text only)",
        ));
    }
    let Some(role) = m.get("role").and_then(J::str) else {
        return Err(bad("messages", "a message has no 'role'"));
    };
    let mut out = vec![("role", s(role)), ("content", content(m, "messages")?)];
    if let Some(J::Str(t)) = m.get("thinking") {
        out.push(("reasoning_content", s(t)));
    }
    match m.get("tool_calls") {
        None | Some(J::Null) => {}
        Some(J::Arr(calls)) if !calls.is_empty() => {
            let mut cs = Vec::new();
            for (i, c) in calls.iter().enumerate() {
                let f = c.get("function");
                let Some(name) = f.and_then(|f| f.get("name")).and_then(J::str) else {
                    return Err(bad("messages", "a tool call has no function name"));
                };
                let args = match f.and_then(|f| f.get("arguments")) {
                    None | Some(J::Null) => J::Obj(Vec::new()),
                    Some(a) => a.clone(),
                };
                let id = match c.get("id") {
                    Some(J::Str(id)) => id.clone(),
                    _ => format!("call_{i}"),
                };
                cs.push(obj(vec![
                    ("id", J::Str(id)),
                    ("type", s("function")),
                    (
                        "function",
                        obj(vec![("name", s(name)), ("arguments", args)]),
                    ),
                ]));
            }
            out.push(("tool_calls", J::Arr(cs)));
        }
        Some(J::Arr(_)) => {}
        Some(_) => return Err(bad("messages", "'tool_calls' must be an array")),
    }
    Ok(obj(out))
}

/// `options` (and `think`) as chat-completions fields.
fn options(o: &[(String, J)], chat: &mut Vec<(&'static str, J)>) -> Result<(), ApiError> {
    let get = |k: &str| {
        o.iter()
            .rev()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v)
            .filter(|v| !matches!(v, J::Null))
    };
    let Some(opts) = get("options") else {
        return Ok(());
    };
    let J::Obj(_) = opts else {
        return Err(bad("options", "'options' must be an object"));
    };
    for (from, to) in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("seed", "seed"),
        ("repeat_penalty", "repeat_penalty"),
        ("repeat_last_n", "repeat_last_n"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("stop", "stop"),
    ] {
        if let Some(v) = opts.get(from).filter(|v| !matches!(v, J::Null)) {
            chat.push((to, v.clone()));
        }
    }
    match opts.get("num_predict") {
        None | Some(J::Null) => {}
        // -1: no limit; -2: fill the context. Both: up to the context.
        Some(J::Num(n)) if n.trim_start().starts_with('-') => {}
        Some(J::Num(n)) if n == "0" => {
            return Err(bad(
                "options",
                "'num_predict' 0 generates nothing; use -1 for no limit",
            ));
        }
        Some(n @ J::Num(_)) => chat.push(("max_tokens", n.clone())),
        Some(_) => return Err(bad("options", "'num_predict' must be a whole number")),
    }
    Ok(())
}

/// Parse an /api/chat (`chat`) or /api/generate body: None for ollama's
/// "load the model" request (no messages, no prompt).
pub fn parse(body: &[u8], chat: bool) -> Result<Option<(GenReq, Req)>, ApiError> {
    let v: Value =
        serde_json::from_slice(body).map_err(|e| bad("body", format!("invalid JSON: {e}")))?;
    let Value::Object(_) = v else {
        return Err(bad("body", "the body must be a JSON object"));
    };
    let Some(J::Obj(o)) = std::str::from_utf8(body).ok().and_then(tools::parse_json) else {
        return Err(bad("body", "the body is nested too deeply"));
    };
    let get = |k: &str| {
        o.iter()
            .rev()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v)
            .filter(|v| !matches!(v, J::Null))
    };
    let model = match get("model") {
        None => None,
        Some(J::Str(m)) => Some(m.clone()),
        Some(_) => return Err(bad("model", "'model' must be a string")),
    };
    match get("format") {
        None => {}
        Some(J::Str(f)) if f.is_empty() => {}
        Some(_) => {
            return Err(bad(
                "format",
                "'format' (structured output) is not supported by axon",
            ));
        }
    }
    let mut body: Vec<(&'static str, J)> = Vec::new();
    if let Some(m) = &model {
        body.push(("model", J::Str(base_name(m).to_string())));
    }
    let mut text = false;
    if chat {
        let msgs = match get("messages") {
            None => Vec::new(),
            Some(J::Arr(a)) => a.iter().map(message).collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(bad("messages", "'messages' must be an array")),
        };
        if msgs.is_empty() {
            return Ok(None);
        }
        body.push(("messages", J::Arr(msgs)));
        match get("tools") {
            None => {}
            Some(t @ J::Arr(_)) => body.push(("tools", t.clone())),
            Some(_) => return Err(bad("tools", "'tools' must be an array")),
        }
    } else {
        for k in ["suffix", "template"] {
            if get(k).is_some_and(|v| v.str() != Some("")) {
                return Err(bad(
                    if k == "suffix" { "suffix" } else { "template" },
                    format!("'{k}' is not supported by axon"),
                ));
            }
        }
        if matches!(get("context"), Some(J::Arr(a)) if !a.is_empty()) {
            return Err(bad(
                "context",
                "'context' is not supported by axon; send the conversation to /api/chat",
            ));
        }
        if get("images").is_some_and(|i| matches!(i, J::Arr(a) if !a.is_empty())) {
            return Err(bad(
                "images",
                "images are not supported by axon (text only)",
            ));
        }
        let prompt = match get("prompt") {
            None => String::new(),
            Some(J::Str(p)) => p.clone(),
            Some(_) => return Err(bad("prompt", "'prompt' must be a string")),
        };
        if prompt.is_empty() {
            return Ok(None);
        }
        let raw = match get("raw") {
            None | Some(J::Bool(false)) => false,
            Some(J::Bool(true)) => true,
            Some(_) => return Err(bad("raw", "'raw' must be true or false")),
        };
        if raw {
            text = true;
            body.push(("prompt", J::Str(prompt)));
        } else {
            let mut msgs = Vec::new();
            match get("system") {
                None => {}
                Some(J::Str(sys)) if sys.is_empty() => {}
                Some(J::Str(sys)) => {
                    msgs.push(obj(vec![("role", s("system")), ("content", s(sys))]))
                }
                Some(_) => return Err(bad("system", "'system' must be a string")),
            }
            msgs.push(obj(vec![("role", s("user")), ("content", J::Str(prompt))]));
            body.push(("messages", J::Arr(msgs)));
        }
    }
    let stream = match get("stream") {
        None => true,
        Some(J::Bool(b)) => *b,
        Some(_) => return Err(bad("stream", "'stream' must be true or false")),
    };
    body.push(("stream", J::Bool(stream)));
    match get("think") {
        None => {}
        // true or a level ("high", ...): think (the model's default).
        Some(J::Bool(true) | J::Str(_)) if !text => body.push((
            "chat_template_kwargs",
            obj(vec![("enable_thinking", J::Bool(true))]),
        )),
        Some(J::Bool(false)) if !text => body.push((
            "chat_template_kwargs",
            obj(vec![("enable_thinking", J::Bool(false))]),
        )),
        Some(J::Bool(_) | J::Str(_)) => {}
        Some(_) => return Err(bad("think", "'think' must be true, false or a level")),
    }
    options(&o, &mut body)?;
    let g = api::parse(obj(body).dumps().as_bytes(), !text).map_err(|mut e| {
        // The sampling fields live in `options` here.
        if let Some(p) = e.param
            && !["messages", "prompt", "model", "tools", "tool_choice"].contains(&p)
        {
            e.msg = format!("options: {}", e.msg);
            e.param = Some("options");
        }
        e
    })?;
    Ok(Some((g, Req { chat, model })))
}

/// ollama's counts and durations (nanoseconds) of a finished request.
fn stats(r: &mut Value, u: &Usage, arrived: Instant) {
    r["total_duration"] = json!(arrived.elapsed().as_nanos() as u64);
    r["load_duration"] = json!(0);
    r["prompt_eval_count"] = json!(u.prompt_tokens);
    r["prompt_eval_duration"] = json!((u.prompt_ms * 1e6) as u64);
    r["eval_count"] = json!(u.completion_tokens);
    r["eval_duration"] = json!((u.decode_ms * 1e6) as u64);
}

fn calls_json(calls: &[Call]) -> Value {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            // ollama's arguments are an object.
            let args = match serde_json::from_str::<Value>(&c.args) {
                Ok(v @ Value::Object(_)) => v,
                _ => json!({}),
            };
            json!({"id": c.id, "function": {"index": i, "name": c.name, "arguments": args}})
        })
        .collect()
}

/// One NDJSON line / object: a piece of the reply (not done).
fn line(x: &Req, model: &(&str, String), content: &str, thinking: &str, calls: &[Call]) -> Value {
    let name = x.model.as_deref().unwrap_or(model.0);
    let mut r = json!({"model": name, "created_at": rfc3339_now(), "done": false});
    if x.chat {
        let mut m = json!({"role": "assistant", "content": content});
        if !thinking.is_empty() {
            m["thinking"] = json!(thinking);
        }
        if !calls.is_empty() {
            m["tool_calls"] = calls_json(calls);
        }
        r["message"] = m;
    } else {
        r["response"] = json!(content);
        if !thinking.is_empty() {
            r["thinking"] = json!(thinking);
        }
    }
    r
}

fn done_reason(finish: &str) -> &'static str {
    if finish == "length" { "length" } else { "stop" }
}

/// A whole (`stream: false`) answer.
pub(super) fn whole(
    c: &mut Conn<TcpStream>,
    keep: bool,
    x: &Req,
    ev: &mut Events,
    model: (&str, String),
    arrived: Instant,
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
            Some(Event::Error(m)) => return json_reply(c, 500, &error_json(&m), keep),
            Some(Event::Done { usage, finish }) => {
                let (text, calls) = super::responses::split(parser, content);
                let mut r = line(x, &model, &text, &reasoning, &calls);
                r["done"] = json!(true);
                r["done_reason"] = json!(done_reason(finish));
                stats(&mut r, &usage, arrived);
                return json_reply(c, 200, &r.to_string(), keep);
            }
        }
    }
}

/// A streamed answer: one JSON object per line.
pub(super) fn stream(
    c: &mut Conn<TcpStream>,
    req: &Request,
    x: &Req,
    ev: &mut Events,
    model: (&str, String),
    arrived: Instant,
    mut parser: Option<tools::Parser>,
) -> io::Result<bool> {
    let keep = req.keep_alive && req.minor == 1;
    c.start_stream(200, "application/x-ndjson", req.minor, keep)?;
    let mut buf = String::new();
    let push = |buf: &mut String, v: Value| {
        buf.push_str(&v.to_string());
        buf.push('\n');
    };
    let pieces = |buf: &mut String, ps: Vec<Piece>| {
        for p in ps {
            let v = match p {
                Piece::Text(t) => line(x, &model, &t, "", &[]),
                Piece::Call(call) => line(x, &model, "", "", &[call]),
            };
            push(buf, v);
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
                        push(&mut buf, line(x, &model, "", &reasoning, &[]));
                    }
                    match parser.as_mut() {
                        Some(p) => pieces(&mut buf, p.push(&content)),
                        None if content.is_empty() => {}
                        None => pieces(&mut buf, vec![Piece::Text(content)]),
                    }
                }
                Event::Error(m) => {
                    push(&mut buf, json!({ "error": m }));
                    done = true;
                }
                Event::Done { usage, finish } => {
                    if let Some(p) = parser.as_mut() {
                        pieces(&mut buf, p.finish());
                    }
                    let mut r = line(x, &model, "", "", &[]);
                    r["done"] = json!(true);
                    r["done_reason"] = json!(done_reason(finish));
                    stats(&mut r, &usage, arrived);
                    push(&mut buf, r);
                    done = true;
                }
            }
            if done {
                break;
            }
            e = ev.try_next();
        }
        c.send(buf.as_bytes())?;
        buf.clear();
        if done {
            c.end_stream()?;
            return Ok(keep);
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

    fn p(v: Value, chat: bool) -> Result<Option<(GenReq, Req)>, ApiError> {
        parse(v.to_string().as_bytes(), chat)
    }

    fn same(v: Value, chat_body: Value) {
        let (g, _) = p(v, true).unwrap().unwrap();
        let c = api::parse(chat_body.to_string().as_bytes(), true).unwrap();
        assert_eq!(g.prompt, c.prompt);
        assert_eq!(g.tools, c.tools);
        assert_eq!(g.max_tokens, c.max_tokens);
        assert_eq!(g.stream, c.stream);
        assert_eq!(g.stop, c.stop);
        assert_eq!(format!("{:?}", g.sample), format!("{:?}", c.sample));
    }

    #[test]
    fn chat_translates() {
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters":
            {"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "string"}}}}}]);
        same(
            json!({"model": "m:latest", "messages": [{"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "", "thinking": "t",
                 "tool_calls": [{"function": {"name": "f", "arguments": {"b": "1"}}}]},
                {"role": "tool", "content": "r", "tool_name": "f"}],
                   "tools": tools, "think": false,
                   "options": {"temperature": 0, "num_predict": 9, "seed": 3, "top_k": 5,
                               "stop": ["x"], "num_ctx": 4096, "num_gpu": 99, "mirostat": 0}}),
            json!({"model": "m", "messages": [{"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "", "reasoning_content": "t",
                 "tool_calls": [{"id": "call_0", "type": "function",
                                 "function": {"name": "f", "arguments": {"b": "1"}}}]},
                {"role": "tool", "content": "r"}],
                   "tools": tools, "stream": true,
                   "chat_template_kwargs": {"enable_thinking": false},
                   "temperature": 0, "max_tokens": 9, "seed": 3, "top_k": 5, "stop": ["x"]}),
        );
        // stream defaults to true; false is a whole answer.
        let (g, x) = p(
            json!({"messages": [{"role": "user", "content": "u"}], "stream": false}),
            true,
        )
        .unwrap()
        .unwrap();
        assert!(!g.stream && x.chat && x.model.is_none());
        // num_predict -1 / -2: no limit.
        for n in [-1, -2] {
            let (g, _) = p(
                json!({"messages": [{"role": "user", "content": "u"}],
                                  "options": {"num_predict": n}}),
                true,
            )
            .unwrap()
            .unwrap();
            assert_eq!(g.max_tokens, None);
        }
    }

    #[test]
    fn generate_translates() {
        let (g, x) = p(json!({"model": "m", "prompt": "hi", "system": "s"}), false)
            .unwrap()
            .unwrap();
        assert!(!x.chat);
        let c = api::parse(
            json!({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]})
                .to_string()
                .as_bytes(),
            true,
        )
        .unwrap();
        assert_eq!(g.prompt, c.prompt);
        let (g, _) = p(json!({"prompt": "raw text", "raw": true}), false)
            .unwrap()
            .unwrap();
        assert_eq!(g.prompt, api::Prompt::Text("raw text".into()));
        // Empty prompt / messages: load only.
        assert!(p(json!({"model": "m"}), false).unwrap().is_none());
        assert!(
            p(json!({"model": "m", "prompt": ""}), false)
                .unwrap()
                .is_none()
        );
        assert!(
            p(json!({"model": "m", "messages": []}), true)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn bad_shapes_are_400() {
        for (v, chat) in [
            (json!([]), true),
            (json!({"messages": {}}), true),
            (json!({"messages": [1]}), true),
            (json!({"messages": [{"content": "x"}]}), true),
            (json!({"messages": [{"role": "user", "content": 5}]}), true),
            (
                json!({"messages": [{"role": "user", "content": "x", "images": ["aGk="]}]}),
                true,
            ),
            (
                json!({"messages": [{"role": "wizard", "content": "x"}]}),
                true,
            ),
            (
                json!({"messages": [{"role": "assistant", "tool_calls": [{"function": {}}]}]}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "format": "json"}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "format": {"type": "object"}}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "tools": {}}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "stream": 1}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "think": 1}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "options": 1}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "options": {"num_predict": 0}}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "options": {"num_predict": "a"}}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "options": {"temperature": "hot"}}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "options": {"top_p": 7}}),
                true,
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "model": 3}),
                true,
            ),
            (json!({"prompt": 3}), false),
            (json!({"prompt": "x", "suffix": "y"}), false),
            (json!({"prompt": "x", "template": "y"}), false),
            (json!({"prompt": "x", "context": [1, 2]}), false),
            (json!({"prompt": "x", "images": ["aGk="]}), false),
            (json!({"prompt": "x", "raw": "yes"}), false),
            (json!({"prompt": "x", "system": 1}), false),
        ] {
            match p(v.clone(), chat) {
                Err(e) => assert_eq!(e.status, 400, "{v}"),
                Ok(_) => panic!("accepted {v}"),
            }
        }
        for b in [&b"{"[..], b"", b"\xff"] {
            assert!(parse(b, true).is_err());
        }
        let deep = format!(
            "{{\"messages\":{}{}}}",
            "[".repeat(100_000),
            "]".repeat(100_000)
        );
        assert!(parse(deep.as_bytes(), true).is_err());
    }

    #[test]
    fn shapes() {
        assert_eq!(rfc3339(0, 0), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(rfc3339(1_792_270_920, 5), "2026-10-17T21:02:00.000000005Z");
        assert_eq!(rfc3339(951_782_400, 0), "2000-02-29T00:00:00.000000000Z");
        let x = Req {
            chat: true,
            model: Some("m:latest".into()),
        };
        let m = ("m", "fp".to_string());
        let c = Call {
            id: "call_1".into(),
            name: "f".into(),
            args: "{\"a\":1}".into(),
        };
        let l = line(&x, &m, "", "", &[c]);
        assert_eq!(l["model"], "m:latest");
        assert_eq!(
            l["message"]["tool_calls"][0]["function"]["arguments"]["a"],
            1
        );
        let x = Req {
            chat: false,
            model: None,
        };
        let l = line(&x, &m, "hi", "t", &[]);
        assert_eq!(l["response"], "hi");
        assert_eq!(l["thinking"], "t");
        assert_eq!(l["model"], "m");
        assert!(serves("GET", "/api/tags") && serves("HEAD", "/"));
        assert!(serves("POST", "/api/chat") && !serves("GET", "/api/chat"));
        assert!(!serves("OPTIONS", "/api/chat") && !serves("GET", "/api/nope"));
        // Exact methods, not substrings of the list.
        assert!(!serves("GE", "/api/tags") && !serves("POS", "/api/chat") && !serves("T", "/"));
        assert!(!serves("ET, HEAD", "/api/tags"));
        assert_eq!(base_name("a:latest"), "a");
        assert_eq!(base_name("a:7b"), "a:7b");
    }
}
