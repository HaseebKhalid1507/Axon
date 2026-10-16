//! The OpenAI-shaped part of `axon serve`: request bodies to generation
//! parameters, response and stream-chunk JSON, stop strings.
//!
//! Sampling parameters map onto sample.rs as follows (anything not given
//! falls back to the server's defaults: `axon serve`'s sampling flags, else
//! `axon chat`'s defaults but a repeat penalty of 1, off; temperature 0 is
//! greedy, as in the CLI):
//!
//! | request field | axon |
//! |---|---|
//! | `temperature` | `--temp` (0 = greedy: argmax, no other setting applies) |
//! | `top_p` | `--top-p` |
//! | `top_k` (extension; -1 or 0 = off) | `--top-k` |
//! | `min_p` (extension) | `--min-p` |
//! | `repetition_penalty` / `repeat_penalty` (extensions) | `--repeat-penalty` (llama.cpp's multiplicative penalty) |
//! | `repeat_last_n` (extension) | `--repeat-last-n` |
//! | `seed` | `--seed` (random per request if absent) |
//! | `presence_penalty`, `frequency_penalty` | only 0: OpenAI's additive penalties over all earlier tokens have no equivalent in axon's sampler (which selects candidates on the device); a non-zero value is refused with 400 rather than silently ignored |
//! | `max_tokens` / `max_completion_tokens` | the generation limit (default: up to the context) |
//! | `stop` (string or up to 16 strings) | stop strings, matched on the output text and not returned |
//! | `n` | 1 only |
//! | `tools`, `tool_choice` (chat) | tool calling: tools.rs (`required` and a named function prefill a call) |
//! | `parallel_tool_calls`, `store`, `reasoning_effort`, `prompt_cache_key`, `user` | accepted and ignored |
//! | `logprobs`, `functions`, `tools` on /v1/completions, `logit_bias`, `response_format` other than text, `echo`, `suffix`, `best_of` > 1 | refused (400) |

use crate::sample::{SampleArgs, SampleCfg};
use serde_json::{Map, Value};

/// A request that can't be served as asked: OpenAI's error shape.
#[derive(Debug, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub msg: String,
    pub kind: &'static str,
    pub param: Option<&'static str>,
    pub code: Option<&'static str>,
}

impl ApiError {
    pub fn bad(param: &'static str, msg: impl Into<String>) -> ApiError {
        ApiError {
            status: 400,
            msg: msg.into(),
            kind: "invalid_request_error",
            param: Some(param),
            code: None,
        }
    }

    pub fn new(status: u16, kind: &'static str, msg: impl Into<String>) -> ApiError {
        ApiError {
            status,
            msg: msg.into(),
            kind,
            param: None,
            code: None,
        }
    }

    pub fn json(&self) -> String {
        serde_json::json!({
            "error": {
                "message": self.msg,
                "type": self.kind,
                "param": self.param,
                "code": self.code,
            }
        })
        .to_string()
    }
}

/// A chat message.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Prompt {
    /// /v1/chat/completions: the messages, and `enable_thinking: false`
    /// (Some(true)) or true (Some(false)) from `chat_template_kwargs`.
    Chat {
        messages: Vec<Message>,
        no_think: Option<bool>,
    },
    /// /v1/completions: raw text, no template (tokenized as `axon gen`
    /// tokenizes its prompt: BOS first if the model adds one).
    Text(String),
}

/// A generation request, parsed.
#[derive(Debug)]
pub struct GenReq {
    pub prompt: Prompt,
    pub model: Option<String>,
    pub stream: bool,
    pub include_usage: bool,
    pub max_tokens: Option<usize>,
    /// The request's sampling fields (None: the server's default).
    pub sample: SampleArgs,
    pub stop: Vec<String>,
    /// Chat with tools or tool turns (None: an ordinary chat, rendered as before).
    pub tools: Option<super::tools::ToolReq>,
}

/// Most stop strings per request.
pub const MAX_STOP: usize = 16;

fn get<'v>(o: &'v Map<String, Value>, k: &str) -> Option<&'v Value> {
    o.get(k).filter(|v| !v.is_null())
}

fn num(o: &Map<String, Value>, k: &'static str, lo: f64, hi: f64) -> Result<Option<f64>, ApiError> {
    match get(o, k) {
        None => Ok(None),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() && x >= lo && x <= hi => Ok(Some(x)),
            _ => Err(ApiError::bad(
                k,
                format!("'{k}' must be a number from {lo} to {hi}"),
            )),
        },
    }
}

fn int(o: &Map<String, Value>, k: &'static str, lo: i64) -> Result<Option<i64>, ApiError> {
    match get(o, k) {
        None => Ok(None),
        Some(v) => match v.as_i64() {
            Some(x) if x >= lo => Ok(Some(x)),
            _ => Err(ApiError::bad(
                k,
                format!("'{k}' must be a whole number of at least {lo}"),
            )),
        },
    }
}

fn boolean(o: &Map<String, Value>, k: &'static str) -> Result<Option<bool>, ApiError> {
    match get(o, k) {
        None => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ApiError::bad(k, format!("'{k}' must be true or false"))),
    }
}

fn unsupported(k: &'static str, what: &str) -> ApiError {
    ApiError::bad(k, format!("'{k}' {what} is not supported by axon"))
}

/// The text of a message's `content`: a string, null (empty), or an array
/// of `{"type": "text", "text": ...}` parts (joined with newlines).
fn content(v: Option<&Value>) -> Result<String, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut texts = Vec::new();
            for p in parts {
                match (p.get("type").and_then(Value::as_str), p.get("text")) {
                    (Some("text"), Some(Value::String(t))) => texts.push(t.as_str()),
                    _ => {
                        return Err(ApiError::bad(
                            "messages",
                            "only text content parts are supported",
                        ));
                    }
                }
            }
            Ok(texts.join("\n"))
        }
        Some(_) => Err(ApiError::bad(
            "messages",
            "a message's content must be a string or an array of text parts",
        )),
    }
}

/// Parse a /v1/chat/completions (`chat`) or /v1/completions body.
pub fn parse(body: &[u8], chat: bool) -> Result<GenReq, ApiError> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::new(400, "invalid_request_error", format!("invalid JSON: {e}")))?;
    let Value::Object(o) = v else {
        return Err(ApiError::new(
            400,
            "invalid_request_error",
            "the body must be a JSON object",
        ));
    };
    let o = &o;

    let mut tools = None;
    let prompt = if chat {
        let Some(Value::Array(msgs)) = get(o, "messages") else {
            return Err(ApiError::bad("messages", "'messages' must be an array"));
        };
        if msgs.is_empty() {
            return Err(ApiError::bad("messages", "'messages' is empty"));
        }
        tools = super::tools::parse(body)?;
        let mut messages = Vec::with_capacity(msgs.len());
        for m in msgs {
            let role = match m.get("role").and_then(Value::as_str) {
                Some("system" | "developer") => Role::System,
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                // Tool results (tools.rs renders them; here: for the size check).
                Some("tool" | "ipython") if tools.is_some() => Role::User,
                Some(r) => {
                    return Err(ApiError::bad(
                        "messages",
                        format!("role '{r}' is not supported (system, user, assistant)"),
                    ));
                }
                None => return Err(ApiError::bad("messages", "a message has no 'role'")),
            };
            if tools.is_none() && m.get("tool_calls").is_some_and(|t| !t.is_null()) {
                return Err(unsupported("messages", "with tool_calls"));
            }
            messages.push(Message {
                role,
                content: content(m.get("content"))?,
            });
        }
        let no_think = match get(o, "chat_template_kwargs") {
            None => None,
            Some(Value::Object(k)) => boolean(k, "enable_thinking")?.map(|t| !t),
            Some(_) => {
                return Err(ApiError::bad(
                    "chat_template_kwargs",
                    "'chat_template_kwargs' must be an object",
                ));
            }
        };
        Prompt::Chat { messages, no_think }
    } else {
        let p = match get(o, "prompt") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => {
                a[0].as_str().unwrap_or("").to_string()
            }
            _ => {
                return Err(ApiError::bad(
                    "prompt",
                    "'prompt' must be a string (or an array of one string)",
                ));
            }
        };
        if boolean(o, "echo")? == Some(true) {
            return Err(unsupported("echo", "true"));
        }
        if get(o, "suffix").is_some_and(|s| s.as_str() != Some("")) {
            return Err(unsupported("suffix", ""));
        }
        if int(o, "best_of", 1)?.is_some_and(|b| b > 1) {
            return Err(unsupported("best_of", "above 1"));
        }
        Prompt::Text(p)
    };

    let model = match get(o, "model") {
        None => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(ApiError::bad("model", "'model' must be a string")),
    };
    let stream = boolean(o, "stream")?.unwrap_or(false);
    let include_usage = match get(o, "stream_options") {
        None => false,
        Some(Value::Object(so)) => boolean(so, "include_usage")?.unwrap_or(false),
        Some(_) => {
            return Err(ApiError::bad(
                "stream_options",
                "'stream_options' must be an object",
            ));
        }
    };
    let max_tokens = match int(o, "max_completion_tokens", 1)? {
        Some(n) => Some(n),
        None => int(o, "max_tokens", 1)?,
    }
    .map(|n| n as usize);

    if int(o, "n", 1)?.is_some_and(|n| n != 1) {
        return Err(unsupported("n", "above 1"));
    }
    for k in ["presence_penalty", "frequency_penalty"] {
        let k: &'static str = k;
        if num(o, k, -2.0, 2.0)?.is_some_and(|x| x != 0.0) {
            return Err(ApiError::bad(
                k,
                format!(
                    "'{k}' other than 0 is not supported by axon; use 'repetition_penalty' \
                     (llama.cpp's multiplicative penalty, with 'repeat_last_n')"
                ),
            ));
        }
    }
    match get(o, "logprobs") {
        None | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => return Err(unsupported("logprobs", "")),
        // /v1/completions: an integer, the number of logprobs (0 = none).
        Some(v) if !chat && v.as_u64() == Some(0) => {}
        Some(_) => return Err(unsupported("logprobs", "")),
    }
    for k in ["tools", "functions"] {
        if k == "tools" && tools.is_some() {
            // tools.rs took them (and refused what isn't a tool).
            continue;
        }
        if get(o, k).is_some_and(|t| t.as_array().is_none_or(|a| !a.is_empty())) {
            return Err(unsupported(
                if k == "tools" { "tools" } else { "functions" },
                "",
            ));
        }
    }
    if get(o, "logit_bias").is_some_and(|b| b.as_object().is_none_or(|m| !m.is_empty())) {
        return Err(unsupported("logit_bias", ""));
    }
    if let Some(rf) = get(o, "response_format")
        && rf.get("type").and_then(Value::as_str) != Some("text")
    {
        return Err(unsupported("response_format", "other than text"));
    }

    let mut sample = SampleArgs {
        temp: num(o, "temperature", 0.0, 100.0)?.map(|x| x as f32),
        top_p: num(o, "top_p", 0.0, 1.0)?.map(|x| x as f32),
        min_p: num(o, "min_p", 0.0, 1.0)?.map(|x| x as f32),
        top_k: int(o, "top_k", -1)?.map(|k| k.max(0) as usize),
        repeat_last_n: int(o, "repeat_last_n", 0)?.map(|n| n as usize),
        seed: match get(o, "seed") {
            None => None,
            // Negative seeds (some clients send -1 for "random") wrap.
            Some(v) => Some(
                v.as_u64()
                    .or_else(|| v.as_i64().map(|i| i as u64))
                    .ok_or_else(|| ApiError::bad("seed", "'seed' must be a whole number"))?,
            ),
        },
        ..SampleArgs::default()
    };
    for k in ["repetition_penalty", "repeat_penalty"] {
        let k: &'static str = k;
        if let Some(p) = num(o, k, 0.0, 100.0)? {
            if p == 0.0 {
                return Err(ApiError::bad(k, format!("'{k}' must be above 0 (1 = off)")));
            }
            sample.repeat_penalty = Some(p as f32);
        }
    }

    let stop = match get(o, "stop") {
        None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) if a.len() <= MAX_STOP && a.iter().all(Value::is_string) => a
            .iter()
            .filter_map(|s| s.as_str().map(str::to_string))
            .collect(),
        Some(_) => {
            return Err(ApiError::bad(
                "stop",
                format!("'stop' must be a string or an array of up to {MAX_STOP} strings"),
            ));
        }
    }
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect();

    Ok(GenReq {
        prompt,
        model,
        stream,
        include_usage,
        max_tokens,
        sample,
        stop,
        tools,
    })
}

/// The settings a request samples with: its own fields over the server's
/// (`base`), then `resolve` (defaults, a random seed); None is greedy.
pub fn sampling(req: &SampleArgs, base: &SampleArgs, default_temp: f32) -> Option<SampleCfg> {
    SampleArgs {
        temp: req.temp.or(base.temp),
        top_k: req.top_k.or(base.top_k),
        top_p: req.top_p.or(base.top_p),
        min_p: req.min_p.or(base.min_p),
        repeat_penalty: req.repeat_penalty.or(base.repeat_penalty),
        repeat_last_n: req.repeat_last_n.or(base.repeat_last_n),
        seed: req.seed.or(base.seed),
    }
    .resolve(default_temp)
}

/// Stop strings over streamed text: text that could still turn out to be
/// the start of a stop string is held back until it can't.
#[derive(Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> StopMatcher {
        StopMatcher {
            stops,
            held: String::new(),
        }
    }

    /// Add text: (text that is now final, whether a stop string ended it;
    /// then nothing more may be pushed).
    pub fn push(&mut self, s: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (s.to_string(), false);
        }
        self.held.push_str(s);
        // The earliest stop string occurrence.
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|t| self.held.find(t.as_str()))
            .min()
        {
            let out = self.held[..at].to_string();
            self.held.clear();
            return (out, true);
        }
        // Hold back the longest tail that is a proper prefix of a stop string.
        let mut keep = 0;
        for t in &self.stops {
            for (i, _) in t.char_indices().skip(1) {
                if i > keep && self.held.ends_with(&t[..i]) {
                    keep = i;
                }
            }
        }
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        (out, false)
    }

    /// End of the output: what was held back is final.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// A JSON string literal.
pub fn quote(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

/// One server-sent event carrying `json`.
pub fn sse(json: &str) -> String {
    format!("data: {json}\n\n")
}

pub const SSE_DONE: &str = "data: [DONE]\n\n";

/// The part of every chunk/response before `choices`.
pub fn head(id: &str, object: &str, created: u64, model: &str) -> String {
    format!(
        "\"id\":{},\"object\":\"{object}\",\"created\":{created},\"model\":{}",
        quote(id),
        quote(model)
    )
}

/// A streamed chat delta (content and/or reasoning; role on the first).
pub fn chat_chunk(
    head: &str,
    role: bool,
    content: &str,
    reasoning: &str,
    finish: Option<&str>,
) -> String {
    let mut delta = String::new();
    if role {
        delta.push_str("\"role\":\"assistant\",");
    }
    if !reasoning.is_empty() {
        delta.push_str(&format!("\"reasoning_content\":{},", quote(reasoning)));
    }
    if !content.is_empty() || role {
        delta.push_str(&format!("\"content\":{},", quote(content)));
    }
    if delta.ends_with(',') {
        delta.pop();
    }
    let finish = finish.map_or("null".into(), quote);
    format!(
        "{{{head},\"choices\":[{{\"index\":0,\"delta\":{{{delta}}},\"logprobs\":null,\"finish_reason\":{finish}}}]}}"
    )
}

/// A chunk JSON object with `"usage":null` added (the chunks before the
/// usage chunk, with `stream_options.include_usage`).
pub fn null_usage(mut j: String) -> String {
    debug_assert!(j.ends_with('}'));
    j.pop();
    j.push_str(",\"usage\":null}");
    j
}

/// A streamed completions chunk.
pub fn text_chunk(head: &str, text: &str, finish: Option<&str>) -> String {
    let finish = finish.map_or("null".into(), quote);
    format!(
        "{{{head},\"choices\":[{{\"index\":0,\"text\":{},\"logprobs\":null,\"finish_reason\":{finish}}}]}}",
        quote(text)
    )
}

/// Token counts and timings of a finished request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Usage {
    pub prompt_tokens: usize,
    /// Of `prompt_tokens`, those whose KV cache was reused from the
    /// request before.
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub prompt_ms: f64,
    pub decode_ms: f64,
}

impl Usage {
    pub fn json(&self) -> String {
        format!(
            "{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{}}}}}",
            self.prompt_tokens,
            self.completion_tokens,
            self.prompt_tokens + self.completion_tokens,
            self.cached_tokens
        )
    }

    /// llama-server's `timings` object (prompt = the tokens actually fed).
    pub fn timings(&self) -> String {
        let fed = self.prompt_tokens - self.cached_tokens;
        let rate = |n: usize, ms: f64| if ms > 0.0 { n as f64 * 1e3 / ms } else { 0.0 };
        format!(
            "{{\"prompt_n\":{fed},\"prompt_ms\":{:.3},\"prompt_per_second\":{:.2},\"predicted_n\":{},\"predicted_ms\":{:.3},\"predicted_per_second\":{:.2}}}",
            self.prompt_ms,
            rate(fed, self.prompt_ms),
            self.completion_tokens,
            self.decode_ms,
            rate(self.completion_tokens, self.decode_ms)
        )
    }
}

/// The trailing usage chunk of a stream (`stream_options.include_usage`).
pub fn usage_chunk(head: &str, u: &Usage) -> String {
    format!(
        "{{{head},\"choices\":[],\"usage\":{},\"timings\":{}}}",
        u.json(),
        u.timings()
    )
}

/// A whole (non-streamed) chat response.
pub fn chat_response(
    head: &str,
    content: &str,
    reasoning: &str,
    finish: &str,
    u: &Usage,
) -> String {
    let reasoning = if reasoning.is_empty() {
        String::new()
    } else {
        format!(",\"reasoning_content\":{}", quote(reasoning))
    };
    format!(
        "{{{head},\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}{reasoning}}},\"logprobs\":null,\"finish_reason\":{}}}],\"usage\":{},\"timings\":{}}}",
        quote(content),
        quote(finish),
        u.json(),
        u.timings()
    )
}

/// A whole (non-streamed) completions response.
pub fn text_response(head: &str, text: &str, finish: &str, u: &Usage) -> String {
    format!(
        "{{{head},\"choices\":[{{\"index\":0,\"text\":{},\"logprobs\":null,\"finish_reason\":{}}}],\"usage\":{},\"timings\":{}}}",
        quote(text),
        quote(finish),
        u.json(),
        u.timings()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(v: Value) -> Result<GenReq, ApiError> {
        parse(v.to_string().as_bytes(), true)
    }

    fn text(v: Value) -> Result<GenReq, ApiError> {
        parse(v.to_string().as_bytes(), false)
    }

    #[test]
    fn chat_request() {
        let r = chat(json!({
            "model": "m", "stream": true, "stream_options": {"include_usage": true},
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "assistant", "content": null},
                {"role": "user", "content": "hi"}
            ],
            "max_tokens": 5, "max_completion_tokens": 7, "stop": "\n\n",
            "chat_template_kwargs": {"enable_thinking": false},
            "n": 1, "logprobs": false, "tools": [], "presence_penalty": 0, "user": "x",
            "top_p": null
        }))
        .unwrap();
        assert_eq!(r.model.as_deref(), Some("m"));
        assert!(r.stream && r.include_usage);
        assert_eq!(r.max_tokens, Some(7));
        assert_eq!(r.stop, ["\n\n"]);
        let m = |role, c: &str| Message {
            role,
            content: c.into(),
        };
        assert_eq!(
            r.prompt,
            Prompt::Chat {
                messages: vec![
                    m(Role::System, "be brief"),
                    m(Role::User, "a\nb"),
                    m(Role::Assistant, ""),
                    m(Role::User, "hi")
                ],
                no_think: Some(true)
            }
        );
        assert_eq!(r.sample.top_p, None);
    }

    #[test]
    fn completion_request() {
        let r = text(json!({"prompt": ["The capital"], "max_tokens": 3, "echo": false})).unwrap();
        assert_eq!(r.prompt, Prompt::Text("The capital".into()));
        assert!(!r.stream);
        assert_eq!(r.max_tokens, Some(3));
        assert!(text(json!({"prompt": ["a", "b"]})).is_err());
        assert!(text(json!({"prompt": [1, 2]})).is_err());
        assert!(text(json!({"prompt": "a", "echo": true})).is_err());
        assert!(text(json!({"prompt": "a", "logprobs": 0})).is_ok());
        assert!(text(json!({"prompt": "a", "logprobs": 2})).is_err());
    }

    #[test]
    fn refused_requests() {
        let m = json!([{"role": "user", "content": "x"}]);
        for (extra, param) in [
            (json!({"n": 2}), "n"),
            (json!({"presence_penalty": 0.5}), "presence_penalty"),
            (json!({"frequency_penalty": -1}), "frequency_penalty"),
            (json!({"logprobs": true}), "logprobs"),
            (json!({"tools": [{"type": "function"}]}), "tools"),
            (json!({"logit_bias": {"1": 2}}), "logit_bias"),
            (
                json!({"response_format": {"type": "json_object"}}),
                "response_format",
            ),
            (json!({"temperature": -1}), "temperature"),
            (json!({"temperature": "hot"}), "temperature"),
            (json!({"top_p": 1.5}), "top_p"),
            (json!({"top_k": -2}), "top_k"),
            (json!({"max_tokens": 0}), "max_tokens"),
            (json!({"repetition_penalty": 0}), "repetition_penalty"),
            (json!({"stop": [1]}), "stop"),
            (json!({"stream": "yes"}), "stream"),
            (json!({"seed": 1.5}), "seed"),
        ] {
            let mut o = extra.as_object().unwrap().clone();
            o.insert("messages".into(), m.clone());
            let e = chat(Value::Object(o)).unwrap_err();
            assert_eq!((e.status, e.param), (400, Some(param)), "{extra}");
        }
        for bad in [
            json!({}),
            json!({"messages": []}),
            // (role "tool" is tool history now: tools.rs.)
            json!({"messages": [{"role": "function", "content": "x"}]}),
            json!({"messages": [{"content": "x"}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}),
            json!({"messages": [{"role": "user", "content": 3}]}),
        ] {
            assert_eq!(
                chat(bad.clone()).unwrap_err().param,
                Some("messages"),
                "{bad}"
            );
        }
        assert_eq!(parse(b"{", true).unwrap_err().status, 400);
        assert_eq!(parse(b"[]", true).unwrap_err().status, 400);
        let e = ApiError::bad("n", "no");
        let v: Value = serde_json::from_str(&e.json()).unwrap();
        assert_eq!(
            v,
            json!({"error": {"message": "no", "type": "invalid_request_error", "param": "n", "code": null}})
        );
    }

    #[test]
    fn sampling_maps_onto_sample_rs() {
        let m = json!([{"role": "user", "content": "x"}]);
        let req = chat(
            json!({"messages": m, "temperature": 0.5, "top_k": -1, "min_p": 0.05,
            "repetition_penalty": 1.3, "repeat_last_n": 32, "seed": 7, "top_p": 0.8}),
        )
        .unwrap();
        let c = sampling(&req.sample, &SampleArgs::default(), 0.8).unwrap();
        assert_eq!(
            c,
            SampleCfg {
                temp: 0.5,
                top_k: 0,
                top_p: 0.8,
                min_p: 0.05,
                repeat_penalty: 1.3,
                repeat_last_n: 32,
                seed: 7
            }
        );
        // Unset fields: the server's flags, then the CLI defaults.
        let base = SampleArgs {
            top_k: Some(20),
            seed: Some(1),
            ..SampleArgs::default()
        };
        let req = chat(json!({"messages": m, "repeat_penalty": 1.0})).unwrap();
        let c = sampling(&req.sample, &base, 0.8).unwrap();
        assert_eq!(
            (
                c.temp,
                c.top_k,
                c.top_p,
                c.min_p,
                c.repeat_penalty,
                c.repeat_last_n,
                c.seed
            ),
            (0.8, 20, 0.9, 0.0, 1.0, 64, 1)
        );
        // Temperature 0: greedy.
        let req = chat(json!({"messages": m, "temperature": 0, "top_k": 5})).unwrap();
        assert_eq!(sampling(&req.sample, &base, 0.8), None);
        // A server started with --temp 0 is greedy unless asked otherwise.
        let base = SampleArgs {
            temp: Some(0.0),
            ..SampleArgs::default()
        };
        let req = chat(json!({"messages": m})).unwrap();
        assert_eq!(sampling(&req.sample, &base, 0.8), None);
        let req = chat(json!({"messages": m, "temperature": 1})).unwrap();
        assert_eq!(sampling(&req.sample, &base, 0.8).unwrap().temp, 1.0);
        let req = chat(json!({"messages": m, "seed": -1})).unwrap();
        assert_eq!(req.sample.seed, Some(u64::MAX));
    }

    #[test]
    fn stop_strings() {
        // Feed `pieces`, collect (text, hit).
        fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
            let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()).collect());
            let mut out = String::new();
            for p in pieces {
                let (s, hit) = m.push(p);
                out.push_str(&s);
                if hit {
                    return (out, true);
                }
            }
            out.push_str(&m.finish());
            (out, false)
        }
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), false));
        assert_eq!(
            run(&["END"], &["abc E", "N", "D tail"]),
            ("abc ".into(), true)
        );
        assert_eq!(run(&["END"], &["abc EN", "x"]), ("abc ENx".into(), false));
        assert_eq!(run(&["END"], &["abc EN"]), ("abc EN".into(), false));
        assert_eq!(run(&["\n\n", "ab"], &["xa", "b\n\n"]), ("x".into(), true));
        assert_eq!(run(&["\n\n", "b\n"], &["xab\n\n"]), ("xa".into(), true));
        assert_eq!(run(&["é!"], &["café", "é", "!"]), ("café".into(), true));
        // Held text comes out as soon as it can't be a stop string.
        let mut m = StopMatcher::new(vec!["STOP".into()]);
        assert_eq!(m.push("hello ST"), ("hello ".into(), false));
        assert_eq!(m.push("A"), ("STA".into(), false));
        assert_eq!(m.push("S"), ("".into(), false));
        assert_eq!(m.finish(), "S");
    }

    #[test]
    fn sse_framing() {
        let h = head("chatcmpl-1", "chat.completion.chunk", 7, "m");
        let first = sse(&chat_chunk(&h, true, "", "", None));
        assert_eq!(
            first,
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n"
        );
        for s in [
            chat_chunk(&h, false, "a\"\n\u{1}é", "", None),
            chat_chunk(&h, false, "", "why", None),
            chat_chunk(&h, false, "", "", Some("stop")),
            text_chunk(&h, "x\ny", Some("length")),
            usage_chunk(&h, &Usage::default()),
        ] {
            // One line of JSON per event: no raw newline can end it early.
            assert!(!s.contains('\n'));
            let v: Value = serde_json::from_str(&s).unwrap();
            assert_eq!(v["id"], "chatcmpl-1");
        }
        let v: Value =
            serde_json::from_str(&chat_chunk(&h, false, "a\"\n\u{1}é", "", None)).unwrap();
        assert_eq!(v["choices"][0]["delta"], json!({"content": "a\"\n\u{1}é"}));
        let v: Value = serde_json::from_str(&chat_chunk(&h, false, "", "why", None)).unwrap();
        assert_eq!(
            v["choices"][0]["delta"],
            json!({"reasoning_content": "why"})
        );
        let v: Value = serde_json::from_str(&chat_chunk(&h, false, "", "", Some("stop"))).unwrap();
        assert_eq!(v["choices"][0]["delta"], json!({}));
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        let u = Usage {
            prompt_tokens: 10,
            cached_tokens: 4,
            completion_tokens: 3,
            prompt_ms: 2.0,
            decode_ms: 3.0,
        };
        let v: Value = serde_json::from_str(&chat_response(&h, "hi", "", "stop", &u)).unwrap();
        assert_eq!(
            v["choices"][0]["message"],
            json!({"role": "assistant", "content": "hi"})
        );
        assert_eq!(
            v["usage"],
            json!({"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
                   "prompt_tokens_details": {"cached_tokens": 4}})
        );
        assert_eq!(v["timings"]["prompt_n"], 6);
        assert_eq!(v["timings"]["predicted_per_second"], 1000.0);
        assert_eq!(SSE_DONE, "data: [DONE]\n\n");
        let v: Value = serde_json::from_str(&null_usage(text_chunk(&h, "x", None))).unwrap();
        assert_eq!(
            (v["usage"].is_null(), v["choices"][0]["text"].as_str()),
            (true, Some("x"))
        );
        assert!(v.as_object().unwrap().contains_key("usage"));
    }
}
