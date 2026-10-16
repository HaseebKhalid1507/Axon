//! `axon serve` end to end: a server on an ephemeral port over the test
//! model, driven over real sockets.

use super::{MAX_CONNS, MAX_QUEUE, Opts, Server, Template, api::Message, api::Role, start};
use crate::lm::{Device, Lm};
use crate::sample::SampleArgs;
use crate::tokenizer::{Tokenizer, Utf8Stream};
use crate::{GenOpts, generate};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

fn server(path: &str, device: Device) -> Server {
    server_with(path, device, |_| {})
}

const ORIGIN: &str = "http://app.test";

fn server_with(path: &str, device: Device, f: impl FnOnce(&mut Opts)) -> Server {
    let mut o = Opts {
        path: path.into(),
        host: "127.0.0.1".into(),
        port: 0,
        device,
        gpu_layers: None,
        kv: crate::kv::KvType::F16,
        ctx: Some(2048),
        max_draft: 0,
        draft_model: None,
        sample: SampleArgs::default(),
        no_think: false,
        think_inline: false,
        alias: Some("alias".into()),
        log: false,
        max_tokens: None,
        cors: Some(ORIGIN.into()),
        max_conns: MAX_CONNS,
        max_queue: MAX_QUEUE,
    };
    f(&mut o);
    start(o).expect("start the server")
}

/// A parsed response: status, headers (names lowercased), body (chunked
/// coding removed).
struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("{e}: {}", self.body))
    }
    fn header(&self, n: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == n)
            .map(|(_, v)| v.as_str())
    }
    /// The `data:` payloads of an SSE body, `[DONE]` included.
    fn events(&self) -> Vec<String> {
        assert!(self.body.ends_with("\n\n"), "{:?}", self.body);
        self.body
            .split("\n\n")
            .filter(|e| !e.is_empty())
            .map(|e| {
                e.strip_prefix("data: ")
                    .unwrap_or_else(|| panic!("not an SSE data line: {e:?}"))
                    .to_string()
            })
            .collect()
    }
}

/// Read one response from `r` (whose bytes may run on past it).
fn read_resp(r: &mut impl Read, buf: &mut Vec<u8>) -> Resp {
    read_resp_to(r, buf, false)
}

/// `read_resp`; `no_body`: the answer to a HEAD request.
fn read_resp_to(r: &mut impl Read, buf: &mut Vec<u8>, no_body: bool) -> Resp {
    fn fill(r: &mut impl Read, buf: &mut Vec<u8>) -> usize {
        let mut t = [0u8; 4096];
        let n = r.read(&mut t).expect("read");
        buf.extend_from_slice(&t[..n]);
        n
    }
    let end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
        assert!(fill(r, buf) > 0, "EOF in head");
    };
    let head = String::from_utf8(buf.drain(..end).collect()).unwrap();
    let mut lines = head.split("\r\n");
    let status = lines.next().unwrap()[9..12].parse().unwrap();
    let headers: Vec<(String, String)> = lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (a, b) = l.split_once(": ").unwrap();
            (a.to_ascii_lowercase(), b.to_string())
        })
        .collect();
    let get = |n: &str| headers.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
    let body = if no_body || status == 204 || status == 100 {
        Vec::new()
    } else if let Some(n) = get("content-length") {
        let n: usize = n.parse().unwrap();
        while buf.len() < n {
            assert!(fill(r, buf) > 0, "EOF in body");
        }
        buf.drain(..n).collect()
    } else if get("transfer-encoding").as_deref() == Some("chunked") {
        let mut body = Vec::new();
        loop {
            let p = loop {
                if let Some(p) = buf.windows(2).position(|w| w == b"\r\n") {
                    break p;
                }
                assert!(fill(r, buf) > 0, "EOF in chunk size");
            };
            let n = usize::from_str_radix(std::str::from_utf8(&buf[..p]).unwrap(), 16).unwrap();
            while buf.len() < p + 2 + n + 2 {
                assert!(fill(r, buf) > 0, "EOF in chunk");
            }
            body.extend_from_slice(&buf[p + 2..p + 2 + n]);
            assert_eq!(&buf[p + 2 + n..p + 4 + n], b"\r\n");
            buf.drain(..p + 4 + n);
            if n == 0 {
                break body;
            }
        }
    } else {
        while fill(r, buf) > 0 {}
        std::mem::take(buf)
    };
    Resp {
        status,
        headers,
        body: String::from_utf8(body).unwrap(),
    }
}

fn request(method: &str, path: &str, body: &str, close: bool) -> String {
    format!(
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}\r\n{body}",
        body.len(),
        if close { "Connection: close\r\n" } else { "" }
    )
}

fn call(addr: SocketAddr, method: &str, path: &str, body: &str) -> Resp {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(300))).unwrap();
    s.write_all(request(method, path, body, true).as_bytes())
        .unwrap();
    let r = read_resp(&mut s, &mut Vec::new());
    assert_eq!(r.header("connection"), Some("close"));
    r
}

fn post(addr: SocketAddr, path: &str, body: Value) -> Resp {
    call(addr, "POST", path, &body.to_string())
}

/// `axon gen`'s greedy text for `prompt` on `net`, as run_lm makes it: the
/// prompt's tokens (`encode_prompt`: BOS if the model adds one), one
/// `prefill_logits`, the end-of-generation stop set (`eog`).
fn gen_text<M: Lm>(net: &M, tok: &Tokenizer, prompt: &str, n: usize) -> String {
    let ids = tok.encode_prompt(prompt);
    let mut cache = net.new_cache(ids.len() + n + 1);
    let logits = net.prefill_logits(&ids, 0, &mut cache);
    let mut text = Utf8Stream::default();
    let mut out = String::new();
    generate(
        net,
        &mut cache,
        &ids,
        logits,
        GenOpts {
            n_gen: n,
            stop: &tok.eog(),
            max_draft: 0,
            sample: None,
        },
        &mut |_, _| Vec::new(),
        &mut |id| {
            out.push_str(&text.push(&tok.decode_bytes(&[id])));
            Ok(())
        },
    )
    .unwrap();
    out.push_str(&text.finish());
    out
}

/// `axon chat --temp 0`'s first reply to `msg` on `net`, at most `n`
/// tokens, with run_chat's default system message for `arch`.
fn chat_text<M: Lm>(net: &M, tok: &Tokenizer, arch: &str, msg: &str, n: usize) -> String {
    let system = (arch == "qwen2").then_some(crate::chat::DEFAULT_SYSTEM);
    let mut c = crate::chat::Chat::new(net, tok, system, 256).unwrap();
    c.max_reply = n;
    let mut text = Utf8Stream::default();
    let mut out = String::new();
    c.reply(msg, &|| false, &mut |id| {
        out.push_str(&text.push(&tok.decode_bytes(&[id])));
        Ok(())
    })
    .unwrap();
    out.push_str(&text.finish());
    out
}

/// Text of a streamed response's events (content, or completions text).
fn streamed_text(r: &Resp, chat: bool) -> (String, Value) {
    let ev = r.events();
    assert_eq!(ev.last().unwrap(), "[DONE]");
    let mut text = String::new();
    let mut last = Value::Null;
    for e in &ev[..ev.len() - 1] {
        let v: Value = serde_json::from_str(e).unwrap();
        if let Some(c) = v["choices"].get(0) {
            let t = if chat {
                &c["delta"]["content"]
            } else {
                &c["text"]
            };
            text.push_str(t.as_str().unwrap_or(""));
        }
        last = v;
    }
    (text, last)
}

const PROMPT: &str = "The capital of France is";
const CHAT_MSG: &str = "Name three colors.";

/// `axon gen`'s and `axon chat`'s greedy output for `path` on `device`
/// (gen_text, chat_text), then `serve_checks` against them.
fn serve_matches_gen_and_chat(path: &str, device: Device) {
    let (gen_ref, chat_ref) = {
        let m = crate::gguf::load_with(path, crate::copy_for(device)).unwrap();
        let tok = Tokenizer::from_gguf(&m);
        let net =
            crate::lm::Backend::new(&m, device, 256, None, crate::kv::KvType::F16, false).unwrap();
        (
            gen_text(&net, &tok, PROMPT, 32),
            chat_text(&net, &tok, &m.arch, CHAT_MSG, 24),
        )
    };
    serve_checks(path, device, &gen_ref, &chat_ref);
}

/// The checks shared by the CPU and GPU runs: greedy completions are
/// `reference` (axon gen's), a greedy chat reply is `chat_ref` (axon chat's).
fn serve_checks(path: &str, device: Device, reference: &str, chat_ref: &str) {
    let s = server(path, device);
    let a = s.addr;

    let r = call(a, "GET", "/v1/models", "");
    assert_eq!(r.status, 200);
    let v = r.json();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][1]["id"], "alias");

    // Greedy completions == axon gen, whole and streamed.
    let req = json!({"model": "alias", "prompt": PROMPT, "max_tokens": 32, "temperature": 0});
    let r = post(a, "/v1/completions", req.clone());
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    assert_eq!(v["choices"][0]["text"], reference, "{device:?}");
    assert_eq!(v["object"], "text_completion");
    let tok = Tokenizer::from_gguf(&crate::gguf::load(path).unwrap());
    let ids = tok.encode_prompt(PROMPT);
    assert_eq!(v["usage"]["prompt_tokens"], ids.len());
    let n_out = v["usage"]["completion_tokens"].as_u64().unwrap();
    let finish = v["choices"][0]["finish_reason"].clone();
    assert_eq!(finish, if n_out == 32 { "length" } else { "stop" });
    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    sreq["stream_options"] = json!({"include_usage": true});
    let r = post(a, "/v1/completions", sreq);
    assert_eq!(r.header("content-type"), Some("text/event-stream"));
    let (text, last) = streamed_text(&r, false);
    assert_eq!(text, reference);
    assert_eq!(last["usage"]["completion_tokens"], n_out);
    // The same prompt again reuses all but its last token.
    assert_eq!(
        last["usage"]["prompt_tokens_details"]["cached_tokens"],
        ids.len() - 1
    );

    // Stop string: the output up to it, finish "stop".
    let cut = reference
        .char_indices()
        .nth(reference.chars().count() / 2)
        .unwrap()
        .0;
    let stop: String = reference[cut..].chars().take(3).collect();
    let mut sreq = req.clone();
    sreq["stop"] = json!(["\u{1}never", stop]);
    let v = post(a, "/v1/completions", sreq).json();
    let want = &reference[..reference.find(&stop).unwrap()];
    assert_eq!(v["choices"][0]["text"], want);
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    // The cache after a stop-string cut: the same prompt again reuses all
    // but its last token and still answers as `gen` does.
    let v = post(a, "/v1/completions", req.clone()).json();
    assert_eq!(v["choices"][0]["text"], reference);
    assert_eq!(
        v["usage"]["prompt_tokens_details"]["cached_tokens"],
        ids.len() - 1
    );

    // Chat: greedy == axon chat, whole == streamed; a follow-up reuses the
    // conversation's cache and answers as a fresh prefill would.
    let msgs = json!([{"role": "user", "content": CHAT_MSG}]);
    let creq = json!({"messages": msgs, "max_tokens": 24, "temperature": 0});
    let v = post(a, "/v1/chat/completions", creq.clone()).json();
    assert_eq!(v["object"], "chat.completion");
    let answer = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(!answer.is_empty());
    assert_eq!(answer, chat_ref, "{device:?}");
    let n_chat = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(
        v["choices"][0]["finish_reason"],
        if n_chat == 24 { "length" } else { "stop" }
    );
    // A short answer ends on the model's end-of-turn token (Llama 3:
    // <|eot_id|>), which never shows in the text.
    let y = post(
        a,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "Reply with just the word: yes"}],
               "max_tokens": 32, "temperature": 0}),
    )
    .json();
    assert_eq!(y["choices"][0]["finish_reason"], "stop", "{y}");
    assert!(
        y["usage"]["completion_tokens"].as_u64().unwrap() < 32,
        "{y}"
    );
    assert!(
        !y["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("<|"),
        "{y}"
    );
    let mut sreq = creq.clone();
    sreq["stream"] = json!(true);
    let r = post(a, "/v1/chat/completions", sreq);
    let ev = r.events();
    let first: Value = serde_json::from_str(&ev[0]).unwrap();
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    let (text, last) = streamed_text(&r, true);
    assert_eq!(text, answer);
    assert_eq!(
        last["choices"][0]["finish_reason"],
        v["choices"][0]["finish_reason"]
    );

    let follow = json!({"messages": [
        {"role": "user", "content": CHAT_MSG},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "And three animals?"}
    ], "max_tokens": 16, "temperature": 0.7, "seed": 42});
    let r1 = post(a, "/v1/chat/completions", follow.clone()).json();
    let cached = r1["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    assert!(cached > 20, "follow-up reused {cached} tokens");
    // Something unrelated in between: the follow-up is then prefilled from scratch.
    post(
        a,
        "/v1/completions",
        json!({"prompt": "Hello", "max_tokens": 2}),
    );
    let r2 = post(a, "/v1/chat/completions", follow).json();
    assert!(
        r2["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            < 5
    );
    assert_eq!(r1["choices"][0]["message"], r2["choices"][0]["message"]);

    // Seeded sampling is reproducible.
    let sreq =
        json!({"prompt": PROMPT, "max_tokens": 16, "temperature": 1.0, "seed": 7, "top_k": 50});
    let x = post(a, "/v1/completions", sreq.clone()).json();
    let y = post(a, "/v1/completions", sreq).json();
    assert_eq!(x["choices"][0]["text"], y["choices"][0]["text"]);

    // A client that leaves mid-stream stops the generation.
    let mut c = TcpStream::connect(a).unwrap();
    let body = json!({"prompt": "Count: 1, 2, 3,", "max_tokens": 1500, "stream": true,
                      "temperature": 0, "repetition_penalty": 1.0})
    .to_string();
    c.write_all(request("POST", "/v1/completions", &body, false).as_bytes())
        .unwrap();
    let mut got = 0;
    let mut b = [0u8; 4096];
    while got < 600 {
        got += c.read(&mut b).unwrap();
    }
    let at = s.generated.load(Ordering::Relaxed);
    drop(c);
    std::thread::sleep(Duration::from_millis(800));
    let n1 = s.generated.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(500));
    let n2 = s.generated.load(Ordering::Relaxed);
    assert_eq!(n1, n2, "still generating after the client left");
    assert!(n1 - at < 1000, "{at} -> {n1}");
    // The cache after a cancel: the same prompt, reused, answers as after
    // an unrelated request (a fresh prefill).
    let count = json!({"prompt": "Count: 1, 2, 3,", "max_tokens": 24, "temperature": 0,
                       "repetition_penalty": 1.0});
    let x = post(a, "/v1/completions", count.clone()).json();
    let np = x["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(x["usage"]["prompt_tokens_details"]["cached_tokens"], np - 1);
    post(
        a,
        "/v1/completions",
        json!({"prompt": "Hello", "max_tokens": 2}),
    );
    let y = post(a, "/v1/completions", count).json();
    assert!(
        y["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            < np - 1
    );
    assert_eq!(x["choices"][0]["text"], y["choices"][0]["text"]);
    // The server goes on.
    let v = post(a, "/v1/completions", req).json();
    assert_eq!(v["choices"][0]["text"], reference);
}

#[test]
fn serve_cpu_matches_gen() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    serve_matches_gen_and_chat(p.to_str().unwrap(), Device::Cpu);
}

#[test]
fn serve_cuda_matches_gen() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    serve_matches_gen_and_chat(p.to_str().unwrap(), Device::Cuda);
}

/// A draft model changes nothing a client sees: completions and chat
/// replies, greedy and sampled with a seed, across requests that reuse the
/// cache, equal a server's without one (1.5B main model, 0.5B drafter).
fn serve_with_draft_model(device: Device) {
    let Some(d) = crate::testutil::model_path() else {
        return;
    };
    let t = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
    if !t.exists() && std::env::var_os("AXON_SKIP_MODEL_TESTS").is_some() {
        return;
    }
    let reqs = [
        (
            "/v1/completions",
            json!({"prompt": PROMPT, "max_tokens": 40, "temperature": 0}),
        ),
        (
            "/v1/completions",
            json!({"prompt": PROMPT, "max_tokens": 40, "temperature": 0.8, "seed": 5}),
        ),
        (
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": CHAT_MSG}], "max_tokens": 40,
                   "temperature": 0}),
        ),
        (
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": CHAT_MSG}], "max_tokens": 40,
                   "temperature": 0.7, "seed": 11}),
        ),
        (
            "/v1/completions",
            json!({"prompt": PROMPT, "max_tokens": 40, "temperature": 0}),
        ),
    ];
    let mut outs = Vec::new();
    for with in [false, true] {
        let s = server_with(t.to_str().unwrap(), device, |o| {
            if with {
                o.draft_model = Some(d.to_str().unwrap().into());
                o.max_draft = 5;
            }
        });
        let mut got = Vec::new();
        for (path, req) in &reqs {
            let mut req = req.clone();
            req["model"] = json!("alias");
            let r = post(s.addr, path, req);
            assert_eq!(r.status, 200, "{}", r.body);
            let v = r.json();
            got.push((
                v["choices"][0].clone(),
                v["usage"]["completion_tokens"].clone(),
            ));
        }
        outs.push(got);
    }
    assert_eq!(outs[0], outs[1], "{device:?}");
}

#[test]
fn serve_cpu_with_draft_model() {
    serve_with_draft_model(Device::Cpu);
}

#[test]
fn serve_cuda_with_draft_model() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    serve_with_draft_model(Device::Cuda);
}

/// Llama 3: completions start with <|begin_of_text|> and stop on the
/// end-of-generation set as `axon gen`; chat is `axon chat`'s template,
/// ending on <|eot_id|>.
#[test]
fn serve_cpu_llama_matches_gen() {
    let Some(p) = crate::testutil::llama_model_path() else {
        return;
    };
    serve_matches_gen_and_chat(p.to_str().unwrap(), Device::Cpu);
}

#[test]
fn serve_cuda_llama_matches_gen() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::llama_model_path() else {
        return;
    };
    serve_matches_gen_and_chat(p.to_str().unwrap(), Device::Cuda);
}

/// The HTTP layer over a real socket: keep-alive, pipelining, errors.
#[test]
fn serve_http_behaviour() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server(p.to_str().unwrap(), Device::Cpu);
    let a = s.addr;
    // Keep-alive: three requests on one connection, the last two pipelined.
    let mut c = TcpStream::connect(a).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    let mut buf = Vec::new();
    c.write_all(request("GET", "/v1/models", "", false).as_bytes())
        .unwrap();
    let r = read_resp(&mut c, &mut buf);
    assert_eq!((r.status, r.header("connection")), (200, None));
    let body = json!({"prompt": "Hi", "max_tokens": 2, "temperature": 0}).to_string();
    let two =
        request("POST", "/v1/completions", &body, false) + &request("GET", "/health", "", true);
    c.write_all(two.as_bytes()).unwrap();
    let r = read_resp(&mut c, &mut buf);
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["usage"]["completion_tokens"], 2);
    let r = read_resp(&mut c, &mut buf);
    assert_eq!((r.status, r.body.as_str()), (200, "{\"status\":\"ok\"}"));
    assert_eq!(
        c.read(&mut [0u8; 1]).unwrap(),
        0,
        "closed after Connection: close"
    );

    // CORS (--cors ORIGIN): preflights, the origin on every response; HEAD;
    // Date; absolute-form targets.
    let mut c = TcpStream::connect(a).unwrap();
    let mut buf = Vec::new();
    c.write_all(b"OPTIONS /v1/chat/completions HTTP/1.1\r\nOrigin: http://app.test\r\nAccess-Control-Request-Method: POST\r\n\r\n")
        .unwrap();
    let r = read_resp(&mut c, &mut buf);
    assert_eq!(r.status, 204);
    assert_eq!(r.header("access-control-allow-origin"), Some(ORIGIN));
    assert_eq!(
        r.header("access-control-allow-methods"),
        Some("POST, OPTIONS")
    );
    assert!(
        r.header("access-control-allow-headers")
            .unwrap()
            .contains("Content-Type")
    );
    c.write_all(b"HEAD /v1/models HTTP/1.1\r\n\r\n").unwrap();
    let h = read_resp_to(&mut c, &mut buf, true);
    c.write_all(b"GET http://localhost/v1/models HTTP/1.1\r\n\r\n")
        .unwrap();
    let g = read_resp(&mut c, &mut buf);
    assert_eq!((h.status, g.status), (200, 200));
    assert_eq!(
        h.header("content-length").unwrap(),
        g.body.len().to_string()
    );
    assert!(h.body.is_empty());
    assert_eq!(g.header("access-control-allow-origin"), Some(ORIGIN));
    assert!(g.header("date").unwrap().ends_with(" GMT"));
    let e = post(a, "/v1/completions", json!({"prompt": "x", "n": 3}));
    assert_eq!(
        (e.status, e.header("access-control-allow-origin")),
        (400, Some(ORIGIN))
    );
    let e = call(a, "DELETE", "/health", "");
    assert_eq!(
        (e.status, e.header("allow")),
        (405, Some("GET, HEAD, OPTIONS"))
    );

    // HTTP/1.0 stream: close-delimited, no chunked coding.
    let mut c = TcpStream::connect(a).unwrap();
    let body = json!({"prompt": "Hi", "max_tokens": 3, "stream": true}).to_string();
    write!(
        c,
        "POST /v1/completions HTTP/1.0\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let r = read_resp(&mut c, &mut Vec::new());
    assert_eq!(
        (r.header("transfer-encoding"), r.header("connection")),
        (None, Some("close"))
    );
    assert_eq!(r.events().last().unwrap(), "[DONE]");

    // Expect: 100-continue over a socket: the 100, then the answer.
    let mut c = TcpStream::connect(a).unwrap();
    let body = json!({"prompt": "Hi", "max_tokens": 1}).to_string();
    write!(
        c,
        "POST /v1/completions HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .unwrap();
    let mut buf = Vec::new();
    assert_eq!(read_resp(&mut c, &mut buf).status, 100);
    c.write_all(body.as_bytes()).unwrap();
    assert_eq!(read_resp(&mut c, &mut buf).status, 200);

    // A prompt too long by its bytes alone is refused before encoding.
    let e = post(
        a,
        "/v1/completions",
        json!({"prompt": "x".repeat(1_000_000)}),
    );
    assert_eq!(e.status, 400);
    let msg = e.json()["error"]["message"].as_str().unwrap().to_string();
    assert!(msg.contains("bytes"), "{msg}");

    // Chunked request body.
    let mut c = TcpStream::connect(a).unwrap();
    let body = json!({"prompt": "Hi", "max_tokens": 1}).to_string();
    let (x, y) = body.split_at(5);
    write!(
        c,
        "POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{x}\r\n{:x}\r\n{y}\r\n0\r\n\r\n",
        x.len(),
        y.len()
    )
    .unwrap();
    let r = read_resp(&mut c, &mut Vec::new());
    assert_eq!(r.status, 200, "{}", r.body);

    // Errors, OpenAI-shaped.
    let e = post(
        a,
        "/v1/completions",
        json!({"model": "gpt-4", "prompt": "x"}),
    );
    assert_eq!(
        (e.status, e.json()["error"]["code"].as_str()),
        (404, Some("model_not_found"))
    );
    let e = call(a, "POST", "/v1/completions", "{nope");
    assert_eq!(e.status, 400);
    assert!(
        e.json()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("JSON")
    );
    let e = post(
        a,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "x"}], "presence_penalty": 1}),
    );
    assert_eq!(
        (e.status, e.json()["error"]["param"].as_str()),
        (400, Some("presence_penalty"))
    );
    let e = call(a, "GET", "/v1/completions", "");
    assert_eq!((e.status, e.header("allow")), (405, Some("POST, OPTIONS")));
    assert_eq!(call(a, "GET", "/nope", "").status, 404);
    let long = "word ".repeat(2100);
    let e = post(a, "/v1/completions", json!({"prompt": long}));
    assert_eq!(
        (e.status, e.json()["error"]["code"].as_str()),
        (400, Some("context_length_exceeded"))
    );
    // Malformed HTTP: an answer, then the connection closes.
    let mut c = TcpStream::connect(a).unwrap();
    c.write_all(b"GET / HTTP/1.1\r\nBad Header\r\n\r\n")
        .unwrap();
    let r = read_resp(&mut c, &mut Vec::new());
    assert_eq!(r.status, 400);
    // max_tokens past the context: cut at the context, finish "length".
    let v = post(
        a,
        "/v1/completions",
        json!({"prompt": "word ".repeat(2040), "temperature": 0, "repetition_penalty": 1.0}),
    )
    .json();
    assert!(v["choices"][0]["finish_reason"].as_str().is_some());
    let u = &v["usage"];
    assert!(
        u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap() <= 2048
    );
}

/// The chat prompt is `axon chat`'s: the same building blocks (frame), the
/// default system message for qwen2, the empty think block for qwen3.
#[test]
fn chat_template_matches_chat() {
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let tok = Tokenizer::from_gguf(&m);
    let t = Template::new(&tok, &m.arch).unwrap();
    let msg = |role, c: &str| Message {
        role,
        content: c.into(),
    };
    // Qwen's stop set: eos, <|im_end|>, <|endoftext|> (eog adds nothing).
    let mut want = vec![
        tok.eos(),
        tok.special_id("<|im_end|>").unwrap(),
        tok.special_id("<|endoftext|>").unwrap(),
    ];
    want.sort_unstable();
    want.dedup();
    assert_eq!(t.stop(), want);
    assert_eq!(tok.eog(), [tok.eos()]);
    assert_eq!(tok.bos(), None);
    let ids = t.render(&tok, &[msg(Role::User, "Hi!")], false);
    assert_eq!(
        tok.decode(&ids),
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n",
            crate::chat::DEFAULT_SYSTEM
        )
    );
    let ids2 = t.render(
        &tok,
        &[
            msg(Role::System, "S"),
            msg(Role::User, "a"),
            msg(Role::Assistant, "b"),
            msg(Role::User, "c"),
        ],
        true,
    );
    assert_eq!(
        tok.decode(&ids2),
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n"
    );

    let Some(m3) = crate::testutil::qwen3_model() else {
        return;
    };
    let tok3 = Tokenizer::from_gguf(&m3);
    let t3 = Template::new(&tok3, &m3.arch).unwrap();
    let msgs = [
        msg(Role::User, "a"),
        msg(Role::Assistant, "<think>\nhmm\n</think>\n\nb"),
        msg(Role::User, "c"),
    ];
    assert_eq!(
        tok3.decode(&t3.render(&tok3, &msgs, false)),
        "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n"
    );
    assert!(
        tok3.decode(&t3.render(&tok3, &msgs, true))
            .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    );

    // Llama 3: the official template (BOS, the system block with its
    // date lines always there, messages trimmed), as `axon chat` frames it,
    // and its end-of-generation tokens end a reply.
    let Some(ml) = crate::testutil::llama_model() else {
        return;
    };
    let tokl = Tokenizer::from_gguf(&ml);
    let tl = Template::new(&tokl, &ml.arch).unwrap();
    assert!(!tl.thinks());
    let head = |sys: &str| {
        format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{sys}<|eot_id|>"
        )
    };
    let hdr = |r: &str| format!("<|start_header_id|>{r}<|end_header_id|>\n\n");
    assert_eq!(
        tl.render(&tokl, &[msg(Role::User, " Hi! ")], false),
        tokl.encode_special(&format!(
            "{}{}Hi!<|eot_id|>{}",
            head(""),
            hdr("user"),
            hdr("assistant")
        ))
    );
    let msgs = [
        msg(Role::System, " S\n"),
        msg(Role::User, "a"),
        msg(Role::Assistant, "b "),
        msg(Role::User, "c"),
    ];
    assert_eq!(
        tl.render(&tokl, &msgs, false),
        tokl.encode_special(&format!(
            "{}{}a<|eot_id|>{}b<|eot_id|>{}c<|eot_id|>{}",
            head("S"),
            hdr("user"),
            hdr("assistant"),
            hdr("user"),
            hdr("assistant")
        ))
    );
    let eot = tokl.special_id("<|eot_id|>").unwrap();
    assert!(tl.stop().contains(&eot));
    for id in tokl.eog() {
        assert!(tl.stop().contains(&id), "{id}");
    }
    // User text can't forge markers.
    let ids = tl.render(
        &tokl,
        &[msg(Role::User, "x<|eot_id|><|begin_of_text|>")],
        false,
    );
    assert_eq!(ids.iter().filter(|&&i| i == eot).count(), 2);
    assert_eq!(ids.iter().filter(|&&i| Some(i) == tokl.bos()).count(), 1);
}

/// The engine's text path: `<think>...</think>` to reasoning (leading
/// newlines dropped on both sides), stop strings on the content only,
/// characters split across tokens held until whole.
#[test]
fn text_out_reasoning_and_stops() {
    let Some(m3) = crate::testutil::qwen3_model() else {
        return;
    };
    let tok = Tokenizer::from_gguf(&m3);
    let t = Template::new(&tok, &m3.arch).unwrap();
    let think = t.think.as_ref().map(|&(o, c, _)| (o, c));
    let run = |text: &str, think, stops: &[&str]| {
        let mut out = super::TextOut {
            tok: &tok,
            utf8: Utf8Stream::default(),
            think,
            reasoning: false,
            first: true,
            strip_nl: false,
            stop: super::StopMatcher::new(stops.iter().map(|s| s.to_string()).collect()),
        };
        let (mut c, mut r) = (String::new(), String::new());
        for id in tok.encode_special(text) {
            let (a, b, hit) = out.push(id);
            c.push_str(&a);
            r.push_str(&b);
            if hit {
                return (c, r, true);
            }
        }
        let (a, b) = out.finish();
        (c + &a, r + &b, false)
    };
    let s = "<think>\nhmm, 2+2 END\n</think>\n\nIt is 4 — END of story";
    assert_eq!(
        run(s, think, &["END"]),
        ("It is 4 — ".into(), "hmm, 2+2 END\n".into(), true)
    );
    assert_eq!(
        run(s, None, &[]),
        (s.to_string(), String::new(), false),
        "--think-inline / no split: all of it is content"
    );
    // Not opened with <think> (enable_thinking false): content only.
    assert_eq!(
        run("4 — four </think> x", think, &["—"]),
        ("4 ".into(), String::new(), true)
    );
}

/// The HTTP layer's own cost over loopback, no model (specs/r4-serve.md's
/// gate: < 0.2 ms per request, < 20 us per SSE chunk): keep-alive round
/// trips through `Conn`, and SSE chunks as `stream` frames them, written
/// one per token. `--ignored --nocapture`.
#[test]
#[ignore]
fn http_overhead() {
    use super::api;
    use super::http::Conn;
    use std::net::TcpListener;
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let a = l.local_addr().unwrap();
    const REQS: usize = 20_000;
    const CHUNKS: usize = 100_000;
    let srv = std::thread::spawn(move || {
        let (s, _) = l.accept().unwrap();
        s.set_nodelay(true).unwrap();
        let mut c = Conn::new(s);
        for _ in 0..REQS {
            let r = c.read_request().unwrap().unwrap();
            assert_eq!(r.path, "/health");
            c.respond(200, "application/json", b"{\"status\":\"ok\"}", true, &[])
                .unwrap();
        }
        let r = c.read_request().unwrap().unwrap();
        c.start_stream(200, "text/event-stream", r.minor, true)
            .unwrap();
        let h = api::head("chatcmpl-1", "chat.completion.chunk", 1, "model");
        let t = Instant::now();
        for i in 0..CHUNKS {
            let j = api::chat_chunk(
                &h,
                false,
                if i % 2 == 0 { " the" } else { " café" },
                "",
                None,
            );
            c.send(api::sse(&j).as_bytes()).unwrap();
        }
        c.send(api::SSE_DONE.as_bytes()).unwrap();
        c.end_stream().unwrap();
        t.elapsed()
    });
    let mut s = TcpStream::connect(a).unwrap();
    s.set_nodelay(true).unwrap();
    let mut buf = Vec::new();
    let req = request("GET", "/health", "", false);
    let mut lat = Vec::with_capacity(REQS);
    for _ in 0..REQS {
        let t = Instant::now();
        s.write_all(req.as_bytes()).unwrap();
        let r = read_resp(&mut s, &mut buf);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
        assert_eq!(r.status, 200);
    }
    lat.sort_by(f64::total_cmp);
    let t = Instant::now();
    s.write_all(request("POST", "/x", "{}", false).as_bytes())
        .unwrap();
    let r = read_resp(&mut s, &mut buf);
    let total = t.elapsed();
    let send = srv.join().unwrap();
    assert_eq!(r.events().len(), CHUNKS + 1);
    eprintln!(
        "keep-alive round trip (client write -> parsed response): p50 {:.1} us, p99 {:.1} us",
        lat[REQS / 2],
        lat[REQS * 99 / 100]
    );
    eprintln!(
        "SSE chunk: format + write {:.2} us each on the server thread; {:.2} us each end to end (client parse included)",
        send.as_secs_f64() * 1e6 / CHUNKS as f64,
        total.as_secs_f64() * 1e6 / CHUNKS as f64
    );
}

/// Limits: a full queue and too many connections are 503s, a client that
/// stops reading its stream neither blocks the engine nor other clients,
/// and the server recovers once they are gone.
#[test]
fn serve_caps_and_slow_reader() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cpu, |o| {
        o.max_conns = 6;
        o.max_queue = 2;
    });
    let a = s.addr;
    let conn = || {
        let c = TcpStream::connect(a).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
        c
    };
    // A: a long stream nobody reads past its first bytes.
    let mut ca = conn();
    let body = json!({"prompt": "Count: 1, 2, 3,", "max_tokens": 3000, "stream": true,
                      "temperature": 0, "repetition_penalty": 1.0})
    .to_string();
    ca.write_all(request("POST", "/v1/completions", &body, false).as_bytes())
        .unwrap();
    let mut b = [0u8; 512];
    assert!(ca.read(&mut b).unwrap() > 0);
    // B, C: queued behind it (the queue holds 2).
    let small = json!({"prompt": "Hi", "max_tokens": 1, "temperature": 0}).to_string();
    let mut cb = conn();
    let mut cc = conn();
    for c in [&mut cb, &mut cc] {
        c.write_all(request("POST", "/v1/completions", &small, false).as_bytes())
            .unwrap();
    }
    std::thread::sleep(Duration::from_millis(500));
    // D: the queue is full.
    let mut cd = conn();
    cd.write_all(request("POST", "/v1/completions", &small, false).as_bytes())
        .unwrap();
    let r = read_resp(&mut cd, &mut Vec::new());
    assert_eq!(r.status, 503, "{}", r.body);
    assert!(r.body.contains("queued"));
    // While A's client reads nothing, the engine goes on and /health answers at once.
    let g0 = s.generated.load(Ordering::Relaxed);
    let mut ce = conn();
    let t = Instant::now();
    ce.write_all(request("GET", "/health", "", false).as_bytes())
        .unwrap();
    assert_eq!(read_resp(&mut ce, &mut Vec::new()).status, 200);
    assert!(
        t.elapsed() < Duration::from_millis(500),
        "{:?}",
        t.elapsed()
    );
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        s.generated.load(Ordering::Relaxed) > g0,
        "the engine stalled"
    );
    // A, B, C, D, E open, F the sixth; G is one too many.
    let _cf = conn();
    std::thread::sleep(Duration::from_millis(100));
    let mut cg = conn();
    let r = read_resp(&mut cg, &mut Vec::new());
    assert_eq!(r.status, 503);
    assert!(r.body.contains("connections"));
    // A leaves; B and C get their answers; everyone leaves; the server recovers.
    drop(ca);
    for c in [&mut cb, &mut cc] {
        assert_eq!(read_resp(c, &mut Vec::new()).status, 200);
    }
    drop((cb, cc, cd, ce, _cf, cg));
    std::thread::sleep(Duration::from_millis(500));
    for _ in 0..8 {
        let v = post(
            a,
            "/v1/completions",
            json!({"prompt": "Hi", "max_tokens": 1}),
        );
        assert_eq!(v.status, 200);
    }
}

/// Qwen3 over HTTP: the reasoning comes as `reasoning_content`, whole or
/// streamed (the same text either way); `enable_thinking: false` has none.
#[test]
fn serve_qwen3_reasoning() {
    if crate::testutil::qwen3_model().is_none() {
        return;
    }
    let path = match std::env::var_os("AXON_TEST_QWEN3") {
        Some(p) => std::path::PathBuf::from(p),
        None => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(crate::testutil::QWEN3_MODEL),
    };
    let s = server(path.to_str().unwrap(), Device::Cpu);
    let a = s.addr;
    // Qwen's recommended thinking-mode sampling, seeded (greedy can loop
    // in the reasoning: it did on the CPU of an sm_89 machine).
    let req = json!({"messages": [{"role": "user", "content": "What is 2+2? One word."}],
                     "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7,
                     "max_tokens": 500});
    let v = post(a, "/v1/chat/completions", req.clone()).json();
    let m = &v["choices"][0]["message"];
    let reasoning = m["reasoning_content"].as_str().unwrap().to_string();
    let content = m["content"].as_str().unwrap().to_string();
    assert!(!reasoning.is_empty(), "{v}");
    assert!(
        !content.contains("think>") && !reasoning.contains("think>"),
        "{v}"
    );
    // A reply that ended (rather than hit max_tokens) has closed its
    // reasoning and answered.
    if v["choices"][0]["finish_reason"] == "stop" {
        assert!(!content.is_empty(), "{v}");
    }

    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    let r = post(a, "/v1/chat/completions", sreq);
    let (mut sr, mut sc) = (String::new(), String::new());
    for e in r.events().iter().filter(|e| *e != "[DONE]") {
        let v: Value = serde_json::from_str(e).unwrap();
        let d = &v["choices"][0]["delta"];
        sr.push_str(d["reasoning_content"].as_str().unwrap_or(""));
        sc.push_str(d["content"].as_str().unwrap_or(""));
    }
    assert_eq!((sr, sc), (reasoning, content));

    let mut nreq = req;
    nreq["chat_template_kwargs"] = json!({"enable_thinking": false});
    let v = post(a, "/v1/chat/completions", nreq).json();
    let m = &v["choices"][0]["message"];
    assert!(m.get("reasoning_content").is_none(), "{v}");
    assert!(!m["content"].as_str().unwrap().is_empty());
}

/// Does prefill cost depend on the cache's size? The server keeps one big
/// cache (--ctx) where `gen` sizes it to the prompt. Times a 560-token
/// prefill (feed_batch + the last forward) into caches of 600, 4096 and
/// 8192 positions, alternating rounds, median and best.
/// `AXON_PROF_MODEL=<gguf>` (default the 0.5B), `--ignored --nocapture`.
#[test]
#[ignore]
fn prefill_vs_cache_size() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let rel = std::env::var("AXON_PROF_MODEL").unwrap_or(crate::testutil::MODEL.into());
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let m = crate::gguf::load_with(path.to_str().unwrap(), Some(false)).unwrap();
    let tok = Tokenizer::from_gguf(&m);
    let net = crate::lm::Backend::new(&m, Device::Cuda, 8192, None, crate::kv::KvType::F16, false)
        .unwrap();
    let sent = "The quick brown fox jumps over the lazy dog near the river bank. ";
    let ctxs = [600usize, 4096, 8192];
    let mut caches: Vec<_> = ctxs.iter().map(|&c| net.new_cache(c)).collect();
    let mut times = vec![Vec::new(); ctxs.len()];
    let pb = net.prefill_batch();
    for round in 0..12 {
        for (i, cache) in caches.iter_mut().enumerate() {
            // Salted: a different first token each time.
            let ids = tok.encode(&format!("{round}{i} {}", sent.repeat(40)));
            let ids = &ids[..560];
            let t = Instant::now();
            let (&last, rest) = ids.split_last().unwrap();
            for (j, b) in rest.chunks(pb).enumerate() {
                net.feed_batch(b, j * pb, cache);
            }
            let l = net.forward(last, rest.len(), cache);
            assert!(!l.is_empty());
            if round >= 2 {
                times[i].push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    for (i, t) in times.iter_mut().enumerate() {
        t.sort_by(f64::total_cmp);
        eprintln!(
            "cache {:5}: 560-token prefill median {:.2} ms, best {:.2} ms",
            ctxs[i],
            t[t.len() / 2],
            t[0]
        );
    }
}
