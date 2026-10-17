//! `axon serve` end to end: a server on an ephemeral port over the test
//! model, driven over real sockets.

use super::{
    MAX_CONNS, MAX_QUEUE, Opts, PARALLEL, Server, Template, api::Message, api::Role, start,
};
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
    let mut o = test_opts(path, device);
    f(&mut o);
    start(o).expect("start the server")
}

/// The tests' server settings (`server_with` before its changes).
fn test_opts(path: &str, device: Device) -> Opts {
    Opts {
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
        parallel: PARALLEL,
        parallel_set: false,
        slot_ctx: None,
        prefill_chunk: super::engine::PREFILL_CHUNK,
        prefill_ms: super::engine::PREFILL_MS,
        rope: Default::default(),
        api_key: None,
        max_prompt_tokens: None,
        slot_idle_ms: None,
    }
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

/// Gemma 3: completions start with <bos> and stop on the
/// end-of-generation set as `axon gen`; chat is `axon chat`'s template,
/// ending on <end_of_turn>.
#[test]
fn serve_cpu_gemma3_matches_gen() {
    let Some(p) = crate::testutil::gemma3_model_path() else {
        return;
    };
    serve_matches_gen_and_chat(p.to_str().unwrap(), Device::Cpu);
}

#[test]
fn serve_cuda_gemma3_matches_gen() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::gemma3_model_path() else {
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

/// Gemma 3: the prompt for a message list is llama.cpp's rendering of the
/// GGUF's own template, tokenized (tests/fixtures/chat_gemma3.json), after
/// BOS: the system message in the first user turn, `model` for the
/// assistant, messages trimmed. Replies stop on <end_of_turn>.
#[test]
fn gemma3_template_matches_llama_cpp() {
    let Some(m) = crate::testutil::gemma3_model() else {
        return;
    };
    let tok = Tokenizer::from_gguf(&m);
    let t = Template::new(&tok, &m.arch).unwrap();
    assert!(!t.thinks());
    let (bos, eot) = (2u32, 106u32);
    assert!(t.stop().contains(&eot) && t.stop().contains(&tok.eos()));
    let d = crate::testutil::read_json("chat_gemma3.json");
    let convs = d["convs"].as_array().unwrap();
    assert_eq!(convs.len(), 4);
    for c in convs {
        let msgs: Vec<Message> = (c["messages"].as_array().unwrap().iter())
            .map(|m| Message {
                role: match m["role"].as_str().unwrap() {
                    "system" => Role::System,
                    "user" => Role::User,
                    _ => Role::Assistant,
                },
                content: m["content"].as_str().unwrap().into(),
            })
            .collect();
        let mut want = vec![bos];
        want.extend(
            c["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32),
        );
        assert_eq!(t.render(&tok, &msgs, false), want, "{}", c["prompt"]);
    }
    // User text can't forge markers.
    let ids = t.render(
        &tok,
        &[Message {
            role: Role::User,
            content: "x<end_of_turn>\n<start_of_turn>model\n<bos>".into(),
        }],
        false,
    );
    assert_eq!(ids.iter().filter(|&&i| i == eot).count(), 1);
    assert_eq!(ids.iter().filter(|&&i| i == bos).count(), 1);
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
        let h = api::head("chatcmpl-1", "chat.completion.chunk", 1, "model", "fp_0");
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

/// Wait until `cond` holds: a liveness bound (2 minutes), not a speed one.
fn wait_for(what: &str, cond: impl Fn() -> bool) {
    let t = Instant::now();
    while !cond() {
        assert!(
            t.elapsed() < Duration::from_secs(120),
            "waited 2 min for {what}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
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
    // The steps below wait for the server's state, not for fixed times: on
    // a loaded machine (a whole suite in parallel, spinning pools on few
    // cores) a token took over 400 ms and a connection thread over 500 ms
    // to enqueue its request (MEASURED: 0.5B decode 47 -> 2.5 tok/s with 6
    // busy loops on 2 cores), and fixed sleeps made "the engine stalled"
    // out of a slow engine. `wait_for`'s deadline is for liveness only.
    wait_for("A to start generating", || {
        s.generated.load(Ordering::Relaxed) > 0
    });
    // B, C: queued behind it (the queue holds 2).
    let small = json!({"prompt": "Hi", "max_tokens": 1, "temperature": 0}).to_string();
    let mut cb = conn();
    let mut cc = conn();
    for c in [&mut cb, &mut cc] {
        c.write_all(request("POST", "/v1/completions", &small, false).as_bytes())
            .unwrap();
    }
    wait_for("B and C to be queued", || {
        s.queued.load(Ordering::Relaxed) == 3
    });
    // D: the queue is full.
    let mut cd = conn();
    cd.write_all(request("POST", "/v1/completions", &small, false).as_bytes())
        .unwrap();
    let r = read_resp(&mut cd, &mut Vec::new());
    assert_eq!(r.status, 503, "{}", r.body);
    assert!(r.body.contains("queued"));
    // While A's client reads nothing, the engine goes on, and /health
    // answers while A is still generating: it isn't queued behind it (A
    // runs to ~2000 tokens, the context; B and C haven't run).
    let g0 = s.generated.load(Ordering::Relaxed);
    let mut ce = conn();
    ce.write_all(request("GET", "/health", "", false).as_bytes())
        .unwrap();
    assert_eq!(read_resp(&mut ce, &mut Vec::new()).status, 200);
    let g1 = s.generated.load(Ordering::Relaxed);
    assert!(g1 < 1000, "/health waited for A's generation ({g1} tokens)");
    wait_for("the engine to go on with A", || {
        s.generated.load(Ordering::Relaxed) > g1.max(g0)
    });
    // A, B, C, D, E open, F the sixth; G is one too many: it closes D,
    // the connection that has waited longest for a request (its 503 was
    // answered first; E and F came later), and is served.
    let _cf = conn();
    std::thread::sleep(Duration::from_millis(100));
    let mut cg = conn();
    cg.write_all(request("GET", "/health", "", false).as_bytes())
        .unwrap();
    assert_eq!(read_resp(&mut cg, &mut Vec::new()).status, 200);
    assert_eq!(cd.read(&mut b).unwrap_or(0), 0, "D still open");
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

/// Slow-loris: idle sockets at the connection limit don't lock clients
/// out. A new connection closes the one that has waited longest for a
/// request (idle, or still sending its head); only when every connection
/// is busy with a request is it a 503.
#[test]
fn serve_evicts_waiting_connections() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cpu, |o| {
        o.max_conns = 4;
        o.max_queue = 8;
    });
    let a = s.addr;
    let conn = || {
        let c = TcpStream::connect(a).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
        c
    };
    let long = json!({"prompt": "Count: 1, 2, 3,", "max_tokens": 3000, "stream": true,
                      "temperature": 0})
    .to_string();
    let mut b = [0u8; 512];
    // A streams (nobody reads past the first bytes); B and C queue behind it.
    let mut ca = conn();
    ca.write_all(request("POST", "/v1/completions", &long, false).as_bytes())
        .unwrap();
    assert!(ca.read(&mut b).unwrap() > 0);
    let mut busy = vec![ca];
    for _ in 0..2 {
        let mut c = conn();
        c.write_all(request("POST", "/v1/completions", &long, false).as_bytes())
            .unwrap();
        busy.push(c);
    }
    wait_for("A, B, C to be queued", || {
        s.queued.load(Ordering::Relaxed) == 3
    });
    // D: idle (slow-loris), the fourth. E closes it and is served.
    let mut cd = conn();
    wait_for("D to wait", || s.waiting() == 1);
    let mut ce = conn();
    ce.write_all(request("GET", "/health", "", false).as_bytes())
        .unwrap();
    assert_eq!(read_resp(&mut ce, &mut Vec::new()).status, 200);
    assert_eq!(cd.read(&mut b).unwrap_or(0), 0, "D still open");
    // E, idle again after its answer, is now the one waiting longest: F
    // closes it. F's own head trickles in, which counts as waiting too: G
    // closes F.
    wait_for("E to wait again", || s.waiting() == 1);
    let mut cf = conn();
    cf.write_all(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\n")
        .unwrap();
    assert_eq!(
        ce.read(&mut b).unwrap_or(0),
        0,
        "E (oldest waiting) still open"
    );
    wait_for("F to wait", || s.waiting() == 1);
    let mut cg = conn();
    assert_eq!(cf.read(&mut b).unwrap_or(0), 0, "F still open");
    // G becomes busy too: A, B, C, G busy, nobody waiting: H is a 503.
    cg.write_all(request("POST", "/v1/completions", &long, false).as_bytes())
        .unwrap();
    wait_for("G to be queued", || s.queued.load(Ordering::Relaxed) == 4);
    let mut ch = conn();
    let r = read_resp(&mut ch, &mut Vec::new());
    assert_eq!(r.status, 503);
    assert!(r.body.contains("connections"));
    drop((busy, cg, ch));
    wait_for("the connections to close", || s.conns() == 0);
    let v = post(
        a,
        "/v1/completions",
        json!({"prompt": "Hi", "max_tokens": 1}),
    );
    assert_eq!(v.status, 200);
}

/// The eviction race: taking a connection off the wait list decides it.
/// A connection that took itself off first (its request arrived) is never
/// evicted; one whose entry is already gone was evicted, its socket shut
/// down, and doesn't serve. An empty list is the 503 path.
#[test]
fn wait_list_take_decides_the_race() {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let a = l.local_addr().unwrap();
    let (mut c1, mut c2) = (
        TcpStream::connect(a).unwrap(),
        TcpStream::connect(a).unwrap(),
    );
    let h1 = std::sync::Arc::new(l.accept().unwrap().0);
    let h2 = std::sync::Arc::new(l.accept().unwrap().0);
    let w = super::WaitList::default();
    assert!(!w.set(1, Some(&h1)));
    assert!(!w.set(2, Some(&h2)));
    // 1 (the oldest) got its request and took itself off first.
    assert!(w.set(1, None));
    assert!(w.evict_oldest());
    assert!(!w.set(2, None), "2 was evicted: it must not serve");
    assert!(!w.evict_oldest());
    c2.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    assert_eq!(c2.read(&mut [0u8; 8]).unwrap_or(0), 0, "2 not shut down");
    // 1 is untouched.
    (&*h1).write_all(b"ok").unwrap();
    c1.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut b = [0u8; 2];
    c1.read_exact(&mut b).unwrap();
    assert_eq!(&b, b"ok");
}

/// A client that leaves while its whole (non-streamed) response is being
/// generated cancels it within a few POLLs even though tokens keep coming;
/// before, the engine ran on to max_tokens while later requests waited.
#[test]
fn serve_whole_response_cancelled_on_disconnect() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let mut devices = vec![Device::Cpu];
    if crate::testutil::gpu().is_some() {
        devices.push(Device::Cuda);
    }
    for device in devices {
        let s = server(p.to_str().unwrap(), device);
        let body = json!({"prompt": "Count: 1, 2, 3,", "max_tokens": 1900, "temperature": 0,
                          "repetition_penalty": 1.0})
        .to_string();
        let mut c = TcpStream::connect(s.addr).unwrap();
        c.write_all(request("POST", "/v1/completions", &body, false).as_bytes())
            .unwrap();
        // Wait until it generates, then leave.
        wait_for("it to start", || s.generated.load(Ordering::Relaxed) >= 5);
        // On the GPU, a second request first: the first hands over to the
        // batch, and the cancel must drop it from there (its slot freed).
        let other = (device == Device::Cuda).then(|| {
            let a = s.addr;
            let g0 = s.generated.load(Ordering::Relaxed);
            let h = std::thread::spawn(move || {
                post(
                    a,
                    "/v1/completions",
                    json!({"prompt": "Hello", "max_tokens": 64,
                                                  "temperature": 0}),
                )
            });
            wait_for("the batch", || {
                s.generated.load(Ordering::Relaxed) >= g0 + 10
            });
            h
        });
        drop(c);
        if let Some(h) = other {
            assert_eq!(h.join().unwrap().status, 200);
        }
        // The next request is served once the engine is free (on the CPU:
        // right after the cancel, or (before) after all 1900 tokens; on the
        // GPU it shares the batch). No clock involved.
        let v = post(
            s.addr,
            "/v1/completions",
            json!({"prompt": "Hi", "max_tokens": 1}),
        );
        assert_eq!(v.status, 200);
        if device == Device::Cuda {
            // The cancelled request leaves the batch: generation stops.
            wait_for("the batch to go idle", || {
                let g = s.generated.load(Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(500));
                s.generated.load(Ordering::Relaxed) == g
            });
        }
        let g = s.generated.load(Ordering::Relaxed);
        assert!(g < 1900, "{device:?}: ran to the end: {g} tokens");
    }
}

/// An error answered before the body is read (413 here) is followed by
/// a lingering close: the server reads and drops what the client is still
/// sending. A client that sends its whole request before reading (most do)
/// then gets the 413; closing with the body unread reset the connection,
/// and the client saw a broken pipe / reset instead.
#[test]
fn serve_error_mid_upload_is_not_a_reset() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server(p.to_str().unwrap(), Device::Cpu);
    for _ in 0..3 {
        let mut c = TcpStream::connect(s.addr).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let n = 3 << 20;
        let head =
            format!("POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {n}\r\n\r\n");
        c.write_all(head.as_bytes()).unwrap();
        c.write_all(&vec![b'x'; n]).expect("upload");
        let r = read_resp(&mut c, &mut Vec::new());
        assert_eq!(r.status, 413, "{}", r.body);
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

/// Tool calling over HTTP on the test models (CPU): `tool_choice:
/// "required"` makes even a small model call; the streamed deltas add up
/// to the whole response's calls and content; a call's first delta has its
/// id and name; the next turn (the result appended) reuses the cache.
#[test]
fn serve_tool_calls() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let mut paths = vec![p];
    if crate::testutil::qwen3_model().is_some() {
        paths.push(std::env::var_os("AXON_TEST_QWEN3").map_or_else(
            || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(crate::testutil::QWEN3_MODEL),
            std::path::PathBuf::from,
        ));
    }
    for path in paths {
        let s = server(path.to_str().unwrap(), Device::Cpu);
        let a = s.addr;
        let tools = json!([{"type": "function", "function": {"name": "get_weather",
            "description": "Get the weather in a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}]);
        let user = json!({"role": "user", "content": "What's the weather in Paris?"});
        let req = json!({"messages": [user], "tools": tools, "tool_choice": "required",
                         "temperature": 0, "max_tokens": 64,
                         "chat_template_kwargs": {"enable_thinking": false}});
        let v = post(a, "/v1/chat/completions", req.clone()).json();
        let ch = &v["choices"][0];
        let m = &ch["message"];
        if ch["finish_reason"] != "tool_calls" {
            // Not a call (cut off, or not JSON): the block is content, verbatim.
            assert!(
                m["content"].as_str().unwrap().contains("<tool_call>"),
                "{v}"
            );
            continue;
        }
        let calls = m["tool_calls"].as_array().unwrap();
        assert!(!calls.is_empty(), "{v}");
        for c in calls {
            assert!(c["id"].as_str().unwrap().starts_with("call_"), "{v}");
            assert_eq!(c["type"], "function");
            let args = c["function"]["arguments"].as_str().unwrap();
            serde_json::from_str::<Value>(args).unwrap_or_else(|e| panic!("{e}: {v}"));
        }

        let mut sreq = req.clone();
        sreq["stream"] = json!(true);
        sreq["stream_options"] = json!({"include_usage": true});
        let r = post(a, "/v1/chat/completions", sreq);
        let ev = r.events();
        assert_eq!(ev.last().map(String::as_str), Some("[DONE]"));
        let (mut content, mut scalls, mut finish) = (String::new(), Vec::<Value>::new(), None);
        for e in ev.iter().filter(|e| *e != "[DONE]") {
            let v: Value = serde_json::from_str(e).unwrap();
            let Some(c) = v["choices"].get(0) else {
                assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
                continue;
            };
            content.push_str(c["delta"]["content"].as_str().unwrap_or(""));
            if let Some(f) = c["finish_reason"].as_str() {
                finish = Some(f.to_string());
                assert!(c["delta"].get("tool_calls").is_none(), "{e}");
            }
            for tc in c["delta"]["tool_calls"].as_array().into_iter().flatten() {
                let i = tc["index"].as_u64().unwrap() as usize;
                if i == scalls.len() {
                    // A call's first delta: id, type and name.
                    assert!(
                        tc["id"].is_string() && tc["function"]["name"].is_string(),
                        "{e}"
                    );
                    scalls.push(json!({"name": tc["function"]["name"], "arguments": ""}));
                }
                let args = format!(
                    "{}{}",
                    scalls[i]["arguments"].as_str().unwrap(),
                    tc["function"]["arguments"].as_str().unwrap_or("")
                );
                scalls[i]["arguments"] = json!(args);
            }
        }
        assert_eq!(finish.as_deref(), Some("tool_calls"));
        assert_eq!(content, m["content"].as_str().unwrap_or(""));
        let whole: Vec<Value> = calls
            .iter()
            .map(
                |c| json!({"name": c["function"]["name"], "arguments": c["function"]["arguments"]}),
            )
            .collect();
        assert_eq!(scalls, whole);

        // The next turn: the call and its result appended; the cache is reused.
        let p1 = v["usage"]["prompt_tokens"].as_u64().unwrap();
        let mut msgs = vec![
            user.clone(),
            json!({"role": "assistant", "content": null, "tool_calls": calls}),
        ];
        for c in calls {
            msgs.push(json!({"role": "tool", "tool_call_id": c["id"], "content": "18C, sunny"}));
        }
        let req2 = json!({"messages": msgs, "tools": tools, "temperature": 0, "max_tokens": 16,
                          "chat_template_kwargs": {"enable_thinking": false}});
        let v2 = post(a, "/v1/chat/completions", req2).json();
        let cached = v2["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap();
        assert!(cached * 10 >= p1 * 9, "reused {cached} of {p1}: {v2}");
        assert!(v2["usage"]["prompt_tokens"].as_u64().unwrap() > p1);
    }
}

/// Continuous batching (stage S2, GPU), `parallel` 8: requests sent
/// together, greedy and seeded sampling mixed (penalties, top_k off: the
/// host selects), get exactly the reply (content, reasoning, token count)
/// each gets alone, on the f16 and q8 caches, qwen2.5 and qwen3 (thinking);
/// also when the first one is already running alone (with drafting) as the
/// others arrive, so it hands over to the batch partway; and with more
/// requests than slots (later ones take freed slots). Then Gemma 3 1B with
/// 4 slots.
#[test]
fn serve_batched_matches_solo() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let mut paths = vec![p];
    if crate::testutil::qwen3_model().is_some() {
        paths.push(std::env::var_os("AXON_TEST_QWEN3").map_or_else(
            || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(crate::testutil::QWEN3_MODEL),
            std::path::PathBuf::from,
        ));
    }
    let bodies = batch_bodies(true);
    for path in &paths {
        for kv in [crate::kv::KvType::F16, crate::kv::KvType::Q8] {
            let s = server_with(path.to_str().unwrap(), Device::Cuda, |o| {
                o.kv = kv;
                o.parallel = 8;
                o.max_draft = 3;
            });
            let what = format!("{} {kv:?}", path.display());
            batched_vs_solo(s.addr, &bodies, &what);
        }
    }
    // Gemma 3 1B (head size 256, sliding window, its chat framing; f16 only
    // on the GPU), 4 slots: the batched step's rows are per-sequence.
    if let Some(gp) = crate::testutil::gemma3_model_path() {
        let s = server_with(gp.to_str().unwrap(), Device::Cuda, |o| {
            o.parallel = 4;
            o.max_draft = 3;
        });
        batched_vs_solo(s.addr, &bodies, &format!("{} F16 parallel 4", gp.display()));
    }
}

/// The batch test's requests: 10 (more than 8 slots), mixed sampling;
/// `max_tokens` on each (else none: up to the context).
fn batch_bodies(max_tokens: bool) -> Vec<Value> {
    [
        ("Write a story about a fox.", 120, json!({"temperature": 0})),
        (
            "Name three colors.",
            40,
            json!({"temperature": 0.8, "seed": 1}),
        ),
        (
            "Explain rain.",
            64,
            json!({"temperature": 0.9, "seed": 2, "top_k": 0}),
        ),
        (
            "List five animals.",
            50,
            json!({"temperature": 0, "repetition_penalty": 1.2}),
        ),
        (
            "Why is grass green?",
            72,
            json!({"temperature": 1.1, "seed": 3, "top_k": 20, "repetition_penalty": 1.1}),
        ),
        ("Count to twenty.", 90, json!({"temperature": 0})),
        (
            "What is 12 times 13?",
            30,
            json!({"temperature": 0.6, "seed": 4, "top_p": 0.8}),
        ),
        (
            "Describe the moon.",
            80,
            json!({"temperature": 0.7, "seed": 5, "min_p": 0.05}),
        ),
        ("Say hello in French.", 20, json!({"temperature": 0})),
        (
            "Give a haiku about snow.",
            48,
            json!({"temperature": 1.0, "seed": 6, "top_k": 256}),
        ),
    ]
    .into_iter()
    .map(|(m, n, mut s)| {
        s["messages"] = json!([{"role": "user", "content": m}]);
        if max_tokens {
            s["max_tokens"] = json!(n);
        }
        s
    })
    .collect()
}

/// Each of `bodies` alone, then all at once and staggered (the first one
/// alone for 60 ms first): the same replies.
fn batched_vs_solo(a: SocketAddr, bodies: &[Value], what: &str) {
    let reply = |r: &Resp| -> (String, String, Value) {
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        let m = &v["choices"][0]["message"];
        (
            m["content"].to_string(),
            m["reasoning_content"].to_string(),
            v["usage"]["completion_tokens"].clone(),
        )
    };
    let solo: Vec<_> = bodies
        .iter()
        .map(|b| reply(&post(a, "/v1/chat/completions", b.clone())))
        .collect();
    for stagger in [0u64, 60] {
        let together: Vec<_> = std::thread::scope(|sc| {
            let hs: Vec<_> = bodies
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    sc.spawn(move || {
                        if i > 0 {
                            std::thread::sleep(Duration::from_millis(stagger));
                        }
                        reply(&post(a, "/v1/chat/completions", b.clone()))
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (i, (t, s1)) in together.iter().zip(&solo).enumerate() {
            assert_eq!(t, s1, "{what} stagger {stagger} ms: request {i}");
        }
    }
}

/// A pool whose slots are shorter than the context (`slot_ctx`, as when
/// --parallel slots of --ctx don't fit): the context served is a slot's
/// (/v1/models, context_length_exceeded past it), and requests without max_tokens (up to that
/// context) batch with the same replies as alone.
#[test]
fn serve_shrunk_pool() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cuda, |o| {
        o.ctx = Some(2048);
        o.slot_ctx = Some(384);
        o.parallel = 8;
    });
    let a = s.addr;
    let v = call(a, "GET", "/v1/models", "").json();
    assert_eq!(v["data"][0]["meta"]["n_ctx"], 384, "{v}");
    let long = "word ".repeat(500);
    let r = post(
        a,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": long}], "max_tokens": 4}),
    );
    assert_eq!(r.status, 400, "{}", r.body);
    assert_eq!(r.json()["error"]["code"], "context_length_exceeded");
    batched_vs_solo(a, &batch_bodies(false), "shrunk pool, no max_tokens");
}

/// `plan_slots` over a model of the memory: s slots of c positions fit
/// in `free` if (s + 1) * c positions do (any cost growing with s and c
/// will do: `plan_slots` only asks `fits`).
fn plan(
    ctx: usize,
    ctx_set: bool,
    parallel: usize,
    parallel_set: bool,
    free: usize,
) -> (usize, usize, bool) {
    let p = super::plan_slots(ctx, ctx_set, parallel, parallel_set, |s, c| {
        (s + 1) * c <= free
    });
    (p.slots, p.ctx, p.note.is_some())
}

/// The default (neither --ctx nor --parallel): the trained context with as
/// many slots as fit; slots shorten only down to min(ctx, AGENT_CTX), else
/// one request at a time with the whole context.
#[test]
fn plan_slots_default() {
    use super::AGENT_CTX;
    // Everything fits: 8 slots of the whole context, nothing to say.
    assert_eq!(
        plan(32768, false, PARALLEL, false, 9 * 32768),
        (8, 32768, false)
    );
    assert_eq!(
        plan(32768, false, PARALLEL, false, 100 * 32768),
        (8, 32768, false)
    );
    // Room for 4 at the whole context: 4, no shortening.
    assert_eq!(
        plan(32768, false, PARALLEL, false, 5 * 32768 + 7),
        (4, 32768, false)
    );
    // 2 fit exactly.
    assert_eq!(
        plan(32768, false, PARALLEL, false, 3 * 32768),
        (2, 32768, false)
    );
    // 2 don't fit at 32768 (the floor for a 32k model): one at a time, 32768.
    assert_eq!(
        plan(32768, false, PARALLEL, false, 3 * 32768 - 1),
        (1, 32768, true)
    );
    assert_eq!(plan(32768, false, PARALLEL, false, 0), (1, 32768, true));
    // A 128k model: 2 slots fit at 50000 -> the multiple of 1024 below,
    // and that's still >= AGENT_CTX, so batching at 49152.
    assert_eq!(
        plan(131072, false, PARALLEL, false, 3 * 50000),
        (2, 49152, true)
    );
    // Context first: room for 3 slots of 65536 is 2 of 65536, not 3 shorter.
    assert_eq!(
        plan(131072, false, PARALLEL, false, 4 * 49152),
        (2, 65536, true)
    );
    // ... and once the whole context fits 2, the rest goes to slots.
    assert_eq!(
        plan(131072, false, PARALLEL, false, 4 * 131072),
        (3, 131072, false)
    );
    // ... but below AGENT_CTX: one at a time with the whole 128k.
    assert_eq!(
        plan(131072, false, PARALLEL, false, 3 * AGENT_CTX - 1),
        (1, 131072, true)
    );
    assert_eq!(
        plan(131072, false, PARALLEL, false, 3 * AGENT_CTX),
        (2, AGENT_CTX, true)
    );
    // A context shorter than AGENT_CTX (a fitted 20000, an 8k model) is
    // never shortened for slots.
    assert_eq!(
        plan(20000, false, PARALLEL, false, 3 * 20000),
        (2, 20000, false)
    );
    assert_eq!(
        plan(20000, false, PARALLEL, false, 3 * 20000 - 1),
        (1, 20000, true)
    );
    assert_eq!(
        plan(8192, false, PARALLEL, false, 9 * 8192),
        (8, 8192, false)
    );
    // --parallel 1 (or 0): one at a time, no note.
    assert_eq!(plan(32768, false, 1, true, 100 * 32768), (1, 32768, false));
    assert_eq!(plan(32768, false, 0, true, 100 * 32768), (1, 32768, false));
}

/// An explicit --ctx is never shortened: the slots that fit with it (up
/// to --parallel), else one request at a time; an explicit --parallel alone
/// gets its slots, at the longest context they fit with.
#[test]
fn plan_slots_explicit() {
    // --ctx 32768: 3 slots fit, the default --parallel 8 says nothing.
    assert_eq!(
        plan(32768, true, PARALLEL, false, 4 * 32768),
        (3, 32768, false)
    );
    // ... 2 don't: one at a time, still 32768 (the default would also
    // keep it, but for a 128k --ctx it would have batched shorter).
    assert_eq!(
        plan(32768, true, PARALLEL, false, 3 * 32768 - 1),
        (1, 32768, true)
    );
    assert_eq!(
        plan(131072, true, PARALLEL, false, 3 * 50000),
        (1, 131072, true)
    );
    // --ctx and --parallel: --ctx wins, fewer slots, said.
    assert_eq!(plan(16384, true, 6, true, 5 * 16384), (4, 16384, true));
    assert_eq!(plan(16384, true, 6, true, 7 * 16384), (6, 16384, false));
    // --parallel 4 alone: 4 slots, the context shortened to fit (a
    // multiple of 1024), down to 1024.
    assert_eq!(plan(32768, false, 4, true, 5 * 32768), (4, 32768, false));
    assert_eq!(plan(32768, false, 4, true, 5 * 20000), (4, 19456, true));
    assert_eq!(plan(32768, false, 4, true, 5 * 1024), (4, 1024, true));
    assert_eq!(plan(32768, false, 4, true, 5 * 1024 - 1), (1, 32768, true));
    // --parallel 16 (more than the default 8) alone: all 16 if they fit.
    assert_eq!(plan(4096, false, 16, true, 17 * 4096), (16, 4096, false));
    // A context under 1024 is kept whole or not batched.
    assert_eq!(plan(1000, false, 2, true, 3 * 1000), (2, 1000, false));
    assert_eq!(plan(1000, false, 2, true, 2999), (1, 1000, true));
}

/// The CPU's default context: the trained one if its cache fits in RAM,
/// else the most (a multiple of 1024) that does, never below the old
/// default min(trained, 8192).
#[test]
fn ram_ctx_fits() {
    use super::ram_ctx;
    let per = 57344; // 7B: 28 layers x 2 x 512 x f16
    assert_eq!(ram_ctx(32768, per, 32768 * per), 32768);
    assert_eq!(ram_ctx(32768, per, u64::MAX), 32768);
    assert_eq!(ram_ctx(32768, per, 32768 * per - 1), 31744);
    assert_eq!(ram_ctx(131072, per, 50000 * per), 49152);
    assert_eq!(ram_ctx(32768, per, 9000 * per), 8192);
    assert_eq!(ram_ctx(32768, per, 0), 8192);
    assert_eq!(ram_ctx(4096, per, 0), 4096);
    assert_eq!(ram_ctx(512, per, 0), 512);
}

/// The CPU arm with MemAvailable injected: less the mapped model and 1 GiB;
/// unknown (None) keeps the trained context; a note only when shorter.
#[test]
fn cpu_ctx_mem_available() {
    use super::cpu_ctx;
    const GIB: u64 = 1 << 30;
    let per = 57344;
    let model = 4 * GIB;
    assert_eq!(cpu_ctx(32768, per, model, None), (32768, None));
    let (c, why) = cpu_ctx(32768, per, model, Some(20 * GIB));
    assert_eq!((c, why), (32768, None));
    // 5 GiB + 1 GiB margin + the model: 1 GiB left = 18724 positions.
    let (c, why) = cpu_ctx(32768, per, model, Some(6 * GIB));
    assert_eq!(c, 18432);
    assert!(why.unwrap().contains("32768"));
    // The mapped model alone is more than what's available: the old default.
    let (c, why) = cpu_ctx(32768, per, model, Some(3 * GIB));
    assert_eq!(c, 8192);
    assert!(why.is_some());
    // Not mapped (copied): its RAM is already out of MemAvailable.
    assert_eq!(cpu_ctx(32768, per, 0, Some(3 * GIB)).0, 32768);
}

/// The one-at-a-time note names the context 2 slots would fit at.
#[test]
fn plan_slots_note_names_the_fit() {
    let p = super::plan_slots(32768, false, PARALLEL, false, |s, c| {
        (s + 1) * c <= 3 * 25000
    });
    assert_eq!((p.slots, p.ctx), (1, 32768));
    let note = p.note.unwrap();
    assert!(
        note.contains("2 at once fit at 24576: --ctx 24576"),
        "{note}"
    );
    let p = super::plan_slots(32768, false, PARALLEL, false, |_, _| false);
    assert!(p.note.unwrap().contains("even at 1024"));
}

/// No --ctx: the context served is the model's trained one where it fits
/// (/v1/models), at least the old default 8192 or what the GPU fits, and
/// on the GPU a prompt past 8192 tokens is answered.
#[test]
fn serve_default_ctx() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let trained = crate::gguf::load(p.to_str().unwrap()).unwrap().ctx_len() as u64;
    let n_ctx = |a| call(a, "GET", "/v1/models", "").json()["data"][0]["meta"]["n_ctx"].clone();
    let s = server_with(p.to_str().unwrap(), Device::Cpu, |o| o.ctx = None);
    let c = n_ctx(s.addr).as_u64().unwrap();
    assert!(
        (trained.min(8192)..=trained).contains(&c),
        "cpu: {c} of {trained}"
    );
    drop(s);
    if crate::testutil::gpu().is_none() {
        return;
    }
    let s = server_with(p.to_str().unwrap(), Device::Cuda, |o| o.ctx = None);
    let c = n_ctx(s.addr).as_u64().unwrap();
    assert!((4096..=trained).contains(&c), "cuda: {c} of {trained}");
    if c < 9500 {
        eprintln!("serve_default_ctx: {c} positions fit, no long prompt");
        return;
    }
    let long = "word ".repeat(9000);
    let r = post(
        s.addr,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": long}], "max_tokens": 2, "temperature": 0}),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let used = r.json()["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(used > 8192 && used < c, "{used} prompt tokens");
}

/// Chunked prefill (`--prefill-chunk`, `--prefill-ms`): long prompts
/// admitted while other requests decode are fed a chunk per step in
/// between those requests' steps (the decoding ones keep producing tokens
/// meanwhile), and every request's reply is the one it gets alone: two
/// long ones sent together (both prefilling at once), greedy and sampled
/// (top_k off: the host selects); chunks of 16 on the f16 and q8 caches,
/// chunks sized by a time budget (varying sizes), and the Gemma 3 model
/// (sliding-window layers, prompts past their window) when present; then
/// again with the long prompts extended, so their slots' rows are reused
/// and only the rest is fed in chunks.
#[test]
fn serve_chunked_prefill_matches_solo() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    use crate::kv::KvType::{F16, Q8};
    // (model, kv, chunk floor, ms budget)
    let mut configs = vec![
        (p.clone(), F16, 16, 0),
        (p.clone(), Q8, 16, 0),
        (p, F16, 16, 20),
    ];
    if let Some(g) = crate::testutil::gemma3_model_path() {
        configs.push((g, F16, 16, 0));
    }
    const N: u64 = 400;
    let decoders = [
        json!({"prompt": "The history of the Roman Empire", "max_tokens": N, "temperature": 0}),
        json!({"prompt": "Once upon a time, in a small village by the sea,", "max_tokens": N,
               "temperature": 0.8, "seed": 7}),
    ];
    let text: String = (0..120)
        .map(|i| {
            format!(
                "Line {i}: the {} cat sat on mat number {}. ",
                ["red", "blue", "green"][i % 3],
                i * 7
            )
        })
        .collect();
    let longs = |tail: &str| {
        [
            json!({"prompt": format!("{text}{tail}Summary:"), "max_tokens": 40, "temperature": 0}),
            json!({"prompt": format!("Notes. {text}{tail}In short,"), "max_tokens": 40,
                   "temperature": 0.9, "seed": 3, "top_k": 0}),
        ]
    };
    let completion = |r: &Resp| -> (String, Value) {
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        (
            v["choices"][0]["text"].to_string(),
            v["usage"]["completion_tokens"].clone(),
        )
    };
    for (path, kv, chunk, ms) in configs {
        let gemma = path.to_string_lossy().contains("gemma");
        let s = server_with(path.to_str().unwrap(), Device::Cuda, |o| {
            o.kv = kv;
            o.parallel = 4;
            o.max_draft = 3;
            o.prefill_chunk = chunk;
            o.prefill_ms = ms;
        });
        let a = s.addr;
        for (round, tail) in ["", "And one more line about a dog. "]
            .into_iter()
            .enumerate()
        {
            let long = longs(tail);
            let solo_dec: Vec<_> = decoders
                .iter()
                .map(|b| completion(&post(a, "/v1/completions", b.clone())))
                .collect();
            // (Gemma's raw completions may end early: then the "decoded
            // meanwhile" check is skipped.)
            let running = solo_dec.iter().all(|d| d.1 == N);
            assert!(
                running || gemma,
                "{kv:?}: the decoding requests must run {N} tokens"
            );
            let solo_long: Vec<_> = long
                .iter()
                .map(|b| completion(&post(a, "/v1/completions", b.clone())))
                .collect();
            let what = format!(
                "{} {kv:?} chunk {chunk} ms {ms} round {round}",
                path.display()
            );
            std::thread::scope(|sc| {
                let g = &s.generated;
                let g0 = g.load(Ordering::Relaxed);
                let dec: Vec<_> = decoders
                    .iter()
                    .map(|b| sc.spawn(move || completion(&post(a, "/v1/completions", b.clone()))))
                    .collect();
                wait_for("the decoding requests", || {
                    g.load(Ordering::Relaxed) >= g0 + 20
                });
                // The first long one streamed: how many tokens the others
                // made between its arrival and its first token.
                let mut b0 = long[0].clone();
                b0["stream"] = json!(true);
                let first = sc.spawn(move || {
                    let mut c = TcpStream::connect(a).unwrap();
                    c.set_read_timeout(Some(Duration::from_secs(300))).unwrap();
                    let req = request("POST", "/v1/completions", &b0.to_string(), true);
                    let mut buf = Vec::new();
                    let g1 = g.load(Ordering::Relaxed);
                    c.write_all(req.as_bytes()).unwrap();
                    while !buf.windows(6).any(|w| w == b"data: ") {
                        let mut t = [0u8; 4096];
                        let n = c.read(&mut t).expect("read");
                        assert!(n > 0, "EOF before the first token");
                        buf.extend_from_slice(&t[..n]);
                    }
                    let during = g.load(Ordering::Relaxed) - g1;
                    let r = read_resp(&mut c, &mut buf);
                    assert_eq!(r.status, 200, "{}", r.body);
                    let (t, _) = streamed_text(&r, false);
                    (during, t)
                });
                let r = post(a, "/v1/completions", long[1].clone());
                let second = completion(&r);
                let v = r.json();
                let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap();
                let prompt = v["usage"]["prompt_tokens"].as_u64().unwrap();
                assert!(prompt > 1000, "{what}: a {prompt}-token prompt");
                if round == 1 {
                    // Its slot holds round 0's prompt, all but the tail.
                    assert!(cached + 30 > prompt, "{what}: {cached} of {prompt} cached");
                }
                let (during, t0) = first.join().unwrap();
                eprintln!("{what}: {during} tokens decoded while the long prompt prefilled");
                assert_eq!(
                    serde_json::to_string(&t0).unwrap(),
                    solo_long[0].0,
                    "{what}: long request 0"
                );
                assert_eq!(second, solo_long[1], "{what}: long request 1");
                for (i, (h, d)) in dec.into_iter().zip(&solo_dec).enumerate() {
                    assert_eq!(&h.join().unwrap(), d, "{what}: decoding request {i}");
                }
                // Over 60 chunks of 16 for a prompt of over 1000 tokens,
                // each followed by a step of the two decoding requests
                // (round 1 feeds only the tail; a time budget's chunks
                // are larger).
                assert!(
                    round == 1 || ms > 0 || !running || during >= 40,
                    "{what}: {during} tokens decoded while a long prompt prefilled"
                );
            });
        }
    }
}

/// A request cancelled while its prompt is being fed in chunks frees its
/// slot with the rows fed so far, and the next prompt sharing that prefix
/// reuses them (cached_tokens) and gets the reply it gets alone.
#[test]
fn serve_cancel_mid_prefill_reuses_rows() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    // One token per chunk, a decode step between chunks: the prompt takes
    // over a second to feed, longer than a disconnect takes to notice.
    let s = server_with(p.to_str().unwrap(), Device::Cuda, |o| {
        o.parallel = 4;
        o.prefill_chunk = 1;
        o.prefill_ms = 0;
    });
    let a = s.addr;
    let text: String = (0..120)
        .map(|i| {
            format!(
                "Item {i}: a {} box holds {} stones. ",
                ["tin", "oak"][i % 2],
                i * 3
            )
        })
        .collect();
    let decoder =
        json!({"prompt": "The history of the Roman Empire", "max_tokens": 1500, "temperature": 0});
    let next = json!({"prompt": format!("{text}Total:"), "max_tokens": 30, "temperature": 0});
    let solo = post(a, "/v1/completions", next.clone()).json();
    let solo_text = solo["choices"][0]["text"].clone();
    std::thread::scope(|sc| {
        let g = &s.generated;
        let g0 = g.load(Ordering::Relaxed);
        let dec = sc.spawn(move || post(a, "/v1/completions", decoder));
        wait_for("the decoding request", || {
            g.load(Ordering::Relaxed) >= g0 + 20
        });
        // Its prompt: the same text with another end; gone partway in.
        let gone = json!({"prompt": format!("{text}Count:"), "max_tokens": 30, "stream": true});
        let mut c = TcpStream::connect(a).unwrap();
        let g1 = g.load(Ordering::Relaxed);
        c.write_all(request("POST", "/v1/completions", &gone.to_string(), true).as_bytes())
            .unwrap();
        // 50 decode steps: 50 chunks of its prompt fed.
        wait_for("its prefill to start", || {
            g.load(Ordering::Relaxed) >= g1 + 50
        });
        drop(c);
        // Once the disconnect is noticed (POLL) the slot is free again.
        std::thread::sleep(Duration::from_millis(600));
        let r = post(a, "/v1/completions", next);
        let v = r.json();
        assert_eq!(r.status, 200, "{}", r.body);
        let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap();
        let prompt = v["usage"]["prompt_tokens"].as_u64().unwrap();
        assert!(
            cached > 0 && cached + 10 < prompt,
            "{cached} of {prompt} tokens cached: the cancelled prompt's partial rows"
        );
        assert_eq!(v["choices"][0]["text"], solo_text);
        let d = dec.join().unwrap();
        assert_eq!(d.status, 200, "{}", d.body);
    });
}

/// The requests `identity_checks` follows: greedy and seeded chat (one
/// long, ~700 tokens of prompt, which a batch admits in chunks), seeded
/// completions; at most 48 tokens each.
fn identity_targets() -> Vec<(bool, Value)> {
    let long = "The river bends past the old mill and the bell tower. ".repeat(60);
    vec![
        (
            true,
            json!({"messages": [{"role": "user", "content": "Tell me about owls."}],
                   "max_tokens": 48, "temperature": 0}),
        ),
        (
            true,
            json!({"messages": [{"role": "user", "content": format!("{long}Summarize.")}],
                   "max_tokens": 32, "temperature": 0.8, "seed": 9, "top_p": 0.9}),
        ),
        (
            false,
            json!({"prompt": PROMPT, "max_tokens": 40, "temperature": 1.0, "seed": 3,
                   "top_k": 40}),
        ),
    ]
}

/// One target's reply text, its `system_fingerprint` and its cached prompt
/// tokens, whole or streamed (every streamed chunk must carry the same
/// fingerprint).
fn identity_reply(a: SocketAddr, chat: bool, body: &Value, stream: bool) -> (String, String, u64) {
    let path = if chat {
        "/v1/chat/completions"
    } else {
        "/v1/completions"
    };
    let mut b = body.clone();
    if stream {
        b["stream"] = json!(true);
        b["stream_options"] = json!({"include_usage": true});
    }
    let r = post(a, path, b);
    assert_eq!(r.status, 200, "{}", r.body);
    if !stream {
        let v = r.json();
        let text = if chat {
            v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_string()
        } else {
            v["choices"][0]["text"].as_str().unwrap().to_string()
        };
        let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0);
        return (
            text,
            v["system_fingerprint"].as_str().unwrap().into(),
            cached,
        );
    }
    let ev = r.events();
    let fps: Vec<String> = (ev[..ev.len() - 1].iter())
        .map(|e| {
            let v: Value = serde_json::from_str(e).unwrap();
            v["system_fingerprint"].as_str().unwrap().to_string()
        })
        .collect();
    assert!(fps.windows(2).all(|w| w[0] == w[1]), "{fps:?}");
    let (text, last) = streamed_text(&r, chat);
    let cached = last["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    (text, fps[0].clone(), cached)
}

/// The determinism contract over HTTP (docs/determinism.md) on server `a`:
/// each target alone, again (its prompt now cached: prefix reuse), streamed,
/// and in the middle of `load` other requests (concurrent: batched on the
/// GPU, queued on the CPU) gives the same text, and every response carries
/// the fingerprint GET /identity and /v1/models report. Returns the
/// identity and the targets' texts.
fn identity_checks(a: SocketAddr, load: &[Value], what: &str) -> (Value, Vec<String>) {
    let id = call(a, "GET", "/identity", "").json();
    let fp = id["system_fingerprint"].as_str().unwrap().to_string();
    assert!(fp.starts_with("fp_"), "{id}");
    assert_eq!(id["numerics"], crate::identity::NUMERICS_VERSION);
    let models = call(a, "GET", "/v1/models", "").json();
    for d in models["data"].as_array().unwrap() {
        assert_eq!(d["meta"]["identity"], id, "{what}");
    }
    let targets = identity_targets();
    let mut solo = Vec::new();
    for (i, (c, b)) in targets.iter().enumerate() {
        let s = identity_reply(a, *c, b, false);
        assert_eq!(s.1, fp, "{what}: target {i}");
        // Right away, so its prompt is still the cache's (one request at a time).
        let again = identity_reply(a, *c, b, false);
        assert_eq!(again.0, s.0, "{what}: target {i} with its prompt cached");
        assert!(again.2 > 0, "{what}: target {i} reused nothing");
        assert_eq!(again.1, fp);
        let st = identity_reply(a, *c, b, true);
        assert_eq!((&st.0, &st.1), (&s.0, &fp), "{what}: target {i} streamed");
        solo.push(s);
    }
    // Under load: the targets go in after the others, interleaved with them.
    let loaded: Vec<_> = std::thread::scope(|sc| {
        let others: Vec<_> = (load.iter())
            .map(|b| sc.spawn(move || post(a, "/v1/chat/completions", b.clone()).status))
            .collect();
        let hs: Vec<_> = (targets.iter().enumerate())
            .map(|(i, (c, b))| {
                sc.spawn(move || {
                    std::thread::sleep(Duration::from_millis(20 + 30 * i as u64));
                    identity_reply(a, *c, b, i == 1)
                })
            })
            .collect();
        for o in others {
            assert_eq!(o.join().unwrap(), 200);
        }
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (i, (l, s)) in loaded.iter().zip(&solo).enumerate() {
        assert_eq!((&l.0, &l.1), (&s.0, &fp), "{what}: target {i} under load");
    }
    (id, solo.into_iter().map(|s| s.0).collect())
}

/// Same request, same fingerprint, same text: alone, with its prefix
/// cached, streamed and under concurrent load (its long prompt prefilled
/// in 64-token chunks between decode steps), on the f16 and q8 caches
/// (8 slots); a restarted server has the same fingerprint, and the cache
/// type (and backend) is part of it.
#[test]
fn serve_identity() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let p = p.to_str().unwrap();
    let load = batch_bodies(true);
    let mut fps = Vec::new();
    for kv in [crate::kv::KvType::F16, crate::kv::KvType::Q8] {
        let s = server_with(p, Device::Cuda, |o| {
            o.kv = kv;
            o.parallel = 8;
            o.max_draft = 3;
            // The long target is admitted in chunks while others decode.
            o.prefill_chunk = 64;
        });
        let (id, _) = identity_checks(s.addr, &load, &format!("cuda {kv:?}"));
        assert_eq!(id["backend"], "cuda");
        assert_eq!(
            id["kv"],
            if kv == crate::kv::KvType::F16 {
                "f16"
            } else {
                "q8"
            }
        );
        assert!(id["gpu"].as_str().unwrap().starts_with("sm_"), "{id}");
        fps.push(id["system_fingerprint"].clone());
    }
    assert_ne!(fps[0], fps[1], "the cache type is in the fingerprint");
    // A restart (other slots, no drafting: neither is in it) has the same one.
    let s = server_with(p, Device::Cuda, |o| o.parallel = 2);
    let again = call(s.addr, "GET", "/identity", "").json();
    assert_eq!(again["system_fingerprint"], fps[0]);
    let s = server_with(p, Device::Cpu, |o| o.parallel = 1);
    let cpu = call(s.addr, "GET", "/identity", "").json();
    assert_ne!(cpu["system_fingerprint"], fps[0], "the backend is in it");
}

/// `serve_identity` on the CPU (one request at a time, the others queued);
/// a restarted server has the same fingerprint and, seeded, the same
/// samples (a fresh engine, nothing cached).
#[test]
fn serve_identity_cpu() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let p = p.to_str().unwrap();
    let load: Vec<Value> = batch_bodies(true).into_iter().take(3).collect();
    let s = server_with(p, Device::Cpu, |_| {});
    let (id, texts) = identity_checks(s.addr, &load, "cpu");
    assert_eq!(id["backend"], "cpu");
    assert!(id.get("gpu").is_none(), "{id}");
    drop(s);
    let s = server_with(p, Device::Cpu, |_| {});
    assert_eq!(call(s.addr, "GET", "/identity", "").json(), id);
    for (i, ((c, b), t)) in identity_targets().iter().zip(&texts).enumerate() {
        let r = identity_reply(s.addr, *c, b, false);
        assert_eq!(&r.0, t, "target {i} after a restart");
        assert!(i > 0 || r.2 == 0, "a fresh engine has nothing cached");
    }
    // A different seed samples differently (the seed is really used).
    let (c, mut b) = identity_targets().swap_remove(2);
    b["seed"] = json!(4);
    assert_ne!(identity_reply(s.addr, c, &b, false).0, texts[2]);
}

/// `call` with extra header lines (`Name: value\r\n` each).
fn call_h(addr: SocketAddr, method: &str, path: &str, body: &str, hdr: &str) -> Resp {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(300))).unwrap();
    let req =
        request(method, path, body, true).replacen("Host: x\r\n", &format!("Host: x\r\n{hdr}"), 1);
    s.write_all(req.as_bytes()).unwrap();
    read_resp(&mut s, &mut Vec::new())
}

/// A strict check of Prometheus text format 0.0.4 (what `prometheus_client`'s
/// parser takes): HELP/TYPE before a family's samples, one TYPE per family,
/// `name{label="v",...} value` lines, histograms cumulative with `+Inf` ==
/// `_count`. Returns the samples, `name{labels}` -> value.
fn prom_parse(text: &str) -> std::collections::BTreeMap<String, f64> {
    let name_ok = |n: &str| {
        !n.is_empty()
            && !n.starts_with(|c: char| c.is_ascii_digit())
            && n.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
    };
    assert!(text.ends_with('\n'), "no final newline");
    let mut types: std::collections::BTreeMap<String, String> = Default::default();
    let mut samples = std::collections::BTreeMap::new();
    for l in text.lines() {
        if let Some(r) = l.strip_prefix("# HELP ") {
            let (n, help) = r.split_once(' ').unwrap_or_else(|| panic!("HELP: {l}"));
            assert!(name_ok(n) && !help.is_empty(), "HELP: {l}");
            continue;
        }
        if let Some(r) = l.strip_prefix("# TYPE ") {
            let (n, t) = r.split_once(' ').unwrap_or_else(|| panic!("TYPE: {l}"));
            assert!(name_ok(n), "TYPE: {l}");
            assert!(["counter", "gauge", "histogram"].contains(&t), "TYPE: {l}");
            assert!(
                types.insert(n.into(), t.into()).is_none(),
                "TYPE twice: {l}"
            );
            continue;
        }
        assert!(!l.starts_with('#') && !l.is_empty(), "stray line: {l:?}");
        let (key, v) = l.rsplit_once(' ').unwrap_or_else(|| panic!("sample: {l}"));
        let v: f64 = match v {
            "+Inf" => f64::INFINITY,
            v => v.parse().unwrap_or_else(|_| panic!("value: {l}")),
        };
        let (name, labels) = match key.split_once('{') {
            Some((n, rest)) => (
                n,
                Some(rest.strip_suffix('}').unwrap_or_else(|| panic!("{l}"))),
            ),
            None => (key, None),
        };
        assert!(name_ok(name), "name: {l}");
        if let Some(ls) = labels {
            for kv in ls.split(',') {
                let (k, q) = kv.split_once('=').unwrap_or_else(|| panic!("label: {l}"));
                assert!(name_ok(k), "label name: {l}");
                assert!(
                    q.len() >= 2 && q.starts_with('"') && q.ends_with('"'),
                    "label value: {l}"
                );
                assert!(!q[1..q.len() - 1].contains(['"', '\\', '\n']), "{l}");
            }
        }
        let fam = ["_bucket", "_sum", "_count"]
            .iter()
            .find_map(|s| {
                name.strip_suffix(s)
                    .filter(|f| types.get(*f).is_some_and(|t| t == "histogram"))
            })
            .unwrap_or(name);
        let t = types
            .get(fam)
            .unwrap_or_else(|| panic!("sample before its TYPE: {l}"));
        if t == "counter" {
            assert!(name.ends_with("_total") && v >= 0.0, "counter: {l}");
        }
        assert!(
            samples.insert(key.to_string(), v).is_none(),
            "duplicate: {l}"
        );
    }
    for (fam, _) in types.iter().filter(|(_, t)| *t == "histogram") {
        let b: Vec<(&String, &f64)> = samples
            .iter()
            .filter(|(k, _)| k.starts_with(&format!("{fam}_bucket{{")))
            .collect();
        assert!(!b.is_empty(), "{fam}: no buckets");
        // BTreeMap order isn't le order: sort by le.
        let mut b: Vec<(f64, f64)> = b
            .iter()
            .map(|(k, v)| {
                let le = k.split("le=\"").nth(1).unwrap().trim_end_matches("\"}");
                (
                    if le == "+Inf" {
                        f64::INFINITY
                    } else {
                        le.parse().unwrap()
                    },
                    **v,
                )
            })
            .collect();
        b.sort_by(|x, y| x.0.total_cmp(&y.0));
        assert!(
            b.windows(2).all(|w| w[0].1 <= w[1].1),
            "{fam} not cumulative: {b:?}"
        );
        assert_eq!(b.last().unwrap().0, f64::INFINITY, "{fam}: no +Inf");
        assert_eq!(
            Some(&b.last().unwrap().1),
            samples.get(&format!("{fam}_count")),
            "{fam}"
        );
        assert!(samples.contains_key(&format!("{fam}_sum")), "{fam}");
    }
    samples
}

#[test]
fn prom_parse_rejects_bad_text() {
    let ok = "# HELP a_total x\n# TYPE a_total counter\na_total{r=\"/\"} 1\n";
    assert_eq!(prom_parse(ok)["a_total{r=\"/\"}"], 1.0);
    for bad in [
        "a_total 1\n",
        "# TYPE a_total counter\na_total one\n",
        "# TYPE a_total counter\na_total 1",
        "# TYPE a_total counter\n# TYPE a_total counter\na_total 1\n",
        "# TYPE a_total counter\na_total{r=/} 1\n",
        "# TYPE h histogram\nh_bucket{le=\"1\"} 2\nh_bucket{le=\"+Inf\"} 1\nh_sum 0\nh_count 1\n",
    ] {
        assert!(
            std::panic::catch_unwind(|| prom_parse(bad)).is_err(),
            "accepted {bad:?}"
        );
    }
}

/// --api-key: every route but /health (and CORS preflights) needs the
/// Bearer key; wrong or missing is a 401 in OpenAI's error shape.
#[test]
fn serve_api_key() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cpu, |o| {
        o.api_key = Some("s3cret-key".into());
    });
    let a = s.addr;
    let right = "Authorization: Bearer s3cret-key\r\n";
    for path in ["/v1/models", "/metrics", "/slots", "/identity", "/nope"] {
        for hdr in [
            "",
            "Authorization: Bearer s3cret-kez\r\n",
            "Authorization: Bearer s3cret-key2\r\n",
            "Authorization: Bearer \r\n",
            "Authorization: Basic s3cret-key\r\n",
            "Authorization: s3cret-key\r\n",
        ] {
            let r = call_h(a, "GET", path, "", hdr);
            assert_eq!(r.status, 401, "{path} {hdr:?}: {}", r.body);
            assert_eq!(r.header("www-authenticate"), Some("Bearer"));
            let e = &r.json()["error"];
            assert_eq!(e["code"], "invalid_api_key", "{}", r.body);
            assert_eq!(e["type"], "invalid_request_error");
            assert!(e["message"].as_str().unwrap().contains("Bearer"));
        }
    }
    for path in ["/v1/models", "/metrics", "/slots", "/identity"] {
        assert_eq!(call_h(a, "GET", path, "", right).status, 200, "{path}");
    }
    assert_eq!(call_h(a, "GET", "/nope", "", right).status, 404);
    let lower = "Authorization: bearer s3cret-key\r\n";
    assert_eq!(call_h(a, "GET", "/v1/models", "", lower).status, 200);
    // /health stays open, as do CORS preflights.
    assert_eq!(call(a, "GET", "/health", "").status, 200);
    assert_eq!(call(a, "OPTIONS", "/v1/chat/completions", "").status, 204);
    // Generation: refused without the key (nothing queued), served with it.
    let body = json!({"prompt": "Hi", "max_tokens": 2, "temperature": 0}).to_string();
    let r = call_h(a, "POST", "/v1/completions", &body, "");
    assert_eq!(r.status, 401);
    assert_eq!(s.queued.load(Ordering::Relaxed), 0);
    let r = call_h(a, "POST", "/v1/completions", &body, right);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["usage"]["completion_tokens"], 2);
    // The 401s are counted.
    let m = prom_parse(&call_h(a, "GET", "/metrics", "", right).body);
    assert_eq!(
        m["axon_requests_total{route=\"/v1/models\",status=\"401\"}"],
        6.0
    );
    assert_eq!(
        m["axon_requests_total{route=\"/v1/completions\",status=\"200\"}"],
        1.0
    );
}

/// GET /metrics parses as Prometheus text and counts what was served;
/// GET /slots has its shape (CPU: the one cache, `solo`); `priority` is
/// accepted and ignored; --max-prompt-tokens refuses before prefill.
#[test]
fn serve_metrics_slots_max_prompt() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cpu, |o| {
        o.max_prompt_tokens = Some(16);
    });
    let a = s.addr;
    // Before any request: zeros, a free slot.
    let m = prom_parse(&call(a, "GET", "/metrics", "").body);
    for (k, v) in [
        ("axon_generated_tokens_total", 0.0),
        ("axon_prompt_tokens_total", 0.0),
        ("axon_queue_depth", 0.0),
        ("axon_slots_busy", 0.0),
        ("axon_slots_free", 1.0),
        ("axon_ttft_seconds_count", 0.0),
    ] {
        assert_eq!(m[k], v, "{k}");
    }
    let sl = call(a, "GET", "/slots", "").json();
    assert_eq!(sl["ctx"], 2048);
    assert_eq!(sl["slots"], json!([]));
    assert_eq!(sl["solo"]["busy"], false);
    assert_eq!(sl["solo"]["request_id"], Value::Null);
    assert_eq!(sl["queue_depth"], 0);

    // Two requests (one with `priority`, ignored); the second reuses the
    // first's prompt rows.
    let mut prompt_tokens = 0;
    let mut cached = 0;
    let mut texts = Vec::new();
    for b in [
        json!({"prompt": PROMPT, "max_tokens": 6, "temperature": 0}),
        json!({"prompt": PROMPT, "max_tokens": 6, "temperature": 0, "priority": 7, "stream": true,
               "stream_options": {"include_usage": true}}),
    ] {
        let r = post(a, "/v1/completions", b.clone());
        assert_eq!(r.status, 200, "{}", r.body);
        let u = if b["stream"] == true {
            let (t, last) = streamed_text(&r, false);
            texts.push(t);
            last["usage"].clone()
        } else {
            texts.push(r.json()["choices"][0]["text"].as_str().unwrap().to_string());
            r.json()["usage"].clone()
        };
        prompt_tokens += u["prompt_tokens"].as_u64().unwrap();
        cached += u["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap();
    }
    assert_eq!(texts[0], texts[1], "priority changes nothing");
    assert!(cached > 0);
    let m = prom_parse(&call(a, "GET", "/metrics", "").body);
    assert_eq!(
        m["axon_generated_tokens_total"],
        s.generated.load(Ordering::Relaxed) as f64
    );
    assert_eq!(m["axon_generated_tokens_total"], 12.0);
    assert_eq!(m["axon_prompt_tokens_total"], prompt_tokens as f64);
    assert_eq!(m["axon_prompt_cached_tokens_total"], cached as f64);
    assert_eq!(
        m["axon_requests_total{route=\"/v1/completions\",status=\"200\"}"],
        2.0
    );
    assert_eq!(
        m["axon_requests_total{route=\"/metrics\",status=\"200\"}"],
        1.0
    );
    assert_eq!(m["axon_ttft_seconds_count"], 2.0);
    assert!(m["axon_ttft_seconds_sum"] > 0.0);
    assert!(m["axon_itl_seconds_count"] >= 2.0, "{m:?}");
    assert_eq!(m["axon_requests_in_flight"], 0.0);
    let sl = call(a, "GET", "/slots", "").json();
    assert_eq!(sl["solo"]["busy"], false);
    assert!(sl["solo"]["held_tokens"].as_u64().unwrap() >= prompt_tokens / 2);

    // While a request runs: busy, its id and age.
    let mut c = TcpStream::connect(a).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    let body = json!({"prompt": "Count: 1, 2, 3,", "max_tokens": 1500, "stream": true,
                      "temperature": 0})
    .to_string();
    c.write_all(request("POST", "/v1/completions", &body, false).as_bytes())
        .unwrap();
    assert!(c.read(&mut [0u8; 256]).unwrap() > 0);
    wait_for("the stream to show in /slots", || {
        call(a, "GET", "/slots", "").json()["solo"]["busy"] == true
    });
    let sl = call(a, "GET", "/slots", "").json();
    assert_eq!(sl["solo"]["request_id"], 3, "{sl}");
    assert!(sl["solo"]["age_ms"].as_u64().is_some(), "{sl}");
    let m = prom_parse(&call(a, "GET", "/metrics", "").body);
    assert_eq!((m["axon_slots_busy"], m["axon_slots_free"]), (1.0, 0.0));
    assert_eq!(m["axon_requests_in_flight"], 1.0);
    drop(c);
    wait_for("the cancelled stream to free the slot", || {
        call(a, "GET", "/slots", "").json()["solo"]["busy"] == false
    });

    // --max-prompt-tokens 16: refused with a 400 before anything is queued.
    let q = s.queued.load(Ordering::Relaxed);
    let long = "one two three four five six seven eight nine ten eleven twelve thirteen \
                fourteen fifteen sixteen seventeen";
    for (path, b, param) in [
        ("/v1/completions", json!({"prompt": long}), "prompt"),
        (
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": long}]}),
            "messages",
        ),
        (
            "/v1/completions",
            json!({"prompt": "x".repeat(200_000)}),
            "prompt",
        ),
    ] {
        let r = post(a, path, b);
        assert_eq!(r.status, 400, "{}", r.body);
        let e = &r.json()["error"];
        assert_eq!(e["code"], "context_length_exceeded", "{}", r.body);
        assert_eq!(e["param"], param);
        assert!(
            e["message"]
                .as_str()
                .unwrap()
                .contains("--max-prompt-tokens")
        );
    }
    assert_eq!(s.queued.load(Ordering::Relaxed), q, "nothing queued");
    let r = post(
        a,
        "/v1/completions",
        json!({"prompt": "Hi", "max_tokens": 1}),
    );
    assert_eq!(r.status, 200, "a short prompt still runs");
    let m = prom_parse(&call(a, "GET", "/metrics", "").body);
    assert_eq!(m["axon_prompt_refused_total"], 3.0);
}

/// Set `s`'s receive buffer (bytes; the kernel doubles it), so a client
/// that stops reading closes its window after a few KiB.
#[cfg(target_os = "linux")]
fn small_rcvbuf(s: &TcpStream, n: i32) {
    use std::os::fd::AsRawFd;
    unsafe extern "C" {
        fn setsockopt(
            fd: std::ffi::c_int,
            level: std::ffi::c_int,
            name: std::ffi::c_int,
            val: *const std::ffi::c_void,
            len: u32,
        ) -> std::ffi::c_int;
    }
    const SOL_SOCKET: std::ffi::c_int = 1;
    const SO_RCVBUF: std::ffi::c_int = 8;
    // SAFETY: an int option on an fd we own.
    let r = unsafe {
        setsockopt(
            s.as_raw_fd(),
            SOL_SOCKET,
            SO_RCVBUF,
            &n as *const i32 as *const std::ffi::c_void,
            4,
        )
    };
    assert_eq!(r, 0, "setsockopt");
}

/// Bytes waiting in `s`'s receive queue (FIONREAD).
#[cfg(target_os = "linux")]
fn inq(s: &TcpStream) -> i32 {
    use std::os::fd::AsRawFd;
    unsafe extern "C" {
        fn ioctl(fd: std::ffi::c_int, req: std::ffi::c_ulong, ...) -> std::ffi::c_int;
    }
    let mut n: std::ffi::c_int = 0;
    // SAFETY: FIONREAD writes one int.
    let r = unsafe { ioctl(s.as_raw_fd(), 0x541B, &mut n as *mut std::ffi::c_int) };
    assert_eq!(r, 0, "FIONREAD");
    n
}

/// A streaming client that stops reading: with --slot-idle-ms N the
/// stream is cancelled (as a disconnect) N ms after its receive window
/// fills, plus a poll and a step, and the slot serves the next request;
/// without it, the same client keeps the slot.
#[cfg(target_os = "linux")]
#[test]
fn serve_slot_idle_frees_a_stalled_stream() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    const IDLE: u64 = 300;
    for idle in [Some(IDLE), None] {
        let s = server_with(p.to_str().unwrap(), Device::Cpu, |o| {
            o.slot_idle_ms = idle;
        });
        let a = s.addr;
        let c = TcpStream::connect(a).unwrap();
        small_rcvbuf(&c, 2048);
        let body = json!({"prompt": "Count: 1, 2, 3,", "max_tokens": 1800, "stream": true,
                          "temperature": 0})
        .to_string();
        (&c).write_all(request("POST", "/v1/completions", &body, false).as_bytes())
            .unwrap();
        // The window is full when the client's queue stops growing while
        // the engine goes on generating (2 more tokens).
        let mut last = (inq(&c), Instant::now(), s.generated.load(Ordering::Relaxed));
        let full = loop {
            std::thread::sleep(Duration::from_millis(10));
            let n = inq(&c);
            let g = s.generated.load(Ordering::Relaxed);
            if n != last.0 {
                last = (n, Instant::now(), g);
            } else if n > 0 && g >= last.2 + 2 {
                break last.1;
            }
            assert!(last.1.elapsed() < Duration::from_secs(120), "never filled");
        };
        let busy = || call(a, "GET", "/slots", "").json()["solo"]["busy"] == true;
        let what = format!("--slot-idle-ms {idle:?}");
        match idle {
            Some(n) => {
                wait_for("the stalled stream to free its slot", || !busy());
                let took = full.elapsed();
                eprintln!("{what}: freed {took:?} after the window filled");
                // The stall time, a poll (n/4), a token step on a loaded
                // machine, and the time spent seeing the window full.
                assert!(
                    took < Duration::from_millis(n + 2500),
                    "{what}: freed {took:?} after the window filled"
                );
                assert!(took >= Duration::from_millis(n / 2), "{what}: {took:?}");
                let m = prom_parse(&call(a, "GET", "/metrics", "").body);
                assert_eq!(m["axon_slot_idle_evictions_total"], 1.0);
                let r = post(
                    a,
                    "/v1/completions",
                    json!({"prompt": "Hi", "max_tokens": 1}),
                );
                assert_eq!(r.status, 200, "the slot serves the next request");
            }
            None => {
                std::thread::sleep(Duration::from_millis(IDLE + 1500));
                assert!(busy(), "{what}: the stalled client still holds the slot");
                let m = prom_parse(&call(a, "GET", "/metrics", "").body);
                assert_eq!(m["axon_slot_idle_evictions_total"], 0.0);
            }
        }
        drop(c);
    }
}

/// GET /slots over a pool (GPU): one entry per slot, two concurrent
/// streams in two busy slots with their ids, freed when they leave.
#[test]
fn serve_cuda_slots() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cuda, |o| {
        o.parallel = 2;
        o.parallel_set = true;
    });
    let a = s.addr;
    let sl = call(a, "GET", "/slots", "").json();
    assert_eq!(sl["slots"].as_array().unwrap().len(), 2, "{sl}");
    let body = json!({"prompt": "Count: 1, 2, 3,", "max_tokens": 1500, "stream": true,
                      "temperature": 0})
    .to_string();
    let cs: Vec<TcpStream> = (0..2)
        .map(|_| {
            let mut c = TcpStream::connect(a).unwrap();
            c.write_all(request("POST", "/v1/completions", &body, false).as_bytes())
                .unwrap();
            c
        })
        .collect();
    wait_for("both streams in slots", || {
        call(a, "GET", "/slots", "").json()["slots"]
            .as_array()
            .unwrap()
            .iter()
            .all(|x| x["busy"] == true)
    });
    let sl = call(a, "GET", "/slots", "").json();
    let mut ids: Vec<u64> = sl["slots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| {
            assert!(x["age_ms"].as_u64().is_some(), "{sl}");
            assert!(x["held_tokens"].as_u64().unwrap() > 0, "{sl}");
            x["request_id"].as_u64().unwrap()
        })
        .collect();
    ids.sort();
    assert_eq!(ids, [1, 2], "{sl}");
    assert_eq!(sl["slots"][0]["id"], 0);
    let m = prom_parse(&call(a, "GET", "/metrics", "").body);
    assert_eq!((m["axon_slots_busy"], m["axon_slots_free"]), (2.0, 0.0));
    drop(cs);
    wait_for("the slots to be freed", || {
        let sl = call(a, "GET", "/slots", "").json();
        sl["slots"]
            .as_array()
            .unwrap()
            .iter()
            .all(|x| x["busy"] == false)
            && sl["solo"]["busy"] == false
    });
}

/// AXON_API_KEY is trimmed (a secret file's newline); set but blank is an
/// error; an empty key can't start a server (it would take an empty token).
#[test]
fn api_key_env_and_empty() {
    assert_eq!(super::env_key(None), Ok(None));
    assert_eq!(super::env_key(Some("k3y\n".into())), Ok(Some("k3y".into())));
    assert_eq!(
        super::env_key(Some("  k3y \r\n".into())),
        Ok(Some("k3y".into()))
    );
    assert!(super::env_key(Some(" \n".into())).is_err());
    assert!(super::env_key(Some(Default::default())).is_err());
    // Not UTF-8: an error, never "unset" (which would turn auth off).
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let bad = std::ffi::OsString::from_vec(vec![b'k', 0xff, b'y']);
        assert!(super::env_key(Some(bad)).unwrap_err().contains("UTF-8"));
    }
    // start() refuses an empty or blank key before binding or loading
    // anything (no model needed).
    for k in ["", " ", "\n"] {
        let mut o = test_opts("/nonexistent/model.gguf", Device::Cpu);
        o.api_key = Some(k.into());
        match start(o) {
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{k:?}: {e}"),
            Ok(_) => panic!("{k:?}: an empty key must not start"),
        }
    }
}

/// --api-key and `Expect: 100-continue`: without the key, a 401 before
/// the body is invited (no `100 Continue`); with it, the usual 100 then
/// the answer.
#[test]
fn serve_api_key_before_100_continue() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cpu, |o| {
        o.api_key = Some("k3y".into());
    });
    let body = json!({"prompt": "Hi", "max_tokens": 1, "temperature": 0}).to_string();
    for (auth, want) in [("", 401), ("Authorization: Bearer k3y\r\n", 200)] {
        let mut c = TcpStream::connect(s.addr).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
        write!(
            c,
            "POST /v1/completions HTTP/1.1\r\nHost: x\r\n{auth}Expect: 100-continue\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        let mut buf = Vec::new();
        let r = read_resp(&mut c, &mut buf);
        if want == 401 {
            assert_eq!(r.status, 401, "{}", r.body);
            assert_eq!(r.json()["error"]["code"], "invalid_api_key");
            assert_eq!(r.header("www-authenticate"), Some("Bearer"));
            continue;
        }
        assert_eq!(r.status, 100);
        c.write_all(body.as_bytes()).unwrap();
        let r = read_resp(&mut c, &mut buf);
        assert_eq!(r.status, 200, "{}", r.body);
    }
}

/// An error answered by `connection` after a HEAD request on the same
/// keep-alive connection still has its body (not the HEAD's headers-only
/// mode); a body-carrying request without the key is refused after its
/// head (no `100 Continue`) and counted under its route.
#[test]
fn serve_api_key_error_after_head() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cpu, |o| {
        o.api_key = Some("k3y".into());
    });
    let mut c = TcpStream::connect(s.addr).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    let mut buf = Vec::new();
    c.write_all(b"HEAD /health HTTP/1.1\r\nHost: x\r\n\r\n")
        .unwrap();
    let r = read_resp_to(&mut c, &mut buf, true);
    assert_eq!((r.status, r.body.as_str()), (200, ""));
    c.write_all(
        b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\n\
          Content-Length: 50\r\n\r\n",
    )
    .unwrap();
    let r = read_resp(&mut c, &mut buf);
    assert_eq!(r.status, 401, "{}", r.body);
    assert_eq!(r.json()["error"]["code"], "invalid_api_key");
    // Without Expect: the body (sent with the head) is never read as a
    // request; the 401 comes, then the connection closes.
    let body = json!({"prompt": "Hi", "max_tokens": 1}).to_string();
    let r = call_h(s.addr, "POST", "/v1/chat/completions", &body, "");
    assert_eq!(r.status, 401, "{}", r.body);
    assert_eq!(r.header("connection"), Some("close"));
    assert_eq!(s.queued.load(Ordering::Relaxed), 0);
    let ok = "Authorization: Bearer k3y\r\n";
    let m = prom_parse(&call_h(s.addr, "GET", "/metrics", "", ok).body);
    assert_eq!(
        m["axon_requests_total{route=\"/v1/completions\",status=\"401\"}"],
        1.0
    );
    assert_eq!(
        m["axon_requests_total{route=\"/v1/chat/completions\",status=\"401\"}"],
        1.0
    );
    assert!(!m.keys().any(|k| k.contains("route=\"other\"")), "{m:?}");
}

/// --api-key wins over AXON_API_KEY: the variable isn't read at all, so a
/// blank or non-UTF-8 one doesn't abort a server started with the flag.
#[test]
fn api_key_flag_wins_over_bad_env() {
    let never = || -> Option<std::ffi::OsString> { panic!("the env var was read") };
    assert_eq!(
        super::api_key_of(Some("k".into()), never),
        Ok((Some("k".into()), "--api-key"))
    );
    assert_eq!(
        super::api_key_of(None, || Some(" k \n".into())),
        Ok((Some("k".into()), "AXON_API_KEY"))
    );
    assert!(super::api_key_of(None, || Some(" ".into())).is_err());
    assert_eq!(super::api_key_of(None, || None), Ok((None, "AXON_API_KEY")));
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let bad = || Some(std::ffi::OsString::from_vec(vec![0xff]));
        assert!(super::api_key_of(None, bad).is_err());
        assert_eq!(
            super::api_key_of(Some("k".into()), bad),
            Ok((Some("k".into()), "--api-key"))
        );
    }
}

/// With --api-key no body gets through without the key, whatever the
/// route: POST /health, OPTIONS and a chunked /health with a body are
/// refused right after the head (only the head is sent: a server that read
/// the body would wait for it), as is a body past MAX_BODY (401, not 413);
/// a refused HEAD gets headers only. Bodiless /health and OPTIONS stay open.
#[test]
fn serve_api_key_no_body_without_the_key() {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cpu, |o| {
        o.api_key = Some("k3y".into());
    });
    let head_only = |head: &str, no_body: bool| {
        let mut c = TcpStream::connect(s.addr).unwrap();
        // A server waiting for the body would answer 408 after 30 s.
        c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        c.write_all(head.as_bytes()).unwrap();
        let mut buf = Vec::new();
        let r = read_resp_to(&mut c, &mut buf, no_body);
        let mut rest = [0u8; 64];
        let n = c.read(&mut rest).unwrap_or(0);
        (r, buf.len() + n)
    };
    for head in [
        format!(
            "POST /health HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n",
            1 << 20
        ),
        "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n".into(),
        "POST /health HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n".into(),
        "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 9000000\r\n\r\n".into(),
        "GET /health HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\n".into(),
    ] {
        let (r, extra) = head_only(&head, false);
        assert_eq!(r.status, 401, "{head:?}: {}", r.body);
        assert_eq!(r.json()["error"]["code"], "invalid_api_key", "{head:?}");
        assert_eq!(r.header("connection"), Some("close"), "{head:?}");
        assert_eq!(extra, 0, "{head:?}: nothing after the 401");
    }
    // HEAD with a body and no key: a 401 with headers only.
    let (r, extra) = head_only(
        "HEAD /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\n",
        true,
    );
    assert_eq!((r.status, r.body.as_str(), extra), (401, "", 0));
    // With the key, past MAX_BODY is still a 413.
    let (r, _) = head_only(
        "POST /v1/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k3y\r\n\
         Content-Length: 9000000\r\n\r\n",
        false,
    );
    assert_eq!(r.status, 413, "{}", r.body);
    // Bodiless /health and OPTIONS stay open.
    assert_eq!(call(s.addr, "GET", "/health", "").status, 200);
    assert_eq!(
        call(s.addr, "OPTIONS", "/v1/chat/completions", "").status,
        204
    );
    let ok = "Authorization: Bearer k3y\r\n";
    let m = prom_parse(&call_h(s.addr, "GET", "/metrics", "", ok).body);
    assert_eq!(
        m["axon_requests_total{route=\"/health\",status=\"401\"}"],
        3.0
    );
}

/// A completion's (text, cached prompt tokens, prompt tokens, completion
/// tokens), greedy.
fn completion_usage(a: SocketAddr, prompt: &str, n: usize) -> (Value, u64, u64, u64) {
    let r = post(
        a,
        "/v1/completions",
        json!({"prompt": prompt, "max_tokens": n, "temperature": 0}),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let u = &v["usage"];
    (
        v["choices"][0]["text"].clone(),
        u["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap(),
        u["prompt_tokens"].as_u64().unwrap(),
        u["completion_tokens"].as_u64().unwrap(),
    )
}

/// The request running alone runs in slot 0 (`Batch::lend_solo` /
/// `end_solo`), so slot 0's rows are shared by the batch and the solo
/// path, and each reuses what the other left there:
/// (a) a request the batch admits into slot 0 (slot 1 busy), then a request
///     alone sharing its prompt's prefix: cached_tokens is that prefix, and
///     the reply is the one it gets alone;
/// (b) a request alone that hands over to the batch partway (a second one
///     arrives) and finishes there, then a request alone sharing its
///     prefix: the same.
#[test]
fn serve_solo_and_batch_share_slot_0() {
    let Some(_) = crate::testutil::gpu() else {
        return;
    };
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let s = server_with(p.to_str().unwrap(), Device::Cuda, |o| {
        o.parallel = 2;
        o.max_draft = 3;
    });
    let a = s.addr;
    let g = &s.generated;
    let text = |w: &str| -> String {
        (0..60)
            .map(|i| {
                format!(
                    "{w} {i}: a {} box holds {} stones. ",
                    ["tin", "oak"][i % 2],
                    i * 7
                )
            })
            .collect()
    };
    let (shared_a, shared_b) = (text("Item"), text("Crate"));
    let decoder = "The history of the Roman Empire";
    let x_prompt = format!("{shared_b}Summary:");
    let (z_prompt, w_prompt) = (format!("{shared_a}Total:"), format!("{shared_a}Count:"));
    let v_prompt = format!("{shared_b}List:");
    // Each alone first (the references).
    let alone = |p: &str, n| completion_usage(a, p, n);
    let w_ref = alone(&w_prompt, 24);
    let v_ref = alone(&v_prompt, 24);
    let x_ref = alone(&x_prompt, 200);
    assert_eq!(x_ref.3, 200, "the hand-over request runs to its length");
    // The shared prefix's tokens: what an extension of it reuses.
    let prefix = |p: &str| completion_usage(a, p, 1).2 - 1;
    let (pre_a, pre_b) = (prefix(&shared_a), prefix(&shared_b));

    std::thread::scope(|sc| {
        // (b) X alone (slot 0); Y arrives after X's 20th token: X hands
        // over and finishes in the batch, Y decodes on in slot 1.
        let g0 = g.load(Ordering::Relaxed);
        let x = sc.spawn(|| completion_usage(a, &x_prompt, 200));
        wait_for("X decoding alone", || g.load(Ordering::Relaxed) >= g0 + 20);
        let y = sc.spawn(|| completion_usage(a, decoder, 1000));
        let x = x.join().unwrap();
        assert_eq!(x.0, x_ref.0, "X, handed over: its reply alone");
        // (a) Z arrives with Y running: the batch admits it into slot 0
        // (the only free one).
        completion_usage(a, &z_prompt, 16);
        assert!(!y.is_finished(), "Y still running while Z was admitted");
        y.join().unwrap();
    });
    // (a) W alone: slot 0 holds Z's prompt.
    let w = completion_usage(a, &w_prompt, 24);
    assert_eq!(w.0, w_ref.0, "W after Z in slot 0: its reply alone");
    assert!(
        w.1 + 4 >= pre_a && w.1 <= w.2,
        "W reuses Z's prefix: {} cached of {} (prefix {pre_a})",
        w.1,
        w.2
    );
    // (b) slot 0 now holds W; X's rows were overwritten. Run X's prefix
    // again through a hand-over, then V alone reuses it.
    std::thread::scope(|sc| {
        let g0 = g.load(Ordering::Relaxed);
        let x = sc.spawn(|| completion_usage(a, &x_prompt, 200));
        wait_for("X decoding alone", || g.load(Ordering::Relaxed) >= g0 + 20);
        let y = sc.spawn(|| completion_usage(a, decoder, 40));
        assert_eq!(x.join().unwrap().0, x_ref.0, "X again, handed over");
        y.join().unwrap();
    });
    // Y (40 tokens) may outlast X, in slot 1: slot 0 still holds X's rows.
    let v = completion_usage(a, &v_prompt, 24);
    assert_eq!(v.0, v_ref.0, "V after X's hand-over: its reply alone");
    assert!(
        v.1 + 4 >= pre_b && v.1 <= v.2,
        "V reuses X's prefix: {} cached of {} (prefix {pre_b})",
        v.1,
        v.2
    );
}
