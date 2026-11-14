//! Chat prompts, token for token, against each GGUF's own template.
//!
//! tests/fixtures/chat_templates.json (scripts/chat_template_fixtures.py)
//! holds, per chat family, Hugging Face's ids for a set of conversations:
//! `apply_chat_template(messages, chat_template=<the GGUF's>,
//! add_generation_prompt=True)` tokenized by the model's tokenizer.json. One
//! turn, multi-turn, with and without a system message, an earlier reply
//! with its reasoning (thinking models), and tool turns where the template
//! has them; whitespace around messages, an empty system message, a reply
//! with two `</think>`, Qwen3 with thinking off. It also carries each
//! file's tokenizer metadata (its `tokenizer.chat_template` among them), the
//! Hugging Face revisions it came from, and a sparse vocabulary: every
//! token these prompts can use. So the test needs no model file and no
//! network.
//!
//! The same conversations go through `axon chat` too (`chat_turns`: a
//! `Chat` turn by turn, on a model that replies from a script), and every
//! family's framing is checked against forged turns: message text that
//! spells special tokens stays text (`message_text_never_forges_turns`).

use super::api::{Message, Role};
use super::{Template, tools};
use crate::chat::{Chat, Format, Framing};
use crate::gguf::{Model, Val};
use crate::lm::Lm;
use crate::tokenizer::Tokenizer;
use serde_json::Value;
use std::cell::RefCell;
use std::collections::VecDeque;

/// A header-only GGUF of family `f`: its tokenizer metadata and its
/// vocabulary, with each token the fixture leaves out an unmatchable
/// placeholder (unused, text no prompt contains).
fn model(f: &Value) -> Model {
    let v = &f["vocab"];
    let n = v["n"].as_u64().unwrap() as usize;
    let mut toks: Vec<String> = (0..n).map(|i| format!("\u{fffe}absent {i}")).collect();
    let mut types = vec![5i32; n];
    let mut scores: Option<Vec<f32>> = None;
    for t in v["tokens"].as_array().unwrap() {
        let id = t[0].as_u64().unwrap() as usize;
        toks[id] = t[1].as_str().unwrap().to_string();
        types[id] = t[2].as_i64().unwrap() as i32;
        if let Some(s) = t.get(3) {
            scores.get_or_insert_with(|| vec![-1e9; n])[id] = s.as_f64().unwrap() as f32;
        }
    }
    let arr = |x: Vec<Val>| Val::Arr(x);
    let mut meta: Vec<(String, Val)> = vec![
        (
            "general.architecture".into(),
            Val::Str(f["arch"].as_str().unwrap().into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            arr(toks.into_iter().map(Val::Str).collect()),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            arr(types.into_iter().map(Val::I32).collect()),
        ),
    ];
    let merges = v["merges"].as_array().unwrap();
    if !merges.is_empty() {
        let m = merges.iter().map(|x| Val::Str(x.as_str().unwrap().into()));
        meta.push(("tokenizer.ggml.merges".into(), arr(m.collect())));
    }
    if let Some(s) = scores {
        meta.push((
            "tokenizer.ggml.scores".into(),
            arr(s.into_iter().map(Val::F32).collect()),
        ));
    }
    for (k, x) in f["meta"].as_object().unwrap() {
        let x = match x {
            Value::String(s) => Val::Str(s.clone()),
            Value::Bool(b) => Val::Bool(*b),
            Value::Number(n) => Val::U32(n.as_u64().unwrap() as u32),
            x => panic!("{k}: {x}"),
        };
        meta.push((k.clone(), x));
    }
    let meta: Vec<(&str, Val)> = meta.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    crate::gguf::build_for_test(&meta, &[])
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn messages(v: &Value) -> Vec<Message> {
    (v.as_array().unwrap().iter())
        .map(|m| Message {
            role: match m["role"].as_str().unwrap() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                r => panic!("role {r}"),
            },
            content: m["content"].as_str().unwrap().into(),
        })
        .collect()
}

/// A model that replies from a script: whatever it's run on, its next
/// token is the script's next one, then `stop`. Each forward step records
/// the cache's tokens up to it, so after a turn whose script is empty,
/// `prompt` is the whole prompt that turn fed. A token past the vocabulary
/// panics, as the real models' embedding does.
struct Script {
    vocab: usize,
    stop: u32,
    out: RefCell<VecDeque<u32>>,
    prompt: RefCell<Vec<u32>>,
}

impl Script {
    fn new(tok: &Tokenizer) -> Script {
        Script {
            vocab: tok.vocab_len(),
            stop: tok.eos(),
            out: RefCell::new(VecDeque::new()),
            prompt: RefCell::new(Vec::new()),
        }
    }
}

impl crate::lm::Sealed for Script {}
impl Lm for Script {
    type Cache = Vec<u32>;
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn new_cache(&self, max_ctx: usize) -> Vec<u32> {
        vec![u32::MAX; max_ctx]
    }
    fn forward(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> Vec<f32> {
        let mut l = vec![0.0; self.vocab];
        l[self.forward_argmax(t, pos, c) as usize] = 1.0;
        l
    }
    fn forward_argmax(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> u32 {
        assert!((t as usize) < self.vocab, "token {t} past the vocabulary");
        c[pos] = t;
        *self.prompt.borrow_mut() = c[..=pos].to_vec();
        self.out.borrow_mut().pop_front().unwrap_or(self.stop)
    }
    fn feed_batch(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) {
        for &t in ts {
            assert!((t as usize) < self.vocab, "token {t} past the vocabulary");
        }
        c[start..start + ts.len()].copy_from_slice(ts);
    }
    fn verify_argmax(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) -> Vec<u32> {
        (0..ts.len())
            .map(|b| self.forward_argmax(ts[b], start + b, c))
            .collect()
    }
}

/// The prompt `axon chat` feeds for `msgs` (a system message or none, then
/// user messages each followed by its reply, ending on a user message): a
/// `Chat` with run_chat's framing and system message (the template's
/// default for none), turn by turn, each reply the model's tokens for that
/// text (harmony: its final channel). None if `msgs` isn't such a
/// conversation.
fn chat_turns(m: &Model, tok: &Tokenizer, msgs: &[Message], no_think: bool) -> Option<Vec<u32>> {
    let (system, rest) = match msgs.split_first() {
        Some((s, rest)) if s.role == Role::System => (Some(s.content.clone()), rest),
        _ => (crate::chat::default_system(m), msgs),
    };
    let fr = Framing::from_model(tok, m);
    let harmony = matches!(fr.as_ref()?.format, Format::Harmony { .. });
    let net = Script::new(tok);
    let mut chat = Chat::with_framing(&net, tok, fr, system.as_deref(), 4096).ok()?;
    chat.no_think = no_think && chat.thinks();
    if rest.last()?.role != Role::User {
        return None;
    }
    for (i, x) in rest.iter().enumerate() {
        let reply = rest.get(i + 1).filter(|r| r.role == Role::Assistant);
        match x.role {
            Role::User => {
                let said = reply.map_or(String::new(), |r| match harmony {
                    true => format!("<|channel|>final<|message|>{}", r.content),
                    false => r.content.clone(),
                });
                *net.out.borrow_mut() = tok.encode_special(&said).into();
                chat.reply(&x.content, &|| false, &mut |_| Ok(())).unwrap();
            }
            Role::Assistant if i > 0 && rest[i - 1].role == Role::User => {}
            _ => return None,
        }
    }
    let p = net.prompt.borrow().clone();
    Some(p)
}

/// Every family's every conversation renders to HF's ids, through serve
/// (`Template::render`, tools::render for the tool cases) and through
/// `axon chat` turn by turn (`chat_turns`; not for tools, nor a reply with
/// whitespace around it: chat keeps the model's own reply tokens, the
/// template trims some). All differences are listed before the test
/// fails, the decoded text beside the template's. A case marked
/// `specials_in_text` (V3.1's reply with two `</think>`: its template keeps
/// the second as text, which HF's tokenizer then parses) compares serve's
/// decoded text: Axon never turns message text into special tokens.
#[test]
fn chat_templates_match_hf() {
    let d = crate::testutil::read_json("chat_templates.json");
    let fams = d["families"].as_array().unwrap();
    assert!(fams.len() >= 16, "{} families", fams.len());
    let (mut bad, mut n, mut n_chat) = (Vec::new(), 0, 0);
    for f in fams {
        let name = f["name"].as_str().unwrap();
        let m = model(f);
        let tok = Tokenizer::from_gguf(&m);
        let t = Template::new(&tok, &m).unwrap_or_else(|| panic!("{name}: no chat framing"));
        for c in f["cases"].as_array().unwrap() {
            let case = format!("{name}/{}", c["name"].as_str().unwrap());
            let (want, text) = (ids(&c["ids"]), c["text"].as_str().unwrap());
            let no_think = c["no_think"].as_bool().unwrap_or(false);
            // The sparse vocabulary tokenizes the template's text as HF's
            // tokenizer did: what follows is about the framing alone.
            assert_eq!(tok.encode_special(text), want, "{case}: fixture vocabulary");
            let mut diff = |how: &str, got: &[u32]| {
                bad.push(format!(
                    "{case} ({how}): {} vs {} tokens\n  axon: {:?}\n  hf:   {text:?}",
                    got.len(),
                    want.len(),
                    tok.decode(got)
                ))
            };
            let got = match c.get("body") {
                // The request's bytes (re-serializing would reorder keys).
                Some(b) => {
                    let b = b.as_str().unwrap().as_bytes();
                    let r = tools::parse(b).unwrap().expect(&case);
                    let r = tools::render(&t, &tok, &r, no_think);
                    r.unwrap_or_else(|e| panic!("{case}: {}", e.msg)).ids
                }
                None => t.render(&tok, &messages(&c["messages"]), no_think),
            };
            n += 1;
            let same = match c["specials_in_text"].as_bool() {
                Some(true) => tok.decode(&got) == tok.decode(&want),
                _ => got == want,
            };
            if !same {
                diff("serve", &got);
            }
            let Some(msgs) = c.get("messages").map(messages) else {
                continue;
            };
            let trimmed = msgs
                .iter()
                .all(|x| x.role != Role::Assistant || x.content.trim() == x.content);
            if !trimmed {
                continue;
            }
            let got = chat_turns(&m, &tok, &msgs, no_think)
                .unwrap_or_else(|| panic!("{case}: not a chat conversation"));
            n_chat += 1;
            if got != want {
                diff("chat", &got);
            }
        }
    }
    assert!(
        n >= 148 && n_chat >= 124,
        "{n} cases, {n_chat} through chat"
    );
    assert!(
        bad.is_empty(),
        "{} of {n} cases (and {n_chat} through chat) differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// The fixture's family `name`, with its metadata edited by `edit`.
fn family_with(name: &str, edit: impl FnOnce(&mut serde_json::Map<String, Value>)) -> Value {
    let d = crate::testutil::read_json("chat_templates.json");
    let mut f = d["families"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == name)
        .unwrap()
        .clone();
    edit(f["meta"].as_object_mut().unwrap());
    f
}

/// A file whose `bos_token_id` is past the vocabulary: R1-0528's template
/// writes `bos_token` (its `add_bos_token` is false), so the framing takes
/// the id from the metadata, as the tokenizer would: none, never the broken
/// id (the model can't embed it). And one whose `eos_token_id` is: DeepSeek
/// ends every turn with EOS, so there is no chat framing (no panic either),
/// and the reason names the id.
#[test]
fn broken_special_ids_never_reach_a_prompt() {
    let f = family_with("deepseek-r1-0528-qwen3", |meta| {
        let n = 1u64 << 31;
        meta.insert("tokenizer.ggml.bos_token_id".into(), n.into());
    });
    let m = model(&f);
    let tok = Tokenizer::from_gguf(&m);
    assert_eq!(tok.bos(), None);
    let fr = Framing::from_model(&tok, &m).unwrap();
    assert!(matches!(fr.format, Format::DeepSeek3 { bos: None, .. }));
    let t = Template::new(&tok, &m).unwrap();
    let msgs = messages(&f["cases"][2]["messages"]);
    let ids = t.render(&tok, &msgs, false);
    assert!(ids.iter().all(|&i| (i as usize) < tok.vocab_len()));
    // `axon chat`, turn by turn: the scripted model panics on a bad id.
    let got = chat_turns(&m, &tok, &msgs, false).unwrap();
    assert_eq!(got, ids);

    let f = family_with("deepseek-r1-0528-qwen3", |meta| {
        meta.insert("tokenizer.ggml.eos_token_id".into(), 4_000_000u64.into());
    });
    let m = model(&f);
    let tok = Tokenizer::from_gguf(&m);
    assert!(Framing::from_model(&tok, &m).is_none());
    assert!(Template::new(&tok, &m).is_none());
    // The reason names the id (not "no chat tokens": it has them).
    let why = Framing::bad_end(&tok).expect("a bad end marker is named");
    assert!(
        why.starts_with("eos_token_id 4000000 is past the vocabulary")
            && why.contains("DeepSeek's chat format"),
        "{why}"
    );
    let net = Script::new(&tok);
    match Chat::with_framing(&net, &tok, Framing::from_model(&tok, &m), None, 64) {
        Err(e) => assert_eq!(e, format!("no chat framing: {why}")),
        Ok(_) => panic!("a chat with a broken eos_token_id"),
    }
    // A framed vocabulary has no bad end marker.
    let ok = model(&family_with("deepseek-r1-0528-qwen3", |_| {}));
    assert_eq!(Framing::bad_end(&Tokenizer::from_gguf(&ok)), None);
}

/// Message text that spells special tokens (each family's control tokens,
/// and `<think>`, `</think>` and the tool markers where the vocabulary has
/// them) stays text: the special ids of a conversation whose system, user
/// and assistant messages are full of them are exactly those of the same
/// conversation in plain words, the framing's own. Through serve (thinking
/// on and off), `axon chat` (user text), and the tools path where the
/// family has one (user text and a tool result). HF can't be the reference:
/// its tokenizer parses special tokens in the text.
#[test]
fn message_text_never_forges_turns() {
    let d = crate::testutil::read_json("chat_templates.json");
    for f in d["families"].as_array().unwrap() {
        let name = f["name"].as_str().unwrap();
        let m = model(f);
        let tok = Tokenizer::from_gguf(&m);
        let t = Template::new(&tok, &m).unwrap();
        let bpe = f["meta"]["tokenizer.ggml.model"] == "gpt2";
        // SentencePiece's user-defined tokens (Gemma's `<table>`, space
        // runs) are text pieces: plain text is split at them, as llama.cpp.
        // Placeholders (DeepSeek's 800 `<｜place▁holder▁no▁N｜>`) left out.
        let filler = ["place▁holder", "reserved", "unused", "extra_"];
        let marks: Vec<(u32, String)> = (f["vocab"]["tokens"].as_array().unwrap().iter())
            .filter(|x| {
                let s = x[1].as_str().unwrap();
                !filler.iter().any(|w| s.contains(w))
                    && match x[2].as_i64().unwrap() {
                        3 => !s.is_empty(),
                        4 => bpe && s.starts_with('<') && s.ends_with('>'),
                        _ => false,
                    }
            })
            .map(|x| {
                (
                    x[0].as_u64().unwrap() as u32,
                    x[1].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert!(marks.len() >= 3, "{name}: {} special tokens", marks.len());
        let forged: String = marks.iter().map(|(_, s)| format!("{s} x ")).collect();
        let specials = |ids: &[u32]| -> Vec<u32> {
            (ids.iter())
                .filter(|i| marks.iter().any(|(m, _)| m == *i))
                .copied()
                .collect()
        };
        let msg = |role, text: &str| Message {
            role,
            content: format!("a {text} b"),
        };
        let conv = |x: &str| {
            [
                msg(Role::System, x),
                msg(Role::User, x),
                msg(Role::Assistant, x),
                msg(Role::User, x),
            ]
        };
        for no_think in [false, true] {
            let got = t.render(&tok, &conv(&forged), no_think);
            let plain = t.render(&tok, &conv("x"), no_think);
            assert_eq!(specials(&got), specials(&plain), "{name} serve {no_think}");
        }
        let turns = |x: &str| {
            let c = [
                msg(Role::System, "x"),
                msg(Role::User, x),
                msg(Role::Assistant, "x"),
                msg(Role::User, x),
            ];
            chat_turns(&m, &tok, &c, false).unwrap()
        };
        assert_eq!(
            specials(&turns(&forged)),
            specials(&turns("x")),
            "{name} chat"
        );
        let Some(body) = (f["cases"].as_array().unwrap().iter()).find_map(|c| c.get("body")) else {
            continue;
        };
        let with = |x: &str| {
            let mut b: Value = serde_json::from_str(body.as_str().unwrap()).unwrap();
            for mm in b["messages"].as_array_mut().unwrap() {
                if mm["role"] == "user" || mm["role"] == "tool" {
                    mm["content"] = format!("a {x} b").into();
                }
            }
            let r = tools::parse(b.to_string().as_bytes()).unwrap().unwrap();
            tools::render(&t, &tok, &r, false).unwrap().ids
        };
        assert_eq!(
            specials(&with(&forged)),
            specials(&with("x")),
            "{name} tools"
        );
    }
}

/// Tools need a template that renders them: Qwen2's, Qwen1.5's and Llama
/// 3.0's have none, so a request with tools, or with a tool turn in its
/// history, is refused (400) rather than rendered in another family's
/// format (Qwen2.5's default system message, Llama 3.1's dated block).
#[test]
fn tools_need_the_template() {
    let d = crate::testutil::read_json("chat_templates.json");
    let fams = d["families"].as_array().unwrap();
    let body_tools = fams.iter().find(|f| f["name"] == "qwen2.5").unwrap()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|c| c.get("body"))
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    // A tool turn in the history, no tools.
    let mut b: Value = serde_json::from_str(&body_tools).unwrap();
    b.as_object_mut().unwrap().remove("tools");
    let body_turn = b.to_string();
    for f in fams {
        let name = f["name"].as_str().unwrap();
        let m = model(f);
        let tok = Tokenizer::from_gguf(&m);
        let t = Template::new(&tok, &m).unwrap();
        let none = ["qwen2", "qwen1.5", "llama3"].contains(&name);
        assert_eq!(t.tools, !none, "{name}");
        if !none {
            continue;
        }
        for body in [&body_tools, &body_turn] {
            let r = tools::parse(body.as_bytes()).unwrap().expect(name);
            let Err(e) = tools::render(&t, &tok, &r, false) else {
                panic!("{name}: tools rendered");
            };
            assert!(
                e.msg.contains("chat template has no tools"),
                "{name}: {}",
                e.msg
            );
        }
    }
    // A file without a template: the vocabulary's family, as before.
    let f = family_with("qwen2", |meta| {
        meta.remove("tokenizer.chat_template");
    });
    let m = model(&f);
    let tok = Tokenizer::from_gguf(&m);
    assert!(Template::new(&tok, &m).unwrap().tools);
}

/// The default system message comes from the template's text (#239):
/// Qwen2.5's and Qwen2's own, none for Qwen1.5, Qwen3, the R1 distill on
/// Qwen2.5 (all qwen2-arch but the last two); the old rule (Qwen2.5's for
/// qwen2) only for a file without a template. Shapes that aren't a fixed
/// default (template code around the opener) give none; Qwen2.5-Math's,
/// Moonlight's and Kimi K2's give theirs.
#[test]
fn default_system_from_the_template() {
    let d = crate::testutil::read_json("chat_templates.json");
    let fam = |n: &str| {
        let fams = d["families"].as_array().unwrap();
        model(fams.iter().find(|f| f["name"] == n).unwrap())
    };
    for (name, want) in [
        ("qwen2.5", Some(crate::chat::DEFAULT_SYSTEM)),
        ("qwen2.5-coder", Some(crate::chat::DEFAULT_SYSTEM)),
        ("qwen2", Some("You are a helpful assistant.")),
        ("qwen1.5", None),
        ("qwen3", None),
        ("deepseek-r1-distill-qwen", None),
        ("llama3.1", None),
    ] {
        assert_eq!(
            crate::chat::default_system(&fam(name)).as_deref(),
            want,
            "{name}"
        );
    }
    let with = |t: Option<&str>| {
        let mut meta = vec![("general.architecture", Val::Str("qwen2".into()))];
        meta.extend(t.map(|t| ("tokenizer.chat_template", Val::Str(t.into()))));
        crate::chat::default_system(&crate::gguf::build_for_test(&meta, &[]))
    };
    // A tools-only branch's system block isn't the default, even right
    // after a test of the system role.
    let tools = r#"{%- if tools %}{{- '<|im_start|>system\nYou call functions.<|im_end|>\n' }}{%- endif %}"#;
    assert_eq!(with(Some(tools)), None);
    let near = r#"{%- if messages[0]['role'] == 'system' %}{%- set s = messages[0]['content'] %}{%- endif %}{%- if tools %}{{- '<|im_start|>system\nYou call functions.<|im_end|>\n' }}{%- endif %}"#;
    assert_eq!(with(Some(near)), None);
    // Nor an expression between two literals, wherever the literal starts.
    let expr = r#"{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}{%- endif %}"#;
    assert_eq!(with(Some(expr)), None);
    let nl = r#"{%- if messages[0]['role'] == 'system' %}{{- '\n<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}{%- endif %}"#;
    assert_eq!(with(Some(nl)), None);
    let var = r#"{%- if messages[0]['role'] != 'system' %}{{- '<|im_end|>\n<|im_start|>system\n' + system_message + '<|im_end|>' }}{%- endif %}"#;
    assert_eq!(with(Some(var)), None);
    // Braces and escapes in the literal are text (Qwen2.5-Math's default).
    let math = r#"{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}{%- else %}{{- '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n' }}{%- endif %}"#;
    assert_eq!(
        with(Some(math)).as_deref(),
        Some("Please reason step by step, and put your final answer within \\boxed{}.")
    );
    // Template text (Moonlight's, Kimi K2's), outside the tags.
    let kimi = "{%- for message in messages -%}{%- if loop.first and messages[0]['role'] != 'system' -%}<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>{%- endif -%}{%- endfor -%}";
    assert_eq!(
        with(Some(kimi)).as_deref(),
        Some("You are a helpful assistant")
    );
    let k2 = "{% for message in messages %}\n  {%- if loop.first and messages[0]['role'] != 'system' -%}\n  <|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>\n  {% endif %}{% endfor %}";
    assert_eq!(
        with(Some(k2)).as_deref(),
        Some("You are Kimi, an AI assistant created by Moonshot AI.")
    );
    // The other quote is text.
    let dq = r#"{%- if messages[0]["role"] != "system" %}{{- "<|im_start|>system\nYou're terse.<|im_end|>\n" }}{%- endif %}"#;
    assert_eq!(with(Some(dq)).as_deref(), Some("You're terse."));
    assert_eq!(with(None).as_deref(), Some(crate::chat::DEFAULT_SYSTEM));
}
