//! Chat prompts, token for token, against each GGUF's own template.
//!
//! tests/fixtures/chat_templates.json (scripts/chat_template_fixtures.py)
//! holds, per chat family, Hugging Face's ids for a set of conversations:
//! `apply_chat_template(messages, chat_template=<the GGUF's>,
//! add_generation_prompt=True)` tokenized by the model's tokenizer.json. One
//! turn, multi-turn, with and without a system message, an earlier reply
//! with its reasoning (thinking models), and tool turns where the template
//! has them. It also carries each file's tokenizer metadata (its
//! `tokenizer.chat_template` among them) and a sparse vocabulary: every
//! token these prompts can use. So the test needs no model file and no
//! network.

use super::api::{Message, Role};
use super::{Template, tools};
use crate::gguf::{Model, Val};
use crate::tokenizer::Tokenizer;
use serde_json::Value;

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

/// Every family's every conversation renders to HF's ids. All differences
/// are listed before the test fails, the decoded text beside the template's.
#[test]
fn chat_templates_match_hf() {
    let d = crate::testutil::read_json("chat_templates.json");
    let fams = d["families"].as_array().unwrap();
    assert!(fams.len() >= 14, "{} families", fams.len());
    let (mut bad, mut n) = (Vec::new(), 0);
    for f in fams {
        let name = f["name"].as_str().unwrap();
        let m = model(f);
        let tok = Tokenizer::from_gguf(&m);
        let t = Template::new(&tok, &m).unwrap_or_else(|| panic!("{name}: no chat framing"));
        for c in f["cases"].as_array().unwrap() {
            let case = format!("{name}/{}", c["name"].as_str().unwrap());
            let (want, text) = (ids(&c["ids"]), c["text"].as_str().unwrap());
            // The sparse vocabulary tokenizes the template's text as HF's
            // tokenizer did: what follows is about the framing alone.
            assert_eq!(tok.encode_special(text), want, "{case}: fixture vocabulary");
            let got = match c.get("body") {
                // The request's bytes (re-serializing would reorder keys).
                Some(b) => {
                    let b = b.as_str().unwrap().as_bytes();
                    let r = tools::parse(b).unwrap().expect(&case);
                    let r = tools::render(&t, &tok, &r, false);
                    r.unwrap_or_else(|e| panic!("{case}: {}", e.msg)).ids
                }
                None => t.render(&tok, &messages(&c["messages"]), false),
            };
            n += 1;
            if got != want {
                bad.push(format!(
                    "{case}: {} vs {} tokens\n  axon: {:?}\n  hf:   {text:?}",
                    got.len(),
                    want.len(),
                    tok.decode(&got)
                ));
            }
        }
    }
    assert!(n >= 68, "{n} cases");
    assert!(
        bad.is_empty(),
        "{} of {n} cases differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// The default system message comes from the template's text (#239):
/// Qwen2.5's and Qwen2's own, none for Qwen1.5, Qwen3, the R1 distill on
/// Qwen2.5 (all qwen2-arch but the last two); the old rule (Qwen2.5's for
/// qwen2) only for a file without a template.
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
    // A tools-only branch's system block isn't the default.
    let tools = r#"{%- if tools %}{{- '<|im_start|>system\nYou call functions.<|im_end|>\n' }}{%- endif %}"#;
    assert_eq!(with(Some(tools)), None);
    // Nor an expression between two literals.
    let expr = r#"{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}{%- endif %}"#;
    assert_eq!(with(Some(expr)), None);
    // The other quote is text.
    let dq = r#"{%- if messages[0]["role"] != "system" %}{{- "<|im_start|>system\nYou're terse.<|im_end|>\n" }}{%- endif %}"#;
    assert_eq!(with(Some(dq)).as_deref(), Some("You're terse."));
    assert_eq!(with(None).as_deref(), Some(crate::chat::DEFAULT_SYSTEM));
}
