//! GPT-2 byte-level BPE (llama.cpp `gpt2` model, `qwen2` pre-tokenizer).

use crate::gguf::{Model, Val};
use fancy_regex::Regex;
use std::collections::HashMap;

// llama.cpp's expanded form of `(?i:'s|'t|...)` — avoids Unicode case folding (ſ -> s, K -> k).
const QWEN2_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;

pub struct Tokenizer {
    tokens: Vec<String>,
    token_type: Vec<i32>,
    #[allow(dead_code)]
    token_to_id: HashMap<String, u32>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// raw byte -> id of its single-char byte-encoded token
    byte_id: [Option<u32>; 256],
    /// unicode char of the byte encoding -> raw byte
    byte_decoder: HashMap<char, u8>,
    /// (text, id), longest text first
    specials: Vec<(String, u32)>,
    re: Regex,
    eos: u32,
}

/// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100.. in order.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    table
}

fn meta<'a>(m: &'a Model, key: &str) -> Option<&'a Val> {
    m.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn val_i64(v: &Val) -> i64 {
    match v {
        Val::U8(x) => *x as i64,
        Val::I8(x) => *x as i64,
        Val::U16(x) => *x as i64,
        Val::I16(x) => *x as i64,
        Val::U32(x) => *x as i64,
        Val::I32(x) => *x as i64,
        Val::U64(x) => *x as i64,
        Val::I64(x) => *x,
        Val::Bool(x) => *x as i64,
        _ => 0,
    }
}

impl Tokenizer {
    pub fn from_gguf(m: &Model) -> Tokenizer {
        let tokens: Vec<String> = match meta(m, "tokenizer.ggml.tokens") {
            Some(Val::Arr(a)) => a
                .iter()
                .map(|v| match v {
                    Val::Str(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => panic!("tokenizer.ggml.tokens missing"),
        };
        let token_type: Vec<i32> = match meta(m, "tokenizer.ggml.token_type") {
            Some(Val::Arr(a)) => a.iter().map(|v| val_i64(v) as i32).collect(),
            _ => vec![1; tokens.len()],
        };

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let mut merges = HashMap::new();
        if let Some(Val::Arr(a)) = meta(m, "tokenizer.ggml.merges") {
            merges.reserve(a.len());
            let mut buf = String::new();
            for (rank, v) in a.iter().enumerate() {
                let Val::Str(s) = v else { continue };
                let Some((l, r)) = s.split_once(' ') else {
                    continue;
                };
                let (Some(&li), Some(&ri)) = (token_to_id.get(l), token_to_id.get(r)) else {
                    continue;
                };
                buf.clear();
                buf.push_str(l);
                buf.push_str(r);
                let Some(&mi) = token_to_id.get(buf.as_str()) else {
                    continue;
                };
                merges.entry((li, ri)).or_insert((rank as u32, mi));
            }
        }

        let enc = bytes_to_unicode();
        let mut byte_id = [None; 256];
        let mut byte_decoder = HashMap::with_capacity(256);
        let mut cbuf = [0u8; 4];
        for b in 0..256 {
            byte_decoder.insert(enc[b], b as u8);
            byte_id[b] = token_to_id.get(&*enc[b].encode_utf8(&mut cbuf)).copied();
        }

        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                let ty = token_type.get(*i).copied().unwrap_or(1);
                !t.is_empty() && (ty == TYPE_CONTROL || ty == TYPE_USER_DEFINED)
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        let eos = meta(m, "tokenizer.ggml.eos_token_id")
            .map(|v| val_i64(v) as u32)
            .unwrap_or(151645);

        Tokenizer {
            tokens,
            token_type,
            token_to_id,
            merges,
            byte_id,
            byte_decoder,
            specials,
            re: Regex::new(QWEN2_PATTERN).expect("qwen2 regex"),
            eos,
        }
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for frag in self.split_specials(text) {
            match frag {
                Frag::Special(id) => out.push(id),
                Frag::Text(s) => self.encode_text(s, &mut out),
            }
        }
        out
    }

    /// Split out special tokens the way llama.cpp does: for each special token
    /// (longest first), cut every remaining raw-text fragment at its occurrences.
    fn split_specials<'t>(&self, text: &'t str) -> Vec<Frag<'t>> {
        let mut frags = vec![Frag::Text(text)];
        for (st, id) in &self.specials {
            if !text.contains(st.as_str()) {
                continue;
            }
            let mut next = Vec::with_capacity(frags.len() + 2);
            for f in frags {
                match f {
                    Frag::Text(s) => {
                        let mut rest = s;
                        while let Some(p) = rest.find(st.as_str()) {
                            if p > 0 {
                                next.push(Frag::Text(&rest[..p]));
                            }
                            next.push(Frag::Special(*id));
                            rest = &rest[p + st.len()..];
                        }
                        if !rest.is_empty() {
                            next.push(Frag::Text(rest));
                        }
                    }
                    sp => next.push(sp),
                }
            }
            frags = next;
        }
        frags
    }

    fn encode_text(&self, text: &str, out: &mut Vec<u32>) {
        let mut last = 0;
        for m in self.re.find_iter(text) {
            match m {
                Ok(m) => {
                    if m.start() > last {
                        self.bpe_word(&text[last..m.start()], out);
                    }
                    if m.end() > m.start() {
                        self.bpe_word(m.as_str(), out);
                    }
                    last = m.end();
                }
                Err(_) => break, // backtrack limit etc.: treat the remainder as one word
            }
        }
        if last < text.len() {
            self.bpe_word(&text[last..], out);
        }
    }

    /// Byte-level BPE on one pre-token: repeatedly merge the lowest-rank adjacent pair (leftmost on ties).
    fn bpe_word(&self, word: &str, out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = word
            .bytes()
            .filter_map(|b| self.byte_id[b as usize])
            .collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None; // (rank, pos, merged id)
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, mid)) = self.merges.get(&(syms[i], syms[i + 1]))
                    && best.is_none_or(|(r, _, _)| rank < r)
                {
                    best = Some((rank, i, mid));
                }
            }
            let Some((_, i, mid)) = best else { break };
            syms[i] = mid;
            syms.remove(i + 1);
        }
        out.extend_from_slice(&syms);
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(t) = self.tokens.get(id as usize) else {
                continue;
            };
            let ty = self.token_type.get(id as usize).copied().unwrap_or(1);
            if ty == TYPE_CONTROL || ty == TYPE_USER_DEFINED {
                bytes.extend_from_slice(t.as_bytes());
                continue;
            }
            for c in t.chars() {
                match self.byte_decoder.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

enum Frag<'t> {
    Text(&'t str),
    Special(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Option<Tokenizer> {
        crate::testutil::model().map(|m| Tokenizer::from_gguf(&m))
    }

    #[test]
    fn byte_table_roundtrip() {
        let t = bytes_to_unicode();
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
        let mut seen = std::collections::HashSet::new();
        for c in t {
            assert!(seen.insert(c));
        }
    }

    #[test]
    fn roundtrip() {
        let Some(t) = tok() else { return };
        assert_eq!(t.eos(), 151645);
        for s in [
            "Hello, world!",
            "The capital of France is",
            "  leading spaces\n\nnewlines",
            "naïve café 東京 🚀",
            "<|im_start|>user\nhi<|im_end|>",
        ] {
            let ids = t.encode(s);
            assert!(!ids.is_empty());
            assert_eq!(t.decode(&ids), s, "ids {ids:?}");
        }
    }

    #[test]
    fn specials_single_ids() {
        let Some(t) = tok() else { return };
        let ids = t.encode("<|im_start|>user\nhi<|im_end|>");
        assert_eq!(ids.first(), Some(&151644));
        assert_eq!(ids.last(), Some(&151645));
        assert_eq!(t.encode("<|endoftext|>"), vec![151643]);
    }

    /// Token ids must match Hugging Face's tokenizer exactly (tests/fixtures/tokens.json).
    #[test]
    fn oracle() {
        let cases = crate::testutil::read_json("tokens.json");
        let cases = cases.as_array().expect("array");
        assert!(!cases.is_empty());
        let Some(t) = tok() else { return };
        for c in cases {
            let text = c["text"].as_str().expect("text");
            let want: Vec<u32> = c["ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(t.encode(text), want, "text {text:?}");
        }
    }
}
