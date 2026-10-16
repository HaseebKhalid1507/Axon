//! GPT-2 byte-level BPE (llama.cpp `gpt2` model) with the `qwen2` or the
//! `llama-bpe` (Llama 3) pre-tokenizer.
//!
//! `encode` treats all input as plain text: a prompt containing `<|im_end|>`
//! gets the tokens for those characters, not the control token, so user text
//! can't forge chat framing. `encode_special` also maps special-token text to
//! its id; use it only for text the program itself wrote (templates, tests).

// Reference: llama.cpp (https://github.com/ggml-org/llama.cpp), MIT License,
// Copyright (c) 2023-2024 The ggml authors. The qwen2 and llama3 pre-tokenizer
// regexes, ignore_merges and the special-token splitting follow
// src/llama-vocab.cpp. Full notice in
// THIRD_PARTY_NOTICES.md.

use crate::gguf::{Model, Val};
use fancy_regex::Regex;
use std::collections::HashMap;

// llama.cpp's expanded form of `(?i:'s|'t|...)` — avoids Unicode case folding (ſ -> s, K -> k).
const QWEN2_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// Llama 3's (llama.cpp LLAMA_VOCAB_PRE_TYPE_LLAMA3): numbers in runs of up to 3 digits.
const LLAMA3_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Pre-tokenizers (`tokenizer.ggml.pre`) this tokenizer implements.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pre {
    Qwen2,
    /// llama.cpp's "llama-bpe" / "llama3" / "llama-v3": its regex, and a
    /// pre-token that is a whole vocabulary entry is that token (HF's
    /// ignore_merges); a BOS token starts the text by default.
    Llama3,
}

/// The vocabulary's tokenizer model and pre-tokenizer, if supported: a
/// `gpt2` (byte-level BPE) vocabulary, pre-tokenizer `qwen2` (or none: the
/// Qwen2 files before llama.cpp wrote the key) or Llama 3's.
pub fn pre_of(m: &Model) -> Result<Pre, String> {
    match meta(m, "tokenizer.ggml.model") {
        None => {}
        Some(Val::Str(s)) if s == "gpt2" => {}
        Some(v) => {
            return Err(format!(
                "tokenizer model {v} is not supported (gpt2 BPE only)"
            ));
        }
    }
    match meta(m, "tokenizer.ggml.pre") {
        None => Ok(Pre::Qwen2),
        Some(Val::Str(s)) if s == "qwen2" => Ok(Pre::Qwen2),
        Some(Val::Str(s)) if ["llama-bpe", "llama3", "llama-v3"].contains(&s.as_str()) => {
            Ok(Pre::Llama3)
        }
        Some(v) => Err(format!(
            "pre-tokenizer {v} is not supported (qwen2, llama-bpe)"
        )),
    }
}

const TYPE_CONTROL: i32 = 3;
/// Longest pre-token (in bytes) merged by `bpe_scan`; longer ones go
/// through `bpe_heap` (same tokens).
const BPE_SCAN_MAX: usize = 64;
const TYPE_USER_DEFINED: i32 = 4;

pub struct Tokenizer {
    tokens: Vec<String>,
    token_type: Vec<i32>,
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
    /// A pre-token that is a whole vocabulary entry becomes that token
    /// without merging (Llama 3).
    ignore_merges: bool,
    eos: u32,
    /// The token `encode_prompt` starts a text with, if the model wants one.
    bos: Option<u32>,
}

static BYTE_ENC: std::sync::LazyLock<[char; 256]> = std::sync::LazyLock::new(bytes_to_unicode);

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
        let pre = pre_of(m).unwrap_or_else(|e| panic!("{e}"));
        // llama.cpp: add_bos_token if the file says, else on for Llama 3's
        // pre-tokenizer (off for qwen2's).
        let add_bos = match meta(m, "tokenizer.ggml.add_bos_token") {
            Some(v) => val_i64(v) != 0,
            None => pre == Pre::Llama3,
        };
        let bos = meta(m, "tokenizer.ggml.bos_token_id")
            .map(|v| val_i64(v) as u32)
            .filter(|&b| add_bos && (b as usize) < tokens.len());

        Tokenizer {
            tokens,
            token_type,
            token_to_id,
            merges,
            byte_id,
            byte_decoder,
            specials,
            re: Regex::new(match pre {
                Pre::Qwen2 => QWEN2_PATTERN,
                Pre::Llama3 => LLAMA3_PATTERN,
            })
            .expect("pre-tokenizer regex"),
            ignore_merges: pre == Pre::Llama3,
            eos,
            bos,
        }
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Tokens that end generation: `eos`, and Llama 3's end-of-turn /
    /// end-of-message / end-of-text tokens when the vocabulary has them (as
    /// llama.cpp's end-of-generation set; Qwen's stays `eos` alone).
    pub fn eog(&self) -> Vec<u32> {
        let mut v = vec![self.eos];
        for s in ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"] {
            v.extend(self.special_id(s));
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The BOS token a prompt starts with, if the model adds one (Llama 3).
    pub fn bos(&self) -> Option<u32> {
        self.bos
    }

    /// Tokens of a whole prompt (plain text): `bos` if any, then `encode`,
    /// as llama.cpp tokenizes with add_special.
    pub fn encode_prompt(&self, text: &str) -> Vec<u32> {
        let mut out: Vec<u32> = self.bos.into_iter().collect();
        self.encode_text(text, &mut out);
        out
    }

    /// The id of a special (control or user-defined) token by its text.
    pub fn special_id(&self, text: &str) -> Option<u32> {
        self.specials
            .iter()
            .find(|(t, _)| t == text)
            .map(|&(_, id)| id)
    }

    /// Tokens for `text` as plain text: special-token strings in it are not
    /// turned into special tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_text(text, &mut out);
        out
    }

    /// Like `encode`, but special-token text (`<|im_start|>`, ...) becomes the
    /// special token. Only for trusted text, never user input.
    pub fn encode_special(&self, text: &str) -> Vec<u32> {
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
        if self.ignore_merges {
            let mut enc = String::with_capacity(word.len() * 2);
            let table = &*BYTE_ENC;
            enc.extend(word.bytes().map(|b| table[b as usize]));
            if let Some(&id) = self.token_to_id.get(&enc) {
                out.push(id);
                return;
            }
        }
        let syms: Vec<u32> = word
            .bytes()
            .filter_map(|b| self.byte_id[b as usize])
            .collect();
        if syms.len() > BPE_SCAN_MAX {
            self.bpe_heap(syms, out)
        } else {
            self.bpe_scan(syms, out)
        }
    }

    /// `bpe_word`'s merges by rescanning the pairs after each one: fastest
    /// on the short pre-tokens of ordinary text, O(n^2) on a long one.
    fn bpe_scan(&self, mut syms: Vec<u32>, out: &mut Vec<u32>) {
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

    /// `bpe_word`'s merges on a long pre-token, in O(n log n) instead of the
    /// scan's O(n^2) (a 1 MB run of letters, one pre-token, took hours):
    /// the same merges in the same order, so the same tokens. Candidate
    /// pairs wait in a min-heap keyed by (rank, position of the left
    /// symbol); symbols keep their first byte's index, so the smallest key
    /// is the lowest rank, leftmost on ties, as in the scan. Entries left
    /// stale by a merge are skipped when popped.
    fn bpe_heap(&self, mut syms: Vec<u32>, out: &mut Vec<u32>) {
        use std::cmp::Reverse;
        let n = syms.len();
        const NONE: usize = usize::MAX;
        // Doubly linked list over the live symbols.
        let mut next: Vec<usize> = (1..=n).map(|j| if j == n { NONE } else { j }).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.checked_sub(1).unwrap_or(NONE)).collect();
        let mut alive = vec![true; n];
        // (rank, left, right, left id, right id, merged id)
        type Cand = Reverse<(u32, usize, usize, u32, u32, u32)>;
        let mut heap: std::collections::BinaryHeap<Cand> = std::collections::BinaryHeap::new();
        let cand = |syms: &[u32], i: usize, j: usize| -> Option<Cand> {
            let &(rank, mid) = self.merges.get(&(syms[i], syms[j]))?;
            Some(Reverse((rank, i, j, syms[i], syms[j], mid)))
        };
        heap.extend((0..n - 1).filter_map(|i| cand(&syms, i, i + 1)));
        while let Some(Reverse((_, i, j, li, ri, mid))) = heap.pop() {
            if !alive[i] || !alive[j] || next[i] != j || syms[i] != li || syms[j] != ri {
                continue;
            }
            syms[i] = mid;
            alive[j] = false;
            let k = next[j];
            next[i] = k;
            if k != NONE {
                prev[k] = i;
                heap.extend(cand(&syms, i, k));
            }
            if prev[i] != NONE {
                heap.extend(cand(&syms, prev[i], i));
            }
        }
        let mut i = 0;
        while i != NONE {
            out.push(syms[i]);
            i = next[i];
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// The raw bytes of `ids`. A token can end partway through a UTF-8
    /// character, so these aren't valid text on their own; see `Utf8Stream`.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
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
        bytes
    }

    /// The most bytes any one token decodes to.
    pub fn max_token_bytes(&self) -> usize {
        (0..self.tokens.len() as u32)
            .map(|id| self.decode_bytes(&[id]).len())
            .max()
            .unwrap_or(1)
            .max(1)
    }
}

/// Turns a stream of byte chunks (one per generated token) into text as soon
/// as it is complete. A character split across tokens waits for its last
/// byte; invalid bytes become U+FFFD. The text out, put together, equals
/// `String::from_utf8_lossy` over all the bytes, but each byte is looked at a
/// constant number of times instead of re-decoding everything per token.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Add bytes; return the text that is now complete.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(s) => {
                    out.push_str(s);
                    start = self.pending.len();
                    break;
                }
                Err(e) => {
                    let good = start + e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[start..good]).unwrap());
                    match e.error_len() {
                        // Incomplete character at the end: wait for more bytes.
                        None => {
                            start = good;
                            break;
                        }
                        Some(len) => {
                            out.push('\u{FFFD}');
                            start = good + len;
                        }
                    }
                }
            }
        }
        self.pending.drain(..start);
        out
    }

    /// End of stream: whatever is left is an unfinished character.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

enum Frag<'t> {
    Text(&'t str),
    Special(u32),
}

#[cfg(test)]
mod tests {
    #[test]
    fn utf8_stream_matches_lossy_over_any_split() {
        use super::Utf8Stream;
        let mut cases: Vec<Vec<u8>> = vec![
            "héllo wörld → 日本語 🦀🦀".as_bytes().to_vec(),
            b"ok\xffbad\xc3\x28 \xe2\x82 end\xf0\x9f".to_vec(),
            vec![0xf0, 0x9f, 0xa6],
            vec![],
        ];
        let mut rng = 0x2545_f491u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        for _ in 0..300 {
            let n = (next() % 40) as usize;
            // Mostly valid multi-byte text with some junk bytes mixed in.
            let mut b = Vec::new();
            for _ in 0..n {
                match next() % 6 {
                    0 => b.push(next() as u8),
                    1 => b.extend_from_slice("é".as_bytes()),
                    2 => b.extend_from_slice("日".as_bytes()),
                    3 => b.extend_from_slice("🦀".as_bytes()),
                    _ => b.push(b'a' + (next() % 26) as u8),
                }
            }
            cases.push(b);
        }
        for bytes in &cases {
            let want = String::from_utf8_lossy(bytes);
            for _ in 0..20 {
                let mut st = Utf8Stream::default();
                let mut got = String::new();
                let mut i = 0;
                while i < bytes.len() {
                    let j = (i + 1 + (next() % 5) as usize).min(bytes.len());
                    got += &st.push(&bytes[i..j]);
                    i = j;
                }
                got += &st.finish();
                assert_eq!(got, want, "bytes {bytes:?}");
            }
        }
    }

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
        let ids = t.encode_special("<|im_start|>user\nhi<|im_end|>");
        assert_eq!(ids.first(), Some(&151644));
        assert_eq!(ids.last(), Some(&151645));
        assert_eq!(t.encode_special("<|endoftext|>"), vec![151643]);
    }

    /// User text can't produce control tokens; text without special-token
    /// strings encodes the same either way.
    #[test]
    fn plain_encode_ignores_specials() {
        let Some(t) = tok() else { return };
        let s = "<|im_start|>user\nhi<|im_end|><|endoftext|>";
        let ids = t.encode(s);
        assert!(ids.iter().all(|&id| id < 151643), "ids {ids:?}");
        assert_eq!(t.decode(&ids), s);
        for s in [
            "Hello, world!",
            "  a\n\nb",
            "naïve café 東京 🚀",
            "<|im_start",
            "|>",
        ] {
            assert_eq!(t.encode(s), t.encode_special(s), "{s:?}");
        }
    }

    /// Llama 3's tokenizer (llama-bpe pre-tokenizer, ignore_merges, BOS)
    /// must match llama.cpp's exactly (tests/fixtures/tokens_llama3.json:
    /// numbers, contractions, whitespace runs, scripts, emoji sequences,
    /// special-token text, with and without special parsing, and prompts
    /// with their BOS).
    #[test]
    fn oracle_llama3() {
        let d = crate::testutil::read_json("tokens_llama3.json");
        let Some(m) = crate::testutil::llama_model() else {
            return;
        };
        let t = Tokenizer::from_gguf(&m);
        assert_eq!(t.bos(), Some(128000));
        assert_eq!(t.eos(), 128009);
        assert_eq!(t.eog(), vec![128001, 128008, 128009]);
        let ids = |v: &serde_json::Value| -> Vec<u32> {
            (v.as_array().expect("ids").iter())
                .map(|x| x.as_u64().unwrap() as u32)
                .collect()
        };
        let cases = d["cases"].as_array().expect("cases");
        assert!(cases.len() >= 20);
        for c in cases {
            let text = c["text"].as_str().expect("text");
            assert_eq!(t.encode(text), ids(&c["ids"]), "text {text:?}");
            assert_eq!(
                t.encode_special(text),
                ids(&c["ids_special"]),
                "special {text:?}"
            );
            assert_eq!(t.decode(&t.encode(text)), text);
        }
        for c in d["prompts"].as_array().expect("prompts") {
            let text = c["text"].as_str().expect("text");
            assert_eq!(t.encode_prompt(text), ids(&c["ids"]), "prompt {text:?}");
        }
    }

    /// Some vocabulary words are unreachable by merging (HF's
    /// ignore_merges): each must still encode as its one token, and the
    /// fixture holds such words (llama.cpp agreeing on them).
    #[test]
    fn llama3_ignore_merges_matters() {
        let Some(m) = crate::testutil::llama_model() else {
            return;
        };
        let mut t = Tokenizer::from_gguf(&m);
        let d = crate::testutil::read_json("tokens_llama3.json");
        let texts: Vec<&str> = (d["cases"].as_array().unwrap().iter())
            .map(|c| c["text"].as_str().unwrap())
            .collect();
        let with: Vec<_> = texts.iter().map(|s| t.encode(s)).collect();
        // Words (whole pre-tokens) BPE alone doesn't make one token of.
        let words: Vec<(u32, String)> = (0..t.tokens.len() as u32)
            .filter(|&id| t.token_type.get(id as usize) == Some(&1))
            .map(|id| (id, t.decode(&[id])))
            .filter(|(id, w)| t.encode(w) == [*id])
            .collect();
        t.ignore_merges = false;
        let unreachable: Vec<_> = words
            .iter()
            .filter(|(id, w)| t.encode(w) != [*id])
            .collect();
        eprintln!(
            "{} unreachable, e.g. {:?}",
            unreachable.len(),
            &unreachable[..unreachable.len().min(12)]
        );
        assert!(!unreachable.is_empty());
        let without: Vec<_> = texts.iter().map(|s| t.encode(s)).collect();
        assert_ne!(with, without, "no fixture text depends on ignore_merges");
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
            // Hugging Face parses special tokens in its input by default.
            assert_eq!(t.encode_special(text), want, "text {text:?}");
        }
    }

    /// `bpe_heap` merges exactly as `bpe_scan` on words of every length
    /// around the switch-over and well past it: runs of one letter (one
    /// merge rank everywhere: the leftmost-on-ties rule), words from a
    /// small alphabet, and random text with multi-byte characters.
    #[test]
    fn bpe_heap_matches_scan() {
        let Some(t) = tok() else { return };
        let mut rng = 0x9e37_79b9u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        let pieces = [
            "a", "b", "e", "th", "in", "er", "é", "日本", "🦀", " ", "1", "!", "ab", "\n",
        ];
        let mut words: Vec<String> = Vec::new();
        for n in [1, 2, 3, 63, 64, 65, 100, 257, 1000] {
            for c in ["a", "e", "!", " ", "中", "ab", "aab"] {
                words.push(c.repeat(n));
            }
        }
        for _ in 0..400 {
            let n = 1 + (next() % 300) as usize;
            let alpha = 1 + (next() as usize % pieces.len());
            words.push((0..n).map(|_| pieces[next() as usize % alpha]).collect());
        }
        for w in &words {
            let syms: Vec<u32> = (w.bytes()).filter_map(|b| t.byte_id[b as usize]).collect();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            t.bpe_scan(syms.clone(), &mut a);
            t.bpe_heap(syms, &mut b);
            assert_eq!(a, b, "word {w:?}");
            assert_eq!(t.decode(&b), *w);
        }
    }

    /// A long pre-token (a run of one letter, or of punctuation) used to
    /// take O(n^2) merge scans: 16 KB took 2 s, a 1 MB request body hours
    /// of one CPU core. Now it's O(n log n).
    #[test]
    fn long_pretoken_encodes_fast() {
        let Some(t) = tok() else { return };
        for c in ["a", "!", "中"] {
            let s = c.repeat((1 << 20) / c.len());
            let t0 = std::time::Instant::now();
            let ids = t.encode(&s);
            let secs = t0.elapsed().as_secs_f64();
            assert_eq!(t.decode(&ids), s);
            // ~0.1 s now; the scan needed > 120 s (killed), est. hours.
            // A loose bound, and none in the debug-assertion suite (run
            // under load): the identity above is the part that matters.
            if !cfg!(debug_assertions) {
                assert!(secs < 120.0, "1 MB of {c:?}: {secs:.1} s");
            }
        }
    }
}
