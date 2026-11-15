//! Unit tests (no model): the JSON automaton against `tools::parse_json`,
//! masks against walking every token, the Hermes grammar on the 8
//! malformed calls of #259, special tokens inside strings, the free /
//! engaged / free cycle, and constrained generation (greedy and sampled,
//! drafted or not) on a toy model.

use super::mask::{Masks, TokenMask};
use super::pda::{self, Builder, Graph, MAX_DEPTH, NodeId, Step, Sym};
use super::vocab::Vocab;
use super::{Constraint, Grammar};
use crate::lm::Lm;
use crate::serve::tools::{J, parse_json};
use std::sync::Arc;

fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn pick<'a, T>(s: &mut u64, v: &'a [T]) -> &'a T {
    &v[(xs(s) % v.len() as u64) as usize]
}

fn pick_str(s: &mut u64, v: &[&'static str]) -> &'static str {
    v[(xs(s) % v.len() as u64) as usize]
}

/// One JSON value between optional whitespace: the oracle's grammar.
fn value_grammar() -> (Graph, NodeId) {
    let mut b = Builder::default();
    let j = pda::json(&mut b);
    let top = b.node();
    let end = b.node();
    b.ws_loop(top);
    b.call(top, j.value, end);
    b.ws_loop(end);
    b.set_accept(end);
    (b.finish().unwrap(), top)
}

/// Whether `g` takes all of `s` from `start` and may end there.
fn accepts_whole(g: &Graph, start: NodeId, s: &[u8]) -> bool {
    let (mut n, mut st) = (start, Vec::new());
    s.iter()
        .all(|&b| g.step(&mut n, &mut st, Sym::Byte(b)) == Step::Ok)
        && g.may_end(n, &st)
}

/// The index of the first symbol `g` refuses from `start`, if any.
fn first_reject(g: &Graph, start: NodeId, syms: &[Sym]) -> Option<usize> {
    let (mut n, mut st) = (start, Vec::new());
    syms.iter()
        .position(|&x| g.step(&mut n, &mut st, x) != Step::Ok)
}

/// Deepest nesting of containers in `s` (strings skipped).
fn depth(s: &str) -> usize {
    let (mut d, mut max, mut in_s, mut esc) = (0usize, 0, false, false);
    for c in s.chars() {
        if in_s {
            match (esc, c) {
                (true, _) => esc = false,
                (false, '\\') => esc = true,
                (false, '"') => in_s = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_s = true,
            '[' | '{' => {
                d += 1;
                max = max.max(d);
            }
            ']' | '}' => d = d.saturating_sub(1),
            _ => {}
        }
    }
    max
}

fn gen_ws(s: &mut u64, out: &mut String) {
    out.push_str(pick_str(s, &["", "", "", " ", "\n", "\t ", "\r\n  "]));
}

fn gen_string(s: &mut u64, out: &mut String) {
    out.push('"');
    for _ in 0..xs(s) % 6 {
        out.push_str(pick_str(
            s,
            &[
                "a",
                "Z",
                "7",
                " ",
                "é",
                "😀",
                "中",
                "\\n",
                "\\\"",
                "\\\\",
                "\\/",
                "\\b",
                "\\f",
                "\\r",
                "\\t",
                "\\u00e9",
                "\\uD83D\\uDE00",
                "\\u12aB",
                "</tool_call>",
                "{",
                "]",
                ",",
                ":",
            ],
        ));
    }
    out.push('"');
}

fn gen_number(s: &mut u64, out: &mut String) {
    if xs(s).is_multiple_of(3) {
        out.push('-');
    }
    if xs(s).is_multiple_of(4) {
        out.push('0');
    } else {
        out.push(char::from(b'1' + (xs(s) % 9) as u8));
        for _ in 0..xs(s) % 4 {
            out.push(char::from(b'0' + (xs(s) % 10) as u8));
        }
    }
    if xs(s).is_multiple_of(3) {
        out.push('.');
        for _ in 0..1 + xs(s) % 3 {
            out.push(char::from(b'0' + (xs(s) % 10) as u8));
        }
    }
    if xs(s).is_multiple_of(4) {
        out.push_str(pick_str(s, &["e", "E", "e+", "E-", "e-"]));
        for _ in 0..1 + xs(s) % 2 {
            out.push(char::from(b'0' + (xs(s) % 10) as u8));
        }
    }
}

fn gen_value(s: &mut u64, d: usize, out: &mut String) {
    let k = xs(s) % if d > 5 { 5 } else { 8 };
    match k {
        0 | 3 => gen_number(s, out),
        1 | 4 => gen_string(s, out),
        2 => out.push_str(pick_str(s, &["true", "false", "null"])),
        5 | 6 => {
            out.push('[');
            let n = xs(s) % 4;
            for i in 0..n {
                gen_ws(s, out);
                gen_value(s, d + 1, out);
                gen_ws(s, out);
                if i + 1 < n {
                    out.push(',');
                }
            }
            gen_ws(s, out);
            out.push(']');
        }
        _ => {
            out.push('{');
            let n = xs(s) % 4;
            for i in 0..n {
                gen_ws(s, out);
                gen_string(s, out);
                gen_ws(s, out);
                out.push(':');
                gen_ws(s, out);
                gen_value(s, d + 1, out);
                gen_ws(s, out);
                if i + 1 < n {
                    out.push(',');
                }
            }
            gen_ws(s, out);
            out.push('}');
        }
    }
}

/// `s` with one random edit at a char boundary.
fn mutate(s: &mut u64, v: &str) -> String {
    let cuts: Vec<usize> = (0..=v.len()).filter(|&i| v.is_char_boundary(i)).collect();
    let at = *pick(s, &cuts);
    let ins = pick_str(
        s,
        &[
            "\"", "\\", "{", "}", "[", "]", ",", ":", "\n", "\u{1}", "0", "1", "e", "E", ".", "+",
            "-", "t", "u", "x", "\\(", "\\u12", " ",
        ],
    );
    match xs(s) % 4 {
        0 => format!("{}{ins}{}", &v[..at], &v[at..]),
        1 if at < v.len() => {
            let next = cuts.iter().find(|&&c| c > at).copied().unwrap_or(v.len());
            format!("{}{}", &v[..at], &v[next..])
        }
        2 if at < v.len() => {
            let next = cuts.iter().find(|&&c| c > at).copied().unwrap_or(v.len());
            format!("{}{ins}{}", &v[..at], &v[next..])
        }
        _ => v[..at].to_string(),
    }
}

/// The automaton takes exactly what `tools::parse_json` parses, up to the
/// depth cap: 10k values, random and mutated, plus hand-picked edges and
/// nesting at 63, 64 and 65.
#[test]
fn json_matches_parse_json() {
    let (g, top) = value_grammar();
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut corpus: Vec<String> = Vec::new();
    for i in 0..10_000 {
        let mut v = String::new();
        gen_ws(&mut s, &mut v);
        gen_value(&mut s, 0, &mut v);
        gen_ws(&mut s, &mut v);
        corpus.push(if i % 2 == 0 { v } else { mutate(&mut s, &v) });
    }
    for e in [
        "0",
        "-0",
        "01",
        "-",
        "1.",
        "1e",
        "1e+",
        ".5",
        "0.0e-0",
        "1E5",
        "-12.50e+03",
        "+1",
        "00",
        "\"\\x\"",
        "\"\\(\"",
        "\"\\u12g4\"",
        "\"\\u00\"",
        "\"a\nb\"",
        "\"\u{7f}\"",
        "tru",
        "nul",
        "true false",
        "[1,]",
        "[,1]",
        "{\"a\"}",
        "{\"a\":}",
        "{\"a\":1,}",
        "{,}",
        "{1:2}",
        "[]",
        "{}",
        " [ ] ",
        "\"\"",
        "[\"\\\"\"]",
        "{\"\\u0041\":[{}]}",
        "",
    ] {
        corpus.push(e.to_string());
    }
    for d in [62usize, 63, 64, 65, 66] {
        corpus.push(format!("{}{}", "[".repeat(d), "]".repeat(d)));
        corpus.push(format!("{}1{}", "{\"a\":".repeat(d), "}".repeat(d)));
        corpus.push(format!(
            "{}\"x\"{}",
            "[{\"k\": ".repeat(d / 2),
            "}]".repeat(d / 2)
        ));
    }
    let (mut ok, mut capped) = (0, 0);
    for v in &corpus {
        let want = parse_json(v).is_some();
        let deep = depth(v) > MAX_DEPTH;
        let got = accepts_whole(&g, top, v.as_bytes());
        assert_eq!(got, want && !deep, "{v:?}");
        ok += usize::from(want);
        capped += usize::from(want && deep);
    }
    assert!(
        ok > 5_000 && capped >= 4,
        "{ok} valid, {capped} past the cap"
    );
}

/// The tool names the unit tests' calls take (`Grammar::hermes`'s enum):
/// #259's (codex's shell tool and the patch tool it called unoffered),
/// short ones the toy model can spell, and a prefix of another.
pub(super) const NAMES: &[&str] = &["get", "x", "a", "ab", "exec_command", "apply_patch"];

/// `names` as `Grammar::hermes` takes them (their JSON string bodies).
pub(super) fn names(n: &[&str]) -> Vec<Vec<u8>> {
    n.iter().map(|s| s.as_bytes().to_vec()).collect()
}

/// The Hermes grammar on `v`, `NAMES` offered.
pub(super) fn hermes(v: Arc<Vocab>) -> Grammar {
    Grammar::hermes(v, &names(NAMES)).unwrap()
}

/// A synthetic vocabulary: every single byte, JSON-ish pieces, random
/// pieces, an empty token, three specials, and ids past the pieces.
fn toy_vocab(seed: u64, n_rand: usize) -> (Arc<Vocab>, Vec<u32>) {
    let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    for p in [
        "{\"",
        "\":",
        "\",",
        "\"}",
        "\"}}",
        "}}",
        "}}}",
        "]}",
        "\"]}",
        "}]}",
        "\n}\n}",
        "\n}",
        "[[",
        "[[[[",
        "[{",
        "{\"a\":[",
        "\": \"",
        "\", \"",
        "\":{",
        "\":[",
        "\"name\"",
        "name",
        "\"arguments\"",
        "arguments",
        "true",
        "null",
        "false",
        "\\n",
        "\\\"",
        "\\\\",
        "\\u",
        "  ",
        "\n\n",
        ", ",
        "\"\n",
        "0.5",
        "-1",
        "1e",
        "</tool_call",
        "</",
        "tool_call",
        "\"}\n",
        "\"}}\n",
        "}\n",
        "é",
        "😀",
        "\"é",
        "\":\"",
        "\"]",
        "],",
        "\"],",
        "]]",
        "]]]",
        "}]",
    ] {
        pieces.push(p.as_bytes().to_vec());
    }
    let alpha = b"{}[]\":,  \n\\abcxyz019.-+eEtrunl/<>";
    let mut s = seed;
    for _ in 0..n_rand {
        let len = 1 + xs(&mut s) % 7;
        pieces.push(
            (0..len)
                .map(|_| alpha[(xs(&mut s) % alpha.len() as u64) as usize])
                .collect(),
        );
    }
    pieces.push(Vec::new()); // decodes to nothing: never allowed
    let mut specials = Vec::new();
    for t in ["<tool_call>", "</tool_call>", "<|im_end|>"] {
        specials.push(pieces.len() as u32);
        pieces.push(t.as_bytes().to_vec());
    }
    let n = pieces.len() + 3; // ids past the pieces: nothing
    (
        Arc::new(Vocab::from_pieces(n, &pieces, &specials)),
        specials,
    )
}

/// Every token the live automaton takes at (node, stack), by walking each.
fn naive(m: &Masks, node: NodeId, stack: &[NodeId]) -> Vec<u32> {
    let mut buf = Vec::new();
    (0..m.vocab.len() as u32)
        .filter(|&id| m.walk(node, stack, id, &mut buf).is_some())
        .collect()
}

fn ids(m: &TokenMask) -> Vec<u32> {
    let mut v: Vec<u32> = m.ids().collect();
    v.sort_unstable();
    v
}

/// The cached masks equal walking every token through the live automaton,
/// bit for bit, at every state of random walks through the Hermes grammar:
/// shallow and deep (to the depth cap and past it), with tokens that pop
/// one frame (`",`), several (`"}}`, deep tokens) or open several (`[[[[`).
/// Count is the popcount. Special tokens are allowed only where the grammar
/// names them (the close), never inside a string.
#[test]
fn masks_match_naive() {
    let (v, sp) = toy_vocab(7, 3000);
    let g = hermes(v.clone());
    let m = &g.masks;
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut states = 0;
    let mut deep_seen = 0;
    let mut max_stack = 0;
    for walk in 0..48 {
        let (mut node, mut stack) = (g.start, Vec::<NodeId>::new());
        // Some walks open containers whenever they can (toward the cap).
        let dive = walk % 3 == 0;
        let mut buf = Vec::new();
        for _ in 0..if dive { 200 } else { 120 } {
            let tm = m.mask(node, &stack, &mut buf);
            let want = naive(m, node, &stack);
            assert_eq!(ids(&tm), want, "walk {walk} at node {node} stack {stack:?}");
            assert_eq!(tm.count(), want.len());
            for id in 0..v.len() as u32 {
                assert_eq!(tm.allows(id), want.binary_search(&id).is_ok());
            }
            let e = m.entry(node, &stack);
            deep_seen += e.deep.len();
            states += 1;
            max_stack = max_stack.max(stack.len());
            let in_str = m.graph.nodes[node as usize].in_str;
            for &x in &sp {
                if in_str || x != sp[1] {
                    assert!(!tm.allows(x), "special {x} allowed at node {node}");
                }
            }
            assert!(!want.is_empty(), "nothing allowed at node {node}");
            // Diving: mostly a token that deepens the stack the most (to
            // the cap and against it), else any.
            let choice = if dive && !xs(&mut s).is_multiple_of(8) {
                let grow = |id: u32, buf: &mut Vec<NodeId>| {
                    m.walk(node, &stack, id, buf);
                    buf.len() as i64 - stack.len() as i64
                };
                let best = (want.iter().map(|&id| grow(id, &mut buf))).max().unwrap();
                let top: Vec<u32> = (want.iter().copied())
                    .filter(|&id| grow(id, &mut buf) == best)
                    .collect();
                *pick(&mut s, &top)
            } else {
                *pick(&mut s, &want)
            };
            let Some(n) = m.walk(node, &stack, choice, &mut buf) else {
                panic!("an allowed token refused");
            };
            node = n;
            stack = buf.clone();
            if m.graph.nodes[node as usize].accept && stack.is_empty() {
                break;
            }
        }
    }
    assert!(states > 3000, "{states} states");
    assert!(deep_seen > 0, "no deep tokens met");
    // At most MAX_DEPTH frames (the arguments object's included): the
    // dives reach the cap, and the masks there match too.
    assert_eq!(max_stack, MAX_DEPTH, "deepest stack");
}

/// The Hermes grammar refuses each of the 8 malformed calls of #259 at the
/// byte where it goes wrong (unescaped quote x4, invalid escape x2, raw
/// newline, a second `<tool_call>` before `</tool_call>`), and the mask
/// there doesn't allow that byte's token (or that special token).
#[test]
fn malformed_259_impossible() {
    let d = crate::testutil::read_json("constrain_259_malformed.json");
    let cases = d["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    let (v, sp) = toy_vocab(3, 500);
    let g = hermes(v.clone());
    let mut whys = std::collections::BTreeMap::new();
    for c in cases {
        let body = c["body"].as_str().unwrap();
        let at = c["reject_at"].as_u64().unwrap() as usize;
        let why = c["why"].as_str().unwrap();
        *whys.entry(why.to_string()).or_insert(0) += 1;
        // The body as the model's tokens: the markers as their special
        // tokens, the rest byte by byte (offsets kept per symbol).
        let mut syms = Vec::new();
        let b = body.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i..].starts_with(b"<tool_call>") {
                syms.push((i, Sym::Special(sp[0]), sp[0]));
                i += 11;
            } else if b[i..].starts_with(b"</tool_call>") {
                syms.push((i, Sym::Special(sp[1]), sp[1]));
                i += 12;
            } else {
                syms.push((i, Sym::Byte(b[i]), b[i] as u32));
                i += 1;
            }
        }
        let only: Vec<Sym> = syms.iter().map(|x| x.1).collect();
        let r = first_reject(&g.masks.graph, g.start, &only).expect("refused somewhere");
        assert_eq!(
            syms[r].0,
            at,
            "{why}: {:?}",
            &body[at.saturating_sub(30)..at + 1]
        );
        // The same through a Constraint and its masks, token by token.
        let mut con = Constraint::new(Arc::new(hermes(v.clone())));
        assert!(con.accept(sp[0]) && con.engaged());
        for (k, &(_, _, id)) in syms.iter().enumerate() {
            let m = con.mask().unwrap().expect("engaged");
            if k == r {
                assert!(!m.allows(id), "{why}: token {id} allowed");
                assert!(!con.accept(id));
                break;
            }
            assert!(m.allows(id) && con.accept(id), "{why}: symbol {k}");
        }
    }
    assert_eq!(whys["unescaped quote inside a string"], 4);
    assert_eq!(whys["invalid escape"], 2);
    assert_eq!(whys["raw control byte in a string"], 1);
    assert_eq!(whys["a second <tool_call> before </tool_call>"], 1);
}

/// Free until `<tool_call>`, engaged (masked) through the call, free again
/// after `</tool_call>` (as its token or spelled), and again for a second
/// call (the maintainer's lazy re-arm). Inside strings no special token is
/// allowed; `in_string` says where strings are.
#[test]
fn hermes_cycle() {
    let (v, sp) = toy_vocab(11, 200);
    let g = Arc::new(hermes(v.clone()));
    let mut c = Constraint::new(g.clone());
    let feed = |c: &mut Constraint, s: &str| {
        for &b in s.as_bytes() {
            let m = c.mask().unwrap().expect("engaged");
            assert!(m.allows(b as u32), "{s:?}: byte {:?}", b as char);
            assert!(c.accept(b as u32));
        }
    };
    assert!(c.mask().unwrap().is_none());
    // Free: anything goes, the stop token included.
    assert!(c.accept(b'x' as u32) && c.accept(sp[2]) && c.accept(sp[1]));
    assert!(!c.engaged());
    assert!(c.accept(sp[0]));
    assert!(c.engaged());
    feed(&mut c, "\n{\"name\": \"ge");
    assert!(c.in_string());
    let m = c.mask().unwrap().unwrap();
    for &x in &sp {
        assert!(!m.allows(x), "special {x} inside a string");
    }
    feed(
        &mut c,
        "t\", \"arguments\": {\"q\": [1, 2], \"s\": \"</tool_call>\", \"o\": {\"a\": null}}}\n",
    );
    assert!(!c.in_string());
    let m = c.mask().unwrap().unwrap();
    assert!(m.allows(sp[1]) && !m.allows(sp[0]) && !m.allows(sp[2]));
    assert!(m.allows(b'<' as u32) && !m.allows(b'x' as u32));
    assert!(c.accept(sp[1]));
    assert!(!c.engaged());
    assert!(c.mask().unwrap().is_none());
    // A second call, closed spelled out.
    assert!(c.accept(b'\n' as u32) && c.accept(sp[0]));
    feed(&mut c, "{\"name\":\"x\",\"arguments\":{}}</tool_call");
    assert!(c.engaged());
    feed(&mut c, ">");
    assert!(!c.engaged());
    // Refused tokens leave the state as it was.
    assert!(c.accept(sp[0]));
    assert!(!c.accept(b'x' as u32));
    assert!(!c.accept(sp[1]));
    feed(&mut c, " {");
    let empty = (0..v.len() as u32).find(|&id| v.bytes(id).is_empty() && !v.is_special(id));
    assert!(!c.accept(empty.unwrap()), "an empty token is never allowed");
    assert!(!c.accept(v.len() as u32 - 1), "an id past the pieces");
}

/// The grammar's name in the identity: stable for a vocabulary, the same
/// whatever tools are offered (they are the request's, as its prompt is),
/// and another grammar text would be another hash (`Hermes::field` is it).
#[test]
fn hermes_field() {
    let (v, _) = toy_vocab(1, 10);
    let a = hermes(v.clone());
    let b = hermes(v.clone());
    let c = Grammar::hermes(v.clone(), &names(&["only"])).unwrap();
    assert_eq!(a.field(), b.field());
    assert_eq!(a.field(), c.field());
    let h = super::Hermes::new(v.clone(), Default::default()).unwrap();
    assert_eq!(h.field(), a.field());
    assert!(a.field().starts_with("hermes-json ") && a.field().len() == 12 + 16);
    // No names: refused (a call couldn't name anything).
    assert!(Grammar::hermes(v, &[]).err().unwrap().contains("no tool names"));
    // No `<tool_call>` token; not byte-complete (no 'q'): refused.
    let bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    // Each refusal says why (the server's startup line).
    let why = |v: Vocab| Grammar::hermes(Arc::new(v), &names(NAMES)).err();
    let w = why(Vocab::from_pieces(256, &bytes, &[])).unwrap();
    assert!(w.contains("no <tool_call>"), "{w}");
    let mut p = bytes.clone();
    p[b'q' as usize] = b"qq".to_vec();
    p.push(b"<tool_call>".to_vec());
    let w = why(Vocab::from_pieces(257, &p, &[256])).unwrap();
    assert!(w.contains("byte-complete: byte 0x71 "), "{w}");
    p[b'q' as usize] = b"q".to_vec();
    assert!(Grammar::hermes(Arc::new(Vocab::from_pieces(257, &p, &[256])), &names(NAMES)).is_ok());
}

/// `cands_masked` is the reference: the `k` best allowed (penalty first),
/// never a forbidden id; with everything allowed it is `cands`; and the
/// device's way (`masked_prefix` of the CAND_MAX best) gives the very same
/// candidates whenever it says it can, on rows with ties, NaN and -inf.
#[test]
fn cands_masked_reference() {
    use crate::sample::{CAND_MAX, CandSpec, cands, cands_masked, masked_prefix, top_k};
    let mut s = 0xfeed_beefu64;
    let n = 3000;
    let (mut fell, mut took) = (0, 0);
    for round in 0..400 {
        let l: Vec<f32> = (0..n)
            .map(|_| match (round % 4, xs(&mut s) % 50) {
                (1, x) => (x % 5) as f32,
                (2, 0) => f32::NAN,
                (2, 1) => f32::NEG_INFINITY,
                _ => (xs(&mut s) % 10_000) as f32 / 300.0 - 10.0,
            })
            .collect();
        let allowed: Vec<u32> = match round % 5 {
            0 => (0..n as u32).collect(),
            1 => (0..n as u32)
                .filter(|_| xs(&mut s).is_multiple_of(3))
                .collect(),
            2 => (0..1 + xs(&mut s) % 8)
                .map(|_| (xs(&mut s) % n as u64) as u32)
                .collect(),
            3 => vec![(xs(&mut s) % n as u64) as u32],
            _ => (0..n as u32)
                .filter(|_| xs(&mut s).is_multiple_of(100))
                .collect(),
        };
        let mask = TokenMask::from_ids(n, &allowed);
        let recent: Vec<u32> = (0..20).map(|_| (xs(&mut s) % n as u64) as u32).collect();
        for k in [1usize, 2, 20, 40, 256, 1000, usize::MAX] {
            let spec = CandSpec {
                k,
                penalty: [1.0, 1.3][round % 2],
                last_n: 16,
            };
            let mut lm = l.clone();
            let got = cands_masked(&mut lm, &spec, &recent, &mask);
            // Brute force: penalize, keep the allowed, sort by (key, id).
            let mut lp = l.clone();
            let all = cands(&mut lp, &CandSpec { k: n, ..spec }, &recent);
            let want: Vec<_> = (all.iter().filter(|c| mask.allows(c.id)))
                .take(k)
                .map(|c| (c.logit.to_bits(), c.id))
                .collect();
            let gb: Vec<_> = got.iter().map(|c| (c.logit.to_bits(), c.id)).collect();
            assert_eq!(gb, want, "round {round} k {k}");
            if allowed.len() == n {
                let mut lc = l.clone();
                let c = cands(&mut lc, &spec, &recent);
                let cb: Vec<_> = c.iter().map(|c| (c.logit.to_bits(), c.id)).collect();
                assert_eq!(gb, cb, "everything allowed: cands");
            }
            if k <= CAND_MAX {
                let best = top_k(&lp, CAND_MAX);
                match masked_prefix(&best, k, &mask) {
                    Some(c) => {
                        took += 1;
                        let cb: Vec<_> = c.iter().map(|c| (c.logit.to_bits(), c.id)).collect();
                        assert_eq!(cb, want, "prefix, round {round} k {k}");
                    }
                    None => fell += 1,
                }
            }
        }
    }
    assert!(fell > 0 && took > 0, "{took} prefixes, {fell} fallbacks");
}

/// A toy model over a JSON-ish vocabulary, for constrained generation: soft
/// logits from a hash of the token, its position and the token before it,
/// with `<tool_call>` likely every few positions and `</tool_call>` always
/// strong (it is allowed only at a call's close).
struct Toy {
    v: Arc<Vocab>,
    trig: u32,
    close: u32,
}

impl Toy {
    fn new(v: Arc<Vocab>, sp: &[u32]) -> Toy {
        Toy {
            v,
            trig: sp[0],
            close: sp[1],
        }
    }

    /// The filler bytes (`toy_json_vocab`): ids past the specials.
    fn rare(&self, j: u32) -> bool {
        j > self.close + 1
    }
}

impl Toy {
    fn logits(&self, t: u32, pos: usize, c: &mut [u32]) -> Vec<f32> {
        c[pos] = t;
        let prev = if pos > 0 { c[pos - 1] } else { 5 };
        (0..self.v.len() as u32)
            .map(|j| {
                let mut h = (t.wrapping_mul(2654435761)
                    ^ (pos as u32).wrapping_mul(40503)
                    ^ prev.wrapping_mul(97)
                    ^ j.wrapping_mul(7919))
                    | 1;
                h ^= h << 13;
                h ^= h >> 17;
                h ^= h << 5;
                let base = (h % 1000) as f32 / 150.0;
                if j == self.trig && pos % 7 == 3 {
                    base + 9.0
                } else if j == self.close {
                    base + 4.0
                } else if self.rare(j) {
                    base - 5.0
                } else {
                    base
                }
            })
            .collect()
    }
}

impl crate::lm::Sealed for Toy {}
impl Lm for Toy {
    type Cache = Vec<u32>;
    fn vocab(&self) -> usize {
        self.v.len()
    }
    fn new_cache(&self, max_ctx: usize) -> Vec<u32> {
        vec![u32::MAX; max_ctx]
    }
    fn forward(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> Vec<f32> {
        self.logits(t, pos, c)
    }
    fn feed_batch(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) {
        c[start..start + ts.len()].copy_from_slice(ts);
    }
    fn verify_argmax(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) -> Vec<u32> {
        (0..ts.len())
            .map(|b| crate::argmax(&self.logits(ts[b], start + b, c)))
            .collect()
    }
}

/// The JSON-ish vocabulary of `Toy`.
fn toy_json_vocab() -> (Arc<Vocab>, Vec<u32>) {
    let mut pieces: Vec<Vec<u8>> = Vec::new();
    for p in [
        "{",
        "}",
        "[",
        "]",
        "\"",
        ":",
        ",",
        " ",
        "\n",
        "a",
        "b",
        "x",
        "0",
        "1",
        "-",
        ".",
        "e",
        "\\",
        "n",
        "\"name\"",
        "\"arguments\"",
        ": \"",
        "\", \"",
        "\"}",
        "\"}}",
        "}}",
        "true",
        "null",
        "{\"",
        "\":",
        "\\n",
        "\\\"",
        "</tool_call",
        ">",
        "Hi",
        " there",
    ] {
        pieces.push(p.as_bytes().to_vec());
    }
    let mut sp = Vec::new();
    for t in ["<tool_call>", "</tool_call>", "<|im_end|>"] {
        sp.push(pieces.len() as u32);
        pieces.push(t.as_bytes().to_vec());
    }
    // Every other byte as a token (byte-complete, as byte-level BPE):
    // `Toy::rare` from here on, unlikely.
    let have: Vec<Vec<u8>> = pieces.clone();
    for b in (0x20..=0xffu8).chain([b'\t', b'\n', b'\r']) {
        if !have.contains(&vec![b]) {
            pieces.push(vec![b]);
        }
    }
    (Arc::new(Vocab::from_pieces(pieces.len(), &pieces, &sp)), sp)
}

/// Generate with `generate`'s loop (drafting by `mode`, constrained if
/// `g`), as `sample::tests::run` does for plain sampling.
#[allow(clippy::too_many_arguments)]
fn run(
    net: &Toy,
    g: Option<&Arc<Grammar>>,
    prompt: &[u32],
    n: usize,
    stop: &[u32],
    k: usize,
    cfg: Option<crate::sample::SampleCfg>,
    mode: &str,
    oracle: &[u32],
) -> crate::GenStats {
    let mut cache = net.new_cache(prompt.len() + n + 1);
    let (&last, rest) = prompt.split_last().unwrap();
    net.feed_batch(rest, 0, &mut cache);
    let logits = net.forward(last, rest.len(), &mut cache);
    let mut sampler = cfg.map(crate::sample::Sampler::new);
    let mut con = g.map(|g| Constraint::new(g.clone()));
    let mut rng = 0x2545_f491u64 ^ cfg.map_or(0, |c| c.seed);
    let mut lookup = crate::spec::PromptLookup::new();
    let mut draft = |h: &[u32], max: usize| -> Vec<u32> {
        let at = h.len() - prompt.len();
        let right = |i: usize| oracle.get(at + i).copied().unwrap_or(1);
        match mode {
            "oracle" => (0..max).map(right).collect(),
            "wrong" => (0..max).map(|i| right(i) ^ 1).collect(),
            "random" => (0..max)
                .map(|_| (xs(&mut rng) % net.v.len() as u64) as u32)
                .collect(),
            _ => {
                lookup.sync(h);
                lookup.draft(max)
            }
        }
    };
    let mut emitted = Vec::new();
    let st = crate::generate(
        net,
        &mut cache,
        prompt,
        logits,
        crate::GenOpts {
            n_gen: n,
            stop,
            max_draft: k,
            sample: sampler.as_mut(),
            constraint: con.as_mut(),
        },
        &mut draft,
        &mut |t| {
            emitted.push(t);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(emitted, st.ids);
    st
}

/// The calls in `ids`, closed ones checked: each a JSON object (by
/// `tools::parse_json`) with a string "name" then an object "arguments".
/// Where a call ends comes from a shadow `Constraint`, not from scanning
/// the text for `</tool_call>`: the model spells that inside string values
/// too (the maintainer's ruling 5 on the spec's §6: PR B's Parser reads the
/// same way). Returns (closed calls, spelled closes seen inside strings).
fn calls_parse(g: &Arc<Grammar>, v: &Vocab, ids: &[u32]) -> (usize, usize) {
    let mut c = Constraint::new(g.clone());
    let (mut calls, mut in_str) = (0, 0);
    let mut body: Option<Vec<u8>> = None;
    for &t in ids {
        let was = c.engaged();
        assert!(c.accept(t), "token {t} refused");
        match (was, c.engaged()) {
            (false, true) => body = Some(Vec::new()),
            (true, true) => {
                let b = body.as_mut().unwrap();
                b.extend_from_slice(v.bytes(t));
                if b.ends_with(b"</tool_call>") {
                    assert!(c.in_string());
                    in_str += 1;
                }
            }
            (true, false) => {
                let mut b = body.take().unwrap();
                if !v.is_special(t) {
                    b.extend_from_slice(v.bytes(t));
                    assert!(b.ends_with(b"</tool_call>"));
                    b.truncate(b.len() - 12);
                }
                let text = String::from_utf8(b).unwrap();
                let Some(J::Obj(m)) = parse_json(&text) else {
                    panic!("not an object: {text:?}");
                };
                assert!(matches!(&m[0], (k, J::Str(_)) if k == "name"), "{text}");
                assert!(
                    matches!(&m[1], (k, J::Obj(_)) if k == "arguments"),
                    "{text}"
                );
                assert_eq!(m.len(), 2, "{text}");
                calls += 1;
            }
            (false, false) => {}
        }
    }
    (calls, in_str)
}

/// Constrained generation, greedy and sampled over 300 seeds: drafts that
/// are always right, always wrong, random or real prompt lookup give the
/// very tokens undrafted generation gives (one RNG number per decided
/// token, the masks carried through a clone); every call parses; and until
/// the trigger the tokens are the unconstrained run's.
#[test]
fn constrained_drafting_matches_plain() {
    let (v, sp) = toy_json_vocab();
    let net = Toy::new(v.clone(), &sp);
    let g = Arc::new(hermes(v.clone()));
    let prompt = [9u32, 10, 34, 35, 9, 10];
    let (mut calls, mut engaged_runs, mut spelled_in_str) = (0, 0, 0);
    for seed in 0..300u64 {
        let cfg = (seed % 3 != 0).then(|| crate::sample::SampleCfg {
            temp: [0.8, 1.5, 0.3][seed as usize % 3],
            top_k: [40, 5, 0][seed as usize % 3],
            top_p: [0.9, 1.0][seed as usize % 2],
            min_p: [0.0, 0.05][seed as usize % 2],
            repeat_penalty: [1.0, 1.1, 1.3][seed as usize % 3],
            repeat_last_n: [64, 0, 3][seed as usize % 3],
            seed,
        });
        let stop: &[u32] = if seed % 4 == 0 { &[sp[2]] } else { &[] };
        let n = 48;
        let plain = run(&net, Some(&g), &prompt, n, stop, 0, cfg, "", &[]);
        let free = run(&net, None, &prompt, n, stop, 0, cfg, "", &[]);
        // Until (and with) the first trigger: the unconstrained tokens.
        let t = plain.ids.iter().position(|&x| x == sp[0]);
        let upto = t.map_or(plain.ids.len(), |t| t + 1);
        assert_eq!(
            plain.ids[..upto],
            free.ids[..upto.min(free.ids.len())],
            "seed {seed}"
        );
        if t.is_none() {
            assert_eq!(plain.ids, free.ids, "seed {seed}: never engaged");
        } else {
            engaged_runs += 1;
        }
        let (c, i) = calls_parse(&g, &v, &plain.ids);
        calls += c;
        spelled_in_str += i;
        for k in [1, 3, 7] {
            for mode in ["oracle", "wrong", "random", "lookup"] {
                let st = run(&net, Some(&g), &prompt, n, stop, k, cfg, mode, &plain.ids);
                let ctx = format!("seed {seed} k {k} {mode}");
                assert_eq!(st.ids, plain.ids, "{ctx}");
                assert_eq!(st.decided, plain.decided, "{ctx}");
                if mode == "oracle" && plain.ids.len() == n {
                    assert!(st.passes < plain.passes, "{ctx}");
                }
            }
        }
    }
    eprintln!(
        "{engaged_runs} runs engaged, {calls} calls closed, {spelled_in_str} spelled closes in strings"
    );
    assert!(
        engaged_runs > 100 && calls > 30,
        "{engaged_runs} engaged, {calls} calls"
    );
}

/// The default `verify_cands_masked` (row by row) is `forward_cands_masked`
/// per row, and `forward_cands_masked` is `cands_masked` of `forward`.
#[test]
fn verify_cands_masked_rows() {
    use crate::sample::{CandSpec, cands_masked};
    let (v, sp) = toy_json_vocab();
    let net = Toy::new(v.clone(), &sp);
    let toks = [9u32, 10, 19, 5, 4];
    let spec = CandSpec {
        k: 5,
        penalty: 1.2,
        last_n: 4,
    };
    let masks: Vec<TokenMask> = (0..toks.len())
        .map(|b| TokenMask::from_ids(v.len(), &[b as u32, 7, 8, 20 + b as u32]))
        .collect();
    let mut c1 = net.new_cache(16);
    let mrefs: Vec<Option<&TokenMask>> = (0..toks.len())
        .map(|b| (b != 2).then(|| &masks[b]))
        .collect();
    let rows = net.verify_cands_masked(&toks, 0, &mut c1, &spec, &toks, &mrefs);
    let mut c2 = net.new_cache(16);
    for (b, row) in rows.iter().enumerate() {
        let mut l = net.forward(toks[b], b, &mut c2);
        let want = match mrefs[b] {
            Some(m) => cands_masked(&mut l, &spec, &toks[..=b], m),
            None => crate::sample::cands(&mut l, &spec, &toks[..=b]),
        };
        assert_eq!(row, &want, "row {b}");
    }
}

/// `lm::verify_masked_via` and `lm::cands_masked_via` (the pipeline's and
/// the CPU/GPU split's routes) == the default per row, on a vocabulary
/// past CAND_MAX: unmasked rows (CAND_MAX's best cut to k), broad masks
/// (the allowed kept from the CAND_MAX best) and masks whose tokens are all
/// below the CAND_MAX best (the position run again for its logits); and
/// past CAND_MAX's k, row by row.
#[test]
fn masked_via_rows() {
    use crate::lm::{cands_masked_via, verify_masked_via};
    use crate::sample::{CAND_MAX, CandSpec, cands, cands_masked};
    let (v, sp) = toy_vocab(3, 700);
    let net = Toy::new(v.clone(), &sp);
    let n = v.len();
    let mut s = 0x5eed_u64;
    let toks: Vec<u32> = (0..20).map(|_| (xs(&mut s) % 900) as u32).collect();
    let masks: Vec<Option<TokenMask>> = (0..toks.len())
        .map(|b| match b % 3 {
            0 => None,
            1 => Some(TokenMask::from_ids(
                n,
                &(0..n as u32)
                    .filter(|j| j % 2 == b as u32 % 2)
                    .collect::<Vec<_>>(),
            )),
            _ => Some(TokenMask::from_ids(
                n,
                &(0..1 + b / 3 % 2)
                    .map(|_| (xs(&mut s) % n as u64) as u32)
                    .collect::<Vec<_>>(),
            )),
        })
        .collect();
    let mrefs: Vec<Option<&TokenMask>> = masks.iter().map(Option::as_ref).collect();
    for k in [5, CAND_MAX + 44] {
        let spec = CandSpec {
            k,
            penalty: 1.2,
            last_n: 8,
        };
        let mut c1 = net.new_cache(32);
        let rows = verify_masked_via(&net, &toks, 0, &mut c1, &spec, &toks, &mrefs);
        let mut below = 0;
        for (b, row) in rows.iter().enumerate() {
            let r = &toks[..=b];
            // The row's position alone, after the tokens before it.
            let mut c2 = net.new_cache(32);
            net.feed_batch(&toks[..b], 0, &mut c2);
            let mut c3 = c2.clone();
            let mut l = net.forward(toks[b], b, &mut c2);
            let want = match mrefs[b] {
                Some(m) => {
                    let mut best = l.clone();
                    let top = cands(
                        &mut best,
                        &CandSpec {
                            k: CAND_MAX,
                            ..spec
                        },
                        r,
                    );
                    below += top.iter().all(|c| !m.allows(c.id)) as usize;
                    assert_eq!(
                        cands_masked_via(&net, toks[b], b, &mut c3, &spec, r, m),
                        cands_masked(&mut l.clone(), &spec, r, m),
                        "k {k}: cands_masked_via row {b}"
                    );
                    cands_masked(&mut l, &spec, r, m)
                }
                None => cands(&mut l, &spec, r),
            };
            assert_eq!(row, &want, "k {k}: verify_masked_via row {b}");
        }
        assert!(below > 0, "k {k}: no row needed its logits");
    }
}

/// The call the mask-cost benches decode: a string with quotes, escapes
/// and a backslash, numbers, nesting, a literal.
pub(super) const COST_CALL: &str = "\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"sed -i \
    's/a/b/' x.py && echo \\\"done\\\"\", \"login\": false, \"n\": [1, 2.5, -3e2], \"m\": [{\"k\": null}], \
    \"deep\": {\"a\": {\"b\": [\"c\", \"d\"], \"e\": {\"d\": true}}}}}\n";

/// The per-token cost of the mask (`Constraint::mask` + `accept`) at a
/// 152k-token vocabulary, cache warm, against the spec's bar (median <= 20
/// us, p99 <= 100 us), and the first-visit compile of the string body.
/// Synthetic vocabulary (single bytes + random JSON-ish pieces), the call
/// a token per byte; model_tests.rs `mask_cost_real` has the real one.
/// `--ignored --nocapture` to see.
#[test]
#[ignore]
fn mask_cost_synthetic() {
    let (v, sp) = toy_vocab(5, 151_400);
    let toks: Vec<u32> = COST_CALL.bytes().map(|b| b as u32).collect();
    mask_cost_on(v, sp[0], &toks);
}

/// `mask_cost_synthetic`'s measurement on `v`: engaged by `trig`, then
/// `toks` (a whole call, closed or not).
pub(super) fn mask_cost_on(v: Arc<Vocab>, trig: u32, toks: &[u32]) {
    let g = Arc::new(hermes(v.clone()));
    let t0 = std::time::Instant::now();
    let mut c = Constraint::new(g.clone());
    c.accept(trig);
    for &t in toks {
        c.mask().unwrap();
        assert!(c.accept(t));
    }
    let cold = t0.elapsed().as_secs_f64() * 1e3;
    let mut us = Vec::new();
    for _ in 0..20 {
        let mut c = Constraint::new(g.clone());
        c.accept(trig);
        for &t in toks {
            let t1 = std::time::Instant::now();
            let m = c.mask().unwrap().unwrap();
            std::hint::black_box(m.allows(t));
            assert!(c.accept(t));
            us.push(t1.elapsed().as_secs_f64() * 1e6);
        }
    }
    us.sort_by(f64::total_cmp);
    let q = |p: f64| us[((us.len() - 1) as f64 * p) as usize];
    // The string body's compile alone (the largest entry: nearly every
    // token is allowed in a string).
    let mut c = Constraint::new(g.clone());
    c.accept(trig);
    let s = state_after(&mut c, &v, b"\n{\"name\": \"");
    let t2 = std::time::Instant::now();
    let e = g.masks.compute(Masks::key(s.node, &s.stack));
    let body_ms = t2.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "mask cost, vocab {} ({} trie nodes), {} tokens x 20: per token median {:.2} us, p99 {:.2} us, \
         max {:.1} us; first pass over the call (cold cache) {cold:.1} ms; string body compile \
         {body_ms:.2} ms ({} allowed, {} deep); {} cached entries",
        v.len(),
        v.trie().len(),
        toks.len(),
        q(0.5),
        q(0.99),
        us[us.len() - 1],
        e.bits_count(),
        e.deep.len(),
        g.masks.cached(),
    );
}

/// `c` after the bytes `b`, a single-byte token each.
fn state_after(c: &mut Constraint, v: &Vocab, b: &[u8]) -> Super {
    for &x in b {
        let id = (0..v.len() as u32)
            .find(|&id| v.bytes(id) == [x] && !v.is_special(id))
            .unwrap();
        assert!(c.accept(id));
    }
    state(c)
}

struct Super {
    node: NodeId,
    stack: Vec<NodeId>,
}

fn state(c: &Constraint) -> Super {
    match &c.phase {
        super::Phase::On { node, stack } => Super {
            node: *node,
            stack: stack.clone(),
        },
        super::Phase::Free => panic!("free"),
    }
}

// ------------------------------------------------------------- PR B: held calls

/// One JSON value between optional whitespace, held as a tool call's
/// arguments are (`Style::HELD`).
fn held_value_grammar() -> (Graph, NodeId) {
    let mut b = Builder::new(pda::Style::HELD);
    let j = pda::json(&mut b);
    let top = b.node();
    let end = b.node();
    b.ws_loop(top);
    b.call(top, j.value, end);
    b.ws_loop(end);
    b.set_accept(end);
    (b.finish().unwrap(), top)
}

/// Every whitespace run outside strings is llama.cpp's `space`: nothing,
/// one space, or one or two newlines then at most 20 spaces or tabs.
fn spaces_ok(s: &str) -> bool {
    let b = s.as_bytes();
    let (mut i, mut ins) = (0, false);
    while i < b.len() {
        let c = b[i];
        if ins {
            match c {
                b'\\' => i += 1,
                b'"' => ins = false,
                _ => {}
            }
            i += 1;
            continue;
        }
        if c == b'"' {
            ins = true;
            i += 1;
            continue;
        }
        if b" \t\n\r".contains(&c) {
            let j = (i..b.len()).find(|&j| !b" \t\n\r".contains(&b[j])).unwrap_or(b.len());
            let run = &b[i..j];
            let nl = run.iter().take_while(|&&x| x == b'\n').count();
            let ok = run == b" "
                || ((1..=2).contains(&nl)
                    && run[nl..].iter().all(|&x| x == b' ' || x == b'\t')
                    && run.len() - nl <= pda::INDENT_MAX);
            if !ok {
                return false;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    true
}

/// Every array's items are of one kind (booleans one kind).
fn one_kind(v: &J) -> bool {
    let k = |x: &J| match x {
        J::Null => 0,
        J::Bool(_) => 1,
        J::Num(_) => 2,
        J::Str(_) => 3,
        J::Arr(_) => 4,
        J::Obj(_) => 5,
    };
    match v {
        J::Arr(a) => a.windows(2).all(|w| k(&w[0]) == k(&w[1])) && a.iter().all(one_kind),
        J::Obj(m) => m.iter().all(|(_, x)| one_kind(x)),
        _ => true,
    }
}

/// The held JSON (`Style::HELD`) takes exactly what `tools::parse_json`
/// parses whose whitespace runs are llama.cpp's and whose arrays keep one
/// kind (and the depth cap): the 10k values of `json_matches_parse_json`,
/// random and mutated, with whitespace llama allows and some it doesn't.
#[test]
fn held_json_matches_its_oracle() {
    let (g, top) = held_value_grammar();
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let (mut ok, mut by_space, mut by_kind) = (0, 0, 0);
    let extra = [
        "[1, 2]", "[{}, {}]", "[{}, false]", "[true, false]", "[null, null]", "[[1], [\"a\"]]",
        "[[1], [2, \"a\"]]", "[1,  2]", "[1,\n\n\n2]", "[1,\n\t\t2]", "[1,\r\n2]", " 1", "1 ",
        "\n1\n", "\t1", "{\"a\":  1}", "{\"a\": [\"x\", 1]}", "[\"x\", \"y\"]", "[[], {}]",
    ];
    let mut corpus: Vec<String> = extra.iter().map(|x| x.to_string()).collect();
    for i in 0..10_000 {
        let mut v = String::new();
        gen_ws(&mut s, &mut v);
        gen_value(&mut s, 0, &mut v);
        gen_ws(&mut s, &mut v);
        corpus.push(if i % 3 == 0 { mutate(&mut s, &v) } else { v });
    }
    corpus.push(format!("[{}]", vec!["1"; 300].join(", ")));
    corpus.push(format!("\n{}1", " ".repeat(20)));
    corpus.push(format!("\n{}1", " ".repeat(21)));
    for v in &corpus {
        let parsed = parse_json(v);
        let sp = spaces_ok(v);
        let kind = parsed.as_ref().is_some_and(one_kind);
        let want = parsed.is_some() && depth(v) <= MAX_DEPTH && sp && kind;
        assert_eq!(accepts_whole(&g, top, v.as_bytes()), want, "{v:?}");
        ok += usize::from(want);
        by_space += usize::from(parsed.is_some() && !sp);
        by_kind += usize::from(parsed.is_some() && sp && !kind);
    }
    assert!(
        ok > 1_000 && by_space > 500 && by_kind > 200,
        "{ok} held, {by_space} refused for whitespace, {by_kind} for mixed arrays"
    );
}

/// The refused byte of `body` after `<tool_call>` through a constraint
/// (mask and accept agreeing at each byte), or None; and whether the call
/// closed.
fn held(g: &Arc<Grammar>, open: u32, body: &str) -> (Option<usize>, bool) {
    let mut c = Constraint::new(g.clone());
    assert!(c.accept(open) && c.engaged());
    for (i, &b) in body.as_bytes().iter().enumerate() {
        let m = c.mask().unwrap().expect("engaged");
        let ok = c.accept(b as u32);
        assert_eq!(ok, m.allows(b as u32), "{body:?}: mask and accept at {i}");
        if !ok {
            return (Some(i), c.engaged());
        }
    }
    (None, !c.engaged())
}

/// A held call's whitespace in its frame and its arguments: llama.cpp's
/// rule (#272: codex's call that wrote 25k tokens of whitespace).
#[test]
fn bounded_space() {
    let (v, sp) = toy_vocab(13, 50);
    let g = Arc::new(hermes(v));
    let call = |ws: &str| {
        format!("\n{{\"name\":{ws}\"get\", \"arguments\": {{\"q\":{ws}[1,{ws}2]}}}}\n</tool_call>")
    };
    let at = call("").find(":\"get").unwrap() + 1;
    let ind = |n: usize, c: &str| c.repeat(n);
    for ok in [
        String::new(),
        " ".into(),
        "\n".into(),
        "\n\n".into(),
        "\n\t".into(),
        "\n\n\t\t  ".into(),
        format!("\n{}", ind(20, " ")),
        format!("\n\n{}", ind(20, "\t")),
    ] {
        assert_eq!(held(&g, sp[0], &call(&ok)), (None, true), "{ok:?}");
    }
    for (bad, off) in [
        ("  ".to_string(), 1),
        ("\n\n\n".into(), 2),
        ("\r\n".into(), 0),
        ("\t".into(), 0),
        (" \n".into(), 1),
        (format!("\n{}", ind(21, " ")), 21),
    ] {
        assert_eq!(held(&g, sp[0], &call(&bad)).0, Some(at + off), "{bad:?}");
    }
    // The frame's own positions: before `{` and before the close.
    assert_eq!(held(&g, sp[0], "\n\n\n{").0, Some(2));
    let head = "{\"name\": \"x\", \"arguments\": {}}";
    assert_eq!(held(&g, sp[0], &format!("{head}  </tool_call>")).0, Some(head.len() + 1));
    assert_eq!(held(&g, sp[0], &format!("{head}\n</tool_call>")), (None, true));
}

/// A call names an offered tool, spelled as the prompt shows it (its JSON
/// string body): an unoffered one (codex's `apply_patch`: 6 calls in #259's
/// job, 29 in #235's constrained arm) is refused at the byte where it leaves every offered name; a name
/// that is a prefix of another is still whole; the mask at the name's start
/// allows only what begins one.
#[test]
fn hermes_names() {
    let (v, sp) = toy_vocab(17, 400);
    let offered = ["exec_command", "read", "read_file", "a\\\"b"];
    let g = Arc::new(Grammar::hermes(v.clone(), &names(&offered)).unwrap());
    let call = |n: &str| format!("\n{{\"name\": \"{n}\", \"arguments\": {{}}}}\n</tool_call>");
    for n in offered {
        assert_eq!(held(&g, sp[0], &call(n)), (None, true), "{n}");
    }
    let at = call("").find(": \"").unwrap() + 3;
    for (n, off) in [("apply_patch", 1), ("exec", 4), ("read_files", 9), ("a\"b", 1), ("", 0)] {
        assert_eq!(held(&g, sp[0], &call(n)).0, Some(at + off), "{n:?}");
    }
    // The mask at the name's first byte: `e`, `r`, `a` and the tokens that
    // go on into an offered name, nothing else.
    let mut c = Constraint::new(g.clone());
    assert!(c.accept(sp[0]));
    for &b in b"\n{\"name\": \"" {
        assert!(c.accept(b as u32));
    }
    let m = c.mask().unwrap().unwrap();
    for id in m.ids() {
        let t = v.bytes(id);
        assert!(
            offered.iter().any(|n| {
                let full = format!("{n}\", ");
                full.as_bytes().starts_with(t) || t.starts_with(n.as_bytes())
            }),
            "{:?} allowed at the name",
            String::from_utf8_lossy(t)
        );
    }
    assert!(m.allows(b'e' as u32) && m.allows(b'r' as u32) && !m.allows(b'x' as u32));
}

/// An array keeps its first item's kind (#272: pi's calls that wrote
/// `false` after an edit object 1900 times).
#[test]
fn one_kind_arrays() {
    let (v, sp) = toy_vocab(19, 50);
    let g = Arc::new(hermes(v));
    let call = |a: &str| format!("\n{{\"name\": \"x\", \"arguments\": {{\"q\": {a}}}}}\n</tool_call>");
    let at = call("").find("\"q\": ").unwrap() + 5;
    for ok in ["[]", "[1, -2.5]", "[{}, {\"a\": [true]}]", "[true, false]", "[[1], [\"a\"]]", "[null]"] {
        assert_eq!(held(&g, sp[0], &call(ok)), (None, true), "{ok}");
    }
    for (bad, off) in [("[{}, false]", 5), ("[1, \"a\"]", 4), ("[[1], [2, \"a\"]]", 10), ("[true, null]", 7)] {
        assert_eq!(held(&g, sp[0], &call(bad)).0, Some(at + off), "{bad}");
    }
}

/// The finisher's masks equal walking every token through it (as the
/// grammar's do: `masks_match_naive`), and from any state of a random call
/// it closes the call: following any token it allows reaches the end.
#[test]
fn finisher_closes_from_anywhere() {
    let (v, sp) = toy_vocab(23, 600);
    let g = hermes(v.clone());
    let (m, f) = (&g.masks, &g.finish);
    let mut s = 0x0dd_ba11u64;
    let mut buf = Vec::new();
    let mut closed = 0;
    for walk in 0..30 {
        let (mut node, mut stack) = (g.start, Vec::<NodeId>::new());
        for step in 0..(20 + walk * 3) {
            let want = naive(m, node, &stack);
            let Some(&id) = (!want.is_empty()).then(|| pick(&mut s, &want)) else {
                break;
            };
            node = m.walk(node, &stack, id, &mut buf).unwrap();
            stack = buf.clone();
            if m.graph.nodes[node as usize].accept && stack.is_empty() {
                break;
            }
            if step % 5 != 0 {
                continue;
            }
            // From here, the shortest way out.
            let (mut n2, mut st2) = (node, stack.clone());
            for k in 0..400 {
                let tm = f.mask(n2, &st2, &mut buf);
                let fw = naive(f, n2, &st2);
                assert_eq!(ids(&tm), fw, "walk {walk}: finisher mask at node {n2}");
                assert!(!fw.is_empty(), "walk {walk}: the finisher allows nothing at node {n2}");
                for &x in &fw {
                    assert!(
                        m.walk(n2, &st2, x, &mut Vec::new()).is_some(),
                        "the finisher allows what the grammar doesn't"
                    );
                    assert!(x != sp[0] && x != sp[2], "special {x}");
                }
                n2 = f.walk(n2, &st2, *pick(&mut s, &fw), &mut buf).unwrap();
                st2 = buf.clone();
                if f.graph.nodes[n2 as usize].accept && st2.is_empty() {
                    closed += 1;
                    break;
                }
                assert!(k < 399, "walk {walk}: the finisher didn't close");
            }
        }
    }
    assert!(closed > 100, "{closed} closes");
}

/// A call whose last `STALL_WINDOW` tokens are at most `STALL_IDS` ids is
/// closed the shortest way: `false, false, ...` in an array, `\n\n\n...` in
/// a string. A long call that keeps writing other tokens doesn't stall.
#[test]
fn stalled_call_closes() {
    use super::{STALL_IDS, STALL_WINDOW};
    let (v, sp) = toy_vocab(29, 50);
    let g = Arc::new(hermes(v.clone()));
    let tok = |s: &str| -> u32 {
        (0..v.len() as u32)
            .find(|&id| v.bytes(id) == s.as_bytes() && !v.is_special(id))
            .unwrap_or_else(|| panic!("no token {s:?}"))
    };
    let start = |c: &mut Constraint, head: &str| {
        assert!(c.accept(sp[0]));
        for &b in head.as_bytes() {
            assert!(c.accept(b as u32), "{head:?}");
        }
    };
    // `[false, false, ...`: three ids (`false`, `,`, ` `).
    let mut c = Constraint::new(g.clone());
    start(&mut c, "\n{\"name\": \"x\", \"arguments\": {\"edits\": [false");
    let n0 = g.counters.stalls.load(std::sync::atomic::Ordering::Relaxed);
    let mut taken = 0;
    while !c.stalled() {
        for t in [",", " ", "false"] {
            assert!(c.accept(tok(t)));
            taken += 1;
        }
        assert!(taken < 2 * STALL_WINDOW, "never stalled");
    }
    assert!(taken + 13 >= STALL_WINDOW, "stalled after {taken}");
    assert_eq!(g.counters.stalls.load(std::sync::atomic::Ordering::Relaxed), n0 + 1);
    // Now only the shortest close: `]`, `}`, `}`, then `</tool_call>`.
    for want in ["]", "}", "}"] {
        let m = c.mask().unwrap().unwrap();
        let only: Vec<String> = m.ids().map(|i| String::from_utf8_lossy(v.bytes(i)).into()).collect();
        assert!(only.iter().all(|t| t.starts_with(want)), "{want}: {only:?}");
        assert!(m.allows(tok(want)) && !m.allows(tok(",")) && !m.allows(tok(" ")));
        assert!(c.accept(tok(want)));
    }
    let m = c.mask().unwrap().unwrap();
    assert!(m.allows(sp[1]) && !m.allows(b'\n' as u32));
    assert!(c.accept(sp[1]) && !c.engaged() && !c.stalled());
    // The next call starts fresh.
    start(&mut c, "\n{\"name\": \"x\", \"arguments\": {\"cmd\": \"");
    assert!(!c.stalled());
    let mut taken = 0;
    while !c.stalled() {
        assert!(c.accept(tok("\\n")));
        taken += 1;
        assert!(taken <= STALL_WINDOW, "never stalled");
    }
    // The window's other ids: the head's last few (`: "`).
    assert!(taken + STALL_IDS >= STALL_WINDOW, "stalled after {taken}");
    let m = c.mask().unwrap().unwrap();
    assert!(m.allows(b'"' as u32) && !m.allows(tok("\\n")) && !m.allows(b'a' as u32));
    // A call writing more than `STALL_IDS` ids over and over never stalls.
    let mut c = Constraint::new(g.clone());
    start(&mut c, "\n{\"name\": \"x\", \"arguments\": {\"cmd\": \"");
    let five: Vec<u32> = ["a", "b", "c", "x", "y"][..STALL_IDS + 1].iter().map(|t| tok(t)).collect();
    for i in 0..4 * STALL_WINDOW {
        assert!(c.accept(five[i % five.len()]));
    }
    assert!(!c.stalled());
}

/// Triggers: `<tool_call>` as its token or spelled out, across tokens or
/// inside one, with the bytes after it in that token taken by the grammar
/// (refused: left free, counted); suspended inside the reply's reasoning
/// (which only its first token opens, as `TextOut` reads it).
#[test]
fn hermes_triggers() {
    use std::sync::atomic::Ordering::Relaxed;
    let (v, sp) = toy_vocab(31, 30);
    let mut pieces: Vec<Vec<u8>> = (0..v.len() as u32).map(|i| v.bytes(i).to_vec()).collect();
    let extra = ["<tool", "_call", ">", ">\n", ">{", ">abc", "<tool_call>\n{", "<think>", "</think>"];
    let first = pieces.len() as u32;
    for p in extra {
        pieces.push(p.as_bytes().to_vec());
    }
    let think = (first + 7, first + 8);
    let mut specials = sp.clone();
    specials.extend([think.0, think.1]);
    let v = Arc::new(Vocab::from_pieces(pieces.len() + 1, &pieces, &specials));
    let g = Arc::new(hermes(v.clone()));
    let t = |s: &str| first + extra.iter().position(|x| *x == s).unwrap() as u32;
    let bytes = |c: &mut Constraint, s: &str| s.bytes().for_each(|b| assert!(c.accept(b as u32)));
    // Spelled across tokens, then the call.
    let mut c = Constraint::new(g.clone());
    bytes(&mut c, "Sure. ");
    for x in ["<tool", "_call"] {
        assert!(c.accept(t(x)) && !c.engaged());
    }
    assert!(c.accept(t(">\n")) && c.engaged(), "engaged, the newline taken");
    let m = c.mask().unwrap().unwrap();
    // After a newline: a second one, indentation, or the object.
    assert!(m.allows(b'{' as u32) && m.allows(b'\n' as u32) && m.allows(b' ' as u32));
    assert!(!m.allows(b'\r' as u32) && !m.allows(b'x' as u32) && !m.allows(sp[1]));
    // Byte by byte, and inside one token with its tail.
    let mut c = Constraint::new(g.clone());
    bytes(&mut c, "<tool_call");
    assert!(!c.engaged());
    assert!(c.accept(t(">{")) && c.engaged());
    assert!(c.mask().unwrap().unwrap().allows(b'"' as u32));
    let mut c = Constraint::new(g.clone());
    assert!(c.accept(t("<tool_call>\n{")) && c.engaged());
    // A tail the grammar refuses: free, counted.
    let r0 = g.counters.triggers_rejected.load(Relaxed);
    let mut c = Constraint::new(g.clone());
    for x in ["<tool", "_call", ">abc"] {
        assert!(c.accept(t(x)));
    }
    assert!(!c.engaged());
    assert_eq!(g.counters.triggers_rejected.load(Relaxed), r0 + 1);
    // Not a trigger: almost, or broken up by other text.
    let mut c = Constraint::new(g.clone());
    bytes(&mut c, "<tool_cal> <tool <tool_call ");
    assert!(c.accept(t(">")) && !c.engaged());
    // Reasoning: a reply that opens with <think> holds no call until
    // </think>, spelled or as the token.
    let mut c = Constraint::new(g.clone()).with_think(Some(think));
    assert!(c.accept(think.0));
    assert!(c.accept(sp[0]) && !c.engaged());
    bytes(&mut c, "<tool_call>");
    assert!(!c.engaged());
    assert!(c.accept(think.1));
    bytes(&mut c, "\n\n");
    assert!(c.accept(sp[0]) && c.engaged());
    // <think> anywhere but first is text, not reasoning (as `TextOut`).
    let mut c = Constraint::new(g.clone()).with_think(Some(think));
    bytes(&mut c, "Hi ");
    assert!(c.accept(think.0) && c.accept(sp[0]) && c.engaged());
    // Spelled during a call is string text; after it, armed again.
    let mut c = Constraint::new(g.clone());
    assert!(c.accept(sp[0]));
    bytes(&mut c, "{\"name\": \"x\", \"arguments\": {\"s\": \"<tool_call>\"}}");
    assert!(c.engaged());
    assert!(c.accept(sp[1]) && !c.engaged());
    bytes(&mut c, "\n<tool_call>");
    assert!(c.engaged());
}

/// tool_choice required and named: the prompt opens the call (its text
/// after `<tool_call>`, `tools::Rendered::seed`), and the constraint starts
/// engaged where that text leaves it: at the name (only offered names'
/// starts), or at the arguments (only `{`). A seed the grammar refuses
/// (a named tool it doesn't offer) is an error, never a wrong state.
#[test]
fn forced_seeds() {
    let (v, _) = toy_vocab(37, 30);
    let g = Arc::new(hermes(v.clone()));
    let c = Constraint::forced(g.clone(), b"\n{\"name\": \"").unwrap();
    assert!(c.engaged() && c.in_string());
    let mut c2 = c.clone();
    let m = c2.mask().unwrap().unwrap();
    for b in 0..=255u8 {
        let starts = NAMES.iter().any(|n| n.as_bytes()[0] == b);
        assert_eq!(m.allows(b as u32), starts, "byte {:?}", b as char);
    }
    let mut c = Constraint::forced(g.clone(), b"\n{\"name\": \"get\", \"arguments\": ").unwrap();
    let m = c.mask().unwrap().unwrap();
    for b in 0..=255u8 {
        assert_eq!(m.allows(b as u32), b == b'{', "byte {:?}", b as char);
    }
    for &b in b"{\"a\": 1}}\n" {
        assert!(c.accept(b as u32));
    }
    assert!(c.engaged());
    assert!(Constraint::forced(g.clone(), b"\n{\"name\": \"rm\", \"arguments\": ").is_err());
    assert!(Constraint::forced(g, b"\n[").is_err());
}

/// `Hermes`: one grammar per set of names (order and duplicates don't
/// matter), kept for the next request, at most `Hermes::KEEP`, the least
/// recently used dropped; built warm (its call's masks compiled).
#[test]
fn hermes_grammars_kept() {
    let (v, _) = toy_vocab(41, 100);
    let h = super::Hermes::new(v, Default::default()).unwrap();
    let a = h.grammar(&names(&["b", "a", "a"])).unwrap();
    assert!(Arc::ptr_eq(&a, &h.grammar(&names(&["a", "b"])).unwrap()));
    assert!(a.masks.cached() > 20, "warmed: {} masks", a.masks.cached());
    let mut warm = b"\n{\"name\": \"b".to_vec();
    warm.extend_from_slice(super::PREWARM_ARGS);
    assert!(a.prewarm(&warm), "the warm-up call is one the grammar takes");
    let other = h.grammar(&names(&["c"])).unwrap();
    assert!(!Arc::ptr_eq(&a, &other));
    for i in 0..super::Hermes::KEEP {
        h.grammar(&names(&[&format!("t{i}")])).unwrap();
    }
    assert!(!Arc::ptr_eq(&a, &h.grammar(&names(&["a", "b"])).unwrap()), "dropped");
    assert!(h.grammar(&[]).is_err());
}

/// AXON_CONSTRAIN is read strictly (as AXON_SHARE_KV): 0, 1, or unset or
/// empty for the default (on).
#[test]
fn constrain_env_strict() {
    use super::env_value;
    assert_eq!(env_value(None), Ok(None));
    assert_eq!(env_value(Some("")), Ok(None));
    assert_eq!(env_value(Some(" 0 ")), Ok(Some(false)));
    assert_eq!(env_value(Some("1")), Ok(Some(true)));
    for bad in ["off", "false", "on", "true", "2", "yes"] {
        let e = env_value(Some(bad)).unwrap_err();
        assert!(e.contains("expected 0") && e.contains(bad), "{e}");
    }
}

/// The spec's §4 parser agreement: 1000 random calls through the Hermes
/// automaton (tokens the mask allows, picked at random, then the shortest
/// close), each the text of a reply after its `<tool_call>` token: the
/// server's tools parser makes exactly one call of it, named one of the
/// offered names, its arguments a JSON object; nothing else but blanks.
#[test]
fn parser_agrees_on_random_calls() {
    use crate::serve::tools::{Piece, hermes_pieces};
    let (v, sp) = toy_vocab(43, 600);
    let g = Arc::new(hermes(v.clone()));
    let mut s = 0x5eed_cafe_f00du64;
    let (mut toks, mut spelled_in_str) = (0, 0);
    for walk in 0..1000 {
        let mut c = Constraint::new(g.clone());
        assert!(c.accept(sp[0]));
        let mut bytes = b"<tool_call>".to_vec();
        let budget = 5 + walk % 150;
        let mut n = 0;
        while c.engaged() {
            if n == budget {
                c.close_now();
            }
            let m = c.mask().unwrap().expect("engaged");
            let ids: Vec<u32> = m.ids().collect();
            let id = *pick(&mut s, &ids);
            assert!(c.accept(id));
            bytes.extend_from_slice(v.bytes(id));
            n += 1;
            assert!(n < budget + 400, "walk {walk}: no close");
        }
        toks += n;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let body = &text[11..text.rfind("</tool_call>").unwrap()];
        spelled_in_str += usize::from(body.contains("</tool_call>"));
        let pieces = hermes_pieces(&text);
        let calls: Vec<_> = (pieces.iter())
            .filter_map(|p| match p {
                Piece::Call(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1, "walk {walk}: {text:?} -> {pieces:?}");
        assert!(NAMES.contains(&calls[0].name.as_str()), "walk {walk}: {:?}", calls[0].name);
        assert!(
            matches!(parse_json(&calls[0].args), Some(J::Obj(_))),
            "walk {walk}: arguments {:?}",
            calls[0].args
        );
        for p in &pieces {
            if let Piece::Text(t) = p {
                assert!(t.trim().is_empty(), "walk {walk}: text {t:?} from {text:?}");
            }
        }
    }
    eprintln!("{toks} tokens; {spelled_in_str} calls spell </tool_call> inside a string");
}

/// A draft's copy (`Constraint::draft_copy`) takes the same tokens and
/// reaches the same state, but only the real constraint's tokens and stalls
/// are counted (a drafted reply would count its guesses twice).
#[test]
fn draft_copies_not_counted() {
    use std::sync::atomic::Ordering::Relaxed;
    let (v, sp) = toy_vocab(47, 30);
    let g = Arc::new(hermes(v.clone()));
    let mut c = Constraint::new(g.clone());
    assert!(c.accept(sp[0]));
    let t0 = g.counters.tokens.load(Relaxed);
    let mut d = c.draft_copy();
    for &b in b"\n{\"name\": \"x\"" {
        assert!(d.accept(b as u32));
    }
    assert_eq!(g.counters.tokens.load(Relaxed), t0, "a draft's tokens");
    for &b in b"\n{\"name\": \"x\"" {
        assert!(c.accept(b as u32));
    }
    assert_eq!(g.counters.tokens.load(Relaxed), t0 + 13);
    assert_eq!(c.phase, d.phase);
}
