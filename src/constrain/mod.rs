//! Constrained decoding (#259, #116; the spec is
//! specs/constrained-decoding.md in the overnight workspace): a request's
//! tokens restricted to what a grammar allows, only where it is engaged.
//!
//! A `Constraint` is one request's state: free until its trigger (Hermes:
//! the `<tool_call>` token), then engaged in a byte-level pushdown
//! automaton (pda.rs: the JSON library plus the format's frame) until the
//! grammar completes (Hermes: `</tool_call>`), then free again (armed for
//! the next call: the maintainer's ruling 2 on the spec's §6, lazy re-arm,
//! so parallel calls and prose after a call stay possible). While engaged,
//! `mask` gives the tokens the automaton takes next (mask.rs), and the
//! sampler only ever sees candidates the mask allows
//! (`sample::cands_masked`, `Lm::forward_cands_masked`).
//!
//! Bits: a free position takes the request's normal head (`RowHead::Argmax`
//! or `Sample` without a mask), so a constrained request's tokens equal an
//! unconstrained one's until the trigger, and a request without a
//! constraint runs no new code at all. An engaged position's candidates are
//! a pure function of (logits, sampling spec, penalty window, mask), and
//! the mask is a function of the tokens so far, so decode, batched steps
//! and speculative verify agree bit for bit (docs/determinism.md).
//!
//! Special tokens (control and user-defined) are symbols to the automaton,
//! never bytes: inside a JSON string none of them (EOS, `<|im_end|>`,
//! `<tool_call>`) can be picked; the grammar names the ones it wants (the
//! Hermes close takes `</tool_call>` as its token or spelled out).
//!
//! Opt-in in this PR (`AXON_CONSTRAIN=1`, `serve::Opts::constrain`): Qwen2.5's
//! Hermes tool calls with `tool_choice` auto. PR B turns it on by default
//! (spelled triggers, `<think>` suspension, forced seeds, the name enum).

mod mask;
#[cfg(test)]
mod model_tests;
mod pda;
#[cfg(test)]
mod tests;
mod vocab;

pub use mask::TokenMask;
pub use vocab::Vocab;

use crate::sha256::Sha256;
use mask::Masks;
use pda::{Builder, ByteSet, NodeId};
use std::sync::Arc;

/// A compiled grammar on one vocabulary, with its mask cache. Built once
/// per (model, format) and shared by the requests that use it.
pub struct Grammar {
    masks: Masks,
    /// Where an engaged constraint starts.
    start: NodeId,
    /// Free: these tokens engage it.
    triggers: Vec<u32>,
    /// What the identity calls it: `<name> <hash>`.
    field: String,
}

impl Grammar {
    /// Hermes (Qwen2.5's `<tool_call>` format) with free JSON: once
    /// `<tool_call>` is picked, `ws* {ws* "name" ws* : ws* STRING ws* ,
    /// ws* "arguments" ws* : ws* OBJECT ws* } ws* </tool_call>` (the close
    /// as its token or spelled), where STRING is any JSON string and
    /// OBJECT any JSON object (nesting capped at `pda::MAX_DEPTH`). None if
    /// the vocabulary has no `<tool_call>` token, or isn't
    /// `Vocab::byte_complete` (a mask could come up empty).
    pub fn hermes(vocab: Arc<Vocab>) -> Option<Grammar> {
        if !vocab.byte_complete() {
            return None;
        }
        let open = vocab.special_of(b"<tool_call>")?;
        let close = vocab.special_of(b"</tool_call>");
        let mut b = Builder::default();
        let json = pda::json(&mut b);
        let h0 = b.node();
        b.ws_loop(h0);
        let h1 = b.node();
        b.bytes(h0, ByteSet::of(b"{"), h1);
        b.ws_loop(h1);
        let h2 = b.lit(h1, b"\"name\"");
        b.ws_loop(h2);
        let h3 = b.node();
        b.bytes(h2, ByteSet::of(b":"), h3);
        b.ws_loop(h3);
        let h4 = b.node();
        b.call(h3, json.string, h4);
        b.ws_loop(h4);
        let h5 = b.node();
        b.bytes(h4, ByteSet::of(b","), h5);
        b.ws_loop(h5);
        let h6 = b.lit(h5, b"\"arguments\"");
        b.ws_loop(h6);
        let h7 = b.node();
        b.bytes(h6, ByteSet::of(b":"), h7);
        b.ws_loop(h7);
        let h8 = b.node();
        b.call(h7, json.object, h8);
        b.ws_loop(h8);
        let h9 = b.node();
        b.bytes(h8, ByteSet::of(b"}"), h9);
        b.ws_loop(h9);
        let done = b.node();
        b.set_accept(done);
        if let Some(c) = close {
            b.special(h9, c, done);
        }
        let spelled = b.lit(h9, b"</tool_call");
        b.bytes(spelled, ByteSet::of(b">"), done);
        let graph = b.finish().expect("the Hermes grammar is well formed");
        // The identity names the grammar by what it was built from (PR C
        // adds the names and schemas): a change to it is a new field.
        let desc = format!(
            "axon-constrain/1\nhermes\nopen={open}\nclose={close:?}\nname=string\narguments=object\nmax_depth={}\n",
            pda::MAX_DEPTH
        );
        let mut h = Sha256::new();
        h.update(desc.as_bytes());
        let hex = h.finish_hex()[..16].to_string();
        Some(Grammar {
            masks: Masks::new(graph, vocab),
            start: h0,
            triggers: vec![open],
            field: format!("hermes-json {hex}"),
        })
    }

    /// How the identity's `serve` field names it (`identity::constrain_field`).
    pub fn field(&self) -> &str {
        &self.field
    }
}

/// Why a constrained request can't go on (it fails, never panics).
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintError(pub String);

impl std::fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "constrained decoding: {}", self.0)
    }
}

impl From<ConstraintError> for std::io::Error {
    fn from(e: ConstraintError) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Phase {
    Free,
    On { node: NodeId, stack: Vec<NodeId> },
}

/// One request's constraint: a grammar and where the reply is in it. Its
/// state is a function of the tokens accepted so far; `Clone` copies the
/// state only (drafting advances a clone through the guesses).
#[derive(Clone)]
pub struct Constraint {
    g: Arc<Grammar>,
    phase: Phase,
    buf: Vec<NodeId>,
}

impl Constraint {
    /// A request's constraint, free until a trigger.
    pub fn new(g: Arc<Grammar>) -> Constraint {
        Constraint {
            g,
            phase: Phase::Free,
            buf: Vec::new(),
        }
    }

    /// The tokens allowed next; None while free (the position takes its
    /// normal head). Err if the grammar allows nothing: never on a
    /// `Vocab::byte_complete` vocabulary (from every node a path of bytes
    /// finishes, by `pda::Builder::finish`'s check, without opening a
    /// container in JSON's case, and every byte is a token); a request that
    /// hits it fails alone.
    pub fn mask(&mut self) -> Result<Option<TokenMask>, ConstraintError> {
        let Phase::On { node, stack } = &self.phase else {
            return Ok(None);
        };
        let m = self.g.masks.mask(*node, stack, &mut self.buf);
        if m.count() == 0 {
            return Err(ConstraintError(format!(
                "no token is allowed at grammar node {node}"
            )));
        }
        Ok(Some(m))
    }

    /// Advance by token `id`, just picked. False if the grammar refuses it
    /// (the state is then unchanged): a pick outside the mask.
    pub fn accept(&mut self, id: u32) -> bool {
        let Phase::On { node, stack } = &mut self.phase else {
            if self.g.triggers.contains(&id) {
                self.phase = Phase::On {
                    node: self.g.start,
                    stack: Vec::new(),
                };
            }
            return true;
        };
        let Some(n) = self.g.masks.walk(*node, stack, id, &mut self.buf) else {
            return false;
        };
        // `buf` holds the stack after the token.
        std::mem::swap(stack, &mut self.buf);
        *node = n;
        if self.g.masks.graph.nodes[n as usize].accept && stack.is_empty() {
            // Complete: armed again for the next call.
            self.phase = Phase::Free;
        }
        true
    }

    /// Engaged (masking) now.
    #[cfg_attr(not(test), allow(dead_code))] // PR B: the Parser's state
    pub fn engaged(&self) -> bool {
        matches!(self.phase, Phase::On { .. })
    }

    /// Engaged inside a JSON string: what PR B's Parser asks to tell a
    /// spelled `</tool_call>` in a string value from the call's real close
    /// (the maintainer's ruling 5 on the spec's §6).
    #[cfg_attr(not(test), allow(dead_code))] // PR B
    pub fn in_string(&self) -> bool {
        match &self.phase {
            Phase::On { node, .. } => self.g.masks.graph.nodes[*node as usize].in_str,
            Phase::Free => false,
        }
    }

    pub fn grammar(&self) -> &Arc<Grammar> {
        &self.g
    }
}

/// `AXON_CONSTRAIN`: constrained tool calls on ("1"), off otherwise (PR A:
/// opt-in; `serve::Opts::constrain` overrides it).
pub fn env_on() -> bool {
    std::env::var_os("AXON_CONSTRAIN").is_some_and(|v| v == "1")
}
