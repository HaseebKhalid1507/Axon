//! Constrained decoding (#259, #116; the spec is
//! specs/constrained-decoding.md in the overnight workspace): a request's
//! tokens restricted to what a grammar allows, only where it is engaged.
//!
//! A `Constraint` is one request's state: free until its trigger (Hermes:
//! `<tool_call>`, as its token or spelled out, outside the reply's
//! reasoning), then engaged in a byte-level pushdown automaton (pda.rs: the
//! JSON library plus the format's frame) until the grammar completes
//! (Hermes: `</tool_call>`), then free again (armed for the next call: the
//! maintainer's ruling 2 on the spec's §6, lazy re-arm, so parallel calls
//! and prose after a call stay possible). While engaged, `mask` gives the
//! tokens the automaton takes next (mask.rs), and the sampler only ever
//! sees candidates the mask allows (`sample::cands_masked`,
//! `Lm::forward_cands_masked`). A reply the prompt opened a call for
//! (`tool_choice` required or named) starts engaged (`Constraint::forced`).
//!
//! Bits: a free position takes the request's normal head (`RowHead::Argmax`
//! or `Sample` without a mask), so a constrained request's tokens equal an
//! unconstrained one's until the trigger, and a request without a
//! constraint takes none of the constrained paths. An engaged position's
//! candidates are a pure function of (logits, sampling spec, penalty
//! window, mask), and the mask is a function of the tokens so far, so
//! decode, batched steps and speculative verify agree bit for bit
//! (docs/determinism.md).
//!
//! Special tokens (control and user-defined) are symbols to the automaton,
//! never bytes: inside a JSON string none of them (EOS, `<|im_end|>`,
//! `<tool_call>`) can be picked; the grammar names the ones it wants (the
//! Hermes close takes `</tool_call>` as its token or spelled out).
//!
//! On by default in `axon serve` for the Hermes families (Qwen2.5, Qwen3;
//! `Hermes`, one per server): a call names one of the request's tools
//! (`Grammar::hermes`'s name enum) and its arguments are a JSON object.
//! A held call can't run on (#272): whitespace is bounded as llama.cpp
//! bounds it and an array keeps its first item's kind (`pda::Style::HELD`),
//! and a call that stalls (its last 256 tokens no more than 8 distinct
//! ones, `false, false, ...` or `\n\n\n...` in a string, or its last 512 one
//! cycle of at most 64 tokens over and over: `STALL_WINDOW`) is closed the
//! shortest way (`pda::Graph::finisher`). The kill switch is
//! `--no-constrain` / `AXON_CONSTRAIN=0`. Per-tool argument schemas are the
//! spec's PR C.

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
use pda::{Builder, ByteSet, NodeId, Step, Style, Sym};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Hermes's call markers, as text (a special token decodes to it).
const OPEN: &[u8] = b"<tool_call>";

/// A call stalls, and is then closed the shortest way, when its last
/// `STALL_WINDOW` tokens are at most `STALL_IDS` distinct ids, or its last
/// `LOOP_WINDOW` tokens repeat one cycle of at most `LOOP_PERIOD` tokens.
/// In the acceptance runs' 6,333 calls (codex, opencode, pi, axon and
/// llama-server; split as a byte-level pre-tokenizer splits text, a token
/// being that piece or less), the 13 runaways trip one rule or both after
/// 461 to 50,831 pieces (`, false` 1,900 times; whitespace;
/// `--exclude=*.pyc` over and over; a list of strings in a loop), and no
/// other call comes near: its longest stretch of at most 8 distinct pieces
/// is 28, its longest periodic one 119.
pub const STALL_WINDOW: usize = 256;
pub const STALL_IDS: usize = 8;
pub const LOOP_WINDOW: usize = 512;
pub const LOOP_PERIOD: usize = 64;

/// What the constrained paths count (one per server: `serve::ops::Metrics`
/// shows them; the logits-row fallbacks are `sample::MASK_FALLBACKS`).
#[derive(Default, Debug)]
pub struct Counters {
    /// Tokens decided while engaged (masked).
    pub tokens: AtomicU64,
    /// A spelled `<tool_call>` whose token went on with bytes the grammar
    /// refuses (`<tool_call>abc`): left free.
    pub triggers_rejected: AtomicU64,
    /// Time in `Constraint::mask` (cache lookups, first-visit compiles,
    /// the live check of deep tokens), nanoseconds.
    pub mask_ns: AtomicU64,
    /// Calls closed the shortest way after they stalled.
    pub stalls: AtomicU64,
}

/// A compiled grammar on one vocabulary, with its mask cache. Built once
/// per (model, format, tool names) and shared by the requests that use it
/// (`Hermes::grammar`).
pub struct Grammar {
    masks: Masks,
    /// The same states' shortest way out (`pda::Graph::finisher`): what a
    /// stalled call takes.
    finish: Masks,
    /// Where an engaged constraint starts.
    start: NodeId,
    /// Free: these tokens engage it.
    triggers: Vec<u32>,
    /// What the identity calls it: `<name> <hash>`.
    field: String,
    counters: Arc<Counters>,
}

/// The Hermes grammar's rules, for the identity's hash (`Grammar::field`):
/// what the tokens of a constrained reply depend on besides the request.
/// The names come from the request (as its prompt does), so they aren't in
/// it. A change to the rules is a new line here.
fn hermes_desc(open: u32, close: Option<u32>) -> String {
    format!(
        "axon-constrain/2\nhermes\nopen={open}\nclose={close:?}\nname=enum(offered)\n\
         arguments=object\nmax_depth={}\nspace=llama({})\narrays=one-kind\n\
         stall={}/{},loop={}/{}:shortest-close\n",
        pda::MAX_DEPTH,
        pda::INDENT_MAX,
        STALL_WINDOW,
        STALL_IDS,
        LOOP_WINDOW,
        LOOP_PERIOD
    )
}

/// The identity field of a grammar described by `desc`.
fn field_of(desc: &str) -> String {
    let mut h = Sha256::new();
    h.update(desc.as_bytes());
    format!("hermes-json {}", &h.finish_hex()[..16])
}

/// The vocabulary checks the Hermes grammar needs: (the `<tool_call>` and
/// `</tool_call>` special tokens), or why it can't be held.
fn hermes_markers(vocab: &Vocab) -> Result<(u32, Option<u32>), String> {
    if let Some(b) = vocab.missing_byte() {
        return Err(format!(
            "the vocabulary isn't byte-complete: byte {b:#04x} is no token by itself"
        ));
    }
    let open = (vocab.special_of(OPEN)).ok_or("the vocabulary has no <tool_call> special token")?;
    Ok((open, vocab.special_of(b"</tool_call>")))
}

impl Grammar {
    /// Hermes (Qwen2.5's and Qwen3's `<tool_call>` format): once
    /// `<tool_call>` is taken, `ws {ws "name" ws : ws NAME ws , ws
    /// "arguments" ws : ws OBJECT ws } ws </tool_call>` (the close as its
    /// token or spelled), where NAME is one of `names` (the offered tools'
    /// names, each its JSON string body as the prompt shows it:
    /// `Builder::names`), OBJECT any JSON object (nesting capped at
    /// `pda::MAX_DEPTH`), and `ws` llama.cpp's bounded whitespace
    /// (`Space::Bounded`), inside the arguments too. Err (why, for the
    /// server's startup line) if the vocabulary isn't
    /// `Vocab::byte_complete` (a mask could come up empty), has no
    /// `<tool_call>` special token, or `names` is empty.
    pub fn hermes(vocab: Arc<Vocab>, names: &[Vec<u8>]) -> Result<Grammar, String> {
        let (open, close) = hermes_markers(&vocab)?;
        if names.is_empty() {
            return Err("no tool names to hold a call to".into());
        }
        let mut b = Builder::new(Style::HELD);
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
        b.names(h3, names, h4);
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
        // Names a JSON string can't hold raw (an escape the trie would
        // have to tell apart) fail here: the request goes unconstrained.
        let graph = b.finish()?;
        Ok(Grammar {
            finish: Masks::new(graph.finisher(), vocab.clone()),
            masks: Masks::new(graph, vocab),
            start: h0,
            triggers: vec![open],
            field: field_of(&hermes_desc(open, close)),
            counters: Arc::default(),
        })
    }

    /// How the identity's `serve` field names it (`identity::constrain_field`).
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Compile the masks a call is most likely to meet (`Hermes::grammar`,
    /// on a request's thread): every state of `text`, a call written out
    /// (its opening `<tool_call>` taken), byte by byte. The engine's steps
    /// then rarely compile one (a first visit to the string body walks
    /// nearly the whole vocabulary: ~3 ms at 152k tokens). False if `text`
    /// isn't a call the grammar takes (nothing more is warmed).
    pub fn prewarm(&self, text: &[u8]) -> bool {
        let (mut node, mut stack) = (self.start, Vec::new());
        for &x in text {
            let _ = self.masks.entry(node, &stack);
            if self.masks.graph.step(&mut node, &mut stack, Sym::Byte(x)) != Step::Ok {
                return false;
            }
        }
        let _ = self.masks.entry(node, &stack);
        true
    }
}

/// The Hermes grammars of one server's vocabulary (`serve`): one per set of
/// tool names, built on first use and kept for the next requests that
/// offer the same tools (a harness offers the same set every turn), at
/// most `Hermes::KEEP` of them.
pub struct Hermes {
    vocab: Arc<Vocab>,
    field: String,
    counters: Arc<Counters>,
    /// Most recently used first.
    grammars: Mutex<Vec<KeptGrammar>>,
}

/// A grammar `Hermes` keeps, under its names (sorted, each once).
type KeptGrammar = (Vec<Vec<u8>>, Arc<Grammar>);

impl Hermes {
    /// Grammars kept (each with its own mask cache).
    pub const KEEP: usize = 8;

    /// Err (why) as `Grammar::hermes`'s for this vocabulary.
    pub fn new(vocab: Arc<Vocab>, counters: Arc<Counters>) -> Result<Hermes, String> {
        let (open, close) = hermes_markers(&vocab)?;
        Ok(Hermes {
            vocab,
            field: field_of(&hermes_desc(open, close)),
            counters,
            grammars: Mutex::new(Vec::new()),
        })
    }

    /// How the identity's `serve` field names its grammars (the same for
    /// any names).
    pub fn field(&self) -> &str {
        &self.field
    }

    /// The grammar holding a call to one of `names` (`Grammar::hermes`'s;
    /// order and duplicates don't matter), warmed on a call to the first
    /// (`Grammar::prewarm`) when it's new. Err if it can't be built (no
    /// names, or names the trie can't hold): the request goes free.
    pub fn grammar(&self, names: &[Vec<u8>]) -> Result<Arc<Grammar>, String> {
        let mut key = names.to_vec();
        key.sort();
        key.dedup();
        let lock = || self.grammars.lock().unwrap_or_else(|p| p.into_inner());
        {
            let mut v = lock();
            if let Some(i) = v.iter().position(|(k, _)| *k == key) {
                let e = v.remove(i);
                let g = e.1.clone();
                v.insert(0, e);
                return Ok(g);
            }
        }
        // Built and warmed off the lock (another request may build the
        // same: the first one in is kept).
        let mut g = Grammar::hermes(self.vocab.clone(), &key)?;
        g.counters = self.counters.clone();
        let mut call = b"\n{\"name\": \"".to_vec();
        call.extend_from_slice(&key[0]);
        call.extend_from_slice(PREWARM_ARGS);
        let warm = g.prewarm(&call);
        debug_assert!(warm, "the warm-up call is one the grammar takes");
        let g = Arc::new(g);
        let mut v = lock();
        if let Some(i) = v.iter().position(|(k, _)| *k == key) {
            return Ok(v[i].1.clone());
        }
        v.insert(0, (key, g.clone()));
        v.truncate(Self::KEEP);
        Ok(g)
    }
}

/// `Hermes::grammar`'s warm-up call after its name: strings as keys, values
/// in an object and in an array, numbers, literals, nesting, and the close.
const PREWARM_ARGS: &[u8] = b"\", \"arguments\": {\"a\": \"b c\", \"d\": -1.5e3, \"e\": [true, false], \
    \"f\": [null], \"g\": [2, 3], \"h\": [\"i\", \"j\"], \"k\": [{\"l\": {}}], \"m\": [[]], \"n\": {\"o\": 0}}}\n\
    </tool_call>";

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
    /// The reply's reasoning markers (`<think>`, `</think>`): triggers are
    /// suspended inside it, as `serve`'s `TextOut` reads it (reasoning only
    /// when the reply's first token opens it; its text never reaches the
    /// tools parser, so a `<tool_call>` there is no call).
    think: Option<(u32, u32)>,
    /// No token taken yet.
    first: bool,
    reasoning: bool,
    /// Free: the reply's last bytes outside reasoning that may still begin
    /// a spelled `<tool_call>` (fewer than its length).
    tail: Vec<u8>,
    /// Engaged: the call's last tokens (a ring of at most `STALL_WINDOW`,
    /// the next write at `at`), how many it has taken, for each period p
    /// (`runs[p - 1]`) how many tokens in a row equal the one p before, and
    /// whether it stalled (its states then take `Grammar::finish`'s edges).
    recent: Vec<u32>,
    at: usize,
    taken: usize,
    runs: [u16; LOOP_PERIOD],
    stalled: bool,
    /// A draft's copy (`draft_copy`): its tokens and stalls aren't counted
    /// (the real constraint counts the ones it takes).
    shadow: bool,
}

impl Constraint {
    /// A request's constraint, free until a trigger.
    pub fn new(g: Arc<Grammar>) -> Constraint {
        Constraint {
            g,
            phase: Phase::Free,
            buf: Vec::new(),
            think: None,
            first: true,
            reasoning: false,
            tail: Vec::new(),
            recent: Vec::new(),
            at: 0,
            taken: 0,
            runs: [0; LOOP_PERIOD],
            stalled: false,
            shadow: false,
        }
    }

    /// A copy to advance through a draft's guesses (`sample::
    /// generate_constrained`): the same state, its counts not counted.
    pub fn draft_copy(&self) -> Constraint {
        Constraint {
            shadow: true,
            ..self.clone()
        }
    }

    /// An engaged token taken: into the stall rules (`STALL_WINDOW`'s
    /// note); stalled (the next states take the shortest way out) when one
    /// trips.
    fn note(&mut self, id: u32) {
        if self.stalled {
            return;
        }
        let mut cycle = false;
        for p in 1..=LOOP_PERIOD {
            let r = &mut self.runs[p - 1];
            if self.taken >= p && self.recent[(self.at + STALL_WINDOW - p) % STALL_WINDOW] == id {
                *r += 1;
                cycle |= *r as usize + p >= LOOP_WINDOW;
            } else {
                *r = 0;
            }
        }
        if self.recent.len() < STALL_WINDOW {
            self.recent.push(id);
        } else {
            self.recent[self.at] = id;
        }
        self.at = (self.at + 1) % STALL_WINDOW;
        self.taken += 1;
        if !cycle && (self.recent.len() < STALL_WINDOW || !self.few_ids()) {
            return;
        }
        self.stalled = true;
        if !self.shadow {
            self.g.counters.stalls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The stall window holds at most `STALL_IDS` distinct ids.
    fn few_ids(&self) -> bool {
        let mut seen = [0u32; STALL_IDS];
        let mut n = 0;
        for &x in &self.recent {
            if !seen[..n].contains(&x) {
                if n == STALL_IDS {
                    return false;
                }
                seen[n] = x;
                n += 1;
            }
        }
        true
    }

    /// Triggers suspended inside the reply's reasoning, `think` its
    /// (`<think>`, `</think>`) ids (`serve`'s `Job::think`: None when the
    /// reply doesn't route reasoning).
    pub fn with_think(mut self, think: Option<(u32, u32)>) -> Constraint {
        self.think = think;
        self
    }

    /// A reply the prompt opened a call for (`tool_choice` required or
    /// named: the prompt ends `<tool_call>` then `seed`, the text the tools
    /// parser starts inside the block with): engaged from its first token,
    /// where `seed` left the grammar. Err if the grammar refuses `seed` (a
    /// named tool it can't hold, or a frame this grammar doesn't have).
    pub fn forced(g: Arc<Grammar>, seed: &[u8]) -> Result<Constraint, ConstraintError> {
        let mut c = Constraint::new(g);
        c.first = false;
        c.engage(seed);
        match c.phase {
            Phase::On { .. } => Ok(c),
            Phase::Free => Err(ConstraintError(format!(
                "the prefilled call {:?} isn't one the grammar takes",
                String::from_utf8_lossy(seed)
            ))),
        }
    }

    /// Engage at the grammar's start, then take `rest` (the bytes of the
    /// trigger's token after it) byte by byte. A byte it refuses leaves the
    /// constraint free: false.
    fn engage(&mut self, rest: &[u8]) -> bool {
        self.tail.clear();
        self.recent.clear();
        self.at = 0;
        self.taken = 0;
        self.runs = [0; LOOP_PERIOD];
        self.stalled = false;
        let (mut node, mut stack) = (self.g.start, Vec::new());
        for &x in rest {
            let gr = &self.g.masks.graph;
            if gr.step(&mut node, &mut stack, Sym::Byte(x)) != Step::Ok {
                self.phase = Phase::Free;
                return false;
            }
            if gr.nodes[node as usize].accept && stack.is_empty() {
                // A whole call inside one token (never in practice): done.
                self.phase = Phase::Free;
                return true;
            }
        }
        self.phase = Phase::On { node, stack };
        true
    }

    /// Free: take token `id`. Reasoning (`think`) as `TextOut` reads it;
    /// outside it, `<tool_call>` engages, as its token or once the reply's
    /// bytes spell it (the tools parser reads text, so both open a block).
    fn free(&mut self, id: u32) {
        let first = std::mem::replace(&mut self.first, false);
        if let Some((open, close)) = self.think {
            if first && id == open {
                self.reasoning = true;
                return;
            }
            if self.reasoning {
                self.reasoning = id != close;
                return;
            }
        }
        if self.g.triggers.contains(&id) {
            self.engage(&[]);
            return;
        }
        let v = &self.g.masks.vocab;
        self.tail.extend_from_slice(v.bytes(id));
        if let Some(at) = (self.tail.windows(OPEN.len())).position(|w| w == OPEN) {
            let rest = self.tail[at + OPEN.len()..].to_vec();
            if !self.engage(&rest) {
                // `<tool_call>abc`: the tools parser holds a block the
                // grammar can't; it decides it as it always has.
                if !self.shadow {
                    self.g
                        .counters
                        .triggers_rejected
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            return;
        }
        // Keep what may still begin `<tool_call>`.
        let keep = (1..OPEN.len())
            .rev()
            .find(|&n| self.tail.ends_with(&OPEN[..n]))
            .unwrap_or(0);
        self.tail.drain(..self.tail.len() - keep);
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
        let t = std::time::Instant::now();
        let masks = match self.stalled {
            true => &self.g.finish,
            false => &self.g.masks,
        };
        let m = masks.mask(*node, stack, &mut self.buf);
        let ns = t.elapsed().as_nanos() as u64;
        self.g.counters.mask_ns.fetch_add(ns, Ordering::Relaxed);
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
            self.free(id);
            return true;
        };
        let masks = match self.stalled {
            true => &self.g.finish,
            false => &self.g.masks,
        };
        let Some(n) = masks.walk(*node, stack, id, &mut self.buf) else {
            return false;
        };
        if !self.shadow {
            self.g.counters.tokens.fetch_add(1, Ordering::Relaxed);
        }
        // `buf` holds the stack after the token.
        std::mem::swap(stack, &mut self.buf);
        *node = n;
        if self.g.masks.graph.nodes[n as usize].accept && stack.is_empty() {
            // Complete: armed again for the next call.
            self.phase = Phase::Free;
        } else {
            self.note(id);
        }
        true
    }

    /// Close the call the shortest way from here, as a stalled one is.
    #[cfg(test)]
    pub(crate) fn close_now(&mut self) {
        self.stalled = true;
    }

    /// Stalled: the call is being closed the shortest way (`note`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stalled(&self) -> bool {
        self.stalled && self.engaged()
    }

    /// Engaged (masking) now.
    #[cfg_attr(not(test), allow(dead_code))] // PR B: the Parser's state
    pub fn engaged(&self) -> bool {
        matches!(self.phase, Phase::On { .. })
    }

    /// Engaged inside a JSON string: tells a spelled `</tool_call>` in a
    /// string from the call's real close (the maintainer's ruling 5 on the
    /// spec's §6). #260's Parser already reads it as text inside a string
    /// in a call's arguments, from its own scan; in the name string, and
    /// when calls are decided whole, the tag still ends the block (PR B).
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

/// AXON_CONSTRAIN's value `v` (None: unset), spaces around it trimmed: "0"
/// off (the kill switch), "1" on, unset or empty None (the default: on).
/// Anything else is an error naming the valid values (the server refuses
/// to start, as for AXON_SHARE_KV; PR A read only "1" as on).
pub fn env_value(v: Option<&str>) -> Result<Option<bool>, String> {
    match v.map(str::trim) {
        None | Some("") => Ok(None),
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        Some(_) => Err(format!(
            "AXON_CONSTRAIN={:?}: expected 0 (constrained tool calls off) or 1 (on), or unset \
             or empty for the default (on)",
            v.unwrap_or_default()
        )),
    }
}

/// `env_value` of AXON_CONSTRAIN (a value that isn't UTF-8 is an error).
pub fn env() -> Result<Option<bool>, String> {
    match std::env::var("AXON_CONSTRAIN") {
        Ok(v) => env_value(Some(&v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("AXON_CONSTRAIN isn't UTF-8: expected 0 or 1, or unset".into())
        }
    }
}
