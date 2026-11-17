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
//! bounds it (`pda::Style::HELD`), and a call that stalls (its last 2048
//! tokens one block of at most 256 over and over: `, false, false, ...`,
//! `\n\n\n...` in a string, `--exclude=*.pyc --exclude=*.pyc ...`:
//! `LOOP_WINDOW`) ends the reply right there (`Constraint::with_end`), the
//! call left as written: cut off and unclosed. The server returns it as it
//! returns a call `max_tokens` cut off mid-arguments: its arguments aren't
//! JSON, and the reply finishes `length` (`Constraint::cut`; Responses:
//! `incomplete`), so a client that refuses a cut call refuses it (pi runs
//! a repaired call unless the finish is `length`: Fable's review of #274,
//! the lead's ruling of 11-17). Closing it the shortest way
//! instead (`pda::Graph::finisher`, as #274 first did) made a valid-looking
//! call that a harness applied: a model that dropped a string's closing
//! quote wrote on inside the string until the rule fired, and the closed
//! call wrote that text into a file (the lead's ruling of 11-17). A stall
//! is logged and counted (`Constraint::stall`, `Counters::stalls`);
//! `AXON_CONSTRAIN_STALL=0` turns the rule off alone (`stall_env`). The
//! kill switch is `--no-constrain` / `AXON_CONSTRAIN=0`. Per-tool argument
//! schemas are the spec's PR C.

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

/// A held call stalls, and the reply ends there, when its last
/// `LOOP_WINDOW` tokens repeat one block of at most `LOOP_PERIOD` tokens
/// (written twice at least): a model in a loop. That cuts the call where it
/// stalled, so the window is wide: when in doubt, don't fire
/// (the lead's ruling of 11-15 on shady's review of #274). In the
/// acceptance runs' 6,333 calls (codex, opencode, pi, axon and
/// llama-server; split as a byte-level pre-tokenizer splits text, a token
/// being that piece or less), 11 of the 13 runaways trip it (periodic
/// stretches of 2,144 to 50,831 pieces: `, false` 1,900 times, whitespace,
/// `--exclude=*.pyc` over and over, a list of strings in a loop); the other
/// two, a `, false` loop of 1,963 pieces and a list loop of 623, ended by
/// themselves first. No other call comes near: 383 pieces periodic at most
/// (119 with periods up to 64). Data that fills 2,048 tokens with one
/// repeated block (8 or more times over for a 256-token block, more for a
/// shorter one) trips it: 256 identical `<div class="cell"></div>` lines after ~2,085
/// tokens, a 32×32 matrix of zeros after ~2,070 (the PR's table of shady's
/// and Fable's cases).
///
/// A second rule closed a call whose last 1,024 tokens were at most 8
/// distinct ids, until Fable's review of #274: it caught one runaway more
/// (that `, false` loop), but it cut data that doesn't repeat (400 dice
/// rolls, a 30×20 tile map of 4 tile ids, a 40×20 one-hot CSV), and at a
/// 2,048 window it caught no runaway this rule misses and still cut a
/// 2,048-bit string and a 32×32 bitmap. Dropped (the lead's ruling of 11-16).
///
/// `LOOP_PERIOD` was 64 until a replay of a runaway (cbcheck, 11-15) wrote
/// one 72-token block of code over and over, 53 times, to `max_tokens`. At
/// 256 a cycle must still fill `LOOP_WINDOW` (8 or more times over), and
/// the acceptance calls' longest periodic stretch at 512 is 522 pieces.
pub const LOOP_WINDOW: usize = 2048;
pub const LOOP_PERIOD: usize = 256;

/// The stall rule's window: `LOOP_WINDOW` in a server, smaller in the serve
/// tests (`serve::Opts::constrain_stall`), so a call there stalls in a few
/// tokens. A grammar without one (`AXON_CONSTRAIN_STALL=0`) never closes a
/// call early: a call in a loop runs to `max_tokens`, as before #274.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stall {
    pub window: usize,
}

impl Stall {
    pub const DEFAULT: Stall = Stall {
        window: LOOP_WINDOW,
    };
}

/// A call that stalled (`Constraint::stall`): after how many of its tokens,
/// the block it repeated (its length: the stretch's smallest period), the
/// window it filled, and whether that ended the reply with the call cut off
/// (`cut`: a constraint `with_end`, the server's) or closed the call the
/// shortest way (no end tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stalled {
    pub after: usize,
    pub period: usize,
    pub window: usize,
    pub cut: bool,
}

impl std::fmt::Display for Stalled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = if self.period == 1 { "" } else { "s" };
        let then = match self.cut {
            true => {
                "the reply ends there, finish `length`, the call cut off unclosed as max_tokens \
                 cuts one (its arguments aren't JSON)"
            }
            false => "the call is closed the shortest way",
        };
        write!(
            f,
            "a held tool call stalled after {} tokens (its last {} repeat one block of {} \
             token{s}): {then}; AXON_CONSTRAIN_STALL=0 turns this off",
            self.after, self.window, self.period
        )
    }
}

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
    /// Calls that stalled (the reply ended there, the call cut off).
    pub stalls: AtomicU64,
    /// Tools requests left unconstrained: the grammar couldn't hold their
    /// tools (`Hermes::grammar`) or the call their prompt opened
    /// (`Constraint::forced`).
    pub unheld: AtomicU64,
    /// Mask caches started over (`mask::CACHE_BUDGET` reached).
    pub cache_clears: AtomicU64,
    /// The most masks one grammar's cache has held at once.
    pub cache_peak_entries: AtomicU64,
}

/// A compiled grammar on one vocabulary, with its mask cache. Built once
/// per (model, format, tool names) and shared by the requests that use it
/// (`Hermes::grammar`).
pub struct Grammar {
    masks: Masks,
    /// The same states' shortest way out (`pda::Graph::finisher`): what a
    /// stalled call takes when the reply has no end tokens (a library
    /// caller's `Constraint`; the server's end the reply instead:
    /// `Constraint::with_end`).
    finish: Masks,
    /// Where an engaged constraint starts.
    start: NodeId,
    /// Free: these tokens engage it.
    triggers: Vec<u32>,
    /// What the identity calls it: `<name> <hash>`.
    field: String,
    counters: Arc<Counters>,
    /// None: calls are never closed early (`AXON_CONSTRAIN_STALL=0`).
    stall: Option<Stall>,
}

/// The Hermes grammar's rules that aren't in its graph, for the identity's
/// hash (`hermes_field`): what the tokens of a constrained reply depend on
/// besides the request and the graph. A change to them is a new line here.
/// The triggers' rules (`Constraint::free`: the open as its token or
/// spelled, suspended in reasoning the reply's first token opens, armed
/// again after each call; after a stalled call only the reply's end) are
/// its `trigger` line: a change to `free` changes that line (Fable's review
/// of #274).
fn hermes_desc(open: u32, close: Option<u32>, stall: Option<Stall>) -> String {
    format!(
        "axon-constrain/5\nhermes\nopen={open}\nclose={close:?}\n\
         trigger=open-token-or-spelled,not-in-first-token-reasoning,re-armed-after-call\n\
         name=enum(offered)\narguments=object\nmax_depth={}\nspace=llama({})\nstall={}\n",
        pda::MAX_DEPTH,
        pda::INDENT_MAX,
        match stall {
            Some(st) => format!(
                "loop={}/{}:end-reply-there,call-unclosed",
                st.window, LOOP_PERIOD
            ),
            None => "off".into(),
        }
    )
}

/// The identity field of the Hermes grammars on markers (`open`, `close`)
/// with stall window `stall` (`Grammar::field`): a hash of `hermes_desc`
/// and of what the rules compile to (`pda::Graph::canon` of the graph and
/// of its finisher), built with one placeholder name: the names come from
/// the request, as its prompt does, so they aren't in it. An edit to
/// `pda::json`, the finisher's ties or the whitespace unrolling changes
/// the hash by itself (shady's review of #274).
fn hermes_field(open: u32, close: Option<u32>, stall: Option<Stall>) -> String {
    let (graph, _) = hermes_graph(close, &[b"x".to_vec()]).expect("one plain name builds");
    let mut canon = Vec::new();
    graph.canon(&mut canon);
    graph.finisher().canon(&mut canon);
    let mut h = Sha256::new();
    h.update(hermes_desc(open, close, stall).as_bytes());
    h.update(&canon);
    format!("hermes-json {}", &h.finish_hex()[..16])
}

/// The Hermes graph (`Grammar::hermes`) and its start node.
/// (`<tool_call>` is the trigger, outside it: a call starts after it.)
fn hermes_graph(close: Option<u32>, names: &[Vec<u8>]) -> Result<(pda::Graph, NodeId), String> {
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
    // Names a JSON string can't hold raw (an escape the trie would have to
    // tell apart) fail here: the request goes unconstrained.
    Ok((b.finish()?, h0))
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
    #[cfg_attr(not(test), allow(dead_code))] // the server's are `Hermes::grammar`'s
    pub fn hermes(vocab: Arc<Vocab>, names: &[Vec<u8>]) -> Result<Grammar, String> {
        Self::build(vocab, names, Arc::default(), Some(Stall::DEFAULT), None)
    }

    /// `hermes` with a test window (`Stall`), or None: never closed early.
    #[cfg(test)]
    pub fn hermes_stall(
        vocab: Arc<Vocab>,
        names: &[Vec<u8>],
        stall: Option<Stall>,
    ) -> Result<Grammar, String> {
        Self::build(vocab, names, Arc::default(), stall, None)
    }

    /// `hermes` counting into `counters`, with stall window `stall`, its
    /// identity field `field` when known (`Hermes::field`: the same for
    /// any names).
    fn build(
        vocab: Arc<Vocab>,
        names: &[Vec<u8>],
        counters: Arc<Counters>,
        stall: Option<Stall>,
        field: Option<&str>,
    ) -> Result<Grammar, String> {
        let (open, close) = hermes_markers(&vocab)?;
        if names.is_empty() {
            return Err("no tool names to hold a call to".into());
        }
        assert!(stall.is_none_or(|st| st.window > 0 && st.window <= u32::MAX as usize));
        let (graph, start) = hermes_graph(close, names)?;
        Ok(Grammar {
            finish: Masks::new(graph.finisher(), vocab.clone(), counters.clone()),
            masks: Masks::new(graph, vocab, counters.clone()),
            start,
            triggers: vec![open],
            field: match field {
                Some(f) => f.to_string(),
                None => hermes_field(open, close, stall),
            },
            counters,
            stall,
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
    stall: Option<Stall>,
    /// Most recently used first.
    grammars: Mutex<Vec<KeptGrammar>>,
    /// Why tools requests were left unconstrained, each logged once
    /// (`unheld`; at most `UNHELD_LOGGED` of them).
    unheld: Mutex<Vec<String>>,
}

/// A grammar `Hermes` keeps, under its names (sorted, each once).
type KeptGrammar = (Vec<Vec<u8>>, Arc<Grammar>);

impl Hermes {
    /// Grammars kept (each with its own mask cache).
    pub const KEEP: usize = 8;

    /// Reasons for unconstrained tools requests logged (`unheld`).
    const UNHELD_LOGGED: usize = 32;

    /// Err (why) as `Grammar::hermes`'s for this vocabulary. `stall`: the
    /// stall rule's window (`Stall::DEFAULT` but in tests), None to never
    /// close a call early (`AXON_CONSTRAIN_STALL=0`).
    pub fn new(
        vocab: Arc<Vocab>,
        counters: Arc<Counters>,
        stall: Option<Stall>,
    ) -> Result<Hermes, String> {
        let (open, close) = hermes_markers(&vocab)?;
        assert!(stall.is_none_or(|st| st.window > 0));
        Ok(Hermes {
            vocab,
            field: hermes_field(open, close, stall),
            counters,
            stall,
            grammars: Mutex::new(Vec::new()),
            unheld: Mutex::new(Vec::new()),
        })
    }

    /// A tools request left unconstrained, `why` (`grammar`'s or
    /// `Constraint::forced`'s Err): counted (`Counters::unheld`); true the
    /// first time `why` is seen (the server logs it then).
    pub fn unheld(&self, why: &str) -> bool {
        self.counters.unheld.fetch_add(1, Ordering::Relaxed);
        let mut v = self.unheld.lock().unwrap_or_else(|p| p.into_inner());
        if v.len() >= Self::UNHELD_LOGGED || v.iter().any(|w| w == why) {
            return false;
        }
        v.push(why.to_string());
        true
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
        let g = Grammar::build(
            self.vocab.clone(),
            &key,
            self.counters.clone(),
            self.stall,
            Some(&self.field),
        )?;
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
    On {
        node: NodeId,
        stack: Vec<NodeId>,
    },
    /// From the token a call stalled at: only the reply's end
    /// (`Constraint::with_end`). The call stays as written, cut off, and a
    /// model in a loop can't open the next call and stall again.
    Ended,
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
    /// Engaged: the call's last `LOOP_PERIOD` tokens (a ring, the next write
    /// at `at`), how many it has taken, for each period p (`runs[p - 1]`)
    /// how many tokens in a row equal the one p before, and whether it
    /// stalled (its states then take `Grammar::finish`'s edges).
    recent: Vec<u32>,
    at: usize,
    taken: usize,
    runs: [u32; LOOP_PERIOD],
    stalled: bool,
    /// The reply's stall, once one happened (`stall`).
    seen: Option<Stalled>,
    /// The tokens that end the reply (`with_end`): all a reply may take
    /// after a stalled call.
    end: Arc<[u32]>,
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
            seen: None,
            end: Arc::new([]),
            shadow: false,
        }
    }

    /// When a call stalls the reply may then take only `end` (its stop
    /// tokens: `serve`'s `Job::stop_ids`), so it ends right there, the call
    /// cut off unclosed where it stalled (`cut`: the tools parser returns
    /// it as a call cut mid-arguments, and the reply finishes `length`).
    /// Empty (the default): the call is closed the shortest way
    /// (`Grammar::finish`), and the reply goes on free, armed for another
    /// call.
    pub fn with_end(mut self, end: &[u32]) -> Constraint {
        self.end = end.into();
        self
    }

    /// The reply's stalled call, if one stalled (the server logs it).
    pub fn stall(&self) -> Option<Stalled> {
        self.seen
    }

    /// The stall rule ended the reply (a stall `with_end`): its call is cut
    /// off unclosed, as `max_tokens` cuts one, so the reply finishes as one
    /// the token limit cut (`serve::finish_reason`: `length`).
    pub fn cut(&self) -> bool {
        self.seen.is_some_and(|s| s.cut)
    }

    /// A copy to advance through a draft's guesses (`sample::
    /// generate_constrained`): the same state, its counts not counted.
    pub fn draft_copy(&self) -> Constraint {
        Constraint {
            shadow: true,
            ..self.clone()
        }
    }

    /// An engaged token taken: into the stall rule (`LOOP_WINDOW`'s note).
    /// When it trips: with end tokens (`with_end`) the reply ends here,
    /// else the next states take the shortest way out (`stalled`).
    fn note(&mut self, id: u32) {
        let Some(Stall { window }) = self.g.stall else {
            return;
        };
        if self.stalled {
            return;
        }
        let ring = LOOP_PERIOD;
        let mut period = None;
        for p in 1..=LOOP_PERIOD {
            let r = &mut self.runs[p - 1];
            // The token p back: `at - p` in the ring (`at` < ring and p <=
            // ring, so it wraps once at most; no division a period).
            let back = self.at + ring - p;
            let back = if back >= ring { back - ring } else { back };
            if self.taken >= p && self.recent[back] == id {
                *r += 1;
                // A cycle: the stretch fills the window, its block written
                // twice at least (with a server's window, 2048, and periods
                // of 256 at most, the first implies the second; a test's
                // window under 2 × LOOP_PERIOD needs it, or one chance match
                // at a long lag would do). The smallest period is the block.
                let run = *r as usize;
                if period.is_none() && run + p >= window && run >= p {
                    period = Some(p);
                }
            } else {
                *r = 0;
            }
        }
        if self.recent.len() < ring {
            self.recent.push(id);
        } else {
            self.recent[self.at] = id;
        }
        self.at += 1;
        if self.at == ring {
            self.at = 0;
        }
        self.taken += 1;
        let Some(period) = period else {
            return;
        };
        if self.end.is_empty() {
            self.stalled = true;
        } else {
            self.phase = Phase::Ended;
        }
        self.seen = Some(Stalled {
            after: self.taken,
            period,
            window,
            cut: !self.end.is_empty(),
        });
        if !self.shadow {
            self.g.counters.stalls.fetch_add(1, Ordering::Relaxed);
        }
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
            Phase::Free | Phase::Ended => Err(ConstraintError(format!(
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
        let (node, stack) = match &self.phase {
            Phase::On { node, stack } => (node, stack),
            Phase::Free => return Ok(None),
            Phase::Ended => {
                let n = self.g.masks.vocab.len();
                return Ok(Some(TokenMask::from_ids(n, &self.end)));
            }
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
        let (node, stack) = match &mut self.phase {
            Phase::On { node, stack } => (node, stack),
            Phase::Free => {
                self.free(id);
                return true;
            }
            Phase::Ended => return self.end.contains(&id),
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
            // Complete: armed again for the next call (a stalled one closed
            // the shortest way too: there are no end tokens to end on).
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

    /// Stalled without end tokens: the call is being closed the shortest
    /// way (`note`).
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
            Phase::Free | Phase::Ended => false,
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
    switch_value("AXON_CONSTRAIN", "constrained tool calls", v)
}

/// AXON_CONSTRAIN_STALL's value `v`, read as `env_value` reads
/// AXON_CONSTRAIN: "0" turns the stall rule off alone (a held call in a
/// loop runs to `max_tokens`, its name and JSON still held: Fable's review
/// of #274, for data the rule would cut), "1" or unset on.
pub fn stall_env_value(v: Option<&str>) -> Result<Option<bool>, String> {
    switch_value("AXON_CONSTRAIN_STALL", "cutting a stalled tool call", v)
}

/// Variable `var`'s value `v` as an on/off switch for `what`.
fn switch_value(var: &str, what: &str, v: Option<&str>) -> Result<Option<bool>, String> {
    match v.map(str::trim) {
        None | Some("") => Ok(None),
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        Some(_) => Err(format!(
            "{var}={:?}: expected 0 ({what} off) or 1 (on), or unset or empty for the default \
             (on)",
            v.unwrap_or_default()
        )),
    }
}

/// `env_value` of AXON_CONSTRAIN (a value that isn't UTF-8 is an error).
pub fn env() -> Result<Option<bool>, String> {
    switch_env("AXON_CONSTRAIN", env_value)
}

/// `stall_env_value` of AXON_CONSTRAIN_STALL.
pub fn stall_env() -> Result<Option<bool>, String> {
    switch_env("AXON_CONSTRAIN_STALL", stall_env_value)
}

fn switch_env(
    var: &str,
    read: fn(Option<&str>) -> Result<Option<bool>, String>,
) -> Result<Option<bool>, String> {
    match std::env::var(var) {
        Ok(v) => read(Some(&v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{var} isn't UTF-8: expected 0 or 1, or unset"))
        }
    }
}
