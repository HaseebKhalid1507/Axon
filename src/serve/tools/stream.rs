//! Tool calls streamed as they are generated (#260): `Parser`'s stream mode.
//!
//! In decide-whole mode (`Parser::new`, main's behaviour before #260) a call
//! goes out only once its block is decided, so a call that runs on (a
//! looping argument, no `max_tokens`) sends nothing for minutes, and a
//! client with an idle timeout hangs up (codex: 300 s). Stream mode sends a
//! call the way OpenAI and llama-server do: its id and name as soon as they
//! are known, then its arguments' text as it arrives.
//!
//! A call can't be taken back once its head is out, so stream mode commits
//! to a call at a fixed point and decides everything after it as a call:
//!
//! - **The commit point.** The block's text so far is a prefix of a JSON
//!   object (`CallScan`: exactly the grammar `parse_object_spans` accepts),
//!   a string `"name"` member has been read (the first one), and the first
//!   arguments member (`"arguments"`, or Llama's `"parameters"`) opens with
//!   `{`. From that `{` on the arguments stream as `Piece::Args`, byte for
//!   byte as written, and the call ends with `Piece::Call` (the whole call,
//!   the same id).
//! - **Before it** the block is held, as in decide-whole mode, but decided
//!   early: as soon as its text can't begin a JSON object any more, it is
//!   content, tags and all, and the rest of the block streams as content.
//!   An object that closes without the commit point (arguments that aren't
//!   an object, or come before the name) is decided whole, as before: a call
//!   if `call_of` takes it (sent as head, arguments and end at once), else
//!   content.
//! - **After it** the call is a call. Arguments that stop being JSON (an
//!   unescaped quote, a raw newline, a bad escape: #259's failures) keep
//!   streaming as written up to the block's end, less the outer object's
//!   `}` and the blanks around it. A reply cut off mid-call (the token limit,
//!   a stop string) ends the call with what was written. Either way its
//!   `arguments` aren't a JSON object, and it is counted
//!   (`axon_tool_calls_invalid_total`). OpenAI's API does the same: arguments
//!   are the model's text, and clients check them (a harness then tells the
//!   model the call failed, and it can try again, where a block returned as
//!   text ended codex's turn).
//! - **After the object** whitespace and `</tool_call>` end the block. Other
//!   text ends it without its tag: a `<tool_call>` there opens the next block
//!   (#259's "a valid object followed by a second `<tool_call>`"), anything
//!   else is content. Llama: `;` and another object is the next call.
//!
//! `</tool_call>` ends a block wherever it appears, inside a JSON string too
//! (as in decide-whole mode). The pieces don't depend on how the text is cut
//! into pushes: every decision is made at a byte, and text is only held while
//! it may still be `</tool_call>`, or the end of arguments that went bad.

use super::{CLOSE, Call, J, Kind, MAX_DEPTH, OPEN, Parser, Piece, call_id, call_of, parse_json};

// ------------------------------------------------------------------ the scan

/// JSON's literals.
const LITS: [&[u8]; 3] = [b"null", b"true", b"false"];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Num {
    /// After `-`.
    Minus,
    /// A leading `0` (nothing but `.`, `e` or the end may follow).
    Zero,
    Int,
    /// After `.`.
    Dot,
    Frac,
    /// After `e`.
    E,
    /// After `e+` / `e-`.
    ESign,
    Exp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum St {
    /// Before the object: whitespace, then `{`.
    Start,
    /// After `{`: a key or `}`.
    Key0,
    /// After `,` in an object: a key.
    Key,
    /// After a key: `:`.
    Colon,
    /// A value (after `:`, or after `,` in an array).
    Val,
    /// After `[`: a value or `]`.
    Val0,
    /// After a value in a container: `,` or the container's close.
    After,
    /// In a string (`key`: an object's key); `esc`: after `\`; `hex`:
    /// `\u` digits still to come.
    Str {
        key: bool,
        esc: bool,
        hex: u8,
    },
    /// In `null`, `true` or `false` (`LITS[.0]`), `.1` bytes in.
    Lit(u8, u8),
    Num(Num),
    /// The object closed: whitespace only.
    Done,
    Bad,
}

/// What a byte did (`CallScan::feed`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Ev {
    /// Still a prefix of a JSON object.
    More,
    /// No JSON object starts like this: `parse_object_spans` refuses any
    /// text that does, however it goes on.
    Bad,
    /// This `{` opens the arguments object of a call whose name is known
    /// (the commit point).
    ArgsOpen,
    /// This `}` closes that arguments object.
    ArgsClose,
    /// This `}` closes the object.
    End,
}

fn ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// A JSON object read a byte at a time, as `parse_object_spans` reads it
/// whole (the same grammar: `Reader`'s numbers, escapes, control bytes and
/// depth limit), with what a call needs from its outer members: the first
/// `"name"`, and where the first arguments member's object opens and closes.
/// O(1) a byte, outside the outer object's keys and its name.
#[derive(Clone, Debug)]
pub(super) struct CallScan {
    st: St,
    /// The open containers (`{` / `[`).
    stack: Vec<u8>,
    /// The outer object's current member's key, decoded.
    key: String,
    /// Raw text of the outer object's key or name string being read.
    cap: Vec<u8>,
    capturing: bool,
    /// The current member's value is the first `"name"`, a string.
    in_name: bool,
    /// The first `"name"`: unread, not a string (`Some(None)`), or its value.
    name: Option<Option<String>>,
    /// The first arguments member has begun.
    args_seen: bool,
    /// Its object is open (and `ArgsOpen` was returned).
    args_open: bool,
}

impl Default for CallScan {
    fn default() -> CallScan {
        CallScan {
            st: St::Start,
            stack: Vec::new(),
            key: String::new(),
            cap: Vec::new(),
            capturing: false,
            in_name: false,
            name: None,
            args_seen: false,
            args_open: false,
        }
    }
}

/// A JSON string's raw text (between its quotes), decoded as `Reader` does.
fn unquote(raw: &[u8]) -> Option<String> {
    let s = format!("\"{}\"", String::from_utf8_lossy(raw));
    match parse_json(&s)? {
        J::Str(s) => Some(s),
        _ => None,
    }
}

impl CallScan {
    /// The call's name, once its (first) `"name"` member has been read.
    pub(super) fn name(&self) -> Option<&str> {
        match &self.name {
            Some(Some(n)) => Some(n),
            _ => None,
        }
    }

    /// One more byte of the text.
    pub(super) fn feed(&mut self, b: u8) -> Ev {
        let ev = self.step(b);
        if ev == Ev::Bad {
            self.st = St::Bad;
        }
        ev
    }

    fn step(&mut self, b: u8) -> Ev {
        match self.st {
            St::Bad => Ev::Bad,
            St::Start if ws(b) => Ev::More,
            St::Start if b == b'{' => {
                self.stack.push(b'{');
                self.st = St::Key0;
                Ev::More
            }
            St::Done if ws(b) => Ev::More,
            St::Start | St::Done => Ev::Bad,
            St::Key0 | St::Key if ws(b) => Ev::More,
            St::Key0 | St::Key if b == b'"' => {
                self.open_str(true);
                Ev::More
            }
            St::Key0 if b == b'}' => self.close(),
            St::Key0 | St::Key => Ev::Bad,
            St::Colon if ws(b) => Ev::More,
            St::Colon if b == b':' => {
                self.st = St::Val;
                Ev::More
            }
            St::Colon => Ev::Bad,
            St::Val | St::Val0 if ws(b) => Ev::More,
            St::Val0 if b == b']' => self.close(),
            St::Val | St::Val0 => self.value(b),
            St::After if ws(b) => Ev::More,
            St::After => match (b, self.stack.last()) {
                (b',', Some(b'{')) => {
                    self.st = St::Key;
                    Ev::More
                }
                (b',', Some(b'[')) => {
                    self.st = St::Val;
                    Ev::More
                }
                (b'}', Some(b'{')) | (b']', Some(b'[')) => self.close(),
                _ => Ev::Bad,
            },
            St::Str { key, esc, hex } => self.string(b, key, esc, hex),
            St::Lit(w, at) => {
                let lit = LITS[w as usize];
                if b != lit[at as usize] {
                    return Ev::Bad;
                }
                if at as usize + 1 == lit.len() {
                    self.value_end()
                } else {
                    self.st = St::Lit(w, at + 1);
                    Ev::More
                }
            }
            St::Num(n) => self.number(b, n),
        }
    }

    fn open_str(&mut self, key: bool) {
        self.cap.clear();
        self.capturing = (key && self.stack.len() == 1) || self.in_name;
        self.st = St::Str {
            key,
            esc: false,
            hex: 0,
        };
    }

    /// A value starts with `b`, in the container on top of the stack.
    fn value(&mut self, b: u8) -> Ev {
        // `Reader::value` refuses any value (scalars too) this deep.
        if self.stack.len() > MAX_DEPTH {
            return Ev::Bad;
        }
        let mut ev = Ev::More;
        if self.stack.len() == 1 {
            // A member of the outer object (always an object: an array
            // there is never a member's value at depth 1).
            if self.key == "name" && self.name.is_none() {
                if b == b'"' {
                    self.in_name = true;
                } else {
                    self.name = Some(None);
                }
            }
            if (self.key == "arguments" || self.key == "parameters") && !self.args_seen {
                self.args_seen = true;
                if b == b'{' && self.name().is_some() {
                    self.args_open = true;
                    ev = Ev::ArgsOpen;
                }
            }
        }
        match b {
            b'{' => {
                self.stack.push(b'{');
                self.st = St::Key0;
            }
            b'[' => {
                self.stack.push(b'[');
                self.st = St::Val0;
            }
            b'"' => self.open_str(false),
            b'n' => self.st = St::Lit(0, 1),
            b't' => self.st = St::Lit(1, 1),
            b'f' => self.st = St::Lit(2, 1),
            b'-' => self.st = St::Num(Num::Minus),
            b'0' => self.st = St::Num(Num::Zero),
            b'1'..=b'9' => self.st = St::Num(Num::Int),
            _ => return Ev::Bad,
        }
        ev
    }

    /// A value ended inside the container on top of the stack.
    fn value_end(&mut self) -> Ev {
        self.st = St::After;
        Ev::More
    }

    /// `}` or `]` closes the container on top of the stack.
    fn close(&mut self) -> Ev {
        self.stack.pop();
        if self.stack.is_empty() {
            self.st = St::Done;
            return Ev::End;
        }
        if self.stack.len() == 1 && self.args_open {
            self.args_open = false;
            self.st = St::After;
            return Ev::ArgsClose;
        }
        self.value_end()
    }

    fn string(&mut self, b: u8, key: bool, esc: bool, hex: u8) -> Ev {
        let st = |esc, hex| St::Str { key, esc, hex };
        if hex > 0 {
            if !b.is_ascii_hexdigit() {
                return Ev::Bad;
            }
            self.st = st(false, hex - 1);
        } else if esc {
            match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => self.st = st(false, 0),
                b'u' => self.st = st(false, 4),
                _ => return Ev::Bad,
            }
        } else {
            match b {
                b'"' => return self.str_end(key),
                b'\\' => self.st = st(true, 0),
                0..0x20 => return Ev::Bad,
                _ => {}
            }
        }
        if self.capturing {
            self.cap.push(b);
        }
        Ev::More
    }

    fn str_end(&mut self, key: bool) -> Ev {
        let raw = std::mem::take(&mut self.cap);
        let captured = std::mem::replace(&mut self.capturing, false);
        if key {
            if captured {
                self.key = unquote(&raw).unwrap_or_default();
            }
            self.st = St::Colon;
            return Ev::More;
        }
        if std::mem::replace(&mut self.in_name, false) {
            self.name = Some(unquote(&raw));
        }
        self.value_end()
    }

    fn number(&mut self, b: u8, n: Num) -> Ev {
        let d = b.is_ascii_digit();
        let next = match (n, b) {
            (Num::Minus, b'0') => Some(Num::Zero),
            (Num::Minus, b'1'..=b'9') => Some(Num::Int),
            (Num::Int, _) if d => Some(Num::Int),
            (Num::Zero | Num::Int, b'.') => Some(Num::Dot),
            (Num::Dot | Num::Frac, _) if d => Some(Num::Frac),
            (Num::Zero | Num::Int | Num::Frac, b'e' | b'E') => Some(Num::E),
            (Num::E, b'+' | b'-') => Some(Num::ESign),
            (Num::E | Num::ESign | Num::Exp, _) if d => Some(Num::Exp),
            _ => None,
        };
        if let Some(n) = next {
            self.st = St::Num(n);
            return Ev::More;
        }
        match n {
            // The number ended before `b`, which its container reads.
            Num::Zero | Num::Int | Num::Frac | Num::Exp => {
                self.value_end();
                self.step(b)
            }
            _ => Ev::Bad,
        }
    }
}

// ------------------------------------------------------------------ blocks

/// Where a block stands (stream mode).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum BSt {
    /// Not a call yet: its text is held while it may still begin one.
    Head,
    /// A call (its head is out): its arguments object, JSON so far, streams.
    Args,
    /// A call whose arguments stopped being JSON: the rest of the block is
    /// its arguments, as written (less `raw_tail`).
    Raw,
    /// The arguments closed: the rest of the object, dropped.
    Tail,
    /// The object closed: whitespace, then the block's end.
    After,
    /// Llama, after `;` past a call: whitespace, then the next call's `{`.
    Sep,
    /// The rest of the block is content. `call`: the block was a call (its
    /// end tag is dropped); else the block is content, tags and all.
    Text { call: bool },
}

/// The open block (Hermes: from after `<tool_call>`; Llama: the reply's
/// object, from its `{`).
pub(super) struct Block {
    pub(super) st: BSt,
    scan: CallScan,
    /// What comes before the held text if the block turns out content:
    /// `<tool_call>`; Llama: the reply's text before its `{` (or the `;`
    /// before a later object).
    pre: String,
    /// Head (and Sep): the text so far.
    held: String,
    /// The call being streamed.
    call: Option<Call>,
    /// Raw: the arguments' end held back.
    tail: String,
}

impl Block {
    pub(super) fn new(kind: Kind, pre: String) -> Block {
        Block {
            st: BSt::Head,
            scan: CallScan::default(),
            pre: if kind == Kind::Hermes {
                OPEN.to_string()
            } else {
                pre
            },
            held: String::new(),
            call: None,
            tail: String::new(),
        }
    }
}

/// How much of the end of `s` (the arguments' text since they went bad) may
/// be the end of the block's object rather than arguments: blanks, at most
/// one `}`, blanks.
fn raw_tail(s: &str) -> usize {
    let t = s.trim_end();
    let t = t.strip_suffix('}').map_or(t, str::trim_end);
    s.len() - t.len()
}

/// The longest end of `s` that may still become `tag`.
fn tag_start(s: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&n| s.ends_with(&tag[..n]))
        .unwrap_or(0)
}

impl Parser {
    /// Stream mode, Hermes: text outside blocks as in decide-whole mode;
    /// each block through `feed_block`.
    pub(super) fn hermes_stream(&mut self, out: &mut Vec<Piece>) {
        loop {
            if !self.inside {
                if let Some(at) = self.buf.find(OPEN) {
                    let text = self.buf[..at].to_string();
                    self.buf.drain(..at + OPEN.len());
                    self.content(&text, out);
                    self.inside = true;
                    self.blk = None;
                    continue;
                }
                let cut = self.buf.len() - tag_start(&self.buf, OPEN);
                let text: String = self.buf.drain(..cut).collect();
                if !text.is_empty() {
                    self.content(&text, out);
                }
                return;
            }
            // The end tag ends the block wherever it is; text that may be
            // its start waits.
            let close = self.buf.find(CLOSE);
            let upto = close.unwrap_or(self.buf.len() - tag_start(&self.buf, CLOSE));
            let text: String = self.buf.drain(..upto).collect();
            if let Some(i) = self.feed_block(&text, out) {
                // Other text after the object: the block is over (without
                // its tag), and that text is outside it.
                self.buf.insert_str(0, &text[i..]);
                self.inside = false;
                self.blk = None;
                continue;
            }
            if close.is_none() {
                return;
            }
            self.buf.drain(..CLOSE.len());
            self.end_block(true, out);
            self.inside = false;
        }
    }

    /// Stream mode, the end of the reply inside a Hermes block or a Llama
    /// object: the held text, then the block's end.
    pub(super) fn finish_stream_block(&mut self, out: &mut Vec<Piece>) {
        let t = std::mem::take(&mut self.buf);
        match self.feed_block(&t, out) {
            Some(i) => {
                self.inside = false;
                self.blk = None;
                self.content(&t[i..], out);
            }
            None => {
                self.end_block(false, out);
                self.inside = false;
            }
        }
    }

    /// More of the open block's text (none of it the end tag): Some(i) if
    /// the block ended before `t[i..]` (Hermes, after its object).
    pub(super) fn feed_block(&mut self, t: &str, out: &mut Vec<Piece>) -> Option<usize> {
        let kind = self.kind;
        let mut b = self
            .blk
            .take()
            .unwrap_or_else(|| Block::new(kind, String::new()));
        let r = self.feed_in(&mut b, t, out);
        self.blk = Some(b);
        r
    }

    fn feed_in(&mut self, b: &mut Block, t: &str, out: &mut Vec<Piece>) -> Option<usize> {
        let bytes = t.as_bytes();
        // `t[mark..]` is the text not yet given to the state's sink.
        let mut mark = 0;
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i];
            match b.st {
                BSt::Head => match b.scan.feed(c) {
                    Ev::More => {}
                    Ev::ArgsOpen => {
                        b.held.push_str(&t[mark..i]);
                        mark = i;
                        self.head(b, out);
                    }
                    Ev::End => {
                        b.held.push_str(&t[mark..=i]);
                        mark = i + 1;
                        match call_of(&b.held) {
                            Some(call) => {
                                b.held.clear();
                                self.whole_call(call, out);
                                b.st = BSt::After;
                            }
                            None => self.not_call(b, out),
                        }
                    }
                    Ev::Bad | Ev::ArgsClose => {
                        b.held.push_str(&t[mark..i]);
                        mark = i;
                        self.not_call(b, out);
                    }
                },
                BSt::Args => match b.scan.feed(c) {
                    Ev::ArgsClose => {
                        self.args(b, &t[mark..=i], out);
                        mark = i + 1;
                        b.st = BSt::Tail;
                    }
                    Ev::Bad => {
                        self.args(b, &t[mark..i], out);
                        mark = i;
                        b.st = BSt::Raw;
                    }
                    _ => {}
                },
                BSt::Tail => match b.scan.feed(c) {
                    Ev::End => {
                        mark = i + 1;
                        self.call_end(b, out);
                        b.st = BSt::After;
                    }
                    Ev::Bad => {
                        mark = i;
                        self.call_end(b, out);
                        b.st = BSt::Text { call: true };
                    }
                    _ => mark = i + 1,
                },
                BSt::After if ws(c) => mark = i + 1,
                BSt::After if self.kind == Kind::Hermes => return Some(i),
                BSt::After => {
                    mark = i;
                    b.held.clear();
                    b.st = if c == b';' {
                        BSt::Sep
                    } else {
                        BSt::Text { call: true }
                    };
                }
                BSt::Sep if ws(c) => {}
                BSt::Sep if c == b'{' => {
                    // The next call: content from its `;` if it isn't one.
                    b.held.push_str(&t[mark..i]);
                    mark = i;
                    let pre = std::mem::take(&mut b.held);
                    *b = Block::new(self.kind, pre);
                    continue;
                }
                BSt::Sep => {
                    b.held.push_str(&t[mark..i]);
                    mark = i;
                    let sep = std::mem::take(&mut b.held);
                    self.text_raw(sep, out);
                    b.st = BSt::Text { call: true };
                }
                BSt::Raw | BSt::Text { .. } => break,
            }
            i += 1;
        }
        let rest = &t[mark..];
        match b.st {
            BSt::Head | BSt::Sep => b.held.push_str(rest),
            BSt::Args => self.args(b, rest, out),
            BSt::Raw => {
                b.tail.push_str(rest);
                let keep = raw_tail(&b.tail);
                let send: String = b.tail.drain(..b.tail.len() - keep).collect();
                self.args(b, &send, out);
            }
            BSt::Text { .. } => self.text_raw(rest.to_string(), out),
            BSt::Tail | BSt::After => {}
        }
        None
    }

    /// The block's end: its end tag (`closed`) or the reply's.
    fn end_block(&mut self, closed: bool, out: &mut Vec<Piece>) {
        let kind = self.kind;
        let mut b = self
            .blk
            .take()
            .unwrap_or_else(|| Block::new(kind, String::new()));
        match b.st {
            BSt::Head => {
                // Never a whole object: content, as in decide-whole mode.
                self.not_call(&mut b, out);
                if closed {
                    self.text_raw(CLOSE.to_string(), out);
                }
            }
            // A call cut off (or its arguments gone bad): what was written.
            BSt::Args | BSt::Raw | BSt::Tail => self.call_end(&mut b, out),
            BSt::Text { call: false } if closed => self.text_raw(CLOSE.to_string(), out),
            BSt::Text { .. } | BSt::After | BSt::Sep => {}
        }
    }

    /// The commit point: the call's head goes out.
    fn head(&mut self, b: &mut Block, out: &mut Vec<Piece>) {
        let c = Call {
            id: call_id(),
            name: b.scan.name().unwrap_or_default().to_string(),
            args: String::new(),
        };
        self.ws.clear();
        out.push(Piece::Head(c.clone()));
        b.call = Some(c);
        b.held.clear();
        b.st = BSt::Args;
    }

    fn args(&mut self, b: &mut Block, s: &str, out: &mut Vec<Piece>) {
        if s.is_empty() {
            return;
        }
        if let Some(c) = b.call.as_mut() {
            c.args.push_str(s);
        }
        out.push(Piece::Args(s.to_string()));
    }

    fn call_end(&mut self, b: &mut Block, out: &mut Vec<Piece>) {
        b.tail.clear();
        if let Some(c) = b.call.take() {
            self.call(c, out);
        }
    }

    /// A call decided whole (its object closed before the commit point):
    /// head, arguments and end at once, as a streamed call's pieces.
    pub(super) fn whole_call(&mut self, c: Call, out: &mut Vec<Piece>) {
        self.ws.clear();
        out.push(Piece::Head(Call {
            args: String::new(),
            ..c.clone()
        }));
        if !c.args.is_empty() {
            out.push(Piece::Args(c.args.clone()));
        }
        self.call(c, out);
    }

    /// Not a call: the held text is content, tags and all, and so is the rest
    /// of the block.
    fn not_call(&mut self, b: &mut Block, out: &mut Vec<Piece>) {
        let ws = std::mem::take(&mut self.ws);
        self.after_call = false;
        let held = std::mem::take(&mut b.held);
        self.dropped(&held);
        out.push(Piece::Text(ws + &b.pre + &held));
        b.st = BSt::Text { call: false };
    }

    /// Content as written (inside a block: no blank-line handling).
    fn text_raw(&mut self, s: String, out: &mut Vec<Piece>) {
        if !s.is_empty() {
            self.after_call = false;
            out.push(Piece::Text(s));
        }
    }
}
