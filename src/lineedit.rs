//! A small line editor for `axon chat` on a terminal.
//!
//! Three layers:
//! - [`Editor`]: the buffer, cursor, history and kill ring as a pure state
//!   machine of [`Op`]s, no terminal involved (unit-tested below).
//! - [`layout`]: where the prompt and buffer land on screen for a given
//!   width, wide characters and wrapping included; also pure.
//! - [`LineReader`], [`RawMode`], [`poll_typeahead`]: the crossterm side,
//!   reading keys and redrawing.
//!
//! The terminal stays in raw mode for the whole session, replies included,
//! so keys typed while a reply streams are not echoed into it: the reply
//! loop collects them with [`poll_typeahead`] and the next prompt replays
//! them. Raw mode turns off output post-processing too, so everything
//! written to the terminal in the meantime must say "\r\n", not "\n".

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::io::{self, Write};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crossterm::ExecutableCommand;
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::terminal;
use unicode_width::UnicodeWidthChar;

/// The prompt of a message's first line, and of each line after it.
const PROMPT: &str = "> ";
const MORE: &str = ". ";
const BOLD: &str = "\x1b[1m";
const PLAIN: &str = "\x1b[0m";

/// Spaces a Tab (typed or pasted) becomes: a tab's width on screen depends
/// on its column, which would make cursor math depend on the terminal's stops.
const TAB: &str = "    ";

/// Columns `c` takes on a terminal: 2 for wide (CJK, most emoji), 0 for
/// combining marks and other zero-width characters, 1 otherwise.
pub fn char_width(c: char) -> usize {
    c.width().unwrap_or(0)
}

/// Zero-width characters ride along with the character before them: the
/// cursor never stops between a letter and its accent.
fn rides(c: char) -> bool {
    c != '\n' && char_width(c) == 0
}

/// A word for Alt-B/F, Ctrl-Left/Right, Alt-D and Alt-Backspace.
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// A word for Ctrl-W: anything up to whitespace, like the shell's.
fn is_unix_word(c: char) -> bool {
    !c.is_whitespace()
}

/// An editing operation; [`op_for`] maps keys to them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// Type a character (control characters are ignored).
    Insert(char),
    /// Insert text as is but tidied: line breaks become "\n", tabs spaces,
    /// other control characters go. Never submits, whatever it holds.
    Paste(String),
    Left,
    Right,
    WordLeft,
    WordRight,
    /// Start / end of the current line.
    Home,
    End,
    Backspace,
    Delete,
    /// Ctrl-D: end of input on an empty buffer, else Delete.
    DeleteOrEof,
    /// Ctrl-W: delete back to whitespace.
    KillUnixWordBack,
    /// Alt-Backspace: delete back to the start of a word.
    KillWordBack,
    /// Alt-D: delete to the end of a word.
    KillWordForward,
    /// Ctrl-U / Ctrl-K: delete to the start / end of the line.
    KillToStart,
    KillToEnd,
    /// Ctrl-Y: insert the text the last kill deleted.
    Yank,
    /// Previous / next line of the buffer, else history entry.
    Up,
    Down,
    /// Submit, or go on to a new line if the buffer ends in a backslash.
    Enter,
    /// Alt-Enter: a line break in the message.
    Newline,
    /// Ctrl-C: clear the buffer. (On an empty one [`LineReader`] shows a
    /// hint instead; it never ends input.)
    Interrupt,
    /// Ctrl-L.
    ClearScreen,
}

/// What an [`Op`] asks of the caller.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The buffer or cursor may have changed: redraw.
    Redraw,
    /// Clear the screen, then redraw.
    ClearScreen,
    /// The message is done.
    Submit(String),
    /// No more input.
    Eof,
}

/// The key bindings.
pub fn op_for(k: &KeyEvent) -> Option<Op> {
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    let alt = k.modifiers.contains(KeyModifiers::ALT);
    Some(match k.code {
        KeyCode::Char(c) if ctrl => match c.to_ascii_lowercase() {
            'a' => Op::Home,
            'e' => Op::End,
            'b' => Op::Left,
            'f' => Op::Right,
            'h' => Op::Backspace,
            'd' => Op::DeleteOrEof,
            'w' => Op::KillUnixWordBack,
            'u' => Op::KillToStart,
            'k' => Op::KillToEnd,
            'y' => Op::Yank,
            'p' => Op::Up,
            'n' => Op::Down,
            'j' | 'm' => Op::Enter,
            'c' => Op::Interrupt,
            'l' => Op::ClearScreen,
            _ => return None,
        },
        KeyCode::Char(c) if alt => match c {
            'b' => Op::WordLeft,
            'f' => Op::WordRight,
            'd' => Op::KillWordForward,
            _ => return None,
        },
        KeyCode::Char(c) => Op::Insert(c),
        KeyCode::Tab => Op::Paste("\t".into()),
        KeyCode::Left if ctrl || alt => Op::WordLeft,
        KeyCode::Right if ctrl || alt => Op::WordRight,
        KeyCode::Left => Op::Left,
        KeyCode::Right => Op::Right,
        KeyCode::Home => Op::Home,
        KeyCode::End => Op::End,
        KeyCode::Backspace if ctrl || alt => Op::KillWordBack,
        KeyCode::Backspace => Op::Backspace,
        KeyCode::Delete => Op::Delete,
        KeyCode::Up => Op::Up,
        KeyCode::Down => Op::Down,
        KeyCode::Enter if alt => Op::Newline,
        KeyCode::Enter => Op::Enter,
        _ => return None,
    })
}

/// Pasted text made fit for the buffer: "\r\n" and "\r" become "\n", tabs
/// spaces, other control characters are dropped.
fn tidy(s: &str) -> String {
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    let mut t = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => t.push('\n'),
            '\t' => t.push_str(TAB),
            c if c.is_control() => {}
            c => t.push(c),
        }
    }
    t
}

/// The editing state: the buffer (lines separated by "\n"), the cursor (a
/// byte offset at a character boundary), the session's history and the
/// last killed text.
#[derive(Default)]
pub struct Editor {
    buf: String,
    pos: usize,
    /// Submitted messages, oldest first; consecutive repeats kept once.
    history: Vec<String>,
    /// The history entry the buffer holds a copy of; history.len() = the
    /// new message, whose text waits in `draft` while history is browsed.
    /// Edits change only the copy, never the entry.
    at: usize,
    draft: String,
    kill: String,
}

impl Editor {
    pub fn text(&self) -> &str {
        &self.buf
    }

    pub fn cursor(&self) -> usize {
        self.pos
    }

    /// Start of the line `p` is on.
    fn line_start(&self, p: usize) -> usize {
        self.buf[..p].rfind('\n').map_or(0, |i| i + 1)
    }

    /// End of the line `p` is on (its "\n", or the buffer's end).
    fn line_end(&self, p: usize) -> usize {
        self.buf[p..].find('\n').map_or(self.buf.len(), |i| p + i)
    }

    /// The cursor position one character (with what rides on it) after `p`.
    /// Nothing rides on a line break: a zero-width character starting a
    /// line is a cursor stop of its own.
    fn next(&self, p: usize) -> usize {
        let mut it = self.buf[p..].char_indices();
        let Some((_, c)) = it.next() else { return p };
        let mut end = p + c.len_utf8();
        if c == '\n' {
            return end;
        }
        for (i, c) in it {
            if !rides(c) {
                break;
            }
            end = p + i + c.len_utf8();
        }
        end
    }

    /// The cursor position one character before `p`: the inverse of `next`.
    fn prev(&self, p: usize) -> usize {
        let mut i = p;
        let mut it = self.buf[..p].chars().rev().peekable();
        while let Some(c) = it.next() {
            i -= c.len_utf8();
            if !rides(c) || matches!(it.peek(), None | Some('\n')) {
                break;
            }
        }
        i
    }

    /// Start of the word before `p`: past non-words, then words.
    fn word_back(&self, p: usize, word: fn(char) -> bool) -> usize {
        let s = &self.buf[..p];
        let mut it = s.char_indices().rev().peekable();
        let mut i = p;
        while let Some(&(j, c)) = it.peek()
            && !word(c)
        {
            i = j;
            it.next();
        }
        while let Some(&(j, c)) = it.peek()
            && word(c)
        {
            i = j;
            it.next();
        }
        i
    }

    /// End of the word after `p`.
    fn word_forward(&self, p: usize, word: fn(char) -> bool) -> usize {
        let mut it = self.buf[p..].char_indices().peekable();
        let mut i = p;
        while let Some(&(j, c)) = it.peek()
            && !word(c)
        {
            i = p + j + c.len_utf8();
            it.next();
        }
        while let Some(&(j, c)) = it.peek()
            && word(c)
        {
            i = p + j + c.len_utf8();
            it.next();
        }
        i
    }

    /// Delete a..b into the kill buffer; the cursor goes to a.
    fn kill(&mut self, a: usize, b: usize) {
        if a < b {
            self.kill = self.buf[a..b].to_string();
            self.buf.replace_range(a..b, "");
        }
        self.pos = a;
    }

    fn insert(&mut self, s: &str) {
        self.buf.insert_str(self.pos, s);
        self.pos += s.len();
    }

    /// Show a copy of history entry `at` (or the draft), cursor at the end.
    fn recall(&mut self, at: usize) {
        if self.at == self.history.len() {
            self.draft = std::mem::take(&mut self.buf);
        }
        self.at = at;
        self.buf = match self.history.get(at) {
            Some(h) => h.clone(),
            None => std::mem::take(&mut self.draft),
        };
        self.pos = self.buf.len();
    }

    /// Move to the same column (in cursor stops: a letter and its accents
    /// are one) of the line starting at `start`, or to that line's end.
    fn move_to_line_at(&mut self, start: usize) {
        let mut col = 0;
        let mut q = self.line_start(self.pos);
        while q < self.pos {
            q = self.next(q);
            col += 1;
        }
        let end = self.line_end(start);
        let mut q = start;
        for _ in 0..col {
            if q == end {
                break;
            }
            q = self.next(q);
        }
        self.pos = q;
    }

    /// Forget the buffer: a new, empty message.
    fn clear(&mut self) {
        self.buf.clear();
        self.pos = 0;
        self.at = self.history.len();
        self.draft.clear();
    }

    pub fn apply(&mut self, op: Op) -> Outcome {
        let p = self.pos;
        match op {
            Op::Insert(c) if !c.is_control() => self.insert(c.encode_utf8(&mut [0; 4])),
            Op::Insert(_) => {}
            Op::Paste(s) => self.insert(&tidy(&s)),
            Op::Newline => self.insert("\n"),
            Op::Left => self.pos = self.prev(p),
            Op::Right => self.pos = self.next(p),
            Op::WordLeft => self.pos = self.word_back(p, is_word),
            Op::WordRight => self.pos = self.word_forward(p, is_word),
            Op::Home => self.pos = self.line_start(p),
            Op::End => self.pos = self.line_end(p),
            Op::Backspace => {
                let a = self.prev(p);
                self.buf.replace_range(a..p, "");
                self.pos = a;
            }
            Op::DeleteOrEof if self.buf.is_empty() => return Outcome::Eof,
            Op::Delete | Op::DeleteOrEof => {
                let b = self.next(p);
                self.buf.replace_range(p..b, "");
            }
            Op::KillUnixWordBack => self.kill(self.word_back(p, is_unix_word), p),
            Op::KillWordBack => self.kill(self.word_back(p, is_word), p),
            Op::KillWordForward => self.kill(p, self.word_forward(p, is_word)),
            // At the start (end) of a line these take its line break, as in Emacs.
            Op::KillToStart => match self.line_start(p) {
                a if a == p => self.kill(p.saturating_sub(1), p),
                a => self.kill(a, p),
            },
            Op::KillToEnd => match self.line_end(p) {
                b if b == p => self.kill(p, self.next(p)),
                b => self.kill(p, b),
            },
            Op::Yank => {
                let k = self.kill.clone();
                self.insert(&k);
            }
            Op::Up => match self.line_start(p) {
                0 if self.at > 0 => self.recall(self.at - 1),
                0 => {}
                s => self.move_to_line_at(self.line_start(s - 1)),
            },
            Op::Down => match self.line_end(p) {
                e if e < self.buf.len() => self.move_to_line_at(e + 1),
                _ if self.at < self.history.len() => self.recall(self.at + 1),
                _ => {}
            },
            Op::Enter if self.buf.ends_with('\\') => {
                self.buf.pop();
                self.buf.push('\n');
                self.pos = self.buf.len();
            }
            Op::Enter => {
                let msg = std::mem::take(&mut self.buf);
                if !msg.trim().is_empty() && self.history.last() != Some(&msg) {
                    self.history.push(msg.clone());
                }
                self.clear();
                return Outcome::Submit(msg);
            }
            Op::Interrupt => self.clear(),
            Op::ClearScreen => return Outcome::ClearScreen,
        }
        Outcome::Redraw
    }
}

/// The prompt and buffer as drawn on a terminal `width` columns wide.
#[derive(Debug, PartialEq, Eq)]
pub struct Layout {
    /// What to write, starting at column 0: "\r\n" between rows, written
    /// out rather than left to the terminal's auto-wrap, which differs
    /// between terminals for a wide character at the last column and for
    /// text exactly filling a row.
    pub text: String,
    /// (row, column) of the cursor and of the end of `text`, rows counted
    /// from the prompt's.
    pub cursor: (usize, usize),
    pub end: (usize, usize),
}

struct Pen {
    text: String,
    row: usize,
    col: usize,
    width: usize,
}

impl Pen {
    fn break_row(&mut self) {
        self.text.push_str("\r\n");
        self.row += 1;
        self.col = 0;
    }

    /// Make room for a character `w` wide: a new row if it doesn't fit.
    fn fit(&mut self, w: usize) {
        if w > 0 && self.col > 0 && self.col + w > self.width {
            self.break_row();
        }
    }

    fn put(&mut self, c: char, w: usize) {
        self.text.push(c);
        self.col += w;
    }

    /// After a full row the cursor must stand on the next one, which the
    /// terminal doesn't do until something is printed there: go there now.
    fn settle(&mut self) {
        if self.col >= self.width {
            self.break_row();
        }
    }
}

/// Lay out `buf` with the cursor at byte `pos` on a terminal `width` wide.
pub fn layout(buf: &str, pos: usize, width: usize) -> Layout {
    let mut p = Pen {
        text: String::with_capacity(buf.len() + 16),
        row: 0,
        col: 0,
        width: width.max(1),
    };
    let mut cursor = (0, 0);
    let mut start = 0;
    for (n, line) in buf.split('\n').enumerate() {
        if n > 0 {
            p.break_row();
        }
        p.text.push_str(BOLD);
        for c in if n == 0 { PROMPT } else { MORE }.chars() {
            p.fit(1);
            p.put(c, 1);
        }
        p.text.push_str(PLAIN);
        for (i, c) in line.char_indices() {
            let w = char_width(c);
            p.fit(w);
            if start + i == pos {
                cursor = (p.row, p.col);
            }
            p.put(c, w);
        }
        p.settle();
        if start + line.len() == pos {
            cursor = (p.row, p.col);
        }
        start += line.len() + 1;
    }
    Layout {
        text: p.text,
        cursor,
        end: (p.row, p.col),
    }
}

/// The terminal's width in columns (80 if it won't say).
fn term_width() -> usize {
    match terminal::size() {
        Ok((w, _)) if w > 0 => w as usize,
        _ => 80,
    }
}

/// Whether the terminal state was changed and needs restoring.
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Bracketed paste off, raw mode off (the termios saved when it went on).
/// Idempotent, and safe to call from the panic hook.
fn restore() {
    if ACTIVE.swap(false, Ordering::SeqCst) {
        let _ = io::stdout().execute(DisableBracketedPaste);
        let _ = terminal::disable_raw_mode();
    }
}

/// Raw mode and bracketed paste while alive. The terminal is put back when
/// this drops (normal return, `?` error return, unwinding) and, before the
/// panic message prints, by a panic hook, so it is never left broken.
pub struct RawMode(());

impl RawMode {
    pub fn enter() -> io::Result<RawMode> {
        static HOOK: Once = Once::new();
        HOOK.call_once(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                restore();
                prev(info);
            }));
        });
        terminal::enable_raw_mode()?;
        ACTIVE.store(true, Ordering::SeqCst);
        let guard = RawMode(());
        io::stdout().execute(EnableBracketedPaste)?;
        Ok(guard)
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        restore();
    }
}

/// File one event read while no prompt was showing: true for a Ctrl-C (which
/// is not queued: it was aimed at the reply), keys and pastes go into `q`
/// for the next prompt, anything else (resizes, focus, mouse) is dropped.
fn queue_event(q: &mut VecDeque<Event>, ev: Event) -> bool {
    match ev {
        Event::Key(k) if op_for(&k) == Some(Op::Interrupt) => return true,
        Event::Key(_) | Event::Paste(_) => q.push_back(ev),
        _ => {}
    }
    false
}

/// Collect the input waiting on the terminal, without blocking, into `q`;
/// true if it held a Ctrl-C (see [`queue_event`]). For use while a reply
/// streams in raw mode: one `poll` syscall when nothing was typed.
pub fn poll_typeahead(q: &mut VecDeque<Event>) -> bool {
    let mut hit = false;
    while event::poll(Duration::ZERO).unwrap_or(false) {
        match event::read() {
            Ok(ev) => hit |= queue_event(q, ev),
            Err(_) => break,
        }
    }
    hit
}

/// How long after a prompt appears a Ctrl-C on it still counts as aimed at
/// the reply before it (mashed, or pressed as the reply ended): dropped.
const LATE_CTRL_C: Duration = Duration::from_millis(300);

/// Shown under `^C` when Ctrl-C hits an empty prompt.
const QUIT_HINT: &str = "\x1b[2m(Ctrl-D or /exit to quit)\x1b[0m";

/// What [`LineReader::feed`] made of an event.
#[derive(Debug, PartialEq, Eq)]
enum Step {
    /// Keep reading.
    More,
    /// The message, or None at end of input.
    Done(Option<String>),
}

/// The interactive prompt: an [`Editor`] and where its last drawing is.
#[derive(Default)]
pub struct LineReader {
    pub ed: Editor,
    /// Row of the cursor, and of the drawing's end, below the prompt's row.
    row: usize,
    end_row: usize,
    /// Terminal width the next drawing is for.
    width: usize,
    /// The screen doesn't show the editor's state yet.
    dirty: bool,
    /// Until then a Ctrl-C on an empty line is dropped (LATE_CTRL_C).
    grace_until: Option<Instant>,
}

impl LineReader {
    /// Redraw the prompt and `buf` (cursor at `pos`) over the last drawing.
    fn draw(&mut self, out: &mut impl Write, buf: &str, pos: usize) -> io::Result<()> {
        let l = layout(buf, pos, self.width);
        let mut s = String::with_capacity(l.text.len() + 24);
        s.push('\r');
        if self.row > 0 {
            let _ = write!(s, "\x1b[{}A", self.row);
        }
        s.push_str("\x1b[J");
        s.push_str(&l.text);
        let ((cr, cc), (er, _)) = (l.cursor, l.end);
        if er > cr {
            let _ = write!(s, "\x1b[{}A", er - cr);
        }
        s.push('\r');
        if cc > 0 {
            let _ = write!(s, "\x1b[{cc}C");
        }
        (self.row, self.end_row) = (cr, er);
        out.write_all(s.as_bytes())?;
        out.flush()
    }

    /// Draw the editor's state if the screen is behind it.
    fn redraw(&mut self, out: &mut impl Write) -> io::Result<()> {
        if self.dirty {
            let (buf, pos) = (self.ed.text().to_string(), self.ed.cursor());
            self.draw(out, &buf, pos)?;
            self.dirty = false;
        }
        Ok(())
    }

    /// Leave the drawing: the cursor to the start of the row below it.
    fn leave(&mut self, out: &mut impl Write) -> io::Result<()> {
        if self.end_row > self.row {
            write!(out, "\x1b[{}B", self.end_row - self.row)?;
        }
        (self.row, self.end_row) = (0, 0);
        write!(out, "\r\n")?;
        out.flush()
    }

    /// Apply one event. `queued`: it was typed while no prompt was showing
    /// (typeahead), so end-of-input from it is dropped, not acted on: a
    /// Ctrl-D pressed during a reply must not end the session unseen.
    /// Drawing is left to [`redraw`](Self::redraw), except for the final
    /// drawing of a submitted message and the Ctrl-C notice.
    fn feed(&mut self, out: &mut impl Write, ev: Event, queued: bool) -> io::Result<Step> {
        let op = match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => op_for(&k),
            Event::Paste(s) => Some(Op::Paste(s)),
            Event::Resize(w, _) => {
                self.width = usize::from(w).max(1);
                self.dirty = true;
                None
            }
            _ => None,
        };
        // Ctrl-C on an empty line: a notice and a fresh prompt, like Python's.
        if op == Some(Op::Interrupt) && self.ed.text().is_empty() {
            if self.grace_until.is_some_and(|t| Instant::now() < t) {
                return Ok(Step::More);
            }
            self.redraw(out)?;
            write!(out, "^C")?;
            self.leave(out)?;
            write!(out, "{QUIT_HINT}\r\n")?;
            self.dirty = true;
            return Ok(Step::More);
        }
        match op.map(|op| self.ed.apply(op)) {
            Some(Outcome::Submit(msg)) => {
                self.draw(out, &msg, msg.len())?;
                self.dirty = false;
                self.leave(out)?;
                return Ok(Step::Done(Some(msg)));
            }
            Some(Outcome::Eof) if queued => {}
            Some(Outcome::Eof) => {
                self.leave(out)?;
                return Ok(Step::Done(None));
            }
            Some(Outcome::ClearScreen) => {
                write!(out, "\x1b[H\x1b[2J")?;
                (self.row, self.end_row) = (0, 0);
                self.dirty = true;
            }
            Some(Outcome::Redraw) => self.dirty = true,
            None => {}
        }
        Ok(Step::More)
    }

    /// Read a message: keys from `typeahead` first, then from the terminal.
    /// None at end of input (Ctrl-D on an empty line). Raw mode must be on,
    /// the cursor at the start of a row.
    pub fn read(
        &mut self,
        out: &mut impl Write,
        typeahead: &mut VecDeque<Event>,
    ) -> io::Result<Option<String>> {
        // Input that came after the reply's last check for it is typeahead
        // too; a Ctrl-C there was meant for the reply and is dropped.
        poll_typeahead(typeahead);
        self.grace_until = Some(Instant::now() + LATE_CTRL_C);
        self.width = term_width();
        (self.row, self.end_row) = (0, 0);
        self.dirty = true;
        self.redraw(out)?;
        loop {
            let (ev, queued) = match typeahead.pop_front() {
                Some(ev) => (ev, true),
                None => (event::read()?, false),
            };
            if let Step::Done(msg) = self.feed(out, ev, queued)? {
                return Ok(msg);
            }
            // A burst of keys (typeahead, or a paste the terminal didn't
            // bracket) draws once, at its end.
            if typeahead.is_empty() && !event::poll(Duration::ZERO)? {
                self.redraw(out)?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer with "|" at the cursor.
    fn show(ed: &Editor) -> String {
        let mut s = ed.text().to_string();
        s.insert(ed.cursor(), '|');
        s
    }

    fn typed(s: &str) -> Editor {
        let mut ed = Editor::default();
        for c in s.chars() {
            ed.apply(Op::Insert(c));
        }
        ed
    }

    fn run(ed: &mut Editor, ops: &[Op]) {
        for op in ops {
            ed.apply(op.clone());
        }
    }

    fn submit(ed: &mut Editor, s: &str) -> Outcome {
        ed.apply(Op::Paste(s.into()));
        ed.apply(Op::Enter)
    }

    #[test]
    fn insert_utf8_and_move() {
        let mut ed = typed("añ日😀");
        assert_eq!(show(&ed), "añ日😀|");
        run(&mut ed, &[Op::Left, Op::Left]);
        assert_eq!(show(&ed), "añ|日😀");
        ed.apply(Op::Insert('x'));
        assert_eq!(show(&ed), "añx|日😀");
        run(&mut ed, &[Op::Right, Op::Right, Op::Right]);
        assert_eq!(show(&ed), "añx日😀|");
        run(&mut ed, &[Op::Home, Op::Left]);
        assert_eq!(show(&ed), "|añx日😀");
        ed.apply(Op::End);
        assert_eq!(show(&ed), "añx日😀|");
        // Control characters typed are not text.
        ed.apply(Op::Insert('\x07'));
        assert_eq!(show(&ed), "añx日😀|");
    }

    #[test]
    fn combining_marks_ride_with_their_letter() {
        // "e" + COMBINING ACUTE, then "x".
        let mut ed = typed("e\u{301}x");
        ed.apply(Op::Left);
        assert_eq!(show(&ed), "e\u{301}|x");
        ed.apply(Op::Left);
        assert_eq!(show(&ed), "|e\u{301}x");
        ed.apply(Op::Right);
        assert_eq!(show(&ed), "e\u{301}|x");
        ed.apply(Op::Backspace);
        assert_eq!(show(&ed), "|x");
        let mut ed = typed("e\u{301}x");
        run(&mut ed, &[Op::Home, Op::Delete]);
        assert_eq!(show(&ed), "|x");
    }

    #[test]
    fn word_motion() {
        let mut ed = typed("foo bar_1, baz");
        ed.apply(Op::WordLeft);
        assert_eq!(show(&ed), "foo bar_1, |baz");
        ed.apply(Op::WordLeft);
        assert_eq!(show(&ed), "foo |bar_1, baz");
        run(&mut ed, &[Op::WordLeft, Op::WordLeft]);
        assert_eq!(show(&ed), "|foo bar_1, baz");
        ed.apply(Op::WordRight);
        assert_eq!(show(&ed), "foo| bar_1, baz");
        ed.apply(Op::WordRight);
        assert_eq!(show(&ed), "foo bar_1|, baz");
        run(&mut ed, &[Op::WordRight, Op::WordRight]);
        assert_eq!(show(&ed), "foo bar_1, baz|");
    }

    #[test]
    fn backspace_and_delete() {
        let mut ed = typed("abc");
        ed.apply(Op::Backspace);
        assert_eq!(show(&ed), "ab|");
        run(&mut ed, &[Op::Home, Op::Backspace]);
        assert_eq!(show(&ed), "|ab");
        ed.apply(Op::Delete);
        assert_eq!(show(&ed), "|b");
        run(&mut ed, &[Op::End, Op::Delete]);
        assert_eq!(show(&ed), "b|");
        let mut ed = typed("日本");
        ed.apply(Op::Backspace);
        assert_eq!(show(&ed), "日|");
    }

    #[test]
    fn ctrl_d_is_eof_only_on_an_empty_buffer() {
        let mut ed = Editor::default();
        assert_eq!(ed.apply(Op::DeleteOrEof), Outcome::Eof);
        let mut ed = typed("ab");
        ed.apply(Op::Home);
        assert_eq!(ed.apply(Op::DeleteOrEof), Outcome::Redraw);
        assert_eq!(show(&ed), "|b");
        // At the end of a non-empty line it does nothing.
        ed.apply(Op::End);
        assert_eq!(ed.apply(Op::DeleteOrEof), Outcome::Redraw);
        assert_eq!(show(&ed), "b|");
    }

    #[test]
    fn ctrl_c_clears_and_never_ends_input() {
        let mut ed = typed("half a thought");
        ed.apply(Op::Left);
        assert_eq!(ed.apply(Op::Interrupt), Outcome::Redraw);
        assert_eq!(show(&ed), "|");
        assert_eq!(ed.apply(Op::Interrupt), Outcome::Redraw);
        assert_eq!(show(&ed), "|");
    }

    #[test]
    fn zero_width_char_starting_a_line_is_its_own_stop() {
        let mut ed = Editor::default();
        ed.apply(Op::Paste("a\n\u{301}x".into()));
        run(&mut ed, &[Op::Home, Op::Up, Op::End]);
        assert_eq!(show(&ed), "a|\n\u{301}x");
        ed.apply(Op::Right);
        assert_eq!(show(&ed), "a\n|\u{301}x");
        ed.apply(Op::Right);
        assert_eq!(show(&ed), "a\n\u{301}|x");
        ed.apply(Op::Left);
        assert_eq!(show(&ed), "a\n|\u{301}x");
        ed.apply(Op::Left);
        assert_eq!(show(&ed), "a|\n\u{301}x");
        // Ctrl-K at the line end takes the line break only.
        ed.apply(Op::KillToEnd);
        assert_eq!(show(&ed), "a|\u{301}x");
        let mut ed = typed("a\u{301}");
        ed.apply(Op::Left);
        assert_eq!(show(&ed), "|a\u{301}");
    }

    #[test]
    fn up_down_keep_the_column_in_cursor_stops() {
        let mut ed = Editor::default();
        ed.apply(Op::Paste("e\u{301}fg\nabc".into()));
        run(&mut ed, &[Op::Home, Op::Right]);
        assert_eq!(show(&ed), "e\u{301}fg\na|bc");
        // Column 1 is after "e" and its accent, not between them.
        ed.apply(Op::Up);
        assert_eq!(show(&ed), "e\u{301}|fg\nabc");
        ed.apply(Op::Right);
        ed.apply(Op::Down);
        assert_eq!(show(&ed), "e\u{301}fg\nab|c");
    }

    #[test]
    fn kills_and_yank() {
        let mut ed = typed("ls -la /tmp/foo.txt");
        ed.apply(Op::KillUnixWordBack);
        assert_eq!(show(&ed), "ls -la |");
        ed.apply(Op::Yank);
        assert_eq!(show(&ed), "ls -la /tmp/foo.txt|");
        ed.apply(Op::KillWordBack);
        assert_eq!(show(&ed), "ls -la /tmp/foo.|");
        run(&mut ed, &[Op::Home, Op::KillWordForward]);
        assert_eq!(show(&ed), "| -la /tmp/foo.");
        run(&mut ed, &[Op::WordRight, Op::KillToEnd]);
        assert_eq!(show(&ed), " -la|");
        ed.apply(Op::KillToStart);
        assert_eq!(show(&ed), "|");
        ed.apply(Op::Yank);
        assert_eq!(show(&ed), " -la|");
    }

    #[test]
    fn line_kills_stay_on_their_line_but_join_at_its_edge() {
        let mut ed = Editor::default();
        ed.apply(Op::Paste("one\ntwo".into()));
        ed.apply(Op::Left);
        ed.apply(Op::KillToStart);
        assert_eq!(show(&ed), "one\n|o");
        ed.apply(Op::KillToStart);
        assert_eq!(show(&ed), "one|o");
        ed.apply(Op::Paste("\n".into()));
        run(&mut ed, &[Op::Up, Op::End, Op::KillToEnd]);
        assert_eq!(show(&ed), "one|o");
        run(&mut ed, &[Op::Left, Op::KillToEnd]);
        assert_eq!(show(&ed), "on|");
    }

    #[test]
    fn home_end_and_up_down_within_a_multiline_message() {
        let mut ed = Editor::default();
        ed.apply(Op::Paste("first line\nab\nthird".into()));
        assert_eq!(show(&ed), "first line\nab\nthird|");
        ed.apply(Op::Home);
        assert_eq!(show(&ed), "first line\nab\n|third");
        run(&mut ed, &[Op::Right, Op::Right, Op::Right, Op::Up]);
        // "ab" is shorter: the end of it.
        assert_eq!(show(&ed), "first line\nab|\nthird");
        ed.apply(Op::Up);
        assert_eq!(show(&ed), "fi|rst line\nab\nthird");
        ed.apply(Op::End);
        assert_eq!(show(&ed), "first line|\nab\nthird");
        ed.apply(Op::Down);
        assert_eq!(show(&ed), "first line\nab|\nthird");
        ed.apply(Op::Down);
        assert_eq!(show(&ed), "first line\nab\nth|ird");
    }

    #[test]
    fn history_browse_keeps_the_draft() {
        let mut ed = Editor::default();
        assert_eq!(submit(&mut ed, "one"), Outcome::Submit("one".into()));
        submit(&mut ed, "two");
        ed.apply(Op::Paste("dra".into()));
        ed.apply(Op::Up);
        assert_eq!(show(&ed), "two|");
        ed.apply(Op::Up);
        assert_eq!(show(&ed), "one|");
        ed.apply(Op::Up); // the oldest: stays
        assert_eq!(show(&ed), "one|");
        ed.apply(Op::Down);
        assert_eq!(show(&ed), "two|");
        ed.apply(Op::Down);
        assert_eq!(show(&ed), "dra|");
        ed.apply(Op::Down); // the newest: stays
        assert_eq!(show(&ed), "dra|");
        // Resubmitting a recalled entry: it's the last one, kept once.
        run(&mut ed, &[Op::Interrupt, Op::Up]);
        assert_eq!(ed.apply(Op::Enter), Outcome::Submit("two".into()));
        assert_eq!(ed.history, ["one", "two"]);
    }

    #[test]
    fn history_collapses_repeats_and_skips_blanks() {
        let mut ed = Editor::default();
        for m in ["a", "a", "b", "a", "", "  "] {
            submit(&mut ed, m);
        }
        assert_eq!(ed.history, ["a", "b", "a"]);
    }

    #[test]
    fn editing_a_recalled_entry_leaves_history_alone() {
        let mut ed = Editor::default();
        submit(&mut ed, "hello");
        submit(&mut ed, "world");
        run(&mut ed, &[Op::Up, Op::Up, Op::Backspace, Op::Insert('!')]);
        assert_eq!(show(&ed), "hell!|");
        run(&mut ed, &[Op::Down, Op::Up]);
        assert_eq!(show(&ed), "hello|");
        ed.apply(Op::Insert('?'));
        assert_eq!(ed.apply(Op::Enter), Outcome::Submit("hello?".into()));
        assert_eq!(ed.history, ["hello", "world", "hello?"]);
        assert_eq!(show(&ed), "|");
    }

    #[test]
    fn multiline_history_entry_up_moves_inside_it_first() {
        let mut ed = Editor::default();
        submit(&mut ed, "a\nb");
        run(&mut ed, &[Op::Up]);
        assert_eq!(show(&ed), "a\nb|");
        run(&mut ed, &[Op::Up]);
        assert_eq!(show(&ed), "a|\nb");
    }

    #[test]
    fn backslash_enter_continues_the_message() {
        let mut ed = typed("line one\\");
        assert_eq!(ed.apply(Op::Enter), Outcome::Redraw);
        assert_eq!(show(&ed), "line one\n|");
        ed.apply(Op::Paste("two".into()));
        assert_eq!(ed.apply(Op::Enter), Outcome::Submit("line one\ntwo".into()));
        // Alt-Enter breaks the line where the cursor is.
        let mut ed = typed("ab");
        run(&mut ed, &[Op::Left, Op::Newline]);
        assert_eq!(show(&ed), "a\n|b");
    }

    #[test]
    fn paste_with_newlines_is_one_message() {
        let mut ed = typed("say: ");
        assert_eq!(
            ed.apply(Op::Paste("one\r\ntwo\rthree\n\tfour\x1b\x07".into())),
            Outcome::Redraw
        );
        assert_eq!(show(&ed), "say: one\ntwo\nthree\n    four|");
        assert_eq!(
            ed.apply(Op::Enter),
            Outcome::Submit("say: one\ntwo\nthree\n    four".into())
        );
    }

    #[test]
    fn key_map() {
        let k = |code, m| op_for(&KeyEvent::new(code, m));
        let (none, ctrl, alt) = (KeyModifiers::NONE, KeyModifiers::CONTROL, KeyModifiers::ALT);
        assert_eq!(k(KeyCode::Char('é'), none), Some(Op::Insert('é')));
        assert_eq!(
            k(KeyCode::Char('A'), KeyModifiers::SHIFT),
            Some(Op::Insert('A'))
        );
        assert_eq!(k(KeyCode::Left, none), Some(Op::Left));
        assert_eq!(k(KeyCode::Left, ctrl), Some(Op::WordLeft));
        assert_eq!(k(KeyCode::Right, ctrl), Some(Op::WordRight));
        assert_eq!(k(KeyCode::Char('b'), alt), Some(Op::WordLeft));
        assert_eq!(k(KeyCode::Char('f'), alt), Some(Op::WordRight));
        assert_eq!(k(KeyCode::Char('a'), ctrl), Some(Op::Home));
        assert_eq!(k(KeyCode::Char('e'), ctrl), Some(Op::End));
        assert_eq!(k(KeyCode::Home, none), Some(Op::Home));
        assert_eq!(k(KeyCode::End, none), Some(Op::End));
        assert_eq!(k(KeyCode::Backspace, none), Some(Op::Backspace));
        assert_eq!(k(KeyCode::Char('h'), ctrl), Some(Op::Backspace));
        assert_eq!(k(KeyCode::Backspace, alt), Some(Op::KillWordBack));
        assert_eq!(k(KeyCode::Delete, none), Some(Op::Delete));
        assert_eq!(k(KeyCode::Char('d'), ctrl), Some(Op::DeleteOrEof));
        assert_eq!(k(KeyCode::Char('w'), ctrl), Some(Op::KillUnixWordBack));
        assert_eq!(k(KeyCode::Char('u'), ctrl), Some(Op::KillToStart));
        assert_eq!(k(KeyCode::Char('k'), ctrl), Some(Op::KillToEnd));
        assert_eq!(k(KeyCode::Char('y'), ctrl), Some(Op::Yank));
        assert_eq!(k(KeyCode::Up, none), Some(Op::Up));
        assert_eq!(k(KeyCode::Down, none), Some(Op::Down));
        assert_eq!(k(KeyCode::Enter, none), Some(Op::Enter));
        assert_eq!(k(KeyCode::Enter, alt), Some(Op::Newline));
        assert_eq!(k(KeyCode::Char('c'), ctrl), Some(Op::Interrupt));
        assert_eq!(k(KeyCode::Char('l'), ctrl), Some(Op::ClearScreen));
        assert_eq!(k(KeyCode::Esc, none), None);
        assert_eq!(k(KeyCode::Char('z'), ctrl), None);
    }

    fn key(code: KeyCode, m: KeyModifiers) -> Event {
        Event::Key(KeyEvent::new(code, m))
    }

    fn ctrl(c: char) -> Event {
        key(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn reader(width: usize) -> LineReader {
        LineReader {
            width,
            ..LineReader::default()
        }
    }

    /// Feed events (typed live), returning the last step and what was written.
    fn feed(lr: &mut LineReader, evs: Vec<Event>) -> (Step, String) {
        let mut out = Vec::new();
        let mut step = Step::More;
        for ev in evs {
            step = lr.feed(&mut out, ev, false).unwrap();
        }
        (step, String::from_utf8(out).unwrap())
    }

    fn redrawn(lr: &mut LineReader) -> String {
        let mut out = Vec::new();
        lr.redraw(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn draw_two_rows_cursor_on_the_first() {
        let mut lr = reader(10);
        let mut out = Vec::new();
        lr.draw(&mut out, "abcdefghij", 3).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\r\x1b[J\x1b[1m> \x1b[0mabcdefgh\r\nij\x1b[1A\r\x1b[5C"
        );
        assert_eq!((lr.row, lr.end_row), (0, 1));
        // From row 0 no move up; the cursor at the end needs none either.
        let mut out = Vec::new();
        lr.draw(&mut out, "abcdefghij", 10).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\r\x1b[J\x1b[1m> \x1b[0mabcdefgh\r\nij\r\x1b[2C"
        );
        // From row 1, up to the prompt's row first.
        let mut out = Vec::new();
        lr.draw(&mut out, "ab", 2).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\r\x1b[1A\x1b[J\x1b[1m> \x1b[0mab\r\x1b[4C"
        );
    }

    #[test]
    fn submit_draws_the_whole_message_and_leaves_below_it() {
        let mut lr = reader(80);
        feed(&mut lr, vec![Event::Paste("ab\ncd".into())]);
        redrawn(&mut lr);
        feed(&mut lr, vec![key(KeyCode::Up, KeyModifiers::NONE)]);
        assert_eq!(
            redrawn(&mut lr),
            "\r\x1b[1A\x1b[J\x1b[1m> \x1b[0mab\r\n\x1b[1m. \x1b[0mcd\x1b[1A\r\x1b[4C"
        );
        let (step, out) = feed(&mut lr, vec![key(KeyCode::Enter, KeyModifiers::NONE)]);
        assert_eq!(step, Step::Done(Some("ab\ncd".into())));
        // Redrawn with the cursor at the end, then a new row.
        assert_eq!(
            out,
            "\r\x1b[J\x1b[1m> \x1b[0mab\r\n\x1b[1m. \x1b[0mcd\r\x1b[4C\r\n"
        );
        assert!(!lr.dirty);
        assert_eq!((lr.row, lr.end_row), (0, 0));
    }

    #[test]
    fn ctrl_c_on_an_empty_line_shows_a_hint_and_a_new_prompt() {
        let mut lr = reader(80);
        lr.dirty = true;
        redrawn(&mut lr);
        let (step, out) = feed(&mut lr, vec![ctrl('c')]);
        assert_eq!(step, Step::More);
        assert_eq!(out, format!("^C\r\n{QUIT_HINT}\r\n"));
        assert_eq!(redrawn(&mut lr), "\r\x1b[J\x1b[1m> \x1b[0m\r\x1b[2C");
        // On a non-empty line it clears it.
        let (step, out) = feed(
            &mut lr,
            vec![key(KeyCode::Char('a'), KeyModifiers::NONE), ctrl('c')],
        );
        assert_eq!((step, out.as_str(), lr.ed.text()), (Step::More, "", ""));
        assert!(lr.dirty);
    }

    #[test]
    fn ctrl_c_right_after_the_prompt_appears_is_dropped() {
        let mut lr = reader(80);
        lr.grace_until = Some(Instant::now() + Duration::from_secs(60));
        assert_eq!(feed(&mut lr, vec![ctrl('c')]), (Step::More, String::new()));
        assert!(!lr.dirty);
        lr.grace_until = Some(Instant::now());
        assert!(feed(&mut lr, vec![ctrl('c')]).1.starts_with("^C"));
    }

    #[test]
    fn late_ctrl_c_is_not_typeahead() {
        let mut q = VecDeque::new();
        assert!(queue_event(&mut q, ctrl('c')));
        assert!(q.is_empty());
        assert!(!queue_event(&mut q, Event::Resize(10, 10)));
        assert!(!queue_event(&mut q, ctrl('d')));
        assert!(!queue_event(&mut q, Event::Paste("x".into())));
        assert_eq!(q, [ctrl('d'), Event::Paste("x".into())]);
    }

    #[test]
    fn queued_ctrl_d_is_dropped_queued_enter_submits() {
        let mut lr = reader(80);
        let mut out = Vec::new();
        assert_eq!(lr.feed(&mut out, ctrl('d'), true).unwrap(), Step::More);
        assert!(out.is_empty());
        lr.feed(&mut out, key(KeyCode::Char('h'), KeyModifiers::NONE), true)
            .unwrap();
        let enter = key(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(
            lr.feed(&mut out, enter, true).unwrap(),
            Step::Done(Some("h".into()))
        );
        // Typed live, Ctrl-D on an empty line ends input.
        assert_eq!(feed(&mut lr, vec![ctrl('d')]).0, Step::Done(None));
    }

    #[test]
    fn a_burst_ending_in_a_no_op_still_draws() {
        let mut lr = reader(80);
        let mut release = KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE);
        release.kind = KeyEventKind::Release;
        feed(
            &mut lr,
            vec![
                key(KeyCode::Char('a'), KeyModifiers::NONE),
                key(KeyCode::Esc, KeyModifiers::NONE),
                key(KeyCode::F(5), KeyModifiers::NONE),
                Event::Key(release),
            ],
        );
        assert!(lr.dirty);
        assert_eq!(redrawn(&mut lr), "\r\x1b[J\x1b[1m> \x1b[0ma\r\x1b[3C");
        assert_eq!(redrawn(&mut lr), "");
        feed(&mut lr, vec![Event::Resize(40, 10)]);
        assert_eq!((lr.width, lr.dirty), (40, true));
    }

    /// The text of a layout without the prompt's bold escapes.
    fn plain(l: &Layout) -> String {
        l.text.replace(BOLD, "").replace(PLAIN, "")
    }

    #[test]
    fn layout_wraps_at_the_width() {
        let l = layout("abcdefgh", 3, 5);
        assert_eq!(plain(&l), "> abc\r\ndefgh\r\n");
        // At "d", which starts the second row.
        assert_eq!(l.cursor, (1, 0));
        assert_eq!(l.end, (2, 0));
        // The cursor before a character that starts a row is on that row.
        let l = layout("abcdefgh", 3, 6);
        assert_eq!(plain(&l), "> abcd\r\nefgh");
        assert_eq!(l.cursor, (0, 5));
        assert_eq!(layout("abcdefgh", 4, 6).cursor, (1, 0));
        assert_eq!(l.end, (1, 4));
    }

    #[test]
    fn layout_cursor_after_a_full_row_is_on_the_next() {
        let l = layout("abcd", 4, 6);
        assert_eq!(plain(&l), "> abcd\r\n");
        assert_eq!((l.cursor, l.end), ((1, 0), (1, 0)));
        let l = layout("", 0, 80);
        assert_eq!(plain(&l), "> ");
        assert_eq!((l.cursor, l.end), ((0, 2), (0, 2)));
    }

    #[test]
    fn layout_wide_characters() {
        // "日" doesn't fit in the last column: it starts the next row.
        let l = layout("ab日本", 2, 5);
        assert_eq!(plain(&l), "> ab\r\n日本");
        assert_eq!(l.cursor, (1, 0));
        assert_eq!(l.end, (1, 4));
        // Exactly filling the row with a wide character.
        let l = layout("a😀x", "a😀".len(), 5);
        assert_eq!(plain(&l), "> a😀\r\nx");
        assert_eq!(l.cursor, (1, 0));
        // Combining marks take no column.
        let l = layout("e\u{301}x", 3, 80);
        assert_eq!(l.cursor, (0, 3));
        assert_eq!(l.end, (0, 4));
    }

    #[test]
    fn layout_multiline_prompts() {
        let l = layout("ab\ncd\n", 4, 80);
        assert_eq!(plain(&l), "> ab\r\n. cd\r\n. ");
        assert_eq!(l.cursor, (1, 3));
        assert_eq!(l.end, (2, 2));
        // A line exactly filling its row: the cursor at its end gets a row
        // of its own rather than standing on the next line's prompt.
        let l = layout("abc\nd", 3, 5);
        assert_eq!(plain(&l), "> abc\r\n\r\n. d");
        assert_eq!(l.cursor, (1, 0));
        assert_eq!(l.end, (2, 3));
    }

    #[test]
    fn layout_narrow_terminal() {
        let l = layout("日x", 4, 1);
        assert_eq!(plain(&l), ">\r\n \r\n日\r\nx\r\n");
        assert_eq!(l.end, (4, 0));
    }

    #[test]
    fn tidy_paste() {
        assert_eq!(tidy("a\r\nb\rc\n\td\x00e"), "a\nb\nc\n    de");
    }
}
