//! Token masks: which tokens the automaton takes from a state, cached per
//! (node, stack top).
//!
//! From a node, walking the vocabulary trie (vocab.rs) with the live
//! stack's top frame known and everything below it unknown, a token is
//! accepted (its bytes never pop below the top), rejected (some prefix is
//! refused: its whole subtree is skipped), or deep: it pops past the top
//! (`",` closes a string and a member, fine; `"}}` closes two containers,
//! deep), and is checked at use against the live stack. So the mask of a
//! state is the cached bits of its (node, top, room) plus the deep tokens
//! that pass now. `room` is how many more frames may open before the depth
//! cap (`pda::MAX_DEPTH`), exact near the cap and `ROOM_FAR` (at least
//! that many) elsewhere: a token opening more containers than that is deep.
//! Special tokens are symbols, checked one by one the same way.

use super::pda::{Graph, MAX_DEPTH, NodeId, Pop, Stack, Step, Sym};
use super::vocab::Vocab;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// `room` at or past this is one cache key (module doc).
const ROOM_FAR: i32 = 32;

/// Bytes of cached masks a grammar keeps at most; past it the cache starts
/// over (a free-JSON grammar holds a few dozen entries, ~19 KB each at a
/// 152k vocabulary).
const CACHE_BUDGET: usize = 64 << 20;

/// A set of token ids out of a vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bits {
    words: Vec<u64>,
    count: u32,
}

impl Bits {
    pub fn new(n: usize) -> Bits {
        Bits {
            words: vec![0; n.div_ceil(64)],
            count: 0,
        }
    }

    fn set(&mut self, id: u32) {
        let w = &mut self.words[(id >> 6) as usize];
        let b = 1u64 << (id & 63);
        if *w & b == 0 {
            *w |= b;
            self.count += 1;
        }
    }

    fn has(&self, id: u32) -> bool {
        (self.words.get((id >> 6) as usize)).is_some_and(|w| (w >> (id & 63)) & 1 == 1)
    }

    fn bytes(&self) -> usize {
        self.words.len() * 8
    }
}

/// The tokens a constrained position may take (`Constraint::mask`): a
/// cached set plus a few tokens checked against the live state. Cheap to
/// clone.
#[derive(Clone, Debug)]
pub struct TokenMask {
    base: Arc<Bits>,
    /// Allowed ids not in `base`, ascending.
    extra: Vec<u32>,
}

impl TokenMask {
    /// Exactly `ids` (any order) out of `n` tokens.
    pub fn from_ids(n: usize, ids: &[u32]) -> TokenMask {
        let mut b = Bits::new(n);
        for &id in ids {
            if (id as usize) < n {
                b.set(id);
            }
        }
        TokenMask {
            base: Arc::new(b),
            extra: Vec::new(),
        }
    }

    pub fn allows(&self, id: u32) -> bool {
        self.base.has(id) || (!self.extra.is_empty() && self.extra.binary_search(&id).is_ok())
    }

    /// How many tokens it allows.
    pub fn count(&self) -> usize {
        self.base.count as usize + self.extra.len()
    }

    /// The allowed ids: `base`'s ascending, then `extra`'s.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        let words = self.base.words.iter().enumerate().flat_map(|(i, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let b = w.trailing_zeros();
                w &= w - 1;
                Some((i as u32) * 64 + b)
            })
        });
        words.chain(self.extra.iter().copied())
    }
}

/// A cached mask: the context-independent part and the deep tokens.
#[derive(Debug)]
pub struct Entry {
    pub bits: Arc<Bits>,
    /// Tokens checked at use against the live stack, ascending.
    pub deep: Vec<u32>,
}

/// (node, the live stack's top, room).
pub type Key = (NodeId, Option<NodeId>, i32);

impl Entry {
    /// Tokens in the context-independent part.
    #[cfg(test)]
    pub fn bits_count(&self) -> usize {
        self.bits.count as usize
    }
}

/// A grammar on a vocabulary, with its mask cache: shared by every request
/// that uses it.
pub struct Masks {
    pub graph: Graph,
    pub vocab: Arc<Vocab>,
    cache: Mutex<(HashMap<Key, Arc<Entry>>, usize)>,
}

/// The arena stack of a walk: `cells[i]` = (frame, the cell below). Cell 0
/// is "unknown" (below the live top), 1 "empty" (the live stack is), 2 the
/// live top.
struct Walk<'a> {
    cells: &'a mut Vec<(NodeId, u32)>,
    sp: u32,
    /// Frames pushed minus popped since the walk began.
    rel: i32,
    room: i32,
}

const UNKNOWN: u32 = 0;
const EMPTY: u32 = 1;
const TOP: u32 = 2;

impl Stack for Walk<'_> {
    fn push(&mut self, ret: NodeId) {
        self.cells.push((ret, self.sp));
        self.sp = (self.cells.len() - 1) as u32;
        self.rel += 1;
    }

    fn pop(&mut self) -> Pop {
        match self.sp {
            UNKNOWN => Pop::Unknown,
            EMPTY => Pop::Empty,
            sp => {
                let (n, below) = self.cells[sp as usize];
                self.sp = below;
                self.rel -= 1;
                Pop::Node(n)
            }
        }
    }

    fn may_open(&self) -> Option<bool> {
        if self.rel <= self.room {
            Some(true)
        } else if self.room >= ROOM_FAR {
            None
        } else {
            Some(false)
        }
    }
}

impl Masks {
    pub fn new(graph: Graph, vocab: Arc<Vocab>) -> Masks {
        Masks {
            graph,
            vocab,
            cache: Mutex::new((HashMap::new(), 0)),
        }
    }

    /// The cache key of (`node`, `stack`).
    pub fn key(node: NodeId, stack: &[NodeId]) -> Key {
        let room = (MAX_DEPTH as i64 - stack.len() as i64).min(ROOM_FAR as i64) as i32;
        (node, stack.last().copied(), room)
    }

    /// The cached entry of (`node`, `stack`), computed on a miss.
    pub fn entry(&self, node: NodeId, stack: &[NodeId]) -> Arc<Entry> {
        let key = Self::key(node, stack);
        if let Some(e) = self.lock().0.get(&key) {
            return e.clone();
        }
        let e = Arc::new(self.compute(key));
        let mut c = self.lock();
        let size = e.bits.bytes() + e.deep.len() * 4;
        if c.1 + size > CACHE_BUDGET {
            c.0.clear();
            c.1 = 0;
        }
        c.1 += size;
        c.0.insert(key, e.clone());
        e
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, (HashMap<Key, Arc<Entry>>, usize)> {
        // A panic elsewhere while holding it leaves a valid map.
        self.cache.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Entries cached now (tests).
    #[cfg(test)]
    pub fn cached(&self) -> usize {
        self.lock().0.len()
    }

    /// Walk the trie and the specials from `key`'s state.
    pub fn compute(&self, (node, top, room): Key) -> Entry {
        let v = &self.vocab;
        let t = v.trie();
        let mut bits = Bits::new(v.len());
        let mut deep = Vec::new();
        let mut cells: Vec<(NodeId, u32)> = vec![(0, 0), (0, 0), (top.unwrap_or(0), UNKNOWN)];
        let sp0 = if top.is_some() { TOP } else { EMPTY };
        // Per depth: (node, sp, rel) after the path's bytes, and the arena's
        // length then (deeper cells are garbage once the walk moves on).
        let mut at = vec![(node, sp0, 0i32, cells.len()); t.max_depth + 1];
        let mut i = 0;
        while i < t.len() {
            let d = t.depth[i] as usize;
            let (n0, sp, rel, len) = at[d - 1];
            cells.truncate(len);
            let mut n = n0;
            let mut w = Walk {
                cells: &mut cells,
                sp,
                rel,
                room,
            };
            match self.graph.step(&mut n, &mut w, Sym::Byte(t.byte[i])) {
                Step::Ok => {
                    let (sp, rel) = (w.sp, w.rel);
                    at[d] = (n, sp, rel, cells.len());
                    for &id in t.toks_at(i) {
                        bits.set(id);
                    }
                    i += 1;
                }
                Step::Reject => i = t.end[i] as usize,
                Step::Unknown => {
                    deep.extend_from_slice(t.toks_under(i));
                    i = t.end[i] as usize;
                }
            }
        }
        for &id in v.specials() {
            cells.truncate(3);
            let mut n = node;
            let mut w = Walk {
                cells: &mut cells,
                sp: sp0,
                rel: 0,
                room,
            };
            match self.graph.step(&mut n, &mut w, Sym::Special(id)) {
                Step::Ok => bits.set(id),
                Step::Reject => {}
                Step::Unknown => deep.push(id),
            }
        }
        deep.sort_unstable();
        Entry {
            bits: Arc::new(bits),
            deep,
        }
    }

    /// Whether token `id` is taken at (`node`, `stack`), and the state
    /// after it: the live check (deep tokens, `Constraint::accept`). `buf`
    /// is scratch for the stack.
    pub fn walk(
        &self,
        node: NodeId,
        stack: &[NodeId],
        id: u32,
        buf: &mut Vec<NodeId>,
    ) -> Option<NodeId> {
        buf.clear();
        buf.extend_from_slice(stack);
        let mut n = node;
        if self.vocab.is_special(id) {
            return (self.graph.step(&mut n, buf, Sym::Special(id)) == Step::Ok).then_some(n);
        }
        let b = self.vocab.bytes(id);
        if b.is_empty() {
            return None;
        }
        for &x in b {
            if self.graph.step(&mut n, buf, Sym::Byte(x)) != Step::Ok {
                return None;
            }
        }
        Some(n)
    }

    /// The mask at (`node`, `stack`): the cached entry plus the deep tokens
    /// that pass on the live stack.
    pub fn mask(&self, node: NodeId, stack: &[NodeId], buf: &mut Vec<NodeId>) -> TokenMask {
        let e = self.entry(node, stack);
        let extra = (e.deep.iter().copied())
            .filter(|&id| self.walk(node, stack, id, buf).is_some())
            .collect();
        TokenMask {
            base: e.bits.clone(),
            extra,
        }
    }
}
