//! The prefix index of a paged pool's KV pages (specs/shared-prefix-pages.md
//! §3.1): which page holds the K/V of each 64-token block after a given
//! prefix, so a prompt that starts the same way maps those pages instead of
//! prefilling them. Host only; the pool counts each page's holders and pins
//! the indexed ones (`PageMap::pin`), the engine keeps the two in step.
//!
//! Why sharing is exact: a page's bytes are a function of the model's
//! identity and the tokens of its positions and every earlier one, never of
//! who wrote it (the bit-identity contract: batched == chunked == decode,
//! the physical page never in the arithmetic, rows past a reader's length
//! never read). One pool is one identity, so the key is the tokens: the
//! kvstore hash chain, `key_b = H(key_{b-1}, tokens of block b)`, with each
//! block's tokens and parent compared on every hit, so a hash collision is
//! a miss, never another prefix's bytes. An indexed page is immutable (the
//! pool's `writable`): a slot about to write into one gives it up first.
//!
//! Eviction is kvstore's: least recently used among the leaves (pages no
//! indexed page extends) that no slot holds, so a chain never loses its
//! root while a longer one is indexed, and a cold conversation goes tail
//! first. A lookup touches deepest first (a prefix is never older than its
//! extension).

#![cfg_attr(not(test), allow(dead_code))] // the engine takes it in C3

use crate::kvsave::PAGE;
use crate::kvstore::{Key, ROOT, chain};
use std::collections::{BTreeSet, HashMap};

/// An indexed page given up (`pop_lru_leaf`, `unindex_subtree`): the caller
/// unpins `page` (a RAM tier may first copy it out under `key`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evicted {
    pub key: Key,
    pub parent: Key,
    /// The block's PAGE tokens.
    pub tokens: Box<[u32]>,
    pub page: u32,
}

/// `insert`'s answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ins {
    /// Indexed now: pin the page.
    New(Key),
    /// The block is indexed already, at that page (the caller's page holds
    /// the same bytes: keep it unindexed, or swap to this one).
    Have(Key, u32),
    /// Not indexed: the parent isn't, or a different block has this key.
    No,
}

struct Node {
    parent: Key,
    tokens: Box<[u32]>,
    page: u32,
    /// Indexed blocks that extend this one.
    kids: Vec<Key>,
    /// Slots holding the page (the pool's refs, mirrored).
    holds: u32,
    last: u64,
}

pub struct PrefixIndex {
    /// The chain's start: kvstore::ROOT (a tenant's salted root later).
    root: Key,
    map: HashMap<Key, Node>,
    /// (last use, key) of the evictable blocks: unheld leaves.
    lru: BTreeSet<(u64, Key)>,
    /// Page -> its key (a page is indexed under one key at most).
    by_page: HashMap<u32, Key>,
    tick: u64,
}

impl Default for PrefixIndex {
    fn default() -> PrefixIndex {
        PrefixIndex::new(ROOT)
    }
}

impl PrefixIndex {
    pub fn new(root: Key) -> PrefixIndex {
        PrefixIndex {
            root,
            map: HashMap::new(),
            lru: BTreeSet::new(),
            by_page: HashMap::new(),
            tick: 0,
        }
    }

    pub fn root(&self) -> Key {
        self.root
    }

    /// Indexed pages.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The key under which `page` is indexed, if it is.
    pub fn key_of(&self, page: u32) -> Option<Key> {
        self.by_page.get(&page).copied()
    }

    pub fn page(&self, key: Key) -> Option<u32> {
        self.map.get(&key).map(|n| n.page)
    }

    fn evictable(n: &Node) -> bool {
        n.kids.is_empty() && n.holds == 0
    }

    fn touch(&mut self, key: Key) {
        self.tick += 1;
        let t = self.tick;
        let n = self.map.get_mut(&key).expect("an indexed block");
        if Self::evictable(n) {
            self.lru.remove(&(n.last, key));
            self.lru.insert((t, key));
        }
        n.last = t;
    }

    /// The indexed pages of `tokens`' longest indexed prefix of whole
    /// blocks, in order, each checked against its tokens and parent;
    /// touched deepest first.
    pub fn walk(&mut self, tokens: &[u32]) -> Vec<(Key, u32)> {
        let mut out: Vec<(Key, u32)> = Vec::new();
        let mut parent = self.root;
        for b in tokens.chunks_exact(PAGE) {
            let key = chain(parent, b);
            match self.map.get(&key) {
                Some(n) if n.parent == parent && *n.tokens == *b => out.push((key, n.page)),
                _ => break,
            }
            parent = key;
        }
        for &(k, _) in out.iter().rev() {
            self.touch(k);
        }
        out
    }

    /// `walk` of a prompt, capped where sharing must stop: whole blocks
    /// below its last position (which admission always feeds again, so the
    /// page holding it is the request's own: specs §2.1 condition 3).
    pub fn walk_prompt(&mut self, prompt: &[u32]) -> Vec<(Key, u32)> {
        let last = prompt.len().saturating_sub(1);
        self.walk(&prompt[..last])
    }

    /// Index `page`, holding the block `tokens` (PAGE of them) after the
    /// prefix `parent` (the root or an indexed block). New blocks start
    /// unheld: the caller `hold`s them for the slot that wrote them.
    pub fn insert(&mut self, parent: Key, tokens: &[u32], page: u32) -> Ins {
        assert_eq!(tokens.len(), PAGE, "a page is PAGE positions");
        if parent != self.root && !self.map.contains_key(&parent) {
            return Ins::No;
        }
        let key = chain(parent, tokens);
        if key == self.root {
            return Ins::No; // a salted root's collision: never a block's key
        }
        if let Some(n) = self.map.get(&key) {
            if n.parent != parent || *n.tokens != *tokens {
                return Ins::No; // a collision: keep what is there
            }
            let have = n.page;
            self.touch(key);
            return Ins::Have(key, have);
        }
        assert!(
            !self.by_page.contains_key(&page),
            "page {page} is indexed already"
        );
        if parent != self.root {
            let p = self.map.get_mut(&parent).expect("checked above");
            if Self::evictable(p) {
                self.lru.remove(&(p.last, parent));
            }
            p.kids.push(key);
        }
        self.tick += 1;
        self.map.insert(
            key,
            Node {
                parent,
                tokens: tokens.into(),
                page,
                kids: Vec::new(),
                holds: 0,
                last: self.tick,
            },
        );
        self.lru.insert((self.tick, key));
        self.by_page.insert(page, key);
        Ins::New(key)
    }

    /// One more slot holds `key`'s page.
    pub fn hold(&mut self, key: Key) {
        let n = self.map.get_mut(&key).expect("an indexed block");
        if Self::evictable(n) {
            self.lru.remove(&(n.last, key));
        }
        n.holds += 1;
    }

    /// A slot let `key`'s page go.
    pub fn release(&mut self, key: Key) {
        let n = self.map.get_mut(&key).expect("an indexed block");
        assert!(n.holds > 0, "release without hold");
        n.holds -= 1;
        if Self::evictable(n) {
            self.lru.insert((n.last, key));
        }
    }

    /// Slots holding `key`'s page.
    pub fn holds(&self, key: Key) -> u32 {
        self.map.get(&key).map_or(0, |n| n.holds)
    }

    /// Drop a block (it has no kids); its parent may become evictable.
    fn remove(&mut self, key: Key) -> Evicted {
        let n = self.map.remove(&key).expect("an indexed block");
        debug_assert!(n.kids.is_empty());
        self.lru.remove(&(n.last, key));
        self.by_page.remove(&n.page);
        if n.parent != self.root {
            let p = self.map.get_mut(&n.parent).expect("parents outlive kids");
            p.kids.retain(|&k| k != key);
            if Self::evictable(p) {
                self.lru.insert((p.last, n.parent));
            }
        }
        Evicted {
            key,
            parent: n.parent,
            tokens: n.tokens,
            page: n.page,
        }
    }

    /// The least recently used unheld leaf, unindexed (the caller unpins
    /// its page); None when every indexed page is held or inner to one.
    pub fn pop_lru_leaf(&mut self) -> Option<Evicted> {
        let &(_, key) = self.lru.iter().next()?;
        Some(self.remove(key))
    }

    /// Unindex `key` and every block extending it, deepest first (the
    /// caller unpins their pages): a slot that alone holds `key`'s page
    /// writes into it next (in-place continuation), and the blocks after
    /// it would name bytes that change. Err (nothing removed) if a slot
    /// other than `key`'s own holder holds one of them: only an unshared
    /// page may be written.
    pub fn unindex_subtree(&mut self, key: Key) -> Result<Vec<Evicted>, String> {
        let n = self.map.get(&key).ok_or("unindex: not indexed")?;
        if n.holds > 1 {
            return Err(format!("unindex: {} slots hold the page", n.holds));
        }
        // Every block under key, parents before kids.
        let mut order = vec![key];
        let mut i = 0;
        while i < order.len() {
            order.extend_from_slice(&self.map[&order[i]].kids);
            i += 1;
        }
        if let Some(k) = order[1..].iter().find(|k| self.map[*k].holds > 0) {
            return Err(format!("unindex: block {k:#x} after it is held"));
        }
        let mut out = Vec::new();
        for &k in order.iter().rev() {
            let n = self.map.get_mut(&k).expect("listed");
            let h = n.holds;
            n.holds = 0; // the own holder's page leaves the index, held
            if h == 0 {
                self.lru.remove(&(n.last, k));
            }
            out.push(self.remove(k));
        }
        Ok(out)
    }

    /// Invariants (tests, debug builds): kids and parents agree, the LRU set
    /// is exactly the unheld leaves, `by_page` the inverse of the pages, and
    /// each block's holds is `holds_of(page)` (the pool's refs).
    #[cfg(any(test, debug_assertions))]
    pub fn check(&self, holds_of: impl Fn(u32) -> u32) {
        let mut kids = 0;
        for (&k, n) in &self.map {
            assert_eq!(chain(n.parent, &n.tokens), k, "block {k:#x}: its key");
            if n.parent != self.root {
                let p = self
                    .map
                    .get(&n.parent)
                    .expect("a block's parent is indexed");
                assert!(p.kids.contains(&k), "block {k:#x}: not its parent's kid");
            }
            for c in &n.kids {
                assert_eq!(self.map[c].parent, k, "kid {c:#x} of {k:#x}");
            }
            kids += n.kids.len();
            assert_eq!(self.by_page.get(&n.page), Some(&k), "page {}", n.page);
            assert_eq!(n.holds, holds_of(n.page), "block {k:#x} page {}", n.page);
            assert_eq!(
                self.lru.contains(&(n.last, k)),
                Self::evictable(n),
                "block {k:#x}: LRU"
            );
        }
        let roots = self.map.values().filter(|n| n.parent == self.root).count();
        assert_eq!(kids + roots, self.map.len());
        assert_eq!(self.by_page.len(), self.map.len());
        assert!(self.lru.iter().all(|(_, k)| self.map.contains_key(k)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::qwen2::pages::PageMap;

    fn toks(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| seed * 1000 + i).collect()
    }

    /// Index every whole page of `t` at pages `first..`; their keys.
    fn put(x: &mut PrefixIndex, t: &[u32], first: u32) -> Vec<Ins> {
        let mut parent = x.root();
        t.chunks_exact(PAGE)
            .enumerate()
            .map(|(i, b)| {
                let r = x.insert(parent, b, first + i as u32);
                parent = match r {
                    Ins::New(k) | Ins::Have(k, _) => k,
                    Ins::No => 0,
                };
                r
            })
            .collect()
    }

    #[test]
    fn walk_is_the_longest_verified_prefix() {
        let mut x = PrefixIndex::default();
        let sys = toks(1, 2 * PAGE);
        let mut a = sys.clone();
        a.extend(toks(2, 2 * PAGE + 10));
        let ia = put(&mut x, &a, 0);
        assert!(ia.iter().all(|r| matches!(r, Ins::New(_))));
        assert_eq!(x.len(), 4);
        let w: Vec<u32> = x.walk(&a).iter().map(|e| e.1).collect();
        assert_eq!(w, [0, 1, 2, 3]);
        // B shares the system prompt only.
        let mut b = sys.clone();
        b.extend(toks(3, PAGE));
        assert_eq!(x.walk(&b).len(), 2);
        // Indexing B's own copy of the shared blocks: Have, at A's pages.
        let ib = put(&mut x, &b, 10);
        assert!(matches!(ib[0], Ins::Have(_, 0)) && matches!(ib[1], Ins::Have(_, 1)));
        assert!(matches!(ib[2], Ins::New(_)));
        assert_eq!(x.len(), 5);
        // A child of an unindexed parent: refused.
        assert_eq!(x.insert(12345, &toks(5, PAGE), 20), Ins::No);
        assert!(x.walk(&toks(9, 3 * PAGE)).is_empty());
        x.check(|_| 0);
    }

    /// The page holding a prompt's last position is never shared: a prompt
    /// of exactly k whole pages shares k - 1, one more token shares k.
    #[test]
    fn walk_prompt_stops_below_the_last_position() {
        let mut x = PrefixIndex::default();
        let a = toks(1, 4 * PAGE);
        put(&mut x, &a, 0);
        assert_eq!(x.walk_prompt(&a).len(), 3, "identical prompt");
        assert_eq!(x.walk_prompt(&a[..3 * PAGE + 1]).len(), 3);
        assert_eq!(x.walk_prompt(&a[..3 * PAGE]).len(), 2);
        assert_eq!(x.walk_prompt(&a[..1]).len(), 0);
        assert_eq!(x.walk_prompt(&[]).len(), 0);
    }

    /// A key collision (same key, other tokens or parent) is a miss, never
    /// an alias: forced by planting a node under a key it doesn't hash to.
    #[test]
    fn a_collision_is_a_miss() {
        let mut x = PrefixIndex::default();
        let a = toks(1, PAGE);
        let b = toks(2, PAGE);
        let kb = chain(ROOT, &b);
        // Plant A's block under B's key.
        x.map.insert(
            kb,
            Node {
                parent: ROOT,
                tokens: a.clone().into(),
                page: 7,
                kids: Vec::new(),
                holds: 0,
                last: 0,
            },
        );
        assert!(x.walk(&b).is_empty(), "B's tokens != the stored ones");
        assert_eq!(x.insert(ROOT, &b, 8), Ins::No);
    }

    #[test]
    fn evicts_lru_unheld_leaves_never_an_inner_or_held_page() {
        let mut x = PrefixIndex::default();
        let a = toks(1, 3 * PAGE);
        let ka: Vec<Key> = put(&mut x, &a, 0)
            .iter()
            .map(|r| match r {
                Ins::New(k) => *k,
                _ => unreachable!(),
            })
            .collect();
        let b = toks(2, PAGE);
        put(&mut x, &b, 3);
        x.hold(ka[2]); // a slot holds A's tail: A's chain stays
        // B's leaf is the only evictable one.
        assert_eq!(x.pop_lru_leaf().map(|e| e.page), Some(3));
        assert_eq!(x.pop_lru_leaf(), None);
        x.release(ka[2]);
        // A goes tail first: 2, 1, 0.
        let order: Vec<u32> = std::iter::from_fn(|| x.pop_lru_leaf().map(|e| e.page)).collect();
        assert_eq!(order, [2, 1, 0]);
        assert!(x.is_empty());
    }

    #[test]
    fn evicted_carries_what_a_ram_tier_needs() {
        let mut x = PrefixIndex::default();
        let a = toks(4, 2 * PAGE);
        put(&mut x, &a, 5);
        let e = x.pop_lru_leaf().unwrap();
        assert_eq!(e.page, 6);
        assert_eq!(&*e.tokens, &a[PAGE..]);
        assert_eq!(e.key, chain(e.parent, &e.tokens));
        assert_eq!(e.parent, chain(ROOT, &a[..PAGE]));
    }

    #[test]
    fn unindex_subtree_frees_warm_descendants_only() {
        let mut x = PrefixIndex::default();
        let a = toks(1, 3 * PAGE);
        let k: Vec<Key> = put(&mut x, &a, 0)
            .iter()
            .map(|r| match r {
                Ins::New(k) => *k,
                _ => unreachable!(),
            })
            .collect();
        // A sibling branch after block 0.
        let mut b = a[..PAGE].to_vec();
        b.extend(toks(9, PAGE));
        put(&mut x, &b, 10);
        // Slot holds blocks 0..=1 (its own, about to rewrite inside 1).
        x.hold(k[0]);
        x.hold(k[1]);
        let gone: Vec<u32> = x
            .unindex_subtree(k[1])
            .unwrap()
            .iter()
            .map(|e| e.page)
            .collect();
        assert_eq!(gone, [2, 1], "deepest first");
        assert_eq!(x.page(k[0]), Some(0));
        assert_eq!(x.len(), 2);
        // Two holders: refused, nothing removed.
        x.hold(k[0]);
        assert_eq!(x.holds(k[0]), 2);
        assert!(x.unindex_subtree(k[0]).is_err());
        assert_eq!(x.len(), 2);
        x.check(|p| if p == 0 { 2 } else { 0 });
    }

    /// The engine's use, modelled on the host (specs §3.3–3.6): slots admit
    /// prompts from a few shared templates (walk, map the indexed pages,
    /// feed the rest), feed and index whole pages, release, get evicted;
    /// the pool and the index in step after every operation, and every
    /// write lands in a page the slot may write. Each page carries the key
    /// of the prefix it was written for ("its bytes"): a page mapped into a
    /// slot always carries the key of that slot's own prefix there, which
    /// is the exactness argument run as a model.
    #[test]
    fn pool_and_index_in_step_fuzz() {
        let (slots, cap, total) = (6usize, 10usize, 40usize);
        for shuffle in [None, Some(5u64)] {
            let mut m = PageMap::new(slots, cap, total, shuffle);
            let mut x = PrefixIndex::default();
            // What each page was last written for: (key of its block).
            let mut bytes: Vec<Option<Key>> = vec![None; total];
            let mut held: Vec<Vec<u32>> = vec![Vec::new(); slots];
            let mut keys: Vec<Vec<Key>> = vec![Vec::new(); slots];
            let mut s = 0x2545_f491_4f6c_dd1du64;
            let mut rnd = |n: usize| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % n as u64) as usize
            };
            let templates: Vec<Vec<u32>> = (0..4).map(|i| toks(i + 1, cap * PAGE)).collect();
            for _ in 0..20000 {
                let sl = rnd(slots);
                match rnd(5) {
                    // Admit a prompt: a template's prefix then private tokens.
                    0 => {
                        for k in keys[sl].drain(..) {
                            x.release(k);
                        }
                        m.release(sl, 0);
                        held[sl].clear();
                        let t = &templates[rnd(templates.len())];
                        let mut prompt = t[..rnd(cap * PAGE / 2)].to_vec();
                        prompt.extend(toks(100 + rnd(1000) as u32, rnd(3 * PAGE)));
                        prompt.truncate(cap * PAGE);
                        if prompt.is_empty() {
                            continue;
                        }
                        let w = x.walk_prompt(&prompt);
                        let pages: Vec<u32> = w.iter().map(|e| e.1).collect();
                        m.map(sl, &pages);
                        for (i, &(k, p)) in w.iter().enumerate() {
                            assert_eq!(bytes[p as usize], Some(k), "mapped page {i}: its bytes");
                            x.hold(k);
                        }
                        keys[sl] = w.iter().map(|e| e.0).collect();
                        held[sl] = prompt[..w.len() * PAGE].to_vec();
                        // Feed the rest (all but the last token), indexing
                        // each page it completes.
                        let rest = prompt[held[sl].len()..prompt.len() - 1].to_vec();
                        let lo = held[sl].len();
                        if m.ensure(sl, lo + rest.len()).is_err() {
                            // Out of pages: evict warm leaves, else give up.
                            while let Some(e) = x.pop_lru_leaf() {
                                m.unpin(e.page);
                            }
                            m.flush();
                            if m.ensure(sl, lo + rest.len()).is_err() {
                                continue;
                            }
                        }
                        m.assert_writable(sl, lo, lo + rest.len());
                        held[sl].extend_from_slice(&rest);
                        let full = held[sl].len() / PAGE;
                        for b in keys[sl].len()..full {
                            let parent = keys[sl].last().copied().unwrap_or(ROOT);
                            let blk = &held[sl][b * PAGE..(b + 1) * PAGE];
                            let page = m.pages(sl)[b];
                            let k = chain(parent, blk);
                            bytes[page as usize] = Some(k); // written now
                            match x.insert(parent, blk, page) {
                                Ins::New(k) => {
                                    m.pin(page);
                                    x.hold(k);
                                    keys[sl].push(k);
                                }
                                // Same bytes elsewhere (the theorem's check).
                                Ins::Have(k, p) => {
                                    assert_eq!(bytes[p as usize], Some(k));
                                    break; // this slot's chain stays unindexed past here
                                }
                                Ins::No => break,
                            }
                        }
                    }
                    // Release from a random point (a cancel / preempt / trim):
                    // round down to whole indexed pages.
                    1 => {
                        let lo = rnd(held[sl].len() + 1);
                        let keep = lo / PAGE;
                        while keys[sl].len() > keep {
                            x.release(keys[sl].pop().unwrap());
                        }
                        m.release(sl, keep * PAGE);
                        held[sl].truncate(keep * PAGE);
                    }
                    // Evict one warm leaf.
                    2 => {
                        if let Some(e) = x.pop_lru_leaf() {
                            assert_eq!(m.refs(e.page), 0, "an evicted page is unheld");
                            m.unpin(e.page);
                        }
                    }
                    // In-place continuation: a slot alone on its last indexed
                    // page unindexes it (and what follows) to write into it.
                    3 => {
                        if let Some(&k) = keys[sl].last() {
                            let p = x.page(k).unwrap();
                            if m.refs(p) == 1 {
                                let gone = x.unindex_subtree(k).unwrap();
                                for e in &gone {
                                    m.unpin(e.page);
                                }
                                keys[sl].pop();
                                let i = keys[sl].len();
                                assert!(m.writable(sl, i), "unindexed and alone: writable");
                                m.assert_writable(sl, i * PAGE, i * PAGE + 1);
                                bytes[p as usize] = None; // about to change
                            } else {
                                assert!(x.unindex_subtree(k).is_err());
                            }
                        }
                    }
                    _ => m.flush(),
                }
                m.check();
                x.check(|p| m.refs(p));
                for p in 0..total as u32 {
                    assert_eq!(
                        x.key_of(p).is_some(),
                        m.pinned(p),
                        "page {p}: pinned == indexed"
                    );
                }
            }
            for (sl, ks) in keys.iter_mut().enumerate() {
                for k in ks.drain(..) {
                    x.release(k);
                }
                m.release(sl, 0);
            }
            while let Some(e) = x.pop_lru_leaf() {
                m.unpin(e.page);
            }
            m.flush();
            assert!(x.is_empty());
            assert_eq!(m.free(), total, "no page leaked");
            m.check();
        }
    }
}
