//! Paged KV's allocator (specs/r11-paged-kv.md §2.3): which physical
//! pages of a pool's cache hold each slot's positions. Host only; the pool
//! (slots.rs) mirrors a slot's list into its row of the device table.
//!
//! A page is `PAGE` consecutive positions of every layer's K and V (and
//! q8 scales): rows 64p..64p + 64 of each array. Slot s's position t is
//! at row `pages[s][t / 64] * 64 + t % 64`. A slot's list only grows at
//! its end (`ensure`, `map`) and shrinks from its end (`release`), so the
//! pages under positions a kernel reads never change while it runs.
//!
//! Shared prefixes (specs/shared-prefix-pages.md): a page may be in more
//! than one slot's list (`map`: another slot's page of the same prefix
//! tokens, whose bytes are the ones this slot would write) and may be
//! pinned by the prefix index (serve/prefix.rs). Every page is in exactly
//! one state: free, retired (released while a step may still read it:
//! free after it, `flush`), or live: `refs` (its occurrences in slot
//! lists) > 0, or pinned (with refs 0: "warm", cached for a later prompt).
//! **A live page is written only while it is private**: refs 1 and not
//! pinned (`writable`). A shared or indexed page is immutable: its bytes
//! are a prefix's, and another reader relies on them. Without sharing
//! (nothing mapped or pinned) refs is always 1 and this is the allocator
//! before it, decision for decision.

use super::ATT_BLK;
use std::ops::Range;

/// Positions per page (decode.cu PG_ROWS, kvsave::PAGE).
pub const PAGE: usize = ATT_BLK;

/// Pages short (`PageMap::ensure`): nothing was taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoPages {
    pub short: usize,
}

pub struct PageMap {
    /// Pages a slot may hold (its positions cap / PAGE, rounded up).
    cap: usize,
    pages: Vec<Vec<u32>>,
    free: Vec<u32>,
    retire: Vec<u32>,
    total: usize,
    /// AXON_PAGE_SHUFFLE: pop a seeded random free page (xorshift state),
    /// not the last one.
    rng: Option<u64>,
    /// Occurrences of each page in slot lists.
    refs: Vec<u32>,
    /// Pages the prefix index holds (kept with refs 0: warm).
    pinned: Vec<bool>,
}

impl PageMap {
    /// `total` pages (ids 0..total) for `slots` slots of at most `cap`
    /// pages each, all free; `shuffle`: pop them in a seeded random order.
    pub fn new(slots: usize, cap: usize, total: usize, shuffle: Option<u64>) -> PageMap {
        assert!(total <= u32::MAX as usize);
        PageMap {
            cap,
            pages: vec![Vec::new(); slots],
            // LIFO: page 0 is popped first.
            free: (0..total as u32).rev().collect(),
            retire: Vec::new(),
            total,
            rng: shuffle.map(|s| s | 1),
            refs: vec![0; total],
            pinned: vec![false; total],
        }
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Pages free now or after the step in flight (`flush`).
    pub fn free(&self) -> usize {
        self.free.len() + self.retire.len()
    }

    /// Slot `slot`'s pages, in position order.
    pub fn pages(&self, slot: usize) -> &[u32] {
        &self.pages[slot]
    }

    /// Pages positions 0..n take.
    pub fn need(n: usize) -> usize {
        n.div_ceil(PAGE)
    }

    /// Pages for slot `slot`'s positions 0..hi: the entries it gained (to
    /// copy into the device table), or how many pages are short (then
    /// nothing was taken). Past the cap: `NoPages` with short 0 is never
    /// returned; the caller checks hi against the cap (a panic here).
    pub fn ensure(&mut self, slot: usize, hi: usize) -> Result<Range<usize>, NoPages> {
        let want = Self::need(hi);
        assert!(want <= self.cap, "{hi} positions: past the slot's cap");
        let have = self.pages[slot].len();
        if want <= have {
            return Ok(have..have);
        }
        let n = want - have;
        if n > self.free.len() {
            // Retired pages are free once the step in flight is done: the
            // caller flushes first.
            return Err(NoPages {
                short: n - self.free.len(),
            });
        }
        for _ in 0..n {
            let i = match self.rng.as_mut() {
                None => self.free.len() - 1,
                Some(s) => {
                    *s ^= *s << 13;
                    *s ^= *s >> 7;
                    *s ^= *s << 17;
                    (*s % self.free.len() as u64) as usize
                }
            };
            let p = self.free.swap_remove(i);
            debug_assert!(self.refs[p as usize] == 0 && !self.pinned[p as usize]);
            self.refs[p as usize] = 1;
            self.pages[slot].push(p);
        }
        Ok(have..want)
    }

    /// Slot `slot` keeps the pages of positions 0..lo; the rest retire
    /// (free after the step in flight: `flush`).
    ///
    /// Shared: a page leaves the slot's list (its refs drop) and retires
    /// only once no slot holds it and the index doesn't pin it.
    pub fn release(&mut self, slot: usize, lo: usize) {
        let keep = Self::need(lo);
        let l = &mut self.pages[slot];
        if keep < l.len() {
            for p in l.drain(keep..).rev() {
                let r = &mut self.refs[p as usize];
                *r -= 1;
                if *r == 0 && !self.pinned[p as usize] {
                    self.retire.push(p);
                }
            }
        }
    }

    /// Append `pages` (live: held by a slot or pinned) to slot `slot`'s
    /// list, each one more reference: the slot's positions from
    /// `pages(slot).len() * PAGE` on are theirs, read only (`writable`).
    /// The entries it gained (to copy into the device table).
    pub fn map(&mut self, slot: usize, pages: &[u32]) -> Range<usize> {
        let have = self.pages[slot].len();
        assert!(have + pages.len() <= self.cap, "map: past the slot's cap");
        for &p in pages {
            let i = p as usize;
            assert!(i < self.total, "map: page {p} of {}", self.total);
            assert!(
                self.refs[i] > 0 || self.pinned[i],
                "map: page {p} isn't live (free or retired)"
            );
            self.refs[i] += 1;
        }
        self.pages[slot].extend_from_slice(pages);
        have..have + pages.len()
    }

    /// The prefix index holds page `p` (live, not pinned yet): kept, with
    /// its bytes, after the last slot releases it.
    pub fn pin(&mut self, p: u32) {
        let i = p as usize;
        assert!(self.refs[i] > 0, "pin: page {p} isn't in a slot");
        assert!(!self.pinned[i], "pin: page {p} is pinned already");
        self.pinned[i] = true;
    }

    /// The index lets page `p` go: it retires if no slot holds it.
    pub fn unpin(&mut self, p: u32) {
        let i = p as usize;
        assert!(self.pinned[i], "unpin: page {p} isn't pinned");
        self.pinned[i] = false;
        if self.refs[i] == 0 {
            self.retire.push(p);
        }
    }

    /// Whether the index pins page `p`.
    pub fn pinned(&self, p: u32) -> bool {
        self.pinned[p as usize]
    }

    /// Slots holding page `p`.
    pub fn refs(&self, p: u32) -> u32 {
        self.refs[p as usize]
    }

    /// Whether slot `slot` may write its `i`th page: it alone holds it and
    /// the index doesn't (a shared or indexed page is immutable).
    pub fn writable(&self, slot: usize, i: usize) -> bool {
        let p = self.pages[slot][i] as usize;
        self.refs[p] == 1 && !self.pinned[p]
    }

    /// Debug builds: a write of slot `slot`'s positions lo..hi touches only
    /// pages it may write (positions past its list get fresh pages).
    #[cfg(any(test, debug_assertions))]
    pub fn assert_writable(&self, slot: usize, lo: usize, hi: usize) {
        let held = self.pages[slot].len();
        for i in lo / PAGE..Self::need(hi).min(held) {
            assert!(
                self.writable(slot, i),
                "slot {slot} writes positions {lo}..{hi} into its page {i} (page {}): shared \
                 or indexed, read only",
                self.pages[slot][i]
            );
        }
    }

    /// Pinned pages no slot holds (warm: the index's to evict).
    pub fn warm(&self) -> usize {
        (0..self.total)
            .filter(|&i| self.pinned[i] && self.refs[i] == 0)
            .count()
    }

    /// Pages slot `slot` alone holds (what releasing it gives up, to the
    /// free list or, pinned, to the warm set).
    pub fn excl(&self, slot: usize) -> usize {
        self.pages[slot]
            .iter()
            .filter(|&&p| self.refs[p as usize] == 1)
            .count()
    }

    /// No step is in flight: retired pages are free.
    pub fn flush(&mut self) {
        self.free.append(&mut self.retire);
    }

    /// Whether retired pages wait for a `flush`.
    pub fn retired(&self) -> bool {
        !self.retire.is_empty()
    }

    /// Every page in exactly one state (free, retired, live), each page's
    /// refs its occurrences in slot lists, a live page held or pinned, no
    /// slot past its cap.
    #[cfg(any(test, debug_assertions))]
    pub fn check(&self) {
        let mut occ = vec![0u32; self.total];
        for &p in self.pages.iter().flatten() {
            assert!((p as usize) < self.total, "page {p} of {}", self.total);
            occ[p as usize] += 1;
        }
        let mut idle = vec![false; self.total];
        for &p in self.free.iter().chain(&self.retire) {
            assert!((p as usize) < self.total, "page {p} of {}", self.total);
            assert!(!idle[p as usize], "page {p} twice in free / retired");
            idle[p as usize] = true;
        }
        for i in 0..self.total {
            assert_eq!(occ[i], self.refs[i], "page {i}: refs vs slot lists");
            let live = occ[i] > 0 || self.pinned[i];
            assert!(
                live != idle[i],
                "page {i}: live {live}, free/retired {}",
                idle[i]
            );
        }
        for (s, l) in self.pages.iter().enumerate() {
            assert!(l.len() <= self.cap, "slot {s} past its cap");
            let mut seen = std::collections::HashSet::new();
            assert!(
                l.iter().all(|p| seen.insert(*p)),
                "slot {s} holds a page twice"
            );
        }
    }
}

/// AXON_PAGE_SHUFFLE=seed (tests: scattered pages for free): the seed.
pub fn shuffle_env() -> Option<u64> {
    let v = std::env::var("AXON_PAGE_SHUFFLE").ok()?;
    let v = v.trim();
    if v.is_empty() || v == "0" {
        return None;
    }
    Some(v.parse().unwrap_or_else(|_| {
        v.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
        })
    }))
}

/// AXON_PAGED_KV=0: pools are plain (every slot its reserved ctx rows, the
/// layout before paging), else paged.
pub fn paged_env() -> bool {
    std::env::var("AXON_PAGED_KV").map_or(true, |v| v.trim() != "0")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocator fuzz (spec §3.1.4): random grow / truncate / release /
    /// flush against the invariants, LIFO and shuffled; a failed ensure
    /// takes nothing; freed pages come back.
    #[test]
    fn pagemap_fuzz() {
        for shuffle in [None, Some(7u64)] {
            let (slots, cap, total) = (6, 9, 30);
            let mut m = PageMap::new(slots, cap, total, shuffle);
            let mut len = vec![0usize; slots];
            let mut s = 0x2545_f491_4f6c_dd1du64;
            let mut rnd = |n: usize| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % n as u64) as usize
            };
            for _ in 0..20000 {
                let slot = rnd(slots);
                match rnd(4) {
                    0 | 1 => {
                        let hi = (len[slot] + rnd(3 * PAGE)).min(cap * PAGE);
                        let before = (m.pages(slot).to_vec(), m.free.len());
                        match m.ensure(slot, hi) {
                            Ok(r) => {
                                assert_eq!(r.end, m.pages(slot).len());
                                assert!(m.pages(slot).len() >= PageMap::need(hi));
                                len[slot] = len[slot].max(hi);
                            }
                            Err(NoPages { short }) => {
                                assert!(short > 0);
                                assert_eq!((m.pages(slot).to_vec(), m.free.len()), before);
                            }
                        }
                    }
                    2 => {
                        let lo = rnd(len[slot] + 1);
                        m.release(slot, lo);
                        len[slot] = lo;
                        assert_eq!(m.pages(slot).len(), PageMap::need(lo));
                    }
                    _ => m.flush(),
                }
                m.check();
            }
            for slot in 0..slots {
                m.release(slot, 0);
            }
            m.flush();
            assert_eq!(m.free(), total);
            m.check();
        }
    }

    /// Shared pages (specs/shared-prefix-pages.md): random ensure / map
    /// (another slot's live pages, or warm ones) / release / pin / unpin /
    /// flush against `check` (refs == occurrences, one state per page),
    /// LIFO and shuffled; a page is writable exactly when one slot holds it
    /// and it isn't pinned; releasing everything and unpinning gives every
    /// page back.
    #[test]
    fn pagemap_share_fuzz() {
        for shuffle in [None, Some(11u64)] {
            let (slots, cap, total) = (5, 8, 26);
            let mut m = PageMap::new(slots, cap, total, shuffle);
            let mut s = 0x9e37_79b9_7f4a_7c15u64;
            let mut rnd = |n: usize| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % n as u64) as usize
            };
            for _ in 0..30000 {
                let slot = rnd(slots);
                let held = m.pages(slot).len();
                match rnd(7) {
                    0 | 1 => {
                        let free = m.free.len();
                        let want = (held + rnd(3)).min(cap);
                        let r = m.ensure(slot, want * PAGE);
                        assert_eq!(r.is_ok(), want - held.min(want) <= free);
                    }
                    2 => {
                        // Another slot's prefix, or a warm page.
                        let other = rnd(slots);
                        let live: Vec<u32> = if other != slot {
                            m.pages(other)
                                .iter()
                                .copied()
                                .filter(|p| !m.pages(slot).contains(p))
                                .collect()
                        } else {
                            (0..total as u32)
                                .filter(|&p| m.pinned(p) && m.refs(p) == 0)
                                .collect()
                        };
                        let n = rnd(live.len() + 1).min(cap - held);
                        if n > 0 {
                            let r = m.map(slot, &live[..n]);
                            assert_eq!(r, held..held + n);
                            for i in r {
                                assert!(!m.writable(slot, i), "a mapped page is shared or pinned");
                            }
                        }
                    }
                    3 => m.release(slot, rnd(held * PAGE + 1)),
                    4 => {
                        if held > 0 {
                            let p = m.pages(slot)[rnd(held)];
                            if !m.pinned(p) {
                                m.pin(p);
                            }
                        }
                    }
                    5 => {
                        let pins: Vec<u32> = (0..total as u32).filter(|&p| m.pinned(p)).collect();
                        if !pins.is_empty() {
                            m.unpin(pins[rnd(pins.len())]);
                        }
                    }
                    _ => m.flush(),
                }
                m.check();
                for sl in 0..slots {
                    for (i, &p) in m.pages(sl).iter().enumerate() {
                        assert_eq!(m.writable(sl, i), m.refs(p) == 1 && !m.pinned(p));
                    }
                    let excl = m.pages(sl).iter().filter(|&&p| m.refs(p) == 1).count();
                    assert_eq!(m.excl(sl), excl);
                }
            }
            for slot in 0..slots {
                m.release(slot, 0);
            }
            let warm = m.warm();
            for p in 0..total as u32 {
                if m.pinned(p) {
                    m.unpin(p);
                }
            }
            m.flush();
            assert!(warm <= total);
            assert_eq!(m.free(), total);
            m.check();
        }
    }

    /// A pinned page outlives its last slot (warm) and can be mapped again;
    /// a shared page frees only with its last holder.
    #[test]
    fn pagemap_refs_and_pins() {
        let mut m = PageMap::new(3, 4, 6, None);
        m.ensure(0, 2 * PAGE + 5).unwrap();
        let a = m.pages(0).to_vec();
        assert_eq!(a, [0, 1, 2]);
        assert!(m.writable(0, 0));
        m.map(1, &a[..2]);
        assert_eq!((m.refs(0), m.refs(2)), (2, 1));
        assert!(!m.writable(0, 0) && !m.writable(1, 1) && m.writable(0, 2));
        assert_eq!((m.excl(0), m.excl(1)), (1, 0));
        m.pin(0);
        m.release(0, 0);
        m.flush();
        // Page 2 was slot 0's alone: free; 0 and 1 still slot 1's.
        assert_eq!(m.free(), 4);
        m.release(1, 0);
        m.flush();
        // Page 0 pinned: warm, not free; page 1 free.
        assert_eq!((m.free(), m.warm()), (5, 1));
        m.map(2, &[0]);
        assert_eq!(m.refs(0), 1);
        assert!(!m.writable(2, 0), "a pinned page is read only");
        m.check();
        m.unpin(0);
        assert!(m.writable(2, 0), "unpinned, slot 2's alone");
        m.release(2, 0);
        m.flush();
        assert_eq!(m.free(), 6);
        m.check();
    }

    /// E3: a write into a shared page is a debug panic, not a corruption.
    #[test]
    #[should_panic(expected = "read only")]
    fn pagemap_write_into_shared_page_panics() {
        let mut m = PageMap::new(2, 4, 8, None);
        m.ensure(0, 2 * PAGE).unwrap();
        let a = m.pages(0).to_vec();
        m.map(1, &a);
        // Slot 1 continues past its prefix: fine.
        m.assert_writable(1, 2 * PAGE, 3 * PAGE);
        // Slot 0 rewrites position 70 of the page it shares: refused.
        m.assert_writable(0, 70, 80);
    }

    /// map refuses a page that isn't live.
    #[test]
    #[should_panic(expected = "isn't live")]
    fn pagemap_map_refuses_a_free_page() {
        let mut m = PageMap::new(2, 4, 8, None);
        m.map(0, &[3]);
    }

    /// Shuffled pops differ from LIFO, LIFO pops page 0 first.
    #[test]
    fn pagemap_orders() {
        let mut a = PageMap::new(1, 8, 8, None);
        a.ensure(0, 8 * PAGE).unwrap();
        assert_eq!(a.pages(0), &[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut b = PageMap::new(1, 8, 8, Some(3));
        b.ensure(0, 8 * PAGE).unwrap();
        assert_ne!(b.pages(0), a.pages(0));
        b.check();
    }
}
