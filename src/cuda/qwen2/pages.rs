//! Paged KV's allocator (specs/r11-paged-kv.md §2.3): which physical
//! pages of a pool's cache hold each slot's positions. Host only; the pool
//! (slots.rs) mirrors a slot's list into its row of the device table.
//!
//! A page is `PAGE` consecutive positions of every layer's K and V (and
//! q8 scales): rows 64p..64p + 64 of each array. Slot s's position t is
//! at row `pages[s][t / 64] * 64 + t % 64`. Pages move between three
//! places, and every page is in exactly one of them: the free list, the
//! retired list (released while a step may still read them: free again
//! after it, `flush`), and one slot's list. A slot's list only grows at
//! its end (`ensure`) and shrinks from its end (`release`), so the pages
//! under positions a kernel reads never change while it runs.

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
            self.pages[slot].push(p);
        }
        Ok(have..want)
    }

    /// Slot `slot` keeps the pages of positions 0..lo; the rest retire
    /// (free after the step in flight: `flush`).
    pub fn release(&mut self, slot: usize, lo: usize) {
        let keep = Self::need(lo);
        let l = &mut self.pages[slot];
        if keep < l.len() {
            self.retire.extend(l.drain(keep..).rev());
        }
    }

    /// No step is in flight: retired pages are free.
    pub fn flush(&mut self) {
        self.free.append(&mut self.retire);
    }

    /// Whether retired pages wait for a `flush`.
    pub fn retired(&self) -> bool {
        !self.retire.is_empty()
    }

    /// Every page in exactly one place, no slot past its cap.
    #[cfg(any(test, debug_assertions))]
    pub fn check(&self) {
        let mut seen = vec![false; self.total];
        let all = self
            .pages
            .iter()
            .flatten()
            .chain(&self.free)
            .chain(&self.retire);
        for &p in all {
            assert!((p as usize) < self.total, "page {p} of {}", self.total);
            assert!(!seen[p as usize], "page {p} twice");
            seen[p as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "a page lost");
        assert!(self.pages.iter().all(|l| l.len() <= self.cap));
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
