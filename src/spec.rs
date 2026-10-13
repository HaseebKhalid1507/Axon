//! Prompt-lookup drafting for speculative decoding: guess the next tokens by
//! finding the most recent earlier occurrence of the last few tokens in the
//! history (prompt + generated) and proposing what followed it. Pays off when
//! the output copies from the prompt (code edits, RAG, summaries). The guesses
//! are only drafts: the gen loop verifies them against the model's own greedy
//! picks, so output is unchanged whatever is proposed.

use std::collections::HashMap;

/// Longest n-gram matched; drafting tries MAX_N, then shorter, down to MIN_N.
const MAX_N: usize = 3;
/// A single matching token is too weak a hint: in prose the token after a
/// common one ("the", ",") is mostly a different one, and every rejected
/// draft token is extra work in the verify pass.
const MIN_N: usize = 2;
/// Draft length cap when only a shorter-than-MAX_N n-gram matched: such a
/// match predicts the next token or two, rarely a long run (prose at K = 4
/// accepted 2/3 and K = 8 was slower than K = 4).
const SHORT_MATCH_MAX: usize = 3;

pub struct PromptLookup {
    hist: Vec<u32>,
    /// (n, n-gram padded with 0s) -> end (exclusive) of its latest occurrence,
    /// over n-grams ending before the last token. The suffix itself is left out
    /// so a lookup of it finds an earlier occurrence, with a token after it.
    index: HashMap<(usize, [u32; MAX_N]), usize>,
}

impl PromptLookup {
    pub fn new() -> PromptLookup {
        PromptLookup {
            hist: Vec::new(),
            index: HashMap::new(),
        }
    }

    fn key(gram: &[u32]) -> (usize, [u32; MAX_N]) {
        let mut k = [0u32; MAX_N];
        k[..gram.len()].copy_from_slice(gram);
        (gram.len(), k)
    }

    /// Append a token. O(MAX_N): indexes the n-grams ending at the previous
    /// last token, which is no longer the suffix.
    pub fn push(&mut self, t: u32) {
        let end = self.hist.len();
        for n in MIN_N..=MAX_N.min(end) {
            self.index.insert(Self::key(&self.hist[end - n..end]), end);
        }
        self.hist.push(t);
    }

    /// Push the tokens of `h` past ours: catch up with an append-only history.
    pub fn sync(&mut self, h: &[u32]) {
        debug_assert!(h.len() >= self.hist.len() && h[..self.hist.len()] == self.hist[..]);
        for &t in &h[self.hist.len()..] {
            self.push(t);
        }
    }

    /// Up to `max` tokens that followed the most recent earlier occurrence of
    /// the longest (MIN_N..=MAX_N) suffix seen before, at most
    /// SHORT_MATCH_MAX if that suffix is shorter than MAX_N. Empty if none.
    pub fn draft(&self, max: usize) -> Vec<u32> {
        let len = self.hist.len();
        if max == 0 {
            return Vec::new();
        }
        for n in (MIN_N..=MAX_N.min(len)).rev() {
            if let Some(&e) = self.index.get(&Self::key(&self.hist[len - n..])) {
                let max = if n < MAX_N {
                    max.min(SHORT_MATCH_MAX)
                } else {
                    max
                };
                return self.hist[e..len.min(e + max)].to_vec();
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(h: &[u32]) -> PromptLookup {
        let mut d = PromptLookup::new();
        d.sync(h);
        d
    }

    /// Brute force: scan back for the latest earlier occurrence.
    fn reference(h: &[u32], max: usize) -> Vec<u32> {
        let len = h.len();
        if max == 0 {
            return Vec::new();
        }
        for n in (MIN_N..=MAX_N.min(len)).rev() {
            let suf = &h[len - n..];
            for e in (n..len).rev() {
                if &h[e - n..e] == suf {
                    let max = if n < MAX_N {
                        max.min(SHORT_MATCH_MAX)
                    } else {
                        max
                    };
                    return h[e..len.min(e + max)].to_vec();
                }
            }
        }
        Vec::new()
    }

    #[test]
    fn copies_what_followed() {
        let d = lookup(&[1, 2, 3, 4, 5, 6, 9, 2, 3, 4]);
        assert_eq!(d.draft(8), vec![5, 6, 9, 2, 3, 4]);
        assert_eq!(d.draft(2), vec![5, 6]);
        assert_eq!(d.draft(0), Vec::<u32>::new());
    }

    #[test]
    fn prefers_longest_then_latest() {
        // 3-gram [7,8,9] occurred once (-> 1); 2-gram [8,9] occurred later (-> 2).
        let d = lookup(&[7, 8, 9, 1, 0, 8, 9, 2, 0, 7, 8, 9]);
        assert_eq!(d.draft(1), vec![1]);
        // Only 2-gram matches: take the latest one.
        let d = lookup(&[4, 5, 1, 6, 4, 5, 2, 7, 4, 5]);
        assert_eq!(d.draft(3), vec![2, 7, 4]);
    }

    #[test]
    fn short_match_drafts_at_most_3() {
        // Only the 2-gram [4,5] matches: 3 tokens, not 5.
        let d = lookup(&[4, 5, 1, 2, 3, 6, 7, 0, 4, 5]);
        assert_eq!(d.draft(7), vec![1, 2, 3]);
        // A 3-gram match drafts the full K.
        let d = lookup(&[9, 4, 5, 1, 2, 3, 6, 7, 0, 9, 4, 5]);
        assert_eq!(d.draft(7), vec![1, 2, 3, 6, 7, 0, 9]);
    }

    #[test]
    fn single_token_match_is_not_enough() {
        assert!(lookup(&[5, 1, 6, 5, 2, 7, 5]).draft(3).is_empty());
    }

    #[test]
    fn nothing_to_copy() {
        assert!(lookup(&[]).draft(4).is_empty());
        assert!(lookup(&[1]).draft(4).is_empty());
        assert!(lookup(&[1, 2, 3]).draft(4).is_empty());
    }

    #[test]
    fn repeats_overlap() {
        // The suffix's earlier occurrence overlaps it: still a valid draft.
        assert_eq!(lookup(&[4, 4, 4, 4]).draft(5), vec![4]);
        assert_eq!(lookup(&[4, 4, 4]).draft(5), vec![4]);
        assert_eq!(lookup(&[1, 2, 1, 2, 1]).draft(5), vec![2, 1]);
    }

    #[test]
    fn incremental_matches_brute_force() {
        let mut seed = 0x2545_f491u32;
        for alphabet in [2u32, 3, 5, 50] {
            let mut d = PromptLookup::new();
            let mut h = Vec::new();
            for _ in 0..400 {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let t = seed % alphabet;
                h.push(t);
                d.push(t);
                for max in [1, 3, 4, 8] {
                    assert_eq!(d.draft(max), reference(&h, max), "{h:?} max {max}");
                }
            }
        }
    }
}
