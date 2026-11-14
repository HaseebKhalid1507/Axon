//! A tokenizer's vocabulary as the automaton reads it: each token's bytes
//! (`Tokenizer::decode_bytes`), the special tokens (one symbol each, never
//! bytes), and a trie over the other tokens' bytes, stored in preorder so
//! that a walk can skip a whole subtree (every token with a rejected
//! prefix) by jumping to its end.

use crate::tokenizer::Tokenizer;

/// One vocabulary, built once per loaded model.
pub struct Vocab {
    /// Token ids 0..n: the model's logits (ids past the tokenizer's list
    /// decode to nothing and are never allowed).
    n: usize,
    /// Token id i's bytes: `bytes[off[i]..off[i + 1]]` (a special's text).
    off: Vec<u32>,
    bytes: Vec<u8>,
    /// The special tokens' ids, ascending.
    specials: Vec<u32>,
    is_special: Vec<bool>,
    trie: Trie,
}

/// The byte trie in preorder: node i (the root isn't stored) is reached by
/// byte `byte[i]` at depth `depth[i]` (>= 1); its subtree is i..end[i];
/// the tokens spelled exactly by the path to it are
/// `toks[tok0[i]..tok0[i + 1]]`, so a subtree's tokens are
/// `toks[tok0[i]..tok0[end[i]]]`.
#[derive(Default)]
pub struct Trie {
    pub byte: Vec<u8>,
    pub depth: Vec<u16>,
    pub end: Vec<u32>,
    pub tok0: Vec<u32>,
    pub toks: Vec<u32>,
    /// The deepest node.
    pub max_depth: usize,
}

impl Vocab {
    /// The vocabulary of `tok` for a model with `n` logits.
    pub fn new(tok: &Tokenizer, n: usize) -> Vocab {
        let specials: Vec<u32> = (tok.special_ids().into_iter())
            .filter(|&id| (id as usize) < n)
            .collect();
        let pieces: Vec<Vec<u8>> = (0..n.min(tok.vocab_len()) as u32)
            .map(|id| tok.decode_bytes(&[id]))
            .collect();
        Vocab::from_pieces(n, &pieces, &specials)
    }

    /// From each token's bytes (ids past `pieces` decode to nothing) and
    /// the special ids: the unit tests' synthetic vocabularies.
    pub fn from_pieces(n: usize, pieces: &[Vec<u8>], specials: &[u32]) -> Vocab {
        let mut off = Vec::with_capacity(n + 1);
        let mut bytes = Vec::new();
        off.push(0u32);
        for id in 0..n {
            if let Some(p) = pieces.get(id) {
                bytes.extend_from_slice(p);
            }
            off.push(bytes.len() as u32);
        }
        let mut is_special = vec![false; n];
        let mut sp: Vec<u32> = specials
            .iter()
            .copied()
            .filter(|&id| (id as usize) < n)
            .collect();
        sp.sort_unstable();
        sp.dedup();
        for &id in &sp {
            is_special[id as usize] = true;
        }
        let mut v = Vocab {
            n,
            off,
            bytes,
            specials: sp,
            is_special,
            trie: Trie::default(),
        };
        v.trie = v.build_trie();
        v
    }

    /// Token ids 0..n.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn bytes(&self, id: u32) -> &[u8] {
        let i = id as usize;
        if i >= self.n {
            return &[];
        }
        &self.bytes[self.off[i] as usize..self.off[i + 1] as usize]
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn specials(&self) -> &[u32] {
        &self.specials
    }

    pub fn trie(&self) -> &Trie {
        &self.trie
    }

    /// Whether every byte a JSON text can hold (tab, newline, carriage
    /// return, 0x20..=0xff) is a token by itself, as in byte-level BPE
    /// (Qwen, Llama 3) and SentencePiece with byte fallback: then every
    /// prefix the automaton takes can be finished, and a mask is never
    /// empty (a grammar refuses a vocabulary without it: `Grammar::hermes`).
    pub fn byte_complete(&self) -> bool {
        let mut have = [false; 256];
        for id in 0..self.n as u32 {
            if let [b] = self.bytes(id)
                && !self.is_special(id)
            {
                have[*b as usize] = true;
            }
        }
        (have.iter().enumerate()).all(|(b, &h)| h || (b < 0x20 && !matches!(b, 0x09 | 0x0a | 0x0d)))
    }

    /// The special token whose text is `text`, if there is one.
    pub fn special_of(&self, text: &[u8]) -> Option<u32> {
        (self.specials.iter().copied()).find(|&id| self.bytes(id) == text)
    }

    /// The trie of every token that isn't special and isn't empty.
    fn build_trie(&self) -> Trie {
        let mut ids: Vec<u32> = (0..self.n as u32)
            .filter(|&id| !self.is_special(id) && !self.bytes(id).is_empty())
            .collect();
        ids.sort_by(|&a, &b| self.bytes(a).cmp(self.bytes(b)).then(a.cmp(&b)));
        let mut t = Trie::default();
        // The path to the current node: node index per depth (1-based).
        let mut path: Vec<u32> = Vec::new();
        let mut prev: &[u8] = &[];
        for &id in &ids {
            let s = self.bytes(id);
            let lcp = prev.iter().zip(s).take_while(|(a, b)| a == b).count();
            // Close the nodes past the common prefix.
            while path.len() > lcp {
                let i = path.pop().unwrap() as usize;
                t.end[i] = t.byte.len() as u32;
            }
            for (d, &b) in s.iter().enumerate().skip(lcp) {
                t.byte.push(b);
                t.depth.push((d + 1) as u16);
                t.end.push(0);
                t.tok0.push(t.toks.len() as u32);
                path.push((t.byte.len() - 1) as u32);
            }
            t.toks.push(id);
            t.max_depth = t.max_depth.max(s.len());
            prev = s;
        }
        while let Some(i) = path.pop() {
            t.end[i as usize] = t.byte.len() as u32;
        }
        t.tok0.push(t.toks.len() as u32);
        t
    }
}

impl Trie {
    pub fn len(&self) -> usize {
        self.byte.len()
    }

    /// The tokens spelled by the path to node i.
    pub fn toks_at(&self, i: usize) -> &[u32] {
        &self.toks[self.tok0[i] as usize..self.tok0[i + 1] as usize]
    }

    /// Every token in node i's subtree.
    pub fn toks_under(&self, i: usize) -> &[u32] {
        let e = self.end[i] as usize;
        let hi = if e < self.len() {
            self.tok0[e]
        } else {
            self.toks.len() as u32
        };
        &self.toks[self.tok0[i] as usize..hi as usize]
    }
}
