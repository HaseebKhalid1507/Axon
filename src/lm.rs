//! The backend seam: what generation and chat need from a language model.
//!
//! `generate` and `chat::Chat` are generic over `Lm` and monomorphized, so
//! the CPU model (`model::Qwen2`) runs exactly the code it ran before the
//! trait existed; a GPU model implements the same five calls.

use crate::model::{KvCache, Qwen2};

/// A causal language model with a KV cache of its own type. Positions are
/// absolute: `pos` is the cache slot the token goes in; tokens before it
/// must already be in the cache.
pub trait Lm {
    /// Keys and values of past positions (on the model's device).
    type Cache;

    /// Logits per position.
    fn vocab(&self) -> usize;

    /// A cache for positions 0..max_ctx.
    fn new_cache(&self, max_ctx: usize) -> Self::Cache;

    /// Run `token` at `pos`; its logits (len `vocab()`).
    fn forward(&self, token: u32, pos: usize, cache: &mut Self::Cache) -> Vec<f32>;

    /// `crate::argmax(&self.forward(..))`: the greedy pick after `token`.
    /// Backends that keep logits off the host (a GPU) override it to return
    /// the index alone instead of copying `vocab()` floats every token; an
    /// override must pick the same index, ties included.
    fn forward_argmax(&self, token: u32, pos: usize, cache: &mut Self::Cache) -> u32 {
        crate::argmax(&self.forward(token, pos, cache))
    }

    /// Run `tokens[b]` at `start + b` for the cache only (prefill).
    fn feed_batch(&self, tokens: &[u32], start: usize, cache: &mut Self::Cache);

    /// The greedy pick after each `tokens[b]` at `start + b`, as
    /// `forward_argmax` on each in order would give (speculative verify).
    fn verify_argmax(&self, tokens: &[u32], start: usize, cache: &mut Self::Cache) -> Vec<u32>;
}

impl Lm for Qwen2<'_> {
    type Cache = KvCache;

    fn vocab(&self) -> usize {
        self.vocab
    }

    fn new_cache(&self, max_ctx: usize) -> KvCache {
        Qwen2::new_cache(self, max_ctx)
    }

    fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        Qwen2::forward(self, token, pos, cache)
    }

    fn feed_batch(&self, tokens: &[u32], start: usize, cache: &mut KvCache) {
        Qwen2::feed_batch(self, tokens, start, cache)
    }

    fn verify_argmax(&self, tokens: &[u32], start: usize, cache: &mut KvCache) -> Vec<u32> {
        Qwen2::verify_argmax(self, tokens, start, cache)
    }
}
