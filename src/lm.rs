//! The backend seam: what generation and chat need from a language model.
//!
//! `generate` and `chat::Chat` are generic over `Lm`: the CPU model
//! (`model::Qwen2`) and the GPU one (`cuda::qwen2::CudaQwen2`) implement the
//! same five calls. The CLI holds either in a `Backend`, which dispatches
//! each call to the one inside (one match per token; the CPU's results are
//! the same bits as calling it directly).

use crate::cuda::qwen2::{CudaCache, CudaQwen2};
use crate::cuda::{Gpu, Kernels, Module};
use crate::gguf::Model;
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

    /// Prompt tokens per `feed_batch` call that suit this backend (callers
    /// may pass any size: results don't depend on the batching).
    fn prefill_batch(&self) -> usize {
        crate::PREFILL_BATCH
    }
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

/// Where `axon gen` / `axon chat` run the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    /// "cpu" or "cuda".
    pub fn parse(s: &str) -> Result<Device, String> {
        match s {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Ok(Device::Cuda),
            _ => Err(format!("unknown device '{s}' (devices: cpu, cuda)")),
        }
    }

    /// $AXON_DEVICE, else the CPU.
    pub fn from_env() -> Result<Device, String> {
        match std::env::var("AXON_DEVICE") {
            Ok(v) => Device::parse(&v).map_err(|e| format!("AXON_DEVICE: {e}")),
            Err(_) => Ok(Device::Cpu),
        }
    }
}

/// The model on the device the command line picked: the CLI's one `Lm`.
/// Each call is a match away from the backend's own.
pub enum Backend<'a> {
    Cpu(Qwen2<'a>),
    /// Its `Gpu` and kernel module are leaked: they live as long as the
    /// process, which is as long as the CLI needs them (the weights and
    /// caches are still freed on drop).
    Cuda(Box<CudaQwen2<'static>>),
}

pub enum BackendCache {
    Cpu(KvCache),
    Cuda(Box<CudaCache<'static>>),
}

impl<'a> Backend<'a> {
    /// The model of `m` on `device`, with room for a `max_ctx`-token cache
    /// (checked up front on the GPU). Err says why CUDA can't be used.
    pub fn new(m: &'a Model, device: Device, max_ctx: usize) -> Result<Backend<'a>, String> {
        match device {
            Device::Cpu => Ok(Backend::Cpu(Qwen2::new(m))),
            Device::Cuda => {
                let gpu: &'static Gpu = Box::leak(Box::new(Gpu::open()?));
                let module: &'static Module<'static> = Box::leak(Box::new(Kernels::module(gpu)?));
                Ok(Backend::Cuda(Box::new(CudaQwen2::new(
                    gpu, module, m, max_ctx,
                )?)))
            }
        }
    }
}

macro_rules! dispatch {
    ($self:expr, $cache:expr, |$n:ident, $c:ident| $body:expr) => {
        match ($self, $cache) {
            (Backend::Cpu($n), BackendCache::Cpu($c)) => $body,
            (Backend::Cuda($n), BackendCache::Cuda($c)) => {
                let ($n, $c) = (&**$n, &mut **$c);
                $body
            }
            _ => panic!("a cache of another backend"),
        }
    };
}

impl Lm for Backend<'_> {
    type Cache = BackendCache;

    fn vocab(&self) -> usize {
        match self {
            Backend::Cpu(n) => n.vocab(),
            Backend::Cuda(n) => n.vocab(),
        }
    }

    fn new_cache(&self, max_ctx: usize) -> BackendCache {
        match self {
            Backend::Cpu(n) => BackendCache::Cpu(Lm::new_cache(n, max_ctx)),
            Backend::Cuda(n) => BackendCache::Cuda(Box::new(n.new_cache(max_ctx))),
        }
    }

    fn forward(&self, token: u32, pos: usize, cache: &mut BackendCache) -> Vec<f32> {
        dispatch!(self, cache, |n, c| Lm::forward(n, token, pos, c))
    }

    fn forward_argmax(&self, token: u32, pos: usize, cache: &mut BackendCache) -> u32 {
        dispatch!(self, cache, |n, c| n.forward_argmax(token, pos, c))
    }

    fn feed_batch(&self, tokens: &[u32], start: usize, cache: &mut BackendCache) {
        dispatch!(self, cache, |n, c| Lm::feed_batch(n, tokens, start, c))
    }

    fn verify_argmax(&self, tokens: &[u32], start: usize, cache: &mut BackendCache) -> Vec<u32> {
        dispatch!(self, cache, |n, c| Lm::verify_argmax(n, tokens, start, c))
    }

    fn prefill_batch(&self) -> usize {
        match self {
            Backend::Cpu(n) => n.prefill_batch(),
            Backend::Cuda(n) => n.prefill_batch(),
        }
    }
}
