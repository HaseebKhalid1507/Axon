//! The backend seam: what generation and chat need from a language model.
//!
//! `generate` and `chat::Chat` are generic over `Lm`: the CPU model
//! (`model::Qwen2`), the GPU one (`cuda::qwen2::CudaQwen2`) and the two
//! split across devices (`hybrid::HybridQwen2`) implement the same calls.
//! The CLI holds one in a `Backend`, which dispatches each call to the one
//! inside (one match per token; the CPU's results are the same bits as
//! calling it directly).

use crate::cuda::qwen2::{CacheKv, CudaCache, CudaQwen2};
use crate::cuda::{Gpu, Kernels, Module};
use crate::gguf::Model;
use crate::hybrid::{HybridCache, HybridQwen2, Placement};
use crate::kv::KvType;
use crate::model::{KvCache, Qwen2};
use crate::sample::{self, Cand, CandSpec};

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

    /// Run `tokens[b]` at `start + b` (non-empty) into the cache; the logits
    /// after the last one: `feed_batch` of all but the last (in
    /// `prefill_batch` chunks) then `forward` of the last, bit for bit.
    /// Backends override it to run the last token inside the final batched
    /// pass (the output head on its row alone) instead of a separate step.
    fn prefill_logits(&self, tokens: &[u32], start: usize, cache: &mut Self::Cache) -> Vec<f32> {
        let (&last, rest) = tokens.split_last().expect("prefill_logits: no tokens");
        let pb = self.prefill_batch();
        for (i, b) in rest.chunks(pb).enumerate() {
            self.feed_batch(b, start + i * pb, cache);
        }
        self.forward(last, start + rest.len(), cache)
    }

    /// Run `token` at `pos`; the sampling candidates of its logits,
    /// `sample::cands` (repeat penalty over the tail of `recent`, which ends
    /// with `token`, then the `spec.k` best, best first). Backends that keep
    /// logits off the host (a GPU) override it to select on the device; an
    /// override must give the same candidates for the same logits.
    fn forward_cands(
        &self,
        token: u32,
        pos: usize,
        cache: &mut Self::Cache,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Vec<Cand> {
        let mut l = self.forward(token, pos, cache);
        sample::cands(&mut l, spec, recent)
    }

    /// `forward_cands` on each `tokens[b]` at `start + b` in order
    /// (speculative verify): `recent` ends with `tokens`, and row b's
    /// penalty window ends at `tokens[b]`.
    fn verify_cands(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &mut Self::Cache,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Vec<Vec<Cand>> {
        let (nb, n) = (tokens.len(), recent.len());
        assert!(
            n >= nb && recent[n - nb..] == *tokens,
            "recent must end with tokens"
        );
        (0..nb)
            .map(|b| {
                self.forward_cands(tokens[b], start + b, cache, spec, &recent[..n - nb + b + 1])
            })
            .collect()
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

    /// One batched pass (`verify_batch`, the same logits as `forward`).
    fn verify_cands(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &mut KvCache,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Vec<Vec<Cand>> {
        let (nb, n) = (tokens.len(), recent.len());
        assert!(
            n >= nb && recent[n - nb..] == *tokens,
            "recent must end with tokens"
        );
        self.verify_batch(tokens, start, cache)
            .into_iter()
            .enumerate()
            .map(|(b, mut l)| sample::cands(&mut l, spec, &recent[..n - nb + b + 1]))
            .collect()
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
    /// Split across the CPU and the GPU (leaked as for `Cuda`).
    Hybrid(Box<HybridQwen2<'a, 'static>>),
}

pub enum BackendCache {
    Cpu(KvCache),
    Cuda(Box<CudaCache<'static>>),
    Hybrid(Box<HybridCache<'static>>),
}

impl<'a> Backend<'a> {
    /// The model of `m` on `device`, with room for a `max_ctx`-token cache
    /// of type `kv` (checked up front on the GPU). On CUDA, `gpu_layers`
    /// Some(g) splits the model, the last g layers on the GPU
    /// (`HybridQwen2`, its GPU half's cache of type `kv`, the CPU half's
    /// f16/f32 as AXON_KV says), or with g >= n_layers is the GPU model;
    /// None puts it all on the GPU if it fits, else as many of the last
    /// layers as fit with a max_ctx cache of type `kv`
    /// (`hybrid::auto_gpu_layers`, told on stderr), but with `fit` a
    /// shorter all-GPU context (`ctx`) may win over the split
    /// (`hybrid::choose_placement` says when). Err says why CUDA can't
    /// be used, or that the CPU has no q8 cache. Some(0) is not a split: the
    /// CLI maps it to `Device::Cpu` before loading (a GPU model maps the
    /// file, a CPU one copies it only if it reads matrices in place).
    pub fn new(
        m: &'a Model,
        device: Device,
        max_ctx: usize,
        gpu_layers: Option<usize>,
        kv: KvType,
        fit: bool,
    ) -> Result<Backend<'a>, String> {
        if device == Device::Cpu {
            // The CPU's f16/f32 choice is AXON_KV's (kv::use_f32).
            if kv == KvType::Q8 {
                return Err("the q8 KV cache is cuda-only (--device cuda)".into());
            }
            return Ok(Backend::Cpu(Qwen2::new(m)));
        }
        if gpu_layers == Some(0) {
            return Err("--gpu-layers 0 is --device cpu".into());
        }
        let gpu: &'static Gpu = Box::leak(Box::new(Gpu::open()?));
        let module: &'static Module<'static> = Box::leak(Box::new(Kernels::module(gpu)?));
        // f32 isn't offered on the GPU: f16 there (as always).
        let ckv = match kv {
            KvType::Q8 => CacheKv::Q8 { group: 0 },
            _ => CacheKv::F16,
        };
        let whole_gpu = |ctx: usize, fit: bool| {
            CudaQwen2::with_kv(gpu, module, m, ctx, ckv, fit).map(|n| Backend::Cuda(Box::new(n)))
        };
        let auto = gpu_layers.is_none();
        let stream = crate::hybrid::stream_setting();
        let g = match gpu_layers {
            // Every layer: the GPU model (same weights, cache and VRAM as a
            // split with no CPU half, and its own sampling paths).
            Some(g) => {
                let n = crate::hybrid::block_count(m)?;
                if g >= n {
                    return whole_gpu(max_ctx, false).map_err(|e| {
                        format!("{g}/{n} layers on the GPU: {e}, or fewer --gpu-layers")
                    });
                }
                g
            }
            None => {
                let free = gpu.info()?.free_bytes;
                let whole = CudaQwen2::vram_curve(gpu, module, m, ckv)?;
                // A streaming split's VRAM (its bigger batch scratch and the
                // stream reserve) is what the split option is weighed by.
                let split =
                    || crate::hybrid::auto_gpu_layers(gpu, module, m, max_ctx, ckv, free, stream);
                match crate::hybrid::choose_placement(free, max_ctx, fit, whole, split)? {
                    Placement::Gpu { ctx } => return whole_gpu(ctx, false),
                    Placement::Split {
                        gpu_layers,
                        gpu_ctx,
                    } => {
                        if let Some(c) = gpu_ctx {
                            eprintln!(
                                "axon: the whole model fits in GPU memory only with a {c}-token \
                                 context; splitting it to keep {max_ctx} (--ctx {c} for all of it \
                                 on the GPU, faster; --gpu-layers N to choose the split{})",
                                if ckv == CacheKv::F16 {
                                    "; --kv q8 halves the KV cache"
                                } else {
                                    ""
                                }
                            );
                        }
                        gpu_layers
                    }
                    // with_kv's error says what it needs (or, if memory
                    // was freed meanwhile, it fits).
                    Placement::Nowhere => {
                        return whole_gpu(max_ctx, fit)
                            .map_err(|e| format!("{e}; not even one layer fits"));
                    }
                }
            }
        };
        let h =
            HybridQwen2::new_with_stream(Some((gpu, module)), m, max_ctx, g, ckv, stream, auto)?;
        eprintln!("split:       {}", h.describe(max_ctx));
        Ok(Backend::Hybrid(Box::new(h)))
    }

    /// The context the cache was planned for: max_ctx, or less on the GPU
    /// with `fit` (never for a split: it has the whole max_ctx).
    pub fn ctx(&self, max_ctx: usize) -> usize {
        match self {
            Backend::Cpu(_) | Backend::Hybrid(_) => max_ctx,
            Backend::Cuda(n) => n.ctx,
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
            (Backend::Hybrid($n), BackendCache::Hybrid($c)) => {
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
            Backend::Hybrid(n) => n.vocab(),
        }
    }

    fn new_cache(&self, max_ctx: usize) -> BackendCache {
        match self {
            Backend::Cpu(n) => BackendCache::Cpu(Lm::new_cache(n, max_ctx)),
            Backend::Cuda(n) => BackendCache::Cuda(Box::new(n.new_cache(max_ctx))),
            Backend::Hybrid(n) => BackendCache::Hybrid(Box::new(n.new_cache(max_ctx))),
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

    fn forward_cands(
        &self,
        token: u32,
        pos: usize,
        cache: &mut BackendCache,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Vec<Cand> {
        dispatch!(self, cache, |n, c| n
            .forward_cands(token, pos, c, spec, recent))
    }

    fn verify_cands(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &mut BackendCache,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Vec<Vec<Cand>> {
        dispatch!(self, cache, |n, c| n
            .verify_cands(tokens, start, c, spec, recent))
    }

    fn prefill_logits(&self, tokens: &[u32], start: usize, cache: &mut BackendCache) -> Vec<f32> {
        dispatch!(self, cache, |n, c| n.prefill_logits(tokens, start, c))
    }

    fn prefill_batch(&self) -> usize {
        match self {
            Backend::Cpu(n) => n.prefill_batch(),
            Backend::Cuda(n) => n.prefill_batch(),
            Backend::Hybrid(n) => n.prefill_batch(),
        }
    }
}
