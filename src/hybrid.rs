//! A model split across devices, like llama.cpp's `-ngl`: the CPU runs the
//! embedding and the first k layers, the GPU the last n - k layers, the
//! output norm, the head and the argmax. For models bigger than VRAM.
//!
//! Per token: CPU layers (`Qwen2::hidden`, the pool), one host-to-device
//! copy of the hidden state (dim floats), GPU layers and head
//! (`CudaQwen2::forward_hidden`): 4 bytes back for a greedy pick, the
//! logits for `forward`. Batches (prefill, speculative verify) take the CPU
//! batched path (`hidden_batch`) over the whole batch, then G3's batched
//! pass from those states. Each layer's KV cache lives with its layer.
//!
//! Numerics: each half is its backend's own code, so k = 0 gives the GPU
//! model's bits and k = n the CPU's; in between, the CPU's bits up to layer
//! k and the GPU's numerics after it (graded vs the CPU in the tests, as
//! the GPU is). Both halves give every token the same bits whatever the
//! batching, so the hybrid does too: chat's cache reuse equals a fresh
//! prefill and drafting is lossless.
//!
//! Threads: strictly sequential per token. The thread that opened the GPU
//! (`Gpu` is !Send) runs everything: it drives the CPU layers as the pool's
//! caller (AXON_THREADS workers take the chunks), then launches the GPU
//! half and waits for it. The workers spin ~200 us after their last chunk,
//! then park, so they don't compete with the driver thread while the GPU
//! runs; waking them costs ~50 us once per token. Overlapping the halves is
//! left for later (PR A streams the CPU layers' weights instead).

use crate::cuda::qwen2::{CudaCache, CudaQwen2};
use crate::cuda::{Gpu, Module};
use crate::gguf::{Model, Tensor};
use crate::lm::Lm;
use crate::model::{KvCache, Qwen2};

/// VRAM the automatic split leaves free beyond what `CudaQwen2` plans for
/// (which keeps 256 MiB for the context's own growth): the desktop's
/// compositor and other programs share the GPU.
///
/// Asymmetric on purpose: the all-GPU plan (`CudaQwen2::new`, tried first)
/// needs only its `HEADROOM`, as it always has; a split, which only happens
/// when the GPU is already full, keeps `DESKTOP_MARGIN` on top so the
/// desktop isn't starved. So a model that just misses the all-GPU plan
/// gets ~2 layers (32B) fewer than `HEADROOM` alone would allow. An
/// explicit --gpu-layers is checked against `HEADROOM` only.
pub const DESKTOP_MARGIN: u64 = 640 << 20;

const GIB: f64 = (1u64 << 30) as f64;

pub struct HybridQwen2<'m, 'g> {
    /// Embedding and layers 0..k (and the head if k == n); None if k == 0.
    cpu: Option<Qwen2<'m>>,
    /// Layers k..n and the head (and the embedding if k == 0); None if k == n.
    gpu: Option<CudaQwen2<'g>>,
    /// Layers on the GPU, of `n_layers`.
    pub gpu_layers: usize,
    pub n_layers: usize,
}

/// Each half's cache: KV of layers 0..k on the CPU, k..n on the GPU.
pub struct HybridCache<'g> {
    cpu: Option<KvCache>,
    gpu: Option<CudaCache<'g>>,
}

/// The model's layer count.
fn block_count(m: &Model) -> Result<usize, String> {
    crate::model::meta_num(m, &format!("{}.block_count", m.arch))
        .map(|v| v as usize)
        .ok_or_else(|| "missing metadata block_count".to_string())
}

/// Most layers the GPU can take (the last ones, plus the head) with room
/// for a `max_ctx`-token cache and `DESKTOP_MARGIN` to spare out of `free`
/// bytes of VRAM (the caller's `gpu.info()?.free_bytes`): 0 if not even
/// one fits. The VRAM a split needs only shrinks as layers move to the CPU,
/// so a binary search over `CudaQwen2::vram_need` finds it.
pub fn auto_gpu_layers(
    gpu: &Gpu,
    module: &Module,
    m: &Model,
    max_ctx: usize,
    free: u64,
) -> Result<usize, String> {
    let n = block_count(m)?;
    let fits = |g: usize| -> Result<bool, String> {
        let need = CudaQwen2::vram_need(gpu, module, m, max_ctx, n - g)?;
        Ok(need + DESKTOP_MARGIN <= free)
    };
    let (mut lo, mut hi) = (0, n); // fits(lo) (no GPU layers); find the largest
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(mid)? {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Ok(lo)
}

impl<'m, 'g> HybridQwen2<'m, 'g> {
    /// The model with its last `gpu_layers` layers (clamped to n) and head
    /// on the GPU (`dev`, needed unless gpu_layers is 0), the rest on the
    /// CPU (repacked unless AXON_REPACK=0, as `Qwen2::new`), with room for
    /// a `max_ctx`-token cache. Each half loads only its own weights, and
    /// the GPU's are handed back to `m` once uploaded (`Model::release`;
    /// on a copy of the file they can't be read again).
    pub fn new(
        dev: Option<(&'g Gpu, &'g Module<'g>)>,
        m: &'m Model,
        max_ctx: usize,
        gpu_layers: usize,
    ) -> Result<HybridQwen2<'m, 'g>, String> {
        let n = block_count(m)?;
        let g = gpu_layers.min(n);
        let k = n - g;
        // The GPU first: a VRAM shortfall fails before the CPU's repack.
        // Each matrix's pages go back right after its upload, so the GPU
        // half (~9 GB on 32B) never sits in the page cache all at once; the
        // norms and biases (read into `plan`) after.
        let gpu = match (g, dev) {
            (0, _) => None,
            (_, None) => return Err("GPU layers without a GPU".into()),
            (_, Some((gpu, module))) => {
                let mut uploaded = |t: &Tensor| {
                    if gpu_part(&t.name, k, n) {
                        m.release(t);
                    }
                };
                let net = CudaQwen2::new_layers(gpu, module, m, max_ctx, k, &mut uploaded)
                    .map_err(|e| {
                        format!("{g}/{n} layers on the GPU: {e}, or fewer --gpu-layers")
                    })?;
                Some(net)
            }
        };
        if g > 0 {
            release_gpu_part(m, k, n);
        }
        let repack = std::env::var("AXON_REPACK").map_or(true, |v| v != "0");
        let cpu = (k > 0).then(|| Qwen2::new_layers(m, repack, k));
        let halves = [
            cpu.as_ref().map(|c| c.n_layers),
            gpu.as_ref().map(|g| g.n_layers),
        ];
        assert!(
            halves.iter().flatten().all(|&l| l == n),
            "halves of different models"
        );
        Ok(HybridQwen2 {
            cpu,
            gpu,
            gpu_layers: g,
            n_layers: n,
        })
    }

    /// VRAM the GPU half holds with a `max_ctx`-token cache (0 if none).
    pub fn vram_bytes(&self, max_ctx: usize) -> usize {
        self.gpu
            .as_ref()
            .map_or(0, |g| g.weight_bytes() + g.cache_bytes(max_ctx))
    }

    /// "gpu layers 30/64, 9.10 GiB on GPU" (with a `max_ctx` cache).
    pub fn describe(&self, max_ctx: usize) -> String {
        format!(
            "gpu layers {}/{}, {:.2} GiB on GPU",
            self.gpu_layers,
            self.n_layers,
            self.vram_bytes(max_ctx) as f64 / GIB
        )
    }

    /// Both halves, each with its cache.
    fn split<'a>(
        &'a self,
        c: &'a mut HybridCache<'g>,
    ) -> (
        Half<'a, 'm>,
        Option<(&'a CudaQwen2<'g>, &'a mut CudaCache<'g>)>,
    ) {
        let cpu = match (&self.cpu, &mut c.cpu) {
            (Some(n), Some(c)) => Some((n, c)),
            (None, None) => None,
            _ => panic!("a HybridCache of another split"),
        };
        let gpu = match (&self.gpu, &mut c.gpu) {
            (Some(n), Some(c)) => Some((n, c)),
            (None, None) => None,
            _ => panic!("a HybridCache of another split"),
        };
        (cpu, gpu)
    }
}

type Half<'a, 'm> = Option<(&'a Qwen2<'m>, &'a mut KvCache)>;

/// Whether tensor `name` is the GPU half's alone (with layers k..n): its
/// layers, the output norm and head, and the embedding if the CPU doesn't
/// read it (k == 0; a tied head uploaded from it when k > 0 stays).
fn gpu_part(name: &str, k: usize, n: usize) -> bool {
    if let Some(rest) = name.strip_prefix("blk.") {
        let l: usize = rest
            .split('.')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        l >= k && l < n
    } else {
        name.starts_with("output") || (k == 0 && name == "token_embd.weight")
    }
}

/// Hand back what the GPU half uploaded (`gpu_part`); the matrices were
/// already released one by one as they went up, so this is the norms and
/// biases (releasing twice is a no-op).
fn release_gpu_part(m: &Model, k: usize, n: usize) {
    for t in m.tensors.iter().filter(|t| gpu_part(&t.name, k, n)) {
        m.release(t);
    }
}

/// Hidden states, token-major in one buffer (`CudaQwen2::feed_hidden`).
fn flat(xs: Vec<Vec<f32>>) -> Vec<f32> {
    xs.concat()
}

impl<'g> Lm for HybridQwen2<'_, 'g> {
    type Cache = HybridCache<'g>;

    fn vocab(&self) -> usize {
        match (&self.cpu, &self.gpu) {
            (_, Some(g)) => g.vocab,
            (Some(c), None) => c.vocab,
            (None, None) => unreachable!("a model with no layers"),
        }
    }

    fn new_cache(&self, max_ctx: usize) -> HybridCache<'g> {
        HybridCache {
            cpu: self.cpu.as_ref().map(|n| n.new_cache(max_ctx)),
            gpu: self.gpu.as_ref().map(|n| Lm::new_cache(n, max_ctx)),
        }
    }

    fn forward(&self, token: u32, pos: usize, c: &mut HybridCache<'g>) -> Vec<f32> {
        match self.split(c) {
            (Some((n, c)), None) => n.forward(token, pos, c),
            (None, Some((n, c))) => Lm::forward(n, token, pos, c),
            (Some((cn, cc)), Some((gn, gc))) => {
                let x = cn.hidden(token, pos, cc);
                gn.forward_hidden(gc, &x, pos)
            }
            (None, None) => unreachable!(),
        }
    }

    fn forward_argmax(&self, token: u32, pos: usize, c: &mut HybridCache<'g>) -> u32 {
        match self.split(c) {
            (Some((n, c)), None) => crate::argmax(&n.forward(token, pos, c)),
            (None, Some((n, c))) => n.forward_argmax(token, pos, c),
            (Some((cn, cc)), Some((gn, gc))) => {
                let x = cn.hidden(token, pos, cc);
                gn.argmax_hidden(gc, &x, pos)
            }
            (None, None) => unreachable!(),
        }
    }

    fn feed_batch(&self, tokens: &[u32], start: usize, c: &mut HybridCache<'g>) {
        match self.split(c) {
            (Some((n, c)), None) => n.feed_batch(tokens, start, c),
            (None, Some((n, c))) => Lm::feed_batch(n, tokens, start, c),
            (Some((cn, cc)), Some((gn, gc))) => {
                let xs = cn.hidden_batch(tokens, start, cc);
                gn.feed_hidden(gc, &flat(xs), start)
            }
            (None, None) => unreachable!(),
        }
    }

    fn verify_argmax(&self, tokens: &[u32], start: usize, c: &mut HybridCache<'g>) -> Vec<u32> {
        match self.split(c) {
            (Some((n, c)), None) => n.verify_argmax(tokens, start, c),
            (None, Some((n, c))) => Lm::verify_argmax(n, tokens, start, c),
            (Some((cn, cc)), Some((gn, gc))) => {
                let xs = cn.hidden_batch(tokens, start, cc);
                gn.verify_hidden(gc, &flat(xs), start)
            }
            (None, None) => unreachable!(),
        }
    }

    /// The CPU half's batch whenever it has layers: its batched pass sets
    /// the pace, and the GPU half chunks any size into its own passes
    /// (its scratch is sized for this: `cuda::qwen2::SPLIT_BATCH`).
    fn prefill_batch(&self) -> usize {
        match (&self.cpu, &self.gpu) {
            (None, Some(g)) => g.prefill_batch(),
            _ => crate::PREFILL_BATCH,
        }
    }
}

#[cfg(test)]
mod tests;
