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

use crate::cuda::qwen2::stream::{HostLayer, StreamKv, Streamer};
use crate::cuda::qwen2::{CacheKv, CudaCache, CudaQwen2};
use crate::cuda::{Gpu, Module};
use crate::gguf::{Model, Tensor};
use crate::kv::Half as F16;
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
    /// Streaming the CPU layers through the GPU for big prefill batches
    /// (None: off, or no room). First, so it drops (unpinning the CPU's
    /// weights) before them.
    stream: Option<Streaming<'g>>,
    /// Embedding and layers 0..k (and the head if k == n); None if k == 0.
    cpu: Option<Qwen2<'m>>,
    /// Layers k..n and the head (and the embedding if k == 0); None if k == n.
    gpu: Option<CudaQwen2<'g>>,
    /// Layers on the GPU, of `n_layers`.
    pub gpu_layers: usize,
    pub n_layers: usize,
    /// The context its caches are planned for (`build`'s max_ctx).
    pub ctx: usize,
}

/// PR A's state: the streamer and when to use it.
struct Streaming<'g> {
    st: std::cell::RefCell<Streamer<'g>>,
    /// Smallest prefill batch streamed (AXON_STREAM_MIN): below it the
    /// CPU computes its layers as without streaming.
    min: usize,
    /// AXON_STREAM_DECODE=1 (experiment): stream one-token steps too.
    decode: bool,
    /// Chunks streamed so far (tests: that streaming really happened).
    chunks: std::cell::Cell<usize>,
}

/// VRAM streaming leaves free beyond what it and the cache take (the
/// automatic split reserves it and keeps `DESKTOP_MARGIN` besides; with
/// --gpu-layers this is all that's kept).
const STREAM_HEADROOM: usize = 256 << 20;

/// Default `Streaming::min`: the break-even batch (NOTES: measured).
pub const STREAM_MIN: usize = 32;

/// Prompt tokens per `feed_batch` while streaming: the chunks a call
/// streams are split evenly from it (each up to the GPU's batch max), so a
/// prompt's tail doesn't make a small chunk that pays a whole stream.
const STREAM_BATCH: usize = 4096;

/// AXON_STREAM_DECODE=1: the experiment that streams one-token decode
/// steps too (AirLLM's idea; ~2x slower than the CPU computing its layers
/// on 32B, NOTES). Not with drafting: verify passes never stream, so the
/// picks would no longer be the per-token ones (main refuses the pair).
pub fn stream_decode_setting() -> bool {
    std::env::var("AXON_STREAM_DECODE").is_ok_and(|v| v == "1")
}

/// Whether streaming is on: AXON_STREAM unset or anything but "0".
pub fn stream_setting() -> bool {
    std::env::var("AXON_STREAM").map_or(true, |v| v != "0")
}

/// Each half's cache: KV of layers 0..k on the CPU, k..n on the GPU.
pub struct HybridCache<'g> {
    /// The streamed layers' VRAM slots, with the CPU cache's buffers
    /// pinned: dropped first, before `cpu`'s memory goes.
    skv: Option<StreamKv<'g>>,
    cpu: Option<KvCache>,
    gpu: Option<CudaCache<'g>>,
}

/// The model's layer count.
pub fn block_count(m: &Model) -> Result<usize, String> {
    crate::model::meta_num(m, &format!("{}.block_count", m.arch))
        .map(|v| v as usize)
        .ok_or_else(|| "missing metadata block_count".to_string())
}

/// Most layers the GPU can take (the last ones, plus the head) with room
/// for a `max_ctx`-token cache of type `kv` (q8's smaller cache leaves room
/// for more layers) and `DESKTOP_MARGIN` to spare out of `free`
/// bytes of VRAM (the caller's `gpu.info()?.free_bytes`): 0 if not even
/// one fits. With `stream` (PR A: prefill streams the CPU's layers), the
/// GPU half's batch scratch is `split_batch(true)` and `stream_reserve`
/// is kept besides, as `HybridQwen2::new_with_stream` will allocate. The
/// VRAM a split needs only shrinks as layers move to the CPU (the reserve
/// is the same for every split), so a binary search over
/// `CudaQwen2::vram_need` finds it.
pub fn auto_gpu_layers(
    gpu: &Gpu,
    module: &Module,
    m: &Model,
    max_ctx: usize,
    kv: CacheKv,
    free: u64,
    stream: bool,
) -> Result<usize, String> {
    let n = block_count(m)?;
    let stream = stream && !crate::kv::use_f32();
    let reserve = match stream {
        true => stream_reserve(m, max_ctx)?,
        false => 0,
    };
    split_gpu_layers(n, free, stream, reserve, |g, batch| {
        CudaQwen2::vram_need(gpu, module, m, max_ctx, n - g, kv, batch)
    })
}

/// `auto_gpu_layers`' sizing, given `need(g, batch)`, the VRAM of a split
/// with g of the n layers on the GPU and batch scratch for `batch` tokens:
/// with `stream`, the GPU half's scratch is `split_batch(true)`
/// (`STREAM_PASS`) and `reserve` (`stream_reserve`) is kept besides; else
/// `SPLIT_BATCH` scratch and no reserve. The most layers that fit
/// (`most_gpu_layers`). Pure: the placement tests drive it with a model
/// of the needs.
pub fn split_gpu_layers(
    n: usize,
    free: u64,
    stream: bool,
    reserve: u64,
    mut need: impl FnMut(usize, usize) -> Result<u64, String>,
) -> Result<usize, String> {
    let batch = crate::cuda::qwen2::split_batch(stream);
    let extra = if stream { reserve } else { 0 };
    most_gpu_layers(n, free, |g| Ok(need(g, batch)? + extra))
}

/// `auto_gpu_layers`' search, given `need(g)`, the VRAM a split with g of
/// the n layers on the GPU needs (it shrinks as g does): the largest g in
/// 0..=n with need(g) plus `DESKTOP_MARGIN` within `free`, 0 if none.
pub fn most_gpu_layers(
    n: usize,
    free: u64,
    mut need: impl FnMut(usize) -> Result<u64, String>,
) -> Result<usize, String> {
    let (mut lo, mut hi) = (0, n); // fits(lo) (no GPU layers); find the largest
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if need(mid)? + DESKTOP_MARGIN <= free {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Ok(lo)
}

/// The largest context `fit` gives the whole model on the GPU, given
/// `need(ctx)`, the VRAM it needs with a ctx-token cache (growing with
/// ctx): max_ctx if that fits in `free`, else the largest multiple of 256
/// below it that does, None if not even 256 does.
pub fn fit_ctx(max_ctx: usize, free: u64, need: impl Fn(usize) -> u64) -> Option<usize> {
    if need(max_ctx) <= free {
        return Some(max_ctx);
    }
    let mut c = max_ctx / 256 * 256;
    while c >= 256 && need(c) > free {
        c -= 256;
    }
    (c >= 256).then_some(c)
}

/// The shortest context `choose_placement` takes over a split for the
/// whole model on the GPU: half of max_ctx, but never more than
/// `KEEP_CTX_FLOOR`.
pub fn keep_ctx(max_ctx: usize) -> usize {
    (max_ctx / 2).min(KEEP_CTX_FLOOR)
}

/// A context this long is enough for a chat whatever max_ctx is: with a
/// longer default (a 32k model, a raised `DEFAULT_CHAT_CTX`), a 4096-token
/// all-GPU chat still beats a split. Below both it and half of max_ctx,
/// the conversation forgets its start too soon, and the split (slower per
/// token, as its CPU layers run at RAM speed) is the better default.
pub const KEEP_CTX_FLOOR: usize = 4096;

/// Where `Backend::new` puts a model when no --gpu-layers says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// All of it on the GPU, with a cache for `ctx` tokens (max_ctx, or
    /// less with `fit`).
    Gpu { ctx: usize },
    /// The last `gpu_layers` layers on the GPU, a max_ctx-token cache.
    /// `gpu_ctx`: the shorter context the whole model would have had on
    /// the GPU (with `fit`), for the note saying why it's split.
    Split {
        gpu_layers: usize,
        gpu_ctx: Option<usize>,
    },
    /// Not even one layer fits, nor the whole model at 256 tokens.
    Nowhere,
}

/// The placement policy, a pure function of the free VRAM, the model's
/// needs (counting the cache type: q8 makes both smaller) and max_ctx.
/// `whole(ctx)`: the VRAM for all of it on the GPU with a ctx-token cache;
/// `split()`: the most GPU layers a max_ctx split fits (`auto_gpu_layers`,
/// only called if the whole model doesn't fit at max_ctx).
///
/// The whole model at max_ctx if it fits. Else, with `fit` (chat's default
/// context), a shorter all-GPU context if it's at least `keep_ctx(max_ctx)`
/// (all-GPU speed for a still-long chat); else the split at max_ctx if it
/// has a GPU layer; else any all-GPU context `fit` finds (>= 256 tokens).
pub fn choose_placement(
    free: u64,
    max_ctx: usize,
    fit: bool,
    whole: impl Fn(usize) -> u64,
    split: impl FnOnce() -> Result<usize, String>,
) -> Result<Placement, String> {
    if whole(max_ctx) <= free {
        return Ok(Placement::Gpu { ctx: max_ctx });
    }
    let shrunk = if fit {
        fit_ctx(max_ctx, free, &whole)
    } else {
        None
    };
    match shrunk {
        Some(c) if c >= keep_ctx(max_ctx) => return Ok(Placement::Gpu { ctx: c }),
        _ => {}
    }
    Ok(match (split()?, shrunk) {
        (0, Some(c)) => Placement::Gpu { ctx: c },
        (0, None) => Placement::Nowhere,
        (g, gpu_ctx) => Placement::Split {
            gpu_layers: g,
            gpu_ctx,
        },
    })
}

/// VRAM streaming a split's CPU layers takes with the smallest ring, from
/// the file's tensor sizes (`CudaQwen2::ring_bounds`, `stream_bytes`,
/// `stream_kv_bytes`), plus STREAM_HEADROOM: what the automatic split
/// leaves for it. Over every layer, so it's the same for any split (the
/// placement search stays monotone): at most a few MB more than the CPU
/// layers alone need.
pub fn stream_reserve(m: &Model, max_ctx: usize) -> Result<u64, String> {
    let k = block_count(m)?;
    let meta = |key: &str| {
        crate::model::meta_num(m, &format!("{}.{key}", m.arch))
            .map(|v| v as usize)
            .ok_or_else(|| format!("missing metadata {key}"))
    };
    let dim = meta("embedding_length")?;
    // head_dim from the GGUF when it says (qwen3), else dim / heads.
    let hd = match meta("attention.key_length") {
        Ok(hd) => hd,
        Err(_) => dim / meta("attention.head_count")?,
    };
    let kvd = meta("attention.head_count_kv")? * hd;
    const MATS: [&str; 7] = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    let mut ring = 0;
    for l in 0..k {
        let sizes: Vec<usize> = MATS
            .iter()
            .map(|s| {
                let name = format!("blk.{l}.{s}.weight");
                m.tensors
                    .iter()
                    .find(|t| t.name == name)
                    .map(|t| (t.n_bytes as usize).next_multiple_of(256))
                    .ok_or_else(|| format!("missing tensor {name}"))
            })
            .collect::<Result<_, _>>()?;
        let biggest = sizes.iter().max().copied().unwrap_or(0);
        ring = ring.max(sizes.iter().sum::<usize>() + biggest);
    }
    // Staging (2 x 16 MiB), norms and biases (a few floats per layer;
    // qwen3's q/k norms 2 head_dim more, each padded to 64).
    let rest = (32 << 20) + k * (3 * dim + 2 * kvd + 2 * hd + 512) * 4;
    let slots = 2 * 2 * max_ctx * kvd * 2;
    Ok((ring + rest + slots + STREAM_HEADROOM) as u64)
}

impl<'m, 'g> HybridQwen2<'m, 'g> {
    /// The model with its last `gpu_layers` layers (clamped to n) and head
    /// on the GPU (`dev`, needed unless gpu_layers is 0), the rest on the
    /// CPU (repacked unless AXON_REPACK=0, as `Qwen2::new`), with room for
    /// a `max_ctx`-token cache (the GPU half's of type `kv`; the CPU half's
    /// is f16, or f32 for `KvType::F32`, kv::use_f32). Each half loads only its own weights, and
    /// the GPU's are handed back to `m` once uploaded (`Model::release`;
    /// on a copy of the file they can't be read again). Streaming off (PR
    /// B's behavior); the CLI uses `new_with_stream`.
    #[cfg_attr(not(test), allow(dead_code))] // tests and PR B's exact path
    pub fn new(
        dev: Option<(&'g Gpu, &'g Module<'g>)>,
        m: &'m Model,
        max_ctx: usize,
        gpu_layers: usize,
        kv: CacheKv,
    ) -> Result<HybridQwen2<'m, 'g>, String> {
        Self::new_with_stream(dev, m, max_ctx, gpu_layers, kv, false, false)
    }

    /// `new`, with `stream`: prefill batches of at least `STREAM_MIN`
    /// tokens (AXON_STREAM_MIN) run the CPU's layers on the GPU, their
    /// weights streamed over (`cuda::qwen2::stream`), if the CPU's KV cache
    /// is f16 and the VRAM it takes (a ring of two layers, by default) fits
    /// next to a `max_ctx` cache; else a note on stderr and no streaming.
    /// `auto`: the split came from `auto_gpu_layers`, so streaming keeps
    /// `DESKTOP_MARGIN` free too (an explicit --gpu-layers only keeps
    /// STREAM_HEADROOM).
    pub fn new_with_stream(
        dev: Option<(&'g Gpu, &'g Module<'g>)>,
        m: &'m Model,
        max_ctx: usize,
        gpu_layers: usize,
        kv: CacheKv,
        stream: bool,
        auto: bool,
    ) -> Result<HybridQwen2<'m, 'g>, String> {
        let repack = std::env::var("AXON_REPACK").map_or(true, |v| v != "0");
        let margin = if auto { DESKTOP_MARGIN as usize } else { 0 };
        Self::build(dev, m, max_ctx, gpu_layers, kv, stream, repack, margin)
    }

    /// `new_with_stream`, the CPU's layers repacked or not (`repack`).
    #[allow(clippy::too_many_arguments)]
    fn build(
        dev: Option<(&'g Gpu, &'g Module<'g>)>,
        m: &'m Model,
        max_ctx: usize,
        gpu_layers: usize,
        kv: CacheKv,
        stream: bool,
        repack: bool,
        margin: usize,
    ) -> Result<HybridQwen2<'m, 'g>, String> {
        if m.arch == "gemma3" {
            return Err(
                "gemma3 doesn't split across the CPU and GPU yet: the whole model \
                        goes on the GPU (a smaller --ctx), or --device cpu"
                    .into(),
            );
        }
        let n = block_count(m)?;
        let g = gpu_layers.min(n);
        let k = n - g;
        // Streaming feeds the GPU half whole prompt chunks: its batch
        // scratch is sized for that, decided up front: if the plan with
        // the bigger scratch and the stream reserve doesn't fit, no
        // streaming and SPLIT_BATCH scratch (as without it).
        let mut wants_stream = stream && k > 0 && g > 0 && !crate::kv::use_f32();
        if let (true, Some((gpu, module))) = (wants_stream, dev) {
            let sb = crate::cuda::qwen2::split_batch(true);
            let need = CudaQwen2::vram_need(gpu, module, m, max_ctx, k, kv, sb)?
                + stream_reserve(m, max_ctx)?
                + margin as u64;
            let free = gpu.info()?.free_bytes;
            if need > free {
                eprintln!(
                    "note: not streaming the CPU layers for prefill: it needs {:.2} GiB of GPU \
                     memory with this split, {:.2} GiB is free; use fewer --gpu-layers",
                    need as f64 / GIB,
                    free as f64 / GIB
                );
                wants_stream = false;
            }
        }
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
                let sb = crate::cuda::qwen2::split_batch(wants_stream);
                let net = CudaQwen2::new_layers(gpu, module, m, max_ctx, k, kv, sb, &mut uploaded)
                    .map_err(|e| {
                        format!("{g}/{n} layers on the GPU: {e}, or fewer --gpu-layers")
                    })?;
                Some(net)
            }
        };
        if g > 0 {
            release_gpu_part(m, k, n);
        }
        let cpu = (k > 0).then(|| Qwen2::new_layers(m, repack, k));
        let halves = [
            cpu.as_ref().map(|c| c.n_layers),
            gpu.as_ref().map(|g| g.n_layers),
        ];
        assert!(
            halves.iter().flatten().all(|&l| l == n),
            "halves of different models"
        );
        let stream = match (&cpu, &gpu) {
            (Some(c), Some(g)) if wants_stream => Self::streaming(c, g, max_ctx, margin)
                .map_err(|e| eprintln!("note: not streaming the CPU layers for prefill: {e}"))
                .ok(),
            _ => None,
        };
        Ok(HybridQwen2 {
            stream,
            cpu,
            gpu,
            gpu_layers: g,
            n_layers: n,
            ctx: max_ctx,
        })
    }

    /// The streamer for the CPU's layers, if it fits beside a `max_ctx` cache.
    /// `margin`: VRAM to leave free besides (DESKTOP_MARGIN for an
    /// automatic split).
    fn streaming(
        cpu: &Qwen2<'m>,
        gpu: &CudaQwen2<'g>,
        max_ctx: usize,
        margin: usize,
    ) -> Result<Streaming<'g>, String> {
        if crate::kv::use_f32() {
            return Err("the CPU half's KV cache is f32 (KvType::F32; the CLI's AXON_KV=f32): streaming needs f16".into());
        }
        let src: Vec<HostLayer> = (0..cpu.layer_count()).map(|l| cpu.host_layer(l)).collect();
        // Room for the cache `new_layers` planned for, its slots and
        // STREAM_HEADROOM first; the ring takes what's left, up to two
        // layers, at least its minimum (AXON_STREAM_RING_MB overrides).
        let cache = gpu.cache_bytes(max_ctx) + gpu.stream_kv_bytes(max_ctx);
        let (lo, hi) = CudaQwen2::ring_bounds(&src);
        let fixed = gpu.stream_bytes(&src, 0) + cache + STREAM_HEADROOM + margin;
        let free = gpu.free_bytes()? as usize;
        let room = free.saturating_sub(fixed);
        let ring = match std::env::var("AXON_STREAM_RING_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(mb) => (mb << 20).max(lo),
            None => room.min(hi),
        };
        if ring > room || ring < lo {
            return Err(format!(
                "it needs {:.2} GiB of GPU memory beside the cache, {:.2} GiB is free; \
                 use fewer --gpu-layers",
                (fixed - cache + ring.max(lo)) as f64 / GIB,
                free.saturating_sub(cache) as f64 / GIB
            ));
        }
        let st = gpu.streamer(&src, cpu.r8_arena(), ring)?;
        let env = |k: &str| std::env::var(k).ok();
        Ok(Streaming {
            st: std::cell::RefCell::new(st),
            min: env("AXON_STREAM_MIN")
                .and_then(|v| v.parse().ok())
                .unwrap_or(STREAM_MIN)
                .max(1),
            decode: stream_decode_setting(),
            chunks: std::cell::Cell::new(0),
        })
    }

    /// Prompt chunks streamed so far (tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn streamed_chunks(&self) -> usize {
        self.stream.as_ref().map_or(0, |s| s.chunks.get())
    }

    /// Stream `tokens` at start.. through every layer if this split and
    /// cache do and the batch is big enough (`min`); false if not (nothing
    /// done). With `head`, the picks (one-token decode experiment).
    fn try_stream(
        &self,
        tokens: &[u32],
        start: usize,
        c: &mut HybridCache<'g>,
        min: usize,
        head: bool,
    ) -> Option<Vec<u32>> {
        let s = self.stream.as_ref()?;
        if tokens.len() < min {
            return None;
        }
        let (cn, gn) = (self.cpu.as_ref()?, self.gpu.as_ref()?);
        let (skv, cc, gc) = match c {
            HybridCache {
                skv: Some(k),
                cpu: Some(cc),
                gpu: Some(gc),
            } => (&*k, cc, gc),
            _ => return None,
        };
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= cn.vocab) {
            panic!("token {t} out of range");
        }
        let src: Vec<HostLayer> = (0..cn.layer_count()).map(|l| cn.host_layer(l)).collect();
        // Even chunks of at most the GPU's batch max.
        let n = tokens.len();
        let parts = n.div_ceil(CudaQwen2::batch_max(gc));
        let mut picks = Vec::new();
        let mut at = 0;
        for p in 0..parts {
            let len = n / parts + (p < n % parts) as usize;
            let chunk = &tokens[at..at + len];
            let xs = cn.embed_batch(chunk);
            let mut hkv = cc.f16_layers().expect("streaming needs an f16 CPU cache");
            let mut st = s.st.borrow_mut();
            picks.extend(gn.stream_feed(gc, &mut st, skv, &src, &mut hkv, &xs, start + at, head));
            s.chunks.set(s.chunks.get() + 1);
            at += len;
        }
        Some(picks)
    }

    /// VRAM the GPU half holds with a `max_ctx`-token cache (0 if none).
    pub fn vram_bytes(&self, max_ctx: usize) -> usize {
        self.gpu
            .as_ref()
            .map_or(0, |g| g.weight_bytes() + g.cache_bytes(max_ctx))
    }

    /// "gpu layers 30/64, 9.10 GiB on GPU" (with a `max_ctx` cache), and
    /// whether prefill streams.
    pub fn describe(&self, max_ctx: usize) -> String {
        let st = match &self.stream {
            Some(s) => format!(", prefill streams the CPU layers (batches >= {})", s.min),
            None => String::new(),
        };
        format!(
            "gpu layers {}/{}, {:.2} GiB on GPU{st}",
            self.gpu_layers,
            self.n_layers,
            self.vram_bytes(max_ctx) as f64 / GIB
        )
    }

    /// The GPU half, if it has GPU layers.
    pub fn gpu_net(&self) -> Option<&CudaQwen2<'g>> {
        self.gpu.as_ref()
    }

    /// With streaming: the smallest prefill batch streamed, and whether
    /// one-token steps stream too (AXON_STREAM_DECODE).
    pub fn streams(&self) -> Option<(usize, bool)> {
        self.stream.as_ref().map(|s| (s.min, s.decode))
    }

    /// The GPU layers' cache type (`CudaQwen2::kv_name`; the CPU layers'
    /// is f16, or f32 for `KvType::F32`).
    pub fn kv_name(&self) -> &'static str {
        self.gpu.as_ref().map_or("f16", |g| g.kv_name())
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
        let mut cpu = self.cpu.as_ref().map(|n| n.new_cache(max_ctx));
        let gpu = self.gpu.as_ref().map(|n| Lm::new_cache(n, max_ctx));
        // The slots after the GPU half's cache, which `new` planned for.
        let skv = match (&self.stream, &self.gpu, &mut cpu) {
            (Some(_), Some(g), Some(c)) => {
                let host: Vec<(&[F16], &[F16])> = c
                    .f16_layers()
                    .expect("streaming needs an f16 CPU cache")
                    .into_iter()
                    .map(|(k, v)| (&*k, &*v))
                    .collect();
                g.stream_kv(max_ctx, &host)
                    .map_err(|e| eprintln!("note: this cache won't stream: {e}"))
                    .ok()
            }
            _ => None,
        };
        HybridCache { skv, cpu, gpu }
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

    fn decode_streams(&self) -> bool {
        self.stream.as_ref().is_some_and(|s| s.decode)
    }

    fn forward_argmax(&self, token: u32, pos: usize, c: &mut HybridCache<'g>) -> u32 {
        if self.stream.as_ref().is_some_and(|s| s.decode)
            && let Some(p) = self.try_stream(&[token], pos, c, 1, true)
        {
            return p[0];
        }
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
        if let Some(s) = &self.stream {
            if self.try_stream(tokens, start, c, s.min, false).is_some() {
                return;
            }
            if tokens.len() > crate::PREFILL_BATCH {
                // The CPU's batch size (same bits, any batching).
                for (i, b) in tokens.chunks(crate::PREFILL_BATCH).enumerate() {
                    self.feed_batch(b, start + i * crate::PREFILL_BATCH, c);
                }
                return;
            }
        }
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

    /// Sampled verify: the CPU half's batched pass, then the GPU half's
    /// batched passes with on-device candidates (`verify_cands_hidden`), so
    /// a sampled draft reads the CPU layers' weights once per pass, as a
    /// greedy one does. Same candidates as `forward_cands` per row: each
    /// half's rows have its per-token bits, and the GPU's candidates equal
    /// the host's `sample::cands` of the same logits.
    fn verify_cands(
        &self,
        tokens: &[u32],
        start: usize,
        c: &mut HybridCache<'g>,
        spec: &crate::sample::CandSpec,
        recent: &[u32],
    ) -> Vec<Vec<crate::sample::Cand>> {
        let (nb, n) = (tokens.len(), recent.len());
        assert!(
            n >= nb && recent[n - nb..] == *tokens,
            "recent must end with tokens"
        );
        match self.split(c) {
            (Some((n, c)), None) => n.verify_cands(tokens, start, c, spec, recent),
            (None, Some((n, c))) => Lm::verify_cands(n, tokens, start, c, spec, recent),
            (Some((cn, cc)), Some((gn, gc))) => {
                let xs = cn.hidden_batch(tokens, start, cc);
                gn.verify_cands_hidden(gc, &flat(xs), start, spec, recent)
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
            _ if self.stream.is_some() => STREAM_BATCH,
            _ => crate::PREFILL_BATCH,
        }
    }
}

#[cfg(test)]
mod tests;
