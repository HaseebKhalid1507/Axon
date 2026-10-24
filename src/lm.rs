//! The backend seam: what generation and chat need from a language model.
//!
//! `generate` and `chat::Chat` are generic over `Lm`: the CPU model
//! (`model::Qwen2`), the GPU one (`cuda::qwen2::CudaQwen2`) and the two
//! split across devices (`hybrid::HybridQwen2`) implement the same calls.
//! The CLI holds one in a `Backend`, which dispatches each call to the one
//! inside (one match per token; the CPU's results are the same bits as
//! calling it directly).

use crate::cuda::qwen2::pipeline::{PipelineCache, PipelineQwen2};
use crate::cuda::qwen2::slots::CudaPool;
use crate::cuda::qwen2::{CacheKv, CudaCache, CudaQwen2};
use crate::cuda::{Gpu, Kernels, Module};
use crate::gguf::Model;
use crate::hybrid::{HybridCache, HybridQwen2, Placement};
use crate::kv::KvType;
use crate::model::{KvCache, Qwen2};
use crate::moe::{MoeCache, MoeQwen3};
use crate::sample::{self, Cand, CandSpec};

/// 0..n in chunks of pb (the last shorter).
pub fn chunk_ranges(n: usize, pb: usize) -> Vec<std::ops::Range<usize>> {
    (0..n)
        .step_by(pb.max(1))
        .map(|s| s..(s + pb).min(n))
        .collect()
}

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

    /// Whether one-token steps (`forward`, `forward_argmax`) may take a
    /// path `verify_argmax` never takes, so their picks can differ: a
    /// CPU/GPU split built with AXON_STREAM_DECODE=1 streams decode steps,
    /// verify passes never stream. Drafting is then not lossless, and
    /// `generate` refuses `max_draft > 0`. Default: no.
    fn decode_streams(&self) -> bool {
        false
    }

    /// The greedy pick after each `tokens[b]` at `start + b`, as
    /// `forward_argmax` on each in order would give (speculative verify).
    fn verify_argmax(&self, tokens: &[u32], start: usize, cache: &mut Self::Cache) -> Vec<u32>;

    /// Prompt tokens per `feed_batch` call that suit this backend (callers
    /// may pass any size: results don't depend on the batching).
    fn prefill_batch(&self) -> usize {
        crate::PREFILL_BATCH
    }

    /// How a caller that feeds a prompt in cancellable pieces (serve's
    /// single stream, chat) should cut n tokens fed from `start` into the
    /// cache: ranges of 0..n in order, each one `feed_batch` /
    /// `prefill_logits` call of at most `prefill_batch` tokens. Results
    /// don't depend on it. Default: `prefill_batch`-sized chunks; the GPU
    /// model gives its own passes (`CudaQwen2::prompt_passes`: sized to the
    /// GEMM waves, AXON_WAVE_SPLIT).
    fn prompt_chunks(
        &self,
        n: usize,
        start: usize,
        cache: &Self::Cache,
    ) -> Vec<std::ops::Range<usize>> {
        let _ = (start, cache);
        chunk_ranges(n, self.prefill_batch())
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

/// One token of a multi-sequence step (`BatchLm::step`): `token` at
/// position `pos` of the sequence in pool slot `slot`.
#[derive(Clone, Copy, Debug)]
pub struct SeqRow<'a> {
    pub token: u32,
    pub pos: usize,
    pub slot: usize,
    pub head: RowHead<'a>,
}

/// What a row of a step returns.
#[derive(Clone, Copy, Debug)]
pub enum RowHead<'a> {
    /// `forward_argmax`'s pick.
    Argmax,
    /// `forward_cands`'s candidates; `recent` ends with the row's token.
    Sample { spec: CandSpec, recent: &'a [u32] },
}

/// A row's result, per its `RowHead`.
#[derive(Clone, Debug, PartialEq)]
pub enum RowOut {
    Pick(u32),
    Cands(Vec<Cand>),
}

/// Why a step failed.
#[derive(Clone, Debug, PartialEq)]
pub enum StepError {
    /// The rows were refused before any work ran (nothing changed): the
    /// requests in them fail, the backend is fine.
    Input(String),
    /// The device failed partway (CUDA errors are sticky).
    Device(String),
}

/// A backend that runs many sequences at once (`axon serve`'s continuous
/// batching): a pool of slots, each a sequence's cache of up to `ctx`
/// positions, and steps of one row per token of any slots. Every row's
/// cache rows and result are bit for bit what the `Lm` calls on one
/// sequence give (`forward_argmax`, `forward_cands`, `feed_batch`), whatever
/// else is in the step. Fallible: a server answers an error rather than
/// panicking.
pub trait BatchLm: Lm {
    type Pool;

    /// A pool of `slots` sequences of up to `ctx` positions each, or why
    /// there is none (no room, or a backend without one).
    fn new_pool(&self, slots: usize, ctx: usize) -> Result<Self::Pool, String>;

    /// A paged pool (specs r11) of `pages` KV pages for up to `slots`
    /// sequences of up to `ctx` positions each.
    fn new_pool_paged(&self, slots: usize, ctx: usize, pages: usize) -> Result<Self::Pool, String>;

    /// `feed_batch` into slot `slot`: `tokens[b]` at `start + b`.
    fn feed_slot(
        &self,
        pool: &mut Self::Pool,
        slot: usize,
        tokens: &[u32],
        start: usize,
    ) -> Result<(), String>;

    /// One pass over `rows` (a slot's rows in ascending positions, each with
    /// its slot's earlier positions already in it, or earlier in `rows`):
    /// each row's keys and values into its slot, and its result.
    fn step(&self, pool: &mut Self::Pool, rows: &[SeqRow]) -> Result<Vec<RowOut>, StepError>;

    /// A cache whose rows are slot `slot`'s (the same memory): what the
    /// `Lm` calls write there, the slot holds, and `adopt` of it into that
    /// slot copies nothing.
    fn slot_cache(&self, pool: &Self::Pool, slot: usize) -> Result<Self::Cache, String>;

    /// Point `cache`, a `slot_cache` of `pool`, at slot `slot` instead.
    fn move_slot_cache(
        &self,
        pool: &Self::Pool,
        cache: &mut Self::Cache,
        slot: usize,
    ) -> Result<(), String>;

    /// Copy positions 0..n of `cache` into slot `slot`.
    fn adopt(
        &self,
        pool: &mut Self::Pool,
        slot: usize,
        cache: &Self::Cache,
        n: usize,
    ) -> Result<(), String>;

    /// Most tokens one of `feed_slot`'s passes takes: a longer feed is
    /// split into equal passes of at most this many.
    fn pass_max(&self, pool: &Self::Pool) -> usize;

    /// KV pages for slot `slot`'s positions 0..hi (a paged pool, specs
    /// r11): Ok(0) once it holds them, or how many pages are short (nothing
    /// taken). `feed_slot` and `step` take them themselves; the `Lm` calls
    /// on a `slot_cache` don't (infallible), so whoever runs those reserves
    /// the positions they write first. A plain pool: Ok(0).
    fn reserve(&self, pool: &mut Self::Pool, slot: usize, hi: usize) -> Result<usize, String>;

    /// Slot `slot` keeps the pages of positions 0..lo; the rest are free.
    fn release(&self, pool: &mut Self::Pool, slot: usize, lo: usize) -> Result<(), String>;

    /// The pool's pages (None: plain, every slot its ctx rows).
    fn pages(&self, pool: &Self::Pool) -> Option<PoolPages>;

    /// Lend the pool's batch scratch to `cache` (its `slot_cache`, which
    /// may have none of its own) for the `Lm` calls on it; the pool's
    /// `feed_slot` / `step` wait for `return_scratch`.
    fn lend_scratch(&self, pool: &mut Self::Pool, cache: &mut Self::Cache) -> Result<(), String>;

    /// `lend_scratch` undone.
    fn return_scratch(&self, pool: &mut Self::Pool, cache: &mut Self::Cache) -> Result<(), String>;
}

/// A paged pool's pages (`BatchLm::pages`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolPages {
    /// Positions per page.
    pub size: usize,
    pub free: usize,
    pub total: usize,
    /// Pages each slot holds.
    pub held: Vec<usize>,
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

/// The GPUs --gpus named (`set_gpus`), in pipeline order.
static GPUS: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();

/// "0,1" (or "1,0", "0,1,2,3"; a device may repeat): device ordinals.
pub fn parse_gpus(s: &str) -> Result<Vec<usize>, String> {
    let v: Vec<usize> = s
        .split(',')
        .map(|d| d.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("GPU ordinals separated by commas (0,1), got '{s}'"))?;
    if v.is_empty() {
        return Err("no GPUs".into());
    }
    Ok(v)
}

/// --gpus: the CUDA devices to run on (`Backend::new`), set once before it.
pub fn set_gpus(v: Vec<usize>) {
    let _ = GPUS.set(v);
}

/// The GPUs to run on: --gpus, else $AXON_GPUS, else None (device 0).
pub fn gpus_setting() -> Result<Option<Vec<usize>>, String> {
    if let Some(v) = GPUS.get() {
        return Ok(Some(v.clone()));
    }
    match std::env::var("AXON_GPUS") {
        Ok(v) => parse_gpus(&v)
            .map(Some)
            .map_err(|e| format!("AXON_GPUS: {e}")),
        Err(_) => Ok(None),
    }
}

/// The CUDA device of the model's output head: the last of --gpus (a
/// pipeline's last stage), else device 0. A draft model goes there.
pub fn head_gpu() -> Result<usize, String> {
    Ok(gpus_setting()?.and_then(|l| l.last().copied()).unwrap_or(0))
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
    /// Layers split across GPUs (--gpus 0,1; leaked as for `Cuda`).
    Pipeline(Box<PipelineQwen2<'static>>),
    /// A mixture-of-experts model: attention on the GPU, experts on the
    /// CPU (leaked as for `Cuda`).
    Moe(Box<MoeQwen3<'a, 'static>>),
}

pub enum BackendCache {
    Cpu(KvCache),
    Cuda(Box<CudaCache<'static>>),
    Hybrid(Box<HybridCache<'static>>),
    Pipeline(Box<PipelineCache<'static>>),
    Moe(Box<MoeCache<'static>>),
}

/// How `Backend::new_capped` places a model beyond `new`'s arguments.
#[derive(Clone, Copy, Debug)]
pub struct Placing {
    /// The context of a CPU/GPU split (and, without `fit`, of everything
    /// but the automatic whole-GPU model): see `new_capped`.
    pub split_ctx: usize,
    /// Leave the GPU model's Q6_K prefill copy to the caller
    /// (`CudaQwen2::make_q6k8_spare`, once serve's caches are made).
    pub defer_q6k8: bool,
    /// A paged KV pool will hold the caches (`axon serve`, specs r11),
    /// placed as main places it with --ctx (`paged_placement`).
    pub paged: bool,
    /// A MoE model's experts resident on the GPU (`--gpu-experts`; None:
    /// $AXON_GPU_EXPERTS, else 0).
    pub gpu_experts: Option<crate::cuda::qwen2::experts::GpuExperts>,
}

/// The shortest cache a paged serve keeps the whole model on the GPU for
/// when no --ctx was given.
pub const PAGED_FLOOR: usize = 8192;

/// Where a paged serve's model goes, placed automatically, with `free`
/// bytes of VRAM and `whole(c)` the whole model with one c-token cache.
#[derive(Debug, PartialEq, Eq)]
pub enum PagedPlace {
    /// Every layer on the GPU, the cache planned at the longest that fits
    /// (no --ctx: the context may be shorter than trained, as main's fit).
    WholeShrunk,
    /// main's rule (`choose_placement`), with this much VRAM: the same
    /// placement main makes (every layer on the GPU whenever one cache of
    /// the context fits; an explicit --ctx is never shortened). If the
    /// pages then hold less than one full --ctx sequence, serve takes
    /// main's plain plan (`serve::paged_plan`).
    Main { free: u64 },
}

/// `PagedPlace` of a model: with --ctx (`!fit`) main's rule as is (a
/// margin for the pool here cost the 14B at --ctx 16384 five layers on the
/// CPU where main keeps all: 1018-044243-ceb5); without, the whole model
/// on the GPU if it fits beside a `PAGED_FLOOR`-token cache (the pool's
/// pages then cap the context).
pub fn paged_placement(
    free: u64,
    max_ctx: usize,
    fit: bool,
    whole: impl Fn(usize) -> u64,
) -> PagedPlace {
    if fit && whole(max_ctx) > free && whole(PAGED_FLOOR.min(max_ctx)) <= free {
        return PagedPlace::WholeShrunk;
    }
    PagedPlace::Main { free }
}

impl Placing {
    /// `new`'s: max_ctx everywhere, the Q6_K copy made with the model.
    pub fn of(max_ctx: usize) -> Placing {
        Placing {
            split_ctx: max_ctx,
            defer_q6k8: false,
            paged: false,
            gpu_experts: None,
        }
    }
}

impl<'a> Backend<'a> {
    /// The model of `m` on `device`, with room for a `max_ctx`-token cache
    /// of type `kv` (checked up front on the GPU). On CUDA, `gpu_layers`
    /// Some(g) splits the model, the last g layers on the GPU
    /// (`HybridQwen2`, its GPU half's cache of type `kv`, the CPU half's
    /// f16, or f32 for `KvType::F32`; process-wide), or with g >= n_layers
    /// is the GPU model;
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
        Self::new_capped(
            m,
            device,
            max_ctx,
            gpu_layers,
            kv,
            fit,
            Placing::of(max_ctx),
        )
    }

    /// `new`, as `p` says (`axon serve`'s default: `Placing`). With
    /// p.split_ctx < max_ctx and `fit`, the whole model on the GPU (placed
    /// automatically, every layer by --gpu-layers, or a multi-GPU
    /// pipeline) gets the longest context up to max_ctx that fits (a
    /// pipeline at least `KEEP_CTX_FLOOR`, else split_ctx); a CPU/GPU split
    /// (automatic or --gpu-layers) gets split_ctx.
    pub(crate) fn new_capped(
        m: &'a Model,
        device: Device,
        max_ctx: usize,
        gpu_layers: Option<usize>,
        kv: KvType,
        fit: bool,
        p: Placing,
    ) -> Result<Backend<'a>, String> {
        let split_ctx = p.split_ctx;
        let long = fit && split_ctx < max_ctx;
        // `kv` is the whole KV choice: this reads no AXON_KV (the CLI parses
        // it and passes the type it names; an embedder passes its own). The
        // CPU caches' f16/f32 choice is process-wide (kv::use_f32, fixed at
        // the first cache): `kv` F32 fixes it f32, F16/Q8 f16, and a type
        // that conflicts with an earlier Backend's is an error.
        let want_f32 = kv == KvType::F32;
        crate::kv::set_f32(want_f32);
        if crate::kv::use_f32() != want_f32 {
            return Err(format!(
                "the {} KV cache: this process already makes {} caches (chosen once, at the first one)",
                if want_f32 { "f32" } else { "f16" },
                if want_f32 { "f16" } else { "f32" },
            ));
        }
        if device == Device::Cpu {
            // The CPU's f16/f32 choice: `kv`, fixed above (kv::use_f32).
            if kv == KvType::Q8 {
                return Err("the q8 KV cache is cuda-only (--device cuda)".into());
            }
            crate::model::moe_check_ram(m, true)?;
            return Ok(Backend::Cpu(Qwen2::new(m)));
        }
        if gpu_layers == Some(0) {
            return Err("--gpu-layers 0 is --device cpu".into());
        }
        if let Some(e) = crate::safetensors::gpu_refusal(m) {
            return Err(e);
        }
        // f32 isn't offered on the GPU: f16 there (as always).
        let ckv = match kv {
            KvType::Q8 => CacheKv::Q8 { group: 0 },
            _ => CacheKv::F16,
        };
        let gpus = gpus_setting()?;
        if let Some(list) = gpus.as_deref().filter(|l| l.len() > 1) {
            if gpu_layers.is_some() {
                return Err(
                    "--gpus (or AXON_GPUS) with more than one GPU runs every layer on \
                            them: drop --gpu-layers (or unset AXON_GPU_LAYERS)"
                        .into(),
                );
            }
            return Self::pipeline(
                m,
                list,
                if long { max_ctx } else { split_ctx },
                split_ctx,
                ckv,
            );
        }
        let ordinal = gpus.map_or(0, |l| l[0]);
        if crate::model::is_moe(m) {
            return Self::moe(
                m,
                ordinal,
                gpu_layers,
                if long { max_ctx } else { split_ctx },
                ckv,
                fit,
                !p.defer_q6k8,
                p.gpu_experts,
            );
        }
        let gpu: &'static Gpu = Box::leak(Box::new(Gpu::open_device(ordinal)?));
        let module: &'static Module<'static> = Box::leak(Box::new(Kernels::module_for(gpu, m)?));
        let whole_gpu = |ctx: usize, fit: bool| {
            CudaQwen2::with_kv_q6k8(gpu, module, m, ctx, ckv, fit, !p.defer_q6k8)
                .map(|n| Backend::Cuda(Box::new(n)))
        };
        let auto = gpu_layers.is_none();
        let stream = crate::hybrid::stream_setting();
        let g = match gpu_layers {
            // Every layer: the GPU model (same weights, cache and VRAM as a
            // split with no CPU half, and its own sampling paths).
            Some(g) => {
                let n = crate::hybrid::block_count(m)?;
                if g >= n {
                    // As all of it placed automatically (-ngl 999).
                    let (c, f) = if long {
                        (max_ctx, true)
                    } else {
                        (split_ctx, false)
                    };
                    return whole_gpu(c, f).map_err(|e| {
                        format!("{g}/{n} layers on the GPU: {e}, or fewer --gpu-layers")
                    });
                }
                g
            }
            None => {
                let mut free = gpu.info()?.free_bytes;
                let whole = CudaQwen2::vram_curve(gpu, module, m, ckv)?;
                if p.paged {
                    match paged_placement(free, max_ctx, fit, &whole) {
                        PagedPlace::WholeShrunk => return whole_gpu(max_ctx, true),
                        PagedPlace::Main { free: f } => free = f,
                    }
                }
                // A streaming split's VRAM (its bigger batch scratch and the
                // stream reserve) is what the split option is weighed by.
                let split =
                    || crate::hybrid::auto_gpu_layers(gpu, module, m, split_ctx, ckv, free, stream);
                match crate::hybrid::choose_placement(free, max_ctx, fit, whole, split)? {
                    Placement::Gpu { ctx } => return whole_gpu(ctx, false),
                    Placement::Split {
                        gpu_layers,
                        gpu_ctx,
                    } => {
                        if let Some(c) = gpu_ctx {
                            eprintln!(
                                "axon: the whole model fits in GPU memory only with a {c}-token \
                                 context; splitting it to keep {split_ctx} (--ctx {c} for all of it \
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
            HybridQwen2::new_with_stream(Some((gpu, module)), m, split_ctx, g, ckv, stream, auto)?;
        eprintln!("split:       {}", h.describe(split_ctx));
        Ok(Backend::Hybrid(Box::new(h)))
    }

    /// A qwen3moe model on GPU `ordinal`: attention, routers and head there
    /// (a `max_ctx` cache of type `kv`, shorter with `fit` if need be), the
    /// experts on the CPU. There's no layer split: --gpu-layers below the
    /// layer count is refused. `q6k8`: make the Q6_K prefill copy now
    /// (`Placing::defer_q6k8` off), as the whole-GPU model.
    #[allow(clippy::too_many_arguments)]
    fn moe(
        m: &'a Model,
        ordinal: usize,
        gpu_layers: Option<usize>,
        max_ctx: usize,
        kv: CacheKv,
        fit: bool,
        q6k8: bool,
        gpu_experts: Option<crate::cuda::qwen2::experts::GpuExperts>,
    ) -> Result<Backend<'a>, String> {
        let n = crate::hybrid::block_count(m)?;
        if gpu_layers.is_some_and(|g| g < n) {
            return Err(format!(
                "{}: a mixture-of-experts model doesn't split by layers (--gpu-layers): its \
                 attention runs on the GPU and its experts on the CPU",
                m.arch
            ));
        }
        let gpu: &'static Gpu = Box::leak(Box::new(Gpu::open_device(ordinal)?));
        let module: &'static Module<'static> = Box::leak(Box::new(Kernels::module_for(gpu, m)?));
        let router: &'static Module<'static> =
            Box::leak(Box::new(Kernels::moe_module_for(gpu, m)?));
        let want = match gpu_experts {
            Some(w) => w,
            None => crate::cuda::qwen2::experts::GpuExperts::from_env()?,
        };
        let n = MoeQwen3::new_with(gpu, module, router, m, max_ctx, kv, fit, q6k8, want)?;
        eprintln!("moe:         {}, kv {}", n.describe(n.ctx), n.kv_name());
        Ok(Backend::Moe(Box::new(n)))
    }

    /// The model's layers over the GPUs `ordinals` (in order), a
    /// `max_ctx`-token cache of type `kv` (`PipelineQwen2::new`: an even
    /// split if it fits; AXON_PP_CUTS sets it), told on stderr. With
    /// max_ctx > split_ctx, the longest context up to max_ctx that fits,
    /// if at least `KEEP_CTX_FLOOR` (`PipelineQwen2::fit_ctx`), else
    /// split_ctx.
    fn pipeline(
        m: &Model,
        ordinals: &[usize],
        max_ctx: usize,
        split_ctx: usize,
        kv: CacheKv,
    ) -> Result<Backend<'a>, String> {
        let mut devs = Vec::with_capacity(ordinals.len());
        for &o in ordinals {
            let gpu: &'static Gpu = Box::leak(Box::new(Gpu::open_device(o)?));
            let module: &'static Module<'static> =
                Box::leak(Box::new(Kernels::module_for(gpu, m)?));
            devs.push((gpu, module));
        }
        let floor = crate::hybrid::KEEP_CTX_FLOOR.min(max_ctx);
        let ctx = match max_ctx > split_ctx {
            true => PipelineQwen2::fit_ctx(&devs, m, max_ctx, floor, kv)?.unwrap_or(split_ctx),
            false => max_ctx,
        };
        let p = PipelineQwen2::new(&devs, m, ctx, kv, None)?;
        eprintln!("pipeline:    {}", p.describe(ctx));
        Ok(Backend::Pipeline(Box::new(p)))
    }

    /// The context the cache was planned for: the GPU's, the split's or
    /// the pipeline's own (max_ctx, or less with `fit`, or `Placing`'s
    /// split_ctx); the CPU's is max_ctx (it plans nothing).
    pub fn ctx(&self, max_ctx: usize) -> usize {
        match self {
            Backend::Cpu(_) => max_ctx,
            Backend::Cuda(n) => n.ctx,
            Backend::Hybrid(n) => n.ctx,
            Backend::Pipeline(n) => n.ctx(),
            Backend::Moe(n) => n.ctx,
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
            (Backend::Pipeline($n), BackendCache::Pipeline($c)) => {
                let ($n, $c) = (&**$n, &mut **$c);
                $body
            }
            (Backend::Moe($n), BackendCache::Moe($c)) => {
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
            Backend::Pipeline(n) => n.vocab(),
            Backend::Moe(n) => n.vocab(),
        }
    }

    fn new_cache(&self, max_ctx: usize) -> BackendCache {
        match self {
            Backend::Cpu(n) => BackendCache::Cpu(Lm::new_cache(n, max_ctx)),
            Backend::Cuda(n) => BackendCache::Cuda(Box::new(n.new_cache(max_ctx))),
            Backend::Hybrid(n) => BackendCache::Hybrid(Box::new(n.new_cache(max_ctx))),
            Backend::Pipeline(n) => BackendCache::Pipeline(Box::new(n.new_cache(max_ctx))),
            Backend::Moe(n) => BackendCache::Moe(Box::new(n.new_cache(max_ctx))),
        }
    }

    fn forward(&self, token: u32, pos: usize, cache: &mut BackendCache) -> Vec<f32> {
        dispatch!(self, cache, |n, c| Lm::forward(n, token, pos, c))
    }

    fn forward_argmax(&self, token: u32, pos: usize, cache: &mut BackendCache) -> u32 {
        dispatch!(self, cache, |n, c| n.forward_argmax(token, pos, c))
    }

    fn decode_streams(&self) -> bool {
        matches!(self, Backend::Hybrid(h) if h.decode_streams())
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
            Backend::Pipeline(n) => n.prefill_batch(),
            Backend::Moe(n) => n.prefill_batch(),
        }
    }

    fn prompt_chunks(
        &self,
        n: usize,
        start: usize,
        cache: &BackendCache,
    ) -> Vec<std::ops::Range<usize>> {
        match (self, cache) {
            (Backend::Cpu(net), BackendCache::Cpu(c)) => net.prompt_chunks(n, start, c),
            (Backend::Cuda(net), BackendCache::Cuda(c)) => net.prompt_chunks(n, start, c),
            (Backend::Hybrid(net), BackendCache::Hybrid(c)) => net.prompt_chunks(n, start, c),
            (Backend::Pipeline(net), BackendCache::Pipeline(c)) => net.prompt_chunks(n, start, c),
            (Backend::Moe(net), BackendCache::Moe(c)) => net.prompt_chunks(n, start, c),
            _ => panic!("a cache of another backend"),
        }
    }
}

/// Continuous batching runs on the GPU model only: the CPU and a split
/// serve one request at a time (`new_pool` says so).
impl BatchLm for Backend<'_> {
    type Pool = Box<CudaPool<'static>>;

    fn new_pool(&self, slots: usize, ctx: usize) -> Result<Self::Pool, String> {
        match self {
            Backend::Cuda(n) => n.pool(slots, ctx).map(Box::new),
            _ => Err("batching many requests is GPU-only (--device cuda, every layer)".into()),
        }
    }

    fn new_pool_paged(&self, slots: usize, ctx: usize, pages: usize) -> Result<Self::Pool, String> {
        match self {
            Backend::Cuda(n) => n
                .pool_with(
                    slots,
                    ctx,
                    Some(pages),
                    crate::cuda::qwen2::pages::shuffle_env(),
                )
                .map(Box::new),
            _ => Err("batching many requests is GPU-only (--device cuda, every layer)".into()),
        }
    }

    fn feed_slot(
        &self,
        pool: &mut Self::Pool,
        slot: usize,
        tokens: &[u32],
        start: usize,
    ) -> Result<(), String> {
        match self {
            Backend::Cuda(n) => n.feed_slot(pool, slot, tokens, start),
            _ => unreachable!("a pool is the GPU model's"),
        }
    }

    fn step(&self, pool: &mut Self::Pool, rows: &[SeqRow]) -> Result<Vec<RowOut>, StepError> {
        match self {
            Backend::Cuda(n) => n.step_rows(pool, rows),
            _ => unreachable!("a pool is the GPU model's"),
        }
    }

    fn slot_cache(&self, pool: &Self::Pool, slot: usize) -> Result<BackendCache, String> {
        match self {
            Backend::Cuda(n) => n
                .slot_view_bare(pool, slot)
                .map(|c| BackendCache::Cuda(Box::new(c))),
            _ => unreachable!("a pool is the GPU model's"),
        }
    }

    fn move_slot_cache(
        &self,
        pool: &Self::Pool,
        cache: &mut BackendCache,
        slot: usize,
    ) -> Result<(), String> {
        match (self, cache) {
            (Backend::Cuda(n), BackendCache::Cuda(c)) => n.retarget_view(pool, c, slot),
            _ => Err("move_slot_cache: a cache of another backend".into()),
        }
    }

    fn adopt(
        &self,
        pool: &mut Self::Pool,
        slot: usize,
        cache: &BackendCache,
        n: usize,
    ) -> Result<(), String> {
        match (self, cache) {
            (Backend::Cuda(m), BackendCache::Cuda(c)) => {
                m.adopt_cache(pool, slot, c, n).map(|_| ())
            }
            _ => Err("adopt: a cache of another backend".into()),
        }
    }

    fn pass_max(&self, pool: &Self::Pool) -> usize {
        pool.pass_max()
    }

    fn reserve(&self, pool: &mut Self::Pool, slot: usize, hi: usize) -> Result<usize, String> {
        match self {
            Backend::Cuda(n) => n.reserve(pool, slot, hi),
            _ => unreachable!("a pool is the GPU model's"),
        }
    }

    fn release(&self, pool: &mut Self::Pool, slot: usize, lo: usize) -> Result<(), String> {
        match self {
            Backend::Cuda(n) => n.release(pool, slot, lo),
            _ => unreachable!("a pool is the GPU model's"),
        }
    }

    fn pages(&self, pool: &Self::Pool) -> Option<PoolPages> {
        pool.page_info()
    }

    fn lend_scratch(&self, pool: &mut Self::Pool, cache: &mut BackendCache) -> Result<(), String> {
        match (self, cache) {
            (Backend::Cuda(n), BackendCache::Cuda(c)) => n.lend_scratch(pool, c),
            _ => Err("lend_scratch: a cache of another backend".into()),
        }
    }

    fn return_scratch(
        &self,
        pool: &mut Self::Pool,
        cache: &mut BackendCache,
    ) -> Result<(), String> {
        match (self, cache) {
            (Backend::Cuda(n), BackendCache::Cuda(c)) => n.return_scratch(pool, c),
            _ => Err("return_scratch: a cache of another backend".into()),
        }
    }
}

#[cfg(test)]
mod paged_place_tests {
    use super::*;
    use crate::hybrid::{Placement, choose_placement};

    const GIB: u64 = 1 << 30;

    /// A model of `w` GiB of weights, `kib` KiB of KV per position and
    /// `scratch` GiB of scratch (+ HEADROOM-like 0.25 GiB).
    fn curve(w: f64, kib: u64, scratch: f64) -> impl Fn(usize) -> u64 {
        move |c| ((w + scratch + 0.25) * GIB as f64) as u64 + c as u64 * kib * 1024
    }

    /// Paged placement then main's rule: (on the GPU, its planned ctx) or a
    /// split; `split_layers` what the split would put on the GPU.
    fn place(free: u64, ctx: usize, fit: bool, w: &dyn Fn(usize) -> u64) -> Placement {
        match paged_placement(free, ctx, fit, w) {
            PagedPlace::WholeShrunk => Placement::Gpu { ctx: 0 },
            PagedPlace::Main { free } => choose_placement(free, ctx, fit, w, || Ok(35)).unwrap(),
        }
    }

    /// Qwen2.5-14B Q4_K_M, `--kv q8 --ctx 32768` on a 12 GB card with
    /// 11.2 GiB free (the bella case): one full 32k sequence doesn't fit
    /// beside the weights, so it splits as main does (no 400 for a prompt
    /// main accepts); `--ctx 16384` fits whole.
    #[test]
    fn paged_placement_14b() {
        let w = curve(8.37, 99, 0.6);
        let free = (11.2 * GIB as f64) as u64;
        assert!(matches!(
            place(free, 32768, false, &w),
            Placement::Split { .. }
        ));
        assert_eq!(place(free, 16384, false, &w), Placement::Gpu { ctx: 16384 });
    }

    /// Squeezed VRAM (another process holds 0.7 GiB more): --ctx 16384 isn't
    /// shortened to what fits all-GPU, it splits; without --ctx the model
    /// stays on the GPU with a shorter context (the pages cap it).
    #[test]
    fn paged_placement_squeezed_splits_rather_than_shrinks() {
        let w = curve(8.37, 99, 0.6);
        let free = (10.5 * GIB as f64) as u64;
        assert!(matches!(
            place(free, 16384, false, &w),
            Placement::Split { .. }
        ));
        assert_eq!(
            paged_placement(free, 32768, true, &w),
            PagedPlace::WholeShrunk
        );
        // Past even the floor: main's rule (it shrinks or splits).
        let tiny = (9.5 * GIB as f64) as u64;
        assert!(matches!(
            paged_placement(tiny, 32768, true, &w),
            PagedPlace::Main { .. }
        ));
    }

    /// Wherever main places every layer on the GPU with one --ctx cache, the
    /// paged placement does too (and splits where main splits): 14B q8 at
    /// --ctx 16384 and 32768 over a sweep of free VRAM.
    #[test]
    fn paged_placement_matches_main_with_ctx() {
        let w = curve(8.37, 99, 0.6);
        for ctx in [8192, 16384, 32768] {
            for tenth in 90..=130 {
                let free = (tenth as f64 / 10.0 * GIB as f64) as u64;
                let main = choose_placement(free, ctx, false, &w, || Ok(35)).unwrap();
                assert_eq!(place(free, ctx, false, &w), main, "ctx {ctx} free {tenth}");
            }
        }
    }

    /// Qwen2.5-7B Q4_K_M f16 at 32768 on the same card: the whole model
    /// with the full context, with --ctx or without.
    #[test]
    fn paged_placement_all_gpu_kept() {
        let w = curve(4.36, 56, 0.8);
        let free = (11.2 * GIB as f64) as u64;
        assert_eq!(place(free, 32768, false, &w), Placement::Gpu { ctx: 32768 });
        assert_eq!(place(free, 32768, true, &w), Placement::Gpu { ctx: 32768 });
    }
}
