//! Qwen2 on the GPU: `CudaQwen2`, an `Lm` whose weights, KV cache and
//! scratch all live in VRAM, running decode.cu's kernels one token at a time
//! (`forward`), and prefill.cu's for batches (`feed_batch`, `verify_argmax`:
//! batch.rs), which give every token the same bits as the one-token step.
//!
//! Weights are uploaded once, in their GGUF block formats (no repacking),
//! into one arena: all of them, or for a model split across devices
//! (`new_layers`, `crate::hybrid`) the last layers and the head only, which
//! then start from the hidden state the CPU's layers left (one copy in).
//! Everything a token needs is allocated by `new` (weights)
//! and `new_cache` (KV cache, activations, logits, scratch for batches of up
//! to `PREFILL_BATCH` tokens), so a forward never allocates; `new` checks
//! that both fit in the free VRAM first.
//!
//! One token is a fixed sequence of 9 kernels per layer (`enqueue`) whose
//! only per-token inputs, the token id and position, are read from a device
//! buffer written by an async copy before each step. So the sequence is
//! captured once into a CUDA graph per cache and replayed (AXON_CUDA_GRAPHS=0
//! launches the kernels one by one instead: same kernels, same bits).
//!
//! Numerics (decode.cu): activations are quantized exactly as the CPU's
//! matvec does (Q8_K / Q8_0), so the weights meet the same 8-bit values in
//! exact integer dot products; the float scaling and every sum (blocks,
//! RMSNorm's squares, attention, softmax) are split and ordered differently,
//! in a fixed order. Results are deterministic run to run and equal with and
//! without graphs, close to the CPU's but not bit-identical (see the tests
//! for how close).
//!
//! Thread model: `Gpu` is !Send and binds its context on every call, and
//! everything here borrows it, so a `CudaQwen2` and its caches are used from
//! the one thread that opened the GPU. That's all generation needs.
//!
//! Errors: `Lm` is infallible, so a CUDA failure after setup (a device
//! fault, a lost GPU) panics with the driver's message. Setup errors (no
//! GPU, out of memory, unsupported model) are `Err`s from `new` / `try_cache`.

use super::{Arg, DevBuf, Func, Gpu, Graph, HostBuf, Module};
use crate::gguf::{Model, Tensor};
use crate::lm::Lm;
use crate::quant::{self, Q4_K, Q5_0, Q6_K, Q8_0};
use crate::sample::{self, CAND_MAX, Cand, CandSpec};

/// Positions per attention tile (decode.cu ATT_TILE) and largest GQA group.
const ATT_TILE: usize = 128;
const MAX_GROUP: usize = 8;
/// Most attention splits (decode.cu MAX_SPLIT).
const MAX_SPLIT: usize = 256;
/// Positions per block of the tensor-core attention (decode.cu ATT_BLK).
const ATT_BLK: usize = 64;

/// The tensor-core attention's score scale (decode.cu PRE): 1 / sqrt(hd)
/// times log2(e), in f32, the same for every kernel.
fn att_pre(head_dim: usize) -> f32 {
    (1.0 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E
}
/// Threads per GEMV block (decode.cu GEMV_THREADS).
const GEMV_THREADS: u32 = 128;
/// Rows per block of the gemm2 GEMMs (prefill.cu G2_BM); tokens per block
/// and stages are 128 and 2 (gemm2_*) or 64 and 3 (gemm2s_*).
const G2_BM: usize = 64;
/// gemm2's dynamic shared memory for type qt, bn tokens per block and st
/// stages (prefill.cu G2L::SMEM; the kernel traps if given less).
fn gemm2_smem(qt: u32, bn: usize, st: usize) -> usize {
    let (raw, kq) = (qt == Q4_K, qt == Q4_K || qt == Q6_K);
    let w = if raw {
        G2_BM * 176
    } else {
        G2_BM * 256 + G2_BM * if qt == Q6_K { 20 } else { 32 }
    };
    let stage = bn * 256 + if raw { bn * 32 } else { 0 } + if kq { bn * 4 } else { bn * 32 };
    let nwb = if raw {
        st
    } else if st >= 3 {
        2
    } else {
        1
    };
    nwb * w + st * stage
}
/// Most tokens per batched pass on the GPU (`Lm::prefill_batch`): big
/// batches keep the tensor cores busy and read the weights fewer times.
/// 9 x 128 (gemm2's token tiles): with equal-size passes (`batch::passes`)
/// a 2100-token prompt is two passes of 1050, not 1024 + 1024 + 52 (the
/// short pass's GEMMs are latency-bound: ~5 % of the prompt's time).
const PREFILL_BATCH: usize = 1152;
/// The batch scratch placement and `fit` budget for (main's 1024 before
/// round 4): `build` takes `PREFILL_BATCH` only when its extra scratch fits
/// besides, so the bigger passes never cost context or tip a placement.
const PLAN_BATCH: usize = 1024;
/// `PREFILL_BATCH` for the GPU half of a split (first > 0, `crate::hybrid`):
/// the CPU half's batched pass sets the pace, so the hybrid feeds at most
/// `crate::PREFILL_BATCH` tokens per call, and a speculative verify pass
/// takes K + 1 (8 at --draft 7; bigger ones are chunked, same bits). The
/// scratch for 512 tokens would cost 160 MiB more on 32B (99 MiB on 14B)
/// that the split's placement (`vram_need`) counts against layers.
/// Placement stays monotone: first == 0 (the whole model) gets the bigger
/// `PREFILL_BATCH` scratch on top of one more layer. PR A: a split that
/// streams its CPU layers for prefill feeds the GPU half whole prompt
/// chunks, so it gets `STREAM_PASS` (`split_batch(true)`).
///
/// Pinned at 64, not `crate::PREFILL_BATCH`: the CPU batch went 64 -> 128
/// (CPU prefill throughput), but doubling this scratch would cost the GPU
/// half ~16 MiB on 7B (~23 MiB on 32B) and could push a layer to the CPU near
/// a placement boundary. The GPU half chunks a 128-token hybrid batch into
/// passes of 64 (same bits), so placement stays what it was at 64.
pub const SPLIT_BATCH: usize = 64;

/// Most tokens per batched pass of a split that streams its CPU layers
/// (`split_batch(true)`): the streamed prefill's chunk size. Its own
/// constant, not `PREFILL_BATCH` (1152 since round 4): the scratch for it
/// is VRAM the streaming ring and the GPU layers don't get.
pub const STREAM_PASS: usize = 1024;

/// The GPU half's batch scratch for a split that streams (`stream`) or not.
pub fn split_batch(stream: bool) -> usize {
    if stream { STREAM_PASS } else { SPLIT_BATCH }
}

/// VRAM kept free beyond what `new` plans for: the context's own growth,
/// graphs, the kernel module.
const HEADROOM: u64 = 256 << 20;

/// VRAM a new cache leaves free beyond its own size (graphs it captures,
/// allocation granularity): less than `HEADROOM`, which `new` already keeps
/// for the first cache.
const CACHE_HEADROOM: u64 = 64 << 20;

/// Bytes after the last matrix in the weight arena (see `new`).
const ARENA_PAD: usize = 16;

const GIB: f64 = (1u64 << 30) as f64;

/// A matrix in the weight arena: type, byte offset, bytes per row, rows,
/// columns (values per row).
#[derive(Clone, Copy, Debug)]
struct Mat {
    qt: u32,
    off: usize,
    rb: usize,
    rows: usize,
    cols: usize,
    /// Q6_K: byte offset of its prefill copy (prefill.cu Q6K8, 288 bytes a
    /// block) in `CudaQwen2::w8`, if made (`make_q6k8`).
    c8: Option<usize>,
}

impl Mat {
    fn bytes(&self) -> usize {
        self.rb * self.rows
    }
}

/// One layer: matrices, and offsets (in floats) into the vector buffer.
#[derive(Clone, Copy)]
struct LayerW {
    attn_norm: usize,
    /// The q8 cache's key shift (decode.cu kv_q8_store), [kv_dim] floats.
    ksh: Option<usize>,
    bq: Option<usize>,
    bk: Option<usize>,
    bv: Option<usize>,
    /// Per-head q/k RMSNorm weights (qwen3), head_dim floats each.
    qn: Option<usize>,
    kn: Option<usize>,
    ffn_norm: usize,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
}

/// How a cache stores keys and values (`CudaQwen2::kv`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheKv {
    /// IEEE half, rounded to nearest even (saturating): the default.
    F16,
    /// int8 by decode.cu's q8 rule, with an f32 scale per `group` values:
    /// head_dim (one per position and kv head; 0 means head_dim) or 32.
    /// Half the bytes (plus 4 / group per value), so twice the context in
    /// the same memory. Results differ from f16's (measured in the tests);
    /// the batching contract holds within the mode.
    Q8 { group: usize },
}

/// The q8 kernels for one scale group size (`CacheKv::Q8`).
struct Q8Fns<'m> {
    /// The tensor-core attention (`tc_attn`): decode, and prefill.cu's
    /// per query (batches of at most `batch::att_per_query`) and flash
    /// kernels (attn_fa_b, attn_fa_bs: `batch::fa_shape`).
    attn_blk: Func<'m>,
    attn_blk_b: Func<'m>,
    attn_fa: [Func<'m>; 2],
    /// Without tensor cores: the split attention, decode and per query.
    attn_split: Func<'m>,
    attn_split_b: Func<'m>,
    /// prefill.cu's batched store.
    store_b: Func<'m>,
}

impl<'m> Q8Fns<'m> {
    /// group: values per scale, head_dim or 32.
    fn new(module: &'m Module<'m>, head_dim: usize, group: usize) -> Result<Q8Fns<'m>, String> {
        let q = if group == head_dim { "q8" } else { "q8g32" };
        let f = |s: &str| module.function(&format!("{s}_{q}_{head_dim}"));
        let fa = [f("attn_fa_b")?, f("attn_fa_bs")?];
        if module.mma() {
            for (small, fa) in fa.iter().enumerate() {
                fa.set_max_shared(batch::fa_shape(head_dim, small == 1, group).2)?;
            }
        }
        Ok(Q8Fns {
            attn_blk: f("attn_blk")?,
            attn_blk_b: f("attn_blk_b")?,
            attn_fa: fa,
            attn_split: f("attn_split")?,
            attn_split_b: f("attn_split_b")?,
            store_b: module.function(&format!("kv_{q}_b_{head_dim}"))?,
        })
    }
}

/// The decode kernels of a module.
struct Fns<'m> {
    rmsnorm_quant: Func<'m>,
    quantize: Func<'m>,
    embed: Func<'m>,
    rope_table: Func<'m>,
    gemv: Func<'m>,
    gemv1: Func<'m>,
    qkv_rope: Func<'m>,
    gate_up: Func<'m>,
    attn_split: Func<'m>,
    attn_combine: Func<'m>,
    /// The tensor-core attention (f16 cache, sm_80+; `tc_attn`): a warp
    /// per 64 positions, then the fold.
    attn_blk: Func<'m>,
    attn_cmb: Func<'m>,
    argmax: Func<'m>,
    /// The q8 cache's: qkv_rope with f32 keys and values out, and per scale
    /// group (head_dim, 32) the attention kernels.
    qkv_rope_f32: Func<'m>,
    q8: [Q8Fns<'m>; 2],
    /// sample.cu.
    samp_topk: Func<'m>,
}

/// Logits per samp_topk block, and its threads (sample.cu).
const TK_SLICE: usize = 4096;
const TK_THREADS: u32 = 512;

/// What a step (or batched pass) does after the layers: nothing (prefill),
/// the greedy pick, or sampling candidates (sample.cu). The step's graph
/// slot. `Last` (batched passes only): the logits of the pass's last token
/// alone, into the batch scratch's logits row 0 (`Lm::prefill_logits`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Head {
    No = 0,
    Argmax = 1,
    Sample = 2,
    Last = 3,
}

impl From<bool> for Head {
    fn from(head: bool) -> Head {
        if head { Head::Argmax } else { Head::No }
    }
}

/// A cache's scratch for sampling candidates, made on first use (a few
/// MiB, within CACHE_HEADROOM): the step's {token, pos} (16 bytes, read in
/// place of `CudaCache::params` by the sampling step) then sample.cu's S,
/// slice picks and merge counters for up to HEAD_SUB rows a launch, and
/// `rows` rows of CAND_MAX candidates out.
pub(super) struct SampBufs<'g> {
    /// u32s `s` holds (4 of params, then S: header and penalty ids).
    cap: usize,
    rows: usize,
    s: DevBuf<'g>,
    host_s: HostBuf<'g>,
    part: DevBuf<'g>,
    count: DevBuf<'g>,
    pub(super) out: DevBuf<'g>,
    pub(super) host_out: HostBuf<'g>,
    /// The {token, pos} the last upload carried.
    params: Option<(u32, usize)>,
}

pub struct CudaQwen2<'g> {
    gpu: &'g Gpu,
    /// Unique per model in the process: caches record it (`CudaCache::net`).
    id: u64,
    f: Fns<'g>,
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// n_heads * head_dim: queries and attention output (qwen3: != dim).
    pub q_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    eps: f32,
    /// Index of the first layer here: 0, or the first after those another
    /// backend runs (`new_layers`), whose hidden state comes in instead of
    /// a token (`Input::Hidden`).
    pub first: usize,
    /// Quantized matrices, embedding included (if first == 0).
    weights: DevBuf<'g>,
    /// The Q6_K layer matrices' prefill copy (`make_q6k8`; `Mat::c8`), or
    /// empty.
    w8: DevBuf<'g>,
    /// Norm weights and biases, f32.
    vecs: DevBuf<'g>,
    /// RoPE frequencies, f64 (base^(-2i/hd), as model.rs rope_table).
    freq: DevBuf<'g>,
    /// None when first > 0: tokens are embedded by the backend that runs
    /// the first layers.
    embd: Option<Mat>,
    output: Mat,
    output_norm: usize,
    /// Layers first..n_layers (index i is layer first + i).
    layers: Vec<LayerW>,
    /// Attention splits (grid y of attn_split; the q8 cache and GPUs
    /// without tensor cores), and the most blocks of an attn_blk launch.
    max_split: usize,
    /// The attention runs on the tensor cores (decode.cu "Tensor-core
    /// attention"), decode and batched alike, either cache: sm_80+.
    tc_attn: bool,
    /// Replay a captured graph per token (else launch kernel by kernel).
    pub graphs: bool,
    bf: batch::BatchFns<'g>,
    /// Most tokens per batched pass (`Lm::prefill_batch`); caches get
    /// scratch for that many (or their size, if smaller), and `vram_need`
    /// counts it. `PREFILL_BATCH`, or `SPLIT_BATCH` for a split's GPU half.
    batch: usize,
    /// How `new_cache` stores keys and values (set by `with_kv`; a test may
    /// change it between caches: each cache keeps its own).
    pub kv: CacheKv,
    /// The context `with_kv` planned memory for: its max_ctx, or less with
    /// `fit` (what's free then held).
    pub ctx: usize,
    /// Batched passes make the FFN's silu(gate) * up in the up product's
    /// epilogue (bgemm mode 2), else by swiglu_quant_b (tests compare them:
    /// the same bits).
    pub(super) swiglu_epi: bool,
    /// Batched passes run gate and up (both Q4_K) as one launch with the
    /// swiglu in its epilogue (prefill.cu gemm2s_q4_k_gu; AXON_GATE_UP=0
    /// turns it off): the same bits as separate products.
    pub(super) gate_up_fused: bool,
}

/// What `CudaQwen2` keeps per sequence: the KV cache, activations, logits,
/// the per-token parameter buffers, and the step's captured graphs.
pub struct CudaCache<'g> {
    /// The `CudaQwen2::id` of the model that made it: its graphs hold that
    /// model's weight addresses, so it only runs with that model.
    net: u64,
    max_ctx: usize,
    /// How kv stores keys and values.
    mode: CacheKv,
    /// F16: per layer, K then V, each [max_ctx][kv_dim] f16. Q8: per layer
    /// K then V int8 [max_ctx][kv_dim], then per layer K then V scales f32
    /// [max_ctx][kv_dim / group] (`kv_args`).
    kv: DevBuf<'g>,
    /// Q8: this token's keys and values, f32 [2][kv_dim], on their way into
    /// the cache (qkv_rope_f32 to attn_split_q8). Empty for F16.
    kvf: DevBuf<'g>,
    x: DevBuf<'g>,
    q: DevBuf<'g>,
    att: DevBuf<'g>,
    hid: DevBuf<'g>,
    act: DevBuf<'g>,
    part_o: DevBuf<'g>,
    part_ml: DevBuf<'g>,
    /// qwen3's q/k norms: this token's raw keys (kv_dim floats), from
    /// qkv_rope to the attention, which normalizes them.
    kraw: DevBuf<'g>,
    logits: DevBuf<'g>,
    best: DevBuf<'g>,
    /// This token's RoPE (sin, cos) per frequency.
    rope: DevBuf<'g>,
    /// {token, pos}, read by the kernels.
    params: DevBuf<'g>,
    host_params: HostBuf<'g>,
    host_out: HostBuf<'g>,
    /// The incoming hidden state (dim floats) when the model's first > 0.
    host_x: Option<HostBuf<'g>>,
    /// Captured steps, by `Head`: without the output head, with the
    /// argmax, with sampling candidates.
    graph: [Option<Graph<'g>>; 3],
    /// Scratch for batched passes (prefill, verify).
    b: Option<batch::BatchBufs<'g>>,
    /// Scratch for sampling candidates (`samp_upload`).
    samp: Option<SampBufs<'g>>,
}

/// Bytes of a quantized-activation buffer for n values (decode.cu Act).
fn act_bytes(n: usize) -> usize {
    let np = n.next_multiple_of(256);
    2 * np + np / 64 + np / 8 + np / 8
}

/// What `CudaQwen2::new_layers` uploads: each matrix's (tensor, arena
/// offset), the arena's size, norms and biases, RoPE frequencies.
struct Plan<'m> {
    uploads: Vec<(&'m Tensor, usize)>,
    total: usize,
    vecs: Vec<f32>,
    freq: Vec<f64>,
}

impl Plan<'_> {
    /// The VRAM bytes that don't depend on the context: the weight arena
    /// (padded) and the small vectors.
    fn fixed(&self) -> usize {
        self.total + ARENA_PAD + self.vecs.len() * 4 + self.freq.len() * 8
    }
}

/// What a step starts from: a token (embedded here, first == 0) or the
/// hidden state another backend's layers 0..first left (dim floats).
#[derive(Clone, Copy)]
pub(crate) enum Input<'x> {
    Token(u32),
    Hidden(&'x [f32]),
}

/// `t` placed in the weight arena after the last of `uploads` (256-byte
/// aligned), and recorded there.
fn place<'t>(t: &'t Tensor, uploads: &mut Vec<(&'t Tensor, usize)>) -> Mat {
    let cols = t.dims[0] as usize;
    let rows = t.n_elements as usize / cols;
    let rb = quant::row_bytes(t.qtype.id, cols);
    let off = uploads
        .last()
        .map_or(0, |&(u, o)| o + (u.n_bytes as usize).next_multiple_of(256));
    uploads.push((t, off));
    Mat {
        qt: t.qtype.id,
        off,
        rb,
        rows,
        cols,
        c8: None,
    }
}

/// Bytes per 256-value block of a Q6_K matrix's prefill copy (prefill.cu
/// Q6K8_BYTES).
const Q6K8_BYTES: usize = 288;

/// Whether a Q6_K prefill copy of `bytes` is made with `free` VRAM left
/// after the weights and a cache of `cache` bytes still to come: the cache,
/// `HEADROOM` and `Q6K8_MARGIN` stay free besides (with `force`, the cache
/// alone). A copy of 0 bytes (no Q6_K matrices) is never made.
fn q6k8_fits(free: u64, cache: u64, bytes: u64, force: bool) -> bool {
    let keep = cache + if force { 0 } else { HEADROOM + Q6K8_MARGIN };
    bytes > 0 && free >= keep + bytes
}

/// VRAM the Q6_K prefill copy must leave free beyond the planned cache and
/// `HEADROOM` (a second cache, the desktop's growth): it is made only from
/// memory nothing else was going to use.
const Q6K8_MARGIN: u64 = 512 << 20;

fn meta(m: &Model, key: &str) -> Option<f64> {
    crate::model::meta_num(m, &format!("{}.{key}", m.arch))
}

fn tensor<'a>(m: &'a Model, name: &str) -> Result<&'a Tensor, String> {
    m.tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("missing tensor {name}"))
}

fn tensor_opt<'a>(m: &'a Model, name: &str) -> Option<&'a Tensor> {
    m.tensors.iter().find(|t| t.name == name)
}

fn vec_f32(m: &Model, t: &Tensor) -> Vec<f32> {
    let mut v = vec![0f32; t.n_elements as usize];
    quant::dequant(t.qtype.id, m.tensor_data(t), &mut v);
    v
}

impl<'g> CudaQwen2<'g> {
    /// The model's weights on `gpu`, with decode.cu's kernels from `module`
    /// (`super::Kernels::module`), f16 caches. Err if the model isn't
    /// supported here or if the weights plus a cache for `max_ctx` positions
    /// don't fit in the free VRAM.
    #[cfg_attr(not(test), allow(dead_code))] // the tests'
    pub fn new(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &Model,
        max_ctx: usize,
    ) -> Result<CudaQwen2<'g>, String> {
        Self::with_kv(gpu, module, m, max_ctx, CacheKv::F16, false)
    }

    /// `new` with caches of type `kv`. With `fit`, a cache for max_ctx
    /// positions that doesn't fit isn't an error: `ctx` becomes the most
    /// (a multiple of 256) that does, if that's at least 256.
    pub fn with_kv(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &Model,
        max_ctx: usize,
        kv: CacheKv,
        fit: bool,
    ) -> Result<CudaQwen2<'g>, String> {
        Self::build(
            gpu,
            module,
            m,
            max_ctx,
            0,
            kv,
            SPLIT_BATCH,
            fit,
            &mut |_| {},
        )
    }

    /// `with_kv` (no `fit`) with layers first..n_layers only, plus the
    /// output norm and head (and the embedding if first == 0): the GPU half
    /// of a model split across devices (`hybrid`). The other layers'
    /// weights are never read or uploaded. With first > 0 every call takes
    /// the hidden state after layer first - 1 (`forward_hidden` & co.), not
    /// tokens, and the cache holds n_layers - first layers. `uploaded` sees
    /// each matrix right after its bytes are on the GPU (the hybrid hands
    /// its pages back to `m` there, so the GPU half never sits in RAM all
    /// at once). With first > 0, caches get batch scratch for
    /// `split_batch` tokens (`split_batch`: `SPLIT_BATCH`, or the streaming
    /// pass's to stream).
    #[allow(clippy::too_many_arguments)]
    pub fn new_layers<'m>(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &'m Model,
        max_ctx: usize,
        first: usize,
        kv: CacheKv,
        split_batch: usize,
        uploaded: &mut dyn FnMut(&'m Tensor),
    ) -> Result<CudaQwen2<'g>, String> {
        Self::build(
            gpu,
            module,
            m,
            max_ctx,
            first,
            kv,
            split_batch,
            false,
            uploaded,
        )
    }

    /// `new_layers` with `with_kv`'s `fit`.
    #[allow(clippy::too_many_arguments)]
    fn build<'m>(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &'m Model,
        max_ctx: usize,
        first: usize,
        kv: CacheKv,
        split_batch: usize,
        fit: bool,
        uploaded: &mut dyn FnMut(&'m Tensor),
    ) -> Result<CudaQwen2<'g>, String> {
        let (mut net, plan) = Self::plan(gpu, module, m, first, kv, split_batch)?;
        // Check the whole plan fits before allocating any of it.
        let free = gpu.info()?.free_bytes;
        let mut ctx = max_ctx;
        if fit {
            // cache_bytes grows with the context: the largest that fits.
            if let Some(c) = crate::hybrid::fit_ctx(max_ctx, free, |c| net.vram_want(&plan, c)) {
                ctx = c;
            }
        }
        let want = net.vram_want(&plan, ctx);
        if want > free {
            let q8 = if matches!(kv, CacheKv::F16) {
                ", --kv q8 (half the KV memory)"
            } else {
                ""
            };
            return Err(format!(
                "the model needs {:.2} GiB of GPU memory ({:.2} GiB weights, {:.2} GiB KV cache \
                 ({}) and scratch for {ctx} tokens), {:.2} GiB is free; try a smaller --ctx \
                 (or -n){q8} or the CPU",
                want as f64 / GIB,
                plan.total as f64 / GIB,
                net.cache_bytes(ctx) as f64 / GIB,
                net.kv_name(),
                free as f64 / GIB
            ));
        }
        net.ctx = ctx;
        if first == 0 {
            // The bigger passes only from VRAM the plan leaves (HEADROOM
            // kept); the context and placement were decided at PLAN_BATCH.
            net.batch = PREFILL_BATCH;
            if net.vram_want(&plan, ctx) > free {
                net.batch = PLAN_BATCH;
            }
        }
        // 16 bytes of slack: the 2-byte-aligned block loads (ld8_a2) read up
        // to 2 bytes past a Q8_0/Q5_0 block, which may be the arena's last.
        net.weights = gpu.alloc(plan.total + ARENA_PAD)?;
        // A thread fills in the page tables of the matrices ahead of the
        // copies (`Model::prefault`): out of a file mapping the driver's
        // pageable copy otherwise stops at a page fault every 64 KB (11.5
        // vs 14.3 GB/s, cuda::tests::h2d_bandwidth). It skips the ones
        // already being copied (`uploaded` may release them).
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|sc| {
            sc.spawn(|| {
                for (i, &(t, _)) in plan.uploads.iter().enumerate() {
                    if next.load(std::sync::atomic::Ordering::Relaxed) <= i {
                        m.prefault(t);
                    }
                }
            });
            let r = (|| {
                for (i, &(t, off)) in plan.uploads.iter().enumerate() {
                    next.store(i + 1, std::sync::atomic::Ordering::Relaxed);
                    net.weights.write_at(off, m.tensor_data(t))?; // synchronous
                    uploaded(t);
                }
                Ok::<(), String>(())
            })();
            next.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
            r
        })?;
        net.vecs = gpu.upload(&plan.vecs)?;
        net.freq = gpu.alloc(plan.freq.len() * 8)?;
        let fb: Vec<u64> = plan.freq.iter().map(|f| f.to_bits()).collect();
        net.freq.write(&fb)?;
        if first == 0 {
            net.make_q6k8(ctx)?;
        }
        Ok(net)
    }

    /// Bytes of the Q6_K layer matrices' prefill copy (prefill.cu Q6K8).
    fn q6k8_bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| [l.q, l.k, l.v, l.o, l.gate, l.up, l.down])
            .filter(|m| m.qt == Q6_K)
            .map(|m| m.rows * m.cols / 256 * Q6K8_BYTES)
            .sum()
    }

    /// The Q6_K layer matrices' prefill copy (int8 values, 16-byte aligned
    /// blocks: gemm2_q6k8 loads them without unpacking; the same products,
    /// bit for bit), if the GPU has the int8 tensor cores and VRAM to spare:
    /// what's free now must hold it, a cache of `ctx` positions,
    /// `HEADROOM` and `Q6K8_MARGIN` besides, so it never costs context
    /// (`vram_want` and the split's placement don't count it: it is made
    /// only from VRAM they leave unused, and never for a split's GPU half).
    /// AXON_Q6K8=0 never makes it, =1 whenever it fits at all. Decode
    /// keeps reading the Q6_K bytes.
    fn make_q6k8(&mut self, ctx: usize) -> Result<(), String> {
        let Some(conv) = &self.bf.q6k_to_i8 else {
            return Ok(());
        };
        let bytes = self.q6k8_bytes();
        let force = match std::env::var("AXON_Q6K8").as_deref() {
            Ok("0") => return Ok(()),
            Ok("1") => true,
            _ => false,
        };
        let free = self.gpu.info()?.free_bytes;
        if !q6k8_fits(free, self.cache_bytes(ctx) as u64, bytes as u64, force) {
            return Ok(());
        }
        let w8 = self.gpu.alloc(bytes)?;
        let mut off = 0;
        for l in self.layers.iter_mut() {
            for m in [
                &mut l.q,
                &mut l.k,
                &mut l.v,
                &mut l.o,
                &mut l.gate,
                &mut l.up,
                &mut l.down,
            ] {
                if m.qt != Q6_K {
                    continue;
                }
                let nblk = m.cols / 256;
                // SAFETY: q6k_to_i8's signature; the matrix is whole in the
                // arena, its copy (rows * nblk blocks) whole in w8 at off.
                unsafe {
                    self.gpu.launch(
                        conv,
                        (nblk as u32, m.rows as u32, 1),
                        (64, 1, 1),
                        0,
                        &[
                            self.weights.arg_at(m.off),
                            Arg::U64(m.rb as u64),
                            Arg::I32(nblk as i32),
                            w8.arg_at(off),
                        ],
                    )?;
                }
                m.c8 = Some(off);
                off += m.rows * nblk * Q6K8_BYTES;
            }
        }
        self.gpu.sync()?;
        self.w8 = w8;
        Ok(())
    }

    /// VRAM of the Q6_K prefill copy (0 if not made: `make_q6k8`).
    pub fn q6k8_vram(&self) -> usize {
        self.w8.len()
    }

    /// Drop the Q6_K prefill copy (tests: the same products without it).
    #[cfg(test)]
    pub(super) fn drop_q6k8(&mut self) -> Result<(), String> {
        for l in self.layers.iter_mut() {
            for m in [
                &mut l.q,
                &mut l.k,
                &mut l.v,
                &mut l.o,
                &mut l.gate,
                &mut l.up,
                &mut l.down,
            ] {
                m.c8 = None;
            }
        }
        self.w8 = self.gpu.alloc(0)?;
        Ok(())
    }

    /// VRAM `new_layers(.., max_ctx, first, kv, split_batch, _)` checks
    /// it has, without allocating anything: the split's placement
    /// (`hybrid::auto_gpu_layers`).
    pub fn vram_need(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &Model,
        max_ctx: usize,
        first: usize,
        kv: CacheKv,
        split_batch: usize,
    ) -> Result<u64, String> {
        let (net, plan) = Self::plan(gpu, module, m, first, kv, split_batch)?;
        Ok(net.vram_want(&plan, max_ctx))
    }

    /// `vram_need` of the whole model (first 0) as a function of max_ctx,
    /// planned once: what `Backend::new`'s placement weighs.
    pub fn vram_curve(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &Model,
        kv: CacheKv,
    ) -> Result<impl Fn(usize) -> u64 + use<'g>, String> {
        let (net, plan) = Self::plan(gpu, module, m, 0, kv, SPLIT_BATCH)?;
        let fixed = plan.fixed();
        Ok(move |ctx| net.vram_want_fixed(fixed, ctx))
    }

    /// Weights, small vectors, cache for `max_ctx` and `HEADROOM`, in bytes.
    fn vram_want(&self, plan: &Plan, max_ctx: usize) -> u64 {
        self.vram_want_fixed(plan.fixed(), max_ctx)
    }

    /// `vram_want` given the plan's `fixed()`.
    fn vram_want_fixed(&self, fixed: usize, max_ctx: usize) -> u64 {
        (fixed + self.cache_bytes(max_ctx)) as u64 + HEADROOM
    }

    /// The model with nothing allocated yet (caches of type `kv`), and what
    /// to upload.
    fn plan<'m>(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &'m Model,
        first: usize,
        kv: CacheKv,
        split_batch: usize,
    ) -> Result<(CudaQwen2<'g>, Plan<'m>), String> {
        crate::model::check_supported(m).map_err(|e| e.to_string())?;
        let need = |k: &str| {
            meta(m, k)
                .map(|v| v as usize)
                .ok_or_else(|| format!("missing metadata {k}"))
        };
        let n_layers = need("block_count")?;
        let dim = need("embedding_length")?;
        let n_heads = need("attention.head_count")?;
        let n_kv_heads = need("attention.head_count_kv")?;
        let head_dim = meta(m, "attention.key_length").map_or(dim / n_heads, |v| v as usize);
        let q_dim = n_heads * head_dim;
        let eps = meta(m, "attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32;
        let rope_base = meta(m, "rope.freq_base").unwrap_or(10000.0);
        let trained = meta(m, "context_length").unwrap_or(32768.0);
        let group = n_heads / n_kv_heads;
        if head_dim != 64 && head_dim != 128 {
            return Err(format!(
                "head size {head_dim} isn't supported on the GPU (64 or 128)"
            ));
        }
        if group > MAX_GROUP {
            return Err(format!(
                "{group} query heads per kv head isn't supported on the GPU (at most {MAX_GROUP})"
            ));
        }
        if first >= n_layers {
            return Err(format!(
                "layers {first}.. of {n_layers}: the GPU needs at least one"
            ));
        }

        // Every matrix's (tensor, arena offset), uploaded once all fit.
        let mut uploads: Vec<(&Tensor, usize)> = Vec::new();
        let gemv_type = |mt: &Mat, name: &str| -> Result<(), String> {
            if [Q4_K, Q6_K, Q5_0, Q8_0].contains(&mt.qt) {
                Ok(())
            } else {
                Err(format!(
                    "tensor {name}: type {} isn't supported by the CUDA backend \
                     (Q4_K, Q6_K, Q5_0, Q8_0)",
                    crate::gguf::qtype(mt.qt).name
                ))
            }
        };
        let embd_t = tensor(m, "token_embd.weight")?;
        let embd = (first == 0).then(|| place(embd_t, &mut uploads));
        let output = match (tensor_opt(m, "output.weight"), embd) {
            (Some(t), _) => place(t, &mut uploads),
            (None, Some(e)) => e,
            // Tied, and the embedding isn't here: upload it as the head.
            (None, None) => place(embd_t, &mut uploads),
        };
        let vocab = output.rows;
        gemv_type(&output, "output.weight")?;

        let mut vecs: Vec<f32> = Vec::new();
        let mut push_vec = |v: Vec<f32>| {
            let off = vecs.len();
            vecs.extend(v);
            vecs.resize(vecs.len().next_multiple_of(64), 0.0);
            off
        };
        let output_norm = push_vec(vec_f32(m, tensor(m, "output_norm.weight")?));
        let mut layers = Vec::with_capacity(n_layers - first);
        for i in first..n_layers {
            let name = |s: &str| format!("blk.{i}.{s}");
            let mut mat = |s: &str| -> Result<Mat, String> {
                let mt = place(tensor(m, &name(s))?, &mut uploads);
                gemv_type(&mt, &name(s))?;
                Ok(mt)
            };
            let (q, k, v, o) = (
                mat("attn_q.weight")?,
                mat("attn_k.weight")?,
                mat("attn_v.weight")?,
                mat("attn_output.weight")?,
            );
            let (gate, up, down) = (
                mat("ffn_gate.weight")?,
                mat("ffn_up.weight")?,
                mat("ffn_down.weight")?,
            );
            let mut opt = |s: &str| tensor_opt(m, &name(s)).map(|t| push_vec(vec_f32(m, t)));
            let (bq, bk, bv) = (opt("attn_q.bias"), opt("attn_k.bias"), opt("attn_v.bias"));
            let (qn, kn) = (opt("attn_q_norm.weight"), opt("attn_k_norm.weight"));
            if qn.is_some() != kn.is_some() {
                return Err(format!(
                    "blk.{i}: a q norm without a k norm (or the reverse)"
                ));
            }
            // The q8 cache stores keys minus the key bias b of the RoPE
            // pairs slow enough that the rotated bias R_t b stays within |b|
            // of b at every trained position t (angle theta t <= pi / 3):
            // Qwen2's huge key channels (|b| up to ~200, at least 10x the
            // rest) are such pairs. 0 for the others. A shift the same at
            // every position adds the same to all of a query's scores, so
            // softmax doesn't see it; quantizing the small remainder does.
            let ksh = tensor_opt(m, &name("attn_k.bias")).map(|t| {
                let b = vec_f32(m, t);
                let slow = |r: usize| {
                    let i = r % head_dim % (head_dim / 2);
                    let theta = rope_base.powf(-2.0 * i as f64 / head_dim as f64);
                    theta * trained <= std::f64::consts::FRAC_PI_3
                };
                push_vec(
                    b.iter()
                        .enumerate()
                        .map(|(r, &x)| if slow(r) { x } else { 0.0 })
                        .collect(),
                )
            });
            layers.push(LayerW {
                ksh,
                attn_norm: push_vec(vec_f32(m, tensor(m, &name("attn_norm.weight"))?)),
                bq,
                bk,
                bv,
                qn,
                kn,
                ffn_norm: push_vec(vec_f32(m, tensor(m, &name("ffn_norm.weight"))?)),
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
            });
        }
        let total = uploads
            .last()
            .map_or(0, |&(t, off)| off + t.n_bytes as usize);
        let ffn = layers[0].gate.rows;
        if layers.iter().any(|l| l.gate.rows != ffn) {
            return Err("layers of different FFN sizes aren't supported on the GPU".into());
        }
        let freq: Vec<f64> = (0..head_dim / 2)
            .map(|i| rope_base.powf(-2.0 * i as f64 / head_dim as f64))
            .collect();

        let sms = gpu.info()?.sms as usize;
        // Up to ~8 attention blocks per SM at long contexts.
        let max_split = (8 * sms / n_kv_heads).clamp(1, MAX_SPLIT);
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut net = CudaQwen2 {
            gpu,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            f: Fns {
                rmsnorm_quant: module.function("rmsnorm_quant")?,
                quantize: module.function("quantize")?,
                embed: module.function("embed")?,
                rope_table: module.function("rope_table")?,
                gemv: module.function("gemv")?,
                gemv1: module.function("gemv1")?,
                qkv_rope: module.function("qkv_rope")?,
                gate_up: module.function("gate_up")?,
                attn_split: module.function(if head_dim == 64 {
                    "attn_split_64"
                } else {
                    "attn_split_128"
                })?,
                attn_combine: module.function("attn_combine")?,
                attn_blk: module.function(if head_dim == 64 {
                    "attn_blk_64"
                } else {
                    "attn_blk_128"
                })?,
                attn_cmb: module.function("attn_cmb")?,
                argmax: module.function("argmax")?,
                qkv_rope_f32: module.function("qkv_rope_f32")?,
                q8: [
                    Q8Fns::new(module, head_dim, head_dim)?,
                    Q8Fns::new(module, head_dim, 32)?,
                ],
                samp_topk: module.function("samp_topk")?,
            },
            n_layers,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            q_dim,
            ffn,
            vocab,
            eps,
            first,
            weights: gpu.alloc(0)?,
            w8: gpu.alloc(0)?,
            vecs: gpu.alloc(0)?,
            freq: gpu.alloc(0)?,
            embd,
            output,
            output_norm,
            layers,
            max_split,
            tc_attn: module.mma(),
            graphs: std::env::var("AXON_CUDA_GRAPHS").map_or(true, |v| v != "0"),
            bf: batch::BatchFns::new(module, head_dim, sms)?,
            batch: if first > 0 { split_batch } else { PLAN_BATCH },
            kv,
            ctx: 0,
            swiglu_epi: true,
            gate_up_fused: std::env::var("AXON_GATE_UP").map_or(true, |v| v != "0"),
        };
        if let CacheKv::Q8 { group } = kv {
            if group == 0 {
                net.kv = CacheKv::Q8 { group: head_dim };
            } else if group != head_dim && group != 32 {
                return Err(format!(
                    "q8 KV scale groups of {group} values (head_dim or 32)"
                ));
            }
        }
        let plan = Plan {
            uploads,
            vecs,
            freq,
            total,
        };
        Ok((net, plan))
    }

    /// The whole KV cache's bytes (f16 bits), every layer here (tests).
    #[cfg(test)]
    pub(crate) fn kv_bits(c: &CudaCache) -> Vec<u16> {
        c.kv.read().unwrap()
    }

    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// "f16", "q8" (a scale per kv head row) or "q8/32" (per 32 values):
    /// what `new_cache` makes.
    pub fn kv_name(&self) -> &'static str {
        match self.kv {
            CacheKv::F16 => "f16",
            CacheKv::Q8 { group } if group == self.head_dim => "q8",
            CacheKv::Q8 { .. } => "q8/32",
        }
    }

    /// Bytes of the keys and values of a `kv` cache for max_ctx positions.
    pub fn kv_cache_bytes(&self, kv: CacheKv, max_ctx: usize) -> usize {
        let n = 2 * self.layers.len() * max_ctx * self.kv_dim();
        match kv {
            CacheKv::F16 => 2 * n,
            // group 0: one scale per kv head (`try_cache_kv`).
            CacheKv::Q8 { group } => n + n / if group == 0 { self.head_dim } else { group } * 4,
        }
    }

    /// Layer l's cache as kernel arguments: F16 [K, V]; Q8 [K, V, K scales,
    /// V scales] (layout at `CudaCache::kv`).
    fn kv_args(&self, c: &CudaCache, l: usize) -> Vec<Arg> {
        let n = c.max_ctx * self.kv_dim();
        match c.mode {
            CacheKv::F16 => vec![c.kv.arg_at(2 * l * n * 2), c.kv.arg_at((2 * l + 1) * n * 2)],
            CacheKv::Q8 { group } => {
                let s0 = 2 * self.layers.len() * n;
                let sn = n / group * 4;
                vec![
                    c.kv.arg_at(2 * l * n),
                    c.kv.arg_at((2 * l + 1) * n),
                    c.kv.arg_at(s0 + 2 * l * sn),
                    c.kv.arg_at(s0 + (2 * l + 1) * sn),
                ]
            }
        }
    }

    /// The q8 kernels for a cache of `mode` (Q8).
    fn q8_fns(&self, mode: CacheKv) -> &Q8Fns<'g> {
        match mode {
            CacheKv::Q8 { group } => &self.f.q8[(group != self.head_dim) as usize],
            CacheKv::F16 => unreachable!("q8 kernels for an f16 cache"),
        }
    }

    /// Blocks of ATT_BLK positions a cache of max_ctx has (the stride of
    /// attn_blk's partials).
    fn nblk_max(max_ctx: usize) -> usize {
        max_ctx.div_ceil(ATT_BLK)
    }

    /// Partials per head of a decode step's attention: a block's
    /// (attn_blk), or a split's (attn_split, without tensor cores).
    fn att_parts(&self, max_ctx: usize) -> usize {
        match self.tc_attn {
            true => Self::nblk_max(max_ctx),
            false => self.max_split,
        }
    }

    /// Tests: run the attention through the split kernels of GPUs without
    /// tensor cores (`tc_attn` false) on any GPU, before making caches
    /// (their scratch is sized for the path). Turning it on needs mma.
    #[cfg(test)]
    pub(crate) fn set_tc_attn(&mut self, on: bool) {
        assert!(
            !on || self.bf.has_mma(),
            "tensor-core attention without mma"
        );
        self.tc_attn = on;
    }

    /// Free VRAM now.
    pub fn free_bytes(&self) -> Result<u64, String> {
        Ok(self.gpu.info()?.free_bytes)
    }

    /// Most tokens per batched pass of cache `c`.
    pub fn batch_max(c: &CudaCache) -> usize {
        c.b.as_ref().map_or(1, |b| b.max)
    }

    /// Longest vector quantized into an activation buffer: the model width,
    /// the attention output (q_dim) or the FFN's hidden width.
    fn act_len(&self) -> usize {
        self.dim.max(self.q_dim).max(self.ffn)
    }

    /// Weight bytes in VRAM.
    pub fn weight_bytes(&self) -> usize {
        self.weights.len()
    }

    /// Bytes of matrices read per token with the output head (`head`) or
    /// without (the embedding row aside).
    pub fn bytes_per_token(&self, head: bool) -> usize {
        let l: usize = self
            .layers
            .iter()
            .map(|l| {
                [l.q, l.k, l.v, l.o, l.gate, l.up, l.down]
                    .iter()
                    .map(Mat::bytes)
                    .sum::<usize>()
            })
            .sum();
        l + if head { self.output.bytes() } else { 0 }
    }

    /// VRAM a cache for `max_ctx` positions takes. On a split's GPU half
    /// (first > 0) it also counts the pinned host buffers for incoming
    /// hidden states (`host_x` here and per batch token in `batch_bytes`),
    /// which live in RAM, not VRAM: a harmless overcount of
    /// (batch + 1) * dim * 4 bytes (~1.3 MB on 32B), kept so `vram_need`
    /// stays one formula.
    pub fn cache_bytes(&self, max_ctx: usize) -> usize {
        self.cache_bytes_kv(self.kv, max_ctx)
    }

    /// `cache_bytes` for a cache of type `kv`.
    fn cache_bytes_kv(&self, kv: CacheKv, max_ctx: usize) -> usize {
        let f = 4;
        let kvf = match kv {
            CacheKv::F16 => 0,
            CacheKv::Q8 { .. } => 2 * self.kv_dim() * f,
        };
        let kv = self.kv_cache_bytes(kv, max_ctx) + kvf;
        let host_x = if self.first > 0 { self.dim * f } else { 0 };
        let act = act_bytes(self.act_len());
        let parts = self.n_heads * self.att_parts(max_ctx) * (self.head_dim + 2) * f;
        kv + (self.dim + 2 * self.q_dim + self.kv_dim() + self.ffn + self.vocab + self.head_dim) * f
            + act
            + parts
            + 64
            + host_x
            + self.batch_bytes(self.batch_for(max_ctx), max_ctx)
    }

    /// A cache for positions 0..max_ctx, or why there's no room for it.
    pub fn try_cache(&self, max_ctx: usize) -> Result<CudaCache<'g>, String> {
        self.try_cache_kv(max_ctx, self.kv)
    }

    /// `try_cache` of type `mode` (rather than `kv`'s).
    pub fn try_cache_kv(&self, max_ctx: usize, mode: CacheKv) -> Result<CudaCache<'g>, String> {
        let g = self.gpu;
        let f = 4;
        let mode = match mode {
            CacheKv::Q8 { group: 0 } => CacheKv::Q8 {
                group: self.head_dim,
            },
            m => m,
        };
        if let CacheKv::Q8 { group } = mode {
            assert!(
                group == self.head_dim || group == 32,
                "q8 group {group} (head_dim or 32)"
            );
        }
        // `new` budgets one cache; each cache brings its own KV and batch
        // scratch (batch_bytes: ~275 MiB on the 7B at 1024 tokens and an 8k
        // context, ~360 MiB at 32k: the attention partials grow with it),
        // so every one is checked against what is free now, before any of
        // it is allocated.
        let want = self.cache_bytes_kv(mode, max_ctx) as u64 + CACHE_HEADROOM;
        let free = g.info()?.free_bytes;
        if want > free {
            return Err(format!(
                "a {max_ctx}-token cache needs {:.2} GiB of GPU memory (KV cache and scratch), \
                 {:.2} GiB is free; try a smaller --ctx (or -n){} or the CPU",
                want as f64 / GIB,
                free as f64 / GIB,
                if mode == CacheKv::F16 {
                    ", --kv q8"
                } else {
                    ""
                }
            ));
        }
        let kv = g.alloc(self.kv_cache_bytes(mode, max_ctx)).map_err(|e| {
            format!("no room for a {max_ctx}-token KV cache on the GPU ({e}); try a smaller --ctx")
        })?;
        let q8 = matches!(mode, CacheKv::Q8 { .. });
        let c = CudaCache {
            net: self.id,
            max_ctx,
            mode,
            kv,
            kvf: g.alloc(if q8 { 2 * self.kv_dim() * f } else { 0 })?,
            x: g.alloc(self.dim * f)?,
            q: g.alloc(self.q_dim * f)?,
            att: g.alloc(self.q_dim * f)?,
            hid: g.alloc(self.ffn * f)?,
            act: g.alloc(act_bytes(self.act_len()))?,
            part_o: g.alloc(self.n_heads * self.att_parts(max_ctx) * self.head_dim * f)?,
            part_ml: g.alloc(self.n_heads * self.att_parts(max_ctx) * 2 * f)?,
            kraw: g.alloc(self.kv_dim() * f)?,
            logits: g.alloc(self.vocab * f)?,
            best: g.alloc(4)?,
            rope: g.alloc(self.head_dim / 2 * 8)?,
            params: g.alloc(16)?,
            host_params: g.host_alloc(16)?,
            host_out: g.host_alloc(self.vocab * f)?,
            host_x: match self.first {
                0 => None,
                _ => Some(g.host_alloc(self.dim * f)?),
            },
            graph: [None, None, None],
            b: Some(self.batch_bufs(self.batch_for(max_ctx), max_ctx)?),
            samp: None,
        };
        // Stale KV rows are never read (attention stops at pos), but zeroed
        // memory keeps a bug from reading NaNs.
        c.kv.zero()?;
        Ok(c)
    }

    /// Launch every kernel of one step on the stream (no sync): the layers,
    /// and with a `head` the output norm, head and argmax (or sampling
    /// candidates). `on_launch(name, bytes)` runs after each launch (profiling).
    fn enqueue(
        &self,
        c: &CudaCache,
        head: Head,
        on_launch: &mut dyn FnMut(&'static str, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        let g = self.gpu;
        let f = &self.f;
        // {token, pos}: the sampling step's come with its S, in one copy.
        let params = match head {
            Head::Sample => c.samp.as_ref().expect("samp_upload first").s.arg(),
            _ => c.params.arg(),
        };
        let w = |mt: &Mat| self.weights.arg_at(mt.off);
        let vec = |off: usize| self.vecs.arg_at(off * 4);
        let null = Arg::Ptr(0);
        let (dim, hd, kvd) = (self.dim, self.head_dim, self.kv_dim());
        let want_k = |n: usize| Arg::I32(n.is_multiple_of(256) as i32);
        let blocks = |n: usize, per: usize| (n.div_ceil(per) as u32, 1, 1);
        let gemv_block = (GEMV_THREADS, 1, 1);
        // SAFETY (every launch below): arguments match decode.cu's kernel
        // signatures, and every buffer is sized for what the kernel reads
        // and writes with them: x holds dim floats, q/att q_dim, hid ffn,
        // act act_bytes(act_len()), kraw kv_dim, the KV cache max_ctx rows
        // per layer (P[pos] < max_ctx: `step` checks; q8: int8 rows and
        // their scales, kvf 2 kv_dim floats), part_* n_heads * att_parts
        // splits, logits vocab; matrices are whole in the arena.
        unsafe {
            // Without the embedding, x was copied in before the step and
            // rope_table is the first kernel: no PDL overlap to ask for.
            if let Some(embd) = &self.embd {
                g.launch(
                    &f.embed,
                    blocks(dim, 256),
                    (256, 1, 1),
                    0,
                    &[
                        Arg::I32(embd.qt as i32),
                        w(embd),
                        Arg::U64(embd.rb as u64),
                        Arg::I32(dim as i32),
                        params,
                        c.x.arg(),
                    ],
                )?;
                on_launch("embed", embd.rb)?;
            }
            g.launch_with(
                &f.rope_table,
                (1, 1, 1),
                ((hd / 2) as u32, 1, 1),
                0,
                &[
                    self.freq.arg(),
                    Arg::I32((hd / 2) as i32),
                    params,
                    c.rope.arg(),
                ],
                self.embd.is_some(),
            )?;
            on_launch("rope", 0)?;
            for (li, l) in self.layers.iter().enumerate() {
                let norm = |off: usize| {
                    self.run(
                        &f.rmsnorm_quant,
                        (1, 1, 1),
                        (1024, 1, 1),
                        0,
                        &[
                            c.x.arg(),
                            vec(off),
                            Arg::F32(self.eps),
                            Arg::I32(dim as i32),
                            want_k(dim),
                            c.act.arg(),
                            null,
                        ],
                    )
                };
                norm(l.attn_norm)?;
                on_launch("rmsnorm", 0)?;
                // Row pairs: query and key RoPE pairs, value rows two by two.
                let qd = self.q_dim;
                let qk_norm = l.qn.is_some();
                let items = qd / 2 + kvd / 2 + kvd / 2;
                let bias = |b: Option<usize>| b.map_or(null, vec);
                let kva = self.kv_args(c, li);
                // q8: keys and values to kvf, which attn_split_q8 stores.
                let (qkv_f, kout, kraw) = match c.mode {
                    CacheKv::F16 => (&f.qkv_rope, [kva[0], kva[1]], c.kraw.arg()),
                    CacheKv::Q8 { .. } => (
                        &f.qkv_rope_f32,
                        [c.kvf.arg(), c.kvf.arg_at(kvd * 4)],
                        c.kvf.arg(),
                    ),
                };
                self.run(
                    qkv_f,
                    (items as u32, 1, 1),
                    gemv_block,
                    0,
                    &[
                        Arg::I32(l.q.qt as i32),
                        w(&l.q),
                        Arg::U64(l.q.rb as u64),
                        Arg::I32(l.k.qt as i32),
                        w(&l.k),
                        Arg::U64(l.k.rb as u64),
                        Arg::I32(l.v.qt as i32),
                        w(&l.v),
                        Arg::U64(l.v.rb as u64),
                        bias(l.bq),
                        bias(l.bk),
                        bias(l.bv),
                        Arg::I32(qd as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(dim as i32),
                        c.act.arg(),
                        c.rope.arg(),
                        params,
                        c.q.arg(),
                        kout[0],
                        kout[1],
                        // Raw q and k for the attention's norms (qwen3):
                        // kraw, or kvf's keys on the q8 cache.
                        if qk_norm { kraw } else { null },
                    ],
                )?;
                on_launch("qkv", l.q.bytes() + l.k.bytes() + l.v.bytes())?;
                let tc = self.tc_attn;
                let nblk_max = Self::nblk_max(c.max_ctx);
                let mut args = vec![c.q.arg()];
                args.extend_from_slice(&kva);
                let attn = match c.mode {
                    CacheKv::F16 if tc => &f.attn_blk,
                    CacheKv::F16 => &f.attn_split,
                    CacheKv::Q8 { .. } => {
                        args.push(c.kvf.arg());
                        let q8 = self.q8_fns(c.mode);
                        if tc { &q8.attn_blk } else { &q8.attn_split }
                    }
                };
                args.extend_from_slice(&[
                    params,
                    Arg::I32(self.n_heads as i32),
                    Arg::I32(self.n_kv_heads as i32),
                    if tc {
                        Arg::F32(att_pre(hd))
                    } else {
                        Arg::F32(1.0 / (hd as f32).sqrt())
                    },
                    Arg::I32(if tc { nblk_max } else { self.max_split } as i32),
                    c.part_o.arg(),
                    c.part_ml.arg(),
                ]);
                if c.mode == CacheKv::F16 {
                    // qwen3's norms (attn_split_t's QkNorm), kc_out = K.
                    args.extend_from_slice(&[c.kraw.arg(), bias(l.qn)]);
                    args.extend_from_slice(&[bias(l.kn), Arg::F32(self.eps), c.rope.arg(), kva[0]]);
                } else {
                    args.push(bias(l.ksh));
                    args.extend_from_slice(&[bias(l.qn), bias(l.kn)]);
                    args.extend_from_slice(&[Arg::F32(self.eps), c.rope.arg()]);
                }
                // attn_blk: at most max_split warps per kv head (each takes
                // blocks y, y + max_split, ..), whatever the position.
                let (gy, bx) = match tc {
                    true => (self.max_split.min(nblk_max), 32),
                    false => (self.max_split, ATT_TILE as u32),
                };
                self.run(
                    attn,
                    (self.n_kv_heads as u32, gy as u32, 1),
                    (bx, 1, 1),
                    0,
                    &args,
                )?;
                on_launch("attn", 0)?;
                self.run(
                    if tc { &f.attn_cmb } else { &f.attn_combine },
                    blocks(qd, 256),
                    (256, 1, 1),
                    0,
                    &[
                        c.part_o.arg(),
                        c.part_ml.arg(),
                        params,
                        Arg::I32(qd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(if tc { nblk_max } else { self.max_split } as i32),
                        want_k(qd),
                        c.act.arg(),
                        c.att.arg(),
                    ],
                )?;
                on_launch("combine", 0)?;
                self.gemv(&l.o, &c.act, null, 1, &c.x)?;
                on_launch("wo", l.o.bytes())?;
                norm(l.ffn_norm)?;
                on_launch("rmsnorm", 0)?;
                self.run(
                    &f.gate_up,
                    (self.ffn as u32, 1, 1),
                    gemv_block,
                    0,
                    &[
                        Arg::I32(l.gate.qt as i32),
                        w(&l.gate),
                        Arg::U64(l.gate.rb as u64),
                        Arg::I32(l.up.qt as i32),
                        w(&l.up),
                        Arg::U64(l.up.rb as u64),
                        Arg::I32(self.ffn as i32),
                        Arg::I32(dim as i32),
                        c.act.arg(),
                        c.hid.arg(),
                    ],
                )?;
                on_launch("gate_up", l.gate.bytes() + l.up.bytes())?;
                self.run(
                    &f.quantize,
                    blocks(self.ffn, 256),
                    (32, 1, 1),
                    0,
                    &[
                        c.hid.arg(),
                        Arg::I32(self.ffn as i32),
                        want_k(self.ffn),
                        c.act.arg(),
                    ],
                )?;
                on_launch("quantize", 0)?;
                self.gemv(&l.down, &c.act, null, 1, &c.x)?;
                on_launch("down", l.down.bytes())?;
            }
            if head != Head::No {
                self.run(
                    &f.rmsnorm_quant,
                    (1, 1, 1),
                    (1024, 1, 1),
                    0,
                    &[
                        c.x.arg(),
                        vec(self.output_norm),
                        Arg::F32(self.eps),
                        Arg::I32(dim as i32),
                        want_k(dim),
                        c.act.arg(),
                        null,
                    ],
                )?;
                on_launch("rmsnorm", 0)?;
                self.gemv(&self.output, &c.act, null, 0, &c.logits)?;
                on_launch("head", self.output.bytes())?;
                if head == Head::Sample {
                    let s = c.samp.as_ref().expect("samp_upload before a sampling step");
                    self.samp_tail(s, c.logits.arg(), self.vocab, 1, 0, 0, true)?;
                    on_launch("sample", 0)?;
                } else {
                    self.run(
                        &f.argmax,
                        (1, 1, 1),
                        (1024, 1, 1),
                        0,
                        &[c.logits.arg(), Arg::I32(self.vocab as i32), c.best.arg()],
                    )?;
                    on_launch("argmax", 0)?;
                }
            }
        }
        Ok(())
    }

    /// `Gpu::launch` allowed to overlap the previous kernel's end (every
    /// decode.cu kernel but `embed` starts with `pdl_wait`).
    ///
    /// # Safety
    /// As `Gpu::launch`.
    unsafe fn run(
        &self,
        f: &Func,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        args: &[Arg],
    ) -> Result<(), String> {
        // SAFETY: the caller's.
        unsafe { self.gpu.launch_with(f, grid, block, shared, args, true) }
    }

    /// out = W act (mode 0, + bias if non-null) or out += W act (mode 1).
    ///
    /// # Safety
    /// act holds W's input quantized (decode.cu Act for mt.cols values), out
    /// has mt.rows floats, bias is null or has mt.rows floats.
    unsafe fn gemv(
        &self,
        mt: &Mat,
        act: &DevBuf,
        bias: Arg,
        mode: i32,
        out: &DevBuf,
    ) -> Result<(), String> {
        // Rows of 4 KiB or more (ffn_down in every model here: 4864+
        // inputs; no other matrix is that wide) get one row per block, so
        // their few rows still make enough blocks for a small last wave (+3%
        // on the 7B's down; same bits either way). The rest: two per block.
        let (f, per) = if mt.rb >= 4096 {
            (&self.f.gemv1, 1)
        } else {
            (&self.f.gemv, 2)
        };
        // SAFETY: the caller's contract; the matrix is whole in the arena.
        unsafe {
            self.run(
                f,
                (mt.rows.div_ceil(per) as u32, 1, 1),
                (GEMV_THREADS, 1, 1),
                0,
                &[
                    Arg::I32(mt.qt as i32),
                    self.weights.arg_at(mt.off),
                    Arg::U64(mt.rb as u64),
                    Arg::I32(mt.rows as i32),
                    Arg::I32(mt.cols as i32),
                    act.arg(),
                    bias,
                    Arg::I32(mode),
                    out.arg(),
                ],
            )
        }
    }

    /// Run `input` at `pos` (logits and argmax too with `head`), up to the
    /// point of reading results: enqueued, not waited for.
    fn step(
        &self,
        c: &mut CudaCache<'g>,
        input: Input,
        pos: usize,
        head: bool,
    ) -> Result<(), String> {
        self.step_with(c, input, pos, head.into())
    }

    /// `step` with any `Head` (Sample: after `samp_upload`).
    fn step_with(
        &self,
        c: &mut CudaCache<'g>,
        input: Input,
        pos: usize,
        head: Head,
    ) -> Result<(), String> {
        assert_eq!(c.net, self.id, "a CudaCache used with another CudaQwen2");
        assert!(head != Head::Last, "Head::Last: batched passes only");
        assert!(
            pos < c.max_ctx,
            "position {pos} exceeds cache size {}",
            c.max_ctx
        );
        let token = match input {
            Input::Token(t) => {
                assert!(
                    self.first == 0,
                    "layers {}..: takes hidden states",
                    self.first
                );
                t
            }
            Input::Hidden(x) => {
                assert!(self.first > 0, "takes tokens");
                assert_eq!(x.len(), self.dim, "hidden state size");
                // Ordered before the step on the stream (the graph reads c.x).
                let hx = c.host_x.as_mut().expect("first > 0: host_x");
                for (d, v) in hx.bytes_mut()?.chunks_exact_mut(4).zip(x) {
                    d.copy_from_slice(&v.to_le_bytes());
                }
                hx.copy_to(&c.x, 0, self.dim * 4)?;
                0
            }
        };
        assert!((token as usize) < self.vocab, "token {token} out of range");
        if head == Head::Sample {
            // samp_upload wrote and copied them, with S.
            let s = c.samp.as_ref().expect("samp_upload first");
            assert_eq!(s.params, Some((token, pos)), "samp_upload for another step");
        } else {
            let p = c.host_params.bytes_mut()?;
            p[0..4].copy_from_slice(&token.to_le_bytes());
            p[4..8].copy_from_slice(&(pos as u32).to_le_bytes());
            c.host_params.copy_to(&c.params, 0, 8)?;
        }
        if !self.graphs {
            return self.enqueue(c, head, &mut |_, _| Ok(()));
        }
        let slot = head as usize;
        if c.graph[slot].is_none() {
            let g = self
                .gpu
                .capture(|| self.enqueue(c, head, &mut |_, _| Ok(())))?;
            c.graph[slot] = Some(g);
        }
        // SAFETY: the graph's buffers are the cache's and the model's, both
        // borrowed for this call; the cache owns the graph, so it can't
        // outlive them.
        unsafe { c.graph[slot].as_ref().unwrap().launch() }
    }

    /// model.rs `hidden_batch`'s checks, with its messages.
    /// Also every token in range (as `step`): embed_b reads row
    /// tokens[z] of the embedding unchecked.
    fn check_batch(&self, tokens: &[u32], start: usize, c: &CudaCache) {
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.vocab) {
            panic!("token {t} out of range");
        }
        Self::check_positions(tokens.len(), start, c);
    }

    /// `check_batch` without tokens (hidden-state input: the CPU half
    /// that made it checked them).
    fn check_positions(nb: usize, start: usize, c: &CudaCache) {
        assert!(nb > 0, "feed_batch: empty batch");
        assert!(
            start + nb <= c.max_ctx,
            "positions {start}..{} exceed cache size {}",
            start + nb,
            c.max_ctx
        );
    }

    fn die(e: String) -> ! {
        panic!("CUDA: {e}")
    }

    /// Logits after `input` at `pos`.
    fn logits(&self, c: &mut CudaCache<'g>, input: Input, pos: usize) -> Result<Vec<f32>, String> {
        self.step(c, input, pos, true)?;
        let n = self.vocab * 4;
        c.host_out.copy_from(&c.logits, 0, n)?;
        let b = &c.host_out.bytes_mut()?[..n]; // waits for the step and copy
        Ok(b.chunks_exact(4)
            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect())
    }

    fn pick(&self, c: &mut CudaCache<'g>, input: Input, pos: usize) -> Result<u32, String> {
        self.step(c, input, pos, true)?;
        c.host_out.copy_from(&c.best, 0, 4)?;
        let b = &c.host_out.bytes_mut()?[..4]; // waits for the step and copy
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Candidates after `token` at `pos` (`Lm::forward_cands`).
    fn cands(
        &self,
        c: &mut CudaCache<'g>,
        token: u32,
        pos: usize,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Result<Vec<Cand>, String> {
        self.samp_upload(c, spec, recent, 1, 1, Some((token, pos)))?;
        self.step_with(c, Input::Token(token), pos, Head::Sample)?;
        Ok(self
            .samp_read(c, spec.k, self.vocab, 1)?
            .pop()
            .expect("one row"))
    }

    /// Get sampling scratch for `rows` rows of candidates (at least) and
    /// write S for `nb` rows whose tokens end `recent` (row b's penalty ids:
    /// the distinct tokens among the last `spec.last_n` up to its token,
    /// as `sample::cands` takes them), after `params` {token, pos} for a
    /// sampling step: one copy to the device on the stream.
    pub(super) fn samp_upload(
        &self,
        c: &mut CudaCache<'g>,
        spec: &CandSpec,
        recent: &[u32],
        nb: usize,
        rows: usize,
        params: Option<(u32, usize)>,
    ) -> Result<(), String> {
        assert!(
            spec.k >= 1 && spec.k <= CAND_MAX,
            "candidates per row: 1..={CAND_MAX}"
        );
        assert!(recent.len() >= nb, "recent must end with the rows' tokens");
        let penalize = spec.penalty != 1.0 && spec.last_n > 0;
        let n = recent.len();
        let ids: Vec<Vec<u32>> = (0..nb)
            .map(|b| {
                if penalize {
                    sample::penalty_ids(&recent[..n - nb + b + 1], spec.last_n)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let need = 4 + 4 + nb + ids.iter().map(Vec::len).sum::<usize>();
        let fits = c
            .samp
            .as_ref()
            .is_some_and(|s| s.cap >= need && s.rows >= rows);
        if !fits {
            // New buffers: the sampling graph holds the old ones' addresses.
            c.graph[Head::Sample as usize] = None;
            let old = c.samp.take();
            let cap = need.max(4 + 5 + 256).max(old.as_ref().map_or(0, |s| s.cap));
            let rows = rows.max(old.as_ref().map_or(1, |s| s.rows));
            let g = self.gpu;
            let slices = self.vocab.div_ceil(TK_SLICE);
            let count = g.alloc(batch::HEAD_SUB * 4)?;
            count.zero()?;
            c.samp = Some(SampBufs {
                cap,
                rows,
                s: g.alloc(cap * 4)?,
                host_s: g.host_alloc(cap * 4)?,
                part: g.alloc(batch::HEAD_SUB * slices * CAND_MAX * 8)?,
                count,
                params: None,
                out: g.alloc(rows * CAND_MAX * 8)?,
                host_out: g.host_alloc(rows * CAND_MAX * 8)?,
            });
        }
        let s = c.samp.as_mut().unwrap();
        let hb = s.host_s.bytes_mut()?;
        let mut put = |i: usize, v: u32| hb[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        s.params = params;
        let (token, pos) = params.unwrap_or((0, 0));
        put(0, token);
        put(1, pos as u32);
        // S, from u32 4 on.
        put(4, spec.k as u32);
        put(5, spec.penalty.to_bits());
        put(6, nb as u32);
        let mut off = 0;
        put(7, 0);
        let base = 8 + nb;
        for (b, w) in ids.iter().enumerate() {
            for (j, &id) in w.iter().enumerate() {
                assert!((id as usize) < self.vocab, "token {id} out of range");
                put(base + off + j, id);
            }
            off += w.len();
            put(8 + b, off as u32);
        }
        s.host_s.copy_to(&s.s, 0, need * 4)
    }

    /// Enqueue sample.cu on `rows` rows of `n` logits at `logits` (`n`
    /// floats apart; not modified): penalty row `row0` + z, candidates to
    /// out row `out_row0` + z. `pdl`: may overlap the previous kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn samp_tail(
        &self,
        s: &SampBufs,
        logits: Arg,
        n: usize,
        rows: usize,
        row0: usize,
        out_row0: usize,
        pdl: bool,
    ) -> Result<(), String> {
        assert!(n >= 1 && n <= self.vocab, "rows of 1..=vocab logits");
        assert!(rows <= batch::HEAD_SUB && out_row0 + rows <= s.rows);
        let slices = n.div_ceil(TK_SLICE) as u32;
        // SAFETY: logits holds `rows` rows of n floats; S (16 bytes into s)
        // was written by samp_upload for at least row0 + rows rows; part
        // has room for HEAD_SUB rows of `slices` x CAND_MAX keys (k <=
        // CAND_MAX, checked there), count for HEAD_SUB rows (zeroed at
        // allocation; each launch's last block per row wraps it back to 0), out for
        // s.rows rows of CAND_MAX pairs.
        unsafe {
            self.gpu.launch_with(
                &self.f.samp_topk,
                (slices, rows as u32, 1),
                (TK_THREADS, 1, 1),
                0,
                &[
                    logits,
                    Arg::U64(n as u64),
                    Arg::I32(n as i32),
                    s.s.arg_at(16),
                    Arg::I32(row0 as i32),
                    Arg::I32(out_row0 as i32),
                    s.part.arg(),
                    s.count.arg(),
                    s.out.arg(),
                ],
                pdl,
            )
        }
    }

    /// The `k` candidates (fewer if rows had `n` < k logits) of out rows
    /// 0..rows, after the step or pass that wrote them: waits for the stream.
    pub(super) fn samp_read(
        &self,
        c: &mut CudaCache<'g>,
        k: usize,
        n: usize,
        rows: usize,
    ) -> Result<Vec<Vec<Cand>>, String> {
        let s = c.samp.as_mut().expect("samp_upload first");
        let m = k.min(n);
        let n = (rows - 1) * k * 8 + m * 8;
        s.host_out.copy_from(&s.out, 0, n)?;
        let hb = &s.host_out.bytes_mut()?[..n]; // waits for the step and copy
        let u =
            |i: usize| u32::from_le_bytes([hb[4 * i], hb[4 * i + 1], hb[4 * i + 2], hb[4 * i + 3]]);
        Ok((0..rows)
            .map(|r| {
                (0..m)
                    .map(|j| Cand {
                        logit: f32::from_bits(u(2 * (r * k + j))),
                        id: u(2 * (r * k + j) + 1),
                    })
                    .collect()
            })
            .collect())
    }

    /// `token` into the cache at `pos` by the one-token step (the batched
    /// passes' reference in the tests; feed_batch of one token uses it too).
    fn feed(&self, c: &mut CudaCache<'g>, token: u32, pos: usize) -> Result<(), String> {
        self.step(c, Input::Token(token), pos, false)?;
        self.gpu.sync()
    }

    /// `Lm::forward` from the hidden state after layer first - 1 (first > 0).
    pub fn forward_hidden(&self, c: &mut CudaCache<'g>, x: &[f32], pos: usize) -> Vec<f32> {
        self.logits(c, Input::Hidden(x), pos)
            .unwrap_or_else(|e| Self::die(e))
    }

    /// `Lm::forward_argmax` from a hidden state (first > 0).
    pub fn argmax_hidden(&self, c: &mut CudaCache<'g>, x: &[f32], pos: usize) -> u32 {
        self.pick(c, Input::Hidden(x), pos)
            .unwrap_or_else(|e| Self::die(e))
    }

    /// `Lm::feed_batch` from hidden states, xs[b * dim..][..dim] at
    /// start + b (first > 0); one state goes through the step's graph.
    pub fn feed_hidden(&self, c: &mut CudaCache<'g>, xs: &[f32], start: usize) {
        let nb = xs.len() / self.dim;
        Self::check_positions(nb, start, c);
        assert_eq!(xs.len(), nb * self.dim, "hidden states: whole rows");
        let r = if nb == 1 {
            self.step(c, Input::Hidden(xs), start, false)
                .and_then(|_| self.gpu.sync())
        } else {
            self.feed_tokens(c, batch::BatchIn::Hidden(xs), start)
        };
        r.unwrap_or_else(|e| Self::die(e));
    }

    /// `Lm::verify_argmax` from hidden states, laid out as `feed_hidden`'s.
    pub fn verify_hidden(&self, c: &mut CudaCache<'g>, xs: &[f32], start: usize) -> Vec<u32> {
        let nb = xs.len() / self.dim;
        Self::check_positions(nb, start, c);
        assert_eq!(xs.len(), nb * self.dim, "hidden states: whole rows");
        self.verify_tokens(c, batch::BatchIn::Hidden(xs), start)
            .unwrap_or_else(|e| Self::die(e))
    }

    /// Per-kernel GPU time of one step (no graph), in first-launch order,
    /// plus the step's total ms. The events recorded between kernels also
    /// serialize them (no PDL overlap), so this is each kernel alone.
    #[cfg_attr(not(test), allow(dead_code))] // the profiling benchmark
    pub fn profile(
        &self,
        c: &mut CudaCache<'g>,
        token: u32,
        pos: usize,
    ) -> Result<(Vec<KernelTime>, f64), String> {
        assert_eq!(c.net, self.id, "a CudaCache used with another CudaQwen2");
        assert!(pos < c.max_ctx && (token as usize) < self.vocab);
        let p = c.host_params.bytes_mut()?;
        p[0..4].copy_from_slice(&token.to_le_bytes());
        p[4..8].copy_from_slice(&(pos as u32).to_le_bytes());
        c.host_params.copy_to(&c.params, 0, 8)?;
        let mut evs = vec![("start", self.gpu.record()?, 0)];
        self.enqueue(c, Head::Argmax, &mut |name, bytes| {
            evs.push((name, self.gpu.record()?, bytes));
            Ok(())
        })?;
        self.gpu.sync()?;
        let mut out: Vec<KernelTime> = Vec::new();
        for i in 1..evs.len() {
            let ms = evs[i - 1].1.ms_to(&evs[i].1)? as f64;
            let (name, bytes) = (evs[i].0, evs[i].2);
            match out.iter_mut().find(|o| o.name == name) {
                Some(o) => {
                    o.ms += ms;
                    o.launches += 1;
                    o.bytes += bytes;
                }
                None => out.push(KernelTime {
                    name,
                    ms,
                    launches: 1,
                    bytes,
                }),
            }
        }
        let total = evs[0].1.ms_to(&evs[evs.len() - 1].1)? as f64;
        Ok((out, total))
    }
}

/// One kernel's share of a profiled step (`CudaQwen2::profile`).
#[cfg_attr(not(test), allow(dead_code))] // the profiling benchmark
#[derive(Clone, Debug)]
pub struct KernelTime {
    pub name: &'static str,
    /// GPU time over all its launches.
    pub ms: f64,
    pub launches: usize,
    /// Matrix bytes those launches read (0 for non-GEMV kernels).
    pub bytes: usize,
}

impl<'g> Lm for CudaQwen2<'g> {
    type Cache = CudaCache<'g>;

    fn vocab(&self) -> usize {
        self.vocab
    }

    fn new_cache(&self, max_ctx: usize) -> CudaCache<'g> {
        self.try_cache(max_ctx).unwrap_or_else(|e| Self::die(e))
    }

    fn forward(&self, token: u32, pos: usize, cache: &mut CudaCache<'g>) -> Vec<f32> {
        self.logits(cache, Input::Token(token), pos)
            .unwrap_or_else(|e| Self::die(e))
    }

    fn forward_argmax(&self, token: u32, pos: usize, cache: &mut CudaCache<'g>) -> u32 {
        self.pick(cache, Input::Token(token), pos)
            .unwrap_or_else(|e| Self::die(e))
    }

    /// Batched passes (batch.rs): the same cache bits as `feed` per token.
    /// Checked up front as the CPU's: a non-empty batch that fits in the cache.
    fn feed_batch(&self, tokens: &[u32], start: usize, cache: &mut CudaCache<'g>) {
        self.check_batch(tokens, start, cache);
        assert_eq!(
            cache.net, self.id,
            "a CudaCache used with another CudaQwen2"
        );
        let r = if tokens.len() == 1 {
            self.feed(cache, tokens[0], start) // the step's graph: fewer launches
        } else {
            self.feed_tokens(cache, batch::BatchIn::Tokens(tokens), start)
        };
        r.unwrap_or_else(|e| Self::die(e));
    }

    /// Batched passes, the last token in the final one with the output head
    /// on its row alone: the KV cache and logit bits of `feed_batch` +
    /// `forward` (batch.rs contract), without the extra decode step.
    fn prefill_logits(&self, tokens: &[u32], start: usize, cache: &mut CudaCache<'g>) -> Vec<f32> {
        self.check_batch(tokens, start, cache);
        assert_eq!(
            cache.net, self.id,
            "a CudaCache used with another CudaQwen2"
        );
        if tokens.len() == 1 {
            return Lm::forward(self, tokens[0], start, cache);
        }
        self.prefill_tokens(cache, tokens, start)
            .unwrap_or_else(|e| Self::die(e))
    }

    /// Batched passes with the output head: the picks `forward_argmax`
    /// gives per token, in order (same logit bits, same tie rule).
    fn verify_argmax(&self, tokens: &[u32], start: usize, cache: &mut CudaCache<'g>) -> Vec<u32> {
        self.check_batch(tokens, start, cache);
        assert_eq!(
            cache.net, self.id,
            "a CudaCache used with another CudaQwen2"
        );
        self.verify_tokens(cache, batch::BatchIn::Tokens(tokens), start)
            .unwrap_or_else(|e| Self::die(e))
    }

    /// Selected on the device (sample.cu): k pairs come back, not the
    /// logits. Past CAND_MAX (top_k off: the whole vocab) the logits come
    /// back and the host selects (`sample::cands`).
    fn forward_cands(
        &self,
        token: u32,
        pos: usize,
        cache: &mut CudaCache<'g>,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Vec<Cand> {
        assert_eq!(
            recent.last(),
            Some(&token),
            "recent must end with the token"
        );
        if spec.k > CAND_MAX {
            let mut l = Lm::forward(self, token, pos, cache);
            return sample::cands(&mut l, spec, recent);
        }
        self.cands(cache, token, pos, spec, recent)
            .unwrap_or_else(|e| Self::die(e))
    }

    /// Batched passes with sample.cu on each row's logits (past CAND_MAX:
    /// `forward_cands` token by token, the same logits bit for bit).
    fn verify_cands(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &mut CudaCache<'g>,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Vec<Vec<Cand>> {
        self.check_batch(tokens, start, cache);
        assert_eq!(
            cache.net, self.id,
            "a CudaCache used with another CudaQwen2"
        );
        let (nb, n) = (tokens.len(), recent.len());
        assert!(
            n >= nb && recent[n - nb..] == *tokens,
            "recent must end with tokens"
        );
        if spec.k > CAND_MAX {
            return (0..nb)
                .map(|b| {
                    let r = &recent[..n - nb + b + 1];
                    self.forward_cands(tokens[b], start + b, cache, spec, r)
                })
                .collect();
        }
        self.verify_cands_tokens(cache, tokens, start, spec, recent)
            .unwrap_or_else(|e| Self::die(e))
    }

    fn prefill_batch(&self) -> usize {
        self.batch
    }
}

mod batch;
#[cfg(test)]
mod sample_tests;
pub(crate) mod stream;
#[cfg(test)]
pub(crate) mod tests;
