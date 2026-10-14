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

/// Positions per attention tile (decode.cu ATT_TILE) and largest GQA group.
const ATT_TILE: usize = 128;
const MAX_GROUP: usize = 8;
/// Most attention splits (decode.cu MAX_SPLIT).
const MAX_SPLIT: usize = 256;
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
const PREFILL_BATCH: usize = 512;
/// `PREFILL_BATCH` for the GPU half of a split (first > 0, `crate::hybrid`):
/// the CPU half's batched pass sets the pace, so the hybrid feeds at most
/// `crate::PREFILL_BATCH` tokens per call, and a speculative verify pass
/// takes K + 1 (8 at --draft 7; bigger ones are chunked, same bits). The
/// scratch for 512 would cost 160 MiB more on 32B (99 MiB on 14B) that the
/// split's placement (`vram_need`) counts against layers. Raise it (the
/// field `batch`, set from this in `plan`) if the GPU half is ever fed
/// bigger chunks, e.g. a prefill whose CPU layers are streamed to the GPU.
/// Placement stays monotone: first == 0 (the whole model) gets the bigger
/// `PREFILL_BATCH` scratch on top of one more layer.
pub const SPLIT_BATCH: usize = crate::PREFILL_BATCH;
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
}

impl Mat {
    fn bytes(&self) -> usize {
        self.rb * self.rows
    }
}

/// One layer: matrices, and offsets (in floats) into the vector buffer.
struct LayerW {
    attn_norm: usize,
    bq: Option<usize>,
    bk: Option<usize>,
    bv: Option<usize>,
    ffn_norm: usize,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
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
    argmax: Func<'m>,
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
    pub ffn: usize,
    pub vocab: usize,
    eps: f32,
    /// Index of the first layer here: 0, or the first after those another
    /// backend runs (`new_layers`), whose hidden state comes in instead of
    /// a token (`Input::Hidden`).
    pub first: usize,
    /// Quantized matrices, embedding included (if first == 0).
    weights: DevBuf<'g>,
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
    /// Attention splits (grid y of attn_split).
    max_split: usize,
    /// Replay a captured graph per token (else launch kernel by kernel).
    pub graphs: bool,
    bf: batch::BatchFns<'g>,
    /// Most tokens per batched pass (`Lm::prefill_batch`); caches get
    /// scratch for that many (or their size, if smaller), and `vram_need`
    /// counts it. `PREFILL_BATCH`, or `SPLIT_BATCH` for a split's GPU half.
    batch: usize,
}

/// What `CudaQwen2` keeps per sequence: the KV cache, activations, logits,
/// the per-token parameter buffers, and the step's captured graphs.
pub struct CudaCache<'g> {
    /// The `CudaQwen2::id` of the model that made it: its graphs hold that
    /// model's weight addresses, so it only runs with that model.
    net: u64,
    max_ctx: usize,
    /// Per layer, K then V, each [max_ctx][kv_dim] f16.
    kv: DevBuf<'g>,
    x: DevBuf<'g>,
    q: DevBuf<'g>,
    att: DevBuf<'g>,
    hid: DevBuf<'g>,
    act: DevBuf<'g>,
    part_o: DevBuf<'g>,
    part_ml: DevBuf<'g>,
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
    /// Captured steps: [without the output head, with it].
    graph: [Option<Graph<'g>>; 2],
    /// Scratch for batched passes (prefill, verify).
    b: Option<batch::BatchBufs<'g>>,
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
    }
}

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
    /// (`super::Kernels::module`). Err if the model isn't supported here or
    /// if the weights plus a cache for `max_ctx` positions don't fit in the
    /// free VRAM.
    pub fn new(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &Model,
        max_ctx: usize,
    ) -> Result<CudaQwen2<'g>, String> {
        Self::new_layers(gpu, module, m, max_ctx, 0, &mut |_| {})
    }

    /// `new` with layers first..n_layers only, plus the output norm and
    /// head (and the embedding if first == 0): the GPU half of a model
    /// split across devices (`hybrid`). The other layers' weights are never
    /// read or uploaded. With first > 0 every call takes the hidden state
    /// after layer first - 1 (`forward_hidden` & co.), not tokens, and the
    /// cache holds n_layers - first layers. `uploaded` sees each matrix
    /// right after its bytes are on the GPU (the hybrid hands its pages
    /// back to `m` there, so the GPU half never sits in RAM all at once).
    pub fn new_layers<'m>(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &'m Model,
        max_ctx: usize,
        first: usize,
        uploaded: &mut dyn FnMut(&'m Tensor),
    ) -> Result<CudaQwen2<'g>, String> {
        let (mut net, plan) = Self::plan(gpu, module, m, first)?;
        let want = net.vram_want(&plan, max_ctx);
        let free = gpu.info()?.free_bytes;
        if want > free {
            return Err(format!(
                "the model needs {:.2} GiB of GPU memory ({:.2} GiB weights, {:.2} GiB KV cache \
                 and scratch for {max_ctx} tokens), {:.2} GiB is free; try a smaller --ctx (or -n) or the CPU",
                want as f64 / GIB,
                plan.total as f64 / GIB,
                net.cache_bytes(max_ctx) as f64 / GIB,
                free as f64 / GIB
            ));
        }
        // 16 bytes of slack: the 2-byte-aligned block loads (ld8_a2) read up
        // to 2 bytes past a Q8_0/Q5_0 block, which may be the arena's last.
        net.weights = gpu.alloc(plan.total + ARENA_PAD)?;
        for (t, off) in plan.uploads {
            net.weights.write_at(off, m.tensor_data(t))?; // synchronous
            uploaded(t);
        }
        net.vecs = gpu.upload(&plan.vecs)?;
        net.freq = gpu.alloc(plan.freq.len() * 8)?;
        let fb: Vec<u64> = plan.freq.iter().map(|f| f.to_bits()).collect();
        net.freq.write(&fb)?;
        Ok(net)
    }

    /// VRAM `new_layers(.., max_ctx, first)` checks it has, without
    /// allocating anything: the split's placement (`hybrid::gpu_layers`).
    pub fn vram_need(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &Model,
        max_ctx: usize,
        first: usize,
    ) -> Result<u64, String> {
        let (net, plan) = Self::plan(gpu, module, m, first)?;
        Ok(net.vram_want(&plan, max_ctx))
    }

    /// Weights, small vectors, cache for `max_ctx` and `HEADROOM`, in bytes.
    fn vram_want(&self, plan: &Plan, max_ctx: usize) -> u64 {
        let small = plan.vecs.len() * 4 + plan.freq.len() * 8;
        (plan.total + ARENA_PAD + small + self.cache_bytes(max_ctx)) as u64 + HEADROOM
    }

    /// The model with nothing allocated yet, and what to upload.
    fn plan<'m>(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        m: &'m Model,
        first: usize,
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
        let head_dim = dim / n_heads;
        let eps = meta(m, "attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32;
        let rope_base = meta(m, "rope.freq_base").unwrap_or(10000.0);
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
            layers.push(LayerW {
                attn_norm: push_vec(vec_f32(m, tensor(m, &name("attn_norm.weight"))?)),
                bq,
                bk,
                bv,
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
        let net = CudaQwen2 {
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
                argmax: module.function("argmax")?,
            },
            n_layers,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn,
            vocab,
            eps,
            first,
            weights: gpu.alloc(0)?,
            vecs: gpu.alloc(0)?,
            freq: gpu.alloc(0)?,
            embd,
            output,
            output_norm,
            layers,
            max_split,
            graphs: std::env::var("AXON_CUDA_GRAPHS").map_or(true, |v| v != "0"),
            bf: batch::BatchFns::new(module, head_dim, sms)?,
            batch: if first > 0 {
                SPLIT_BATCH
            } else {
                PREFILL_BATCH
            },
        };
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
        let f = 4;
        let kv = 2 * self.layers.len() * max_ctx * self.kv_dim() * 2;
        let host_x = if self.first > 0 { self.dim * f } else { 0 };
        let act = act_bytes(self.dim.max(self.ffn));
        let parts = self.n_heads * self.max_split * (self.head_dim + 2) * f;
        kv + (3 * self.dim + self.ffn + self.vocab + self.head_dim) * f
            + act
            + parts
            + 64
            + host_x
            + self.batch_bytes(self.batch_for(max_ctx))
    }

    /// A cache for positions 0..max_ctx, or why there's no room for it.
    pub fn try_cache(&self, max_ctx: usize) -> Result<CudaCache<'g>, String> {
        let g = self.gpu;
        let f = 4;
        // `new` budgets one cache; each cache brings its own KV and batch
        // scratch (up to ~200 MiB on the 7B), so every one is checked
        // against what is free now, before any of it is allocated.
        let want = self.cache_bytes(max_ctx) as u64 + CACHE_HEADROOM;
        let free = g.info()?.free_bytes;
        if want > free {
            return Err(format!(
                "a {max_ctx}-token cache needs {:.2} GiB of GPU memory (KV cache and scratch), \
                 {:.2} GiB is free; try a smaller --ctx (or -n) or the CPU",
                want as f64 / GIB,
                free as f64 / GIB
            ));
        }
        let kv = g
            .alloc(2 * self.layers.len() * max_ctx * self.kv_dim() * 2)
            .map_err(|e| {
                format!(
                    "no room for a {max_ctx}-token KV cache on the GPU ({e}); try a smaller --ctx"
                )
            })?;
        let c = CudaCache {
            net: self.id,
            max_ctx,
            kv,
            x: g.alloc(self.dim * f)?,
            q: g.alloc(self.dim * f)?,
            att: g.alloc(self.dim * f)?,
            hid: g.alloc(self.ffn * f)?,
            act: g.alloc(act_bytes(self.dim.max(self.ffn)))?,
            part_o: g.alloc(self.n_heads * self.max_split * self.head_dim * f)?,
            part_ml: g.alloc(self.n_heads * self.max_split * 2 * f)?,
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
            graph: [None, None],
            b: Some(self.batch_bufs(self.batch_for(max_ctx))?),
        };
        // Stale KV rows are never read (attention stops at pos), but zeroed
        // memory keeps a bug from reading NaNs.
        c.kv.zero()?;
        Ok(c)
    }

    /// K (`v` false) or V cache of layer l: byte offset in `kv`.
    fn kv_off(&self, c: &CudaCache, l: usize, v: bool) -> usize {
        (2 * l + v as usize) * c.max_ctx * self.kv_dim() * 2
    }

    /// Launch every kernel of one step on the stream (no sync): the layers,
    /// and with `head` the output norm, head and argmax. `on_launch(name,
    /// bytes)` runs after each launch (profiling).
    fn enqueue(
        &self,
        c: &CudaCache,
        head: bool,
        on_launch: &mut dyn FnMut(&'static str, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        let g = self.gpu;
        let f = &self.f;
        let w = |mt: &Mat| self.weights.arg_at(mt.off);
        let vec = |off: usize| self.vecs.arg_at(off * 4);
        let null = Arg::Ptr(0);
        let (dim, hd, kvd) = (self.dim, self.head_dim, self.kv_dim());
        let want_k = |n: usize| Arg::I32(n.is_multiple_of(256) as i32);
        let blocks = |n: usize, per: usize| (n.div_ceil(per) as u32, 1, 1);
        let gemv_block = (GEMV_THREADS, 1, 1);
        // SAFETY (every launch below): arguments match decode.cu's kernel
        // signatures, and every buffer is sized for what the kernel reads
        // and writes with them: x/q/att hold dim floats, hid ffn, act
        // act_bytes(max(dim, ffn)), the KV cache max_ctx rows per layer
        // (P[pos] < max_ctx: `step` checks), part_* n_heads * max_split
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
                        c.params.arg(),
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
                    c.params.arg(),
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
                let items = dim / 2 + kvd / 2 + kvd / 2;
                let bias = |b: Option<usize>| b.map_or(null, vec);
                self.run(
                    &f.qkv_rope,
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
                        Arg::I32(dim as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(dim as i32),
                        c.act.arg(),
                        c.rope.arg(),
                        c.params.arg(),
                        c.q.arg(),
                        c.kv.arg_at(self.kv_off(c, li, false)),
                        c.kv.arg_at(self.kv_off(c, li, true)),
                    ],
                )?;
                on_launch("qkv", l.q.bytes() + l.k.bytes() + l.v.bytes())?;
                self.run(
                    &f.attn_split,
                    (self.n_kv_heads as u32, self.max_split as u32, 1),
                    (ATT_TILE as u32, 1, 1),
                    0,
                    &[
                        c.q.arg(),
                        c.kv.arg_at(self.kv_off(c, li, false)),
                        c.kv.arg_at(self.kv_off(c, li, true)),
                        c.params.arg(),
                        Arg::I32(self.n_heads as i32),
                        Arg::I32(self.n_kv_heads as i32),
                        Arg::F32(1.0 / (hd as f32).sqrt()),
                        Arg::I32(self.max_split as i32),
                        c.part_o.arg(),
                        c.part_ml.arg(),
                    ],
                )?;
                on_launch("attn", 0)?;
                self.run(
                    &f.attn_combine,
                    blocks(dim, 256),
                    (256, 1, 1),
                    0,
                    &[
                        c.part_o.arg(),
                        c.part_ml.arg(),
                        c.params.arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(self.max_split as i32),
                        want_k(dim),
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
            if head {
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
        assert_eq!(c.net, self.id, "a CudaCache used with another CudaQwen2");
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
        let p = c.host_params.bytes_mut()?;
        p[0..4].copy_from_slice(&token.to_le_bytes());
        p[4..8].copy_from_slice(&(pos as u32).to_le_bytes());
        c.host_params.copy_to(&c.params, 0, 8)?;
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
        self.enqueue(c, true, &mut |name, bytes| {
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

    fn prefill_batch(&self) -> usize {
        self.batch
    }
}

mod batch;
#[cfg(test)]
pub(crate) mod tests;
