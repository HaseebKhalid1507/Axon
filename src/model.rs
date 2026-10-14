//! qwen2 forward pass. Matrix-vector products quantize the input to 8 bits and
//! run integer dot products on the worker pool (pool.rs); everything else is
//! plain f32 loops.

use crate::gguf::{Model, Tensor, Val};
use crate::kv::{Half, KvData, KvElem, Layers};
use crate::pool::{self, Pool};
use crate::quant::{
    DecodeCfg, F16, F32, Q4_K, Q5_0, Q6_K, Q8_0, QuantizedAct, batch_slots, dequant,
    dot_row_q8_multi, dot_rows_q8, dot_rows_q8_multi, r8, r8q, row_bytes,
};
use crate::vmath;

/// A weight matrix: its bytes in the model's mapping, or its 8-row
/// interleaved repack (quant::r8 for Q4_K/Q6_K, quant::r8q for Q5_0/Q8_0).
/// A repacked matrix goes through those kernels in every path (decode,
/// prefill, swiglu, the verify head), so all of them give the same bits; the
/// others use the per-row kernels.
struct Weight<'a> {
    t: &'a Tensor,
    /// The tensor's bytes; None when repacked (they may have been released).
    data: Option<&'a [u8]>,
    r8: Option<R8>,
}

/// A repacked matrix: bytes off..off + len of the shared arena.
struct R8 {
    map: std::sync::Arc<memmap2::Mmap>,
    off: usize,
    len: usize,
}

impl std::ops::Deref for R8 {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.map[self.off..self.off + self.len]
    }
}

/// Bytes of the r8/r8q form of `t`, if it has one.
fn repacked_len(t: &Tensor) -> Option<usize> {
    let (qt, rows, n) = (t.qtype.id, t.dims[1] as usize, t.dims[0] as usize);
    if r8q::can_repack(qt, rows, n) {
        Some(r8q::repacked_len(qt, rows, n))
    } else if r8::can_repack(qt, rows, n) {
        Some(r8::repacked_len(qt, rows, n))
    } else {
        None
    }
}

/// One anonymous mapping for every repacked matrix, in whole 2 MB pages
/// with transparent huge pages requested before the first touch (decode
/// walks all of it every token; on 4 KB pages the dTLB misses cost more than
/// the arithmetic. AXON_R8_THP=0 skips the request). One mapping, not one
/// per matrix: the kernel only 2 MB-aligns mappings whose length is a whole
/// number of 2 MB pages, and a huge page per small matrix (attn_k) would
/// waste most of it.
fn r8_arena(len: usize) -> memmap2::MmapMut {
    const HUGE: usize = 2 << 20;
    let map = memmap2::MmapMut::map_anon(len.max(1).next_multiple_of(HUGE))
        .expect("r8: anonymous mapping");
    #[cfg(target_os = "linux")]
    if std::env::var("AXON_R8_THP").map_or(true, |v| v != "0") {
        // Failure only costs speed.
        let _ = map.advise(memmap2::Advice::HugePage);
    }
    map
}

impl<'a> Weight<'a> {
    fn qt(&self) -> u32 {
        self.t.qtype.id
    }

    fn rows(&self) -> usize {
        self.t.dims[1] as usize
    }

    /// With `dst` (`repacked_len(t)` bytes of the arena), repack the matrix
    /// into it; `r8` is attached once the arena is read-only (`build`).
    /// Then, if `release`, the original bytes are handed back
    /// (`Model::release`) so the weights aren't resident twice. Never
    /// release a tensor read elsewhere: the token embedding (a tied output
    /// matrix is the same tensor).
    fn load(m: &'a Model, t: &'a Tensor, dst: Option<&mut [u8]>, release: bool) -> Weight<'a> {
        let Some(buf) = dst else {
            return Weight {
                t,
                data: Some(m.tensor_data(t)),
                r8: None,
            };
        };
        let (qt, rows, n) = (t.qtype.id, t.dims[1] as usize, t.dims[0] as usize);
        if r8q::can_repack(qt, rows, n) {
            r8q::repack_into(qt, m.tensor_data(t), rows, n, buf);
        } else {
            r8::repack_into(qt, m.tensor_data(t), rows, n, buf);
        }
        if release {
            m.release(t);
        }
        Weight {
            t,
            data: None,
            r8: None,
        }
    }

    /// Whether the repack is r8q's (Q5_0/Q8_0) rather than r8's (K-quants).
    fn is_r8q(&self) -> bool {
        matches!(self.qt(), Q5_0 | Q8_0)
    }

    /// Repacked decode: `r8::gemv` or `r8q::gemv` (same contract).
    fn gemv(&self, groups: &[u8], x: &QuantizedAct, out: &mut [f32], pf: usize) {
        if self.is_r8q() {
            r8q::gemv(self.qt(), groups, x, out, pf);
        } else {
            r8::gemv(self.qt(), groups, x, out, pf);
        }
    }

    /// Repacked batch: `r8::gemm` or `r8q::gemm` (same contract).
    fn gemm(&self, groups: &[u8], xs: &[&QuantizedAct], out: &mut [f32], stride: usize) {
        if self.is_r8q() {
            r8q::gemm(self.qt(), groups, xs, out, stride);
        } else {
            r8::gemm(self.qt(), groups, xs, out, stride);
        }
    }

    /// The tensor's bytes (not repacked).
    fn bytes(&self) -> &'a [u8] {
        self.data.expect("repacked weight has no raw bytes")
    }

    /// Rows r0..r0 + k of a repacked matrix as whole r8 groups, plus the
    /// rest of the matrix when `to_end` (prefetch room, see `r8::gemv`).
    fn groups(&self, n_in: usize, r0: usize, k: usize, to_end: bool) -> &[u8] {
        let p: &[u8] = self.r8.as_deref().expect("not repacked");
        assert!(
            r0.is_multiple_of(8) && k.is_multiple_of(8),
            "{}: r8 rows not whole groups",
            self.t.name
        );
        let gb = if self.is_r8q() {
            r8q::group_bytes(self.qt(), n_in)
        } else {
            r8::group_bytes(self.qt(), n_in)
        };
        let end = if to_end { p.len() } else { (r0 + k) / 8 * gb };
        &p[r0 / 8 * gb..end]
    }
}

/// The weights of one transformer block, borrowed from the mapping or repacked.
struct Layer<'a> {
    attn_norm: Vec<f32>,
    wq: Weight<'a>,
    wk: Weight<'a>,
    wv: Weight<'a>,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    wo: Weight<'a>,
    ffn_norm: Vec<f32>,
    w_gate: Weight<'a>,
    w_up: Weight<'a>,
    w_down: Weight<'a>,
}

pub struct Qwen2<'a> {
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    eps: f32,
    rope_base: f64,
    token_embd: &'a Tensor,
    /// token_embd's bytes (never released: see `Weight::load`)
    embd: &'a [u8],
    /// Empty, and `output` None, in a model of the first layers only
    /// (`new_layers`): the output head is another backend's.
    output_norm: Vec<f32>,
    output: Option<Weight<'a>>,
    /// Layers 0..layers.len() (all n_layers unless `new_layers`).
    layers: Vec<Layer<'a>>,
}

/// Keys and values of every past position, per layer.
/// k[layer] is laid out as [ctx][n_kv_heads * head_dim], same for v.
pub struct KvCache {
    /// Per layer, [t][kv_dim] keys and values: f16 unless AXON_KV=f32.
    data: KvData,
    kv_dim: usize,
    max_ctx: usize,
}

impl KvCache {
    /// Every layer's K then V, all positions, as f32 bits (tests: exact,
    /// the f16 cache widens losslessly).
    #[cfg(test)]
    pub(crate) fn bits(&self) -> Vec<u32> {
        fn all<T: KvElem>(c: &Layers<T>) -> Vec<u32> {
            let kv =
                c.k.iter()
                    .zip(&c.v)
                    .flat_map(|(k, v)| k.iter().chain(v.iter()));
            kv.map(|x| x.to_f32().to_bits()).collect()
        }
        match &self.data {
            KvData::F16(c) => all(c),
            KvData::F32(c) => all(c),
        }
    }
}

impl KvCache {
    /// Every layer's K and V buffers if the cache is f16 (None for f32):
    /// what streaming reads and writes in place (`cuda::qwen2::stream`).
    pub(crate) fn f16_layers(&mut self) -> Option<Vec<(&mut [Half], &mut [Half])>> {
        match &mut self.data {
            KvData::F16(c) => Some(
                c.k.iter_mut()
                    .zip(c.v.iter_mut())
                    .map(|(k, v)| (&mut k[..], &mut v[..]))
                    .collect(),
            ),
            KvData::F32(_) => None,
        }
    }
}

/// Run `$body` with `$k`, `$v` bound to layer `$l`'s K and V cache slices
/// (`&mut [T]`, T the cache's element type).
macro_rules! with_kv_layer {
    ($cache:expr, $l:expr, |$k:ident, $v:ident| $body:block) => {
        match &mut $cache.data {
            KvData::F16(c) => {
                let ($k, $v) = (&mut c.k[$l][..], &mut c.v[$l][..]);
                $body
            }
            KvData::F32(c) => {
                let ($k, $v) = (&mut c.k[$l][..], &mut c.v[$l][..]);
                $body
            }
        }
    };
}

pub(crate) fn meta_num(m: &Model, key: &str) -> Option<f64> {
    m.metadata
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| match v {
            Val::U32(n) => Some(*n as f64),
            Val::I32(n) => Some(*n as f64),
            Val::U64(n) => Some(*n as f64),
            Val::I64(n) => Some(*n as f64),
            Val::F32(n) => Some(*n as f64),
            Val::F64(n) => Some(*n),
            _ => None,
        })
}

fn find<'a>(m: &'a Model, name: &str) -> &'a Tensor {
    m.tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("missing tensor {name}"))
}

fn find_opt<'a>(m: &'a Model, name: &str) -> Option<&'a Tensor> {
    m.tensors.iter().find(|t| t.name == name)
}

/// Fully dequantize a small tensor (norm weights, biases) into f32.
fn load_vec(m: &Model, t: &Tensor) -> Vec<f32> {
    let mut out = vec![0.0f32; t.n_elements as usize];
    dequant(t.qtype.id, m.tensor_data(t), &mut out);
    out
}

/// Tensor types the kernels run (`quant::dequant` and `dot_row_q8`).
const SUPPORTED_TYPES: [u32; 6] = [F32, F16, Q5_0, Q8_0, Q4_K, Q6_K];

/// Everything `Qwen2::new` and `Tokenizer::from_gguf` rely on, checked up front
/// so a bad or unsupported model is an error instead of a panic: the
/// architecture, the hyperparameters, every tensor's presence, type and shape,
/// and the vocabulary. Reads metadata only.
pub fn check_supported(m: &Model) -> std::io::Result<()> {
    let bad = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    if m.arch != "qwen2" {
        return Err(bad(format!(
            "architecture '{}' is not supported (only qwen2 for now)",
            if m.arch.is_empty() { "<none>" } else { &m.arch }
        )));
    }
    let need = |k: &str| -> std::io::Result<usize> {
        let key = format!("qwen2.{k}");
        match meta_num(m, &key) {
            Some(v) if v.fract() == 0.0 && v >= 1.0 && v <= u32::MAX as f64 => Ok(v as usize),
            Some(v) => Err(bad(format!(
                "metadata {key} = {v} is not a positive integer"
            ))),
            None => Err(bad(format!("metadata {key} is missing or not a number"))),
        }
    };
    let n_layers = need("block_count")?;
    let dim = need("embedding_length")?;
    let n_heads = need("attention.head_count")?;
    let n_kv_heads = need("attention.head_count_kv")?;
    if !dim.is_multiple_of(n_heads) || !(dim / n_heads).is_multiple_of(2) {
        return Err(bad(format!(
            "embedding_length {dim} doesn't split into {n_heads} heads of even size"
        )));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(bad(format!(
            "head_count {n_heads} is not a multiple of head_count_kv {n_kv_heads}"
        )));
    }
    let kv_dim = n_kv_heads * (dim / n_heads);

    // `rows` = None: a vector of `cols` elements; Some(r): a cols x r matrix.
    let check = |name: &str, cols: usize, rows: Option<usize>| -> std::io::Result<&Tensor> {
        let t = find_opt(m, name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        let q = t.qtype;
        if !SUPPORTED_TYPES.contains(&q.id) {
            return Err(bad(format!(
                "tensor {name}: type {} ({}) is not supported (F32, F16, Q5_0, Q8_0, Q4_K, Q6_K)",
                q.name, q.id
            )));
        }
        let ok = match rows {
            None => t.n_elements == cols as u64,
            Some(r) => {
                t.dims.len() >= 2
                    && t.dims[0] == cols as u64
                    && (if r == 0 {
                        t.dims[1] > 0
                    } else {
                        t.dims[1] == r as u64
                    })
                    && t.dims[2..].iter().all(|&d| d == 1)
            }
        };
        if !ok
            || t.dims
                .first()
                .is_some_and(|&d| d % q.block_size as u64 != 0)
        {
            return Err(bad(format!(
                "tensor {name}: shape {:?} doesn't fit the model ({cols} x {})",
                t.dims,
                rows.map_or("1".into(), |r| if r == 0 {
                    "n".into()
                } else {
                    r.to_string()
                })
            )));
        }
        Ok(t)
    };

    let embd = check("token_embd.weight", dim, Some(0))?;
    let vocab = embd.dims[1] as usize;
    if find_opt(m, "output.weight").is_some() {
        check("output.weight", dim, Some(vocab))?;
    }
    check("output_norm.weight", dim, None)?;
    for i in 0..n_layers {
        let t = |s: &str, cols, rows| check(&format!("blk.{i}.{s}"), cols, rows);
        t("attn_norm.weight", dim, None)?;
        t("attn_q.weight", dim, Some(dim))?;
        t("attn_k.weight", dim, Some(kv_dim))?;
        t("attn_v.weight", dim, Some(kv_dim))?;
        for (b, n) in [
            ("attn_q.bias", dim),
            ("attn_k.bias", kv_dim),
            ("attn_v.bias", kv_dim),
        ] {
            if find_opt(m, &format!("blk.{i}.{b}")).is_some() {
                t(b, n, None)?;
            }
        }
        t("attn_output.weight", dim, Some(dim))?;
        t("ffn_norm.weight", dim, None)?;
        let ffn = t("ffn_gate.weight", dim, Some(0))?.dims[1] as usize;
        t("ffn_up.weight", dim, Some(ffn))?;
        t("ffn_down.weight", ffn, Some(dim))?;
    }

    match m
        .metadata
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.tokens")
    {
        Some((_, Val::Arr(a))) if !a.is_empty() && a.len() <= vocab => Ok(()),
        Some((_, Val::Arr(a))) => Err(bad(format!(
            "tokenizer has {} tokens for an embedding of {vocab} rows",
            a.len()
        ))),
        _ => Err(bad("tokenizer.ggml.tokens is missing".into())),
    }
}

impl<'a> Qwen2<'a> {
    /// The model, with Q4_K/Q6_K and Q5_0/Q8_0 matrices repacked for the r8
    /// and r8q kernels unless AXON_REPACK=0 (the per-row kernels only: the
    /// branch's bits).
    pub fn new(m: &'a Model) -> Qwen2<'a> {
        Self::new_with(m, std::env::var("AXON_REPACK").map_or(true, |v| v != "0"))
    }

    pub fn new_with(m: &'a Model, repack: bool) -> Qwen2<'a> {
        Self::build(m, repack, false, usize::MAX)
    }

    /// The embedding and layers 0..k only (all of them, and the output head,
    /// if k >= n_layers): the CPU half of a model split across devices
    /// (`hybrid`). Only these weights are read, repacked and released; the
    /// rest of the mapping is never touched. Without the head it runs
    /// `hidden` / `hidden_batch` only; its cache holds k layers.
    pub fn new_layers(m: &'a Model, repack: bool, k: usize) -> Qwen2<'a> {
        Self::build(m, repack, false, k)
    }

    /// `tie`: use the token embedding as the output matrix even if the model
    /// has its own (tests: the tied-embedding case on any model). `keep`:
    /// layers 0..keep only, the output head only if that's all of them.
    fn build(m: &'a Model, repack: bool, tie: bool, keep: usize) -> Qwen2<'a> {
        if let Err(e) = check_supported(m) {
            panic!("{e}");
        }
        let a = &m.arch;
        let key = |k: &str| format!("{a}.{k}");
        let need =
            |k: &str| meta_num(m, &key(k)).unwrap_or_else(|| panic!("missing metadata {}", key(k)));

        let n_layers = need("block_count") as usize;
        let (k, head) = (keep.min(n_layers), keep >= n_layers);
        let dim = need("embedding_length") as usize;
        let n_heads = need("attention.head_count") as usize;
        let n_kv_heads = need("attention.head_count_kv") as usize;
        let head_dim = dim / n_heads;
        let eps = meta_num(m, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f32;
        let rope_base = meta_num(m, &key("rope.freq_base")).unwrap_or(10000.0);

        let token_embd = find(m, "token_embd.weight");
        // ne1 of the embedding matrix is the vocab size.
        let vocab = token_embd.dims[1] as usize;
        // Some models tie output to the embedding; qwen2.5-0.5b ships a separate one.
        let out_t = match find_opt(m, "output.weight") {
            Some(t) if !tie => t,
            _ => token_embd,
        };
        // Every repacked matrix in one arena, in this order: output, then
        // per layer MATS (64-byte aligned offsets).
        const MATS: [&str; 7] = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];
        let lt = |i: usize, s: &str| find(m, &format!("blk.{i}.{s}"));
        let mut plan: Vec<Option<(usize, usize)>> = Vec::new();
        let mut total = 0;
        let out_ts = head.then_some(out_t);
        for t in out_ts
            .into_iter()
            .chain((0..k).flat_map(|i| MATS.map(|s| lt(i, s))))
        {
            let len = if repack { repacked_len(t) } else { None };
            plan.push(len.map(|len| (total, len)));
            total += len.map_or(0, |l| l.next_multiple_of(64));
        }
        let mut arena = (total > 0).then(|| r8_arena(total));
        // Disjoint slices of the arena, in plan order.
        let mut slots: Vec<Option<&mut [u8]>> = Vec::with_capacity(plan.len());
        {
            let mut rest: &mut [u8] = arena.as_deref_mut().unwrap_or(&mut []);
            let mut at = 0;
            for p in &plan {
                slots.push(p.map(|(off, len)| {
                    let (_, r) = std::mem::take(&mut rest).split_at_mut(off - at);
                    let (d, r) = r.split_at_mut(len);
                    rest = r;
                    at = off + len;
                    d
                }));
            }
        }
        let mut slots = slots.into_iter();
        // A tied output matrix is the embedding, which `embed` keeps reading.
        let mut output =
            out_ts.map(|t| Weight::load(m, t, slots.next().unwrap(), !std::ptr::eq(t, token_embd)));
        let output_norm = match head {
            true => load_vec(m, find(m, "output_norm.weight")),
            false => Vec::new(),
        };
        let layer_slots: Vec<std::sync::Mutex<Vec<Option<&mut [u8]>>>> = (0..k)
            .map(|_| std::sync::Mutex::new(slots.by_ref().take(MATS.len()).collect()))
            .collect();

        // One pool task per layer (`floats` set so it never runs inline):
        // repacking is a pass over all the weights.
        let mut layers = par_map(k, INLINE_WORK + 1, |i| {
            let t = |s: &str| lt(i, s);
            let mut dst = std::mem::take(&mut *layer_slots[i].lock().unwrap()).into_iter();
            let mut w = |s: &str| Weight::load(m, t(s), dst.next().unwrap(), true);
            // Norms and biases are copied out; when repacking, their bytes go
            // back too, so `Model::release` can drop the blocks they share
            // with the small repacked matrices around them.
            let vals = |t: &Tensor| {
                let v = load_vec(m, t);
                if repack {
                    m.release(t);
                }
                v
            };
            let bias = |s: &str| find_opt(m, &format!("blk.{i}.{s}")).map(vals);
            Layer {
                attn_norm: vals(t("attn_norm.weight")),
                wq: w(MATS[0]),
                wk: w(MATS[1]),
                wv: w(MATS[2]),
                bq: bias("attn_q.bias"),
                bk: bias("attn_k.bias"),
                bv: bias("attn_v.bias"),
                wo: w(MATS[3]),
                ffn_norm: vals(t("ffn_norm.weight")),
                w_gate: w(MATS[4]),
                w_up: w(MATS[5]),
                w_down: w(MATS[6]),
            }
        });
        drop(layer_slots);
        drop(slots);
        if let Some(a) = arena.take() {
            let map = std::sync::Arc::new(a.make_read_only().expect("r8: mprotect"));
            let ws = output.iter_mut().chain(layers.iter_mut().flat_map(|l| {
                [
                    &mut l.wq,
                    &mut l.wk,
                    &mut l.wv,
                    &mut l.wo,
                    &mut l.w_gate,
                    &mut l.w_up,
                    &mut l.w_down,
                ]
            }));
            for (w, p) in ws.zip(&plan) {
                w.r8 = p.map(|(off, len)| R8 {
                    map: map.clone(),
                    off,
                    len,
                });
            }
        }

        Qwen2 {
            n_layers,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab,
            eps,
            rope_base,
            token_embd,
            embd: m.tensor_data(token_embd),
            output_norm,
            output,
            layers,
        }
    }

    /// Layer l's weights as the CPU holds them, for streaming them to the
    /// GPU (`cuda::qwen2::stream`): repacked bytes or the file's.
    pub(crate) fn host_layer(&self, l: usize) -> crate::cuda::qwen2::stream::HostLayer<'_> {
        use crate::cuda::qwen2::stream::{HostLayer, HostMat};
        fn mat<'s>(w: &'s Weight<'_>) -> HostMat<'s> {
            HostMat {
                qt: w.qt(),
                rows: w.rows(),
                cols: w.t.dims[0] as usize,
                bytes: match &w.r8 {
                    Some(r) => r,
                    None => w.bytes(),
                },
                repacked: w.r8.is_some(),
            }
        }
        let ly = &self.layers[l];
        HostLayer {
            mats: [
                mat(&ly.wq),
                mat(&ly.wk),
                mat(&ly.wv),
                mat(&ly.wo),
                mat(&ly.w_gate),
                mat(&ly.w_up),
                mat(&ly.w_down),
            ],
            attn_norm: &ly.attn_norm,
            ffn_norm: &ly.ffn_norm,
            bq: ly.bq.as_deref(),
            bk: ly.bk.as_deref(),
            bv: ly.bv.as_deref(),
        }
    }

    /// The whole mapping the repacked matrices live in (None if none is).
    pub(crate) fn r8_arena(&self) -> Option<&[u8]> {
        let w = self
            .layers
            .iter()
            .flat_map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down])
            .chain(self.output.as_ref())
            .find_map(|w| w.r8.as_ref())?;
        Some(&w.map[..])
    }

    /// Layers here (0..k of the model).
    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// The embeddings of `tokens`, token-major (dim floats each).
    pub(crate) fn embed_batch(&self, tokens: &[u32]) -> Vec<f32> {
        let rows = par_map(tokens.len(), self.dim, |b| self.embed(tokens[b]));
        rows.concat()
    }

    pub fn new_cache(&self, max_ctx: usize) -> KvCache {
        let kv_dim = self.n_kv_heads * self.head_dim;
        let len = max_ctx * kv_dim;
        KvCache {
            data: if crate::kv::use_f32() {
                KvData::F32(Layers::new(self.layers.len(), len))
            } else {
                KvData::F16(Layers::new(self.layers.len(), len))
            },
            kv_dim,
            max_ctx,
        }
    }

    /// Decode kernel settings (read once per matvec), and whether one thread
    /// runs every chunk: then prefetch may run on into the next chunk, which
    /// is its own work too.
    fn decode_cfg() -> (DecodeCfg, bool) {
        (DecodeCfg::get(), Pool::global().threads() == 1)
    }

    /// y = W x. W has ne1 rows of ne0 values each; y[o] = row_o · x.
    fn matvec(&self, w: &Weight, x: &[f32]) -> Vec<f32> {
        let [y] = self.matvec_many([w], x);
        y
    }

    /// Joint row space of `ws` for input width `n_in`: starts[i] is where
    /// matrix i begins, plus the total and the row granularity pool chunks
    /// must keep: 8 (whole r8 groups) if any matrix is repacked, every start
    /// then a multiple of 8; else 1.
    fn joint_rows<const N: usize>(ws: [&Weight; N], n_in: usize) -> ([usize; N], usize, usize) {
        let mut starts = [0usize; N];
        let mut total = 0;
        for (i, w) in ws.iter().enumerate() {
            assert_eq!(w.t.dims[0] as usize, n_in, "matmul {}: input len", w.t.name);
            starts[i] = total;
            total += w.rows();
        }
        let align = if ws.iter().any(|w| w.r8.is_some()) {
            8
        } else {
            1
        };
        assert!(
            starts.iter().all(|s| s % align == 0),
            "repacked matrices must start at a multiple of 8 rows"
        );
        (starts, total, align)
    }

    /// Decode rows r0..r0 + out.len() of `w` · x. Prefetch stays inside the
    /// rows the caller owns, or runs to the matrix's end when `own_all`.
    fn rows_vec(
        &self,
        w: &Weight,
        r0: usize,
        x: &QuantizedAct,
        out: &mut [f32],
        cfg: DecodeCfg,
        own_all: bool,
    ) {
        let k = out.len();
        if w.r8.is_some() {
            let groups = w.groups(x.len(), r0, k, own_all);
            w.gemv(groups, x, out, cfg.prefetch);
        } else {
            let (qt, data) = (w.qt(), w.bytes());
            let rb = row_bytes(qt, x.len());
            let lim = if own_all { data.len() } else { (r0 + k) * rb };
            dot_rows_q8(qt, &data[r0 * rb..lim], x, out, cfg);
        }
    }

    /// Several matrices applied to the same input in one parallel pass:
    /// one fork/join and one quantization of x instead of one per matrix.
    /// Returns one output vector per matrix, in order.
    fn matvec_many<const N: usize>(&self, ws: [&Weight; N], x: &[f32]) -> [Vec<f32>; N] {
        // Quantize x to 8 bits once; every row then does integer multiply-adds
        // against its packed weights instead of decoding them to floats.
        let types = ws.map(|w| w.qt());
        let xq = QuantizedAct::for_types(x, &types);
        // All matrices' rows form one index space; starts[i] is where matrix i begins.
        let (starts, total, align) = Self::joint_rows(ws, x.len());
        // Every output row is an independent dot product, so split rows across
        // cores. About 4 chunks per thread lets the fast and efficiency cores
        // even out; at most 64 rows keeps chunk claiming cheap.
        let mut out = vec![0f32; total];
        let chunk = (total / (Pool::global().threads() * 4))
            .clamp(4, 64)
            .next_multiple_of(align);
        let (cfg, own_all) = Self::decode_cfg();
        pool::for_each_chunk_mut(&mut out, chunk, |c, mut ys| {
            // The chunk's rows, one run per matrix it touches, several rows
            // per kernel call.
            let mut g = c * chunk;
            while !ys.is_empty() {
                let t = starts.iter().rposition(|&s| s <= g).unwrap();
                let end = starts.get(t + 1).copied().unwrap_or(total);
                let k = (end - g).min(ys.len());
                let (run, rest) = std::mem::take(&mut ys).split_at_mut(k);
                self.rows_vec(ws[t], g - starts[t], &xq, run, cfg, own_all);
                g += k;
                ys = rest;
            }
        });
        // Split the joint output back into one vector per matrix (no copy when N == 1).
        let mut res: [Vec<f32>; N] = std::array::from_fn(|_| Vec::new());
        for i in (1..N).rev() {
            res[i] = out.split_off(starts[i]);
        }
        res[0] = out;
        res
    }

    /// silu(gate x) * (up x), row by row: each task computes row r of both
    /// matrices and combines them, so the exp() runs on every core instead of
    /// in a serial pass after the matvec.
    fn matvec_swiglu(&self, gate: &Weight, up: &Weight, x: &[f32]) -> Vec<f32> {
        let n = gate.rows();
        let (_, _, a0) = Self::joint_rows([gate], x.len());
        let (_, _, a1) = Self::joint_rows([up], x.len());
        assert_eq!(up.rows(), n, "swiglu: gate/up row counts differ");
        let xq = QuantizedAct::for_types(x, &[gate.qt(), up.qt()]);
        let mut out = vec![0f32; n];
        // Each row is two dot products, so half the rows per chunk of matvec_many.
        const MAX_CHUNK: usize = 32;
        let chunk = (n / (Pool::global().threads() * 8))
            .clamp(4, MAX_CHUNK)
            .next_multiple_of(a0.max(a1));
        let (cfg, own_all) = Self::decode_cfg();
        pool::for_each_chunk_mut(&mut out, chunk, |c, ys| {
            let (r, k) = (c * chunk, ys.len());
            let mut ub = [0f32; MAX_CHUNK];
            let ub = &mut ub[..k];
            self.rows_vec(gate, r, &xq, ys, cfg, own_all);
            self.rows_vec(up, r, &xq, ub, cfg, own_all);
            vmath::swiglu(ys, ub);
        });
        out
    }

    /// Rows per pool chunk for a batched matmul over `nb` tokens: an even
    /// count so the 2-row tile kernels never get a lone row at a chunk edge.
    /// ~32 chunks per thread for prefill-sized batches: a chunk is a lot of
    /// work (every row times every token), so a few big chunks leave threads
    /// idle while the last one finishes, and claiming is cheap next to it.
    /// Sweep on bella (560-token prompt): 4/thread ~611 tok/s, 32 ~646, 64+
    /// slower again. A small batch (a speculative verify, 2..9 tokens) is
    /// compute-bound on the E-cores (half-width SIMD) while the P-cores
    /// finish early, so it gets 16 small chunks per thread to balance them:
    /// a 0.5B verify pass at 4 per thread left ~21% of pool time waiting, and
    /// drafting (K=7) on low-repetition text went from -5% to -1% vs no
    /// drafting with 16. Chunking never changes results.
    /// With repacked matrices (`align` 8) chunks are whole r8 groups.
    fn rows_per_chunk(rows: usize, nb: usize, align: usize) -> usize {
        let per_thread = if nb <= 9 { 16 } else { (nb / 2).clamp(4, 32) };
        rows.div_ceil(Pool::global().threads() * per_thread)
            .max(1)
            .next_multiple_of(align.max(2))
    }

    /// Token slots for a batched matmul over `ws`: `quant::batch_slots` of
    /// the types that go through the per-row kernels (their x4 tiles want
    /// padding); the r8 GEMM takes any count, so all-repacked needs none.
    fn slots_for(ws: &[&Weight], n_in: usize, nb: usize) -> usize {
        let types: Vec<u32> = ws
            .iter()
            .filter(|w| w.r8.is_none())
            .map(|w| w.qt())
            .collect();
        if types.is_empty() {
            nb
        } else {
            batch_slots(&types, n_in, nb)
        }
    }

    /// Rows r0..r0 + ys.len() / slots of `w` times the batch, into
    /// ys[r * slots + b]: r8 GEMM over the `nb` real tokens (padding slots
    /// untouched), or the per-row tile kernels over every slot of `xr`.
    fn rows_batch(&self, w: &Weight, r0: usize, xr: &[&QuantizedAct], nb: usize, ys: &mut [f32]) {
        let slots = xr.len();
        let n_in = xr[0].len();
        let k = ys.len() / slots;
        if w.r8.is_some() {
            w.gemm(w.groups(n_in, r0, k, false), &xr[..nb], ys, slots);
        } else {
            let (qt, data) = (w.qt(), w.bytes());
            let rb = row_bytes(qt, n_in);
            let row_of = |g: usize| (qt, &data[g * rb..(g + 1) * rb]);
            dot_rows_chunk(ys, slots, r0, &row_of, xr);
        }
    }

    /// The batch's activations quantized for `types`, one per token. The
    /// matmuls hand the kernels `quant::batch_slots` references to them, the
    /// extra slots repeating the last token (their outputs are ignored).
    /// Quantizing is about a nanosecond per float (896 floats ~1 us), the
    /// same scale as `per_token`'s cost unit, so a verify batch's 896-wide
    /// inputs quantize inline instead of paying a fork/join.
    fn quantize_batch<X: AsRef<[f32]> + Sync>(xs: &[X], types: &[u32]) -> Vec<QuantizedAct> {
        let n_in = xs[0].as_ref().len();
        par_map(xs.len(), n_in, |b| {
            let x = xs[b].as_ref();
            assert_eq!(x.len(), n_in, "matmul: ragged batch");
            QuantizedAct::for_types(x, types)
        })
    }

    /// `quantize_batch` of the first `nb` token columns of a row-major
    /// (row, token slot) buffer: the transpose and the quantization in one
    /// pass (one fork/join at most), same input values as `unbatch` gives.
    fn quantize_cols(y: &[f32], slots: usize, nb: usize, types: &[u32]) -> Vec<QuantizedAct> {
        let n = y.len() / slots;
        par_map(nb, n, |b| {
            QuantizedAct::for_types(&gather_col(y, slots, b, n), types)
        })
    }

    /// `matvec_many` for a batch of inputs: res[i][b] = ws[i] · xs[b].
    /// See `matmul_raw`; this transposes its output to per-token vectors.
    fn matmul_many<const N: usize, X: AsRef<[f32]> + Sync>(
        &self,
        ws: [&Weight; N],
        xs: &[X],
    ) -> [Vec<Vec<f32>>; N] {
        let nb = xs.len();
        let (out, starts, slots) = self.matmul_raw(ws, xs);
        std::array::from_fn(|i| {
            let n = ws[i].rows();
            unbatch(&out[starts[i] * slots..(starts[i] + n) * slots], slots, nb)
        })
    }

    /// Batched matmul, output left row-major over (row, token slot):
    /// y[r * S + b] for the joint row space of all `ws`, where starts[i] is
    /// matrix i's first row and S >= B the slot count (`quant::batch_slots`;
    /// slots past B are padding).
    ///
    /// Rows are split across the pool. A task loads each weight row once and
    /// dots it with every token's input while it's still in L1 (row outer,
    /// token inner), so the weights stream from RAM once per batch, not once
    /// per token. Each dot is the same `dot_row_q8` on the same quantized input
    /// as `matvec_many`, so results are bit-identical.
    fn matmul_raw<const N: usize, X: AsRef<[f32]> + Sync>(
        &self,
        ws: [&Weight; N],
        xs: &[X],
    ) -> (Vec<f32>, [usize; N], usize) {
        let types = ws.map(|w| w.qt());
        self.matmul_q(ws, &Self::quantize_batch(xs, &types))
    }

    /// `matmul_raw` on inputs already quantized for `ws`' types.
    fn matmul_q<const N: usize>(
        &self,
        ws: [&Weight; N],
        xq: &[QuantizedAct],
    ) -> (Vec<f32>, [usize; N], usize) {
        let nb = xq.len();
        let n_in = xq[0].len();
        let (starts, total, align) = Self::joint_rows(ws, n_in);
        let slots = Self::slots_for(&ws, n_in, nb);
        let xr: Vec<&QuantizedAct> = (0..slots).map(|b| &xq[b.min(nb - 1)]).collect();
        let mut out = vec![0f32; total * slots];
        let rows = Self::rows_per_chunk(total, nb, align);
        if align == 1 {
            // Nothing repacked: the per-row tile kernels, rows paired across
            // matrices as they come.
            let info = ws.map(|w| (w.qt(), row_bytes(w.qt(), n_in), w.bytes()));
            let row_of = |g: usize| {
                let t = starts.iter().rposition(|&s| s <= g).unwrap();
                let (qt, rb, data) = info[t];
                let o = g - starts[t];
                (qt, &data[o * rb..(o + 1) * rb])
            };
            pool::for_each_chunk_mut(&mut out, rows * slots, |c, ys| {
                dot_rows_chunk(ys, slots, c * rows, &row_of, &xr)
            });
        } else {
            // One run per matrix the chunk touches (whole r8 groups).
            pool::for_each_chunk_mut(&mut out, rows * slots, |c, mut ys| {
                let mut g = c * rows;
                while !ys.is_empty() {
                    let t = starts.iter().rposition(|&s| s <= g).unwrap();
                    let end = starts.get(t + 1).copied().unwrap_or(total);
                    let k = (end - g).min(ys.len() / slots);
                    let (run, rest) = std::mem::take(&mut ys).split_at_mut(k * slots);
                    self.rows_batch(ws[t], g - starts[t], &xr, nb, run);
                    g += k;
                    ys = rest;
                }
            });
        }
        (out, starts, slots)
    }

    /// The output head for a batch, reduced to each token's greedy pick:
    /// picks[b] == argmax(w · xs[b]) with `argmax`'s rules (see `Best`).
    /// Each pool chunk computes its rows into thread-local scratch and keeps
    /// only a `Best` per token; the chunks are then merged in row order. No
    /// vocab-sized logits are allocated or transposed. Each logit is the
    /// same dot as in `matmul_raw` (and so `matvec`), bit for bit.
    fn matmul_argmax<X: AsRef<[f32]> + Sync>(&self, w: &Weight, xs: &[X]) -> Vec<u32> {
        let nb = xs.len();
        let qt = w.qt();
        let xq = Self::quantize_batch(xs, &[qt]);
        let n_in = xq[0].len();
        let (_, total, align) = Self::joint_rows([w], n_in);
        let slots = Self::slots_for(&[w], n_in, nb);
        let xr: Vec<&QuantizedAct> = (0..slots).map(|b| &xq[b.min(nb - 1)]).collect();
        let rows = Self::rows_per_chunk(total, nb, align);
        let n_chunks = total.div_ceil(rows);
        // best[c * nb + b]: chunk c's best row for token b.
        let mut best = vec![Best::NONE; n_chunks * nb];
        pool::for_each_chunk_mut(&mut best, nb, |c, bs| {
            let first = c * rows;
            let n_rows = rows.min(total - first);
            with_rows_scratch(n_rows * slots, |ys| {
                self.rows_batch(w, first, &xr, nb, ys);
                best_rows(ys, slots, first, bs);
            });
        });
        (0..nb)
            .map(|b| {
                let mut acc = Best::NONE;
                for c in 0..n_chunks {
                    acc.merge(best[c * nb + b]);
                }
                acc.index(total)
            })
            .collect()
    }

    /// `matvec_swiglu` for a batch: per row r, per token b, the gate and up
    /// dots combined with the same formula. Layout and slot count returned as
    /// in `matmul_raw`.
    fn matmul_swiglu<X: AsRef<[f32]> + Sync>(
        &self,
        gate: &Weight,
        up: &Weight,
        xs: &[X],
    ) -> (Vec<f32>, usize) {
        let nb = xs.len();
        let n_in = xs[0].as_ref().len();
        let n = gate.rows();
        let (_, _, a0) = Self::joint_rows([gate], n_in);
        let (_, _, a1) = Self::joint_rows([up], n_in);
        assert_eq!(up.rows(), n, "swiglu: gate/up row counts differ");
        let types = [gate.qt(), up.qt()];
        let xq = Self::quantize_batch(xs, &types);
        let slots = Self::slots_for(&[gate, up], n_in, nb);
        let xr: Vec<&QuantizedAct> = (0..slots).map(|b| &xq[b.min(nb - 1)]).collect();
        let mut out = vec![0f32; n * slots];
        let align = a0.max(a1);
        let rows = Self::rows_per_chunk(n, nb, align);
        if align == 1 {
            // Nothing repacked: each gate row paired with its up row.
            let [g, u] = [gate, up].map(|w| (w.qt(), row_bytes(w.qt(), n_in), w.bytes()));
            pool::for_each_chunk_mut(&mut out, rows * slots, |c, ys| {
                with_rows_scratch(slots, |uv| {
                    for (j, yr) in ys.chunks_mut(slots).enumerate() {
                        let r = c * rows + j;
                        let gr = &g.2[r * g.1..(r + 1) * g.1];
                        let ur = &u.2[r * u.1..(r + 1) * u.1];
                        dot_rows_q8_multi([g.0, u.0], [gr, ur], &xr, [&mut *yr, &mut *uv]);
                        vmath::swiglu(yr, uv);
                    }
                });
            });
        } else {
            // The chunk's gate rows, then its up rows, then combine (real
            // tokens only: padding slots may be unwritten on r8 rows).
            pool::for_each_chunk_mut(&mut out, rows * slots, |c, ys| {
                let r0 = c * rows;
                with_rows_scratch(ys.len(), |uv| {
                    self.rows_batch(gate, r0, &xr, nb, ys);
                    self.rows_batch(up, r0, &xr, nb, uv);
                    for (yr, ur) in ys.chunks_mut(slots).zip(uv.chunks(slots)) {
                        vmath::swiglu(&mut yr[..nb], &ur[..nb]);
                    }
                });
            });
        }
        (out, slots)
    }

    /// RMSNorm: scale x so its root-mean-square is 1, then multiply by a learned weight.
    fn rmsnorm(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len().min(w.len())];
        self.rmsnorm_into(x, w, &mut out);
        out
    }

    /// `rmsnorm` into an existing buffer.
    fn rmsnorm_into(&self, x: &[f32], w: &[f32], out: &mut [f32]) {
        rmsnorm_into(x, w, self.eps, out);
    }

    /// (sin, cos) of each RoPE pair's angle at `pos`: pair i turns by
    /// pos * base^(-2i/head_dim). Low i spins fast, high i spins slowly.
    /// Every head in every layer uses the same angles, so this runs once a token.
    fn rope_table(&self, pos: usize) -> Vec<(f32, f32)> {
        (0..self.head_dim / 2)
            .map(|i| {
                let freq = self.rope_base.powf(-2.0 * i as f64 / self.head_dim as f64);
                let angle = pos as f64 * freq;
                (angle.sin() as f32, angle.cos() as f32)
            })
            .collect()
    }

    /// Run one token at position `pos` and return logits (len vocab).
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let x = self.hidden(token, pos, cache);
        // 9. Final norm, then the output head scores every vocab entry.
        let h = self.rmsnorm(&x, &self.output_norm);
        self.matvec(self.head(), &h)
    }

    /// Run one token at position `pos` for its effect on the cache only. Prompt
    /// tokens before the last need no logits, and the output head is the
    /// biggest matrix in the model (37% of a token's weight reads here).
    /// Prompts go through `feed_batch`; this is its reference in the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn feed(&self, token: u32, pos: usize, cache: &mut KvCache) {
        self.hidden(token, pos, cache);
    }

    /// The token's row of the embedding matrix.
    fn embed(&self, token: u32) -> Vec<f32> {
        let mut x = vec![0.0f32; self.dim];
        let rb = row_bytes(self.token_embd.qtype.id, self.dim);
        let t = token as usize;
        dequant(
            self.token_embd.qtype.id,
            &self.embd[t * rb..(t + 1) * rb],
            &mut x,
        );
        x
    }

    /// Run `tokens[b]` at position `start + b` for every b, for the cache only.
    /// Same result, bit for bit, as `feed` on each token in order, but every
    /// weight row is read once per batch instead of once per token.
    pub fn feed_batch(&self, tokens: &[u32], start: usize, cache: &mut KvCache) {
        self.hidden_batch(tokens, start, cache);
    }

    /// Logits after each of `tokens[b]` at position `start + b`: row b is,
    /// bit for bit, what `forward(tokens[b], start + b, ..)` returns after
    /// the earlier tokens went in. The speculative-decoding verify pass: the
    /// whole model, output head included, reads each weight row once for the
    /// batch. Same final norm as `forward`; the head goes through
    /// `matmul_many`, whose per-row dots equal `matvec`'s. Cache slots
    /// `start..start + len` are overwritten; slots past the accepted prefix
    /// are simply rewritten by the next call (attention reads only 0..=pos).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn verify_batch(&self, tokens: &[u32], start: usize, cache: &mut KvCache) -> Vec<Vec<f32>> {
        let hs = self.final_norm_batch(tokens, start, cache);
        let [logits] = self.matmul_many([self.head()], &hs);
        logits
    }

    /// `verify_batch` reduced to what greedy decoding uses: picks[b] ==
    /// argmax(verify_batch(..)[b]), same index on ties (the last of equal
    /// maxima, never a NaN unless all are: `Best`), without materializing
    /// any logits (`matmul_argmax`).
    pub fn verify_argmax(&self, tokens: &[u32], start: usize, cache: &mut KvCache) -> Vec<u32> {
        let hs = self.final_norm_batch(tokens, start, cache);
        self.matmul_argmax(self.head(), &hs)
    }

    /// The output matrix (absent in a model of the first layers only).
    fn head(&self) -> &Weight<'a> {
        self.output
            .as_ref()
            .expect("a model of the first layers only has no output head")
    }

    /// `hidden_batch` then the final norm, as `forward` does per token.
    fn final_norm_batch(&self, tokens: &[u32], start: usize, cache: &mut KvCache) -> Vec<Vec<f32>> {
        let xs = self.hidden_batch(tokens, start, cache);
        par_map(xs.len(), self.dim, |b| {
            self.rmsnorm(&xs[b], &self.output_norm)
        })
    }

    /// `hidden` for a batch: `feed_batch`'s work, returning every token's
    /// final hidden state (before the output norm).
    pub(crate) fn hidden_batch(
        &self,
        tokens: &[u32],
        start: usize,
        cache: &mut KvCache,
    ) -> Vec<Vec<f32>> {
        let nb = tokens.len();
        assert!(nb > 0, "feed_batch: empty batch");
        assert!(
            start + nb <= cache.max_ctx,
            "positions {start}..{} exceed cache size {}",
            start + nb,
            cache.max_ctx
        );
        let (dim, hd) = (self.dim, self.head_dim);
        let kv_dim = cache.kv_dim;
        let n_kv = self.n_kv_heads;
        let group = self.n_heads / n_kv;

        // Every per-token step runs over the pool by token (`per_token`): with
        // a few hundred tokens, a serial pass here leaves every worker idle.
        // A few tokens' worth (a speculative verify) runs inline instead.
        let mut xs: Vec<Vec<f32>> = par_map(nb, dim, |b| self.embed(tokens[b]));
        let ropes: Vec<_> = par_map(nb, hd * 16, |b| self.rope_table(start + b));
        // Reused across layers; every element is overwritten before it's read.
        let mut hs: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; nb];
        let mut att = vec![0.0f32; nb * dim];
        let qn = group * hd; // floats per (token, kv head) of attention output
        debug_assert_eq!(dim, n_kv * qn, "feed_batch: dim != heads * head_dim");
        for (l, layer) in self.layers.iter().enumerate() {
            // Steps as in `hidden`, each applied to every token of the batch.
            per_token(&mut hs, dim, |b, h| {
                self.rmsnorm_into(&xs[b], &layer.attn_norm, h)
            });
            let [mut qs, mut ks, mut vs] = self.matmul_many([&layer.wq, &layer.wk, &layer.wv], &hs);
            with_kv_layer!(cache, l, |kc, vc| {
                // Bias, RoPE, and k/v into the cache, per token: same values,
                // same ops as `hidden`, then the same rounding to the cache type.
                let span = start * kv_dim..(start + nb) * kv_dim;
                let (kl, vl) = (&mut kc[span.clone()], &mut vc[span]);
                let mut jobs: Vec<_> = qs
                    .iter_mut()
                    .zip(ks.iter_mut().zip(vs.iter_mut()))
                    .zip(kl.chunks_mut(kv_dim).zip(vl.chunks_mut(kv_dim)))
                    .collect();
                per_token(&mut jobs, dim + 2 * kv_dim, |b, ((q, (k, v)), (kd, vd))| {
                    if let Some(bias) = &layer.bq {
                        add_assign(q, bias);
                    }
                    if let Some(bias) = &layer.bk {
                        add_assign(k, bias);
                    }
                    if let Some(bias) = &layer.bv {
                        add_assign(v, bias);
                    }
                    rope_heads(q, hd, &ropes[b]);
                    rope_heads(k, hd, &ropes[b]);
                    KvElem::store(kd, k);
                    KvElem::store(vd, v);
                });
                let (kc, vc) = (&*kc, &*vc);

                // Causal attention, one (token, kv head) pair per pool task,
                // longest first: all `group` query heads sharing that kv head
                // together, so each key and value row is loaded once for all of
                // them. Token b sees positions 0..=start + b only. Per query head
                // the arithmetic is exactly `hidden`'s: same dot, same softmax
                // order, same fused `o += w v` order over t (`attend_group`).
                pool::for_each_chunk_mut_rev(&mut att, qn, |c, out| {
                    let (b, kvh) = (c / n_kv, c % n_kv);
                    let np = start + b + 1;
                    let qg = &qs[b][kvh * qn..(kvh + 1) * qn];
                    attend_group(qg, kc, vc, kv_dim, hd, kvh, np, out);
                });
            });

            let att_rows: Vec<&[f32]> = att.chunks(dim).collect();
            let (os, _, slots) = self.matmul_raw([&layer.wo], &att_rows);
            add_unbatched(&mut xs, &os, slots);

            per_token(&mut hs, dim, |b, h| {
                self.rmsnorm_into(&xs[b], &layer.ffn_norm, h)
            });
            let (acts, a_slots) = self.matmul_swiglu(&layer.w_gate, &layer.w_up, &hs);
            let aq = Self::quantize_cols(&acts, a_slots, nb, &[layer.w_down.qt()]);
            let (ds, _, slots) = self.matmul_q([&layer.w_down], &aq);
            add_unbatched(&mut xs, &ds, slots);
        }
        xs
    }

    /// Every layer for one token: store its keys and values in the cache and
    /// return the final hidden state (before the output norm).
    pub(crate) fn hidden(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        assert!(
            pos < cache.max_ctx,
            "position {pos} exceeds cache size {}",
            cache.max_ctx
        );
        let (dim, hd) = (self.dim, self.head_dim);
        let kv_dim = cache.kv_dim;

        // 1. Embedding: the token's row of the embedding matrix is its starting vector.
        let mut x = self.embed(token);

        let rope = self.rope_table(pos);
        // Attention scratch (step 5), kept on this thread across tokens so
        // decode attention allocates nothing once warm.
        let np = pos + 1;
        let tile = decode_tile(np, self.n_kv_heads);
        let threads = Pool::global().threads();
        let (mut buf, mut att_out) = DECODE.take();
        att_out.resize(dim, 0.0);
        for (l, layer) in self.layers.iter().enumerate() {
            // ---- Attention sub-block ----
            // 2. Normalize, then project into queries, keys and values (+ bias).
            let h = self.rmsnorm(&x, &layer.attn_norm);
            let [mut q, mut k, mut v] = self.matvec_many([&layer.wq, &layer.wk, &layer.wv], &h);
            if let Some(b) = &layer.bq {
                add_assign(&mut q, b);
            }
            if let Some(b) = &layer.bk {
                add_assign(&mut k, b);
            }
            if let Some(b) = &layer.bv {
                add_assign(&mut v, b);
            }

            // 3. Rotate q and k by position (RoPE), one head at a time.
            rope_heads(&mut q, hd, &rope);
            rope_heads(&mut k, hd, &rope);

            // 4. Store this position's k and v in the cache so later tokens can see them.
            //    (rounded to the cache's type: f16 unless AXON_KV=f32).
            with_kv_layer!(cache, l, |kc, vc| {
                let row = pos * kv_dim..(pos + 1) * kv_dim;
                KvElem::store(&mut kc[row.clone()], &k);
                KvElem::store(&mut vc[row], &v);

                // 5. Each query head attends over all positions 0..=pos (the work
                //    grows with context, so it can't stay serial). GQA: query head
                //    qh reads kv head qh / group. See `attend`.
                attend(
                    &q,
                    kc,
                    vc,
                    kv_dim,
                    hd,
                    np,
                    tile,
                    threads,
                    &mut buf,
                    &mut att_out,
                );
            });

            // 6. Mix the heads back together and add to the residual stream.
            let o = self.matvec(&layer.wo, &att_out);
            add_assign(&mut x[..dim], &o[..dim]);

            // ---- Feed-forward sub-block (SwiGLU) ----
            // 7. Normalize, then gate * up: silu(gate(h)) decides how much of up(h) passes.
            let h = self.rmsnorm(&x, &layer.ffn_norm);
            let act = self.matvec_swiglu(&layer.w_gate, &layer.w_up, &h);
            // 8. Project back down to model width and add to the residual stream.
            let d = self.matvec(&layer.w_down, &act);
            add_assign(&mut x[..dim], &d[..dim]);
        }
        DECODE.set((buf, att_out));

        x
    }
}

/// Below this much work (items x floats touched per item) a per-token step
/// runs inline on the caller: a pool fork/join costs more than, say, an
/// rmsnorm of a few tokens. A verify batch's elementwise steps stay inline;
/// prefill batches, the output-head transpose and quantization go wide.
const INLINE_WORK: usize = 16 * 1024;

/// Call `f(i, &mut data[i])` for every i, each item touching about
/// `floats` floats: inline if that's little work in total, else spread over
/// the pool in chunks of consecutive items (~4 chunks per thread).
fn per_token<T: Send>(data: &mut [T], floats: usize, f: impl Fn(usize, &mut T) + Sync) {
    if data.len() <= 1 || data.len() * floats <= INLINE_WORK {
        data.iter_mut().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let chunk = data.len().div_ceil(Pool::global().threads() * 4).max(1);
    pool::for_each_chunk_mut(data, chunk, |c, xs| {
        for (j, x) in xs.iter_mut().enumerate() {
            f(c * chunk + j, x);
        }
    });
}

/// `(0..n).map(f).collect()`, through `per_token` (same `floats` meaning).
fn par_map<T: Send>(n: usize, floats: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    per_token(&mut out, floats, |i, o| *o = Some(f(i)));
    out.into_iter().map(|o| o.unwrap()).collect()
}

/// Transpose a row-major (row, token slot) buffer, y[r * slots + b], into
/// one vector per token for the first `nb` slots.
fn unbatch(y: &[f32], slots: usize, nb: usize) -> Vec<Vec<f32>> {
    let n = y.len() / slots;
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); nb];
    per_token(&mut out, n, |b, x| *x = gather_col(y, slots, b, n));
    out
}

/// xs[b][i] += y[i * slots + b]: the residual add straight from a row-major
/// (row, token slot) buffer, same single add per element as adding
/// `unbatch(y)`.
fn add_unbatched(xs: &mut [Vec<f32>], y: &[f32], slots: usize) {
    let n = xs.first().map_or(0, |x| x.len());
    per_token(xs, n, |b, x| add_col(x, y, slots, b));
}

/// One pool chunk of a batched matmul: rows `first..first + ys.len() / slots`
/// (`row_of(g)` = (qtype, packed row g)) times every token slot of `xr`, into
/// ys[r * slots + b]. Rows two at a time (possibly from different matrices):
/// the pair shares each token's activation loads. An odd last row goes alone.
fn dot_rows_chunk<'w>(
    ys: &mut [f32],
    slots: usize,
    first: usize,
    row_of: &impl Fn(usize) -> (u32, &'w [u8]),
    xr: &[&QuantizedAct],
) {
    let n_rows = ys.len() / slots;
    let mut pairs = ys.chunks_exact_mut(2 * slots);
    for (j, yp) in pairs.by_ref().enumerate() {
        let g = first + 2 * j;
        let (q0, r0) = row_of(g);
        let (q1, r1) = row_of(g + 1);
        let (y0, y1) = yp.split_at_mut(slots);
        dot_rows_q8_multi([q0, q1], [r0, r1], xr, [y0, y1]);
    }
    let yr = pairs.into_remainder();
    if !yr.is_empty() {
        let (qt, row) = row_of(first + n_rows - 1);
        dot_row_q8_multi(qt, row, xr, yr);
    }
}

/// Running argmax with main.rs `argmax`'s exact rules: the largest value
/// by `>`/`==` (so NaN is never picked and -0.0 ties +0.0), and of equal
/// maxima the LAST index. Pushing values in index order, or merging the
/// `Best`s of consecutive index ranges in order, gives that index: a later
/// value/range replaces the current one iff it is greater or equal.
/// `index(n)` is n - 1 when nothing qualified (all NaN), as `argmax` does.
#[derive(Clone, Copy, Debug)]
pub struct Best {
    v: f32,
    i: Option<u32>,
}

impl Best {
    pub const NONE: Best = Best {
        v: f32::NEG_INFINITY,
        i: None,
    };

    #[inline]
    pub fn push(&mut self, v: f32, i: u32) {
        // `>=` is false when either side is NaN, so NaN is skipped.
        if v >= self.v {
            *self = Best { v, i: Some(i) };
        }
    }

    pub fn merge(&mut self, later: Best) {
        if let Some(i) = later.i {
            self.push(later.v, i);
        }
    }

    pub fn index(self, n: usize) -> u32 {
        self.i.unwrap_or(n as u32 - 1)
    }
}

thread_local! {
    /// Per-task matmul scratch (head rows of `matmul_argmax`, swiglu's up
    /// values); grows to the largest seen and is reused across passes.
    static ROWS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` on this thread's matmul scratch, exactly `n` long. Contents are
/// stale; `f` must write before reading.
///
/// Not re-entrant: `f` must not call `with_rows_scratch` again, directly or
/// through anything it runs (`dot_rows_chunk` and the quant kernels don't).
/// A nested call would borrow the RefCell twice; it panics with a message
/// saying so rather than aliasing the buffer.
fn with_rows_scratch<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    ROWS.with(|cell| {
        let mut s = cell
            .try_borrow_mut()
            .expect("with_rows_scratch re-entered: one scratch per thread");
        if s.len() < n {
            s.resize(n, 0.0);
        }
        f(&mut s[..n])
    })
}

thread_local! {
    /// Prefill attention scores of the task in hand; grows to the largest seen.
    static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` on this thread's score scratch, at least `n` long. Contents are
/// left over from earlier tasks: write before reading.
fn with_scores<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    SCORES.with_borrow_mut(|s| {
        if s.len() < n {
            s.resize(n, 0.0);
        }
        f(&mut s[..n])
    })
}

thread_local! {
    /// Decode attention scratch (`attend`'s buffer, attention output), taken
    /// by `hidden` for a token and put back after.
    static DECODE: std::cell::Cell<(Vec<f32>, Vec<f32>)> = const { std::cell::Cell::new((Vec::new(), Vec::new())) };
}

/// Decode contexts shorter than this score one task per query head (plain
/// `dot`): K is small enough to stay in cache across the group's re-reads,
/// and it's one fork/join instead of two. Untuned guess (no benchmarks).
const TILE_MIN_NP: usize = 512;
/// Fewest positions per tiled scoring task.
const TILE_MIN_ROWS: usize = 32;

/// `attend`'s scoring plan for `np` positions: None for per-query-head tasks,
/// or Some(rows per (kv head, t-range) task), aiming at ~3 tasks per thread.
fn decode_tile(np: usize, n_kv: usize) -> Option<usize> {
    if np < TILE_MIN_NP {
        return None;
    }
    let ranges = (Pool::global().threads() * 3).div_ceil(n_kv).max(1);
    Some(np.div_ceil(ranges).max(TILE_MIN_ROWS))
}

/// Decode attention of one token over cache positions 0..np: out[qh * hd..]
/// = sum over t of softmax_t(q_qh · k_t / sqrt(hd)) * v_t, with q head qh on
/// kv head qh / group. kc and vc are [t][kv_dim] (kv_dim = n_kv * hd) of the
/// cache's element type, widened to f32 exactly on load; `buf` is reusable
/// scratch, grown as needed.
///
/// Scores, `tile` None: one pool task per query head, `dot` per position.
/// `tile` Some(tc): one task per (kv head, tc positions), each K row loaded
/// once for all `group` heads (`dot_group`, which is `dot` per head) into
/// sc[kvh][t][g]; then one task per query head runs `softmax_from` on its
/// strided scores into wts. Either way wts[qh][t] gets the same dot, the same
/// `* scale` and the same softmax ops in the same order.
///
/// Values: one task per (kv head, column of one 64-byte line of each V row
/// (`line_dims`), subset of the group's heads), the subsets just enough that
/// there are at least `threads` tasks (`vsum_split`); each task one pass over
/// its column (`vsum_rows`), then copied into place. Every output element is
/// the same fused-multiply-add chain over t however it's split.
#[allow(clippy::too_many_arguments)]
fn attend<T: KvElem>(
    q: &[f32],
    kc: &[T],
    vc: &[T],
    kv_dim: usize,
    hd: usize,
    np: usize,
    tile: Option<usize>,
    threads: usize,
    buf: &mut Vec<f32>,
    out: &mut [f32],
) {
    let n_heads = q.len() / hd;
    let n_kv = kv_dim / hd;
    let group = n_heads / n_kv;
    let qn = group * hd;
    debug_assert_eq!(n_kv * qn, q.len(), "attend: heads * head_dim != q len");
    assert!(np * kv_dim <= kc.len().min(vc.len()) && out.len() >= q.len());
    let scale = 1.0 / (hd as f32).sqrt();
    // Per kv head, `nt` t-ranges of `tc` rows; the last may be short and
    // its tail of sc is left unused, so chunks never straddle kv heads.
    let (tc, nt) = tile.map_or((0, 0), |tc| (tc, np.div_ceil(tc)));
    let cw = line_dims::<T>();
    let ncol = hd.div_ceil(cw);
    let nsub = vsum_split(threads, n_kv * ncol, group);
    let hmax = group.div_ceil(nsub);
    let (nw, ns, nv) = (
        n_heads * np,
        n_kv * nt * tc * group,
        n_kv * ncol * nsub * hmax * cw,
    );
    if buf.len() < nw + ns + nv {
        buf.resize(nw + ns + nv, 0.0);
    }
    let (wts, rest) = buf.split_at_mut(nw);
    let (sc, rest) = rest.split_at_mut(ns);
    let vparts = &mut rest[..nv];

    if tile.is_none() {
        pool::for_each_chunk_mut(wts, np, |qh, s| {
            let qv = &q[qh * hd..(qh + 1) * hd];
            score_softmax(qv, kc, kv_dim, qh / group * hd, scale, s);
        });
    } else {
        pool::for_each_chunk_mut(sc, tc * group, |c, s| {
            let (kvh, t0) = (c / nt, c % nt * tc);
            let qg = &q[kvh * qn..(kvh + 1) * qn];
            score_tile(qg, kc, kv_dim, hd, kvh * hd, t0..np.min(t0 + tc), scale, s);
        });
        let sc = &*sc;
        pool::for_each_chunk_mut(wts, np, |qh, w| {
            let (kvh, g) = (qh / group, qh % group);
            softmax_from(&sc[kvh * nt * tc * group + g..], group, w);
        });
    }

    // vparts[c]: task c = (kvh, col, sub), heads g0..g1 of the group, dims
    // col * cw..+ width, as [head][width].
    let wts = &*wts;
    let task = |c: usize| {
        let (kvh, col, j) = (c / (ncol * nsub), c / nsub % ncol, c % nsub);
        let (g0, g1) = (j * group / nsub, (j + 1) * group / nsub);
        let d0 = col * cw;
        (kvh, g0, g1, d0, cw.min(hd - d0))
    };
    pool::for_each_chunk_mut(vparts, hmax * cw, |c, o| {
        let (kvh, g0, g1, d0, width) = task(c);
        let w = &wts[(kvh * group + g0) * np..];
        let vb = &vc[kvh * hd + d0..];
        vsum_rows(
            w,
            np,
            np,
            g1 - g0,
            vb,
            kv_dim,
            &mut o[..(g1 - g0) * width],
            width,
        );
    });
    for (c, part) in vparts.chunks_exact(hmax * cw).enumerate() {
        let (kvh, g0, g1, d0, width) = task(c);
        for (g, p) in (g0..g1).zip(part.chunks_exact(width)) {
            let o0 = (kvh * group + g) * hd + d0;
            out[o0..o0 + width].copy_from_slice(p);
        }
    }
}

/// Elements of T per 64-byte cache line: 32 halves or 16 floats.
fn line_dims<T>() -> usize {
    64 / std::mem::size_of::<T>()
}

/// Head subsets per (kv head, column) V-sum task: enough that `cols`
/// columns give at least `threads` tasks, but no more than `group` (one head
/// per task).
fn vsum_split(threads: usize, cols: usize, group: usize) -> usize {
    threads.div_ceil(cols.max(1)).clamp(1, group.max(1))
}

/// Prefill attention of one token's `group` query heads qg (contiguous) on
/// kv head kvh over cache positions 0..np, into out ([head][hd]): the same
/// dot, `* scale`, softmax and V-sum order per head as `attend`.
#[allow(clippy::too_many_arguments)]
fn attend_group<T: KvElem>(
    qg: &[f32],
    kc: &[T],
    vc: &[T],
    kv_dim: usize,
    hd: usize,
    kvh: usize,
    np: usize,
    out: &mut [f32],
) {
    let group = qg.len() / hd;
    let scale = 1.0 / (hd as f32).sqrt();
    with_scores((np + 1) * group, |sc| {
        attend_causal(qg, kc, vc, kv_dim, hd, kvh, np, scale, sc, out)
    });
}

// ---------------------------------------------------------------------------
// Attention, multiversioned (see mv.rs) with the fused-multiply-add switch F:
// every `a * b + c` step below is `vmath::fma::<F>`, one rounding when the CPU
// has FMA (both variants), a multiply and an add when it doesn't. The `V3`
// variants use the hand-written AVX2 + FMA helpers (`*_v3`), the same
// operations in the same order as the scalar code with F.

multiversion! {
    /// Decode scores of one query head, then `softmax`: s[t] =
    /// dot(qv, k_t) * scale with k_t = kc[t * kv_dim + k_off..][..qv.len()].
    fn score_softmax<V3, F, T: KvElem>(
        qv: &[f32],
        kc: &[T],
        kv_dim: usize,
        k_off: usize,
        scale: f32,
        s: &mut [f32],
    ) {
        let hd = qv.len();
        for (t, s) in s.iter_mut().enumerate() {
            let k0 = t * kv_dim + k_off;
            // SAFETY: V3 only in the v3 variant.
            *s = unsafe { dot::<V3, F, T>(qv, &kc[k0..k0 + hd]) } * scale;
        }
        // SAFETY: as above.
        unsafe { vmath::softmax::<V3, F>(s) };
    }
}

multiversion! {
    /// Decode scores of one kv head's query group qg (heads of hd floats)
    /// over positions `ts`: s[j * group + g] = dot(q_g, k_t) * scale for the
    /// j-th t, `dot_group` per position (k_t = kc[t * kv_dim + k_off..][..hd]).
    fn score_tile<V3, F, T: KvElem>(
        qg: &[f32],
        kc: &[T],
        kv_dim: usize,
        hd: usize,
        k_off: usize,
        ts: std::ops::Range<usize>,
        scale: f32,
        s: &mut [f32],
    ) {
        let group = qg.len() / hd;
        for (t, row) in ts.zip(s.chunks_exact_mut(group)) {
            let k0 = t * kv_dim + k_off;
            // SAFETY: V3 only in the v3 variant.
            unsafe { dot_group::<V3, F, T>(qg, &kc[k0..k0 + hd], row) };
            for x in row.iter_mut() {
                *x *= scale;
            }
        }
    }
}

multiversion! {
    /// `attend_group`'s body: `sc` is scratch of (np + 1) * group floats,
    /// sc[g * np + t] head g's score at t (contiguous per head for the
    /// softmax), the rest one position's dots.
    fn attend_causal<V3, F, T: KvElem>(
        qg: &[f32],
        kc: &[T],
        vc: &[T],
        kv_dim: usize,
        hd: usize,
        kvh: usize,
        np: usize,
        scale: f32,
        sc: &mut [f32],
        out: &mut [f32],
    ) {
        let group = qg.len() / hd;
        let (sc, row) = sc.split_at_mut(np * group);
        for t in 0..np {
            let k0 = t * kv_dim + kvh * hd;
            // SAFETY (every call): V3 only in the v3 variant.
            unsafe { dot_group::<V3, F, T>(qg, &kc[k0..k0 + hd], row) };
            for (g, &s) in row.iter().enumerate() {
                sc[g * np + t] = s * scale;
            }
        }
        for s in sc.chunks_exact_mut(np) {
            unsafe { vmath::softmax::<V3, F>(s) };
        }
        let vb = &vc[kvh * hd..];
        unsafe { vsum_rows::imp::<V3, F, T>(sc, np, np, group, vb, kv_dim, out, hd) };
    }
}

multiversion! {
    /// `softmax` of the scores src[t * stride], t in 0..dst.len(), written to
    /// dst: gathered, then `vmath::softmax` in place, so the same bits as
    /// `softmax` on a copy.
    fn softmax_from<V3, F>(src: &[f32], stride: usize, dst: &mut [f32]) {
        let n = dst.len();
        assert!(
            n == 0 || (n - 1) * stride < src.len(),
            "softmax_from: short src"
        );
        for (t, d) in dst.iter_mut().enumerate() {
            *d = src[t * stride];
        }
        // SAFETY: V3 only in the v3 variant.
        unsafe { vmath::softmax::<V3, F>(dst) };
    }
}

multiversion! {
    /// o[h * os + d] = sum over t in 0..np, in order from 0.0, of
    /// w[h * ws + t] * v_t[d] (each step `fma::<F>`, v widened to f32), for
    /// heads h in 0..nh and d in 0..os, with value rows at a fixed stride:
    /// v_t = vb[t * vs..][..os].
    ///
    /// V3: blocks of up to 3 heads x 32 dims (12 accumulators) per pass over
    /// the rows (`vsum_rows_v3`), each row slice loaded once per block. Per
    /// element that's the scalar `o = fma(w, v, o)` over t, so the blocking
    /// never changes a bit.
    fn vsum_rows<V3, F, T: KvElem>(
        w: &[f32],
        ws: usize,
        np: usize,
        nh: usize,
        vb: &[T],
        vs: usize,
        o: &mut [f32],
        os: usize,
    ) {
        assert!(
            np <= ws && (nh == 0 || (nh - 1) * ws + np <= w.len()) && nh * os <= o.len(),
            "vsum_rows: short w/o"
        );
        assert!(
            np == 0 || (np - 1) * vs + os <= vb.len(),
            "vsum_rows: short v"
        );
        let mut d0 = 0;
        #[cfg(target_arch = "x86_64")]
        if V3 {
            // SAFETY: the v3 variant; bounds asserted above.
            unsafe { vsum_rows_v3(w, ws, np, nh, vb, vs, o, os) };
            d0 = os / 8 * 8;
        }
        for h in 0..nh {
            for d in d0..os {
                let mut a = 0.0f32;
                for t in 0..np {
                    a = vmath::fma::<F>(w[h * ws + t], vb[t * vs + d].to_f32(), a);
                }
                o[h * os + d] = a;
            }
        }
    }
}

/// The whole 8-dim chunks of `vsum_rows` (d < os / 8 * 8) with AVX2 + FMA.
///
/// # Safety
/// The CPU must have the x86-64-v3 features; the bounds `vsum_rows` asserts.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,f16c,fma,lzcnt")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn vsum_rows_v3<T: KvElem>(
    w: &[f32],
    ws: usize,
    np: usize,
    nh: usize,
    vb: &[T],
    vs: usize,
    o: &mut [f32],
    os: usize,
) {
    let nc = os / 8;
    for h0 in (0..nh).step_by(3) {
        let g = (nh - h0).min(3);
        let mut c0 = 0;
        while c0 < nc {
            let c = match nc - c0 {
                1 => 1,
                2 | 3 => 2,
                _ => 4,
            };
            let (wp, vp) = (w[h0 * ws..].as_ptr(), vb[8 * c0..].as_ptr());
            let op = o[h0 * os + 8 * c0..].as_mut_ptr();
            // SAFETY: heads h0..h0 + g <= nh, chunks c0..c0 + c <= nc, so
            // every read and write is inside the bounds `vsum_rows` asserts.
            unsafe {
                match (g, c) {
                    (1, 1) => vsum_block::<T, 1, 1>(wp, ws, np, vp, vs, op, os),
                    (1, 2) => vsum_block::<T, 1, 2>(wp, ws, np, vp, vs, op, os),
                    (1, _) => vsum_block::<T, 1, 4>(wp, ws, np, vp, vs, op, os),
                    (2, 1) => vsum_block::<T, 2, 1>(wp, ws, np, vp, vs, op, os),
                    (2, 2) => vsum_block::<T, 2, 2>(wp, ws, np, vp, vs, op, os),
                    (2, _) => vsum_block::<T, 2, 4>(wp, ws, np, vp, vs, op, os),
                    (_, 1) => vsum_block::<T, 3, 1>(wp, ws, np, vp, vs, op, os),
                    (_, 2) => vsum_block::<T, 3, 2>(wp, ws, np, vp, vs, op, os),
                    (_, _) => vsum_block::<T, 3, 4>(wp, ws, np, vp, vs, op, os),
                }
            }
            c0 += c;
        }
    }
}

/// G heads x C 8-dim chunks of `vsum_rows`: o[g * os + 8c..+ 8] = the fused
/// chain over t of w[g * ws + t] * vb[t * vs + 8c..+ 8].
///
/// # Safety
/// The CPU must have the x86-64-v3 features. w[g * ws + t] (g < G, t < np)
/// readable, vb[t * vs..+ 8 * C] (t < np) readable, o[g * os..+ 8 * C]
/// writable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,f16c,fma,lzcnt")]
#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // fixed-size register tiles: indexed loops unroll cleanly
unsafe fn vsum_block<T: KvElem, const G: usize, const C: usize>(
    w: *const f32,
    ws: usize,
    np: usize,
    vb: *const T,
    vs: usize,
    o: *mut f32,
    os: usize,
) {
    use std::arch::x86_64::*;
    // SAFETY: the caller's contract; the row pointer only advances with
    // wrapping_add and is dereferenced for t < np.
    unsafe {
        let mut acc = [[_mm256_setzero_ps(); C]; G];
        let mut vp = vb;
        for t in 0..np {
            for c in 0..C {
                let y = T::load8(vp.add(8 * c));
                for g in 0..G {
                    let x = _mm256_set1_ps(*w.add(g * ws + t));
                    acc[g][c] = _mm256_fmadd_ps(x, y, acc[g][c]);
                }
            }
            vp = vp.wrapping_add(vs);
        }
        for g in 0..G {
            for c in 0..C {
                _mm256_storeu_ps(o.add(g * os + 8 * c), acc[g][c]);
            }
        }
    }
}

/// out[g] = dot(q_g, k) for the query vectors q_g = q[g * k.len()..][..k.len()],
/// g in 0..out.len(), bit for bit; up to 8 at a time (`dot_multi_v3` when V3,
/// else `dot` per vector). Not dispatched on its own: the attention bodies
/// call it with their own `V3` and `F`.
///
/// # Safety
/// With V3, the CPU must have the x86-64-v3 features.
#[inline(always)]
unsafe fn dot_group<const V3: bool, const F: bool, T: KvElem>(q: &[f32], k: &[T], out: &mut [f32]) {
    let n = k.len();
    debug_assert_eq!(q.len(), n * out.len());
    #[cfg(target_arch = "x86_64")]
    if V3 {
        for (qs, os) in q.chunks(8 * n).zip(out.chunks_mut(8)) {
            // SAFETY (every arm): our caller's contract.
            unsafe {
                match os.len() {
                    8 => dot_multi_v3::<T, 8>(qs, k, os),
                    7 => dot_multi_v3::<T, 7>(qs, k, os),
                    6 => dot_multi_v3::<T, 6>(qs, k, os),
                    5 => dot_multi_v3::<T, 5>(qs, k, os),
                    4 => dot_multi_v3::<T, 4>(qs, k, os),
                    3 => dot_multi_v3::<T, 3>(qs, k, os),
                    2 => dot_multi_v3::<T, 2>(qs, k, os),
                    _ => dot_multi_v3::<T, 1>(qs, k, os),
                }
            }
        }
        return;
    }
    for (g, o) in out.iter_mut().enumerate() {
        // SAFETY: V3 false here.
        *o = unsafe { dot::<false, F, T>(&q[g * n..(g + 1) * n], k) };
    }
}

/// `dot` of G query vectors (contiguous in q) with one k, with AVX2 + FMA.
/// Each result uses `dot`'s exact lanes, tail and final sum, so it's
/// identical to `dot`; the G accumulator chains run side by side and k is
/// loaded (and widened) once.
///
/// Written with intrinsics: the plain-Rust version (an `[[f32; 8]; G]`
/// accumulator indexed through per-head slices) didn't vectorize and ran as
/// scalar loads/adds through the stack, ~40% of prefill at 2k tokens.
/// `_mm256_fmadd_ps(x, y, acc)` is `dot`'s `acc[i] = fma(x[i], y[i], acc[i])`
/// lane for lane, and `vmath::x86::hsum8` is `vmath::sum8`.
///
/// # Safety
/// The CPU must have the x86-64-v3 features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,f16c,fma,lzcnt")]
#[inline]
unsafe fn dot_multi_v3<T: KvElem, const G: usize>(q: &[f32], k: &[T], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = k.len();
    assert!(q.len() >= G * n && out.len() >= G, "dot_multi: short q/out");
    let nc = n / 8;
    let (qp, kp) = (q.as_ptr(), k.as_ptr());
    // SAFETY: Every load reads q[g * n + 8 * j .. + 8] with g < G and
    // 8 * j + 8 <= n, inside q (asserted above), or k[8 * j..+ 8], inside k.
    let acc = unsafe {
        let mut acc = [_mm256_setzero_ps(); G];
        for j in 0..nc {
            let y = T::load8(kp.add(8 * j));
            for (g, a) in acc.iter_mut().enumerate() {
                let x = _mm256_loadu_ps(qp.add(g * n + 8 * j));
                *a = _mm256_fmadd_ps(x, y, *a);
            }
        }
        acc
    };
    for (g, a) in acc.iter().enumerate() {
        let ar = &q[g * n + 8 * nc..(g + 1) * n];
        out[g] = vmath::x86::hsum8(*a) + dot_tail::<true, T>(ar, &k[8 * nc..]);
    }
}

/// Elements past the last whole 8-chunk: `fma::<F>` in order from 0.0.
#[inline(always)]
fn dot_tail<const F: bool, T: KvElem>(a: &[f32], b: &[T]) -> f32 {
    let mut s = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        s = vmath::fma::<F>(x, y.to_f32(), s);
    }
    s
}

/// Dot product with 8 independent partial sums (lane i: `fma::<F>` over
/// elements i mod 8 of the whole chunks), combined by `vmath::sum8`, plus
/// the tail. `dot_multi_v3::<1>` when V3. Not dispatched on its own.
///
/// # Safety
/// With V3, the CPU must have the x86-64-v3 features.
#[inline(always)]
unsafe fn dot<const V3: bool, const F: bool, T: KvElem>(a: &[f32], b: &[T]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if V3 {
        let mut o = [0.0f32];
        // SAFETY: our caller's contract.
        unsafe { dot_multi_v3::<T, 1>(a, b, &mut o) };
        return o[0];
    }
    let mut acc = [0.0f32; 8];
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..8 {
            acc[i] = vmath::fma::<F>(x[i], y[i].to_f32(), acc[i]);
        }
    }
    vmath::sum8(&acc) + dot_tail::<F, T>(ar, br)
}

// ---------------------------------------------------------------------------
// The hot loops outside the quant kernels, each compiled twice (portable and
// x86-64-v3, chosen at runtime: see mv.rs). Bodies are closure-free.

multiversion! {
    /// RMSNorm: scale x so its root-mean-square is 1, then multiply by a
    /// learned weight. The sum of squares is in order from -0.0 (`sum`'s order).
    fn rmsnorm_into<V3>(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
        let mut ss = -0.0f32;
        for v in x {
            ss += v * v;
        }
        let ms = ss / x.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
            *o = v * inv * g;
        }
    }
}

/// NEOX-style RoPE on one head vector (len head_dim) in place: dim i is
/// paired with dim i + head_dim/2 and the pair rotated by `table[i]`.
#[inline(always)]
fn rope_head(v: &mut [f32], table: &[(f32, f32)]) {
    let half = table.len();
    for (i, &(sin, cos)) in table.iter().enumerate() {
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * cos - b * sin;
        v[i + half] = a * sin + b * cos;
    }
}

multiversion! {
    /// `rope_head` on every head of v (heads of `hd` floats).
    fn rope_heads<V3>(v: &mut [f32], hd: usize, table: &[(f32, f32)]) {
        for head in v.chunks_mut(hd) {
            rope_head(head, table);
        }
    }
}

multiversion! {
    /// a[i] += b[i] (bias and residual adds).
    fn add_assign<V3>(a: &mut [f32], b: &[f32]) {
        for (a, b) in a.iter_mut().zip(b) {
            *a += b;
        }
    }
}

multiversion! {
    /// x[i] += y[i * slots + b]: one token's residual add from a row-major
    /// (row, token slot) buffer.
    fn add_col<V3>(x: &mut [f32], y: &[f32], slots: usize, b: usize) {
        for (i, x) in x.iter_mut().enumerate() {
            *x += y[i * slots + b];
        }
    }
}

multiversion! {
    /// Token slot b's column of a row-major (row, token slot) buffer of n rows.
    fn gather_col<V3>(y: &[f32], slots: usize, b: usize, n: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; n];
        for (r, x) in x.iter_mut().enumerate() {
            *x = y[r * slots + b];
        }
        x
    }
}

multiversion! {
    /// `Best::push` of every row's logit for each token: ys is row-major
    /// (row, token slot), rows numbered from `first`, in row order.
    fn best_rows<V3>(ys: &[f32], slots: usize, first: usize, bs: &mut [Best]) {
        for (b, best) in bs.iter_mut().enumerate() {
            for (r, y) in ys.chunks_exact(slots).enumerate() {
                best.push(y[b], (first + r) as u32);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::vmath::fmadd;

    /// `dot` on the path the model runs (`quant::v3()`, `vmath::fused()`).
    pub(crate) fn dot<T: KvElem>(a: &[f32], b: &[T]) -> f32 {
        // SAFETY: V3 only when v3() detected the features.
        unsafe {
            match (crate::quant::v3(), vmath::fused()) {
                (true, _) => super::dot::<true, true, T>(a, b),
                (false, true) => super::dot::<false, true, T>(a, b),
                (false, false) => super::dot::<false, false, T>(a, b),
            }
        }
    }

    /// `dot_group` on the path the model runs.
    pub(crate) fn dot_group<T: KvElem>(q: &[f32], k: &[T], out: &mut [f32]) {
        // SAFETY: as in `dot`.
        unsafe {
            match (crate::quant::v3(), vmath::fused()) {
                (true, _) => super::dot_group::<true, true, T>(q, k, out),
                (false, true) => super::dot_group::<false, true, T>(q, k, out),
                (false, false) => super::dot_group::<false, false, T>(q, k, out),
            }
        }
    }
    use crate::kv::Half;

    /// The CPU's decode attention (`attend`) of query heads q over f16 keys
    /// and values [t][kv_dim] at positions 0..np, `tile` as `decode_tile`.
    pub(crate) fn attend_cpu(
        q: &[f32],
        kc: &[Half],
        vc: &[Half],
        kv_dim: usize,
        hd: usize,
        np: usize,
        tile: Option<usize>,
    ) -> Vec<f32> {
        let mut out = vec![0f32; q.len()];
        let mut buf = Vec::new();
        let threads = Pool::global().threads();
        attend(q, kc, vc, kv_dim, hd, np, tile, threads, &mut buf, &mut out);
        out
    }

    /// The CPU's RMSNorm.
    pub(crate) fn rmsnorm_cpu(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        rmsnorm_into(x, w, eps, &mut out);
        out
    }

    /// A tiny well-formed qwen2: dim 64, 2 heads (1 kv), ffn 128, vocab 8, `layers`
    /// layers; `edit` changes the tensor list before the file is built.
    fn tiny(edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>)) -> std::io::Result<()> {
        tiny_meta(edit, |_| {})
    }

    fn tiny_meta(
        edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>),
        edit_meta: impl FnOnce(&mut Vec<(&'static str, Val)>),
    ) -> std::io::Result<()> {
        let mut ts: Vec<(String, Vec<u64>, u32)> = vec![
            ("token_embd.weight".into(), vec![64, 8], Q8_0),
            ("output_norm.weight".into(), vec![64], F32),
        ];
        for i in 0..2 {
            for (n, d, ty) in [
                ("attn_norm.weight", vec![64], F32),
                ("attn_q.weight", vec![64, 64], Q8_0),
                ("attn_k.weight", vec![64, 32], Q8_0),
                ("attn_v.weight", vec![64, 32], Q8_0),
                ("attn_q.bias", vec![64], F32),
                ("attn_output.weight", vec![64, 64], Q8_0),
                ("ffn_norm.weight", vec![64], F32),
                ("ffn_gate.weight", vec![64, 128], Q8_0),
                ("ffn_up.weight", vec![64, 128], Q8_0),
                ("ffn_down.weight", vec![128, 64], Q8_0),
            ] {
                ts.push((format!("blk.{i}.{n}"), d, ty));
            }
        }
        edit(&mut ts);
        let mut meta = vec![
            ("general.architecture", Val::Str("qwen2".into())),
            ("qwen2.block_count", Val::U32(2)),
            ("qwen2.embedding_length", Val::U32(64)),
            ("qwen2.attention.head_count", Val::U32(2)),
            ("qwen2.attention.head_count_kv", Val::U32(1)),
            (
                "tokenizer.ggml.tokens",
                Val::Arr((0..8).map(|i| Val::Str(format!("t{i}"))).collect()),
            ),
        ];
        edit_meta(&mut meta);
        let ts: Vec<(&str, &[u64], u32)> = ts
            .iter()
            .map(|(n, d, t)| (n.as_str(), &d[..], *t))
            .collect();
        check_supported(&crate::gguf::build_for_test(&meta, &ts))
    }

    fn rejects(edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>), want: &str) {
        let e = tiny(edit).expect_err("malformed model accepted");
        assert!(e.to_string().contains(want), "{e}");
    }

    #[test]
    fn check_supported_accepts_tiny_model() {
        tiny(|_| {}).expect("tiny model");
    }

    #[test]
    fn check_supported_rejects_bad_models() {
        let set = |name: &'static str, dims: Vec<u64>, ty: u32| {
            move |ts: &mut Vec<(String, Vec<u64>, u32)>| {
                let t = ts.iter_mut().find(|t| t.0 == name).unwrap();
                (t.1, t.2) = (dims, ty);
            }
        };
        rejects(
            |ts| ts.retain(|t| t.0 != "blk.1.ffn_up.weight"),
            "missing tensor blk.1.ffn_up.weight",
        );
        // Q4_0: a real GGUF type the kernels don't run; 99: unknown.
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], 2),
            "Q4_0 (2) is not supported",
        );
        rejects(
            set("blk.1.ffn_down.weight", vec![128, 64], 99),
            "?? (99) is not supported",
        );
        rejects(
            set("blk.0.attn_k.weight", vec![64, 64], Q8_0),
            "attn_k.weight: shape",
        );
        rejects(
            set("blk.0.ffn_down.weight", vec![64, 64], Q8_0),
            "ffn_down.weight: shape",
        );
        rejects(
            set("output_norm.weight", vec![32], F32),
            "output_norm.weight: shape",
        );
        // K-quant rows must be whole 256-blocks: 64 isn't.
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], Q4_K),
            "attn_q.weight: shape",
        );
        rejects(set("token_embd.weight", vec![64, 4], Q8_0), "8 tokens");
        let meta = |edit: fn(&mut Vec<(&'static str, Val)>), want: &str| {
            let e = tiny_meta(|_| {}, edit).expect_err("malformed metadata accepted");
            assert!(e.to_string().contains(want), "{e}");
        };
        meta(
            |m| m.retain(|kv| kv.0 != "qwen2.block_count"),
            "qwen2.block_count is missing",
        );
        meta(
            |m| m[3].1 = Val::Str("2".into()),
            "head_count is missing or not a number",
        );
        meta(|m| m[3].1 = Val::U32(0), "not a positive integer");
        meta(|m| m[4].1 = Val::U32(3), "not a multiple of head_count_kv");
        meta(|m| m[1].1 = Val::U32(3), "missing tensor blk.2.");
        meta(
            |m| m.retain(|kv| kv.0 != "general.architecture"),
            "'<none>' is not supported",
        );
    }

    /// Batched prefill in uneven chunks must leave the same cache, and so give
    /// the same next-token logits, bit for bit, as feeding one token at a time.
    #[test]
    fn feed_batch_matches_feed() {
        for repack in [true, false] {
            // A fresh load each: repacking releases the originals.
            let Some(m) = crate::testutil::model() else {
                return;
            };
            let net = Qwen2::new_with(&m, repack);
            let repacked = net.layers.iter().filter(|l| l.w_down.r8.is_some()).count();
            assert_eq!(
                repacked > 0,
                repack,
                "repack {repack}: {repacked} ffn_down repacked"
            );
            // Every Q5_0/Q8_0 matrix (0.5B: attention, gate/up, the head)
            // goes through r8q when repacking, none otherwise.
            let all = net
                .layers
                .iter()
                .flat_map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]);
            for w in all.chain([net.head()]) {
                if matches!(w.qt(), Q5_0 | Q8_0) {
                    assert_eq!(w.r8.is_some(), repack, "{}", w.t.name);
                }
            }
            feed_batch_matches_feed_on(&net);
        }
    }

    /// Bit fingerprints of every matmul path on the real model, as
    /// `Qwen2::new` builds it (AXON_REPACK / AXON_SIMD pick the kernels):
    /// decode (`forward` token by token), prefill (`feed_batch` then
    /// `forward`) and verify (`verify_batch`, the batched head). Asserts the
    /// three agree bit for bit and prints one hash, so runs under different
    /// settings or builds can be compared (`AXON_REPACK=0` must print the
    /// same hash before and after a repack change; scalar/avx2/vnni must
    /// print the same hash when everything is repacked).
    #[test]
    fn path_hashes() {
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let net = Qwen2::new(&m);
        let ids: Vec<u32> = (0..24u64)
            .map(|i| (i * 2_654_435_761 % 150_000) as u32)
            .collect();
        let hash = |ls: &[Vec<f32>]| {
            ls.iter()
                .flatten()
                .fold(0u64, |h, v| h.rotate_left(5) ^ v.to_bits() as u64)
        };
        let split = 15;
        // decode: every position one token at a time
        let mut c = net.new_cache(ids.len());
        let dec: Vec<Vec<f32>> = (0..ids.len())
            .map(|p| net.forward(ids[p], p, &mut c))
            .collect();
        // prefill: a batch, then decode the rest
        let mut c = net.new_cache(ids.len());
        net.feed_batch(&ids[..split], 0, &mut c);
        let pre: Vec<Vec<f32>> = (split..ids.len())
            .map(|p| net.forward(ids[p], p, &mut c))
            .collect();
        // verify: the rest as one batch through the batched head
        let mut c = net.new_cache(ids.len());
        net.feed_batch(&ids[..split], 0, &mut c);
        let ver = net.verify_batch(&ids[split..], split, &mut c);
        let tail = &dec[split..];
        let (hd, hp, hv) = (hash(tail), hash(&pre), hash(&ver));
        let repacked: Vec<&str> = net
            .layers
            .iter()
            .flat_map(|l| [&l.wq, &l.wv, &l.w_gate, &l.w_down])
            .chain([net.head()])
            .filter(|w| w.r8.is_some())
            .map(|w| w.t.qtype.name)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        println!(
            "path hashes: all decode {:016x} tail decode {hd:016x} prefill {hp:016x} \
             verify {hv:016x}; repacked types {repacked:?}",
            hash(&dec)
        );
        assert_eq!(hd, hp, "prefill differs from decode");
        assert_eq!(hd, hv, "verify differs from decode");
    }

    /// The r8 kernels against the per-row ones on the real model: same
    /// prompt, logits compared. Prints the deltas only. This random-token
    /// prompt is chaotic: today's per-row kernels alone disagree with
    /// themselves here (AXON_SIMD=scalar argmax 220, avx2/vnni 2090, max
    /// |dlogprob| ~0.3), so r8's delta of the same size proves nothing either
    /// way. The bound is only a gross-bug tripwire; accuracy is asserted per
    /// row against f64 (`repack_rows_match_exact_on_model`, quant r8 tests)
    /// and on real text by `matches_ollama`.
    #[test]
    fn repack_logit_deltas() {
        if crate::testutil::model_path().is_none() {
            return;
        }
        let ids: Vec<u32> = (0..40u64)
            .map(|i| (i * 2_654_435_761 % 150_000) as u32)
            .collect();
        let run = |repack: bool| {
            let m = crate::testutil::model().unwrap();
            let net = Qwen2::new_with(&m, repack);
            let mut c = net.new_cache(ids.len() + 1);
            net.feed_batch(&ids[..ids.len() - 1], 0, &mut c);
            net.forward(ids[ids.len() - 1], ids.len() - 1, &mut c)
        };
        let (a, b) = (run(true), run(false));
        let lse = |l: &[f32]| {
            let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            mx + l.iter().map(|v| (v - mx).exp()).sum::<f32>().ln()
        };
        let (la, lb) = (lse(&a), lse(&b));
        let mut dl = 0f32;
        let mut dlp = 0f32;
        for (x, y) in a.iter().zip(&b) {
            dl = dl.max((x - y).abs());
            dlp = dlp.max(((x - la) - (y - lb)).abs());
        }
        let argmax = |l: &[f32]| (0..l.len()).max_by(|&i, &j| l[i].total_cmp(&l[j])).unwrap();
        let spread = a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - a.iter().copied().fold(f32::INFINITY, f32::min);
        println!(
            "repack vs per-row: max |dlogit| {dl:.2e} (logit range {spread:.1}), \
             max |dlogprob| {dlp:.2e}, argmax {} vs {}",
            argmax(&a),
            argmax(&b)
        );
        let hash = |l: &[f32]| {
            l.iter()
                .fold(0u64, |h, v| h.rotate_left(5) ^ v.to_bits() as u64)
        };
        println!(
            "logit hashes: r8 {:016x} per-row {:016x}; per-row argmax {} top {:.4}",
            hash(&a),
            hash(&b),
            argmax(&b),
            b[argmax(&b)]
        );
        assert!(dlp < 1.0, "r8 moved a logprob by {dlp}");
    }

    /// Every repacked matrix of the real model: r8/r8q rows vs the per-row
    /// kernel vs an f64 dot of the dequantized row and quantized input, on a
    /// real activation (the layer's own input for one token; wo gets the
    /// attention's input as a stand-in, the head the final norm).
    #[test]
    fn repack_rows_match_exact_on_model() {
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let net = Qwen2::new_with(&m, true);
        // The repack released the originals from `m`: raw bytes from a second load.
        let raw = crate::testutil::model().unwrap();
        let mut worst = (0f64, 0f64);
        let mut seen = std::collections::BTreeSet::new();
        let mut check = |w: &Weight, inp: &[f32]| {
            if w.r8.is_none() {
                return;
            }
            seen.insert(w.qt());
            let data = raw.tensor_data(w.t);
            let plain = Weight {
                t: w.t,
                data: Some(data),
                r8: None,
            };
            let a = net.matvec(w, inp);
            let b = net.matvec(&plain, inp);
            let n = inp.len();
            let rb = row_bytes(w.qt(), n);
            let xq = quantized_values(w.qt(), inp);
            let mut wr = vec![0f32; n];
            for r in 0..w.rows() {
                dequant(w.qt(), &data[r * rb..(r + 1) * rb], &mut wr);
                let exact: f64 = wr.iter().zip(&xq).map(|(p, q)| *p as f64 * q).sum();
                let mag: f64 = wr.iter().zip(&xq).map(|(p, q)| (*p as f64 * q).abs()).sum();
                let ea = (a[r] as f64 - exact).abs() / mag.max(f64::MIN_POSITIVE);
                let eb = (b[r] as f64 - exact).abs() / mag.max(f64::MIN_POSITIVE);
                worst.0 = worst.0.max(ea);
                worst.1 = worst.1.max(eb);
                assert!(
                    ea < 1e-5,
                    "{} row {r}: repacked {} per-row {} exact {exact}",
                    w.t.name,
                    a[r],
                    b[r]
                );
            }
        };
        let mut x = net.embed(9707);
        for layer in &net.layers {
            let ha = net.rmsnorm(&x, &layer.attn_norm);
            for w in [&layer.wq, &layer.wk, &layer.wv, &layer.wo] {
                check(w, &ha);
            }
            let h = net.rmsnorm(&x, &layer.ffn_norm);
            let act = net.matvec_swiglu(&layer.w_gate, &layer.w_up, &h);
            check(&layer.w_down, &act);
            check(&layer.w_gate, &h);
            check(&layer.w_up, &h);
            let d = net.matvec(&layer.w_down, &act);
            for i in 0..x.len() {
                x[i] += d[i];
            }
        }
        check(net.head(), &net.rmsnorm(&x, &net.output_norm));
        println!(
            "max rel err vs f64: repacked {:.2e}, per-row {:.2e}; types {seen:?}",
            worst.0, worst.1
        );
    }

    /// x as the kernels for `qt` see it: quantized (Q8_K 256-blocks for the
    /// K-quants, Q8_0 32-blocks for Q5_0/Q8_0) and dequantized, in f64.
    fn quantized_values(qt: u32, inp: &[f32]) -> Vec<f64> {
        let mut out = Vec::with_capacity(inp.len());
        if matches!(qt, Q5_0 | Q8_0) {
            for c in inp.chunks(32) {
                let amax = c.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = crate::quant::f16_to_f32(crate::quant::f32_to_f16(amax / 127.0));
                let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
                for v in c {
                    let q = ((v * id).round_ties_even() as i32).clamp(-128, 127);
                    out.push((q as f32 * d) as f64);
                }
            }
            return out;
        }
        for c in inp.chunks(256) {
            let amax = c.iter().fold(0f32, |m, v| m.max(v.abs()));
            let max = *c.iter().find(|v| v.abs() == amax).unwrap();
            let is = -127.0 / max;
            for v in c {
                let q = ((is * v).round_ties_even() as i32).min(127);
                out.push((q as f32 * (1.0 / is)) as f64);
            }
        }
        out
    }

    /// Releasing repacked originals must never hand out zeros, on the
    /// anonymous copy (where released pages ARE zeros) and on a file mapping.
    /// The output matrix is forced to the (tied) embedding, so it's repacked
    /// as the head (0.5B: Q5_0, r8q; 1.5B+: Q4_K, r8) while `embed` keeps
    /// reading its raw rows, and every tensor not released (blocks join
    /// across neighbouring released tensors) keeps its bytes.
    #[test]
    fn release_keeps_embedding_on_copy_and_mmap() {
        let Some(fresh) = crate::testutil::model_with(Some(false)) else {
            return;
        };
        let ids = [9707u32, 11, 1879, 0, 151642, 77, 5000];
        let mut logits: Vec<Vec<f32>> = Vec::new();
        for copy in [true, false] {
            let m = crate::testutil::model_with(Some(copy)).unwrap();
            assert_eq!(m.file_backed(), !copy);
            let net = Qwen2::build(&m, true, true, usize::MAX);
            let repacked: Vec<&Tensor> = net
                .layers
                .iter()
                .flat_map(|l| [&l.w_down, &l.w_gate, &l.wq])
                .filter(|w| w.r8.is_some())
                .map(|w| w.t)
                .collect();
            assert!(!repacked.is_empty(), "nothing repacked");
            // The embedding (= the tied head) still reads its real bytes,
            // and so does every other tensor that wasn't released, however
            // the released blocks around it were joined up.
            assert!(!m.is_released(net.token_embd));
            let mut kept = 0;
            for t in &m.tensors {
                if !m.is_released(t) {
                    assert_eq!(m.tensor_data(t), fresh.tensor_data(t), "{}", t.name);
                    kept += 1;
                }
            }
            assert!(kept > 0 && kept < m.tensors.len(), "kept {kept}");
            for &id in &ids {
                assert_eq!(net.embed(id), {
                    let mut x = vec![0f32; net.dim];
                    let rb = row_bytes(net.token_embd.qtype.id, net.dim);
                    let t = id as usize;
                    dequant(
                        net.token_embd.qtype.id,
                        &fresh.tensor_data(net.token_embd)[t * rb..(t + 1) * rb],
                        &mut x,
                    );
                    x
                });
            }
            // A released original: the copy refuses it, the file map faults
            // the same bytes back in.
            let t = repacked[0];
            if copy {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    m.tensor_data(t).len()
                }));
                assert!(r.is_err(), "released tensor readable on the anonymous copy");
            } else {
                assert_eq!(m.tensor_data(t), fresh.tensor_data(t));
            }
            let mut c = net.new_cache(ids.len());
            net.feed_batch(&ids[..ids.len() - 1], 0, &mut c);
            logits.push(net.forward(ids[ids.len() - 1], ids.len() - 1, &mut c));
            println!(
                "copy {copy}: head repacked {}, {} sampled repacked tensors",
                net.head().r8.is_some(),
                repacked.len()
            );
        }
        // Same weights either way: bit-identical logits.
        let eq = logits[0]
            .iter()
            .zip(&logits[1])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(eq, "copy and file map give different logits");
    }

    fn feed_batch_matches_feed_on(net: &Qwen2) {
        // 150 spread-out ordinary token ids (Qwen2's special tokens start at 151643).
        let ids: Vec<u32> = (0..150u64)
            .map(|i| (i * 2_654_435_761 % 150_000) as u32)
            .collect();
        let next = 9707;
        let mut a = net.new_cache(ids.len() + 1);
        let mut b = net.new_cache(ids.len() + 1);
        for (pos, &id) in ids.iter().enumerate() {
            net.feed(id, pos, &mut a);
        }
        let mut s = 0;
        // Every verify-sized batch (1..=9, so every padding remainder), an
        // odd one, a full prefill batch and the rest.
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 64, ids.len() - 122] {
            net.feed_batch(&ids[s..s + n], s, &mut b);
            s += n;
        }
        assert_eq!(s, ids.len());
        let la = net.forward(next, ids.len(), &mut a);
        let lb = net.forward(next, ids.len(), &mut b);

        let first_diff = |x: &[f32], y: &[f32]| {
            assert_eq!(x.len(), y.len());
            (0..x.len()).find(|&i| x[i].to_bits() != y[i].to_bits())
        };
        for l in 0..net.n_layers {
            for (what, x, y) in [
                ("k", kv_f32(&a, l, true), kv_f32(&b, l, true)),
                ("v", kv_f32(&a, l, false), kv_f32(&b, l, false)),
            ] {
                if let Some(i) = first_diff(&x[..], &y[..]) {
                    panic!(
                        "layer {l} {what} differs at pos {} elem {}: {} vs {}",
                        i / a.kv_dim,
                        i % a.kv_dim,
                        x[i],
                        y[i]
                    );
                }
            }
        }
        if let Some(i) = first_diff(&la[..], &lb[..]) {
            panic!("logit {i} differs: {} vs {}", la[i], lb[i]);
        }
    }

    /// Layer l's K (or V) cache, widened to f32 (exact).
    fn kv_f32(c: &KvCache, l: usize, k: bool) -> Vec<f32> {
        fn get<T: KvElem>(c: &Layers<T>, l: usize, k: bool) -> Vec<f32> {
            let x = if k { &c.k[l] } else { &c.v[l] };
            x.iter().map(|v| v.to_f32()).collect()
        }
        match &c.data {
            KvData::F16(c) => get(c, l, k),
            KvData::F32(c) => get(c, l, k),
        }
    }

    /// The grouped scorer must equal `dot` per query, bit for bit.
    #[test]
    fn dot_group_matches_dot() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
        };
        for n in [64, 37, 8, 5] {
            for g in [1, 2, 7, 9, 14] {
                for _ in 0..20 {
                    let q: Vec<f32> = (0..g * n).map(|_| rnd()).collect();
                    let k: Vec<f32> = (0..n).map(|_| rnd()).collect();
                    let kh: Vec<Half> = k.iter().map(|&x| Half::from_f32(x)).collect();
                    let kw: Vec<f32> = kh.iter().map(|x| x.to_f32()).collect();
                    let mut out = vec![0.0f32; g];
                    let mut outh = vec![0.0f32; g];
                    dot_group(&q, &k, &mut out);
                    dot_group(&q, &kh, &mut outh);
                    for (i, (&o, &oh)) in out.iter().zip(&outh).enumerate() {
                        let qi = &q[i * n..(i + 1) * n];
                        let want = dot(qi, &k);
                        assert_eq!(o.to_bits(), want.to_bits(), "n={n} g={g} head {i}");
                        // f16 keys: the same as f32 keys holding the widened values
                        assert_eq!(oh.to_bits(), dot(qi, &kw).to_bits(), "f16 n={n} g={g}");
                        assert_eq!(dot(qi, &kh).to_bits(), oh.to_bits(), "f16 n={n} g={g}");
                    }
                }
            }
        }
    }

    /// The shared-pass V sum must equal summing one head at a time (the
    /// per-head order: `o = fmadd(w, v, o)` over t from 0.0, as `weighted_sum`), bit
    /// for bit, for value rows strided inside a bigger buffer.
    #[test]
    fn vsum_matches_per_head() {
        fn weighted_sum<'v, const H: usize>(
            n: usize,
            w: impl Fn(usize) -> f32,
            v: impl Fn(usize) -> &'v [f32],
            o: &mut [f32; H],
        ) {
            let mut acc = [0.0f32; H];
            for t in 0..n {
                let w = w(t);
                let v: &[f32; H] = v(t).try_into().unwrap();
                for i in 0..H {
                    acc[i] = fmadd(w, v[i], acc[i]);
                }
            }
            *o = acc;
        }
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
        };
        for hd in [64, 37, 8, 5] {
            for nh in (1..=9).chain([14]) {
                for np in [1, 2, 7, 64, 300] {
                    let (stride, ws) = (hd + 3, np + 2);
                    let vb: Vec<f32> = (0..np * stride + 1)
                        .map(|_| Half::from_f32(rnd()).to_f32())
                        .collect();
                    let vh: Vec<Half> = vb.iter().map(|&x| Half::from_f32(x)).collect();
                    let w: Vec<f32> = (0..nh * ws).map(|_| rnd()).collect();
                    let v = |t: usize| &vb[t * stride + 1..t * stride + 1 + hd];
                    let mut got = vec![f32::NAN; nh * hd];
                    vsum_rows(&w, ws, np, nh, &vb[1..], stride, &mut got, hd);
                    let mut goth = vec![f32::NAN; nh * hd];
                    vsum_rows(&w, ws, np, nh, &vh[1..], stride, &mut goth, hd);
                    assert!(
                        got.iter()
                            .zip(&goth)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "f16 rows differ: hd={hd} nh={nh} np={np}"
                    );
                    for h in 0..nh {
                        let mut want = vec![0.0f32; hd];
                        for t in 0..np {
                            let wt = w[h * ws + t];
                            for (o, &x) in want.iter_mut().zip(v(t)) {
                                *o = fmadd(wt, x, *o);
                            }
                        }
                        if hd == 64 {
                            let mut ws64 = [0.0f32; 64];
                            weighted_sum::<64>(np, |t| w[h * ws + t], v, &mut ws64);
                            assert_eq!(
                                ws64.map(f32::to_bits)[..],
                                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>()[..]
                            );
                        }
                        for d in 0..hd {
                            assert_eq!(
                                got[h * hd + d].to_bits(),
                                want[d].to_bits(),
                                "hd={hd} nh={nh} np={np} head {h} d {d}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The CPU can run the `v3` variants (as `quant::has_avx2`).
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn can_v3() -> bool {
        let ok = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("bmi1")
            && is_x86_feature_detected!("bmi2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("lzcnt");
        if !ok {
            println!("::warning::no x86-64-v3 on this CPU: v3 variants not compared");
        }
        ok
    }

    /// Bit patterns, every NaN as one canonical NaN: Rust leaves NaN sign
    /// and payload unspecified (x86 keeps the first operand's NaN, so it
    /// varies with register allocation), so only "is NaN" is comparable.
    /// Everything else, -0.0 and infinities included, must match exactly.
    pub(crate) fn bits(x: &[f32]) -> Vec<u32> {
        x.iter()
            .map(|v| if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() })
            .collect()
    }

    /// Random values with the edge cases mixed in: +-0, +-inf, NaN, tiny and
    /// huge magnitudes (`edgy` of them, the rest uniform in [-scale, scale)).
    pub(crate) fn mixed(n: usize, seed: u64, scale: f32, edgy: bool) -> Vec<f32> {
        let mut st = seed | 1;
        let edges = [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-38,
            -3e38,
            1e-45,
        ];
        (0..n)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                if edgy && st.is_multiple_of(17) {
                    edges[(st >> 8) as usize % edges.len()]
                } else {
                    ((st >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0) * scale
                }
            })
            .collect()
    }

    /// Every multiversioned function in this file: the portable and the
    /// x86-64-v3 variant on the same inputs give the same bits (NaNs as in
    /// `bits`). Sizes cover whole 8-lane chunks, tails and empty inputs.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn multiversion_bit_identical() {
        if !can_v3() {
            return;
        }
        let mut seed = 1u64;
        let mut next = || {
            seed += 0x9e37_79b9;
            seed
        };
        for n in [0usize, 1, 5, 8, 37, 64, 128, 896] {
            for edgy in [false, true] {
                let x = mixed(n, next(), 3.0, edgy);
                let w = mixed(n, next(), 1.5, edgy);
                for eps in [1e-6f32, 0.0] {
                    let (mut a, mut b) = (vec![f32::NAN; n], vec![1.0; n]);
                    rmsnorm_into::portable(&x, &w, eps, &mut a);
                    unsafe { rmsnorm_into::v3(&x, &w, eps, &mut b) };
                    assert_eq!(bits(&a), bits(&b), "rmsnorm n={n}");
                }
                let (mut a, mut b) = (x.clone(), x.clone());
                add_assign::portable(&mut a, &w);
                unsafe { add_assign::v3(&mut b, &w) };
                assert_eq!(bits(&a), bits(&b), "add_assign n={n}");

                let mut best = |slots: usize| {
                    let ys = mixed(n * slots, next(), 2.0, edgy);
                    let mut a = vec![Best::NONE; slots];
                    let mut b = vec![Best::NONE; slots];
                    best_rows::portable(&ys, slots, 11, &mut a);
                    unsafe { best_rows::v3(&ys, slots, 11, &mut b) };
                    for (a, b) in a.iter().zip(&b) {
                        assert_eq!((bits(&[a.v]), a.i), (bits(&[b.v]), b.i), "best n={n}");
                    }
                };
                best(1);
                best(4);
                best(12);
            }
        }
        for hd in [64, 128, 6] {
            let half = hd / 2;
            let table: Vec<(f32, f32)> = mixed(half, next(), 1.0, false)
                .iter()
                .map(|&a| (a.sin(), a.cos()))
                .collect();
            let v = mixed(7 * hd, next(), 2.0, true);
            let (mut a, mut b) = (v.clone(), v.clone());
            rope_heads::portable(&mut a, hd, &table);
            unsafe { rope_heads::v3(&mut b, hd, &table) };
            assert_eq!(bits(&a), bits(&b), "rope hd={hd}");
        }
        for (rows, slots) in [(896, 4), (37, 12), (8, 1), (0, 4)] {
            let y = mixed(rows * slots, next(), 2.0, true);
            for b in 0..slots {
                let ga = gather_col::portable(&y, slots, b, rows);
                let gb = unsafe { gather_col::v3(&y, slots, b, rows) };
                assert_eq!(bits(&ga), bits(&gb), "gather_col");
                let x = mixed(rows, next(), 2.0, true);
                let (mut a, mut c) = (x.clone(), x.clone());
                add_col::portable(&mut a, &y, slots, b);
                unsafe { add_col::v3(&mut c, &y, slots, b) };
                assert_eq!(bits(&a), bits(&c), "add_col");
            }
        }
        // softmax over plain scores (no inf/NaN unless edgy)
        for n in [0usize, 1, 5, 8, 37, 64, 300] {
            for edgy in [false, true] {
                for stride in [1, 3] {
                    let src = mixed(n * stride, next(), 8.0, edgy);
                    let (mut a, mut b) = (vec![0.0; n], vec![1.0; n]);
                    softmax_from::portable(&src, stride, &mut a);
                    unsafe { softmax_from::v3(&src, stride, &mut b) };
                    assert_eq!(bits(&a), bits(&b), "softmax_from n={n}");
                    unsafe { softmax_from::fma(&src, stride, &mut b) };
                    assert_eq!(bits(&a), bits(&b), "softmax_from fma n={n}");
                }
            }
        }
        // Attention pieces, on the model's shapes and odd ones, f32 and f16
        // caches. The portable variant fuses like v3 here (the CPU has FMA).
        assert!(vmath::fused());
        for (hd, group, n_kv) in [(64, 7, 2), (64, 6, 2), (128, 3, 1), (37, 9, 2), (5, 1, 1)] {
            let kv_dim = n_kv * hd;
            for np in [1usize, 2, 7, 33, 300] {
                let qg = mixed(group * hd, next(), 4.0, false);
                let kc = mixed((np + 2) * kv_dim, next(), 1.0, false);
                let vc = mixed((np + 2) * kv_dim, next(), 1.0, false);
                let kh: Vec<Half> = kc.iter().map(|&x| Half::from_f32(x)).collect();
                let vh: Vec<Half> = vc.iter().map(|&x| Half::from_f32(x)).collect();
                let scale = 1.0 / (hd as f32).sqrt();
                let kvh = n_kv - 1;
                let off = kvh * hd;
                let tc = np.div_ceil(3);
                let t0 = np - np.min(tc);
                let wts = mixed(group * (np + 1), next(), 1.0, false);
                let vb = &vc[off..];
                let vbh = &vh[off..];

                let (mut a, mut b) = (vec![0.0; group], vec![1.0; group]);
                // SAFETY (every v3 call below): can_v3() above.
                unsafe { super::dot_group::<false, true, f32>(&qg, &kc[..hd], &mut a) };
                unsafe { super::dot_group::<true, true, f32>(&qg, &kc[..hd], &mut b) };
                assert_eq!(bits(&a), bits(&b), "dot_group hd={hd} g={group}");

                let (mut a, mut b) = (vec![0.0; np], vec![1.0; np]);
                score_softmax::portable(&qg[..hd], &kc, kv_dim, off, scale, &mut a);
                unsafe { score_softmax::v3(&qg[..hd], &kc, kv_dim, off, scale, &mut b) };
                assert_eq!(bits(&a), bits(&b), "score_softmax hd={hd} np={np}");
                score_softmax::portable(&qg[..hd], &kh, kv_dim, off, scale, &mut a);
                unsafe { score_softmax::v3(&qg[..hd], &kh, kv_dim, off, scale, &mut b) };
                assert_eq!(bits(&a), bits(&b), "score_softmax f16 hd={hd} np={np}");
                unsafe { score_softmax::fma(&qg[..hd], &kh, kv_dim, off, scale, &mut b) };
                assert_eq!(bits(&a), bits(&b), "score_softmax fma hd={hd} np={np}");

                let (mut a, mut b) = (vec![0.0; tc * group], vec![1.0; tc * group]);
                score_tile::portable(&qg, &kh, kv_dim, hd, off, t0..np, scale, &mut a);
                unsafe { score_tile::v3(&qg, &kh, kv_dim, hd, off, t0..np, scale, &mut b) };
                assert_eq!(bits(&a), bits(&b), "score_tile hd={hd} np={np}");
                unsafe { score_tile::fma(&qg, &kh, kv_dim, hd, off, t0..np, scale, &mut b) };
                assert_eq!(bits(&a), bits(&b), "score_tile fma hd={hd} np={np}");

                let (mut a, mut b) = (vec![0.0; group * hd], vec![1.0; group * hd]);
                vsum_rows::portable(&wts, np + 1, np, group, vb, kv_dim, &mut a, hd);
                unsafe { vsum_rows::v3(&wts, np + 1, np, group, vb, kv_dim, &mut b, hd) };
                assert_eq!(bits(&a), bits(&b), "vsum_rows hd={hd} np={np}");
                vsum_rows::portable(&wts, np + 1, np, group, vbh, kv_dim, &mut a, hd);
                unsafe { vsum_rows::v3(&wts, np + 1, np, group, vbh, kv_dim, &mut b, hd) };
                assert_eq!(bits(&a), bits(&b), "vsum_rows f16 hd={hd} np={np}");
                unsafe { vsum_rows::fma(&wts, np + 1, np, group, vbh, kv_dim, &mut b, hd) };
                assert_eq!(bits(&a), bits(&b), "vsum_rows fma hd={hd} np={np}");

                let (mut sa, mut sb) = (vec![0.0; (np + 1) * group], vec![1.0; (np + 1) * group]);
                let (mut a, mut b) = (vec![0.0; group * hd], vec![1.0; group * hd]);
                attend_causal::portable(&qg, &kh, &vh, kv_dim, hd, kvh, np, scale, &mut sa, &mut a);
                unsafe {
                    attend_causal::v3(&qg, &kh, &vh, kv_dim, hd, kvh, np, scale, &mut sb, &mut b)
                };
                assert_eq!(bits(&a), bits(&b), "attend_causal hd={hd} np={np}");
                unsafe {
                    attend_causal::fma(&qg, &kh, &vh, kv_dim, hd, kvh, np, scale, &mut sb, &mut b)
                };
                assert_eq!(bits(&a), bits(&b), "attend_causal fma hd={hd} np={np}");
            }
        }
    }

    /// The unfused path (CPUs without FMA) on its own terms: `dot` is 8
    /// lanes of separate multiply and add + `sum8` + the tail, the V sum the
    /// same chain per element, softmax `exp_f::<false>`; equal bit for bit to
    /// plain references, and decode equals prefill attention on it too.
    #[test]
    fn unfused_attention_paths() {
        let mut seed = 0x5151_2626u64;
        let mut next = || {
            seed += 0x9e37_79b9;
            seed
        };
        for (hd, group) in [(64, 7), (37, 3), (5, 1)] {
            for np in [1usize, 9, 40] {
                let q = mixed(group * hd, next(), 4.0, false);
                let kc = mixed(np * hd, next(), 1.0, false);
                let vc = mixed(np * hd, next(), 1.0, false);
                let scale = 1.0 / (hd as f32).sqrt();
                let dot_ref = |a: &[f32], b: &[f32]| {
                    let mut acc = [0.0f32; 8];
                    let n8 = a.len() / 8 * 8;
                    for i in 0..n8 {
                        acc[i % 8] += a[i] * b[i];
                    }
                    let mut t = 0.0f32;
                    for i in n8..a.len() {
                        t += a[i] * b[i];
                    }
                    vmath::sum8(&acc) + t
                };
                let mut got = vec![0.0f32; group];
                // SAFETY: V3 false.
                unsafe { super::dot_group::<false, false, f32>(&q, &kc[..hd], &mut got) };
                for g in 0..group {
                    let want = dot_ref(&q[g * hd..(g + 1) * hd], &kc[..hd]);
                    assert_eq!(got[g].to_bits(), want.to_bits(), "dot hd={hd} g={g}");
                }
                // Prefill body vs the per-head reference.
                let mut sc = vec![f32::NAN; (np + 1) * group];
                let mut out = vec![f32::NAN; group * hd];
                // SAFETY: V3 false.
                unsafe {
                    attend_causal::imp::<false, false, f32>(
                        &q, &kc, &vc, hd, hd, 0, np, scale, &mut sc, &mut out,
                    )
                };
                for g in 0..group {
                    let qv = &q[g * hd..(g + 1) * hd];
                    let s: Vec<f32> = (0..np)
                        .map(|t| dot_ref(qv, &kc[t * hd..(t + 1) * hd]) * scale)
                        .collect();
                    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f32> = s.iter().map(|&x| vmath::exp_f::<false>(x - m)).collect();
                    let (c, r) = e.as_chunks::<8>();
                    let mut lanes = [0.0f32; 8];
                    for c in c {
                        for i in 0..8 {
                            lanes[i] += c[i];
                        }
                    }
                    let inv = 1.0 / (vmath::sum8(&lanes) + r.iter().fold(0.0f32, |a, &x| a + x));
                    for d in 0..hd {
                        let mut o = 0.0f32;
                        for t in 0..np {
                            o += e[t] * inv * vc[t * hd + d];
                        }
                        assert_eq!(
                            out[g * hd + d].to_bits(),
                            o.to_bits(),
                            "attend hd={hd} np={np}"
                        );
                    }
                    // Decode scoring of the same head: same weights.
                    let mut w = vec![f32::NAN; np];
                    // SAFETY: V3 false.
                    unsafe {
                        score_softmax::imp::<false, false, f32>(qv, &kc, hd, 0, scale, &mut w)
                    };
                    for t in 0..np {
                        assert_eq!(w[t].to_bits(), (e[t] * inv).to_bits(), "weights t={t}");
                    }
                }
            }
        }
    }

    /// Decode attention, per-head and every tiling and V split, and
    /// prefill's `attend_group` must equal a plain scalar
    /// one-head-at-a-time reference bit for bit: `dot` * scale, softmax (max,
    /// `vmath::exp`, sum in 8 lanes by t mod 8 + `sum8` + the tail in order,
    /// times 1/sum), then o = fmadd(w, v, o) over t from 0.0, on f32 and f16
    /// caches (v widened). The cache has extra rows past np that must not be
    /// read, and the scratch is reused across calls with stale contents.
    #[test]
    fn attend_matches_per_head() {
        let mut seed = 0x6a09_e667_f3bc_c909u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
        };
        let mut buf = Vec::new();
        for hd in [64, 128] {
            for group in [6, 7] {
                for n_kv in [2, 4] {
                    let kv_dim = n_kv * hd;
                    let nps = (1..300).chain([1000, 2047, 3001]);
                    for np in nps {
                        let q: Vec<f32> = (0..n_kv * group * hd).map(|_| rnd() * 4.0).collect();
                        let kc: Vec<f32> = (0..(np + 3) * kv_dim).map(|_| rnd()).collect();
                        let vc: Vec<f32> = (0..(np + 3) * kv_dim).map(|_| rnd()).collect();
                        attend_check(&q, &kc, &vc, kv_dim, hd, np, &mut buf);
                        let half = |x: &Vec<f32>| -> Vec<Half> {
                            let mut h = vec![Half(0); x.len()];
                            Half::store(&mut h, x);
                            h
                        };
                        attend_check(&q, &half(&kc), &half(&vc), kv_dim, hd, np, &mut buf);
                    }
                }
            }
        }
    }

    fn attend_check<T: KvElem>(
        q: &[f32],
        kc: &[T],
        vc: &[T],
        kv_dim: usize,
        hd: usize,
        np: usize,
        buf: &mut Vec<f32>,
    ) {
        let nq = q.len();
        let n_kv = kv_dim / hd;
        let group = nq / hd / n_kv;
        let ty = std::any::type_name::<T>();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut want = vec![0.0f32; nq];
        for qh in 0..n_kv * group {
            let kvh = qh / group;
            let qv = &q[qh * hd..(qh + 1) * hd];
            let mut w: Vec<f32> = (0..np)
                .map(|t| {
                    let k0 = t * kv_dim + kvh * hd;
                    dot(qv, &kc[k0..k0 + hd]) * scale
                })
                .collect();
            let max = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let (mut lanes, mut tail) = ([0.0f32; 8], 0.0f32);
            let whole = np / 8 * 8;
            for (t, e) in w.iter_mut().enumerate() {
                *e = vmath::exp(*e - max);
                if t < whole {
                    lanes[t % 8] += *e;
                } else {
                    tail += *e;
                }
            }
            let inv = 1.0 / (vmath::sum8(&lanes) + tail);
            w.iter_mut().for_each(|e| *e *= inv);
            let o = &mut want[qh * hd..(qh + 1) * hd];
            for (t, &wt) in w.iter().enumerate() {
                let v0 = t * kv_dim + kvh * hd;
                for (o, &v) in o.iter_mut().zip(&vc[v0..v0 + hd]) {
                    *o = fmadd(wt, v.to_f32(), *o);
                }
            }
        }
        let tiles = [
            None,
            Some(1),
            Some(7),
            Some(64),
            Some(np.div_ceil(5)),
            decode_tile(np.max(TILE_MIN_NP), n_kv),
        ];
        // Prefill's per-(token, kv head) attention.
        let mut got = vec![f32::NAN; nq];
        for (kvh, o) in got.chunks_exact_mut(group * hd).enumerate() {
            let qg = &q[kvh * group * hd..(kvh + 1) * group * hd];
            attend_group(qg, kc, vc, kv_dim, hd, kvh, np, o);
        }
        if let Some(i) = (0..nq).find(|&i| got[i].to_bits() != want[i].to_bits()) {
            panic!(
                "{ty} prefill hd={hd} group={group} n_kv={n_kv} np={np}: elem {i} {} vs {}",
                got[i], want[i]
            );
        }
        // Decode: every tiling x the V split for 1, 5, 18 and 64 threads.
        for tile in tiles {
            for threads in [1, 5, 18, 64] {
                let mut got = vec![f32::NAN; nq];
                attend(q, kc, vc, kv_dim, hd, np, tile, threads, buf, &mut got);
                if let Some(i) = (0..nq).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                    panic!(
                        "{ty} hd={hd} group={group} n_kv={n_kv} np={np} tile={tile:?} \
                         threads={threads}: elem {i} {} vs {}",
                        got[i], want[i]
                    );
                }
            }
        }
    }

    /// Run each prompt in tests/fixtures/gen/ and follow Ollama's greedy path token by
    /// token. At every step axon must give Ollama's token nearly the same logprob, and
    /// must pick the same token unless Ollama's top two were a near-tie.
    #[test]
    fn matches_ollama() {
        const MAX_DLOGPROB: f32 = 0.15;
        let dir = crate::testutil::fixtures().join("gen");
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|f| f.ends_with(".json"))
            .collect();
        names.sort();
        assert!(!names.is_empty(), "no gen fixtures in {}", dir.display());
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
        let net = Qwen2::new(&m);

        for name in &names {
            let fx = crate::testutil::read_json(&format!("gen/{name}"));
            let prompt = fx["prompt"].as_str().expect("prompt");
            let steps = fx["steps"].as_array().expect("steps");
            let ids = tok.encode(prompt);
            let mut cache = net.new_cache(ids.len() + steps.len() + 1);
            let (&last, rest) = ids.split_last().expect("prompt has tokens");
            for (pos, &id) in rest.iter().enumerate() {
                net.feed(id, pos, &mut cache);
            }
            let mut logits = net.forward(last, rest.len(), &mut cache);
            let mut worst = 0f32;
            for (i, s) in steps.iter().enumerate() {
                let want = s["id"].as_u64().expect("id") as usize;
                let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = mx + logits.iter().map(|l| (l - mx).exp()).sum::<f32>().ln();
                let lp = logits[want] - lse;
                let d = (lp - s["logprob"].as_f64().expect("logprob") as f32).abs();
                worst = worst.max(d);
                let ctx = format!("{name} step {i} ({:?})", s["token"].as_str().unwrap_or("?"));
                assert!(d <= MAX_DLOGPROB, "{ctx}: logprob off by {d:.3}");
                if !s["near_tie"].as_bool().unwrap_or(false) {
                    let argmax = (0..logits.len())
                        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                        .unwrap();
                    assert_eq!(
                        argmax,
                        want,
                        "{ctx}: axon picked {:?}",
                        tok.decode(&[argmax as u32])
                    );
                }
                logits = net.forward(want as u32, ids.len() + i, &mut cache);
            }
            println!("{name}: {} steps, max |dlogprob| {worst:.3}", steps.len());
        }
    }
}
