//! qwen2 forward pass. Matrix-vector products quantize the input to 8 bits and
//! run integer dot products on the worker pool (pool.rs); everything else is
//! plain f32 loops.

use crate::gguf::{Model, Tensor, Val};
use crate::kv::{Half, KvData, KvElem, Layers};
use crate::pool::{self, Kind, Pool};
use crate::quant::{
    DecodeCfg, F16, F32, IQ4_NL, MXFP4, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0,
    QuantizedAct, batch_slots, dequant, dot_row_q8_multi, dot_rows_q8, dot_rows_q8_multi, r8, r8q,
    row_bytes,
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
        let repacked = dst.is_some();
        let w = Self::load_bytes(t, m.tensor_data(t), dst);
        if repacked && release {
            m.release(t);
        }
        w
    }

    /// `load` of the matrix whose bytes are `data` (all of `t`, or one
    /// expert's slice of a 3-D `t`: a dims[1] x dims[0] matrix either way),
    /// releasing nothing.
    fn load_bytes(t: &'a Tensor, data: &'a [u8], dst: Option<&mut [u8]>) -> Weight<'a> {
        let Some(buf) = dst else {
            return Weight {
                t,
                data: Some(data),
                r8: None,
            };
        };
        let (qt, rows, n) = (t.qtype.id, t.dims[1] as usize, t.dims[0] as usize);
        if r8q::can_repack(qt, rows, n) {
            r8q::repack_into(qt, data, rows, n, buf);
        } else {
            r8::repack_into(qt, data, rows, n, buf);
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
    /// Per-head RMSNorm weights (head_dim) for q and k, before RoPE (qwen3).
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    wo: Weight<'a>,
    ffn_norm: Vec<f32>,
    ffn: Ffn<'a>,
    /// gemma3: RMSNorm weights of the attention output and of the FFN
    /// output, applied before each residual add.
    post_attn_norm: Option<Vec<f32>>,
    post_ffn_norm: Option<Vec<f32>>,
    /// gemma3 local layer: attends to the last `window` positions only
    /// (`window_start`) and uses the local RoPE table.
    window: Option<usize>,
    /// gpt-oss: the attention output's bias, and one sink score per query
    /// head (`vmath::softmax_sink`).
    bo: Option<Vec<f32>>,
    sinks: Option<Vec<f32>>,
}

/// A layer's feed-forward block: the gated FFN, or experts and a router.
enum Ffn<'a> {
    Dense {
        gate: Weight<'a>,
        up: Weight<'a>,
        down: Weight<'a>,
    },
    Moe(Box<moe::Moe<'a>>),
}

impl<'a> Layer<'a> {
    /// The gated FFN's gate, up and down (a dense layer: every caller but
    /// the MoE paths, which no other backend takes).
    fn dense(&self) -> (&Weight<'a>, &Weight<'a>, &Weight<'a>) {
        match &self.ffn {
            Ffn::Dense { gate, up, down } => (gate, up, down),
            Ffn::Moe(_) => panic!("a mixture-of-experts layer has no dense FFN"),
        }
    }

    /// Every matrix, in the r8 arena's order (`Qwen2::build`).
    fn weights(&self) -> Vec<&Weight<'a>> {
        let mut v = vec![&self.wq, &self.wk, &self.wv, &self.wo];
        match &self.ffn {
            Ffn::Dense { gate, up, down } => v.extend([gate, up, down]),
            Ffn::Moe(m) => v.extend(m.weights()),
        }
        v
    }

    fn weights_mut(&mut self) -> Vec<&mut Weight<'a>> {
        let mut v = vec![&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo];
        match &mut self.ffn {
            Ffn::Dense { gate, up, down } => v.extend([gate, up, down]),
            Ffn::Moe(m) => v.extend(m.weights_mut()),
        }
        v
    }
}

pub struct Qwen2<'a> {
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// n_heads * head_dim: the attention width (qwen3 can differ from dim).
    pub q_dim: usize,
    pub vocab: usize,
    eps: f32,
    /// RoPE frequency per pair (`rope_freqs`), head_dim / 2 of them; on
    /// gemma3 with local layers, then theirs (`rope_freqs_swa`).
    /// (A boxed slice, not a Vec: `lm::Backend` holds this struct inline,
    /// near clippy's size bound.)
    freqs: Box<[f64]>,
    /// YaRN's (sin, cos) multiplier on the global table (`rope::mscale`;
    /// 1 without YaRN).
    rope_m: f32,
    /// gemma3: GELU (tanh form) in `glu`, the embedding times sqrt(dim),
    /// and `freqs` followed by the local layers' table. (Flags, not data:
    /// `lm::Backend` holds this struct inline, near clippy's size bound.)
    gemma: bool,
    token_embd: &'a Tensor,
    /// token_embd's bytes (never released: see `Weight::load`)
    embd: &'a [u8],
    /// Empty, and `output` None, in a model of the first layers only
    /// (`new_layers`): the output head is another backend's.
    /// (A boxed slice, as `freqs`.)
    output_norm: Box<[f32]>,
    output: Option<Weight<'a>>,
    /// Layers 0..layers.len() (all n_layers unless `new_layers`); empty
    /// on deepseek2, whose blocks are `ds2`'s.
    layers: Vec<Layer<'a>>,
    ds2: Option<Box<deepseek2::Ds2<'a>>>,
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

impl KvCache {
    /// (element type name, bytes per element): what an image of the cache
    /// stores (kvsave).
    pub(crate) fn elem(&self) -> (&'static str, usize) {
        match &self.data {
            KvData::F16(_) => ("f16", 2),
            KvData::F32(_) => ("f32", 4),
        }
    }

    pub(crate) fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// `arrays_mut`, read-only.
    pub(crate) fn arrays(&self) -> Vec<&[u8]> {
        fn bytes<T: KvElem>(c: &Layers<T>) -> Vec<&[u8]> {
            c.k.iter()
                .zip(c.v.iter())
                .flat_map(|(k, v)| [&k[..], &v[..]])
                .map(|a| {
                    // SAFETY: plain data (KvElem) viewed as its bytes.
                    unsafe { std::slice::from_raw_parts(a.as_ptr().cast::<u8>(), size_of_val(a)) }
                })
                .collect()
        }
        match &self.data {
            KvData::F16(c) => bytes(c),
            KvData::F32(c) => bytes(c),
        }
    }

    /// Every layer's K then V, each `[max_ctx][kv_dim]`, as bytes (kvsave:
    /// a cache's rows to and from a file).
    pub(crate) fn arrays_mut(&mut self) -> Vec<&mut [u8]> {
        fn bytes<T: KvElem>(c: &mut Layers<T>) -> Vec<&mut [u8]> {
            c.k.iter_mut()
                .zip(c.v.iter_mut())
                .flat_map(|(k, v)| [&mut k[..], &mut v[..]])
                .map(|a| {
                    let n = size_of_val(a);
                    // SAFETY: plain data (KvElem) viewed as its bytes, the
                    // same borrow; any bytes written back are valid Ts.
                    unsafe { std::slice::from_raw_parts_mut(a.as_mut_ptr().cast::<u8>(), n) }
                })
                .collect()
        }
        match &mut self.data {
            KvData::F16(c) => bytes(c),
            KvData::F32(c) => bytes(c),
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
        .and_then(|(_, v)| v.as_f64())
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

/// Architectures `Qwen2` runs: qwen3 is qwen2 without the qkv biases, plus
/// a per-head RMSNorm on q and k before RoPE and a head size of its own.
/// llama (Llama 2/3.x) is qwen2 without the biases, with its attn_q/attn_k
/// rows put back in rotate-half order at load (`gguf::rope_unpermute_plan`)
/// and Llama 3's RoPE frequency factors (`rope_freqs`).
/// gemma3 (Gemma 3 1B/4B/12B, llama.cpp src/models/gemma3.cpp) is qwen3
/// plus: the embedding times sqrt(dim), an RMSNorm on the attention and on
/// the FFN output before each residual add (`post_attention_norm`,
/// `post_ffw_norm`), GELU (tanh form) instead of SiLU in the gated FFN,
/// and sliding-window layers (`Gemma`). Its norm weights are stored with
/// HF's `1 + w` already added by llama.cpp's converter, so they are plain
/// RMSNorm weights here.
/// qwen3moe (Qwen3-30B-A3B & co.) is qwen3 with each FFN replaced by
/// experts and a router (`moe`; on the GPU, `crate::moe`).
/// gpt-oss (gpt-oss-20b/120b, llama.cpp src/models/openai-moe.cpp) is
/// qwen2 (qkv biases) plus a bias on the attention output, per-head
/// attention sinks, sliding-window (128) layers alternating with global
/// ones (even layers local), YaRN on every layer, and MoE FFNs (`moe`) with
/// a biased router, top-4 then softmax over the four, biased experts and
/// the clamped SwiGLU (`vmath::swiglu_oai`); the pre-FFN norm is its
/// `post_attention_norm`. Experts are MXFP4 (quant/mxfp4.rs), on the CPU;
/// with --device cuda the rest runs on the GPU (`crate::moe`, cuda/oss.cu).
/// deepseek2 (DeepSeek V2/V3, Kimi K2, Moonlight: MLA attention, shared
/// experts) is its own block (`deepseek2`), on the CPU only.
pub const ARCHS: [&str; 7] = [
    "qwen2",
    "qwen3",
    "llama",
    "gemma3",
    "qwen3moe",
    "gpt-oss",
    "deepseek2",
];

#[path = "model/deepseek2.rs"]
mod deepseek2;
#[path = "model/moe.rs"]
mod moe;
pub(crate) use deepseek2::info_lines as deepseek2_info;
pub(crate) use deepseek2::{GpuCfg as Ds2GpuCfg, gpu_cfg as deepseek2_gpu_cfg};

pub(crate) use moe::{
    MoeDims, MoeInfo, Route as MoeRoute, check_ram as moe_check_ram,
    gpu_refusal as moe_gpu_refusal, is_moe, moe_dims,
};

/// Gemma 3's layer layout: local (sliding-window) layers attend to the last
/// `window` positions only (key t is seen from position p iff p - t <
/// window, llama.cpp's LLAMA_SWA_TYPE_STANDARD) and turn by their own RoPE
/// table (base `rope.freq_base_swa`, default 10000, never scaled); the
/// global layers see everything and use `rope.freq_base` with the model's
/// linear RoPE scaling (`rope_freqs`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Swa {
    pub window: usize,
    /// Per layer: local (true) or global.
    pub local: Vec<bool>,
}

/// The sliding-window layout of a gemma3 model (None: every layer global,
/// llama.cpp's swa_type NONE when the file has no window), as llama.cpp's
/// `load_swa_pattern(ml, 6)`: an array `attention.sliding_window_pattern`
/// gives each layer; a number n (default 6) makes layer il local iff
/// il % n < n - 1 (every n-th layer global).
///
/// gpt-oss the same with llama.cpp's `load_swa_pattern(ml, 2)`: by default
/// the even layers are local.
pub(crate) fn gemma_swa(m: &Model, n_layers: usize) -> Option<Swa> {
    let pattern = match m.arch.as_str() {
        "gemma3" => 6,
        "gpt-oss" => 2,
        _ => return None,
    };
    let a = &m.arch;
    let window =
        meta_num(m, &format!("{a}.attention.sliding_window")).filter(|&w| w > 0.0)? as usize;
    let key = format!("{a}.attention.sliding_window_pattern");
    let local = match m.metadata.iter().find(|(k, _)| *k == key).map(|(_, v)| v) {
        Some(Val::Arr(a)) => (0..n_layers)
            .map(|i| a.get(i).and_then(|v| v.as_f64()).is_some_and(|v| v != 0.0))
            .collect(),
        Some(v) => {
            let n = v.as_f64().unwrap_or(pattern as f64) as usize;
            (0..n_layers).map(|i| n == 0 || i % n < n - 1).collect()
        }
        None => (0..n_layers).map(|i| i % pattern < pattern - 1).collect(),
    };
    Some(Swa { window, local })
}

/// First cache position a layer attends to from position `pos`: 0, or for
/// a local layer of window w, pos + 1 - w once that is positive.
#[inline]
pub(crate) fn window_start(window: Option<usize>, pos: usize) -> usize {
    window.map_or(0, |w| (pos + 1).saturating_sub(w))
}

/// Gemma 3's local-layer RoPE frequencies: base `rope.freq_base_swa`
/// (llama.cpp's default 10000), no scaling.
pub(crate) fn rope_freqs_swa(m: &Model, head_dim: usize) -> Vec<f64> {
    let base = meta_num(m, &format!("{}.rope.freq_base_swa", m.arch)).unwrap_or(10000.0);
    (0..head_dim / 2)
        .map(|i| base.powf(-2.0 * i as f64 / head_dim as f64))
        .collect()
}

/// gemma3's linear RoPE scale for the global layers: 1 / rope.scaling.factor
/// (or rope.scale_linear) unless the scaling type is "none" (llama.cpp:
/// rope_freq_scale_train, and freq scale 1 for type none). 1 elsewhere
/// (`check_supported` refuses scaling on the other architectures).
pub(crate) fn rope_linear_scale(m: &Model) -> f64 {
    if m.arch != "gemma3" {
        return 1.0;
    }
    let num = |k: &str| meta_num(m, &format!("{}.{k}", m.arch));
    let none = m.metadata.iter().any(|(k, v)| {
        *k == format!("{}.rope.scaling.type", m.arch) && matches!(v, Val::Str(t) if t == "none")
    });
    match num("rope.scaling.factor").or(num("rope.scale_linear")) {
        Some(f) if f != 0.0 && !none => 1.0 / f,
        _ => 1.0,
    }
}

/// RoPE frequency of each pair i < head_dim / 2: base^(-2i/head_dim),
/// divided by the model's factor for that pair if it has `rope_freqs.weight`
/// (Llama 3.1/3.2's long-context scaling, as llama.cpp's freq_factors).
/// Shared by the CPU and CUDA backends. `check_supported` validates the
/// factors.
pub(crate) fn rope_freqs(m: &Model, head_dim: usize) -> Vec<f64> {
    let base = meta_num(m, &format!("{}.rope.freq_base", m.arch)).unwrap_or(10000.0);
    let factors = find_opt(m, "rope_freqs.weight").map(|t| load_vec(m, t));
    let scale = rope_linear_scale(m);
    let mut freqs: Vec<f64> = (0..head_dim / 2)
        .map(|i| {
            let f = base.powf(-2.0 * i as f64 / head_dim as f64);
            let f = match &factors {
                Some(v) => f / v[i] as f64,
                None => f,
            };
            if scale == 1.0 { f } else { f * scale }
        })
        .collect();
    // YaRN (rope.rs), after Llama 3's factors as llama.cpp's theta / ff.
    if let Some(y) = crate::rope::yarn(m) {
        y.apply(&mut freqs, base, head_dim);
    }
    freqs
}

/// Tensor types the kernels run (`quant::dequant` and `dot_row_q8`).
const SUPPORTED_TYPES: [u32; 13] = [
    F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL,
];

/// gpt-oss's expert matrices only (`check_supported`).
const EXPERT_TYPES: [u32; 1] = [MXFP4];

/// The refusal for a tensor of a type the kernels don't run: the type by
/// name, what is supported, and which file to get instead.
/// A mixed file (an "IQ2_M" is mostly IQ2_S/IQ3_S, with IQ4_NL where a
/// row isn't whole 256-blocks) also gets every unsupported type it holds.
fn unsupported_type(m: &Model, name: &str, q: crate::gguf::QType) -> String {
    let mut seen: Vec<(&str, usize)> = Vec::new();
    for t in m.tensors.iter().filter(|t| {
        !(SUPPORTED_TYPES.contains(&t.qtype.id)
            || m.arch == "gpt-oss" && EXPERT_TYPES.contains(&t.qtype.id))
    }) {
        match seen.iter_mut().find(|s| s.0 == t.qtype.name) {
            Some(s) => s.1 += 1,
            None => seen.push((t.qtype.name, 1)),
        }
    }
    let all = if seen.len() > 1 {
        let l: Vec<String> = seen.iter().map(|(n, c)| format!("{c} {n}")).collect();
        format!(" (the file has unsupported tensors: {})", l.join(", "))
    } else {
        String::new()
    };
    let n = q.name;
    let hint = if n.starts_with("Q4_0_") {
        "; it is an ARM repacked layout that llama.cpp has dropped: use the Q4_0 file"
    } else if n.starts_with("IQ4_NL_") {
        "; it is an ARM repacked layout that llama.cpp has dropped: use the IQ4_NL file"
    } else if n.starts_with("IQ") || n.starts_with("TQ") {
        "; this i-quant or ternary type is not supported: use a K-quant file \
         (Q4_K_M, Q5_K_M, Q6_K) or Q8_0"
    } else if n == "BF16" {
        "; use the F16 or Q8_0 file"
    } else {
        ""
    };
    let ok: Vec<&str> = SUPPORTED_TYPES
        .iter()
        .map(|&id| crate::gguf::qtype(id).name)
        .collect();
    format!(
        "tensor {name}: type {n} ({}) is not supported (supported: {}){all}{hint}",
        q.id,
        ok.join(", ")
    )
}

/// Whether `Qwen2::new` reads any weight matrix in place from the model's
/// bytes on every token (AXON_REPACK=0, or a matrix with no r8/r8q form),
/// so that the anonymous huge-page copy of the file pays (`gguf::load_then`).
/// Otherwise every matrix, the output head included, is repacked into the
/// r8 arena (huge pages itself) and the file is read once: the embedding
/// (one row per token) and the norm/bias vectors (copied out) are all
/// that's left in the mapping.
///
/// A MoE file is always mapped: its experts are most of it, and a copy
/// beside the r8 arena would hold them twice (the 30B: 18.6 + 17.9 GB).
pub fn reads_matrices_in_place(m: &Model) -> bool {
    !moe::is_moe(m)
        && reads_in_place_with(m, std::env::var("AXON_REPACK").map_or(true, |v| v != "0"))
}

/// `reads_matrices_in_place` for `Qwen2::new_with(m, repack)`.
fn reads_in_place_with(m: &Model, repack: bool) -> bool {
    if !repack {
        return true;
    }
    let tied = find_opt(m, "output.weight").is_none();
    m.tensors
        .iter()
        .filter(|t| {
            t.dims.len() >= 2
                && t.dims[1] > 1
                && !t.name.ends_with(".ffn_gate_inp.weight")
                && (t.name.starts_with("blk.")
                    || t.name == "output.weight"
                    || (tied && t.name == "token_embd.weight"))
        })
        .any(|t| repacked_len(t).is_none())
}

/// Everything `Qwen2::new` and `Tokenizer::from_gguf` rely on, checked up front
/// so a bad or unsupported model is an error instead of a panic: the
/// architecture, the hyperparameters, every tensor's presence, type and shape,
/// and the vocabulary. Reads metadata only.
pub fn check_supported(m: &Model) -> std::io::Result<()> {
    let bad = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    if !ARCHS.contains(&m.arch.as_str()) {
        return Err(bad(format!(
            "architecture '{}' is not supported (qwen2, qwen3, llama, gemma3, qwen3moe, gpt-oss, deepseek2)",
            if m.arch.is_empty() { "<none>" } else { &m.arch }
        )));
    }
    if m.arch == "deepseek2" {
        return deepseek2::check(m).map(|_| ());
    }
    let need = |k: &str| -> std::io::Result<usize> {
        let key = format!("{}.{k}", m.arch);
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
    let head_dim = match meta_num(m, &format!("{}.attention.key_length", m.arch)) {
        Some(_) => need("attention.key_length")?,
        None if dim.is_multiple_of(n_heads) => dim / n_heads,
        None => {
            return Err(bad(format!(
                "embedding_length {dim} doesn't split into {n_heads} heads"
            )));
        }
    };
    if meta_num(m, &format!("{}.attention.value_length", m.arch)).is_some()
        && need("attention.value_length")? != head_dim
    {
        return Err(bad(format!(
            "attention.value_length differs from the head size {head_dim}"
        )));
    }
    if !head_dim.is_multiple_of(2) {
        return Err(bad(format!("head size {head_dim} is odd")));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(bad(format!(
            "head_count {n_heads} is not a multiple of head_count_kv {n_kv_heads}"
        )));
    }
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let num = |k: &str| meta_num(m, &format!("{}.{k}", m.arch));
    if num("rope.dimension_count").is_some_and(|v| v != head_dim as f64) {
        return Err(bad(format!(
            "rope.dimension_count differs from the head size {head_dim} (partial RoPE)"
        )));
    }
    let moe = moe::is_moe(m);
    if !moe && num("expert_count").is_some_and(|v| v != 0.0) {
        return Err(bad(format!(
            "mixture-of-experts models are not supported on {} (qwen3moe is)",
            m.arch
        )));
    }
    let moe = match moe {
        false => None,
        true => {
            let d = moe::MoeDims {
                n_expert: need("expert_count")?,
                k: need("expert_used_count")?,
                ffn: need("expert_feed_forward_length")?,
            };
            if d.k > d.n_expert {
                return Err(bad(format!(
                    "expert_used_count {} exceeds expert_count {}",
                    d.k, d.n_expert
                )));
            }
            if num("expert_shared_count").is_some_and(|v| v != 0.0) {
                return Err(bad("shared experts are not supported".into()));
            }
            if let Some(v) = num("expert_weights_scale").filter(|&v| v != 0.0 && v != 1.0) {
                return Err(bad(format!("expert_weights_scale {v} is not supported")));
            }
            // llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE: 0 (unset) or 1 softmax.
            if let Some(v) = num("expert_gating_func").filter(|&v| v != 0.0 && v != 1.0) {
                return Err(bad(format!(
                    "expert_gating_func {v} is not supported (softmax only)"
                )));
            }
            if num("expert_group_count").is_some_and(|v| v > 1.0) {
                return Err(bad("grouped expert selection is not supported".into()));
            }
            if num("expert_weights_norm").is_some_and(|v| v == 0.0) {
                return Err(bad(
                    "expert_weights_norm false is not supported (qwen3moe renormalizes)".into(),
                ));
            }
            Some(d)
        }
    };
    if num("attention.scale").is_some_and(|v| v != 0.0) {
        return Err(bad("a custom attention.scale is not supported".into()));
    }
    // RoPE scaling: none, or a linear / YaRN entry that changes nothing
    // (factor 1, no attention factor). A factor without a type is
    // llama.cpp's linear scaling, so any factor other than 1 is refused.
    let factor = num("rope.scaling.factor").or(num("rope.scale_linear"));
    let gemma = m.arch == "gemma3";
    let yarn = m.metadata.iter().any(|(k, v)| {
        *k == format!("{}.rope.scaling.type", m.arch) && matches!(v, Val::Str(t) if t == "yarn")
    });
    if yarn && let Some(f) = factor.filter(|&f| f != 1.0 && f != 0.0) {
        // YaRN (rope.rs): any factor above 1, not on gemma3.
        if gemma || !(f > 1.0 && f.is_finite()) {
            return Err(bad(format!(
                "RoPE scaling yarn (factor {f}) is not supported"
            )));
        }
        let ok = |k: &str, lo: f64| num(k).is_none_or(|v| v.is_finite() && v >= lo);
        if !ok("rope.scaling.original_context_length", 1.0)
            || !ok("rope.scaling.yarn_beta_fast", f64::MIN_POSITIVE)
            || !ok("rope.scaling.yarn_beta_slow", f64::MIN_POSITIVE)
            || !ok("rope.scaling.yarn_ext_factor", 0.0)
            || !ok("rope.scaling.attn_factor", f64::MIN_POSITIVE)
            || !ok("rope.scaling.axon_mscale", f64::MIN_POSITIVE)
        {
            return Err(bad("RoPE scaling yarn: a parameter is out of range".into()));
        }
        if num("rope.scale_linear").is_some_and(|v| v != 1.0 && v != 0.0) {
            return Err(bad(
                "RoPE scaling yarn with rope.scale_linear is not supported".into(),
            ));
        }
        if num("rope.scaling.yarn_log_multiplier").is_some_and(|v| v != 0.0) {
            return Err(bad(
                "RoPE scaling yarn with yarn_log_multiplier is not supported".into(),
            ));
        }
    } else if let Some(f) = factor.filter(|&f| f != 1.0 && f != 0.0) {
        // gemma3's global layers: linear scaling (`rope_linear_scale`).
        let linear = !m.metadata.iter().any(|(k, v)| {
            *k == format!("{}.rope.scaling.type", m.arch)
                && !matches!(v, Val::Str(t) if t == "linear" || t == "none")
        });
        if !(gemma && linear && f > 0.0 && f.is_finite()) {
            return Err(bad(format!("RoPE scaling (factor {f}) is not supported")));
        }
    }
    let oss = m.arch == "gpt-oss";
    if oss
        && let Some(b) = num("rope.freq_base_swa")
        && Some(b) != num("rope.freq_base")
    {
        return Err(bad(format!(
            "gpt-oss rope.freq_base_swa {b} (other than rope.freq_base) is not supported"
        )));
    }
    if oss
        && let Some(w) = num("attention.sliding_window").filter(|&w| w > 0.0)
        && (w.fract() != 0.0 || w > u32::MAX as f64)
    {
        return Err(bad(format!(
            "attention.sliding_window {w} is not an integer"
        )));
    }
    if gemma {
        for k in ["final_logit_softcapping", "attn_logit_softcapping"] {
            if num(k).is_some_and(|v| v != 0.0) {
                return Err(bad(format!("gemma3 {k} is not supported")));
            }
        }
        // llama.cpp scales gemma3 27B's queries by 1/sqrt(dim / heads), not
        // 1/sqrt(head_dim) as the attention here does.
        if n_layers == 62 && dim / n_heads != head_dim {
            return Err(bad(
                "gemma3 27B (attention scale 1/sqrt(dim/heads)) is not supported".into(),
            ));
        }
        if let Some(w) = num("attention.sliding_window").filter(|&w| w > 0.0)
            && (w.fract() != 0.0 || w > u32::MAX as f64)
        {
            return Err(bad(format!(
                "attention.sliding_window {w} is not an integer"
            )));
        }
    }
    if let Some(f) =
        num("rope.scaling.attn_factor").filter(|&f| f != 1.0 && crate::rope::yarn(m).is_none())
    {
        return Err(bad(format!(
            "RoPE scaling (attention factor {f}) is not supported"
        )));
    }
    let scaling = m
        .metadata
        .iter()
        .find(|(k, _)| *k == format!("{}.rope.scaling.type", m.arch));
    match scaling {
        None => {}
        Some((_, Val::Str(t))) if ["none", "linear", "yarn"].contains(&t.as_str()) => {}
        Some((_, v)) => {
            return Err(bad(format!("RoPE scaling {v} is not supported")));
        }
    }
    if let Some(t) = find_opt(m, "rope_freqs.weight") {
        if t.qtype.id != F32 || t.n_elements != (head_dim / 2) as u64 {
            return Err(bad(format!(
                "rope_freqs.weight: {} {:?} (want {} F32 values)",
                t.qtype.name,
                t.dims,
                head_dim / 2
            )));
        }
        if !load_vec(m, t).iter().all(|&f| f.is_finite() && f > 0.0) {
            return Err(bad("rope_freqs.weight: factors must be positive".into()));
        }
    }

    // `rows` = None: a vector of `cols` elements; Some(r): a cols x r matrix.
    let check = |name: &str, cols: usize, rows: Option<usize>| -> std::io::Result<&Tensor> {
        let t = find_opt(m, name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        let q = t.qtype;
        if !SUPPORTED_TYPES.contains(&q.id) {
            return Err(bad(unsupported_type(m, name, q)));
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
        t("attn_q.weight", dim, Some(q_dim))?;
        t("attn_k.weight", dim, Some(kv_dim))?;
        t("attn_v.weight", dim, Some(kv_dim))?;
        // Per-head q/k RMSNorm before RoPE: qwen3 has it, qwen2 doesn't,
        // llama's graph has none (one in the file would be ignored there).
        for b in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            let has = find_opt(m, &format!("blk.{i}.{b}")).is_some();
            if m.arch == "llama" && has {
                return Err(bad(format!("blk.{i}.{b}: llama has no q/k norm")));
            }
            if oss && has {
                return Err(bad(format!("blk.{i}.{b}: gpt-oss has no q/k norm")));
            }
            if m.arch == "qwen3" || (moe.is_some() && !oss) || gemma || has {
                t(b, head_dim, None)?;
            }
        }
        // gemma3's norms of the attention and FFN outputs ("sandwich").
        // gpt-oss: its `post_attention_norm` is the pre-FFN norm (llama.cpp's
        // attn_post_norm), and it has an output bias and attention sinks.
        for b in ["post_attention_norm.weight", "post_ffw_norm.weight"] {
            if gemma || (oss && b == "post_attention_norm.weight") {
                t(b, dim, None)?;
            } else if find_opt(m, &format!("blk.{i}.{b}")).is_some() {
                return Err(bad(format!("blk.{i}.{b} is not supported on {}", m.arch)));
            }
        }
        if oss {
            t("attn_output.bias", dim, None)?;
            let s = t("attn_sinks.weight", n_heads, None)?;
            if s.qtype.id != F32 {
                return Err(bad(format!(
                    "blk.{i}.attn_sinks.weight: type {} (want F32)",
                    s.qtype.name
                )));
            }
        } else if find_opt(m, &format!("blk.{i}.attn_sinks.weight")).is_some() {
            return Err(bad(format!(
                "blk.{i}.attn_sinks.weight is not supported on {}",
                m.arch
            )));
        }
        // Biases nothing here would add.
        for b in ["ffn_gate.bias", "ffn_up.bias", "ffn_down.bias"] {
            if find_opt(m, &format!("blk.{i}.{b}")).is_some() {
                return Err(bad(format!("blk.{i}.{b} is not supported")));
            }
        }
        for (b, n) in [
            ("attn_q.bias", q_dim),
            ("attn_k.bias", kv_dim),
            ("attn_v.bias", kv_dim),
        ] {
            if find_opt(m, &format!("blk.{i}.{b}")).is_some() {
                t(b, n, None)?;
            }
        }
        if !oss && find_opt(m, &format!("blk.{i}.attn_output.bias")).is_some() {
            return Err(bad(format!("blk.{i}.attn_output.bias is not supported")));
        }
        t("attn_output.weight", q_dim, Some(dim))?;
        if !oss {
            t("ffn_norm.weight", dim, None)?;
        }
        let Some(d) = moe else {
            let ffn = t("ffn_gate.weight", dim, Some(0))?.dims[1] as usize;
            t("ffn_up.weight", dim, Some(ffn))?;
            t("ffn_down.weight", ffn, Some(dim))?;
            continue;
        };
        // Experts: [cols, rows, n_expert] tensors, and the router.
        for b in [
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "ffn_gate_inp_shexp.weight",
            "ffn_gate_inp.bias",
            "exp_probs_b.bias",
        ] {
            if oss && b == "ffn_gate_inp.bias" {
                t(b, d.n_expert, None)?;
                continue;
            }
            if find_opt(m, &format!("blk.{i}.{b}")).is_some() {
                return Err(bad(format!("blk.{i}.{b} is not supported on {}", m.arch)));
            }
        }
        let r = t("ffn_gate_inp.weight", dim, Some(d.n_expert))?;
        if r.qtype.id != F32 && r.qtype.id != F16 {
            return Err(bad(format!(
                "blk.{i}.ffn_gate_inp.weight: type {} (want F32 or F16)",
                r.qtype.name
            )));
        }
        for (b, cols, rows) in [
            ("ffn_gate_exps.weight", dim, d.ffn),
            ("ffn_up_exps.weight", dim, d.ffn),
            ("ffn_down_exps.weight", d.ffn, dim),
        ] {
            let name = format!("blk.{i}.{b}");
            let t = find_opt(m, &name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
            let q = t.qtype;
            if !(SUPPORTED_TYPES.contains(&q.id) || oss && EXPERT_TYPES.contains(&q.id)) {
                return Err(bad(unsupported_type(m, &name, q)));
            }
            let want = [cols as u64, rows as u64, d.n_expert as u64];
            if t.dims.len() < 3
                || t.dims[..3] != want
                || t.dims[3..].iter().any(|&x| x != 1)
                || !t.dims[0].is_multiple_of(q.block_size as u64)
            {
                return Err(bad(format!(
                    "tensor {name}: shape {:?} doesn't fit the model ({cols} x {rows} x {})",
                    t.dims, d.n_expert
                )));
            }
            // gpt-oss: each expert's bias, [rows, n_expert].
            let bn = format!("blk.{i}.{}", b.replace(".weight", ".bias"));
            match (oss, find_opt(m, &bn)) {
                (true, _) => {
                    check(&bn, rows * d.n_expert, None)?;
                }
                (false, Some(_)) => {
                    return Err(bad(format!("{bn} is not supported on {}", m.arch)));
                }
                (false, None) => {}
            }
        }
    }

    crate::tokenizer::pre_of(m).map_err(bad)?;
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
        Self::build(m, repack, false, usize::MAX, false)
    }

    /// The experts of a MoE model (every layer's, repacked unless
    /// AXON_REPACK=0), for the GPU backend that runs everything else
    /// (`crate::moe`): no output head, and the attention matrices are left
    /// in the mapping, never read. It runs `moe_layer` / `moe_layer_batch`
    /// only.
    pub(crate) fn new_experts(m: &'a Model) -> Qwen2<'a> {
        let repack = std::env::var("AXON_REPACK").map_or(true, |v| v != "0");
        Self::build(m, repack, false, usize::MAX, true)
    }

    /// The embedding and layers 0..k only (all of them, and the output head,
    /// if k >= n_layers): the CPU half of a model split across devices
    /// (`hybrid`). Only these weights are read, repacked and released; the
    /// rest of the mapping is never touched. Without the head it runs
    /// `hidden` / `hidden_batch` only; its cache holds k layers.
    pub fn new_layers(m: &'a Model, repack: bool, k: usize) -> Qwen2<'a> {
        Self::build(m, repack, false, k, false)
    }

    /// `tie`: use the token embedding as the output matrix even if the model
    /// has its own (tests: the tied-embedding case on any model). `keep`:
    /// layers 0..keep only, the output head only if that's all of them.
    /// `experts`: `new_experts`'s model (no head, attention in place).
    fn build(m: &'a Model, repack: bool, tie: bool, keep: usize, experts: bool) -> Qwen2<'a> {
        if let Err(e) = check_supported(m) {
            panic!("{e}");
        }
        if m.arch == "deepseek2" {
            assert!(
                !tie && keep == usize::MAX,
                "deepseek2: the whole model (or its experts) only"
            );
            return deepseek2::build(m, repack, experts);
        }
        let a = &m.arch;
        let key = |k: &str| format!("{a}.{k}");
        let need =
            |k: &str| meta_num(m, &key(k)).unwrap_or_else(|| panic!("missing metadata {}", key(k)));

        let n_layers = need("block_count") as usize;
        let (k, head) = (keep.min(n_layers), keep >= n_layers && !experts);
        let dim = need("embedding_length") as usize;
        let n_heads = need("attention.head_count") as usize;
        let n_kv_heads = need("attention.head_count_kv") as usize;
        let head_dim =
            meta_num(m, &key("attention.key_length")).map_or(dim / n_heads, |v| v as usize);
        let eps = meta_num(m, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f32;
        let freqs = rope_freqs(m, head_dim);
        let gemma = m.arch == "gemma3";
        let swa = gemma_swa(m, n_layers);
        let oss = m.arch == "gpt-oss";
        // gpt-oss's local layers turn as its global ones (YaRN too).
        let freqs_swa = match &swa {
            Some(_) if oss => freqs.clone(),
            Some(_) => rope_freqs_swa(m, head_dim),
            None => Vec::new(),
        };

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
        // A MoE layer: the attention matrices, then each expert tensor's
        // n_expert matrices (one repack each: `moe::Moe`).
        let moe_dims = moe::moe_dims(m);
        let layer_mats = |i: usize| -> Vec<&Tensor> {
            match moe_dims {
                None => MATS.map(|s| lt(i, s)).to_vec(),
                Some(d) => MATS[..4]
                    .iter()
                    .map(|s| lt(i, s))
                    .chain(
                        moe::EXPS
                            .iter()
                            .flat_map(|s| std::iter::repeat_n(lt(i, s), d.n_expert)),
                    )
                    .collect(),
            }
        };
        let mut plan: Vec<Option<(usize, usize)>> = Vec::new();
        let mut total = 0;
        let out_ts = head.then_some(out_t);
        // `experts`: a layer's first 4 (attention) stay in the mapping.
        let held = |i: usize| {
            layer_mats(i)
                .into_iter()
                .enumerate()
                .map(move |(j, t)| (t, !(experts && j < 4)))
        };
        for (t, hold) in out_ts
            .map(|t| (t, true))
            .into_iter()
            .chain((0..k).flat_map(held))
        {
            let len = if repack && hold {
                repacked_len(t)
            } else {
                None
            };
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
            .map(|i| std::sync::Mutex::new(slots.by_ref().take(layer_mats(i).len()).collect()))
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
            let [wq, wk, wv, wo] = [0, 1, 2, 3].map(|j| w(MATS[j]));
            let ffn = match moe_dims {
                None => Ffn::Dense {
                    gate: w(MATS[4]),
                    up: w(MATS[5]),
                    down: w(MATS[6]),
                },
                Some(d) => {
                    // Each expert's slice of the 3-D tensor, repacked on its
                    // own; the tensor's pages go back once all are.
                    let [gate, up, down] = moe::EXPS.map(|s| {
                        let t = t(s);
                        // Read in place (MXFP4: no repack): the page tables
                        // filled now, not by a fault per 4 KB on the first
                        // tokens (file mapping only; no extra memory).
                        if repacked_len(t).is_none() || !repack {
                            m.prefault(t);
                        }
                        let ws: Vec<Weight> = (0..d.n_expert)
                            .map(|e| {
                                let dst = dst.next().unwrap();
                                let repacked = dst.is_some();
                                let w = Weight::load_bytes(t, moe::expert_bytes(m, t, e), dst);
                                if repacked && e + 1 == d.n_expert {
                                    m.release(t);
                                }
                                w
                            })
                            .collect();
                        ws
                    });
                    let router = vals(t("ffn_gate_inp.weight"));
                    let mut moe = moe::Moe::new(d, router, gate, up, down, moe::Gate::SOFTMAX_NORM);
                    if oss {
                        let [router, gate, up, down] = [
                            "ffn_gate_inp.bias",
                            "ffn_gate_exps.bias",
                            "ffn_up_exps.bias",
                            "ffn_down_exps.bias",
                        ]
                        .map(|s| vals(t(s)));
                        moe.oai = Some(Box::new(moe::Oai {
                            router,
                            gate,
                            up,
                            down,
                        }));
                    }
                    Ffn::Moe(Box::new(moe))
                }
            };
            Layer {
                attn_norm: vals(t("attn_norm.weight")),
                wq,
                wk,
                wv,
                bq: bias("attn_q.bias"),
                bk: bias("attn_k.bias"),
                bv: bias("attn_v.bias"),
                q_norm: bias("attn_q_norm.weight"),
                k_norm: bias("attn_k_norm.weight"),
                wo,
                // gpt-oss's post_attention_norm is its pre-FFN norm.
                ffn_norm: vals(t(match oss {
                    true => "post_attention_norm.weight",
                    false => "ffn_norm.weight",
                })),
                ffn,
                post_attn_norm: if oss {
                    None
                } else {
                    bias("post_attention_norm.weight")
                },
                post_ffn_norm: bias("post_ffw_norm.weight"),
                window: swa.as_ref().filter(|s| s.local[i]).map(|s| s.window),
                bo: bias("attn_output.bias"),
                sinks: bias("attn_sinks.weight"),
            }
        });
        drop(layer_slots);
        drop(slots);
        if let Some(a) = arena.take() {
            let map = std::sync::Arc::new(a.make_read_only().expect("r8: mprotect"));
            let ws = output
                .iter_mut()
                .chain(layers.iter_mut().flat_map(|l| l.weights_mut()));
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
            q_dim: n_heads * head_dim,
            vocab,
            eps,
            freqs: [freqs, freqs_swa].concat().into_boxed_slice(),
            rope_m: crate::rope::mscale(m),
            gemma,
            token_embd,
            embd: m.tensor_data(token_embd),
            output_norm: output_norm.into_boxed_slice(),
            output,
            layers,
            ds2: None,
        }
    }

    /// Layer l's weights as the CPU holds them, for streaming them to the
    /// GPU (`cuda::qwen2::stream`): repacked bytes or the file's. Not for
    /// deepseek2 (its blocks are `ds2`'s; the split that streams is refused).
    pub(crate) fn host_layer(&self, l: usize) -> crate::cuda::qwen2::stream::HostLayer<'_> {
        assert!(
            !self.is_mla(),
            "host_layer: deepseek2 has no streamable layers"
        );
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
        let (gate, up, down) = ly.dense();
        HostLayer {
            mats: [
                mat(&ly.wq),
                mat(&ly.wk),
                mat(&ly.wv),
                mat(&ly.wo),
                mat(gate),
                mat(up),
                mat(down),
            ],
            attn_norm: &ly.attn_norm,
            ffn_norm: &ly.ffn_norm,
            bq: ly.bq.as_deref(),
            bk: ly.bk.as_deref(),
            bv: ly.bv.as_deref(),
            qn: ly.q_norm.as_deref(),
            kn: ly.k_norm.as_deref(),
        }
    }

    /// The whole mapping the repacked matrices live in (None if none is).
    pub(crate) fn r8_arena(&self) -> Option<&[u8]> {
        let w = self
            .layers
            .iter()
            .flat_map(|l| l.weights())
            .chain(self.output.as_ref())
            .find_map(|w| w.r8.as_ref())?;
        Some(&w.map[..])
    }

    /// A deepseek2 (MLA) model: K-only cache rows of `head_dim` (576).
    pub(crate) fn is_mla(&self) -> bool {
        self.ds2.is_some()
    }

    /// Layers here (0..k of the model): `Layer`s, so 0 on deepseek2, whose
    /// blocks are `ds2`'s (`n_layers` counts them; only the refused split
    /// and streaming paths call this).
    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// The embeddings of `tokens`, token-major (dim floats each).
    pub(crate) fn embed_batch(&self, tokens: &[u32]) -> Vec<f32> {
        let rows = par_map(tokens.len(), self.dim, |b| self.embed(tokens[b]));
        rows.concat()
    }

    pub fn new_cache(&self, max_ctx: usize) -> KvCache {
        if self.ds2.is_some() {
            return deepseek2::new_cache(self.n_layers, self.head_dim, max_ctx);
        }
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
        pool::for_each_chunk_mut_in(Kind::Decode, &mut out, chunk, |c, mut ys| {
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
        pool::for_each_chunk_mut_in(Kind::Decode, &mut out, chunk, |c, ys| {
            let (r, k) = (c * chunk, ys.len());
            let mut ub = [0f32; MAX_CHUNK];
            let ub = &mut ub[..k];
            self.rows_vec(gate, r, &xq, ys, cfg, own_all);
            self.rows_vec(up, r, &xq, ub, cfg, own_all);
            self.glu(ys, ub);
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
    /// Tokens go in blocks (`per_block`), each block's columns gathered
    /// row by row (`gather_cols`: a row's slots are adjacent) into scratch.
    fn quantize_cols(y: &[f32], slots: usize, nb: usize, types: &[u32]) -> Vec<QuantizedAct> {
        let n = y.len() / slots;
        let mut out: Vec<Option<QuantizedAct>> = (0..nb).map(|_| None).collect();
        per_block(&mut out, n, |b0, blk| {
            with_tail_scratch(blk.len() * n, |sc| {
                let mut cols: Vec<&mut [f32]> = sc.chunks_exact_mut(n).collect();
                gather_cols(y, slots, b0, &mut cols);
                for (o, x) in blk.iter_mut().zip(cols) {
                    *o = Some(QuantizedAct::for_types(x, types));
                }
            });
        });
        out.into_iter().map(|o| o.unwrap()).collect()
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
        self.matmul_q_in(ws, xq, Vec::new())
    }

    /// `matmul_q` into `out`, a recycled buffer (resized; its old contents
    /// are overwritten, but for padding slots, which nothing reads): no
    /// allocation and no serial zeroing of megabytes per call.
    fn matmul_q_in<const N: usize>(
        &self,
        ws: [&Weight; N],
        xq: &[QuantizedAct],
        out: Vec<f32>,
    ) -> (Vec<f32>, [usize; N], usize) {
        let nb = xq.len();
        let n_in = xq[0].len();
        let (starts, total, align) = Self::joint_rows(ws, n_in);
        let slots = Self::slots_for(&ws, n_in, nb);
        let xr: Vec<&QuantizedAct> = (0..slots).map(|b| &xq[b.min(nb - 1)]).collect();
        let mut out = recycle(out, total * slots);
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
            let all_r8 = ws.iter().all(|w| w.r8.is_some());
            run_split(
                &mut out,
                slots,
                nb,
                rows,
                all_r8,
                |r0, k, t0, t1, mut ys| {
                    let (xt, st) = if t1 - t0 == nb {
                        (&xr[..], slots)
                    } else {
                        (&xr[t0..t1], t1 - t0)
                    };
                    let mut g = r0;
                    let end_g = r0 + k;
                    while g < end_g {
                        let t = starts.iter().rposition(|&s| s <= g).unwrap();
                        let end = starts.get(t + 1).copied().unwrap_or(total);
                        let kk = (end - g).min(end_g - g);
                        let (run, rest) = std::mem::take(&mut ys).split_at_mut(kk * st);
                        self.rows_batch(ws[t], g - starts[t], xt, t1 - t0, run);
                        g += kk;
                        ys = rest;
                    }
                },
            );
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

    /// `matvec_swiglu` for a batch of inputs quantized for both matrices'
    /// types: per row r, per token b, the gate and up dots combined with the
    /// same formula. Layout and slot count returned as in `matmul_raw`.
    /// The output goes into `out`, recycled as in `matmul_q_in`.
    fn matmul_swiglu_q(
        &self,
        gate: &Weight,
        up: &Weight,
        xq: &[QuantizedAct],
        out: Vec<f32>,
    ) -> (Vec<f32>, usize) {
        let nb = xq.len();
        let n_in = xq[0].len();
        let n = gate.rows();
        let (_, _, a0) = Self::joint_rows([gate], n_in);
        let (_, _, a1) = Self::joint_rows([up], n_in);
        assert_eq!(up.rows(), n, "swiglu: gate/up row counts differ");
        let slots = Self::slots_for(&[gate, up], n_in, nb);
        let xr: Vec<&QuantizedAct> = (0..slots).map(|b| &xq[b.min(nb - 1)]).collect();
        let mut out = recycle(out, n * slots);
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
                        self.glu(yr, uv);
                    }
                });
            });
        } else {
            // The chunk's gate rows, then its up rows, then combine (real
            // tokens only: padding slots may be unwritten on r8 rows).
            let all_r8 = gate.r8.is_some() && up.r8.is_some();
            run_split(&mut out, slots, nb, rows, all_r8, |r0, _, t0, t1, ys| {
                let (xt, st) = if t1 - t0 == nb {
                    (&xr[..], slots)
                } else {
                    (&xr[t0..t1], t1 - t0)
                };
                with_rows_scratch(ys.len(), |uv| {
                    self.rows_batch(gate, r0, xt, t1 - t0, ys);
                    self.rows_batch(up, r0, xt, t1 - t0, uv);
                    for (yr, ur) in ys.chunks_mut(st).zip(uv.chunks(st)) {
                        self.glu(&mut yr[..t1 - t0], &ur[..t1 - t0]);
                    }
                });
            });
        }
        (out, slots)
    }

    /// The gated FFN's combine in place: g = act(g) * u, act SiLU, or
    /// GELU (tanh form) on gemma3.
    #[inline]
    fn glu(&self, g: &mut [f32], u: &[f32]) {
        if self.gemma {
            vmath::geglu(g, u);
        } else {
            vmath::swiglu(g, u);
        }
    }

    /// gemma3's post-norm of a matmul output in (row, slot) layout: each
    /// token b < nb's column replaced by `rmsnorm` of it with `w`, the same
    /// call as the one-token path's on the same values.
    fn norm_cols(&self, y: &mut [f32], slots: usize, nb: usize, w: &[f32]) {
        let dim = self.dim;
        let cols = par_map(nb, dim, |b| self.rmsnorm(&gather_col(y, slots, b, dim), w));
        let rows = 64;
        pool::for_each_chunk_mut(&mut y[..dim * slots], rows * slots, |c, ys| {
            for (j, yr) in ys.chunks_exact_mut(slots).enumerate() {
                let r = c * rows + j;
                for (b, col) in cols.iter().enumerate() {
                    yr[b] = col[r];
                }
            }
        });
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
    /// pos * freqs[i] (base^(-2i/head_dim), over Llama 3's factor). Low i
    /// spins fast, high i spins slowly. Every head in every layer uses the
    /// same angles, so this runs once a token.
    fn rope_table(&self, pos: usize) -> Vec<(f32, f32)> {
        Self::rope_table_of(&self.freqs[..self.head_dim / 2], pos, self.rope_m)
    }

    /// `rope_table` of the local layers (`freqs_swa`; empty if none). The
    /// mscale is YaRN's, so 1 on gemma3 (which refuses YaRN): gpt-oss's
    /// local layers are scaled as its global ones.
    fn rope_table_swa(&self, pos: usize) -> Vec<(f32, f32)> {
        Self::rope_table_of(&self.freqs[self.head_dim / 2..], pos, self.rope_m)
    }

    /// The table at `pos`, each value times `m` (YaRN's mscale) in f64
    /// before the rounding, as decode.cu's rope_table: m = 1 leaves the
    /// bits as they were.
    fn rope_table_of(freqs: &[f64], pos: usize, m: f32) -> Vec<(f32, f32)> {
        let m = m as f64;
        freqs
            .iter()
            .map(|&freq| {
                let angle = pos as f64 * freq;
                ((angle.sin() * m) as f32, (angle.cos() * m) as f32)
            })
            .collect()
    }

    /// RoPE on q and k (heads of head_dim), each head RMS-normalized first
    /// when the layer has q/k norms (qwen3). Shared by `hidden` and
    /// `hidden_batch` so both run the same ops.
    #[inline]
    fn qk_rope(&self, layer: &Layer, q: &mut [f32], k: &mut [f32], rope: &[(f32, f32)]) {
        let hd = self.head_dim;
        if let Some(w) = &layer.q_norm {
            rmsnorm_heads(q, hd, w, self.eps);
        }
        if let Some(w) = &layer.k_norm {
            rmsnorm_heads(k, hd, w, self.eps);
        }
        rope_heads(q, hd, rope);
        rope_heads(k, hd, rope);
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
        if self.gemma {
            // llama.cpp's build_inp_embd(tok_embd, sqrtf(n_embd)): an f32
            // multiply.
            let sc = (self.dim as f32).sqrt();
            for v in &mut x {
                *v *= sc;
            }
        }
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
        if self.ds2.is_some() {
            return self.ds2_hidden_batch(tokens, start, cache);
        }
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

        // Every per-token step runs over the pool by token (`per_token`,
        // `per_block`): with a few hundred tokens, a serial pass here leaves
        // every worker idle. A few tokens' worth (a speculative verify) runs
        // inline instead. Steps between two matmuls are fused into one pass
        // (residual add + norm + quantize; gather + bias + RoPE + cache
        // store): each pass is a fork/join, and a join waits for the slowest
        // core. Same ops on the same values as `hidden`, in the same order.
        let mut xs: Vec<Vec<f32>> = par_map(nb, dim, |b| self.embed(tokens[b]));
        let ropes: Vec<_> = par_map(nb, hd * 16, |b| self.rope_table(start + b));
        let ropes_swa: Vec<_> = match self.freqs.len() > hd / 2 {
            false => Vec::new(),
            true => par_map(nb, hd * 16, |b| self.rope_table_swa(start + b)),
        };
        let q_dim = self.q_dim;
        // Reused across layers; every element is overwritten before it's read.
        let mut qbuf = vec![0.0f32; nb * q_dim];
        let mut att = vec![0.0f32; nb * q_dim];
        let qn = group * hd; // floats per (token, kv head) of attention output
        debug_assert_eq!(q_dim, n_kv * qn, "feed_batch: q_dim != heads * head_dim");
        // The last matmul's output, (row, slot) layout, not yet added to xs.
        let mut pending: Option<(Vec<f32>, usize)> = None;
        // Matmul outputs, recycled across layers (`matmul_q_in`).
        let (mut qkv_buf, mut o_buf, mut act_buf, mut down_buf) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (l, layer) in self.layers.iter().enumerate() {
            // Steps as in `hidden`, each applied to every token of the batch.
            let res = pending.take();
            let types = [layer.wq.qt(), layer.wk.qt(), layer.wv.qt()];
            let hq = self.add_norm_quant(&mut xs, res.as_ref(), &layer.attn_norm, &types);
            if let Some((y, _)) = res {
                down_buf = y;
            }
            let qkvw = [&layer.wq, &layer.wk, &layer.wv];
            let (qkv, starts, slots) = self.matmul_q_in(qkvw, &hq, std::mem::take(&mut qkv_buf));
            drop(hq);
            with_kv_layer!(cache, l, |kc, vc| {
                // Bias, RoPE, and k/v into the cache, per token: same values,
                // same ops as `hidden`, then the same rounding to the cache type.
                let span = start * kv_dim..(start + nb) * kv_dim;
                let (kl, vl) = (&mut kc[span.clone()], &mut vc[span]);
                let mut jobs: Vec<_> = qbuf
                    .chunks_mut(q_dim)
                    .zip(kl.chunks_mut(kv_dim).zip(vl.chunks_mut(kv_dim)))
                    .collect();
                let (yq, yk, yv) = (
                    &qkv[starts[0] * slots..starts[1] * slots],
                    &qkv[starts[1] * slots..starts[2] * slots],
                    &qkv[starts[2] * slots..(starts[2] + kv_dim) * slots],
                );
                per_block(&mut jobs, q_dim + 2 * kv_dim, |b0, blk| {
                    let mut qs: Vec<&mut [f32]> = blk.iter_mut().map(|(q, _)| &mut **q).collect();
                    gather_cols(yq, slots, b0, &mut qs);
                    with_rows_scratch(2 * kv_dim, |kv| {
                        let (k, v) = kv.split_at_mut(kv_dim);
                        for (j, (q, (kd, vd))) in blk.iter_mut().enumerate() {
                            let b = b0 + j;
                            gather_cols(yk, slots, b, &mut [&mut *k]);
                            gather_cols(yv, slots, b, &mut [&mut *v]);
                            if let Some(bias) = &layer.bq {
                                add_assign(q, bias);
                            }
                            if let Some(bias) = &layer.bk {
                                add_assign(k, bias);
                            }
                            if let Some(bias) = &layer.bv {
                                add_assign(v, bias);
                            }
                            let rope = match layer.window {
                                Some(_) => &ropes_swa[b],
                                None => &ropes[b],
                            };
                            self.qk_rope(layer, q, k, rope);
                            KvElem::store(kd, k);
                            KvElem::store(vd, v);
                        }
                    });
                });
                drop(jobs);
                let (kc, vc) = (&*kc, &*vc);
                let qs = &qbuf;

                // Causal attention, one (token, kv head) pair per pool task,
                // longest first: all `group` query heads sharing that kv head
                // together, so each key and value row is loaded once for all of
                // them. Token b sees positions 0..=start + b only. Per query head
                // the arithmetic is exactly `hidden`'s: same dot, same softmax
                // order, same fused `o += w v` order over t (`attend_group`).
                // A gemma3 local layer sees positions s0..=start + b only
                // (`window_start`), the same slice as `hidden`'s.
                pool::for_each_chunk_mut_rev(&mut att, qn, |c, out| {
                    let (b, kvh) = (c / n_kv, c % n_kv);
                    let np = start + b + 1;
                    let s0 = window_start(layer.window, start + b) * kv_dim;
                    let qg = &qs[b * q_dim + kvh * qn..b * q_dim + (kvh + 1) * qn];
                    let (kw, vw) = (&kc[s0..], &vc[s0..]);
                    let sk = layer
                        .sinks
                        .as_deref()
                        .map(|s| &s[kvh * group..(kvh + 1) * group]);
                    attend_group(qg, kw, vw, kv_dim, hd, kvh, np - s0 / kv_dim, sk, out);
                });
            });
            qkv_buf = qkv;

            let att_rows: Vec<&[f32]> = att.chunks(q_dim).collect();
            let aq = Self::quantize_batch(&att_rows, &[layer.wo.qt()]);
            let (mut os, _, slots) = self.matmul_q_in([&layer.wo], &aq, std::mem::take(&mut o_buf));
            drop(aq);
            if let Some(b) = &layer.bo {
                bias_cols(&mut os, slots, nb, b);
            }
            if let Some(w) = &layer.post_attn_norm {
                self.norm_cols(&mut os, slots, nb, w);
            }

            let res = (os, slots);
            let (gu, gw, w_down) = match &layer.ffn {
                Ffn::Dense { gate, up, down } => (gate, up, down),
                Ffn::Moe(moe) => {
                    let hs = self.add_norm(&mut xs, Some(&res), &layer.ffn_norm);
                    o_buf = res.0;
                    let y = self.moe_ffn_batch(moe, &hs, std::mem::take(&mut down_buf));
                    pending = Some((y, nb));
                    continue;
                }
            };
            let types = [gu.qt(), gw.qt()];
            let hq = self.add_norm_quant(&mut xs, Some(&res), &layer.ffn_norm, &types);
            o_buf = res.0;
            let (acts, a_slots) = self.matmul_swiglu_q(gu, gw, &hq, std::mem::take(&mut act_buf));
            drop(hq);
            let aq = Self::quantize_cols(&acts, a_slots, nb, &[w_down.qt()]);
            act_buf = acts;
            let mut down = self.matmul_q_in([w_down], &aq, std::mem::take(&mut down_buf));
            if let Some(w) = &layer.post_ffn_norm {
                self.norm_cols(&mut down.0, down.2, nb, w);
            }
            pending = Some((down.0, down.2));
        }
        if let Some((y, slots)) = pending {
            add_unbatched(&mut xs, &y, slots);
        }
        xs
    }

    /// Per token b: xs[b] += column b of `res` (a matmul output in (row,
    /// slot) layout) if given, then RMSNorm with `w` and quantize for
    /// `types`, in one pass. The same ops as `add_unbatched`, `rmsnorm_into`
    /// and `quantize_batch` one after another.
    fn add_norm_quant(
        &self,
        xs: &mut [Vec<f32>],
        res: Option<&(Vec<f32>, usize)>,
        w: &[f32],
        types: &[u32],
    ) -> Vec<QuantizedAct> {
        let dim = self.dim;
        let mut out: Vec<Option<QuantizedAct>> = (0..xs.len()).map(|_| None).collect();
        let mut items: Vec<_> = xs.iter_mut().zip(out.iter_mut()).collect();
        per_block(&mut items, dim, |b0, blk| {
            if let Some((y, slots)) = res {
                let mut rows: Vec<&mut [f32]> = blk.iter_mut().map(|(x, _)| &mut x[..]).collect();
                add_cols(y, *slots, b0, &mut rows);
            }
            with_rows_scratch(dim, |h| {
                for (x, o) in blk.iter_mut() {
                    self.rmsnorm_into(x, w, h);
                    **o = Some(QuantizedAct::for_types(h, types));
                }
            });
        });
        out.into_iter().map(|o| o.unwrap()).collect()
    }

    /// `add_norm_quant` without the quantization: the normed vectors (the
    /// MoE router reads them in f32).
    fn add_norm(
        &self,
        xs: &mut [Vec<f32>],
        res: Option<&(Vec<f32>, usize)>,
        w: &[f32],
    ) -> Vec<Vec<f32>> {
        let dim = self.dim;
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); xs.len()];
        let mut items: Vec<_> = xs.iter_mut().zip(out.iter_mut()).collect();
        per_block(&mut items, dim, |b0, blk| {
            if let Some((y, slots)) = res {
                let mut rows: Vec<&mut [f32]> = blk.iter_mut().map(|(x, _)| &mut x[..]).collect();
                add_cols(y, *slots, b0, &mut rows);
            }
            for (x, o) in blk.iter_mut() {
                **o = self.rmsnorm(x, w);
            }
        });
        out
    }

    /// Every layer for one token: store its keys and values in the cache and
    /// return the final hidden state (before the output norm).
    pub(crate) fn hidden(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        if self.ds2.is_some() {
            return self.ds2_hidden(token, pos, cache);
        }
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
        let rope_swa = self.rope_table_swa(pos);
        // Attention scratch (step 5), kept on this thread across tokens so
        // decode attention allocates nothing once warm.
        let np = pos + 1;
        let threads = Pool::global().threads();
        let (mut buf, mut att_out) = DECODE.take();
        att_out.resize(self.q_dim, 0.0);
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

            // 3. Rotate q and k by position (RoPE), one head at a time, each
            //    head RMS-normalized first if the model says so (qwen3).
            let table = if layer.window.is_some() {
                &rope_swa
            } else {
                &rope
            };
            self.qk_rope(layer, &mut q, &mut k, table);

            // 4. Store this position's k and v in the cache so later tokens can see them.
            //    (rounded to the cache's type: f16 unless AXON_KV=f32).
            with_kv_layer!(cache, l, |kc, vc| {
                let row = pos * kv_dim..(pos + 1) * kv_dim;
                KvElem::store(&mut kc[row.clone()], &k);
                KvElem::store(&mut vc[row], &v);

                // 5. Each query head attends over all positions 0..=pos (the work
                //    grows with context, so it can't stay serial). GQA: query head
                //    qh reads kv head qh / group. See `attend`. A gemma3 local
                //    layer sees positions s0..=pos only (`window_start`).
                let s0 = window_start(layer.window, pos);
                let tile = decode_tile(np - s0, self.n_kv_heads);
                attend(
                    &q,
                    &kc[s0 * kv_dim..],
                    &vc[s0 * kv_dim..],
                    kv_dim,
                    hd,
                    np - s0,
                    tile,
                    threads,
                    layer.sinks.as_deref(),
                    &mut buf,
                    &mut att_out,
                );
            });

            // 6. Mix the heads back together and add to the residual stream.
            let mut o = self.matvec(&layer.wo, &att_out);
            if let Some(b) = &layer.bo {
                add_assign(&mut o, b);
            }
            if let Some(w) = &layer.post_attn_norm {
                o = self.rmsnorm(&o, w);
            }
            add_assign(&mut x[..dim], &o[..dim]);

            // ---- Feed-forward sub-block (SwiGLU) ----
            // 7. Normalize, then gate * up: silu(gate(h)) decides how much of up(h) passes.
            let h = self.rmsnorm(&x, &layer.ffn_norm);
            let mut d = match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    let act = self.matvec_swiglu(gate, up, &h);
                    // 8. Project back down to model width and add to the residual stream.
                    self.matvec(down, &act)
                }
                // 7-8 with experts: the router's top k, their FFNs summed.
                Ffn::Moe(moe) => self.moe_ffn(moe, &h),
            };
            if let Some(w) = &layer.post_ffn_norm {
                d = self.rmsnorm(&d, w);
            }
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

/// `v` resized to `n` for reuse as a matmul output: only growth is zeroed.
fn recycle(mut v: Vec<f32>, n: usize) -> Vec<f32> {
    if v.len() < n {
        v.resize(n, 0.0);
    } else {
        v.truncate(n);
    }
    v
}

/// Split a batched matmul's output (`out`: rows x `slots`, row-major) into
/// pool chunks and call `f(r0, k, t0, t1, ys)` for each: rows r0..r0 + k
/// times tokens t0..t1, results into `ys` (k rows of stride ys.len() / k).
/// Chunks of `rows` rows over the whole batch (t0..t1 = 0..nb, ys = those
/// rows of `out` in place, stride `slots`) go first; when `split` allows
/// (every matrix repacked: the r8 GEMM takes any token range) the last
/// row chunk per thread is cut into token ranges, computed into scratch
/// and copied out, so the join waits for a quarter of a chunk on the
/// slowest core instead of a whole one. Each output is still one dot of
/// one row and one token, as in an unsplit run: results don't change.
fn run_split(
    out: &mut [f32],
    slots: usize,
    nb: usize,
    rows: usize,
    split: bool,
    f: impl Fn(usize, usize, usize, usize, &mut [f32]) + Sync,
) {
    let total = out.len() / slots;
    let n_rc = total.div_ceil(rows);
    let threads = Pool::global().threads();
    let parts = if !split || nb < 16 || threads == 1 {
        1
    } else {
        4
    };
    let n_split = if parts > 1 { n_rc.min(threads) } else { 0 };
    let full = n_rc - n_split;
    let part = nb.div_ceil(parts).next_multiple_of(4);
    let base = pool::SendPtr(out.as_mut_ptr());
    Pool::global().run(full + n_split * parts, &|c| {
        let (rc, t0, t1) = if c < full {
            (c, 0, nb)
        } else {
            let s = c - full;
            let p = s % parts;
            (
                full + s / parts,
                (p * part).min(nb),
                ((p + 1) * part).min(nb),
            )
        };
        let r0 = rc * rows;
        let k = rows.min(total - r0);
        if t0 >= t1 {
            return;
        }
        if t1 - t0 == nb {
            // SAFETY: rows r0..r0 + k of `out`, this chunk's alone (each row
            // chunk is either whole or split, never both), inside `out`,
            // which is borrowed mutably for the whole run.
            let ys =
                unsafe { std::slice::from_raw_parts_mut(base.get().add(r0 * slots), k * slots) };
            f(r0, k, t0, t1, ys);
        } else {
            let nt = t1 - t0;
            with_tail_scratch(k * nt, |ys| {
                f(r0, k, t0, t1, ys);
                for (r, yr) in ys.chunks_exact(nt).enumerate() {
                    // SAFETY: row r0 + r, slots t0..t1: this (row chunk,
                    // token range) is claimed once, ranges are disjoint.
                    let o = unsafe {
                        std::slice::from_raw_parts_mut(base.get().add((r0 + r) * slots + t0), nt)
                    };
                    o.copy_from_slice(yr);
                }
            });
        }
    });
}

thread_local! {
    /// A split chunk's outputs before they're copied into place.
    static TAIL: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn with_tail_scratch<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    TAIL.with_borrow_mut(|s| {
        if s.len() < n {
            s.resize(n, 0.0);
        }
        f(&mut s[..n])
    })
}

/// `per_token` for steps that want several consecutive tokens at once (a
/// transpose out of a (row, slot) buffer reads a row's slots together):
/// `f(first, items)` over blocks of consecutive items (~2 blocks per thread).
fn per_block<T: Send>(data: &mut [T], floats: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if data.len() <= 1 || data.len() * floats <= INLINE_WORK {
        f(0, data);
        return;
    }
    let chunk = data.len().div_ceil(Pool::global().threads() * 2).max(1);
    pool::for_each_chunk_mut(data, chunk, |c, xs| f(c * chunk, xs));
}

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
/// y[r * slots + b] += bias[r] for the first `nb` slots (`add_assign`'s
/// op per element, in the (row, slot) layout of a matmul output).
fn bias_cols(y: &mut [f32], slots: usize, nb: usize, bias: &[f32]) {
    for (yr, &bv) in y.chunks_exact_mut(slots).zip(bias) {
        for v in &mut yr[..nb] {
            *v += bv;
        }
    }
}

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
    sinks: Option<&[f32]>,
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
        pool::for_each_chunk_mut_in(Kind::Decode, wts, np, |qh, s| {
            let qv = &q[qh * hd..(qh + 1) * hd];
            let sink = sinks.map(|v| v[qh]);
            score_softmax(qv, kc, kv_dim, qh / group * hd, scale, sink, s);
        });
    } else {
        pool::for_each_chunk_mut_in(Kind::Decode, sc, tc * group, |c, s| {
            let (kvh, t0) = (c / nt, c % nt * tc);
            let qg = &q[kvh * qn..(kvh + 1) * qn];
            score_tile(qg, kc, kv_dim, hd, kvh * hd, t0..np.min(t0 + tc), scale, s);
        });
        let sc = &*sc;
        pool::for_each_chunk_mut_in(Kind::Decode, wts, np, |qh, w| {
            let (kvh, g) = (qh / group, qh % group);
            let sink = sinks.map(|v| v[qh]);
            softmax_from(&sc[kvh * nt * tc * group + g..], group, sink, w);
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
    pool::for_each_chunk_mut_in(Kind::Decode, vparts, hmax * cw, |c, o| {
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
/// dot, `* scale`, softmax and V-sum order per head as `attend`. `sinks`:
/// the group's sink scores (gpt-oss), as `attend`'s.
#[allow(clippy::too_many_arguments)]
fn attend_group<T: KvElem>(
    qg: &[f32],
    kc: &[T],
    vc: &[T],
    kv_dim: usize,
    hd: usize,
    kvh: usize,
    np: usize,
    sinks: Option<&[f32]>,
    out: &mut [f32],
) {
    let group = qg.len() / hd;
    let scale = 1.0 / (hd as f32).sqrt();
    with_scores((np + 1) * group, |sc| {
        attend_causal(qg, kc, vc, kv_dim, hd, kvh, np, scale, sinks, sc, out)
    });
}

// ---------------------------------------------------------------------------
// Attention, multiversioned (see mv.rs) with the fused-multiply-add switch F:
// every `a * b + c` step below is `vmath::fma::<F>`, one rounding when the CPU
// has FMA (both variants), a multiply and an add when it doesn't. The `V3`
// variants use the hand-written AVX2 + FMA helpers (`*_v3`), the same
// operations in the same order as the scalar code with F.

multiversion! {
    /// Decode scores of one query head, then `softmax` (`softmax_sink`
    /// with a sink): s[t] = dot(qv, k_t) * scale with k_t = kc[t * kv_dim +
    /// k_off..][..qv.len()].
    fn score_softmax<V3, F, T: KvElem>(
        qv: &[f32],
        kc: &[T],
        kv_dim: usize,
        k_off: usize,
        scale: f32,
        sink: Option<f32>,
        s: &mut [f32],
    ) {
        let hd = qv.len();
        for (t, s) in s.iter_mut().enumerate() {
            let k0 = t * kv_dim + k_off;
            // SAFETY: V3 only in the v3 variant.
            *s = unsafe { dot::<V3, F, T>(qv, &kc[k0..k0 + hd]) } * scale;
        }
        // SAFETY: as above.
        unsafe { softmax_opt::<V3, F>(s, sink) };
    }
}

/// `vmath::softmax`, or `vmath::softmax_sink` with a sink.
///
/// # Safety
/// As `vmath::softmax`.
#[inline(always)]
unsafe fn softmax_opt<const V3: bool, const F: bool>(s: &mut [f32], sink: Option<f32>) {
    // SAFETY: our caller's contract.
    unsafe {
        match sink {
            None => vmath::softmax::<V3, F>(s),
            Some(k) => vmath::softmax_sink::<V3, F>(s, k),
        }
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
        sinks: Option<&[f32]>,
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
        for (g, s) in sc.chunks_exact_mut(np).enumerate() {
            unsafe { softmax_opt::<V3, F>(s, sinks.map(|v| v[g])) };
        }
        let vb = &vc[kvh * hd..];
        unsafe { vsum_rows::imp::<V3, F, T>(sc, np, np, group, vb, kv_dim, out, hd) };
    }
}

multiversion! {
    /// `softmax` of the scores src[t * stride], t in 0..dst.len(), written to
    /// dst: gathered, then `vmath::softmax` in place, so the same bits as
    /// `softmax` on a copy.
    fn softmax_from<V3, F>(src: &[f32], stride: usize, sink: Option<f32>, dst: &mut [f32]) {
        let n = dst.len();
        assert!(
            n == 0 || (n - 1) * stride < src.len(),
            "softmax_from: short src"
        );
        for (t, d) in dst.iter_mut().enumerate() {
            *d = src[t * stride];
        }
        // SAFETY: V3 only in the v3 variant.
        unsafe { softmax_opt::<V3, F>(dst, sink) };
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

multiversion! {
    /// `rmsnorm_into` on every head of v in place (heads of `hd` floats),
    /// same ops in the same order.
    fn rmsnorm_heads<V3>(v: &mut [f32], hd: usize, w: &[f32], eps: f32) {
        for head in v.chunks_mut(hd) {
            let mut ss = -0.0f32;
            for x in head.iter() {
                ss += x * x;
            }
            let ms = ss / hd as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for (x, g) in head.iter_mut().zip(w) {
                *x = *x * inv * g;
            }
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
    /// xs[j][i] += y[i * slots + b0 + j]: `add_col` for tokens b0.. at once,
    /// row outer (a row's slots are read together), same single add per element.
    fn add_cols<V3>(y: &[f32], slots: usize, b0: usize, xs: &mut [&mut [f32]]) {
        let n = xs.first().map_or(0, |x| x.len());
        for (i, yr) in y.chunks_exact(slots).take(n).enumerate() {
            for (x, &v) in xs.iter_mut().zip(&yr[b0..]) {
                x[i] += v;
            }
        }
    }
}

multiversion! {
    /// xs[j][i] = y[i * slots + b0 + j]: the columns of tokens b0.. of a
    /// row-major (row, token slot) buffer (`gather_col` for several tokens).
    fn gather_cols<V3>(y: &[f32], slots: usize, b0: usize, xs: &mut [&mut [f32]]) {
        let n = xs.first().map_or(0, |x| x.len());
        for (i, yr) in y.chunks_exact(slots).take(n).enumerate() {
            for (x, &v) in xs.iter_mut().zip(&yr[b0..]) {
                x[i] = v;
            }
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
#[path = "model/gemma3_tests.rs"]
pub(crate) mod gemma3_tests;

#[cfg(test)]
#[path = "model/moe_tests.rs"]
mod moe_tests;

#[cfg(test)]
#[path = "model/deepseek2_tests.rs"]
pub(crate) mod deepseek2_tests;

#[cfg(test)]
#[path = "model/gptoss_tests.rs"]
pub(crate) mod gptoss_tests;

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
        attend(
            q, kc, vc, kv_dim, hd, np, tile, threads, None, &mut buf, &mut out,
        );
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
        check_supported(&tiny_model(edit, edit_meta))
    }

    /// The file `tiny_meta` checks.
    fn tiny_model(
        edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>),
        edit_meta: impl FnOnce(&mut Vec<(&'static str, Val)>),
    ) -> Model {
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
        crate::gguf::build_for_test(&meta, &ts)
    }

    /// The copy-vs-map choice (`reads_matrices_in_place`): a copy only when
    /// some matrix the CPU reads per token isn't repacked.
    #[test]
    fn reads_in_place_when_a_matrix_isnt_repacked() {
        type Ts = Vec<(String, Vec<u64>, u32)>;
        let reads = |edit: &dyn Fn(&mut Ts), repack: bool| {
            let m = tiny_model(edit, |_| {});
            reads_in_place_with(&m, repack)
        };
        let set = |name: &'static str, dims: Vec<u64>, ty: u32| {
            move |ts: &mut Ts| {
                let t = ts.iter_mut().find(|t| t.0 == name).unwrap();
                (t.1, t.2) = (dims.clone(), ty);
            }
        };
        // All Q8_0, tied output from a repackable embedding (8 rows).
        assert!(!reads(&|_| {}, true));
        // AXON_REPACK=0: everything is read in place.
        assert!(reads(&|_| {}, false));
        // A 2-D F16 matrix in a block has no r8/r8q form.
        assert!(reads(&set("blk.1.ffn_up.weight", vec![64, 128], F16), true));
        // 12 rows: not whole groups of 8.
        assert!(reads(&set("blk.0.attn_k.weight", vec![64, 12], Q8_0), true));
        // Tied to an F16 embedding: the head reads it in place.
        let f16_embd = set("token_embd.weight", vec![64, 8], F16);
        assert!(reads(&f16_embd, true));
        // With its own repackable head the F16 embedding is one row a token.
        let own_head = |ts: &mut Ts| {
            f16_embd(ts);
            ts.push(("output.weight".into(), vec![64, 8], Q8_0));
        };
        assert!(!reads(&own_head, true));
    }

    /// A tied output head over a full-precision (F16 / F32) embedding, as
    /// F16 GGUFs and llama.cpp's --token-embedding-type f16 make: loads and
    /// runs with every other matrix repacked and released (on the anonymous
    /// copy, where reading a released tensor panics), with and without the
    /// repack. The fuzzer's panic here came from overlapping tensors, now
    /// refused by `gguf::parse`; real files of this shape (Qwen3-0.6B and
    /// Llama-3.2-1B with an F16 embedding) load on main too.
    #[test]
    fn tied_full_precision_embedding_runs() {
        for (ty, repack) in [(F16, true), (F16, false), (F32, true), (F32, false)] {
            // A fresh model each time: a repacked one's released tensors
            // can't be read again.
            let m = tiny_model(
                |ts| {
                    let t = ts.iter_mut().find(|t| t.0 == "token_embd.weight").unwrap();
                    t.2 = ty;
                },
                |_| {},
            );
            assert!(!m.file_backed());
            assert!(find_opt(&m, "output.weight").is_none());
            {
                let net = Qwen2::new_with(&m, repack);
                let mut c = net.new_cache(4);
                for p in 0..4 {
                    let l = net.forward(p as u32, p, &mut c);
                    assert_eq!(l.len(), 8);
                    assert!(l.iter().all(|x| x.is_finite()), "type {ty} repack {repack}");
                }
            }
        }
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
        // Q8_1: a real GGUF type the kernels don't run; 99: unknown.
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], 9),
            "Q8_1 (9) is not supported",
        );
        rejects(
            set("blk.1.ffn_down.weight", vec![128, 64], 99),
            "?? (99) is not supported",
        );
        // Zoo sweep: i-quants and repacked types were "?? (23)" with no way out.
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], 23),
            "type IQ4_XS (23) is not supported (supported: F32",
        );
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], 19),
            "type IQ1_S (19) is not supported",
        );
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], 19),
            "use a K-quant file (Q4_K_M, Q5_K_M, Q6_K) or Q8_0",
        );
        rejects(
            |ts: &mut Vec<(String, Vec<u64>, u32)>| {
                for (n, ty) in [("blk.0.attn_q.weight", 22), ("blk.1.attn_q.weight", 19)] {
                    ts.iter_mut().find(|t| t.0 == n).unwrap().2 = ty;
                }
            },
            "(the file has unsupported tensors: 1 IQ2_S, 1 IQ1_S); this i-quant",
        );
        let ok: Vec<&str> = SUPPORTED_TYPES
            .iter()
            .map(|&id| crate::gguf::qtype(id).name)
            .collect();
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], 31),
            &format!(
                "Q4_0_4_4 (31) is not supported (supported: {}); it is an ARM repacked layout \
                 that llama.cpp has dropped: use the Q4_0 file",
                ok.join(", ")
            ),
        );
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], 36),
            "IQ4_NL_4_4 (36) is not supported",
        );
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], 36),
            "dropped: use the IQ4_NL file",
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

    /// The tiny model as qwen3: q_dim (4 heads of 32 = 128) != dim (64), q/k
    /// norms, no biases.
    pub(crate) fn tiny3(
        edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>),
        edit_meta: impl FnOnce(&mut Vec<(&'static str, Val)>),
    ) -> std::io::Result<()> {
        tiny_meta(
            |ts| {
                ts.retain(|t| !t.0.ends_with(".bias"));
                for t in ts.iter_mut() {
                    if t.0.ends_with("attn_q.weight") {
                        t.1 = vec![64, 128];
                    } else if t.0.ends_with("attn_output.weight") {
                        t.1 = vec![128, 64];
                    }
                }
                for i in 0..2 {
                    for n in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                        ts.push((format!("blk.{i}.{n}"), vec![32], F32));
                    }
                }
                edit(ts);
            },
            |m| {
                for kv in m.iter_mut() {
                    if let Some(k) = kv.0.strip_prefix("qwen2.") {
                        kv.0 = format!("qwen3.{k}").leak();
                    }
                }
                m[0].1 = Val::Str("qwen3".into());
                m[3].1 = Val::U32(4);
                m.push(("qwen3.attention.key_length", Val::U32(32)));
                m.push(("qwen3.attention.value_length", Val::U32(32)));
                edit_meta(m);
            },
        )
    }

    /// A small random qwen3 for checking the forward pass against `qwen3_ref`:
    /// dim 128, 4 query heads and 2 kv heads of 64 (q_dim 256 != dim), ffn
    /// 256, vocab 64, 2 layers, tied output, eps 0.5 (so a wrong eps moves
    /// the q/k norms by ~40%: raw q and k have RMS ~0.7), q/k norm gains in
    /// [0.25, 4) (RoPE commutes with the norm but for the gains: wide gains
    /// make the order matter). Matrices of type `mt` (F32: the CPU's exact float rows;
    /// Q8_0 for the GPU), norms F32.
    pub(crate) fn qwen3_fixture(mt: u32) -> Model {
        dense_fixture("qwen3", mt).0
    }

    /// The tensors of a fixture as written to its GGUF: (name, dims, type,
    /// bytes), before `load` touches anything (llama's q/k row order).
    pub(crate) type RawTensors = Vec<(String, Vec<u64>, u32, Vec<u8>)>;

    /// A small random llama like `qwen3_fixture` (head_dim 64, 2 query heads
    /// per kv head) without q/k norms, with RoPE factors in [1, 8) per pair
    /// (`rope_freqs.weight`; base 10000, which turns more pairs than Llama
    /// 3's 500000 over a short test), and its tensors as
    /// written. The file's attn_q/attn_k rows are in the GGUF's order (the
    /// converter's permutation of random rows is random rows): `load`
    /// reorders them; `llama_ref` reads the raw ones with llama.cpp's
    /// adjacent-pair RoPE, so the two only agree if the reordering is right.
    pub(crate) fn llama_fixture(mt: u32) -> (Model, RawTensors) {
        dense_fixture("llama", mt)
    }

    /// `qwen3_fixture` / `llama_fixture` (arch "qwen3" or "llama"), or a
    /// qwen3moe (`MOE_FIXTURE` experts of ffn 256, top-2, an F32 router)
    /// that is the qwen3 fixture otherwise.
    pub(crate) fn dense_fixture(arch: &str, mt: u32) -> (Model, RawTensors) {
        let (dim, nh, nkv, hd, ffn, vocab, nl) = (128usize, 4, 2, 64, 256, 64, 2);
        let moe = arch == "qwen3moe";
        let (n_exp, k_exp) = MOE_FIXTURE;
        let (qd, kvd) = (nh * hd, nkv * hd);
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut uni = move |lo: f32, hi: f32| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            lo + (hi - lo) * ((st >> 40) as f32 / (1u64 << 24) as f32)
        };
        // (name, [cols, rows], f32 values)
        let mut ts: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        let mut mat = |name: String, cols: usize, rows: usize, rms: f32| {
            let a = rms * (3.0 / cols as f32).sqrt();
            let v = (0..cols * rows).map(|_| uni(-a, a)).collect();
            (name, vec![cols as u64, rows as u64], v)
        };
        let mut t = vec![mat("token_embd.weight".into(), dim, vocab, 0.0)];
        for i in 0..nl {
            let n = |s: &str| format!("blk.{i}.{s}");
            // llama has no q/k norm to scale them up: larger q and k
            // instead, so attention is far from uniform and RoPE matters.
            let qk = if arch == "llama" { 3.0 } else { 0.7 };
            t.push(mat(n("attn_q.weight"), dim, qd, qk));
            t.push(mat(n("attn_k.weight"), dim, kvd, qk));
            t.push(mat(n("attn_v.weight"), dim, kvd, 1.0));
            t.push(mat(n("attn_output.weight"), qd, dim, 1.0));
            if moe {
                // [cols, rows, n_exp]: n_exp matrices one after another.
                let mut exps = |s: &str, cols: usize, rows: usize| {
                    let (name, mut d, v) = mat(n(s), cols, rows * n_exp, 1.0);
                    d[1] = rows as u64;
                    d.push(n_exp as u64);
                    (name, d, v)
                };
                t.push(exps("ffn_gate_exps.weight", dim, ffn));
                t.push(exps("ffn_up_exps.weight", dim, ffn));
                t.push(exps("ffn_down_exps.weight", ffn, dim));
                // Router logits of a few units: a clear but not saturated
                // preference.
                t.push(mat(n("ffn_gate_inp.weight"), dim, n_exp, 3.0));
            } else {
                t.push(mat(n("ffn_gate.weight"), dim, ffn, 1.0));
                t.push(mat(n("ffn_up.weight"), dim, ffn, 1.0));
                t.push(mat(n("ffn_down.weight"), ffn, dim, 1.0));
            }
        }
        ts.extend(t);
        // The embedding (and tied head): uniform in [-1, 1).
        ts[0].2 = (0..dim * vocab).map(|_| uni(-1.0, 1.0)).collect();
        let mut vecs = |name: String, n: usize, lo: f32, hi: f32| {
            ts.push((name, vec![n as u64], (0..n).map(|_| uni(lo, hi)).collect()))
        };
        vecs("output_norm.weight".into(), dim, 0.8, 1.2);
        for i in 0..nl {
            let n = |s: &str| format!("blk.{i}.{s}");
            vecs(n("attn_norm.weight"), dim, 0.8, 1.2);
            vecs(n("ffn_norm.weight"), dim, 0.8, 1.2);
            if arch == "qwen3" || moe {
                vecs(n("attn_q_norm.weight"), hd, 0.25, 4.0);
                vecs(n("attn_k_norm.weight"), hd, 0.25, 4.0);
            }
        }
        if arch == "llama" {
            vecs("rope_freqs.weight".into(), hd / 2, 1.0, 8.0);
        }
        let ty = |t: &(String, Vec<u64>, Vec<f32>)| {
            if t.1.len() >= 2 && !t.0.ends_with("ffn_gate_inp.weight") {
                mt
            } else {
                F32
            }
        };
        let bytes = |t: &(String, Vec<u64>, Vec<f32>)| -> Vec<u8> {
            match ty(t) {
                F32 => t.2.iter().flat_map(|v| v.to_le_bytes()).collect(),
                Q8_0 => t
                    .2
                    .chunks(32)
                    .flat_map(|b| {
                        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                        let d = crate::quant::f16_to_f32(crate::quant::f32_to_f16(amax / 127.0));
                        let mut out = crate::quant::f32_to_f16(d).to_le_bytes().to_vec();
                        out.extend(b.iter().map(|&v| {
                            (if d == 0.0 { 0.0 } else { (v / d).round() }).clamp(-127.0, 127.0)
                                as i8 as u8
                        }));
                        out
                    })
                    .collect(),
                other => unimplemented!("dense_fixture: type {other}"),
            }
        };
        let k = |s: &str| -> &'static str { format!("{arch}.{s}").leak() };
        let mut meta = vec![
            ("general.architecture", Val::Str(arch.into())),
            (k("block_count"), Val::U32(nl as u32)),
            (k("embedding_length"), Val::U32(dim as u32)),
            (k("feed_forward_length"), Val::U32(ffn as u32)),
            (k("attention.head_count"), Val::U32(nh as u32)),
            (k("attention.head_count_kv"), Val::U32(nkv as u32)),
            (k("attention.key_length"), Val::U32(hd as u32)),
            (k("attention.value_length"), Val::U32(hd as u32)),
            (k("attention.layer_norm_rms_epsilon"), Val::F32(0.5)),
            (k("rope.freq_base"), Val::F32(10000.0)),
            (k("context_length"), Val::U32(64)),
            (
                "tokenizer.ggml.tokens",
                Val::Arr((0..vocab).map(|i| Val::Str(format!("t{i}"))).collect()),
            ),
        ];
        if moe {
            meta.push((k("expert_count"), Val::U32(n_exp as u32)));
            meta.push((k("expert_used_count"), Val::U32(k_exp as u32)));
            meta.push((k("expert_feed_forward_length"), Val::U32(ffn as u32)));
        }
        let list: Vec<(&str, &[u64], u32)> =
            ts.iter().map(|t| (t.0.as_str(), &t.1[..], ty(t))).collect();
        let data: Vec<Vec<u8>> = ts.iter().map(bytes).collect();
        let m = crate::gguf::build_for_test_with(&meta, &list, &data);
        let raw = ts
            .iter()
            .zip(data)
            .map(|(t, d)| (t.0.clone(), t.1.clone(), ty(t), d))
            .collect();
        (m, raw)
    }

    /// The qwen3moe fixture's (experts, experts used per token).
    pub(crate) const MOE_FIXTURE: (usize, usize) = (4, 2);

    /// Ways to get qwen3's q/k norm wrong, for showing `qwen3_ref` (and so
    /// the tests built on it) tells them apart from the right one.
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub(crate) enum NormBug {
        None,
        /// No q/k norm (qwen2's graph).
        Dropped,
        /// After RoPE instead of before.
        AfterRope,
        /// eps 1e-6 instead of the GGUF's.
        Eps,
        /// The gains' halves swapped (g[j + hd/2] on dim j).
        GainHalves,
    }

    /// The logits after each of `ids`, by a plain f64 forward pass written
    /// from the Qwen3 graph (llama.cpp's LLM_ARCH_QWEN3, HF's modeling_qwen3),
    /// none of the model's code: embed, then per layer rmsnorm, q k v,
    /// per-head RMSNorm of q and k (the GGUF's eps), NEOX RoPE, causal
    /// softmax attention (GQA), wo, rmsnorm, SwiGLU; final norm, head (the
    /// output matrix or the tied embedding). Weights are the file's,
    /// dequantized; the one rounding mirrored is the KV cache's (f16 unless
    /// AXON_KV=f32), stored after the rotation.
    pub(crate) fn qwen3_ref(m: &Model, ids: &[u32], bug: NormBug) -> Vec<Vec<f32>> {
        dense_ref(m, None, ids, bug, RopeBug::None)
    }

    /// Ways to get llama's RoPE wrong, for showing `llama_ref` tells them
    /// apart from the right one.
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub(crate) enum RopeBug {
        None,
        /// The file's q/k rows taken as they are with rotate-half RoPE (no
        /// reordering at load).
        Unpermuted,
        /// Llama 3's frequency factors ignored.
        NoFactors,
    }

    /// The logits after each of `ids`, by a plain f64 forward pass written
    /// from llama.cpp's LLM_ARCH_LLAMA graph on the GGUF's own tensors
    /// (`raw`, as `llama_fixture` wrote them, not the loaded model's): no
    /// biases or q/k norms, llama.cpp's "normal" RoPE on adjacent pairs
    /// (2i, 2i + 1) with frequency base^(-2i/hd) / rope_freqs[i]. Otherwise
    /// as `qwen3_ref` (f16 cache rounding mirrored).
    pub(crate) fn llama_ref(
        m: &Model,
        raw: &RawTensors,
        ids: &[u32],
        bug: RopeBug,
    ) -> Vec<Vec<f32>> {
        dense_ref(m, Some(raw), ids, NormBug::Dropped, bug)
    }

    /// `qwen3_ref` (raw None: the loaded model's tensors, NEOX RoPE, q/k
    /// norms) or `llama_ref` (raw: the file's tensors, adjacent-pair RoPE
    /// with the file's frequency factors, no q/k norm: `bug` Dropped).
    pub(crate) fn dense_ref(
        m: &Model,
        raw: Option<&RawTensors>,
        ids: &[u32],
        bug: NormBug,
        rbug: RopeBug,
    ) -> Vec<Vec<f32>> {
        let a = &m.arch;
        let num = |k: &str| meta_num(m, &format!("{a}.{k}")).unwrap();
        let (dim, nh, nkv) = (
            num("embedding_length") as usize,
            num("attention.head_count") as usize,
            num("attention.head_count_kv") as usize,
        );
        let hd = num("attention.key_length") as usize;
        let eps = num("attention.layer_norm_rms_epsilon");
        let base = meta_num(m, &format!("{a}.rope.freq_base")).unwrap_or(10000.0);
        let nl = num("block_count") as usize;
        let w = |name: &str| -> (Vec<f64>, usize) {
            let (qt, dims, bytes): (u32, &[u64], &[u8]) = match raw {
                Some(r) => {
                    let t = r.iter().find(|t| t.0 == name).unwrap();
                    (t.2, &t.1, &t.3)
                }
                None => {
                    let t = m.tensors.iter().find(|t| t.name == name).unwrap();
                    (t.qtype.id, &t.dims, m.tensor_data(t))
                }
            };
            let mut v = vec![0f32; dims.iter().product::<u64>() as usize];
            dequant(qt, bytes, &mut v);
            (v.into_iter().map(f64::from).collect(), dims[0] as usize)
        };
        // llama: the pair (2i, 2i + 1) turns at base^(-2i/hd) / factor[i].
        let llama = raw.is_some();
        let factors: Vec<f64> = match raw {
            Some(_) if rbug != RopeBug::NoFactors => w("rope_freqs.weight").0,
            _ => vec![1.0; hd / 2],
        };
        let mv = |(wv, cols): &(Vec<f64>, usize), x: &[f64]| -> Vec<f64> {
            wv.chunks(*cols)
                .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let rms = |x: &[f64], g: &[f64], eps: f64| -> Vec<f64> {
            let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
            x.iter().zip(g).map(|(v, g)| v * inv * g).collect()
        };
        // YaRN as HF transformers' _compute_yarn_parameters (truncate):
        // inv_freq = interp (1 - extrap) + extrap_freq extrap, extrap =
        // 1 - linear ramp over [low, high]; cos and sin times the
        // attention factor.
        let yarn = crate::rope::yarn(m);
        let inv_freq = |i: usize, f: f64| -> f64 {
            let pf = base.powf(2.0 * i as f64 / hd as f64) * f;
            let Some(y) = yarn else { return 1.0 / pf };
            let corr = |rot: f64| {
                (hd as f64 * (y.orig / (rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln())
            };
            let low = corr(y.beta_fast).floor().max(0.0);
            let mut high = corr(y.beta_slow).ceil().min(hd as f64 - 1.0);
            if high == low {
                high += 0.001;
            }
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let extrap = 1.0 - ramp;
            (1.0 / (y.factor * pf)) * (1.0 - extrap) + (1.0 / pf) * extrap
        };
        let att = yarn.map_or(1.0, |y| y.mscale as f64);
        let rope = |v: &mut [f64], pos: usize| {
            for head in v.chunks_mut(hd) {
                for (i, f) in factors.iter().enumerate() {
                    let th = pos as f64 * inv_freq(i, *f);
                    let (s, c) = (th.sin() * att, th.cos() * att);
                    let (a, b) = if llama && rbug != RopeBug::Unpermuted {
                        (2 * i, 2 * i + 1)
                    } else {
                        (i, i + hd / 2)
                    };
                    let (x0, x1) = (head[a], head[b]);
                    head[a] = x0 * c - x1 * s;
                    head[b] = x0 * s + x1 * c;
                }
            }
        };
        let head_norm = |v: &mut [f64], g: &[f64]| {
            let g: Vec<f64> = if bug == NormBug::GainHalves {
                g[hd / 2..].iter().chain(&g[..hd / 2]).copied().collect()
            } else {
                g.to_vec()
            };
            let e = if bug == NormBug::Eps { 1e-6 } else { eps };
            for head in v.chunks_mut(hd) {
                let n = rms(head, &g, e);
                head.copy_from_slice(&n);
            }
        };
        let cache = |x: f64| -> f64 {
            if crate::kv::use_f32() {
                x as f32 as f64
            } else {
                Half::from_f32(x as f32).to_f32() as f64
            }
        };
        let (embd, _) = w("token_embd.weight");
        let out = if m.tensors.iter().any(|t| t.name == "output.weight") {
            w("output.weight")
        } else {
            (embd.clone(), dim)
        };
        // qwen3moe (llama.cpp's build_moe_ffn: softmax gating, top k,
        // weights renormalized, no shared expert): its experts, top k.
        let moe = m.arch == "qwen3moe";
        let (n_exp, k_exp) = match moe {
            true => (
                num("expert_count") as usize,
                num("expert_used_count") as usize,
            ),
            false => (0, 0),
        };
        let lw: Vec<_> = (0..nl)
            .map(|i| {
                let n = |s: &str| w(&format!("blk.{i}.{s}"));
                [
                    n("attn_norm.weight"),
                    n("attn_q.weight"),
                    n("attn_k.weight"),
                    n("attn_v.weight"),
                    if llama {
                        (Vec::new(), 0)
                    } else {
                        n("attn_q_norm.weight")
                    },
                    if llama {
                        (Vec::new(), 0)
                    } else {
                        n("attn_k_norm.weight")
                    },
                    n("attn_output.weight"),
                    n("ffn_norm.weight"),
                    n(if moe {
                        "ffn_gate_exps.weight"
                    } else {
                        "ffn_gate.weight"
                    }),
                    n(if moe {
                        "ffn_up_exps.weight"
                    } else {
                        "ffn_up.weight"
                    }),
                    n(if moe {
                        "ffn_down_exps.weight"
                    } else {
                        "ffn_down.weight"
                    }),
                    if moe {
                        n("ffn_gate_inp.weight")
                    } else {
                        (Vec::new(), 0)
                    },
                ]
            })
            .collect();
        let (out_norm, _) = w("output_norm.weight");
        // Per layer, the cached (rounded) keys and values of every position.
        let mut kc: Vec<Vec<Vec<f64>>> = vec![Vec::new(); nl];
        let mut vc: Vec<Vec<Vec<f64>>> = vec![Vec::new(); nl];
        let mut logits = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            let mut x: Vec<f64> = embd[id as usize * dim..(id as usize + 1) * dim].to_vec();
            for (l, [an, wq, wk, wv, qn, kn, wo, fnm, wg, wu, wd, wr]) in lw.iter().enumerate() {
                let h = rms(&x, &an.0, eps);
                let (mut q, mut k, v) = (mv(wq, &h), mv(wk, &h), mv(wv, &h));
                match bug {
                    NormBug::Dropped => {}
                    NormBug::AfterRope => {
                        rope(&mut q, pos);
                        rope(&mut k, pos);
                    }
                    _ => {
                        head_norm(&mut q, &qn.0);
                        head_norm(&mut k, &kn.0);
                    }
                }
                if bug == NormBug::AfterRope {
                    head_norm(&mut q, &qn.0);
                    head_norm(&mut k, &kn.0);
                } else {
                    rope(&mut q, pos);
                    rope(&mut k, pos);
                }
                kc[l].push(k.iter().map(|&x| cache(x)).collect());
                vc[l].push(v.iter().map(|&x| cache(x)).collect());
                let mut att = vec![0f64; nh * hd];
                for qh in 0..nh {
                    let kh = qh / (nh / nkv);
                    let qv = &q[qh * hd..(qh + 1) * hd];
                    let sc: Vec<f64> = kc[l]
                        .iter()
                        .map(|kr| {
                            qv.iter()
                                .zip(&kr[kh * hd..(kh + 1) * hd])
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                                / (hd as f64).sqrt()
                        })
                        .collect();
                    let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for (t, p) in e.iter().enumerate() {
                        for d in 0..hd {
                            att[qh * hd + d] += p / z * vc[l][t][kh * hd + d];
                        }
                    }
                }
                for (x, o) in x.iter_mut().zip(mv(wo, &att)) {
                    *x += o;
                }
                let h = rms(&x, &fnm.0, eps);
                let ffn =
                    |wg: &(Vec<f64>, usize), wu: &(Vec<f64>, usize), wd: &(Vec<f64>, usize)| {
                        let (g, u) = (mv(wg, &h), mv(wu, &h));
                        let act: Vec<f64> = g
                            .iter()
                            .zip(&u)
                            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                            .collect();
                        mv(wd, &act)
                    };
                if !moe {
                    for (x, d) in x.iter_mut().zip(ffn(wg, wu, wd)) {
                        *x += d;
                    }
                    continue;
                }
                let lg = mv(wr, &h);
                let mx = lg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = lg.iter().map(|v| (v - mx).exp()).sum();
                let p: Vec<f64> = lg.iter().map(|v| (v - mx).exp() / z).collect();
                let mut order: Vec<usize> = (0..n_exp).collect();
                order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
                let top = &order[..k_exp];
                let s: f64 = top.iter().map(|&e| p[e]).sum();
                let part = |t: &(Vec<f64>, usize), e: usize| {
                    let per = t.0.len() / n_exp;
                    (t.0[e * per..(e + 1) * per].to_vec(), t.1)
                };
                for &e in top {
                    let d = ffn(&part(wg, e), &part(wu, e), &part(wd, e));
                    for (x, d) in x.iter_mut().zip(d) {
                        *x += p[e] / s * d;
                    }
                }
            }
            let h = rms(&x, &out_norm, eps);
            logits.push(mv(&out, &h).into_iter().map(|v| v as f32).collect());
        }
        logits
    }

    /// Largest |a - b| over two sets of logit rows.
    pub(crate) fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// Bound on |dlogit| vs `qwen3_ref` for 8-bit activations (the Q8_0
    /// fixture, CPU and GPU): measured 0.175 (CPU) and ~0.2 (GPU, f16 and q8
    /// caches); every `NormBug` moves the logits by >= 3.3 there.
    pub(crate) const QWEN3_Q8_TOL: f32 = 0.5;

    /// `QWEN3_Q8_TOL` for the llama fixture on the GPU, whose larger q and
    /// k (no norm to scale them) make 8-bit activations and a q8 cache cost
    /// more: measured GPU f16 cache 0.44, q8 cache 0.54; each `RopeBug`
    /// moves the logits by >= 10.6 there.
    pub(crate) const LLAMA_Q8_TOL: f32 = 1.0;
    /// The CPU's bound there (8-bit activations, f16 cache): measured 0.38.
    pub(crate) const LLAMA_Q8_TOL_CPU: f32 = 0.6;

    /// The CPU's qwen3 (decode and batched prefill, repacked or not) against
    /// the independent `qwen3_ref`. On the F32 fixture the CPU's only
    /// rounding besides f32 arithmetic is the f16 KV cache, which the
    /// reference mirrors (measured max |dlogit| 4.5e-4: f32 sums, and a key
    /// now and then rounding to the other neighbouring half); on the Q8_0
    /// one its 8-bit activations (QWEN3_Q8_TOL). Each `NormBug` moves the
    /// reference's logits by 3.3 to 10 on either (checked), so a norm bug in
    /// the model's code, shared by every backend or not, fails here (and in
    /// cuda's gpu_qwen3_matches_reference_forward).
    #[test]
    fn qwen3_matches_reference_forward() {
        const TOL: f32 = 2e-3;
        let m = qwen3_fixture(F32);
        let ids = [3u32, 17, 42, 9];
        let want = qwen3_ref(&m, &ids, NormBug::None);
        for bug in [
            NormBug::Dropped,
            NormBug::AfterRope,
            NormBug::Eps,
            NormBug::GainHalves,
        ] {
            let d = max_diff(&qwen3_ref(&m, &ids, bug), &want);
            eprintln!("reference with {bug:?}: max |dlogit| {d:.4}");
            assert!(d > 100.0 * TOL, "{bug:?} hardly changes the logits ({d})");
        }
        // The Q8_0 fixture (the GPU test's): the CPU's 8-bit activations
        // against the bugs' size there.
        {
            let m = qwen3_fixture(Q8_0);
            let want = qwen3_ref(&m, &ids, NormBug::None);
            let mn = qwen3_fixture(Q8_0); // the net's (repacking releases)
            let net = Qwen2::new(&mn);
            let mut c = net.new_cache(ids.len());
            let l: Vec<Vec<f32>> = (0..ids.len())
                .map(|p| net.forward(ids[p], p, &mut c))
                .collect();
            let d = max_diff(&l, &want);
            eprintln!("Q8_0 fixture: cpu vs reference {d:.4}");
            assert!(d <= QWEN3_Q8_TOL, "Q8_0 fixture: cpu vs reference {d}");
            for bug in [
                NormBug::Dropped,
                NormBug::AfterRope,
                NormBug::Eps,
                NormBug::GainHalves,
            ] {
                let d = max_diff(&qwen3_ref(&m, &ids, bug), &want);
                eprintln!("Q8_0 fixture: reference with {bug:?}: {d:.4}");
                assert!(d > 5.0 * QWEN3_Q8_TOL, "Q8_0 fixture: {bug:?} only {d}");
            }
        }
        for repack in [true, false] {
            // A fresh fixture each: repacking releases what it repacks.
            let m = qwen3_fixture(F32);
            let net = Qwen2::new_with(&m, repack);
            let mut c = net.new_cache(ids.len());
            let dec: Vec<Vec<f32>> = (0..ids.len())
                .map(|p| net.forward(ids[p], p, &mut c))
                .collect();
            let mut c = net.new_cache(ids.len());
            net.feed_batch(&ids[..3], 0, &mut c);
            let pre = net.forward(ids[3], 3, &mut c);
            let (d, dp) = (max_diff(&dec, &want), max_diff(&[pre], &want[3..]));
            eprintln!(
                "cpu (repack {repack}) vs reference: max |dlogit| {d:.2e}, after prefill {dp:.2e}"
            );
            assert!(d <= TOL && dp <= TOL, "cpu vs reference: {d} / {dp}");
        }
    }

    /// YaRN (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 16`, so a
    /// short test crosses the ramp: the first few pairs blend, the rest are
    /// divided by 4) on the CPU against the reference's own YaRN (HF's formulation),
    /// decode bit-identical to batched prefill, and the scaling moving the
    /// logits far beyond the tolerance (so a dropped mscale or ramp fails).
    #[test]
    fn yarn_matches_reference_forward() {
        const TOL: f32 = 2e-3;
        let ids = [3u32, 17, 42, 9, 5, 11, 60, 2, 33, 21, 8, 40];
        let n = ids.len();
        let args = crate::rope::RopeArgs {
            scaling: Some("yarn".into()),
            factor: Some(4.0),
            orig_ctx: Some(16),
            ..Default::default()
        };
        let plain = qwen3_ref(&qwen3_fixture(F32), &ids, NormBug::None);
        let mut m = qwen3_fixture(F32);
        args.apply(&mut m).unwrap();
        check_supported(&m).unwrap();
        assert_eq!(m.ctx_len(), 64);
        let y = crate::rope::yarn(&m).unwrap();
        let base = meta_num(&m, "qwen3.rope.freq_base").unwrap_or(10000.0);
        let (low, high) = y.corr_dims(base, 64);
        eprintln!("yarn {y:?}: blend pairs {low}..{high}");
        assert!(high > low && high < 31.0, "{low}..{high}");
        let want = qwen3_ref(&m, &ids, NormBug::None);
        let moved = max_diff(&plain, &want);
        eprintln!("yarn vs plain reference: max |dlogit| {moved:.3}");
        assert!(
            moved > 100.0 * TOL,
            "yarn hardly changes the logits ({moved})"
        );
        let net = Qwen2::new_with(&m, false);
        assert_eq!(net.rope_m, y.mscale);
        let mut c = net.new_cache(n);
        let dec: Vec<Vec<f32>> = (0..n).map(|p| net.forward(ids[p], p, &mut c)).collect();
        let mut c = net.new_cache(n);
        net.feed_batch(&ids[..n - 1], 0, &mut c);
        let pre = net.forward(ids[n - 1], n - 1, &mut c);
        let d = max_diff(&dec, &want);
        eprintln!("cpu yarn vs reference: max |dlogit| {d:.2e}");
        assert!(d <= TOL, "cpu yarn vs reference: {d}");
        assert_eq!(
            pre.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            dec[n - 1].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "yarn: decode != batched prefill"
        );
        // The mscale alone moves the logits far beyond the tolerance too (a
        // table builder ignoring it would fail above).
        let mut m1 = qwen3_fixture(F32);
        let no_m = crate::rope::RopeArgs {
            mscale: Some(1.0),
            ..args.clone()
        };
        no_m.apply(&mut m1).unwrap();
        let d1 = max_diff(&qwen3_ref(&m1, &ids, NormBug::None), &want);
        eprintln!("yarn without mscale vs yarn reference: max |dlogit| {d1:.3}");
        assert!(d1 > 100.0 * TOL, "mscale hardly changes the logits ({d1})");
        // Off: --rope-scaling none on a plain model changes nothing; on a
        // YaRN model (here the flags' keys) it restores the original context.
        let none = crate::rope::RopeArgs {
            scaling: Some("none".into()),
            ..Default::default()
        };
        none.apply(&mut m).unwrap();
        assert!(crate::rope::yarn(&m).is_none());
        assert_eq!(m.ctx_len(), 16);
        check_supported(&m).unwrap();
        let mut m = qwen3_fixture(F32);
        none.apply(&mut m).unwrap();
        assert!(crate::rope::yarn(&m).is_none());
        assert_eq!(m.ctx_len(), 64);
        // No --rope-scale: ceil(--ctx / orig) (spec L2), else an error.
        let mut m = qwen3_fixture(F32);
        let by_ctx = crate::rope::RopeArgs {
            scaling: Some("yarn".into()),
            orig_ctx: Some(16),
            ctx: Some(40),
            ..Default::default()
        };
        by_ctx.apply(&mut m).unwrap();
        assert_eq!(crate::rope::yarn(&m).unwrap().factor, 3.0);
        assert_eq!(m.ctx_len(), 48);
        let mut m = qwen3_fixture(F32);
        let short = crate::rope::RopeArgs {
            ctx: Some(16),
            ..by_ctx.clone()
        };
        assert!(short.apply(&mut m).is_err());
        // yarn with a linear scale too: refused.
        let mut m = qwen3_fixture(F32);
        args.apply(&mut m).unwrap();
        m.metadata
            .push(("qwen3.rope.scale_linear".into(), Val::F32(2.0)));
        assert!(check_supported(&m).is_err());
        // Flags without the type, and yarn on gemma3, are refused.
        let mut m = qwen3_fixture(F32);
        let bare = crate::rope::RopeArgs {
            factor: Some(4.0),
            ..Default::default()
        };
        assert!(bare.apply(&mut m).is_err());
    }

    /// The tiny model as llama: no biases, llama.* keys, `edit` /
    /// `edit_meta` after.
    fn tiny_llama(
        edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>),
        edit_meta: impl FnOnce(&mut Vec<(&'static str, Val)>),
    ) -> std::io::Result<()> {
        tiny_meta(
            |ts| {
                ts.retain(|t| !t.0.ends_with(".bias"));
                ts.push(("rope_freqs.weight".into(), vec![16], F32));
                edit(ts);
            },
            |m| {
                for kv in m.iter_mut() {
                    if let Some(k) = kv.0.strip_prefix("qwen2.") {
                        kv.0 = format!("llama.{k}").leak();
                    }
                }
                m[0].1 = Val::Str("llama".into());
                m.push(("tokenizer.ggml.model", Val::Str("gpt2".into())));
                m.push(("tokenizer.ggml.pre", Val::Str("llama-bpe".into())));
                edit_meta(m);
            },
        )
    }

    #[test]
    fn check_supported_llama() {
        // rope_freqs is zero-filled in the tiny model: factors must be > 0.
        let rej = |e: std::io::Result<()>, want: &str| {
            let e = e.expect_err("malformed llama accepted");
            assert!(e.to_string().contains(want), "{e}");
        };
        rej(tiny_llama(|_| {}, |_| {}), "factors must be positive");
        tiny_llama(|ts| ts.retain(|t| t.0 != "rope_freqs.weight"), |_| {}).expect("tiny llama");
        let no_freqs =
            |ts: &mut Vec<(String, Vec<u64>, u32)>| ts.retain(|t| t.0 != "rope_freqs.weight");
        rej(
            tiny_llama(
                |ts| {
                    no_freqs(ts);
                    ts.push(("rope_freqs.weight".into(), vec![32], F32));
                },
                |_| {},
            ),
            "rope_freqs.weight",
        );
        rej(
            tiny_llama(
                |ts| {
                    no_freqs(ts);
                    ts.push(("blk.0.attn_q_norm.weight".into(), vec![32], F32));
                },
                |_| {},
            ),
            "no q/k norm",
        );
        rej(
            tiny_llama(
                |ts| {
                    no_freqs(ts);
                    ts.push(("blk.1.ffn_down.bias".into(), vec![64], F32));
                },
                |_| {},
            ),
            "ffn_down.bias is not supported",
        );
        for (k, v, want) in [
            ("llama.rope.dimension_count", Val::U32(16), "partial RoPE"),
            ("llama.expert_count", Val::U32(8), "mixture-of-experts"),
            ("llama.attention.scale", Val::F32(0.5), "attention.scale"),
            (
                "llama.rope.scaling.type",
                Val::Str("linear".into()),
                "RoPE scaling",
            ),
            (
                "llama.rope.scaling.type",
                Val::Str("longrope".into()),
                "RoPE scaling",
            ),
            ("llama.rope.scaling.factor", Val::F32(4.0), "factor 4"),
            ("llama.rope.scale_linear", Val::F32(2.0), "factor 2"),
            (
                "llama.rope.scaling.attn_factor",
                Val::F32(0.5),
                "attention factor",
            ),
            (
                "tokenizer.ggml.pre",
                Val::Str("deepseek-llm".into()),
                "pre-tokenizer",
            ),
            (
                "tokenizer.ggml.model",
                Val::Str("t5".into()),
                "tokenizer model",
            ),
        ] {
            let e = tiny_llama(no_freqs, |m| {
                m.retain(|kv| kv.0 != k);
                m.push((k, v));
                if k == "llama.rope.scaling.type" {
                    m.push(("llama.rope.scaling.factor", Val::F32(8.0)));
                }
            });
            rej(e, want);
        }
        // Accepted: the head size as rope.dimension_count, scaling "none",
        // or linear / yarn with factor 1.
        tiny_llama(no_freqs, |m| {
            m.push(("llama.rope.dimension_count", Val::U32(32)));
            m.push(("llama.rope.scaling.type", Val::Str("none".into())));
        })
        .expect("tiny llama with rope keys");
        // A SentencePiece vocabulary (Mistral, Llama 2, TinyLlama) is supported.
        tiny_llama(no_freqs, |m| {
            m.retain(|kv| kv.0 != "tokenizer.ggml.model" && kv.0 != "tokenizer.ggml.pre");
            m.push(("tokenizer.ggml.model", Val::Str("llama".into())));
        })
        .expect("SentencePiece vocabulary");
        for t in ["linear", "yarn"] {
            tiny_llama(no_freqs, |m| {
                m.push(("llama.rope.scaling.type", Val::Str(t.into())));
                m.push(("llama.rope.scaling.factor", Val::F32(1.0)));
            })
            .expect("factor-1 scaling");
        }
    }

    /// The CPU's llama (decode and batched prefill, repacked or not) against
    /// the independent `llama_ref` on the GGUF's own tensors: proves the
    /// q/k row reordering at load plus NEOX RoPE equals llama.cpp's
    /// adjacent-pair RoPE on the file's rows, and that the RoPE factors are
    /// applied. Each `RopeBug` moves the reference far beyond the tolerance.
    #[test]
    fn llama_matches_reference_forward() {
        const TOL: f32 = 2e-3;
        let ids = [3u32, 17, 42, 9, 5, 61, 33, 2, 28, 50, 11, 7];
        let n = ids.len();
        let (m, raw) = llama_fixture(F32);
        let want = llama_ref(&m, &raw, &ids, RopeBug::None);
        for bug in [RopeBug::Unpermuted, RopeBug::NoFactors] {
            let d = max_diff(&llama_ref(&m, &raw, &ids, bug), &want);
            eprintln!("llama reference with {bug:?}: max |dlogit| {d:.4}");
            assert!(d > 100.0 * TOL, "{bug:?} hardly changes the logits ({d})");
        }
        for repack in [true, false] {
            let (m, _) = llama_fixture(F32);
            let net = Qwen2::new_with(&m, repack);
            let mut c = net.new_cache(n);
            let dec: Vec<Vec<f32>> = (0..n).map(|p| net.forward(ids[p], p, &mut c)).collect();
            let mut c = net.new_cache(n);
            net.feed_batch(&ids[..n - 1], 0, &mut c);
            let pre = net.forward(ids[n - 1], n - 1, &mut c);
            let (d, dp) = (max_diff(&dec, &want), max_diff(&[pre], &want[n - 1..]));
            eprintln!("llama cpu (repack {repack}) vs reference: {d:.2e}, after prefill {dp:.2e}");
            assert!(d <= TOL && dp <= TOL, "cpu vs reference: {d} / {dp}");
        }
        let (m, raw) = llama_fixture(Q8_0);
        let want = llama_ref(&m, &raw, &ids, RopeBug::None);
        let net = Qwen2::new(&m);
        let mut c = net.new_cache(n);
        let l: Vec<Vec<f32>> = (0..n).map(|p| net.forward(ids[p], p, &mut c)).collect();
        let d = max_diff(&l, &want);
        eprintln!("llama Q8_0 fixture: cpu vs reference {d:.4}");
        assert!(d <= LLAMA_Q8_TOL_CPU, "Q8_0 fixture: cpu vs reference {d}");
        for bug in [RopeBug::Unpermuted, RopeBug::NoFactors] {
            let d = max_diff(&llama_ref(&m, &raw, &ids, bug), &want);
            eprintln!("llama Q8_0 fixture: reference with {bug:?}: {d:.4}");
            assert!(d > 5.0 * LLAMA_Q8_TOL, "Q8_0 fixture: {bug:?} only {d}");
        }
    }

    /// The real llama file loads the same copied or mapped (both reorder
    /// attn_q/attn_k once: the copy in place, the mapping into side copies),
    /// other tensors untouched, and the reordered rows are the file's rows
    /// moved.
    #[test]
    fn llama_load_paths_agree() {
        let Some(m0) = crate::testutil::llama_model() else {
            return;
        };
        let path = crate::testutil::LLAMA_MODEL;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        let path = path.to_str().unwrap();
        let file = std::fs::read(path).unwrap();
        let (a, b) = (
            crate::gguf::load_with(path, Some(true)).unwrap(),
            crate::gguf::load_with(path, Some(false)).unwrap(),
        );
        assert!(b.file_backed() && !a.file_backed());
        let hd = 64;
        for t in &m0.tensors {
            let (x, y) = (a.tensor_data(t), b.tensor_data(t));
            assert_eq!(x, y, "{}", t.name);
            let s = (m0.data_offset + t.offset) as usize;
            let f = &file[s..s + t.n_bytes as usize];
            if t.kind == "attn_q.weight" || t.kind == "attn_k.weight" {
                let rb = t.n_bytes as usize / t.dims[1] as usize;
                for r in [0usize, 1, 2, 31, 32, 33, 63, 64, 100] {
                    let (h, j) = (r / hd, r % hd);
                    let src = h * hd
                        + if j < hd / 2 {
                            2 * j
                        } else {
                            2 * (j - hd / 2) + 1
                        };
                    assert_eq!(
                        &x[r * rb..(r + 1) * rb],
                        &f[src * rb..(src + 1) * rb],
                        "{} row {r}",
                        t.name
                    );
                }
                assert_ne!(x, f, "{} not reordered", t.name);
            } else {
                assert_eq!(x, f, "{} changed", t.name);
            }
        }
        // The file mapping itself is untouched (shared, read-only): the
        // reordered rows are a side copy, kept through a release (slices
        // taken before stay valid, as a file mapping's pages fault back in);
        // the copy refuses a released tensor, as always.
        let q = (m0.tensors.iter())
            .find(|t| t.name == "blk.3.attn_q.weight")
            .unwrap();
        let s = (m0.data_offset + q.offset) as usize;
        let r = s..s + q.n_bytes as usize;
        assert_eq!(&b.data[r.clone()], &file[r.clone()]);
        assert_ne!(b.tensor_data(q), &file[r]);
        let want = a.tensor_data(q).to_vec();
        let held = b.tensor_data(q);
        b.release(q);
        b.release(q);
        assert_eq!(held, &want[..], "a slice taken before the release");
        assert_eq!(b.tensor_data(q), &want[..]);
        a.release(q);
        let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.tensor_data(q).len()));
        assert!(e.is_err(), "the copy read back a released tensor");
        // Neighbours stay readable.
        let k = (m0.tensors.iter())
            .find(|t| t.name == "blk.3.attn_k.weight")
            .unwrap();
        assert_eq!(a.tensor_data(k), b.tensor_data(k));
    }

    #[test]
    fn check_supported_qwen3() {
        tiny3(|_| {}, |_| {}).expect("tiny qwen3");
        let rej = |e: std::io::Result<()>, want: &str| {
            let e = e.expect_err("malformed qwen3 accepted");
            assert!(e.to_string().contains(want), "{e}");
        };
        // qwen3 needs its q/k norms; they must be head_dim long.
        rej(
            tiny3(
                |ts| ts.retain(|t| t.0 != "blk.1.attn_k_norm.weight"),
                |_| {},
            ),
            "missing tensor blk.1.attn_k_norm.weight",
        );
        rej(
            tiny3(
                |ts| {
                    ts.iter_mut().for_each(|t| {
                        if t.0 == "blk.0.attn_q_norm.weight" {
                            t.1 = vec![64];
                        }
                    })
                },
                |_| {},
            ),
            "attn_q_norm.weight: shape",
        );
        // attn_q is dim x q_dim, attn_output q_dim x dim.
        rej(
            tiny3(
                |ts| {
                    ts.iter_mut().for_each(|t| {
                        if t.0 == "blk.0.attn_output.weight" {
                            t.1 = vec![64, 64];
                        }
                    })
                },
                |_| {},
            ),
            "attn_output.weight: shape",
        );
        rej(
            tiny3(|_| {}, |m| m.retain(|kv| kv.0 != "qwen3.block_count")),
            "qwen3.block_count is missing",
        );
        rej(
            tiny3(|_| {}, |m| m.last_mut().unwrap().1 = Val::U32(16)),
            "value_length differs",
        );
        rej(
            tiny3(|_| {}, |m| m[0].1 = Val::Str("gemma".into())),
            "'gemma' is not supported",
        );
        // Without key_length, head_dim is dim / heads (16): attn_k doesn't fit.
        rej(
            tiny3(
                |_| {},
                |m| m.retain(|kv| !kv.0.ends_with("_length") || kv.0.contains("embedding")),
            ),
            "shape",
        );
    }

    /// `feed_batch_matches_feed` on Qwen3-0.6B (q_dim != dim, q/k norms,
    /// tied Q8_0 embedding).
    #[test]
    fn feed_batch_matches_feed_qwen3() {
        for repack in [true, false] {
            let Some(m) = crate::testutil::qwen3_model() else {
                return;
            };
            let net = Qwen2::new_with(&m, repack);
            assert!(net.q_dim != net.dim && net.layers[0].q_norm.is_some());
            feed_batch_matches_feed_on(&net);
        }
    }

    /// `feed_batch_matches_feed` on the small Q8_0 qwen3 fixture (no model
    /// file needed): repacked, its 17- to 130-token batches go through the
    /// split matmul chunks (token ranges of every remainder mod 4) and the
    /// parallel fused per-token passes. Run it with AXON_THREADS >= 32 too:
    /// then the fixture's 128-row matrices have fewer row chunks than threads.
    #[test]
    fn feed_batch_matches_feed_fixture() {
        for repack in [true, false] {
            let m = qwen3_fixture(Q8_0);
            let net = Qwen2::new_with(&m, repack);
            feed_batch_matches_feed_on(&net);
        }
    }

    /// `feed_batch_matches_feed` on Llama-3.2-1B (RoPE factors, head_dim
    /// 64, GQA 4, tied Q6_K head).
    #[test]
    fn feed_batch_matches_feed_llama() {
        for repack in [true, false] {
            let Some(m) = crate::testutil::llama_model() else {
                return;
            };
            let net = Qwen2::new_with(&m, repack);
            assert!(net.head_dim == 64 && net.layers[0].bq.is_none());
            feed_batch_matches_feed_on(&net);
        }
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
            let repacked = net
                .layers
                .iter()
                .filter(|l| l.dense().2.r8.is_some())
                .count();
            assert_eq!(
                repacked > 0,
                repack,
                "repack {repack}: {repacked} ffn_down repacked"
            );
            // Every Q5_0/Q8_0 matrix (0.5B: attention, gate/up, the head)
            // goes through r8q when repacking, none otherwise.
            let all = net.layers.iter().flat_map(|l| l.weights());
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
            .flat_map(|l| [&l.wq, &l.wv, l.dense().0, l.dense().2])
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
            let (w_gate, w_up, w_down) = layer.dense();
            let act = net.matvec_swiglu(w_gate, w_up, &h);
            check(w_down, &act);
            check(w_gate, &h);
            check(w_up, &h);
            let d = net.matvec(w_down, &act);
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
        if matches!(qt, Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | IQ4_NL) {
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
            let net = Qwen2::build(&m, true, true, usize::MAX, false);
            let repacked: Vec<&Tensor> = net
                .layers
                .iter()
                .flat_map(|l| [l.dense().2, l.dense().0, &l.wq])
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

    pub(crate) fn feed_batch_matches_feed_on(net: &Qwen2) {
        // 483 spread-out ordinary token ids (Qwen2's special tokens start at
        // 151643, Llama 3's at 128000). (Taken mod the vocab: the same ids
        // on a real model.)
        let v = net.vocab as u64;
        let span = if net.vocab > 150_000 {
            150_000
        } else {
            128_000
        };
        let ids: Vec<u32> = (0..483u64)
            .map(|i| (i * 2_654_435_761 % span % v) as u32)
            .collect();
        let next = (9707 % v) as u32;
        let mut a = net.new_cache(ids.len() + 1);
        let mut b = net.new_cache(ids.len() + 1);
        for (pos, &id) in ids.iter().enumerate() {
            net.feed(id, pos, &mut a);
        }
        let mut s = 0;
        // Every verify-sized batch (1..=9, so every padding remainder), odd
        // ones (13; 17, 86: split token ranges of 1 and 14 tokens), the old
        // and the current full prefill batch (64, 128) and one past it (130).
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 64, 86, 128, 130] {
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
            // qwen3's per-head norm: both variants, and `rmsnorm_into` per
            // head (the same ops) as the reference.
            let w = mixed(hd, next(), 1.5, false);
            let v = mixed(5 * hd, next(), 3.0, true);
            let (mut a, mut b) = (v.clone(), v.clone());
            rmsnorm_heads::portable(&mut a, hd, &w, 1e-6);
            unsafe { rmsnorm_heads::v3(&mut b, hd, &w, 1e-6) };
            assert_eq!(bits(&a), bits(&b), "rmsnorm_heads hd={hd}");
            let mut r = vec![0f32; v.len()];
            for (x, o) in v.chunks(hd).zip(r.chunks_mut(hd)) {
                rmsnorm_into::portable(x, &w, 1e-6, o);
            }
            assert_eq!(bits(&a), bits(&r), "rmsnorm_heads vs rmsnorm_into hd={hd}");
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
                    softmax_from::portable(&src, stride, None, &mut a);
                    unsafe { softmax_from::v3(&src, stride, None, &mut b) };
                    assert_eq!(bits(&a), bits(&b), "softmax_from n={n}");
                    unsafe { softmax_from::fma(&src, stride, None, &mut b) };
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
                score_softmax::portable(&qg[..hd], &kc, kv_dim, off, scale, None, &mut a);
                unsafe { score_softmax::v3(&qg[..hd], &kc, kv_dim, off, scale, None, &mut b) };
                assert_eq!(bits(&a), bits(&b), "score_softmax hd={hd} np={np}");
                score_softmax::portable(&qg[..hd], &kh, kv_dim, off, scale, None, &mut a);
                unsafe { score_softmax::v3(&qg[..hd], &kh, kv_dim, off, scale, None, &mut b) };
                assert_eq!(bits(&a), bits(&b), "score_softmax f16 hd={hd} np={np}");
                unsafe { score_softmax::fma(&qg[..hd], &kh, kv_dim, off, scale, None, &mut b) };
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
                attend_causal::portable(
                    &qg, &kh, &vh, kv_dim, hd, kvh, np, scale, None, &mut sa, &mut a,
                );
                unsafe {
                    attend_causal::v3(
                        &qg, &kh, &vh, kv_dim, hd, kvh, np, scale, None, &mut sb, &mut b,
                    )
                };
                assert_eq!(bits(&a), bits(&b), "attend_causal hd={hd} np={np}");
                unsafe {
                    attend_causal::fma(
                        &qg, &kh, &vh, kv_dim, hd, kvh, np, scale, None, &mut sb, &mut b,
                    )
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
                        &q, &kc, &vc, hd, hd, 0, np, scale, None, &mut sc, &mut out,
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
                        score_softmax::imp::<false, false, f32>(qv, &kc, hd, 0, scale, None, &mut w)
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
            attend_group(qg, kc, vc, kv_dim, hd, kvh, np, None, o);
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
                attend(
                    q, kc, vc, kv_dim, hd, np, tile, threads, None, buf, &mut got,
                );
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
