//! `deepseek2` (DeepSeek V2/V3/V3.1, Kimi K2, Moonlight) on the CPU: the
//! exact reference (r12 stage 0).
//!
//! A deepseek2 block is a qwen3moe block with two things changed.
//!
//! Attention is MLA in llama.cpp's *absorbed* form (src/models/deepseek2.cpp,
//! the `is_mla` branch), the only form current GGUFs carry. Per token, with
//! `n_h` heads of nope (128) + rope (64) query dims, a latent of `lat` (512)
//! and values of `hv` (128) per head:
//!
//! ```text
//! q    = Wq h               (attn_q; or attn_q_b · rmsnorm(attn_q_a h) with q_lora)
//! c    = Wkv_a h            (attn_kv_a_mqa: [lat latent | rope k_pe])
//! K    = [rmsnorm(c[..lat]) | rope(c[lat..])]      one cache row per position (576)
//! Q_h  = [Wk_b[h] q_nope_h | rope(q_pe_h)]         (attn_k_b: head h's lat x nope)
//! o_h  = softmax_t(Q_h · K_t * kq_scale) · K_t[..lat]        (V is K's first lat)
//! y    = Wo [Wv_b[h] o_h for every h]              (attn_v_b: head h's hv x lat)
//! ```
//!
//! The cache holds K only (`KvCache` with empty V arrays). RoPE is llama's
//! "normal" (adjacent-pair) kind on the 64 rope dims; the loader puts those
//! rows of attn_q / attn_q_b / attn_kv_a_mqa in rotate-half order
//! (`gguf::rope_unpermute_plan`), so the NEOX `rope_head` applies. The
//! attention scale is `rope::deepseek2_scales`'s (YaRN's mscale² over
//! sqrt(nope + rope)), not 1/sqrt(576).
//!
//! The FFN: the first `leading_dense_block_count` layers are dense; the
//! rest are MoE (`moe.rs`, with the router's `Gate`: sigmoid or softmax,
//! `exp_probs_b` for selection, renorm, the weight scale) plus shared
//! experts, a dense FFN of width `ffn_exp * n_shared` added to the experts'
//! sum.
//!
//! Exactness: decode (`ds2_hidden`) and a batch (`ds2_hidden_batch`) run the
//! same ops per token. Every matrix product is one weight row times one
//! token's quantized input with the dense kernels (gemv == gemm), per head
//! for the absorb/expand matrices; the attention of a token is
//! `mla_causal`, whatever the heads per task (`dot_group` per head is `dot`;
//! the V sum per element one fma chain over t). Absorbed everywhere: never
//! the expanded form for prefill (it rounds differently).

use super::{
    F16, F32, Ffn, Kind, KvCache, KvData, KvElem, Layers, Model, Pool, QuantizedAct, Qwen2,
    SUPPORTED_TYPES, Tensor, Weight, add_assign, add_unbatched, dot_group, find, find_opt,
    load_vec, meta_num, moe, par_map, pool, r8_arena, repacked_len, rope_head, unbatch,
    unsupported_type, vmath, vsum_rows, with_scores,
};
use crate::gguf::Val;
use std::io;

/// One deepseek2 block's weights.
pub(super) struct MlaLayer<'a> {
    attn_norm: Vec<f32>,
    /// q_lora (V3, K2): attn_q_a and attn_q_a_norm; then `wq` is attn_q_b.
    q_a: Option<(Weight<'a>, Vec<f32>)>,
    /// attn_q (lite: Moonlight, V2-Lite) or attn_q_b: n_h * (nope + rope) rows.
    wq: Weight<'a>,
    /// attn_kv_a_mqa: lat + rope rows.
    wkv_a: Weight<'a>,
    kv_a_norm: Vec<f32>,
    /// Per head: attn_k_b's lat x nope slice (absorb), attn_v_b's hv x lat (expand).
    wk_b: Vec<Weight<'a>>,
    wv_b: Vec<Weight<'a>>,
    wo: Weight<'a>,
    ffn_norm: Vec<f32>,
    ffn: Ffn<'a>,
    /// Shared experts' gate, up, down (MoE layers with expert_shared_count > 0).
    shared: Option<[Weight<'a>; 3]>,
}

impl<'a> MlaLayer<'a> {
    /// Every matrix, in the r8 arena's order (`layer_items`).
    fn weights_mut(&mut self) -> Vec<&mut Weight<'a>> {
        let mut v: Vec<&mut Weight<'a>> = Vec::new();
        if let Some((w, _)) = &mut self.q_a {
            v.push(w);
        }
        v.push(&mut self.wq);
        v.push(&mut self.wkv_a);
        v.extend(self.wk_b.iter_mut());
        v.extend(self.wv_b.iter_mut());
        v.push(&mut self.wo);
        match &mut self.ffn {
            Ffn::Dense { gate, up, down } => v.extend([gate, up, down]),
            Ffn::Moe(m) => v.extend(m.weights_mut()),
        }
        if let Some(s) = &mut self.shared {
            v.extend(s.iter_mut());
        }
        v
    }
}

/// The deepseek2 part of a `Qwen2` (its `layers` are empty).
pub(crate) struct Ds2<'a> {
    pub(super) layers: Vec<MlaLayer<'a>>,
    pub(super) d: Dims,
}

/// MLA shapes and scales.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Dims {
    pub n_heads: usize,
    pub nope: usize,
    pub rope: usize,
    /// kv_lora_rank: the latent (V) width; K rows are lat + rope.
    pub lat: usize,
    /// value_length_mla: per-head output width.
    pub hv: usize,
    pub q_lora: Option<usize>,
    pub kq_scale: f32,
    pub rope_m: f32,
}

impl Dims {
    /// Query head width before absorption (nope + rope).
    pub fn hk(&self) -> usize {
        self.nope + self.rope
    }

    /// Cache row width (lat + rope).
    pub fn krow(&self) -> usize {
        self.lat + self.rope
    }
}

fn num(m: &Model, k: &str) -> Option<f64> {
    meta_num(m, &format!("deepseek2.{k}"))
}

/// A boolean key (GGUF bool, or a number: non-zero), None if absent.
fn flag(m: &Model, k: &str) -> Option<bool> {
    let key = format!("deepseek2.{k}");
    match m.metadata.iter().find(|(n, _)| *n == key).map(|(_, v)| v) {
        Some(Val::Bool(b)) => Some(*b),
        Some(v) => v.as_f64().map(|x| x != 0.0),
        None => None,
    }
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

const LEGACY: &str = "legacy non-MLA deepseek2 GGUF (an unsplit attn_kv_b, head_count_kv > 1, \
                      no key_length_mla): re-convert it from the Hugging Face model with a \
                      current llama.cpp (convert_hf_to_gguf.py, 2025-04 or later), which writes \
                      attn_k_b / attn_v_b";

/// llama.cpp's expert_gating_func for the file: 1 softmax, 2 sigmoid; a
/// file without the key is softmax (DeepSeek V2), except GLM 4.7 Lite
/// (47 or 48 layers, 154880 tokens), sigmoid (deepseek2.cpp).
fn sigmoid_gating(m: &Model, n_layers: usize, vocab: usize) -> io::Result<bool> {
    match num(m, "expert_gating_func").map(|v| v as i64) {
        None | Some(0) => Ok((n_layers == 47 || n_layers == 48) && vocab == 154880),
        Some(1) => Ok(false),
        Some(2) => Ok(true),
        Some(v) => Err(bad(format!(
            "expert_gating_func {v} is not supported (softmax or sigmoid)"
        ))),
    }
}

/// Everything `build` relies on (`model::check_supported` for deepseek2).
pub(super) fn check(m: &Model) -> io::Result<Dims> {
    let num = |k: &str| self::num(m, k);
    let need = |k: &str| -> io::Result<usize> {
        match num(k) {
            Some(v) if v.fract() == 0.0 && v >= 1.0 && v <= u32::MAX as f64 => Ok(v as usize),
            Some(v) => Err(bad(format!(
                "metadata deepseek2.{k} = {v} is not a positive integer"
            ))),
            None => {
                let key = format!("deepseek2.{k}");
                Err(bad(match m.metadata.iter().find(|(n, _)| *n == key) {
                    // llama.cpp reads some keys per layer (get_key_or_arr:
                    // expert_feed_forward_length); one value per model here.
                    Some((_, Val::Arr(a))) => format!(
                        "metadata {key} is an array ({} values, per layer): per-layer values \
                         are not supported",
                        a.len()
                    ),
                    Some((_, v)) => format!("metadata {key} = {v} is not a number"),
                    None => format!("metadata {key} is missing"),
                }))
            }
        }
    };
    let n_layers = need("block_count")?;
    let dim = need("embedding_length")?;
    let n_heads = need("attention.head_count")?;
    // The legacy layout first: everything below would only misreport it.
    let legacy = (0..n_layers).any(|i| find_opt(m, &format!("blk.{i}.attn_kv_b.weight")).is_some());
    if legacy
        || num("attention.head_count_kv").is_some_and(|v| v != 1.0)
        || num("attention.key_length_mla").is_none()
    {
        return Err(bad(LEGACY.into()));
    }
    let lat = need("attention.kv_lora_rank")?;
    let rope = need("rope.dimension_count")?;
    let hk = need("attention.key_length_mla")?;
    let hv = need("attention.value_length_mla")?;
    if need("attention.key_length")? != lat + rope || need("attention.value_length")? != lat {
        return Err(bad(format!(
            "deepseek2: key_length/value_length must be kv_lora_rank + rope ({}) / kv_lora_rank \
             ({lat}) in the absorbed form",
            lat + rope
        )));
    }
    if hk <= rope || !rope.is_multiple_of(2) {
        return Err(bad(format!(
            "deepseek2: key_length_mla {hk} with a {rope}-dim rope part"
        )));
    }
    let nope = hk - rope;
    for (k, what) in [
        ("attention.scale", "a custom attention.scale"),
        (
            "attention.temperature_scale",
            "attention temperature scaling",
        ),
        ("nextn_predict_layers", "NextN/MTP layers"),
    ] {
        if num(k).is_some_and(|v| v != 0.0) {
            return Err(bad(format!("deepseek2: {what} is not supported")));
        }
    }
    if num("expert_group_count").is_some_and(|v| v > 1.0) {
        return Err(bad(
            "deepseek2: grouped expert selection (expert_group_count > 1) is not supported".into(),
        ));
    }
    // RoPE scaling: none, or YaRN (rope.rs; its yarn_log_multiplier is
    // deepseek2's mscale_all_dim rule).
    let rs = |k: &str| num(&format!("rope.scaling.{k}"));
    let ty = m
        .metadata
        .iter()
        .find(|(k, _)| k == "deepseek2.rope.scaling.type")
        .map(|(_, v)| v);
    match ty {
        None => {}
        Some(Val::Str(t)) if t == "none" => {}
        Some(Val::Str(t)) if t == "yarn" => {
            let f = rs("factor").unwrap_or(1.0);
            let ok = |k: &str, lo: f64| rs(k).is_none_or(|v| v.is_finite() && v >= lo);
            let fine = f.is_finite()
                && f >= 1.0
                && ok("original_context_length", 1.0)
                && ok("yarn_beta_fast", f64::MIN_POSITIVE)
                && ok("yarn_beta_slow", f64::MIN_POSITIVE)
                && ok("yarn_ext_factor", 0.0)
                && ok("attn_factor", f64::MIN_POSITIVE)
                && ok("yarn_log_multiplier", 0.0);
            if !fine {
                return Err(bad("RoPE scaling yarn: a parameter is out of range".into()));
            }
            if rs("axon_mscale").is_some() {
                return Err(bad(
                    "--yarn-mscale doesn't apply to deepseek2 (its mscale is the attention \
                     scale's: rope::deepseek2_scales)"
                        .into(),
                ));
            }
        }
        Some(v) => return Err(bad(format!("RoPE scaling {v} is not supported"))),
    }
    if num("rope.scale_linear").is_some_and(|v| v != 1.0 && v != 0.0)
        || (ty.is_none() && rs("factor").is_some_and(|v| v != 1.0 && v != 0.0))
    {
        return Err(bad("deepseek2: linear RoPE scaling is not supported".into()));
    }
    if find_opt(m, "rope_freqs.weight").is_some() {
        return Err(bad("deepseek2: rope_freqs.weight is not supported".into()));
    }

    let tensor = |name: &str, want: &[u64]| -> io::Result<&Tensor> {
        let t = find_opt(m, name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        let q = t.qtype;
        if !SUPPORTED_TYPES.contains(&q.id) {
            return Err(bad(unsupported_type(m, name, q)));
        }
        let ok = t.dims.len() >= want.len()
            && t.dims[..want.len()]
                .iter()
                .zip(want)
                .all(|(&d, &w)| (w == 0 && d > 0) || d == w)
            && t.dims[want.len()..].iter().all(|&d| d == 1)
            && t.dims[0].is_multiple_of(q.block_size as u64);
        if !ok {
            return Err(bad(format!(
                "tensor {name}: shape {:?} doesn't fit the model ({want:?})",
                t.dims
            )));
        }
        Ok(t)
    };
    let embd = tensor("token_embd.weight", &[dim as u64, 0])?;
    let vocab = embd.dims[1] as usize;
    if find_opt(m, "output.weight").is_some() {
        tensor("output.weight", &[dim as u64, vocab as u64])?;
    }
    tensor("output_norm.weight", &[dim as u64])?;
    let lead = num("leading_dense_block_count").map_or(0, |v| v as usize);
    let n_expert = need("expert_count")?;
    let k = need("expert_used_count")?;
    let ffe = need("expert_feed_forward_length")?;
    let shared = num("expert_shared_count").map_or(0, |v| v as usize);
    if k > n_expert {
        return Err(bad(format!(
            "expert_used_count {k} exceeds expert_count {n_expert}"
        )));
    }
    if lead > n_layers {
        return Err(bad(format!(
            "leading_dense_block_count {lead} exceeds block_count {n_layers}"
        )));
    }
    sigmoid_gating(m, n_layers, vocab)?;
    if let Some(v) = num("expert_weights_scale").filter(|v| !v.is_finite()) {
        return Err(bad(format!("expert_weights_scale {v} is not finite")));
    }
    let q_lora = (0..n_layers)
        .any(|i| find_opt(m, &format!("blk.{i}.attn_q_a.weight")).is_some())
        .then(|| need("attention.q_lora_rank"))
        .transpose()?;
    let (d64, nh) = (dim as u64, n_heads as u64);
    for i in 0..n_layers {
        let mut known: Vec<String> = Vec::new();
        let mut t = |s: &str, want: &[u64]| -> io::Result<&Tensor> {
            known.push(s.to_string());
            tensor(&format!("blk.{i}.{s}"), want)
        };
        t("attn_norm.weight", &[d64])?;
        match q_lora {
            Some(r) => {
                t("attn_q_a.weight", &[d64, r as u64])?;
                t("attn_q_a_norm.weight", &[r as u64])?;
                t("attn_q_b.weight", &[r as u64, nh * hk as u64])?;
            }
            None => {
                t("attn_q.weight", &[d64, nh * hk as u64])?;
            }
        }
        t("attn_kv_a_mqa.weight", &[d64, (lat + rope) as u64])?;
        t("attn_kv_a_norm.weight", &[lat as u64])?;
        t("attn_k_b.weight", &[nope as u64, lat as u64, nh])?;
        t("attn_v_b.weight", &[lat as u64, hv as u64, nh])?;
        t("attn_output.weight", &[nh * hv as u64, d64])?;
        t("ffn_norm.weight", &[d64])?;
        if i < lead {
            let ffn = t("ffn_gate.weight", &[d64, 0])?.dims[1];
            t("ffn_up.weight", &[d64, ffn])?;
            t("ffn_down.weight", &[ffn, d64])?;
        } else {
            let r = t("ffn_gate_inp.weight", &[d64, n_expert as u64])?;
            if r.qtype.id != F32 && r.qtype.id != F16 {
                return Err(bad(format!(
                    "blk.{i}.ffn_gate_inp.weight: type {} (want F32 or F16)",
                    r.qtype.name
                )));
            }
            if find_opt(m, &format!("blk.{i}.exp_probs_b.bias")).is_some() {
                let b = t("exp_probs_b.bias", &[n_expert as u64])?;
                if b.qtype.id != F32 {
                    return Err(bad(format!("blk.{i}.exp_probs_b.bias: want F32")));
                }
            }
            let (e, f) = (n_expert as u64, ffe as u64);
            t("ffn_gate_exps.weight", &[d64, f, e])?;
            t("ffn_up_exps.weight", &[d64, f, e])?;
            t("ffn_down_exps.weight", &[f, d64, e])?;
            if shared > 0 {
                let sw = (ffe * shared) as u64;
                t("ffn_gate_shexp.weight", &[d64, sw])?;
                t("ffn_up_shexp.weight", &[d64, sw])?;
                t("ffn_down_shexp.weight", &[sw, d64])?;
            }
        }
        // Anything else in the block would be ignored: refuse it.
        let pre = format!("blk.{i}.");
        if let Some(x) = m.tensors.iter().find(|x| {
            x.name
                .strip_prefix(&pre)
                .is_some_and(|s| !known.iter().any(|k| k == s))
        }) {
            return Err(bad(format!(
                "tensor {} is not supported on deepseek2",
                x.name
            )));
        }
    }
    crate::tokenizer::pre_of(m).map_err(bad)?;
    match m
        .metadata
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.tokens")
    {
        Some((_, Val::Arr(a))) if !a.is_empty() && a.len() <= vocab => {}
        Some((_, Val::Arr(a))) => {
            return Err(bad(format!(
                "tokenizer has {} tokens for an embedding of {vocab} rows",
                a.len()
            )));
        }
        _ => return Err(bad("tokenizer.ggml.tokens is missing".into())),
    }
    let (rope_m, kq_scale) = crate::rope::deepseek2_scales(m, hk);
    Ok(Dims {
        n_heads,
        nope,
        rope,
        lat,
        hv,
        q_lora,
        kq_scale,
        rope_m,
    })
}

/// Layer i's matrices in arena order: (tensor, its slice for 3-D ones).
fn layer_items<'m>(
    m: &'m Model,
    d: &Dims,
    i: usize,
    lead: usize,
) -> Vec<(&'m Tensor, Option<usize>)> {
    let t = |s: &str| find(m, &format!("blk.{i}.{s}"));
    let mut v: Vec<(&Tensor, Option<usize>)> = Vec::new();
    if d.q_lora.is_some() {
        v.push((t("attn_q_a.weight"), None));
        v.push((t("attn_q_b.weight"), None));
    } else {
        v.push((t("attn_q.weight"), None));
    }
    v.push((t("attn_kv_a_mqa.weight"), None));
    for s in ["attn_k_b.weight", "attn_v_b.weight"] {
        v.extend((0..d.n_heads).map(|h| (t(s), Some(h))));
    }
    v.push((t("attn_output.weight"), None));
    if i < lead {
        for s in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
            v.push((t(s), None));
        }
    } else {
        for s in moe::EXPS {
            let x = t(s);
            v.extend((0..x.dims[2] as usize).map(|e| (x, Some(e))));
        }
        for s in [
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ] {
            if let Some(x) = find_opt(m, &format!("blk.{i}.{s}")) {
                v.push((x, None));
            }
        }
    }
    v
}

/// What the GPU backend (`cuda::qwen2::mla`) reads beyond the tensors: the
/// MLA shapes and scales, the dense lead, the router's rule (`moe::Gate`
/// less the per-layer bias).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuCfg {
    pub d: Dims,
    pub lead: usize,
    pub sigmoid: bool,
    pub norm: bool,
    pub scale: f32,
}

/// `GpuCfg` of a deepseek2 model (Err: `check`'s refusals).
pub(crate) fn gpu_cfg(m: &Model) -> io::Result<GpuCfg> {
    let d = check(m)?;
    let n_layers = num(m, "block_count").map_or(0, |v| v as usize);
    let vocab = find(m, "token_embd.weight").dims[1] as usize;
    Ok(GpuCfg {
        d,
        lead: num(m, "leading_dense_block_count").map_or(0, |v| v as usize),
        sigmoid: sigmoid_gating(m, n_layers, vocab)?,
        norm: flag(m, "expert_weights_norm").unwrap_or(false),
        scale: num(m, "expert_weights_scale")
            .filter(|&v| v != 0.0)
            .map_or(1.0, |v| v as f32),
    })
}

/// Whether arena item `t` is a routed expert's matrix.
fn is_expert(t: &Tensor) -> bool {
    moe::EXPS.iter().any(|s| t.name.ends_with(s))
}

/// `Qwen2::new_with` of a deepseek2 model: the whole model (no layer cut;
/// `lm` refuses the placements that would want one). `experts`: the routed
/// experts only repacked (`Qwen2::new_experts`, the CPU half of
/// `crate::moe`'s split); every other matrix stays in the mapping, never
/// read here.
pub(super) fn build<'a>(m: &'a Model, repack: bool, experts: bool) -> Qwen2<'a> {
    build_only(m, repack, experts, None)
}

/// `build`, repacking only the matrices of layers `only` if given (the rest
/// stay in the mapping, read in place): tests that run a few layers'
/// experts on the CPU without the whole model's r8 arena.
pub(super) fn build_only<'a>(
    m: &'a Model,
    repack: bool,
    experts: bool,
    only: Option<&[usize]>,
) -> Qwen2<'a> {
    let num = |k: &str| self::num(m, k);
    let d = check(m).unwrap_or_else(|e| panic!("{e}"));
    let need = |k: &str| num(k).unwrap_or_else(|| panic!("missing metadata deepseek2.{k}"));
    let n_layers = need("block_count") as usize;
    let dim = need("embedding_length") as usize;
    let eps = num("attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32;
    let lead = num("leading_dense_block_count").map_or(0, |v| v as usize);
    let freqs = super::rope_freqs(m, d.rope);
    let token_embd = find(m, "token_embd.weight");
    let vocab = token_embd.dims[1] as usize;
    let out_t = find_opt(m, "output.weight").unwrap_or(token_embd);
    let gate = moe::Gate {
        sigmoid: sigmoid_gating(m, n_layers, vocab).expect("checked"),
        bias: None,
        norm: flag(m, "expert_weights_norm").unwrap_or(false),
        scale: num("expert_weights_scale")
            .filter(|&v| v != 0.0)
            .map_or(1.0, |v| v as f32),
    };
    let dims = moe::MoeDims {
        n_expert: need("expert_count") as usize,
        k: need("expert_used_count") as usize,
        ffn: need("expert_feed_forward_length") as usize,
    };

    // The arena: output, then each layer's `layer_items` (64-byte aligned).
    let items: Vec<Vec<(&Tensor, Option<usize>)>> =
        (0..n_layers).map(|i| layer_items(m, &d, i, lead)).collect();
    let mut plan: Vec<Option<(usize, usize)>> = Vec::new();
    let mut total = 0;
    for t in std::iter::once(out_t).chain(items.iter().flatten().map(|(t, _)| *t)) {
        let picked = only.is_none_or(|o| usize::try_from(t.layer).is_ok_and(|l| o.contains(&l)));
        let len = if repack && picked && (!experts || is_expert(t)) {
            repacked_len(t)
        } else {
            None
        };
        plan.push(len.map(|len| (total, len)));
        total += len.map_or(0, |l| l.next_multiple_of(64));
    }
    let mut arena = (total > 0).then(|| r8_arena(total));
    let mut slots: Vec<Option<&mut [u8]>> = Vec::with_capacity(plan.len());
    {
        let mut rest: &mut [u8] = arena.as_deref_mut().unwrap_or(&mut []);
        let mut at = 0;
        for p in &plan {
            slots.push(p.map(|(off, len)| {
                let (_, r) = std::mem::take(&mut rest).split_at_mut(off - at);
                let (s, r) = r.split_at_mut(len);
                rest = r;
                at = off + len;
                s
            }));
        }
    }
    let mut slots = slots.into_iter();
    let mut output = Weight::load(
        m,
        out_t,
        slots.next().unwrap(),
        !std::ptr::eq(out_t, token_embd),
    );
    let output_norm = load_vec(m, find(m, "output_norm.weight"));
    let layer_slots: Vec<std::sync::Mutex<Vec<Option<&mut [u8]>>>> = items
        .iter()
        .map(|it| std::sync::Mutex::new(slots.by_ref().take(it.len()).collect()))
        .collect();

    let mut layers = par_map(n_layers, super::INLINE_WORK + 1, |i| {
        let mut dst = std::mem::take(&mut *layer_slots[i].lock().unwrap()).into_iter();
        let mut its = items[i].iter();
        let vals = |t: &Tensor| {
            let v = load_vec(m, t);
            if repack {
                m.release(t);
            }
            v
        };
        let tv = |s: &str| vals(find(m, &format!("blk.{i}.{s}")));
        // The next item, whole or one slice (released after its last).
        let mut next = || {
            let &(t, slice) = its.next().unwrap();
            let dst = dst.next().unwrap();
            match slice {
                None => Weight::load(m, t, dst, true),
                Some(e) => {
                    let repacked = dst.is_some();
                    let w = Weight::load_bytes(t, moe::expert_bytes(m, t, e), dst);
                    if repacked && e + 1 == t.dims[2] as usize {
                        m.release(t);
                    }
                    w
                }
            }
        };
        let q_a = d.q_lora.map(|_| (next(), tv("attn_q_a_norm.weight")));
        let wq = next();
        let wkv_a = next();
        let wk_b: Vec<Weight> = (0..d.n_heads).map(|_| next()).collect();
        let wv_b: Vec<Weight> = (0..d.n_heads).map(|_| next()).collect();
        let wo = next();
        let (ffn, shared) = if i < lead {
            let [gate, up, down] = [(); 3].map(|_| next());
            (Ffn::Dense { gate, up, down }, None)
        } else {
            let [ge, ue, de] = [(); 3].map(|_| (0..dims.n_expert).map(|_| next()).collect());
            let shared = find_opt(m, &format!("blk.{i}.ffn_gate_shexp.weight"))
                .map(|_| [(); 3].map(|_| next()));
            let mut g = gate.clone();
            g.bias = find_opt(m, &format!("blk.{i}.exp_probs_b.bias")).map(vals);
            let router = tv("ffn_gate_inp.weight");
            let moe = moe::Moe::new(dims, router, ge, ue, de, g);
            (Ffn::Moe(Box::new(moe)), shared)
        };
        MlaLayer {
            attn_norm: tv("attn_norm.weight"),
            q_a,
            wq,
            wkv_a,
            kv_a_norm: tv("attn_kv_a_norm.weight"),
            wk_b,
            wv_b,
            wo,
            ffn_norm: tv("ffn_norm.weight"),
            ffn,
            shared,
        }
    });
    drop(layer_slots);
    drop(slots);
    if let Some(a) = arena.take() {
        let map = std::sync::Arc::new(a.make_read_only().expect("r8: mprotect"));
        let ws =
            std::iter::once(&mut output).chain(layers.iter_mut().flat_map(|l| l.weights_mut()));
        for (w, p) in ws.zip(&plan) {
            w.r8 = p.map(|(off, len)| super::R8 {
                map: map.clone(),
                off,
                len,
            });
        }
    }
    Qwen2 {
        n_layers,
        dim,
        n_heads: d.n_heads,
        n_kv_heads: 1,
        head_dim: d.krow(),
        q_dim: d.n_heads * d.hv,
        vocab,
        eps,
        freqs: freqs.into_boxed_slice(),
        rope_m: d.rope_m,
        gemma: false,
        token_embd,
        embd: m.tensor_data(token_embd),
        output_norm: output_norm.into_boxed_slice(),
        output: Some(output),
        layers: Vec::new(),
        ds2: Some(Box::new(Ds2 { layers, d })),
    }
}

/// A K-only cache of `n_layers` layers, rows of `krow` (V is each row's
/// first `lat`: no V arrays).
pub(super) fn new_cache(n_layers: usize, krow: usize, max_ctx: usize) -> KvCache {
    new_cache_of(n_layers, krow, max_ctx, crate::kv::use_f32())
}

/// `new_cache` with f32 (else f16) elements.
pub(super) fn new_cache_of(n_layers: usize, krow: usize, max_ctx: usize, f32: bool) -> KvCache {
    fn k_only<T: KvElem>(n: usize, len: usize) -> Layers<T> {
        Layers {
            k: (0..n).map(|_| crate::kv::KvBuf::zeroed(len)).collect(),
            v: (0..n).map(|_| crate::kv::KvBuf::zeroed(0)).collect(),
        }
    }
    let len = max_ctx * krow;
    KvCache {
        data: if f32 {
            KvData::F32(k_only(n_layers, len))
        } else {
            KvData::F16(k_only(n_layers, len))
        },
        kv_dim: krow,
        max_ctx,
    }
}

/// Run `$body` with `$k` bound to layer `$l`'s K cache (`&mut [T]`).
macro_rules! with_k_layer {
    ($cache:expr, $l:expr, |$k:ident| $body:block) => {
        match &mut $cache.data {
            KvData::F16(c) => {
                let $k = &mut c.k[$l][..];
                $body
            }
            KvData::F32(c) => {
                let $k = &mut c.k[$l][..];
                $body
            }
        }
    };
}

multiversion! {
    /// The attention of one token's `g` query heads qg ([head][lat + rope],
    /// absorbed) over cache rows 0..np of kc ([t][lat + rope]): scores
    /// `dot_group` * scale, a softmax per head, then out[h][..lat] = sum
    /// over t of p_t K_t[..lat] (`vsum_rows`, V being K's prefix). `sc` is
    /// scratch of (np + 1) * g floats. The same per-head bits for any g.
    fn mla_causal<V3, F, T: KvElem>(
        qg: &[f32],
        kc: &[T],
        krow: usize,
        lat: usize,
        np: usize,
        scale: f32,
        sc: &mut [f32],
        out: &mut [f32],
    ) {
        let g = qg.len() / krow;
        let (sc, row) = sc.split_at_mut(np * g);
        for t in 0..np {
            // SAFETY (every call): V3 only in the v3 variant.
            unsafe { dot_group::<V3, F, T>(qg, &kc[t * krow..(t + 1) * krow], row) };
            for (h, &s) in row.iter().enumerate() {
                sc[h * np + t] = s * scale;
            }
        }
        for s in sc.chunks_exact_mut(np) {
            unsafe { vmath::softmax::<V3, F>(s) };
        }
        unsafe { vsum_rows::imp::<V3, F, T>(sc, np, np, g, kc, krow, out, lat) };
    }
}

impl<'a> Qwen2<'a> {
    fn ds2(&self) -> &Ds2<'a> {
        self.ds2.as_deref().expect("a deepseek2 model")
    }

    /// Layer l's FFN (`moe_layer`'s experts on a deepseek2 model).
    pub(super) fn ds2_layer_ffn(&self, l: usize) -> &Ffn<'a> {
        &self.ds2().layers[l].ffn
    }

    /// `heads[h] · xs[h]` for every head h (each its own input), in one
    /// fork/join: out[h * rows..] (rows = each matrix's row count).
    fn heads_vec(&self, ws: &[Weight], xs: &[&[f32]]) -> Vec<f32> {
        let rows = ws[0].rows();
        let xq: Vec<QuantizedAct> = ws
            .iter()
            .zip(xs)
            .map(|(w, x)| QuantizedAct::for_types(x, &[w.qt()]))
            .collect();
        let align = if ws[0].r8.is_some() { 8 } else { 1 };
        let threads = Pool::global().threads();
        let target = (rows * ws.len() / (threads * 4)).clamp(4, 64);
        let chunk = moe::chunk_dividing(rows, target, align);
        let mut out = vec![0f32; rows * ws.len()];
        let (cfg, own_all) = Self::decode_cfg();
        pool::for_each_chunk_mut_in(Kind::Decode, &mut out, chunk, |c, ys| {
            let g = c * chunk;
            let (h, r) = (g / rows, g % rows);
            self.rows_vec(&ws[h], r, &xq[h], ys, cfg, own_all);
        });
        out
    }

    /// `heads_vec` for a batch: xs[b][h] is token b's input to head h;
    /// returns per token the heads' outputs concatenated.
    fn heads_batch(&self, ws: &[Weight], xs: &[Vec<&[f32]>]) -> Vec<Vec<f32>> {
        let nb = xs.len();
        let rows = ws[0].rows();
        let mut out = vec![vec![0f32; rows * ws.len()]; nb];
        for (h, w) in ws.iter().enumerate() {
            let col: Vec<&[f32]> = xs.iter().map(|x| x[h]).collect();
            let xq = Self::quantize_batch(&col, &[w.qt()]);
            let (y, _, slots) = self.matmul_q([w], &xq);
            for (b, o) in out.iter_mut().enumerate() {
                for r in 0..rows {
                    o[h * rows + r] = y[r * slots + b];
                }
            }
        }
        out
    }

    /// The cache row of a token from its kv_a output c: the latent normed,
    /// the rope part turned (`table`).
    fn mla_krow(&self, ly: &MlaLayer, d: &Dims, c: &[f32], table: &[(f32, f32)]) -> Vec<f32> {
        let mut k = self.rmsnorm(&c[..d.lat], &ly.kv_a_norm);
        let mut pe = c[d.lat..d.krow()].to_vec();
        rope_head(&mut pe, table);
        k.extend_from_slice(&pe);
        k
    }

    /// The queries' rope parts turned in place, and each head's nope part
    /// (the absorb inputs).
    fn mla_qrope(d: &Dims, q: &mut [f32], table: &[(f32, f32)]) {
        let hk = d.hk();
        for h in 0..d.n_heads {
            rope_head(&mut q[h * hk + d.nope..(h + 1) * hk], table);
        }
    }

    /// Absorbed queries [head][lat + rope] from the absorb outputs qa
    /// ([head][lat]) and the turned rope parts of q.
    fn mla_qcat(d: &Dims, qa: &[f32], q: &[f32]) -> Vec<f32> {
        let (hk, kr) = (d.hk(), d.krow());
        let mut out = vec![0f32; d.n_heads * kr];
        for h in 0..d.n_heads {
            out[h * kr..h * kr + d.lat].copy_from_slice(&qa[h * d.lat..(h + 1) * d.lat]);
            out[h * kr + d.lat..(h + 1) * kr].copy_from_slice(&q[h * hk + d.nope..(h + 1) * hk]);
        }
        out
    }

    /// deepseek2's FFN of one normed hidden state.
    fn ds2_ffn(&self, ly: &MlaLayer, h: &[f32]) -> Vec<f32> {
        match &ly.ffn {
            Ffn::Dense { gate, up, down } => {
                let act = self.matvec_swiglu(gate, up, h);
                self.matvec(down, &act)
            }
            Ffn::Moe(moe) => {
                let mut y = self.moe_ffn(moe, h);
                if let Some([g, u, dn]) = &ly.shared {
                    let act = self.matvec_swiglu(g, u, h);
                    // llama.cpp: moe_out + ffn_shexp (one add per element).
                    add_assign(&mut y, &self.matvec(dn, &act));
                }
                y
            }
        }
    }

    /// `hidden` of a deepseek2 model.
    pub(super) fn ds2_hidden(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        assert!(
            pos < cache.max_ctx,
            "position {pos} exceeds cache size {}",
            cache.max_ctx
        );
        let ds = self.ds2();
        let d = &ds.d;
        let (kr, np) = (d.krow(), pos + 1);
        let table = Self::rope_table_of(&self.freqs, pos, self.rope_m);
        let mut x = self.embed(token);
        for (l, ly) in ds.layers.iter().enumerate() {
            let h = self.rmsnorm(&x, &ly.attn_norm);
            let (mut q, c) = match &ly.q_a {
                None => {
                    let [q, c] = self.matvec_many([&ly.wq, &ly.wkv_a], &h);
                    (q, c)
                }
                Some((wqa, qan)) => {
                    let [qa, c] = self.matvec_many([wqa, &ly.wkv_a], &h);
                    (self.matvec(&ly.wq, &self.rmsnorm(&qa, qan)), c)
                }
            };
            let k = self.mla_krow(ly, d, &c, &table);
            Self::mla_qrope(d, &mut q, &table);
            let hk = d.hk();
            let nopes: Vec<&[f32]> = (0..d.n_heads)
                .map(|h| &q[h * hk..h * hk + d.nope])
                .collect();
            let qa = self.heads_vec(&ly.wk_b, &nopes);
            let qall = Self::mla_qcat(d, &qa, &q);
            let mut att = vec![0f32; d.n_heads * d.lat];
            with_k_layer!(cache, l, |kc| {
                KvElem::store(&mut kc[pos * kr..(pos + 1) * kr], &k);
                let kc = &*kc;
                pool::for_each_chunk_mut_in(Kind::Decode, &mut att, d.lat, |hh, o| {
                    with_scores(np + 1, |sc| {
                        let qh = &qall[hh * kr..(hh + 1) * kr];
                        mla_causal(qh, kc, kr, d.lat, np, d.kq_scale, sc, o)
                    })
                });
            });
            let os: Vec<&[f32]> = att.chunks_exact(d.lat).collect();
            let v = self.heads_vec(&ly.wv_b, &os);
            let o = self.matvec(&ly.wo, &v);
            add_assign(&mut x, &o);
            let h = self.rmsnorm(&x, &ly.ffn_norm);
            let y = self.ds2_ffn(ly, &h);
            add_assign(&mut x, &y);
        }
        x
    }

    /// `hidden_batch` of a deepseek2 model: per token the same ops as
    /// `ds2_hidden`, every matrix read once per batch.
    pub(super) fn ds2_hidden_batch(
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
        let ds = self.ds2();
        let d = &ds.d;
        let (dim, kr, hk) = (self.dim, d.krow(), d.hk());
        let mut xs: Vec<Vec<f32>> = par_map(nb, dim, |b| self.embed(tokens[b]));
        let tables: Vec<_> = par_map(nb, d.rope * 16, |b| {
            Self::rope_table_of(&self.freqs, start + b, self.rope_m)
        });
        let mut pending: Option<(Vec<f32>, usize)> = None;
        for (l, ly) in ds.layers.iter().enumerate() {
            let res = pending.take();
            let first = ly.q_a.as_ref().map_or(&ly.wq, |(w, _)| w);
            let hq = self.add_norm_quant(
                &mut xs,
                res.as_ref(),
                &ly.attn_norm,
                &[first.qt(), ly.wkv_a.qt()],
            );
            let (y, starts, slots) = self.matmul_q([first, &ly.wkv_a], &hq);
            drop(hq);
            let n0 = first.rows();
            let mut qs = unbatch(&y[..n0 * slots], slots, nb);
            let cs = unbatch(&y[starts[1] * slots..], slots, nb);
            drop(y);
            if let Some((_, qan)) = &ly.q_a {
                let qn: Vec<Vec<f32>> = par_map(nb, n0, |b| self.rmsnorm(&qs[b], qan));
                let xq = Self::quantize_batch(&qn, &[ly.wq.qt()]);
                let (y, _, slots) = self.matmul_q([&ly.wq], &xq);
                qs = unbatch(&y, slots, nb);
            }
            let ks: Vec<Vec<f32>> = par_map(nb, kr, |b| self.mla_krow(ly, d, &cs[b], &tables[b]));
            super::per_token(&mut qs, hk * d.n_heads, |b, q| {
                Self::mla_qrope(d, q, &tables[b])
            });
            let nopes: Vec<Vec<&[f32]>> = qs
                .iter()
                .map(|q| {
                    (0..d.n_heads)
                        .map(|h| &q[h * hk..h * hk + d.nope])
                        .collect()
                })
                .collect();
            let qa = self.heads_batch(&ly.wk_b, &nopes);
            let qall: Vec<Vec<f32>> =
                par_map(nb, d.n_heads * kr, |b| Self::mla_qcat(d, &qa[b], &qs[b]));
            let w = d.n_heads * d.lat;
            let mut att = vec![0f32; nb * w];
            with_k_layer!(cache, l, |kc| {
                for (b, k) in ks.iter().enumerate() {
                    let p = start + b;
                    KvElem::store(&mut kc[p * kr..(p + 1) * kr], k);
                }
                let kc = &*kc;
                // One task per token, longest first, all heads together
                // (each K row loaded once for them): per head the bits of
                // `ds2_hidden`'s one-head tasks.
                pool::for_each_chunk_mut_rev(&mut att, w, |b, o| {
                    let np = start + b + 1;
                    with_scores((np + 1) * d.n_heads, |sc| {
                        mla_causal(&qall[b], kc, kr, d.lat, np, d.kq_scale, sc, o)
                    })
                });
            });
            let os: Vec<Vec<&[f32]>> = att
                .chunks_exact(w)
                .map(|a| a.chunks_exact(d.lat).collect())
                .collect();
            let vs = self.heads_batch(&ly.wv_b, &os);
            let xq = Self::quantize_batch(&vs, &[ly.wo.qt()]);
            let (o, _, slots) = self.matmul_q([&ly.wo], &xq);
            let res = (o, slots);
            pending = Some(match &ly.ffn {
                Ffn::Dense { gate, up, down } => {
                    let hq = self.add_norm_quant(
                        &mut xs,
                        Some(&res),
                        &ly.ffn_norm,
                        &[gate.qt(), up.qt()],
                    );
                    let (acts, a_slots) = self.matmul_swiglu_q(gate, up, &hq, Vec::new());
                    let aq = Self::quantize_cols(&acts, a_slots, nb, &[down.qt()]);
                    let (y, _, s) = self.matmul_q([down], &aq);
                    (y, s)
                }
                Ffn::Moe(moe) => {
                    let hs = self.add_norm(&mut xs, Some(&res), &ly.ffn_norm);
                    let mut y = self.moe_ffn_batch(moe, &hs, Vec::new());
                    if let Some([g, u, dn]) = &ly.shared {
                        let hq = Self::quantize_batch(&hs, &[g.qt(), u.qt()]);
                        let (acts, a_slots) = self.matmul_swiglu_q(g, u, &hq, Vec::new());
                        let aq = Self::quantize_cols(&acts, a_slots, nb, &[dn.qt()]);
                        let (sy, _, s) = self.matmul_q([dn], &aq);
                        for r in 0..dim {
                            for t in 0..nb {
                                y[r * nb + t] += sy[r * s + t];
                            }
                        }
                    }
                    (y, nb)
                }
            });
        }
        if let Some((y, slots)) = pending {
            add_unbatched(&mut xs, &y, slots);
        }
        xs
    }
}

/// `axon info`'s deepseek2 lines (None for other architectures): the MLA
/// shapes and cache size, the RoPE/attention scale, the routing rule.
pub(crate) fn info_lines(m: &Model) -> Option<Vec<String>> {
    if m.arch != "deepseek2" {
        return None;
    }
    let num = |k: &str| self::num(m, k);
    let n = |k: &str| num(k).map_or(0, |v| v as usize);
    let (layers, heads, lat, rope) = (
        n("block_count"),
        n("attention.head_count"),
        n("attention.kv_lora_rank"),
        n("rope.dimension_count"),
    );
    let (hk, hv) = (
        n("attention.key_length_mla"),
        n("attention.value_length_mla"),
    );
    let mut v = Vec::new();
    if hk == 0 || n("attention.head_count_kv") > 1 {
        v.push(format!("mla: none ({LEGACY})"));
        return Some(v);
    }
    let row = lat + rope;
    let q = match n("attention.q_lora_rank") {
        0 => "direct".to_string(),
        r => format!("q_lora {r}"),
    };
    v.push(format!(
        "mla: {heads} heads, q {hk} (nope {} + rope {rope}), latent {lat}, v {hv}, {q} | K-only \
         cache row {row}: {:.1} KB/token f16 ({layers} layers x {} B)",
        hk.saturating_sub(rope),
        (layers * row * 2) as f64 / 1e3,
        row * 2
    ));
    let (rope_m, kq) = crate::rope::deepseek2_scales(m, hk.max(1));
    let base = num("rope.freq_base").unwrap_or(10000.0);
    let yarn = match crate::rope::yarn(m) {
        None => "no scaling".to_string(),
        Some(y) => format!(
            "yarn x{} (orig {}, beta {}/{}, log_mul {:.4})",
            y.factor,
            y.orig,
            y.beta_fast,
            y.beta_slow,
            num("rope.scaling.yarn_log_multiplier").unwrap_or(0.0)
        ),
    };
    v.push(format!(
        "rope: base {base}, {yarn} | kq_scale {kq:.6} (mscale {:.6}), table x{rope_m:.7}",
        (kq * (hk.max(1) as f32).sqrt()).sqrt()
    ));
    let vocab = find_opt(m, "token_embd.weight")
        .map_or(0, |t| t.dims.get(1).copied().unwrap_or(0) as usize);
    let sig = sigmoid_gating(m, layers, vocab).unwrap_or(false);
    let bias = m.tensors.iter().any(|t| t.kind == "exp_probs_b.bias");
    let scale = num("expert_weights_scale")
        .filter(|&s| s != 0.0)
        .unwrap_or(1.0);
    let norm = flag(m, "expert_weights_norm").unwrap_or(false);
    v.push(format!(
        "routing: {}{}, top-{} of {}, {}, x{scale:.4} | shared {} (width {}) | dense lead {} (ffn {})",
        if sig { "sigmoid" } else { "softmax" },
        if bias { " + exp_probs_b" } else { "" },
        n("expert_used_count"),
        n("expert_count"),
        if norm {
            "renormalized"
        } else {
            "not renormalized"
        },
        n("expert_shared_count"),
        n("expert_shared_count") * n("expert_feed_forward_length"),
        n("leading_dense_block_count"),
        n("feed_forward_length"),
    ));
    Some(v)
}
