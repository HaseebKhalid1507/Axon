//! deepseek2 on the GPU (r12 stage 1): MLA attention in the absorbed
//! form, the router and the shared experts here, the routed experts on the
//! CPU (`crate::moe`, the #67 split).
//!
//! A deepseek2 `CudaQwen2` has no `layers`: its `mla` holds the MLA layers
//! and `head_dim` is the RoPE width (the step's table). The embedding, the
//! output head and the batch scratch are the dense model's; the cache's
//! `kv` holds one array per layer, [rows][lat + rope] f16 (the normed
//! latent, then k_pe turned), V being each row's first lat (llama.cpp's
//! K-only MLA cache).
//!
//! Per layer and token (mla.cu's kernels; spec r12 §1.1):
//!
//! ```text
//! h = rmsnorm(x)                 q = Wq h (or Wq_b rmsnorm(Wq_a h))   c = Wkv_a h
//! row pos = [rmsnorm(c[..lat]) | rope(c[lat..])]                      (mla_kv_store)
//! Q_h = [Wk_b[h] q_nope_h | rope(q_pe_h)]                             (mla_heads_mv, mla_qrope)
//! o_h = softmax(Q_h . K * kq_scale) . K[..lat]                        (mla_attn)
//! x += Wo [Wv_b[h] o_h]                                               (mla_heads_mv, bgemm)
//! FFN: dense (the lead layers), or router (moe_logits, mla_topk) + shared experts here,
//!      routed experts on the CPU, x += y                              (moe_add)
//! ```
//!
//! Exactness: one code path for decode and batched passes. Every matrix
//! product is `bgemm` (decode at nb = 1: the batched kernels' canonical
//! order, G3), norms and quantization the batched kernels', and every
//! mla.cu kernel works per token with its position from P (decode) or
//! start (a pass): a token gets the same bits alone or in any batch, so
//! decode == feed_batch == verify and cache reuse == a fresh prefill.
//! Against the CPU model (absorbed too, f32) it is graded (tests).

use super::batch::{BatchIn, act_stride, passes};
use super::*;
use crate::cuda::SyncEvent;
use crate::model::{Ds2GpuCfg, MoeDims};

/// mla.cu MLA_TILE (rows per attention tile) and MLA_MV_T (tokens per
/// warp of mla_heads_mv).
const MLA_TILE: usize = 32;
const MLA_MV_T: usize = 8;
/// mla.cu MLA_MAXI: krow <= 64 * MLA_MAXI.
const MLA_MAX_KROW: usize = 576;
/// moe.cu MOE_THREADS / MOE_LG_WARPS.
const MOE_THREADS: u32 = 128;
const MOE_LG_WARPS: usize = 8;

/// One MLA layer's matrices and vectors (offsets in floats into `vecs`).
#[derive(Clone, Copy)]
struct MlaW {
    attn_norm: usize,
    /// q_lora (V3, K2): attn_q_a and attn_q_a_norm; `wq` is then attn_q_b.
    q_a: Option<(Mat, usize)>,
    wq: Mat,
    wkv_a: Mat,
    kv_a_norm: usize,
    /// attn_k_b as one matrix of n_h * lat rows (nope columns), attn_v_b
    /// of n_h * hv rows (lat columns): head h's rows are its slice.
    wk_b: Mat,
    wv_b: Mat,
    wo: Mat,
    ffn_norm: usize,
    ffn: MlaFfn,
}

#[derive(Clone, Copy)]
enum MlaFfn {
    Dense {
        gate: Mat,
        up: Mat,
        down: Mat,
    },
    /// The router [n_expert][dim] f32 and the selection bias in `vecs`,
    /// the shared experts' gate, up, down.
    Moe {
        router: usize,
        bias: Option<usize>,
        shared: Option<(Mat, Mat, Mat)>,
    },
}

/// The MLA half of a deepseek2 `CudaQwen2` (`CudaQwen2::mla`).
pub(crate) struct MlaNet {
    cfg: Ds2GpuCfg,
    layers: Vec<MlaW>,
    /// The widest FFN here (dense lead or shared experts): scratch width.
    ffn_max: usize,
    /// The widest vector quantized into the scratch's activations.
    act_max: usize,
}

impl MlaNet {
    /// The latent cache's bytes for max_ctx positions (f16).
    pub(crate) fn kv_bytes(&self, max_ctx: usize) -> usize {
        self.layers.len() * max_ctx * self.cfg.d.krow() * 2
    }

    /// Heads per attention block (mla.cu mla_attn's hpb).
    fn hpb(&self) -> usize {
        self.cfg.d.n_heads.min(8)
    }
}

/// deepseek2's GPU shapes (None for other architectures), or why it can't
/// run here.
pub(super) fn dims(m: &Model, kv: CacheKv) -> Result<Option<crate::model::Ds2GpuCfg>, String> {
    if m.arch != "deepseek2" {
        return Ok(None);
    }
    let c = crate::model::deepseek2_gpu_cfg(m).map_err(|e| e.to_string())?;
    let d = &c.d;
    if kv != CacheKv::F16 {
        return Err("deepseek2 has no q8 KV cache on the GPU yet (--kv f16)".into());
    }
    let hpb = d.n_heads.min(8);
    if !d.n_heads.is_multiple_of(hpb) || !hpb.is_power_of_two() {
        return Err(format!(
            "deepseek2 with {} heads isn't supported on the GPU (a multiple of 8, or 1, 2, 4)",
            d.n_heads
        ));
    }
    if d.krow() > MLA_MAX_KROW || d.lat % 2 != 0 || d.rope % 2 != 0 {
        return Err(format!(
            "deepseek2 with a {}-wide cache row (latent {}, rope {}) isn't supported on the \
             GPU (even widths, at most {MLA_MAX_KROW})",
            d.krow(),
            d.lat,
            d.rope
        ));
    }
    Ok(Some(c))
}

/// The layers' matrices placed in the arena (`uploads`) and their vectors
/// pushed (`push_vec`): `CudaQwen2::plan`'s deepseek2 part.
pub(super) fn plan_layers<'m>(
    m: &'m Model,
    cfg: Ds2GpuCfg,
    uploads: &mut Vec<(&'m Tensor, usize)>,
    push_vec: &mut dyn FnMut(Vec<f32>) -> usize,
) -> Result<MlaNet, String> {
    let n_layers = meta(m, "block_count").map_or(0, |v| v as usize);
    let d = cfg.d;
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let name = |s: &str| format!("blk.{i}.{s}");
        let mut mat = |s: &str| -> Result<Mat, String> {
            let mt = place(tensor(m, &name(s))?, uploads);
            if ![
                Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, IQ4_NL,
            ]
            .contains(&mt.qt)
            {
                return Err(format!(
                    "tensor {}: type {} isn't supported by the CUDA backend",
                    name(s),
                    crate::gguf::qtype(mt.qt).name
                ));
            }
            Ok(mt)
        };
        let q_a = match d.q_lora {
            Some(_) => Some(mat("attn_q_a.weight")?),
            None => None,
        };
        let wq = mat(if d.q_lora.is_some() {
            "attn_q_b.weight"
        } else {
            "attn_q.weight"
        })?;
        let wkv_a = mat("attn_kv_a_mqa.weight")?;
        let wk_b = mat("attn_k_b.weight")?;
        let wv_b = mat("attn_v_b.weight")?;
        let wo = mat("attn_output.weight")?;
        let ffn_kind = if i < cfg.lead {
            MlaFfn::Dense {
                gate: mat("ffn_gate.weight")?,
                up: mat("ffn_up.weight")?,
                down: mat("ffn_down.weight")?,
            }
        } else {
            let shared = match tensor_opt(m, &name("ffn_gate_shexp.weight")) {
                Some(_) => Some((
                    mat("ffn_gate_shexp.weight")?,
                    mat("ffn_up_shexp.weight")?,
                    mat("ffn_down_shexp.weight")?,
                )),
                None => None,
            };
            MlaFfn::Moe {
                router: 0,
                bias: None,
                shared,
            }
        };
        let mut v = |s: &str| -> Result<Option<usize>, String> {
            match tensor_opt(m, &name(s)) {
                Some(t) => Ok(Some(push_vec(vec_f32(m, t)))),
                None if s.ends_with(".bias") => Ok(None),
                None => Err(format!("missing tensor {}", name(s))),
            }
        };
        let ffn = match ffn_kind {
            MlaFfn::Moe { shared, .. } => MlaFfn::Moe {
                router: v("ffn_gate_inp.weight")?.unwrap(),
                bias: v("exp_probs_b.bias")?,
                shared,
            },
            f => f,
        };
        let mut v = |s: &str| v(s).map(Option::unwrap);
        layers.push(MlaW {
            attn_norm: v("attn_norm.weight")?,
            q_a: match q_a {
                Some(w) => Some((w, v("attn_q_a_norm.weight")?)),
                None => None,
            },
            wq,
            wkv_a,
            kv_a_norm: v("attn_kv_a_norm.weight")?,
            wk_b,
            wv_b,
            wo,
            ffn_norm: v("ffn_norm.weight")?,
            ffn,
        });
    }
    let ffn_max = layers
        .iter()
        .map(|l| match l.ffn {
            MlaFfn::Dense { gate, .. } => gate.rows,
            MlaFfn::Moe { shared, .. } => shared.map_or(0, |s| s.0.rows),
        })
        .max()
        .unwrap_or(0);
    let dim = meta(m, "embedding_length").map_or(0, |v| v as usize);
    let act_max = dim
        .max(ffn_max)
        .max(d.n_heads * d.hv)
        .max(d.q_lora.unwrap_or(0));
    Ok(MlaNet {
        cfg,
        layers,
        ffn_max,
        act_max,
    })
}

/// The kernels the MLA layers launch: mla.cu's (with moe.cu's logits and
/// add, `Kernels::mla_module`) and the dense module's batched norm,
/// quantization and RoPE table.
pub(crate) struct MlaFns<'m> {
    kv_store: Func<'m>,
    qrope: Func<'m>,
    heads_mv: Func<'m>,
    attn: Func<'m>,
    topk: Func<'m>,
    logits: Func<'m>,
    add: Func<'m>,
    norm: Func<'m>,
    quantize: Func<'m>,
    rope_table: Func<'m>,
}

impl<'m> MlaFns<'m> {
    /// `mla`: `Kernels::mla_module`; `dense`: the model's module.
    pub(crate) fn new(mla: &'m Module<'m>, dense: &'m Module<'m>) -> Result<MlaFns<'m>, String> {
        Ok(MlaFns {
            kv_store: mla.function("mla_kv_store")?,
            qrope: mla.function("mla_qrope")?,
            heads_mv: mla.function("mla_heads_mv")?,
            attn: mla.function("mla_attn")?,
            topk: mla.function("mla_topk")?,
            logits: mla.function("moe_logits")?,
            add: mla.function("moe_add")?,
            norm: dense.function("rmsnorm_quant_b")?,
            quantize: dense.function("quantize_b")?,
            rope_table: dense.function("rope_table_b")?,
        })
    }
}

/// A cache's MLA scratch for passes of up to `max` tokens (token major):
/// activations, the query/latent products, absorbed queries, attention
/// output, value heads, FFN gate/up, the pass's RoPE table; the router's
/// output (h [nb][dim], ids [nb][k] i32, w [nb][k], packed for nb), its
/// logits, the experts' y and their pinned host twins; the decode graphs.
pub struct MlaBufs<'g> {
    max: usize,
    ast: usize,
    act: DevBuf<'g>,
    q: DevBuf<'g>,
    ql: DevBuf<'g>,
    c: DevBuf<'g>,
    qa: DevBuf<'g>,
    att: DevBuf<'g>,
    v: DevBuf<'g>,
    gate: DevBuf<'g>,
    up: DevBuf<'g>,
    rope: DevBuf<'g>,
    r: DevBuf<'g>,
    lg: DevBuf<'g>,
    y: DevBuf<'g>,
    host_r: HostBuf<'g>,
    host_y: HostBuf<'g>,
    /// Per segment (`n_layers + 1`): its captured decode graph, then per
    /// layer its shared experts' (`n_layers`).
    graphs: Vec<Option<Graph<'g>>>,
    /// Recorded after the router's copy down (`mla_round_trip`).
    ev: SyncEvent<'g>,
    /// The cache's `graph_gen` when `graphs` were captured.
    graph_gen: u64,
}

/// `b` (4-byte aligned) as floats in place.
fn as_f32(b: &[u8]) -> &[f32] {
    // SAFETY: any 4 bytes are a valid f32.
    let (pre, v, post) = unsafe { b.align_to::<f32>() };
    assert!(pre.is_empty() && post.is_empty(), "f32 alignment");
    v
}

/// `b` (4-byte aligned) as i32s in place.
fn as_i32(b: &[u8]) -> &[i32] {
    // SAFETY: any 4 bytes are a valid i32.
    let (pre, v, post) = unsafe { b.align_to::<i32>() };
    assert!(pre.is_empty() && post.is_empty(), "i32 alignment");
    v
}

/// f32s of little-endian bytes.
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

/// Per-token floats of the scratch (`mla_bufs`), and the activations'
/// stride.
fn per_token(ml: &MlaNet, dim: usize, k: usize, n_e: usize) -> (usize, usize) {
    let d = &ml.cfg.d;
    let floats = d.n_heads * d.hk()
        + d.q_lora.unwrap_or(0)
        + d.krow()
        + d.n_heads * d.krow()
        + d.n_heads * d.lat
        + d.n_heads * d.hv
        + 2 * ml.ffn_max
        + d.rope
        + 2 * (dim + 2 * k)
        + n_e
        + 2 * dim;
    (floats, act_stride(ml.act_max))
}

impl<'g> CudaQwen2<'g> {
    fn ml(&self) -> &MlaNet {
        self.mla.as_deref().expect("a deepseek2 model")
    }

    fn moe_dims(&self) -> MoeDims {
        self.moe.expect("a deepseek2 model's experts")
    }

    /// VRAM (and pinned RAM) of `mla_bufs(max)`; 0 without MLA layers.
    pub(super) fn mla_scratch_bytes(&self, max: usize) -> usize {
        let Some(ml) = &self.mla else { return 0 };
        let e = self.moe_dims();
        let (f, ast) = per_token(ml, self.dim, e.k, e.n_expert);
        max * (4 * f + ast) + 16 * 256
    }

    /// Scratch for passes of up to `max` tokens (the cache's batch max).
    pub(crate) fn mla_bufs(&self, max: usize) -> Result<MlaBufs<'g>, String> {
        let ml = self.ml();
        let d = &ml.cfg.d;
        let e = self.moe_dims();
        let g = self.gpu;
        let f = 4;
        let ast = act_stride(ml.act_max);
        let rb = max * (self.dim + 2 * e.k) * f;
        let yb = max * self.dim * f;
        Ok(MlaBufs {
            max,
            ast,
            act: g.alloc(max * ast)?,
            q: g.alloc(max * d.n_heads * d.hk() * f)?,
            ql: g.alloc(max * d.q_lora.unwrap_or(0) * f)?,
            c: g.alloc(max * d.krow() * f)?,
            qa: g.alloc(max * d.n_heads * d.krow() * f)?,
            att: g.alloc(max * d.n_heads * d.lat * f)?,
            v: g.alloc(max * d.n_heads * d.hv * f)?,
            gate: g.alloc(max * ml.ffn_max * f)?,
            up: g.alloc(max * ml.ffn_max * f)?,
            rope: g.alloc(max * d.rope / 2 * 8)?,
            r: g.alloc(rb)?,
            lg: g.alloc(max * e.n_expert * f)?,
            y: g.alloc(yb)?,
            host_r: g.host_alloc(rb)?,
            host_y: g.host_alloc(yb)?,
            graphs: (0..=2 * self.n_layers).map(|_| None).collect(),
            graph_gen: 0,
            ev: g.new_event()?,
        })
    }

    /// Whether layer li's FFN is the experts' (a host round trip).
    fn mla_is_moe(&self, li: usize) -> bool {
        matches!(self.ml().layers[li].ffn, MlaFfn::Moe { .. })
    }

    /// Layer li's latent cache from position 0, as a kernel argument.
    fn mla_kv(&self, c: &CudaCache, li: usize) -> Arg {
        let kr = self.ml().cfg.d.krow();
        c.kv.arg_at((li * c.rows + c.row0) * kr * 2)
    }

    /// act[z] = rmsnorm(x[z]) * g (n floats per row; rmsnorm_quant_b) for
    /// nb rows.
    ///
    /// # Safety
    /// x holds nb rows of n floats, `g` n floats; nb <= mb.max.
    unsafe fn mla_norm(
        &self,
        mf: &MlaFns,
        mb: &MlaBufs,
        x: Arg,
        g: usize,
        n: usize,
        nb: usize,
    ) -> Result<(), String> {
        // SAFETY: rmsnorm_quant_b's signature; act holds nb rows of ast.
        unsafe {
            self.gpu.launch(
                &mf.norm,
                (nb as u32, 1, 1),
                (256, 1, 1),
                0,
                &[
                    x,
                    self.vecs.arg_at(g * 4),
                    Arg::F32(self.eps),
                    Arg::I32(n as i32),
                    Arg::I32(n.is_multiple_of(256) as i32),
                    mb.act.arg(),
                    Arg::U64(mb.ast as u64),
                ],
            )
        }
    }

    /// act[z] = quantized x[z] (n floats per row) for nb rows.
    ///
    /// # Safety
    /// As `mla_norm`.
    unsafe fn mla_quant(
        &self,
        mf: &MlaFns,
        mb: &MlaBufs,
        x: Arg,
        n: usize,
        nb: usize,
    ) -> Result<(), String> {
        // SAFETY: quantize_b's signature; act holds nb rows of ast.
        unsafe {
            self.gpu.launch(
                &mf.quantize,
                (n.div_ceil(256) as u32, nb as u32, 1),
                (32, 1, 1),
                0,
                &[
                    x,
                    Arg::I32(n as i32),
                    Arg::I32(n.is_multiple_of(256) as i32),
                    mb.act.arg(),
                    Arg::U64(mb.ast as u64),
                ],
            )
        }
    }

    /// out[z] (mt.rows floats per row) = W act[z] (mode 0), += (mode 1)
    /// or silu(gate[z]) * W act[z] (mode 2, gate laid out as out).
    ///
    /// # Safety
    /// act holds nb rows of mt.cols values; out nb rows of mt.rows floats
    /// (mode 1: x, `ostride` dim).
    #[allow(clippy::too_many_arguments)]
    unsafe fn mla_mm(
        &self,
        mb: &MlaBufs,
        mt: &Mat,
        nb: usize,
        bias: Arg,
        mode: i32,
        out: Arg,
        ostride: usize,
    ) -> Result<(), String> {
        // SAFETY: the caller's contract; the matrix is whole in the arena.
        unsafe {
            self.bgemm(
                &self.weights,
                mt,
                mb.act.arg(),
                mb.ast,
                nb,
                bias,
                mode,
                out,
                ostride,
            )
        }
    }

    /// x += down(silu(gate h) * up h), h = rmsnorm(x) * g (a dense FFN:
    /// the lead layers', the shared experts').
    ///
    /// # Safety
    /// x holds nb rows of dim floats; nb <= mb.max.
    #[allow(clippy::too_many_arguments)]
    unsafe fn mla_ffn(
        &self,
        mf: &MlaFns,
        mb: &MlaBufs,
        x: Arg,
        norm: usize,
        (gate, up, down): (&Mat, &Mat, &Mat),
        nb: usize,
    ) -> Result<(), String> {
        let null = Arg::Ptr(0);
        let w = gate.rows;
        // SAFETY: the caller's contract; gate/up hold nb rows of ffn_max >= w.
        unsafe {
            self.mla_norm(mf, mb, x, norm, self.dim, nb)?;
            self.mla_mm(mb, gate, nb, null, 0, mb.gate.arg(), w)?;
            self.mla_mm(mb, up, nb, mb.gate.arg(), 2, mb.up.arg(), w)?;
            self.mla_quant(mf, mb, mb.up.arg(), w, nb)?;
            self.mla_mm(mb, down, nb, null, 1, x, self.dim)
        }
    }

    /// Enqueue layer li over nb tokens of x (rows of dim floats) at
    /// positions pos0.. (pos0 = P's position if `p` isn't null, else
    /// `start`), with their RoPE table at `rope` ([nb][rope / 2] (sin,
    /// cos)): the attention and, for a dense layer, the FFN (x updated);
    /// for a MoE layer the router into mb.r and the shared experts (x +=
    /// theirs): the caller's round trip adds the routed experts.
    ///
    /// # Safety
    /// x holds nb <= mb.max rows; the cache holds positions pos0 + nb.
    #[allow(clippy::too_many_arguments)]
    unsafe fn mla_layer(
        &self,
        mf: &MlaFns,
        c: &CudaCache,
        mb: &MlaBufs,
        li: usize,
        x: Arg,
        nb: usize,
        rope: Arg,
        p: Arg,
        start: usize,
    ) -> Result<(), String> {
        let ml = self.ml();
        let d = &ml.cfg.d;
        let l = &ml.layers[li];
        let (dim, nh, kr) = (self.dim, d.n_heads, d.krow());
        let null = Arg::Ptr(0);
        let vec = |off: usize| self.vecs.arg_at(off * 4);
        let w = |mt: &Mat| self.weights.arg_at(mt.off);
        let kv = self.mla_kv(c, li);
        let g = self.gpu;
        let nbu = nb as u32;
        assert!(nb <= mb.max);
        // SAFETY (every launch): mla.cu's / moe.cu's / prefill.cu's
        // signatures; the scratch holds mb.max >= nb rows of each buffer
        // (mla_bufs), the cache the rows written and read (the caller's
        // contract), matrices are whole in the arena.
        unsafe {
            self.mla_norm(mf, mb, x, l.attn_norm, dim, nb)?;
            match &l.q_a {
                None => self.mla_mm(mb, &l.wq, nb, null, 0, mb.q.arg(), l.wq.rows)?,
                Some((wqa, _)) => self.mla_mm(mb, wqa, nb, null, 0, mb.ql.arg(), wqa.rows)?,
            }
            self.mla_mm(mb, &l.wkv_a, nb, null, 0, mb.c.arg(), kr)?;
            if let Some((wqa, qn)) = &l.q_a {
                self.mla_norm(mf, mb, mb.ql.arg(), *qn, wqa.rows, nb)?;
                self.mla_mm(mb, &l.wq, nb, null, 0, mb.q.arg(), l.wq.rows)?;
            }
            g.launch(
                &mf.kv_store,
                (nbu, 1, 1),
                (256, 1, 1),
                0,
                &[
                    mb.c.arg(),
                    Arg::I32(kr as i32),
                    vec(l.kv_a_norm),
                    Arg::F32(self.eps),
                    Arg::I32(d.lat as i32),
                    Arg::I32(d.rope as i32),
                    rope,
                    p,
                    Arg::I32(start as i32),
                    kv,
                    Arg::I32(kr as i32),
                ],
            )?;
            g.launch(
                &mf.qrope,
                (nbu, 1, 1),
                (256, 1, 1),
                0,
                &[
                    mb.q.arg(),
                    Arg::I32(nh as i32),
                    Arg::I32(d.hk() as i32),
                    Arg::I32(d.nope as i32),
                    Arg::I32(d.rope as i32),
                    rope,
                    mb.qa.arg(),
                    Arg::I32(kr as i32),
                    Arg::I32(d.lat as i32),
                ],
            )?;
            // The absorbed queries: Wk_b[h] q_nope_h.
            let heads = |mt: &Mat, rph: usize, xin: Arg, xs: usize, out: Arg, os: usize| {
                g.launch(
                    &mf.heads_mv,
                    (
                        (nh * rph).div_ceil(8) as u32,
                        nb.div_ceil(MLA_MV_T) as u32,
                        1,
                    ),
                    (256, 1, 1),
                    0,
                    &[
                        Arg::I32(mt.qt as i32),
                        w(mt),
                        Arg::U64(mt.rb as u64),
                        Arg::I32(rph as i32),
                        Arg::I32(mt.cols as i32),
                        Arg::I32(nh as i32),
                        xin,
                        Arg::I32(xs as i32),
                        Arg::U64((nh * xs) as u64),
                        out,
                        Arg::I32(os as i32),
                        Arg::U64((nh * os) as u64),
                        Arg::I32(nb as i32),
                    ],
                )
            };
            heads(&l.wk_b, d.lat, mb.q.arg(), d.hk(), mb.qa.arg(), kr)?;
            let hpb = ml.hpb();
            g.launch(
                &mf.attn,
                ((nh / hpb) as u32, nbu, 1),
                (256, 1, 1),
                (MLA_TILE * kr * 2) as u32,
                &[
                    mb.qa.arg(),
                    kv,
                    Arg::I32(kr as i32),
                    Arg::I32(d.lat as i32),
                    Arg::I32(nh as i32),
                    Arg::I32(hpb as i32),
                    Arg::F32(d.kq_scale),
                    p,
                    Arg::I32(start as i32),
                    mb.att.arg(),
                ],
            )?;
            heads(&l.wv_b, d.hv, mb.att.arg(), d.lat, mb.v.arg(), d.hv)?;
            self.mla_quant(mf, mb, mb.v.arg(), nh * d.hv, nb)?;
            self.mla_mm(mb, &l.wo, nb, null, 1, x, dim)?;
            match &l.ffn {
                MlaFfn::Dense { gate, up, down } => {
                    self.mla_ffn(mf, mb, x, l.ffn_norm, (gate, up, down), nb)?;
                }
                MlaFfn::Moe {
                    router,
                    bias,
                    shared,
                } => {
                    let e = self.moe_dims();
                    let (n_e, k) = (e.n_expert, e.k);
                    let hb = nb * dim * 4;
                    g.launch(
                        &mf.logits,
                        (n_e.div_ceil(MOE_LG_WARPS) as u32, nbu, 1),
                        ((MOE_LG_WARPS * 32) as u32, 1, 1),
                        (dim * 4) as u32,
                        &[
                            x,
                            vec(l.ffn_norm),
                            Arg::F32(self.eps),
                            Arg::I32(dim as i32),
                            vec(*router),
                            Arg::I32(n_e as i32),
                            mb.r.arg(),
                            mb.lg.arg(),
                        ],
                    )?;
                    let cfg = &ml.cfg;
                    g.launch(
                        &mf.topk,
                        (nbu, 1, 1),
                        (MOE_THREADS, 1, 1),
                        (2 * n_e * 4) as u32,
                        &[
                            mb.lg.arg(),
                            Arg::I32(n_e as i32),
                            Arg::I32(k as i32),
                            Arg::I32(cfg.sigmoid as i32),
                            bias.map_or(null, vec),
                            Arg::I32(cfg.norm as i32),
                            Arg::F32(cfg.scale),
                            mb.r.arg_at(hb),
                            mb.r.arg_at(hb + nb * k * 4),
                        ],
                    )?;
                    let _ = shared; // mla_shared, after the router's copy down
                }
            }
        }
        Ok(())
    }

    /// Enqueue MoE layer li's shared experts over nb tokens of x (x +=
    /// theirs), after its `mla_layer` (the same x: the FFN's input). The
    /// round trip enqueues them behind the router's copy down, so they run
    /// while the CPU computes the routed experts.
    ///
    /// # Safety
    /// As `mla_layer`.
    unsafe fn mla_shared(
        &self,
        mf: &MlaFns,
        mb: &MlaBufs,
        li: usize,
        x: Arg,
        nb: usize,
    ) -> Result<(), String> {
        let l = &self.ml().layers[li];
        if let MlaFfn::Moe {
            shared: Some((sg, su, sd)),
            ..
        } = &l.ffn
        {
            // SAFETY: the caller's contract.
            unsafe { self.mla_ffn(mf, mb, x, l.ffn_norm, (sg, su, sd), nb)? };
        }
        Ok(())
    }

    /// Enqueue x += mb.y for nb tokens.
    ///
    /// # Safety
    /// x holds nb rows of dim floats; nb <= mb.max.
    unsafe fn mla_add(&self, mf: &MlaFns, mb: &MlaBufs, x: Arg, nb: usize) -> Result<(), String> {
        assert!(nb <= mb.max);
        let n = nb * self.dim;
        // SAFETY: moe_add's signature; both hold n floats.
        unsafe {
            self.gpu.launch(
                &mf.add,
                (n.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &[x, mb.y.arg(), Arg::I32(n as i32)],
            )
        }
    }

    /// The host round trip of MoE layer li after its `mla_layer`
    /// (enqueued): the router's output down, then `shared` (the shared
    /// experts, enqueued behind the copy: they run meanwhile), the CPU
    /// experts once the copy is in, y up (enqueued, ordered before what
    /// comes next on the stream).
    fn mla_round_trip(
        &self,
        mb: &mut MlaBufs<'g>,
        li: usize,
        nb: usize,
        shared: &mut dyn FnMut(&mut MlaBufs<'g>) -> Result<(), String>,
        experts: &mut super::moe::Experts,
    ) -> Result<(), String> {
        let k = self.moe_dims().k;
        let dim = self.dim;
        let n = nb * (dim + 2 * k) * 4;
        mb.host_r.copy_from(&mb.r, 0, n)?;
        self.gpu.record_main(&mb.ev)?;
        shared(mb)?;
        mb.ev.sync()?; // the layer and the copy down, not the shared experts
        // SAFETY: the event fired after the copy down and, earlier on the
        // stream, host_y's last copy up (the previous layer's y): neither
        // buffer has a copy in flight.
        unsafe {
            mb.host_r.assume_synced();
            mb.host_y.assume_synced();
        }
        let b = &mb.host_r.bytes_mut()?[..n];
        let (hb, rest) = b.split_at(nb * dim * 4);
        let (ib, wb) = rest.split_at(nb * k * 4);
        let (h, ids, w) = (as_f32(hb), as_i32(ib), as_f32(wb));
        let yb = &mut mb.host_y.bytes_mut()?[..nb * dim * 4];
        // SAFETY: any bytes are a valid f32; the prefix is empty (asserted).
        let (pre, y, _) = unsafe { yb.align_to_mut::<f32>() };
        assert!(pre.is_empty(), "pinned buffer alignment");
        experts(self.first + li, h, ids, w, y);
        mb.host_y.copy_to(&mb.y, 0, nb * dim * 4)
    }

    /// One decode step of `token` at `pos` (the logits and argmax into the
    /// cache), the routed experts by `experts`. Waits for every MoE
    /// layer's router; the head is enqueued, not waited for.
    pub(crate) fn mla_step(
        &self,
        mf: &MlaFns,
        c: &mut CudaCache<'g>,
        mb: &mut MlaBufs<'g>,
        token: u32,
        pos: usize,
        experts: &mut super::moe::Experts,
    ) -> Result<(), String> {
        assert_eq!(c.net, self.id, "a CudaCache used with another CudaQwen2");
        assert!(
            pos < c.max_ctx,
            "position {pos} exceeds cache size {}",
            c.max_ctx
        );
        assert!(c.tab.is_none(), "deepseek2: no paged KV on the GPU yet");
        assert!((token as usize) < self.vocab, "token {token} out of range");
        let p = c.host_params.bytes_mut()?;
        p[0..4].copy_from_slice(&token.to_le_bytes());
        p[4..8].copy_from_slice(&(pos as u32).to_le_bytes());
        c.host_params.copy_to(&c.params, 0, 8)?;
        let n = self.n_layers;
        if mb.graph_gen != c.graph_gen {
            mb.graphs.iter_mut().for_each(|g| *g = None);
            mb.graph_gen = c.graph_gen;
        }
        for seg in 0..=n {
            let c = &*c;
            let run = |mb: &MlaBufs| -> Result<(), String> {
                let params = c.params.arg();
                let on = &mut |_: &'static str, _: usize| Ok(());
                // SAFETY: x holds dim floats, mb one token's scratch; the
                // cache holds pos + 1 positions (checked).
                unsafe {
                    if seg == 0 {
                        self.enqueue_input(c, params, on)?;
                    } else if self.mla_is_moe(seg - 1) {
                        self.mla_add(mf, mb, c.x.arg(), 1)?;
                    }
                    if seg < n {
                        self.mla_layer(mf, c, mb, seg, c.x.arg(), 1, c.rope.arg(), params, 0)?;
                    } else {
                        self.enqueue_head(c, Head::Argmax, on)?;
                    }
                }
                Ok(())
            };
            if self.graphs {
                if mb.graphs[seg].is_none() {
                    let g = self.gpu.capture(|| run(mb))?;
                    mb.graphs[seg] = Some(g);
                }
                // SAFETY: the graph's buffers are the cache's, the scratch's
                // and the model's, all borrowed for this call; the scratch
                // owns the graph and lives with the cache.
                unsafe { mb.graphs[seg].as_ref().unwrap().launch()? };
            } else {
                run(mb)?;
            }
            if seg < n && self.mla_is_moe(seg) {
                let gi = n + 1 + seg;
                let mut shared = |mb: &mut MlaBufs<'g>| -> Result<(), String> {
                    // SAFETY: as `run`'s.
                    let run = |mb: &MlaBufs| unsafe { self.mla_shared(mf, mb, seg, c.x.arg(), 1) };
                    if !self.graphs {
                        return run(mb);
                    }
                    if mb.graphs[gi].is_none() {
                        let g = self.gpu.capture(|| run(mb))?;
                        mb.graphs[gi] = Some(g);
                    }
                    // SAFETY: as the segment's graph.
                    unsafe { mb.graphs[gi].as_ref().unwrap().launch() }
                };
                self.mla_round_trip(mb, seg, 1, &mut shared, experts)?;
            }
        }
        Ok(())
    }

    /// `feed_batch` (head No), `verify_argmax` (Argmax: the picks) or
    /// `prefill_logits` (Last: the logits) of `tokens` at start.., in
    /// passes of up to the cache's batch max.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_tokens(
        &self,
        mf: &MlaFns,
        c: &mut CudaCache<'g>,
        mb: &mut MlaBufs<'g>,
        tokens: &[u32],
        start: usize,
        head: Head,
        experts: &mut super::moe::Experts,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        assert!(
            matches!(head, Head::No | Head::Argmax | Head::Last),
            "mla_tokens: {head:?}"
        );
        self.check_batch(tokens, start, c);
        assert!(c.tab.is_none(), "deepseek2: no paged KV on the GPU yet");
        let max = c.b.as_ref().expect("batch scratch").max.min(mb.max);
        let ps = passes(tokens.len(), max);
        let embd = self.embd.expect("deepseek2: the whole model is here");
        let ast_b = act_stride(self.act_len());
        let half = self.head_dim / 2;
        let null = Arg::Ptr(0);
        let mut picks = Vec::new();
        for (i, r) in ps.iter().enumerate() {
            let h = match head {
                Head::Last if i + 1 < ps.len() => Head::No,
                h => h,
            };
            let nb = r.len();
            let s0 = start + r.start;
            self.load_input(c.b.as_mut().unwrap(), BatchIn::Tokens(&tokens[r.clone()]))?;
            let b = c.b.as_ref().unwrap();
            let (g, bf) = (self.gpu, &self.bf);
            // SAFETY (every launch): prefill.cu's signatures; b and mb hold
            // nb <= their max rows; the cache holds s0 + nb positions
            // (check_batch).
            unsafe {
                g.launch(
                    &bf.embed,
                    (self.dim.div_ceil(256) as u32, nb as u32, 1),
                    (256, 1, 1),
                    0,
                    &[
                        Arg::I32(embd.qt as i32),
                        self.weights.arg_at(embd.off),
                        Arg::U64(embd.rb as u64),
                        Arg::I32(self.dim as i32),
                        b.tokens.arg(),
                        b.x.arg(),
                    ],
                )?;
                g.launch(
                    &mf.rope_table,
                    (nb as u32, 1, 1),
                    (half as u32, 1, 1),
                    0,
                    &[
                        self.freq.arg(),
                        Arg::I32(half as i32),
                        Arg::I32(s0 as i32),
                        mb.rope.arg(),
                        null,
                        Arg::F32(self.rope_m),
                    ],
                )?;
            }
            for li in 0..self.n_layers {
                let b = c.b.as_ref().unwrap();
                // SAFETY: as above.
                unsafe {
                    self.mla_layer(mf, c, mb, li, b.x.arg(), nb, mb.rope.arg(), null, s0)?;
                }
                if self.mla_is_moe(li) {
                    let x = b.x.arg();
                    // SAFETY: as above.
                    let mut shared =
                        |mb: &mut MlaBufs<'g>| unsafe { self.mla_shared(mf, mb, li, x, nb) };
                    self.mla_round_trip(mb, li, nb, &mut shared, experts)?;
                    // SAFETY: as above.
                    unsafe { self.mla_add(mf, mb, b.x.arg(), nb)? };
                }
            }
            let b = c.b.as_ref().unwrap();
            // SAFETY: as above; b.act holds nb rows of ast_b, b.logits
            // HEAD_SUB rows of vocab floats, b.best nb picks.
            unsafe {
                if h == Head::Last {
                    self.norm_rows(b, &self.vecs, self.output_norm, nb - 1, 1)?;
                    self.bgemm(
                        &self.weights,
                        &self.output,
                        b.act.arg(),
                        ast_b,
                        1,
                        null,
                        0,
                        b.logits.arg(),
                        self.vocab,
                    )?;
                } else if h == Head::Argmax {
                    self.norm_b(b, &self.vecs, self.output_norm, nb)?;
                    let v = self.vocab;
                    for z0 in (0..nb).step_by(batch::HEAD_SUB) {
                        let nz = (nb - z0).min(batch::HEAD_SUB);
                        self.bgemm(
                            &self.weights,
                            &self.output,
                            b.act.arg_at(z0 * ast_b),
                            ast_b,
                            nz,
                            null,
                            0,
                            b.logits.arg(),
                            v,
                        )?;
                        g.launch(
                            &bf.argmax,
                            (nz as u32, 1, 1),
                            (1024, 1, 1),
                            0,
                            &[b.logits.arg(), Arg::I32(v as i32), b.best.arg_at(z0 * 4)],
                        )?;
                    }
                }
            }
            if h == Head::Argmax {
                picks.extend(c.b.as_mut().unwrap().picks(nb)?);
            }
        }
        let logits = match head {
            Head::Last => {
                let n = self.vocab * 4;
                let b = c.b.as_ref().unwrap();
                c.host_out.copy_from(&b.logits, 0, n)?;
                f32s(&c.host_out.bytes_mut()?[..n])
            }
            _ => {
                self.gpu.sync()?;
                Vec::new()
            }
        };
        Ok((picks, logits))
    }
}

#[cfg(test)]
#[path = "mla_tests.rs"]
mod tests;
