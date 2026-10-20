//! One MoE layer's routed experts resident on the GPU (specs r16 pick 4, PR
//! B): moe_res.cu's kernels, bit for bit the CPU's `moe_layer_batch`. Test
//! only for now (`CudaExperts::layer`); the placement and the model's use of
//! it are PR C.
//!
//! The batched path needs the twin GEMMs (tensor cores, sm_80+) and refuses
//! below; the decode path (`moe_gemv_tw`) runs on every arch.

use super::batch::act_stride;
use super::gemm2_smem;
use crate::cuda::{Arg, DevBuf, Gpu, Module};
use crate::gguf::{Model, Tensor};
use crate::quant::{self, Q4_K, Q5_0, Q6_K, Q8_0};

/// The weight types the twins cover.
const TWIN_TYPES: [u32; 4] = [Q4_K, Q5_0, Q8_0, Q6_K];

/// moe_res.cu's MOE_BN (tokens per grouped-GEMM tile).
const MOE_BN: usize = 64;

/// moe_res.cu's name for a twin type.
fn tw_name(qt: u32) -> &'static str {
    match qt {
        Q4_K => "q4_k",
        Q5_0 => "q5_0",
        Q8_0 => "q8_0",
        _ => "q6_k",
    }
}

/// One of a layer's three expert tensors on the device: n_e matrices of
/// `rows` x `cols`, back to back.
struct Mats<'g> {
    buf: DevBuf<'g>,
    qt: u32,
    rows: usize,
    cols: usize,
    /// Bytes per row, per expert.
    rb: usize,
    estride: usize,
}

impl<'g> Mats<'g> {
    fn upload(g: &'g Gpu, m: &Model, t: &Tensor, n_e: usize) -> Result<Mats<'g>, String> {
        let qt = t.qtype.id;
        if !TWIN_TYPES.contains(&qt) {
            return Err(format!("{}: type {qt} has no GPU twin", t.name));
        }
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        if t.dims.len() != 3 || t.dims[2] as usize != n_e {
            return Err(format!("{}: not {n_e} experts ({:?})", t.name, t.dims));
        }
        let rb = quant::row_bytes(qt, cols);
        let data = m.tensor_data(t);
        assert_eq!(data.len(), rb * rows * n_e, "{}: size", t.name);
        Ok(Mats {
            buf: g.upload(data)?,
            qt,
            rows,
            cols,
            rb,
            estride: rb * rows,
        })
    }
}

/// Layer l's routed experts on the GPU.
pub(crate) struct CudaExperts<'g> {
    g: &'g Gpu,
    km: &'g Module<'g>,
    n_e: usize,
    k: usize,
    dim: usize,
    ffn: usize,
    gate: Mats<'g>,
    up: Mats<'g>,
    down: Mats<'g>,
    /// The batched path's twin GEMMs (sm_80+).
    mma: bool,
}

impl<'g> CudaExperts<'g> {
    /// Layer l's experts of `m` (its `ffn_{gate,up,down}_exps`), k per token.
    pub(crate) fn upload(
        g: &'g Gpu,
        km: &'g Module<'g>,
        m: &Model,
        l: usize,
        k: usize,
    ) -> Result<CudaExperts<'g>, String> {
        let t = |s: &str| {
            let name = format!("blk.{l}.ffn_{s}_exps.weight");
            m.tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| format!("{name}: missing"))
        };
        let tg = t("gate")?;
        let n_e = tg.dims.get(2).copied().unwrap_or(0) as usize;
        let gate = Mats::upload(g, m, tg, n_e)?;
        let up = Mats::upload(g, m, t("up")?, n_e)?;
        let down = Mats::upload(g, m, t("down")?, n_e)?;
        let (dim, ffn) = (gate.cols, gate.rows);
        if up.cols != dim || up.rows != ffn || down.cols != ffn || down.rows != dim {
            return Err(format!("layer {l}: expert shapes don't chain"));
        }
        if gate.qt != up.qt {
            return Err(format!("layer {l}: gate and up of different types"));
        }
        if !(1..=n_e).contains(&k) {
            return Err(format!("layer {l}: k = {k} of {n_e} experts"));
        }
        Ok(CudaExperts {
            g,
            km,
            n_e,
            k,
            dim,
            ffn,
            gate,
            up,
            down,
            mma: g.info()?.cc.0 >= 8,
        })
    }

    /// Check a batch's shapes and expert ids; nb.
    fn check(&self, h: &[f32], ids: &[i32], w: &[f32]) -> Result<usize, String> {
        let nb = h.len() / self.dim;
        if nb == 0 || h.len() != nb * self.dim || ids.len() != nb * self.k || w.len() != ids.len() {
            return Err("CudaExperts: h, ids and w don't match".into());
        }
        if let Some(e) = ids.iter().find(|&&e| e < 0 || e as usize >= self.n_e) {
            return Err(format!("CudaExperts: expert id {e} out of range"));
        }
        Ok(nb)
    }

    /// The layer's experts of nb tokens: h (nb x dim, token major), each
    /// token's k expert ids (rank order) and weights; y token major. nb = 1:
    /// the decode GEMV (any arch); more: the grouped twin GEMMs (sm_80+;
    /// refused below).
    #[cfg(test)]
    pub(crate) fn layer(&self, h: &[f32], ids: &[i32], w: &[f32]) -> Result<Vec<f32>, String> {
        if self.check(h, ids, w)? == 1 {
            return self.layer_gemv(h, ids, w);
        }
        Ok(self.batched(h, ids, w, 1)?.0)
    }

    /// `layer`'s batched path, its kernels run `reps` times back to back
    /// (the same outputs each time): y and the GPU time per run in ms
    /// (events around the kernels; uploads and allocations excluded).
    pub(crate) fn batched(
        &self,
        h: &[f32],
        ids: &[i32],
        w: &[f32],
        reps: usize,
    ) -> Result<(Vec<f32>, f32), String> {
        let nb = self.check(h, ids, w)?;
        if !self.mma {
            return Err("CudaExperts: twin GEMMs need sm_80+ (tensor cores)".into());
        }
        let (g, km, k, dim, ffn) = (self.g, self.km, self.k, self.dim, self.ffn);
        let np = nb * k;
        let (ast, ast2) = (act_stride(dim), act_stride(ffn));
        let hb = g.upload(h)?;
        let idb = g.upload(ids)?;
        let wb = g.upload(w)?;
        let act = g.alloc(nb * ast)?;
        act.zero()?;
        let acts = g.alloc(np * ast)?;
        let off = g.alloc((self.n_e + 1) * 4)?;
        let pair_at = g.alloc(np * 4)?;
        let pos_of = g.alloc(np * 4)?;
        let max_tiles = np.div_ceil(MOE_BN) + self.n_e;
        let tiles = g.alloc(max_tiles * 16)?;
        let ntiles = g.alloc(4)?;
        let gb = g.alloc(np * ffn * 4)?;
        let ub = g.alloc(np * ffn * 4)?;
        let act2 = g.alloc(np * ast2)?;
        act2.zero()?;
        let db = g.alloc(np * dim * 4)?;
        let yb = g.alloc(nb * dim * 4)?;
        let f = |n: &str| km.function(n);
        let reps = reps.max(1);
        let t0 = g.record()?;
        for _ in 0..reps {
            // SAFETY: moe_res.cu's / prefill.cu's signatures; every buffer is
            // sized above for nb tokens and np pairs, and lives past the sync.
            unsafe {
                g.launch(
                    &f("quantize_b")?,
                    (dim.div_ceil(256) as u32, nb as u32, 1),
                    (32, 1, 1),
                    0,
                    &[
                        hb.arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(dim.is_multiple_of(256) as i32),
                        act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )?;
                g.launch(
                    &f("moe_sort")?,
                    (1, 1, 1),
                    (1024, 1, 1),
                    ((self.n_e + 1) * 4) as u32,
                    &[
                        idb.arg(),
                        Arg::I32(np as i32),
                        Arg::I32(self.n_e as i32),
                        off.arg(),
                        pair_at.arg(),
                        pos_of.arg(),
                        tiles.arg(),
                        ntiles.arg(),
                    ],
                )?;
                g.launch(
                    &f("moe_gather")?,
                    (np as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    &[
                        act.arg(),
                        Arg::U64(ast as u64),
                        pair_at.arg(),
                        Arg::I32(k as i32),
                        acts.arg(),
                    ],
                )?;
                for (mats, out) in [(&self.gate, &gb), (&self.up, &ub)] {
                    self.grouped(mats, &acts, ast, &tiles, &ntiles, max_tiles, out)?;
                }
                g.launch(
                    &f("swiglu_quant_tw")?,
                    (ffn.div_ceil(256) as u32, np as u32, 1),
                    (32, 1, 1),
                    0,
                    &[
                        gb.arg(),
                        ub.arg(),
                        Arg::I32(ffn as i32),
                        Arg::I32(crate::vmath::fused() as i32),
                        Arg::I32(ffn.is_multiple_of(256) as i32),
                        act2.arg(),
                        Arg::U64(ast2 as u64),
                    ],
                )?;
                self.grouped(&self.down, &act2, ast2, &tiles, &ntiles, max_tiles, &db)?;
                g.launch(
                    &f("moe_sum")?,
                    (dim.div_ceil(128) as u32, nb as u32, 1),
                    (128, 1, 1),
                    0,
                    &[
                        db.arg(),
                        pos_of.arg(),
                        wb.arg(),
                        Arg::I32(k as i32),
                        Arg::I32(dim as i32),
                        yb.arg(),
                    ],
                )?;
            }
        }
        let t1 = g.record()?;
        g.sync()?;
        let ms = t0.ms_to(&t1)? / reps as f32;
        Ok((yb.read()?, ms))
    }

    /// moe_gemm_tw_* of `mats` over the tile list: out row s (rows floats)
    /// = its expert times act row s (astride bytes).
    ///
    /// # Safety
    /// act holds every sorted pair's row, tiles / ntiles moe_sort's, out
    /// np * mats.rows floats.
    #[allow(clippy::too_many_arguments)]
    unsafe fn grouped(
        &self,
        mats: &Mats,
        act: &DevBuf,
        astride: usize,
        tiles: &DevBuf,
        ntiles: &DevBuf,
        max_tiles: usize,
        out: &DevBuf,
    ) -> Result<(), String> {
        let f = self
            .km
            .function(&format!("moe_gemm_tw_{}", tw_name(mats.qt)))?;
        let sm = gemm2_smem(mats.qt, MOE_BN, 3);
        f.set_max_shared(sm)?;
        // SAFETY: the caller's; moe_gemm_tw's signature.
        unsafe {
            self.g.launch(
                &f,
                (max_tiles as u32, mats.rows.div_ceil(64) as u32, 1),
                (256, 1, 1),
                sm as u32,
                &[
                    mats.buf.arg(),
                    Arg::U64(mats.estride as u64),
                    Arg::U64(mats.rb as u64),
                    Arg::I32(mats.rows as i32),
                    Arg::I32(mats.cols as i32),
                    act.arg(),
                    Arg::U64(astride as u64),
                    tiles.arg(),
                    ntiles.arg(),
                    out.arg(),
                    Arg::U64(mats.rows as u64),
                ],
            )
        }
    }

    /// `layer` of each token on its own by the decode GEMV (moe_gemv_tw),
    /// any arch: the decode path, and nb invariance's reference.
    #[cfg(test)]
    pub(crate) fn layer_gemv(&self, h: &[f32], ids: &[i32], w: &[f32]) -> Result<Vec<f32>, String> {
        let nb = self.check(h, ids, w)?;
        let (g, km, k, dim, ffn) = (self.g, self.km, self.k, self.dim, self.ffn);
        let (ast, ast2) = (act_stride(dim), act_stride(ffn));
        let act = g.alloc(ast)?;
        let act2 = g.alloc(k * ast2)?;
        let gb = g.alloc(k * ffn * 4)?;
        let ub = g.alloc(k * ffn * 4)?;
        let db = g.alloc(k * dim * 4)?;
        let yb = g.alloc(dim * 4)?;
        let pos = g.upload(&(0..k as i32).collect::<Vec<i32>>())?;
        let (fq, fv, fs, fy) = (
            km.function("quantize")?,
            km.function("moe_gemv_tw")?,
            km.function("swiglu_quant_tw")?,
            km.function("moe_sum")?,
        );
        let mut y = Vec::with_capacity(nb * dim);
        for t in 0..nb {
            let hb = g.upload(&h[t * dim..(t + 1) * dim])?;
            let idb = g.upload(&ids[t * k..(t + 1) * k])?;
            let wb = g.upload(&w[t * k..(t + 1) * k])?;
            act.zero()?;
            act2.zero()?;
            let gemv = |mats: &Mats, a: &DevBuf, aslot: usize, out: &DevBuf| {
                // SAFETY: moe_gemv_tw's signature; a holds k slots (or one,
                // aslot 0) of mats.cols inputs, out k * mats.rows floats.
                unsafe {
                    g.launch(
                        &fv,
                        (mats.rows as u32, k as u32, 1),
                        (128, 1, 1),
                        0,
                        &[
                            Arg::I32(mats.qt as i32),
                            mats.buf.arg(),
                            Arg::U64(mats.estride as u64),
                            Arg::U64(mats.rb as u64),
                            Arg::I32(mats.rows as i32),
                            Arg::I32(mats.cols as i32),
                            idb.arg(),
                            a.arg(),
                            Arg::U64(aslot as u64),
                            out.arg(),
                            Arg::U64(mats.rows as u64),
                        ],
                    )
                }
            };
            // SAFETY: decode.cu's / moe_res.cu's signatures; the buffers
            // above, one token.
            unsafe {
                g.launch(
                    &fq,
                    (dim.div_ceil(256) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    &[
                        hb.arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(dim.is_multiple_of(256) as i32),
                        act.arg(),
                    ],
                )?;
            }
            gemv(&self.gate, &act, 0, &gb)?;
            gemv(&self.up, &act, 0, &ub)?;
            // SAFETY: as above.
            unsafe {
                g.launch(
                    &fs,
                    (ffn.div_ceil(256) as u32, k as u32, 1),
                    (32, 1, 1),
                    0,
                    &[
                        gb.arg(),
                        ub.arg(),
                        Arg::I32(ffn as i32),
                        Arg::I32(crate::vmath::fused() as i32),
                        Arg::I32(ffn.is_multiple_of(256) as i32),
                        act2.arg(),
                        Arg::U64(ast2 as u64),
                    ],
                )?;
            }
            gemv(&self.down, &act2, ast2, &db)?;
            // SAFETY: as above.
            unsafe {
                g.launch(
                    &fy,
                    (dim.div_ceil(128) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    &[
                        db.arg(),
                        pos.arg(),
                        wb.arg(),
                        Arg::I32(k as i32),
                        Arg::I32(dim as i32),
                        yb.arg(),
                    ],
                )?;
            }
            g.sync()?;
            y.extend(yb.read::<f32>()?);
        }
        Ok(y)
    }
}

#[cfg(test)]
#[path = "experts_tests.rs"]
mod tests;
