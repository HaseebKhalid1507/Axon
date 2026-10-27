//! MoE layers' routed experts resident on the GPU (specs r16 pick 4, PRs B
//! and C): moe_res.cu's kernels, bit for bit the CPU's `moe_layer` /
//! `moe_layer_batch`, so which layers are resident never changes a result.
//!
//! `CudaExperts::enqueue` runs one layer on the stream from device inputs
//! (the router's normed states, ids and weights) into device outputs, with
//! no host sync and no allocation (`ResBufs`, per cache): graph-capturable.
//! nb = 1 takes the decode GEMV (`moe_gemv_tw`, any arch); more, the grouped
//! twin GEMMs (sm_80+). `place` picks the resident layers (`--gpu-experts`).

use super::batch::act_stride;
use super::gemm2_smem;
use crate::cuda::{Arg, DevBuf, Gpu, Module};
use crate::gguf::{Model, Tensor};
use crate::quant::{self, Q4_K, Q5_0, Q8_0};

pub(crate) use super::{twin_gemm_ok, twin_type};

/// Err if layer l of `m` has expert biases (gpt-oss): the CPU adds them
/// (`Moe::act`, `down_bias`); the expert kernels have no bias epilogue yet.
/// `CudaExperts::upload` checks it before uploading anything.
fn refuse_expert_biases(m: &Model, l: usize) -> Result<(), String> {
    for p in ["gate", "up", "down"] {
        let name = format!("blk.{l}.ffn_{p}_exps.bias");
        if m.tensors.iter().any(|t| t.name == name) {
            return Err(format!(
                "{name}: expert biases aren't supported on the GPU yet"
            ));
        }
    }
    Ok(())
}

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
        twin_type(qt).map_err(|e| format!("{}: {e}", t.name))?;
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
    /// The batched path's twin GEMMs: `twin_gemm_ok` of the device.
    mma: Result<(), String>,
    /// F of every swiglu_twin launch of the layer, read once at upload: the
    /// host CPU's `vmath::fused()`, the F its `moe_layer_batch` runs with.
    /// One field, so no launch can pass another (#121 item 4).
    fused: bool,
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
        refuse_expert_biases(m, l)?;
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
            mma: twin_gemm_ok(g.info()?.cc),
            fused: crate::vmath::fused(),
        })
    }

    /// Check a batch's shapes and expert ids; nb.
    #[cfg(test)]
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

    /// The layer's device bytes (its three expert tensors).
    pub(crate) fn bytes(&self) -> usize {
        self.gate.buf.len() + self.up.buf.len() + self.down.buf.len()
    }

    /// The F the layer's swiglu_twin launches pass.
    pub(crate) fn fused(&self) -> bool {
        self.fused
    }

    /// Enqueue the layer's experts of nb tokens: h at `h` (nb x dim f32,
    /// token major), each token's k ids (i32, rank order) at `ids` and
    /// weights (f32) at `w`; y (nb x dim f32, token major) at `y`. nb = 1:
    /// the decode GEMV; more: the grouped twin GEMMs (refused below sm_80).
    /// No sync, no allocation: capturable.
    ///
    /// # Safety
    /// h, ids, w and y hold nb tokens' values, ids in 0..n_e; nb <= rb.max;
    /// rb is this model's (`ResBufs::new` of the same dims), not in use by
    /// another stream.
    pub(crate) unsafe fn enqueue(
        &self,
        rb: &ResBufs,
        h: Arg,
        ids: Arg,
        w: Arg,
        nb: usize,
        y: Arg,
    ) -> Result<(), String> {
        // SAFETY: the caller's.
        unsafe { self.enqueue_path(rb, h, ids, w, nb, y, nb > 1, None) }
    }

    /// The decode step's experts of one token, added into the residual:
    /// `x[r] = x[r] + y[r]` with y `enqueue`'s (moe_sum_add: the weighted sum
    /// then moe.cu moe_add's op, in that order, so the same bits as `enqueue`
    /// then `moe_add`), one launch fewer.
    ///
    /// # Safety
    /// As `enqueue` with nb = 1; x holds dim floats.
    pub(crate) unsafe fn enqueue_add(
        &self,
        rb: &ResBufs,
        h: Arg,
        ids: Arg,
        w: Arg,
        x: Arg,
    ) -> Result<(), String> {
        // SAFETY: the caller's.
        unsafe { self.enqueue_path(rb, h, ids, w, 1, Arg::Ptr(0), false, Some(x)) }
    }

    /// `enqueue`, the grouped GEMMs if `grouped` (any nb), else the decode
    /// GEMV (nb = 1).
    ///
    /// # Safety
    /// As `enqueue`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn enqueue_path(
        &self,
        rb: &ResBufs,
        h: Arg,
        ids: Arg,
        w: Arg,
        nb: usize,
        y: Arg,
        grouped: bool,
        add_to: Option<Arg>,
    ) -> Result<(), String> {
        assert!(grouped || nb == 1, "the decode GEMV takes one token");
        assert!(add_to.is_none() || !grouped, "add_to: the decode path only");
        assert!(
            nb >= 1 && nb <= rb.max,
            "CudaExperts: nb {nb} of {}",
            rb.max
        );
        assert!(rb.dim == self.dim && rb.ffn == self.ffn && rb.k == self.k);
        let (g, km, k, dim, ffn) = (self.g, self.km, self.k, self.dim, self.ffn);
        let (ast, ast2) = (act_stride(dim), act_stride(ffn));
        let f = |n: &str| km.function(n);
        let fused = Arg::I32(self.fused as i32);
        if !grouped {
            // SAFETY (every launch): decode.cu's / moe_res.cu's signatures;
            // rb holds max >= 1 token's scratch, k slots of each.
            unsafe {
                g.launch(
                    &f("quantize")?,
                    (dim.div_ceil(256) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    &[
                        h,
                        Arg::I32(dim as i32),
                        Arg::I32(dim.is_multiple_of(256) as i32),
                        rb.act.arg(),
                    ],
                )?;
                let fv = f("moe_gemv_tw")?;
                let gemv = |mats: &Mats, a: Arg, aslot: usize, out: Arg| {
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
                            ids,
                            a,
                            Arg::U64(aslot as u64),
                            out,
                            Arg::U64(mats.rows as u64),
                        ],
                    )
                };
                // T2b: gate and up in one launch, two rows of each per
                // block (moe_gu_tw), down two rows per block (moe_gemv2_tw):
                // each row's chain unchanged, so the same bits as one row per
                // block (the CPU-equality tests check them at nb 1); the old
                // launches stay for other shapes (gate / up types differ, odd
                // rows).
                let v2 = self.gate.qt == self.up.qt
                    && self.gate.rows.is_multiple_of(2)
                    && self.down.rows.is_multiple_of(2);
                if v2 {
                    let fgu = f("moe_gu_tw")?;
                    g.launch(
                        &fgu,
                        ((self.gate.rows / 2) as u32, k as u32, 1),
                        (128, 1, 1),
                        0,
                        &[
                            Arg::I32(self.gate.qt as i32),
                            self.gate.buf.arg(),
                            self.up.buf.arg(),
                            Arg::U64(self.gate.estride as u64),
                            Arg::U64(self.gate.rb as u64),
                            Arg::I32(self.gate.rows as i32),
                            Arg::I32(self.gate.cols as i32),
                            ids,
                            rb.act.arg(),
                            rb.gb.arg(),
                            rb.ub.arg(),
                        ],
                    )?;
                } else {
                    gemv(&self.gate, rb.act.arg(), 0, rb.gb.arg())?;
                    gemv(&self.up, rb.act.arg(), 0, rb.ub.arg())?;
                }
                g.launch(
                    &f("swiglu_quant_tw")?,
                    (ffn.div_ceil(256) as u32, k as u32, 1),
                    (32, 1, 1),
                    0,
                    &[
                        rb.gb.arg(),
                        rb.ub.arg(),
                        Arg::I32(ffn as i32),
                        fused,
                        Arg::I32(ffn.is_multiple_of(256) as i32),
                        rb.act2.arg(),
                        Arg::U64(ast2 as u64),
                    ],
                )?;
                if v2 {
                    let d = &self.down;
                    g.launch(
                        &f("moe_gemv2_tw")?,
                        ((d.rows / 2) as u32, k as u32, 1),
                        (128, 1, 1),
                        0,
                        &[
                            Arg::I32(d.qt as i32),
                            d.buf.arg(),
                            Arg::U64(d.estride as u64),
                            Arg::U64(d.rb as u64),
                            Arg::I32(d.rows as i32),
                            Arg::I32(d.cols as i32),
                            ids,
                            rb.act2.arg(),
                            Arg::U64(ast2 as u64),
                            rb.db.arg(),
                            Arg::U64(d.rows as u64),
                        ],
                    )?;
                } else {
                    gemv(&self.down, rb.act2.arg(), ast2, rb.db.arg())?;
                }
                // y = the weighted sum (moe_sum), or with `add_to` x += it
                // in the same launch (moe_sum_add: moe_sum's then moe_add's
                // ops, in order).
                let (sum, out) = match add_to {
                    Some(x) => ("moe_sum_add", x),
                    None => ("moe_sum", y),
                };
                g.launch(
                    &f(sum)?,
                    (dim.div_ceil(128) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    &[
                        rb.db.arg(),
                        rb.pos_id.arg(),
                        w,
                        Arg::I32(k as i32),
                        Arg::I32(dim as i32),
                        out,
                    ],
                )?;
            }
            return Ok(());
        }
        self.mma.clone().map_err(|e| format!("CudaExperts: {e}"))?;
        let np = nb * k;
        let max_tiles = np.div_ceil(MOE_BN) + self.n_e;
        // SAFETY (every launch): moe_res.cu's / prefill.cu's signatures; rb
        // holds max >= nb tokens' and max * k pairs' scratch.
        unsafe {
            g.launch(
                &f("quantize_b")?,
                (dim.div_ceil(256) as u32, nb as u32, 1),
                (32, 1, 1),
                0,
                &[
                    h,
                    Arg::I32(dim as i32),
                    Arg::I32(dim.is_multiple_of(256) as i32),
                    rb.act.arg(),
                    Arg::U64(ast as u64),
                ],
            )?;
            g.launch(
                &f("moe_sort")?,
                (1, 1, 1),
                (1024, 1, 1),
                ((self.n_e + 1) * 4) as u32,
                &[
                    ids,
                    Arg::I32(np as i32),
                    Arg::I32(self.n_e as i32),
                    rb.off.arg(),
                    rb.pair_at.arg(),
                    rb.pos_of.arg(),
                    rb.tiles.arg(),
                    rb.ntiles.arg(),
                ],
            )?;
            g.launch(
                &f("moe_gather")?,
                (np as u32, 1, 1),
                (128, 1, 1),
                0,
                &[
                    rb.act.arg(),
                    Arg::U64(ast as u64),
                    rb.pair_at.arg(),
                    Arg::I32(k as i32),
                    rb.acts.arg(),
                ],
            )?;
            for (mats, out) in [(&self.gate, &rb.gb), (&self.up, &rb.ub)] {
                self.grouped(mats, &rb.acts, ast, rb, max_tiles, out)?;
            }
            g.launch(
                &f("swiglu_quant_tw")?,
                (ffn.div_ceil(256) as u32, np as u32, 1),
                (32, 1, 1),
                0,
                &[
                    rb.gb.arg(),
                    rb.ub.arg(),
                    Arg::I32(ffn as i32),
                    fused,
                    Arg::I32(ffn.is_multiple_of(256) as i32),
                    rb.act2.arg(),
                    Arg::U64(ast2 as u64),
                ],
            )?;
            self.grouped(&self.down, &rb.act2, ast2, rb, max_tiles, &rb.db)?;
            g.launch(
                &f("moe_sum")?,
                (dim.div_ceil(128) as u32, nb as u32, 1),
                (128, 1, 1),
                0,
                &[
                    rb.db.arg(),
                    rb.pos_of.arg(),
                    w,
                    Arg::I32(k as i32),
                    Arg::I32(dim as i32),
                    y,
                ],
            )
        }
    }

    /// moe_gemm_tw_* of `mats` over the tile list: out row s (rows floats)
    /// = its expert times act row s (astride bytes).
    ///
    /// # Safety
    /// act holds every sorted pair's row, rb's tiles / ntiles moe_sort's,
    /// out np * mats.rows floats.
    unsafe fn grouped(
        &self,
        mats: &Mats,
        act: &DevBuf,
        astride: usize,
        rb: &ResBufs,
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
                    rb.tiles.arg(),
                    rb.ntiles.arg(),
                    out.arg(),
                    Arg::U64(mats.rows as u64),
                ],
            )
        }
    }

    /// The layer's experts of nb tokens from host slices (tests): h (nb x
    /// dim, token major), each token's k expert ids (rank order) and
    /// weights; y token major. nb = 1: the decode GEMV; more: grouped.
    #[cfg(test)]
    pub(crate) fn layer(&self, h: &[f32], ids: &[i32], w: &[f32]) -> Result<Vec<f32>, String> {
        let nb = self.check(h, ids, w)?;
        Ok(self.run_host(h, ids, w, nb, 1, nb > 1)?.0)
    }

    /// `layer`'s grouped path (any nb, 1 too), its kernels enqueued `reps` times
    /// back to back: y and the GPU time per run in ms (events around the
    /// kernels; uploads and allocations excluded).
    #[cfg(test)]
    pub(crate) fn batched(
        &self,
        h: &[f32],
        ids: &[i32],
        w: &[f32],
        reps: usize,
    ) -> Result<(Vec<f32>, f32), String> {
        let nb = self.check(h, ids, w)?;
        self.run_host(h, ids, w, nb, reps, true)
    }

    /// Upload, `enqueue` reps times (timed), read y back.
    #[cfg(test)]
    fn run_host(
        &self,
        h: &[f32],
        ids: &[i32],
        w: &[f32],
        nb: usize,
        reps: usize,
        grouped: bool,
    ) -> Result<(Vec<f32>, f32), String> {
        let g = self.g;
        let rb = ResBufs::new(g, nb, self.dim, self.ffn, self.k)?;
        let (hb, idb, wb) = (g.upload(h)?, g.upload(ids)?, g.upload(w)?);
        let yb = g.alloc(nb * self.dim * 4)?;
        let reps = reps.max(1);
        let t0 = g.record()?;
        for _ in 0..reps {
            // SAFETY: the buffers above hold nb tokens; ids checked.
            unsafe {
                self.enqueue_path(
                    &rb,
                    hb.arg(),
                    idb.arg(),
                    wb.arg(),
                    nb,
                    yb.arg(),
                    grouped,
                    None,
                )?
            };
        }
        let t1 = g.record()?;
        g.sync()?;
        let ms = t0.ms_to(&t1)? / reps as f32;
        Ok((yb.read()?, ms))
    }

    /// The decode path's `enqueue_add` for one token: x + y in place, read
    /// back (T2b: moe_sum_add).
    #[cfg(test)]
    pub(crate) fn layer_add(
        &self,
        h: &[f32],
        ids: &[i32],
        w: &[f32],
        x: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = self.check(h, ids, w)?;
        assert!(nb == 1 && x.len() == self.dim);
        let g = self.g;
        let rb = ResBufs::new(g, 1, self.dim, self.ffn, self.k)?;
        let (hb, idb, wb, xb) = (g.upload(h)?, g.upload(ids)?, g.upload(w)?, g.upload(x)?);
        // SAFETY: the buffers hold one token; ids checked; xb dim floats.
        unsafe { self.enqueue_add(&rb, hb.arg(), idb.arg(), wb.arg(), xb.arg())? };
        g.sync()?;
        xb.read::<f32>()
    }

    /// `layer` of each token on its own by the decode GEMV, any arch: the
    /// decode path, and nb invariance's reference.
    #[cfg(test)]
    pub(crate) fn layer_gemv(&self, h: &[f32], ids: &[i32], w: &[f32]) -> Result<Vec<f32>, String> {
        let nb = self.check(h, ids, w)?;
        let (dim, k) = (self.dim, self.k);
        let mut y = Vec::with_capacity(nb * dim);
        for t in 0..nb {
            let r = self.run_host(
                &h[t * dim..(t + 1) * dim],
                &ids[t * k..(t + 1) * k],
                &w[t * k..(t + 1) * k],
                1,
                1,
                false,
            )?;
            y.extend(r.0);
        }
        Ok(y)
    }
}

/// The model's resident layers' experts, by layer.
pub(crate) struct Resident<'g> {
    layers: Vec<Option<CudaExperts<'g>>>,
}

impl<'g> Resident<'g> {
    /// `layers` uploaded (global layer indices of a model of n layers).
    pub(crate) fn upload(
        g: &'g Gpu,
        km: &'g Module<'g>,
        m: &Model,
        n: usize,
        layers: &[usize],
        k: usize,
    ) -> Result<Resident<'g>, String> {
        let mut v: Vec<Option<CudaExperts<'g>>> = (0..n).map(|_| None).collect();
        for &l in layers {
            v[l] = Some(CudaExperts::upload(g, km, m, l, k)?);
        }
        Ok(Resident { layers: v })
    }

    pub(crate) fn get(&self, l: usize) -> Option<&CudaExperts<'g>> {
        self.layers.get(l)?.as_ref()
    }

    /// The resident layers.
    pub(crate) fn which(&self) -> Vec<usize> {
        (0..self.layers.len())
            .filter(|&l| self.layers[l].is_some())
            .collect()
    }

    /// Device bytes of the resident experts.
    pub(crate) fn bytes(&self) -> usize {
        self.layers.iter().flatten().map(|e| e.bytes()).sum()
    }

    /// A resident layer's (dim, ffn, k), if any (the scratch's shapes).
    pub(crate) fn dims(&self) -> Option<(usize, usize, usize)> {
        let e = self.layers.iter().flatten().next()?;
        Some((e.dim, e.ffn, e.k))
    }
}

/// `CudaExperts::enqueue`'s scratch for up to `max` tokens of one model's
/// expert shapes (every resident layer of it shares one: they run in order
/// on the stream). Zeroed once: the kernels write the same parts of it on
/// every run, so the padding stays zero.
pub(crate) struct ResBufs<'g> {
    max: usize,
    dim: usize,
    ffn: usize,
    k: usize,
    act: DevBuf<'g>,
    acts: DevBuf<'g>,
    off: DevBuf<'g>,
    pair_at: DevBuf<'g>,
    pos_of: DevBuf<'g>,
    tiles: DevBuf<'g>,
    ntiles: DevBuf<'g>,
    gb: DevBuf<'g>,
    ub: DevBuf<'g>,
    act2: DevBuf<'g>,
    db: DevBuf<'g>,
    /// 0..k (moe_sum's positions for one token).
    pos_id: DevBuf<'g>,
}

/// Most experts per layer the scratch's tile list allows for (moe_sort's
/// worst case adds one partial tile per expert).
const MAX_EXPERTS: usize = 1024;

impl<'g> ResBufs<'g> {
    pub(crate) fn new(
        g: &'g Gpu,
        max: usize,
        dim: usize,
        ffn: usize,
        k: usize,
    ) -> Result<ResBufs<'g>, String> {
        let np = max * k;
        let (ast, ast2) = (act_stride(dim), act_stride(ffn));
        let z = |n: usize| -> Result<DevBuf<'g>, String> {
            let b = g.alloc(n.max(4))?;
            b.zero()?;
            Ok(b)
        };
        Ok(ResBufs {
            max,
            dim,
            ffn,
            k,
            act: z(max * ast)?,
            acts: z(np * ast)?,
            off: z((MAX_EXPERTS + 1) * 4)?,
            pair_at: z(np * 4)?,
            pos_of: z(np * 4)?,
            tiles: z((np.div_ceil(MOE_BN) + MAX_EXPERTS) * 16)?,
            ntiles: z(4)?,
            gb: z(np * ffn * 4)?,
            ub: z(np * ffn * 4)?,
            act2: z(np * ast2)?,
            db: z(np * dim * 4)?,
            pos_id: g.upload(&(0..k as i32).collect::<Vec<i32>>())?,
        })
    }

    /// `new`'s device bytes.
    pub(crate) fn bytes(max: usize, dim: usize, ffn: usize, k: usize) -> usize {
        let np = max * k;
        max * act_stride(dim)
            + np * act_stride(dim)
            + np * (8 + 8 * ffn + act_stride(ffn) + 4 * dim)
            + (np.div_ceil(MOE_BN) + MAX_EXPERTS) * 16
            + (MAX_EXPERTS + 1) * 4
    }
}

/// `--gpu-experts`: how many MoE layers' experts to keep on the GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuExperts {
    /// Exactly this many (0: none, today's behaviour; fewer if they don't fit).
    N(usize),
    /// As many as fit after the context's cache (ctx first).
    Auto,
}

impl GpuExperts {
    /// "auto", or a count.
    pub fn parse(s: &str) -> Result<GpuExperts, String> {
        match s {
            "auto" => Ok(GpuExperts::Auto),
            n => n
                .parse()
                .map(GpuExperts::N)
                .map_err(|_| format!("--gpu-experts {n}: a count or auto")),
        }
    }

    /// $AXON_GPU_EXPERTS, else 0 (the default until the kill-rule bench
    /// passes: today's behaviour).
    pub fn from_env() -> Result<GpuExperts, String> {
        match std::env::var("AXON_GPU_EXPERTS") {
            Ok(v) if !v.is_empty() => Self::parse(&v),
            _ => Ok(GpuExperts::N(0)),
        }
    }
}

/// The resident layers: of `layers` (layer, device bytes), smallest first
/// (ties: the lower layer), as many as `want` asks that fit in `budget`
/// bytes. Sorted by layer.
pub(crate) fn place(layers: &[(usize, usize)], want: GpuExperts, budget: usize) -> Vec<usize> {
    let cap = match want {
        GpuExperts::N(n) => n,
        GpuExperts::Auto => usize::MAX,
    };
    let mut by: Vec<(usize, usize)> = layers.to_vec();
    by.sort_by_key(|&(l, b)| (b, l));
    let mut used = 0;
    let mut out = Vec::new();
    for (l, b) in by {
        if out.len() == cap || used + b > budget {
            break;
        }
        used += b;
        out.push(l);
    }
    out.sort();
    out
}

/// Whether resident experts give the CPU experts' bits here (#121 item 5):
/// the twins are the r8 / r8q kernels' numerics, so the CPU must repack
/// (`AXON_REPACK=0` reads the file's rows in place, other sums), and the
/// twins' F must be the host's `vmath::fused()`. Err says why not.
pub(crate) fn twin_identity(repack: bool, cpu_fused: bool, twin_fused: bool) -> Result<(), String> {
    if !repack {
        return Err(
            "AXON_REPACK=0: the CPU experts aren't the r8 kernels the GPU twins match".into(),
        );
    }
    if cpu_fused != twin_fused {
        return Err(format!(
            "the twins' F ({twin_fused}) isn't the host CPU's vmath::fused() ({cpu_fused})"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "experts_tests.rs"]
mod tests;
