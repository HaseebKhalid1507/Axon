//! Mixture-of-experts FFN (`qwen3moe`) on the CPU.
//!
//! A MoE layer replaces the gated FFN with `n_expert` small ones and a
//! router. Per token: router logits (an f32 `[n_expert x dim]` matrix times
//! the normed hidden state), softmax over all of them, the top `k`
//! (ties to the lower expert index), their probabilities renormalized to
//! sum to 1 (llama.cpp's `norm_w`, hardcoded for this arch), and the output
//! sum_i w_i * down_{id_i}(silu(gate_{id_i} h) * up_{id_i} h).
//!
//! Exactness. The router runs per token with one fixed op order (logits
//! accumulated over the input in ascending order, multiply then add; the
//! softmax sum in ascending expert order; the renorm by division), so a
//! token's experts and weights don't depend on what else is in its batch.
//! Each expert matrix is its own `Weight` (an r8 repack of that slice of
//! the 3-D tensor, or its bytes in place), so the expert dots are the dense
//! kernels' (`r8::gemm` per token == `r8::gemv`). The weighted sum is in
//! rank order, `y = w_0 d_0; y = y + w_i d_i`, in both paths, per output
//! row on one thread: no reduction across threads. Hence decode ==
//! `feed_batch` == verify, bit for bit.
//!
//! Decode takes two pool fork/joins per layer: gate+up+SwiGLU over the k
//! selected experts' rows, then down + the weighted sum over output rows.
//! A batch groups its (token, slot) pairs by expert (counting sort, in
//! ascending pair order) and runs each expert's rows once over its tokens.

use super::{Kind, Pool, QuantizedAct, Qwen2, Weight, par_map, pool, with_rows_scratch};
use crate::gguf::{Model, Tensor};

/// llama.cpp's clamp of the renormalization sum (`build_moe_ffn`).
const W_SUM_MIN: f32 = 6.103_515_6e-5;

/// A batch's (token, slot) pairs are processed this many tokens at a time:
/// the scratch (k x (ffn + dim) floats per token) stays bounded. Bits don't
/// depend on it (every token is routed and computed on its own).
const MOE_BATCH: usize = 512;

/// One layer's experts and router.
pub(super) struct Moe<'a> {
    pub(super) n_expert: usize,
    /// Experts used per token (top-k).
    pub(super) k: usize,
    /// Rows of each expert's gate/up matrix (`expert_feed_forward_length`).
    pub(super) ffn: usize,
    /// The router transposed: `router_t[c * n_expert + e]` = W[e][c].
    router_t: Vec<f32>,
    pub(super) gate: Vec<Weight<'a>>,
    pub(super) up: Vec<Weight<'a>>,
    pub(super) down: Vec<Weight<'a>>,
}

/// The MoE hyperparameters of a model (`qwen3moe.*`), None for a dense one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MoeDims {
    pub n_expert: usize,
    pub k: usize,
    pub ffn: usize,
}

pub(crate) fn is_moe(m: &Model) -> bool {
    m.arch == "qwen3moe"
}

pub(crate) fn moe_dims(m: &Model) -> Option<MoeDims> {
    if !is_moe(m) {
        return None;
    }
    let num = |k: &str| super::meta_num(m, &format!("{}.{k}", m.arch)).map(|v| v as usize);
    Some(MoeDims {
        n_expert: num("expert_count")?,
        k: num("expert_used_count")?,
        ffn: num("expert_feed_forward_length")?,
    })
}

/// The expert tensors of a layer, in the arena's order.
pub(super) const EXPS: [&str; 3] = [
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
];

/// Bytes of expert `e`'s matrix in a `[cols, rows, n_expert]` tensor.
pub(super) fn expert_bytes<'m>(m: &'m Model, t: &Tensor, e: usize) -> &'m [u8] {
    let n = t.dims[2] as usize;
    let per = t.n_bytes as usize / n;
    &m.tensor_data(t)[e * per..(e + 1) * per]
}

/// Bytes of one expert matrix of `t` as the CPU holds it: repacked if it
/// has an r8/r8q form, else as in the file.
fn held_bytes(t: &Tensor) -> u64 {
    super::repacked_len(t).map_or(t.n_bytes / t.dims[2], |l| l as u64)
}

/// What `axon info` says about a MoE model: sizes from the tensor table.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MoeInfo {
    pub dims: MoeDims,
    pub shared: usize,
    /// All expert tensors, as in the file and as the CPU holds them (r8).
    pub expert_bytes: u64,
    pub expert_held: u64,
    /// k experts of every layer: what one token reads.
    pub active_bytes: u64,
    pub active_held: u64,
    /// Everything else (attention, routers, norms, embedding, head).
    pub dense_bytes: u64,
}

impl MoeInfo {
    pub fn of(m: &Model) -> Option<MoeInfo> {
        let dims = moe_dims(m)?;
        let mut i = MoeInfo {
            dims,
            shared: super::meta_num(m, &format!("{}.expert_shared_count", m.arch))
                .map_or(0, |v| v as usize),
            expert_bytes: 0,
            expert_held: 0,
            active_bytes: 0,
            active_held: 0,
            dense_bytes: 0,
        };
        for t in &m.tensors {
            if EXPS.iter().any(|s| t.name.ends_with(s)) && t.dims.len() >= 3 && t.dims[2] > 0 {
                let n = t.dims[2];
                i.expert_bytes += t.n_bytes;
                i.expert_held += held_bytes(t) * n;
                i.active_bytes += t.n_bytes / n * dims.k as u64;
                i.active_held += held_bytes(t) * dims.k as u64;
            } else {
                i.dense_bytes += t.n_bytes;
            }
        }
        Some(i)
    }

    /// Two lines for `axon info`.
    pub fn lines(&self) -> [String; 2] {
        let gb = |b: u64| format!("{:.2} GB", b as f64 / 1e9);
        let d = self.dims;
        [
            format!(
                "moe: {} experts, top-{}, expert ffn {}, {} shared | CPU only for now",
                d.n_expert,
                d.k,
                d.ffn,
                if self.shared == 0 {
                    "none".to_string()
                } else {
                    self.shared.to_string()
                }
            ),
            format!(
                "experts {} B ({}; r8 {}) | active/token {} B ({}; r8 {}) | non-expert {} B ({})",
                self.expert_bytes,
                gb(self.expert_bytes),
                gb(self.expert_held),
                self.active_bytes,
                gb(self.active_bytes),
                gb(self.active_held),
                self.dense_bytes,
                gb(self.dense_bytes)
            ),
        ]
    }
}

/// The CPU model's RAM beyond the file: refuse a MoE model that wouldn't fit
/// in `MemAvailable` (the r8 arena plus 1.5 GiB) instead of swapping.
pub(crate) fn check_ram(m: &Model) -> Result<(), String> {
    let Some(i) = MoeInfo::of(m) else {
        return Ok(());
    };
    let want = i.expert_held + i.dense_bytes + (3 << 29);
    match crate::gguf::mem_available() {
        Some(avail) if avail < want => Err(format!(
            "{}: the CPU model needs about {:.1} GB of RAM (its repacked experts {:.1} GB) and \
             only {:.1} GB is available",
            m.arch,
            want as f64 / 1e9,
            i.expert_held as f64 / 1e9,
            avail as f64 / 1e9
        )),
        _ => Ok(()),
    }
}

/// The refusal of any GPU placement for now.
pub(crate) fn gpu_refusal(m: &Model) -> String {
    format!(
        "{}: mixture-of-experts models run on the CPU only for now (--device cpu); \
         attention on the GPU with the experts on the CPU is not implemented yet",
        m.arch
    )
}

/// Largest multiple of `align` that is at most `target` (at least `align`)
/// and divides `n`; `align` itself if none does.
fn chunk_dividing(n: usize, target: usize, align: usize) -> usize {
    let mut c = target.max(align).min(n) / align * align;
    while c > align && !n.is_multiple_of(c) {
        c -= align;
    }
    c.max(align)
}

/// exp with `vmath`'s own polynomial (fused as the CPU's other paths are).
#[inline]
fn exp(x: f32) -> f32 {
    if crate::vmath::fused() {
        crate::vmath::exp_f::<true>(x)
    } else {
        crate::vmath::exp_f::<false>(x)
    }
}

/// `y = w d` for the first expert, `y = y + w d` for the rest: the
/// weighted sum's one op order (decode and batch).
#[inline(always)]
fn combine(y: &mut f32, w: f32, d: f32, first: bool) {
    *y = if first { w * d } else { *y + w * d };
}

impl<'a> Moe<'a> {
    pub(super) fn new(
        dims: MoeDims,
        router: Vec<f32>,
        gate: Vec<Weight<'a>>,
        up: Vec<Weight<'a>>,
        down: Vec<Weight<'a>>,
    ) -> Moe<'a> {
        let n = dims.n_expert;
        let dim = router.len() / n;
        let mut router_t = vec![0f32; n * dim];
        for e in 0..n {
            for c in 0..dim {
                router_t[c * n + e] = router[e * dim + c];
            }
        }
        Moe {
            n_expert: n,
            k: dims.k,
            ffn: dims.ffn,
            router_t,
            gate,
            up,
            down,
        }
    }

    /// Every expert matrix, gate then up then down, expert order: the
    /// arena's order.
    pub(super) fn weights_mut(&mut self) -> impl Iterator<Item = &mut Weight<'a>> {
        self.gate
            .iter_mut()
            .chain(self.up.iter_mut())
            .chain(self.down.iter_mut())
    }

    pub(super) fn weights(&self) -> impl Iterator<Item = &Weight<'a>> {
        self.gate.iter().chain(&self.up).chain(&self.down)
    }

    /// Row granularity of pool chunks over `ws`' rows: 8 if repacked
    /// (whole r8 groups).
    fn align(ws: &[&Weight]) -> usize {
        if ws.iter().any(|w| w.r8.is_some()) {
            8
        } else {
            1
        }
    }

    /// `align` of the gate/up rows (stage A) and of the down rows (B).
    fn aligns(&self) -> (usize, usize) {
        (
            Self::align(&[&self.gate[0], &self.up[0]]),
            Self::align(&[&self.down[0]]),
        )
    }

    /// The token's experts in rank order and their weights: softmax over the
    /// router logits of `h`, the top k (higher probability first, the lower
    /// index on ties), renormalized by their sum (at least `W_SUM_MIN`).
    pub(super) fn route(&self, h: &[f32], ids: &mut [usize], w: &mut [f32]) {
        let n = self.n_expert;
        let mut p = vec![0f32; n];
        // logit_e = sum over c ascending of W[e][c] * h[c], multiply then
        // add (elementwise across e: the vector code gives the same bits).
        for (c, &x) in h.iter().enumerate() {
            let row = &self.router_t[c * n..(c + 1) * n];
            for (l, &wv) in p.iter_mut().zip(row) {
                *l += wv * x;
            }
        }
        let max = p.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0f32;
        for v in p.iter_mut() {
            *v = exp(*v - max);
            sum += *v;
        }
        for v in p.iter_mut() {
            *v /= sum;
        }
        let mut taken = vec![false; n];
        for (id, wi) in ids.iter_mut().zip(w.iter_mut()) {
            let mut best = usize::MAX;
            for e in 0..n {
                if !taken[e] && (best == usize::MAX || p[e] > p[best]) {
                    best = e;
                }
            }
            taken[best] = true;
            *id = best;
            *wi = p[best];
        }
        let mut s = 0f32;
        for &v in w.iter() {
            s += v;
        }
        let s = s.max(W_SUM_MIN);
        for v in w.iter_mut() {
            *v /= s;
        }
    }
}

impl<'a> Qwen2<'a> {
    /// The MoE FFN of one token's normed hidden state `h`: two fork/joins.
    pub(super) fn moe_ffn(&self, moe: &Moe, h: &[f32]) -> Vec<f32> {
        let (k, ffn, dim) = (moe.k, moe.ffn, self.dim);
        let mut ids = vec![0usize; k];
        let mut w = vec![0f32; k];
        moe.route(h, &mut ids, &mut w);
        let xq = QuantizedAct::for_types(h, &[moe.gate[0].qt(), moe.up[0].qt()]);
        let (align_a, align_b) = moe.aligns();
        let threads = Pool::global().threads();
        let (cfg, own_all) = Self::decode_cfg();

        // Stage A: rows of the k experts' gate and up, SwiGLU, into act
        // (slot i's ffn rows at i * ffn). Chunks never straddle two slots.
        const MAX_A: usize = 32;
        let chunk_a = chunk_dividing(ffn, (k * ffn / (threads * 8)).clamp(4, MAX_A), align_a);
        let mut act = vec![0f32; k * ffn];
        pool::for_each_chunk_mut_in(Kind::Decode, &mut act, chunk_a, |c, ys| {
            let g = c * chunk_a;
            let (e, r) = (ids[g / ffn], g % ffn);
            let mut ub = [0f32; MAX_A];
            let ub = &mut ub[..ys.len()];
            self.rows_vec(&moe.gate[e], r, &xq, ys, cfg, own_all);
            self.rows_vec(&moe.up[e], r, &xq, ub, cfg, own_all);
            self.glu(ys, ub);
        });
        let dt = [moe.down[0].qt()];
        let aq: Vec<QuantizedAct> = act
            .chunks_exact(ffn)
            .map(|a| QuantizedAct::for_types(a, &dt))
            .collect();

        // Stage B: per output row chunk, the k experts' down rows in rank
        // order, summed with their weights as they come.
        const MAX_B: usize = 64;
        let chunk_b = (dim / (threads * 4))
            .clamp(8, MAX_B)
            .next_multiple_of(align_b)
            .min(MAX_B);
        let mut y = vec![0f32; dim];
        pool::for_each_chunk_mut_in(Kind::Decode, &mut y, chunk_b, |c, ys| {
            let r0 = c * chunk_b;
            let mut d = [0f32; MAX_B];
            let d = &mut d[..ys.len()];
            for (i, &e) in ids.iter().enumerate() {
                self.rows_vec(&moe.down[e], r0, &aq[i], d, cfg, own_all);
                for (yv, &dv) in ys.iter_mut().zip(d.iter()) {
                    combine(yv, w[i], dv, i == 0);
                }
            }
        });
        y
    }

    /// `moe_ffn` for a batch of normed hidden states: the output in (row,
    /// token slot) layout, y[r * nb + t] (`out` recycled), each token's
    /// column bit for bit `moe_ffn` of it.
    pub(super) fn moe_ffn_batch(&self, moe: &Moe, hs: &[Vec<f32>], out: Vec<f32>) -> Vec<f32> {
        let nb = hs.len();
        let dim = self.dim;
        let mut y = super::recycle(out, dim * nb);
        for t0 in (0..nb).step_by(MOE_BATCH) {
            let t1 = (t0 + MOE_BATCH).min(nb);
            self.moe_sub_batch(moe, &hs[t0..t1], &mut y, nb, t0);
        }
        y
    }

    /// Tokens t0..t0 + hs.len() of `moe_ffn_batch` into their columns of y
    /// (`slots` columns).
    fn moe_sub_batch(&self, moe: &Moe, hs: &[Vec<f32>], y: &mut [f32], slots: usize, t0: usize) {
        let nb = hs.len();
        let (k, ffn, dim, n_e) = (moe.k, moe.ffn, self.dim, moe.n_expert);
        let types = [moe.gate[0].qt(), moe.up[0].qt()];
        // Route and quantize each token as `moe_ffn` does.
        let routed: Vec<(Vec<usize>, Vec<f32>, QuantizedAct)> =
            par_map(nb, n_e * dim / 8 + dim, |t| {
                let mut ids = vec![0usize; k];
                let mut w = vec![0f32; k];
                moe.route(&hs[t], &mut ids, &mut w);
                (ids, w, QuantizedAct::for_types(&hs[t], &types))
            });
        // Pairs p = t * k + i grouped by expert, ascending p within each.
        let mut off = vec![0usize; n_e + 1];
        for (ids, _, _) in &routed {
            for &e in ids {
                off[e + 1] += 1;
            }
        }
        for e in 0..n_e {
            off[e + 1] += off[e];
        }
        let mut pairs = vec![0usize; nb * k];
        let mut at = off.clone();
        for (t, (ids, _, _)) in routed.iter().enumerate() {
            for (i, &e) in ids.iter().enumerate() {
                pairs[at[e]] = t * k + i;
                at[e] += 1;
            }
        }
        let (align_a, align_b) = moe.aligns();
        let used: Vec<usize> = (0..n_e).filter(|&e| off[e + 1] > off[e]).collect();

        // Stage A: act[p * ffn + r] = SwiGLU of expert e's rows for pair p.
        let rows_a = chunk_dividing(ffn, 32, align_a);
        let items_a: Vec<(usize, usize)> = used
            .iter()
            .flat_map(|&e| (0..ffn).step_by(rows_a).map(move |r| (e, r)))
            .collect();
        let mut act = vec![0f32; nb * k * ffn];
        let base = pool::SendPtr(act.as_mut_ptr());
        Pool::global().run(items_a.len(), &|c| {
            let (e, r0) = items_a[c];
            let ps = &pairs[off[e]..off[e + 1]];
            let n = ps.len();
            let (gate, up) = (&moe.gate[e], &moe.up[e]);
            let slots = Self::slots_for(&[gate, up], dim, n);
            let xr: Vec<&QuantizedAct> = (0..slots)
                .map(|b| &routed[ps[b.min(n - 1)] / k].2)
                .collect();
            with_rows_scratch(2 * rows_a * slots, |sc| {
                let (gs, us) = sc.split_at_mut(rows_a * slots);
                self.rows_batch(gate, r0, &xr, n, gs);
                self.rows_batch(up, r0, &xr, n, us);
                for (gr, ur) in gs.chunks_mut(slots).zip(us.chunks(slots)) {
                    self.glu(&mut gr[..n], &ur[..n]);
                }
                for (b, &p) in ps.iter().enumerate() {
                    for r in 0..rows_a {
                        // SAFETY: pair p belongs to expert e alone and rows
                        // r0..r0 + rows_a to this item alone: disjoint
                        // writes inside `act`, alive for the whole run.
                        unsafe { *base.get().add(p * ffn + r0 + r) = gs[r * slots + b] };
                    }
                }
            });
        });
        let dt = [moe.down[0].qt()];
        let aq: Vec<QuantizedAct> = par_map(nb * k, ffn, |p| {
            QuantizedAct::for_types(&act[p * ffn..(p + 1) * ffn], &dt)
        });
        drop(act);

        // Stage B: d[p * dim + r] = expert e's down rows for pair p.
        let rows_b = chunk_dividing(dim, 32, align_b);
        let items_b: Vec<(usize, usize)> = used
            .iter()
            .flat_map(|&e| (0..dim).step_by(rows_b).map(move |r| (e, r)))
            .collect();
        let mut d = vec![0f32; nb * k * dim];
        let base = pool::SendPtr(d.as_mut_ptr());
        Pool::global().run(items_b.len(), &|c| {
            let (e, r0) = items_b[c];
            let ps = &pairs[off[e]..off[e + 1]];
            let n = ps.len();
            let down = &moe.down[e];
            let slots = Self::slots_for(&[down], ffn, n);
            let xr: Vec<&QuantizedAct> = (0..slots).map(|b| &aq[ps[b.min(n - 1)]]).collect();
            let rows = rows_b.min(dim - r0);
            with_rows_scratch(rows * slots, |ds| {
                self.rows_batch(down, r0, &xr, n, ds);
                for (b, &p) in ps.iter().enumerate() {
                    for r in 0..rows {
                        // SAFETY: as in stage A.
                        unsafe { *base.get().add(p * dim + r0 + r) = ds[r * slots + b] };
                    }
                }
            });
        });

        // Stage C: the weighted sums, rank order, per (row, token).
        let rows_c = 16;
        pool::for_each_chunk_mut(y, rows_c * slots, |c, ys| {
            for (j, yr) in ys.chunks_exact_mut(slots).enumerate() {
                let r = c * rows_c + j;
                for (t, (_, w, _)) in routed.iter().enumerate() {
                    let yv = &mut yr[t0 + t];
                    for (i, &wi) in w.iter().enumerate() {
                        combine(yv, wi, d[(t * k + i) * dim + r], i == 0);
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_dividing_divides() {
        assert_eq!(chunk_dividing(768, 32, 8), 32);
        assert_eq!(chunk_dividing(768, 30, 8), 24);
        assert_eq!(chunk_dividing(3072, 4, 8), 8);
        assert_eq!(chunk_dividing(100, 32, 1), 25);
        assert_eq!(chunk_dividing(7, 32, 1), 7);
        assert_eq!(chunk_dividing(8, 32, 8), 8);
    }
}
