//! Batched passes (prefill.cu): `feed_batch` and `verify_argmax` run nb
//! tokens through each layer together, so each weight row is read once per
//! batch instead of once per token.
//!
//! The contract is the CPU's: every token's KV cache rows and logits are
//! bit for bit what the one-token step (`CudaQwen2::step`) gives, whatever
//! the batch size or the token's place in it; prefill.cu's header says how.
//! So chat's cache reuse equals a fresh prefill, and drafting is lossless.
//!
//! Products of more than `SMALL` tokens run on int8 tensor cores (gemm2_*,
//! sm_80+); smaller ones (the verify pass of a draft) too, by gemvm_*, which
//! reads the weights once for the batch like a decode step. On GPUs without
//! int8 mma (sm < 80) gemvn, the decode GEMV with a token loop (dp4a), does
//! every batch, 16 tokens per pass over the weights: correct, not fast.
//!
//! The attention of a batch (sm_80+) is the decode kernels' canonical order
//! (decode.cu "Tensor-core attention"): one flash-attention launch per layer
//! (attn_fa_b) for the whole pass, or the decode kernel per query
//! (attn_blk_b) for a few queries (`att_per_query`).

use super::*;

/// Batches of up to this many tokens use gemvn (weight-read-once, CUDA
/// cores) even where the tensor-core GEMM exists.
pub(super) const SMALL: usize = 16;
/// Query tokens per batched attention launch without tensor cores (sizes
/// `part_o`).
pub(super) const ATT_SUB: usize = 64;

/// Batches of up to this many queries take the decode attention per query
/// (attn_blk_b) rather than the flash kernel (attn_fa_b), which walks a
/// block's positions in order and so is latency-bound with few blocks of
/// rows: the crossover measured by cuda_attn_bench (RTX 5070 Ti Laptop: 64
/// wide heads ~48-64 queries, 128 wide ~16-24). Either gives the same bits.
pub(super) fn att_per_query(head_dim: usize) -> usize {
    if head_dim == 64 { 48 } else { SMALL }
}
/// Tokens per output-head launch in `verify_argmax` (sizes the logits).
pub(super) const HEAD_SUB: usize = 16;

/// attn_fa_b's variants for head size hd and q8 scale group kg (0: the
/// f16 cache): (rows per block, threads, dynamic shared memory: f16, its
/// stages of K and V tiles of ATT_BLK rows; q8, one f16 tile and per stage
/// the raw rows and their scales). `small`: attn_fa_bs (half the rows,
/// more blocks, for short batches).
pub(super) fn fa_shape(hd: usize, small: bool, kg: usize) -> (usize, usize, usize) {
    let (rows, stages) = match (hd, small, kg) {
        (64, false, _) => (64, 3),
        (64, true, _) => (32, 2),
        (_, false, _) => (128, 2),
        (_, true, 0) => (64, 1),
        (_, true, _) => (64, 2),
    };
    let smem = match kg {
        0 => stages * 2 * ATT_BLK * hd * 2,
        _ => ATT_BLK * hd * 2 + stages * 2 * ATT_BLK * (hd + hd / kg * 4),
    };
    (rows, rows * 2, smem)
}

/// Bytes per token of a batched activation buffer: a decode.cu Act, padded
/// so every token's starts 256-byte aligned (the GEMMs' 16-byte copies).
pub(super) fn act_stride(n: usize) -> usize {
    act_bytes(n).next_multiple_of(256)
}

/// What a batched pass starts from: tokens (first == 0), or nb hidden
/// states of dim floats each, token-major (first > 0; see `Input`).
#[derive(Clone, Copy)]
pub(crate) enum BatchIn<'x> {
    Tokens(&'x [u32]),
    Hidden(&'x [f32]),
    /// This many states already in the batch scratch's residual stream,
    /// and the RoPE table of their positions in its `rope` (left there by
    /// the streamed layers, stream.rs).
    Resident(usize),
}

impl BatchIn<'_> {
    fn len(&self, dim: usize) -> usize {
        match self {
            BatchIn::Tokens(t) => t.len(),
            BatchIn::Hidden(x) => x.len() / dim,
            BatchIn::Resident(n) => *n,
        }
    }

    /// Tokens r.
    fn range(&self, r: std::ops::Range<usize>, dim: usize) -> Self {
        match self {
            BatchIn::Tokens(t) => BatchIn::Tokens(&t[r]),
            BatchIn::Hidden(x) => BatchIn::Hidden(&x[r.start * dim..r.end * dim]),
            BatchIn::Resident(n) => {
                assert!(r == (0..*n), "resident states: one pass");
                BatchIn::Resident(*n)
            }
        }
    }
}

/// The batched kernels of a module.
pub(super) struct BatchFns<'m> {
    embed: Func<'m>,
    rope_table: Func<'m>,
    rmsnorm_quant: Func<'m>,
    swiglu_quant: Func<'m>,
    rope_kv: Func<'m>,
    /// rope_kv with qwen3's per-head q/k norms.
    rope_kv_norm: Func<'m>,
    /// The split decode attention per query (attn_split_t: the q8 cache's
    /// kernels' f16 twin, for GPUs without tensor cores), and its combine.
    attn_split: Func<'m>,
    attn_combine: Func<'m>,
    /// The tensor-core attention per query (decode.cu attn_blk_t), and its
    /// fold: batches of at most att_per_query tokens.
    attn_blk: Func<'m>,
    attn_cmb: Func<'m>,
    /// attn_fa_b and attn_fa_bs (fa_shape) for this head size (larger
    /// batches), then quantize_b.
    attn_fa: [Func<'m>; 2],
    quantize: Func<'m>,
    argmax: Func<'m>,
    gemvn8: Func<'m>,
    gemvn16: Func<'m>,
    /// gemvm8_*, gemvm16_* per type (as `gemm`); None without tensor cores.
    gemvm: Option<[[Func<'m>; 4]; 2]>,
    /// Per type (Q4_K, Q6_K, Q8_0, Q5_0): gemm2_* (128 tokens per block)
    /// and gemm2s_* (64); None without int8 tensor cores.
    gemm: Option<[[Func<'m>; 2]; 4]>,
    /// gemm2_q6k8 and gemm2s_q6k8 (Q6_K by its prefill copy, `Mat::c8`),
    /// and q6k_to_i8 which makes it; None without int8 tensor cores.
    gemm_q6k8: Option<[Func<'m>; 2]>,
    /// gemm2s_q4_k_gu: gate and up (both Q4_K) in one launch, silu(gate) *
    /// up out (`CudaQwen2::gate_up_fused`).
    pub(super) gemm_gu: Option<Func<'m>>,
    pub(super) q6k_to_i8: Option<Func<'m>>,
    /// SMs of the GPU (the GEMM tile choice).
    sms: usize,
    /// stream.cu: r8/r8q blocks back to GGUF rows.
    pub(super) unpermute: Func<'m>,
}

impl<'m> BatchFns<'m> {
    /// Whether the module has the tensor-core kernels (sm_80+).
    #[cfg(test)]
    pub(super) fn has_mma(&self) -> bool {
        self.gemm.is_some()
    }

    pub(super) fn new(
        module: &'m Module<'m>,
        head_dim: usize,
        sms: usize,
    ) -> Result<BatchFns<'m>, String> {
        let gemm = if module.mma() {
            let mut fs = Vec::new();
            for (qt, tn) in [
                (Q4_K, "q4_k"),
                (Q6_K, "q6_k"),
                (Q8_0, "q8_0"),
                (Q5_0, "q5_0"),
            ] {
                let big = module.function(&format!("gemm2_{tn}"))?;
                big.set_max_shared(gemm2_smem(qt, 128, 2))?;
                let small = module.function(&format!("gemm2s_{tn}"))?;
                small.set_max_shared(gemm2_smem(qt, 64, 3))?;
                fs.push([big, small]);
            }
            let f: [[Func<'m>; 2]; 4] = fs.try_into().map_err(|_| "gemm2 kernels")?;
            Some(f)
        } else {
            None
        };
        let gemm_q6k8 = if module.mma() {
            let big = module.function("gemm2_q6k8")?;
            big.set_max_shared(gemm2_smem(Q6_K, 128, 2))?;
            let small = module.function("gemm2s_q6k8")?;
            small.set_max_shared(gemm2_smem(Q6_K, 64, 3))?;
            Some([big, small])
        } else {
            None
        };
        let gemvm = if module.mma() {
            let f = |nf: usize| -> Result<[Func<'m>; 4], String> {
                let n = |t: &str| module.function(&format!("gemvm{nf}_{t}"));
                Ok([n("q4_k")?, n("q6_k")?, n("q8_0")?, n("q5_0")?])
            };
            Some([f(8)?, f(16)?])
        } else {
            None
        };
        Ok(BatchFns {
            embed: module.function("embed_b")?,
            rope_table: module.function("rope_table_b")?,
            rmsnorm_quant: module.function("rmsnorm_quant_b")?,
            swiglu_quant: module.function("swiglu_quant_b")?,
            rope_kv: module.function("rope_kv_b")?,
            rope_kv_norm: module.function("rope_kv_norm_b")?,
            attn_split: module.function(if head_dim == 64 {
                "attn_split_b_64"
            } else {
                "attn_split_b_128"
            })?,
            attn_combine: module.function("attn_combine_b")?,
            attn_blk: module.function(if head_dim == 64 {
                "attn_blk_b_64"
            } else {
                "attn_blk_b_128"
            })?,
            attn_cmb: module.function("attn_cmb_b")?,
            attn_fa: {
                let (a, b) = (
                    module.function(&format!("attn_fa_b_{head_dim}"))?,
                    module.function(&format!("attn_fa_bs_{head_dim}"))?,
                );
                if module.mma() {
                    a.set_max_shared(fa_shape(head_dim, false, 0).2)?;
                    b.set_max_shared(fa_shape(head_dim, true, 0).2)?;
                }
                [a, b]
            },
            quantize: module.function("quantize_b")?,
            argmax: module.function("argmax_b")?,
            gemvn8: module.function("gemvn_2x8")?,
            gemvn16: module.function("gemvn_2x16")?,
            gemvm,
            gemm,
            gemm_q6k8,
            gemm_gu: if module.mma() {
                let f = module.function("gemm2s_q4_k_gu")?;
                f.set_max_shared(gemm2_smem(Q4_K, 64, 3))?;
                Some(f)
            } else {
                None
            },
            q6k_to_i8: if module.mma() {
                Some(module.function("q6k_to_i8")?)
            } else {
                None
            },
            sms,
            unpermute: module.function("unpermute_r8")?,
        })
    }
}

/// A cache's scratch for batched passes of up to `max` tokens.
pub(super) struct BatchBufs<'g> {
    pub(super) max: usize,
    /// Blocks of ATT_BLK positions of the cache it's for (the stride of
    /// attn_blk_b's partials).
    nblk_max: usize,
    tokens: DevBuf<'g>,
    host_tokens: HostBuf<'g>,
    /// Incoming hidden states, [max][dim] (first > 0 only).
    host_x: Option<HostBuf<'g>>,
    /// Residual stream, [max][dim].
    x: DevBuf<'g>,
    /// Products of the q, k, v matrices (biases added), [max][q_dim + 2 kv_dim].
    qkv: DevBuf<'g>,
    /// Rotated queries, [max][q_dim].
    q: DevBuf<'g>,
    gate: DevBuf<'g>,
    up: DevBuf<'g>,
    /// Quantized activations, [max][act_stride(act_len())].
    act: DevBuf<'g>,
    rope: DevBuf<'g>,
    part_o: DevBuf<'g>,
    part_ml: DevBuf<'g>,
    /// HEAD_SUB rows, reused per head launch; with `logits_rows` (tests:
    /// `keep_logits`) one row per token of the pass instead.
    pub(super) logits: DevBuf<'g>,
    pub(super) logits_rows: bool,
    best: DevBuf<'g>,
    host_best: HostBuf<'g>,
}

impl<'g> CudaQwen2<'g> {
    /// Tokens per batched pass a cache of `max_ctx` positions gets scratch for.
    pub(super) fn batch_for(&self, max_ctx: usize) -> usize {
        self.batch.min(max_ctx).max(1)
    }

    /// Partials (per head, hd + 2 floats each) of the batched attention:
    /// att_per_query queries' blocks of ATT_BLK positions (attn_blk_b), or
    /// without tensor cores ATT_SUB queries' splits (attn_split_b).
    fn batch_parts(&self, max: usize, max_ctx: usize) -> usize {
        match self.tc_attn {
            true => max.min(att_per_query(self.head_dim)) * Self::nblk_max(max_ctx),
            false => max.min(ATT_SUB) * self.max_split,
        }
    }

    /// VRAM of the batch scratch for `max` tokens of a cache of max_ctx
    /// positions.
    pub(super) fn batch_bytes(&self, max: usize, max_ctx: usize) -> usize {
        let (f, kvd) = (4, self.kv_dim());
        let parts = self.n_heads * self.batch_parts(max, max_ctx) * (self.head_dim + 2) * f;
        let per_tok = (self.dim + 2 * self.q_dim + 2 * kvd + 2 * self.ffn + self.head_dim) * f
            + act_stride(self.act_len())
            + 8
            + if self.first > 0 { self.dim * f } else { 0 };
        max * per_tok + parts + max.min(HEAD_SUB) * self.vocab * f + 256
    }

    pub(super) fn batch_bufs(&self, max: usize, max_ctx: usize) -> Result<BatchBufs<'g>, String> {
        let (g, f, kvd) = (self.gpu, 4, self.kv_dim());
        let parts = self.n_heads * self.batch_parts(max, max_ctx);
        Ok(BatchBufs {
            max,
            nblk_max: Self::nblk_max(max_ctx),
            tokens: g.alloc(max * 4)?,
            host_tokens: g.host_alloc(max * 4)?,
            host_x: match self.first {
                0 => None,
                _ => Some(g.host_alloc(max * self.dim * f)?),
            },
            x: g.alloc(max * self.dim * f)?,
            qkv: g.alloc(max * (self.q_dim + 2 * kvd) * f)?,
            q: g.alloc(max * self.q_dim * f)?,
            gate: g.alloc(max * self.ffn * f)?,
            up: g.alloc(max * self.ffn * f)?,
            act: g.alloc(max * act_stride(self.act_len()))?,
            rope: g.alloc(max * self.head_dim / 2 * 8)?,
            part_o: g.alloc(parts * self.head_dim * f)?,
            part_ml: g.alloc(parts * 2 * f)?,
            logits: g.alloc(max.min(HEAD_SUB) * self.vocab * f)?,
            logits_rows: false,
            best: g.alloc(max * 4)?,
            host_best: g.host_alloc(max * 4)?,
        })
    }

    /// out[z][r] (row stride ostride floats) = W act[z] (+ bias) for mode 0
    /// (W at mt.off in `wts`), += for mode 1, silu(bias[z][r]) * W act[z]
    /// for mode 2 (bias: the gate product, laid out as out; prefill.cu
    /// gm_store), for tokens z < nb: gemvn for small batches (or without
    /// tensor cores), else the type's GEMM (Q6_K: by its prefill copy if
    /// made, `Mat::c8`). Same bits either way.
    ///
    /// # Safety
    /// act holds nb tokens of mt.cols inputs at astride bytes apart; out has
    /// room for nb rows of ostride floats (mt.rows <= ostride); bias is null
    /// or has mt.rows floats (mode 2: nb rows of ostride floats); the matrix
    /// is whole in `wts`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn bgemm(
        &self,
        wts: &DevBuf,
        mt: &Mat,
        act: Arg,
        astride: usize,
        nb: usize,
        bias: Arg,
        mode: i32,
        out: Arg,
        ostride: usize,
    ) -> Result<(), String> {
        let bf = &self.bf;
        // Q6_K by its prefill copy where made (the same products).
        let q8c = match (mt.c8, &bf.gemm_q6k8) {
            (Some(c8), Some(f)) if nb > SMALL => Some((c8, f)),
            _ => None,
        };
        let (wa, rb) = match q8c {
            Some((c8, _)) => (self.w8.arg_at(c8), mt.cols / 256 * Q6K8_BYTES),
            None => (wts.arg_at(mt.off), mt.rb),
        };
        let common = [
            wa,
            Arg::U64(rb as u64),
            Arg::I32(mt.rows as i32),
            Arg::I32(mt.cols as i32),
            act,
            Arg::U64(astride as u64),
            Arg::I32(nb as i32),
            bias,
            Arg::I32(mode),
            out,
            Arg::U64(ostride as u64),
        ];
        let ti = match mt.qt {
            Q4_K => 0,
            Q6_K => 1,
            Q8_0 => 2,
            _ => 3,
        };
        // SAFETY: the caller's contract; the matrix is whole in `wts`.
        unsafe {
            match (&bf.gemm, &bf.gemvm) {
                (Some(fs), _) if nb > SMALL => {
                    // Measured (cuda_gemm_bench): Q4_K is fastest in 64 x 64
                    // tiles (three stages, one barrier each) at every size;
                    // the register-staged types in 64 x 128, given at least
                    // ~1.5 blocks per SM.
                    let ctas = nb.div_ceil(128) * mt.rows.div_ceil(G2_BM);
                    let big = mt.qt != Q4_K && 2 * ctas >= 3 * bf.sms;
                    let (bn, st) = if big { (128, 2) } else { (64, 3) };
                    let (f, smem) = match q8c {
                        Some((_, f8)) => (&f8[!big as usize], gemm2_smem(Q6_K, bn, st)),
                        None => (&fs[ti][!big as usize], gemm2_smem(mt.qt, bn, st)),
                    };
                    self.gpu.launch(
                        f,
                        (nb.div_ceil(bn) as u32, mt.rows.div_ceil(G2_BM) as u32, 1),
                        (256, 1, 1),
                        smem as u32,
                        &common,
                    )
                }
                (_, Some(fs)) if nb <= SMALL => self.gpu.launch(
                    &fs[(nb > 8) as usize][ti],
                    (mt.rows.div_ceil(16) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    &common,
                ),
                _ => {
                    let (f, t) = if nb <= 8 {
                        (&bf.gemvn8, 8)
                    } else {
                        (&bf.gemvn16, 16)
                    };
                    let mut args = vec![Arg::I32(mt.qt as i32)];
                    args.extend_from_slice(&common);
                    self.gpu.launch(
                        f,
                        (mt.rows.div_ceil(2) as u32, nb.div_ceil(t) as u32, 1),
                        (GEMV_THREADS, 1, 1),
                        0,
                        &args,
                    )
                }
            }
        }
    }

    /// Tokens at positions start.. (nb <= the cache's batch max) through
    /// every layer, their K/V into the cache; with `head`, each one's greedy
    /// pick into b.best. Enqueued, not waited for.
    pub(super) fn pass(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        start: usize,
        head: bool,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
    ) -> Result<(), String> {
        self.pass_with(c, input, start, head.into(), 0, on)
    }

    /// `pass` with any `Head`: Sample puts token z's candidates in out row
    /// z, with penalty row `row0` + z of S (`samp_upload` first).
    pub(super) fn pass_with(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        start: usize,
        head: Head,
        row0: usize,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
    ) -> Result<(), String> {
        let nb = input.len(self.dim);
        let dim = self.dim;
        let kvas: Vec<Vec<Arg>> = (0..self.layers.len()).map(|l| self.kv_args(c, l)).collect();
        let mode = c.mode;
        let b = c.b.as_mut().expect("a cache with batch scratch");
        assert!(nb > 0 && nb <= b.max);
        self.load_input(b, input)?;
        let b = &*b;
        let (g, bf) = (self.gpu, &self.bf);
        let w = |mt: &Mat| self.weights.arg_at(mt.off);
        let null = Arg::Ptr(0);
        let ast = act_stride(self.act_len());
        // SAFETY (every launch): arguments match prefill.cu's signatures;
        // the batch buffers hold b.max >= nb tokens of what each kernel
        // reads and writes (sizes in batch_bufs); positions start..start + nb
        // are in the cache (the callers check); matrices are whole in the arena.
        unsafe {
            if let (Some(embd), BatchIn::Tokens(_)) = (&self.embd, input) {
                g.launch(
                    &bf.embed,
                    (dim.div_ceil(256) as u32, nb as u32, 1),
                    (256, 1, 1),
                    0,
                    &[
                        Arg::I32(embd.qt as i32),
                        w(embd),
                        Arg::U64(embd.rb as u64),
                        Arg::I32(dim as i32),
                        b.tokens.arg(),
                        b.x.arg(),
                    ],
                )?;
                on("embed")?;
            }
            // Resident states come with their RoPE table (stream.rs
            // computed it for the same positions).
            if !matches!(input, BatchIn::Resident(_)) {
                self.rope_b(b, nb, start)?;
                on("rope")?;
            }
            for (li, l) in self.layers.iter().enumerate() {
                self.layer(
                    b,
                    l,
                    &self.weights,
                    &self.vecs,
                    &kvas[li],
                    mode,
                    nb,
                    start,
                    on,
                )?;
            }
            if head == Head::Last {
                // The last row alone: its norm into act row 0, the head
                // (nb = 1: gemvm, as decode's gemv == gemv_ref) into logits
                // row 0.
                self.norm_rows(b, &self.vecs, self.output_norm, nb - 1, 1)?;
                on("rmsnorm")?;
                self.bgemm(
                    &self.weights,
                    &self.output,
                    b.act.arg(),
                    ast,
                    1,
                    null,
                    0,
                    b.logits.arg(),
                    self.vocab,
                )?;
                on("head")?;
            } else if head != Head::No {
                self.norm_b(b, &self.vecs, self.output_norm, nb)?;
                on("rmsnorm")?;
                let v = self.vocab;
                for z0 in (0..nb).step_by(HEAD_SUB) {
                    let nz = (nb - z0).min(HEAD_SUB);
                    let lg = b.logits.arg_at(if b.logits_rows { z0 * v * 4 } else { 0 });
                    self.bgemm(
                        &self.weights,
                        &self.output,
                        b.act.arg_at(z0 * ast),
                        ast,
                        nz,
                        null,
                        0,
                        lg,
                        v,
                    )?;
                    if head == Head::Sample {
                        let s = c.samp.as_ref().expect("samp_upload before a sampling pass");
                        self.samp_tail(s, lg, v, nz, row0 + z0, z0, false)?;
                    } else {
                        g.launch(
                            &bf.argmax,
                            (nz as u32, 1, 1),
                            (1024, 1, 1),
                            0,
                            &[lg, Arg::I32(v as i32), b.best.arg_at(z0 * 4)],
                        )?;
                    }
                    on("head")?;
                }
            }
        }
        Ok(())
    }

    /// Enqueue the pass's input: tokens into b.tokens, or hidden states
    /// straight into the residual stream b.x ([max][dim] floats, the
    /// embedding's output otherwise); `Resident` states are there already.
    pub(super) fn load_input(&self, b: &mut BatchBufs<'g>, input: BatchIn) -> Result<(), String> {
        match input {
            BatchIn::Tokens(tokens) => {
                assert!(
                    self.first == 0,
                    "layers {}..: takes hidden states",
                    self.first
                );
                let hb = b.host_tokens.bytes_mut()?;
                for (i, t) in tokens.iter().enumerate() {
                    hb[4 * i..4 * i + 4].copy_from_slice(&t.to_le_bytes());
                }
                b.host_tokens.copy_to(&b.tokens, 0, 4 * tokens.len())?;
            }
            BatchIn::Hidden(xs) => {
                let hx = b.host_x.as_mut().expect("first > 0: host_x");
                for (d, v) in hx.bytes_mut()?.chunks_exact_mut(4).zip(xs) {
                    d.copy_from_slice(&v.to_le_bytes());
                }
                hx.copy_to(&b.x, 0, xs.len() * 4)?;
            }
            BatchIn::Resident(_) => {}
        }
        Ok(())
    }

    /// b.rope = the RoPE table of positions start..start + nb.
    ///
    /// # Safety
    /// nb <= b.max.
    pub(super) unsafe fn rope_b(
        &self,
        b: &BatchBufs,
        nb: usize,
        start: usize,
    ) -> Result<(), String> {
        let hd = self.head_dim;
        // SAFETY: rope_table_b's signature; b.rope holds b.max >= nb rows.
        unsafe {
            self.gpu.launch(
                &self.bf.rope_table,
                (nb as u32, 1, 1),
                ((hd / 2) as u32, 1, 1),
                0,
                &[
                    self.freq.arg(),
                    Arg::I32((hd / 2) as i32),
                    Arg::I32(start as i32),
                    b.rope.arg(),
                ],
            )
        }
    }

    /// b.act = rmsnorm_quant of b.x's nb rows with the weights at `off`
    /// floats into `vecs`.
    ///
    /// # Safety
    /// nb <= b.max; vecs holds dim floats at off.
    unsafe fn norm_b(
        &self,
        b: &BatchBufs,
        vecs: &DevBuf,
        off: usize,
        nb: usize,
    ) -> Result<(), String> {
        // SAFETY: the caller's contract.
        unsafe { self.norm_rows(b, vecs, off, 0, nb) }
    }

    /// `norm_b` of b.x's rows z0..z0 + n into b.act's rows 0..n.
    ///
    /// # Safety
    /// z0 + n <= b.max; vecs holds dim floats at off.
    unsafe fn norm_rows(
        &self,
        b: &BatchBufs,
        vecs: &DevBuf,
        off: usize,
        z0: usize,
        n: usize,
    ) -> Result<(), String> {
        let dim = self.dim;
        let want_k = |n: usize| Arg::I32(n.is_multiple_of(256) as i32);
        // SAFETY: rmsnorm_quant_b's signature; sizes as `pass`.
        unsafe {
            self.gpu.launch(
                &self.bf.rmsnorm_quant,
                (n as u32, 1, 1),
                (256, 1, 1),
                0,
                &[
                    b.x.arg_at(z0 * dim * 4),
                    vecs.arg_at(off * 4),
                    Arg::F32(self.eps),
                    Arg::I32(dim as i32),
                    want_k(dim),
                    b.act.arg(),
                    Arg::U64(act_stride(self.act_len()) as u64),
                ],
            )
        }
    }

    /// One layer over the nb tokens in b.x at positions start..: matrices
    /// at their offsets into `wts`, norms and biases at theirs into `vecs`,
    /// the layer's cache at `kv` (`kv_args`: F16 K and V rows of kv_dim
    /// halves per position; Q8 int8 K and V rows and their scales) in
    /// format `mode`, which picks the attention kernels. b.rope holds the
    /// batch's table. `on` runs after each step, by name.
    ///
    /// # Safety
    /// nb <= b.max; the matrices are whole in `wts`, the vectors in `vecs`;
    /// the K and V buffers hold positions 0..start + nb, rows before start
    /// written; all stay live until the work completes.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn layer(
        &self,
        b: &BatchBufs,
        l: &LayerW,
        wts: &DevBuf,
        vecs: &DevBuf,
        kv: &[Arg],
        mode: CacheKv,
        nb: usize,
        start: usize,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
    ) -> Result<(), String> {
        let (dim, hd, kvd, ffn) = (self.dim, self.head_dim, self.kv_dim(), self.ffn);
        let qd = self.q_dim;
        let (g, bf) = (self.gpu, &self.bf);
        let vec = |off: usize| vecs.arg_at(off * 4);
        let null = Arg::Ptr(0);
        let want_k = |n: usize| Arg::I32(n.is_multiple_of(256) as i32);
        let ast = act_stride(self.act_len());
        let nbu = nb as u32;
        let qkvw = qd + 2 * kvd;
        // SAFETY (every launch): as `pass`, with the caller's contract for
        // the matrices, vectors and cache rows.
        unsafe {
            self.norm_b(b, vecs, l.attn_norm, nb)?;
            on("rmsnorm")?;
            let bias = |o: Option<usize>| o.map_or(null, vec);
            let qkv_at = |off: usize| b.qkv.arg_at(off * 4);
            self.bgemm(
                wts,
                &l.q,
                b.act.arg(),
                ast,
                nb,
                bias(l.bq),
                0,
                qkv_at(0),
                qkvw,
            )?;
            on("q")?;
            self.bgemm(
                wts,
                &l.k,
                b.act.arg(),
                ast,
                nb,
                bias(l.bk),
                0,
                qkv_at(qd),
                qkvw,
            )?;
            on("k")?;
            self.bgemm(
                wts,
                &l.v,
                b.act.arg(),
                ast,
                nb,
                bias(l.bv),
                0,
                qkv_at(qd + kvd),
                qkvw,
            )?;
            // (Profile segments by type; "qkv", "gate_up", "down" are the
            // stream's weight-group events, stream.rs.)
            on(if l.v.qt == Q6_K { "v6" } else { "v4" })?;
            on("qkv")?;
            // q8: rope_kv_b (rope_kv_norm_b) does the queries alone (null
            // cache), kv_q8_b the keys and values (normalized first on
            // qwen3, as decode).
            let q8 = matches!(mode, CacheKv::Q8 { .. });
            let (kc, vc) = if q8 { (null, null) } else { (kv[0], kv[1]) };
            if let (Some(qn), Some(kn)) = (l.qn, l.kn) {
                // qwen3: a warp per head and token (decode.cu warp_norm_rope).
                let heads = (qd + 2 * kvd) / hd;
                g.launch(
                    &bf.rope_kv_norm,
                    (heads.div_ceil(4) as u32, nbu, 1),
                    (128, 1, 1),
                    0,
                    &[
                        b.qkv.arg(),
                        Arg::I32(qd as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(start as i32),
                        b.rope.arg(),
                        vec(qn),
                        vec(kn),
                        Arg::F32(self.eps),
                        b.q.arg(),
                        kc,
                        vc,
                    ],
                )?;
            } else {
                let items = if q8 { qd / 2 } else { qd / 2 + kvd / 2 + kvd };
                g.launch(
                    &bf.rope_kv,
                    (items.div_ceil(128) as u32, nbu, 1),
                    (128, 1, 1),
                    0,
                    &[
                        b.qkv.arg(),
                        Arg::I32(qd as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(start as i32),
                        b.rope.arg(),
                        b.q.arg(),
                        kc,
                        vc,
                    ],
                )?;
            }
            if q8 {
                let mut args = vec![
                    b.qkv.arg(),
                    Arg::I32(qd as i32),
                    Arg::I32(kvd as i32),
                    Arg::I32(start as i32),
                    b.rope.arg(),
                ];
                args.extend_from_slice(kv);
                args.push(bias(l.ksh));
                args.push(bias(l.kn));
                args.push(Arg::F32(self.eps));
                g.launch(
                    &self.q8_fns(mode).store_b,
                    (self.n_kv_heads as u32, nbu, 1),
                    (128, 1, 1),
                    0,
                    &args,
                )?;
            }
            on("rope_kv")?;
            let pre = Arg::F32(att_pre(hd));
            let kg = match mode {
                CacheKv::F16 => 0,
                CacheKv::Q8 { group } => group,
            };
            assert!(start + nb <= b.nblk_max * ATT_BLK, "batch past its cache");
            if self.tc_attn && nb > att_per_query(hd) {
                // Flash attention, every query of the pass in one launch,
                // y in place of q; then act (the decode kernels' bits,
                // prefill.cu attn_fa_bt). Half-size blocks when the full
                // ones don't fill the GPU.
                let group = self.n_heads / self.n_kv_heads;
                let full = fa_shape(hd, false, kg).0;
                let small = self.n_kv_heads * (nb * group).div_ceil(full) < bf.sms;
                let (rows, threads, smem) = fa_shape(hd, small, kg);
                let f = match q8 {
                    false => &bf.attn_fa[small as usize],
                    true => &self.q8_fns(mode).attn_fa[small as usize],
                };
                let mut args = vec![b.q.arg()];
                args.extend_from_slice(kv);
                args.extend_from_slice(&[
                    Arg::I32(start as i32),
                    Arg::I32(nb as i32),
                    Arg::I32(self.n_heads as i32),
                    Arg::I32(self.n_kv_heads as i32),
                    pre,
                    b.q.arg(),
                ]);
                g.launch(
                    f,
                    (
                        self.n_kv_heads as u32,
                        (nb * group).div_ceil(rows) as u32,
                        1,
                    ),
                    (threads as u32, 1, 1),
                    smem as u32,
                    &args,
                )?;
                on("attn")?;
                g.launch(
                    &bf.quantize,
                    (qd.div_ceil(256) as u32, nbu, 1),
                    (32, 1, 1),
                    0,
                    &[
                        b.q.arg(),
                        Arg::I32(qd as i32),
                        want_k(qd),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )?;
                on("combine")?;
            } else {
                // A few queries (the verify pass), or no tensor cores: the
                // decode kernels per query (the same code), attn_blk_t, or
                // attn_split_t ATT_SUB queries per launch.
                let tc = self.tc_attn;
                let per = if tc { att_per_query(hd) } else { ATT_SUB };
                let (gy_blk, ms) = (self.max_split.min(b.nblk_max) as u32, self.max_split as u32);
                for z0 in (0..nb).step_by(per) {
                    let nz = (nb - z0).min(per) as u32;
                    // (kernel, blocks per kv head and query, threads, the
                    // partials' stride: blocks or splits).
                    let (f, grid, block, stride) = match (tc, q8) {
                        (true, false) => (&bf.attn_blk, gy_blk, 32, b.nblk_max),
                        (true, true) => (&self.q8_fns(mode).attn_blk_b, gy_blk, 32, b.nblk_max),
                        (false, false) => (&bf.attn_split, ms, ATT_TILE as u32, self.max_split),
                        (false, true) => (
                            &self.q8_fns(mode).attn_split_b,
                            ms,
                            ATT_TILE as u32,
                            self.max_split,
                        ),
                    };
                    let mut args = vec![b.q.arg()];
                    args.extend_from_slice(kv);
                    args.extend_from_slice(&[
                        Arg::I32(start as i32),
                        Arg::I32(z0 as i32),
                        Arg::I32(self.n_heads as i32),
                        Arg::I32(self.n_kv_heads as i32),
                        if tc {
                            pre
                        } else {
                            Arg::F32(1.0 / (hd as f32).sqrt())
                        },
                        Arg::I32(stride as i32),
                        b.part_o.arg(),
                        b.part_ml.arg(),
                    ]);
                    g.launch(
                        f,
                        (self.n_kv_heads as u32, grid, nz),
                        (block, 1, 1),
                        0,
                        &args,
                    )?;
                    on("attn")?;
                    g.launch(
                        if tc { &bf.attn_cmb } else { &bf.attn_combine },
                        (qd.div_ceil(256) as u32, nz, 1),
                        (256, 1, 1),
                        0,
                        &[
                            b.part_o.arg(),
                            b.part_ml.arg(),
                            Arg::I32(start as i32),
                            Arg::I32(z0 as i32),
                            Arg::I32(qd as i32),
                            Arg::I32(hd as i32),
                            Arg::I32(stride as i32),
                            want_k(qd),
                            b.act.arg(),
                            Arg::U64(ast as u64),
                        ],
                    )?;
                    on("combine")?;
                }
            }
            self.bgemm(wts, &l.o, b.act.arg(), ast, nb, null, 1, b.x.arg(), dim)?;
            on("wo")?;
            self.norm_b(b, vecs, l.ffn_norm, nb)?;
            on("rmsnorm")?;
            // Both Q4_K, a tensor-core batch: one launch for gate and up.
            let fuse = self.gate_up_fused
                && bf.gemm_gu.is_some()
                && nb > SMALL
                && l.gate.qt == Q4_K
                && l.up.qt == Q4_K
                && l.gate.cols == dim
                && l.up.cols == dim;
            if !fuse {
                self.bgemm(
                    wts,
                    &l.gate,
                    b.act.arg(),
                    ast,
                    nb,
                    null,
                    0,
                    b.gate.arg(),
                    ffn,
                )?;
            }
            on("gate")?;
            if let (Some(f), true) = (&bf.gemm_gu, fuse) {
                // gate and up in one launch, hid = silu(gate) * up out
                // (the same floats and ops as below), then quantize_b.
                g.launch(
                    f,
                    (nb.div_ceil(64) as u32, ffn.div_ceil(32) as u32, 1),
                    (256, 1, 1),
                    gemm2_smem(Q4_K, 64, 3) as u32,
                    &[
                        wts.arg_at(l.gate.off),
                        wts.arg_at(l.up.off),
                        Arg::U64(l.gate.rb as u64),
                        Arg::I32(ffn as i32),
                        Arg::I32(dim as i32),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                        Arg::I32(nb as i32),
                        b.up.arg(),
                        Arg::U64(ffn as u64),
                    ],
                )?;
                on("gate_up")?;
                g.launch(
                    &bf.quantize,
                    (ffn.div_ceil(256) as u32, nbu, 1),
                    (32, 1, 1),
                    0,
                    &[
                        b.up.arg(),
                        Arg::I32(ffn as i32),
                        want_k(ffn),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )?;
            } else if self.swiglu_epi {
                // The up product's epilogue (mode 2) makes hid = silu(gate) *
                // up from the gate product and its own sum (swiglu_quant_b's
                // ops on the same floats) into b.up, then quantize_b: one
                // fp32 product less written and read back per layer.
                self.bgemm(
                    wts,
                    &l.up,
                    b.act.arg(),
                    ast,
                    nb,
                    b.gate.arg(),
                    2,
                    b.up.arg(),
                    ffn,
                )?;
                on("gate_up")?;
                g.launch(
                    &bf.quantize,
                    (ffn.div_ceil(256) as u32, nbu, 1),
                    (32, 1, 1),
                    0,
                    &[
                        b.up.arg(),
                        Arg::I32(ffn as i32),
                        want_k(ffn),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )?;
            } else {
                self.bgemm(wts, &l.up, b.act.arg(), ast, nb, null, 0, b.up.arg(), ffn)?;
                on("gate_up")?;
                g.launch(
                    &bf.swiglu_quant,
                    (ffn.div_ceil(256) as u32, nbu, 1),
                    (32, 1, 1),
                    0,
                    &[
                        b.gate.arg(),
                        b.up.arg(),
                        Arg::I32(ffn as i32),
                        want_k(ffn),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )?;
            }
            on("swiglu")?;
            self.bgemm(wts, &l.down, b.act.arg(), ast, nb, null, 1, b.x.arg(), dim)?;
            on(if l.down.qt == Q6_K { "down6" } else { "down4" })?;
            on("down")?;
        }
        Ok(())
    }

    /// `feed_batch`: passes of up to the cache's batch max (`passes`), then
    /// a wait (so a device fault surfaces here, as with `feed`).
    pub(super) fn feed_tokens(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        start: usize,
    ) -> Result<(), String> {
        let max = c.b.as_ref().expect("batch scratch").max;
        for r in passes(input.len(self.dim), max) {
            let ch = input.range(r.clone(), self.dim);
            self.pass(c, ch, start + r.start, false, &mut |_| Ok(()))?;
        }
        self.gpu.sync()
    }

    /// `prefill_logits` (nb >= 2): the passes of `feed_tokens`, the last
    /// with `Head::Last`; its logits (waits for the passes).
    pub(super) fn prefill_tokens(
        &self,
        c: &mut CudaCache<'g>,
        tokens: &[u32],
        start: usize,
    ) -> Result<Vec<f32>, String> {
        let max = c.b.as_ref().expect("batch scratch").max;
        let ps = passes(tokens.len(), max);
        for (i, r) in ps.iter().enumerate() {
            let head = if i + 1 == ps.len() {
                Head::Last
            } else {
                Head::No
            };
            let ch = BatchIn::Tokens(&tokens[r.clone()]);
            self.pass_with(c, ch, start + r.start, head, 0, &mut |_| Ok(()))?;
        }
        let n = self.vocab * 4;
        let b = c.b.as_ref().unwrap();
        c.host_out.copy_from(&b.logits, 0, n)?;
        let hb = &c.host_out.bytes_mut()?[..n]; // waits for the passes and copy
        Ok(hb
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect())
    }

    /// `verify_argmax`: each token's greedy pick.
    pub(super) fn verify_tokens(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        start: usize,
    ) -> Result<Vec<u32>, String> {
        let max = c.b.as_ref().expect("batch scratch").max;
        let nt = input.len(self.dim);
        let mut picks = Vec::with_capacity(nt);
        for r in passes(nt, max) {
            let ch = input.range(r.clone(), self.dim);
            self.pass(c, ch, start + r.start, true, &mut |_| Ok(()))?;
            let b = c.b.as_mut().unwrap();
            picks.extend(b.picks(ch.len(self.dim))?);
        }
        Ok(picks)
    }
}

/// The passes of n tokens with at most max per pass: as few as possible,
/// of equal size (the first ones one token longer), so a prompt just past a
/// multiple of max doesn't end in a short pass (latency-bound GEMMs: the
/// weights read once more for a few tokens). Results don't depend on it.
pub(super) fn passes(n: usize, max: usize) -> Vec<std::ops::Range<usize>> {
    let np = n.div_ceil(max);
    let (q, r) = (n / np.max(1), n % np.max(1));
    let mut v = Vec::with_capacity(np);
    let mut s = 0;
    for i in 0..np {
        let e = s + q + (i < r) as usize;
        v.push(s..e);
        s = e;
    }
    v
}

impl BatchBufs<'_> {
    /// The first n greedy picks of the pass enqueued (waits for it).
    pub(super) fn picks(&mut self, n: usize) -> Result<Vec<u32>, String> {
        let n = 4 * n;
        self.host_best.copy_from(&self.best, 0, n)?;
        let hb = &self.host_best.bytes_mut()?[..n]; // waits for the pass
        Ok(hb
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect())
    }
}

impl<'g> CudaQwen2<'g> {
    /// `verify_cands`: each token's sampling candidates, in passes of up to
    /// the cache's batch max.
    pub(super) fn verify_cands_tokens(
        &self,
        c: &mut CudaCache<'g>,
        tokens: &[u32],
        start: usize,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Result<Vec<Vec<Cand>>, String> {
        let max = c.b.as_ref().expect("batch scratch").max;
        self.samp_upload(c, spec, recent, tokens.len(), tokens.len().min(max), None)?;
        let mut rows = Vec::with_capacity(tokens.len());
        for (i, ch) in tokens.chunks(max).enumerate() {
            let input = BatchIn::Tokens(ch);
            self.pass_with(
                c,
                input,
                start + i * max,
                Head::Sample,
                i * max,
                &mut |_| Ok(()),
            )?;
            rows.extend(self.samp_read(c, spec.k, self.vocab, ch.len())?);
        }
        Ok(rows)
    }
}
