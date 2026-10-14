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

use super::*;

/// Batches of up to this many tokens use gemvn (weight-read-once, CUDA
/// cores) even where the tensor-core GEMM exists.
pub(super) const SMALL: usize = 16;
/// Query tokens per batched attention launch (sizes `part_o`).
pub(super) const ATT_SUB: usize = 64;
/// Tokens per output-head launch in `verify_argmax` (sizes the logits).
pub(super) const HEAD_SUB: usize = 16;

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
}

impl BatchIn<'_> {
    fn len(&self, dim: usize) -> usize {
        match self {
            BatchIn::Tokens(t) => t.len(),
            BatchIn::Hidden(x) => x.len() / dim,
        }
    }

    /// Tokens i * max.. (at most max of them).
    fn chunk(&self, i: usize, max: usize, dim: usize) -> Self {
        let n = self.len(dim);
        let r = i * max..((i + 1) * max).min(n);
        match self {
            BatchIn::Tokens(t) => BatchIn::Tokens(&t[r]),
            BatchIn::Hidden(x) => BatchIn::Hidden(&x[r.start * dim..r.end * dim]),
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
    attn_split: Func<'m>,
    /// attn_split_b for queries at most 32 * max_split positions in.
    attn_warp: Func<'m>,
    attn_combine: Func<'m>,
    argmax: Func<'m>,
    gemvn8: Func<'m>,
    gemvn16: Func<'m>,
    /// gemvm8_*, gemvm16_* per type (as `gemm`); None without tensor cores.
    gemvm: Option<[[Func<'m>; 4]; 2]>,
    /// Per type (Q4_K, Q6_K, Q8_0, Q5_0): gemm2_* (128 tokens per block)
    /// and gemm2s_* (64); None without int8 tensor cores.
    gemm: Option<[[Func<'m>; 2]; 4]>,
    /// SMs of the GPU (the GEMM tile choice).
    sms: usize,
}

impl<'m> BatchFns<'m> {
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
            attn_split: module.function(if head_dim == 64 {
                "attn_split_b_64"
            } else {
                "attn_split_b_128"
            })?,
            attn_warp: module.function(if head_dim == 64 {
                "attn_warp_b_64"
            } else {
                "attn_warp_b_128"
            })?,
            attn_combine: module.function("attn_combine_b")?,
            argmax: module.function("argmax_b")?,
            gemvn8: module.function("gemvn_2x8")?,
            gemvn16: module.function("gemvn_2x16")?,
            gemvm,
            gemm,
            sms,
        })
    }
}

/// A cache's scratch for batched passes of up to `max` tokens.
pub(super) struct BatchBufs<'g> {
    pub(super) max: usize,
    tokens: DevBuf<'g>,
    host_tokens: HostBuf<'g>,
    /// Incoming hidden states, [max][dim] (first > 0 only).
    host_x: Option<HostBuf<'g>>,
    /// Residual stream, [max][dim].
    x: DevBuf<'g>,
    /// Products of the q, k, v matrices (biases added), [max][dim + 2 kv_dim].
    qkv: DevBuf<'g>,
    /// Rotated queries, [max][dim].
    q: DevBuf<'g>,
    gate: DevBuf<'g>,
    up: DevBuf<'g>,
    /// Quantized activations, [max][act_stride(max(dim, ffn))].
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

    /// VRAM of the batch scratch for `max` tokens.
    pub(super) fn batch_bytes(&self, max: usize) -> usize {
        let (f, kvd) = (4, self.kv_dim());
        let att = max.min(ATT_SUB);
        let parts = att * self.n_heads * self.max_split * (self.head_dim + 2) * f;
        let per_tok = (3 * self.dim + 2 * kvd + 2 * self.ffn + self.head_dim) * f
            + act_stride(self.dim.max(self.ffn))
            + 8
            + if self.first > 0 { self.dim * f } else { 0 };
        max * per_tok + parts + max.min(HEAD_SUB) * self.vocab * f + 256
    }

    pub(super) fn batch_bufs(&self, max: usize) -> Result<BatchBufs<'g>, String> {
        let (g, f, kvd) = (self.gpu, 4, self.kv_dim());
        let att = max.min(ATT_SUB);
        Ok(BatchBufs {
            max,
            tokens: g.alloc(max * 4)?,
            host_tokens: g.host_alloc(max * 4)?,
            host_x: match self.first {
                0 => None,
                _ => Some(g.host_alloc(max * self.dim * f)?),
            },
            x: g.alloc(max * self.dim * f)?,
            qkv: g.alloc(max * (self.dim + 2 * kvd) * f)?,
            q: g.alloc(max * self.dim * f)?,
            gate: g.alloc(max * self.ffn * f)?,
            up: g.alloc(max * self.ffn * f)?,
            act: g.alloc(max * act_stride(self.dim.max(self.ffn)))?,
            rope: g.alloc(max * self.head_dim / 2 * 8)?,
            part_o: g.alloc(att * self.n_heads * self.max_split * self.head_dim * f)?,
            part_ml: g.alloc(att * self.n_heads * self.max_split * 2 * f)?,
            logits: g.alloc(max.min(HEAD_SUB) * self.vocab * f)?,
            logits_rows: false,
            best: g.alloc(max * 4)?,
            host_best: g.host_alloc(max * 4)?,
        })
    }

    /// out[z][r] (row stride ostride floats) = W act[z] (+ bias) for mode 0,
    /// += for mode 1, for tokens z < nb: gemvn for small batches (or without
    /// tensor cores), else the type's GEMM. Same bits either way.
    ///
    /// # Safety
    /// act holds nb tokens of mt.cols inputs at astride bytes apart; out has
    /// room for nb rows of ostride floats (mt.rows <= ostride); bias is null
    /// or has mt.rows floats.
    #[allow(clippy::too_many_arguments)]
    unsafe fn bgemm(
        &self,
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
        let common = [
            self.weights.arg_at(mt.off),
            Arg::U64(mt.rb as u64),
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
        // SAFETY: the caller's contract; the matrix is whole in the arena.
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
                    self.gpu.launch(
                        &fs[ti][!big as usize],
                        (nb.div_ceil(bn) as u32, mt.rows.div_ceil(G2_BM) as u32, 1),
                        (256, 1, 1),
                        gemm2_smem(mt.qt, bn, st) as u32,
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
        let nb = input.len(self.dim);
        let (dim, hd, kvd, ffn) = (self.dim, self.head_dim, self.kv_dim(), self.ffn);
        let kvas: Vec<Vec<Arg>> = (0..self.layers.len()).map(|l| self.kv_args(c, l)).collect();
        let mode = c.mode;
        let b = c.b.as_mut().expect("a cache with batch scratch");
        assert!(nb > 0 && nb <= b.max);
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
                b.host_tokens.copy_to(&b.tokens, 0, 4 * nb)?;
            }
            BatchIn::Hidden(xs) => {
                // The states go straight into the residual stream b.x
                // ([max][dim] floats, the embedding's output otherwise).
                let hx = b.host_x.as_mut().expect("first > 0: host_x");
                for (d, v) in hx.bytes_mut()?.chunks_exact_mut(4).zip(xs) {
                    d.copy_from_slice(&v.to_le_bytes());
                }
                hx.copy_to(&b.x, 0, xs.len() * 4)?;
            }
        }
        let b = &*b;
        let (g, bf) = (self.gpu, &self.bf);
        let w = |mt: &Mat| self.weights.arg_at(mt.off);
        let vec = |off: usize| self.vecs.arg_at(off * 4);
        let null = Arg::Ptr(0);
        let want_k = |n: usize| Arg::I32(n.is_multiple_of(256) as i32);
        let ast = act_stride(dim.max(ffn));
        let nbu = nb as u32;
        let qkvw = dim + 2 * kvd;
        // SAFETY (every launch): arguments match prefill.cu's signatures;
        // the batch buffers hold b.max >= nb tokens of what each kernel
        // reads and writes (sizes in batch_bufs); positions start..start + nb
        // are in the cache (the callers check); matrices are whole in the arena.
        unsafe {
            if let (Some(embd), BatchIn::Tokens(_)) = (&self.embd, input) {
                g.launch(
                    &bf.embed,
                    (dim.div_ceil(256) as u32, nbu, 1),
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
            g.launch(
                &bf.rope_table,
                (nbu, 1, 1),
                ((hd / 2) as u32, 1, 1),
                0,
                &[
                    self.freq.arg(),
                    Arg::I32((hd / 2) as i32),
                    Arg::I32(start as i32),
                    b.rope.arg(),
                ],
            )?;
            on("rope")?;
            let norm = |off: usize| {
                g.launch(
                    &bf.rmsnorm_quant,
                    (nbu, 1, 1),
                    (1024, 1, 1),
                    0,
                    &[
                        b.x.arg(),
                        vec(off),
                        Arg::F32(self.eps),
                        Arg::I32(dim as i32),
                        want_k(dim),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )
            };
            for (li, l) in self.layers.iter().enumerate() {
                norm(l.attn_norm)?;
                on("rmsnorm")?;
                let bias = |o: Option<usize>| o.map_or(null, vec);
                let qkv_at = |off: usize| b.qkv.arg_at(off * 4);
                self.bgemm(&l.q, b.act.arg(), ast, nb, bias(l.bq), 0, qkv_at(0), qkvw)?;
                self.bgemm(&l.k, b.act.arg(), ast, nb, bias(l.bk), 0, qkv_at(dim), qkvw)?;
                self.bgemm(
                    &l.v,
                    b.act.arg(),
                    ast,
                    nb,
                    bias(l.bv),
                    0,
                    qkv_at(dim + kvd),
                    qkvw,
                )?;
                on("qkv")?;
                let kva = &kvas[li];
                // q8: rope_kv_b does the queries alone (null cache), kv_q8_b
                // the keys and values.
                let q8 = matches!(mode, CacheKv::Q8 { .. });
                let items = if q8 { dim / 2 } else { dim / 2 + kvd / 2 + kvd };
                let (kc, vc) = if q8 { (null, null) } else { (kva[0], kva[1]) };
                g.launch(
                    &bf.rope_kv,
                    (items.div_ceil(128) as u32, nbu, 1),
                    (128, 1, 1),
                    0,
                    &[
                        b.qkv.arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(start as i32),
                        b.rope.arg(),
                        b.q.arg(),
                        kc,
                        vc,
                    ],
                )?;
                if q8 {
                    let mut args = vec![
                        b.qkv.arg(),
                        Arg::I32(dim as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(start as i32),
                        b.rope.arg(),
                    ];
                    args.extend_from_slice(kva);
                    args.push(bias(l.ksh));
                    g.launch(
                        &self.q8_fns(mode).store_b,
                        (self.n_kv_heads as u32, nbu, 1),
                        (128, 1, 1),
                        0,
                        &args,
                    )?;
                }
                on("rope_kv")?;
                for z0 in (0..nb).step_by(ATT_SUB) {
                    let nz = (nb - z0).min(ATT_SUB) as u32;
                    // Splits of at most 32 positions for every query here:
                    // a warp each (same bits, prefill.cu attn_warp_bt).
                    let short = start + z0 + nz as usize <= 32 * self.max_split;
                    let (f, grid, block) = match (short, q8) {
                        (true, false) => (&bf.attn_warp, self.max_split.div_ceil(4) as u32, 128),
                        (false, false) => (&bf.attn_split, self.max_split as u32, ATT_TILE as u32),
                        (true, true) => (
                            &self.q8_fns(mode).attn_warp_b,
                            self.max_split.div_ceil(4) as u32,
                            128,
                        ),
                        (false, true) => (
                            &self.q8_fns(mode).attn_split_b,
                            self.max_split as u32,
                            ATT_TILE as u32,
                        ),
                    };
                    let mut args = vec![b.q.arg()];
                    args.extend_from_slice(kva);
                    args.extend_from_slice(&[
                        Arg::I32(start as i32),
                        Arg::I32(z0 as i32),
                        Arg::I32(self.n_heads as i32),
                        Arg::I32(self.n_kv_heads as i32),
                        Arg::F32(1.0 / (hd as f32).sqrt()),
                        Arg::I32(self.max_split as i32),
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
                        &bf.attn_combine,
                        (dim.div_ceil(256) as u32, nz, 1),
                        (256, 1, 1),
                        0,
                        &[
                            b.part_o.arg(),
                            b.part_ml.arg(),
                            Arg::I32(start as i32),
                            Arg::I32(z0 as i32),
                            Arg::I32(dim as i32),
                            Arg::I32(hd as i32),
                            Arg::I32(self.max_split as i32),
                            want_k(dim),
                            b.act.arg(),
                            Arg::U64(ast as u64),
                        ],
                    )?;
                    on("combine")?;
                }
                self.bgemm(&l.o, b.act.arg(), ast, nb, null, 1, b.x.arg(), dim)?;
                on("wo")?;
                norm(l.ffn_norm)?;
                on("rmsnorm")?;
                self.bgemm(&l.gate, b.act.arg(), ast, nb, null, 0, b.gate.arg(), ffn)?;
                self.bgemm(&l.up, b.act.arg(), ast, nb, null, 0, b.up.arg(), ffn)?;
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
                on("swiglu")?;
                self.bgemm(&l.down, b.act.arg(), ast, nb, null, 1, b.x.arg(), dim)?;
                on("down")?;
            }
            if head {
                norm(self.output_norm)?;
                on("rmsnorm")?;
                let v = self.vocab;
                for z0 in (0..nb).step_by(HEAD_SUB) {
                    let nz = (nb - z0).min(HEAD_SUB);
                    let lg = b.logits.arg_at(if b.logits_rows { z0 * v * 4 } else { 0 });
                    self.bgemm(
                        &self.output,
                        b.act.arg_at(z0 * ast),
                        ast,
                        nz,
                        null,
                        0,
                        lg,
                        v,
                    )?;
                    g.launch(
                        &bf.argmax,
                        (nz as u32, 1, 1),
                        (1024, 1, 1),
                        0,
                        &[lg, Arg::I32(v as i32), b.best.arg_at(z0 * 4)],
                    )?;
                    on("head")?;
                }
            }
        }
        Ok(())
    }

    /// `feed_batch`: passes of up to the cache's batch max, then a wait (so
    /// a device fault surfaces here, as with `feed`).
    pub(super) fn feed_tokens(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        start: usize,
    ) -> Result<(), String> {
        let max = c.b.as_ref().expect("batch scratch").max;
        let n = input.len(self.dim);
        for i in 0..n.div_ceil(max) {
            let ch = input.chunk(i, max, self.dim);
            self.pass(c, ch, start + i * max, false, &mut |_| Ok(()))?;
        }
        self.gpu.sync()
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
        for i in 0..nt.div_ceil(max) {
            let ch = input.chunk(i, max, self.dim);
            self.pass(c, ch, start + i * max, true, &mut |_| Ok(()))?;
            let b = c.b.as_mut().unwrap();
            let n = 4 * ch.len(self.dim);
            b.host_best.copy_from(&b.best, 0, n)?;
            let hb = &b.host_best.bytes_mut()?[..n]; // waits for the pass
            picks.extend(
                hb.chunks_exact(4)
                    .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]])),
            );
        }
        Ok(picks)
    }
}
