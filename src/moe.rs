//! A MoE model (qwen3moe, gpt-oss) with attention, norms, routers, embedding and head on the
//! GPU and the experts on the CPU (r11 stage 2; llama.cpp's `--cpu-moe`).
//!
//! The GPU half is a `CudaQwen2` without dense FFNs (`with_kv_moe`: its
//! layers' `Ffn::Moe` hold the routers); the CPU half is the experts alone
//! (`Qwen2::new_experts`: r8-repacked, the attention never read). Per layer
//! and step: attention and router on the GPU, the normed states, expert ids
//! and weights down to the host, the experts on the CPU pool, y back up
//! (`cuda::qwen2::moe`). All KV is on the GPU.
//!
//! Numerics: the attention is the GPU's (batch-invariant), the router
//! `moe_route` (per token), the experts the CPU's stage-1 kernels given the
//! GPU's routing (per token, whatever the batch). So decode, `feed_batch`
//! and verify give a token the same bits: chat's cache reuse equals a fresh
//! prefill and drafting is lossless. Against the CPU-only model it is
//! graded, as the GPU is (tests).
//!
//! gpt-oss runs the same way (stage 1): its windowed attention with sinks
//! is cuda/oss.cu's (the model's module, `Kernels::module_for`), its biased
//! router moe.cu's `moe_topk_oai`, and its MXFP4 experts are read in place
//! from the mapping (`Qwen2::new_experts`: no repack, no copy).
//!
//! Threads: as the hybrid's, strictly sequential: the thread that opened
//! the GPU drives the CPU pool for the experts between the GPU's segments.

use crate::cuda::qwen2::mla::{MlaBufs, MlaFns};
use crate::cuda::qwen2::moe::{MoeBufs, MoeFns};
use crate::cuda::qwen2::{CacheKv, CudaCache, CudaQwen2, Head};
use crate::cuda::{Gpu, Module};
use crate::gguf::Model;
use crate::lm::Lm;
use crate::model::{MoeRoute, Qwen2};

pub struct MoeQwen3<'m, 'g> {
    gpu: CudaQwen2<'g>,
    cpu: Qwen2<'m>,
    mf: MoeFns<'g>,
    /// deepseek2: the MLA layers' kernels (`cuda::qwen2::mla`; the GPU
    /// half's `mla`); None for qwen3moe.
    xf: Option<MlaFns<'g>>,
    /// The context its caches are planned for (with `fit`, maybe less than
    /// asked).
    pub ctx: usize,
}

/// The GPU cache and the experts' exchange scratch.
pub struct MoeCache<'g> {
    /// Before `c`: its graphs hold the cache's addresses.
    mb: MoeBufs<'g>,
    /// deepseek2: the MLA layers' scratch (then `mb` is a one-token stub).
    xb: Option<MlaBufs<'g>>,
    c: CudaCache<'g>,
}

impl<'m, 'g> MoeQwen3<'m, 'g> {
    /// `m` (a qwen3moe model) with its attention on `gpu` (a `max_ctx`
    /// cache of type `kv`; with `fit` a shorter one if that doesn't fit)
    /// and its experts on the CPU. `module` is the model's kernel module,
    /// `moe_module` moe.cu's (`Kernels::moe_module`). `q6k8`: the Q6_K
    /// prefill copy now (as `CudaQwen2::with_kv_q6k8`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        moe_module: &'g Module<'g>,
        m: &'m Model,
        max_ctx: usize,
        kv: CacheKv,
        fit: bool,
        q6k8: bool,
    ) -> Result<MoeQwen3<'m, 'g>, String> {
        crate::model::moe_check_ram(m, false)?;
        let net = CudaQwen2::with_kv_moe(gpu, module, m, max_ctx, kv, fit, q6k8)?;
        let mf = MoeFns::new(moe_module)?;
        let cpu = Qwen2::new_experts(m);
        Ok(MoeQwen3 {
            ctx: net.ctx,
            gpu: net,
            cpu,
            mf,
            xf: None,
        })
    }

    /// `new` for a deepseek2 model: `mla_module` is `Kernels::mla_module`
    /// (MLA attention and router; moe.cu's add and logits with it).
    #[allow(clippy::too_many_arguments)]
    pub fn new_mla(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        mla_module: &'g Module<'g>,
        m: &'m Model,
        max_ctx: usize,
        kv: CacheKv,
        fit: bool,
        q6k8: bool,
    ) -> Result<MoeQwen3<'m, 'g>, String> {
        crate::model::moe_check_ram(m, false)?;
        let net = CudaQwen2::with_kv_moe(gpu, module, m, max_ctx, kv, fit, q6k8)?;
        let mf = MoeFns::new(mla_module)?;
        let xf = MlaFns::new(mla_module, module)?;
        let cpu = Qwen2::new_experts(m);
        Ok(MoeQwen3 {
            ctx: net.ctx,
            gpu: net,
            cpu,
            mf,
            xf: Some(xf),
        })
    }

    /// The GPU half (identity, profiles).
    pub fn gpu_net(&self) -> &CudaQwen2<'g> {
        &self.gpu
    }

    pub fn kv_name(&self) -> &'static str {
        self.gpu.kv_name()
    }

    /// One line for the banner.
    pub fn describe(&self, max_ctx: usize) -> String {
        let d = self.gpu.moe.expect("a mixture-of-experts model");
        format!(
            "attention + router on the GPU ({:.2} GiB, ctx {max_ctx}), {} experts x {} \
             layers on the CPU (top-{})",
            self.gpu.weight_bytes() as f64 / (1u64 << 30) as f64,
            d.n_expert,
            self.gpu.mla_moe_layers().unwrap_or(self.gpu.n_layers),
            d.k
        )
    }

    fn die(e: String) -> ! {
        panic!("CUDA: {e}")
    }

    /// The CPU experts of a layer for nb tokens (token major in and out).
    fn experts(&self, l: usize, h: &[f32], ids: &[i32], w: &[f32], y: &mut [f32]) {
        let dim = self.gpu.dim;
        let nb = h.len() / dim;
        if nb == 1 {
            self.cpu.moe_layer(l, h, ids, w, y);
            return;
        }
        let k = ids.len() / nb;
        let route = |t: usize| -> MoeRoute {
            (
                ids[t * k..(t + 1) * k]
                    .iter()
                    .map(|&e| e as usize)
                    .collect(),
                w[t * k..(t + 1) * k].to_vec(),
            )
        };
        let hs: Vec<&[f32]> = h.chunks_exact(dim).collect();
        let routes: Vec<MoeRoute> = (0..nb).map(route).collect();
        let yc = self.cpu.moe_layer_batch(l, &hs, &routes);
        transpose_tiled(&yc, nb, dim, y);
    }

    fn step(&self, token: u32, pos: usize, c: &mut MoeCache<'g>) {
        let mut ex =
            |l: usize, h: &[f32], i: &[i32], w: &[f32], y: &mut [f32]| self.experts(l, h, i, w, y);
        if let (Some(xf), Some(xb)) = (&self.xf, &mut c.xb) {
            return self
                .gpu
                .mla_step(xf, &mut c.c, xb, token, pos, &mut ex)
                .unwrap_or_else(|e| Self::die(e));
        }
        self.gpu
            .moe_step(
                &self.mf,
                &mut c.c,
                &mut c.mb,
                token,
                pos,
                Head::Argmax,
                &mut ex,
            )
            .unwrap_or_else(|e| Self::die(e));
    }

    fn tokens(
        &self,
        tokens: &[u32],
        start: usize,
        c: &mut MoeCache<'g>,
        head: Head,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut ex =
            |l: usize, h: &[f32], i: &[i32], w: &[f32], y: &mut [f32]| self.experts(l, h, i, w, y);
        if let (Some(xf), Some(xb)) = (&self.xf, &mut c.xb) {
            return self
                .gpu
                .mla_tokens(xf, &mut c.c, xb, tokens, start, head, &mut ex)
                .unwrap_or_else(|e| Self::die(e));
        }
        self.gpu
            .moe_tokens(&self.mf, &mut c.c, &mut c.mb, tokens, start, head, &mut ex)
            .unwrap_or_else(|e| Self::die(e))
    }
}

/// y[t][r] = yc[r][t] (yc: dim rows of nb, y: nb rows of dim), in
/// parallel over blocks of `TILE` tokens, each copied in TILE x TILE tiles
/// (a row of yc read TILE floats at a time, not one per 4 * nb bytes).
fn transpose_tiled(yc: &[f32], nb: usize, dim: usize, y: &mut [f32]) {
    const TILE: usize = 32;
    assert!(yc.len() == nb * dim && y.len() == nb * dim);
    crate::pool::for_each_chunk_mut(y, TILE * dim, |c, ys| {
        let t0 = c * TILE;
        let nt = ys.len() / dim;
        for r0 in (0..dim).step_by(TILE) {
            let nr = TILE.min(dim - r0);
            for r in r0..r0 + nr {
                let src = &yc[r * nb + t0..r * nb + t0 + nt];
                for (j, &v) in src.iter().enumerate() {
                    ys[j * dim + r] = v;
                }
            }
        }
    });
}

impl<'g> Lm for MoeQwen3<'_, 'g> {
    type Cache = MoeCache<'g>;

    fn vocab(&self) -> usize {
        self.gpu.vocab
    }

    fn new_cache(&self, max_ctx: usize) -> MoeCache<'g> {
        let c = Lm::new_cache(&self.gpu, max_ctx);
        let max = CudaQwen2::batch_max(&c);
        let xb = self.xf.as_ref().map(|_| {
            self.gpu
                .mla_bufs(max, max_ctx)
                .unwrap_or_else(|e| Self::die(e))
        });
        let mb = self
            .gpu
            .moe_bufs(if xb.is_some() { 1 } else { max })
            .unwrap_or_else(|e| Self::die(e));
        MoeCache { mb, xb, c }
    }

    fn forward(&self, token: u32, pos: usize, c: &mut MoeCache<'g>) -> Vec<f32> {
        self.step(token, pos, c);
        self.gpu
            .moe_logits(&mut c.c)
            .unwrap_or_else(|e| Self::die(e))
    }

    fn forward_argmax(&self, token: u32, pos: usize, c: &mut MoeCache<'g>) -> u32 {
        self.step(token, pos, c);
        self.gpu.moe_pick(&mut c.c).unwrap_or_else(|e| Self::die(e))
    }

    fn feed_batch(&self, tokens: &[u32], start: usize, c: &mut MoeCache<'g>) {
        self.tokens(tokens, start, c, Head::No);
    }

    fn verify_argmax(&self, tokens: &[u32], start: usize, c: &mut MoeCache<'g>) -> Vec<u32> {
        self.tokens(tokens, start, c, Head::Argmax).0
    }

    fn prefill_logits(&self, tokens: &[u32], start: usize, c: &mut MoeCache<'g>) -> Vec<f32> {
        match tokens {
            [] => panic!("prefill_logits: no tokens"),
            [t] => self.forward(*t, start, c),
            _ => self.tokens(tokens, start, c, Head::Last).1,
        }
    }

    fn prefill_batch(&self) -> usize {
        self.gpu.prefill_batch()
    }
}

#[cfg(test)]
mod ds2_tests;
#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod oss_tests;
