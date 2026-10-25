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
//! router moe.cu's `moe_router` (`MoeKind::Oai`), and its MXFP4 experts are read in place
//! from the mapping (`Qwen2::new_experts`: no repack, no copy).
//!
//! Threads: as the hybrid's, strictly sequential: the thread that opened
//! the GPU drives the CPU pool for the experts between the GPU's segments.

use crate::cuda::qwen2::experts::{self, GpuExperts, ResBufs, Resident};
use crate::cuda::qwen2::mla::MlaBufs;
use crate::cuda::qwen2::moe::{MoeBufs, MoeKind};
use crate::cuda::qwen2::{CacheKv, CudaCache, CudaQwen2, Head};
use crate::cuda::{Gpu, Module};
use crate::gguf::Model;
use crate::lm::Lm;
use crate::model::{MoeRoute, Qwen2};

pub(crate) struct MoeQwen3<'m, 'g> {
    gpu: CudaQwen2<'g>,
    cpu: Qwen2<'m>,
    kind: MoeKind<'g>,
    /// The context its caches are planned for (with `fit`, maybe less than
    /// asked).
    pub(crate) ctx: usize,
    /// The MoE layers whose experts are resident on the GPU
    /// (`--gpu-experts`; deepseek2 only for now), and why there are none if
    /// some were asked for.
    res: Option<Resident<'g>>,
    res_note: Option<String>,
    dev: &'g Gpu,
}

/// The GPU cache and the experts' exchange scratch.
pub(crate) struct MoeCache<'g> {
    /// Before `c`: its graphs hold the cache's addresses.
    s: Scratch<'g>,
    c: CudaCache<'g>,
}

/// A cache's scratch, by `MoeKind`: deepseek2's MLA layers have their own.
enum Scratch<'g> {
    Moe(MoeBufs<'g>),
    /// With the resident experts' scratch if any are.
    Mla(Box<MlaBufs<'g>>, Option<ResBufs<'g>>),
}

/// A pass `MoeQwen3::run` dispatches: a decode step (token, pos) or
/// `moe_tokens`' (tokens, start, head).
enum Pass<'a> {
    Step(u32, usize),
    Tokens(&'a [u32], usize, Head),
}

#[cfg(test)]
impl<'g> MoeCache<'g> {
    /// The GPU cache and the qwen3moe-style scratch.
    pub(crate) fn moe_parts(&mut self) -> (&mut CudaCache<'g>, &mut MoeBufs<'g>) {
        match &mut self.s {
            Scratch::Moe(mb) => (&mut self.c, mb),
            Scratch::Mla(..) => panic!("an MLA cache"),
        }
    }
}

impl<'m, 'g> MoeQwen3<'m, 'g> {
    /// `m` (a qwen3moe, gpt-oss or deepseek2 model) with its attention on
    /// `gpu` (a `max_ctx` cache of type `kv`; with `fit` a shorter one if
    /// that doesn't fit) and its experts on the CPU. `module` is the
    /// model's kernel module, `router` `Kernels::moe_module_for(m)`.
    /// `q6k8`: the Q6_K prefill copy now (as `CudaQwen2::with_kv_q6k8`).
    /// The tests'; the product passes `--gpu-experts` (`new_with`).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        router: &'g Module<'g>,
        m: &'m Model,
        max_ctx: usize,
        kv: CacheKv,
        fit: bool,
        q6k8: bool,
    ) -> Result<MoeQwen3<'m, 'g>, String> {
        Self::new_with(
            gpu,
            module,
            router,
            m,
            max_ctx,
            kv,
            fit,
            q6k8,
            GpuExperts::from_env()?,
        )
    }

    /// `new` with `want` MoE layers' experts resident on the GPU (deepseek2;
    /// the default `N(0)` is `new`'s behaviour). Residency never changes
    /// a bit (the twins: `cuda::qwen2::experts`); where it couldn't keep
    /// them, it falls back to none and says why (`describe`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        router: &'g Module<'g>,
        m: &'m Model,
        max_ctx: usize,
        kv: CacheKv,
        fit: bool,
        q6k8: bool,
        want: GpuExperts,
    ) -> Result<MoeQwen3<'m, 'g>, String> {
        let repack = std::env::var("AXON_REPACK").map_or(true, |v| v != "0");
        let env = TwinEnv {
            repack,
            cpu_fused: crate::vmath::fused(),
        };
        Self::new_inner(gpu, module, router, m, max_ctx, kv, fit, q6k8, want, env)
    }

    /// `new_with` under `env` (tests force the refusals with it). The CPU
    /// half is built exactly once, after the placement: a build releases an
    /// in-memory model's bytes (`Model::release`), so neither a second build
    /// nor an upload may come after it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_inner(
        gpu: &'g Gpu,
        module: &'g Module<'g>,
        router: &'g Module<'g>,
        m: &'m Model,
        max_ctx: usize,
        kv: CacheKv,
        fit: bool,
        q6k8: bool,
        want: GpuExperts,
        env: TwinEnv,
    ) -> Result<MoeQwen3<'m, 'g>, String> {
        let net = CudaQwen2::with_kv_moe(gpu, module, m, max_ctx, kv, fit, q6k8)?;
        let kind = MoeKind::new(m, router, module)?;
        let (res, res_note) = match want {
            GpuExperts::N(0) => (None, None),
            _ => match place_resident(&net, &kind, gpu, module, m, want, env) {
                Ok(r) if !r.which().is_empty() => (Some(r), None),
                Ok(_) => (None, Some("none fit".to_string())),
                Err(e) => (None, Some(e)),
            },
        };
        let skip = res.as_ref().map_or(Vec::new(), |r| r.which());
        let moe_n = expert_layers(m).len();
        match skip.is_empty() {
            true => crate::model::moe_check_ram(m, false)?,
            false => crate::model::moe_check_ram_keep(m, moe_n - skip.len(), moe_n)?,
        }
        let cpu = Qwen2::new_experts_except(m, &skip);
        Ok(MoeQwen3 {
            ctx: net.ctx,
            gpu: net,
            cpu,
            kind,
            res,
            res_note,
            dev: gpu,
        })
    }
}

/// What residency's bit identity depends on (#121 item 5): whether the CPU
/// experts repack (AXON_REPACK) and the host's `vmath::fused()`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TwinEnv {
    pub repack: bool,
    pub cpu_fused: bool,
}

/// The resident layers `want` asks for that fit after a cache of `net`'s
/// planned context (ctx first), smallest first, uploaded from the mapping;
/// Err if residency can't keep the CPU's bits here or isn't supported. Reads
/// the model before any CPU build (which may release its bytes).
fn place_resident<'g>(
    net: &CudaQwen2<'g>,
    kind: &MoeKind<'g>,
    gpu: &'g Gpu,
    module: &'g Module<'g>,
    m: &Model,
    want: GpuExperts,
    env: TwinEnv,
) -> Result<Resident<'g>, String> {
    {
        if !matches!(kind, MoeKind::Mla(_)) {
            return Err(format!(
                "resident experts: deepseek2 only for now ({})",
                m.arch
            ));
        }
        experts::twin_identity(env.repack, env.cpu_fused, crate::vmath::fused())?;
        experts::twin_gemm_ok(gpu.info()?.cc)?;
        let layers = expert_layers(m);
        let d = net.moe.expect("a mixture-of-experts model");
        let dim = net.dim;
        // ctx first: the free memory with one cache of the planned context
        // (the GPU cache and the MLA scratch, as `new_cache` makes).
        let free = {
            let probe = Lm::new_cache(net, net.ctx);
            let max = CudaQwen2::batch_max(&probe);
            let _mb = net.mla_bufs(max, net.ctx)?;
            let f = gpu.info()?.free_bytes as usize;
            f.saturating_sub(ResBufs::bytes(max, dim, d.ffn, d.k))
        };
        let budget = free.saturating_sub(RESIDENT_HEADROOM);
        let pick = experts::place(&layers, want, budget);
        if let GpuExperts::N(n) = want
            && pick.len() < n
        {
            eprintln!(
                "moe:         --gpu-experts {n}: {} fit after a ctx-{} cache",
                pick.len(),
                net.ctx
            );
        }
        let r = Resident::upload(gpu, module, m, net.n_layers, &pick, d.k)?;
        for l in r.which() {
            let f = r.get(l).expect("resident").fused();
            experts::twin_identity(env.repack, env.cpu_fused, f)?;
        }
        Ok(r)
    }
}

impl<'g> MoeQwen3<'_, 'g> {
    /// The GPU half (identity, profiles).
    pub(crate) fn gpu_net(&self) -> &CudaQwen2<'g> {
        &self.gpu
    }

    pub(crate) fn kv_name(&self) -> &'static str {
        self.gpu.kv_name()
    }

    /// One line for the banner.
    pub(crate) fn describe(&self, max_ctx: usize) -> String {
        let d = self.gpu.moe.expect("a mixture-of-experts model");
        let n = self.gpu.mla_moe_layers().unwrap_or(self.gpu.n_layers);
        let res = match (&self.res, &self.res_note) {
            (Some(r), _) => format!(
                "; {} of {n} expert layers resident on the GPU ({:.2} GiB)",
                r.which().len(),
                r.bytes() as f64 / (1u64 << 30) as f64
            ),
            (None, Some(e)) => format!("; no experts resident: {e}"),
            _ => String::new(),
        };
        format!(
            "attention + router on the GPU ({:.2} GiB, ctx {max_ctx}), {} experts x {} \
             layers on the CPU (top-{}){res}",
            self.gpu.weight_bytes() as f64 / (1u64 << 30) as f64,
            d.n_expert,
            n - self.resident_layers().len(),
            d.k
        )
    }

    /// The resident layers (`--gpu-experts`).
    pub fn resident_layers(&self) -> Vec<usize> {
        self.res.as_ref().map_or(Vec::new(), |r| r.which())
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

    /// Every pass, by kind: the one dispatch. Steps return nothing.
    fn run(&self, c: &mut MoeCache<'g>, p: Pass) -> (Vec<u32>, Vec<f32>) {
        let mut ex =
            |l: usize, h: &[f32], i: &[i32], w: &[f32], y: &mut [f32]| self.experts(l, h, i, w, y);
        let g = &self.gpu;
        let none = |r: Result<(), String>| r.map(|_| (Vec::new(), Vec::new()));
        match (&self.kind, &mut c.s, p) {
            (MoeKind::Mla(xf), Scratch::Mla(xb, rb), Pass::Step(t, pos)) => {
                let res = self.res.as_ref().zip(rb.as_ref());
                none(g.mla_step(xf, &mut c.c, xb, t, pos, &mut ex, res))
            }
            (MoeKind::Mla(xf), Scratch::Mla(xb, rb), Pass::Tokens(ts, start, head)) => {
                let res = self.res.as_ref().zip(rb.as_ref());
                g.mla_tokens(xf, &mut c.c, xb, ts, start, head, &mut ex, res)
            }
            (k, Scratch::Moe(mb), Pass::Step(t, pos)) => {
                none(g.moe_step(k, &mut c.c, mb, t, pos, Head::Argmax, &mut ex))
            }
            (k, Scratch::Moe(mb), Pass::Tokens(ts, start, head)) => {
                g.moe_tokens(k, &mut c.c, mb, ts, start, head, &mut ex)
            }
            _ => unreachable!("a cache of another kind"),
        }
        .unwrap_or_else(|e| Self::die(e))
    }

    fn step(&self, token: u32, pos: usize, c: &mut MoeCache<'g>) {
        self.run(c, Pass::Step(token, pos));
    }

    fn tokens(
        &self,
        tokens: &[u32],
        start: usize,
        c: &mut MoeCache<'g>,
        head: Head,
    ) -> (Vec<u32>, Vec<f32>) {
        self.run(c, Pass::Tokens(tokens, start, head))
    }
}

/// VRAM left free after the resident experts (the driver, graphs, the
/// batch passes' transient buffers).
const RESIDENT_HEADROOM: usize = 512 << 20;

/// Each MoE layer's routed experts: (layer, bytes of its three tensors).
fn expert_layers(m: &Model) -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = Vec::new();
    for t in &m.tensors {
        if crate::model::MOE_EXPS.iter().any(|s| t.name.ends_with(s))
            && let Ok(l) = usize::try_from(t.layer)
        {
            match v.iter_mut().find(|e| e.0 == l) {
                Some(e) => e.1 += t.n_bytes as usize,
                None => v.push((l, t.n_bytes as usize)),
            }
        }
    }
    v.sort();
    v
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

impl<'g> crate::lm::Sealed for MoeQwen3<'_, 'g> {}
impl<'g> Lm for MoeQwen3<'_, 'g> {
    type Cache = MoeCache<'g>;

    fn vocab(&self) -> usize {
        self.gpu.vocab
    }

    fn new_cache(&self, max_ctx: usize) -> MoeCache<'g> {
        let c = Lm::new_cache(&self.gpu, max_ctx);
        let max = CudaQwen2::batch_max(&c);
        let s = match self.kind {
            MoeKind::Mla(_) => self.gpu.mla_bufs(max, max_ctx).and_then(|b| {
                let rb = match self.res.as_ref().and_then(|r| r.dims()) {
                    Some((dim, ffn, k)) => Some(ResBufs::new(self.dev, max, dim, ffn, k)?),
                    None => None,
                };
                Ok(Scratch::Mla(Box::new(b), rb))
            }),
            _ => self.gpu.moe_bufs(max).map(Scratch::Moe),
        }
        .unwrap_or_else(|e| Self::die(e));
        MoeCache { s, c }
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

#[cfg(test)]
mod router_legacy_tests;
