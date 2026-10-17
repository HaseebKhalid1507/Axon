//! Pipeline parallel: a model's layers split across GPUs (stages), like
//! `crate::hybrid` with GPUs in place of the CPU. Stage s holds layers
//! `cuts[s]..cuts[s + 1]` (`CudaQwen2::new_stage`): the first one also the
//! embedding, the last one the output norm and head (argmax, sampling).
//!
//! Per token: each stage runs its step (its own captured graph), then
//! copies its hidden state (dim floats) into the next stage's (`Gpu::
//! copy_peer`: P2P if the devices allow it, else staged by the driver), and
//! records its `done` event; the next stage's stream waits for it. Batched
//! passes (prefill, verify) go the same way per chunk of tokens, and the
//! chunks are pipelined: stage s runs chunk i + 1 while stage s + 1 runs
//! chunk i (a stage's copy into the next waits only for the next stage's
//! previous chunk to be done with its buffers). One host thread drives every
//! device (`Gpu` is !Send).
//!
//! Numerics: every layer runs whole on one device with the same kernels as
//! on one GPU, and the copies move f32 bits, so the results (logits, picks,
//! sampling candidates, KV bytes) are the one-GPU model's bit for bit
//! (tests: two stages on one device). Results don't depend on the batching
//! (batch.rs contract), so the pipeline's chunk size doesn't matter either.

use super::batch::{BatchIn, passes};
use super::{CacheKv, CudaCache, CudaQwen2, Head, Input, PLAN_BATCH};
use crate::cuda::{Gpu, Module, SyncEvent};
use crate::gguf::Model;
use crate::lm::Lm;
use crate::sample::{self, CAND_MAX, Cand, CandSpec};

const GIB: f64 = (1u64 << 30) as f64;

/// One device's share of the model.
struct Stage<'g> {
    gpu: &'g Gpu,
    net: CudaQwen2<'g>,
    /// Recorded on this stage's stream after each piece of work (a step or
    /// pass and the copy of its output to the next stage): the next stage
    /// waits for it before reading its input, the previous one before
    /// overwriting this stage's input.
    done: SyncEvent<'g>,
}

pub struct PipelineQwen2<'g> {
    stages: Vec<Stage<'g>>,
    /// Layer boundaries: stage s holds layers cuts[s]..cuts[s + 1]
    /// (cuts[0] = 0, the last = n_layers).
    pub cuts: Vec<usize>,
    /// Per boundary s -> s + 1: whether the copy goes peer to peer (else
    /// the driver stages it through host memory).
    pub p2p: Vec<bool>,
}

/// Each stage's cache (its layers' KV and scratch).
pub struct PipelineCache<'g> {
    c: Vec<CudaCache<'g>>,
}

/// Where the layers go: `n` layers over devices with `free[s]` bytes free,
/// given `need(s, first, last)`, the VRAM stage s needs with layers
/// first..last (growing with the layer count), and `margin` bytes kept free
/// on each device besides. Even layer counts if they fit (the stages'
/// steps then take about as long: what overlapping chunks wants); else the
/// same share t of every device's free memory, as small as works: front
/// to back, each stage the most layers within t of its memory (leaving one
/// for each later stage), the last the rest. Err if not even all of every
/// device's memory holds the model. Pure: the tests drive it with a model
/// of the needs.
pub fn pipeline_cuts(
    n: usize,
    free: &[u64],
    margin: u64,
    mut need: impl FnMut(usize, usize, usize) -> Result<u64, String>,
) -> Result<Vec<usize>, String> {
    let ns = free.len();
    if ns == 0 || ns > n {
        return Err(format!("{n} layers over {ns} GPUs"));
    }
    let even: Vec<usize> = (0..=ns).map(|s| s * n / ns).collect();
    let mut ok = true;
    for s in 0..ns {
        if need(s, even[s], even[s + 1])? + margin > free[s] {
            ok = false;
            break;
        }
    }
    if ok {
        return Ok(even);
    }
    // The cuts with share t of each device's free memory, if they fit.
    let mut at = |t: f64| -> Result<Option<Vec<usize>>, String> {
        let budget = |s: usize| (free[s] as f64 * t) as u64;
        let mut cuts = vec![0];
        for s in 0..ns - 1 {
            let a = cuts[s];
            let (mut lo, mut hi) = (0, n - a - (ns - 1 - s));
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                if need(s, a, a + mid)? + margin <= budget(s) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            if lo == 0 {
                return Ok(None);
            }
            cuts.push(a + lo);
        }
        let a = cuts[ns - 1];
        if need(ns - 1, a, n)? + margin > budget(ns - 1) {
            return Ok(None);
        }
        cuts.push(n);
        Ok(Some(cuts))
    };
    let Some(mut best) = at(1.0)? else {
        return Err(format!(
            "the model doesn't fit on these {ns} GPUs ({:.2} GiB free in all)",
            free.iter().sum::<u64>() as f64 / GIB
        ));
    };
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..30 {
        let mid = (lo + hi) / 2.0;
        match at(mid)? {
            Some(c) => {
                best = c;
                hi = mid;
            }
            None => lo = mid,
        }
    }
    Ok(best)
}

/// Each device's free VRAM, shared evenly by the stages listed on it.
fn free_shares(devs: &[(&Gpu, &Module)]) -> Result<Vec<u64>, String> {
    let mut free = Vec::with_capacity(devs.len());
    for &(g, _) in devs {
        let share = devs.iter().filter(|(o, _)| o.same_device(g)).count();
        free.push(g.info()?.free_bytes / share as u64);
    }
    Ok(free)
}

/// `cuts` from AXON_PP_CUTS (the inner boundaries, "14" or "10,20"), if set.
fn cuts_env(n: usize, stages: usize) -> Result<Option<Vec<usize>>, String> {
    match std::env::var("AXON_PP_CUTS") {
        Ok(v) => parse_cuts(&v, n, stages)
            .map(Some)
            .map_err(|e| format!("AXON_PP_CUTS: {e}")),
        Err(_) => Ok(None),
    }
}

/// The cuts of n layers over `stages` GPUs from their inner boundaries
/// ("14", "10,20": the first layer of each GPU after the first).
fn parse_cuts(v: &str, n: usize, stages: usize) -> Result<Vec<usize>, String> {
    let inner: Vec<usize> = v
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("layer numbers separated by commas, got '{v}'"))?;
    let mut cuts = vec![0];
    cuts.extend(inner);
    cuts.push(n);
    check_cuts(&cuts, n, stages)?;
    Ok(cuts)
}

/// Ok if `cuts` splits n layers into `stages` (at least one) non-empty
/// ranges.
fn check_cuts(cuts: &[usize], n: usize, stages: usize) -> Result<(), String> {
    if stages == 0 {
        return Err("a pipeline of no GPUs".into());
    }
    if cuts.len() != stages + 1
        || cuts[0] != 0
        || cuts[stages] != n
        || cuts.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(format!(
            "{} inner cuts for {stages} GPUs, each of the {n} layers once, ascending",
            stages - 1
        ));
    }
    Ok(())
}

impl<'g> PipelineQwen2<'g> {
    /// The model over `devs` (in order: the first takes the embedding and
    /// first layers, the last the head), caches of type `kv` for max_ctx
    /// positions. `cuts`: the layer boundaries (`pipeline_cuts`' form), else
    /// AXON_PP_CUTS, else `pipeline_cuts` over each device's free VRAM (a
    /// device listed twice shares it evenly). Peer access is turned on
    /// between neighbors where the devices allow it.
    pub fn new(
        devs: &[(&'g Gpu, &'g Module<'g>)],
        m: &Model,
        max_ctx: usize,
        kv: CacheKv,
        cuts: Option<Vec<usize>>,
    ) -> Result<PipelineQwen2<'g>, String> {
        let ns = devs.len();
        let n = crate::hybrid::block_count(m)?;
        let cuts = match cuts {
            Some(c) => {
                check_cuts(&c, n, ns)?;
                c
            }
            None => match cuts_env(n, ns)? {
                Some(c) => c,
                None => {
                    let free = free_shares(devs)?;
                    pipeline_cuts(n, &free, 0, |s, a, b| {
                        let (g, km) = devs[s];
                        CudaQwen2::stage_need(g, km, m, max_ctx, a, b, kv, PLAN_BATCH)
                    })?
                }
            },
        };
        // The last stage first: on a device shared with an earlier stage
        // (tests), the first stage's optional extras (its bigger prefill
        // scratch, the Q6_K copy: `build`) only take what the rest leaves.
        let mut stages = Vec::with_capacity(ns);
        for s in (0..ns).rev() {
            let (gpu, km) = devs[s];
            let net =
                CudaQwen2::new_stage(gpu, km, m, max_ctx, cuts[s], cuts[s + 1], kv, PLAN_BATCH)
                    .map_err(|e| {
                        format!(
                            "pipeline stage {s} (GPU {}, layers {}..{}): {e}",
                            gpu.ordinal(),
                            cuts[s],
                            cuts[s + 1]
                        )
                    })?;
            stages.push(Stage {
                gpu,
                net,
                done: gpu.new_event()?,
            });
        }
        stages.reverse();
        let mut p2p = Vec::with_capacity(ns.saturating_sub(1));
        for w in stages.windows(2) {
            p2p.push(w[0].gpu.enable_peer(w[1].gpu));
        }
        Ok(PipelineQwen2 { stages, cuts, p2p })
    }

    /// The context the stages' caches are planned for (`new`'s max_ctx).
    pub fn ctx(&self) -> usize {
        self.stages[0].net.ctx
    }

    /// The longest context `new` (without `cuts`) places on `devs`: max_ctx
    /// if it fits, else the most (a multiple of 256) down to `min_ctx`;
    /// None if not even that. The same placement (AXON_PP_CUTS, else
    /// `pipeline_cuts`) and needs (`stage_need`), nothing allocated.
    pub fn fit_ctx(
        devs: &[(&'g Gpu, &'g Module<'g>)],
        m: &Model,
        max_ctx: usize,
        min_ctx: usize,
        kv: CacheKv,
    ) -> Result<Option<usize>, String> {
        let (ns, n) = (devs.len(), crate::hybrid::block_count(m)?);
        let (free, env) = (free_shares(devs)?, cuts_env(n, ns)?);
        let fits = |c: usize| {
            let need = |s: usize, a, b| {
                CudaQwen2::stage_need(devs[s].0, devs[s].1, m, c, a, b, kv, PLAN_BATCH)
            };
            match &env {
                Some(k) => (0..ns).all(|s| need(s, k[s], k[s + 1]).is_ok_and(|v| v <= free[s])),
                None => pipeline_cuts(n, &free, 0, need).is_ok(),
            }
        };
        if fits(max_ctx) {
            return Ok(Some(max_ctx));
        }
        // fits(k * 256) for k in lo..=hi only gets harder as k grows.
        let (mut lo, mut hi) = (min_ctx.div_ceil(256), (max_ctx - 1) / 256);
        if lo > hi || !fits(lo * 256) {
            return Ok(None);
        }
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if fits(mid * 256) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Ok(Some(lo * 256))
    }

    /// "pipeline 2 GPUs: layers 0..33 on 0 (11.20 GiB), 33..64 on 1 (...)"
    /// with a `max_ctx` cache, and how the hops go.
    pub fn describe(&self, max_ctx: usize) -> String {
        let parts: Vec<String> = self
            .stages
            .iter()
            .enumerate()
            .map(|(s, st)| {
                format!(
                    "layers {}..{} on {} ({:.2} GiB)",
                    self.cuts[s],
                    self.cuts[s + 1],
                    st.gpu.ordinal(),
                    (st.net.weight_bytes() + st.net.cache_bytes(max_ctx)) as f64 / GIB
                )
            })
            .collect();
        let hops = if self
            .stages
            .windows(2)
            .all(|w| w[0].gpu.same_device(w[1].gpu))
        {
            "on one device"
        } else if self.p2p.iter().all(|&p| p) {
            "p2p"
        } else if self.p2p.iter().any(|&p| p) {
            "p2p on some, host-staged on others"
        } else {
            "host-staged"
        };
        format!(
            "pipeline {} GPUs: {}; hops {hops}",
            self.stages.len(),
            parts.join(", ")
        )
    }

    /// Each stage's model, in order (its device: `CudaQwen2::gpu`).
    pub fn stages(&self) -> Vec<&CudaQwen2<'g>> {
        self.stages.iter().map(|s| &s.net).collect()
    }

    /// The cache type (`CudaQwen2::kv_name`; every stage's is the same).
    pub fn kv_name(&self) -> &'static str {
        self.stages[0].net.kv_name()
    }

    fn last(&self) -> &Stage<'g> {
        self.stages.last().expect("a stage")
    }

    fn die(e: String) -> ! {
        panic!("CUDA: {e}")
    }

    /// The caches belong to this pipeline's stages (step_with checks each
    /// one again; pass_with doesn't).
    fn check_cache(&self, c: &PipelineCache<'g>) {
        assert!(
            c.c.len() == self.stages.len()
                && self.stages.iter().zip(&c.c).all(|(s, c)| c.net == s.net.id),
            "a PipelineCache of another pipeline"
        );
    }

    /// Wait for every stage's stream (a fault anywhere surfaces here). Every
    /// `Lm` call ends here. First the last stage records its `done` after
    /// its last work (decode steps don't otherwise: `front` leaves the last
    /// step to the caller), so a stage that next copies into it waits for
    /// that work by its event, not only by this host wait.
    fn sync_all(&self) -> Result<(), String> {
        let last = self.last();
        last.gpu.record_main(&last.done)?;
        for s in &self.stages {
            s.gpu.sync()?;
        }
        Ok(())
    }

    /// `token` at `pos` through every stage but the last (enqueued, not
    /// waited for), the last's stream made to wait for its input: what the
    /// last stage's step starts from.
    fn front(
        &self,
        c: &mut PipelineCache<'g>,
        token: u32,
        pos: usize,
    ) -> Result<Input<'static>, String> {
        let ns = self.stages.len();
        if ns == 1 {
            return Ok(Input::Token(token));
        }
        for s in 0..ns - 1 {
            let st = &self.stages[s];
            let input = if s == 0 {
                Input::Token(token)
            } else {
                st.gpu.wait_main(&self.stages[s - 1].done)?;
                Input::Resident
            };
            st.net.step(&mut c.c[s], input, pos, false)?;
            // The next stage is done with its x (its previous step).
            st.gpu.wait_main(&self.stages[s + 1].done)?;
            st.gpu
                .copy_peer(&c.c[s + 1].x, 0, &c.c[s].x, 0, st.net.dim * 4)?;
            st.gpu.record_main(&st.done)?;
        }
        self.last().gpu.wait_main(&self.stages[ns - 2].done)?;
        Ok(Input::Resident)
    }

    /// Tokens at positions start.. (at most every stage's batch max)
    /// through every stage, the last with `head` (`row0`: its sampling
    /// rows, `pass_with`); enqueued, not waited for. Stage s's copy into
    /// stage s + 1 waits only for s + 1's previous pass, so successive
    /// calls overlap across the stages.
    fn pass_chunk(
        &self,
        c: &mut PipelineCache<'g>,
        tokens: &[u32],
        start: usize,
        head: Head,
        row0: usize,
    ) -> Result<(), String> {
        let ns = self.stages.len();
        let nb = tokens.len();
        for s in 0..ns {
            let st = &self.stages[s];
            let input = if s == 0 {
                BatchIn::Tokens(tokens)
            } else {
                st.gpu.wait_main(&self.stages[s - 1].done)?;
                BatchIn::Placed(nb)
            };
            let h = if s + 1 == ns { head } else { Head::No };
            st.net
                .pass_with(&mut c.c[s], input, start, h, row0, &mut |_| Ok(()))?;
            if s + 1 < ns {
                st.gpu.wait_main(&self.stages[s + 1].done)?;
                let src = c.c[s].b.as_ref().expect("batch scratch").x();
                let dst = c.c[s + 1].b.as_ref().expect("batch scratch").x();
                st.gpu.copy_peer(dst, 0, src, 0, nb * st.net.dim * 4)?;
            }
            st.gpu.record_main(&st.done)?;
        }
        Ok(())
    }

    /// Most tokens per pipelined chunk with these caches.
    fn chunk(&self, c: &PipelineCache<'g>) -> usize {
        c.c.iter().map(CudaQwen2::batch_max).min().unwrap_or(1)
    }

    /// `CudaQwen2::check_batch` for the pipeline: tokens in range, positions
    /// in the caches.
    fn check_batch(&self, tokens: &[u32], start: usize, c: &PipelineCache<'g>) {
        self.check_cache(c);
        let vocab = self.last().net.vocab;
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= vocab) {
            panic!("token {t} out of range");
        }
        CudaQwen2::check_positions(tokens.len(), start, &c.c[0]);
    }

    fn feed(&self, c: &mut PipelineCache<'g>, tokens: &[u32], start: usize) -> Result<(), String> {
        if tokens.len() == 1 {
            // The steps' graphs: fewer launches.
            let input = self.front(c, tokens[0], start)?;
            let last = self.stages.len() - 1;
            self.last().net.step(&mut c.c[last], input, start, false)?;
        } else {
            let max = self.chunk(c);
            for r in passes(tokens.len(), max) {
                self.pass_chunk(c, &tokens[r.clone()], start + r.start, Head::No, 0)?;
            }
        }
        self.sync_all()
    }

    fn verify(
        &self,
        c: &mut PipelineCache<'g>,
        tokens: &[u32],
        start: usize,
    ) -> Result<Vec<u32>, String> {
        let max = self.chunk(c);
        let last = self.stages.len() - 1;
        let mut picks = Vec::with_capacity(tokens.len());
        for r in passes(tokens.len(), max) {
            self.pass_chunk(c, &tokens[r.clone()], start + r.start, Head::Argmax, 0)?;
            let b = c.c[last].b.as_mut().expect("batch scratch");
            picks.extend(b.picks(r.len())?);
        }
        self.sync_all()?;
        Ok(picks)
    }

    fn prefill(
        &self,
        c: &mut PipelineCache<'g>,
        tokens: &[u32],
        start: usize,
    ) -> Result<Vec<f32>, String> {
        let max = self.chunk(c);
        let ps = passes(tokens.len(), max);
        for (i, r) in ps.iter().enumerate() {
            let head = if i + 1 == ps.len() {
                Head::Last
            } else {
                Head::No
            };
            self.pass_chunk(c, &tokens[r.clone()], start + r.start, head, 0)?;
        }
        let last = self.stages.len() - 1;
        let n = self.last().net.vocab * 4;
        let lc = &mut c.c[last];
        let b = lc.b.as_ref().expect("batch scratch");
        lc.host_out.copy_from(&b.logits, 0, n)?;
        let hb = &lc.host_out.bytes_mut()?[..n]; // waits for the passes and copy
        let l = hb
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect();
        self.sync_all()?;
        Ok(l)
    }

    fn cands(
        &self,
        c: &mut PipelineCache<'g>,
        token: u32,
        pos: usize,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Result<Vec<Cand>, String> {
        let last = self.stages.len() - 1;
        let net = &self.last().net;
        // The step's {token, pos}: a hidden state's token is 0 (`step_with`).
        let t = if last == 0 { token } else { 0 };
        net.samp_upload(&mut c.c[last], spec, recent, 1, 1, Some((t, pos)))?;
        let input = self.front(c, token, pos)?;
        net.step_with(&mut c.c[last], input, pos, Head::Sample)?;
        let row = net
            .samp_read(&mut c.c[last], spec.k, net.vocab, 1)?
            .pop()
            .expect("one row");
        self.sync_all()?;
        Ok(row)
    }

    fn verify_cands_tokens(
        &self,
        c: &mut PipelineCache<'g>,
        tokens: &[u32],
        start: usize,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Result<Vec<Vec<Cand>>, String> {
        let max = self.chunk(c);
        let last = self.stages.len() - 1;
        let net = &self.last().net;
        let nb = tokens.len();
        net.samp_upload(&mut c.c[last], spec, recent, nb, nb.min(max), None)?;
        let mut rows = Vec::with_capacity(nb);
        for r in passes(nb, max) {
            self.pass_chunk(
                c,
                &tokens[r.clone()],
                start + r.start,
                Head::Sample,
                r.start,
            )?;
            rows.extend(net.samp_read(&mut c.c[last], spec.k, net.vocab, r.len())?);
        }
        self.sync_all()?;
        Ok(rows)
    }

    /// Each layer's KV bytes (keys, values, then a q8 cache's scales), in
    /// layer order over the stages (tests: `layer_kv` of the one-GPU cache).
    #[cfg(test)]
    pub(crate) fn kv_layers(&self, c: &PipelineCache) -> Vec<Vec<u8>> {
        self.stages
            .iter()
            .zip(&c.c)
            .flat_map(|(s, c)| layer_kv(&s.net, c))
            .collect()
    }
}

/// Cache `c`'s KV bytes per layer of `net` (`CudaCache::kv`'s layout: f16
/// K, V per layer; q8 int8 K, V per layer, then every layer's scales).
#[cfg(test)]
pub(crate) fn layer_kv(net: &CudaQwen2, c: &CudaCache) -> Vec<Vec<u8>> {
    let all: Vec<u8> = c.kv.read().unwrap();
    let nl = net.layers.len();
    let n = c.max_ctx * net.kv_dim();
    match c.mode {
        CacheKv::F16 => all.chunks(4 * n).map(<[u8]>::to_vec).collect(),
        CacheKv::Q8 { group } => {
            let sn = n / group * 4;
            let (vals, scales) = all.split_at(2 * nl * n);
            vals.chunks(2 * n)
                .zip(scales.chunks(2 * sn))
                .map(|(v, s)| [v, s].concat())
                .collect()
        }
    }
}

impl<'g> Lm for PipelineQwen2<'g> {
    type Cache = PipelineCache<'g>;

    fn vocab(&self) -> usize {
        self.last().net.vocab
    }

    fn new_cache(&self, max_ctx: usize) -> PipelineCache<'g> {
        let c = self
            .stages
            .iter()
            .map(|s| s.net.try_cache(max_ctx))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|e| Self::die(e));
        PipelineCache { c }
    }

    fn forward(&self, token: u32, pos: usize, c: &mut PipelineCache<'g>) -> Vec<f32> {
        self.check_cache(c);
        let last = self.stages.len() - 1;
        self.front(c, token, pos)
            .and_then(|input| self.last().net.logits(&mut c.c[last], input, pos))
            .and_then(|l| self.sync_all().map(|_| l))
            .unwrap_or_else(|e| Self::die(e))
    }

    fn forward_argmax(&self, token: u32, pos: usize, c: &mut PipelineCache<'g>) -> u32 {
        self.check_cache(c);
        let last = self.stages.len() - 1;
        self.front(c, token, pos)
            .and_then(|input| self.last().net.pick(&mut c.c[last], input, pos))
            .and_then(|p| self.sync_all().map(|_| p))
            .unwrap_or_else(|e| Self::die(e))
    }

    fn feed_batch(&self, tokens: &[u32], start: usize, c: &mut PipelineCache<'g>) {
        self.check_batch(tokens, start, c);
        self.feed(c, tokens, start).unwrap_or_else(|e| Self::die(e));
    }

    fn prefill_logits(&self, tokens: &[u32], start: usize, c: &mut PipelineCache<'g>) -> Vec<f32> {
        self.check_batch(tokens, start, c);
        if tokens.len() == 1 {
            return Lm::forward(self, tokens[0], start, c);
        }
        self.prefill(c, tokens, start)
            .unwrap_or_else(|e| Self::die(e))
    }

    fn verify_argmax(&self, tokens: &[u32], start: usize, c: &mut PipelineCache<'g>) -> Vec<u32> {
        self.check_batch(tokens, start, c);
        self.verify(c, tokens, start)
            .unwrap_or_else(|e| Self::die(e))
    }

    fn forward_cands(
        &self,
        token: u32,
        pos: usize,
        c: &mut PipelineCache<'g>,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Vec<Cand> {
        assert_eq!(
            recent.last(),
            Some(&token),
            "recent must end with the token"
        );
        if spec.k > CAND_MAX {
            let mut l = Lm::forward(self, token, pos, c);
            return sample::cands(&mut l, spec, recent);
        }
        self.check_cache(c);
        self.cands(c, token, pos, spec, recent)
            .unwrap_or_else(|e| Self::die(e))
    }

    fn verify_cands(
        &self,
        tokens: &[u32],
        start: usize,
        c: &mut PipelineCache<'g>,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Vec<Vec<Cand>> {
        self.check_batch(tokens, start, c);
        let (nb, n) = (tokens.len(), recent.len());
        assert!(
            n >= nb && recent[n - nb..] == *tokens,
            "recent must end with tokens"
        );
        if spec.k > CAND_MAX {
            return (0..nb)
                .map(|b| {
                    let r = &recent[..n - nb + b + 1];
                    self.forward_cands(tokens[b], start + b, c, spec, r)
                })
                .collect();
        }
        self.verify_cands_tokens(c, tokens, start, spec, recent)
            .unwrap_or_else(|e| Self::die(e))
    }

    /// The smallest stage's batch (the first stage may have the bigger
    /// `PREFILL_BATCH`).
    fn prefill_batch(&self) -> usize {
        self.stages.iter().map(|s| s.net.batch).min().unwrap_or(1)
    }
}

#[cfg(test)]
mod tests;
