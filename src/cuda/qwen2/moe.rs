//! qwen3moe with attention on the GPU and the experts on the CPU (r11
//! stage 2): the GPU half of `crate::moe::MoeQwen3`.
//!
//! Per layer the GPU runs the attention sub-block (`enqueue_attn` /
//! `attn_half`, the dense kernels unchanged) and the router (`moe_route`:
//! moe.cu `moe_logits`, the FFN norm and a warp per expert's logit, then
//! `moe_topk`, softmax, exact top k, renorm; fixed op orders per token, so a
//! token's bits don't depend on its batch). One copy brings the normed
//! states, expert ids and weights to the host; the caller's CPU experts
//! (`Qwen2::moe_layer[_batch]`) make y; one copy takes y back and `moe_add`
//! adds it into the residual stream before the next layer. KV lives on the
//! GPU only.
//!
//! Decode replays a captured graph per segment (layer i's add of y_{i-1},
//! attention and router; the last segment the add and the head), so the
//! round trip per layer is two copies, a sync and one graph launch.
//! Batched passes (verify, prefill) launch kernel by kernel, as the dense
//! batched passes do.
//!
//! Exactness: decode and batched passes share the attention's bits (G3:
//! batched == decode), `moe_route` is per token, and the CPU experts give a
//! token the same bits alone or in a batch: decode == feed_batch == verify,
//! bit for bit.

use super::batch::{BatchIn, Place};
use super::*;

/// moe.cu's kernels (`Kernels::moe_module`).
pub(crate) struct MoeFns<'m> {
    logits: Func<'m>,
    topk: Func<'m>,
    /// gpt-oss: the biased logits' top k, softmax over those k.
    topk_oai: Func<'m>,
    add: Func<'m>,
}

impl<'m> MoeFns<'m> {
    pub(crate) fn new(module: &'m Module<'m>) -> Result<MoeFns<'m>, String> {
        Ok(MoeFns {
            logits: module.function("moe_logits")?,
            topk: module.function("moe_topk")?,
            topk_oai: module.function("moe_topk_oai")?,
            add: module.function("moe_add")?,
        })
    }
}

/// AXON_MOE_PROFILE's report interval, in decode steps.
const PROFILE_EVERY: f64 = 64.0;

/// AXON_MOE_PROFILE's report: per layer and the sum.
fn profile_report(p: &[Prof]) {
    const NAMES: [&str; 5] = ["gpu", "down", "in", "experts", "up"];
    let mut sum = [0.0; 5];
    for (l, x) in p.iter().enumerate() {
        let n = x[5];
        let cols: Vec<String> = (0..5)
            .map(|i| {
                sum[i] += x[i] / n;
                format!("{} {:.1}", NAMES[i], x[i] / n * 1e6)
            })
            .collect();
        eprintln!(
            "[moe profile] layer {l} (us): {} ({n} steps)",
            cols.join(", ")
        );
    }
    let cols: Vec<String> = (0..5)
        .map(|i| format!("{} {:.3}", NAMES[i], sum[i] * 1e3))
        .collect();
    eprintln!(
        "[moe profile] per step (ms): {}, {} layers",
        cols.join(", "),
        p.len()
    );
}

/// AXON_MOE_PROFILE's seconds per layer: the GPU segment (synced for the
/// profile), the copy down and its wait, the host's unpacking, the CPU
/// experts, y's packing and copy up; then the steps.
type Prof = [f64; 6];

/// moe.cu MOE_THREADS (moe_topk) and MOE_LG_WARPS (moe_logits).
const MOE_THREADS: u32 = 128;
const MOE_LG_WARPS: usize = 8;

/// A cache's MoE scratch: for up to `max` tokens the router's output
/// (normed states [nb][dim] f32, then ids [nb][k] i32, then weights [nb][k]
/// f32, packed for nb), the experts' y [nb][dim], their pinned host twins,
/// and the decode step's graphs.
pub struct MoeBufs<'g> {
    max: usize,
    r: DevBuf<'g>,
    /// The router logits, [max][n_expert] f32.
    lg: DevBuf<'g>,
    y: DevBuf<'g>,
    host_r: HostBuf<'g>,
    host_y: HostBuf<'g>,
    /// Per segment (`n_layers + 1` of them) 4 variants: layers by cmb64,
    /// the last by `Head`.
    graphs: Vec<Option<Graph<'g>>>,
    /// The cache's `graph_gen` when `graphs` were captured: a different
    /// one (a table set, a view moved) drops them before a replay.
    graph_gen: u64,
    /// AXON_MOE_PROFILE=1: decode steps' `Prof` per layer, told on
    /// stderr every `PROFILE_EVERY` steps (the segments synced: slower).
    prof: Option<Vec<Prof>>,
}

impl MoeBufs<'_> {
    /// Decode graphs held, and the cache generation they were made for.
    #[cfg(test)]
    pub(crate) fn graphs_held(&self) -> (usize, u64) {
        (self.graphs.iter().flatten().count(), self.graph_gen)
    }
}

/// What the CPU experts get per layer: the layer, nb normed states (token
/// major), each token's k expert ids (rank order) and weights; they write
/// y, token major.
pub(crate) type Experts<'a> = dyn FnMut(usize, &[f32], &[i32], &[f32], &mut [f32]) + 'a;

/// `b` (4-byte aligned, a whole number of floats) as floats in place.
fn as_f32(b: &[u8]) -> &[f32] {
    // SAFETY: any 4 bytes are a valid f32.
    let (pre, v, post) = unsafe { b.align_to::<f32>() };
    assert!(pre.is_empty() && post.is_empty(), "f32 alignment");
    v
}

/// `b` (4-byte aligned, a whole number of ints) as i32s in place.
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

impl<'g> CudaQwen2<'g> {
    fn moe_k(&self) -> (usize, usize) {
        let d = self.moe.expect("a mixture-of-experts model");
        (d.n_expert, d.k)
    }

    /// Scratch for passes of up to `max` tokens (the cache's batch max).
    pub(crate) fn moe_bufs(&self, max: usize) -> Result<MoeBufs<'g>, String> {
        let (n_e, k) = self.moe_k();
        let (rb, yb) = (max * (self.dim + 2 * k) * 4, max * self.dim * 4);
        let g = self.gpu;
        Ok(MoeBufs {
            max,
            r: g.alloc(rb)?,
            lg: g.alloc(max * n_e * 4)?,
            y: g.alloc(yb)?,
            host_r: g.host_alloc(rb)?,
            host_y: g.host_alloc(yb)?,
            graphs: (0..(self.n_layers + 1) * 4).map(|_| None).collect(),
            graph_gen: 0,
            prof: std::env::var("AXON_MOE_PROFILE")
                .is_ok_and(|v| v == "1")
                .then(|| vec![[0.0; 6]; self.n_layers]),
        })
    }

    /// The router's outputs for nb tokens in `mb.r`: (h, ids, w) args.
    fn route_args(&self, mb: &MoeBufs, nb: usize) -> (Arg, Arg, Arg) {
        let (_, k) = self.moe_k();
        let hb = nb * self.dim * 4;
        (mb.r.arg(), mb.r.arg_at(hb), mb.r.arg_at(hb + nb * k * 4))
    }

    /// Enqueue the router of layer li over nb states at x into `mb.r`:
    /// moe.cu `moe_logits` (the FFN norm and the logits) then `moe_topk`.
    ///
    /// # Safety
    /// x holds nb rows of dim floats; nb <= mb.max.
    unsafe fn moe_route(
        &self,
        mf: &MoeFns,
        mb: &MoeBufs,
        li: usize,
        x: Arg,
        nb: usize,
    ) -> Result<(), String> {
        let (n_e, k) = self.moe_k();
        let l = &self.layers[li];
        let Ffn::Moe { router } = l.ffn else {
            panic!("layer {li} is not a mixture-of-experts layer")
        };
        assert!(nb <= mb.max);
        let (h, ids, w) = self.route_args(mb, nb);
        // SAFETY: the kernels' signatures; the router is [n_e][dim] floats
        // at `router` in vecs, the norm dim floats at ffn_norm; mb.r holds
        // nb tokens' h, ids and w (route_args), mb.lg nb rows of n_e logits;
        // dynamic shared memory dim (moe_logits) and n_e (moe_topk) floats.
        unsafe {
            self.gpu.launch(
                &mf.logits,
                (n_e.div_ceil(MOE_LG_WARPS) as u32, nb as u32, 1),
                ((MOE_LG_WARPS * 32) as u32, 1, 1),
                (self.dim * 4) as u32,
                &[
                    x,
                    self.vecs.arg_at(l.ffn_norm * 4),
                    Arg::F32(self.eps),
                    Arg::I32(self.dim as i32),
                    self.vecs.arg_at(router * 4),
                    Arg::I32(n_e as i32),
                    h,
                    mb.lg.arg(),
                ],
            )?;
            if let Some(o) = l.oss {
                // gpt-oss: the router's bias (n_e floats), moe_topk_oai.
                return self.gpu.launch(
                    &mf.topk_oai,
                    (nb as u32, 1, 1),
                    (MOE_THREADS, 1, 1),
                    (n_e * 4) as u32,
                    &[
                        mb.lg.arg(),
                        self.vecs.arg_at(o.router_b * 4),
                        Arg::I32(n_e as i32),
                        Arg::I32(k as i32),
                        ids,
                        w,
                    ],
                );
            }
            self.gpu.launch(
                &mf.topk,
                (nb as u32, 1, 1),
                (MOE_THREADS, 1, 1),
                (n_e * 4) as u32,
                &[
                    mb.lg.arg(),
                    Arg::I32(n_e as i32),
                    Arg::I32(k as i32),
                    ids,
                    w,
                ],
            )
        }
    }

    /// Enqueue x += mb.y for nb tokens.
    ///
    /// # Safety
    /// x holds nb rows of dim floats; nb <= mb.max.
    unsafe fn moe_add(&self, mf: &MoeFns, mb: &MoeBufs, x: Arg, nb: usize) -> Result<(), String> {
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

    /// The host round trip of layer li after its `moe_route` (enqueued):
    /// the router's output down, the CPU experts, y up (enqueued, ordered
    /// before what comes next on the stream).
    fn moe_round_trip(
        &self,
        mb: &mut MoeBufs<'g>,
        li: usize,
        nb: usize,
        experts: &mut Experts,
    ) -> Result<(), String> {
        let (_, k) = self.moe_k();
        let dim = self.dim;
        let n = nb * (dim + 2 * k) * 4;
        let t0 = std::time::Instant::now();
        mb.host_r.copy_from(&mb.r, 0, n)?;
        let b = &mb.host_r.bytes_mut()?[..n]; // waits for the layer and copy
        let t1 = std::time::Instant::now();
        // The pinned buffers are page-aligned: read and write them as
        // floats (and ints) in place.
        let (hb, rest) = b.split_at(nb * dim * 4);
        let (ib, wb) = rest.split_at(nb * k * 4);
        let (h, ids, w) = (as_f32(hb), as_i32(ib), as_f32(wb));
        // SAFETY: host_r's wait above synced the stream, after host_y's
        // last copy (the previous layer's y): no second wait for it.
        unsafe { mb.host_y.assume_synced() };
        let yb = &mut mb.host_y.bytes_mut()?[..nb * dim * 4];
        // SAFETY: any bytes are a valid f32; the prefix is empty (asserted).
        let (pre, y, _) = unsafe { yb.align_to_mut::<f32>() };
        assert!(pre.is_empty(), "pinned buffer alignment");
        let t2 = std::time::Instant::now();
        experts(self.first + li, h, ids, w, y);
        let t3 = std::time::Instant::now();
        mb.host_y.copy_to(&mb.y, 0, nb * dim * 4)?;
        if let (Some(p), 1) = (&mut mb.prof, nb) {
            let t4 = std::time::Instant::now();
            for (i, d) in [t1 - t0, t2 - t1, t3 - t2, t4 - t3].iter().enumerate() {
                p[li][1 + i] += d.as_secs_f64();
            }
            p[li][5] += 1.0;
            if li + 1 == p.len() && p[li][5] >= PROFILE_EVERY {
                profile_report(p);
                p.iter_mut().for_each(|x| *x = [0.0; 6]);
            }
        }
        Ok(())
    }

    /// One decode step of `token` at `pos` (logits and the argmax into the
    /// cache; `head` must be `Head::Argmax`), the experts by `experts`. Waits for every
    /// layer's router; the head is enqueued, not waited for.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_step(
        &self,
        mf: &MoeFns,
        c: &mut CudaCache<'g>,
        mb: &mut MoeBufs<'g>,
        token: u32,
        pos: usize,
        head: Head,
        experts: &mut Experts,
    ) -> Result<(), String> {
        assert_eq!(c.net, self.id, "a CudaCache used with another CudaQwen2");
        // Head::No would only add the last layer's FFN into x, which the
        // next step's embedding overwrites: no caller wants it.
        assert!(head == Head::Argmax, "moe_step: {head:?}");
        assert!(
            pos < c.max_ctx,
            "position {pos} exceeds cache size {}",
            c.max_ctx
        );
        c.debug_covers(0, pos + 1);
        assert!((token as usize) < self.vocab, "token {token} out of range");
        let p = c.host_params.bytes_mut()?;
        p[0..4].copy_from_slice(&token.to_le_bytes());
        p[4..8].copy_from_slice(&(pos as u32).to_le_bytes());
        c.host_params.copy_to(&c.params, 0, 8)?;
        let cmb64 = cmb64_at(pos);
        let n = self.layers.len();
        if mb.graph_gen != c.graph_gen {
            mb.graphs.iter_mut().for_each(|g| *g = None);
            mb.graph_gen = c.graph_gen;
        }
        for seg in 0..=n {
            let c = &*c;
            let run = |mb: &MoeBufs| -> Result<(), String> {
                let params = c.params.arg();
                let on = &mut |_: &'static str, _: usize| Ok(());
                // SAFETY: x holds dim floats, mb one token's scratch.
                unsafe {
                    if seg == 0 {
                        self.enqueue_input(c, params, on)?;
                    } else {
                        self.moe_add(mf, mb, c.x.arg(), 1)?;
                    }
                    if seg < n {
                        self.enqueue_attn(c, seg, params, cmb64, on)?;
                        self.moe_route(mf, mb, seg, c.x.arg(), 1)?;
                    } else {
                        self.enqueue_head(c, head, on)?;
                    }
                }
                Ok(())
            };
            let ts = std::time::Instant::now();
            let v = seg * 4
                + if seg < n {
                    cmb64 as usize
                } else {
                    head as usize
                };
            if self.graphs {
                if mb.graphs[v].is_none() {
                    let g = self.gpu.capture(|| run(mb))?;
                    mb.graphs[v] = Some(g);
                }
                // SAFETY: the graph's buffers are the cache's, the scratch's
                // and the model's, all borrowed for this call; the scratch
                // owns the graph and lives with the cache.
                unsafe { mb.graphs[v].as_ref().unwrap().launch()? };
            } else {
                run(mb)?;
            }
            if seg < n {
                if let Some(p) = &mut mb.prof {
                    self.gpu.sync()?;
                    p[seg][0] += ts.elapsed().as_secs_f64();
                }
                self.moe_round_trip(mb, seg, 1, experts)?;
            }
        }
        Ok(())
    }

    /// The logits after `moe_step` with `Head::Argmax`.
    pub(crate) fn moe_logits(&self, c: &mut CudaCache<'g>) -> Result<Vec<f32>, String> {
        let n = self.vocab * 4;
        c.host_out.copy_from(&c.logits, 0, n)?;
        Ok(f32s(&c.host_out.bytes_mut()?[..n]))
    }

    /// The greedy pick after `moe_step` with `Head::Argmax`.
    pub(crate) fn moe_pick(&self, c: &mut CudaCache<'g>) -> Result<u32, String> {
        c.host_out.copy_from(&c.best, 0, 4)?;
        let b = &c.host_out.bytes_mut()?[..4];
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// `feed_batch` (head None), `verify_argmax` (Argmax: the picks) or
    /// `prefill_logits` (Last: the logits) of `tokens` at start.., in
    /// passes of up to the cache's batch max.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_tokens(
        &self,
        mf: &MoeFns,
        c: &mut CudaCache<'g>,
        mb: &mut MoeBufs<'g>,
        tokens: &[u32],
        start: usize,
        head: Head,
        experts: &mut Experts,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        self.check_batch(tokens, start, c);
        let max = c.b.as_ref().expect("batch scratch").max.min(mb.max);
        let ps = match head {
            Head::No | Head::Last => self.prompt_passes(c, tokens.len(), start),
            _ => batch::passes(tokens.len(), max),
        }
        .into_iter()
        .flat_map(|r| {
            batch::passes(r.len(), max)
                .into_iter()
                .map(move |q| r.start + q.start..r.start + q.end)
        })
        .collect::<Vec<_>>();
        let mut picks = Vec::new();
        for (i, r) in ps.iter().enumerate() {
            let h = match head {
                Head::Last if i + 1 < ps.len() => Head::No,
                h => h,
            };
            let nb = r.len();
            assert!(nb <= mb.max);
            // The dense pass (embedding, attention, head), each MoE layer's
            // FFN here: the router, the host round trip, x += y.
            let mut ffn = |li: usize, x: Arg, n: usize| -> Result<(), String> {
                // SAFETY: x holds the pass's n <= mb.max rows.
                unsafe { self.moe_route(mf, mb, li, x, n)? };
                self.moe_round_trip(mb, li, n, experts)?;
                // SAFETY: as above.
                unsafe { self.moe_add(mf, mb, x, n) }
            };
            let input = BatchIn::Tokens(&tokens[r.clone()]);
            let place = Place::Seq {
                start: start + r.start,
                base: 0,
            };
            let on = &mut |_: &'static str| Ok(());
            self.pass_hooked(c, input, place, h, None, 0, on, Some(&mut ffn))?;
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
mod tests {
    use super::*;
    use crate::cuda::Kernels;

    /// A dense batched pass (`pass_at`: no MoE hook, as a branch that
    /// forgets `moe_tokens` would call it) on a MoE model is an Err, not
    /// `LayerW::dense`'s panic.
    #[test]
    fn moe_gpu_dense_pass_is_an_err() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let Some(m) = crate::testutil::moe_model(true) else {
            return;
        };
        let km = Kernels::module_for(&g, &m).unwrap();
        let net = CudaQwen2::with_kv_moe(&g, &km, &m, 256, CacheKv::F16, false, false).unwrap();
        let mut c = net.try_cache(256).unwrap();
        let tokens = [1u32, 2, 3];
        let e = net.pass_with(
            &mut c,
            BatchIn::Tokens(&tokens),
            0,
            Head::Argmax,
            0,
            &mut |_| Ok(()),
        );
        assert!(e.is_err_and(|e| e.contains("mixture-of-experts")));
    }

    /// The dense decode step (`step`, `profile`) on a MoE model is an Err
    /// before anything is written or captured: with graphs on, the stream
    /// is not left in capture mode, so a `moe_step` after it runs.
    #[test]
    fn moe_gpu_dense_step_is_an_err() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let Some(m) = crate::testutil::moe_model(true) else {
            return;
        };
        let km = Kernels::module_for(&g, &m).unwrap();
        let mm = Kernels::moe_module(&g).unwrap();
        let mut net = CudaQwen2::with_kv_moe(&g, &km, &m, 256, CacheKv::F16, false, false).unwrap();
        net.graphs = true;
        let mut c = net.try_cache(256).unwrap();
        let moe_err = |r: Result<(), String>| r.is_err_and(|e| e.contains("mixture-of-experts"));
        assert!(moe_err(net.step(&mut c, Input::Token(1), 0, true)));
        assert!(moe_err(net.profile(&mut c, 1, 0).map(|_| ())));
        let mf = MoeFns::new(&mm).unwrap();
        let mut mb = net.moe_bufs(CudaQwen2::batch_max(&c)).unwrap();
        // Zero experts: only the stream's health is under test.
        let mut ex = |_: usize, _: &[f32], _: &[i32], _: &[f32], y: &mut [f32]| y.fill(0.0);
        net.moe_step(&mf, &mut c, &mut mb, 1, 0, Head::Argmax, &mut ex)
            .unwrap();
        assert!((net.moe_pick(&mut c).unwrap() as usize) < net.vocab);
    }
}
