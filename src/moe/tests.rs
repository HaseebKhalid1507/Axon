//! MoE stage 2 tests: `moe_route` on its own (batch invariance, ties),
//! decode == batched prefill == verify bit for bit on the dev models, and the
//! GPU-attention backend graded against the CPU-only one.

use super::*;
use crate::cuda::qwen2::tests::{Agree, PROMPTS, assert_graded, compare, report};
use crate::cuda::{Arg, Kernels};

struct Rng(u64);
impl Rng {
    /// Uniform in [-1, 1).
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
    }
}

/// The GPU router (moe.cu `moe_logits` + `moe_topk`) over nb tokens x
/// [nb][dim], router `wr` [n_e][dim]: (h, ids, w).
fn route_gpu(
    g: &Gpu,
    mm: &Module,
    x: &[f32],
    gain: &[f32],
    wr: &[f32],
    n_e: usize,
    k: usize,
) -> (Vec<f32>, Vec<i32>, Vec<f32>) {
    let dim = gain.len();
    let nb = x.len() / dim;
    let (fl, ft) = (
        mm.function("moe_logits").unwrap(),
        mm.function("moe_topk").unwrap(),
    );
    let (xd, gd, wd) = (
        g.upload(x).unwrap(),
        g.upload(gain).unwrap(),
        g.upload(wr).unwrap(),
    );
    let h = g.alloc(nb * dim * 4).unwrap();
    let lg = g.alloc(nb * n_e * 4).unwrap();
    let ids = g.alloc(nb * k * 4).unwrap();
    let w = g.alloc(nb * k * 4).unwrap();
    // SAFETY: the kernels' signatures; buffers sized for nb tokens.
    unsafe {
        g.launch(
            &fl,
            (n_e.div_ceil(8) as u32, nb as u32, 1),
            (256, 1, 1),
            (dim * 4) as u32,
            &[
                xd.arg(),
                gd.arg(),
                Arg::F32(1e-6),
                Arg::I32(dim as i32),
                wd.arg(),
                Arg::I32(n_e as i32),
                h.arg(),
                lg.arg(),
            ],
        )
        .unwrap();
        g.launch(
            &ft,
            (nb as u32, 1, 1),
            (128, 1, 1),
            (n_e * 4) as u32,
            &[
                lg.arg(),
                Arg::I32(n_e as i32),
                Arg::I32(k as i32),
                ids.arg(),
                w.arg(),
            ],
        )
        .unwrap();
    }
    g.sync().unwrap();
    (h.read().unwrap(), ids.read().unwrap(), w.read().unwrap())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Spec r11 stage 2 gate (a): a token's normed state, experts and weights
/// are the same bits alone (nb = 1) as in a batch of 512; ties go to the
/// lower expert index (in rank order, equal weights); the weights sum to 1.
/// The 30B's shapes (dim 2048, 128 experts, top 8) and the 4x0.6B's (1024,
/// 4, top 2).
#[test]
fn moe_route_batch_invariant_and_ties() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let mm = Kernels::moe_module(&g).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (dim, n_e, k) in [(2048, 128, 8), (1024, 4, 2), (256, 200, 6)] {
        let nb = 512;
        let x: Vec<f32> = (0..nb * dim).map(|_| 3.0 * rng.f()).collect();
        let gain: Vec<f32> = (0..dim).map(|_| 1.0 + 0.5 * rng.f()).collect();
        let mut wt: Vec<f32> = (0..dim * n_e).map(|_| 0.05 * rng.f()).collect();
        let (h, ids, w) = route_gpu(&g, &mm, &x, &gain, &wt, n_e, k);
        for t in [0, 1, 77, 300, 511] {
            let (h1, i1, w1) = route_gpu(&g, &mm, &x[t * dim..(t + 1) * dim], &gain, &wt, n_e, k);
            assert_eq!(
                bits(&h1),
                bits(&h[t * dim..(t + 1) * dim]),
                "h of token {t}"
            );
            assert_eq!(i1, ids[t * k..(t + 1) * k], "ids of token {t}");
            assert_eq!(bits(&w1), bits(&w[t * k..(t + 1) * k]), "w of token {t}");
        }
        for t in 0..nb {
            let (it, wt_) = (&ids[t * k..(t + 1) * k], &w[t * k..(t + 1) * k]);
            let mut s = it.to_vec();
            s.sort();
            s.dedup();
            assert_eq!(s.len(), k, "distinct experts");
            assert!(it.iter().all(|&e| (e as usize) < n_e));
            assert!(wt_.windows(2).all(|p| p[0] >= p[1]), "rank order {wt_:?}");
            let sum: f32 = wt_.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "weights sum {sum}");
        }
        // Ties: experts a < b get identical router columns, scaled up so
        // they win: both picked, a first, the same weight.
        let (a, b) = (n_e / 3, n_e - 1);
        for c in 0..dim {
            let v = wt[a * dim + c] + 0.2 * x[c].signum();
            wt[a * dim + c] = v;
            wt[b * dim + c] = v;
        }
        let (_, i1, w1) = route_gpu(&g, &mm, &x[..dim], &gain, &wt, n_e, k);
        let pa = i1.iter().position(|&e| e as usize == a).expect("a picked");
        let pb = i1.iter().position(|&e| e as usize == b).expect("b picked");
        assert_eq!(pb, pa + 1, "tie: lower index first {i1:?}");
        assert_eq!(w1[pa].to_bits(), w1[pb].to_bits());
        // Every expert the same: the first k in index order.
        let flat: Vec<f32> = (0..dim * n_e).map(|i| 0.01 * (i % dim) as f32).collect();
        let (_, i1, w1) = route_gpu(&g, &mm, &x[..dim], &gain, &flat, n_e, k);
        assert_eq!(i1, (0..k as i32).collect::<Vec<_>>());
        assert!(w1.iter().all(|v| v.to_bits() == w1[0].to_bits()));
    }
}

/// The MoE backend of `m` on `g`.
pub(crate) fn moe_net<'m, 'g>(
    g: &'g Gpu,
    km: &'g Module<'g>,
    mm: &'g Module<'g>,
    m: &'m Model,
    ctx: usize,
) -> MoeQwen3<'m, 'g> {
    MoeQwen3::new(g, km, mm, m, ctx, CacheKv::F16, false).unwrap()
}

/// `model::tests::feed_batch_matches_feed_on` for the MoE backend: 483
/// tokens decoded one by one vs fed in batches of every verify size, odd
/// ones and past the GPU's small-batch kernels: the KV caches and the next
/// logits bit for bit.
fn feed_batch_matches_feed(small: bool) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::moe_model(small) else {
        return;
    };
    let km = Kernels::module_for(&g, &m).unwrap();
    let mm = Kernels::moe_module(&g).unwrap();
    let net = moe_net(&g, &km, &mm, &m, 512);
    let v = net.vocab() as u64;
    let ids: Vec<u32> = (0..483u64)
        .map(|i| (i * 2_654_435_761 % 150_000 % v) as u32)
        .collect();
    let next = (9707 % v) as u32;
    let mut a = net.new_cache(ids.len() + 1);
    let mut b = net.new_cache(ids.len() + 1);
    for (pos, &id) in ids.iter().enumerate() {
        net.forward_argmax(id, pos, &mut a);
    }
    let mut s = 0;
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 64, 86, 128, 130] {
        net.feed_batch(&ids[s..s + n], s, &mut b);
        s += n;
    }
    assert_eq!(s, ids.len());
    let ka = CudaQwen2::kv_bits(&a.c);
    let kb = CudaQwen2::kv_bits(&b.c);
    assert!(ka == kb, "KV caches differ");
    let la = net.forward(next, ids.len(), &mut a);
    let lb = net.forward(next, ids.len(), &mut b);
    assert_eq!(bits(&la), bits(&lb), "logits differ");
    // prefill_logits (batched passes, the head on the last row) == feed +
    // forward.
    let mut c = net.new_cache(ids.len() + 1);
    let lc = net.prefill_logits(&ids[..200], 0, &mut c);
    let mut d = net.new_cache(ids.len() + 1);
    net.feed_batch(&ids[..199], 0, &mut d);
    let ld = net.forward(ids[199], 199, &mut d);
    assert_eq!(
        bits(&lc),
        bits(&ld),
        "prefill_logits != feed_batch + forward"
    );
}

#[test]
fn moe_gpu_feed_batch_matches_feed_pruned() {
    feed_batch_matches_feed(false);
}

#[test]
fn moe_gpu_feed_batch_matches_feed_small() {
    feed_batch_matches_feed(true);
}

/// verify_argmax (batched, with the head) == forward_argmax per token, and
/// the caches it leaves the same bits, after a batched prefill.
fn verify_matches_forward(small: bool) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::moe_model(small) else {
        return;
    };
    let km = Kernels::module_for(&g, &m).unwrap();
    let mm = Kernels::moe_module(&g).unwrap();
    let net = moe_net(&g, &km, &mm, &m, 128);
    let ids: Vec<u32> = (0..60u64)
        .map(|i| (i * 2_654_435_761 % 150_000) as u32)
        .collect();
    let (pre, tail) = ids.split_at(28);
    for tail in [&tail[..1], &tail[..5], &tail[..16], tail] {
        let mut a = net.new_cache(ids.len());
        net.feed_batch(pre, 0, &mut a);
        let mut b = net.new_cache(ids.len());
        net.feed_batch(pre, 0, &mut b);
        let picks = net.verify_argmax(tail, pre.len(), &mut a);
        for (j, &t) in tail.iter().enumerate() {
            let f = net.forward(t, pre.len() + j, &mut b);
            assert_eq!(picks[j], crate::argmax(&f), "row {j} of {}", tail.len());
        }
        assert!(
            CudaQwen2::kv_bits(&a.c) == CudaQwen2::kv_bits(&b.c),
            "verify's cache"
        );
    }
}

#[test]
fn moe_gpu_verify_matches_forward_pruned() {
    verify_matches_forward(false);
}

#[test]
fn moe_gpu_verify_matches_forward_small() {
    verify_matches_forward(true);
}

/// Spec r11 stage 2 gate (c): the GPU-attention MoE vs the CPU-only MoE,
/// graded as the GPU is against the CPU (`cuda::qwen2::tests::logits_match`):
/// logits after each prompt and 96 greedy steps (AXON_TEST_MOE_STEPS),
/// against the CPU's own spread between its two kernel sets (r8 vs
/// per-row). 96, not the dense tests' 32: at 32 (198 positions) the
/// Pruned-2B's top-1 was 189 vs the CPU's self-agreement 193 and its mean KL
/// 2.4x the CPU's (7.8e-3 vs 3.2e-3), at 96 (582 positions) 568 vs 568 and
/// 1.76x (5.2e-3 vs 2.9e-3): the same flips (near-ties, the widest 0.89 and
/// 0.61) dominate a short run. MEASURED, bella sm_120. The 4x0.6B passes at
/// 32 too (191 vs 191, 7.1e-3 vs 6.8e-3).
fn logits_graded(small: bool, name: &str) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    // Mapped: the per-row CPU model reads the experts after the repacking
    // ones release their pages.
    let Some(path) = crate::testutil::moe_model_path(small) else {
        return;
    };
    let m = crate::gguf::load_with(path.to_str().unwrap(), Some(false)).unwrap();
    let km = Kernels::module_for(&g, &m).unwrap();
    let mm = Kernels::moe_module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    // Per-row first: the repacked models release the bytes they repack.
    let rows = Qwen2::new_with(&m, false);
    let net = moe_net(&g, &km, &mm, &m, 256);
    let cpu = Qwen2::new(&m);
    let steps = std::env::var("AXON_TEST_MOE_STEPS").map_or(96, |v| v.parse().unwrap());
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    for p in PROMPTS {
        let ids = tok.encode(p);
        compare(&cpu, &net, &ids, steps, &mut ag);
        compare(&cpu, &rows, &ids, steps, &mut ac);
    }
    report(&format!("{name} moe gpu vs cpu"), &ag);
    report(&format!("{name} cpu rows vs cpu"), &ac);
    assert_graded(&ag, &ac);
}

#[test]
fn moe_gpu_logits_graded_pruned() {
    logits_graded(false, "Pruned-2B");
}

#[test]
fn moe_gpu_logits_graded_small() {
    logits_graded(true, "4x0.6B");
}

/// A MoE model through `Backend::new` on the GPU: the MoE backend (no
/// layer split: --gpu-layers below the layer count is refused).
#[test]
fn backend_places_moe() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(m) = crate::testutil::moe_model(true) else {
        return;
    };
    use crate::lm::{Backend, Device};
    let b = Backend::new(&m, Device::Cuda, 512, None, crate::kv::KvType::F16, false).unwrap();
    assert!(matches!(b, Backend::Moe(_)));
    assert_eq!(b.ctx(512), 512);
    let e = Backend::new(
        &m,
        Device::Cuda,
        512,
        Some(3),
        crate::kv::KvType::F16,
        false,
    );
    assert!(e.is_err_and(|e| e.contains("doesn't split by layers")));
}

/// The MoE backend through a shuffled page table (`CudaCache::set_tab`,
/// paged KV #64: the attention's _pg kernels) runs the same script as a
/// plain cache with the same picks (batched prefill in two passes, graphed
/// decode, a verify pass). And the decode graphs, kept beside the cache,
/// go when the cache drops its own (`drop_graphs`: a table set, a view
/// moved), then are made again with the same results.
#[test]
fn moe_gpu_paged_and_graph_generation() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::moe_model(true) else {
        return;
    };
    let km = Kernels::module_for(&g, &m).unwrap();
    let mm = Kernels::moe_module(&g).unwrap();
    const PAGE: usize = 64;
    let ctx = 640;
    let net = moe_net(&g, &km, &mm, &m, ctx);
    let v = net.vocab() as u64;
    let prompt: Vec<u32> = (0..300u64)
        .map(|i| (i * 2_654_435_761 % 150_000 % v) as u32)
        .collect();
    fn script<'g>(
        net: &MoeQwen3<'_, 'g>,
        prompt: &[u32],
        c: &mut MoeCache<'g>,
        drop_at: Option<usize>,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        net.feed_batch(&prompt[..200], 0, c);
        net.feed_batch(&prompt[200..250], 200, c);
        let mut tok = prompt[250];
        for i in 0..20 {
            if drop_at == Some(i) {
                let held = c.mb.graphs_held();
                assert!(held.0 > 0, "graphs captured by now");
                c.c.drop_graphs();
            }
            tok = net.forward_argmax(tok, 250 + i, c);
            out.push(tok);
            if drop_at == Some(i) {
                let (n, at) = c.mb.graphs_held();
                assert!(
                    n > 0 && at == c.c.graph_gen,
                    "graphs remade for the new generation"
                );
            }
        }
        out.extend(net.verify_argmax(&out[..5].to_vec(), 270, c));
        out
    }
    let mut c0 = net.new_cache(ctx);
    let want = script(&net, &prompt, &mut c0, None);
    let mut c2 = net.new_cache(ctx);
    assert_eq!(
        script(&net, &prompt, &mut c2, Some(7)),
        want,
        "after drop_graphs"
    );
    let pages = ctx / PAGE;
    let phys = pages + pages / 2 + 3;
    let mut pool: Vec<u32> = (0..phys as u32).rev().collect();
    pool.truncate(pages);
    let mut c1 = net.new_cache(phys * PAGE);
    c1.c.set_tab(Some(g.upload(&pool).unwrap()));
    assert_eq!(script(&net, &prompt, &mut c1, None), want, "paged cache");
}
