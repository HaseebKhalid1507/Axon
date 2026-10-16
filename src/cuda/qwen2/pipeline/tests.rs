use super::*;
use crate::cuda::Kernels;
use crate::cuda::qwen2::tests::PROMPTS;
use std::path::PathBuf;

fn mapped(rel: &str) -> Option<Model> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    if !p.exists() {
        eprintln!("skipping: {} not present", p.display());
        return None;
    }
    Some(crate::gguf::load_with(p.to_str().unwrap(), Some(false)).unwrap())
}

const M05: &str = crate::testutil::MODEL;
const M15: &str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
const Q3: &str = crate::testutil::QWEN3_MODEL;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `n` prompt tokens (the PROMPTS, repeated).
fn long_ids(m: &Model, n: usize) -> Vec<u32> {
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let mut ids = Vec::new();
    let mut i = 0;
    while ids.len() < n {
        ids.extend(tok.encode(&format!("{} ", PROMPTS[i % PROMPTS.len()])));
        i += 1;
    }
    ids.truncate(n);
    ids
}

/// Device 0 opened `n` times (one `Gpu` per stage: separate streams on the
/// same context, so a peer copy is a device-to-device copy), each with
/// its kernels.
fn devices(n: usize) -> Option<Vec<Gpu>> {
    crate::testutil::gpu()?;
    Some((0..n).map(|_| Gpu::open_device(0).unwrap()).collect())
}

/// What a run leaves: logits after the prompt and each greedy step, the
/// KV bits.
fn trace<M: Lm>(net: &M, ids: &[u32], chunk: usize, steps: usize) -> (Vec<Vec<u32>>, M::Cache) {
    let mut c = net.new_cache(ids.len() + steps + 1);
    let (&last, rest) = ids.split_last().unwrap();
    for (i, b) in rest.chunks(chunk).enumerate() {
        net.feed_batch(b, i * chunk, &mut c);
    }
    let mut out = Vec::new();
    let mut t = last;
    for s in 0..=steps {
        let l = net.forward(t, rest.len() + s, &mut c);
        t = crate::argmax(&l);
        out.push(bits(&l));
    }
    (out, c)
}

/// The cut sets the spec names: 2 stages at 1, 7 and n - 1; 3 stages.
fn cut_sets(n: usize) -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, n],
        vec![0, 7, n],
        vec![0, n - 1, n],
        vec![0, 1, 7, n],
        vec![0, 7, n - 1, n],
    ]
}

/// Two and three stages on one device give the one-GPU model's logits and
/// KV bytes, for prefill in passes of several sizes, then greedy decode
/// steps; and prefill_logits the same logits.
fn same_device_matches_cuda(rel: &str, kv: CacheKv) {
    let Some(m) = mapped(rel) else {
        return;
    };
    let Some(gs) = devices(3) else {
        return;
    };
    let kms: Vec<Module> = gs
        .iter()
        .map(|g| Kernels::module_for(g, &m).unwrap())
        .collect();
    let n = crate::hybrid::block_count(&m).unwrap();
    let ctx = 1400;
    let one = CudaQwen2::with_kv(&gs[0], &kms[0], &m, ctx, kv, false).unwrap();
    let ids = long_ids(&m, 300);
    for chunk in [1usize, 64, 299] {
        let (want, wc) = trace(&one, &ids, chunk, 6);
        let want_kv = layer_kv(&one, &wc);
        drop(wc);
        for cuts in cut_sets(n) {
            let ns = cuts.len() - 1;
            let devs: Vec<(&Gpu, &Module)> = (0..ns).map(|s| (&gs[s], &kms[s])).collect();
            let p = PipelineQwen2::new(&devs, &m, ctx, kv, Some(cuts.clone())).unwrap();
            assert!(p.p2p.iter().all(|&x| x), "one device: always direct");
            let (got, gc) = trace(&p, &ids, chunk, 6);
            assert!(got == want, "{rel} cuts {cuts:?} chunk {chunk}: logits");
            assert!(
                p.kv_layers(&gc) == want_kv,
                "{rel} cuts {cuts:?} chunk {chunk}: KV bytes"
            );
        }
    }
    // prefill_logits (the last token in the final pass, Head::Last).
    let (&last, rest) = ids.split_last().unwrap();
    let mut wc = one.new_cache(ids.len() + 1);
    let want = bits(&one.prefill_logits(&ids, 0, &mut wc));
    let mut fc = one.new_cache(ids.len() + 1);
    one.feed_batch(rest, 0, &mut fc);
    assert_eq!(bits(&Lm::forward(&one, last, rest.len(), &mut fc)), want);
    for cuts in cut_sets(n) {
        let ns = cuts.len() - 1;
        let devs: Vec<(&Gpu, &Module)> = (0..ns).map(|s| (&gs[s], &kms[s])).collect();
        let p = PipelineQwen2::new(&devs, &m, ctx, kv, Some(cuts.clone())).unwrap();
        let mut c = p.new_cache(ids.len() + 1);
        assert!(
            bits(&p.prefill_logits(&ids, 0, &mut c)) == want,
            "{rel} cuts {cuts:?}: prefill_logits"
        );
        assert!(p.kv_layers(&c) == layer_kv(&one, &wc));
    }
    eprintln!(
        "{rel} {kv:?}: pipelines == one GPU over {} cut sets",
        cut_sets(n).len()
    );
}

#[test]
fn pipeline_same_device_matches_cuda() {
    same_device_matches_cuda(M05, CacheKv::F16);
}

#[test]
fn pipeline_same_device_matches_cuda_1_5b() {
    same_device_matches_cuda(M15, CacheKv::F16);
}

#[test]
fn pipeline_same_device_matches_cuda_q8() {
    same_device_matches_cuda(M05, CacheKv::Q8 { group: 0 });
}

#[test]
fn pipeline_same_device_matches_cuda_qwen3() {
    same_device_matches_cuda(Q3, CacheKv::F16);
}

/// feed_batch in chunks of 3, 7 and 64 (each pipelined across the stages
/// in passes) leaves the KV bytes one call does, and the next logits.
#[test]
fn pipeline_feed_batch_matches_feed() {
    let Some(m) = mapped(M05) else {
        return;
    };
    let Some(gs) = devices(2) else {
        return;
    };
    let kms: Vec<Module> = gs.iter().map(|g| Kernels::module(g).unwrap()).collect();
    let n = crate::hybrid::block_count(&m).unwrap();
    let devs = [(&gs[0], &kms[0]), (&gs[1], &kms[1])];
    let p = PipelineQwen2::new(&devs, &m, 3000, CacheKv::F16, Some(vec![0, n / 2, n])).unwrap();
    // Past the batch max: one call is several pipelined passes.
    let ids = long_ids(&m, 2300);
    let (&last, rest) = ids.split_last().unwrap();
    let run = |chunk: usize| {
        let mut c = p.new_cache(ids.len() + 1);
        for (i, b) in rest.chunks(chunk).enumerate() {
            p.feed_batch(b, i * chunk, &mut c);
        }
        let l = bits(&Lm::forward(&p, last, rest.len(), &mut c));
        (l, p.kv_layers(&c))
    };
    let want = run(rest.len());
    for chunk in [3, 7, 64] {
        assert!(run(chunk) == want, "chunks of {chunk}");
    }
    eprintln!(
        "0.5B: feed_batch in chunks of 3, 7, 64 == one call of {}",
        rest.len()
    );
}

/// verify_argmax's picks equal forward_argmax per token and the one-GPU
/// model's verify, and leave the same KV, for right, wrong and random
/// drafts, small and past the batch max.
#[test]
fn pipeline_verify_matches_forward_argmax() {
    let Some(m) = mapped(M05) else {
        return;
    };
    let Some(gs) = devices(3) else {
        return;
    };
    let kms: Vec<Module> = gs.iter().map(|g| Kernels::module(g).unwrap()).collect();
    let n = crate::hybrid::block_count(&m).unwrap();
    let one = CudaQwen2::new(&gs[0], &kms[0], &m, 3000).unwrap();
    let prompt = long_ids(&m, 90);
    let (&last, rest) = prompt.split_last().unwrap();
    let start = rest.len();
    let mut seed = 77u64;
    let mut rnd = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n) as u32
    };
    let mut cases: Vec<Vec<u32>> = Vec::new();
    for nb in [1, 2, 5, 8, 9, 16] {
        cases.push(
            std::iter::once(last)
                .chain(rest[..nb - 1].iter().copied())
                .collect(),
        );
        cases.push((0..nb).map(|_| rnd(one.vocab as u64)).collect());
    }
    cases.push((0..1300).map(|_| rnd(5000)).collect());
    for cuts in [vec![0, n / 2, n], vec![0, 1, n - 1, n]] {
        let ns = cuts.len() - 1;
        let devs: Vec<(&Gpu, &Module)> = (0..ns).map(|s| (&gs[s], &kms[s])).collect();
        let p = PipelineQwen2::new(&devs, &m, 3000, CacheKv::F16, Some(cuts.clone())).unwrap();
        for toks in &cases {
            let ctx = start + toks.len() + 1;
            let (mut a, mut b, mut o) = (p.new_cache(ctx), p.new_cache(ctx), one.new_cache(ctx));
            p.feed_batch(rest, 0, &mut a);
            p.feed_batch(rest, 0, &mut b);
            one.feed_batch(rest, 0, &mut o);
            // Stale slots: a different pass over the same positions first.
            let junk: Vec<u32> = toks.iter().map(|t| (t + 1) % 1000).collect();
            p.verify_argmax(&junk, start, &mut a);
            let got = p.verify_argmax(toks, start, &mut a);
            let want: Vec<u32> = (0..toks.len())
                .map(|i| p.forward_argmax(toks[i], start + i, &mut b))
                .collect();
            assert_eq!(got, want, "cuts {cuts:?}: verify of {}", toks.len());
            assert_eq!(got, one.verify_argmax(toks, start, &mut o));
            let ka = p.kv_layers(&a);
            assert!(ka == p.kv_layers(&b), "cuts {cuts:?}: KV");
            assert!(ka == layer_kv(&one, &o), "cuts {cuts:?}: KV vs one GPU");
        }
    }
    eprintln!(
        "0.5B: {} verify cases == per-token picks == one GPU",
        cases.len()
    );
}

/// Sampling candidates (on the last stage's device) equal the one-GPU
/// model's, per token and per verify pass, top-k within CAND_MAX and past
/// it, with and without a repeat penalty.
#[test]
fn pipeline_cands_match_cuda() {
    let Some(m) = mapped(M05) else {
        return;
    };
    let Some(gs) = devices(2) else {
        return;
    };
    let kms: Vec<Module> = gs.iter().map(|g| Kernels::module(g).unwrap()).collect();
    let n = crate::hybrid::block_count(&m).unwrap();
    let one = CudaQwen2::new(&gs[0], &kms[0], &m, 600).unwrap();
    let devs = [(&gs[0], &kms[0]), (&gs[1], &kms[1])];
    let p = PipelineQwen2::new(&devs, &m, 600, CacheKv::F16, Some(vec![0, 9, n])).unwrap();
    let ids = long_ids(&m, 120);
    let (rest, tail) = ids.split_at(100);
    for spec in [
        CandSpec {
            k: 40,
            penalty: 1.1,
            last_n: 64,
        },
        CandSpec {
            k: 1,
            penalty: 1.0,
            last_n: 0,
        },
        CandSpec {
            k: CAND_MAX + 1,
            penalty: 1.3,
            last_n: 16,
        },
    ] {
        let (mut a, mut o) = (p.new_cache(200), one.new_cache(200));
        p.feed_batch(rest, 0, &mut a);
        one.feed_batch(rest, 0, &mut o);
        for (i, &t) in tail[..8].iter().enumerate() {
            let recent = &ids[..rest.len() + i + 1];
            let got = p.forward_cands(t, rest.len() + i, &mut a, &spec, recent);
            let want = one.forward_cands(t, rest.len() + i, &mut o, &spec, recent);
            assert!(got == want, "{spec:?}: forward_cands at {i}");
        }
        let toks = &tail[8..];
        let got = p.verify_cands(toks, 108, &mut a, &spec, &ids);
        let want = one.verify_cands(toks, 108, &mut o, &spec, &ids);
        assert!(got == want, "{spec:?}: verify_cands");
        assert!(p.kv_layers(&a) == layer_kv(&one, &o));
    }
}

/// A one-stage pipeline is the one-GPU model.
#[test]
fn pipeline_one_stage_is_cuda() {
    let Some(m) = mapped(M05) else {
        return;
    };
    let Some(gs) = devices(1) else {
        return;
    };
    let km = Kernels::module(&gs[0]).unwrap();
    let n = crate::hybrid::block_count(&m).unwrap();
    let one = CudaQwen2::new(&gs[0], &km, &m, 600).unwrap();
    let p = PipelineQwen2::new(&[(&gs[0], &km)], &m, 600, CacheKv::F16, None).unwrap();
    assert_eq!(p.cuts, vec![0, n]);
    let ids = long_ids(&m, 200);
    let (want, wc) = trace(&one, &ids, 64, 4);
    let (got, gc) = trace(&p, &ids, 64, 4);
    assert!(got == want);
    assert!(p.kv_layers(&gc) == layer_kv(&one, &wc));
}

/// Placement: even when it fits, else the same share of each device's
/// memory; Err when it can't fit; each stage gets a layer.
#[test]
fn pipeline_cuts_balance() {
    let gib = 1u64 << 30;
    // 64 layers of 0.3 GiB, the head 1 GiB on the last stage, 0.5 GiB fixed.
    let need = |ns: usize| {
        move |s: usize, a: usize, b: usize| -> Result<u64, String> {
            let head = if s + 1 == ns { gib } else { 0 };
            Ok(gib / 2 + (b - a) as u64 * 3 * gib / 10 + head)
        }
    };
    assert_eq!(
        pipeline_cuts(64, &[24 * gib, 24 * gib], 0, need(2)).unwrap(),
        vec![0, 32, 64]
    );
    assert_eq!(
        pipeline_cuts(64, &[24 * gib; 4], 0, need(4)).unwrap(),
        vec![0, 16, 32, 48, 64]
    );
    // 24 + 12 GiB: even (32 layers = 10.1 GiB + head on the 12) fits.
    assert_eq!(
        pipeline_cuts(64, &[24 * gib, 12 * gib], 0, need(2)).unwrap(),
        vec![0, 32, 64]
    );
    // 24 + 8: even doesn't fit on the second; both get the same share of
    // their memory (about 2/3: 51 layers + 13 and the head).
    let valid = |c: &[usize], free: &[u64], margin: u64, ns: usize| {
        check_cuts(c, 64, ns).unwrap();
        for s in 0..ns {
            assert!(
                need(ns)(s, c[s], c[s + 1]).unwrap() + margin <= free[s],
                "{c:?}"
            );
        }
    };
    let c = pipeline_cuts(64, &[24 * gib, 8 * gib], 0, need(2)).unwrap();
    valid(&c, &[24 * gib, 8 * gib], 0, 2);
    assert!((49..=53).contains(&c[1]), "{c:?}");
    // Margin on every device: 32 + 32 needs 10.1 / 11.1 GiB, so with 1 GiB
    // more kept, not even on 24 + 12.
    let c = pipeline_cuts(64, &[24 * gib, 12 * gib], gib, need(2)).unwrap();
    valid(&c, &[24 * gib, 12 * gib], gib, 2);
    assert!(c[1] > 32, "{c:?}");
    // Too little memory in all; not enough layers for the stages; no devices.
    assert!(pipeline_cuts(64, &[10 * gib, 10 * gib], 0, need(2)).is_err());
    assert!(pipeline_cuts(2, &[gib; 3], 0, need(3)).is_err());
    assert!(pipeline_cuts(2, &[], 0, need(1)).is_err());
    // A device that fits no layer.
    assert!(pipeline_cuts(8, &[gib / 4, 24 * gib], 0, need(2)).is_err());
    assert!(check_cuts(&[0, 5, 5, 10], 10, 3).is_err());
    assert!(check_cuts(&[0, 5, 10], 10, 2).is_ok());
    assert!(check_cuts(&[0, 5, 9], 10, 2).is_err());
    assert!(check_cuts(&[0], 0, 0).is_err());
    assert!(check_cuts(&[10], 10, 0).is_err());
}

/// AXON_PP_CUTS's parse: inner boundaries, checked against the layers and GPUs.
#[test]
fn pipeline_parse_cuts() {
    assert_eq!(parse_cuts("14", 28, 2).unwrap(), vec![0, 14, 28]);
    assert_eq!(parse_cuts(" 10, 20 ", 28, 3).unwrap(), vec![0, 10, 20, 28]);
    assert!(parse_cuts("14", 28, 3).is_err()); // too few cuts
    assert!(parse_cuts("0", 28, 2).is_err()); // an empty first stage
    assert!(parse_cuts("28", 28, 2).is_err()); // an empty last stage
    assert!(parse_cuts("20,10", 28, 3).is_err()); // not ascending
    assert!(parse_cuts("x", 28, 2).is_err());
    assert!(parse_cuts("", 28, 2).is_err());
}

/// Median and min of `v` (sorted in place).
fn med_min(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0])
}

/// The r7 §4 experiment on a multi-GPU box (devices 0 and 1; AXON_P2P=1
/// to try peer access, else the driver-staged path): peer access, hop latency per size (copy
/// then host wait), back-to-back copy bandwidth, and the TP all-reduce
/// loop: per "token" 128 rounds of a dim-float copy each way, a cross
/// event wait and a one-block kernel on each device. AXON_PP_DIM sets dim
/// (default 5120, 32B). Run: pp-test --ignored pp_probe --nocapture.
#[test]
#[ignore]
fn pp_probe() {
    let g0 = Gpu::open_device(0).unwrap();
    let Ok(g1) = Gpu::open_device(1) else {
        eprintln!("pp_probe: one device only");
        return;
    };
    for (a, b, name) in [(&g0, &g1, "0->1"), (&g1, &g0, "1->0")] {
        let mut can = 0;
        // SAFETY: valid devices; writes one c_int.
        unsafe { (a.d.cuDeviceCanAccessPeer)(&mut can, a.dev, b.dev) };
        eprintln!("probe: cuDeviceCanAccessPeer {name} = {can}");
    }
    let on = g0.enable_peer(&g1) & g1.enable_peer(&g0);
    eprintln!(
        "probe: peer access {} (AXON_P2P={})",
        if on { "ON" } else { "OFF (driver-staged)" },
        std::env::var("AXON_P2P").unwrap_or_default()
    );
    let dim: usize = std::env::var("AXON_PP_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5120);
    // Hop latency: one copy, then the host waits for it.
    for (bytes, iters) in [
        (14336usize, 1000usize),
        (20480, 1000),
        (327680, 1000),
        (21 << 20, 100),
        (268 << 20, 10),
    ] {
        let src = g0.alloc(bytes).unwrap();
        let dst = g1.alloc(bytes).unwrap();
        src.zero().unwrap();
        for _ in 0..3 {
            g0.copy_peer(&dst, 0, &src, 0, bytes).unwrap();
            g0.sync().unwrap();
        }
        let mut lat = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            g0.copy_peer(&dst, 0, &src, 0, bytes).unwrap();
            g0.sync().unwrap();
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let (med, min) = med_min(&mut lat);
        // Back to back: iters copies, one wait.
        let t = std::time::Instant::now();
        for _ in 0..iters {
            g0.copy_peer(&dst, 0, &src, 0, bytes).unwrap();
        }
        g0.sync().unwrap();
        let per = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        eprintln!(
            "probe: hop {bytes} B: copy+wait median {med:.1} us (min {min:.1}); \
             back-to-back {per:.1} us/copy = {:.2} GB/s",
            bytes as f64 / per / 1e3
        );
    }
    // The all-reduce loop.
    let (k0, k1) = (Kernels::module(&g0).unwrap(), Kernels::module(&g1).unwrap());
    let (f0, f1) = (
        k0.function("rmsnorm_quant").unwrap(),
        k1.function("rmsnorm_quant").unwrap(),
    );
    let ones = vec![1f32; dim];
    fn bufs<'g>(g: &'g Gpu, ones: &[f32]) -> [crate::cuda::DevBuf<'g>; 4] {
        [
            g.upload(ones).unwrap(),
            g.upload(ones).unwrap(),
            g.alloc(ones.len() * 4).unwrap(),
            g.alloc(super::super::act_bytes(ones.len())).unwrap(),
        ]
    }
    let [x0, w0, r0, a0] = bufs(&g0, &ones);
    let [x1, w1, r1, a1] = bufs(&g1, &ones);
    let (e0, e1) = (g0.new_event().unwrap(), g1.new_event().unwrap());
    let kernel = |g: &Gpu,
                  f: &crate::cuda::Func,
                  x: &crate::cuda::DevBuf,
                  w: &crate::cuda::DevBuf,
                  a: &crate::cuda::DevBuf| {
        // SAFETY: rmsnorm_quant(x, g, eps, n, want_k, act, null): x and w
        // hold dim floats, act act_bytes(dim).
        unsafe {
            g.launch(
                f,
                (1, 1, 1),
                (1024, 1, 1),
                0,
                &[
                    x.arg(),
                    w.arg(),
                    crate::cuda::Arg::F32(1e-6),
                    crate::cuda::Arg::I32(dim as i32),
                    crate::cuda::Arg::I32(dim.is_multiple_of(256) as i32),
                    a.arg(),
                    crate::cuda::Arg::Ptr(0),
                ],
            )
            .unwrap();
        }
    };
    let rounds = 128;
    for (label, copies) in [("kernels only", false), ("all-reduce", true)] {
        let mut per_tok = Vec::new();
        for _tok in 0..200 {
            let t = std::time::Instant::now();
            for _ in 0..rounds {
                kernel(&g0, &f0, &x0, &w0, &a0);
                kernel(&g1, &f1, &x1, &w1, &a1);
                if copies {
                    g0.copy_peer(&r1, 0, &x0, 0, dim * 4).unwrap();
                    g1.copy_peer(&r0, 0, &x1, 0, dim * 4).unwrap();
                    g0.record_main(&e0).unwrap();
                    g1.record_main(&e1).unwrap();
                    g0.wait_main(&e1).unwrap();
                    g1.wait_main(&e0).unwrap();
                }
            }
            g0.sync().unwrap();
            g1.sync().unwrap();
            per_tok.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (med, min) = med_min(&mut per_tok[20..]);
        eprintln!(
            "probe: {label}: {rounds} rounds of dim {dim}: median {med:.3} ms/token (min {min:.3}), \
             {:.1} us/round",
            med * 1e3 / rounds as f64
        );
    }
}
