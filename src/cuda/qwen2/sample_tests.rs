//! sample.cu against sample.rs: the candidates the GPU selects are the
//! CPU's for the same logits (ids, order, logit bits), on synthetic rows
//! built to break it (ties, NaN, infinities, signed zeros, short rows) and
//! on the model's own; and sampling on the GPU keeps drafting lossless.

use super::*;
use crate::cuda::Kernels;
use crate::sample::{self, Cand, CandSpec, SampleArgs, SampleCfg};

fn bits(c: &[Cand]) -> Vec<(u32, u32)> {
    c.iter().map(|c| (c.logit.to_bits(), c.id)).collect()
}

fn xorshift(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

/// Rows of `n` synthetic logits, `round` picking the kind.
fn synthetic(n: usize, round: usize, s: &mut u32) -> Vec<f32> {
    let specials = [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.0, 0.0];
    (0..n)
        .map(|_| {
            let r = xorshift(s);
            match round % 6 {
                0 => r as f32 / 1e7 - 200.0,
                1 => (r % 7) as f32 - 3.0, // heavy ties
                2 if r.is_multiple_of(13) => specials[(r >> 8) as usize % 5],
                2 => (r % 1000) as f32 / 37.0,
                3 if r.is_multiple_of(3) => f32::NAN, // few non-NaN
                3 => (r % 5) as f32,
                4 => 1.5,                             // all equal
                _ => (r % 2000) as f32 / 100.0 - 4.0, // logit-like
            }
        })
        .collect()
}

/// samp_tail on uploaded rows == sample::cands per row, for row lengths
/// around the slice size and the vocab, k up to CAND_MAX, penalty on and
/// off, several rows per launch (row offsets into S).
#[test]
fn kernels_match_cpu_candidates() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::new(&g, &km, &m, 64).unwrap();
    let mut c = net.new_cache(64);
    let v = net.vocab;
    let mut s = 0x5eed_1234u32;
    let mut checked = 0;
    for n in [
        1usize, 2, 39, 40, 41, 255, 256, 257, 2047, 2048, 2049, 4097, 50_000, v,
    ] {
        for round in 0..6 {
            for (k, penalty, rows) in [(1, 1.0, 1), (40, 1.1, 3), (256, 1.3, 2), (7, 0.7, 16)] {
                if rows * n > batch::HEAD_SUB * v {
                    continue;
                }
                let spec = CandSpec {
                    k,
                    penalty,
                    last_n: 64,
                };
                let ls: Vec<f32> = (0..rows)
                    .flat_map(|_| synthetic(n, round, &mut s))
                    .collect();
                // recent: random ids below n, with repeats; row b's window
                // ends at its token (the last `rows` of recent).
                let recent: Vec<u32> = (0..70 + rows)
                    .map(|i| xorshift(&mut s) % n.min(5 + i) as u32)
                    .collect();
                net.samp_upload(&mut c, &spec, &recent, rows, rows, None)
                    .unwrap();
                let lg = &c.b.as_ref().unwrap().logits;
                lg.write(&ls).unwrap();
                net.samp_tail(c.samp.as_ref().unwrap(), lg.arg(), n, rows, 0, 0, false)
                    .unwrap();
                let got = net.samp_read(&mut c, k, n, rows).unwrap();
                let nr = recent.len();
                for b in 0..rows {
                    let mut l = ls[b * n..(b + 1) * n].to_vec();
                    let want = sample::cands(&mut l, &spec, &recent[..nr - rows + b + 1]);
                    assert_eq!(
                        bits(&got[b]),
                        bits(&want),
                        "n {n} round {round} k {k} p {penalty} row {b}/{rows}"
                    );
                    checked += 1;
                }
            }
        }
    }
    eprintln!("{checked} rows checked");
}

/// On the model's logits: forward_cands == sample::cands(forward), and the
/// batched verify's rows == forward_cands per token, for batch sizes past a
/// head launch (16 rows) and with penalty windows that run into the batch.
#[test]
fn model_candidates_match() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::new(&g, &km, &m, 256).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let ids = tok.encode("The quick brown fox jumps over the lazy dog. The quick brown fox");
    let toks: Vec<u32> = (0..17u32).map(|i| 500 + i * 7919 % 60_000).collect();
    for spec in [
        CandSpec {
            k: 40,
            penalty: 1.1,
            last_n: 64,
        },
        CandSpec {
            k: 256,
            penalty: 1.0,
            last_n: 64,
        },
        CandSpec {
            k: 1,
            penalty: 1.5,
            last_n: 8,
        },
        // Past CAND_MAX: the host path.
        CandSpec {
            k: 1000,
            penalty: 1.1,
            last_n: 64,
        },
        CandSpec {
            k: sample::ALL,
            penalty: 1.1,
            last_n: 64,
        },
    ] {
        let mut a = net.new_cache(256);
        let (&last, rest) = ids.split_last().unwrap();
        net.feed_batch(rest, 0, &mut a);
        let mut b = net.new_cache(256);
        net.feed_batch(rest, 0, &mut b);
        // The whole sequence: prompt, then toks fed one by one.
        let mut seq = ids.clone();
        let mut want = Vec::new();
        for (i, &t) in std::iter::once(&last).chain(&toks).enumerate() {
            if i > 0 {
                seq.push(t);
            }
            let pos = rest.len() + i;
            let mut l = net.forward(t, pos, &mut a);
            let cpu = sample::cands(&mut l, &spec, &seq);
            let gpu = net.forward_cands(t, pos, &mut a, &spec, &seq);
            assert_eq!(bits(&gpu), bits(&cpu), "{spec:?} pos {pos}");
            assert_eq!(gpu[0].id, crate::argmax(&l));
            want.push(gpu);
        }
        // Verify batches starting at the last prompt token.
        for nb in [1usize, 2, 4, 8, 12, 16, 17] {
            let batch: Vec<u32> = std::iter::once(last)
                .chain(toks.iter().copied())
                .take(nb)
                .collect();
            let recent = &seq[..ids.len() + nb - 1];
            let got = net.verify_cands(&batch, rest.len(), &mut b, &spec, recent);
            assert_eq!(got.len(), nb);
            for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(bits(g), bits(w), "{spec:?} nb {nb} row {r}");
            }
        }
    }
}

/// sample.rs's real-model checks (drafting == plain sampling, seeds) on
/// the GPU's forward_cands / verify_cands.
#[test]
fn real_model_sampling_cuda() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::new(&g, &km, &m, 256).unwrap();
    crate::sample::tests::real_model_sampling_on(&net, &m);
}

/// CPU and GPU sampling with one seed: the same tokens until the logits
/// differ. At the first token that differs, the two backends' candidates
/// for that position (same history) differ too.
#[test]
fn cpu_and_gpu_differ_only_where_logits_do() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let gpu = CudaQwen2::new(&g, &km, &m, 256).unwrap();
    let cpu = crate::model::Qwen2::new(&m);
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let (mut same, mut diverged) = (0, 0);
    for p in [
        "The capital of France is",
        "Once upon a time, in a small village by the sea, there lived",
        "Q: What is 17 times 23?\nA:",
    ] {
        let ids = tok.encode(p);
        for seed in 1..=4u64 {
            let cfg: SampleCfg = SampleArgs {
                seed: Some(seed),
                ..Default::default()
            }
            .resolve(sample::DEFAULT_TEMP)
            .unwrap();
            let n = 32;
            let a = crate::sample::tests::run(&cpu, &ids, n, &[], 0, cfg, "", &[]);
            let b = crate::sample::tests::run(&gpu, &ids, n, &[], 0, cfg, "", &[]);
            let Some(i) = (0..n).find(|&i| a.ids[i] != b.ids[i]) else {
                same += 1;
                continue;
            };
            diverged += 1;
            // Both backends' candidates at token i after the common prefix,
            // and the uniform number token i drew (one per token).
            let mut seq = ids.clone();
            seq.extend_from_slice(&a.ids[..i]);
            let spec = cfg.spec();
            let mut lc = cpu.logits_after(&seq);
            let mut lg = gpu.logits_after(&seq);
            let cc = sample::cands(&mut lc, &spec, &seq);
            let gc = sample::cands(&mut lg, &spec, &seq);
            let mut rng = sample::Rng::new(seed);
            for _ in 0..i {
                rng.f64();
            }
            let u = rng.f64();
            let ctx = format!("{p:?} seed {seed} token {i}");
            assert_eq!(sample::pick(&cfg, &cc, u), a.ids[i], "{ctx}: cpu");
            assert_eq!(sample::pick(&cfg, &gc, u), b.ids[i], "{ctx}: gpu");
            assert_ne!(bits(&cc), bits(&gc), "{ctx}");
            let d = lc
                .iter()
                .zip(&lg)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            eprintln!("{ctx}: diverged, u {u:.4}, max |logit diff| {d:.4}");
        }
    }
    eprintln!("{same} runs identical, {diverged} diverged where the logits did");
}

/// Logits after a whole sequence, on a fresh cache.
trait LogitsAfter: Lm {
    fn logits_after(&self, seq: &[u32]) -> Vec<f32> {
        let mut c = self.new_cache(seq.len() + 1);
        let (&last, rest) = seq.split_last().unwrap();
        if !rest.is_empty() {
            self.feed_batch(rest, 0, &mut c);
        }
        self.forward(last, rest.len(), &mut c)
    }
}

impl<M: Lm> LogitsAfter for M {}

/// Per-token cost of sampling on the GPU: `forward_argmax` (greedy) vs
/// `forward_cands` (penalty + top-k on the device, k pairs back), same
/// token and position, alternating rounds; medians. AXON_TEST_MODEL picks
/// the model. `--ignored --nocapture` to see.
#[test]
#[ignore]
fn step_cost_cuda() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::new(&g, &km, &m, 512).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let ids = tok.encode("The history of the Roman Empire");
    let mut c = net.new_cache(512);
    let (&last, rest) = ids.split_last().unwrap();
    net.feed_batch(rest, 0, &mut c);
    let pos = rest.len();
    let recent: Vec<u32> = (0..64).map(|i| 1000 + i * 37).chain([last]).collect();
    let specs = [
        (
            "cands k40 pen1.1",
            CandSpec {
                k: 40,
                penalty: 1.1,
                last_n: 64,
            },
        ),
        (
            "cands k256 pen1.0",
            CandSpec {
                k: 256,
                penalty: 1.0,
                last_n: 64,
            },
        ),
    ];
    let n = 200;
    let mut t_greedy = Vec::new();
    let mut t_s: Vec<Vec<f64>> = vec![Vec::new(); specs.len()];
    for round in 0..9 {
        let t = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(net.forward_argmax(last, pos, &mut c));
        }
        let g_us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        let mut s_us = Vec::new();
        for (_, spec) in &specs {
            let t = std::time::Instant::now();
            for _ in 0..n {
                std::hint::black_box(net.forward_cands(last, pos, &mut c, spec, &recent));
            }
            s_us.push(t.elapsed().as_secs_f64() * 1e6 / n as f64);
        }
        if round > 0 {
            t_greedy.push(g_us);
            for (v, x) in t_s.iter_mut().zip(s_us) {
                v.push(x);
            }
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let g_med = med(&mut t_greedy);
    eprintln!("{} | greedy step {g_med:.1} us", m.arch);
    for ((name, _), v) in specs.iter().zip(&mut t_s) {
        let s = med(v);
        eprintln!(
            "{name}: {s:.1} us, +{:.1} us ({:+.2}%)",
            s - g_med,
            100.0 * (s - g_med) / g_med
        );
    }
}

/// GPU time of each sample.cu kernel on a real logits row (events between
/// launches, so no overlap), and the host side of `forward_cands` beyond
/// the step. `--ignored --nocapture` to see.
#[test]
#[ignore]
fn samp_kernel_profile() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::new(&g, &km, &m, 64).unwrap();
    let mut c = net.new_cache(64);
    let v = net.vocab;
    for k in [40usize, 256] {
        let spec = CandSpec {
            k,
            penalty: 1.1,
            last_n: 64,
        };
        let recent: Vec<u32> = (0..65).map(|i| 1000 + i * 37).collect();
        let mut t = Vec::new();
        for _ in 0..60 {
            net.forward(100, 3, &mut c);
            net.samp_upload(&mut c, &spec, &recent, 1, 1, None).unwrap();
            let s = c.samp.as_ref().unwrap();
            let e0 = g.record().unwrap();
            net.samp_tail(s, c.logits.arg(), v, 1, 0, 0, false).unwrap();
            let e1 = g.record().unwrap();
            // The greedy step's argmax, for scale.
            unsafe {
                g.launch(
                    &net.f.argmax,
                    (1, 1, 1),
                    (1024, 1, 1),
                    0,
                    &[c.logits.arg(), Arg::I32(v as i32), c.best.arg()],
                )
                .unwrap();
            }
            let e2 = g.record().unwrap();
            g.sync().unwrap();
            t.push((
                e0.ms_to(&e1).unwrap() as f64 * 1e3,
                e1.ms_to(&e2).unwrap() as f64 * 1e3,
            ));
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let mut a: Vec<f64> = t.iter().map(|x| x.0).collect();
        let mut b: Vec<f64> = t.iter().map(|x| x.1).collect();
        eprintln!(
            "k {k} samp_topk: {:.1} us, argmax: {:.1} us",
            med(&mut a),
            med(&mut b)
        );
        // Host side: upload + read-back + CPU pick, no step.
        let mut host = Vec::new();
        for _ in 0..200 {
            let t0 = std::time::Instant::now();
            net.samp_upload(&mut c, &spec, &recent, 1, 1, None).unwrap();
            std::hint::black_box(net.samp_read(&mut c, k, v, 1).unwrap());
            host.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        eprintln!("k {k} upload + read (no kernels): {:.1} us", med(&mut host));
        // The greedy step's host side: params copy + 4-byte read.
        let mut gh = Vec::new();
        for _ in 0..200 {
            let t0 = std::time::Instant::now();
            let p = c.host_params.bytes_mut().unwrap();
            p[0..4].copy_from_slice(&7u32.to_le_bytes());
            c.host_params.copy_to(&c.params, 0, 8).unwrap();
            c.host_out.copy_from(&c.best, 0, 4).unwrap();
            std::hint::black_box(c.host_out.bytes_mut().unwrap()[0]);
            gh.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        eprintln!("greedy params copy + read: {:.1} us", med(&mut gh));
        // Phases of forward_cands vs forward_argmax.
        let (mut up, mut st, mut rd, mut ga) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..200 {
            let t0 = std::time::Instant::now();
            net.samp_upload(&mut c, &spec, &recent, 1, 1, Some((100, 3)))
                .unwrap();
            let t1 = std::time::Instant::now();
            net.step_with(&mut c, Input::Token(100), 3, Head::Sample)
                .unwrap();
            let t2 = std::time::Instant::now();
            std::hint::black_box(net.samp_read(&mut c, k, v, 1).unwrap());
            let t3 = std::time::Instant::now();
            std::hint::black_box(net.forward_argmax(100, 3, &mut c));
            let t4 = std::time::Instant::now();
            up.push((t1 - t0).as_secs_f64() * 1e6);
            st.push((t2 - t1).as_secs_f64() * 1e6);
            rd.push((t3 - t2).as_secs_f64() * 1e6);
            ga.push((t4 - t3).as_secs_f64() * 1e6);
        }
        eprintln!(
            "k {k} phases: upload {:.1}, launch {:.1}, wait+read {:.1} us; greedy step {:.1} us",
            med(&mut up),
            med(&mut st),
            med(&mut rd),
            med(&mut ga)
        );
    }
}

/// `real_model_sampling_cuda` with a q8 cache: the sampling step and
/// batched verify on q8 attention.
#[test]
fn real_model_sampling_cuda_q8() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let net = CudaQwen2::with_kv(&g, &km, &m, 256, CacheKv::Q8 { group: 0 }, false).unwrap();
    crate::sample::tests::real_model_sampling_on(&net, &m);
}

/// qwen3 (samp_topk at the end of its decode graph and in batched verify;
/// q_dim != dim): `real_model_sampling_cuda` with f16 and q8 caches.
#[test]
fn real_model_sampling_cuda_qwen3() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::qwen3_model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
        let net = CudaQwen2::with_kv(&g, &km, &m, 256, kv, false).unwrap();
        crate::sample::tests::real_model_sampling_on(&net, &m);
    }
}

/// llama (128K vocabulary, tied Q6_K head): `real_model_sampling_cuda`
/// with f16 and q8 caches.
#[test]
fn real_model_sampling_cuda_llama() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::llama_model() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
        let net = CudaQwen2::with_kv(&g, &km, &m, 256, kv, false).unwrap();
        crate::sample::tests::real_model_sampling_on(&net, &m);
    }
}
