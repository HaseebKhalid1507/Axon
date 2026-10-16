use super::*;
use crate::cuda::Kernels;
use crate::cuda::qwen2::CacheKv;
use crate::cuda::qwen2::tests::{Agree, PROMPTS, assert_graded, compare, kv_compare, report};
use std::path::PathBuf;

/// A model file under models/, mapped (not copied): several models are
/// built from one load, and a mapped file reads back what's been released.
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
/// Qwen3-0.6B: q/k norms, q_dim (2048) != dim (1024).
const Q3: &str = crate::testutil::QWEN3_MODEL;
const L1: &str = crate::testutil::LLAMA_MODEL;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Logits of `net` after each prompt and each of `steps` greedy tokens
/// (prefill through feed_batch in the backend's chunks, then forward).
fn trace<M: Lm>(net: &M, m: &Model, steps: usize) -> Vec<Vec<f32>> {
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let mut out = Vec::new();
    for p in PROMPTS {
        let ids = tok.encode(p);
        let mut c = net.new_cache(ids.len() + steps + 1);
        let (&last, rest) = ids.split_last().unwrap();
        let pb = net.prefill_batch();
        for (i, b) in rest.chunks(pb).enumerate() {
            net.feed_batch(b, i * pb, &mut c);
        }
        let mut t = last;
        for s in 0..=steps {
            let l = net.forward(t, rest.len() + s, &mut c);
            t = crate::argmax(&l);
            out.push(l);
        }
    }
    out
}

/// Every layer on the GPU is the GPU model, bit for bit; every layer on
/// the CPU is the CPU model (no GPU opened at all).
#[test]
fn hybrid_ends_are_the_backends_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(M05) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let want = trace(&CudaQwen2::new(&g, &km, &m, 128).unwrap(), &m, 16);
    let got = trace(
        &HybridQwen2::new(Some((&g, &km)), &m, 128, n, CacheKv::F16).unwrap(),
        &m,
        16,
    );
    assert!(
        got.iter().zip(&want).all(|(a, b)| bits(a) == bits(b)),
        "all on GPU"
    );
    // More than n is all of them too.
    let got = trace(
        &HybridQwen2::new(Some((&g, &km)), &m, 128, n + 5, CacheKv::F16).unwrap(),
        &m,
        2,
    );
    let want2 = trace(&CudaQwen2::new(&g, &km, &m, 128).unwrap(), &m, 2);
    assert!(
        got.iter().zip(&want2).all(|(a, b)| bits(a) == bits(b)),
        "n + 5"
    );

    let want = trace(&Qwen2::new(&m), &m, 16);
    let h = HybridQwen2::new(None, &m, 128, 0, CacheKv::F16).unwrap();
    assert!(h.gpu.is_none() && h.cpu.as_ref().unwrap().n_layers == n);
    let got = trace(&h, &m, 16);
    assert!(
        got.iter().zip(&want).all(|(a, b)| bits(a) == bits(b)),
        "all on CPU"
    );
    eprintln!(
        "0.5B: hybrid g = n / g = 0 == GPU / CPU bits, {} positions",
        got.len()
    );
}

/// Mixed splits (k CPU layers in {1, n/2, n-1}) vs the CPU model, graded
/// as the GPU model is (`assert_graded`: against the CPU's own spread
/// between its kernel sets).
fn mixed_graded(rel: &str, name: &str) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(rel) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let n = block_count(&m).unwrap();
    let rows = Qwen2::new_with(&m, false);
    let cpu = Qwen2::new(&m);
    let mut ac = Agree::default();
    for p in PROMPTS {
        compare(&cpu, &rows, &tok.encode(p), 32, &mut ac);
    }
    drop(rows);
    report(&format!("{name} cpu rows vs cpu"), &ac);
    for k in [1, n / 2, n - 1] {
        let h = HybridQwen2::new(Some((&g, &km)), &m, 256, n - k, CacheKv::F16).unwrap();
        let mut ag = Agree::default();
        for p in PROMPTS {
            compare(&cpu, &h, &tok.encode(p), 32, &mut ag);
        }
        report(&format!("{name} hybrid k={k} vs cpu"), &ag);
        assert_graded(&ag, &ac);
    }
}

#[test]
fn hybrid_logits_match_cpu_0_5b() {
    mixed_graded(M05, "0.5B");
}

#[test]
fn hybrid_logits_match_cpu_1_5b() {
    mixed_graded(M15, "1.5B");
}

/// ~n tokens of mixed text, for prefill tests.
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

/// Prefill through feed_batch in `sizes` (cycled) chunks, 0 meaning one
/// token by `forward` (the per-token path both halves take in decode),
/// then `forward` of the last: (CPU KV bits, GPU KV bits, logits).
fn prefill_with(h: &HybridQwen2, ids: &[u32], sizes: &[usize]) -> (Vec<u32>, Vec<u16>, Vec<f32>) {
    let mut c = h.new_cache(ids.len() + 1);
    let (&last, rest) = ids.split_last().unwrap();
    let mut pos = 0;
    for &s in sizes.iter().cycle() {
        if pos == rest.len() {
            break;
        }
        if s == 0 {
            h.forward(rest[pos], pos, &mut c);
            pos += 1;
            continue;
        }
        let e = (pos + s).min(rest.len());
        h.feed_batch(&rest[pos..e], pos, &mut c);
        pos = e;
    }
    let l = h.forward(last, rest.len(), &mut c);
    let cpu = c.cpu.as_ref().unwrap().bits();
    let gpu = CudaQwen2::kv_bits(c.gpu.as_ref().unwrap());
    (cpu, gpu, l)
}

/// The hybrid's prefill contract: whatever the batching (per token, 1, 7,
/// the CPU's 64, past the GPU's 512, ragged, mixed with decode steps), both
/// halves' KV caches and the next token's logits are bit-identical, with
/// the GPU half's cache of type `kv` (q8: its batched store and attention
/// are q8 decode's bits).
fn prefill_bits(rel: &str, name: &str, kv: CacheKv) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(rel) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let ids = long_ids(&m, 700);
    let h = HybridQwen2::new(Some((&g, &km)), &m, ids.len() + 1, n - n / 2, kv).unwrap();
    let gpu_max = h.gpu.as_ref().unwrap().prefill_batch();
    assert!(ids.len() > gpu_max, "covers the GPU's chunking");
    let (c0, g0, l0) = prefill_with(&h, &ids, &[0]);
    for sizes in [
        vec![1],
        vec![7],
        vec![h.prefill_batch()],
        vec![600],
        vec![5, 100, 3, 17, 260, 1, 33],
        vec![16, 0, 9, 0, 0, 200],
    ] {
        let (c, gk, l) = prefill_with(&h, &ids, &sizes);
        assert!(c == c0, "{name} {sizes:?}: CPU KV differs");
        assert!(gk == g0, "{name} {sizes:?}: GPU KV differs");
        assert_eq!(bits(&l), bits(&l0), "{name} {sizes:?}: logits");
    }
    eprintln!(
        "{name} k={} {kv:?}: prefill bit-identical over 6 batchings of {} tokens",
        n / 2,
        ids.len()
    );
}

#[test]
fn hybrid_prefill_batchings_bit_identical_0_5b() {
    prefill_bits(M05, "0.5B", CacheKv::F16);
}

#[test]
fn hybrid_prefill_batchings_bit_identical_1_5b() {
    prefill_bits(M15, "1.5B", CacheKv::F16);
}

#[test]
fn hybrid_prefill_batchings_bit_identical_q8_0_5b() {
    prefill_bits(M05, "0.5B", CacheKv::Q8 { group: 0 });
}

#[test]
fn hybrid_prefill_batchings_bit_identical_q8_1_5b() {
    prefill_bits(M15, "1.5B", CacheKv::Q8 { group: 0 });
}

/// qwen3 split: CPU layers by the CPU's qwen3, the GPU half from hidden
/// states, f16 and q8 GPU caches.
#[test]
fn hybrid_prefill_batchings_bit_identical_qwen3() {
    prefill_bits(Q3, "qwen3 0.6B", CacheKv::F16);
    prefill_bits(Q3, "qwen3 0.6B", CacheKv::Q8 { group: 0 });
}

#[test]
fn hybrid_logits_match_cpu_qwen3() {
    mixed_graded(Q3, "qwen3 0.6B");
}

/// llama split (q/k rows reordered at load, RoPE factors on both halves):
/// f16 and q8 GPU caches keep the batchings' bits; logits graded vs CPU.
#[test]
fn hybrid_prefill_batchings_bit_identical_llama() {
    prefill_bits(L1, "llama 1B", CacheKv::F16);
    prefill_bits(L1, "llama 1B", CacheKv::Q8 { group: 0 });
}

#[test]
fn hybrid_logits_match_cpu_llama() {
    mixed_graded(L1, "llama 1B");
}

/// verify_argmax's picks equal forward_argmax per token (and leave the
/// same KV in both halves), for right, wrong and random drafts, small and
/// past the GPU's max batch; with k = 1 and n - 1 too.
#[test]
fn hybrid_verify_matches_forward_argmax() {
    verify_bits(CacheKv::F16);
}

#[test]
fn hybrid_verify_matches_forward_argmax_q8() {
    verify_bits(CacheKv::Q8 { group: 0 });
}

fn verify_bits(kv: CacheKv) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(M05) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let prompt = long_ids(&m, 90);
    let (&last, rest) = prompt.split_last().unwrap();
    for k in [1, n / 2, n - 1] {
        let h = HybridQwen2::new(Some((&g, &km)), &m, 1400, n - k, kv).unwrap();
        let mut right = Vec::new();
        {
            let mut c = h.new_cache(prompt.len() + 20);
            h.feed_batch(rest, 0, &mut c);
            let mut t = last;
            for i in 0..16 {
                t = h.forward_argmax(t, rest.len() + i, &mut c);
                right.push(t);
            }
        }
        let mut seed = 77u64 + k as u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n) as u32
        };
        let mut cases: Vec<Vec<u32>> = Vec::new();
        for nb in [1, 2, 5, 8, 9, 16] {
            let mut d = vec![last];
            d.extend_from_slice(&right[..nb - 1]);
            cases.push(d.clone());
            if nb > 2 {
                d[nb / 2] = rnd(h.vocab() as u64);
                cases.push(d);
            }
            cases.push((0..nb).map(|_| rnd(h.vocab() as u64)).collect());
        }
        cases.push((0..600).map(|_| rnd(5000)).collect());
        let start = rest.len();
        for toks in &cases {
            let mut a = h.new_cache(start + toks.len() + 1);
            let mut b = h.new_cache(start + toks.len() + 1);
            h.feed_batch(rest, 0, &mut a);
            h.feed_batch(rest, 0, &mut b);
            // Stale slots: a different pass over the same positions first.
            let junk: Vec<u32> = toks.iter().map(|t| (t + 1) % 1000).collect();
            h.verify_argmax(&junk, start, &mut a);
            let got = h.verify_argmax(toks, start, &mut a);
            let want: Vec<u32> = (0..toks.len())
                .map(|i| h.forward_argmax(toks[i], start + i, &mut b))
                .collect();
            assert_eq!(got, want, "k={k}: verify of {} tokens", toks.len());
            assert!(
                a.cpu.as_ref().unwrap().bits() == b.cpu.as_ref().unwrap().bits(),
                "k={k}: CPU KV after verify of {}",
                toks.len()
            );
            assert!(
                CudaQwen2::kv_bits(a.gpu.as_ref().unwrap())
                    == CudaQwen2::kv_bits(b.gpu.as_ref().unwrap()),
                "k={k}: GPU KV after verify of {}",
                toks.len()
            );
        }
        eprintln!(
            "0.5B k={k} {kv:?}: {} verify passes == per-token picks",
            cases.len()
        );
    }
}

/// Sampled verify: verify_cands' candidates equal forward_cands' per token
/// (bit for bit: logits and ids) and leave the same KV, for a split's
/// batched path (CPU hidden_batch, GPU on-device candidates), with and
/// without the repeat penalty, k up to CAND_MAX and past it (the per-row
/// fallback), drafts small and past the GPU's max batch.
#[test]
fn hybrid_verify_cands_matches_forward_cands() {
    use crate::sample::{CAND_MAX, CandSpec};
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(M05) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let prompt = long_ids(&m, 90);
    let (&last, rest) = prompt.split_last().unwrap();
    let start = rest.len();
    let h = HybridQwen2::new(Some((&g, &km)), &m, 1400, n / 2, CacheKv::F16).unwrap();
    let mut seed = 99u64;
    let mut rnd = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n) as u32
    };
    let specs = [
        CandSpec {
            k: 40,
            penalty: 1.0,
            last_n: 64,
        },
        CandSpec {
            k: 40,
            penalty: 1.1,
            last_n: 64,
        },
        CandSpec {
            k: CAND_MAX,
            penalty: 1.1,
            last_n: 8,
        },
        CandSpec {
            k: CAND_MAX + 1,
            penalty: 1.1,
            last_n: 64,
        },
    ];
    for spec in &specs {
        for nb in [1, 2, 8, 9, 300] {
            let mut toks = vec![last];
            toks.extend((1..nb).map(|_| rnd(3000)));
            let recent: Vec<u32> = prompt[..start]
                .iter()
                .copied()
                .chain(toks.iter().copied())
                .collect();
            let mut a = h.new_cache(start + nb + 1);
            let mut b = h.new_cache(start + nb + 1);
            h.feed_batch(rest, 0, &mut a);
            h.feed_batch(rest, 0, &mut b);
            let got = h.verify_cands(&toks, start, &mut a, spec, &recent);
            let n = recent.len();
            let want: Vec<_> = (0..nb)
                .map(|i| {
                    h.forward_cands(toks[i], start + i, &mut b, spec, &recent[..n - nb + i + 1])
                })
                .collect();
            assert_eq!(got.len(), nb);
            for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                let bits = |v: &Vec<crate::sample::Cand>| -> Vec<(u32, u32)> {
                    v.iter().map(|c| (c.id, c.logit.to_bits())).collect()
                };
                assert_eq!(bits(x), bits(y), "{spec:?} nb {nb}: row {i}");
            }
            assert!(
                a.cpu.as_ref().unwrap().bits() == b.cpu.as_ref().unwrap().bits(),
                "{spec:?} nb {nb}: CPU KV"
            );
            assert!(
                CudaQwen2::kv_bits(a.gpu.as_ref().unwrap())
                    == CudaQwen2::kv_bits(b.gpu.as_ref().unwrap()),
                "{spec:?} nb {nb}: GPU KV"
            );
        }
    }
}

/// Whether `m` refuses to hand out `t`'s bytes (released on a copy).
fn released(m: &Model, t: &crate::gguf::Tensor) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        m.tensor_data(t);
    }))
    .is_err()
}

/// Each half loads only its own weights: on a copy of the file (where
/// released bytes can't be read back), the CPU half of a split leaves the
/// GPU's layers and head untouched, and once the hybrid is built every
/// tensor has been handed back but the embedding the CPU reads.
#[test]
fn hybrid_halves_own_their_weights() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::model_with(Some(true)) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let k = n / 2;
    let layer = |name: &str| -> Option<usize> {
        name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
    };
    {
        let cpu = Qwen2::new_layers(&m, true, k);
        assert_eq!(
            cpu.new_cache(4).bits().len(),
            2 * k * 4 * cpu.n_kv_heads * cpu.head_dim
        );
        for t in &m.tensors {
            let gpu_side = layer(&t.name).is_none_or(|l| l >= k) && t.name != "token_embd.weight";
            if gpu_side {
                assert!(!released(&m, t), "{} released by the CPU half", t.name);
            }
        }
    }
    let m = crate::testutil::model_with(Some(true)).unwrap();
    let h = HybridQwen2::new(Some((&g, &km)), &m, 64, n - k, CacheKv::F16).unwrap();
    for t in &m.tensors {
        let kept = t.name == "token_embd.weight";
        assert_eq!(released(&m, t), !kept, "{}", t.name);
    }
    // The GPU half's batch scratch is the split's, not a whole model's.
    assert_eq!(
        h.gpu.as_ref().unwrap().prefill_batch(),
        crate::cuda::qwen2::SPLIT_BATCH
    );
    // And it runs (no released bytes read).
    let mut c = h.new_cache(8);
    h.feed_batch(&[1, 2, 3], 0, &mut c);
    assert!(h.forward_argmax(4, 3, &mut c) < h.vocab() as u32);
}

/// The automatic split: the most GPU layers whose plan plus the desktop
/// margin fit the free VRAM, one more doesn't; the plan shrinks as layers
/// move to the CPU.
#[test]
fn hybrid_auto_split_is_the_most_that_fit() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(M15) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let need = |first: usize, ctx: usize| {
        CudaQwen2::vram_need(
            &g,
            &km,
            &m,
            ctx,
            first,
            CacheKv::F16,
            crate::cuda::qwen2::SPLIT_BATCH,
        )
        .unwrap()
    };
    for f in 1..n {
        assert!(need(f, 4096) < need(f - 1, 4096), "plan at first {f}");
    }
    let free = g.info().unwrap().free_bytes;
    // A context too big for the whole model on any GPU this runs on: all
    // layers' KV would take 1.5x the free VRAM, so about half fit.
    let per_tok = (need(0, 2048) - need(0, 1024)) / 1024;
    let ctx = (free * 3 / 2 / per_tok) as usize;
    let auto = auto_gpu_layers(&g, &km, &m, ctx, CacheKv::F16, free, false).unwrap();
    eprintln!(
        "1.5B ctx {ctx}: auto split {auto}/{n} of {:.2} GiB free",
        free as f64 / GIB
    );
    let fits = |gl: usize| need(n - gl, ctx) + DESKTOP_MARGIN <= free;
    assert!(auto == 0 || fits(auto), "chosen split doesn't fit");
    assert!(
        auto == n || !fits(auto + 1),
        "one more layer would have fit"
    );
    assert!(auto > 0 && auto < n, "a split, not all or nothing");
    // Streaming (PR A): the GPU half's bigger batch scratch and the stream
    // reserve, the same search; the plan stays monotone.
    let sb = crate::cuda::qwen2::split_batch(true);
    let need_s =
        |first: usize| CudaQwen2::vram_need(&g, &km, &m, ctx, first, CacheKv::F16, sb).unwrap();
    for f in 1..n {
        assert!(need_s(f) < need_s(f - 1), "streaming plan at first {f}");
    }
    let reserve = stream_reserve(&m, ctx).unwrap();
    let auto_s = auto_gpu_layers(&g, &km, &m, ctx, CacheKv::F16, free, true).unwrap();
    let fits_s = |gl: usize| need_s(n - gl) + reserve + DESKTOP_MARGIN <= free;
    assert!(auto_s == 0 || fits_s(auto_s), "streaming split doesn't fit");
    assert!(
        auto_s == n || !fits_s(auto_s + 1),
        "streaming: one more would fit"
    );
    assert!(auto_s <= auto, "streaming takes more VRAM, not less");
    eprintln!("1.5B ctx {ctx}: streaming auto split {auto_s}/{n}");
    // An explicit count that doesn't fit (all: 1.5x free in KV alone) says
    // which and what to do.
    let e = HybridQwen2::new(Some((&g, &km)), &m, ctx, n, CacheKv::F16)
        .err()
        .expect("all layers can't fit");
    assert!(
        e.contains(&format!("{n}/{n} layers")) && e.contains("or fewer --gpu-layers"),
        "{e}"
    );
}

/// A split's GPU half with a q8 cache: all layers on the GPU is the q8 GPU
/// model, bit for bit.
#[test]
fn hybrid_q8_all_on_gpu_is_the_q8_model() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(M05) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let q8 = CacheKv::Q8 { group: 0 };
    let want = trace(
        &CudaQwen2::with_kv(&g, &km, &m, 128, q8, false).unwrap(),
        &m,
        16,
    );
    let h = HybridQwen2::new(Some((&g, &km)), &m, 128, n, q8).unwrap();
    assert_eq!(h.kv_name(), "q8");
    let got = trace(&h, &m, 16);
    assert!(
        got.iter().zip(&want).all(|(a, b)| bits(a) == bits(b)),
        "all on GPU, q8"
    );
}

/// Mixed splits with a q8 GPU half vs the same split with f16, graded with
/// the q8 cache's own bounds (`cuda::qwen2::tests::q8_accuracy`: text mean
/// KL <= 1.5e-2 and top-1 >= 90%, prompts mean KL <= 3e-2, any position
/// KL < 4): PROMPTS with 32 greedy steps, natural text 0..300 and
/// 1024..1152 teacher-forced. (Against the CPU, 0.5B's q8 is 2-5x the CPU's
/// own noise, so `assert_graded` would measure q8, not the split.)
fn q8_split_graded(rel: &str, name: &str) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(rel) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    for k in [1, n / 2] {
        let f16 = HybridQwen2::new(Some((&g, &km)), &m, 1200, n - k, CacheKv::F16).unwrap();
        let q8 =
            HybridQwen2::new(Some((&g, &km)), &m, 1200, n - k, CacheKv::Q8 { group: 0 }).unwrap();
        let a = kv_compare(&f16, &q8, &m, 300, 1024, 128);
        for (w, what) in a.iter().zip(["prompts", "text 0..300", "text 1024+128"]) {
            report(&format!("{name} k={k} hybrid q8 vs f16 {what}"), w);
            assert!(w.max_kl < 4.0, "{name} k={k} {what}: max KL {}", w.max_kl);
        }
        assert!(
            a[0].sum_kl <= 3e-2 * a[0].positions as f64,
            "{name} k={k} prompts: mean KL"
        );
        for w in &a[1..] {
            assert!(
                w.sum_kl <= 1.5e-2 * w.positions as f64,
                "{name} k={k} text: mean KL"
            );
            assert!(
                w.top1 * 100 >= 90 * w.positions,
                "{name} k={k} text: top-1 {}",
                w.top1
            );
        }
    }
}

#[test]
fn hybrid_q8_close_to_f16_0_5b() {
    q8_split_graded(M05, "0.5B");
}

#[test]
fn hybrid_q8_close_to_f16_1_5b() {
    q8_split_graded(M15, "1.5B");
}

/// Placement counts the q8 cache's bytes: every split's plan is smaller
/// than f16's, and at a context where the KV dominates, the automatic
/// split puts more layers on the GPU (the point of q8 on a split).
#[test]
fn hybrid_auto_split_q8_fits_more_layers() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(M15) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let q8 = CacheKv::Q8 { group: 0 };
    let need = |first: usize, ctx: usize, kv: CacheKv| {
        CudaQwen2::vram_need(&g, &km, &m, ctx, first, kv, crate::cuda::qwen2::SPLIT_BATCH).unwrap()
    };
    for f in 0..n {
        assert!(
            need(f, 4096, q8) < need(f, 4096, CacheKv::F16),
            "plan at first {f}"
        );
    }
    let free = g.info().unwrap().free_bytes;
    let per_tok = (need(0, 2048, CacheKv::F16) - need(0, 1024, CacheKv::F16)) / 1024;
    let ctx = (free * 3 / 2 / per_tok) as usize;
    let a16 = auto_gpu_layers(&g, &km, &m, ctx, CacheKv::F16, free, false).unwrap();
    let a8 = auto_gpu_layers(&g, &km, &m, ctx, q8, free, false).unwrap();
    eprintln!("1.5B ctx {ctx}: auto split f16 {a16}/{n}, q8 {a8}/{n}");
    let fits = |gl: usize| need(n - gl, ctx, q8) + DESKTOP_MARGIN <= free;
    assert!(a8 == 0 || fits(a8), "chosen q8 split doesn't fit");
    assert!(a8 == n || !fits(a8 + 1), "one more q8 layer would have fit");
    assert!(a8 > a16, "q8 fits more layers ({a8} vs {a16})");
}

// ---------------------------------------------------------------------------
// Streaming (PR A): prefill batches run the CPU's layers on the GPU.

/// The split with streaming on for batches of at least `min` tokens.
fn streamed<'m, 'g>(
    g: &'g Gpu,
    km: &'g Module<'g>,
    m: &'m Model,
    ctx: usize,
    gpu_layers: usize,
    min: usize,
) -> HybridQwen2<'m, 'g> {
    streamed_kv(g, km, m, ctx, gpu_layers, min, CacheKv::F16)
}

/// `streamed` with the GPU half's cache of type `kv`.
fn streamed_kv<'m, 'g>(
    g: &'g Gpu,
    km: &'g Module<'g>,
    m: &'m Model,
    ctx: usize,
    gpu_layers: usize,
    min: usize,
    kv: CacheKv,
) -> HybridQwen2<'m, 'g> {
    let mut h =
        HybridQwen2::new_with_stream(Some((g, km)), m, ctx, gpu_layers, kv, true, false).unwrap();
    h.stream.as_mut().expect("streaming set up").min = min;
    h
}

/// Every layer's KV rows as f32 bits, CPU half then GPU half (the CPU
/// cache's layout: per layer K then V, all positions).
fn kv_all(c: &HybridCache) -> Vec<u32> {
    let mut v = c.cpu.as_ref().unwrap().bits();
    let gk = CudaQwen2::kv_bits(c.gpu.as_ref().unwrap());
    v.extend(gk.iter().map(|&h| crate::quant::f16_to_f32(h).to_bits()));
    v
}

/// Streamed prefill computes every layer with the GPU's kernels from the
/// same embeddings, so whatever the batching (each batch streamed, history
/// rows read back from the CPU cache), both halves' KV caches hold the
/// all-GPU model's bits: the CPU layers' rows written back into the CPU
/// cache included. Batches under `min` don't stream: PR B's bits.
fn stream_prefill_bits(rel: &str, name: &str) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(rel) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let ids = long_ids(&m, 700);
    let ctx = ids.len() + 1;
    let rest = &ids[..ids.len() - 1];
    let want = {
        let gpu = CudaQwen2::new(&g, &km, &m, ctx).unwrap();
        let mut c = gpu.new_cache(ctx);
        gpu.feed_batch(rest, 0, &mut c);
        let k = CudaQwen2::kv_bits(&c);
        k.iter()
            .map(|&h| crate::quant::f16_to_f32(h).to_bits())
            .collect::<Vec<u32>>()
    };
    for k in [1, n / 2, n - 1] {
        let h = streamed(&g, &km, &m, ctx, n - k, 1);
        assert_eq!(
            h.gpu.as_ref().unwrap().prefill_batch(),
            crate::cuda::qwen2::split_batch(true)
        );
        for sizes in [vec![rest.len()], vec![5, 100, 3, 17, 260, 1, 33]] {
            let mut c = h.new_cache(ctx);
            assert!(c.skv.is_some(), "{name} k={k}: this cache streams");
            let before = h.streamed_chunks();
            let mut pos = 0;
            for &s in sizes.iter().cycle() {
                if pos == rest.len() {
                    break;
                }
                let e = (pos + s).min(rest.len());
                h.feed_batch(&rest[pos..e], pos, &mut c);
                pos = e;
            }
            assert!(
                h.streamed_chunks() >= before + sizes.len().min(7),
                "streamed"
            );
            let got = kv_all(&c);
            assert_eq!(got.len(), want.len());
            if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
                let gp = h.gpu.as_ref().unwrap();
                let per_layer = 2 * ctx * gp.n_kv_heads * gp.head_dim;
                panic!(
                    "{name} k={k} {sizes:?}: KV differs from the GPU model's at {i} (layer ~{})",
                    i / per_layer.max(1)
                );
            }
        }
        // Never streaming (min past every batch): PR B's bits.
        let off = streamed(&g, &km, &m, ctx, n - k, usize::MAX);
        let plain = HybridQwen2::new(Some((&g, &km)), &m, ctx, n - k, CacheKv::F16).unwrap();
        let sizes = [64, 300, 7];
        let a = prefill_with(&off, &ids, &sizes);
        let b = prefill_with(&plain, &ids, &sizes);
        assert!(a.0 == b.0 && a.1 == b.1, "{name} k={k}: below min != PR B");
        assert_eq!(bits(&a.2), bits(&b.2));
    }
    // CPU layers not repacked: their GGUF bytes are copied as they are
    // (from pageable memory), no un-permute.
    let mut h = HybridQwen2::build(
        Some((&g, &km)),
        &m,
        ctx,
        n - n / 2,
        CacheKv::F16,
        true,
        false,
        0,
    )
    .unwrap();
    h.stream.as_mut().expect("streaming set up").min = 1;
    let mut c = h.new_cache(ctx);
    assert!(c.skv.is_some(), "{name}: not repacked: this cache streams");
    h.feed_batch(&rest[..300], 0, &mut c);
    assert_eq!(h.streamed_chunks(), 1);
    h.feed_batch(&rest[300..], 300, &mut c);
    assert!(
        kv_all(&c) == want,
        "{name}: not repacked: KV differs from the GPU model's"
    );
    eprintln!(
        "{name}: streamed prefill KV == all-GPU bits (k = 1, n/2, n-1; 2 batchings of {} tokens; \
         and not repacked)",
        rest.len()
    );
}

#[test]
fn stream_prefill_is_gpu_bits_0_5b() {
    stream_prefill_bits(M05, "0.5B");
}

#[test]
fn stream_prefill_is_gpu_bits_1_5b() {
    stream_prefill_bits(M15, "1.5B");
}

/// qwen3 streamed prefill: the streamed layers' q/k norms ride with their
/// vectors (`streamer`), so the CPU half's KV is the all-GPU model's bits.
#[test]
fn stream_prefill_is_gpu_bits_qwen3() {
    stream_prefill_bits(Q3, "qwen3 0.6B");
    stream_q8_bits(Q3, "qwen3 0.6B");
}

#[test]
fn stream_logits_match_cpu_qwen3() {
    stream_graded(Q3, "qwen3 0.6B");
}

/// llama streamed prefill (the streamed layers' reordered q/k rows come
/// from the CPU's copy): GPU bits, f16 and q8; logits graded vs CPU.
#[test]
fn stream_prefill_is_gpu_bits_llama() {
    stream_prefill_bits(L1, "llama 1B");
    stream_q8_bits(L1, "llama 1B");
}

#[test]
fn stream_logits_match_cpu_llama() {
    stream_graded(L1, "llama 1B");
}

/// Streamed prefill, then decode with the CPU computing its layers: graded
/// vs the CPU model like the split without streaming (`mixed_graded`).
fn stream_graded(rel: &str, name: &str) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(rel) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let n = block_count(&m).unwrap();
    let rows = Qwen2::new_with(&m, false);
    let cpu = Qwen2::new(&m);
    let mut ac = Agree::default();
    for p in PROMPTS {
        compare(&cpu, &rows, &tok.encode(p), 32, &mut ac);
    }
    drop(rows);
    report(&format!("{name} cpu rows vs cpu"), &ac);
    for k in [1, n / 2, n - 1] {
        // min 2: every prompt here streams.
        let h = streamed(&g, &km, &m, 256, n - k, 2);
        let mut ag = Agree::default();
        for p in PROMPTS {
            compare(&cpu, &h, &tok.encode(p), 32, &mut ag);
        }
        report(&format!("{name} streamed k={k} vs cpu"), &ag);
        // Every prompt's prefill streamed (each cache had its slots).
        assert_eq!(h.streamed_chunks(), PROMPTS.len(), "{name} k={k}: streamed");
        assert_graded(&ag, &ac);
    }
}

#[test]
fn stream_logits_match_cpu_0_5b() {
    stream_graded(M05, "0.5B");
}

#[test]
fn stream_logits_match_cpu_1_5b() {
    stream_graded(M15, "1.5B");
}

/// verify_argmax never streams: after a streamed prefill its picks equal
/// forward_argmax per token and leave the same KV, also for batches far
/// past the streaming threshold.
#[test]
fn stream_verify_matches_forward_argmax() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(M05) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let prompt = long_ids(&m, 90);
    let rest = &prompt[..prompt.len() - 1];
    let start = rest.len();
    for k in [1, n / 2] {
        let h = streamed(&g, &km, &m, 1400, n - k, STREAM_MIN);
        let mut seed = 99u64 + k as u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n) as u32
        };
        let mut cases: Vec<Vec<u32>> = [1, 5, 16, 40]
            .iter()
            .map(|&nb| (0..nb).map(|_| rnd(h.vocab() as u64)).collect())
            .collect();
        cases.push((0..600).map(|_| rnd(5000)).collect());
        for toks in &cases {
            let mut a = h.new_cache(start + toks.len() + 1);
            let mut b = h.new_cache(start + toks.len() + 1);
            assert!(a.skv.is_some() && b.skv.is_some(), "k={k}: caches stream");
            let before = h.streamed_chunks();
            h.feed_batch(rest, 0, &mut a);
            h.feed_batch(rest, 0, &mut b);
            assert_eq!(h.streamed_chunks(), before + 2, "k={k}: prefills streamed");
            let got = h.verify_argmax(toks, start, &mut a);
            let want: Vec<u32> = (0..toks.len())
                .map(|i| h.forward_argmax(toks[i], start + i, &mut b))
                .collect();
            assert_eq!(got, want, "k={k}: verify of {} tokens", toks.len());
            assert!(kv_all(&a) == kv_all(&b), "k={k}: KV after verify");
        }
        eprintln!(
            "0.5B streamed k={k}: {} verify passes == per-token picks",
            cases.len()
        );
    }
}

/// Streaming with a q8 GPU half (--kv q8 on a split): the streamed (CPU)
/// layers keep the CPU cache's f16 rows and take the f16 attention
/// kernels, the GPU's own layers their q8 cache and kernels (`layer`
/// follows each layer's format). So the CPU half's rows are the all-GPU
/// f16 model's (its layers 0..k see the same inputs), the GPU half's q8
/// cache and the next logits are the same whatever the batching (each
/// batch streamed), and verify (never streamed) == per-token picks.
fn stream_q8_bits(rel: &str, name: &str) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(rel) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let ids = long_ids(&m, 700);
    let ctx = ids.len() + 8;
    let (&last, rest) = ids.split_last().unwrap();
    let f16_rows = {
        let gpu = CudaQwen2::new(&g, &km, &m, ctx).unwrap();
        let mut c = gpu.new_cache(ctx);
        gpu.feed_batch(rest, 0, &mut c);
        CudaQwen2::kv_bits(&c)
    };
    let q8 = CacheKv::Q8 { group: 0 };
    for k in [1, n / 2] {
        let h = streamed_kv(&g, &km, &m, ctx, n - k, 1, q8);
        assert_eq!(h.kv_name(), "q8");
        let run = |sizes: &[usize]| {
            let mut c = h.new_cache(ctx);
            assert!(c.skv.is_some(), "{name} k={k}: this cache streams");
            let before = h.streamed_chunks();
            let mut pos = 0;
            for &s in sizes.iter().cycle() {
                if pos == rest.len() {
                    break;
                }
                let e = (pos + s).min(rest.len());
                h.feed_batch(&rest[pos..e], pos, &mut c);
                pos = e;
            }
            assert!(h.streamed_chunks() > before, "{name} k={k}: streamed");
            let l = h.forward(last, rest.len(), &mut c);
            let cpu = c.cpu.as_ref().unwrap().bits();
            let gpu = CudaQwen2::kv_bits(c.gpu.as_ref().unwrap());
            (cpu, gpu, l, c)
        };
        let (c0, g0, l0, mut cache) = run(&[rest.len()]);
        // The CPU half's rows: the f16 GPU model's, layers 0..k, positions
        // 0..rest.len() (the CPU cache holds f16 bits widened to f32; the
        // last position came from the CPU's own decode step).
        let per = ctx * h.gpu.as_ref().unwrap().n_kv_heads * h.gpu.as_ref().unwrap().head_dim;
        for kv in 0..2 * k {
            for i in 0..rest.len() * per / ctx {
                let want = crate::quant::f16_to_f32(f16_rows[kv * per + i]).to_bits();
                assert_eq!(
                    c0[kv * per + i],
                    want,
                    "{name} k={k}: CPU half row bits (layer {}, {})",
                    kv / 2,
                    if kv % 2 == 0 { "K" } else { "V" }
                );
            }
        }
        for sizes in [vec![5, 100, 3, 17, 260, 1, 33], vec![64, 300]] {
            let (c, gk, l, _) = run(&sizes);
            assert!(c == c0, "{name} k={k} {sizes:?}: CPU KV differs");
            assert!(gk == g0, "{name} k={k} {sizes:?}: GPU q8 KV differs");
            assert_eq!(bits(&l), bits(&l0), "{name} k={k} {sizes:?}: logits");
        }
        // Verify never streams, on top of a streamed q8 prefill.
        let toks: Vec<u32> = (0..6).map(|i| (i * 977 + 13) % 5000).collect();
        let mut other = run(&[rest.len()]).3;
        let start = rest.len() + 1;
        let got = h.verify_argmax(&toks, start, &mut cache);
        let want: Vec<u32> = (0..toks.len())
            .map(|i| h.forward_argmax(toks[i], start + i, &mut other))
            .collect();
        assert_eq!(
            got, want,
            "{name} k={k}: verify after a streamed q8 prefill"
        );
        eprintln!("{name} k={k}: streamed q8 prefill bit contracts hold");
    }
}

#[test]
fn stream_q8_prefill_bits_0_5b() {
    stream_q8_bits(M05, "0.5B");
}

#[test]
fn stream_q8_prefill_bits_1_5b() {
    stream_q8_bits(M15, "1.5B");
}

/// A 32B-shaped model's VRAM needs without a GPU: 64 layers of 280 MiB,
/// a 600 MiB head, 256 MiB headroom; the KV per token and layer is 4 KiB
/// in f16 (8 kv heads x 128, keys and values), about half in q8.
struct Toy {
    q8: bool,
}

const MIB: u64 = 1 << 20;
const TOY_N: usize = 64;

impl Toy {
    fn kv_tok_layer(&self) -> u64 {
        if self.q8 {
            2 * 8 * 128 + 2 * 8 * 4
        } else {
            2 * 8 * 128 * 2
        }
    }
    /// g of the layers (and the head) on the GPU, a ctx-token cache.
    fn need(&self, g: usize, ctx: usize) -> u64 {
        256 * MIB + 600 * MIB + g as u64 * (280 * MIB + self.kv_tok_layer() * ctx as u64)
    }
    fn place(&self, free: u64, max_ctx: usize, fit: bool) -> Placement {
        choose_placement(
            free,
            max_ctx,
            fit,
            |c| self.need(TOY_N, c),
            || most_gpu_layers(TOY_N, free, |g| Ok(self.need(g, max_ctx))),
        )
        .unwrap()
    }
}

const F16: Toy = Toy { q8: false };
const Q8: Toy = Toy { q8: true };

/// A split is the most layers that fit at max_ctx with the margin.
fn assert_most(t: &Toy, p: Placement, free: u64, max_ctx: usize) -> usize {
    let Placement::Split { gpu_layers: g, .. } = p else {
        panic!("{p:?}: not a split");
    };
    assert!(g > 0 && g < TOY_N, "{p:?}");
    assert!(
        t.need(g, max_ctx) + DESKTOP_MARGIN <= free,
        "{p:?} doesn't fit"
    );
    assert!(
        t.need(g + 1, max_ctx) + DESKTOP_MARGIN > free,
        "{p:?}: one more fits"
    );
    g
}

#[test]
fn placement_weights_dont_fit_splits() {
    let free = 12 << 30;
    for t in [&F16, &Q8] {
        for fit in [false, true] {
            let p = t.place(free, 8192, fit);
            assert!(matches!(p, Placement::Split { gpu_ctx: None, .. }), "{p:?}");
            assert_most(t, p, free, 8192);
        }
    }
    // Nothing fits at all.
    assert_eq!(F16.place(MIB, 8192, true), Placement::Nowhere);
}

#[test]
fn placement_full_ctx_fits_whole_gpu() {
    let free = F16.need(TOY_N, 8192);
    for fit in [false, true] {
        assert_eq!(F16.place(free, 8192, fit), Placement::Gpu { ctx: 8192 });
        assert_eq!(Q8.place(free, 8192, fit), Placement::Gpu { ctx: 8192 });
    }
}

/// Room for the whole model with only a short context: the split at the
/// full context wins, and says what the all-GPU context would have been.
#[test]
fn placement_tiny_ctx_splits_at_full_ctx() {
    for ctx in [256, 1024, 2048, 4095] {
        let free = F16.need(TOY_N, ctx);
        let p = F16.place(free, 8192, true);
        let want = ctx / 256 * 256;
        assert!(
            matches!(p, Placement::Split { gpu_ctx: Some(c), .. } if c == want),
            "room for {ctx}: {p:?}"
        );
        assert_most(&F16, p, free, 8192);
        // Without fit (an explicit --ctx) the same split, no note.
        let q = F16.place(free, 8192, false);
        assert!(matches!(q, Placement::Split { gpu_ctx: None, .. }), "{q:?}");
    }
}

/// Room for at least half the context (or 4096): the whole model on the
/// GPU with the shorter context.
#[test]
fn placement_long_enough_ctx_stays_on_gpu() {
    assert_eq!(keep_ctx(8192), 4096);
    assert_eq!(keep_ctx(4096), 2048);
    assert_eq!(keep_ctx(32768), 4096);
    for (max_ctx, ctx) in [
        (8192, 4096),
        (8192, 6000),
        (8192, 8191),
        (4096, 2048),
        (32768, 4096),
    ] {
        let free = F16.need(TOY_N, ctx);
        assert_eq!(
            F16.place(free, max_ctx, true),
            Placement::Gpu {
                ctx: ctx / 256 * 256
            },
            "room for {ctx} of {max_ctx}"
        );
        // An explicit --ctx max_ctx: the split.
        assert!(matches!(
            F16.place(free, max_ctx, false),
            Placement::Split { gpu_ctx: None, .. }
        ));
    }
    // Just under the threshold of a 4096 model: split.
    let free = F16.need(TOY_N, 2047);
    assert!(matches!(
        F16.place(free, 4096, true),
        Placement::Split {
            gpu_ctx: Some(1792),
            ..
        }
    ));
}

/// q8's smaller cache changes the decision where it should: the same VRAM
/// that gives f16 a short context (a split) gives q8 twice the tokens
/// (all on the GPU), and q8 fits the full context where f16 must shrink.
#[test]
fn placement_q8_changes_the_decision() {
    let free = F16.need(TOY_N, 2560);
    assert!(matches!(
        F16.place(free, 8192, true),
        Placement::Split {
            gpu_ctx: Some(2560),
            ..
        }
    ));
    let p = Q8.place(free, 8192, true);
    assert!(matches!(p, Placement::Gpu { ctx } if ctx >= 4096), "{p:?}");

    let free = Q8.need(TOY_N, 8192);
    assert_eq!(Q8.place(free, 8192, true), Placement::Gpu { ctx: 8192 });
    let p = F16.place(free, 8192, true);
    assert!(
        matches!(p, Placement::Gpu { ctx } if (4096..8192).contains(&ctx)),
        "{p:?}"
    );

    // Weights alone don't fit: both split, q8 with more GPU layers.
    let free = 16 << 30;
    let g16 = assert_most(&F16, F16.place(free, 8192, true), free, 8192);
    let g8 = assert_most(&Q8, Q8.place(free, 8192, true), free, 8192);
    assert!(g8 > g16, "q8 {g8} vs f16 {g16}");
}

/// --gpu-layers n (or more) is the plain GPU model, not a split with no
/// CPU half: same bits.
#[test]
fn all_gpu_layers_is_the_cuda_backend() {
    use crate::lm::{Backend, Device};
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(M05) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let want = trace(&CudaQwen2::new(&g, &km, &m, 128).unwrap(), &m, 8);
    for gl in [n, n + 3] {
        let b = Backend::new(
            &m,
            Device::Cuda,
            128,
            Some(gl),
            crate::kv::KvType::F16,
            false,
        )
        .unwrap();
        assert!(matches!(b, Backend::Cuda(_)), "--gpu-layers {gl}");
        assert_eq!(b.ctx(128), 128);
        let got = trace(&b, &m, 8);
        assert!(
            want.iter().zip(&got).all(|(a, b)| bits(a) == bits(b)),
            "--gpu-layers {gl}: not the GPU model's bits"
        );
    }
}

/// The toy's batch scratch per token of a split's GPU half (32B-like:
/// ~160 MiB at 512 tokens).
const TOY_SCRATCH_TOK: u64 = 320 << 10;

impl Toy {
    /// `need` of a split's GPU half with batch scratch for `batch` tokens.
    fn split_need(&self, g: usize, ctx: usize, batch: usize) -> u64 {
        self.need(g, ctx) + batch as u64 * TOY_SCRATCH_TOK
    }

    /// `place` with the split option sized as `auto_gpu_layers` sizes it
    /// (`split_gpu_layers`): streaming or not, `reserve` for the stream.
    fn place_stream(
        &self,
        free: u64,
        max_ctx: usize,
        fit: bool,
        stream: bool,
        reserve: u64,
    ) -> Placement {
        choose_placement(
            free,
            max_ctx,
            fit,
            |c| self.need(TOY_N, c),
            || {
                split_gpu_layers(TOY_N, free, stream, reserve, |g, b| {
                    Ok(self.split_need(g, max_ctx, b))
                })
            },
        )
        .unwrap()
    }
}

/// The split option with streaming on is sized for the streamed pass
/// (`STREAM_PASS` scratch) plus the stream reserve; off, `SPLIT_BATCH`
/// scratch and nothing else. Each is the most layers that fit with that
/// need, and streaming gets fewer (never more) layers.
#[test]
fn placement_split_counts_streaming() {
    use crate::cuda::qwen2::{SPLIT_BATCH, STREAM_PASS, split_batch};
    assert_eq!(split_batch(false), SPLIT_BATCH);
    assert_eq!(split_batch(true), STREAM_PASS);
    let reserve = 900 * MIB;
    let free = 12 << 30;
    for t in [&F16, &Q8] {
        // The batch the sizing asks for.
        let mut seen = Vec::new();
        split_gpu_layers(TOY_N, free, true, reserve, |g, b| {
            seen.push(b);
            Ok(t.split_need(g, 8192, b))
        })
        .unwrap();
        assert!(seen.iter().all(|&b| b == STREAM_PASS), "{seen:?}");
        let mut layers = [0; 2];
        for (i, stream) in [false, true].into_iter().enumerate() {
            let p = t.place_stream(free, 8192, true, stream, reserve);
            let Placement::Split { gpu_layers: g, .. } = p else {
                panic!("{p:?}: not a split");
            };
            let (batch, extra) = if stream {
                (STREAM_PASS, reserve)
            } else {
                (SPLIT_BATCH, 0)
            };
            let need = |g: usize| t.split_need(g, 8192, batch) + extra + DESKTOP_MARGIN;
            assert!(need(g) <= free, "stream {stream}: {g} doesn't fit");
            assert!(need(g + 1) > free, "stream {stream}: one more fits");
            layers[i] = g;
        }
        assert!(layers[1] < layers[0], "streaming {layers:?}");
        // A reserve of 0 still sizes the streamed pass's scratch.
        let g0 =
            split_gpu_layers(TOY_N, free, true, 0, |g, b| Ok(t.split_need(g, 8192, b))).unwrap();
        assert!(g0 <= layers[0] && g0 >= layers[1], "{g0} vs {layers:?}");
        // Off ignores the reserve.
        let off = split_gpu_layers(TOY_N, free, false, 1 << 40, |g, b| {
            Ok(t.split_need(g, 8192, b))
        })
        .unwrap();
        assert_eq!(off, layers[0]);
    }
}

/// Streaming can change the decision: VRAM for a split with a few GPU
/// layers when not streaming has none left once the streamed pass and the
/// reserve are counted, so the placement is Nowhere (the whole model never
/// fits here), and with room for a short all-GPU context it falls back to
/// that context rather than a split it can't hold.
#[test]
fn placement_stream_changes_the_decision() {
    let reserve = 900 * MIB;
    // One GPU layer without streaming, none with it.
    let free = F16.split_need(1, 8192, crate::cuda::qwen2::SPLIT_BATCH) + DESKTOP_MARGIN;
    assert!(matches!(
        F16.place_stream(free, 8192, true, false, reserve),
        Placement::Split { gpu_layers: 1, .. }
    ));
    assert_eq!(
        F16.place_stream(free, 8192, true, true, reserve),
        Placement::Nowhere
    );
    // A toy whose whole model fits at 1024 tokens (below keep_ctx) and
    // whose stream reserve leaves no split layer: the shrunk all-GPU
    // context; without streaming, the split at the full context.
    let free = F16.need(TOY_N, 1024);
    let big = free; // a reserve no split can hold beside its layers
    let p = F16.place_stream(free, 8192, true, true, big);
    assert_eq!(p, Placement::Gpu { ctx: 1024 });
    let p = F16.place_stream(free, 8192, true, false, big);
    assert!(
        matches!(
            p,
            Placement::Split {
                gpu_ctx: Some(1024),
                ..
            }
        ),
        "{p:?}"
    );
    // The full context fitting whole never asks about a split.
    let free = F16.need(TOY_N, 8192);
    assert_eq!(
        F16.place_stream(free, 8192, true, true, big),
        Placement::Gpu { ctx: 8192 }
    );
}

/// Sampling on a split (`Lm::forward_cands` / `verify_cands` by default
/// through `forward`: the logits come back, the CPU filters them; verify
/// is token by token, never streamed): sample.rs's real-model checks
/// (drafting == plain sampling for the same seed, seeds differ) with f16
/// and q8 GPU halves, and with streamed prefill (every prompt streams).
#[test]
fn hybrid_real_model_sampling() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    for rel in [M05, Q3] {
        real_model_sampling_split(&g, rel);
    }
}

fn real_model_sampling_split(g: &crate::cuda::Gpu, rel: &str) {
    let Some(m) = mapped(rel) else {
        return;
    };
    let km = Kernels::module(g).unwrap();
    let n = block_count(&m).unwrap();
    for kv in [CacheKv::F16, CacheKv::Q8 { group: 0 }] {
        let h = HybridQwen2::new(Some((g, &km)), &m, 256, n / 2, kv).unwrap();
        crate::sample::tests::real_model_sampling_on(&h, &m);
        let h = streamed_kv(g, &km, &m, 256, n / 2, 1, kv);
        crate::sample::tests::real_model_sampling_on(&h, &m);
        assert!(h.streamed_chunks() > 0, "{kv:?}: prefills streamed");
    }
}
