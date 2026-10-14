use super::*;
use crate::cuda::Kernels;
use crate::cuda::qwen2::tests::{Agree, PROMPTS, assert_graded, compare, report};
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
        &HybridQwen2::new(
            Some((&g, &km)),
            &m,
            128,
            n,
            crate::cuda::qwen2::CacheKv::F16,
        )
        .unwrap(),
        &m,
        16,
    );
    assert!(
        got.iter().zip(&want).all(|(a, b)| bits(a) == bits(b)),
        "all on GPU"
    );
    // More than n is all of them too.
    let got = trace(
        &HybridQwen2::new(
            Some((&g, &km)),
            &m,
            128,
            n + 5,
            crate::cuda::qwen2::CacheKv::F16,
        )
        .unwrap(),
        &m,
        2,
    );
    let want2 = trace(&CudaQwen2::new(&g, &km, &m, 128).unwrap(), &m, 2);
    assert!(
        got.iter().zip(&want2).all(|(a, b)| bits(a) == bits(b)),
        "n + 5"
    );

    let want = trace(&Qwen2::new(&m), &m, 16);
    let h = HybridQwen2::new(None, &m, 128, 0, crate::cuda::qwen2::CacheKv::F16).unwrap();
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
        let h = HybridQwen2::new(
            Some((&g, &km)),
            &m,
            256,
            n - k,
            crate::cuda::qwen2::CacheKv::F16,
        )
        .unwrap();
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
/// halves' KV caches and the next token's logits are bit-identical.
fn prefill_bits(rel: &str, name: &str) {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = mapped(rel) else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let n = block_count(&m).unwrap();
    let ids = long_ids(&m, 700);
    let h = HybridQwen2::new(
        Some((&g, &km)),
        &m,
        ids.len() + 1,
        n - n / 2,
        crate::cuda::qwen2::CacheKv::F16,
    )
    .unwrap();
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
        "{name} k={}: prefill bit-identical over 6 batchings of {} tokens",
        n / 2,
        ids.len()
    );
}

#[test]
fn hybrid_prefill_batchings_bit_identical_0_5b() {
    prefill_bits(M05, "0.5B");
}

#[test]
fn hybrid_prefill_batchings_bit_identical_1_5b() {
    prefill_bits(M15, "1.5B");
}

/// verify_argmax's picks equal forward_argmax per token (and leave the
/// same KV in both halves), for right, wrong and random drafts, small and
/// past the GPU's max batch; with k = 1 and n - 1 too.
#[test]
fn hybrid_verify_matches_forward_argmax() {
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
        let h = HybridQwen2::new(
            Some((&g, &km)),
            &m,
            1400,
            n - k,
            crate::cuda::qwen2::CacheKv::F16,
        )
        .unwrap();
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
            "0.5B k={k}: {} verify passes == per-token picks",
            cases.len()
        );
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
    let h = HybridQwen2::new(
        Some((&g, &km)),
        &m,
        64,
        n - k,
        crate::cuda::qwen2::CacheKv::F16,
    )
    .unwrap();
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
        CudaQwen2::vram_need(&g, &km, &m, ctx, first, crate::cuda::qwen2::CacheKv::F16).unwrap()
    };
    for f in 1..n {
        assert!(need(f, 4096) < need(f - 1, 4096), "plan at first {f}");
    }
    let free = g.info().unwrap().free_bytes;
    // A context too big for the whole model on any GPU this runs on: all
    // layers' KV would take 1.5x the free VRAM, so about half fit.
    let per_tok = (need(0, 2048) - need(0, 1024)) / 1024;
    let ctx = (free * 3 / 2 / per_tok) as usize;
    let auto = auto_gpu_layers(&g, &km, &m, ctx, crate::cuda::qwen2::CacheKv::F16, free).unwrap();
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
    // An explicit count that doesn't fit (all: 1.5x free in KV alone) says
    // which and what to do.
    let e = HybridQwen2::new(
        Some((&g, &km)),
        &m,
        ctx,
        n,
        crate::cuda::qwen2::CacheKv::F16,
    )
    .err()
    .expect("all layers can't fit");
    assert!(
        e.contains(&format!("{n}/{n} layers")) && e.contains("or fewer --gpu-layers"),
        "{e}"
    );
}
