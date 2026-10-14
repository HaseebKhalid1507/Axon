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
        CudaQwen2::vram_need(&g, &km, &m, ctx, first, CacheKv::F16).unwrap()
    };
    for f in 1..n {
        assert!(need(f, 4096) < need(f - 1, 4096), "plan at first {f}");
    }
    let free = g.info().unwrap().free_bytes;
    // A context too big for the whole model on any GPU this runs on: all
    // layers' KV would take 1.5x the free VRAM, so about half fit.
    let per_tok = (need(0, 2048) - need(0, 1024)) / 1024;
    let ctx = (free * 3 / 2 / per_tok) as usize;
    let auto = auto_gpu_layers(&g, &km, &m, ctx, CacheKv::F16, free).unwrap();
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
        CudaQwen2::vram_need(&g, &km, &m, ctx, first, kv).unwrap()
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
    let a16 = auto_gpu_layers(&g, &km, &m, ctx, CacheKv::F16, free).unwrap();
    let a8 = auto_gpu_layers(&g, &km, &m, ctx, q8, free).unwrap();
    eprintln!("1.5B ctx {ctx}: auto split f16 {a16}/{n}, q8 {a8}/{n}");
    let fits = |gl: usize| need(n - gl, ctx, q8) + DESKTOP_MARGIN <= free;
    assert!(a8 == 0 || fits(a8), "chosen q8 split doesn't fit");
    assert!(a8 == n || !fits(a8 + 1), "one more q8 layer would have fit");
    assert!(a8 > a16, "q8 fits more layers ({a8} vs {a16})");
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
