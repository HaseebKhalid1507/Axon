//! gpt-oss stage 1 tests: attention (sinks, alternating windows) and the
//! biased router on the GPU, the MXFP4 experts on the CPU (`MoeQwen3`).
//! decode == batched prefill == verify bit for bit, cache reuse == a fresh
//! prefill, a paged cache == a plain one; the router against its CPU twin on
//! identical states; the GPU half against the CPU-only model on a small
//! fixture (with the sinks, the window and the router bias each shown to
//! matter) and, graded, on the real model when it's there.

use super::tests::moe_net;
use super::*;
use crate::cuda::Kernels;
use crate::cuda::qwen2::tests::{Agree, PROMPTS, assert_graded_floor, compare, report};
use crate::cuda::qwen2::{CacheKv, Head};
use crate::gguf::Val;
use crate::model::gptoss_tests::{OssDims, Ts, oss_fixture_dims};

/// (layers, dim, q heads, kv heads, head dim, expert ffn, experts, top-k,
/// vocab, window): head size 64 and 8 query heads per kv head (the 20B's),
/// 3 layers (local, global, local), window 4.
const GPU_OSS: OssDims = (3, 256, 16, 2, 64, 64, 4, 2, 64, 4);

fn fixture() -> Model {
    oss_fixture_dims(GPU_OSS, |_| {}, |_| {})
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn argmax(l: &[f32]) -> u32 {
    crate::argmax(l)
}

/// The fixture with every layer's tensor `name` (F32) set to `v`.
fn with_tensor(name: &'static str, v: f32) -> impl FnOnce(&mut Ts) {
    move |ts: &mut Ts| {
        for t in ts.iter_mut().filter(|t| t.0.ends_with(name)) {
            let n = t.3.len() / 4;
            t.3 = (0..n).flat_map(|_| v.to_le_bytes()).collect();
        }
    }
}

/// decode == feed_batch (every verify size, odd ones, past the window and
/// the GPU's small-batch kernels: 483 tokens), KV and next logits bit for
/// bit; prefill_logits == feed + forward; verify_argmax == forward per row
/// with the same cache; a cache built as prefix + rest == one fresh
/// prefill.
fn contract(net: &MoeQwen3, ids: &[u32], next: u32) {
    let n = ids.len();
    let mut a = net.new_cache(n + 1);
    let mut b = net.new_cache(n + 1);
    for (pos, &id) in ids.iter().enumerate() {
        net.forward_argmax(id, pos, &mut a);
    }
    let mut s = 0;
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 64, 86, 128, 130] {
        let k = k.min(n - s);
        net.feed_batch(&ids[s..s + k], s, &mut b);
        s += k;
    }
    assert_eq!(s, n);
    assert!(
        CudaQwen2::kv_bits(&a.c) == CudaQwen2::kv_bits(&b.c),
        "KV caches differ"
    );
    let la = net.forward(next, n, &mut a);
    let lb = net.forward(next, n, &mut b);
    assert_eq!(bits(&la), bits(&lb), "logits differ");
    // prefill_logits == feed_batch + forward.
    let cut = 200.min(n);
    let mut c = net.new_cache(n + 1);
    let lc = net.prefill_logits(&ids[..cut], 0, &mut c);
    let mut d = net.new_cache(n + 1);
    net.feed_batch(&ids[..cut - 1], 0, &mut d);
    let ld = net.forward(ids[cut - 1], cut - 1, &mut d);
    assert_eq!(bits(&lc), bits(&ld), "prefill_logits != feed + forward");
    // verify == forward (1, 5, 16, 32 rows after a 28-token prefill).
    let (pre, tail) = ids[..60].split_at(28);
    for tail in [&tail[..1], &tail[..5], &tail[..16], tail] {
        let mut a = net.new_cache(n + 1);
        net.feed_batch(pre, 0, &mut a);
        let mut b = net.new_cache(n + 1);
        net.feed_batch(pre, 0, &mut b);
        let picks = net.verify_argmax(tail, pre.len(), &mut a);
        for (j, &t) in tail.iter().enumerate() {
            let f = net.forward(t, pre.len() + j, &mut b);
            assert_eq!(picks[j], argmax(&f), "verify row {j} of {}", tail.len());
        }
        assert!(
            CudaQwen2::kv_bits(&a.c) == CudaQwen2::kv_bits(&b.c),
            "verify's cache"
        );
    }
    // Reuse: a prefix, then the rest, then decode == one prefill + decode.
    let r = n / 3;
    let mut e = net.new_cache(n + 1);
    net.feed_batch(&ids[..r], 0, &mut e);
    net.feed_batch(&ids[r..n - 1], r, &mut e);
    let le = net.forward(ids[n - 1], n - 1, &mut e);
    let mut f = net.new_cache(n + 1);
    net.feed_batch(&ids[..n - 1], 0, &mut f);
    let lf = net.forward(ids[n - 1], n - 1, &mut f);
    assert!(
        CudaQwen2::kv_bits(&e.c) == CudaQwen2::kv_bits(&f.c),
        "reused-prefix cache != fresh prefill"
    );
    assert_eq!(bits(&le), bits(&lf), "reused-prefix logits");
}

/// A prefill in two passes, 20 graphed decode steps (graphs dropped and
/// remade at step `drop_at`), a verify pass: the picks.
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
            c.c.drop_graphs();
        }
        tok = net.forward_argmax(tok, 250 + i, c);
        out.push(tok);
    }
    let guess = out[..5].to_vec();
    out.extend(net.verify_argmax(&guess, 270, c));
    out
}

/// `script` on a plain cache == after drop_graphs == through a shuffled
/// page table (oss.cu's kernels with a table): the same picks, then one
/// more `forward` with the same logits, bit for bit.
fn paged(g: &Gpu, net: &MoeQwen3, prompt: &[u32]) {
    const PAGE: usize = 64;
    let ctx = 640;
    let mut c0 = net.new_cache(ctx);
    let want = script(net, prompt, &mut c0, None);
    let mut c2 = net.new_cache(ctx);
    assert_eq!(script(net, prompt, &mut c2, Some(7)), want, "drop_graphs");
    let pages = ctx / PAGE;
    let phys = pages + pages / 2 + 3;
    let mut pool: Vec<u32> = (0..phys as u32).rev().collect();
    pool.truncate(pages);
    let mut c1 = net.new_cache(phys * PAGE);
    c1.c.set_tab(Some(g.upload(&pool).unwrap()));
    assert_eq!(script(net, prompt, &mut c1, None), want, "paged cache");
    // script leaves positions 0..275 written: one more step on each.
    let next = want[want.len() - 1];
    let l0 = net.forward(next, 275, &mut c0);
    assert_eq!(
        bits(&net.forward(next, 275, &mut c2)),
        bits(&l0),
        "drop_graphs logits"
    );
    assert_eq!(
        bits(&net.forward(next, 275, &mut c1)),
        bits(&l0),
        "paged logits"
    );
}

fn ids_mod(n: u64, v: u64) -> Vec<u32> {
    (0..n)
        .map(|i| (i * 2_654_435_761 % 150_000 % v) as u32)
        .collect()
}

#[test]
fn oss_gpu_contract_fixture() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let m = fixture();
    let km = Kernels::module_for(&g, &m).unwrap();
    let mm = Kernels::moe_module(&g).unwrap();
    let net = moe_net(&g, &km, &mm, &m, 1024);
    let v = net.vocab() as u64;
    contract(&net, &ids_mod(483, v), (9707 % v) as u32);
    paged(&g, &net, &ids_mod(300, v));
}

/// The real model: the same contract (window 128: 483 tokens cross it), the
/// paged cache, and a sanity answer.
#[test]
fn oss_gpu_contract_model() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::gptoss_model() else {
        return;
    };
    let km = Kernels::module_for(&g, &m).unwrap();
    let mm = Kernels::moe_module(&g).unwrap();
    let net = moe_net(&g, &km, &mm, &m, 1024);
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let p = tok.encode_prompt("The capital of France is");
    let mut c = net.new_cache(64);
    let l = net.prefill_logits(&p, 0, &mut c);
    assert_eq!(tok.decode(&[argmax(&l)]), " Paris");
    let v = net.vocab() as u64;
    contract(&net, &ids_mod(483, v), (9707 % v) as u32);
    paged(&g, &net, &ids_mod(300, v));
}

/// max |dlogit| between two models' logits at every position of `ids`
/// (each fed one token at a time).
fn max_dlogit<A: Lm, B: Lm>(a: &A, b: &B, ids: &[u32]) -> f32 {
    let (mut ca, mut cb) = (a.new_cache(ids.len()), b.new_cache(ids.len()));
    let mut d = 0f32;
    for (p, &id) in ids.iter().enumerate() {
        let (la, lb) = (a.forward(id, p, &mut ca), b.forward(id, p, &mut cb));
        for (x, y) in la.iter().zip(&lb) {
            d = d.max((x - y).abs());
        }
    }
    d
}

/// The GPU half vs the CPU-only model on the fixture (40 positions, 10x
/// the window): close; and the CPU model with no sinks (-1e4: exp 0), no
/// window, or no router bias is far from both, so each is the GPU's.
#[test]
fn oss_gpu_matches_cpu_fixture() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let m = fixture();
    let km = Kernels::module_for(&g, &m).unwrap();
    let mm = Kernels::moe_module(&g).unwrap();
    let net = moe_net(&g, &km, &mm, &m, 64);
    let ids = ids_mod(40, net.vocab() as u64);
    // A model of its own: the GPU half's CPU experts released what they
    // repacked.
    let mc = fixture();
    let cpu = crate::model::Qwen2::new(&mc);
    let d = max_dlogit(&net, &cpu, &ids);
    let bugs: [(&str, Model); 3] = [
        (
            "no sinks",
            oss_fixture_dims(GPU_OSS, with_tensor("attn_sinks.weight", -1e4), |_| {}),
        ),
        (
            "no window",
            oss_fixture_dims(
                GPU_OSS,
                |_| {},
                |m| {
                    for kv in m.iter_mut() {
                        if kv.0 == "gpt-oss.attention.sliding_window" {
                            kv.1 = Val::U32(4096);
                        }
                    }
                },
            ),
        ),
        (
            "no router bias",
            oss_fixture_dims(GPU_OSS, with_tensor("ffn_gate_inp.bias", 0.0), |_| {}),
        ),
    ];
    let mut far = Vec::new();
    for (name, bm) in &bugs {
        let bad = crate::model::Qwen2::new(bm);
        far.push((*name, max_dlogit(&net, &bad, &ids)));
    }
    eprintln!("oss fixture: GPU vs CPU max |dlogit| {d:.3}; vs the CPU with a bug: {far:?}");
    assert!(d < 0.5, "GPU vs CPU max |dlogit| {d}");
    for (name, b) in far {
        assert!(b > 3.0 * d.max(0.1), "{name}: {b} vs {d}");
    }
}

/// The real model, GPU half vs CPU-only, graded as the dense GPU is against
/// the CPU (`assert_graded_floor`): PROMPTS x 32 greedy steps
/// (AXON_TEST_MOE_STEPS), against the CPU's own spread (r8 vs per-row
/// kernels).
#[test]
fn oss_gpu_logits_graded_model() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::gptoss_model() else {
        return;
    };
    let km = Kernels::module_for(&g, &m).unwrap();
    let mm = Kernels::moe_module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let rows = crate::model::Qwen2::new_with(&m, false);
    let net = moe_net(&g, &km, &mm, &m, 256);
    let cpu = crate::model::Qwen2::new_with(&m, true);
    let steps = std::env::var("AXON_TEST_MOE_STEPS").map_or(32, |v| v.parse().unwrap());
    let (mut ag, mut ac) = (Agree::default(), Agree::default());
    for p in PROMPTS {
        let ids = tok.encode_prompt(p);
        compare(&cpu, &net, &ids, steps, &mut ag);
        compare(&cpu, &rows, &ids, steps, &mut ac);
    }
    report("gpt-oss-20b moe gpu vs cpu", &ag);
    report("gpt-oss-20b cpu rows vs cpu", &ac);
    assert_graded_floor(&ag, &ac, 90);
}

/// The GPU router (moe_logits + moe_router's OAI gate) vs its CPU twin
/// (`Moe::route_oai`) on IDENTICAL normed states (those the GPU handed the
/// CPU experts over a prefill): sets differ only at near-ties of the CPU's
/// own k-th / (k+1)-th probabilities, same-set weights within 1e-5.
fn route_twin(net: &MoeQwen3, ids: &[u32], name: &str) {
    type Seen = (usize, Vec<f32>, Vec<i32>, Vec<f32>);
    let mut seen: Vec<Seen> = Vec::new();
    let mut c = net.new_cache(ids.len());
    let mut ex = |l: usize, h: &[f32], i: &[i32], w: &[f32], y: &mut [f32]| {
        seen.push((l, h.to_vec(), i.to_vec(), w.to_vec()));
        net.experts(l, h, i, w, y)
    };
    let (cc, mb) = c.moe_parts();
    net.gpu
        .moe_tokens(&net.kind, cc, mb, ids, 0, Head::No, &mut ex)
        .unwrap();
    let dim = net.gpu.dim;
    let (mut n, mut sets, mut wmax, mut gap_max) = (0, 0, 0f32, 0f32);
    for (l, h, gi, gw) in &seen {
        let k = gi.len() / (h.len() / dim);
        for (t, hr) in h.chunks_exact(dim).enumerate() {
            let (ci, cw) = net.cpu.moe_route(*l, hr, false);
            let gi: Vec<usize> = gi[t * k..(t + 1) * k].iter().map(|&e| e as usize).collect();
            n += 1;
            if ci != gi {
                sets += 1;
                let (_, p) = net.cpu.moe_route(*l, hr, true);
                gap_max = gap_max.max(p[k - 1] - p[k]);
                continue;
            }
            for (x, y) in cw.iter().zip(&gw[t * k..(t + 1) * k]) {
                wmax = wmax.max((x - y).abs());
            }
        }
    }
    eprintln!(
        "{name}: gpt-oss router GPU vs CPU twin on identical h: {n} routes, {sets} differ \
         (widest CPU k-th vs (k+1)-th probability gap there {gap_max:.2e}), same-set max |w| \
         diff {wmax:.2e}"
    );
    assert!(n > 0);
    assert!(gap_max < 1e-4, "{name}: a set differs at gap {gap_max}");
    assert!(wmax < 1e-5, "{name}: weights differ by {wmax}");
}

#[test]
fn oss_gpu_route_matches_cpu_twin() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let m = fixture();
    let km = Kernels::module_for(&g, &m).unwrap();
    let mm = Kernels::moe_module(&g).unwrap();
    let net = moe_net(&g, &km, &mm, &m, 512);
    route_twin(&net, &ids_mod(300, net.vocab() as u64), "fixture");
    let Some(m) = crate::testutil::gptoss_model() else {
        return;
    };
    let km = Kernels::module_for(&g, &m).unwrap();
    let net = moe_net(&g, &km, &mm, &m, 1024);
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let ids: Vec<u32> = PROMPTS.iter().flat_map(|p| tok.encode(p)).collect();
    route_twin(&net, &ids, "gpt-oss-20b");
}

/// moe_router's OAI gate on its own: a token's ids and weights are the same bits
/// alone as in a batch of 512; ties (equal biased logits) go to the lower
/// index first with equal weights; the weights are the softmax of the top
/// k biased logits (to 1e-6) and sum to 1.
#[test]
fn oss_topk_batch_invariant_and_ties() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let mm = Kernels::moe_module(&g).unwrap();
    let f = mm.function("moe_router").unwrap();
    let run = |lg: &[f32], b: &[f32], k: usize| -> (Vec<i32>, Vec<f32>) {
        let n_e = b.len();
        let nb = lg.len() / n_e;
        let (ld, bd) = (g.upload(lg).unwrap(), g.upload(b).unwrap());
        let (ids, w) = (g.alloc(nb * k * 4).unwrap(), g.alloc(nb * k * 4).unwrap());
        // SAFETY: moe_router's signature; buffers sized for nb tokens.
        unsafe {
            crate::cuda::qwen2::moe::launch_router(
                &g,
                &f,
                ld.arg(),
                (n_e, k, nb),
                crate::cuda::qwen2::moe::GATE_OAI,
                bd.arg(),
                false,
                1.0,
                (ids.arg(), w.arg()),
            )
            .unwrap();
        }
        g.sync().unwrap();
        (ids.read().unwrap(), w.read().unwrap())
    };
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    let mut uni = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
    };
    for (n_e, k) in [(32, 4), (128, 4), (200, 6)] {
        let nb = 512;
        let lg: Vec<f32> = (0..nb * n_e).map(|_| 4.0 * uni()).collect();
        let b: Vec<f32> = (0..n_e).map(|_| uni()).collect();
        let (ids, w) = run(&lg, &b, k);
        for t in 0..nb {
            let row = &lg[t * n_e..(t + 1) * n_e];
            if [0, 1, 77, 300, 511].contains(&t) {
                let (i1, w1) = run(row, &b, k);
                assert_eq!(i1, ids[t * k..(t + 1) * k], "ids of token {t}");
                assert_eq!(bits(&w1), bits(&w[t * k..(t + 1) * k]), "w of token {t}");
            }
            let l: Vec<f32> = row.iter().zip(&b).map(|(x, y)| x + y).collect();
            let mut o: Vec<usize> = (0..n_e).collect();
            o.sort_by(|&x, &y| l[y].total_cmp(&l[x]).then(x.cmp(&y)));
            let want: Vec<i32> = o[..k].iter().map(|&e| e as i32).collect();
            assert_eq!(ids[t * k..(t + 1) * k], want, "top {k} of token {t}");
            let e: Vec<f64> = o[..k]
                .iter()
                .map(|&i| ((l[i] - l[o[0]]) as f64).exp())
                .collect();
            let s: f64 = e.iter().sum();
            for (j, ej) in e.iter().enumerate() {
                let got = w[t * k + j] as f64;
                assert!((got - ej / s).abs() < 1e-6, "w[{j}] {got} vs {}", ej / s);
            }
        }
        // Ties: two experts with the same biased logit, above the rest.
        let mut row = vec![0f32; n_e];
        let (a, c) = (n_e / 3, n_e - 1);
        let mut b2 = b.clone();
        b2[c] = b2[a];
        row[a] = 10.0;
        row[c] = 10.0;
        let (i1, w1) = run(&row, &b2, k);
        assert_eq!(&i1[..2], &[a as i32, c as i32], "tie: lower index first");
        assert_eq!(w1[0].to_bits(), w1[1].to_bits());
    }
}

/// `Backend::new` on the GPU places gpt-oss as the MoE backend; --gpu-layers
/// below the layer count and the q8 KV cache are refused.
#[test]
fn backend_places_gpt_oss() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let m = fixture();
    use crate::lm::{Backend, Device, Inner};
    let b = Backend::new(&m, Device::Cuda, 64, None, crate::kv::KvType::F16, false).unwrap();
    assert!(matches!(b.0, Inner::Moe(_)));
    let e = Backend::new(&m, Device::Cuda, 64, Some(1), crate::kv::KvType::F16, false);
    assert!(e.is_err_and(|e| e.contains("doesn't split by layers")));
    let e = Backend::new(&m, Device::Cuda, 64, None, crate::kv::KvType::Q8, false);
    assert!(e.is_err_and(|e| e.contains("q8")));
    // The plain qwen2 path (CudaQwen2::with_kv) refuses it too.
    let g = crate::testutil::gpu().unwrap();
    let km = Kernels::module_for(&g, &m).unwrap();
    assert!(CudaQwen2::with_kv(&g, &km, &m, 64, CacheKv::F16, false).is_err());
}
