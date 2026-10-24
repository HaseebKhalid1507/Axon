//! deepseek2 on the GPU (r12 stage 1, `cuda::qwen2::mla`): decode ==
//! batched prefill == verify bit for bit and graded against the CPU-only
//! model, on the synthetic fixture (every option: q_lora, YaRN) and, when
//! AXON_TEST_DS2 names it, on Moonlight.

use super::*;
use crate::cuda::Kernels;
use crate::lm::Lm;
use crate::model::deepseek2_tests::{Opt, ds2_fixture};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The fixture's vocabulary (`deepseek2_tests::FX`).
const FX_VOCAB: usize = crate::model::deepseek2_tests::FX.vocab;

/// The fixture's four option sets.
const OPTS: [Opt; 4] = [
    Opt {
        q_lora: false,
        yarn: false,
    },
    Opt {
        q_lora: true,
        yarn: false,
    },
    Opt {
        q_lora: false,
        yarn: true,
    },
    Opt {
        q_lora: true,
        yarn: true,
    },
];

/// The deepseek2 backend of `m` on `g` (f16 cache of `ctx`).
fn ds2_net<'m, 'g>(
    g: &'g Gpu,
    km: &'g Module<'g>,
    ml: &'g Module<'g>,
    m: &'m Model,
    ctx: usize,
) -> MoeQwen3<'m, 'g> {
    MoeQwen3::new(g, km, ml, m, ctx, CacheKv::F16, false, true).unwrap()
}

/// Token ids < v, spread.
fn ids_for(n: usize, v: usize, seed: u64) -> Vec<u32> {
    (0..n as u64)
        .map(|i| ((i + seed) * 2_654_435_761 % 150_000 % v as u64) as u32)
        .collect()
}

/// decode one by one == feed_batch in passes of many sizes (the caches and
/// the next logits, bit for bit); prefill_logits == feed_batch + forward;
/// verify_argmax == forward_argmax per row, the same caches; graphs off ==
/// on.
fn bits_on(net: &mut MoeQwen3, ids: &[u32], splits: &[usize], what: &str) {
    let n = ids.len();
    let next = ids[0];
    let mut a = net.new_cache(n + 1);
    let mut dec = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        dec.push(net.forward(id, pos, &mut a));
    }
    let bad = dec.iter().flatten().filter(|v| !v.is_finite()).count();
    assert_eq!(bad, 0, "{what}: non-finite decode logits");
    let mut b = net.new_cache(n + 1);
    let mut s = 0;
    for &k in splits.iter().cycle() {
        if s == n {
            break;
        }
        let k = k.min(n - s);
        net.feed_batch(&ids[s..s + k], s, &mut b);
        s += k;
    }
    assert!(
        CudaQwen2::kv_bits(&a.c) == CudaQwen2::kv_bits(&b.c),
        "{what}: KV caches differ (decode vs feed_batch)"
    );
    let la = net.forward(next, n, &mut a);
    let lb = net.forward(next, n, &mut b);
    assert_eq!(bits(&la), bits(&lb), "{what}: logits after feed_batch");
    // prefill_logits (one pass, the head on the last row).
    for cut in [2, n / 2, n] {
        let mut c = net.new_cache(n + 1);
        let lc = net.prefill_logits(&ids[..cut], 0, &mut c);
        assert_eq!(
            bits(&lc),
            bits(&dec[cut - 1]),
            "{what}: prefill_logits of {cut} vs decode"
        );
    }
    // verify after a batched prefix.
    let pre = n / 3;
    for len in [1, 5, n - pre] {
        let tail = &ids[pre..pre + len];
        let mut v = net.new_cache(n + 1);
        net.feed_batch(&ids[..pre], 0, &mut v);
        let picks = net.verify_argmax(tail, pre, &mut v);
        for (j, p) in picks.iter().enumerate() {
            assert_eq!(
                *p,
                crate::argmax(&dec[pre + j]),
                "{what}: verify row {j} of {len}"
            );
        }
        if len == n - pre {
            // The cache verify left: the same next logits as decode's.
            let lv = net.forward(next, n, &mut v);
            assert_eq!(bits(&lv), bits(&la), "{what}: logits after verify");
        }
    }
    // Graphs off: the same kernels launched one by one.
    let graphs = net.gpu.graphs;
    net.gpu.graphs = !graphs;
    let mut e = net.new_cache(n + 1);
    for (pos, &id) in ids.iter().enumerate() {
        let l = net.forward(id, pos, &mut e);
        assert_eq!(bits(&l), bits(&dec[pos]), "{what}: graphs {}", !graphs);
    }
    net.gpu.graphs = graphs;
}

/// Spec r12 stage 1 gate (b) on the fixture: 60 positions (two attention
/// tiles; 150 with YaRN: two splits), every option.
#[test]
fn ds2_gpu_fixture_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let ml = Kernels::mla_module(&g).unwrap();
    for o in OPTS {
        let (m, _) = ds2_fixture(o, crate::quant::Q8_0);
        // YaRN's 256-position fixture: past an attention split (128 rows).
        let n = if o.yarn { 150 } else { 60 };
        let mut net = ds2_net(&g, &km, &ml, &m, n + 1);
        let ids = ids_for(n, net.vocab(), 3);
        bits_on(
            &mut net,
            &ids,
            &[1, 2, 3, 5, 8, 13, 17, 11],
            &format!("{o:?}"),
        );
    }
}

/// Per-position teacher-forced logits of `net` over `ids` (decode).
fn decode_all<L: Lm>(net: &L, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut c = net.new_cache(ids.len());
    (0..ids.len())
        .map(|p| net.forward(ids[p], p, &mut c))
        .collect()
}

/// (top-1 agreements, mean |dlogit|, max |dlogit|) of `a` against the
/// reference `r`, per position.
fn vs_ref(r: &[Vec<f32>], a: &[Vec<f32>]) -> (usize, f64, f32) {
    let (mut top, mut sum, mut mx, mut n) = (0, 0.0, 0f32, 0);
    for (lr, la) in r.iter().zip(a) {
        top += (crate::argmax(lr) == crate::argmax(la)) as usize;
        for (x, y) in lr.iter().zip(la) {
            let d = (x - y).abs();
            sum += d as f64;
            mx = mx.max(d);
            n += 1;
        }
    }
    (top, sum / n as f64, mx)
}

/// Gate (c) on the fixture: the GPU model and the CPU-only one (both
/// absorbed, both quantizing activations to 8 bits, in other kernels)
/// against the f64 *expanded* reference (`deepseek2_ref`, the stage 0
/// oracle) on teacher-forced ids: the GPU no further from it than the CPU
/// (mean |dlogit| at most 1.5x the CPU's, top-1 at most 2 positions
/// behind). Not GPU vs CPU directly: on the fixture (random weights,
/// vocab 64, near-tied routers everywhere) two correct 8-bit models flip
/// experts and top-1 against each other often.
#[test]
fn ds2_gpu_fixture_graded() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let ml = Kernels::mla_module(&g).unwrap();
    for o in OPTS {
        // Copies: each model releases what it repacks.
        let (m, raw) = ds2_fixture(o, crate::quant::Q8_0);
        let (mc, _) = ds2_fixture(o, crate::quant::Q8_0);
        let (mr, _) = ds2_fixture(o, crate::quant::Q8_0);
        let n = if o.yarn { 160 } else { 60 };
        let ids = ids_for(n, FX_VOCAB, 7);
        let r = crate::model::deepseek2_tests::deepseek2_ref(
            &mr,
            &raw,
            &ids,
            crate::model::deepseek2_tests::Bug::None,
        );
        let net = ds2_net(&g, &km, &ml, &m, n);
        let gl = decode_all(&net, &ids);
        drop(net);
        let cl = decode_all(&Qwen2::new(&mc), &ids);
        let (gt, gm, gx) = vs_ref(&r, &gl);
        let (ct, cm, cx) = vs_ref(&r, &cl);
        eprintln!(
            "ds2 fixture {o:?} vs the f64 reference ({n} positions): gpu top-1 {gt}, mean \
             |dlogit| {gm:.4}, max {gx:.3}; cpu top-1 {ct}, mean {cm:.4}, max {cx:.3}"
        );
        assert!(
            graded_ok((gt, gm), (ct, cm)),
            "{o:?}: gpu top-1 {gt}, mean |dlogit| {gm} vs cpu {ct}, {cm}"
        );
    }
}

/// `ds2_gpu_fixture_graded`'s criterion: a model's (top-1, mean |dlogit|)
/// against the reference, given the CPU model's.
fn graded_ok((top, mean): (usize, f64), (ct, cm): (usize, f64)) -> bool {
    mean <= 1.5 * cm && top + 2 >= ct
}

/// Whether `ds2_gpu_fixture_graded`'s criterion has teeth: each of stage
/// 0's deliberate bugs, run through the f64 reference (a GPU model with
/// that bug would sit about there), graded against the correct reference
/// with the CPU model as the yardstick, must fail it. CPU only.
#[test]
fn ds2_fixture_graded_criterion_catches_bugs() {
    use crate::model::deepseek2_tests::{Bug, deepseek2_ref};
    let mut slipped = Vec::new();
    for o in OPTS {
        let (mc, raw) = ds2_fixture(o, crate::quant::Q8_0);
        let (mr, _) = ds2_fixture(o, crate::quant::Q8_0);
        let n = if o.yarn { 160 } else { 60 };
        let ids = ids_for(n, FX_VOCAB, 7);
        let r = deepseek2_ref(&mr, &raw, &ids, Bug::None);
        let (ct, cm, _) = vs_ref(&r, &decode_all(&Qwen2::new(&mc), &ids));
        let bugs = [
            Bug::RopeWrongPart,
            Bug::ScaleOfKeyRow,
            Bug::BiasInWeights,
            Bug::ScaleBeforeRenorm,
            Bug::NormOverRow,
            Bug::NoYarn,
            Bug::MscaleOnTable,
            Bug::NoMscale,
        ];
        for bug in bugs {
            if !o.yarn && matches!(bug, Bug::NoYarn | Bug::MscaleOnTable | Bug::NoMscale) {
                continue; // no YaRN: the same model
            }
            let (bt, bm, bx) = vs_ref(&r, &deepseek2_ref(&mr, &raw, &ids, bug));
            eprintln!(
                "{o:?} {bug:?}: top-1 {bt} (cpu {ct}), mean |dlogit| {bm:.4} = {:.1}x cpu's \
                 {cm:.4}, max {bx:.3}",
                bm / cm
            );
            if graded_ok((bt, bm), (ct, cm)) {
                slipped.push((o, bug));
            }
        }
    }
    assert!(slipped.is_empty(), "bugs the criterion misses: {slipped:?}");
}

/// Gate (b) and (e) on Moonlight (AXON_TEST_DS2=1): a 200-token prompt
/// decoded vs fed in passes vs verified, bit for bit; and a two-turn chat
/// in one cache vs the whole history fed fresh.
#[test]
fn ds2_gpu_moonlight_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::ds2_model() else {
        return;
    };
    let km = Kernels::module_for(&g, &m).unwrap();
    let ml = Kernels::mla_module(&g).unwrap();
    let mut net = ds2_net(&g, &km, &ml, &m, 1024);
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let mut ids =
        tok.encode(&"The quick brown fox jumps over the lazy dog. Then it rests. ".repeat(12));
    ids.truncate(200);
    bits_on(&mut net, &ids, &[1, 7, 64, 3, 100], "moonlight");
    let l = net.prefill_logits(&ids[..20], 0, &mut net.new_cache(32));
    let top = crate::argmax(&l);
    eprintln!("moonlight gpu: next token {top} {:?}", tok.decode(&[top]));

    use crate::chat::Chat;
    let sys = crate::chat::template_default_system(&m);
    let mut chat = Chat::new(&net, &tok, sys.as_deref(), 512).unwrap();
    chat.max_reply = 16;
    let never = || false;
    let r1 = chat
        .reply("Name a color.", &never, &mut |_| Ok(()))
        .unwrap();
    let r2 = chat
        .reply("And another one?", &never, &mut |_| Ok(()))
        .unwrap();
    let hist = chat.history_for_test();
    let cut = (0..=hist.len() - r2.ids.len())
        .rev()
        .find(|&p| hist[p..p + r2.ids.len()] == r2.ids[..])
        .expect("reply in history");
    let mut c = net.new_cache(512);
    net.feed_batch(&hist[..cut - 1], 0, &mut c);
    let mut last = net.forward(hist[cut - 1], cut - 1, &mut c);
    for (i, &t) in r2.ids.iter().enumerate() {
        assert_eq!(crate::argmax(&last), t, "reply token {i}: reuse vs fresh");
        last = net.forward(t, cut + i, &mut c);
    }
    eprintln!(
        "moonlight gpu chat: {:?} / {:?}",
        tok.decode(&r1.ids),
        tok.decode(&r2.ids)
    );
}

/// The logits of `net` over `ids`: decode one by one, then the same
/// prompt fed in passes of `splits` and its last logits (prefill_logits),
/// then a greedy continuation of `ngen` tokens from there (decode).
fn resident_trace(net: &MoeQwen3, ids: &[u32], splits: &[usize], ngen: usize) -> Vec<Vec<f32>> {
    let n = ids.len();
    let mut out = Vec::new();
    let mut a = net.new_cache(n + ngen + 1);
    for (pos, &id) in ids.iter().enumerate() {
        out.push(net.forward(id, pos, &mut a));
    }
    let mut b = net.new_cache(n + ngen + 1);
    let mut s = 0;
    for &k in splits.iter().cycle() {
        if s + 1 >= n {
            break;
        }
        let k = k.min(n - 1 - s);
        net.feed_batch(&ids[s..s + k], s, &mut b);
        s += k;
    }
    let mut last = net.prefill_logits(&ids[s..], s, &mut b);
    out.push(last.clone());
    for i in 0..ngen {
        let t = crate::argmax(&last);
        last = net.forward(t, n + i, &mut b);
        out.push(last.clone());
    }
    out
}

/// PR C's key gate on the fixture: every `--gpu-experts` placement (none,
/// one MoE layer, both, auto) gives the same bits, decode and batched
/// (every option set; YaRN's crosses an attention split).
#[test]
fn ds2_resident_fixture_bits() {
    use crate::cuda::qwen2::experts::GpuExperts;
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    if crate::cuda::qwen2::experts::twin_gemm_ok(g.info().unwrap().cc).is_err() {
        // Residency falls back to none below sm_80 (and says so).
        let (m, _) = ds2_fixture(OPTS[0], crate::quant::Q8_0);
        let km = Kernels::module(&g).unwrap();
        let ml = Kernels::mla_module(&g).unwrap();
        let net = MoeQwen3::new_with(
            &g,
            &km,
            &ml,
            &m,
            64,
            CacheKv::F16,
            false,
            true,
            GpuExperts::Auto,
        )
        .unwrap();
        assert!(net.resident_layers().is_empty());
        assert!(net.describe(64).contains("sm_80"), "{}", net.describe(64));
        eprintln!("SKIPPED (fell back, as designed): {}", net.describe(64));
        return;
    }
    let km = Kernels::module(&g).unwrap();
    let ml = Kernels::mla_module(&g).unwrap();
    for o in OPTS {
        let n = if o.yarn { 150 } else { 60 };
        let mut base: Option<Vec<Vec<f32>>> = None;
        for want in [
            GpuExperts::N(0),
            GpuExperts::N(1),
            GpuExperts::N(2),
            GpuExperts::Auto,
        ] {
            let (m, _) = ds2_fixture(o, crate::quant::Q8_0);
            let net = MoeQwen3::new_with(&g, &km, &ml, &m, n + 40, CacheKv::F16, false, true, want)
                .unwrap();
            let r = net.resident_layers();
            let expect = match want {
                GpuExperts::N(k) => k,
                GpuExperts::Auto => 2,
            };
            assert_eq!(r.len(), expect, "{o:?} {want:?}: {}", net.describe(n));
            let ids = ids_for(n, net.vocab(), 5);
            let tr = resident_trace(&net, &ids, &[1, 7, 13, 2, 33], 24);
            assert!(tr.iter().flatten().all(|v| v.is_finite()));
            match &base {
                None => base = Some(tr),
                Some(b) => {
                    for (i, (x, y)) in b.iter().zip(&tr).enumerate() {
                        assert!(
                            bits(x) == bits(y),
                            "{o:?} {want:?} (resident {r:?}): logits {i} differ from none resident"
                        );
                    }
                }
            }
            eprintln!("fixture {o:?} {want:?}: resident {r:?}, bits equal");
        }
    }
}

/// PR C's key gate on Moonlight (AXON_TEST_DS2=1): `--gpu-experts` 0, 6,
/// 12 and auto give the same bits on a 400-token prompt (3 attention
/// splits; fed in passes) and a 64-token greedy continuation, and on the
/// prompt's first 40 positions decoded one by one.
#[test]
fn ds2_resident_moonlight_bits() {
    use crate::cuda::qwen2::experts::GpuExperts;
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::ds2_model() else {
        return;
    };
    if crate::cuda::qwen2::experts::twin_gemm_ok(g.info().unwrap().cc).is_err() {
        eprintln!("SKIPPED: resident experts need sm_80+");
        return;
    }
    let km = Kernels::module_for(&g, &m).unwrap();
    let ml = Kernels::mla_module(&g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let text = "Mixture-of-experts models route each token to a few experts. ".repeat(40);
    let mut ids = tok.encode(&text);
    ids.truncate(400);
    type Logits = Vec<Vec<f32>>;
    let mut base: Option<(Logits, Logits)> = None;
    for want in [
        GpuExperts::N(0),
        GpuExperts::N(6),
        GpuExperts::N(12),
        GpuExperts::Auto,
    ] {
        let net =
            MoeQwen3::new_with(&g, &km, &ml, &m, 512, CacheKv::F16, false, true, want).unwrap();
        let r = net.resident_layers();
        eprintln!("moonlight {want:?}: {}", net.describe(512));
        if let GpuExperts::N(k) = want {
            assert!(r.len() <= k);
        }
        // The prompt fed (passes of 100, 37, 263) + the continuation.
        let fed = {
            let mut c = net.new_cache(512);
            net.feed_batch(&ids[..100], 0, &mut c);
            net.feed_batch(&ids[100..137], 100, &mut c);
            let mut last = net.prefill_logits(&ids[137..], 137, &mut c);
            let mut out = vec![last.clone()];
            for i in 0..64 {
                let t = crate::argmax(&last);
                last = net.forward(t, 400 + i, &mut c);
                out.push(last.clone());
            }
            out
        };
        let mut c = net.new_cache(64);
        let dec: Vec<Vec<f32>> = ids[..40]
            .iter()
            .enumerate()
            .map(|(p, &t)| net.forward(t, p, &mut c))
            .collect();
        match &base {
            None => base = Some((fed, dec)),
            Some((bf, bd)) => {
                for (i, (x, y)) in bf.iter().zip(&fed).enumerate() {
                    assert!(
                        bits(x) == bits(y),
                        "{want:?} (resident {r:?}): fed logits {i} differ"
                    );
                }
                for (i, (x, y)) in bd.iter().zip(&dec).enumerate() {
                    assert!(
                        bits(x) == bits(y),
                        "{want:?} (resident {r:?}): decode logits {i} differ"
                    );
                }
            }
        }
        eprintln!("moonlight {want:?}: resident {r:?}: bits equal");
    }
}
