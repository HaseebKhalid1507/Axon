//! qwen3moe's routed experts resident on the GPU (`--gpu-experts`, PR E):
//! the same twins as deepseek2's (`cuda::qwen2::experts`), so every
//! placement gives the CPU experts' bits, decode == batched prefill ==
//! verify (picks). On the synthetic fixture (Q8_0, in memory: a released tensor
//! panics on read) and the two real test models (Qwen3-MOE-4x0.6B: Q6_K
//! and Q4_K down; the pruned 30B-A3B: the 30B's shapes, 128 experts top-8).

use super::ds2_tests::resident_trace;
use super::*;
use crate::cuda::Kernels;
use crate::cuda::qwen2::experts::GpuExperts;

fn bits(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter()
        .map(|x| x.iter().map(|f| f.to_bits()).collect())
        .collect()
}

/// The GPU if it runs the resident experts (sm_80+), else None after a
/// visible skip.
fn gpu_mma() -> Option<Gpu> {
    let g = crate::testutil::gpu()?;
    if let Err(e) = crate::cuda::qwen2::experts::twin_gemm_ok(g.info().unwrap().cc) {
        eprintln!("SKIPPED: resident experts: {e}");
        return None;
    }
    Some(g)
}

/// Every expert tensor of the resident layers `r` reads as released (`m`
/// anonymous memory: `tensor_data` panics on a released tensor).
fn assert_released(m: &Model, r: &[usize], what: &str) {
    for t in m.tensors.iter().filter(|t| {
        crate::model::MOE_EXPS.iter().any(|s| t.name.ends_with(s))
            && usize::try_from(t.layer).is_ok_and(|l| r.contains(&l))
    }) {
        let read =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.tensor_data(t).len()));
        assert!(read.is_err(), "{what}: {} not released", t.name);
    }
}

/// verify (`verify_argmax`: the batched pass with `Head::Argmax`) picks the
/// argmax of each decode logits row in `tr` (its first ids.len() rows).
fn check_verify(net: &MoeQwen3, ids: &[u32], tr: &[Vec<f32>], what: &str) {
    let mut c = net.new_cache(ids.len() + 1);
    let picks = net.verify_argmax(ids, 0, &mut c);
    let want: Vec<u32> = tr[..ids.len()].iter().map(|l| crate::argmax(l)).collect();
    assert_eq!(picks, want, "{what}: verify != decode argmax");
}

/// The fixture: N(0), N(1), every layer and auto give the same bits, decode
/// and fed in odd passes, plus a greedy continuation; the resident layers'
/// host bytes are released and never read.
#[test]
fn q3_resident_fixture_bits() {
    let Some(g) = gpu_mma() else {
        return;
    };
    let g = &g;
    let (m0, _) = crate::model::tests::dense_fixture("qwen3moe", crate::quant::Q8_0);
    let n_moe = expert_layers(&m0).len();
    assert!(n_moe >= 2, "the fixture has {n_moe} MoE layers");
    let km = Kernels::module_for(g, &m0).unwrap();
    let mm = Kernels::moe_module_for(g, &m0).unwrap();
    let n = 60;
    let ids: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % 50 + 1).collect();
    let mut base: Option<Vec<Vec<u32>>> = None;
    for want in [
        GpuExperts::N(0),
        GpuExperts::N(1),
        GpuExperts::N(n_moe),
        GpuExperts::Auto,
    ] {
        let (m, _) = crate::model::tests::dense_fixture("qwen3moe", crate::quant::Q8_0);
        assert!(!m.file_backed());
        let net =
            MoeQwen3::new_with(g, &km, &mm, &m, n + 40, CacheKv::F16, false, true, want).unwrap();
        let r = net.resident_layers();
        let expect = match want {
            GpuExperts::N(k) => k,
            GpuExperts::Auto => n_moe,
        };
        assert_eq!(r.len(), expect, "{want:?}: {}", net.describe(n));
        assert_released(&m, &r, &format!("{want:?}"));
        let tr = resident_trace(&net, &ids, &[1, 7, 13, 2, 33], 24);
        assert!(tr.iter().flatten().all(|v| v.is_finite()));
        check_verify(&net, &ids, &tr, &format!("fixture {want:?}"));
        let b = bits(&tr);
        match &base {
            None => base = Some(b),
            Some(x) => {
                for (i, (p, q)) in x.iter().zip(&b).enumerate() {
                    assert!(
                        p == q,
                        "{want:?} (resident {r:?}): logits {i} differ from N(0)"
                    );
                }
            }
        }
        eprintln!("qwen3moe fixture {want:?}: resident {r:?}, bits equal");
    }
}

/// The real test models (each loaded afresh per placement, mapped, under
/// the strict release guard: `gguf::load`): none, about half the MoE
/// layers, auto. N > 0 must place N (or all that fit, said), auto at
/// least one; bits equal to N(0) over a ~120-token prompt fed in passes
/// and decoded one by one, plus 24 generated tokens.
#[test]
fn q3_resident_model_bits() {
    let Some(g) = gpu_mma() else {
        return;
    };
    let g = &g;
    for small in [true, false] {
        let Some(path) = crate::testutil::moe_model_path(small) else {
            eprintln!(
                "SKIPPED: q3_resident_model_bits: the {} qwen3moe test model is absent",
                if small { "4x0.6B" } else { "pruned 30B-A3B" }
            );
            continue;
        };
        let p = path.to_str().unwrap();
        let m0 = crate::gguf::load(p).unwrap();
        let km = Kernels::module_for(g, &m0).unwrap();
        let mm = Kernels::moe_module_for(g, &m0).unwrap();
        let tok = crate::tokenizer::Tokenizer::from_gguf(&m0);
        let mut ids =
            tok.encode(&"Mixture-of-experts models route each token to a few experts. ".repeat(12));
        ids.truncate(120);
        let n_moe = expert_layers(&m0).len();
        let half = n_moe.div_ceil(2);
        drop(m0);
        let mut base: Option<Vec<Vec<u32>>> = None;
        for want in [GpuExperts::N(0), GpuExperts::N(half), GpuExperts::Auto] {
            let m = crate::gguf::load(p).unwrap();
            let net =
                MoeQwen3::new_with(g, &km, &mm, &m, 512, CacheKv::F16, false, true, want).unwrap();
            let r = net.resident_layers();
            let d = net.describe(512);
            match want {
                GpuExperts::N(k) => assert_eq!(r.len(), k, "{p} {want:?}: {d}"),
                GpuExperts::Auto => assert!(!r.is_empty(), "{p} auto: {d}"),
            }
            let tr = resident_trace(&net, &ids, &[1, 7, 13, 2, 33], 24);
            check_verify(&net, &ids, &tr, &format!("{p} {want:?}"));
            let b = bits(&tr);
            match &base {
                None => base = Some(b),
                Some(x) => {
                    for (i, (p0, q)) in x.iter().zip(&b).enumerate() {
                        assert!(
                            p0 == q,
                            "{p} {want:?} (resident {r:?}): logits {i} differ from N(0)"
                        );
                    }
                }
            }
            eprintln!(
                "{} {want:?}: {d}; resident {r:?}: bits equal",
                path.display()
            );
        }
    }
}

/// The refusals still fall back to CPU experts on qwen3moe with N(0)'s
/// bits: AXON_REPACK=0 and a twin F that isn't the host's (forced through
/// `new_inner`); gpt-oss (MXFP4 experts, expert biases) is refused.
#[test]
fn q3_resident_fallbacks() {
    use super::TwinEnv;
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let g = &g;
    let (m0, _) = crate::model::tests::dense_fixture("qwen3moe", crate::quant::Q8_0);
    let km = Kernels::module_for(g, &m0).unwrap();
    let mm = Kernels::moe_module_for(g, &m0).unwrap();
    let ids: Vec<u32> = (0..20u32).map(|i| (i * 5 + 1) % 50 + 1).collect();
    let run = |net: &MoeQwen3| bits(&resident_trace(net, &ids, &[3, 7], 4));
    let base = {
        let net = MoeQwen3::new_with(
            g,
            &km,
            &mm,
            &m0,
            64,
            CacheKv::F16,
            false,
            true,
            GpuExperts::N(0),
        )
        .unwrap();
        run(&net)
    };
    let f = crate::vmath::fused();
    for (env, why) in [
        (
            TwinEnv {
                repack: false,
                cpu_fused: f,
                ram: None,
            },
            "AXON_REPACK=0",
        ),
        (
            TwinEnv {
                repack: true,
                cpu_fused: !f,
                ram: None,
            },
            "fused",
        ),
    ] {
        let (m, _) = crate::model::tests::dense_fixture("qwen3moe", crate::quant::Q8_0);
        let net = MoeQwen3::new_inner(
            g,
            &km,
            &mm,
            &m,
            64,
            CacheKv::F16,
            false,
            true,
            GpuExperts::Auto,
            env,
        )
        .unwrap();
        let d = net.describe(64);
        assert!(
            net.resident_layers().is_empty() && d.contains(why),
            "{why}: {d}"
        );
        assert!(run(&net) == base, "{why}: bits differ from N(0)");
        eprintln!("qwen3moe fallback ({why}): {d}");
    }
    let o = crate::model::gptoss_tests::oss_fixture_dims(
        (3, 256, 16, 2, 64, 64, 4, 2, 64, 4),
        |_| {},
        |_| {},
    );
    let ok = Kernels::module_for(g, &o).unwrap();
    let om = Kernels::moe_module_for(g, &o).unwrap();
    let net = MoeQwen3::new_with(
        g,
        &ok,
        &om,
        &o,
        32,
        CacheKv::F16,
        false,
        true,
        GpuExperts::Auto,
    )
    .unwrap();
    let d = net.describe(32);
    assert!(
        net.resident_layers().is_empty() && d.contains("deepseek2 and qwen3moe only"),
        "{d}"
    );
    eprintln!("gpt-oss fallback: {d}");
}
