//! gpt-oss's experts resident on the GPU (`--gpu-experts`,
//! specs/gptoss-gpu-experts.md stage B): oss.cu's MXFP4 twins (bias
//! epilogues, `swiglu_oai`) give the CPU experts' bits, so placement never
//! changes an output (#120's rule) and decode == batched prefill == verify
//! holds at every placement. The fixture (3 layers, MXFP4, biases) runs on
//! any GPU: MXFP4's batched twin is dp4a (sm_61+); the real model where it
//! is (AXON_TEST_GPTOSS).

use super::ds2_tests::resident_trace;
use super::oss_tests::{GPU_OSS, contract, ids_mod, paged};
use super::q3res_tests::{assert_released, bits, check_verify};
use super::{TwinEnv, *};
use crate::cuda::Kernels;
use crate::cuda::qwen2::experts::GpuExperts;
use crate::model::gptoss_tests::oss_fixture_dims;

fn fixture() -> Model {
    oss_fixture_dims(GPU_OSS, |_| {}, |_| {})
}

fn net<'m, 'g>(
    g: &'g Gpu,
    km: &'g Module<'g>,
    mm: &'g Module<'g>,
    m: &'m Model,
    ctx: usize,
    want: GpuExperts,
) -> MoeQwen3<'m, 'g> {
    MoeQwen3::new_with(g, km, mm, m, ctx, CacheKv::F16, false, true, want).unwrap()
}

/// The fixture: N(0), N(1), every layer and auto give the same logits bits,
/// fed in odd passes and decoded, plus a greedy continuation; verify picks
/// decode's argmax; the resident layers' host bytes are released.
#[test]
fn oss_resident_fixture_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let g = &g;
    let m0 = fixture();
    let n_moe = expert_layers(&m0).len();
    assert_eq!(n_moe, 3);
    let km = Kernels::module_for(g, &m0).unwrap();
    let mm = Kernels::moe_module_for(g, &m0).unwrap();
    let v = GPU_OSS.8 as u64;
    let ids = ids_mod(60, v);
    let mut base: Option<Vec<Vec<u32>>> = None;
    for want in [
        GpuExperts::N(0),
        GpuExperts::N(1),
        GpuExperts::N(n_moe),
        GpuExperts::Auto,
    ] {
        let m = fixture();
        assert!(!m.file_backed());
        let net = net(g, &km, &mm, &m, 100, want);
        let r = net.resident_layers();
        let d = net.describe(60);
        let expect = match want {
            GpuExperts::N(k) => k,
            GpuExperts::Auto => n_moe,
        };
        assert_eq!(r.len(), expect, "{want:?}: {d}");
        assert_released(&m, &r, &format!("{want:?}"));
        let tr = resident_trace(&net, &ids, &[1, 7, 13, 2, 33], 24);
        assert!(tr.iter().flatten().all(|v| v.is_finite()));
        check_verify(&net, &ids, &tr, &format!("oss fixture {want:?}"));
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
        eprintln!("oss fixture {want:?}: {d}; resident {r:?}: bits equal");
    }
}

/// The contract (decode == batched prefill over 15 batchings == verify,
/// reused prefix == fresh, paged == plain) with every layer resident and
/// with one, on the fixture.
#[test]
fn oss_resident_contract_fixture() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let m0 = fixture();
    let km = Kernels::module_for(&g, &m0).unwrap();
    let mm = Kernels::moe_module_for(&g, &m0).unwrap();
    for want in [GpuExperts::N(1), GpuExperts::Auto] {
        let m = fixture();
        let net = net(&g, &km, &mm, &m, 1024, want);
        assert!(!net.resident_layers().is_empty(), "{}", net.describe(64));
        let v = net.vocab() as u64;
        contract(&net, &ids_mod(483, v), (9707 % v) as u32);
        paged(&g, &net, &ids_mod(300, v));
        eprintln!(
            "oss fixture contract {want:?}: resident {:?}",
            net.resident_layers()
        );
    }
}

/// MXFP4's twins don't depend on AXON_REPACK (the CPU never repacks MXFP4):
/// with it off the layers are still placed, with N(0)'s bits. A twin F
/// that isn't the host's is still refused (falls back to the CPU, N(0)'s
/// bits).
#[test]
fn oss_resident_repack_and_fused() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let g = &g;
    let m0 = fixture();
    let km = Kernels::module_for(g, &m0).unwrap();
    let mm = Kernels::moe_module_for(g, &m0).unwrap();
    let ids = ids_mod(20, GPU_OSS.8 as u64);
    let run = |net: &MoeQwen3| bits(&resident_trace(net, &ids, &[3, 7], 4));
    let base = run(&net(g, &km, &mm, &m0, 64, GpuExperts::N(0)));
    let f = crate::vmath::fused();
    for (repack, cpu_fused, placed) in [(false, f, true), (true, !f, false)] {
        let m = fixture();
        let env = TwinEnv {
            repack,
            cpu_fused,
            ram: None,
        };
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
        assert_eq!(
            net.resident_layers().len(),
            if placed { 3 } else { 0 },
            "{d}"
        );
        if !placed {
            assert!(d.contains("fused"), "{d}");
        }
        assert!(
            run(&net) == base,
            "repack {repack}, F {cpu_fused}: bits differ from N(0)"
        );
        eprintln!("oss repack {repack}, F {cpu_fused}: {d}");
    }
}

/// The real model (AXON_TEST_GPTOSS), loaded afresh per placement: none,
/// 6 layers, auto. Bits equal to N(0) over a ~120-token prompt fed in passes
/// and decoded one by one, plus 24 generated tokens; verify picks decode's
/// argmax; with every placement the contract.
#[test]
fn oss_resident_model_bits() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let g = &g;
    let Some(path) = crate::testutil::gptoss_model_path() else {
        return;
    };
    let p = path.to_str().unwrap();
    let m0 = crate::gguf::load(p).unwrap();
    let km = Kernels::module_for(g, &m0).unwrap();
    let mm = Kernels::moe_module_for(g, &m0).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m0);
    let mut ids =
        tok.encode(&"Mixture-of-experts models route each token to a few experts. ".repeat(12));
    ids.truncate(120);
    drop(m0);
    let mut base: Option<Vec<Vec<u32>>> = None;
    for want in [GpuExperts::N(0), GpuExperts::N(6), GpuExperts::Auto] {
        let m = crate::gguf::load(p).unwrap();
        let net = net(g, &km, &mm, &m, 1024, want);
        let r = net.resident_layers();
        let d = net.describe(512);
        match want {
            GpuExperts::N(k) => assert_eq!(r.len(), k, "{want:?}: {d}"),
            GpuExperts::Auto => assert!(!r.is_empty(), "auto: {d}"),
        }
        let tr = resident_trace(&net, &ids, &[1, 7, 13, 2, 33], 24);
        check_verify(&net, &ids, &tr, &format!("gpt-oss {want:?}"));
        let b = bits(&tr);
        match &base {
            None => base = Some(b),
            Some(x) => {
                for (i, (p0, q)) in x.iter().zip(&b).enumerate() {
                    assert!(
                        p0 == q,
                        "{want:?} (resident {r:?}): logits {i} differ from N(0)"
                    );
                }
            }
        }
        let v = net.vocab() as u64;
        contract(&net, &ids_mod(483, v), (9707 % v) as u32);
        eprintln!("gpt-oss {want:?}: {d}; resident {r:?}: bits equal, contract holds");
    }
}
