//! `CudaExperts` == the CPU's `moe_layer_batch`, bit for bit (specs r16 pick
//! 4, PR B gates): on the deepseek2 fixture and, with AXON_TEST_DS2, on three
//! real Moonlight layers (one with Q8_0 down, one with Q5_0 down, and the last
//! layer), at nb 1, 6, 48, 510 and 1024; and nb invariance (each token alone
//! through the decode GEMV == its column of the batch).

use super::*;
use crate::cuda::Kernels;
use crate::model::Qwen2;
use crate::model::deepseek2_tests::{Opt, ds2_fixture};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// Bits of a float, every NaN as one value.
fn key(v: f32) -> u32 {
    if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() }
}

/// Positions where a and b differ in `key`: the count and the first few.
fn diffs(a: &[f32], b: &[f32]) -> (usize, Vec<(usize, f32, f32)>) {
    assert_eq!(a.len(), b.len());
    let d: Vec<usize> = (0..a.len()).filter(|&i| key(a[i]) != key(b[i])).collect();
    (
        d.len(),
        d.iter().take(4).map(|&i| (i, a[i], b[i])).collect(),
    )
}

/// The host-side refusals (#121 items 1, 2): no GPU needed.
#[test]
fn twin_refusals() {
    for cc in [(6, 1), (7, 0), (7, 5)] {
        let e = twin_gemm_ok(cc).unwrap_err();
        assert!(e.contains("sm_80"), "{e}");
    }
    for cc in [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)] {
        twin_gemm_ok(cc).unwrap();
    }
    use crate::quant::*;
    for qt in [Q4_K, Q5_0, Q8_0, Q6_K] {
        twin_type(qt).unwrap();
    }
    for qt in [Q4_0, Q4_1, Q5_1, Q2_K, Q3_K, Q5_K, IQ4_NL, F16, F32] {
        assert!(twin_type(qt).is_err(), "type {qt} must be refused");
    }
}

/// The batch sizes of the gate.
const NBS: [usize; 5] = [1, 6, 48, 510, 1024];

/// Layer l of `m` on the GPU vs `cpu.moe_layer_batch` at every nb of NBS:
/// nb random hidden states (every third token 8x larger), routed by the CPU's
/// router; then nb invariance on the largest batch (the decode GEMV, token by
/// token). Below sm_80 the batched path must refuse, and the GEMV path is
/// checked against the CPU instead.
fn check_layer(g: &Gpu, ex: &CudaExperts, cpu: &Qwen2, l: usize, what: &str) {
    let mut rng = Rng(0x6d6f_6572_6573_0000 ^ l as u64);
    let (dim, k) = (cpu.dim, ex.k);
    let mma = twin_gemm_ok(g.info().unwrap().cc).is_ok();
    assert_eq!(ex.fused(), crate::vmath::fused(), "one F: the host CPU's");
    let mut used = vec![0usize; ex.n_e];
    for nb in NBS {
        let h: Vec<f32> = (0..nb * dim)
            .map(|i| rng.f() * if (i / dim) % 3 == 0 { 8.0 } else { 1.0 })
            .collect();
        let hs: Vec<&[f32]> = h.chunks_exact(dim).collect();
        let routes: Vec<_> = hs.iter().map(|x| cpu.moe_route(l, x, false)).collect();
        for (ids, _) in &routes {
            for &e in ids {
                used[e] += 1;
            }
        }
        let ids: Vec<i32> = routes
            .iter()
            .flat_map(|r| r.0.iter().map(|&e| e as i32))
            .collect();
        let ws: Vec<f32> = routes.iter().flat_map(|r| r.1.iter().copied()).collect();
        // CPU: (row, token); to token major.
        let yc = cpu.moe_layer_batch(l, &hs, &routes);
        let cpu_t: Vec<f32> = (0..nb * dim)
            .map(|i| yc[(i % dim) * nb + i / dim])
            .collect();
        assert!(
            cpu_t.iter().all(|v| v.is_finite()),
            "{what} layer {l} nb {nb}: CPU output not finite"
        );
        let got = if nb > 1 && !mma {
            let e = ex.layer(&h, &ids, &ws).unwrap_err();
            assert!(e.contains("sm_80"), "{what}: refusal says why: {e}");
            ex.layer_gemv(&h, &ids, &ws).unwrap()
        } else {
            ex.layer(&h, &ids, &ws).unwrap()
        };
        let (n, first) = diffs(&got, &cpu_t);
        assert_eq!(
            n,
            0,
            "{what} layer {l} nb {nb}: {n} of {} outputs differ from the CPU, e.g. {first:?}",
            nb * dim
        );
        if nb == 1 && mma {
            // `layer` sends nb 1 to the decode GEMV: the grouped kernels
            // (moe_sort, moe_gemm_tw_*, moe_sum) at one token, forced.
            let (b1, _) = ex.batched(&h, &ids, &ws, 1).unwrap();
            let (n, first) = diffs(&b1, &cpu_t);
            assert_eq!(
                n, 0,
                "{what} layer {l}: batched path at nb 1 differs: {first:?}"
            );
        }
        if nb == 1024 && mma {
            // nb invariance: tokens alone (decode GEMV) == their batch columns.
            let pick: Vec<usize> = (0..8).chain([511, 512, 1023]).collect();
            let hp: Vec<f32> = pick
                .iter()
                .flat_map(|&t| h[t * dim..(t + 1) * dim].iter().copied())
                .collect();
            let ip: Vec<i32> = pick
                .iter()
                .flat_map(|&t| ids[t * k..(t + 1) * k].iter().copied())
                .collect();
            let wp: Vec<f32> = pick
                .iter()
                .flat_map(|&t| ws[t * k..(t + 1) * k].iter().copied())
                .collect();
            let alone = ex.layer_gemv(&hp, &ip, &wp).unwrap();
            for (j, &t) in pick.iter().enumerate() {
                let (n, first) =
                    diffs(&alone[j * dim..(j + 1) * dim], &got[t * dim..(t + 1) * dim]);
                assert_eq!(
                    n, 0,
                    "{what} layer {l}: token {t} alone != in the batch of 1024: {first:?}"
                );
            }
        }
        eprintln!(
            "{what} layer {l} nb {nb}: {} outputs equal the CPU",
            nb * dim
        );
    }
    let unused = used.iter().filter(|&&c| c == 0).count();
    eprintln!(
        "{what} layer {l}: types gate {} / down {}, {} experts, k {k}, {unused} never routed",
        ex.gate.qt, ex.down.qt, ex.n_e
    );
}

/// Layers `layers` of `m` uploaded (before the CPU's repack, which may
/// release an in-memory model's bytes), k from the metadata.
fn upload_all<'g>(
    g: &'g Gpu,
    km: &'g Module<'g>,
    m: &Model,
    layers: &[usize],
) -> Vec<CudaExperts<'g>> {
    let key = format!("{}.expert_used_count", m.arch);
    let k = crate::model::meta_num(m, &key).expect("expert_used_count") as usize;
    layers
        .iter()
        .map(|&l| CudaExperts::upload(g, km, m, l, k).unwrap())
        .collect()
}

/// qwen3moe's Q4_K_M experts, Q6_K down included (#122 review): the
/// pruned Qwen3-30B-A3B (128 experts, top-8, the 30B's shapes; layer 2's
/// down Q6_K), all 3 layers; and Qwen3-MOE-4x0.6B (4 experts, top-2) layer
/// 0 (Q6_K down) and 3 (Q4_K down). Skipped if the files are missing.
#[test]
fn cuda_experts_qwen3moe_match_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    for (small, layers) in [(false, vec![0, 1, 2]), (true, vec![0, 3])] {
        let Some(m) = crate::testutil::moe_model(small) else {
            continue;
        };
        let km = Kernels::module_for(&g, &m).unwrap();
        let exs = upload_all(&g, &km, &m, &layers);
        let cpu = Qwen2::new_experts(&m);
        for (ex, &l) in exs.iter().zip(&layers) {
            check_layer(
                &g,
                ex,
                &cpu,
                l,
                if small {
                    "qwen3moe-4x0.6b"
                } else {
                    "qwen3moe-pruned"
                },
            );
        }
    }
}

/// The fixture (deepseek2, Q8_0 experts, 4 experts, k 2): layers 1 and 2.
#[test]
fn cuda_experts_fixture_match_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let (m, _) = ds2_fixture(
        Opt {
            q_lora: false,
            yarn: false,
        },
        crate::quant::Q8_0,
    );
    let km = Kernels::module_for(&g, &m).unwrap();
    let exs = upload_all(&g, &km, &m, &[1, 2]);
    let cpu = Qwen2::new_experts(&m);
    for (ex, l) in exs.iter().zip([1, 2]) {
        check_layer(&g, ex, &cpu, l, "fixture");
    }
}

/// Moonlight (AXON_TEST_DS2): the first layer with Q8_0 down, the first with
/// Q5_0 down, and the last layer; only these layers' experts repacked.
#[test]
fn cuda_experts_moonlight_match_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::ds2_model() else {
        return;
    };
    let down = |l: usize| {
        m.tensors
            .iter()
            .find(|t| t.name == format!("blk.{l}.ffn_down_exps.weight"))
            .map(|t| t.qtype.id)
    };
    let n_layers = (0..)
        .take_while(|&l| {
            m.tensors
                .iter()
                .any(|t| t.name.starts_with(&format!("blk.{l}.")))
        })
        .count();
    let first = |qt: u32| (0..n_layers).find(|&l| down(l) == Some(qt));
    let mut layers: Vec<usize> = [first(crate::quant::Q8_0), first(crate::quant::Q5_0)]
        .into_iter()
        .flatten()
        .collect();
    layers.push(n_layers - 1);
    layers.dedup();
    assert!(
        layers.len() == 3,
        "Moonlight layers {layers:?}: expected a Q8_0 and a Q5_0 down"
    );
    let km = Kernels::module_for(&g, &m).unwrap();
    let exs = upload_all(&g, &km, &m, &layers);
    let cpu = Qwen2::new_experts_of(&m, &layers);
    for (ex, &l) in exs.iter().zip(&layers) {
        check_layer(&g, ex, &cpu, l, "moonlight");
    }
}

/// PR B's speed gate (spec: per-layer GPU time at 510 tokens <= 5 ms on
/// bella): Moonlight's first Q8_0-down and Q5_0-down layers at nb 48, 510
/// and 1024, the batched kernels' GPU time (events, 3 warm-up runs, then the
/// median of 5 runs of 10) and the CPU's `moe_layer_batch` wall time (median
/// of 3). AXON_TEST_DS2=1; run exclusive (`--timing`).
#[test]
#[ignore]
fn cuda_experts_moonlight_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::ds2_model() else {
        return;
    };
    if g.info().unwrap().cc.0 < 8 {
        eprintln!("skipping: the batched path needs sm_80+");
        return;
    }
    let down = |l: usize| {
        m.tensors
            .iter()
            .find(|t| t.name == format!("blk.{l}.ffn_down_exps.weight"))
            .map(|t| t.qtype.id)
    };
    let layers: Vec<usize> = [crate::quant::Q8_0, crate::quant::Q5_0]
        .iter()
        .filter_map(|&qt| (0..64).find(|&l| down(l) == Some(qt)))
        .collect();
    let km = Kernels::module_for(&g, &m).unwrap();
    let exs = upload_all(&g, &km, &m, &layers);
    let cpu = Qwen2::new_experts_of(&m, &layers);
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    for (ex, &l) in exs.iter().zip(&layers) {
        let mut rng = Rng(0xbe9c ^ l as u64);
        for nb in [48, 510, 1024] {
            let h: Vec<f32> = (0..nb * cpu.dim).map(|_| rng.f()).collect();
            let hs: Vec<&[f32]> = h.chunks_exact(cpu.dim).collect();
            let routes: Vec<_> = hs.iter().map(|x| cpu.moe_route(l, x, false)).collect();
            let ids: Vec<i32> = routes
                .iter()
                .flat_map(|r| r.0.iter().map(|&e| e as i32))
                .collect();
            let ws: Vec<f32> = routes.iter().flat_map(|r| r.1.iter().copied()).collect();
            ex.batched(&h, &ids, &ws, 3).unwrap();
            let gpu = med((0..5)
                .map(|_| ex.batched(&h, &ids, &ws, 10).unwrap().1 as f64)
                .collect());
            let cpu_ms = med((0..3)
                .map(|_| {
                    let t = std::time::Instant::now();
                    std::hint::black_box(cpu.moe_layer_batch(l, &hs, &routes));
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect());
            eprintln!(
                "BENCH moonlight layer {l} (down type {}) nb {nb}: gpu {gpu:.3} ms, cpu {cpu_ms:.2} ms, \
                 x{:.1}",
                ex.down.qt,
                cpu_ms / gpu
            );
        }
    }
}

/// `place`: smallest first (ties to the lower layer), as many as asked and
/// fit, sorted by layer; `GpuExperts::parse`; `twin_identity` (#121 item 5).
#[test]
fn resident_placement_and_identity() {
    let ls = [(1, 400), (2, 334), (3, 334), (4, 400), (5, 334)];
    assert_eq!(
        place(&ls, GpuExperts::N(0), usize::MAX),
        Vec::<usize>::new()
    );
    assert_eq!(place(&ls, GpuExperts::N(2), usize::MAX), vec![2, 3]);
    assert_eq!(place(&ls, GpuExperts::N(4), usize::MAX), vec![1, 2, 3, 5]);
    assert_eq!(
        place(&ls, GpuExperts::Auto, usize::MAX),
        vec![1, 2, 3, 4, 5]
    );
    assert_eq!(place(&ls, GpuExperts::Auto, 1100), vec![2, 3, 5]);
    assert_eq!(place(&ls, GpuExperts::Auto, 1001), vec![2, 3]);
    assert_eq!(place(&ls, GpuExperts::N(9), 1500), vec![1, 2, 3, 5]);
    assert_eq!(place(&ls, GpuExperts::Auto, 100), Vec::<usize>::new());
    assert_eq!(GpuExperts::parse("auto"), Ok(GpuExperts::Auto));
    assert_eq!(GpuExperts::parse("12"), Ok(GpuExperts::N(12)));
    assert!(GpuExperts::parse("-1").is_err() && GpuExperts::parse("all").is_err());
    twin_identity(true, true, true).unwrap();
    twin_identity(true, false, false).unwrap();
    assert!(
        twin_identity(false, true, true)
            .unwrap_err()
            .contains("AXON_REPACK=0")
    );
    assert!(
        twin_identity(true, true, false)
            .unwrap_err()
            .contains("fused")
    );
    assert!(twin_identity(true, false, true).is_err());
}
