//! #99's regression guard: `moe_router` (the production modules, launched by
//! `launch_router` with the `GATE_*` constants) against main 764792f's three
//! router kernels (`tests/fixtures/moe_router_legacy.cu`: moe_topk,
//! moe_topk_oai, mla_topk), ids and weight bits, on adversarial logits.
//! Ignored by default (a GPU and the legacy kernels' NVRTC build); run with
//! `--ignored router_bits_match_legacy` on a release build (debug builds
//! compile MOE_DEBUG, whose guard traps on the NaN rows).

use crate::cuda::qwen2::moe::{GATE_OAI, GATE_SIGMOID, GATE_SOFTMAX, launch_router};
use crate::cuda::{Arg, Gpu, KERNEL_OPTS, Kernels, Module};

const LEGACY: &str = include_str!("../../tests/fixtures/moe_router_legacy.cu");

struct Rng(u64);
impl Rng {
    /// Uniform in [-1, 1).
    fn u(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
    }
    /// Uniform in [0, 1).
    fn p(&mut self) -> f32 {
        (self.u() + 1.0) / 2.0
    }
}

/// Row categories below `SAFE` are finite or -inf with at least one finite
/// logit (qwen3moe's softmax stays defined: the old kernel had no guard).
const SAFE: usize = 8;
const CATEGORIES: usize = 13;

/// A row of n_e logits of category c.
fn row(r: &mut Rng, c: usize, n_e: usize, k: usize) -> Vec<f32> {
    let ni = f32::NEG_INFINITY;
    match c {
        0 => (0..n_e).map(|_| 4.0 * r.u()).collect(),
        1 => (0..n_e).map(|_| 50.0 * r.u()).collect(), // saturated sigmoid: ties
        2 => (0..n_e).map(|_| (r.u() * 4.0).round() / 4.0).collect(), // exact ties
        3 => vec![0.37; n_e],
        4 => (0..n_e).map(|_| 1e3 * r.u()).collect(),
        5 => {
            let mut v: Vec<f32> = (0..n_e)
                .map(|_| if r.p() < 0.3 { ni } else { 4.0 * r.u() })
                .collect();
            v[(r.p() * n_e as f32) as usize % n_e] = 1.5;
            v
        }
        6 => (0..n_e).map(|_| 1e-30 * r.u()).collect(),
        7 => (0..n_e).map(|_| 88.0 + 4.0 * r.u()).collect(),
        // Not SAFE: NaN / +inf (gpt-oss and deepseek2 only).
        8 => (0..n_e)
            .map(|_| match r.p() {
                q if q < 0.1 => ni,
                q if q < 0.15 => f32::INFINITY,
                q if q < 0.2 => f32::NAN,
                _ => 4.0 * r.u(),
            })
            .collect(),
        9 => vec![f32::NAN; n_e],
        10 => (0..n_e)
            .map(|e| if e < k / 2 { r.u() } else { f32::NAN })
            .collect(),
        11 => vec![ni; n_e],
        _ => (0..n_e)
            .map(|e| if e % 3 == 0 { f32::INFINITY } else { r.u() })
            .collect(),
    }
}

/// A router's output: ids and weights, [nb][k].
type Out = (Vec<i32>, Vec<f32>);

/// One launch of `name` from `m` over the rows `lg` (n_e each), `smem`
/// floats of dynamic shared memory, args (logits, ids, w) -> the list.
#[allow(clippy::too_many_arguments)]
fn legacy(
    g: &Gpu,
    m: &Module,
    name: &str,
    lg: &[f32],
    (n_e, k): (usize, usize),
    smem: usize,
    args: &dyn Fn(Arg, Arg, Arg) -> Vec<Arg>,
) -> Out {
    let f = m.function(name).unwrap();
    let nb = lg.len() / n_e;
    let ld = g.upload(lg).unwrap();
    let (ids, w) = (g.alloc(nb * k * 4).unwrap(), g.alloc(nb * k * 4).unwrap());
    let a = args(ld.arg(), ids.arg(), w.arg());
    // SAFETY: the legacy kernels' signatures; buffers sized for nb tokens.
    unsafe {
        g.launch(&f, (nb as u32, 1, 1), (128, 1, 1), (smem * 4) as u32, &a)
            .unwrap()
    };
    g.sync().unwrap();
    (ids.read().unwrap(), w.read().unwrap())
}

/// `moe_router` from `m` through `launch_router`.
#[allow(clippy::too_many_arguments)]
fn router(
    g: &Gpu,
    m: &Module,
    lg: &[f32],
    (n_e, k): (usize, usize),
    gate: i32,
    bias: Arg,
    norm: bool,
    scale: f32,
) -> Out {
    let f = m.function("moe_router").unwrap();
    let nb = lg.len() / n_e;
    let ld = g.upload(lg).unwrap();
    let (ids, w) = (g.alloc(nb * k * 4).unwrap(), g.alloc(nb * k * 4).unwrap());
    // SAFETY: buffers sized for nb tokens; bias n_e floats or null.
    unsafe {
        launch_router(
            g,
            &f,
            ld.arg(),
            (n_e, k, nb),
            gate,
            bias,
            norm,
            scale,
            (ids.arg(), w.arg()),
        )
        .unwrap()
    };
    g.sync().unwrap();
    (ids.read().unwrap(), w.read().unwrap())
}

/// Slots whose id or weight bits differ (the first few told).
fn differ(what: &str, a: &Out, b: &Out) -> usize {
    let mut n = 0;
    for i in 0..a.0.len() {
        if a.0[i] != b.0[i] || a.1[i].to_bits() != b.1[i].to_bits() {
            if n < 5 {
                eprintln!(
                    "{what}: slot {i}: legacy ({}, {:#x}) moe_router ({}, {:#x})",
                    a.0[i],
                    a.1[i].to_bits(),
                    b.0[i],
                    b.1[i].to_bits()
                );
            }
            n += 1;
        }
    }
    n
}

#[test]
#[ignore = "GPU + legacy kernels; release builds: --ignored router_bits_match_legacy"]
fn router_bits_match_legacy() {
    if cfg!(debug_assertions) {
        eprintln!("skipping: MOE_DEBUG traps on the guard rows (release builds only)");
        return;
    }
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let src = format!("{}{LEGACY}", include_str!("../cuda/moe.cu"));
    let lm = Module::load(&g, &src, &KERNEL_OPTS).unwrap().0;
    let mm = Kernels::moe_module(&g).unwrap();
    let ml = Kernels::mla_module(&g).unwrap();
    let mut r = Rng(0x1234_5678_9abc_def1);
    let (mut configs, mut slots, mut bad) = (0usize, 0usize, 0usize);
    let null = Arg::Ptr(0);
    let shapes = [
        (4, 1),
        (4, 2),
        (8, 2),
        (32, 4),
        (60, 4),
        (64, 6),
        (128, 8),
        (200, 6),
        (256, 8),
        (384, 8),
    ];
    for (n_e, k) in shapes {
        let (nb, sh, (ne, ki)) = (2048, (n_e, k), (n_e as i32, k as i32));
        let all: Vec<f32> = (0..nb)
            .flat_map(|t| row(&mut r, t % CATEGORIES, n_e, k))
            .collect();
        let safe: Vec<f32> = (0..nb)
            .flat_map(|t| row(&mut r, t % SAFE, n_e, k))
            .collect();
        let mut check = |what: String, a: Out, b: Out| {
            bad += differ(&what, &a, &b);
            configs += 1;
            slots += a.0.len();
        };
        // qwen3moe: moe_topk vs softmax, no bias, norm, scale 1.
        let a = legacy(&g, &lm, "moe_topk", &safe, sh, n_e, &|l, i, w| {
            vec![l, Arg::I32(ne), Arg::I32(ki), i, w]
        });
        let b = router(&g, &mm, &safe, sh, GATE_SOFTMAX, null, true, 1.0);
        check(format!("qwen3 {n_e}/{k}"), a, b);
        // gpt-oss: moe_topk_oai vs OAI, biases plain, zero and special.
        for bk in 0..3 {
            let bias: Vec<f32> = (0..n_e)
                .map(|e| match bk {
                    0 => r.u(),
                    1 => 0.0,
                    _ => [f32::NEG_INFINITY, f32::NAN, f32::INFINITY, 0.5, -0.25][e % 5],
                })
                .collect();
            let bd = g.upload(&bias).unwrap();
            let ba = bd.arg();
            let a = legacy(&g, &lm, "moe_topk_oai", &all, sh, n_e, &|l, i, w| {
                vec![l, ba, Arg::I32(ne), Arg::I32(ki), i, w]
            });
            let b = router(&g, &mm, &all, sh, GATE_OAI, ba, false, 1.0);
            check(format!("oai {n_e}/{k} bias {bk}"), a, b);
        }
        // deepseek2: mla_topk vs sigmoid / softmax, exp_probs_b none, plain
        // or special, norm and scale.
        for sigm in [false, true] {
            for bk in 0..3 {
                let bias: Vec<f32> = (0..n_e)
                    .map(|e| match bk {
                        1 => 0.3 * r.u(),
                        _ => [
                            f32::NEG_INFINITY,
                            f32::NAN,
                            f32::INFINITY,
                            0.5,
                            -0.25,
                            0.0,
                            1e-3,
                        ][e % 7],
                    })
                    .collect();
                let bd = g.upload(&bias).unwrap();
                let ba = if bk == 0 { null } else { bd.arg() };
                for (norm, scale) in [(true, 1.0f32), (true, 2.446), (false, 2.5), (false, 1.0)] {
                    let a = legacy(&g, &lm, "mla_topk", &all, sh, 2 * n_e, &|l, i, w| {
                        vec![
                            l,
                            Arg::I32(ne),
                            Arg::I32(ki),
                            Arg::I32(sigm as i32),
                            ba,
                            Arg::I32(norm as i32),
                            Arg::F32(scale),
                            i,
                            w,
                        ]
                    });
                    let gate = if sigm { GATE_SIGMOID } else { GATE_SOFTMAX };
                    let b = router(&g, &ml, &all, sh, gate, ba, norm, scale);
                    check(
                        format!("mla {n_e}/{k} sigmoid {sigm} bias {bk} norm {norm} scale {scale}"),
                        a,
                        b,
                    );
                }
            }
        }
    }
    eprintln!("router_bits_match_legacy: {configs} configs, {slots} slots, {bad} differ");
    assert_eq!(configs, 280);
    assert_eq!(bad, 0);
}
