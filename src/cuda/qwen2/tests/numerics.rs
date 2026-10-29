//! Fast mode against exact mode, graded: the scaffolding of
//! docs/design/determinism-modes.md §4.2 (ADR step M3), built before any
//! fast kernel exists so the first one's PR has a gate waiting for it.
//!
//! The gate is `assert_graded`, the GPU-vs-CPU gate since G3, unchanged,
//! pointed at a new reference: the net under test in its numerics mode
//! against the same device in exact mode, graded relative to an
//! *exact-vs-exact floor*, two placements the project already accepts as
//! "the same model" (`Floor`):
//! - a second GPU of another architecture in this process (sm_89 vs sm_120:
//!   their tensor-core attention sums f32 in other orders), if the host
//!   has one;
//! - else this GPU's f16 KV cache against its q8 one, both exact;
//! - on the CPU, the r8 repacked kernels against the per-row ones (the
//!   floor the GPU-vs-CPU gate grades against).
//!
//! Positions: `PROMPTS` with `STEPS` greedy steps of the reference each
//! (its picks fed to every net), then `WINDOWS` windows of natural text,
//! teacher-forced, where it also takes the ΔNLL (mode - exact, nats per
//! position) with an SE clustered by window (methodology.md
//! "Numerics-changing work"). `scripts/numerics_dnll.py` is that ΔNLL on
//! any model and corpus through `axon logits` (AXON_LOGITS_NLL), for a PR
//! table.
//!
//! Today it runs green in exact vs exact (the sanity check: a mode graded
//! against itself has KL 0, ΔNLL 0, every top-1 the same) and fast vs exact
//! (no fast kernel yet: fast runs exact's kernels, so the same), and it
//! fails another quantization of the same weights
//! (`graded_catches_another_model`): the gate has teeth.

use super::{
    Agree, PROMPTS, assert_graded, compare_trace, log_softmax, natural_ids, report, trace,
};
use crate::cuda::Gpu;
use crate::gguf::Model;
use crate::kv::KvType;
use crate::lm::{Backend, BackendOptions, Device, Lm};
use crate::numerics::{self, Numerics};
use std::path::Path;

/// Greedy steps after each prompt.
const STEPS: usize = 16;
/// Teacher-forced windows of natural text, of `WINDOW` tokens each.
const WINDOWS: usize = 3;
const WINDOW: usize = 48;
/// The nets' context.
const CTX: usize = 512;

/// One grading: the net under test and the floor, each against the exact
/// reference over the same positions, index for index (`assert_graded`
/// looks the floor's flips up by index).
pub(crate) struct Graded {
    pub(crate) mode: Agree,
    pub(crate) floor: Agree,
    /// Which floor (`Floor::name`).
    pub(crate) floor_name: &'static str,
    /// Teacher-forced ΔNLL against the reference, (mean, SE clustered by
    /// window): the net under test's and the floor's.
    pub(crate) dnll: (f64, f64),
    pub(crate) floor_dnll: (f64, f64),
}

/// The exact-vs-exact floor (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Floor {
    /// Exact on GPU 1, of another architecture than GPU 0's.
    CrossArch,
    /// Exact with the q8 KV cache, on the reference's GPU.
    KvQ8,
    /// Exact on the CPU's per-row kernels (AXON_REPACK=0's).
    CpuRows,
}

impl Floor {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Floor::CrossArch => "exact on another GPU architecture",
            Floor::KvQ8 => "exact with the q8 KV cache",
            Floor::CpuRows => "exact on the CPU's per-row kernels",
        }
    }
}

/// Mean of `d`'s values and its standard error clustered by `d`'s keys
/// (Liang-Zeger with the CR1 factor C/(C-1)); SE NaN for one cluster.
pub(crate) fn clustered_mean(d: &[(usize, f64)]) -> (f64, f64) {
    if d.is_empty() {
        return (0.0, 0.0);
    }
    let n = d.len() as f64;
    let mean = d.iter().map(|x| x.1).sum::<f64>() / n;
    let mut sums = std::collections::BTreeMap::<usize, f64>::new();
    for &(c, x) in d {
        *sums.entry(c).or_default() += x - mean;
    }
    let k = sums.len() as f64;
    let v = sums.values().map(|s| s * s).sum::<f64>() / (n * n) * k / (k - 1.0);
    (mean, v.sqrt())
}

/// `-ln p(next)` under logits `l`.
fn nll(l: &[f32], next: u32) -> f64 {
    -log_softmax(l)[next as usize]
}

/// Grade `o` against the exact reference `r`, with the floor `f` (exact
/// too: another accepted placement), on `m`'s tokenizer (module doc).
pub(crate) fn grade<R: Lm, F: Lm, O: Lm>(r: &R, f: &F, o: &O, m: &Model, floor: Floor) -> Graded {
    let tok = crate::tokenizer::Tokenizer::from_gguf(m);
    let (mut am, mut af) = (Agree::default(), Agree::default());
    for p in PROMPTS {
        let t = trace(r, &tok.encode(p), STEPS);
        compare_trace(&t, f, &mut af);
        compare_trace(&t, o, &mut am);
    }
    // Teacher-forced: every net fed the text, a token per step (decode).
    let ids = natural_ids(m, WINDOWS * WINDOW);
    let (mut dm, mut df) = (Vec::new(), Vec::new());
    for (w, seq) in ids.chunks(WINDOW).enumerate() {
        let (mut rc, mut fc, mut oc) = (r.new_cache(CTX), f.new_cache(CTX), o.new_cache(CTX));
        (am.last_kl, af.last_kl) = (0.0, 0.0);
        for (pos, &t) in seq.iter().enumerate() {
            let lr = r.forward(t, pos, &mut rc);
            let lf = f.forward(t, pos, &mut fc);
            let lo = o.forward(t, pos, &mut oc);
            af.tally(&lr, &lf);
            am.tally(&lr, &lo);
            if let Some(&next) = seq.get(pos + 1) {
                let n = nll(&lr, next);
                df.push((w, nll(&lf, next) - n));
                dm.push((w, nll(&lo, next) - n));
            }
        }
    }
    Graded {
        mode: am,
        floor: af,
        floor_name: floor.name(),
        dnll: clustered_mean(&dm),
        floor_dnll: clustered_mean(&df),
    }
}

/// The gate: `g.mode` graded against `g.floor` (`assert_graded`, the
/// GPU-vs-CPU bounds); with `bits` (the net under test runs no kernel of
/// its own: exact, or fast with no fast kernel, `numerics::fast_kernels`)
/// also nothing at all to grade: every logit equal, ΔNLL 0.
pub(crate) fn assert_graded_vs_exact(what: &str, g: &Graded, bits: bool) {
    report(&format!("{what} vs exact"), &g.mode);
    report(
        &format!("{what}: floor, {} vs exact", g.floor_name),
        &g.floor,
    );
    eprintln!(
        "{what}: ΔNLL vs exact {:+.2e} ± {:.1e} (floor {:+.2e} ± {:.1e}), {} teacher-forced positions",
        g.dnll.0,
        g.dnll.1,
        g.floor_dnll.0,
        g.floor_dnll.1,
        WINDOWS * (WINDOW - 1)
    );
    if bits {
        assert!(
            g.mode.top1 == g.mode.positions && g.mode.max_abs == 0.0 && g.mode.sum_kl == 0.0,
            "{what}: no kernel of its own, so exact's bits: {:?}",
            g.mode
        );
        assert_eq!(g.dnll.0, 0.0, "{what}: ΔNLL");
    }
    assert_graded(&g.mode, &g.floor);
}

/// A GPU of another architecture than GPU 0's, as device 1 (the cross-arch
/// floor), if this host has one.
fn other_arch_gpu() -> Option<&'static Gpu> {
    let a0 = Gpu::open_device(0).ok()?.info().ok()?.arch();
    let g1 = Gpu::open_device(1).ok()?;
    if g1.info().ok()?.arch() == a0 {
        return None;
    }
    Some(Box::leak(Box::new(g1)))
}

/// The model at `path` placed in `mode` on `device` (`BackendOptions`).
fn net(m: &Model, device: Device, kv: KvType, mode: Numerics) -> Backend<'_> {
    let o = BackendOptions::default()
        .device(device)
        .max_ctx(CTX)
        .kv(kv)
        .numerics(mode);
    Backend::new(m, o).unwrap()
}

fn load(path: &Path) -> Model {
    crate::gguf::load_with(path.to_str().unwrap(), Some(false)).unwrap()
}

/// The model at `path` (`under`: the net under test's file, the same by
/// default) graded in numerics mode `mode` against exact on `device`, with
/// this host's floor; and whether the net under test has no kernel of its
/// own (`assert_graded_vs_exact`'s `bits`).
pub(crate) fn graded_vs_exact(
    path: &Path,
    under: Option<&Path>,
    mode: Numerics,
    device: Device,
) -> (Graded, bool) {
    // A model each: a CPU model releases the bytes it repacks.
    let (mr, mo, mf) = (load(path), load(under.unwrap_or(path)), load(path));
    let r = net(&mr, device, KvType::F16, Numerics::Exact);
    let o = net(&mo, device, KvType::F16, mode);
    let bits = under.is_none() && (mode == Numerics::Exact || !numerics::fast_kernels(&o));
    let g = match device {
        Device::Cpu => {
            let f = crate::model::Qwen2::new_with(&mf, false);
            grade(&r, &f, &o, &mr, Floor::CpuRows)
        }
        _ => match other_arch_gpu() {
            Some(g1) => {
                let km = Box::leak(Box::new(crate::cuda::Kernels::module_for(g1, &mf).unwrap()));
                let f = crate::cuda::qwen2::CudaQwen2::new(g1, km, &mf, CTX).unwrap();
                grade(&r, &f, &o, &mr, Floor::CrossArch)
            }
            None => {
                let f = net(&mf, device, KvType::Q8, Numerics::Exact);
                grade(&r, &f, &o, &mr, Floor::KvQ8)
            }
        },
    };
    (g, bits)
}

/// `graded_vs_exact` for `mode` on the GPU, asserted.
fn gpu_mode(mode: Numerics) {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let (g, bits) = graded_vs_exact(&p, None, mode, Device::Cuda);
    assert_graded_vs_exact(&format!("0.5B cuda {mode}"), &g, bits);
}

/// `graded_vs_exact` for `mode` on the CPU, asserted.
fn cpu_mode(mode: Numerics) {
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let (g, bits) = graded_vs_exact(&p, None, mode, Device::Cpu);
    assert_eq!(g.floor_name, Floor::CpuRows.name());
    assert_graded_vs_exact(&format!("0.5B cpu {mode}"), &g, bits);
}

/// The sanity check (ADR M3): exact graded against exact is green, with
/// nothing to grade (KL 0, ΔNLL 0), against a floor that isn't zero.
#[test]
fn graded_exact_vs_exact() {
    gpu_mode(Numerics::Exact);
}

/// Fast graded against exact: green; while no fast kernel exists, exact's
/// bits (the fast module is the exact source under `cuda::FAST_DEFINE`).
#[test]
fn graded_fast_vs_exact() {
    gpu_mode(Numerics::Fast);
}

#[test]
fn graded_exact_vs_exact_cpu() {
    cpu_mode(Numerics::Exact);
}

#[test]
fn graded_fast_vs_exact_cpu() {
    cpu_mode(Numerics::Fast);
}

/// The gate has teeth: another quantization of the same weights (Q4_0 for
/// the Q4_K_M reference), which a fast mode must never be (it changes the
/// float order, not the model), fails it on the GPU.
#[test]
fn graded_catches_another_model() {
    if crate::testutil::gpu().is_none() {
        return;
    }
    let Some(p) = crate::testutil::model_path() else {
        return;
    };
    let other =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/Qwen2.5-0.5B-Instruct-Q4_0.gguf");
    if !other.exists() {
        eprintln!("skipping: {} not present", other.display());
        return;
    }
    let (g, bits) = graded_vs_exact(&p, Some(&other), Numerics::Exact, Device::Cuda);
    assert!(!bits);
    report("0.5B Q4_0 vs Q4_K_M exact", &g.mode);
    eprintln!("ΔNLL {:+.2e} ± {:.1e}", g.dnll.0, g.dnll.1);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_graded(&g.mode, &g.floor)
    }));
    assert!(r.is_err(), "another model passed the fast-vs-exact gate");
}

/// The clustered SE: a hand-worked case, and one cluster has none.
#[test]
fn clustered_mean_by_hand() {
    // Clusters {0: 1, 3}, {1: 2, 2}: mean 2; cluster sums of deviations
    // (-1 + 1) = 0 and (0 + 0) = 0 -> SE 0.
    assert_eq!(
        clustered_mean(&[(0, 1.0), (0, 3.0), (1, 2.0), (1, 2.0)]),
        (2.0, 0.0)
    );
    // {0: 0, 0}, {1: 2, 2}: mean 1, sums -2 and 2: v = 8 / 16 * 2 = 1.
    assert_eq!(
        clustered_mean(&[(0, 0.0), (0, 0.0), (1, 2.0), (1, 2.0)]),
        (1.0, 1.0)
    );
    assert!(clustered_mean(&[(0, 1.0), (0, 2.0)]).1.is_nan());
    assert_eq!(clustered_mean(&[]), (0.0, 0.0));
}
