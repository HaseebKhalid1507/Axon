//! Numerics modes ([ADR](../docs/design/determinism-modes.md), accepted
//! 2026-10-29).
//!
//! - `exact`, the default everywhere: the contract docs/determinism.md
//!   states (decode == batched prefill == verify, bit for bit; one canonical
//!   float order per product; batched == alone; cache == fresh prefill),
//!   gated by the md5 of `gen` outputs.
//! - `fast`: kernels whose float order may depend on more than the model's
//!   shapes and the token's position (split-K with an arrival-order fixup,
//!   NCCL-style all-reduce, FP8 tensor-core accumulation) are allowed, gated
//!   by tolerance against exact mode on the same device (the ADR's §4.2)
//!   instead of by md5. `axon serve --datacenter` turns it on.
//!
//! A mode belongs to a `Backend` (`BackendOptions::numerics`), and in the
//! CLI to the whole process: a cache written by one mode's kernels can't be
//! continued by the other's bit for bit, so there is no per-request mode
//! (run two servers).
//!
//! **No fast kernel exists yet** (the ADR's §4.6: none is built before a
//! measured gap names it, with a kill rule). Until one does, `fast` runs
//! the exact kernels, says so once (`FAST_NOTE`), and still *identifies* as
//! fast: its own fingerprint (`numerics=fast/N.M` in the identity tuple), its
//! own NVRTC modules (`cuda::FAST_DEFINE` in their options, so in the
//! kernel cache's key), its own kvsave identity (a session saved in one
//! mode is restored into the other as tokens, prefilled again, never as
//! bytes). A fast fingerprint promises less than an exact one, never more:
//! a client that gets exact bits under it today was promised tolerance,
//! and the first fast kernel bumps `NUMERICS_VERSION_FAST`, so a client
//! caching on the fingerprint sees that too.

use crate::identity::{NUMERICS_VERSION_EXACT, NUMERICS_VERSION_FAST};
use crate::lm::Backend;

/// How a `Backend` orders its float arithmetic (module doc): `Exact` (the
/// default) or `Fast`. Non-exhaustive: a stricter third mode (the ADR's
/// `exact-xarch`, cross-architecture exact) may come.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Numerics {
    /// Bit-exact: decode == batched prefill == verify, one canonical order.
    #[default]
    Exact,
    /// Order-changing kernels allowed; graded against `Exact`, not equal.
    Fast,
}

impl Numerics {
    /// `--numerics`' value: `exact` or `fast` (an error names both).
    pub fn parse(s: &str) -> Result<Numerics, String> {
        match s {
            "exact" => Ok(Numerics::Exact),
            "fast" => Ok(Numerics::Fast),
            _ => Err(format!("--numerics takes exact or fast, got '{s}'")),
        }
    }

    /// `exact` or `fast`.
    pub fn name(self) -> &'static str {
        match self {
            Numerics::Exact => "exact",
            Numerics::Fast => "fast",
        }
    }

    /// The mode's numerics version (identity.rs states when each counter
    /// bumps): exact's counter, `1`; fast's is exact's and its own,
    /// `1.1`. Fast mode runs on top of the exact kernels and the
    /// request-to-text code, so a change to those (an exact bump) moves
    /// fast's version too, with nobody having to remember a second bump.
    pub fn version(self) -> String {
        match self {
            Numerics::Exact => format!("{NUMERICS_VERSION_EXACT}"),
            Numerics::Fast => format!("{NUMERICS_VERSION_EXACT}.{NUMERICS_VERSION_FAST}"),
        }
    }

    /// The identity tuple's `numerics` value: `exact/1`, `fast/1.2`.
    pub fn tag(self) -> String {
        format!("{}/{}", self.name(), self.version())
    }
}

impl std::fmt::Display for Numerics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Whether `net` runs any fast-set kernel (`Backend::fast_kernels`): an FP8
/// checkpoint's native matrices on the GPU (fp8.cu, docs/design/fp8-weights.md),
/// the only fast kernels; every launch calls `CudaQwen2::fast_launched`. The tests that hold fast mode to exact mode's bits while it's
/// false (`identity_checks`' path equalities under `serve_identity_fast`,
/// `graded_vs_exact`'s bit check, the md5 gate's fast column) become
/// graded-only there, and the graded gate then requires the launch count
/// to move in the arms it grades: a fast kernel that never ran can't pass
/// it.
pub(crate) fn fast_kernels(net: &Backend) -> bool {
    net.gpu_nets().iter().any(|n| n.f8_native())
}

/// What `--numerics fast` says once where no fast kernel exists yet.
pub(crate) const FAST_NOTE: &str =
    "no fast kernels on this backend yet; running exact kernels; identity says fast";

/// The note for `net` (`--numerics fast` without fast kernels), if any.
pub(crate) fn fast_note(net: &Backend) -> Option<String> {
    (net.numerics() == Numerics::Fast && !fast_kernels(net))
        .then(|| format!("axon: --numerics fast: {FAST_NOTE}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flag's values and the tuple's tags; anything else is refused by
    /// name, and the default is exact.
    #[test]
    fn parse_and_tags() {
        assert_eq!(Numerics::parse("exact"), Ok(Numerics::Exact));
        assert_eq!(Numerics::parse("fast"), Ok(Numerics::Fast));
        for bad in ["", "Fast", "fastest", "exact/1", "approx"] {
            let e = Numerics::parse(bad).unwrap_err();
            assert!(e.contains("exact or fast"), "{bad}: {e}");
        }
        assert_eq!(Numerics::default(), Numerics::Exact);
        assert_eq!(
            Numerics::Exact.tag(),
            format!("exact/{NUMERICS_VERSION_EXACT}")
        );
        assert_eq!(
            Numerics::Fast.tag(),
            format!("fast/{NUMERICS_VERSION_EXACT}.{NUMERICS_VERSION_FAST}")
        );
        assert_eq!(Numerics::Fast.version(), "1.1");
        assert_eq!(Numerics::Fast.to_string(), "fast");
    }

    /// A CPU backend in fast mode: no fast kernel, so the note, and the
    /// mode it was asked for; exact says nothing.
    #[test]
    fn fast_note_on_a_fixture() {
        crate::kv::set_f32(false);
        // A model each: a CPU model releases the tensors it repacks.
        let fixture = || crate::model::tests::qwen3_fixture(crate::quant::F32);
        let (me, mf) = (fixture(), fixture());
        let mk = |m, n| {
            let o = crate::lm::BackendOptions::default()
                .device(crate::lm::Device::Cpu)
                .max_ctx(64)
                .numerics(n);
            Backend::new(m, o).unwrap()
        };
        let (e, f) = (mk(&me, Numerics::Exact), mk(&mf, Numerics::Fast));
        assert_eq!(
            (e.numerics(), f.numerics()),
            (Numerics::Exact, Numerics::Fast)
        );
        assert_eq!(fast_note(&e), None);
        assert!(!fast_kernels(&f));
        let n = fast_note(&f).unwrap();
        assert!(
            n.contains(FAST_NOTE) && n.contains("--numerics fast"),
            "{n}"
        );
    }
}
