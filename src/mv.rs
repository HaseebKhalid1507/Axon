//! Function multiversioning for the hot loops outside the quant kernels.
//!
//! The default build targets baseline x86-64, so it runs on any x86-64 CPU.
//! The loops that gain from AVX2 (norms, attention, activation quantization,
//! argmax, ...) are written once and compiled twice: `name::portable`, and
//! `name::v3` with `#[target_feature(enable = "avx2,bmi1,bmi2,f16c,fma,lzcnt")]`
//! (x86-64-v3; the quant kernels need only avx2/fma/f16c, see
//! `quant::has_avx2`). `name` picks one at runtime through `quant::v3()`:
//! the v3 variant on a SIMD kernel path when the CPU also has bmi1/bmi2/lzcnt
//! (`AXON_SIMD=scalar` runs the portable code).
//!
//! Both variants are the same source: the body becomes an `#[inline(always)]`
//! `imp::<V3>` inlined into each, so LLVM vectorizes it for each feature set.
//! The const parameter (named by the caller) lets a body pick a hand-written
//! AVX helper when it is true. Bit-identical by construction: Rust never
//! contracts a * b + c into an FMA and never reorders float sums, so the
//! features change instruction selection, not results (the `*_v3` helpers keep
//! the portable lanes and order; `model::tests::multiversion_bit_identical`,
//! `quant::tests::multiversion_quantize_bit_identical` and
//! `tests::argmax_multiversion_identical` check every pair). The one
//! exception is a NaN's sign/payload, which Rust doesn't specify.
//!
//! Closures defined in a `#[target_feature]` fn inherit its features (Rust
//! 1.86+), but these bodies live in `imp`, which has none of its own, so a
//! closure there is only vectorized for v3 if LLVM inlines it (small ones
//! normally are). The current bodies use explicit loops, so nothing depends
//! on that. Float sums that replace `Iterator::sum` start from -0.0, as `sum`
//! does, so a sign of zero can't change.

/// `multiversion! { /// docs
/// fn name<V3>(a: A, b: B) -> R { body } }` defines `fn name` (the runtime
/// dispatcher) and `mod name { fn portable; unsafe fn v3 }` (x86_64 only for
/// `v3`). Arguments must be plain identifiers; `V3` is a `const bool` usable
/// in the body (true in `v3`).
macro_rules! multiversion {
    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident<$v3:ident>($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block
    ) => {
        $(#[$attr])*
        #[inline]
        #[allow(clippy::too_many_arguments)]
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            if crate::quant::v3() {
                // SAFETY: v3() is true only after the x86-64-v3 features were detected.
                return unsafe { $name::v3($($arg),*) };
            }
            $name::portable($($arg),*)
        }

        $vis mod $name {
            #[allow(unused_imports)]
            use super::*;

            /// The shared body; other multiversioned bodies call `imp::<V3>`
            /// directly to stay in their own variant.
            ///
            /// # Safety
            /// With `V3` true, the CPU must have the x86-64-v3 features.
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn imp<const $v3: bool>($($arg: $ty),*) $(-> $ret)? $body

            /// The baseline x86-64 (or other target) variant.
            #[allow(clippy::too_many_arguments)]
            pub fn portable($($arg: $ty),*) $(-> $ret)? {
                // SAFETY: V3 false needs nothing.
                unsafe { imp::<false>($($arg),*) }
            }

            /// The x86-64-v3 variant.
            ///
            /// # Safety
            /// The CPU must have avx2, bmi1, bmi2, f16c, fma and lzcnt.
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,bmi1,bmi2,f16c,fma,lzcnt")]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn v3($($arg: $ty),*) $(-> $ret)? {
                // SAFETY: our caller's contract.
                unsafe { imp::<true>($($arg),*) }
            }
        }
    };
}
