//! f32 math for attention and SwiGLU: fused multiply-add, an 8-wide exp, and
//! the softmax and SiLU built on it.
//!
//! Every function here gives the same bits from its AVX2 path and its scalar
//! path (tails, `AXON_SIMD=scalar`, CPUs without AVX2): the scalar code runs
//! the same operations in the same order, with `fma::<F>` fused exactly when
//! the vector code's FMA is. So results never depend on where a slice
//! starts, how long it is, or how work was split across threads.
//!
//! Fused or not is a property of the CPU, not of the path: `fused()` is true
//! when the CPU has FMA, and then every path fuses (the AVX2 code with
//! `vfmadd`, the scalar code with `f32::mul_add`). A CPU without FMA has no
//! vector path here and runs a multiply and an add: a software fused
//! multiply-add would cost more than attention itself. The hot callers take
//! `F` as a const parameter (`multiversion!`'s `<V3, F>` form), so the choice
//! is made once per call, not per element.

/// Whether attention and SwiGLU fuse their multiply-adds: the CPU has FMA
/// (any build), or the build targets it. (Detected on x86 only: elsewhere,
/// aarch64 included, only a build that enables `fma` fuses.)
///
/// Fused code is inline `vfmadd` in the `v3` and `fma` variants. The one
/// remaining `f32::mul_add` lowered to libm `fmaf` is `portable` with F on
/// a CPU that has FMA but not sse4.1 (none known), or called directly (the
/// tests); its bits are only right with a correctly rounded `fmaf`, which
/// glibc's and musl's are.
#[inline]
pub fn fused() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        cfg!(target_feature = "fma") || is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        cfg!(target_feature = "fma")
    }
}

/// The CPU can run `multiversion!`'s `fma` variants (fma and sse4.1;
/// every FMA CPU has both). Decided once by std's feature cache.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn fast_fma() -> bool {
    is_x86_feature_detected!("fma") && is_x86_feature_detected!("sse4.1")
}

/// a * b + c, fused (one rounding) when F, else a multiply and an add.
#[inline(always)]
pub fn fma<const F: bool>(a: f32, b: f32, c: f32) -> f32 {
    if F { a.mul_add(b, c) } else { a * b + c }
}

/// `fma` with F = `fused()`: the tests' reference.
#[cfg(test)]
pub fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    if fused() {
        fma::<true>(a, b, c)
    } else {
        fma::<false>(a, b, c)
    }
}

/// The 8 lanes summed as ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)):
/// the order `hsum8` uses.
#[inline(always)]
pub fn sum8(l: &[f32; 8]) -> f32 {
    let a = [l[0] + l[4], l[1] + l[5], l[2] + l[6], l[3] + l[7]];
    let b = [a[0] + a[2], a[1] + a[3]];
    b[0] + b[1]
}

// exp(x) = 2^n * exp(r), n = x * log2 e rounded to nearest (ties even),
// r = x - n ln 2 in [-ln 2 / 2, ln 2 / 2] (Cody-Waite: ln 2 split in a short
// high part, exact in n * LN2_HI, and the rest). exp(r) = 1 + r * q(r), q
// degree 5: a relative minimax fit on [-ln 2 / 2, ln 2 / 2], coefficients
// rounded to f32 one at a time with the rest refitted (max error 3.1e-9,
// ~0.03 ulp). The polynomial is evaluated as 2 exp(r) (coefficients doubled,
// exact) and scaled by 2^(n - 1) built from its exponent bits, so n = 128
// still has a normal scale (2^127; results past f32::MAX round to +inf).
// x is clamped to [EXP_LO, EXP_HI] first, so n is in -126..=128; n = -126
// gives 2^(n - 1) = +0.0: results below 2^-125.5 (x < -125.5 ln 2 ~ -86.99)
// flush to 0, never subnormal, and exp(-inf) = 0. NaN propagates.
const EXP_LO: f32 = -87.3;
const EXP_HI: f32 = 88.8;
const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 355.0 / 512.0;
const LN2_LO: f32 = (std::f64::consts::LN_2 - 355.0 / 512.0) as f32;
// 2 * (the fit's c1..c6); exp(r) ~ 1 + c1 r + ... + c6 r^6.
const D1: f32 = f32::from_bits(0x4000_0000);
const D2: f32 = f32::from_bits(0x3f7f_fffe);
const D3: f32 = f32::from_bits(0x3eaa_aa49);
const D4: f32 = f32::from_bits(0x3daa_ac58);
const D5: f32 = f32::from_bits(0x3c89_1cd5);
const D6: f32 = f32::from_bits(0x3b35_280e);

/// exp(x), max error ~1 ulp where the result is at least 2^-125.5 (see the
/// tests for the measured sweep), 0 below that, +inf past f32::MAX. With F,
/// the same bits as the AVX2 `exp8` lane for lane.
#[inline(always)]
pub fn exp_f<const F: bool>(x: f32) -> f32 {
    // max_ps(LO, x) / min_ps(HI, x): the second operand (x) wins on NaN.
    let x = if EXP_LO > x { EXP_LO } else { x };
    let x = if EXP_HI < x { EXP_HI } else { x };
    let n = (x * LOG2E).round_ties_even();
    let r = fma::<F>(n, -LN2_HI, x);
    let r = fma::<F>(n, -LN2_LO, r);
    let mut q = D6;
    q = fma::<F>(q, r, D5);
    q = fma::<F>(q, r, D4);
    q = fma::<F>(q, r, D3);
    q = fma::<F>(q, r, D2);
    q = fma::<F>(q, r, D1);
    let p = fma::<F>(q, r, 2.0);
    // n is an integer in -126..=128 (or NaN, where p is NaN already).
    p * f32::from_bits(((n as i32 + 126) as u32) << 23)
}

/// `exp_f` with F = `fused()`.
#[cfg(test)]
pub fn exp(x: f32) -> f32 {
    if fused() {
        exp_f::<true>(x)
    } else {
        exp_f::<false>(x)
    }
}

/// silu(g) * u = g / (1 + exp(-g)) * u, SwiGLU's combine.
#[inline(always)]
pub fn swiglu1_f<const F: bool>(g: f32, u: f32) -> f32 {
    g / (1.0 + exp_f::<F>(-g)) * u
}

// GELU, tanh form: 0.5 x (1 + tanh(z)), z = sqrt(2/pi) (x + 0.044715 x^3).
// 0.5 (1 + tanh(z)) = 1 / (1 + exp(-2z)), so gelu(x) = x / (1 + exp(-2z)):
// the same exp as SiLU (llama.cpp's GELU_COEF_A and SQRT_2_OVER_PI).
const GELU_A: f32 = 0.044715;
/// -2 sqrt(2/pi).
const GELU_M2K: f32 = -2.0 * 0.797_884_6;

/// gelu_tanh(g) * u, GeGLU's combine (Gemma): g / (1 + exp(GELU_M2K *
/// fma(GELU_A * g * g, g, g))) * u.
#[inline(always)]
pub fn geglu1_f<const F: bool>(g: f32, u: f32) -> f32 {
    let z = fma::<F>(GELU_A * g * g, g, g);
    g / (1.0 + exp_f::<F>(GELU_M2K * z)) * u
}

/// `swiglu1_f` with F = `fused()`.
#[cfg(test)]
pub fn swiglu1(g: f32, u: f32) -> f32 {
    if fused() {
        swiglu1_f::<true>(g, u)
    } else {
        swiglu1_f::<false>(g, u)
    }
}

/// AVX2 + FMA versions, `#[target_feature]` so the default build carries no
/// AVX outside them. Callers: the `V3` variants of `multiversion!` bodies.
#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    /// `exp_f::<true>` on 8 lanes.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub fn exp8(x: __m256) -> __m256 {
        let x = _mm256_max_ps(_mm256_set1_ps(EXP_LO), x);
        let x = _mm256_min_ps(_mm256_set1_ps(EXP_HI), x);
        let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_mul_ps(x, _mm256_set1_ps(LOG2E)),
        );
        let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-LN2_HI), x);
        let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-LN2_LO), r);
        let mut q = _mm256_set1_ps(D6);
        q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(D5));
        q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(D4));
        q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(D3));
        q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(D2));
        q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(D1));
        let p = _mm256_fmadd_ps(q, r, _mm256_set1_ps(2.0));
        let e = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(126));
        _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32::<23>(e)))
    }

    /// `sum8` of the lanes, same order.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub fn hsum8(v: __m256) -> f32 {
        let a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let b = _mm_add_ps(a, _mm_movehl_ps(a, a));
        _mm_cvtss_f32(_mm_add_ss(b, _mm_shuffle_ps::<1>(b, b)))
    }

    /// `super::max`.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub fn max(s: &[f32]) -> f32 {
        let (c, r) = s.as_chunks::<8>();
        let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
        for c in c {
            // SAFETY: a whole 8-float chunk.
            m = _mm256_max_ps(m, unsafe { _mm256_loadu_ps(c.as_ptr()) });
        }
        let mut l = [0.0f32; 8];
        // SAFETY: storing 8 floats into an 8-float array.
        unsafe { _mm256_storeu_ps(l.as_mut_ptr(), m) };
        r.iter()
            .copied()
            .fold(l.into_iter().fold(f32::NEG_INFINITY, f32::max), f32::max)
    }

    /// `super::exp_sum` with F.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub fn exp_sum(s: &mut [f32], max: f32) -> f32 {
        let (c, r) = s.as_chunks_mut::<8>();
        let (mut acc, m) = (_mm256_setzero_ps(), _mm256_set1_ps(max));
        for c in c {
            // SAFETY: loads and stores are whole 8-float chunks of s.
            unsafe {
                let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), m));
                _mm256_storeu_ps(c.as_mut_ptr(), e);
                acc = _mm256_add_ps(acc, e);
            }
        }
        hsum8(acc) + super::exp_sum_tail::<true>(r, max)
    }

    /// `super::swiglu` with F.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub fn swiglu(g: &mut [f32], u: &[f32]) {
        assert_eq!(g.len(), u.len(), "swiglu: gate/up lengths differ");
        let (gc, gr) = g.as_chunks_mut::<8>();
        let (uc, ur) = u.as_chunks::<8>();
        let (one, sign) = (_mm256_set1_ps(1.0), _mm256_set1_ps(-0.0));
        for (g, u) in gc.iter_mut().zip(uc) {
            // SAFETY: whole 8-float chunks.
            unsafe {
                let x = _mm256_loadu_ps(g.as_ptr());
                let e = exp8(_mm256_xor_ps(x, sign));
                let y = _mm256_div_ps(x, _mm256_add_ps(one, e));
                _mm256_storeu_ps(
                    g.as_mut_ptr(),
                    _mm256_mul_ps(y, _mm256_loadu_ps(u.as_ptr())),
                );
            }
        }
        for (g, &u) in gr.iter_mut().zip(ur) {
            *g = swiglu1_f::<true>(*g, u);
        }
    }

    /// `super::geglu` with F.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub fn geglu(g: &mut [f32], u: &[f32]) {
        assert_eq!(g.len(), u.len(), "geglu: gate/up lengths differ");
        let (gc, gr) = g.as_chunks_mut::<8>();
        let (uc, ur) = u.as_chunks::<8>();
        let (one, a, m2k) = (
            _mm256_set1_ps(1.0),
            _mm256_set1_ps(GELU_A),
            _mm256_set1_ps(GELU_M2K),
        );
        for (g, u) in gc.iter_mut().zip(uc) {
            // SAFETY: whole 8-float chunks.
            unsafe {
                let x = _mm256_loadu_ps(g.as_ptr());
                let ax2 = _mm256_mul_ps(_mm256_mul_ps(a, x), x);
                let z = _mm256_fmadd_ps(ax2, x, x);
                let e = exp8(_mm256_mul_ps(m2k, z));
                let y = _mm256_div_ps(x, _mm256_add_ps(one, e));
                _mm256_storeu_ps(
                    g.as_mut_ptr(),
                    _mm256_mul_ps(y, _mm256_loadu_ps(u.as_ptr())),
                );
            }
        }
        for (g, &u) in gr.iter_mut().zip(ur) {
            *g = geglu1_f::<true>(*g, u);
        }
    }
}

/// Largest element (-inf for empty), as `fold(-inf, f32::max)`.
///
/// # Safety
/// With V3, the CPU must have AVX2 and FMA.
#[inline(always)]
pub unsafe fn max<const V3: bool>(s: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if V3 {
        // SAFETY: our caller's contract.
        return unsafe { x86::max(s) };
    }
    s.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// s[t] = exp(s[t] - max) in place; returns their sum in a fixed order:
/// 8 lane sums (lane i: t = i mod 8, over the whole 8-chunks, ascending),
/// combined by `sum8`, plus the tail (t past the last whole chunk) summed in
/// order from 0.0.
///
/// # Safety
/// With V3, the CPU must have AVX2 and FMA (and V3 implies F).
#[inline(always)]
pub unsafe fn exp_sum<const V3: bool, const F: bool>(s: &mut [f32], max: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if V3 {
        // SAFETY: our caller's contract.
        return unsafe { x86::exp_sum(s, max) };
    }
    let (c, r) = s.as_chunks_mut::<8>();
    let mut acc = [0.0f32; 8];
    for c in c {
        for (a, e) in acc.iter_mut().zip(c) {
            *e = exp_f::<F>(*e - max);
            *a += *e;
        }
    }
    sum8(&acc) + exp_sum_tail::<F>(r, max)
}

#[inline(always)]
fn exp_sum_tail<const F: bool>(r: &mut [f32], max: f32) -> f32 {
    let mut tail = 0.0f32;
    for e in r {
        *e = exp_f::<F>(*e - max);
        tail += *e;
    }
    tail
}

/// Attention weights from scaled scores, in place: subtract the max, exp,
/// sum (`exp_sum`'s order), then multiply each by 1 / sum.
///
/// # Safety
/// With V3, the CPU must have AVX2 and FMA (and V3 implies F).
#[inline(always)]
pub unsafe fn softmax<const V3: bool, const F: bool>(s: &mut [f32]) {
    // SAFETY: our caller's contract.
    let m = unsafe { max::<V3>(s) };
    // SAFETY: as above.
    let inv = 1.0 / unsafe { exp_sum::<V3, F>(s, m) };
    for e in s.iter_mut() {
        *e *= inv;
    }
}

multiversion! {
    /// g[i] = silu(g[i]) * u[i] (`swiglu1_f`).
    pub fn swiglu<V3, F>(g: &mut [f32], u: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        if V3 {
            // SAFETY: V3 only in the v3 variant.
            return unsafe { x86::swiglu(g, u) };
        }
        assert_eq!(g.len(), u.len(), "swiglu: gate/up lengths differ");
        for (g, &u) in g.iter_mut().zip(u) {
            *g = swiglu1_f::<F>(*g, u);
        }
    }
}

multiversion! {
    /// g[i] = gelu_tanh(g[i]) * u[i] (`geglu1_f`).
    pub fn geglu<V3, F>(g: &mut [f32], u: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        if V3 {
            // SAFETY: V3 only in the v3 variant.
            return unsafe { x86::geglu(g, u) };
        }
        assert_eq!(g.len(), u.len(), "geglu: gate/up lengths differ");
        for (g, &u) in g.iter_mut().zip(u) {
            *g = geglu1_f::<F>(*g, u);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// |got - want| in ulps of want's binade (f32), want a normal f32.
    fn ulps(got: f32, want: f64) -> f64 {
        let e = ((want.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        (got as f64 - want).abs() / 2f64.powi(e - 23)
    }

    /// exp over every `step`-th f32 bit pattern with |x| <= 89 (both signs)
    /// against f64 exp, where the true result is a finite f32 >= 2^-125.4
    /// (below 2^-125.4 it may also flush to 0; below 2^-126.5 it must).
    /// Returns the max ulp error and where, overall and on x in [-1, 0].
    fn exp_sweep(step: u32) -> [(f64, f32); 2] {
        let mut worst = [(0f64, 0f32); 2];
        for sign in [0u32, 0x8000_0000] {
            let mut b = 0u32;
            loop {
                let x = f32::from_bits(b | sign);
                if x.abs() > 89.0 {
                    break;
                }
                b += step;
                let (want, got) = ((x as f64).exp(), exp(x));
                if want > f32::MAX as f64 {
                    assert_eq!(got, f32::INFINITY, "exp({x:e})");
                    continue;
                }
                if want < 2f64.powf(-125.4) {
                    let flush = want < 2f64.powf(-126.5);
                    assert!(
                        got == 0.0 || (!flush && ulps(got, want) <= 2.0),
                        "exp({x:e}) = {got:e}"
                    );
                    continue;
                }
                let u = ulps(got, want);
                for (i, w) in worst.iter_mut().enumerate() {
                    if u > w.0 && (i == 0 || (-1.0..=0.0).contains(&x)) {
                        *w = (u, x);
                    }
                }
            }
        }
        worst
    }

    fn exp_accuracy_on(step: u32) {
        let [(worst, at), (w1, at1)] = exp_sweep(step);
        println!(
            "exp, every {step} of the f32s: max {worst:.3} ulp at {at:e}; on [-1, 0] {w1:.3} at {at1:e}"
        );
        assert!(worst <= 2.0, "exp: {worst} ulp at {at:e}");
    }

    /// Every 97th f32; `exp_accuracy_every_float` is the full sweep.
    #[test]
    fn exp_accuracy() {
        exp_accuracy_on(97);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-0.0), 1.0);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp(-88.5), 0.0, "below 2^-125.5 flushes");
        assert!(exp(-86.9) > 0.0 && exp(-86.9) >= f32::MIN_POSITIVE);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp(88.8), f32::INFINITY);
        assert!(exp(88.72).is_finite() && exp(88.72) > 3.3e38);
        assert!(exp(f32::NAN).is_nan());
        // SiLU near its extremes: -0 and g, as with libm's exp.
        assert_eq!(swiglu1(-200.0, 1.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(swiglu1(200.0, 1.0), 200.0);
    }

    /// Every f32 with |x| <= 89 (~2.2e9 values; ~20 s optimized).
    #[test]
    #[ignore = "slow: full sweep, run with --ignored --nocapture"]
    fn exp_accuracy_every_float() {
        exp_accuracy_on(1);
    }

    /// The CPU has AVX2 + FMA (the `x86` functions can run).
    #[cfg(target_arch = "x86_64")]
    fn can_avx2() -> bool {
        let ok = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        if !ok {
            println!("::warning::no AVX2+FMA on this CPU: vmath SIMD paths not compared");
        }
        ok
    }

    /// The AVX2 lanes against the scalar fused `exp_f`, bit for bit, and
    /// SiLU / softmax against f64.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_matches_scalar() {
        use std::arch::x86_64::*;
        if !can_avx2() {
            return;
        }
        let mut b = 0u32;
        let mut xs = [0f32; 8];
        while b < u32::MAX - 4099 {
            for (i, x) in xs.iter_mut().enumerate() {
                *x = f32::from_bits(b.wrapping_add(i as u32 * 523));
            }
            let mut got = [0f32; 8];
            // SAFETY: AVX2 + FMA checked above; 8-float arrays.
            unsafe { _mm256_storeu_ps(got.as_mut_ptr(), x86::exp8(_mm256_loadu_ps(xs.as_ptr()))) };
            for (x, g) in xs.iter().zip(got) {
                let w = exp_f::<true>(*x);
                assert!(
                    g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                    "exp8({x:e}) {g:e} vs exp {w:e}"
                );
            }
            b += 4099;
        }
        let l: [f32; 8] = std::array::from_fn(|i| 1.0 / (i as f32 + 3.0));
        // SAFETY: as above.
        let h = unsafe { x86::hsum8(_mm256_loadu_ps(l.as_ptr())) };
        assert_eq!(h.to_bits(), sum8(&l).to_bits());
    }

    /// softmax on the path `quant::v3()` picks (as attention runs it).
    fn softmax_rt(s: &mut [f32]) {
        // SAFETY: V3 only when v3() detected the features.
        unsafe {
            if crate::quant::v3() {
                softmax::<true, true>(s)
            } else if fused() {
                softmax::<false, true>(s)
            } else {
                softmax::<false, false>(s)
            }
        }
    }

    /// Every path of softmax and SwiGLU this CPU can run (scalar unfused,
    /// scalar fused if it has FMA, AVX2 if it has AVX2 + FMA) against its
    /// scalar definition with the same F, bit for bit, at every length.
    #[test]
    fn softmax_swiglu_paths() {
        let mut seed = 0x0bad_5eed_1234_5678u64;
        #[cfg(target_arch = "x86_64")]
        let avx = can_avx2();
        #[cfg(not(target_arch = "x86_64"))]
        let avx = false;
        let f = fused();
        for n in (0..70).chain([255, 1000]) {
            let s = rnd_vec(&mut seed, n, 20.0);
            let u = rnd_vec(&mut seed, n, 3.0);
            let g0 = rnd_vec(&mut seed, n, 20.0);
            let want = |fz: bool| {
                let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = s
                    .iter()
                    .map(|&x| {
                        if fz {
                            exp_f::<true>(x - m)
                        } else {
                            exp_f::<false>(x - m)
                        }
                    })
                    .collect();
                let (c, r) = e.as_chunks::<8>();
                let mut acc = [0f32; 8];
                for c in c {
                    for i in 0..8 {
                        acc[i] += c[i];
                    }
                }
                let inv = 1.0 / (sum8(&acc) + r.iter().fold(0.0f32, |a, &x| a + x));
                let sw: Vec<u32> = g0
                    .iter()
                    .zip(&u)
                    .map(|(&g, &u)| {
                        if fz {
                            swiglu1_f::<true>(g, u)
                        } else {
                            swiglu1_f::<false>(g, u)
                        }
                        .to_bits()
                    })
                    .collect();
                (
                    e.iter().map(|x| (x * inv).to_bits()).collect::<Vec<_>>(),
                    sw,
                )
            };
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            let check = |fz: bool, sm: &[f32], sw: &[f32], what: &str| {
                let (ws, ww) = want(fz);
                assert_eq!(bits(sm), ws, "softmax {what} n={n}");
                assert_eq!(bits(sw), ww, "swiglu {what} n={n}");
            };
            // SAFETY (every block): V3 paths only after can_avx2().
            let (mut a, mut g) = (s.clone(), g0.clone());
            unsafe { softmax::<false, false>(&mut a) };
            unsafe { swiglu::imp::<false, false>(&mut g, &u) };
            check(false, &a, &g, "unfused");
            if f {
                let (mut a, mut g) = (s.clone(), g0.clone());
                unsafe { softmax::<false, true>(&mut a) };
                unsafe { swiglu::imp::<false, true>(&mut g, &u) };
                check(true, &a, &g, "fused scalar");
            }
            #[cfg(target_arch = "x86_64")]
            if fast_fma() {
                let (mut a, mut g) = (s.clone(), g0.clone());
                unsafe { softmax::<false, true>(&mut a) };
                unsafe { swiglu::fma(&mut g, &u) };
                check(true, &a, &g, "fma variant");
            }
            if avx {
                let (mut a, mut g) = (s.clone(), g0.clone());
                unsafe { softmax::<true, true>(&mut a) };
                unsafe { swiglu::imp::<true, true>(&mut g, &u) };
                check(true, &a, &g, "avx2");
            }
        }
    }

    /// Every GeGLU path this CPU can run equals `geglu1_f` with the same F
    /// bit for bit at every length, and the f64 tanh-form GELU closely.
    #[test]
    fn geglu_paths() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        #[cfg(target_arch = "x86_64")]
        let avx = can_avx2();
        #[cfg(not(target_arch = "x86_64"))]
        let avx = false;
        let f = fused();
        let mut err = 0f64;
        for n in (0..70).chain([255, 1000]) {
            for scale in [0.5f32, 4.0, 30.0] {
                let u = rnd_vec(&mut seed, n, 3.0);
                let g0 = rnd_vec(&mut seed, n, scale);
                let want = |fz: bool| -> Vec<u32> {
                    g0.iter()
                        .zip(&u)
                        .map(|(&g, &u)| {
                            if fz {
                                geglu1_f::<true>(g, u)
                            } else {
                                geglu1_f::<false>(g, u)
                            }
                            .to_bits()
                        })
                        .collect()
                };
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                let mut g = g0.clone();
                // SAFETY (every block): V3 paths only after can_avx2().
                unsafe { geglu::imp::<false, false>(&mut g, &u) };
                assert_eq!(bits(&g), want(false), "unfused n={n}");
                if f {
                    let mut g = g0.clone();
                    unsafe { geglu::imp::<false, true>(&mut g, &u) };
                    assert_eq!(bits(&g), want(true), "fused scalar n={n}");
                }
                #[cfg(target_arch = "x86_64")]
                if fast_fma() {
                    let mut g = g0.clone();
                    unsafe { geglu::fma(&mut g, &u) };
                    assert_eq!(bits(&g), want(true), "fma variant n={n}");
                }
                if avx {
                    let mut g = g0.clone();
                    unsafe { geglu::imp::<true, true>(&mut g, &u) };
                    assert_eq!(bits(&g), want(true), "avx2 n={n}");
                }
                let mut g = g0.clone();
                geglu(&mut g, &u);
                for i in 0..n {
                    let x = g0[i] as f64;
                    let k = (2.0 / std::f64::consts::PI).sqrt();
                    let gl = 0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh());
                    let want = gl * u[i] as f64;
                    if want.abs() > 1e-20 {
                        err = err.max((g[i] as f64 - want).abs() / want.abs().max(1e-3));
                    }
                }
            }
        }
        println!("geglu max rel err {err:.2e}");
        assert!(err < 1e-5, "geglu {err:e}");
    }

    fn rnd_vec(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                ((*seed >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Vector softmax and SwiGLU equal their scalar definitions bit for bit at
    /// every length (whole chunks and tails), and stay close to f64.
    #[test]
    fn softmax_swiglu_match_scalar() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        let (mut sm_err, mut lm_err, mut sw_err) = (0f64, 0f64, 0f64);
        for n in (1..70).chain([255, 1000]) {
            for scale in [0.5, 8.0, 40.0] {
                let s = rnd_vec(&mut seed, n, scale);
                let mut got = s.clone();
                softmax_rt(&mut got);
                // Scalar reference in exp_sum's order.
                let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = s.iter().map(|&x| exp(x - m)).collect();
                let (c, r) = e.as_chunks::<8>();
                let mut acc = [0f32; 8];
                for c in c {
                    for i in 0..8 {
                        acc[i] += c[i];
                    }
                }
                let sum = sum8(&acc) + r.iter().fold(0.0f32, |a, &x| a + x);
                let inv = 1.0 / sum;
                // f64 on the same f32 differences (their rounding is the
                // subtraction's, not exp's); libm's f32 softmax for scale.
                let d: Vec<f64> = s.iter().map(|&x| (x - m) as f64).collect();
                let z: f64 = d.iter().map(|x| x.exp()).sum();
                let zl: f32 = s.iter().map(|&x| (x - m).exp()).sum();
                for (i, (&g, &e)) in got.iter().zip(&e).enumerate() {
                    assert_eq!(g.to_bits(), (e * inv).to_bits(), "softmax n={n} t={i}");
                    let want = d[i].exp() / z;
                    if want < 1e-30 {
                        continue; // subnormal weights: no relative precision
                    }
                    sm_err = sm_err.max((g as f64 - want).abs() / want);
                    let l = (s[i] - m).exp() / zl;
                    lm_err = lm_err.max((l as f64 - want).abs() / want);
                }

                let u = rnd_vec(&mut seed, n, 3.0);
                let mut g = rnd_vec(&mut seed, n, scale);
                let g0 = g.clone();
                swiglu(&mut g, &u);
                for i in 0..n {
                    assert_eq!(
                        g[i].to_bits(),
                        swiglu1(g0[i], u[i]).to_bits(),
                        "swiglu n={n}"
                    );
                    let x = g0[i] as f64;
                    let want = x / (1.0 + (-x).exp()) * u[i] as f64;
                    if want.abs() > 1e-30 {
                        sw_err = sw_err.max((g[i] as f64 - want).abs() / want.abs());
                    }
                }
            }
        }
        println!("softmax max rel err {sm_err:.2e} (libm expf {lm_err:.2e}), swiglu {sw_err:.2e}");
        assert!(
            sm_err < 1e-6 && sw_err < 1e-6,
            "softmax {sm_err:e} swiglu {sw_err:e}"
        );
    }
}
