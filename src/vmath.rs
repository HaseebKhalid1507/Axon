//! f32 math for attention and SwiGLU: fused multiply-add, an 8-wide exp, and
//! the softmax and SiLU built on it.
//!
//! Every function here gives the same bits from its AVX2 path and its scalar
//! path (tails, or builds without AVX2 + FMA): the scalar code runs the same
//! operations in the same order, with `fmadd` fused exactly when the vector
//! code's FMA is. So results never depend on where a slice starts, how long
//! it is, or how work was split across threads.

/// a * b + c, fused (one rounding) when the build has FMA, else a multiply
/// and an add. Builds with FMA are the ones with the AVX2 paths (x86-64-v3),
/// so scalar and vector code agree in every build; without FMA there is no
/// vector path, and a software fused multiply-add would cost more than
/// attention itself.
#[inline(always)]
pub fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        a * b + c
    }
}

/// The 8 lanes summed as ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)):
/// the order `hsum8` uses.
#[inline(always)]
#[cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ),
    allow(dead_code)
)] // non-AVX dot + tests
pub fn sum8(l: &[f32; 8]) -> f32 {
    let a = [l[0] + l[4], l[1] + l[5], l[2] + l[6], l[3] + l[7]];
    let b = [a[0] + a[2], a[1] + a[3]];
    b[0] + b[1]
}

// exp(x) = 2^n * exp(r), n = floor(x * log2 e), r = x - n ln 2 in [0, ln 2)
// (Cody-Waite: ln 2 split in a short high part, exact in n * LN2_HI, and the
// rest). exp(r) = 1 + r * q(r), q degree 5: a relative minimax fit on
// [0, ln 2] (max error 2e-9 in f64, 1.3e-8 with the coefficients rounded to
// f32, ~0.1 ulp). 2^n is built from its exponent bits. x is clamped to
// [EXP_LO, EXP_HI] first: n = -127 gives 2^n = +0.0 (results below
// f32::MIN_POSITIVE flush to 0, so exp(-inf) = 0), n = 128 gives +inf.
// NaN propagates.
const EXP_LO: f32 = -88.0;
const EXP_HI: f32 = 88.8;
const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 355.0 / 512.0;
const LN2_LO: f32 = (std::f64::consts::LN_2 - 355.0 / 512.0) as f32;
const C1: f32 = f32::from_bits(0x3f7f_fffd);
const C2: f32 = f32::from_bits(0x3f00_0062);
const C3: f32 = f32::from_bits(0x3e2a_9bd9);
const C4: f32 = f32::from_bits(0x3d2b_af0f);
const C5: f32 = f32::from_bits(0x3bff_564b);
const C6: f32 = f32::from_bits(0x3aff_6dee);

/// exp(x), max error ~1.5 ulp where the result is a normal f32 (see the
/// tests for the measured sweep), 0 below that, +inf past f32::MAX. The same
/// bits as the AVX2 `exp8` lane for lane.
#[inline]
pub fn exp(x: f32) -> f32 {
    // max_ps(LO, x) / min_ps(HI, x): the second operand (x) wins on NaN.
    let x = if EXP_LO > x { EXP_LO } else { x };
    let x = if EXP_HI < x { EXP_HI } else { x };
    let n = (x * LOG2E).floor();
    let r = fmadd(n, -LN2_HI, x);
    let r = fmadd(n, -LN2_LO, r);
    let mut q = C6;
    q = fmadd(q, r, C5);
    q = fmadd(q, r, C4);
    q = fmadd(q, r, C3);
    q = fmadd(q, r, C2);
    q = fmadd(q, r, C1);
    let p = fmadd(q, r, 1.0);
    // n is an integer in -127..=128 (or NaN, where p is NaN already).
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// silu(g) * u = g / (1 + exp(-g)) * u, SwiGLU's combine.
#[inline]
pub fn swiglu1(g: f32, u: f32) -> f32 {
    g / (1.0 + exp(-g)) * u
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    /// `exp` on 8 lanes.
    #[inline(always)]
    pub fn exp8(x: __m256) -> __m256 {
        // SAFETY: AVX2 and FMA are enabled at compile time (cfg above); no
        // memory is touched.
        unsafe {
            let x = _mm256_max_ps(_mm256_set1_ps(EXP_LO), x);
            let x = _mm256_min_ps(_mm256_set1_ps(EXP_HI), x);
            let n = _mm256_floor_ps(_mm256_mul_ps(x, _mm256_set1_ps(LOG2E)));
            let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-LN2_HI), x);
            let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-LN2_LO), r);
            let mut q = _mm256_set1_ps(C6);
            q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C5));
            q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C4));
            q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C3));
            q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C2));
            q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C1));
            let p = _mm256_fmadd_ps(q, r, _mm256_set1_ps(1.0));
            let e = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127));
            _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32::<23>(e)))
        }
    }

    /// `sum8` of the lanes, same order.
    #[inline(always)]
    pub fn hsum8(v: __m256) -> f32 {
        // SAFETY: AVX enabled at compile time; registers only.
        unsafe {
            let a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
            let b = _mm_add_ps(a, _mm_movehl_ps(a, a));
            _mm_cvtss_f32(_mm_add_ss(b, _mm_shuffle_ps::<1>(b, b)))
        }
    }

    pub fn max(s: &[f32]) -> f32 {
        let (c, r) = s.as_chunks::<8>();
        // SAFETY: AVX enabled at compile time; loads are whole 8-float chunks.
        let m = unsafe {
            let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
            for c in c {
                m = _mm256_max_ps(m, _mm256_loadu_ps(c.as_ptr()));
            }
            let mut l = [0.0f32; 8];
            _mm256_storeu_ps(l.as_mut_ptr(), m);
            l
        };
        r.iter()
            .copied()
            .fold(m.into_iter().fold(f32::NEG_INFINITY, f32::max), f32::max)
    }

    pub fn exp_sum(s: &mut [f32], max: f32) -> f32 {
        let (c, r) = s.as_chunks_mut::<8>();
        // SAFETY: AVX2 + FMA enabled at compile time; loads and stores are
        // whole 8-float chunks of s.
        let lanes = unsafe {
            let (mut acc, m) = (_mm256_setzero_ps(), _mm256_set1_ps(max));
            for c in c {
                let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), m));
                _mm256_storeu_ps(c.as_mut_ptr(), e);
                acc = _mm256_add_ps(acc, e);
            }
            hsum8(acc)
        };
        lanes + super::exp_sum_tail(r, max)
    }

    pub fn swiglu(g: &mut [f32], u: &[f32]) {
        assert_eq!(g.len(), u.len(), "swiglu: gate/up lengths differ");
        let (gc, gr) = g.as_chunks_mut::<8>();
        let (uc, ur) = u.as_chunks::<8>();
        // SAFETY: AVX2 + FMA enabled at compile time; whole 8-float chunks.
        unsafe {
            let (one, sign) = (_mm256_set1_ps(1.0), _mm256_set1_ps(-0.0));
            for (g, u) in gc.iter_mut().zip(uc) {
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
            *g = swiglu1(*g, u);
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    test
))]
pub use x86::exp8;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub use x86::hsum8;

/// Largest element (-inf for empty), as `fold(-inf, f32::max)`.
pub fn max(s: &[f32]) -> f32 {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        x86::max(s)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        s.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
}

/// s[t] = exp(s[t] - max) in place; returns their sum in a fixed order:
/// 8 lane sums (lane i: t = i mod 8, over the whole 8-chunks, ascending),
/// combined by `sum8`, plus the tail (t past the last whole chunk) summed in
/// order from 0.0.
pub fn exp_sum(s: &mut [f32], max: f32) -> f32 {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        x86::exp_sum(s, max)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        let (c, r) = s.as_chunks_mut::<8>();
        let mut acc = [0.0f32; 8];
        for c in c {
            for (a, e) in acc.iter_mut().zip(c) {
                *e = exp(*e - max);
                *a += *e;
            }
        }
        sum8(&acc) + exp_sum_tail(r, max)
    }
}

fn exp_sum_tail(r: &mut [f32], max: f32) -> f32 {
    let mut tail = 0.0f32;
    for e in r {
        *e = exp(*e - max);
        tail += *e;
    }
    tail
}

/// Attention weights from scaled scores, in place: subtract the max, exp,
/// sum (`exp_sum`'s order), then multiply each by 1 / sum.
pub fn softmax(s: &mut [f32]) {
    let m = max(s);
    let inv = 1.0 / exp_sum(s, m);
    for e in s.iter_mut() {
        *e *= inv;
    }
}

/// g[i] = silu(g[i]) * u[i] (`swiglu1`).
pub fn swiglu(g: &mut [f32], u: &[f32]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        x86::swiglu(g, u)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        assert_eq!(g.len(), u.len(), "swiglu: gate/up lengths differ");
        for (g, &u) in g.iter_mut().zip(u) {
            *g = swiglu1(*g, u);
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

    /// exp over every `step`-th f32 in [EXP_LO, ln f32::MAX] (both signs'
    /// bit patterns walked) against f64 exp; the max ulp error, where.
    fn exp_sweep(step: u32) -> (f64, f32) {
        let (mut worst, mut at) = (0f64, 0f32);
        let mut check = |x: f32| {
            let want = (x as f64).exp();
            if want < f32::MIN_POSITIVE as f64 || want > f32::MAX as f64 {
                return;
            }
            let u = ulps(exp(x), want);
            if u > worst {
                (worst, at) = (u, x);
            }
        };
        for sign in [0u32, 0x8000_0000] {
            let mut b = 0u32;
            loop {
                let x = f32::from_bits(b | sign);
                if x.abs() > 89.0 {
                    break;
                }
                check(x);
                b += step;
            }
        }
        (worst, at)
    }

    #[test]
    fn exp_accuracy() {
        let (worst, at) = exp_sweep(97);
        println!("exp: max {worst:.3} ulp at {at:e}");
        assert!(worst <= 2.0, "exp: {worst} ulp at {at:e}");
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-0.0), 1.0);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp(-88.5), 0.0, "below f32::MIN_POSITIVE flushes");
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp(88.8), f32::INFINITY);
        assert!(exp(88.72).is_finite() && exp(88.72) > 3.3e38);
        assert!(exp(f32::NAN).is_nan());
        // SiLU near its extremes: -0 and g, as with libm's exp.
        assert_eq!(swiglu1(-200.0, 1.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(swiglu1(200.0, 1.0), 200.0);
    }

    /// The AVX2 lanes against the scalar `exp`, bit for bit, and SiLU /
    /// softmax against f64.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[test]
    fn simd_matches_scalar() {
        use std::arch::x86_64::*;
        let mut b = 0u32;
        let mut xs = [0f32; 8];
        while b < u32::MAX - 4099 {
            for (i, x) in xs.iter_mut().enumerate() {
                *x = f32::from_bits(b.wrapping_add(i as u32 * 523));
            }
            let mut got = [0f32; 8];
            // SAFETY: AVX enabled (cfg); 8-float arrays.
            unsafe { _mm256_storeu_ps(got.as_mut_ptr(), exp8(_mm256_loadu_ps(xs.as_ptr()))) };
            for (x, g) in xs.iter().zip(got) {
                let w = exp(*x);
                assert!(
                    g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                    "exp8({x:e}) {g:e} vs exp {w:e}"
                );
            }
            b += 4099;
        }
        let l: [f32; 8] = std::array::from_fn(|i| 1.0 / (i as f32 + 3.0));
        // SAFETY: as above.
        let h = unsafe { hsum8(_mm256_loadu_ps(l.as_ptr())) };
        assert_eq!(h.to_bits(), sum8(&l).to_bits());
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
                softmax(&mut got);
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
