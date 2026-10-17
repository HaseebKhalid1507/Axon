//! IQ4_NL: 32-value blocks of 4-bit indices into a fixed non-linear table
//! (ggml's kvalues_iq4nl), one f16 scale. The 0.5B Q2_K / Q3_K_M GGUFs use it
//! for their 896-wide rows (llama.cpp's fallback when a row isn't a multiple
//! of 256).
//!
//! Dot against Q8_0 activations, per block: s = Σ table[q] x (exact), then
//! acc += d dx f32(s) - Q4_0's step, in every path (scalar, AVX2, four
//! tokens), so they all give the same bits.

use super::{QK8_0, rd_f16};

// block_iq4_nl { ggml_half d; uint8_t qs[16]; }
pub(super) const IQ4_NL_BYTES: usize = 2 + QK8_0 / 2; // 18

pub(super) const KVALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[allow(clippy::needless_range_loop)] // indexes several arrays
pub(super) fn dequant_iq4_nl_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..18];
    for j in 0..16 {
        y[j] = d * KVALUES[(qs[j] & 0xf) as usize] as f32;
        y[j + 16] = d * KVALUES[(qs[j] >> 4) as usize] as f32;
    }
}

#[allow(clippy::needless_range_loop)] // indexes several arrays
#[inline(always)]
fn block_sum(b: &[u8; IQ4_NL_BYTES], x: &[i8; 32]) -> i32 {
    let mut s = 0i32;
    for j in 0..16 {
        s += KVALUES[(b[2 + j] & 0xf) as usize] as i32 * x[j] as i32;
        s += KVALUES[(b[2 + j] >> 4) as usize] as i32 * x[j + 16] as i32;
    }
    s
}

#[inline(always)]
pub(super) fn step(b: &[u8; IQ4_NL_BYTES], dx: f32, s: i32) -> f32 {
    rd_f16(b, 0) * dx * s as f32
}

pub(super) fn dot_iq4_nl_q8(row: &[u8], dx: &[f32], qx: &[[i8; 32]]) -> f32 {
    let mut sum = 0f32;
    for (b, (&d, x)) in row
        .as_chunks::<IQ4_NL_BYTES>()
        .0
        .iter()
        .zip(dx.iter().zip(qx))
    {
        sum += step(b, d, block_sum(b, x));
    }
    sum
}

#[cfg(target_arch = "x86_64")]
pub(super) mod x86 {
    use super::{IQ4_NL_BYTES, KVALUES, step};
    use std::arch::x86_64::*;

    /// The block's 32 signed weights (low nibbles, then high).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn weights(b: &[u8; IQ4_NL_BYTES]) -> __m256i {
        // SAFETY: 16 bytes each.
        let (qs, lut) = unsafe {
            (
                _mm_loadu_si128(b[2..].as_ptr().cast()),
                _mm_loadu_si128(KVALUES.as_ptr().cast()),
            )
        };
        let m4 = _mm_set1_epi8(0xf);
        let lo = _mm_shuffle_epi8(lut, _mm_and_si128(qs, m4));
        let hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(qs, 4), m4));
        _mm256_set_m128i(hi, lo)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn dot(w: __m256i, x: &[i8; 32]) -> i32 {
        // SAFETY: 32 bytes.
        let x = unsafe { _mm256_loadu_si256(x.as_ptr().cast()) };
        // |w| <= 127, |x| <= 127: pairs fit i16.
        let p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
        let v = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01));
        _mm_cvtsi128_si32(s)
    }

    /// # Safety
    /// The CPU has avx2, fma and f16c.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn iq4_nl_avx2(row: &[u8], dx: &[f32], qx: &[[i8; 32]]) -> f32 {
        let mut sum = 0f32;
        for (b, (&d, x)) in row
            .as_chunks::<IQ4_NL_BYTES>()
            .0
            .iter()
            .zip(dx.iter().zip(qx))
        {
            sum += step(b, d, dot(weights(b), x));
        }
        sum
    }

    /// Four tokens: each token's `iq4_nl_avx2`, bit for bit.
    ///
    /// # Safety
    /// As `iq4_nl_avx2`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn iq4_nl_avx2_x4(row: &[u8], dx: [&[f32]; 4], qx: [&[[i8; 32]]; 4]) -> [f32; 4] {
        let mut sum = [0f32; 4];
        for (i, b) in row.as_chunks::<IQ4_NL_BYTES>().0.iter().enumerate() {
            let w = weights(b);
            for t in 0..4 {
                sum[t] += step(b, dx[t][i], dot(w, &qx[t][i]));
            }
        }
        sum
    }
}
