//! Q2_K and Q3_K: dequant (ggml's dequantize_row_q2_K / q3_K, bit for bit)
//! and the integer dots against Q8_K activations.
//!
//! Both are 16 sub-blocks of 16 values; value e = 16 g + l of a block (g
//! the sub-block) has its 2 low bits at bits 2j of qs[32 n + 16 h + l],
//! with g = 8 n + 2 j + h. Q3_K's third bit is bit 4 n + j of hmask[16 h + l]
//! (set: q, clear: q - 4).
//!
//! The dots compute exact integers per block and then a fixed few float
//! ops (`q2_k_step` / `q3_k_step`), so the scalar and the AVX2 kernels, one
//! token or four, give the same bits:
//!   Q2_K: S = Σ_g sc_g Σ q x, M = Σ_g m_g bsum_g, acc += dx (d S - dmin M)
//!   Q3_K: S = Σ_g sc_g Σ q x (q in -4..3, sc in -32..31), acc += dx d S
//! (Q4_K's and Q6_K's per-block forms; the GPU's canonical order is in
//! decode.cu "Row dot products".)

use super::{KBlock, QK_K, rd_f16};

// block_q2_K { u8 scales[16]; u8 qs[64]; ggml_half d; ggml_half dmin; }
pub(super) const Q2_K_BYTES: usize = QK_K / 16 + QK_K / 4 + 4; // 84
// block_q3_K { u8 hmask[32]; u8 qs[64]; u8 scales[12]; ggml_half d; }
pub(super) const Q3_K_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + 2; // 110

/// Sub-block g's (byte offset into qs, shift): qs[32 n + 16 h ..], 2 j.
#[inline(always)]
fn place(g: usize) -> (usize, usize) {
    let (n, j, h) = (g / 8, (g % 8) / 2, g % 2);
    (32 * n + 16 * h, 2 * j)
}

/// Q3_K's 16 scales, 6-bit (ggml's aux unpacking), minus 32.
#[inline(always)]
pub(super) fn q3_k_scales(s: &[u8]) -> [i8; 16] {
    let mut out = [0i8; 16];
    for (i, o) in out.iter_mut().enumerate() {
        let lo = if i < 8 { s[i] & 0xf } else { s[i - 8] >> 4 };
        let hi = (s[8 + i % 4] >> (2 * (i / 4))) & 3;
        *o = (lo | (hi << 4)) as i8 - 32;
    }
    out
}

#[allow(clippy::needless_range_loop)] // indexes several arrays
pub(super) fn dequant_q2_k_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 80);
    let min = rd_f16(b, 82);
    let qs = &b[16..80];
    for g in 0..16 {
        let sc = b[g];
        let dl = d * (sc & 0xf) as f32;
        let ml = min * (sc >> 4) as f32;
        let (o, sh) = place(g);
        for l in 0..16 {
            y[16 * g + l] = dl * ((qs[o + l] >> sh) & 3) as f32 - ml;
        }
    }
}

#[allow(clippy::needless_range_loop)] // indexes several arrays
pub(super) fn dequant_q3_k_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 108);
    let hm = &b[0..32];
    let qs = &b[32..96];
    let sc = q3_k_scales(&b[96..108]);
    for g in 0..16 {
        let dl = d * sc[g] as f32;
        let (o, sh) = place(g);
        let hb = 4 * (g / 8) + (g % 8) / 2;
        let h = (g % 2) * 16;
        for l in 0..16 {
            let q = ((qs[o + l] >> sh) & 3) as i32 - if (hm[h + l] >> hb) & 1 != 0 { 0 } else { 4 };
            y[16 * g + l] = dl * q as f32;
        }
    }
}

/// The block's float step (every path's).
#[inline(always)]
pub(super) fn q2_k_step(b: &[u8; Q2_K_BYTES], dx: f32, sumi: i32, summ: i32) -> f32 {
    let d = rd_f16(b, 80);
    let dmin = rd_f16(b, 82);
    dx * (d * sumi as f32 - dmin * summ as f32)
}

#[inline(always)]
pub(super) fn q3_k_step(b: &[u8; Q3_K_BYTES], dx: f32, sumi: i32) -> f32 {
    dx * rd_f16(b, 108) * sumi as f32
}

/// (S, M) of one Q2_K block.
#[allow(clippy::needless_range_loop)] // g indexes several arrays
#[inline(always)]
fn q2_k_sums(b: &[u8; Q2_K_BYTES], a: &KBlock) -> (i32, i32) {
    let qs = &b[16..80];
    let (mut sumi, mut summ) = (0i32, 0i32);
    for g in 0..16 {
        let (o, sh) = place(g);
        let mut s = 0i32;
        for l in 0..16 {
            s += ((qs[o + l] >> sh) & 3) as i32 * a.qs[16 * g + l] as i32;
        }
        sumi += (b[g] & 0xf) as i32 * s;
        summ += (b[g] >> 4) as i32 * a.bsums[g] as i32;
    }
    (sumi, summ)
}

/// S of one Q3_K block.
#[allow(clippy::needless_range_loop)] // g indexes several arrays
#[inline(always)]
fn q3_k_sums(b: &[u8; Q3_K_BYTES], a: &KBlock) -> i32 {
    let hm = &b[0..32];
    let qs = &b[32..96];
    let sc = q3_k_scales(&b[96..108]);
    let mut sumi = 0i32;
    for g in 0..16 {
        let (o, sh) = place(g);
        let hb = 4 * (g / 8) + (g % 8) / 2;
        let h = (g % 2) * 16;
        let mut s = 0i32;
        for l in 0..16 {
            let q = ((qs[o + l] >> sh) & 3) as i32 - 4 * (1 - ((hm[h + l] >> hb) & 1) as i32);
            s += q * a.qs[16 * g + l] as i32;
        }
        sumi += sc[g] as i32 * s;
    }
    sumi
}

pub(super) fn dot_q2_k_q8(row: &[u8], xk: &[KBlock]) -> f32 {
    let mut sum = 0f32;
    for (b, a) in row.as_chunks::<Q2_K_BYTES>().0.iter().zip(xk) {
        let (s, m) = q2_k_sums(b, a);
        sum += q2_k_step(b, a.d, s, m);
    }
    sum
}

pub(super) fn dot_q3_k_q8(row: &[u8], xk: &[KBlock]) -> f32 {
    let mut sum = 0f32;
    for (b, a) in row.as_chunks::<Q3_K_BYTES>().0.iter().zip(xk) {
        sum += q3_k_step(b, a.d, q3_k_sums(b, a));
    }
    sum
}

#[cfg(target_arch = "x86_64")]
pub(super) mod x86 {
    //! AVX2: the same integers as the scalar sums (maddubs pairs < 2^15,
    //! madd by the sub-block scales into i32), the same float steps.
    use super::{KBlock, Q2_K_BYTES, Q3_K_BYTES, q2_k_step, q3_k_step};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn ld32(p: &[u8]) -> __m256i {
        assert!(p.len() >= 32);
        // SAFETY: 32 readable bytes; loadu has no alignment requirement.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn ld16(p: &[u8]) -> __m128i {
        assert!(p.len() >= 16);
        // SAFETY: as above, 16 bytes.
        unsafe { _mm_loadu_si128(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn hsum_i32x8(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01));
        _mm_cvtsi128_si32(s)
    }

    /// i8 scales of sub-blocks 2i, 2i + 1, each repeated 8 times, as i16.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn pair_scale(scales: __m128i, i: usize) -> __m256i {
        let lo = (2 * i) as i64 * 0x0101010101010101;
        let hi = (2 * i + 1) as i64 * 0x0101010101010101;
        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, _mm_set_epi64x(hi, lo)))
    }

    /// The 8 chunks (n, j) of 32 values of a Q2_K / Q3_K block (unsigned
    /// 2-bit values, sub-blocks 8n + 2j and + 1), chunk 4n + j.
    #[inline(always)]
    fn low2(qs: &[u8]) -> [__m256i; 8] {
        // SAFETY (intrinsics): the callers' avx2 target feature.
        unsafe {
            let m3 = _mm256_set1_epi8(3);
            let q = [ld32(&qs[0..32]), ld32(&qs[32..64])];
            std::array::from_fn(|c| {
                let (n, j) = (c / 4, c % 4);
                _mm256_and_si256(_mm256_srl_epi16(q[n], _mm_cvtsi32_si128(2 * j as i32)), m3)
            })
        }
    }

    /// T tokens' (S, M) of a Q2_K block.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn q2_k_sums<const T: usize>(b: &[u8; Q2_K_BYTES], a: [&KBlock; T]) -> [(i32, i32); T] {
        let sb = ld16(&b[0..16]);
        let sc = _mm_and_si128(sb, _mm_set1_epi8(0xf));
        let mn = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sb, 4), _mm_set1_epi8(0xf)));
        let w = low2(&b[16..80]);
        let s: [__m256i; 8] = std::array::from_fn(|c| pair_scale(sc, c));
        std::array::from_fn(|t| {
            // SAFETY: bsums is 16 i16 = 32 bytes.
            let bs = unsafe { _mm256_loadu_si256(a[t].bsums.as_ptr().cast()) };
            let summ = hsum_i32x8(_mm256_madd_epi16(mn, bs));
            let mut acc = _mm256_setzero_si256();
            for c in 0..8 {
                // SAFETY: 32 bytes of the 256.
                let x = unsafe { _mm256_loadu_si256(a[t].qs[32 * c..].as_ptr().cast()) };
                let p = _mm256_maddubs_epi16(w[c], x);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s[c], p));
            }
            (hsum_i32x8(acc), summ)
        })
    }

    /// `q3_k_scales` as 16 i8 lanes (ggml's aux word unpacking).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn q3_k_scales_v(s: &[u8]) -> __m128i {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let w = |i: usize| u32::from_le_bytes([s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]]);
        let (a0, a1, t) = (w(0), w(1), w(2));
        let r2 = ((a0 >> 4) & KMASK2) | (((t >> 4) & KMASK1) << 4);
        let r3 = ((a1 >> 4) & KMASK2) | (((t >> 6) & KMASK1) << 4);
        let r0 = (a0 & KMASK2) | ((t & KMASK1) << 4);
        let r1 = (a1 & KMASK2) | (((t >> 2) & KMASK1) << 4);
        _mm_sub_epi8(
            _mm_set_epi32(r3 as i32, r2 as i32, r1 as i32, r0 as i32),
            _mm_set1_epi8(32),
        )
    }

    /// T tokens' S of a Q3_K block.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn q3_k_sums<const T: usize>(b: &[u8; Q3_K_BYTES], a: [&KBlock; T]) -> [i32; T] {
        let sc = q3_k_scales_v(&b[96..108]);
        let lo = low2(&b[32..96]);
        let hm = ld32(&b[0..32]);
        let m1 = _mm256_set1_epi8(1);
        let four = _mm256_set1_epi8(4);
        // Signed values q - 4 (bit clear) as |w| and the sign applied to x.
        let w: [__m256i; 8] = std::array::from_fn(|c| {
            let h = _mm256_and_si256(_mm256_srl_epi16(hm, _mm_cvtsi32_si128(c as i32)), m1);
            _mm256_sub_epi8(_mm256_or_si256(lo[c], _mm256_slli_epi16(h, 2)), four)
        });
        let aw: [__m256i; 8] = std::array::from_fn(|c| _mm256_sign_epi8(w[c], w[c]));
        let s: [__m256i; 8] = std::array::from_fn(|c| pair_scale(sc, c));
        std::array::from_fn(|t| {
            let mut acc = _mm256_setzero_si256();
            for c in 0..8 {
                // SAFETY: 32 bytes of the 256.
                let x = unsafe { _mm256_loadu_si256(a[t].qs[32 * c..].as_ptr().cast()) };
                let p = _mm256_maddubs_epi16(aw[c], _mm256_sign_epi8(x, w[c]));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s[c], p));
            }
            hsum_i32x8(acc)
        })
    }

    /// # Safety
    /// The CPU has avx2, fma and f16c.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q2_k_avx2(row: &[u8], xk: &[KBlock]) -> f32 {
        let mut sum = 0f32;
        for (b, a) in row.as_chunks::<Q2_K_BYTES>().0.iter().zip(xk) {
            let [(s, m)] = q2_k_sums(b, [a]);
            sum += q2_k_step(b, a.d, s, m);
        }
        sum
    }

    /// # Safety
    /// As `q2_k_avx2`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q3_k_avx2(row: &[u8], xk: &[KBlock]) -> f32 {
        let mut sum = 0f32;
        for (b, a) in row.as_chunks::<Q3_K_BYTES>().0.iter().zip(xk) {
            let [s] = q3_k_sums(b, [a]);
            sum += q3_k_step(b, a.d, s);
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn prefetch(rows: &[u8], off: usize) {
        if off < rows.len() {
            // SAFETY: off is in bounds; a prefetch never faults anyway.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(rows.as_ptr().add(off).cast()) }
        }
    }

    /// Decode rows: `out[r]` = `q2_k_avx2` of row r (rows of `rb` bytes),
    /// bit for bit, prefetching `pf` bytes ahead.
    ///
    /// # Safety
    /// As `q2_k_avx2`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q2_k_rows(rows: &[u8], rb: usize, xk: &[KBlock], out: &mut [f32], pf: usize) {
        for (r, o) in out.iter_mut().enumerate() {
            let row = &rows[r * rb..(r + 1) * rb];
            let mut sum = 0f32;
            for (i, (b, a)) in row.as_chunks::<Q2_K_BYTES>().0.iter().zip(xk).enumerate() {
                if pf != 0 {
                    let off = r * rb + i * Q2_K_BYTES + pf;
                    prefetch(rows, off);
                    prefetch(rows, off + 64);
                }
                let [(s, m)] = q2_k_sums(b, [a]);
                sum += q2_k_step(b, a.d, s, m);
            }
            *o = sum;
        }
    }

    /// Decode rows: `out[r]` = `q3_k_avx2` of row r, bit for bit.
    ///
    /// # Safety
    /// As `q2_k_avx2`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q3_k_rows(rows: &[u8], rb: usize, xk: &[KBlock], out: &mut [f32], pf: usize) {
        for (r, o) in out.iter_mut().enumerate() {
            let row = &rows[r * rb..(r + 1) * rb];
            let mut sum = 0f32;
            for (i, (b, a)) in row.as_chunks::<Q3_K_BYTES>().0.iter().zip(xk).enumerate() {
                if pf != 0 {
                    let off = r * rb + i * Q3_K_BYTES + pf;
                    prefetch(rows, off);
                    prefetch(rows, off + 64);
                }
                let [s] = q3_k_sums(b, [a]);
                sum += q3_k_step(b, a.d, s);
            }
            *o = sum;
        }
    }

    /// Four tokens: each token's `q2_k_avx2`, bit for bit.
    ///
    /// # Safety
    /// As `q2_k_avx2`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q2_k_avx2_x4(row: &[u8], xk: [&[KBlock]; 4]) -> [f32; 4] {
        let mut sum = [0f32; 4];
        for (i, b) in row.as_chunks::<Q2_K_BYTES>().0.iter().enumerate() {
            let r = q2_k_sums(b, [&xk[0][i], &xk[1][i], &xk[2][i], &xk[3][i]]);
            for t in 0..4 {
                sum[t] += q2_k_step(b, xk[t][i].d, r[t].0, r[t].1);
            }
        }
        sum
    }

    /// Four tokens: each token's `q3_k_avx2`, bit for bit.
    ///
    /// # Safety
    /// As `q2_k_avx2`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q3_k_avx2_x4(row: &[u8], xk: [&[KBlock]; 4]) -> [f32; 4] {
        let mut sum = [0f32; 4];
        for (i, b) in row.as_chunks::<Q3_K_BYTES>().0.iter().enumerate() {
            let r = q3_k_sums(b, [&xk[0][i], &xk[1][i], &xk[2][i], &xk[3][i]]);
            for t in 0..4 {
                sum[t] += q3_k_step(b, xk[t][i].d, r[t]);
            }
        }
        sum
    }
}
