//! MXFP4 (ggml GGML_TYPE_MXFP4, gpt-oss's expert matrices): 32-value blocks
//! of 4-bit E2M1 codes and one E8M0 (power-of-two) scale.
//!
//! `block_mxfp4 { uint8_t e; uint8_t qs[16]; }`: value j < 16 is the low
//! nibble of qs[j], value j + 16 its high nibble; a code is an index into
//! `KVALUES` (E2M1 doubled, so integers), and the block's scale is
//! 2^(e - 128) (`e8m0_half`: ggml's GGML_E8M0_TO_FP32_HALF, the half that
//! undoes the doubling). Dequant is `KVALUES[q] * e8m0_half(e)`, exactly
//! llama.cpp's `dequantize_row_mxfp4`.
//!
//! Dot against Q8_0 activations (`B32`): per block, eight exact i32 lane
//! sums (lane l = Σ w[i] x[i] over i in 4l..4l + 4, |w| <= 12, |x| <= 128),
//! then per lane `acc[l] = fma(e8m0_half(e) * dx, f32(lane), acc[l])`, and
//! at the end the eight lanes summed in `hsum8`'s tree, the order of the
//! AVX2 kernels' horizontal sum. The scalar path does exactly that with
//! `f32::mul_add`, so every path (scalar, AVX2, AVX-VNNI; one row, several
//! rows, several tokens) gives the same bits.
//!
//! The layout is the file's (17-byte blocks, rows contiguous, experts
//! contiguous in a 3-D tensor): no repack, so the CPU reads the mapping in
//! place and a GPU upload is the same bytes llama.cpp's CUDA kernels take.

use super::QK8_0;

pub(super) const MXFP4_BYTES: usize = 1 + QK8_0 / 2; // 17

/// E2M1 values, doubled (ggml kvalues_mxfp4).
pub(super) const KVALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// 2^(e - 128) as f32, bit for bit ggml_e8m0_to_fp32_half (subnormal for e < 2).
#[inline(always)]
pub(super) fn e8m0_half(e: u8) -> f32 {
    let bits = if e < 2 {
        0x0020_0000u32 << e
    } else {
        ((e as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

/// The 32 values of block `b` (17 bytes) into `y`.
pub(super) fn dequant_mxfp4_block(b: &[u8], y: &mut [f32]) {
    let d = e8m0_half(b[0]);
    for j in 0..16 {
        let q = b[1 + j];
        y[j] = KVALUES[(q & 0xf) as usize] as f32 * d;
        y[j + 16] = KVALUES[(q >> 4) as usize] as f32 * d;
    }
}

/// The eight lanes' sum in the AVX2 horizontal sum's order:
/// ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)).
#[inline(always)]
pub(super) fn hsum8(l: [f32; 8]) -> f32 {
    let r = [l[0] + l[4], l[1] + l[5], l[2] + l[6], l[3] + l[7]];
    (r[0] + r[2]) + (r[1] + r[3])
}

/// The scalar kernel (the definition the SIMD ones match bit for bit).
pub(super) fn dot_mxfp4_q8(row: &[u8], dx: &[f32], qx: &[[i8; 32]]) -> f32 {
    let mut acc = [0f32; 8];
    for (b, (&dxb, x)) in row
        .as_chunks::<MXFP4_BYTES>()
        .0
        .iter()
        .zip(dx.iter().zip(qx))
    {
        let mut w = [0i32; 32];
        for j in 0..16 {
            w[j] = KVALUES[(b[1 + j] & 0xf) as usize] as i32;
            w[j + 16] = KVALUES[(b[1 + j] >> 4) as usize] as i32;
        }
        let d = e8m0_half(b[0]) * dxb;
        for (l, a) in acc.iter_mut().enumerate() {
            let s: i32 = (4 * l..4 * l + 4).map(|i| w[i] * x[i] as i32).sum();
            *a = d.mul_add(s as f32, *a);
        }
    }
    hsum8(acc)
}

#[cfg(target_arch = "x86_64")]
pub(super) mod x86 {
    //! AVX2 (maddubs + madd) and AVX-VNNI (dpbusd) kernels: the same exact
    //! lane integers, so the same bits (module doc). One copy per feature
    //! set (`kernels!`), every helper compiled for it so all of it inlines.

    macro_rules! kernels {
        ($m:ident, $feat:literal, $dot:ident) => {
            pub mod $m {
                use super::super::{KVALUES, MXFP4_BYTES, e8m0_half};
                use std::arch::x86_64::*;

                /// The block's 32 signed weights (low nibbles, then high).
                #[inline]
                #[target_feature(enable = $feat)]
                fn weights(b: &[u8; MXFP4_BYTES], lut: __m128i) -> __m256i {
                    // SAFETY: 16 bytes from b[1..17].
                    let qs = unsafe { _mm_loadu_si128(b[1..].as_ptr().cast()) };
                    let m4 = _mm_set1_epi8(0xf);
                    let lo = _mm_shuffle_epi8(lut, _mm_and_si128(qs, m4));
                    let hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(qs, 4), m4));
                    _mm256_set_m128i(hi, lo)
                }

                #[inline]
                #[target_feature(enable = $feat)]
                fn lut() -> __m128i {
                    // SAFETY: 16 bytes.
                    unsafe { _mm_loadu_si128(KVALUES.as_ptr().cast()) }
                }

                /// Lane l = Σ w[i] x[i], i in 4l..4l + 4 (exact: |pair| <= 2 * 12 * 128).
                #[inline]
                #[target_feature(enable = $feat)]
                fn lanes(w: __m256i, x: &[i8; 32]) -> __m256i {
                    // SAFETY: 32 bytes.
                    let x = unsafe { _mm256_loadu_si256(x.as_ptr().cast()) };
                    let (ax, sy) = (_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
                    super::$dot(ax, sy)
                }

                /// The order of `super::super::hsum8`.
                #[inline]
                #[target_feature(enable = $feat)]
                fn hsum(x: __m256) -> f32 {
                    let mut r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
                    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
                    r = _mm_add_ss(r, _mm_movehdup_ps(r));
                    _mm_cvtss_f32(r)
                }

                #[inline]
                #[target_feature(enable = $feat)]
                fn prefetch(rows: &[u8], off: usize) {
                    if off < rows.len() {
                        // SAFETY: in bounds; a prefetch never faults anyway.
                        unsafe { _mm_prefetch::<_MM_HINT_T0>(rows.as_ptr().add(off).cast()) }
                    }
                }

                /// R rows (each `rb` bytes, consecutive in `rows`) against one
                /// token, prefetching `pf` bytes ahead of each row's current
                /// block (within `rows` only).
                ///
                /// # Safety
                /// The CPU has the enabled features.
                #[allow(clippy::needless_range_loop)] // r indexes several arrays
                #[target_feature(enable = $feat)]
                pub unsafe fn rows<const R: usize>(
                    rows: &[u8],
                    rb: usize,
                    dx: &[f32],
                    qx: &[[i8; 32]],
                    pf: usize,
                ) -> [f32; R] {
                    let n = rb / MXFP4_BYTES;
                    let (dx, qx) = (&dx[..n], &qx[..n]);
                    let mut blk: [&[[u8; MXFP4_BYTES]]; R] = [&[]; R];
                    for r in 0..R {
                        blk[r] = &rows[r * rb..(r + 1) * rb].as_chunks::<MXFP4_BYTES>().0[..n];
                    }
                    let lut = lut();
                    let mut acc = [_mm256_setzero_ps(); R];
                    for i in 0..n {
                        if pf != 0 && i % 4 == 0 {
                            for r in 0..R {
                                prefetch(rows, r * rb + i * MXFP4_BYTES + pf);
                                prefetch(rows, r * rb + i * MXFP4_BYTES + pf + 64);
                            }
                        }
                        for r in 0..R {
                            let b = &blk[r][i];
                            let d = _mm256_set1_ps(e8m0_half(b[0]) * dx[i]);
                            let s = _mm256_cvtepi32_ps(lanes(weights(b, lut), &qx[i]));
                            acc[r] = _mm256_fmadd_ps(d, s, acc[r]);
                        }
                    }
                    let mut out = [0f32; R];
                    for r in 0..R {
                        out[r] = hsum(acc[r]);
                    }
                    out
                }

                /// One row against four tokens, each the single-token bits.
                ///
                /// # Safety
                /// The CPU has the enabled features.
                #[allow(clippy::needless_range_loop)] // t indexes several arrays
                #[target_feature(enable = $feat)]
                pub unsafe fn x4(row: &[u8], dx: [&[f32]; 4], qx: [&[[i8; 32]]; 4]) -> [f32; 4] {
                    let blocks = row.as_chunks::<MXFP4_BYTES>().0;
                    let n = blocks.len();
                    let dx = [&dx[0][..n], &dx[1][..n], &dx[2][..n], &dx[3][..n]];
                    let qx = [&qx[0][..n], &qx[1][..n], &qx[2][..n], &qx[3][..n]];
                    let lut = lut();
                    let mut acc = [_mm256_setzero_ps(); 4];
                    for (i, b) in blocks.iter().enumerate() {
                        let w = weights(b, lut);
                        let e = e8m0_half(b[0]);
                        for t in 0..4 {
                            let d = _mm256_set1_ps(e * dx[t][i]);
                            let s = _mm256_cvtepi32_ps(lanes(w, &qx[t][i]));
                            acc[t] = _mm256_fmadd_ps(d, s, acc[t]);
                        }
                    }
                    [hsum(acc[0]), hsum(acc[1]), hsum(acc[2]), hsum(acc[3])]
                }
            }
        };
    }

    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn dot_avx2(ax: __m256i, sy: __m256i) -> __m256i {
        _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), _mm256_set1_epi16(1))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    fn dot_vnni(ax: __m256i, sy: __m256i) -> __m256i {
        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy)
    }

    kernels!(avx2, "avx2,fma,f16c", dot_avx2);
    kernels!(vnni, "avx2,fma,f16c,avxvnni", dot_vnni);
}

#[cfg(test)]
mod tests {
    use super::super::{
        QuantizedAct, Simd, dequant, dot_row_q8_multi_with, dot_row_q8_with,
        dot_rows_q8_multi_with, dot_rows_q8_with,
    };
    use super::*;

    /// FNV-1a 64 over the f32 bits (the hash `mxdq` prints).
    fn fnv(y: &[f32], mut h: u64) -> u64 {
        for v in y {
            for b in v.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        h
    }
    const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

    /// Every exponent with every code byte, dequantized bit for bit as
    /// llama.cpp 4413a19's `dequantize_row_mxfp4` (ggml type traits; the
    /// hash from the scratch tool `mxdq synth`, docs/log/2026-10-18-gpt-oss-s0.md).
    #[test]
    fn dequant_matches_llama_cpp_bits() {
        let mut h = FNV0;
        let mut y = [0f32; 32];
        for e in 0..=255u8 {
            for j in 0..16u8 {
                let mut b = [0u8; MXFP4_BYTES];
                b[0] = e;
                for i in 0..16u8 {
                    b[1 + i as usize] = 16 * j + i;
                }
                dequant(super::super::MXFP4, &b, &mut y);
                if e == 127 && j == 1 {
                    let want = [
                        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5,
                    ];
                    assert_eq!(&y[..12], &want);
                    assert_eq!(y[16..], [0.5; 16]);
                }
                h = fnv(&y, h);
            }
        }
        assert_eq!(h, 0x7c32_d88a_05ac_3625);
        assert_eq!(e8m0_half(0).to_bits(), 0x0020_0000);
        assert_eq!(e8m0_half(1).to_bits(), 0x0040_0000);
        assert_eq!(e8m0_half(128), 1.0);
    }

    /// The real file's experts dequantized == llama.cpp's (`mxdq MODEL
    /// TENSOR`); optional: skipped without the gpt-oss model.
    #[test]
    fn dequant_matches_llama_cpp_on_gpt_oss() {
        let Some(m) = crate::testutil::gptoss_model() else {
            return;
        };
        for (name, want) in [
            ("blk.0.ffn_down_exps.weight", 0xd50e_53c7_a7e3_95c5u64),
            ("blk.23.ffn_gate_exps.weight", 0x35ad_7bdd_ff77_2387),
        ] {
            let t = m.tensors.iter().find(|t| t.name == name).unwrap();
            assert_eq!(t.qtype.id, super::super::MXFP4);
            let data = m.tensor_data(t);
            let mut h = FNV0;
            let mut y = vec![0f32; 32 * 4096];
            for c in data.chunks(MXFP4_BYTES * 4096) {
                let y = &mut y[..c.len() / MXFP4_BYTES * 32];
                dequant(super::super::MXFP4, c, y);
                h = fnv(y, h);
            }
            assert_eq!(h, want, "{name}");
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn u(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn f(&mut self) -> f32 {
            self.u() as f32 / (1u64 << 30) as f32 - 1.0
        }
    }

    /// Rows of random codes, exponents near 2^-8..2^0 (gpt-oss's range), a
    /// few extreme blocks (all ±12).
    fn rows(n: usize, nr: usize, rng: &mut Lcg) -> Vec<u8> {
        let mut v = Vec::with_capacity(nr * n / 32 * MXFP4_BYTES);
        for b in 0..nr * n / 32 {
            v.push(118 + (rng.u() % 12) as u8);
            for _ in 0..16 {
                v.push(match b % 7 {
                    3 => 0x77,
                    5 => 0xff,
                    _ => rng.u() as u8,
                });
            }
        }
        v
    }

    fn paths() -> Vec<Simd> {
        #[allow(unused_mut)]
        let mut v = vec![Simd::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if super::super::has_avx2() {
                v.push(Simd::Avx2);
            }
            if super::super::has_vnni() {
                v.push(Simd::Vnni);
            }
        }
        v
    }

    /// Every path and entry point (one row, decode rows with and without
    /// prefetch, 4-token and 2-row batches) gives the scalar kernel's bits;
    /// the scalar kernel is the exact dot of the dequantized values to f32
    /// rounding.
    #[test]
    fn every_path_same_bits() {
        let qt = super::super::MXFP4;
        let mut rng = Lcg(0x6f55);
        for n in [32usize, 64, 2880] {
            let nr = 9;
            let w = rows(n, nr, &mut rng);
            let rb = n / 32 * MXFP4_BYTES;
            let xs: Vec<Vec<f32>> = (0..6)
                .map(|t| (0..n).map(|_| rng.f() * [1.0, 30.0, 0.01][t % 3]).collect())
                .collect();
            let qa: Vec<QuantizedAct> = xs
                .iter()
                .map(|x| QuantizedAct::for_types(x, &[qt]))
                .collect();
            let qr: Vec<&QuantizedAct> = qa.iter().collect();
            // SAFETY: Scalar needs nothing.
            let want: Vec<Vec<f32>> = (0..nr)
                .map(|r| {
                    qr.iter()
                        .map(|x| unsafe { dot_row_q8_with(Simd::Scalar, qt, &w[r * rb..], x) })
                        .collect()
                })
                .collect();
            // The scalar kernel vs f64 over the dequantized operands.
            let mut wd = vec![0f32; n];
            for r in 0..nr {
                dequant(qt, &w[r * rb..(r + 1) * rb], &mut wd);
                for (t, x) in qa.iter().enumerate() {
                    let b = x.b32.as_ref().unwrap();
                    let (mut e, mut mag) = (0f64, 0f64);
                    for (i, &wv) in wd.iter().enumerate() {
                        let xv = b.d[i / 32] as f64 * b.q[i / 32][i % 32] as f64;
                        e += wv as f64 * xv;
                        mag += (wv as f64 * xv).abs();
                    }
                    let got = want[r][t] as f64;
                    assert!(
                        (got - e).abs() <= 1e-5 * mag + 1e-30,
                        "n {n} r {r} t {t}: {got} vs {e}"
                    );
                }
            }
            for p in paths() {
                for pf in [0usize, 4096] {
                    for max_rows in [1usize, 4] {
                        for (t, x) in qr.iter().enumerate() {
                            let mut out = vec![0f32; nr];
                            // SAFETY: a detected path.
                            unsafe { dot_rows_q8_with(p, qt, &w, x, &mut out, pf, max_rows) };
                            for r in 0..nr {
                                assert_eq!(
                                    out[r].to_bits(),
                                    want[r][t].to_bits(),
                                    "{p:?} rows n {n}"
                                );
                            }
                        }
                    }
                }
                for r in 0..nr {
                    let mut out = vec![0f32; qr.len()];
                    // SAFETY: as above.
                    unsafe { dot_row_q8_multi_with(p, qt, &w[r * rb..], &qr, &mut out) };
                    for t in 0..qr.len() {
                        assert_eq!(out[t].to_bits(), want[r][t].to_bits(), "{p:?} multi n {n}");
                    }
                }
                let (mut o0, mut o1) = (vec![0f32; qr.len()], vec![0f32; qr.len()]);
                // SAFETY: as above.
                unsafe {
                    dot_rows_q8_multi_with(
                        p,
                        [qt, qt],
                        [&w[..rb], &w[rb..2 * rb]],
                        &qr,
                        [&mut o0, &mut o1],
                    )
                };
                for t in 0..qr.len() {
                    assert_eq!(o0[t].to_bits(), want[0][t].to_bits(), "{p:?} pair");
                    assert_eq!(o1[t].to_bits(), want[1][t].to_bits(), "{p:?} pair");
                }
            }
        }
    }
}
