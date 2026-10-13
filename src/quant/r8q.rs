//! 8-row interleaved ("r8q") Q5_0 / Q8_0 weights with integer-first kernels.
//!
//! NUMERICS PROPOSAL, like r8.rs: one f32 step per (row, 32-block), in block
//! order, no FMA, on every path (scalar, AVX2, AVX-VNNI; GEMV and GEMM), so
//! they all agree bit for bit. That step is exactly the per-row *scalar*
//! kernels' (`dot_q5_0_q8` / `dot_q8_0_q8`: sum += (d_w * d_a) * f32(isum)),
//! so r8q also equals today's Scalar path bit for bit; today's AVX2/VNNI
//! per-row kernels (8 f32 lanes per row + fmadd + hsum) differ in low bits.
//!
//! Layout. A matrix with rows % 8 == 0 becomes rows / 8 groups; a group is
//! its 32-blocks in order. Both types store u, the unsigned quant, in VNNI4
//! order (one i32 lane = 4 consecutive k of one row r, bytes 4r..4r+4):
//!   Q5_0: u = nibble | high bit << 4 (0..31), w = u - 16
//!   Q8_0: u = w ^ 0x80 = w + 128 (0..255)
//! Q8_0 block, 272 bytes (= 8 x 34, the original size):
//!   [0..16)    d, f16 x 8 (row r at 2r): one vcvtph2ps
//!   [16..272)  vector v (0..8) at 16 + 32v, byte [4r + i] = u_r[4v + i]
//! Q5_0 block, 176 bytes (= 8 x 22, the original size):
//!   [0..16)    d, f16 x 8
//!   [16..144)  nibble vector c (0..4) at 16 + 32c, byte [4r + i] =
//!              qs_r[4c + i] (low nibble: k = 4c + i, high: k = 16 + 4c + i)
//!   [144..176) H, byte [4r + i] bit v = high bit of k = 4v + i
//!   Vector v of u (as Q8_0's): N = nibble vector v % 4,
//!   (v < 4 ? N & 15 : N >> 4 & 15) | (H shifted by 4 - v) & 0x10:
//!   4-5 ALU ops (16-bit shifts never move bit 4 across a byte here).
//!   The GEMM unpacks a whole group once into the Q8_0-shaped form (scratch,
//!   L1/L2) and runs the same tile as Q8_0; the GEMV unpacks inline.
//!
//! Integer identity per (row r, block) with activation block a (Q8_0, S = Σ a):
//!   isum_r = Σ_k w_r[k] a[k] = Σ_k u_r[k] a[k] - OFF * S,  OFF = 16 | 128
//! VNNI starts the dpbusd chain at -OFF * S and adds 8 dpbusd(u_v, bcast4(a)).
//! AVX2 Q5_0: same with maddubs/madd (pairs 2*31*127 = 7874, no saturation).
//! AVX2 Q8_0: u + 128 pairs would saturate maddubs (2*255*127), so it uses
//! w = u ^ 0x80 and maddubs(|w|, sign(a, w)) (2*128*127 = 32512) from 0.
//! All give the exact Σ w a: |a| <= 127, |isum| <= 32*128*127 = 520192,
//! |Σ u a| <= 32*255*128 = 1.04M; dpbusd doesn't saturate. Then
//!   acc_r += (d_r * d_a) * f32(isum_r)     (f32, block order, mul/mul/add)

use super::{B32, Q5_0, Q5_0_BYTES, Q8_0, Q8_0_BYTES, QK8_0, QuantizedAct, Simd, f16_to_f32};

pub const R8Q5: usize = 176;
pub const R8Q8: usize = 272;
const QV: usize = 16;
const Q5_H: usize = 144;

/// Bytes of one 8-row block of `qtype_id`.
pub fn block_bytes(qtype_id: u32) -> usize {
    match qtype_id {
        Q5_0 => R8Q5,
        Q8_0 => R8Q8,
        _ => panic!("r8q: Q5_0/Q8_0 only"),
    }
}

/// Whether a `rows` x `n` matrix of `qtype_id` can be repacked.
pub fn can_repack(qtype_id: u32, rows: usize, n: usize) -> bool {
    matches!(qtype_id, Q5_0 | Q8_0) && rows.is_multiple_of(8) && n.is_multiple_of(QK8_0) && n > 0
}

/// Bytes of one 8-row group (all of its blocks).
pub fn group_bytes(qtype_id: u32, n: usize) -> usize {
    block_bytes(qtype_id) * (n / QK8_0)
}

/// Bytes of the r8q form of a `rows` x `n` matrix (= the original's).
pub fn repacked_len(qtype_id: u32, rows: usize, n: usize) -> usize {
    rows / 8 * group_bytes(qtype_id, n)
}

/// Repack a `rows` x `n` matrix (`can_repack` must hold) into r8q groups.
#[cfg_attr(not(test), allow(dead_code))]
pub fn repack(qtype_id: u32, data: &[u8], rows: usize, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; repacked_len(qtype_id, rows, n)];
    repack_into(qtype_id, data, rows, n, &mut out);
    out
}

/// One row's original block `src` into row r of the 8-row block `o`.
fn repack_row(qtype_id: u32, src: &[u8], r: usize, o: &mut [u8]) {
    o[2 * r..2 * r + 2].copy_from_slice(&src[0..2]);
    if qtype_id == Q5_0 {
        for c in 0..4 {
            let p = QV + 32 * c + 4 * r;
            o[p..p + 4].copy_from_slice(&src[6 + 4 * c..10 + 4 * c]);
        }
        let qh = u32::from_le_bytes([src[2], src[3], src[4], src[5]]);
        for i in 0..4 {
            let mut h = 0u8;
            for v in 0..8 {
                h |= (((qh >> (4 * v + i)) & 1) as u8) << v;
            }
            o[Q5_H + 4 * r + i] = h;
        }
    } else {
        for k in 0..32 {
            o[QV + 32 * (k / 4) + 4 * r + k % 4] = src[2 + k] ^ 0x80;
        }
    }
}

/// `repack` into `out` (`repacked_len` bytes).
pub fn repack_into(qtype_id: u32, data: &[u8], rows: usize, n: usize, out: &mut [u8]) {
    assert!(can_repack(qtype_id, rows, n), "r8q: can't repack");
    let sb = if qtype_id == Q5_0 {
        Q5_0_BYTES
    } else {
        Q8_0_BYTES
    };
    let bb = block_bytes(qtype_id);
    let rb = super::row_bytes(qtype_id, n);
    assert!(data.len() >= rows * rb, "r8q: short tensor data");
    let gb = group_bytes(qtype_id, n);
    assert_eq!(out.len(), rows / 8 * gb, "r8q: output size");
    for (grp, og) in out.chunks_exact_mut(gb).enumerate() {
        for (blk, o) in og.chunks_exact_mut(bb).enumerate() {
            for r in 0..8 {
                let src = &data[(8 * grp + r) * rb + blk * sb..][..sb];
                repack_row(qtype_id, src, r, o);
            }
        }
    }
}

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// Unsigned quant of row r at k (0..32) of an r8q block.
#[inline]
fn u_at(qtype_id: u32, b: &[u8], r: usize, k: usize) -> u8 {
    let (v, i) = (k / 4, k % 4);
    if qtype_id == Q5_0 {
        let nb = b[QV + 32 * (v % 4) + 4 * r + i];
        let nib = if v < 4 { nb & 15 } else { nb >> 4 };
        nib | ((b[Q5_H + 4 * r + i] >> v) & 1) << 4
    } else {
        b[QV + 32 * v + 4 * r + i]
    }
}

fn offset(qtype_id: u32) -> i32 {
    if qtype_id == Q5_0 { 16 } else { 128 }
}

/// Dequantize row r (0..8) of an r8q group into `out` (n values), with
/// quant.rs's dequant_q5_0_block / dequant_q8_0_block float formula.
#[cfg_attr(not(test), allow(dead_code))]
pub fn dequant_row(qtype_id: u32, group: &[u8], r: usize, out: &mut [f32]) {
    let off = offset(qtype_id);
    for (b, y) in group
        .chunks_exact(block_bytes(qtype_id))
        .zip(out.as_chunks_mut::<QK8_0>().0)
    {
        let d = f16_at(b, 2 * r);
        for (k, yk) in y.iter_mut().enumerate() {
            *yk = (u_at(qtype_id, b, r, k) as i32 - off) as f32 * d;
        }
    }
}

// ---------------------------------------------------------------------------
// Activations: the Q8_0 view plus the i32 start of each block's chain.

struct Act<'a> {
    d: &'a [f32],
    q: &'a [[i8; 32]],
    /// per block: -OFF * Σ a (0 for the AVX2 Q8_0 sign path)
    s0: Vec<i32>,
}

fn act<'a>(x: &'a QuantizedAct, off: i32) -> Act<'a> {
    let B32 { d, q, .. } = x.b32.as_ref().unwrap_or_else(|| {
        panic!(
            "r8q: needs input len multiple of 32, got {} (or the 32-block view wasn't built)",
            x.len()
        )
    });
    let s0 = q
        .iter()
        .map(|b| -off * b.iter().map(|&v| v as i32).sum::<i32>())
        .collect();
    Act { d, q, s0 }
}

/// The chain offset `simd` uses for `qtype_id`.
fn chain_offset(simd: Simd, qtype_id: u32) -> i32 {
    match (simd, qtype_id) {
        #[cfg(target_arch = "x86_64")]
        (Simd::Avx2, Q8_0) => 0,
        _ => offset(qtype_id),
    }
}

// ---------------------------------------------------------------------------
// Scalar reference.

/// One r8q group x one token: out[r] for the group's 8 rows.
#[allow(clippy::needless_range_loop)]
fn group_scalar(qtype_id: u32, group: &[u8], x: &Act, out: &mut [f32; 8]) {
    let mut acc = [0f32; 8];
    for (i, b) in group.chunks_exact(block_bytes(qtype_id)).enumerate() {
        for r in 0..8 {
            let mut isum = x.s0[i];
            for k in 0..32 {
                isum += u_at(qtype_id, b, r, k) as i32 * x.q[i][k] as i32;
            }
            acc[r] += (f16_at(b, 2 * r) * x.d[i]) * isum as f32;
        }
    }
    *out = acc;
}

// ---------------------------------------------------------------------------
// Dispatch.

/// out[8g + r] = row r of group g of `groups` (whole r8q groups of
/// `qtype_id`, width x.len()) · x, for the first out.len() / 8 groups.
/// `groups` may run on past them: software prefetch (`pf` bytes ahead, 0 =
/// off) reads up to its end and never beyond, so callers bound it to the
/// rows they own (as r8::gemv).
pub fn gemv(qtype_id: u32, groups: &[u8], x: &QuantizedAct, out: &mut [f32], pf: usize) {
    // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
    unsafe { gemv_with(super::simd(), qtype_id, groups, x, out, pf) }
}

/// `gemv` on an explicit path.
///
/// # Safety
/// As for quant.rs's `dot_row_q8_with`.
pub(super) unsafe fn gemv_with(
    simd: Simd,
    qtype_id: u32,
    groups: &[u8],
    x: &QuantizedAct,
    out: &mut [f32],
    pf: usize,
) {
    let xa = act(x, chain_offset(simd, qtype_id));
    let gb = group_bytes(qtype_id, x.len());
    let ng = out.len() / 8;
    assert!(
        out.len() == 8 * ng && groups.len() >= ng * gb,
        "r8q gemv: sizes"
    );
    // SAFETY (below): guaranteed by our caller.
    match (simd, qtype_id) {
        (Simd::Scalar, _) => {
            for (g, o) in groups.chunks_exact(gb).zip(out.as_chunks_mut::<8>().0) {
                group_scalar(qtype_id, g, &xa, o);
            }
        }
        #[cfg(target_arch = "x86_64")]
        (Simd::Avx2, Q5_0) => unsafe { x86::avx2::gemv_q5(groups, gb, &xa, out, pf) },
        #[cfg(target_arch = "x86_64")]
        (Simd::Avx2, _) => unsafe { x86::avx2::gemv::<true>(groups, gb, &xa, out, pf) },
        #[cfg(target_arch = "x86_64")]
        (Simd::Vnni, Q5_0) => unsafe { x86::vnni::gemv_q5(groups, gb, &xa, out, pf) },
        #[cfg(target_arch = "x86_64")]
        (Simd::Vnni, _) => unsafe { x86::vnni::gemv::<false>(groups, gb, &xa, out, pf) },
    }
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// One Q5_0 group unpacked to the Q8_0 block shape, for the GEMM.
    static UNPACKED: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Batched: out[(8g + r) * stride + t] = row r of group g · xs[t] for
/// t < xs.len() <= stride (row-major over (row, token slot); slots past
/// xs.len() are left alone), bit for bit `gemv` per token (as r8::gemm).
pub fn gemm(qtype_id: u32, groups: &[u8], xs: &[&QuantizedAct], out: &mut [f32], stride: usize) {
    // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
    unsafe { gemm_with(super::simd(), qtype_id, groups, xs, out, stride) }
}

/// `gemm` on an explicit path.
///
/// # Safety
/// As for quant.rs's `dot_row_q8_with`.
pub(super) unsafe fn gemm_with(
    simd: Simd,
    qtype_id: u32,
    groups: &[u8],
    xs: &[&QuantizedAct],
    out: &mut [f32],
    stride: usize,
) {
    let nt = xs.len();
    assert!(nt <= stride, "r8q gemm: more tokens than slots");
    if nt == 0 {
        return;
    }
    let n = xs[0].len();
    assert!(xs.iter().all(|x| x.len() == n), "r8q gemm: ragged batch");
    let off = chain_offset(simd, qtype_id);
    let xa: Vec<Act> = xs.iter().map(|x| act(x, off)).collect();
    let gb = group_bytes(qtype_id, n);
    let ng = out.len() / (8 * stride);
    assert!(
        out.len() == 8 * stride * ng && groups.len() >= ng * gb,
        "r8q gemm: sizes"
    );
    let groups = &groups[..ng * gb];
    if simd == Simd::Scalar {
        let mut o = [0f32; 8];
        for (g, og) in groups
            .chunks_exact(gb)
            .zip(out.chunks_exact_mut(8 * stride))
        {
            for (t, x) in xa.iter().enumerate() {
                group_scalar(qtype_id, g, x, &mut o);
                for r in 0..8 {
                    og[r * stride + t] = o[r];
                }
            }
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY (below): guaranteed by our caller.
        let run = |g: &[u8], ugb: usize, o: &mut [f32]| match (simd, qtype_id) {
            (Simd::Avx2, Q8_0) => unsafe { x86::avx2::gemm::<true>(g, ugb, &xa, o, stride) },
            (Simd::Avx2, _) => unsafe { x86::avx2::gemm::<false>(g, ugb, &xa, o, stride) },
            _ => unsafe { x86::vnni::gemm::<false>(g, ugb, &xa, o, stride) },
        };
        if qtype_id == Q8_0 {
            run(groups, gb, out);
            return;
        }
        let ugb = group_bytes(Q8_0, n);
        UNPACKED.with_borrow_mut(|u| {
            u.resize(ugb, 0);
            for (g, og) in groups
                .chunks_exact(gb)
                .zip(out.chunks_exact_mut(8 * stride))
            {
                // SAFETY: Avx2/Vnni imply avx2 (caller).
                unsafe { x86::unpack_q5_group(g, u) };
                run(u, ugb, og);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// x86 kernels. One tile function: T tokens x 8 rows, all blocks of a group
// (Q8_0-shaped: Q8_0, or Q5_0 unpacked); per block the 8 weight vectors are
// loaded once and used by all T tokens.
//
// Register budget (T = 4): f32 acc 4 + i32 chains 4 + weight 1 + broadcast
// 1 + d_w 1 + f32 temps 2 = 13 of 16 ymm (AVX2 adds the 1s constant and a
// maddubs temp; the Q8_0 sign path |w| and sign(a, w)).

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Act, Q5_H, QV, R8Q5, R8Q8};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    fn ld32<const N: usize>(b: &[u8; N], off: usize) -> __m256i {
        let p: &[u8; 32] = b[off..off + 32].try_into().unwrap();
        // SAFETY: p points to 32 readable bytes; loadu has no alignment requirement.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn ld16<const N: usize>(b: &[u8; N]) -> __m128i {
        let p: &[u8; 16] = b[..16].try_into().unwrap();
        // SAFETY: as ld32 (16 bytes).
        unsafe { _mm_loadu_si128(p.as_ptr().cast()) }
    }

    /// q[4v..4v + 4] broadcast to every i32 lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn bc(q: &[i8; 32], v: usize) -> __m256i {
        _mm256_set1_epi32(i32::from_le_bytes([
            q[4 * v] as u8,
            q[4 * v + 1] as u8,
            q[4 * v + 2] as u8,
            q[4 * v + 3] as u8,
        ]))
    }

    /// Hint [off, off + len) of `all` into L2, clamped to its end (64-byte
    /// lines). L2, not L1 (T1, not T0): a burst of T0 hints ties up the L1
    /// fill buffers the demand loads need (as r8bi measured); callers also
    /// spread a block's hints across its loop instead of one burst.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn prefetch(all: &[u8], off: usize, len: usize) {
        let end = (off + len).min(all.len());
        let mut p = off;
        while p < end {
            // SAFETY: p < all.len(); prefetch never faults anyway.
            unsafe { _mm_prefetch::<_MM_HINT_T1>(all.as_ptr().add(p).cast()) };
            p += 64;
        }
    }

    /// Vector v (0..8) of a compact Q5_0 block's u: nibbles | high bit << 4.
    /// `h` is the block's H vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn q5_vec(b: &[u8; R8Q5], h: __m256i, v: usize) -> __m256i {
        let m15 = _mm256_set1_epi8(15);
        let n = ld32(b, QV + 32 * (v % 4));
        let nib = if v < 4 {
            _mm256_and_si256(n, m15)
        } else {
            _mm256_and_si256(_mm256_srli_epi16::<4>(n), m15)
        };
        // bit v of each H byte to bit 4 (16-bit shifts by <= 4: bit 4 of a
        // byte only ever comes from the same byte)
        let hb = match v {
            0 => _mm256_slli_epi16::<4>(h),
            1 => _mm256_slli_epi16::<3>(h),
            2 => _mm256_slli_epi16::<2>(h),
            3 => _mm256_slli_epi16::<1>(h),
            4 => h,
            5 => _mm256_srli_epi16::<1>(h),
            6 => _mm256_srli_epi16::<2>(h),
            _ => _mm256_srli_epi16::<3>(h),
        };
        _mm256_or_si256(nib, _mm256_and_si256(hb, _mm256_set1_epi8(0x10)))
    }

    /// A compact Q5_0 group into `out`, Q8_0-shaped (same scales, u vectors).
    ///
    /// # Safety
    /// The CPU must support avx2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_q5_group(group: &[u8], out: &mut [u8]) {
        let src = group.as_chunks::<R8Q5>().0;
        let dst = out.as_chunks_mut::<R8Q8>().0;
        for (b, o) in src.iter().zip(dst) {
            o[..QV].copy_from_slice(&b[..QV]);
            let h = ld32(b, Q5_H);
            for v in 0..8 {
                let p: &mut [u8; 32] = (&mut o[QV + 32 * v..QV + 32 * v + 32]).try_into().unwrap();
                // SAFETY: p is 32 writable bytes; storeu is unaligned.
                unsafe { _mm256_storeu_si256(p.as_mut_ptr().cast(), q5_vec(b, h, v)) };
            }
        }
    }

    /// acc + Σ u·a per i32 lane (4 products each), maddubs/madd. Exact
    /// while each pair stays < 2^15 (Q5_0: u <= 31).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn dp_avx2<const SGN: bool>(acc: __m256i, u: __m256i, a: __m256i) -> __m256i {
        let p = if SGN {
            // Q8_0: w = u ^ 0x80 (i8), Σ |w| sign(a, w) = Σ w a
            let w = _mm256_xor_si256(u, _mm256_set1_epi8(0x80u8 as i8));
            _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a, w))
        } else {
            _mm256_maddubs_epi16(u, a)
        };
        _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)))
    }

    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    fn dp_vnni<const SGN: bool>(acc: __m256i, u: __m256i, a: __m256i) -> __m256i {
        _mm256_dpbusd_avx_epi32(acc, u, a)
    }

    macro_rules! kernels {
        ($m:ident, $feat:literal, $dp:ident) => {
            pub mod $m {
                use super::*;

                /// T tokens through every block of one Q8_0-shaped group.
                /// `all`/`base` locate the group for the prefetch (pf = 0: none).
                #[allow(clippy::needless_range_loop)]
                #[inline]
                #[target_feature(enable = $feat)]
                fn tile<const T: usize, const SGN: bool>(
                    all: &[u8],
                    base: usize,
                    gb: usize,
                    x: &[&Act; T],
                    pf: usize,
                ) -> [__m256; T] {
                    let blocks = all[base..base + gb].as_chunks::<R8Q8>().0;
                    let nb = blocks.len();
                    let mut d = [&x[0].d[..nb]; T];
                    let mut q = [&x[0].q[..nb]; T];
                    let mut s0 = [&x[0].s0[..nb]; T];
                    for t in 1..T {
                        d[t] = &x[t].d[..nb];
                        q[t] = &x[t].q[..nb];
                        s0[t] = &x[t].s0[..nb];
                    }
                    let mut acc = [_mm256_setzero_ps(); T];
                    for (i, b) in blocks.iter().enumerate() {
                        let mut s = [_mm256_setzero_si256(); T];
                        for t in 0..T {
                            s[t] = _mm256_set1_epi32(s0[t][i]);
                        }
                        // the block `pf` bytes ahead, half before each
                        // half of the vectors (two loops: a branch inside
                        // one stops it being unrolled)
                        let ahead = base + i * R8Q8 + pf;
                        if pf > 0 {
                            prefetch(all, ahead, R8Q8 / 2);
                        }
                        for v in 0..4 {
                            let u = ld32(b, QV + 32 * v);
                            for t in 0..T {
                                s[t] = $dp::<SGN>(s[t], u, bc(&q[t][i], v));
                            }
                        }
                        if pf > 0 {
                            prefetch(all, ahead + R8Q8 / 2, R8Q8 / 2);
                        }
                        for v in 4..8 {
                            let u = ld32(b, QV + 32 * v);
                            for t in 0..T {
                                s[t] = $dp::<SGN>(s[t], u, bc(&q[t][i], v));
                            }
                        }
                        let dw = _mm256_cvtph_ps(ld16(b));
                        for t in 0..T {
                            let dd = _mm256_mul_ps(dw, _mm256_set1_ps(d[t][i]));
                            acc[t] =
                                _mm256_add_ps(acc[t], _mm256_mul_ps(dd, _mm256_cvtepi32_ps(s[t])));
                        }
                    }
                    acc
                }

                /// GEMV over whole Q8_0 groups (`gb` bytes each), one token.
                ///
                /// # Safety
                /// The CPU must support the enabled target features.
                #[target_feature(enable = $feat)]
                pub unsafe fn gemv<const SGN: bool>(
                    groups: &[u8],
                    gb: usize,
                    x: &Act,
                    out: &mut [f32],
                    pf: usize,
                ) {
                    for (g, o) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let [acc] = tile::<1, SGN>(groups, g * gb, gb, &[x], pf);
                        // SAFETY: o is 32 writable bytes; storeu is unaligned.
                        unsafe { _mm256_storeu_ps(o.as_mut_ptr(), acc) };
                    }
                }

                /// GEMV over whole compact Q5_0 groups, one token; unpacks
                /// each block inline (decode is memory-bound).
                ///
                /// # Safety
                /// The CPU must support the enabled target features.
                #[target_feature(enable = $feat)]
                pub unsafe fn gemv_q5(
                    groups: &[u8],
                    gb: usize,
                    x: &Act,
                    out: &mut [f32],
                    pf: usize,
                ) {
                    for (g, o) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let base = g * gb;
                        let blocks = groups[base..base + gb].as_chunks::<R8Q5>().0;
                        let nb = blocks.len();
                        let (d, q, s0) = (&x.d[..nb], &x.q[..nb], &x.s0[..nb]);
                        let mut acc = _mm256_setzero_ps();
                        for (i, b) in blocks.iter().enumerate() {
                            let h = ld32(b, Q5_H);
                            let mut s = _mm256_set1_epi32(s0[i]);
                            // as in `tile`
                            let ahead = base + i * R8Q5 + pf;
                            if pf > 0 {
                                prefetch(groups, ahead, R8Q5 / 2);
                            }
                            for v in 0..4 {
                                s = $dp::<false>(s, q5_vec(b, h, v), bc(&q[i], v));
                            }
                            if pf > 0 {
                                prefetch(groups, ahead + R8Q5 / 2, R8Q5 / 2);
                            }
                            for v in 4..8 {
                                s = $dp::<false>(s, q5_vec(b, h, v), bc(&q[i], v));
                            }
                            let dd = _mm256_mul_ps(_mm256_cvtph_ps(ld16(b)), _mm256_set1_ps(d[i]));
                            acc = _mm256_add_ps(acc, _mm256_mul_ps(dd, _mm256_cvtepi32_ps(s)));
                        }
                        // SAFETY: o is 32 writable bytes; storeu is unaligned.
                        unsafe { _mm256_storeu_ps(o.as_mut_ptr(), acc) };
                    }
                }

                /// T tokens from t0 through group g, into out (row-major
                /// over (row, token slot), `stride` slots per row).
                #[allow(clippy::needless_range_loop)]
                #[inline]
                #[target_feature(enable = $feat)]
                fn tokens<const T: usize, const SGN: bool>(
                    groups: &[u8],
                    g: usize,
                    gb: usize,
                    xa: &[Act],
                    t0: usize,
                    out: &mut [f32],
                    stride: usize,
                ) {
                    let mut x = [&xa[t0]; T];
                    for t in 1..T {
                        x[t] = &xa[t0 + t];
                    }
                    let acc = tile::<T, SGN>(groups, g * gb, gb, &x, 0);
                    let og = &mut out[8 * stride * g..8 * stride * (g + 1)];
                    for t in 0..T {
                        let mut o = [0f32; 8];
                        // SAFETY: o is 32 writable bytes.
                        unsafe { _mm256_storeu_ps(o.as_mut_ptr(), acc[t]) };
                        for r in 0..8 {
                            og[r * stride + t0 + t] = o[r];
                        }
                    }
                }

                /// Every Q8_0-shaped group x every token, 4 tokens per tile.
                ///
                /// # Safety
                /// The CPU must support the enabled target features.
                #[target_feature(enable = $feat)]
                pub unsafe fn gemm<const SGN: bool>(
                    groups: &[u8],
                    gb: usize,
                    xa: &[Act],
                    out: &mut [f32],
                    stride: usize,
                ) {
                    let nt = xa.len();
                    for g in 0..out.len() / (8 * stride) {
                        let mut t0 = 0;
                        while t0 + 4 <= nt {
                            tokens::<4, SGN>(groups, g, gb, xa, t0, out, stride);
                            t0 += 4;
                        }
                        match nt - t0 {
                            1 => tokens::<1, SGN>(groups, g, gb, xa, t0, out, stride),
                            2 => tokens::<2, SGN>(groups, g, gb, xa, t0, out, stride),
                            3 => tokens::<3, SGN>(groups, g, gb, xa, t0, out, stride),
                            _ => {}
                        }
                    }
                }
            }
        };
    }

    kernels!(avx2, "avx2,fma,f16c", dp_avx2);
    kernels!(vnni, "avx2,fma,f16c,avxvnni", dp_vnni);
}

#[cfg(test)]
mod tests {
    use super::super::{F32, dequant, dot_row_q8, dot_row_q8_with, row_bytes};
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        /// uniform in [-1, 1)
        fn f(&mut self) -> f32 {
            self.next_u32() as f32 / (1u64 << 30) as f32 - 1.0
        }
        /// a sane f16 bit pattern, |v| in [2^-8, 2^-4), random sign
        fn f16_bits(&mut self) -> u16 {
            let r = self.next_u32();
            (0x1c00 + (r % 0x1000) as u16) | ((((r >> 16) & 1) as u16) << 15)
        }
    }

    fn block_bytes(qt: u32) -> usize {
        if qt == Q5_0 { Q5_0_BYTES } else { Q8_0_BYTES }
    }

    /// Random row; `extreme`: Q5_0 every raw value 0..31 per block, Q8_0
    /// -128/127/-127/0/±1 pinned in every block.
    fn row(qt: u32, n: usize, extreme: bool, seed: usize, rng: &mut Lcg) -> Vec<u8> {
        let bb = block_bytes(qt);
        let mut row: Vec<u8> = (0..row_bytes(qt, n))
            .map(|_| rng.next_u32() as u8)
            .collect();
        for (b, blk) in row.chunks_exact_mut(bb).enumerate() {
            blk[0..2].copy_from_slice(&rng.f16_bits().to_le_bytes());
            if !extreme {
                continue;
            }
            if qt == Q5_0 {
                let r = |j: usize| ((j + 7 * b + seed) % 32) as u8;
                let mut qh = 0u32;
                for j in 0..32 {
                    qh |= ((r(j) >> 4) as u32) << j;
                }
                blk[2..6].copy_from_slice(&qh.to_le_bytes());
                for j in 0..16 {
                    blk[6 + j] = (r(j) & 0xf) | ((r(j + 16) & 0xf) << 4);
                }
            } else {
                let pins = [-128i8, 127, -127, 0, 1, -1, -128, 127];
                for (k, &v) in pins.iter().enumerate() {
                    blk[2 + (k * 4 + b + seed) % 32] = v as u8;
                }
            }
        }
        row
    }

    fn matrix(qt: u32, rows: usize, n: usize, extreme: bool, rng: &mut Lcg) -> Vec<u8> {
        (0..rows)
            .flat_map(|r| row(qt, n, extreme, r * 5, rng))
            .collect()
    }

    /// Activations with per-256 magnitude changes and one all-zero region.
    fn synthetic_x(n: usize, rng: &mut Lcg) -> Vec<f32> {
        let scales = [1.0f32, 0.01, 30.0, 0.0, 3.0];
        (0..n)
            .map(|i| rng.f() * scales[(i / 256) % scales.len()])
            .collect()
    }

    /// Activations hitting q = ±127 often, including whole lanes of ±127.
    fn extreme_x(n: usize, rng: &mut Lcg) -> Vec<f32> {
        let mode = rng.next_u32() % 4;
        (0..n)
            .map(|i| match mode {
                0 => 1.0,
                1 => -1.0,
                _ => match i % 5 {
                    0 => 1.0,
                    1 => -1.0,
                    _ if (i / 32) % 3 == 0 => -(((i / 4) % 2) as f32 * 2.0 - 1.0),
                    _ => rng.f(),
                },
            })
            .collect()
    }

    fn paths() -> Vec<Simd> {
        #[allow(unused_mut)]
        let mut v = vec![Simd::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if super::super::has_avx2() {
                v.push(Simd::Avx2);
            } else {
                println!("SKIP avx2 path");
            }
            if super::super::has_vnni() {
                v.push(Simd::Vnni);
            } else {
                println!("SKIP vnni path");
            }
        }
        v
    }

    fn act_values(qa: &QuantizedAct) -> Vec<f32> {
        let B32 { d, q, .. } = qa.b32.as_ref().unwrap();
        d.iter()
            .zip(q)
            .flat_map(|(&d, q)| q.map(|v| d * v as f32))
            .collect()
    }

    #[test]
    fn r8q_repack_roundtrip() {
        let mut rng = Lcg(0x5038);
        for qt in [Q5_0, Q8_0] {
            for n in [896usize, 1536] {
                for extreme in [false, true] {
                    let rows = 16;
                    let m = matrix(qt, rows, n, extreme, &mut rng);
                    let p = repack(qt, &m, rows, n);
                    let gb = group_bytes(qt, n);
                    assert_eq!(p.len(), rows / 8 * gb);
                    assert_eq!(p.len(), rows * row_bytes(qt, n));
                    #[cfg(target_arch = "x86_64")]
                    if qt == Q5_0 && super::super::has_avx2() {
                        // the GEMM's unpacked form holds the same u values
                        let mut u = vec![0u8; group_bytes(Q8_0, n)];
                        for g in p.chunks_exact(gb) {
                            // SAFETY: avx2 detected.
                            unsafe { x86::unpack_q5_group(g, &mut u) };
                            for (bc, bu) in g.chunks_exact(R8Q5).zip(u.chunks_exact(R8Q8)) {
                                assert_eq!(bc[..16], bu[..16]);
                                for r in 0..8 {
                                    for k in 0..32 {
                                        assert_eq!(u_at(Q5_0, bc, r, k), u_at(Q8_0, bu, r, k));
                                    }
                                }
                            }
                        }
                    }
                    let rb = row_bytes(qt, n);
                    for r in 0..rows {
                        let mut want = vec![0f32; n];
                        dequant(qt, &m[r * rb..(r + 1) * rb], &mut want);
                        let mut got = vec![0f32; n];
                        dequant_row(qt, &p[r / 8 * gb..(r / 8 + 1) * gb], r % 8, &mut got);
                        for k in 0..n {
                            assert_eq!(
                                got[k].to_bits(),
                                want[k].to_bits(),
                                "qt {qt} n {n} extreme {extreme} row {r} k {k}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Scalar GEMV per token is the reference: it must equal today's per-row
    /// Scalar kernel, and every path's GEMV (every prefetch distance) and
    /// GEMM (every batch size) must equal it, bit for bit.
    #[test]
    fn r8q_paths_and_gemm_match_scalar() {
        let paths = paths();
        let mut rng = Lcg(0x5088);
        let counts: Vec<usize> = (1..=9).chain([64, 67]).collect();
        for qt in [Q5_0, Q8_0] {
            for (n, extreme) in [(896usize, false), (896, true), (1536, false), (4864, true)] {
                let rows = 16;
                let m = matrix(qt, rows, n, extreme, &mut rng);
                let rb = row_bytes(qt, n);
                let p = repack(qt, &m, rows, n);
                for &nt in &counts {
                    if n == 4864 && !matches!(nt, 1 | 5 | 67) {
                        continue;
                    }
                    let qas: Vec<QuantizedAct> = (0..nt)
                        .map(|t| {
                            let x = if extreme || t % 2 == 1 {
                                extreme_x(n, &mut rng)
                            } else {
                                synthetic_x(n, &mut rng)
                            };
                            QuantizedAct::for_types(&x, &[qt])
                        })
                        .collect();
                    let xs: Vec<&QuantizedAct> = qas.iter().collect();
                    let want: Vec<Vec<f32>> = qas
                        .iter()
                        .map(|qa| {
                            let mut o = vec![f32::NAN; rows];
                            // SAFETY: Scalar is always valid.
                            unsafe { gemv_with(Simd::Scalar, qt, &p, qa, &mut o, 0) };
                            for r in 0..rows {
                                // SAFETY: as above.
                                let old = unsafe {
                                    dot_row_q8_with(Simd::Scalar, qt, &m[r * rb..(r + 1) * rb], qa)
                                };
                                assert_eq!(
                                    o[r].to_bits(),
                                    old.to_bits(),
                                    "scalar per-row qt {qt} n {n} row {r}"
                                );
                            }
                            o
                        })
                        .collect();
                    for &path in &paths {
                        for pf in [0usize, 64, 8192] {
                            for (t, qa) in qas.iter().enumerate() {
                                let mut o = vec![f32::NAN; rows];
                                // SAFETY: Scalar, or a detected path.
                                unsafe { gemv_with(path, qt, &p, qa, &mut o, pf) };
                                for r in 0..rows {
                                    assert_eq!(
                                        o[r].to_bits(),
                                        want[t][r].to_bits(),
                                        "gemv qt {qt} n {n} {path:?} pf {pf} token {t} row {r}: {} vs {}",
                                        o[r],
                                        want[t][r]
                                    );
                                }
                            }
                        }
                        // Slot layout wider than the batch: padding slots untouched.
                        let st = nt + 3;
                        let mut w = vec![f32::NAN; rows * st];
                        // SAFETY: as above.
                        unsafe { gemm_with(path, qt, &p, &xs, &mut w, st) };
                        for r in 0..rows {
                            for t in 0..st {
                                let (got, want) = (w[r * st + t], want.get(t).map(|v| v[r]));
                                match want {
                                    Some(v) => assert_eq!(got.to_bits(), v.to_bits()),
                                    None => assert!(got.is_nan(), "padding slot written"),
                                }
                            }
                        }
                        let mut o = vec![f32::NAN; rows * nt];
                        // SAFETY: as above.
                        unsafe { gemm_with(path, qt, &p, &xs, &mut o, nt) };
                        for r in 0..rows {
                            for t in 0..nt {
                                assert_eq!(
                                    o[r * nt + t].to_bits(),
                                    want[t][r].to_bits(),
                                    "gemm qt {qt} n {n} {path:?} B {nt} token {t} row {r}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The checks above catch a broken offset, swapped scales, swapped
    /// interleave lanes and (Q5_0) a flipped high bit or swapped nibbles
    /// (each mutation must change some output bits).
    #[test]
    fn r8q_mutations_are_caught() {
        let mut rng = Lcg(0x3e3e);
        let (rows, n) = (8usize, 896usize);
        for qt in [Q5_0, Q8_0] {
            let m = matrix(qt, rows, n, true, &mut rng);
            let p = repack(qt, &m, rows, n);
            let x = synthetic_x(n, &mut rng);
            let qa = QuantizedAct::for_types(&x, &[qt]);
            for path in paths() {
                let off = chain_offset(path, qt);
                let mut want = vec![0f32; rows];
                // SAFETY: Scalar, or a detected path.
                unsafe { gemv_with(path, qt, &p, &qa, &mut want, 0) };
                let run = |p: &[u8], xa: &Act| {
                    let mut o = [0f32; 8];
                    match path {
                        Simd::Scalar => group_scalar(qt, p, xa, &mut o),
                        // SAFETY (below): detected paths.
                        #[cfg(target_arch = "x86_64")]
                        Simd::Avx2 if qt == Q8_0 => unsafe {
                            x86::avx2::gemv::<true>(p, p.len(), xa, &mut o, 0)
                        },
                        #[cfg(target_arch = "x86_64")]
                        Simd::Avx2 => unsafe { x86::avx2::gemv_q5(p, p.len(), xa, &mut o, 0) },
                        #[cfg(target_arch = "x86_64")]
                        Simd::Vnni if qt == Q8_0 => unsafe {
                            x86::vnni::gemv::<false>(p, p.len(), xa, &mut o, 0)
                        },
                        #[cfg(target_arch = "x86_64")]
                        Simd::Vnni => unsafe { x86::vnni::gemv_q5(p, p.len(), xa, &mut o, 0) },
                    }
                    o
                };
                let same = |o: [f32; 8]| (0..8).all(|r| o[r].to_bits() == want[r].to_bits());
                assert!(same(run(&p, &act(&qa, off))), "{path:?} unmutated");
                // offset: chain start of block 3 off by the offset of one unit of a
                let mut xa = act(&qa, off);
                xa.s0[3] += offset(qt);
                assert!(
                    !same(run(&p, &xa)),
                    "{path:?} qt {qt}: broken offset not caught"
                );
                // wrong offset everywhere (e.g. Q5_0's 16 for Q8_0)
                if off != 0 {
                    let xa = act(&qa, if off == 16 { 128 } else { 16 });
                    assert!(
                        !same(run(&p, &xa)),
                        "{path:?} qt {qt}: wrong OFF not caught"
                    );
                }
                // swap rows 2 and 5 scales in block 0
                let mut q = p.clone();
                q.swap(4, 10);
                q.swap(5, 11);
                if q != p {
                    let o = run(&q, &act(&qa, off));
                    assert!(!same(o), "{path:?} qt {qt}: swapped scale not caught");
                }
                // swap two rows' lanes in vector 1 of block 7
                let mut q = p.clone();
                let b = 7 * block_bytes(qt) + QV + 32;
                for i in 0..4 {
                    q.swap(b + i, b + 12 + i);
                }
                assert!(
                    !same(run(&q, &act(&qa, off))),
                    "{path:?} qt {qt}: lane swap not caught"
                );
                if qt == Q5_0 {
                    // flip one high bit (row 6, k = 4*5 + 2) of block 9
                    let mut q = p.clone();
                    q[9 * R8Q5 + Q5_H + 4 * 6 + 2] ^= 1 << 5;
                    assert!(
                        !same(run(&q, &act(&qa, off))),
                        "{path:?}: high bit flip not caught"
                    );
                    // swap the nibbles of one lane byte: low/high k mixed up
                    let mut q = p.clone();
                    let i = 4 * R8Q5 + QV + 32 * 2 + 4;
                    q[i] = q[i].rotate_left(4);
                    if q != p {
                        assert!(
                            !same(run(&q, &act(&qa, off))),
                            "{path:?}: nibble swap not caught"
                        );
                    }
                }
            }
        }
    }

    /// Real 0.5B Q5_0 / Q8_0 rows: r8q against an f64 dot of the dequantized
    /// weights and activations (relative to Σ|w x|), next to today's kernel
    /// (default SIMD path). Asserts the f64 bound; prints both errors.
    #[test]
    fn r8q_accuracy_on_model() {
        let Some(model) = crate::testutil::model() else {
            return;
        };
        let mut rng = Lcg(0xacc5);
        let mut seen = std::collections::BTreeSet::new();
        for t in &model.tensors {
            let qt = t.qtype.id;
            if !matches!(qt, Q5_0 | Q8_0) || t.dims.len() != 2 {
                continue;
            }
            let (n, all_rows) = (t.dims[0] as usize, t.dims[1] as usize);
            // one tensor per (type, kind)
            if !seen.insert((qt, t.kind.clone())) || !can_repack(qt, all_rows, n) {
                continue;
            }
            let rows = all_rows.min(64);
            let rb = row_bytes(qt, n);
            let m = &model.tensor_data(t)[..rows * rb];
            let p = repack(qt, m, rows, n);
            let (mut new_rel, mut old_rel, mut cnt) = (0f64, 0f64, 0usize);
            for _ in 0..4 {
                let x = synthetic_x(n, &mut rng);
                let qa = QuantizedAct::for_types(&x, &[qt, F32]);
                let xq = act_values(&qa);
                let mut o = vec![0f32; rows];
                gemv(qt, &p, &qa, &mut o, 8192);
                for r in 0..rows {
                    let row = &m[r * rb..(r + 1) * rb];
                    let mut w = vec![0f32; n];
                    dequant(qt, row, &mut w);
                    let exact: f64 = w.iter().zip(&xq).map(|(a, b)| *a as f64 * *b as f64).sum();
                    let mag: f64 = w
                        .iter()
                        .zip(&xq)
                        .map(|(a, b)| (*a as f64 * *b as f64).abs())
                        .sum::<f64>()
                        .max(f64::MIN_POSITIVE);
                    let old = dot_row_q8(qt, row, &qa);
                    let e_new = (o[r] as f64 - exact).abs() / mag;
                    assert!(
                        e_new <= 1e-5,
                        "{} row {r}: r8q {} exact {exact}",
                        t.name,
                        o[r]
                    );
                    new_rel = new_rel.max(e_new);
                    old_rel = old_rel.max((old as f64 - exact).abs() / mag);
                    cnt += 1;
                }
            }
            println!(
                "r8q accuracy {} ({qt}, {rows}x{n}, {cnt} dots): max rel err vs f64 r8q {new_rel:.2e}, today {old_rel:.2e}",
                t.name
            );
        }
        assert!(!seen.is_empty(), "no Q5_0/Q8_0 tensors in the model");
    }
}
