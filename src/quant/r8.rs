//! 8-row interleaved ("r8") Q4_K / Q6_K weights with integer-first kernels.
//!
//! NUMERICS PROPOSAL: results differ in the low bits from the per-row kernels
//! in quant.rs (one f32 step per (row, super-block) instead of 8 f32 lanes per
//! row). Every path here (scalar, AVX2, AVX-VNNI; GEMV and GEMM) computes the
//! same exact i32 sums and then the same f32 ops in the same order, so they
//! agree bit for bit with each other.
//!
//! Prior art: llama.cpp ggml-cpu/repack.cpp (block_q4_Kx8), ik_llama.cpp R8
//! row-interleaved k-quants, Georganas et al. arXiv:2508.06753 (VNNI4 order).
//!
//! Layout. A matrix with rows % 8 == 0 becomes rows / 8 groups; a group is
//! its super-blocks in order, each a fixed-size block for 8 rows. In every
//! 32-byte quant vector, bytes 4r..4r+4 are 4 consecutive k values of row r
//! (VNNI4: one dpbusd lane = one row).
//!
//! Q4_K block, 1184 bytes (8 x 148; the original is 8 x 144):
//!   [0..16)      d, f16 x 8 (row r at 2r)
//!   [16..32)     dmin, f16 x 8
//!   [32..96)     scales: pair c (sub-blocks 2c, 2c+1), 16 bytes, [2r + h] = sc[2c + h]
//!   [96..160)    mins, same layout as the scales
//!   [160..1184)  qs: vector (c, g) at 160 + 32 (8c + g), [4r + i] = qs_r[32c + 4g + i]
//!                (low nibble: k = 64c + 4g + i, high nibble: k + 32)
//! Q6_K block, 1680 bytes (8 x 210, same as the original):
//!   [0..16)      d, f16 x 8
//!   [16..144)    scales (i8): pair p (sub-blocks 2p, 2p+1), [2r + h] = sc[2p + h]
//!   [144..1680)  per (h, g), 96 bytes at 144 + 96 (8h + g): A, B, H vectors,
//!                [4r + i] of A = ql_r[64h + 4g + i], B = ql_r[64h + 32 + 4g + i],
//!                H = qh_r[32h + 4g + i]. With K0 = 128h + 4g + i:
//!                q[K0] = A & 15 | (H & 3) << 4, q[K0+32] = B & 15 | (H >> 2 & 3) << 4,
//!                q[K0+64] = A >> 4 | (H >> 4 & 3) << 4, q[K0+96] = B >> 4 | (H >> 6) << 4.
//!
//! Integer sums per (row, super-block) and activation block a (Q8_K):
//!   Q4_K: s_j = Σ_{k in sub-block j} u[k] a[k] (u = 0..15),
//!         sumi = Σ_j sc_j s_j, summ = Σ_j m_j (bsums[2j] + bsums[2j+1]);
//!         acc += (d * a.d) * f32(sumi) - (dmin * a.d) * f32(summ)
//!   Q6_K: t_j = Σ_{k in j} q[k] a[k] - 32 bsums[j] (q = 0..63; -32 offset folded in),
//!         isum = Σ_j sc_j t_j;  acc += (d * a.d) * f32(isum)
//! The f32 step is exactly that op order, no FMA, on every path.
//!
//! Overflow (|a| <= 127: quantize_q8_k clamps to [-127, 127]). Everything
//! integer is per super-block, so the row width (4864, 18944, ...) only sets
//! how many f32 steps there are.
//!   Q4_K: |s_j| <= 32*15*127 = 60960; |sumi| <= 8*63*60960 = 30.7M;
//!         |summ| <= 8*63*(32*127) = 2.05M.
//!   Q6_K: |t_j| <= 16*63*127 + 32*16*127 = 193040; |isum| <= 16*128*193040 = 395M.
//!   All < 2^31. dpbusd lanes: 4 products of u8 x i8, no saturation. AVX2
//!   maddubs pairs: 2*63*127 = 16002 < 32767 (Q6_K), 2*15*127 (Q4_K): no
//!   saturation, so the AVX2 sums equal the VNNI ones.

use super::{
    K_SCALE_SIZE, KBlock, Q4_K, Q4_K_BYTES, Q6_K, Q6_K_BYTES, QK_K, QuantizedAct, Simd, f16_to_f32,
    get_scale_min_k4,
};

pub const R8_Q4_K: usize = 1184;
pub const R8_Q6_K: usize = 1680;

const Q4_SC: usize = 32;
const Q4_MN: usize = 96;
const Q4_QS: usize = 160;
const Q6_SC: usize = 16;
const Q6_Q: usize = 144;

/// Bytes of one 8-row super-block for `qtype_id`, or None if it has no r8 form.
pub fn block_bytes(qtype_id: u32) -> Option<usize> {
    match qtype_id {
        Q4_K => Some(R8_Q4_K),
        Q6_K => Some(R8_Q6_K),
        _ => None,
    }
}

/// Whether a `rows` x `n` matrix of `qtype_id` can be repacked.
pub fn can_repack(qtype_id: u32, rows: usize, n: usize) -> bool {
    block_bytes(qtype_id).is_some() && rows.is_multiple_of(8) && n.is_multiple_of(QK_K) && n > 0
}

/// Bytes of one 8-row group (all of its super-blocks).
pub fn group_bytes(qtype_id: u32, n: usize) -> usize {
    block_bytes(qtype_id).expect("r8: Q4_K/Q6_K only") * (n / QK_K)
}

fn repack_q4_k_block(src: [&[u8; Q4_K_BYTES]; 8], out: &mut [u8; R8_Q4_K]) {
    for (r, b) in src.iter().enumerate() {
        out[2 * r..2 * r + 2].copy_from_slice(&b[0..2]);
        out[16 + 2 * r..16 + 2 * r + 2].copy_from_slice(&b[2..4]);
        let scales: &[u8] = &b[4..4 + K_SCALE_SIZE];
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, scales);
            out[Q4_SC + 16 * (j / 2) + 2 * r + j % 2] = sc;
            out[Q4_MN + 16 * (j / 2) + 2 * r + j % 2] = m;
        }
        let qs = &b[4 + K_SCALE_SIZE..];
        for c in 0..4 {
            for g in 0..8 {
                let o = Q4_QS + 32 * (8 * c + g) + 4 * r;
                out[o..o + 4].copy_from_slice(&qs[32 * c + 4 * g..32 * c + 4 * g + 4]);
            }
        }
    }
}

fn repack_q6_k_block(src: [&[u8; Q6_K_BYTES]; 8], out: &mut [u8; R8_Q6_K]) {
    for (r, b) in src.iter().enumerate() {
        out[2 * r..2 * r + 2].copy_from_slice(&b[208..210]);
        for p in 0..8 {
            out[Q6_SC + 16 * p + 2 * r] = b[192 + 2 * p];
            out[Q6_SC + 16 * p + 2 * r + 1] = b[192 + 2 * p + 1];
        }
        for h in 0..2 {
            for g in 0..8 {
                let o = Q6_Q + 96 * (8 * h + g) + 4 * r;
                let l = 4 * g;
                out[o..o + 4].copy_from_slice(&b[64 * h + l..64 * h + l + 4]);
                out[o + 32..o + 36].copy_from_slice(&b[64 * h + 32 + l..64 * h + 36 + l]);
                out[o + 64..o + 68].copy_from_slice(&b[128 + 32 * h + l..128 + 32 * h + l + 4]);
            }
        }
    }
}

/// Repack a `rows` x `n` matrix (`can_repack` must hold) into r8 groups.
#[cfg_attr(not(test), allow(dead_code))]
pub fn repack(qtype_id: u32, data: &[u8], rows: usize, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; repacked_len(qtype_id, rows, n)];
    repack_into(qtype_id, data, rows, n, &mut out);
    out
}

/// Bytes of the r8 form of a `rows` x `n` matrix.
pub fn repacked_len(qtype_id: u32, rows: usize, n: usize) -> usize {
    rows / 8 * group_bytes(qtype_id, n)
}

/// `repack` into `out` (`repacked_len` bytes).
pub fn repack_into(qtype_id: u32, data: &[u8], rows: usize, n: usize, out: &mut [u8]) {
    assert!(can_repack(qtype_id, rows, n), "r8: can't repack");
    let nsb = n / QK_K;
    let rb = super::row_bytes(qtype_id, n);
    assert!(data.len() >= rows * rb, "r8: short tensor data");
    let gb = group_bytes(qtype_id, n);
    assert_eq!(out.len(), rows / 8 * gb, "r8: output size");
    for (grp, og) in out.chunks_exact_mut(gb).enumerate() {
        let row = |r: usize| &data[(8 * grp + r) * rb..(8 * grp + r + 1) * rb];
        match qtype_id {
            Q4_K => {
                let ob = og.as_chunks_mut::<R8_Q4_K>().0;
                for (sb, o) in ob.iter_mut().enumerate().take(nsb) {
                    let src: [&[u8; Q4_K_BYTES]; 8] =
                        std::array::from_fn(|r| &row(r).as_chunks::<Q4_K_BYTES>().0[sb]);
                    repack_q4_k_block(src, o);
                }
            }
            _ => {
                let ob = og.as_chunks_mut::<R8_Q6_K>().0;
                for (sb, o) in ob.iter_mut().enumerate().take(nsb) {
                    let src: [&[u8; Q6_K_BYTES]; 8] =
                        std::array::from_fn(|r| &row(r).as_chunks::<Q6_K_BYTES>().0[sb]);
                    repack_q6_k_block(src, o);
                }
            }
        }
    }
}

/// Unsigned 4-bit quant of row r at k (0..256) of a Q4_K r8 block.
#[inline]
fn q4_at(b: &[u8; R8_Q4_K], r: usize, k: usize) -> u8 {
    let (c, hi, l) = (k / 64, (k / 32) % 2, k % 32);
    let v = b[Q4_QS + 32 * (8 * c + l / 4) + 4 * r + l % 4];
    if hi == 1 { v >> 4 } else { v & 15 }
}

/// Unsigned 6-bit quant (0..63) of row r at k of a Q6_K r8 block.
#[inline]
fn q6_at(b: &[u8; R8_Q6_K], r: usize, k: usize) -> u8 {
    let (h, q, l) = (k / 128, (k % 128) / 32, k % 32);
    let o = Q6_Q + 96 * (8 * h + l / 4) + 4 * r + l % 4;
    let (a, bb, hh) = (b[o], b[o + 32], b[o + 64]);
    let low = match q {
        0 => a & 15,
        1 => bb & 15,
        2 => a >> 4,
        _ => bb >> 4,
    };
    low | ((hh >> (2 * q)) & 3) << 4
}

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// Dequantize row r (0..8) of an r8 group into `out` (n values). The same
/// float formulas as quant.rs's dequant_q4_k_block / dequant_q6_k_block.
#[cfg_attr(not(test), allow(dead_code))]
pub fn dequant_row(qtype_id: u32, group: &[u8], r: usize, out: &mut [f32]) {
    let nsb = out.len() / QK_K;
    match qtype_id {
        Q4_K => {
            for (b, y) in group.as_chunks::<R8_Q4_K>().0[..nsb]
                .iter()
                .zip(out.as_chunks_mut::<QK_K>().0)
            {
                let d = f16_at(b, 2 * r);
                let dmin = f16_at(b, 16 + 2 * r);
                for (k, yk) in y.iter_mut().enumerate() {
                    let j = k / 32;
                    let sc = b[Q4_SC + 16 * (j / 2) + 2 * r + j % 2];
                    let m = b[Q4_MN + 16 * (j / 2) + 2 * r + j % 2];
                    *yk = (d * sc as f32) * q4_at(b, r, k) as f32 - dmin * m as f32;
                }
            }
        }
        Q6_K => {
            for (b, y) in group.as_chunks::<R8_Q6_K>().0[..nsb]
                .iter()
                .zip(out.as_chunks_mut::<QK_K>().0)
            {
                let d = f16_at(b, 2 * r);
                for (k, yk) in y.iter_mut().enumerate() {
                    let j = k / 16;
                    let sc = b[Q6_SC + 16 * (j / 2) + 2 * r + j % 2] as i8;
                    *yk = d * sc as f32 * (q6_at(b, r, k) as i32 - 32) as f32;
                }
            }
        }
        _ => panic!("r8: Q4_K/Q6_K only"),
    }
}

// ---------------------------------------------------------------------------
// Scalar reference, in the new order.

#[inline]
fn bsum32(a: &KBlock, j: usize) -> i32 {
    a.bsums[2 * j] as i32 + a.bsums[2 * j + 1] as i32
}

/// (sumi, summ) per row for one Q4_K block.
#[allow(clippy::needless_range_loop)]
fn q4_k_sums(b: &[u8; R8_Q4_K], a: &KBlock) -> ([i32; 8], [i32; 8]) {
    let mut sumi = [0i32; 8];
    let mut summ = [0i32; 8];
    for r in 0..8 {
        for j in 0..8 {
            let mut s = 0i32;
            for k in 32 * j..32 * j + 32 {
                s += q4_at(b, r, k) as i32 * a.qs[k] as i32;
            }
            let sc = b[Q4_SC + 16 * (j / 2) + 2 * r + j % 2] as i32;
            let m = b[Q4_MN + 16 * (j / 2) + 2 * r + j % 2] as i32;
            sumi[r] += sc * s;
            summ[r] += m * bsum32(a, j);
        }
    }
    (sumi, summ)
}

/// isum per row for one Q6_K block.
#[allow(clippy::needless_range_loop)]
fn q6_k_sums(b: &[u8; R8_Q6_K], a: &KBlock) -> [i32; 8] {
    let mut isum = [0i32; 8];
    for r in 0..8 {
        for j in 0..16 {
            let mut t = -32 * a.bsums[j] as i32;
            for k in 16 * j..16 * j + 16 {
                t += q6_at(b, r, k) as i32 * a.qs[k] as i32;
            }
            let sc = b[Q6_SC + 16 * (j / 2) + 2 * r + j % 2] as i8 as i32;
            isum[r] += sc * t;
        }
    }
    isum
}

/// The f32 step for one Q4_K block (see the module comment for the order).
#[allow(clippy::needless_range_loop)]
fn q4_k_step(acc: &mut [f32; 8], b: &[u8; R8_Q4_K], ad: f32, sumi: &[i32; 8], summ: &[i32; 8]) {
    for r in 0..8 {
        let dd = f16_at(b, 2 * r) * ad;
        let dm = f16_at(b, 16 + 2 * r) * ad;
        acc[r] += dd * sumi[r] as f32 - dm * summ[r] as f32;
    }
}

#[allow(clippy::needless_range_loop)]
fn q6_k_step(acc: &mut [f32; 8], b: &[u8; R8_Q6_K], ad: f32, isum: &[i32; 8]) {
    for r in 0..8 {
        let dd = f16_at(b, 2 * r) * ad;
        acc[r] += dd * isum[r] as f32;
    }
}

/// One r8 group x one token, scalar: out[r] for the group's 8 rows.
fn group_scalar(qtype_id: u32, group: &[u8], xk: &[KBlock], out: &mut [f32; 8]) {
    let mut acc = [0f32; 8];
    match qtype_id {
        Q4_K => {
            for (b, a) in group.as_chunks::<R8_Q4_K>().0.iter().zip(xk) {
                let (sumi, summ) = q4_k_sums(b, a);
                q4_k_step(&mut acc, b, a.d, &sumi, &summ);
            }
        }
        _ => {
            for (b, a) in group.as_chunks::<R8_Q6_K>().0.iter().zip(xk) {
                let isum = q6_k_sums(b, a);
                q6_k_step(&mut acc, b, a.d, &isum);
            }
        }
    }
    *out = acc;
}

// ---------------------------------------------------------------------------
// Dispatch.

fn kview(x: &QuantizedAct) -> &[KBlock] {
    x.k.as_deref().unwrap_or_else(|| {
        panic!(
            "r8: needs input len multiple of 256, got {} (or the 256-block view wasn't built)",
            x.len()
        )
    })
}

/// out[8g + r] = row r of group g of `groups` (whole r8 groups of
/// `qtype_id`, width x.len()) · x, for the first out.len() / 8 groups.
/// `groups` may run on past them: software prefetch (`pf` bytes ahead, 0 =
/// off) reads up to its end and never beyond, so callers bound it to the
/// rows they own.
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
    let xk = kview(x);
    let gb = group_bytes(qtype_id, x.len());
    let ng = out.len() / 8;
    assert!(
        out.len() == 8 * ng && groups.len() >= ng * gb,
        "r8 gemv: sizes"
    );
    match simd {
        Simd::Scalar => {
            for (g, o) in groups.chunks_exact(gb).zip(out.as_chunks_mut::<8>().0) {
                group_scalar(qtype_id, g, xk, o);
            }
        }
        // SAFETY (below): guaranteed by our caller.
        #[cfg(target_arch = "x86_64")]
        Simd::Avx2 => unsafe {
            if qtype_id == Q4_K {
                x86::avx2::gemv_q4_k(groups, gb, xk, out, pf)
            } else {
                x86::avx2::gemv_q6_k(groups, gb, xk, out, pf)
            }
        },
        #[cfg(target_arch = "x86_64")]
        Simd::Vnni => unsafe {
            if qtype_id == Q4_K {
                x86::vnni::gemv_q4_k(groups, gb, xk, out, pf)
            } else {
                x86::vnni::gemv_q6_k(groups, gb, xk, out, pf)
            }
        },
    }
}

/// `gemm`'s software prefetch distance in bytes for `nt` tokens: decode's
/// (`AXON_PREFETCH`) up to `GEMM_PREFETCH_MAX_NT` tokens, where a pass is
/// still bound by reading the weights (a speculative verify), else 0 (a
/// prefill chunk computes long enough per block for the hardware
/// prefetcher). AXON_GEMM_PREFETCH=0 turns it off (A/B; same bits).
fn gemm_prefetch(nt: usize) -> usize {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("AXON_GEMM_PREFETCH").map_or(true, |v| v != "0"));
    if on && nt <= GEMM_PREFETCH_MAX_NT {
        crate::quant::DecodeCfg::get().prefetch
    } else {
        0
    }
}

/// Most tokens `gemm` prefetches for (see `gemm_prefetch`).
const GEMM_PREFETCH_MAX_NT: usize = 16;

thread_local! {
    /// Per-token f32 accumulators (8 rows) for the GEMM.
    static ACC: std::cell::RefCell<Vec<[f32; 8]>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Batched: out[(8g + r) * stride + t] = row r of group g · xs[t] for
/// t < xs.len() <= stride (row-major over (row, token slot); slots past
/// xs.len() are left alone), bit for bit `gemv` per token.
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
    assert!(nt <= stride, "r8 gemm: more tokens than slots");
    if nt == 0 {
        return;
    }
    let n = xs[0].len();
    let xk: Vec<&[KBlock]> = xs.iter().map(|x| kview(x)).collect();
    assert!(xs.iter().all(|x| x.len() == n), "r8 gemm: ragged batch");
    let gb = group_bytes(qtype_id, n);
    let ng = out.len() / (8 * stride);
    assert!(
        out.len() == 8 * stride * ng && groups.len() >= ng * gb,
        "r8 gemm: sizes"
    );
    if simd == Simd::Scalar {
        let mut o = [0f32; 8];
        for (g, og) in groups
            .chunks_exact(gb)
            .zip(out.chunks_exact_mut(8 * stride))
        {
            for (t, x) in xk.iter().enumerate() {
                group_scalar(qtype_id, g, x, &mut o);
                for r in 0..8 {
                    og[r * stride + t] = o[r];
                }
            }
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    ACC.with_borrow_mut(|acc| {
        if acc.len() < nt {
            acc.resize(nt, [0f32; 8]);
        }
        let acc = &mut acc[..nt];
        // Software prefetch as decode's (`gemv`): a small batch is still
        // a stream over the weights, and the hardware prefetcher alone
        // leaves DRAM idle while a block's tiles compute.
        let pf = gemm_prefetch(nt);
        for (gi, og) in out.chunks_exact_mut(8 * stride).enumerate().take(ng) {
            let base = gi * gb;
            // SAFETY (below): guaranteed by our caller.
            match (simd, qtype_id) {
                (Simd::Vnni, Q4_K) => unsafe {
                    x86::vnni::gemm_q4_k(groups, base, gb, &xk, acc, pf)
                },
                (Simd::Vnni, _) => unsafe { x86::vnni::gemm_q6_k(groups, base, gb, &xk, acc, pf) },
                (_, Q4_K) => unsafe { x86::avx2::gemm_q4_k(groups, base, gb, &xk, acc, pf) },
                (_, _) => unsafe { x86::avx2::gemm_q6_k(groups, base, gb, &xk, acc, pf) },
            }
            for (t, a) in acc.iter().enumerate() {
                for r in 0..8 {
                    og[r * stride + t] = a[r];
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// x86 kernels: one module per feature set, generated by `kernels!`.

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{KBlock, Q4_MN, Q4_QS, Q4_SC, Q6_Q, Q6_SC, R8_Q4_K, R8_Q6_K};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    fn ld32(b: &[u8], off: usize) -> __m256i {
        let p: &[u8; 32] = b[off..off + 32].try_into().unwrap();
        // SAFETY: p points to 32 readable bytes; loadu has no alignment requirement.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn ld16(b: &[u8], off: usize) -> __m128i {
        let p: &[u8; 16] = b[off..off + 16].try_into().unwrap();
        // SAFETY: as ld32 (16 bytes).
        unsafe { _mm_loadu_si128(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn ldv(p: &[u8; 32]) -> __m256i {
        // SAFETY: as ld32.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn stv(p: &mut [u8; 32], v: __m256i) {
        // SAFETY: p points to 32 writable bytes; storeu has no alignment requirement.
        unsafe { _mm256_storeu_si256(p.as_mut_ptr().cast(), v) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn ldf(p: &[f32; 8]) -> __m256 {
        // SAFETY: 32 readable bytes.
        unsafe { _mm256_loadu_ps(p.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn stf(p: &mut [f32; 8], v: __m256) {
        // SAFETY: 32 writable bytes.
        unsafe { _mm256_storeu_ps(p.as_mut_ptr(), v) }
    }

    /// a.qs[k..k + 4] broadcast to every i32 lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn bc(a: &KBlock, k: usize) -> __m256i {
        let q: &[i8; 4] = a.qs[k..k + 4].try_into().unwrap();
        _mm256_set1_epi32(i32::from_le_bytes([
            q[0] as u8, q[1] as u8, q[2] as u8, q[3] as u8,
        ]))
    }

    /// i32 lane r = bsum32(2c) | bsum32(2c + 1) << 16 (two i16), every lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn bsum_pair(a: &KBlock, c: usize) -> __m256i {
        let lo = super::bsum32(a, 2 * c) as u16 as u32;
        let hi = super::bsum32(a, 2 * c + 1) as u16 as u32;
        _mm256_set1_epi32((lo | hi << 16) as i32)
    }

    /// Σ_c m_{2c+h} * bsum32(2c + h) per row lane: the Q4_K summ.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn q4_k_summ(b: &[u8; R8_Q4_K], a: &KBlock) -> __m256i {
        let mut s = _mm256_setzero_si256();
        for c in 0..4 {
            let m = _mm256_cvtepu8_epi16(ld16(b, Q4_MN + 16 * c));
            s = _mm256_add_epi32(s, _mm256_madd_epi16(m, bsum_pair(a, c)));
        }
        s
    }

    /// acc + ((d * ad) * f32(sumi) - (dmin * ad) * f32(summ)), lane-wise.
    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    fn q4_k_step(acc: __m256, b: &[u8; R8_Q4_K], ad: f32, sumi: __m256i, summ: __m256i) -> __m256 {
        let ad = _mm256_set1_ps(ad);
        let dd = _mm256_mul_ps(_mm256_cvtph_ps(ld16(b, 0)), ad);
        let dm = _mm256_mul_ps(_mm256_cvtph_ps(ld16(b, 16)), ad);
        let t = _mm256_sub_ps(
            _mm256_mul_ps(dd, _mm256_cvtepi32_ps(sumi)),
            _mm256_mul_ps(dm, _mm256_cvtepi32_ps(summ)),
        );
        _mm256_add_ps(acc, t)
    }

    /// acc + (d * ad) * f32(isum), lane-wise.
    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    fn q6_k_step(acc: __m256, b: &[u8; R8_Q6_K], ad: f32, isum: __m256i) -> __m256 {
        let dd = _mm256_mul_ps(_mm256_cvtph_ps(ld16(b, 0)), _mm256_set1_ps(ad));
        _mm256_add_ps(acc, _mm256_mul_ps(dd, _mm256_cvtepi32_ps(isum)))
    }

    /// Q4_K scale pair c: (sc[2c], sc[2c + 1]) per row as i32 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn q4_k_scales(b: &[u8; R8_Q4_K], c: usize) -> (__m256i, __m256i) {
        let v = _mm256_cvtepu8_epi16(ld16(b, Q4_SC + 16 * c));
        (
            _mm256_and_si256(v, _mm256_set1_epi32(0xffff)),
            _mm256_srli_epi32::<16>(v),
        )
    }

    /// Q6_K scale pair p: (sc[2p], sc[2p + 1]) per row as sign-extended i32 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn q6_k_scales(b: &[u8; R8_Q6_K], p: usize) -> (__m256i, __m256i) {
        let v = _mm256_cvtepi8_epi16(ld16(b, Q6_SC + 16 * p));
        (
            _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(v)),
            _mm256_srai_epi32::<16>(v),
        )
    }

    /// The 4 Q6_K weight vectors (K0, K0+32, K0+64, K0+96) of (h, g).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn q6_k_weights(b: &[u8; R8_Q6_K], h: usize, g: usize) -> [__m256i; 4] {
        let o = Q6_Q + 96 * (8 * h + g);
        let (qa, qb, qh) = (ld32(b, o), ld32(b, o + 32), ld32(b, o + 64));
        let m15 = _mm256_set1_epi8(15);
        let m3 = _mm256_set1_epi8(3);
        [
            _mm256_or_si256(
                _mm256_and_si256(qa, m15),
                _mm256_slli_epi16::<4>(_mm256_and_si256(qh, m3)),
            ),
            _mm256_or_si256(
                _mm256_and_si256(qb, m15),
                _mm256_slli_epi16::<2>(_mm256_and_si256(qh, _mm256_set1_epi8(12))),
            ),
            _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16::<4>(qa), m15),
                _mm256_and_si256(qh, _mm256_set1_epi8(48)),
            ),
            _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16::<4>(qb), m15),
                _mm256_srli_epi16::<2>(_mm256_and_si256(qh, _mm256_set1_epi8(-64))),
            ),
        ]
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

    /// Integer dot of u8 weights x i8 activations, 4 per i32 lane, added to acc.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn dp_avx2(acc: __m256i, w: __m256i, a: __m256i) -> __m256i {
        _mm256_add_epi32(
            acc,
            _mm256_madd_epi16(_mm256_maddubs_epi16(w, a), _mm256_set1_epi16(1)),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    fn dp_vnni(acc: __m256i, w: __m256i, a: __m256i) -> __m256i {
        _mm256_dpbusd_avx_epi32(acc, w, a)
    }

    /// Weights of one block unpacked to one byte per value, k/4-th vector at
    /// index k/4, plus the per-sub-block scales as i32 lanes. Shared by every
    /// token of a GEMM.
    pub struct Unpacked {
        w: [[u8; 32]; 64],
        sc: [[u8; 32]; 16],
    }

    impl Unpacked {
        const ZERO: Unpacked = Unpacked {
            w: [[0; 32]; 64],
            sc: [[0; 32]; 16],
        };
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn unpack_q4_k(b: &[u8; R8_Q4_K], u: &mut Unpacked) {
        let m4 = _mm256_set1_epi8(15);
        for c in 0..4 {
            for g in 0..8 {
                let q = ld32(b, Q4_QS + 32 * (8 * c + g));
                // sub-block 2c is vectors 16c..16c+8, 2c + 1 the next 8
                stv(&mut u.w[16 * c + g], _mm256_and_si256(q, m4));
                stv(
                    &mut u.w[16 * c + 8 + g],
                    _mm256_and_si256(_mm256_srli_epi16::<4>(q), m4),
                );
            }
            let (s0, s1) = q4_k_scales(b, c);
            stv(&mut u.sc[2 * c], s0);
            stv(&mut u.sc[2 * c + 1], s1);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn unpack_q6_k(b: &[u8; R8_Q6_K], u: &mut Unpacked) {
        for h in 0..2 {
            for g in 0..8 {
                let w = q6_k_weights(b, h, g);
                for (q, wq) in w.iter().enumerate() {
                    // k = 128h + 32q + 4g + i
                    stv(&mut u.w[32 * h + 8 * q + g], *wq);
                }
            }
        }
        for p in 0..8 {
            let (s0, s1) = q6_k_scales(b, p);
            stv(&mut u.sc[2 * p], s0);
            stv(&mut u.sc[2 * p + 1], s1);
        }
    }

    macro_rules! kernels {
        ($m:ident, $feat:literal, $dp:ident) => {
            pub mod $m {
                use super::*;

                /// GEMV over whole Q4_K groups (`gb` bytes each), one token.
                ///
                /// # Safety
                /// The CPU must support the enabled target features.
                #[allow(clippy::needless_range_loop)]
                #[target_feature(enable = $feat)]
                pub unsafe fn gemv_q4_k(
                    groups: &[u8],
                    gb: usize,
                    xk: &[KBlock],
                    out: &mut [f32],
                    pf: usize,
                ) {
                    let m4 = _mm256_set1_epi8(15);
                    let z = _mm256_setzero_si256();
                    for (gi, o) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let base = gi * gb;
                        let blocks = groups[base..base + gb].as_chunks::<R8_Q4_K>().0;
                        let mut acc = _mm256_setzero_ps();
                        for (sb, (b, a)) in blocks.iter().zip(xk).enumerate() {
                            // the block `pf` bytes ahead, a quarter per c
                            let ahead = base + sb * R8_Q4_K + pf;
                            let mut sumi = z;
                            for c in 0..4 {
                                if pf > 0 {
                                    let part = R8_Q4_K / 4;
                                    prefetch(groups, ahead + c * part, part);
                                }
                                // even/odd g in separate chains
                                let (mut s0, mut s1, mut t0, mut t1) = (z, z, z, z);
                                for g in (0..8).step_by(2) {
                                    let q = ld32(b, Q4_QS + 32 * (8 * c + g));
                                    let k = 64 * c + 4 * g;
                                    s0 = $dp(s0, _mm256_and_si256(q, m4), bc(a, k));
                                    t0 = $dp(
                                        t0,
                                        _mm256_and_si256(_mm256_srli_epi16::<4>(q), m4),
                                        bc(a, k + 32),
                                    );
                                    let q = ld32(b, Q4_QS + 32 * (8 * c + g + 1));
                                    s1 = $dp(s1, _mm256_and_si256(q, m4), bc(a, k + 4));
                                    t1 = $dp(
                                        t1,
                                        _mm256_and_si256(_mm256_srli_epi16::<4>(q), m4),
                                        bc(a, k + 36),
                                    );
                                }
                                let (sc0, sc1) = q4_k_scales(b, c);
                                sumi = _mm256_add_epi32(
                                    sumi,
                                    _mm256_add_epi32(
                                        _mm256_mullo_epi32(_mm256_add_epi32(s0, s1), sc0),
                                        _mm256_mullo_epi32(_mm256_add_epi32(t0, t1), sc1),
                                    ),
                                );
                            }
                            acc = q4_k_step(acc, b, a.d, sumi, q4_k_summ(b, a));
                        }
                        stf(o, acc);
                    }
                }

                /// GEMV over whole Q6_K groups, one token.
                ///
                /// # Safety
                /// The CPU must support the enabled target features.
                #[allow(clippy::needless_range_loop)]
                #[target_feature(enable = $feat)]
                pub unsafe fn gemv_q6_k(
                    groups: &[u8],
                    gb: usize,
                    xk: &[KBlock],
                    out: &mut [f32],
                    pf: usize,
                ) {
                    for (gi, o) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let base = gi * gb;
                        let blocks = groups[base..base + gb].as_chunks::<R8_Q6_K>().0;
                        let mut acc = _mm256_setzero_ps();
                        for (sb, (b, a)) in blocks.iter().zip(xk).enumerate() {
                            // the block `pf` bytes ahead, a quarter per (h, half)
                            let ahead = base + sb * R8_Q6_K + pf;
                            let mut isum = _mm256_setzero_si256();
                            for h in 0..2 {
                                for half in 0..2 {
                                    if pf > 0 {
                                        let part = R8_Q6_K / 4;
                                        prefetch(groups, ahead + (2 * h + half) * part, part);
                                    }
                                    // sub-blocks j = 8h + 2q + half, q = 0..4
                                    let mut s = [_mm256_setzero_si256(); 4];
                                    for q in 0..4 {
                                        let j = 8 * h + 2 * q + half;
                                        s[q] = _mm256_set1_epi32(-32 * a.bsums[j] as i32);
                                    }
                                    for g in 4 * half..4 * half + 4 {
                                        let w = q6_k_weights(b, h, g);
                                        let k = 128 * h + 4 * g;
                                        for q in 0..4 {
                                            s[q] = $dp(s[q], w[q], bc(a, k + 32 * q));
                                        }
                                    }
                                    for q in 0..4 {
                                        let (sc0, sc1) = q6_k_scales(b, 4 * h + q);
                                        let sc = if half == 0 { sc0 } else { sc1 };
                                        isum = _mm256_add_epi32(isum, _mm256_mullo_epi32(s[q], sc));
                                    }
                                }
                            }
                            acc = q6_k_step(acc, b, a.d, isum);
                        }
                        stf(o, acc);
                    }
                }

                /// sumi for T tokens over an unpacked block: NS sub-blocks
                /// of VPS vectors. Q6 starts each sub-block at -32 bsums[j].
                #[allow(clippy::needless_range_loop)]
                #[inline]
                #[target_feature(enable = $feat)]
                fn tile<const T: usize, const NS: usize, const VPS: usize, const Q6: bool>(
                    u: &Unpacked,
                    x: &[&KBlock; T],
                ) -> [__m256i; T] {
                    let mut sumi = [_mm256_setzero_si256(); T];
                    for j in 0..NS {
                        let mut s = [_mm256_setzero_si256(); T];
                        if Q6 {
                            for t in 0..T {
                                s[t] = _mm256_set1_epi32(-32 * x[t].bsums[j] as i32);
                            }
                        }
                        for v in VPS * j..VPS * j + VPS {
                            let w = ldv(&u.w[v]);
                            for t in 0..T {
                                s[t] = $dp(s[t], w, bc(x[t], 4 * v));
                            }
                        }
                        let sc = ldv(&u.sc[j]);
                        for t in 0..T {
                            sumi[t] = _mm256_add_epi32(sumi[t], _mm256_mullo_epi32(s[t], sc));
                        }
                    }
                    sumi
                }

                /// T tokens starting at t0 through one Q4_K block.
                #[allow(clippy::needless_range_loop)]
                #[inline]
                #[target_feature(enable = $feat)]
                fn tokens_q4_k<const T: usize>(
                    u: &Unpacked,
                    b: &[u8; R8_Q4_K],
                    xk: &[&[KBlock]],
                    sb: usize,
                    t0: usize,
                    acc: &mut [[f32; 8]],
                ) {
                    let mut x = [&xk[t0][sb]; T];
                    for t in 1..T {
                        x[t] = &xk[t0 + t][sb];
                    }
                    let sumi = tile::<T, 8, 8, false>(u, &x);
                    for t in 0..T {
                        let a = ldf(&acc[t0 + t]);
                        let r = q4_k_step(a, b, x[t].d, sumi[t], q4_k_summ(b, x[t]));
                        stf(&mut acc[t0 + t], r);
                    }
                }

                /// T tokens starting at t0 through one Q6_K block.
                #[allow(clippy::needless_range_loop)]
                #[inline]
                #[target_feature(enable = $feat)]
                fn tokens_q6_k<const T: usize>(
                    u: &Unpacked,
                    b: &[u8; R8_Q6_K],
                    xk: &[&[KBlock]],
                    sb: usize,
                    t0: usize,
                    acc: &mut [[f32; 8]],
                ) {
                    let mut x = [&xk[t0][sb]; T];
                    for t in 1..T {
                        x[t] = &xk[t0 + t][sb];
                    }
                    let isum = tile::<T, 16, 4, true>(u, &x);
                    for t in 0..T {
                        let a = ldf(&acc[t0 + t]);
                        stf(&mut acc[t0 + t], q6_k_step(a, b, x[t].d, isum[t]));
                    }
                }

                /// One Q4_K group x every token: acc[t][r] (acc.len() == xk.len()).
                /// Each block is unpacked once for all tokens; tokens go 4 at a time.
                ///
                /// # Safety
                /// The CPU must support the enabled target features.
                #[target_feature(enable = $feat)]
                pub unsafe fn gemm_q4_k(
                    groups: &[u8],
                    base: usize,
                    gb: usize,
                    xk: &[&[KBlock]],
                    acc: &mut [[f32; 8]],
                    pf: usize,
                ) {
                    let nt = xk.len();
                    acc.fill([0.0; 8]);
                    let mut u = Unpacked::ZERO;
                    let group = &groups[base..base + gb];
                    for (sb, b) in group.as_chunks::<R8_Q4_K>().0.iter().enumerate() {
                        if pf > 0 {
                            prefetch(groups, base + sb * R8_Q4_K + pf, R8_Q4_K);
                        }
                        unpack_q4_k(b, &mut u);
                        let mut t0 = 0;
                        while t0 + 4 <= nt {
                            tokens_q4_k::<4>(&u, b, xk, sb, t0, acc);
                            t0 += 4;
                        }
                        match nt - t0 {
                            1 => tokens_q4_k::<1>(&u, b, xk, sb, t0, acc),
                            2 => tokens_q4_k::<2>(&u, b, xk, sb, t0, acc),
                            3 => tokens_q4_k::<3>(&u, b, xk, sb, t0, acc),
                            _ => {}
                        }
                    }
                }

                /// One Q6_K group x every token, as gemm_q4_k.
                ///
                /// # Safety
                /// The CPU must support the enabled target features.
                #[target_feature(enable = $feat)]
                pub unsafe fn gemm_q6_k(
                    groups: &[u8],
                    base: usize,
                    gb: usize,
                    xk: &[&[KBlock]],
                    acc: &mut [[f32; 8]],
                    pf: usize,
                ) {
                    let nt = xk.len();
                    acc.fill([0.0; 8]);
                    let mut u = Unpacked::ZERO;
                    let group = &groups[base..base + gb];
                    for (sb, b) in group.as_chunks::<R8_Q6_K>().0.iter().enumerate() {
                        if pf > 0 {
                            prefetch(groups, base + sb * R8_Q6_K + pf, R8_Q6_K);
                        }
                        unpack_q6_k(b, &mut u);
                        let mut t0 = 0;
                        while t0 + 4 <= nt {
                            tokens_q6_k::<4>(&u, b, xk, sb, t0, acc);
                            t0 += 4;
                        }
                        match nt - t0 {
                            1 => tokens_q6_k::<1>(&u, b, xk, sb, t0, acc),
                            2 => tokens_q6_k::<2>(&u, b, xk, sb, t0, acc),
                            3 => tokens_q6_k::<3>(&u, b, xk, sb, t0, acc),
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
