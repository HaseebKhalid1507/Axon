//! 8-row interleaved ("r8-bi") Q4_K / Q6_K prefill GEMM, bit-identical to the
//! per-row kernels in quant.rs (`q4_k_avx2` / `q6_k_avx2` and their 4-token /
//! row-pair forms, which all agree bit for bit).
//!
//! Reference order, per (row, token), for each super-block in order:
//!   i32 lane l (0..8) = Σ_c sc(c, l) · Σ_{i<4} u[32c + 4l + i] · a[32c + 4l + i]
//!     (c = 32-wide chunk 0..8; Q4_K sc(c, l) = sc[c]; Q6_K sc(c, l) =
//!     sc16[2c + (l >= 4)], and the lane starts at -32 (bs[2l] sc16[2l] +
//!     bs[2l+1] sc16[2l+1]))
//!   Q4_K min lane l (0..4) = min[2l] q8s[2l] + min[2l+1] q8s[2l+1],
//!     q8s[j] = bs[2j] + bs[2j+1]
//!   acc_l = fma(a.d * d, f32(lane l), acc_l); accm_l = fma(-a.d * dmin, f32(min l), accm_l)
//! then ((acc0+acc4)+(acc2+acc6)) + ((acc1+acc5)+(acc3+acc7)) [+ ((m0+m2)+(m1+m3))].
//!
//! Here the 8 rows of a group ride the 8 i32 lanes of a vector instead: the
//! reference lane l of every row is the vector S_l (lane r = row r), so each
//! S_l holds exactly the reference integer, and the per-lane f32 ops are the
//! reference's, lane-parallel over rows: S_l is converted and fma'd into
//! ACC_l with dd = f32(d_r) * a.d per lane (the same product as the scalar
//! a.d * d), the mins into ACCM_l, and the final hsum tree becomes 11
//! vertical adds whose lane r is row r's result. Integer sums are exact (no
//! saturation: maddubs pairs <= 2*63*128 = 16128), so any integer grouping
//! gives the same i32; only the float order matters, and it is unchanged.
//!
//! Layout (the r8 layout of branch numerics/r8): a matrix with rows % 8 == 0
//! is rows / 8 groups; a group is its super-blocks in order, each a
//! fixed-size block for 8 rows. In every 32-byte quant vector, bytes
//! 4r..4r+4 are 4 consecutive k values of row r.
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
//! GEMM: per group, every super-block is unpacked once into thread-local
//! scratch (`UBlock`, 2.9 KB per super-block: one byte per weight, per-row
//! i16 scale pairs; ~100 KB at width 8960, so it lives in L2), then tokens
//! go 8 at a time (then 4, then 1..3) over all blocks, ACC (12 vectors per
//! token) in a stack tile that stays in L1.
//!
//! GEMV (decode): one token, each group streamed once from the r8 layout
//! with no unpack, ACC on the stack; the same lanes and f32 ops, so decode
//! and prefill both read only this copy and the original can be released.

use super::{
    K_SCALE_SIZE, KBlock, Q4_K, Q4_K_BYTES, Q6_K, Q6_K_BYTES, QK_K, QuantizedAct, Simd,
    get_scale_min_k4,
};

pub const R8_Q4_K: usize = 1184;
pub const R8_Q6_K: usize = 1680;

const Q4_SC: usize = 32;
const Q4_MN: usize = 96;
const Q4_QS: usize = 160;
const Q6_SC: usize = 16;
const Q6_Q: usize = 144;

fn block_bytes(qtype_id: u32) -> Option<usize> {
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
    block_bytes(qtype_id).expect("r8bi: Q4_K/Q6_K only") * (n / QK_K)
}

/// Whether prefill should repack its K-quant matrices: on unless
/// AXON_REPACK=0, and only where an r8bi kernel runs (AVX2 or VNNI).
pub fn enabled() -> bool {
    std::env::var("AXON_REPACK").map_or(true, |v| v != "0") && available()
}

/// Whether this CPU runs an r8bi kernel (AVX2 or VNNI).
pub fn available() -> bool {
    super::simd() != Simd::Scalar
}

/// `b` as 4-byte words (a dword-sized copy is one move, no per-byte checks).
fn words<const N: usize>(b: &[u8]) -> &[[u8; 4]; N] {
    b.as_chunks::<4>().0.try_into().unwrap()
}

fn words_mut<const N: usize>(b: &mut [u8]) -> &mut [[u8; 4]; N] {
    b.as_chunks_mut::<4>().0.try_into().unwrap()
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
    }
    // qs word k (bytes 4k..4k+4, k = 8c + g) of row r -> word 8k + r
    let dst = words_mut::<256>(&mut out[Q4_QS..]);
    for (r, b) in src.iter().enumerate() {
        let q = words::<32>(&b[4 + K_SCALE_SIZE..]);
        for (k, &w) in q.iter().enumerate() {
            dst[8 * k + r] = w;
        }
    }
}

fn repack_q6_k_block(src: [&[u8; Q6_K_BYTES]; 8], out: &mut [u8; R8_Q6_K]) {
    for (r, b) in src.iter().enumerate() {
        out[2 * r..2 * r + 2].copy_from_slice(&b[208..210]);
        let sc = words::<4>(&b[192..208]);
        // scale pair p (bytes 2p, 2p+1) of row r -> bytes Q6_SC + 16p + 2r
        for p in 0..8 {
            let w = sc[p / 2];
            let o = Q6_SC + 16 * p + 2 * r;
            out[o..o + 2].copy_from_slice(&w[2 * (p % 2)..2 * (p % 2) + 2]);
        }
    }
    // per (h, g), 24 words: A (ql word 16h + g), B (16h + 8 + g), H (qh word
    // 8h + g), each 8 rows
    let dst = words_mut::<384>(&mut out[Q6_Q..]);
    for (r, b) in src.iter().enumerate() {
        let q = words::<48>(&b[..192]);
        for h in 0..2 {
            for g in 0..8 {
                let o = 24 * (8 * h + g) + r;
                dst[o] = q[16 * h + g];
                dst[o + 8] = q[16 * h + 8 + g];
                dst[o + 16] = q[32 + 8 * h + g];
            }
        }
    }
}

/// Repack whole groups: `out` (a multiple of `group_bytes`) gets groups
/// `g0..` of the row-major `data` (rows of width `n`).
pub fn repack_groups(qtype_id: u32, data: &[u8], n: usize, g0: usize, out: &mut [u8]) {
    let nsb = n / QK_K;
    let rb = super::row_bytes(qtype_id, n);
    let gb = group_bytes(qtype_id, n);
    assert!(out.len().is_multiple_of(gb), "r8bi: partial group");
    assert!(
        data.len() >= (g0 * 8 + out.len() / gb * 8) * rb,
        "r8bi: short tensor data"
    );
    for (i, og) in out.chunks_exact_mut(gb).enumerate() {
        let grp = g0 + i;
        let rows: [&[u8]; 8] =
            [0, 1, 2, 3, 4, 5, 6, 7].map(|r| &data[(8 * grp + r) * rb..(8 * grp + r + 1) * rb]);
        if qtype_id == Q4_K {
            let ob = og.as_chunks_mut::<R8_Q4_K>().0;
            for (sb, o) in ob.iter_mut().enumerate().take(nsb) {
                let src = rows.map(|r| &r.as_chunks::<Q4_K_BYTES>().0[sb]);
                repack_q4_k_block(src, o);
            }
        } else {
            let ob = og.as_chunks_mut::<R8_Q6_K>().0;
            for (sb, o) in ob.iter_mut().enumerate().take(nsb) {
                let src = rows.map(|r| &r.as_chunks::<Q6_K_BYTES>().0[sb]);
                repack_q6_k_block(src, o);
            }
        }
    }
}

/// Bytes of the r8bi form of a `rows` x `n` matrix, or None if it can't be
/// repacked.
pub fn repacked_len(qtype_id: u32, rows: usize, n: usize) -> Option<usize> {
    can_repack(qtype_id, rows, n).then(|| rows / 8 * group_bytes(qtype_id, n))
}

/// Where repacked matrices live: one anonymous mapping for all of a model's,
/// with transparent huge pages requested. Decode streams every byte per
/// token, and on 4 KB pages the page walks cost more than the arithmetic
/// (the original weights' copy does the same: gguf.rs `map_weights`). One
/// mapping, not one per matrix: a 2 MB page per small matrix (attn_k is
/// 220 KB on the 1.5B) would waste most of it.
pub struct Arena(memmap2::MmapMut);

impl Arena {
    /// `len` zeroed bytes (the mapping is rounded up to whole 2 MB pages:
    /// the kernel aligns such mappings to 2 MB, so all of it can be huge).
    /// None if the mapping fails (callers fall back to the per-row kernels).
    pub fn new(len: usize) -> Option<Arena> {
        const HUGE: usize = 2 << 20;
        let map = memmap2::MmapMut::map_anon(len.max(1).next_multiple_of(HUGE)).ok()?;
        // Advise before touching any page (THP=madvise turns huge pages on
        // here; defrag=defer+madvise compacts for them). Failure only costs
        // speed.
        #[cfg(target_os = "linux")]
        let _ = map.advise(memmap2::Advice::HugePage);
        Some(Arena(map))
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }

    /// Read-only from here on, shared by the matrices placed in it.
    pub fn freeze(self) -> std::sync::Arc<memmap2::Mmap> {
        std::sync::Arc::new(self.0.make_read_only().expect("r8bi: mprotect"))
    }
}

/// A repacked matrix: `rows` x `n` of `qtype_id`, `rows / 8` groups, at
/// bytes off..off + len of an `Arena`.
pub struct Mat {
    qtype_id: u32,
    rows: usize,
    n: usize,
    buf: std::sync::Arc<memmap2::Mmap>,
    off: usize,
    len: usize,
}

impl Mat {
    /// The matrix whose r8bi bytes (filled by `repack_groups`) are at `off`
    /// in `buf`.
    pub fn in_arena(
        buf: &std::sync::Arc<memmap2::Mmap>,
        off: usize,
        qtype_id: u32,
        rows: usize,
        n: usize,
    ) -> Mat {
        let len = repacked_len(qtype_id, rows, n).expect("r8bi: can't repack");
        assert!(off + len <= buf.len(), "r8bi: matrix outside its arena");
        Mat {
            qtype_id,
            rows,
            n,
            buf: buf.clone(),
            off,
            len,
        }
    }

    /// Single-threaded repack of a whole matrix into its own arena (tests;
    /// model.rs puts all of them in one, repacked on the pool).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(qtype_id: u32, data: &[u8], rows: usize, n: usize) -> Option<Mat> {
        let len = repacked_len(qtype_id, rows, n)?;
        let mut a = Arena::new(len)?;
        repack_groups(qtype_id, data, n, 0, &mut a.bytes_mut()[..len]);
        Some(Mat::in_arena(&a.freeze(), 0, qtype_id, rows, n))
    }

    /// A matrix from already-repacked bytes (tests: mutated layouts).
    #[cfg(test)]
    fn from_repacked(qtype_id: u32, rows: usize, n: usize, bytes: &[u8]) -> Mat {
        let mut a = Arena::new(bytes.len()).expect("r8bi: map");
        a.bytes_mut()[..bytes.len()].copy_from_slice(bytes);
        Mat::in_arena(&a.freeze(), 0, qtype_id, rows, n)
    }

    /// The r8bi bytes.
    pub fn data(&self) -> &[u8] {
        &self.buf[self.off..self.off + self.len]
    }

    pub fn group_bytes(&self) -> usize {
        group_bytes(self.qtype_id, self.n)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Resident bytes of the repacked copy.
    pub fn bytes(&self) -> usize {
        self.len
    }
}

fn kview(x: &QuantizedAct) -> &[KBlock] {
    x.k.as_deref().unwrap_or_else(|| {
        panic!(
            "r8bi: needs input len multiple of 256, got {} (or the 256-block view wasn't built)",
            x.len()
        )
    })
}

/// out[(r - first) * stride + t] = row r of `m` · xs[t], for the rows
/// first..first + out.len().div_ceil(stride) (whole groups: `first` and the
/// row count multiples of 8) and every t < xs.len() <= stride. Bit for bit
/// `dot_row_q8(qtype, row r, xs[t])`.
pub fn gemm(m: &Mat, first: usize, xs: &[&QuantizedAct], out: &mut [f32], stride: usize) {
    // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
    unsafe { gemm_with(super::simd(), m, first, xs, out, stride) }
}

/// Decode: out[r - first] = row r of `m` · x for rows first..first +
/// out.len() (whole groups), bit for bit `dot_row_q8(qtype, row r, x)`.
/// Each group is read once, straight from the r8 layout. Prefetch runs `pf`
/// bytes ahead (0 = off) but never past row `pf_end` (a multiple of 8: the
/// end of the rows the caller owns).
pub fn gemv(m: &Mat, first: usize, x: &QuantizedAct, out: &mut [f32], pf: usize, pf_end: usize) {
    // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
    unsafe { gemv_with(super::simd(), m, first, x, out, pf, pf_end) }
}

/// `gemv` on an explicit path.
///
/// # Safety
/// As for `gemm_with`.
#[allow(unused_variables)]
pub(super) unsafe fn gemv_with(
    simd: Simd,
    m: &Mat,
    first: usize,
    x: &QuantizedAct,
    out: &mut [f32],
    pf: usize,
    pf_end: usize,
) {
    let n_rows = out.len();
    assert!(
        first.is_multiple_of(8) && n_rows.is_multiple_of(8) && first + n_rows <= m.rows,
        "r8bi: rows {first}..{} aren't whole groups of {}",
        first + n_rows,
        m.rows
    );
    assert_eq!(x.len(), m.n, "r8bi: input len");
    let xk = kview(x);
    let gb = m.group_bytes();
    let end = pf_end
        .clamp(first + n_rows, m.rows)
        .next_multiple_of(8)
        .min(m.rows);
    let groups = &m.data()[first / 8 * gb..end / 8 * gb];
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY (unsafe calls below): guaranteed by our caller.
        match (m.qtype_id, simd) {
            (Q4_K, Simd::Vnni) => unsafe { x86::vnni::gemv_q4_k(groups, gb, xk, out, pf) },
            (Q4_K, Simd::Avx2) => unsafe { x86::avx2::gemv_q4_k(groups, gb, xk, out, pf) },
            (_, Simd::Vnni) => unsafe { x86::vnni::gemv_q6_k(groups, gb, xk, out, pf) },
            (_, Simd::Avx2) => unsafe { x86::avx2::gemv_q6_k(groups, gb, xk, out, pf) },
            (_, Simd::Scalar) => panic!("r8bi: no scalar kernel"),
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        panic!("r8bi: x86_64 only")
    }
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// The group being multiplied, unpacked (~100 KB at width 8960: L2,
    /// streamed once per token tile); grows to the widest seen.
    static UNPACKED: std::cell::RefCell<Vec<x86::UBlock>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `gemm` on an explicit path.
///
/// # Safety
/// `simd` must be a path whose CPU features were detected (`has_avx2()` for
/// Avx2, `has_vnni()` for Vnni). Scalar panics.
#[allow(unused_variables)]
pub(super) unsafe fn gemm_with(
    simd: Simd,
    m: &Mat,
    first: usize,
    xs: &[&QuantizedAct],
    out: &mut [f32],
    stride: usize,
) {
    let nt = xs.len();
    if nt == 0 || out.is_empty() {
        return;
    }
    let n_rows = out.len().div_ceil(stride);
    assert!(
        first.is_multiple_of(8) && n_rows.is_multiple_of(8) && first + n_rows <= m.rows,
        "r8bi: rows {first}..{} aren't whole groups of {}",
        first + n_rows,
        m.rows
    );
    assert!(nt <= stride, "r8bi: more tokens than the output stride");
    assert!(
        out.len() >= (n_rows - 1) * stride + nt,
        "r8bi: output too short"
    );
    let mut xk: Vec<&[KBlock]> = Vec::with_capacity(nt);
    for x in xs {
        assert_eq!(x.len(), m.n, "r8bi: input len");
        xk.push(kview(x));
    }
    #[cfg(target_arch = "x86_64")]
    {
        let nsb = m.n / QK_K;
        let gb = m.group_bytes();
        UNPACKED.with_borrow_mut(|ub| {
            if ub.len() < nsb {
                ub.resize(nsb, x86::UBlock::ZERO);
            }
            let ub = &mut ub[..nsb];
            for gi in 0..n_rows / 8 {
                let g = first / 8 + gi;
                let grp = &m.data()[g * gb..(g + 1) * gb];
                let o = &mut out[gi * 8 * stride..];
                // SAFETY (unsafe calls below): guaranteed by our caller.
                match (m.qtype_id, simd) {
                    (Q4_K, Simd::Vnni) => unsafe { x86::vnni::gemm_q4_k(grp, ub, &xk, o, stride) },
                    (Q4_K, Simd::Avx2) => unsafe { x86::avx2::gemm_q4_k(grp, ub, &xk, o, stride) },
                    (_, Simd::Vnni) => unsafe { x86::vnni::gemm_q6_k(grp, ub, &xk, o, stride) },
                    (_, Simd::Avx2) => unsafe { x86::avx2::gemm_q6_k(grp, ub, &xk, o, stride) },
                    (_, Simd::Scalar) => panic!("r8bi: no scalar kernel"),
                }
            }
        });
        return;
    }
    #[allow(unreachable_code)]
    {
        panic!("r8bi: x86_64 only")
    }
}

// ---------------------------------------------------------------------------
// x86 kernels: one module per feature set, generated by `kernels!`.

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{KBlock, Q4_MN, Q4_QS, Q4_SC, Q6_Q, Q6_SC, R8_Q4_K, R8_Q6_K};
    use std::arch::x86_64::*;

    /// One super-block of a group, unpacked for every token of the GEMM.
    #[repr(C, align(32))]
    #[derive(Clone)]
    pub struct UBlock {
        /// [8c + g]: bytes 4r..4r+4 = row r's unsigned weights at chunk c,
        /// positions 4g..4g+4 (Q4_K 0..15, Q6_K 0..63 before the -32)
        u: [[u8; 32]; 64],
        /// [h][c]: i16 lanes 2r, 2r+1 = row r's scale for chunk c, used by
        /// lanes g with g / 4 == h (Q6_K: of 16-group 2c + h; Q4_K: the
        /// same in both halves)
        sc: [[[i16; 16]; 8]; 2],
        /// Q4_K [l < 4]: lanes 2r, 2r+1 = row r's (min 2l, min 2l+1);
        /// Q6_K [g]: row r's (-32 sc16[2g], -32 sc16[2g+1]), the chain start
        m: [[i16; 16]; 8],
        d: [f32; 8],
        dmin: [f32; 8],
    }

    impl UBlock {
        pub const ZERO: UBlock = UBlock {
            u: [[0; 32]; 64],
            sc: [[[0; 16]; 8]; 2],
            m: [[0; 16]; 8],
            d: [0.0; 8],
            dmin: [0.0; 8],
        };
    }

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
    fn ldw(p: &[i16; 16]) -> __m256i {
        // SAFETY: 32 readable bytes.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn stw(p: &mut [i16; 16], v: __m256i) {
        // SAFETY: 32 writable bytes.
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

    /// 4 bytes at `p[k..k + 4]` (i8 or i16 data) broadcast to every i32 lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn bc4(p: &[u8], k: usize) -> __m256i {
        let q: [u8; 4] = p[k..k + 4].try_into().unwrap();
        _mm256_set1_epi32(i32::from_le_bytes(q))
    }

    #[inline]
    fn qs_bytes(a: &KBlock) -> &[u8] {
        // SAFETY: i8 and u8 have the same layout.
        unsafe { std::slice::from_raw_parts(a.qs.as_ptr().cast(), a.qs.len()) }
    }

    #[inline]
    fn bsums_bytes(a: &KBlock) -> &[u8] {
        // SAFETY: plain i16 data, read as its little-endian bytes.
        unsafe { std::slice::from_raw_parts(a.bsums.as_ptr().cast(), 2 * a.bsums.len()) }
    }

    /// Dword r = (lo i16 of dword r) twice.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn dup_lo(v: __m256i) -> __m256i {
        _mm256_shufflehi_epi16::<0xA0>(_mm256_shufflelo_epi16::<0xA0>(v))
    }

    /// Dword r = (hi i16 of dword r) twice.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn dup_hi(v: __m256i) -> __m256i {
        _mm256_shufflehi_epi16::<0xF5>(_mm256_shufflelo_epi16::<0xF5>(v))
    }

    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    fn unpack_q4_k(b: &[u8; R8_Q4_K], u: &mut UBlock) {
        stf(&mut u.d, _mm256_cvtph_ps(ld16(b, 0)));
        stf(&mut u.dmin, _mm256_cvtph_ps(ld16(b, 16)));
        let m4 = _mm256_set1_epi8(15);
        for c in 0..4 {
            let p = _mm256_cvtepu8_epi16(ld16(b, Q4_SC + 16 * c));
            // both halves: the kernels index sc[g / 4] for either type, so
            // the scale loads stay in the g loop (as memory operands)
            // instead of 8 hoisted registers
            for h in 0..2 {
                stw(&mut u.sc[h][2 * c], dup_lo(p));
                stw(&mut u.sc[h][2 * c + 1], dup_hi(p));
            }
            stw(&mut u.m[c], _mm256_cvtepu8_epi16(ld16(b, Q4_MN + 16 * c)));
            for g in 0..8 {
                let q = ld32(b, Q4_QS + 32 * (8 * c + g));
                stv(&mut u.u[8 * (2 * c) + g], _mm256_and_si256(q, m4));
                stv(
                    &mut u.u[8 * (2 * c + 1) + g],
                    _mm256_and_si256(_mm256_srli_epi16::<4>(q), m4),
                );
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    fn unpack_q6_k(b: &[u8; R8_Q6_K], u: &mut UBlock) {
        stf(&mut u.d, _mm256_cvtph_ps(ld16(b, 0)));
        for p in 0..8 {
            let s = _mm256_cvtepi8_epi16(ld16(b, Q6_SC + 16 * p));
            stw(&mut u.sc[0][p], dup_lo(s));
            stw(&mut u.sc[1][p], dup_hi(s));
            // -32 * scale; |scale| <= 128, so it fits i16
            stw(
                &mut u.m[p],
                _mm256_sub_epi16(_mm256_setzero_si256(), _mm256_slli_epi16::<5>(s)),
            );
        }
        for h in 0..2 {
            for g in 0..8 {
                let w = q6_weights(b, h, g);
                for (q, &wq) in w.iter().enumerate() {
                    stv(&mut u.u[8 * (4 * h + q) + g], wq);
                }
            }
        }
    }

    /// The 4 Q6_K weight vectors of (h, g): chunk 4h + q, lane g (the same
    /// bit assembly as q6_k_avx2).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn q6_weights(b: &[u8; R8_Q6_K], h: usize, g: usize) -> [__m256i; 4] {
        let m15 = _mm256_set1_epi8(15);
        let m3 = _mm256_set1_epi8(3);
        let o = Q6_Q + 96 * (8 * h + g);
        let (qa, qb, qh) = (ld32(b, o), ld32(b, o + 32), ld32(b, o + 64));
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

    /// Hint `len` bytes at `all[off..]` into L2, clamped to `all` (the rows
    /// the caller owns: hints past them would be other threads' rows).
    /// L2, not L1 (T1, not T0): a whole block of T0 hints ties up the L1
    /// fill buffers the demand loads need; on a P-core, 1.5B decode, 1
    /// thread: T0 14.1G cycles / 64 tokens (head's kernels 13.5G), T1 10.3G.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn prefetch(all: &[u8], off: usize, len: usize) {
        let end = (off + len).min(all.len());
        let mut p = off;
        while p < end {
            // SAFETY: p < all.len(); a prefetch never faults anyway.
            unsafe { _mm_prefetch::<_MM_HINT_T1>(all.as_ptr().add(p).cast()) };
            p += 64;
        }
    }

    /// The final per-row sums of a group's 12 accumulators into out[..8].
    #[inline]
    #[target_feature(enable = "avx2")]
    fn finish<const Q6: bool>(acc: &[[f32; 8]; 12], out: &mut [f32]) {
        let mut f = tree8(&acc[..8]);
        if !Q6 {
            let m = _mm256_add_ps(
                _mm256_add_ps(ldf(&acc[8]), ldf(&acc[10])),
                _mm256_add_ps(ldf(&acc[9]), ldf(&acc[11])),
            );
            f = _mm256_add_ps(f, m);
        }
        let o: &mut [f32; 8] = (&mut out[..8]).try_into().unwrap();
        stf(o, f);
    }

    /// sumi + madd(sc, p16), as the AVX2 K-quant kernels do it.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn scale_acc_avx2(sumi: __m256i, sc: __m256i, p16: __m256i) -> __m256i {
        _mm256_add_epi32(sumi, _mm256_madd_epi16(sc, p16))
    }

    /// The same i32 lanes in one instruction.
    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    fn scale_acc_vnni(sumi: __m256i, sc: __m256i, p16: __m256i) -> __m256i {
        _mm256_dpwssd_avx_epi32(sumi, sc, p16)
    }

    /// ((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7)): hsum_f32x8's tree, lane-wise.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn tree8(a: &[[f32; 8]]) -> __m256 {
        let (v0, v1, v2, v3) = (ldf(&a[0]), ldf(&a[1]), ldf(&a[2]), ldf(&a[3]));
        let (v4, v5, v6, v7) = (ldf(&a[4]), ldf(&a[5]), ldf(&a[6]), ldf(&a[7]));
        _mm256_add_ps(
            _mm256_add_ps(_mm256_add_ps(v0, v4), _mm256_add_ps(v2, v6)),
            _mm256_add_ps(_mm256_add_ps(v1, v5), _mm256_add_ps(v3, v7)),
        )
    }

    macro_rules! kernels {
        ($m:ident, $feat:literal, $acc:ident) => {
            pub mod $m {
                use super::*;

                /// T tokens (xk[..T]) x one group, every super-block (`ub`,
                /// unpacked). ACC[t][0..8] = S lanes, [8..12] = Q4_K min lanes.
                #[allow(clippy::needless_range_loop)]
                #[inline]
                #[target_feature(enable = $feat)]
                fn tile<const T: usize, const Q6: bool>(
                    ub: &[UBlock],
                    xk: &[&[KBlock]],
                    out: &mut [f32],
                    t0: usize,
                    stride: usize,
                ) {
                    let mut x: [&[KBlock]; T] = [&[]; T];
                    for t in 0..T {
                        x[t] = &xk[t][..ub.len()];
                    }
                    let mut acc = [[[0f32; 8]; 12]; T];
                    for (i, u) in ub.iter().enumerate() {
                        let d8 = ldf(&u.d);
                        let mut dd = [_mm256_setzero_ps(); T];
                        for t in 0..T {
                            dd[t] = _mm256_mul_ps(d8, _mm256_set1_ps(x[t][i].d));
                        }
                        if !Q6 {
                            // min lanes: madd((min 2l, min 2l+1), (q8s 2l, q8s 2l+1))
                            let dm8 = ldf(&u.dmin);
                            for t in 0..T {
                                let a = &x[t][i];
                                let bs = ldw(&a.bsums);
                                let q8s = _mm_hadd_epi16(
                                    _mm256_castsi256_si128(bs),
                                    _mm256_extracti128_si256::<1>(bs),
                                );
                                let mut q = [0u8; 16];
                                // SAFETY: q is 16 writable bytes.
                                unsafe { _mm_storeu_si128(q.as_mut_ptr().cast(), q8s) };
                                let dmn = _mm256_mul_ps(dm8, _mm256_set1_ps(-a.d));
                                for l in 0..4 {
                                    let p = _mm256_madd_epi16(ldw(&u.m[l]), bc4(&q, 4 * l));
                                    let k = &mut acc[t][8 + l];
                                    stf(k, _mm256_fmadd_ps(dmn, _mm256_cvtepi32_ps(p), ldf(k)));
                                }
                            }
                        }
                        for g in 0..8 {
                            let mut s = [_mm256_setzero_si256(); T];
                            if Q6 {
                                // chain start: -32 (bs 2g sc 2g + bs 2g+1 sc 2g+1)
                                let m = ldw(&u.m[g]);
                                for t in 0..T {
                                    s[t] = _mm256_madd_epi16(m, bc4(bsums_bytes(&x[t][i]), 4 * g));
                                }
                            }
                            let sc = &u.sc[g / 4];
                            for c in 0..8 {
                                let w = ldv(&u.u[8 * c + g]);
                                let scv = ldw(&sc[c]);
                                for t in 0..T {
                                    let a = bc4(qs_bytes(&x[t][i]), 32 * c + 4 * g);
                                    s[t] = $acc(s[t], scv, _mm256_maddubs_epi16(w, a));
                                }
                            }
                            for t in 0..T {
                                let k = &mut acc[t][g];
                                stf(k, _mm256_fmadd_ps(dd[t], _mm256_cvtepi32_ps(s[t]), ldf(k)));
                            }
                        }
                    }
                    for t in 0..T {
                        let mut f = tree8(&acc[t][..8]);
                        if !Q6 {
                            let a = &acc[t];
                            let m = _mm256_add_ps(
                                _mm256_add_ps(ldf(&a[8]), ldf(&a[10])),
                                _mm256_add_ps(ldf(&a[9]), ldf(&a[11])),
                            );
                            f = _mm256_add_ps(f, m);
                        }
                        let mut r = [0f32; 8];
                        stf(&mut r, f);
                        for (j, &v) in r.iter().enumerate() {
                            out[j * stride + t0 + t] = v;
                        }
                    }
                }

                /// Every token of `xk` x the group `grp`, tokens 4 at a time.
                #[target_feature(enable = $feat)]
                fn tokens<const Q6: bool>(
                    ub: &[UBlock],
                    xk: &[&[KBlock]],
                    out: &mut [f32],
                    stride: usize,
                ) {
                    let nt = xk.len();
                    let mut t = 0;
                    while t + 8 <= nt {
                        tile::<8, Q6>(ub, &xk[t..], out, t, stride);
                        t += 8;
                    }
                    if t + 4 <= nt {
                        tile::<4, Q6>(ub, &xk[t..], out, t, stride);
                        t += 4;
                    }
                    match nt - t {
                        1 => tile::<1, Q6>(ub, &xk[t..], out, t, stride),
                        2 => tile::<2, Q6>(ub, &xk[t..], out, t, stride),
                        3 => tile::<3, Q6>(ub, &xk[t..], out, t, stride),
                        _ => {}
                    }
                }

                /// One Q4_K group (`grp`, r8 layout) x every token: out[r *
                /// stride + t]. `ub` is scratch, one per super-block. (No
                /// unpack prefetch: measured ~4% slower at 18 threads.)
                ///
                /// # Safety
                /// The CPU must support the enabled target features.
                #[target_feature(enable = $feat)]
                pub unsafe fn gemm_q4_k(
                    grp: &[u8],
                    ub: &mut [UBlock],
                    xk: &[&[KBlock]],
                    out: &mut [f32],
                    stride: usize,
                ) {
                    let blocks = &grp.as_chunks::<R8_Q4_K>().0[..ub.len()];
                    for (b, u) in blocks.iter().zip(ub.iter_mut()) {
                        unpack_q4_k(b, u);
                    }
                    tokens::<false>(&ub[..blocks.len()], xk, out, stride);
                }

                /// `gemm_q4_k` for Q6_K.
                ///
                /// # Safety
                /// The CPU must support the enabled target features.
                #[target_feature(enable = $feat)]
                pub unsafe fn gemm_q6_k(
                    grp: &[u8],
                    ub: &mut [UBlock],
                    xk: &[&[KBlock]],
                    out: &mut [f32],
                    stride: usize,
                ) {
                    let blocks = &grp.as_chunks::<R8_Q6_K>().0[..ub.len()];
                    for (b, u) in blocks.iter().zip(ub.iter_mut()) {
                        unpack_q6_k(b, u);
                    }
                    tokens::<true>(&ub[..blocks.len()], xk, out, stride);
                }

                /// One token x whole Q4_K groups, straight from the r8
                /// layout (no unpack): out[8j + r] = row r of group j, for
                /// out.len() / 8 groups. The same i32 lanes and f32 ops as
                /// `tile::<1, false>`. `groups` may run past them: prefetch
                /// (`pf` bytes ahead, 0 = off) stays inside it.
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
                    let nsb = gb / R8_Q4_K;
                    let xk = &xk[..nsb];
                    for (j, o) in out.chunks_exact_mut(8).enumerate() {
                        let grp = groups[j * gb..(j + 1) * gb].as_chunks::<R8_Q4_K>().0;
                        let mut acc = [[0f32; 8]; 12];
                        for (i, (b, a)) in grp.iter().zip(xk).enumerate() {
                            // prefetch: the block `pf` bytes ahead, a quarter
                            // per chunk pair (spread out, not one burst)
                            let ahead = j * gb + i * R8_Q4_K + pf;
                            // min lanes
                            let bs = ldw(&a.bsums);
                            let q8s = _mm_hadd_epi16(
                                _mm256_castsi256_si128(bs),
                                _mm256_extracti128_si256::<1>(bs),
                            );
                            let mut q = [0u8; 16];
                            // SAFETY: q is 16 writable bytes.
                            unsafe { _mm_storeu_si128(q.as_mut_ptr().cast(), q8s) };
                            let dmn =
                                _mm256_mul_ps(_mm256_cvtph_ps(ld16(b, 16)), _mm256_set1_ps(-a.d));
                            for l in 0..4 {
                                let mn = _mm256_cvtepu8_epi16(ld16(b, Q4_MN + 16 * l));
                                let p = _mm256_madd_epi16(mn, bc4(&q, 4 * l));
                                let k = &mut acc[8 + l];
                                stf(k, _mm256_fmadd_ps(dmn, _mm256_cvtepi32_ps(p), ldf(k)));
                            }
                            let dd =
                                _mm256_mul_ps(_mm256_cvtph_ps(ld16(b, 0)), _mm256_set1_ps(a.d));
                            let qs = qs_bytes(a);
                            // The block in address order (c outer, all 8
                            // lanes g inner): one sequential stream, which
                            // the hardware prefetchers follow; a lane-outer
                            // walk hops 256 B back and forth and they don't.
                            let mut s = [_mm256_setzero_si256(); 8];
                            for c in 0..4 {
                                let p = _mm256_cvtepu8_epi16(ld16(b, Q4_SC + 16 * c));
                                let (slo, shi) = (dup_lo(p), dup_hi(p));
                                if pf != 0 {
                                    prefetch(groups, ahead + c * (R8_Q4_K / 4), R8_Q4_K / 4);
                                }
                                for g in 0..8 {
                                    let w = ld32(b, Q4_QS + 32 * (8 * c + g));
                                    let lo = _mm256_and_si256(w, m4);
                                    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(w), m4);
                                    let alo = bc4(qs, 64 * c + 4 * g);
                                    let ahi = bc4(qs, 64 * c + 32 + 4 * g);
                                    s[g] = $acc(s[g], slo, _mm256_maddubs_epi16(lo, alo));
                                    s[g] = $acc(s[g], shi, _mm256_maddubs_epi16(hi, ahi));
                                }
                            }
                            for g in 0..8 {
                                let k = &mut acc[g];
                                stf(k, _mm256_fmadd_ps(dd, _mm256_cvtepi32_ps(s[g]), ldf(k)));
                            }
                        }
                        finish::<false>(&acc, o);
                    }
                }

                /// `gemv_q4_k` for Q6_K.
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
                    let nsb = gb / R8_Q6_K;
                    let xk = &xk[..nsb];
                    for (j, o) in out.chunks_exact_mut(8).enumerate() {
                        let grp = groups[j * gb..(j + 1) * gb].as_chunks::<R8_Q6_K>().0;
                        let mut acc = [[0f32; 8]; 12];
                        for (i, (b, a)) in grp.iter().zip(xk).enumerate() {
                            // prefetch: the block `pf` bytes ahead, a 16th
                            // per (h, g) step (spread out, not one burst)
                            let ahead = j * gb + i * R8_Q6_K + pf;
                            let dd =
                                _mm256_mul_ps(_mm256_cvtph_ps(ld16(b, 0)), _mm256_set1_ps(a.d));
                            let qs = qs_bytes(a);
                            let bsb = bsums_bytes(a);
                            // scales [half][c] (lanes g / 4 == half use
                            // 16-group 2c + half) and the chain starts
                            // -32 (bs 2g sc 2g + bs 2g+1 sc 2g+1)
                            let mut sc = [[_mm256_setzero_si256(); 8]; 2];
                            let mut s = [_mm256_setzero_si256(); 8];
                            for c in 0..8 {
                                let p = _mm256_cvtepi8_epi16(ld16(b, Q6_SC + 16 * c));
                                sc[0][c] = dup_lo(p);
                                sc[1][c] = dup_hi(p);
                                let m32 = _mm256_sub_epi16(
                                    _mm256_setzero_si256(),
                                    _mm256_slli_epi16::<5>(p),
                                );
                                s[c] = _mm256_madd_epi16(m32, bc4(bsb, 4 * c));
                            }
                            // the block in address order ((h, g) groups of
                            // 96 B): one sequential stream (see gemv_q4_k)
                            for h in 0..2 {
                                for g in 0..8 {
                                    if pf != 0 {
                                        let part = R8_Q6_K / 16;
                                        prefetch(groups, ahead + (8 * h + g) * part, part);
                                    }
                                    let w = q6_weights(b, h, g);
                                    for q in 0..4 {
                                        let c = 4 * h + q;
                                        let av = bc4(qs, 32 * c + 4 * g);
                                        let p = _mm256_maddubs_epi16(w[q], av);
                                        s[g] = $acc(s[g], sc[g / 4][c], p);
                                    }
                                }
                            }
                            for g in 0..8 {
                                let k = &mut acc[g];
                                stf(k, _mm256_fmadd_ps(dd, _mm256_cvtepi32_ps(s[g]), ldf(k)));
                            }
                        }
                        finish::<true>(&acc, o);
                    }
                }
            }
        };
    }

    kernels!(avx2, "avx2,fma,f16c", scale_acc_avx2);
    kernels!(vnni, "avx2,fma,f16c,avxvnni", scale_acc_vnni);
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::super::{
        dot_row_q8_with, dot_rows_q8_multi_with, dot_rows_q8_with, has_avx2, has_vnni, row_bytes,
    };
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
        fn f(&mut self) -> f32 {
            self.next_u32() as f32 / (1u64 << 31) as f32 - 1.0
        }
        /// a finite f16 with |v| in about [2^-14, 2^1)
        fn f16(&mut self) -> [u8; 2] {
            let e = 1 + self.next_u32() % 16;
            let m = self.next_u32() & 0x3ff;
            let s = self.next_u32() & 1;
            ((s << 15 | e << 10 | m) as u16).to_le_bytes()
        }
    }

    /// `rows` random rows of width n (any bytes are valid K-quant data, f16s
    /// kept finite).
    fn random_rows(qt: u32, rows: usize, n: usize, rng: &mut Lcg) -> Vec<u8> {
        let rb = row_bytes(qt, n);
        let mut v: Vec<u8> = (0..rows * rb).map(|_| rng.next_u32() as u8).collect();
        let (bb, dpos): (usize, &[usize]) = if qt == Q4_K {
            (Q4_K_BYTES, &[0, 2])
        } else {
            (Q6_K_BYTES, &[208])
        };
        for b in v.chunks_exact_mut(bb) {
            for &p in dpos {
                b[p..p + 2].copy_from_slice(&rng.f16());
            }
        }
        v
    }

    fn random_x(n: usize, rng: &mut Lcg) -> Vec<f32> {
        // mostly small, a few outliers (like real activations)
        (0..n)
            .map(|i| {
                let v = rng.f();
                if i % 97 == 3 { v * 40.0 } else { v }
            })
            .collect()
    }

    fn acts(qt: u32, n: usize, nt: usize, rng: &mut Lcg) -> Vec<QuantizedAct> {
        (0..nt)
            .map(|_| QuantizedAct::for_types(&random_x(n, rng), &[qt]))
            .collect()
    }

    fn paths() -> Vec<Simd> {
        let mut p = Vec::new();
        if has_avx2() {
            p.push(Simd::Avx2);
        }
        if has_vnni() {
            p.push(Simd::Vnni);
        }
        assert!(!p.is_empty(), "r8bi tests need AVX2");
        p
    }

    /// r8bi output for rows first..first + count of `m`, out[r * stride + t].
    fn run(simd: Simd, m: &Mat, first: usize, count: usize, xs: &[&QuantizedAct]) -> Vec<f32> {
        let stride = xs.len() + 3; // a stride wider than the batch
        let mut out = vec![f32::NAN; count * stride];
        unsafe { gemm_with(simd, m, first, xs, &mut out, stride) };
        let mut r = Vec::with_capacity(count * xs.len());
        for row in out.chunks(stride) {
            r.extend_from_slice(&row[..xs.len()]);
            assert!(
                row[xs.len()..].iter().all(|v| v.is_nan()),
                "wrote past the batch"
            );
        }
        r
    }

    /// Every (row, token) of r8bi vs the per-row kernel on each SIMD path
    /// (single-token `dot_row_q8_with`), and vs the row-pair path; returns
    /// the number of mismatching bits patterns (0 = bit-identical).
    #[allow(clippy::too_many_arguments)]
    fn mismatches(
        qt: u32,
        data: &[u8],
        rows: usize,
        n: usize,
        first: usize,
        count: usize,
        xs: &[&QuantizedAct],
        m: &Mat,
    ) -> usize {
        let rb = row_bytes(qt, n);
        let nt = xs.len();
        let mut bad = 0;
        let simd = *paths().last().unwrap();
        let mut reference = vec![0f32; count * nt];
        for r in 0..count {
            let row = &data[(first + r) * rb..(first + r + 1) * rb];
            for (t, x) in xs.iter().enumerate() {
                reference[r * nt + t] = unsafe { dot_row_q8_with(simd, qt, row, x) };
            }
        }
        // the batched row-pair kernels agree with the single-row one
        for r in (0..count).step_by(2) {
            let rows2 = [
                &data[(first + r) * rb..(first + r + 1) * rb],
                &data[(first + r + 1) * rb..(first + r + 2) * rb],
            ];
            let (mut o0, mut o1) = (vec![0f32; nt], vec![0f32; nt]);
            unsafe { dot_rows_q8_multi_with(simd, [qt, qt], rows2, xs, [&mut o0, &mut o1]) };
            for t in 0..nt {
                assert_eq!(o0[t].to_bits(), reference[r * nt + t].to_bits());
                assert_eq!(o1[t].to_bits(), reference[(r + 1) * nt + t].to_bits());
            }
        }
        assert!(first + count <= rows);
        for p in paths() {
            let got = run(p, m, first, count, xs);
            bad += got
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
        // decode: today's multi-row decode kernel agrees with the reference,
        // and so must the r8bi GEMV (prefetch on and off)
        let mut y = vec![0f32; count];
        for (t, x) in xs.iter().enumerate() {
            let rows_from = &data[first * rb..];
            unsafe { dot_rows_q8_with(simd, qt, rows_from, x, &mut y, 8192, 4) };
            for r in 0..count {
                assert_eq!(
                    y[r].to_bits(),
                    reference[r * nt + t].to_bits(),
                    "decode kernel"
                );
            }
            for p in paths() {
                for pf in [0, 8192] {
                    y.fill(f32::NAN);
                    unsafe { gemv_with(p, m, first, x, &mut y, pf, rows) };
                    bad += (0..count)
                        .filter(|&r| y[r].to_bits() != reference[r * nt + t].to_bits())
                        .count();
                }
            }
        }
        bad
    }

    #[test]
    fn r8bi_random_rows_bit_identical() {
        let mut rng = Lcg(0x5eed);
        for qt in [Q4_K, Q6_K] {
            for n in [1536, 4864, 8960] {
                let rows = 24;
                let data = random_rows(qt, rows, n, &mut rng);
                let m = Mat::new(qt, &data, rows, n).unwrap();
                let xq = acts(qt, n, 67, &mut rng);
                for nt in [1, 2, 3, 4, 5, 6, 7, 8, 9, 64, 67] {
                    let xs: Vec<&QuantizedAct> = xq[..nt].iter().collect();
                    // all rows, and a sub-range of groups
                    let bad = mismatches(qt, &data, rows, n, 0, rows, &xs, &m)
                        + mismatches(qt, &data, rows, n, 8, 16, &xs, &m);
                    assert_eq!(bad, 0, "qt {qt} n {n} nt {nt}: {bad} mismatches");
                }
            }
        }
    }

    /// Saturation edge: every weight at its max, every activation at +-127.
    #[test]
    fn r8bi_extreme_bit_identical() {
        let mut rng = Lcg(77);
        let n = 1536;
        for qt in [Q4_K, Q6_K] {
            let rows = 16;
            let mut data = random_rows(qt, rows, n, &mut rng);
            let bb = if qt == Q4_K { Q4_K_BYTES } else { Q6_K_BYTES };
            for (i, b) in data.chunks_exact_mut(bb).enumerate() {
                if qt == Q4_K {
                    b[4..].fill(0xff); // scales/mins 63, nibbles 15
                } else {
                    b[..192].fill(0xff); // q = 63
                    let s = if i % 2 == 0 { 0x80 } else { 0x7f }; // -128 / 127
                    b[192..208].fill(s);
                }
            }
            let m = Mat::new(qt, &data, rows, n).unwrap();
            let xq: Vec<QuantizedAct> = (0..9)
                .map(|t| {
                    let x: Vec<f32> = (0..n)
                        .map(|i| if (i + t) % 3 == 0 { -1.0 } else { 1.0 })
                        .collect();
                    QuantizedAct::for_types(&x, &[qt])
                })
                .collect();
            let xs: Vec<&QuantizedAct> = xq.iter().collect();
            assert_eq!(mismatches(qt, &data, rows, n, 0, rows, &xs, &m), 0);
        }
    }

    #[test]
    fn r8bi_rejects_unrepackable() {
        assert!(!can_repack(Q4_K, 896, 896)); // 896 % 256 != 0
        assert!(!can_repack(Q6_K, 12, 1536)); // rows % 8 != 0
        assert!(!can_repack(super::super::Q8_0, 8, 1536));
        assert!(can_repack(Q4_K, 896, 4864));
    }

    /// Real rows of the models under models/ (both ffn_down types of the
    /// 0.5B, all 896 rows: the 896 of "896/1536/4864/8960"; the 1.5B's
    /// Q4_K/Q6_K matrices at widths 1536 and 8960).
    #[test]
    fn r8bi_real_rows_bit_identical() {
        let models = [
            (
                "models/qwen2.5-0.5b-q4_k_m.gguf",
                &["blk.0.ffn_down.weight", "blk.2.ffn_down.weight"][..],
            ),
            (
                "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                &[
                    "blk.0.attn_q.weight",
                    "blk.0.attn_v.weight",
                    "blk.1.attn_v.weight",
                    "blk.0.ffn_gate.weight",
                    "blk.0.ffn_down.weight",
                    "blk.2.ffn_down.weight",
                ][..],
            ),
        ];
        let mut rng = Lcg(2024);
        let mut seen = Vec::new();
        for (path, names) in models {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
            if !p.exists() {
                assert!(
                    std::env::var_os("AXON_SKIP_MODEL_TESTS").is_some(),
                    "{path} missing"
                );
                continue;
            }
            let model = crate::gguf::load(p.to_str().unwrap()).unwrap();
            for name in names {
                let t = model.tensors.iter().find(|t| t.name == *name).unwrap();
                let (qt, n, rows) = (t.qtype.id, t.dims[0] as usize, t.dims[1] as usize);
                let data = model.tensor_data(t);
                let m = Mat::new(qt, data, rows, n).unwrap();
                // whole matrix for 9 tokens, a 64-row slice for 67
                let xq = acts(qt, n, 67, &mut rng);
                let x9: Vec<&QuantizedAct> = xq[..9].iter().collect();
                let x67: Vec<&QuantizedAct> = xq.iter().collect();
                let count = rows.min(1536);
                let bad = mismatches(qt, data, rows, n, 0, count, &x9, &m)
                    + mismatches(qt, data, rows, n, rows - 64, 64, &x67, &m);
                assert_eq!(bad, 0, "{path} {name}: {bad} mismatches");
                seen.push((qt, n, rows));
            }
        }
        eprintln!("r8bi real matrices checked (qtype, width, rows): {seen:?}");
    }

    /// The bit-identity check must catch near-miss kernels: a swapped scale
    /// and a lane-mapping error that leaves the exact dot product unchanged.
    #[test]
    fn r8bi_mutations_are_caught() {
        let mut rng = Lcg(99);
        let n = 4864;
        let rows = 16;
        for qt in [Q4_K, Q6_K] {
            let data = random_rows(qt, rows, n, &mut rng);
            let xq = acts(qt, n, 5, &mut rng);
            let xs: Vec<&QuantizedAct> = xq.iter().collect();
            let good = Mat::new(qt, &data, rows, n).unwrap();
            assert_eq!(mismatches(qt, &data, rows, n, 0, rows, &xs, &good), 0);
            let bb = block_bytes(qt).unwrap();

            // 1. row 3's scales of sub-blocks 0 and 1 swapped (first super-block)
            let mut bytes = good.data().to_vec();
            let sc = if qt == Q4_K { Q4_SC } else { Q6_SC } + 2 * 3;
            bytes.swap(sc, sc + 1);
            let differ = bytes[sc] != bytes[sc + 1];
            let m = Mat::from_repacked(qt, rows, n, &bytes);
            let bad = mismatches(qt, &data, rows, n, 0, rows, &xs, &m);
            assert!(!differ || bad > 0, "scale swap not caught");
            assert!(differ, "pick a seed where the two scales differ");

            // 2. lanes 0 and 1 swapped in weights AND activations: the exact
            // dot product is unchanged (bsums too: both lanes are in one
            // 16-group, the same Q6_K scale half), only the lane each
            // product sum lands in moves.
            let mut bytes = good.data().to_vec();
            for blk in bytes.chunks_exact_mut(bb) {
                if qt == Q4_K {
                    for c in 0..4 {
                        let o = Q4_QS + 32 * 8 * c;
                        let (a, b) = blk[o..o + 64].split_at_mut(32);
                        a.swap_with_slice(b);
                    }
                } else {
                    for h in 0..2 {
                        let o = Q6_Q + 96 * 8 * h;
                        let (a, b) = blk[o..o + 192].split_at_mut(96);
                        a.swap_with_slice(b);
                    }
                }
            }
            let m = Mat::from_repacked(qt, rows, n, &bytes);
            let swapped: Vec<QuantizedAct> = xq
                .iter()
                .map(|x| {
                    let k =
                        x.k.as_ref()
                            .unwrap()
                            .iter()
                            .map(|b| {
                                let mut qs = b.qs;
                                for c in 0..8 {
                                    for i in 0..4 {
                                        qs.swap(32 * c + i, 32 * c + 4 + i);
                                    }
                                }
                                KBlock {
                                    d: b.d,
                                    qs,
                                    bsums: b.bsums,
                                }
                            })
                            .collect();
                    QuantizedAct {
                        n: x.n,
                        x: Vec::new(),
                        b32: None,
                        k: Some(k),
                    }
                })
                .collect();
            let sx: Vec<&QuantizedAct> = swapped.iter().collect();
            let rb = row_bytes(qt, n);
            for p in paths() {
                let got = run(p, &m, 0, rows, &sx);
                let mut bad = 0;
                for r in 0..rows {
                    for t in 0..xs.len() {
                        let want =
                            unsafe { dot_row_q8_with(p, qt, &data[r * rb..(r + 1) * rb], xs[t]) };
                        let g = got[r * xs.len() + t];
                        assert!(
                            (g - want).abs() <= 1e-4 * want.abs().max(1.0),
                            "{g} vs {want}"
                        );
                        bad += (g.to_bits() != want.to_bits()) as usize;
                    }
                }
                let mut gv = vec![0f32; rows];
                let mut bad_gemv = 0;
                for t in 0..xs.len() {
                    unsafe { gemv_with(p, &m, 0, sx[t], &mut gv, 0, rows) };
                    for r in 0..rows {
                        let want =
                            unsafe { dot_row_q8_with(p, qt, &data[r * rb..(r + 1) * rb], xs[t]) };
                        bad_gemv += (gv[r].to_bits() != want.to_bits()) as usize;
                    }
                }
                assert!(bad > 0 && bad_gemv > 0, "lane swap not caught");
                eprintln!(
                    "qt {qt} {p:?}: lane swap changes {bad}/{} results",
                    rows * xs.len()
                );
            }
        }
    }
}
