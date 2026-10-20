// Dequantization kernels for GGUF tensor types.
//
// Reference: ggml (https://github.com/ggml-org/llama.cpp), MIT License,
// Copyright (c) 2023-2024 The ggml authors. Block layouts follow
// ggml/src/ggml-common.h; dequant math follows dequantize_row_* and
// get_scale_min_k4 in ggml/src/ggml-quants.c. All reads are little-endian
// from byte slices (no transmutes).

const QK4_0: usize = 32;
const QK5_0: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

// block_q4_0 { ggml_half d; uint8_t qs[16]; }
const Q4_0_BYTES: usize = 2 + QK4_0 / 2; // 18
// block_q4_1 { ggml_half d; ggml_half m; uint8_t qs[16]; }
const Q4_1_BYTES: usize = 2 + 2 + QK4_0 / 2; // 20
// block_q5_1 { ggml_half d; ggml_half m; uint8_t qh[4]; uint8_t qs[16]; }
const Q5_1_BYTES: usize = 2 + 2 + 4 + QK5_0 / 2; // 24
// block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; }
const Q5_0_BYTES: usize = 2 + 4 + QK5_0 / 2; // 22
// block_q8_0 { ggml_half d; int8_t qs[32]; }
const Q8_0_BYTES: usize = 2 + QK8_0; // 34
// block_q4_K { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[128]; }
const Q4_K_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2; // 144
// block_q5_K { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }
const Q5_K_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2; // 176
// block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; ggml_half d; }
const Q6_K_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 210

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q4_1: u32 = 3;
pub const Q5_0: u32 = 6;
pub const Q5_1: u32 = 7;
pub const Q8_0: u32 = 8;
pub const Q2_K: u32 = 10;
pub const Q3_K: u32 = 11;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;
pub const IQ4_NL: u32 = 20;
pub const MXFP4: u32 = 39;

mod iq4nl;
mod k23;
mod mxfp4;
pub mod r8;
pub mod r8q;

use iq4nl::IQ4_NL_BYTES;
use k23::{Q2_K_BYTES, Q3_K_BYTES};
use mxfp4::MXFP4_BYTES;

/// MXFP4's code table (tests build fixtures with it).
#[cfg(test)]
pub(crate) fn mxfp4_kvalues() -> [i8; 16] {
    mxfp4::KVALUES
}

/// (values per block, bytes per block)
fn block_info(qtype_id: u32) -> (usize, usize) {
    match qtype_id {
        F32 => (1, 4),
        F16 => (1, 2),
        Q4_0 => (QK4_0, Q4_0_BYTES),
        Q4_1 => (QK4_0, Q4_1_BYTES),
        Q5_1 => (QK5_0, Q5_1_BYTES),
        Q5_0 => (QK5_0, Q5_0_BYTES),
        Q8_0 => (QK8_0, Q8_0_BYTES),
        IQ4_NL => (QK8_0, IQ4_NL_BYTES),
        Q2_K => (QK_K, Q2_K_BYTES),
        Q3_K => (QK_K, Q3_K_BYTES),
        Q4_K => (QK_K, Q4_K_BYTES),
        Q5_K => (QK_K, Q5_K_BYTES),
        Q6_K => (QK_K, Q6_K_BYTES),
        MXFP4 => (QK8_0, MXFP4_BYTES),
        _ => panic!(
            "quant: unsupported qtype {} ({})",
            qtype_id,
            crate::gguf::qtype(qtype_id).name
        ),
    }
}

#[inline]
fn rd_f16(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        // zero or subnormal: mant * 2^-24 (exact in f32)
        let v = mant as f32 * (1.0 / 16_777_216.0);
        return if sign != 0 { -v } else { v };
    }
    let bits = if exp == 31 {
        // inf (mant == 0) or NaN (payload preserved)
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn row_bytes(qtype_id: u32, n: usize) -> usize {
    let (bs, bb) = block_info(qtype_id);
    assert!(
        n.is_multiple_of(bs),
        "row_bytes: n={} not a multiple of block size {}",
        n,
        bs
    );
    n / bs * bb
}

fn dequant_q5_0_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..6 + QK5_0 / 2];
    for j in 0..QK5_0 / 2 {
        let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
        let x0 = ((qs[j] & 0x0f) | xh_0) as i32 - 16;
        let x1 = ((qs[j] >> 4) | xh_1) as i32 - 16;
        y[j] = x0 as f32 * d;
        y[j + QK5_0 / 2] = x1 as f32 * d;
    }
}

fn dequant_q4_0_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..2 + QK4_0 / 2];
    for j in 0..QK4_0 / 2 {
        let x0 = (qs[j] & 0x0f) as i32 - 8;
        let x1 = (qs[j] >> 4) as i32 - 8;
        y[j] = x0 as f32 * d;
        y[j + QK4_0 / 2] = x1 as f32 * d;
    }
}

fn dequant_q4_1_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let m = rd_f16(b, 2);
    let qs = &b[4..4 + QK4_0 / 2];
    for j in 0..QK4_0 / 2 {
        y[j] = (qs[j] & 0x0f) as f32 * d + m;
        y[j + QK4_0 / 2] = (qs[j] >> 4) as f32 * d + m;
    }
}

fn dequant_q5_1_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let m = rd_f16(b, 2);
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..8 + QK5_0 / 2];
    for j in 0..QK5_0 / 2 {
        let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
        y[j] = ((qs[j] & 0x0f) | xh_0) as f32 * d + m;
        y[j + QK5_0 / 2] = ((qs[j] >> 4) | xh_1) as f32 * d + m;
    }
}

fn dequant_q8_0_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    for j in 0..QK8_0 {
        y[j] = (b[2 + j] as i8) as f32 * d;
    }
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let min = rd_f16(b, 2);
    let scales = &b[4..4 + K_SCALE_SIZE];
    let qs = &b[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 2];
    let mut is = 0;
    let mut qo = 0; // offset into qs
    let mut yo = 0; // offset into y
    for _ in (0..QK_K).step_by(64) {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        let q = &qs[qo..qo + 32];
        for l in 0..32 {
            y[yo + l] = d1 * (q[l] & 0xf) as f32 - m1;
        }
        for l in 0..32 {
            y[yo + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
        }
        yo += 64;
        qo += 32;
        is += 2;
    }
}

fn dequant_q5_k_block(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let min = rd_f16(b, 2);
    let scales = &b[4..4 + K_SCALE_SIZE];
    let qh = &b[16..48];
    let qs = &b[48..Q5_K_BYTES];
    for c in 0..QK_K / 64 {
        let (sc, m) = get_scale_min_k4(2 * c, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(2 * c + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        let q = &qs[32 * c..32 * c + 32];
        for l in 0..32 {
            let h = (qh[l] >> (2 * c)) & 1;
            y[64 * c + l] = d1 * ((q[l] & 0xf) | (h << 4)) as f32 - m1;
        }
        for l in 0..32 {
            let h = (qh[l] >> (2 * c + 1)) & 1;
            y[64 * c + 32 + l] = d2 * ((q[l] >> 4) | (h << 4)) as f32 - m2;
        }
    }
}

fn dequant_q6_k_block(b: &[u8], y: &mut [f32]) {
    let ql_all = &b[0..QK_K / 2];
    let qh_all = &b[QK_K / 2..QK_K / 2 + QK_K / 4];
    let sc_all = &b[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16];
    let d = rd_f16(b, QK_K / 2 + QK_K / 4 + QK_K / 16);
    for half in 0..QK_K / 128 {
        let ql = &ql_all[half * 64..];
        let qh = &qh_all[half * 32..];
        let sc = &sc_all[half * 8..];
        let yy = &mut y[half * 128..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            yy[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            yy[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            yy[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            yy[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

/// Dequantize `n = out.len()` consecutive values (n % block_size == 0) from `bytes` into `out`.
pub fn dequant(qtype_id: u32, bytes: &[u8], out: &mut [f32]) {
    let n = out.len();
    let (bs, bb) = block_info(qtype_id);
    assert!(
        n.is_multiple_of(bs),
        "dequant: n={} not a multiple of block size {}",
        n,
        bs
    );
    let nb = n / bs;
    assert!(
        bytes.len() >= nb * bb,
        "dequant: need {} bytes, got {}",
        nb * bb,
        bytes.len()
    );
    match qtype_id {
        F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                let p = i * 4;
                *o = f32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]);
            }
        }
        F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = rd_f16(bytes, i * 2);
            }
        }
        _ => {
            let f: fn(&[u8], &mut [f32]) = match qtype_id {
                Q4_0 => dequant_q4_0_block,
                Q4_1 => dequant_q4_1_block,
                Q5_1 => dequant_q5_1_block,
                Q5_0 => dequant_q5_0_block,
                Q8_0 => dequant_q8_0_block,
                IQ4_NL => iq4nl::dequant_iq4_nl_block,
                Q2_K => k23::dequant_q2_k_block,
                Q3_K => k23::dequant_q3_k_block,
                Q4_K => dequant_q4_k_block,
                Q5_K => dequant_q5_k_block,
                Q6_K => dequant_q6_k_block,
                MXFP4 => mxfp4::dequant_mxfp4_block,
                _ => unreachable!(),
            };
            for i in 0..nb {
                f(&bytes[i * bb..(i + 1) * bb], &mut out[i * bs..(i + 1) * bs]);
            }
        }
    }
}

pub fn dot_row(qtype_id: u32, row: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    let (bs, bb) = block_info(qtype_id);
    assert!(
        n.is_multiple_of(bs),
        "dot_row: len {} not a multiple of block size {}",
        n,
        bs
    );
    // decode in chunks of up to 256 values (one K-block, or several small blocks)
    let chunk = QK_K; // multiple of every supported block size
    let mut buf = [0f32; QK_K];
    let mut sum = 0f32;
    let mut i = 0;
    while i < n {
        let len = chunk.min(n - i);
        let boff = i / bs * bb;
        let bend = boff + len / bs * bb;
        dequant(qtype_id, &row[boff..bend], &mut buf[..len]);
        let xs = &x[i..i + len];
        let mut s = 0f32;
        for k in 0..len {
            s += buf[k] * xs[k];
        }
        sum += s;
        i += len;
    }
    sum
}

// ---------------------------------------------------------------------------
// Integer dot products against 8-bit-quantized activations (llama.cpp's
// vec_dot_*_q8_0 / vec_dot_*_q8_K approach). Activations are quantized once
// per matvec; each row then needs only i8*i8 -> i32 sums plus one f32 multiply
// per (sub-)block.

/// One 256-value block of "Q8_K" activations.
struct KBlock {
    d: f32,
    qs: [i8; QK_K],
    /// sums of qs over each 16-element group
    bsums: [i16; QK_K / 16],
}

/// 32-block ("Q8_0") activations, block i = 32 values.
struct B32 {
    /// per-block scale
    d: Vec<f32>,
    /// per-block i8 values
    q: Vec<[i8; 32]>,
    /// per block, lane l (bytes 4l..4l+4 of q, one i32 lane of a 256-bit
    /// vector): -16 * (sum of those 4 values). The VNNI kernels feed unsigned
    /// weights u to dpbusd and start the accumulator here to undo the offset:
    /// Q5_0 u = w + 16 uses it as is, Q8_0 u = w + 128 uses it << 3.
    c: Vec<[i32; 8]>,
    /// per block: -128 * (sum of its 32 values), the start of the r8q Q8_0
    /// VNNI chain (r8q.rs), kept here so a GEMM chunk doesn't redo the sums.
    n128: Vec<i32>,
}

#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
#[inline(always)]
fn neg16_lane_sums(q: &[i8; 32]) -> [i32; 8] {
    let mut c = [0i32; 8];
    for l in 0..8 {
        let s = q[4 * l] as i32 + q[4 * l + 1] as i32 + q[4 * l + 2] as i32 + q[4 * l + 3] as i32;
        c[l] = -16 * s;
    }
    c
}

/// x quantized once per matvec, in the layouts its weight types need.
pub struct QuantizedAct {
    n: usize,
    /// original values, for the F32/F16 float fallback; empty unless built
    x: Vec<f32>,
    /// "Q8_0" view (32-blocks); None if len % 32 != 0 or not built
    b32: Option<B32>,
    /// "Q8_K" view (256-blocks); None if len % 256 != 0 or not built
    k: Option<Vec<KBlock>>,
}

/// Round to nearest, ties to even, for |v| < 2^22: the same result as
/// `f32::round_ties_even`, but plain arithmetic. Adding 1.5 * 2^23 leaves no
/// fraction bits, so the FPU's own ties-to-even rounding does the work (the
/// trick llama.cpp's `nearest_int` uses). Without SSE4.1 at compile time,
/// `round_ties_even` is a libm call per element: 63 us to quantize one
/// 4864-wide activation, half the time of the matvec that uses it.
#[inline(always)]
fn round_even_small(v: f32) -> i32 {
    const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23
    (v + MAGIC).to_bits() as i32 - MAGIC.to_bits() as i32
}

/// Symmetric 8-bit quantization of one block: d = amax/127, q = round(x/d).
#[inline]
/// f32 -> IEEE half bits, round to nearest even (what F16C's `vcvtps2ph` does).
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        // inf, or NaN quieted with its top payload bits kept (as vcvtps2ph)
        return sign
            | 0x7c00
            | if man != 0 {
                0x200 | (man >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    let (r, rem, half) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        (m >> shift, m & ((1 << shift) - 1), 1u32 << (shift - 1))
    } else {
        (((e as u32) << 10) | (man >> 13), man & 0x1fff, 0x1000)
    };
    let r = if rem > half || (rem == half && r & 1 == 1) {
        r + 1
    } else {
        r
    };
    sign | r as u16
}

/// Activation block for Q5_0/Q8_0 weights, exactly as llama.cpp's x86
/// quantize_row_q8_0: d = amax/127 stored as f16, q = round-half-even(x * 127/amax).
/// Largest |v| in x, with 8 lane-wise maxima so it vectorizes (max is exact,
/// so the order doesn't change the result). NaNs are ignored, as f32::max does.
#[inline(always)]
fn abs_max(x: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let (chunks, tail) = x.as_chunks::<8>();
    for c in chunks {
        for i in 0..8 {
            lanes[i] = lanes[i].max(c[i].abs());
        }
    }
    let mut m = 0f32;
    for v in tail {
        m = m.max(v.abs());
    }
    for b in lanes {
        m = m.max(b);
    }
    m
}

#[inline(always)]
fn quantize_q8_0(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = abs_max(x);
    let d = f16_to_f32(f32_to_f16(amax / 127.0));
    let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
    for (qi, &xi) in q.iter_mut().zip(x) {
        *qi = round_even_small(xi * id).clamp(-128, 127) as i8;
    }
    d
}

/// Activation block for K-quant weights, exactly as llama.cpp's
/// quantize_row_q8_K_ref: scale from the signed largest-magnitude value,
/// q = min(127, round-half-even(x * -127/max)), d = 1/iscale.
#[inline(always)]
fn quantize_q8_k(x: &[f32], q: &mut [i8]) -> f32 {
    // The signed value with the largest magnitude, first one on ties (the
    // reference loop keeps the first because it only replaces on `>`). Two
    // passes so both vectorize: the magnitude, then where it first occurs.
    let amax = abs_max(x);
    if amax == 0.0 {
        q.fill(0);
        return 0.0;
    }
    // amax is the |v| of some v in x (NaNs are ignored, and all-NaN gives 0).
    let mut max = f32::NAN;
    for &v in x {
        if v.abs() == amax {
            max = v;
            break;
        }
    }
    debug_assert!(!max.is_nan(), "amax comes from x");
    let iscale = -127.0 / max;
    for (qi, &xi) in q.iter_mut().zip(x) {
        *qi = round_even_small(iscale * xi).min(127) as i8;
    }
    1.0 / iscale
}

multiversion! {
    /// The "Q8_0" view of x (len a multiple of 32).
    fn quantize_b32<V3>(x: &[f32]) -> B32 {
        let nb = x.len() / QK8_0;
        let mut d = Vec::with_capacity(nb);
        let mut q = vec![[0i8; 32]; nb];
        let mut c = Vec::with_capacity(nb);
        let mut n128 = Vec::with_capacity(nb);
        for (xb, qb) in x.as_chunks::<QK8_0>().0.iter().zip(q.iter_mut()) {
            d.push(quantize_q8_0(xb, qb));
            let cb = neg16_lane_sums(qb);
            // Σ c = -16 Σ q, exactly (|Σ q| <= 32 * 127)
            n128.push(8 * cb.iter().sum::<i32>());
            c.push(cb);
        }
        B32 { d, q, c, n128 }
    }
}

multiversion! {
    /// The "Q8_K" view of x (len a multiple of 256).
    fn quantize_k<V3>(x: &[f32]) -> Vec<KBlock> {
        let blocks = x.as_chunks::<QK_K>().0;
        let mut out = Vec::with_capacity(blocks.len());
        for xb in blocks {
            let mut qs = [0i8; QK_K];
            let d = quantize_q8_k(xb, &mut qs);
            let mut bsums = [0i16; QK_K / 16];
            for (s, g) in bsums.iter_mut().zip(qs.as_chunks::<16>().0) {
                let mut t = 0i16;
                for &v in g {
                    t += v as i16;
                }
                *s = t;
            }
            out.push(KBlock { d, qs, bsums });
        }
        out
    }
}

impl QuantizedAct {
    /// Every layout x's length allows.
    #[cfg(test)]
    pub fn new(x: &[f32]) -> QuantizedAct {
        Self::for_types(x, &[F32, Q8_0, Q4_K])
    }

    /// Only the layouts the given weight types read, so a matvec doesn't pay
    /// to quantize views it never uses.
    pub fn for_types(x: &[f32], qtypes: &[u32]) -> QuantizedAct {
        let n = x.len();
        let wants = |ts: &[u32]| qtypes.iter().any(|t| ts.contains(t));
        let b32 = (wants(&[Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, MXFP4])
            && n.is_multiple_of(QK8_0))
        .then(|| quantize_b32(x));
        let k = (wants(&[Q2_K, Q3_K, Q4_K, Q5_K, Q6_K]) && n.is_multiple_of(QK_K))
            .then(|| quantize_k(x));
        let x = if wants(&[F32, F16]) {
            x.to_vec()
        } else {
            Vec::new()
        };
        QuantizedAct { n, x, b32, k }
    }

    pub fn len(&self) -> usize {
        self.n
    }
}

/// The views as built, for the GPU twin's equality test
/// (`cuda::qwen2::tests::twin`): Q8_0 (per-block scales, values) and Q8_K
/// (per-block scales, values, sums of 16).
#[cfg(test)]
pub type Q8Views = (
    Option<(Vec<f32>, Vec<i8>)>,
    Option<(Vec<f32>, Vec<i8>, Vec<i16>)>,
);

#[cfg(test)]
impl QuantizedAct {
    pub fn views(&self) -> Q8Views {
        let b32 = self
            .b32
            .as_ref()
            .map(|b| (b.d.clone(), b.q.iter().flatten().copied().collect()));
        let k = self.k.as_ref().map(|k| {
            (
                k.iter().map(|b| b.d).collect(),
                k.iter().flat_map(|b| b.qs).collect(),
                k.iter().flat_map(|b| b.bsums).collect(),
            )
        });
        (b32, k)
    }
}

#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
#[inline(always)]
fn dot_i8_32(a: &[i8; 32], b: &[i8; 32]) -> i32 {
    let mut s = 0i32;
    for i in 0..32 {
        s += a[i] as i32 * b[i] as i32;
    }
    s
}

#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
fn dot_q8_0_q8(row: &[u8], dx: &[f32], qx: &[[i8; 32]]) -> f32 {
    let mut sum = 0f32;
    for (b, (&dxb, qxb)) in row
        .as_chunks::<Q8_0_BYTES>()
        .0
        .iter()
        .zip(dx.iter().zip(qx))
    {
        let d = rd_f16(b, 0);
        let qw: &[u8; 32] = b[2..2 + QK8_0].try_into().unwrap();
        let mut s = 0i32;
        for i in 0..32 {
            s += (qw[i] as i8) as i32 * qxb[i] as i32;
        }
        sum += d * dxb * s as f32;
    }
    sum
}

#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
fn dot_q5_0_q8(row: &[u8], dx: &[f32], qx: &[[i8; 32]]) -> f32 {
    let mut sum = 0f32;
    for (b, (&dxb, qxb)) in row
        .as_chunks::<Q5_0_BYTES>()
        .0
        .iter()
        .zip(dx.iter().zip(qx))
    {
        let d = rd_f16(b, 0);
        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
        let qs: &[u8; 16] = b[6..6 + QK5_0 / 2].try_into().unwrap();
        // same bit order as dequant_q5_0_block: element j gets qh bit j
        let mut w = [0i8; 32];
        for j in 0..16 {
            let h0 = (((qh >> j) & 1) << 4) as u8;
            let h1 = (((qh >> (j + 16)) & 1) << 4) as u8;
            w[j] = ((qs[j] & 0x0f) | h0) as i8 - 16;
            w[j + 16] = ((qs[j] >> 4) | h1) as i8 - 16;
        }
        sum += d * dxb * dot_i8_32(&w, qxb) as f32;
    }
    sum
}

#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
fn dot_q4_0_q8(row: &[u8], dx: &[f32], qx: &[[i8; 32]]) -> f32 {
    let mut sum = 0f32;
    for (b, (&dxb, qxb)) in row
        .as_chunks::<Q4_0_BYTES>()
        .0
        .iter()
        .zip(dx.iter().zip(qx))
    {
        let d = rd_f16(b, 0);
        let qs: &[u8; 16] = b[2..2 + QK4_0 / 2].try_into().unwrap();
        // same order as dequant_q4_0_block: low nibbles, then high
        let mut w = [0i8; 32];
        for j in 0..16 {
            w[j] = (qs[j] & 0x0f) as i8 - 8;
            w[j + 16] = (qs[j] >> 4) as i8 - 8;
        }
        sum += d * dxb * dot_i8_32(&w, qxb) as f32;
    }
    sum
}

/// The unsigned 32 weights of a Q4_1 / Q5_1 block (dequant's order) and
/// its (d, m).
#[inline(always)]
fn q_1_block(qtype_id: u32, b: &[u8]) -> ([u8; 32], f32, f32) {
    let mut w = [0u8; 32];
    let (qh, qs) = if qtype_id == Q5_1 {
        (u32::from_le_bytes([b[4], b[5], b[6], b[7]]), &b[8..24])
    } else {
        (0, &b[4..20])
    };
    for j in 0..16 {
        w[j] = (qs[j] & 0x0f) | ((((qh >> j) & 1) << 4) as u8);
        w[j + 16] = (qs[j] >> 4) | ((((qh >> (j + 16)) & 1) << 4) as u8);
    }
    (w, rd_f16(b, 0), rd_f16(b, 2))
}

/// Q4_1 / Q5_1 against Q8_0 activations, per block (the GPU's canonical
/// order too): dx * (d * Σ w qx + m * Σ qx), the sums exact integers. The
/// activation sum comes from the view's lane sums (c = -16 Σ, exact).
/// Every CPU path runs this scalar code (these types have no SIMD kernels
/// beyond the integer sum), so all of them give the same bits.
#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
fn dot_q_1_q8(qtype_id: u32, row: &[u8], x: &B32) -> f32 {
    let bb = if qtype_id == Q5_1 {
        Q5_1_BYTES
    } else {
        Q4_1_BYTES
    };
    let mut sum = 0f32;
    for (i, b) in row.chunks_exact(bb).enumerate() {
        let (w, d, m) = q_1_block(qtype_id, b);
        let qx = &x.q[i];
        let mut s = 0i32;
        for k in 0..32 {
            s += w[k] as i32 * qx[k] as i32;
        }
        let bs = x.c[i].iter().sum::<i32>() / -16;
        sum += x.d[i] * (d * s as f32 + m * bs as f32);
    }
    sum
}

#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
fn dot_q5_k_q8(row: &[u8], xk: &[KBlock]) -> f32 {
    let mut sum = 0f32;
    for (b, a) in row.as_chunks::<Q5_K_BYTES>().0.iter().zip(xk) {
        let d = rd_f16(b, 0);
        let dmin = rd_f16(b, 2);
        let scales: &[u8; K_SCALE_SIZE] = b[4..4 + K_SCALE_SIZE].try_into().unwrap();
        let qh: &[u8; 32] = b[16..48].try_into().unwrap();
        let qs: &[u8; QK_K / 2] = b[48..Q5_K_BYTES].try_into().unwrap();
        let mut sumi = 0i32; // Σ_j sc_j * Σ q*qx
        let mut summ = 0i32; // Σ_j m_j * Σ qx
        for c in 0..4 {
            let (sc1, m1) = get_scale_min_k4(2 * c, scales);
            let (sc2, m2) = get_scale_min_k4(2 * c + 1, scales);
            let q: &[u8; 32] = qs[32 * c..32 * c + 32].try_into().unwrap();
            let x1: &[i8; 32] = a.qs[64 * c..64 * c + 32].try_into().unwrap();
            let x2: &[i8; 32] = a.qs[64 * c + 32..64 * c + 64].try_into().unwrap();
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            for l in 0..32 {
                let w1 = (q[l] & 0x0f) | (((qh[l] >> (2 * c)) & 1) << 4);
                let w2 = (q[l] >> 4) | (((qh[l] >> (2 * c + 1)) & 1) << 4);
                s1 += w1 as i32 * x1[l] as i32;
                s2 += w2 as i32 * x2[l] as i32;
            }
            sumi += sc1 as i32 * s1 + sc2 as i32 * s2;
            let bs = &a.bsums[4 * c..4 * c + 4];
            summ += m1 as i32 * (bs[0] as i32 + bs[1] as i32)
                + m2 as i32 * (bs[2] as i32 + bs[3] as i32);
        }
        sum += a.d * (d * sumi as f32 - dmin * summ as f32);
    }
    sum
}

#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
fn dot_q4_k_q8(row: &[u8], xk: &[KBlock]) -> f32 {
    let mut sum = 0f32;
    for (b, a) in row.as_chunks::<Q4_K_BYTES>().0.iter().zip(xk) {
        let d = rd_f16(b, 0);
        let dmin = rd_f16(b, 2);
        let scales: &[u8; K_SCALE_SIZE] = b[4..4 + K_SCALE_SIZE].try_into().unwrap();
        let qs: &[u8; QK_K / 2] = b[4 + K_SCALE_SIZE..Q4_K_BYTES].try_into().unwrap();
        let mut sumi = 0i32; // Σ_j sc_j * Σ q*qx
        let mut summ = 0i32; // Σ_j m_j * Σ qx
        for c in 0..4 {
            let (sc1, m1) = get_scale_min_k4(2 * c, scales);
            let (sc2, m2) = get_scale_min_k4(2 * c + 1, scales);
            let q: &[u8; 32] = qs[32 * c..32 * c + 32].try_into().unwrap();
            let x1: &[i8; 32] = a.qs[64 * c..64 * c + 32].try_into().unwrap();
            let x2: &[i8; 32] = a.qs[64 * c + 32..64 * c + 64].try_into().unwrap();
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            for l in 0..32 {
                s1 += (q[l] & 0x0f) as i32 * x1[l] as i32;
                s2 += (q[l] >> 4) as i32 * x2[l] as i32;
            }
            sumi += sc1 as i32 * s1 + sc2 as i32 * s2;
            let bs = &a.bsums[4 * c..4 * c + 4];
            summ += m1 as i32 * (bs[0] as i32 + bs[1] as i32)
                + m2 as i32 * (bs[2] as i32 + bs[3] as i32);
        }
        sum += a.d * (d * sumi as f32 - dmin * summ as f32);
    }
    sum
}

#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
fn dot_q6_k_q8(row: &[u8], xk: &[KBlock]) -> f32 {
    const QL: usize = QK_K / 2;
    const QH: usize = QL + QK_K / 4;
    const SC: usize = QH + QK_K / 16;
    let mut sum = 0f32;
    for (b, a) in row.as_chunks::<Q6_K_BYTES>().0.iter().zip(xk) {
        let ql_all: &[u8; QL] = b[0..QL].try_into().unwrap();
        let qh_all: &[u8; QK_K / 4] = b[QL..QH].try_into().unwrap();
        let sc: &[u8; QK_K / 16] = b[QH..SC].try_into().unwrap();
        let d = rd_f16(b, SC);
        // decode to signed 6-bit values (q6 - 32), same layout as dequant_q6_k_block
        let mut w = [0i8; QK_K];
        for half in 0..2 {
            let ql: &[u8; 64] = ql_all[half * 64..half * 64 + 64].try_into().unwrap();
            let qh: &[u8; 32] = qh_all[half * 32..half * 32 + 32].try_into().unwrap();
            let y: &mut [i8; 128] = (&mut w[half * 128..half * 128 + 128]).try_into().unwrap();
            for l in 0..32 {
                y[l] = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8 - 32;
                y[l + 32] = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                y[l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                y[l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
            }
        }
        let mut sumi = 0i32;
        for (g, (wg, xg)) in w
            .as_chunks::<16>()
            .0
            .iter()
            .zip(a.qs.as_chunks::<16>().0)
            .enumerate()
        {
            let mut s = 0i32;
            for l in 0..16 {
                s += wg[l] as i32 * xg[l] as i32;
            }
            sumi += (sc[g] as i8) as i32 * s;
        }
        sum += a.d * d * sumi as f32;
    }
    sum
}

// ---------------------------------------------------------------------------
// SIMD kernels, chosen once at runtime. They follow llama.cpp's AVX2 code
// (ggml/src/ggml-cpu/arch/x86/quants.c) and produce the same integer sums per
// block as the scalar kernels above; only the f32 accumulation order differs.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Simd {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Vnni,
}

/// The kernel set, picked on first use: the best the CPU supports, or
/// `AXON_SIMD=scalar|avx2|vnni` (panics if the requested path is unavailable).
fn simd() -> Simd {
    static SIMD: std::sync::OnceLock<Simd> = std::sync::OnceLock::new();
    *SIMD.get_or_init(|| choose_simd(std::env::var("AXON_SIMD").ok().as_deref()))
}

/// What the SIMD kernels need: every `x86` kernel and the `_v3` glue entry
/// points are compiled with `#[target_feature(enable = "avx2,fma,f16c")]`
/// (VNNI ones add avxvnni), so the default build runs on any x86-64 CPU and
/// only these functions use AVX. Unchanged from the target-cpu days, so an
/// AVX2 CPU without BMI (a VM with a masked CPU model) keeps its kernels.
#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
}

/// The full x86-64-v3 set the multiversioned non-kernel functions (`mv.rs`)
/// are compiled for: the kernel set plus bmi1, bmi2 and lzcnt.
#[cfg(target_arch = "x86_64")]
fn has_v3() -> bool {
    has_avx2()
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("bmi2")
        && is_x86_feature_detected!("lzcnt")
}

/// Whether the `_v3` variants of the multiversioned functions run (see
/// `multiversion!`): on a SIMD kernel path (so `AXON_SIMD=scalar` runs the
/// portable code everywhere) when the CPU has the full x86-64-v3 set. Both
/// variants give the same bits, so this only changes speed. Decided once; an
/// AVX2 CPU that misses bmi1/bmi2/lzcnt says so on stderr.
#[inline]
pub fn v3() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static V3: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V3.get_or_init(|| {
            let ok = simd() != Simd::Scalar && has_v3();
            if simd() != Simd::Scalar && !ok {
                eprintln!(
                    "axon: CPU has AVX2 but not bmi1/bmi2/lzcnt: SIMD kernels on, \
                     other hot loops portable or scalar FMA (same results, slower)"
                );
            }
            ok
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
fn has_vnni() -> bool {
    has_avx2() && is_x86_feature_detected!("avxvnni")
}

fn simd_unavailable(p: &str, avx2: bool, vnni: bool) -> ! {
    panic!("AXON_SIMD={p}: this CPU/target doesn't support it (avx2={avx2}, vnni={vnni})")
}

fn choose_simd(req: Option<&str>) -> Simd {
    #[cfg(target_arch = "x86_64")]
    let (avx2, vnni) = (has_avx2(), has_vnni());
    #[cfg(not(target_arch = "x86_64"))]
    let (avx2, vnni) = (false, false);
    match req {
        Some("scalar") => Simd::Scalar,
        Some("avx2") => {
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                return Simd::Avx2;
            }
            simd_unavailable("avx2", avx2, vnni)
        }
        Some("vnni") => {
            #[cfg(target_arch = "x86_64")]
            if vnni {
                return Simd::Vnni;
            }
            simd_unavailable("vnni", avx2, vnni)
        }
        None | Some("") => {
            #[cfg(target_arch = "x86_64")]
            {
                if vnni {
                    return Simd::Vnni;
                }
                if avx2 {
                    return Simd::Avx2;
                }
                if is_x86_feature_detected!("avx2") {
                    eprintln!(
                        "axon: CPU has AVX2 but not fma/f16c: using the scalar kernels \
                         (much slower; results within tolerance of the SIMD ones)"
                    );
                }
            }
            Simd::Scalar
        }
        Some(other) => panic!("AXON_SIMD={other}: expected scalar, avx2 or vnni"),
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{
        B32, KBlock, Q4_0_BYTES, Q4_1_BYTES, Q4_K_BYTES, Q5_0_BYTES, Q5_1_BYTES, Q5_K_BYTES,
        Q6_K_BYTES, Q8_0_BYTES,
    };
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn load_u8x32(p: &[u8; 32]) -> __m256i {
        // SAFETY: p points to 32 readable bytes; loadu has no alignment requirement.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn load_i8x32(p: &[i8; 32]) -> __m256i {
        // SAFETY: as above.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn load_i16x16(p: &[i16; 16]) -> __m256i {
        // SAFETY: as above (32 bytes).
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn load_u8x16(p: &[u8; 16]) -> __m128i {
        // SAFETY: as above (16 bytes).
        unsafe { _mm_loadu_si128(p.as_ptr().cast()) }
    }

    /// little-endian f16 -> f32 (exact, same value as `rd_f16`)
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn h2f(b: [u8; 2]) -> f32 {
        let h = u16::from_le_bytes(b) as i32;
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn hsum_f32x8(x: __m256) -> f32 {
        let mut r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        _mm_cvtss_f32(r)
    }

    /// Byte i = 0xFF if bit i of x is set, else 0x00 (ggml bytes_from_bits_32).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn bytes_from_bits_32(x: u32) -> __m256i {
        let shuf = _mm256_set_epi64x(
            0x0303030303030303,
            0x0202020202020202,
            0x0101010101010101,
            0,
        );
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x as i32), shuf);
        let bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
        _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1))
    }

    /// Σ over each 4 consecutive byte products ax(u8)*sy(i8), as 8 f32 lanes.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn mul_sum_us8_avx2(ax: __m256i, sy: __m256i) -> __m256 {
        let dot = _mm256_maddubs_epi16(ax, sy);
        _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), dot))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    fn mul_sum_us8_vnni(ax: __m256i, sy: __m256i) -> __m256 {
        _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy))
    }

    /// 32 Q5_0 weights (nibble | high bit << 4) - 16 as i8.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) fn q5_0_weights(b: &[u8; Q5_0_BYTES]) -> __m256i {
        let qs = load_u8x16(b[6..22].try_into().unwrap());
        let nib = _mm256_and_si256(
            _mm256_set_m128i(_mm_srli_epi16(qs, 4), qs),
            _mm256_set1_epi8(0x0f),
        );
        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
        // 0xF0 (= -16 after the OR) where the high bit is clear
        let hi = _mm256_andnot_si256(bytes_from_bits_32(qh), _mm256_set1_epi8(0xF0u8 as i8));
        _mm256_or_si256(nib, hi)
    }

    /// 32 Q4_0 weights nibble - 8 as i8 (low nibbles, then high).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn q4_0_weights(b: &[u8; Q4_0_BYTES]) -> __m256i {
        let qs = load_u8x16(b[2..18].try_into().unwrap());
        let nib = _mm256_and_si256(
            _mm256_set_m128i(_mm_srli_epi16(qs, 4), qs),
            _mm256_set1_epi8(0x0f),
        );
        _mm256_sub_epi8(nib, _mm256_set1_epi8(8))
    }

    /// `dot_q_1_q8` with the block's integer sum Σ w qx by maddubs (exact:
    /// |pair| <= 2 * 31 * 128): the same integers and float ops, so the same
    /// bits as the scalar kernel.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c (`has_avx2()`).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q_1_avx2(q5: bool, row: &[u8], x: &B32) -> f32 {
        let bb = if q5 { Q5_1_BYTES } else { Q4_1_BYTES };
        let ones = _mm256_set1_epi16(1);
        let mut sum = 0f32;
        for (i, b) in row.chunks_exact(bb).enumerate() {
            let qs = load_u8x16(b[bb - 16..bb].try_into().unwrap());
            let mut w = _mm256_and_si256(
                _mm256_set_m128i(_mm_srli_epi16(qs, 4), qs),
                _mm256_set1_epi8(0x0f),
            );
            if q5 {
                let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let hi = _mm256_and_si256(bytes_from_bits_32(qh), _mm256_set1_epi8(0x10));
                w = _mm256_or_si256(w, hi);
            }
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(w, load_i8x32(&x.q[i])), ones);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
            let s64 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0b01_00_11_10));
            let s = _mm_cvtsi128_si32(_mm_add_epi32(s64, _mm_shuffle_epi32(s64, 0b10_11_00_01)));
            let d = h2f([b[0], b[1]]);
            let m = h2f([b[2], b[3]]);
            let bs = x.c[i].iter().sum::<i32>() / -16;
            sum += x.d[i] * (d * s as f32 + m * bs as f32);
        }
        sum
    }

    /// Broadcast i16 scale `i` of `scales` (8 i16, both lanes) to all 16 lanes.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn k4_scale(scales: __m256i, i: i32) -> __m256i {
        _mm256_shuffle_epi8(
            scales,
            _mm256_set1_epi16(((2 * i) | ((2 * i + 1) << 8)) as i16),
        )
    }

    /// Q6_K i8 scales of 16-groups 2i and 2i+1, each repeated 8 times, as i16.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn q6_scale(scales: __m128i, i: usize) -> __m256i {
        let lo = (2 * i) as i64 * 0x0101010101010101;
        let hi = (2 * i + 1) as i64 * 0x0101010101010101;
        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, _mm_set_epi64x(hi, lo)))
    }

    macro_rules! small_block_kernels {
        ($feat:literal, $q8:ident, $q5:ident, $q4:ident, $mul:ident) => {
            /// # Safety
            /// The CPU must support the enabled target features.
            #[target_feature(enable = $feat)]
            pub unsafe fn $q4(row: &[u8], dx: &[f32], qx: &[[i8; 32]]) -> f32 {
                let mut acc = _mm256_setzero_ps();
                for (b, (&dxb, qxb)) in row
                    .as_chunks::<Q4_0_BYTES>()
                    .0
                    .iter()
                    .zip(dx.iter().zip(qx))
                {
                    let d = _mm256_set1_ps(h2f([b[0], b[1]]) * dxb);
                    let qw = q4_0_weights(b);
                    let qy = load_i8x32(qxb);
                    let ax = _mm256_sign_epi8(qw, qw);
                    let sy = _mm256_sign_epi8(qy, qw);
                    acc = _mm256_fmadd_ps(d, $mul(ax, sy), acc);
                }
                hsum_f32x8(acc)
            }

            /// # Safety
            /// The CPU must support the enabled target features.
            #[target_feature(enable = $feat)]
            pub unsafe fn $q8(row: &[u8], dx: &[f32], qx: &[[i8; 32]]) -> f32 {
                let mut acc = _mm256_setzero_ps();
                for (b, (&dxb, qxb)) in row
                    .as_chunks::<Q8_0_BYTES>()
                    .0
                    .iter()
                    .zip(dx.iter().zip(qx))
                {
                    let d = _mm256_set1_ps(h2f([b[0], b[1]]) * dxb);
                    let qw = load_u8x32(b[2..34].try_into().unwrap());
                    let qy = load_i8x32(qxb);
                    let ax = _mm256_sign_epi8(qw, qw);
                    let sy = _mm256_sign_epi8(qy, qw);
                    acc = _mm256_fmadd_ps(d, $mul(ax, sy), acc);
                }
                hsum_f32x8(acc)
            }

            /// # Safety
            /// The CPU must support the enabled target features.
            #[target_feature(enable = $feat)]
            pub unsafe fn $q5(row: &[u8], dx: &[f32], qx: &[[i8; 32]]) -> f32 {
                let mut acc = _mm256_setzero_ps();
                for (b, (&dxb, qxb)) in row
                    .as_chunks::<Q5_0_BYTES>()
                    .0
                    .iter()
                    .zip(dx.iter().zip(qx))
                {
                    let d = _mm256_set1_ps(h2f([b[0], b[1]]) * dxb);
                    let qw = q5_0_weights(b);
                    let qy = load_i8x32(qxb);
                    let ax = _mm256_sign_epi8(qw, qw);
                    let sy = _mm256_sign_epi8(qy, qw);
                    acc = _mm256_fmadd_ps(d, $mul(ax, sy), acc);
                }
                hsum_f32x8(acc)
            }
        };
    }

    small_block_kernels!(
        "avx2,fma,f16c",
        q8_0_avx2,
        q5_0_avx2,
        q4_0_avx2,
        mul_sum_us8_avx2
    );
    small_block_kernels!(
        "avx2,fma,f16c,avxvnni",
        q8_0_vnni,
        q5_0_vnni,
        q4_0_vnni,
        mul_sum_us8_vnni
    );

    /// ggml_vec_dot_q4_K_q8_K (AVX2). Mins via bsums.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c (`has_avx2()`).
    #[allow(clippy::needless_range_loop)] // j picks several chunks
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q4_k_avx2(row: &[u8], xk: &[KBlock]) -> f32 {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
        let m4 = _mm256_set1_epi8(0xF);
        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();
        for (b, a) in row.as_chunks::<Q4_K_BYTES>().0.iter().zip(xk) {
            let d = a.d * h2f([b[0], b[1]]);
            let dmin = -a.d * h2f([b[2], b[3]]);

            // unpack the 6-bit scales/mins: utmp[0..2] = 8 scales, utmp[2..4] = 8 mins
            let u = |i: usize| {
                u32::from_le_bytes([b[4 + 4 * i], b[5 + 4 * i], b[6 + 4 * i], b[7 + 4 * i]])
            };
            let (u0, u1, u2) = (u(0), u(1), u(2));
            let t3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
            let t2 = u1 & KMASK1;
            let t1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
            let t0 = u0 & KMASK1;
            let mins_and_scales =
                _mm256_cvtepu8_epi16(_mm_set_epi32(t3 as i32, t2 as i32, t1 as i32, t0 as i32));

            let q8sums = load_i16x16(&a.bsums);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_set_m128i(sc128, sc128);

            let q4 = b[16..Q4_K_BYTES].as_chunks::<32>().0;
            let q8 = a.qs.as_chunks::<32>().0;
            let mut sumi = _mm256_setzero_si256();
            for j in 0..4 {
                let scale_l = k4_scale(scales, 2 * j as i32);
                let scale_h = k4_scale(scales, 2 * j as i32 + 1);

                let q4bits = load_u8x32(&q4[j]);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                let p16l =
                    _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, load_i8x32(&q8[2 * j])));
                let p16h = _mm256_madd_epi16(
                    scale_h,
                    _mm256_maddubs_epi16(q4h, load_i8x32(&q8[2 * j + 1])),
                );
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }
        acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
        hsum_f32x8(acc) + _mm_cvtss_f32(acc_m)
    }

    /// ggml_vec_dot_q5_K_q8_K (AVX2). Mins via bsums; bit 4 from qh.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c (`has_avx2()`).
    #[allow(clippy::needless_range_loop)] // j picks several chunks
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q5_k_avx2(row: &[u8], xk: &[KBlock]) -> f32 {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
        let m4 = _mm256_set1_epi8(0xF);
        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();
        for (b, a) in row.as_chunks::<Q5_K_BYTES>().0.iter().zip(xk) {
            let d = a.d * h2f([b[0], b[1]]);
            let dmin = -a.d * h2f([b[2], b[3]]);

            // unpack the 6-bit scales/mins: utmp[0..2] = 8 scales, utmp[2..4] = 8 mins
            let u = |i: usize| {
                u32::from_le_bytes([b[4 + 4 * i], b[5 + 4 * i], b[6 + 4 * i], b[7 + 4 * i]])
            };
            let (u0, u1, u2) = (u(0), u(1), u(2));
            let t3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
            let t2 = u1 & KMASK1;
            let t1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
            let t0 = u0 & KMASK1;
            let mins_and_scales =
                _mm256_cvtepu8_epi16(_mm_set_epi32(t3 as i32, t2 as i32, t1 as i32, t0 as i32));

            let q8sums = load_i16x16(&a.bsums);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_set_m128i(sc128, sc128);

            let q4 = b[48..Q5_K_BYTES].as_chunks::<32>().0;
            let hbits = load_u8x32(b[16..48].try_into().unwrap());
            let m1 = _mm256_set1_epi8(1);
            let q8 = a.qs.as_chunks::<32>().0;
            let mut sumi = _mm256_setzero_si256();
            for j in 0..4 {
                let scale_l = k4_scale(scales, 2 * j as i32);
                let scale_h = k4_scale(scales, 2 * j as i32 + 1);

                let q4bits = load_u8x32(&q4[j]);
                // bit 2j (low nibbles) / 2j + 1 (high) of qh, moved to bit 4
                let hl =
                    _mm256_and_si256(_mm256_srl_epi16(hbits, _mm_cvtsi32_si128(2 * j as i32)), m1);
                let hh = _mm256_and_si256(
                    _mm256_srl_epi16(hbits, _mm_cvtsi32_si128(2 * j as i32 + 1)),
                    m1,
                );
                let q4l = _mm256_or_si256(_mm256_and_si256(q4bits, m4), _mm256_slli_epi16(hl, 4));
                let q4h = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4),
                    _mm256_slli_epi16(hh, 4),
                );

                let p16l =
                    _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, load_i8x32(&q8[2 * j])));
                let p16h = _mm256_madd_epi16(
                    scale_h,
                    _mm256_maddubs_epi16(q4h, load_i8x32(&q8[2 * j + 1])),
                );
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }
        acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
        hsum_f32x8(acc) + _mm_cvtss_f32(acc_m)
    }

    /// ggml_vec_dot_q6_K_q8_K (AVX2). The -32 offset is applied via bsums.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c (`has_avx2()`).
    #[allow(clippy::needless_range_loop)] // j picks several chunks
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q6_k_avx2(row: &[u8], xk: &[KBlock]) -> f32 {
        let m3 = _mm256_set1_epi8(3);
        let m15 = _mm256_set1_epi8(15);
        let mut acc = _mm256_setzero_ps();
        for (b, a) in row.as_chunks::<Q6_K_BYTES>().0.iter().zip(xk) {
            let d = a.d * h2f([b[208], b[209]]);
            let ql = b[0..128].as_chunks::<32>().0;
            let qh = b[128..192].as_chunks::<32>().0;
            let q8 = a.qs.as_chunks::<32>().0;

            let scales = load_u8x16(b[192..208].try_into().unwrap());
            let q8sums = load_i16x16(&a.bsums);
            let q8sclsub =
                _mm256_slli_epi32(_mm256_madd_epi16(q8sums, _mm256_cvtepi8_epi16(scales)), 5);

            let mut sumi = _mm256_setzero_si256();
            for j in 0..2 {
                let q4bits1 = load_u8x32(&ql[2 * j]);
                let q4bits2 = load_u8x32(&ql[2 * j + 1]);
                let q4bits_h = load_u8x32(&qh[j]);

                let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
                let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(12)), 2);
                let q4h_2 = _mm256_and_si256(q4bits_h, _mm256_set1_epi8(48));
                let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(-64)), 2);

                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                let q4_2 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
                let q4_3 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

                let p16_0 = _mm256_maddubs_epi16(q4_0, load_i8x32(&q8[4 * j]));
                let p16_1 = _mm256_maddubs_epi16(q4_1, load_i8x32(&q8[4 * j + 1]));
                let p16_2 = _mm256_maddubs_epi16(q4_2, load_i8x32(&q8[4 * j + 2]));
                let p16_3 = _mm256_maddubs_epi16(q4_3, load_i8x32(&q8[4 * j + 3]));

                let p16_0 = _mm256_madd_epi16(q6_scale(scales, 4 * j), p16_0);
                let p16_1 = _mm256_madd_epi16(q6_scale(scales, 4 * j + 1), p16_1);
                let p16_2 = _mm256_madd_epi16(q6_scale(scales, 4 * j + 2), p16_2);
                let p16_3 = _mm256_madd_epi16(q6_scale(scales, 4 * j + 3), p16_3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
            }
            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_f32x8(acc)
    }

    // ---- 4-token kernels: the weight-only work of each block is done once and
    // shared by 4 tokens. Per token, the float ops are exactly those of the
    // single-token kernel above (same products, same fmadd chain, same hsum),
    // so each result is bit-identical; only the integer sums are reordered,
    // and i32 wrapping adds are associative.

    /// 32 Q8_0 weights as i8 (in a u8 vector).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) fn q8_0_weights(b: &[u8; Q8_0_BYTES]) -> __m256i {
        load_u8x32(b[2..34].try_into().unwrap())
    }

    macro_rules! small_block_kernels_x4 {
        ($feat:literal, $name:ident, $bytes:ident, $weights:ident, $mul:ident) => {
            /// 4-token version of the single-token kernel of the same type.
            ///
            /// # Safety
            /// The CPU must support the enabled target features.
            #[allow(clippy::needless_range_loop)] // t indexes several arrays
            #[target_feature(enable = $feat)]
            pub unsafe fn $name(row: &[u8], dx: [&[f32]; 4], qx: [&[[i8; 32]]; 4]) -> [f32; 4] {
                let blocks = row.as_chunks::<$bytes>().0;
                let n = blocks.len();
                // Written out, not `.map(..)`: a closure doesn't inherit this
                // fn's target features, so it compiles out of line and costs a
                // call per row in the hot path.
                let dx = [&dx[0][..n], &dx[1][..n], &dx[2][..n], &dx[3][..n]];
                let qx = [&qx[0][..n], &qx[1][..n], &qx[2][..n], &qx[3][..n]];
                let mut acc = [_mm256_setzero_ps(); 4];
                for (i, b) in blocks.iter().enumerate() {
                    let hw = h2f([b[0], b[1]]);
                    let qw = $weights(b);
                    let ax = _mm256_sign_epi8(qw, qw);
                    for t in 0..4 {
                        let d = _mm256_set1_ps(hw * dx[t][i]);
                        let qy = load_i8x32(&qx[t][i]);
                        let sy = _mm256_sign_epi8(qy, qw);
                        acc[t] = _mm256_fmadd_ps(d, $mul(ax, sy), acc[t]);
                    }
                }
                [
                    hsum_f32x8(acc[0]),
                    hsum_f32x8(acc[1]),
                    hsum_f32x8(acc[2]),
                    hsum_f32x8(acc[3]),
                ]
            }
        };
    }

    small_block_kernels_x4!(
        "avx2,fma,f16c",
        q8_0_avx2_x4,
        Q8_0_BYTES,
        q8_0_weights,
        mul_sum_us8_avx2
    );
    small_block_kernels_x4!(
        "avx2,fma,f16c",
        q5_0_avx2_x4,
        Q5_0_BYTES,
        q5_0_weights,
        mul_sum_us8_avx2
    );
    small_block_kernels_x4!(
        "avx2,fma,f16c,avxvnni",
        q8_0_vnni_x4,
        Q8_0_BYTES,
        q8_0_weights,
        mul_sum_us8_vnni
    );
    small_block_kernels_x4!(
        "avx2,fma,f16c,avxvnni",
        q5_0_vnni_x4,
        Q5_0_BYTES,
        q5_0_weights,
        mul_sum_us8_vnni
    );
    small_block_kernels_x4!(
        "avx2,fma,f16c",
        q4_0_avx2_x4,
        Q4_0_BYTES,
        q4_0_weights,
        mul_sum_us8_avx2
    );
    small_block_kernels_x4!(
        "avx2,fma,f16c,avxvnni",
        q4_0_vnni_x4,
        Q4_0_BYTES,
        q4_0_weights,
        mul_sum_us8_vnni
    );

    // ---- 2-row x 4-token tiles: per 32-weight block, unpack 2 rows' weights
    // once and load each token's activations once, for 8 (row, token) dots.
    // Per (row, token) the float ops are exactly the single-token kernel's, so
    // results are bit-identical. Both rows have the same type. (AVX2 path;
    // VNNI uses the unpack-once kernels below.)

    macro_rules! small_block_kernels_2x4 {
        ($feat:literal, $name:ident, $bytes:ident, $weights:ident, $mul:ident) => {
            /// 2-row x 4-token version of the single-token kernel of the same
            /// type. `rows` must have the same number of blocks.
            ///
            /// # Safety
            /// The CPU must support the enabled target features.
            #[allow(clippy::needless_range_loop)] // t indexes several arrays
            #[target_feature(enable = $feat)]
            pub unsafe fn $name(
                rows: [&[u8]; 2],
                dx: [&[f32]; 4],
                qx: [&[[i8; 32]]; 4],
            ) -> [[f32; 4]; 2] {
                let b0 = rows[0].as_chunks::<$bytes>().0;
                let n = b0.len();
                let b1 = &rows[1].as_chunks::<$bytes>().0[..n];
                // Written out, not `.map(..)`: see small_block_kernels_x4.
                let dx = [&dx[0][..n], &dx[1][..n], &dx[2][..n], &dx[3][..n]];
                let qx = [&qx[0][..n], &qx[1][..n], &qx[2][..n], &qx[3][..n]];
                let mut acc0 = [_mm256_setzero_ps(); 4];
                let mut acc1 = [_mm256_setzero_ps(); 4];
                for i in 0..n {
                    let (w0, w1) = (&b0[i], &b1[i]);
                    let hw0 = h2f([w0[0], w0[1]]);
                    let hw1 = h2f([w1[0], w1[1]]);
                    let qw0 = $weights(w0);
                    let qw1 = $weights(w1);
                    let ax0 = _mm256_sign_epi8(qw0, qw0);
                    let ax1 = _mm256_sign_epi8(qw1, qw1);
                    for t in 0..4 {
                        let dxt = dx[t][i];
                        let qy = load_i8x32(&qx[t][i]);
                        let d0 = _mm256_set1_ps(hw0 * dxt);
                        let sy0 = _mm256_sign_epi8(qy, qw0);
                        acc0[t] = _mm256_fmadd_ps(d0, $mul(ax0, sy0), acc0[t]);
                        let d1 = _mm256_set1_ps(hw1 * dxt);
                        let sy1 = _mm256_sign_epi8(qy, qw1);
                        acc1[t] = _mm256_fmadd_ps(d1, $mul(ax1, sy1), acc1[t]);
                    }
                }
                [
                    [
                        hsum_f32x8(acc0[0]),
                        hsum_f32x8(acc0[1]),
                        hsum_f32x8(acc0[2]),
                        hsum_f32x8(acc0[3]),
                    ],
                    [
                        hsum_f32x8(acc1[0]),
                        hsum_f32x8(acc1[1]),
                        hsum_f32x8(acc1[2]),
                        hsum_f32x8(acc1[3]),
                    ],
                ]
            }
        };
    }

    small_block_kernels_2x4!(
        "avx2,fma,f16c",
        q8_0_avx2_2x4,
        Q8_0_BYTES,
        q8_0_weights,
        mul_sum_us8_avx2
    );
    small_block_kernels_2x4!(
        "avx2,fma,f16c",
        q5_0_avx2_2x4,
        Q5_0_BYTES,
        q5_0_weights,
        mul_sum_us8_avx2
    );

    // ---- VNNI 2-row x many-token: unpack a row pair once, then run every
    // 4-token group over the unpacked blocks. Weights go to dpbusd unsigned
    // (u = w + 16 for Q5_0, w + 128 for Q8_0) with the accumulator started at
    // -offset * (lane sum of the activations) (`B32::c`), which gives the same
    // i32 per lane as the single-token kernel's dpbusd(0, |w|, sign(a, w)):
    // both are the exact Σ w·a (|a| <= 127, no overflow, dpbusd doesn't
    // saturate). The scale product row_d * token_d is a vector mul of the
    // same two f32s, and the fmadd chain is unchanged, so results are
    // bit-identical to the single-token kernel.

    /// One block of a row pair, unpacked: unsigned weights and f32 scales.
    #[derive(Clone, Copy)]
    #[repr(C)]
    pub struct Pair32 {
        u: [[u8; 32]; 2],
        d: [f32; 2],
    }

    impl Pair32 {
        pub const ZERO: Pair32 = Pair32 {
            u: [[0; 32]; 2],
            d: [0.0; 2],
        };
    }

    /// 32 Q5_0 raw values nibble | high bit << 4, in 0..32 (= w + 16).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn q5_0_raw(b: &[u8; Q5_0_BYTES]) -> __m256i {
        let qs = load_u8x16(b[6..22].try_into().unwrap());
        let nib = _mm256_and_si256(
            _mm256_set_m128i(_mm_srli_epi16(qs, 4), qs),
            _mm256_set1_epi8(0x0f),
        );
        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
        let hi = _mm256_and_si256(bytes_from_bits_32(qh), _mm256_set1_epi8(0x10));
        _mm256_or_si256(nib, hi)
    }

    /// 32 Q8_0 weights + 128 as u8.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn q8_0_raw(b: &[u8; Q8_0_BYTES]) -> __m256i {
        _mm256_xor_si256(q8_0_weights(b), _mm256_set1_epi8(0x80u8 as i8))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn store_u8x32(p: &mut [u8; 32], v: __m256i) {
        // SAFETY: p points to 32 writable bytes; storeu has no alignment requirement.
        unsafe { _mm256_storeu_si256(p.as_mut_ptr().cast(), v) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn load_i32x8(p: &[i32; 8]) -> __m256i {
        // SAFETY: as load_u8x32 (32 bytes).
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    macro_rules! unpack_pair {
        ($name:ident, $bytes:ident, $raw:ident) => {
            /// Unpack `out.len()` blocks of both rows into `out`.
            ///
            /// # Safety
            /// The CPU must support avx2 and f16c.
            #[target_feature(enable = "avx2,fma,f16c")]
            pub unsafe fn $name(rows: [&[u8]; 2], out: &mut [Pair32]) {
                let n = out.len();
                let b0 = &rows[0].as_chunks::<$bytes>().0[..n];
                let b1 = &rows[1].as_chunks::<$bytes>().0[..n];
                for (p, (w0, w1)) in out.iter_mut().zip(b0.iter().zip(b1)) {
                    store_u8x32(&mut p.u[0], $raw(w0));
                    store_u8x32(&mut p.u[1], $raw(w1));
                    p.d = [h2f([w0[0], w0[1]]), h2f([w1[0], w1[1]])];
                }
            }
        };
    }

    unpack_pair!(unpack_q5_0_pair, Q5_0_BYTES, q5_0_raw);
    unpack_pair!(unpack_q8_0_pair, Q8_0_BYTES, q8_0_raw);

    /// 2 unpacked rows x 4 tokens. `Q8` picks the Q8_0 offset (c << 3)
    /// over Q5_0's (c as stored).
    ///
    /// # Safety
    /// The CPU must support avx2, fma, f16c and avxvnni (`has_vnni()`).
    #[allow(clippy::needless_range_loop)] // t indexes several arrays
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub unsafe fn vnni_pair_x4<const Q8: bool>(
        w: &[Pair32],
        dx: [&[f32]; 4],
        qx: [&[[i8; 32]]; 4],
        c: [&[[i32; 8]]; 4],
    ) -> [[f32; 4]; 2] {
        let n = w.len();
        // Written out, not `.map(..)`: see small_block_kernels_x4.
        let dx = [&dx[0][..n], &dx[1][..n], &dx[2][..n], &dx[3][..n]];
        let qx = [&qx[0][..n], &qx[1][..n], &qx[2][..n], &qx[3][..n]];
        let c = [&c[0][..n], &c[1][..n], &c[2][..n], &c[3][..n]];
        let mut acc0 = [_mm256_setzero_ps(); 4];
        let mut acc1 = [_mm256_setzero_ps(); 4];
        for i in 0..n {
            let p = &w[i];
            let u0 = load_u8x32(&p.u[0]);
            let u1 = load_u8x32(&p.u[1]);
            let d0 = _mm256_set1_ps(p.d[0]);
            let d1 = _mm256_set1_ps(p.d[1]);
            for t in 0..4 {
                let qy = load_i8x32(&qx[t][i]);
                let mut ct = load_i32x8(&c[t][i]);
                if Q8 {
                    ct = _mm256_slli_epi32::<3>(ct);
                }
                let dt = _mm256_set1_ps(dx[t][i]);
                let s0 = _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(ct, u0, qy));
                acc0[t] = _mm256_fmadd_ps(_mm256_mul_ps(d0, dt), s0, acc0[t]);
                let s1 = _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(ct, u1, qy));
                acc1[t] = _mm256_fmadd_ps(_mm256_mul_ps(d1, dt), s1, acc1[t]);
            }
        }
        [
            [
                hsum_f32x8(acc0[0]),
                hsum_f32x8(acc0[1]),
                hsum_f32x8(acc0[2]),
                hsum_f32x8(acc0[3]),
            ],
            [
                hsum_f32x8(acc1[0]),
                hsum_f32x8(acc1[1]),
                hsum_f32x8(acc1[2]),
                hsum_f32x8(acc1[3]),
            ],
        ]
    }

    /// 4-token `q4_k_avx2`.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c (`has_avx2()`).
    #[allow(clippy::needless_range_loop)] // j, t pick several chunks
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q4_k_avx2_x4(row: &[u8], xk: [&[KBlock]; 4]) -> [f32; 4] {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
        let m4 = _mm256_set1_epi8(0xF);
        let blocks = row.as_chunks::<Q4_K_BYTES>().0;
        let n = blocks.len();
        // Written out, not `.map(..)`: see small_block_kernels_x4.
        let xk = [&xk[0][..n], &xk[1][..n], &xk[2][..n], &xk[3][..n]];
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut acc_m = [_mm_setzero_ps(); 4];
        for (i, b) in blocks.iter().enumerate() {
            let hd = h2f([b[0], b[1]]);
            let hmin = h2f([b[2], b[3]]);

            let u = |i: usize| {
                u32::from_le_bytes([b[4 + 4 * i], b[5 + 4 * i], b[6 + 4 * i], b[7 + 4 * i]])
            };
            let (u0, u1, u2) = (u(0), u(1), u(2));
            let t3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
            let t2 = u1 & KMASK1;
            let t1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
            let t0 = u0 & KMASK1;
            let mins_and_scales =
                _mm256_cvtepu8_epi16(_mm_set_epi32(t3 as i32, t2 as i32, t1 as i32, t0 as i32));
            let mins = _mm256_extracti128_si256(mins_and_scales, 1);

            for t in 0..4 {
                let a = &xk[t][i];
                let dmin = -a.d * hmin;
                let q8sums = load_i16x16(&a.bsums);
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(mins, q8s);
                acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            }

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_set_m128i(sc128, sc128);

            let q4 = b[16..Q4_K_BYTES].as_chunks::<32>().0;
            let mut sumi = [_mm256_setzero_si256(); 4];
            for j in 0..4 {
                let scale_l = k4_scale(scales, 2 * j as i32);
                let scale_h = k4_scale(scales, 2 * j as i32 + 1);

                let q4bits = load_u8x32(&q4[j]);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                for t in 0..4 {
                    let q8 = xk[t][i].qs.as_chunks::<32>().0;
                    let p16l = _mm256_madd_epi16(
                        scale_l,
                        _mm256_maddubs_epi16(q4l, load_i8x32(&q8[2 * j])),
                    );
                    let p16h = _mm256_madd_epi16(
                        scale_h,
                        _mm256_maddubs_epi16(q4h, load_i8x32(&q8[2 * j + 1])),
                    );
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p16l, p16h));
                }
            }
            for t in 0..4 {
                let d = xk[t][i].d * hd;
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
        }
        let mut out = [0f32; 4];
        for t in 0..4 {
            let mut m = acc_m[t];
            m = _mm_add_ps(m, _mm_movehl_ps(m, m));
            m = _mm_add_ss(m, _mm_movehdup_ps(m));
            out[t] = hsum_f32x8(acc[t]) + _mm_cvtss_f32(m);
        }
        out
    }

    /// 4-token `q5_k_avx2`.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c (`has_avx2()`).
    #[allow(clippy::needless_range_loop)] // j, t pick several chunks
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q5_k_avx2_x4(row: &[u8], xk: [&[KBlock]; 4]) -> [f32; 4] {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
        let m4 = _mm256_set1_epi8(0xF);
        let blocks = row.as_chunks::<Q5_K_BYTES>().0;
        let n = blocks.len();
        // Written out, not `.map(..)`: see small_block_kernels_x4.
        let xk = [&xk[0][..n], &xk[1][..n], &xk[2][..n], &xk[3][..n]];
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut acc_m = [_mm_setzero_ps(); 4];
        for (i, b) in blocks.iter().enumerate() {
            let hd = h2f([b[0], b[1]]);
            let hmin = h2f([b[2], b[3]]);

            let u = |i: usize| {
                u32::from_le_bytes([b[4 + 4 * i], b[5 + 4 * i], b[6 + 4 * i], b[7 + 4 * i]])
            };
            let (u0, u1, u2) = (u(0), u(1), u(2));
            let t3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
            let t2 = u1 & KMASK1;
            let t1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
            let t0 = u0 & KMASK1;
            let mins_and_scales =
                _mm256_cvtepu8_epi16(_mm_set_epi32(t3 as i32, t2 as i32, t1 as i32, t0 as i32));
            let mins = _mm256_extracti128_si256(mins_and_scales, 1);

            for t in 0..4 {
                let a = &xk[t][i];
                let dmin = -a.d * hmin;
                let q8sums = load_i16x16(&a.bsums);
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(mins, q8s);
                acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            }

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_set_m128i(sc128, sc128);

            let q4 = b[48..Q5_K_BYTES].as_chunks::<32>().0;
            let hbits = load_u8x32(b[16..48].try_into().unwrap());
            let m1 = _mm256_set1_epi8(1);
            let mut sumi = [_mm256_setzero_si256(); 4];
            for j in 0..4 {
                let scale_l = k4_scale(scales, 2 * j as i32);
                let scale_h = k4_scale(scales, 2 * j as i32 + 1);

                let q4bits = load_u8x32(&q4[j]);
                // bit 2j (low nibbles) / 2j + 1 (high) of qh, moved to bit 4
                let hl =
                    _mm256_and_si256(_mm256_srl_epi16(hbits, _mm_cvtsi32_si128(2 * j as i32)), m1);
                let hh = _mm256_and_si256(
                    _mm256_srl_epi16(hbits, _mm_cvtsi32_si128(2 * j as i32 + 1)),
                    m1,
                );
                let q4l = _mm256_or_si256(_mm256_and_si256(q4bits, m4), _mm256_slli_epi16(hl, 4));
                let q4h = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4),
                    _mm256_slli_epi16(hh, 4),
                );

                for t in 0..4 {
                    let q8 = xk[t][i].qs.as_chunks::<32>().0;
                    let p16l = _mm256_madd_epi16(
                        scale_l,
                        _mm256_maddubs_epi16(q4l, load_i8x32(&q8[2 * j])),
                    );
                    let p16h = _mm256_madd_epi16(
                        scale_h,
                        _mm256_maddubs_epi16(q4h, load_i8x32(&q8[2 * j + 1])),
                    );
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p16l, p16h));
                }
            }
            for t in 0..4 {
                let d = xk[t][i].d * hd;
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
        }
        let mut out = [0f32; 4];
        for t in 0..4 {
            let mut m = acc_m[t];
            m = _mm_add_ps(m, _mm_movehl_ps(m, m));
            m = _mm_add_ss(m, _mm_movehdup_ps(m));
            out[t] = hsum_f32x8(acc[t]) + _mm_cvtss_f32(m);
        }
        out
    }

    /// 4-token `q6_k_avx2`.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c (`has_avx2()`).
    #[allow(clippy::needless_range_loop)] // j, t pick several chunks
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q6_k_avx2_x4(row: &[u8], xk: [&[KBlock]; 4]) -> [f32; 4] {
        let m3 = _mm256_set1_epi8(3);
        let m15 = _mm256_set1_epi8(15);
        let blocks = row.as_chunks::<Q6_K_BYTES>().0;
        let n = blocks.len();
        // Written out, not `.map(..)`: see small_block_kernels_x4.
        let xk = [&xk[0][..n], &xk[1][..n], &xk[2][..n], &xk[3][..n]];
        let mut acc = [_mm256_setzero_ps(); 4];
        for (i, b) in blocks.iter().enumerate() {
            let hd = h2f([b[208], b[209]]);
            let ql = b[0..128].as_chunks::<32>().0;
            let qh = b[128..192].as_chunks::<32>().0;

            let scales = load_u8x16(b[192..208].try_into().unwrap());
            let scales16 = _mm256_cvtepi8_epi16(scales);

            let mut sumi = [_mm256_setzero_si256(); 4];
            for j in 0..2 {
                let q4bits1 = load_u8x32(&ql[2 * j]);
                let q4bits2 = load_u8x32(&ql[2 * j + 1]);
                let q4bits_h = load_u8x32(&qh[j]);

                let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
                let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(12)), 2);
                let q4h_2 = _mm256_and_si256(q4bits_h, _mm256_set1_epi8(48));
                let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(-64)), 2);

                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                let q4_2 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
                let q4_3 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

                let s0 = q6_scale(scales, 4 * j);
                let s1 = q6_scale(scales, 4 * j + 1);
                let s2 = q6_scale(scales, 4 * j + 2);
                let s3 = q6_scale(scales, 4 * j + 3);

                for t in 0..4 {
                    let q8 = xk[t][i].qs.as_chunks::<32>().0;
                    let p16_0 = _mm256_maddubs_epi16(q4_0, load_i8x32(&q8[4 * j]));
                    let p16_1 = _mm256_maddubs_epi16(q4_1, load_i8x32(&q8[4 * j + 1]));
                    let p16_2 = _mm256_maddubs_epi16(q4_2, load_i8x32(&q8[4 * j + 2]));
                    let p16_3 = _mm256_maddubs_epi16(q4_3, load_i8x32(&q8[4 * j + 3]));

                    let p16_0 = _mm256_madd_epi16(s0, p16_0);
                    let p16_1 = _mm256_madd_epi16(s1, p16_1);
                    let p16_2 = _mm256_madd_epi16(s2, p16_2);
                    let p16_3 = _mm256_madd_epi16(s3, p16_3);

                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p16_0, p16_1));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p16_2, p16_3));
                }
            }
            for t in 0..4 {
                let a = &xk[t][i];
                let q8sums = load_i16x16(&a.bsums);
                let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), 5);
                let s = _mm256_sub_epi32(sumi[t], q8sclsub);
                let d = a.d * hd;
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s), acc[t]);
            }
        }
        [
            hsum_f32x8(acc[0]),
            hsum_f32x8(acc[1]),
            hsum_f32x8(acc[2]),
            hsum_f32x8(acc[3]),
        ]
    }

    // ---- K-quant 2-row x many-token. Per super-block, both rows are unpacked
    // once (weights one byte each, scales decoded and widened to i16 lanes)
    // and every 4-token group of the call runs over that block before the
    // next one; the per-(row, token) f32 accumulators live in a `KAcc` array.
    // So each block is unpacked once per call and each activation block is
    // read once per row pair (the x4 kernels read it once per row).
    //
    // Per 32-weight chunk and token the integer work is maddubs(u, a), then
    // "sumi += madd(scale, p16)": on Vnni a single dpwssd, which is that exact
    // sum (i16*i16 products, no saturation). Every i32 lane of sumi is the
    // single-token kernel's lane, and the float ops per (row, token) (d
    // product, the per-block fmadd chain, the mins chain, the final hsum) are
    // unchanged, so results are bit-identical.

    /// One super-block of a K-quant row pair, unpacked. Chunk c (32 weights)
    /// lines up with activation bytes 32c..32c+32.
    #[repr(C, align(32))]
    struct KPair {
        /// unsigned weights: Q4_K nibble 0..15, Q6_K q 0..63 (before -32)
        u: [[[u8; 32]; 8]; 2],
        /// per chunk, the i16 scale of each maddubs lane (Q4_K: one scale for
        /// all 16; Q6_K: 16-group 2c in lanes 0..8, 2c+1 in lanes 8..16)
        sc: [[[i16; 16]; 8]; 2],
        /// Q4_K: the 8 mins in lanes 0..8; Q6_K: -32 * the 16 scales, so
        /// madd(bsums, m) is the -32 offset correction per i32 lane
        m: [[i16; 16]; 2],
        /// f16 d of each row (Q4_K and Q6_K)
        d: [f32; 2],
        /// f16 dmin of each row (Q4_K)
        dmin: [f32; 2],
    }

    impl KPair {
        const ZERO: KPair = KPair {
            u: [[[0; 32]; 8]; 2],
            sc: [[[0; 16]; 8]; 2],
            m: [[0; 16]; 2],
            d: [0.0; 2],
            dmin: [0.0; 2],
        };
    }

    /// One token's f32 accumulators for a row pair (`accm`: Q4_K mins only).
    #[derive(Clone, Copy)]
    #[repr(C, align(32))]
    pub struct KAcc {
        acc: [[f32; 8]; 2],
        accm: [[f32; 4]; 2],
    }

    impl KAcc {
        pub const ZERO: KAcc = KAcc {
            acc: [[0.0; 8]; 2],
            accm: [[0.0; 4]; 2],
        };
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn store_i16x16(p: &mut [i16; 16], v: __m256i) {
        // SAFETY: p points to 32 writable bytes; storeu has no alignment requirement.
        unsafe { _mm256_storeu_si256(p.as_mut_ptr().cast(), v) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn load_i16x8(p: &[i16; 8]) -> __m128i {
        // SAFETY: as load_u8x16 (16 bytes).
        unsafe { _mm_loadu_si128(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn load_f32x8(p: &[f32; 8]) -> __m256 {
        // SAFETY: p points to 8 readable f32s; loadu has no alignment requirement.
        unsafe { _mm256_loadu_ps(p.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn store_f32x8(p: &mut [f32; 8], v: __m256) {
        // SAFETY: p points to 8 writable f32s; storeu has no alignment requirement.
        unsafe { _mm256_storeu_ps(p.as_mut_ptr(), v) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn load_f32x4(p: &[f32; 4]) -> __m128 {
        // SAFETY: as load_f32x8 (4 f32s).
        unsafe { _mm_loadu_ps(p.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn store_f32x4(p: &mut [f32; 4], v: __m128) {
        // SAFETY: as store_f32x8 (4 f32s).
        unsafe { _mm_storeu_ps(p.as_mut_ptr(), v) }
    }

    /// Unpack Q4_K super-block `b` as row `r` of `p`.
    #[allow(clippy::needless_range_loop)] // j, k index several arrays
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn unpack_q4_k_block(b: &[u8; Q4_K_BYTES], r: usize, p: &mut KPair) {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
        let m4 = _mm256_set1_epi8(0xF);
        p.d[r] = h2f([b[0], b[1]]);
        p.dmin[r] = h2f([b[2], b[3]]);
        // the same decode as q4_k_avx2: t0, t1 = 8 scales, t2, t3 = 8 mins
        let u0 = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        let u1 = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
        let u2 = u32::from_le_bytes([b[12], b[13], b[14], b[15]]);
        let t3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
        let t2 = u1 & KMASK1;
        let t1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
        let t0 = u0 & KMASK1;
        let sc = [t0.to_le_bytes(), t1.to_le_bytes()];
        let mn = [t2.to_le_bytes(), t3.to_le_bytes()];
        for k in 0..8 {
            store_i16x16(&mut p.sc[r][k], _mm256_set1_epi16(sc[k / 4][k % 4] as i16));
            p.m[r][k] = mn[k / 4][k % 4] as i16;
        }
        let q4 = b[16..Q4_K_BYTES].as_chunks::<32>().0;
        for j in 0..4 {
            let q4bits = load_u8x32(&q4[j]);
            store_u8x32(&mut p.u[r][2 * j], _mm256_and_si256(q4bits, m4));
            store_u8x32(
                &mut p.u[r][2 * j + 1],
                _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4),
            );
        }
    }

    /// Unpack Q6_K super-block `b` as row `r` of `p`.
    #[allow(clippy::needless_range_loop)] // j, k index several arrays
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn unpack_q6_k_block(b: &[u8; Q6_K_BYTES], r: usize, p: &mut KPair) {
        let m3 = _mm256_set1_epi8(3);
        let m15 = _mm256_set1_epi8(15);
        p.d[r] = h2f([b[208], b[209]]);
        let scales = load_u8x16(b[192..208].try_into().unwrap());
        // -32 * scale; |scale| <= 128, so it fits i16
        let s16 = _mm256_cvtepi8_epi16(scales);
        store_i16x16(
            &mut p.m[r],
            _mm256_sub_epi16(_mm256_setzero_si256(), _mm256_slli_epi16(s16, 5)),
        );
        for c in 0..8 {
            store_i16x16(&mut p.sc[r][c], q6_scale(scales, c));
        }
        // the same bit assembly as q6_k_avx2
        let ql = b[0..128].as_chunks::<32>().0;
        let qh = b[128..192].as_chunks::<32>().0;
        let u = &mut p.u[r];
        for j in 0..2 {
            let q4bits1 = load_u8x32(&ql[2 * j]);
            let q4bits2 = load_u8x32(&ql[2 * j + 1]);
            let q4bits_h = load_u8x32(&qh[j]);

            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
            let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(12)), 2);
            let q4h_2 = _mm256_and_si256(q4bits_h, _mm256_set1_epi8(48));
            let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(-64)), 2);

            let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            store_u8x32(&mut u[4 * j], q4_0);
            store_u8x32(&mut u[4 * j + 1], q4_1);
            store_u8x32(&mut u[4 * j + 2], q4_2);
            store_u8x32(&mut u[4 * j + 3], q4_3);
        }
    }

    /// sumi + madd(sc, p16), as the AVX2 K-quant kernels do it.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn scale_acc_avx2(sumi: __m256i, sc: __m256i, p16: __m256i) -> __m256i {
        _mm256_add_epi32(sumi, _mm256_madd_epi16(sc, p16))
    }

    /// The same i32 lanes in one instruction.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    fn scale_acc_vnni(sumi: __m256i, sc: __m256i, p16: __m256i) -> __m256i {
        _mm256_dpwssd_avx_epi32(sumi, sc, p16)
    }

    /// Σ over the block's 8 chunks of scale * (u . a) for 2 rows x 4 tokens,
    /// added to `sumi0/1` (which hold each lane's starting value).
    macro_rules! kquant_chunks {
        ($acc:ident, $p:ident, $x4:ident, $sumi0:ident, $sumi1:ident) => {
            for c in 0..8 {
                let u0 = load_u8x32(&$p.u[0][c]);
                let u1 = load_u8x32(&$p.u[1][c]);
                let s0 = load_i16x16(&$p.sc[0][c]);
                let s1 = load_i16x16(&$p.sc[1][c]);
                for t in 0..4 {
                    let a = load_i8x32(&$x4[t].qs.as_chunks::<32>().0[c]);
                    $sumi0[t] = $acc($sumi0[t], s0, _mm256_maddubs_epi16(u0, a));
                    $sumi1[t] = $acc($sumi1[t], s1, _mm256_maddubs_epi16(u1, a));
                }
            }
        };
    }

    macro_rules! kquant_pair_kernels {
        ($feat:literal, $q4k:ident, $q6k:ident, $acc:ident) => {
            /// Two Q4_K rows x `xk.len()` tokens (a multiple of 4): `out[r][t]`
            /// is `q4_k_avx2(rows[r], xk[t])`, bit for bit. `acc` is scratch,
            /// at least `xk.len()` long.
            ///
            /// # Safety
            /// The CPU must support the enabled target features.
            #[allow(clippy::needless_range_loop)] // c, t index several arrays
            #[target_feature(enable = $feat)]
            pub unsafe fn $q4k(
                rows: [&[u8]; 2],
                xk: &[&[KBlock]],
                acc: &mut [KAcc],
                out: [&mut [f32]; 2],
            ) {
                let b0 = rows[0].as_chunks::<Q4_K_BYTES>().0;
                let n = b0.len();
                let b1 = &rows[1].as_chunks::<Q4_K_BYTES>().0[..n];
                let nt = xk.len();
                assert!(nt % 4 == 0 && acc.len() >= nt && out[0].len() >= nt && out[1].len() >= nt);
                let acc = &mut acc[..nt];
                acc.fill(KAcc::ZERO);
                let mut p = KPair::ZERO;
                for i in 0..n {
                    unpack_q4_k_block(&b0[i], 0, &mut p);
                    unpack_q4_k_block(&b1[i], 1, &mut p);
                    let mins0 = load_i16x8(p.m[0][..8].try_into().unwrap());
                    let mins1 = load_i16x8(p.m[1][..8].try_into().unwrap());
                    for (x4, k4) in xk.chunks_exact(4).zip(acc.chunks_exact_mut(4)) {
                        let x4 = [&x4[0][i], &x4[1][i], &x4[2][i], &x4[3][i]];
                        let mut sumi0 = [_mm256_setzero_si256(); 4];
                        let mut sumi1 = [_mm256_setzero_si256(); 4];
                        kquant_chunks!($acc, p, x4, sumi0, sumi1);
                        for t in 0..4 {
                            let a = x4[t];
                            let k = &mut k4[t];
                            let q8sums = load_i16x16(&a.bsums);
                            let q8s = _mm_hadd_epi16(
                                _mm256_extracti128_si256(q8sums, 0),
                                _mm256_extracti128_si256(q8sums, 1),
                            );
                            let dmin0 = -a.d * p.dmin[0];
                            let dmin1 = -a.d * p.dmin[1];
                            let prod0 = _mm_cvtepi32_ps(_mm_madd_epi16(mins0, q8s));
                            let prod1 = _mm_cvtepi32_ps(_mm_madd_epi16(mins1, q8s));
                            let m0 =
                                _mm_fmadd_ps(_mm_set1_ps(dmin0), prod0, load_f32x4(&k.accm[0]));
                            let m1 =
                                _mm_fmadd_ps(_mm_set1_ps(dmin1), prod1, load_f32x4(&k.accm[1]));
                            store_f32x4(&mut k.accm[0], m0);
                            store_f32x4(&mut k.accm[1], m1);
                            let d0 = a.d * p.d[0];
                            let d1 = a.d * p.d[1];
                            let s0 = _mm256_cvtepi32_ps(sumi0[t]);
                            let s1 = _mm256_cvtepi32_ps(sumi1[t]);
                            let a0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), s0, load_f32x8(&k.acc[0]));
                            let a1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), s1, load_f32x8(&k.acc[1]));
                            store_f32x8(&mut k.acc[0], a0);
                            store_f32x8(&mut k.acc[1], a1);
                        }
                    }
                }
                let [o0, o1] = out;
                for (t, k) in acc.iter().enumerate() {
                    let mut m = load_f32x4(&k.accm[0]);
                    m = _mm_add_ps(m, _mm_movehl_ps(m, m));
                    m = _mm_add_ss(m, _mm_movehdup_ps(m));
                    o0[t] = hsum_f32x8(load_f32x8(&k.acc[0])) + _mm_cvtss_f32(m);
                    let mut m = load_f32x4(&k.accm[1]);
                    m = _mm_add_ps(m, _mm_movehl_ps(m, m));
                    m = _mm_add_ss(m, _mm_movehdup_ps(m));
                    o1[t] = hsum_f32x8(load_f32x8(&k.acc[1])) + _mm_cvtss_f32(m);
                }
            }

            /// Two Q6_K rows x `xk.len()` tokens (a multiple of 4): `out[r][t]`
            /// is `q6_k_avx2(rows[r], xk[t])`, bit for bit. sumi starts at the
            /// -32 offset correction madd(bsums, -32 * scales) instead of
            /// subtracting madd(bsums, scales) << 5 at the end: the same i32
            /// per lane. `acc` is scratch, at least `xk.len()` long.
            ///
            /// # Safety
            /// The CPU must support the enabled target features.
            #[allow(clippy::needless_range_loop)] // c, t index several arrays
            #[target_feature(enable = $feat)]
            pub unsafe fn $q6k(
                rows: [&[u8]; 2],
                xk: &[&[KBlock]],
                acc: &mut [KAcc],
                out: [&mut [f32]; 2],
            ) {
                let b0 = rows[0].as_chunks::<Q6_K_BYTES>().0;
                let n = b0.len();
                let b1 = &rows[1].as_chunks::<Q6_K_BYTES>().0[..n];
                let nt = xk.len();
                assert!(nt % 4 == 0 && acc.len() >= nt && out[0].len() >= nt && out[1].len() >= nt);
                let acc = &mut acc[..nt];
                acc.fill(KAcc::ZERO);
                let mut p = KPair::ZERO;
                for i in 0..n {
                    unpack_q6_k_block(&b0[i], 0, &mut p);
                    unpack_q6_k_block(&b1[i], 1, &mut p);
                    let m0 = load_i16x16(&p.m[0]);
                    let m1 = load_i16x16(&p.m[1]);
                    for (x4, k4) in xk.chunks_exact(4).zip(acc.chunks_exact_mut(4)) {
                        let x4 = [&x4[0][i], &x4[1][i], &x4[2][i], &x4[3][i]];
                        let mut sumi0 = [_mm256_setzero_si256(); 4];
                        let mut sumi1 = [_mm256_setzero_si256(); 4];
                        for t in 0..4 {
                            let q8sums = load_i16x16(&x4[t].bsums);
                            sumi0[t] = _mm256_madd_epi16(q8sums, m0);
                            sumi1[t] = _mm256_madd_epi16(q8sums, m1);
                        }
                        kquant_chunks!($acc, p, x4, sumi0, sumi1);
                        for t in 0..4 {
                            let k = &mut k4[t];
                            let ad = x4[t].d;
                            let d0 = ad * p.d[0];
                            let d1 = ad * p.d[1];
                            let s0 = _mm256_cvtepi32_ps(sumi0[t]);
                            let s1 = _mm256_cvtepi32_ps(sumi1[t]);
                            let a0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), s0, load_f32x8(&k.acc[0]));
                            let a1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), s1, load_f32x8(&k.acc[1]));
                            store_f32x8(&mut k.acc[0], a0);
                            store_f32x8(&mut k.acc[1], a1);
                        }
                    }
                }
                let [o0, o1] = out;
                for (t, k) in acc.iter().enumerate() {
                    o0[t] = hsum_f32x8(load_f32x8(&k.acc[0]));
                    o1[t] = hsum_f32x8(load_f32x8(&k.acc[1]));
                }
            }
        };
    }

    kquant_pair_kernels!(
        "avx2,fma,f16c",
        q4_k_pair_avx2,
        q6_k_pair_avx2,
        scale_acc_avx2
    );
    kquant_pair_kernels!(
        "avx2,fma,f16c,avxvnni",
        q4_k_pair_vnni,
        q6_k_pair_vnni,
        scale_acc_vnni
    );

    // ---- Decode (single-token) kernels over R consecutive rows. Row r is
    // `rows[r * rb..(r + 1) * rb]`; `rows` may run further (to the end of the
    // tensor) and is only read there by prefetch hints. Each row keeps its
    // own fmadd chain, so R chains overlap, and each activation block is
    // loaded once for all R rows. Per row the float ops are exactly the
    // single-row kernel's: d = h2f(row d) * x d, fmadd(d, cvt(lanes), acc) in
    // block order, then the same hsum. Only how the i32 lanes are formed
    // differs, and every lane is the exact Σ w·a of its 4 products (proved
    // in the tests), so results are bit-identical.

    /// Hint the line at `rows[off]` into L1; nothing if `off` is past the
    /// slice (the caller's rows end there; hints beyond would be other
    /// threads' rows, i.e. extra DRAM traffic).
    #[inline(always)]
    fn prefetch(rows: &[u8], off: usize) {
        if off < rows.len() {
            // SAFETY: off is in bounds; a prefetch never faults anyway.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(rows.as_ptr().add(off).cast()) }
        }
    }

    /// Q5_0 lanes, unsigned: u = w + 16 in 0..32 via dpbusd from c = -16 Σa.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub fn q5_0_lanes_vnni(b: &[u8; Q5_0_BYTES], qy: __m256i, c: &[i32; 8]) -> __m256i {
        _mm256_dpbusd_avx_epi32(load_i32x8(c), q5_0_raw(b), qy)
    }

    /// Q5_0 lanes, unsigned on plain AVX2: maddubs(u, a) pairs are at most
    /// 2 * 31 * 128 = 7936 in magnitude (no i16 saturation), madd sums them
    /// exactly, then + c = -16 Σa.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn q5_0_lanes_avx2(b: &[u8; Q5_0_BYTES], qy: __m256i, c: &[i32; 8]) -> __m256i {
        let p = _mm256_madd_epi16(_mm256_maddubs_epi16(q5_0_raw(b), qy), _mm256_set1_epi16(1));
        _mm256_add_epi32(p, load_i32x8(c))
    }

    /// Q8_0 lanes, unsigned: u = w ^ 0x80 = w + 128 via dpbusd from
    /// c << 3 = -128 Σa.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub fn q8_0_lanes_vnni(b: &[u8; Q8_0_BYTES], qy: __m256i, c: &[i32; 8]) -> __m256i {
        let c = _mm256_slli_epi32::<3>(load_i32x8(c));
        _mm256_dpbusd_avx_epi32(c, q8_0_raw(b), qy)
    }

    /// Q8_0 lanes on plain AVX2: the sign trick, as the single-row kernel
    /// (maddubs with u = w + 128 could saturate). `c` unused.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn q8_0_lanes_avx2(b: &[u8; Q8_0_BYTES], qy: __m256i, _c: &[i32; 8]) -> __m256i {
        let qw = q8_0_weights(b);
        let dot = _mm256_maddubs_epi16(_mm256_sign_epi8(qw, qw), _mm256_sign_epi8(qy, qw));
        _mm256_madd_epi16(_mm256_set1_epi16(1), dot)
    }

    /// The single-row kernels' lanes (sign trick), for the lane-equality tests.
    #[cfg(test)]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn sign_lanes_avx2(qw: __m256i, qy: __m256i) -> __m256i {
        let dot = _mm256_maddubs_epi16(_mm256_sign_epi8(qw, qw), _mm256_sign_epi8(qy, qw));
        _mm256_madd_epi16(_mm256_set1_epi16(1), dot)
    }

    #[cfg(test)]
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub fn sign_lanes_vnni(qw: __m256i, qy: __m256i) -> __m256i {
        let (ax, sy) = (_mm256_sign_epi8(qw, qw), _mm256_sign_epi8(qy, qw));
        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy)
    }

    macro_rules! small_block_rows {
        ($feat:literal, $name:ident, $bytes:ident, $lanes:ident) => {
            /// R rows of the single-row kernel of the same type and path.
            ///
            /// # Safety
            /// The CPU must support the enabled target features.
            #[allow(clippy::needless_range_loop)] // r indexes several arrays
            #[target_feature(enable = $feat)]
            pub unsafe fn $name<const R: usize>(
                rows: &[u8],
                rb: usize,
                dx: &[f32],
                qx: &[[i8; 32]],
                c: &[[i32; 8]],
                pf: usize,
            ) -> [f32; R] {
                let n = rb / $bytes;
                let (dx, qx, c) = (&dx[..n], &qx[..n], &c[..n]);
                let mut blk: [&[[u8; $bytes]]; R] = [&[]; R];
                for r in 0..R {
                    blk[r] = &rows[r * rb..(r + 1) * rb].as_chunks::<$bytes>().0[..n];
                }
                let mut acc = [_mm256_setzero_ps(); R];
                for i in 0..n {
                    if pf != 0 {
                        for r in 0..R {
                            prefetch(rows, r * rb + i * $bytes + pf);
                        }
                    }
                    let dxb = dx[i];
                    let qy = load_i8x32(&qx[i]);
                    for r in 0..R {
                        let b = &blk[r][i];
                        let d = _mm256_set1_ps(h2f([b[0], b[1]]) * dxb);
                        let s = _mm256_cvtepi32_ps($lanes(b, qy, &c[i]));
                        acc[r] = _mm256_fmadd_ps(d, s, acc[r]);
                    }
                }
                let mut out = [0f32; R];
                for r in 0..R {
                    out[r] = hsum_f32x8(acc[r]);
                }
                out
            }
        };
    }

    small_block_rows!("avx2,fma,f16c", q5_0_rows_avx2, Q5_0_BYTES, q5_0_lanes_avx2);
    small_block_rows!("avx2,fma,f16c", q8_0_rows_avx2, Q8_0_BYTES, q8_0_lanes_avx2);
    small_block_rows!(
        "avx2,fma,f16c,avxvnni",
        q5_0_rows_vnni,
        Q5_0_BYTES,
        q5_0_lanes_vnni
    );
    small_block_rows!(
        "avx2,fma,f16c,avxvnni",
        q8_0_rows_vnni,
        Q8_0_BYTES,
        q8_0_lanes_vnni
    );

    /// Q4_K packed 6-bit scales/mins as 16 u16: 8 scales, then 8 mins (as in
    /// `q4_k_avx2`).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn q4_k_mins_and_scales(b: &[u8; Q4_K_BYTES]) -> __m256i {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
        let u0 = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        let u1 = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
        let u2 = u32::from_le_bytes([b[12], b[13], b[14], b[15]]);
        let t3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
        let t2 = u1 & KMASK1;
        let t1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
        let t0 = u0 & KMASK1;
        _mm256_cvtepu8_epi16(_mm_set_epi32(t3 as i32, t2 as i32, t1 as i32, t0 as i32))
    }

    /// R rows of `q4_k_avx2`, bit for bit per row.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c (`has_avx2()`).
    #[allow(clippy::needless_range_loop)] // j, r index several arrays
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q4_k_rows_avx2<const R: usize>(
        rows: &[u8],
        rb: usize,
        xk: &[KBlock],
        pf: usize,
    ) -> [f32; R] {
        let m4 = _mm256_set1_epi8(0xF);
        let n = rb / Q4_K_BYTES;
        let xk = &xk[..n];
        let mut blk: [&[[u8; Q4_K_BYTES]]; R] = [&[]; R];
        for r in 0..R {
            blk[r] = &rows[r * rb..(r + 1) * rb].as_chunks::<Q4_K_BYTES>().0[..n];
        }
        let mut acc = [_mm256_setzero_ps(); R];
        let mut acc_m = [_mm_setzero_ps(); R];
        for i in 0..n {
            if pf != 0 {
                for r in 0..R {
                    // a block spans 3 lines; hint them all
                    let off = r * rb + i * Q4_K_BYTES + pf;
                    prefetch(rows, off);
                    prefetch(rows, off + 64);
                    prefetch(rows, off + 128);
                }
            }
            let a = &xk[i];
            let q8sums = load_i16x16(&a.bsums);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let q8 = a.qs.as_chunks::<32>().0;
            let mut d = [0f32; R];
            let mut scales = [_mm256_setzero_si256(); R];
            let mut sumi = [_mm256_setzero_si256(); R];
            for r in 0..R {
                let b = &blk[r][i];
                d[r] = a.d * h2f([b[0], b[1]]);
                let dmin = -a.d * h2f([b[2], b[3]]);
                let mins_and_scales = q4_k_mins_and_scales(b);
                let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m[r] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[r]);
                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                scales[r] = _mm256_set_m128i(sc128, sc128);
            }
            for j in 0..4 {
                let yl = load_i8x32(&q8[2 * j]);
                let yh = load_i8x32(&q8[2 * j + 1]);
                for r in 0..R {
                    let scale_l = k4_scale(scales[r], 2 * j as i32);
                    let scale_h = k4_scale(scales[r], 2 * j as i32 + 1);
                    let q4bits = load_u8x32(&blk[r][i][16..Q4_K_BYTES].as_chunks::<32>().0[j]);
                    let q4l = _mm256_and_si256(q4bits, m4);
                    let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                    let p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, yl));
                    let p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, yh));
                    sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16l, p16h));
                }
            }
            for r in 0..R {
                acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d[r]), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
            }
        }
        let mut out = [0f32; R];
        for r in 0..R {
            let mut m = acc_m[r];
            m = _mm_add_ps(m, _mm_movehl_ps(m, m));
            m = _mm_add_ss(m, _mm_movehdup_ps(m));
            out[r] = hsum_f32x8(acc[r]) + _mm_cvtss_f32(m);
        }
        out
    }

    /// R rows of `q6_k_avx2`, bit for bit per row.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c (`has_avx2()`).
    #[allow(clippy::needless_range_loop)] // j, r index several arrays
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q6_k_rows_avx2<const R: usize>(
        rows: &[u8],
        rb: usize,
        xk: &[KBlock],
        pf: usize,
    ) -> [f32; R] {
        let m3 = _mm256_set1_epi8(3);
        let m15 = _mm256_set1_epi8(15);
        let n = rb / Q6_K_BYTES;
        let xk = &xk[..n];
        let mut blk: [&[[u8; Q6_K_BYTES]]; R] = [&[]; R];
        for r in 0..R {
            blk[r] = &rows[r * rb..(r + 1) * rb].as_chunks::<Q6_K_BYTES>().0[..n];
        }
        let mut acc = [_mm256_setzero_ps(); R];
        for i in 0..n {
            if pf != 0 {
                for r in 0..R {
                    // a block spans 4 lines; hint them all
                    let off = r * rb + i * Q6_K_BYTES + pf;
                    prefetch(rows, off);
                    prefetch(rows, off + 64);
                    prefetch(rows, off + 128);
                    prefetch(rows, off + 192);
                }
            }
            let a = &xk[i];
            let q8 = a.qs.as_chunks::<32>().0;
            let q8sums = load_i16x16(&a.bsums);
            let mut d = [0f32; R];
            let mut scales = [_mm_setzero_si128(); R];
            let mut sumi = [_mm256_setzero_si256(); R];
            let mut q8sclsub = [_mm256_setzero_si256(); R];
            for r in 0..R {
                let b = &blk[r][i];
                d[r] = a.d * h2f([b[208], b[209]]);
                scales[r] = load_u8x16(b[192..208].try_into().unwrap());
                q8sclsub[r] = _mm256_slli_epi32(
                    _mm256_madd_epi16(q8sums, _mm256_cvtepi8_epi16(scales[r])),
                    5,
                );
            }
            for j in 0..2 {
                let y0 = load_i8x32(&q8[4 * j]);
                let y1 = load_i8x32(&q8[4 * j + 1]);
                let y2 = load_i8x32(&q8[4 * j + 2]);
                let y3 = load_i8x32(&q8[4 * j + 3]);
                for r in 0..R {
                    let b = &blk[r][i];
                    let ql = b[0..128].as_chunks::<32>().0;
                    let qh = b[128..192].as_chunks::<32>().0;
                    let q4bits1 = load_u8x32(&ql[2 * j]);
                    let q4bits2 = load_u8x32(&ql[2 * j + 1]);
                    let q4bits_h = load_u8x32(&qh[j]);

                    let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
                    let q4h_1 =
                        _mm256_slli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(12)), 2);
                    let q4h_2 = _mm256_and_si256(q4bits_h, _mm256_set1_epi8(48));
                    let q4h_3 =
                        _mm256_srli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(-64)), 2);

                    let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                    let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                    let q4_2 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15),
                        q4h_2,
                    );
                    let q4_3 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15),
                        q4h_3,
                    );

                    let p16_0 = _mm256_maddubs_epi16(q4_0, y0);
                    let p16_1 = _mm256_maddubs_epi16(q4_1, y1);
                    let p16_2 = _mm256_maddubs_epi16(q4_2, y2);
                    let p16_3 = _mm256_maddubs_epi16(q4_3, y3);

                    let p16_0 = _mm256_madd_epi16(q6_scale(scales[r], 4 * j), p16_0);
                    let p16_1 = _mm256_madd_epi16(q6_scale(scales[r], 4 * j + 1), p16_1);
                    let p16_2 = _mm256_madd_epi16(q6_scale(scales[r], 4 * j + 2), p16_2);
                    let p16_3 = _mm256_madd_epi16(q6_scale(scales[r], 4 * j + 3), p16_3);

                    sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_0, p16_1));
                    sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_2, p16_3));
                }
            }
            for r in 0..R {
                let s = _mm256_sub_epi32(sumi[r], q8sclsub[r]);
                acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d[r]), _mm256_cvtepi32_ps(s), acc[r]);
            }
        }
        let mut out = [0f32; R];
        for r in 0..R {
            out[r] = hsum_f32x8(acc[r]);
        }
        out
    }
}

/// Same meaning as dot_row(qtype_id, row, x) but with x pre-quantized.
/// Decode matvecs use `dot_rows_q8`; kept as the single-row reference.
#[cfg_attr(not(test), allow(dead_code))]
pub fn dot_row_q8(qtype_id: u32, row: &[u8], x: &QuantizedAct) -> f32 {
    // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
    unsafe { dot_row_q8_with(simd(), qtype_id, row, x) }
}

/// dot_row_q8 on an explicit kernel path.
///
/// # Safety
/// `simd` must be Scalar or a path whose CPU features were detected
/// (`has_avx2()` for Avx2, `has_vnni()` for Vnni).
unsafe fn dot_row_q8_with(simd: Simd, qtype_id: u32, row: &[u8], x: &QuantizedAct) -> f32 {
    let n = x.len();
    let name = || crate::gguf::qtype(qtype_id).name;
    match qtype_id {
        F32 | F16 => {
            assert_eq!(
                x.x.len(),
                n,
                "dot_row_q8: {} input wasn't built for float weights",
                name()
            );
            dot_row(qtype_id, row, &x.x)
        }
        Q4_0 | Q5_0 | Q8_0 => {
            let B32 { d: dx, q: qx, .. } = x.b32.as_ref().unwrap_or_else(|| {
                panic!(
                    "dot_row_q8: {} needs input len multiple of 32, got {} (or the 32-block view wasn't built)",
                    name(),
                    n
                )
            });
            let row = &row[..row_bytes(qtype_id, n)];
            // SAFETY (unsafe calls below): guaranteed by our caller.
            match (qtype_id, simd) {
                #[cfg(target_arch = "x86_64")]
                (Q4_0, Simd::Avx2) => unsafe { x86::q4_0_avx2(row, dx, qx) },
                #[cfg(target_arch = "x86_64")]
                (Q4_0, Simd::Vnni) => unsafe { x86::q4_0_vnni(row, dx, qx) },
                #[cfg(target_arch = "x86_64")]
                (Q8_0, Simd::Avx2) => unsafe { x86::q8_0_avx2(row, dx, qx) },
                #[cfg(target_arch = "x86_64")]
                (Q8_0, Simd::Vnni) => unsafe { x86::q8_0_vnni(row, dx, qx) },
                #[cfg(target_arch = "x86_64")]
                (_, Simd::Avx2) => unsafe { x86::q5_0_avx2(row, dx, qx) },
                #[cfg(target_arch = "x86_64")]
                (_, Simd::Vnni) => unsafe { x86::q5_0_vnni(row, dx, qx) },
                (Q4_0, Simd::Scalar) => dot_q4_0_q8(row, dx, qx),
                (Q8_0, Simd::Scalar) => dot_q8_0_q8(row, dx, qx),
                (_, Simd::Scalar) => dot_q5_0_q8(row, dx, qx),
            }
        }
        Q4_K | Q5_K | Q6_K => {
            let xk = x.k.as_deref().unwrap_or_else(|| {
                panic!(
                    "dot_row_q8: {} needs input len multiple of 256, got {} (or the 256-block view wasn't built)",
                    name(),
                    n
                )
            });
            let row = &row[..row_bytes(qtype_id, n)];
            // Single token: the vnni path uses the AVX2 K-quant kernels (the
            // batched row-pair path uses dpwssd: dot_rows_kquant).
            match (qtype_id, simd) {
                #[cfg(target_arch = "x86_64")]
                (Q4_K, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q4_k_avx2(row, xk) },
                #[cfg(target_arch = "x86_64")]
                (Q5_K, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q5_k_avx2(row, xk) },
                #[cfg(target_arch = "x86_64")]
                (_, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q6_k_avx2(row, xk) },
                (Q4_K, Simd::Scalar) => dot_q4_k_q8(row, xk),
                (Q5_K, Simd::Scalar) => dot_q5_k_q8(row, xk),
                (_, Simd::Scalar) => dot_q6_k_q8(row, xk),
            }
        }
        MXFP4 => {
            let B32 { d: dx, q: qx, .. } = x.b32.as_ref().unwrap_or_else(|| {
                panic!(
                    "dot_row_q8: {} needs input len multiple of 32, got {} (or the 32-block view wasn't built)",
                    name(),
                    n
                )
            });
            let row = &row[..row_bytes(qtype_id, n)];
            let rb = row.len();
            // SAFETY (unsafe calls below): guaranteed by our caller.
            match simd {
                #[cfg(target_arch = "x86_64")]
                Simd::Avx2 => unsafe { mxfp4::x86::avx2::rows::<1>(row, rb, dx, qx, 0)[0] },
                #[cfg(target_arch = "x86_64")]
                Simd::Vnni => unsafe { mxfp4::x86::vnni::rows::<1>(row, rb, dx, qx, 0)[0] },
                Simd::Scalar => mxfp4::dot_mxfp4_q8(row, dx, qx),
            }
        }
        IQ4_NL => {
            let B32 { d: dx, q: qx, .. } = x.b32.as_ref().unwrap_or_else(|| {
                panic!(
                    "dot_row_q8: {} needs input len multiple of 32, got {} (or the 32-block view wasn't built)",
                    name(),
                    n
                )
            });
            let row = &row[..row_bytes(qtype_id, n)];
            match simd {
                // SAFETY: guaranteed by our caller.
                #[cfg(target_arch = "x86_64")]
                Simd::Avx2 | Simd::Vnni => unsafe { iq4nl::x86::iq4_nl_avx2(row, dx, qx) },
                Simd::Scalar => iq4nl::dot_iq4_nl_q8(row, dx, qx),
            }
        }
        Q2_K | Q3_K => {
            let xk = x.k.as_deref().unwrap_or_else(|| {
                panic!(
                    "dot_row_q8: {} needs input len multiple of 256, got {} (or the 256-block view wasn't built)",
                    name(),
                    n
                )
            });
            let row = &row[..row_bytes(qtype_id, n)];
            match (qtype_id, simd) {
                // SAFETY: guaranteed by our caller.
                #[cfg(target_arch = "x86_64")]
                (Q2_K, Simd::Avx2 | Simd::Vnni) => unsafe { k23::x86::q2_k_avx2(row, xk) },
                #[cfg(target_arch = "x86_64")]
                (_, Simd::Avx2 | Simd::Vnni) => unsafe { k23::x86::q3_k_avx2(row, xk) },
                (Q2_K, Simd::Scalar) => k23::dot_q2_k_q8(row, xk),
                (_, Simd::Scalar) => k23::dot_q3_k_q8(row, xk),
            }
        }
        Q4_1 | Q5_1 => {
            let b = x.b32.as_ref().unwrap_or_else(|| {
                panic!(
                    "dot_row_q8: {} needs input len multiple of 32, got {} (or the 32-block view wasn't built)",
                    name(),
                    n
                )
            });
            let row = &row[..row_bytes(qtype_id, n)];
            match simd {
                // SAFETY: guaranteed by our caller.
                #[cfg(target_arch = "x86_64")]
                Simd::Avx2 | Simd::Vnni => unsafe { x86::q_1_avx2(qtype_id == Q5_1, row, b) },
                Simd::Scalar => dot_q_1_q8(qtype_id, row, b),
            }
        }
        _ => panic!("dot_row_q8: unsupported qtype {} ({})", qtype_id, name()),
    }
}

/// Default decode prefetch distance in bytes (the 1-thread optimum measured
/// on the 0.5B model: 512 B 42.6 tok/s, 8 KiB 57.6, 32 KiB 53.7).
const DEFAULT_PREFETCH: usize = 8192;

/// Decode kernel settings, read from the environment once:
/// `AXON_PREFETCH` = prefetch distance in bytes ahead of the current block
/// (0 = off, capped at 1 MiB so the address arithmetic can't wrap), `AXON_DECODE_ROWS` = most rows per kernel call (1, 2 or 4).
#[derive(Clone, Copy, Debug)]
pub struct DecodeCfg {
    pub prefetch: usize,
    pub max_rows: usize,
}

impl DecodeCfg {
    pub fn get() -> DecodeCfg {
        static CFG: std::sync::OnceLock<DecodeCfg> = std::sync::OnceLock::new();
        *CFG.get_or_init(|| {
            let var = |k: &str, d: usize| match std::env::var(k) {
                Ok(s) if !s.trim().is_empty() => s
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("{k}={s}: expected a number")),
                _ => d,
            };
            let max_rows = var("AXON_DECODE_ROWS", 4);
            assert!(
                matches!(max_rows, 1 | 2 | 4),
                "AXON_DECODE_ROWS={max_rows}: expected 1, 2 or 4"
            );
            DecodeCfg {
                prefetch: var("AXON_PREFETCH", DEFAULT_PREFETCH).min(1 << 20),
                max_rows,
            }
        })
    }
}

/// Decode matvec rows: `out[r] = dot_row_q8(qtype_id, &rows[r * rb..(r + 1) * rb], x)`
/// for every r, bit for bit, with `rb = row_bytes(qtype_id, x.len())`. `rows`
/// may extend past the last row; the kernels only prefetch from there, and
/// never outside `rows`, so callers bound it to the rows they own. Q5_0/Q8_0
/// rows go up to 4 at a time, Q4_K/Q6_K 2 (at most `cfg.max_rows`), each
/// with its own accumulator chain; the remainder goes in smaller groups.
pub fn dot_rows_q8(qtype_id: u32, rows: &[u8], x: &QuantizedAct, out: &mut [f32], cfg: DecodeCfg) {
    #[cfg(target_arch = "x86_64")]
    {
        let (simd, pf, mr) = (simd(), cfg.prefetch, cfg.max_rows);
        if simd != Simd::Scalar {
            // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
            return unsafe { dot_rows_q8_v3(simd, qtype_id, rows, x, out, pf, mr) };
        }
    }
    let _ = cfg;
    let rb = row_bytes(qtype_id, x.len());
    assert!(
        rows.len() >= out.len() * rb,
        "dot_rows_q8: {} rows of {rb} bytes need {} bytes, got {}",
        out.len(),
        out.len() * rb,
        rows.len()
    );
    dot_rows_scalar(qtype_id, rb, rows, x, out);
}

/// The scalar path of the row loops: `dot_row_q8` per row. The dispatchers
/// call this instead of the `_with` glue, so the glue (and the SIMD kernels it
/// calls) is only reached from the `_v3` copies, where the kernels have a
/// single call site and inline like they did under a global target-cpu.
fn dot_rows_scalar(qtype_id: u32, rb: usize, rows: &[u8], x: &QuantizedAct, out: &mut [f32]) {
    for (r, o) in out.iter_mut().enumerate() {
        // SAFETY: Scalar needs nothing.
        *o = unsafe { dot_row_q8_with(Simd::Scalar, qtype_id, &rows[r * rb..(r + 1) * rb], x) };
    }
}

/// `dot_row_q8_multi` on the scalar path (see `dot_rows_scalar`).
fn dot_row_multi_scalar(qtype_id: u32, row: &[u8], xs: &[&QuantizedAct], out: &mut [f32]) {
    assert_eq!(
        xs.len(),
        out.len(),
        "dot_row_q8_multi: xs/out lengths differ"
    );
    for (o, x) in out.iter_mut().zip(xs) {
        // SAFETY: Scalar needs nothing.
        *o = unsafe { dot_row_q8_with(Simd::Scalar, qtype_id, row, x) };
    }
}

/// `dot_rows_q8_with` compiled for the kernels' features, so they inline
/// into the row loop (the glue functions are `#[inline(always)]` and
/// closure-free for this).
///
/// # Safety
/// As for `dot_row_q8_with`, and the CPU has avx2, fma and f16c.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_rows_q8_v3(
    simd: Simd,
    qtype_id: u32,
    rows: &[u8],
    x: &QuantizedAct,
    out: &mut [f32],
    pf: usize,
    max_rows: usize,
) {
    // SAFETY: our caller's contract.
    unsafe { dot_rows_q8_with(simd, qtype_id, rows, x, out, pf, max_rows) }
}

/// dot_rows_q8 on an explicit kernel path and prefetch distance.
///
/// # Safety
/// As for `dot_row_q8_with`.
#[inline(always)]
unsafe fn dot_rows_q8_with(
    simd: Simd,
    qtype_id: u32,
    rows: &[u8],
    x: &QuantizedAct,
    out: &mut [f32],
    pf: usize,
    max_rows: usize,
) {
    let nr = out.len();
    if nr == 0 {
        return;
    }
    let n = x.len();
    let rb = row_bytes(qtype_id, n);
    assert!(
        rows.len() >= nr * rb,
        "dot_rows_q8: {} rows of {rb} bytes need {} bytes, got {}",
        nr,
        nr * rb,
        rows.len()
    );
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (pf, max_rows);
    #[cfg(target_arch = "x86_64")]
    if simd != Simd::Scalar {
        let mut done = 0;
        match qtype_id {
            Q5_0 | Q8_0 if x.b32.is_some() => {
                let b = x.b32.as_ref().expect("checked");
                while max_rows >= 4 && nr - done >= 4 {
                    let o: &mut [f32; 4] = (&mut out[done..done + 4]).try_into().unwrap();
                    // SAFETY: guaranteed by our caller.
                    *o = unsafe { small_rows::<4>(simd, qtype_id, &rows[done * rb..], rb, b, pf) };
                    done += 4;
                }
                while max_rows >= 2 && nr - done >= 2 {
                    let o: &mut [f32; 2] = (&mut out[done..done + 2]).try_into().unwrap();
                    // SAFETY: as above.
                    *o = unsafe { small_rows::<2>(simd, qtype_id, &rows[done * rb..], rb, b, pf) };
                    done += 2;
                }
                while nr - done >= 1 {
                    // SAFETY: as above.
                    let [v] =
                        unsafe { small_rows::<1>(simd, qtype_id, &rows[done * rb..], rb, b, pf) };
                    out[done] = v;
                    done += 1;
                }
                return;
            }
            // SAFETY (the three below): guaranteed by our caller.
            Q2_K if x.k.is_some() => {
                let xk = x.k.as_deref().expect("checked");
                return unsafe { k23::x86::q2_k_rows(rows, rb, xk, out, pf) };
            }
            Q3_K if x.k.is_some() => {
                let xk = x.k.as_deref().expect("checked");
                return unsafe { k23::x86::q3_k_rows(rows, rb, xk, out, pf) };
            }
            IQ4_NL if x.b32.is_some() => {
                let b = x.b32.as_ref().expect("checked");
                return unsafe { iq4nl::x86::iq4_nl_rows(rows, rb, &b.d, &b.q, out, pf) };
            }
            MXFP4 if x.b32.is_some() => {
                let b = x.b32.as_ref().expect("checked");
                let (dx, qx) = (&b.d[..], &b.q[..]);
                // SAFETY (all four): guaranteed by our caller.
                let rows4 = |r: &[u8]| unsafe {
                    match simd {
                        Simd::Vnni => mxfp4::x86::vnni::rows::<4>(r, rb, dx, qx, pf),
                        _ => mxfp4::x86::avx2::rows::<4>(r, rb, dx, qx, pf),
                    }
                };
                let rows1 = |r: &[u8]| unsafe {
                    match simd {
                        Simd::Vnni => mxfp4::x86::vnni::rows::<1>(r, rb, dx, qx, pf),
                        _ => mxfp4::x86::avx2::rows::<1>(r, rb, dx, qx, pf),
                    }
                };
                while max_rows >= 4 && nr - done >= 4 {
                    let o = rows4(&rows[done * rb..]);
                    out[done..done + 4].copy_from_slice(&o);
                    done += 4;
                }
                while nr - done >= 1 {
                    out[done] = rows1(&rows[done * rb..])[0];
                    done += 1;
                }
                return;
            }
            Q4_K | Q6_K if x.k.is_some() => {
                let xk = x.k.as_deref().expect("checked");
                while max_rows >= 2 && nr - done >= 2 {
                    let o: &mut [f32; 2] = (&mut out[done..done + 2]).try_into().unwrap();
                    // SAFETY: as above.
                    *o = unsafe { kquant_rows::<2>(qtype_id, &rows[done * rb..], rb, xk, pf) };
                    done += 2;
                }
                while nr - done >= 1 {
                    // SAFETY: as above.
                    let [v] = unsafe { kquant_rows::<1>(qtype_id, &rows[done * rb..], rb, xk, pf) };
                    out[done] = v;
                    done += 1;
                }
                return;
            }
            _ => {}
        }
    }
    // Scalar, float types, or a missing view (dot_row_q8_with reports it).
    for (r, o) in out.iter_mut().enumerate() {
        // SAFETY: guaranteed by our caller.
        *o = unsafe { dot_row_q8_with(simd, qtype_id, &rows[r * rb..(r + 1) * rb], x) };
    }
}

/// R Q5_0/Q8_0 rows on the Avx2 or Vnni path.
///
/// # Safety
/// As for `dot_row_q8_with`, and `simd` is not Scalar.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn small_rows<const R: usize>(
    simd: Simd,
    qtype_id: u32,
    rows: &[u8],
    rb: usize,
    b: &B32,
    pf: usize,
) -> [f32; R] {
    let (dx, qx, c) = (&b.d[..], &b.q[..], &b.c[..]);
    // SAFETY (unsafe calls below): guaranteed by our caller.
    match (qtype_id, simd) {
        (Q8_0, Simd::Vnni) => unsafe { x86::q8_0_rows_vnni::<R>(rows, rb, dx, qx, c, pf) },
        (Q8_0, _) => unsafe { x86::q8_0_rows_avx2::<R>(rows, rb, dx, qx, c, pf) },
        (_, Simd::Vnni) => unsafe { x86::q5_0_rows_vnni::<R>(rows, rb, dx, qx, c, pf) },
        (_, _) => unsafe { x86::q5_0_rows_avx2::<R>(rows, rb, dx, qx, c, pf) },
    }
}

/// R Q4_K/Q6_K rows; both SIMD paths use the AVX2 kernels, as in
/// `dot_row_q8_with`.
///
/// # Safety
/// The CPU must support avx2, fma and f16c (`has_avx2()`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn kquant_rows<const R: usize>(
    qtype_id: u32,
    rows: &[u8],
    rb: usize,
    xk: &[KBlock],
    pf: usize,
) -> [f32; R] {
    // SAFETY (unsafe calls below): guaranteed by our caller.
    if qtype_id == Q4_K {
        unsafe { x86::q4_k_rows_avx2::<R>(rows, rb, xk, pf) }
    } else {
        unsafe { x86::q6_k_rows_avx2::<R>(rows, rb, xk, pf) }
    }
}

/// How many token slots to give `dot_row_q8_multi` / `dot_rows_q8_multi`
/// for `nb` tokens of length `n` against weights of types `qtypes`: `nb`
/// rounded up to a multiple of 4 when every type has a 4-token SIMD kernel,
/// else `nb`. The batched kernels take tokens 4 at a time and run any
/// leftover ones through the single-token kernel, a whole extra pass over
/// the row per leftover token (for 2 or 3 tokens, no batching at all).
/// Callers fill the extra slots by repeating a token's `QuantizedAct` (a
/// reference, no copy) and ignore their outputs: every token's result is
/// computed on its own lanes, so the real ones are unchanged, bit for bit.
/// A lone token isn't padded: 4x the arithmetic for the same one row read.
pub fn batch_slots(qtypes: &[u32], n: usize, nb: usize) -> usize {
    let grouped = qtypes.iter().all(|&qt| match qt {
        Q4_0 | Q5_0 | Q8_0 | IQ4_NL | MXFP4 => n.is_multiple_of(QK8_0),
        Q2_K | Q3_K | Q4_K | Q5_K | Q6_K => n.is_multiple_of(QK_K),
        _ => false,
    });
    if nb >= 2 && grouped && simd() != Simd::Scalar {
        nb.next_multiple_of(4)
    } else {
        nb
    }
}

/// `out[b] = dot_row_q8(qtype_id, row, xs[b])`, bit for bit, for every b.
/// Tokens go through the SIMD kernels in groups of 4, so each weight block is
/// unpacked once per group rather than once per token.
pub fn dot_row_q8_multi(qtype_id: u32, row: &[u8], xs: &[&QuantizedAct], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let simd = simd();
        if simd != Simd::Scalar {
            // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
            return unsafe { dot_row_q8_multi_v3(simd, qtype_id, row, xs, out) };
        }
    }
    dot_row_multi_scalar(qtype_id, row, xs, out);
}

/// `dot_row_q8_multi_with` compiled for the kernels' features (see `dot_rows_q8_v3`).
///
/// # Safety
/// As for `dot_row_q8_with`, and the CPU has avx2, fma and f16c.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_row_q8_multi_v3(
    simd: Simd,
    qtype_id: u32,
    row: &[u8],
    xs: &[&QuantizedAct],
    out: &mut [f32],
) {
    // SAFETY: our caller's contract.
    unsafe { dot_row_q8_multi_with(simd, qtype_id, row, xs, out) }
}

/// dot_row_q8_multi on an explicit kernel path.
///
/// # Safety
/// As for `dot_row_q8_with`.
#[inline(always)]
unsafe fn dot_row_q8_multi_with(
    simd: Simd,
    qtype_id: u32,
    row: &[u8],
    xs: &[&QuantizedAct],
    out: &mut [f32],
) {
    assert_eq!(
        xs.len(),
        out.len(),
        "dot_row_q8_multi: xs/out lengths differ"
    );
    let Some(first) = xs.first() else { return };
    // Debug only: this runs once per weight row, and a ragged batch can't
    // cause harm (every view is sliced with bounds checks). matmul callers
    // check it once per batch.
    debug_assert!(
        xs.iter().all(|x| x.len() == first.len()),
        "dot_row_q8_multi: ragged inputs"
    );
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    for (g, o) in xs.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
        let g: [&QuantizedAct; 4] = g.try_into().unwrap();
        // SAFETY: guaranteed by our caller.
        match unsafe { dot_row_q8_x4(simd, qtype_id, row, g) } {
            Some(r) => o.copy_from_slice(&r),
            None => break,
        }
        done += 4;
    }
    for (o, x) in out[done..].iter_mut().zip(&xs[done..]) {
        // SAFETY: guaranteed by our caller.
        *o = unsafe { dot_row_q8_with(simd, qtype_id, row, x) };
    }
}

/// 4 tokens through a 4-token SIMD kernel, or None if there is none for this
/// path/type (or a view is missing: the single-token path then reports it).
///
/// # Safety
/// As for `dot_row_q8_with`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::needless_range_loop)] // t indexes xs and the view arrays
#[inline(always)]
unsafe fn dot_row_q8_x4(
    simd: Simd,
    qtype_id: u32,
    row: &[u8],
    xs: [&QuantizedAct; 4],
) -> Option<[f32; 4]> {
    let n = xs[0].len();
    match qtype_id {
        Q4_0 | Q5_0 | Q8_0 => {
            let mut dx: [&[f32]; 4] = [&[]; 4];
            let mut qx: [&[[i8; 32]]; 4] = [&[]; 4];
            for t in 0..4 {
                let b = xs[t].b32.as_ref()?;
                dx[t] = b.d.as_slice();
                qx[t] = b.q.as_slice();
            }
            let row = &row[..row_bytes(qtype_id, n)];
            // SAFETY (unsafe calls below): guaranteed by our caller.
            Some(match (qtype_id, simd) {
                (Q4_0, Simd::Avx2) => unsafe { x86::q4_0_avx2_x4(row, dx, qx) },
                (Q4_0, Simd::Vnni) => unsafe { x86::q4_0_vnni_x4(row, dx, qx) },
                (Q8_0, Simd::Avx2) => unsafe { x86::q8_0_avx2_x4(row, dx, qx) },
                (Q8_0, Simd::Vnni) => unsafe { x86::q8_0_vnni_x4(row, dx, qx) },
                (_, Simd::Avx2) => unsafe { x86::q5_0_avx2_x4(row, dx, qx) },
                (_, Simd::Vnni) => unsafe { x86::q5_0_vnni_x4(row, dx, qx) },
                (_, Simd::Scalar) => return None,
            })
        }
        Q4_K | Q5_K | Q6_K => {
            let mut xk: [&[KBlock]; 4] = [&[]; 4];
            for t in 0..4 {
                xk[t] = xs[t].k.as_deref()?;
            }
            let row = &row[..row_bytes(qtype_id, n)];
            // The vnni path uses the AVX2 K-quant kernels, as in dot_row_q8_with.
            Some(match (qtype_id, simd) {
                (Q4_K, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q4_k_avx2_x4(row, xk) },
                (Q5_K, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q5_k_avx2_x4(row, xk) },
                (_, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q6_k_avx2_x4(row, xk) },
                (_, Simd::Scalar) => return None,
            })
        }
        MXFP4 => {
            let mut dx: [&[f32]; 4] = [&[]; 4];
            let mut qx: [&[[i8; 32]]; 4] = [&[]; 4];
            for t in 0..4 {
                let b = xs[t].b32.as_ref()?;
                dx[t] = b.d.as_slice();
                qx[t] = b.q.as_slice();
            }
            let row = &row[..row_bytes(qtype_id, n)];
            // SAFETY: guaranteed by our caller.
            Some(match simd {
                Simd::Avx2 => unsafe { mxfp4::x86::avx2::x4(row, dx, qx) },
                Simd::Vnni => unsafe { mxfp4::x86::vnni::x4(row, dx, qx) },
                Simd::Scalar => return None,
            })
        }
        IQ4_NL => {
            let mut dx: [&[f32]; 4] = [&[]; 4];
            let mut qx: [&[[i8; 32]]; 4] = [&[]; 4];
            for t in 0..4 {
                let b = xs[t].b32.as_ref()?;
                dx[t] = b.d.as_slice();
                qx[t] = b.q.as_slice();
            }
            let row = &row[..row_bytes(qtype_id, n)];
            Some(match simd {
                Simd::Avx2 | Simd::Vnni => unsafe { iq4nl::x86::iq4_nl_avx2_x4(row, dx, qx) },
                Simd::Scalar => return None,
            })
        }
        Q2_K | Q3_K => {
            let mut xk: [&[KBlock]; 4] = [&[]; 4];
            for t in 0..4 {
                xk[t] = xs[t].k.as_deref()?;
            }
            let row = &row[..row_bytes(qtype_id, n)];
            Some(match (qtype_id, simd) {
                (Q2_K, Simd::Avx2 | Simd::Vnni) => unsafe { k23::x86::q2_k_avx2_x4(row, xk) },
                (_, Simd::Avx2 | Simd::Vnni) => unsafe { k23::x86::q3_k_avx2_x4(row, xk) },
                (_, Simd::Scalar) => return None,
            })
        }
        _ => None,
    }
}

/// Two rows at once: `out[k][b] = dot_row_q8(qtypes[k], rows[k], xs[b])`, bit
/// for bit. When both rows have the same small-block type (Q5_0/Q8_0) on a
/// SIMD path, tokens go 4 at a time through a 2-row x 4-token kernel (on Vnni
/// the pair is unpacked once for all groups: `dot_rows_q8_vnni`). Two Q4_K or
/// two Q6_K rows are unpacked once and go 4 tokens at a time on both SIMD
/// paths (`dot_rows_kquant`). The rest (other types, mixed pairs, < 4
/// leftover tokens) goes per row through `dot_row_q8_multi`.
pub fn dot_rows_q8_multi(
    qtypes: [u32; 2],
    rows: [&[u8]; 2],
    xs: &[&QuantizedAct],
    out: [&mut [f32]; 2],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let simd = simd();
        if simd != Simd::Scalar {
            // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
            return unsafe { dot_rows_q8_multi_v3(simd, qtypes, rows, xs, out) };
        }
    }
    let [o0, o1] = out;
    assert!(
        o0.len() == xs.len() && o1.len() == xs.len(),
        "dot_rows_q8_multi: xs/out lengths differ"
    );
    dot_row_multi_scalar(qtypes[0], rows[0], xs, o0);
    dot_row_multi_scalar(qtypes[1], rows[1], xs, o1);
}

/// `dot_rows_q8_multi_with` compiled for the kernels' features (see `dot_rows_q8_v3`).
///
/// # Safety
/// As for `dot_row_q8_with`, and the CPU has avx2, fma and f16c.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_rows_q8_multi_v3(
    simd: Simd,
    qtypes: [u32; 2],
    rows: [&[u8]; 2],
    xs: &[&QuantizedAct],
    out: [&mut [f32]; 2],
) {
    // SAFETY: our caller's contract.
    unsafe { dot_rows_q8_multi_with(simd, qtypes, rows, xs, out) }
}

/// dot_rows_q8_multi on an explicit kernel path.
///
/// # Safety
/// As for `dot_row_q8_with`.
#[inline(always)]
unsafe fn dot_rows_q8_multi_with(
    simd: Simd,
    qtypes: [u32; 2],
    rows: [&[u8]; 2],
    xs: &[&QuantizedAct],
    out: [&mut [f32]; 2],
) {
    let [o0, o1] = out;
    assert!(
        o0.len() == xs.len() && o1.len() == xs.len(),
        "dot_rows_q8_multi: xs/out lengths differ"
    );
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if qtypes[0] == qtypes[1] && matches!(qtypes[0], Q4_K | Q6_K) {
        // SAFETY: guaranteed by our caller.
        done = unsafe { dot_rows_kquant(simd, qtypes[0], rows, xs, [&mut *o0, &mut *o1]) };
    } else if qtypes[0] == qtypes[1] && simd == Simd::Vnni {
        // SAFETY: guaranteed by our caller.
        done = unsafe { dot_rows_q8_vnni(qtypes[0], rows, xs, [&mut *o0, &mut *o1]) };
    } else if qtypes[0] == qtypes[1] {
        for g in xs.chunks_exact(4) {
            let g: [&QuantizedAct; 4] = g.try_into().unwrap();
            // SAFETY: guaranteed by our caller.
            match unsafe { dot_rows_q8_2x4(simd, qtypes[0], rows, g) } {
                Some([r0, r1]) => {
                    o0[done..done + 4].copy_from_slice(&r0);
                    o1[done..done + 4].copy_from_slice(&r1);
                }
                None => break,
            }
            done += 4;
        }
    }
    // SAFETY: guaranteed by our caller.
    unsafe {
        dot_row_q8_multi_with(simd, qtypes[0], rows[0], &xs[done..], &mut o0[done..]);
        dot_row_q8_multi_with(simd, qtypes[1], rows[1], &xs[done..], &mut o1[done..]);
    }
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// Unpacked row pair for dot_rows_q8_vnni; grows to the widest row seen.
    static PAIR_SCRATCH: std::cell::RefCell<Vec<x86::Pair32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The VNNI part of dot_rows_q8_multi for two rows of type `qtype_id`:
/// unpack the pair once, then every whole group of 4 tokens. Returns how many
/// leading tokens it did (a multiple of 4; 0 if the type isn't Q5_0/Q8_0 or
/// a 32-block view is missing).
///
/// # Safety
/// The CPU must support the Vnni path (`has_vnni()`).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::needless_range_loop)] // t indexes xs and the view arrays
#[inline(always)]
unsafe fn dot_rows_q8_vnni(
    qtype_id: u32,
    rows: [&[u8]; 2],
    xs: &[&QuantizedAct],
    out: [&mut [f32]; 2],
) -> usize {
    let groups = xs.len() / 4;
    if !matches!(qtype_id, Q5_0 | Q8_0) || groups == 0 {
        return 0;
    }
    for x in &xs[..groups * 4] {
        if x.b32.is_none() {
            return 0;
        }
    }
    let n = xs[0].len();
    let nb = n / QK8_0;
    let rb = row_bytes(qtype_id, n);
    let rows = [&rows[0][..rb], &rows[1][..rb]];
    let [o0, o1] = out;
    // Taken out and put back rather than borrowed in a closure: a closure
    // wouldn't get the caller's target features, so the kernels wouldn't
    // inline (see `dot_rows_q8_v3`).
    let mut scratch = PAIR_SCRATCH.take();
    if scratch.len() < nb {
        scratch.resize(nb, x86::Pair32::ZERO);
    }
    let w = &mut scratch[..nb];
    // SAFETY (unsafe calls below): has_vnni() implies avx2, fma and f16c.
    if qtype_id == Q8_0 {
        unsafe { x86::unpack_q8_0_pair(rows, w) };
    } else {
        unsafe { x86::unpack_q5_0_pair(rows, w) };
    }
    let w = &*w;
    for (g, x4) in xs.chunks_exact(4).enumerate() {
        let mut dx: [&[f32]; 4] = [&[]; 4];
        let mut qx: [&[[i8; 32]]; 4] = [&[]; 4];
        let mut c: [&[[i32; 8]]; 4] = [&[]; 4];
        for t in 0..4 {
            let b = x4[t].b32.as_ref().expect("checked above");
            dx[t] = b.d.as_slice();
            qx[t] = b.q.as_slice();
            c[t] = b.c.as_slice();
        }
        let [r0, r1] = if qtype_id == Q8_0 {
            unsafe { x86::vnni_pair_x4::<true>(w, dx, qx, c) }
        } else {
            unsafe { x86::vnni_pair_x4::<false>(w, dx, qx, c) }
        };
        o0[4 * g..4 * g + 4].copy_from_slice(&r0);
        o1[4 * g..4 * g + 4].copy_from_slice(&r1);
    }
    PAIR_SCRATCH.set(scratch);
    groups * 4
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// Per-token accumulators for dot_rows_kquant; grows to the largest batch seen.
    static KACC_SCRATCH: std::cell::RefCell<Vec<x86::KAcc>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The SIMD part of dot_rows_q8_multi for two Q4_K or two Q6_K rows: every
/// whole group of 4 tokens, each super-block of the pair unpacked once.
/// Returns how many leading tokens it did (a multiple of 4; 0 on Scalar, for
/// other types, or if a 256-block view is missing).
///
/// # Safety
/// As for `dot_row_q8_with`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_rows_kquant(
    simd: Simd,
    qtype_id: u32,
    rows: [&[u8]; 2],
    xs: &[&QuantizedAct],
    out: [&mut [f32]; 2],
) -> usize {
    let nt = xs.len() / 4 * 4;
    if !matches!(qtype_id, Q4_K | Q6_K) || simd == Simd::Scalar || nt == 0 {
        return 0;
    }
    let mut xk: Vec<&[KBlock]> = Vec::with_capacity(nt);
    for x in &xs[..nt] {
        match x.k.as_deref() {
            Some(k) => xk.push(k),
            None => return 0,
        }
    }
    let n = xs[0].len();
    let rb = row_bytes(qtype_id, n);
    let rows = [&rows[0][..rb], &rows[1][..rb]];
    let [o0, o1] = out;
    let out = [&mut o0[..nt], &mut o1[..nt]];
    // Taken out and put back, not borrowed in a closure (see dot_rows_q8_vnni).
    let mut acc = KACC_SCRATCH.take();
    if acc.len() < nt {
        acc.resize(nt, x86::KAcc::ZERO);
    }
    let a = &mut acc[..];
    // SAFETY (unsafe calls below): guaranteed by our caller; Avx2 and
    // Vnni both imply avx2, fma and f16c (`has_avx2()`), Vnni also avxvnni.
    match (qtype_id, simd) {
        (Q4_K, Simd::Vnni) => unsafe { x86::q4_k_pair_vnni(rows, &xk, a, out) },
        (Q4_K, _) => unsafe { x86::q4_k_pair_avx2(rows, &xk, a, out) },
        (_, Simd::Vnni) => unsafe { x86::q6_k_pair_vnni(rows, &xk, a, out) },
        (_, _) => unsafe { x86::q6_k_pair_avx2(rows, &xk, a, out) },
    }
    KACC_SCRATCH.set(acc);
    nt
}

/// 2 rows of type `qtype_id` x 4 tokens through a 2x4 SIMD kernel, or None if
/// there is none for this path/type (or a view is missing).
///
/// # Safety
/// As for `dot_row_q8_with`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::needless_range_loop)] // t indexes xs and the view arrays
#[inline(always)]
unsafe fn dot_rows_q8_2x4(
    simd: Simd,
    qtype_id: u32,
    rows: [&[u8]; 2],
    xs: [&QuantizedAct; 4],
) -> Option<[[f32; 4]; 2]> {
    if !matches!(qtype_id, Q5_0 | Q8_0) || simd == Simd::Scalar {
        return None;
    }
    let n = xs[0].len();
    let mut dx: [&[f32]; 4] = [&[]; 4];
    let mut qx: [&[[i8; 32]]; 4] = [&[]; 4];
    for t in 0..4 {
        let b = xs[t].b32.as_ref()?;
        dx[t] = b.d.as_slice();
        qx[t] = b.q.as_slice();
    }
    let rb = row_bytes(qtype_id, n);
    let rows = [&rows[0][..rb], &rows[1][..rb]];
    // SAFETY (unsafe calls below): guaranteed by our caller.
    Some(match (qtype_id, simd) {
        (Q8_0, Simd::Avx2) => unsafe { x86::q8_0_avx2_2x4(rows, dx, qx) },
        (_, Simd::Avx2) => unsafe { x86::q5_0_avx2_2x4(rows, dx, qx) },
        // Vnni has its own unpack-once path (dot_rows_q8_vnni).
        (_, Simd::Vnni | Simd::Scalar) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_q8_k_matches_reference_loop() {
        fn reference(x: &[f32], q: &mut [i8]) -> f32 {
            let (mut max, mut amax) = (0f32, 0f32);
            for &v in x {
                if v.abs() > amax {
                    amax = v.abs();
                    max = v;
                }
            }
            if amax == 0.0 {
                q.fill(0);
                return 0.0;
            }
            let iscale = -127.0 / max;
            for (qi, &xi) in q.iter_mut().zip(x) {
                *qi = round_even_small(iscale * xi).min(127) as i8;
            }
            1.0 / iscale
        }
        let mut rng = 0x9e37_79b9u32;
        let mut next = || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let mut cases: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..256).map(|_| next() * 8.0).collect())
            .collect();
        cases.push(vec![0.0; 256]);
        // ties between +m and -m: the first one wins, and decides the sign
        let mut tie = vec![0.25f32; 256];
        tie[17] = -3.0;
        tie[200] = 3.0;
        cases.push(tie.clone());
        tie[17] = 3.0;
        tie[200] = -3.0;
        cases.push(tie);
        for x in &cases {
            let (mut a, mut b) = ([0i8; 256], [0i8; 256]);
            assert_eq!(
                quantize_q8_k(x, &mut a).to_bits(),
                reference(x, &mut b).to_bits()
            );
            assert_eq!(a, b);
        }
    }

    #[test]
    fn round_even_small_matches_round_ties_even() {
        let mut v = -200.0f32;
        while v <= 200.0 {
            for x in [v, v + 0.5, v + 0.25, v - 0.5, v * 1.000_001] {
                assert_eq!(round_even_small(x), x.round_ties_even() as i32, "x={x}");
            }
            v += 0.125;
        }
        let mut rng = 0x1234_5678u32;
        for _ in 0..200_000 {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = (rng as f32 / u32::MAX as f32 - 0.5) * 300.0;
            assert_eq!(round_even_small(x), x.round_ties_even() as i32, "x={x}");
        }
    }

    #[test]
    fn for_types_builds_only_needed_views() {
        let x: Vec<f32> = (0..512).map(|i| (i as f32 * 0.37).sin()).collect();
        let k_only = QuantizedAct::for_types(&x, &[Q6_K]);
        assert!(k_only.k.is_some() && k_only.b32.is_none() && k_only.x.is_empty());
        let b_only = QuantizedAct::for_types(&x, &[Q5_0, Q8_0]);
        assert!(b_only.b32.is_some() && b_only.k.is_none() && b_only.x.is_empty());
        // Same numbers as the all-views constructor.
        let all = QuantizedAct::new(&x);
        let mut row = vec![0u8; row_bytes(Q8_0, 512)];
        for (i, b) in row.iter_mut().enumerate() {
            *b = (i * 7) as u8;
        }
        for blk in row.chunks_mut(Q8_0_BYTES) {
            blk[..2].copy_from_slice(&0x2c00u16.to_le_bytes());
        }
        assert_eq!(
            dot_row_q8(Q8_0, &row, &b_only),
            dot_row_q8(Q8_0, &row, &all)
        );
    }

    #[test]
    fn f32_to_f16_roundtrip_and_rounding() {
        for h in 0u16..=0xffff {
            let f = f16_to_f32(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "h={h:#06x} f={f}");
        }
        // Exactly halfway between 1.0 (0x3c00) and the next half: ties go to even.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // rounds up to infinity
        assert_eq!(f32_to_f16(1e-9), 0);
    }

    #[test]
    fn f16_edges() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        let z = f16_to_f32(0x0000);
        assert_eq!(z, 0.0);
        assert!(z.is_sign_positive());
        let nz = f16_to_f32(0x8000);
        assert_eq!(nz, 0.0);
        assert!(nz.is_sign_negative());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert!(f16_to_f32(0xfc01).is_nan());
    }

    #[test]
    fn q8_0_block() {
        let mut b = vec![0u8; Q8_0_BYTES];
        b[0..2].copy_from_slice(&0xc000u16.to_le_bytes()); // d = -2.0
        let qs: [i8; 32] = [
            -128, 127, 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10,
            50, -50, 100, -100, 64, -64, 126, -127, 33,
        ];
        for j in 0..32 {
            b[2 + j] = qs[j] as u8;
        }
        let mut out = [0f32; 32];
        dequant(Q8_0, &b, &mut out);
        for j in 0..32 {
            assert_eq!(out[j], qs[j] as f32 * -2.0, "j={}", j);
        }
        assert_eq!(out[0], 256.0);
        assert_eq!(out[1], -254.0);
        assert_eq!(row_bytes(Q8_0, 896), 28 * 34);
    }

    #[test]
    fn q5_0_block() {
        let mut b = vec![0u8; Q5_0_BYTES];
        b[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
        // qh bit 0 -> high bit of element 0, bit 31 -> high bit of element 31
        b[2..6].copy_from_slice(&0x8000_0001u32.to_le_bytes());
        for j in 0..16 {
            b[6 + j] = (j as u8 & 0xf) | (((15 - j) as u8 & 0xf) << 4);
        }
        let mut out = [0f32; 32];
        dequant(Q5_0, &b, &mut out);
        let expect: [f32; 32] = [
            0.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5,
            -1.0, -0.5, // elements 0..16: low nibbles j (0 has hi bit -> 16)
            -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0,
            -7.5, 0.0, // elements 16..32: high nibbles 15-j (31 has hi bit -> 16)
        ];
        assert_eq!(out, expect);
    }

    #[test]
    fn q4_k_block() {
        let mut b = vec![0u8; Q4_K_BYTES];
        b[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        b[2..4].copy_from_slice(&0x3c00u16.to_le_bytes()); // dmin = 1
        let sc: [u8; 12] = [0x41, 2, 3, 4, 0xc5, 6, 7, 8, 0x9a, 0, 0, 0];
        b[4..16].copy_from_slice(&sc);
        assert_eq!(get_scale_min_k4(0, &sc), (1, 5));
        assert_eq!(get_scale_min_k4(3, &sc), (4, 8));
        assert_eq!(get_scale_min_k4(4, &sc), (26, 57));
        assert_eq!(get_scale_min_k4(5, &sc), (0, 0));
        for v in &mut b[16..] {
            *v = 0x21; // low nibble 1, high nibble 2
        }
        let mut out = [0f32; 256];
        dequant(Q4_K, &b, &mut out);
        let per_sub = [-4.0, -2.0, -4.0, 0.0, -31.0, 0.0, 0.0, 0.0];
        for e in 0..256 {
            assert_eq!(out[e], per_sub[e / 32], "e={}", e);
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // `e` picks the expected value too
    fn q6_k_block() {
        let mut b = vec![0u8; Q6_K_BYTES];
        let (ql, qh, sc) = (0usize, 128usize, 192usize);
        for i in 0..16 {
            b[sc + i] = (i as i8 + 1) as u8;
        }
        b[sc + 15] = (-1i8) as u8;
        b[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        b[ql] = 0x21;
        b[ql + 32] = 0x43;
        b[qh] = 0xe4; // 11 10 01 00
        b[ql + 64] = 0x0f;
        b[qh + 32] = 0x03;
        let mut out = [0f32; 256];
        dequant(Q6_K, &b, &mut out);
        let scale = |e: usize| (b[sc + e / 16] as i8) as f32;
        for e in 0..256 {
            let want = match e {
                0 => -31.0,
                32 => -39.0,
                64 => 10.0,
                96 => 140.0,
                128 => 279.0,
                _ => -32.0 * scale(e),
            };
            assert_eq!(out[e], want, "e={}", e);
        }
        assert_eq!(out[255], 32.0);
    }

    /// ggml's dequantize_row_* on deterministic random blocks (xorshift64*,
    /// every f16 scale field an edge value - zero, -0, subnormals, max, inf -
    /// in the first 22 blocks, then random magnitudes): FNV-1a of the output
    /// bits (NaNs canonical) against the hashes the C harness got from
    /// llama.cpp's libggml-base (ggml-quants.c), bit for bit.
    #[test]
    fn dequant_matches_ggml_reference_hashes() {
        const EDGE: [u16; 11] = [
            0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0x3c00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x3555,
        ];
        fn hash(qt: u32, seed: u64, bb: usize, bs: usize, offs: &[usize]) -> u64 {
            const NB: usize = 4096;
            let mut s = seed;
            let mut nx = || {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                s.wrapping_mul(0x2545_F491_4F6C_DD1D)
            };
            let mut b: Vec<u8> = (0..NB * bb).map(|_| nx() as u8).collect();
            for i in 0..NB {
                for (f, &o) in offs.iter().enumerate() {
                    let v = if i < 22 * offs.len() {
                        EDGE[(i + f * 3) % 11]
                    } else {
                        let r = nx();
                        ((r & 0x83ff) | ((((r >> 16) % 20) + 1) << 10)) as u16
                    };
                    b[i * bb + o..i * bb + o + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            let mut y = vec![0f32; NB * bs];
            dequant(qt, &b, &mut y);
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for v in y {
                let mut bits = v.to_bits();
                if v.is_nan() {
                    bits = 0x7fc0_0000;
                }
                for k in 0..4 {
                    h ^= ((bits >> (8 * k)) & 0xff) as u64;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
            }
            h
        }
        let k = 0xD1B5_4A32_D192_ED03;
        assert_eq!(
            hash(Q4_0, 0x9E37_79B9_7F4A_7C15, 18, 32, &[0]),
            0xd73c_b41a_793f_78a8
        );
        assert_eq!(hash(Q5_K, k, 176, 256, &[0, 2]), 0xc348_1e06_2f21_aeac);
        assert_eq!(
            hash(Q4_1, 0x243F_6A88_85A3_08D3, 20, 32, &[0, 2]),
            0xebfa_d763_1936_e044
        );
        assert_eq!(
            hash(Q5_1, 0x1319_8A2E_0370_7344, 24, 32, &[0, 2]),
            0x5f34_2d15_154a_6849
        );
        assert_eq!(
            hash(Q2_K, 0xA409_3822_299F_31D0, 84, 256, &[80, 82]),
            0xc1fc_a545_4ea5_59d5
        );
        assert_eq!(
            hash(Q3_K, 0x082E_FA98_EC4E_6C89, 110, 256, &[108]),
            0x3ea6_511d_8885_b60a
        );
        assert_eq!(
            hash(IQ4_NL, 0xBE54_66CF_34E9_0C6C, 18, 32, &[0]),
            0x111d_48f7_b813_57f3
        );
        // The harness's control: an existing type through the same generator.
        assert_eq!(hash(Q4_K, k, 144, 256, &[0, 2]), 0x3046_51e0_52ca_e34f);
    }

    #[test]
    fn q4_0_q5_k_blocks() {
        // Q4_0: d = 0.5, qs[j] = j | (15 - j) << 4.
        let mut b = vec![0u8; Q4_0_BYTES];
        b[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        for j in 0..16 {
            b[2 + j] = (j as u8) | ((15 - j as u8) << 4);
        }
        let mut out = [0f32; 32];
        dequant(Q4_0, &b, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], (j as f32 - 8.0) * 0.5);
            assert_eq!(out[16 + j], (7.0 - j as f32) * 0.5);
        }
        // Q5_K: d = 1, dmin = 2, sub-block scale j + 1, min j; qs = 0x5a,
        // qh bit 2c (low nibbles of chunk c) set for even l only.
        let mut b = vec![0u8; Q5_K_BYTES];
        b[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x4000u16.to_le_bytes());
        for j in 0..4 {
            b[4 + j] = j as u8 + 1; // scales 0..4
            b[8 + j] = j as u8; // mins 0..4
            b[12 + j] = ((j as u8 + 5) & 0xf) | (((j as u8 + 4) & 0xf) << 4); // 4..8 low bits
        }
        for l in 0..32 {
            b[16 + l] = if l % 2 == 0 { 0x55 } else { 0 };
        }
        b[48..].fill(0x5a);
        let mut out = [0f32; 256];
        dequant(Q5_K, &b, &mut out);
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &b[4..16]);
            assert_eq!((sc, m), (j as u8 + 1, j as u8));
            for l in 0..32 {
                let nib = if j % 2 == 0 { 0xa } else { 0x5 };
                let h = if j % 2 == 0 && l % 2 == 0 { 16 } else { 0 };
                let want = sc as f32 * (nib + h) as f32 - 2.0 * m as f32;
                assert_eq!(out[32 * j + l], want, "sub-block {j} l {l}");
            }
        }
    }

    #[test]
    fn dot_row_matches_dequant_synthetic() {
        // two Q8_0 blocks + exercise chunking with F32 of odd length
        let mut b = vec![0u8; 2 * Q8_0_BYTES];
        for blk in 0..2 {
            b[blk * 34..blk * 34 + 2].copy_from_slice(&0x3400u16.to_le_bytes()); // 0.25
            for j in 0..32 {
                b[blk * 34 + 2 + j] = ((j as i32 * 7 - 100 + blk as i32) as i8) as u8;
            }
        }
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut w = [0f32; 64];
        dequant(Q8_0, &b, &mut w);
        let naive: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((dot_row(Q8_0, &b, &x) - naive).abs() < 1e-4);

        let n = 300;
        let fb: Vec<u8> = (0..n)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        let x: Vec<f32> = (0..n).map(|i| 1.0 / (1 + i) as f32).collect();
        let naive: f32 = (0..n).map(|i| i as f32 * 0.5 * x[i]).sum();
        assert!((dot_row(F32, &fb, &x) - naive).abs() < 1e-3);
    }

    /// Every quant type in the model, checked row by row against gguf-py's
    /// dequantizer. Needs no model: the fixtures carry the raw row bytes.
    #[test]
    fn golden_rows() {
        let dir = crate::testutil::fixtures().join("golden");
        let mut stems: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter_map(|f| f.strip_suffix(".json").map(str::to_string))
            .collect();
        stems.sort();
        let mut covered = std::collections::BTreeSet::new();
        for stem in &stems {
            let meta = crate::testutil::read_json(&format!("golden/{stem}.json"));
            let qt = meta["qtype"].as_u64().expect("qtype") as u32;
            let row_len = meta["row_len"].as_u64().expect("row_len") as usize;
            let bytes = std::fs::read(dir.join(format!("{stem}.raw"))).expect("raw");
            let want = std::fs::read(dir.join(format!("{stem}.f32"))).expect("f32");
            assert_eq!(bytes.len(), row_bytes(qt, row_len), "{stem}: raw size");
            assert_eq!(want.len(), row_len * 4, "{stem}: f32 size");

            let mut got = vec![0f32; row_len];
            dequant(qt, &bytes, &mut got);
            for (i, (g, w)) in got.iter().zip(want.as_chunks::<4>().0).enumerate() {
                let w = f32::from_le_bytes(*w);
                assert!((g - w).abs() <= 1e-6, "{stem}: idx {i} got {g} want {w}");
            }

            // dot_row must agree with dequant + a plain dot product.
            let x: Vec<f32> = (0..row_len)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
                .collect();
            let naive: f32 = got.iter().zip(&x).map(|(a, b)| a * b).sum();
            let mag: f32 = got.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
            let d = dot_row(qt, &bytes, &x);
            assert!(
                (d - naive).abs() <= 1e-5 * mag + 1e-6,
                "{stem}: dot_row {d} vs naive {naive}"
            );
            covered.insert(meta["qtype_name"].as_str().expect("qtype_name").to_string());
        }
        let want: std::collections::BTreeSet<String> = ["F32", "Q4_K", "Q5_0", "Q6_K", "Q8_0"]
            .map(String::from)
            .into();
        assert_eq!(
            covered, want,
            "golden fixtures should cover every quant type in the model"
        );
    }

    // ---- integer dot products (dot_row_q8) ----

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

    /// Random row bytes with sane f16 scale fields.
    fn synthetic_row(qt: u32, n: usize, rng: &mut Lcg) -> Vec<u8> {
        let mut b: Vec<u8> = (0..row_bytes(qt, n))
            .map(|_| rng.next_u32() as u8)
            .collect();
        let (bs, bb) = block_info(qt);
        let d_offs: &[usize] = match qt {
            Q4_0 | Q5_0 | Q8_0 => &[0],
            Q4_K | Q5_K | Q4_1 | Q5_1 => &[0, 2],
            Q6_K => &[208],
            Q2_K => &[80, 82],
            Q3_K => &[108],
            IQ4_NL => &[0],
            F16 => &[0],
            F32 => {
                return (0..n).flat_map(|_| rng.f().to_le_bytes()).collect();
            }
            _ => unreachable!(),
        };
        for blk in 0..n / bs {
            for &o in d_offs {
                let p = blk * bb + o;
                b[p..p + 2].copy_from_slice(&rng.f16_bits().to_le_bytes());
            }
        }
        b
    }

    /// Activations with per-256 magnitude changes and one all-zero region.
    fn synthetic_x(n: usize, rng: &mut Lcg) -> Vec<f32> {
        let scales = [1.0f32, 0.01, 30.0, 0.0, 3.0];
        (0..n)
            .map(|i| rng.f() * scales[(i / 256) % scales.len()])
            .collect()
    }

    /// The activations as dot_row_q8 sees them for weight type `qt`.
    fn dequant_act(qt: u32, qa: &QuantizedAct) -> Vec<f32> {
        match qt {
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | IQ4_NL => {
                let B32 { d, q, c, n128 } = qa.b32.as_ref().unwrap();
                for ((q, c), &s) in q.iter().zip(c).zip(n128) {
                    for (lane, &cl) in q.as_chunks::<4>().0.iter().zip(c) {
                        assert_eq!(cl, -16 * lane.iter().map(|&v| v as i32).sum::<i32>());
                    }
                    assert_eq!(s, -128 * q.iter().map(|&v| v as i32).sum::<i32>());
                }
                d.iter()
                    .zip(q)
                    .flat_map(|(&d, q)| q.map(|v| d * v as f32))
                    .collect()
            }
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K => {
                let k = qa.k.as_ref().unwrap();
                for b in k {
                    for (g, s) in b.qs.as_chunks::<16>().0.iter().zip(b.bsums) {
                        assert_eq!(g.iter().map(|&v| v as i16).sum::<i16>(), s);
                    }
                }
                k.iter()
                    .flat_map(|b| b.qs.map(|v| b.d * v as f32))
                    .collect()
            }
            _ => qa.x.clone(),
        }
    }

    fn check_q8(qt: u32, row: &[u8], x: &[f32], label: &str) {
        let n = x.len();
        let qa = QuantizedAct::new(x);
        assert_eq!(qa.len(), n);
        let mut w = vec![0f32; n];
        dequant(qt, row, &mut w);
        let xq = dequant_act(qt, &qa);
        let reference: f64 = w.iter().zip(&xq).map(|(a, b)| *a as f64 * *b as f64).sum();
        let mag: f64 = w
            .iter()
            .zip(&xq)
            .map(|(a, b)| (*a as f64 * *b as f64).abs())
            .sum();
        let got = dot_row_q8(qt, row, &qa);
        assert!(
            (got as f64 - reference).abs() <= 1e-4 * mag + 1e-5,
            "{label}: dot_row_q8 {got} vs exact-on-quantized-x {reference} (mag {mag})"
        );
        let fl = dot_row(qt, row, x);
        let magf: f32 = w.iter().zip(x).map(|(a, b)| (a * b).abs()).sum();
        assert!(
            (got - fl).abs() <= 2e-2 * magf + 1e-4,
            "{label}: dot_row_q8 {got} vs float dot_row {fl} (mag {magf})"
        );
    }

    #[test]
    fn dot_row_q8_synthetic() {
        let mut rng = Lcg(0x5eed);
        for &(qt, n) in &[
            (Q5_0, 896),
            (Q8_0, 896),
            (Q5_0, 1280),
            (Q8_0, 1280),
            (Q4_K, 1280),
            (Q6_K, 1280),
            (Q4_K, 4864),
            (Q6_K, 4864),
            (Q4_0, 896),
            (Q4_0, 4864),
            (Q5_K, 1280),
            (Q5_K, 4864),
            (IQ4_NL, 896),
            (IQ4_NL, 4864),
            (Q2_K, 1280),
            (Q2_K, 4864),
            (Q3_K, 1280),
            (Q3_K, 4864),
            (Q4_1, 896),
            (Q5_1, 896),
            (Q5_1, 4864),
            (F32, 896),
            (F16, 896),
        ] {
            for rep in 0..8 {
                let row = synthetic_row(qt, n, &mut rng);
                let x = synthetic_x(n, &mut rng);
                check_q8(qt, &row, &x, &format!("qt {qt} n {n} rep {rep}"));
            }
        }
    }

    /// Random scales, every other byte set to `fill`.
    fn extreme_row(qt: u32, n: usize, fill: u8, rng: &mut Lcg) -> Vec<u8> {
        let mut row = synthetic_row(qt, n, rng);
        let (bs, bb) = block_info(qt);
        // byte range holding the f16 scale(s), left random
        let keep = match qt {
            Q4_0 | Q5_0 | Q8_0 => 0..2,
            Q4_K | Q5_K | Q4_1 | Q5_1 => 0..4,
            Q6_K => 208..210,
            Q2_K => 80..84,
            Q3_K => 108..110,
            IQ4_NL => 0..2,
            _ => unreachable!(),
        };
        for blk in 0..n / bs {
            for i in 0..bb {
                if !keep.contains(&i) {
                    row[blk * bb + i] = fill;
                }
            }
        }
        row
    }

    fn golden_stems() -> Vec<String> {
        let dir = crate::testutil::fixtures().join("golden");
        let mut stems: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter_map(|f| f.strip_suffix(".json").map(str::to_string))
            .collect();
        stems.sort();
        assert!(!stems.is_empty());
        stems
    }

    #[test]
    fn dot_row_q8_extreme_weights() {
        // saturated quants (all bits set / all clear) stress the i32 accumulators
        let mut rng = Lcg(7);
        for qt in [
            Q5_0, Q8_0, Q4_K, Q6_K, Q4_0, Q5_K, Q4_1, Q5_1, Q2_K, Q3_K, IQ4_NL,
        ] {
            for fill in [0x00u8, 0xff, 0x80, 0x7f] {
                let n = 512;
                let row = extreme_row(qt, n, fill, &mut rng);
                let x: Vec<f32> = (0..n)
                    .map(|i| if i % 2 == 0 { 1.0 } else { 0.999 })
                    .collect();
                check_q8(qt, &row, &x, &format!("qt {qt} fill {fill:#x}"));
            }
        }
    }

    #[test]
    fn dot_row_q8_golden() {
        let dir = crate::testutil::fixtures().join("golden");
        let mut rng = Lcg(42);
        for stem in &golden_stems() {
            let meta = crate::testutil::read_json(&format!("golden/{stem}.json"));
            let qt = meta["qtype"].as_u64().expect("qtype") as u32;
            let row_len = meta["row_len"].as_u64().expect("row_len") as usize;
            let bytes = std::fs::read(dir.join(format!("{stem}.raw"))).expect("raw");
            for rep in 0..4 {
                let x = synthetic_x(row_len, &mut rng);
                check_q8(qt, &bytes, &x, &format!("{stem} rep {rep}"));
            }
            let x: Vec<f32> = (0..row_len)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
                .collect();
            check_q8(qt, &bytes, &x, &format!("{stem} ramp"));
        }
    }

    #[test]
    fn quantized_act_896_has_no_k_view() {
        let mut rng = Lcg(3);
        let x = synthetic_x(896, &mut rng);
        let qa = QuantizedAct::new(&x);
        assert_eq!(qa.len(), 896);
        assert!(qa.b32.is_some());
        assert!(qa.k.is_none());
        for qt in [Q5_0, Q8_0] {
            let row = synthetic_row(qt, 896, &mut rng);
            check_q8(qt, &row, &x, &format!("qt {qt} n 896"));
        }
        // all-zero input: d = 0, everything quantizes to 0
        let qa = QuantizedAct::new(&[0.0; 512]);
        let row = synthetic_row(Q4_K, 512, &mut rng);
        assert_eq!(dot_row_q8(Q4_K, &row, &qa), 0.0);
    }

    #[test]
    #[should_panic(expected = "Q4_K needs input len multiple of 256, got 896")]
    fn dot_row_q8_q4_k_rejects_896() {
        let qa = QuantizedAct::new(&[0.5; 896]);
        dot_row_q8(Q4_K, &[0u8; 4 * Q4_K_BYTES], &qa);
    }

    #[test]
    #[should_panic(expected = "Q6_K needs input len multiple of 256, got 896")]
    fn dot_row_q8_q6_k_rejects_896() {
        let qa = QuantizedAct::new(&[0.5; 896]);
        dot_row_q8(Q6_K, &[0u8; 4 * Q6_K_BYTES], &qa);
    }

    // ---- SIMD kernels vs scalar ----

    /// SIMD paths this CPU supports; unsupported ones are reported, not hidden.
    fn simd_paths() -> Vec<Simd> {
        #[allow(unused_mut)]
        let mut v = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() {
                v.push(Simd::Avx2);
            } else {
                println!("::warning::SIMD test skipped the avx2 path: CPU lacks avx2/fma/f16c");
            }
            if has_vnni() {
                v.push(Simd::Vnni);
            } else {
                println!("::warning::SIMD test skipped the vnni path: CPU lacks avxvnni");
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        println!("::warning::SIMD test skipped every SIMD path: not x86_64");
        println!("SIMD paths under test: {v:?}");
        v
    }

    /// Every SIMD path must match the scalar kernel on the same QuantizedAct.
    fn check_simd(qt: u32, row: &[u8], x: &[f32], paths: &[Simd], label: &str) {
        let qa = QuantizedAct::new(x);
        let mut w = vec![0f32; x.len()];
        dequant(qt, row, &mut w);
        let xq = dequant_act(qt, &qa);
        let mag: f32 = w.iter().zip(&xq).map(|(a, b)| (a * b).abs()).sum();
        // SAFETY: Scalar is always valid.
        let want = unsafe { dot_row_q8_with(Simd::Scalar, qt, row, &qa) };
        for &p in paths {
            // SAFETY: simd_paths() only lists detected paths.
            let got = unsafe { dot_row_q8_with(p, qt, row, &qa) };
            assert!(
                (got - want).abs() <= 1e-5 * mag + 1e-6,
                "{label} {p:?}: simd {got} vs scalar {want} (mag {mag})"
            );
        }
    }

    #[test]
    fn simd_matches_scalar_synthetic() {
        let paths = simd_paths();
        let mut rng = Lcg(0xb10c);
        for &(qt, n) in &[
            (Q8_0, 32),
            (Q5_0, 32),
            (Q8_0, 896),
            (Q5_0, 896),
            (Q4_K, 256),
            (Q6_K, 256),
            (Q4_K, 1280),
            (Q6_K, 1280),
            (Q4_K, 4864),
            (Q6_K, 4864),
            (Q4_0, 32),
            (Q4_0, 896),
            (Q5_K, 256),
            (Q5_K, 4864),
            (Q4_1, 896),
            (Q5_1, 896),
            (IQ4_NL, 32),
            (IQ4_NL, 896),
            (Q2_K, 256),
            (Q2_K, 4864),
            (Q3_K, 256),
            (Q3_K, 4864),
        ] {
            for rep in 0..8 {
                let row = synthetic_row(qt, n, &mut rng);
                let x = synthetic_x(n, &mut rng);
                check_simd(qt, &row, &x, &paths, &format!("qt {qt} n {n} rep {rep}"));
            }
        }
    }

    /// Q2_K / Q3_K: every path (scalar, AVX2, the 4-token kernels) computes
    /// the same exact integers and the same float step per block, so the
    /// same bits, on random, extreme and all-max blocks.
    #[test]
    fn q2_k_q3_k_simd_bits_match_scalar() {
        let mut paths = vec![Simd::Scalar];
        paths.extend(simd_paths());
        let mut rng = Lcg(0x2b3c);
        for qt in [Q2_K, Q3_K, IQ4_NL] {
            for n in [256usize, 1536, 4864] {
                for rep in 0..12 {
                    let row = match rep % 4 {
                        1 => extreme_row(qt, n, 0x00, &mut rng),
                        2 => extreme_row(qt, n, 0xff, &mut rng),
                        3 => extreme_row(qt, n, 0xaa, &mut rng),
                        _ => synthetic_row(qt, n, &mut rng),
                    };
                    let x = if rep % 2 == 0 {
                        synthetic_x(n, &mut rng)
                    } else {
                        extreme_x(n, &mut rng)
                    };
                    let qa = QuantizedAct::for_types(&x, &[qt]);
                    // SAFETY: Scalar needs nothing.
                    let want = unsafe { dot_row_q8_with(Simd::Scalar, qt, &row, &qa) };
                    for &p in &paths {
                        // SAFETY: Scalar, or a path listed by simd_paths().
                        let got = unsafe { dot_row_q8_with(p, qt, &row, &qa) };
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "qt {qt} n {n} rep {rep} {p:?}"
                        );
                    }
                }
            }
        }
    }

    /// Q2_K / Q3_K blocks by hand: dequant is the documented layout.
    #[test]
    #[allow(clippy::needless_range_loop)] // e, l index several arrays
    fn q2_k_q3_k_blocks() {
        // Q2_K: d = 1, dmin = 0.5; sub-block g scale g & 15, min 15 - g.
        let mut b = vec![0u8; Q2_K_BYTES];
        for g in 0..16 {
            b[g] = (g as u8) | ((15 - g as u8) << 4);
        }
        // qs: byte k's 2-bit field j = (k + j) & 3.
        for k in 0..64 {
            b[16 + k] = (0..4).map(|j| (((k + j) & 3) << (2 * j)) as u8).sum();
        }
        b[80..82].copy_from_slice(&0x3c00u16.to_le_bytes());
        b[82..84].copy_from_slice(&0x3800u16.to_le_bytes());
        let mut out = [0f32; 256];
        dequant(Q2_K, &b, &mut out);
        for e in 0..256 {
            let (g, l) = (e / 16, e % 16);
            let (n, j, h) = (g / 8, (g % 8) / 2, g % 2);
            let q = (32 * n + 16 * h + l + j) & 3;
            assert_eq!(
                out[e],
                g as f32 * q as f32 - 0.5 * (15 - g) as f32,
                "q2_k e={e}"
            );
        }
        // Q3_K: d = 2, scales 0..16 - 32 (6-bit packing), qs as above,
        // hmask bit 4n + j set for odd l only.
        let mut b = vec![0u8; Q3_K_BYTES];
        for k in 0..64 {
            b[32 + k] = (0..4).map(|j| (((k + j) & 3) << (2 * j)) as u8).sum();
        }
        for l in 0..32 {
            b[l] = if l % 2 == 1 { 0xff } else { 0 };
        }
        let sc: Vec<u8> = (0..16).map(|i| (i * 4 + 1) as u8).collect(); // 6-bit
        for i in 0..16 {
            if i < 8 {
                b[96 + i] |= sc[i] & 0xf;
            } else {
                b[96 + i - 8] |= (sc[i] & 0xf) << 4;
            }
            b[96 + 8 + i % 4] |= ((sc[i] >> 4) & 3) << (2 * (i / 4));
        }
        b[108..110].copy_from_slice(&0x4000u16.to_le_bytes());
        dequant(Q3_K, &b, &mut out);
        for e in 0..256 {
            let (g, l) = (e / 16, e % 16);
            let (n, j, h) = (g / 8, (g % 8) / 2, g % 2);
            let q = ((32 * n + 16 * h + l + j) & 3) as i32 - if l % 2 == 1 { 0 } else { 4 };
            let want = 2.0 * (sc[g] as i32 - 32) as f32 * q as f32;
            assert_eq!(out[e], want, "q3_k e={e}");
        }
    }

    /// Q4_1 / Q5_1: the SIMD kernels only sum the integers differently, so
    /// every path gives the scalar kernel's bits.
    #[test]
    fn q_1_simd_bits_match_scalar() {
        let mut rng = Lcg(0x0001_0051);
        for qt in [Q4_1, Q5_1] {
            for n in [32, 896, 4864] {
                for rep in 0..6 {
                    let row = if rep < 3 {
                        synthetic_row(qt, n, &mut rng)
                    } else {
                        extreme_row(qt, n, [0x00, 0xff, 0xa5][rep - 3], &mut rng)
                    };
                    let x = if rep % 2 == 0 {
                        synthetic_x(n, &mut rng)
                    } else {
                        extreme_x(n, &mut rng)
                    };
                    let qa = QuantizedAct::for_types(&x, &[qt]);
                    // SAFETY: Scalar, or a path listed by simd_paths().
                    let want = unsafe { dot_row_q8_with(Simd::Scalar, qt, &row, &qa) };
                    for p in simd_paths() {
                        let got = unsafe { dot_row_q8_with(p, qt, &row, &qa) };
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "qt {qt} n {n} rep {rep} {p:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn simd_matches_scalar_extreme() {
        let paths = simd_paths();
        let mut rng = Lcg(0xe7);
        for qt in [
            Q5_0, Q8_0, Q4_K, Q6_K, Q4_0, Q5_K, Q4_1, Q5_1, Q2_K, Q3_K, IQ4_NL,
        ] {
            for fill in [0x00u8, 0xff, 0x80, 0x7f, 0x55, 0xaa] {
                for n in [256, 1280] {
                    let row = extreme_row(qt, n, fill, &mut rng);
                    // full-scale activations of both signs: q = ±127
                    let alt: Vec<f32> = (0..n)
                        .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                        .collect();
                    let pos: Vec<f32> = (0..n).map(|i| 1.0 - (i % 2) as f32 * 1e-3).collect();
                    for (xn, x) in [("alt", &alt), ("pos", &pos)] {
                        let label = format!("qt {qt} fill {fill:#x} n {n} x {xn}");
                        check_simd(qt, &row, x, &paths, &label);
                    }
                }
            }
        }
    }

    #[test]
    fn simd_matches_scalar_golden() {
        let paths = simd_paths();
        let dir = crate::testutil::fixtures().join("golden");
        let mut rng = Lcg(0x901d);
        for stem in &golden_stems() {
            let meta = crate::testutil::read_json(&format!("golden/{stem}.json"));
            let qt = meta["qtype"].as_u64().expect("qtype") as u32;
            let row_len = meta["row_len"].as_u64().expect("row_len") as usize;
            let bytes = std::fs::read(dir.join(format!("{stem}.raw"))).expect("raw");
            for rep in 0..4 {
                let x = synthetic_x(row_len, &mut rng);
                check_simd(qt, &bytes, &x, &paths, &format!("{stem} rep {rep}"));
            }
        }
    }

    #[test]
    fn dot_row_q8_multi_matches_single() {
        let mut paths = vec![Simd::Scalar];
        paths.extend(simd_paths());
        let mut rng = Lcg(0x4a11);
        for &(qt, n) in &[
            (Q5_0, 896),
            (Q8_0, 896),
            (Q4_K, 4864),
            (Q6_K, 4864),
            (Q4_0, 896),
            (Q5_K, 4864),
            (Q4_1, 896),
            (Q5_1, 896),
            (Q2_K, 4864),
            (Q3_K, 4864),
            (IQ4_NL, 896),
        ] {
            for nb in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 13] {
                let row = synthetic_row(qt, n, &mut rng);
                let qas: Vec<QuantizedAct> = (0..nb)
                    .map(|_| QuantizedAct::new(&synthetic_x(n, &mut rng)))
                    .collect();
                let xs: Vec<&QuantizedAct> = qas.iter().collect();
                for &p in &paths {
                    let mut out = vec![f32::NAN; nb];
                    // SAFETY: Scalar, or a path listed by simd_paths().
                    unsafe { dot_row_q8_multi_with(p, qt, &row, &xs, &mut out) };
                    for (b, (&got, qa)) in out.iter().zip(&qas).enumerate() {
                        // SAFETY: as above.
                        let want = unsafe { dot_row_q8_with(p, qt, &row, qa) };
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "qt {qt} n {n} B {nb} token {b} {p:?}: multi {got} vs single {want}"
                        );
                    }
                }
            }
        }
    }

    /// The `batch_slots` padding: real tokens followed by repeats of the last
    /// one must give each real token exactly its single-token result.
    #[test]
    fn padded_batches_match_single() {
        let mut paths = vec![Simd::Scalar];
        paths.extend(simd_paths());
        let mut rng = Lcg(0x9ad);
        for &(qts, n) in &[
            ([Q5_0, Q8_0], 896),
            ([Q8_0, Q8_0], 896),
            ([Q4_K, Q4_K], 896 * 2 + 256),
            ([Q6_K, Q6_K], 4864),
            ([Q4_K, Q6_K], 4864),
            ([Q2_K, Q3_K], 4864),
            ([Q3_K, Q6_K], 4864),
            ([IQ4_NL, Q5_0], 896),
        ] {
            assert_eq!(batch_slots(&qts, n, 1), 1);
            assert!(batch_slots(&[F32], n, 3) == 3);
            for nb in 1usize..=9 {
                let slots = nb.next_multiple_of(4);
                let r0 = synthetic_row(qts[0], n, &mut rng);
                let r1 = synthetic_row(qts[1], n, &mut rng);
                let qas: Vec<QuantizedAct> = (0..nb)
                    .map(|_| QuantizedAct::new(&synthetic_x(n, &mut rng)))
                    .collect();
                let xs: Vec<&QuantizedAct> = (0..slots).map(|b| &qas[b.min(nb - 1)]).collect();
                for &p in &paths {
                    let mut o0 = vec![f32::NAN; slots];
                    let mut o1 = vec![f32::NAN; slots];
                    let mut o2 = vec![f32::NAN; slots];
                    // SAFETY: Scalar, or a path listed by simd_paths().
                    unsafe {
                        dot_rows_q8_multi_with(
                            p,
                            qts,
                            [&r0[..], &r1[..]],
                            &xs,
                            [&mut o0[..], &mut o1[..]],
                        );
                        dot_row_q8_multi_with(p, qts[1], &r1, &xs, &mut o2);
                    }
                    for b in 0..nb {
                        // SAFETY: as above.
                        let want = unsafe {
                            [
                                dot_row_q8_with(p, qts[0], &r0, &qas[b]),
                                dot_row_q8_with(p, qts[1], &r1, &qas[b]),
                            ]
                        };
                        let got = [o0[b], o1[b], o2[b]];
                        for (k, g) in got.iter().enumerate() {
                            assert_eq!(
                                g.to_bits(),
                                want[k.min(1)].to_bits(),
                                "qts {qts:?} n {n} B {nb} token {b} out {k} {p:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dot_rows_q8_multi_matches_single() {
        let mut paths = vec![Simd::Scalar];
        paths.extend(simd_paths());
        let mut rng = Lcg(0x2a4);
        for &(qts, n) in &[
            ([Q5_0, Q5_0], 896),
            ([Q8_0, Q8_0], 896),
            ([Q5_0, Q8_0], 896),
            ([Q8_0, Q5_0], 896),
            ([Q4_K, Q4_K], 4864),
            ([Q6_K, Q6_K], 4864),
            ([Q4_K, Q6_K], 4864),
            ([Q4_K, Q4_K], 256),
            ([Q6_K, Q6_K], 256),
            ([Q4_0, Q4_0], 896),
            ([Q5_K, Q5_K], 4864),
            ([Q5_K, Q4_K], 4864),
            ([Q4_0, Q5_0], 896),
            ([Q5_1, Q5_1], 896),
            ([Q4_1, Q8_0], 896),
        ] {
            for nb in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 13] {
                let r0 = synthetic_row(qts[0], n, &mut rng);
                let r1 = synthetic_row(qts[1], n, &mut rng);
                let qas: Vec<QuantizedAct> = (0..nb)
                    .map(|_| QuantizedAct::new(&synthetic_x(n, &mut rng)))
                    .collect();
                let xs: Vec<&QuantizedAct> = qas.iter().collect();
                for &p in &paths {
                    let mut o0 = vec![f32::NAN; nb];
                    let mut o1 = vec![f32::NAN; nb];
                    // SAFETY: Scalar, or a path listed by simd_paths().
                    unsafe {
                        dot_rows_q8_multi_with(
                            p,
                            qts,
                            [&r0[..], &r1[..]],
                            &xs,
                            [&mut o0[..], &mut o1[..]],
                        )
                    };
                    for (k, (row, out)) in [(&r0, &o0), (&r1, &o1)].into_iter().enumerate() {
                        for (b, (&got, qa)) in out.iter().zip(&qas).enumerate() {
                            // SAFETY: as above.
                            let want = unsafe { dot_row_q8_with(p, qts[k], row, qa) };
                            assert_eq!(
                                got.to_bits(),
                                want.to_bits(),
                                "qts {qts:?} n {n} row {k} B {nb} token {b} {p:?}: rows {got} vs single {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Q5_0 row whose element j of block b has raw value (j + 7b + seed) % 32,
    /// so every raw value 0..31 shows up in every block.
    #[allow(clippy::needless_range_loop)] // j indexes both qh bits and qs bytes
    fn q5_0_all_raw_row(n: usize, seed: usize, rng: &mut Lcg) -> Vec<u8> {
        let mut row = synthetic_row(Q5_0, n, rng);
        for (b, blk) in row.chunks_exact_mut(Q5_0_BYTES).enumerate() {
            let r = |j: usize| ((j + 7 * b + seed) % 32) as u8;
            let mut qh = 0u32;
            for j in 0..32 {
                qh |= ((r(j) >> 4) as u32) << j;
            }
            blk[2..6].copy_from_slice(&qh.to_le_bytes());
            for j in 0..16 {
                blk[6 + j] = (r(j) & 0xf) | ((r(j + 16) & 0xf) << 4);
            }
        }
        row
    }

    /// Q8_0 row with -128, 127, -127, 0, ±1 in every block, the rest random.
    fn q8_0_extreme_row(n: usize, rng: &mut Lcg) -> Vec<u8> {
        let mut row = synthetic_row(Q8_0, n, rng);
        let pins = [-128i8, 127, -127, 0, 1, -1, -128, 127];
        for (b, blk) in row.chunks_exact_mut(Q8_0_BYTES).enumerate() {
            for (k, &v) in pins.iter().enumerate() {
                blk[2 + (k * 4 + b) % 32] = v as u8;
            }
        }
        row
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

    #[test]
    fn dot_rows_q8_multi_extreme_matches_single() {
        let mut paths = vec![Simd::Scalar];
        paths.extend(simd_paths());
        let mut rng = Lcg(0xe8e8);
        for n in [32usize, 896] {
            for qt in [Q5_0, Q8_0] {
                let (r0, r1) = if qt == Q5_0 {
                    (
                        q5_0_all_raw_row(n, 0, &mut rng),
                        q5_0_all_raw_row(n, 13, &mut rng),
                    )
                } else {
                    (q8_0_extreme_row(n, &mut rng), q8_0_extreme_row(n, &mut rng))
                };
                for nb in [1usize, 3, 5, 64, 67] {
                    let qas: Vec<QuantizedAct> = (0..nb)
                        .map(|_| QuantizedAct::for_types(&extreme_x(n, &mut rng), &[qt]))
                        .collect();
                    let xs: Vec<&QuantizedAct> = qas.iter().collect();
                    for &p in &paths {
                        let mut o0 = vec![f32::NAN; nb];
                        let mut o1 = vec![f32::NAN; nb];
                        // SAFETY: Scalar, or a path listed by simd_paths().
                        unsafe {
                            dot_rows_q8_multi_with(
                                p,
                                [qt, qt],
                                [&r0[..], &r1[..]],
                                &xs,
                                [&mut o0[..], &mut o1[..]],
                            )
                        };
                        for (k, (row, out)) in [(&r0, &o0), (&r1, &o1)].into_iter().enumerate() {
                            for (b, (&got, qa)) in out.iter().zip(&qas).enumerate() {
                                // SAFETY: as above.
                                let want = unsafe { dot_row_q8_with(p, qt, row, qa) };
                                assert_eq!(
                                    got.to_bits(),
                                    want.to_bits(),
                                    "qt {qt} n {n} row {k} B {nb} token {b} {p:?}: rows {got} vs single {want}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// K-quant row with saturated fields, varying per block: f16 scale(s) at
    /// max |value| (±65504) or random; Q4_K 6-bit scales/mins all 0 or all 63
    /// (or random), nibbles all 0 or all 15 (or random); Q6_K quants all 0 or
    /// all 63 (or random), i8 scales -128 / 127 (or random).
    fn kquant_extreme_row(qt: u32, n: usize, seed: usize, rng: &mut Lcg) -> Vec<u8> {
        let mut row = synthetic_row(qt, n, rng);
        let (_, bb) = block_info(qt);
        for (b, blk) in row.chunks_exact_mut(bb).enumerate() {
            let k = b + seed;
            let fill = |v: u8, r: &mut [u8], rng: &mut Lcg| match v {
                0 => r.fill(0x00),
                1 => r.fill(0xff),
                _ => r.iter_mut().for_each(|x| *x = rng.next_u32() as u8),
            };
            let maxd = |rng: &mut Lcg| -> [u8; 2] {
                (0x7bffu16 | (((rng.next_u32() & 1) as u16) << 15)).to_le_bytes()
            };
            if qt == Q4_K || qt == Q5_K {
                if k.is_multiple_of(2) {
                    let v = maxd(rng);
                    blk[0..2].copy_from_slice(&v);
                }
                if k % 3 != 1 {
                    let v = maxd(rng);
                    blk[2..4].copy_from_slice(&v);
                }
                fill((k % 3) as u8, &mut blk[4..16], rng);
                fill(((k / 3) % 3) as u8, &mut blk[16..bb], rng);
            } else {
                if k.is_multiple_of(2) {
                    let v = maxd(rng);
                    blk[208..210].copy_from_slice(&v);
                }
                fill((k % 3) as u8, &mut blk[0..192], rng);
                let sc = &mut blk[192..208];
                match (k / 3) % 4 {
                    0 => sc.fill(0x80),
                    1 => sc.fill(0x7f),
                    2 => sc
                        .iter_mut()
                        .enumerate()
                        .for_each(|(j, x)| *x = if j % 2 == 0 { 0x80 } else { 0x7f }),
                    _ => sc.iter_mut().for_each(|x| *x = rng.next_u32() as u8),
                }
            }
        }
        row
    }

    #[test]
    fn dot_rows_q8_multi_kquant_extreme_matches_single() {
        let mut paths = vec![Simd::Scalar];
        paths.extend(simd_paths());
        let mut rng = Lcg(0x4b4b);
        for n in [256usize, 4864] {
            for qt in [Q4_K, Q6_K] {
                let r0 = kquant_extreme_row(qt, n, 0, &mut rng);
                let r1 = kquant_extreme_row(qt, n, 5, &mut rng);
                for nb in [1usize, 3, 4, 5, 64, 67] {
                    let qas: Vec<QuantizedAct> = (0..nb)
                        .map(|_| QuantizedAct::for_types(&extreme_x(n, &mut rng), &[qt]))
                        .collect();
                    let xs: Vec<&QuantizedAct> = qas.iter().collect();
                    for &p in &paths {
                        let mut o0 = vec![f32::NAN; nb];
                        let mut o1 = vec![f32::NAN; nb];
                        // SAFETY: Scalar, or a path listed by simd_paths().
                        unsafe {
                            dot_rows_q8_multi_with(
                                p,
                                [qt, qt],
                                [&r0[..], &r1[..]],
                                &xs,
                                [&mut o0[..], &mut o1[..]],
                            )
                        };
                        for (k, (row, out)) in [(&r0, &o0), (&r1, &o1)].into_iter().enumerate() {
                            for (b, (&got, qa)) in out.iter().zip(&qas).enumerate() {
                                // SAFETY: as above.
                                let want = unsafe { dot_row_q8_with(p, qt, row, qa) };
                                assert!(want.is_finite(), "qt {qt} n {n}: non-finite {want}");
                                assert_eq!(
                                    got.to_bits(),
                                    want.to_bits(),
                                    "qt {qt} n {n} row {k} B {nb} token {b} {p:?}: rows {got} vs single {want}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The unpacked K-quant pair kernels against the single-token kernels on
    /// rows whose bytes are all one value (every nibble/quant/scale/min at an
    /// extreme at once), on both SIMD paths, called directly.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kquant_pair_kernels_uniform_rows() {
        let paths = simd_paths();
        let mut rng = Lcg(0x6c6c);
        let n = 512;
        for qt in [Q4_K, Q6_K] {
            for fill in [0x00u8, 0xff, 0x0f, 0xf0, 0x80, 0x7f] {
                let r0 = extreme_row(qt, n, fill, &mut rng);
                let r1 = extreme_row(qt, n, !fill, &mut rng);
                let qas: Vec<QuantizedAct> = (0..4)
                    .map(|_| QuantizedAct::for_types(&extreme_x(n, &mut rng), &[qt]))
                    .collect();
                let xs: Vec<&QuantizedAct> = qas.iter().collect();
                for &p in &paths {
                    let mut o0 = vec![f32::NAN; 4];
                    let mut o1 = vec![f32::NAN; 4];
                    // SAFETY: a path listed by simd_paths().
                    let done = unsafe {
                        dot_rows_kquant(p, qt, [&r0[..], &r1[..]], &xs, [&mut o0[..], &mut o1[..]])
                    };
                    assert_eq!(done, 4);
                    for (k, (row, out)) in [(&r0, &o0), (&r1, &o1)].into_iter().enumerate() {
                        for (b, (&got, qa)) in out.iter().zip(&qas).enumerate() {
                            // SAFETY: as above.
                            let want = unsafe { dot_row_q8_with(p, qt, row, qa) };
                            assert_eq!(
                                got.to_bits(),
                                want.to_bits(),
                                "qt {qt} fill {fill:#x} row {k} token {b} {p:?}: pair {got} vs single {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Rows of every kind the decode kernels must get bit-identical: random,
    /// saturated (Q5_0 all raw 0 / all 31 / every raw value; Q8_0 all -128 /
    /// all 127 / pinned extremes; K-quant extreme fields), and random rows with
    /// every block scale at max |d| (±65504). Row r is kind r % kinds.
    fn decode_rows(qt: u32, n: usize, count: usize, rng: &mut Lcg) -> Vec<u8> {
        let (bs, bb) = block_info(qt);
        let mut out = Vec::new();
        for r in 0..count {
            let row = match (qt, r % 5) {
                (_, 0) => synthetic_row(qt, n, rng),
                (Q5_0, 1) => extreme_row(qt, n, 0x00, rng),
                (Q5_0, 2) => extreme_row(qt, n, 0xff, rng),
                (Q5_0, 3) => q5_0_all_raw_row(n, r, rng),
                (Q8_0, 1) => extreme_row(qt, n, 0x80, rng),
                (Q8_0, 2) => extreme_row(qt, n, 0x7f, rng),
                (Q8_0, 3) => q8_0_extreme_row(n, rng),
                (Q4_0, 1) => extreme_row(qt, n, 0x00, rng),
                (Q4_0, 2) => extreme_row(qt, n, 0xff, rng),
                (Q4_0, 3) => extreme_row(qt, n, 0x0f, rng),
                (Q4_1 | Q5_1, 1) => extreme_row(qt, n, 0x00, rng),
                (Q4_1 | Q5_1, 2) => extreme_row(qt, n, 0xff, rng),
                (Q4_1 | Q5_1, 3) => extreme_row(qt, n, 0xf0, rng),
                (IQ4_NL, 1) => extreme_row(qt, n, 0x00, rng),
                (IQ4_NL, 2) => extreme_row(qt, n, 0xff, rng),
                (IQ4_NL, 3) => extreme_row(qt, n, 0x0f, rng),
                (Q2_K | Q3_K, 1) => extreme_row(qt, n, 0x00, rng),
                (Q2_K | Q3_K, 2) => extreme_row(qt, n, 0xff, rng),
                (Q2_K | Q3_K, 3) => extreme_row(qt, n, 0x55, rng),
                (_, 4) => {
                    let mut row = synthetic_row(qt, n, rng);
                    let d_off = match qt {
                        Q6_K => 208,
                        Q2_K => 80,
                        Q3_K => 108,
                        _ => 0,
                    };
                    for blk in 0..n / bs {
                        let v = 0x7bffu16 | (((rng.next_u32() & 1) as u16) << 15);
                        let p = blk * bb + d_off;
                        row[p..p + 2].copy_from_slice(&v.to_le_bytes());
                    }
                    row
                }
                (_, k) => kquant_extreme_row(qt, n, k + r, rng),
            };
            out.extend_from_slice(&row);
        }
        out
    }

    #[test]
    fn dot_rows_q8_matches_single_row() {
        let mut paths = vec![Simd::Scalar];
        paths.extend(simd_paths());
        let mut rng = Lcg(0xdec0de);
        for (qt, widths) in [
            (Q5_0, [896usize, 4864]),
            (Q8_0, [896, 4864]),
            (Q4_K, [256, 4864]),
            (Q6_K, [256, 4864]),
            (Q4_0, [896, 4864]),
            (Q5_K, [256, 4864]),
            (Q4_1, [896, 4864]),
            (Q5_1, [896, 4864]),
            (IQ4_NL, [896, 4864]),
            (Q2_K, [256, 4864]),
            (Q3_K, [256, 4864]),
        ] {
            for n in widths {
                let rb = row_bytes(qt, n);
                let xs = [
                    extreme_x(n, &mut rng),
                    extreme_x(n, &mut rng),
                    synthetic_x(n, &mut rng),
                ];
                for count in 1..=9 {
                    // one spare row at the end: prefetch may look, compute may not
                    let rows = decode_rows(qt, n, count + 1, &mut rng);
                    for x in &xs {
                        let qa = QuantizedAct::for_types(x, &[qt]);
                        for &p in &paths {
                            // SAFETY: Scalar, or a path listed by simd_paths().
                            let want: Vec<f32> = (0..count)
                                .map(|r| unsafe {
                                    dot_row_q8_with(p, qt, &rows[r * rb..(r + 1) * rb], &qa)
                                })
                                .collect();
                            for (pf, mr) in [
                                (0usize, 4),
                                (64, 4),
                                (DEFAULT_PREFETCH, 4),
                                (1 << 40, 4),
                                (512, 2),
                                (512, 1),
                            ] {
                                for tail in [count * rb, rows.len()] {
                                    let mut got = vec![f32::NAN; count];
                                    // SAFETY: as above.
                                    unsafe {
                                        dot_rows_q8_with(
                                            p,
                                            qt,
                                            &rows[..tail],
                                            &qa,
                                            &mut got,
                                            pf,
                                            mr,
                                        )
                                    };
                                    for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                                        assert_eq!(
                                            g.to_bits(),
                                            w.to_bits(),
                                            "qt {qt} n {n} rows {count} row {r} {p:?} pf {pf} max_rows {mr}: {g} vs single {w}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // the public entry (env-chosen path and prefetch) agrees too
        let n = 896;
        let rows = decode_rows(Q5_0, n, 7, &mut rng);
        let qa = QuantizedAct::for_types(&extreme_x(n, &mut rng), &[Q5_0]);
        let mut got = vec![0f32; 7];
        dot_rows_q8(Q5_0, &rows, &qa, &mut got, DecodeCfg::get());
        let rb = row_bytes(Q5_0, n);
        for (r, g) in got.iter().enumerate() {
            let w = dot_row_q8(Q5_0, &rows[r * rb..(r + 1) * rb], &qa);
            assert_eq!(g.to_bits(), w.to_bits(), "public dot_rows_q8 row {r}");
        }
    }

    /// The decode kernels' unsigned lanes equal the single-row kernels'
    /// sign-trick lanes, and both the exact Σ w·a per 4-value lane, for every
    /// weight value against every activation value in -127..=127.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(clippy::needless_range_loop)] // j indexes several arrays
    fn decode_unsigned_lanes_match_sign_trick() {
        use std::arch::x86_64::{__m256i, _mm256_loadu_si256};
        let (avx2, vnni) = (has_avx2(), has_vnni());
        if !avx2 {
            println!("::warning::lane test skipped: CPU lacks avx2/fma/f16c");
            return;
        }
        if !vnni {
            println!("::warning::lane test skipped the vnni lanes: CPU lacks avxvnni");
        }
        let lanes = |v: __m256i| -> [i32; 8] {
            // SAFETY: same size, plain integers.
            unsafe { std::mem::transmute(v) }
        };
        let exact = |w: &[i32; 32], a: &[i8; 32]| -> [i32; 8] {
            let mut e = [0i32; 8];
            for j in 0..32 {
                e[j / 4] += w[j] * a[j] as i32;
            }
            e
        };
        let acts = |seed: usize| -> [i8; 32] {
            let mut a = [0i8; 32];
            for j in 0..32 {
                a[j] = (((seed + 37 * j) % 255) as i32 - 127) as i8;
            }
            a
        };
        let mut a_sets: Vec<[i8; 32]> = (0..255).map(acts).collect();
        a_sets.push([127; 32]);
        a_sets.push([-127; 32]);
        // Q5_0: raw (j + shift) % 32, plus all 0 and all 31
        let mut raws: Vec<[u8; 32]> = (0..32)
            .map(|s| std::array::from_fn(|j| ((j + s) % 32) as u8))
            .collect();
        raws.push([0; 32]);
        raws.push([31; 32]);
        for raw in &raws {
            let mut b = [0u8; Q5_0_BYTES];
            let mut qh = 0u32;
            for j in 0..32 {
                qh |= ((raw[j] >> 4) as u32) << j;
            }
            b[2..6].copy_from_slice(&qh.to_le_bytes());
            for j in 0..16 {
                b[6 + j] = (raw[j] & 0xf) | ((raw[j + 16] & 0xf) << 4);
            }
            let w: [i32; 32] = std::array::from_fn(|j| raw[j] as i32 - 16);
            for a in &a_sets {
                let c = neg16_lane_sums(a);
                let e = exact(&w, a);
                // SAFETY: avx2 detected above; vnni only when detected.
                unsafe {
                    let qy = _mm256_loadu_si256(a.as_ptr().cast());
                    let qw = x86::q5_0_weights(&b);
                    assert_eq!(lanes(x86::sign_lanes_avx2(qw, qy)), e, "q5_0 sign avx2");
                    assert_eq!(lanes(x86::q5_0_lanes_avx2(&b, qy, &c)), e, "q5_0 avx2");
                    if vnni {
                        assert_eq!(lanes(x86::sign_lanes_vnni(qw, qy)), e, "q5_0 sign vnni");
                        assert_eq!(lanes(x86::q5_0_lanes_vnni(&b, qy, &c)), e, "q5_0 vnni");
                    }
                }
            }
        }
        // Q8_0: w = (k + 8j) as i8 for every k, plus all -128 and all 127
        let mut ws: Vec<[u8; 32]> = (0..256)
            .map(|k| std::array::from_fn(|j| ((k + 8 * j) % 256) as u8))
            .collect();
        ws.push([0x80; 32]);
        ws.push([0x7f; 32]);
        for wb in &ws {
            let mut b = [0u8; Q8_0_BYTES];
            b[2..].copy_from_slice(wb);
            let w: [i32; 32] = std::array::from_fn(|j| wb[j] as i8 as i32);
            for a in &a_sets {
                let c = neg16_lane_sums(a);
                let e = exact(&w, a);
                // SAFETY: as above.
                unsafe {
                    let qy = _mm256_loadu_si256(a.as_ptr().cast());
                    let qw = x86::q8_0_weights(&b);
                    assert_eq!(lanes(x86::sign_lanes_avx2(qw, qy)), e, "q8_0 sign avx2");
                    assert_eq!(lanes(x86::q8_0_lanes_avx2(&b, qy, &c)), e, "q8_0 avx2");
                    if vnni {
                        assert_eq!(lanes(x86::sign_lanes_vnni(qw, qy)), e, "q8_0 sign vnni");
                        assert_eq!(lanes(x86::q8_0_lanes_vnni(&b, qy, &c)), e, "q8_0 vnni");
                    }
                }
            }
        }
    }

    /// The activation quantizers' portable and x86-64-v3 variants give the
    /// same bits (see `model::tests::multiversion_bit_identical`).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn multiversion_quantize_bit_identical() {
        use crate::model::tests::{bits, can_v3, mixed};
        if !can_v3() {
            return;
        }
        for (i, n) in [0usize, 32, 256, 896, 4864].into_iter().enumerate() {
            for edgy in [false, true] {
                for scale in [1.0f32, 1e-30, 3e30] {
                    let x = mixed(n, 77 + i as u64, scale, edgy);
                    if n % 32 == 0 {
                        let a = quantize_b32::portable(&x);
                        let b = unsafe { quantize_b32::v3(&x) };
                        assert_eq!(bits(&a.d), bits(&b.d), "b32 d n={n}");
                        assert_eq!((a.q, a.c, a.n128), (b.q, b.c, b.n128), "b32 q/c n={n}");
                    }
                    if n % 256 == 0 {
                        let a = quantize_k::portable(&x);
                        let b = unsafe { quantize_k::v3(&x) };
                        assert_eq!(a.len(), b.len());
                        for (a, b) in a.iter().zip(&b) {
                            assert_eq!(a.d.to_bits(), b.d.to_bits(), "k d n={n}");
                            assert_eq!((a.qs, a.bsums), (b.qs, b.bsums), "k qs n={n}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn simd_choice() {
        assert_eq!(choose_simd(Some("scalar")), Simd::Scalar);
        let best = choose_simd(None);
        assert_eq!(choose_simd(Some("")), best);
        #[cfg(target_arch = "x86_64")]
        {
            if has_vnni() {
                assert_eq!(best, Simd::Vnni);
                assert_eq!(choose_simd(Some("vnni")), Simd::Vnni);
            } else if has_avx2() {
                assert_eq!(best, Simd::Avx2);
            }
            if has_avx2() {
                assert_eq!(choose_simd(Some("avx2")), Simd::Avx2);
            }
        }
        println!("default SIMD path: {best:?}");
    }

    #[test]
    #[should_panic(expected = "expected scalar, avx2 or vnni")]
    fn simd_choice_rejects_unknown() {
        choose_simd(Some("sse9"));
    }

    // ---- r8 (8-row interleaved K-quant) kernels ----

    /// Single-thread r8 (Q4_K / Q6_K) GEMM speed, as r8q_gemm_speed (run
    /// pinned, --ignored --nocapture): per 8-row-group calls vs one call.
    #[test]
    #[ignore]
    fn r8_gemm_speed() {
        let mut rng = Lcg(0x5eed);
        for qt in [Q4_K, Q6_K] {
            for n in [1536usize, 8960] {
                let (rows, nt) = (128, 64);
                let p = r8::repack(qt, &r8_matrix(qt, rows, n, false, &mut rng), rows, n);
                let qas: Vec<QuantizedAct> = (0..nt)
                    .map(|_| QuantizedAct::for_types(&synthetic_x(n, &mut rng), &[qt]))
                    .collect();
                let xs: Vec<&QuantizedAct> = qas.iter().collect();
                let mut o = vec![0f32; rows * nt];
                let gb = r8::group_bytes(qt, n);
                for chunked in [true, false] {
                    let mut reps = 0usize;
                    let t = std::time::Instant::now();
                    while t.elapsed().as_secs_f64() < 0.6 {
                        if chunked {
                            for (g, og) in o.chunks_mut(8 * nt).enumerate() {
                                r8::gemm(qt, &p[g * gb..(g + 1) * gb], &xs, og, nt);
                            }
                        } else {
                            r8::gemm(qt, &p, &xs, &mut o, nt);
                        }
                        reps += 1;
                    }
                    let s = t.elapsed().as_secs_f64();
                    let gmac = (reps * rows * n * nt) as f64 / s / 1e9;
                    println!(
                        "qt {qt} n {n}: {}: {gmac:.1} GMAC/s ({:.1} us per 8-row group)",
                        if chunked {
                            "per-group calls"
                        } else {
                            "one call"
                        },
                        s / (reps * rows / 8) as f64 * 1e6
                    );
                }
            }
        }
    }

    /// Single-thread GMAC/s of the batched CPU path per type (64 tokens,
    /// 128 rows): the new types' per-row 4-token kernels next to Q4_K's r8
    /// GEMM and per-row kernel (`--ignored --nocapture`).
    #[test]
    #[ignore]
    fn q23_multi_speed() {
        let mut rng = Lcg(0x5eed);
        let (rows, nt, n) = (128usize, 64usize, 3584usize);
        for qt in [Q4_K, Q2_K, Q3_K, IQ4_NL, Q4_0, Q5_0] {
            let w: Vec<u8> = (0..rows)
                .flat_map(|_| synthetic_row(qt, n, &mut rng))
                .collect();
            let rb = row_bytes(qt, n);
            let qas: Vec<QuantizedAct> = (0..nt)
                .map(|_| QuantizedAct::for_types(&synthetic_x(n, &mut rng), &[qt]))
                .collect();
            let xs: Vec<&QuantizedAct> = qas.iter().collect();
            let mut o = vec![0f32; nt];
            let mut reps = 0usize;
            let t = std::time::Instant::now();
            while t.elapsed().as_secs_f64() < 0.6 {
                for r in 0..rows {
                    dot_row_q8_multi(qt, &w[r * rb..(r + 1) * rb], &xs, &mut o);
                }
                reps += 1;
            }
            let gmac = (reps * rows * n * nt) as f64 / t.elapsed().as_secs_f64() / 1e9;
            println!(
                "{} per-row x4: {gmac:.2} GMAC/s",
                crate::gguf::qtype(qt).name
            );
            if qt == Q4_K {
                let p = r8::repack(qt, &w, rows, n);
                let mut o = vec![0f32; rows * nt];
                let mut reps = 0usize;
                let t = std::time::Instant::now();
                while t.elapsed().as_secs_f64() < 0.6 {
                    r8::gemm(qt, &p, &xs, &mut o, nt);
                    reps += 1;
                }
                let gmac = (reps * rows * n * nt) as f64 / t.elapsed().as_secs_f64() / 1e9;
                println!("Q4_K r8 gemm: {gmac:.2} GMAC/s");
            }
        }
    }

    /// `rows` rows of `qt`, width n: random, and K-quant extremes per block.
    fn r8_matrix(qt: u32, rows: usize, n: usize, extreme: bool, rng: &mut Lcg) -> Vec<u8> {
        (0..rows)
            .flat_map(|r| {
                if extreme {
                    kquant_extreme_row(qt, n, r * 7, rng)
                } else {
                    synthetic_row(qt, n, rng)
                }
            })
            .collect()
    }

    #[test]
    fn r8_repack_roundtrip() {
        let mut rng = Lcg(0x7238);
        for qt in [Q4_K, Q6_K] {
            for n in [256usize, 1024] {
                for extreme in [false, true] {
                    let rows = 16;
                    let m = r8_matrix(qt, rows, n, extreme, &mut rng);
                    let p = r8::repack(qt, &m, rows, n);
                    let gb = r8::group_bytes(qt, n);
                    assert_eq!(p.len(), rows / 8 * gb);
                    // Q6_K keeps its size; Q4_K grows 144 -> 148 bytes per row block.
                    let per_row_block = if qt == Q4_K { 148 } else { 210 };
                    assert_eq!(p.len(), rows * (n / QK_K) * per_row_block);
                    let rb = row_bytes(qt, n);
                    for r in 0..rows {
                        let mut want = vec![0f32; n];
                        dequant(qt, &m[r * rb..(r + 1) * rb], &mut want);
                        let mut got = vec![0f32; n];
                        r8::dequant_row(qt, &p[r / 8 * gb..(r / 8 + 1) * gb], r % 8, &mut got);
                        for k in 0..n {
                            assert_eq!(
                                got[k].to_bits(),
                                want[k].to_bits(),
                                "qt {qt} n {n} extreme {extreme} row {r} k {k}: {} vs {}",
                                got[k],
                                want[k]
                            );
                        }
                    }
                }
            }
        }
    }

    /// Scalar GEMV per token is the reference; every path's GEMV and GEMM
    /// (every batch size) must equal it bit for bit.
    #[test]
    fn r8_paths_and_gemm_match_scalar_gemv() {
        let mut paths = vec![Simd::Scalar];
        paths.extend(simd_paths());
        let mut rng = Lcg(0x8888);
        let counts: Vec<usize> = (1..=9).chain([64, 67]).collect();
        for qt in [Q4_K, Q6_K] {
            for (n, extreme) in [(256usize, false), (256, true), (1024, false), (4864, true)] {
                let rows = 16;
                let m = r8_matrix(qt, rows, n, extreme, &mut rng);
                let p = r8::repack(qt, &m, rows, n);
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
                            unsafe { r8::gemv_with(Simd::Scalar, qt, &p, qa, &mut o, 0) };
                            o
                        })
                        .collect();
                    for &path in &paths {
                        for pf in [0usize, 64, 8192] {
                            for (t, qa) in qas.iter().enumerate() {
                                let mut o = vec![f32::NAN; rows];
                                // SAFETY: Scalar, or a path listed by simd_paths().
                                unsafe { r8::gemv_with(path, qt, &p, qa, &mut o, pf) };
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
                        unsafe { r8::gemm_with(path, qt, &p, &xs, &mut w, st) };
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
                        unsafe { r8::gemm_with(path, qt, &p, &xs, &mut o, nt) };
                        for r in 0..rows {
                            for t in 0..nt {
                                assert_eq!(
                                    o[r * nt + t].to_bits(),
                                    want[t][r].to_bits(),
                                    "gemm qt {qt} n {n} {path:?} B {nt} token {t} row {r}: {} vs {}",
                                    o[r * nt + t],
                                    want[t][r]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Distance in ulps between two finite f32s (sign-aware).
    fn ulps(a: f32, b: f32) -> u64 {
        let key = |v: f32| {
            let i = v.to_bits() as i32;
            if i < 0 {
                i32::MIN as i64 - i as i64
            } else {
                i as i64
            }
        };
        (key(a) - key(b)).unsigned_abs()
    }

    /// The new scheme against an f64 dot of the dequantized weights and
    /// activations (relative to Σ|w x|), and against today's kernel. Prints
    /// the deltas; asserts only the f64 bound.
    #[test]
    fn r8_accuracy() {
        let mut rng = Lcg(0xacc0);
        for qt in [Q4_K, Q6_K] {
            for n in [256, 4864, 18944] {
                let rows = 64;
                let m = r8_matrix(qt, rows, n, false, &mut rng);
                let p = r8::repack(qt, &m, rows, n);
                let rb = row_bytes(qt, n);
                let (mut new_rel, mut old_rel, mut max_ulp, mut sum_ulp, mut cnt) =
                    (0f64, 0f64, 0u64, 0u64, 0u64);
                for _ in 0..4 {
                    let x = synthetic_x(n, &mut rng);
                    let qa = QuantizedAct::for_types(&x, &[qt]);
                    let xq = dequant_act(qt, &qa);
                    let mut o = vec![0f32; rows];
                    r8::gemv(qt, &p, &qa, &mut o, 8192);
                    for r in 0..rows {
                        let row = &m[r * rb..(r + 1) * rb];
                        let mut w = vec![0f32; n];
                        dequant(qt, row, &mut w);
                        let exact: f64 =
                            w.iter().zip(&xq).map(|(a, b)| *a as f64 * *b as f64).sum();
                        let mag: f64 = w
                            .iter()
                            .zip(&xq)
                            .map(|(a, b)| (*a as f64 * *b as f64).abs())
                            .sum::<f64>()
                            .max(f64::MIN_POSITIVE);
                        let old = dot_row_q8(qt, row, &qa);
                        let e_new = (o[r] as f64 - exact).abs() / mag;
                        let e_old = (old as f64 - exact).abs() / mag;
                        assert!(
                            e_new <= 1e-5,
                            "qt {qt} n {n} row {r}: r8 {} vs exact {exact} (rel {e_new:e})",
                            o[r]
                        );
                        new_rel = new_rel.max(e_new);
                        old_rel = old_rel.max(e_old);
                        let u = ulps(o[r], old);
                        max_ulp = max_ulp.max(u);
                        sum_ulp += u;
                        cnt += 1;
                    }
                }
                println!(
                    "r8 accuracy qt {qt} n {n}: max rel err vs f64 r8 {new_rel:.2e} old {old_rel:.2e}; \
                     r8 vs old max {max_ulp} ulp, mean {:.1} ulp",
                    sum_ulp as f64 / cnt as f64
                );
            }
        }
    }
}
