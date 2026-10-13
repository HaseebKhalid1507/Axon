// Dequantization kernels for GGUF tensor types.
//
// Reference: ggml (https://github.com/ggml-org/llama.cpp), MIT License,
// Copyright (c) 2023-2024 The ggml authors. Block layouts follow
// ggml/src/ggml-common.h; dequant math follows dequantize_row_* and
// get_scale_min_k4 in ggml/src/ggml-quants.c. All reads are little-endian
// from byte slices (no transmutes).

const QK5_0: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

// block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; }
const Q5_0_BYTES: usize = 2 + 4 + QK5_0 / 2; // 22
// block_q8_0 { ggml_half d; int8_t qs[32]; }
const Q8_0_BYTES: usize = 2 + QK8_0; // 34
// block_q4_K { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[128]; }
const Q4_K_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2; // 144
// block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; ggml_half d; }
const Q6_K_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 210

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q5_0: u32 = 6;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q6_K: u32 = 14;

/// (values per block, bytes per block)
fn block_info(qtype_id: u32) -> (usize, usize) {
    match qtype_id {
        F32 => (1, 4),
        F16 => (1, 2),
        Q5_0 => (QK5_0, Q5_0_BYTES),
        Q8_0 => (QK8_0, Q8_0_BYTES),
        Q4_K => (QK_K, Q4_K_BYTES),
        Q6_K => (QK_K, Q6_K_BYTES),
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
                Q5_0 => dequant_q5_0_block,
                Q8_0 => dequant_q8_0_block,
                Q4_K => dequant_q4_k_block,
                Q6_K => dequant_q6_k_block,
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
}

#[allow(clippy::needless_range_loop)] // fixed-size arrays: indexing vectorizes cleanly
#[inline]
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
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
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
#[inline]
fn abs_max(x: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let (chunks, tail) = x.as_chunks::<8>();
    for c in chunks {
        for i in 0..8 {
            lanes[i] = lanes[i].max(c[i].abs());
        }
    }
    let m = tail.iter().fold(0f32, |m, v| m.max(v.abs()));
    lanes.iter().fold(m, |a, &b| a.max(b))
}

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
fn quantize_q8_k(x: &[f32], q: &mut [i8]) -> f32 {
    // The signed value with the largest magnitude, first one on ties (the
    // reference loop keeps the first because it only replaces on `>`). Two
    // passes so both vectorize: the magnitude, then where it first occurs.
    let amax = abs_max(x);
    if amax == 0.0 {
        q.fill(0);
        return 0.0;
    }
    let max = *x
        .iter()
        .find(|v| v.abs() == amax)
        .expect("amax comes from x");
    let iscale = -127.0 / max;
    for (qi, &xi) in q.iter_mut().zip(x) {
        *qi = round_even_small(iscale * xi).min(127) as i8;
    }
    1.0 / iscale
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
        let b32 = (wants(&[Q5_0, Q8_0]) && n.is_multiple_of(QK8_0)).then(|| {
            let nb = n / QK8_0;
            let mut d = Vec::with_capacity(nb);
            let mut q = vec![[0i8; 32]; nb];
            let mut c = Vec::with_capacity(nb);
            for (xb, qb) in x.as_chunks::<QK8_0>().0.iter().zip(q.iter_mut()) {
                d.push(quantize_q8_0(xb, qb));
                c.push(neg16_lane_sums(qb));
            }
            B32 { d, q, c }
        });
        let k = (wants(&[Q4_K, Q6_K]) && n.is_multiple_of(QK_K)).then(|| {
            x.as_chunks::<QK_K>()
                .0
                .iter()
                .map(|xb| {
                    let mut qs = [0i8; QK_K];
                    let d = quantize_q8_k(xb, &mut qs);
                    let mut bsums = [0i16; QK_K / 16];
                    for (s, g) in bsums.iter_mut().zip(qs.as_chunks::<16>().0) {
                        *s = g.iter().map(|&v| v as i16).sum();
                    }
                    KBlock { d, qs, bsums }
                })
                .collect()
        });
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

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
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
            }
            Simd::Scalar
        }
        Some(other) => panic!("AXON_SIMD={other}: expected scalar, avx2 or vnni"),
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{KBlock, Q4_K_BYTES, Q5_0_BYTES, Q6_K_BYTES, Q8_0_BYTES};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    fn load_u8x32(p: &[u8; 32]) -> __m256i {
        // SAFETY: p points to 32 readable bytes; loadu has no alignment requirement.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn load_i8x32(p: &[i8; 32]) -> __m256i {
        // SAFETY: as above.
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn load_i16x16(p: &[i16; 16]) -> __m256i {
        // SAFETY: as above (32 bytes).
        unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn load_u8x16(p: &[u8; 16]) -> __m128i {
        // SAFETY: as above (16 bytes).
        unsafe { _mm_loadu_si128(p.as_ptr().cast()) }
    }

    /// little-endian f16 -> f32 (exact, same value as `rd_f16`)
    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    fn h2f(b: [u8; 2]) -> f32 {
        let h = u16::from_le_bytes(b) as i32;
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn hsum_f32x8(x: __m256) -> f32 {
        let mut r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        _mm_cvtss_f32(r)
    }

    /// Byte i = 0xFF if bit i of x is set, else 0x00 (ggml bytes_from_bits_32).
    #[inline]
    #[target_feature(enable = "avx2")]
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
    #[target_feature(enable = "avx2")]
    fn mul_sum_us8_avx2(ax: __m256i, sy: __m256i) -> __m256 {
        let dot = _mm256_maddubs_epi16(ax, sy);
        _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), dot))
    }

    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    fn mul_sum_us8_vnni(ax: __m256i, sy: __m256i) -> __m256 {
        _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy))
    }

    /// 32 Q5_0 weights (nibble | high bit << 4) - 16 as i8.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn q5_0_weights(b: &[u8; Q5_0_BYTES]) -> __m256i {
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

    /// Broadcast i16 scale `i` of `scales` (8 i16, both lanes) to all 16 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn k4_scale(scales: __m256i, i: i32) -> __m256i {
        _mm256_shuffle_epi8(
            scales,
            _mm256_set1_epi16(((2 * i) | ((2 * i + 1) << 8)) as i16),
        )
    }

    /// Q6_K i8 scales of 16-groups 2i and 2i+1, each repeated 8 times, as i16.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn q6_scale(scales: __m128i, i: usize) -> __m256i {
        let lo = (2 * i) as i64 * 0x0101010101010101;
        let hi = (2 * i + 1) as i64 * 0x0101010101010101;
        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, _mm_set_epi64x(hi, lo)))
    }

    macro_rules! small_block_kernels {
        ($feat:literal, $q8:ident, $q5:ident, $mul:ident) => {
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

    small_block_kernels!("avx2,fma,f16c", q8_0_avx2, q5_0_avx2, mul_sum_us8_avx2);
    small_block_kernels!(
        "avx2,fma,f16c,avxvnni",
        q8_0_vnni,
        q5_0_vnni,
        mul_sum_us8_vnni
    );

    /// ggml_vec_dot_q4_K_q8_K (AVX2). Mins via bsums.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c.
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

    /// ggml_vec_dot_q6_K_q8_K (AVX2). The -32 offset is applied via bsums.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c.
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
    #[target_feature(enable = "avx2")]
    fn q8_0_weights(b: &[u8; Q8_0_BYTES]) -> __m256i {
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
    #[target_feature(enable = "avx2")]
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
    #[target_feature(enable = "avx2")]
    fn q8_0_raw(b: &[u8; Q8_0_BYTES]) -> __m256i {
        _mm256_xor_si256(q8_0_weights(b), _mm256_set1_epi8(0x80u8 as i8))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn store_u8x32(p: &mut [u8; 32], v: __m256i) {
        // SAFETY: p points to 32 writable bytes; storeu has no alignment requirement.
        unsafe { _mm256_storeu_si256(p.as_mut_ptr().cast(), v) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
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
            #[target_feature(enable = "avx2,f16c")]
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
    /// The CPU must support avx2, fma and avxvnni.
    #[allow(clippy::needless_range_loop)] // t indexes several arrays
    #[target_feature(enable = "avx2,fma,avxvnni")]
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
    /// The CPU must support avx2, fma and f16c.
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

    /// 4-token `q6_k_avx2`.
    ///
    /// # Safety
    /// The CPU must support avx2, fma and f16c.
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
}

/// Same meaning as dot_row(qtype_id, row, x) but with x pre-quantized.
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
        Q5_0 | Q8_0 => {
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
                (Q8_0, Simd::Avx2) => unsafe { x86::q8_0_avx2(row, dx, qx) },
                #[cfg(target_arch = "x86_64")]
                (Q8_0, Simd::Vnni) => unsafe { x86::q8_0_vnni(row, dx, qx) },
                #[cfg(target_arch = "x86_64")]
                (_, Simd::Avx2) => unsafe { x86::q5_0_avx2(row, dx, qx) },
                #[cfg(target_arch = "x86_64")]
                (_, Simd::Vnni) => unsafe { x86::q5_0_vnni(row, dx, qx) },
                (Q8_0, Simd::Scalar) => dot_q8_0_q8(row, dx, qx),
                (_, Simd::Scalar) => dot_q5_0_q8(row, dx, qx),
            }
        }
        Q4_K | Q6_K => {
            let xk = x.k.as_deref().unwrap_or_else(|| {
                panic!(
                    "dot_row_q8: {} needs input len multiple of 256, got {} (or the 256-block view wasn't built)",
                    name(),
                    n
                )
            });
            let row = &row[..row_bytes(qtype_id, n)];
            // AVX-VNNI gains nothing here (the scale multiply needs madd anyway),
            // so the vnni path uses the AVX2 K-quant kernels.
            match (qtype_id, simd) {
                #[cfg(target_arch = "x86_64")]
                (Q4_K, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q4_k_avx2(row, xk) },
                #[cfg(target_arch = "x86_64")]
                (_, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q6_k_avx2(row, xk) },
                (Q4_K, Simd::Scalar) => dot_q4_k_q8(row, xk),
                (_, Simd::Scalar) => dot_q6_k_q8(row, xk),
            }
        }
        _ => panic!("dot_row_q8: unsupported qtype {} ({})", qtype_id, name()),
    }
}

/// `out[b] = dot_row_q8(qtype_id, row, xs[b])`, bit for bit, for every b.
/// Tokens go through the SIMD kernels in groups of 4, so each weight block is
/// unpacked once per group rather than once per token.
pub fn dot_row_q8_multi(qtype_id: u32, row: &[u8], xs: &[&QuantizedAct], out: &mut [f32]) {
    // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
    unsafe { dot_row_q8_multi_with(simd(), qtype_id, row, xs, out) }
}

/// dot_row_q8_multi on an explicit kernel path.
///
/// # Safety
/// As for `dot_row_q8_with`.
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
unsafe fn dot_row_q8_x4(
    simd: Simd,
    qtype_id: u32,
    row: &[u8],
    xs: [&QuantizedAct; 4],
) -> Option<[f32; 4]> {
    let n = xs[0].len();
    match qtype_id {
        Q5_0 | Q8_0 => {
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
                (Q8_0, Simd::Avx2) => unsafe { x86::q8_0_avx2_x4(row, dx, qx) },
                (Q8_0, Simd::Vnni) => unsafe { x86::q8_0_vnni_x4(row, dx, qx) },
                (_, Simd::Avx2) => unsafe { x86::q5_0_avx2_x4(row, dx, qx) },
                (_, Simd::Vnni) => unsafe { x86::q5_0_vnni_x4(row, dx, qx) },
                (_, Simd::Scalar) => return None,
            })
        }
        Q4_K | Q6_K => {
            let mut xk: [&[KBlock]; 4] = [&[]; 4];
            for t in 0..4 {
                xk[t] = xs[t].k.as_deref()?;
            }
            let row = &row[..row_bytes(qtype_id, n)];
            // The vnni path uses the AVX2 K-quant kernels, as in dot_row_q8_with.
            Some(match (qtype_id, simd) {
                (Q4_K, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q4_k_avx2_x4(row, xk) },
                (_, Simd::Avx2 | Simd::Vnni) => unsafe { x86::q6_k_avx2_x4(row, xk) },
                (_, Simd::Scalar) => return None,
            })
        }
        _ => None,
    }
}

/// Two rows at once: `out[k][b] = dot_row_q8(qtypes[k], rows[k], xs[b])`, bit
/// for bit. When both rows have the same small-block type (Q5_0/Q8_0) on a
/// SIMD path, tokens go 4 at a time through a 2-row x 4-token kernel (on Vnni
/// the pair is unpacked once for all groups: `dot_rows_q8_vnni`); the rest
/// (other types, mixed pairs, < 4 leftover tokens) goes per row through
/// `dot_row_q8_multi`.
pub fn dot_rows_q8_multi(
    qtypes: [u32; 2],
    rows: [&[u8]; 2],
    xs: &[&QuantizedAct],
    out: [&mut [f32]; 2],
) {
    // SAFETY: simd() only returns Avx2/Vnni after detecting their features.
    unsafe { dot_rows_q8_multi_with(simd(), qtypes, rows, xs, out) }
}

/// dot_rows_q8_multi on an explicit kernel path.
///
/// # Safety
/// As for `dot_row_q8_with`.
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
    if qtypes[0] == qtypes[1] && simd == Simd::Vnni {
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
unsafe fn dot_rows_q8_vnni(
    qtype_id: u32,
    rows: [&[u8]; 2],
    xs: &[&QuantizedAct],
    out: [&mut [f32]; 2],
) -> usize {
    let groups = xs.len() / 4;
    if !matches!(qtype_id, Q5_0 | Q8_0)
        || groups == 0
        || xs[..groups * 4].iter().any(|x| x.b32.is_none())
    {
        return 0;
    }
    let n = xs[0].len();
    let nb = n / QK8_0;
    let rb = row_bytes(qtype_id, n);
    let rows = [&rows[0][..rb], &rows[1][..rb]];
    let [o0, o1] = out;
    PAIR_SCRATCH.with_borrow_mut(|scratch| {
        if scratch.len() < nb {
            scratch.resize(nb, x86::Pair32::ZERO);
        }
        let w = &mut scratch[..nb];
        // SAFETY (unsafe calls below): has_vnni() implies avx2, fma, f16c.
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
    });
    groups * 4
}

/// 2 rows of type `qtype_id` x 4 tokens through a 2x4 SIMD kernel, or None if
/// there is none for this path/type (or a view is missing).
///
/// # Safety
/// As for `dot_row_q8_with`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::needless_range_loop)] // t indexes xs and the view arrays
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
            Q5_0 | Q8_0 => &[0],
            Q4_K => &[0, 2],
            Q6_K => &[208],
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
            Q5_0 | Q8_0 => {
                let B32 { d, q, c } = qa.b32.as_ref().unwrap();
                for (q, c) in q.iter().zip(c) {
                    for (lane, &cl) in q.as_chunks::<4>().0.iter().zip(c) {
                        assert_eq!(cl, -16 * lane.iter().map(|&v| v as i32).sum::<i32>());
                    }
                }
                d.iter()
                    .zip(q)
                    .flat_map(|(&d, q)| q.map(|v| d * v as f32))
                    .collect()
            }
            Q4_K | Q6_K => {
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
            Q5_0 | Q8_0 => 0..2,
            Q4_K => 0..4,
            Q6_K => 208..210,
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
        for qt in [Q5_0, Q8_0, Q4_K, Q6_K] {
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
                println!("SKIP avx2 path: CPU lacks avx2/fma/f16c");
            }
            if has_vnni() {
                v.push(Simd::Vnni);
            } else {
                println!("SKIP vnni path: CPU lacks avxvnni");
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        println!("SKIP all SIMD paths: not x86_64");
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
        ] {
            for rep in 0..8 {
                let row = synthetic_row(qt, n, &mut rng);
                let x = synthetic_x(n, &mut rng);
                check_simd(qt, &row, &x, &paths, &format!("qt {qt} n {n} rep {rep}"));
            }
        }
    }

    #[test]
    fn simd_matches_scalar_extreme() {
        let paths = simd_paths();
        let mut rng = Lcg(0xe7);
        for qt in [Q5_0, Q8_0, Q4_K, Q6_K] {
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
        for &(qt, n) in &[(Q5_0, 896), (Q8_0, 896), (Q4_K, 4864), (Q6_K, 4864)] {
            for nb in [1usize, 3, 4, 5, 8, 13] {
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

    #[test]
    fn dot_rows_q8_multi_matches_single() {
        let mut paths = vec![Simd::Scalar];
        paths.extend(simd_paths());
        let mut rng = Lcg(0x2a4);
        let n = 896;
        for &qts in &[[Q5_0, Q5_0], [Q8_0, Q8_0], [Q5_0, Q8_0], [Q8_0, Q5_0]] {
            for nb in [1usize, 3, 4, 5, 8, 13] {
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
                                "qts {qts:?} row {k} B {nb} token {b} {p:?}: rows {got} vs single {want}"
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
}
