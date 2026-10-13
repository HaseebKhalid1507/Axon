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

#[cfg(test)]
mod tests {
    use super::*;

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
            for (i, (g, w)) in got.iter().zip(want.chunks_exact(4)).enumerate() {
                let w = f32::from_le_bytes(w.try_into().unwrap());
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
}
