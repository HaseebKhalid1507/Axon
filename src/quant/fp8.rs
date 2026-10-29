//! FP8 (e4m3, the OCP "e4m3fn" variant: no infinities, NaN = S.1111.111)
//! weights of per-channel checkpoints (docs/design/fp8-weights.md):
//!
//! - the codec: `e4m3_to_f32` (exact) and `f32_to_e4m3` (round to nearest
//!   even, saturating to +-448: PTX `cvt.rn.satfinite.e4m3x2.f32`), in
//!   integer arithmetic, the GPU quantizer's twin (D2);
//! - the activation quantizer, per token (`quantize_act_f8`) and per 128
//!   values (`quantize_act_f8_g128`), and their round trips (the CPU
//!   emulation of D7);
//! - `F8S`: the CPU's form of a per-channel matrix, blocks of 128 e4m3
//!   bytes and the row's f32 scale, so `dequant` gives `e4m3 * scale`
//!   (exact in f32: 4 x 8 significant bits), HF F32's weights;
//! - the Q8_0 converters of the exact-mode fallback and the head (D1, D3);
//! - `gemv_f8_ref` (the CUDA GEMV's order, D6) and `f8_gemm_f64` (the
//!   independent anchor of every FP8 kernel, §3).

use super::{F16, Q8_0, Q8_0_BYTES, QK8_0, f16_to_f32, f32_to_f16, quantize_q8_0};

/// The file's raw e4m3 matrix (a safetensors element type: one byte a value).
pub const F8_E4M3: u32 = crate::safetensors::QT_F8_E4M3;
/// `F8S`: blocks of `F8S_QK` e4m3 values and the row's f32 scale; the CPU's
/// per-channel FP8 matrix, activations in f32 (`fp8-ref`).
pub const F8S: u32 = 1120;
/// `F8S`, activations rounded through e4m3 per token first (emulation).
pub const F8S_TOK: u32 = 1121;
/// `F8S`, activations rounded through e4m3 per 128 values first (emulation).
pub const F8S_G128: u32 = 1122;
/// Values per `F8S` block.
pub const F8S_QK: usize = 128;
/// Bytes per `F8S` block.
pub const F8S_BYTES: usize = F8S_QK + 4;
/// The largest finite e4m3 value.
pub const E4M3_MAX: f32 = 448.0;

/// e4m3 code `x` as f32, exactly: its exponent and mantissa bits placed in
/// an f32's (a value 2^-120 of the true one, normal or subnormal alike),
/// times 2^120. The NaN codes (0x7f, 0xff) come out as +-480; the loader
/// refuses files that hold them.
#[inline(always)]
pub fn e4m3_to_f32(x: u8) -> f32 {
    let b = ((x as u32 & 0x80) << 24) | ((x as u32 & 0x7f) << 20);
    f32::from_bits(b) * f32::from_bits((127 + 120) << 23)
}

/// f32 -> e4m3, round to nearest even, saturating: |x| past 448 (and
/// infinity) gives +-448, NaN gives 0x7f with x's sign, the sign of a zero
/// result is x's (as `cvt.rn.satfinite.e4m3x2.f32`).
pub fn f32_to_e4m3(x: f32) -> u8 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) as u8) << 7;
    let a = bits & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return sign | 0x7f;
    }
    if a >= E4M3_MAX.to_bits() {
        return sign | 0x7e;
    }
    // Below 2^-6 (the smallest normal): a multiple of 2^-9, 0..=8 (8 is the
    // smallest normal's code, 0x08). a * 512 is exact, its rounding RNE.
    if a < 0x3c80_0000 {
        let q = (f32::from_bits(a) * 512.0).round_ties_even() as u8;
        return sign | q;
    }
    let exp = (a >> 23) as i32 - 127; // -6..=8
    let mant = a & 0x7f_ffff;
    let mut m = mant >> 20;
    let rem = mant & 0xf_ffff;
    if rem > 0x8_0000 || (rem == 0x8_0000 && m & 1 == 1) {
        m += 1;
    }
    let mut e = exp + 7;
    if m == 8 {
        m = 0;
        e += 1;
    }
    // a < 448 rounds to at most 448 (e 15, m 6): never past it.
    sign | ((e as u8) << 3) | m as u8
}

/// The amax of `x` as both quantizers take it: |v| folded with `f32::max`
/// from 0 (NaN ignored, as the GPU's fmaxf).
fn amax(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |a, &v| a.max(v.abs()))
}

/// One group's codes into `q` and its scale: s = amax / 448, q = e4m3(x / s)
/// (both divisions rounded once, `__fdiv_rn`), all 0 when s is.
fn quant_group(x: &[f32], q: &mut [u8]) -> f32 {
    let s = amax(x) / E4M3_MAX;
    for (o, &v) in q.iter_mut().zip(x) {
        *o = if s == 0.0 { 0 } else { f32_to_e4m3(v / s) };
    }
    s
}

/// A token's activation as e4m3 codes and one scale (the GPU's
/// `quant_f8_tok`, D2).
pub fn quantize_act_f8(x: &[f32]) -> (Vec<u8>, f32) {
    let mut q = vec![0u8; x.len()];
    let s = quant_group(x, &mut q);
    (q, s)
}

/// Per 128 values (the last group may be short): codes and one scale per
/// group.
pub fn quantize_act_f8_g128(x: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut q = vec![0u8; x.len()];
    let s = x
        .chunks(F8S_QK)
        .zip(q.chunks_mut(F8S_QK))
        .map(|(xg, qg)| quant_group(xg, qg))
        .collect();
    (q, s)
}

/// x rounded through the activation quantizer of `qt` (`F8S_TOK`: per
/// token, `F8S_G128`: per 128): each value `e4m3 * s` in f32. Any other
/// type: x unchanged.
pub fn act_round_trip(qt: u32, x: &[f32]) -> Vec<f32> {
    match qt {
        F8S_TOK => {
            let (q, s) = quantize_act_f8(x);
            q.iter().map(|&c| e4m3_to_f32(c) * s).collect()
        }
        F8S_G128 => {
            let (q, s) = quantize_act_f8_g128(x);
            q.iter()
                .enumerate()
                .map(|(i, &c)| e4m3_to_f32(c) * s[i / F8S_QK])
                .collect()
        }
        _ => x.to_vec(),
    }
}

/// `rows` x `k` raw e4m3 bytes and their per-row scales as `F8S` blocks.
pub fn f8_to_f8s(raw: &[u8], scales: &[f32], k: usize, out: &mut [u8]) {
    assert!(
        k.is_multiple_of(F8S_QK),
        "F8S: row length {k} isn't a multiple of 128"
    );
    let rb = k / F8S_QK * F8S_BYTES;
    for ((row, &s), o) in raw
        .chunks_exact(k)
        .zip(scales)
        .zip(out.chunks_exact_mut(rb))
    {
        for (b, ob) in row.chunks_exact(F8S_QK).zip(o.chunks_exact_mut(F8S_BYTES)) {
            ob[..F8S_QK].copy_from_slice(b);
            ob[F8S_QK..].copy_from_slice(&s.to_le_bytes());
        }
    }
}

/// `dequant` of `F8S` blocks: e4m3 * the block's scale.
pub(super) fn dequant_f8s_block(b: &[u8], y: &mut [f32]) {
    let s = f32::from_le_bytes([b[F8S_QK], b[F8S_QK + 1], b[F8S_QK + 2], b[F8S_QK + 3]]);
    for (o, &c) in y.iter_mut().zip(&b[..F8S_QK]) {
        *o = e4m3_to_f32(c) * s;
    }
}

/// Q8_0 blocks of f32 values `x` (whole 32-blocks) into `out`.
fn q8_0_of(x: &[f32], out: &mut [u8]) {
    let mut q = [0i8; QK8_0];
    for (xb, ob) in x.chunks_exact(QK8_0).zip(out.chunks_exact_mut(Q8_0_BYTES)) {
        let d = quantize_q8_0(xb, &mut q);
        ob[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (o, &v) in ob[2..].iter_mut().zip(&q) {
            *o = v as u8;
        }
    }
}

/// Rows of a raw e4m3 matrix (`k` values each, scale `scales[r]`) as Q8_0:
/// the exact f32 dequant, then the Q8_0 quantizer (the fallback, D1).
pub fn f8_rows_to_q8_0(raw: &[u8], scales: &[f32], k: usize, out: &mut [u8]) {
    assert!(k.is_multiple_of(QK8_0));
    let rb = super::row_bytes(Q8_0, k);
    let mut x = vec![0f32; k];
    for ((row, &s), o) in raw
        .chunks_exact(k)
        .zip(scales)
        .zip(out.chunks_exact_mut(rb))
    {
        for (v, &c) in x.iter_mut().zip(row) {
            *v = e4m3_to_f32(c) * s;
        }
        q8_0_of(&x, o);
    }
}

/// Rows of an F16 matrix (`k` values each) as Q8_0 (the GPU's head, D3).
pub fn f16_rows_to_q8_0(raw: &[u8], k: usize, out: &mut [u8]) {
    assert!(k.is_multiple_of(QK8_0));
    let (ib, rb) = (super::row_bytes(F16, k), super::row_bytes(Q8_0, k));
    let mut x = vec![0f32; k];
    for (row, o) in raw.chunks_exact(ib).zip(out.chunks_exact_mut(rb)) {
        for (v, h) in x.iter_mut().zip(row.chunks_exact(2)) {
            *v = f16_to_f32(u16::from_le_bytes([h[0], h[1]]));
        }
        q8_0_of(&x, o);
    }
}

#[cfg(test)]
/// The CUDA `gemv_f8`'s products (D6), bit for bit: `w` rows x `k` raw e4m3
/// with scales `ws`, `qa` nb x k codes with scales `das`; out[t * rows + r].
/// Per output: lane l of 32 owns bytes 4l..4l + 4 of each 128-byte chunk and
/// chains fma over them in order from 0; an xor tree (16, 8, 4, 2, 1) of
/// the lanes' sums; then (sum * ws[r]) * das[t].
pub fn gemv_f8_ref(w: &[u8], ws: &[f32], k: usize, qa: &[u8], das: &[f32]) -> Vec<f32> {
    let rows = ws.len();
    let mut out = vec![0f32; das.len() * rows];
    for (t, (a, &da)) in qa.chunks_exact(k).zip(das).enumerate() {
        for (r, (wr, &s)) in w.chunks_exact(k).zip(ws).enumerate() {
            let mut lane = [0f32; 32];
            for c in 0..k / F8S_QK {
                for (l, acc) in lane.iter_mut().enumerate() {
                    for j in 0..4 {
                        let i = c * F8S_QK + 4 * l + j;
                        *acc = e4m3_to_f32(wr[i]).mul_add(e4m3_to_f32(a[i]), *acc);
                    }
                }
            }
            for off in [16, 8, 4, 2, 1] {
                let v = lane;
                for (l, x) in lane.iter_mut().enumerate() {
                    *x = v[l] + v[l ^ off];
                }
            }
            out[t * rows + r] = (lane[0] * s) * da;
        }
    }
    out
}

#[cfg(test)]
/// The anchor (docs/design/fp8-weights.md §3): per output, the exact value
/// E = ws * das * sum(w * a) in f64 (e4m3 products are exact; the f64 sum
/// all but), and the tolerance 2^-23 (k/32 + 40) ws das sum|w a|.
/// out[t * rows + r] = (E, tol).
pub fn f8_gemm_f64(w: &[u8], ws: &[f32], k: usize, qa: &[u8], das: &[f32]) -> Vec<(f64, f64)> {
    let rows = ws.len();
    let mut out = Vec::with_capacity(das.len() * rows);
    let c = (k as f64 / 32.0 + 40.0) * 2f64.powi(-23);
    for (a, &da) in qa.chunks_exact(k).zip(das) {
        for (wr, &s) in w.chunks_exact(k).zip(ws) {
            let (mut e, mut m) = (0f64, 0f64);
            for (&x, &y) in wr.iter().zip(a) {
                let p = e4m3_to_f32(x) as f64 * e4m3_to_f32(y) as f64;
                e += p;
                m += p.abs();
            }
            let sc = s as f64 * da as f64;
            out.push((e * sc, c * m * sc.abs()));
        }
    }
    out
}

#[cfg(test)]
/// Every product of `got` (nb x rows, `gemv_f8_ref`'s layout) within the
/// anchor's tolerance of `want` (`f8_gemm_f64`'s); the first miss if not.
pub fn check_anchor(got: &[f32], want: &[(f64, f64)], slack: f64) -> Result<(), String> {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &(e, tol))) in got.iter().zip(want).enumerate() {
        let err = (g as f64 - e).abs();
        if err.is_nan() || err > slack * tol {
            return Err(format!(
                "output {i}: {g} vs exact {e}: error {err:e} > {slack} x tolerance {tol:e}"
            ));
        }
    }
    Ok(())
}

/// Random finite e4m3 codes (no NaN), for tests here and the GPU's.
#[cfg(test)]
pub(crate) fn test_codes(n: usize, seed: &mut u64) -> Vec<u8> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let c = (*seed >> 33) as u8;
            if c & 0x7f == 0x7f { c ^ 1 } else { c }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// e4m3's value by its definition, in f64.
    fn value(c: u8) -> f64 {
        let (s, e, m) = (c >> 7, (c >> 3) & 15, c & 7);
        let v = if e == 0 {
            m as f64 / 8.0 * 2f64.powi(-6)
        } else {
            (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn e4m3_codes_round_trip() {
        for c in 0..=255u8 {
            let f = e4m3_to_f32(c);
            if c & 0x7f == 0x7f {
                assert_eq!(f.abs(), 480.0, "NaN code {c:#x}");
                assert_eq!(f32_to_e4m3(f32::NAN) & 0x7f, 0x7f);
                continue;
            }
            assert_eq!(f as f64, value(c), "code {c:#x}");
            assert_eq!(f32_to_e4m3(f), c, "code {c:#x} ({f})");
        }
    }

    /// Every f32 between two neighbouring codes: below the midpoint the
    /// lower, above the upper, at it the even one; past 448 saturated.
    #[test]
    fn e4m3_rounding_ties_and_saturation() {
        let pos: Vec<u8> = (0..0x7fu8).collect();
        for w in pos.windows(2) {
            let (lo, hi) = (e4m3_to_f32(w[0]), e4m3_to_f32(w[1]));
            let mid = (lo + hi) / 2.0; // exact: a few significant bits
            let even = if w[0] & 1 == 0 { w[0] } else { w[1] };
            assert_eq!(f32_to_e4m3(mid), even, "tie {lo}..{hi}");
            assert_eq!(f32_to_e4m3(-mid), 0x80 | even);
            assert_eq!(f32_to_e4m3(f32::from_bits(mid.to_bits() - 1)), w[0]);
            assert_eq!(f32_to_e4m3(f32::from_bits(mid.to_bits() + 1)), w[1]);
        }
        for v in [448.0f32, 449.0, 463.9, 464.0, 1e9, f32::INFINITY] {
            assert_eq!(f32_to_e4m3(v), 0x7e, "{v}");
            assert_eq!(f32_to_e4m3(-v), 0xfe, "{v}");
        }
        assert_eq!(f32_to_e4m3(-0.0), 0x80);
        assert_eq!(f32_to_e4m3(2f32.powi(-11)), 0); // a quarter of the smallest subnormal
        assert_eq!(f32_to_e4m3(f32::from_bits(1)), 0);
    }

    #[test]
    fn act_quantizers() {
        let x: Vec<f32> = (0..896)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        let (q, s) = quantize_act_f8(&x);
        let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert_eq!(s, amax / 448.0);
        for (&c, &v) in q.iter().zip(&x) {
            // Within half an e4m3 step of x / s.
            let y = (v / s) as f64;
            let got = value(c);
            let step = if y.abs() < 2f64.powi(-6) {
                2f64.powi(-9)
            } else {
                2f64.powi(y.abs().log2().floor() as i32 - 3)
            };
            assert!((got - y).abs() <= step / 2.0 + 1e-12, "{v}: {got} vs {y}");
        }
        assert!(q.iter().any(|&c| c & 0x7f == 0x7e), "amax maps to 448");
        let (q0, s0) = quantize_act_f8(&[0.0; 256]);
        assert_eq!((q0.iter().all(|&c| c == 0), s0), (true, 0.0));
        // Per 128: 896 = 7 groups, each its own amax.
        let (qg, sg) = quantize_act_f8_g128(&x);
        assert_eq!(sg.len(), 7);
        for (g, &sv) in sg.iter().enumerate() {
            let xg = &x[g * 128..(g + 1) * 128];
            assert_eq!(sv, xg.iter().fold(0f32, |a, v| a.max(v.abs())) / 448.0);
            assert_eq!(&qg[g * 128..(g + 1) * 128], &quantize_act_f8(xg).0[..]);
        }
    }

    use super::test_codes as codes;

    /// gemv_f8_ref against the f64 anchor at the design's shapes.
    #[test]
    fn gemv_ref_within_anchor() {
        let mut seed = 7;
        for (rows, k, nb) in [(64, 128, 1), (128, 896, 3), (9, 4864, 2)] {
            let w = codes(rows * k, &mut seed);
            let ws: Vec<f32> = (0..rows).map(|r| 1e-3 * (1.0 + r as f32 / 7.0)).collect();
            let qa = codes(nb * k, &mut seed);
            let das: Vec<f32> = (0..nb).map(|t| 0.01 + t as f32 * 0.003).collect();
            let got = gemv_f8_ref(&w, &ws, k, &qa, &das);
            check_anchor(&got, &f8_gemm_f64(&w, &ws, k, &qa, &das), 1.0).unwrap();
        }
    }

    /// A deliberately wrong product (one weight dropped) misses the anchor:
    /// the tolerance is tight enough to see one term.
    #[test]
    fn anchor_has_teeth() {
        let mut seed = 3;
        let (k, rows) = (896, 16);
        let w = codes(rows * k, &mut seed);
        let ws = vec![1.0f32; rows];
        let qa = codes(k, &mut seed);
        let want = f8_gemm_f64(&w, &ws, k, &qa, &[1.0]);
        let mut w2 = w.clone();
        for r in 0..rows {
            // The largest product of each row, zeroed.
            let i = (0..k)
                .max_by(|&a, &b| {
                    let p = |i: usize| (e4m3_to_f32(w[r * k + i]) * e4m3_to_f32(qa[i])).abs();
                    p(a).total_cmp(&p(b))
                })
                .unwrap();
            w2[r * k + i] = 0;
        }
        let got = gemv_f8_ref(&w2, &ws, k, &qa, &[1.0]);
        assert!(check_anchor(&got, &want, 1.0).is_err());
    }

    #[test]
    fn converters() {
        let mut seed = 11;
        let (rows, k) = (3, 256);
        let raw = codes(rows * k, &mut seed);
        let sc = [0.5f32, 1.0 / 3.0, 7.25];
        let mut f8s = vec![0u8; rows * k / F8S_QK * F8S_BYTES];
        f8_to_f8s(&raw, &sc, k, &mut f8s);
        let mut y = vec![0f32; rows * k];
        super::super::dequant(F8S, &f8s, &mut y);
        for (i, &v) in y.iter().enumerate() {
            assert_eq!(
                v as f64,
                value(raw[i]) * sc[i / k] as f64,
                "exact dequant {i}"
            );
        }
        let mut q8 = vec![0u8; rows * super::super::row_bytes(Q8_0, k)];
        f8_rows_to_q8_0(&raw, &sc, k, &mut q8);
        let mut want = vec![0u8; q8.len()];
        q8_0_of(&y, &mut want);
        assert_eq!(q8, want);
    }
}
