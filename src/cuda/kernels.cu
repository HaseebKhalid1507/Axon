// axon's CUDA kernels, compiled at run time by NVRTC (src/cuda/mod.rs).
//
// Block decoding mirrors src/quant.rs (itself ggml's dequantize_row_* and
// get_scale_min_k4), and is BIT-IDENTICAL to quant::dequant: every float
// op is an explicit round-to-nearest intrinsic (__fmul_rn, __fsub_rn), which
// the compiler never contracts into an FMA, in the CPU's order (Rust rounds
// every op, never fuses). For these formats it couldn't differ anyway: an
// f16 scale (11 significant bits) times integers of at most 8 bits is exact
// in f32, so only the last op rounds, contracted or not. The intrinsics make
// it hold by construction, also for a future format where it wouldn't. Int -> float
// conversions are of small integers (|q| <= 255), exact.
//
// The deq_* functions decode element i of one block; GEMVs reuse them.

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

// GGML type ids (quant.rs).
#define QT_F32 0
#define QT_F16 1
#define QT_Q5_0 6
#define QT_Q8_0 8
#define QT_Q4_K 12
#define QT_Q6_K 14

// Little-endian reads from bytes: blocks are only 2-byte aligned (Q5_0 is 22
// bytes, Q6_K 210), so no wider loads.
__device__ __forceinline__ u32 rd_u16(const u8* p) {
    return (u32)p[0] | ((u32)p[1] << 8);
}

// quant::f16_to_f32: exact, NaN payloads kept. The sign is set with a bit
// op, not a negation, so -0 and negative subnormals come out exactly.
__device__ __forceinline__ float f16_to_f32(u32 h) {
    u32 sign = (h >> 15) & 1;
    u32 exp = (h >> 10) & 0x1f;
    u32 mant = h & 0x3ff;
    if (exp == 0) {
        // zero or subnormal: mant * 2^-24, exact in f32
        float v = __fmul_rn((float)mant, 1.0f / 16777216.0f);
        return __uint_as_float(__float_as_uint(v) | (sign << 31));
    }
    u32 bits = exp == 31 ? (sign << 31) | 0x7f800000u | (mant << 13)
                         : (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float rd_f16(const u8* p) {
    return f16_to_f32(rd_u16(p));
}

// block_q5_0 { half d; u8 qh[4]; u8 qs[16]; }, 32 values, 22 bytes.
__device__ __forceinline__ float deq_q5_0(const u8* b, int i) {
    float d = rd_f16(b);
    u32 qh = (u32)b[2] | ((u32)b[3] << 8) | ((u32)b[4] << 16) | ((u32)b[5] << 24);
    int j = i & 15;
    int x = i < 16 ? (int)((b[6 + j] & 0x0f) | (((qh >> j) << 4) & 0x10)) - 16
                   : (int)((b[6 + j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
    return __fmul_rn((float)x, d);
}

// block_q8_0 { half d; int8 qs[32]; }, 32 values, 34 bytes.
__device__ __forceinline__ float deq_q8_0(const u8* b, int i) {
    return __fmul_rn((float)(signed char)b[2 + i], rd_f16(b));
}

// quant::get_scale_min_k4: 6-bit scale and min of sub-block j (0..8).
__device__ __forceinline__ void scale_min_k4(int j, const u8* q, u32* sc, u32* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// block_q4_K { half d; half dmin; u8 scales[12]; u8 qs[128]; }, 256 values,
// 144 bytes. 64-value chunks c: low nibbles of qs[32c..] with sub-block 2c,
// then the high nibbles with 2c + 1.
__device__ __forceinline__ float deq_q4_k(const u8* b, int i) {
    int c = i >> 6, w = i & 63, l = w & 31, hi = w >> 5;
    u32 sc, m;
    scale_min_k4(2 * c + hi, b + 4, &sc, &m);
    float d1 = __fmul_rn(rd_f16(b), (float)sc);
    float m1 = __fmul_rn(rd_f16(b + 2), (float)m);
    u8 q = b[16 + 32 * c + l];
    u32 nib = hi ? q >> 4 : q & 0xf;
    return __fsub_rn(__fmul_rn(d1, (float)nib), m1);
}

// block_q6_K { u8 ql[128]; u8 qh[64]; int8 scales[16]; half d; }, 256
// values, 210 bytes. Halves of 128; in each, quarter k of 32 takes the low
// (k < 2) or high nibbles of ql[l + 32 (k & 1)] and bits 2k..2k+1 of qh[l].
__device__ __forceinline__ float deq_q6_k(const u8* b, int i) {
    int half = i >> 7, r = i & 127, k = r >> 5, l = r & 31, is = l >> 4;
    const u8* ql = b + 64 * half;
    const u8* qh = b + 128 + 32 * half;
    const u8* sc = b + 192 + 8 * half;
    u8 qlb = ql[l + 32 * (k & 1)];
    u32 nib = k < 2 ? qlb & 0xf : qlb >> 4;
    int q = (int)(nib | (((qh[l] >> (2 * k)) & 3) << 4)) - 32;
    float ds = __fmul_rn(rd_f16(b + 208), (float)(signed char)sc[is + 2 * k]);
    return __fmul_rn(ds, (float)q);
}

// Element i of a row of type qt (one of the QT_* ids).
__device__ __forceinline__ float deq(int qt, const u8* row, int i) {
    switch (qt) {
    case QT_F32:
        return __uint_as_float((u32)row[4 * i] | ((u32)row[4 * i + 1] << 8) |
                               ((u32)row[4 * i + 2] << 16) | ((u32)row[4 * i + 3] << 24));
    case QT_F16:
        return rd_f16(row + 2 * i);
    case QT_Q5_0:
        return deq_q5_0(row + 22 * (i >> 5), i & 31);
    case QT_Q8_0:
        return deq_q8_0(row + 34 * (i >> 5), i & 31);
    case QT_Q4_K:
        return deq_q4_k(row + 144 * (i >> 8), i & 255);
    default: // QT_Q6_K; the host only launches supported types
        return deq_q6_k(row + 210 * (i >> 8), i & 255);
    }
}

// out[r][i] = element i of row rows[r] of a matrix of type qt with
// row_bytes bytes per row and n values per row. Grid: (rows, ceil(n / block)).
extern "C" __global__ void dequant_rows(int qt, const u8* __restrict__ src, u64 row_bytes, int n,
                                        const u32* __restrict__ rows, float* __restrict__ out) {
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 r = blockIdx.x;
    out[r * (u64)n + i] = deq(qt, src + (u64)rows[r] * row_bytes, i);
}
