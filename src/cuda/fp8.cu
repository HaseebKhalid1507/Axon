
// ---------------------------------------------------------------------------
// FP8 (e4m3) per-channel matrices (docs/design/fp8-weights.md). Compiled only
// into the fast-mode module of an FP8 checkpoint on sm >= 89 (-DAXON_F8):
// these kernels are in the fast set and promise tolerance, not bits, against
// the exact kernels and against each other. Each has a bits reference of its
// own order and the f64 anchor (quant::fp8::f8_gemm_f64), tests.rs.
//
// Activations (D2): quant_warp also stores its f32 values at act row +
// f8_off(n) (decode.cu, under AXON_F8); quant_f8_tok makes them e4m3 per
// token into qa [nb][K] (row stride K, 16-byte aligned: a TMA 2D box) with
// one f32 scale per token in das [nb] (contiguous, token-major).
// Weights: w [rows][K] e4m3 as in the file, ws [rows] f32 (the row scale).
// Every product: y[t][r] = (sum_k w[r][k] * qa[t][k] * ws[r]) * das[t], into
// gm_store's epilogue (bias, residual, swiglu mode 2).

// e4m3 code x as f32 (exact; --ftz=false keeps the subnormals): its bits in
// an f32's places, times 2^120 (quant::fp8::e4m3_to_f32).
__device__ __forceinline__ float f8f(u32 x) {
    return __int_as_float(((x & 0x80u) << 24) | ((x & 0x7fu) << 20)) * 0x1p120f;
}

// Two floats as two e4m3 codes, round to nearest even, saturating (lo in the
// low byte).
__device__ __forceinline__ unsigned short f8x2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

// Test: x[i] as e4m3 codes (the Rust twin quant::fp8::f32_to_e4m3). n even.
extern "C" __global__ void f8_cvt(const float* __restrict__ x, int n, u8* __restrict__ out) {
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) ((unsigned short*)out)[i / 2] = f8x2(x[i], x[i + 1]);
}

// Per token z (grid nb x 256): amax over the n f32 values at act + z *
// astride + f8_off(n) (fmaxf: NaN ignored), s = amax / 448, qa[z][e] =
// e4m3(x / s) (0 where s == 0), das[z] = s. quant::fp8::quantize_act_f8.
extern "C" __global__ void __launch_bounds__(256)
quant_f8_tok(const u8* __restrict__ act, u64 astride, int n, u8* __restrict__ qa,
             float* __restrict__ das) {
    __shared__ float sh[8];
    int z = blockIdx.x, tid = threadIdx.x;
    const float* x = (const float*)(act + (u64)z * astride + f8_off(n));
    float m = 0.0f;
    for (int i = tid; i < n; i += 256) m = fmaxf(m, fabsf(x[i]));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((tid & 31) == 0) sh[tid >> 5] = m;
    __syncthreads();
    m = sh[0];
    for (int w = 1; w < 8; w++) m = fmaxf(m, sh[w]);
    float s = __fdiv_rn(m, 448.0f);
    unsigned short* q = (unsigned short*)(qa + (u64)z * n);
    for (int i = 2 * tid; i < n; i += 512)
        q[i / 2] = s == 0.0f ? 0 : f8x2(__fdiv_rn(x[i], s), __fdiv_rn(x[i + 1], s));
    if (tid == 0) das[z] = s;
}

// GEMV for nb <= 16 tokens (D6), CUDA cores. Block of 8 warps, a row each
// (rows 8 * blockIdx.x ..). The tokens' codes, decoded to f32, are staged
// in shared memory 512 values at a time ([nb][512] floats, dynamic). Lane l
// owns bytes 4l..4l + 4 of each 128-byte chunk: acc[t] = fma(w, a, acc[t])
// in k order; then the xor tree 16, 8, 4, 2, 1; then (sum * ws) * das.
// quant::fp8::gemv_f8_ref is this order.
#define F8V_T 512
extern "C" __global__ void __launch_bounds__(256)
gemv_f8(const u8* __restrict__ w, int rows, int K, const u8* __restrict__ qa,
        const float* __restrict__ das, int nb, const float* __restrict__ ws,
        const float* __restrict__ bias, int mode, float* __restrict__ out, u64 ostride) {
    extern __shared__ float xs[];  // [nb][F8V_T]
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int r = blockIdx.x * 8 + warp;
    bool live = r < rows;
    const u8* wr = w + (u64)(live ? r : 0) * K;
    float acc[16];
#pragma unroll
    for (int t = 0; t < 16; t++) acc[t] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += F8V_T) {
        int kn = min(F8V_T, K - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < nb * kn; i += 256) {
            int t = i / kn, k = i - t * kn;
            xs[t * F8V_T + k] = f8f(qa[(u64)t * K + k0 + k]);
        }
        __syncthreads();
        if (live) {
            for (int c = 0; c < kn; c += 128) {
                u32 wv = __ldg((const u32*)(wr + k0 + c + 4 * lane));
                float w0 = f8f(wv), w1 = f8f(wv >> 8), w2 = f8f(wv >> 16), w3 = f8f(wv >> 24);
#pragma unroll
                for (int t = 0; t < 16; t++) {
                    if (t < nb) {
                        float4 a = *(const float4*)&xs[t * F8V_T + c + 4 * lane];
                        acc[t] = __fmaf_rn(w0, a.x, acc[t]);
                        acc[t] = __fmaf_rn(w1, a.y, acc[t]);
                        acc[t] = __fmaf_rn(w2, a.z, acc[t]);
                        acc[t] = __fmaf_rn(w3, a.w, acc[t]);
                    }
                }
            }
        }
    }
    if (!live) return;
    float s = ws[r];
#pragma unroll
    for (int t = 0; t < 16; t++) {
        if (t < nb) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
            if (lane == t) gm_store(__fmul_rn(__fmul_rn(v, s), das[t]), bias, mode, out, (u64)t * ostride, r);
        }
    }
}

// mma.sync.m16n8k32 e4m3 x e4m3 -> f32 (sm_89+): D = A B + C. Fragments
// (the int8 m16n8k32 layout; g = lane / 4, q = lane % 4): a0 (row g, k 4q..),
// a1 (row g + 8, k 4q..), a2 (row g, k 16 + 4q..), a3 (row g + 8, k 16 + 4q..);
// b0 (col g, k 4q..), b1 (col g, k 16 + 4q..); c0, c1 (row g, cols 2q, 2q + 1),
// c2, c3 (row g + 8, same cols).
__device__ __forceinline__ void mma_f8(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// GEMM for nb > 16 (D5): 64 rows x 64 tokens a block of 8 warps (2 x 4: 32
// rows x 16 tokens each, 2 x 2 mma tiles), K in stages of 128 bytes through
// two cp.async buffers, rows padded to 144 bytes (no bank conflicts). Each
// output's accumulator carries the whole K through the mma chain, k ascending
// (A1's form), then (acc * ws) * das. Rows and tokens past the ends are
// zero-filled, not stored.
#define F8G_LD 144
#define F8G_STAGE (2 * 64 * F8G_LD)
extern "C" __global__ void __launch_bounds__(256)
gemm_f8(const u8* __restrict__ w, int rows, int K, const u8* __restrict__ qa,
        const float* __restrict__ das, int nb, const float* __restrict__ ws,
        const float* __restrict__ bias, int mode, float* __restrict__ out, u64 ostride) {
    extern __shared__ __align__(16) u8 sm[];  // [2][W 64 x 144 | X 64 x 144]
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
    int t0 = blockIdx.x * 64, r0 = blockIdx.y * 64;
    int wm = (warp >> 2) * 32, wn = (warp & 3) * 16;
    // Each thread copies 2 x 16 bytes of W and of X per stage: 64 rows x 8
    // chunks of 16 bytes each.
    auto load = [&](int st, int k0) {
        u8* s = sm + st * F8G_STAGE;
        for (int i = tid; i < 512; i += 256) {
            int row = i >> 3, ch = (i & 7) * 16;
            int rr = r0 + row, tt = t0 + row;
            cp16q(s + row * F8G_LD + ch, w + (u64)(rr < rows ? rr : 0) * K + k0 + ch, rr < rows);
            cp16q(s + 64 * F8G_LD + row * F8G_LD + ch, qa + (u64)(tt < nb ? tt : 0) * K + k0 + ch,
                  tt < nb);
        }
        cp_commit();
    };
    float acc[2][2][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    int ns = K / 128;
    load(0, 0);
    for (int s = 0; s < ns; s++) {
        if (s + 1 < ns) {
            load((s + 1) & 1, (s + 1) * 128);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const u8* W = sm + (s & 1) * F8G_STAGE;
        const u8* X = W + 64 * F8G_LD;
#pragma unroll
        for (int kk = 0; kk < 128; kk += 32) {
            u32 a[2][4], b[2][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const u8* p = W + (wm + 16 * i + g) * F8G_LD + kk + 4 * q;
                a[i][0] = *(const u32*)p;
                a[i][1] = *(const u32*)(p + 8 * F8G_LD);
                a[i][2] = *(const u32*)(p + 16);
                a[i][3] = *(const u32*)(p + 8 * F8G_LD + 16);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const u8* p = X + (wn + 8 * j + g) * F8G_LD + kk + 4 * q;
                b[j][0] = *(const u32*)p;
                b[j][1] = *(const u32*)(p + 16);
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                    mma_f8(acc[i][j], a[i][0], a[i][1], a[i][2], a[i][3], b[j][0], b[j][1]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = r0 + wm + 16 * i + g + (e >> 1) * 8;
                int t = t0 + wn + 8 * j + 2 * q + (e & 1);
                if (r < rows && t < nb)
                    gm_store(__fmul_rn(__fmul_rn(acc[i][j][e], ws[r]), das[t]), bias, mode, out,
                             (u64)t * ostride, r);
            }
}

// gemm_f8's bits reference: one warp per 16-row x 8-token tile, fragments
// loaded straight from global memory (its own loads, not gemm_f8's), the same
// mma chain over k ascending, the same epilogue. Grid (ceil(nb / 8),
// ceil(rows / 16)) x 32.
extern "C" __global__ void __launch_bounds__(32)
gemm_ref_f8(const u8* __restrict__ w, int rows, int K, const u8* __restrict__ qa,
            const float* __restrict__ das, int nb, const float* __restrict__ ws,
            const float* __restrict__ bias, int mode, float* __restrict__ out, u64 ostride) {
    int lane = threadIdx.x, g = lane >> 2, q = lane & 3;
    int tb = blockIdx.x * 8, rb = blockIdx.y * 16;
    int ra = rb + g, rc = rb + g + 8, tn = tb + g;
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < K; k += 32) {
        int ka = k + 4 * q, kb = k + 16 + 4 * q;
        u32 a0 = ra < rows ? *(const u32*)(w + (u64)ra * K + ka) : 0u;
        u32 a1 = rc < rows ? *(const u32*)(w + (u64)rc * K + ka) : 0u;
        u32 a2 = ra < rows ? *(const u32*)(w + (u64)ra * K + kb) : 0u;
        u32 a3 = rc < rows ? *(const u32*)(w + (u64)rc * K + kb) : 0u;
        u32 b0 = tn < nb ? *(const u32*)(qa + (u64)tn * K + ka) : 0u;
        u32 b1 = tn < nb ? *(const u32*)(qa + (u64)tn * K + kb) : 0u;
        mma_f8(d, a0, a1, a2, a3, b0, b1);
    }
    for (int e = 0; e < 4; e++) {
        int r = rb + g + (e >> 1) * 8, t = tb + 2 * q + (e & 1);
        if (r < rows && t < nb)
            gm_store(__fmul_rn(__fmul_rn(d[e], ws[r]), das[t]), bias, mode, out, (u64)t * ostride, r);
    }
}
