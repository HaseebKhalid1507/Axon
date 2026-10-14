
// ---------------------------------------------------------------------------
// Decode kernels: one token through a Qwen2 model (src/cuda/qwen2.rs).
//
// Numerics. Matrix-vector products quantize the activation exactly as the
// CPU does (quant.rs quantize_q8_k / quantize_q8_0: same scale, same
// round-half-even, same first-largest-magnitude rule), so the weights meet
// the same 8-bit activations, in integer dot products (dp4a here, i8
// multiply-adds there) that are exact. What differs from the CPU is how the
// float scaling and sums are split: a K-quant block's sub-block sums are
// combined in integers and scaled once per block (the CPU: per 32 values),
// and the blocks are added in a fused chain ("Row dot products" below);
// likewise RMSNorm's sum of squares, scores and value rows.
// Every reduction has a fixed order (no atomics), so results are identical
// run to run.
//
// Per-token inputs (token id, position) are read from a small device buffer,
// `P` = {token, pos}, so one captured CUDA graph replays for every token.

#define INFINITY __int_as_float(0x7f800000)
#define NAN_F __int_as_float(0x7fffffff)

// Programmatic dependent launch (sm_90+, Gpu::launch_with): a kernel may be
// started before the previous one on the stream has finished. Every kernel
// launched that way calls pdl_wait() before touching global memory, which
// returns once the previous kernel has completed and its writes are
// visible; then pdl_launch() lets the next kernel's blocks be scheduled
// once all of this one's have started. Without PDL both are no-ops.
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_launch() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
#define PDL_START() \
    do {            \
        pdl_wait(); \
        pdl_launch(); \
    } while (0)

// Pull [p, p + bytes) into L2 (the block's threads share the lines). The
// GEMVs do it for their rows before pdl_wait: weights don't depend on the
// previous kernel, so the first blocks stream them in while it finishes.
__device__ __forceinline__ void prefetch_l2(const u8* p, u64 bytes) {
    for (u64 off = threadIdx.x * 128; off < bytes; off += 128 * blockDim.x)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p + off));
}

#define P_TOKEN 0
#define P_POS 1

// Attention: positions per tile (= threads per attention block).
#define ATT_TILE 128
// Largest GQA group (query heads per kv head) the attention kernel handles.
#define MAX_GROUP 8

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// f32 <-> IEEE half bits, round to nearest even (as quant::f32_to_f16).
__device__ __forceinline__ unsigned short f2h(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}
__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// KvElem for Half::from_f32: saturate to +-65504 (NaN passes: comparisons
// with NaN are false), then round to f16.
__device__ __forceinline__ float kv_sat(float x) {
    return x > 65504.0f ? 65504.0f : (x < -65504.0f ? -65504.0f : x);
}
__device__ __forceinline__ unsigned short kv_half(float x) { return f2h(kv_sat(x)); }

// ---------------------------------------------------------------------------
// Quantized activations. For n values (np = n rounded up to 256), one buffer:
//   qk  i8[np]        Q8_K values (if n % 256 == 0)
//   q0  i8[np]        Q8_0 values
//   dk  f32[np/256]   Q8_K scales
//   b32 i32[np/32]    Q8_K sums of each 32 values (sums of two CPU bsums)
//   d0  f32[np/32]    Q8_0 scales (f16-rounded, as the CPU stores them)
struct Act {
    const signed char* qk;
    const signed char* q0;
    const float* dk;
    const int* b32;
    const float* d0;
};

__device__ __forceinline__ int act_np(int n) { return (n + 255) & ~255; }

__device__ __forceinline__ Act act_view(const u8* a, int n) {
    int np = act_np(n);
    Act v;
    v.qk = (const signed char*)a;
    v.q0 = (const signed char*)(a + np);
    v.dk = (const float*)(a + 2 * np);
    v.b32 = (const int*)(a + 2 * np + np / 64);
    v.d0 = (const float*)(a + 2 * np + np / 64 + np / 8);
    return v;
}

// One warp quantizes 256-block `blk` of a vector of n values: lane l holds
// values blk*256 + 8l .. +8 in v (garbage past n). want_k: the Q8_K view
// (only called with it when n % 256 == 0); the Q8_0 view always.
__device__ void quant_warp(const float v[8], int n, int blk, u8* act, bool want_k) {
    int lane = threadIdx.x & 31;
    int np = act_np(n);
    int e0 = blk * 256 + lane * 8;
    bool valid = e0 < n; // n % 32 == 0: whole 4-lane groups are in or out
    signed char* q0 = (signed char*)(act + np);
    float* d0 = (float*)(act + 2 * np + np / 64 + np / 8);
    // Q8_0: 32-value blocks = 4 lanes.
    float am = 0.0f;
    for (int i = 0; i < 8; i++) am = fmaxf(am, valid ? fabsf(v[i]) : 0.0f);
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    if (valid) {
        float d = h2f(f2h(__fdiv_rn(am, 127.0f)));
        float id = am != 0.0f ? __fdiv_rn(127.0f, am) : 0.0f;
        u32 w[2] = {0, 0};
        for (int i = 0; i < 8; i++) {
            int q = __float2int_rn(__fmul_rn(v[i], id));
            q = q < -128 ? -128 : (q > 127 ? 127 : q);
            w[i >> 2] |= (u32)(q & 0xff) << (8 * (i & 3));
        }
        *(uint2*)(q0 + e0) = make_uint2(w[0], w[1]);
        if ((lane & 3) == 0) d0[e0 / 32] = d;
    }
    if (!want_k) return;
    // Q8_K: the whole warp's 256 values.
    signed char* qk = (signed char*)act;
    float* dk = (float*)(act + 2 * np);
    int* b32 = (int*)(act + 2 * np + np / 64);
    float amax = 0.0f;
    for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    u32 w[2] = {0, 0};
    int s = 0;
    float d = 0.0f;
    if (amax != 0.0f) {
        // The first value of largest magnitude, signed.
        int first = 1 << 30;
        float cand = 0.0f;
        for (int i = 7; i >= 0; i--)
            if (fabsf(v[i]) == amax) {
                first = lane * 8 + i;
                cand = v[i];
            }
        int best = first;
        for (int o = 16; o > 0; o >>= 1) best = min(best, __shfl_xor_sync(0xffffffffu, best, o));
        float mx = __shfl_sync(0xffffffffu, cand, best >> 3);
        float iscale = __fdiv_rn(-127.0f, mx);
        for (int i = 0; i < 8; i++) {
            int q = __float2int_rn(__fmul_rn(iscale, v[i]));
            q = q > 127 ? 127 : q;
            s += q;
            w[i >> 2] |= (u32)(q & 0xff) << (8 * (i & 3));
        }
        d = __fdiv_rn(1.0f, iscale);
    }
    *(uint2*)(qk + e0) = make_uint2(w[0], w[1]);
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    if ((lane & 3) == 0) b32[e0 / 32] = s;
    if (lane == 0) dk[blk] = d;
}

// Block-wide sum over blockDim.x threads (a multiple of 32, at most 1024),
// fixed order; every thread gets the result. `sh` holds 32 floats.
__device__ float block_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = lane < nw ? sh[lane] : 0.0f;
    t = warp_sum(t);
    return t;
}

// out = rmsnorm(x) * g (model.rs rmsnorm_into's ops), quantized into act;
// also stored as f32 in y if y is non-null. One block of 1024 threads.
// The body of rmsnorm_quant, shared with the batched rmsnorm_quant_b
// (prefill.cu): one block per vector.
__device__ void rmsnorm_quant_dev(const float* __restrict__ x, const float* __restrict__ g,
                                  float eps, int n, int want_k, u8* __restrict__ act,
                                  float* __restrict__ y) {
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss = __fadd_rn(ss, __fmul_rn(x[i], x[i]));
    ss = block_sum(ss, sh);
    float ms = __fdiv_rn(ss, (float)n);
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ms, eps)));
    int w = threadIdx.x >> 5, nw = blockDim.x >> 5, lane = threadIdx.x & 31;
    for (int blk = w; blk * 256 < n; blk += nw) {
        float v[8];
        int e0 = blk * 256 + lane * 8;
        for (int i = 0; i < 8; i++) {
            int e = e0 + i;
            v[i] = e < n ? __fmul_rn(__fmul_rn(x[e], inv), g[e]) : 0.0f;
            if (y && e < n) y[e] = v[i];
        }
        quant_warp(v, n, blk, act, want_k);
    }
}

extern "C" __global__ void rmsnorm_quant(const float* __restrict__ x, const float* __restrict__ g,
                                         float eps, int n, int want_k, u8* __restrict__ act,
                                         float* __restrict__ y) {
    PDL_START();
    rmsnorm_quant_dev(x, g, eps, n, want_k, act, y);
}

// x (n values) quantized into act. Grid: ceil(n / 256) blocks of 32 threads.
extern "C" __global__ void quantize(const float* __restrict__ x, int n, int want_k,
                                    u8* __restrict__ act) {
    PDL_START();
    int lane = threadIdx.x & 31, blk = blockIdx.x;
    float v[8];
    for (int i = 0; i < 8; i++) {
        int e = blk * 256 + lane * 8 + i;
        v[i] = e < n ? x[e] : 0.0f;
    }
    quant_warp(v, n, blk, act, want_k);
}

// x = row P[token] of the embedding (type qt, row_bytes per row), dequantized
// bit-exactly. Grid: ceil(n / 256) x 256 threads. The step's first kernel,
// launched without PDL (its predecessor is the copy that writes P), so no
// PDL_START: nothing to wait for, and without launch_dependents the next
// kernel's blocks start only once this one has finished.
extern "C" __global__ void embed(int qt, const u8* __restrict__ w, u64 row_bytes, int n,
                                 const u32* __restrict__ P, float* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = deq(qt, w + (u64)P[P_TOKEN] * row_bytes, i);
}

// RoPE angles of this token, once for all layers: rope[i] = (sin, cos) of
// P[pos] * freq[i] in f64, rounded to f32, as model.rs rope_table does with
// Rust's f64 sin/cos. CUDA's f64 sincos is within 2 ulp (of f64), so the f32
// values are the same in practice (equal in the tests up to position
// 32767) but not guaranteed bit-identical. One block of `half` threads.
extern "C" __global__ void rope_table(const double* __restrict__ freq, int half,
                                      const u32* __restrict__ P, float2* __restrict__ rope) {
    PDL_START();
    int i = threadIdx.x;
    if (i < half) {
        double s, c;
        sincos((double)P[P_POS] * freq[i], &s, &c);
        rope[i] = make_float2((float)s, (float)c);
    }
}

// ---------------------------------------------------------------------------
// Row dot products, in the canonical order every GPU product follows
// (decode GEMV, the batched gemvn / gemvm / gemm2, the gemv_ref reference):
// per output, per block b of the row, in order b = 0, 1, ..:
//   Q4_K: S = sum_j sc_j sumi_j, M = sum_j m_j bsum_j (integers, exact:
//         sumi_j the integer dot of 32-value sub-block j with the Q8_K
//         values, bsum_j the sum of those 32 values), then
//         v = d S - dmin M (fmul, fmul, fsub; S and M rounded to f32 by
//         cvt.rn: |S| < 2^25 may round, M < 2^22 is exact), acc = fma(dk, v, acc)
//   Q6_K: S = sum_j sc_j sumi_j over the 16 sub-blocks of 16 values,
//         acc = fma(dk * d, f32(S), acc)
//   Q8_0, Q5_0 (32-value blocks): s the integer dot,
//         acc = fma(dw * d0, f32(s), acc)   (|s| < 2^20: exact)
// with acc starting at 0. The integers are exact however they are summed
// (dp4a here, mma in the batched kernels), so only the float ops - five or
// six per block, one fma chain per output - fix the bits. This is not the
// CPU's order (quant.rs sums 32-value scaled pieces in floats): GPU and CPU
// agree to f32 rounding, GPU decode and GPU batched passes bit for bit.
//
// The GEMV: a block of GEMV_THREADS threads computes R rows (1, 2) of one
// matrix for T tokens (1 here; the dp4a gemvn of prefill.cu uses more): its
// lane groups (8 lanes per 256-value block, 4 per 32-value block) stride
// over the rows' blocks, each group loading a block of every row before
// using any, so a thread has R weight loads in flight. A group adds its
// lanes' integer shares by shuffles, and its first lane writes the block's
// (scale, value) pair to shared memory; then one thread per output runs
// the fma chain over the blocks in order. Rows of more blocks than the
// buffer holds go in chunks.

#define GEMV_THREADS 128
// Blocks per SM the GEMV kernels are compiled for (registers capped so
// that many fit): more blocks, more weight loads in flight.
#define GEMV_MIN_BLOCKS 8
// (scale, value) pairs per chunk, all outputs: 4 KiB of shared memory.
#define GEMV_PAIRS 512

__device__ __forceinline__ u32 byte_of(uint4 h, int j) {
    // header bytes 4..15 are the 12 scale bytes: scales[j] = byte 4 + j.
    int k = 4 + j;
    u32 w = k < 8 ? h.y : (k < 12 ? h.z : h.w);
    return (w >> (8 * (k & 3))) & 0xff;
}

// quant.rs get_scale_min_k4 on a header held in a uint4.
__device__ __forceinline__ void scale_min_h(uint4 h, int j, int* sc, int* m) {
    if (j < 4) {
        *sc = byte_of(h, j) & 63;
        *m = byte_of(h, j + 4) & 63;
    } else {
        *sc = (byte_of(h, j + 4) & 0xf) | ((byte_of(h, j - 4) >> 6) << 4);
        *m = (byte_of(h, j + 4) >> 4) | ((byte_of(h, j) >> 6) << 4);
    }
}

// 8 bytes at a 2-byte aligned address: two aligned words, or three
// funnel-shifted (Q6_K/Q8_0/Q5_0 blocks are only 2-byte aligned).
__device__ __forceinline__ uint2 ld8_a2(const u8* p) {
    if (((size_t)p & 3) == 0) return make_uint2(__ldg((const u32*)p), __ldg((const u32*)(p + 4)));
    const u32* w = (const u32*)(p - 2);
    u32 a = __ldg(w), b = __ldg(w + 1), c = __ldg(w + 2);
    return make_uint2(__funnelshift_r(a, b, 16), __funnelshift_r(b, c, 16));
}

__device__ __forceinline__ u32 ld4_a2(const u8* p) {
    const unsigned short* s = (const unsigned short*)p;
    return (u32)__ldg(s) | ((u32)__ldg(s + 1) << 16);
}

// The high bits b0..b3 of x moved to bit 4 of bytes 0..3.
__device__ __forceinline__ u32 spread4(u32 x) {
    return ((x << 4) & 0x10) | ((x << 11) & 0x1000) | ((x << 18) & 0x100000) |
           ((x << 25) & 0x10000000);
}

// The f32 of an integer, cvt.rn (round to nearest even, as the CPU's `as`).
__device__ __forceinline__ float i2f_rn(int i) { return __int2float_rn(i); }

// Pair (scale, value) of output (r, j) of block b of a chunk starting at
// block base: the chain does acc = fma(scale, value, acc).
#define GV_PAIR(sh, CAP, r, j, T, b, base) (sh)[((r) * (T) + (j)) * (CAP) + (b) - (base)]

// Q4_K (quant.rs dot_q4_k_q8's integers): lane p of a group takes 16 bytes
// of qs, 32 nibbles: 16 of sub-block 2c and 16 of 2c + 1 (c = p / 2), and
// the min terms of sub-blocks 2c, 2c + 1 (even lanes). Token j's
// activations at act + min(j, zmax) * astride.
template <int R, int T, int CAP>
__device__ __forceinline__ void blk_q4_k(const u8* const* rows, int base, int end, const u8* act,
                                         u64 astride, int n, int zmax, float2* sh) {
    int t = threadIdx.x, g = t >> 3, p = t & 7, c = p >> 1, hf = p & 1;
    u32 mask = 0xffu << (t & 24);
    for (int b = base + g; b < end; b += GEMV_THREADS / 8) {
        uint4 h[R], q[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const u8* blk = rows[r] + (u64)b * 144;
            h[r] = __ldg((const uint4*)blk);
            q[r] = __ldg((const uint4*)(blk + 16 + 16 * p));
        }
#pragma unroll
        for (int j = 0; j < T; j++) {
            Act a = act_view(act + (u64)min(j, zmax) * astride, n);
            const signed char* xb = a.qk + b * 256 + 64 * c + 16 * hf;
            int4 xl = *(const int4*)xb;
            int4 xh = *(const int4*)(xb + 32);
            int bs0 = a.b32[b * 8 + 2 * c], bs1 = a.b32[b * 8 + 2 * c + 1];
            float dk = a.dk[b];
#pragma unroll
            for (int r = 0; r < R; r++) {
                int s1 = 0, s2 = 0;
                s1 = __dp4a((int)(q[r].x & 0x0f0f0f0f), xl.x, s1);
                s1 = __dp4a((int)(q[r].y & 0x0f0f0f0f), xl.y, s1);
                s1 = __dp4a((int)(q[r].z & 0x0f0f0f0f), xl.z, s1);
                s1 = __dp4a((int)(q[r].w & 0x0f0f0f0f), xl.w, s1);
                s2 = __dp4a((int)((q[r].x >> 4) & 0x0f0f0f0f), xh.x, s2);
                s2 = __dp4a((int)((q[r].y >> 4) & 0x0f0f0f0f), xh.y, s2);
                s2 = __dp4a((int)((q[r].z >> 4) & 0x0f0f0f0f), xh.z, s2);
                s2 = __dp4a((int)((q[r].w >> 4) & 0x0f0f0f0f), xh.w, s2);
                int sc1, m1, sc2, m2;
                scale_min_h(h[r], 2 * c, &sc1, &m1);
                scale_min_h(h[r], 2 * c + 1, &sc2, &m2);
                int S = sc1 * s1 + sc2 * s2;
                int M = hf ? 0 : m1 * bs0 + m2 * bs1;
#pragma unroll
                for (int o = 1; o < 8; o <<= 1) {
                    S += __shfl_xor_sync(mask, S, o);
                    M += __shfl_xor_sync(mask, M, o);
                }
                if (p == 0) {
                    float d = h2f(h[r].x & 0xffff), dmin = h2f(h[r].x >> 16);
                    float v = __fsub_rn(__fmul_rn(d, i2f_rn(S)), __fmul_rn(dmin, i2f_rn(M)));
                    GV_PAIR(sh, CAP, r, j, T, b, base) = make_float2(dk, v);
                }
            }
        }
    }
}

// Q6_K (dot_q6_k_q8's integers): lane p of a group takes half hh = p / 4,
// positions l0..l0+8 (l0 = 8 (p % 4)) of each of that half's 4 quarters.
template <int R, int T, int CAP>
__device__ __forceinline__ void blk_q6_k(const u8* const* rows, int base, int end, const u8* act,
                                         u64 astride, int n, int zmax, float2* sh) {
    int t = threadIdx.x, g = t >> 3, p = t & 7, hh = p >> 2, l0 = (p & 3) * 8;
    u32 mask = 0xffu << (t & 24);
    for (int b = base + g; b < end; b += GEMV_THREADS / 8) {
        uint2 la[R], lb[R], qh[R];
        float dw[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const u8* blk = rows[r] + (u64)b * 210;
            la[r] = ld8_a2(blk + 64 * hh + l0);      // ql[l]: quarters 0 (low), 2 (high)
            lb[r] = ld8_a2(blk + 64 * hh + 32 + l0); // ql[l + 32]: quarters 1, 3
            qh[r] = ld8_a2(blk + 128 + 32 * hh + l0);
            dw[r] = h2f(rd_u16(blk + 208));
        }
#pragma unroll
        for (int j = 0; j < T; j++) {
            Act a = act_view(act + (u64)min(j, zmax) * astride, n);
            const signed char* xb = a.qk + b * 256 + 128 * hh + l0;
            uint2 x[4];
#pragma unroll
            for (int k = 0; k < 4; k++) x[k] = *(const uint2*)(xb + 32 * k);
            float dk = a.dk[b];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const u8* blk = rows[r] + (u64)b * 210;
                const signed char* sc = (const signed char*)(blk + 192 + 8 * hh + (l0 >> 4));
                int S = 0;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    uint2 l = (k & 1) ? lb[r] : la[r];
                    u32 lo0 = k < 2 ? (l.x & 0x0f0f0f0f) : ((l.x >> 4) & 0x0f0f0f0f);
                    u32 lo1 = k < 2 ? (l.y & 0x0f0f0f0f) : ((l.y >> 4) & 0x0f0f0f0f);
                    u32 w0 = __vsub4(lo0 | (((qh[r].x >> (2 * k)) & 0x03030303) << 4), 0x20202020);
                    u32 w1 = __vsub4(lo1 | (((qh[r].y >> (2 * k)) & 0x03030303) << 4), 0x20202020);
                    int s = __dp4a((int)w0, (int)x[k].x, 0);
                    s = __dp4a((int)w1, (int)x[k].y, s);
                    S += (int)sc[2 * k] * s;
                }
#pragma unroll
                for (int o = 1; o < 8; o <<= 1) S += __shfl_xor_sync(mask, S, o);
                if (p == 0)
                    GV_PAIR(sh, CAP, r, j, T, b, base) =
                        make_float2(__fmul_rn(dk, dw[r]), i2f_rn(S));
            }
        }
    }
}

// Q8_0 (dot_q8_0_q8): 4 lanes per block, 8 values each.
template <int R, int T, int CAP>
__device__ __forceinline__ void blk_q8_0(const u8* const* rows, int base, int end, const u8* act,
                                         u64 astride, int n, int zmax, float2* sh) {
    int t = threadIdx.x, g = t >> 2, p = t & 3;
    u32 mask = 0xfu << (t & 28);
    for (int b = base + g; b < end; b += GEMV_THREADS / 4) {
        uint2 q[R];
        float dw[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            q[r] = ld8_a2(rows[r] + (u64)b * 34 + 2 + 8 * p);
            dw[r] = h2f(rd_u16(rows[r] + (u64)b * 34));
        }
#pragma unroll
        for (int j = 0; j < T; j++) {
            Act a = act_view(act + (u64)min(j, zmax) * astride, n);
            uint2 x = *(const uint2*)(a.q0 + b * 32 + 8 * p);
            float d0 = a.d0[b];
#pragma unroll
            for (int r = 0; r < R; r++) {
                int s = __dp4a((int)q[r].x, (int)x.x, 0);
                s = __dp4a((int)q[r].y, (int)x.y, s);
                s += __shfl_xor_sync(mask, s, 1);
                s += __shfl_xor_sync(mask, s, 2);
                if (p == 0)
                    GV_PAIR(sh, CAP, r, j, T, b, base) =
                        make_float2(__fmul_rn(dw[r], d0), i2f_rn(s));
            }
        }
    }
}

// Q5_0 (dot_q5_0_q8): 4 lanes per block; lane p has values 8p..8p+8: low
// nibbles of qs[8p..] for p < 2, high nibbles of qs[8(p-2)..] after, high
// bits 8p.. of qh.
template <int R, int T, int CAP>
__device__ __forceinline__ void blk_q5_0(const u8* const* rows, int base, int end, const u8* act,
                                         u64 astride, int n, int zmax, float2* sh) {
    int t = threadIdx.x, g = t >> 2, p = t & 3;
    u32 mask = 0xfu << (t & 28);
    for (int b = base + g; b < end; b += GEMV_THREADS / 4) {
        u32 qh[R];
        uint2 q[R];
        float dw[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const u8* blk = rows[r] + (u64)b * 22;
            qh[r] = ld4_a2(blk + 2) >> (8 * p);
            q[r] = ld8_a2(blk + 6 + 8 * (p & 1));
            dw[r] = h2f(rd_u16(blk));
        }
#pragma unroll
        for (int j = 0; j < T; j++) {
            Act a = act_view(act + (u64)min(j, zmax) * astride, n);
            uint2 x = *(const uint2*)(a.q0 + b * 32 + 8 * p);
            float d0 = a.d0[b];
#pragma unroll
            for (int r = 0; r < R; r++) {
                u32 n0 = p < 2 ? (q[r].x & 0x0f0f0f0f) : ((q[r].x >> 4) & 0x0f0f0f0f);
                u32 n1 = p < 2 ? (q[r].y & 0x0f0f0f0f) : ((q[r].y >> 4) & 0x0f0f0f0f);
                u32 w0 = __vsub4(n0 | spread4(qh[r] & 0xf), 0x10101010);
                u32 w1 = __vsub4(n1 | spread4((qh[r] >> 4) & 0xf), 0x10101010);
                int s = __dp4a((int)w0, (int)x.x, 0);
                s = __dp4a((int)w1, (int)x.y, s);
                s += __shfl_xor_sync(mask, s, 1);
                s += __shfl_xor_sync(mask, s, 2);
                if (p == 0)
                    GV_PAIR(sh, CAP, r, j, T, b, base) =
                        make_float2(__fmul_rn(dw[r], d0), i2f_rn(s));
            }
        }
    }
}

// The (scale, value) pairs of dot_rows_t's chunks (one buffer for every
// call of a kernel).
__shared__ float2 gv_pairs[GEMV_PAIRS];

// acc = fma(c[b].x, c[b].y, acc) for b = 0..nb-1, in order (loads ahead).
__device__ __forceinline__ float gv_chain(const float2* c, int nb, float acc) {
    int b = 0;
    for (; b + 4 <= nb; b += 4) {
        float2 e0 = c[b], e1 = c[b + 1], e2 = c[b + 2], e3 = c[b + 3];
        acc = __fmaf_rn(e0.x, e0.y, acc);
        acc = __fmaf_rn(e1.x, e1.y, acc);
        acc = __fmaf_rn(e2.x, e2.y, acc);
        acc = __fmaf_rn(e3.x, e3.y, acc);
    }
    for (; b < nb; b++) acc = __fmaf_rn(c[b].x, c[b].y, acc);
    return acc;
}

// rows[0..R) of a matrix of type qt and n columns times T tokens' (token j
// at act + min(j, zmax) * astride): thread i < R T runs output i's chain
// (r = i / T, j = i % T) and returns it; with T = 1 (decode) the lanes of
// warp 0 also get every y[r] (shuffled from lane r). Every thread of the block must call it;
// `again` if the kernel called it before (whose chains may still be
// reading the pairs).
template <int R, int T>
__device__ __forceinline__ float dot_rows_t(int qt, const u8* const* rows, int n, const u8* act,
                                            u64 astride, int zmax, float* y, bool again = false) {
    const int RT = R * T, CAP = GEMV_PAIRS / RT;
    float2* sh = gv_pairs;
    bool kq = qt == QT_Q4_K || qt == QT_Q6_K;
    int nb = kq ? n >> 8 : n >> 5;
    int i = threadIdx.x;
    float acc = 0.0f;
    if (again) __syncthreads();
    for (int base = 0; base < nb; base += CAP) {
        if (base > 0) __syncthreads(); // the last chunk's chains are done
        int end = min(nb, base + CAP);
        switch (qt) {
        case QT_Q4_K: blk_q4_k<R, T, CAP>(rows, base, end, act, astride, n, zmax, sh); break;
        case QT_Q6_K: blk_q6_k<R, T, CAP>(rows, base, end, act, astride, n, zmax, sh); break;
        case QT_Q8_0: blk_q8_0<R, T, CAP>(rows, base, end, act, astride, n, zmax, sh); break;
        default: blk_q5_0<R, T, CAP>(rows, base, end, act, astride, n, zmax, sh); break; // QT_Q5_0
        }
        __syncthreads();
        if (i < RT) acc = gv_chain(sh + i * CAP, end - base, acc);
    }
    if (T == 1 && y && i < 32) {
#pragma unroll
        for (int r = 0; r < R; r++) y[r] = __shfl_sync(0xffffffffu, acc, r);
    }
    return acc;
}

// The one-token form, activations in a: y[0..R) in warp 0 (thread 0 is what
// the kernels use).
template <int R>
__device__ __forceinline__ void dot_rows(int qt, const u8* const* rows, int n, const u8* act,
                                         float* y, bool again = false) {
    dot_rows_t<R, 1>(qt, rows, n, act, 0, 0, y, again);
}

// Rows r0 and r1 of one matrix (r1 may equal r0 - then it's computed once).
__device__ __forceinline__ void dot_pair(int qt, const u8* w, u64 rb, int r0, int r1, int n,
                                         const u8* act, float* y) {
    if (r1 == r0) {
        const u8* rows[1] = {w + (u64)r0 * rb};
        dot_rows<1>(qt, rows, n, act, y);
        y[1] = y[0];
    } else {
        const u8* rows[2] = {w + (u64)r0 * rb, w + (u64)r1 * rb};
        dot_rows<2>(qt, rows, n, act, y);
    }
}

// y = W x for one matrix, two rows per block: out[r] = y[r] (+ bias[r]) for
// mode 0, out[r] += y[r] (the residual add) for mode 1. Grid ceil(rows / 2).
extern "C" __global__ void __launch_bounds__(GEMV_THREADS, GEMV_MIN_BLOCKS)
gemv(int qt, const u8* __restrict__ w, u64 row_bytes, int rows, int n, const u8* __restrict__ act,
     const float* __restrict__ bias, int mode, float* __restrict__ out) {
    int r0 = 2 * blockIdx.x, r1 = min(r0 + 1, rows - 1);
    prefetch_l2(w + (u64)r0 * row_bytes, (u64)(r1 - r0 + 1) * row_bytes);
    PDL_START();
    float y[2];
    dot_pair(qt, w, row_bytes, r0, r1, n, act, y);
    if (threadIdx.x == 0) {
        for (int i = 0; i < 1 + (r1 != r0); i++) {
            int r = r0 + i;
            float v = y[i];
            if (bias) v = __fadd_rn(v, bias[r]);
            out[r] = mode == 1 ? __fadd_rn(out[r], v) : v;
        }
    }
}

// `gemv` with one row per block (grid rows): twice the blocks, so the last
// wave is a smaller share of the time. The host uses it for rows of 4 KiB
// or more, which here means ffn_down (wide rows, few of them); every other
// matrix goes through gemv.
// Same bits as gemv (a row's sums don't depend on its block-mates).
extern "C" __global__ void __launch_bounds__(GEMV_THREADS, GEMV_MIN_BLOCKS)
gemv1(int qt, const u8* __restrict__ w, u64 row_bytes, int rows, int n, const u8* __restrict__ act,
      const float* __restrict__ bias, int mode, float* __restrict__ out) {
    int r = blockIdx.x;
    prefetch_l2(w + (u64)r * row_bytes, row_bytes);
    PDL_START();
    float y[1];
    const u8* rr[1] = {w + (u64)r * row_bytes};
    dot_rows<1>(qt, rr, n, act, y);
    if (threadIdx.x == 0) {
        float v = y[0];
        if (bias) v = __fadd_rn(v, bias[r]);
        out[r] = mode == 1 ? __fadd_rn(out[r], v) : v;
    }
}

// One head's RMSNorm (model.rs rmsnorm_heads: x * inv * g) then RoPE, by
// one warp, for qwen3's q/k norms: lane l holds pairs j = l + 32 i (i <
// hd / 64, a = x[j], b = x[j + hd/2]) in a[i], b[i] and gets them back
// normalized and rotated (NEOX, rope[j]). The sum of squares is fixed: per
// lane a*a + b*b over its pairs in order (from 0), then warp_sum. Shared by
// the decode attention (attn_split_t's prologue) and rope_kv_norm_b
// (prefill.cu), so both give the same bits.
__device__ __forceinline__ void warp_norm_rope(float* a, float* b, int hd,
                                               const float* __restrict__ g, float eps,
                                               const float2* __restrict__ rope) {
    int half = hd >> 1, np = half >> 5, lane = threadIdx.x & 31;
    float ss = 0.0f;
    for (int i = 0; i < np; i++) ss = __fadd_rn(ss, __fadd_rn(__fmul_rn(a[i], a[i]), __fmul_rn(b[i], b[i])));
    ss = warp_sum(ss);
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)hd), eps)));
    for (int i = 0; i < np; i++) {
        int j = lane + 32 * i;
        float y0 = __fmul_rn(__fmul_rn(a[i], inv), g[j]);
        float y1 = __fmul_rn(__fmul_rn(b[i], inv), g[j + half]);
        float2 sc = rope[j];
        a[i] = __fsub_rn(__fmul_rn(y0, sc.y), __fmul_rn(y1, sc.x));
        b[i] = __fadd_rn(__fmul_rn(y0, sc.x), __fmul_rn(y1, sc.y));
    }
}

// Queries, keys and values for one token, with biases, RoPE and the cache
// store fused. Work items, one per block: nq/2 query row pairs, nk/2 key
// row pairs, nk/2 value row pairs (nk even). A query or key pair is rows
// (i, i + hd/2) of one head: the two halves RoPE rotates together
// (model.rs rope_head, NEOX style), with (sin, cos) of P[pos] * freq[i]
// from the rope_table kernel. Rotated queries go to q (f32); keys
// and values, rounded to f16, to the caches' row P[pos] (nk halves a row).
//
// F32 (the q8 cache's qkv_rope_f32): keys and values go to kf / vf (nk
// floats each, this token only) unrounded instead; attn_split_q8 quantizes
// them into the cache.
//
// With per-head q/k norms (kraw non-null: qwen3) a head's RoPE needs all its
// rows first, so this stores the raw products instead (q in q, keys in
// kraw, f32: the q8 kernel passes kf; values as usual); the decode
// attention then normalizes and rotates them as it loads them
// (attn_split_t's QkNorm): no extra launch, no handshake between blocks.
template <bool F32>
__device__ __forceinline__ void
qkv_rope_t(int qtq, const u8* __restrict__ wq, u64 rbq, int qtk, const u8* __restrict__ wk, u64 rbk,
           int qtv, const u8* __restrict__ wv, u64 rbv, const float* __restrict__ bq,
           const float* __restrict__ bk, const float* __restrict__ bv, int nq, int nk, int hd, int n,
           const u8* __restrict__ act, const float2* __restrict__ rope, const u32* __restrict__ P,
           float* __restrict__ q, void* __restrict__ kc, void* __restrict__ vc, float* kraw) {
    int item = blockIdx.x;
    int half = hd >> 1, npq = nq >> 1, npk = nk >> 1;
    if (item >= npq + npk) {
        prefetch_l2(wv + (u64)(2 * (item - npq - npk)) * rbv, 2 * rbv);
    } else {
        bool isq = item < npq;
        int it = isq ? item : item - npq;
        int r0 = (it / half) * hd + it % half;
        const u8* w = isq ? wq : wk;
        u64 rb = isq ? rbq : rbk;
        prefetch_l2(w + (u64)r0 * rb, rb);
        prefetch_l2(w + (u64)(r0 + half) * rb, rb);
    }
    PDL_START();
    u32 pos = P[P_POS];
    float y[2];
    if (item >= npq + npk) {
        int r0 = 2 * (item - npq - npk);
        dot_pair(qtv, wv, rbv, r0, r0 + 1, n, act, y);
        if (threadIdx.x == 0) {
            for (int i = 0; i < 2; i++) {
                float v = bv ? __fadd_rn(y[i], bv[r0 + i]) : y[i];
                if (F32)
                    ((float*)vc)[r0 + i] = v;
                else
                    ((unsigned short*)vc)[(u64)pos * nk + r0 + i] = kv_half(v);
            }
        }
        return;
    }
    bool isq = item < npq;
    int it = isq ? item : item - npq;
    int r0 = (it / half) * hd + it % half, r1 = r0 + half;
    const float* bias = isq ? bq : bk;
    dot_pair(isq ? qtq : qtk, isq ? wq : wk, isq ? rbq : rbk, r0, r1, n, act, y);
    if (threadIdx.x != 0) return;
    float y0 = y[0], y1 = y[1];
    if (bias) {
        y0 = __fadd_rn(y0, bias[r0]);
        y1 = __fadd_rn(y1, bias[r1]);
    }
    if (kraw) {
        float* raw = isq ? q : kraw;
        raw[r0] = y0;
        raw[r1] = y1;
        return;
    }
    float2 sc = rope[it % half];
    float sn = sc.x, cs = sc.y;
    float o0 = __fsub_rn(__fmul_rn(y0, cs), __fmul_rn(y1, sn));
    float o1 = __fadd_rn(__fmul_rn(y0, sn), __fmul_rn(y1, cs));
    if (isq) {
        q[r0] = o0;
        q[r1] = o1;
    } else if (F32) {
        ((float*)kc)[r0] = o0;
        ((float*)kc)[r1] = o1;
    } else {
        ((unsigned short*)kc)[(u64)pos * nk + r0] = kv_half(o0);
        ((unsigned short*)kc)[(u64)pos * nk + r1] = kv_half(o1);
    }
}

#define QKV_ROPE_ARGS                                                                              \
    int qtq, const u8 *__restrict__ wq, u64 rbq, int qtk, const u8 *__restrict__ wk, u64 rbk,     \
        int qtv, const u8 *__restrict__ wv, u64 rbv, const float *__restrict__ bq,                 \
        const float *__restrict__ bk, const float *__restrict__ bv, int nq, int nk, int hd, int n, \
        const u8 *__restrict__ act, const float2 *__restrict__ rope, const u32 *__restrict__ P,    \
        float *__restrict__ q
#define QKV_ROPE_PASS qtq, wq, rbq, qtk, wk, rbk, qtv, wv, rbv, bq, bk, bv, nq, nk, hd, n, act, rope, P, q

extern "C" __global__ void __launch_bounds__(GEMV_THREADS, GEMV_MIN_BLOCKS)
qkv_rope(QKV_ROPE_ARGS, unsigned short* __restrict__ kc, unsigned short* __restrict__ vc,
         float* __restrict__ kraw) {
    qkv_rope_t<false>(QKV_ROPE_PASS, kc, vc, kraw);
}

extern "C" __global__ void __launch_bounds__(GEMV_THREADS, GEMV_MIN_BLOCKS)
qkv_rope_f32(QKV_ROPE_ARGS, float* kf, float* __restrict__ vf, float* kraw) {
    qkv_rope_t<true>(QKV_ROPE_PASS, kf, vf, kraw);
}

// out[r] = silu(gate_r . x) * (up_r . x) (vmath swiglu1's ops, expf for
// exp). One row of each matrix per block (both loaded together when they
// have the same type).
extern "C" __global__ void __launch_bounds__(GEMV_THREADS, GEMV_MIN_BLOCKS)
gate_up(int qtg, const u8* __restrict__ wg, u64 rbg, int qtu, const u8* __restrict__ wu, u64 rbu,
        int rows, int n, const u8* __restrict__ act, float* __restrict__ out) {
    int r = blockIdx.x;
    prefetch_l2(wg + (u64)r * rbg, rbg);
    prefetch_l2(wu + (u64)r * rbu, rbu);
    PDL_START();
    float y[2];
    if (qtg == qtu) {
        const u8* rr[2] = {wg + (u64)r * rbg, wu + (u64)r * rbu};
        dot_rows<2>(qtg, rr, n, act, y);
    } else {
        const u8* r0[1] = {wg + (u64)r * rbg};
        const u8* r1[1] = {wu + (u64)r * rbu};
        dot_rows<1>(qtg, r0, n, act, y);
        dot_rows<1>(qtu, r1, n, act, y + 1, true);
    }
    if (threadIdx.x == 0)
        out[r] = __fmul_rn(__fdiv_rn(y[0], __fadd_rn(1.0f, expf(-y[0]))), y[1]);
}

// ---------------------------------------------------------------------------
// 8-bit KV cache (AXON_KV=q8, `--kv q8`). A layer's keys are kept as int8
// [pos][kv_dim] with one f32 scale per KG consecutive values (a "group":
// KG = head_dim, one scale per (position, kv head), or 32), at [pos][kv_dim
// / KG]; values likewise. The rule, per group x[0..KG) of a token's rotated
// keys minus the layer's key shift (qwen2.rs: the key bias of the slowest
// RoPE pairs, else 0), or of its values, each value first saturated to
// +-65504 as the f16 cache does:
//   amax = max |x_i|,  d = amax / 127,  q_i = clamp(rne(x_i * (127 / amax)),
//   -127, 127)   (all q_i = 0 if amax = 0; d = NaN if any x_i is NaN),
// f32 ops correctly rounded. Attention reads them as d * q_i, folded in:
// a score is sum over groups of d * (q . k_q) (the group's dot product an fma
// chain over its values in order), a value term (p d) * v_q. The shift is
// never added back: it moves all of a query's scores by the same q . shift,
// which softmax cancels; without it Qwen2's few huge, near-constant key
// channels (|k| up to ~200, the rest ~1-5) set every row's scale, and their
// rounding error (q8: ~0.4% of 200, f16 ~0.05%) times the queries' equally
// large components there moves scores by whole units.
struct Q8Kv {
    const signed char* k;
    const signed char* v;
    const float* ks;
    const float* vs;
    // decode only: this token's keys and values, f32 [2][kv_dim] (qkv_rope_f32),
    // which the block holding its position quantizes into the cache first.
    const float* knew;
    // The layer's key shift [kv_dim] (null: none).
    const float* ksh;
};

// Store one kv head's key and value rows, xs = 2 * HD floats (shared:
// keys, then values), into the cache at element e0 (= pos * kv_dim + kvh *
// HD) by the rule above. Called by every thread of the block (a multiple of
// 32); xs must be complete (synced) on entry; ends with a barrier, so the
// block can read the rows back.
template <int HD, int KG>
__device__ void kv_q8_store(float* xs, const Q8Kv& c, u64 e0, const float* ksh) {
    __shared__ float dsh[2 * HD / KG], ish[2 * HD / KG];
    int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = blockDim.x >> 5;
    if (ksh) {
        for (int i = tid; i < HD; i += blockDim.x) xs[i] = __fsub_rn(xs[i], ksh[i]);
        __syncthreads();
    }
    for (int gi = w; gi < 2 * HD / KG; gi += nw) {
        float m = 0.0f;
        bool nan = false;
        for (int i = lane; i < KG; i += 32) {
            float x = kv_sat(xs[gi * KG + i]);
            m = fmaxf(m, fabsf(x)); // fmaxf drops NaN
            nan |= x != x;
        }
        m = warp_max(m);
        nan = __any_sync(0xffffffffu, nan);
        if (lane == 0) {
            dsh[gi] = nan ? NAN_F : __fdiv_rn(m, 127.0f);
            ish[gi] = m != 0.0f ? __fdiv_rn(127.0f, m) : 0.0f;
        }
    }
    __syncthreads();
    signed char* kq = (signed char*)c.k + e0;
    signed char* vq = (signed char*)c.v + e0;
    for (int i = tid; i < 2 * HD; i += blockDim.x) {
        // NaN * id -> cvt gives 0 (its group's d is NaN).
        int q = __float2int_rn(__fmul_rn(kv_sat(xs[i]), ish[i / KG]));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        if (i < HD)
            kq[i] = (signed char)q;
        else
            vq[i - HD] = (signed char)q;
    }
    if (tid < 2 * HD / KG) {
        float* s = (float*)(tid < HD / KG ? c.ks : c.vs);
        s[e0 / KG + tid % (HD / KG)] = dsh[tid];
    }
    __syncthreads();
}

// The 16 int8 of w as floats (exact).
__device__ __forceinline__ void i8x16(uint4 w, float k[16]) {
    u32 a[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 16; i++) k[i] = (float)(int)(signed char)(a[i >> 2] >> (8 * (i & 3)));
}

// ---------------------------------------------------------------------------
// Decode attention, split over positions (flash-decoding). np = P[pos] + 1
// positions in nsplit = min(max_split, ceil(np / ATT_MIN)) splits of
// `chunk` = ceil(np / nsplit) positions. Block (kv head, split) scores all
// `group` query heads of its kv head over its positions, ATT_TILE at a time
// with an online softmax, and writes per head the unnormalized sum of
// p_t v_t with its max m and sum l; attn_combine merges the splits. Grid
// (n_kv, max_split) whatever the position (a graph replays it): blocks past
// nsplit return at once, and max_split (a few blocks per SM) keeps long
// contexts spread over the whole GPU.

// Fewest positions per split.
#define ATT_MIN 32

__device__ __forceinline__ void att_split(int np, int max_split, int* nsplit, int* chunk) {
    *nsplit = min(max_split, (np + ATT_MIN - 1) / ATT_MIN);
    *chunk = (np + *nsplit - 1) / *nsplit;
}

// qwen3's q/k norms in the decode attention (qn null: none). q holds the
// raw queries and kraw the raw keys of position np - 1 (qkv_rope's norm
// path); kc_out is the K cache (kc) that position's keys are stored in.
struct QkNorm {
    const float* kraw;
    const float* qn;
    const float* kn;
    float eps;
    const float2* rope;
    unsigned short* kc_out;
};

// part_o: [head][split][hd]; part_ml: [head][split][2] (m, l). Head size
// HD 64 or 128, group <= MAX_GROUP (the host checks).
// Block (kvh, s) of a query whose position is np - 1; shared with the
// batched attn_split_b (prefill.cu), which runs it per query token.
// KG = 0: the f16 cache kc / vc; else the q8 cache q8 with KG values per
// scale (kc, vc unused). nm: qwen3's q/k norms (see QkNorm), either cache.
template <int HD, int KG = 0>
__device__ void attn_split_t(const float* __restrict__ q, const unsigned short* __restrict__ kc,
                             const unsigned short* __restrict__ vc, int np, int kvh, int s,
                             int n_heads, int n_kv, float scale, int max_split,
                             float* __restrict__ part_o, float* __restrict__ part_ml,
                             Q8Kv q8 = Q8Kv{}, QkNorm nm = QkNorm{}) {
    const int hd = HD;
    __shared__ float qs[MAX_GROUP * 128];
    __shared__ float ps[MAX_GROUP][ATT_TILE];
    __shared__ float osum[MAX_GROUP * ATT_TILE * 2];
    __shared__ float mrun[MAX_GROUP], lrun[MAX_GROUP], alpha[MAX_GROUP];
    __shared__ __align__(16) unsigned short kh[HD];
    int tid = threadIdx.x;
    int nsplit, chunk;
    att_split(np, max_split, &nsplit, &chunk);
    if (s >= nsplit) return;
    int tbeg = s * chunk, tend = min(np, tbeg + chunk);
    int group = n_heads / n_kv, kv_dim = n_kv * hd;
    if (tbeg >= tend) { // an empty last split (rounding): a partial that weighs 0
        for (int i = tid; i < group * hd; i += ATT_TILE)
            part_o[((u64)(kvh * group + i / hd) * max_split + s) * hd + i % hd] = 0.0f;
        if (tid < group) {
            part_ml[((u64)(kvh * group + tid) * max_split + s) * 2] = -INFINITY;
            part_ml[((u64)(kvh * group + tid) * max_split + s) * 2 + 1] = 0.0f;
        }
        return;
    }
    if constexpr (KG > 0) {
        // Decode: the new token's rows, quantized here (block-uniform test),
        // by the block whose split holds them, before anyone reads them.
        if (q8.knew && np - 1 >= tbeg && np - 1 < tend) {
            __shared__ float xs[2 * HD];
            for (int i = tid; i < 2 * HD; i += ATT_TILE)
                xs[i] = q8.knew[(i < HD ? 0 : kv_dim - HD) + kvh * HD + i];
            __syncthreads();
            if (nm.qn) {
                // qwen3: knew holds the raw keys; normalize and rotate them
                // (as the f16 path below) before quantizing.
                if (tid < 32) {
                    float a[2], b[2];
                    for (int i = 0; i < (HD >> 6); i++) {
                        a[i] = xs[tid + 32 * i];
                        b[i] = xs[tid + 32 * i + HD / 2];
                    }
                    warp_norm_rope(a, b, HD, nm.kn, nm.eps, nm.rope);
                    for (int i = 0; i < (HD >> 6); i++) {
                        xs[tid + 32 * i] = a[i];
                        xs[tid + 32 * i + HD / 2] = b[i];
                    }
                }
                __syncthreads();
            }
            kv_q8_store<HD, KG>(xs, q8, (u64)(np - 1) * kv_dim + kvh * HD,
                                q8.ksh ? q8.ksh + kvh * HD : nullptr);
        }
    }
    // With q/k norms (decode, qwen3): a warp per query head normalizes and
    // rotates it into qs, and on the f16 cache the split holding position
    // np - 1 does its keys too, into the cache and kh; that row is scored
    // from kh (the cache's copy isn't read back by the kernel that wrote it).
    bool kpos = KG == 0 && nm.qn && tbeg <= np - 1 && np - 1 < tend;
    if (nm.qn) {
        int lane = tid & 31, w = tid >> 5, half = hd >> 1;
        for (int g = w; g < group + kpos; g += ATT_TILE / 32) {
            bool isk = g == group;
            const float* x = isk ? nm.kraw + kvh * hd : q + (kvh * group + g) * hd;
            float a[2], b[2];
            for (int i = 0; i < (half >> 5); i++) {
                a[i] = x[lane + 32 * i];
                b[i] = x[lane + 32 * i + half];
            }
            warp_norm_rope(a, b, hd, isk ? nm.kn : nm.qn, nm.eps, nm.rope);
            for (int i = 0; i < (half >> 5); i++) {
                int j = lane + 32 * i;
                if (isk) {
                    unsigned short h0 = kv_half(a[i]), h1 = kv_half(b[i]);
                    kh[j] = h0;
                    kh[j + half] = h1;
                    unsigned short* kr = nm.kc_out + (u64)(np - 1) * kv_dim + kvh * hd;
                    kr[j] = h0;
                    kr[j + half] = h1;
                } else {
                    qs[g * hd + j] = a[i];
                    qs[g * hd + j + half] = b[i];
                }
            }
        }
    } else {
        for (int i = tid; i < group * hd; i += ATT_TILE) qs[i] = q[kvh * group * hd + i];
    }
    if (tid < group) mrun[tid] = -INFINITY, lrun[tid] = 0.0f;
    // Value sums: thread = (row slot rs, dim pair dp), rows i = rs mod nrs.
    int pairs = hd >> 1, nrs = ATT_TILE / pairs, rs = tid / pairs, dp = tid % pairs;
    float o0[MAX_GROUP], o1[MAX_GROUP];
    for (int g = 0; g < MAX_GROUP; g++) o0[g] = o1[g] = 0.0f;
    int lane = tid & 31, wid = tid >> 5;
    __syncthreads();
    for (int t0 = tbeg; t0 < tend; t0 += ATT_TILE) {
        int t = t0 + tid;
        // Scores, thread per position: q_g . k_t * scale for every head g.
        float sc[MAX_GROUP];
        for (int g = 0; g < MAX_GROUP; g++) sc[g] = 0.0f;
        if (KG > 0 && t < tend) {
            u64 e = (u64)t * kv_dim + kvh * HD;
            const uint4* kr = (const uint4*)(q8.k + e);
            const float* sr = q8.ks + e / (KG > 0 ? KG : 1);
#pragma unroll
            for (int j = 0; j < (KG > 0 ? HD / KG : 0); j++) {
                float a[MAX_GROUP];
#pragma unroll
                for (int g = 0; g < MAX_GROUP; g++) a[g] = 0.0f;
#pragma unroll
                for (int d16 = j * KG / 16; d16 < (j + 1) * KG / 16; d16++) {
                    float k[16];
                    i8x16(kr[d16], k);
#pragma unroll
                    for (int g = 0; g < MAX_GROUP; g++) {
                        if (g < group) {
                            const float* qq = qs + g * HD + d16 * 16;
#pragma unroll
                            for (int i = 0; i < 16; i++) a[g] = __fmaf_rn(qq[i], k[i], a[g]);
                        }
                    }
                }
                float d = sr[j];
#pragma unroll
                for (int g = 0; g < MAX_GROUP; g++)
                    if (g < group) sc[g] = __fmaf_rn(d, a[g], sc[g]);
            }
        } else if (KG == 0 && t < tend) {
            const uint4* kr = kpos && t == np - 1
                                  ? (const uint4*)kh
                                  : (const uint4*)(kc + (u64)t * kv_dim + kvh * hd);
#pragma unroll
            for (int d8 = 0; d8 < HD / 8; d8++) {
                uint4 kv = kr[d8];
                float k[8] = {h2f(kv.x & 0xffff), h2f(kv.x >> 16), h2f(kv.y & 0xffff),
                              h2f(kv.y >> 16),    h2f(kv.z & 0xffff), h2f(kv.z >> 16),
                              h2f(kv.w & 0xffff), h2f(kv.w >> 16)};
#pragma unroll
                for (int g = 0; g < MAX_GROUP; g++) {
                    if (g < group) {
                        const float* qq = qs + g * hd + d8 * 8;
                        float a = sc[g];
#pragma unroll
                        for (int i = 0; i < 8; i++) a = __fmaf_rn(qq[i], k[i], a);
                        sc[g] = a;
                    }
                }
            }
        }
#pragma unroll
        for (int g = 0; g < MAX_GROUP; g++)
            if (g < group) ps[g][tid] = t < tend ? __fmul_rn(sc[g], scale) : -INFINITY;
        __syncthreads();
        // Per head (a warp each): new running max, p = exp(s - m), running sum.
        for (int g = wid; g < group; g += ATT_TILE / 32) {
            float mx = -INFINITY;
            for (int i = lane; i < ATT_TILE; i += 32) mx = fmaxf(mx, ps[g][i]);
            mx = warp_max(mx);
            float mo = mrun[g], mn = fmaxf(mo, mx), ls = 0.0f;
            for (int i = lane; i < ATT_TILE; i += 32) {
                float p = expf(__fsub_rn(ps[g][i], mn));
                ps[g][i] = p;
                ls = __fadd_rn(ls, p);
            }
            ls = warp_sum(ls);
            __syncwarp();
            if (lane == 0) {
                float a = mo == -INFINITY ? 0.0f : expf(__fsub_rn(mo, mn));
                alpha[g] = a;
                lrun[g] = __fadd_rn(__fmul_rn(lrun[g], a), ls);
                mrun[g] = mn;
            }
        }
        __syncthreads();
        int tn = min(ATT_TILE, tend - t0);
#pragma unroll
        for (int g = 0; g < MAX_GROUP; g++) {
            o0[g] = __fmul_rn(o0[g], alpha[g < group ? g : 0]);
            o1[g] = __fmul_rn(o1[g], alpha[g < group ? g : 0]);
        }
        if constexpr (KG > 0) {
#pragma unroll 4
            for (int i = rs; i < tn; i += nrs) {
                u64 e = (u64)(t0 + i) * kv_dim + kvh * HD + 2 * dp;
                u32 vv = *(const unsigned short*)(q8.v + e);
                float d = q8.vs[e / KG];
                float v0 = (float)(int)(signed char)(vv & 0xff);
                float v1 = (float)(int)(signed char)(vv >> 8);
#pragma unroll
                for (int g = 0; g < MAX_GROUP; g++) {
                    if (g < group) {
                        float p = __fmul_rn(ps[g][i], d);
                        o0[g] = __fmaf_rn(p, v0, o0[g]);
                        o1[g] = __fmaf_rn(p, v1, o1[g]);
                    }
                }
            }
        } else
#pragma unroll 4
        for (int i = rs; i < tn; i += nrs) {
            u32 vv = *(const u32*)(vc + (u64)(t0 + i) * kv_dim + kvh * hd + 2 * dp);
            float v0 = h2f(vv & 0xffff), v1 = h2f(vv >> 16);
#pragma unroll
            for (int g = 0; g < MAX_GROUP; g++) {
                if (g < group) {
                    float p = ps[g][i];
                    o0[g] = __fmaf_rn(p, v0, o0[g]);
                    o1[g] = __fmaf_rn(p, v1, o1[g]);
                }
            }
        }
        __syncthreads();
    }
    // Merge the row slots in order, then write this split's partials.
#pragma unroll
    for (int g = 0; g < MAX_GROUP; g++) {
        if (g < group) {
            osum[(g * nrs + rs) * hd + 2 * dp] = o0[g];
            osum[(g * nrs + rs) * hd + 2 * dp + 1] = o1[g];
        }
    }
    __syncthreads();
    for (int i = tid; i < group * hd; i += ATT_TILE) {
        int g = i / hd, d = i % hd;
        float a = 0.0f;
        for (int r = 0; r < nrs; r++) a = __fadd_rn(a, osum[(g * nrs + r) * hd + d]);
        int h = kvh * group + g;
        part_o[((u64)h * max_split + s) * hd + d] = a;
    }
    if (tid < group) {
        int h = kvh * group + tid;
        part_ml[((u64)h * max_split + s) * 2] = mrun[tid];
        part_ml[((u64)h * max_split + s) * 2 + 1] = lrun[tid];
    }
}

extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_64(const float* __restrict__ q, const unsigned short* __restrict__ kc,
              const unsigned short* __restrict__ vc, const u32* __restrict__ P, int n_heads,
              int n_kv, float scale, int max_split, float* __restrict__ part_o,
              float* __restrict__ part_ml,
              const float* __restrict__ kraw, const float* __restrict__ qn,
              const float* __restrict__ kn, float eps, const float2* __restrict__ rope,
              unsigned short* kc_out) {
    PDL_START();
    attn_split_t<64>(q, kc, vc, (int)P[P_POS] + 1, blockIdx.x, blockIdx.y, n_heads, n_kv, scale,
                      max_split, part_o, part_ml, Q8Kv{},
                      QkNorm{kraw, qn, kn, eps, rope, kc_out});
}

extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_128(const float* __restrict__ q, const unsigned short* __restrict__ kc,
               const unsigned short* __restrict__ vc, const u32* __restrict__ P, int n_heads,
               int n_kv, float scale, int max_split, float* __restrict__ part_o,
               float* __restrict__ part_ml,
               const float* __restrict__ kraw, const float* __restrict__ qn,
               const float* __restrict__ kn, float eps, const float2* __restrict__ rope,
               unsigned short* kc_out) {
    PDL_START();
    attn_split_t<128>(q, kc, vc, (int)P[P_POS] + 1, blockIdx.x, blockIdx.y, n_heads, n_kv, scale,
                      max_split, part_o, part_ml, Q8Kv{},
                      QkNorm{kraw, qn, kn, eps, rope, kc_out});
}

// attn_split on the q8 cache (KG values per scale), quantizing this
// token's keys and values (kf: [2][kv_dim] f32, from qkv_rope_f32) first.
// The cache pointers aren't __restrict__: the kernel writes what it reads.
// qn non-null (qwen3): q and kf's keys are raw; normalized and rotated here.
#define ATTN_SPLIT_Q8(NAME, HD, KG)                                                                \
    extern "C" __global__ void __launch_bounds__(ATT_TILE)                                         \
        NAME(const float* __restrict__ q, const signed char* kq, const signed char* vq,            \
             const float* ks, const float* vs, const float* __restrict__ kf,                       \
             const u32* __restrict__ P, int n_heads, int n_kv, float scale, int max_split,         \
             float* __restrict__ part_o, float* __restrict__ part_ml,                              \
             const float* __restrict__ ksh, const float* __restrict__ qn,                          \
             const float* __restrict__ kn, float eps, const float2* __restrict__ rope) {           \
        PDL_START();                                                                               \
        attn_split_t<HD, KG>(q, nullptr, nullptr, (int)P[P_POS] + 1, blockIdx.x, blockIdx.y,       \
                             n_heads, n_kv, scale, max_split, part_o, part_ml,                     \
                             Q8Kv{kq, vq, ks, vs, kf, ksh},                                        \
                             QkNorm{nullptr, qn, kn, eps, rope, nullptr});                         \
    }
ATTN_SPLIT_Q8(attn_split_q8_64, 64, 64)
ATTN_SPLIT_Q8(attn_split_q8_128, 128, 128)
ATTN_SPLIT_Q8(attn_split_q8g32_64, 64, 32)
ATTN_SPLIT_Q8(attn_split_q8g32_128, 128, 32)

// Most splits attn_combine merges (the host keeps max_split below).
#define MAX_SPLIT 256

// The splits merged: out[h * hd + d] = sum_s w_s o_s[d] / sum_s w_s l_s with
// w_s = exp(m_s - max m); then quantized for the output projection (and
// stored as f32 in y). Grid n / 256 blocks of 256 threads (hd divides 256:
// a block has 256 / hd whole heads). Warp j works out head j's weights
// (sums in a fixed shuffle order), then every thread sums its element over
// the splits in order.
// Block blk of the combine for a query at position np - 1 (shared with
// attn_combine_b, prefill.cu).
__device__ void attn_combine_dev(const float* __restrict__ part_o,
                                 const float* __restrict__ part_ml, int np, int blk, int n,
                                 int hd, int max_split, int want_k, u8* __restrict__ act,
                                 float* __restrict__ y) {
    __shared__ float vals[256];
    __shared__ float ws[256 / 64][MAX_SPLIT];
    __shared__ float linv[256 / 64];
    int e = blk * 256 + threadIdx.x, lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int nsplit, chunk;
    att_split(np, max_split, &nsplit, &chunk);
    int h0 = blk * 256 / hd, nh = min(256 / hd, n / hd - h0);
    if (wid < nh) {
        const float* ml = part_ml + (u64)(h0 + wid) * max_split * 2;
        float M = -INFINITY;
        for (int s = lane; s < nsplit; s += 32) M = fmaxf(M, ml[2 * s]);
        M = warp_max(M);
        float L = 0.0f;
        for (int s = lane; s < nsplit; s += 32) {
            float w = expf(__fsub_rn(ml[2 * s], M));
            ws[wid][s] = w;
            L = __fadd_rn(L, __fmul_rn(w, ml[2 * s + 1]));
        }
        L = warp_sum(L);
        if (lane == 0) linv[wid] = L;
    }
    __syncthreads();
    float out = 0.0f;
    if (e < n) {
        int j = threadIdx.x / hd, d = e % hd;
        const float* po = part_o + (u64)(h0 + j) * max_split * hd + d;
        float O = 0.0f;
#pragma unroll 4
        for (int s = 0; s < nsplit; s++) O = __fmaf_rn(ws[j][s], po[(u64)s * hd], O);
        out = __fdiv_rn(O, linv[j]);
        if (y) y[e] = out;
    }
    vals[threadIdx.x] = out;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v[8];
        for (int i = 0; i < 8; i++) v[i] = vals[threadIdx.x * 8 + i];
        quant_warp(v, n, blk, act, want_k);
    }
}

extern "C" __global__ void attn_combine(const float* __restrict__ part_o,
                                        const float* __restrict__ part_ml,
                                        const u32* __restrict__ P, int n, int hd, int max_split,
                                        int want_k, u8* __restrict__ act, float* __restrict__ y) {
    PDL_START();
    attn_combine_dev(part_o, part_ml, (int)P[P_POS] + 1, blockIdx.x, n, hd, max_split, want_k, act,
                     y);
}

// Index of the largest of x[0..n): the last of equal maxima, NaN never
// picked, n - 1 if all are NaN (main.rs argmax, model.rs Best). One block of
// 1024 threads; the merge rule is order-independent, the order fixed anyway.
extern "C" __global__ void argmax(const float* __restrict__ x, int n, u32* __restrict__ out) {
    PDL_START();
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = -INFINITY;
    int bi = -1;
    // Thread t scans float4s t, t + 1024, ... (16-byte loads, 4 in flight),
    // each in index order, then the tail; `>=` is false for NaN.
    int n4 = n / 4;
    const float4* x4 = (const float4*)x;
    for (int i0 = threadIdx.x; i0 < n4; i0 += 4 * blockDim.x) {
        float4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            int i = i0 + u * blockDim.x;
            v[u] = i < n4 ? x4[i] : make_float4(NAN_F, NAN_F, NAN_F, NAN_F);
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            int i = 4 * (i0 + u * blockDim.x);
            if (v[u].x >= bv) bv = v[u].x, bi = i;
            if (v[u].y >= bv) bv = v[u].y, bi = i + 1;
            if (v[u].z >= bv) bv = v[u].z, bi = i + 2;
            if (v[u].w >= bv) bv = v[u].w, bi = i + 3;
        }
    }
    for (int i = 4 * n4 + threadIdx.x; i < n; i += blockDim.x)
        if (x[i] >= bv) bv = x[i], bi = i;
    // (bv, bi) beats (v, i) unless i is valid and v > bv or (v == bv, i > bi).
    for (int o = 16; o > 0; o >>= 1) {
        float v = __shfl_xor_sync(0xffffffffu, bv, o);
        int i = __shfl_xor_sync(0xffffffffu, bi, o);
        if (i >= 0 && (bi < 0 || v > bv || (v == bv && i > bi))) bv = v, bi = i;
    }
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) sv[w] = bv, si[w] = bi;
    __syncthreads();
    if (w == 0) {
        int nw = blockDim.x >> 5;
        bv = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : -1;
        for (int o = 16; o > 0; o >>= 1) {
            float v = __shfl_xor_sync(0xffffffffu, bv, o);
            int i = __shfl_xor_sync(0xffffffffu, bi, o);
            if (i >= 0 && (bi < 0 || v > bv || (v == bv && i > bi))) bv = v, bi = i;
        }
        if (lane == 0) *out = bi < 0 ? (u32)(n - 1) : (u32)bi;
    }
}

// Bandwidth probe for the benchmarks: XOR of n uint4s (written so the
// loads can't be dropped). Grid-stride, 4 loads in flight per thread.
extern "C" __global__ void stream_read(const uint4* __restrict__ p, u64 n, u32* __restrict__ out) {
    u32 acc = 0;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += 4 * stride) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] = i + u * stride < n ? __ldg(p + i + u * stride) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x12345678u) *out = acc;
}
