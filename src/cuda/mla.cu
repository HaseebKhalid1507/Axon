// deepseek2 (MLA, the absorbed form) on the GPU: r12 stage 1. Its own
// module (`Kernels::mla_module`: kernels.cu's deq helpers, moe.cu's router
// logits and add, then this file), loaded by the deepseek2 backend only:
// the dense and qwen3moe modules are not touched.
//
// Every kernel works per token (grid y or x = the token), with the token's
// position read from P (the decode step's {token, pos}) or given as start:
// nothing a token computes depends on the batch it is in, so decode ==
// batched prefill == verify, bit for bit (the same kernels run both).
//
// Layouts (per token z): c [z][cs] the kv_a product (latent | k_pe);
// q [z][n_h][hk] the query product (nope | rope); qa [z][n_h][krow] the
// absorbed queries (Wk_b q_nope | rope(q_pe)); the cache [row][krow] f16
// per layer (latent normed | k_pe turned), V = a row's first lat; att
// [z][n_h][lat]; the rope table [z][rope / 2] (sin, cos).

#define MLA_THREADS 256

__device__ __forceinline__ unsigned short mla_f2h(float x) {
    // Saturated to +-65504 then rounded to nearest even (decode.cu kv_half).
    x = x > 65504.0f ? 65504.0f : (x < -65504.0f ? -65504.0f : x);
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}

__device__ __forceinline__ float mla_h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ int mla_pos0(const u32* P, int start) {
    return P ? (int)P[1] : start;
}

// Fixed-shape tree over MLA_THREADS values (sum).
__device__ float mla_block_sum(float v, float* s) {
    int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int h = MLA_THREADS / 2; h > 0; h >>= 1) {
        if (t < h) s[t] = __fadd_rn(s[t], s[t + h]);
        __syncthreads();
    }
    float r = s[0];
    __syncthreads();
    return r;
}

// NEOX turn of pair (a, b) by (sin, cos) (decode.cu qkv_rope's ops).
__device__ __forceinline__ float2 mla_turn(float a, float b, float2 sc) {
    return make_float2(__fsub_rn(__fmul_rn(a, sc.y), __fmul_rn(b, sc.x)),
                       __fadd_rn(__fmul_rn(a, sc.x), __fmul_rn(b, sc.y)));
}

// Cache rows of nb tokens (grid nb, MLA_THREADS): row pos0 + z of kv =
// [rmsnorm(c[..lat]) * g | rope(c[lat..lat + rope])] in f16. The norm: per
// thread squares of i = t, t + 256, .. in order, the fixed tree, inv =
// 1 / sqrt(ss / lat + eps), v = (x * inv) * g.
extern "C" __global__ void __launch_bounds__(MLA_THREADS)
mla_kv_store(const float* __restrict__ c, int cs, const float* __restrict__ g, float eps, int lat,
             int rope, const float2* __restrict__ tab, const u32* __restrict__ P, int start,
             unsigned short* __restrict__ kv, int krow) {
    __shared__ float red[MLA_THREADS];
    int t = threadIdx.x, z = blockIdx.x;
    const float* cz = c + (u64)z * cs;
    float ss = 0.0f;
    for (int i = t; i < lat; i += MLA_THREADS) ss = __fadd_rn(ss, __fmul_rn(cz[i], cz[i]));
    ss = mla_block_sum(ss, red);
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)lat), eps)));
    unsigned short* row = kv + (u64)(mla_pos0(P, start) + z) * krow;
    for (int i = t; i < lat; i += MLA_THREADS) row[i] = mla_f2h(__fmul_rn(__fmul_rn(cz[i], inv), g[i]));
    int half = rope >> 1;
    const float2* tz = tab + (u64)z * half;
    for (int i = t; i < half; i += MLA_THREADS) {
        float2 o = mla_turn(cz[lat + i], cz[lat + half + i], tz[i]);
        row[lat + i] = mla_f2h(o.x);
        row[lat + half + i] = mla_f2h(o.y);
    }
}

// The queries' rope parts turned into the absorbed queries' tails (grid
// nb): qa[z][h][lat..krow] = rope(q[z][h][nope..hk]).
extern "C" __global__ void __launch_bounds__(MLA_THREADS)
mla_qrope(const float* __restrict__ q, int n_h, int hk, int nope, int rope,
          const float2* __restrict__ tab, float* __restrict__ qa, int krow, int lat) {
    int z = blockIdx.x, half = rope >> 1;
    const float2* tz = tab + (u64)z * half;
    for (int it = threadIdx.x; it < n_h * half; it += MLA_THREADS) {
        int h = it / half, i = it % half;
        const float* qh = q + ((u64)z * n_h + h) * hk + nope;
        float2 o = mla_turn(qh[i], qh[half + i], tz[i]);
        float* d = qa + ((u64)z * n_h + h) * krow + lat;
        d[i] = o.x;
        d[half + i] = o.y;
    }
}

// Per-head products (absorb: Wk_b q_nope; expand: Wv_b o):
//   out[z][h * os + r] = sum_c W[h * rph + r][c] * x[z][h * xs + c]
// for heads h, rows r < rph, c < cols; x rows xt floats apart, out ot.
// W: type qt, rb bytes per row, n_h * rph rows, dequantized exactly (deq).
// A warp per matrix row and MLA_MV_T tokens: lane l takes c = l, l + 32, ..
// ascending, one fma chain per token, then the fixed xor tree. A token's
// sums are the same in any batch.
//
// Grid (ceil(n_h * rph / rpb), ceil(nb / MLA_MV_T)), mv_smem bytes of
// dynamic shared memory (mla.rs `mv_smem`). A block stages its rpb
// consecutive rows' bytes (the widest loads their alignment allows), its
// tokens' x slices of the heads those rows are in, and the rows' block
// scales, in shared memory; then warp w computes rows w, w + 8, .. from
// there (a lone token: 4 rows per warp at a time, interleaved). Each
// output's float operations and their order are those of main
// be13f1f's kernel (kept as mla_heads_mv_ref in test builds; mla_tests.rs
// compares bits): the same chains and tree, the same x floats, and each
// weight the value deq(qt, row, c) gives, by the same float operations on
// the same operands. For the common types the per-block factors deq
// computes for every element (Q5_0 / Q8_0: d; Q4_K: d1 = d * sc and m1 =
// dmin * m; Q6_K: ds = d * scale) are computed once per block, by the same
// expressions, into the scale table, and the element is then deq's last
// step on them: (float)x * d, d1 * (float)nib - m1, ds * (float)q. Other
// types call deq itself. What changed is only that, where bytes come from,
// and NT = 1 for a lone token (the decode step), which skips the 7 idle
// chains and trees.
#define MLA_MV_T 8

// Scale-table floats per row for qt over cols values (mla.rs mirrors it in
// `mv_smem`): one per 32 (Q5_0, Q8_0), two per 32 (Q4_K), one per 16 (Q6_K).
__device__ __forceinline__ int mla_tab_n(int qt, int cols) {
    switch (qt) {
    case QT_Q5_0:
    case QT_Q8_0:
        return cols / 32;
    case QT_Q4_K:
        return cols / 16;
    case QT_Q6_K:
        return cols / 16;
    default:
        return 0;
    }
}

// Entry e of a row's scale table (deq_*'s per-element factors).
__device__ __forceinline__ float mla_tab_entry(int qt, const u8* row, int e) {
    switch (qt) {
    case QT_Q5_0:  // deq_q5_0's d
        return rd_f16(row + 22 * e);
    case QT_Q8_0:  // deq_q8_0's rd_f16(b)
        return rd_f16(row + 34 * e);
    case QT_Q4_K: {  // deq_q4_k's d1 (even e) and m1 (odd e) of sub-block e / 2
        int sb = e >> 1, b = sb >> 3, j = sb & 7;
        const u8* blk = row + 144 * b;
        u32 sc, m;
        scale_min_k4(j, blk + 4, &sc, &m);
        return (e & 1) ? __fmul_rn(rd_f16(blk + 2), (float)m) : __fmul_rn(rd_f16(blk), (float)sc);
    }
    default: {  // QT_Q6_K: deq_q6_k's ds of 16-value group e
        int b = e >> 4, gi = e & 15;  // group gi = 8 half + 2 k + is
        int half = gi >> 3, k = (gi >> 1) & 3, is = gi & 1;
        const u8* blk = row + 210 * b;
        const u8* sc = blk + 192 + 8 * half;
        return __fmul_rn(rd_f16(blk + 208), (float)(signed char)sc[is + 2 * k]);
    }
    }
}

// The weight deq(QT, row, c) gives, from the row's bytes and scale table.
template <int QT>
__device__ __forceinline__ float mla_w(int qt, const u8* row, const float* tab, int c) {
    if (QT == QT_Q5_0) {  // deq_q5_0
        const u8* b = row + 22 * (c >> 5);
        int i = c & 31, j = i & 15;
        u32 qh = (u32)b[2] | ((u32)b[3] << 8) | ((u32)b[4] << 16) | ((u32)b[5] << 24);
        int x = i < 16 ? (int)((b[6 + j] & 0x0f) | (((qh >> j) << 4) & 0x10)) - 16
                       : (int)((b[6 + j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
        return __fmul_rn((float)x, tab[c >> 5]);
    } else if (QT == QT_Q8_0) {  // deq_q8_0
        return __fmul_rn((float)(signed char)row[34 * (c >> 5) + 2 + (c & 31)], tab[c >> 5]);
    } else if (QT == QT_Q4_K) {  // deq_q4_k
        const u8* b = row + 144 * (c >> 8);
        int i = c & 255;
        int ch = i >> 6, w = i & 63, l = w & 31, hi = w >> 5;
        u8 q = b[16 + 32 * ch + l];
        u32 nib = hi ? q >> 4 : q & 0xf;
        int sb = c >> 5;  // = 8 (c >> 8) + 2 ch + hi
        return __fsub_rn(__fmul_rn(tab[2 * sb], (float)nib), tab[2 * sb + 1]);
    } else if (QT == QT_Q6_K) {  // deq_q6_k
        const u8* b = row + 210 * (c >> 8);
        int i = c & 255;
        int half = i >> 7, r = i & 127, k = r >> 5, l = r & 31, is = l >> 4;
        const u8* ql = b + 64 * half;
        const u8* qh = b + 128 + 32 * half;
        u8 qlb = ql[l + 32 * (k & 1)];
        u32 nib = k < 2 ? qlb & 0xf : qlb >> 4;
        int q = (int)(nib | (((qh[l] >> (2 * k)) & 3) << 4)) - 32;
        return __fmul_rn(tab[16 * (c >> 8) + 8 * half + 2 * k + is], (float)q);
    } else {
        return deq(QT < 0 ? qt : QT, row, c);
    }
}

// One row's outputs for NT tokens (the chain and the tree above).
template <int QT, int NT>
__device__ __forceinline__ void mla_mv_row(int qt, const u8* row, const float* tab, int gr,
                                           int rph, int cols, const float* xz,
                                           float* __restrict__ out, int os, u64 ot, int z0,
                                           int nt) {
    int lane = threadIdx.x & 31;
    int h = gr / rph, r = gr % rph;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        float wv = mla_w<QT>(qt, row, tab, c);
#pragma unroll
        for (int t = 0; t < NT; t++)
            if (NT == 1 || t < nt) acc[t] = __fmaf_rn(wv, xz[t * cols + c], acc[t]);
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        if (NT == 1 || t < nt) {  // nt is the block's: uniform
            float v = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
            if (lane == 0) out[(u64)(z0 + t) * ot + (u64)h * os + r] = v;
        }
    }
}

// A lone token: 4 rows per warp at a time (4g .. 4g + 3, g = w, w + 8, ..),
// their chains interleaved (each row's still in c order, its own tree), a
// short last group one row at a time.
template <int QT>
__device__ __forceinline__ void mla_mv_rows1(int qt, const u8* sh, u64 rb, const float* tabs,
                                             int tn, int nrow, int gr0, int rph, int cols,
                                             const float* xsh, int h_lo,
                                             float* __restrict__ out, int os, u64 ot, int z0) {
    int lane = threadIdx.x & 31;
    for (int g = threadIdx.x >> 5; 4 * g < nrow; g += MLA_THREADS / 32) {
        int i0 = 4 * g;
        if (i0 + 3 >= nrow) {
            for (int i = i0; i < nrow; i++)
                mla_mv_row<QT, 1>(qt, sh + (u64)i * rb, tabs + i * tn, gr0 + i, rph, cols,
                                  xsh + ((gr0 + i) / rph - h_lo) * cols, out, os, ot, z0, 1);
            continue;
        }
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        const u8* r0 = sh + (u64)i0 * rb;
        const float* t0 = tabs + i0 * tn;
        const float* x0 = xsh + ((gr0 + i0) / rph - h_lo) * cols;
        const float* x1 = xsh + ((gr0 + i0 + 1) / rph - h_lo) * cols;
        const float* x2 = xsh + ((gr0 + i0 + 2) / rph - h_lo) * cols;
        const float* x3 = xsh + ((gr0 + i0 + 3) / rph - h_lo) * cols;
        for (int c = lane; c < cols; c += 32) {
            a0 = __fmaf_rn(mla_w<QT>(qt, r0, t0, c), x0[c], a0);
            a1 = __fmaf_rn(mla_w<QT>(qt, r0 + rb, t0 + tn, c), x1[c], a1);
            a2 = __fmaf_rn(mla_w<QT>(qt, r0 + 2 * rb, t0 + 2 * tn, c), x2[c], a2);
            a3 = __fmaf_rn(mla_w<QT>(qt, r0 + 3 * rb, t0 + 3 * tn, c), x3[c], a3);
        }
        // The four rows' xor trees, transposed: every sum is the tree's
        // (lane i's value plus lane i ^ o's, of the same row), but a lane
        // carries only the rows its bits 4 and 3 pick, so 6 shuffles for
        // the 4 trees instead of 20. Row lane >> 3 ends in lanes 8 k ..
        // 8 k + 7.
        bool b4 = lane & 16, b3 = lane & 8;
        float k0 = b4 ? a2 : a0, k1 = b4 ? a3 : a1;
        float y0 = b4 ? a0 : a2, y1 = b4 ? a1 : a3;  // sent
        k0 = __fadd_rn(k0, __shfl_xor_sync(0xffffffffu, y0, 16));
        k1 = __fadd_rn(k1, __shfl_xor_sync(0xffffffffu, y1, 16));
        float k = b3 ? k1 : k0, y = b3 ? k0 : k1;
        k = __fadd_rn(k, __shfl_xor_sync(0xffffffffu, y, 8));
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) k = __fadd_rn(k, __shfl_xor_sync(0xffffffffu, k, o));
        if ((lane & 7) == 0) {
            int gr = gr0 + i0 + (lane >> 3);
            out[(u64)z0 * ot + (u64)(gr / rph) * os + gr % rph] = k;
        }
    }
}

template <int QT, int NT>
__device__ __forceinline__ void mla_mv_rows(int qt, const u8* sh, u64 rb, const float* tabs,
                                            int tn, int nrow, int gr0, int rph, int cols,
                                            const float* xsh, int h_lo,
                                            float* __restrict__ out, int os, u64 ot, int z0,
                                            int nt) {
    for (int i = threadIdx.x >> 5; i < nrow; i += MLA_THREADS / 32) {
        int gr = gr0 + i;
        mla_mv_row<QT, NT>(qt, sh + (u64)i * rb, tabs + i * tn, gr, rph, cols,
                           xsh + (gr / rph - h_lo) * nt * cols, out, os, ot, z0, nt);
    }
}

template <int QT>
__device__ __forceinline__ void mla_mv_nt(int qt, const u8* sh, u64 rb, const float* tabs,
                                          int tn, int nrow, int gr0, int rph, int cols,
                                          const float* xsh, int h_lo, float* out, int os,
                                          u64 ot, int z0, int nt) {
    if (nt == 1)
        mla_mv_rows1<QT>(qt, sh, rb, tabs, tn, nrow, gr0, rph, cols, xsh, h_lo, out, os, ot, z0);
    else
        mla_mv_rows<QT, MLA_MV_T>(qt, sh, rb, tabs, tn, nrow, gr0, rph, cols, xsh, h_lo, out,
                                  os, ot, z0, nt);
}

// n bytes from src to dst (shared), the widest loads both allow.
__device__ __forceinline__ void mla_copy(u8* dst, const u8* src, u64 n) {
    u64 al = (u64)src | (u64)dst | n;
    if ((al & 15) == 0) {
        for (u64 i = threadIdx.x; i < n / 16; i += MLA_THREADS)
            ((uint4*)dst)[i] = ((const uint4*)src)[i];
    } else if ((al & 3) == 0) {
        for (u64 i = threadIdx.x; i < n / 4; i += MLA_THREADS) ((u32*)dst)[i] = ((const u32*)src)[i];
    } else {
        for (u64 i = threadIdx.x; i < n; i += MLA_THREADS) dst[i] = src[i];
    }
}

// Shared memory: the rows' bytes (rpb * rb, rounded up to 16), the scale
// tables (rpb * mla_tab_n floats), then x[head - h_lo][t][c] for the heads
// h_lo..h_hi the rows are in.
extern "C" __global__ void __launch_bounds__(MLA_THREADS)
mla_heads_mv(int qt, const u8* __restrict__ w, u64 rb, int rph, int cols, int n_h,
             const float* __restrict__ x, int xs, u64 xt, float* __restrict__ out, int os,
             u64 ot, int nb, int rpb) {
    extern __shared__ uint4 mv_sh[];
    int gr0 = blockIdx.x * rpb;
    int nrow = min(rpb, n_h * rph - gr0);
    if (nrow <= 0) return;
    u8* sh = (u8*)mv_sh;
    mla_copy(sh, w + (u64)gr0 * rb, (u64)nrow * rb);
    int z0 = blockIdx.y * MLA_MV_T;
    int nt = min(MLA_MV_T, nb - z0);
    int h_lo = gr0 / rph, h_hi = (gr0 + nrow - 1) / rph;
    int tn = mla_tab_n(qt, cols);
    float* tabs = (float*)(sh + (((u64)rpb * rb + 15) & ~(u64)15));
    float* xsh = tabs + rpb * tn;
    int per = nt * cols;
    for (int i = threadIdx.x; i < (h_hi - h_lo + 1) * per; i += MLA_THREADS) {
        int hl = i / per, t = (i % per) / cols, c = i % cols;
        xsh[i] = x[(u64)(z0 + t) * xt + (u64)(h_lo + hl) * xs + c];
    }
    __syncthreads();
    // The scale tables, from the staged bytes.
    for (int i = threadIdx.x; i < nrow * tn; i += MLA_THREADS)
        tabs[i] = mla_tab_entry(qt, sh + (u64)(i / tn) * rb, i % tn);
    __syncthreads();
#define MLA_MV_CASE(T)                                                                     \
    case T:                                                                                \
        mla_mv_nt<T>(qt, sh, rb, tabs, tn, nrow, gr0, rph, cols, xsh, h_lo, out, os, ot, z0, \
                     nt);                                                                  \
        break;
    switch (qt) {
        MLA_MV_CASE(QT_Q5_0)
        MLA_MV_CASE(QT_Q8_0)
        MLA_MV_CASE(QT_Q4_K)
        MLA_MV_CASE(QT_Q6_K)
        MLA_MV_CASE(QT_F16)
        MLA_MV_CASE(QT_F32)
    default:
        mla_mv_nt<-1>(qt, sh, rb, tabs, tn, nrow, gr0, rph, cols, xsh, h_lo, out, os, ot, z0, nt);
    }
#undef MLA_MV_CASE
}

// mla_heads_mv's products for a batch (nb > 1), as a register-tiled f32
// GEMM with the same bits. Per output, mla_heads_mv computes 32 lane chains,
// chain l = fma over c = l, l + 32, .. ascending from 0, then the xor tree:
// at offset 16 it adds chains l and l + 16, then 8, .., 1. That tree is a
// fixed binary tree whose leaves, in order, are the chains l = brev5(i), i
// = 0..31 (brev5: 5-bit reversal: 0, 16, 8, 24, 4, ..), each internal node
// the __fadd_rn of its two children. Here a thread computes chain brev5(i)
// for i = 0, 1, .. in turn (the same fma chain) and folds it into the tree
// as a binary counter does (in groups of 8 leaves, then the groups' roots):
// every node is the same __fadd_rn of the same two values, so the output
// is the same float. The columns are staged permuted, in that order (chain
// by chain), and the weights are deq's values. Cols must be a multiple of
// 32 (every chain cols / 32 long), and a block's rows one head's.
// Grid (n_h * rph / MLA_MM_R, ceil(nb / MLA_MM_T)), MLA_THREADS threads,
// thread (rg, tg) = (t % 16, t / 16): rows 4 rg .. + 4 x tokens 4 tg .. + 4
// of the block's MLA_MM_R x MLA_MM_T. Shared memory: MLA_MM_SC permuted
// columns of the rows (dequantized, column-major: [j][r]) and of x ([j][t]).
#define MLA_MM_R 64
#define MLA_MM_T 64
#define MLA_MM_SC 32
#define MLA_MM_BYTES 18432  // 64 rows of 288 bytes (Q4_K x 512)

__device__ __forceinline__ int mla_brev5(int i) { return __brev((unsigned)i) >> 27; }


// The weight deq(qt, row, c) gives, from the row's staged bytes and scale
// table (mla_w, its type resolved here).
__device__ __forceinline__ float mla_mm_w(int qt, const u8* row, const float* tab, int c) {
    switch (qt) {
    case QT_Q5_0:
        return mla_w<QT_Q5_0>(qt, row, tab, c);
    case QT_Q8_0:
        return mla_w<QT_Q8_0>(qt, row, tab, c);
    case QT_Q4_K:
        return mla_w<QT_Q4_K>(qt, row, tab, c);
    case QT_Q6_K:
        return mla_w<QT_Q6_K>(qt, row, tab, c);
    default:
        return deq(qt, row, c);
    }
}

// mla_heads_mm's block state, for its helpers.
struct MlaMm {
    int qt;
    const u8* w;
    u64 rb;
    bool staged;
    const u8* rows;
    const float* tabs;
    int tn;
    const float* x;
    int xs;
    u64 xt;
    int nb;
    float* ws;
    float* xsh;
    int gr0, z0, h, L, per, t, rg, tg;
};

// Stage chains i .. i + n - 1 (n = min(per, 32 - i)): their permuted columns
// brev5(chain) + 32 k of the block's rows (deq's values) and of x.
__device__ __noinline__ void mla_mm_stage(const MlaMm& m, int i) {
    __syncthreads();
    int n = min(m.per, 32 - i) * m.L;
    for (int e = m.t; e < n * MLA_MM_R; e += MLA_THREADS) {
        int j = e / MLA_MM_R, r = e % MLA_MM_R;
        int c = mla_brev5(i + j / m.L) + 32 * (j % m.L);
        m.ws[j * MLA_MM_R + r] = m.staged ? mla_mm_w(m.qt, m.rows + (u64)r * m.rb, m.tabs + r * m.tn, c)
                                          : deq(m.qt, m.w + (u64)(m.gr0 + r) * m.rb, c);
    }
    for (int e = m.t; e < n * MLA_MM_T; e += MLA_THREADS) {
        int j = e / MLA_MM_T, tt = e % MLA_MM_T;
        int c = mla_brev5(i + j / m.L) + 32 * (j % m.L);
        m.xsh[j * MLA_MM_T + tt] =
            m.z0 + tt < m.nb ? m.x[(u64)(m.z0 + tt) * m.xt + (u64)m.h * m.xs + c] : 0.0f;
    }
    __syncthreads();
}

// Leaf i = 8 g + V: chain brev5(i) of the thread's 4 x 4 outputs, folded
// into the group's subtree (ls: its open left children; y: the root at V = 7).
template <int V>
__device__ __forceinline__ void mla_mm_leaf(const MlaMm& m, int i, float (*ls)[4][3],
                                            float (*y)[4]) {
    int ii = i % m.per;
    if (ii == 0) mla_mm_stage(m, i);
    float a[4][4];
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
        for (int u = 0; u < 4; u++) a[q][u] = 0.0f;
    for (int k = 0; k < m.L; k++) {
        int jj = ii * m.L + k;
        float4 w4 = *(const float4*)(m.ws + jj * MLA_MM_R + 4 * m.rg);
        float4 x4 = *(const float4*)(m.xsh + jj * MLA_MM_T + 4 * m.tg);
        float wr[4] = {w4.x, w4.y, w4.z, w4.w};
        float xr[4] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int u = 0; u < 4; u++) a[q][u] = __fmaf_rn(wr[q], xr[u], a[q][u]);
    }
    // Close the levels whose bit of V is set, lowest first; wait at the first
    // clear one.
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
        for (int u = 0; u < 4; u++) {
            float v = a[q][u];
            if (V & 1) {
                v = __fadd_rn(ls[q][u][0], v);
                if (V & 2) {
                    v = __fadd_rn(ls[q][u][1], v);
                    if (V & 4) y[q][u] = __fadd_rn(ls[q][u][2], v);
                    else ls[q][u][2] = v;
                } else {
                    ls[q][u][1] = v;
                }
            } else {
                ls[q][u][0] = v;
            }
        }
}

extern "C" __global__ void __launch_bounds__(MLA_THREADS, 1)
mla_heads_mm(int qt, const u8* __restrict__ w, u64 rb, int rph, int cols, int n_h,
             const float* __restrict__ x, int xs, u64 xt, float* __restrict__ out, int os,
             u64 ot, int nb) {
    __shared__ __align__(16) float ws[MLA_MM_SC * MLA_MM_R];
    __shared__ __align__(16) float xsh[MLA_MM_SC * MLA_MM_T];
    __shared__ __align__(16) u8 rows[MLA_MM_BYTES];  // the block's rows, when they fit
    __shared__ float tabs[MLA_MM_R * 32];             // their scale tables (<= 32 per row)
    int t = threadIdx.x, rg = t % 16, tg = t / 16;
    int gr0 = blockIdx.x * MLA_MM_R, z0 = blockIdx.y * MLA_MM_T;
    int h = gr0 / rph, r0 = gr0 % rph;
    int L = cols / 32;                // chain length
    int per = MLA_MM_SC / L;          // chains per stage (the host: L <= MLA_MM_SC)
    // The rows' bytes and scale tables in shared memory when they fit
    // (Moonlight's k_b and v_b); else deq reads the rows in global memory.
    int tn = mla_tab_n(qt, cols);
    bool staged = (u64)MLA_MM_R * rb <= MLA_MM_BYTES && tn <= 32;
    if (staged) {
        mla_copy(rows, w + (u64)gr0 * rb, (u64)MLA_MM_R * rb);
        __syncthreads();
        for (int i = t; i < MLA_MM_R * tn; i += MLA_THREADS)
            tabs[i] = mla_tab_entry(qt, rows + (u64)(i / tn) * rb, i % tn);
    }
    // The tree in groups of 8 leaves (8 g .. 8 g + 7: a whole subtree of
    // depth 3, its levels folded at compile time: mla_mm_leaf<V>), then each
    // group's root as leaf g of the top two levels (gs).
    MlaMm m{qt, w, rb, staged, rows, tabs, tn, x, xs, xt, nb, ws, xsh, gr0, z0, h, L, per, t, rg, tg};
    float gs[4][4][2];
    float res[4][4];
    for (int g = 0; g < 4; g++) {
        float ls[4][4][3];
        float y[4][4];
        mla_mm_leaf<0>(m, 8 * g, ls, y);
        mla_mm_leaf<1>(m, 8 * g + 1, ls, y);
        mla_mm_leaf<2>(m, 8 * g + 2, ls, y);
        mla_mm_leaf<3>(m, 8 * g + 3, ls, y);
        mla_mm_leaf<4>(m, 8 * g + 4, ls, y);
        mla_mm_leaf<5>(m, 8 * g + 5, ls, y);
        mla_mm_leaf<6>(m, 8 * g + 6, ls, y);
        mla_mm_leaf<7>(m, 8 * g + 7, ls, y);
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int u = 0; u < 4; u++) {
                float v = y[q][u];  // the group's root
                if (g == 0 || g == 2) {
                    gs[q][u][0] = v;
                } else {
                    v = __fadd_rn(gs[q][u][0], v);
                    if (g == 1) gs[q][u][1] = v;
                    else res[q][u] = __fadd_rn(gs[q][u][1], v);
                }
            }
    }
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
        for (int u = 0; u < 4; u++) {
            int z = z0 + 4 * tg + u;
            if (z < nb) out[(u64)z * ot + (u64)h * os + r0 + 4 * rg + q] = res[q][u];
        }
}

// The attention of nb tokens' absorbed queries over their cache rows 0..np
// (np = pos0 + z + 1): o_h = softmax(Q_h . K_t * scale) . K_t[..lat], in
// splits of `split` rows (a multiple of MLA_TILE) from row 0: split sp of a
// token covers rows sp * split .. min(np, (sp + 1) * split), so a token has
// ceil(np / split) of them, whatever its batch. Rows in tiles of MLA_TILE,
// the tile's K rows in shared memory (f16), an online softmax per head
// across the split's tiles in order. Per head and tile: each row's score
// (lane l: columns 2l, 2l + 1, 2l + 64, ..: one fma chain, then the xor
// tree, times scale), lane r holding row r's; the tile's max m_new, p_r =
// exp(s_r - m_new) and their xor-tree sum, the head's m and l; then each
// column pair rescaled by exp(m_old - m_new) and fma'd over the tile's
// rows in order. Out: the split's (m, l, acc[lat]) per head at
// part[((z * ns + sp) * n_h + h) * (lat + 2)] (mla_attn_cmb combines them).
// The ops of a (token, head) depend on np only.
//
// Who computes what doesn't change them (main be13f1f's kernel is kept as
// mla_attn_ref in test builds; mla_tests.rs compares bits):
// - Grid (nblk, nb), MLA_THREADS threads. A token's blocks work for hpb
//   heads each, hpb picked per token by the work: the largest power of two
//   <= hpb_max (dividing n_h) with at least `busy` busy blocks (n_h / hpb
//   times the token's splits times nb), else 1. Few splits (short
//   positions, a lone decode token): more blocks, fewer heads each. Block
//   b works for split b / (n_h / hpb), heads (b % (n_h / hpb)) * hpb ..
//   + hpb; past the token's splits it does nothing. nblk covers every
//   token's (mla.rs `attn_blocks`).
// - The tile's row scores: warp w takes head h0 + w % hpb, rows w / hpb,
//   w / hpb + 8 / hpb, .. (all 8 warps busy at any hpb), into shared
//   memory; warp w < hpb then reads its head's back as lane r = row r.
// - Thread t accumulates head t / tph's column pairs 2 (t % tph) + 2 tph j
//   (tph = MLA_THREADS / hpb).
// - dbuf (sm_80+, krow a multiple of 8, 2 tiles of dynamic shared memory):
//   the next tile's rows are copied (cp.async) while this one is used.
#define MLA_TILE 32
#define MLA_MAXI 9   // krow <= 576: 64-column steps of the score
#define MLA_MAXC 16  // lat <= 512 at tph 32: column pairs per thread

__device__ __forceinline__ void mla_tile_load(unsigned short* dst, const unsigned short* kv,
                                              int r0, int nr, int krow, bool dbuf) {
#if __CUDA_ARCH__ >= 800
    if (dbuf) {
        int ch = krow >> 3;  // 16-byte chunks per row
        const unsigned short* src = kv + (u64)r0 * krow;
        for (int i = threadIdx.x; i < nr * ch; i += MLA_THREADS) {
            u32 s = (u32)__cvta_generic_to_shared(dst + 8 * i);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src + 8 * i));
        }
        asm volatile("cp.async.commit_group;");
        return;
    }
#endif
    const unsigned* src = (const unsigned*)(kv + (u64)r0 * krow);
    int kw = krow >> 1;  // u32 words per row
    for (int i = threadIdx.x; i < nr * kw; i += MLA_THREADS) ((unsigned*)dst)[i] = src[i];
}

// Wait for all but the newest `pending` (0 or 1) cp.async groups.
__device__ __forceinline__ void mla_tile_wait(int pending, bool dbuf) {
#if __CUDA_ARCH__ >= 800
    if (dbuf) {
        if (pending) asm volatile("cp.async.wait_group 1;" ::: "memory");
        else asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
#endif
}

// A cache row's score against a head's query (lane l: columns 2l, 2l + 1,
// 2l + 64, ..: one fma chain, then the xor tree), unscaled.
__device__ __forceinline__ float mla_score(const float2* qr, const unsigned short* kr, int krow,
                                           int lane) {
    float d = 0.0f;
#pragma unroll
    for (int i = 0; i < MLA_MAXI; i++) {
        int c = 2 * lane + 64 * i;
        if (c < krow) {
            unsigned kk = *(const unsigned*)(kr + c);
            d = __fmaf_rn(qr[i].x, mla_h2f(kk & 0xffff), d);
            d = __fmaf_rn(qr[i].y, mla_h2f(kk >> 16), d);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) d = __fadd_rn(d, __shfl_xor_sync(0xffffffffu, d, o));
    return d;
}

extern "C" __global__ void __launch_bounds__(MLA_THREADS)
mla_attn(const float* __restrict__ qa, const unsigned short* __restrict__ kv, int krow, int lat,
         int n_h, int hpb_max, float scale, const u32* __restrict__ P, int start, int split,
         int ns, float* __restrict__ part, int busy, int dbuf_on) {
    // 16-byte aligned: cp.async writes 16-byte chunks.
    extern __shared__ __align__(16) unsigned short ks[];  // [1 or 2][MLA_TILE][krow]
    __shared__ float ps[8][MLA_TILE];
    __shared__ float ss[8][MLA_TILE];
    __shared__ float ms[8], ls[8], al[8];
    int t = threadIdx.x, lane = t & 31, w = t >> 5;
    int z = blockIdx.y;
    int np = mla_pos0(P, start) + z + 1;
    int nsp = (np + split - 1) / split;
    int hpb = hpb_max;
    while (hpb > 1 && (n_h / hpb) * nsp * (int)gridDim.y < busy) hpb >>= 1;
    int ng_h = n_h / hpb;  // head groups
    int sp = blockIdx.x / ng_h;
    if (sp >= nsp) return;
    int h0 = (blockIdx.x % ng_h) * hpb;
    int rb = sp * split, re = min(np, rb + split);
    int tph = MLA_THREADS / hpb;
    bool dbuf = dbuf_on != 0;
    // The score warp's head and rows, its query's column pairs.
    int sh = w % hpb, sg = w / hpb, ng = (MLA_THREADS / 32) / hpb;
    float2 qr[MLA_MAXI];
    {
        const float* qh = qa + ((u64)z * n_h + h0 + sh) * krow;
#pragma unroll
        for (int i = 0; i < MLA_MAXI; i++) {
            int c = 2 * lane + 64 * i;
            qr[i] = c < krow ? make_float2(qh[c], qh[c + 1]) : make_float2(0.0f, 0.0f);
        }
    }
    if (w < hpb && lane == 0) {
        ms[w] = __int_as_float(0xff800000);  // -inf
        ls[w] = 0.0f;
    }
    // The accumulating thread's head and columns.
    int ah = t / tph, ac = 2 * (t % tph);
    float2 acc[MLA_MAXC];
#pragma unroll
    for (int j = 0; j < MLA_MAXC; j++) acc[j] = make_float2(0.0f, 0.0f);
    int tile = MLA_TILE * krow;
    int buf = 0;
    if (dbuf) mla_tile_load(ks, kv, rb, min(MLA_TILE, re - rb), krow, true);
    for (int r0 = rb; r0 < re; r0 += MLA_TILE) {
        int nr = min(MLA_TILE, re - r0);
        // Everyone is done with the tile read last (and, dbuf, with the
        // buffer the next tile goes into).
        __syncthreads();
        unsigned short* kt = ks + buf * tile;
        if (dbuf) {
            int n1 = r0 + MLA_TILE;
            bool more = n1 < re;
            if (more) mla_tile_load(ks + (buf ^ 1) * tile, kv, n1, min(MLA_TILE, re - n1), krow, true);
            mla_tile_wait(more ? 1 : 0, true);
        } else {
            mla_tile_load(kt, kv, r0, nr, krow, false);
        }
        __syncthreads();
        // ng == 1 (hpb 8): warp w scores all of head w's rows itself, the
        // score in lane r's register (main's loop); else warps share a
        // head's rows, through ss.
        float s1 = __int_as_float(0xff800000);
        if (ng == 1) {
            for (int r = 0; r < nr; r++) {
                float d = mla_score(qr, kt + r * krow, krow, lane);
                if (lane == r) s1 = __fmul_rn(d, scale);
            }
        } else {
            for (int r = sg; r < nr; r += ng) {
                float d = mla_score(qr, kt + r * krow, krow, lane);
                if (lane == 0) ss[sh][r] = __fmul_rn(d, scale);
            }
            __syncthreads();
        }
        if (w < hpb) {
            float s = ng == 1 ? s1 : (lane < nr ? ss[w][lane] : __int_as_float(0xff800000));
            float mx = s;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            float m_old = ms[w];
            float m_new = fmaxf(m_old, mx);
            float p = lane < nr ? expf(__fsub_rn(s, m_new)) : 0.0f;
            float sum = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
            ps[w][lane] = p;
            __syncwarp();
            if (lane == 0) {
                float a = expf(__fsub_rn(m_old, m_new));
                al[w] = a;
                ms[w] = m_new;
                ls[w] = __fadd_rn(__fmul_rn(ls[w], a), sum);
            }
        }
        __syncthreads();
        if (ah < hpb) {
            float a = al[ah];
#pragma unroll
            for (int j = 0; j < MLA_MAXC; j++) {
                int c = ac + 2 * tph * j;
                if (c < lat) {
                    float2 v = make_float2(__fmul_rn(acc[j].x, a), __fmul_rn(acc[j].y, a));
                    for (int r = 0; r < nr; r++) {
                        unsigned kk = *(const unsigned*)(kt + r * krow + c);
                        float p = ps[ah][r];
                        v.x = __fmaf_rn(p, mla_h2f(kk & 0xffff), v.x);
                        v.y = __fmaf_rn(p, mla_h2f(kk >> 16), v.y);
                    }
                    acc[j] = v;
                }
            }
        }
        if (dbuf) buf ^= 1;
    }
    __syncthreads();
    if (ah < hpb) {
        float* o = part + (((u64)z * ns + sp) * n_h + h0 + ah) * (lat + 2);
        if (t % tph == 0) {
            o[0] = ms[ah];
            o[1] = ls[ah];
        }
#pragma unroll
        for (int j = 0; j < MLA_MAXC; j++) {
            int c = ac + 2 * tph * j;
            if (c < lat) {
                o[2 + c] = acc[j].x;
                o[2 + c + 1] = acc[j].y;
            }
        }
    }
}

// mla_attn for a batch (nb > 1), the same bits per (token, head, split):
// a block takes one split and MLA_AB_Q (token, head) pairs, MLA_AB_Q / hb
// consecutive tokens x hb heads (hb = min(n_h, MLA_AB_Q), a power of two),
// so each K tile is read and converted once for all of them. Per (token,
// head) and tile, everything mla_attn does, in its order: the rows' scores
// (lane l's chain over columns 2l, 2l + 1, 2l + 64, .., the xor tree, times
// scale), the tile's max, p_r and their xor-tree sum (lane r = row r), m, l
// and the rescale, then each column's fma over the tile's rows in order. A
// token takes part in a tile only below its own end (min(np, split end)),
// with its own row count; a token whose cache ends before the split does
// nothing there (as mla_attn's block). What differs is only who computes:
// - the tile's K rows converted to f32 in shared memory once (cvt.f32.f16
//   is exact; mla_attn converts at each use);
// - warp w scores pairs 4w .. 4w + 3 together over the tile's rows (the
//   four xor trees transposed: the same additions, pair k's sum in lanes
//   8k .. 8k + 7);
// - thread t accumulates pairs 4 (t / 32) .. + 3, columns 2 (t % 32) +
//   64 j (+1), each K value it loads used by its 4 pairs.
// Grid (splits x n_h / hb, ceil(nb / (MLA_AB_Q / hb))), MLA_AB_THREADS
// threads, attn_b_smem bytes of dynamic shared memory (sm_80+: over 48 KiB).
// Out: part as mla_attn's; mla_attn_cmb combines them.
#define MLA_AB_Q 32
#define MLA_AB_THREADS 256
#define MLA_AB_CJ 8  // lat <= 512: column pairs per thread and pair (64 apart)
extern "C" __global__ void __launch_bounds__(MLA_AB_THREADS, 1)
mla_attn_b(const float* __restrict__ qa, const unsigned short* __restrict__ kv, int krow,
           int lat, int n_h, float scale, int start, int split, int ns, int nb,
           float* __restrict__ part) {
    extern __shared__ __align__(16) float kf[];  // [MLA_TILE][krow] f32
    __shared__ float ss[MLA_AB_Q][MLA_TILE];
    __shared__ __align__(16) float pst[MLA_TILE][MLA_AB_Q];  // p, row-major by row
    __shared__ float ms[MLA_AB_Q], ls[MLA_AB_Q], al[MLA_AB_Q];
    int t = threadIdx.x, lane = t & 31, w = t >> 5;
    int hb = n_h < MLA_AB_Q ? n_h : MLA_AB_Q;
    int qt = MLA_AB_Q / hb;  // tokens per block
    int hg = n_h / hb;
    int sp = blockIdx.x / hg, h0 = (blockIdx.x % hg) * hb;
    int z0 = blockIdx.y * qt;
    int rb = sp * split;
    // Pair q = (token z0 + q / hb, head h0 + q % hb); its end re (0: none).
    auto pair_end = [&](int q) -> int {
        int z = z0 + q / hb;
        if (z >= nb) return 0;
        int np = start + z + 1;
        return np > rb ? min(np, rb + split) : 0;
    };
    int last = min(nb, z0 + qt) - 1;  // the group's longest cache
    if (last < z0 || start + last + 1 <= rb) return;
    int re_max = min(start + last + 1, rb + split);
    // The score warp's four pairs 4w .. 4w + 3 and their queries' column pairs.
    float2 qr[4][MLA_MAXI];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        int q = 4 * w + k;
        int z = min(z0 + q / hb, nb - 1);
        const float* qp = qa + ((u64)z * n_h + h0 + q % hb) * krow;
#pragma unroll
        for (int i = 0; i < MLA_MAXI; i++) {
            int c = 2 * lane + 64 * i;
            qr[k][i] = c < krow ? make_float2(qp[c], qp[c + 1]) : make_float2(0.0f, 0.0f);
        }
    }
    if (t < MLA_AB_Q) {
        ms[t] = __int_as_float(0xff800000);  // -inf
        ls[t] = 0.0f;
    }
    // The accumulating thread's 4 pairs aq0 .. aq0 + 3 and its column pairs
    // ac + 64 j: each K value it loads feeds 4 pairs.
    int aq0 = 4 * (t / 32), ac = 2 * (t % 32);
    int a_re[4];
#pragma unroll
    for (int u = 0; u < 4; u++) a_re[u] = pair_end(aq0 + u);
    float2 acc[4][MLA_AB_CJ];
#pragma unroll
    for (int u = 0; u < 4; u++)
#pragma unroll
        for (int j = 0; j < MLA_AB_CJ; j++) acc[u][j] = make_float2(0.0f, 0.0f);
    int kw = krow >> 1;
    for (int r0 = rb; r0 < re_max; r0 += MLA_TILE) {
        int nr = min(MLA_TILE, re_max - r0);  // the group's rows (a pair's may be fewer)
        __syncthreads();
        const unsigned* src = (const unsigned*)(kv + (u64)r0 * krow);
        for (int i = t; i < nr * kw; i += MLA_AB_THREADS) {
            unsigned kk = src[i];
            ((float2*)kf)[i] = make_float2(mla_h2f(kk & 0xffff), mla_h2f(kk >> 16));
        }
        __syncthreads();
        // Scores of the warp's 4 pairs for the tile's rows: 4 chains per lane,
        // the 4 xor trees transposed (every sum the tree's; 6 shuffles, pair
        // k's sum ends in lanes 8 k .. 8 k + 7).
        for (int r = 0; r < nr; r++) {
            const float* kr = kf + r * krow;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int i = 0; i < MLA_MAXI; i++) {
                int c = 2 * lane + 64 * i;
                if (c < krow) {
                    float2 k2 = *(const float2*)(kr + c);
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        d[k] = __fmaf_rn(qr[k][i].x, k2.x, d[k]);
                        d[k] = __fmaf_rn(qr[k][i].y, k2.y, d[k]);
                    }
                }
            }
            bool b4 = lane & 16, b3 = lane & 8;
            float k0 = b4 ? d[2] : d[0], k1 = b4 ? d[3] : d[1];
            float y0 = b4 ? d[0] : d[2], y1 = b4 ? d[1] : d[3];
            k0 = __fadd_rn(k0, __shfl_xor_sync(0xffffffffu, y0, 16));
            k1 = __fadd_rn(k1, __shfl_xor_sync(0xffffffffu, y1, 16));
            float kk = b3 ? k1 : k0, yy = b3 ? k0 : k1;
            kk = __fadd_rn(kk, __shfl_xor_sync(0xffffffffu, yy, 8));
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) kk = __fadd_rn(kk, __shfl_xor_sync(0xffffffffu, kk, o));
            if ((lane & 7) == 0) ss[4 * w + (lane >> 3)][r] = __fmul_rn(kk, scale);
        }
        __syncthreads();
        // Softmax state per pair: warp w takes pairs w and w + 16.
        for (int q = w; q < MLA_AB_Q; q += MLA_AB_THREADS / 32) {
            int re = pair_end(q);
            if (r0 >= re) continue;  // not in this tile: unchanged
            int nq = min(MLA_TILE, re - r0);
            float s = lane < nq ? ss[q][lane] : __int_as_float(0xff800000);
            float mx = s;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            float m_old = ms[q];
            float m_new = fmaxf(m_old, mx);
            float p = lane < nq ? expf(__fsub_rn(s, m_new)) : 0.0f;
            float sum = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
            pst[lane][q] = p;
            __syncwarp();
            if (lane == 0) {
                float a = expf(__fsub_rn(m_old, m_new));
                al[q] = a;
                ms[q] = m_new;
                ls[q] = __fadd_rn(__fmul_rn(ls[q], a), sum);
            }
        }
        __syncthreads();
        // Each pair: its rescale, then its rows in order (its own count;
        // a pair not in this tile, or past its rows, does nothing).
        int nq[4];
        float al4[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            nq[u] = r0 < a_re[u] ? min(MLA_TILE, a_re[u] - r0) : 0;
            al4[u] = al[aq0 + u];
        }
        int nmax = max(max(nq[0], nq[1]), max(nq[2], nq[3]));
        if (nmax > 0) {
#pragma unroll
            for (int j = 0; j < MLA_AB_CJ; j++) {
                int c = ac + 64 * j;
                if (c < lat) {
#pragma unroll
                    for (int u = 0; u < 4; u++)
                        if (nq[u] > 0)
                            acc[u][j] = make_float2(__fmul_rn(acc[u][j].x, al4[u]),
                                                    __fmul_rn(acc[u][j].y, al4[u]));
                    for (int r = 0; r < nmax; r++) {
                        float2 k2 = *(const float2*)(kf + r * krow + c);
                        float4 p4 = *(const float4*)(&pst[r][aq0]);
                        float pr[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                        for (int u = 0; u < 4; u++)
                            if (r < nq[u]) {
                                acc[u][j].x = __fmaf_rn(pr[u], k2.x, acc[u][j].x);
                                acc[u][j].y = __fmaf_rn(pr[u], k2.y, acc[u][j].y);
                            }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
        if (a_re[u] == 0) continue;
        int q = aq0 + u;
        int z = z0 + q / hb, h = h0 + q % hb;
        float* o = part + (((u64)z * ns + sp) * n_h + h) * (lat + 2);
        if (t % 32 == 0) {
            o[0] = ms[q];
            o[1] = ls[q];
        }
#pragma unroll
        for (int j = 0; j < MLA_AB_CJ; j++) {
            int c = ac + 64 * j;
            if (c < lat) {
                o[2 + c] = acc[u][j].x;
                o[2 + c + 1] = acc[u][j].y;
            }
        }
    }
}

// mla_attn's splits of each (token, head) combined (grid (n_h, nb),
// MLA_THREADS): M = max m_sp, w_sp = exp(m_sp - M), L = sum l_sp w_sp and
// out[c] = (sum acc_sp[c] w_sp) / L, both fma chains over sp ascending. One
// split: w = 1, L = l, out = acc / l exactly. At most MLA_MAX_SPLITS splits.
#define MLA_MAX_SPLITS 2048
extern "C" __global__ void __launch_bounds__(MLA_THREADS)
mla_attn_cmb(const float* __restrict__ part, int lat, int n_h, const u32* __restrict__ P,
             int start, int split, int ns, float* __restrict__ att) {
    __shared__ float wsh[MLA_MAX_SPLITS];
    int h = blockIdx.x, z = blockIdx.y, t = threadIdx.x;
    int np = mla_pos0(P, start) + z + 1;
    int n = (np + split - 1) / split;
    const float* pz = part + ((u64)z * ns * n_h + h) * (lat + 2);
    u64 st = (u64)n_h * (lat + 2);
    float mx = __int_as_float(0xff800000);
    for (int sp = 0; sp < n; sp++) mx = fmaxf(mx, pz[sp * st]);
    for (int sp = t; sp < n; sp += MLA_THREADS) wsh[sp] = expf(__fsub_rn(pz[sp * st], mx));
    __syncthreads();
    float l = 0.0f;
    for (int sp = 0; sp < n; sp++) l = __fmaf_rn(pz[sp * st + 1], wsh[sp], l);
    float* o = att + ((u64)z * n_h + h) * lat;
    for (int c = t; c < lat; c += MLA_THREADS) {
        float v = 0.0f;
        for (int sp = 0; sp < n; sp++) v = __fmaf_rn(pz[sp * st + 2 + c], wsh[sp], v);
        o[c] = __fdiv_rn(v, l);
    }
}
