
// ---------------------------------------------------------------------------
// gpt-oss (src/cuda/qwen2.rs with `oss`; llama.cpp src/models/openai-moe.cpp):
// head size 64, a sliding window (128) on alternate layers and one sink
// score per query head. Appended to KERNELS as a module of its own
// (`Kernels::module_for` on a gpt-oss model): the dense models' source,
// cache key and cubin are unchanged.
//
// The attention is decode.cu's split attention (attn_split_t) over the
// window, as gemma.cu's at head size 256: the queries' positions lo .. np
// with lo = win_lo(np, win) (win 0: every position), the splits of that
// length. The sink enters in the combine: a score with no value row, so the
// max over the splits' maxima and the sink, and the normalizer sum_s w_s l_s
// + exp(sink - M) (soft_max_ext with sinks). The batched twins run the same
// device code per query token, so decode == batched, bit for bit.

// Decode: attn_split_t<64> over the window of the query at P[P_POS], through
// `tab` (null: the plain cache). The arguments of attn_split_64 plus `win`.
extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_ws_64(const float* __restrict__ q, const unsigned short* kc,
                 const unsigned short* __restrict__ vc, const u32* __restrict__ P, int n_heads,
                 int n_kv, float scale, int max_split, float* __restrict__ part_o,
                 float* __restrict__ part_ml, const float* __restrict__ kraw,
                 const float* __restrict__ qn, const float* __restrict__ kn, float eps,
                 const float2* __restrict__ rope, unsigned short* kc_out, int win,
                 const u32* __restrict__ tab) {
    PDL_START();
    int np = (int)P[P_POS] + 1, lo = win_lo(np, win);
    attn_split_t<64>(q, kc, vc, np - lo, blockIdx.x, blockIdx.y, n_heads, n_kv, scale, max_split,
                     part_o, part_ml, Q8Kv{}, QkNorm{kraw, qn, kn, eps, rope, kc_out}, tab, lo);
}

// The splits of a query whose window has np positions merged, with the sink:
// attn_combine_dev's code (weights w_s = exp(m_s - M), the element's sum
// over the splits in order by fma), M = max(max_s m_s, sink_h) and L =
// sum_s w_s l_s (warp_sum) + exp(sink_h - M). Block blk of n / 256; act and
// y (if non-null) as attn_combine_dev's.
__device__ void oss_combine_dev(const float* __restrict__ part_o,
                                const float* __restrict__ part_ml, int np, int blk, int n,
                                int hd, int max_split, int want_k, u8* __restrict__ act,
                                float* __restrict__ y, const float* __restrict__ sinks) {
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
        float sink = sinks[h0 + wid];
        M = fmaxf(M, sink);
        float L = 0.0f;
        for (int s = lane; s < nsplit; s += 32) {
            float w = expf(__fsub_rn(ml[2 * s], M));
            ws[wid][s] = w;
            L = __fadd_rn(L, __fmul_rn(w, ml[2 * s + 1]));
        }
        L = warp_sum(L);
        if (lane == 0) linv[wid] = __fadd_rn(L, expf(__fsub_rn(sink, M)));
    }
    __syncthreads();
    float out = 0.0f;
    if (e < n) {
        int j = threadIdx.x / hd, d = e % hd;
        const float* po = part_o + (u64)(h0 + j) * max_split * hd + d;
        float O = 0.0f;
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

// attn_split_ws_64's combine: attn_combine's arguments plus win and the
// layer's sinks (n / hd floats).
extern "C" __global__ void attn_combine_ws(const float* __restrict__ part_o,
                                           const float* __restrict__ part_ml,
                                           const u32* __restrict__ P, int n, int hd, int max_split,
                                           int want_k, u8* __restrict__ act, float* __restrict__ y,
                                           int win, const float* __restrict__ sinks) {
    PDL_START();
    int np = (int)P[P_POS] + 1;
    oss_combine_dev(part_o, part_ml, np - win_lo(np, win), blockIdx.x, n, hd, max_split, want_k,
                    act, y, sinks);
}

// Batched (prefill.cu conventions): attn_split_ws_64 per query token z0 +
// blockIdx.z (its position and rows by rp; tab: paged, null: plain). Grid
// (n_kv, max_split, nz) x ATT_TILE; part_* per query of the launch.
extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_bws_64(const float* __restrict__ q, const unsigned short* __restrict__ kc,
                  const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
                  float scale, int max_split, float* __restrict__ part_o,
                  float* __restrict__ part_ml, const u32* __restrict__ rp, int win,
                  const u32* __restrict__ tab) {
    int zl = blockIdx.z, z = z0 + zl;
    u64 nh = n_heads;
    int np = (int)rp_pos(rp, start, z) + 1, lo = win_lo(np, win);
    u64 off = rp_base(rp, z, tab) * n_kv * 64;
    attn_split_t<64>(q + (u64)z * nh * 64, kc + off, vc + off, np - lo, blockIdx.x, blockIdx.y,
                     n_heads, n_kv, scale, max_split, part_o + (u64)zl * nh * max_split * 64,
                     part_ml + (u64)zl * nh * max_split * 2, Q8Kv{}, QkNorm{}, rp_tab(rp, z, tab),
                     lo);
}

// attn_split_bws_64's combine (attn_combine_ws per query token). Grid
// (n / 256, nz) x 256.
extern "C" __global__ void attn_combine_bws(const float* __restrict__ part_o,
                                            const float* __restrict__ part_ml, int start, int z0,
                                            int n, int hd, int max_split, int want_k,
                                            u8* __restrict__ act, u64 astride,
                                            const u32* __restrict__ rp, int win,
                                            const float* __restrict__ sinks) {
    int zl = blockIdx.y, z = z0 + zl;
    u64 nh = n / hd;
    int np = (int)rp_pos(rp, start, z) + 1;
    oss_combine_dev(part_o + (u64)zl * nh * max_split * hd, part_ml + (u64)zl * nh * max_split * 2,
                    np - win_lo(np, win), blockIdx.x, n, hd, max_split, want_k, act + z * astride,
                    nullptr, sinks);
}

// ---------------------------------------------------------------------------
// gpt-oss's routed experts on the GPU (specs/gptoss-gpu-experts.md, PR A):
// the twin of quant/mxfp4.rs `dot_mxfp4_q8`, bit for bit, and of the CPU's
// gpt-oss expert epilogues (model/moe.rs `Moe::act`, `down_bias`). Per
// 32-value block b of a row against a Q8_0 activation block (scale dx, 32
// i8): eight exact lane sums s_l = sum of w_i x_i over i in 4l..4l + 4,
// d = e8m0_half(e) * dx (one rounding), acc[l] = fma(d, f32(s_l), acc[l])
// for blocks ascending, then hsum8: ((l0 + l4) + (l2 + l6)) + ((l1 + l5) +
// (l3 + l7)). A lane sum is one dp4a seeded with GM_MAGIC (prefill.cu): its
// bits are 1.5 * 2^23 + s_l, so one subtract gives f32(s_l) exactly (|s_l|
// <= 4 * 12 * 128 < 2^22), no I2F (a quarter-rate conversion).
//
// The weights are in the twin's layout (`mx_repack`, at upload): a row's
// blocks' 16 qs bytes back to back (16-aligned), and their e8m0 bytes in an
// array of their own (`es` bytes per row, padded with zeros). Same values,
// so the same bits as the file's 17-byte blocks.

// ggml kvalues_mxfp4 (E2M1 doubled), 4 int8 per word.
#define MX_T0 0x03020100u // 0 1 2 3
#define MX_T1 0x0C080604u // 4 6 8 12
#define MX_T2 0xFDFEFF00u // 0 -1 -2 -3
#define MX_T3 0xF4F8FAFCu // -4 -6 -8 -12

// The 8 weights of qs word q (block bytes 4w..4w + 4): lane w's (the low
// nibbles: values 4w..4w + 4) in .x, lane w + 4's (the high nibbles: values
// 16 + 4w..) in .y, each 4 int8 in value order (llama.cpp's
// get_int_from_table_16).
__device__ __forceinline__ int2 mx_lanes(u32 q) {
    u32 hl = 0x32103210u | ((q & 0x88888888u) >> 1);
    u32 t0 = __byte_perm(__byte_perm(MX_T0, MX_T1, q), __byte_perm(MX_T2, MX_T3, q), hl);
    u32 q1 = q >> 16;
    u32 t1 = __byte_perm(__byte_perm(MX_T0, MX_T1, q1), __byte_perm(MX_T2, MX_T3, q1), hl >> 16);
    return make_int2((int)__byte_perm(t0, t1, 0x6420), (int)__byte_perm(t0, t1, 0x7531));
}

// e8m0_half(e) = 2^(e - 128), bit for bit ggml_e8m0_to_fp32_half (subnormal
// for e < 2; the module keeps subnormals: --ftz=false).
__device__ __forceinline__ float mx_scale(u32 e) {
    return __int_as_float(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
}

// f32(lane sum of w and x), exactly.
__device__ __forceinline__ float mx_lane(int w, int x) {
    return __fsub_rn(__int_as_float(__dp4a(w, x, GM_MAGIC)), GM_MAGICF);
}

// hsum8's tree (quant/mxfp4.rs; the AVX2 kernels' horizontal sum).
__device__ __forceinline__ float mx_hsum8(const float* l) {
    return __fadd_rn(__fadd_rn(__fadd_rn(l[0], l[4]), __fadd_rn(l[2], l[6])),
                     __fadd_rn(__fadd_rn(l[1], l[5]), __fadd_rn(l[3], l[7])));
}

// vmath.rs swiglu_oai1_f::<F> (alpha 1.702, limit 7): x = g.min(7) (fminf:
// NaN g gives 7, as f32::min), y = u.clamp(-7, 7) by comparisons (NaN u
// stays NaN, as f32::clamp; fminf / fmaxf would make it -7 or 7), then
// x / (1 + exp(-(1.702 x))) * (y + 1) with moe_twin.cu's exp_twin.
__device__ __forceinline__ float swiglu_oai_twin(float g, float u, bool F) {
    const float ALPHA = __int_as_float(0x3fd9db23); // 1.702f
    float x = fminf(g, 7.0f);
    float y = u < -7.0f ? -7.0f : u;
    y = y > 7.0f ? 7.0f : y;
    float e = exp_twin(-__fmul_rn(ALPHA, x), F);
    return __fmul_rn(__fdiv_rn(x, __fadd_rn(1.0f, e)), __fadd_rn(y, 1.0f));
}

// out[i] = swiglu_oai_twin(g[i], u[i]) for i < n (the tests' handle).
extern "C" __global__ void swiglu_oai_twin_k(const float* __restrict__ g,
                                             const float* __restrict__ u, int n, int fused,
                                             float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = swiglu_oai_twin(g[i], u[i], fused != 0);
}

// Row z (n floats, biases added already): swiglu_oai_twin(g, u) (F =
// fused), quantized (quant_warp) into act row z: the CPU's `Moe::act` then
// `QuantizedAct::for_types` of one pair. swiglu_quant_tw's shape: grid
// (n / 256 rounded up, rows), 32 threads.
extern "C" __global__ void swiglu_oai_quant_tw(const float* __restrict__ g,
                                               const float* __restrict__ u, int n, int fused,
                                               int want_k, u8* __restrict__ act, u64 astride) {
    int lane = threadIdx.x & 31, blk = blockIdx.x, z = blockIdx.y;
    const float* gz = g + (u64)z * n;
    const float* uz = u + (u64)z * n;
    float v[8];
    for (int i = 0; i < 8; i++) {
        int e = blk * 256 + lane * 8 + i;
        v[i] = e < n ? swiglu_oai_twin(gz[e], uz[e], fused != 0) : 0.0f;
    }
    quant_warp(v, n, blk, act + (u64)z * astride, want_k != 0);
}

// The file's MXFP4 rows (nrows rows of nblk 17-byte blocks, back to back) to
// the twin's layout: qs (16 bytes a block, nblk * 16 a row) and e (es bytes
// a row; the caller zeroes it, so the padding past nblk is 0). A thread per
// (row, block). Grid (nrows * nblk / 256 rounded up), 256 threads.
extern "C" __global__ void mx_repack(const u8* __restrict__ src, u64 nrows, int nblk, int es,
                                     u8* __restrict__ qs, u8* __restrict__ e) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * (u64)nblk) return;
    u64 r = i / (u64)nblk;
    int b = (int)(i - r * (u64)nblk);
    const u8* s = src + i * 17;
    e[r * (u64)es + b] = s[0];
    u32 w[4];
    for (int j = 0; j < 4; j++)
        w[j] = (u32)s[1 + 4 * j] | ((u32)s[2 + 4 * j] << 8) | ((u32)s[3 + 4 * j] << 16) |
               ((u32)s[4 + 4 * j] << 24);
    *(uint4*)(qs + i * 16) = make_uint4(w[0], w[1], w[2], w[3]);
}

// The grouped GEMM's tile: MX_BM = 64 rows x MX_BN = 32 tokens of one expert
// a block (128 threads, each 4 rows x 4 tokens x 8 lanes = 128 f32 chains),
// K staged MX_KB blocks (of 32 values) at a time in shared memory: the
// weights' qs and e (cp.async), then expanded to int8 lanes and f32 scales;
// the tokens' Q8_0 values and scales (cp.async, double buffered).
#define MX_BM 64
#define MX_BN 32
#define MX_KB 8
#define MX_THREADS 128
// A weight row's expanded bytes in shared memory, +16: rows 16 banks' worth
// apart (mod 128 B), so 8 lanes reading 8 consecutive rows don't conflict.
#define MX_WROW (MX_KB * 32 + 16)
// Shared bytes (moe_gemm_mx's dynamic shared memory; mx_smem in experts.rs).
#define MX_SH_WEXP 0
#define MX_SH_WS (MX_SH_WEXP + MX_BM * MX_WROW)
#define MX_SH_RAWQ (MX_SH_WS + MX_KB * MX_BM * 4)
#define MX_SH_RAWE (MX_SH_RAWQ + MX_BM * MX_KB * 16)
#define MX_SH_X (MX_SH_RAWE + MX_BM * MX_KB)
#define MX_SH_XD (MX_SH_X + 2 * MX_BN * MX_KB * 32)
#define MX_SH_BYTES (MX_SH_XD + 2 * MX_KB * MX_BN * 4)

// 16 (or 4) bytes global -> shared, zero-filled unless ok (src is then
// `base`, any valid global address: not read): cp.async on sm_80+, a plain
// copy below (the sm_61 path is out of PR A's scope, but the oss module must
// compile on every arch it loads on).
__device__ __forceinline__ void mx_cp16(void* dst, const void* src, const void* base, bool ok) {
#if __CUDA_ARCH__ >= 800
    cp16q(dst, ok ? src : base, ok);
#else
    *(uint4*)dst = ok ? *(const uint4*)src : make_uint4(0, 0, 0, 0);
#endif
}
__device__ __forceinline__ void mx_cp4(void* dst, const void* src, const void* base, bool ok) {
#if __CUDA_ARCH__ >= 800
    cp4q(dst, ok ? src : base, ok);
#else
    *(u32*)dst = ok ? *(const u32*)src : 0u;
#endif
}
__device__ __forceinline__ void mx_commit() {
#if __CUDA_ARCH__ >= 800
    cp_commit();
#endif
}
__device__ __forceinline__ void mx_wait() {
#if __CUDA_ARCH__ >= 800
    cp_wait<0>();
#endif
}

// out row s (sorted position s, ostride floats) = expert e's rows times act
// row s, in the CPU's numerics, plus bias[e * rows + r] (null: none), for
// the positions of the block's tile: tile blockIdx.x of the 32-token tiles
// of each expert's run (moe_sort's off[]: expert e's pairs are sorted
// positions off[e]..off[e + 1]; blocks past the last tile exit). qs / e the
// experts' matrices in the twin's layout (qestride / eestride bytes apart,
// es e bytes a row), n = cols (a multiple of 32). Grid (np / 32 + n_e
// rounded up, rows / 64 rounded up), MX_THREADS threads, MX_SH_BYTES of
// dynamic shared memory.
extern "C" __global__ void __launch_bounds__(MX_THREADS, 2)
moe_gemm_mx(const u8* __restrict__ qs, const u8* __restrict__ ee, u64 qestride, u64 eestride,
            int es, int rows, int n, const u8* __restrict__ act, u64 astride,
            const int* __restrict__ off, int n_e, float* __restrict__ out, u64 ostride,
            const float* __restrict__ bias) {
    extern __shared__ __align__(16) u8 mx_sh[];
    // The tile: expert ex, sorted positions s0..s0 + cnt.
    int j = blockIdx.x, ex = -1, s0 = 0, cnt = 0;
    for (int e = 0; e < n_e; e++) {
        int c = off[e + 1] - off[e], nt = (c + MX_BN - 1) / MX_BN;
        if (j < nt) {
            ex = e;
            s0 = off[e] + j * MX_BN;
            cnt = min(MX_BN, c - j * MX_BN);
            break;
        }
        j -= nt;
    }
    if (ex < 0) return;
    int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    int rq = (w & 1) * 8 + (lane & 7); // rows r0 + rq + 16 i, i < 4
    int tq = (w >> 1) * 4 + (lane >> 3); // tokens 4 tq + jj, jj < 4
    int r0 = blockIdx.y * MX_BM;
    int nk = n / 32, np = act_np(n);
    const u8* wq = qs + (u64)ex * qestride;
    const u8* we = ee + (u64)ex * eestride;
    u64 qrow = (u64)nk * 16;
    u8* wexp = mx_sh + MX_SH_WEXP;
    float* wsc = (float*)(mx_sh + MX_SH_WS);
    u8* rawq = mx_sh + MX_SH_RAWQ;
    u8* rawe = mx_sh + MX_SH_RAWE;
    // Chunk c's copies: the weights' qs and e into raw*, the tokens' values
    // and scales into buffer c & 1.
    auto issue = [&](int c) {
        int kb0 = c * MX_KB, kbn = min(MX_KB, nk - kb0);
        for (int p = tid; p < MX_BM * MX_KB; p += MX_THREADS) {
            int r = p / MX_KB, b = p % MX_KB;
            bool ok = r0 + r < rows && b < kbn;
            mx_cp16(rawq + p * 16, wq + (u64)(r0 + r) * qrow + (u64)(kb0 + b) * 16, wq, ok);
        }
        for (int p = tid; p < MX_BM * MX_KB / 4; p += MX_THREADS) {
            int r = p / (MX_KB / 4), b4 = p % (MX_KB / 4);
            bool ok = r0 + r < rows && b4 * 4 < kbn;
            mx_cp4(rawe + p * 4, we + (u64)(r0 + r) * es + kb0 + b4 * 4, we, ok);
        }
        u8* xs = mx_sh + MX_SH_X + (c & 1) * (MX_BN * MX_KB * 32);
        float* xd = (float*)(mx_sh + MX_SH_XD + (c & 1) * (MX_KB * MX_BN * 4));
        // (an act row holds np / 32 blocks, a multiple of 8 >= kb0 + MX_KB:
        // a whole chunk is always in the row)
        for (int p = tid; p < MX_BN * MX_KB * 2; p += MX_THREADS) {
            int t = p / (MX_KB * 2), h = p % (MX_KB * 2);
            const u8* a = act + (u64)(s0 + min(t, cnt - 1)) * astride;
            mx_cp16(xs + p * 16, a + np + kb0 * 32 + h * 16, act, t < cnt);
        }
        // xd[b][t]: block b's scale of token t.
        for (int p = tid; p < MX_KB * MX_BN; p += MX_THREADS) {
            int b = p / MX_BN, t = p % MX_BN;
            const u8* a = act + (u64)(s0 + min(t, cnt - 1)) * astride;
            mx_cp4(xd + p, a + 2 * np + np / 64 + np / 8 + (kb0 + b) * 4, act, t < cnt);
        }
        mx_commit();
    };
    float acc[4][4][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int jj = 0; jj < 4; jj++)
#pragma unroll
            for (int l = 0; l < 8; l++) acc[i][jj][l] = 0.0f;
    int nc = (nk + MX_KB - 1) / MX_KB;
    issue(0);
    for (int c = 0; c < nc; c++) {
        int kbn = min(MX_KB, nk - c * MX_KB);
        mx_wait();
        __syncthreads(); // chunk c landed; every thread is done with chunk c - 1
        for (int p = tid; p < MX_BM * MX_KB; p += MX_THREADS) {
            int r = p / MX_KB, b = p % MX_KB;
            uint4 q = *(const uint4*)(rawq + p * 16);
            int2 v0 = mx_lanes(q.x), v1 = mx_lanes(q.y), v2 = mx_lanes(q.z), v3 = mx_lanes(q.w);
            int4* d = (int4*)(wexp + r * MX_WROW + b * 32);
            d[0] = make_int4(v0.x, v1.x, v2.x, v3.x); // lanes 0..3
            d[1] = make_int4(v0.y, v1.y, v2.y, v3.y); // lanes 4..7
            wsc[b * MX_BM + r] = mx_scale(rawe[p]);
        }
        __syncthreads(); // expanded; raw* free for the next chunk
        if (c + 1 < nc) issue(c + 1);
        const u8* xs = mx_sh + MX_SH_X + (c & 1) * (MX_BN * MX_KB * 32);
        const float* xd = (const float*)(mx_sh + MX_SH_XD + (c & 1) * (MX_KB * MX_BN * 4));
        for (int b = 0; b < kbn; b++) {
            float d[4][4];
            float4 dx = *(const float4*)(xd + b * MX_BN + tq * 4);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float s = wsc[b * MX_BM + rq + 16 * i];
                d[i][0] = __fmul_rn(s, dx.x);
                d[i][1] = __fmul_rn(s, dx.y);
                d[i][2] = __fmul_rn(s, dx.z);
                d[i][3] = __fmul_rn(s, dx.w);
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int4 wv[4];
#pragma unroll
                for (int i = 0; i < 4; i++)
                    wv[i] = *(const int4*)(wexp + (rq + 16 * i) * MX_WROW + b * 32 + h * 16);
#pragma unroll
                for (int jj = 0; jj < 4; jj++) {
                    int4 xv = *(const int4*)(xs + ((tq * 4 + jj) * MX_KB + b) * 32 + h * 16);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        float* a = acc[i][jj] + 4 * h;
                        a[0] = __fmaf_rn(d[i][jj], mx_lane(wv[i].x, xv.x), a[0]);
                        a[1] = __fmaf_rn(d[i][jj], mx_lane(wv[i].y, xv.y), a[1]);
                        a[2] = __fmaf_rn(d[i][jj], mx_lane(wv[i].z, xv.z), a[2]);
                        a[3] = __fmaf_rn(d[i][jj], mx_lane(wv[i].w, xv.w), a[3]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = r0 + rq + 16 * i;
        if (r >= rows) continue;
        float bv = bias ? bias[(u64)ex * rows + r] : 0.0f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            int t = tq * 4 + jj;
            if (t >= cnt) continue;
            float y = mx_hsum8(acc[i][jj]);
            if (bias) y = __fadd_rn(y, bv);
            out[(u64)(s0 + t) * ostride + r] = y;
        }
    }
}

// The decode GEMV's row: thread lane l (< 8) of a row runs lane l's chain
// over the row's blocks in order (its 4 weights from qs word l & 3, low or
// high nibbles), the 8 chains are summed in hsum8's tree by shuffles (xor 4,
// 2, 1: lane 0 gets ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))), and
// the bias added. The value is in lane 0 of the row's 8 threads.
__device__ __forceinline__ float mx_row(const u8* __restrict__ q, const u8* __restrict__ e,
                                        int nk, const u8* __restrict__ act, int n, int l) {
    int np = act_np(n);
    const int* xq = (const int*)(act + np);
    const float* xd = (const float*)(act + 2 * np + np / 64 + np / 8);
    float a = 0.0f;
    for (int b = 0; b < nk; b++) {
        u32 qw = *(const u32*)(q + b * 16 + 4 * (l & 3));
        int2 v = mx_lanes(qw);
        float d = __fmul_rn(mx_scale(e[b]), xd[b]);
        a = __fmaf_rn(d, mx_lane(l < 4 ? v.x : v.y, xq[b * 8 + l]), a);
    }
    // hsum8: lane l < 4 adds l + 4's first (l0 + l4 ...), then pairs.
    a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 4));
    a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 2));
    a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 1));
    return a;
}

// The decode expert GEMV (moe_gemv_tw's job for MXFP4): out[i * ostride + r]
// = row r of expert ids[i] (qestride / eestride bytes apart) times slot i's
// activations (act + i aslot; aslot 0: one token's for every slot), plus
// bias[ids[i] * rows + r] (null: none). 16 rows a block, 8 threads a row.
// Grid (rows / 16 rounded up, k), 128 threads.
extern "C" __global__ void __launch_bounds__(128)
moe_gemv_mx(const u8* __restrict__ qs, const u8* __restrict__ ee, u64 qestride, u64 eestride,
            int es, int rows, int n, const int* __restrict__ ids, const u8* __restrict__ act,
            u64 aslot, float* __restrict__ out, u64 ostride, const float* __restrict__ bias) {
    PDL_START();
    int r = blockIdx.x * 16 + (threadIdx.x >> 3), l = threadIdx.x & 7, i = blockIdx.y;
    if (r >= rows) r = rows - 1; // (whole warps shuffle; the duplicate isn't stored)
    int nk = n / 32;
    u64 e = (u64)ids[i];
    float y = mx_row(qs + e * qestride + (u64)r * nk * 16, ee + e * eestride + (u64)r * es, nk,
                     act + (u64)i * aslot, n, l);
    if (l == 0 && blockIdx.x * 16 + (threadIdx.x >> 3) < rows) {
        if (bias) y = __fadd_rn(y, bias[e * rows + r]);
        out[(u64)i * ostride + r] = y;
    }
}

// Decode: gate (z = 0) and up (z = 1) of slot i in one launch, each
// moe_gemv_mx's chain for its row (the same bits), one token's activations
// for every slot. Gate and up of one shape. Grid (rows / 16 rounded up, k,
// 2), 128 threads.
extern "C" __global__ void __launch_bounds__(128)
moe_gu_mx(const u8* __restrict__ gq, const u8* __restrict__ ge, const u8* __restrict__ uq,
          const u8* __restrict__ ue, u64 qestride, u64 eestride, int es, int rows, int n,
          const int* __restrict__ ids, const u8* __restrict__ act, float* __restrict__ gout,
          float* __restrict__ uout, const float* __restrict__ gbias,
          const float* __restrict__ ubias) {
    PDL_START();
    int r = blockIdx.x * 16 + (threadIdx.x >> 3), l = threadIdx.x & 7, i = blockIdx.y;
    bool up = blockIdx.z == 1;
    if (r >= rows) r = rows - 1;
    int nk = n / 32;
    u64 e = (u64)ids[i];
    float y = mx_row((up ? uq : gq) + e * qestride + (u64)r * nk * 16,
                     (up ? ue : ge) + e * eestride + (u64)r * es, nk, act, n, l);
    if (l == 0 && blockIdx.x * 16 + (threadIdx.x >> 3) < rows) {
        const float* bias = up ? ubias : gbias;
        if (bias) y = __fadd_rn(y, bias[e * rows + r]);
        (up ? uout : gout)[(u64)i * rows + r] = y;
    }
}
