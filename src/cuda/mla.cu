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
// A warp per matrix row and MLA_MV_T tokens (grid (ceil(n_h * rph / 8),
// ceil(nb / MLA_MV_T))): lane l takes c = l, l + 32, .. ascending, one
// fma chain per token, then the fixed xor tree. A token's sums are the same
// in any batch.
#define MLA_MV_T 8
extern "C" __global__ void __launch_bounds__(MLA_THREADS)
mla_heads_mv(int qt, const u8* __restrict__ w, u64 rb, int rph, int cols, int n_h,
             const float* __restrict__ x, int xs, u64 xt, float* __restrict__ out, int os,
             u64 ot, int nb) {
    int lane = threadIdx.x & 31;
    int gr = blockIdx.x * (MLA_THREADS / 32) + (threadIdx.x >> 5);
    if (gr >= n_h * rph) return;
    int h = gr / rph, r = gr % rph;
    int z0 = blockIdx.y * MLA_MV_T;
    int nt = min(MLA_MV_T, nb - z0);
    const u8* row = w + (u64)gr * rb;
    const float* xz = x + (u64)z0 * xt + (u64)h * xs;
    float acc[MLA_MV_T];
#pragma unroll
    for (int t = 0; t < MLA_MV_T; t++) acc[t] = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        float wv = deq(qt, row, c);
#pragma unroll
        for (int t = 0; t < MLA_MV_T; t++)
            if (t < nt) acc[t] = __fmaf_rn(wv, xz[(u64)t * xt + c], acc[t]);
    }
#pragma unroll
    for (int t = 0; t < MLA_MV_T; t++) {
        float v = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0 && t < nt) out[(u64)(z0 + t) * ot + (u64)h * os + r] = v;
    }
}

// The attention of nb tokens' absorbed queries over their cache rows 0..np
// (np = pos0 + z + 1): o_h = softmax(Q_h . K_t * scale) . K_t[..lat], in
// splits of `split` rows (a multiple of MLA_TILE) from row 0: split sp of a
// token covers rows sp * split .. min(np, (sp + 1) * split), so a token has
// ceil(np / split) of them, whatever its batch.
// Grid (n_h / hpb, nb, max splits), MLA_THREADS threads; hpb heads per block
// (a power of two <= 8 dividing n_h); a block past its token's last split
// does nothing. Rows in tiles of MLA_TILE, the tile's K rows in shared
// memory (f16), an online softmax per head across the split's tiles in
// order. Warp w scores head w of the block (lane l: columns 2l, 2l + 1,
// 2l + 64, ..: one fma chain per row, then the xor tree), keeps the head's
// max m, sum l and the tile's p_t (= exp(s_t - m_new)); thread t then
// rescales and accumulates head t / tph's columns 2 (t % tph) + 2 tph j
// (+1) over the tile's rows in order. Out: the split's (m, l, acc[lat])
// per head at part[((z * ns + sp) * n_h + h) * (lat + 2)] (mla_attn_cmb
// combines them). The ops of a (token, head) depend on np only.
#define MLA_TILE 32
#define MLA_MAXI 9   // krow <= 576: 64-column steps of the score
#define MLA_MAXC 16  // lat <= 512 at tph 32: column pairs per thread
extern "C" __global__ void __launch_bounds__(MLA_THREADS)
mla_attn(const float* __restrict__ qa, const unsigned short* __restrict__ kv, int krow, int lat,
         int n_h, int hpb, float scale, const u32* __restrict__ P, int start, int split, int ns,
         float* __restrict__ part) {
    extern __shared__ unsigned short ks[];  // [MLA_TILE][krow]
    __shared__ float ps[8][MLA_TILE];
    __shared__ float ms[8], ls[8], al[8];
    int t = threadIdx.x, lane = t & 31, w = t >> 5;
    int z = blockIdx.y, h0 = blockIdx.x * hpb, sp = blockIdx.z;
    int np = mla_pos0(P, start) + z + 1;
    int rb = sp * split, re = min(np, rb + split);
    if (rb >= np) return;
    int tph = MLA_THREADS / hpb;
    // The score warp's query (head h0 + w, if w < hpb): column pairs.
    float2 qr[MLA_MAXI];
    bool scorer = w < hpb;
    if (scorer) {
        const float* qh = qa + ((u64)z * n_h + h0 + w) * krow;
#pragma unroll
        for (int i = 0; i < MLA_MAXI; i++) {
            int c = 2 * lane + 64 * i;
            qr[i] = c < krow ? make_float2(qh[c], qh[c + 1]) : make_float2(0.0f, 0.0f);
        }
        if (lane == 0) {
            ms[w] = __int_as_float(0xff800000);  // -inf
            ls[w] = 0.0f;
        }
    }
    // The accumulating thread's head and columns.
    int ah = t / tph, ac = 2 * (t % tph);
    float2 acc[MLA_MAXC];
#pragma unroll
    for (int j = 0; j < MLA_MAXC; j++) acc[j] = make_float2(0.0f, 0.0f);
    int kw = krow >> 1;  // u32 words per row
    for (int r0 = rb; r0 < re; r0 += MLA_TILE) {
        int nr = min(MLA_TILE, re - r0);
        __syncthreads();
        const unsigned* src = (const unsigned*)(kv + (u64)r0 * krow);
        for (int i = t; i < nr * kw; i += MLA_THREADS) ((unsigned*)ks)[i] = src[i];
        __syncthreads();
        if (scorer) {
            float s = __int_as_float(0xff800000);
            for (int r = 0; r < nr; r++) {
                const unsigned short* kr = ks + r * krow;
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
                if (lane == r) s = __fmul_rn(d, scale);
            }
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
                        unsigned kk = *(const unsigned*)(ks + r * krow + c);
                        float p = ps[ah][r];
                        v.x = __fmaf_rn(p, mla_h2f(kk & 0xffff), v.x);
                        v.y = __fmaf_rn(p, mla_h2f(kk >> 16), v.y);
                    }
                    acc[j] = v;
                }
            }
        }
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

// deepseek2's router after moe_logits (one token per block, MOE_THREADS;
// grid nb): p = sigmoid(logits) (sigm 1: 1 / (1 + exp(-x))) or softmax
// (moe_topk's ops); selection by p + bias (bias null: p), k rounds of
// argmax (ties to the lower index); w_i = p[id_i] (unbiased), renormalized
// (norm: s = max(((w_0 + w_1) + ..), 6.103515625e-5), w /= s) then times
// scale. llama.cpp build_moe_ffn's order. Dynamic shared memory: 2 n_e
// floats.
extern "C" __global__ void __launch_bounds__(MOE_THREADS)
mla_topk(const float* __restrict__ logits, int n_e, int k, int sigm,
         const float* __restrict__ bias, int norm, float scale, int* __restrict__ ids,
         float* __restrict__ w) {
    extern __shared__ float sp[];
    float* p = sp;
    float* sel = sp + n_e;
    __shared__ float red[MOE_THREADS];
    __shared__ int rid[MOE_THREADS];
    int t = threadIdx.x;
    size_t tok = blockIdx.x;
    for (int e = t; e < n_e; e += MOE_THREADS) p[e] = logits[tok * n_e + e];
    __syncthreads();
    if (sigm) {
        for (int e = t; e < n_e; e += MOE_THREADS)
            p[e] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-p[e])));
    } else {
        float mx = __int_as_float(0xff800000);
        for (int e = t; e < n_e; e += MOE_THREADS) mx = fmaxf(mx, p[e]);
        red[t] = mx;
        __syncthreads();
        for (int s = MOE_THREADS / 2; s > 0; s >>= 1) {
            if (t < s) red[t] = fmaxf(red[t], red[t + s]);
            __syncthreads();
        }
        mx = red[0];
        __syncthreads();
        float sum = 0.0f;
        for (int e = t; e < n_e; e += MOE_THREADS) {
            float v = expf(__fsub_rn(p[e], mx));
            p[e] = v;
            sum = __fadd_rn(sum, v);
        }
        sum = moe_tree_sum(sum, red);
        for (int e = t; e < n_e; e += MOE_THREADS) p[e] = __fdiv_rn(p[e], sum);
    }
    __syncthreads();
    for (int e = t; e < n_e; e += MOE_THREADS) sel[e] = bias ? __fadd_rn(p[e], bias[e]) : p[e];
    __syncthreads();
    for (int r = 0; r < k; r++) {
        float bv = __int_as_float(0xff800000);
        int bi = 0x7fffffff;
        for (int e = t; e < n_e; e += MOE_THREADS)
            if (sel[e] == sel[e] && moe_beats(sel[e], e, bv, bi)) {
                bv = sel[e];
                bi = e;
            }
        red[t] = bv;
        rid[t] = bi;
        __syncthreads();
        for (int s = MOE_THREADS / 2; s > 0; s >>= 1) {
            if (t < s && moe_beats(red[t + s], rid[t + s], red[t], rid[t])) {
                red[t] = red[t + s];
                rid[t] = rid[t + s];
            }
            __syncthreads();
        }
        if (t == 0) {
            int e = rid[0] == 0x7fffffff ? 0 : rid[0];
            ids[tok * k + r] = e;
            w[tok * k + r] = p[e];
            sel[e] = __int_as_float(0xff800000);
        }
        __syncthreads();
    }
    if (t == 0) {
        if (norm) {
            float s = 0.0f;
            for (int r = 0; r < k; r++) s = __fadd_rn(s, w[tok * k + r]);
            s = fmaxf(s, 6.103515625e-5f);
            for (int r = 0; r < k; r++) w[tok * k + r] = __fdiv_rn(w[tok * k + r], s);
        }
        if (scale != 1.0f)
            for (int r = 0; r < k; r++) w[tok * k + r] = __fmul_rn(w[tok * k + r], scale);
    }
}
