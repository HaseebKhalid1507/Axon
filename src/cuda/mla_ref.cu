// main be13f1f's mla_heads_mv and mla_attn, verbatim but renamed: the
// reference the rewritten kernels must equal bit for bit (mla_tests.rs).
// Test builds only (Kernels::mla_ref_module).

// Per-head products (absorb: Wk_b q_nope; expand: Wv_b o):
//   out[z][h * os + r] = sum_c W[h * rph + r][c] * x[z][h * xs + c]
// for heads h, rows r < rph, c < cols; x rows xt floats apart, out ot.
// W: type qt, rb bytes per row, n_h * rph rows, dequantized exactly (deq).
// A warp per matrix row and MLA_MV_T tokens (grid (ceil(n_h * rph / 8),
// ceil(nb / MLA_MV_T))): lane l takes c = l, l + 32, .. ascending, one
// fma chain per token, then the fixed xor tree. A token's sums are the same
// in any batch.
extern "C" __global__ void __launch_bounds__(MLA_THREADS)
mla_heads_mv_ref(int qt, const u8* __restrict__ w, u64 rb, int rph, int cols, int n_h,
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
extern "C" __global__ void __launch_bounds__(MLA_THREADS)
mla_attn_ref(const float* __restrict__ qa, const unsigned short* __restrict__ kv, int krow, int lat,
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

