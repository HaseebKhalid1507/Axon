
// ---------------------------------------------------------------------------
// Batched kernels: nb tokens through a Qwen2 model at once (prefill and the
// speculative verify pass, src/cuda/qwen2.rs).
//
// The contract: every token's KV cache rows and logits are BIT-IDENTICAL to
// the decode kernels' (one token at a time), whatever the batch size or the
// token's place in it. The elementwise kernels run decode.cu's code per
// token. The matrix products reproduce the decode GEMV's arithmetic exactly:
// the same integer block sums (exact, however they are computed), then the
// same float ops in the same order. That order is the GEMV's: lane t = 8g + p
// of its 128-thread block (K-quants; 4g + p for 32-value blocks) keeps a
// running sum over blocks b = g, g + G, g + 2G, ... (G groups), and cta_sum
// adds the 128 lane sums by a fixed tree (a perfect binary tree per warp,
// then the 4 warps in order). A batched product therefore keeps, per output,
// the running sums of the lanes it is in the middle of, and folds each group
// into that tree when its last block is done: groups in order, blocks of a
// group in order.
//
// Token-batched buffers: token z's activation (decode.cu Act) at act + z *
// astride; f32 rows at x + z * stride.

// ---------------------------------------------------------------------------
// Elementwise, one token per grid row (blockIdx.y or blockIdx.x as noted).

// x[z] = embedding row tokens[z] (embed). Grid (ceil(n / 256), nb) x 256.
extern "C" __global__ void embed_b(int qt, const u8* __restrict__ w, u64 row_bytes, int n,
                                   const u32* __restrict__ tokens, float* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, z = blockIdx.y;
    if (i < n) x[(u64)z * n + i] = deq(qt, w + (u64)tokens[z] * row_bytes, i);
}

// rope[z] = rope_table at position start + z. Grid nb x half.
extern "C" __global__ void rope_table_b(const double* __restrict__ freq, int half, int start,
                                        float2* __restrict__ rope) {
    int i = threadIdx.x, z = blockIdx.x;
    if (i < half) {
        double s, c;
        sincos((double)(u32)(start + z) * freq[i], &s, &c);
        rope[(u64)z * half + i] = make_float2((float)s, (float)c);
    }
}

// rmsnorm_quant of row z of x (n floats per row) into act[z]. Grid nb x 1024.
extern "C" __global__ void rmsnorm_quant_b(const float* __restrict__ x, const float* __restrict__ g,
                                           float eps, int n, int want_k, u8* __restrict__ act,
                                           u64 astride) {
    int z = blockIdx.x;
    rmsnorm_quant_dev(x + (u64)z * n, g, eps, n, want_k, act + z * astride, nullptr);
}

// hid = silu(gate) * up (gate_up's ops), quantized into act[z] (the decode
// step's `quantize` of hid). Grid (ceil(n / 256), nb) x 32.
extern "C" __global__ void swiglu_quant_b(const float* __restrict__ gate,
                                          const float* __restrict__ up, int n, int want_k,
                                          u8* __restrict__ act, u64 astride) {
    int lane = threadIdx.x & 31, blk = blockIdx.x, z = blockIdx.y;
    const float* gz = gate + (u64)z * n;
    const float* uz = up + (u64)z * n;
    float v[8];
    for (int i = 0; i < 8; i++) {
        int e = blk * 256 + lane * 8 + i;
        float y0 = e < n ? gz[e] : 0.0f, y1 = e < n ? uz[e] : 0.0f;
        float h = __fmul_rn(__fdiv_rn(y0, __fadd_rn(1.0f, expf(-y0))), y1);
        v[i] = e < n ? h : 0.0f;
    }
    quant_warp(v, n, blk, act + z * astride, want_k);
}

// RoPE and the cache store of qkv_rope, on the products (biases added) in
// qkv: row z = [nq queries | nk keys | nk values]. Items per token: nq / 2
// query pairs, nk / 2 key pairs, nk values. Grid (ceil(items / 128), nb).
extern "C" __global__ void rope_kv_b(const float* __restrict__ qkv, int nq, int nk, int hd,
                                     int start, const float2* __restrict__ rope,
                                     float* __restrict__ q, unsigned short* __restrict__ kc,
                                     unsigned short* __restrict__ vc) {
    int z = blockIdx.y, item = blockIdx.x * blockDim.x + threadIdx.x;
    int half = hd >> 1, npq = nq >> 1, npk = nk >> 1;
    const float* y = qkv + (u64)z * (nq + 2 * nk);
    u64 pos = (u64)(start + z);
    if (item >= npq + npk + nk) return;
    if (item >= npq + npk) {
        int r = item - npq - npk;
        vc[pos * nk + r] = kv_half(y[nq + nk + r]);
        return;
    }
    bool isq = item < npq;
    int it = isq ? item : item - npq;
    int r0 = (it / half) * hd + it % half, r1 = r0 + half;
    const float* yy = isq ? y : y + nq;
    float y0 = yy[r0], y1 = yy[r1];
    float2 sc = rope[(u64)z * half + it % half];
    float sn = sc.x, cs = sc.y;
    float o0 = __fsub_rn(__fmul_rn(y0, cs), __fmul_rn(y1, sn));
    float o1 = __fadd_rn(__fmul_rn(y0, sn), __fmul_rn(y1, cs));
    if (isq) {
        q[(u64)z * nq + r0] = o0;
        q[(u64)z * nq + r1] = o1;
    } else {
        kc[pos * nk + r0] = kv_half(o0);
        kc[pos * nk + r1] = kv_half(o1);
    }
}

// Decode attention for query tokens z0 + blockIdx.z (position start + z),
// all of whose keys and values are in the cache: the decode schedule and
// code per query (same splits, tiles and sums). q: [token][n_heads * hd];
// part_*: per query of this launch, as attn_split's.
template <int HD>
__device__ __forceinline__ void attn_split_bt(const float* q, const unsigned short* kc,
                                              const unsigned short* vc, int start, int z0,
                                              int n_heads, int n_kv, float scale, int max_split,
                                              float* part_o, float* part_ml) {
    int zl = blockIdx.z, z = z0 + zl;
    u64 nh = n_heads;
    attn_split_t<HD>(q + (u64)z * nh * HD, kc, vc, start + z + 1, blockIdx.x, blockIdx.y, n_heads,
                     n_kv, scale, max_split, part_o + (u64)zl * nh * max_split * HD,
                     part_ml + (u64)zl * nh * max_split * 2);
}

extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_b_64(const float* __restrict__ q, const unsigned short* __restrict__ kc,
                const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
                float scale, int max_split, float* __restrict__ part_o,
                float* __restrict__ part_ml) {
    attn_split_bt<64>(q, kc, vc, start, z0, n_heads, n_kv, scale, max_split, part_o, part_ml);
}

extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_b_128(const float* __restrict__ q, const unsigned short* __restrict__ kc,
                 const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
                 float scale, int max_split, float* __restrict__ part_o,
                 float* __restrict__ part_ml) {
    attn_split_bt<128>(q, kc, vc, start, z0, n_heads, n_kv, scale, max_split, part_o, part_ml);
}

// attn_split_b for queries whose every split has at most 32 positions (np <=
// 32 max_split: att_split's chunk <= ATT_MIN): one warp per split, 4 splits
// of one (query, kv head) per block, instead of a 128-thread block per split
// with a 128-position tile 3/4 empty. The ops per value are attn_split_t's,
// with its one tile: score chains over d in order; the max; p = exp(s - m);
// each lane's partial sum of the tile's p as it adds them (its own entry,
// then the three padding entries of its column, exp(-inf - m)); warp_sum;
// value sums per row slot (rows rs, rs + nrs, ...) in order, slots merged in
// order. Grid (n_kv, ceil(max_split / 4), nz) x 128.
template <int HD>
__device__ __forceinline__ void attn_warp_bt(const float* __restrict__ q,
                                             const unsigned short* __restrict__ kc,
                                             const unsigned short* __restrict__ vc, int start,
                                             int z0, int n_heads, int n_kv, float scale,
                                             int max_split, float* __restrict__ part_o,
                                             float* __restrict__ part_ml) {
    __shared__ float qs[MAX_GROUP * HD];
    __shared__ float ps[4][MAX_GROUP][32];
    int kvh = blockIdx.x, zl = blockIdx.z, z = z0 + zl, tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31, s = blockIdx.y * 4 + warp;
    int np = start + z + 1, nsplit, chunk;
    att_split(np, max_split, &nsplit, &chunk);
    int group = n_heads / n_kv, kv_dim = n_kv * HD;
    u64 nh = n_heads;
    const float* qz = q + (u64)z * nh * HD + (u64)kvh * group * HD;
    for (int i = tid; i < group * HD; i += 128) qs[i] = qz[i];
    __syncthreads();
    if (s >= nsplit) return;
    float* po = part_o + (u64)zl * nh * max_split * HD;
    float* pm = part_ml + (u64)zl * nh * max_split * 2;
    if (chunk > 32) __trap(); // the host sends these to attn_split_b
    int tbeg = s * chunk, tend = min(np, tbeg + chunk), tn = tend - tbeg;
    if (tn <= 0) { // attn_split_t's empty split
        for (int i = lane; i < group * HD; i += 32)
            po[((u64)(kvh * group + i / HD) * max_split + s) * HD + i % HD] = 0.0f;
        if (lane < group) {
            pm[((u64)(kvh * group + lane) * max_split + s) * 2] = -INFINITY;
            pm[((u64)(kvh * group + lane) * max_split + s) * 2 + 1] = 0.0f;
        }
        return;
    }
    // Scores: lane = position tbeg + lane.
    float sc[MAX_GROUP];
#pragma unroll
    for (int g = 0; g < MAX_GROUP; g++) sc[g] = 0.0f;
    {
        // Every lane (no divergence before the shuffles below): lanes past
        // the split score a valid row, and their scores are dropped.
        const uint4* kr =
            (const uint4*)(kc + (u64)(tbeg + min(lane, tn - 1)) * kv_dim + kvh * HD);
        // The row in two halves (fewer registers live), loads before use.
        uint4 kk[HD / 16];
#pragma unroll
        for (int d8 = 0; d8 < HD / 8; d8++) {
            if (d8 % (HD / 16) == 0)
#pragma unroll
                for (int u = 0; u < HD / 16; u++) kk[u] = kr[d8 + u];
            uint4 kv = kk[d8 % (HD / 16)];
            float k[8] = {h2f(kv.x & 0xffff), h2f(kv.x >> 16), h2f(kv.y & 0xffff),
                          h2f(kv.y >> 16),    h2f(kv.z & 0xffff), h2f(kv.z >> 16),
                          h2f(kv.w & 0xffff), h2f(kv.w >> 16)};
#pragma unroll
            for (int g = 0; g < MAX_GROUP; g++) {
                if (g < group) {
                    const float* qq = qs + g * HD + d8 * 8;
                    float a = sc[g];
#pragma unroll
                    for (int i = 0; i < 8; i++) a = __fmaf_rn(qq[i], k[i], a);
                    sc[g] = a;
                }
            }
        }
    }
    const int pairs = HD / 2, nrs = ATT_TILE / pairs, KP = HD / 64;
    // Softmax of every head at once (each head's reductions are warp_max's
    // and warp_sum's trees, the heads interleaved round by round).
    float sg[MAX_GROUP], mx[MAX_GROUP], mrun[MAX_GROUP], lrun[MAX_GROUP];
#pragma unroll
    for (int g = 0; g < MAX_GROUP; g++) {
        sg[g] = lane < tn && g < group ? __fmul_rn(sc[g], scale) : -INFINITY;
        mx[g] = sg[g];
    }
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int g = 0; g < MAX_GROUP; g++)
            mx[g] = fmaxf(mx[g], __shfl_xor_sync(0xffffffffu, mx[g], o));
    float ls[MAX_GROUP];
#pragma unroll
    for (int g = 0; g < MAX_GROUP; g++) {
        float mn = fmaxf(-INFINITY, mx[g]);
        float p = expf(__fsub_rn(sg[g], mn));
        float pad = expf(__fsub_rn(-INFINITY, mn));
        ls[g] = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.0f, p), pad), pad), pad);
        if (g < group) ps[warp][g][lane] = p;
        mrun[g] = mn;
    }
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int g = 0; g < MAX_GROUP; g++)
            ls[g] = __fadd_rn(ls[g], __shfl_xor_sync(0xffffffffu, ls[g], o));
#pragma unroll
    for (int g = 0; g < MAX_GROUP; g++)
        lrun[g] = __fadd_rn(__fmul_rn(0.0f, 0.0f), ls[g]); // alpha = 0: the first tile
    __syncwarp();
    // Values of all (<= 32) rows loaded at once: one wait, not one per row.
    const unsigned short* vb = vc + (u64)tbeg * kv_dim + kvh * HD;
    u32 vv[32][KP];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int k = 0; k < KP; k++)
            vv[i][k] = i < tn ? *(const u32*)(vb + (u64)i * kv_dim + 2 * (lane + 32 * k)) : 0;
    // Values: lane owns dim pairs dp = lane + 32 k (k < HD / 64) of every row
    // slot rs < nrs; rows rs, rs + nrs, ... in order.
    float o0[MAX_GROUP][KP][nrs], o1[MAX_GROUP][KP][nrs];
#pragma unroll
    for (int g = 0; g < MAX_GROUP; g++)
#pragma unroll
        for (int k = 0; k < KP; k++)
#pragma unroll
            for (int r = 0; r < nrs; r++) {
                o0[g][k][r] = __fmul_rn(0.0f, 0.0f);
                o1[g][k][r] = __fmul_rn(0.0f, 0.0f);
            }
#pragma unroll
    for (int i0 = 0; i0 < 32; i0 += nrs) {
#pragma unroll
        for (int r = 0; r < nrs; r++) {
            int i = i0 + r;
            if (i < tn) {
                float v0[KP], v1[KP];
#pragma unroll
                for (int k = 0; k < KP; k++) v0[k] = h2f(vv[i][k] & 0xffff), v1[k] = h2f(vv[i][k] >> 16);
#pragma unroll
                for (int g = 0; g < MAX_GROUP; g++) {
                    if (g < group) {
                        float p = ps[warp][g][i];
#pragma unroll
                        for (int k = 0; k < KP; k++) {
                            o0[g][k][r] = __fmaf_rn(p, v0[k], o0[g][k][r]);
                            o1[g][k][r] = __fmaf_rn(p, v1[k], o1[g][k][r]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int g = 0; g < MAX_GROUP; g++) {
        if (g < group) {
            int h = kvh * group + g;
#pragma unroll
            for (int k = 0; k < KP; k++) {
                float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                for (int r = 0; r < nrs; r++) {
                    a0 = __fadd_rn(a0, o0[g][k][r]);
                    a1 = __fadd_rn(a1, o1[g][k][r]);
                }
                int d = 2 * (lane + 32 * k);
                po[((u64)h * max_split + s) * HD + d] = a0;
                po[((u64)h * max_split + s) * HD + d + 1] = a1;
            }
            if (lane == 0) {
                pm[((u64)h * max_split + s) * 2] = mrun[g];
                pm[((u64)h * max_split + s) * 2 + 1] = lrun[g];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 4)
attn_warp_b_64(const float* __restrict__ q, const unsigned short* __restrict__ kc,
               const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
               float scale, int max_split, float* __restrict__ part_o,
               float* __restrict__ part_ml) {
    attn_warp_bt<64>(q, kc, vc, start, z0, n_heads, n_kv, scale, max_split, part_o, part_ml);
}

extern "C" __global__ void __launch_bounds__(128, 4)
attn_warp_b_128(const float* __restrict__ q, const unsigned short* __restrict__ kc,
                const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
                float scale, int max_split, float* __restrict__ part_o,
                float* __restrict__ part_ml) {
    attn_warp_bt<128>(q, kc, vc, start, z0, n_heads, n_kv, scale, max_split, part_o, part_ml);
}

// attn_combine for the queries of an attn_split_b launch, into act[z]: its
// ops in its order (attn_combine_dev), the split sums' loads batched 16 at
// a time. Grid (n / 256, nz) x 256.
extern "C" __global__ void attn_combine_b(const float* __restrict__ part_o,
                                          const float* __restrict__ part_ml, int start, int z0,
                                          int n, int hd, int max_split, int want_k,
                                          u8* __restrict__ act, u64 astride) {
    __shared__ float vals[256];
    __shared__ float ws[256 / 64][MAX_SPLIT];
    __shared__ float linv[256 / 64];
    int zl = blockIdx.y, z = z0 + zl, blk = blockIdx.x;
    u64 nh = n / hd;
    const float* pob = part_o + (u64)zl * nh * max_split * hd;
    const float* pmb = part_ml + (u64)zl * nh * max_split * 2;
    u8* actz = act + z * astride;
    int e = blk * 256 + threadIdx.x, lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int np = start + z + 1, nsplit, chunk;
    att_split(np, max_split, &nsplit, &chunk);
    int h0 = blk * 256 / hd, nhb = min(256 / hd, n / hd - h0);
    if (wid < nhb) {
        const float* ml = pmb + (u64)(h0 + wid) * max_split * 2;
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
        const float* po = pob + (u64)(h0 + j) * max_split * hd + d;
        float O = 0.0f;
        for (int s0 = 0; s0 < nsplit; s0 += 16) {
            float pv[16];
#pragma unroll
            for (int u = 0; u < 16; u++) pv[u] = s0 + u < nsplit ? po[(u64)(s0 + u) * hd] : 0.0f;
#pragma unroll
            for (int u = 0; u < 16; u++)
                if (s0 + u < nsplit) O = __fmaf_rn(ws[j][s0 + u], pv[u], O);
        }
        out = __fdiv_rn(O, linv[j]);
    }
    vals[threadIdx.x] = out;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v[8];
        for (int i = 0; i < 8; i++) v[i] = vals[threadIdx.x * 8 + i];
        quant_warp(v, n, blk, actz, want_k);
    }
}

// argmax of row z of x (n per row) into out[z]. Grid nb x 1024.
extern "C" __global__ void argmax_b(const float* __restrict__ x, int n, u32* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* xz = x + (u64)blockIdx.x * n;
    float bv = -INFINITY;
    int bi = -1;
    // argmax's scan (the merge rule makes any order give the same index).
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (xz[i] >= bv) bv = xz[i], bi = i;
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
        if (lane == 0) out[blockIdx.x] = bi < 0 ? (u32)(n - 1) : (u32)bi;
    }
}

// The output of a batched product: v = the row's sum (+ bias[r]); out = v
// (mode 0) or out += v (mode 1), as gemv's.
__device__ __forceinline__ void gm_store(float y, const float* bias, int mode, float* out, int r) {
    float v = y;
    if (bias) v = __fadd_rn(v, bias[r]);
    out[r] = mode == 1 ? __fadd_rn(out[r], v) : v;
}

// 0x4B400000 = 1.5 * 2^23: for |i| < 2^22, float(i) = bits(MAGIC + i) - 1.5 * 2^23,
// exactly (one integer add, one float subtract, instead of a slow I2F).
#define GM_MAGIC 0x4B400000
#define GM_MAGICF 12582912.0f

__device__ __forceinline__ float gm_i2f(int i) {
    return __fsub_rn(__int_as_float(GM_MAGIC + i), GM_MAGICF);
}

// ---------------------------------------------------------------------------
// gemvn: the decode GEMV over T tokens at once (CUDA cores, dp4a). Exactly
// gemv's lanes and ops, with a token loop inside each lane's block step, so
// a row's weights are read once for T tokens: the small-batch verify path
// (weight-bound like a decode step), and the batched products' fallback on
// GPUs without int8 tensor cores (sm < 80).

template <int R, int T>
__device__ void dotn_q4_k(const u8* const* rows, int nb, const u8* act, u64 astride, int n,
                          int zmax, float* acc) {
    int t = threadIdx.x, g = t >> 3, p = t & 7, c = p >> 1, hf = p & 1;
    // Software-pipelined: block b + 16's weights load while b's are used.
    uint4 h[R], q[R], hn[R], qn[R];
    auto load = [&](int b, uint4* hh, uint4* qq) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const u8* blk = rows[r] + (u64)b * 144;
            hh[r] = __ldg((const uint4*)blk);
            qq[r] = __ldg((const uint4*)(blk + 16 + 16 * p));
        }
    };
    if (g < nb) load(g, hn, qn);
    for (int b = g; b < nb; b += GEMV_THREADS / 8) {
#pragma unroll
        for (int r = 0; r < R; r++) h[r] = hn[r], q[r] = qn[r];
        if (b + GEMV_THREADS / 8 < nb) load(b + GEMV_THREADS / 8, hn, qn);
        // What depends on the weights alone, once for the T tokens.
        u32 wl[R][4], wh[R][4];
        int sc1[R], m1[R], sc2[R], m2[R];
        float d[R], dmin[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            wl[r][0] = q[r].x & 0x0f0f0f0f, wh[r][0] = (q[r].x >> 4) & 0x0f0f0f0f;
            wl[r][1] = q[r].y & 0x0f0f0f0f, wh[r][1] = (q[r].y >> 4) & 0x0f0f0f0f;
            wl[r][2] = q[r].z & 0x0f0f0f0f, wh[r][2] = (q[r].z >> 4) & 0x0f0f0f0f;
            wl[r][3] = q[r].w & 0x0f0f0f0f, wh[r][3] = (q[r].w >> 4) & 0x0f0f0f0f;
            scale_min_h(h[r], 2 * c, &sc1[r], &m1[r]);
            scale_min_h(h[r], 2 * c + 1, &sc2[r], &m2[r]);
            d[r] = f16_to_f32(h[r].x & 0xffff), dmin[r] = f16_to_f32(h[r].x >> 16);
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
                s1 = __dp4a((int)wl[r][0], xl.x, s1);
                s1 = __dp4a((int)wl[r][1], xl.y, s1);
                s1 = __dp4a((int)wl[r][2], xl.z, s1);
                s1 = __dp4a((int)wl[r][3], xl.w, s1);
                s2 = __dp4a((int)wh[r][0], xh.x, s2);
                s2 = __dp4a((int)wh[r][1], xh.y, s2);
                s2 = __dp4a((int)wh[r][2], xh.z, s2);
                s2 = __dp4a((int)wh[r][3], xh.w, s2);
                int sumi = sc1[r] * s1 + sc2[r] * s2;
                int summ = hf ? 0 : m1[r] * bs0 + m2[r] * bs1;
                // gm_i2f: exact (|sumi| < 2^22), = (float)sumi.
                float v = __fsub_rn(__fmul_rn(d[r], gm_i2f(sumi)), __fmul_rn(dmin[r], gm_i2f(summ)));
                acc[r * T + j] = __fmaf_rn(dk, v, acc[r * T + j]);
            }
        }
    }
}

template <int R, int T>
__device__ void dotn_q6_k(const u8* const* rows, int nb, const u8* act, u64 astride, int n,
                          int zmax, float* acc) {
    int t = threadIdx.x, g = t >> 3, p = t & 7, hh = p >> 2, l0 = (p & 3) * 8;
    uint2 la[R], lb[R], qh[R], lan[R], lbn[R], qhn[R];
    int scn[R];
    float dn[R];
    auto load = [&](int b) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const u8* blk = rows[r] + (u64)b * 210;
            lan[r] = ld8_a2(blk + 64 * hh + l0);
            lbn[r] = ld8_a2(blk + 64 * hh + 32 + l0);
            qhn[r] = ld8_a2(blk + 128 + 32 * hh + l0);
            // scales 8 hh + (l0 >> 4) + 2k, k = 0..3: bytes 0, 2, 4, 6 of 8.
            uint2 sc = ld8_a2(blk + 192 + 8 * hh);
            scn[r] = (l0 >> 4) ? (int)__byte_perm(sc.x, sc.y, 0x7531)
                               : (int)__byte_perm(sc.x, sc.y, 0x6420);
            dn[r] = rd_f16(blk + 208);
        }
    };
    if (g < nb) load(g);
    for (int b = g; b < nb; b += GEMV_THREADS / 8) {
        int scs[R];
        float ds[R];
#pragma unroll
        for (int r = 0; r < R; r++)
            la[r] = lan[r], lb[r] = lbn[r], qh[r] = qhn[r], scs[r] = scn[r], ds[r] = dn[r];
        if (b + GEMV_THREADS / 8 < nb) load(b + GEMV_THREADS / 8);
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
                int sumi = 0;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    uint2 l = (k & 1) ? lb[r] : la[r];
                    u32 lo0 = k < 2 ? (l.x & 0x0f0f0f0f) : ((l.x >> 4) & 0x0f0f0f0f);
                    u32 lo1 = k < 2 ? (l.y & 0x0f0f0f0f) : ((l.y >> 4) & 0x0f0f0f0f);
                    u32 w0 = __vsub4(lo0 | (((qh[r].x >> (2 * k)) & 0x03030303) << 4), 0x20202020);
                    u32 w1 = __vsub4(lo1 | (((qh[r].y >> (2 * k)) & 0x03030303) << 4), 0x20202020);
                    int s = __dp4a((int)w0, (int)x[k].x, 0);
                    s = __dp4a((int)w1, (int)x[k].y, s);
                    sumi += ((int)(scs[r] << (24 - 8 * k)) >> 24) * s;
                }
                acc[r * T + j] = __fmaf_rn(__fmul_rn(dk, ds[r]), (float)sumi, acc[r * T + j]);
            }
        }
    }
}

template <int R, int T>
__device__ void dotn_q8_0(const u8* const* rows, int nb, const u8* act, u64 astride, int n,
                          int zmax, float* acc) {
    int t = threadIdx.x, g = t >> 2, p = t & 3;
    u32 mask = 0xfu << (t & 28);
    uint2 q[R], qn[R];
    float dw[R], dwn[R];
    auto load = [&](int b) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            qn[r] = ld8_a2(rows[r] + (u64)b * 34 + 2 + 8 * p);
            dwn[r] = rd_f16(rows[r] + (u64)b * 34);
        }
    };
    if (g < nb) load(g);
    for (int b = g; b < nb; b += GEMV_THREADS / 4) {
#pragma unroll
        for (int r = 0; r < R; r++) q[r] = qn[r], dw[r] = dwn[r];
        if (b + GEMV_THREADS / 4 < nb) load(b + GEMV_THREADS / 4);
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
                    acc[r * T + j] = __fadd_rn(
                        acc[r * T + j],
                        __fmul_rn(__fmul_rn(dw[r], d0), gm_i2f(s)));
            }
        }
    }
}

template <int R, int T>
__device__ void dotn_q5_0(const u8* const* rows, int nb, const u8* act, u64 astride, int n,
                          int zmax, float* acc) {
    int t = threadIdx.x, g = t >> 2, p = t & 3;
    u32 mask = 0xfu << (t & 28);
    u32 qh[R], qhn[R];
    uint2 q[R], qn[R];
    float dw[R], dwn[R];
    auto load = [&](int b) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const u8* blk = rows[r] + (u64)b * 22;
            qhn[r] = ld4_a2(blk + 2) >> (8 * p);
            qn[r] = ld8_a2(blk + 6 + 8 * (p & 1));
            dwn[r] = rd_f16(blk);
        }
    };
    if (g < nb) load(g);
    for (int b = g; b < nb; b += GEMV_THREADS / 4) {
#pragma unroll
        for (int r = 0; r < R; r++) qh[r] = qhn[r], q[r] = qn[r], dw[r] = dwn[r];
        if (b + GEMV_THREADS / 4 < nb) load(b + GEMV_THREADS / 4);
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
                    acc[r * T + j] = __fadd_rn(
                        acc[r * T + j],
                        __fmul_rn(__fmul_rn(dw[r], d0), gm_i2f(s)));
            }
        }
    }
}

// cta_sum for N values per thread (same tree per value).
template <int N>
__device__ void cta_sum_n(float* acc) {
    __shared__ float sh[GEMV_THREADS / 32][N];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
#pragma unroll
    for (int r = 0; r < N; r++) {
        float v = acc[r];
        for (int o = 1; o < 32; o <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) sh[w][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < N) {
        int r = threadIdx.x;
        float v = sh[0][r];
        for (int i = 1; i < GEMV_THREADS / 32; i++) v = __fadd_rn(v, sh[i][r]);
        acc[0] = v;
    }
}

// Rows R * blockIdx.x .. (+R) for tokens T * blockIdx.y .. (+T, < nb):
// out[z * ostride + r] as gemv's. Thread i < R * T ends with the sum of row
// i / T, token i % T.
template <int R, int T>
__device__ void gemvn_t(int qt, const u8* w, u64 rb, int rows, int n, const u8* act, u64 astride,
                        int nb, const float* bias, int mode, float* out, u64 ostride) {
    int r0 = R * blockIdx.x, z0 = T * blockIdx.y;
    const u8* rr[R];
#pragma unroll
    for (int r = 0; r < R; r++) rr[r] = w + (u64)min(r0 + r, rows - 1) * rb;
    const u8* a = act + (u64)z0 * astride;
    int zmax = nb - 1 - z0;
    float acc[R * T];
#pragma unroll
    for (int i = 0; i < R * T; i++) acc[i] = 0.0f;
    switch (qt) {
    case QT_Q4_K: dotn_q4_k<R, T>(rr, n >> 8, a, astride, n, zmax, acc); break;
    case QT_Q6_K: dotn_q6_k<R, T>(rr, n >> 8, a, astride, n, zmax, acc); break;
    case QT_Q8_0: dotn_q8_0<R, T>(rr, n >> 5, a, astride, n, zmax, acc); break;
    default: dotn_q5_0<R, T>(rr, n >> 5, a, astride, n, zmax, acc); break;
    }
    cta_sum_n<R * T>(acc);
    int i = threadIdx.x;
    if (i < R * T) {
        int r = r0 + i / T, z = z0 + i % T;
        if (r < rows && z < nb) gm_store(acc[0], bias, mode, out + (u64)z * ostride, r);
    }
}

#define GEMVN(NAME, R, T)                                                                         \
    extern "C" __global__ void __launch_bounds__(GEMV_THREADS) NAME(                              \
        int qt, const u8* __restrict__ w, u64 rb, int rows, int n, const u8* __restrict__ act,    \
        u64 astride, int nb, const float* __restrict__ bias, int mode, float* __restrict__ out,  \
        u64 ostride) {                                                                            \
        gemvn_t<R, T>(qt, w, rb, rows, n, act, astride, nb, bias, mode, out, ostride);            \
    }
GEMVN(gemvn_2x8, 2, 8)
GEMVN(gemvn_2x16, 2, 16)

// ---------------------------------------------------------------------------
// gemm_*: the batched products on int8 tensor cores (sm_80+, mma.sync).
//
// A block computes GM_BM rows x GM_BN tokens over all of K: 8 warps, 2 along
// rows x 4 along tokens, each 16 rows x 16 tokens (two m16n8 tiles). An mma
// gives the exact integer sum over one GEMV lane's share of one sub-block:
// k16 tiles for K-quant lanes (16 values of a 32-value Q4_K sub-block; 8 of
// a 16-value Q6_K one, with the other 8 in the tile's second row half: A
// rows g / g + 8 are one matrix row's first / second half-sub-block, zero
// elsewhere), k32 for Q8_0/Q5_0 blocks. The float epilogue is the GEMV
// lane's ops on that sum, into that lane's running sum; when a group's last
// block is done its lanes fold into the cta_sum tree (see the top).
//
// A stage is one block of 256 values (K-quants) or 8 blocks of 32, in the
// lanes' order (g-major, a table in shared memory); two stages are in shared
// memory, one computed while the next one's copies are in flight. Q4_K
// weights and all activations are copied as they are (cp.async); Q4_K's
// nibbles are unpacked when the fragments are built. The other types'
// blocks (2-byte aligned) go through registers and are unpacked to int8.

#define GM_BM 32
#define GM_THREADS 256
// Most blocks per row the sequence table holds (K <= 32768 for Q8_0/Q5_0).
#define GM_MAXSEQ 1024

template <int BN>
struct GmStage {
    // Q4_K: the raw 144-byte blocks (9 chunks); others: int8 values [row][256].
    // Rows padded to 272 bytes, so a warp's fragment loads (rows g = 0..7,
    // words t = 0..3: banks 4g + t + 4j) hit 32 different banks.
    uint4 w[GM_BM][17];
    uint4 x[BN][17];
    // Per row: Q6_K d; Q8_0/Q5_0 d of each of the 8 blocks.
    float wd[GM_BM][8];
    // Q6_K: the 16 int8 scales.
    u32 wsc[GM_BM][4];
    // Per token: Q8_K d (xd[0]) and the 8 sums of 32 (xb); Q8_0 d of each block.
    float xd[BN][8];
    int xb[BN][8];
};

__device__ __forceinline__ uint4* gm_chunk(uint4 (*a)[17], int r, int j) { return &a[r][j]; }
__device__ __forceinline__ u32 gm_word(const uint4 (*a)[17], int r, int j, int t) {
    return ((const u32*)&a[r][j])[t];
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    u32 s = (u32)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    u32 s = (u32)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void mma16(int* d, u32 a0, u32 a1, u32 b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
                 "{%7,%7,%7,%7};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}
__device__ __forceinline__ void mma32(int* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}
#endif

// N words of the bytes at a 2-byte aligned p.
template <int N>
__device__ __forceinline__ void ld_a2(const u8* p, u32* out) {
    if (((size_t)p & 3) == 0) {
#pragma unroll
        for (int i = 0; i < N; i++) out[i] = __ldg((const u32*)p + i);
    } else {
        const u32* w = (const u32*)(p - 2);
        u32 prev = __ldg(w);
#pragma unroll
        for (int i = 0; i < N; i++) {
            u32 nx = __ldg(w + i + 1);
            out[i] = __funnelshift_r(prev, nx, 16);
            prev = nx;
        }
    }
}

// Step q of the lanes' sequence over nblk blocks in G groups: group g holds
// blocks g, g + G, ... (nblk / G of them, or one more); groups in order.
// Packed: block | group << 16 | (last of its group) << 31.
__device__ __forceinline__ u32 gm_seq(int q, int nblk, int G) {
    int lmin = nblk / G, rem = nblk % G, g, j, len;
    if (q < rem * (lmin + 1)) {
        g = q / (lmin + 1);
        j = q % (lmin + 1);
        len = lmin + 1;
    } else {
        int q2 = q - rem * (lmin + 1);
        g = rem + q2 / lmin;
        j = q2 % lmin;
        len = lmin;
    }
    return (u32)(g + G * j) | ((u32)g << 16) | ((u32)(j == len - 1) << 31);
}
__device__ __forceinline__ int sq_blk(u32 e) { return e & 0xffff; }
__device__ __forceinline__ int sq_grp(u32 e) { return (e >> 16) & 0x7fff; }
__device__ __forceinline__ bool sq_last(u32 e) { return e >> 31; }

// One output's cta_sum tree so far: groups fold in order. K-quant groups
// are 8 lanes (4 groups per warp), 32-value ones 4 lanes (8 per warp).
struct GmTree {
    float r, s0, s1, s2;
};

// Fold group g's value (its lanes' subtree) into the tree: 4 (GPW = 4) or 8
// leaves per warp, a perfect binary tree each, then the warps in order.
template <int GPW>
__device__ __forceinline__ void gm_fold(GmTree& t, int g, float v) {
    int i = g % GPW, wq = g / GPW;
    float w = 0.0f;
    bool done = false;
    if (GPW == 4) {
        if (i == 0) t.s0 = v;
        else if (i == 1) t.s0 = __fadd_rn(t.s0, v);
        else if (i == 2) t.s1 = v;
        else {
            w = __fadd_rn(t.s0, __fadd_rn(t.s1, v));
            done = true;
        }
    } else {
        if (i == 0) t.s0 = v;
        else if (i == 1) t.s0 = __fadd_rn(t.s0, v);
        else if (i == 2) t.s1 = v;
        else if (i == 3) t.s0 = __fadd_rn(t.s0, __fadd_rn(t.s1, v));
        else if (i == 4) t.s1 = v;
        else if (i == 5) t.s1 = __fadd_rn(t.s1, v);
        else if (i == 6) t.s2 = v;
        else {
            w = __fadd_rn(t.s0, __fadd_rn(t.s1, __fadd_rn(t.s2, v)));
            done = true;
        }
    }
    if (done) t.r = wq == 0 ? w : __fadd_rn(t.r, w);
}

// The 8 lanes of a K-quant group: ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7)).
__device__ __forceinline__ float gm_tree8(const float* c) {
    return __fadd_rn(__fadd_rn(__fadd_rn(c[0], c[1]), __fadd_rn(c[2], c[3])),
                     __fadd_rn(__fadd_rn(c[4], c[5]), __fadd_rn(c[6], c[7])));
}
// The 4 lanes of a 32-value group, only the first with a sum: (c + 0) + (0 + 0).
__device__ __forceinline__ float gm_tree4(float c) {
    return __fadd_rn(__fadd_rn(c, 0.0f), __fadd_rn(0.0f, 0.0f));
}

// NF n8 tiles per warp: GM_BN tokens per block; per thread outputs o =
// 4f + i of tile f, mma accumulator element i: row g (i < 2) or g + 8,
// token 2t + (i & 1).
#define GM_BN(NF) (32 * (NF))

template <int QT>
struct GmType;
template <>
struct GmType<QT_Q4_K> {
    // G groups, SB blocks per stage of VB values, register-staged items per row.
    static const int G = 16, SB = 1, VB = 256, ITEMS = 0, BYTES = 144;
};
template <>
struct GmType<QT_Q6_K> {
    // 4 x (2 x 16 bytes of ql + 16 of qh), 1 header.
    static const int G = 16, SB = 1, VB = 256, ITEMS = 5, BYTES = 210;
};
template <>
struct GmType<QT_Q8_0> {
    static const int G = 32, SB = 8, VB = 32, ITEMS = 8, BYTES = 34;
};
template <>
struct GmType<QT_Q5_0> {
    static const int G = 32, SB = 8, VB = 32, ITEMS = 8, BYTES = 22;
};

// Register-staged weight words per item.
#define GM_WREGS 13

// Into registers: stage s's weights for the register-staged types.
template <int QT>
__device__ __forceinline__ void gm_load_w(const u8* w, u64 rb, int rows, int r0, int nblk, int s,
                                          const u32* seq, u32 (*reg)[GM_WREGS]) {
    const int ITEMS = GmType<QT>::ITEMS > 0 ? GmType<QT>::ITEMS : 1, SB = GmType<QT>::SB;
    const int NI = (GM_BM * GmType<QT>::ITEMS + GM_THREADS - 1) / GM_THREADS;
#pragma unroll
    for (int k = 0; k < NI; k++) {
        int id = threadIdx.x + k * GM_THREADS;
        if (id >= GM_BM * ITEMS) break;
        int rr = id / ITEMS, it = id % ITEMS;
        const u8* row = w + (u64)min(r0 + rr, rows - 1) * rb;
        u32* o = reg[k];
        if (QT == QT_Q6_K) {
            const u8* blk = row + (u64)sq_blk(seq[s]) * 210;
            if (it < 4) {
                int h = it >> 1, l0 = 16 * (it & 1);
                ld_a2<4>(blk + 64 * h + l0, o);
                ld_a2<4>(blk + 64 * h + 32 + l0, o + 4);
                ld_a2<4>(blk + 128 + 32 * h + l0, o + 8);
            } else {
                ld_a2<4>(blk + 192, o);
                o[4] = rd_u16(blk + 208);
            }
        } else {
            int q = s * SB + it;
            if (q >= nblk) continue;
            const u8* blk = row + (u64)sq_blk(seq[q]) * GmType<QT>::BYTES;
            o[0] = rd_u16(blk);
            if (QT == QT_Q8_0) ld_a2<8>(blk + 2, o + 1);
            else ld_a2<5>(blk + 2, o + 1); // qh, qs
        }
    }
}

// The registers of gm_load_w, unpacked into stage st.
template <int QT, class ST>
__device__ __forceinline__ void gm_store_w(ST* st, int nblk, int s, u32 (*reg)[GM_WREGS]) {
    const int ITEMS = GmType<QT>::ITEMS > 0 ? GmType<QT>::ITEMS : 1, SB = GmType<QT>::SB;
    const int NI = (GM_BM * GmType<QT>::ITEMS + GM_THREADS - 1) / GM_THREADS;
#pragma unroll
    for (int k = 0; k < NI; k++) {
        int id = threadIdx.x + k * GM_THREADS;
        if (id >= GM_BM * ITEMS) break;
        int rr = id / ITEMS, it = id % ITEMS;
        const u32* o = reg[k];
        if (QT == QT_Q6_K) {
            if (it < 4) {
                // Values 128 h + 32 k + l0 .. + 16: chunk 8 h + 2 k + (l0 / 16).
                int h = it >> 1, lh = it & 1;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    u32 v[4];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        u32 l = o[(k & 1) * 4 + i];
                        u32 lo = k < 2 ? (l & 0x0f0f0f0f) : ((l >> 4) & 0x0f0f0f0f);
                        v[i] = __vsub4(lo | (((o[8 + i] >> (2 * k)) & 0x03030303) << 4),
                                       0x20202020);
                    }
                    *gm_chunk(st->w, rr, 8 * h + 2 * k + lh) = make_uint4(v[0], v[1], v[2], v[3]);
                }
            } else {
                st->wsc[rr][0] = o[0], st->wsc[rr][1] = o[1];
                st->wsc[rr][2] = o[2], st->wsc[rr][3] = o[3];
                st->wd[rr][0] = h2f(o[4]);
            }
        } else {
            if (s * SB + it >= nblk) continue;
            st->wd[rr][it] = h2f(o[0]);
            if (QT == QT_Q8_0) {
                *gm_chunk(st->w, rr, 2 * it) = make_uint4(o[1], o[2], o[3], o[4]);
                *gm_chunk(st->w, rr, 2 * it + 1) = make_uint4(o[5], o[6], o[7], o[8]);
            } else {
                // o[1] = qh; qs = o[2..6): element i < 16 low nibble of qs[i],
                // i >= 16 high nibble of qs[i - 16]; bit 4 = qh bit i.
                u32 qh = o[1], v[8];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    v[u] = __vsub4((o[2 + u] & 0x0f0f0f0f) | spread4((qh >> (4 * u)) & 0xf),
                                   0x10101010);
                    v[4 + u] = __vsub4(((o[2 + u] >> 4) & 0x0f0f0f0f) |
                                           spread4((qh >> (16 + 4 * u)) & 0xf),
                                       0x10101010);
                }
                *gm_chunk(st->w, rr, 2 * it) = make_uint4(v[0], v[1], v[2], v[3]);
                *gm_chunk(st->w, rr, 2 * it + 1) = make_uint4(v[4], v[5], v[6], v[7]);
            }
        }
    }
}

// Start the copies of stage s: Q4_K weights, and the activations of tokens
// z0.. (clamped to nb - 1).
template <int QT, int NF>
__device__ __forceinline__ void gm_copy(GmStage<GM_BN(NF)>* st, const u8* w, u64 rb, int rows, int r0,
                                        const u8* act, u64 astride, int n, int nb, int z0,
                                        int nblk, int s, const u32* seq) {
#if __CUDA_ARCH__ >= 800
    u64 np = act_np(n);
    int tid = threadIdx.x;
    if (QT == QT_Q4_K || QT == QT_Q6_K) {
        u64 b = sq_blk(seq[s]);
        if (QT == QT_Q4_K) {
            for (int id = tid; id < GM_BM * 9; id += GM_THREADS) {
                int rr = id / 9, ch = id - 9 * rr;
                cp16(gm_chunk(st->w, rr, ch),
                     w + (u64)min(r0 + rr, rows - 1) * rb + b * 144 + 16 * ch);
            }
        }
        int c = tid & 15;
#pragma unroll
        for (int k = 0; k < GM_BN(NF) * 16 / GM_THREADS; k++) {
            int tk = (tid >> 4) + k * (GM_THREADS / 16);
            const u8* a = act + (u64)min(z0 + tk, nb - 1) * astride;
            cp16(gm_chunk(st->x, tk, c), a + b * 256 + 16 * c);
        }
        if (QT == QT_Q4_K) {
            int j = tid & 7;
#pragma unroll
            for (int k = 0; k < GM_BN(NF) * 8 / GM_THREADS; k++) {
                int tk = (tid >> 3) + k * (GM_THREADS / 8);
                const u8* a = act + (u64)min(z0 + tk, nb - 1) * astride;
                cp4(&st->xb[tk][j], a + 2 * np + np / 64 + 4 * (b * 8 + j));
            }
        }
        if (tid < GM_BN(NF)) {
            const u8* a = act + (u64)min(z0 + tid, nb - 1) * astride;
            cp4(&st->xd[tid][0], a + 2 * np + 4 * b);
        }
    } else {
        int e = (tid >> 1) & 7, h = tid & 1, q = s * 8 + e;
        if (q < nblk) {
            u64 b = sq_blk(seq[q]);
#pragma unroll
            for (int k = 0; k < GM_BN(NF) * 16 / GM_THREADS; k++) {
                int tk = (tid >> 4) + k * (GM_THREADS / 16);
                const u8* a = act + (u64)min(z0 + tk, nb - 1) * astride;
                cp16(gm_chunk(st->x, tk, 2 * e + h), a + np + 32 * b + 16 * h);
            }
        }
        e = tid & 7, q = s * 8 + e;
        if (q < nblk) {
            u64 b = sq_blk(seq[q]);
#pragma unroll
            for (int k = 0; k < GM_BN(NF) * 8 / GM_THREADS; k++) {
                int tk = (tid >> 3) + k * (GM_THREADS / 8);
                const u8* a = act + (u64)min(z0 + tk, nb - 1) * astride;
                cp4(&st->xd[tk][e], a + 2 * np + np / 64 + np / 8 + 4 * b);
            }
        }
    }
#endif
}

// Signed byte j of the 16 packed in w[0..4).
__device__ __forceinline__ int gm_sbyte(const u32* w, int j) {
    return (int)(w[j >> 2] << (24 - 8 * (j & 3))) >> 24;
}
__device__ __forceinline__ int gm_ubyte(u32 w, int j) { return __byte_perm(w, 0, 0x4440 + (j & 3)); }

// One stage's products into the lanes' running sums c[lane][output] (and,
// for 32-value blocks, groups folded as they end). wr / wc: the warp's row /
// token tile.
// CH: groups of several blocks (lanes keep running sums c across stages);
// without, every group is one block (nblk <= G) and its 8 lane values are
// added up as they come (same tree, gm_tree8's), folded at once.
template <int QT, int NF, bool CH>
__device__ __forceinline__ void gm_compute(const GmStage<GM_BN(NF)>* st, int nblk, int s,
                                           const u32* seq, int wr, int wc, float (*c)[4 * NF],
                                           GmTree* tree) {
    const int NO = 4 * NF;
    float s01[NO], s45[NO], s67[NO]; // !CH: partial gm_tree8 sums
#if __CUDA_ARCH__ >= 800
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int ra = 16 * wr + g, rbw = ra + 8; // the thread's rows (outputs i < 2, i >= 2)
    int tb[NF]; // B fragment tokens of the n8 tiles
#pragma unroll
    for (int f = 0; f < NF; f++) tb[f] = 8 * NF * wc + 8 * f + g;
    int tk[NO]; // output tokens
#pragma unroll
    for (int o = 0; o < NO; o++) tk[o] = 8 * NF * wc + 8 * (o >> 2) + 2 * t + (o & 1);
    if constexpr (QT == QT_Q4_K) {
        // The rows' headers (chunk 0): d, dmin, 6-bit scales and mins (quant.rs
        // get_scale_min_k4, four at a time).
        float d[2], dmin[2], dm0[2];
        u32 scw[2][2], mw[2][2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            uint4 hd = st->w[h ? rbw : ra][0];
            // The hardware conversion: as exact as f16_to_f32 in the
            // products it feeds (gpu_hw_f16_scales_exact_in_products).
            d[h] = h2f(hd.x & 0xffff);
            dmin[h] = h2f(hd.x >> 16);
            dm0[h] = __fmul_rn(dmin[h], 0.0f);
            scw[h][0] = hd.y & 0x3f3f3f3f;
            scw[h][1] = (hd.w & 0x0f0f0f0f) | ((hd.y >> 2) & 0x30303030);
            mw[h][0] = hd.z & 0x3f3f3f3f;
            mw[h][1] = ((hd.w >> 4) & 0x0f0f0f0f) | ((hd.z >> 2) & 0x30303030);
        }
        float dk[NO];
#pragma unroll
        for (int o = 0; o < NO; o++) dk[o] = st->xd[tk[o]][0];
#pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            int sc1[2], sc2[2], m1[2], m2[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                sc1[h] = gm_ubyte(scw[h][cc >> 1], 2 * cc);
                sc2[h] = gm_ubyte(scw[h][cc >> 1], 2 * cc + 1);
                m1[h] = gm_ubyte(mw[h][cc >> 1], 2 * cc);
                m2[h] = gm_ubyte(mw[h][cc >> 1], 2 * cc + 1);
            }
            float dm[NO]; // dmin * summ of sub-blocks 2c, 2c + 1 (lanes with hf = 0)
#pragma unroll
            for (int o = 0; o < NO; o++) {
                int h = (o & 2) >> 1;
                int2 bs = *(const int2*)&st->xb[tk[o]][2 * cc];
                dm[o] = __fmul_rn(dmin[h], gm_i2f(m1[h] * bs.x + m2[h] * bs.y));
            }
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                // qs bytes 32 c + 16 hf + 4 t.. (chunk 1 + 2c + hf): low nibbles
                // are sub-block 2c's values 16 hf + 4 t.., high ones 2c + 1's.
                int p = 2 * cc + hf, k = 1 + 2 * cc + hf;
                u32 qa = gm_word(st->w, ra, k, t), qb = gm_word(st->w, rbw, k, t);
                u32 a10 = qa & 0x0f0f0f0f, a11 = qb & 0x0f0f0f0f;
                u32 a20 = (qa >> 4) & 0x0f0f0f0f, a21 = (qb >> 4) & 0x0f0f0f0f;
                int j1 = 4 * cc + hf, j2 = j1 + 2; // activation chunks
#pragma unroll
                for (int f = 0; f < NF; f++) {
                    int d1[4], d2[4];
                    mma16(d1, a10, a11, gm_word(st->x, tb[f], j1, t));
                    mma16(d2, a20, a21, gm_word(st->x, tb[f], j2, t));
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int o = 4 * f + i, h = i >> 1;
                        float si = gm_i2f(sc1[h] * d1[i] + sc2[h] * d2[i]);
                        float v = __fsub_rn(__fmul_rn(d[h], si), hf ? dm0[h] : dm[o]);
                        if constexpr (CH) {
                            c[p][o] = __fmaf_rn(dk[o], v, c[p][o]);
                        } else {
                            // ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7)), in order.
                            float cp = __fmaf_rn(dk[o], v, 0.0f);
                            if (p == 0) s01[o] = cp;
                            else if (p == 1) s01[o] = __fadd_rn(s01[o], cp);
                            else if (p == 2) s45[o] = cp; // c2 for now
                            else if (p == 3) s01[o] = __fadd_rn(s01[o], __fadd_rn(s45[o], cp));
                            else if (p == 4) s45[o] = cp;
                            else if (p == 5) s45[o] = __fadd_rn(s45[o], cp);
                            else if (p == 6) s67[o] = cp;
                            else c[0][o] = __fadd_rn(s01[o], __fadd_rn(s45[o], __fadd_rn(s67[o], cp)));
                        }
                    }
                }
            }
        }
    } else if constexpr (QT == QT_Q6_K) {
        float d[2] = {st->wd[ra][0], st->wd[rbw][0]};
        float dkd[NO];
#pragma unroll
        for (int o = 0; o < NO; o++) dkd[o] = __fmul_rn(st->xd[tk[o]][0], d[(o & 2) >> 1]);
#pragma unroll
        for (int pp = 0; pp < 4; pp++) {
            int hh = pp >> 1, qhi = pp & 1, px = 4 * hh + 2 * qhi;
            int sx[NO], sy[NO];
#pragma unroll
            for (int o = 0; o < NO; o++) sx[o] = sy[o] = 0;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                int j = 8 * hh + 2 * k + qhi;
                int sca = gm_sbyte(st->wsc[ra], j), scb = gm_sbyte(st->wsc[rbw], j);
                u32 wa = gm_word(st->w, ra, j, t), wb = gm_word(st->w, rbw, j, t);
                u32 xa0 = t < 2 ? wa : 0, xa1 = t < 2 ? 0 : wa;
                u32 xb0 = t < 2 ? wb : 0, xb1 = t < 2 ? 0 : wb;
#pragma unroll
                for (int f = 0; f < NF; f++) {
                    u32 bb = gm_word(st->x, tb[f], j, t);
                    int da[4], db[4];
                    mma16(da, xa0, xa1, bb);
                    mma16(db, xb0, xb1, bb);
                    sx[4 * f + 0] += sca * da[0];
                    sx[4 * f + 1] += sca * da[1];
                    sy[4 * f + 0] += sca * da[2];
                    sy[4 * f + 1] += sca * da[3];
                    sx[4 * f + 2] += scb * db[0];
                    sx[4 * f + 3] += scb * db[1];
                    sy[4 * f + 2] += scb * db[2];
                    sy[4 * f + 3] += scb * db[3];
                }
            }
#pragma unroll
            for (int o = 0; o < NO; o++) {
                if constexpr (CH) {
                    c[px][o] = __fmaf_rn(dkd[o], (float)sx[o], c[px][o]);
                    c[px + 1][o] = __fmaf_rn(dkd[o], (float)sy[o], c[px + 1][o]);
                } else {
                    float cx = __fmaf_rn(dkd[o], (float)sx[o], 0.0f);
                    float cy = __fmaf_rn(dkd[o], (float)sy[o], 0.0f);
                    float pr = __fadd_rn(cx, cy);
                    if (pp == 0) s01[o] = pr;
                    else if (pp == 1) s01[o] = __fadd_rn(s01[o], pr);
                    else if (pp == 2) s45[o] = pr;
                    else c[0][o] = __fadd_rn(s01[o], __fadd_rn(s45[o], pr));
                }
            }
        }
    } else {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int q = s * 8 + e;
            if (q >= nblk) break;
            u32 sq = seq[q];
            u32 a0 = gm_word(st->w, ra, 2 * e, t), a1 = gm_word(st->w, rbw, 2 * e, t);
            u32 a2 = gm_word(st->w, ra, 2 * e + 1, t), a3 = gm_word(st->w, rbw, 2 * e + 1, t);
            float wd[2] = {st->wd[ra][e], st->wd[rbw][e]};
#pragma unroll
            for (int f = 0; f < NF; f++) {
                int dd[4];
                mma32(dd, a0, a1, a2, a3, gm_word(st->x, tb[f], 2 * e, t),
                      gm_word(st->x, tb[f], 2 * e + 1, t));
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    int o = 4 * f + i;
                    float sd = __fmul_rn(wd[i >> 1], st->xd[tk[o]][e]);
                    c[0][o] = __fadd_rn(c[0][o], __fmul_rn(sd, gm_i2f(dd[i])));
                }
            }
            if (sq_last(sq)) {
#pragma unroll
                for (int o = 0; o < NO; o++) {
                    gm_fold<8>(tree[o], sq_grp(sq), gm_tree4(c[0][o]));
                    c[0][o] = 0.0f;
                }
            }
        }
    }
#endif
}

template <int QT, int NF, bool CH>
__device__ __forceinline__ void gemm_t(const u8* w, u64 rb, int rows, int n, const u8* act,
                                       u64 astride, int nb, const float* bias, int mode,
                                       float* out, u64 ostride) {
#if __CUDA_ARCH__ >= 800
    const int G = GmType<QT>::G, SB = GmType<QT>::SB, KQ = G == 16;
    const int NI = (GM_BM * GmType<QT>::ITEMS + GM_THREADS - 1) / GM_THREADS;
    extern __shared__ uint4 gm_smem[];
    typedef GmStage<GM_BN(NF)> St;
    const int NO = 4 * NF;
    St* st = (St*)gm_smem;
    u32* seq = (u32*)(st + 2);
    int nblk = n / GmType<QT>::VB, nsteps = (nblk + SB - 1) / SB;
    u32 dsz;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    // The host's GEMM_SMEM out of date, or a matrix too wide for the table.
    if (dsz < 2 * sizeof(St) + 4 * GM_MAXSEQ || nblk > GM_MAXSEQ) __trap();
    for (int q = threadIdx.x; q < nblk; q += GM_THREADS) seq[q] = gm_seq(q, nblk, G);
    __syncthreads();
    int r0 = blockIdx.y * GM_BM, z0 = blockIdx.x * GM_BN(NF);
    int warp = threadIdx.x >> 5, wr = warp & 1, wc = warp >> 1;
    u32 reg[NI > 0 ? NI : 1][GM_WREGS];
    float c[KQ && CH ? 8 : 1][NO];
    GmTree tree[NO];
#pragma unroll
    for (int o = 0; o < NO; o++) {
        tree[o].r = tree[o].s0 = tree[o].s1 = tree[o].s2 = 0.0f;
#pragma unroll
        for (int p = 0; p < (KQ && CH ? 8 : 1); p++) c[p][o] = 0.0f;
    }
    if (KQ && !CH && nblk > G) __trap(); // the host picks CH for wider rows
    if (NI > 0) gm_load_w<QT>(w, rb, rows, r0, nblk, 0, seq, reg);
    gm_copy<QT, NF>(&st[0], w, rb, rows, r0, act, astride, n, nb, z0, nblk, 0, seq);
    cp_commit();
    if (NI > 0) gm_store_w<QT>(&st[0], nblk, 0, reg);
    for (int s = 0; s < nsteps; s++) {
        bool more = s + 1 < nsteps;
        if (more) {
            if (NI > 0) gm_load_w<QT>(w, rb, rows, r0, nblk, s + 1, seq, reg);
            gm_copy<QT, NF>(&st[(s + 1) & 1], w, rb, rows, r0, act, astride, n, nb, z0, nblk, s + 1,
                        seq);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        gm_compute<QT, NF, CH>(&st[s & 1], nblk, s, seq, wr, wc, c, tree);
        if (KQ && !CH) {
#pragma unroll
            for (int o = 0; o < NO; o++) gm_fold<4>(tree[o], s, c[0][o]);
        } else if (KQ && sq_last(seq[s])) {
            int gi = sq_grp(seq[s]);
#pragma unroll
            for (int o = 0; o < NO; o++) {
                float cs[8];
#pragma unroll
                for (int p = 0; p < 8; p++) {
                    cs[p] = c[p][o];
                    c[p][o] = 0.0f;
                }
                gm_fold<4>(tree[o], gi, gm_tree8(cs));
            }
        }
        if (more && NI > 0) gm_store_w<QT>(&st[(s + 1) & 1], nblk, s + 1, reg);
        __syncthreads();
    }
    // Groups without blocks (nblk < G): lanes that summed nothing.
    for (int gi = nblk; gi < G; gi++) {
#pragma unroll
        for (int o = 0; o < NO; o++) {
            if (KQ) gm_fold<4>(tree[o], gi, 0.0f); // gm_tree8 of 8 zeros
            else gm_fold<8>(tree[o], gi, gm_tree4(0.0f));
        }
    }
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int o = 0; o < NO; o++) {
        int r = r0 + 16 * wr + g + ((o & 2) ? 8 : 0);
        int z = z0 + 8 * NF * wc + 8 * (o >> 2) + 2 * t + (o & 1);
        if (r < rows && z < nb) gm_store(tree[o].r, bias, mode, out + (u64)z * ostride, r);
    }
#else
    __trap();
#endif
}

#define GEMM(NAME, QT, NF, CH)                                                                    \
    extern "C" __global__ void __launch_bounds__(GM_THREADS, NF == 1 ? 2 : 1) NAME(               \
        const u8* __restrict__ w, u64 rb, int rows, int n, const u8* __restrict__ act,           \
        u64 astride, int nb, const float* __restrict__ bias, int mode, float* __restrict__ out,  \
        u64 ostride) {                                                                            \
        gemm_t<QT, NF, CH>(w, rb, rows, n, act, astride, nb, bias, mode, out, ostride);           \
    }
// gemm_*: 64 tokens per block; gemm1_*: 32 (two blocks per SM); gemms1_*:
// K-quant rows of at most 16 blocks (no running sums kept), 32 tokens. The
// variants qwen2/batch.rs picks (the fastest measured, cuda_gemm_bench).
GEMM(gemm_q4_k, QT_Q4_K, 2, true)
GEMM(gemm1_q6_k, QT_Q6_K, 1, true)
GEMM(gemm1_q8_0, QT_Q8_0, 1, true)
GEMM(gemm1_q5_0, QT_Q5_0, 1, true)
GEMM(gemms1_q4_k, QT_Q4_K, 1, false)
GEMM(gemms1_q6_k, QT_Q6_K, 1, false)

// ---------------------------------------------------------------------------
// gemvm_*: small batches (nb <= 8 NF) on tensor cores, weight-read-once: the
// verify pass of a draft. A block is the decode GEMV's block remapped: warp
// w does what decode warp w's lanes do (groups GPW w .. GPW w + GPW - 1 of
// every row) for 16 rows x 8 NF tokens, with mma sums per lane share as in
// gemm_*, weights loaded straight from global memory (the next block's while
// one is used); then cta_sum's last step, ((W0 + W1) + W2) + W3, through
// shared memory. 4 warps, rows / 16 blocks.

#define GV_THREADS 128

// A word of the bytes at a 2-byte aligned p.
__device__ __forceinline__ u32 ldw_a2(const u8* p) {
    if (((size_t)p & 3) == 0) return __ldg((const u32*)p);
    const u32* w = (const u32*)(p - 2);
    return __funnelshift_r(__ldg(w), __ldg(w + 1), 16);
}

// The weights a thread needs of one block of rows ra (r = 0) and rb (r = 1).
template <int QT>
struct GvW;
template <>
struct GvW<QT_Q4_K> {
    uint4 h[2];
    u32 q[2][8]; // qs word (32 c + 16 hf + 4 t) / 4, piece p = 2c + hf
    __device__ __forceinline__ void load(const u8* const* rows, int b, int t) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const u8* blk = rows[r] + (u64)b * 144;
            h[r] = __ldg((const uint4*)blk);
#pragma unroll
            for (int p = 0; p < 8; p++) q[r][p] = __ldg((const u32*)(blk + 16 + 16 * p) + t);
        }
    }
};
template <>
struct GvW<QT_Q6_K> {
    // (h, lh): ql words of quarter parity 0 / 1, qh word; scales; d.
    u32 ql[2][4][2], qh[2][4], sc[2][4];
    float d[2];
    __device__ __forceinline__ void load(const u8* const* rows, int b, int t) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const u8* blk = rows[r] + (u64)b * 210;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int h = i >> 1, lh = i & 1, l = 16 * lh + 4 * t;
                ql[r][i][0] = ldw_a2(blk + 64 * h + l);
                ql[r][i][1] = ldw_a2(blk + 64 * h + 32 + l);
                qh[r][i] = ldw_a2(blk + 128 + 32 * h + l);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) sc[r][i] = ldw_a2(blk + 192 + 4 * i);
            d[r] = h2f(rd_u16(blk + 208));
        }
    }
};
template <int QT>
struct GvW32 {
    // Q8_0 / Q5_0: words 4t.. and 16 + 4t.. of the int8 values; d.
    u32 a[2][2];
    float d[2];
    __device__ __forceinline__ void load(const u8* const* rows, int b, int t) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            if constexpr (QT == QT_Q8_0) {
                const u8* blk = rows[r] + (u64)b * 34;
                a[r][0] = ldw_a2(blk + 2 + 4 * t);
                a[r][1] = ldw_a2(blk + 18 + 4 * t);
                d[r] = h2f(rd_u16(blk));
            } else {
                const u8* blk = rows[r] + (u64)b * 22;
                u32 qh = ldw_a2(blk + 2), qs = ldw_a2(blk + 6 + 4 * t);
                // Values 4t.. (low nibbles) and 16 + 4t.. (high), bit 4 from qh.
                a[r][0] = __vsub4((qs & 0x0f0f0f0f) | spread4((qh >> (4 * t)) & 0xf), 0x10101010);
                a[r][1] = __vsub4(((qs >> 4) & 0x0f0f0f0f) | spread4((qh >> (16 + 4 * t)) & 0xf),
                                  0x10101010);
                d[r] = h2f(rd_u16(blk));
            }
        }
    }
};
template <>
struct GvW<QT_Q8_0> : GvW32<QT_Q8_0> {};
template <>
struct GvW<QT_Q5_0> : GvW32<QT_Q5_0> {};

template <int QT, int NF>
__device__ __forceinline__ void gemvm_t(const u8* w, u64 rb, int rows, int n, const u8* act,
                                        u64 astride, int nb, const float* bias, int mode,
                                        float* out, u64 ostride) {
#if __CUDA_ARCH__ >= 800
    const int KQ = QT == QT_Q4_K || QT == QT_Q6_K;
    const int G = KQ ? 16 : 32, GPW = G / 4, NO = 4 * NF;
    __shared__ float sh[4][16][16];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    int r0 = blockIdx.x * 16;
    const u8* rws[2] = {w + (u64)min(r0 + g, rows - 1) * rb, w + (u64)min(r0 + g + 8, rows - 1) * rb};
    u64 np = act_np(n);
    int nblk = n / (KQ ? 256 : 32);
    // B fragment tokens (n = g of each tile) and output tokens (2t + e).
    const u8* xa[NF];
    const u8* xo[NO];
#pragma unroll
    for (int f = 0; f < NF; f++) xa[f] = act + (u64)min(8 * f + g, nb - 1) * astride;
#pragma unroll
    for (int o = 0; o < NO; o++)
        xo[o] = act + (u64)min(8 * (o >> 2) + 2 * t + (o & 1), nb - 1) * astride;
    float c[KQ ? 8 : 1][NO];
    GmTree tree[NO];
#pragma unroll
    for (int o = 0; o < NO; o++) {
        tree[o].r = tree[o].s0 = tree[o].s1 = tree[o].s2 = 0.0f;
#pragma unroll
        for (int p = 0; p < (KQ ? 8 : 1); p++) c[p][o] = 0.0f;
    }
    // This warp's blocks in the lanes' order: groups GPW warp.., each b = g + G j.
    int gi = GPW * warp, b = gi;
    GvW<QT> cur, nxt;
    if (b < nblk) nxt.load(rws, b, t);
    for (int i = 0; i < GPW; i++, gi++) {
        for (b = gi; b < nblk; b += G) {
            cur = nxt;
            int bn = b + G < nblk ? b + G : (i + 1 < GPW && gi + 1 < nblk ? gi + 1 : -1);
            if (bn >= 0) nxt.load(rws, bn, t);
            if constexpr (QT == QT_Q4_K) {
                float d[2], dmin[2], dm0[2];
                u32 scw[2][2], mw[2][2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    uint4 hd = cur.h[h];
                    d[h] = h2f(hd.x & 0xffff);
                    dmin[h] = h2f(hd.x >> 16);
                    dm0[h] = __fmul_rn(dmin[h], 0.0f);
                    scw[h][0] = hd.y & 0x3f3f3f3f;
                    scw[h][1] = (hd.w & 0x0f0f0f0f) | ((hd.y >> 2) & 0x30303030);
                    mw[h][0] = hd.z & 0x3f3f3f3f;
                    mw[h][1] = ((hd.w >> 4) & 0x0f0f0f0f) | ((hd.z >> 2) & 0x30303030);
                }
                float dk[NO];
#pragma unroll
                for (int o = 0; o < NO; o++) dk[o] = *(const float*)(xo[o] + 2 * np + 4 * (u64)b);
#pragma unroll
                for (int cc = 0; cc < 4; cc++) {
                    int sc1[2], sc2[2], m1[2], m2[2];
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        sc1[h] = gm_ubyte(scw[h][cc >> 1], 2 * cc);
                        sc2[h] = gm_ubyte(scw[h][cc >> 1], 2 * cc + 1);
                        m1[h] = gm_ubyte(mw[h][cc >> 1], 2 * cc);
                        m2[h] = gm_ubyte(mw[h][cc >> 1], 2 * cc + 1);
                    }
                    float dm[NO];
#pragma unroll
                    for (int o = 0; o < NO; o++) {
                        int h = (o & 2) >> 1;
                        // (4-byte aligned only: two loads)
                        const int* bs = (const int*)(xo[o] + 2 * np + np / 64) + (u64)b * 8 + 2 * cc;
                        dm[o] = __fmul_rn(dmin[h], gm_i2f(m1[h] * bs[0] + m2[h] * bs[1]));
                    }
#pragma unroll
                    for (int hf = 0; hf < 2; hf++) {
                        int p = 2 * cc + hf;
                        u32 qa = cur.q[0][p], qb = cur.q[1][p];
                        u32 a10 = qa & 0x0f0f0f0f, a11 = qb & 0x0f0f0f0f;
                        u32 a20 = (qa >> 4) & 0x0f0f0f0f, a21 = (qb >> 4) & 0x0f0f0f0f;
                        int j1 = 4 * cc + hf, j2 = j1 + 2;
#pragma unroll
                        for (int f = 0; f < NF; f++) {
                            const u8* xb = xa[f] + (u64)b * 256 + 4 * t;
                            int d1[4], d2[4];
                            mma16(d1, a10, a11, *(const u32*)(xb + 16 * j1));
                            mma16(d2, a20, a21, *(const u32*)(xb + 16 * j2));
#pragma unroll
                            for (int e = 0; e < 4; e++) {
                                int o = 4 * f + e, h = e >> 1;
                                float si = gm_i2f(sc1[h] * d1[e] + sc2[h] * d2[e]);
                                float v = __fsub_rn(__fmul_rn(d[h], si), hf ? dm0[h] : dm[o]);
                                c[p][o] = __fmaf_rn(dk[o], v, c[p][o]);
                            }
                        }
                    }
                }
            } else if constexpr (QT == QT_Q6_K) {
                float dkd[NO];
#pragma unroll
                for (int o = 0; o < NO; o++)
                    dkd[o] = __fmul_rn(*(const float*)(xo[o] + 2 * np + 4 * (u64)b),
                                       cur.d[(o & 2) >> 1]);
#pragma unroll
                for (int pp = 0; pp < 4; pp++) {
                    int hh = pp >> 1, qhi = pp & 1, px = 4 * hh + 2 * qhi;
                    int sx[NO], sy[NO];
#pragma unroll
                    for (int o = 0; o < NO; o++) sx[o] = sy[o] = 0;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int j = 8 * hh + 2 * k + qhi, ii = 2 * hh + qhi;
                        int sca = gm_sbyte(cur.sc[0], j), scb = gm_sbyte(cur.sc[1], j);
                        u32 wv[2];
#pragma unroll
                        for (int r = 0; r < 2; r++) {
                            u32 l = cur.ql[r][ii][k & 1];
                            u32 lo = k < 2 ? (l & 0x0f0f0f0f) : ((l >> 4) & 0x0f0f0f0f);
                            wv[r] = __vsub4(lo | (((cur.qh[r][ii] >> (2 * k)) & 0x03030303) << 4),
                                            0x20202020);
                        }
                        u32 xa0 = t < 2 ? wv[0] : 0, xa1 = t < 2 ? 0 : wv[0];
                        u32 xb0 = t < 2 ? wv[1] : 0, xb1 = t < 2 ? 0 : wv[1];
#pragma unroll
                        for (int f = 0; f < NF; f++) {
                            u32 bb = *(const u32*)(xa[f] + (u64)b * 256 + 16 * j + 4 * t);
                            int da[4], db[4];
                            mma16(da, xa0, xa1, bb);
                            mma16(db, xb0, xb1, bb);
                            sx[4 * f + 0] += sca * da[0];
                            sx[4 * f + 1] += sca * da[1];
                            sy[4 * f + 0] += sca * da[2];
                            sy[4 * f + 1] += sca * da[3];
                            sx[4 * f + 2] += scb * db[0];
                            sx[4 * f + 3] += scb * db[1];
                            sy[4 * f + 2] += scb * db[2];
                            sy[4 * f + 3] += scb * db[3];
                        }
                    }
#pragma unroll
                    for (int o = 0; o < NO; o++) {
                        c[px][o] = __fmaf_rn(dkd[o], (float)sx[o], c[px][o]);
                        c[px + 1][o] = __fmaf_rn(dkd[o], (float)sy[o], c[px + 1][o]);
                    }
                }
            } else {
#pragma unroll
                for (int f = 0; f < NF; f++) {
                    const u8* xb = xa[f] + np + (u64)b * 32 + 4 * t;
                    int dd[4];
                    mma32(dd, cur.a[0][0], cur.a[1][0], cur.a[0][1], cur.a[1][1], *(const u32*)xb,
                          *(const u32*)(xb + 16));
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        int o = 4 * f + e;
                        float d0 = *(const float*)(xo[o] + 2 * np + np / 64 + np / 8 + 4 * (u64)b);
                        float sd = __fmul_rn(cur.d[e >> 1], d0);
                        c[0][o] = __fadd_rn(c[0][o], __fmul_rn(sd, gm_i2f(dd[e])));
                    }
                }
            }
        }
        // Group gi is done (or had no blocks): fold its lanes.
#pragma unroll
        for (int o = 0; o < NO; o++) {
            if (KQ) {
                float cs[8];
#pragma unroll
                for (int p = 0; p < 8; p++) {
                    cs[p] = c[p][o];
                    c[p][o] = 0.0f;
                }
                gm_fold<4>(tree[o], i, gm_tree8(cs));
            } else {
                gm_fold<8>(tree[o], i, gm_tree4(c[0][o]));
                c[0][o] = 0.0f;
            }
        }
    }
    // tree[o].r = this warp's W; cta_sum: ((W0 + W1) + W2) + W3.
#pragma unroll
    for (int o = 0; o < NO; o++)
        sh[warp][g + ((o & 2) ? 8 : 0)][8 * (o >> 2) + 2 * t + (o & 1)] = tree[o].r;
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int o = 0; o < NO; o++) {
            int rr = g + ((o & 2) ? 8 : 0), zz = 8 * (o >> 2) + 2 * t + (o & 1);
            float v = sh[0][rr][zz];
            for (int k = 1; k < 4; k++) v = __fadd_rn(v, sh[k][rr][zz]);
            if (r0 + rr < rows && zz < nb) gm_store(v, bias, mode, out + (u64)zz * ostride, r0 + rr);
        }
    }
#else
    __trap();
#endif
}

#define GEMVM(NAME, QT, NF)                                                                       \
    extern "C" __global__ void __launch_bounds__(GV_THREADS) NAME(                                \
        const u8* __restrict__ w, u64 rb, int rows, int n, const u8* __restrict__ act,           \
        u64 astride, int nb, const float* __restrict__ bias, int mode, float* __restrict__ out,  \
        u64 ostride) {                                                                            \
        gemvm_t<QT, NF>(w, rb, rows, n, act, astride, nb, bias, mode, out, ostride);              \
    }
GEMVM(gemvm8_q4_k, QT_Q4_K, 1)
GEMVM(gemvm8_q6_k, QT_Q6_K, 1)
GEMVM(gemvm8_q8_0, QT_Q8_0, 1)
GEMVM(gemvm8_q5_0, QT_Q5_0, 1)
GEMVM(gemvm16_q4_k, QT_Q4_K, 2)
GEMVM(gemvm16_q6_k, QT_Q6_K, 2)
GEMVM(gemvm16_q8_0, QT_Q8_0, 2)
GEMVM(gemvm16_q5_0, QT_Q5_0, 2)

// Test: for every half h and a few x, fmul(h2f(h), x) == fmul(f16_to_f32(h),
// x) bit for bit (the GEMMs convert block scales with the hardware cvt, and
// only ever multiply them; the two conversions differ in NaN payloads only).
// out[0] = mismatches. Grid 256 x 256.
extern "C" __global__ void h2f_mul_check(u32* out) {
    u32 h = blockIdx.x * 256 + threadIdx.x;
    const float xs[6] = {1.0f, -3.5f, 0.0f, 1e-30f, INFINITY, NAN_F};
    float a = f16_to_f32(h), b = h2f((unsigned short)h);
    for (int i = 0; i < 6; i++)
        if (__float_as_uint(__fmul_rn(a, xs[i])) != __float_as_uint(__fmul_rn(b, xs[i])))
            atomicAdd(out, 1u); // a count: order doesn't matter
}

// ---------------------------------------------------------------------------
// gemmf_q4_k: NOT bit-identical, for measurement only (cuda_gemm_bench): what
// the exact epilogue costs. The usual fast form (llama.cpp's MMQ): an mma
// sum per 32-value sub-block, d sc / dmin m scaled once per sub-block, one
// running sum per output in another order. Same tiles and copies as
// gemms1_q4_k (32 rows x 32 tokens).
extern "C" __global__ void __launch_bounds__(GM_THREADS, 2)
gemmf_q4_k(const u8* __restrict__ w, u64 rb, int rows, int n, const u8* __restrict__ act,
           u64 astride, int nb, const float* __restrict__ bias, int mode, float* __restrict__ out,
           u64 ostride) {
#if __CUDA_ARCH__ >= 800
    typedef GmStage<GM_BN(1)> St;
    extern __shared__ uint4 gm_smem[];
    St* st = (St*)gm_smem;
    u32* seq = (u32*)(st + 2);
    int nblk = n / 256;
    for (int q = threadIdx.x; q < nblk; q += GM_THREADS) seq[q] = gm_seq(q, nblk, 16);
    __syncthreads();
    int r0 = blockIdx.y * GM_BM, z0 = blockIdx.x * GM_BN(1);
    int warp = threadIdx.x >> 5, wr = warp & 1, wc = warp >> 1;
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int ra = 16 * wr + g, rbw = ra + 8, tb = 8 * wc + g;
    int tk[4];
#pragma unroll
    for (int o = 0; o < 4; o++) tk[o] = 8 * wc + 2 * t + (o & 1);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    gm_copy<QT_Q4_K, 1>(&st[0], w, rb, rows, r0, act, astride, n, nb, z0, nblk, 0, seq);
    cp_commit();
    for (int s = 0; s < nblk; s++) {
        if (s + 1 < nblk) {
            gm_copy<QT_Q4_K, 1>(&st[(s + 1) & 1], w, rb, rows, r0, act, astride, n, nb, z0, nblk,
                                s + 1, seq);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const St* S = &st[s & 1];
        float dsc[2][8], dmm[2][8];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            uint4 hd = S->w[h ? rbw : ra][0];
            float d = h2f(hd.x & 0xffff), dmin = h2f(hd.x >> 16);
            u32 sc0 = hd.y & 0x3f3f3f3f, sc1 = (hd.w & 0x0f0f0f0f) | ((hd.y >> 2) & 0x30303030);
            u32 m0 = hd.z & 0x3f3f3f3f, m1 = ((hd.w >> 4) & 0x0f0f0f0f) | ((hd.z >> 2) & 0x30303030);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                dsc[h][j] = d * (float)gm_ubyte(j < 4 ? sc0 : sc1, j);
                dmm[h][j] = dmin * (float)gm_ubyte(j < 4 ? m0 : m1, j);
            }
        }
#pragma unroll
        for (int c = 0; c < 4; c++) {
            u32 qa0 = gm_word(S->w, ra, 1 + 2 * c, t), qa1 = gm_word(S->w, ra, 2 + 2 * c, t);
            u32 qb0 = gm_word(S->w, rbw, 1 + 2 * c, t), qb1 = gm_word(S->w, rbw, 2 + 2 * c, t);
            u32 x0 = gm_word(S->x, tb, 4 * c, t), x1 = gm_word(S->x, tb, 4 * c + 1, t);
            u32 x2 = gm_word(S->x, tb, 4 * c + 2, t), x3 = gm_word(S->x, tb, 4 * c + 3, t);
            int lo[4], hi[4];
            mma32(lo, qa0 & 0x0f0f0f0f, qb0 & 0x0f0f0f0f, qa1 & 0x0f0f0f0f, qb1 & 0x0f0f0f0f, x0, x1);
            mma32(hi, (qa0 >> 4) & 0x0f0f0f0f, (qb0 >> 4) & 0x0f0f0f0f, (qa1 >> 4) & 0x0f0f0f0f,
                  (qb1 >> 4) & 0x0f0f0f0f, x2, x3);
#pragma unroll
            for (int o = 0; o < 4; o++) {
                int h = o >> 1;
                float dk = S->xd[tk[o]][0];
                float b0 = (float)S->xb[tk[o]][2 * c], b1 = (float)S->xb[tk[o]][2 * c + 1];
                float v = dsc[h][2 * c] * gm_i2f(lo[o]) - dmm[h][2 * c] * b0 +
                          dsc[h][2 * c + 1] * gm_i2f(hi[o]) - dmm[h][2 * c + 1] * b1;
                acc[o] = fmaf(dk, v, acc[o]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int o = 0; o < 4; o++) {
        int r = r0 + 16 * wr + g + ((o & 2) ? 8 : 0), z = z0 + tk[o];
        if (r < rows && z < nb) gm_store(acc[o], bias, mode, out + (u64)z * ostride, r);
    }
#else
    __trap();
#endif
}
