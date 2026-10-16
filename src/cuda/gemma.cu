
// ---------------------------------------------------------------------------
// Gemma 3 (src/cuda/qwen2.rs with `gemma`): head size 256, sliding-window
// layers, the embedding times sqrt(dim), RMSNorms of the attention and FFN
// outputs before their residual adds, GELU (tanh form) in the gated FFN.
// Decode kernels, then their batched twins; every batched one runs the
// decode one's code per token, so the batching contract (prefill.cu) holds.
// Head size 256 runs the split attention (attn_split_t) on either path: no
// tensor-core attention or q8 cache instantiations at 256 (the host turns
// them off).

// The first position a layer of window `win` attends to from the query at
// np - 1 (llama.cpp LLAMA_SWA_TYPE_STANDARD: key t is masked iff
// np - 1 - t >= win). win 0: a global layer, every position.
__device__ __forceinline__ int win_lo(int np, int win) {
    return win > 0 && np > win ? np - win : 0;
}

// gelu_tanh(g) * u, as vmath::geglu1_f: g / (1 + exp(-2 sqrt(2/pi) z)) * u
// with z = fma(0.044715 g g, g, g); expf for exp (as gate_up's silu).
__device__ __forceinline__ float geglu1(float g, float u) {
    float z = __fmaf_rn(__fmul_rn(__fmul_rn(0.044715f, g), g), g, g);
    float e = expf(__fmul_rn(-2.0f * 0.7978846f, z));
    return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, e)), u);
}

// x += rmsnorm(y) * g (model.rs: `rmsnorm` of the attention or FFN output,
// then the residual add): the sum of squares strided over the block's
// threads in order, block_sum, inv = 1 / sqrt(ss / n + eps), then x[i] +
// y[i] * inv * g[i]. One block of 1024 threads per vector.
__device__ void rmsnorm_add_dev(const float* __restrict__ y, const float* __restrict__ g, float eps,
                                int n, float* __restrict__ x) {
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss = __fadd_rn(ss, __fmul_rn(y[i], y[i]));
    ss = block_sum(ss, sh);
    float ms = __fdiv_rn(ss, (float)n);
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ms, eps)));
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        x[i] = __fadd_rn(x[i], __fmul_rn(__fmul_rn(y[i], inv), g[i]));
}

extern "C" __global__ void rmsnorm_add(const float* __restrict__ y, const float* __restrict__ g,
                                       float eps, int n, float* __restrict__ x) {
    PDL_START();
    rmsnorm_add_dev(y, g, eps, n, x);
}

// embed, times `scale` (an f32 multiply after the exact dequantization, as
// model.rs embed). Launched without PDL like embed.
extern "C" __global__ void embed_s(int qt, const u8* __restrict__ w, u64 row_bytes, int n,
                                   const u32* __restrict__ P, float scale, float* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fmul_rn(deq(qt, w + (u64)P[P_TOKEN] * row_bytes, i), scale);
}

// gate_up with GELU (tanh form) in place of SiLU: out[r] = geglu1(gate_r .
// x, up_r . x). One row of each matrix per block.
extern "C" __global__ void __launch_bounds__(GEMV_THREADS, GEMV_MIN_BLOCKS)
gate_up_gelu(int qtg, const u8* __restrict__ wg, u64 rbg, int qtu, const u8* __restrict__ wu,
             u64 rbu, int rows, int n, const u8* __restrict__ act, float* __restrict__ out) {
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
    if (threadIdx.x == 0) out[r] = geglu1(y[0], y[1]);
}

// attn_split_256 over the layer's window: the positions win_lo..np of the
// cache, as a cache that starts there (the split schedule then depends on
// the window's length, which attn_combine_w uses too). qn non-null: the q/k
// norms and RoPE (QkNorm) with `rope` this layer's table.
extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_w_256(const float* __restrict__ q, const unsigned short* kc,
                 const unsigned short* __restrict__ vc, const u32* __restrict__ P, int n_heads,
                 int n_kv, float scale, int max_split, float* __restrict__ part_o,
                 float* __restrict__ part_ml, const float* __restrict__ kraw,
                 const float* __restrict__ qn, const float* __restrict__ kn, float eps,
                 const float2* __restrict__ rope, unsigned short* kc_out, int win) {
    PDL_START();
    int np = (int)P[P_POS] + 1, lo = win_lo(np, win);
    u64 off = (u64)lo * n_kv * 256;
    attn_split_t<256>(q, kc + off, vc + off, np - lo, blockIdx.x, blockIdx.y, n_heads, n_kv, scale,
                      max_split, part_o, part_ml, Q8Kv{},
                      QkNorm{kraw, qn, kn, eps, rope, kc_out ? kc_out + off : nullptr});
}

// attn_combine for attn_split_w_256's partials (the window's length).
extern "C" __global__ void attn_combine_w(const float* __restrict__ part_o,
                                          const float* __restrict__ part_ml,
                                          const u32* __restrict__ P, int n, int hd, int max_split,
                                          int want_k, u8* __restrict__ act, float* __restrict__ y,
                                          int win) {
    PDL_START();
    int np = (int)P[P_POS] + 1;
    attn_combine_dev(part_o, part_ml, np - win_lo(np, win), blockIdx.x, n, hd, max_split, want_k,
                     act, y);
}

// ---------------------------------------------------------------------------
// Batched twins (prefill.cu conventions: rows at x + z * stride, activations
// at act + z * astride; token z's position and cache row by its Placement:
// rp null, start + z for both; else rp's).

// embed_b times `scale` (embed_s per token). Grid (ceil(n / 256), nb) x 256.
extern "C" __global__ void embed_bs(int qt, const u8* __restrict__ w, u64 row_bytes, int n,
                                    const u32* __restrict__ tokens, float scale,
                                    float* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, z = blockIdx.y;
    if (i < n) x[(u64)z * n + i] = __fmul_rn(deq(qt, w + (u64)tokens[z] * row_bytes, i), scale);
}

// rmsnorm_add per token: x[z] += rmsnorm(y[z]) * g, y rows of `ystride`
// floats, x rows of n. Grid nb x 1024 (the decode kernel's block).
extern "C" __global__ void rmsnorm_add_b(const float* __restrict__ y, u64 ystride,
                                         const float* __restrict__ g, float eps, int n,
                                         float* __restrict__ x) {
    int z = blockIdx.x;
    rmsnorm_add_dev(y + (u64)z * ystride, g, eps, n, x + (u64)z * n);
}

// hid = geglu1(gate, up) (gate_up_gelu's op), quantized into act[z]
// (swiglu_quant_b with GELU). Grid (ceil(n / 256), nb) x 32.
extern "C" __global__ void geglu_quant_b(const float* __restrict__ gate,
                                         const float* __restrict__ up, int n, int want_k,
                                         u8* __restrict__ act, u64 astride) {
    int lane = threadIdx.x & 31, blk = blockIdx.x, z = blockIdx.y;
    const float* gz = gate + (u64)z * n;
    const float* uz = up + (u64)z * n;
    float v[8];
    for (int i = 0; i < 8; i++) {
        int e = blk * 256 + lane * 8 + i;
        float y0 = e < n ? gz[e] : 0.0f, y1 = e < n ? uz[e] : 0.0f;
        float h = geglu1(y0, y1);
        v[i] = e < n ? h : 0.0f;
    }
    quant_warp(v, n, blk, act + z * astride, want_k);
}

// rope_kv_norm_b for head size 256 (warp_norm_rope's 4 pairs a lane): the
// same per-head work, the same ops. Grid (ceil(heads / 4), nb) x 128.
extern "C" __global__ void rope_kv_norm_b_256(const float* __restrict__ qkv, int nq, int nk,
                                              int start, const float2* __restrict__ rope,
                                              const float* __restrict__ qn,
                                              const float* __restrict__ kn, float eps,
                                              float* __restrict__ q,
                                              unsigned short* __restrict__ kc,
                                              unsigned short* __restrict__ vc,
                                              const u32* __restrict__ rp) {
    const int hd = 256, half = 128;
    int z = blockIdx.y, h = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    int hq = nq / hd, hk = nk / hd;
    if (h >= hq + 2 * hk) return;
    const float* y = qkv + (u64)z * (nq + 2 * nk);
    u64 pos = rp_row(rp, start, z); // the cache row
    if (h >= hq + hk) {
        for (int e = (h - hq - hk) * hd + lane; e < (h - hq - hk + 1) * hd; e += 32)
            vc[pos * nk + e] = kv_half(y[nq + nk + e]);
        return;
    }
    bool isq = h < hq;
    int e0 = (isq ? h : h - hq) * hd;
    const float* yy = isq ? y : y + nq;
    float a[4], b[4];
    for (int i = 0; i < 4; i++) {
        a[i] = yy[e0 + lane + 32 * i];
        b[i] = yy[e0 + lane + 32 * i + half];
    }
    warp_norm_rope(a, b, hd, isq ? qn : kn, eps, rope + (u64)z * half);
    for (int i = 0; i < 4; i++) {
        int e = e0 + lane + 32 * i;
        if (isq) {
            q[(u64)z * nq + e] = a[i];
            q[(u64)z * nq + e + half] = b[i];
        } else {
            kc[pos * nk + e] = kv_half(a[i]);
            kc[pos * nk + e + half] = kv_half(b[i]);
        }
    }
}

// attn_split_b at head size 256 over each query's window (attn_split_w_256
// per query token: the same cache view, splits and code; rp: the token's
// sequence's rows). Grid (n_kv, max_split, nz) x ATT_TILE; part_* per query
// of the launch.
extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_bw_256(const float* __restrict__ q, const unsigned short* __restrict__ kc,
                  const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
                  float scale, int max_split, float* __restrict__ part_o,
                  float* __restrict__ part_ml, const u32* __restrict__ rp, int win) {
    int zl = blockIdx.z, z = z0 + zl;
    u64 nh = n_heads;
    int np = (int)rp_pos(rp, start, z) + 1, lo = win_lo(np, win);
    u64 off = (rp_base(rp, z) + (u64)lo) * n_kv * 256;
    attn_split_t<256>(q + (u64)z * nh * 256, kc + off, vc + off, np - lo, blockIdx.x, blockIdx.y,
                      n_heads, n_kv, scale, max_split, part_o + (u64)zl * nh * max_split * 256,
                      part_ml + (u64)zl * nh * max_split * 2);
}

// attn_combine_b for attn_split_bw_256's partials: its code with the
// window's length for np (keep the two in step).
extern "C" __global__ void attn_combine_bw(const float* __restrict__ part_o,
                                           const float* __restrict__ part_ml, int start, int z0,
                                           int n, int hd, int max_split, int want_k,
                                           u8* __restrict__ act, u64 astride,
                                           const u32* __restrict__ rp, int win) {
    __shared__ float vals[256];
    __shared__ float ws[256 / 64][MAX_SPLIT];
    __shared__ float linv[256 / 64];
    int zl = blockIdx.y, z = z0 + zl, blk = blockIdx.x;
    u64 nh = n / hd;
    const float* pob = part_o + (u64)zl * nh * max_split * hd;
    const float* pmb = part_ml + (u64)zl * nh * max_split * 2;
    u8* actz = act + z * astride;
    int e = blk * 256 + threadIdx.x, lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int np = (int)rp_pos(rp, start, z) + 1, nsplit, chunk;
    np -= win_lo(np, win);
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
