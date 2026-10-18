
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
