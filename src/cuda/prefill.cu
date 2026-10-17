
// ---------------------------------------------------------------------------
// Batched kernels: nb tokens through a Qwen2 model at once (prefill and the
// speculative verify pass, src/cuda/qwen2.rs).
//
// The contract: every token's KV cache rows and logits are BIT-IDENTICAL to
// the decode kernels' (one token at a time), whatever the batch size or the
// token's place in it. The elementwise kernels run decode.cu's code per
// token. The matrix products follow the canonical order of decode.cu's
// "Row dot products": per output, the exact integer sums of each block
// (however computed: dp4a, mma), then a few float ops per block into one
// fma chain over the blocks in order. gemv_ref spells it out. The attention
// follows decode.cu's "Tensor-core attention" order: attn_fa_b does a
// block of rows at a time over all positions, attn_blk_b runs the decode
// kernel per query (GPUs without tensor cores: attn_split_b, the split
// decode kernel per query).
//
// Token-batched buffers: token z's activation (decode.cu Act) at act + z *
// astride; f32 rows at x + z * stride.
//
// Placement: a pass is one sequence (rp null: token z at position start +
// z, cache row start + z) or, for the server's multi-sequence decode step
// (serve S2), one row per token from rp: rp[2z] = its position, rp[2z + 1]
// = its cache row; its sequence's rows start at row - position (a slot of
// the cache). Every per-token op is the same on the same position and
// rows, so the bits are those of the one-sequence pass.
// Paged (tab non-null, decode.cu pg_row): one sequence's rows through tab
// (its table); with rp, rp[2z + 1] is the entry of the pool's table tab
// where token z's sequence's table starts, and its rows go through that.
__device__ __forceinline__ u32 rp_pos(const u32* rp, int start, int z) {
    return rp ? rp[2 * z] : (u32)(start + z);
}
// Token z's table (null: none).
__device__ __forceinline__ const u32* rp_tab(const u32* rp, int z, const u32* tab) {
    return tab && rp ? tab + rp[2 * z + 1] : tab;
}
// Token z's cache row.
__device__ __forceinline__ u64 rp_row(const u32* rp, int start, int z, const u32* tab) {
    if (tab) return pg_row(rp_tab(rp, z, tab), (int)rp_pos(rp, start, z));
    return rp ? (u64)rp[2 * z + 1] : (u64)(start + z);
}
// The row token z's sequence's rows are counted from (0 for one sequence
// and for a table: the table has them).
__device__ __forceinline__ u64 rp_base(const u32* rp, int z, const u32* tab) {
    return rp && !tab ? (u64)(rp[2 * z + 1] - rp[2 * z]) : 0;
}

// ---------------------------------------------------------------------------
// Elementwise, one token per grid row (blockIdx.y or blockIdx.x as noted).

// x[z] = embedding row tokens[z] (embed). Grid (ceil(n / 256), nb) x 256.
extern "C" __global__ void embed_b(int qt, const u8* __restrict__ w, u64 row_bytes, int n,
                                   const u32* __restrict__ tokens, float* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, z = blockIdx.y;
    if (i < n) x[(u64)z * n + i] = deq(qt, w + (u64)tokens[z] * row_bytes, i);
}

// rope[z] = rope_table at position start + z (rp: rp_pos), times m as
// rope_table. Grid nb x half.
extern "C" __global__ void rope_table_b(const double* __restrict__ freq, int half, int start,
                                        float2* __restrict__ rope, const u32* __restrict__ rp,
                                        float m) {
    int i = threadIdx.x, z = blockIdx.x;
    if (i < half) {
        double s, c;
        sincos((double)rp_pos(rp, start, z) * freq[i], &s, &c);
        rope[(u64)z * half + i] = make_float2((float)(s * (double)m), (float)(c * (double)m));
    }
}

// rmsnorm_quant of row z of x (n floats per row) into act[z], by 256
// threads, each standing in for 4 of rmsnorm_quant_dev's 1024 (threads t,
// t + 256, t + 512, t + 768: virtual warps w, w + 8, w + 16, w + 24): the
// same sums of squares in the same order (each virtual thread's strided
// chain, warp_sum per virtual warp, then the 32 warp sums by warp_sum), the
// same elementwise ops and quant_warp per 256 values; a quarter of the
// threads keeps more rows in flight per SM. Grid nb x 256.
extern "C" __global__ void __launch_bounds__(256)
rmsnorm_quant_b(const float* __restrict__ x, const float* __restrict__ g, float eps, int n,
                int want_k, u8* __restrict__ act, u64 astride) {
    __shared__ float sh[32];
    int z = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const float* xz = x + (u64)z * n;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float a = 0.0f;
        for (int i = tid + 256 * j; i < n; i += 1024) a = __fadd_rn(a, __fmul_rn(xz[i], xz[i]));
        a = warp_sum(a);
        if (lane == 0) sh[w + 8 * j] = a;
    }
    __syncthreads();
    float t = warp_sum(sh[lane]);
    float ms = __fdiv_rn(t, (float)n);
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ms, eps)));
    u8* az = act + z * astride;
    for (int blk = w; blk * 256 < n; blk += 8) {
        float v[8];
        int e0 = blk * 256 + lane * 8;
        for (int i = 0; i < 8; i++) {
            int e = e0 + i;
            v[i] = e < n ? __fmul_rn(__fmul_rn(xz[e], inv), g[e]) : 0.0f;
        }
        quant_warp(v, n, blk, az, want_k);
    }
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
// With a null kc (the q8 cache: kv_q8_b stores keys and values), queries only.
extern "C" __global__ void rope_kv_b(const float* __restrict__ qkv, int nq, int nk, int hd,
                                     int start, const float2* __restrict__ rope,
                                     float* __restrict__ q, unsigned short* __restrict__ kc,
                                     unsigned short* __restrict__ vc,
                                     const u32* __restrict__ rp, const u32* __restrict__ tab) {
    int z = blockIdx.y, item = blockIdx.x * blockDim.x + threadIdx.x;
    int half = hd >> 1, npq = nq >> 1, npk = nk >> 1;
    const float* y = qkv + (u64)z * (nq + 2 * nk);
    u64 pos = rp_row(rp, start, z, tab); // the cache row
    if (item >= npq + npk + nk || (!kc && item >= npq)) return;
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

// The q8 cache store for token z (position start + z), kv head blockIdx.x:
// its keys rotated as rope_kv_b does (the same ops), its values, quantized
// by kv_q8_store (decode.cu's rule and code). Grid (n_kv, nb) x 128.
// kn non-null (qwen3): the keys are normalized and rotated by
// warp_norm_rope instead, as the decode q8 attention does.
template <int HD, int KG>
__device__ __forceinline__ void kv_q8_bt(const float* __restrict__ qkv, int nq, int nk, int start,
                                         const float2* __restrict__ rope, Q8Kv c,
                                         const float* __restrict__ kn, float eps,
                                         const u32* __restrict__ rp, const u32* __restrict__ tab) {
    __shared__ float xs[2 * HD];
    int kvh = blockIdx.x, z = blockIdx.y, tid = threadIdx.x, half = HD / 2;
    const float* y = qkv + (u64)z * (nq + 2 * nk) + nq + kvh * HD;
    if (kn) {
        if (tid < 32) {
            float a[2], b[2];
            for (int i = 0; i < (HD >> 6); i++) {
                a[i] = y[tid + 32 * i];
                b[i] = y[tid + 32 * i + half];
            }
            warp_norm_rope(a, b, HD, kn, eps, rope + (u64)z * half);
            for (int i = 0; i < (HD >> 6); i++) {
                xs[tid + 32 * i] = a[i];
                xs[tid + 32 * i + half] = b[i];
            }
        }
    } else if (tid < half) {
        float y0 = y[tid], y1 = y[tid + half];
        float2 sc = rope[(u64)z * half + tid];
        float sn = sc.x, cs = sc.y;
        xs[tid] = __fsub_rn(__fmul_rn(y0, cs), __fmul_rn(y1, sn));
        xs[tid + half] = __fadd_rn(__fmul_rn(y0, sn), __fmul_rn(y1, cs));
    }
    for (int i = tid; i < HD; i += blockDim.x) xs[HD + i] = y[nk + i];
    __syncthreads();
    kv_q8_store<HD, KG>(xs, c, rp_row(rp, start, z, tab) * nk + kvh * HD,
                        c.ksh ? c.ksh + kvh * HD : nullptr);
}

#define KV_Q8_B(NAME, HD, KG)                                                                      \
    extern "C" __global__ void __launch_bounds__(128)                                              \
        NAME(const float* __restrict__ qkv, int nq, int nk, int start,                             \
             const float2* __restrict__ rope, signed char* __restrict__ kq,                        \
             signed char* __restrict__ vq, float* __restrict__ ks, float* __restrict__ vs,         \
             const float* __restrict__ ksh, const float* __restrict__ kn, float eps,               \
             const u32* __restrict__ rp, const u32* __restrict__ tab) {                            \
        kv_q8_bt<HD, KG>(qkv, nq, nk, start, rope, Q8Kv{kq, vq, ks, vs, nullptr, ksh}, kn, eps,    \
                         rp, tab);                                                                 \
    }
KV_Q8_B(kv_q8_b_64, 64, 64)
KV_Q8_B(kv_q8_b_128, 128, 128)
KV_Q8_B(kv_q8g32_b_64, 64, 32)
KV_Q8_B(kv_q8g32_b_128, 128, 32)

// rope_kv_b with qwen3's per-head q/k norms (decode: attn_split_t's
// QkNorm). Grid (ceil(heads / 4), nb) of 128 threads, a warp per head of
// [nq / hd query | nk / hd key | nk / hd value heads]: queries and keys
// through warp_norm_rope, values copied to the cache. A null kc (the q8
// cache: kv_q8_b normalizes, rotates and stores keys and values): queries
// only.
extern "C" __global__ void rope_kv_norm_b(const float* __restrict__ qkv, int nq, int nk, int hd,
                                          int start, const float2* __restrict__ rope,
                                          const float* __restrict__ qn,
                                          const float* __restrict__ kn, float eps,
                                          float* __restrict__ q, unsigned short* __restrict__ kc,
                                          unsigned short* __restrict__ vc,
                                          const u32* __restrict__ rp, const u32* __restrict__ tab) {
    int z = blockIdx.y, h = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    int half = hd >> 1, hq = nq / hd, hk = nk / hd;
    if (h >= hq + 2 * hk || (!kc && h >= hq)) return;
    const float* y = qkv + (u64)z * (nq + 2 * nk);
    u64 pos = rp_row(rp, start, z, tab); // the cache row
    if (h >= hq + hk) {
        for (int e = (h - hq - hk) * hd + lane; e < (h - hq - hk + 1) * hd; e += 32)
            vc[pos * nk + e] = kv_half(y[nq + nk + e]);
        return;
    }
    bool isq = h < hq;
    int e0 = (isq ? h : h - hq) * hd;
    const float* yy = isq ? y : y + nq;
    float a[2], b[2];
    for (int i = 0; i < (half >> 5); i++) {
        a[i] = yy[e0 + lane + 32 * i];
        b[i] = yy[e0 + lane + 32 * i + half];
    }
    warp_norm_rope(a, b, hd, isq ? qn : kn, eps, rope + (u64)z * half);
    for (int i = 0; i < (half >> 5); i++) {
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

// Decode attention for query tokens z0 + blockIdx.z (position start + z),
// all of whose keys and values are in the cache: the decode schedule and
// code per query (same splits, tiles and sums). q: [token][n_heads * hd];
// part_*: per query of this launch, as attn_split's.
// The cache from element e on (e = a sequence's first row x kv_dim): f16
// K and V pointers, or the q8 rows and their scales (group KG).
__device__ __forceinline__ const unsigned short* kv_at(const unsigned short* c, u64 e) {
    return c ? c + e : c;
}
template <int KG>
__device__ __forceinline__ Q8Kv q8_at(Q8Kv c, u64 e) {
    if constexpr (KG > 0) {
        if (c.k) {
            c.k += e;
            c.v += e;
            c.ks += e / KG;
            c.vs += e / KG;
        }
    }
    return c;
}

template <int HD, int KG = 0>
__device__ __forceinline__ void attn_split_bt(const float* q, const unsigned short* kc,
                                              const unsigned short* vc, int start, int z0,
                                              int n_heads, int n_kv, float scale, int max_split,
                                              float* part_o, float* part_ml, const u32* rp,
                                              const u32* tab, Q8Kv q8 = Q8Kv{}) {
    int zl = blockIdx.z, z = z0 + zl;
    u64 nh = n_heads, e = rp_base(rp, z, tab) * (u64)(n_kv * HD);
    attn_split_t<HD, KG>(q + (u64)z * nh * HD, kv_at(kc, e), kv_at(vc, e),
                         rp_pos(rp, start, z) + 1, blockIdx.x, blockIdx.y, n_heads, n_kv, scale,
                         max_split, part_o + (u64)zl * nh * max_split * HD,
                         part_ml + (u64)zl * nh * max_split * 2, q8_at<KG>(q8, e), QkNorm{},
                         rp_tab(rp, z, tab));
}

extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_b_64(const float* __restrict__ q, const unsigned short* __restrict__ kc,
                const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
                float scale, int max_split, float* __restrict__ part_o,
                float* __restrict__ part_ml, const u32* __restrict__ rp,
                const u32* __restrict__ tab) {
    attn_split_bt<64>(q, kc, vc, start, z0, n_heads, n_kv, scale, max_split, part_o, part_ml, rp,
                      tab);
}

extern "C" __global__ void __launch_bounds__(ATT_TILE)
attn_split_b_128(const float* __restrict__ q, const unsigned short* __restrict__ kc,
                 const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
                 float scale, int max_split, float* __restrict__ part_o,
                 float* __restrict__ part_ml, const u32* __restrict__ rp,
                 const u32* __restrict__ tab) {
    attn_split_bt<128>(q, kc, vc, start, z0, n_heads, n_kv, scale, max_split, part_o, part_ml, rp,
                       tab);
}

// The tensor-core decode attention (decode.cu attn_blk_t) for query tokens
// z0 + blockIdx.z (position start + z), all of whose keys and values are in
// the cache (queries already normalized on qwen3: rope_kv_norm_b). part_*:
// per query of this launch, as attn_blk's. Grid (n_kv, <= nblk_max, nz) x 32.
template <int HD>
__device__ __forceinline__ void attn_blk_bt(const float* q, const unsigned short* kc,
                                            const unsigned short* vc, int start, int z0,
                                            int n_heads, int n_kv, float pre, int nblk_max,
                                            float* part_o, float* part_ml, const u32* rp,
                                            const u32* tab) {
    int zl = blockIdx.z, z = z0 + zl;
    u64 nh = n_heads, e = rp_base(rp, z, tab) * (u64)(n_kv * HD);
    attn_blk_t<HD>(q + (u64)z * nh * HD, kv_at(kc, e), kv_at(vc, e), rp_pos(rp, start, z) + 1,
                   blockIdx.x, n_heads, n_kv, pre, nblk_max,
                   part_o + (u64)zl * nh * nblk_max * HD, part_ml + (u64)zl * nh * nblk_max * 2,
                   QkNorm{}, Q8Kv{}, rp_tab(rp, z, tab));
}

extern "C" __global__ void __launch_bounds__(32)
attn_blk_b_64(const float* __restrict__ q, const unsigned short* __restrict__ kc,
              const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
              float pre, int nblk_max, float* __restrict__ part_o, float* __restrict__ part_ml,
              const u32* __restrict__ rp, const u32* __restrict__ tab) {
    attn_blk_bt<64>(q, kc, vc, start, z0, n_heads, n_kv, pre, nblk_max, part_o, part_ml, rp,
                    tab);
}

extern "C" __global__ void __launch_bounds__(32)
attn_blk_b_128(const float* __restrict__ q, const unsigned short* __restrict__ kc,
               const unsigned short* __restrict__ vc, int start, int z0, int n_heads, int n_kv,
               float pre, int nblk_max, float* __restrict__ part_o, float* __restrict__ part_ml,
               const u32* __restrict__ rp, const u32* __restrict__ tab) {
    attn_blk_bt<128>(q, kc, vc, start, z0, n_heads, n_kv, pre, nblk_max, part_o, part_ml, rp,
                     tab);
}

// attn_blk_b on the q8 cache (written by kv_q8_b; nothing to store here).
#define ATTN_BLK_B_Q8(NAME, HD, KG)                                                                \
    extern "C" __global__ void __launch_bounds__(32)                                               \
        NAME(const float* __restrict__ q, const signed char* __restrict__ kq,                      \
             const signed char* __restrict__ vq, const float* __restrict__ ks,                     \
             const float* __restrict__ vs, int start, int z0, int n_heads, int n_kv, float pre,    \
             int nblk_max, float* __restrict__ part_o, float* __restrict__ part_ml,                \
             const u32* __restrict__ rp, const u32* __restrict__ tab) {                            \
        int zl = blockIdx.z, z = z0 + zl;                                                          \
        u64 nh = n_heads, e = rp_base(rp, z, tab) * (u64)(n_kv * HD);                              \
        attn_blk_t<HD, KG>(q + (u64)z * nh * HD, nullptr, nullptr, rp_pos(rp, start, z) + 1,       \
                           blockIdx.x, n_heads, n_kv, pre, nblk_max,                               \
                           part_o + (u64)zl * nh * nblk_max * HD,                                  \
                           part_ml + (u64)zl * nh * nblk_max * 2, QkNorm{},                        \
                           q8_at<KG>(Q8Kv{kq, vq, ks, vs, nullptr, nullptr}, e),                   \
                           rp_tab(rp, z, tab));                                                    \
    }
ATTN_BLK_B_Q8(attn_blk_b_q8_64, 64, 64)
ATTN_BLK_B_Q8(attn_blk_b_q8_128, 128, 128)
ATTN_BLK_B_Q8(attn_blk_b_q8g32_64, 64, 32)
ATTN_BLK_B_Q8(attn_blk_b_q8g32_128, 128, 32)

// attn_cmb for the queries of an attn_blk_b launch, into act[z]. Grid (n /
// 256, nz) x 256.
extern "C" __global__ void attn_cmb_b(const float* __restrict__ part_o,
                                      const float* __restrict__ part_ml, int start, int z0, int n,
                                      int hd, int nblk_max, int want_k, u8* __restrict__ act,
                                      u64 astride, const u32* __restrict__ rp) {
    int zl = blockIdx.y, z = z0 + zl;
    u64 nh = n / hd;
    attn_cmb_dev(part_o + (u64)zl * nh * nblk_max * hd, part_ml + (u64)zl * nh * nblk_max * 2,
                 rp_pos(rp, start, z) + 1, blockIdx.x, n, hd, nblk_max, want_k, act + z * astride,
                 nullptr);
}

// The prefill attention on the tensor cores (decode.cu "Tensor-core
// attention", steps 1-10 in one block): rows r = zl * group + g (query zl <
// nb at position start + zl, head kvh * group + g) of kv head blockIdx.x,
// 16 NW per block (16 per warp), the last row tiles first (they have the
// most positions). Every block of 64 positions up to the block's last
// query's np_max goes through ST shared-memory stages (K and V tiles,
// cp.async, rows past np_max zero-filled); each warp scores its rows
// against it, and folds it into the rows with 64b < np in registers; then
// out = O / L goes to y (row zl: n_heads * HD floats, in place of q: a
// row's queries are in registers before any of y is written, and no other
// block reads them). quantize_b then fills act. Grid (n_kv, ceil(nb *
// group / (16 NW))) x 32 NW, ST * 4 * 64 * HD bytes of dynamic shared memory.
template <int HD, int NW, int ST, int KG = 0>
__device__ __forceinline__ void attn_fa_bt(const float* q, const unsigned short* __restrict__ kc,
                                           const unsigned short* __restrict__ vc, int start,
                                           int nb, int n_heads, int n_kv, float pre, float* y,
                                           Q8Kv q8 = Q8Kv{}, const u32* __restrict__ tab = nullptr) {
#if __CUDA_ARCH__ >= 800
    constexpr bool Q8 = KG > 0;
    constexpr int KGs = Q8 ? KG : 1;
    extern __shared__ __align__(16) unsigned short fa_sm[];
    const int TILE = ATT_BLK * HD;
    // q8: one f16 tile (K, then V), then per stage the raw K and V rows
    // and their scales.
    const int RAW = 2 * ATT_BLK * HD + 2 * ATT_BLK * (HD / KGs) * 4;
    auto raw8 = [&](int st) { return (signed char*)(fa_sm + TILE) + st * RAW; };
    auto rawsc = [&](int st) { return (float*)(raw8(st) + 2 * ATT_BLK * HD); };
    int kvh = blockIdx.x, cta = gridDim.y - 1 - blockIdx.y, tid = threadIdx.x;
    int lane = tid & 31, w = tid >> 5, group = n_heads / n_kv, kv_dim = n_kv * HD;
    int rows = nb * group, r0 = cta * 16 * NW;
    int ra = r0 + 16 * w + (lane >> 2), rb = ra + 8;
    int np0 = ra < rows ? start + ra / group + 1 : 0;
    int np1 = rb < rows ? start + rb / group + 1 : 0;
    // The warp's fewest and most positions, the block's most.
    int wlo = start + min(r0 + 16 * w, rows - 1) / group + 1;
    int whi = r0 + 16 * w < rows ? start + min(r0 + 16 * w + 15, rows - 1) / group + 1 : 0;
    int np_max = start + min(r0 + 16 * NW - 1, rows - 1) / group + 1;
    int nblk = (np_max + ATT_BLK - 1) / ATT_BLK;
    u64 qs = (u64)n_heads * HD;
    auto qrow = [&](int r) -> u64 { return (u64)(r / group) * qs + (kvh * group + r % group) * HD; };
    auto load = [&](int bl) {
        int st = bl % ST, t0 = bl * ATT_BLK;
        u64 rb = att_rb(tab, t0);
        if constexpr (Q8) {
            signed char* r8 = raw8(st);
            float* rs = rawsc(st);
            att_load_q8<HD, KGs>(r8, rs, q8.k, q8.ks, t0, np_max, kv_dim, kvh, tid, 32 * NW, rb);
            att_load_q8<HD, KGs>(r8 + ATT_BLK * HD, rs + ATT_BLK * HD / KGs, q8.v, q8.vs, t0,
                                 np_max, kv_dim, kvh, tid, 32 * NW, rb);
        } else {
            att_load<HD>(fa_sm + 2 * st * TILE, kc, t0, np_max, kv_dim, kvh, tid, 32 * NW, rb);
            att_load<HD>(fa_sm + (2 * st + 1) * TILE, vc, t0, np_max, kv_dim, kvh, tid, 32 * NW,
                         rb);
        }
    };
    u32 qa[HD / 16][4];
    att_qfrag<HD>(qa, ra < rows ? q + qrow(ra) : nullptr, rb < rows ? q + qrow(rb) : nullptr, pre);
    float O[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; n++) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.0f;
    float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f};
#pragma unroll
    for (int st = 0; st < ST - 1; st++) {
        if (st < nblk) load(st);
        att_commit();
    }
    for (int b = 0; b < nblk; b++) {
        if (b + ST - 1 < nblk) load(b + ST - 1);
        att_commit();
        att_wait<ST - 1>();
        __syncthreads();
        int t0 = b * ATT_BLK, st = b % ST;
        unsigned short* kt = Q8 ? fa_sm : fa_sm + 2 * st * TILE;
        unsigned short* vt = Q8 ? fa_sm : kt + TILE;
        if constexpr (Q8) {
            att_cvt_q8<HD, KGs>(kt, raw8(st), rawsc(st), tid, 32 * NW);
            __syncthreads();
        }
        bool act = t0 < whi;
        float m[2], l[2], al[2], wt[2];
        bool on[2] = {t0 < np0, t0 < np1};
        u32 pa[4][4];
        if (act) {
            float s[8][4];
            att_scores<HD>(s, qa, kt);
            if (t0 + ATT_BLK > wlo)
                att_softmax<true>(s, t0, np0, np1, m, l, pa);
            else
                att_softmax<false>(s, t0, np0, np1, m, l, pa);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                float mn;
                att_fold_w(M[i], m[i], &mn, &al[i], &wt[i]);
                if (on[i]) {
                    M[i] = mn;
                    L[i] = att_fold_o(L[i], l[i], al[i], wt[i]);
                }
            }
        }
        if constexpr (Q8) {
            __syncthreads();
            att_cvt_q8<HD, KGs>(vt, raw8(st) + ATT_BLK * HD, rawsc(st) + ATT_BLK * HD / KGs, tid,
                                32 * NW);
            __syncthreads();
        }
        if (act) {
#pragma unroll
            for (int h = 0; h < HD / 64; h++) {
                float o[8][4];
                if (h == 0)
                    att_pv<HD, 0, 64>(o, pa, vt);
                else
                    att_pv<HD, 64, 64>(o, pa, vt);
#pragma unroll
                for (int n = 0; n < 8; n++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int i = j >> 1;
                        if (on[i]) O[8 * h + n][j] = att_fold_o(O[8 * h + n][j], o[n][j], al[i], wt[i]);
                    }
                }
            }
        }
        __syncthreads();
    }
    int c = 2 * (lane & 3);
    if (ra < rows) {
        float* yr = y + qrow(ra) + c;
#pragma unroll
        for (int n = 0; n < HD / 8; n++)
            *(float2*)(yr + 8 * n) = make_float2(__fdiv_rn(O[n][0], L[0]), __fdiv_rn(O[n][1], L[0]));
    }
    if (rb < rows) {
        float* yr = y + qrow(rb) + c;
#pragma unroll
        for (int n = 0; n < HD / 8; n++)
            *(float2*)(yr + 8 * n) = make_float2(__fdiv_rn(O[n][2], L[1]), __fdiv_rn(O[n][3], L[1]));
    }
#endif
}

#define ATTN_FA_B(NAME, HD, NW, ST, MINB)                                                          \
    extern "C" __global__ void __launch_bounds__(32 * NW, MINB)                                    \
        NAME(const float* q, const unsigned short* __restrict__ kc,                                \
             const unsigned short* __restrict__ vc, int start, int nb, int n_heads, int n_kv,      \
             float pre, float* y, const u32* __restrict__ tab) {                                   \
        attn_fa_bt<HD, NW, ST>(q, kc, vc, start, nb, n_heads, n_kv, pre, y, Q8Kv{}, tab);          \
    }
// Registers (ptxas, sm_120): the hd-128 variants sit at the cap - attn_fa_b_128
// 254, attn_fa_bs_128 242, the q8 hd-128 ones 254-255 (CUDA 13.4 nvcc: no
// spills; cuda_attn_bench prints NVRTC's count of local bytes per variant,
// where a few bytes of spill have been seen). __launch_bounds__(256, 1) is
// exactly satisfiable: 256 x 255 = 65280 of the SM's 65536 registers. A
// smaller m-tile per warp for hd 128 (32 rows per 2 warps, less O and S
// live per thread) is the follow-up if a toolkit starts spilling here.
ATTN_FA_B(attn_fa_b_64, 64, 4, 3, 2)
ATTN_FA_B(attn_fa_b_128, 128, 8, 2, 1)
// Fewer rows per block, more blocks (short batches: the same bits).
ATTN_FA_B(attn_fa_bs_64, 64, 2, 2, 3)
ATTN_FA_B(attn_fa_bs_128, 128, 4, 1, 2)

// attn_fa_bt's rows (the f16 cache) with K and V on separate copies, for
// small blocks several to an SM: three tiles - K of blocks b and b + 1, V
// of block b. Per block: wait for K(b), copy V(b) during the scores and
// softmax, wait for V(b), copy K(b + 1) during P V. The same steps on the
// same tiles as attn_fa_bt; only when the bytes arrive changes. Grid as
// attn_fa_b, 3 * 64 * HD * 2 bytes of dynamic shared memory.
template <int HD, int NW>
__device__ __forceinline__ void attn_fa_kt(const float* q, const unsigned short* __restrict__ kc,
                                           const unsigned short* __restrict__ vc, int start,
                                           int nb, int n_heads, int n_kv, float pre, float* y,
                                           const u32* __restrict__ tab = nullptr) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned short fa_sm[];
    const int TILE = ATT_BLK * HD;
    int kvh = blockIdx.x, cta = gridDim.y - 1 - blockIdx.y, tid = threadIdx.x;
    int lane = tid & 31, w = tid >> 5, group = n_heads / n_kv, kv_dim = n_kv * HD;
    int rows = nb * group, r0 = cta * 16 * NW;
    int ra = r0 + 16 * w + (lane >> 2), rb = ra + 8;
    int np0 = ra < rows ? start + ra / group + 1 : 0;
    int np1 = rb < rows ? start + rb / group + 1 : 0;
    int wlo = start + min(r0 + 16 * w, rows - 1) / group + 1;
    int whi = r0 + 16 * w < rows ? start + min(r0 + 16 * w + 15, rows - 1) / group + 1 : 0;
    int np_max = start + min(r0 + 16 * NW - 1, rows - 1) / group + 1;
    int nblk = (np_max + ATT_BLK - 1) / ATT_BLK;
    u64 qs = (u64)n_heads * HD;
    auto qrow = [&](int r) -> u64 { return (u64)(r / group) * qs + (kvh * group + r % group) * HD; };
    unsigned short* vt = fa_sm + 2 * TILE;
    u32 qa[HD / 16][4];
    att_qfrag<HD>(qa, ra < rows ? q + qrow(ra) : nullptr, rb < rows ? q + qrow(rb) : nullptr, pre);
    float O[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; n++) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.0f;
    float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f};
    att_load<HD>(fa_sm, kc, 0, np_max, kv_dim, kvh, tid, 32 * NW, att_rb(tab, 0));
    att_commit();
    for (int b = 0; b < nblk; b++) {
        int t0 = b * ATT_BLK;
        unsigned short* kt = fa_sm + (b & 1) * TILE;
        att_wait<0>();
        __syncthreads(); // K(b) in; everyone past P V(b - 1): vt free
        att_load<HD>(vt, vc, t0, np_max, kv_dim, kvh, tid, 32 * NW, att_rb(tab, t0));
        att_commit();
        bool act = t0 < whi;
        float m[2], l[2], al[2], wt[2];
        bool on[2] = {t0 < np0, t0 < np1};
        u32 pa[4][4];
        if (act) {
            float s[8][4];
            att_scores<HD>(s, qa, kt);
            if (t0 + ATT_BLK > wlo)
                att_softmax<true>(s, t0, np0, np1, m, l, pa);
            else
                att_softmax<false>(s, t0, np0, np1, m, l, pa);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                float mn;
                att_fold_w(M[i], m[i], &mn, &al[i], &wt[i]);
                if (on[i]) {
                    M[i] = mn;
                    L[i] = att_fold_o(L[i], l[i], al[i], wt[i]);
                }
            }
        }
        att_wait<0>();
        __syncthreads(); // V(b) in (K(b + 1)'s tile: last read by b - 1's scores)
        if (b + 1 < nblk)
            att_load<HD>(fa_sm + ((b + 1) & 1) * TILE, kc, t0 + ATT_BLK, np_max, kv_dim, kvh, tid,
                         32 * NW, att_rb(tab, t0 + ATT_BLK));
        att_commit();
        if (act) {
#pragma unroll
            for (int h = 0; h < HD / 64; h++) {
                float o[8][4];
                if (h == 0)
                    att_pv<HD, 0, 64>(o, pa, vt);
                else
                    att_pv<HD, 64, 64>(o, pa, vt);
#pragma unroll
                for (int n = 0; n < 8; n++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int i = j >> 1;
                        if (on[i]) O[8 * h + n][j] = att_fold_o(O[8 * h + n][j], o[n][j], al[i], wt[i]);
                    }
                }
            }
        }
    }
    int c = 2 * (lane & 3);
    if (ra < rows) {
        float* yr = y + qrow(ra) + c;
#pragma unroll
        for (int n = 0; n < HD / 8; n++)
            *(float2*)(yr + 8 * n) = make_float2(__fdiv_rn(O[n][0], L[0]), __fdiv_rn(O[n][1], L[0]));
    }
    if (rb < rows) {
        float* yr = y + qrow(rb) + c;
#pragma unroll
        for (int n = 0; n < HD / 8; n++)
            *(float2*)(yr + 8 * n) = make_float2(__fdiv_rn(O[n][2], L[1]), __fdiv_rn(O[n][3], L[1]));
    }
#endif
}
#define ATTN_FA_K(NAME, HD, NW, MINB)                                                              \
    extern "C" __global__ void __launch_bounds__(32 * NW, MINB)                                    \
        NAME(const float* q, const unsigned short* __restrict__ kc,                                \
             const unsigned short* __restrict__ vc, int start, int nb, int n_heads, int n_kv,      \
             float pre, float* y, const u32* __restrict__ tab) {                                   \
        attn_fa_kt<HD, NW>(q, kc, vc, start, nb, n_heads, n_kv, pre, y, tab);                      \
    }
// Registers (ptxas, CUDA 13.4): attn_fa_k_128 254 on sm_120 / 245 on sm_89,
// attn_fa_k_64 156 / 154 (3 blocks per SM need <= 168); no spills. The host
// picks a variant per GPU and batch (batch.rs fa_pick).
ATTN_FA_K(attn_fa_k_64, 64, 4, 3)
ATTN_FA_K(attn_fa_k_128, 128, 4, 2)

// attn_fa_b on the q8 cache (written by kv_q8_b), KG values per scale.
// Dynamic shared memory: 2 * 64 * HD bytes of tile, then per stage
// (2 * 64 * HD + 2 * 64 * HD / KG * 4) bytes.
#define ATTN_FA_B_Q8(NAME, HD, NW, ST, MINB, KG)                                                   \
    extern "C" __global__ void __launch_bounds__(32 * NW, MINB)                                    \
        NAME(const float* q, const signed char* __restrict__ kq,                                   \
             const signed char* __restrict__ vq, const float* __restrict__ ks,                     \
             const float* __restrict__ vs, int start, int nb, int n_heads, int n_kv,               \
             float pre, float* y, const u32* __restrict__ tab) {                                   \
        attn_fa_bt<HD, NW, ST, KG>(q, nullptr, nullptr, start, nb, n_heads, n_kv, pre, y,          \
                                   Q8Kv{kq, vq, ks, vs, nullptr, nullptr}, tab);                   \
    }
ATTN_FA_B_Q8(attn_fa_b_q8_64, 64, 4, 3, 2, 64)
ATTN_FA_B_Q8(attn_fa_b_q8_128, 128, 8, 2, 1, 128)
ATTN_FA_B_Q8(attn_fa_bs_q8_64, 64, 2, 2, 3, 64)
ATTN_FA_B_Q8(attn_fa_bs_q8_128, 128, 4, 2, 2, 128)
ATTN_FA_B_Q8(attn_fa_b_q8g32_64, 64, 4, 3, 2, 32)
ATTN_FA_B_Q8(attn_fa_b_q8g32_128, 128, 8, 2, 1, 32)
ATTN_FA_B_Q8(attn_fa_bs_q8g32_64, 64, 2, 2, 3, 32)
ATTN_FA_B_Q8(attn_fa_bs_q8g32_128, 128, 4, 2, 2, 32)

// quantize (decode.cu) of row z of x (n floats per row) into act[z]. Grid
// (ceil(n / 256), nb) x 32.
extern "C" __global__ void quantize_b(const float* __restrict__ x, int n, int want_k,
                                      u8* __restrict__ act, u64 astride) {
    int lane = threadIdx.x & 31, blk = blockIdx.x, z = blockIdx.y;
    const float* xz = x + (u64)z * n;
    float v[8];
    for (int i = 0; i < 8; i++) {
        int e = blk * 256 + lane * 8 + i;
        v[i] = e < n ? xz[e] : 0.0f;
    }
    quant_warp(v, n, blk, act + z * astride, want_k);
}

// attn_split_b on the q8 cache (written by kv_q8_b; nothing to store here).
#define ATTN_SPLIT_B_Q8(NAME, HD, KG)                                                              \
    extern "C" __global__ void __launch_bounds__(ATT_TILE)                                         \
        NAME(const float* __restrict__ q, const signed char* __restrict__ kq,                      \
             const signed char* __restrict__ vq, const float* __restrict__ ks,                     \
             const float* __restrict__ vs, int start, int z0, int n_heads, int n_kv, float scale,  \
             int max_split, float* __restrict__ part_o, float* __restrict__ part_ml,               \
             const u32* __restrict__ rp, const u32* __restrict__ tab) {                            \
        attn_split_bt<HD, KG>(q, nullptr, nullptr, start, z0, n_heads, n_kv, scale, max_split,     \
                              part_o, part_ml, rp, tab, Q8Kv{kq, vq, ks, vs, nullptr, nullptr});   \
    }
ATTN_SPLIT_B_Q8(attn_split_b_q8_64, 64, 64)
ATTN_SPLIT_B_Q8(attn_split_b_q8_128, 128, 128)
ATTN_SPLIT_B_Q8(attn_split_b_q8g32_64, 64, 32)
ATTN_SPLIT_B_Q8(attn_split_b_q8g32_128, 128, 32)


// attn_combine for the queries of an attn_split_b launch, into act[z]: its
// ops in its order (attn_combine_dev), the split sums' loads batched 16 at
// a time. Grid (n / 256, nz) x 256.
extern "C" __global__ void attn_combine_b(const float* __restrict__ part_o,
                                          const float* __restrict__ part_ml, int start, int z0,
                                          int n, int hd, int max_split, int want_k,
                                          u8* __restrict__ act, u64 astride,
                                          const u32* __restrict__ rp) {
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

// The output of a batched product, token row z at zo = z * ostride floats
// into out: v = the row's sum (+ bias[r]); out = v (mode 0) or out += v
// (mode 1), as gemv's. Mode 2 (the up product, no bias): `bias` is the gate
// product (laid out as out) and out = silu(gate) * v, swiglu_quant_b's ops
// on the same two floats (gate_up's), for quantize_b.
// gm_put: the store with what mode 1/2 read already in p (out[zo + r] for
// mode 1, the gate value for mode 2; unused for mode 0), so a caller can
// issue all its loads before its first store.
__device__ __forceinline__ void gm_put(float y, float p, const float* bias, int mode, float* out,
                                       u64 zo, int r) {
    if (mode == 2) {
        out[zo + r] = __fmul_rn(__fdiv_rn(p, __fadd_rn(1.0f, expf(-p))), y);
        return;
    }
    float v = y;
    if (bias) v = __fadd_rn(v, bias[r]);
    out[zo + r] = mode == 1 ? __fadd_rn(p, v) : v;
}

// What gm_put needs read for output (zo, r).
__device__ __forceinline__ float gm_pre(const float* bias, int mode, const float* out, u64 zo,
                                        int r) {
    return mode == 0 ? 0.0f : (mode == 1 ? out : bias)[zo + r];
}

__device__ __forceinline__ void gm_store(float y, const float* bias, int mode, float* out, u64 zo,
                                         int r) {
    gm_put(y, gm_pre(bias, mode, out, zo, r), bias, mode, out, zo, r);
}

// 0x4B400000 = 1.5 * 2^23: for |i| < 2^22, float(i) = bits(MAGIC + i) - 1.5 * 2^23,
// exactly (one integer add, one float subtract, instead of a slow I2F).
#define GM_MAGIC 0x4B400000
#define GM_MAGICF 12582912.0f

__device__ __forceinline__ float gm_i2f(int i) {
    return __fsub_rn(__int_as_float(GM_MAGIC + i), GM_MAGICF);
}

// ---------------------------------------------------------------------------
// gemvn: the decode GEMV over T tokens at once (CUDA cores, dp4a): the same
// code (decode.cu dot_rows_t) with a token loop inside each lane's block
// step, so a row's weights are read once for T tokens. The batched
// products' path on GPUs without int8 tensor cores (sm < 80).

// Rows R * blockIdx.x .. (+R) for tokens T * blockIdx.y .. (+T, < nb):
// out[z * ostride + r] as gemv's.
template <int R, int T>
__device__ void gemvn_t(int qt, const u8* w, u64 rb, int rows, int n, const u8* act, u64 astride,
                        int nb, const float* bias, int mode, float* out, u64 ostride) {
    int r0 = R * blockIdx.x, z0 = T * blockIdx.y;
    const u8* rr[R];
#pragma unroll
    for (int r = 0; r < R; r++) rr[r] = w + (u64)min(r0 + r, rows - 1) * rb;
    float y = dot_rows_t<R, T>(qt, rr, n, act + (u64)z0 * astride, astride, nb - 1 - z0, nullptr);
    int i = threadIdx.x;
    if (i < R * T) {
        int r = r0 + i / T, z = z0 + i % T;
        if (r < rows && z < nb) gm_store(y, bias, mode, out, (u64)z * ostride, r);
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
// gemv_ref: the canonical order (decode.cu, "Row dot products") written out
// literally, one thread per output, every integer from the raw block bytes
// (quant.rs's layouts). Slow; the tests' reference for every product
// kernel. Grid (ceil(rows / 128), nb) x 128; arguments as gemvn's.
extern "C" __global__ void gemv_ref(int qt, const u8* __restrict__ w, u64 rb, int rows, int n,
                                    const u8* __restrict__ act, u64 astride, int nb,
                                    const float* __restrict__ bias, int mode,
                                    float* __restrict__ out, u64 ostride) {
    int r = blockIdx.x * blockDim.x + threadIdx.x, z = blockIdx.y;
    if (r >= rows || z >= nb) return;
    Act a = act_view(act + (u64)z * astride, n);
    const u8* row = w + (u64)r * rb;
    float acc = 0.0f;
    if (qt == QT_Q4_K) {
        for (int b = 0; b < n / 256; b++) {
            const u8* blk = row + (u64)b * 144;
            int S = 0, M = 0;
            for (int j = 0; j < 8; j++) {
                u32 sc, m;
                scale_min_k4(j, blk + 4, &sc, &m);
                int sumi = 0;
                for (int i = 0; i < 32; i++) {
                    u32 qb = blk[16 + 32 * (j >> 1) + i];
                    int q = (j & 1) ? (int)(qb >> 4) : (int)(qb & 15);
                    sumi += q * (int)a.qk[b * 256 + 32 * j + i];
                }
                S += (int)sc * sumi;
                M += (int)m * a.b32[b * 8 + j];
            }
            float d = h2f(rd_u16(blk)), dmin = h2f(rd_u16(blk + 2));
            float v = __fsub_rn(__fmul_rn(d, i2f_rn(S)), __fmul_rn(dmin, i2f_rn(M)));
            acc = __fmaf_rn(a.dk[b], v, acc);
        }
    } else if (qt == QT_Q5_K) {
        for (int b = 0; b < n / 256; b++) {
            const u8* blk = row + (u64)b * 176;
            int S = 0, M = 0;
            for (int j = 0; j < 8; j++) {
                u32 sc, m;
                scale_min_k4(j, blk + 4, &sc, &m);
                int sumi = 0;
                for (int i = 0; i < 32; i++) {
                    u32 qb = blk[48 + 32 * (j >> 1) + i];
                    int q = (j & 1) ? (int)(qb >> 4) : (int)(qb & 15);
                    q |= (int)((blk[16 + i] >> j) & 1) << 4;
                    sumi += q * (int)a.qk[b * 256 + 32 * j + i];
                }
                S += (int)sc * sumi;
                M += (int)m * a.b32[b * 8 + j];
            }
            float d = h2f(rd_u16(blk)), dmin = h2f(rd_u16(blk + 2));
            float v = __fsub_rn(__fmul_rn(d, i2f_rn(S)), __fmul_rn(dmin, i2f_rn(M)));
            acc = __fmaf_rn(a.dk[b], v, acc);
        }
    } else if (qt == QT_Q6_K) {
        for (int b = 0; b < n / 256; b++) {
            const u8* blk = row + (u64)b * 210;
            int S = 0;
            for (int j = 0; j < 16; j++) {
                int sumi = 0;
                for (int i = 0; i < 16; i++) {
                    int e = 16 * j + i, hh = e >> 7, k = (e >> 5) & 3, l = e & 31;
                    u32 ql = blk[64 * hh + l + 32 * (k & 1)];
                    u32 lo = k < 2 ? (ql & 15) : (ql >> 4);
                    u32 hi = (blk[128 + 32 * hh + l] >> (2 * k)) & 3;
                    sumi += ((int)(lo | (hi << 4)) - 32) * (int)a.qk[b * 256 + e];
                }
                S += (int)(signed char)blk[192 + j] * sumi;
            }
            acc = __fmaf_rn(__fmul_rn(a.dk[b], h2f(rd_u16(blk + 208))), i2f_rn(S), acc);
        }
    } else if (qt == QT_Q4_1 || qt == QT_Q5_1) {
        bool q5 = qt == QT_Q5_1;
        for (int b = 0; b < n / 32; b++) {
            const u8* blk = row + (u64)b * (q5 ? 24 : 20);
            u32 qh = q5 ? rd_u16(blk + 4) | (rd_u16(blk + 6) << 16) : 0;
            int s = 0, bs = 0;
            for (int i = 0; i < 32; i++) {
                u32 qs = blk[(q5 ? 8 : 4) + (i & 15)];
                int q = (int)(((i < 16 ? qs : qs >> 4) & 15) | (((qh >> i) & 1) << 4));
                s += q * (int)a.q0[b * 32 + i];
                bs += (int)a.q0[b * 32 + i];
            }
            float v = __fadd_rn(__fmul_rn(h2f(rd_u16(blk)), i2f_rn(s)),
                                __fmul_rn(h2f(rd_u16(blk + 2)), i2f_rn(bs)));
            acc = __fmaf_rn(a.d0[b], v, acc);
        }
    } else {
        bool q8 = qt == QT_Q8_0, q4 = qt == QT_Q4_0;
        for (int b = 0; b < n / 32; b++) {
            const u8* blk = row + (u64)b * (q8 ? 34 : q4 ? 18 : 22);
            int s = 0;
            u32 qh = q8 || q4 ? 0 : rd_u16(blk + 2) | (rd_u16(blk + 4) << 16);
            for (int i = 0; i < 32; i++) {
                int q;
                if (q8) {
                    q = (int)(signed char)blk[2 + i];
                } else if (q4) {
                    u32 qs = blk[2 + (i & 15)];
                    q = (int)((i < 16 ? qs : qs >> 4) & 15) - 8;
                } else {
                    u32 qs = blk[6 + (i & 15)];
                    q = (int)(((i < 16 ? qs : qs >> 4) & 15) | (((qh >> i) & 1) << 4)) - 16;
                }
                s += q * (int)a.q0[b * 32 + i];
            }
            acc = __fmaf_rn(__fmul_rn(h2f(rd_u16(blk)), a.d0[b]), i2f_rn(s), acc);
        }
    }
    gm_store(acc, bias, mode, out, (u64)z * ostride, r);
}

// ---------------------------------------------------------------------------
// Tensor-core helpers (sm_80+): async copies and int8 mma.sync.

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
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
                 "{%7,%7,%7,%7};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}
__device__ __forceinline__ void mma32(int* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, "
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

// Per weight type: SB blocks of VB values per 256-value stage, register-
// staged items per row (gemm2), bytes per block.
template <int QT>
struct GmType;
template <>
struct GmType<QT_Q4_K> {
    static const int SB = 1, VB = 256, ITEMS = 0, BYTES = 144;
};
template <>
struct GmType<QT_Q5_K> {
    static const int SB = 1, VB = 256, ITEMS = 0, BYTES = 176;
};
template <>
struct GmType<QT_Q6_K> {
    // 4 x (2 x 16 bytes of ql + 16 of qh), 1 header.
    static const int SB = 1, VB = 256, ITEMS = 5, BYTES = 210;
};
// Q6_K's prefill copy (not a GGUF type; the host makes it at load when
// VRAM allows, q6k_to_i8): per 256-value block 256 int8 values (q - 32, in
// order), the 16 int8 scales, d as f32 (h2f of the block's f16), 12 bytes
// of padding: 288 bytes, 16-byte aligned. Register-staged as Q6_K, but by
// aligned 16-byte loads with nothing to unpack: the same weight buffer
// bytes, so g2_compute's Q6_K code (the same bits) does the products.
#define QT_Q6K8 1014
#define Q6K8_BYTES 288
template <>
struct GmType<QT_Q6K8> {
    // 4 x 16 chunks of values per row, then 1 header (scales, d).
    static const int SB = 1, VB = 256, ITEMS = 20, BYTES = Q6K8_BYTES;
};
template <>
struct GmType<QT_Q8_0> {
    static const int SB = 8, VB = 32, ITEMS = 8, BYTES = 34;
};
template <>
struct GmType<QT_Q5_0> {
    static const int SB = 8, VB = 32, ITEMS = 8, BYTES = 22;
};
template <>
struct GmType<QT_Q4_0> {
    static const int SB = 8, VB = 32, ITEMS = 8, BYTES = 18;
};
template <>
struct GmType<QT_Q4_1> {
    static const int SB = 8, VB = 32, ITEMS = 8, BYTES = 20;
};
template <>
struct GmType<QT_Q5_1> {
    static const int SB = 8, VB = 32, ITEMS = 8, BYTES = 24;
};

// Signed byte j of the 16 packed in w[0..4).
__device__ __forceinline__ int gm_sbyte(const u32* w, int j) {
    return (int)(w[j >> 2] << (24 - 8 * (j & 3))) >> 24;
}
__device__ __forceinline__ int gm_ubyte(u32 w, int j) { return __byte_perm(w, 0, 0x4440 + (j & 3)); }

// ---------------------------------------------------------------------------
// gemm2_*: the batched products on int8 tensor cores (sm_80+, mma.sync), in
// the canonical order (decode.cu, "Row dot products"): per output, the
// block's integer sums S and Q4_K's M from mma tiles (and integer
// multiply-adds by the sub-block scales), then the block's few float ops
// into one running fma chain, blocks in order - what a K loop does anyway.
//
// A block computes 64 rows x BN = 32 NT tokens over all of K: 8 warps, 2
// along rows x 4 along tokens, each 32 rows x 8 NT tokens (two m16 tiles x
// NT n8 tiles). A stage is 256 values of K (a K-quant block, eight 32-value
// blocks): mma m16n8k32 per 32-value (sub-)block for Q4_K, Q8_0, Q5_0,
// m16n8k16 per 16-value Q6_K sub-block, and Q4_K's M = sum_j m_j bsum_j by
// two m16n8k16 per tile (bsum_j split into int8 halves). Activations are
// copied as they are (cp.async, ST stages), 16-byte chunks XOR-swizzled by
// row (chunk c of token z at c ^ (z & 7)) so fragment loads hit 32 banks.
// Q4_K weights too, raw (rows padded to 176 bytes, again conflict-free),
// their nibbles unpacked as fragments are built. Q6_K/Q8_0/Q5_0 blocks are
// only 2-byte aligned: loaded into registers one stage ahead and stored
// unpacked to int8 (rows swizzled as the activations).

#define G2_BM 64
#define G2_THREADS 256
#define G2_Q4K_ROW 176

// Shared memory layout (bytes) of gemm2 for type QT, BN tokens and ST
// stages: [weight buffers | ST stages of activations]. Q4_K: a weight
// buffer per stage (copied with it); the register-staged types: one, or
// two with three or more stages (stored one stage ahead).
template <int QT, int BN, int ST>
struct G2L {
    static const bool Q6 = QT == QT_Q6_K || QT == QT_Q6K8;
    // Q5_K's raw 176-byte rows need no padding: the stride is Q4_K's.
    static const bool RAW = QT == QT_Q4_K || QT == QT_Q5_K,
                      KQ = QT == QT_Q4_K || QT == QT_Q5_K || Q6;
    // Weights: Q4_K raw rows; others int8 [64][256], then Q6_K's scales
    // [64][16] and d [64], or Q8_0/Q5_0's d [64][8].
    // Q4_1 / Q5_1: d [64][8], then m [64][8].
    static const bool MIN1 = QT == QT_Q4_1 || QT == QT_Q5_1;
    static const int W = RAW ? G2_BM * G2_Q4K_ROW
                             : G2_BM * 256 + (Q6     ? G2_BM * 20
                                              : MIN1 ? G2_BM * 64
                                                     : G2_BM * 32);
    static const int NWB = RAW ? ST : (ST >= 3 ? 2 : 1);
    // Activations: values [BN][256]; Q4_K the sums of 32 [BN][8]; scales:
    // Q8_K d [BN] or Q8_0 d [BN][8].
    static const int X = BN * 256, XB = RAW ? BN * 32 : 0, XD = KQ ? BN * 4 : BN * 32;
    static const int STAGE = X + XB + XD;
    static const int SMEM = NWB * W + ST * STAGE;
};

// Byte k of w, sign-extended (prmt's sign-replicating selectors; the
// __byte_perm intrinsic masks them off).
__device__ __forceinline__ int g2_sbyte(u32 w, int k) {
    u32 r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(w), "r"((0x8880 | k) + 0x1110 * k));
    return (int)r;
}

static_assert(G2_THREADS == 4 * G2_BM, "g2_item: a Q6_K value item per thread");

// A thread's k-th register-staged item: row rr, item it. Q6_K: the four
// value items of a row to four threads, then the scales of row tid to
// threads < 64 (every thread one big item); others in order.
template <int QT>
__device__ __forceinline__ bool g2_item(int k, int* rr, int* it) {
    int tid = threadIdx.x;
    if (QT == QT_Q6_K) {
        if (k == 0) {
            *rr = tid >> 2, *it = tid & 3;
            return true;
        }
        *rr = tid, *it = 4;
        return k == 1 && tid < G2_BM;
    }
    if (QT == QT_Q6K8) {
        // Chunk it (0..16) of row rr for k < 4, then the header of row tid.
        if (k < 4) {
            int id = tid + k * G2_THREADS;
            *rr = id >> 4, *it = id & 15;
            return true;
        }
        *rr = tid, *it = 16;
        return k == 4 && tid < G2_BM;
    }
    int id = tid + k * G2_THREADS;
    *rr = id / GmType<QT>::ITEMS, *it = id % GmType<QT>::ITEMS;
    return id < G2_BM * GmType<QT>::ITEMS;
}

// Into registers: stage s's weights for the register-staged types (rows
// r0.., clamped).
#define G2_WREGS 13
template <int QT>
__device__ __forceinline__ void g2_load_w(const u8* w, u64 rb, int rows, int r0, int nblk, int s,
                                          u32 (*reg)[G2_WREGS]) {
    const int SB = GmType<QT>::SB;
    const int NI = (G2_BM * GmType<QT>::ITEMS + G2_THREADS - 1) / G2_THREADS;
#pragma unroll
    for (int k = 0; k < NI; k++) {
        int rr, it;
        if (!g2_item<QT>(k, &rr, &it)) break;
        const u8* row = w + (u64)min(r0 + rr, rows - 1) * rb;
        u32* o = reg[k];
        if (QT == QT_Q6K8) {
            const u8* blk = row + (u64)s * Q6K8_BYTES;
            uint4 v = __ldg((const uint4*)(blk + (it < 16 ? 16 * it : 256)));
            o[0] = v.x, o[1] = v.y, o[2] = v.z, o[3] = v.w;
            if (it == 16) o[4] = __ldg((const u32*)(blk + 272));
        } else if (QT == QT_Q6_K) {
            const u8* blk = row + (u64)s * 210;
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
            const u8* blk = row + (u64)q * GmType<QT>::BYTES;
            if (QT == QT_Q4_1 || QT == QT_Q5_1) {
                o[0] = __ldg((const u32*)blk); // d, m (blocks 4-byte aligned)
                if (QT == QT_Q4_1) ld_a2<4>(blk + 4, o + 1); // qs
                else ld_a2<5>(blk + 4, o + 1);               // qh, qs
                continue;
            }
            o[0] = rd_u16(blk);
            if (QT == QT_Q8_0) ld_a2<8>(blk + 2, o + 1);
            else if (QT == QT_Q4_0) ld_a2<4>(blk + 2, o + 1); // qs
            else ld_a2<5>(blk + 2, o + 1);                    // qh, qs
        }
    }
}

// Row r's 16-byte chunk c of int8 weights (swizzled).
__device__ __forceinline__ uint4* g2_w8(u8* wb, int r, int c) {
    return (uint4*)(wb + r * 256 + 16 * (c ^ (r & 7)));
}

// The registers of g2_load_w, unpacked into the weight buffer.
template <int QT>
__device__ __forceinline__ void g2_store_w(u8* wb, int nblk, int s, u32 (*reg)[G2_WREGS]) {
    const int SB = GmType<QT>::SB;
    const int NI = (G2_BM * GmType<QT>::ITEMS + G2_THREADS - 1) / G2_THREADS;
    u8* meta = wb + G2_BM * 256;
#pragma unroll
    for (int k = 0; k < NI; k++) {
        int rr, it;
        if (!g2_item<QT>(k, &rr, &it)) break;
        const u32* o = reg[k];
        if (QT == QT_Q6K8) {
            // As Q6_K's below: values chunk it of row rr (swizzled), the
            // scales, d (f32 bits of the same h2f).
            if (it < 16) {
                *g2_w8(wb, rr, it) = make_uint4(o[0], o[1], o[2], o[3]);
            } else {
                *(uint4*)(meta + 16 * rr) = make_uint4(o[0], o[1], o[2], o[3]);
                ((u32*)(meta + 16 * G2_BM))[rr] = o[4];
            }
        } else if (QT == QT_Q6_K) {
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
                    *g2_w8(wb, rr, 8 * h + 2 * k + lh) = make_uint4(v[0], v[1], v[2], v[3]);
                }
            } else {
                *(uint4*)(meta + 16 * rr) = make_uint4(o[0], o[1], o[2], o[3]);
                ((float*)(meta + 16 * G2_BM))[rr] = h2f(o[4]);
            }
        } else {
            if (s * SB + it >= nblk) continue;
            ((float*)meta)[rr * 8 + it] = h2f(o[0] & 0xffff);
            if (QT == QT_Q4_1 || QT == QT_Q5_1) {
                ((float*)meta)[G2_BM * 8 + rr * 8 + it] = h2f(o[0] >> 16);
                // Unsigned values: low nibbles, then high (bit 4 from qh).
                u32 v[8];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const u32 qs = o[(QT == QT_Q5_1 ? 2 : 1) + u];
                    u32 qh = QT == QT_Q5_1 ? o[1] : 0;
                    v[u] = (qs & 0x0f0f0f0f) | spread4((qh >> (4 * u)) & 0xf);
                    v[4 + u] = ((qs >> 4) & 0x0f0f0f0f) | spread4((qh >> (16 + 4 * u)) & 0xf);
                }
                *g2_w8(wb, rr, 2 * it) = make_uint4(v[0], v[1], v[2], v[3]);
                *g2_w8(wb, rr, 2 * it + 1) = make_uint4(v[4], v[5], v[6], v[7]);
            } else if (QT == QT_Q8_0) {
                *g2_w8(wb, rr, 2 * it) = make_uint4(o[1], o[2], o[3], o[4]);
                *g2_w8(wb, rr, 2 * it + 1) = make_uint4(o[5], o[6], o[7], o[8]);
            } else if (QT == QT_Q4_0) {
                u32 v[8];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    v[u] = __vsub4(o[1 + u] & 0x0f0f0f0f, 0x08080808);
                    v[4 + u] = __vsub4((o[1 + u] >> 4) & 0x0f0f0f0f, 0x08080808);
                }
                *g2_w8(wb, rr, 2 * it) = make_uint4(v[0], v[1], v[2], v[3]);
                *g2_w8(wb, rr, 2 * it + 1) = make_uint4(v[4], v[5], v[6], v[7]);
            } else {
                u32 qh = o[1], v[8];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    v[u] = __vsub4((o[2 + u] & 0x0f0f0f0f) | spread4((qh >> (4 * u)) & 0xf),
                                   0x10101010);
                    v[4 + u] = __vsub4(((o[2 + u] >> 4) & 0x0f0f0f0f) |
                                           spread4((qh >> (16 + 4 * u)) & 0xf),
                                       0x10101010);
                }
                *g2_w8(wb, rr, 2 * it) = make_uint4(v[0], v[1], v[2], v[3]);
                *g2_w8(wb, rr, 2 * it + 1) = make_uint4(v[4], v[5], v[6], v[7]);
            }
        }
    }
}

// Start the copies of stage s: Q4_K weights (into wb), the activations of
// tokens z0.. (clamped to nb - 1) into st.
template <int QT, int BN, int ST, int GU = 0>
__device__ __forceinline__ void g2_copy(u8* wb, u8* st, const u8* w, u64 rb, int rows, int r0,
                                        const u8* act, u64 astride, int n, int nb, int z0,
                                        int nblk, int s, const u8* w2 = nullptr) {
#if __CUDA_ARCH__ >= 800
    typedef G2L<QT, BN, ST> L;
    u64 np = act_np(n);
    int tid = threadIdx.x;
    if (L::RAW) {
        // 16-byte chunks per block: 9 (Q4_K, 144 bytes), 11 (Q5_K, 176).
        const int CH = GmType<QT>::BYTES / 16;
        for (int id = tid; id < G2_BM * CH; id += G2_THREADS) {
            int rr = id / CH, ch = id - CH * rr;
            // GU: rows 0..32 of the tile from w (gate), 32.. the same rows
            // of w2 (up).
            const u8* row = GU ? (rr < 32 ? w : w2) + (u64)min(r0 + (rr & 31), rows - 1) * rb
                               : w + (u64)min(r0 + rr, rows - 1) * rb;
            cp16(wb + rr * G2_Q4K_ROW + 16 * ch, row + (u64)s * GmType<QT>::BYTES + 16 * ch);
        }
    }
    u8* xs = st;
    u8* xb = st + L::X;
    u8* xd = st + L::X + L::XB;
    // Values: K-quants the Q8_K view at 0, others the Q8_0 view at np.
    u64 voff = (L::KQ ? 0 : np) + (u64)s * 256;
    int c = tid & 15;
    bool cvalid = L::KQ || s * 8 + (c >> 1) < nblk;
#pragma unroll
    for (int k = 0; k < BN * 16 / G2_THREADS; k++) {
        int tk = (tid >> 4) + k * (G2_THREADS / 16);
        const u8* a = act + (u64)min(z0 + tk, nb - 1) * astride;
        if (cvalid) cp16(xs + tk * 256 + 16 * (c ^ (tk & 7)), a + voff + 16 * c);
    }
    if (L::RAW) {
        int j = tid & 7;
#pragma unroll
        for (int k = 0; k < BN * 8 / G2_THREADS; k++) {
            int tk = (tid >> 3) + k * (G2_THREADS / 8);
            const u8* a = act + (u64)min(z0 + tk, nb - 1) * astride;
            cp4(xb + tk * 32 + 4 * j, a + 2 * np + np / 64 + 4 * ((u64)s * 8 + j));
        }
    }
    if (L::KQ) {
        for (int tk = tid; tk < BN; tk += G2_THREADS) {
            const u8* a = act + (u64)min(z0 + tk, nb - 1) * astride;
            cp4(xd + 4 * tk, a + 2 * np + 4 * (u64)s);
        }
    } else {
        int e = tid & 7;
        if (s * 8 + e < nblk) {
#pragma unroll
            for (int k = 0; k < BN * 8 / G2_THREADS; k++) {
                int tk = (tid >> 3) + k * (G2_THREADS / 8);
                const u8* a = act + (u64)min(z0 + tk, nb - 1) * astride;
                cp4(xd + tk * 32 + 4 * e, a + 2 * np + np / 64 + np / 8 + 4 * ((u64)s * 8 + e));
            }
        }
    }
#endif
}

__device__ __forceinline__ u32 g2_ld(const u8* p) { return *(const u32*)p; }

// One stage's products into the chains acc[mi][f][e] (row 32 wr + 16 mi +
// g + 8 (e >> 1), token 8 NT wc + 8 f + 2t + (e & 1)).
template <int QT, int NT, int ST>
__device__ __forceinline__ void g2_compute(const u8* wb, const u8* st, int nblk, int s, int wr,
                                           int wc, float (*acc)[NT][4]) {
#if __CUDA_ARCH__ >= 800
    typedef G2L<QT, 32 * NT, ST> L;
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const u8* xs = st;
    const u8* xb = st + L::X;
    const float* xd = (const float*)(st + L::X + L::XB);
    // B fragment rows: token tb[f] (tb & 7 == g), word t of a chunk.
    const u8* xrow[NT];
#pragma unroll
    for (int f = 0; f < NT; f++) xrow[f] = xs + (8 * NT * wc + 8 * f + g) * 256 + 4 * t;
#define G2_X(f, c) g2_ld(xrow[f] + 16 * ((c) ^ g))
    // The thread's output tokens (within the block).
    int tk0 = 8 * NT * wc + 2 * t;
    if constexpr (QT == QT_Q4_K || QT == QT_Q5_K) {
        // Q5_K: qs after the 32 qh bytes; qh bytes 4t.. and 16 + 4t.. hold
        // bit 4 of the values at the qs bytes this thread reads.
        const bool Q5 = QT == QT_Q5_K;
        const int QS = Q5 ? 48 : 16;
        // Rows ra = 32 wr + 16 mi + 8 h + g: header d, dmin, scales, mins.
        float d[2][2], dmin[2][2];
        u32 scw[2][2][2], mw[2][2][2];
        const u8* wrow[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                wrow[mi][h] = wb + (32 * wr + 16 * mi + 8 * h + g) * G2_Q4K_ROW;
                uint4 hd = *(const uint4*)wrow[mi][h];
                d[mi][h] = h2f(hd.x & 0xffff);
                dmin[mi][h] = h2f(hd.x >> 16);
                scw[mi][h][0] = hd.y & 0x3f3f3f3f;
                scw[mi][h][1] = (hd.w & 0x0f0f0f0f) | ((hd.y >> 2) & 0x30303030);
                mw[mi][h][0] = hd.z & 0x3f3f3f3f;
                mw[mi][h][1] = ((hd.w >> 4) & 0x0f0f0f0f) | ((hd.z >> 2) & 0x30303030);
            }
        u32 qhw[2][2][2];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int k = 0; k < 2; k++)
                    qhw[mi][h][k] = Q5 ? g2_ld(wrow[mi][h] + 16 + 16 * k + 4 * t) : 0;
        int S[2][NT][4], M[2][NT][4];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int f = 0; f < NT; f++)
#pragma unroll
                for (int e = 0; e < 4; e++) S[mi][f][e] = 0;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            // B: sub-blocks 2c (chunks 4c, 4c + 1) and 2c + 1 (4c + 2, 4c + 3).
            u32 bf[NT][4];
#pragma unroll
            for (int f = 0; f < NT; f++)
#pragma unroll
                for (int i = 0; i < 4; i++) bf[f][i] = G2_X(f, 4 * c + i);
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                // qs bytes 32c + 4t.. (chunk 1 + 2c) and 32c + 16 + 4t.. (2 + 2c).
                u32 q[2][2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    q[h][0] = g2_ld(wrow[mi][h] + QS + 32 * c + 4 * t);
                    q[h][1] = g2_ld(wrow[mi][h] + QS + 32 * c + 16 + 4 * t);
                }
                int sc1[2], sc2[2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    sc1[h] = gm_ubyte(scw[mi][h][c >> 1], 2 * c);
                    sc2[h] = gm_ubyte(scw[mi][h][c >> 1], 2 * c + 1);
                }
                u32 lo0 = q[0][0] & 0x0f0f0f0f, lo1 = q[1][0] & 0x0f0f0f0f;
                u32 lo2 = q[0][1] & 0x0f0f0f0f, lo3 = q[1][1] & 0x0f0f0f0f;
                u32 hi0 = (q[0][0] >> 4) & 0x0f0f0f0f, hi1 = (q[1][0] >> 4) & 0x0f0f0f0f;
                u32 hi2 = (q[0][1] >> 4) & 0x0f0f0f0f, hi3 = (q[1][1] >> 4) & 0x0f0f0f0f;
                if (Q5) {
#define G2_H5(w, sh) ((((w) >> (sh)) & 0x01010101) << 4)
                    lo0 |= G2_H5(qhw[mi][0][0], 2 * c), lo1 |= G2_H5(qhw[mi][1][0], 2 * c);
                    lo2 |= G2_H5(qhw[mi][0][1], 2 * c), lo3 |= G2_H5(qhw[mi][1][1], 2 * c);
                    hi0 |= G2_H5(qhw[mi][0][0], 2 * c + 1), hi1 |= G2_H5(qhw[mi][1][0], 2 * c + 1);
                    hi2 |= G2_H5(qhw[mi][0][1], 2 * c + 1), hi3 |= G2_H5(qhw[mi][1][1], 2 * c + 1);
#undef G2_H5
                }
#pragma unroll
                for (int f = 0; f < NT; f++) {
                    int d1[4], d2[4];
                    mma32(d1, lo0, lo1, lo2, lo3, bf[f][0], bf[f][1]);
                    mma32(d2, hi0, hi1, hi2, hi3, bf[f][2], bf[f][3]);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        int h = e >> 1;
                        S[mi][f][e] += sc1[h] * d1[e] + sc2[h] * d2[e];
                    }
                }
            }
        }
        {
            // M by mma: sum_j m_j bsum_j with bsum_j = lo_j + 64 hi_j (lo in
            // [0, 64), hi in [-64, 64): int8). B (k16) of token g: k < 8 the
            // lo_j, k >= 8 the hi_j; A rows: the mins at k < 8 (lo product)
            // or k >= 8 (hi product), zeros elsewhere.
            u32 bm[NT];
#pragma unroll
            for (int f = 0; f < NT; f++) {
                int4 v = *(const int4*)(xb + (8 * NT * wc + 8 * f + g) * 32 + 16 * (t & 1));
                int sh = t < 2 ? 0 : 6;
                u32 p01 = __byte_perm((u32)(v.x >> sh), (u32)(v.y >> sh), 0x0040);
                u32 p23 = __byte_perm((u32)(v.z >> sh), (u32)(v.w >> sh), 0x0040);
                bm[f] = __byte_perm(p01, p23, 0x5410) & (t < 2 ? 0x3f3f3f3fu : 0xffffffffu);
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                u32 m0 = (t & 1) ? mw[mi][0][1] : mw[mi][0][0];
                u32 m1 = (t & 1) ? mw[mi][1][1] : mw[mi][1][0];
                u32 alo0 = t < 2 ? m0 : 0, alo1 = t < 2 ? m1 : 0;
                u32 ahi0 = t < 2 ? 0 : m0, ahi1 = t < 2 ? 0 : m1;
#pragma unroll
                for (int f = 0; f < NT; f++) {
                    int ml[4], mh[4];
                    mma16(ml, alo0, alo1, bm[f]);
                    mma16(mh, ahi0, ahi1, bm[f]);
#pragma unroll
                    for (int e = 0; e < 4; e++) M[mi][f][e] = ml[e] + mh[e] * 64;
                }
            }
        }
        float dk[NT][2];
#pragma unroll
        for (int f = 0; f < NT; f++)
#pragma unroll
            for (int e = 0; e < 2; e++) dk[f][e] = xd[tk0 + 8 * f + e];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int f = 0; f < NT; f++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int h = e >> 1;
                    float v = __fsub_rn(__fmul_rn(d[mi][h], i2f_rn(S[mi][f][e])),
                                        __fmul_rn(dmin[mi][h], i2f_rn(M[mi][f][e])));
                    acc[mi][f][e] = __fmaf_rn(dk[f][e & 1], v, acc[mi][f][e]);
                }
    } else if constexpr (L::Q6) {
        const u8* meta = wb + G2_BM * 256;
        // Rows 32 wr + 16 mi + 8 h + g.
        const u8* wrow[2][2];
        u32 scw[2][2][4];
        float dw[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int r = 32 * wr + 16 * mi + 8 * h + g;
                wrow[mi][h] = wb + r * 256 + 4 * t;
                uint4 sc = *(const uint4*)(meta + 16 * r);
                scw[mi][h][0] = sc.x, scw[mi][h][1] = sc.y, scw[mi][h][2] = sc.z,
                scw[mi][h][3] = sc.w;
                dw[mi][h] = ((const float*)(meta + 16 * G2_BM))[r];
            }
        int S[2][NT][4];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int f = 0; f < NT; f++)
#pragma unroll
                for (int e = 0; e < 4; e++) S[mi][f][e] = 0;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            u32 bf[NT];
#pragma unroll
            for (int f = 0; f < NT; f++) bf[f] = G2_X(f, j);
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                u32 a0 = g2_ld(wrow[mi][0] + 16 * (j ^ g)), a1 = g2_ld(wrow[mi][1] + 16 * (j ^ g));
                int sa = (int)g2_sbyte(scw[mi][0][j >> 2], j & 3);
                int sb = (int)g2_sbyte(scw[mi][1][j >> 2], j & 3);
#pragma unroll
                for (int f = 0; f < NT; f++) {
                    int dd[4];
                    mma16(dd, a0, a1, bf[f]);
                    S[mi][f][0] += sa * dd[0];
                    S[mi][f][1] += sa * dd[1];
                    S[mi][f][2] += sb * dd[2];
                    S[mi][f][3] += sb * dd[3];
                }
            }
        }
        float dk[NT][2];
#pragma unroll
        for (int f = 0; f < NT; f++)
#pragma unroll
            for (int e = 0; e < 2; e++) dk[f][e] = xd[tk0 + 8 * f + e];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int f = 0; f < NT; f++)
#pragma unroll
                for (int e = 0; e < 4; e++)
                    acc[mi][f][e] = __fmaf_rn(__fmul_rn(dk[f][e & 1], dw[mi][e >> 1]),
                                              i2f_rn(S[mi][f][e]), acc[mi][f][e]);
    } else {
        const float* wd = (const float*)(wb + G2_BM * 256);
        const u8* wrow[2][2];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int h = 0; h < 2; h++)
                wrow[mi][h] = wb + (32 * wr + 16 * mi + 8 * h + g) * 256 + 4 * t;
#pragma unroll
        for (int e8 = 0; e8 < 8; e8++) {
            if (s * 8 + e8 >= nblk) break;
            u32 bf[NT][2];
#pragma unroll
            for (int f = 0; f < NT; f++) {
                bf[f][0] = G2_X(f, 2 * e8);
                bf[f][1] = G2_X(f, 2 * e8 + 1);
            }
            float d0[NT][2];
#pragma unroll
            for (int f = 0; f < NT; f++)
#pragma unroll
                for (int e = 0; e < 2; e++) d0[f][e] = xd[(tk0 + 8 * f + e) * 8 + e8];
#pragma unroll
            // Q4_1 / Q5_1: each token's sum of the block's activations
            // (an all-ones A: the same in every row).
            int bsum[NT][2];
            if constexpr (L::MIN1) {
#pragma unroll
                for (int f = 0; f < NT; f++) {
                    int dd[4];
                    const u32 one = 0x01010101u;
                    mma32(dd, one, one, one, one, bf[f][0], bf[f][1]);
                    bsum[f][0] = dd[0], bsum[f][1] = dd[1];
                }
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                u32 a0 = g2_ld(wrow[mi][0] + 16 * ((2 * e8) ^ g));
                u32 a1 = g2_ld(wrow[mi][1] + 16 * ((2 * e8) ^ g));
                u32 a2 = g2_ld(wrow[mi][0] + 16 * ((2 * e8 + 1) ^ g));
                u32 a3 = g2_ld(wrow[mi][1] + 16 * ((2 * e8 + 1) ^ g));
                float dwr[2], mwr[2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    int r = 32 * wr + 16 * mi + 8 * h + g;
                    dwr[h] = wd[r * 8 + e8];
                    mwr[h] = L::MIN1 ? wd[G2_BM * 8 + r * 8 + e8] : 0.0f;
                }
#pragma unroll
                for (int f = 0; f < NT; f++) {
                    int dd[4];
                    mma32(dd, a0, a1, a2, a3, bf[f][0], bf[f][1]);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        if constexpr (L::MIN1) {
                            float v = __fadd_rn(__fmul_rn(dwr[e >> 1], gm_i2f(dd[e])),
                                                __fmul_rn(mwr[e >> 1], gm_i2f(bsum[f][e & 1])));
                            acc[mi][f][e] = __fmaf_rn(d0[f][e & 1], v, acc[mi][f][e]);
                        } else {
                            acc[mi][f][e] = __fmaf_rn(__fmul_rn(dwr[e >> 1], d0[f][e & 1]),
                                                      gm_i2f(dd[e]), acc[mi][f][e]);
                        }
                    }
                }
            }
        }
    }
#undef G2_X
#endif
}

template <int QT, int NT, int ST, int GU = 0>
__device__ __forceinline__ void gemm2_t(const u8* w, u64 rb, int rows, int n, const u8* act,
                                        u64 astride, int nb, const float* bias, int mode,
                                        float* out, u64 ostride, const u8* w2 = nullptr) {
#if __CUDA_ARCH__ >= 800
    const int BN = 32 * NT;
    typedef G2L<QT, BN, ST> L;
    extern __shared__ uint4 g2_smem[];
    u8* sm = (u8*)g2_smem;
    u32 dsz;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz < L::SMEM) __trap(); // the host's size out of date
    u8* wbuf = sm;
    u8* stg = sm + L::NWB * L::W;
    int nblk = n / (L::KQ ? 256 : 32), nst = (n + 255) / 256;
    int r0 = blockIdx.y * (GU ? 32 : G2_BM), z0 = blockIdx.x * BN;
    int warp = threadIdx.x >> 5, wr = warp & 1, wc = warp >> 1;
    float acc[2][NT][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int f = 0; f < NT; f++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mi][f][e] = 0.0f;
    const int NI = L::RAW ? 1 : (G2_BM * GmType<QT>::ITEMS + G2_THREADS - 1) / G2_THREADS;
    u32 reg[NI][G2_WREGS];
    // Stage s's buffers.
    auto wsl = [&](int s) { return wbuf + (L::RAW ? s % ST : (L::NWB == 2 ? s & 1 : 0)) * L::W; };
    auto xsl = [&](int s) { return stg + (s % ST) * L::STAGE; };
    auto copy = [&](int s) {
        g2_copy<QT, BN, ST, GU>(wsl(s), xsl(s), w, rb, rows, r0, act, astride, n, nb, z0, nblk, s,
                                w2);
    };
    if constexpr (!L::RAW) {
        g2_load_w<QT>(w, rb, rows, r0, nblk, 0, reg);
        g2_store_w<QT>(wsl(0), nblk, 0, reg);
        if (nst > 1) g2_load_w<QT>(w, rb, rows, r0, nblk, 1, reg);
    }
    if constexpr (ST == 2) {
        // Two stages: copy s + 1 while s is computed; two barriers a stage.
        copy(0);
        cp_commit();
        for (int s = 0; s < nst; s++) {
            if (s + 1 < nst) {
                copy(s + 1);
                cp_commit();
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncthreads();
            g2_compute<QT, NT, ST>(wsl(s), xsl(s), nblk, s, wr, wc, acc);
            __syncthreads();
            if constexpr (!L::RAW) {
                if (s + 1 < nst) {
                    g2_store_w<QT>(wsl(s + 1), nblk, s + 1, reg);
                    if (s + 2 < nst) g2_load_w<QT>(w, rb, rows, r0, nblk, s + 2, reg);
                }
            }
        }
    } else {
        // ST stages, ST - 1 copies in flight, one barrier a stage: after it,
        // every thread is done with stage s - 1, whose buffers (and the
        // weight buffer s + 1 shares with it) are refilled.
#pragma unroll
        for (int p = 0; p < ST - 1; p++) {
            if (p < nst) copy(p);
            cp_commit();
        }
        for (int s = 0; s < nst; s++) {
            cp_wait<ST - 2>();
            __syncthreads();
            if (s + ST - 1 < nst) copy(s + ST - 1);
            cp_commit();
            if constexpr (!L::RAW) {
                if (s + 1 < nst) {
                    g2_store_w<QT>(wsl(s + 1), nblk, s + 1, reg);
                    if (s + 2 < nst) g2_load_w<QT>(w, rb, rows, r0, nblk, s + 2, reg);
                }
            }
            g2_compute<QT, NT, ST>(wsl(s), xsl(s), nblk, s, wr, wc, acc);
        }
    }
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if constexpr (GU) {
        // The gate warps' sums (wr 0) to the up warps holding the same row
        // and token (wr 1) through shared memory (the stage buffers, free
        // once every warp is past its last stage); out = silu(gate) * up,
        // gm_store mode 2's ops on the same two floats.
        float* xch = (float*)sm;
        const int ID = (threadIdx.x >> 6) * 32 + lane;
        __syncthreads();
        if (wr == 0) {
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
#pragma unroll
                for (int f = 0; f < NT; f++)
#pragma unroll
                    for (int e = 0; e < 4; e++) xch[((mi * NT + f) * 4 + e) * 128 + ID] = acc[mi][f][e];
        }
        __syncthreads();
        if (wr == 1) {
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
#pragma unroll
                for (int f = 0; f < NT; f++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        int r = r0 + 16 * mi + 8 * (e >> 1) + g;
                        int z = z0 + 8 * NT * wc + 8 * f + 2 * t + (e & 1);
                        float gv = xch[((mi * NT + f) * 4 + e) * 128 + ID];
                        if (r < rows && z < nb)
                            out[(u64)z * ostride + r] =
                                __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), acc[mi][f][e]);
                    }
        }
        return;
    }
    // Every load the stores need first (out for mode 1, the gate product
    // for mode 2), then the stores: the 2 * NT * 4 loads in flight at once,
    // not one dependent load per store (a one-CTA-per-SM tail hides
    // nothing). gm_store's ops on the same floats.
    float pre[2][NT][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int f = 0; f < NT; f++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = r0 + 32 * wr + 16 * mi + 8 * (e >> 1) + g;
                int z = z0 + 8 * NT * wc + 8 * f + 2 * t + (e & 1);
                pre[mi][f][e] = 0.0f;
                if (mode != 0 && r < rows && z < nb)
                    pre[mi][f][e] = gm_pre(bias, mode, out, (u64)z * ostride, r);
            }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int f = 0; f < NT; f++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = r0 + 32 * wr + 16 * mi + 8 * (e >> 1) + g;
                int z = z0 + 8 * NT * wc + 8 * f + 2 * t + (e & 1);
                if (r < rows && z < nb)
                    gm_put(acc[mi][f][e], pre[mi][f][e], bias, mode, out, (u64)z * ostride, r);
            }
#else
    __trap();
#endif
}

#define GEMM2(NAME, QT, NT, ST)                                                                   \
    extern "C" __global__ void __launch_bounds__(G2_THREADS, 1) NAME(                             \
        const u8* __restrict__ w, u64 rb, int rows, int n, const u8* __restrict__ act,           \
        u64 astride, int nb, const float* __restrict__ bias, int mode, float* __restrict__ out,  \
        u64 ostride) {                                                                            \
        gemm2_t<QT, NT, ST>(w, rb, rows, n, act, astride, nb, bias, mode, out, ostride);          \
    }
// gemm2_*: 64 rows x 128 tokens, two stages; gemm2s_*: 64 x 64, three
// stages, for matrices too small to fill the GPU with the big tiles
// (qwen2/batch.rs picks).
GEMM2(gemm2_q4_k, QT_Q4_K, 4, 2)
GEMM2(gemm2_q6_k, QT_Q6_K, 4, 2)
GEMM2(gemm2_q8_0, QT_Q8_0, 4, 2)
GEMM2(gemm2_q5_0, QT_Q5_0, 4, 2)
GEMM2(gemm2_q5_k, QT_Q5_K, 4, 2)
GEMM2(gemm2_q4_0, QT_Q4_0, 4, 2)
GEMM2(gemm2_q4_1, QT_Q4_1, 4, 2)
GEMM2(gemm2_q5_1, QT_Q5_1, 4, 2)
GEMM2(gemm2s_q4_k, QT_Q4_K, 2, 3)
GEMM2(gemm2s_q6_k, QT_Q6_K, 2, 3)
GEMM2(gemm2s_q8_0, QT_Q8_0, 2, 3)
GEMM2(gemm2s_q5_0, QT_Q5_0, 2, 3)
GEMM2(gemm2s_q5_k, QT_Q5_K, 2, 3)
GEMM2(gemm2s_q4_0, QT_Q4_0, 2, 3)
GEMM2(gemm2s_q4_1, QT_Q4_1, 2, 3)
GEMM2(gemm2s_q5_1, QT_Q5_1, 2, 3)

// gemm2s_q4_k_gu: the gate and up products of a layer in one launch and
// out = silu(gate) * up (gate_up's ops; swiglu_quant_b's floats): a block
// takes rows 32 y.. of both (gate in its row-warps 0, up in 1), so the
// gate product never goes to memory. Grid (ceil(nb / 64), ceil(rows / 32)).
extern "C" __global__ void __launch_bounds__(G2_THREADS, 1)
    gemm2s_q4_k_gu(const u8* __restrict__ wg, const u8* __restrict__ wu, u64 rb, int rows, int n,
                   const u8* __restrict__ act, u64 astride, int nb, float* __restrict__ out,
                   u64 ostride) {
    gemm2_t<QT_Q4_K, 2, 3, 1>(wg, rb, rows, n, act, astride, nb, nullptr, 0, out, ostride, wu);
}

GEMM2(gemm2_q6k8, QT_Q6K8, 4, 2)
GEMM2(gemm2s_q6k8, QT_Q6K8, 2, 3)

// Q6_K rows (rb bytes each, nblk blocks) to their Q6K8 copy (nblk * 288
// bytes a row): values (q - 32 as int8, gemv_ref's q), scales, d = h2f.
// Grid (nblk, rows) x 64: four values a thread.
extern "C" __global__ void q6k_to_i8(const u8* __restrict__ w, u64 rb, int nblk,
                                     u8* __restrict__ out) {
    int b = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const u8* blk = w + (u64)r * rb + (u64)b * 210;
    u8* o = out + ((u64)r * nblk + b) * Q6K8_BYTES;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int e = 4 * t + i, hh = e >> 7, k = (e >> 5) & 3, l = e & 31;
        u32 ql = blk[64 * hh + l + 32 * (k & 1)];
        u32 lo = k < 2 ? (ql & 15) : (ql >> 4);
        u32 hi = (blk[128 + 32 * hh + l] >> (2 * k)) & 3;
        o[e] = (u8)((int)(lo | (hi << 4)) - 32);
    }
    if (t < 16) o[256 + t] = blk[192 + t];
    if (t == 0) *(float*)(o + 272) = h2f(rd_u16(blk + 208));
    if (t >= 16 && t < 28) o[276 + t - 16] = 0;
}

// ---------------------------------------------------------------------------
// gemvm_*: small batches (nb <= 8 NF) on tensor cores, weight-read-once: the
// verify pass of a draft. A block: 16 rows x 8 NF tokens, 4 warps splitting
// K (warp w takes blocks w, w + 4, ...), weights loaded straight from global
// memory (the next block's while one is used). Each block's value per
// output (and its row and token scales) goes to shared memory, GVM_CH blocks
// at a time; then each thread runs the fma chains of its outputs over them.

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
struct GvW<QT_Q5_K> {
    uint4 h[2];
    u32 q[2][8];  // qs word (32 c + 16 hf + 4 t) / 4, piece p = 2c + hf
    u32 qh[2][2]; // qh words (16 hf + 4 t) / 4: bit 4 of piece (c, hf)'s values in bit 2c / 2c + 1
    __device__ __forceinline__ void load(const u8* const* rows, int b, int t) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const u8* blk = rows[r] + (u64)b * 176;
            h[r] = __ldg((const uint4*)blk);
#pragma unroll
            for (int k = 0; k < 2; k++) qh[r][k] = __ldg((const u32*)(blk + 16 + 16 * k) + t);
#pragma unroll
            for (int p = 0; p < 8; p++) q[r][p] = __ldg((const u32*)(blk + 48 + 16 * p) + t);
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
            } else if constexpr (QT == QT_Q4_0) {
                const u8* blk = rows[r] + (u64)b * 18;
                u32 qs = ldw_a2(blk + 2 + 4 * t);
                // Values 4t.. (low nibbles) and 16 + 4t.. (high), minus 8.
                a[r][0] = __vsub4(qs & 0x0f0f0f0f, 0x08080808);
                a[r][1] = __vsub4((qs >> 4) & 0x0f0f0f0f, 0x08080808);
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
template <>
struct GvW<QT_Q4_0> : GvW32<QT_Q4_0> {};

// Blocks per chunk of gemvm's (block, output) values in shared memory.
#define GVM_CH 8

template <int QT, int NF>
__device__ __forceinline__ void gemvm_t(const u8* w, u64 rb, int rows, int n, const u8* act,
                                        u64 astride, int nb, const float* bias, int mode,
                                        float* out, u64 ostride) {
#if __CUDA_ARCH__ >= 800
    const int KQ = QT == QT_Q4_K || QT == QT_Q5_K || QT == QT_Q6_K;
    const bool K45 = QT == QT_Q4_K || QT == QT_Q5_K; // (d, dmin) types
    const int NT = 8 * NF, NO = 4 * NF;
    // Per chunk (two, alternating): each block's value per output, and its
    // per-row (Q6_K d, Q8_0/Q5_0 dw) and per-token (dk, d0) scales.
    __shared__ float vs[2][GVM_CH][16][NT];
    __shared__ float rs[2][GVM_CH][16];
    __shared__ float ts[2][GVM_CH][NT];
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
    const u8* xs = act + (u64)min(lane, nb - 1) * astride; // lane < NT: token lane's scales
    float acc[NF];
#pragma unroll
    for (int k = 0; k < NF; k++) acc[k] = 0.0f;
    GvW<QT> cur = {}, nxt = {}; // (warps past the last block copy nxt unloaded)
    if (warp < nblk) nxt.load(rws, warp, t);
    for (int base = 0, it = 0; base < nblk; base += GVM_CH, it++) {
        int buf = it & 1, end = min(nblk, base + GVM_CH);
        // Warp w: blocks w, w + 4, ...
        for (int b = base + warp; b < end; b += 4) {
            int sl = b - base;
            cur = nxt;
            if (b + 4 < nblk) nxt.load(rws, b + 4, t);
            float v[NO];
            if constexpr (K45) {
                float d[2], dmin[2];
                u32 scw[2][2], mw[2][2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    uint4 hd = cur.h[h];
                    d[h] = h2f(hd.x & 0xffff);
                    dmin[h] = h2f(hd.x >> 16);
                    scw[h][0] = hd.y & 0x3f3f3f3f;
                    scw[h][1] = (hd.w & 0x0f0f0f0f) | ((hd.y >> 2) & 0x30303030);
                    mw[h][0] = hd.z & 0x3f3f3f3f;
                    mw[h][1] = ((hd.w >> 4) & 0x0f0f0f0f) | ((hd.z >> 2) & 0x30303030);
                }
                int S[NO], M[NO];
#pragma unroll
                for (int o = 0; o < NO; o++) S[o] = M[o] = 0;
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
#pragma unroll
                    for (int o = 0; o < NO; o++) {
                        int h = (o & 2) >> 1;
                        // (4-byte aligned only: two loads)
                        const int* bs = (const int*)(xo[o] + 2 * np + np / 64) + (u64)b * 8 + 2 * cc;
                        M[o] += m1[h] * bs[0] + m2[h] * bs[1];
                    }
                    // qs bytes 32 cc + 4t.. and 32 cc + 16 + 4t..: low nibbles
                    // sub-block 2cc, high ones 2cc + 1.
                    u32 qa0 = cur.q[0][2 * cc], qa1 = cur.q[0][2 * cc + 1];
                    u32 qb0 = cur.q[1][2 * cc], qb1 = cur.q[1][2 * cc + 1];
                    // Q5_K: bit 4 of each nibble (low: bit 2cc, high: 2cc + 1).
                    u32 la0 = 0, la1 = 0, lb0 = 0, lb1 = 0, ha0 = 0, ha1 = 0, hb0 = 0, hb1 = 0;
                    if constexpr (QT == QT_Q5_K) {
#define GV_H5(w, sh) ((((w) >> (sh)) & 0x01010101) << 4)
                        la0 = GV_H5(cur.qh[0][0], 2 * cc), la1 = GV_H5(cur.qh[0][1], 2 * cc);
                        lb0 = GV_H5(cur.qh[1][0], 2 * cc), lb1 = GV_H5(cur.qh[1][1], 2 * cc);
                        ha0 = GV_H5(cur.qh[0][0], 2 * cc + 1), ha1 = GV_H5(cur.qh[0][1], 2 * cc + 1);
                        hb0 = GV_H5(cur.qh[1][0], 2 * cc + 1), hb1 = GV_H5(cur.qh[1][1], 2 * cc + 1);
#undef GV_H5
                    }
#pragma unroll
                    for (int f = 0; f < NF; f++) {
                        const u8* xb = xa[f] + (u64)b * 256 + 64 * cc + 4 * t;
                        int d1[4], d2[4];
                        mma32(d1, (qa0 & 0x0f0f0f0f) | la0, (qb0 & 0x0f0f0f0f) | lb0,
                              (qa1 & 0x0f0f0f0f) | la1, (qb1 & 0x0f0f0f0f) | lb1, *(const u32*)xb,
                              *(const u32*)(xb + 16));
                        mma32(d2, ((qa0 >> 4) & 0x0f0f0f0f) | ha0, ((qb0 >> 4) & 0x0f0f0f0f) | hb0,
                              ((qa1 >> 4) & 0x0f0f0f0f) | ha1, ((qb1 >> 4) & 0x0f0f0f0f) | hb1,
                              *(const u32*)(xb + 32), *(const u32*)(xb + 48));
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            int h = e >> 1;
                            S[4 * f + e] += sc1[h] * d1[e] + sc2[h] * d2[e];
                        }
                    }
                }
#pragma unroll
                for (int o = 0; o < NO; o++) {
                    int h = (o & 2) >> 1;
                    v[o] = __fsub_rn(__fmul_rn(d[h], i2f_rn(S[o])),
                                     __fmul_rn(dmin[h], i2f_rn(M[o])));
                }
            } else if constexpr (QT == QT_Q6_K) {
                int S[NO];
#pragma unroll
                for (int o = 0; o < NO; o++) S[o] = 0;
#pragma unroll
                for (int ii = 0; ii < 4; ii++) {
                    int hh = ii >> 1, qhi = ii & 1;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        // Sub-block j: values 128 hh + 32 k + 16 qhi ..
                        int j = 8 * hh + 2 * k + qhi;
                        int sca = gm_sbyte(cur.sc[0], j), scb = gm_sbyte(cur.sc[1], j);
                        u32 wv[2];
#pragma unroll
                        for (int r = 0; r < 2; r++) {
                            u32 l = cur.ql[r][ii][k & 1];
                            u32 lo = k < 2 ? (l & 0x0f0f0f0f) : ((l >> 4) & 0x0f0f0f0f);
                            wv[r] = __vsub4(lo | (((cur.qh[r][ii] >> (2 * k)) & 0x03030303) << 4),
                                            0x20202020);
                        }
#pragma unroll
                        for (int f = 0; f < NF; f++) {
                            int dd[4];
                            mma16(dd, wv[0], wv[1],
                                  *(const u32*)(xa[f] + (u64)b * 256 + 16 * j + 4 * t));
                            S[4 * f + 0] += sca * dd[0];
                            S[4 * f + 1] += sca * dd[1];
                            S[4 * f + 2] += scb * dd[2];
                            S[4 * f + 3] += scb * dd[3];
                        }
                    }
                }
#pragma unroll
                for (int o = 0; o < NO; o++) v[o] = i2f_rn(S[o]);
                if (t == 0) rs[buf][sl][g] = cur.d[0], rs[buf][sl][g + 8] = cur.d[1];
            } else {
#pragma unroll
                for (int f = 0; f < NF; f++) {
                    const u8* xb = xa[f] + np + (u64)b * 32 + 4 * t;
                    int dd[4];
                    mma32(dd, cur.a[0][0], cur.a[1][0], cur.a[0][1], cur.a[1][1], *(const u32*)xb,
                          *(const u32*)(xb + 16));
#pragma unroll
                    for (int e = 0; e < 4; e++) v[4 * f + e] = i2f_rn(dd[e]);
                }
                if (t == 0) rs[buf][sl][g] = cur.d[0], rs[buf][sl][g + 8] = cur.d[1];
            }
#pragma unroll
            for (int o = 0; o < NO; o++)
                vs[buf][sl][g + ((o & 2) ? 8 : 0)][8 * (o >> 2) + 2 * t + (o & 1)] = v[o];
            if (lane < NT)
                ts[buf][sl][lane] = *(const float*)(xs + 2 * np + (KQ ? 0 : np / 64 + np / 8) +
                                                    4 * (u64)b);
        }
        __syncthreads();
        // The chains: output q = threadIdx.x + 128 k (row q / NT, token q % NT).
#pragma unroll
        for (int k = 0; k < NF; k++) {
            int q = threadIdx.x + 128 * k, rr = q / NT, zz = q % NT;
            for (int sl = 0; sl < end - base; sl++) {
                float sv = vs[buf][sl][rr][zz];
                if constexpr (K45) acc[k] = __fmaf_rn(ts[buf][sl][zz], sv, acc[k]);
                else acc[k] = __fmaf_rn(__fmul_rn(ts[buf][sl][zz], rs[buf][sl][rr]), sv, acc[k]);
            }
        }
        // (No second barrier: the next chunk writes the other buffer, and
        // this one again only after the next chunk's barrier.)
    }
#pragma unroll
    for (int k = 0; k < NF; k++) {
        int q = threadIdx.x + 128 * k, rr = q / NT, zz = q % NT;
        if (r0 + rr < rows && zz < nb)
            gm_store(acc[k], bias, mode, out, (u64)zz * ostride, r0 + rr);
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
GEMVM(gemvm8_q5_k, QT_Q5_K, 1)
GEMVM(gemvm8_q4_0, QT_Q4_0, 1)
GEMVM(gemvm16_q4_k, QT_Q4_K, 2)
GEMVM(gemvm16_q6_k, QT_Q6_K, 2)
GEMVM(gemvm16_q8_0, QT_Q8_0, 2)
GEMVM(gemvm16_q5_0, QT_Q5_0, 2)
GEMVM(gemvm16_q5_k, QT_Q5_K, 2)
GEMVM(gemvm16_q4_0, QT_Q4_0, 2)

// Test: for every half h and a few x, fmul(h2f(h), x) == fmul(f16_to_f32(h),
// x) bit for bit (the products convert block scales with the hardware cvt,
// and only ever multiply them). The conversions agree on every non-NaN half (f16
// is a subset of f32); on NaN, h2f gives the canonical NaN while f16_to_f32
// keeps the payload, which the products then lose (fmul returns the
// canonical NaN): what this checks, on the GPU at hand.
// out[0] = mismatches. Grid 256 x 256.
extern "C" __global__ void h2f_mul_check(u32* out) {
    u32 h = blockIdx.x * 256 + threadIdx.x;
    const float xs[6] = {1.0f, -3.5f, 0.0f, 1e-30f, INFINITY, NAN_F};
    float a = f16_to_f32(h), b = h2f((unsigned short)h);
    for (int i = 0; i < 6; i++)
        if (__float_as_uint(__fmul_rn(a, xs[i])) != __float_as_uint(__fmul_rn(b, xs[i])))
            atomicAdd(out, 1u); // a count: order doesn't matter
}

// Benchmark (cuda_gemm_bench): int8 mma.sync throughput, 8 independent
// m16n8k32 per step. out[blockIdx.x] keeps the results live.
extern "C" __global__ void __launch_bounds__(256) mma_peak(int iters, int* out) {
#if __CUDA_ARCH__ >= 800
    int d[8][4];
    u32 a = threadIdx.x * 0x01010101u, b = blockIdx.x * 0x01030507u;
    int s = 0;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int k = 0; k < 8; k++) mma32(d[k], a + k, a ^ i, a + 2, a + 3, b, b + k);
#pragma unroll
        for (int k = 0; k < 8; k++) s += d[k][0] ^ d[k][3];
    }
    if (s == 0x12345678) out[blockIdx.x] = s;
#endif
}

// Test (gpu_mma_f16_is_deterministic): D = A B^T, A 64 x K and B N x K
// row-major f16 (K <= 128, N <= 64, both multiples of 16; Bt = B^T, K x N),
// through mma_h chains over the k-steps in ascending order from +0, with
// the 64 rows placed in m16 tiles by perm (tile slot s holds row perm[s])
// and the tiles spread over the block's warps (any block size). A
// fragments from global (amode 0) or by ldmatrix (1); B fragments by
// ldmatrix from B (bmode 0), by ldmatrix.trans from Bt (1), or from global
// (2). D: 64 x N f32, by row. Every combination must give the same bits.
extern "C" __global__ void mma_f16_check(const unsigned short* __restrict__ A,
                                         const unsigned short* __restrict__ B,
                                         const unsigned short* __restrict__ Bt, int K, int N,
                                         const int* __restrict__ perm, int amode, int bmode,
                                         float* __restrict__ D) {
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(16) unsigned short as[64 * 128], bs[64 * 128];
    int tid = threadIdx.x, lane = tid & 31, nw = blockDim.x >> 5;
    for (int i = tid; i < 64 * K; i += blockDim.x) as[i] = A[perm[i / K] * K + i % K];
    for (int i = tid; i < N * K; i += blockDim.x) bs[i] = bmode == 1 ? Bt[i] : B[i];
    __syncthreads();
    int c = 2 * (lane & 3);
    for (int mt = tid >> 5; mt < 4; mt += nw) {
        const unsigned short* ra = A + (u64)perm[mt * 16 + lane / 4] * K;
        const unsigned short* rb = A + (u64)perm[mt * 16 + lane / 4 + 8] * K;
        for (int n2 = 0; n2 < N / 16; n2++) {
            float d[2][4] = {};
            for (int kk = 0; kk < K / 16; kk++) {
                u32 a[4], b[4];
                if (amode == 0) {
                    int k = kk * 16 + c;
                    a[0] = *(const u32*)(ra + k);
                    a[1] = *(const u32*)(rb + k);
                    a[2] = *(const u32*)(ra + k + 8);
                    a[3] = *(const u32*)(rb + k + 8);
                } else {
                    ldsm4(a, as + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * K + kk * 16 +
                                 (lane >> 4) * 8);
                }
                if (bmode == 0) {
                    ldsm4(b, bs + (n2 * 16 + (lane & 7) + (lane >> 4) * 8) * K + kk * 16 +
                                 ((lane >> 3) & 1) * 8);
                } else if (bmode == 1) {
                    ldsm4t(b, bs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * N + n2 * 16 +
                                  (lane >> 4) * 8);
                } else {
                    for (int j = 0; j < 2; j++) {
                        const unsigned short* r = B + (u64)(n2 * 16 + 8 * j + lane / 4) * K;
                        b[2 * j] = *(const u32*)(r + kk * 16 + c);
                        b[2 * j + 1] = *(const u32*)(r + kk * 16 + c + 8);
                    }
                }
                mma_h(d[0], a, b[0], b[1]);
                mma_h(d[1], a, b[2], b[3]);
            }
            for (int j = 0; j < 2; j++) {
                int n = n2 * 16 + 8 * j + c;
                float* d0 = D + (u64)perm[mt * 16 + lane / 4] * N + n;
                float* d1 = D + (u64)perm[mt * 16 + lane / 4 + 8] * N + n;
                d0[0] = d[j][0], d0[1] = d[j][1], d1[0] = d[j][2], d1[1] = d[j][3];
            }
        }
    }
#endif
}

// Benchmark (cuda_gemm_bench): f16 x f16 -> f32 mma.sync throughput, 8
// independent m16n8k16 chains per step.
extern "C" __global__ void __launch_bounds__(256) mma_peak_h(int iters, float* out) {
#if __CUDA_ARCH__ >= 800
    float d[8][4] = {};
    u32 a[4] = {threadIdx.x * 0x00010001u, 0x3c003c00u, 0x3c00u, 0x3800u};
    u32 b = blockIdx.x * 0x00030001u;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int k = 0; k < 8; k++) mma_h(d[k], a, b + k, b ^ i);
    }
    float s = 0.0f;
    for (int k = 0; k < 8; k++) s += d[k][0] + d[k][3];
    if (s == 1234.5f) out[blockIdx.x] = s;
#endif
}
