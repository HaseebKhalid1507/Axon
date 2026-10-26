
// ---------------------------------------------------------------------------
// Resident MoE experts (specs r16 pick 4, PR B): one layer's routed experts
// on the GPU, bit for bit the CPU's `moe_layer_batch` (model/moe.rs). The
// products are moe_twin.cu's twins (gemm2_t / dot_rows_t with TW), the
// activation is swiglu_twin, the quantizer quant_warp (== QuantizedAct), and
// the weighted sum is the CPU's `combine` in rank order. A token's output
// never depends on what else is in its batch: each (token, slot) pair is its
// own column of the grouped products, and each output's K chain stays in one
// thread in block order.
//
// Batched (sm_80+): moe_sort groups the nb * k pairs by expert (ascending
// pair order within each, as the CPU's counting sort) and cuts each expert's
// run into 64-token tiles; moe_gather copies each pair's quantized input in
// that order; moe_gemm_tw_* run gate, up and down over the tile list
// (blockIdx.x = tile); swiglu_quant_tw makes the down's input; moe_sum the
// weighted sums. Decode (any arch): moe_gemv_tw over the k slots.

// The tile width of the grouped GEMMs (gemm2s: 64 tokens a block).
#define MOE_BN 64

// ids[0..np) (pair p = token * k + slot: its expert) grouped by expert:
// off[e] = the first sorted position of expert e (off[n_e] = np),
// pair_at[s] = the pair at sorted position s, pos_of[p] = its position;
// tiles[j] = (expert, first position, count <= MOE_BN, 0) in expert order,
// *ntiles of them. One block; a warp per expert, ballots in pair order.
// Shared: n_e + 1 ints.
extern "C" __global__ void moe_sort(const int* __restrict__ ids, int np, int n_e,
                                    int* __restrict__ off, int* __restrict__ pair_at,
                                    int* __restrict__ pos_of, int4* __restrict__ tiles,
                                    int* __restrict__ ntiles) {
    extern __shared__ int ms_at[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int e = warp; e < n_e; e += nw) {
        int c = 0;
        for (int p0 = 0; p0 < np; p0 += 32) {
            int p = p0 + lane;
            c += __popc(__ballot_sync(0xffffffffu, p < np && ids[p] == e));
        }
        if (lane == 0) ms_at[e] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0, nt = 0;
        for (int e = 0; e < n_e; e++) {
            int c = ms_at[e];
            off[e] = s;
            for (int j = 0; j < c; j += MOE_BN) tiles[nt++] = make_int4(e, s + j, min(MOE_BN, c - j), 0);
            ms_at[e] = s;
            s += c;
        }
        off[n_e] = s;
        *ntiles = nt;
    }
    __syncthreads();
    for (int e = warp; e < n_e; e += nw) {
        int at = ms_at[e];
        for (int p0 = 0; p0 < np; p0 += 32) {
            int p = p0 + lane;
            bool hit = p < np && ids[p] == e;
            unsigned m = __ballot_sync(0xffffffffu, hit);
            if (hit) {
                int s = at + __popc(m & ((1u << lane) - 1u));
                pair_at[s] = p;
                pos_of[p] = s;
            }
            at += __popc(m);
        }
    }
}

// dst row s = src row pair_at[s] / k (astride bytes each, a multiple of 16):
// each pair's token's quantized input, in sorted order. Grid np.
extern "C" __global__ void moe_gather(const u8* __restrict__ src, u64 astride,
                                      const int* __restrict__ pair_at, int k,
                                      u8* __restrict__ dst) {
    int s = blockIdx.x;
    const uint4* a = (const uint4*)(src + (u64)(pair_at[s] / k) * astride);
    uint4* d = (uint4*)(dst + (u64)s * astride);
    for (int i = threadIdx.x; i < (int)(astride / 16); i += blockDim.x) d[i] = a[i];
}

// out row s (sorted position s, ostride floats) = expert tiles[j].x's rows
// times act row s, in the CPU twin's numerics, for the positions of tile j =
// blockIdx.x (blocks past *ntiles exit). w: the experts' matrices back to
// back, estride bytes apart. Grid (max tiles, rows / 64 rounded up).
#define MOE_GEMM_TW(NAME, QT)                                                                     \
    extern "C" __global__ void __launch_bounds__(G2_THREADS, 1) NAME(                             \
        const u8* __restrict__ w, u64 estride, u64 rb, int rows, int n,                          \
        const u8* __restrict__ act, u64 astride, const int4* __restrict__ tiles,                 \
        const int* __restrict__ ntiles, float* __restrict__ out, u64 ostride) {                   \
        int j = blockIdx.x;                                                                       \
        if (j >= *ntiles) return;                                                                 \
        int4 t = tiles[j];                                                                        \
        gemm2_t<QT, 2, 3, 0, 2, true>(w + (u64)t.x * estride, rb, rows, n,                        \
                                      act + (u64)t.y * astride, astride, t.z, nullptr, 0,         \
                                      out + (u64)t.y * ostride, ostride, nullptr, 0);             \
    }
MOE_GEMM_TW(moe_gemm_tw_q4_k, QT_Q4_K)
MOE_GEMM_TW(moe_gemm_tw_q5_0, QT_Q5_0)
MOE_GEMM_TW(moe_gemm_tw_q8_0, QT_Q8_0)
MOE_GEMM_TW(moe_gemm_tw_q6_k, QT_Q6_K)

// Row z (n floats): swiglu_twin(g, u) (F = fused), quantized (quant_warp) into
// act row z: the CPU's `glu` then `QuantizedAct::for_types` of one pair.
// Grid (n / 256 rounded up, rows), 32 threads.
extern "C" __global__ void swiglu_quant_tw(const float* __restrict__ g, const float* __restrict__ u,
                                           int n, int fused, int want_k, u8* __restrict__ act,
                                           u64 astride) {
    int lane = threadIdx.x & 31, blk = blockIdx.x, z = blockIdx.y;
    const float* gz = g + (u64)z * n;
    const float* uz = u + (u64)z * n;
    float v[8];
    for (int i = 0; i < 8; i++) {
        int e = blk * 256 + lane * 8 + i;
        v[i] = e < n ? swiglu_twin(gz[e], uz[e], fused != 0) : 0.0f;
    }
    quant_warp(v, n, blk, act + (u64)z * astride, want_k != 0);
}

// y[t * dim + r] = sum over slots i in rank order of w[p] d[pos_of[p]][r]
// (p = t k + i): the CPU's `combine`, y = w0 d0, then y + wi di (a multiply
// then an add). Grid (dim / 128 rounded up, nb), 128 threads.
extern "C" __global__ void moe_sum(const float* __restrict__ d, const int* __restrict__ pos_of,
                                   const float* __restrict__ w, int k, int dim,
                                   float* __restrict__ y) {
    int r = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (r >= dim) return;
    float acc = 0.0f;
    for (int i = 0; i < k; i++) {
        int p = t * k + i;
        float v = __fmul_rn(w[p], d[(u64)pos_of[p] * dim + r]);
        acc = i == 0 ? v : __fadd_rn(acc, v);
    }
    y[(u64)t * dim + r] = acc;
}

// The decode expert GEMV: out[i * ostride + r] = row r of expert ids[i]
// (estride bytes apart in w) times slot i's activations (act + i aslot;
// aslot 0: one token's for every slot), gemv_tw's numerics. Grid (rows, k).
extern "C" __global__ void __launch_bounds__(GEMV_THREADS, GEMV_MIN_BLOCKS)
moe_gemv_tw(int qt, const u8* __restrict__ w, u64 estride, u64 row_bytes, int rows, int n,
            const int* __restrict__ ids, const u8* __restrict__ act, u64 aslot,
            float* __restrict__ out, u64 ostride) {
    // Before reading ids and act (the router's and the quantizer's outputs):
    // under programmatic dependent launch it waits for them, as gemv_tw.
    PDL_START();
    int r = blockIdx.x, i = blockIdx.y;
    const u8* we = w + (u64)ids[i] * estride;
    prefetch_l2(we + (u64)r * row_bytes, row_bytes);
    const u8* rr[1] = {we + (u64)r * row_bytes};
    float y = dot_rows_t<1, 1, true>(qt, rr, 0, n, act + (u64)i * aslot, 0, 0, nullptr);
    if (threadIdx.x == 0) out[(u64)i * ostride + r] = y;
}
