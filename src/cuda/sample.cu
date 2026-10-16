// Sampling candidates on the GPU (sample.rs): the repeat penalty, then the
// k best logits of each row as (logit, id), best first, ties to the larger
// id: exactly sample::cands on the same logits. Only those k pairs go back
// to the host, never the vocab-sized row. One kernel, samp_topk, in place of
// the greedy step's argmax:
//
//   1. each block stages a slice of TK_SLICE logits in shared memory,
//      applies the penalty to the row's penalty ids that fall in it
//      (l > 0 ? l / p : l * p, as sample::penalize; the logits buffer is
//      left alone), and selects the slice's k best keys into `part`;
//   2. the last block of the row to finish (a counter that wraps to 0 on
//      it: atomicInc) merges
//      the slices' picks: the k best of them, sorted, to `out`.
//
// Order: key = tk_key(logit) << 32 | id, a 64-bit integer that is distinct
// per logit and orders as the candidates must (sample::key). Selection is a
// radix select on it, 8 bits per pass from the top, stopping as soon as the
// bin holding the k-th key is taken whole; so the picked set is exact
// (the k largest keys), whatever the thread timing, and only its order in
// `part` varies, which the merge's rank sort removes.
//
// S, the parameters (host-written, see qwen2.rs `samp_upload`), as u32s:
//   S[0] = k (<= TK_MAX), S[1] = penalty (f32 bits), S[2] = rows,
//   S[3 .. 3 + rows + 1] = offsets into ids of each row's penalty ids
//   (distinct, sorted), ids from S[4 + rows].
// Per-row penalties (the server's multi-sequence step, serve S2):
//   S[1] = SP_PER_ROW, S[4 + rows .. 4 + 2 rows] = each row's penalty (f32
//   bits), ids from S[4 + 2 rows]. k stays per launch: the rows of a step
//   take the largest k, and a row wanting fewer keeps the first ones (the
//   k best, best first, are a prefix of any larger k's).

#define SP_K 0
#define SP_PEN 1
#define SP_ROWS 2
#define SP_OFF 3
#define SP_PER_ROW 0xffffffffu
#define TK_SLICE 4096
#define TK_THREADS 512
#define TK_MAX 256
// Slice picks the merge stages in shared memory (more: read from L2 each pass).
#define TK_STAGE (TK_SLICE / 2)

// sample::key: NaN lowest (0), -0 == +0, else order-preserving.
__device__ __forceinline__ u32 tk_key(float v) {
    if (v != v) return 0u;
    u32 b = v == 0.0f ? 0u : __float_as_uint(v);
    return (b >> 31) ? ~b : (b | 0x80000000u);
}

// sample::unkey.
__device__ __forceinline__ float tk_val(u32 k) {
    return __uint_as_float((k >> 31) ? (k & 0x7fffffffu) : ~k);
}

// The smallest P with exactly min(k, nvalid) of the valid keys >= P (keys
// distinct). get(i, &key) for i < n says whether slot i holds a key. hist:
// 256 u32 of shared memory, bc: 3. Called by the whole block (blockDim a
// multiple of 32), uniformly.
template <class Get>
__device__ u64 tk_select(Get get, int n, int nvalid, int k, u32* hist, u32* bc) {
    if (nvalid <= k) return 0;
    const int lane = threadIdx.x & 31;
    u64 prefix = 0, mask = 0;
    u32 kk = (u32)k; // keys still to take among those matching prefix
    for (int shift = 56;; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i0 = 0; i0 < n; i0 += blockDim.x) {
            int i = i0 + threadIdx.x;
            u64 x = 0;
            bool on = i < n && get(i, x) && (x & mask) == prefix;
            u32 bin = on ? (u32)(x >> shift) & 255u : 256u + lane;
#if __CUDA_ARCH__ >= 700
            // One atomic per distinct bin in the warp: logits crowd a few bins.
            u32 same = __match_any_sync(0xffffffffu, bin);
            if (on && __ffs(same) - 1 == lane) atomicAdd(&hist[bin], __popc(same));
#else
            if (on) atomicAdd(&hist[bin], 1u);
#endif
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            // Lane l holds bins 255 - 8l down to 248 - 8l; find the bin d
            // with (keys in bins above d) < kk <= (... and in d).
            u32 c[8], s = 0;
#pragma unroll
            for (int j = 0; j < 8; j++) c[j] = hist[255 - 8 * lane - j], s += c[j];
            u32 inc = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                u32 t = __shfl_up_sync(0xffffffffu, inc, o);
                if (lane >= o) inc += t;
            }
            u32 above = inc - s;
            if (above < kk && kk <= inc) {
                for (int j = 0; j < 8; j++) {
                    if (above + c[j] >= kk) {
                        bc[0] = 255 - 8 * lane - j;
                        bc[1] = kk - above;
                        bc[2] = c[j];
                        break;
                    }
                    above += c[j];
                }
            }
        }
        __syncthreads();
        u32 d = bc[0], cnt = bc[2];
        kk = bc[1];
        prefix |= (u64)d << shift;
        mask |= 255ull << shift;
        // Every key in bin d is wanted: the keys >= prefix are exactly the
        // k. (At shift 0 the bin is one key.)
        if (cnt == kk || shift == 0) return prefix;
        __syncthreads(); // bc read by all before the next pass rewrites it
    }
}

// Grid (slices = ceil(n / TK_SLICE), rows) x TK_THREADS: row blockIdx.y of
// x (n logits, row stride `stride`; penalty row row0 + blockIdx.y of S) to
// out row out_row0 + blockIdx.y: min(k, n) (logit bits, id) pairs, k pairs
// apart, best first. part: rows x slices x k u64 scratch; count: rows u32,
// zero when allocated; each launch leaves it zero (atomicInc).
extern "C" __global__ void __launch_bounds__(TK_THREADS)
    samp_topk(const float* __restrict__ x, u64 stride, int n, const u32* __restrict__ S, int row0,
              int out_row0, u64* __restrict__ part, u32* __restrict__ count, u32* __restrict__ out) {
    PDL_START();
    __shared__ union {
        float f[TK_SLICE];
        u32 k[TK_SLICE];
        u64 m[TK_STAGE];
    } sh;
    __shared__ u64 sel[TK_MAX];
    __shared__ u32 hist[256];
    __shared__ u32 bc[3];
    __shared__ u32 slot;
    __shared__ int is_last;
    const int k = (int)S[SP_K];
    const int slices = gridDim.x, row = blockIdx.y;
    const float* xr = x + (u64)row * stride;
    const int base = blockIdx.x * TK_SLICE;
    const int len = min(TK_SLICE, n - base);
    for (int i = threadIdx.x; i < len; i += blockDim.x) sh.f[i] = xr[base + i];
    if (threadIdx.x == 0) slot = 0;
    __syncthreads();
    const int r = row0 + row;
    const bool per_row = S[SP_PEN] == SP_PER_ROW;
    const float p = __uint_as_float(per_row ? S[SP_OFF + S[SP_ROWS] + 1 + r] : S[SP_PEN]);
    if (p != 1.0f) {
        const u32* ids = S + SP_OFF + S[SP_ROWS] + 1 + (per_row ? S[SP_ROWS] : 0);
        for (u32 j = S[SP_OFF + r] + threadIdx.x; j < S[SP_OFF + r + 1]; j += blockDim.x) {
            u32 i = ids[j] - (u32)base; // ids are distinct: one thread per logit
            if (i < (u32)len) {
                float l = sh.f[i];
                sh.f[i] = l > 0.0f ? __fdiv_rn(l, p) : __fmul_rn(l, p);
            }
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < len; i += blockDim.x) sh.k[i] = tk_key(sh.f[i]);
    __syncthreads();
    auto local = [&](int i, u64& key) {
        key = ((u64)sh.k[i] << 32) | (u32)(base + i);
        return true;
    };
    u64 P = tk_select(local, len, len, k, hist, bc);
    u64* pr = part + (u64)row * slices * k;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        u64 key = ((u64)sh.k[i] << 32) | (u32)(base + i);
        if (key >= P) pr[(u64)blockIdx.x * k + atomicAdd(&slot, 1u)] = key;
    }
    // The last block of the row to get here merges (threadFenceReduction).
    __threadfence();
    __syncthreads();
    // atomicInc wraps the counter to 0 on the last block: zero again after
    // every launch, with no reset to forget, whatever `slices` the next
    // launch on this row has (graph replays included).
    if (threadIdx.x == 0) is_last = atomicInc(&count[row], (u32)slices - 1) == (u32)slices - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    // Slice j kept min(k, its length) keys; all but the last are full.
    const int last = min(k, n - (slices - 1) * TK_SLICE);
    const int full = min(k, TK_SLICE);
    const int nvalid = (slices - 1) * full + last;
    const int n_slots = slices * k;
    const bool staged = n_slots <= TK_STAGE;
    auto valid = [&](int i) {
        int j = i / k, s = i - j * k;
        return s < (j == slices - 1 ? last : full);
    };
    if (staged) {
        // sh.f / sh.k are done with. __ldcg: other blocks wrote part (L2).
        for (int i = threadIdx.x; i < n_slots; i += blockDim.x)
            sh.m[i] = valid(i) ? __ldcg(pr + i) : 0;
        __syncthreads();
    }
    auto merged = [&](int i, u64& key) {
        if (!valid(i)) return false;
        key = staged ? sh.m[i] : __ldcg(pr + i);
        return true;
    };
    u64 Q = tk_select(merged, n_slots, nvalid, k, hist, bc);
    if (threadIdx.x == 0) slot = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n_slots; i += blockDim.x) {
        u64 key;
        if (merged(i, key) && key >= Q) sel[atomicAdd(&slot, 1u)] = key;
    }
    __syncthreads();
    const int m = min(k, nvalid);
    u32* o = out + (u64)(out_row0 + row) * k * 2;
    for (int t = threadIdx.x; t < m; t += blockDim.x) {
        u64 me = sel[t];
        int rank = 0;
        for (int j = 0; j < m; j++) rank += sel[j] > me;
        o[2 * rank] = __float_as_uint(tk_val((u32)(me >> 32)));
        o[2 * rank + 1] = (u32)me;
    }
}
