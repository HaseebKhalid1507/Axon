// qwen3moe's router on the GPU (r11 stage 2: attention and routing here,
// the experts on the CPU). Its own module (`Kernels::moe_module`), loaded by
// the MoE backend only: the dense models' kernels are not touched.

#define MOE_THREADS 128

// Fixed-shape tree over MOE_THREADS values in s (sum): the same pairing for
// every launch.
__device__ float moe_tree_sum(float v, float* s) {
    int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int h = MOE_THREADS / 2; h > 0; h >>= 1) {
        if (t < h) s[t] = __fadd_rn(s[t], s[t + h]);
        __syncthreads();
    }
    float r = s[0];
    __syncthreads();
    return r;
}

// a beats b: higher value, or the same value and a lower index.
__device__ bool moe_beats(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

// Router logits, one warp per (token, expert), 8 experts per block of 256
// threads; grid (ceil(n_e / 8), nb). Each block normalizes its token's x
// (ss over i = t, t + 256, .. then a fixed tree; rmsnorm_quant's ops on
// each value) into shared memory, and block x == 0 also stores it to h
// (every block computes the same bits). Warp w: expert e = 8 blockIdx.x + w,
// lane l sums W[e][c] * h[c] (multiply then add) over c = l, l + 32, ..
// ascending, then the lanes fold by a fixed shuffle tree. No step depends
// on nb: a token's bits are the same in any batch. w: the router as in the
// GGUF, [n_e][dim]. Dynamic shared memory: dim floats.
#define MOE_LG_WARPS 8
extern "C" __global__ void __launch_bounds__(MOE_LG_WARPS * 32)
moe_logits(const float* __restrict__ x, const float* __restrict__ g, float eps, int dim,
           const float* __restrict__ wr, int n_e, float* __restrict__ h,
           float* __restrict__ logits) {
    extern __shared__ float hs[];
    __shared__ float red[MOE_LG_WARPS * 32];
    int t = threadIdx.x, nt = MOE_LG_WARPS * 32;
    size_t tok = blockIdx.y;
    const float* xr = x + tok * dim;
    float ss = 0.0f;
    for (int i = t; i < dim; i += nt) ss = __fadd_rn(ss, __fmul_rn(xr[i], xr[i]));
    red[t] = ss;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = __fadd_rn(red[t], red[t + s]);
        __syncthreads();
    }
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(red[0], (float)dim), eps)));
    for (int i = t; i < dim; i += nt) {
        float v = __fmul_rn(__fmul_rn(xr[i], inv), g[i]);
        hs[i] = v;
        if (blockIdx.x == 0) h[tok * dim + i] = v;
    }
    __syncthreads();
    int lane = t & 31, e = blockIdx.x * MOE_LG_WARPS + (t >> 5);
    if (e >= n_e) return;
    const float* row = wr + (size_t)e * dim;
    float acc = 0.0f;
    int c = lane;
    // 8 loads in flight per lane (the same adds, in the same order).
    for (; c + 7 * 32 < dim; c += 8 * 32) {
        float wv[8];
#pragma unroll
        for (int u = 0; u < 8; u++) wv[u] = row[c + u * 32];
#pragma unroll
        for (int u = 0; u < 8; u++) acc = __fadd_rn(acc, __fmul_rn(wv[u], hs[c + u * 32]));
    }
    for (; c < dim; c += 32) acc = __fadd_rn(acc, __fmul_rn(row[c], hs[c]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffff, acc, o));
    if (lane == 0) logits[tok * n_e + e] = acc;
}

// One token per block (MOE_THREADS threads; grid = tokens): its n_e
// logits (moe_logits) to the k experts in rank order and their weights:
//   p = softmax(logits) (max and sum by fixed trees, expf),
//   k rounds of argmax (ties to the lower index), w_i = p[id_i] / s with
//   s = max(((w_0 + w_1) + ..) + w_{k-1}, 6.103515625e-5) (llama.cpp CPU).
// Dynamic shared memory: n_e floats.
extern "C" __global__ void __launch_bounds__(MOE_THREADS)
moe_topk(const float* __restrict__ logits, int n_e, int k, int* __restrict__ ids,
         float* __restrict__ w) {
    extern __shared__ float p[];
    __shared__ float red[MOE_THREADS];
    __shared__ int rid[MOE_THREADS];
    int t = threadIdx.x;
    size_t tok = blockIdx.x;
    for (int e = t; e < n_e; e += MOE_THREADS) p[e] = logits[tok * n_e + e];
    __syncthreads();
    // Softmax: the max and the sum, each thread's experts ascending, then
    // the tree.
    float mx = __int_as_float(0xff800000);  // -inf
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
    __syncthreads();

    // Top k: each round the best untaken expert (taken ones are -1 < any
    // probability), then it is taken.
    for (int r = 0; r < k; r++) {
        float bv = -2.0f;
        int bi = 0x7fffffff;
        for (int e = t; e < n_e; e += MOE_THREADS)
            if (moe_beats(p[e], e, bv, bi)) {
                bv = p[e];
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
            int e = rid[0];
            ids[tok * k + r] = e;
            w[tok * k + r] = red[0];
            p[e] = -1.0f;
        }
        __syncthreads();
    }
    if (t == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; r++) s = __fadd_rn(s, w[tok * k + r]);
        s = fmaxf(s, 6.103515625e-5f);
        for (int r = 0; r < k; r++) w[tok * k + r] = __fdiv_rn(w[tok * k + r], s);
    }
}

// x[i] += y[i], i < n (the experts' output into the residual stream).
extern "C" __global__ void moe_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fadd_rn(x[i], y[i]);
}
