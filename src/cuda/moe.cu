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

// The router after moe_logits, one token per block (MOE_THREADS threads;
// grid = tokens): its n_e logits to the k experts in rank order and their
// weights. The gate makes p from the logits:
//   MOE_GATE_SOFTMAX (qwen3moe, deepseek2): softmax (max and sum by fixed
//     trees, expf);
//   MOE_GATE_SIGMOID (deepseek2): 1 / (1 + exp(-x));
//   MOE_GATE_OAI (gpt-oss): the logits themselves.
// Selection by sel = p + bias (bias null: p; deepseek2's exp_probs_b:
// selection only; gpt-oss: the router's bias, required), k rounds of tree
// argmax over the untaken non-NaN scores (softmax/sigmoid: finite or +inf;
// OAI: -inf too), ties to the lower index; w_i = p[id_i] (OAI: sel[id_i],
// the biased logit). Then OAI: the softmax over those k (max = the first,
// e_i = exp(w_i - max), s = ((e_0 + e_1) + ..) + e_{k-1}, w_i = e_i *
// (1 / s)); norm: s = max(((w_0 + w_1) + ..) + w_{k-1}, 6.103515625e-5),
// w_i /= s (llama.cpp CPU); then times scale if not 1. qwen3moe: SOFTMAX,
// no bias, norm, scale 1; gpt-oss: OAI, no norm, scale 1.
// No eligible score left (a NaN router): expert 0, __trap() under
// MOE_DEBUG (debug builds; an unknown gate traps there too). (gpt-oss's
// CPU twin, Moe::route_oai, picks the first untaken expert instead: NaN
// routers differ, by design nothing to match.) Dynamic shared memory:
// 2 n_e floats.
#define MOE_GATE_SOFTMAX 0
#define MOE_GATE_SIGMOID 1
#define MOE_GATE_OAI 2
extern "C" __global__ void __launch_bounds__(MOE_THREADS)
moe_router(const float* __restrict__ logits, int n_e, int k, int gate,
           const float* __restrict__ bias, int norm, float scale, int* __restrict__ ids,
           float* __restrict__ w) {
    extern __shared__ float sp[];
    float* p = sp;
    float* sel = sp + n_e;
    __shared__ float red[MOE_THREADS];
    __shared__ int rid[MOE_THREADS];
    const float ninf = __int_as_float(0xff800000), nan = __int_as_float(0x7fc00000);
    int t = threadIdx.x;
    size_t tok = blockIdx.x;
#ifdef MOE_DEBUG
    if (gate < MOE_GATE_SOFTMAX || gate > MOE_GATE_OAI) __trap();
#endif
    for (int e = t; e < n_e; e += MOE_THREADS) p[e] = logits[tok * n_e + e];
    __syncthreads();
    if (gate == MOE_GATE_SIGMOID) {
        for (int e = t; e < n_e; e += MOE_THREADS)
            p[e] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-p[e])));
    } else if (gate == MOE_GATE_SOFTMAX) {
        // The max and the sum, each thread's experts ascending, then the
        // tree.
        float mx = ninf;
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
    for (int e = t; e < n_e; e += MOE_THREADS) {
        float s = bias ? __fadd_rn(p[e], bias[e]) : p[e];
        if (gate != MOE_GATE_OAI && s == ninf) s = nan;  // -inf: never selected
        sel[e] = s;
    }
    __syncthreads();
    // k rounds of argmax over the untaken scores. moe_beats is a total
    // order (higher value, then lower index), so its maximum is one expert
    // however the candidates are paired: each warp folds its lanes by a
    // shuffle tree, thread 0 folds the warps' winners. The selected experts
    // and their weights are the same as any fixed tree's (T2b: 2 barriers a
    // round instead of a block tree's 8; no float op on the scores).
    const int NW = MOE_THREADS / 32;
    int lane = t & 31, wid = t >> 5;
    for (int r = 0; r < k; r++) {
        float bv = ninf;
        int bi = 0x7fffffff;
        for (int e = t; e < n_e; e += MOE_THREADS)
            // Non-NaN only: taken experts (NaN) never again.
            if (sel[e] == sel[e] && moe_beats(sel[e], e, bv, bi)) {
                bv = sel[e];
                bi = e;
            }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (moe_beats(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            red[wid] = bv;
            rid[wid] = bi;
        }
        __syncthreads();
        if (t == 0) {
            for (int j = 1; j < NW; j++)
                if (moe_beats(red[j], rid[j], red[0], rid[0])) {
                    red[0] = red[j];
                    rid[0] = rid[j];
                }
#ifdef MOE_DEBUG
            if (rid[0] == 0x7fffffff) __trap();
#endif
            int e = rid[0] == 0x7fffffff ? 0 : rid[0];
            ids[tok * k + r] = e;
            // OAI: the selected score (-inf if none left), else p
            // (expert 0's).
            w[tok * k + r] = gate == MOE_GATE_OAI ? red[0] : p[e];
            sel[e] = nan;
        }
        __syncthreads();
    }
    if (t == 0) {
        float* wt = w + tok * k;
        if (gate == MOE_GATE_OAI) {
            float mx = wt[0];
            float s = 0.0f;
            for (int r = 0; r < k; r++) {
                float v = expf(__fsub_rn(wt[r], mx));
                wt[r] = v;
                s = __fadd_rn(s, v);
            }
            float inv = __fdiv_rn(1.0f, s);
            for (int r = 0; r < k; r++) wt[r] = __fmul_rn(wt[r], inv);
        }
        if (norm) {
            float s = 0.0f;
            for (int r = 0; r < k; r++) s = __fadd_rn(s, wt[r]);
            s = fmaxf(s, 6.103515625e-5f);
            for (int r = 0; r < k; r++) wt[r] = __fdiv_rn(wt[r], s);
        }
        if (scale != 1.0f)
            for (int r = 0; r < k; r++) wt[r] = __fmul_rn(wt[r], scale);
    }
}

// x[i] += y[i], i < n (the experts' output into the residual stream).
extern "C" __global__ void moe_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fadd_rn(x[i], y[i]);
}
