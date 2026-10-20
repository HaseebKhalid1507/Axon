// main 764792f's three MoE router kernels, verbatim (MLA_DEBUG renamed MOE_DEBUG): the reference for
// src/moe/router_legacy_tests.rs (moe_router, #99, must give the same bits). Compiled after moe.cu (its
// MOE_THREADS, moe_beats, moe_tree_sum). Test fixture only: not in any production module.
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
extern "C" __global__ void __launch_bounds__(MOE_THREADS)
moe_topk_oai(const float* __restrict__ logits, const float* __restrict__ bias, int n_e, int k,
             int* __restrict__ ids, float* __restrict__ w) {
    extern __shared__ float p[];
    __shared__ float red[MOE_THREADS];
    __shared__ int rid[MOE_THREADS];
    int t = threadIdx.x;
    size_t tok = blockIdx.x;
    for (int e = t; e < n_e; e += MOE_THREADS) p[e] = __fadd_rn(logits[tok * n_e + e], bias[e]);
    __syncthreads();
    for (int r = 0; r < k; r++) {
        float bv = __int_as_float(0xff800000);  // -inf
        int bi = 0x7fffffff;
        for (int e = t; e < n_e; e += MOE_THREADS)
            if (p[e] == p[e] && moe_beats(p[e], e, bv, bi)) {
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
#ifdef MOE_DEBUG
            // No untaken non-NaN logit left (a NaN router): debug builds stop
            // here; release ones route to expert 0 (a NaN weight), as
            // mla.cu's mla_topk. (The CPU twin, Moe::route_oai, picks the
            // first untaken expert instead: NaN routers differ, by design
            // nothing to match.)
            if (rid[0] == 0x7fffffff) __trap();
#endif
            int e = rid[0] == 0x7fffffff ? 0 : rid[0];
            ids[tok * k + r] = e;
            w[tok * k + r] = red[0];
            p[e] = __int_as_float(0x7fc00000);  // NaN: taken
        }
        __syncthreads();
    }
    if (t == 0) {
        float mx = w[tok * k];
        float s = 0.0f;
        for (int r = 0; r < k; r++) {
            float v = expf(__fsub_rn(w[tok * k + r], mx));
            w[tok * k + r] = v;
            s = __fadd_rn(s, v);
        }
        float inv = __fdiv_rn(1.0f, s);
        for (int r = 0; r < k; r++) w[tok * k + r] = __fmul_rn(w[tok * k + r], inv);
    }
}
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
            // Finite only: NaN never, taken experts (-inf) never again.
            if (sel[e] > __int_as_float(0xff800000) && moe_beats(sel[e], e, bv, bi)) {
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
#ifdef MOE_DEBUG
            // No finite selection score left (a NaN router): debug builds
            // stop here; release ones route to expert 0 (a NaN weight).
            if (rid[0] == 0x7fffffff) __trap();
#endif
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
