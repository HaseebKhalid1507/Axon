
// ---------------------------------------------------------------------------
// The CPU's expert numerics on the GPU, bit for bit ("twins"): resident
// experts (specs r16 pick 4) must give the CPU's r8/r8q expert bits, so
// which layers sit in VRAM never changes a result. Family-generic: the
// same primitives serve qwen3moe (Q4_K / Q6_K experts), deepseek2 (Q4_K /
// Q5_0 / Q8_0) and, with an MXFP4 twin later, gpt-oss.
//
// exp_twin / swiglu_twin: vmath.rs exp_f::<F> / swiglu1_f::<F>, op for op.
// Every float op is an explicit _rn intrinsic (nvcc would otherwise contract
// a multiply and an add into an FMA); `fused` is the host CPU's
// `vmath::fused()` (F), passed in by the host: with it the polynomial's
// steps are FMAs, without it a multiply then an add, as the CPU's. The AVX2
// `exp8` is the F = true sequence lane for lane, so one twin serves both.

// a * b + c, fused (one rounding) when F, else a multiply and an add.
__device__ __forceinline__ float tw_fma(bool F, float a, float b, float c) {
    return F ? __fmaf_rn(a, b, c) : __fadd_rn(__fmul_rn(a, b), c);
}

// vmath.rs exp_f::<F>: the same constants (as bits), clamp, Cody-Waite
// reduction, degree-6 polynomial of 2 exp(r) and scale 2^(n - 1) from bits.
__device__ __forceinline__ float exp_twin(float x, bool F) {
    const float LO = __int_as_float(0xc2ae999a);    // -87.3
    const float HI = __int_as_float(0x42b1999a);    // 88.8
    const float LOG2E = __int_as_float(0x3fb8aa3b);
    const float NLN2_HI = __int_as_float(0xbf318000); // -355 / 512
    const float NLN2_LO = __int_as_float(0x395e8083); // -(ln 2 - 355 / 512)
    // NaN keeps x (the comparisons are false), as the CPU's.
    x = LO > x ? LO : x;
    x = HI < x ? HI : x;
    float n = rintf(__fmul_rn(x, LOG2E)); // round half to even
    float r = tw_fma(F, n, NLN2_HI, x);
    r = tw_fma(F, n, NLN2_LO, r);
    float q = __int_as_float(0x3b35280e);         // D6
    q = tw_fma(F, q, r, __int_as_float(0x3c891cd5)); // D5
    q = tw_fma(F, q, r, __int_as_float(0x3daaac58)); // D4
    q = tw_fma(F, q, r, __int_as_float(0x3eaaaa49)); // D3
    q = tw_fma(F, q, r, __int_as_float(0x3f7ffffe)); // D2
    q = tw_fma(F, q, r, __int_as_float(0x40000000)); // D1
    float p = tw_fma(F, q, r, 2.0f);
    // n is an integer in -126..=128 (NaN: p is NaN already).
    return __fmul_rn(p, __int_as_float(((int)n + 126) << 23));
}

// vmath.rs swiglu1_f::<F>: g / (1 + exp(-g)) * u.
__device__ __forceinline__ float swiglu_twin(float g, float u, bool F) {
    return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, exp_twin(-g, F))), u);
}

// out[i] = swiglu_twin(g[i], u[i]) for i < n (the tests' handle on the
// device functions; the expert kernels call them inline).
extern "C" __global__ void swiglu_twin_k(const float* __restrict__ g, const float* __restrict__ u,
                                         int n, int fused, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = swiglu_twin(g[i], u[i], fused != 0);
}

// out[i] = exp_twin(x[i]) for i < n (tests).
extern "C" __global__ void exp_twin_k(const float* __restrict__ x, int n, int fused,
                                      float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp_twin(x[i], fused != 0);
}
