
// #190 diagnostic kernels (tests only, appended to KERNELS_F8 by
// gpu_fp8_mma_diag). diag_chain_f8: gemm_ref_f8's loads and mma chain, raw
// accumulators (no scales) out[t * rows + r]; P > 0: every P mmas the accumulators are added into f32 registers
// (__fadd_rn) and restarted from 0 (a promotion); h16: each k32 step as two
// m16n8k16 f16 mmas of the codes converted to f16 (exact) instead, else the
// e4m3 mma itself on every architecture (diag_mma_e4m3, not fp8.cu's mma_f8).
__device__ __forceinline__ u32 diag_h2(u32 v) {
    u32 r;
    unsigned short s = (unsigned short)v;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(s));
    return r;
}
__device__ __forceinline__ void diag_mma_h(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void diag_mma_e4m3(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
extern "C" __global__ void __launch_bounds__(32)
diag_chain_f8(const u8* __restrict__ w, int rows, int K, const u8* __restrict__ qa, int nb,
              int P, int h16, float* __restrict__ out) {
    int lane = threadIdx.x, g = lane >> 2, q = lane & 3;
    int tb = blockIdx.x * 8, rb = blockIdx.y * 16;
    int ra = rb + g, rc = rb + g + 8, tn = tb + g;
    float d[4], acc[4];
    for (int e = 0; e < 4; e++) d[e] = acc[e] = 0.0f;
    int n = 0;
    for (int k = 0; k < K; k += 32) {
        int ka = k + 4 * q, kb = k + 16 + 4 * q;
        u32 a0 = ra < rows ? *(const u32*)(w + (u64)ra * K + ka) : 0u;
        u32 a1 = rc < rows ? *(const u32*)(w + (u64)rc * K + ka) : 0u;
        u32 a2 = ra < rows ? *(const u32*)(w + (u64)ra * K + kb) : 0u;
        u32 a3 = rc < rows ? *(const u32*)(w + (u64)rc * K + kb) : 0u;
        u32 b0 = tn < nb ? *(const u32*)(qa + (u64)tn * K + ka) : 0u;
        u32 b1 = tn < nb ? *(const u32*)(qa + (u64)tn * K + kb) : 0u;
        if (h16) {
            diag_mma_h(d, diag_h2(a0), diag_h2(a1), diag_h2(a0 >> 16), diag_h2(a1 >> 16), diag_h2(b0),
                       diag_h2(b0 >> 16));
            diag_mma_h(d, diag_h2(a2), diag_h2(a3), diag_h2(a2 >> 16), diag_h2(a3 >> 16), diag_h2(b1),
                       diag_h2(b1 >> 16));
        } else {
            diag_mma_e4m3(d, a0, a1, a2, a3, b0, b1);
        }
        if (P > 0 && ++n == P) {
            n = 0;
            for (int e = 0; e < 4; e++) {
                acc[e] = __fadd_rn(acc[e], d[e]);
                d[e] = 0.0f;
            }
        }
    }
    for (int e = 0; e < 4; e++) {
        int r = rb + g + (e >> 1) * 8, t = tb + 2 * q + (e & 1);
        if (r < rows && t < nb) out[(u64)t * rows + r] = P > 0 ? __fadd_rn(acc[e], d[e]) : d[e];
    }
}
