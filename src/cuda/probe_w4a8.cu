
// ---------------------------------------------------------------------------
// TEST-ONLY (T3 / r11 rule P, Hopper W4A8 measured upper bound). Never part
// of KERNELS: tests.rs compiles it as KERNELS' helpers + this file into a
// module of its own (its own cache key), so no product path can load it.
//
// Four kernels on one warp-specialized, persistent structure (CTA 384
// threads, 1 CTA / SM, grid = #SMs, tiles of 128 rows x 64 tokens, tokens
// fastest, a PW_ST-stage TMA ring that streams across tiles):
// - probe_w4a8_n64   Q4_K skeleton: INTEGER-ONLY, no per-256 drain. Writes
//   S = sum_b sum_j sc_bj sumi_bj, M = sum_b sum_j m_bj bsum_bj over all of
//   K (wrapping int32). Not the GEMM: its speed bounds an exact kernel's.
// - probe_w4a8x_n64  Q4_K EXACT: the same tensor work plus the real per-256
//   drain in gemmh2_q4_k's float ops and order (v = d S - dmin M, acc =
//   fma(dk, v, acc), block after block), out = acc: bit-identical to
//   gemmh2_q4_k (mode 0) by construction (test: probe_w4a8x vs gemmh2).
// - probe_w6a8_n64   Q6_K skeleton on the probe copy (PW6, below): three
//   signed-digit planes of the scales, u8 x s8 RS wgmma, integer-only.
//   Writes S6 = P0 + 8 P1 + 64 P2 (= sum sc q x + 9344 sum x) over all K.
// - probe_w6a8x_n64  Q6_K EXACT: per-256 drain in gemm2_q6k8's float ops,
//   acc = fma(dk dw, float(S_b), acc): bit-identical to gemm2(s)_q6k8.
//
// Roles: warpgroup 0 = producer (setmaxnreg 40): warp 0 lane 0 issues the
// TMA loads of a stage; warps 1-3 build per-stage token data from global
// (Q4_K: the bsum-piece tile for the M wgmmas; EXACT: dk, and Q6 the
// block's activation sum). Warpgroups 1, 2 = consumers (setmaxnreg 232):
// rows 64 (wg - 1) .. +63 of the tile x 64 tokens; per 256-block 8 k32
// steps, each unpacking its A fragments into one of 4 rotating register
// slots and issuing its wgmmas as one commit group (the unpack of step j+1
// overlaps the wgmmas of j..j-3).
//
// PW6, the probe's Q6_K copy (288 bytes a 256-block, as Q6K8): bytes 0..255
// the values q - 32 as int8 (Q6K8's, unchanged); 256..279 three u64 digit
// planes p: byte j holds e = d + 3 (0..7) of sub-block 2j (low nibble) and
// 2j+1 (high), where sc = d0 + 8 d1 + 64 d2, d in [-3, 4] (balanced radix
// 8: every int8 has exactly one such form); 280..283 d (f32, h2f of the
// f16, as Q6K8); 284..287 zero. A load-time copy as q6k_to_i8 is: the same
// information, laid out for this kernel. Per 32-bit A word of 4 values:
//   U = (W ^ 0x20202020) & 0x3f3f3f3f   (bytes q + 32, 0..63)
//   Q = U - 0x20202020                   (= sum_k q_k 256^k mod 2^32)
//   A_p = Q d_p + 0x80808080             (bytes q_k d_p + 128 in 0..252:
//         exact, no byte overflows, since q d in [-128, 124])
// so sum_k A_pk x_k = d_p sum q x + 128 sum x, and sum_p 8^p (...) =
// sum_j sc_j sumq_j + 128 * 73 * sum x: S_b = P0 + 8 P1 + 64 P2 - 9344 bs_b.
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

#define PW_ST 4
#define PW_ACT (64 * 256)  // two 128-byte swizzled boxes of 64 token rows
#define PW_BS (64 * 128)   // Q4_K: the bsum-piece tile, 128-byte swizzled
#define PW_RAW (128 * 144) // 128 rows x 144 B: a Q4_K block, half a PW6 block
#define PW_STAGE(QK) ((QK) == 4 ? PW_ACT + PW_BS + PW_RAW : PW_ACT + 2 * PW_RAW)
#define PW_META 512        // per stage: dk[64] (f32), bs[64] (i32)
#define PW_OFF_META(QK) (PW_ST * PW_STAGE(QK))
#define PW_OFF_BAR(QK) (PW_OFF_META(QK) + PW_ST * PW_META)
#define PROBE_W4A8_SMEM(QK) (PW_OFF_BAR(QK) + 2 * PW_ST * 8 + 1024)
#define PW_BUILDERS 96
#define PW6_BYTES 288

__device__ __forceinline__ void pw_setmaxnreg_dec40() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
}
__device__ __forceinline__ void pw_setmaxnreg_inc232() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
}
// decode.cu's h2f / i2f_rn and prefill.cu's gm_i2f, the same ops (an f16 ->
// f32 conversion is exact; the magic-number i2f is exact for |i| < 2^22).
__device__ __forceinline__ float pw_h2f(u32 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)h));
    return f;
}
__device__ __forceinline__ float pw_i2f_rn(int i) { return __int2float_rn(i); }
__device__ __forceinline__ float pw_gm_i2f(int i) {
    return __fsub_rn(__int_as_float(0x4B400000 + i), 12582912.0f);
}

template <int QK, bool EXACT>
__device__ __forceinline__ void probe_t(const TMap* mw, const TMap* mx, const u8* act, u64 astride,
                                        int rows, int n, int nb, void* out0, void* out1) {
    extern __shared__ u8 pw_dsm[];
    u32 dsz;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz < PROBE_W4A8_SMEM(QK)) __trap();
    constexpr int ST = PW_ST, STAGE = PW_STAGE(QK);
    u8* sm = (u8*)(((u64)pw_dsm + 1023) & ~(u64)1023);
    u64* full = (u64*)(sm + PW_OFF_BAR(QK));
    u64* empty = full + ST;
    const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127, w = t >> 5, l = t & 31;
    const int nblk = n / 256;
    const int rtiles = (rows + 127) / 128, ttiles = (nb + 63) / 64;
    const int ntiles = rtiles * ttiles;
    if (tid == 0) {
        for (int s = 0; s < ST; s++) {
            mbar_init(&full[s], 1 + PW_BUILDERS);
            mbar_init(&empty[s], 8); // lane 0 of each consumer warp
        }
        mbar_init_fence();
    }
    __syncthreads();

    if (wg == 0) {
        // ---------------- producer ----------------
        pw_setmaxnreg_dec40();
        int it = 0;
        for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
            const int z0 = (tile % ttiles) * 64, r0 = (tile / ttiles) * 128;
            for (int b = 0; b < nblk; b++, it++) {
                const int s = it % ST;
                u8* st = sm + s * STAGE;
                if (w == 0) {
                    if (l == 0) {
                        if (it >= ST) mbar_wait(&empty[s], ((it / ST) & 1) ^ 1);
                        if (QK == 4) {
                            mbar_expect(&full[s], PW_RAW + PW_ACT);
                            tma_load_2d(st + PW_ACT + PW_BS, mw, &full[s], 144 * b, r0);
                        } else {
                            mbar_expect(&full[s], 2 * PW_RAW + PW_ACT);
                            tma_load_2d(st + PW_ACT, mw, &full[s], PW6_BYTES * b, r0);
                            tma_load_2d(st + PW_ACT + PW_RAW, mw, &full[s], PW6_BYTES * b + 144, r0);
                        }
                        tma_load_2d(st, mx, &full[s], 256 * b, z0);
                        tma_load_2d(st + PW_ACT / 2, mx, &full[s], 256 * b + 128, z0);
                    }
                } else {
                    const int bt = t - 32;
                    if (it >= ST) mbar_wait(&empty[s], ((it / ST) & 1) ^ 1);
                    u8* bs = st + PW_ACT;
                    float* mdk = (float*)(sm + PW_OFF_META(QK) + s * PW_META);
                    int* mbs = (int*)(mdk + 64);
                    if (QK == 4 || EXACT) {
                        for (int zz = bt; zz < 64; zz += PW_BUILDERS) {
                            const int z = z0 + zz;
                            u32 lo[2] = {0, 0}, hi[2] = {0, 0};
                            int sum = 0;
                            float dk = 0.0f;
                            if (z < nb) {
                                // Scalar loads: b32 sits at 2 np + np / 64 bytes into the
                                // activation (decode.cu act_view), only 4-byte aligned
                                // (np 3584: +7224). An int4 load here was the 10-26 H100
                                // run's CUDA_ERROR_MISALIGNED_ADDRESS.
                                Act a = act_view(act + (u64)z * astride, n);
                                const int* b32 = a.b32 + b * 8;
#pragma unroll
                                for (int j = 0; j < 8; j++) {
                                    const int x = b32[j];
                                    lo[j >> 2] |= (u32)(x & 63) << (8 * (j & 3));
                                    hi[j >> 2] |= (u32)((x >> 6) & 255) << (8 * (j & 3));
                                    sum += x;
                                }
                                if (EXACT) dk = a.dk[b];
                            }
                            if (QK == 4) {
                                // Bytes 0..15 and 32..47 of the row (k32 steps 0 and 1);
                                // 16..31 and 48..63 zero.
                                *(uint4*)(bs + sw128(zz, 0)) = make_uint4(lo[0], lo[1], 0, 0);
                                *(uint4*)(bs + sw128(zz, 16)) = make_uint4(0, 0, 0, 0);
                                *(uint4*)(bs + sw128(zz, 32)) = make_uint4(hi[0], hi[1], 0, 0);
                                *(uint4*)(bs + sw128(zz, 48)) = make_uint4(0, 0, 0, 0);
                            }
                            if (EXACT) {
                                mdk[zz] = dk;
                                mbs[zz] = sum;
                            }
                        }
                        if (QK == 4) fence_proxy_async();
                    }
                    mbar_arrive(&full[s]);
                }
            }
        }
        return;
    }

    // ---------------- consumers ----------------
    pw_setmaxnreg_inc232();
    const int cw = wg - 1; // consumer warpgroup 0 / 1
    const int g = l >> 2, q = l & 3, rl = 16 * w + g;
    constexpr int NP = QK == 4 ? 2 : 3; // accumulator planes (Q4_K: lo, hi; Q6: digits)
    int dd[NP][32];
    int mlo[QK == 4 ? 32 : 1], mhi[QK == 4 ? 32 : 1];
    float acc[EXACT ? 32 : 1];
    u32 aa[NP][4][4], am[4];
#pragma unroll
    for (int i = 0; i < 32; i++) {
#pragma unroll
        for (int p = 0; p < NP; p++) dd[p][i] = 0;
        if constexpr (QK == 4) mlo[i] = 0, mhi[i] = 0;
    }
    int it = 0;
    bool pend = false; // skeleton: the previous block's stage not yet released
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int z0 = (tile % ttiles) * 64, r0 = (tile / ttiles) * 128;
        if constexpr (EXACT) {
#pragma unroll
            for (int i = 0; i < 32; i++) acc[i] = 0.0f;
        }
        for (int b = 0; b < nblk; b++, it++) {
            const int s = it % ST;
            const u8* st = sm + s * STAGE;
            const u32 b0 = smem_addr(st), bsa = smem_addr(st + PW_ACT);
            mbar_wait(&full[s], (it / ST) & 1);
            // Skeleton: accumulate over the tile's whole K (scale-d 0 on its
            // first block). Exact: every block from zero, drained below.
            const int first = EXACT ? 0 : b != 0;
            uint4 hd[2];
            u32 sc[2][2];
            float dw[2];
            const u8* raw;   // Q4_K: the row's block; Q6: half 0 (values 0..143)
            const u8* raw1;  // Q6: half 1 (values 144..255, digits, d)
            if constexpr (QK == 4) {
                raw = st + PW_ACT + PW_BS + (64 * cw + rl) * 144;
                raw1 = raw;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    hd[h] = *(const uint4*)(raw + 8 * h * 144);
                    sc[h][0] = hd[h].y & 0x3f3f3f3fu;
                    sc[h][1] = (hd[h].w & 0x0f0f0f0fu) | ((hd[h].y >> 2) & 0x30303030u);
                }
            } else {
                raw = st + PW_ACT + (64 * cw + rl) * 144;
                raw1 = raw + PW_RAW;
                if (EXACT) {
#pragma unroll
                    for (int h = 0; h < 2; h++) dw[h] = *(const float*)(raw1 + 8 * h * 144 + 136);
                }
            }
            u32 wv[2][2];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int sl = j & 3;
                wg_wait<3>(); // the group that last read slot sl (step j - 4) retired
                if (!EXACT && j == 3 && pend) {
                    // group 8 (it-1) + 7 retired: block it-1's stage is free.
                    if (l == 0) mbar_arrive(&empty[(it + ST - 1) % ST]);
                    pend = false;
                }
                if constexpr (QK == 4) {
                    const int p = j >> 1, o = j & 1;
                    if (j == 0) {
                        // M's A fragment: the rows' 8 mins in K lanes 0..7.
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            u32 m0 = hd[h].z & 0x3f3f3f3fu;
                            u32 m1 = ((hd[h].w >> 4) & 0x0f0f0f0fu) | ((hd[h].z >> 2) & 0x30303030u);
                            am[h] = q == 0 ? m0 : (q == 1 ? m1 : 0u);
                        }
                        am[2] = 0u;
                        am[3] = 0u;
                    }
                    if (o == 0) {
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            wv[h][0] = *(const u32*)(raw + 8 * h * 144 + 16 + 32 * p + 4 * q);
                            wv[h][1] = *(const u32*)(raw + 8 * h * 144 + 16 + 32 * p + 16 + 4 * q);
                        }
                    }
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        u32 scj = (sc[h][j >> 2] >> (8 * (j & 3))) & 255;
                        u32 v0 = (o ? wv[h][0] >> 4 : wv[h][0]) & 0x0f0f0f0fu;
                        u32 v1 = (o ? wv[h][1] >> 4 : wv[h][1]) & 0x0f0f0f0fu;
                        aa[0][sl][h] = v0 * (scj & 15);
                        aa[0][sl][2 + h] = v1 * (scj & 15);
                        aa[1][sl][h] = v0 * (scj >> 4);
                        aa[1][sl][2 + h] = v1 * (scj >> 4);
                    }
                } else {
                    // Values 32 j + 4 q (sub-block 2j) and + 16 (2j + 1): half 0
                    // below 144, half 1 from there (j compile-time).
                    const int K0 = 32 * j, K1 = 32 * j + 16;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const u8* r0p = raw + 8 * h * 144;
                        const u8* r1p = raw1 + 8 * h * 144;
                        u32 W[2];
                        W[0] = *(const u32*)((K0 < 144 ? r0p + K0 : r1p + (K0 - 144)) + 4 * q);
                        W[1] = *(const u32*)((K1 < 144 ? r0p + K1 : r1p + (K1 - 144)) + 4 * q);
#pragma unroll
                        for (int c = 0; c < 2; c++) {
                            const u32 Q = ((W[c] ^ 0x20202020u) & 0x3f3f3f3fu) - 0x20202020u;
                            const u32 R = 0x80808080u - 3u * Q; // A = Q e + R = Q (e - 3) + 0x80..
#pragma unroll
                            for (int p = 0; p < 3; p++) {
                                const u32 eb = r1p[112 + 8 * p + j];
                                const u32 e = c ? eb >> 4 : eb & 15u;
                                aa[p][sl][2 * c + h] = Q * e + R;
                            }
                        }
                    }
                }
#pragma unroll
                for (int p = 0; p < NP; p++) wg_pin_n64(dd[p]);
                if constexpr (QK == 4) {
                    wg_pin_n64(mlo);
                    wg_pin_n64(mhi);
                }
                wg_fence();
                const u64 db = wg_desc(b0 + (j >> 2) * (PW_ACT / 2)) + 2 * (j & 3);
#pragma unroll
                for (int p = 0; p < NP; p++) wg_mma_rs_n64_u8(dd[p], aa[p][sl], db, first | j);
                if constexpr (QK == 4) {
                    if (j == 0) {
                        wg_mma_rs_n64_u8(mlo, am, wg_desc(bsa), first);
                        wg_mma_rs_n64_u8(mhi, am, wg_desc(bsa) + 2, first);
                    }
                }
                wg_commit();
            }
            if constexpr (EXACT) {
                // The per-256 drain: every wgmma of the block retired, the
                // block's integers into the float chain (gemmh2_q4_k's /
                // gemm2_q6k8's ops, in block order).
                wg_wait<0>();
#pragma unroll
                for (int p = 0; p < NP; p++) wg_pin_n64(dd[p]);
                if constexpr (QK == 4) {
                    wg_pin_n64(mlo);
                    wg_pin_n64(mhi);
                }
                const float* mdk = (const float*)(sm + PW_OFF_META(QK) + s * PW_META);
                const int* mbs = (const int*)(mdk + 64);
                float dr[2], dmr[2];
                if constexpr (QK == 4) {
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        dr[h] = pw_h2f(hd[h].x & 0xffff);
                        dmr[h] = pw_h2f(hd[h].x >> 16);
                    }
                }
#pragma unroll
                for (int i = 0; i < 8; i++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int x = 4 * i + e, h = e >> 1, zl = 8 * i + 2 * q + (e & 1);
                        const float dk = mdk[zl];
                        if constexpr (QK == 4) {
                            const int S = 16 * dd[1][x] + dd[0][x];
                            const int M = 64 * mhi[x] + mlo[x];
                            const float v = __fsub_rn(__fmul_rn(dr[h], pw_i2f_rn(S)),
                                                      __fmul_rn(dmr[h], pw_gm_i2f(M)));
                            acc[x] = __fmaf_rn(dk, v, acc[x]);
                        } else {
                            const int S = (int)((u32)dd[0][x] + 8u * (u32)dd[1][x] +
                                                64u * (u32)dd[2][x] - 9344u * (u32)mbs[zl]);
                            acc[x] = __fmaf_rn(__fmul_rn(dk, dw[h]), pw_i2f_rn(S), acc[x]);
                        }
                    }
                __syncwarp();
                if (l == 0) mbar_arrive(&empty[s]); // the stage fully read
            } else {
                pend = true;
            }
        }
        // Tile end. Skeleton: drain once (the cheat: one drain per tile, not
        // per 256), release the last stage, store the integers. Exact: store.
        if constexpr (!EXACT) {
            wg_wait<0>();
#pragma unroll
            for (int p = 0; p < NP; p++) wg_pin_n64(dd[p]);
            if constexpr (QK == 4) {
                wg_pin_n64(mlo);
                wg_pin_n64(mhi);
            }
            if (pend) {
                if (l == 0) mbar_arrive(&empty[(it + ST - 1) % ST]);
                pend = false;
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int x = 4 * i + e;
                const int r = r0 + 64 * cw + rl + 8 * (e >> 1), z = z0 + 8 * i + 2 * q + (e & 1);
                if (r < rows && z < nb) {
                    const u64 o = (u64)z * rows + r;
                    if constexpr (EXACT) {
                        ((float*)out0)[o] = acc[x];
                    } else if constexpr (QK == 4) {
                        ((int*)out0)[o] = 16 * dd[1][x] + dd[0][x];
                        ((int*)out1)[o] = 64 * mhi[x] + mlo[x];
                    } else {
                        ((int*)out0)[o] =
                            (int)((u32)dd[0][x] + 8u * (u32)dd[1][x] + 64u * (u32)dd[2][x]);
                    }
                }
            }
    }
}

#define PROBE_KERNEL(NAME, QK, EXACT)                                                              \
    extern "C" __global__ void __launch_bounds__(384, 1)                                           \
        NAME(const __grid_constant__ TMap mw, const __grid_constant__ TMap mx,                     \
             const u8* __restrict__ act, u64 astride, int rows, int n, int nb, void* out0,         \
             void* out1) {                                                                         \
        probe_t<QK, EXACT>(&mw, &mx, act, astride, rows, n, nb, out0, out1);                       \
    }
PROBE_KERNEL(probe_w4a8_n64, 4, false)
PROBE_KERNEL(probe_w4a8x_n64, 4, true)
PROBE_KERNEL(probe_w6a8_n64, 6, false)
PROBE_KERNEL(probe_w6a8x_n64, 6, true)

#endif // __CUDA_ARCH_FEAT_SM90_ALL
