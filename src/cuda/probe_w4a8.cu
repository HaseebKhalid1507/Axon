
// ---------------------------------------------------------------------------
// TEST-ONLY (T3, Hopper W4A8 round 2 upper-bound probe). Never part of
// KERNELS: tests.rs compiles it as KERNELS + this file into a module of its
// own (its own cache key), so no product path can load it.
//
// probe_w4a8_n64: a warp-specialized, persistent Q4_K x Q8_K GEMM skeleton
// with INTEGER-ONLY accumulation and NO per-256 drain. Its results are not
// the GEMM (no d / dmin / dk); it writes the integer sums over all of K
//   S[z][r] = sum_b sum_j sc_bj * sumi_bj      (sumi = sum of q * x over sub-block j)
//   M[z][r] = sum_b sum_j m_bj * bsum_bj
// (wrapping int32), which the probe's bit test checks against the CPU. Its
// speed bounds what any exact kernel built on this structure can reach: the
// exact one does the same tensor work plus the per-256 float drain.
//
// CTA = 384 threads, 1 CTA / SM, grid = #SMs (persistent, tiles t =
// blockIdx.x + i * gridDim.x, tokens fastest):
// - warpgroup 0, producer (setmaxnreg 40): warp 0 lane 0 issues the TMA loads
//   of a stage (raw weights 128 rows x 144 B, activations 2 x {128 B, N}
//   swizzled); warps 1-3 build the stage's bsum-piece tile (B of the M
//   wgmmas: token row z, bytes 0..7 = the 8 sub-block sums' low 6 bits,
//   bytes 32..39 = their high parts as s8; zeros elsewhere) from global,
//   fence.proxy.async, arrive.
// - warpgroups 1, 2, consumers (setmaxnreg 232): rows 64 (wg - 1) .. +63 of
//   the tile x all N tokens. Per 256-block: 8 k32 steps, each unpacks its
//   lo / hi A fragments (4+2 scale split as gemmh2) into one of 4 rotating
//   register slots and issues wgmma lo + hi as one commit group, so the
//   unpack of step j+1 overlaps the wgmmas of step j..j-3; the M pair
//   (A = the row's 8 mins in K lanes 0..7) rides step 0's group. The stage
//   of block it-1 is released once step 3 of block it has waited (group
//   8 (it-1) + 7 retired).
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

#define PW_ST(N) ((N) == 64 ? 4 : 3)
#define PW_ACT(N) ((N) * 256)          // two 128-byte swizzled boxes of N rows
#define PW_BS(N) ((N) * 128)           // the bsum-piece tile, 128-byte swizzled
#define PW_RAW (128 * 144)             // raw Q4_K rows
#define PW_STAGE(N) (PW_ACT(N) + PW_BS(N) + PW_RAW)
#define PW_OFF_BAR(N) (PW_ST(N) * PW_STAGE(N))
#define PROBE_W4A8_SMEM(N) (PW_OFF_BAR(N) + 2 * PW_ST(N) * 8 + 1024)
#define PW_BUILDERS 96

__device__ __forceinline__ void pw_setmaxnreg_dec40() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
}
__device__ __forceinline__ void pw_setmaxnreg_inc232() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void pw_mma(int* d, const u32* a, u64 b, int acc) {
    if constexpr (N == 64) wg_mma_rs_n64_u8(d, a, b, acc);
    else wg_mma_rs_n128_u8(d, a, b, acc);
}
template <int N>
__device__ __forceinline__ void pw_pin(int* d) {
    if constexpr (N == 64) wg_pin_n64(d);
    else wg_pin_n128(d);
}

template <int N>
__device__ __forceinline__ void probe_w4a8_t(const TMap* mw, const TMap* mx, const u8* act,
                                             u64 astride, int rows, int n, int nb, int* outS,
                                             int* outM) {
    extern __shared__ u8 pw_dsm[];
    u32 dsz;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz < PROBE_W4A8_SMEM(N)) __trap();
    constexpr int ST = PW_ST(N);
    u8* sm = (u8*)(((u64)pw_dsm + 1023) & ~(u64)1023);
    u64* full = (u64*)(sm + PW_OFF_BAR(N));
    u64* empty = full + ST;
    const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127, w = t >> 5, l = t & 31;
    const int nblk = n / 256;
    const int rtiles = (rows + 127) / 128, ttiles = (nb + N - 1) / N;
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
            const int z0 = (tile % ttiles) * N, r0 = (tile / ttiles) * 128;
            for (int b = 0; b < nblk; b++, it++) {
                const int s = it % ST;
                u8* st = sm + s * PW_STAGE(N);
                if (w == 0) {
                    if (l == 0) {
                        if (it >= ST) mbar_wait(&empty[s], ((it / ST) & 1) ^ 1);
                        mbar_expect(&full[s], PW_RAW + PW_ACT(N));
                        tma_load_2d(st + PW_ACT(N) + PW_BS(N), mw, &full[s], 144 * b, r0);
                        tma_load_2d(st, mx, &full[s], 256 * b, z0);
                        tma_load_2d(st + PW_ACT(N) / 2, mx, &full[s], 256 * b + 128, z0);
                    }
                } else {
                    const int bt = t - 32;
                    if (it >= ST) mbar_wait(&empty[s], ((it / ST) & 1) ^ 1);
                    u8* bs = st + PW_ACT(N);
                    for (int zz = bt; zz < N; zz += PW_BUILDERS) {
                        const int z = z0 + zz;
                        u32 lo[2] = {0, 0}, hi[2] = {0, 0};
                        if (z < nb) {
                            // Scalar loads: b32 sits at 2 np + np / 64 bytes into the
                            // activation (decode.cu act_view), only 4-byte aligned
                            // (np 3584: +7224). An int4 load here was the 10-26 H100
                            // run's CUDA_ERROR_MISALIGNED_ADDRESS.
                            const int* b32 = act_view(act + (u64)z * astride, n).b32 + b * 8;
#pragma unroll
                            for (int j = 0; j < 8; j++) {
                                const int x = b32[j];
                                lo[j >> 2] |= (u32)(x & 63) << (8 * (j & 3));
                                hi[j >> 2] |= (u32)((x >> 6) & 255) << (8 * (j & 3));
                            }
                        }
                        // Bytes 0..15 and 32..47 of the row (k32 steps 0 and 1);
                        // 16..31 and 48..63 zero.
                        *(uint4*)(bs + sw128(zz, 0)) = make_uint4(lo[0], lo[1], 0, 0);
                        *(uint4*)(bs + sw128(zz, 16)) = make_uint4(0, 0, 0, 0);
                        *(uint4*)(bs + sw128(zz, 32)) = make_uint4(hi[0], hi[1], 0, 0);
                        *(uint4*)(bs + sw128(zz, 48)) = make_uint4(0, 0, 0, 0);
                    }
                    fence_proxy_async();
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
    int dlo[N / 2], dhi[N / 2], mlo[N / 2], mhi[N / 2];
    u32 alo[4][4], ahi[4][4], am[4];
#pragma unroll
    for (int i = 0; i < N / 2; i++) dlo[i] = 0, dhi[i] = 0, mlo[i] = 0, mhi[i] = 0;
    int it = 0;
    bool pend = false; // the previous block's stage not yet released
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int z0 = (tile % ttiles) * N, r0 = (tile / ttiles) * 128;
        for (int b = 0; b < nblk; b++, it++) {
            const int s = it % ST;
            const u8* st = sm + s * PW_STAGE(N);
            const u32 b0 = smem_addr(st), bsa = smem_addr(st + PW_ACT(N));
            const u8* raw = st + PW_ACT(N) + PW_BS(N) + (64 * cw + rl) * 144;
            mbar_wait(&full[s], (it / ST) & 1);
            uint4 hd[2];
            u32 sc[2][2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                hd[h] = *(const uint4*)(raw + 8 * h * 144);
                sc[h][0] = hd[h].y & 0x3f3f3f3fu;
                sc[h][1] = (hd[h].w & 0x0f0f0f0fu) | ((hd[h].y >> 2) & 0x30303030u);
            }
            const int first = b != 0; // scale-d 0 on a tile's first block
            u32 wv[2][2];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int sl = j & 3, p = j >> 1, o = j & 1;
                wg_wait<3>(); // the group that last read slot sl (step j - 4) retired
                if (j == 3 && pend) {
                    // group 8 (it-1) + 7 retired: block it-1's stage is free.
                    if (l == 0) mbar_arrive(&empty[(it + ST - 1) % ST]);
                    pend = false;
                }
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
                    alo[sl][h] = v0 * (scj & 15);
                    alo[sl][2 + h] = v1 * (scj & 15);
                    ahi[sl][h] = v0 * (scj >> 4);
                    ahi[sl][2 + h] = v1 * (scj >> 4);
                }
                pw_pin<N>(dlo);
                pw_pin<N>(dhi);
                pw_pin<N>(mlo);
                pw_pin<N>(mhi);
                wg_fence();
                const u64 db = wg_desc(b0 + (j >> 2) * (PW_ACT(N) / 2)) + 2 * (j & 3);
                pw_mma<N>(dlo, alo[sl], db, first | j);
                pw_mma<N>(dhi, ahi[sl], db, first | j);
                if (j == 0) {
                    pw_mma<N>(mlo, am, wg_desc(bsa), first);
                    pw_mma<N>(mhi, am, wg_desc(bsa) + 2, first);
                }
                wg_commit();
            }
            pend = true;
        }
        // Tile end: drain once (the cheat: one drain per tile, not per 256),
        // release the last stage, store the integers.
        wg_wait<0>();
        pw_pin<N>(dlo);
        pw_pin<N>(dhi);
        pw_pin<N>(mlo);
        pw_pin<N>(mhi);
        if (pend) {
            if (l == 0) mbar_arrive(&empty[(it + ST - 1) % ST]);
            pend = false;
        }
#pragma unroll
        for (int i = 0; i < N / 8; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int r = r0 + 64 * cw + rl + 8 * (e >> 1), z = z0 + 8 * i + 2 * q + (e & 1);
                if (r < rows && z < nb) {
                    outS[(u64)z * rows + r] = 16 * dhi[4 * i + e] + dlo[4 * i + e];
                    outM[(u64)z * rows + r] = 64 * mhi[4 * i + e] + mlo[4 * i + e];
                }
            }
    }
}

extern "C" __global__ void __launch_bounds__(384, 1)
probe_w4a8_n64(const __grid_constant__ TMap mw, const __grid_constant__ TMap mx,
               const u8* __restrict__ act, u64 astride, int rows, int n, int nb,
               int* __restrict__ outS, int* __restrict__ outM) {
    probe_w4a8_t<64>(&mw, &mx, act, astride, rows, n, nb, outS, outM);
}

#endif // __CUDA_ARCH_FEAT_SM90_ALL
