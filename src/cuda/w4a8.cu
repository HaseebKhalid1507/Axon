
// ---------------------------------------------------------------------------
// w4a8.cu (docs/design/w4a8-prefill.md): the exact W4A8 prefill GEMMs on
// sm_90a, the r11 probe's exact variants (#165) made production. Same bits as
// today's kernels, by construction and by test: every output equals
// gemmh2_q4_k (modes 0 / 1, bias), gemmh2_q4_k_gu (silu(gate) * up) and
// gemm2(s)_q6k8 (modes 0 / 1, bias) on the same inputs: an exact integer
// block sum per 256 values, then the same float chain over the blocks in
// order (gemv_ref's order). Only how the integers are formed differs.
// Routed by batch.rs (`w4a8_route`): AXON_W4A8=1, feed passes of nb >= T.
//
// Structure (CTA 384 threads, 1 CTA / SM, persistent: grid <= #SMs, tiles of
// 128 rows x 64 tokens, tokens fastest; a W8_ST-stage TMA ring streaming
// across tiles):
// - warpgroup 0, producer (setmaxnreg 40): warp 0 lane 0 issues a stage's
//   TMA loads (raw rows; activations 2 x {128 B, 64 rows} SW128). Warps 1-3
//   (96 builders) write the stage's per-token dk and 256-sums (meta), and:
//   Q4_K the M wgmma's B tile (64 tokens x the 8 b32 pieces, lo6 / hi s8);
//   Q6 the rows' scale digits (below) from the Q6K8 copy in global memory.
// - warpgroups 1, 2, consumers (setmaxnreg 232): 64 rows each x 64 tokens.
//   Per 256-block 8 k32 steps; each unpacks its A fragments into one of 4
//   rotating register slots and issues its RS wgmmas (u8 x s8) as one
//   commit group. After the block: wait, the block's integers into the
//   float chain (the per-256 drain), release the stage.
//   Plain: rows 64 (wg - 1)..; gu: consumer 1 the gate's 64 rows, consumer 2
//   the same rows of up, gate's chain handed over through smem at the tile.
//
// Q4_K: gemmh2's 4 + 2 scale split (two u8 passes, S = 16 hi + lo) and M by
// a wgmma pair (A = the rows' 8 mins in K lanes 0..7, B = the bsum pieces,
// M = 64 hi + lo). Drain: v = d S - dmin M (M by gm_i2f, as gemmh2),
// acc = fma(dk, v, acc).
// Q6 (the Q6K8 prefill copy, prefill.cu: 256 int8 values q - 32, 16 int8
// scales, d as f32, 288 B a block; read as two 144-byte TMA boxes): each
// scale sc = d0 + 8 d1 + 64 d2, d in [-3, 4] (balanced radix 8; every int8
// has one), computed by the builders into smem as e = d + 3 nibbles. Per
// 32-bit word of 4 values W: Q = ((W ^ 0x20..) & 0x3f..) - 0x20.. (= sum q_k
// 256^k mod 2^32), A_p = Q e_p + (0x80808080 - 3 Q) = bytes q_k d_p + 128 in
// [0, 252] (no byte overflows since q d in [-128, 124]): sum_k A_pk x_k =
// d_p sum q x + 128 sum x, so S_b = P0 + 8 P1 + 64 P2 - 9344 sum_b x, exactly
// gemm2's sum_j sc_j sum q x. Drain: acc = fma(dk dw, float(S_b), acc).
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

#define W8_ST 4
#define W8_ACT (64 * 256)  // two 128-byte swizzled boxes of 64 token rows
#define W8_BS (64 * 128)   // Q4_K: the bsum-piece tile, 128-byte swizzled
#define W8_RAW (128 * 144) // 128 rows x 144 B: a Q4_K block, half a Q6K8 block
#define W8_DIG (128 * 24)  // Q6: 128 rows x 3 planes x 8 bytes of digit nibbles
#define W8_STAGE(QK) ((QK) == 4 ? W8_ACT + W8_BS + W8_RAW : W8_ACT + 2 * W8_RAW + W8_DIG)
#define W8_META 512        // per stage: dk[64] (f32), the 256-sums [64] (i32)
#define W8_XCH (32 * 128 * 4) // gu: gate's chain, 32 floats x 128 threads
#define W8_OFF_META(QK) (W8_ST * W8_STAGE(QK))
#define W8_OFF_XCH(QK) (W8_OFF_META(QK) + W8_ST * W8_META)
#define W8_OFF_BAR(QK, GU) (W8_OFF_XCH(QK) + ((GU) ? W8_XCH : 0))
#define W4A8_SMEM(QK, GU) (W8_OFF_BAR(QK, GU) + 2 * W8_ST * 8 + 1024)
#define W8_BUILDERS 96
#define W8_Q6K8 288

__device__ __forceinline__ void w8_setmaxnreg_dec40() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
}
__device__ __forceinline__ void w8_setmaxnreg_inc232() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
}
// The two consumer warpgroups (256 threads), named barrier id.
__device__ __forceinline__ void w8_cbar(int id) {
    asm volatile("bar.sync %0, 256;" ::"r"(id) : "memory");
}

// The 3 balanced radix-8 digits of an int8 scale, as e = d + 3 (0..7).
__device__ __forceinline__ void w8_digits(int sc, u32* e0, u32* e1, u32* e2) {
    int a = (sc + 3) & 7, s1 = (sc + 3 - a) >> 3;
    int b = (s1 + 3) & 7, s2 = (s1 + 3 - b) >> 3;
    *e0 = (u32)a, *e1 = (u32)b, *e2 = (u32)(s2 + 3);
}

// QK 4 (Q4_K) or 6 (Q6K8); GU: the fused gate (mw) + up (mw2) with
// silu(gate) * up out (QK 4 only). w8 / rb8: the Q6K8 copy in global memory
// (its scales, for the builders' digits).
template <int QK, bool GU>
__device__ __forceinline__ void w4a8_t(const TMap* mw, const TMap* mw2, const TMap* mx,
                                       const u8* act, u64 astride, int rows, int n, int nb,
                                       const float* bias, int mode, float* out, u64 ostride,
                                       const u8* w8, u64 rb8) {
    extern __shared__ u8 w8_dsm[];
    u32 dsz;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz < W4A8_SMEM(QK, GU)) __trap(); // the host's size out of date
    constexpr int ST = W8_ST, STAGE = W8_STAGE(QK);
    constexpr int RT = GU ? 64 : 128; // rows of a tile (gu: of each of gate and up)
    u8* sm = (u8*)(((u64)w8_dsm + 1023) & ~(u64)1023);
    u64* full = (u64*)(sm + W8_OFF_BAR(QK, GU));
    u64* empty = full + ST;
    const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127, w = t >> 5, l = t & 31;
    const int nblk = n / 256;
    const int rtiles = (rows + RT - 1) / RT, ttiles = (nb + 63) / 64;
    const int ntiles = rtiles * ttiles;
    if (tid == 0) {
        for (int s = 0; s < ST; s++) {
            mbar_init(&full[s], 1 + W8_BUILDERS);
            mbar_init(&empty[s], 8); // lane 0 of each consumer warp
        }
        mbar_init_fence();
    }
    __syncthreads();

    if (wg == 0) {
        // ---------------- producer ----------------
        w8_setmaxnreg_dec40();
        int it = 0;
        for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
            const int z0 = (tile % ttiles) * 64, r0 = (tile / ttiles) * RT;
            for (int b = 0; b < nblk; b++, it++) {
                const int s = it % ST;
                u8* st = sm + s * STAGE;
                if (w == 0) {
                    if (l == 0) {
                        if (it >= ST) mbar_wait(&empty[s], ((it / ST) & 1) ^ 1);
                        if (QK == 4) {
                            mbar_expect(&full[s], W8_RAW + W8_ACT);
                            u8* raw = st + W8_ACT + W8_BS;
                            if (GU) {
                                tma_load_2d(raw, mw, &full[s], 144 * b, r0);
                                tma_load_2d(raw + 64 * 144, mw2, &full[s], 144 * b, r0);
                            } else {
                                tma_load_2d(raw, mw, &full[s], 144 * b, r0);
                            }
                        } else {
                            mbar_expect(&full[s], 2 * W8_RAW + W8_ACT);
                            tma_load_2d(st + W8_ACT, mw, &full[s], W8_Q6K8 * b, r0);
                            tma_load_2d(st + W8_ACT + W8_RAW, mw, &full[s], W8_Q6K8 * b + 144, r0);
                        }
                        tma_load_2d(st, mx, &full[s], 256 * b, z0);
                        tma_load_2d(st + W8_ACT / 2, mx, &full[s], 256 * b + 128, z0);
                    }
                } else {
                    const int bt = t - 32;
                    if (it >= ST) mbar_wait(&empty[s], ((it / ST) & 1) ^ 1);
                    float* mdk = (float*)(sm + W8_OFF_META(QK) + s * W8_META);
                    int* mbs = (int*)(mdk + 64);
                    u8* bs = st + W8_ACT;
                    for (int zz = bt; zz < 64; zz += W8_BUILDERS) {
                        const int z = z0 + zz;
                        u32 lo[2] = {0, 0}, hi[2] = {0, 0};
                        int sum = 0;
                        float dk = 0.0f;
                        if (z < nb) {
                            // Scalar loads: b32 is only 4-byte aligned (decode.cu
                            // act_view: 2 np + np / 64 bytes in).
                            Act a = act_view(act + (u64)z * astride, n);
                            const int* b32 = a.b32 + b * 8;
#pragma unroll
                            for (int j = 0; j < 8; j++) {
                                const int x = b32[j];
                                lo[j >> 2] |= (u32)(x & 63) << (8 * (j & 3));
                                hi[j >> 2] |= (u32)((x >> 6) & 255) << (8 * (j & 3));
                                sum += x;
                            }
                            dk = a.dk[b];
                        }
                        if (QK == 4) {
                            *(uint4*)(bs + sw128(zz, 0)) = make_uint4(lo[0], lo[1], 0, 0);
                            *(uint4*)(bs + sw128(zz, 16)) = make_uint4(0, 0, 0, 0);
                            *(uint4*)(bs + sw128(zz, 32)) = make_uint4(hi[0], hi[1], 0, 0);
                            *(uint4*)(bs + sw128(zz, 48)) = make_uint4(0, 0, 0, 0);
                        }
                        mdk[zz] = dk;
                        mbs[zz] = sum;
                    }
                    if (QK == 6) {
                        // The rows' scale digits: plane p's byte j = e of
                        // sub-block 2j (low nibble), 2j + 1 (high).
                        u8* dig = st + W8_ACT + 2 * W8_RAW;
                        for (int rr = bt; rr < 128; rr += W8_BUILDERS) {
                            const int r = r0 + rr;
                            uint4 sc = make_uint4(0, 0, 0, 0);
                            if (r < rows)
                                sc = *(const uint4*)(w8 + (u64)r * rb8 + (u64)b * W8_Q6K8 + 256);
                            const u32 sw[4] = {sc.x, sc.y, sc.z, sc.w};
                            u32 pl[3][2] = {{0, 0}, {0, 0}, {0, 0}};
#pragma unroll
                            for (int j = 0; j < 16; j++) {
                                u32 e[3];
                                w8_digits((int)(signed char)(sw[j >> 2] >> (8 * (j & 3))), &e[0], &e[1],
                                          &e[2]);
                                const int sh = 4 * (j & 15) % 32;
#pragma unroll
                                for (int p = 0; p < 3; p++) pl[p][j >> 3] |= e[p] << sh;
                            }
                            uint2* dp = (uint2*)(dig + rr * 24);
#pragma unroll
                            for (int p = 0; p < 3; p++) dp[p] = make_uint2(pl[p][0], pl[p][1]);
                        }
                    }
                    if (QK == 4) fence_proxy_async(); // the bsum tile, for the wgmmas
                    mbar_arrive(&full[s]);
                }
            }
        }
        return;
    }

    // ---------------- consumers ----------------
    w8_setmaxnreg_inc232();
    const int cw = wg - 1; // consumer warpgroup 0 / 1
    const int g = l >> 2, q = l & 3, rl = 16 * w + g;
    constexpr int NP = QK == 4 ? 2 : 3;
    int dd[NP][32];
    int mlo[QK == 4 ? 32 : 1], mhi[QK == 4 ? 32 : 1];
    float acc[32];
    u32 aa[NP][4][4], am[4];
#pragma unroll
    for (int i = 0; i < 32; i++) {
#pragma unroll
        for (int p = 0; p < NP; p++) dd[p][i] = 0;
        if constexpr (QK == 4) mlo[i] = 0, mhi[i] = 0;
    }
    float* xch = (float*)(sm + W8_OFF_XCH(QK));
    int it = 0;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int z0 = (tile % ttiles) * 64, r0 = (tile / ttiles) * RT;
#pragma unroll
        for (int i = 0; i < 32; i++) acc[i] = 0.0f;
        for (int b = 0; b < nblk; b++, it++) {
            const int s = it % ST;
            const u8* st = sm + s * STAGE;
            const u32 b0 = smem_addr(st), bsa = smem_addr(st + W8_ACT);
            mbar_wait(&full[s], (it / ST) & 1);
            uint4 hd[2];
            u32 sc[2][2];
            float dw[2];
            const u8* raw; // Q4_K: the row's block; Q6: half 0 (values 0..143)
            const u8* raw1; // Q6: half 1 (values 144..255, scales, d)
            const u8* dig = st + W8_ACT + 2 * W8_RAW + (64 * cw + rl) * 24;
            if constexpr (QK == 4) {
                raw = st + W8_ACT + W8_BS + (64 * cw + rl) * 144;
                raw1 = raw;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    hd[h] = *(const uint4*)(raw + 8 * h * 144);
                    sc[h][0] = hd[h].y & 0x3f3f3f3fu;
                    sc[h][1] = (hd[h].w & 0x0f0f0f0fu) | ((hd[h].y >> 2) & 0x30303030u);
                }
            } else {
                raw = st + W8_ACT + (64 * cw + rl) * 144;
                raw1 = raw + W8_RAW;
#pragma unroll
                for (int h = 0; h < 2; h++) dw[h] = *(const float*)(raw1 + 8 * h * 144 + 128);
            }
            u32 wv[2][2];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int sl = j & 3;
                wg_wait<3>(); // the group that last read slot sl (step j - 4) retired
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
                    // below 144, half 1 from there.
                    const int K0 = 32 * j, K1 = 32 * j + 16;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const u8* r0p = raw + 8 * h * 144;
                        const u8* r1p = raw1 + 8 * h * 144;
                        const u8* dg = dig + 8 * h * 24;
                        u32 W[2];
                        W[0] = *(const u32*)((K0 < 144 ? r0p + K0 : r1p + (K0 - 144)) + 4 * q);
                        W[1] = *(const u32*)((K1 < 144 ? r0p + K1 : r1p + (K1 - 144)) + 4 * q);
#pragma unroll
                        for (int c = 0; c < 2; c++) {
                            const u32 Q = ((W[c] ^ 0x20202020u) & 0x3f3f3f3fu) - 0x20202020u;
                            const u32 R = 0x80808080u - 3u * Q; // A = Q e + R = Q (e - 3) + 0x80..
#pragma unroll
                            for (int p = 0; p < 3; p++) {
                                const u32 eb = dg[8 * p + j];
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
                const u64 db = wg_desc(b0 + (j >> 2) * (W8_ACT / 2)) + 2 * (j & 3);
#pragma unroll
                for (int p = 0; p < NP; p++) wg_mma_rs_n64_u8(dd[p], aa[p][sl], db, j);
                if constexpr (QK == 4) {
                    if (j == 0) {
                        wg_mma_rs_n64_u8(mlo, am, wg_desc(bsa), 0);
                        wg_mma_rs_n64_u8(mhi, am, wg_desc(bsa) + 2, 0);
                    }
                }
                wg_commit();
            }
            // The per-256 drain: every wgmma of the block retired, its
            // integers into the float chain, in block order.
            wg_wait<0>();
#pragma unroll
            for (int p = 0; p < NP; p++) wg_pin_n64(dd[p]);
            if constexpr (QK == 4) {
                wg_pin_n64(mlo);
                wg_pin_n64(mhi);
            }
            const float* mdk = (const float*)(sm + W8_OFF_META(QK) + s * W8_META);
            const int* mbs = (const int*)(mdk + 64);
            float dr[2], dmr[2];
            if constexpr (QK == 4) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    dr[h] = h2f(hd[h].x & 0xffff);
                    dmr[h] = h2f(hd[h].x >> 16);
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
                        const float v = __fsub_rn(__fmul_rn(dr[h], i2f_rn(S)),
                                                  __fmul_rn(dmr[h], gm_i2f(M)));
                        acc[x] = __fmaf_rn(dk, v, acc[x]);
                    } else {
                        const int S = (int)((u32)dd[0][x] + 8u * (u32)dd[1][x] +
                                            64u * (u32)dd[2][x] - 9344u * (u32)mbs[zl]);
                        acc[x] = __fmaf_rn(__fmul_rn(dk, dw[h]), i2f_rn(S), acc[x]);
                    }
                }
            __syncwarp();
            if (l == 0) mbar_arrive(&empty[s]); // the stage fully read
        }
        // Tile end: the stores.
        if constexpr (GU) {
            // out = silu(gate) * up, gemmh2_q4_k_gu's ops on the same floats.
            if (cw == 0) {
#pragma unroll
                for (int i = 0; i < 32; i++) xch[i * 128 + t] = acc[i];
            }
            w8_cbar(1);
            if (cw == 1) {
#pragma unroll
                for (int i = 0; i < 8; i++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int r = r0 + rl + 8 * (e >> 1), z = z0 + 8 * i + 2 * q + (e & 1);
                        const float gv = xch[(4 * i + e) * 128 + t];
                        if (r < rows && z < nb)
                            out[(u64)z * ostride + r] =
                                __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), acc[4 * i + e]);
                    }
            }
            w8_cbar(2); // xch read before gate's next tile writes it
        } else {
            // Every load the stores need first, then the stores (gemm2's epilogue).
            float pre[32];
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = r0 + 64 * cw + rl + 8 * (e >> 1), z = z0 + 8 * i + 2 * q + (e & 1);
                    pre[4 * i + e] = 0.0f;
                    if (mode != 0 && r < rows && z < nb)
                        pre[4 * i + e] = gm_pre(bias, mode, out, (u64)z * ostride, r);
                }
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = r0 + 64 * cw + rl + 8 * (e >> 1), z = z0 + 8 * i + 2 * q + (e & 1);
                    if (r < rows && z < nb)
                        gm_put(acc[4 * i + e], pre[4 * i + e], bias, mode, out, (u64)z * ostride, r);
                }
        }
    }
}

// Arguments as gemmh2_q4_k's (+ unused w8 / rb8); grid <= #SMs x 384 threads,
// W4A8_SMEM(4, false) bytes of dynamic smem.
extern "C" __global__ void __launch_bounds__(384, 1)
w4a8_q4_k(const __grid_constant__ TMap mw, const __grid_constant__ TMap mx,
          const u8* __restrict__ act, u64 astride, int rows, int n, int nb,
          const float* __restrict__ bias, int mode, float* __restrict__ out, u64 ostride) {
    w4a8_t<4, false>(&mw, &mw, &mx, act, astride, rows, n, nb, bias, mode, out, ostride, nullptr, 0);
}

// As gemmh2_q4_k_gu (rows: of each of gate and up; tensor maps of 64-row
// boxes); W4A8_SMEM(4, true) bytes.
extern "C" __global__ void __launch_bounds__(384, 1)
w4a8_q4_k_gu(const __grid_constant__ TMap mw, const __grid_constant__ TMap mw2,
             const __grid_constant__ TMap mx, const u8* __restrict__ act, u64 astride, int rows,
             int n, int nb, float* __restrict__ out, u64 ostride) {
    w4a8_t<4, true>(&mw, &mw2, &mx, act, astride, rows, n, nb, nullptr, 0, out, ostride, nullptr, 0);
}

// The Q6K8 copy (mw: its rows, 288 B a block, boxes of 144 B x 128 rows; w8 /
// rb8 the same bytes and row stride, for the builders' scale reads), else as
// gemm2_q6k8 (bias, mode, out, ostride); W4A8_SMEM(6, false) bytes.
extern "C" __global__ void __launch_bounds__(384, 1)
w4a8_q6(const __grid_constant__ TMap mw, const __grid_constant__ TMap mx,
        const u8* __restrict__ act, u64 astride, int rows, int n, int nb,
        const float* __restrict__ bias, int mode, float* __restrict__ out, u64 ostride,
        const u8* __restrict__ w8, u64 rb8) {
    w4a8_t<6, false>(&mw, &mw, &mx, act, astride, rows, n, nb, bias, mode, out, ostride, w8, rb8);
}

#endif // __CUDA_ARCH_FEAT_SM90_ALL
