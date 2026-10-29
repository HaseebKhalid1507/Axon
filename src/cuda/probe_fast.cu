
// ---------------------------------------------------------------------------
// TEST-ONLY (WP-A1, 2026-10-29): the H100 probe of two `--numerics fast` GEMM
// candidates on the T3 skeleton (probe_w4a8.cu, feat/t3-w4a8-probe). Never
// part of KERNELS: tests.rs builds hopper.cu's helper prefix + this file into
// a module of its own (its own cache key), so no product path can load it.
// The results are not bit-exact by design (fast mode); each candidate is
// checked against a CPU reference within a tolerance (tests.rs a1_*).
//
// Both candidates share the T3 CTA: 384 threads, 1 CTA / SM, a persistent
// grid of #SMs, tiles of R rows x T tokens (tokens fastest), K in 256-blocks
// through an mbarrier ring:
// - warpgroup 0, producer (setmaxnreg 40): thread 0 issues every TMA load of
//   a stage (nothing is built by threads: both operands come by TMA);
// - warpgroups 1, 2, consumers (setmaxnreg 232): rows R/2 (wg - 1) .. of the
//   tile x all T tokens, f32 accumulators over the CTA's whole K range, one
//   epilogue per range.
// Two tile shapes per candidate, the same instructions: T3's 128 x 64 and a
// bigger one that halves an operand's L2 -> SM bytes per MAC (bw / T + bx / R:
// FP8 128 x 64 needs 8.2 TB/s at 350 TMAC/s, 128 x 128 5.5; W4A16 128 x 64
// 5.8 TB/s at 230, 256 x 64 4.0).
// Work schedule (a1_next, the same in producer and consumers), `ksplit`:
// - 1: data-parallel tiles (t = blockIdx.x + i gridDim.x), plain stores;
// - S >= 2: split-K, units (tile, part) with part fastest, K-blocks
//   [part nblk / S, (part + 1) nblk / S);
// - 0: stream-K, CTA c takes [c T / G, (c + 1) T / G) of the T = tiles x
//   nblk K-blocks in (tile, block) order, cut at tile edges.
// With ksplit != 1 every range adds its scaled partial into `out` with
// atomicAdd (red.global.add.f32: the arrival-order fixup fast mode allows);
// the host zeroes `out` first (a1_zero_f32, inside the timed region).
//
// (a) probe_fp8_128x64 / probe_fp8_128x128: SS wgmma.m64n64k32.f32.e4m3.e4m3,
//     T / 64 per k32 step per warpgroup. Stage = activations 2 x {128 B, T
//     tokens} + weights 2 x {128 B, 128 rows}, all SW128. Epilogue out = acc
//     * ws[row] * xs[token].
// (b) probe_w4a16_128x64 / probe_w4a16_256x64: RS wgmma.m64n64k16.f32.f16.f16,
//     R / 128 m64 row blocks per warpgroup. Weights are Q4_K repacked at upload
//     (tests.rs a1_q4f_pack): nibbles in the consumer threads' fragment order
//     (one uint4 per thread per 4 sub-blocks, no bank conflicts) and per-32
//     half2 (s, -z) = (d sc, -dmin m) rounded to fp16. Registers: LOP3 magic
//     (0x6400 | nibble) -> HSUB2 / HFMA2 -> exact q, then one HFMA2 q s - z.
//     Stage = activations 4 x {128 B, 64 tokens} SW128 (256 fp16 K) + R x 128 B
//     nibbles + R x 32 B scales.
// Every mbarrier wait traps after 4 s (a1_wait): a parity or count bug fails
// the launch fast instead of hanging the GPU until the stage's process limit.
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

// FP8, T tokens per tile (64, 128): stages, activation and weight bytes.
#define A1F_ST(T) ((T) == 64 ? 4 : 3)
#define A1F_X(T) ((T) * 256)
#define A1F_W (128 * 256)
#define A1F_STAGE(T) (A1F_X(T) + A1F_W)
#define A1F_SMEM(T) (A1F_ST(T) * A1F_STAGE(T) + 2 * A1F_ST(T) * 8 + 1024)
// W4A16, R rows per tile (128, 256).
#define A1H_ST(R) ((R) == 128 ? 4 : 3)
#define A1H_X (64 * 512)
#define A1H_N(R) ((R) * 128)
#define A1H_S(R) ((R) * 32)
#define A1H_STAGE(R) (A1H_X + A1H_N(R) + A1H_S(R))
#define A1H_SMEM(R) (A1H_ST(R) * A1H_STAGE(R) + 2 * A1H_ST(R) * 8 + 1024)

__device__ __forceinline__ void a1_setmaxnreg_dec40() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
}
__device__ __forceinline__ void a1_setmaxnreg_inc232() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
}

__device__ __forceinline__ u64 a1_ns() {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ bool a1_try(u64* bar, u32 parity) {
    u32 ok;
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ok)
        : "r"(smem_addr(bar)), "r"(parity)
        : "memory");
    return ok != 0;
}
// mbar_wait with a limit: 4 s of failed try_waits traps the launch (the host
// sees a launch error at its next sync). No printf here: a call inside the
// wgmma pipeline makes ptxas serialize every wgmma (C7510).
__device__ __forceinline__ void a1_wait(u64* bar, u32 parity) {
    if (a1_try(bar, parity)) return;
    const u64 t0 = a1_ns();
    while (!a1_try(bar, parity))
        if (a1_ns() - t0 > 4000000000ull) __trap();
}

struct A1Sched {
    int ksplit, nblk, ntiles, u;
    long long pos, end;
};
__device__ __forceinline__ A1Sched a1_sched(int ksplit, int nblk, int ntiles) {
    A1Sched s;
    s.ksplit = ksplit;
    s.nblk = nblk;
    s.ntiles = ntiles;
    s.u = blockIdx.x;
    const long long tot = (long long)ntiles * nblk;
    s.pos = tot * blockIdx.x / gridDim.x;
    s.end = tot * (blockIdx.x + 1) / gridDim.x;
    return s;
}
// This CTA's next (tile, K-block range [kb0, kb1)), never empty; false when done.
__device__ __forceinline__ bool a1_next(A1Sched& s, int& tile, int& kb0, int& kb1) {
    if (s.ksplit == 0) {
        if (s.pos >= s.end) return false;
        tile = (int)(s.pos / s.nblk);
        kb0 = (int)(s.pos % s.nblk);
        const long long left = s.end - s.pos;
        kb1 = left < (long long)(s.nblk - kb0) ? kb0 + (int)left : s.nblk;
        s.pos += kb1 - kb0;
        return true;
    }
    for (;;) {
        if (s.u >= s.ntiles * s.ksplit) return false;
        tile = s.u / s.ksplit;
        const int p = s.u % s.ksplit;
        kb0 = p * s.nblk / s.ksplit;
        kb1 = (p + 1) * s.nblk / s.ksplit;
        s.u += gridDim.x;
        if (kb1 > kb0) return true;
    }
}

// D (64 x 64 f32, wgmma fragment order) += A (e4m3, desc a) B^T (e4m3, desc b);
// D = A B^T instead if !acc. FP8 is K-major only (no transpose immediates).
__device__ __forceinline__ void a1_mma_e4m3_n64(float* d, u64 a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(a), "l"(b), "r"(acc));
}

// D (64 x 64 f32) += A (f16, registers a[4]: rows 16 w + g (+8), K 2q, 2q+1
// and 2q+8, 2q+9, the m16n8k16 A fragment) B^T (f16, desc b, K-major: trans-b
// 0); D = A B^T instead if !acc.
__device__ __forceinline__ void a1_mma_f16_rs_n64(float* d, const u32* a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
}

// Pin the 32 f32 accumulators (no reordering of their uses across the
// wgmma fence / wait: avoids ptxas serializing the wgmmas).
__device__ __forceinline__ void a1_pin(float* d) {
    asm volatile("" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) :: "memory");
}

// out[z rows + r] = v (one writer: ksplit 1) or += v (atomic, arrival order).
__device__ __forceinline__ void a1_put(float* out, u64 i, float v, int ksplit) {
    if (ksplit == 1) out[i] = v;
    else atomicAdd(out + i, v);
}

// (a) T tokens per tile: each consumer warpgroup 64 rows x T tokens, T / 64
// accumulators (token sub-tile h at + h 8 KiB in each activation box).
template <int T>
__device__ __forceinline__ void probe_fp8_t(const TMap* mw, const TMap* mx, const float* ws,
                                            const float* xs, int rows, int n, int nb, int ksplit,
                                            float* out) {
    constexpr int ST = A1F_ST(T), H = T / 64;
    extern __shared__ u8 a1_dsm[];
    u32 dsz;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz < A1F_SMEM(T)) __trap();
    u8* sm = (u8*)(((u64)a1_dsm + 1023) & ~(u64)1023);
    u64* full = (u64*)(sm + ST * A1F_STAGE(T));
    u64* empty = full + ST;
    const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127, w = t >> 5, l = t & 31;
    const int nblk = n / 256, ttiles = (nb + T - 1) / T, ntiles = ((rows + 127) / 128) * ttiles;
    if (tid == 0) {
        for (int s = 0; s < ST; s++) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], 8); // lane 0 of each consumer warp
        }
        mbar_init_fence();
    }
    __syncthreads();
    A1Sched sc = a1_sched(ksplit, nblk, ntiles);
    int tile, kb0, kb1;

    if (wg == 0) {
        // ---------------- producer: thread 0 only ----------------
        a1_setmaxnreg_dec40();
        if (tid == 0) {
            int it = 0;
            while (a1_next(sc, tile, kb0, kb1)) {
                const int z0 = (tile % ttiles) * T, r0 = (tile / ttiles) * 128;
                for (int kb = kb0; kb < kb1; kb++, it++) {
                    const int s = it % ST;
                    u8* st = sm + s * A1F_STAGE(T);
                    if (it >= ST) a1_wait(&empty[s], ((it / ST) & 1) ^ 1);
                    mbar_expect(&full[s], A1F_STAGE(T));
                    tma_load_2d(st, mx, &full[s], 256 * kb, z0);
                    tma_load_2d(st + A1F_X(T) / 2, mx, &full[s], 256 * kb + 128, z0);
                    tma_load_2d(st + A1F_X(T), mw, &full[s], 256 * kb, r0);
                    tma_load_2d(st + A1F_X(T) + A1F_W / 2, mw, &full[s], 256 * kb + 128, r0);
                }
            }
        }
        return;
    }

    // ---------------- consumers ----------------
    a1_setmaxnreg_inc232();
    const int cw = wg - 1, g = l >> 2, q = l & 3;
    float d[H][32];
#pragma unroll
    for (int h = 0; h < H; h++)
#pragma unroll
        for (int i = 0; i < 32; i++) d[h][i] = 0.f;
    int it = 0;
    while (a1_next(sc, tile, kb0, kb1)) {
        const int z0 = (tile % ttiles) * T, r0 = (tile / ttiles) * 128;
        for (int kb = kb0; kb < kb1; kb++, it++) {
            const int s = it % ST;
            const u32 sx = smem_addr(sm + s * A1F_STAGE(T)), sw = sx + A1F_X(T) + cw * 8192;
            a1_wait(&full[s], (it / ST) & 1);
#pragma unroll
            for (int h = 0; h < H; h++) a1_pin(d[h]);
            wg_fence();
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const u64 da = wg_desc(sw + (j >> 2) * (A1F_W / 2)) + 2 * (j & 3);
                const u64 db = wg_desc(sx + (j >> 2) * (A1F_X(T) / 2)) + 2 * (j & 3);
#pragma unroll
                for (int h = 0; h < H; h++) a1_mma_e4m3_n64(d[h], da, db + 512 * h, kb > kb0 || j > 0);
            }
            wg_commit();
            wg_wait<1>(); // the previous block's group retired: its stage is free
#pragma unroll
            for (int h = 0; h < H; h++) a1_pin(d[h]);
            if (kb > kb0 && l == 0) mbar_arrive(&empty[(it + ST - 1) % ST]);
        }
        wg_wait<0>();
#pragma unroll
        for (int h = 0; h < H; h++) a1_pin(d[h]);
        if (l == 0) mbar_arrive(&empty[(it + ST - 1) % ST]);
        const int rb = r0 + 64 * cw + 16 * w + g;
        const float w0 = rb < rows ? ws[rb] : 0.f, w1 = rb + 8 < rows ? ws[rb + 8] : 0.f;
#pragma unroll
        for (int h = 0; h < H; h++)
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = rb + 8 * (e >> 1), z = z0 + 64 * h + 8 * i + 2 * q + (e & 1);
                    if (r < rows && z < nb)
                        a1_put(out, (u64)z * rows + r, d[h][4 * i + e] * ((e >> 1) ? w1 : w0) * xs[z],
                               ksplit);
                }
    }
}

extern "C" __global__ void __launch_bounds__(384, 1)
probe_fp8_128x64(const __grid_constant__ TMap mw, const __grid_constant__ TMap mx,
                 const float* __restrict__ ws, const float* __restrict__ xs, int rows, int n, int nb,
                 int ksplit, float* __restrict__ out) {
    probe_fp8_t<64>(&mw, &mx, ws, xs, rows, n, nb, ksplit, out);
}
extern "C" __global__ void __launch_bounds__(384, 1)
probe_fp8_128x128(const __grid_constant__ TMap mw, const __grid_constant__ TMap mx,
                  const float* __restrict__ ws, const float* __restrict__ xs, int rows, int n, int nb,
                  int ksplit, float* __restrict__ out) {
    probe_fp8_t<128>(&mw, &mx, ws, xs, rows, n, nb, ksplit, out);
}

// (a & mask) | magic, one LOP3.
__device__ __forceinline__ u32 a1_lop3(u32 a, u32 mask, u32 magic) {
    u32 r;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(r) : "r"(a), "r"(mask), "r"(magic));
    return r;
}
__device__ __forceinline__ u32 a1_hfma2(u32 a, u32 b, u32 c) {
    u32 r;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
__device__ __forceinline__ u32 a1_hsub2(u32 a, u32 b) {
    u32 r;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
// 8 nibbles (n0 = bits 0..3 .. n7 = bits 28..31) and the sub-block's half2
// (s, -z) -> half2 (n0, n4), (n1, n5), (n2, n6), (n3, n7), each n s - z
// (one rounding: the q's are exact: 0x6400 | n = 1024 + n, 0x6400 | n << 4
// = 1024 + 16 n, (1024 + 16 n) / 16 - 64 = n).
__device__ __forceinline__ void a1_dq4(u32 v, u32 sz, u32* h) {
    const u32 s2 = __byte_perm(sz, 0, 0x1010), z2 = __byte_perm(sz, 0, 0x3232);
    const u32 M = 0x64006400u, I16 = 0x2c002c00u, N64 = 0xd400d400u;
    const u32 v2 = v >> 8;
    h[0] = a1_hfma2(a1_hsub2(a1_lop3(v, 0x000f000fu, M), M), s2, z2);
    h[1] = a1_hfma2(a1_hfma2(a1_lop3(v, 0x00f000f0u, M), I16, N64), s2, z2);
    h[2] = a1_hfma2(a1_hsub2(a1_lop3(v2, 0x000f000fu, M), M), s2, z2);
    h[3] = a1_hfma2(a1_hfma2(a1_lop3(v2, 0x00f000f0u, M), I16, N64), s2, z2);
}

// (b) R rows per tile: each consumer warpgroup R / 2 rows = MB = R / 128 m64
// blocks x 64 tokens. mn: the repacked nibbles, 128 B lines x (nblk rows_p),
// block kb's tile rows r0.. at line kb rows_p + r0 (no swizzle; rows_p a
// multiple of 256); msc: the half2 (s, -z), 32 B per (kb, row) the same way;
// mx: fp16 activations (2 n bytes x nb), SW128 boxes of 64 halves.
template <int R>
__device__ __forceinline__ void probe_w4a16_t(const TMap* mn, const TMap* msc, const TMap* mx,
                                              int rows, int rows_p, int n, int nb, int ksplit,
                                              float* out) {
    // NS register slots for the A fragments (MB x 8 u32 each): a slot is
    // rewritten once the group that read it (NS sub-blocks back) retired.
    // 4 slots at MB 1 (as T3); 2 at MB 2, where a group is already 4 wgmmas
    // deep (4 slots there spill: the consumers are allocated 168 registers).
    constexpr int ST = A1H_ST(R), MB = R / 128, NS = MB == 1 ? 4 : 2;
    extern __shared__ u8 a1_dsm[];
    u32 dsz;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz < A1H_SMEM(R)) __trap();
    u8* sm = (u8*)(((u64)a1_dsm + 1023) & ~(u64)1023);
    u64* full = (u64*)(sm + ST * A1H_STAGE(R));
    u64* empty = full + ST;
    const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127, w = t >> 5, l = t & 31;
    const int nblk = n / 256, ttiles = (nb + 63) / 64, ntiles = ((rows + R - 1) / R) * ttiles;
    if (tid == 0) {
        for (int s = 0; s < ST; s++) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], 8);
        }
        mbar_init_fence();
    }
    __syncthreads();
    A1Sched sc = a1_sched(ksplit, nblk, ntiles);
    int tile, kb0, kb1;

    if (wg == 0) {
        a1_setmaxnreg_dec40();
        if (tid == 0) {
            int it = 0;
            while (a1_next(sc, tile, kb0, kb1)) {
                const int z0 = (tile % ttiles) * 64, r0 = (tile / ttiles) * R;
                for (int kb = kb0; kb < kb1; kb++, it++) {
                    const int s = it % ST;
                    u8* st = sm + s * A1H_STAGE(R);
                    if (it >= ST) a1_wait(&empty[s], ((it / ST) & 1) ^ 1);
                    mbar_expect(&full[s], A1H_STAGE(R));
#pragma unroll
                    for (int i = 0; i < 4; i++)
                        tma_load_2d(st + i * (A1H_X / 4), mx, &full[s], 512 * kb + 128 * i, z0);
                    tma_load_2d(st + A1H_X, mn, &full[s], 0, kb * rows_p + r0);
                    tma_load_2d(st + A1H_X + A1H_N(R), msc, &full[s], 0, kb * rows_p + r0);
                }
            }
        }
        return;
    }

    a1_setmaxnreg_inc232();
    const int cw = wg - 1, g = l >> 2, q = l & 3;
    float d[MB][32];
    u32 af[NS][MB][8];
#pragma unroll
    for (int m = 0; m < MB; m++)
#pragma unroll
        for (int i = 0; i < 32; i++) d[m][i] = 0.f;
    int it = 0;
    bool pend = false; // the previous block's stage not yet released
    while (a1_next(sc, tile, kb0, kb1)) {
        const int z0 = (tile % ttiles) * 64, r0 = (tile / ttiles) * R;
        for (int kb = kb0; kb < kb1; kb++, it++) {
            const int s = it % ST;
            const u8* st = sm + s * A1H_STAGE(R);
            const u32 sx = smem_addr(st);
            a1_wait(&full[s], (it / ST) & 1);
            uint4 nv[MB][2];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int sl = j % NS;
                wg_wait<NS - 1>(); // the group that last read slot sl (sub-block j - NS) retired
                if (j == NS - 1 && pend) {
                    // group 8 (it-1) + 7 retired: block it-1's stage is free.
                    if (l == 0) mbar_arrive(&empty[(it + ST - 1) % ST]);
                    pend = false;
                }
#pragma unroll
                for (int m = 0; m < MB; m++) {
                    // Row block m: the warp's 16-row group (R / 32) cw + 4 m + w
                    // (2 KiB each); this thread's uint4 h (h = 2 rr + u: row g
                    // + 8 rr, sub-blocks 4 u ..) at (32 h + lane) * 16.
                    const int grp = (R / 32) * cw + 4 * m + w;
                    const uint4* nib = (const uint4*)(st + A1H_X + grp * 2048) + l;
                    const u32* scl = (const u32*)(st + A1H_X + A1H_N(R)) + (16 * grp + g) * 8;
                    if ((j & 3) == 0) {
                        nv[m][0] = nib[32 * (j >> 2)];
                        nv[m][1] = nib[32 * (2 + (j >> 2))];
                    }
#pragma unroll
                    for (int rr = 0; rr < 2; rr++) {
                        const u32 v = (j & 3) == 0   ? nv[m][rr].x
                                      : (j & 3) == 1 ? nv[m][rr].y
                                      : (j & 3) == 2 ? nv[m][rr].z
                                                     : nv[m][rr].w;
                        u32 hv[4];
                        a1_dq4(v, scl[64 * rr + j], hv);
                        af[sl][m][rr] = hv[0];     // k16 step 0: K 2q, 2q+1
                        af[sl][m][2 + rr] = hv[1]; //             K 2q+8, 2q+9
                        af[sl][m][4 + rr] = hv[2]; // k16 step 1
                        af[sl][m][6 + rr] = hv[3];
                    }
                }
#pragma unroll
                for (int m = 0; m < MB; m++) a1_pin(d[m]);
                wg_fence();
                // Sub-block j = halves 32 j .. of the block: box j / 2, + 64 B (j odd).
                const u64 db = wg_desc(sx + (j >> 1) * (A1H_X / 4)) + 4 * (j & 1);
#pragma unroll
                for (int m = 0; m < MB; m++) {
                    a1_mma_f16_rs_n64(d[m], &af[sl][m][0], db, kb > kb0 || j > 0);
                    a1_mma_f16_rs_n64(d[m], &af[sl][m][4], db + 2, 1);
                }
                wg_commit();
            }
            pend = true;
        }
        wg_wait<0>();
#pragma unroll
        for (int m = 0; m < MB; m++) a1_pin(d[m]);
        if (pend) {
            if (l == 0) mbar_arrive(&empty[(it + ST - 1) % ST]);
            pend = false;
        }
#pragma unroll
        for (int m = 0; m < MB; m++) {
            const int rb = r0 + (R / 2) * cw + 64 * m + 16 * w + g;
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = rb + 8 * (e >> 1), z = z0 + 8 * i + 2 * q + (e & 1);
                    if (r < rows && z < nb) a1_put(out, (u64)z * rows + r, d[m][4 * i + e], ksplit);
                }
        }
    }
}

extern "C" __global__ void __launch_bounds__(384, 1)
probe_w4a16_128x64(const __grid_constant__ TMap mn, const __grid_constant__ TMap msc,
                   const __grid_constant__ TMap mx, int rows, int rows_p, int n, int nb, int ksplit,
                   float* __restrict__ out) {
    probe_w4a16_t<128>(&mn, &msc, &mx, rows, rows_p, n, nb, ksplit, out);
}
extern "C" __global__ void __launch_bounds__(384, 1)
probe_w4a16_256x64(const __grid_constant__ TMap mn, const __grid_constant__ TMap msc,
                   const __grid_constant__ TMap mx, int rows, int rows_p, int n, int nb, int ksplit,
                   float* __restrict__ out) {
    probe_w4a16_t<256>(&mn, &msc, &mx, rows, rows_p, n, nb, ksplit, out);
}

// out[0 .. n) = 0 (the split-K / stream-K launches' output, before them).
extern "C" __global__ void a1_zero_f32(float* __restrict__ out, u64 n) {
    const u64 n4 = n / 4, step = (u64)gridDim.x * blockDim.x;
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n4; i += step)
        ((float4*)out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (blockIdx.x == 0 && threadIdx.x < n - 4 * n4) out[4 * n4 + threadIdx.x] = 0.f;
}

// Ceilings (as hopper.cu wg_peak_t / wg_peak_rs_t): each warpgroup chains
// iters x 4 wgmmas over a 128-byte K tile in static smem into f32
// accumulators. a1_peak_fp8_ss_n64: SS m64n64k32 e4m3 (the probe_fp8_n64
// instruction); a1_peak_f16_rs_n64: RS m64n64k16 f16 (probe_w4a16_n64's).
// Operand bytes are masked so no value is NaN / inf. out keeps d live.
extern "C" __global__ void __launch_bounds__(384, 1) a1_peak_fp8_ss_n64(int iters, int* out) {
    __shared__ __align__(1024) u8 sa[64 * 128];
    __shared__ __align__(1024) u8 sb[64 * 128];
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
        sa[i] = (u8)(i * 2654435761u >> 13) & 0xb7;
        sb[i] = (u8)(i * 2246822519u >> 11) & 0xb7;
    }
    fence_proxy_async();
    __syncthreads();
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0.f;
    const u64 da = wg_desc(smem_addr(sa)), db = wg_desc(smem_addr(sb));
    for (int it = 0; it < iters; it++) {
        a1_pin(d);
        wg_fence();
#pragma unroll
        for (int k = 0; k < 4; k++) a1_mma_e4m3_n64(d, da + 2 * k, db + 2 * k, 1);
        wg_commit();
        wg_wait<1>();
    }
    wg_wait<0>();
    a1_pin(d);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) s += d[i];
    if (s == 1234.5f) out[blockIdx.x] = 1;
}
extern "C" __global__ void __launch_bounds__(384, 1) a1_peak_f16_rs_n64(int iters, int* out) {
    __shared__ __align__(1024) u8 sb[64 * 128];
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) sb[i] = (u8)(i * 2246822519u >> 11) & 0xbb;
    fence_proxy_async();
    __syncthreads();
    u32 a[4];
#pragma unroll
    for (int k = 0; k < 4; k++) a[k] = (threadIdx.x * 2654435761u + k * 0x9e3779b9u) & 0xbbffbbffu;
    float d[32];
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0.f;
    const u64 db = wg_desc(smem_addr(sb));
    for (int it = 0; it < iters; it++) {
        a1_pin(d);
        wg_fence();
#pragma unroll
        for (int k = 0; k < 4; k++) a1_mma_f16_rs_n64(d, a, db + 2 * k, 1);
        wg_commit();
        wg_wait<1>();
    }
    wg_wait<0>();
    a1_pin(d);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i++) s += d[i];
    if (s == 1234.5f) out[blockIdx.x] = 1;
}

#endif // __CUDA_ARCH_FEAT_SM90_ALL
