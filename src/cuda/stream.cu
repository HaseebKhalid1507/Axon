
// ---------------------------------------------------------------------------
// Streaming (src/cuda/qwen2/stream.rs): the CPU layers of a split model are
// held repacked for the CPU's kernels (src/quant/r8.rs, r8q.rs: 8 rows
// interleaved per block). For a big prefill batch they are copied to the GPU
// as they are and turned back into GGUF rows here, the exact inverse of the
// repack (every byte of the original is kept, so this is a permutation; for
// Q4_K the 12 packed 6-bit scale bytes are re-packed from the unpacked ones).

// One output byte o (0..bb) of row r's block from the 8-row block t.
__device__ __forceinline__ u8 unperm_byte(int qt, const u8* t, int r, int o) {
    if (qt == QT_Q4_K) {
        if (o < 2) return t[2 * r + o];
        if (o < 4) return t[16 + 2 * r + o - 2];
        if (o < 16) {
            int q = o - 4;
#define SC(j) t[32 + 16 * ((j) / 2) + 2 * r + (j) % 2]
#define MN(j) t[96 + 16 * ((j) / 2) + 2 * r + (j) % 2]
            if (q < 4) return (SC(q) & 63) | ((SC(q + 4) >> 4) << 6);
            if (q < 8) return (MN(q - 4) & 63) | ((MN(q) >> 4) << 6);
            return (SC(q - 4) & 15) | ((MN(q - 4) & 15) << 4);
#undef SC
#undef MN
        }
        int x = o - 16, c = x >> 5, g = (x & 31) >> 2, i = x & 3;
        return t[160 + 32 * (8 * c + g) + 4 * r + i];
    }
    if (qt == QT_Q6_K) {
        if (o < 128) {
            int h = o >> 6, l = o & 63, hi = l >> 5;
            l &= 31;
            return t[144 + 96 * (8 * h + (l >> 2)) + 32 * hi + 4 * r + (l & 3)];
        }
        if (o < 192) {
            int x = o - 128, h = x >> 5, l = x & 31;
            return t[144 + 96 * (8 * h + (l >> 2)) + 64 + 4 * r + (l & 3)];
        }
        if (o < 208) {
            int x = o - 192;
            return t[16 + 16 * (x >> 1) + 2 * r + (x & 1)];
        }
        return t[2 * r + o - 208];
    }
    if (qt == QT_Q8_0) {
        if (o < 2) return t[2 * r + o];
        int k = o - 2;
        return t[16 + 32 * (k >> 2) + 4 * r + (k & 3)] ^ 0x80;
    }
    // Q5_0
    if (o < 2) return t[2 * r + o];
    if (o < 6) {
        int qb = o - 2;
        u8 b = 0;
        for (int s = 0; s < 8; s++) {
            int bit = 8 * qb + s, v = bit >> 2, i = bit & 3;
            b |= ((t[144 + 4 * r + i] >> v) & 1) << s;
        }
        return b;
    }
    int x = o - 6;
    return t[16 + 32 * (x >> 2) + 4 * r + (x & 3)];
}

// The 8-row blocks tile0.. (up to UNP_TILES of them, nt) of a matrix of
// nblk blocks per row, staged in shared memory, out as GGUF rows: one
// 16-bit word per thread step (every block size is even, and rows start at
// even offsets), QT and its sizes known at compile time.
#define UNP_TILES 4
template <int QT, int BB8, int BB>
__device__ __forceinline__ void unperm_tiles(const u8* __restrict__ src, u8* __restrict__ dst,
                                             u64 row_bytes, int nblk, u64 tile0, int nt, u32* s) {
    const u32* w = (const u32*)(src + tile0 * BB8);
    for (int i = threadIdx.x; i < nt * (BB8 / 4); i += blockDim.x) s[i] = w[i];
    __syncthreads();
    const u8* sm = (const u8*)s;
    const int words = 8 * BB / 2;
    for (int j = threadIdx.x; j < nt * words; j += blockDim.x) {
        int t = j / words, x = j % words, r = x / (BB / 2), o = 2 * (x % (BB / 2));
        u64 tile = tile0 + t;
        u64 g = tile / nblk;
        int b = (int)(tile % nblk);
        const u8* tb = sm + t * BB8;
        unsigned short v = unperm_byte(QT, tb, r, o) | (unperm_byte(QT, tb, r, o + 1) << 8);
        *(unsigned short*)(dst + (8 * g + r) * row_bytes + (u64)b * BB + o) = v;
    }
    __syncthreads();
}

// GGUF rows from r8/r8q groups: src holds ngroups groups of nblk 8-row
// blocks of bb8 bytes (a multiple of 4; src 4-byte aligned), dst gets rows
// 8g..8g+8 of row_bytes each (block b of row at b * bb, bb = bb8 / 8 except
// Q4_K: 144 from 1184; dst and row_bytes even). Grid-stride over groups of
// UNP_TILES blocks, 128 threads.
extern "C" __global__ void unpermute_r8(int qt, const u8* __restrict__ src, u8* __restrict__ dst,
                                        u64 row_bytes, int nblk, int bb8, int bb, u64 ngroups) {
    __shared__ u32 s[UNP_TILES * 1680 / 4];
    u64 total = ngroups * (u64)nblk;
    for (u64 t0 = (u64)blockIdx.x * UNP_TILES; t0 < total; t0 += (u64)gridDim.x * UNP_TILES) {
        int nt = (int)min((u64)UNP_TILES, total - t0);
        switch (qt) {
        case QT_Q4_K: unperm_tiles<QT_Q4_K, 1184, 144>(src, dst, row_bytes, nblk, t0, nt, s); break;
        case QT_Q6_K: unperm_tiles<QT_Q6_K, 1680, 210>(src, dst, row_bytes, nblk, t0, nt, s); break;
        case QT_Q8_0: unperm_tiles<QT_Q8_0, 272, 34>(src, dst, row_bytes, nblk, t0, nt, s); break;
        default: unperm_tiles<QT_Q5_0, 176, 22>(src, dst, row_bytes, nblk, t0, nt, s); break;
        }
    }
}
