
// ---------------------------------------------------------------------------
// Hopper (sm_90a: H100/H200/H20/GH200) only: TMA, mbarriers and wgmma.
// These instructions exist only for the arch-specific target, so the whole
// section compiles only there (__CUDA_ARCH_FEAT_SM90_ALL, which NVRTC
// defines for --gpu-architecture=sm_90a); for every other target it is
// empty and the module's other kernels are unchanged. The host picks sm_90a
// for compute capability 9.0 (mod.rs pick_target) and looks these kernels
// up only when Module::hopper() says they are there.
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

// A CUtensorMap (cuTensorMapEncodeTiled), passed by value as a
// __grid_constant__ kernel parameter (mod.rs Arg::Blob128).
struct __align__(64) TMap {
    u64 v[16];
};

__device__ __forceinline__ u32 smem_addr(const void* p) {
    return (u32)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_init(u64* bar, u32 count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(count)
                 : "memory");
}

// Make the barrier inits visible to the async proxy (TMA) before use; then
// a __syncthreads publishes them to the other threads.
__device__ __forceinline__ void mbar_init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// Arrive on bar, announcing tx bytes the TMA copies will complete.
__device__ __forceinline__ void mbar_expect(u64* bar, u32 tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(bar)),
                 "r"(tx)
                 : "memory");
}

// Plain arrive (count 1, no tx).
__device__ __forceinline__ void mbar_arrive(u64* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(bar)) : "memory");
}

// Wait until the barrier's phase with this parity has completed.
__device__ __forceinline__ void mbar_wait(u64* bar, u32 parity) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" ::"r"(smem_addr(bar)),
        "r"(parity)
        : "memory");
}

// TMA: the box of map at (c0 innermost, c1) into smem dst; completes tx
// bytes on bar.
__device__ __forceinline__ void tma_load_2d(void* dst, const TMap* map, u64* bar, int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_addr(dst)),
        "l"((u64)map), "r"(c0), "r"(c1), "r"(smem_addr(bar))
        : "memory");
}

// Generic-proxy smem writes (st.shared) become visible to the async proxy
// (wgmma operands, TMA) only after this fence.
__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

// wgmma smem matrix descriptor of a K-major operand in the 128-byte swizzle
// layout (what TMA's SWIZZLE_128B writes): rows of 128 bytes (128 int8 K
// values), 16-byte chunk c of row r at chunk c ^ (r & 7), 8-row groups
// 1024 bytes apart (SBO), the tile 1024-byte aligned. A k32 step within the
// 128-byte row is +32 bytes on the start address.
__device__ __forceinline__ u64 wg_desc(u32 saddr) {
    return (u64)((saddr & 0x3ffff) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) |
           ((u64)1 << 62);
}

// Byte offset of (row r, byte k < 128) in a 128-byte swizzled tile.
__device__ __forceinline__ u32 sw128(int r, int k) {
    return (u32)(r * 128 + ((((k >> 4) ^ r) & 7) << 4) + (k & 15));
}

// D (64 x 64 s32, wgmma fragment order) += A (s8, desc a) B^T (s8, desc b);
// D = A B^T instead if !acc.
__device__ __forceinline__ void wg_mma_n64_s8(int* d, u64 a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(a), "l"(b), "r"(acc));
}

// D (64 x 64 s32, wgmma fragment order) += A (u8, desc a) B^T (s8, desc b);
// D = A B^T instead if !acc.
__device__ __forceinline__ void wg_mma_n64_u8(int* d, u64 a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(a), "l"(b), "r"(acc));
}

// Pin the 32 accumulator registers (no reordering of their uses across
// the wgmma fence/wait: avoids ptxas serializing the wgmmas).
__device__ __forceinline__ void wg_pin_n64(int* d) {
    asm volatile("" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]) :: "memory");
}

// D (64 x 128 s32, wgmma fragment order) += A (s8, desc a) B^T (s8, desc b);
// D = A B^T instead if !acc.
__device__ __forceinline__ void wg_mma_n128_s8(int* d, u64 a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
        : "l"(a), "l"(b), "r"(acc));
}

// D (64 x 128 s32, wgmma fragment order) += A (u8, desc a) B^T (s8, desc b);
// D = A B^T instead if !acc.
__device__ __forceinline__ void wg_mma_n128_u8(int* d, u64 a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
        : "l"(a), "l"(b), "r"(acc));
}

// Pin the 64 accumulator registers (no reordering of their uses across
// the wgmma fence/wait: avoids ptxas serializing the wgmmas).
__device__ __forceinline__ void wg_pin_n128(int* d) {
    asm volatile("" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]) :: "memory");
}

// D (64 x 256 s32, wgmma fragment order) += A (s8, desc a) B^T (s8, desc b);
// D = A B^T instead if !acc.
__device__ __forceinline__ void wg_mma_n256_s8(int* d, u64 a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]), "+r"(d[64]), "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), "+r"(d[68]), "+r"(d[69]), "+r"(d[70]), "+r"(d[71]), "+r"(d[72]), "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), "+r"(d[76]), "+r"(d[77]), "+r"(d[78]), "+r"(d[79]), "+r"(d[80]), "+r"(d[81]), "+r"(d[82]), "+r"(d[83]), "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]), "+r"(d[88]), "+r"(d[89]), "+r"(d[90]), "+r"(d[91]), "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]), "+r"(d[96]), "+r"(d[97]), "+r"(d[98]), "+r"(d[99]), "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]), "+r"(d[104]), "+r"(d[105]), "+r"(d[106]), "+r"(d[107]), "+r"(d[108]), "+r"(d[109]), "+r"(d[110]), "+r"(d[111]), "+r"(d[112]), "+r"(d[113]), "+r"(d[114]), "+r"(d[115]), "+r"(d[116]), "+r"(d[117]), "+r"(d[118]), "+r"(d[119]), "+r"(d[120]), "+r"(d[121]), "+r"(d[122]), "+r"(d[123]), "+r"(d[124]), "+r"(d[125]), "+r"(d[126]), "+r"(d[127])
        : "l"(a), "l"(b), "r"(acc));
}

// D (64 x 256 s32, wgmma fragment order) += A (u8, desc a) B^T (s8, desc b);
// D = A B^T instead if !acc.
__device__ __forceinline__ void wg_mma_n256_u8(int* d, u64 a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]), "+r"(d[64]), "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), "+r"(d[68]), "+r"(d[69]), "+r"(d[70]), "+r"(d[71]), "+r"(d[72]), "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), "+r"(d[76]), "+r"(d[77]), "+r"(d[78]), "+r"(d[79]), "+r"(d[80]), "+r"(d[81]), "+r"(d[82]), "+r"(d[83]), "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]), "+r"(d[88]), "+r"(d[89]), "+r"(d[90]), "+r"(d[91]), "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]), "+r"(d[96]), "+r"(d[97]), "+r"(d[98]), "+r"(d[99]), "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]), "+r"(d[104]), "+r"(d[105]), "+r"(d[106]), "+r"(d[107]), "+r"(d[108]), "+r"(d[109]), "+r"(d[110]), "+r"(d[111]), "+r"(d[112]), "+r"(d[113]), "+r"(d[114]), "+r"(d[115]), "+r"(d[116]), "+r"(d[117]), "+r"(d[118]), "+r"(d[119]), "+r"(d[120]), "+r"(d[121]), "+r"(d[122]), "+r"(d[123]), "+r"(d[124]), "+r"(d[125]), "+r"(d[126]), "+r"(d[127])
        : "l"(a), "l"(b), "r"(acc));
}

// Pin the 128 accumulator registers (no reordering of their uses across
// the wgmma fence/wait: avoids ptxas serializing the wgmmas).
__device__ __forceinline__ void wg_pin_n256(int* d) {
    asm volatile("" : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]), "+r"(d[64]), "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), "+r"(d[68]), "+r"(d[69]), "+r"(d[70]), "+r"(d[71]), "+r"(d[72]), "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), "+r"(d[76]), "+r"(d[77]), "+r"(d[78]), "+r"(d[79]), "+r"(d[80]), "+r"(d[81]), "+r"(d[82]), "+r"(d[83]), "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]), "+r"(d[88]), "+r"(d[89]), "+r"(d[90]), "+r"(d[91]), "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]), "+r"(d[96]), "+r"(d[97]), "+r"(d[98]), "+r"(d[99]), "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]), "+r"(d[104]), "+r"(d[105]), "+r"(d[106]), "+r"(d[107]), "+r"(d[108]), "+r"(d[109]), "+r"(d[110]), "+r"(d[111]), "+r"(d[112]), "+r"(d[113]), "+r"(d[114]), "+r"(d[115]), "+r"(d[116]), "+r"(d[117]), "+r"(d[118]), "+r"(d[119]), "+r"(d[120]), "+r"(d[121]), "+r"(d[122]), "+r"(d[123]), "+r"(d[124]), "+r"(d[125]), "+r"(d[126]), "+r"(d[127]) :: "memory");
}

// D (64 x 64 s32) += A (u8, registers a[4]: rows 16 w + g (+8), K 4t.. and
// 16 + 4t.., the m16n8k32 A fragment) B^T (s8, desc b); D = A B^T if !acc.
__device__ __forceinline__ void wg_mma_rs_n64_u8(int* d, const u32* a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
}

// D (64 x 128 s32) += A (u8, registers a[4]: rows 16 w + g (+8), K 4t.. and
// 16 + 4t.., the m16n8k32 A fragment) B^T (s8, desc b); D = A B^T if !acc.
__device__ __forceinline__ void wg_mma_rs_n128_u8(int* d, const u32* a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
}

// D (64 x 256 s32) += A (u8, registers a[4]: rows 16 w + g (+8), K 4t.. and
// 16 + 4t.., the m16n8k32 A fragment) B^T (s8, desc b); D = A B^T if !acc.
__device__ __forceinline__ void wg_mma_rs_n256_u8(int* d, const u32* a, u64 b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]), "+r"(d[64]), "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), "+r"(d[68]), "+r"(d[69]), "+r"(d[70]), "+r"(d[71]), "+r"(d[72]), "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), "+r"(d[76]), "+r"(d[77]), "+r"(d[78]), "+r"(d[79]), "+r"(d[80]), "+r"(d[81]), "+r"(d[82]), "+r"(d[83]), "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]), "+r"(d[88]), "+r"(d[89]), "+r"(d[90]), "+r"(d[91]), "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]), "+r"(d[96]), "+r"(d[97]), "+r"(d[98]), "+r"(d[99]), "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]), "+r"(d[104]), "+r"(d[105]), "+r"(d[106]), "+r"(d[107]), "+r"(d[108]), "+r"(d[109]), "+r"(d[110]), "+r"(d[111]), "+r"(d[112]), "+r"(d[113]), "+r"(d[114]), "+r"(d[115]), "+r"(d[116]), "+r"(d[117]), "+r"(d[118]), "+r"(d[119]), "+r"(d[120]), "+r"(d[121]), "+r"(d[122]), "+r"(d[123]), "+r"(d[124]), "+r"(d[125]), "+r"(d[126]), "+r"(d[127])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
}

// Test (gpu_hopper_tma_wgmma): one TMA + wgmma round trip, bit-checked on
// the host. One warpgroup (128 threads); dynamic smem HOPPER_CHECK_SMEM.
// - ma: s8 or u8 matrix A, box {128 bytes, 64 rows}, SWIZZLE_128B, loaded
//   at (ka, ra); mb: s8 matrix B (tokens x K), box {128, 64}, SWIZZLE_128B,
//   at (kb, tb). D (64 x 64 s32, row-major) = A_box B_box^T over the 128 K
//   bytes: four m64n64k32 wgmmas from smem descriptors.
// - mraw: the Q4_K weight shape, box {144 bytes, 128 rows}, no swizzle,
//   loaded at (0, rr) and copied out to raw (128 x 144; rows past the
//   matrix come back zero).
// - ag non-null: A's box comes from ag (arows x ak bytes) written by the
//   threads into the swizzled layout (sw128, gemmh's producer path)
//   instead of by TMA.
#define HOPPER_CHECK_SMEM (1024 + 8192 + 8192 + 18432 + 8)
template <bool U8>
__device__ __forceinline__ void hopper_check_t(const TMap* ma, const TMap* mb, const TMap* mraw,
                                               int ka, int ra, int kb, int tb, int rr,
                                               const u8* ag, int ak, int arows, int* D,
                                               u8* raw) {
    extern __shared__ u8 hc_dsm[];
    u8* sm = (u8*)(((u64)hc_dsm + 1023) & ~(u64)1023);
    u8 *sa = sm, *sb = sm + 8192, *sr = sm + 16384;
    u64* bar = (u64*)(sm + 16384 + 18432);
    int t = threadIdx.x;
    if (t == 0) {
        mbar_init(bar, 1);
        mbar_init_fence();
    }
    __syncthreads();
    if (t == 0) {
        mbar_expect(bar, (ag ? 0 : 8192) + 8192 + 18432);
        if (!ag) tma_load_2d(sa, ma, bar, ka, ra);
        tma_load_2d(sb, mb, bar, kb, tb);
        tma_load_2d(sr, mraw, bar, 0, rr);
    }
    if (ag) {
        // A written by the threads (generic proxy) in the swizzled layout,
        // as gemmh's unpacked weights: sw128 + fence.proxy.async.
        for (int i = t; i < 64 * 8; i += 128) {
            int r = i >> 3, c = i & 7;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (ra + r < arows) v = *(const uint4*)(ag + (u64)(ra + r) * ak + ka + 16 * c);
            *(uint4*)(sa + sw128(r, 16 * c)) = v;
        }
        fence_proxy_async();
        __syncthreads();
    }
    mbar_wait(bar, 0);
    for (int i = t; i < 18432; i += 128) raw[i] = sr[i];
    int d[32];
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0;
    u64 da = wg_desc(smem_addr(sa)), db = wg_desc(smem_addr(sb));
    wg_pin_n64(d);
    wg_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) {
        if (U8)
            wg_mma_n64_u8(d, da + 2 * k, db + 2 * k, k);
        else
            wg_mma_n64_s8(d, da + 2 * k, db + 2 * k, k);
    }
    wg_commit();
    wg_wait<0>();
    wg_pin_n64(d);
    int w = t >> 5, l = t & 31, r0 = 16 * w + l / 4;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int c = 8 * i + 2 * (l & 3);
        D[r0 * 64 + c] = d[4 * i];
        D[r0 * 64 + c + 1] = d[4 * i + 1];
        D[(r0 + 8) * 64 + c] = d[4 * i + 2];
        D[(r0 + 8) * 64 + c + 1] = d[4 * i + 3];
    }
}

// A as s8 (hopper_check_s8) or u8 (hopper_check_u8).
extern "C" __global__ void __launch_bounds__(128)
hopper_check_s8(const __grid_constant__ TMap ma, const __grid_constant__ TMap mb,
                const __grid_constant__ TMap mraw, int ka, int ra, int kb, int tb, int rr,
                const u8* __restrict__ ag, int ak, int arows, int* __restrict__ D,
                u8* __restrict__ raw) {
    hopper_check_t<false>(&ma, &mb, &mraw, ka, ra, kb, tb, rr, ag, ak, arows, D, raw);
}
extern "C" __global__ void __launch_bounds__(128)
hopper_check_u8(const __grid_constant__ TMap ma, const __grid_constant__ TMap mb,
                const __grid_constant__ TMap mraw, int ka, int ra, int kb, int tb, int rr,
                const u8* __restrict__ ag, int ak, int arows, int* __restrict__ D,
                u8* __restrict__ raw) {
    hopper_check_t<true>(&ma, &mb, &mraw, ka, ra, kb, tb, rr, ag, ak, arows, D, raw);
}

// Test (gpu_hopper_tma_wgmma): RS wgmma (A from registers). A: 64 rows x
// 128 bytes u8 at ag (row stride ak, rows ra.., zero past arows), each
// thread's m16n8k32 A fragments loaded straight from global (warp w rows
// 16 w + g and + 8, K 4t.. and 16 + 4t.. of each k32 step); B by TMA as in
// hopper_check (mb at (kb, tb)). D (64 x 64 s32, row-major) = A B^T.
extern "C" __global__ void __launch_bounds__(128)
hopper_check_rs(const __grid_constant__ TMap mb, int kb, int tb, const u8* __restrict__ ag,
                int ak, int ra, int arows, int* __restrict__ D) {
    extern __shared__ u8 rs_dsm[];
    u8* sm = (u8*)(((u64)rs_dsm + 1023) & ~(u64)1023);
    u8* sb = sm;
    u64* bar = (u64*)(sm + 8192);
    int t = threadIdx.x, w = t >> 5, l = t & 31, g = l >> 2, q = l & 3;
    if (t == 0) {
        mbar_init(bar, 1);
        mbar_init_fence();
    }
    __syncthreads();
    if (t == 0) {
        mbar_expect(bar, 8192);
        tma_load_2d(sb, &mb, bar, kb, tb);
    }
    u32 a[4][4];
#pragma unroll
    for (int k = 0; k < 4; k++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int r = ra + 16 * w + g + 8 * (e & 1), c = 32 * k + 16 * (e >> 1) + 4 * q;
            a[k][e] = r < arows ? *(const u32*)(ag + (u64)r * ak + c) : 0u;
        }
    mbar_wait(bar, 0);
    int d[32];
#pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0;
    u64 db = wg_desc(smem_addr(sb));
    wg_pin_n64(d);
    wg_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) wg_mma_rs_n64_u8(d, a[k], db + 2 * k, k);
    wg_commit();
    wg_wait<0>();
    wg_pin_n64(d);
    int r0 = 16 * w + g;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int c = 8 * i + 2 * q;
        D[r0 * 64 + c] = d[4 * i];
        D[r0 * 64 + c + 1] = d[4 * i + 1];
        D[(r0 + 8) * 64 + c] = d[4 * i + 2];
        D[(r0 + 8) * 64 + c + 1] = d[4 * i + 3];
    }
}

// Benchmark (cuda_hopper_bench): wgmma throughput from smem descriptors,
// each warpgroup of the block (blockDim / 128 of them) chaining iters x 4
// m64nNk32 into one accumulator over a 128-byte K tile. A (64 x 128) and B
// (256 x 128) are pseudo-random bytes in static smem, shared by the
// warpgroups. out[blockIdx.x] keeps the results live.
template <int N, bool U8>
__device__ __forceinline__ void wg_peak_t(int iters, int* out) {
    __shared__ __align__(1024) u8 sa[64 * 128];
    __shared__ __align__(1024) u8 sb[256 * 128];
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) sa[i] = (u8)(i * 2654435761u >> 13);
    for (int i = threadIdx.x; i < 256 * 128; i += blockDim.x) sb[i] = (u8)(i * 2246822519u >> 11);
    fence_proxy_async();
    __syncthreads();
    int d[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; i++) d[i] = 0;
    u64 da = wg_desc(smem_addr(sa)), db = wg_desc(smem_addr(sb));
    for (int it = 0; it < iters; it++) {
        if (N == 64) wg_pin_n64(d);
        if (N == 128) wg_pin_n128(d);
        if (N == 256) wg_pin_n256(d);
        wg_fence();
#pragma unroll
        for (int k = 0; k < 4; k++) {
            if (N == 64) U8 ? wg_mma_n64_u8(d, da + 2 * k, db + 2 * k, 1) : wg_mma_n64_s8(d, da + 2 * k, db + 2 * k, 1);
            if (N == 128) U8 ? wg_mma_n128_u8(d, da + 2 * k, db + 2 * k, 1) : wg_mma_n128_s8(d, da + 2 * k, db + 2 * k, 1);
            if (N == 256) U8 ? wg_mma_n256_u8(d, da + 2 * k, db + 2 * k, 1) : wg_mma_n256_s8(d, da + 2 * k, db + 2 * k, 1);
        }
        wg_commit();
        wg_wait<1>();
    }
    wg_wait<0>();
    if (N == 64) wg_pin_n64(d);
    if (N == 128) wg_pin_n128(d);
    if (N == 256) wg_pin_n256(d);
    int s = 0;
#pragma unroll
    for (int i = 0; i < N / 2; i++) s ^= d[i];
    if (s == 0x12345678) out[blockIdx.x] = s;
}

extern "C" __global__ void __launch_bounds__(384, 1) wg_peak_n64(int iters, int* out) {
    wg_peak_t<64, false>(iters, out);
}
extern "C" __global__ void __launch_bounds__(384, 1) wg_peak_n128(int iters, int* out) {
    wg_peak_t<128, false>(iters, out);
}
extern "C" __global__ void __launch_bounds__(384, 1) wg_peak_n256(int iters, int* out) {
    wg_peak_t<256, false>(iters, out);
}
extern "C" __global__ void __launch_bounds__(384, 1) wg_peak_n256u(int iters, int* out) {
    wg_peak_t<256, true>(iters, out);
}

// Benchmark (cuda_hopper_bench): wg_peak with A (u8) from registers (RS
// wgmma: only B is read from smem), as a register-unpacking GEMM would.
template <int N>
__device__ __forceinline__ void wg_peak_rs_t(int iters, int* out) {
    __shared__ __align__(1024) u8 sb[256 * 128];
    for (int i = threadIdx.x; i < 256 * 128; i += blockDim.x) sb[i] = (u8)(i * 2246822519u >> 11);
    fence_proxy_async();
    __syncthreads();
    u32 a[4];
#pragma unroll
    for (int k = 0; k < 4; k++) a[k] = (threadIdx.x * 2654435761u + k * 0x9e3779b9u) & 0x7f7f7f7fu;
    int d[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; i++) d[i] = 0;
    u64 db = wg_desc(smem_addr(sb));
    for (int it = 0; it < iters; it++) {
        if (N == 64) wg_pin_n64(d);
        if (N == 128) wg_pin_n128(d);
        if (N == 256) wg_pin_n256(d);
        wg_fence();
#pragma unroll
        for (int k = 0; k < 4; k++) {
            if (N == 64) wg_mma_rs_n64_u8(d, a, db + 2 * k, 1);
            if (N == 128) wg_mma_rs_n128_u8(d, a, db + 2 * k, 1);
            if (N == 256) wg_mma_rs_n256_u8(d, a, db + 2 * k, 1);
        }
        wg_commit();
        wg_wait<1>();
    }
    wg_wait<0>();
    if (N == 64) wg_pin_n64(d);
    if (N == 128) wg_pin_n128(d);
    if (N == 256) wg_pin_n256(d);
    int s = 0;
#pragma unroll
    for (int i = 0; i < N / 2; i++) s ^= d[i];
    if (s == 0x12345678) out[blockIdx.x] = s;
}

extern "C" __global__ void __launch_bounds__(384, 1) wg_peak_rs_n64(int iters, int* out) {
    wg_peak_rs_t<64>(iters, out);
}
extern "C" __global__ void __launch_bounds__(384, 1) wg_peak_rs_n128(int iters, int* out) {
    wg_peak_rs_t<128>(iters, out);
}
extern "C" __global__ void __launch_bounds__(384, 1) wg_peak_rs_n256(int iters, int* out) {
    wg_peak_rs_t<256>(iters, out);
}

// ---------------------------------------------------------------------------
// gemmh_q4_k: the Q4_K x Q8_K batched product on wgmma (S2 v1). gemv_ref's
// canonical order: per output and 256-K block the exact integers S and M,
// then v = d S - dmin M (d * i2f(S), dmin * i2f(M), fsub) and acc = fma(dk,
// v, acc) over the blocks in order from +0; gm_store at the end.
// S = sum_j sc_j sumi_j is computed as 16 Dhi + Dlo, with sc_j = lo4 + 16 hi2
// (lo4 = sc & 15, hi2 = sc >> 4): the weights are unpacked to u8 q * lo4
// (<= 225) and q * hi2 (<= 45), each pass one u8 x s8 wgmma chain over the
// block's 256 K into an exact int32 accumulator (|Dlo| <= 7.3M, |16 Dhi| <=
// 23.4M). M = sum_j m_j b32_j by dp4a with b32 = lo + 64 hi (lo = b32 & 63,
// hi = b32 >> 6, both int8). No float op differs from gemv_ref.
//
// A block: WGS warpgroups, each 64 weight rows x 64 tokens (m64n64k32);
// rows 64 WGS y.., tokens 64 x... Grid (ceil(nb / 64), ceil(rows / (64 WGS)))
// x 128 WGS, GEMMH_SMEM(WGS) bytes of dynamic smem. Two stages: TMA brings
// block b + 1's raw weight rows (box 144 x 64 WGS) and activation values
// (two 128 x 64 boxes, 128-byte swizzle), the threads unpack them while
// block b's wgmmas run. mw: the weights (rb bytes x rows, stride rb); mx:
// the activations' Q8_K values (n bytes x nb, stride astride).
#define GH_SA(W) ((W) * 32768)
#define GH_SB 16384
#define GH_SR(W) ((W) * 64 * 144)
#define GH_SH(W) ((W) * 64 * 16)
#define GH_OFF_B(W) (2 * GH_SA(W))
#define GH_OFF_RAW(W) (GH_OFF_B(W) + 2 * GH_SB)
#define GH_OFF_XD(W) (GH_OFF_RAW(W) + 2 * GH_SR(W))
#define GH_OFF_XW(W) (GH_OFF_XD(W) + 2 * 256)
#define GH_OFF_H(W) (GH_OFF_XW(W) + 2 * 1024)
#define GH_OFF_BAR(W) (GH_OFF_H(W) + 2 * GH_SH(W))
#define GEMMH_SMEM(W) (GH_OFF_BAR(W) + 16 + 1024)

template <int WGS>
__device__ __forceinline__ void gemmh_q4k_t(const TMap* mw, const TMap* mx, const u8* act,
                                            u64 astride, int rows, int n, int nb,
                                            const float* bias, int mode, float* out,
                                            u64 ostride) {
    extern __shared__ u8 gh_dsm[];
    u32 dsz;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz < GEMMH_SMEM(WGS)) __trap(); // the host's size out of date
    u8* sm = (u8*)(((u64)gh_dsm + 1023) & ~(u64)1023);
    u64* full = (u64*)(sm + GH_OFF_BAR(WGS));
    const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127, w = t >> 5, l = t & 31;
    const int z0 = blockIdx.x * 64, r0 = blockIdx.y * 64 * WGS, nblk = n / 256;
    auto sA = [&](int s) { return sm + s * GH_SA(WGS) + wg * 32768; };
    auto sB = [&](int s) { return sm + GH_OFF_B(WGS) + s * GH_SB; };
    auto sRaw = [&](int s) { return sm + GH_OFF_RAW(WGS) + s * GH_SR(WGS); };
    auto sXd = [&](int s) { return (float*)(sm + GH_OFF_XD(WGS) + s * 256); };
    auto sXw = [&](int s) { return (uint4*)(sm + GH_OFF_XW(WGS) + s * 1024); };
    auto sH = [&](int s) { return (uint4*)(sm + GH_OFF_H(WGS) + s * GH_SH(WGS)) + wg * 64; };
    if (tid == 0) {
        mbar_init(&full[0], 1);
        mbar_init(&full[1], 1);
        mbar_init_fence();
    }
    __syncthreads();
    auto issue = [&](int b, int s) {
        mbar_expect(&full[s], 64 * WGS * 144 + GH_SB);
        tma_load_2d(sRaw(s), mw, &full[s], 144 * b, r0);
        tma_load_2d(sB(s), mx, &full[s], 256 * b, z0);
        tma_load_2d(sB(s) + 8192, mx, &full[s], 256 * b + 128, z0);
    };
    if (tid == 0) {
        issue(0, 0);
        if (nblk > 1) issue(1, 1);
    }
    // Block b into stage s: dk and b32's int8 pieces of the 64 tokens, the
    // weights unpacked (this warpgroup's rows) with d, dmin, m per row.
    auto prep = [&](int b, int s) {
        if (tid < 64) {
            int z = z0 + tid;
            float dk = 0.0f;
            uint4 xw = make_uint4(0, 0, 0, 0);
            if (z < nb) {
                Act a = act_view(act + (u64)z * astride, n);
                dk = a.dk[b];
                u32 lo[2] = {0, 0}, hi[2] = {0, 0};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    int v = a.b32[b * 8 + j];
                    lo[j >> 2] |= (u32)(v & 63) << (8 * (j & 3));
                    hi[j >> 2] |= (u32)((v >> 6) & 255) << (8 * (j & 3));
                }
                xw = make_uint4(lo[0], lo[1], hi[0], hi[1]);
            }
            sXd(s)[tid] = dk;
            sXw(s)[tid] = xw;
        }
        mbar_wait(&full[s], (b >> 1) & 1);
        const u8* raw = sRaw(s) + wg * 64 * 144;
        u8* ap = sA(s);
        uint4* hp = sH(s);
#pragma unroll
        for (int it = 0; it < 4; it++) {
            int idx = t + 128 * it, r = idx >> 3, c = idx & 7, p = c >> 1;
            const u8* blk = raw + r * 144;
            u32 sc0, m0, sc1, m1;
            scale_min_k4(2 * p, blk + 4, &sc0, &m0);
            scale_min_k4(2 * p + 1, blk + 4, &sc1, &m1);
            uint4 q = *(const uint4*)(blk + 16 + 16 * c);
            u32 qw[4] = {q.x, q.y, q.z, q.w}, o[4][4];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                u32 L = qw[e] & 0x0f0f0f0fu, H = (qw[e] >> 4) & 0x0f0f0f0fu;
                o[0][e] = L * (sc0 & 15);
                o[1][e] = L * (sc0 >> 4);
                o[2][e] = H * (sc1 & 15);
                o[3][e] = H * (sc1 >> 4);
            }
            int kl = 64 * p + 16 * (c & 1), kh = kl + 32;
            u32 ol = (kl >> 7) * 8192 + sw128(r, kl & 127);
            u32 oh = (kh >> 7) * 8192 + sw128(r, kh & 127);
            *(uint4*)(ap + ol) = make_uint4(o[0][0], o[0][1], o[0][2], o[0][3]);
            *(uint4*)(ap + 16384 + ol) = make_uint4(o[1][0], o[1][1], o[1][2], o[1][3]);
            *(uint4*)(ap + oh) = make_uint4(o[2][0], o[2][1], o[2][2], o[2][3]);
            *(uint4*)(ap + 16384 + oh) = make_uint4(o[3][0], o[3][1], o[3][2], o[3][3]);
            if ((c & 1) == 0) {
                u8* mb = (u8*)&hp[r] + 8;
                mb[2 * p] = (u8)m0;
                mb[2 * p + 1] = (u8)m1;
            }
            if (c == 0) {
                hp[r].x = __float_as_uint(h2f(rd_u16(blk)));
                hp[r].y = __float_as_uint(h2f(rd_u16(blk + 2)));
            }
        }
        fence_proxy_async(); // the unpacked weights, for the wgmmas (async proxy)
    };
    prep(0, 0);
    __syncthreads();
    float acc[32];
    int dlo[32], dhi[32], M[32];
#pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f, dlo[i] = 0, dhi[i] = 0;
    // The thread's outputs: e of 4 i + e at row 16 w + l / 4 + 8 (e >> 1),
    // token 8 i + 2 (l & 3) + (e & 1) (within the warpgroup's tile).
    const int rl = 16 * w + l / 4, cl = 2 * (l & 3);
    for (int b = 0; b < nblk; b++) {
        int s = b & 1;
        u32 a0 = smem_addr(sA(s)), b0 = smem_addr(sB(s));
        wg_pin_n64(dlo);
        wg_pin_n64(dhi);
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < 8; kk++)
            wg_mma_n64_u8(dlo, wg_desc(a0 + (kk >> 2) * 8192) + 2 * (kk & 3),
                          wg_desc(b0 + (kk >> 2) * 8192) + 2 * (kk & 3), kk);
#pragma unroll
        for (int kk = 0; kk < 8; kk++)
            wg_mma_n64_u8(dhi, wg_desc(a0 + 16384 + (kk >> 2) * 8192) + 2 * (kk & 3),
                          wg_desc(b0 + (kk >> 2) * 8192) + 2 * (kk & 3), kk);
        wg_commit();
        // M of block b while the wgmmas run.
        {
            const uint4* hp = sH(s);
            const uint4* xw = sXw(s);
            uint4 h0 = hp[rl], h1 = hp[rl + 8];
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    uint4 x = xw[8 * i + cl + (e & 1)];
                    uint4 h = (e >> 1) ? h1 : h0;
                    int ml = __dp4a((int)h.z, (int)x.x, __dp4a((int)h.w, (int)x.y, 0));
                    int mh = __dp4a((int)h.z, (int)x.z, __dp4a((int)h.w, (int)x.w, 0));
                    M[4 * i + e] = ml + 64 * mh;
                }
        }
        if (b + 1 < nblk) prep(b + 1, s ^ 1);
        wg_wait<0>();
        wg_pin_n64(dlo);
        wg_pin_n64(dhi);
        {
            const uint4* hp = sH(s);
            const float* xd = sXd(s);
            uint4 h0 = hp[rl], h1 = hp[rl + 8];
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    uint4 h = (e >> 1) ? h1 : h0;
                    int S = 16 * dhi[4 * i + e] + dlo[4 * i + e];
                    float v = __fsub_rn(__fmul_rn(__uint_as_float(h.x), i2f_rn(S)),
                                        __fmul_rn(__uint_as_float(h.y), gm_i2f(M[4 * i + e])));
                    acc[4 * i + e] = __fmaf_rn(xd[8 * i + cl + (e & 1)], v, acc[4 * i + e]);
                }
        }
        __syncthreads();
        if (tid == 0 && b + 2 < nblk) issue(b + 2, s);
    }
    // Every load the stores need first, then the stores (gemm2's epilogue).
    float pre[32];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int r = r0 + 64 * wg + rl + 8 * (e >> 1), z = z0 + 8 * i + cl + (e & 1);
            pre[4 * i + e] = 0.0f;
            if (mode != 0 && r < rows && z < nb)
                pre[4 * i + e] = gm_pre(bias, mode, out, (u64)z * ostride, r);
        }
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int r = r0 + 64 * wg + rl + 8 * (e >> 1), z = z0 + 8 * i + cl + (e & 1);
            if (r < rows && z < nb)
                gm_put(acc[4 * i + e], pre[4 * i + e], bias, mode, out, (u64)z * ostride, r);
        }
}

#define GEMMH(NAME, WGS)                                                                          \
    extern "C" __global__ void __launch_bounds__(128 * WGS, 1)                                    \
        NAME(const __grid_constant__ TMap mw, const __grid_constant__ TMap mx,                    \
             const u8* __restrict__ act, u64 astride, int rows, int n, int nb,                    \
             const float* __restrict__ bias, int mode, float* __restrict__ out, u64 ostride) {    \
        gemmh_q4k_t<WGS>(&mw, &mx, act, astride, rows, n, nb, bias, mode, out, ostride);          \
    }
GEMMH(gemmh_q4_k, 2)
GEMMH(gemmh1_q4_k, 1)

// ---------------------------------------------------------------------------
// gemmh2_q4_k: gemmh_q4_k's arithmetic (same S, M, float ops) with the
// weights unpacked straight into registers: RS wgmma (A from registers, B
// from smem), so no unpacked copy goes through smem. Two warpgroups, each
// 64 rows x 64 tokens, sharing a GH2_ST-stage TMA ring (raw Q4_K rows of
// both, the 64 tokens' values); each runs on its own (warpgroup barriers
// only), so one's epilogue overlaps the other's wgmmas. A stage is
// refilled by thread 0 once both warpgroups released it (mbarrier empty,
// count 2). Grid (ceil(nb / 64), ceil(rows / 128)) x 256, GEMMH2_SMEM bytes
// of dynamic smem; arguments as gemmh_q4_k's.
// gemmh2_q4_k_gu: the gate and up products of a layer in one launch, out =
// silu(gate) * up (gemm2s_q4_k_gu's ops on the same floats): rows 64 y.. of
// the gate matrix (mw) in warpgroup 0 and of the up matrix (mw2) in 1; grid
// (ceil(nb / 64), ceil(rows / 64)); bias and mode unused.
#define GH2_ST 4
#define GH2_OFF_RAW (GH2_ST * 16384)
#define GH2_OFF_X (GH2_OFF_RAW + GH2_ST * 128 * 144)
#define GH2_OFF_BAR (GH2_OFF_X + GH2_ST * 2 * 1280)
#define GEMMH2_SMEM (GH2_OFF_BAR + 2 * GH2_ST * 8 + 1024)

__device__ __forceinline__ void wg_bar(int id) {
    asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory");
}

__device__ __forceinline__ void pin4(u32* a) {
    asm volatile("" : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])::"memory");
}

template <bool GU>
__device__ __forceinline__ void gemmh2_t(const TMap* mw, const TMap* mw2, const TMap* mx,
                                         const u8* act, u64 astride, int rows, int n, int nb,
                                         const float* bias, int mode, float* out, u64 ostride) {
    extern __shared__ u8 gh2_dsm[];
    u32 dsz;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
    if (dsz < GEMMH2_SMEM) __trap(); // the host's size out of date
    u8* sm = (u8*)(((u64)gh2_dsm + 1023) & ~(u64)1023);
    u64* full = (u64*)(sm + GH2_OFF_BAR);
    u64* empty = full + GH2_ST;
    const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127, w = t >> 5, l = t & 31;
    const int g = l >> 2, q = l & 3;
    const int z0 = blockIdx.x * 64, r0 = blockIdx.y * (GU ? 64 : 128), nblk = n / 256;
    if (tid == 0) {
        for (int s = 0; s < GH2_ST; s++) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], 2);
        }
        mbar_init_fence();
    }
    __syncthreads();
    auto issue = [&](int b, int s) {
        mbar_expect(&full[s], 128 * 144 + 16384);
        if (GU) {
            tma_load_2d(sm + GH2_OFF_RAW + s * 128 * 144, mw, &full[s], 144 * b, r0);
            tma_load_2d(sm + GH2_OFF_RAW + s * 128 * 144 + 64 * 144, mw2, &full[s], 144 * b, r0);
        } else {
            tma_load_2d(sm + GH2_OFF_RAW + s * 128 * 144, mw, &full[s], 144 * b, r0);
        }
        tma_load_2d(sm + s * 16384, mx, &full[s], 256 * b, z0);
        tma_load_2d(sm + s * 16384 + 8192, mx, &full[s], 256 * b + 128, z0);
    };
    if (tid == 0)
        for (int b = 0; b < GH2_ST && b < nblk; b++) issue(b, b);
    float acc[32];
    int dlo[32], dhi[32], M[32];
#pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f, dlo[i] = 0, dhi[i] = 0;
    // The thread's rows (within the warpgroup): 16 w + g and + 8; its
    // outputs as gemmh_q4_k's.
    const int rl = 16 * w + g, cl = 2 * q;
    for (int b = 0; b < nblk; b++) {
        const int s = b % GH2_ST;
        if (tid == 0 && b >= 1 && b - 1 + GH2_ST < nblk) {
            int sp = (b - 1) % GH2_ST;
            mbar_wait(&empty[sp], ((b - 1) / GH2_ST) & 1);
            issue(b - 1 + GH2_ST, sp);
        }
        float* xd = (float*)(sm + GH2_OFF_X + (s * 2 + wg) * 1280);
        uint4* xw = (uint4*)(xd + 64);
        if (t < 64) {
            int z = z0 + t;
            float dk = 0.0f;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (z < nb) {
                Act a = act_view(act + (u64)z * astride, n);
                dk = a.dk[b];
                u32 lo[2] = {0, 0}, hi[2] = {0, 0};
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    int x = a.b32[b * 8 + j];
                    lo[j >> 2] |= (u32)(x & 63) << (8 * (j & 3));
                    hi[j >> 2] |= (u32)((x >> 6) & 255) << (8 * (j & 3));
                }
                v = make_uint4(lo[0], lo[1], hi[0], hi[1]);
            }
            xd[t] = dk;
            xw[t] = v;
        }
        mbar_wait(&full[s], (b / GH2_ST) & 1);
        // Unpack: this thread's A fragments of the 8 k32 steps (sub-blocks
        // j), both passes; the rows' d, dmin and mins.
        u32 alo[8][4], ahi[8][4];
        float dr[2], dmr[2];
        u32 mw[2][2];
        const u8* raw = sm + GH2_OFF_RAW + s * 128 * 144 + (64 * wg + rl) * 144;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const u8* blk = raw + 8 * h * 144;
            uint4 hd = *(const uint4*)blk;
            dr[h] = h2f(hd.x & 0xffff);
            dmr[h] = h2f(hd.x >> 16);
            u32 sc[2] = {hd.y & 0x3f3f3f3fu, (hd.w & 0x0f0f0f0fu) | ((hd.y >> 2) & 0x30303030u)};
            mw[h][0] = hd.z & 0x3f3f3f3fu;
            mw[h][1] = ((hd.w >> 4) & 0x0f0f0f0fu) | ((hd.z >> 2) & 0x30303030u);
#pragma unroll
            for (int p = 0; p < 4; p++) {
                u32 w0 = *(const u32*)(blk + 16 + 32 * p + 4 * q);
                u32 w1 = *(const u32*)(blk + 16 + 32 * p + 16 + 4 * q);
#pragma unroll
                for (int o = 0; o < 2; o++) { // j = 2p + o: low (o 0) or high nibbles
                    int j = 2 * p + o;
                    u32 scj = (sc[j >> 2] >> (8 * (j & 3))) & 255;
                    u32 v0 = (o ? w0 >> 4 : w0) & 0x0f0f0f0fu;
                    u32 v1 = (o ? w1 >> 4 : w1) & 0x0f0f0f0fu;
                    alo[j][h] = v0 * (scj & 15);
                    alo[j][2 + h] = v1 * (scj & 15);
                    ahi[j][h] = v0 * (scj >> 4);
                    ahi[j][2 + h] = v1 * (scj >> 4);
                }
            }
        }
        wg_bar(1 + wg); // this warpgroup's xd / xw written
        u32 b0 = smem_addr(sm + s * 16384);
        wg_pin_n64(dlo);
        wg_pin_n64(dhi);
        wg_fence();
#pragma unroll
        for (int j = 0; j < 8; j++)
            wg_mma_rs_n64_u8(dlo, alo[j], wg_desc(b0 + (j >> 2) * 8192) + 2 * (j & 3), j);
#pragma unroll
        for (int j = 0; j < 8; j++)
            wg_mma_rs_n64_u8(dhi, ahi[j], wg_desc(b0 + (j >> 2) * 8192) + 2 * (j & 3), j);
        wg_commit();
        // M of block b while the wgmmas run.
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                uint4 x = xw[8 * i + cl + (e & 1)];
                const u32* m = mw[e >> 1];
                int ml = __dp4a((int)m[0], (int)x.x, __dp4a((int)m[1], (int)x.y, 0));
                int mh = __dp4a((int)m[0], (int)x.z, __dp4a((int)m[1], (int)x.w, 0));
                M[4 * i + e] = ml + 64 * mh;
            }
        wg_wait<0>();
        wg_pin_n64(dlo);
        wg_pin_n64(dhi);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            pin4(alo[j]);
            pin4(ahi[j]);
        }
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int h = e >> 1;
                int S = 16 * dhi[4 * i + e] + dlo[4 * i + e];
                float v = __fsub_rn(__fmul_rn(dr[h], i2f_rn(S)),
                                    __fmul_rn(dmr[h], gm_i2f(M[4 * i + e])));
                acc[4 * i + e] = __fmaf_rn(xd[8 * i + cl + (e & 1)], v, acc[4 * i + e]);
            }
        wg_bar(1 + wg); // this warpgroup is done with stage s
        if (t == 0) mbar_arrive(&empty[s]);
    }
    if constexpr (GU) {
        // The gate sums (warpgroup 0) to the up threads holding the same
        // row and token (warpgroup 1) through the stage buffers (free: every
        // block issued has been consumed); out = silu(gate) * up.
        float* xch = (float*)sm;
        __syncthreads();
        if (wg == 0) {
#pragma unroll
            for (int i = 0; i < 32; i++) xch[i * 128 + t] = acc[i];
        }
        __syncthreads();
        if (wg == 1) {
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int r = r0 + rl + 8 * (e >> 1), z = z0 + 8 * i + cl + (e & 1);
                    float gv = xch[(4 * i + e) * 128 + t];
                    if (r < rows && z < nb)
                        out[(u64)z * ostride + r] =
                            __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), acc[4 * i + e]);
                }
        }
    } else {
        float pre[32];
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = r0 + 64 * wg + rl + 8 * (e >> 1), z = z0 + 8 * i + cl + (e & 1);
                pre[4 * i + e] = 0.0f;
                if (mode != 0 && r < rows && z < nb)
                    pre[4 * i + e] = gm_pre(bias, mode, out, (u64)z * ostride, r);
            }
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = r0 + 64 * wg + rl + 8 * (e >> 1), z = z0 + 8 * i + cl + (e & 1);
                if (r < rows && z < nb)
                    gm_put(acc[4 * i + e], pre[4 * i + e], bias, mode, out, (u64)z * ostride, r);
            }
    }
}

extern "C" __global__ void __launch_bounds__(256, 1)
gemmh2_q4_k(const __grid_constant__ TMap mw, const __grid_constant__ TMap mx,
            const u8* __restrict__ act, u64 astride, int rows, int n, int nb,
            const float* __restrict__ bias, int mode, float* __restrict__ out, u64 ostride) {
    gemmh2_t<false>(&mw, &mw, &mx, act, astride, rows, n, nb, bias, mode, out, ostride);
}

extern "C" __global__ void __launch_bounds__(256, 1)
gemmh2_q4_k_gu(const __grid_constant__ TMap mw, const __grid_constant__ TMap mw2,
               const __grid_constant__ TMap mx, const u8* __restrict__ act, u64 astride, int rows,
               int n, int nb, float* __restrict__ out, u64 ostride) {
    gemmh2_t<true>(&mw, &mw2, &mx, act, astride, rows, n, nb, nullptr, 0, out, ostride);
}

#endif // __CUDA_ARCH_FEAT_SM90_ALL
