//! The `axon` command line. All of it lives in the library (`axon::cli`),
//! next to the engine it drives; this is only the entry point.

fn main() {
    // A release build carries its GPU kernels precompiled (`axon kernels
    // build`, docs/models.md "NVIDIA GPUs"): built with `--cfg
    // axon_kernel_pack` and AXON_BUILD_KERNEL_PACK naming the pack file.
    #[cfg(axon_kernel_pack)]
    axon::cli::set_kernel_pack(include_bytes!(env!("AXON_BUILD_KERNEL_PACK")));
    axon::cli::main()
}
