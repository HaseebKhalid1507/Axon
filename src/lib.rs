//! Axon: CPU and CUDA inference for GGUF language models (the README lists
//! the supported families), bit-exact across decode, batched prefill and
//! speculative verify.
//!
//! The `axon` binary is a thin wrapper over this crate. The public API is
//! deliberately small; everything else stays private to the crate:
//!
//! - [`load`]: map (or copy) a GGUF and check that Axon can run it;
//! - [`Backend::new`]: place it on the CPU or a CUDA GPU, as an [`Lm`];
//! - [`Tokenizer`]: the model's own tokenizer, from the same file;
//! - [`Lm`]: caches, prefill and decode steps;
//! - [`generate`]: the greedy decode loop behind `axon gen` (sampling stays
//!   the CLI's in 0.1), with optional lossless drafting,
//!   streaming each token to a callback;
//! - [`serve`]: the OpenAI-compatible HTTP server behind `axon serve`.
//!
//! ```no_run
//! use axon::{Backend, Device, GenOpts, KvType, Lm, Tokenizer};
//!
//! let model = axon::load("models/qwen2.5-0.5b-instruct-q4_k_m.gguf", Device::Cpu)?;
//! let tok = Tokenizer::from_gguf(&model);
//! let ids = tok.encode_prompt("The capital of France is");
//! let ctx = ids.len() + 16 + 1;
//! let net = Backend::new(&model, Device::Cpu, ctx, None, KvType::F16, false)?;
//! let mut cache = net.new_cache(ctx);
//! let logits = net.prefill_logits(&ids, 0, &mut cache);
//! let stop = tok.eog();
//! let opts = GenOpts::greedy(16, &stop);
//! let st = axon::generate(&net, &mut cache, &ids, logits, opts, &mut |_, _| Vec::new(), &mut |_| Ok(()))
//!     .map_err(|e| e.to_string())?;
//! println!("{}", tok.decode(&st.ids));
//! # Ok::<(), String>(())
//! ```
//!
//! The choices the API takes as arguments (device, KV type, GPU layers,
//! context) are never read from the environment: the CLI reads
//! `AXON_DEVICE`, `AXON_KV` and `AXON_GPU_LAYERS` and passes them, and an
//! embedder passes its own (`Device::from_env` and `KvType::parse` parse
//! them as the CLI does; examples/embed.rs). Engine tuning variables are
//! still read inside the library: some once, at first use (`AXON_SIMD`,
//! `AXON_THREADS`, the decode config), others at every `load` /
//! `Backend::new` (`AXON_MMAP`, `AXON_REPACK`, `AXON_Q6K8`, `AXON_GATE_UP`,
//! `AXON_CUDA_GRAPHS`, `AXON_PP_CUTS`), so changing them between loads
//! changes behaviour. `AXON_GPUS` picks the GPU(s): the only way in 0.1
//! (read at each `Backend::new`). A value the engine can't use for
//! `AXON_SIMD`, `AXON_DECODE_ROWS` or `AXON_PREFETCH` is an error from
//! `load` and `Backend::new`. README; #118.
//!
//! [`Backend`] and [`BackendCache`] are opaque, [`Lm`] is sealed, and
//! [`Device`], [`KvType`] and [`GenStats`] are non-exhaustive: a new
//! backend, device, KV type or statistic is not a breaking change.

#[macro_use]
mod mv;
mod chat;
#[doc(hidden)]
pub mod cli;
mod cuda;
mod draft;
mod generate;
mod gguf;
mod hybrid;
mod identity;
mod kv;
mod kvsave;
mod kvstore;
mod lineedit;
mod lm;
mod model;
mod moe;
mod pool;
mod quant;
mod rope;
mod safetensors;
mod sample;
mod serve;
mod spec;
#[cfg(test)]
mod testutil;
mod tokenizer;
mod vmath;

pub use generate::{GenOpts, GenStats, generate};
pub use gguf::Model;
pub use kv::KvType;
pub use lm::{Backend, BackendCache, Device, Lm};
pub use tokenizer::{Tokenizer, Utf8Stream};

// Crate-root paths the modules use (`crate::PREFILL_BATCH`, `crate::fail`,
// ...), from before the library split, when they lived in main.rs.
pub(crate) use cli::{
    DEFAULT_CHAT_CTX, backend_and_drafter, check_stream_decode, copy_for, device_env, draft_env,
    draft_env_set, fail, kv_env, kv_flag, open_model_with, placement, round_draft,
};
pub(crate) use generate::{PREFILL_BATCH, argmax};

/// Load the GGUF (or safetensors directory) at `path` to run on `device`:
/// mapped for the GPU; for the CPU, copied into huge pages when the model
/// reads its weights in place (AXON_MMAP overrides). An error names the
/// file and says why it can't be read or run, or names an engine variable
/// with a value Axon can't use (AXON_SIMD, AXON_DECODE_ROWS,
/// AXON_PREFETCH: checked before anything is read).
pub fn load(path: &str, device: Device) -> Result<Model, String> {
    quant::env_check()?;
    let mut unsupported = None;
    let m = gguf::load_to_run(path, (device == Device::Cuda).then_some(false), |m| {
        if let Err(e) = model::check_supported(m) {
            unsupported = Some(e);
            return gguf::CopyWish::MapOnly;
        }
        match model::is_moe(m) {
            true => gguf::CopyWish::MapOnly,
            false => gguf::CopyWish::Want(model::reads_matrices_in_place(m)),
        }
    })
    .map_err(|e| format!("{path}: {e}"))?;
    match unsupported {
        Some(e) => Err(format!("{path}: {e}")),
        None => Ok(m),
    }
}

/// Run `axon serve` in this process, until it's stopped: `args` is the
/// command line, `["axon", "serve", "model.gguf", flags...]` (README,
/// `axon serve --help`). Blocks. It behaves as the CLI does, process exits
/// included: a usage error (flags, or a bad `AXON_DEVICE` / `AXON_KV` /
/// `AXON_DRAFT`) prints a message and exits with code 2, and an engine
/// failure (a panic in the engine thread) exits with code 70. Not yet
/// safe to call from a process that must outlive the server.
pub fn serve(args: &[String]) -> std::io::Result<()> {
    serve::run(args)
}

#[cfg(test)]
mod tests {
    use super::{Backend, Device, KvType};

    /// The CPU caches' f16/f32 choice is process-wide: once it's f16 (every
    /// test makes f16 caches), asking for f32 is an error, not a silent f16.
    #[test]
    fn kv_f32_after_f16_is_an_error() {
        crate::kv::set_f32(false);
        assert!(!crate::kv::use_f32());
        let m = crate::gguf::build_for_test(&[], &[]);
        let e = Backend::new(&m, Device::Cpu, 64, None, KvType::F32, false)
            .err()
            .expect("f32 after f16");
        assert!(e.contains("f32 KV cache"), "{e}");
        let e = Backend::new(&m, Device::Cpu, 64, None, KvType::Q8, false)
            .err()
            .expect("q8 on the CPU");
        assert!(e.contains("cuda-only"), "{e}");
    }
}
