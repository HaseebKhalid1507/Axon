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
//! - [`generate`]: the greedy (or sampled) decode loop behind `axon gen`,
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
//! let opts = GenOpts { n_gen: 16, stop: &tok.eog(), max_draft: 0, sample: None };
//! let st = axon::generate(&net, &mut cache, &ids, logits, opts, &mut |_, _| Vec::new(), &mut |_| Ok(()))
//!     .map_err(|e| e.to_string())?;
//! println!("{}", tok.decode(&st.ids));
//! # Ok::<(), String>(())
//! ```
//!
//! Environment variables (`AXON_*`) tune the engine as they tune the CLI
//! (README); the defaults are the CLI's.

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
/// file and says why it can't be read or run.
pub fn load(path: &str, device: Device) -> Result<Model, String> {
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
/// `axon serve --help`). Blocks. As on the command line, a usage error
/// prints a message and exits the process.
pub fn serve(args: &[String]) -> std::io::Result<()> {
    serve::run(args)
}
