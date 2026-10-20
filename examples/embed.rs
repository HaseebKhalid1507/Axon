//! Axon as a library: load a GGUF, encode a prompt, prefill it, and decode
//! N tokens greedily, streaming them to stdout.
//!
//!     cargo run --release --example embed -- model.gguf "The capital of France is" 32
//!
//! The library takes every choice as an argument and reads none of these
//! from the environment; this example reads them as `axon gen` does, so
//! that under the same environment it prints exactly what `axon gen` prints:
//! AXON_DEVICE (cpu, the default, or cuda), AXON_KV (f16, the default, f32
//! or q8) and, on cuda, AXON_GPU_LAYERS (0 = the CPU). Not mirrored: the
//! CLI's flags, drafting (AXON_DRAFT; lossless, so the tokens are the same)
//! and `gen`'s cap at the model's trained context. Engine tuning variables
//! (AXON_THREADS, AXON_GPUS, AXON_MMAP, ...) are read inside the library,
//! for both. scripts/embed_check.sh checks the two agree.

use axon::{Backend, Device, GenOpts, KvType, Lm, Tokenizer, Utf8Stream};
use std::io::Write;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let [_, path, prompt, rest @ ..] = &args[..] else {
        return Err("usage: embed <model.gguf> <prompt> [n_tokens]".into());
    };
    let n: usize = match rest.first() {
        Some(v) => v
            .parse()
            .map_err(|_| format!("n_tokens: '{v}' isn't a number"))?,
        None => 32,
    };
    let mut device = Device::from_env()?;
    let kv = KvType::parse(&std::env::var("AXON_KV").unwrap_or_default())
        .map_err(|e| format!("AXON_KV: {e}"))?;
    let mut gpu_layers = match std::env::var("AXON_GPU_LAYERS") {
        Ok(v) if device == Device::Cuda => Some(
            v.parse::<usize>()
                .map_err(|_| format!("AXON_GPU_LAYERS needs a whole number, got '{v}'"))?,
        ),
        _ => None,
    };
    if gpu_layers == Some(0) {
        (device, gpu_layers) = (Device::Cpu, None);
    }

    let model = axon::load(path, device)?;
    let tok = Tokenizer::from_gguf(&model);
    let ids = tok.encode_prompt(prompt);
    if ids.is_empty() {
        return Err("the prompt encodes to zero tokens".into());
    }
    // The cache holds the prompt, the generated tokens and one more.
    let ctx = ids.len() + n + 1;
    let net = Backend::new(&model, device, ctx, gpu_layers, kv, false)?;
    let mut cache = net.new_cache(ctx);
    let logits = net.prefill_logits(&ids, 0, &mut cache);

    let mut out = std::io::stdout();
    print!("{prompt}");
    // A character can be split across tokens: the stream holds it back until whole.
    let mut text = Utf8Stream::default();
    let stop = tok.eog();
    let opts = GenOpts::greedy(n, &stop);
    let st = axon::generate(
        &net,
        &mut cache,
        &ids,
        logits,
        opts,
        &mut |_, _| Vec::new(),
        &mut |id| {
            print!("{}", text.push(&tok.decode_bytes(&[id])));
            out.flush()
        },
    )
    .map_err(|e| e.to_string())?;
    println!("{}", text.finish());
    eprintln!(
        "embed: {} prompt tokens, {} generated: {:?}",
        ids.len(),
        st.ids.len(),
        st.ids
    );
    Ok(())
}
