//! Axon as a library: load a GGUF, encode a prompt, prefill it, and decode
//! N tokens greedily, streaming them to stdout.
//!
//!     cargo run --release --example embed -- model.gguf "The capital of France is" 32
//!
//! The device is $AXON_DEVICE (cpu, the default, or cuda), as for `axon`.
//! Greedy decoding here is `axon gen`'s: the same tokens, bit for bit.

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
    let device = Device::from_env()?;

    let model = axon::load(path, device)?;
    let tok = Tokenizer::from_gguf(&model);
    let ids = tok.encode_prompt(prompt);
    if ids.is_empty() {
        return Err("the prompt encodes to zero tokens".into());
    }
    // The cache holds the prompt, the generated tokens and one more.
    let ctx = ids.len() + n + 1;
    let net = Backend::new(&model, device, ctx, None, KvType::F16, false)?;
    let mut cache = net.new_cache(ctx);
    let logits = net.prefill_logits(&ids, 0, &mut cache);

    let mut out = std::io::stdout();
    print!("{prompt}");
    // A character can be split across tokens: the stream holds it back until whole.
    let mut text = Utf8Stream::default();
    let opts = GenOpts {
        n_gen: n,
        stop: &tok.eog(),
        max_draft: 0,
        sample: None,
    };
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
