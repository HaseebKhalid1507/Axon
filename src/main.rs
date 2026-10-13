mod gguf;
mod model;
mod quant;
#[cfg(test)]
mod testutil;
mod tokenizer;
mod viz;

use std::env;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("usage: axon <model.gguf>        # interactive TUI");
        eprintln!("       axon info <model.gguf>    # print summary and exit");
        eprintln!("       axon gen <model.gguf> \"<prompt>\" [-n 32]     # greedy generation");
        eprintln!("       axon logits <model.gguf> \"<prompt>\" [-k 10]  # top-k logits at last position");
        std::process::exit(1);
    }

    if args[1] == "gen" || args[1] == "logits" {
        return run_lm(&args);
    }

    let (cmd, path) = if args.len() >= 3 && args[1] == "info" {
        ("info", &args[2])
    } else {
        ("viz", &args[1])
    };

    let model = gguf::load(path)?;

    match cmd {
        "info" => {
            let a = &model.arch;
            println!("axon v{}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("{} | {}L / {}d / {}h (kv {}) | ctx {}",
                a, model.n_layers(), model.embed_dim(), model.n_heads(), model.n_kv_heads(), model.ctx_len());
            println!("{} tensors | {:.2}B params | {}",
                model.tensors.len(), model.total_params() as f64 / 1e9, human_bytes(model.total_bytes()));
        }
        _ => {
            viz::run(&model)?;
        }
    }

    Ok(())
}

/// `gen` and `logits` subcommands: args = [axon, cmd, model, prompt, flags...]
fn run_lm(args: &[String]) -> std::io::Result<()> {
    use std::io::Write;
    use std::time::Instant;

    let cmd = args[1].as_str();
    if args.len() < 4 {
        eprintln!("usage: axon {cmd} <model.gguf> \"<prompt>\" [{}]", if cmd == "gen" { "-n 32" } else { "-k 10" });
        std::process::exit(1);
    }
    let (path, prompt) = (&args[2], &args[3]);
    let mut n_gen = 32usize;
    let mut top_k = 10usize;
    let mut i = 4;
    while i < args.len() {
        let val = args.get(i + 1).and_then(|s| s.parse::<usize>().ok());
        match (args[i].as_str(), val) {
            ("-n", Some(v)) => n_gen = v,
            ("-k", Some(v)) => top_k = v,
            (f, _) => { eprintln!("bad flag or value: {f}"); std::process::exit(1); }
        }
        i += 2;
    }

    let model = gguf::load(path)?;
    let tok = tokenizer::Tokenizer::from_gguf(&model);
    let net = model::Qwen2::new(&model);

    let ids = tok.encode(prompt);
    if ids.is_empty() {
        eprintln!("prompt encodes to zero tokens");
        std::process::exit(1);
    }
    let extra = if cmd == "gen" { n_gen } else { 0 };
    let mut cache = net.new_cache(ids.len() + extra + 1);

    // Feed the prompt one token at a time; keep the logits of the last position.
    let t0 = Instant::now();
    let mut logits = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        logits = net.forward(id, pos, &mut cache);
    }
    let prompt_secs = t0.elapsed().as_secs_f64();

    if cmd == "logits" {
        println!("prompt ids ({}): {:?}", ids.len(), ids);
        // log-softmax normaliser, so we can print logprobs comparable to Ollama's
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = mx + logits.iter().map(|l| (l - mx).exp()).sum::<f32>().ln();
        println!("top-{top_k} of {} at last position (id, logit, logprob, text):", net.vocab);
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        for &id in order.iter().take(top_k) {
            println!("{:>7}  {:>10.4}  {:>8.3}  {:?}", id, logits[id], logits[id] - lse, tok.decode(&[id as u32]));
        }
        return Ok(());
    }

    // Greedy decoding: always pick the highest-scoring token.
    let argmax = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
    let mut out = std::io::stdout();
    print!("{prompt}");
    out.flush()?;
    let mut gen_ids: Vec<u32> = Vec::new();
    let mut printed = 0usize; // bytes of decoded text already printed
    let mut decode_steps = 0usize;
    let t1 = Instant::now();
    for step in 0..n_gen {
        let next = argmax(&logits[..]);
        if next == tok.eos() { break; }
        gen_ids.push(next);
        // Decode the whole tail so multi-byte characters split across tokens come out whole.
        let text = tok.decode(&gen_ids);
        if !text.ends_with('\u{FFFD}') && text.len() > printed && text.is_char_boundary(printed) {
            print!("{}", &text[printed..]);
            out.flush()?;
            printed = text.len();
        }
        if step + 1 < n_gen {
            logits = net.forward(next, ids.len() + step, &mut cache);
            decode_steps += 1;
        }
    }
    let decode_secs = t1.elapsed().as_secs_f64();
    let text = tok.decode(&gen_ids);
    if text.len() > printed && text.is_char_boundary(printed) { print!("{}", &text[printed..]); }
    println!();
    eprintln!("prompt eval: {} tok in {:.2}s ({:.2} tok/s)",
        ids.len(), prompt_secs, ids.len() as f64 / prompt_secs);
    eprintln!("decode:      {} tok, {} forwards in {:.2}s ({:.2} tok/s)",
        gen_ids.len(), decode_steps, decode_secs,
        if decode_secs > 0.0 { decode_steps as f64 / decode_secs } else { 0.0 });
    Ok(())
}

fn human_bytes(b: u64) -> String {
    if b >= 1 << 30 { format!("{:.2} GB", b as f64 / (1u64 << 30) as f64) }
    else if b >= 1 << 20 { format!("{:.1} MB", b as f64 / (1u64 << 20) as f64) }
    else { format!("{b} B") }
}
