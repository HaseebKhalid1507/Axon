mod gguf;
mod model;
mod pool;
mod quant;
#[cfg(test)]
mod testutil;
mod tokenizer;
mod viz;

use std::env;

/// Prompt tokens per `feed_batch` call.
const PREFILL_BATCH: usize = 64;

const USAGE: &str = "\
usage:
  axon <model.gguf>                         browse the model in a terminal UI
  axon info <model.gguf>                    print a summary and exit
  axon gen <model.gguf> \"<prompt>\" [-n N]   greedy generation
  axon logits <model.gguf> \"<prompt>\" [-k K]
                                            top-K next-token logits after the prompt
  axon -h | --help                          this help
  axon -V | --version                       print the version

flags:
  -n N   gen: tokens to generate (default 32)
  -k K   logits: how many logits to print (default 10)

environment:
  AXON_THREADS=N             worker threads (default: cores minus a quarter)
  AXON_SIMD=scalar|avx2|vnni force a kernel set (default: the best the CPU has)
  AXON_POOL_ADAPT=0          keep every worker in every job (no adaptive drop-out)
  AXON_POOL_DEBUG=1          gen: print thread-pool stats at the end
  AXON_MMAP=0|1              0: copy the model into huge-page memory, 1: map the file
                             (default: copy, unless the file is >60% of free memory)

The model must be a qwen2-architecture GGUF (e.g. Qwen2.5-0.5B-Instruct Q4_K_M).";

/// Usage error: one line, a hint, exit code 2.
fn fail(msg: &str) -> ! {
    eprintln!("axon: {msg}");
    eprintln!("try 'axon --help'");
    std::process::exit(2);
}

/// Load a GGUF, exiting with code 2 and a one-line message if it can't be read.
fn open_model(path: &str, lm: bool) -> gguf::Model {
    let m = gguf::load(path).unwrap_or_else(|e| match e.kind() {
        std::io::ErrorKind::NotFound => fail(&format!("cannot open '{path}': no such file")),
        _ => fail(&format!("cannot load '{path}': {e}")),
    });
    if lm && let Err(e) = model::check_supported(&m) {
        fail(&format!("{path}: {e}"));
    }
    m
}

fn main() {
    if let Err(e) = run() {
        eprintln!("axon: {e}");
        std::process::exit(1);
    }
}

fn run() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(String::as_str) {
        None => {
            eprintln!("axon {}\n\n{USAGE}", env!("CARGO_PKG_VERSION"));
            std::process::exit(2);
        }
        Some("-h" | "--help" | "help") => {
            println!(
                "axon {}: CPU inference for Qwen2 GGUF models\n\n{USAGE}",
                env!("CARGO_PKG_VERSION")
            );
            return Ok(());
        }
        Some("-V" | "--version") => {
            println!("axon {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        _ => {}
    }

    if args[1] == "gen" || args[1] == "logits" {
        return run_lm(&args);
    }

    let (cmd, path) = match args[1].as_str() {
        "info" if args.len() == 3 => ("info", &args[2]),
        "info" => fail("usage: axon info <model.gguf>"),
        f if f.starts_with('-') => fail(&format!("unknown flag '{f}'")),
        // A lone argument is a model to browse, unless it can't be one.
        p if args.len() == 2 && (p.ends_with(".gguf") || std::path::Path::new(p).exists()) => {
            ("viz", &args[1])
        }
        c => fail(&format!(
            "unknown command '{c}' (commands: gen, logits, info)"
        )),
    };

    let model = open_model(path, false);

    match cmd {
        "info" => {
            let a = &model.arch;
            println!("axon v{}", env!("CARGO_PKG_VERSION"));
            println!();
            println!(
                "{} | {}L / {}d / {}h (kv {}) | ctx {}",
                a,
                model.n_layers(),
                model.embed_dim(),
                model.n_heads(),
                model.n_kv_heads(),
                model.ctx_len()
            );
            println!(
                "{} tensors | {:.2}B params | {}",
                model.tensors.len(),
                model.total_params() as f64 / 1e9,
                human_bytes(model.total_bytes())
            );
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
        fail(&format!(
            "usage: axon {cmd} <model.gguf> \"<prompt>\" [{}]",
            if cmd == "gen" { "-n 32" } else { "-k 10" }
        ));
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
            (f @ ("-n" | "-k"), None) => fail(&format!(
                "{f} needs a whole number, got {}",
                args.get(i + 1)
                    .map_or("nothing".into(), |v| format!("'{v}'"))
            )),
            (f, _) => fail(&format!("unknown argument '{f}' for {cmd}")),
        }
        i += 2;
    }

    let model = open_model(path, true);
    let tok = tokenizer::Tokenizer::from_gguf(&model);
    let net = model::Qwen2::new(&model);

    let ids = tok.encode(prompt);
    if ids.is_empty() {
        eprintln!("prompt encodes to zero tokens");
        std::process::exit(1);
    }
    let extra = if cmd == "gen" { n_gen } else { 0 };
    let mut cache = net.new_cache(ids.len() + extra + 1);

    // Feed the prompt in batches of PREFILL_BATCH tokens, so each weight row
    // is read from RAM once per batch instead of once per token. Only the last
    // position's logits are used, so the others skip the output head.
    let t0 = Instant::now();
    let (&last, rest) = ids.split_last().expect("checked non-empty above");
    for (i, batch) in rest.chunks(PREFILL_BATCH).enumerate() {
        net.feed_batch(batch, i * PREFILL_BATCH, &mut cache);
    }
    let mut logits = net.forward(last, rest.len(), &mut cache);
    let prompt_secs = t0.elapsed().as_secs_f64();

    if cmd == "logits" {
        println!("prompt ids ({}): {:?}", ids.len(), ids);
        // log-softmax normaliser, so we can print logprobs comparable to Ollama's
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = mx + logits.iter().map(|l| (l - mx).exp()).sum::<f32>().ln();
        println!(
            "top-{top_k} of {} at last position (id, logit, logprob, text):",
            net.vocab
        );
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        for &id in order.iter().take(top_k) {
            println!(
                "{:>7}  {:>10.4}  {:>8.3}  {:?}",
                id,
                logits[id],
                logits[id] - lse,
                tok.decode(&[id as u32])
            );
        }
        return Ok(());
    }

    // Greedy decoding: always pick the highest-scoring token.
    let mut out = std::io::stdout();
    print!("{prompt}");
    out.flush()?;
    let mut gen_ids: Vec<u32> = Vec::new();
    // Characters can be split across tokens; the stream holds them back until whole.
    let mut text = tokenizer::Utf8Stream::default();
    let mut decode_steps = 0usize;
    let t1 = Instant::now();
    for step in 0..n_gen {
        let next = argmax(&logits[..]);
        if next == tok.eos() {
            break;
        }
        gen_ids.push(next);
        let done = text.push(&tok.decode_bytes(&[next]));
        if !done.is_empty() {
            print!("{done}");
            out.flush()?;
        }
        if step + 1 < n_gen {
            logits = net.forward(next, ids.len() + step, &mut cache);
            decode_steps += 1;
        }
    }
    let decode_secs = t1.elapsed().as_secs_f64();
    print!("{}", text.finish());
    println!();
    eprintln!(
        "prompt eval: {} tok in {:.2}s ({:.2} tok/s)",
        ids.len(),
        prompt_secs,
        ids.len() as f64 / prompt_secs
    );
    eprintln!(
        "decode:      {} tok, {} forwards in {:.2}s ({:.2} tok/s)",
        gen_ids.len(),
        decode_steps,
        decode_secs,
        if decode_secs > 0.0 {
            decode_steps as f64 / decode_secs
        } else {
            0.0
        }
    );
    if env::var_os("AXON_POOL_DEBUG").is_some() {
        let pool = pool::Pool::global();
        eprintln!(
            "pool:        {} threads, {} active now, {} at least; {}",
            pool.threads(),
            pool.active(),
            pool.min_active(),
            pool.stats()
        );
    }
    Ok(())
}

/// Index of the largest logit, the last one on ties (as `Iterator::max_by`
/// returns). Two passes that each vectorize: a lane-wise max, then a scan
/// back from the end, 8 at a time, for the last logit equal to it. A vocab is
/// ~152k logits, every token. Unlike a `total_cmp` max, NaN logits are never
/// picked (unless all are NaN) and -0.0 ties with +0.0.
fn argmax(l: &[f32]) -> u32 {
    assert!(!l.is_empty(), "argmax of no logits");
    let (chunks, tail) = l.as_chunks::<8>();
    let mut lanes = [f32::NEG_INFINITY; 8];
    for ch in chunks {
        for i in 0..8 {
            // Not f32::max: its NaN rule costs extra instructions per lane on
            // x86. A plain compare is one maxps, and skips NaN all the same.
            lanes[i] = if ch[i] > lanes[i] { ch[i] } else { lanes[i] };
        }
    }
    let m = tail
        .iter()
        .chain(&lanes)
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let base = chunks.len() * 8;
    if let Some(j) = tail.iter().rposition(|&v| v == m) {
        return (base + j) as u32;
    }
    for (c, ch) in chunks.iter().enumerate().rev() {
        if ch.iter().fold(false, |hit, &v| hit | (v == m)) {
            return (c * 8 + ch.iter().rposition(|&v| v == m).unwrap()) as u32;
        }
    }
    (l.len() - 1) as u32 // all NaN
}

fn human_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    } else {
        format!("{b} B")
    }
}

#[cfg(test)]
mod tests {
    use super::argmax;

    fn reference(l: &[f32]) -> u32 {
        (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32
    }

    #[test]
    fn argmax_matches_max_by_including_ties() {
        let mut rng = 0x1357_9bdfu32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        for len in [1usize, 2, 7, 8, 9, 15, 16, 17, 100, 1000, 151_936] {
            for _ in 0..20 {
                // Few distinct values, so ties are common, all within one lane or across.
                let l: Vec<f32> = (0..len).map(|_| (next() % 9) as f32 - 4.0).collect();
                assert_eq!(argmax(&l), reference(&l), "len {len}");
                let l: Vec<f32> = (0..len).map(|_| next() as f32 / 1e7 - 200.0).collect();
                assert_eq!(argmax(&l), reference(&l), "len {len}");
            }
        }
        assert_eq!(argmax(&[f32::NEG_INFINITY; 5]), 4);
    }

    #[test]
    #[ignore = "timing, run with --ignored --nocapture"]
    fn argmax_speed() {
        let l: Vec<f32> = (0..151_936)
            .map(|i| ((i * 7919) % 10007) as f32 * 0.01)
            .collect();
        for (name, f) in [
            ("max_by", reference as fn(&[f32]) -> u32),
            ("argmax", argmax),
        ] {
            let t = std::time::Instant::now();
            let mut s = 0u64;
            for _ in 0..200 {
                s += f(std::hint::black_box(&l)) as u64;
            }
            eprintln!(
                "{name}: {:.1} us ({s})",
                t.elapsed().as_secs_f64() * 1e6 / 200.0
            );
        }
    }
}
