#[macro_use]
mod mv;
mod gguf;
mod kv;
mod model;
mod pool;
mod quant;
mod spec;
#[cfg(test)]
mod testutil;
mod tokenizer;
mod viz;
mod vmath;

use std::env;

/// Prompt tokens per `feed_batch` call.
const PREFILL_BATCH: usize = 64;

/// Draft tokens per speculative step unless --draft / AXON_DRAFT say
/// otherwise: 0, off. Drafting is lossless and K = 7 (a verify pass of K + 1
/// tokens fills whole groups of 4 slots, `quant::batch_slots`) is 1.2-4x
/// faster on repetitive text, but on low-repetition text it runs ~1-2%
/// slower than plain decode (BENCH.md), so it stays opt-in.
const DEFAULT_DRAFT: usize = 0;

/// K adjusted so K + 1 fills whole groups of 4 slots. Down for K >= 3
/// (4 -> 3, 8 -> 7, 12 -> 11): the guesses past the last full group would
/// each need a new group of 4 slots, mostly padding. Up for K = 1, 2 (-> 3):
/// their verify pass is padded to 4 slots anyway, so a third guess rides in
/// a slot that's computed regardless; a miss there costs nothing. 0 = off.
fn round_draft(k: usize) -> usize {
    match k {
        0 => 0,
        1..=3 => 3,
        _ => k - (k + 1) % 4,
    }
}

const USAGE: &str = "\
usage:
  axon <model.gguf>                         browse the model in a terminal UI
  axon info <model.gguf>                    print a summary and exit
  axon gen <model.gguf> \"<prompt>\" [-n N] [--draft K]
                                            greedy generation
  axon logits <model.gguf> \"<prompt>\" [-k K]
                                            top-K next-token logits after the prompt
  axon -h | --help                          this help
  axon -V | --version                       print the version

flags:
  -n N        gen: tokens to generate (default 32)
  --draft K   gen: draft up to K tokens per step by prompt lookup (default 0 = off; try 7);
              lossless: the output is the plain greedy output either way
  -k K        logits: how many logits to print (default 10)

environment:
  AXON_THREADS=N             worker threads (default: cores minus a quarter)
  AXON_SIMD=scalar|avx2|vnni force a kernel set (default: the best the CPU has)
  AXON_POOL_ADAPT=0          keep every worker in every job (no adaptive drop-out)
  AXON_POOL_DEBUG=1          gen: print thread-pool stats at the end
  AXON_DRAFT=K               same as --draft K
  AXON_MMAP=0|1              0: copy the model into huge-page memory, 1: map the file
                             (default: copy, unless the file is >60% of free memory)
  AXON_PREFETCH=N            decode prefetch distance in bytes (default 8192, 0 = off)
  AXON_DECODE_ROWS=1|2|4     decode rows per kernel call (default 4)
  AXON_KV=f16|f32            KV cache element type (default f16: half the memory and
                             attention bandwidth; f32 for A/B)

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
    // Speculative decoding: draft up to K tokens per step by prompt lookup
    // (spec.rs); output is the same either way. Off by default
    // (DEFAULT_DRAFT); --draft K or AXON_DRAFT=K turns it on, see round_draft.
    let mut max_draft = match env::var("AXON_DRAFT") {
        Ok(v) => v
            .parse::<usize>()
            .unwrap_or_else(|_| fail(&format!("AXON_DRAFT needs a whole number, got '{v}'"))),
        Err(_) => DEFAULT_DRAFT,
    };
    let mut i = 4;
    while i < args.len() {
        let val = args.get(i + 1).and_then(|s| s.parse::<usize>().ok());
        match (args[i].as_str(), val) {
            ("-n", Some(v)) => n_gen = v,
            ("-k", Some(v)) => top_k = v,
            ("--draft", Some(v)) => max_draft = v,
            (f @ ("-n" | "-k" | "--draft"), None) => fail(&format!(
                "{f} needs a whole number, got {}",
                args.get(i + 1)
                    .map_or("nothing".into(), |v| format!("'{v}'"))
            )),
            (f, _) => fail(&format!("unknown argument '{f}' for {cmd}")),
        }
        i += 2;
    }
    let max_draft = round_draft(max_draft);

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
    let logits = net.forward(last, rest.len(), &mut cache);
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
    // Characters can be split across tokens; the stream holds them back until whole.
    let mut text = tokenizer::Utf8Stream::default();
    let mut lookup = spec::PromptLookup::new();
    let t1 = Instant::now();
    let st = generate(
        &net,
        &mut cache,
        &ids,
        logits,
        GenOpts {
            n_gen,
            eos: tok.eos(),
            max_draft,
        },
        &mut |h, max| {
            lookup.sync(h);
            lookup.draft(max)
        },
        &mut |id| {
            let done = text.push(&tok.decode_bytes(&[id]));
            if !done.is_empty() {
                print!("{done}");
                out.flush()?;
            }
            Ok(())
        },
    )?;
    let decode_secs = t1.elapsed().as_secs_f64();
    print!("{}", text.finish());
    println!();
    eprintln!(
        "prompt eval: {} tok in {:.2}s ({:.2} tok/s)",
        ids.len(),
        prompt_secs,
        ids.len() as f64 / prompt_secs
    );
    // tok/s: decided tokens (picks after the prompt's, a final EOS included)
    // over wall time; without drafting that's one per forward, the old metric.
    eprintln!(
        "decode:      {} tok, {} passes in {:.2}s ({:.2} tok/s)",
        st.ids.len(),
        st.passes,
        decode_secs,
        if decode_secs > 0.0 {
            st.decided as f64 / decode_secs
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
    if max_draft == 0 {
        eprintln!("draft:       off");
    } else {
        let ms = |(n, secs): (usize, f64)| {
            format!(
                "{n} x {:.2} ms",
                if n > 0 { secs * 1e3 / n as f64 } else { 0.0 }
            )
        };
        let verify: Vec<String> = (0..st.verify.len())
            .filter(|&nb| st.verify[nb].0 > 0)
            .map(|nb| format!("nb{nb} {}", ms(st.verify[nb])))
            .collect();
        eprintln!(
            "draft:       K={}, {} drafted, {} accepted ({:.1}%), {:.2} tok/pass, \
             (decode tok/s = decided tok incl. EOS / s); forward {}; verify {}",
            max_draft,
            st.drafted,
            st.accepted,
            if st.drafted > 0 {
                100.0 * st.accepted as f64 / st.drafted as f64
            } else {
                0.0
            },
            if st.passes > 0 {
                st.decided as f64 / st.passes as f64
            } else {
                0.0
            },
            ms(st.forward),
            if verify.is_empty() {
                "none".to_string()
            } else {
                verify.join(", ")
            }
        );
    }
    Ok(())
}

/// What `generate` did.
#[derive(Default)]
struct GenStats {
    /// The emitted tokens (EOS excluded).
    ids: Vec<u32>,
    /// Weight passes: `forward` and `verify_argmax` calls.
    passes: usize,
    /// Tokens picked from logits those passes produced, a final EOS included.
    decided: usize,
    /// Draft tokens verified, and how many of them the model agreed with.
    drafted: usize,
    accepted: usize,
    /// (count, seconds) of `forward` passes, and of verify passes by batch
    /// size: verify[nb] for nb = 1 + guesses tokens.
    forward: (usize, f64),
    verify: Vec<(usize, f64)>,
}

/// `generate`'s settings.
struct GenOpts {
    /// Stop after this many tokens.
    n_gen: usize,
    /// Stop before this token.
    eos: u32,
    /// Draft up to this many tokens per pass; 0 = plain greedy.
    max_draft: usize,
}

/// Greedy decoding of up to `o.n_gen` tokens, stopping before `o.eos`. `logits`
/// is the prediction after `prompt`, whose tokens fill cache positions
/// 0..prompt.len(). Every token goes to `emit` as it's picked.
///
/// With `o.max_draft` > 0, `draft(history, max)` may guess up to `max` tokens
/// to follow history (prompt + emitted). [last token, guesses...] then go
/// through one `verify_argmax`, which gives `argmax` of each logits row that
/// `forward` would give (same logits bit for bit, same tie rule), and each
/// guess is kept only while it equals the model's own pick: pick i is the
/// right one exactly when guesses 0..i were the tokens emitted. So the output is the plain greedy output, whatever the drafts;
/// they only change how many tokens one weight pass yields. A rejected
/// guess's cache slot is rewritten by the next pass before anything reads it.
///
/// No low-acceptance gate: replaying the 5 bench prompts' exact greedy streams
/// through this loop and the real drafter (r5 policy sim) showed every gate
/// tried (backoff, n-gram EMA, expected-value) losing more on prose than it
/// saves on story; drafting there costs < 1% by the measured pass costs.
fn generate(
    net: &model::Qwen2,
    cache: &mut model::KvCache,
    prompt: &[u32],
    logits: Vec<f32>,
    o: GenOpts,
    draft: &mut dyn FnMut(&[u32], usize) -> Vec<u32>,
    emit: &mut dyn FnMut(u32) -> std::io::Result<()>,
) -> std::io::Result<GenStats> {
    let mut st = GenStats::default();
    let mut hist = prompt.to_vec();
    // (guess, the model's pick after it) from the last verify pass, in order.
    let mut queue = std::collections::VecDeque::new();
    let GenOpts {
        n_gen,
        eos,
        max_draft,
    } = o;
    let mut pick = argmax(&logits[..]);
    let mut picks = 0usize;
    while st.ids.len() < n_gen {
        let next = pick;
        picks += 1;
        if next == eos {
            break;
        }
        st.ids.push(next);
        hist.push(next);
        emit(next)?;
        let left = n_gen - st.ids.len();
        if left == 0 {
            break;
        }
        match queue.pop_front() {
            Some((guess, p)) if guess == next => {
                pick = p;
                st.accepted += 1;
                continue;
            }
            _ => queue.clear(),
        }
        // Guesses past the token after the last one needing logits are useless:
        // at most left - 1, so no pass goes beyond plain greedy's positions.
        let max = max_draft.min(left - 1);
        let mut guesses = if max > 0 {
            draft(&hist, max)
        } else {
            Vec::new()
        };
        guesses.truncate(max);
        let pos = hist.len() - 1;
        let t = std::time::Instant::now();
        if guesses.is_empty() {
            pick = argmax(&net.forward(next, pos, cache));
            st.forward.0 += 1;
            st.forward.1 += t.elapsed().as_secs_f64();
        } else {
            st.drafted += guesses.len();
            let mut toks = vec![next];
            toks.extend_from_slice(&guesses);
            let mut ps = net.verify_argmax(&toks, pos, cache).into_iter();
            pick = ps.next().expect("verify_argmax: one pick per token");
            queue.extend(guesses.into_iter().zip(ps));
            if st.verify.len() <= toks.len() {
                st.verify.resize(toks.len() + 1, (0, 0.0));
            }
            let v = &mut st.verify[toks.len()];
            v.0 += 1;
            v.1 += t.elapsed().as_secs_f64();
        }
        st.passes += 1;
    }
    st.decided = picks.saturating_sub(1);
    Ok(st)
}

multiversion! {
    /// Index of the largest logit, the last one on ties (as `Iterator::max_by`
    /// returns). Two passes that each vectorize: a lane-wise max, then a scan
    /// back from the end, 8 at a time, for the last logit equal to it. A vocab is
    /// ~152k logits, every token. Unlike a `total_cmp` max, NaN logits are never
    /// picked (unless all are NaN) and -0.0 ties with +0.0.
    fn argmax<V3>(l: &[f32]) -> u32 {
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
        let mut m = f32::NEG_INFINITY;
        for &b in tail.iter().chain(&lanes) {
            m = m.max(b);
        }
        let base = chunks.len() * 8;
        for (j, &v) in tail.iter().enumerate().rev() {
            if v == m {
                return (base + j) as u32;
            }
        }
        for (c, ch) in chunks.iter().enumerate().rev() {
            let mut hit = false;
            for &v in ch {
                hit |= v == m;
            }
            if hit {
                for (j, &v) in ch.iter().enumerate().rev() {
                    if v == m {
                        return (c * 8 + j) as u32;
                    }
                }
            }
        }
        (l.len() - 1) as u32 // all NaN
    }
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
    use super::{GenOpts, argmax, generate};
    use crate::model::{KvCache, Qwen2};
    use crate::spec::PromptLookup;

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

    /// argmax's portable and x86-64-v3 variants pick the same index.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn argmax_multiversion_identical() {
        use crate::model::tests::{can_v3, mixed};
        if !can_v3() {
            return;
        }
        for n in [1usize, 7, 8, 9, 64, 151_936] {
            for (k, edgy) in [false, true].into_iter().enumerate() {
                let mut l = mixed(n, 5 + k as u64, 9.0, edgy);
                let a = super::argmax::portable(&l);
                assert_eq!(a, unsafe { super::argmax::v3(&l) }, "n={n}");
                l.iter_mut().for_each(|v| *v = v.round()); // ties
                let a = super::argmax::portable(&l);
                assert_eq!(a, unsafe { super::argmax::v3(&l) }, "ties n={n}");
            }
        }
        let nan = [f32::NAN; 11];
        assert_eq!(super::argmax::portable(&nan), unsafe {
            super::argmax::v3(&nan)
        });
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

    /// Cache holding `ids` and the logits after them, as `run_lm` does it.
    fn prefill(net: &Qwen2, ids: &[u32], cap: usize) -> (KvCache, Vec<f32>) {
        let mut cache = net.new_cache(cap);
        let (&last, rest) = ids.split_last().unwrap();
        for (i, batch) in rest.chunks(super::PREFILL_BATCH).enumerate() {
            net.feed_batch(batch, i * super::PREFILL_BATCH, &mut cache);
        }
        let logits = net.forward(last, rest.len(), &mut cache);
        (cache, logits)
    }

    const PROMPTS: [&str; 2] = [
        "fn add(a: i32, b: i32) -> i32 { a + b }\nfn sub(a: i32, b: i32) -> i32 {",
        "The quick brown fox jumps over the lazy dog. The quick brown fox",
    ];

    /// verify_batch's row b must equal `forward` on token b, bit for bit,
    /// also when its cache slots hold a rejected earlier draft.
    #[test]
    fn verify_batch_matches_forward() {
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
        let net = Qwen2::new(&m);
        let ids = tok.encode(PROMPTS[0]);
        let toks: Vec<u32> = (0..9u32).map(|i| 1000 + i * 7919 % 50_000).collect();
        let (mut a, _) = prefill(&net, &ids, ids.len() + toks.len());
        let (mut b, _) = prefill(&net, &ids, ids.len() + toks.len());
        let want: Vec<Vec<f32>> = toks
            .iter()
            .enumerate()
            .map(|(i, &t)| net.forward(t, ids.len() + i, &mut a))
            .collect();
        // Stale slots: a wrong draft at the same positions first.
        let junk: Vec<u32> = toks.iter().map(|t| t + 13).collect();
        net.verify_batch(&junk, ids.len(), &mut b);
        for n in 1..=toks.len() {
            let got = net.verify_batch(&toks[..n], ids.len(), &mut b);
            assert_eq!(got.len(), n);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.len(), w.len());
                if let Some(j) = (0..g.len()).find(|&j| g[j].to_bits() != w[j].to_bits()) {
                    panic!("batch {n} row {i} logit {j}: {} vs {}", g[j], w[j]);
                }
            }
        }
    }

    /// verify_argmax's picks must be `argmax` of `forward`'s logits per
    /// position, index for index, for every batch size 1..=17 (all padding
    /// remainders, more than one group of 4 and past the default K + 1).
    #[test]
    fn verify_argmax_matches_forward_argmax() {
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
        let net = Qwen2::new(&m);
        let ids = tok.encode(PROMPTS[1]);
        let toks: Vec<u32> = (0..17u32).map(|i| 500 + i * 7919 % 60_000).collect();
        let (mut a, _) = prefill(&net, &ids, ids.len() + toks.len());
        let (mut b, _) = prefill(&net, &ids, ids.len() + toks.len());
        let want: Vec<u32> = toks
            .iter()
            .enumerate()
            .map(|(i, &t)| argmax(&net.forward(t, ids.len() + i, &mut a)))
            .collect();
        for n in 1..=toks.len() {
            let got = net.verify_argmax(&toks[..n], ids.len(), &mut b);
            assert_eq!(got, want[..n], "batch {n}");
        }
        // And the logits path agrees with it on the same cache.
        let rows = net.verify_batch(&toks, ids.len(), &mut b);
        let from_rows: Vec<u32> = rows.iter().map(|r| argmax(r)).collect();
        assert_eq!(from_rows, want);
    }

    /// `Best`, pushed in order or merged over consecutive ranges of any
    /// size, must pick exactly `argmax`'s index: ties (last wins), -0.0 vs
    /// +0.0, NaNs, -inf, all-NaN.
    #[test]
    fn best_matches_argmax() {
        use crate::model::Best;
        let mut rng = 0x2468_ace1u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        let pick = |l: &[f32], chunk: usize| {
            let mut acc = Best::NONE;
            for (c, ch) in l.chunks(chunk).enumerate() {
                let mut b = Best::NONE;
                for (j, &v) in ch.iter().enumerate() {
                    b.push(v, (c * chunk + j) as u32);
                }
                acc.merge(b);
            }
            acc.index(l.len())
        };
        let vals = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.5,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        for len in [1usize, 2, 7, 8, 9, 31, 100, 1000] {
            for round in 0..200 {
                // Few distinct values so ties are common, randomly negated;
                // smaller pools leave out inf/NaN/2.5/±1 so the max varies.
                let pool = [8, 6, 5, 4, 2][round % 5];
                let l: Vec<f32> = (0..len)
                    .map(|_| {
                        let r = next() as usize;
                        let v = vals[r % pool];
                        if (r >> 8).is_multiple_of(7) { -v } else { v }
                    })
                    .collect();
                for chunk in [1, 2, 3, 8, 64, len] {
                    assert_eq!(pick(&l, chunk), argmax(&l), "{l:?} chunk {chunk}");
                }
            }
            let nan = vec![f32::NAN; len];
            assert_eq!(pick(&nan, 3), argmax(&nan));
            let ninf = vec![f32::NEG_INFINITY; len];
            assert_eq!(pick(&ninf, 3), argmax(&ninf));
        }
    }

    #[test]
    fn draft_k_rounds_to_full_slot_groups() {
        use super::round_draft;
        let got: Vec<usize> = (0..=13).map(round_draft).collect();
        assert_eq!(got, [0, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11]);
    }

    /// Greedy output with drafting must equal plain greedy, token for token,
    /// for drafts that are always right, always wrong, random, and real
    /// prompt lookup; for several K, -n limits, and an EOS hit mid-way.
    #[test]
    fn drafting_matches_greedy() {
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
        let net = Qwen2::new(&m);
        const N: usize = 24;
        for prompt in PROMPTS {
            let ids = tok.encode(prompt);
            let cap = ids.len() + N + 1;
            // Plain greedy reference: every pick, as the original loop does it.
            let (mut cache, mut logits) = prefill(&net, &ids, cap);
            let mut greedy = Vec::new();
            for step in 0..N {
                greedy.push(argmax(&logits));
                if step + 1 < N {
                    logits = net.forward(greedy[step], ids.len() + step, &mut cache);
                }
            }
            // Also stop at a token greedy emits mid-way, as if it were EOS.
            let eos_mid = greedy[N / 2];
            for (eos, n) in [
                (tok.eos(), 1),
                (tok.eos(), 2),
                (tok.eos(), 7),
                (tok.eos(), N),
                (eos_mid, N),
            ] {
                let mut want: Vec<u32> = greedy[..n].to_vec();
                if let Some(e) = want.iter().position(|&t| t == eos) {
                    want.truncate(e);
                }
                for k in [0, 1, 3, 8] {
                    for mode in ["right", "wrong", "random", "lookup"] {
                        if k == 0 && mode != "right" {
                            continue; // no drafting: the mode is moot
                        }
                        let mut rng = 0x9e37_79b9u32 ^ k as u32;
                        let mut lookup = PromptLookup::new();
                        let mut draft = |h: &[u32], max: usize| -> Vec<u32> {
                            let at = h.len() - ids.len(); // index of the next pick
                            let right = |i: usize| greedy.get(at + i).copied().unwrap_or(7);
                            match mode {
                                "right" => (0..max).map(right).collect(),
                                "wrong" => (0..max).map(|i| right(i) ^ 1).collect(),
                                "random" => (0..max)
                                    .map(|i| {
                                        rng ^= rng << 13;
                                        rng ^= rng >> 17;
                                        rng ^= rng << 5;
                                        if rng.is_multiple_of(3) {
                                            right(i) ^ 1
                                        } else {
                                            right(i)
                                        }
                                    })
                                    .collect(),
                                _ => {
                                    lookup.sync(h);
                                    lookup.draft(max)
                                }
                            }
                        };
                        let (mut cache, logits) = prefill(&net, &ids, cap);
                        let mut emitted = Vec::new();
                        let st = generate(
                            &net,
                            &mut cache,
                            &ids,
                            logits,
                            GenOpts {
                                n_gen: n,
                                eos,
                                max_draft: k,
                            },
                            &mut draft,
                            &mut |t| {
                                emitted.push(t);
                                Ok(())
                            },
                        )
                        .unwrap();
                        let ctx = format!("{prompt:?} n {n} eos {eos} k {k} {mode}");
                        assert_eq!(st.ids, want, "{ctx}");
                        assert_eq!(emitted, want, "{ctx}");
                        assert!(st.accepted <= st.drafted, "{ctx}");
                        if k == 0 {
                            assert_eq!(st.drafted, 0, "{ctx}");
                        }
                        if mode == "wrong" {
                            assert_eq!(st.accepted, 0, "{ctx}");
                        }
                        if mode == "right" && k > 0 && n == N && eos != eos_mid {
                            assert!(st.passes < n - 1, "{ctx}: {} passes", st.passes);
                        }
                    }
                }
            }
        }
    }
}
