#[macro_use]
mod mv;
mod chat;
mod cuda;
mod gguf;
mod hybrid;
mod kv;
mod lineedit;
mod lm;
mod model;
mod pool;
mod quant;
mod sample;
mod spec;
#[cfg(test)]
mod testutil;
mod tokenizer;
mod vmath;

use lm::{Backend, Device, Lm};
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
  axon <model.gguf>                         same as axon info
  axon info <model.gguf>                    print a summary and exit
  axon gen <model.gguf> \"<prompt>\" [-n N] [--draft K] [--device D] [--gpu-layers N]
            [--kv T] [sampling flags]       generation (greedy unless --temp > 0)
  axon logits <model.gguf> \"<prompt>\" [-k K]
                                            top-K next-token logits after the prompt
  axon chat <model.gguf> [--system TEXT] [--ctx N] [-n N] [--draft K] [--device D]
            [--gpu-layers N] [--kv T] [sampling flags]
                                            talk to an instruct model (sampled; --temp 0: greedy)
  axon -h | --help                          this help
  axon -V | --version                       print the version

flags:
  -n N        gen: tokens to generate (default 32); chat: most tokens per reply (default: no limit)
  --draft K   gen, chat: draft up to K tokens per step by prompt lookup (default 0 = off; try 7);
              lossless: the output is the same either way (greedy, or sampled with the same seed)
  -k K        logits: how many logits to print (default 10)
  --device D  gen, chat: cpu (default) or cuda (an NVIDIA GPU; a model too big for its memory
              is split: the last layers that fit on the GPU, the first ones on the CPU)
  --gpu-layers N  gen, chat: with --device cuda, run the last N layers (and the output head) on
              the GPU and the rest on the CPU (0 = all on the CPU, as --device cpu; default: all
              if they fit with 256 MiB to spare, else as many as fit with another 640 MiB left
              for the desktop; an explicit N only needs the 256 MiB)
  --kv T      gen, chat: KV cache type, f16 (default) or q8 (cuda only: int8 with a scale per
              position and head, half the memory; slightly different results; on a split, the
              GPU layers' cache, so more layers fit); as AXON_KV
  --system T  chat: the system message (default: Qwen2.5's)
  --ctx N     chat: context size in tokens (default: the model's, at most 8192; on cuda, if the
              whole model doesn't fit with it, less if at least half (or 4096) fits, else a
              split); when the conversation outgrows it, the oldest exchanges are forgotten

sampling flags (gen, chat; applied in this order, as llama.cpp):
  --repeat-penalty P  divide (multiply, if negative) the logits of tokens among the last N by P
                      (default 1.1; 1 = off)
  --repeat-last-n N   how many tokens back the penalty looks, prompt included (default 64; 0 = off)
  --top-k K           keep the K most likely tokens (default 40; 0 = off: all of them, slower,
                      as is K > 256 on the GPU)
  --top-p P           then the fewest whose probability adds up to P (default 0.9; 1 = off)
  --min-p P           then those at least P times as likely as the top one (default 0 = off)
  --temp T            then divide the logits by T and draw (default: gen 0, chat 0.8);
                      0 = greedy: the highest logit, no penalty, no other flag applies
  --seed S            the random seed (default: a new one each run, printed by gen)

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
  AXON_KV=f16|f32|q8         KV cache element type (default f16: half the memory and
                             attention bandwidth; f32 for A/B, CPU only; q8 cuda only)
  AXON_DEVICE=cpu|cuda       same as --device
  AXON_GPU_LAYERS=N          same as --gpu-layers N
  AXON_STREAM=0              split model: don't stream the CPU's layers through the GPU for
                             prefill batches (default on: their weights are copied over
                             once per prompt chunk; results then get the GPU's numerics)
  AXON_STREAM_MIN=N          split model: smallest prefill batch streamed (default 32);
                             streaming pins the CPU layers' KV cache in RAM (32B at
                             --ctx 8192: ~1.2 GB, resident while the cache lives)
  AXON_STREAM_DECODE=1       experiment: stream one-token decode steps too (slower: ~2x
                             on 32B; not with --draft)
  AXON_CUDA_GRAPHS=0         cuda: launch each token's kernels one by one instead of
                             replaying a captured CUDA graph (same results, slower)

The model must be a qwen2-architecture GGUF (e.g. Qwen2.5-0.5B-Instruct Q4_K_M).";

/// AXON_STREAM_DECODE=1 with drafting: refused (drafting's picks come
/// from verify passes, which never stream, so they could differ from the
/// streamed per-token steps: not lossless).
fn check_stream_decode(max_draft: usize) {
    if max_draft > 0 && hybrid::stream_decode_setting() {
        fail("AXON_STREAM_DECODE=1 can't be combined with --draft / AXON_DRAFT (not lossless)");
    }
}

/// Usage error: one line, a hint, exit code 2.
fn fail(msg: &str) -> ! {
    eprintln!("axon: {msg}");
    eprintln!("try 'axon --help'");
    std::process::exit(2);
}

/// Load a GGUF, exiting with code 2 and a one-line message if it can't be read.
fn open_model(path: &str, lm: bool) -> gguf::Model {
    open_model_with(path, lm, None)
}

/// `open_model` with `gguf::load_with`'s copy choice (the GPU reads the
/// weights once, so mapping the file beats copying it).
fn open_model_with(path: &str, lm: bool, copy: Option<bool>) -> gguf::Model {
    let m = match copy {
        None => gguf::load(path),
        c => gguf::load_with(path, c),
    };
    let m = m.unwrap_or_else(|e| match e.kind() {
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
    if args[1] == "chat" {
        return run_chat(&args);
    }

    // `axon info <model>`, or just `axon <model>`.
    let path = match args[1].as_str() {
        "info" if args.len() == 3 => &args[2],
        "info" => fail("usage: axon info <model.gguf>"),
        f if f.starts_with('-') => fail(&format!("unknown flag '{f}'")),
        // A lone argument is a model to summarize, unless it can't be one.
        p if args.len() == 2 && (p.ends_with(".gguf") || std::path::Path::new(p).exists()) => {
            &args[1]
        }
        c => fail(&format!(
            "unknown command '{c}' (commands: gen, logits, chat, info)"
        )),
    };

    let model = open_model(path, false);
    println!("axon v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!(
        "{} | {}L / {}d / {}h (kv {}) | ctx {}",
        model.arch,
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
    println!("{}", cuda::summary());
    Ok(())
}

/// Speculative decoding: draft up to K tokens per step by prompt lookup
/// (spec.rs); output is the same either way. Off by default (DEFAULT_DRAFT);
/// --draft K or AXON_DRAFT=K turns it on, see round_draft. This is AXON_DRAFT.
fn draft_env() -> usize {
    match env::var("AXON_DRAFT") {
        Ok(v) => v
            .parse::<usize>()
            .unwrap_or_else(|_| fail(&format!("AXON_DRAFT needs a whole number, got '{v}'"))),
        Err(_) => DEFAULT_DRAFT,
    }
}

/// KV cache type (kv.rs): checked up front so a typo is a usage error, not a
/// panic in the first new_cache. --kv overrides it.
fn kv_env() -> kv::KvType {
    kv::kv_setting().unwrap_or_else(|e| fail(&e))
}

/// --kv's value (None: the flag was last, without one): f16 or q8. f32
/// (CPU only) is AXON_KV's alone; --kv f16 over AXON_KV=f32 means f16.
fn kv_flag(v: Option<&str>) -> kv::KvType {
    let v = v.unwrap_or_else(|| fail("--kv needs a value (f16 or q8)"));
    match kv::KvType::parse(v) {
        Ok(kv::KvType::F32) | Err(_) => fail(&format!("--kv takes f16 or q8, got '{v}'")),
        Ok(t) => {
            // The CPU reads its f16/f32 choice where it makes caches.
            kv::set_f32(false);
            t
        }
    }
}

/// --device's default: $AXON_DEVICE, else the CPU.
fn device_env() -> Device {
    Device::from_env().unwrap_or_else(|e| fail(&e))
}

/// $AXON_GPU_LAYERS (--gpu-layers' default on CUDA), if set.
fn gpu_layers_env() -> Option<usize> {
    let v = env::var("AXON_GPU_LAYERS").ok()?;
    Some(
        v.parse::<usize>()
            .unwrap_or_else(|_| fail(&format!("AXON_GPU_LAYERS needs a whole number, got '{v}'"))),
    )
}

/// --device and --gpu-layers (else $AXON_GPU_LAYERS on CUDA) as the
/// backend to load: a usage error if --gpu-layers came without CUDA, and
/// 0 GPU layers is the plain CPU backend (loaded as --device cpu loads it:
/// a copy of the file, not a map). Before `open_model_with`.
fn placement(device: Device, gpu_layers: Option<usize>) -> (Device, Option<usize>) {
    if device == Device::Cpu && gpu_layers.is_some() {
        fail("--gpu-layers needs --device cuda");
    }
    let gpu_layers = gpu_layers.or_else(|| (device == Device::Cuda).then(gpu_layers_env)?);
    match gpu_layers {
        Some(0) => (Device::Cpu, None),
        g => (device, g),
    }
}

/// The model on `device` with room for `ctx` tokens, split as `gpu_layers`
/// says (`placement`'s; see `Backend::new`); a usage error if the GPU can't
/// be used (no driver or device, too little memory, ...).
fn backend<'a>(
    model: &'a gguf::Model,
    device: Device,
    ctx: usize,
    gpu_layers: Option<usize>,
    kv: kv::KvType,
    fit: bool,
) -> Backend<'a> {
    Backend::new(model, device, ctx, gpu_layers, kv, fit).unwrap_or_else(|e| match device {
        Device::Cuda => fail(&format!("--device cuda: {e}")),
        Device::Cpu => fail(&e),
    })
}

/// The copy-vs-map choice for loading a model for `device`.
fn copy_for(device: Device) -> Option<bool> {
    (device == Device::Cuda).then_some(false)
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
    let mut max_draft = draft_env();
    let mut device = device_env();
    let mut gpu_layers = None;
    let mut kv = kv_env();
    let mut sargs = sample::SampleArgs::default();
    let mut i = 4;
    while i < args.len() {
        let val = args.get(i + 1).and_then(|s| s.parse::<usize>().ok());
        match (args[i].as_str(), val) {
            (f, _) if sample::SampleArgs::is_flag(f) => sargs
                .set(f, args.get(i + 1).map(String::as_str))
                .unwrap_or_else(|e| fail(&e)),
            ("--device", _) => {
                let v = args.get(i + 1).map_or("", String::as_str);
                device = Device::parse(v).unwrap_or_else(|e| fail(&e));
            }
            ("--kv", _) => kv = kv_flag(args.get(i + 1).map(String::as_str)),
            ("-n", Some(v)) => n_gen = v,
            ("-k", Some(v)) => top_k = v,
            ("--draft", Some(v)) => max_draft = v,
            ("--gpu-layers", Some(v)) => gpu_layers = Some(v),
            (f @ ("-n" | "-k" | "--draft" | "--gpu-layers"), None) => fail(&format!(
                "{f} needs a whole number, got {}",
                args.get(i + 1)
                    .map_or("nothing".into(), |v| format!("'{v}'"))
            )),
            (f, _) => fail(&format!("unknown argument '{f}' for {cmd}")),
        }
        i += 2;
    }
    let max_draft = round_draft(max_draft);
    check_stream_decode(max_draft);
    // gen is greedy unless --temp says otherwise: reproducible runs and benchmarks.
    let mut sampler = sargs.resolve(0.0).map(sample::Sampler::new);
    if sampler.is_none() && cmd == "gen" && sargs.any_besides_temp() {
        eprintln!("axon: note: gen is greedy without --temp > 0; the sampling flags do nothing");
    }

    if let Some(s) = &sampler {
        // Before generating: a run that's cut short can still be repeated.
        let c = s.cfg;
        eprintln!(
            "sampling:    temp {} top-k {} top-p {} min-p {} repeat-penalty {} repeat-last-n {} seed {}",
            c.temp, c.top_k, c.top_p, c.min_p, c.repeat_penalty, c.repeat_last_n, c.seed
        );
    }

    let (device, gpu_layers) = placement(device, gpu_layers);
    let model = open_model_with(path, true, copy_for(device));
    let tok = tokenizer::Tokenizer::from_gguf(&model);

    let ids = tok.encode(prompt);
    if ids.is_empty() {
        eprintln!("prompt encodes to zero tokens");
        std::process::exit(1);
    }
    let extra = if cmd == "gen" { n_gen } else { 0 };
    let net = backend(&model, device, ids.len() + extra + 1, gpu_layers, kv, false);
    let mut cache = net.new_cache(ids.len() + extra + 1);

    // Feed the prompt in batches (the backend's size: PREFILL_BATCH on the
    // CPU), so each weight row is read once per batch instead of once per
    // token. Only the last position's logits are used, so the others skip
    // the output head.
    let t0 = Instant::now();
    let (&last, rest) = ids.split_last().expect("checked non-empty above");
    let pb = net.prefill_batch();
    for (i, batch) in rest.chunks(pb).enumerate() {
        net.feed_batch(batch, i * pb, &mut cache);
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
            net.vocab()
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
            stop: &[tok.eos()],
            max_draft,
            sample: sampler.as_mut(),
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
    if let Backend::Cuda(n) = &net {
        eprintln!(
            "device:      cuda, {} of weights in VRAM ({} read per token), KV cache {} {}, \
             CUDA graphs {}",
            human_bytes(n.weight_bytes() as u64),
            human_bytes(n.bytes_per_token(true) as u64),
            n.kv_name(),
            human_bytes(n.kv_cache_bytes(n.kv, ids.len() + extra + 1) as u64),
            if n.graphs { "on" } else { "off" }
        );
    }
    if let Backend::Hybrid(n) = &net {
        eprintln!(
            "device:      cuda + cpu, {}, kv {}",
            n.describe(ids.len() + extra + 1),
            n.kv_name()
        );
    }
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

/// Context size `axon chat` uses unless --ctx says otherwise (if the model was
/// trained on at least this much). The cache is allocated lazily, page by
/// page as the conversation grows, so a big one costs nothing up front.
const DEFAULT_CHAT_CTX: usize = 8192;

const CHAT_HELP: &str = concat!(
    "  /reset          forget the conversation (the system message stays)\n",
    "  /system TEXT    a new system message (forgets the conversation)\n",
    "  /exit           quit (or Ctrl-D)\n",
    "  end a line with \\ (or press Alt-Enter) to go on with the message on a new line;\n",
    "  pasted text stays one message, line breaks and all; //text sends /text\n",
    "keys: Left/Right, Ctrl-Left/Right or Alt-B/F (word), Home/End or Ctrl-A/E,\n",
    "  Backspace/Delete, Ctrl-W (delete word back), Ctrl-U/Ctrl-K (delete to start/end),\n",
    "  Ctrl-Y (paste deleted text), Up/Down (history), Ctrl-L (clear screen),\n",
    "  Ctrl-C (stop a reply; clear the line), Ctrl-D on an empty line (quit)",
);

/// `chat` subcommand: args = [axon, chat, model, flags...]. Each line of
/// stdin is a message; on a terminal it's an interactive session.
fn run_chat(args: &[String]) -> std::io::Result<()> {
    const CHAT_USAGE: &str = "usage: axon chat <model.gguf> [--system TEXT] [--ctx N] [-n N] [--draft K] [--device D] \
         [--gpu-layers N] [--kv T]";
    let Some(path) = args.get(2).filter(|p| !p.starts_with('-')) else {
        fail(CHAT_USAGE);
    };
    let mut system = chat::DEFAULT_SYSTEM.to_string();
    let mut ctx = None;
    let mut max_reply = usize::MAX;
    let mut max_draft = draft_env();
    let mut device = device_env();
    let mut gpu_layers = None;
    let mut kv = kv_env();
    let mut sargs = sample::SampleArgs::default();
    let mut i = 3;
    while i < args.len() {
        let f = args[i].as_str();
        if sample::SampleArgs::is_flag(f) {
            sargs
                .set(f, args.get(i + 1).map(String::as_str))
                .unwrap_or_else(|e| fail(&e));
            i += 2;
            continue;
        }
        if !matches!(
            f,
            "--system" | "--ctx" | "-n" | "--draft" | "--device" | "--gpu-layers" | "--kv"
        ) {
            fail(&format!("unknown argument '{f}' for chat"));
        }
        let Some(v) = args.get(i + 1) else {
            fail(&format!("{f} needs a value"));
        };
        if f == "--system" {
            system = v.clone();
        } else if f == "--device" {
            device = Device::parse(v).unwrap_or_else(|e| fail(&e));
        } else if f == "--kv" {
            kv = kv_flag(Some(v));
        } else {
            let n = v
                .parse::<usize>()
                .ok()
                .filter(|&n| n > 0 || f == "--draft" || f == "--gpu-layers")
                .unwrap_or_else(|| fail(&format!("{f} needs a positive whole number, got '{v}'")));
            match f {
                "--ctx" => ctx = Some(n),
                "-n" => max_reply = n,
                "--gpu-layers" => gpu_layers = Some(n),
                _ => max_draft = n,
            }
        }
        i += 2;
    }

    let (device, gpu_layers) = placement(device, gpu_layers);
    let model = open_model_with(path, true, copy_for(device));
    // RoPE past the trained length gives garbage; 0 = the GGUF doesn't say.
    let trained = model.ctx_len() as usize;
    let fit = ctx.is_none();
    let ctx = match ctx {
        Some(n) if trained > 0 && n > trained => fail(&format!(
            "--ctx {n} is past the {trained} tokens {path} was trained on"
        )),
        Some(n) => n,
        None if trained > 0 => trained.min(DEFAULT_CHAT_CTX),
        None => DEFAULT_CHAT_CTX,
    };
    let tok = tokenizer::Tokenizer::from_gguf(&model);
    // The default context may shrink to keep the model on the GPU (Backend::new).
    let net = backend(&model, device, ctx, gpu_layers, kv, fit);
    let (want, ctx) = (ctx, net.ctx(ctx));
    if ctx < want {
        eprintln!(
            "axon: a {want}-token context doesn't fit in GPU memory with the whole model; using \
             {ctx} (--ctx {want} splits it across the CPU and GPU instead, slower; --gpu-layers N \
             to choose the split{})",
            if kv == kv::KvType::Q8 {
                ""
            } else {
                "; --kv q8 halves the KV cache"
            }
        );
    }
    let mut conv =
        chat::Chat::new(&net, &tok, &system, ctx).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    conv.max_reply = max_reply;
    conv.max_draft = round_draft(max_draft);
    check_stream_decode(conv.max_draft);
    // chat samples by default (as ollama does): greedy replies loop.
    conv.sampler = sargs
        .resolve(sample::DEFAULT_TEMP)
        .map(sample::Sampler::new);

    let name = std::path::Path::new(path)
        .file_name()
        .map_or(path.as_str().into(), |n| n.to_string_lossy());
    let kvs = match &net {
        Backend::Cuda(n) => format!(" | kv {}", n.kv_name()),
        Backend::Hybrid(n) => format!(" | {} | kv {}", n.describe(ctx), n.kv_name()),
        Backend::Cpu(_) => String::new(),
    };
    let seed = conv
        .sampler
        .as_ref()
        .map_or(" | greedy".into(), |s| format!(" | seed {}", s.cfg.seed));
    match chat_repl(&mut conv, &tok, &format!("{name} | ctx {ctx}{kvs}{seed}")) {
        // stdout went away (a closed pipe): nobody is listening.
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        r => r,
    }
}

/// One message from piped stdin: a line, or several joined while they end
/// in a backslash. (text, whether stdin ended).
fn read_piped_message(input: &mut impl std::io::BufRead) -> std::io::Result<(String, bool)> {
    let mut msg = String::new();
    loop {
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            return Ok((msg, true));
        }
        let line = line.trim_end_matches(['\n', '\r']);
        match line.strip_suffix('\\') {
            Some(head) => {
                msg.push_str(head);
                msg.push('\n');
            }
            None => {
                msg.push_str(line);
                return Ok((msg, false));
            }
        }
    }
}

/// The `chat` read-reply loop. Output goes through `write!`, not `print!`,
/// so a closed stdout is an error to return rather than a panic.
///
/// On a terminal, messages come from the line editor, and the terminal
/// stays in raw mode throughout (see lineedit.rs): keys typed while a reply
/// streams are polled for between tokens, Ctrl-C stopping the reply and the
/// rest replayed at the next prompt. Raw mode means "\r\n" for each "\n".
/// Piped, each line of stdin is a message and SIGINT stops a reply.
fn chat_repl<M: Lm>(
    conv: &mut chat::Chat<'_, M>,
    tok: &tokenizer::Tokenizer,
    banner: &str,
) -> std::io::Result<()> {
    use std::io::{IsTerminal, Write};

    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    // Interactive: the line editor only when a person is at both ends.
    let tty = stdin.is_terminal() && out.is_terminal();
    let err_tty = std::io::stderr().is_terminal();
    let (dim, undim) = if err_tty {
        ("\x1b[2m", "\x1b[0m")
    } else {
        ("", "")
    };
    // Line ends for stdout, and for stderr notes.
    let nl = if tty { "\r\n" } else { "\n" };
    let enl = if tty && err_tty { "\r\n" } else { "\n" };
    let note = |s: &str| eprint!("{}{enl}", s.replace('\n', enl));
    if tty {
        eprintln!("{dim}axon chat | {banner} | /help for commands and keys, Ctrl-D quits{undim}");
    }
    let _raw = if tty {
        Some(lineedit::RawMode::enter()?)
    } else {
        None
    };
    let mut editor = lineedit::LineReader::default();
    let typeahead = std::cell::RefCell::new(std::collections::VecDeque::new());
    let mut input = stdin.lock();
    loop {
        let (msg, eof) = if tty {
            match editor.read(&mut out, &mut typeahead.borrow_mut())? {
                Some(m) => (m, false),
                None => return Ok(()),
            }
        } else {
            read_piped_message(&mut input)?
        };
        let text = msg.trim();
        if eof && text.is_empty() {
            return Ok(());
        }
        if text.is_empty() {
            continue;
        }
        if let Some(cmd) = text.strip_prefix('/')
            && !cmd.starts_with('/')
        {
            let (name, arg) = cmd.split_once(char::is_whitespace).unwrap_or((cmd, ""));
            match name {
                "exit" | "quit" | "bye" => return Ok(()),
                "reset" | "clear" => {
                    conv.reset();
                    note(&format!("{dim}(conversation cleared){undim}"));
                }
                "system" if arg.trim().is_empty() => note("axon: usage: /system TEXT"),
                "system" => match conv.set_system(arg.trim()) {
                    Ok(()) => note(&format!(
                        "{dim}(new system message; conversation cleared){undim}"
                    )),
                    Err(e) => note(&format!("axon: {e}")),
                },
                "help" | "?" => note(CHAT_HELP),
                _ => note(&format!("axon: unknown command /{name}; /help lists them")),
            }
            if eof {
                return Ok(());
            }
            continue;
        }
        let text = text.strip_prefix('/').unwrap_or(text); // "//x" sends "/x"

        // Characters can be split across tokens; the stream holds them back until whole.
        let mut stream = tokenizer::Utf8Stream::default();
        // Polled before every token: one non-blocking poll(2) when no key waits.
        let stopped = std::cell::Cell::new(false);
        // Keeps draining after a stop, so a mashed Ctrl-C is spent here, not at the prompt.
        let poll_keys = || {
            if lineedit::poll_typeahead(&mut typeahead.borrow_mut()) {
                stopped.set(true);
            }
            stopped.get()
        };
        let cancel: &dyn Fn() -> bool = if tty {
            &poll_keys
        } else {
            chat::sigint::arm();
            &chat::sigint::hit
        };
        let res = conv.reply(text, cancel, &mut |id| {
            let s = stream.push(&tok.decode_bytes(&[id]));
            if !s.is_empty() {
                if tty {
                    write!(out, "{}", s.replace('\n', "\r\n"))?;
                } else {
                    write!(out, "{s}")?;
                }
                out.flush()?;
            }
            Ok(())
        });
        if !tty {
            chat::sigint::disarm();
        }
        let r = match res {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => return Err(e),
            Err(e) => {
                note(&format!("axon: {e}"));
                continue;
            }
        };
        write!(out, "{}{nl}", stream.finish().replace('\n', nl))?;
        out.flush()?;
        let rate = |n: usize, s: f64| if s > 0.0 { n as f64 / s } else { 0.0 };
        let (used, size) = conv.used();
        let mut notes = String::new();
        if r.drafted > 0 {
            notes.push_str(&format!(" | drafts {}/{} kept", r.accepted, r.drafted));
        }
        if r.interrupted {
            notes.push_str(" | stopped");
        } else if r.cut {
            notes.push_str(&format!(" | cut at {} tokens", r.ids.len()));
        }
        if r.dropped > 0 {
            notes.push_str(&format!(
                " | forgot the oldest {} exchange{} to fit",
                r.dropped,
                if r.dropped == 1 { "" } else { "s" }
            ));
        }
        note(&format!(
            "{dim}[{} tok, {:.1} tok/s | prompt {} tok, {:.0} tok/s | ctx {used}/{size}{notes}]{undim}",
            r.ids.len(),
            rate(r.decided, r.decode_secs),
            r.prompt_tokens,
            rate(r.prompt_tokens, r.prompt_secs),
        ));
        if tty {
            write!(out, "{nl}")?;
        }
        if eof {
            return Ok(());
        }
    }
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
struct GenOpts<'a> {
    /// Stop after this many tokens.
    n_gen: usize,
    /// Stop before any of these tokens.
    stop: &'a [u32],
    /// Draft up to this many tokens per pass; 0 = plain greedy.
    max_draft: usize,
    /// Sample (sample.rs) instead of picking greedily.
    sample: Option<&'a mut sample::Sampler>,
}

/// Greedy decoding of up to `o.n_gen` tokens, stopping before any of `o.stop`. `logits`
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
fn generate<M: Lm>(
    net: &M,
    cache: &mut M::Cache,
    prompt: &[u32],
    logits: Vec<f32>,
    o: GenOpts<'_>,
    draft: &mut dyn FnMut(&[u32], usize) -> Vec<u32>,
    emit: &mut dyn FnMut(u32) -> std::io::Result<()>,
) -> std::io::Result<GenStats> {
    if let Some(s) = o.sample {
        return sample::generate(
            net,
            cache,
            prompt,
            logits,
            o.n_gen,
            o.stop,
            o.max_draft,
            s,
            draft,
            emit,
        );
    }
    let mut st = GenStats::default();
    let mut hist = prompt.to_vec();
    // (guess, the model's pick after it) from the last verify pass, in order.
    let mut queue = std::collections::VecDeque::new();
    let GenOpts {
        n_gen,
        stop,
        max_draft,
        ..
    } = o;
    let mut pick = argmax(&logits[..]);
    let mut picks = 0usize;
    while st.ids.len() < n_gen {
        let next = pick;
        picks += 1;
        if stop.contains(&next) {
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
            pick = net.forward_argmax(next, pos, cache);
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
    use super::{GenOpts, argmax, generate, placement};
    use crate::lm::Device;
    use crate::model::Qwen2;
    use crate::spec::PromptLookup;

    fn reference(l: &[f32]) -> u32 {
        (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32
    }

    /// --gpu-layers 0 is the CPU backend, decided before the model loads
    /// (so it's copied, as --device cpu, not mapped); other counts stay.
    #[test]
    fn gpu_layers_zero_is_the_cpu() {
        assert_eq!(placement(Device::Cuda, Some(0)), (Device::Cpu, None));
        assert_eq!(placement(Device::Cuda, Some(7)), (Device::Cuda, Some(7)));
        assert_eq!(placement(Device::Cpu, None), (Device::Cpu, None));
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

    /// A weightless `Lm`: the pick after token t at position p is a hash of
    /// t, p and the token cached at p - 1 (so a misused cache shows), and
    /// its logits peak there. Counts the calls `generate` makes.
    #[derive(Default)]
    struct Toy {
        forwards: std::cell::Cell<usize>,
        argmaxes: std::cell::Cell<usize>,
        verifies: std::cell::Cell<usize>,
    }

    impl Toy {
        fn pick(t: u32, pos: usize, c: &mut [u32]) -> u32 {
            c[pos] = t;
            let prev = if pos > 0 { c[pos - 1] } else { 5 };
            (t * 31 + pos as u32 * 7 + prev * 3) % 97
        }
    }

    impl crate::Lm for Toy {
        type Cache = Vec<u32>;
        fn vocab(&self) -> usize {
            97
        }
        fn new_cache(&self, max_ctx: usize) -> Vec<u32> {
            vec![u32::MAX; max_ctx]
        }
        fn forward(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> Vec<f32> {
            self.forwards.set(self.forwards.get() + 1);
            let mut l = vec![0.0; 97];
            l[Toy::pick(t, pos, c) as usize] = 1.0;
            l
        }
        fn forward_argmax(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> u32 {
            self.argmaxes.set(self.argmaxes.get() + 1);
            Toy::pick(t, pos, c)
        }
        fn feed_batch(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) {
            c[start..start + ts.len()].copy_from_slice(ts);
        }
        fn verify_argmax(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) -> Vec<u32> {
            self.verifies.set(self.verifies.get() + 1);
            (0..ts.len())
                .map(|b| Toy::pick(ts[b], start + b, c))
                .collect()
        }
    }

    /// `generate` through the `Lm` seam on a backend that isn't Qwen2:
    /// plain decode takes `forward_argmax` (never `forward`'s logits),
    /// drafting `verify_argmax`, and both give the reference greedy output.
    #[test]
    fn generate_uses_the_lm_seam() {
        use crate::Lm;
        const N: usize = 20;
        let prompt = [1u32, 2, 3];
        let net = Toy::default();
        let mut c = net.new_cache(prompt.len() + N + 1);
        net.feed_batch(&prompt[..2], 0, &mut c);
        let first = net.forward(3, 2, &mut c);
        let mut want = vec![argmax(&first)];
        while want.len() < N {
            let pos = prompt.len() + want.len() - 1;
            want.push(Toy::pick(*want.last().unwrap(), pos, &mut c));
        }
        for k in [0, 3] {
            let net = Toy::default();
            let mut cache = net.new_cache(prompt.len() + N + 1);
            net.feed_batch(&prompt[..2], 0, &mut cache);
            let logits = net.forward(3, 2, &mut cache);
            let st = generate(
                &net,
                &mut cache,
                &prompt,
                logits,
                GenOpts {
                    n_gen: N,
                    stop: &[],
                    max_draft: k,
                    sample: None,
                },
                &mut |h, max| {
                    let at = h.len() - prompt.len();
                    want[at..].iter().take(max).copied().collect()
                },
                &mut |_| Ok(()),
            )
            .unwrap();
            assert_eq!(st.ids, want, "k {k}");
            assert_eq!(net.forwards.get(), 1, "k {k}: only the prompt's forward");
            if k == 0 {
                assert_eq!((net.argmaxes.get(), net.verifies.get()), (N - 1, 0));
            } else {
                assert_eq!(net.argmaxes.get(), 0);
                assert_eq!(net.verifies.get(), st.passes);
                assert!(st.passes < N - 1);
            }
        }
    }

    /// Cache holding `ids` and the logits after them, as `run_lm` does it.
    fn prefill<M: crate::Lm>(net: &M, ids: &[u32], cap: usize) -> (M::Cache, Vec<f32>) {
        let mut cache = net.new_cache(cap);
        let (&last, rest) = ids.split_last().unwrap();
        let pb = net.prefill_batch();
        for (i, batch) in rest.chunks(pb).enumerate() {
            net.feed_batch(batch, i * pb, &mut cache);
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
        drafting_matches_greedy_on(&Qwen2::new(&m), &m);
    }

    /// `drafting_matches_greedy` on the GPU: its batched verify pass.
    #[test]
    fn drafting_matches_greedy_cuda() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let km = crate::cuda::Kernels::module(&g).unwrap();
        let net = crate::cuda::qwen2::CudaQwen2::new(&g, &km, &m, 256).unwrap();
        drafting_matches_greedy_on(&net, &m);
    }

    /// `drafting_matches_greedy` split across devices: layers 0..12 on the
    /// CPU, 12..24 and the head on the GPU (verify = both batched paths).
    #[test]
    fn drafting_matches_greedy_hybrid() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let km = crate::cuda::Kernels::module(&g).unwrap();
        let net = crate::hybrid::HybridQwen2::new(
            Some((&g, &km)),
            &m,
            256,
            12,
            crate::cuda::qwen2::CacheKv::F16,
        )
        .unwrap();
        drafting_matches_greedy_on(&net, &m);
    }

    fn drafting_matches_greedy_on<M: crate::Lm>(net: &M, m: &crate::gguf::Model) {
        let tok = crate::tokenizer::Tokenizer::from_gguf(m);
        const N: usize = 24;
        for prompt in PROMPTS {
            let ids = tok.encode(prompt);
            let cap = ids.len() + N + 1;
            // Plain greedy reference: every pick, as the original loop does it.
            let (mut cache, mut logits) = prefill(net, &ids, cap);
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
                        let (mut cache, logits) = prefill(net, &ids, cap);
                        let mut emitted = Vec::new();
                        let st = generate(
                            net,
                            &mut cache,
                            &ids,
                            logits,
                            GenOpts {
                                n_gen: n,
                                stop: &[eos],
                                max_draft: k,
                                sample: None,
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
