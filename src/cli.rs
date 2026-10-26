//! The `axon` command line (`src/main.rs` calls `main`): argument parsing
//! and the `gen`, `logits`, `info`, `chat` and `serve` subcommands. Not a
//! stable API: usage errors print and exit the process (`fail`). Embedders
//! use the crate root's API instead.

use crate::generate::{GenOpts, generate};
use crate::{
    chat, cuda, draft, gguf, hybrid, identity, kv, kvsave, lineedit, lm, model, pool, rope, sample,
    serve, spec, tokenizer,
};
use lm::{Backend, Device, Inner, Lm};
use std::env;

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
pub(crate) fn round_draft(k: usize) -> usize {
    match k {
        0 => 0,
        1..=3 => 3,
        _ => k - (k + 1) % 4,
    }
}

const USAGE: &str = "\
usage:
  axon <model.gguf>                         same as axon info
  axon info <model.gguf> [rope flags]       print a summary and exit
  axon gen <model.gguf> \"<prompt>\" [-n N] [--draft K] [--draft-model PATH] [--device D]
            [--ctx N] [--gpu-layers N] [--gpus L] [--kv T] [--verbose] [sampling flags]
                                            generation (greedy unless --temp > 0;
                                            --verbose: the full identity tuple)
  axon logits <model.gguf> \"<prompt>\" [-k K]
                                            top-K next-token logits after the prompt
  axon chat <model.gguf> [--system TEXT] [--no-think] [--ctx N] [--forget-to F] [-n N]
            [--draft K] [--draft-model PATH] [--device D] [--gpu-layers N] [--gpus L] [--kv T]
            [--session F] [sampling flags]
                                            talk to an instruct model (sampled; --temp 0: greedy)
  axon serve <model.gguf> [--host H] [--port P] [--ctx N] [--draft K] [--draft-model PATH]
            [--device D] [--gpu-layers N] [--kv T] [--alias NAME] [--max-tokens N] [--cors [ORIGIN]]
            [--max-conns N] [--max-queue N] [--parallel N] [--prefill-chunk N] [--prefill-ms MS]
            [--api-key KEY] [--max-prompt-tokens N] [--slot-idle-ms MS]
            [--no-think] [--think-inline] [--quiet] [sampling flags; the repeat penalty defaults to 1 (off)]
                                            an OpenAI-compatible HTTP server (/v1/models,
                                            /v1/chat/completions, /v1/completions; on cuda
                                            up to --parallel (8) requests decode together,
                                            on the CPU or a --gpu-layers split one at a
                                            time; README: axon serve)
  axon -h | --help                          this help
  axon -V | --version                       print the version

flags:
  -n N        gen: tokens to generate (default 32); chat: most tokens per reply (default: no limit)
  --draft K   gen, chat, serve: draft up to K tokens per step by prompt lookup (default 0 = off;
              try 7), or with --draft-model by the draft model (default 3; greedy gen and chat:
              7 next to a model split across the CPU and GPU; 0 = off;
              the gain is when greedy: little or none when sampling);
              lossless: the output is the same either way (greedy, or sampled with the same seed)
  --draft-model PATH  gen, chat, serve: a small model with the same vocabulary (e.g.
              Qwen2.5-0.5B-Instruct for Qwen2.5-7B, Qwen3-0.6B for Qwen3-8B, Llama-3.2-1B for
              Llama-3.1-8B) guesses the next tokens greedily, the model checks them in one pass;
              on the same --device (on cuda, all of it on the GPU, its memory counted before the
              model is placed); refused if the vocabularies or end tokens differ. gen, chat:
              \"auto\" picks the smallest GGUF next to the model, at most 1/8 its size, that
              can draft for it (none: drafting off)
  -k K        logits: how many logits to print (default 10)
  --device D  gen, chat: cpu (default) or cuda (an NVIDIA GPU; a model too big for its memory
              is split: the last layers that fit on the GPU, the first ones on the CPU)
  --gpu-layers N  gen, chat: with --device cuda, run the last N layers (and the output head) on
              the GPU and the rest on the CPU (0 = all on the CPU, as --device cpu; default: all
              if they fit with 256 MiB to spare, else as many as fit with another 640 MiB left
              for the desktop; an explicit N only needs the 256 MiB)
  --gpus L    gen, chat: the CUDA devices to run on (implies --device cuda): one (\"1\") or a
              pipeline over several (\"0,1\": the layers split across them in order, the first
              runs the embedding, the last the head; same results as one GPU)
  --gpu-experts N|auto  gen, chat: with --device cuda, a MoE model's (deepseek2) routed experts
              of N layers (auto: as many as fit after the context's cache; smallest first) resident
              on the GPU instead of the CPU; same results (the GPU twins of the CPU's kernels);
              default 0; as AXON_GPU_EXPERTS
  --kv T      gen, chat: KV cache type, f16 (default) or q8 (cuda only: int8 with a scale per
              position and head, half the memory; slightly different results; on a split, the
              GPU layers' cache, so more layers fit); as AXON_KV
  --system T  chat: the system message (default: Qwen2.5's for qwen2 models, none for qwen3/llama)
  --no-think  chat: qwen3: answer without reasoning first (by default it thinks, as ollama does)
  --ctx N     chat: context size in tokens (default: the model's, at most 8192; on cuda, if the
              whole model doesn't fit with it, less if at least half (or 4096) fits, else a
              split); when the conversation outgrows it, the oldest exchanges are forgotten
              (see --forget-to);
              serve: per request (default: the model's trained context if the whole model fits
              on the GPU(s) with it, else the longest that does if at least 4096; on the CPU, as
              much of it as fits in free RAM, at least 8192; a CPU/GPU split, at most 8192; a
              shorter one is said on stderr); an explicit N is never shortened;
              gen: place the model and size its cache for N tokens, as chat would (default: the
              prompt and -n), to see a long context's speed (it may split the model)
  --forget-to F  chat: when the conversation outgrows --ctx, forget the oldest exchanges until
              the rest (with the new message) fills at most F of the context (default 0.5; the
              newest exchange stays unless the new message needs its room). Forgetting shifts
              positions, so what's kept is prefilled again; with room left, the next turns append
              to the cache instead (the room is what's free less a reply's: with -n unlimited a
              reply reserves a quarter of the context, so 0.5 leaves a quarter). min, 0 or 1:
              forget only as many as make room for the reply (once the context is full, nearly
              every turn prefills it all again)
  --session F chat: keep the conversation in file F: loaded at start if it exists, saved after
              every reply (new positions appended). Its KV cache comes back from the file if
              it was made with this model, KV type, backend and GPU; else it is prefilled again
              from the saved tokens (slower; on the CPU or GPU the same cache, so replies continue
              as if never stopped; on a CPU/GPU split a re-prefilled cache is close, not exact)
              One process per session file: a session in use by another axon process is refused
              (an advisory lock on F.lock beside it, left in place; where the filesystem can't lock,
              a warning and no lock)
  --parallel N  serve: requests decoded together on the GPU (default 8; 1: one at a time;
              N slots cost N KV caches). By default the context comes first: as many
              slots (2 to 8) as fit with the whole context, or with a shorter one down to
              32768 (or the whole one, if less); if not even 2 fit, one request at a time.
              With --ctx N, the slots that fit with N; with --parallel N alone, N slots of
              the longest context that fits them
  --prefill-ms MS  serve: while requests decode, a step feeds new prompts up to what fits in
              MS ms beside their decode step (default 60; 0: exactly --prefill-chunk tokens)
  --prefill-chunk N  serve: at least N prompt tokens per such step (default 256; 0: prompts
              fed whole, the others' streams wait)
  --api-key KEY  serve: every route but /health (and CORS preflights) needs
              `Authorization: Bearer KEY` (else 401); AXON_API_KEY sets it too, out of `ps`
              (trimmed; set but blank is an error)
  --max-prompt-tokens N  serve: a longer prompt is refused (400, context_length_exceeded)
              before prefill (the rendered prompt: template and tool tokens count)
  --slot-idle-ms MS  serve: a stream whose client stops reading for MS ms is cancelled
              and its slot freed, as if it had disconnected (default: off; at least 1000).
              The clock starts once the client's receive buffer is full (README)
  --rope-scaling yarn --rope-scale F [--yarn-orig-ctx N]
              gen, logits, chat, serve: YaRN RoPE scaling (llama.cpp's flags), off by default:
              the model's context becomes N x F (N: default the model's trained context), e.g.
              Qwen2.5 / Qwen3 to 131072 with --rope-scale 4 --yarn-orig-ctx 32768. Changes the
              output at every length, short prompts too: only for prompts past the trained
              context (docs/long-context.md). --yarn-beta-fast B (32), --yarn-beta-slow B (1),
              --yarn-mscale M (the q and k scale, default 0.1 ln F + 1; not llama.cpp's
              --yarn-attn-factor); without --rope-scale, chat/serve take F = ceil(--ctx / N);
              --rope-scaling none drops a GGUF's own rope.scaling keys

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
  AXON_DRAFT_MODEL=PATH      same as --draft-model PATH
  AXON_MMAP=0|1              cpu: 0: copy the model into huge-page memory, 1: map the
                             file (default: map if every matrix is repacked, else copy
                             unless the file is >60% of free memory; a mapped file must
                             not be overwritten in place while axon runs)
  AXON_READAHEAD=0           don't read a mapped model ahead on a background thread
  AXON_PREFETCH=N            decode prefetch distance in bytes (default 8192, 0 = off)
  AXON_DECODE_ROWS=1|2|4     decode rows per kernel call (default 4)
  AXON_KV=f16|f32|q8         KV cache element type (default f16: half the memory and
                             attention bandwidth; f32 for A/B, CPU only; q8 cuda only)
  AXON_DEVICE=cpu|cuda       same as --device
  AXON_GPU_LAYERS=N          same as --gpu-layers N
  AXON_GPU_EXPERTS=N|auto    same as --gpu-experts N|auto (serve too)
  AXON_GPUS=L                same as --gpus L (serve too; with --device cuda)
  AXON_PP_CUTS=K[,K2..]      --gpus pipeline: the first layer of each GPU after the first
                             (default: even, else what fits)
  AXON_P2P=1                 --gpus pipeline: try peer access (P2P) between the GPUs.
                             UNTESTED: no P2P-capable box has run it yet. Default: the
                             hidden state is copied through host memory by the driver
                             (tested; ~10 us per token hop on 2x4090 PCIe)
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

The model must be a qwen2, qwen3 or llama GGUF (e.g. Qwen2.5-0.5B-Instruct, Qwen3-4B or Llama-3.2-3B-Instruct, Q4_K_M),
or, on the CPU, a Hugging Face checkpoint directory of one (config.json, tokenizer.json and
model.safetensors or its shards; BF16/F16/F32 weights, run as the GGUF F16 of the same weights).";

/// Whether `net` is split across the CPU and the GPU (some layers on each):
/// a drafter's default K is then `draft::SPLIT_MODEL_DRAFT`.
fn is_split(net: &Backend) -> bool {
    matches!(&net.0, Inner::Hybrid(h) if h.gpu_layers > 0 && h.gpu_layers < h.n_layers)
}

/// AXON_STREAM_DECODE=1 with drafting: refused (drafting's picks come
/// from verify passes, which never stream, so they could differ from the
/// streamed per-token steps: not lossless).
pub(crate) fn check_stream_decode(max_draft: usize) {
    if max_draft > 0 && hybrid::stream_decode_setting() {
        fail("AXON_STREAM_DECODE=1 can't be combined with --draft / AXON_DRAFT (not lossless)");
    }
}

/// Usage error: one line, a hint, exit code 2.
pub(crate) fn fail(msg: &str) -> ! {
    eprintln!("axon: {msg}");
    eprintln!("try 'axon --help'");
    cuda::settle();
    std::process::exit(2);
}

/// Load a GGUF, exiting with code 2 and a one-line message if it can't be read.
/// Mapped, not copied, nor read ahead: `info` reads the header only.
fn open_model(path: &str, lm: bool) -> gguf::Model {
    let m = gguf::load_header(path).unwrap_or_else(|e| load_failed(path, e));
    if lm && let Err(e) = model::check_supported(&m) {
        fail(&format!("{path}: {e}"));
    }
    m
}

/// A model to run (`gguf::load_to_run`), checked before it's copied or
/// read: `copy` forces the copy-vs-map choice, None copies only if the CPU
/// model reads its matrices in place (`model::reads_matrices_in_place`); a
/// MoE model is always mapped, whatever `copy` and AXON_MMAP say.
/// `rope`: the --rope-scaling flags, written into the model's metadata
/// (`rope::RopeArgs::apply`) before the model is checked, so that
/// `--rope-scaling none` can run a file whose own scaling is refused.
pub(crate) fn open_model_with(
    path: &str,
    copy: Option<bool>,
    rope: &rope::RopeArgs,
) -> gguf::Model {
    // The CPU kernels' environment, before anything is read (as `load`).
    if let Err(e) = crate::quant::env_check() {
        fail(&e);
    }
    let mut m = gguf::load_to_run(path, copy, |m| {
        // With rope flags the check waits for them (below).
        if rope.is_empty()
            && let Err(e) = model::check_supported(m)
        {
            fail(&format!("{path}: {e}"));
        }
        // A MoE file is always mapped (`reads_matrices_in_place`'s note).
        model::cpu_copy_wish(m)
    })
    .unwrap_or_else(|e| load_failed(path, e));
    if !rope.is_empty() {
        for line in rope.apply(&mut m).unwrap_or_else(|e| fail(&e)) {
            eprintln!("axon: {line}");
        }
        if let Err(e) = model::check_supported(&m) {
            fail(&format!("{path}: {e}"));
        }
    }
    m
}

fn load_failed(path: &str, e: std::io::Error) -> ! {
    match e.kind() {
        std::io::ErrorKind::NotFound => fail(&format!("cannot open '{path}': no such file")),
        _ => fail(&format!("cannot load '{path}': {e}")),
    }
}

/// The `axon` binary's entry point.
pub fn main() {
    // A panic must not exit (libcuda's exit handlers) while a `cuda::preinit`
    // thread is inside the driver.
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default(info);
        cuda::settle();
    }));
    if let Err(e) = run() {
        eprintln!("axon: {e}");
        cuda::settle();
        std::process::exit(1);
    }
}

fn run() -> std::io::Result<()> {
    let args = utf8_args(env::args_os()).unwrap_or_else(|e| fail(&e));

    match args.get(1).map(String::as_str) {
        None => {
            eprintln!("axon {}\n\n{USAGE}", env!("CARGO_PKG_VERSION"));
            std::process::exit(2);
        }
        Some("-h" | "--help" | "help") => {
            println!(
                "axon {}: CPU and CUDA inference for Qwen2/Qwen3/Llama 3 GGUF models\n\n{USAGE}",
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
    if args[1] == "serve" {
        return serve::run(&args);
    }

    // `axon info <model>`, or just `axon <model>`.
    let path = match args[1].as_str() {
        "info" if args.len() >= 3 => &args[2],
        "info" => fail("usage: axon info <model.gguf | checkpoint dir> [rope flags]"),
        f if f.starts_with('-') => fail(&format!("unknown flag '{f}'")),
        // A lone argument is a model to summarize, unless it can't be one.
        p if args.len() == 2 && (p.ends_with(".gguf") || std::path::Path::new(p).exists()) => {
            &args[1]
        }
        c => fail(&format!(
            "unknown command '{c}' (commands: gen, logits, chat, serve, info)"
        )),
    };

    // --rope-scaling and friends, as gen takes them: the fingerprints
    // below are for the model as those flags run it.
    let mut rargs = rope::RopeArgs::default();
    let mut i = 3;
    while i < args.len() {
        let f = args[i].as_str();
        if !rope::RopeArgs::is_flag(f) {
            fail(&format!(
                "unknown argument '{f}' for info (it takes the rope flags)"
            ));
        }
        rargs
            .set(f, args.get(i + 1).map(String::as_str))
            .unwrap_or_else(|e| fail(&e));
        i += 2;
    }
    let mut model = open_model(path, false);
    for line in rargs.apply(&mut model).unwrap_or_else(|e| fail(&e)) {
        eprintln!("axon: {line}");
    }
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
    if let gguf::Source::Safetensors { dir, files } = &model.source {
        println!(
            "safetensors checkpoint {} ({} files); tensors (HF name -> GGUF name, type, dims):",
            dir.display(),
            files.len()
        );
        for t in &model.tensors {
            let (hf, from) = match &t.hf {
                Some(h) if h.dtype != t.qtype.name => {
                    (h.name.as_str(), format!(" (from {})", h.dtype))
                }
                Some(h) => (h.name.as_str(), String::new()),
                None => ("(computed)", String::new()),
            };
            println!("  {hf} -> {} {}{from} {:?}", t.name, t.qtype.name, t.dims);
        }
    }
    if let Some(i) = model::MoeInfo::of(&model) {
        for l in i.lines() {
            println!("{l}");
        }
    }
    for l in model::deepseek2_info(&model).unwrap_or_default() {
        println!("{l}");
    }
    println!("{}", cuda::summary());
    // What decides the output bits (docs/determinism.md), per placement.
    match identity::predicted(path, &model) {
        Ok(ids) => {
            if let Some((_, Ok(i))) = ids.first() {
                let model = &i.fields[1].1;
                println!(
                    "model digest: {model} (sampled SHA-256, identity numerics {})",
                    identity::NUMERICS_VERSION
                );
            }
            for (what, i) in ids {
                match i {
                    Ok(i) => println!("fingerprint {what}: {}", i.line()),
                    Err(why) => println!("fingerprint {what}: none ({why})"),
                }
            }
        }
        Err(e) => println!("fingerprint: none ({e})"),
    }
    Ok(())
}

/// Speculative decoding: draft up to K tokens per step by prompt lookup
/// (spec.rs); output is the same either way. Off by default (DEFAULT_DRAFT);
/// --draft K or AXON_DRAFT=K turns it on, see round_draft. This is AXON_DRAFT.
pub(crate) fn draft_env() -> usize {
    draft_env_set().unwrap_or(DEFAULT_DRAFT)
}

/// AXON_DRAFT's K, if it's set.
pub(crate) fn draft_env_set() -> Option<usize> {
    let v = env::var("AXON_DRAFT").ok()?;
    Some(
        v.parse::<usize>()
            .unwrap_or_else(|_| fail(&format!("AXON_DRAFT needs a whole number, got '{v}'"))),
    )
}

/// The draft model for --draft-model PATH (else $AXON_DRAFT_MODEL), checked
/// against the main model `m`, or None if there's none or K is 0: a usage
/// error if it can't draft for `m`. After `open_model_with` of the main
/// model, before loading it (`draft::load_pair`). "auto": the smallest
/// GGUF next to the model (at `model_path`) that can draft for it
/// (`draft::auto_path`), or none (said on stderr).
fn open_draft_model(
    path: Option<String>,
    model_path: &str,
    k: usize,
    device: Device,
    m: &gguf::Model,
    tok: &tokenizer::Tokenizer,
) -> Option<draft::DraftModel> {
    let mut path = path.or_else(draft::model_env).filter(|_| k > 0)?;
    if path == "auto" {
        match draft::auto_path(model_path, m, tok) {
            Some(p) => {
                eprintln!("axon: --draft-model auto: {p}");
                path = p;
            }
            None => {
                eprintln!(
                    "axon: --draft-model auto: no smaller GGUF next to {model_path} can draft \
                     for it; drafting off"
                );
                return None;
            }
        }
    }
    Some(
        draft::DraftModel::open(&path, copy_for(device), m, tok)
            .unwrap_or_else(|e| fail(&format!("--draft-model {e}"))),
    )
}

/// `backend` with the draft model `dm`'s VRAM held while the main model is
/// placed (`draft::load_pair`: for a `draft_ctx`-token cache), then the
/// drafter for its context (at most draft_ctx).
#[allow(clippy::too_many_arguments)]
pub(crate) fn backend_and_drafter<'a>(
    model: &'a gguf::Model,
    dm: Option<&draft::DraftModel>,
    device: Device,
    ctx: usize,
    gpu_layers: Option<usize>,
    kv: kv::KvType,
    fit: bool,
    place: lm::Placing,
    draft_ctx: usize,
) -> (Backend<'a>, Option<draft::ModelDrafter<Backend<'static>>>) {
    draft::load_pair(dm, device, ctx, draft_ctx, kv, || {
        backend(model, device, ctx, gpu_layers, kv, fit, place)
    })
    .unwrap_or_else(|e| match device {
        Device::Cuda => fail(&format!("--device cuda: --draft-model: {e}")),
        Device::Cpu => fail(&format!("--draft-model: {e}")),
    })
}

/// --gpu-experts' value: a count or "auto" (a usage error otherwise).
fn gpu_experts_flag(v: Option<&String>) -> crate::cuda::qwen2::experts::GpuExperts {
    let v = v.unwrap_or_else(|| fail("--gpu-experts needs a value (a count or auto)"));
    crate::cuda::qwen2::experts::GpuExperts::parse(v).unwrap_or_else(|e| fail(&e))
}

/// KV cache type (kv.rs): checked up front so a typo is a usage error, not a
/// panic in the first new_cache. --kv overrides it.
pub(crate) fn kv_env() -> kv::KvType {
    kv::kv_setting().unwrap_or_else(|e| fail(&e))
}

/// --kv's value (None: the flag was last, without one): f16 or q8. f32
/// (CPU only) is AXON_KV's alone; --kv f16 over AXON_KV=f32 means f16.
pub(crate) fn kv_flag(v: Option<&str>) -> kv::KvType {
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

/// --gpus LIST: the CUDA devices (`lm::set_gpus`); CUDA is the device.
fn gpus_flag(v: Option<&str>) -> Device {
    let v = v.unwrap_or_else(|| fail("--gpus needs GPU ordinals (0 or 0,1)"));
    lm::set_gpus(lm::parse_gpus(v).unwrap_or_else(|e| fail(&format!("--gpus: {e}"))));
    Device::Cuda
}

/// --device's default: $AXON_DEVICE, else the CPU.
pub(crate) fn device_env() -> Device {
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
/// 0 GPU layers is the plain CPU backend (loaded as --device cpu loads it,
/// see `copy_for`). Before `open_model_with`.
pub(crate) fn placement(device: Device, gpu_layers: Option<usize>) -> (Device, Option<usize>) {
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
/// says (`placement`'s) and `place` (`Backend::new_capped`); a usage error
/// if the GPU can't be used (no driver or device, too little memory, ...).
fn backend<'a>(
    model: &'a gguf::Model,
    device: Device,
    ctx: usize,
    gpu_layers: Option<usize>,
    kv: kv::KvType,
    fit: bool,
    place: lm::Placing,
) -> Backend<'a> {
    let net = Backend::new_capped(model, device, ctx, gpu_layers, kv, fit, place);
    net.unwrap_or_else(|e| match device {
        Device::Cuda => fail(&format!("--device cuda: {e}")),
        Device::Cpu => fail(&e),
    })
}

/// The copy-vs-map choice for loading a model for `device`: the GPU reads
/// the weights once, so it maps the file; the CPU decides once it's parsed.
pub(crate) fn copy_for(device: Device) -> Option<bool> {
    (device == Device::Cuda).then_some(false)
}

/// The command line as UTF-8 strings: an error naming the first argument
/// that isn't (`env::args` panics on it).
fn utf8_args(args: impl Iterator<Item = std::ffi::OsString>) -> Result<Vec<String>, String> {
    args.enumerate()
        .map(|(i, a)| {
            a.into_string()
                .map_err(|a| format!("argument {i} ({}) is not valid UTF-8", a.to_string_lossy()))
        })
        .collect()
}

/// How many tokens `gen` may generate after a `prompt`-token prompt when
/// asked for `n`: the cache holds the prompt, those tokens and one more,
/// and stays within the `trained` context (0: the GGUF doesn't say), as
/// `chat` and `serve` do (RoPE past it gives garbage; a -n near usize::MAX
/// overflowed the cache size). An error if the prompt alone is too long.
fn gen_budget(prompt: usize, n: usize, trained: usize) -> Result<usize, String> {
    if trained > 0 {
        if prompt >= trained {
            return Err(format!(
                "the prompt is {prompt} tokens; the model was trained on {trained}"
            ));
        }
        return Ok(n.min(trained - prompt - 1));
    }
    match prompt.checked_add(n).and_then(|t| t.checked_add(1)) {
        Some(_) => Ok(n),
        None => Err(format!("-n {n} is too large")),
    }
}

/// The top `k` of `logits` (id, logit, logprob, text), as `logits` prints them.
fn print_top(tok: &tokenizer::Tokenizer, logits: &[f32], k: usize, vocab: usize) {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = mx + logits.iter().map(|l| (l - mx).exp()).sum::<f32>().ln();
    println!("top-{k} of {vocab} (id, logit, logprob, text):");
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let digits = env::var("AXON_LOGITS_DIGITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let (dl, dp) = digits.map_or((4, 3), |d| (d, d));
    for &id in order.iter().take(k) {
        println!(
            "{:>7}  {:>10.dl$}  {:>8.dp$}  {:?}",
            id,
            logits[id],
            logits[id] - lse,
            tok.decode(&[id as u32])
        );
    }
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
    let mut max_draft = None;
    let mut draft_path = None;
    let mut device = device_env();
    let mut gpu_layers = None;
    let mut gpu_experts = None;
    let mut gen_ctx = None;
    let mut kv = kv_env();
    let mut sargs = sample::SampleArgs::default();
    let mut rargs = rope::RopeArgs::default();
    let mut verbose = false;
    let mut i = 4;
    while i < args.len() {
        if args[i] == "--verbose" {
            verbose = true;
            i += 1;
            continue;
        }
        let val = args.get(i + 1).and_then(|s| s.parse::<usize>().ok());
        match (args[i].as_str(), val) {
            (f, _) if sample::SampleArgs::is_flag(f) => sargs
                .set(f, args.get(i + 1).map(String::as_str))
                .unwrap_or_else(|e| fail(&e)),
            (f, _) if rope::RopeArgs::is_flag(f) => rargs
                .set(f, args.get(i + 1).map(String::as_str))
                .unwrap_or_else(|e| fail(&e)),
            ("--device", _) => {
                let v = args.get(i + 1).map_or("", String::as_str);
                device = Device::parse(v).unwrap_or_else(|e| fail(&e));
            }
            ("--kv", _) => kv = kv_flag(args.get(i + 1).map(String::as_str)),
            ("--draft-model", _) => {
                let v = args.get(i + 1).filter(|v| !v.is_empty());
                draft_path = Some(
                    v.cloned()
                        .unwrap_or_else(|| fail("--draft-model needs a path")),
                );
            }
            ("--gpus", _) => device = gpus_flag(args.get(i + 1).map(String::as_str)),
            ("-n", Some(v)) => n_gen = v,
            ("-k", Some(v)) => top_k = v,
            ("--draft", Some(v)) => max_draft = Some(v),
            ("--gpu-layers", Some(v)) => gpu_layers = Some(v),
            ("--gpu-experts", _) => gpu_experts = Some(gpu_experts_flag(args.get(i + 1))),
            ("--ctx", Some(v)) if cmd == "gen" => gen_ctx = Some(v),
            (f @ ("-n" | "-k" | "--draft" | "--gpu-layers" | "--ctx"), None) => fail(&format!(
                "{f} needs a whole number, got {}",
                args.get(i + 1)
                    .map_or("nothing".into(), |v| format!("'{v}'"))
            )),
            (f, _) => fail(&format!("unknown argument '{f}' for {cmd}")),
        }
        i += 2;
    }
    if cmd == "logits" && draft_path.is_some() {
        fail("--draft-model is for gen, chat and serve, not logits");
    }
    let with_model = cmd == "gen" && (draft_path.is_some() || draft::model_env().is_some());
    let k_given = max_draft.or_else(draft_env_set);
    let max_draft = draft::draft_k(k_given, with_model);
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
    if device == Device::Cuda {
        cuda::preinit();
    }
    // gen --ctx N, as chat's: YaRN's default factor comes from it too.
    rargs.ctx = gen_ctx;
    let model = open_model_with(path, copy_for(device), &rargs);
    let tok = tokenizer::Tokenizer::from_gguf(&model);

    let mut ids = tok.encode_prompt(prompt);
    // AXON_LOGITS_IDS=1,2,3: `logits` of these token ids instead of the
    // prompt's (oracles compare on ids, never on text); with
    // AXON_LOGITS_ALL, several sequences separated by ';' (one load).
    let mut seqs: Vec<Vec<u32>> = Vec::new();
    if let Ok(v) = env::var("AXON_LOGITS_IDS").map(|v| v.trim().to_string())
        && cmd == "logits"
        && !v.is_empty()
    {
        for part in v.split(';').filter(|p| !p.trim().is_empty()) {
            let seq: Vec<u32> = part
                .trim()
                .trim_matches(['[', ']'])
                .split(',')
                .map(|x| x.trim().parse::<u32>())
                .collect::<Result<_, _>>()
                .unwrap_or_else(|_| fail("AXON_LOGITS_IDS needs comma-separated token ids"));
            if let Some(&bad) = seq.iter().find(|&&t| t as usize >= tok.vocab_len()) {
                fail(&format!(
                    "AXON_LOGITS_IDS: token {bad} is past the vocabulary"
                ));
            }
            seqs.push(seq);
        }
        ids = seqs
            .iter()
            .max_by_key(|s| s.len())
            .cloned()
            .unwrap_or_default();
    }
    if ids.is_empty() {
        eprintln!("prompt encodes to zero tokens");
        cuda::settle();
        std::process::exit(1);
    }
    // The cache holds the prompt, the tokens generated and one more. RoPE
    // past the trained length gives garbage (0: the GGUF doesn't say), and
    // a -n near usize::MAX overflowed the sizes: generation stops at the
    // trained context, as `chat` and `serve` do.
    // `logits` generates nothing: no budget, as before (its prompt alone
    // may run past the trained context).
    let want = if cmd == "gen" { n_gen } else { 0 };
    let extra = if cmd == "gen" {
        gen_budget(ids.len(), want, model.ctx_len() as usize)
            .unwrap_or_else(|e| fail(&format!("{path}: {e}")))
    } else {
        0
    };
    if extra < want {
        eprintln!(
            "axon: note: -n {want} runs past the {}-token context {path} was trained on; \
             stopping after {extra} tokens",
            model.ctx_len()
        );
    }
    let n_gen = if cmd == "gen" { extra } else { n_gen };
    // The cache (and so the placement, and the drafter's cache) is for the
    // prompt and the tokens generated, or --ctx's context if bigger: the
    // model placed as chat --ctx N places it (a long context can split it).
    let trained = model.ctx_len() as usize;
    if let Some(n) = gen_ctx.filter(|&n| trained > 0 && n > trained) {
        let yarn = rope::yarn(&model).is_some();
        fail(&format!(
            "--ctx {n} is past the {trained} tokens {path} {}{}",
            if yarn {
                "runs with YaRN"
            } else {
                "was trained on"
            },
            if yarn {
                String::new()
            } else {
                rope::ctx_hint(n, trained)
            }
        ));
    }
    let need = ids.len() + extra + 1;
    if let Some(n) = gen_ctx.filter(|&n| n < need) {
        eprintln!(
            "axon: note: --ctx {n} is less than the prompt and -n need ({need} tokens); \
             using {need}"
        );
    }
    let cache_ctx = need.max(gen_ctx.unwrap_or(0));
    let dm = if with_model {
        open_draft_model(draft_path, path, max_draft, device, &model, &tok)
    } else {
        None
    };
    let (net, mut drafter) = backend_and_drafter(
        &model,
        dm.as_ref(),
        device,
        cache_ctx,
        gpu_layers,
        kv,
        false,
        lm::Placing {
            gpu_experts,
            ..lm::Placing::of(cache_ctx)
        },
        cache_ctx,
    );
    // On a split, sampled drafting pays too (docs/design/spec-split.md).
    if sampler.is_some() && drafter.is_some() && !is_split(&net) {
        eprintln!("{}", draft::SAMPLED_NOTE);
    }
    let max_draft = draft::split_k(
        k_given,
        max_draft,
        with_model,
        drafter.is_some(),
        is_split(&net),
        sampler.is_none(),
    );
    let mut cache = net.new_cache(cache_ctx);

    // Feed the prompt in batches (the backend's size: PREFILL_BATCH on the
    // CPU), so each weight row is read once per batch instead of once per
    // token. Only the last position's logits are used, so the others skip
    // the output head (the GPU runs the last token in the final batched
    // pass, the head on its row alone: `Lm::prefill_logits`).
    // AXON_LOGITS_ALL=N: `logits` after each of the last N positions
    // (prefill of the rest, then one decode step each).
    let all = env::var("AXON_LOGITS_ALL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|_| cmd == "logits")
        .map_or(0, |n| n.min(ids.len()));
    let t0 = Instant::now();
    let logits = if all > 0 {
        if seqs.is_empty() {
            seqs.push(ids.clone());
        }
        let mut last = Vec::new();
        for (k, ids) in seqs.iter().enumerate() {
            if k > 0 {
                cache = net.new_cache(cache_ctx);
            }
            let first = ids.len() - all.min(ids.len());
            if first > 0 {
                net.feed_batch(&ids[..first], 0, &mut cache);
            }
            println!("prompt ids ({}): {:?}", ids.len(), ids);
            for (p, &t) in ids.iter().enumerate().skip(first) {
                last = net.forward(t, p, &mut cache);
                println!("position {p}:");
                print_top(&tok, &last, top_k, net.vocab());
            }
        }
        last
    } else {
        net.prefill_logits(&ids, 0, &mut cache)
    };
    let prompt_secs = t0.elapsed().as_secs_f64();

    if cmd == "logits" {
        if all > 0 {
            return Ok(());
        }
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
        // AXON_LOGITS_DIGITS: more decimals for scripts (oracle/llama_logits.py).
        let digits = env::var("AXON_LOGITS_DIGITS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let (dl, dp) = digits.map_or((4, 3), |d| (d, d));
        for &id in order.iter().take(top_k) {
            println!(
                "{:>7}  {:>10.dl$}  {:>8.dp$}  {:?}",
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
    let mut source = match drafter.as_mut() {
        Some(d) => draft::Drafter::Model(d),
        None => draft::Drafter::Lookup(spec::PromptLookup::new()),
    };
    let t1 = Instant::now();
    let st = generate(
        &net,
        &mut cache,
        &ids,
        logits,
        GenOpts {
            n_gen,
            stop: &tok.eog(),
            max_draft,
            sample: sampler.as_mut(),
        },
        &mut |h, max| source.draft(h, max),
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
    if let Inner::Cuda(n) = &net.0 {
        eprintln!(
            "device:      cuda, {} of weights in VRAM ({} read per token), KV cache {} {}, \
             CUDA graphs {}",
            human_bytes(n.weight_bytes() as u64),
            human_bytes(n.bytes_per_token(true) as u64),
            n.kv_name(),
            human_bytes(n.kv_cache_bytes(n.kv, cache_ctx) as u64),
            if n.graphs { "on" } else { "off" }
        );
        if n.q6k8_vram() > 0 {
            eprintln!(
                "prefill:     Q6_K int8 copy {} (spare VRAM; AXON_Q6K8=0 off)",
                human_bytes(n.q6k8_vram() as u64)
            );
        }
    }
    if let Inner::Hybrid(n) = &net.0 {
        eprintln!(
            "device:      cuda + cpu, {}, kv {}",
            n.describe(cache_ctx),
            n.kv_name()
        );
    }
    if let Inner::Pipeline(n) = &net.0 {
        eprintln!(
            "device:      cuda, {}, kv {}",
            n.describe(cache_ctx),
            n.kv_name()
        );
    }
    // Inner::Moe: its one banner line is `moe:` at load (lm.rs).
    let id = identity::of(path, &model, &net);
    match verbose {
        true => eprint!("identity:    {}\n{}", id.fingerprint(), id.canonical()),
        false => eprintln!("identity:    {}", id.fingerprint()),
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
            "draft:       K={}{}, {} drafted, {} accepted ({:.1}%), {:.2} tok/pass, \
             (decode tok/s = decided tok incl. EOS / s); forward {}; verify {}",
            max_draft,
            if drafter.is_some() {
                " (draft model)"
            } else {
                " (prompt lookup)"
            },
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
        if let (Some(d), Some(dm)) = (&drafter, &dm) {
            eprintln!("drafter:     {}", draft::describe(&dm.name(), d));
        }
    }
    if !matches!(&net.0, Inner::Cpu(_)) {
        // Everything's written: skip tearing the CUDA context down (freeing
        // each buffer, the driver's exit handlers: ~0.15-0.2 s); the kernel
        // drops the process's GPU memory with it.
        out.flush()?;
        exit_now();
    }
    Ok(())
}

/// End the process at once, code 0, without running exit handlers or
/// destructors (`_exit`). Only once all output has been flushed: stdout's
/// buffer would be lost (stderr has none).
fn exit_now() -> ! {
    unsafe extern "C" {
        fn _exit(code: i32) -> !;
    }
    // SAFETY: _exit takes any status and doesn't return; nothing this
    // process still has to do is skipped (the caller flushed its output).
    unsafe { _exit(0) }
}

/// Context size `axon chat` uses unless --ctx says otherwise (if the model was
/// trained on at least this much). The cache is allocated lazily, page by
/// page as the conversation grows, so a big one costs nothing up front.
pub(crate) const DEFAULT_CHAT_CTX: usize = 8192;

const CHAT_HELP: &str = concat!(
    "  /reset          forget the conversation (the system message stays)\n",
    "  /system TEXT    a new system message (forgets the conversation)\n",
    "  /save [FILE]    save the conversation and its cache (to FILE, then saves go there)\n",
    "  /load [FILE]    load a saved conversation (the session file if no FILE)\n",
    "  /exit           quit (or Ctrl-D)\n",
    "  end a line with \\ (or press Alt-Enter) to go on with the message on a new line;\n",
    "  pasted text stays one message, line breaks and all; //text sends /text\n",
    "keys: Left/Right, Ctrl-Left/Right or Alt-B/F (word), Home/End or Ctrl-A/E,\n",
    "  Backspace/Delete, Ctrl-W (delete word back), Ctrl-U/Ctrl-K (delete to start/end),\n",
    "  Ctrl-Y (paste deleted text), Up/Down (history), Ctrl-L (clear screen),\n",
    "  Ctrl-C (stop a reply; clear the line), Ctrl-D on an empty line (quit)",
);

/// `chat --forget-to`: a fraction of the context in [0, 1], or "min" (0).
fn forget_to_flag(v: &str) -> Result<f64, String> {
    if v == "min" {
        return Ok(0.0);
    }
    v.parse::<f64>()
        .ok()
        .filter(|f| (0.0..=1.0).contains(f))
        .ok_or_else(|| format!("--forget-to needs a fraction from 0 to 1 or min, got '{v}'"))
}

/// `chat` subcommand: args = [axon, chat, model, flags...]. Each line of
/// stdin is a message; on a terminal it's an interactive session.
fn run_chat(args: &[String]) -> std::io::Result<()> {
    const CHAT_USAGE: &str = "usage: axon chat <model.gguf> [--system TEXT] [--no-think] [--ctx N] \
         [--forget-to F] [-n N] [--draft K] [--draft-model PATH] [--device D] [--gpu-layers N] \
         [--gpus L] [--kv T] [--session FILE]";
    let Some(path) = args.get(2).filter(|p| !p.starts_with('-')) else {
        fail(CHAT_USAGE);
    };
    let mut system = None;
    let mut no_think = false;
    let mut ctx = None;
    let mut max_reply = usize::MAX;
    let mut forget_to = chat::FORGET_TO;
    let mut max_draft = None;
    let mut draft_path = None;
    let mut device = device_env();
    let mut gpu_layers = None;
    let mut gpu_experts = None;
    let mut kv = kv_env();
    let mut sargs = sample::SampleArgs::default();
    let mut session = None;
    let mut rargs = rope::RopeArgs::default();
    let mut i = 3;
    while i < args.len() {
        let f = args[i].as_str();
        if f == "--no-think" {
            no_think = true;
            i += 1;
            continue;
        }
        if sample::SampleArgs::is_flag(f) {
            sargs
                .set(f, args.get(i + 1).map(String::as_str))
                .unwrap_or_else(|e| fail(&e));
            i += 2;
            continue;
        }
        if rope::RopeArgs::is_flag(f) {
            rargs
                .set(f, args.get(i + 1).map(String::as_str))
                .unwrap_or_else(|e| fail(&e));
            i += 2;
            continue;
        }
        if !matches!(
            f,
            "--system"
                | "--ctx"
                | "--forget-to"
                | "-n"
                | "--draft"
                | "--draft-model"
                | "--device"
                | "--gpu-layers"
                | "--gpu-experts"
                | "--kv"
                | "--gpus"
                | "--session"
        ) {
            fail(&format!("unknown argument '{f}' for chat"));
        }
        let Some(v) = args.get(i + 1) else {
            fail(&format!("{f} needs a value"));
        };
        if f == "--system" {
            system = Some(v.clone());
        } else if f == "--forget-to" {
            forget_to = forget_to_flag(v).unwrap_or_else(|e| fail(&e));
        } else if f == "--session" {
            if v.is_empty() {
                fail("--session needs a path");
            }
            session = Some(std::path::PathBuf::from(v));
        } else if f == "--draft-model" {
            draft_path = Some(v.clone()).filter(|v| !v.is_empty());
            if draft_path.is_none() {
                fail("--draft-model needs a path");
            }
        } else if f == "--device" {
            device = Device::parse(v).unwrap_or_else(|e| fail(&e));
        } else if f == "--kv" {
            kv = kv_flag(Some(v));
        } else if f == "--gpu-experts" {
            gpu_experts = Some(gpu_experts_flag(Some(v)));
        } else if f == "--gpus" {
            device = gpus_flag(Some(v));
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
                _ => max_draft = Some(n),
            }
        }
        i += 2;
    }

    // --session's lock first: a session in use fails before the model loads.
    let session = session.map(|p| {
        let mut s = kvsave::Session::new(kvsave::canonical(&p), serde_json::Value::Null);
        match s.lock() {
            Err(e) => fail(&e.to_string()),
            Ok(Some(w)) => eprintln!("axon: {w}"),
            Ok(None) => {}
        }
        s
    });
    let (device, gpu_layers) = placement(device, gpu_layers);
    if device == Device::Cuda {
        cuda::preinit();
    }
    rargs.ctx = ctx;
    let model = open_model_with(path, copy_for(device), &rargs);
    // RoPE past the trained length gives garbage; 0 = the GGUF doesn't say.
    let trained = model.ctx_len() as usize;
    let yarn = rope::yarn(&model).is_some();
    let fit = ctx.is_none();
    let ctx = match ctx {
        Some(n) if trained > 0 && n > trained => fail(&format!(
            "--ctx {n} is past the {trained} tokens {path} {}{}",
            if yarn {
                "runs with YaRN"
            } else {
                "was trained on"
            },
            if yarn {
                String::new()
            } else {
                rope::ctx_hint(n, trained)
            }
        )),
        Some(n) => n,
        None if trained > 0 => trained.min(DEFAULT_CHAT_CTX),
        None => DEFAULT_CHAT_CTX,
    };
    if yarn && fit {
        eprintln!(
            "axon: note: YaRN is on but chat's context is {ctx} (the default); --ctx N for a longer one"
        );
    }
    let tok = tokenizer::Tokenizer::from_gguf(&model);
    let with_model = draft_path.is_some() || draft::model_env().is_some();
    let k_given = max_draft.or_else(draft_env_set);
    let max_draft = draft::draft_k(k_given, with_model);
    check_stream_decode(max_draft);
    let dm = open_draft_model(draft_path, path, max_draft, device, &model, &tok);
    // The default context may shrink to keep the model on the GPU (Backend::new).
    let (net, drafter) = backend_and_drafter(
        &model,
        dm.as_ref(),
        device,
        ctx,
        gpu_layers,
        kv,
        fit,
        lm::Placing {
            gpu_experts,
            ..lm::Placing::of(ctx)
        },
        ctx,
    );
    let split = is_split(&net);
    let max_draft = draft::split_k(
        k_given,
        max_draft,
        with_model,
        drafter.is_some(),
        split,
        sargs.resolve(sample::DEFAULT_TEMP).is_none(),
    );
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
    // Qwen2.5's template brings a default system message, Qwen3's none.
    let system = system
        .or_else(|| (model.arch == "qwen2").then(|| chat::DEFAULT_SYSTEM.into()))
        .or_else(|| chat::template_default_system(&model));
    let mut conv = chat::Chat::new(&net, &tok, system.as_deref(), ctx)
        .unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    if no_think && !conv.thinks() {
        fail(&format!(
            "--no-think: {path} has no <think> token (not a thinking model)"
        ));
    }
    conv.no_think = no_think;
    conv.max_reply = max_reply;
    conv.forget_to = forget_to;
    conv.max_draft = max_draft;
    conv.drafter = drafter.map(|d| Box::new(d) as Box<dyn draft::Draft>);
    // chat samples by default (as ollama does): greedy replies loop.
    conv.sampler = sargs
        .resolve(sample::DEFAULT_TEMP)
        .map(sample::Sampler::new);
    if conv.sampler.is_some() && conv.drafter.is_some() && !split {
        eprintln!("{}", draft::SAMPLED_NOTE);
    }

    let name = std::path::Path::new(path)
        .file_name()
        .map_or(path.as_str().into(), |n| n.to_string_lossy());
    let kvs = match &net.0 {
        Inner::Cuda(n) => format!(" | kv {}", n.kv_name()),
        Inner::Hybrid(n) => format!(" | {} | kv {}", n.describe(ctx), n.kv_name()),
        Inner::Pipeline(n) => format!(" | {} | kv {}", n.describe(ctx), n.kv_name()),
        Inner::Moe(n) => format!(" | {} | kv {}", n.describe(ctx), n.kv_name()),
        Inner::Cpu(_) => String::new(),
    };
    let seed = conv
        .sampler
        .as_ref()
        .map_or(" | greedy".into(), |s| format!(" | seed {}", s.cfg.seed));
    let drafts = dm.as_ref().map_or(String::new(), |d| {
        format!(" | draft model {} K={max_draft}", d.name())
    });
    // A session's identity: the model file, hashed once (then remembered).
    let model_ident = std::cell::OnceCell::new();
    let head_dim = kvsave::KvIo::kv_shape(&net).2;
    let mk_session = |p: std::path::PathBuf,
                      locked: Option<kvsave::Session>|
     -> std::io::Result<kvsave::Session> {
        let m = match model_ident.get() {
            Some(m) => m,
            None => {
                let m = kvsave::model_ident(&model, std::path::Path::new(path), head_dim)?;
                model_ident.get_or_init(|| m)
            }
        };
        // One process per session file (see `Session::lock`).
        let mut s = match locked {
            Some(s) => s,
            None => {
                let mut s = kvsave::Session::new(p, serde_json::Value::Null);
                if let Some(w) = s.lock()? {
                    eprintln!("axon: {w}");
                }
                s
            }
        };
        s.set_model(m.clone());
        Ok(s)
    };
    let explicit_seed = sargs.seed;
    let sess = ChatSession {
        make: &mk_session,
        cur: None,
        start: session,
        explicit_seed,
    };
    match chat_repl(
        &mut conv,
        &tok,
        &format!("{name} | ctx {ctx}{kvs}{seed}{drafts}"),
        sess,
    ) {
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

/// `axon chat`'s session file (--session, /save, /load).
struct ChatSession<'a> {
    /// A session of a canonical path (`kvsave::canonical`), locked, with the
    /// model's identity; given an already locked one, that one.
    make:
        &'a dyn Fn(std::path::PathBuf, Option<kvsave::Session>) -> std::io::Result<kvsave::Session>,
    cur: Option<kvsave::Session>,
    /// --session: loaded (if it exists) before the first message.
    /// (Locked before the model loaded.)
    start: Option<kvsave::Session>,
    /// --seed: wins over a loaded session's seed.
    explicit_seed: Option<u64>,
}

impl ChatSession<'_> {
    /// Load `p` into `conv`; a note saying how, or why not.
    fn load<M: kvsave::KvIo>(
        &mut self,
        conv: &mut chat::Chat<'_, M>,
        p: std::path::PathBuf,
        new_ok: bool,
    ) -> String {
        // Our own file again (by its canonical path): the same Session and
        // lock.
        let p = kvsave::canonical(&p);
        let reused = self.cur.as_ref().is_some_and(|c| c.path == p);
        let mut s = match self.cur.take_if(|_| reused) {
            Some(mut s) => {
                s.forget();
                s
            }
            None => match (self.make)(p.clone(), self.start.take_if(|s| s.path == p)) {
                Ok(s) => s,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    return format!("axon: {e}");
                }
                Err(e) => return format!("axon: session {}: {e}", p.display()),
            },
        };
        let shown = p.display().to_string();
        match conv.load_session(&mut s) {
            Ok(None) if new_ok => {
                self.cur = Some(s);
                format!("(new session {shown})")
            }
            Ok(None) => {
                if reused {
                    self.cur = Some(s);
                }
                format!("axon: no session file {shown}")
            }
            Ok(Some(r)) => {
                if let (Some(seed), Some(smp)) = (self.explicit_seed, conv.sampler.as_mut())
                    && smp.cfg.seed != seed
                {
                    *smp = sample::Sampler::new(sample::SampleCfg { seed, ..smp.cfg });
                }
                self.cur = Some(s);
                let how = if r.from_file {
                    "loaded from the file".to_string()
                } else {
                    format!(
                        "prefilled again ({})",
                        r.why.as_deref().unwrap_or("no cache in the file")
                    )
                };
                format!(
                    "(session {shown}: {} positions {how} in {:.0} ms{})",
                    r.positions,
                    r.secs * 1e3,
                    if r.rng { "; sampler continues" } else { "" }
                )
            }
            Err(e) => {
                if reused {
                    self.cur = Some(s);
                }
                format!("axon: session {shown}: {e}")
            }
        }
    }

    /// Save to the current file, if any: a note on failure only.
    fn save<M: kvsave::KvIo>(&mut self, conv: &chat::Chat<'_, M>) -> Option<String> {
        let s = self.cur.as_mut()?;
        conv.save_session(s)
            .err()
            .map(|e| format!("axon: saving {}: {e}", s.path.display()))
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
fn chat_repl<M: kvsave::KvIo>(
    conv: &mut chat::Chat<'_, M>,
    tok: &tokenizer::Tokenizer,
    banner: &str,
    mut sess: ChatSession,
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
    if let Some(p) = sess.start.as_ref().map(|s| s.path.clone()) {
        let n = sess.load(conv, p, true);
        if n.starts_with("axon:") {
            // A session that can't be used isn't overwritten: stop here.
            note(&n);
            return Err(std::io::Error::other("--session: not loaded"));
        }
        note(&format!("{dim}{n}{undim}"));
    }
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
                    if let Some(e) = sess.save(conv) {
                        note(&e);
                    }
                }
                "save" => {
                    let arg = arg.trim();
                    let canon = kvsave::canonical(std::path::Path::new(arg));
                    let same = sess.cur.as_ref().is_some_and(|c| c.path == canon);
                    if !arg.is_empty() && !same {
                        match (sess.make)(canon, None) {
                            Ok(s) => sess.cur = Some(s),
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                note(&format!("axon: {e}"));
                                continue;
                            }
                            Err(e) => {
                                note(&format!("axon: session {arg}: {e}"));
                                continue;
                            }
                        }
                    }
                    match &sess.cur {
                        None => note("axon: usage: /save FILE (no session file yet)"),
                        Some(s) => {
                            let shown = s.path.display().to_string();
                            match sess.save(conv) {
                                Some(e) => note(&e),
                                None => note(&format!("{dim}(saved {shown}){undim}")),
                            }
                        }
                    }
                }
                "load" => {
                    let arg = arg.trim();
                    let p = if arg.is_empty() {
                        sess.cur.as_ref().map(|s| s.path.clone())
                    } else {
                        Some(arg.into())
                    };
                    match p {
                        None => note("axon: usage: /load FILE"),
                        Some(p) => {
                            let n = sess.load(conv, p, false);
                            if n.starts_with("axon:") {
                                note(&n);
                            } else {
                                note(&format!("{dim}{n}{undim}"));
                            }
                        }
                    }
                }
                "system" if arg.trim().is_empty() => note("axon: usage: /system TEXT"),
                "system" => match conv.set_system(Some(arg.trim())) {
                    Ok(()) => {
                        note(&format!(
                            "{dim}(new system message; conversation cleared){undim}"
                        ));
                        if let Some(e) = sess.save(conv) {
                            note(&e);
                        }
                    }
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
        // gpt-oss (harmony): only the final channel is shown.
        let mut channels = conv.channels();
        let mut reasoning = 0usize;
        let res = conv.reply(text, cancel, &mut |id| {
            let bytes = tok.decode_bytes(&[id]);
            if let Some(ch) = channels.as_mut() {
                match ch.push(id, &bytes) {
                    chat::Part::Answer => {}
                    chat::Part::Reasoning => {
                        reasoning += 1;
                        return Ok(());
                    }
                    chat::Part::Hidden => return Ok(()),
                }
            }
            let s = stream.push(&bytes);
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
        if let Some(e) = sess.save(conv) {
            note(&e);
        }
        let rate = |n: usize, s: f64| if s > 0.0 { n as f64 / s } else { 0.0 };
        let (used, size) = conv.used();
        let mut notes = String::new();
        if reasoning > 0 {
            notes.push_str(&format!(" | reasoning {reasoning} tok, not shown"));
        }
        if r.drafted > 0 {
            notes.push_str(&format!(" | drafts {}/{} kept", r.accepted, r.drafted));
        }
        if r.interrupted {
            notes.push_str(" | stopped");
        } else if r.cut {
            notes.push_str(&format!(" | cut at {} tokens", r.ids.len()));
        }
        if r.dropped > 0 {
            let f = conv.forget_to;
            let to = if f <= 0.0 || f >= 1.0 {
                "to fit".to_string()
            } else if f == 0.5 {
                "(down to half the context)".to_string()
            } else {
                format!("(down to {:.0}% of the context)", f * 100.0)
            };
            notes.push_str(&format!(
                " | forgot the oldest {} exchange{} {to}",
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
    use super::{gen_budget, placement, utf8_args};
    use crate::generate::{GenOpts, argmax, generate};

    /// `gen -n` past the trained context stops there; usize::MAX no longer
    /// overflows the cache size (it wrapped to 1 and panicked, or to a
    /// short slice); a prompt alone past the context is an error.
    #[test]
    fn gen_budget_caps_and_never_overflows() {
        assert_eq!(gen_budget(5, 32, 32768), Ok(32));
        assert_eq!(gen_budget(5, 32762, 32768), Ok(32762));
        assert_eq!(gen_budget(5, 32763, 32768), Ok(32762));
        assert_eq!(gen_budget(5, usize::MAX, 32768), Ok(32762));
        assert_eq!(gen_budget(1, 1 << 63, 32768), Ok(32766));
        assert_eq!(gen_budget(32767, 10, 32768), Ok(0));
        assert!(
            gen_budget(32768, 0, 32768)
                .unwrap_err()
                .contains("trained on 32768")
        );
        assert_eq!(gen_budget(5, 1000, 0), Ok(1000));
        assert!(
            gen_budget(5, usize::MAX, 0)
                .unwrap_err()
                .contains("too large")
        );
        assert!(gen_budget(1, usize::MAX - 1, 0).is_err());
        assert_eq!(gen_budget(1, usize::MAX - 2, 0), Ok(usize::MAX - 2));
    }

    /// A non-UTF-8 argument is an error naming it, not a panic.
    #[test]
    fn non_utf8_argument_is_an_error() {
        use std::os::unix::ffi::OsStringExt;
        let ok = ["axon", "gen"].map(std::ffi::OsString::from);
        assert_eq!(utf8_args(ok.into_iter()).unwrap(), ["axon", "gen"]);
        let bad = [
            "axon".into(),
            "gen".into(),
            std::ffi::OsString::from_vec(b"m\xff.gguf".to_vec()),
        ];
        let e = utf8_args(bad.into_iter()).unwrap_err();
        assert!(e.starts_with("argument 2 (m\u{fffd}.gguf)"), "{e}");
    }
    use crate::lm::Device;
    use crate::model::Qwen2;
    use crate::spec::PromptLookup;

    /// --gpu-layers 0 is the CPU backend, decided before the model loads
    /// (so it's copied, as --device cpu, not mapped); other counts stay.
    #[test]
    fn gpu_layers_zero_is_the_cpu() {
        assert_eq!(placement(Device::Cuda, Some(0)), (Device::Cpu, None));
        assert_eq!(placement(Device::Cuda, Some(7)), (Device::Cuda, Some(7)));
        assert_eq!(placement(Device::Cpu, None), (Device::Cpu, None));
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
