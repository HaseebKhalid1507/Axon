//! `axon serve`: an OpenAI-compatible HTTP server (stages S1 and S2-lite
//! of specs/r4-serve.md: on the GPU, many requests decode together,
//! `engine.rs`; on the CPU or a split, one request at a time).
//!
//! Threads: the acceptor (the caller of `run`), one per connection (HTTP
//! parsing, JSON, tokenizing the prompt, writing the response or the SSE
//! stream), and the engine, which owns the model and its one KV cache (the
//! GPU handle is `!Send`: everything that touches the backend stays on that
//! thread). Connections hand the engine a `Job` over a channel and get
//! `Event`s back on one of their own; the engine runs jobs in arrival order:
//! a job alone through `generate` exactly as `axon gen` / `axon chat` do,
//! several together in batched steps over KV slots (`engine.rs`), each
//! with the tokens it would get alone.
//!
//! The engine detokenizes and matches stop strings itself, so it stops at a
//! stop string on the token that completes it and counts tokens exactly;
//! per token that is a few microseconds next to a forward pass.
//!
//! Prefix reuse: the engine remembers which tokens the cache holds (`hist`).
//! A request whose prompt starts with some of them (a chat follow-up resends
//! the conversation; requests sharing a system prompt) feeds only the rest.
//! Exact: a token's cache rows depend only on the tokens before it, and the
//! backends give the same bits however a prompt is split into batches (the
//! contract `axon chat` relies on).
//!
//! Cancelling: a client that goes away (a failed write, or EOF seen while
//! waiting) sets the job's flag; the engine checks it between prefill
//! batches and before every token, so generation stops within one step.
//!
//! Failure: a panic in the engine (CUDA errors panic, and are sticky) is
//! logged and the process exits with code 70 for a supervisor to restart
//! it, instead of answering 500 forever while /health says ok.

pub mod api;
mod engine;
pub mod http;
pub mod tools;

use crate::chat::{self, Framing, special};
use crate::cuda::qwen2::CacheKv;
use crate::kv::KvType;
use crate::lm::{Backend, BackendCache, BatchLm, Device, Lm};
use crate::sample::{SampleArgs, SampleCfg, Sampler};
use crate::spec::PromptLookup;
use crate::tokenizer::{Tokenizer, Utf8Stream};
use crate::{GenOpts, generate};
use api::{ApiError, Message, Prompt, Role, StopMatcher, Usage};
use engine::{Batch, Seq};
use http::{Conn, Peer, Request};
use std::collections::VecDeque;
use std::io::{self, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::time::{Duration, Instant};

/// Most open connections (default). One more closes the connection that
/// has waited longest for a request (`Shared::waiting`), or if every one is
/// busy with a request, is answered 503 and closed.
pub const MAX_CONNS: usize = 256;
/// Most requests waiting for the engine (default); more are answered 503.
pub const MAX_QUEUE: usize = 128;
/// Most requests decoded together on the GPU (default, `--parallel`).
pub const PARALLEL: usize = 8;
/// An idle keep-alive connection is closed after this long, and a request
/// must arrive within it once started.
const IDLE: Duration = Duration::from_secs(30);
/// A client that doesn't take a response's bytes for this long is dropped
/// (its generation cancelled).
const WRITE_TIMEOUT: Duration = Duration::from_secs(60);
/// How often a connection waiting for tokens checks that its client is
/// still there (a closed socket otherwise shows only at the next write).
const POLL: Duration = Duration::from_millis(200);
/// `linger`: longest wait, most bytes dropped.
const LINGER: Duration = Duration::from_secs(2);
const LINGER_BYTES: usize = 4 << 20;
/// Most bytes the acceptor drops from a connection it answers 503.
const ACCEPT_DRAIN: usize = 64 << 10;

/// `axon serve`'s settings.
pub struct Opts {
    pub path: String,
    pub host: String,
    pub port: u16,
    pub device: Device,
    pub gpu_layers: Option<usize>,
    pub kv: KvType,
    pub ctx: Option<usize>,
    pub max_draft: usize,
    /// A draft model guesses the tokens (draft.rs), if `max_draft` > 0.
    pub draft_model: Option<String>,
    /// Sampling flags: the default for fields a request leaves out.
    pub sample: SampleArgs,
    /// Qwen3: answer without reasoning unless a request asks for it.
    pub no_think: bool,
    /// Qwen3: leave `<think>...</think>` in `content` instead of moving it
    /// to `reasoning_content`.
    pub think_inline: bool,
    /// Another name the model answers to.
    pub alias: Option<String>,
    /// Print a line per request on stderr.
    pub log: bool,
    /// Most tokens a request may generate (None: up to the context).
    pub max_tokens: Option<usize>,
    /// `Access-Control-Allow-Origin` for every response, and CORS
    /// preflights answered (None: no CORS headers).
    pub cors: Option<String>,
    pub max_conns: usize,
    pub max_queue: usize,
    /// Most requests decoded together (continuous batching, GPU only);
    /// 1: one at a time.
    pub parallel: usize,
    /// `--parallel` was given (not the default): a backend that can't batch
    /// is then an error, not a quiet one-at-a-time.
    pub parallel_set: bool,
    /// Positions per slot at most (None: the context; tests shrink the pool
    /// with it).
    pub slot_ctx: Option<usize>,
    /// Prompt tokens a batched step feeds at least while requests decode
    /// (`--prefill-chunk`; 0: prompts fed whole): a long prompt prefills in
    /// chunks between the running requests' steps.
    pub prefill_chunk: usize,
    /// A batched step's time budget for its chunk and decode, in ms
    /// (`--prefill-ms`; 0: chunks of exactly `prefill_chunk`).
    pub prefill_ms: usize,
    /// `--rope-scaling yarn` and friends (rope.rs).
    pub rope: crate::rope::RopeArgs,
}

/// The chat framing `axon chat` uses (chat.rs: ChatML, or Llama 3's
/// headers), for whole message lists.
pub struct Template {
    fr: Framing,
    /// Qwen3: `<think>`, `</think>`, and the empty reasoning block
    /// `enable_thinking: false` opens a reply with.
    think: Option<(u32, u32, Vec<u32>)>,
    /// Used when the messages don't start with a system message (Qwen2.5's
    /// template brings one, Qwen3's and Llama 3's none).
    default_system: Option<&'static str>,
}

impl Template {
    /// None if the vocabulary has no chat markers (no chat endpoint).
    pub fn new(tok: &Tokenizer, arch: &str) -> Option<Template> {
        let fr = Framing::new(tok)?;
        let think = match (special(tok, "<think>"), special(tok, "</think>")) {
            (Some(open), Some(close)) => {
                Some((open, close, tok.encode_special("<think>\n\n</think>\n\n")))
            }
            _ => None,
        };
        Some(Template {
            fr,
            think,
            // As `axon chat`: Qwen2.5's default system message for qwen2.
            default_system: (arch == "qwen2").then_some(chat::DEFAULT_SYSTEM),
        })
    }

    /// Any of these ends a reply (`axon chat`'s set).
    pub fn stop(&self) -> &[u32] {
        &self.fr.stop
    }

    /// The prompt for `msgs`: the system block (a leading system message,
    /// else the default one; Llama 3 always has one), each message framed,
    /// then the assistant's opening (with the empty reasoning block if
    /// `no_think` and the model thinks), as `axon chat` frames a
    /// conversation. An earlier reply of a thinking model is kept as its
    /// text after `</think>`, leading newlines dropped, as the Qwen3
    /// template (and `axon chat`) render it.
    pub fn render(&self, tok: &Tokenizer, msgs: &[Message], no_think: bool) -> Vec<u32> {
        let (system, rest) = match msgs.split_first() {
            Some((m, rest)) if m.role == Role::System => (Some(m.content.as_str()), rest),
            _ => (self.default_system, msgs),
        };
        let mut t = self.fr.system(tok, system);
        for (i, m) in rest.iter().enumerate() {
            let text = match (m.role, &self.think) {
                (Role::Assistant, Some(_)) => match m.content.rsplit_once("</think>") {
                    Some((_, c)) => c.trim_start_matches('\n'),
                    None => &m.content,
                },
                _ => &m.content,
            };
            // Mistral's first user message carries the system message
            // (llama.cpp's mistral-v3 placement; mistral_common V2/V3 and HF's
            // v0.3 template use the last user message: chat.rs module doc).
            t.extend(match (i, m.role) {
                (0, Role::User) => self.fr.frame_first(tok, "user", system, text, None),
                _ => self.fr.frame(tok, m.role.name(), text, None),
            });
        }
        t.extend(self.fr.opening(tok, "assistant"));
        if no_think && let Some((_, _, empty)) = &self.think {
            t.extend_from_slice(empty);
        }
        t
    }

    pub fn thinks(&self) -> bool {
        self.think.is_some()
    }
}

/// What a connection asks the engine for.
struct Job {
    prompt: Vec<u32>,
    /// At most this many tokens (the context's room already counted).
    n_gen: usize,
    sample: Option<SampleCfg>,
    stop_ids: Vec<u32>,
    stops: Vec<String>,
    /// Route `<think>...</think>` (these ids) to reasoning.
    think: Option<(u32, u32)>,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<Event>,
    id: u64,
}

enum Event {
    Text { content: String, reasoning: String },
    Done { usage: Usage, finish: &'static str },
    Error(String),
}

/// What the connection threads share.
struct Shared {
    /// Set once the model is loaded; until then generation and /health
    /// answer 503.
    model: OnceLock<Loaded>,
    model_id: String,
    alias: Option<String>,
    no_think: bool,
    think_inline: bool,
    sample: SampleArgs,
    max_tokens: Option<usize>,
    /// The `extra` header lines of every response (CORS).
    extra: String,
    cors: bool,
    max_conns: usize,
    /// Server start, for /v1/models.
    created: u64,
    /// Bounded: a full queue is a 503, never unbounded memory.
    jobs: mpsc::SyncSender<Job>,
    conns: AtomicUsize,
    next_id: AtomicU64,
    stop: Arc<AtomicBool>,
    /// Connections waiting for (or in the middle of receiving) a request,
    /// or lingering after an error, oldest first: (connection id, a handle
    /// to shut it down). At the connection limit a new connection closes
    /// the first one, so idle or trickling sockets (slow-loris) can't lock
    /// everyone out for the `IDLE` timeout. Taking an entry off the list
    /// (here, or by its own thread once a request has arrived) is the one
    /// decision under this lock: a thread that finds its entry already
    /// gone was evicted and doesn't serve the request it may just have
    /// read (its client sees the connection closed, as for any idle
    /// keep-alive connection a server closes; a non-idempotent request
    /// sent at that instant is lost, which only happens at the limit). A
    /// request whose thread took its entry first is never evicted. Each
    /// connection holds a second fd for this handle (2 per connection:
    /// with --max-conns N, allow ~2N fds in `ulimit -n`).
    waiting: WaitList,
    next_conn: AtomicU64,
    /// `Server::queued`.
    queued: Arc<AtomicU64>,
}

/// `Shared::waiting`: (connection id, a handle to shut it down), oldest first.
#[derive(Default)]
struct WaitList(std::sync::Mutex<Vec<(u64, Arc<TcpStream>)>>);

impl WaitList {
    /// Close the connection that has waited longest for a request, if
    /// any. Its thread sees the end of the stream and exits.
    fn evict_oldest(&self) -> bool {
        let mut w = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if w.is_empty() {
            return false;
        }
        let (_, s) = w.remove(0);
        let _ = s.shutdown(std::net::Shutdown::Both);
        true
    }

    /// Register connection `id` as waiting (`Some`) or take it off the
    /// list (`None`). Returns whether it was on the list: false after a
    /// `None` means `evict_oldest` took it (and shut it down) first.
    fn set(&self, id: u64, s: Option<&Arc<TcpStream>>) -> bool {
        let mut w = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let n = w.len();
        w.retain(|(i, _)| *i != id);
        let was = w.len() != n;
        if let Some(s) = s {
            w.push((id, s.clone()));
        }
        was
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// Decrements the connection count however the connection's thread ends.
struct ConnGuard<'a>(&'a AtomicUsize);

impl Drop for ConnGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Shared {
    fn serves(&self, model: &str) -> bool {
        model == self.model_id || self.alias.as_deref() == Some(model)
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The engine's side of the text: detokenizing, the reasoning split, stop
/// strings.
struct TextOut<'t> {
    tok: &'t Tokenizer,
    utf8: Utf8Stream,
    think: Option<(u32, u32)>,
    /// Inside `<think>...</think>`.
    reasoning: bool,
    /// Before the first token (a thinking reply opens with `<think>`).
    first: bool,
    /// Drop newlines until the next other character.
    strip_nl: bool,
    stop: StopMatcher,
}

impl TextOut<'_> {
    /// One more token: (content, reasoning, a stop string was hit).
    fn push(&mut self, id: u32) -> (String, String, bool) {
        let first = std::mem::replace(&mut self.first, false);
        if let Some((open, close)) = self.think {
            if first && id == open {
                self.reasoning = true;
                self.strip_nl = true;
                return Default::default();
            }
            if self.reasoning && id == close {
                self.reasoning = false;
                self.strip_nl = true;
                return (String::new(), self.utf8.finish(), false);
            }
        }
        let mut s = self.utf8.push(&self.tok.decode_bytes(&[id]));
        if self.strip_nl {
            s = s.trim_start_matches('\n').to_string();
            self.strip_nl = s.is_empty();
        }
        if self.reasoning {
            return (String::new(), s, false);
        }
        let (c, hit) = self.stop.push(&s);
        (c, String::new(), hit)
    }

    /// The end of the output: what was held back.
    fn finish(&mut self) -> (String, String) {
        let s = self.utf8.finish();
        if self.reasoning {
            return (self.stop.finish(), s);
        }
        let (mut c, hit) = self.stop.push(&s);
        if !hit {
            c.push_str(&self.stop.finish());
        }
        (c, String::new())
    }
}

/// The engine: the model, its one cache, and what the cache holds.
struct Engine<'a> {
    net: &'a Backend<'a>,
    tok: &'a Tokenizer,
    cache: BackendCache,
    /// The tokens whose rows the cache holds, from position 0.
    hist: Vec<u32>,
    max_draft: usize,
    /// Guesses from a draft model instead of prompt lookup.
    drafter: Option<crate::draft::ModelDrafter<Backend<'static>>>,
    log: bool,
    /// Tokens generated so far, all requests.
    generated: Arc<AtomicU64>,
}

/// Run `f`; a panic in it (CUDA errors panic, and are sticky: every later
/// job would fail too, and the caches' contents are unknown) exits the
/// process (code 70, EX_SOFTWARE) for a supervisor to restart the server,
/// rather than answer 500 forever while /health says ok.
fn or_exit<T>(what: &str, f: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(t) => t,
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| p.downcast_ref::<&str>().copied())
                .unwrap_or("?");
            eprintln!("axon serve: the engine failed on {what}: {msg}; exiting");
            std::process::exit(70);
        }
    }
}

/// AXON_SERVE_DEBUG=1: a line on stderr per request saying how its prompt
/// is prefilled (alone, the single-stream path, or in chunks in a slot);
/// benchmarks use it to tell the paths apart.
fn debug() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var_os("AXON_SERVE_DEBUG").is_some_and(|v| v == "1"))
}

/// The detokenizer and stop matcher of `job`'s reply.
fn text_out<'t>(tok: &'t Tokenizer, job: &Job) -> TextOut<'t> {
    TextOut {
        tok,
        utf8: Utf8Stream::default(),
        think: job.think,
        reasoning: false,
        first: true,
        strip_nl: false,
        stop: StopMatcher::new(job.stops.clone()),
    }
}

/// Whether `j` can run in a slot of `slot_ctx` positions (its prompt and
/// all it may generate).
fn fits_slot(j: &Job, slot_ctx: usize) -> bool {
    j.n_gen > 0 && j.prompt.len() + j.n_gen <= slot_ctx
}

/// When a request running alone (`Engine::job`) hands over to the batch:
/// as soon as a newly arrived request could join it.
struct Handover<'r> {
    jobs: &'r mpsc::Receiver<Job>,
    waiting: &'r mut VecDeque<Job>,
    /// The batch's positions per slot, if it has a free slot.
    slot_ctx: Option<usize>,
    /// Most requests to hold off the channel (the batch's slots).
    room: usize,
}

impl Handover<'_> {
    /// Take new requests (while none of those waiting fits a slot, up to
    /// `room`); whether one of them could run in a slot, and so should not
    /// wait for the one running alone.
    fn due(&mut self) -> bool {
        let Some(ctx) = self.slot_ctx else {
            return false;
        };
        self.waiting.retain(|j| !j.cancel.load(Ordering::Relaxed));
        while !self.waiting.iter().any(|j| fits_slot(j, ctx)) && self.waiting.len() < self.room {
            match self.jobs.try_recv() {
                Ok(j) => self.waiting.push_back(j),
                Err(_) => break,
            }
        }
        self.waiting.iter().any(|j| fits_slot(j, ctx))
    }
}

impl<'a> Engine<'a> {
    fn run(&mut self, job: Job, ho: Option<Handover>) -> Option<Seq<'a>> {
        if job.cancel.load(Ordering::Relaxed) {
            return None;
        }
        let id = job.id;
        or_exit(&format!("request #{id}"), || self.job(job, ho))
    }

    /// Run `job` alone on the engine's cache (stage S1). With `ho`, it
    /// hands over (returns its state, the cache holding its rows) when
    /// another request arrives that could share a step with it.
    fn job(&mut self, owned: Job, mut ho: Option<Handover>) -> Option<Seq<'a>> {
        let job = &owned;
        let cancelled = || job.cancel.load(Ordering::Relaxed);
        let t0 = Instant::now();
        let p = &job.prompt;
        let last = p.len() - 1;
        if debug() {
            eprintln!(
                "axon serve: #{} prefill: single-stream, {} tok",
                job.id,
                p.len()
            );
        }
        // Reuse the cached prefix; the last prompt token is always fed (its logits).
        let reuse = self
            .hist
            .iter()
            .zip(p)
            .take_while(|(a, b)| a == b)
            .count()
            .min(last);
        self.hist.truncate(reuse);
        let pb = self.net.prefill_batch();
        for batch in p[reuse..last].chunks(pb) {
            if cancelled() {
                return None;
            }
            self.net.feed_batch(batch, self.hist.len(), &mut self.cache);
            self.hist.extend_from_slice(batch);
        }
        if cancelled() {
            return None;
        }
        let logits = self.net.forward(p[last], last, &mut self.cache);
        self.hist.push(p[last]);
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut sampler = job.sample.map(Sampler::new);
        let mut source = match self.drafter.as_mut() {
            Some(d) => crate::draft::Drafter::Model(d),
            None => crate::draft::Drafter::Lookup(PromptLookup::new()),
        };
        let mut out = text_out(self.tok, job);
        let mut ids = Vec::new();
        let generated = &self.generated;
        // A request that fits a slot may hand over to the batch.
        if ho
            .as_ref()
            .is_some_and(|h| h.slot_ctx.is_none_or(|c| p.len() + job.n_gen > c))
        {
            ho = None;
        }
        // Why the emit callback ended generation: a stop string, the client
        // left, or another request came that it hands over for.
        let (mut hit, mut gone, mut handed) = (false, false, false);
        let t1 = Instant::now();
        let res = generate(
            self.net,
            &mut self.cache,
            p,
            logits,
            GenOpts {
                n_gen: job.n_gen,
                stop: &job.stop_ids,
                max_draft: self.max_draft,
                sample: sampler.as_mut(),
            },
            &mut |h, max| source.draft(h, max),
            &mut |id| {
                if cancelled() {
                    gone = true;
                    return Err(io::ErrorKind::Interrupted.into());
                }
                ids.push(id);
                generated.fetch_add(1, Ordering::Relaxed);
                let (content, reasoning, h) = out.push(id);
                if (!content.is_empty() || !reasoning.is_empty())
                    && job.tx.send(Event::Text { content, reasoning }).is_err()
                {
                    gone = true;
                }
                hit = h;
                if hit || gone {
                    return Err(io::ErrorKind::Interrupted.into());
                }
                if ids.len() < job.n_gen && ho.as_mut().is_some_and(Handover::due) {
                    handed = true;
                    return Err(io::ErrorKind::Interrupted.into());
                }
                Ok(())
            },
        );
        if handed {
            // Rows 0..hist.len() - 1 are in the cache: all but the token
            // just picked, the batch's first row for it.
            // The draft model (if any) is left as it is: the batch doesn't
            // draft, and its next draft call matches its cached tokens
            // against that request's history first (draft.rs).
            let mut hist = p.clone();
            hist.extend_from_slice(&ids);
            self.hist = hist[..hist.len() - 1].to_vec();
            return Some(Seq {
                job: owned,
                slot: 0,
                hist,
                ids,
                sampler,
                out,
                cached: reuse,
                prompt_ms,
                t1,
            });
        }
        let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        // Every token but the last went through the model; the last too if
        // a stop token ended the reply (its pass picked the stop token).
        let by_stop_id = res.is_ok() && ids.len() < job.n_gen;
        let fed = ids.len() - usize::from(!by_stop_id && !ids.is_empty());
        self.hist.extend_from_slice(&ids[..fed]);
        if let Err(e) = res
            && !hit
            && !gone
        {
            let _ = job.tx.send(Event::Error(e.to_string()));
            return None;
        }
        if gone {
            if self.log {
                eprintln!(
                    "axon serve: #{} cancelled after {} tokens",
                    job.id,
                    ids.len()
                );
            }
            return None;
        }
        let (content, reasoning) = out.finish();
        if !content.is_empty() || !reasoning.is_empty() {
            let _ = job.tx.send(Event::Text { content, reasoning });
        }
        let finish = if hit || by_stop_id { "stop" } else { "length" };
        let usage = Usage {
            prompt_tokens: p.len(),
            cached_tokens: reuse,
            completion_tokens: ids.len(),
            prompt_ms,
            decode_ms,
        };
        if self.log {
            eprintln!(
                "axon serve: #{} prompt {} tok ({} cached) in {:.1} ms, {} tok in {:.1} ms ({:.1} tok/s), {finish}",
                job.id,
                p.len(),
                reuse,
                prompt_ms,
                ids.len(),
                decode_ms,
                if decode_ms > 0.0 {
                    ids.len() as f64 * 1e3 / decode_ms
                } else {
                    0.0
                }
            );
        }
        let _ = job.tx.send(Event::Done { usage, finish });
        None
    }
}

/// What the engine thread reports once the model is loaded.
struct Loaded {
    tok: &'static Tokenizer,
    template: Option<Template>,
    ctx: usize,
    /// The longest token's bytes: a prompt of more than `ctx` times this
    /// many bytes is more than `ctx` tokens, refused before encoding it.
    max_token_bytes: usize,
    banner: String,
    /// What decides the output bits (`crate::identity`): every response's
    /// `system_fingerprint`, GET /identity.
    identity: crate::identity::Identity,
}

/// Load the model and serve jobs until every sender is gone. Runs on its
/// own thread: the backend never leaves it.
fn engine(
    o: EngineOpts,
    ready: mpsc::Sender<Loaded>,
    jobs: mpsc::Receiver<Job>,
    stop: Arc<AtomicBool>,
) {
    let (device, gpu_layers) = crate::placement(o.device, o.gpu_layers);
    let rope = crate::rope::RopeArgs {
        ctx: o.ctx,
        ..o.rope.clone()
    };
    let model = crate::open_model_with(&o.path, crate::copy_for(device), &rope);
    let trained = model.ctx_len() as usize;
    let yarn = crate::rope::yarn(&model).is_some();
    let fit = o.ctx.is_none();
    // The default: the trained context (the whole model on one GPU: as
    // much of it as fits, `Backend::new`'s fit); a split, a pipeline or
    // --gpu-layers keep the old cap (`SPLIT_CTX`).
    let want = match o.ctx {
        Some(n) if trained > 0 && n > trained => crate::fail(&format!(
            "--ctx {n} is past the {trained} tokens {} {}{}",
            o.path,
            if yarn {
                "runs with YaRN"
            } else {
                "was trained on"
            },
            if yarn {
                String::new()
            } else {
                crate::rope::ctx_hint(n, trained)
            }
        )),
        Some(n) => n,
        None if trained > 0 => trained,
        None => crate::DEFAULT_CHAT_CTX,
    };
    let split_ctx = if fit { want.min(SPLIT_CTX) } else { want };
    // The drafter's reserve (`load_pair`): a split's context if
    // --gpu-layers makes one, else at most AGENT_CTX (a 131072-token
    // reserve would take the main model's memory); it drafts up to the
    // context the main model gets, if that is shorter.
    let split = gpu_layers.is_some_and(|g| crate::hybrid::block_count(&model).is_ok_and(|n| g < n));
    let draft_ctx = match (fit, split) {
        (false, _) => want,
        (true, true) => split_ctx,
        (true, false) => want.min(AGENT_CTX),
    };
    let tok: &'static Tokenizer = Box::leak(Box::new(Tokenizer::from_gguf(&model)));
    let dm = o.draft_model.as_ref().filter(|_| o.max_draft > 0).map(|p| {
        crate::draft::DraftModel::open(p, crate::copy_for(device), &model, tok)
            .unwrap_or_else(|e| crate::fail(&format!("--draft-model {e}")))
    });
    // The Q6_K prefill copy waits for the cache and slots (below).
    let place = crate::lm::Placing {
        split_ctx,
        defer_q6k8: true,
    };
    let (mut net, drafter) = crate::backend_and_drafter(
        &model,
        dm.as_ref(),
        device,
        want,
        gpu_layers,
        o.kv,
        fit,
        place,
        draft_ctx,
    );
    let ctx = net.ctx(want);
    let why = match &net {
        _ if ctx >= want => None,
        Backend::Cuda(_) => Some(gpu_fit_note(ctx, want, o.kv, gpu_layers.is_some())),
        Backend::Pipeline(_) => Some(format!(
            "{ctx} fit on the GPUs beside the weights, of the model's {want} \
             (--ctx N for another{})",
            if o.kv == KvType::Q8 {
                ""
            } else {
                "; --kv q8 halves the KV cache"
            }
        )),
        Backend::Hybrid(_) => Some(format!(
            "the default for a CPU/GPU split (automatic or --gpu-layers): a cache for the \
             model's {want} tokens would take GPU memory from its layers \
             (--ctx N for another)"
        )),
        Backend::Cpu(_) => None,
    };
    let (ctx, why) = match &net {
        Backend::Cpu(n) if fit => {
            let elem = if crate::kv::use_f32() { 4 } else { 2 };
            let per_pos = (n.n_layers * 2 * n.n_kv_heads * n.head_dim * elem) as u64;
            // A mapped model's pages must stay in memory too.
            let mapped = match model.file_backed() {
                true => std::fs::metadata(&o.path).map_or(0, |m| m.len()),
                false => 0,
            };
            cpu_ctx(want, per_pos, mapped, crate::gguf::mem_available())
        }
        _ => (ctx, why),
    };
    if let Some(why) = why.filter(|_| o.log) {
        eprintln!("axon serve: context {ctx} tokens: {why}");
    }
    let template = Template::new(tok, &model.arch);
    let kvs = match &net {
        Backend::Cuda(n) => format!("cuda, kv {}", n.kv_name()),
        Backend::Hybrid(n) => format!("{}, kv {}", n.describe(ctx), n.kv_name()),
        Backend::Pipeline(n) => format!("{}, kv {}", n.describe(ctx), n.kv_name()),
        Backend::Cpu(_) => "cpu".into(),
    };
    let drafts = dm.as_ref().map_or(String::new(), |d| {
        format!(", draft model {} K={}", d.name(), o.max_draft)
    });
    // Continuous batching runs on one GPU holding every layer: the CPU, a
    // CPU/GPU split and a multi-GPU pipeline (AXON_GPUS) serve one request
    // at a time, and an explicit --parallel N there is refused.
    if o.parallel > 1 && o.parallel_set && !matches!(net, Backend::Cuda(_)) {
        crate::fail(&format!(
            "--parallel {}: batching many requests needs the whole model on one GPU \
             (--device cuda; not with --gpu-layers or several AXON_GPUS); this backend \
             serves one request at a time (--parallel 1)",
            o.parallel
        ));
    }
    // The slots and their context (`plan_slots`), measured against the GPU
    // memory left before the single-request cache is made: a pool of N
    // slots and that cache are N + 1 caches.
    let plan = match &net {
        Backend::Cuda(n) if o.parallel > 1 => match n.free_bytes() {
            Ok(free) => {
                let free = free as usize;
                let fits = |kv, s, c| n.serve_bytes_kv(kv, s, c) + POOL_SPARE <= free;
                let kv = match o.kv {
                    KvType::Q8 => CacheKv::Q8 { group: 0 },
                    _ => CacheKv::F16,
                };
                let mut p =
                    plan_slots(ctx, !fit, o.parallel, o.parallel_set, |s, c| fits(kv, s, c));
                // The q8 cache's slots at this context, for the note.
                if p.slots == 1 && kv == CacheKv::F16 && !o.parallel_set {
                    let q8 = CacheKv::Q8 { group: 0 };
                    let s8 = (2..=o.parallel).rev().find(|&s| fits(q8, s, ctx));
                    if let (Some(s8), Some(note)) = (s8, p.note.as_mut()) {
                        note.push_str(&format!("; --kv q8: {s8} at {ctx}"));
                    }
                }
                p
            }
            Err(e) => SlotPlan::one(ctx, Some(e)),
        },
        _ => SlotPlan::one(ctx, None),
    };
    if let Some(note) = plan.note.as_ref().filter(|_| o.log) {
        eprintln!("axon serve: {note}");
    }
    // With a pool, the single-request cache is a slot's length: the
    // context served is a slot's (every request fits one).
    let ctx = plan.ctx;
    let cache = net.new_cache(ctx);
    let pool = match plan.slots {
        0 | 1 => None,
        n => {
            // Tests shrink the slots below the context (`Opts::slot_ctx`).
            let slot_ctx = o.slot_ctx.map_or(ctx, |c| c.min(ctx));
            match net.new_pool(n, slot_ctx) {
                Ok(pool) => Some((pool, n, slot_ctx)),
                Err(why) => {
                    if o.log {
                        eprintln!("axon serve: one request at a time ({why})");
                    }
                    None
                }
            }
        }
    };
    // The Q6_K prefill copy only out of what the cache and slots leave
    // (prefill-only: the same products without it, slower).
    if let Backend::Cuda(n) = &mut net {
        match n.make_q6k8_spare() {
            Ok(Some(false)) if o.log => eprintln!(
                "axon serve: no Q6_K prefill copy ({:.2} GiB free beside the caches; it \
                 speeds up prefill): a smaller --ctx or --parallel, or --kv q8, makes room",
                n.free_bytes().unwrap_or(0) as f64 / GIB
            ),
            Err(e) => eprintln!("axon serve: no Q6_K prefill copy: {e}"),
            _ => {}
        }
    }
    let net = net;
    let mut e = Engine {
        net: &net,
        tok,
        cache,
        hist: Vec::new(),
        max_draft: o.max_draft,
        drafter,
        log: o.log,
        generated: o.generated,
    };
    let mut batch = pool.map(|(pool, n, slot_ctx)| {
        let policy = engine::ChunkPolicy {
            floor: o.prefill_chunk,
            ms: o.prefill_ms,
        };
        Batch::new(&net, pool, n, slot_ctx, policy, o.log, e.generated.clone())
    });
    let served = batch.as_ref().map_or(ctx, |b| b.ctx.min(ctx));
    let banner = match &batch {
        Some(b) => format!("{kvs}, ctx {served}{drafts}, {} requests at once", b.slots),
        None => format!("{kvs}, ctx {ctx}{drafts}, one request at a time"),
    };
    let identity = crate::identity::of(&o.path, &model, &net);
    if ready
        .send(Loaded {
            tok,
            template,
            ctx: served,
            max_token_bytes: tok.max_token_bytes(),
            banner,
            identity,
        })
        .is_err()
    {
        return;
    }
    let mut waiting: VecDeque<Job> = VecDeque::new();
    loop {
        if batch.as_ref().is_none_or(Batch::idle) && waiting.is_empty() {
            match jobs.recv_timeout(Duration::from_millis(100)) {
                Ok(job) => waiting.push_back(job),
                Err(mpsc::RecvTimeoutError::Timeout) if !stop.load(Ordering::Relaxed) => continue,
                Err(_) => return,
            }
        }
        let Some(b) = batch.as_mut() else {
            if let Some(job) = waiting.pop_front() {
                e.run(job, None);
            }
            continue;
        };
        // Take requests off the channel only as far as slots can start
        // them (and a few that can't, which wait to run alone): the bounded
        // channel stays the queue (full: a 503).
        waiting.retain(|j| !j.cancel.load(Ordering::Relaxed));
        loop {
            let fit = waiting.iter().filter(|j| fits_slot(j, b.ctx)).count();
            if fit > b.free() || waiting.len() > b.slots {
                break;
            }
            match jobs.try_recv() {
                Ok(job) => waiting.push_back(job),
                Err(_) => break,
            }
        }
        // Admit the first request that fits a slot (its prompt then fed a
        // chunk per step) if others run or wait; one alone runs the S1 way,
        // as does one that fits no slot, once the batch is idle.
        let first_fit = waiting.iter().position(|j| fits_slot(j, b.ctx));
        if let Some(i) = first_fit
            && b.has_free()
            && (!b.idle() || waiting.len() > 1)
        {
            let job = waiting.remove(i).unwrap();
            let out = text_out(e.tok, &job);
            b.admit(job, out);
        } else if b.idle() {
            if let Some(job) = waiting.pop_front() {
                let ho = Handover {
                    jobs: &jobs,
                    waiting: &mut waiting,
                    slot_ctx: Some(b.ctx),
                    room: b.slots,
                };
                if let Some(seq) = e.run(job, Some(ho)) {
                    let tx = seq.job.tx.clone();
                    if let Err(err) = or_exit("a hand-over", || b.adopt(seq, &e.cache)) {
                        fatal(b, &err, Some(tx));
                    }
                }
            }
            continue;
        }
        if let Err(err) = or_exit("a batched step", || b.step()) {
            fatal(b, &err, None);
        }
    }
}

/// A device error in the batch: every request in it (and `tx`'s) gets it,
/// then the process exits (CUDA errors are sticky; see `or_exit`).
fn fatal(b: &mut Batch, err: &str, tx: Option<mpsc::Sender<Event>>) -> ! {
    eprintln!("axon serve: the engine failed: {err}; exiting");
    b.fail_all(err);
    if let Some(tx) = tx {
        let _ = tx.send(Event::Error(err.to_string()));
    }
    std::process::exit(70);
}

/// The default context where the trained one isn't served: a CPU/GPU
/// split, a multi-GPU pipeline, --gpu-layers (a long cache there costs GPU
/// layers or the fit; `Backend::new_capped`).
const SPLIT_CTX: usize = crate::DEFAULT_CHAT_CTX;

/// The shortest slot the default trades context for: agent harnesses send
/// 10k-40k-token prompts, so below min(context, this) one request at a
/// time with the whole context wins over batching (`plan_slots`).
pub const AGENT_CTX: usize = 32768;

/// VRAM a pool leaves free beyond itself and the single-request cache (the
/// hand-over copy and other transient buffers).
const POOL_SPARE: usize = 256 << 20;

/// RAM a CPU server's default context leaves free beyond its KV cache.
const RAM_MARGIN: u64 = 1 << 30;

const GIB: f64 = (1u64 << 30) as f64;

/// The CPU's default context: `want` if its cache (`per_pos` bytes a
/// position) fits in `avail` bytes of RAM, else the most positions that do
/// (a multiple of 1024), but never less than the old default,
/// min(want, `SPLIT_CTX`) (the cache is mapped lazily: RAM goes as the
/// context fills).
fn ram_ctx(want: usize, per_pos: u64, avail: u64) -> usize {
    if (want as u64).saturating_mul(per_pos) <= avail {
        return want;
    }
    let c = (avail / per_pos.max(1)) as usize / 1024 * 1024;
    c.max(want.min(SPLIT_CTX))
}

/// Why the GPU's default context is `c`, not the trained `want`
/// (`all_layers`: --gpu-layers put every layer there).
fn gpu_fit_note(c: usize, want: usize, kv: KvType, all_layers: bool) -> String {
    format!(
        "{c} fit in GPU memory beside the weights, of the model's {want} \
         (--ctx N for another: {}{})",
        if all_layers {
            "a longer one doesn't fit with every layer on the GPU"
        } else {
            "a longer one splits the model across the CPU and GPU, slower"
        },
        if kv == KvType::Q8 {
            ""
        } else {
            "; --kv q8 halves the KV cache"
        }
    )
}

/// The CPU's default context and why, if shorter than the trained `want`:
/// `ram_ctx` of `avail` (MemAvailable; None: unknown, want) less the
/// mapped model's `mapped` bytes and `RAM_MARGIN`.
fn cpu_ctx(want: usize, per_pos: u64, mapped: u64, avail: Option<u64>) -> (usize, Option<String>) {
    let Some(a) = avail else {
        return (want, None);
    };
    let a = a.saturating_sub(mapped).saturating_sub(RAM_MARGIN);
    let c = ram_ctx(want, per_pos, a);
    (c, (c < want).then(|| ram_note(want, per_pos, a)))
}

/// Why the CPU's default context is shorter than the trained `want`.
fn ram_note(want: usize, per_pos: u64, avail: u64) -> String {
    format!(
        "a KV cache of the model's {want} tokens takes {:.1} GiB, {:.1} GiB \
         of RAM is free beside the model (--ctx N for another)",
        (want as u64 * per_pos) as f64 / GIB,
        avail as f64 / GIB
    )
}

/// Continuous batching's shape: `slots` requests at once (1: no pool) of
/// `ctx` positions each, and why, if it isn't what was asked for.
#[derive(Debug, PartialEq, Eq)]
struct SlotPlan {
    slots: usize,
    ctx: usize,
    note: Option<String>,
}

impl SlotPlan {
    fn one(ctx: usize, note: Option<String>) -> SlotPlan {
        SlotPlan {
            slots: 1,
            ctx,
            note,
        }
    }
}

/// The longest of `hi`, then the multiples of 1024 below it, down to `lo`,
/// for which `ok` holds (it only gets harder as the context grows).
fn longest_ctx(lo: usize, hi: usize, ok: impl Fn(usize) -> bool) -> Option<usize> {
    if ok(hi) {
        return Some(hi);
    }
    let mut c = (hi - 1) / 1024 * 1024;
    while c >= lo.max(1) {
        if ok(c) {
            return Some(c);
        }
        c -= 1024;
    }
    None
}

/// Slots of continuous batching for a model loaded with a `ctx`-token
/// context, `fits(s, c)`: whether s slots of c positions fit beside the
/// single-request cache of c positions. `ctx_set`: --ctx was given (`ctx`
/// is it); `parallel`: most slots (--parallel, default `PARALLEL`).
///
/// - --ctx N: the context is N, whatever --parallel says; the most slots
///   up to `parallel` that fit with it, at least 2, else one request at a
///   time.
/// - --parallel N alone: N slots, of the longest context they fit with
///   (down to 1024 positions), else one request at a time.
/// - Neither: context first, then slots. The whole context if 2 slots fit
///   with it, else the longest that does down to min(ctx, `AGENT_CTX`);
///   then the most slots (up to 8) at that context. If not even 2 fit at
///   min(ctx, AGENT_CTX), one request at a time with the whole context.
fn plan_slots(
    ctx: usize,
    ctx_set: bool,
    parallel: usize,
    parallel_set: bool,
    fits: impl Fn(usize, usize) -> bool,
) -> SlotPlan {
    if parallel <= 1 {
        return SlotPlan::one(ctx, None);
    }
    let most = |c: usize| (2..=parallel).rev().find(|&s| fits(s, c));
    if ctx_set {
        return match most(ctx) {
            Some(s) => SlotPlan {
                slots: s,
                ctx,
                note: (parallel_set && s < parallel).then(|| {
                    format!(
                        "{s} requests at once: {parallel} slots of --ctx {ctx} don't fit in GPU \
                         memory (a smaller --ctx or --kv q8 for more)"
                    )
                }),
            },
            None => SlotPlan::one(
                ctx,
                Some(format!(
                    "one request at a time: 2 slots of --ctx {ctx} don't fit in GPU memory (a \
                     smaller --ctx or --kv q8 to batch)"
                )),
            ),
        };
    }
    if parallel_set {
        return match longest_ctx(ctx.min(1024), ctx, |c| fits(parallel, c)) {
            Some(c) => SlotPlan {
                slots: parallel,
                ctx: c,
                note: (c < ctx).then(|| {
                    format!(
                        "context {c} tokens: {parallel} slots (--parallel) of {ctx} don't fit \
                         in GPU memory (fewer --parallel or --kv q8 for more)"
                    )
                }),
            },
            None => SlotPlan::one(
                ctx,
                Some(format!(
                    "one request at a time: {parallel} slots (--parallel) don't fit in GPU \
                     memory even at 1024 positions"
                )),
            ),
        };
    }
    let floor = ctx.min(AGENT_CTX);
    match longest_ctx(floor, ctx, |c| fits(2, c)) {
        Some(c) => SlotPlan {
            slots: most(c).unwrap_or(2),
            ctx: c,
            note: (c < ctx).then(|| {
                format!(
                    "context {c} tokens: the longest that 2 requests at once fit with in GPU \
                     memory, of {ctx} (--parallel 1 for {ctx}; --ctx N or --parallel N to choose)"
                )
            }),
        },
        None => {
            // A pool of N slots is N + 1 caches (the single-request one).
            let at = match longest_ctx(ctx.min(1024), ctx, |c| fits(2, c)) {
                Some(c2) => format!("2 at once fit at {c2}: --ctx {c2}"),
                None => "2 at once don't fit even at 1024".into(),
            };
            SlotPlan::one(
                ctx,
                Some(format!(
                    "one request at a time, to keep the {ctx}-token context (2 slots and the \
                     single-request cache are 3 caches); {at}"
                )),
            )
        }
    }
}

struct EngineOpts {
    /// Requests decoded together at most (`--parallel`); 1: one at a time.
    parallel: usize,
    parallel_set: bool,
    slot_ctx: Option<usize>,
    prefill_chunk: usize,
    prefill_ms: usize,
    path: String,
    device: Device,
    gpu_layers: Option<usize>,
    kv: KvType,
    ctx: Option<usize>,
    max_draft: usize,
    draft_model: Option<String>,
    log: bool,
    generated: Arc<AtomicU64>,
    rope: crate::rope::RopeArgs,
}

/// A running server. Dropping it (or `shutdown`) stops it: the acceptor
/// and engine threads end (the engine after its current request) and the
/// model is freed; open connections finish what they are doing.
pub struct Server {
    #[cfg_attr(not(test), allow(dead_code))]
    pub addr: SocketAddr,
    accept: Option<std::thread::JoinHandle<()>>,
    engine: Option<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    /// Tokens generated so far, all requests (the tests watch it).
    #[cfg_attr(not(test), allow(dead_code))]
    pub generated: Arc<AtomicU64>,
    /// Completion requests handed to the engine's queue so far (the tests
    /// wait on it instead of sleeping).
    #[cfg_attr(not(test), allow(dead_code))]
    pub queued: Arc<AtomicU64>,
    #[cfg(test)]
    sh: Arc<Shared>,
}

#[cfg(test)]
impl Server {
    /// Connections waiting for (or receiving) a request (`Shared::waiting`).
    pub fn waiting(&self) -> usize {
        self.sh.waiting.len()
    }

    /// Open connections.
    pub fn conns(&self) -> usize {
        self.sh.conns.load(Ordering::Relaxed)
    }
}

impl Server {
    /// Serve until the acceptor ends (never, short of `shutdown` elsewhere).
    pub fn wait(mut self) -> io::Result<()> {
        if let Some(a) = self.accept.take() {
            a.join()
                .map_err(|_| io::Error::other("the acceptor thread failed"))?;
        }
        Ok(())
    }

    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(a) = self.accept.take() {
            // Wake the blocking accept.
            let _ = TcpStream::connect(self.addr);
            let _ = a.join();
        }
        if let Some(e) = self.engine.take() {
            let _ = e.join();
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Bind, start accepting (503 until the model is loaded), load the model
/// on the engine thread; returns once it is loaded.
pub fn start(o: Opts) -> io::Result<Server> {
    let listener = TcpListener::bind((o.host.as_str(), o.port))?;
    let addr = listener.local_addr()?;
    let (jobs_tx, jobs_rx) = mpsc::sync_channel(o.max_queue);
    let (ready_tx, ready_rx) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let generated = Arc::new(AtomicU64::new(0));
    let queued = Arc::new(AtomicU64::new(0));
    let eo = EngineOpts {
        parallel: o.parallel,
        parallel_set: o.parallel_set,
        slot_ctx: o.slot_ctx,
        prefill_chunk: o.prefill_chunk,
        prefill_ms: o.prefill_ms,
        generated: generated.clone(),
        path: o.path.clone(),
        device: o.device,
        gpu_layers: o.gpu_layers,
        kv: o.kv,
        ctx: o.ctx,
        max_draft: o.max_draft,
        draft_model: o.draft_model.clone(),
        log: o.log,
        rope: o.rope.clone(),
    };
    let stop2 = stop.clone();
    let engine = std::thread::Builder::new()
        .name("axon-engine".into())
        .spawn(move || engine(eo, ready_tx, jobs_rx, stop2))?;
    let model_id = std::path::Path::new(&o.path)
        .file_stem()
        .map_or(o.path.clone(), |s| s.to_string_lossy().into_owned());
    // A request's own repetition_penalty, else the server's flag, else 1
    // (off, as OpenAI and llama-server; `axon chat`'s 1.1 would quietly
    // penalize the punctuation and indentation of code and JSON).
    let mut sample = o.sample;
    sample.repeat_penalty = sample.repeat_penalty.or(Some(1.0));
    let extra = o.cors.as_ref().map_or(String::new(), |origin| {
        let vary = if origin == "*" {
            ""
        } else {
            "Vary: Origin\r\n"
        };
        format!("Access-Control-Allow-Origin: {origin}\r\n{vary}")
    });
    let sh = Arc::new(Shared {
        model: OnceLock::new(),
        model_id: model_id.clone(),
        alias: o.alias,
        no_think: o.no_think,
        think_inline: o.think_inline,
        sample,
        max_tokens: o.max_tokens,
        extra,
        cors: o.cors.is_some(),
        max_conns: o.max_conns,
        created: unix_now(),
        jobs: jobs_tx,
        conns: AtomicUsize::new(0),
        next_id: AtomicU64::new(1),
        stop: stop.clone(),
        waiting: Default::default(),
        next_conn: AtomicU64::new(0),
        queued: queued.clone(),
    });
    let sh2 = sh.clone();
    let accept = std::thread::Builder::new()
        .name("axon-accept".into())
        .spawn(move || accept_loop(listener, sh2))?;
    let mut server = Server {
        addr,
        accept: Some(accept),
        engine: Some(engine),
        stop,
        generated,
        queued,
        #[cfg(test)]
        sh: sh.clone(),
    };
    let Ok(mut l) = ready_rx.recv() else {
        server.shutdown();
        return Err(io::Error::other("the engine failed to start"));
    };
    if o.no_think && !l.template.as_ref().is_some_and(Template::thinks) {
        crate::fail(&format!(
            "--no-think: {} has no <think> token (not a thinking model)",
            o.path
        ));
    }
    let serve = crate::identity::serve_field(o.no_think, o.think_inline, &o.sample);
    l.identity = l.identity.with_serve(serve);
    if o.log {
        eprintln!(
            "axon serve: {model_id} ({}) on http://{addr}/v1{}",
            l.banner,
            if l.template.is_none() {
                " (no chat tokens: /v1/completions only)"
            } else {
                ""
            }
        );
        eprintln!("axon serve: identity {}", l.identity.line());
    }
    let _ = sh.model.set(l);
    Ok(server)
}

fn accept_loop(listener: TcpListener, sh: Arc<Shared>) {
    for s in listener.incoming() {
        if sh.stop.load(Ordering::Relaxed) {
            return;
        }
        let Ok(mut s) = s else {
            // EMFILE and the like: don't spin.
            std::thread::sleep(Duration::from_millis(10));
            continue;
        };
        if sh.conns.load(Ordering::Relaxed) >= sh.max_conns && !sh.waiting.evict_oldest() {
            let _ = s.set_write_timeout(Some(Duration::from_secs(1)));
            let e = ApiError::new(503, "server_error", "too many connections");
            {
                let mut c = Conn::new(&mut s);
                c.extra = sh.extra.clone();
                let _ = c.respond(503, "application/json", e.json().as_bytes(), false, &[]);
            }
            // What the client already sent, unread at close, would make
            // it a reset that can destroy the 503 before the client reads it.
            let _ = s.shutdown(std::net::Shutdown::Write);
            // Only what has already arrived, and at most ACCEPT_DRAIN
            // bytes: this is the one accept thread.
            if s.set_nonblocking(true).is_ok() {
                let (mut b, mut n) = ([0u8; 4096], 0);
                while n < ACCEPT_DRAIN {
                    match io::Read::read(&mut s, &mut b) {
                        Ok(k) if k > 0 => n += k,
                        _ => break,
                    }
                }
            }
            continue;
        }
        sh.conns.fetch_add(1, Ordering::Relaxed);
        let sh2 = sh.clone();
        let spawned = std::thread::Builder::new()
            .name("axon-conn".into())
            .spawn(move || {
                let _guard = ConnGuard(&sh2.conns);
                connection(s, &sh2);
            });
        if spawned.is_err() {
            sh.conns.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

fn connection(s: TcpStream, sh: &Shared) {
    // SSE chunks are small: send each at once.
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(IDLE));
    let _ = s.set_write_timeout(Some(WRITE_TIMEOUT));
    let id = sh.next_conn.fetch_add(1, Ordering::Relaxed);
    let handle = s.try_clone().ok().map(Arc::new);
    let mut c = Conn::new(s);
    c.head_timeout = IDLE;
    c.extra = sh.extra.clone();
    loop {
        sh.waiting.set(id, handle.as_ref());
        let r = c.read_request();
        if !sh.waiting.set(id, None) && handle.is_some() {
            // Evicted (`Shared::waiting`): the socket is shut down.
            return;
        }
        match r {
            Ok(None) => return,
            Err(e) => {
                let body = ApiError::new(e.status, "invalid_request_error", e.msg).json();
                let _ = c.respond(e.status, "application/json", body.as_bytes(), false, &[]);
                // Evictable while it lingers: it serves nothing more.
                sh.waiting.set(id, handle.as_ref());
                linger(&c.s);
                sh.waiting.set(id, None);
                return;
            }
            Ok(Some(req)) => {
                c.head_only = req.method == "HEAD";
                match route(&mut c, &req, sh) {
                    Ok(true) if req.keep_alive => {}
                    _ => return,
                }
            }
        }
    }
}

/// Before closing after an error answered mid-request (413, 431, ...):
/// stop sending, then read and drop what the client is still sending, for
/// up to `LINGER` or `LINGER_BYTES`. Closing with its bytes unread would
/// reset the connection, and a reset can destroy the error response
/// before the client reads it (it sees ECONNRESET instead of the 413).
fn linger(s: &TcpStream) {
    let _ = s.shutdown(std::net::Shutdown::Write);
    let t = Instant::now();
    let mut b = [0u8; 16384];
    let mut n = 0;
    while t.elapsed() < LINGER && n < LINGER_BYTES {
        let left = LINGER
            .saturating_sub(t.elapsed())
            .max(Duration::from_millis(1));
        if s.set_read_timeout(Some(left)).is_err() {
            return;
        }
        match io::Read::read(&mut &*s, &mut b) {
            Ok(0) | Err(_) => return,
            Ok(k) => n += k,
        }
    }
}

fn json_reply<S: io::Read + Write>(
    c: &mut Conn<S>,
    status: u16,
    body: &str,
    keep: bool,
) -> io::Result<bool> {
    c.respond(status, "application/json", body.as_bytes(), keep, &[])?;
    Ok(keep)
}

fn api_err<S: io::Read + Write>(c: &mut Conn<S>, e: ApiError, keep: bool) -> io::Result<bool> {
    json_reply(c, e.status, &e.json(), keep)
}

/// Answer one request; Ok(true) if the connection can take another.
fn route(c: &mut Conn<TcpStream>, req: &Request, sh: &Shared) -> io::Result<bool> {
    let keep = req.keep_alive;
    let methods = match req.path.as_str() {
        "/v1/models" | "/health" | "/identity" => "GET, HEAD, OPTIONS",
        "/v1/chat/completions" | "/v1/completions" => "POST, OPTIONS",
        _ => "",
    };
    let loading = || {
        ApiError::new(
            503,
            "server_error",
            "the model is loading; try again shortly",
        )
    };
    match (req.method.as_str(), req.path.as_str()) {
        (_, _) if methods.is_empty() => api_err(
            c,
            ApiError::new(
                404,
                "invalid_request_error",
                format!("no route {}", req.path),
            ),
            keep,
        ),
        ("OPTIONS", _) => {
            // A CORS preflight (when --cors is on), or a plain OPTIONS.
            let mut h = vec![("Allow", methods)];
            if sh.cors {
                h.extend([
                    ("Access-Control-Allow-Methods", methods),
                    (
                        "Access-Control-Allow-Headers",
                        "Content-Type, Authorization",
                    ),
                    ("Access-Control-Max-Age", "86400"),
                ]);
            }
            c.respond(204, "", b"", keep, &h)?;
            Ok(keep)
        }
        ("GET" | "HEAD", "/v1/models") => {
            let Some(m) = sh.model.get() else {
                return api_err(c, loading(), keep);
            };
            let meta = serde_json::json!({"n_ctx": m.ctx, "identity": m.identity.json()});
            let mut data = vec![serde_json::json!({
                "id": sh.model_id, "object": "model", "created": sh.created, "owned_by": "axon",
                "meta": meta,
            })];
            if let Some(a) = &sh.alias {
                data.push(serde_json::json!({
                    "id": a, "object": "model", "created": sh.created, "owned_by": "axon",
                    "meta": meta,
                }));
            }
            let body = serde_json::json!({"object": "list", "data": data}).to_string();
            json_reply(c, 200, &body, keep)
        }
        ("GET" | "HEAD", "/identity") => match sh.model.get() {
            Some(m) => json_reply(c, 200, &m.identity.json().to_string(), keep),
            None => api_err(c, loading(), keep),
        },
        ("GET" | "HEAD", "/health") => match sh.model.get() {
            Some(_) => json_reply(c, 200, "{\"status\":\"ok\"}", keep),
            None => json_reply(c, 503, "{\"status\":\"loading\"}", keep),
        },
        ("POST", "/v1/chat/completions") => completion(c, req, sh, true),
        ("POST", "/v1/completions") => completion(c, req, sh, false),
        (_, _) => {
            let e = ApiError::new(405, "invalid_request_error", format!("use {methods}"));
            c.respond(
                405,
                "application/json",
                e.json().as_bytes(),
                keep,
                &[("Allow", methods)],
            )?;
            Ok(keep)
        }
    }
}

/// /v1/chat/completions (`chat`) and /v1/completions.
fn completion(c: &mut Conn<TcpStream>, req: &Request, sh: &Shared, chat: bool) -> io::Result<bool> {
    let keep = req.keep_alive;
    let Some(m) = sh.model.get() else {
        return api_err(
            c,
            ApiError::new(
                503,
                "server_error",
                "the model is loading; try again shortly",
            ),
            keep,
        );
    };
    let created = unix_now();
    let g = match api::parse(&req.body, chat) {
        Ok(g) => g,
        Err(e) => return api_err(c, e, keep),
    };
    if let Some(m) = &g.model
        && !sh.serves(m)
    {
        let mut e = ApiError::new(
            404,
            "invalid_request_error",
            format!("model '{m}' not found; this server has '{}'", sh.model_id),
        );
        e.param = Some("model");
        e.code = Some("model_not_found");
        return api_err(c, e, keep);
    }
    // More bytes than the context's worth of the longest token: more
    // tokens than the context, without spending the time to encode them.
    let bytes = match &g.prompt {
        Prompt::Chat { messages, .. } => messages.iter().map(|m| m.content.len()).sum(),
        Prompt::Text(s) => s.len(),
    } + g.tools.as_ref().map_or(0, tools::ToolReq::extra_bytes);
    let too_long = |n: String| {
        let mut e = ApiError::bad(
            if chat { "messages" } else { "prompt" },
            format!("the prompt is {n}; the context is {} tokens (--ctx)", m.ctx),
        );
        e.code = Some("context_length_exceeded");
        e
    };
    if bytes > m.ctx.saturating_mul(m.max_token_bytes) {
        return api_err(c, too_long(format!("{bytes} bytes")), keep);
    }
    let mut parser = None;
    let (prompt, stop_ids, think) = match &g.prompt {
        Prompt::Chat { messages, no_think } => {
            let Some(t) = &m.template else {
                return api_err(
                    c,
                    ApiError::new(
                        400,
                        "invalid_request_error",
                        "this model has no chat tokens (ChatML, Llama 3, Mistral or Gemma); use \
                         /v1/completions",
                    ),
                    keep,
                );
            };
            let no_think = no_think.unwrap_or(sh.no_think) && t.thinks();
            let think = t
                .think
                .as_ref()
                .filter(|_| !no_think && !sh.think_inline)
                .map(|&(o, c, _)| (o, c));
            match &g.tools {
                None => (
                    t.render(m.tok, messages, no_think),
                    t.stop().to_vec(),
                    think,
                ),
                Some(tr) => match tools::render(t, m.tok, tr, no_think) {
                    Ok(r) => {
                        parser = r.parser;
                        let think = think.filter(|_| !r.no_reasoning);
                        (r.ids, t.stop().to_vec(), think)
                    }
                    Err(e) => return api_err(c, e, keep),
                },
            }
        }
        // As `axon gen`: BOS if the model adds one, its end-of-generation set.
        Prompt::Text(s) => (m.tok.encode_prompt(s), m.tok.eog(), None),
    };
    if prompt.is_empty() {
        return api_err(c, ApiError::bad("prompt", "the prompt is empty"), keep);
    }
    if prompt.len() >= m.ctx {
        return api_err(c, too_long(format!("{} tokens", prompt.len())), keep);
    }
    let n = sh.next_id.fetch_add(1, Ordering::Relaxed);
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let job = Job {
        n_gen: g
            .max_tokens
            .unwrap_or(usize::MAX)
            .min(sh.max_tokens.unwrap_or(usize::MAX))
            .min(m.ctx - prompt.len()),
        prompt,
        sample: api::sampling(&g.sample, &sh.sample, crate::sample::DEFAULT_TEMP),
        stop_ids,
        stops: g.stop.clone(),
        think,
        cancel: cancel.clone(),
        tx,
        id: n,
    };
    match sh.jobs.try_send(job) {
        Ok(()) => {
            sh.queued.fetch_add(1, Ordering::Relaxed);
        }
        Err(mpsc::TrySendError::Full(_)) => {
            return api_err(
                c,
                ApiError::new(503, "server_error", "too many queued requests"),
                keep,
            );
        }
        Err(mpsc::TrySendError::Disconnected(_)) => {
            return api_err(
                c,
                ApiError::new(503, "server_error", "the server is shutting down"),
                false,
            );
        }
    }
    let id = format!(
        "{}-{n:x}{:x}",
        if chat { "chatcmpl" } else { "cmpl" },
        sh.created
    );
    let model = g.model.as_deref().unwrap_or(&sh.model_id);
    let model = (model, m.identity.fingerprint());
    let r = if g.stream {
        stream(c, req, &g, &rx, chat, &id, model, created, parser)
    } else {
        whole(c, keep, &rx, chat, &id, model, created, parser)
    };
    if !matches!(r, Ok(true)) {
        cancel.store(true, Ordering::Relaxed);
    }
    r
}

/// The next event, checking every `POLL` that the client is still there
/// (`checked`: when it last was), also while events keep coming: a whole
/// (non-streamed) response writes nothing until the end, so before, a
/// client that left mid-generation went unnoticed and the engine ran on
/// to max_tokens or the context's end (minutes at --ctx 32768) while
/// every later request waited. None: it isn't there (or the engine is gone).
fn next_event(
    c: &Conn<TcpStream>,
    rx: &mpsc::Receiver<Event>,
    checked: &mut Instant,
) -> Option<Event> {
    loop {
        if checked.elapsed() >= POLL {
            *checked = Instant::now();
            if c.s.closed() {
                return None;
            }
        }
        match rx.recv_timeout(POLL) {
            Ok(e) => return Some(e),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn whole(
    c: &mut Conn<TcpStream>,
    keep: bool,
    rx: &mpsc::Receiver<Event>,
    chat: bool,
    id: &str,
    // The model's name and the fingerprint.
    model: (&str, String),
    created: u64,
    parser: Option<tools::Parser>,
) -> io::Result<bool> {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut checked = Instant::now();
    loop {
        match next_event(c, rx, &mut checked) {
            None => return Ok(false),
            Some(Event::Text {
                content: t,
                reasoning: r,
            }) => {
                content.push_str(&t);
                reasoning.push_str(&r);
            }
            Some(Event::Error(m)) => {
                return api_err(c, ApiError::new(500, "server_error", m), keep);
            }
            Some(Event::Done { usage, finish }) => {
                if let Some(mut p) = parser {
                    // Tools: the reply's text into content and calls.
                    let (mut text, mut calls) = (String::new(), Vec::new());
                    for piece in p.push(&content).into_iter().chain(p.finish()) {
                        match piece {
                            tools::Piece::Text(t) => text.push_str(&t),
                            tools::Piece::Call(c) => calls.push(c),
                        }
                    }
                    let h = api::head(id, "chat.completion", created, model.0, &model.1);
                    let body = if calls.is_empty() {
                        api::chat_response(&h, &text, &reasoning, finish, &usage)
                    } else {
                        let finish = if finish == "stop" {
                            "tool_calls"
                        } else {
                            finish
                        };
                        tools::chat_response(&h, &text, &reasoning, &calls, finish, &usage)
                    };
                    return json_reply(c, 200, &body, keep);
                }
                let body = if chat {
                    let h = api::head(id, "chat.completion", created, model.0, &model.1);
                    api::chat_response(&h, &content, &reasoning, finish, &usage)
                } else {
                    let h = api::head(id, "text_completion", created, model.0, &model.1);
                    api::text_response(&h, &content, finish, &usage)
                };
                return json_reply(c, 200, &body, keep);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stream(
    c: &mut Conn<TcpStream>,
    req: &Request,
    g: &api::GenReq,
    rx: &mpsc::Receiver<Event>,
    chat: bool,
    id: &str,
    // The model's name and the fingerprint.
    model: (&str, String),
    created: u64,
    mut parser: Option<tools::Parser>,
) -> io::Result<bool> {
    let keep = req.keep_alive && req.minor == 1;
    c.start_stream(200, "text/event-stream", req.minor, keep)?;
    let h = api::head(
        id,
        if chat {
            "chat.completion.chunk"
        } else {
            "text_completion"
        },
        created,
        model.0,
        &model.1,
    );
    // With include_usage every chunk has `"usage": null` but the last (OpenAI's shape).
    let u = |j: String| {
        if g.include_usage {
            api::null_usage(j)
        } else {
            j
        }
    };
    if chat {
        c.send(api::sse(&u(api::chat_chunk(&h, true, "", "", None))).as_bytes())?;
    }
    let mut buf = String::new();
    // Tools: the content and calls the parser has decided, as chunks.
    let mut ncalls = 0;
    let mut pieces = |buf: &mut String, ps: Vec<tools::Piece>| {
        for p in ps {
            match p {
                tools::Piece::Text(t) => {
                    buf.push_str(&api::sse(&u(api::chat_chunk(&h, false, &t, "", None))));
                }
                tools::Piece::Call(call) => {
                    for j in tools::call_chunks(&h, ncalls, &call) {
                        buf.push_str(&api::sse(&u(j)));
                    }
                    ncalls += 1;
                }
            }
        }
    };
    let mut checked = Instant::now();
    let mut ev = next_event(c, rx, &mut checked);
    loop {
        // Everything already there goes out in one write.
        let mut done = false;
        while let Some(e) = ev {
            match e {
                Event::Text { content, reasoning } if parser.is_some() => {
                    if !reasoning.is_empty() {
                        let j = api::chat_chunk(&h, false, "", &reasoning, None);
                        buf.push_str(&api::sse(&u(j)));
                    }
                    if let Some(p) = parser.as_mut() {
                        pieces(&mut buf, p.push(&content));
                    }
                }
                Event::Text { content, reasoning } => {
                    let j = if chat {
                        api::chat_chunk(&h, false, &content, &reasoning, None)
                    } else {
                        api::text_chunk(&h, &content, None)
                    };
                    buf.push_str(&api::sse(&u(j)));
                }
                Event::Error(m) => {
                    let e = ApiError::new(500, "server_error", m);
                    buf.push_str(&api::sse(&e.json()));
                    buf.push_str(api::SSE_DONE);
                    done = true;
                }
                Event::Done { usage, mut finish } => {
                    if let Some(p) = parser.as_mut() {
                        pieces(&mut buf, p.finish());
                        if p.calls() > 0 && finish == "stop" {
                            finish = "tool_calls";
                        }
                    }
                    let j = if chat {
                        api::chat_chunk(&h, false, "", "", Some(finish))
                    } else {
                        api::text_chunk(&h, "", Some(finish))
                    };
                    buf.push_str(&api::sse(&u(j)));
                    if g.include_usage {
                        buf.push_str(&api::sse(&api::usage_chunk(&h, &usage)));
                    }
                    buf.push_str(api::SSE_DONE);
                    done = true;
                }
            }
            if done {
                break;
            }
            ev = rx.try_recv().ok();
        }
        c.send(buf.as_bytes())?;
        buf.clear();
        if done {
            c.end_stream()?;
            return Ok(keep);
        }
        ev = next_event(c, rx, &mut checked);
        if ev.is_none() {
            return Ok(false);
        }
    }
}

const SERVE_USAGE: &str = "usage: axon serve <model.gguf> [--host H] [--port P] [--ctx N] [--device D] \
     [--gpu-layers N] [--kv T] [--draft K] [--draft-model PATH] [--alias NAME] [--max-tokens N] \
     [--cors [ORIGIN]] [--max-conns N] [--max-queue N] [--parallel N] [--prefill-chunk N] \
     [--prefill-ms MS] [--no-think] [--think-inline] [--quiet] [sampling flags]";

/// `serve` subcommand: args = [axon, serve, model, flags...]
pub fn run(args: &[String]) -> io::Result<()> {
    let Some(path) = args.get(2).filter(|p| !p.starts_with('-')) else {
        crate::fail(SERVE_USAGE);
    };
    let mut o = Opts {
        path: path.clone(),
        host: "127.0.0.1".into(),
        port: 8080,
        device: crate::device_env(),
        gpu_layers: None,
        kv: crate::kv_env(),
        ctx: None,
        max_draft: 0,
        draft_model: crate::draft::model_env(),
        sample: SampleArgs::default(),
        no_think: false,
        think_inline: false,
        alias: None,
        log: true,
        max_tokens: None,
        cors: None,
        max_conns: MAX_CONNS,
        max_queue: MAX_QUEUE,
        parallel: PARALLEL,
        parallel_set: false,
        slot_ctx: None,
        prefill_chunk: engine::PREFILL_CHUNK,
        prefill_ms: engine::PREFILL_MS,
        rope: Default::default(),
    };
    let mut k = None;
    let mut i = 3;
    while i < args.len() {
        let f = args[i].as_str();
        match f {
            "--no-think" | "--think-inline" | "--quiet" => {
                match f {
                    "--no-think" => o.no_think = true,
                    "--think-inline" => o.think_inline = true,
                    _ => o.log = false,
                }
                i += 1;
                continue;
            }
            "--cors" => {
                // The origin is optional: `--cors` alone allows any (*).
                match args.get(i + 1).filter(|v| !v.starts_with("--")) {
                    Some(v) => {
                        o.cors = Some(v.clone());
                        i += 2;
                    }
                    None => {
                        o.cors = Some("*".into());
                        i += 1;
                    }
                }
                continue;
            }
            _ if SampleArgs::is_flag(f) => {
                o.sample
                    .set(f, args.get(i + 1).map(String::as_str))
                    .unwrap_or_else(|e| crate::fail(&e));
                i += 2;
                continue;
            }
            _ if crate::rope::RopeArgs::is_flag(f) => {
                o.rope
                    .set(f, args.get(i + 1).map(String::as_str))
                    .unwrap_or_else(|e| crate::fail(&e));
                i += 2;
                continue;
            }
            "--host" | "--port" | "--ctx" | "--draft" | "--draft-model" | "--device"
            | "--gpu-layers" | "--kv" | "--alias" | "--max-tokens" | "--max-conns"
            | "--max-queue" | "--parallel" | "--prefill-chunk" | "--prefill-ms" => {}
            _ => crate::fail(&format!("unknown argument '{f}' for serve")),
        }
        let Some(v) = args.get(i + 1) else {
            crate::fail(&format!("{f} needs a value"));
        };
        let whole = |pos: bool| {
            v.parse::<usize>()
                .ok()
                .filter(|&n| n > 0 || !pos)
                .unwrap_or_else(|| crate::fail(&format!("{f} needs a whole number, got '{v}'")))
        };
        match f {
            "--host" => o.host = v.clone(),
            "--alias" => o.alias = Some(v.clone()),
            "--draft-model" if v.is_empty() => crate::fail("--draft-model needs a path"),
            "--draft-model" => o.draft_model = Some(v.clone()),
            "--device" => o.device = Device::parse(v).unwrap_or_else(|e| crate::fail(&e)),
            "--kv" => o.kv = crate::kv_flag(Some(v)),
            "--port" => {
                o.port = v.parse::<u16>().unwrap_or_else(|_| {
                    crate::fail(&format!("--port needs a port number, got '{v}'"))
                })
            }
            "--ctx" => o.ctx = Some(whole(true)),
            "--gpu-layers" => o.gpu_layers = Some(whole(false)),
            "--max-tokens" => o.max_tokens = Some(whole(true)),
            "--max-conns" => o.max_conns = whole(true),
            "--max-queue" => o.max_queue = whole(true),
            "--prefill-chunk" => o.prefill_chunk = whole(false),
            "--prefill-ms" => o.prefill_ms = whole(false),
            "--parallel" => {
                o.parallel = whole(true);
                o.parallel_set = true;
            }
            _ => k = Some(whole(false)),
        }
        i += 2;
    }
    o.max_draft = crate::draft::draft_k(k.or_else(crate::draft_env_set), o.draft_model.is_some());
    crate::check_stream_decode(o.max_draft);
    start(o)?.wait()
}

#[cfg(test)]
mod tests;
