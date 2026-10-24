//! Draft-model speculative decoding: a second, smaller model of the same
//! vocabulary (`--draft-model small.gguf`) guesses the next tokens greedily,
//! and the gen loop verifies them in one batched pass of the main model, as
//! it does prompt lookup's (spec.rs). The output doesn't depend on the
//! guesses, so with or without a drafter it's the same, greedy or sampled with
//! the same seed. Only the speed changes.
//!
//! `ModelDrafter` keeps its own KV cache and remembers which tokens it holds
//! (`cached`). Each call feeds whatever part of the history it doesn't
//! already have, after the longest common prefix. That covers the tokens the
//! main model emitted, guesses it rejected (overwritten in place: positions
//! are absolute) and histories the caller rewrote (chat forgetting its oldest
//! exchange, serve switching requests) with no reset hook.

use crate::cuda::qwen2::{CacheKv, CudaQwen2};
use crate::cuda::{DevBuf, Gpu, Kernels, Module};
use crate::gguf::{Model, Val};
use crate::kv::KvType;
use crate::lm::{Backend, Device, Inner, Lm};
use crate::tokenizer::Tokenizer;
use std::time::Instant;

/// Draft tokens per pass when a draft model is given and --draft isn't.
/// 3: the best K of the go/no-go A/B on 7B, Qwen3-8B and Llama-3.1-8B
/// greedy (tg128 medians +37/+44/+63 % at K=3; K=5 +32/+30/+56 %).
pub const DEFAULT_MODEL_DRAFT: usize = 3;

/// New history tokens up to this many are caught up one `forward_argmax` at a
/// time (a CUDA graph step each) instead of one `feed_batch`. Usually 0 or 1
/// are new: the main model's correction token, which the first draft step
/// runs anyway, and the last guess of a fully accepted draft, which no draft
/// step ran.
const STEP_CATCH_UP: usize = 2;

/// Guesses the next tokens with a model of its own (see the module doc).
pub struct ModelDrafter<M: Lm> {
    net: M,
    cache: M::Cache,
    /// Positions the cache has room for.
    ctx: usize,
    /// The tokens whose rows the cache holds, from position 0.
    cached: Vec<u32>,
    /// Drafting stops before any of these (the main model's EOG set: a guess
    /// of one would only be verified by a row that is never used).
    stop: Vec<u32>,
    /// What each draft call cost: (calls, model steps, catch-up tokens, s).
    pub stats: DraftStats,
}

#[derive(Default, Clone, Copy)]
pub struct DraftStats {
    pub calls: usize,
    pub steps: usize,
    pub caught_up: usize,
    pub secs: f64,
}

impl<M: Lm> ModelDrafter<M> {
    /// A drafter running `net`, with a cache of `ctx` positions (the main
    /// model's context: drafts never go past it), stopping before `stop`.
    pub fn new(net: M, ctx: usize, stop: &[u32]) -> ModelDrafter<M> {
        let cache = net.new_cache(ctx);
        ModelDrafter {
            net,
            cache,
            ctx,
            cached: Vec::new(),
            stop: stop.to_vec(),
            stats: DraftStats::default(),
        }
    }

    /// Up to `max` greedy guesses of the tokens after `hist`, stopping before
    /// a stop token. Empty if `max` is 0 or the guesses would pass the
    /// context.
    pub fn draft(&mut self, hist: &[u32], max: usize) -> Vec<u32> {
        let n = hist.len();
        // Guess i is the drafter's pick after the token at position n - 1 + i.
        if max == 0 || n == 0 || n - 1 + max > self.ctx {
            return Vec::new();
        }
        let t = Instant::now();
        let last = n - 1;
        let common = self
            .cached
            .iter()
            .zip(&hist[..last])
            .take_while(|(a, b)| a == b)
            .count();
        self.cached.truncate(common);
        let vocab = self.net.vocab();
        // A main-model id past the drafter's vocabulary (padding rows of a
        // bigger embedding; the token tables match up to the drafter's
        // size): the drafter never sees it, it gets token 0 there. Its
        // guesses may suffer, never the output.
        let id = |t: u32| if (t as usize) < vocab { t } else { 0 };
        let gap: Vec<u32> = hist[common..last].iter().map(|&t| id(t)).collect();
        if gap.len() <= STEP_CATCH_UP {
            for (i, &t) in gap.iter().enumerate() {
                self.net.forward_argmax(t, common + i, &mut self.cache);
            }
        } else {
            let pb = self.net.prefill_batch();
            for (i, b) in gap.chunks(pb).enumerate() {
                self.net.feed_batch(b, common + i * pb, &mut self.cache);
            }
        }
        self.stats.caught_up += gap.len();
        self.cached.extend_from_slice(&hist[common..last]);
        let mut out = Vec::with_capacity(max);
        let mut tok = hist[last];
        for i in 0..max {
            let p = self.net.forward_argmax(id(tok), last + i, &mut self.cache);
            self.cached.push(tok);
            self.stats.steps += 1;
            if self.stop.contains(&p) {
                break;
            }
            out.push(p);
            tok = p;
        }
        self.stats.calls += 1;
        self.stats.secs += t.elapsed().as_secs_f64();
        out
    }
}

/// A source of guesses for the gen loop: `draft(history, max)` proposes up
/// to `max` tokens to follow `history`.
pub trait Draft {
    fn draft(&mut self, hist: &[u32], max: usize) -> Vec<u32>;
}

impl<M: Lm> Draft for ModelDrafter<M> {
    fn draft(&mut self, hist: &[u32], max: usize) -> Vec<u32> {
        ModelDrafter::draft(self, hist, max)
    }
}

/// Where the gen loop's guesses come from: prompt lookup (spec.rs), or a
/// draft model.
pub enum Drafter<'a> {
    Lookup(crate::spec::PromptLookup),
    Model(&'a mut dyn Draft),
}

impl Drafter<'_> {
    pub fn draft(&mut self, hist: &[u32], max: usize) -> Vec<u32> {
        match self {
            Drafter::Lookup(l) => {
                l.sync(hist);
                l.draft(max)
            }
            Drafter::Model(d) => d.draft(hist, max),
        }
    }
}

/// $AXON_DRAFT_MODEL: --draft-model's default.
pub fn model_env() -> Option<String> {
    std::env::var("AXON_DRAFT_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
}

/// --draft K as given (None: the default), adjusted: with a draft model,
/// `DEFAULT_MODEL_DRAFT` and K as is (each guess costs a drafter step, so
/// padding K up to fill the CPU's slot groups doesn't come free; `round_draft`
/// would also take 5 down to 3); else `crate::round_draft` of K (or of
/// AXON_DRAFT's default).
pub fn draft_k(k: Option<usize>, with_model: bool) -> usize {
    match (k, with_model) {
        (Some(k), true) => k,
        (None, true) => DEFAULT_MODEL_DRAFT,
        (k, false) => crate::round_draft(k.unwrap_or_else(crate::draft_env)),
    }
}

/// Draft tokens per pass with a draft model next to a model split across
/// the CPU and the GPU, when --draft isn't given. A split's step is bound by
/// reading the CPU layers' weights from RAM, which a verify pass does once
/// for all its rows, while a draft step on the GPU costs ~1-2 ms: longer
/// drafts pay (NOTES: bella 32B sweep).
pub const SPLIT_MODEL_DRAFT: usize = 7;

/// The K to run with, once the main model is placed: `k` (`draft_k`'s), or
/// `SPLIT_MODEL_DRAFT` if a draft model runs next to a split model, the
/// decoding is greedy and K wasn't given (`given`: --draft / AXON_DRAFT),
/// or 0 if a draft model was asked for (`with_model`) but none runs
/// (`--draft-model auto` found none: not prompt lookup instead). Sampled
/// decoding keeps `k`: there a pick equals a greedy guess less often, and
/// on bella's 32B sampled chat K=7 was no faster than K=3 (code and
/// summaries were; docs/design/spec-split.md), so --draft 7 is opt-in.
pub fn split_k(
    given: Option<usize>,
    k: usize,
    with_model: bool,
    drafter: bool,
    split: bool,
    greedy: bool,
) -> usize {
    match (given, with_model, drafter, split && greedy) {
        (_, true, false, _) => 0,
        (None, true, true, true) => SPLIT_MODEL_DRAFT,
        _ => k,
    }
}

/// `--draft-model auto`: the smallest GGUF next to the main model (`main`,
/// at `path`) that can draft for it (`check_vocab`) and is at most
/// `AUTO_MAX_FRACTION` of its size: None if there's none. Opt-in (the
/// flag): drafting is off unless asked for.
pub fn auto_path(path: &str, main: &Model, tok: &Tokenizer) -> Option<String> {
    let p = std::path::Path::new(path);
    let size = std::fs::metadata(p).ok()?.len();
    let dir = match p.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => std::path::Path::new("."),
    };
    let mut cands: Vec<(u64, String)> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let q = e.path();
            // fs::metadata follows symlinks (a Hugging Face cache's
            // snapshot files are links into its blobs): the file's size.
            let len = std::fs::metadata(&q).ok()?.len();
            let gguf = q
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("gguf"));
            (gguf && len > 0 && len <= size / AUTO_MAX_FRACTION && q != p)
                .then(|| (len, q.to_string_lossy().into_owned()))
        })
        .collect();
    cands.sort();
    cands.into_iter().map(|(_, c)| c).find(|c| {
        crate::gguf::load_with(c, Some(false)).is_ok_and(|d| {
            crate::model::check_supported(&d).is_ok() && check_vocab(&d, main, tok).is_ok()
        })
    })
}

/// `auto_path`'s size limit: a draft model at most 1/AUTO_MAX_FRACTION of
/// the main model's file (Qwen2.5-0.5B's 0.49 GB Q4_K_M for 7B's 4.7 GB).
pub const AUTO_MAX_FRACTION: u64 = 8;

/// The token table (`tokenizer.ggml.tokens`) of a GGUF.
fn tokens(m: &Model) -> Option<&[Val]> {
    m.metadata.iter().find_map(|(k, v)| match v {
        Val::Arr(a) if k == "tokenizer.ggml.tokens" => Some(&a[..]),
        _ => None,
    })
}

/// The end-of-generation ids of model `m`, by `Tokenizer::eog`'s rule:
/// `tokenizer.ggml.eos_token_id` and Llama 3's end tokens, looked up in the
/// main model's tokenizer `tok` (whose table `m`'s is a prefix of) and kept
/// if `m`'s table has them.
fn eog_of(m: &Model, tok: &Tokenizer, n_tokens: usize) -> Vec<u32> {
    let eos = m
        .metadata
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.eos_token_id")
        .and_then(|(_, v)| v.as_f64())
        .map_or(151645, |v| v as u32);
    let mut v = vec![eos];
    for s in ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"] {
        v.extend(tok.special_id(s).filter(|&i| (i as usize) < n_tokens));
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Err unless draft model `d` can draft for main model `m` (tokenizer
/// `tok`): `d`'s token table must equal the start of `m`'s (Qwen2.5-0.5B's
/// 151936 entries are the first of 7B's 152064), and both must end
/// generation on the same tokens.
pub fn check_vocab(d: &Model, m: &Model, tok: &Tokenizer) -> Result<(), String> {
    let (Some(dt), Some(mt)) = (tokens(d), tokens(m)) else {
        return Err("a model has no token table (tokenizer.ggml.tokens)".into());
    };
    if dt.len() > mt.len() {
        return Err(format!(
            "its vocabulary has {} tokens, the model's {}",
            dt.len(),
            mt.len()
        ));
    }
    let same = |a: &Val, b: &Val| match (a, b) {
        (Val::Str(a), Val::Str(b)) => a == b,
        _ => false,
    };
    if let Some(i) = dt.iter().zip(mt).position(|(a, b)| !same(a, b)) {
        return Err(format!(
            "its vocabulary differs from the model's at token {i} ({:?} vs {:?}): not the same \
             tokenizer",
            dt[i], mt[i]
        ));
    }
    let (de, me) = (eog_of(d, tok, dt.len()), tok.eog());
    if let Some(&i) = me.iter().find(|&&i| i as usize >= dt.len()) {
        return Err(format!(
            "the model ends generation on token {i}, past its {} tokens",
            dt.len()
        ));
    }
    if de != me {
        return Err(format!(
            "it ends generation on tokens {de:?}, the model on {me:?}"
        ));
    }
    Ok(())
}

/// A draft model's file, checked against the main model's vocabulary.
/// Leaked: it lives as long as the process (as the CLI's `Backend`s).
pub struct DraftModel {
    pub path: String,
    pub model: &'static Model,
    /// The main model's EOG set (= the drafter's: `check_vocab`).
    stop: Vec<u32>,
}

/// GPU memory held for the drafter while the main model is placed.
pub struct Reserve {
    gpu: &'static Gpu,
    module: &'static Module<'static>,
    _buf: DevBuf<'static>,
}

impl DraftModel {
    /// Open `path` (mapped for `device` as the main model is, `copy`) and
    /// check it against main model `m`: Err says why it can't draft for it.
    pub fn open(
        path: &str,
        copy: Option<bool>,
        m: &Model,
        tok: &Tokenizer,
    ) -> Result<DraftModel, String> {
        if path == "auto" {
            return Err("auto: picking a draft model is for gen and chat; give a path".into());
        }
        let d = crate::gguf::load_with(path, copy).map_err(|e| format!("{path}: {e}"))?;
        crate::model::check_supported(&d).map_err(|e| format!("{path}: {e}"))?;
        check_vocab(&d, m, tok).map_err(|e| format!("{path} can't draft for this model: {e}"))?;
        Ok(DraftModel {
            path: path.into(),
            model: Box::leak(Box::new(d)),
            stop: tok.eog(),
        })
    }

    /// On CUDA, the drafter's VRAM for a `ctx`-token cache of type `kv`
    /// (weights, cache, scratch and headroom: `CudaQwen2::vram_curve`),
    /// allocated now so the main model's placement (`Backend::new`) sees
    /// the memory as taken; `load` frees it and puts the drafter there.
    /// None on the CPU.
    pub fn reserve(
        &self,
        device: Device,
        ctx: usize,
        kv: KvType,
    ) -> Result<Option<Reserve>, String> {
        if device == Device::Cpu {
            return Ok(None);
        }
        // The device the main model's head runs on (--gpus: the last one),
        // where verify passes read the drafter's guesses.
        let gpu: &'static Gpu = Box::leak(Box::new(Gpu::open_device(crate::lm::head_gpu()?)?));
        let module: &'static Module<'static> =
            Box::leak(Box::new(Kernels::module_for(gpu, self.model)?));
        let need = CudaQwen2::vram_curve(gpu, module, self.model, cache_kv(kv))?(ctx);
        let free = gpu.info()?.free_bytes;
        if need > free {
            return Err(format!(
                "the draft model needs {:.2} GiB of GPU memory, {:.2} GiB is free",
                need as f64 / GIB,
                free as f64 / GIB
            ));
        }
        let buf = gpu.alloc(need as usize)?;
        Ok(Some(Reserve {
            gpu,
            module,
            _buf: buf,
        }))
    }

    /// The drafter on `device` (all of it on the GPU, in `reserve`'s
    /// memory) with a cache for `ctx` positions (the main model's, after
    /// its placement) of type `kv`.
    pub fn load(
        &self,
        device: Device,
        reserve: Option<Reserve>,
        ctx: usize,
        kv: KvType,
    ) -> Result<ModelDrafter<Backend<'static>>, String> {
        let net = match reserve {
            None => Backend::new(self.model, Device::Cpu, ctx, None, kv, false)?,
            Some(r) => {
                debug_assert_eq!(device, Device::Cuda);
                let (gpu, module) = (r.gpu, r.module);
                drop(r);
                // No Q6_K prefill copy (a spare-VRAM extra, same bits), so
                // the drafter doesn't take memory the main model's cache
                // (made after it) planned for. Its batch scratch may still
                // widen to PREFILL_BATCH from free VRAM (128 tokens' worth).
                let n = CudaQwen2::with_kv_q6k8(
                    gpu,
                    module,
                    self.model,
                    ctx,
                    cache_kv(kv),
                    false,
                    false,
                )
                .map_err(|e| format!("the draft model: {e}"))?;
                Backend(Inner::Cuda(Box::new(n)))
            }
        };
        Ok(ModelDrafter::new(net, ctx, &self.stop))
    }
}

const GIB: f64 = (1u64 << 30) as f64;

/// The GPU cache type for `kv` (as `Backend::new` maps it).
fn cache_kv(kv: KvType) -> CacheKv {
    match kv {
        KvType::Q8 => CacheKv::Q8 { group: 0 },
        _ => CacheKv::F16,
    }
}

/// The main model and its drafter, if `dm`: the drafter's VRAM for a
/// `draft_ctx`-token cache is held while `target` places the main model
/// (so a 7B + 0.5B pair is sized together), then the drafter is loaded
/// with the main model's context (`ctx` = what `Backend::ctx` gives for
/// the requested `want_ctx`), at most draft_ctx (it doesn't draft past
/// its context). gen and chat: draft_ctx = want_ctx; serve sizes the
/// drafter for less than a long trained context.
pub fn load_pair<'a>(
    dm: Option<&DraftModel>,
    device: Device,
    want_ctx: usize,
    draft_ctx: usize,
    kv: KvType,
    target: impl FnOnce() -> Backend<'a>,
) -> Result<(Backend<'a>, Option<ModelDrafter<Backend<'static>>>), String> {
    let Some(dm) = dm else {
        return Ok((target(), None));
    };
    let r = dm.reserve(device, draft_ctx, kv)?;
    let net = target();
    let ctx = net.ctx(want_ctx).min(draft_ctx);
    // A split model gets no warning: its decode is bound by reading the CPU
    // layers' weights, which a verify pass does once for all of its tokens,
    // so drafting pays even though the drafter's VRAM costs the GPU a layer
    // or two (docs/design/spec-split.md).
    if !matches!(&net.0, Inner::Hybrid(_)) && ctx < want_ctx.min(draft_ctx) {
        eprintln!(
            "axon: warning: with the draft model's memory set aside, the context shrank from \
             {} to {ctx} tokens (--draft 0 for the whole context)",
            want_ctx.min(draft_ctx)
        );
    }
    let d = dm.load(device, r, ctx, kv)?;
    Ok((net, Some(d)))
}

impl DraftModel {
    /// The file's name, as gen, chat and serve print it.
    pub fn name(&self) -> String {
        std::path::Path::new(&self.path)
            .file_name()
            .map_or(self.path.clone(), |n| n.to_string_lossy().into_owned())
    }
}

/// The note gen and chat print once when they sample with a draft model.
pub const SAMPLED_NOTE: &str = "axon: note: a draft model gives little or no gain when sampling \
     (measured ~0% at K=3 on 7B, temp 0.8; slower at K=5, 7); --temp 0 for the speedup";

/// The drafter's report line for gen.
pub fn describe<M: Lm>(path: &str, d: &ModelDrafter<M>) -> String {
    let s = d.stats;
    let per = |x: f64, n: usize| if n > 0 { x * 1e3 / n as f64 } else { 0.0 };
    format!(
        "{path}: {} calls, {} steps, {} tokens caught up, {:.1} ms total ({:.2} ms per call)",
        s.calls,
        s.steps,
        s.caught_up,
        s.secs * 1e3,
        per(s.secs, s.calls)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf;
    use crate::model::Qwen2;
    use crate::sample::{SampleArgs, Sampler};
    use crate::{GenOpts, GenStats, generate};

    const TARGET: &str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    const PROMPTS: [&str; 3] = [
        "The history of the Roman Empire",
        "fn add(a: i32, b: i32) -> i32 { a + b }\nfn sub(a: i32, b: i32) -> i32 {",
        "<|im_start|>user\nList three colors.<|im_end|>\n<|im_start|>assistant\n",
    ];

    /// A model file under the worktree's models/ (or `$var`), or None to
    /// skip with AXON_SKIP_MODEL_TESTS set, as `testutil::model_path`.
    fn load(var: Option<&str>, rel: &str) -> Option<Model> {
        let p = match var.and_then(std::env::var_os) {
            Some(p) => std::path::PathBuf::from(p),
            None => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel),
        };
        if !p.exists() {
            if std::env::var_os("AXON_SKIP_MODEL_TESTS").is_some() {
                eprintln!("skipping: {} not present", p.display());
                return None;
            }
            panic!(
                "{} is missing (or set AXON_SKIP_MODEL_TESTS=1)",
                p.display()
            );
        }
        Some(gguf::load(p.to_str().unwrap()).expect("load model"))
    }

    /// The 1.5B target (`AXON_TEST_TARGET` overrides it).
    fn target() -> Option<Model> {
        load(Some("AXON_TEST_TARGET"), TARGET)
    }

    /// The generated ids for `ids` (greedy, or sampled with `seed`) with
    /// guesses from `d` (None: none) of up to `k` tokens.
    #[allow(clippy::too_many_arguments)]
    fn gen_ids<M: Lm>(
        net: &M,
        ids: &[u32],
        n: usize,
        stop: &[u32],
        seed: Option<u64>,
        k: usize,
        d: Option<&mut dyn Draft>,
    ) -> GenStats {
        let mut cache = net.new_cache(ids.len() + n + 1);
        let logits = net.prefill_logits(ids, 0, &mut cache);
        let mut sampler = seed.map(|s| {
            Sampler::new(
                SampleArgs {
                    seed: Some(s),
                    temp: Some(0.8),
                    ..Default::default()
                }
                .resolve(0.0)
                .unwrap(),
            )
        });
        let mut src = match d {
            Some(d) => Drafter::Model(d),
            None => Drafter::Lookup(crate::spec::PromptLookup::new()),
        };
        generate(
            net,
            &mut cache,
            ids,
            logits,
            GenOpts {
                n_gen: n,
                stop,
                max_draft: k,
                sample: sampler.as_mut(),
            },
            &mut |h, max| src.draft(h, max),
            &mut |_| Ok(()),
        )
        .unwrap()
    }

    /// (a): with a draft model, greedy and sampled output equal the output
    /// without drafting, for several K, and some guesses are kept.
    /// One drafter for every run: it follows each new history by itself.
    fn same_output<M: Lm, D: Lm>(net: &M, tok: &Tokenizer, d: D, n: usize) {
        let stop = tok.eog();
        let mut d = ModelDrafter::new(d, 512, &stop);
        let (mut drafted, mut accepted) = (0, 0);
        for prompt in PROMPTS {
            let ids = tok.encode_prompt(prompt);
            for seed in [None, Some(7), Some(12345)] {
                let want = gen_ids(net, &ids, n, &stop, seed, 0, None).ids;
                for k in [1, 3, 5, 7] {
                    let st = gen_ids(net, &ids, n, &stop, seed, k, Some(&mut d));
                    assert_eq!(st.ids, want, "{prompt:?} seed {seed:?} k {k}");
                    assert!(st.accepted <= st.drafted);
                    drafted += st.drafted;
                    accepted += st.accepted;
                }
            }
        }
        eprintln!("accepted {accepted}/{drafted}");
        assert!(accepted * 4 > drafted, "accepted {accepted}/{drafted}");
    }

    #[test]
    fn draft_model_output_unchanged_cpu() {
        let (Some(m), Some(dm)) = (target(), crate::testutil::model()) else {
            return;
        };
        let tok = Tokenizer::from_gguf(&m);
        check_vocab(&dm, &m, &tok).unwrap();
        same_output(&Qwen2::new(&m), &tok, Qwen2::new(&dm), 40);
    }

    #[test]
    fn draft_model_output_unchanged_cuda() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let (Some(m), Some(dm)) = (target(), crate::testutil::model()) else {
            return;
        };
        let tok = Tokenizer::from_gguf(&m);
        let km = Kernels::module_for(&g, &m).unwrap();
        let net = CudaQwen2::new(&g, &km, &m, 512).unwrap();
        // The drafter in its own context handle (as the CLI does it).
        let g2 = Gpu::open().unwrap();
        let km2 = Kernels::module_for(&g2, &dm).unwrap();
        same_output(&net, &tok, CudaQwen2::new(&g2, &km2, &dm, 512).unwrap(), 64);
    }

    /// (a) on the big pairs, on the GPU: AXON_TEST_PAIRS =
    /// "target.gguf:draft.gguf,..." (paths under models/), default the
    /// 0.5B -> 7B, Qwen3 0.6B -> 8B and Llama 3.2 1B -> 3.1 8B pairs.
    #[test]
    #[ignore]
    fn draft_model_output_unchanged_cuda_big() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let pairs = std::env::var("AXON_TEST_PAIRS").unwrap_or(
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf:qwen2.5-0.5b-q4_k_m.gguf,\
             Qwen3-8B-Q4_K_M.gguf:Qwen3-0.6B-Q8_0.gguf,\
             Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf:Llama-3.2-1B-Instruct-Q4_K_M.gguf"
                .into(),
        );
        for pair in pairs.split(',') {
            let (t, d) = pair.trim().split_once(':').unwrap();
            let (Some(m), Some(dm)) = (
                load(None, &format!("models/{t}")),
                load(None, &format!("models/{d}")),
            ) else {
                return;
            };
            eprintln!("{t} <- {d}");
            let tok = Tokenizer::from_gguf(&m);
            check_vocab(&dm, &m, &tok).unwrap();
            let km = Kernels::module_for(&g, &m).unwrap();
            let net = CudaQwen2::new(&g, &km, &m, 512).unwrap();
            let g2 = Gpu::open().unwrap();
            let km2 = Kernels::module_for(&g2, &dm).unwrap();
            same_output(&net, &tok, CudaQwen2::new(&g2, &km2, &dm, 512).unwrap(), 64);
        }
    }

    /// A weightless `Lm`: the pick after token t at position p hashes t, p
    /// and the token cached at p - 1, so a drafter whose cache rows don't
    /// match its history guesses differently from a fresh one.
    struct Toy {
        steps: std::cell::Cell<usize>,
        fed: std::cell::Cell<usize>,
    }

    impl Toy {
        fn new() -> Toy {
            Toy {
                steps: 0.into(),
                fed: 0.into(),
            }
        }
        fn pick(t: u32, pos: usize, c: &mut [u32]) -> u32 {
            c[pos] = t;
            let prev = if pos > 0 { c[pos - 1] } else { 5 };
            (t * 31 + pos as u32 * 7 + prev * 3) % 23
        }
    }

    impl crate::lm::Sealed for Toy {}
    impl Lm for Toy {
        type Cache = Vec<u32>;
        fn vocab(&self) -> usize {
            23
        }
        fn new_cache(&self, max_ctx: usize) -> Vec<u32> {
            vec![u32::MAX; max_ctx]
        }
        fn forward(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> Vec<f32> {
            let mut l = vec![0.0; 23];
            l[self.forward_argmax(t, pos, c) as usize] = 1.0;
            l
        }
        fn forward_argmax(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> u32 {
            assert!(t < 23, "a token past the vocabulary");
            self.steps.set(self.steps.get() + 1);
            Toy::pick(t, pos, c)
        }
        fn feed_batch(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) {
            assert!(ts.iter().all(|&t| t < 23));
            self.fed.set(self.fed.get() + ts.len());
            c[start..start + ts.len()].copy_from_slice(ts);
        }
        fn verify_argmax(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) -> Vec<u32> {
            (0..ts.len())
                .map(|b| Toy::pick(ts[b], start + b, c))
                .collect()
        }
    }

    fn xorshift(s: &mut u32) -> u32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    }

    /// The drafter's cache follows any history: appended tokens, rejected
    /// guesses, edits in the middle (chat forgetting an exchange), a new
    /// request, ids past its vocabulary. Its guesses always equal a fresh
    /// drafter's for the same history, and it only feeds what's new.
    #[test]
    fn drafter_follows_edited_histories() {
        let mut d = ModelDrafter::new(Toy::new(), 200, &[]);
        let mut hist: Vec<u32> = vec![1, 2, 3];
        let mut rng = 0x1234_5678u32;
        for round in 0..400 {
            let max = (xorshift(&mut rng) % 8) as usize;
            let before = (d.net.steps.get(), d.net.fed.get());
            let got = d.draft(&hist, max);
            let mut fresh = ModelDrafter::new(Toy::new(), 200, &[]);
            let want = fresh.draft(&hist, max);
            assert_eq!(got, want, "round {round} hist {hist:?}");
            assert_eq!(got.len(), max, "round {round}: no stop tokens, room left");
            let work = d.net.steps.get() + d.net.fed.get() - before.0 - before.1;
            assert!(work <= hist.len() - 1 + max, "round {round}");
            // The next history: accept some guesses and add a correction,
            // or an edit.
            match xorshift(&mut rng) % 10 {
                0 => {
                    let at = 1 + xorshift(&mut rng) as usize % hist.len();
                    hist.truncate(at);
                }
                1 if hist.len() > 4 => {
                    let i = xorshift(&mut rng) as usize % (hist.len() - 1);
                    hist.drain(i..i + 2);
                }
                2 => hist.push(500), // past the vocabulary: token 0 for it
                _ => {
                    let keep = xorshift(&mut rng) as usize % (got.len() + 1);
                    hist.extend_from_slice(&got[..keep]);
                    hist.push(xorshift(&mut rng) % 23);
                }
            }
            if hist.len() > 150 {
                hist.truncate(3);
            }
        }
        // A fully accepted draft costs one catch-up step, then the draft.
        let mut d = ModelDrafter::new(Toy::new(), 200, &[]);
        let mut hist = vec![1u32, 2, 3];
        let g = d.draft(&hist, 5);
        hist.extend_from_slice(&g);
        hist.push(4);
        assert_eq!(d.net.steps.get(), 2 + 5, "[1, 2] caught up by steps");
        d.draft(&hist, 5);
        assert_eq!((d.net.steps.get(), d.net.fed.get()), (7 + 1 + 5, 0));
        // Many new tokens: one batch.
        hist.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let common = d
            .cached
            .iter()
            .zip(&hist)
            .take_while(|(a, b)| a == b)
            .count();
        let gap = hist.len() - 1 - common;
        assert!(gap > STEP_CATCH_UP);
        d.draft(&hist, 5);
        assert_eq!((d.net.steps.get(), d.net.fed.get()), (13 + 5, gap));
        // Past the context: no guesses.
        assert!(d.draft(&[1; 198], 5).is_empty());
        assert_eq!(d.draft(&[1; 196], 5).len(), 5);
    }

    /// (b): drafting stops before a stop token (the main model's EOG), and
    /// the output with such drafts equals the output without.
    #[test]
    fn drafter_stops_before_eog() {
        let toy = Toy::new();
        let mut c = toy.new_cache(64);
        // The drafter's own chain from [1, 2, 3]: stop at its third pick.
        toy.feed_batch(&[1, 2], 0, &mut c);
        let mut chain = vec![toy.forward_argmax(3, 2, &mut c)];
        for i in 0..6 {
            let t = *chain.last().unwrap();
            chain.push(toy.forward_argmax(t, 3 + i, &mut c));
        }
        let eog = chain[2];
        assert!(!chain[..2].contains(&eog), "{chain:?}");
        let mut d = ModelDrafter::new(Toy::new(), 64, &[eog]);
        assert_eq!(d.draft(&[1, 2, 3], 6), chain[..2]);
        // Through the gen loop (the toy as the main model too: its picks
        // are the drafter's, so it ends at that EOG).
        let net = Toy::new();
        let ids = [1u32, 2, 3];
        let want = gen_ids(&net, &ids, 20, &[eog], None, 0, None);
        for k in [1, 2, 3, 6] {
            let mut d = ModelDrafter::new(Toy::new(), 64, &[eog]);
            let st = gen_ids(&net, &ids, 20, &[eog], None, k, Some(&mut d));
            assert_eq!(st.ids, want.ids, "k {k}");
            assert_eq!(st.ids, chain[..2], "k {k}");
            assert_eq!(st.accepted, st.drafted, "k {k}");
        }
        // EOG mid-draft on a real model: stop on a token greedy emits.
        let (Some(m), Some(dm)) = (target(), crate::testutil::model()) else {
            return;
        };
        let tok = Tokenizer::from_gguf(&m);
        let net = Qwen2::new(&m);
        let ids = tok.encode_prompt(PROMPTS[0]);
        let free = gen_ids(&net, &ids, 24, &tok.eog(), None, 0, None).ids;
        let stop = [free[10]];
        let want = gen_ids(&net, &ids, 24, &stop, None, 0, None).ids;
        assert!(want.len() <= 10);
        let mut d = ModelDrafter::new(Qwen2::new(&dm), ids.len() + 25, &stop);
        for k in [3, 5] {
            let st = gen_ids(&net, &ids, 24, &stop, None, k, Some(&mut d));
            assert_eq!(st.ids, want, "k {k}");
        }
    }

    #[test]
    fn draft_k_defaults() {
        assert_eq!(draft_k(None, true), DEFAULT_MODEL_DRAFT);
        assert_eq!(DEFAULT_MODEL_DRAFT, 3);
        // An explicit K is kept with a draft model, rounded for prompt lookup.
        assert_eq!(draft_k(Some(5), true), 5);
        assert_eq!(draft_k(Some(0), true), 0);
        assert_eq!(draft_k(Some(5), false), crate::round_draft(5));
    }

    #[test]
    fn split_k_defaults() {
        // A draft model next to a split model, greedy: the split's K unless given.
        let d = DEFAULT_MODEL_DRAFT;
        assert_eq!(split_k(None, d, true, true, true, true), SPLIT_MODEL_DRAFT);
        assert_eq!(split_k(Some(3), 3, true, true, true, true), 3);
        assert_eq!(split_k(Some(0), 0, true, false, true, true), 0);
        // Sampled: K as draft_k gave it, split or not.
        assert_eq!(split_k(None, d, true, true, true, false), d);
        // Not split, or prompt lookup: K as draft_k gave it.
        assert_eq!(split_k(None, d, true, true, false, true), d);
        assert_eq!(split_k(None, 7, false, false, true, true), 7);
        // A draft model asked for (auto) but none found: off, not lookup.
        assert_eq!(split_k(None, d, true, false, true, true), 0);
        assert_eq!(split_k(Some(5), 5, true, false, false, true), 0);
    }

    fn fake(tokens: &[Val], eos: u32) -> Model {
        gguf::build_for_test(
            &[
                ("tokenizer.ggml.tokens", Val::Arr(tokens.to_vec())),
                ("tokenizer.ggml.eos_token_id", Val::U32(eos)),
            ],
            &[],
        )
    }

    /// (c): a draft model with another vocabulary or other end tokens is
    /// refused; one whose table is a prefix of the model's is not.
    #[test]
    fn vocab_mismatch_is_refused() {
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let tok = Tokenizer::from_gguf(&m);
        let t = tokens(&m).unwrap().to_vec();
        let eos = tok.eos();
        check_vocab(&m, &m, &tok).unwrap();
        // A prefix (a smaller embedding) is fine, if it holds the EOS.
        check_vocab(&fake(&t[..eos as usize + 1], eos), &m, &tok).unwrap();
        check_vocab(&fake(&t, eos), &m, &tok).unwrap();
        let e = check_vocab(&fake(&t[..eos as usize], eos), &m, &tok).unwrap_err();
        assert!(e.contains("ends generation"), "{e}");
        // Longer than the model's.
        let mut long = t.clone();
        long.push(Val::Str("extra".into()));
        let e = check_vocab(&fake(&long, eos), &m, &tok).unwrap_err();
        assert!(e.contains("tokens"), "{e}");
        // One token differs.
        let mut other = t.clone();
        other[1000] = Val::Str("zzz".into());
        let e = check_vocab(&fake(&other, eos), &m, &tok).unwrap_err();
        assert!(e.contains("at token 1000"), "{e}");
        // Another EOS.
        let e = check_vocab(&fake(&t, eos - 1), &m, &tok).unwrap_err();
        assert!(e.contains("ends generation"), "{e}");
        // No table.
        assert!(check_vocab(&gguf::build_for_test(&[], &[]), &m, &tok).is_err());
        // Real families: Llama 3 for Qwen2.5, Qwen3 for Qwen2.5.
        if let Some(l) = crate::testutil::llama_model() {
            let e = check_vocab(&l, &m, &tok).unwrap_err();
            assert!(e.contains("tokens") || e.contains("differs"), "{e}");
            let lt = Tokenizer::from_gguf(&l);
            assert!(check_vocab(&m, &l, &lt).is_err());
        }
        if let Some(q3) = crate::testutil::qwen3_model() {
            assert!(check_vocab(&q3, &m, &tok).is_err());
        }
    }

    /// (d): chat turns with a draft model are the turns without one,
    /// greedy and sampled, also when the context forgets old exchanges.
    #[test]
    fn chat_with_draft_model_unchanged() {
        let (Some(m), Some(dm)) = (target(), crate::testutil::model()) else {
            return;
        };
        let tok = Tokenizer::from_gguf(&m);
        let net = Qwen2::new(&m);
        let msgs = [
            "What is the capital of France?",
            "Write fn add(a: i32, b: i32) in Rust.",
            "Name three colors.",
            "And three animals?",
            "Thanks!",
        ];
        drop(dm);
        for seed in [None, Some(3u64)] {
            // A CPU model is made once per loaded file (it repacks and
            // releases the file's matrices): a fresh load per drafter.
            let dm = crate::testutil::model().unwrap();
            let mut runs = Vec::new();
            for with in [false, true] {
                let mut chat = crate::chat::Chat::new(&net, &tok, Some("Be brief."), 160).unwrap();
                chat.max_reply = 24;
                chat.max_draft = if with { 5 } else { 0 };
                chat.sampler = seed.map(|s| {
                    Sampler::new(
                        SampleArgs {
                            seed: Some(s),
                            ..Default::default()
                        }
                        .resolve(crate::sample::DEFAULT_TEMP)
                        .unwrap(),
                    )
                });
                if with {
                    chat.drafter = Some(Box::new(ModelDrafter::new(
                        Qwen2::new(&dm),
                        160,
                        &tok.eog(),
                    )));
                }
                let (mut replies, mut dropped, mut accepted) = (Vec::new(), 0, 0);
                for msg in msgs.iter().chain(&msgs) {
                    let r = chat.reply(msg, &|| false, &mut |_| Ok(())).unwrap();
                    dropped += r.dropped;
                    accepted += r.accepted;
                    replies.push(r.ids);
                }
                assert!(dropped > 0, "the context never filled");
                assert_eq!(accepted > 0, with, "seed {seed:?}");
                runs.push(replies);
            }
            assert_eq!(runs[0], runs[1], "seed {seed:?}");
        }
    }
}
