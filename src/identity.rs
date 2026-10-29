//! The identity tuple: everything that decides Axon's output bits, hashed
//! into one fingerprint (OpenAI's `system_fingerprint`). Two runs with the
//! same fingerprint, prompt ids, sampling params and seed give the same
//! tokens, bit for bit, however the requests were batched, chunked, cached
//! or drafted (docs/determinism.md states the contract and its limits).
//!
//! In the tuple (`Identity::fields`, in this order):
//! - `numerics`: the numerics mode and its version, `exact/N` or `fast/N`
//!   (`NUMERICS_VERSION_EXACT`, `NUMERICS_VERSION_FAST`; `crate::numerics`):
//!   Axon's own kernels and sampler.
//! - `model`: `model_digest` of the GGUF file (its whole header, so every
//!   metadata key, tensor name, shape and quant type, plus a sample of the
//!   tensor data), and `types`, the quant types (readable; the header hash
//!   already covers them).
//! - `backend`: cpu, cuda, cuda+cpu (with the layer split, and whether its
//!   CPU layers stream through the GPU for prefill: then those batches get
//!   GPU numerics), or a pipeline (with its cuts).
//! - `gpu`: per GPU, its exact architecture (sm_XY). Audited (see
//!   docs/determinism.md): two sm_89 cards with different NVRTC (12.8,
//!   13.3) and drivers agree bit for bit, so the toolkit isn't in it; sm_61
//!   and sm_89 differ from prefill on (split f32 attention below sm_80,
//!   tensor cores from sm_80); other pairs are unaudited, so each sm_XY is
//!   its own. Also the code loaded when it isn't the device's own cubin,
//!   and without tensor-core attention the split count (`max_split`, from
//!   the SM count), which orders that path's sums.
//! - `cpu`: what decides CPU bits (fused multiply-add or not, r8 repack or
//!   per-row kernels) when any layer runs on the CPU.
//! - `kv`: the KV cache type (f16, q8, q8/32; f32 on the CPU).
//! - `rope`: the effective RoPE scaling (`rope_field`: YaRN from the GGUF
//!   or --rope-scaling, gemma3's linear scale) when not plain RoPE.
//! - `serve`: `axon serve`'s settings a request inherits when it leaves
//!   them out (`serve_field`), only when not the defaults.
//!
//! Not in it, because they never change bits: drafting (lossless, prompt
//! lookup or a draft model), the context size (--ctx or serve's default,
//! the trained length where it fits: attention runs over the positions
//! filled, in 64-position blocks, so a bigger cache only lets a request run
//! longer and never changes a token below the limit), --parallel and the
//! slot count, chunked prefill, thread counts, CUDA graphs and PDL. All of
//! that is for a given placement: on an automatic CPU/GPU split, a draft
//! model's VRAM reserve and the cache size (--ctx, gen's too) feed where the
//! split falls, so they can move it, which changes bits, and `backend`'s
//! layer split says so. Pin --gpu-layers for reproducible runs. Sampling
//! params and the seed are per request, so they're the request's, not the
//! server's. A flag that does change bits must add a field here, set only
//! when the flag is (as --rope-scaling's `rope`).
//!
//! The model digest reads the file itself (never the loaded copy, which
//! may have been rewritten or released), so it's a function of the file's
//! bytes alone.

use crate::gguf::Model;
use crate::lm::{Backend, Inner};
use crate::numerics::Numerics;
use std::io;
use std::path::Path;

/// Exact mode's numerics version (`numerics=exact/N`). RULE: bump it in
/// the same commit as any change that can change an output bit in exact
/// mode for some model on some backend: a kernel's arithmetic or reduction
/// order (CPU or CUDA), the attention fold, RoPE, norms, the sampler's order
/// or RNG, a quant format's dequant; or what turns a request into tokens or
/// text: the tokenizer, chat templates and prompt rendering (serve.rs
/// `Template`, tools.rs), stop tokens and strings, think handling. Changes
/// that keep every bit (speed, scheduling, a kernel that matches the old one
/// bit for bit, proven by the md5 gate) don't bump it. A bump changes every
/// exact fingerprint, so a client caching on it knows its old outputs are
/// stale. (Before 2026-10-29 the tuple's line was `numerics=1`: the same
/// kernels as `exact/1`, another fingerprint.)
pub const NUMERICS_VERSION_EXACT: u32 = 1;

/// Fast mode's numerics version (`numerics=fast/N`,
/// docs/design/determinism-modes.md §4.3). RULE: bump it in the same commit
/// as any change that can change an output bit in fast mode: a fast-set
/// kernel's arithmetic or its routing (which products run a fast kernel,
/// at which shapes). Code both modes run bumps both counters: every kernel
/// today (fast mode has no kernel of its own yet and runs the exact ones),
/// and the request-to-text parts `NUMERICS_VERSION_EXACT` lists. The first
/// fast kernel is a bump.
pub const NUMERICS_VERSION_FAST: u32 = 1;

/// The tuple's format, the first line of `canonical`.
const SCHEME: &str = "axon-identity/1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// (name, value), in the order the module doc lists them.
    pub fields: Vec<(&'static str, String)>,
    /// The fields that couldn't be read (their value says why): then no
    /// identity is claimed (`fingerprint` is `fp_unknown`).
    pub unknown: Vec<&'static str>,
}

impl Identity {
    /// The tuple from its parts (`of` reads them off a loaded model).
    pub fn new(
        model: &str,
        types: &str,
        backend: &str,
        gpu: Option<&str>,
        cpu: Option<&str>,
        kv: &str,
    ) -> Identity {
        let mut f = vec![
            ("numerics", Numerics::Exact.tag()),
            ("model", model.to_string()),
            ("types", types.to_string()),
            ("backend", backend.to_string()),
        ];
        if let Some(g) = gpu {
            f.push(("gpu", g.to_string()));
        }
        if let Some(c) = cpu {
            f.push(("cpu", c.to_string()));
        }
        f.push(("kv", kv.to_string()));
        Identity {
            fields: f,
            unknown: Vec::new(),
        }
    }

    /// The tuple in numerics mode `n` (`new`'s is exact): its `numerics`
    /// field says the mode and that mode's version.
    pub fn with_numerics(mut self, n: Numerics) -> Identity {
        if let Some(f) = self.fields.iter_mut().find(|(k, _)| *k == "numerics") {
            f.1 = n.tag();
        }
        self
    }

    /// The tuple with `field` marked unreadable (`unknown`).
    pub fn unreadable(mut self, field: &'static str) -> Identity {
        if !self.unknown.contains(&field) {
            self.unknown.push(field);
        }
        self
    }

    /// The tuple with the effective RoPE scaling (`rope_field`), if any,
    /// after `kv`.
    pub fn with_rope(mut self, rope: Option<String>) -> Identity {
        if let Some(r) = rope {
            self.fields.push(("rope", r));
        }
        self
    }

    /// What the fingerprint hashes: the scheme, then `name=value` lines.
    pub fn canonical(&self) -> String {
        let mut s = format!("{SCHEME}\n");
        for (k, v) in &self.fields {
            s.push_str(&format!("{k}={v}\n"));
        }
        s
    }

    /// The tuple with the server's `serve_field`, if any, last.
    pub fn with_serve(mut self, serve: Option<String>) -> Identity {
        if let Some(s) = serve {
            self.fields.push(("serve", s));
        }
        self
    }

    /// `fp_` and the first 16 hex digits of the SHA-256 of `canonical`;
    /// `fp_unknown` if a part couldn't be read (see `of`).
    pub fn fingerprint(&self) -> String {
        if !self.unknown.is_empty() {
            return "fp_unknown".into();
        }
        format!("fp_{}", &hex(&sha256(self.canonical().as_bytes()))[..16])
    }

    /// The tuple as a JSON object, the fingerprint included (GET /identity).
    pub fn json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("system_fingerprint".into(), self.fingerprint().into());
        m.insert("scheme".into(), SCHEME.into());
        for (k, v) in &self.fields {
            m.insert((*k).into(), v.clone().into());
        }
        if !self.unknown.is_empty() {
            m.insert("unknown".into(), self.unknown.clone().into());
        }
        serde_json::Value::Object(m)
    }

    /// One line for logs and `gen`: the fingerprint and the tuple.
    pub fn line(&self) -> String {
        let parts: Vec<String> = (self.fields.iter())
            .map(|(k, v)| match *k {
                "model" => format!("model {}", &v[..v.len().min(16)]),
                _ => format!("{k} {v}"),
            })
            .collect();
        format!("{} ({})", self.fingerprint(), parts.join(", "))
    }
}

/// The identity of `net`, loaded from the GGUF file at `path` (`m`). A
/// part that can't be read (the file, the GPU) says why ("unknown (...)")
/// and is marked `unreadable`: the fingerprint is then `fp_unknown`, no
/// identity is claimed. `m` is the model as run: after `RopeArgs::apply`,
/// so `rope` sees --rope-scaling too.
pub fn of(path: &str, m: &Model, net: &Backend) -> Identity {
    let digest = model_digest(Path::new(path), m);
    let d = digest
        .as_ref()
        .map_or_else(|e| format!("unknown ({e})"), |d| d.clone());
    let types = quant_types(m);
    let cpu_kv = if crate::kv::use_f32() { "f32" } else { "f16" };
    let mut gpu_ok = true;
    let mut gpu = |n: &crate::cuda::qwen2::CudaQwen2| {
        let (g, ok) = gpu(n);
        gpu_ok &= ok;
        g
    };
    let id = match &net.0 {
        Inner::Cpu(_) => Identity::new(&d, &types, "cpu", None, Some(&cpu()), cpu_kv),
        Inner::Cuda(n) => {
            let g = gpu(n);
            Identity::new(&d, &types, "cuda", Some(&g), None, n.kv_name())
        }
        Inner::Hybrid(n) => {
            let g = n.gpu_net().map(&mut gpu);
            let b = hybrid_backend(n.gpu_layers, n.n_layers, n.streams());
            let kv = format!("{} (cpu layers {cpu_kv})", n.kv_name());
            Identity::new(&d, &types, &b, g.as_deref(), Some(&cpu()), &kv)
        }
        Inner::Moe(n) => {
            let g = gpu(n.gpu_net());
            let kv = format!("{} (experts on the cpu)", n.kv_name());
            Identity::new(&d, &types, "cuda+cpu experts", Some(&g), Some(&cpu()), &kv)
        }
        Inner::Pipeline(n) => {
            let gs: Vec<String> = n.stages().into_iter().map(&mut gpu).collect();
            let b = pipeline_backend(&n.cuts);
            Identity::new(&d, &types, &b, Some(&gs.join("; ")), None, n.kv_name())
        }
    };
    let mut id = id.with_rope(rope_field(m)).with_numerics(net.numerics());
    if digest.is_err() {
        id = id.unreadable("model");
    }
    if !gpu_ok {
        id = id.unreadable("gpu");
    }
    id
}

/// The `rope` field: the model's effective RoPE scaling, read from its
/// metadata as run (the GGUF's own keys, rewritten in memory by
/// --rope-scaling), when it isn't plain RoPE: YaRN's parameters (factor,
/// original context, betas, ext factor, (sin, cos) multiplier) and gemma3's
/// linear scale, with the frequency base. None for plain RoPE, so plain
/// models' fingerprints carry no `rope` field. The model digest can't do
/// this: it hashes the file, which the flags don't touch.
pub fn rope_field(m: &Model) -> Option<String> {
    let y = crate::rope::yarn(m);
    let lin = crate::model::rope_linear_scale(m);
    if y.is_none() && lin == 1.0 {
        return None;
    }
    let base = crate::model::meta_num(m, &format!("{}.rope.freq_base", m.arch)).unwrap_or(10000.0);
    let mut v = Vec::new();
    if let Some(y) = y {
        v.push(format!(
            "yarn factor {} orig {} beta_fast {} beta_slow {} ext {} mscale {}",
            y.factor, y.orig, y.beta_fast, y.beta_slow, y.ext, y.mscale
        ));
    }
    if lin != 1.0 {
        v.push(format!("linear {lin}"));
    }
    v.push(format!("freq_base {base}"));
    Some(v.join(", "))
}

/// The backend field of a CPU/GPU split: its layer split and, when the
/// CPU's layers stream through the GPU for prefill batches of at least
/// `min` tokens, that: those batches run the CPU layers with GPU numerics
/// (`cuda::qwen2::stream`), so a streamed split is not prefill == decode
/// invariant, and whether it streams depends on free VRAM at load.
pub fn hybrid_backend(
    gpu_layers: usize,
    n_layers: usize,
    streams: Option<(usize, bool)>,
) -> String {
    let mut b = format!("cuda+cpu, gpu layers {gpu_layers}/{n_layers}");
    if let Some((min, decode)) = streams {
        b.push_str(&format!(", prefill streams >={min}"));
        if decode {
            b.push_str(", decode streams");
        }
    }
    b
}

/// The backend field of a pipeline over GPUs: its layer cuts.
pub fn pipeline_backend(cuts: &[usize]) -> String {
    let cuts: Vec<String> = cuts.iter().map(|c| c.to_string()).collect();
    format!("cuda pipeline, cuts {}", cuts.join(","))
}

/// A GPU's field: its exact architecture; the code loaded for it when that
/// isn't the device's own cubin (an older cubin of the major, PTX the driver
/// compiles, sm_90a); and without tensor-core attention (head size 256, or
/// below sm_80) that path's split count, which follows the SM count.
pub fn gpu_field(arch: &str, code: &str, split: Option<usize>) -> String {
    let mut s = arch.to_string();
    if code != arch {
        s.push_str(&format!(" [code {code}]"));
    }
    if let Some(k) = split {
        s.push_str(&format!(", split attention {k}"));
    }
    s
}

/// `gpu_field` of a loaded GPU model, and whether every part of it could
/// be read.
fn gpu(n: &crate::cuda::qwen2::CudaQwen2) -> (String, bool) {
    let g = n.gpu();
    let (info, code) = (g.info(), crate::cuda::code_target(g));
    let ok = info.is_ok() && code.is_ok();
    let arch = info.map_or_else(|e| format!("unknown ({e})"), |i| i.arch());
    let code = code.map_or_else(|e| format!("unknown ({e})"), |c| c.0);
    (gpu_field(&arch, &code, n.split_attention()), ok)
}

/// The server's settings that change a request's output when it leaves
/// them out: --no-think (the prompt's rendering), --think-inline (where the
/// reasoning goes), and its default sampling flags. None when all are the
/// defaults, so default fingerprints carry no `serve` field. (A request's
/// own params, seed and `max_tokens` are its own; without `max_tokens` a
/// reply runs to the context or --max-tokens, which bound its length, not
/// its tokens.)
pub fn serve_field(
    no_think: bool,
    think_inline: bool,
    s: &crate::sample::SampleArgs,
) -> Option<String> {
    let mut v = Vec::new();
    if no_think {
        v.push("no-think".to_string());
    }
    if think_inline {
        v.push("think-inline".to_string());
    }
    let mut f = |name: &str, x: Option<String>| {
        if let Some(x) = x {
            v.push(format!("{name} {x}"));
        }
    };
    f("temp", s.temp.map(|x| x.to_string()));
    f("top-k", s.top_k.map(|x| x.to_string()));
    f("top-p", s.top_p.map(|x| x.to_string()));
    f("min-p", s.min_p.map(|x| x.to_string()));
    f("repeat-penalty", s.repeat_penalty.map(|x| x.to_string()));
    f("repeat-last-n", s.repeat_last_n.map(|x| x.to_string()));
    f("seed", s.seed.map(|x| x.to_string()));
    (!v.is_empty()).then(|| v.join(", "))
}

/// The identities `m` (at `path`) would have without loading it (`axon
/// info`): on the CPU (its cache type as AXON_KV says) and, on CUDA device
/// 0, with each cache type if the weights fit its free VRAM (else why
/// not: a split's identity depends on where it's placed), as (label,
/// identity or why none).
pub fn predicted(path: &str, m: &Model) -> io::Result<Vec<(String, Result<Identity, String>)>> {
    let digest = model_digest(Path::new(path), m)?;
    let types = quant_types(m);
    let cpu_kv = if crate::kv::use_f32() { "f32" } else { "f16" };
    let mut v = vec![(
        "cpu".to_string(),
        Ok(
            Identity::new(&digest, &types, "cpu", None, Some(&cpu()), cpu_kv)
                .with_rope(rope_field(m)),
        ),
    )];
    if crate::model::is_moe(m) {
        let why = "not predicted here (attention on the GPU, experts on the CPU; it runs): \
                   gen --device cuda prints its identity";
        v.push(("cuda".to_string(), Err(why.to_string())));
        return Ok(v);
    }
    let Ok(g) = crate::cuda::Gpu::open() else {
        return Ok(v);
    };
    let (Ok(info), Ok((code, code_arch))) = (g.info(), crate::cuda::code_target(&g)) else {
        return Ok(v);
    };
    if m.total_bytes() >= info.free_bytes {
        let why = format!(
            "the weights don't fit {}'s free VRAM: gen would split the model, and a split's \
             identity depends on its placement (gen --verbose prints it)",
            info.arch()
        );
        v.push(("cuda".to_string(), Err(why)));
        return Ok(v);
    }
    // CudaQwen2's rule: tensor-core attention needs mma (sm_80+ code) and
    // a head size other than 256.
    let n_kv = m.n_kv_heads().max(1) as usize;
    let head = (m.meta_u32(&format!("{}.attention.key_length", m.arch)))
        .unwrap_or(m.embed_dim() / m.n_heads().max(1));
    let split = (code_arch < 80 || head == 256)
        .then(|| crate::cuda::qwen2::max_split(info.sms as usize, n_kv));
    let gs = gpu_field(&info.arch(), &code, split);
    for kv in ["f16", "q8"] {
        let id = match kv == "q8" && head == 256 {
            // The q8 cache has no head size 256 kernels: gen refuses it.
            true => Err("no q8 cache at head size 256".to_string()),
            false => Ok(Identity::new(&digest, &types, "cuda", Some(&gs), None, kv)
                .with_rope(rope_field(m))),
        };
        v.push((format!("cuda --kv {kv}"), id));
    }
    Ok(v)
}

/// The CPU's part: what changes its layers' bits. SIMD and scalar paths
/// agree bit for bit at any thread count (the `simd_matches_scalar*` and
/// `multiversion_*` tests; an AVX2 and an AVX-VNNI host agree 20/20, see
/// docs/determinism.md), so the SIMD path isn't in it; FMA is (fused or
/// not, `vmath::fused`), and so is AXON_REPACK=0 (per-row kernels round
/// differently from the r8 repack).
fn cpu() -> String {
    let repack = std::env::var("AXON_REPACK").map_or(true, |v| v != "0");
    format!(
        "{} {}, {}",
        std::env::consts::ARCH,
        if crate::vmath::fused() {
            "fma"
        } else {
            "no fma"
        },
        if repack { "r8" } else { "per-row" }
    )
}

/// The quant types of `m`'s tensors with their counts, by type id.
fn quant_types(m: &Model) -> String {
    let mut c: Vec<(u32, &str, usize)> = Vec::new();
    for t in &m.tensors {
        match c.iter_mut().find(|x| x.0 == t.qtype.id) {
            Some(x) => x.2 += 1,
            None => c.push((t.qtype.id, t.qtype.name, 1)),
        }
    }
    c.sort();
    let s: Vec<String> = c.iter().map(|(_, n, k)| format!("{n}:{k}")).collect();
    s.join(" ")
}

/// Bytes read at each end of every tensor, and the size and count of the
/// evenly spaced windows over the whole data section.
const TENSOR_EDGE: u64 = 256;
const WINDOW: u64 = 4096;
const WINDOWS: u64 = 1024;

/// The model's digest: SHA-256 (hex) of "axon-model-digest/2\0", the file
/// length (u64 LE), then the bytes of the union of these ranges, in file
/// order (overlaps once): the whole header (metadata and tensor table, up
/// to the data), the first and last `TENSOR_EDGE` bytes of every tensor,
/// and `WINDOWS` windows of `WINDOW` bytes evenly spaced over the data (all
/// clamped to the file). A few MB in ~1-2k sorted reads with readahead off
/// (`fadvise` RANDOM), ~40 ms warm for a 7B, against seconds for every byte
/// of a 5 GB file on every start. Another model, a requant, a fine-tune or
/// a merge changes the header or (nearly) every block, so the samples see
/// it; a deliberate edit of bytes between samples would not (this names a
/// file; it isn't a tamper seal). It reopens the file by path: a file
/// replaced after loading is digested as the new one.
pub fn model_digest(path: &Path, m: &Model) -> io::Result<String> {
    if let crate::gguf::Source::Safetensors { files, .. } = &m.source {
        return checkpoint_digest(files);
    }
    let tensors: Vec<(u64, u64)> = (m.tensors.iter())
        .map(|t| (m.data_offset + t.offset, t.n_bytes))
        .collect();
    digest_file(path, m.data_offset, &tensors)
}

/// `model_digest` of a safetensors checkpoint: SHA-256 of
/// "axon-model-digest/st1\0" and, per file (`Source::Safetensors::files`),
/// its name and its `digest_file` digest: a shard's with its header and its
/// tensors' edges, a config or tokenizer file's whole (its "header").
fn checkpoint_digest(files: &[std::path::PathBuf]) -> io::Result<String> {
    let mut h = Sha256::new();
    h.update(b"axon-model-digest/st1\0");
    for f in files {
        let name = f.file_name().unwrap_or_default().to_string_lossy();
        let (data, tensors) = match name.ends_with(".safetensors") {
            true => {
                let file = std::fs::File::open(f)?;
                // SAFETY: read-only, only read here.
                let map = unsafe { memmap2::Mmap::map(&file)? };
                let hd = crate::safetensors::parse_header(&map)?;
                let start = 8 + u64::from_le_bytes(map[..8].try_into().unwrap());
                let ts: Vec<(u64, u64)> = (hd.tensors.iter())
                    .map(|t| (t.begin, t.end - t.begin))
                    .collect();
                (start, ts)
            }
            // Config and tokenizer files: hashed whole (all "header").
            false => (u64::MAX, Vec::new()),
        };
        h.update(name.as_bytes());
        h.update(&[0]);
        h.update(digest_file(f, data, &tensors)?.as_bytes());
    }
    Ok(hex(&h.finish()))
}

/// The ranges `digest_file` hashes, for a file of `len` bytes: sorted,
/// overlapping or touching ones merged, empty ones dropped.
fn digest_ranges(len: u64, data_offset: u64, tensors: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut r = vec![(0, data_offset)];
    for &(at, n) in tensors {
        let e = n.min(TENSOR_EDGE);
        r.push((at, at + e));
        r.push((at + n - e, at + n));
    }
    let data = len.saturating_sub(data_offset);
    if data > 0 {
        let span = data.saturating_sub(WINDOW);
        for i in 0..WINDOWS {
            let at = data_offset + span * i / (WINDOWS - 1);
            r.push((at, at + WINDOW));
        }
    }
    let mut r: Vec<(u64, u64)> = (r.into_iter())
        .map(|(a, b)| (a.min(len), b.min(len)))
        .filter(|(a, b)| a < b)
        .collect();
    r.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(r.len());
    for (a, b) in r {
        match out.last_mut() {
            Some(l) if a <= l.1 => l.1 = l.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// `model_digest` for a file whose data starts at `data_offset`, with
/// tensors at (absolute offset, length).
fn digest_file(path: &Path, data_offset: u64, tensors: &[(u64, u64)]) -> io::Result<String> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    fadvise_random(&f);
    let mut h = Sha256::new();
    h.update(b"axon-model-digest/2\0");
    h.update(&len.to_le_bytes());
    let mut buf = Vec::new();
    for (a, b) in digest_ranges(len, data_offset, tensors) {
        buf.resize((b - a) as usize, 0);
        f.read_exact_at(&mut buf, a)?;
        h.update(&buf);
    }
    Ok(hex(&h.finish()))
}

/// No readahead on `f` (POSIX_FADV_RANDOM, Linux): the digest's reads are
/// small and scattered. Best effort.
fn fadvise_random(f: &std::fs::File) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        unsafe extern "C" {
            fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
        }
        // SAFETY: a valid open fd; the call only sets a hint on it.
        let _ = unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, 1) };
    }
    #[cfg(not(target_os = "linux"))]
    let _ = f;
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// SHA-256 of `b` (FIPS 180-4).
pub fn sha256(b: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b);
    h.finish()
}

/// SHA-256, streaming. Plain and portable: it hashes a few MB per start.
struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    n: usize,
    len: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Sha256 {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0; 64],
            n: 0,
            len: 0,
        }
    }

    fn update(&mut self, mut b: &[u8]) {
        self.len += b.len() as u64;
        while !b.is_empty() {
            let k = (64 - self.n).min(b.len());
            self.buf[self.n..self.n + k].copy_from_slice(&b[..k]);
            self.n += k;
            b = &b[k..];
            if self.n == 64 {
                let blk = self.buf;
                self.block(&blk);
                self.n = 0;
            }
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.n != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        let mut out = [0u8; 32];
        for (o, h) in out.chunks_exact_mut(4).zip(self.h) {
            o.copy_from_slice(&h.to_be_bytes());
        }
        out
    }

    fn block(&mut self, b: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b_, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b_) ^ (a & c) ^ (b_ & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b_;
            b_ = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in self.h.iter_mut().zip([a, b_, c, d, e, f, g, h]) {
            *x = x.wrapping_add(y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let two = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            hex(&sha256(two)),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Streaming in odd pieces == one shot.
        let big: Vec<u8> = (0..10_000u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut h = Sha256::new();
        for c in big.chunks(37) {
            h.update(c);
        }
        assert_eq!(h.finish(), sha256(&big));
    }

    fn base() -> Identity {
        Identity::new("aa", "Q4_K:1", "cuda", Some("sm_120"), None, "f16")
    }

    /// The fingerprint is a pure function of the tuple: the same every time
    /// (and so on every run: no clock, pid or address goes in), and any part
    /// changing changes it.
    #[test]
    fn fingerprint_stable_and_sensitive() {
        let a = base();
        assert_eq!(a.fingerprint(), base().fingerprint());
        assert_eq!(a.fingerprint().len(), 3 + 16);
        assert!(a.fingerprint().starts_with("fp_"));
        // Pinned: a change here means every deployed fingerprint changed,
        // which needs a NUMERICS_VERSION_* or SCHEME reason (2026-10-29: the
        // numerics mode, `numerics=1` -> `numerics=exact/1`, ADR M1).
        assert_eq!(
            a.canonical(),
            "axon-identity/1\nnumerics=exact/1\nmodel=aa\ntypes=Q4_K:1\nbackend=cuda\n\
             gpu=sm_120\nkv=f16\n"
        );
        // Golden (Python's hashlib over the line above; `numerics=1` was
        // fp_e43c8e0d7c77ef03).
        assert_eq!(a.fingerprint(), "fp_1e9019e52a04cc54");
        let g = Some("sm_120");
        let variants = [
            Identity::new("aa", "Q4_K:1", "cuda", g, None, "q8"),
            Identity::new("ab", "Q4_K:1", "cuda", g, None, "f16"),
            Identity::new("aa", "Q4_K:1", "cpu", None, Some("x86_64 fma, r8"), "f16"),
            Identity::new("aa", "Q4_K:1", "cuda", Some("sm_89"), None, "f16"),
            Identity::new("aa", "Q4_K:1", "cuda", Some("sm_61"), None, "f16"),
            Identity::new("aa", "Q6_K:1", "cuda", g, None, "f16"),
            base().with_numerics(Numerics::Fast),
        ];
        let mut seen = vec![a.fingerprint()];
        for v in &variants {
            let f = v.fingerprint();
            assert!(!seen.contains(&f), "{v:?} collides");
            seen.push(f);
        }
        let mut n = base();
        n.fields[0].1 = "exact/2".into();
        assert_ne!(n.fingerprint(), a.fingerprint(), "numerics version");
        let j = a.json();
        assert_eq!(j["system_fingerprint"], a.fingerprint());
        assert_eq!(j["numerics"], "exact/1");
        assert_eq!(j["kv"], "f16");
    }

    /// The numerics mode (ADR M1): `numerics=exact/N` by default,
    /// `fast/N` in fast mode, the rest of the tuple the same; fast is its
    /// own fingerprint, stable, and exact is `new`'s. Pinned, as the exact
    /// one above.
    #[test]
    fn numerics_mode_in_the_fingerprint() {
        let e = base();
        assert_eq!(e.clone().with_numerics(Numerics::Exact), e);
        let f = base().with_numerics(Numerics::Fast);
        assert_eq!(
            f.canonical(),
            "axon-identity/1\nnumerics=fast/1\nmodel=aa\ntypes=Q4_K:1\nbackend=cuda\n\
             gpu=sm_120\nkv=f16\n"
        );
        assert_eq!(f.fingerprint(), "fp_380702fdbbb38616", "golden (hashlib)");
        assert_ne!(f.fingerprint(), e.fingerprint());
        assert_eq!(f.fields[1..], e.fields[1..], "only the numerics field");
        assert_eq!(f.json()["numerics"], "fast/1");
        assert!(f.line().contains("numerics fast/1"), "{}", f.line());
        // Back to exact: the same tuple as never fast.
        assert_eq!(f.with_numerics(Numerics::Exact), e);
        // Each counter is in its own mode's line only.
        assert_eq!(
            Numerics::Exact.tag(),
            format!("exact/{NUMERICS_VERSION_EXACT}")
        );
        assert_eq!(
            Numerics::Fast.tag(),
            format!("fast/{NUMERICS_VERSION_FAST}")
        );
        // `of`: the mode a backend was made in (a CPU fixture: no model file,
        // so its digest is unreadable, which doesn't touch the field).
        crate::kv::set_f32(false);
        let m = crate::model::tests::qwen3_fixture(crate::quant::F32);
        for n in [Numerics::Exact, Numerics::Fast] {
            let o = crate::lm::BackendOptions::default()
                .device(crate::lm::Device::Cpu)
                .max_ctx(64)
                .numerics(n);
            let net = Backend::new(&m, o).unwrap();
            let id = of("/nonexistent/axon-fixture.gguf", &m, &net);
            assert_eq!(id.fields[0], ("numerics", n.tag()), "{id:?}");
            assert_eq!(id.fingerprint(), "fp_unknown");
        }
    }

    /// Test file bytes: a fixed pseudo-random pattern (the golden values
    /// come from an independent Python/hashlib model of `digest_file`,
    /// scripts/identity_digest_ref.py).
    fn pattern(n: usize) -> Vec<u8> {
        (0..n as u64)
            .map(|i| (((i * 2654435761) & 0xffff_ffff) >> 13) as u8)
            .collect()
    }

    fn digest_of(name: &str, b: &[u8], off: u64, ts: &[(u64, u64)]) -> String {
        let dir = std::env::temp_dir().join(format!("axon-golden-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, b).unwrap();
        let d = digest_file(&p, off, ts).unwrap();
        std::fs::remove_file(&p).unwrap();
        d
    }

    /// Pinned digests: a change here changes every model's fingerprint
    /// (bump the "axon-model-digest/N" tag). A file over 4 MiB (sparse
    /// windows, gaps between them), tensors of 0, 100 and 300 bytes; data
    /// under 4 KiB; no data at all.
    #[test]
    fn model_digest_golden() {
        let len = 6 * 1024 * 1024 + 123;
        let big = pattern(len);
        let l = len as u64;
        let tb = [
            (777, 0),
            (777, 300),
            (1077, 100),
            (1177, 5_000_000),
            (5_001_177, l - 5_001_177),
        ];
        let r = digest_ranges(l, 777, &tb);
        let covered: u64 = r.iter().map(|(a, b)| b - a).sum();
        assert_eq!(covered, 4_195_462);
        assert!(r.windows(2).all(|w| w[0].1 < w[1].0), "sorted, merged");
        let d = digest_of("big", &big, 777, &tb);
        assert_eq!(
            d,
            "1e6f07123ce8b7cba2671873e0febf5e7e346ade633dff6a89499481e28a947c"
        );
        // A byte in a gap between windows isn't read; one in a window is.
        let gap = r[r.len() / 2].1 as usize;
        let mut b = big.clone();
        b[gap] ^= 1;
        assert_eq!(digest_of("big", &b, 777, &tb), d, "gap byte {gap}");
        let mut b = big.clone();
        b[gap - 1] ^= 1;
        assert_ne!(digest_of("big", &b, 777, &tb), d);
        let tiny = pattern(3000);
        assert_eq!(
            digest_of("tiny", &tiny, 100, &[(100, 0), (100, 2900)]),
            "d333bd1f9f5ce3d645d0d2ae9b88be75c9cbbe9bf39584439299a021f56a2356"
        );
        assert_eq!(
            digest_ranges(3000, 100, &[(100, 0), (100, 2900)]),
            [(0, 3000)]
        );
        assert_eq!(
            digest_of("nodata", &pattern(50), 50, &[]),
            "452b8b418068e72260f60eb9b41410324904a2246ed6ec4534a125e365f6f837"
        );
    }

    /// The field builders: a split names its streaming (prefill, decode),
    /// a GPU its code when it isn't the device's own cubin and its split
    /// count without tensor-core attention, a server its non-default
    /// settings; an unreadable part gives no fingerprint.
    #[test]
    fn fields() {
        assert_eq!(hybrid_backend(12, 24, None), "cuda+cpu, gpu layers 12/24");
        assert_eq!(
            hybrid_backend(12, 24, Some((32, false))),
            "cuda+cpu, gpu layers 12/24, prefill streams >=32"
        );
        assert_eq!(
            hybrid_backend(12, 24, Some((32, true))),
            "cuda+cpu, gpu layers 12/24, prefill streams >=32, decode streams"
        );
        assert_eq!(
            pipeline_backend(&[0, 10, 24]),
            "cuda pipeline, cuts 0,10,24"
        );
        assert_eq!(gpu_field("sm_89", "sm_89", None), "sm_89");
        assert_eq!(gpu_field("sm_90", "sm_90a", None), "sm_90 [code sm_90a]");
        assert_eq!(
            gpu_field("sm_120", "compute_90 PTX, driver 12.8", None),
            "sm_120 [code compute_90 PTX, driver 12.8]"
        );
        assert_eq!(
            gpu_field("sm_89", "sm_89", Some(64)),
            "sm_89, split attention 64"
        );
        let base = |g: &str| Identity::new("aa", "Q4_K:1", "cuda", Some(g), None, "f16");
        let fps: Vec<String> = [
            "sm_89",
            "sm_89, split attention 64",
            "sm_89, split attention 256",
        ]
        .iter()
        .map(|g| base(g).fingerprint())
        .collect();
        assert!(fps[0] != fps[1] && fps[1] != fps[2] && fps[0] != fps[2]);
        use crate::sample::SampleArgs;
        assert_eq!(serve_field(false, false, &SampleArgs::default()), None);
        let s = SampleArgs {
            temp: Some(0.7),
            seed: Some(5),
            ..SampleArgs::default()
        };
        assert_eq!(
            serve_field(true, false, &s).as_deref(),
            Some("no-think, temp 0.7, seed 5")
        );
        let plain = base("sm_89");
        assert_eq!(plain.clone().with_serve(None), plain);
        let srv = plain.clone().with_serve(serve_field(true, false, &s));
        assert_ne!(srv.fingerprint(), plain.fingerprint());
        assert!(
            srv.canonical()
                .ends_with("serve=no-think, temp 0.7, seed 5\n")
        );
        let unk = Identity::new("unknown (gone)", "Q4_K:1", "cpu", None, None, "f16");
        // The text is no sentinel: only `unreadable` withholds the fp.
        assert_ne!(unk.fingerprint(), "fp_unknown");
        assert_eq!(unk.clone().unreadable("model").fingerprint(), "fp_unknown");
        assert_eq!(base("sm_89").unreadable("gpu").fingerprint(), "fp_unknown");
    }

    /// `of` on a real CPU/GPU split and a two-stage pipeline (both stages on
    /// device 0): the backend and gpu fields as `fields` builds them.
    #[test]
    fn of_split_and_pipeline() {
        let Some(g0) = crate::testutil::gpu() else {
            return;
        };
        drop(g0);
        let Some(mp) = crate::testutil::model_path() else {
            return;
        };
        let path = mp.to_str().unwrap();
        let m = crate::gguf::load_with(path, Some(false)).unwrap();
        let n = crate::hybrid::block_count(&m).unwrap();
        let kv = crate::kv::KvType::F16;
        let net = Backend::new(
            &m,
            crate::lm::BackendOptions::default()
                .device(crate::lm::Device::Cuda)
                .max_ctx(256)
                .gpu_layers(Some(n / 2))
                .kv(kv),
        )
        .unwrap();
        let Inner::Hybrid(h) = &net.0 else {
            panic!("not a split")
        };
        let id = of(path, &m, &net);
        let get = |id: &Identity, k: &str| {
            (id.fields.iter())
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.clone())
        };
        let b = get(&id, "backend").unwrap();
        assert_eq!(b, hybrid_backend(n / 2, n, h.streams()), "{id:?}");
        assert!(b.starts_with(&format!("cuda+cpu, gpu layers {}/{n}", n / 2)));
        assert!(get(&id, "cpu").is_some() && get(&id, "gpu").unwrap().starts_with("sm_"));
        assert_ne!(id.fingerprint(), "fp_unknown");
        drop(net);
        let gpu: &'static crate::cuda::Gpu =
            Box::leak(Box::new(crate::cuda::Gpu::open_device(0).unwrap()));
        let module: &'static crate::cuda::Module<'static> =
            Box::leak(Box::new(crate::cuda::Kernels::module_for(gpu, &m).unwrap()));
        let devs = [(gpu, module), (gpu, module)];
        let cuts = vec![0, n / 2, n];
        let p = crate::cuda::qwen2::pipeline::PipelineQwen2::new(
            &devs,
            &m,
            256,
            crate::cuda::qwen2::CacheKv::F16,
            Some(cuts.clone()),
        )
        .unwrap();
        let net = Backend(Inner::Pipeline(Box::new(p)), Numerics::Exact);
        let id = of(path, &m, &net);
        assert_eq!(get(&id, "backend").unwrap(), pipeline_backend(&cuts));
        let g = get(&id, "gpu").unwrap();
        let parts: Vec<&str> = g.split("; ").collect();
        assert_eq!(parts.len(), 2, "{g}");
        assert_eq!(parts[0], parts[1]);
        assert_eq!(get(&id, "kv").unwrap(), "f16");
    }

    /// B2: Gemma 3 (head size 256) runs the split attention on every GPU:
    /// its `gpu` field carries `max_split` for this card's SM count, `axon
    /// info` predicts the same fingerprint gen gets, and no q8 one.
    #[test]
    fn of_split_attention_gemma3() {
        let Some(g) = crate::testutil::gpu() else {
            return;
        };
        let Some(gp) = crate::testutil::gemma3_model_path() else {
            return;
        };
        let sms = g.info().unwrap().sms as usize;
        drop(g);
        let path = gp.to_str().unwrap();
        let m = crate::gguf::load_with(path, Some(false)).unwrap();
        let kv = crate::kv::KvType::F16;
        let net = Backend::new(
            &m,
            crate::lm::BackendOptions::default()
                .device(crate::lm::Device::Cuda)
                .max_ctx(256)
                .kv(kv),
        )
        .unwrap();
        let Inner::Cuda(n) = &net.0 else {
            panic!("not on the GPU")
        };
        let k = crate::cuda::qwen2::max_split(sms, m.n_kv_heads() as usize);
        assert_eq!(n.split_attention(), Some(k));
        let id = of(path, &m, &net);
        let gf = &id.fields.iter().find(|(f, _)| *f == "gpu").unwrap().1;
        assert!(gf.ends_with(&format!(", split attention {k}")), "{gf}");
        let p = predicted(path, &m).unwrap();
        let f16 = p.iter().find(|(l, _)| l == "cuda --kv f16").unwrap();
        assert_eq!(f16.1.as_ref().unwrap().fingerprint(), id.fingerprint());
        let q8 = p.iter().find(|(l, _)| l == "cuda --kv q8").unwrap();
        assert!(q8.1.is_err());
    }

    /// `rope`: --rope-scaling yarn (and another --rope-scale) changes the
    /// fingerprint, though the file and its digest don't; --rope-scaling
    /// none on a plain model is the same as no flag; `predicted` (axon
    /// info with the same flags) gives the fingerprint `of` does.
    #[test]
    fn rope_scaling_in_the_fingerprint() {
        let Some(mp) = crate::testutil::model_path() else {
            return;
        };
        let path = mp.to_str().unwrap();
        let kv = crate::kv::KvType::F16;
        let run = |r: crate::rope::RopeArgs| {
            let mut m = crate::gguf::load_with(path, Some(false)).unwrap();
            r.apply(&mut m).unwrap();
            let net = Backend::new(
                &m,
                crate::lm::BackendOptions::default()
                    .device(crate::lm::Device::Cpu)
                    .max_ctx(64)
                    .kv(kv),
            )
            .unwrap();
            let id = of(path, &m, &net);
            let p = predicted(path, &m).unwrap().remove(0);
            assert_eq!(p.0, "cpu");
            assert_eq!(p.1.unwrap(), id, "axon info predicts gen's");
            (id, rope_field(&m))
        };
        let yarn = |f: f64| crate::rope::RopeArgs {
            scaling: Some("yarn".into()),
            factor: Some(f),
            ..Default::default()
        };
        let (plain, r0) = run(Default::default());
        assert_eq!(r0, None, "plain RoPE: no rope field");
        assert!(!plain.canonical().contains("rope="));
        let (none, _) = run(crate::rope::RopeArgs {
            scaling: Some("none".into()),
            ..Default::default()
        });
        assert_eq!(none.fingerprint(), plain.fingerprint());
        let (y4, r4) = run(yarn(4.0));
        let (y2, _) = run(yarn(2.0));
        let r4 = r4.unwrap();
        assert!(r4.starts_with("yarn factor 4 orig 32768 "), "{r4}");
        assert!(r4.contains("freq_base "), "{r4}");
        let field = |id: &Identity, k: &str| {
            id.fields
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(field(&y4, "model"), field(&plain, "model"), "same file");
        let fps = [plain.fingerprint(), y4.fingerprint(), y2.fingerprint()];
        assert!(
            fps[0] != fps[1] && fps[1] != fps[2] && fps[0] != fps[2],
            "{fps:?}"
        );
    }

    /// The model digest reads the file: stable, and it sees a change in the
    /// header, at a tensor's edge and in the data windows.
    #[test]
    fn model_digest_sees_the_file() {
        let dir = std::env::temp_dir().join(format!("axon-digest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.gguf");
        let bytes: Vec<u8> = (0..3_000_000u32).map(|i| (i ^ (i >> 9)) as u8).collect();
        let (off, ts) = (1000u64, [(1000u64, 1_000_000u64), (1_000_000, 1_999_000)]);
        let d = |b: &[u8]| {
            std::fs::write(&p, b).unwrap();
            digest_file(&p, off, &ts).unwrap()
        };
        let d0 = d(&bytes);
        assert_eq!(d0.len(), 64);
        assert_eq!(d0, d(&bytes), "stable");
        for at in [10usize, 999, 1000, 999_999, 1_000_000, 1_500_000, 2_999_999] {
            let mut b = bytes.clone();
            b[at] ^= 1;
            assert_ne!(d(&b), d0, "byte {at}");
        }
        let mut b = bytes.clone();
        b.push(0);
        assert_ne!(d(&b), d0, "length");
        // A real model: two reads agree.
        if let Some(mp) = crate::testutil::model_path() {
            let m = crate::gguf::load(mp.to_str().unwrap()).unwrap();
            let a = model_digest(&mp, &m).unwrap();
            assert_eq!(a, model_digest(&mp, &m).unwrap());
            assert_ne!(a, d0);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
