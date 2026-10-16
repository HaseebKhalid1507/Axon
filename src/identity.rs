//! The identity tuple: everything that decides Axon's output bits, hashed
//! into one fingerprint (OpenAI's `system_fingerprint`). Two runs with the
//! same fingerprint, prompt ids, sampling params and seed give the same
//! tokens, bit for bit, however the requests were batched, chunked, cached
//! or drafted (docs/determinism.md states the contract and its limits).
//!
//! In the tuple (`Identity::fields`, in this order):
//! - `numerics`: `NUMERICS_VERSION`, Axon's own kernels and sampler.
//! - `model`: `model_digest` of the GGUF file (its whole header, so every
//!   metadata key, tensor name, shape and quant type, plus a sample of the
//!   tensor data), and `types`, the quant types (readable; the header hash
//!   already covers them).
//! - `backend`: cpu, cuda, cuda+cpu (with the layer split), or a pipeline
//!   (with its cuts): a layer's numerics are its device's.
//! - `gpu`: per GPU, its exact architecture (sm_XY). Audited (see
//!   docs/determinism.md): two sm_89 cards with different NVRTC (12.8,
//!   13.3) and drivers agree bit for bit, so the toolkit isn't in it; sm_61
//!   and sm_89 differ from prefill on (split f32 attention below sm_80,
//!   tensor cores from sm_80); other pairs are unaudited, so each sm_XY is
//!   its own.
//! - `cpu`: what decides CPU bits (fused multiply-add or not, r8 repack or
//!   per-row kernels) when any layer runs on the CPU.
//! - `kv`: the KV cache type (f16, q8, q8/32; f32 on the CPU).
//!
//! Not in it, because they never change bits: drafting (lossless, prompt
//! lookup or a draft model), the context size (--ctx or serve's default,
//! the trained length where it fits: attention runs over the positions
//! filled, in 64-position blocks, so a bigger cache only lets a request run
//! longer and never changes a token below the limit), --parallel and the
//! slot count, chunked prefill, thread counts, CUDA graphs and PDL. Sampling
//! params and the seed are per request, so they're the request's, not the
//! server's. A future flag that does change bits (a RoPE scaling override,
//! say) must add a field here, set only when the flag is.
//!
//! The model digest reads the file itself (never the loaded copy, which
//! may have been rewritten or released), so it's a function of the file's
//! bytes alone.

use crate::gguf::Model;
use crate::lm::Backend;
use std::io;
use std::path::Path;

/// Axon's numerics version. RULE: bump it in the same commit as any change
/// that can change an output bit for some model on some backend: a kernel's
/// arithmetic or reduction order (CPU or CUDA), the attention fold, RoPE,
/// norms, the sampler's order or RNG, a quant format's dequant. Changes that
/// keep every bit (speed, scheduling, a kernel that matches the old one bit
/// for bit, proven by the md5 gate) don't bump it. A bump changes every
/// fingerprint, so a client caching on it knows its old outputs are stale.
pub const NUMERICS_VERSION: u32 = 1;

/// The tuple's format, the first line of `canonical`.
const SCHEME: &str = "axon-identity/1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// (name, value), in the order the module doc lists them.
    pub fields: Vec<(&'static str, String)>,
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
            ("numerics", NUMERICS_VERSION.to_string()),
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
        Identity { fields: f }
    }

    /// What the fingerprint hashes: the scheme, then `name=value` lines.
    pub fn canonical(&self) -> String {
        let mut s = format!("{SCHEME}\n");
        for (k, v) in &self.fields {
            s.push_str(&format!("{k}={v}\n"));
        }
        s
    }

    /// `fp_` and the first 16 hex digits of the SHA-256 of `canonical`.
    pub fn fingerprint(&self) -> String {
        format!("fp_{}", &hex(&sha256(self.canonical().as_bytes()))[..16])
    }

    /// The tuple as a JSON object, the fingerprint included (GET /identity).
    pub fn json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("system_fingerprint".into(), self.fingerprint().into());
        m.insert("scheme".into(), SCHEME.into());
        for (k, v) in &self.fields {
            let v = match *k {
                "numerics" => serde_json::Value::from(NUMERICS_VERSION),
                _ => v.clone().into(),
            };
            m.insert((*k).into(), v);
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

/// The identity of `net`, loaded from the GGUF file at `path` (`m`).
pub fn of(path: &str, m: &Model, net: &Backend) -> io::Result<Identity> {
    let digest = model_digest(Path::new(path), m)?;
    let types = quant_types(m);
    let cpu_kv = if crate::kv::use_f32() { "f32" } else { "f16" };
    let id = match net {
        Backend::Cpu(_) => Identity::new(&digest, &types, "cpu", None, Some(&cpu()), cpu_kv),
        Backend::Cuda(n) => {
            let g = gpu(n.gpu());
            Identity::new(&digest, &types, "cuda", Some(&g), None, n.kv_name())
        }
        Backend::Hybrid(n) => {
            let g = n.gpu_net().map(|g| gpu(g.gpu()));
            let b = format!("cuda+cpu, gpu layers {}/{}", n.gpu_layers, m.n_layers());
            let kv = format!("{} (cpu layers {cpu_kv})", n.kv_name());
            Identity::new(&digest, &types, &b, g.as_deref(), Some(&cpu()), &kv)
        }
        Backend::Pipeline(n) => {
            let cuts: Vec<String> = n.cuts.iter().map(|c| c.to_string()).collect();
            let gs: Vec<String> = n.gpus().into_iter().map(gpu).collect();
            let b = format!("cuda pipeline, cuts {}", cuts.join(","));
            Identity::new(&digest, &types, &b, Some(&gs.join("; ")), None, n.kv_name())
        }
    };
    Ok(id)
}

/// The identities `m` (at `path`) would have without loading it (`axon
/// info`): on the CPU (its cache type as AXON_KV says) and, if a GPU can
/// run it, on CUDA device 0 with each cache type, as (label, identity).
pub fn predicted(path: &str, m: &Model) -> io::Result<Vec<(String, Identity)>> {
    let digest = model_digest(Path::new(path), m)?;
    let types = quant_types(m);
    let cpu_kv = if crate::kv::use_f32() { "f32" } else { "f16" };
    let mut v = vec![(
        "cpu".to_string(),
        Identity::new(&digest, &types, "cpu", None, Some(&cpu()), cpu_kv),
    )];
    if let Ok(g) = crate::cuda::Gpu::open() {
        let gs = gpu(&g);
        for kv in ["f16", "q8"] {
            let id = Identity::new(&digest, &types, "cuda", Some(&gs), None, kv);
            v.push((format!("cuda --kv {kv}"), id));
        }
    }
    Ok(v)
}

/// A GPU's part: its architecture.
fn gpu(g: &crate::cuda::Gpu) -> String {
    g.info().map_or_else(|_| "unknown".into(), |i| i.arch())
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

/// The model's digest: SHA-256 (hex) over the file length, the whole header
/// (metadata and tensor table, up to the data), the first and last
/// `TENSOR_EDGE` bytes of every tensor, and `WINDOWS` windows of `WINDOW`
/// bytes evenly spaced over the data: a few MB read, ~40 ms warm (7B), against
/// seconds for every byte of a 5 GB file on every start. Another model, a
/// requant, a fine-tune or a merge changes the header or (nearly) every
/// block, so the samples see it; a deliberate edit of bytes between
/// samples would not (this names a file; it isn't a tamper seal).
pub fn model_digest(path: &Path, m: &Model) -> io::Result<String> {
    let tensors: Vec<(u64, u64)> = (m.tensors.iter())
        .map(|t| (m.data_offset + t.offset, t.n_bytes))
        .collect();
    digest_file(path, m.data_offset, &tensors)
}

/// `model_digest` for a file whose data starts at `data_offset`, with
/// tensors at (absolute offset, length).
fn digest_file(path: &Path, data_offset: u64, tensors: &[(u64, u64)]) -> io::Result<String> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let mut h = Sha256::new();
    h.update(b"axon-model-digest/1\0");
    h.update(&len.to_le_bytes());
    let mut buf = Vec::new();
    let mut read = |h: &mut Sha256, at: u64, n: u64| -> io::Result<()> {
        let n = n.min(len.saturating_sub(at));
        buf.resize(n as usize, 0);
        f.read_exact_at(&mut buf, at)?;
        h.update(&buf);
        Ok(())
    };
    read(&mut h, 0, data_offset)?;
    for &(at, n) in tensors {
        read(&mut h, at, n.min(TENSOR_EDGE))?;
        let e = n.min(TENSOR_EDGE);
        read(&mut h, at + n - e, e)?;
    }
    let data = len.saturating_sub(data_offset);
    if data > 0 {
        let span = data.saturating_sub(WINDOW);
        for i in 0..WINDOWS {
            read(&mut h, data_offset + span * i / (WINDOWS - 1), WINDOW)?;
        }
    }
    Ok(hex(&h.finish()))
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
        // which needs a NUMERICS_VERSION or SCHEME reason.
        assert_eq!(
            a.canonical(),
            "axon-identity/1\nnumerics=1\nmodel=aa\ntypes=Q4_K:1\nbackend=cuda\n\
             gpu=sm_120\nkv=f16\n"
        );
        let g = Some("sm_120");
        let variants = [
            Identity::new("aa", "Q4_K:1", "cuda", g, None, "q8"),
            Identity::new("ab", "Q4_K:1", "cuda", g, None, "f16"),
            Identity::new("aa", "Q4_K:1", "cpu", None, Some("x86_64 fma, r8"), "f16"),
            Identity::new("aa", "Q4_K:1", "cuda", Some("sm_89"), None, "f16"),
            Identity::new("aa", "Q4_K:1", "cuda", Some("sm_61"), None, "f16"),
            Identity::new("aa", "Q6_K:1", "cuda", g, None, "f16"),
        ];
        let mut seen = vec![a.fingerprint()];
        for v in &variants {
            let f = v.fingerprint();
            assert!(!seen.contains(&f), "{v:?} collides");
            seen.push(f);
        }
        let mut n = base();
        n.fields[0].1 = "2".into();
        assert_ne!(n.fingerprint(), a.fingerprint(), "numerics version");
        let j = a.json();
        assert_eq!(j["system_fingerprint"], a.fingerprint());
        assert_eq!(j["numerics"], NUMERICS_VERSION);
        assert_eq!(j["kv"], "f16");
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
