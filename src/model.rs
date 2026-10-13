//! qwen2 forward pass. Matrix-vector products quantize the input to 8 bits and
//! run integer dot products on the worker pool (pool.rs); everything else is
//! plain f32 loops.

use crate::gguf::{Model, Tensor, Val};
use crate::pool::{self, Pool};
use crate::quant::{
    DecodeCfg, F16, F32, Q4_K, Q5_0, Q6_K, Q8_0, QuantizedAct, batch_slots, dequant,
    dot_row_q8_multi, dot_rows_q8, dot_rows_q8_multi, r8, row_bytes,
};

/// A weight matrix: its bytes in the model's mapping, or its 8-row
/// interleaved repack (quant::r8). A repacked matrix goes through the r8
/// kernels in every path (decode, prefill, swiglu, the verify head), so all
/// of them give the same bits; the others use the per-row kernels.
struct Weight<'a> {
    t: &'a Tensor,
    /// The tensor's bytes; None when repacked (they may have been released).
    data: Option<&'a [u8]>,
    r8: Option<memmap2::Mmap>,
}

impl<'a> Weight<'a> {
    fn qt(&self) -> u32 {
        self.t.qtype.id
    }

    fn rows(&self) -> usize {
        self.t.dims[1] as usize
    }

    /// Repack a Q4_K/Q6_K matrix with rows % 8 == 0 if `repack`, into
    /// anonymous memory with transparent huge pages requested (decode walks
    /// it every token; AXON_R8_THP=0 skips the request). Then, if `release`,
    /// the original bytes are handed back (`Model::release`) so the weights
    /// aren't resident twice. Never release a tensor read elsewhere: the
    /// token embedding (a tied output matrix is the same tensor).
    fn load(m: &'a Model, t: &'a Tensor, repack: bool, release: bool) -> Weight<'a> {
        let (qt, rows, n) = (t.qtype.id, t.dims[1] as usize, t.dims[0] as usize);
        if !repack || !r8::can_repack(qt, rows, n) {
            return Weight {
                t,
                data: Some(m.tensor_data(t)),
                r8: None,
            };
        }
        let len = r8::repacked_len(qt, rows, n);
        let mut buf = memmap2::MmapMut::map_anon(len).expect("r8: anonymous mapping");
        #[cfg(target_os = "linux")]
        if std::env::var("AXON_R8_THP").map_or(true, |v| v != "0") {
            // Before the first touch, as in gguf::map_weights. Failure only costs speed.
            let _ = buf.advise(memmap2::Advice::HugePage);
        }
        r8::repack_into(qt, m.tensor_data(t), rows, n, &mut buf);
        if release {
            m.release(t);
        }
        Weight {
            t,
            data: None,
            r8: Some(buf.make_read_only().expect("r8: mprotect")),
        }
    }

    /// The tensor's bytes (not repacked).
    fn bytes(&self) -> &'a [u8] {
        self.data.expect("repacked weight has no raw bytes")
    }

    /// Rows r0..r0 + k of a repacked matrix as whole r8 groups, plus the
    /// rest of the matrix when `to_end` (prefetch room, see `r8::gemv`).
    fn groups(&self, n_in: usize, r0: usize, k: usize, to_end: bool) -> &[u8] {
        let p = self.r8.as_deref().expect("not repacked");
        assert!(
            r0.is_multiple_of(8) && k.is_multiple_of(8),
            "{}: r8 rows not whole groups",
            self.t.name
        );
        let gb = r8::group_bytes(self.qt(), n_in);
        let end = if to_end { p.len() } else { (r0 + k) / 8 * gb };
        &p[r0 / 8 * gb..end]
    }
}

/// The weights of one transformer block, borrowed from the mapping or repacked.
struct Layer<'a> {
    attn_norm: Vec<f32>,
    wq: Weight<'a>,
    wk: Weight<'a>,
    wv: Weight<'a>,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    wo: Weight<'a>,
    ffn_norm: Vec<f32>,
    w_gate: Weight<'a>,
    w_up: Weight<'a>,
    w_down: Weight<'a>,
}

pub struct Qwen2<'a> {
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    eps: f32,
    rope_base: f64,
    token_embd: &'a Tensor,
    /// token_embd's bytes (never released: see `Weight::load`)
    embd: &'a [u8],
    output_norm: Vec<f32>,
    output: Weight<'a>,
    layers: Vec<Layer<'a>>,
}

/// Keys and values of every past position, per layer.
/// k[layer] is laid out as [ctx][n_kv_heads * head_dim], same for v.
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kv_dim: usize,
    max_ctx: usize,
}

fn meta_num(m: &Model, key: &str) -> Option<f64> {
    m.metadata
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| match v {
            Val::U32(n) => Some(*n as f64),
            Val::I32(n) => Some(*n as f64),
            Val::U64(n) => Some(*n as f64),
            Val::I64(n) => Some(*n as f64),
            Val::F32(n) => Some(*n as f64),
            Val::F64(n) => Some(*n),
            _ => None,
        })
}

fn find<'a>(m: &'a Model, name: &str) -> &'a Tensor {
    m.tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("missing tensor {name}"))
}

fn find_opt<'a>(m: &'a Model, name: &str) -> Option<&'a Tensor> {
    m.tensors.iter().find(|t| t.name == name)
}

/// Fully dequantize a small tensor (norm weights, biases) into f32.
fn load_vec(m: &Model, t: &Tensor) -> Vec<f32> {
    let mut out = vec![0.0f32; t.n_elements as usize];
    dequant(t.qtype.id, m.tensor_data(t), &mut out);
    out
}

/// Tensor types the kernels run (`quant::dequant` and `dot_row_q8`).
const SUPPORTED_TYPES: [u32; 6] = [F32, F16, Q5_0, Q8_0, Q4_K, Q6_K];

/// Everything `Qwen2::new` and `Tokenizer::from_gguf` rely on, checked up front
/// so a bad or unsupported model is an error instead of a panic: the
/// architecture, the hyperparameters, every tensor's presence, type and shape,
/// and the vocabulary. Reads metadata only.
pub fn check_supported(m: &Model) -> std::io::Result<()> {
    let bad = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    if m.arch != "qwen2" {
        return Err(bad(format!(
            "architecture '{}' is not supported (only qwen2 for now)",
            if m.arch.is_empty() { "<none>" } else { &m.arch }
        )));
    }
    let need = |k: &str| -> std::io::Result<usize> {
        let key = format!("qwen2.{k}");
        match meta_num(m, &key) {
            Some(v) if v.fract() == 0.0 && v >= 1.0 && v <= u32::MAX as f64 => Ok(v as usize),
            Some(v) => Err(bad(format!(
                "metadata {key} = {v} is not a positive integer"
            ))),
            None => Err(bad(format!("metadata {key} is missing or not a number"))),
        }
    };
    let n_layers = need("block_count")?;
    let dim = need("embedding_length")?;
    let n_heads = need("attention.head_count")?;
    let n_kv_heads = need("attention.head_count_kv")?;
    if !dim.is_multiple_of(n_heads) || !(dim / n_heads).is_multiple_of(2) {
        return Err(bad(format!(
            "embedding_length {dim} doesn't split into {n_heads} heads of even size"
        )));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(bad(format!(
            "head_count {n_heads} is not a multiple of head_count_kv {n_kv_heads}"
        )));
    }
    let kv_dim = n_kv_heads * (dim / n_heads);

    // `rows` = None: a vector of `cols` elements; Some(r): a cols x r matrix.
    let check = |name: &str, cols: usize, rows: Option<usize>| -> std::io::Result<&Tensor> {
        let t = find_opt(m, name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        let q = t.qtype;
        if !SUPPORTED_TYPES.contains(&q.id) {
            return Err(bad(format!(
                "tensor {name}: type {} ({}) is not supported (F32, F16, Q5_0, Q8_0, Q4_K, Q6_K)",
                q.name, q.id
            )));
        }
        let ok = match rows {
            None => t.n_elements == cols as u64,
            Some(r) => {
                t.dims.len() >= 2
                    && t.dims[0] == cols as u64
                    && (if r == 0 {
                        t.dims[1] > 0
                    } else {
                        t.dims[1] == r as u64
                    })
                    && t.dims[2..].iter().all(|&d| d == 1)
            }
        };
        if !ok
            || t.dims
                .first()
                .is_some_and(|&d| d % q.block_size as u64 != 0)
        {
            return Err(bad(format!(
                "tensor {name}: shape {:?} doesn't fit the model ({cols} x {})",
                t.dims,
                rows.map_or("1".into(), |r| if r == 0 {
                    "n".into()
                } else {
                    r.to_string()
                })
            )));
        }
        Ok(t)
    };

    let embd = check("token_embd.weight", dim, Some(0))?;
    let vocab = embd.dims[1] as usize;
    if find_opt(m, "output.weight").is_some() {
        check("output.weight", dim, Some(vocab))?;
    }
    check("output_norm.weight", dim, None)?;
    for i in 0..n_layers {
        let t = |s: &str, cols, rows| check(&format!("blk.{i}.{s}"), cols, rows);
        t("attn_norm.weight", dim, None)?;
        t("attn_q.weight", dim, Some(dim))?;
        t("attn_k.weight", dim, Some(kv_dim))?;
        t("attn_v.weight", dim, Some(kv_dim))?;
        for (b, n) in [
            ("attn_q.bias", dim),
            ("attn_k.bias", kv_dim),
            ("attn_v.bias", kv_dim),
        ] {
            if find_opt(m, &format!("blk.{i}.{b}")).is_some() {
                t(b, n, None)?;
            }
        }
        t("attn_output.weight", dim, Some(dim))?;
        t("ffn_norm.weight", dim, None)?;
        let ffn = t("ffn_gate.weight", dim, Some(0))?.dims[1] as usize;
        t("ffn_up.weight", dim, Some(ffn))?;
        t("ffn_down.weight", ffn, Some(dim))?;
    }

    match m
        .metadata
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.tokens")
    {
        Some((_, Val::Arr(a))) if !a.is_empty() && a.len() <= vocab => Ok(()),
        Some((_, Val::Arr(a))) => Err(bad(format!(
            "tokenizer has {} tokens for an embedding of {vocab} rows",
            a.len()
        ))),
        _ => Err(bad("tokenizer.ggml.tokens is missing".into())),
    }
}

impl<'a> Qwen2<'a> {
    /// The model, with Q4_K/Q6_K matrices repacked for the r8 kernels unless
    /// AXON_REPACK=0 (the per-row kernels only: the branch's bits).
    pub fn new(m: &'a Model) -> Qwen2<'a> {
        Self::new_with(m, std::env::var("AXON_REPACK").map_or(true, |v| v != "0"))
    }

    pub fn new_with(m: &'a Model, repack: bool) -> Qwen2<'a> {
        Self::build(m, repack, false)
    }

    /// `tie`: use the token embedding as the output matrix even if the model
    /// has its own (tests: the tied-embedding case on any model).
    fn build(m: &'a Model, repack: bool, tie: bool) -> Qwen2<'a> {
        if let Err(e) = check_supported(m) {
            panic!("{e}");
        }
        let a = &m.arch;
        let key = |k: &str| format!("{a}.{k}");
        let need =
            |k: &str| meta_num(m, &key(k)).unwrap_or_else(|| panic!("missing metadata {}", key(k)));

        let n_layers = need("block_count") as usize;
        let dim = need("embedding_length") as usize;
        let n_heads = need("attention.head_count") as usize;
        let n_kv_heads = need("attention.head_count_kv") as usize;
        let head_dim = dim / n_heads;
        let eps = meta_num(m, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f32;
        let rope_base = meta_num(m, &key("rope.freq_base")).unwrap_or(10000.0);

        let token_embd = find(m, "token_embd.weight");
        // ne1 of the embedding matrix is the vocab size.
        let vocab = token_embd.dims[1] as usize;
        // Some models tie output to the embedding; qwen2.5-0.5b ships a separate one.
        let out_t = match find_opt(m, "output.weight") {
            Some(t) if !tie => t,
            _ => token_embd,
        };
        // A tied output matrix is the embedding, which `embed` keeps reading.
        let output = Weight::load(m, out_t, repack, !std::ptr::eq(out_t, token_embd));
        let output_norm = load_vec(m, find(m, "output_norm.weight"));

        // One pool task per layer (`floats` set so it never runs inline):
        // repacking is a pass over all the weights.
        let layers = par_map(n_layers, INLINE_WORK + 1, |i| {
            let t = |s: &str| find(m, &format!("blk.{i}.{s}"));
            let w = |s: &str| Weight::load(m, t(s), repack, true);
            let bias = |s: &str| find_opt(m, &format!("blk.{i}.{s}")).map(|t| load_vec(m, t));
            Layer {
                attn_norm: load_vec(m, t("attn_norm.weight")),
                wq: w("attn_q.weight"),
                wk: w("attn_k.weight"),
                wv: w("attn_v.weight"),
                bq: bias("attn_q.bias"),
                bk: bias("attn_k.bias"),
                bv: bias("attn_v.bias"),
                wo: w("attn_output.weight"),
                ffn_norm: load_vec(m, t("ffn_norm.weight")),
                w_gate: w("ffn_gate.weight"),
                w_up: w("ffn_up.weight"),
                w_down: w("ffn_down.weight"),
            }
        });

        Qwen2 {
            n_layers,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab,
            eps,
            rope_base,
            token_embd,
            embd: m.tensor_data(token_embd),
            output_norm,
            output,
            layers,
        }
    }

    pub fn new_cache(&self, max_ctx: usize) -> KvCache {
        let kv_dim = self.n_kv_heads * self.head_dim;
        KvCache {
            k: (0..self.n_layers)
                .map(|_| vec![0.0; max_ctx * kv_dim])
                .collect(),
            v: (0..self.n_layers)
                .map(|_| vec![0.0; max_ctx * kv_dim])
                .collect(),
            kv_dim,
            max_ctx,
        }
    }

    /// Decode kernel settings (read once per matvec), and whether one thread
    /// runs every chunk: then prefetch may run on into the next chunk, which
    /// is its own work too.
    fn decode_cfg() -> (DecodeCfg, bool) {
        (DecodeCfg::get(), Pool::global().threads() == 1)
    }

    /// y = W x. W has ne1 rows of ne0 values each; y[o] = row_o · x.
    fn matvec(&self, w: &Weight, x: &[f32]) -> Vec<f32> {
        let [y] = self.matvec_many([w], x);
        y
    }

    /// Joint row space of `ws` for input width `n_in`: starts[i] is where
    /// matrix i begins, plus the total and the row granularity pool chunks
    /// must keep: 8 (whole r8 groups) if any matrix is repacked, every start
    /// then a multiple of 8; else 1.
    fn joint_rows<const N: usize>(ws: [&Weight; N], n_in: usize) -> ([usize; N], usize, usize) {
        let mut starts = [0usize; N];
        let mut total = 0;
        for (i, w) in ws.iter().enumerate() {
            assert_eq!(w.t.dims[0] as usize, n_in, "matmul {}: input len", w.t.name);
            starts[i] = total;
            total += w.rows();
        }
        let align = if ws.iter().any(|w| w.r8.is_some()) {
            8
        } else {
            1
        };
        assert!(
            starts.iter().all(|s| s % align == 0),
            "repacked matrices must start at a multiple of 8 rows"
        );
        (starts, total, align)
    }

    /// Decode rows r0..r0 + out.len() of `w` · x. Prefetch stays inside the
    /// rows the caller owns, or runs to the matrix's end when `own_all`.
    fn rows_vec(
        &self,
        w: &Weight,
        r0: usize,
        x: &QuantizedAct,
        out: &mut [f32],
        cfg: DecodeCfg,
        own_all: bool,
    ) {
        let k = out.len();
        if w.r8.is_some() {
            let groups = w.groups(x.len(), r0, k, own_all);
            r8::gemv(w.qt(), groups, x, out, cfg.prefetch);
        } else {
            let (qt, data) = (w.qt(), w.bytes());
            let rb = row_bytes(qt, x.len());
            let lim = if own_all { data.len() } else { (r0 + k) * rb };
            dot_rows_q8(qt, &data[r0 * rb..lim], x, out, cfg);
        }
    }

    /// Several matrices applied to the same input in one parallel pass:
    /// one fork/join and one quantization of x instead of one per matrix.
    /// Returns one output vector per matrix, in order.
    fn matvec_many<const N: usize>(&self, ws: [&Weight; N], x: &[f32]) -> [Vec<f32>; N] {
        // Quantize x to 8 bits once; every row then does integer multiply-adds
        // against its packed weights instead of decoding them to floats.
        let types = ws.map(|w| w.qt());
        let xq = QuantizedAct::for_types(x, &types);
        // All matrices' rows form one index space; starts[i] is where matrix i begins.
        let (starts, total, align) = Self::joint_rows(ws, x.len());
        // Every output row is an independent dot product, so split rows across
        // cores. About 4 chunks per thread lets the fast and efficiency cores
        // even out; at most 64 rows keeps chunk claiming cheap.
        let mut out = vec![0f32; total];
        let chunk = (total / (Pool::global().threads() * 4))
            .clamp(4, 64)
            .next_multiple_of(align);
        let (cfg, own_all) = Self::decode_cfg();
        pool::for_each_chunk_mut(&mut out, chunk, |c, mut ys| {
            // The chunk's rows, one run per matrix it touches, several rows
            // per kernel call.
            let mut g = c * chunk;
            while !ys.is_empty() {
                let t = starts.iter().rposition(|&s| s <= g).unwrap();
                let end = starts.get(t + 1).copied().unwrap_or(total);
                let k = (end - g).min(ys.len());
                let (run, rest) = std::mem::take(&mut ys).split_at_mut(k);
                self.rows_vec(ws[t], g - starts[t], &xq, run, cfg, own_all);
                g += k;
                ys = rest;
            }
        });
        // Split the joint output back into one vector per matrix (no copy when N == 1).
        let mut res: [Vec<f32>; N] = std::array::from_fn(|_| Vec::new());
        for i in (1..N).rev() {
            res[i] = out.split_off(starts[i]);
        }
        res[0] = out;
        res
    }

    /// silu(gate x) * (up x), row by row: each task computes row r of both
    /// matrices and combines them, so the exp() runs on every core instead of
    /// in a serial pass after the matvec.
    fn matvec_swiglu(&self, gate: &Weight, up: &Weight, x: &[f32]) -> Vec<f32> {
        let n = gate.rows();
        let (_, _, a0) = Self::joint_rows([gate], x.len());
        let (_, _, a1) = Self::joint_rows([up], x.len());
        assert_eq!(up.rows(), n, "swiglu: gate/up row counts differ");
        let xq = QuantizedAct::for_types(x, &[gate.qt(), up.qt()]);
        let mut out = vec![0f32; n];
        // Each row is two dot products, so half the rows per chunk of matvec_many.
        const MAX_CHUNK: usize = 32;
        let chunk = (n / (Pool::global().threads() * 8))
            .clamp(4, MAX_CHUNK)
            .next_multiple_of(a0.max(a1));
        let (cfg, own_all) = Self::decode_cfg();
        pool::for_each_chunk_mut(&mut out, chunk, |c, ys| {
            let (r, k) = (c * chunk, ys.len());
            let mut ub = [0f32; MAX_CHUNK];
            let ub = &mut ub[..k];
            self.rows_vec(gate, r, &xq, ys, cfg, own_all);
            self.rows_vec(up, r, &xq, ub, cfg, own_all);
            for (y, &uv) in ys.iter_mut().zip(ub.iter()) {
                let gv = *y;
                *y = (gv / (1.0 + (-gv).exp())) * uv;
            }
        });
        out
    }

    /// Rows per pool chunk for a batched matmul over `nb` tokens: an even
    /// count so the 2-row tile kernels never get a lone row at a chunk edge.
    /// ~32 chunks per thread for prefill-sized batches: a chunk is a lot of
    /// work (every row times every token), so a few big chunks leave threads
    /// idle while the last one finishes, and claiming is cheap next to it.
    /// Sweep on bella (560-token prompt): 4/thread ~611 tok/s, 32 ~646, 64+
    /// slower again. A small batch (a speculative verify, 2..9 tokens) is
    /// compute-bound on the E-cores (half-width SIMD) while the P-cores
    /// finish early, so it gets 16 small chunks per thread to balance them:
    /// a 0.5B verify pass at 4 per thread left ~21% of pool time waiting, and
    /// drafting (K=7) on low-repetition text went from -5% to -1% vs no
    /// drafting with 16. Chunking never changes results.
    /// With repacked matrices (`align` 8) chunks are whole r8 groups.
    fn rows_per_chunk(rows: usize, nb: usize, align: usize) -> usize {
        let per_thread = if nb <= 9 { 16 } else { (nb / 2).clamp(4, 32) };
        rows.div_ceil(Pool::global().threads() * per_thread)
            .max(1)
            .next_multiple_of(align.max(2))
    }

    /// Token slots for a batched matmul over `ws`: `quant::batch_slots` of
    /// the types that go through the per-row kernels (their x4 tiles want
    /// padding); the r8 GEMM takes any count, so all-repacked needs none.
    fn slots_for(ws: &[&Weight], n_in: usize, nb: usize) -> usize {
        let types: Vec<u32> = ws
            .iter()
            .filter(|w| w.r8.is_none())
            .map(|w| w.qt())
            .collect();
        if types.is_empty() {
            nb
        } else {
            batch_slots(&types, n_in, nb)
        }
    }

    /// Rows r0..r0 + ys.len() / slots of `w` times the batch, into
    /// ys[r * slots + b]: r8 GEMM over the `nb` real tokens (padding slots
    /// untouched), or the per-row tile kernels over every slot of `xr`.
    fn rows_batch(&self, w: &Weight, r0: usize, xr: &[&QuantizedAct], nb: usize, ys: &mut [f32]) {
        let slots = xr.len();
        let n_in = xr[0].len();
        let k = ys.len() / slots;
        if w.r8.is_some() {
            r8::gemm(w.qt(), w.groups(n_in, r0, k, false), &xr[..nb], ys, slots);
        } else {
            let (qt, data) = (w.qt(), w.bytes());
            let rb = row_bytes(qt, n_in);
            let row_of = |g: usize| (qt, &data[g * rb..(g + 1) * rb]);
            dot_rows_chunk(ys, slots, r0, &row_of, xr);
        }
    }

    /// The batch's activations quantized for `types`, one per token. The
    /// matmuls hand the kernels `quant::batch_slots` references to them, the
    /// extra slots repeating the last token (their outputs are ignored).
    /// Quantizing is about a nanosecond per float (896 floats ~1 us), the
    /// same scale as `per_token`'s cost unit, so a verify batch's 896-wide
    /// inputs quantize inline instead of paying a fork/join.
    fn quantize_batch<X: AsRef<[f32]> + Sync>(xs: &[X], types: &[u32]) -> Vec<QuantizedAct> {
        let n_in = xs[0].as_ref().len();
        par_map(xs.len(), n_in, |b| {
            let x = xs[b].as_ref();
            assert_eq!(x.len(), n_in, "matmul: ragged batch");
            QuantizedAct::for_types(x, types)
        })
    }

    /// `quantize_batch` of the first `nb` token columns of a row-major
    /// (row, token slot) buffer: the transpose and the quantization in one
    /// pass (one fork/join at most), same input values as `unbatch` gives.
    fn quantize_cols(y: &[f32], slots: usize, nb: usize, types: &[u32]) -> Vec<QuantizedAct> {
        let n = y.len() / slots;
        par_map(nb, n, |b| {
            QuantizedAct::for_types(&gather_col(y, slots, b, n), types)
        })
    }

    /// `matvec_many` for a batch of inputs: res[i][b] = ws[i] · xs[b].
    /// See `matmul_raw`; this transposes its output to per-token vectors.
    fn matmul_many<const N: usize, X: AsRef<[f32]> + Sync>(
        &self,
        ws: [&Weight; N],
        xs: &[X],
    ) -> [Vec<Vec<f32>>; N] {
        let nb = xs.len();
        let (out, starts, slots) = self.matmul_raw(ws, xs);
        std::array::from_fn(|i| {
            let n = ws[i].rows();
            unbatch(&out[starts[i] * slots..(starts[i] + n) * slots], slots, nb)
        })
    }

    /// Batched matmul, output left row-major over (row, token slot):
    /// y[r * S + b] for the joint row space of all `ws`, where starts[i] is
    /// matrix i's first row and S >= B the slot count (`quant::batch_slots`;
    /// slots past B are padding).
    ///
    /// Rows are split across the pool. A task loads each weight row once and
    /// dots it with every token's input while it's still in L1 (row outer,
    /// token inner), so the weights stream from RAM once per batch, not once
    /// per token. Each dot is the same `dot_row_q8` on the same quantized input
    /// as `matvec_many`, so results are bit-identical.
    fn matmul_raw<const N: usize, X: AsRef<[f32]> + Sync>(
        &self,
        ws: [&Weight; N],
        xs: &[X],
    ) -> (Vec<f32>, [usize; N], usize) {
        let types = ws.map(|w| w.qt());
        self.matmul_q(ws, &Self::quantize_batch(xs, &types))
    }

    /// `matmul_raw` on inputs already quantized for `ws`' types.
    fn matmul_q<const N: usize>(
        &self,
        ws: [&Weight; N],
        xq: &[QuantizedAct],
    ) -> (Vec<f32>, [usize; N], usize) {
        let nb = xq.len();
        let n_in = xq[0].len();
        let (starts, total, align) = Self::joint_rows(ws, n_in);
        let slots = Self::slots_for(&ws, n_in, nb);
        let xr: Vec<&QuantizedAct> = (0..slots).map(|b| &xq[b.min(nb - 1)]).collect();
        let mut out = vec![0f32; total * slots];
        let rows = Self::rows_per_chunk(total, nb, align);
        if align == 1 {
            // Nothing repacked: the per-row tile kernels, rows paired across
            // matrices as they come.
            let info = ws.map(|w| (w.qt(), row_bytes(w.qt(), n_in), w.bytes()));
            let row_of = |g: usize| {
                let t = starts.iter().rposition(|&s| s <= g).unwrap();
                let (qt, rb, data) = info[t];
                let o = g - starts[t];
                (qt, &data[o * rb..(o + 1) * rb])
            };
            pool::for_each_chunk_mut(&mut out, rows * slots, |c, ys| {
                dot_rows_chunk(ys, slots, c * rows, &row_of, &xr)
            });
        } else {
            // One run per matrix the chunk touches (whole r8 groups).
            pool::for_each_chunk_mut(&mut out, rows * slots, |c, mut ys| {
                let mut g = c * rows;
                while !ys.is_empty() {
                    let t = starts.iter().rposition(|&s| s <= g).unwrap();
                    let end = starts.get(t + 1).copied().unwrap_or(total);
                    let k = (end - g).min(ys.len() / slots);
                    let (run, rest) = std::mem::take(&mut ys).split_at_mut(k * slots);
                    self.rows_batch(ws[t], g - starts[t], &xr, nb, run);
                    g += k;
                    ys = rest;
                }
            });
        }
        (out, starts, slots)
    }

    /// The output head for a batch, reduced to each token's greedy pick:
    /// picks[b] == argmax(w · xs[b]) with `argmax`'s rules (see `Best`).
    /// Each pool chunk computes its rows into thread-local scratch and keeps
    /// only a `Best` per token; the chunks are then merged in row order. No
    /// vocab-sized logits are allocated or transposed. Each logit is the
    /// same dot as in `matmul_raw` (and so `matvec`), bit for bit.
    fn matmul_argmax<X: AsRef<[f32]> + Sync>(&self, w: &Weight, xs: &[X]) -> Vec<u32> {
        let nb = xs.len();
        let qt = w.qt();
        let xq = Self::quantize_batch(xs, &[qt]);
        let n_in = xq[0].len();
        let (_, total, align) = Self::joint_rows([w], n_in);
        let slots = Self::slots_for(&[w], n_in, nb);
        let xr: Vec<&QuantizedAct> = (0..slots).map(|b| &xq[b.min(nb - 1)]).collect();
        let rows = Self::rows_per_chunk(total, nb, align);
        let n_chunks = total.div_ceil(rows);
        // best[c * nb + b]: chunk c's best row for token b.
        let mut best = vec![Best::NONE; n_chunks * nb];
        pool::for_each_chunk_mut(&mut best, nb, |c, bs| {
            let first = c * rows;
            let n_rows = rows.min(total - first);
            with_rows_scratch(n_rows * slots, |ys| {
                self.rows_batch(w, first, &xr, nb, ys);
                best_rows(ys, slots, first, bs);
            });
        });
        (0..nb)
            .map(|b| {
                let mut acc = Best::NONE;
                for c in 0..n_chunks {
                    acc.merge(best[c * nb + b]);
                }
                acc.index(total)
            })
            .collect()
    }

    /// `matvec_swiglu` for a batch: per row r, per token b, the gate and up
    /// dots combined with the same formula. Layout and slot count returned as
    /// in `matmul_raw`.
    fn matmul_swiglu<X: AsRef<[f32]> + Sync>(
        &self,
        gate: &Weight,
        up: &Weight,
        xs: &[X],
    ) -> (Vec<f32>, usize) {
        let nb = xs.len();
        let n_in = xs[0].as_ref().len();
        let n = gate.rows();
        let (_, _, a0) = Self::joint_rows([gate], n_in);
        let (_, _, a1) = Self::joint_rows([up], n_in);
        assert_eq!(up.rows(), n, "swiglu: gate/up row counts differ");
        let types = [gate.qt(), up.qt()];
        let xq = Self::quantize_batch(xs, &types);
        let slots = Self::slots_for(&[gate, up], n_in, nb);
        let xr: Vec<&QuantizedAct> = (0..slots).map(|b| &xq[b.min(nb - 1)]).collect();
        let mut out = vec![0f32; n * slots];
        let align = a0.max(a1);
        let rows = Self::rows_per_chunk(n, nb, align);
        if align == 1 {
            // Nothing repacked: each gate row paired with its up row.
            let [g, u] = [gate, up].map(|w| (w.qt(), row_bytes(w.qt(), n_in), w.bytes()));
            pool::for_each_chunk_mut(&mut out, rows * slots, |c, ys| {
                with_rows_scratch(slots, |uv| {
                    for (j, yr) in ys.chunks_mut(slots).enumerate() {
                        let r = c * rows + j;
                        let gr = &g.2[r * g.1..(r + 1) * g.1];
                        let ur = &u.2[r * u.1..(r + 1) * u.1];
                        dot_rows_q8_multi([g.0, u.0], [gr, ur], &xr, [&mut *yr, &mut *uv]);
                        for (y, &uv) in yr.iter_mut().zip(&*uv) {
                            let gv = *y;
                            *y = (gv / (1.0 + (-gv).exp())) * uv;
                        }
                    }
                });
            });
        } else {
            // The chunk's gate rows, then its up rows, then combine (real
            // tokens only: padding slots may be unwritten on r8 rows).
            pool::for_each_chunk_mut(&mut out, rows * slots, |c, ys| {
                let r0 = c * rows;
                with_rows_scratch(ys.len(), |uv| {
                    self.rows_batch(gate, r0, &xr, nb, ys);
                    self.rows_batch(up, r0, &xr, nb, uv);
                    for (yr, ur) in ys.chunks_mut(slots).zip(uv.chunks(slots)) {
                        for (y, &uv) in yr[..nb].iter_mut().zip(&ur[..nb]) {
                            let gv = *y;
                            *y = (gv / (1.0 + (-gv).exp())) * uv;
                        }
                    }
                });
            });
        }
        (out, slots)
    }

    /// RMSNorm: scale x so its root-mean-square is 1, then multiply by a learned weight.
    fn rmsnorm(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len().min(w.len())];
        self.rmsnorm_into(x, w, &mut out);
        out
    }

    /// `rmsnorm` into an existing buffer.
    fn rmsnorm_into(&self, x: &[f32], w: &[f32], out: &mut [f32]) {
        rmsnorm_into(x, w, self.eps, out);
    }

    /// (sin, cos) of each RoPE pair's angle at `pos`: pair i turns by
    /// pos * base^(-2i/head_dim). Low i spins fast, high i spins slowly.
    /// Every head in every layer uses the same angles, so this runs once a token.
    fn rope_table(&self, pos: usize) -> Vec<(f32, f32)> {
        (0..self.head_dim / 2)
            .map(|i| {
                let freq = self.rope_base.powf(-2.0 * i as f64 / self.head_dim as f64);
                let angle = pos as f64 * freq;
                (angle.sin() as f32, angle.cos() as f32)
            })
            .collect()
    }

    /// Run one token at position `pos` and return logits (len vocab).
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let x = self.hidden(token, pos, cache);
        // 9. Final norm, then the output head scores every vocab entry.
        let h = self.rmsnorm(&x, &self.output_norm);
        self.matvec(&self.output, &h)
    }

    /// Run one token at position `pos` for its effect on the cache only. Prompt
    /// tokens before the last need no logits, and the output head is the
    /// biggest matrix in the model (37% of a token's weight reads here).
    /// Prompts go through `feed_batch`; this is its reference in the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn feed(&self, token: u32, pos: usize, cache: &mut KvCache) {
        self.hidden(token, pos, cache);
    }

    /// The token's row of the embedding matrix.
    fn embed(&self, token: u32) -> Vec<f32> {
        let mut x = vec![0.0f32; self.dim];
        let rb = row_bytes(self.token_embd.qtype.id, self.dim);
        let t = token as usize;
        dequant(
            self.token_embd.qtype.id,
            &self.embd[t * rb..(t + 1) * rb],
            &mut x,
        );
        x
    }

    /// Run `tokens[b]` at position `start + b` for every b, for the cache only.
    /// Same result, bit for bit, as `feed` on each token in order, but every
    /// weight row is read once per batch instead of once per token.
    pub fn feed_batch(&self, tokens: &[u32], start: usize, cache: &mut KvCache) {
        self.hidden_batch(tokens, start, cache);
    }

    /// Logits after each of `tokens[b]` at position `start + b`: row b is,
    /// bit for bit, what `forward(tokens[b], start + b, ..)` returns after
    /// the earlier tokens went in. The speculative-decoding verify pass: the
    /// whole model, output head included, reads each weight row once for the
    /// batch. Same final norm as `forward`; the head goes through
    /// `matmul_many`, whose per-row dots equal `matvec`'s. Cache slots
    /// `start..start + len` are overwritten; slots past the accepted prefix
    /// are simply rewritten by the next call (attention reads only 0..=pos).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn verify_batch(&self, tokens: &[u32], start: usize, cache: &mut KvCache) -> Vec<Vec<f32>> {
        let hs = self.final_norm_batch(tokens, start, cache);
        let [logits] = self.matmul_many([&self.output], &hs);
        logits
    }

    /// `verify_batch` reduced to what greedy decoding uses: picks[b] ==
    /// argmax(verify_batch(..)[b]), same index on ties (the last of equal
    /// maxima, never a NaN unless all are: `Best`), without materializing
    /// any logits (`matmul_argmax`).
    pub fn verify_argmax(&self, tokens: &[u32], start: usize, cache: &mut KvCache) -> Vec<u32> {
        let hs = self.final_norm_batch(tokens, start, cache);
        self.matmul_argmax(&self.output, &hs)
    }

    /// `hidden_batch` then the final norm, as `forward` does per token.
    fn final_norm_batch(&self, tokens: &[u32], start: usize, cache: &mut KvCache) -> Vec<Vec<f32>> {
        let xs = self.hidden_batch(tokens, start, cache);
        par_map(xs.len(), self.dim, |b| {
            self.rmsnorm(&xs[b], &self.output_norm)
        })
    }

    /// `hidden` for a batch: `feed_batch`'s work, returning every token's
    /// final hidden state (before the output norm).
    fn hidden_batch(&self, tokens: &[u32], start: usize, cache: &mut KvCache) -> Vec<Vec<f32>> {
        let nb = tokens.len();
        assert!(nb > 0, "feed_batch: empty batch");
        assert!(
            start + nb <= cache.max_ctx,
            "positions {start}..{} exceed cache size {}",
            start + nb,
            cache.max_ctx
        );
        let (dim, hd) = (self.dim, self.head_dim);
        let kv_dim = cache.kv_dim;
        let n_kv = self.n_kv_heads;
        let group = self.n_heads / n_kv;

        // Every per-token step runs over the pool by token (`per_token`): with
        // a few hundred tokens, a serial pass here leaves every worker idle.
        // A few tokens' worth (a speculative verify) runs inline instead.
        let mut xs: Vec<Vec<f32>> = par_map(nb, dim, |b| self.embed(tokens[b]));
        let ropes: Vec<_> = par_map(nb, hd * 16, |b| self.rope_table(start + b));
        // Reused across layers; every element is overwritten before it's read.
        let mut hs: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; nb];
        let mut att = vec![0.0f32; nb * dim];
        let qn = group * hd; // floats per (token, kv head) of attention output
        debug_assert_eq!(dim, n_kv * qn, "feed_batch: dim != heads * head_dim");
        for (l, layer) in self.layers.iter().enumerate() {
            // Steps as in `hidden`, each applied to every token of the batch.
            per_token(&mut hs, dim, |b, h| {
                self.rmsnorm_into(&xs[b], &layer.attn_norm, h)
            });
            let [mut qs, ks, vs] = self.matmul_many([&layer.wq, &layer.wk, &layer.wv], &hs);
            {
                // Bias, RoPE, and k/v into the cache, per token. k and v get
                // their bias and RoPE in the cache slot itself: same values,
                // same ops as `hidden`, which does them before the copy.
                let span = start * kv_dim..(start + nb) * kv_dim;
                let (kl, vl) = (&mut cache.k[l][span.clone()], &mut cache.v[l][span]);
                let mut jobs: Vec<_> = qs
                    .iter_mut()
                    .zip(kl.chunks_mut(kv_dim))
                    .zip(vl.chunks_mut(kv_dim))
                    .collect();
                per_token(&mut jobs, dim + 2 * kv_dim, |b, ((q, kd), vd)| {
                    kd.copy_from_slice(&ks[b]);
                    vd.copy_from_slice(&vs[b]);
                    if let Some(bias) = &layer.bq {
                        add_assign(q, bias);
                    }
                    if let Some(bias) = &layer.bk {
                        add_assign(kd, bias);
                    }
                    if let Some(bias) = &layer.bv {
                        add_assign(vd, bias);
                    }
                    rope_heads(q, hd, &ropes[b]);
                    rope_heads(kd, hd, &ropes[b]);
                });
            }
            let (kc, vc) = (&cache.k[l], &cache.v[l]);

            // Causal attention, one (token, kv head) pair per pool task,
            // longest first: all `group` query heads sharing that kv head
            // together, so each key and value row is loaded once for all of
            // them. Token b sees positions 0..=start + b only. Per query head
            // the arithmetic is exactly `hidden`'s: same dot, same softmax
            // order, same `o += w v` order over t.
            let scale = 1.0 / (hd as f32).sqrt();
            pool::for_each_chunk_mut_rev(&mut att, qn, |c, out| {
                let (b, kvh) = (c / n_kv, c % n_kv);
                let np = start + b + 1;
                let qg = &qs[b][kvh * qn..(kvh + 1) * qn];
                with_scores((np + 1) * group, |sc| {
                    attend_causal(qg, kc, vc, kv_dim, hd, kvh, np, scale, sc, out)
                });
            });

            let att_rows: Vec<&[f32]> = att.chunks(dim).collect();
            let (os, _, slots) = self.matmul_raw([&layer.wo], &att_rows);
            add_unbatched(&mut xs, &os, slots);

            per_token(&mut hs, dim, |b, h| {
                self.rmsnorm_into(&xs[b], &layer.ffn_norm, h)
            });
            let (acts, a_slots) = self.matmul_swiglu(&layer.w_gate, &layer.w_up, &hs);
            let aq = Self::quantize_cols(&acts, a_slots, nb, &[layer.w_down.qt()]);
            let (ds, _, slots) = self.matmul_q([&layer.w_down], &aq);
            add_unbatched(&mut xs, &ds, slots);
        }
        xs
    }

    /// Every layer for one token: store its keys and values in the cache and
    /// return the final hidden state (before the output norm).
    fn hidden(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        assert!(
            pos < cache.max_ctx,
            "position {pos} exceeds cache size {}",
            cache.max_ctx
        );
        let (dim, hd) = (self.dim, self.head_dim);
        let kv_dim = cache.kv_dim;

        // 1. Embedding: the token's row of the embedding matrix is its starting vector.
        let mut x = self.embed(token);

        let rope = self.rope_table(pos);
        // Attention scratch (step 5), kept on this thread across tokens so
        // decode attention allocates nothing once warm.
        let np = pos + 1;
        let tile = decode_tile(np, self.n_kv_heads);
        let (mut buf, mut att_out) = DECODE.take();
        att_out.resize(dim, 0.0);
        for (l, layer) in self.layers.iter().enumerate() {
            // ---- Attention sub-block ----
            // 2. Normalize, then project into queries, keys and values (+ bias).
            let h = self.rmsnorm(&x, &layer.attn_norm);
            let [mut q, mut k, mut v] = self.matvec_many([&layer.wq, &layer.wk, &layer.wv], &h);
            if let Some(b) = &layer.bq {
                add_assign(&mut q, b);
            }
            if let Some(b) = &layer.bk {
                add_assign(&mut k, b);
            }
            if let Some(b) = &layer.bv {
                add_assign(&mut v, b);
            }

            // 3. Rotate q and k by position (RoPE), one head at a time.
            rope_heads(&mut q, hd, &rope);
            rope_heads(&mut k, hd, &rope);

            // 4. Store this position's k and v in the cache so later tokens can see them.
            cache.k[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            cache.v[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
            let (kc, vc) = (&cache.k[l], &cache.v[l]);

            // 5. Each query head attends over all positions 0..=pos (the work
            //    grows with context, so it can't stay serial). GQA: query head
            //    qh reads kv head qh / group. See `attend`.
            attend(&q, kc, vc, kv_dim, hd, np, tile, &mut buf, &mut att_out);

            // 6. Mix the heads back together and add to the residual stream.
            let o = self.matvec(&layer.wo, &att_out);
            add_assign(&mut x[..dim], &o[..dim]);

            // ---- Feed-forward sub-block (SwiGLU) ----
            // 7. Normalize, then gate * up: silu(gate(h)) decides how much of up(h) passes.
            let h = self.rmsnorm(&x, &layer.ffn_norm);
            let act = self.matvec_swiglu(&layer.w_gate, &layer.w_up, &h);
            // 8. Project back down to model width and add to the residual stream.
            let d = self.matvec(&layer.w_down, &act);
            add_assign(&mut x[..dim], &d[..dim]);
        }
        DECODE.set((buf, att_out));

        x
    }
}

/// Below this much work (items x floats touched per item) a per-token step
/// runs inline on the caller: a pool fork/join costs more than, say, an
/// rmsnorm of a few tokens. A verify batch's elementwise steps stay inline;
/// prefill batches, the output-head transpose and quantization go wide.
const INLINE_WORK: usize = 16 * 1024;

/// Call `f(i, &mut data[i])` for every i, each item touching about
/// `floats` floats: inline if that's little work in total, else spread over
/// the pool in chunks of consecutive items (~4 chunks per thread).
fn per_token<T: Send>(data: &mut [T], floats: usize, f: impl Fn(usize, &mut T) + Sync) {
    if data.len() <= 1 || data.len() * floats <= INLINE_WORK {
        data.iter_mut().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let chunk = data.len().div_ceil(Pool::global().threads() * 4).max(1);
    pool::for_each_chunk_mut(data, chunk, |c, xs| {
        for (j, x) in xs.iter_mut().enumerate() {
            f(c * chunk + j, x);
        }
    });
}

/// `(0..n).map(f).collect()`, through `per_token` (same `floats` meaning).
fn par_map<T: Send>(n: usize, floats: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    per_token(&mut out, floats, |i, o| *o = Some(f(i)));
    out.into_iter().map(|o| o.unwrap()).collect()
}

/// Transpose a row-major (row, token slot) buffer, y[r * slots + b], into
/// one vector per token for the first `nb` slots.
fn unbatch(y: &[f32], slots: usize, nb: usize) -> Vec<Vec<f32>> {
    let n = y.len() / slots;
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); nb];
    per_token(&mut out, n, |b, x| *x = gather_col(y, slots, b, n));
    out
}

/// xs[b][i] += y[i * slots + b]: the residual add straight from a row-major
/// (row, token slot) buffer, same single add per element as adding
/// `unbatch(y)`.
fn add_unbatched(xs: &mut [Vec<f32>], y: &[f32], slots: usize) {
    let n = xs.first().map_or(0, |x| x.len());
    per_token(xs, n, |b, x| add_col(x, y, slots, b));
}

/// One pool chunk of a batched matmul: rows `first..first + ys.len() / slots`
/// (`row_of(g)` = (qtype, packed row g)) times every token slot of `xr`, into
/// ys[r * slots + b]. Rows two at a time (possibly from different matrices):
/// the pair shares each token's activation loads. An odd last row goes alone.
fn dot_rows_chunk<'w>(
    ys: &mut [f32],
    slots: usize,
    first: usize,
    row_of: &impl Fn(usize) -> (u32, &'w [u8]),
    xr: &[&QuantizedAct],
) {
    let n_rows = ys.len() / slots;
    let mut pairs = ys.chunks_exact_mut(2 * slots);
    for (j, yp) in pairs.by_ref().enumerate() {
        let g = first + 2 * j;
        let (q0, r0) = row_of(g);
        let (q1, r1) = row_of(g + 1);
        let (y0, y1) = yp.split_at_mut(slots);
        dot_rows_q8_multi([q0, q1], [r0, r1], xr, [y0, y1]);
    }
    let yr = pairs.into_remainder();
    if !yr.is_empty() {
        let (qt, row) = row_of(first + n_rows - 1);
        dot_row_q8_multi(qt, row, xr, yr);
    }
}

/// Running argmax with main.rs `argmax`'s exact rules: the largest value
/// by `>`/`==` (so NaN is never picked and -0.0 ties +0.0), and of equal
/// maxima the LAST index. Pushing values in index order, or merging the
/// `Best`s of consecutive index ranges in order, gives that index: a later
/// value/range replaces the current one iff it is greater or equal.
/// `index(n)` is n - 1 when nothing qualified (all NaN), as `argmax` does.
#[derive(Clone, Copy, Debug)]
pub struct Best {
    v: f32,
    i: Option<u32>,
}

impl Best {
    pub const NONE: Best = Best {
        v: f32::NEG_INFINITY,
        i: None,
    };

    #[inline]
    pub fn push(&mut self, v: f32, i: u32) {
        // `>=` is false when either side is NaN, so NaN is skipped.
        if v >= self.v {
            *self = Best { v, i: Some(i) };
        }
    }

    pub fn merge(&mut self, later: Best) {
        if let Some(i) = later.i {
            self.push(later.v, i);
        }
    }

    pub fn index(self, n: usize) -> u32 {
        self.i.unwrap_or(n as u32 - 1)
    }
}

thread_local! {
    /// Per-task matmul scratch (head rows of `matmul_argmax`, swiglu's up
    /// values); grows to the largest seen and is reused across passes.
    static ROWS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` on this thread's matmul scratch, exactly `n` long. Contents are
/// stale; `f` must write before reading.
///
/// Not re-entrant: `f` must not call `with_rows_scratch` again, directly or
/// through anything it runs (`dot_rows_chunk` and the quant kernels don't).
/// A nested call would borrow the RefCell twice; it panics with a message
/// saying so rather than aliasing the buffer.
fn with_rows_scratch<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    ROWS.with(|cell| {
        let mut s = cell
            .try_borrow_mut()
            .expect("with_rows_scratch re-entered: one scratch per thread");
        if s.len() < n {
            s.resize(n, 0.0);
        }
        f(&mut s[..n])
    })
}

thread_local! {
    /// Prefill attention scores of the task in hand; grows to the largest seen.
    static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` on this thread's score scratch, at least `n` long. Contents are
/// left over from earlier tasks: write before reading.
fn with_scores<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    SCORES.with_borrow_mut(|s| {
        if s.len() < n {
            s.resize(n, 0.0);
        }
        f(&mut s[..n])
    })
}

thread_local! {
    /// Decode attention scratch (`attend`'s buffer, attention output), taken
    /// by `hidden` for a token and put back after.
    static DECODE: std::cell::Cell<(Vec<f32>, Vec<f32>)> = const { std::cell::Cell::new((Vec::new(), Vec::new())) };
}

/// Decode contexts shorter than this score one task per query head (plain
/// `dot`): K is small enough to stay in cache across the group's re-reads,
/// and it's one fork/join instead of two. Untuned guess (no benchmarks).
const TILE_MIN_NP: usize = 512;
/// Fewest positions per tiled scoring task.
const TILE_MIN_ROWS: usize = 32;

/// `attend`'s scoring plan for `np` positions: None for per-query-head tasks,
/// or Some(rows per (kv head, t-range) task), aiming at ~3 tasks per thread.
fn decode_tile(np: usize, n_kv: usize) -> Option<usize> {
    if np < TILE_MIN_NP {
        return None;
    }
    let ranges = (Pool::global().threads() * 3).div_ceil(n_kv).max(1);
    Some(np.div_ceil(ranges).max(TILE_MIN_ROWS))
}

/// Decode attention of one token over cache positions 0..np: out[qh * hd..]
/// = sum over t of softmax_t(q_qh · k_t / sqrt(hd)) * v_t, with q head qh on
/// kv head qh / group. kc and vc are [t][kv_dim] (kv_dim = n_kv * hd); `buf`
/// is reusable scratch, grown as needed.
///
/// Scores, `tile` None: one pool task per query head, `dot` per position.
/// `tile` Some(tc): one task per (kv head, tc positions), each K row loaded
/// once for all `group` heads (`dot_group`, which is `dot` per head) into
/// sc[kvh][t][g]; then one task per query head runs `softmax_from` on its
/// strided scores into wts. Either way wts[qh][t] gets the same dot, the same
/// `* scale` and the same softmax ops in the same order.
///
/// Values: one task per (kv head, 8 floats of hd), all of its heads in one
/// pass over the V rows (`vsum_rows`), then copied into place.
#[allow(clippy::too_many_arguments)]
fn attend(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    kv_dim: usize,
    hd: usize,
    np: usize,
    tile: Option<usize>,
    buf: &mut Vec<f32>,
    out: &mut [f32],
) {
    let n_heads = q.len() / hd;
    let n_kv = kv_dim / hd;
    let group = n_heads / n_kv;
    let (qn, nd) = (group * hd, hd.div_ceil(8));
    debug_assert_eq!(n_kv * qn, q.len(), "attend: heads * head_dim != q len");
    assert!(np * kv_dim <= kc.len().min(vc.len()) && out.len() >= q.len());
    let scale = 1.0 / (hd as f32).sqrt();
    // Per kv head, `nt` t-ranges of `tc` rows; the last may be short and
    // its tail of sc is left unused, so chunks never straddle kv heads.
    let (tc, nt) = tile.map_or((0, 0), |tc| (tc, np.div_ceil(tc)));
    let (nw, ns, nv) = (n_heads * np, n_kv * nt * tc * group, n_kv * nd * group * 8);
    if buf.len() < nw + ns + nv {
        buf.resize(nw + ns + nv, 0.0);
    }
    let (wts, rest) = buf.split_at_mut(nw);
    let (sc, rest) = rest.split_at_mut(ns);
    let vparts = &mut rest[..nv];

    if tile.is_none() {
        pool::for_each_chunk_mut(wts, np, |qh, s| {
            let qv = &q[qh * hd..(qh + 1) * hd];
            score_softmax(qv, kc, kv_dim, qh / group * hd, scale, s);
        });
    } else {
        pool::for_each_chunk_mut(sc, tc * group, |c, s| {
            let (kvh, t0) = (c / nt, c % nt * tc);
            let qg = &q[kvh * qn..(kvh + 1) * qn];
            score_tile(qg, kc, kv_dim, hd, kvh * hd, t0..np.min(t0 + tc), scale, s);
        });
        let sc = &*sc;
        pool::for_each_chunk_mut(wts, np, |qh, w| {
            let (kvh, g) = (qh / group, qh % group);
            softmax_from(&sc[kvh * nt * tc * group + g..], group, w);
        });
    }

    // vparts[c]: task c = (kvh, dc), its heads' sums for d in dc * 8..+ width,
    // as [head][width].
    let wts = &*wts;
    pool::for_each_chunk_mut(vparts, group * 8, |c, o| {
        let (kvh, d0) = (c / nd, c % nd * 8);
        let width = 8.min(hd - d0);
        let w = &wts[kvh * group * np..];
        let vb = &vc[kvh * hd + d0..];
        vsum_rows(w, np, np, group, vb, kv_dim, &mut o[..group * width], width);
    });
    for (c, part) in vparts.chunks_exact(group * 8).enumerate() {
        let (kvh, d0) = (c / nd, c % nd * 8);
        let width = 8.min(hd - d0);
        for (g, p) in part[..group * width].chunks_exact(width).enumerate() {
            let o0 = (kvh * group + g) * hd + d0;
            out[o0..o0 + width].copy_from_slice(p);
        }
    }
}

// ---------------------------------------------------------------------------
// The hot loops outside the quant kernels, each compiled twice (portable and
// x86-64-v3, chosen at runtime: see mv.rs). Bodies are closure-free.

multiversion! {
    /// RMSNorm: scale x so its root-mean-square is 1, then multiply by a
    /// learned weight. The sum of squares is in order from -0.0 (`sum`'s order).
    fn rmsnorm_into<V3>(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
        let mut ss = -0.0f32;
        for v in x {
            ss += v * v;
        }
        let ms = ss / x.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
            *o = v * inv * g;
        }
    }
}

/// NEOX-style RoPE on one head vector (len head_dim) in place: dim i is
/// paired with dim i + head_dim/2 and the pair rotated by `table[i]`.
#[inline(always)]
fn rope_head(v: &mut [f32], table: &[(f32, f32)]) {
    let half = table.len();
    for (i, &(sin, cos)) in table.iter().enumerate() {
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * cos - b * sin;
        v[i + half] = a * sin + b * cos;
    }
}

multiversion! {
    /// `rope_head` on every head of v (heads of `hd` floats).
    fn rope_heads<V3>(v: &mut [f32], hd: usize, table: &[(f32, f32)]) {
        for head in v.chunks_mut(hd) {
            rope_head(head, table);
        }
    }
}

multiversion! {
    /// a[i] += b[i] (bias and residual adds).
    fn add_assign<V3>(a: &mut [f32], b: &[f32]) {
        for (a, b) in a.iter_mut().zip(b) {
            *a += b;
        }
    }
}

multiversion! {
    /// x[i] += y[i * slots + b]: one token's residual add from a row-major
    /// (row, token slot) buffer.
    fn add_col<V3>(x: &mut [f32], y: &[f32], slots: usize, b: usize) {
        for (i, x) in x.iter_mut().enumerate() {
            *x += y[i * slots + b];
        }
    }
}

multiversion! {
    /// Token slot b's column of a row-major (row, token slot) buffer of n rows.
    fn gather_col<V3>(y: &[f32], slots: usize, b: usize, n: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; n];
        for (r, x) in x.iter_mut().enumerate() {
            *x = y[r * slots + b];
        }
        x
    }
}

multiversion! {
    /// `Best::push` of every row's logit for each token: ys is row-major
    /// (row, token slot), rows numbered from `first`, in row order.
    fn best_rows<V3>(ys: &[f32], slots: usize, first: usize, bs: &mut [Best]) {
        for (b, best) in bs.iter_mut().enumerate() {
            for (r, y) in ys.chunks_exact(slots).enumerate() {
                best.push(y[b], (first + r) as u32);
            }
        }
    }
}

multiversion! {
    /// Decode scores of one query head, then `softmax`: s[t] =
    /// dot(qv, k_t) * scale with k_t = kc[t * kv_dim + k_off..][..qv.len()].
    fn score_softmax<V3>(
        qv: &[f32],
        kc: &[f32],
        kv_dim: usize,
        k_off: usize,
        scale: f32,
        s: &mut [f32],
    ) {
        let hd = qv.len();
        for (t, s) in s.iter_mut().enumerate() {
            let k0 = t * kv_dim + k_off;
            *s = dot(qv, &kc[k0..k0 + hd]) * scale;
        }
        softmax(s);
    }
}

multiversion! {
    /// Decode scores of one kv head's query group qg (heads of hd floats)
    /// over positions `ts`: s[j * group + g] = dot(q_g, k_t) * scale for the
    /// j-th t, `dot_group` per position (k_t = kc[t * kv_dim + k_off..][..hd]).
    fn score_tile<V3>(
        qg: &[f32],
        kc: &[f32],
        kv_dim: usize,
        hd: usize,
        k_off: usize,
        ts: std::ops::Range<usize>,
        scale: f32,
        s: &mut [f32],
    ) {
        let group = qg.len() / hd;
        for (t, row) in ts.zip(s.chunks_exact_mut(group)) {
            let k0 = t * kv_dim + k_off;
            // SAFETY: V3 only in the v3 variant.
            unsafe { dot_group::<V3>(qg, &kc[k0..k0 + hd], row) };
            for x in row.iter_mut() {
                *x *= scale;
            }
        }
    }
}

multiversion! {
    /// One (token, kv head) task of `hidden_batch`'s causal attention over
    /// positions 0..np: per query head the arithmetic is exactly `hidden`'s
    /// (same dot, same softmax order, same `o += w v` order over t). `sc` is
    /// scratch of (np + 1) * group floats: sc[g * np + t] holds head g's score
    /// at t (contiguous per head for the softmax), the rest one position's dots.
    fn attend_causal<V3>(
        qg: &[f32],
        kc: &[f32],
        vc: &[f32],
        kv_dim: usize,
        hd: usize,
        kvh: usize,
        np: usize,
        scale: f32,
        sc: &mut [f32],
        out: &mut [f32],
    ) {
        let group = qg.len() / hd;
        let (sc, row) = sc.split_at_mut(np * group);
        for t in 0..np {
            let k0 = t * kv_dim + kvh * hd;
            // SAFETY (both calls): V3 only in the v3 variant.
            unsafe { dot_group::<V3>(qg, &kc[k0..k0 + hd], row) };
            for (g, &s) in row.iter().enumerate() {
                sc[g * np + t] = s * scale;
            }
        }
        for s in sc.chunks_exact_mut(np) {
            softmax(s);
        }
        let vb = &vc[kvh * hd..];
        unsafe { vsum_rows::imp::<V3>(sc, np, np, group, vb, kv_dim, out, hd) };
    }
}

multiversion! {
    /// `softmax` of the scores src[t * stride], t in 0..dst.len(), written to
    /// dst: the same ops in the same order as `softmax` on a copy.
    fn softmax_from<V3>(src: &[f32], stride: usize, dst: &mut [f32]) {
        let n = dst.len();
        assert!(
            n == 0 || (n - 1) * stride < src.len(),
            "softmax_from: short src"
        );
        let mut max = f32::NEG_INFINITY;
        for t in 0..n {
            max = max.max(src[t * stride]);
        }
        let mut sum = 0.0f32;
        for (t, e) in dst.iter_mut().enumerate() {
            *e = (src[t * stride] - max).exp();
            sum += *e;
        }
        let inv = 1.0 / sum;
        for e in dst.iter_mut() {
            *e *= inv;
        }
    }
}

/// Scaled scores to attention weights, in place: subtract the max (for
/// stability), exp, sum in t order, then multiply each by 1 / sum.
#[inline(always)]
fn softmax(s: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in s.iter() {
        max = max.max(v);
    }
    let mut sum = 0.0f32;
    for e in s.iter_mut() {
        *e = (*e - max).exp();
        sum += *e;
    }
    let inv = 1.0 / sum;
    for e in s.iter_mut() {
        *e *= inv;
    }
}

multiversion! {
    /// o[h * os + d] = sum over t in 0..np, in order from 0.0, of
    /// w[h * ws + t] * vb[t * vs + d], for heads h in 0..nh and d in 0..os:
    /// every head sharing one pass over the value rows (each `os` long, at
    /// stride vs). Up to 8 heads per pass.
    fn vsum_rows<V3>(
        w: &[f32],
        ws: usize,
        np: usize,
        nh: usize,
        vb: &[f32],
        vs: usize,
        o: &mut [f32],
        os: usize,
    ) {
        for h0 in (0..nh).step_by(8) {
            let (w, o) = (&w[h0 * ws..], &mut o[h0 * os..]);
            // SAFETY (every arm): V3 only in the v3 variant.
            unsafe {
                match nh - h0 {
                    1 => vsum_rows_group::<V3, 1>(w, ws, np, vb, vs, o, os),
                    2 => vsum_rows_group::<V3, 2>(w, ws, np, vb, vs, o, os),
                    3 => vsum_rows_group::<V3, 3>(w, ws, np, vb, vs, o, os),
                    4 => vsum_rows_group::<V3, 4>(w, ws, np, vb, vs, o, os),
                    5 => vsum_rows_group::<V3, 5>(w, ws, np, vb, vs, o, os),
                    6 => vsum_rows_group::<V3, 6>(w, ws, np, vb, vs, o, os),
                    7 => vsum_rows_group::<V3, 7>(w, ws, np, vb, vs, o, os),
                    _ => vsum_rows_group::<V3, 8>(w, ws, np, vb, vs, o, os),
                }
            }
        }
    }
}

/// `vsum_rows` for G heads: `vsum_rows_group_v3` when V3, else the scalar
/// loop. Both do the scalar `o[d] += w * v[d]` over t, per element.
///
/// # Safety
/// With V3, the CPU must have the x86-64-v3 features.
#[inline(always)]
unsafe fn vsum_rows_group<const V3: bool, const G: usize>(
    w: &[f32],
    ws: usize,
    np: usize,
    vb: &[f32],
    vs: usize,
    o: &mut [f32],
    os: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if V3 {
        // SAFETY: our caller's contract.
        return unsafe { vsum_rows_group_v3::<G>(w, ws, np, vb, vs, o, os) };
    }
    for g in 0..G {
        for d in 0..os {
            let mut a = 0.0f32;
            for t in 0..np {
                a += w[g * ws + t] * vb[t * vs + d];
            }
            o[g * os + d] = a;
        }
    }
}

/// `vsum_rows` for G heads with AVX. Per 8-float slice of d: G accumulators
/// start at 0.0, and for t ascending the v slice is loaded once and each head
/// does acc = acc + w * v (a separate multiply and add; Rust never contracts
/// to FMA). That's the scalar `o[d] += w * v[d]` over t, per element, so the
/// result is bit-identical to one head at a time. Registers: G accumulators,
/// the v slice and one broadcast weight, so 10 of 16 at G = 8. Leftover d (os
/// not a multiple of 8) runs scalar in the same order. Bounds are checked
/// once up front instead of per row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,f16c,fma,lzcnt")]
#[inline]
fn vsum_rows_group_v3<const G: usize>(
    w: &[f32],
    ws: usize,
    np: usize,
    vb: &[f32],
    vs: usize,
    o: &mut [f32],
    os: usize,
) {
    use std::arch::x86_64::*;
    assert!(
        np <= ws && (G - 1) * ws + np <= w.len() && G * os <= o.len(),
        "vsum_rows: short w/o"
    );
    assert!(
        np == 0 || (np - 1) * vs + os <= vb.len(),
        "vsum_rows: short v"
    );
    let op = o.as_mut_ptr();
    let mut wr = [w.as_ptr(); G];
    for (g, p) in wr.iter_mut().enumerate() {
        *p = w.as_ptr().wrapping_add(g * ws);
    }
    let nc = os / 8;
    for c in 0..nc {
        // SAFETY: AVX is enabled (target_feature above). Weight reads
        // wr[g] + t = w[g * ws + t] with t < np <= ws are inside w; v loads
        // read vb[t * vs + 8 * c..+ 8] with t < np and 8 * c + 8 <= os, inside
        // vb; stores o[g * os + 8 * c..+ 8] are inside o (all asserted above).
        // The pointers only advance with wrapping_add and are dereferenced
        // for t < np.
        unsafe {
            let mut acc = [_mm256_setzero_ps(); G];
            let mut vp = vb.as_ptr().wrapping_add(8 * c);
            for t in 0..np {
                let y = _mm256_loadu_ps(vp);
                for (a, wg) in acc.iter_mut().zip(&wr) {
                    let x = _mm256_set1_ps(*wg.add(t));
                    *a = _mm256_add_ps(*a, _mm256_mul_ps(x, y));
                }
                vp = vp.wrapping_add(vs);
            }
            for (g, a) in acc.iter().enumerate() {
                _mm256_storeu_ps(op.add(g * os + 8 * c), *a);
            }
        }
    }
    for d in 8 * nc..os {
        for g in 0..G {
            let mut a = 0.0f32;
            for t in 0..np {
                a += w[g * ws + t] * vb[t * vs + d];
            }
            o[g * os + d] = a;
        }
    }
}

/// out[g] = dot(q_g, k) for the query vectors q_g = q[g * k.len()..][..k.len()],
/// g in 0..out.len(), bit for bit; up to 8 at a time. Not dispatched on its
/// own: the attention bodies call it with their own `V3`.
///
/// # Safety
/// With V3, the CPU must have the x86-64-v3 features.
#[inline(always)]
unsafe fn dot_group<const V3: bool>(q: &[f32], k: &[f32], out: &mut [f32]) {
    let n = k.len();
    debug_assert_eq!(q.len(), n * out.len());
    for (qs, os) in q.chunks(8 * n).zip(out.chunks_mut(8)) {
        // SAFETY (every arm): our caller's contract.
        unsafe {
            match os.len() {
                8 => dot_multi::<V3, 8>(qs, k, os),
                7 => dot_multi::<V3, 7>(qs, k, os),
                6 => dot_multi::<V3, 6>(qs, k, os),
                5 => dot_multi::<V3, 5>(qs, k, os),
                4 => dot_multi::<V3, 4>(qs, k, os),
                3 => dot_multi::<V3, 3>(qs, k, os),
                2 => dot_multi::<V3, 2>(qs, k, os),
                _ => dot_multi::<V3, 1>(qs, k, os),
            }
        }
    }
}

/// `dot` of G query vectors (contiguous in q) with one k: `dot_multi_v3`
/// when V3, else `dot` per vector (the same bits).
///
/// # Safety
/// With V3, the CPU must have the x86-64-v3 features.
#[inline(always)]
unsafe fn dot_multi<const V3: bool, const G: usize>(q: &[f32], k: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if V3 {
        // SAFETY: our caller's contract.
        return unsafe { dot_multi_v3::<G>(q, k, out) };
    }
    let n = k.len();
    for (g, o) in out[..G].iter_mut().enumerate() {
        *o = dot(&q[g * n..(g + 1) * n], k);
    }
}

/// `dot` of G query vectors with one k, with AVX. Each result uses `dot`'s
/// exact lanes, tail and final sum, so it's identical to `dot`; the G
/// accumulator chains run side by side and k is loaded once.
///
/// Written with intrinsics: the plain-Rust version (an `[[f32; 8]; G]`
/// accumulator indexed through per-head slices) didn't vectorize and ran as
/// scalar loads/adds through the stack, ~40% of prefill at 2k tokens.
/// `_mm256_add_ps(acc, _mm256_mul_ps(x, y))` is `dot`'s `acc[i] += x[i] * y[i]`
/// lane for lane (Rust never contracts to FMA), so results are unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,f16c,fma,lzcnt")]
#[inline]
fn dot_multi_v3<const G: usize>(q: &[f32], k: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = k.len();
    assert!(q.len() >= G * n && out.len() >= G, "dot_multi: short q/out");
    let (kc, kr) = k.as_chunks::<8>();
    let nc = kc.len();
    let qp = q.as_ptr();
    // SAFETY: AVX is enabled (target_feature above). Every load reads
    // q[g * n + 8 * j .. + 8] with g < G and 8 * j + 8 <= n, inside q
    // (asserted above), or a whole 8-float chunk of k.
    let acc = unsafe {
        let mut acc = [_mm256_setzero_ps(); G];
        for (j, y) in kc.iter().enumerate() {
            let y = _mm256_loadu_ps(y.as_ptr());
            for (g, a) in acc.iter_mut().enumerate() {
                let x = _mm256_loadu_ps(qp.add(g * n + 8 * j));
                *a = _mm256_add_ps(*a, _mm256_mul_ps(x, y));
            }
        }
        acc
    };
    for (g, a) in acc.iter().enumerate() {
        let mut lanes = [0.0f32; 8];
        // SAFETY: storing 8 floats into an 8-float array; AVX as above.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), *a) };
        out[g] = sum_lanes(&lanes) + dot_tail(&q[g * n + 8 * nc..(g + 1) * n], kr);
    }
}

/// `lanes.iter().sum()`: in order from -0.0.
#[inline(always)]
fn sum_lanes(lanes: &[f32; 8]) -> f32 {
    let mut s = -0.0f32;
    for &v in lanes {
        s += v;
    }
    s
}

/// The products a[i] * b[i] summed in order from -0.0 (as `sum` does).
#[inline(always)]
fn dot_tail(a: &[f32], b: &[f32]) -> f32 {
    let mut s = -0.0f32;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// Dot product with 8 independent partial sums. A plain `.sum()` must add in
/// order, so it can't use SIMD lanes; splitting it lets the compiler vectorize.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    sum_lanes(&acc) + dot_tail(ar, br)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tiny well-formed qwen2: dim 64, 2 heads (1 kv), ffn 128, vocab 8, `layers`
    /// layers; `edit` changes the tensor list before the file is built.
    fn tiny(edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>)) -> std::io::Result<()> {
        tiny_meta(edit, |_| {})
    }

    fn tiny_meta(
        edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>),
        edit_meta: impl FnOnce(&mut Vec<(&'static str, Val)>),
    ) -> std::io::Result<()> {
        let mut ts: Vec<(String, Vec<u64>, u32)> = vec![
            ("token_embd.weight".into(), vec![64, 8], Q8_0),
            ("output_norm.weight".into(), vec![64], F32),
        ];
        for i in 0..2 {
            for (n, d, ty) in [
                ("attn_norm.weight", vec![64], F32),
                ("attn_q.weight", vec![64, 64], Q8_0),
                ("attn_k.weight", vec![64, 32], Q8_0),
                ("attn_v.weight", vec![64, 32], Q8_0),
                ("attn_q.bias", vec![64], F32),
                ("attn_output.weight", vec![64, 64], Q8_0),
                ("ffn_norm.weight", vec![64], F32),
                ("ffn_gate.weight", vec![64, 128], Q8_0),
                ("ffn_up.weight", vec![64, 128], Q8_0),
                ("ffn_down.weight", vec![128, 64], Q8_0),
            ] {
                ts.push((format!("blk.{i}.{n}"), d, ty));
            }
        }
        edit(&mut ts);
        let mut meta = vec![
            ("general.architecture", Val::Str("qwen2".into())),
            ("qwen2.block_count", Val::U32(2)),
            ("qwen2.embedding_length", Val::U32(64)),
            ("qwen2.attention.head_count", Val::U32(2)),
            ("qwen2.attention.head_count_kv", Val::U32(1)),
            (
                "tokenizer.ggml.tokens",
                Val::Arr((0..8).map(|i| Val::Str(format!("t{i}"))).collect()),
            ),
        ];
        edit_meta(&mut meta);
        let ts: Vec<(&str, &[u64], u32)> = ts
            .iter()
            .map(|(n, d, t)| (n.as_str(), &d[..], *t))
            .collect();
        check_supported(&crate::gguf::build_for_test(&meta, &ts))
    }

    fn rejects(edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>), want: &str) {
        let e = tiny(edit).expect_err("malformed model accepted");
        assert!(e.to_string().contains(want), "{e}");
    }

    #[test]
    fn check_supported_accepts_tiny_model() {
        tiny(|_| {}).expect("tiny model");
    }

    #[test]
    fn check_supported_rejects_bad_models() {
        let set = |name: &'static str, dims: Vec<u64>, ty: u32| {
            move |ts: &mut Vec<(String, Vec<u64>, u32)>| {
                let t = ts.iter_mut().find(|t| t.0 == name).unwrap();
                (t.1, t.2) = (dims, ty);
            }
        };
        rejects(
            |ts| ts.retain(|t| t.0 != "blk.1.ffn_up.weight"),
            "missing tensor blk.1.ffn_up.weight",
        );
        // Q4_0: a real GGUF type the kernels don't run; 99: unknown.
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], 2),
            "Q4_0 (2) is not supported",
        );
        rejects(
            set("blk.1.ffn_down.weight", vec![128, 64], 99),
            "?? (99) is not supported",
        );
        rejects(
            set("blk.0.attn_k.weight", vec![64, 64], Q8_0),
            "attn_k.weight: shape",
        );
        rejects(
            set("blk.0.ffn_down.weight", vec![64, 64], Q8_0),
            "ffn_down.weight: shape",
        );
        rejects(
            set("output_norm.weight", vec![32], F32),
            "output_norm.weight: shape",
        );
        // K-quant rows must be whole 256-blocks: 64 isn't.
        rejects(
            set("blk.0.attn_q.weight", vec![64, 64], Q4_K),
            "attn_q.weight: shape",
        );
        rejects(set("token_embd.weight", vec![64, 4], Q8_0), "8 tokens");
        let meta = |edit: fn(&mut Vec<(&'static str, Val)>), want: &str| {
            let e = tiny_meta(|_| {}, edit).expect_err("malformed metadata accepted");
            assert!(e.to_string().contains(want), "{e}");
        };
        meta(
            |m| m.retain(|kv| kv.0 != "qwen2.block_count"),
            "qwen2.block_count is missing",
        );
        meta(
            |m| m[3].1 = Val::Str("2".into()),
            "head_count is missing or not a number",
        );
        meta(|m| m[3].1 = Val::U32(0), "not a positive integer");
        meta(|m| m[4].1 = Val::U32(3), "not a multiple of head_count_kv");
        meta(|m| m[1].1 = Val::U32(3), "missing tensor blk.2.");
        meta(
            |m| m.retain(|kv| kv.0 != "general.architecture"),
            "'<none>' is not supported",
        );
    }

    /// Batched prefill in uneven chunks must leave the same cache, and so give
    /// the same next-token logits, bit for bit, as feeding one token at a time.
    #[test]
    fn feed_batch_matches_feed() {
        for repack in [true, false] {
            // A fresh load each: repacking releases the originals.
            let Some(m) = crate::testutil::model() else {
                return;
            };
            let net = Qwen2::new_with(&m, repack);
            let repacked = net.layers.iter().filter(|l| l.w_down.r8.is_some()).count();
            assert_eq!(
                repacked > 0,
                repack,
                "repack {repack}: {repacked} ffn_down repacked"
            );
            feed_batch_matches_feed_on(&net);
        }
    }

    /// The r8 kernels against the per-row ones on the real model: same
    /// prompt, logits compared. Prints the deltas only. This random-token
    /// prompt is chaotic: today's per-row kernels alone disagree with
    /// themselves here (AXON_SIMD=scalar argmax 220, avx2/vnni 2090, max
    /// |dlogprob| ~0.3), so r8's delta of the same size proves nothing either
    /// way. The bound is only a gross-bug tripwire; accuracy is asserted per
    /// row against f64 (`repack_rows_match_exact_on_model`, quant r8 tests)
    /// and on real text by `matches_ollama`.
    #[test]
    fn repack_logit_deltas() {
        if crate::testutil::model_path().is_none() {
            return;
        }
        let ids: Vec<u32> = (0..40u64)
            .map(|i| (i * 2_654_435_761 % 150_000) as u32)
            .collect();
        let run = |repack: bool| {
            let m = crate::testutil::model().unwrap();
            let net = Qwen2::new_with(&m, repack);
            let mut c = net.new_cache(ids.len() + 1);
            net.feed_batch(&ids[..ids.len() - 1], 0, &mut c);
            net.forward(ids[ids.len() - 1], ids.len() - 1, &mut c)
        };
        let (a, b) = (run(true), run(false));
        let lse = |l: &[f32]| {
            let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            mx + l.iter().map(|v| (v - mx).exp()).sum::<f32>().ln()
        };
        let (la, lb) = (lse(&a), lse(&b));
        let mut dl = 0f32;
        let mut dlp = 0f32;
        for (x, y) in a.iter().zip(&b) {
            dl = dl.max((x - y).abs());
            dlp = dlp.max(((x - la) - (y - lb)).abs());
        }
        let argmax = |l: &[f32]| (0..l.len()).max_by(|&i, &j| l[i].total_cmp(&l[j])).unwrap();
        let spread = a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - a.iter().copied().fold(f32::INFINITY, f32::min);
        println!(
            "repack vs per-row: max |dlogit| {dl:.2e} (logit range {spread:.1}), \
             max |dlogprob| {dlp:.2e}, argmax {} vs {}",
            argmax(&a),
            argmax(&b)
        );
        let hash = |l: &[f32]| {
            l.iter()
                .fold(0u64, |h, v| h.rotate_left(5) ^ v.to_bits() as u64)
        };
        println!(
            "logit hashes: r8 {:016x} per-row {:016x}; per-row argmax {} top {:.4}",
            hash(&a),
            hash(&b),
            argmax(&b),
            b[argmax(&b)]
        );
        assert!(dlp < 1.0, "r8 moved a logprob by {dlp}");
    }

    /// Every repacked matrix of the real model: r8 rows vs the per-row
    /// kernel vs an f64 dot of the dequantized row and quantized input, on a
    /// real activation (the layer's own input for one token).
    #[test]
    fn repack_rows_match_exact_on_model() {
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let net = Qwen2::new_with(&m, true);
        // The repack released the originals from `m`: raw bytes from a second load.
        let raw = crate::testutil::model().unwrap();
        let mut x = net.embed(9707);
        let mut worst = (0f64, 0f64);
        for layer in &net.layers {
            let h = net.rmsnorm(&x, &layer.ffn_norm);
            let act = net.matvec_swiglu(&layer.w_gate, &layer.w_up, &h);
            for (w, inp) in [
                (&layer.w_down, &act),
                (&layer.w_gate, &h),
                (&layer.w_up, &h),
            ] {
                let Some(_) = &w.r8 else { continue };
                let data = raw.tensor_data(w.t);
                let plain = Weight {
                    t: w.t,
                    data: Some(data),
                    r8: None,
                };
                let a = net.matvec(w, inp);
                let b = net.matvec(&plain, inp);
                let n = inp.len();
                let rb = row_bytes(w.qt(), n);
                // quantized activations, dequantized
                let mut xq = vec![0f64; n];
                {
                    let mut tmp = vec![0f32; n];
                    for (i, c) in inp.chunks(256).enumerate() {
                        let mut q = [0i8; 256];
                        let amax = c.iter().fold(0f32, |m, v| m.max(v.abs()));
                        let max = *c.iter().find(|v| v.abs() == amax).unwrap();
                        let is = -127.0 / max;
                        for (k, v) in c.iter().enumerate() {
                            q[k] = ((is * v).round_ties_even() as i32).min(127) as i8;
                        }
                        for k in 0..256 {
                            tmp[i * 256 + k] = q[k] as f32 * (1.0 / is);
                        }
                    }
                    for (d, s) in xq.iter_mut().zip(&tmp) {
                        *d = *s as f64;
                    }
                }
                let mut wr = vec![0f32; n];
                for r in 0..w.rows() {
                    dequant(w.qt(), &data[r * rb..(r + 1) * rb], &mut wr);
                    let exact: f64 = wr.iter().zip(&xq).map(|(p, q)| *p as f64 * q).sum();
                    let mag: f64 = wr.iter().zip(&xq).map(|(p, q)| (*p as f64 * q).abs()).sum();
                    let ea = (a[r] as f64 - exact).abs() / mag;
                    let eb = (b[r] as f64 - exact).abs() / mag;
                    worst.0 = worst.0.max(ea);
                    worst.1 = worst.1.max(eb);
                    assert!(
                        ea < 1e-5,
                        "{} row {r}: r8 {} per-row {} exact {exact}",
                        w.t.name,
                        a[r],
                        b[r]
                    );
                }
            }
            let d = net.matvec(&layer.w_down, &act);
            for i in 0..x.len() {
                x[i] += d[i];
            }
        }
        println!(
            "max rel err vs f64: r8 {:.2e}, per-row {:.2e}",
            worst.0, worst.1
        );
    }

    /// Releasing repacked originals must never hand out zeros, on the
    /// anonymous copy (where released pages ARE zeros) and on a file mapping.
    /// The output matrix is forced to the (tied) embedding, so if the
    /// embedding is a K-quant (1.5B+: AXON_TEST_MODEL) it's repacked as the
    /// head while `embed` keeps reading its raw rows.
    #[test]
    fn release_keeps_embedding_on_copy_and_mmap() {
        let Some(fresh) = crate::testutil::model_with(Some(false)) else {
            return;
        };
        let ids = [9707u32, 11, 1879, 0, 151642, 77, 5000];
        let mut logits: Vec<Vec<f32>> = Vec::new();
        for copy in [true, false] {
            let m = crate::testutil::model_with(Some(copy)).unwrap();
            assert_eq!(m.file_backed(), !copy);
            let net = Qwen2::build(&m, true, true);
            let repacked: Vec<&Tensor> = net
                .layers
                .iter()
                .flat_map(|l| [&l.w_down, &l.w_gate, &l.wq])
                .filter(|w| w.r8.is_some())
                .map(|w| w.t)
                .collect();
            assert!(!repacked.is_empty(), "nothing repacked");
            // The embedding (= the tied head) still reads its real bytes.
            assert_eq!(
                m.tensor_data(net.token_embd),
                fresh.tensor_data(net.token_embd)
            );
            for &id in &ids {
                assert_eq!(net.embed(id), {
                    let mut x = vec![0f32; net.dim];
                    let rb = row_bytes(net.token_embd.qtype.id, net.dim);
                    let t = id as usize;
                    dequant(
                        net.token_embd.qtype.id,
                        &fresh.tensor_data(net.token_embd)[t * rb..(t + 1) * rb],
                        &mut x,
                    );
                    x
                });
            }
            // A released original: the copy refuses it, the file map faults
            // the same bytes back in.
            let t = repacked[0];
            if copy {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    m.tensor_data(t).len()
                }));
                assert!(r.is_err(), "released tensor readable on the anonymous copy");
            } else {
                assert_eq!(m.tensor_data(t), fresh.tensor_data(t));
            }
            let mut c = net.new_cache(ids.len());
            net.feed_batch(&ids[..ids.len() - 1], 0, &mut c);
            logits.push(net.forward(ids[ids.len() - 1], ids.len() - 1, &mut c));
            println!(
                "copy {copy}: head repacked {}, {} sampled repacked tensors",
                net.output.r8.is_some(),
                repacked.len()
            );
        }
        // Same weights either way: bit-identical logits.
        let eq = logits[0]
            .iter()
            .zip(&logits[1])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(eq, "copy and file map give different logits");
    }

    fn feed_batch_matches_feed_on(net: &Qwen2) {
        // 150 spread-out ordinary token ids (Qwen2's special tokens start at 151643).
        let ids: Vec<u32> = (0..150u64)
            .map(|i| (i * 2_654_435_761 % 150_000) as u32)
            .collect();
        let next = 9707;
        let mut a = net.new_cache(ids.len() + 1);
        let mut b = net.new_cache(ids.len() + 1);
        for (pos, &id) in ids.iter().enumerate() {
            net.feed(id, pos, &mut a);
        }
        let mut s = 0;
        // Every verify-sized batch (1..=9, so every padding remainder), an
        // odd one, a full prefill batch and the rest.
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 64, ids.len() - 122] {
            net.feed_batch(&ids[s..s + n], s, &mut b);
            s += n;
        }
        assert_eq!(s, ids.len());
        let la = net.forward(next, ids.len(), &mut a);
        let lb = net.forward(next, ids.len(), &mut b);

        let first_diff = |x: &[f32], y: &[f32]| {
            assert_eq!(x.len(), y.len());
            (0..x.len()).find(|&i| x[i].to_bits() != y[i].to_bits())
        };
        for l in 0..net.n_layers {
            for (what, x, y) in [("k", &a.k[l], &b.k[l]), ("v", &a.v[l], &b.v[l])] {
                if let Some(i) = first_diff(&x[..], &y[..]) {
                    panic!(
                        "layer {l} {what} differs at pos {} elem {}: {} vs {}",
                        i / a.kv_dim,
                        i % a.kv_dim,
                        x[i],
                        y[i]
                    );
                }
            }
        }
        if let Some(i) = first_diff(&la[..], &lb[..]) {
            panic!("logit {i} differs: {} vs {}", la[i], lb[i]);
        }
    }

    /// The grouped scorer must equal `dot` per query, bit for bit.
    #[test]
    fn dot_group_matches_dot() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
        };
        for n in [64, 37, 8, 5] {
            for g in [1, 2, 7, 9, 14] {
                for _ in 0..20 {
                    let q: Vec<f32> = (0..g * n).map(|_| rnd()).collect();
                    let k: Vec<f32> = (0..n).map(|_| rnd()).collect();
                    let mut out = vec![0.0f32; g];
                    // SAFETY: V3 false needs nothing.
                    unsafe { dot_group::<false>(&q, &k, &mut out) };
                    for (i, &o) in out.iter().enumerate() {
                        let want = dot(&q[i * n..(i + 1) * n], &k);
                        assert_eq!(o.to_bits(), want.to_bits(), "n={n} g={g} head {i}");
                    }
                }
            }
        }
    }

    /// The shared-pass V sum must equal summing one head at a time (today's
    /// per-head order: `o += w * v` over t from 0.0, as `weighted_sum`), bit
    /// for bit, for value rows strided inside a bigger buffer.
    #[test]
    fn vsum_matches_per_head() {
        fn weighted_sum<'v, const H: usize>(
            n: usize,
            w: impl Fn(usize) -> f32,
            v: impl Fn(usize) -> &'v [f32],
            o: &mut [f32; H],
        ) {
            let mut acc = [0.0f32; H];
            for t in 0..n {
                let w = w(t);
                let v: &[f32; H] = v(t).try_into().unwrap();
                for i in 0..H {
                    acc[i] += w * v[i];
                }
            }
            *o = acc;
        }
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
        };
        for hd in [64, 37, 8, 5] {
            for nh in (1..=9).chain([14]) {
                for np in [1, 2, 7, 64, 300] {
                    let (stride, ws) = (hd + 3, np + 2);
                    let vb: Vec<f32> = (0..np * stride + 1).map(|_| rnd()).collect();
                    let w: Vec<f32> = (0..nh * ws).map(|_| rnd()).collect();
                    let v = |t: usize| &vb[t * stride + 1..t * stride + 1 + hd];
                    let mut got = vec![f32::NAN; nh * hd];
                    vsum_rows(&w, ws, np, nh, &vb[1..], stride, &mut got, hd);
                    for h in 0..nh {
                        let mut want = vec![0.0f32; hd];
                        for t in 0..np {
                            let wt = w[h * ws + t];
                            for (o, &x) in want.iter_mut().zip(v(t)) {
                                *o += wt * x;
                            }
                        }
                        if hd == 64 {
                            let mut ws64 = [0.0f32; 64];
                            weighted_sum::<64>(np, |t| w[h * ws + t], v, &mut ws64);
                            assert_eq!(
                                ws64.map(f32::to_bits)[..],
                                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>()[..]
                            );
                        }
                        for d in 0..hd {
                            assert_eq!(
                                got[h * hd + d].to_bits(),
                                want[d].to_bits(),
                                "hd={hd} nh={nh} np={np} head {h} d {d}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The CPU can run the `v3` variants (as `quant::has_avx2`).
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn can_v3() -> bool {
        let ok = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("bmi1")
            && is_x86_feature_detected!("bmi2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("lzcnt");
        if !ok {
            println!("::warning::no x86-64-v3 on this CPU: v3 variants not compared");
        }
        ok
    }

    /// Bit patterns, every NaN as one canonical NaN: Rust leaves NaN sign
    /// and payload unspecified (x86 keeps the first operand's NaN, so it
    /// varies with register allocation), so only "is NaN" is comparable.
    /// Everything else, -0.0 and infinities included, must match exactly.
    pub(crate) fn bits(x: &[f32]) -> Vec<u32> {
        x.iter()
            .map(|v| if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() })
            .collect()
    }

    /// Random values with the edge cases mixed in: +-0, +-inf, NaN, tiny and
    /// huge magnitudes (`edgy` of them, the rest uniform in [-scale, scale)).
    pub(crate) fn mixed(n: usize, seed: u64, scale: f32, edgy: bool) -> Vec<f32> {
        let mut st = seed | 1;
        let edges = [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-38,
            -3e38,
            1e-45,
        ];
        (0..n)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                if edgy && st.is_multiple_of(17) {
                    edges[(st >> 8) as usize % edges.len()]
                } else {
                    ((st >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0) * scale
                }
            })
            .collect()
    }

    /// Every multiversioned function in this file: the portable and the
    /// x86-64-v3 variant on the same inputs give the same bits (NaNs as in
    /// `bits`). Sizes cover whole 8-lane chunks, tails and empty inputs.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn multiversion_bit_identical() {
        if !can_v3() {
            return;
        }
        let mut seed = 1u64;
        let mut next = || {
            seed += 0x9e37_79b9;
            seed
        };
        for n in [0usize, 1, 5, 8, 37, 64, 128, 896] {
            for edgy in [false, true] {
                let x = mixed(n, next(), 3.0, edgy);
                let w = mixed(n, next(), 1.5, edgy);
                for eps in [1e-6f32, 0.0] {
                    let (mut a, mut b) = (vec![f32::NAN; n], vec![1.0; n]);
                    rmsnorm_into::portable(&x, &w, eps, &mut a);
                    unsafe { rmsnorm_into::v3(&x, &w, eps, &mut b) };
                    assert_eq!(bits(&a), bits(&b), "rmsnorm n={n}");
                }
                let (mut a, mut b) = (x.clone(), x.clone());
                add_assign::portable(&mut a, &w);
                unsafe { add_assign::v3(&mut b, &w) };
                assert_eq!(bits(&a), bits(&b), "add_assign n={n}");

                // softmax over plain scores (no inf/NaN unless edgy)
                for stride in [1, 3] {
                    let src = mixed(n * stride, next(), 8.0, edgy);
                    let (mut a, mut b) = (vec![0.0; n], vec![1.0; n]);
                    softmax_from::portable(&src, stride, &mut a);
                    unsafe { softmax_from::v3(&src, stride, &mut b) };
                    assert_eq!(bits(&a), bits(&b), "softmax_from n={n}");
                }

                let mut best = |slots: usize| {
                    let ys = mixed(n * slots, next(), 2.0, edgy);
                    let mut a = vec![Best::NONE; slots];
                    let mut b = vec![Best::NONE; slots];
                    best_rows::portable(&ys, slots, 11, &mut a);
                    unsafe { best_rows::v3(&ys, slots, 11, &mut b) };
                    for (a, b) in a.iter().zip(&b) {
                        assert_eq!((bits(&[a.v]), a.i), (bits(&[b.v]), b.i), "best n={n}");
                    }
                };
                best(1);
                best(4);
                best(12);
            }
        }
        for hd in [64, 128, 6] {
            let half = hd / 2;
            let table: Vec<(f32, f32)> = mixed(half, next(), 1.0, false)
                .iter()
                .map(|&a| (a.sin(), a.cos()))
                .collect();
            let v = mixed(7 * hd, next(), 2.0, true);
            let (mut a, mut b) = (v.clone(), v.clone());
            rope_heads::portable(&mut a, hd, &table);
            unsafe { rope_heads::v3(&mut b, hd, &table) };
            assert_eq!(bits(&a), bits(&b), "rope hd={hd}");
        }
        for (rows, slots) in [(896, 4), (37, 12), (8, 1), (0, 4)] {
            let y = mixed(rows * slots, next(), 2.0, true);
            for b in 0..slots {
                let ga = gather_col::portable(&y, slots, b, rows);
                let gb = unsafe { gather_col::v3(&y, slots, b, rows) };
                assert_eq!(bits(&ga), bits(&gb), "gather_col");
                let x = mixed(rows, next(), 2.0, true);
                let (mut a, mut c) = (x.clone(), x.clone());
                add_col::portable(&mut a, &y, slots, b);
                unsafe { add_col::v3(&mut c, &y, slots, b) };
                assert_eq!(bits(&a), bits(&c), "add_col");
            }
        }
        // Attention pieces, on the model's shapes and odd ones.
        for (hd, group, n_kv) in [(64, 7, 2), (64, 6, 2), (128, 3, 1), (37, 9, 2), (5, 1, 1)] {
            let kv_dim = n_kv * hd;
            for np in [1usize, 2, 7, 33, 300] {
                let qg = mixed(group * hd, next(), 4.0, false);
                let kc = mixed((np + 2) * kv_dim, next(), 1.0, false);
                let vc = mixed((np + 2) * kv_dim, next(), 1.0, false);
                let scale = 1.0 / (hd as f32).sqrt();
                let kvh = n_kv - 1;

                let (mut a, mut b) = (vec![0.0; group], vec![1.0; group]);
                // SAFETY: can_v3() above.
                unsafe { dot_group::<false>(&qg, &kc[..hd], &mut a) };
                unsafe { dot_group::<true>(&qg, &kc[..hd], &mut b) };
                assert_eq!(bits(&a), bits(&b), "dot_group hd={hd} g={group}");

                let (mut a, mut b) = (vec![0.0; np], vec![1.0; np]);
                score_softmax::portable(&qg[..hd], &kc, kv_dim, kvh * hd, scale, &mut a);
                unsafe { score_softmax::v3(&qg[..hd], &kc, kv_dim, kvh * hd, scale, &mut b) };
                assert_eq!(bits(&a), bits(&b), "score_softmax hd={hd} np={np}");

                let tc = np.div_ceil(3);
                let t0 = np - np.min(tc);
                let (mut a, mut b) = (vec![0.0; tc * group], vec![1.0; tc * group]);
                let off = kvh * hd;
                score_tile::portable(&qg, &kc, kv_dim, hd, off, t0..np, scale, &mut a);
                unsafe { score_tile::v3(&qg, &kc, kv_dim, hd, off, t0..np, scale, &mut b) };
                assert_eq!(bits(&a), bits(&b), "score_tile hd={hd} np={np}");

                let wts = mixed(group * (np + 1), next(), 1.0, false);
                let (mut a, mut b) = (vec![0.0; group * hd], vec![1.0; group * hd]);
                let vb = &vc[kvh * hd..];
                vsum_rows::portable(&wts, np + 1, np, group, vb, kv_dim, &mut a, hd);
                unsafe { vsum_rows::v3(&wts, np + 1, np, group, vb, kv_dim, &mut b, hd) };
                assert_eq!(bits(&a), bits(&b), "vsum_rows hd={hd} np={np}");

                let (mut sa, mut sb) = (vec![0.0; (np + 1) * group], vec![1.0; (np + 1) * group]);
                let (mut a, mut b) = (vec![0.0; group * hd], vec![1.0; group * hd]);
                attend_causal::portable(&qg, &kc, &vc, kv_dim, hd, kvh, np, scale, &mut sa, &mut a);
                unsafe {
                    attend_causal::v3(&qg, &kc, &vc, kv_dim, hd, kvh, np, scale, &mut sb, &mut b)
                };
                assert_eq!(bits(&a), bits(&b), "attend_causal hd={hd} np={np}");

                // and the prefill task equals decode `attend` for the same heads
                let mut q = vec![0.0f32; n_kv * group * hd];
                q[kvh * group * hd..].copy_from_slice(&qg);
                let mut got = vec![0.0f32; q.len()];
                let mut buf = Vec::new();
                attend(&q, &kc, &vc, kv_dim, hd, np, None, &mut buf, &mut got);
                assert_eq!(
                    bits(&got[kvh * group * hd..]),
                    bits(&a),
                    "attend_causal vs attend"
                );
            }
        }
    }

    /// Decode attention, per-head and every tiling, must equal a plain
    /// one-head-at-a-time reference bit for bit: `dot` * scale, softmax (max,
    /// exp + sum in t order, * 1/sum), then o += w * v over t from 0.0. The
    /// cache has extra rows past np that must not be read, and the scratch is
    /// reused across calls with stale contents.
    #[test]
    fn attend_matches_per_head() {
        let mut seed = 0x6a09_e667_f3bc_c909u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
        };
        let mut buf = Vec::new();
        for hd in [64, 128] {
            for group in [6, 7] {
                for n_kv in [2, 4] {
                    let (kv_dim, nq) = (n_kv * hd, n_kv * group * hd);
                    let nps = (1..300).chain([1000, 2047, 3001]);
                    for np in nps {
                        let q: Vec<f32> = (0..nq).map(|_| rnd() * 4.0).collect();
                        let kc: Vec<f32> = (0..(np + 3) * kv_dim).map(|_| rnd()).collect();
                        let vc: Vec<f32> = (0..(np + 3) * kv_dim).map(|_| rnd()).collect();
                        let scale = 1.0 / (hd as f32).sqrt();
                        let mut want = vec![0.0f32; nq];
                        for qh in 0..n_kv * group {
                            let kvh = qh / group;
                            let qv = &q[qh * hd..(qh + 1) * hd];
                            let mut w: Vec<f32> = (0..np)
                                .map(|t| {
                                    let k0 = t * kv_dim + kvh * hd;
                                    dot(qv, &kc[k0..k0 + hd]) * scale
                                })
                                .collect();
                            let max = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let mut sum = 0.0f32;
                            for e in w.iter_mut() {
                                *e = (*e - max).exp();
                                sum += *e;
                            }
                            let inv = 1.0 / sum;
                            w.iter_mut().for_each(|e| *e *= inv);
                            let o = &mut want[qh * hd..(qh + 1) * hd];
                            for (t, &wt) in w.iter().enumerate() {
                                let v0 = t * kv_dim + kvh * hd;
                                for (o, &v) in o.iter_mut().zip(&vc[v0..v0 + hd]) {
                                    *o += wt * v;
                                }
                            }
                        }
                        let tiles = [
                            None,
                            Some(1),
                            Some(7),
                            Some(64),
                            Some(np.div_ceil(5)),
                            decode_tile(np.max(TILE_MIN_NP), n_kv),
                        ];
                        for tile in tiles {
                            let mut got = vec![f32::NAN; nq];
                            attend(&q, &kc, &vc, kv_dim, hd, np, tile, &mut buf, &mut got);
                            if let Some(i) =
                                (0..nq).find(|&i| got[i].to_bits() != want[i].to_bits())
                            {
                                panic!(
                                    "hd={hd} group={group} n_kv={n_kv} np={np} tile={tile:?}: \
                                     elem {i} {} vs {}",
                                    got[i], want[i]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run each prompt in tests/fixtures/gen/ and follow Ollama's greedy path token by
    /// token. At every step axon must give Ollama's token nearly the same logprob, and
    /// must pick the same token unless Ollama's top two were a near-tie.
    #[test]
    fn matches_ollama() {
        const MAX_DLOGPROB: f32 = 0.15;
        let dir = crate::testutil::fixtures().join("gen");
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|f| f.ends_with(".json"))
            .collect();
        names.sort();
        assert!(!names.is_empty(), "no gen fixtures in {}", dir.display());
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
        let net = Qwen2::new(&m);

        for name in &names {
            let fx = crate::testutil::read_json(&format!("gen/{name}"));
            let prompt = fx["prompt"].as_str().expect("prompt");
            let steps = fx["steps"].as_array().expect("steps");
            let ids = tok.encode(prompt);
            let mut cache = net.new_cache(ids.len() + steps.len() + 1);
            let (&last, rest) = ids.split_last().expect("prompt has tokens");
            for (pos, &id) in rest.iter().enumerate() {
                net.feed(id, pos, &mut cache);
            }
            let mut logits = net.forward(last, rest.len(), &mut cache);
            let mut worst = 0f32;
            for (i, s) in steps.iter().enumerate() {
                let want = s["id"].as_u64().expect("id") as usize;
                let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = mx + logits.iter().map(|l| (l - mx).exp()).sum::<f32>().ln();
                let lp = logits[want] - lse;
                let d = (lp - s["logprob"].as_f64().expect("logprob") as f32).abs();
                worst = worst.max(d);
                let ctx = format!("{name} step {i} ({:?})", s["token"].as_str().unwrap_or("?"));
                assert!(d <= MAX_DLOGPROB, "{ctx}: logprob off by {d:.3}");
                if !s["near_tie"].as_bool().unwrap_or(false) {
                    let argmax = (0..logits.len())
                        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                        .unwrap();
                    assert_eq!(
                        argmax,
                        want,
                        "{ctx}: axon picked {:?}",
                        tok.decode(&[argmax as u32])
                    );
                }
                logits = net.forward(want as u32, ids.len() + i, &mut cache);
            }
            println!("{name}: {} steps, max |dlogprob| {worst:.3}", steps.len());
        }
    }
}
