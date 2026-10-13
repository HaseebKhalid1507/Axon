//! qwen2 forward pass. Matrix-vector products quantize the input to 8 bits and
//! run integer dot products on the worker pool (pool.rs); everything else is
//! plain f32 loops.

use crate::gguf::{Model, Tensor, Val};
use crate::pool::{self, Pool};
use crate::quant::{QuantizedAct, dequant, dot_row_q8, row_bytes};

/// The weights of one transformer block, borrowed from the mmap.
struct Layer<'a> {
    attn_norm: Vec<f32>,
    wq: &'a Tensor,
    wk: &'a Tensor,
    wv: &'a Tensor,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    wo: &'a Tensor,
    ffn_norm: Vec<f32>,
    w_gate: &'a Tensor,
    w_up: &'a Tensor,
    w_down: &'a Tensor,
}

pub struct Qwen2<'a> {
    m: &'a Model,
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    eps: f32,
    rope_base: f64,
    token_embd: &'a Tensor,
    output_norm: Vec<f32>,
    output: &'a Tensor,
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

impl<'a> Qwen2<'a> {
    pub fn new(m: &'a Model) -> Qwen2<'a> {
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
        let output = find_opt(m, "output.weight").unwrap_or(token_embd);
        let output_norm = load_vec(m, find(m, "output_norm.weight"));

        let layers = (0..n_layers)
            .map(|i| {
                let t = |s: &str| find(m, &format!("blk.{i}.{s}"));
                let bias = |s: &str| find_opt(m, &format!("blk.{i}.{s}")).map(|t| load_vec(m, t));
                Layer {
                    attn_norm: load_vec(m, t("attn_norm.weight")),
                    wq: t("attn_q.weight"),
                    wk: t("attn_k.weight"),
                    wv: t("attn_v.weight"),
                    bq: bias("attn_q.bias"),
                    bk: bias("attn_k.bias"),
                    bv: bias("attn_v.bias"),
                    wo: t("attn_output.weight"),
                    ffn_norm: load_vec(m, t("ffn_norm.weight")),
                    w_gate: t("ffn_gate.weight"),
                    w_up: t("ffn_up.weight"),
                    w_down: t("ffn_down.weight"),
                }
            })
            .collect();

        Qwen2 {
            m,
            n_layers,
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab,
            eps,
            rope_base,
            token_embd,
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

    /// y = W x. W has ne1 rows of ne0 values each; y[o] = row_o · x.
    fn matvec(&self, w: &Tensor, x: &[f32]) -> Vec<f32> {
        let [y] = self.matvec_many([w], x);
        y
    }

    /// Several matrices applied to the same input in one parallel pass:
    /// one fork/join and one quantization of x instead of one per matrix.
    /// Returns one output vector per matrix, in order.
    fn matvec_many<const N: usize>(&self, ws: [&Tensor; N], x: &[f32]) -> [Vec<f32>; N] {
        // Quantize x to 8 bits once; every row then does integer multiply-adds
        // against its packed weights instead of decoding them to floats.
        let types = ws.map(|w| w.qtype.id);
        let xq = QuantizedAct::for_types(x, &types);
        // All matrices' rows form one index space; starts[i] is where matrix i begins.
        let mut starts = [0usize; N];
        let mut total = 0;
        for (i, w) in ws.iter().enumerate() {
            assert_eq!(w.dims[0] as usize, x.len(), "matvec {}: input len", w.name);
            starts[i] = total;
            total += w.dims[1] as usize;
        }
        let info = ws.map(|w| {
            let qt = w.qtype.id;
            (qt, row_bytes(qt, x.len()), self.m.tensor_data(w))
        });
        // Every output row is an independent dot product, so split rows across
        // cores. About 4 chunks per thread lets the fast and efficiency cores
        // even out; at most 64 rows keeps chunk claiming cheap.
        let mut out = vec![0f32; total];
        let chunk = (total / (Pool::global().threads() * 4)).clamp(4, 64);
        pool::for_each_chunk_mut(&mut out, chunk, |c, ys| {
            for (j, y) in ys.iter_mut().enumerate() {
                let g = c * chunk + j;
                let t = starts.iter().rposition(|&s| s <= g).unwrap();
                let (qt, rb, data) = info[t];
                let o = g - starts[t];
                *y = dot_row_q8(qt, &data[o * rb..(o + 1) * rb], &xq);
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
    fn matvec_swiglu(&self, gate: &Tensor, up: &Tensor, x: &[f32]) -> Vec<f32> {
        let n = gate.dims[1] as usize;
        for w in [gate, up] {
            assert_eq!(w.dims[0] as usize, x.len(), "matvec {}: input len", w.name);
        }
        assert_eq!(up.dims[1] as usize, n, "swiglu: gate/up row counts differ");
        let xq = QuantizedAct::for_types(x, &[gate.qtype.id, up.qtype.id]);
        let [g, u] = [gate, up].map(|w| {
            let qt = w.qtype.id;
            (qt, row_bytes(qt, x.len()), self.m.tensor_data(w))
        });
        let row = |(qt, rb, data): (u32, usize, &[u8]), r: usize| {
            dot_row_q8(qt, &data[r * rb..(r + 1) * rb], &xq)
        };
        let mut out = vec![0f32; n];
        // Each row is two dot products, so half the rows per chunk of matvec_many.
        let chunk = (n / (Pool::global().threads() * 8)).clamp(4, 32);
        pool::for_each_chunk_mut(&mut out, chunk, |c, ys| {
            for (j, y) in ys.iter_mut().enumerate() {
                let r = c * chunk + j;
                let gv = row(g, r);
                *y = (gv / (1.0 + (-gv).exp())) * row(u, r);
            }
        });
        out
    }

    /// Rows per pool task for a batched matmul over `rows` weight rows: about
    /// 4 chunks per thread. Each row already does one dot per token, so there
    /// is no upper clamp; claiming a chunk is cheap next to its work.
    fn rows_per_chunk(rows: usize) -> usize {
        rows.div_ceil(Pool::global().threads() * 4).max(1)
    }

    /// `matvec_many` for a batch of inputs: res[i][b] = ws[i] · xs[b].
    ///
    /// Rows are split across the pool. A task loads each weight row once and
    /// dots it with every token's input while it's still in L1 (row outer,
    /// token inner), so the weights stream from RAM once per batch, not once
    /// per token. Output is row-major over (row, token), y[r * B + b], so each
    /// task owns a contiguous slice; it is transposed to per-token vectors at
    /// the end. Each dot is the same `dot_row_q8` on the same quantized input
    /// as `matvec_many`, so results are bit-identical.
    fn matmul_many<const N: usize, X: AsRef<[f32]>>(
        &self,
        ws: [&Tensor; N],
        xs: &[X],
    ) -> [Vec<Vec<f32>>; N] {
        let nb = xs.len();
        let n_in = xs[0].as_ref().len();
        let types = ws.map(|w| w.qtype.id);
        let xq: Vec<QuantizedAct> = xs
            .iter()
            .map(|x| {
                assert_eq!(x.as_ref().len(), n_in, "matmul: ragged batch");
                QuantizedAct::for_types(x.as_ref(), &types)
            })
            .collect();
        let mut starts = [0usize; N];
        let mut total = 0;
        for (i, w) in ws.iter().enumerate() {
            assert_eq!(w.dims[0] as usize, n_in, "matmul {}: input len", w.name);
            starts[i] = total;
            total += w.dims[1] as usize;
        }
        let info = ws.map(|w| {
            let qt = w.qtype.id;
            (qt, row_bytes(qt, n_in), self.m.tensor_data(w))
        });
        let mut out = vec![0f32; total * nb];
        let rows = Self::rows_per_chunk(total);
        pool::for_each_chunk_mut(&mut out, rows * nb, |c, ys| {
            for (j, yr) in ys.chunks_mut(nb).enumerate() {
                let g = c * rows + j;
                let t = starts.iter().rposition(|&s| s <= g).unwrap();
                let (qt, rb, data) = info[t];
                let o = g - starts[t];
                let row = &data[o * rb..(o + 1) * rb];
                for (y, xq) in yr.iter_mut().zip(&xq) {
                    *y = dot_row_q8(qt, row, xq);
                }
            }
        });
        std::array::from_fn(|i| {
            let n = ws[i].dims[1] as usize;
            unbatch(&out[starts[i] * nb..(starts[i] + n) * nb], nb)
        })
    }

    /// `matvec_swiglu` for a batch: per row r, per token b, the gate and up
    /// dots combined with the same formula. Layout as in `matmul_many`.
    fn matmul_swiglu<X: AsRef<[f32]>>(
        &self,
        gate: &Tensor,
        up: &Tensor,
        xs: &[X],
    ) -> Vec<Vec<f32>> {
        let nb = xs.len();
        let n_in = xs[0].as_ref().len();
        let n = gate.dims[1] as usize;
        for w in [gate, up] {
            assert_eq!(w.dims[0] as usize, n_in, "matmul {}: input len", w.name);
        }
        assert_eq!(up.dims[1] as usize, n, "swiglu: gate/up row counts differ");
        let xq: Vec<QuantizedAct> = xs
            .iter()
            .map(|x| {
                assert_eq!(x.as_ref().len(), n_in, "matmul: ragged batch");
                QuantizedAct::for_types(x.as_ref(), &[gate.qtype.id, up.qtype.id])
            })
            .collect();
        let [g, u] = [gate, up].map(|w| {
            let qt = w.qtype.id;
            (qt, row_bytes(qt, n_in), self.m.tensor_data(w))
        });
        let mut out = vec![0f32; n * nb];
        let rows = Self::rows_per_chunk(n);
        pool::for_each_chunk_mut(&mut out, rows * nb, |c, ys| {
            for (j, yr) in ys.chunks_mut(nb).enumerate() {
                let r = c * rows + j;
                let gr = &g.2[r * g.1..(r + 1) * g.1];
                let ur = &u.2[r * u.1..(r + 1) * u.1];
                for (y, xq) in yr.iter_mut().zip(&xq) {
                    let gv = dot_row_q8(g.0, gr, xq);
                    *y = (gv / (1.0 + (-gv).exp())) * dot_row_q8(u.0, ur, xq);
                }
            }
        });
        unbatch(&out, nb)
    }

    /// RMSNorm: scale x so its root-mean-square is 1, then multiply by a learned weight.
    fn rmsnorm(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (ms + self.eps).sqrt();
        x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
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

    /// NEOX-style RoPE on one head vector (len head_dim) in place: dim i is
    /// paired with dim i + head_dim/2 and the pair rotated by `table[i]`.
    fn rope(v: &mut [f32], table: &[(f32, f32)]) {
        let half = table.len();
        for (i, &(sin, cos)) in table.iter().enumerate() {
            let (a, b) = (v[i], v[i + half]);
            v[i] = a * cos - b * sin;
            v[i + half] = a * sin + b * cos;
        }
    }

    /// Run one token at position `pos` and return logits (len vocab).
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let x = self.hidden(token, pos, cache);
        // 9. Final norm, then the output head scores every vocab entry.
        let h = self.rmsnorm(&x, &self.output_norm);
        self.matvec(self.output, &h)
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
            &self.m.tensor_data(self.token_embd)[t * rb..(t + 1) * rb],
            &mut x,
        );
        x
    }

    /// Run `tokens[b]` at position `start + b` for every b, for the cache only.
    /// Same result, bit for bit, as `feed` on each token in order, but every
    /// weight row is read once per batch instead of once per token.
    pub fn feed_batch(&self, tokens: &[u32], start: usize, cache: &mut KvCache) {
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
        let n_heads = self.n_heads;
        let group = n_heads / self.n_kv_heads;

        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&t| self.embed(t)).collect();
        let ropes: Vec<_> = (0..nb).map(|b| self.rope_table(start + b)).collect();
        for (l, layer) in self.layers.iter().enumerate() {
            // Steps as in `hidden`, each applied to every token of the batch.
            let hs: Vec<Vec<f32>> = xs
                .iter()
                .map(|x| self.rmsnorm(x, &layer.attn_norm))
                .collect();
            let [mut qs, mut ks, mut vs] = self.matmul_many([layer.wq, layer.wk, layer.wv], &hs);
            for b in 0..nb {
                let (q, k, v) = (&mut qs[b], &mut ks[b], &mut vs[b]);
                if let Some(bias) = &layer.bq {
                    q.iter_mut().zip(bias).for_each(|(a, b)| *a += b);
                }
                if let Some(bias) = &layer.bk {
                    k.iter_mut().zip(bias).for_each(|(a, b)| *a += b);
                }
                if let Some(bias) = &layer.bv {
                    v.iter_mut().zip(bias).for_each(|(a, b)| *a += b);
                }
                for head in q.chunks_mut(hd).chain(k.chunks_mut(hd)) {
                    Self::rope(head, &ropes[b]);
                }
                let pos = start + b;
                cache.k[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k);
                cache.v[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v);
            }
            let (kc, vc) = (&cache.k[l], &cache.v[l]);

            // Causal attention, one (token, query head) pair per pool task:
            // token b sees positions 0..=start + b only. The body is the same
            // arithmetic as in `hidden`.
            let scale = 1.0 / (hd as f32).sqrt();
            let mut att = vec![0.0f32; nb * dim];
            pool::for_each_chunk_mut(&mut att, hd, |c, out| {
                let (b, qh) = (c / n_heads, c % n_heads);
                let pos = start + b;
                let kvh = qh / group;
                let qv = &qs[b][qh * hd..(qh + 1) * hd];
                let head = |t: usize| t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd;
                let mut scores: Vec<f32> =
                    (0..=pos).map(|t| dot(qv, &kc[head(t)]) * scale).collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for (t, &w) in scores.iter().enumerate() {
                    let w = w * inv;
                    for (o, &v) in out.iter_mut().zip(&vc[head(t)]) {
                        *o += w * v;
                    }
                }
            });

            let att_rows: Vec<&[f32]> = att.chunks(dim).collect();
            let [os] = self.matmul_many([layer.wo], &att_rows);
            for (x, o) in xs.iter_mut().zip(&os) {
                for i in 0..dim {
                    x[i] += o[i];
                }
            }

            let hs: Vec<Vec<f32>> = xs
                .iter()
                .map(|x| self.rmsnorm(x, &layer.ffn_norm))
                .collect();
            let acts = self.matmul_swiglu(layer.w_gate, layer.w_up, &hs);
            let [ds] = self.matmul_many([layer.w_down], &acts);
            for (x, d) in xs.iter_mut().zip(&ds) {
                for i in 0..dim {
                    x[i] += d[i];
                }
            }
        }
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
        let group = self.n_heads / self.n_kv_heads; // q heads that share one kv head

        // 1. Embedding: the token's row of the embedding matrix is its starting vector.
        let mut x = self.embed(token);

        let rope = self.rope_table(pos);
        for (l, layer) in self.layers.iter().enumerate() {
            // ---- Attention sub-block ----
            // 2. Normalize, then project into queries, keys and values (+ bias).
            let h = self.rmsnorm(&x, &layer.attn_norm);
            let [mut q, mut k, mut v] = self.matvec_many([layer.wq, layer.wk, layer.wv], &h);
            if let Some(b) = &layer.bq {
                q.iter_mut().zip(b).for_each(|(a, b)| *a += b);
            }
            if let Some(b) = &layer.bk {
                k.iter_mut().zip(b).for_each(|(a, b)| *a += b);
            }
            if let Some(b) = &layer.bv {
                v.iter_mut().zip(b).for_each(|(a, b)| *a += b);
            }

            // 3. Rotate q and k by position (RoPE), one head at a time.
            for head in q.chunks_mut(hd).chain(k.chunks_mut(hd)) {
                Self::rope(head, &rope);
            }

            // 4. Store this position's k and v in the cache so later tokens can see them.
            cache.k[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            cache.v[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
            let (kc, vc) = (&cache.k[l], &cache.v[l]);

            // 5. Each query head attends over all positions 0..=pos, one head per
            //    pool task (the work grows with context, so it can't stay serial).
            //    GQA: query head qh reads kv head qh / group.
            let scale = 1.0 / (hd as f32).sqrt();
            let mut att_out = vec![0.0f32; dim];
            pool::for_each_chunk_mut(&mut att_out, hd, |qh, out| {
                let kvh = qh / group;
                let qv = &q[qh * hd..(qh + 1) * hd];
                let head = |t: usize| t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd;
                // Score = (q · k_t) / sqrt(head_dim) for every past position t.
                let mut scores: Vec<f32> =
                    (0..=pos).map(|t| dot(qv, &kc[head(t)]) * scale).collect();
                // Softmax turns scores into weights summing to 1 (subtract max for stability).
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                // Output = weighted sum of the value vectors.
                for (t, &w) in scores.iter().enumerate() {
                    let w = w * inv;
                    for (o, &v) in out.iter_mut().zip(&vc[head(t)]) {
                        *o += w * v;
                    }
                }
            });

            // 6. Mix the heads back together and add to the residual stream.
            let o = self.matvec(layer.wo, &att_out);
            for i in 0..dim {
                x[i] += o[i];
            }

            // ---- Feed-forward sub-block (SwiGLU) ----
            // 7. Normalize, then gate * up: silu(gate(h)) decides how much of up(h) passes.
            let h = self.rmsnorm(&x, &layer.ffn_norm);
            let act = self.matvec_swiglu(layer.w_gate, layer.w_up, &h);
            // 8. Project back down to model width and add to the residual stream.
            let d = self.matvec(layer.w_down, &act);
            for i in 0..dim {
                x[i] += d[i];
            }
        }

        x
    }
}

/// Transpose a row-major (row, token) buffer, y[r * nb + b], into one vector
/// per token.
fn unbatch(y: &[f32], nb: usize) -> Vec<Vec<f32>> {
    let n = y.len() / nb;
    (0..nb)
        .map(|b| (0..n).map(|r| y[r * nb + b]).collect())
        .collect()
}

/// Dot product with 8 independent partial sums. A plain `.sum()` must add in
/// order, so it can't use SIMD lanes; splitting it lets the compiler vectorize.
#[inline]
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
    let tail: f32 = ar.iter().zip(br).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Batched prefill in uneven chunks must leave the same cache, and so give
    /// the same next-token logits, bit for bit, as feeding one token at a time.
    #[test]
    fn feed_batch_matches_feed() {
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let net = Qwen2::new(&m);
        // 100 spread-out ordinary token ids (Qwen2's special tokens start at 151643).
        let ids: Vec<u32> = (0..100u64)
            .map(|i| (i * 2_654_435_761 % 150_000) as u32)
            .collect();
        let next = 9707;
        let mut a = net.new_cache(ids.len() + 1);
        let mut b = net.new_cache(ids.len() + 1);
        for (pos, &id) in ids.iter().enumerate() {
            net.feed(id, pos, &mut a);
        }
        let mut s = 0;
        for n in [1, 7, 64, ids.len() - 72] {
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
