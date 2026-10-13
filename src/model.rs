//! qwen2 forward pass. Matrix-vector products run across all cores (rayon);
//! everything else is plain loops with f32 activations.

use crate::gguf::{Model, Tensor, Val};
use crate::quant::{dequant, dot_row, row_bytes};
use rayon::prelude::*;

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
        let (n_in, n_out) = (w.dims[0] as usize, w.dims[1] as usize);
        assert_eq!(x.len(), n_in, "matvec {}: input len", w.name);
        let rb = row_bytes(w.qtype.id, n_in);
        let data = self.m.tensor_data(w);
        // Every output row is an independent dot product, so split rows across
        // cores. Chunks of 64 rows keep scheduling overhead small, and rayon's
        // work stealing evens out the fast and efficiency cores.
        let mut out = vec![0f32; n_out];
        out.par_chunks_mut(64).enumerate().for_each(|(c, ys)| {
            for (j, y) in ys.iter_mut().enumerate() {
                let o = c * 64 + j;
                *y = dot_row(w.qtype.id, &data[o * rb..(o + 1) * rb], x);
            }
        });
        out
    }

    /// RMSNorm: scale x so its root-mean-square is 1, then multiply by a learned weight.
    fn rmsnorm(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (ms + self.eps).sqrt();
        x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
    }

    /// NEOX-style RoPE on one head vector (len head_dim) in place.
    /// Dim i is paired with dim i + head_dim/2, and the pair is rotated by
    /// angle pos * base^(-2i/head_dim). Low i spins fast, high i spins slowly.
    fn rope(&self, v: &mut [f32], pos: usize) {
        let half = self.head_dim / 2;
        for i in 0..half {
            let freq = self.rope_base.powf(-2.0 * i as f64 / self.head_dim as f64);
            let angle = pos as f64 * freq;
            let (sin, cos) = (angle.sin() as f32, angle.cos() as f32);
            let (a, b) = (v[i], v[i + half]);
            v[i] = a * cos - b * sin;
            v[i + half] = a * sin + b * cos;
        }
    }

    /// Run one token at position `pos` and return logits (len vocab).
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        assert!(
            pos < cache.max_ctx,
            "position {pos} exceeds cache size {}",
            cache.max_ctx
        );
        let (dim, hd) = (self.dim, self.head_dim);
        let kv_dim = cache.kv_dim;
        let group = self.n_heads / self.n_kv_heads; // q heads that share one kv head

        // 1. Embedding: the token's row of the embedding matrix is its starting vector.
        let mut x = vec![0.0f32; dim];
        let rb = row_bytes(self.token_embd.qtype.id, dim);
        let t = token as usize;
        dequant(
            self.token_embd.qtype.id,
            &self.m.tensor_data(self.token_embd)[t * rb..(t + 1) * rb],
            &mut x,
        );

        for (l, layer) in self.layers.iter().enumerate() {
            // ---- Attention sub-block ----
            // 2. Normalize, then project into queries, keys and values (+ bias).
            let h = self.rmsnorm(&x, &layer.attn_norm);
            let mut q = self.matvec(layer.wq, &h);
            let mut k = self.matvec(layer.wk, &h);
            let mut v = self.matvec(layer.wv, &h);
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
            for head in q.chunks_mut(hd) {
                self.rope(head, pos);
            }
            for head in k.chunks_mut(hd) {
                self.rope(head, pos);
            }

            // 4. Store this position's k and v in the cache so later tokens can see them.
            cache.k[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            cache.v[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
            let (kc, vc) = (&cache.k[l], &cache.v[l]);

            // 5. Each query head attends over all positions 0..=pos.
            //    GQA: query head qh reads kv head qh / group.
            let scale = 1.0 / (hd as f32).sqrt();
            let mut att_out = vec![0.0f32; dim];
            let mut scores = vec![0.0f32; pos + 1];
            for qh in 0..self.n_heads {
                let kvh = qh / group;
                let qv = &q[qh * hd..(qh + 1) * hd];
                // Score = (q · k_t) / sqrt(head_dim) for every past position t.
                for t in 0..=pos {
                    let kv = &kc[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    scores[t] = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                // Softmax turns scores into weights summing to 1 (subtract max for stability).
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                // Output = weighted sum of the value vectors.
                let out = &mut att_out[qh * hd..(qh + 1) * hd];
                for t in 0..=pos {
                    let vv = &vc[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    let w = scores[t];
                    for d in 0..hd {
                        out[d] += w * vv[d];
                    }
                }
            }

            // 6. Mix the heads back together and add to the residual stream.
            let o = self.matvec(layer.wo, &att_out);
            for i in 0..dim {
                x[i] += o[i];
            }

            // ---- Feed-forward sub-block (SwiGLU) ----
            // 7. Normalize, then gate * up: silu(gate(h)) decides how much of up(h) passes.
            let h = self.rmsnorm(&x, &layer.ffn_norm);
            let gate = self.matvec(layer.w_gate, &h);
            let up = self.matvec(layer.w_up, &h);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            // 8. Project back down to model width and add to the residual stream.
            let d = self.matvec(layer.w_down, &act);
            for i in 0..dim {
                x[i] += d[i];
            }
        }

        // 9. Final norm, then the output head scores every vocab entry.
        let h = self.rmsnorm(&x, &self.output_norm);
        self.matvec(self.output, &h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let mut logits = Vec::new();
            for (pos, &id) in ids.iter().enumerate() {
                logits = net.forward(id, pos, &mut cache);
            }
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
