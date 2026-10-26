//! deepseek2 (MLA) on the CPU: the synthetic fixture against an
//! independent f64 reference in the *expanded* form, the decode == batch ==
//! verify contract, the router rule, the loader's refusals; and, when the
//! Moonlight GGUF is in models/, the same contract on it.

use super::tests::{RawTensors, max_diff};
use super::*;
use crate::gguf::Val;

/// Fixture shapes: dim 64, 2 heads of nope 32 + rope 16, latent 32, values
/// 32 per head, a dense lead layer (ffn 128) then 2 MoE layers (4 experts
/// of 64, top-2, 1 shared), vocab 64; latent 64 so that the key row
/// (80) differs from nope + rope (48). (The spec's nope 16 / rope 8 don't
/// make whole Q8_0 rows: 32-wide blocks.)
pub(crate) const FX: Fx = Fx {
    dim: 64,
    nh: 2,
    nope: 32,
    rope: 16,
    lat: 64,
    hv: 32,
    q_lora: 32,
    ffn: 128,
    n_exp: 4,
    k_exp: 2,
    ffe: 64,
    vocab: 64,
    nl: 3,
};

#[derive(Clone, Copy)]
pub(crate) struct Fx {
    pub dim: usize,
    pub nh: usize,
    pub nope: usize,
    pub rope: usize,
    pub lat: usize,
    pub hv: usize,
    pub q_lora: usize,
    pub ffn: usize,
    pub n_exp: usize,
    pub k_exp: usize,
    pub ffe: usize,
    pub vocab: usize,
    pub nl: usize,
}

/// Fixture options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Opt {
    /// attn_q_a / attn_q_a_norm / attn_q_b (V3, K2) instead of attn_q.
    pub q_lora: bool,
    /// YaRN x4 from 64 positions, yarn_log_multiplier 0.1 (V3/K2's rule).
    pub yarn: bool,
}

/// The fixture's vocabulary: GPT-2 byte-level texts of `n` (<= 65) single
/// bytes (newline first), so llama.cpp's BPE loader takes it too.
fn vocab_texts(n: usize) -> Vec<String> {
    let enc = crate::tokenizer::byte_encoder();
    let mut v = vec![enc[b'\n' as usize].to_string()];
    v.extend(
        (b'a'..=b'z')
            .chain(b'A'..=b'Z')
            .chain(b'0'..=b'9')
            .chain([b' ', b'.'])
            .map(|b| enc[b as usize].to_string()),
    );
    v.truncate(n);
    v
}

/// The metadata, tensors (name, dims, f32 values) of the fixture.
#[allow(clippy::type_complexity)]
fn fixture_parts(o: Opt) -> (Vec<(&'static str, Val)>, Vec<(String, Vec<u64>, Vec<f32>)>) {
    let f = FX;
    let (hk, kr) = (f.nope + f.rope, f.lat + f.rope);
    let mut st = 0x2545_f491_4f6c_dd1du64;
    let mut uni = move |lo: f32, hi: f32| {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        lo + (hi - lo) * ((st >> 40) as f32 / (1u64 << 24) as f32)
    };
    let mut ts: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut mat = |name: String, cols: usize, rows: usize, n: usize, rms: f32| {
        let a = rms * (3.0 / cols as f32).sqrt();
        let v = (0..cols * rows * n).map(|_| uni(-a, a)).collect();
        let mut d = vec![cols as u64, rows as u64];
        if n > 1 {
            d.push(n as u64);
        }
        ts.push((name, d, v));
    };
    mat("token_embd.weight".into(), f.dim, f.vocab, 1, 1.7);
    mat("output.weight".into(), f.dim, f.vocab, 1, 1.0);
    for i in 0..f.nl {
        let n = |s: &str| format!("blk.{i}.{s}");
        if o.q_lora {
            mat(n("attn_q_a.weight"), f.dim, f.q_lora, 1, 1.0);
            mat(n("attn_q_b.weight"), f.q_lora, f.nh * hk, 1, 2.0);
        } else {
            mat(n("attn_q.weight"), f.dim, f.nh * hk, 1, 2.0);
        }
        mat(n("attn_kv_a_mqa.weight"), f.dim, kr, 1, 2.0);
        mat(n("attn_k_b.weight"), f.nope, f.lat, f.nh, 1.0);
        mat(n("attn_v_b.weight"), f.lat, f.hv, f.nh, 1.0);
        mat(n("attn_output.weight"), f.nh * f.hv, f.dim, 1, 1.0);
        if i == 0 {
            mat(n("ffn_gate.weight"), f.dim, f.ffn, 1, 1.0);
            mat(n("ffn_up.weight"), f.dim, f.ffn, 1, 1.0);
            mat(n("ffn_down.weight"), f.ffn, f.dim, 1, 1.0);
        } else {
            mat(n("ffn_gate_inp.weight"), f.dim, f.n_exp, 1, 3.0);
            mat(n("ffn_gate_exps.weight"), f.dim, f.ffe, f.n_exp, 1.0);
            mat(n("ffn_up_exps.weight"), f.dim, f.ffe, f.n_exp, 1.0);
            mat(n("ffn_down_exps.weight"), f.ffe, f.dim, f.n_exp, 1.0);
            mat(n("ffn_gate_shexp.weight"), f.dim, f.ffe, 1, 1.0);
            mat(n("ffn_up_shexp.weight"), f.dim, f.ffe, 1, 1.0);
            mat(n("ffn_down_shexp.weight"), f.ffe, f.dim, 1, 1.0);
        }
    }
    let mut vecs = |name: String, n: usize, lo: f32, hi: f32| {
        ts.push((name, vec![n as u64], (0..n).map(|_| uni(lo, hi)).collect()))
    };
    vecs("output_norm.weight".into(), f.dim, 0.8, 1.2);
    for i in 0..f.nl {
        let n = |s: &str| format!("blk.{i}.{s}");
        vecs(n("attn_norm.weight"), f.dim, 0.8, 1.2);
        vecs(n("ffn_norm.weight"), f.dim, 0.8, 1.2);
        vecs(n("attn_kv_a_norm.weight"), f.lat, 0.5, 2.0);
        if o.q_lora {
            vecs(n("attn_q_a_norm.weight"), f.q_lora, 0.5, 2.0);
        }
        if i > 0 {
            // Selection bias of the size of the sigmoid's spread: it moves
            // the picks.
            vecs(n("exp_probs_b.bias"), f.n_exp, -0.3, 0.3);
        }
    }
    let k = |s: &str| -> &'static str { format!("deepseek2.{s}").leak() };
    let mut meta = vec![
        ("general.architecture", Val::Str("deepseek2".into())),
        (k("block_count"), Val::U32(f.nl as u32)),
        (k("context_length"), Val::U32(if o.yarn { 256 } else { 64 })),
        (k("embedding_length"), Val::U32(f.dim as u32)),
        (k("feed_forward_length"), Val::U32(f.ffn as u32)),
        (k("attention.head_count"), Val::U32(f.nh as u32)),
        (k("attention.head_count_kv"), Val::U32(1)),
        (k("rope.freq_base"), Val::F32(10000.0)),
        (k("attention.layer_norm_rms_epsilon"), Val::F32(0.05)),
        (k("expert_used_count"), Val::U32(f.k_exp as u32)),
        (k("leading_dense_block_count"), Val::U32(1)),
        (k("vocab_size"), Val::U32(f.vocab as u32)),
        (k("attention.kv_lora_rank"), Val::U32(f.lat as u32)),
        (k("attention.key_length"), Val::U32(kr as u32)),
        (k("attention.value_length"), Val::U32(f.lat as u32)),
        (k("attention.key_length_mla"), Val::U32(hk as u32)),
        (k("attention.value_length_mla"), Val::U32(f.hv as u32)),
        (k("expert_feed_forward_length"), Val::U32(f.ffe as u32)),
        (k("expert_count"), Val::U32(f.n_exp as u32)),
        (k("expert_shared_count"), Val::U32(1)),
        (k("expert_weights_scale"), Val::F32(2.446)),
        (k("expert_weights_norm"), Val::Bool(true)),
        (k("expert_gating_func"), Val::U32(2)),
        (k("rope.dimension_count"), Val::U32(f.rope as u32)),
        ("tokenizer.ggml.model", Val::Str("gpt2".into())),
        ("tokenizer.ggml.pre", Val::Str("kimi-k2".into())),
        (
            "tokenizer.ggml.tokens",
            Val::Arr(vocab_texts(f.vocab).into_iter().map(Val::Str).collect()),
        ),
        (
            "tokenizer.ggml.token_type",
            Val::Arr((0..f.vocab).map(|_| Val::I32(1)).collect()),
        ),
        ("tokenizer.ggml.bos_token_id", Val::U32(1)),
        ("tokenizer.ggml.eos_token_id", Val::U32(2)),
    ];
    if o.q_lora {
        meta.push((k("attention.q_lora_rank"), Val::U32(f.q_lora as u32)));
    }
    if o.yarn {
        meta.push((k("rope.scaling.type"), Val::Str("yarn".into())));
        meta.push((k("rope.scaling.factor"), Val::F32(4.0)));
        meta.push((k("rope.scaling.original_context_length"), Val::U32(64)));
        meta.push((k("rope.scaling.yarn_log_multiplier"), Val::F32(0.1)));
    }
    (meta, ts)
}

/// Q8_0 bytes of `v` (32-value blocks: f16 scale amax/127, rounded values).
fn q8_0(v: &[f32]) -> Vec<u8> {
    v.chunks(32)
        .flat_map(|b| {
            let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = crate::quant::f16_to_f32(crate::quant::f32_to_f16(amax / 127.0));
            let mut out = crate::quant::f32_to_f16(d).to_le_bytes().to_vec();
            out.extend(b.iter().map(|&v| {
                (if d == 0.0 { 0.0 } else { (v / d).round() }).clamp(-127.0, 127.0) as i8 as u8
            }));
            out
        })
        .collect()
}

/// The fixture's tensor list and bytes (matrices of type `mt`: F32 or
/// Q8_0; the router, norms and biases F32).
#[allow(clippy::type_complexity)]
pub(crate) fn fixture_tensors(
    o: Opt,
    mt: u32,
) -> (
    Vec<(&'static str, Val)>,
    Vec<(String, Vec<u64>, u32)>,
    Vec<Vec<u8>>,
) {
    let (meta, ts) = fixture_parts(o);
    let ty = |t: &(String, Vec<u64>, Vec<f32>)| {
        if t.1.len() >= 2 && !t.0.ends_with("ffn_gate_inp.weight") {
            mt
        } else {
            F32
        }
    };
    let list = ts
        .iter()
        .map(|t| (t.0.clone(), t.1.clone(), ty(t)))
        .collect();
    let data = ts
        .iter()
        .map(|t| match ty(t) {
            F32 => t.2.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Q8_0 => q8_0(&t.2),
            other => unimplemented!("ds2 fixture: type {other}"),
        })
        .collect();
    (meta, list, data)
}

/// The fixture as a model, and its tensors as written (the rope rows in the
/// file's adjacent-pair order: `deepseek2_ref` reads those).
pub(crate) fn ds2_fixture(o: Opt, mt: u32) -> (Model, RawTensors) {
    let (meta, list, data) = fixture_tensors(o, mt);
    let l: Vec<(&str, &[u64], u32)> = list.iter().map(|t| (t.0.as_str(), &t.1[..], t.2)).collect();
    let m = crate::gguf::build_for_test_with(&meta, &l, &data);
    let raw = list
        .into_iter()
        .zip(data)
        .map(|((n, d, t), b)| (n, d, t, b))
        .collect();
    (m, raw)
}

/// Deliberate mistakes `deepseek2_ref` can make, to show its tolerance
/// tells them apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Bug {
    None,
    /// RoPE on the first 16 query/key dims of each head (nope's) instead of
    /// the last.
    RopeWrongPart,
    /// 1/sqrt(576-style key width) instead of mscale²/sqrt(nope + rope).
    ScaleOfKeyRow,
    /// exp_probs_b added into the weights too, not just the selection.
    BiasInWeights,
    /// The weight scale before the renorm (which then cancels it).
    ScaleBeforeRenorm,
    /// kv_a_norm over the whole kv_a output (rope part too).
    NormOverRow,
    /// No YaRN frequency interpolation (mscale kept).
    NoYarn,
    /// YaRN's mscale on the (sin, cos) table (Axon's qwen rule) instead of
    /// in the attention scale: kq = 1/sqrt(nope + rope).
    MscaleOnTable,
    /// YaRN's frequencies, but no mscale anywhere: kq = 1/sqrt(nope + rope).
    NoMscale,
}

/// The fixture's logits after each of `ids`, in f64, the *expanded* MLA
/// form (HF modeling_deepseek's): per head k = [W_kb^T kv | k_pe] and
/// v = W_vb kv materialized, q·k over nope + rope dims, softmax with
/// mscale²/sqrt(nope + rope), the adjacent-pair RoPE on the raw rows, YaRN
/// as HF's DeepseekV3YarnRotaryEmbedding. FFN: dense lead layer, then
/// sigmoid router + exp_probs_b selection, renorm, x2.446, plus the shared
/// expert.
pub(crate) fn deepseek2_ref(m: &Model, raw: &RawTensors, ids: &[u32], bug: Bug) -> Vec<Vec<f32>> {
    deepseek2_ref_routes(m, raw, ids, bug, None).0
}

/// Routes per position, per MoE layer (rank order).
type Routes = Vec<Vec<Vec<usize>>>;

/// `deepseek2_ref`, and the experts each MoE layer would pick at each
/// position (routes[pos][moe layer], rank order) with the selection margin
/// there (the k-th selection score minus the (k+1)-th). With `forced`, each
/// MoE layer runs the forced experts instead (their unbiased probabilities,
/// renorm, scale: the same weight rule), so a run's routing discontinuities
/// (a near-tied router flipping an expert, and everything downstream of
/// it, later positions' cache rows included) are taken out of a comparison.
#[allow(clippy::type_complexity)]
fn deepseek2_ref_routes(
    m: &Model,
    raw: &RawTensors,
    ids: &[u32],
    bug: Bug,
    forced: Option<&Routes>,
) -> (Vec<Vec<f32>>, Vec<Vec<(Vec<usize>, f64)>>) {
    let f = FX;
    let num = |k: &str| meta_num(m, &format!("deepseek2.{k}"));
    let eps = num("attention.layer_norm_rms_epsilon").unwrap();
    let base = num("rope.freq_base").unwrap();
    let hk = f.nope + f.rope;
    let w = |name: &str| -> (Vec<f64>, usize) {
        let t = raw
            .iter()
            .find(|t| t.0 == name)
            .unwrap_or_else(|| panic!("{name}"));
        let mut v = vec![0f32; t.1.iter().product::<u64>() as usize];
        dequant(t.2, &t.3, &mut v);
        (v.into_iter().map(f64::from).collect(), t.1[0] as usize)
    };
    let has = |name: &str| raw.iter().any(|t| t.0 == name);
    let mv = |(wv, cols): &(Vec<f64>, usize), x: &[f64]| -> Vec<f64> {
        wv.chunks(*cols)
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    };
    let rms = |x: &[f64], g: &[f64]| -> Vec<f64> {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
        x.iter().zip(g).map(|(v, g)| v * inv * g).collect()
    };
    let silu = |x: f64| x / (1.0 + (-x).exp());
    // YaRN (HF): inv_freq = interp (1 - mask) + extra mask, mask = 1 -
    // ramp(low, high); softmax scale times mscale² (mscale_all_dim 1).
    let yarn = meta_num(m, "deepseek2.rope.scaling.factor");
    let (factor, orig) = (
        yarn.unwrap_or(1.0),
        num("rope.scaling.original_context_length").unwrap_or(0.0),
    );
    let inv_freq = |i: usize| -> f64 {
        let extra = base.powf(-(2.0 * i as f64) / f.rope as f64);
        if yarn.is_none() || bug == Bug::NoYarn {
            return extra;
        }
        let corr = |rot: f64| {
            (f.rope as f64 * (orig / (rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln())
        };
        let low = corr(32.0).floor().max(0.0);
        let mut high = corr(1.0).ceil().min(f.rope as f64 - 1.0);
        if high == low {
            high += 0.001;
        }
        let mask = 1.0 - ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
        extra / factor * (1.0 - mask) + extra * mask
    };
    let mscale = if yarn.is_some() {
        0.1 * (num("rope.scaling.yarn_log_multiplier").unwrap() / 0.1) * factor.ln() + 1.0
    } else {
        1.0
    };
    let scale = match bug {
        Bug::ScaleOfKeyRow => 1.0 / ((f.lat + f.rope) as f64).sqrt(),
        Bug::MscaleOnTable | Bug::NoMscale => 1.0 / (hk as f64).sqrt(),
        _ => mscale * mscale / (hk as f64).sqrt(),
    };
    // Adjacent pairs (2i, 2i + 1) turn by pos * inv_freq(i).
    let table_m = if bug == Bug::MscaleOnTable {
        mscale
    } else {
        1.0
    };
    let rope = |v: &mut [f64], pos: usize| {
        for i in 0..v.len() / 2 {
            let (s, c) = (pos as f64 * inv_freq(i)).sin_cos();
            let (s, c) = (s * table_m, c * table_m);
            let (a, b) = (v[2 * i], v[2 * i + 1]);
            v[2 * i] = a * c - b * s;
            v[2 * i + 1] = a * s + b * c;
        }
    };
    let embd = w("token_embd.weight");
    let out_w = w("output.weight");
    let out_norm = w("output_norm.weight").0;
    // Per layer, per position: expanded keys [head][hk] and values [head][hv].
    type Heads = Vec<Vec<f64>>;
    let mut keys: Vec<Vec<Heads>> = vec![Vec::new(); f.nl];
    let mut vals: Vec<Vec<Heads>> = vec![Vec::new(); f.nl];
    let mut out = Vec::new();
    let mut routes: Vec<Vec<(Vec<usize>, f64)>> = Vec::new();
    for (pos, &tok) in ids.iter().enumerate() {
        routes.push(Vec::new());
        let mut x: Vec<f64> = embd.0[tok as usize * f.dim..(tok as usize + 1) * f.dim].to_vec();
        for l in 0..f.nl {
            let n = |s: &str| format!("blk.{l}.{s}");
            let h = rms(&x, &w(&n("attn_norm.weight")).0);
            let q = if has(&n("attn_q_a.weight")) {
                let qa = rms(
                    &mv(&w(&n("attn_q_a.weight")), &h),
                    &w(&n("attn_q_a_norm.weight")).0,
                );
                mv(&w(&n("attn_q_b.weight")), &qa)
            } else {
                mv(&w(&n("attn_q.weight")), &h)
            };
            let c = mv(&w(&n("attn_kv_a_mqa.weight")), &h);
            let kv = match bug {
                Bug::NormOverRow => {
                    let mut g = w(&n("attn_kv_a_norm.weight")).0;
                    g.resize(f.lat + f.rope, 1.0);
                    rms(&c, &g)[..f.lat].to_vec()
                }
                _ => rms(&c[..f.lat], &w(&n("attn_kv_a_norm.weight")).0),
            };
            let mut kpe = c[f.lat..].to_vec();
            let (r0, r1) = match bug {
                Bug::RopeWrongPart => (0, f.rope),
                _ => (f.nope, hk),
            };
            // (RopeWrongPart turns the expanded key's first dims below.)
            if bug != Bug::RopeWrongPart {
                rope(&mut kpe, pos);
            }
            let (kb, _) = w(&n("attn_k_b.weight"));
            let (vb, _) = w(&n("attn_v_b.weight"));
            let mut kh = Vec::new();
            let mut vh = Vec::new();
            for hh in 0..f.nh {
                // k_nope[j] = sum_r Wkb[h][r][j] kv[r]; v[i] = sum_r Wvb[h][i][r] kv[r].
                let mut k = vec![0f64; hk];
                for r in 0..f.lat {
                    for j in 0..f.nope {
                        k[j] += kb[(hh * f.lat + r) * f.nope + j] * kv[r];
                    }
                }
                k[f.nope..].copy_from_slice(&kpe);
                if bug == Bug::RopeWrongPart {
                    rope(&mut k[r0..r1], pos);
                }
                let v: Vec<f64> = (0..f.hv)
                    .map(|i| {
                        (0..f.lat)
                            .map(|r| vb[(hh * f.hv + i) * f.lat + r] * kv[r])
                            .sum()
                    })
                    .collect();
                kh.push(k);
                vh.push(v);
            }
            keys[l].push(kh);
            vals[l].push(vh);
            let mut att = vec![0f64; f.nh * f.hv];
            for hh in 0..f.nh {
                let mut qh = q[hh * hk..(hh + 1) * hk].to_vec();
                rope(&mut qh[r0..r1], pos);
                let s: Vec<f64> = (0..=pos)
                    .map(|t| {
                        qh.iter()
                            .zip(&keys[l][t][hh])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for t in 0..=pos {
                    for i in 0..f.hv {
                        att[hh * f.hv + i] += e[t] / z * vals[l][t][hh][i];
                    }
                }
            }
            let o = mv(&w(&n("attn_output.weight")), &att);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
            let h = rms(&x, &w(&n("ffn_norm.weight")).0);
            let ffn = |g: &(Vec<f64>, usize), u: &(Vec<f64>, usize), d: &(Vec<f64>, usize)| {
                let a: Vec<f64> = mv(g, &h)
                    .iter()
                    .zip(mv(u, &h))
                    .map(|(g, u)| silu(*g) * u)
                    .collect();
                mv(d, &a)
            };
            let y = if l == 0 {
                ffn(
                    &w(&n("ffn_gate.weight")),
                    &w(&n("ffn_up.weight")),
                    &w(&n("ffn_down.weight")),
                )
            } else {
                let p: Vec<f64> = mv(&w(&n("ffn_gate_inp.weight")), &h)
                    .iter()
                    .map(|v| 1.0 / (1.0 + (-v).exp()))
                    .collect();
                let b = w(&n("exp_probs_b.bias")).0;
                let sel: Vec<f64> = p.iter().zip(&b).map(|(p, b)| p + b).collect();
                let mut order: Vec<usize> = (0..f.n_exp).collect();
                order.sort_by(|&a, &c| sel[c].total_cmp(&sel[a]).then(a.cmp(&c)));
                let mine = &routes.last().unwrap();
                let picks: Vec<usize> = match forced {
                    Some(fr) => fr[pos][mine.len()].clone(),
                    None => order[..f.k_exp].to_vec(),
                };
                routes.last_mut().unwrap().push((
                    order[..f.k_exp].to_vec(),
                    sel[order[f.k_exp - 1]] - sel[order[f.k_exp]],
                ));
                let mut ws: Vec<f64> = picks
                    .iter()
                    .map(|&e| {
                        if bug == Bug::BiasInWeights {
                            sel[e]
                        } else {
                            p[e]
                        }
                    })
                    .collect();
                if bug == Bug::ScaleBeforeRenorm {
                    ws.iter_mut().for_each(|v| *v *= 2.446);
                }
                let sum: f64 = ws.iter().sum();
                ws.iter_mut().for_each(|v| *v /= sum);
                if bug != Bug::ScaleBeforeRenorm {
                    ws.iter_mut().for_each(|v| *v *= 2.446);
                }
                let slice = |s: &str, e: usize| {
                    let (v, cols) = w(&n(s));
                    let per = v.len() / f.n_exp;
                    (v[e * per..(e + 1) * per].to_vec(), cols)
                };
                let mut y = ffn(
                    &w(&n("ffn_gate_shexp.weight")),
                    &w(&n("ffn_up_shexp.weight")),
                    &w(&n("ffn_down_shexp.weight")),
                );
                for (&e, &wt) in picks.iter().zip(&ws) {
                    let d = ffn(
                        &slice("ffn_gate_exps.weight", e),
                        &slice("ffn_up_exps.weight", e),
                        &slice("ffn_down_exps.weight", e),
                    );
                    y.iter_mut().zip(d).for_each(|(a, b)| *a += wt * b);
                }
                y
            };
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
        }
        let h = rms(&x, &out_norm);
        out.push(mv(&out_w, &h).into_iter().map(|v| v as f32).collect());
    }
    (out, routes)
}

const IDS: [u32; 14] = [3, 17, 42, 9, 5, 61, 33, 2, 28, 50, 11, 7, 44, 19];
/// F32 fixture: CPU (exact float rows) vs the f64 reference, with an f32
/// cache (MEASURED ~1e-5) and with the default f16 one (~5e-3).
const TOL: f32 = 2e-4;
const TOL_F16: f32 = 1.5e-2;

fn all_opts() -> [Opt; 4] {
    let o = |q_lora, yarn| Opt { q_lora, yarn };
    [
        o(false, false),
        o(false, true),
        o(true, false),
        o(true, true),
    ]
}

/// The absorbed CPU forward (decode, and prefill then one step) agrees with
/// the expanded reference, with and without q_lora and YaRN; every
/// deliberate bug moves the reference far past the tolerance.
#[test]
fn deepseek2_matches_expanded_reference() {
    let n = IDS.len();
    for o in all_opts() {
        let (m, raw) = ds2_fixture(o, F32);
        let want = deepseek2_ref(&m, &raw, &IDS, Bug::None);
        for bug in [
            Bug::RopeWrongPart,
            Bug::ScaleOfKeyRow,
            Bug::BiasInWeights,
            Bug::ScaleBeforeRenorm,
            Bug::NormOverRow,
            Bug::NoYarn,
            Bug::MscaleOnTable,
            Bug::NoMscale,
        ] {
            let yarn_only = matches!(bug, Bug::NoYarn | Bug::MscaleOnTable | Bug::NoMscale);
            if yarn_only && !o.yarn {
                continue;
            }
            let d = max_diff(&deepseek2_ref(&m, &raw, &IDS, bug), &want);
            eprintln!("{o:?}: reference with {bug:?}: max |dlogit| {d:.4}");
            assert!(
                d > 30.0 * TOL_F16,
                "{o:?}: {bug:?} hardly changes the logits ({d})"
            );
        }
        for (repack, f32) in [(true, true), (false, true), (true, false)] {
            let (m, _) = ds2_fixture(o, F32);
            let net = Qwen2::new_with(&m, repack);
            let cache = || super::deepseek2::new_cache_of(net.n_layers, net.head_dim, n, f32);
            let mut c = cache();
            let dec: Vec<Vec<f32>> = (0..n).map(|p| net.forward(IDS[p], p, &mut c)).collect();
            let mut c = cache();
            net.feed_batch(&IDS[..n - 1], 0, &mut c);
            let pre = net.forward(IDS[n - 1], n - 1, &mut c);
            let (d, dp) = (max_diff(&dec, &want), max_diff(&[pre], &want[n - 1..]));
            // The f32 cache: the math alone; f16: plus the cache's rounding
            // of the latent (which is V too).
            let tol = if f32 { TOL } else { TOL_F16 };
            eprintln!(
                "{o:?} cpu (repack {repack}, f32 cache {f32}) vs reference: {d:.2e}, after prefill {dp:.2e}"
            );
            assert!(d <= tol && dp <= tol, "{o:?}: cpu vs reference {d} / {dp}");
        }
        q8_smoke(o, n);
    }
}

/// The Q8_0 (r8q) fixture against the reference on its dequantized weights:
/// a smoke check of the quantized path (exactness is the F32 rows' gate).
///
/// Routing first. A router near-tie (selection margin of a few 1e-2) flips
/// an expert under 8-bit activations, and that changes the FFN output, the
/// next layers, and every later position through the cache: MEASURED, the
/// q_lora/no-yarn max |dlogit| 1.45 is exactly that (position 6, MoE layer
/// 1: experts [0,3] vs the reference's [0,1] at a margin of 0.009; then
/// position 9 at 0.005 and 11 at 0.045), not a q_lora r8q error: on the
/// routes the run took, the same config is at 0.32. So the logits are compared
/// with the reference *forced onto this run's routes*, which isolates the
/// quantization noise, and the routes are gated on their own: wherever this
/// run's expert set differs from what the forced reference itself would
/// pick, the reference's margin must be under `ROUTE_MARGIN`.
///
/// Logits: max |dlogit| <= `Q8_BOUND` (2x this fixture's MEASURED maximum,
/// below), every logit finite, and a greedy pick may flip only where the
/// reference's top-1/top-2 gap is under `Q8_BOUND` (a perturbation of size
/// d can't flip a pick whose gap is over 2d). Any number of such flips is
/// allowed: the earlier "at most one" was a count over 14 positions, not a
/// property of the path. The gate must be able to fire: at least `Q8_GATED`
/// positions have a gap at or above `Q8_BOUND` (asserted).
fn q8_smoke(o: Opt, n: usize) {
    let (m8, raw8) = ds2_fixture(o, Q8_0);
    let net = Qwen2::new(&m8);
    let mut c = net.new_cache(n);
    let mut got: Routes = Vec::new();
    let l: Vec<Vec<f32>> = (0..n)
        .map(|p| {
            moe::ROUTES.with_borrow_mut(|r| r.clear());
            let v = net.forward(IDS[p], p, &mut c);
            got.push(moe::ROUTES.with_borrow(|r| r.clone()));
            v
        })
        .collect();
    assert!(
        l.iter().flatten().all(|v| v.is_finite()),
        "{o:?} Q8_0: non-finite logits"
    );
    let (free, _) = deepseek2_ref_routes(&m8, &raw8, &IDS, Bug::None, None);
    let (want8, own) = deepseek2_ref_routes(&m8, &raw8, &IDS, Bug::None, Some(&got));
    let d = max_diff(&l, &want8);
    let d_free = max_diff(&l, &free);
    let top = |v: &[f32]| (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    // The reference's top-1 minus top-2 logit at each position.
    let gap = |v: &[f32]| {
        let t = top(v);
        v[t] - (0..v.len())
            .filter(|&i| i != t)
            .map(|i| v[i])
            .fold(f32::MIN, f32::max)
    };
    let gaps: Vec<f32> = want8.iter().map(|v| gap(v)).collect();
    let flips: Vec<f32> = l
        .iter()
        .zip(&want8)
        .filter(|(a, b)| top(a) != top(b))
        .map(|(_, b)| gap(b))
        .collect();
    let gated = gaps.iter().filter(|&&g| g >= Q8_BOUND).count();
    let set = |v: &[usize]| {
        let mut v = v.to_vec();
        v.sort_unstable();
        v
    };
    let mut route_flips = Vec::new();
    for p in 0..n {
        for (layer, (ours, (theirs, margin))) in got[p].iter().zip(&own[p]).enumerate() {
            if set(ours) != set(theirs) {
                route_flips.push((p, layer, *margin));
            }
        }
    }
    eprintln!(
        "{o:?} Q8_0 fixture (r8q): max |dlogit| {d:.4} vs the reference on this run's routes \
         ({d_free:.4} on its own), bound {Q8_BOUND}; top-1 flips at gaps {flips:?}; \
         {gated}/{n} positions gated; expert flips (pos, moe layer, ref margin) {route_flips:?}; \
         reference gaps {gaps:.3?}"
    );
    assert!(
        d <= Q8_BOUND,
        "{o:?} Q8_0: max |dlogit| {d} over {Q8_BOUND}"
    );
    assert!(
        flips.iter().all(|&g| g < Q8_BOUND),
        "{o:?} Q8_0: top-1 flips at reference gaps {flips:?} (bound {Q8_BOUND})"
    );
    assert!(
        gated >= Q8_GATED,
        "{o:?} Q8_0: only {gated} positions have a reference gap >= {Q8_BOUND}: the flip gate can't fire"
    );
    assert!(
        route_flips.iter().all(|&(_, _, mg)| mg < ROUTE_MARGIN),
        "{o:?} Q8_0: an expert flipped at a clear router margin: {route_flips:?}"
    );
}

/// `q8_smoke`'s logit bound on the route-forced comparison. MEASURED max
/// |dlogit| (1018-053935-7964): 0.33 / 0.18 without q_lora (no yarn /
/// yarn), 0.32 / 0.42 with (q_lora/no-yarn was 1.45 against the unforced
/// reference: the expert flips at positions 6, 9, 11). 0.7 is 1.65-3.9x
/// those; the margin can't grow to 2x the largest (0.85) without leaving
/// the q_lora/yarn config's flip gate fewer than `Q8_GATED` positions with
/// a gap above it (its gaps: one over 0.9, five over 0.7).
const Q8_BOUND: f32 = 0.7;

/// Largest reference router margin (k-th minus (k+1)-th selection score)
/// at which an expert may flip under 8-bit activations: 2x the MEASURED
/// largest flip margin (0.045; the flips: 0.009, 0.005, 0.045, all
/// q_lora/no-yarn, 1018-053935-7964), rounded up. The fixture's margins run
/// to 0.9, so the gate can fire.
const ROUTE_MARGIN: f64 = 0.1;

/// Positions (of IDS' 14) whose reference gap must reach `Q8_BOUND` for
/// the flip gate to mean anything.
const Q8_GATED: usize = 4;

/// decode == feed_batch == verify, bit for bit: the cache after a batched
/// prefill equals the one after feeding token by token, the logits after
/// it equal decode's, and verify_batch's rows equal forward's. Batches
/// split at every size, f16 and (AXON_KV aside) the fixture's types.
#[test]
fn deepseek2_batch_matches_decode_bits() {
    let n = IDS.len();
    for o in all_opts() {
        for mt in [F32, Q8_0] {
            let (m, _) = ds2_fixture(o, mt);
            let net = Qwen2::new(&m);
            let mut a = net.new_cache(n + 1);
            let dec: Vec<Vec<f32>> = (0..n).map(|p| net.forward(IDS[p], p, &mut a)).collect();
            for split in [1, 3, n - 1, n] {
                let mut b = net.new_cache(n + 1);
                let mut at = 0;
                while at < n {
                    let e = (at + split).min(n);
                    net.feed_batch(&IDS[at..e], at, &mut b);
                    at = e;
                }
                assert_eq!(
                    a.bits(),
                    b.bits(),
                    "{o:?} type {mt}: cache after batches of {split}"
                );
                let mut v = net.new_cache(n + 1);
                net.feed_batch(&IDS[..2], 0, &mut v);
                let rows = net.verify_batch(&IDS[2..], 2, &mut v);
                for (r, d) in rows.iter().zip(&dec[2..]) {
                    assert_eq!(
                        tests::bits(r),
                        tests::bits(d),
                        "{o:?} type {mt}: verify row"
                    );
                }
                let picks = net.verify_argmax(&IDS[2..], 2, &mut v);
                for (p, d) in picks.iter().zip(&dec[2..]) {
                    let best = (0..d.len()).fold(0, |b, i| if d[i] >= d[b] { i } else { b });
                    assert_eq!(*p as usize, best, "{o:?}: verify_argmax");
                }
            }
            // Cache reuse: a prefix fed, then the rest, == all at once.
            let mut c = net.new_cache(n + 1);
            net.feed_batch(&IDS[..5], 0, &mut c);
            net.feed_batch(&IDS[5..n - 1], 5, &mut c);
            let l = net.forward(IDS[n - 1], n - 1, &mut c);
            assert_eq!(
                tests::bits(&l),
                tests::bits(&dec[n - 1]),
                "{o:?}: cache reuse"
            );
        }
    }
}

/// Long batches (the matmuls' token-split path, nb >= 16; several
/// attention tiles): feed_batch == feed bit for bit, every logit finite.
#[test]
fn deepseek2_long_batch_matches_decode_bits() {
    let ids: Vec<u32> = (0..150u32).map(|i| (i * 37 + 11) % 64).collect();
    let n = ids.len();
    for mt in [F32, Q8_0] {
        let (m, _) = ds2_fixture(
            Opt {
                q_lora: false,
                yarn: true,
            },
            mt,
        );
        let net = Qwen2::new(&m);
        let mut a = net.new_cache(n + 1);
        let dec: Vec<Vec<f32>> = (0..n).map(|p| net.forward(ids[p], p, &mut a)).collect();
        assert!(
            dec.iter().flatten().all(|v| v.is_finite()),
            "type {mt}: decode"
        );
        let mut b = net.new_cache(n + 1);
        net.feed_batch(&ids[..n - 1], 0, &mut b);
        let l = net.forward(ids[n - 1], n - 1, &mut b);
        assert!(
            l.iter().all(|v| v.is_finite()),
            "type {mt}: after feed_batch"
        );
        assert_eq!(
            a.bits(),
            b.bits(),
            "type {mt}: cache after one batch of {}",
            n - 1
        );
        assert_eq!(
            tests::bits(&l),
            tests::bits(&dec[n - 1]),
            "type {mt}: logits"
        );
    }
}

/// The cache is K only: one lat + rope row per layer and position, no V.
#[test]
fn deepseek2_fixture_vocab_is_whole() {
    assert_eq!(vocab_texts(FX.vocab).len(), FX.vocab);
}

#[test]
fn deepseek2_cache_is_k_only() {
    let (m, _) = ds2_fixture(
        Opt {
            q_lora: false,
            yarn: false,
        },
        Q8_0,
    );
    let net = Qwen2::new(&m);
    let c = net.new_cache(10);
    // No session image yet (kvsave): a saved chat is prefilled again.
    assert!(crate::kvsave::KvIo::kv_ident(&net, &c).is_none());
    let arrays = c.arrays();
    assert_eq!(arrays.len(), 2 * FX.nl);
    let row = (FX.lat + FX.rope) * c.elem().1;
    for (i, a) in arrays.iter().enumerate() {
        assert_eq!(a.len(), if i % 2 == 0 { 10 * row } else { 0 }, "array {i}");
    }
}

fn route_with(gate: moe::Gate, logits_scale: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let n = logits_scale.len();
    let h = [1.0f32, 0.0, 0.0, 0.0];
    let router: Vec<f32> = logits_scale
        .iter()
        .flat_map(|&s| [s, 0.0, 0.0, 0.0])
        .collect();
    let dims = moe::MoeDims {
        n_expert: n,
        k,
        ffn: 8,
    };
    let r = moe::Moe::new(dims, router, Vec::new(), Vec::new(), Vec::new(), gate);
    let (mut ids, mut w) = (vec![0; k], vec![0.0; k]);
    r.route(&h, &mut ids, &mut w);
    (ids, w)
}

/// Sigmoid routing: the bias picks, the weights stay the sigmoid's, the
/// renorm divides by their sum, the scale multiplies after it; ties go to
/// the lower expert.
#[test]
fn deepseek2_router_rule() {
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let logits = [0.5f32, 1.0, -0.2, 1.0];
    let g = |bias: Option<Vec<f32>>, norm, scale| moe::Gate {
        sigmoid: true,
        bias,
        norm,
        scale,
    };
    let (ids, w) = route_with(g(None, false, 1.0), &logits, 2);
    assert_eq!(ids, [1, 3], "tie to the lower index");
    assert!((w[0] - sig(1.0)).abs() < 1e-5);
    // The bias moves expert 2 to the top; its weight stays unbiased.
    let (ids, w) = route_with(g(Some(vec![0.0, 0.0, 2.0, 0.0]), false, 1.0), &logits, 2);
    assert_eq!(ids, [2, 1]);
    assert!((w[0] - sig(-0.2)).abs() < 1e-5, "{w:?}");
    // Renorm, then the scale.
    let (_, w) = route_with(g(None, true, 2.5), &logits, 2);
    assert!((w[0] + w[1] - 2.5).abs() < 1e-5, "{w:?}");
    assert_eq!(w[0].to_bits(), w[1].to_bits());
}

/// The file checks: the legacy layout and what isn't implemented are
/// refused with a reason; the fixture passes.
#[test]
fn deepseek2_check_supported() {
    let o = Opt {
        q_lora: true,
        yarn: true,
    };
    let (meta, list, _) = fixture_tensors(o, Q8_0);
    // Zero tensors: the checks read metadata and the tensor table only.
    let build = |meta: &[(&'static str, Val)], list: &[(String, Vec<u64>, u32)]| {
        let l: Vec<(&str, &[u64], u32)> =
            list.iter().map(|t| (t.0.as_str(), &t.1[..], t.2)).collect();
        crate::gguf::build_for_test(meta, &l)
    };
    check_supported(&build(&meta, &list)).unwrap();
    let refuse =
        |meta: Vec<(&'static str, Val)>, list: Vec<(String, Vec<u64>, u32)>, want: &str| {
            let e = check_supported(&build(&meta, &list))
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{want:?} not in {e:?}");
        };
    let set = |k: &'static str, v: Val| {
        let mut m = meta.clone();
        match m.iter_mut().find(|(n, _)| *n == k) {
            Some(e) => e.1 = v,
            None => m.push((k, v)),
        }
        m
    };
    refuse(
        set("deepseek2.attention.head_count_kv", Val::U32(2)),
        list.clone(),
        "re-convert",
    );
    let no_mla: Vec<_> = meta
        .iter()
        .filter(|(k, _)| *k != "deepseek2.attention.key_length_mla")
        .cloned()
        .collect();
    refuse(no_mla, list.clone(), "re-convert");
    let mut legacy = list.clone();
    legacy.push(("blk.1.attn_kv_b.weight".into(), vec![32, 128], Q8_0));
    refuse(meta.clone(), legacy, "re-convert");
    refuse(
        set("deepseek2.expert_group_count", Val::U32(8)),
        list.clone(),
        "grouped",
    );
    refuse(
        set("deepseek2.expert_gating_func", Val::U32(3)),
        list.clone(),
        "gating",
    );
    refuse(
        set("deepseek2.nextn_predict_layers", Val::U32(1)),
        list.clone(),
        "MTP",
    );
    refuse(
        set("deepseek2.attention.key_length", Val::U32(64)),
        list.clone(),
        "absorbed",
    );
    let mut extra = list.clone();
    extra.push(("blk.2.attn_q.bias".into(), vec![96], F32));
    refuse(meta.clone(), extra, "not supported on deepseek2");
    // An i-quant the kernels don't run: the shared refusal (model.rs unsupported_type).
    let mut iq = list.clone();
    iq.iter_mut()
        .find(|t| t.0 == "blk.2.attn_v_b.weight")
        .unwrap()
        .2 = 22;
    refuse(
        meta.clone(),
        iq,
        "tensor blk.2.attn_v_b.weight: type IQ2_S (22) is not supported (supported: F32",
    );
    let mut short = list.clone();
    short.retain(|t| t.0 != "blk.2.attn_v_b.weight");
    refuse(meta.clone(), short, "missing tensor blk.2.attn_v_b.weight");
}

/// The fixture as a GGUF file at $AXON_DS2_FIXTURE_OUT (for the llama.cpp
/// oracle: fxoracle.sh); q_lora and YaRN on (llama.cpp wants q_lora_rank on
/// a 3-layer deepseek2), F32 matrices or (AXON_DS2_FIXTURE_TYPE=q8_0) Q8_0.
#[test]
#[ignore = "writes a file; run with AXON_DS2_FIXTURE_OUT=path"]
fn deepseek2_write_fixture() {
    let path = std::env::var("AXON_DS2_FIXTURE_OUT").expect("AXON_DS2_FIXTURE_OUT");
    // AXON_DS2_FIXTURE_TYPE=q8_0: Q8_0 matrices (the r8q path), else F32.
    let mt = match std::env::var("AXON_DS2_FIXTURE_TYPE").as_deref() {
        Ok("q8_0") => Q8_0,
        _ => F32,
    };
    let o = Opt {
        q_lora: true,
        yarn: true,
    };
    let (meta, list, data) = fixture_tensors(o, mt);
    let l: Vec<(&str, &[u64], u32)> = list.iter().map(|t| (t.0.as_str(), &t.1[..], t.2)).collect();
    std::fs::write(&path, crate::gguf::gguf_bytes_for_test(&meta, &l, &data)).unwrap();
    let (m, raw) = ds2_fixture(o, mt);
    let want = deepseek2_ref(&m, &raw, &IDS, Bug::None);
    let p: Vec<String> = IDS.iter().map(|i| i.to_string()).collect();
    eprintln!(
        "wrote {path} (type {mt}); ids {}; reference logits[last][..4] {:?}",
        p.join(","),
        &want[IDS.len() - 1][..4]
    );
}

/// Moonlight-16B-A3B (if present): decode == feed_batch == verify on a
/// real prompt, and a chat conversation's cache reuse == a fresh prefill.
#[test]
fn deepseek2_moonlight_bits() {
    let Some(m) = crate::testutil::ds2_model() else {
        return;
    };
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let ids = tok
        .encode("The capital of France is Paris. The capital of Germany is Berlin. The capital of");
    let n = ids.len();
    let net = Qwen2::new(&m);
    let mut a = net.new_cache(n + 8);
    let dec: Vec<Vec<f32>> = (0..n).map(|p| net.forward(ids[p], p, &mut a)).collect();
    let mut b = net.new_cache(n + 8);
    net.feed_batch(&ids[..n - 1], 0, &mut b);
    let l = net.forward(ids[n - 1], n - 1, &mut b);
    assert_eq!(a.bits(), b.bits(), "moonlight: cache after feed_batch");
    assert_eq!(
        tests::bits(&l),
        tests::bits(&dec[n - 1]),
        "moonlight: logits after feed_batch"
    );
    let mut v = net.new_cache(n + 8);
    net.feed_batch(&ids[..4], 0, &mut v);
    for (r, d) in net.verify_batch(&ids[4..], 4, &mut v).iter().zip(&dec[4..]) {
        assert_eq!(tests::bits(r), tests::bits(d), "moonlight: verify row");
    }
    let best = |d: &[f32]| (0..d.len()).fold(0, |b, i| if d[i] >= d[b] { i } else { b });
    // A longer prompt in one batch (the chat check below feeds ~70).
    let long = tok.encode(&"The quick brown fox jumps over the lazy dog. ".repeat(10));
    let nl = long.len();
    let mut c1 = net.new_cache(nl + 1);
    let d1: Vec<Vec<f32>> = (0..nl).map(|p| net.forward(long[p], p, &mut c1)).collect();
    let mut c2 = net.new_cache(nl + 1);
    net.feed_batch(&long[..nl - 1], 0, &mut c2);
    let l2 = net.forward(long[nl - 1], nl - 1, &mut c2);
    let bad = d1.iter().flatten().filter(|v| !v.is_finite()).count();
    eprintln!("moonlight: {nl}-token prompt, {bad} non-finite decode logits");
    assert_eq!(bad, 0);
    assert_eq!(c1.bits(), c2.bits(), "moonlight: long feed_batch cache");
    assert_eq!(
        tests::bits(&l2),
        tests::bits(&d1[nl - 1]),
        "moonlight: long feed_batch logits"
    );
    let top = best(&dec[n - 1]);
    eprintln!(
        "moonlight: next token {top} {:?}",
        tok.decode(&[top as u32])
    );

    // Chat: two turns in one cache (reuse) vs the whole history fed fresh.
    use crate::chat::Chat;
    let sys = crate::chat::template_default_system(&m);
    assert_eq!(sys.as_deref(), Some("You are a helpful assistant"));
    let mut chat = Chat::new(&net, &tok, sys.as_deref(), 512).unwrap();
    chat.max_reply = 12;
    let never = || false;
    let r1 = chat
        .reply("Name a color.", &never, &mut |_| Ok(()))
        .unwrap();
    let r2 = chat
        .reply("And another one?", &never, &mut |_| Ok(()))
        .unwrap();
    let mut fresh = Chat::new(&net, &tok, sys.as_deref(), 512).unwrap();
    fresh.max_reply = 12;
    let f1 = fresh
        .reply("Name a color.", &never, &mut |_| Ok(()))
        .unwrap();
    assert_eq!(r1.ids, f1.ids);
    let hist = chat.history_for_test();
    let mut c = net.new_cache(512);
    // The reply's start in the history (a stop token may follow it there).
    let cut = (0..=hist.len() - r2.ids.len())
        .rev()
        .find(|&p| hist[p..p + r2.ids.len()] == r2.ids[..])
        .expect("reply in history");
    net.feed_batch(&hist[..cut - 1], 0, &mut c);
    let mut last = net.forward(hist[cut - 1], cut - 1, &mut c);
    eprintln!(
        "moonlight chat: history {} tokens, reply {:?}, {} non-finite logits at the reply's start",
        hist.len(),
        r2.ids,
        last.iter().filter(|v| !v.is_finite()).count()
    );
    for (i, &t) in r2.ids.iter().enumerate() {
        assert_eq!(best(&last), t as usize, "reply token {i}: reuse vs fresh");
        last = net.forward(t, cut + i, &mut c);
    }
    eprintln!(
        "moonlight chat: {:?} / {:?}",
        tok.decode(&r1.ids),
        tok.decode(&r2.ids)
    );
}

/// The `kimi-k2` tokenizer against Hugging Face's (Moonlight's
/// tokenization_moonshot.py, tiktoken): every case token for token, plain
/// and with special-token parsing; the Kimi chat framing against its
/// `apply_chat_template`. Needs only the GGUF's header (mapped, not run).
#[test]
fn kimi_tokenizer_matches_hf() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(crate::testutil::DS2_MODEL);
    if !path.exists() {
        eprintln!("skipping: {} not present (optional)", path.display());
        return;
    }
    let m = crate::gguf::load_with(path.to_str().unwrap(), Some(false)).unwrap();
    let t = crate::tokenizer::Tokenizer::from_gguf(&m);
    let d = crate::testutil::read_json("tokens_kimi.json");
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        (v.as_array().expect("ids").iter())
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let cases = d["cases"].as_array().expect("cases");
    assert!(cases.len() >= 40);
    let mut bad = Vec::new();
    for c in cases {
        let text = c["text"].as_str().expect("text");
        if t.encode(text) != ids(&c["ids"]) || t.encode_special(text) != ids(&c["ids_special"]) {
            bad.push(text.chars().take(60).collect::<String>());
        }
        assert_eq!(t.decode(&t.encode(text)), text);
    }
    assert!(
        bad.is_empty(),
        "{} of {} differ from HF: {bad:?}",
        bad.len(),
        cases.len()
    );
    // No BOS (the file has no add_bos_token; the template adds none).
    assert_eq!(t.encode_prompt("hi"), t.encode("hi"));
    let fr = crate::chat::Framing::new(&t).expect("framing");
    assert!(matches!(fr.format, crate::chat::Format::Kimi { .. }));
    let sys = crate::chat::template_default_system(&m);
    let conv = |c: &serde_json::Value, sys: Option<&str>| {
        let msgs = c["messages"].as_array().unwrap();
        let mut out = fr.system(&t, sys);
        for (i, msg) in msgs.iter().enumerate() {
            let role = msg["role"].as_str().unwrap();
            let next = (i + 1 == msgs.len()).then_some("assistant");
            out.extend(fr.frame(&t, role, msg["content"].as_str().unwrap(), next));
        }
        out
    };
    assert_eq!(
        conv(&d["chat"], sys.as_deref()),
        ids(&d["chat"]["ids"]),
        "chat"
    );
    let s = d["chat_sys"]["system"].as_str();
    assert_eq!(
        conv(&d["chat_sys"], s),
        ids(&d["chat_sys"]["ids"]),
        "chat + system"
    );
}

/// The `deepseek-v3` pre-tokenizer (three sequential splits) on DeepSeek
/// V3.1's vocabulary against Hugging Face's tokenizer.json (through
/// `tokenizers` itself: transformers 5's AutoTokenizer rebuilds it as a
/// LlamaTokenizer, which drops the byte-level spaces), and the
/// DeepSeek3 chat framing against its template (thinking off). The GGUF:
/// AXON_TEST_DS3_HEADER, a header-only stub of unsloth's V3.1 Q4_K_M shard
/// 1 (sparse; only the header is read). Skipped without it.
#[test]
fn deepseek3_tokenizer_matches_hf() {
    let Ok(path) = std::env::var("AXON_TEST_DS3_HEADER") else {
        eprintln!("skipping: AXON_TEST_DS3_HEADER not set (optional)");
        return;
    };
    let m = crate::gguf::load_with(&path, Some(false)).unwrap();
    assert_eq!(
        crate::tokenizer::pre_of(&m),
        Ok(crate::tokenizer::Pre::DeepSeek3)
    );
    let t = crate::tokenizer::Tokenizer::from_gguf(&m);
    let d = crate::testutil::read_json("tokens_deepseek3.json");
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        (v.as_array().expect("ids").iter())
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let mut bad = Vec::new();
    let cases = d["cases"].as_array().expect("cases");
    for c in cases {
        let text = c["text"].as_str().expect("text");
        let want = ids(&c["ids_special"]);
        // Plain encode equals it when the text has no special-token strings.
        let plain = c["has_special"].as_bool() == Some(true) || t.encode(text) == want;
        let got = t.encode_special(text);
        if !plain || got != want {
            let i = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
            let ctx = |v: &[u32]| t.decode(&v[i.saturating_sub(3)..(i + 4).min(v.len())]);
            eprintln!(
                "deepseek-v3 differs at token {i}: axon {:?} vs HF {:?}",
                ctx(&got),
                ctx(&want)
            );
            bad.push(text.chars().take(60).collect::<String>());
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} differ from HF: {bad:?}",
        bad.len(),
        cases.len()
    );
    let fr = crate::chat::Framing::new(&t).expect("framing");
    assert!(matches!(fr.format, crate::chat::Format::DeepSeek3 { .. }));
    let c = &d["chat_sys"];
    let msgs = c["messages"].as_array().unwrap();
    let mut out = fr.system(&t, c["system"].as_str());
    // A user message opens the reply after it (as `Chat` frames turns).
    for msg in msgs {
        let role = msg["role"].as_str().unwrap();
        let next = (role == "user").then_some("assistant");
        out.extend(fr.frame(&t, role, msg["content"].as_str().unwrap(), next));
    }
    assert_eq!(out, ids(&c["ids"]), "chat");
}
