//! gpt-oss tests (a child of `model::tests`): load checks, the forward pass
//! against an independent f64 reference (llama.cpp's openai-moe graph
//! written out: sinks, sliding windows, biases, YaRN, top-k then softmax,
//! the clamped SwiGLU), and decode == batched prefill == verify, bit for
//! bit, on a small fixture and on the real model when it's there.

use super::tests::feed_batch_matches_feed_on;
use super::*;

/// A fixture's tensors: (name, dims, type, bytes).
type Ts = Vec<(String, Vec<u64>, u32, Vec<u8>)>;

/// (layers, dim, q heads, kv heads, head dim, expert ffn, experts, top-k,
/// vocab, window): a gpt-oss in miniature, 3 layers (local, global, local).
const OSS: (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
) = (3, 64, 4, 2, 16, 64, 4, 2, 64, 4);

/// 32 values as one MXFP4 block (17 bytes): the scale 2^(e-128) from the
/// largest magnitude, each value the nearest code.
fn mxfp4_block(x: &[f32]) -> Vec<u8> {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    // codes go up to 12 (6.0 doubled): 12 * 2^(e - 128) >= amax.
    let e = if amax == 0.0 {
        0
    } else {
        ((amax / 12.0).log2().ceil() as i32 + 128).clamp(2, 254) as u8
    };
    let d = f32::from_bits(((e as u32) - 1) << 23);
    let code = |v: f32| -> u8 {
        let kv = crate::quant::mxfp4_kvalues();
        let mut best = 0;
        for (i, &k) in kv.iter().enumerate() {
            if (k as f32 * d - v).abs() < (kv[best] as f32 * d - v).abs() {
                best = i;
            }
        }
        best as u8
    };
    let mut b = vec![e];
    for j in 0..16 {
        b.push(code(x[j]) | (code(x[j + 16]) << 4));
    }
    b
}

fn q8_0_bytes(v: &[f32]) -> Vec<u8> {
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

/// The fixture: attention Q8_0 with biases and sinks, MXFP4 experts with
/// biases, F32 router (+ bias), YaRN (factor 4 over 16), window 4 on the
/// even layers. `edit` changes the tensor list, `edit_meta` the metadata.
pub(super) fn oss_fixture_with(
    edit: impl FnOnce(&mut Ts),
    edit_meta: impl FnOnce(&mut Vec<(&'static str, Val)>),
) -> Model {
    let (nl, dim, nh, nkv, hd, ffn, ne, k, vocab, win) = OSS;
    let (qd, kvd) = (nh * hd, nkv * hd);
    let mut st = 0x2545_f491_4f6c_dd1du64;
    let mut uni = move |lo: f32, hi: f32| {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        lo + (hi - lo) * ((st >> 40) as f32 / (1u64 << 24) as f32)
    };
    let mut ts: Ts = Vec::new();
    let f32b = |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mat = |uni: &mut dyn FnMut(f32, f32) -> f32, cols: usize, rows: usize, rms: f32| {
        let a = rms * (3.0 / cols as f32).sqrt();
        (0..cols * rows).map(|_| uni(-a, a)).collect::<Vec<f32>>()
    };
    let embd: Vec<f32> = (0..dim * vocab).map(|_| uni(-1.0, 1.0)).collect();
    ts.push((
        "token_embd.weight".into(),
        vec![dim as u64, vocab as u64],
        Q8_0,
        q8_0_bytes(&embd),
    ));
    let out = mat(&mut uni, dim, vocab, 1.0);
    ts.push((
        "output.weight".into(),
        vec![dim as u64, vocab as u64],
        Q8_0,
        q8_0_bytes(&out),
    ));
    let on: Vec<f32> = (0..dim).map(|_| uni(0.8, 1.2)).collect();
    ts.push(("output_norm.weight".into(), vec![dim as u64], F32, f32b(on)));
    for i in 0..nl {
        let n = |s: &str| format!("blk.{i}.{s}");
        for (s, cols, rows, rms) in [
            ("attn_q.weight", dim, qd, 3.0),
            ("attn_k.weight", dim, kvd, 3.0),
            ("attn_v.weight", dim, kvd, 1.0),
            ("attn_output.weight", qd, dim, 1.0),
        ] {
            let v = mat(&mut uni, cols, rows, rms);
            ts.push((n(s), vec![cols as u64, rows as u64], Q8_0, q8_0_bytes(&v)));
        }
        for (s, len, lo, hi) in [
            ("attn_q.bias", qd, -0.5, 0.5),
            ("attn_k.bias", kvd, -0.5, 0.5),
            ("attn_v.bias", kvd, -0.5, 0.5),
            ("attn_output.bias", dim, -0.3, 0.3),
            ("attn_sinks.weight", nh, -1.0, 2.0),
            ("attn_norm.weight", dim, 0.8, 1.2),
            ("post_attention_norm.weight", dim, 0.8, 1.2),
            ("ffn_gate_inp.bias", ne, -0.5, 0.5),
        ] {
            let v: Vec<f32> = (0..len).map(|_| uni(lo, hi)).collect();
            ts.push((n(s), vec![len as u64], F32, f32b(v)));
        }
        let r = mat(&mut uni, dim, ne, 3.0);
        ts.push((
            n("ffn_gate_inp.weight"),
            vec![dim as u64, ne as u64],
            F32,
            f32b(r),
        ));
        // gate/up of a few units so the clamps (gate <= 7, |up| <= 7) bite.
        for (s, cols, rows, rms) in [
            ("ffn_gate_exps", dim, ffn, 4.0),
            ("ffn_up_exps", dim, ffn, 4.0),
            ("ffn_down_exps", ffn, dim, 1.0),
        ] {
            let v = mat(&mut uni, cols, rows * ne, rms);
            let bytes = v.chunks(32).flat_map(mxfp4_block).collect();
            let d = vec![cols as u64, rows as u64, ne as u64];
            ts.push((n(&format!("{s}.weight")), d, MXFP4, bytes));
            let b: Vec<f32> = (0..rows * ne).map(|_| uni(-0.5, 0.5)).collect();
            ts.push((
                n(&format!("{s}.bias")),
                vec![rows as u64, ne as u64],
                F32,
                f32b(b),
            ));
        }
    }
    edit(&mut ts);
    let a = "gpt-oss";
    let key = |s: &str| -> &'static str { format!("{a}.{s}").leak() };
    let mut meta = vec![
        ("general.architecture", Val::Str(a.into())),
        (key("block_count"), Val::U32(nl as u32)),
        (key("embedding_length"), Val::U32(dim as u32)),
        (key("feed_forward_length"), Val::U32(ffn as u32)),
        (key("attention.head_count"), Val::U32(nh as u32)),
        (key("attention.head_count_kv"), Val::U32(nkv as u32)),
        (key("attention.key_length"), Val::U32(hd as u32)),
        (key("attention.value_length"), Val::U32(hd as u32)),
        (key("attention.layer_norm_rms_epsilon"), Val::F32(1e-5)),
        (key("attention.sliding_window"), Val::U32(win as u32)),
        (key("rope.freq_base"), Val::F32(150000.0)),
        (key("rope.scaling.type"), Val::Str("yarn".into())),
        (key("rope.scaling.factor"), Val::F32(4.0)),
        (key("rope.scaling.original_context_length"), Val::U32(16)),
        (key("rope.scaling.yarn_beta_fast"), Val::F32(32.0)),
        (key("rope.scaling.yarn_beta_slow"), Val::F32(1.0)),
        (key("context_length"), Val::U32(64)),
        (key("expert_count"), Val::U32(ne as u32)),
        (key("expert_used_count"), Val::U32(k as u32)),
        (key("expert_feed_forward_length"), Val::U32(ffn as u32)),
        (
            "tokenizer.ggml.tokens",
            Val::Arr((0..vocab).map(|i| Val::Str(format!("t{i}"))).collect()),
        ),
    ];
    edit_meta(&mut meta);
    let list: Vec<(&str, &[u64], u32)> = ts.iter().map(|t| (t.0.as_str(), &t.1[..], t.2)).collect();
    let data: Vec<Vec<u8>> = ts.iter().map(|t| t.3.clone()).collect();
    crate::gguf::build_for_test_with(&meta, &list, &data)
}

fn oss_fixture() -> Model {
    oss_fixture_with(|_| {}, |_| {})
}

/// A tensor dequantized, in f64.
fn deq(m: &Model, name: &str) -> Vec<f64> {
    let t = find(m, name);
    let mut v = vec![0f32; t.n_elements as usize];
    dequant(t.qtype.id, m.tensor_data(t), &mut v);
    v.into_iter().map(f64::from).collect()
}

/// `w` (rows of `cols`) times x.
fn mv(w: &[f64], cols: usize, x: &[f64]) -> Vec<f64> {
    w.chunks(cols)
        .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn rms(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let s = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v / s * g).collect()
}

/// Ways to get gpt-oss wrong, for showing the reference tells them apart.
#[derive(Clone, Copy, PartialEq, Debug)]
enum OssBug {
    None,
    /// No attention sinks.
    NoSinks,
    /// Every layer global (no sliding window).
    NoWindow,
    /// No router bias.
    NoRouterBias,
    /// Plain SwiGLU (no clamp, no alpha, no +1).
    PlainSwiglu,
    /// No YaRN (plain RoPE frequencies, mscale 1).
    NoYarn,
}

/// The logits after each of `ids` by an f64 forward pass written from
/// llama.cpp's openai-moe graph (src/models/openai-moe.cpp, build_moe_ffn
/// with SOFTMAX_WEIGHT gating and swiglu_oai, soft_max_ext with sinks,
/// ggml's YaRN rope, NEOX pairs), none of the model's code.
fn oss_ref(m: &Model, ids: &[u32], bug: OssBug) -> Vec<Vec<f32>> {
    let (nl, dim, nh, nkv, hd, ffn, ne, k, _vocab, win) = OSS;
    let eps = 1e-5f64;
    let (base, factor, orig) = (150000f64, 4f64, 16f64);
    let half = hd / 2;
    // ggml_rope_yarn_corr_dims and rope_yarn: theta_extrap = base^(-2i/hd).
    let corr =
        |rot: f64| hd as f64 * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
    let (low, high) = (
        corr(32.0).floor().max(0.0),
        corr(1.0).ceil().min(hd as f64 - 1.0),
    );
    let theta: Vec<f64> = (0..half)
        .map(|i| {
            let ex = base.powf(-2.0 * i as f64 / hd as f64);
            if bug == OssBug::NoYarn {
                return ex;
            }
            let y = (i as f64 - low) / (high - low).max(0.001);
            let ramp = 1.0 - y.clamp(0.0, 1.0);
            ex / factor * (1.0 - ramp) + ex * ramp
        })
        .collect();
    let msc = if bug == OssBug::NoYarn {
        1.0
    } else {
        0.1 * factor.ln() + 1.0
    };
    let rope = |v: &mut [f64], pos: usize| {
        for h in v.chunks_mut(hd) {
            for i in 0..half {
                let a = pos as f64 * theta[i];
                let (s, c) = (a.sin() * msc, a.cos() * msc);
                let (x0, x1) = (h[i], h[i + half]);
                h[i] = x0 * c - x1 * s;
                h[i + half] = x0 * s + x1 * c;
            }
        }
    };
    let embd = deq(m, "token_embd.weight");
    let mut ks: Vec<Vec<Vec<f64>>> = vec![Vec::new(); nl];
    let mut vs: Vec<Vec<Vec<f64>>> = vec![Vec::new(); nl];
    let mut out = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        let mut x = embd[id as usize * dim..(id as usize + 1) * dim].to_vec();
        for l in 0..nl {
            let g = |s: &str| deq(m, &format!("blk.{l}.{s}"));
            let h = rms(&x, &g("attn_norm.weight"), eps);
            let add = |a: Vec<f64>, b: Vec<f64>| -> Vec<f64> {
                a.iter().zip(&b).map(|(x, y)| x + y).collect()
            };
            let mut q = add(mv(&g("attn_q.weight"), dim, &h), g("attn_q.bias"));
            let mut kk = add(mv(&g("attn_k.weight"), dim, &h), g("attn_k.bias"));
            let v = add(mv(&g("attn_v.weight"), dim, &h), g("attn_v.bias"));
            rope(&mut q, pos);
            rope(&mut kk, pos);
            ks[l].push(kk);
            vs[l].push(v);
            let local = l % 2 == 0 && bug != OssBug::NoWindow;
            let t0 = if local {
                (pos + 1).saturating_sub(win)
            } else {
                0
            };
            let sinks = g("attn_sinks.weight");
            let mut att = vec![0f64; nh * hd];
            for qh in 0..nh {
                let kvh = qh / (nh / nkv);
                let qv = &q[qh * hd..(qh + 1) * hd];
                let sc: Vec<f64> = (t0..=pos)
                    .map(|t| {
                        let kv = &ks[l][t][kvh * hd..(kvh + 1) * hd];
                        qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f64>() / (hd as f64).sqrt()
                    })
                    .collect();
                let sink = if bug == OssBug::NoSinks {
                    f64::NEG_INFINITY
                } else {
                    sinks[qh]
                };
                let mx = sc.iter().copied().fold(sink, f64::max);
                let e: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                let z = e.iter().sum::<f64>() + (sink - mx).exp();
                for (j, t) in (t0..=pos).enumerate() {
                    let vv = &vs[l][t][kvh * hd..(kvh + 1) * hd];
                    for d in 0..hd {
                        att[qh * hd + d] += e[j] / z * vv[d];
                    }
                }
            }
            let o = add(
                mv(&g("attn_output.weight"), nh * hd, &att),
                g("attn_output.bias"),
            );
            x = add(x, o);
            let h = rms(&x, &g("post_attention_norm.weight"), eps);
            let mut logits = mv(&g("ffn_gate_inp.weight"), dim, &h);
            if bug != OssBug::NoRouterBias {
                logits = add(logits, g("ffn_gate_inp.bias"));
            }
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            let sel = &order[..k];
            // Softmax over the k selected (over all experts then
            // renormalized over the k is the same thing).
            let w: Vec<f64> = {
                let mx = logits[sel[0]];
                let e: Vec<f64> = sel.iter().map(|&e| (logits[e] - mx).exp()).collect();
                let s: f64 = e.iter().sum();
                e.iter().map(|v| v / s).collect()
            };
            let (wg, wu, wd) = (
                g("ffn_gate_exps.weight"),
                g("ffn_up_exps.weight"),
                g("ffn_down_exps.weight"),
            );
            let (bg, bu, bd) = (
                g("ffn_gate_exps.bias"),
                g("ffn_up_exps.bias"),
                g("ffn_down_exps.bias"),
            );
            let mut y = vec![0f64; dim];
            for (&e, &we) in sel.iter().zip(&w) {
                let gt = mv(&wg[e * ffn * dim..(e + 1) * ffn * dim], dim, &h);
                let up = mv(&wu[e * ffn * dim..(e + 1) * ffn * dim], dim, &h);
                let act: Vec<f64> = (0..ffn)
                    .map(|r| {
                        let (gv, uv) = (gt[r] + bg[e * ffn + r], up[r] + bu[e * ffn + r]);
                        if bug == OssBug::PlainSwiglu {
                            return gv / (1.0 + (-gv).exp()) * uv;
                        }
                        let xg = gv.min(7.0);
                        let yu = uv.clamp(-7.0, 7.0);
                        xg / (1.0 + (-1.702 * xg).exp()) * (yu + 1.0)
                    })
                    .collect();
                let d = mv(&wd[e * dim * ffn..(e + 1) * dim * ffn], ffn, &act);
                for r in 0..dim {
                    y[r] += we * (d[r] + bd[e * dim + r]);
                }
            }
            x = add(x, y);
        }
        let h = rms(&x, &deq(m, "output_norm.weight"), eps);
        out.push(
            mv(&deq(m, "output.weight"), dim, &h)
                .into_iter()
                .map(|v| v as f32)
                .collect(),
        );
    }
    out
}

/// `Best`'s pick (greedy's rules).
fn argmax(l: &[f32]) -> u32 {
    let mut b = Best::NONE;
    for (i, &v) in l.iter().enumerate() {
        b.push(v, i as u32);
    }
    b.index(l.len())
}

fn ids(n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| (i * 37 + 11) % OSS.8 as u32)
        .collect()
}

#[test]
fn check_supported_gpt_oss() {
    check_supported(&oss_fixture()).expect("gpt-oss fixture");
    let rejects = |edit: &dyn Fn(&mut Ts), want: &str| {
        let m = oss_fixture_with(|ts| edit(ts), |_| {});
        let e = check_supported(&m).expect_err("malformed gpt-oss accepted");
        assert!(e.to_string().contains(want), "{e}");
    };
    rejects(
        &|ts| ts.retain(|t| t.0 != "blk.1.attn_sinks.weight"),
        "missing tensor blk.1.attn_sinks.weight",
    );
    rejects(
        &|ts| ts.retain(|t| t.0 != "blk.0.attn_output.bias"),
        "missing tensor blk.0.attn_output.bias",
    );
    rejects(
        &|ts| ts.retain(|t| t.0 != "blk.2.ffn_up_exps.bias"),
        "missing tensor blk.2.ffn_up_exps.bias",
    );
    rejects(
        &|ts| ts.retain(|t| t.0 != "blk.0.ffn_gate_inp.bias"),
        "missing tensor blk.0.ffn_gate_inp.bias",
    );
    rejects(
        &|ts| ts.retain(|t| t.0 != "blk.0.post_attention_norm.weight"),
        "missing tensor blk.0.post_attention_norm.weight",
    );
    let m = oss_fixture_with(
        |_| {},
        |m| m.push(("gpt-oss.rope.freq_base_swa", Val::F32(10000.0))),
    );
    assert!(
        check_supported(&m)
            .unwrap_err()
            .to_string()
            .contains("freq_base_swa")
    );
    // MXFP4 is gpt-oss's alone: a qwen3moe with MXFP4 experts is refused.
    let (mut q, _) = super::tests::dense_fixture("qwen3moe", Q8_0);
    assert!(check_supported(&q).is_ok());
    if let Some(t) = q
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ffn_up_exps.weight")
    {
        t.qtype = crate::gguf::qtype(MXFP4);
    }
    assert!(check_supported(&q).is_err());
    // A gpt-oss file with an unsupported attention tensor: the refusal names
    // it, and the mixed-file summary leaves out the (supported) MXFP4 experts.
    let mut m = oss_fixture();
    let retype = |m: &mut Model, name: &str, id: u32| {
        m.tensors.iter_mut().find(|t| t.name == name).unwrap().qtype = crate::gguf::qtype(id);
    };
    retype(&mut m, "blk.0.attn_q.weight", 20);
    let e = check_supported(&m).unwrap_err().to_string();
    assert!(e.contains("blk.0.attn_q.weight: type IQ4_NL (20)"), "{e}");
    assert!(
        !e.contains("the file has unsupported tensors") && !e.contains("MXFP4"),
        "{e}"
    );
    retype(&mut m, "blk.1.attn_k.weight", 22);
    let e = check_supported(&m).unwrap_err().to_string();
    assert!(
        e.contains("(the file has unsupported tensors: 1 IQ4_NL, 1 IQ2_S)"),
        "{e}"
    );
    assert!(!e.contains("MXFP4"), "{e}");
    // The layout: even layers local, window 4.
    let swa = gemma_swa(&oss_fixture(), 3).unwrap();
    assert_eq!((swa.window, swa.local), (4, vec![true, false, true]));
}

/// Axon's CPU forward vs the f64 reference: within 8-bit activation
/// quantization error, while each `OssBug` is well off (logits ~2-3).
#[test]
fn gpt_oss_matches_reference_forward() {
    let ids = ids(24);
    let want = oss_ref(&oss_fixture(), &ids, OssBug::None);
    let m = oss_fixture();
    let net = Qwen2::new(&m);
    let mut c = net.new_cache(ids.len());
    let got: Vec<Vec<f32>> = ids
        .iter()
        .enumerate()
        .map(|(p, &id)| net.forward(id, p, &mut c))
        .collect();
    let d = super::tests::max_diff(&got, &want);
    eprintln!("gpt-oss fixture vs f64 reference: max |dlogit| {d}");
    for bug in [
        OssBug::NoSinks,
        OssBug::NoWindow,
        OssBug::NoRouterBias,
        OssBug::PlainSwiglu,
        OssBug::NoYarn,
    ] {
        let b = oss_ref(&oss_fixture(), &ids, bug);
        let db = super::tests::max_diff(&b, &want);
        eprintln!("  {bug:?}: {db}");
        assert!(
            db > 2.5 * d.max(0.05),
            "{bug:?} only moves the logits by {db}"
        );
    }
    // MEASURED 0.32 (one position; the rest <= 0.19): 8-bit activations,
    // and with top-2 of 4 a near tie of two experts can flip.
    assert!(d < 0.4, "max |dlogit| {d}");
}

/// decode == batched prefill (every batch size, past the window) ==
/// verify, bit for bit; the cache after a reused prefix == a fresh prefill.
#[test]
fn gpt_oss_decode_batch_verify_bits() {
    for repack in [true, false] {
        let m = oss_fixture();
        let net = Qwen2::new_with(&m, repack);
        feed_batch_matches_feed_on(&net);
        contract(&net, 40);
    }
}

/// verify_batch rows == forward, and a cache that reuses a prefix (fed
/// before, then the rest in another batch) == one fresh prefill.
fn contract(net: &Qwen2, n: usize) {
    let v = net.vocab as u32;
    let ids: Vec<u32> = (0..n as u32)
        .map(|i| (i * 7919 + 13) % v.min(150_000))
        .collect();
    let mut a = net.new_cache(n);
    let fwd: Vec<Vec<f32>> = ids
        .iter()
        .enumerate()
        .map(|(p, &id)| net.forward(id, p, &mut a))
        .collect();
    let mut b = net.new_cache(n);
    let ver = net.verify_batch(&ids, 0, &mut b);
    for (p, (x, y)) in fwd.iter().zip(&ver).enumerate() {
        assert!(
            x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
            "verify != forward at {p}"
        );
    }
    let picks = net.verify_argmax(&ids, 0, &mut net.new_cache(n));
    assert_eq!(picks, fwd.iter().map(|l| argmax(l)).collect::<Vec<_>>());
    // Reuse: a prefix, then the rest, vs all at once.
    let cut = n / 3;
    let mut c = net.new_cache(n);
    net.feed_batch(&ids[..cut], 0, &mut c);
    net.feed_batch(&ids[cut..n - 1], cut, &mut c);
    let l1 = net.forward(ids[n - 1], n - 1, &mut c);
    let mut d = net.new_cache(n);
    net.feed_batch(&ids[..n - 1], 0, &mut d);
    let l2 = net.forward(ids[n - 1], n - 1, &mut d);
    assert_eq!(c.bits(), d.bits(), "reused-prefix cache != fresh prefill");
    assert!(l1.iter().zip(&l2).all(|(p, q)| p.to_bits() == q.to_bits()));
    assert!(
        fwd[n - 1]
            .iter()
            .zip(&l2)
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
}

/// The real model (optional, CPU): the same contract, and a sanity answer.
#[test]
fn gpt_oss_model_bits() {
    let Some(m) = crate::testutil::gptoss_model() else {
        return;
    };
    let net = Qwen2::new(&m);
    let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
    let p = tok.encode_prompt("The capital of France is");
    let mut c = net.new_cache(16);
    net.feed_batch(&p[..p.len() - 1], 0, &mut c);
    let l = net.forward(p[p.len() - 1], p.len() - 1, &mut c);
    assert_eq!(tok.decode(&[argmax(&l)]), " Paris");
    contract(&net, 200);
    feed_batch_matches_feed_on(&net);
}
