//! Gemma 3: an independent f64 reference forward pass on a real GGUF and
//! the CPU model's checks against it (sliding window, sandwich norms,
//! GeGLU, embedding scale, two RoPE tables).

use super::*;
use crate::kv::Half;

/// An f64 forward pass of a gemma3 GGUF written from llama.cpp's
/// src/models/gemma3.cpp, sharing nothing with `Qwen2` but `dequant`:
/// embedding * sqrtf(dim); per layer RMSNorm -> q, k, v; per-head RMSNorm
/// of q and k; NEOX RoPE (local layers base `rope.freq_base_swa` or 10000,
/// global layers `rope.freq_base` times the linear scale); keys and values
/// rounded to the cache's type (f16 unless AXON_KV=f32), as every backend
/// stores them; softmax(q k / sqrt(hd)) over positions in the window
/// (local layers: p - t < window) -> attn_output -> RMSNorm
/// (post_attention_norm) -> residual; RMSNorm -> gelu_tanh(gate) * up ->
/// down -> RMSNorm (post_ffw_norm) -> residual; output norm, logits on the
/// tied embedding (or output.weight). Weights are dequantized row by row,
/// activations stay f64 (no 8-bit quantization): the model's numbers
/// without the engines' activation rounding. Returns the logits at every
/// position in `at`.
pub(crate) fn gemma3_ref(m: &Model, ids: &[u32], at: &[usize]) -> Vec<Vec<f64>> {
    let a = &m.arch;
    assert_eq!(a, "gemma3");
    let num = |k: &str| meta_num(m, &format!("{a}.{k}"));
    let (dim, nh, nkv, hd, nl) = (
        num("embedding_length").unwrap() as usize,
        num("attention.head_count").unwrap() as usize,
        num("attention.head_count_kv").unwrap() as usize,
        num("attention.key_length").unwrap() as usize,
        num("block_count").unwrap() as usize,
    );
    let eps = num("attention.layer_norm_rms_epsilon").unwrap();
    let base_g = num("rope.freq_base").unwrap_or(10000.0);
    let base_l = num("rope.freq_base_swa").unwrap_or(10000.0);
    let lin = match num("rope.scaling.factor") {
        Some(f) if f != 0.0 => 1.0 / f,
        _ => 1.0,
    };
    let window = num("attention.sliding_window").map(|w| w as usize);
    // Pattern 6: layer il local iff il % 6 < 5 (the files here have no key).
    let local = |il: usize| window.is_some() && il % 6 < 5;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let tensor = |name: &str| m.tensors.iter().find(|t| t.name == name).unwrap();
    let vecf = |name: &str| -> Vec<f64> {
        let t = tensor(name);
        let mut v = vec![0f32; t.n_elements as usize];
        dequant(t.qtype.id, m.tensor_data(t), &mut v);
        v.into_iter().map(f64::from).collect()
    };
    // y = W x, rows dequantized on the fly, split over threads.
    let mv = |name: &str, x: &[f64]| -> Vec<f64> {
        let t = tensor(name);
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        assert_eq!(cols, x.len(), "{name}");
        let rb = row_bytes(t.qtype.id, cols);
        let data = m.tensor_data(t);
        let mut y = vec![0f64; rows];
        let per = rows.div_ceil(threads);
        std::thread::scope(|s| {
            for (c, ys) in y.chunks_mut(per).enumerate() {
                s.spawn(move || {
                    let mut row = vec![0f32; cols];
                    for (j, yv) in ys.iter_mut().enumerate() {
                        let r = c * per + j;
                        dequant(t.qtype.id, &data[r * rb..(r + 1) * rb], &mut row);
                        *yv = row.iter().zip(x).map(|(&w, &v)| w as f64 * v).sum();
                    }
                });
            }
        });
        y
    };
    let rms = |x: &[f64], g: &[f64]| -> Vec<f64> {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
        x.iter().zip(g).map(|(v, g)| v * inv * g).collect()
    };
    let rope = |v: &mut [f64], pos: usize, loc: bool| {
        for head in v.chunks_mut(hd) {
            for i in 0..hd / 2 {
                let th = if loc {
                    pos as f64 * base_l.powf(-2.0 * i as f64 / hd as f64)
                } else {
                    pos as f64 * base_g.powf(-2.0 * i as f64 / hd as f64) * lin
                };
                let (s, c) = (th.sin(), th.cos());
                let (x0, x1) = (head[i], head[i + hd / 2]);
                head[i] = x0 * c - x1 * s;
                head[i + hd / 2] = x0 * s + x1 * c;
            }
        }
    };
    let cache = |x: f64| -> f64 {
        if crate::kv::use_f32() {
            x as f32 as f64
        } else {
            Half::from_f32(x as f32).to_f32() as f64
        }
    };
    let gelu = |x: f64| {
        let k = (2.0 / std::f64::consts::PI).sqrt();
        0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
    };
    let embd = tensor("token_embd.weight");
    let erb = row_bytes(embd.qtype.id, dim);
    let out_name = match m.tensors.iter().any(|t| t.name == "output.weight") {
        true => "output.weight",
        false => "token_embd.weight",
    };
    let sc = (dim as f32).sqrt() as f64;
    let mut kc: Vec<Vec<Vec<f64>>> = vec![Vec::new(); nl];
    let mut vc: Vec<Vec<Vec<f64>>> = vec![Vec::new(); nl];
    let mut logits = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        let mut row = vec![0f32; dim];
        let i = id as usize;
        dequant(
            embd.qtype.id,
            &m.tensor_data(embd)[i * erb..(i + 1) * erb],
            &mut row,
        );
        let mut x: Vec<f64> = row.iter().map(|&v| v as f64 * sc).collect();
        for l in 0..nl {
            let n = |s: &str| format!("blk.{l}.{s}");
            let h = rms(&x, &vecf(&n("attn_norm.weight")));
            let (mut q, mut k, v) = (
                mv(&n("attn_q.weight"), &h),
                mv(&n("attn_k.weight"), &h),
                mv(&n("attn_v.weight"), &h),
            );
            let (qn, kn) = (
                vecf(&n("attn_q_norm.weight")),
                vecf(&n("attn_k_norm.weight")),
            );
            for hq in q.chunks_mut(hd) {
                let r = rms(hq, &qn);
                hq.copy_from_slice(&r);
            }
            for hk in k.chunks_mut(hd) {
                let r = rms(hk, &kn);
                hk.copy_from_slice(&r);
            }
            rope(&mut q, pos, local(l));
            rope(&mut k, pos, local(l));
            kc[l].push(k.iter().map(|&x| cache(x)).collect());
            vc[l].push(v.iter().map(|&x| cache(x)).collect());
            let t0 = match (local(l), window) {
                (true, Some(w)) => (pos + 1).saturating_sub(w),
                _ => 0,
            };
            let mut att = vec![0f64; nh * hd];
            for qh in 0..nh {
                let kh = qh / (nh / nkv);
                let qv = &q[qh * hd..(qh + 1) * hd];
                let s: Vec<f64> = (t0..=pos)
                    .map(|t| {
                        let kr = &kc[l][t][kh * hd..(kh + 1) * hd];
                        qv.iter().zip(kr).map(|(a, b)| a * b).sum::<f64>() / (hd as f64).sqrt()
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for (j, p) in e.iter().enumerate() {
                    for d in 0..hd {
                        att[qh * hd + d] += p / z * vc[l][t0 + j][kh * hd + d];
                    }
                }
            }
            let o = rms(
                &mv(&n("attn_output.weight"), &att),
                &vecf(&n("post_attention_norm.weight")),
            );
            for (x, o) in x.iter_mut().zip(&o) {
                *x += o;
            }
            let h = rms(&x, &vecf(&n("ffn_norm.weight")));
            let (g, u) = (mv(&n("ffn_gate.weight"), &h), mv(&n("ffn_up.weight"), &h));
            let act: Vec<f64> = g.iter().zip(&u).map(|(&g, &u)| gelu(g) * u).collect();
            let d = rms(
                &mv(&n("ffn_down.weight"), &act),
                &vecf(&n("post_ffw_norm.weight")),
            );
            for (x, d) in x.iter_mut().zip(&d) {
                *x += d;
            }
        }
        if at.contains(&pos) {
            let h = rms(&x, &vecf("output_norm.weight"));
            logits.push(mv(out_name, &h));
        }
    }
    logits
}

fn log_softmax(l: &[f64]) -> Vec<f64> {
    let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = l.iter().map(|v| (v - mx).exp()).sum::<f64>().ln() + mx;
    l.iter().map(|v| v - z).collect()
}

/// Index of the largest value.
pub(crate) fn top1(v: &[f64]) -> usize {
    top(v, 1)[0].0
}

fn top(lp: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut o: Vec<usize> = (0..lp.len()).collect();
    o.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
    o[..k].iter().map(|&i| (i, lp[i])).collect()
}

/// Probe: AXON_REF_IDS (comma-separated token ids, BOS included) on
/// AXON_TEST_GEMMA3's model: top-5 log-probabilities after the last id from
/// `gemma3_ref` and from the CPU model (feed_batch then forward).
#[test]
#[ignore]
fn gemma3_ref_probe() {
    let ids: Vec<u32> = std::env::var("AXON_REF_IDS")
        .expect("AXON_REF_IDS")
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let path = crate::testutil::gemma3_model_path().expect("model");
    let m = crate::gguf::load(path.to_str().unwrap()).unwrap();
    let n = ids.len();
    let r = gemma3_ref(&m, &ids, &[n - 1]);
    let net = Qwen2::new(&m);
    let mut c = net.new_cache(n + 1);
    net.feed_batch(&ids[..n - 1], 0, &mut c);
    let l: Vec<f64> = net
        .forward(ids[n - 1], n - 1, &mut c)
        .into_iter()
        .map(f64::from)
        .collect();
    println!("ref  {:?}", top(&log_softmax(&r[0]), 5));
    println!("axon {:?}", top(&log_softmax(&l), 5));
}

/// Two logits closer than this in the reference are a tie for the top-token
/// check. With max |dlogit| <= d enforced, a top-1 flip is only possible
/// where the reference's top-2 gap is under 2d (< 2 at the 1.0 bound), so
/// the band is set from data, not from the bound. Issue #106's
/// measurements (Gemma 3 1B Q4_K_M; 4 prompts, every position, windows
/// None and 6, 140 points per CPU build, job 1020-021440-ff16): every
/// top-1 miss with d < 1 had a reference gap of at most 0.348 (0.348, 0.109,
/// 0.062, all in place, window 6). Every miss at a larger gap had d > 1 and
/// fails the bound. 0.5 sits just above 0.348. The 0.348 one is this test's
/// last position at window 6: the reference has 7001 at 17.984 and 506 at
/// 17.637, the in-place build (per-row kernels) picks 506 by 0.017 and the
/// r8 build picks 7001. The two builds differ only in the matmul kernels'
/// f32 accumulation order (docs/determinism.md), which rounding amplifies.
/// Be clear what this costs: where the reference gap is under 0.5 (the last
/// position in both window configs here) the top check accepts either of
/// the top two. Telling the window apart is left to `NEIGHBOR_MIN`.
const TOP_TIE: f64 = 0.5;

/// At window 6, at least this far (max |dlogit|) from the window-5 and
/// window-7 references wherever they differ from window 6's: an off-by-one
/// window bound fails here even if its top token falls in the `TOP_TIE`
/// band (the window-5 reference's top at the last position is 506, the
/// in-place build's pick). Measured (#106, job 1020-031341-ba0c): both
/// builds are 2.92-5.63 from the window-5 reference at positions 5-10 and
/// 3.88-7.92 from the window-7 one at positions 6-10, against at most 0.885
/// from window 6's.
const NEIGHBOR_MIN: f64 = 2.0;

/// The CPU model on the Gemma 3 1B file against `gemma3_ref`, one-token
/// decode and batched prefill: |dlogit| bounded (8-bit activations), and
/// the same top token up to a reference near-tie (`TOP_TIE`). A window
/// shorter than the prompt (the model's own 512 is set to 6 here via a
/// metadata edit) checks the local layers' masking at every position from
/// 5 on: the windowed reference and model agree, differ from the
/// full-attention ones, and are `NEIGHBOR_MIN` away from windows 5 and 7.
/// Both CPU builds run, whatever AXON_REPACK says: in place (per-row
/// kernels) and repacked (r8/r8q), each from its own load (repacking
/// releases the file's tensors).
#[test]
fn gemma3_matches_reference_forward() {
    let Some(path) = crate::testutil::gemma3_model_path() else {
        return;
    };
    let ids: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563, 9079, 236761, 669, 5279, 529];
    let n = ids.len();
    // Position 3 is inside any window here; 5..=10 are where window 6
    // differs from window 5 (and 6..=10 from window 7 and None).
    let at: Vec<usize> = vec![3, 5, 6, 7, 8, 9, n - 1];
    let load = |window: Option<usize>| {
        let mut m = crate::gguf::load(path.to_str().unwrap()).unwrap();
        if let Some(w) = window {
            for (k, v) in m.metadata.iter_mut() {
                if k == "gemma3.attention.sliding_window" {
                    *v = Val::U32(w as u32);
                }
            }
        }
        m
    };
    let maxd = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .fold(0f64, |mx, (x, y)| mx.max((x - y).abs()))
    };
    let mut full: Option<Vec<f64>> = None;
    for window in [None, Some(6usize)] {
        let want = gemma3_ref(&load(window), &ids, &at);
        // The window must matter at the last position (it's past 6).
        match &full {
            None => full = Some(want[at.len() - 1].clone()),
            Some(f) => {
                let d = maxd(f, &want[at.len() - 1]);
                assert!(d > 3.0, "the window changed the reference by {d} only");
            }
        }
        // Window 6's neighbours, and the first position each differs from it at.
        let neighbors: Vec<(usize, Vec<Vec<f64>>)> = match window {
            Some(w) => [w - 1, w + 1]
                .into_iter()
                .map(|nw| (nw, gemma3_ref(&load(Some(nw)), &ids, &at)))
                .collect(),
            None => Vec::new(),
        };
        for repack in [false, true] {
            let m = load(window);
            let net = Qwen2::new_with(&m, repack);
            let tag = format!("window {window:?}, repack {repack}");
            // Decode, one token at a time.
            let mut c = net.new_cache(n);
            let mut dec = Vec::new();
            for (p, &id) in ids.iter().enumerate() {
                let l = net.forward(id, p, &mut c);
                if at.contains(&p) {
                    dec.push(l);
                }
            }
            // Batched prefill, then the logits of the last position.
            let mut c2 = net.new_cache(n);
            net.feed_batch(&ids[..n - 1], 0, &mut c2);
            let last = net.forward(ids[n - 1], n - 1, &mut c2);
            assert_eq!(
                last.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                dec[at.len() - 1]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{tag}: batched prefill != decode"
            );
            for (i, (got, want)) in dec.iter().zip(&want).enumerate() {
                let p = at[i];
                let g64: Vec<f64> = got.iter().map(|&v| v as f64).collect();
                let d = maxd(&g64, want);
                let (tg, t2) = (top1(&g64), top(want, 2));
                let (tw, gap) = (t2[0].0, t2[0].1 - t2[1].1);
                println!(
                    "{tag}, pos {p}: max |dlogit| {d:.4}, top {tg} vs {tw}, reference top-2 gap {gap:.4}"
                );
                assert!(d < 1.0, "{tag}, pos {p}: max |dlogit| {d}");
                // The model's top token is the reference's, or within
                // TOP_TIE of it in the reference, and the model scores the
                // reference's top within TOP_TIE of its own.
                assert!(
                    tg == tw || (want[tw] - want[tg] < TOP_TIE && g64[tg] - g64[tw] < TOP_TIE),
                    "{tag}, pos {p}: top token {tg} vs reference {tw}: reference gap {}, model gap {}",
                    want[tw] - want[tg],
                    g64[tg] - g64[tw]
                );
                // Far from the neighbouring windows' references wherever
                // those differ from this window's (p >= min(nw, w)).
                for (nw, r) in &neighbors {
                    if p >= (*nw).min(window.unwrap()) {
                        let dn = maxd(&g64, &r[i]);
                        println!("{tag}, pos {p}: max |dlogit| vs window {nw} reference {dn:.4}");
                        assert!(
                            dn > NEIGHBOR_MIN,
                            "{tag}, pos {p}: only {dn} from the window {nw} reference"
                        );
                    }
                }
            }
        }
    }
}
