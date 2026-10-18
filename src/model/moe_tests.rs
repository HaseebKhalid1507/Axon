//! qwen3moe tests (a child of `model::tests`): load checks, the forward
//! pass against the independent f64 reference, and decode == batched
//! prefill == verify, bit for bit, on the fixture and the dev models.

use super::tests::{
    NormBug, QWEN3_Q8_TOL, RopeBug, dense_fixture, dense_ref, feed_batch_matches_feed_on, max_diff,
    tiny3,
};
use super::*;

/// `tiny3` turned into a qwen3moe: 4 experts of ffn 128, top-2, an F32
/// router; `edit` / `edit_meta` then change it.
fn tiny_moe(
    edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>),
    edit_meta: impl FnOnce(&mut Vec<(&'static str, Val)>),
) -> std::io::Result<()> {
    tiny3(
        |ts| {
            ts.retain(|t| !t.0.contains(".ffn_gate.") && !t.0.contains(".ffn_up."));
            ts.retain(|t| !t.0.contains(".ffn_down."));
            for i in 0..2 {
                for (n, d, ty) in [
                    ("ffn_gate_inp.weight", vec![64, 4], F32),
                    ("ffn_gate_exps.weight", vec![64, 128, 4], Q8_0),
                    ("ffn_up_exps.weight", vec![64, 128, 4], Q8_0),
                    ("ffn_down_exps.weight", vec![128, 64, 4], Q8_0),
                ] {
                    ts.push((format!("blk.{i}.{n}"), d, ty));
                }
            }
            edit(ts);
        },
        |m| {
            for kv in m.iter_mut() {
                if let Some(k) = kv.0.strip_prefix("qwen3.") {
                    kv.0 = format!("qwen3moe.{k}").leak();
                }
            }
            m[0].1 = Val::Str("qwen3moe".into());
            m.push(("qwen3moe.expert_count", Val::U32(4)));
            m.push(("qwen3moe.expert_used_count", Val::U32(2)));
            m.push(("qwen3moe.expert_feed_forward_length", Val::U32(128)));
            edit_meta(m);
        },
    )
}

fn moe_rejects(
    edit: impl FnOnce(&mut Vec<(String, Vec<u64>, u32)>),
    edit_meta: impl FnOnce(&mut Vec<(&'static str, Val)>),
    want: &str,
) {
    let e = tiny_moe(edit, edit_meta).expect_err("malformed qwen3moe accepted");
    assert!(e.to_string().contains(want), "{e}");
}

#[test]
fn check_supported_qwen3moe() {
    tiny_moe(|_| {}, |_| {}).expect("tiny qwen3moe");
    let meta = |k: &'static str, v: Val| move |m: &mut Vec<(&'static str, Val)>| m.push((k, v));
    let none = |_: &mut Vec<(String, Vec<u64>, u32)>| {};
    let no_meta = |_: &mut Vec<(&'static str, Val)>| {};
    moe_rejects(
        none,
        meta("qwen3moe.expert_shared_count", Val::U32(1)),
        "shared experts",
    );
    moe_rejects(
        none,
        meta("qwen3moe.expert_weights_scale", Val::F32(2.5)),
        "expert_weights_scale 2.5",
    );
    moe_rejects(
        none,
        meta("qwen3moe.expert_gating_func", Val::U32(2)),
        "softmax only",
    );
    moe_rejects(
        none,
        meta("qwen3moe.expert_group_count", Val::U32(4)),
        "grouped expert",
    );
    moe_rejects(
        none,
        |m| {
            let kv = m.iter_mut().find(|kv| kv.0 == "qwen3moe.expert_used_count");
            kv.unwrap().1 = Val::U32(5);
        },
        "expert_used_count 5 exceeds expert_count 4",
    );
    moe_rejects(
        none,
        |m| m.retain(|kv| kv.0 != "qwen3moe.expert_feed_forward_length"),
        "expert_feed_forward_length is missing",
    );
    moe_rejects(
        |ts| ts.push(("blk.1.ffn_gate.weight".into(), vec![64, 128], Q8_0)),
        no_meta,
        "blk.1.ffn_gate.weight is not supported on qwen3moe",
    );
    moe_rejects(
        |ts| ts.push(("blk.0.ffn_up_shexp.weight".into(), vec![64, 128], Q8_0)),
        no_meta,
        "blk.0.ffn_up_shexp.weight is not supported",
    );
    moe_rejects(
        |ts| ts.retain(|t| t.0 != "blk.1.ffn_down_exps.weight"),
        no_meta,
        "missing tensor blk.1.ffn_down_exps.weight",
    );
    moe_rejects(
        |ts| ts.retain(|t| t.0 != "blk.0.ffn_gate_inp.weight"),
        no_meta,
        "missing tensor blk.0.ffn_gate_inp.weight",
    );
    let set = |name: &'static str, dims: Vec<u64>, ty: u32| {
        move |ts: &mut Vec<(String, Vec<u64>, u32)>| {
            let t = ts.iter_mut().find(|t| t.0 == name).unwrap();
            (t.1, t.2) = (dims, ty);
        }
    };
    moe_rejects(
        set("blk.0.ffn_up_exps.weight", vec![64, 128, 3], Q8_0),
        no_meta,
        "blk.0.ffn_up_exps.weight: shape [64, 128, 3]",
    );
    moe_rejects(
        set("blk.1.ffn_down_exps.weight", vec![64, 128, 4], Q8_0),
        no_meta,
        "blk.1.ffn_down_exps.weight: shape",
    );
    moe_rejects(
        set("blk.0.ffn_gate_inp.weight", vec![64, 4], Q8_0),
        no_meta,
        "ffn_gate_inp.weight: type Q8_0",
    );
    moe_rejects(
        set("blk.0.ffn_gate_exps.weight", vec![64, 128, 4], 9),
        no_meta,
        "Q8_1 (9) is not supported",
    );
    // qwen3moe needs its q/k norms, as qwen3.
    moe_rejects(
        |ts| ts.retain(|t| t.0 != "blk.0.attn_k_norm.weight"),
        no_meta,
        "missing tensor blk.0.attn_k_norm.weight",
    );
    // Experts on another architecture: still refused.
    let e = tiny3(|_| {}, |m| m.push(("qwen3.expert_count", Val::U32(4)))).unwrap_err();
    assert!(e.to_string().contains("not supported on qwen3"), "{e}");
}

/// The router, built on its own: experts' rows are scaled copies of one
/// direction, so logits (and probabilities) tie exactly where the scales do.
fn route_of(scales: &[f32], k: usize, h: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let dims = moe::MoeDims {
        n_expert: scales.len(),
        k,
        ffn: 8,
    };
    let router: Vec<f32> = scales
        .iter()
        .flat_map(|&s| (0..h.len()).map(move |c| s * (1.0 + c as f32 / 8.0)))
        .collect();
    let r = moe::Moe::new(
        dims,
        router,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        moe::Gate::SOFTMAX_NORM,
    );
    let (mut ids, mut w) = (vec![0; k], vec![0.0; k]);
    r.route(h, &mut ids, &mut w);
    (ids, w)
}

/// Exact top-k: rank order by probability, ties to the lower expert index;
/// weights renormalized by division.
#[test]
fn route_top_k_ties_and_renorm() {
    let h = [0.25f32, -0.5, 0.75, 0.125];
    let (ids, w) = route_of(&[0.1, 0.4, 0.4, -0.3, 0.4, 0.2], 4, &h);
    assert_eq!(ids, [1, 2, 4, 5]);
    // Tied experts get identical weights.
    assert_eq!(w[0].to_bits(), w[1].to_bits());
    assert_eq!(w[1].to_bits(), w[2].to_bits());
    let s: f32 = w.iter().sum();
    assert!((s - 1.0).abs() < 1e-6, "{w:?}");
    // k == n: every expert, ties in index order.
    let (ids, _) = route_of(&[0.3, 0.3, 0.3], 3, &h);
    assert_eq!(ids, [0, 1, 2]);
    // The same h always routes the same, whatever was routed before.
    let a = route_of(&[0.7, -0.2, 0.9, 0.1], 2, &h);
    let _ = route_of(&[0.1, 0.4, 0.4, -0.3, 0.4, 0.2], 4, &h);
    let b = route_of(&[0.7, -0.2, 0.9, 0.1], 2, &h);
    assert_eq!(a.0, b.0);
    assert_eq!(a.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), {
        b.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    });
    assert_eq!(a.0, [2, 0]);
}

/// The CPU qwen3moe (decode and batched prefill, repacked or not) against
/// the f64 reference (`dense_ref`'s MoE FFN: softmax, top k, renormalized
/// weights). F32 fixture: f32 arithmetic and the f16 cache only; Q8_0: the
/// 8-bit activations (the qwen3 fixture's bound).
#[test]
fn moe_matches_reference_forward() {
    const TOL: f32 = 2e-3;
    let ids = [3u32, 17, 42, 9, 30, 5];
    for (mt, tol) in [(F32, TOL), (Q8_0, QWEN3_Q8_TOL)] {
        let want = {
            let (m, _) = dense_fixture("qwen3moe", mt);
            dense_ref(&m, None, &ids, NormBug::None, RopeBug::None)
        };
        for repack in [true, false] {
            let (m, _) = dense_fixture("qwen3moe", mt);
            let net = Qwen2::new_with(&m, repack);
            let mut c = net.new_cache(ids.len());
            let dec: Vec<Vec<f32>> = (0..ids.len())
                .map(|p| net.forward(ids[p], p, &mut c))
                .collect();
            let mut c = net.new_cache(ids.len());
            net.feed_batch(&ids[..5], 0, &mut c);
            let pre = net.forward(ids[5], 5, &mut c);
            let (d, dp) = (max_diff(&dec, &want), max_diff(&[pre], &want[5..]));
            eprintln!("moe fixture type {mt} repack {repack}: vs reference {d:.2e} / {dp:.2e}");
            assert!(
                d <= tol && dp <= tol,
                "type {mt} repack {repack}: {d} / {dp}"
            );
        }
    }
}

/// decode == feed_batch, bit for bit (every verify batch size, the split
/// token ranges, the parallel passes), on the Q8_0 fixture (r8q experts)
/// and the F32 one (in place).
#[test]
fn moe_feed_batch_matches_feed_fixture() {
    for (mt, repack) in [(Q8_0, true), (Q8_0, false), (F32, true)] {
        let (m, _) = dense_fixture("qwen3moe", mt);
        let net = Qwen2::new_with(&m, repack);
        assert!(matches!(net.layers[0].ffn, Ffn::Moe(_)));
        feed_batch_matches_feed_on(&net);
    }
}

/// `feed_batch_matches_feed_on` on a dev model, repacked and not.
fn moe_feed_batch_matches_feed(small: bool) {
    for repack in [true, false] {
        // A fresh load each: repacking releases the originals.
        let Some(m) = crate::testutil::moe_model(small) else {
            return;
        };
        let net = Qwen2::new_with(&m, repack);
        let Ffn::Moe(moe) = &net.layers[0].ffn else {
            panic!("not a MoE layer")
        };
        assert_eq!(moe.down[0].r8.is_some(), repack);
        feed_batch_matches_feed_on(&net);
    }
}

#[test]
fn moe_feed_batch_matches_feed_pruned() {
    moe_feed_batch_matches_feed(false);
}

#[test]
fn moe_feed_batch_matches_feed_small() {
    moe_feed_batch_matches_feed(true);
}

/// verify (batched, with the head) == forward per token, bit for bit, and
/// verify_argmax == argmax of each row, after a batched prefill.
fn moe_verify_matches_forward(small: bool) {
    let Some(m) = crate::testutil::moe_model(small) else {
        return;
    };
    let net = Qwen2::new(&m);
    let ids: Vec<u32> = (0..40u64)
        .map(|i| (i * 2_654_435_761 % 150_000) as u32)
        .collect();
    let (pre, tail) = ids.split_at(28);
    let mut a = net.new_cache(ids.len());
    net.feed_batch(pre, 0, &mut a);
    let mut b = net.new_cache(ids.len());
    net.feed_batch(pre, 0, &mut b);
    let mut c = net.new_cache(ids.len());
    net.feed_batch(pre, 0, &mut c);
    let ver = net.verify_batch(tail, pre.len(), &mut a);
    let picks = net.verify_argmax(tail, pre.len(), &mut c);
    for (j, &t) in tail.iter().enumerate() {
        let f = net.forward(t, pre.len() + j, &mut b);
        let same = f
            .iter()
            .zip(&ver[j])
            .all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same, "row {j}: verify != forward");
        assert_eq!(picks[j], crate::argmax(&f), "row {j}: verify_argmax");
    }
}

#[test]
fn moe_verify_matches_forward_pruned() {
    moe_verify_matches_forward(false);
}

#[test]
fn moe_verify_matches_forward_small() {
    moe_verify_matches_forward(true);
}

/// AXON_REPACK=0 (experts read in place, per-row kernels) vs the r8 repack,
/// graded as `repack_logit_deltas`: logprobs within 1 after a prompt.
#[test]
fn moe_repack_logit_deltas() {
    for small in [true, false] {
        if crate::testutil::moe_model_path(small).is_none() {
            continue;
        }
        let ids: Vec<u32> = (0..40u64)
            .map(|i| (i * 2_654_435_761 % 150_000) as u32)
            .collect();
        let run = |repack: bool| {
            let m = crate::testutil::moe_model(small).unwrap();
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
        let dlp = a
            .iter()
            .zip(&b)
            .fold(0f32, |d, (x, y)| d.max(((x - la) - (y - lb)).abs()));
        let (ta, tb) = (crate::argmax(&a), crate::argmax(&b));
        eprintln!("moe small {small}: r8 vs per-row max |dlogprob| {dlp:.3e}, argmax {ta} vs {tb}");
        assert!(dlp < 1.0, "small {small}: r8 moved a logprob by {dlp}");
    }
}

/// `axon info`'s MoE numbers on the Pruned-2B (the 30B's per-layer shapes):
/// the tensor table's byte counts, spec r11 §1.
#[test]
fn moe_info_counts() {
    let Some(m) = crate::testutil::moe_model(false) else {
        return;
    };
    let info = MoeInfo::of(&m).unwrap();
    assert_eq!(
        info.dims,
        moe::MoeDims {
            n_expert: 128,
            k: 8,
            ffn: 768
        }
    );
    // 2 Q4_K-down layers (339.74 MB of experts) + 1 Q6_K-down (391.64 MB).
    assert_eq!(info.expert_bytes, 2 * 339_738_624 + 391_643_136);
    assert_eq!(
        info.active_bytes,
        8 * (2 * 2 * 884_736 + 2 * 884_736 + 1_290_240 + 2 * 884_736)
    );
    assert_eq!(
        info.expert_bytes + info.dense_bytes,
        m.tensors.iter().map(|t| t.n_bytes).sum::<u64>()
    );
}

/// Load plumbing: every expert is its own repacked matrix (r8 for the
/// K-quants) of one span of the arena, Q6_K down in layer 2 only.
#[test]
fn moe_layout() {
    let Some(m) = crate::testutil::moe_model(false) else {
        return;
    };
    let net = Qwen2::new(&m);
    for (l, layer) in net.layers.iter().enumerate() {
        let Ffn::Moe(moe) = &layer.ffn else {
            panic!("layer {l} not MoE")
        };
        assert_eq!((moe.n_expert, moe.k, moe.ffn), (128, 8, 768));
        let want_down = if l == 2 { Q6_K } else { Q4_K };
        assert_eq!(moe.down[0].qt(), want_down, "layer {l}");
        let mut last = None;
        for w in moe.weights() {
            let r = w.r8.as_ref().expect("expert repacked");
            if let Some(end) = last {
                assert!(r.off >= end, "experts overlap");
            }
            last = Some(r.off + r.len);
            assert_eq!(r.len, repacked_len(w.t).unwrap());
        }
    }
}
