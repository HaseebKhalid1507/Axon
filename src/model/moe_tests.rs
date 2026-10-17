//! qwen3moe tests: load checks and `axon info`'s numbers.

use super::tests::tiny3;
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
