//! Mixture-of-experts models (`qwen3moe`): what the loader checks and
//! `axon info` reports. A MoE layer replaces the gated FFN with `n_expert`
//! small ones and a router (an f32 `[n_expert x dim]` matrix); per token
//! the top `k` experts by softmax probability run, their outputs summed
//! with the probabilities renormalized to 1. Running them is not
//! implemented yet: every backend refuses these models with an error.

use crate::gguf::{Model, Tensor};

/// The MoE hyperparameters of a model (`qwen3moe.*`), None for a dense one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MoeDims {
    pub n_expert: usize,
    pub k: usize,
    pub ffn: usize,
}

pub(crate) fn is_moe(m: &Model) -> bool {
    m.arch == "qwen3moe"
}

pub(crate) fn moe_dims(m: &Model) -> Option<MoeDims> {
    if !is_moe(m) {
        return None;
    }
    let num = |k: &str| super::meta_num(m, &format!("{}.{k}", m.arch)).map(|v| v as usize);
    Some(MoeDims {
        n_expert: num("expert_count")?,
        k: num("expert_used_count")?,
        ffn: num("expert_feed_forward_length")?,
    })
}

/// The expert tensors of a layer, in the arena's order.
pub(super) const EXPS: [&str; 3] = [
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
];

/// Bytes of one expert matrix of `t` as the CPU holds it: repacked if it
/// has an r8/r8q form, else as in the file.
fn held_bytes(t: &Tensor) -> u64 {
    super::repacked_len(t).map_or(t.n_bytes / t.dims[2], |l| l as u64)
}

/// What `axon info` says about a MoE model: sizes from the tensor table.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MoeInfo {
    pub dims: MoeDims,
    pub shared: usize,
    /// All expert tensors, as in the file and as the CPU holds them (r8).
    pub expert_bytes: u64,
    pub expert_held: u64,
    /// k experts of every layer: what one token reads.
    pub active_bytes: u64,
    pub active_held: u64,
    /// Everything else (attention, routers, norms, embedding, head).
    pub dense_bytes: u64,
}

impl MoeInfo {
    pub fn of(m: &Model) -> Option<MoeInfo> {
        let dims = moe_dims(m)?;
        let mut i = MoeInfo {
            dims,
            shared: super::meta_num(m, &format!("{}.expert_shared_count", m.arch))
                .map_or(0, |v| v as usize),
            expert_bytes: 0,
            expert_held: 0,
            active_bytes: 0,
            active_held: 0,
            dense_bytes: 0,
        };
        for t in &m.tensors {
            if EXPS.iter().any(|s| t.name.ends_with(s)) && t.dims.len() >= 3 && t.dims[2] > 0 {
                let n = t.dims[2];
                i.expert_bytes += t.n_bytes;
                i.expert_held += held_bytes(t) * n;
                i.active_bytes += t.n_bytes / n * dims.k as u64;
                i.active_held += held_bytes(t) * dims.k as u64;
            } else {
                i.dense_bytes += t.n_bytes;
            }
        }
        Some(i)
    }

    /// Two lines for `axon info`.
    pub fn lines(&self) -> [String; 2] {
        let gb = |b: u64| format!("{:.2} GB", b as f64 / 1e9);
        let d = self.dims;
        [
            format!(
                "moe: {} experts, top-{}, expert ffn {}, {} shared | not runnable yet",
                d.n_expert,
                d.k,
                d.ffn,
                if self.shared == 0 {
                    "none".to_string()
                } else {
                    self.shared.to_string()
                }
            ),
            format!(
                "experts {} B ({}; r8 {}) | active/token {} B ({}; r8 {}) | non-expert {} B ({})",
                self.expert_bytes,
                gb(self.expert_bytes),
                gb(self.expert_held),
                self.active_bytes,
                gb(self.active_bytes),
                gb(self.active_held),
                self.dense_bytes,
                gb(self.dense_bytes)
            ),
        ]
    }
}

/// The refusal of any GPU placement for now.
pub(crate) fn gpu_refusal(m: &Model) -> String {
    format!(
        "{}: mixture-of-experts models run on the CPU only for now (--device cpu); \
         attention on the GPU with the experts on the CPU is not implemented yet",
        m.arch
    )
}

/// The CPU's refusal until the forward pass exists.
pub(crate) fn cpu_refusal(m: &Model) -> String {
    format!(
        "{}: mixture-of-experts models are recognized (axon info) but can't run yet",
        m.arch
    )
}
