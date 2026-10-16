//! YaRN RoPE scaling (Peng et al. 2023, arXiv 2309.00071), static, as
//! llama.cpp, vLLM and HF transformers run it: past the trained context
//! (Qwen2.5 / Qwen3: 32768) to `factor` times it.
//!
//! Frequency side ("NTK-by-parts"): pair i keeps its frequency if it turns
//! at least `beta_fast` times within the original context, is divided by
//! `factor` if it turns at most `beta_slow` times, and blends linearly in
//! between (llama.cpp ggml.c ggml_rope_yarn_corr_dims, ops.cpp rope_yarn).
//! Magnitude side: every (sin, cos) of the table times `mscale`, by default
//! 0.1 ln(factor) + 1, so q and k both scale and the logits by mscale².
//!
//! Off unless asked for: by the GGUF (`rope.scaling.type` = yarn with a
//! factor > 1, as llama.cpp's converter writes for a `rope_type: yarn`
//! config) or by `--rope-scaling yarn --rope-scale F` (`RopeArgs`), which
//! writes the same keys into the model's metadata before it's checked and
//! built, and its `context_length` to orig x factor (llama.cpp's
//! n_ctx_train rewrite). With it off the tables are what they were, bit
//! for bit.

use crate::gguf::{Model, Val};
use crate::model::meta_num;

/// The YaRN parameters of a model (`yarn`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Yarn {
    pub factor: f64,
    /// The context the model was trained on (`original_context_length`).
    pub orig: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    /// Extrapolation mix (llama.cpp's ext_factor; 1).
    pub ext: f64,
    /// (sin, cos) multiplier, f32 as llama.cpp's.
    pub mscale: f32,
}

fn key(m: &Model, k: &str) -> String {
    format!("{}.rope.scaling.{k}", m.arch)
}

fn scaling_type(m: &Model) -> Option<&str> {
    let k = key(m, "type");
    m.metadata
        .iter()
        .find(|(n, _)| *n == k)
        .and_then(|(_, v)| match v {
            Val::Str(s) => Some(s.as_str()),
            _ => None,
        })
}

/// The default magnitude scale for `factor`: 0.1 ln(factor) + 1 (1 at or
/// below 1), in f32 as llama.cpp's get_mscale.
pub(crate) fn default_mscale(factor: f32) -> f32 {
    if factor <= 1.0 {
        1.0
    } else {
        0.1 * factor.ln() + 1.0
    }
}

/// Axon's own key for an explicit `--yarn-attn-factor` (the whole (sin,
/// cos) multiplier); GGUF's `attn_factor` multiplies the default instead.
const MSCALE_KEY: &str = "axon_mscale";

/// The model's YaRN parameters if it scales by YaRN (type yarn, factor > 1;
/// a factor of 1 changes nothing, as llama.cpp's). `check_supported`
/// validates them.
pub(crate) fn yarn(m: &Model) -> Option<Yarn> {
    if scaling_type(m) != Some("yarn") {
        return None;
    }
    let num = |k: &str| meta_num(m, &key(m, k));
    let factor = num("factor").filter(|&f| f > 1.0)?;
    let orig = num("original_context_length")
        .filter(|&n| n > 0.0)
        .or_else(|| meta_num(m, &format!("{}.context_length", m.arch)))
        .unwrap_or(0.0);
    let mscale = match num(MSCALE_KEY) {
        Some(s) => s as f32,
        None => num("attn_factor").unwrap_or(1.0) as f32 * default_mscale(factor as f32),
    };
    Some(Yarn {
        factor,
        orig,
        beta_fast: num("yarn_beta_fast").unwrap_or(32.0),
        beta_slow: num("yarn_beta_slow").unwrap_or(1.0),
        ext: num("yarn_ext_factor").filter(|&e| e >= 0.0).unwrap_or(1.0),
        mscale,
    })
}

impl Yarn {
    /// The first and last pair of the blend (llama.cpp's corr_dims): pairs
    /// up to `low` keep their frequency, from `high` on are divided by the
    /// factor.
    pub fn corr_dims(&self, base: f64, head_dim: usize) -> (f64, f64) {
        let n = head_dim as f64;
        let dim = |rot: f64| {
            n * (self.orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
        };
        let low = dim(self.beta_fast).floor().max(0.0);
        let high = dim(self.beta_slow).ceil().min(n - 1.0);
        (low, high)
    }

    /// Scale each pair's frequency (`freqs[i]`, Llama 3's factors already
    /// applied) in place.
    pub fn apply(&self, freqs: &mut [f64], base: f64, head_dim: usize) {
        let (low, high) = self.corr_dims(base, head_dim);
        for (i, f) in freqs.iter_mut().enumerate() {
            let y = (i as f64 - low) / (high - low).max(0.001);
            let ramp = (1.0 - y.clamp(0.0, 1.0)) * self.ext;
            // theta_interp (1 - ramp) + theta_extrap ramp.
            *f = *f / self.factor * (1.0 - ramp) + *f * ramp;
        }
    }

    pub fn ctx(&self) -> usize {
        (self.orig * self.factor) as usize
    }
}

/// The (sin, cos) multiplier of the model's RoPE tables: YaRN's mscale, or 1.
pub(crate) fn mscale(m: &Model) -> f32 {
    yarn(m).map_or(1.0, |y| y.mscale)
}

/// `--rope-scaling`, `--rope-scale`, `--yarn-orig-ctx`, `--yarn-beta-fast`,
/// `--yarn-beta-slow`, `--yarn-attn-factor` (llama.cpp's names), for gen,
/// logits, chat and serve.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RopeArgs {
    pub scaling: Option<String>,
    pub factor: Option<f64>,
    pub orig_ctx: Option<usize>,
    pub beta_fast: Option<f64>,
    pub beta_slow: Option<f64>,
    pub attn_factor: Option<f64>,
}

pub const FLAGS: [&str; 6] = [
    "--rope-scaling",
    "--rope-scale",
    "--yarn-orig-ctx",
    "--yarn-beta-fast",
    "--yarn-beta-slow",
    "--yarn-attn-factor",
];

impl RopeArgs {
    pub fn is_flag(f: &str) -> bool {
        FLAGS.contains(&f)
    }

    /// Set flag `f` (one of `FLAGS`) from its value.
    pub fn set(&mut self, f: &str, v: Option<&str>) -> Result<(), String> {
        let v = v.ok_or_else(|| format!("{f} needs a value"))?;
        let pos = || {
            v.parse::<f64>()
                .ok()
                .filter(|x| x.is_finite() && *x > 0.0)
                .ok_or_else(|| format!("{f} needs a positive number, got '{v}'"))
        };
        match f {
            "--rope-scaling" => match v {
                "yarn" | "none" => self.scaling = Some(v.into()),
                _ => {
                    return Err(format!(
                        "--rope-scaling: '{v}' isn't supported (yarn or none)"
                    ));
                }
            },
            "--rope-scale" => {
                self.factor = Some(pos()?).filter(|&x| x >= 1.0).map_or_else(
                    || Err(format!("--rope-scale needs a number >= 1, got '{v}'")),
                    |x| Ok(Some(x)),
                )?
            }
            "--yarn-orig-ctx" => {
                self.orig_ctx = Some(
                    v.parse::<usize>()
                        .ok()
                        .filter(|&n| n > 0 && n <= u32::MAX as usize)
                        .ok_or_else(|| format!("{f} needs a positive whole number, got '{v}'"))?,
                )
            }
            "--yarn-beta-fast" => self.beta_fast = Some(pos()?),
            "--yarn-beta-slow" => self.beta_slow = Some(pos()?),
            "--yarn-attn-factor" => self.attn_factor = Some(pos()?),
            _ => return Err(format!("unknown flag {f}")),
        }
        Ok(())
    }

    /// Nothing given: the model runs as its file says.
    pub fn is_empty(&self) -> bool {
        *self == RopeArgs::default()
    }

    /// Write the flags into `m`'s metadata (before `check_supported`):
    /// a usage error if they don't make sense. Returns a line describing
    /// the scaling, if any.
    pub fn apply(&self, m: &mut Model) -> Result<Option<String>, String> {
        if self.is_empty() {
            return Ok(None);
        }
        let yarn_only = self.factor.is_some()
            || self.orig_ctx.is_some()
            || self.beta_fast.is_some()
            || self.beta_slow.is_some()
            || self.attn_factor.is_some();
        match self.scaling.as_deref() {
            Some("none") if yarn_only => {
                return Err("--rope-scaling none takes no --rope-scale / --yarn-* flags".into());
            }
            Some("none") => {
                let p = key(m, "");
                m.metadata.retain(|(k, _)| !k.starts_with(&p));
                return Ok(Some("rope scaling: none".into()));
            }
            None => {
                return Err("--rope-scale and --yarn-* need --rope-scaling yarn".into());
            }
            _ => {}
        }
        if m.arch == "gemma3" {
            return Err("--rope-scaling yarn isn't supported on gemma3".into());
        }
        let ctx_key = format!("{}.context_length", m.arch);
        let trained = meta_num(m, &ctx_key);
        let file = yarn(m);
        let factor = self
            .factor
            .or(file.map(|y| y.factor))
            .ok_or("--rope-scaling yarn needs --rope-scale F (e.g. 4: 32768 -> 131072)")?;
        let orig = match (self.orig_ctx, file) {
            (Some(n), _) => n as f64,
            (None, Some(y)) => y.orig,
            (None, None) => {
                trained.ok_or("the model has no context_length: give --yarn-orig-ctx")?
            }
        };
        fn set(m: &mut Model, k: String, v: Val) {
            match m.metadata.iter_mut().find(|(n, _)| *n == k) {
                Some(e) => e.1 = v,
                None => m.metadata.push((k, v)),
            }
        }
        set(m, key(m, "type"), Val::Str("yarn".into()));
        set(m, key(m, "factor"), Val::F32(factor as f32));
        set(m, key(m, "original_context_length"), Val::U32(orig as u32));
        if let Some(b) = self.beta_fast {
            set(m, key(m, "yarn_beta_fast"), Val::F32(b as f32));
        }
        if let Some(b) = self.beta_slow {
            set(m, key(m, "yarn_beta_slow"), Val::F32(b as f32));
        }
        if let Some(s) = self.attn_factor {
            set(m, key(m, MSCALE_KEY), Val::F32(s as f32));
        }
        let y = yarn(m);
        // llama.cpp: n_ctx_train = n_ctx_orig_yarn / freq_scale.
        let ctx = y.map_or(orig as usize, |y| y.ctx());
        set(m, ctx_key, Val::U32(ctx.min(u32::MAX as usize) as u32));
        Ok(Some(match y {
            Some(y) => format!(
                "rope scaling: yarn, factor {}, original context {}, context {ctx}, beta {}/{}, mscale {:.4}",
                y.factor, y.orig, y.beta_fast, y.beta_slow, y.mscale
            ),
            None => format!("rope scaling: yarn, factor {factor} (no change), context {ctx}"),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn y(factor: f64) -> Yarn {
        Yarn {
            factor,
            orig: 32768.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            ext: 1.0,
            mscale: default_mscale(factor as f32),
        }
    }

    /// The Qwen2.5-7B recipe (research-longctx.md L2): pairs 0..=23 keep
    /// their frequency, 40.. are divided by 4, mscale 0.1 ln 4 + 1.
    #[test]
    fn qwen_7b_recipe() {
        let y = y(4.0);
        assert_eq!(y.corr_dims(1e6, 128), (23.0, 40.0));
        assert!((y.mscale - 1.138_629_4).abs() < 1e-6);
        let base: Vec<f64> = (0..64)
            .map(|i| 1e6f64.powf(-2.0 * i as f64 / 128.0))
            .collect();
        let mut f = base.clone();
        y.apply(&mut f, 1e6, 128);
        for i in 0..=23 {
            assert_eq!(f[i], base[i], "pair {i}");
        }
        for i in 40..64 {
            assert_eq!(f[i], base[i] / 4.0, "pair {i}");
        }
        for i in 24..40 {
            assert!(f[i] < base[i] && f[i] > base[i] / 4.0, "pair {i}");
        }
        assert_eq!(y.ctx(), 131072);
    }

    #[test]
    fn flags() {
        let mut a = RopeArgs::default();
        assert!(a.is_empty());
        a.set("--rope-scaling", Some("yarn")).unwrap();
        a.set("--rope-scale", Some("4")).unwrap();
        a.set("--yarn-orig-ctx", Some("32768")).unwrap();
        assert_eq!(a.factor, Some(4.0));
        assert!(a.set("--rope-scaling", Some("linear")).is_err());
        assert!(a.set("--rope-scale", Some("0.5")).is_err());
        assert!(a.set("--yarn-orig-ctx", Some("x")).is_err());
        assert!(a.set("--yarn-beta-fast", None).is_err());
    }
}
