//! Sampling: temperature, top-k, top-p, min-p and a repeat penalty, with a
//! seeded RNG, in llama.cpp's order: penalty -> top-k -> top-p -> min-p ->
//! temperature -> draw. Temperature 0 is not sampling: generation then takes
//! the greedy path in `generate` (the backend's argmax), untouched.
//!
//! Split in two so no vocab-sized work reaches the host on a GPU:
//!
//! 1. Candidates (the backend, `Lm::forward_cands` / `verify_cands`): the
//!    repeat penalty on the logits, then the `k` largest of them as
//!    (logit, id), best first, ties to the larger id (`argmax`'s rule: the
//!    first candidate is the greedy pick of the penalized logits). On the CPU
//!    `cands` (one vectorized scan); on the GPU sample.cu, which copies back
//!    only the k pairs. Both give the same candidates for the same logits.
//! 2. The draw (`pick`, here): top-p, min-p, temperature, then one uniform
//!    number from the RNG.
//!
//! `k` is top_k; top_k 0 (off) keeps every token, as llama.cpp does: the
//! candidates are then the whole vocab, sorted (a slower path: a full sort
//! per token, and on the GPU the logits come back to the host; top_k above
//! `CAND_MAX` takes it too). Either way each step is exactly llama.cpp's:
//! its top-p and min-p only ever see the top-k candidates.
//!
//! Speculative decoding (`generate`) samples losslessly: for a drafted token x
//! the position's token is drawn from its distribution p with the position's
//! own uniform number, and x is kept iff the draw is x. That accepts x with
//! probability p(x) and, on a reject, emits a token distributed as p without
//! x, renormalized: the standard accept/reject rule for a deterministic
//! (one-hot) draft. Since every decided token takes exactly one number from
//! the RNG either way, the output is the same token for token as without
//! drafting, for the same seed.

use crate::GenStats;
use crate::lm::Lm;
use std::collections::VecDeque;
use std::io;
use std::time::Instant;

/// Most candidates the GPU selects on the device per position; a bigger
/// top_k (or 0, off) has the host select from the whole row.
pub const CAND_MAX: usize = 256;

/// `CandSpec::k` for top_k off: every token.
pub const ALL: usize = usize::MAX;

/// One sampling candidate: a (penalized) logit and its token id.
// doc(hidden): reachable through `Lm::forward_cands`, not 0.1 API until
// the sampler is exported (#118 item 8).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cand {
    pub logit: f32,
    pub id: u32,
}

/// What a backend selects per position: the `k` best candidates of the
/// logits after the repeat penalty over the last `last_n` tokens.
// doc(hidden), as `Cand` (#118 item 8).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandSpec {
    pub k: usize,
    pub penalty: f32,
    pub last_n: usize,
}

/// Sampling settings. Defaults are ollama's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleCfg {
    /// > 0 (0 is greedy decoding, which never builds a `SampleCfg`).
    pub temp: f32,
    /// 0 = off (every token).
    pub top_k: usize,
    /// >= 1 = off.
    pub top_p: f32,
    /// 0 = off.
    pub min_p: f32,
    /// 1 = off.
    pub repeat_penalty: f32,
    /// Tokens the penalty looks back over (prompt included); 0 = off.
    pub repeat_last_n: usize,
    pub seed: u64,
}

pub const DEFAULT_TEMP: f32 = 0.8;

impl SampleCfg {
    pub fn spec(&self) -> CandSpec {
        CandSpec {
            k: match self.top_k {
                0 => ALL,
                k => k,
            },
            penalty: self.repeat_penalty,
            last_n: self.repeat_last_n,
        }
    }
}

/// Sampling flags as given on the command line; `resolve` fills in defaults.
#[derive(Clone, Copy, Debug, Default)]
pub struct SampleArgs {
    pub temp: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub seed: Option<u64>,
}

impl SampleArgs {
    pub fn is_flag(f: &str) -> bool {
        matches!(
            f,
            "--temp"
                | "--top-k"
                | "--top-p"
                | "--min-p"
                | "--repeat-penalty"
                | "--repeat-last-n"
                | "--seed"
        )
    }

    /// Set `flag` (one `is_flag` accepts) from its value; Err says what's wrong.
    pub fn set(&mut self, flag: &str, v: Option<&str>) -> Result<(), String> {
        let v = v.ok_or_else(|| format!("{flag} needs a value"))?;
        let float = |lo: f32, hi: f32| -> Result<f32, String> {
            match v.parse::<f32>() {
                Ok(x) if x.is_finite() && x >= lo && x <= hi => Ok(x),
                _ => Err(format!(
                    "{flag} needs a number from {lo} to {hi}, got '{v}'"
                )),
            }
        };
        let whole = || -> Result<usize, String> {
            v.parse::<usize>()
                .map_err(|_| format!("{flag} needs a whole number, got '{v}'"))
        };
        match flag {
            "--temp" => self.temp = Some(float(0.0, 100.0)?),
            "--top-k" => self.top_k = Some(whole()?),
            "--top-p" => self.top_p = Some(float(0.0, 1.0)?),
            "--min-p" => self.min_p = Some(float(0.0, 1.0)?),
            "--repeat-penalty" => {
                let p = float(0.0, 100.0)?;
                if p == 0.0 {
                    return Err(format!("{flag} must be above 0 (1 = off)"));
                }
                self.repeat_penalty = Some(p);
            }
            "--repeat-last-n" => self.repeat_last_n = Some(whole()?),
            "--seed" => {
                self.seed = Some(
                    v.parse::<u64>()
                        .map_err(|_| format!("{flag} needs a whole number, got '{v}'"))?,
                )
            }
            _ => return Err(format!("unknown sampling flag '{flag}'")),
        }
        Ok(())
    }

    /// Any flag besides --temp given.
    pub fn any_besides_temp(&self) -> bool {
        self.top_k.is_some()
            || self.top_p.is_some()
            || self.min_p.is_some()
            || self.repeat_penalty.is_some()
            || self.repeat_last_n.is_some()
            || self.seed.is_some()
    }

    /// The settings, `default_temp` unless --temp was given; None for
    /// temperature 0 (greedy). The seed is random unless --seed was given.
    pub fn resolve(&self, default_temp: f32) -> Option<SampleCfg> {
        let temp = self.temp.unwrap_or(default_temp);
        (temp > 0.0).then(|| SampleCfg {
            temp,
            top_k: self.top_k.unwrap_or(40),
            top_p: self.top_p.unwrap_or(0.9),
            min_p: self.min_p.unwrap_or(0.0),
            repeat_penalty: self.repeat_penalty.unwrap_or(1.1),
            repeat_last_n: self.repeat_last_n.unwrap_or(64),
            seed: self.seed.unwrap_or_else(random_seed),
        })
    }
}

/// A seed from the clock and the process id.
fn random_seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let mut s = t ^ (u64::from(std::process::id()) << 32);
    splitmix(&mut s)
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// xoshiro256**, seeded through splitmix64.
#[derive(Clone, Debug)]
pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut s = seed;
        Rng([
            splitmix(&mut s),
            splitmix(&mut s),
            splitmix(&mut s),
            splitmix(&mut s),
        ])
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1), 53 bits.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Settings plus the RNG's state: one per generation stream (a chat keeps
/// one across replies).
pub struct Sampler {
    pub cfg: SampleCfg,
    rng: Rng,
}

impl Sampler {
    pub fn new(cfg: SampleCfg) -> Sampler {
        Sampler {
            rng: Rng::new(cfg.seed),
            cfg,
        }
    }

    /// The RNG's state (kvsave: a saved chat's sampler continues its stream).
    pub fn rng_state(&self) -> [u64; 4] {
        self.rng.0
    }

    /// A sampler whose RNG is at `state` (`rng_state`).
    pub fn with_rng_state(cfg: SampleCfg, state: [u64; 4]) -> Sampler {
        Sampler {
            cfg,
            rng: Rng(state),
        }
    }

    /// The next token from a position's candidates; takes one number from
    /// the RNG, always.
    pub fn pick(&mut self, c: &[Cand]) -> u32 {
        let u = self.rng.f64();
        pick(&self.cfg, c, u)
    }
}

/// The order key of a logit: larger logit, larger key; -0.0 ties +0.0; NaN
/// below everything (0). sample.cu `tk_key`.
pub fn key(v: f32) -> u32 {
    if v.is_nan() {
        return 0;
    }
    let b = if v == 0.0 { 0 } else { v.to_bits() };
    if b >> 31 == 1 { !b } else { b | 0x8000_0000 }
}

/// The logit of a key (-0.0 comes back +0.0, NaN as 0xffff_ffff).
pub fn unkey(k: u32) -> f32 {
    f32::from_bits(if k >> 31 == 1 { k & 0x7fff_ffff } else { !k })
}

/// The candidate order, as one integer: logit, then id (argmax's tie rule).
fn key64(v: f32, id: usize) -> u64 {
    (u64::from(key(v)) << 32) | id as u64
}

fn cand(k: u64) -> Cand {
    Cand {
        logit: unkey((k >> 32) as u32),
        id: k as u32,
    }
}

/// The distinct ids the repeat penalty applies to after `recent` (the
/// tokens so far, the last one included): its last `last_n`, sorted.
pub fn penalty_ids(recent: &[u32], last_n: usize) -> Vec<u32> {
    let mut w = recent[recent.len().saturating_sub(last_n)..].to_vec();
    w.sort_unstable();
    w.dedup();
    w
}

/// llama.cpp's repeat penalty on each of `ids` (distinct): a positive
/// logit divided by `p`, any other multiplied by it.
pub fn penalize(l: &mut [f32], ids: &[u32], p: f32) {
    if p == 1.0 {
        return;
    }
    for &id in ids {
        let v = &mut l[id as usize];
        *v = if *v > 0.0 { *v / p } else { *v * p };
    }
}

/// A position's candidates from its logits (penalized in place): `spec.k`
/// best, best first. `recent` ends with the token the logits follow.
pub fn cands(l: &mut [f32], spec: &CandSpec, recent: &[u32]) -> Vec<Cand> {
    if spec.penalty != 1.0 && spec.last_n > 0 {
        penalize(l, &penalty_ids(recent, spec.last_n), spec.penalty);
    }
    top_k(l, spec.k)
}

/// The `k` best of `l` (fewer if it's shorter), best first: by logit, then
/// by the larger id; NaN last. Exact, by sorting every key: the reference
/// for `top_k`, and its fallback.
pub fn top_k_ref(l: &[f32], k: usize) -> Vec<Cand> {
    let mut keys: Vec<u64> = l.iter().enumerate().map(|(i, &v)| key64(v, i)).collect();
    keys.sort_unstable_by(|a, b| b.cmp(a));
    keys.truncate(k);
    keys.into_iter().map(cand).collect()
}

/// Keep the `k` largest keys of `buf` (unordered); the k-th largest's logit.
fn keep_top(buf: &mut Vec<u64>, k: usize) -> f32 {
    buf.select_nth_unstable_by(k - 1, |a, b| b.cmp(a));
    buf.truncate(k);
    unkey((buf[k - 1] >> 32) as u32)
}

/// `top_k_ref`, in about the time of one `argmax`: a scan that tests 8
/// logits at a time against the k-th best so far and only then pushes
/// (key, id)s into a small buffer, cut back to k when it fills up. Logits
/// equal to the threshold are pushed (a later id wins the tie). NaNs are
/// skipped; if fewer than k logits aren't NaN, the exact sort decides.
pub fn top_k(l: &[f32], k: usize) -> Vec<Cand> {
    let k = k.min(l.len());
    if k == 0 {
        return Vec::new();
    }
    if k >= l.len() / 8 {
        return top_k_ref(l, k); // most of the row: sort it
    }
    let mut buf = Vec::with_capacity(4 * k + 16);
    top_k_scan(l, k, &mut buf);
    if buf.len() < k {
        return top_k_ref(l, k);
    }
    keep_top(&mut buf, k);
    buf.sort_unstable_by(|a, b| b.cmp(a));
    buf.into_iter().map(cand).collect()
}

multiversion! {
    /// `top_k`'s scan: buf ends up holding the k best non-NaN logits' keys
    /// (and maybe more).
    fn top_k_scan<V3>(l: &[f32], k: usize, buf: &mut Vec<u64>) {
        let cap = 4 * k + 8;
        let mut thr = f32::NEG_INFINITY;
        let (chunks, tail) = l.as_chunks::<8>();
        for (c, ch) in chunks.iter().enumerate() {
            let mut hit = false;
            for &v in ch {
                hit |= v >= thr;
            }
            if hit {
                for (j, &v) in ch.iter().enumerate() {
                    if v >= thr {
                        buf.push(key64(v, c * 8 + j));
                    }
                }
                if buf.len() >= cap {
                    thr = keep_top(buf, k);
                }
            }
        }
        let base = chunks.len() * 8;
        for (j, &v) in tail.iter().enumerate() {
            if v >= thr {
                buf.push(key64(v, base + j));
            }
        }
    }
}

/// How many of `c` (a position's candidates, best first) survive top-p then
/// min-p: at least 1. NaN candidates (last) never do.
fn kept(cfg: &SampleCfg, c: &[Cand]) -> usize {
    let mut n = c.iter().take_while(|x| !x.logit.is_nan()).count();
    let l0 = f64::from(c[0].logit);
    if n <= 1 || !l0.is_finite() {
        return 1;
    }
    if cfg.top_p < 1.0 {
        // llama.cpp: the smallest prefix whose softmax (temperature 1) mass
        // reaches top_p.
        n = with_scratch(n, |e| {
            let mut z = 0.0;
            for (ei, x) in e.iter_mut().zip(c) {
                *ei = (f64::from(x.logit) - l0).exp();
                z += *ei;
            }
            let mut cum = 0.0;
            for (i, ei) in e.iter().enumerate() {
                cum += ei / z;
                if cum >= f64::from(cfg.top_p) {
                    return i + 1;
                }
            }
            e.len()
        });
    }
    if cfg.min_p > 0.0 {
        // p_i >= min_p * p_max, on logits: l_i >= l_0 + ln(min_p).
        let min = l0 + f64::from(cfg.min_p).ln();
        n = c[..n]
            .iter()
            .take_while(|x| f64::from(x.logit) >= min)
            .count()
            .max(1);
    }
    n
}

/// The distribution `pick` draws from: (id, probability) of each kept
/// candidate, in candidate order.
#[cfg(test)]
pub fn dist(cfg: &SampleCfg, c: &[Cand]) -> Vec<(u32, f64)> {
    let n = kept(cfg, c);
    let l0 = f64::from(c[0].logit);
    let t = f64::from(cfg.temp);
    let w: Vec<f64> = if n == 1 {
        vec![1.0]
    } else {
        c[..n]
            .iter()
            .map(|x| ((f64::from(x.logit) - l0) / t).exp())
            .collect()
    };
    let z: f64 = w.iter().sum();
    c[..n].iter().zip(w).map(|(x, w)| (x.id, w / z)).collect()
}

/// The token `u` (uniform in [0, 1)) picks from the candidates `c` (best
/// first, non-empty): top-p, min-p, temperature, then the inverse CDF.
pub fn pick(cfg: &SampleCfg, c: &[Cand], u: f64) -> u32 {
    let n = kept(cfg, c);
    if n == 1 {
        return c[0].id;
    }
    let l0 = f64::from(c[0].logit);
    let t = f64::from(cfg.temp);
    with_scratch(n, |w| {
        for (wi, x) in w.iter_mut().zip(c) {
            *wi = ((f64::from(x.logit) - l0) / t).exp();
        }
        c[draw(w, u)].id
    })
}

/// The inverse CDF of weights `w` (w[0] > 0) at `u` in [0, 1): the first i
/// whose cumulative weight passes u * total. Never a zero-weight index:
/// the target stays below the total, and past the end (u out of range)
/// the last positive weight is taken.
fn draw(w: &[f64], u: f64) -> usize {
    let z: f64 = w.iter().sum();
    let target = (u * z).min(z.next_down());
    let (mut cum, mut last) = (0.0, 0);
    for (i, &wi) in w.iter().enumerate() {
        if wi > 0.0 {
            last = i;
        }
        cum += wi;
        if target < cum {
            return i;
        }
    }
    last
}

/// `f` on n f64s of scratch: on the stack for up to CAND_MAX (top_k on).
fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
    if n <= CAND_MAX {
        f(&mut [0f64; CAND_MAX][..n])
    } else {
        f(&mut vec![0f64; n])
    }
}

/// `crate::generate` when sampling: the same loop, with each position's
/// candidates in place of its greedy pick (one `forward_cands` per token,
/// or one `verify_cands` per draft) and `s.pick` choosing from them, lazily,
/// in token order. A drafted token is kept iff the pick at its position is
/// that token (see the module doc: lossless, and the same output as
/// without drafting for the same seed).
#[allow(clippy::too_many_arguments)]
pub fn generate<M: Lm>(
    net: &M,
    cache: &mut M::Cache,
    prompt: &[u32],
    mut logits: Vec<f32>,
    n_gen: usize,
    stop: &[u32],
    max_draft: usize,
    s: &mut Sampler,
    draft: &mut dyn FnMut(&[u32], usize) -> Vec<u32>,
    emit: &mut dyn FnMut(u32) -> io::Result<()>,
) -> io::Result<GenStats> {
    let spec = s.cfg.spec();
    let mut st = GenStats::default();
    let mut hist = prompt.to_vec();
    // (guess, the candidates after it) from the last verify pass, in order.
    let mut queue: VecDeque<(u32, Vec<Cand>)> = VecDeque::new();
    let mut cur = cands(&mut logits, &spec, &hist);
    drop(logits);
    let mut picks = 0usize;
    while st.ids.len() < n_gen {
        let next = s.pick(&cur);
        picks += 1;
        if stop.contains(&next) {
            break;
        }
        st.ids.push(next);
        hist.push(next);
        emit(next)?;
        let left = n_gen - st.ids.len();
        if left == 0 {
            break;
        }
        match queue.pop_front() {
            Some((guess, c)) if guess == next => {
                cur = c;
                st.accepted += 1;
                continue;
            }
            _ => queue.clear(),
        }
        let max = max_draft.min(left - 1);
        let mut guesses = if max > 0 {
            draft(&hist, max)
        } else {
            Vec::new()
        };
        guesses.truncate(max);
        let pos = hist.len() - 1;
        let t = Instant::now();
        if guesses.is_empty() {
            cur = net.forward_cands(next, pos, cache, &spec, &hist);
            st.forward.0 += 1;
            st.forward.1 += t.elapsed().as_secs_f64();
        } else {
            st.drafted += guesses.len();
            let mut toks = vec![next];
            toks.extend_from_slice(&guesses);
            // Each row's penalty window: the history, then the guesses before
            // it. At least `next` (verify_cands wants recent to end with toks,
            // also when the window is off).
            let tail = spec.last_n.clamp(1, hist.len());
            let mut recent = hist[hist.len() - tail..].to_vec();
            recent.extend_from_slice(&guesses);
            let mut rows = net
                .verify_cands(&toks, pos, cache, &spec, &recent)
                .into_iter();
            cur = rows.next().expect("verify_cands: one row per token");
            queue.extend(guesses.into_iter().zip(rows));
            if st.verify.len() <= toks.len() {
                st.verify.resize(toks.len() + 1, (0, 0.0));
            }
            let v = &mut st.verify[toks.len()];
            v.0 += 1;
            v.1 += t.elapsed().as_secs_f64();
        }
        st.passes += 1;
    }
    st.decided = picks.saturating_sub(1);
    Ok(st)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::argmax;

    fn xorshift(s: &mut u32) -> u32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    }

    fn cfg() -> SampleCfg {
        SampleArgs {
            seed: Some(7),
            ..Default::default()
        }
        .resolve(DEFAULT_TEMP)
        .unwrap()
    }

    fn cs(l: &[f32]) -> Vec<Cand> {
        top_k_ref(l, l.len())
    }

    fn ids(c: &[Cand]) -> Vec<u32> {
        c.iter().map(|c| c.id).collect()
    }

    /// Bits and ids, so NaN and -0 compare exactly.
    fn bits(c: &[Cand]) -> Vec<(u32, u32)> {
        c.iter().map(|c| (c.logit.to_bits(), c.id)).collect()
    }

    #[test]
    fn defaults_are_ollamas() {
        let c = cfg();
        assert_eq!(
            (
                c.temp,
                c.top_k,
                c.top_p,
                c.min_p,
                c.repeat_penalty,
                c.repeat_last_n
            ),
            (0.8, 40, 0.9, 0.0, 1.1, 64)
        );
        assert_eq!(SampleArgs::default().resolve(0.0), None, "gen: greedy");
        let t0 = SampleArgs {
            temp: Some(0.0),
            ..Default::default()
        };
        assert_eq!(t0.resolve(DEFAULT_TEMP), None, "--temp 0: greedy");
        assert_ne!(SampleArgs::default().resolve(0.8).unwrap().seed, 0);
    }

    #[test]
    fn flags_parse_and_reject() {
        let mut a = SampleArgs::default();
        for (f, v) in [
            ("--temp", "0.5"),
            ("--top-k", "0"),
            ("--top-p", "1"),
            ("--min-p", "0.05"),
            ("--repeat-penalty", "1.3"),
            ("--repeat-last-n", "0"),
            ("--seed", "18446744073709551615"),
        ] {
            assert!(SampleArgs::is_flag(f));
            a.set(f, Some(v)).unwrap();
        }
        let c = a.resolve(0.0).unwrap();
        assert_eq!(c.seed, u64::MAX);
        assert_eq!(c.spec().k, ALL, "top-k 0 = off = every token");
        for (f, v) in [
            ("--temp", "-1"),
            ("--temp", "nan"),
            ("--top-p", "1.5"),
            ("--min-p", "x"),
            ("--repeat-penalty", "0"),
            ("--top-k", "-3"),
            ("--seed", "1.5"),
        ] {
            assert!(a.set(f, Some(v)).is_err(), "{f} {v}");
        }
        assert!(a.set("--temp", None).is_err());
        assert!(!SampleArgs::is_flag("--draft"));
        let big = SampleArgs {
            top_k: Some(100_000),
            temp: Some(1.0),
            ..Default::default()
        };
        assert_eq!(big.resolve(0.0).unwrap().spec().k, 100_000);
    }

    #[test]
    fn key_orders_like_argmax() {
        let vals = [
            f32::NEG_INFINITY,
            -1e30,
            -1.0,
            -1e-40,
            -0.0,
            0.0,
            1e-40,
            1.0,
            3.5,
            f32::INFINITY,
        ];
        for w in vals.windows(2) {
            if w[0] == w[1] {
                assert_eq!(key(w[0]), key(w[1]), "-0 ties +0");
            } else {
                assert!(key(w[0]) < key(w[1]), "{} < {}", w[0], w[1]);
            }
        }
        assert!(key(f32::NAN) < key(f32::NEG_INFINITY));
        assert_eq!(key(-f32::NAN), 0);
        for v in vals {
            let back = unkey(key(v));
            assert_eq!(back, v);
            if v != 0.0 {
                assert_eq!(back.to_bits(), v.to_bits());
            }
        }
        assert_eq!(unkey(key(-0.0)).to_bits(), 0, "-0 comes back +0");
        assert!(unkey(0).is_nan());
    }

    /// The fast scan equals the full sort: random, tied, NaN-ridden,
    /// -inf, signed zeros, k past the vocab; and its first candidate is
    /// argmax's pick (ties to the last index).
    #[test]
    fn top_k_matches_reference() {
        let mut s = 0x1234_5679u32;
        let specials = [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.0, 0.0];
        for n in [1usize, 2, 7, 8, 9, 40, 41, 63, 64, 65, 300, 2049, 151_936] {
            for round in 0..12 {
                let l: Vec<f32> = (0..n)
                    .map(|_| {
                        let r = xorshift(&mut s);
                        match round % 4 {
                            0 => r as f32 / 1e7 - 200.0,
                            1 => (r % 7) as f32 - 3.0, // heavy ties
                            2 if r.is_multiple_of(13) => specials[(r >> 8) as usize % 5],
                            2 => (r % 1000) as f32 / 37.0,
                            _ if r.is_multiple_of(3) => f32::NAN, // few non-NaN
                            _ => (r % 5) as f32,
                        }
                    })
                    .collect();
                for k in [1, 2, 3, 40, 256, n, n + 5] {
                    let got = top_k(&l, k);
                    assert_eq!(bits(&got), bits(&top_k_ref(&l, k)), "n {n} k {k} r {round}");
                    assert_eq!(got.len(), k.min(n));
                }
                assert_eq!(top_k(&l, 1)[0].id, argmax(&l), "n {n} r {round}");
            }
        }
        // All NaN: argmax's n - 1.
        assert_eq!(top_k(&[f32::NAN; 9], 3)[0].id, argmax(&[f32::NAN; 9]));
        // Hand-checked ties: equal logits, larger id first.
        let l = [1.0, 3.0, 2.0, 3.0, 1.0, 3.0];
        assert_eq!(ids(&top_k(&l, 4)), [5, 3, 1, 2]);
        assert_eq!(ids(&top_k(&l, 10)), [5, 3, 1, 2, 4, 0]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn top_k_multiversion_identical() {
        if !crate::model::tests::can_v3() {
            return;
        }
        let mut s = 99u32;
        let l: Vec<f32> = (0..151_936)
            .map(|_| (xorshift(&mut s) % 4000) as f32 / 100.0)
            .collect();
        for k in [1, 40, 256] {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            top_k_scan::portable(&l, k, &mut a);
            unsafe { top_k_scan::v3(&l, k, &mut b) };
            assert_eq!(a, b, "k {k}");
        }
    }

    #[test]
    fn repeat_penalty() {
        // Positive divided, negative (and zero) multiplied, each id once
        // however often it repeats, only the last `last_n` tokens.
        let spec = CandSpec {
            k: 6,
            penalty: 2.0,
            last_n: 4,
        };
        let mut l = [4.0, -1.0, 0.0, 3.0, 1.0, 2.0];
        let recent = [3, 5, 1, 1, 2, 1]; // window: [1, 2, 1] + ..: last 4 = [1, 1, 2, 1]
        let c = cands(&mut l, &spec, &recent);
        assert_eq!(l, [4.0, -2.0, 0.0, 3.0, 1.0, 2.0]);
        assert_eq!(ids(&c), [0, 3, 5, 4, 2, 1]);
        assert_eq!(penalty_ids(&recent, 4), [1, 2]);
        assert_eq!(penalty_ids(&recent, 100), [1, 2, 3, 5]);
        assert_eq!(penalty_ids(&recent, 0), Vec::<u32>::new());
        // 1 and last_n 0 are off.
        for (p, n) in [(1.0, 4), (2.0, 0)] {
            let mut l2 = [4.0, -1.0, 0.0, 3.0, 1.0, 2.0];
            let s = CandSpec {
                k: 6,
                penalty: p,
                last_n: n,
            };
            cands(&mut l2, &s, &recent);
            assert_eq!(l2, [4.0, -1.0, 0.0, 3.0, 1.0, 2.0]);
        }
        // Below 1 it boosts: 4.0 / 0.5, 1.0 / 0.5.
        let mut l3 = [4.0, 1.0, 0.0, 3.0, 1.0, 2.0];
        penalize(&mut l3, &[1, 0], 0.5);
        assert_eq!(l3, [8.0, 2.0, 0.0, 3.0, 1.0, 2.0]);
        // The penalty changes the order: 3.0 / 1.1 < 2.8.
        let mut l4 = [3.0, 2.8];
        let s = CandSpec {
            k: 2,
            penalty: 1.1,
            last_n: 64,
        };
        assert_eq!(ids(&cands(&mut l4, &s, &[0])), [1, 0]);
        assert_eq!(l4[0], 3.0f32 / 1.1f32);
    }

    /// Probabilities 0.5, 0.3, 0.2 (as logits): top-p, min-p, top-k, temp
    /// by hand.
    #[test]
    fn filters_by_hand() {
        let l = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        let c = cs(&l);
        let base = SampleCfg {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: 0,
        };
        let kept = |cfg: SampleCfg| dist(&cfg, &c).len();
        let probs = |cfg: SampleCfg| -> Vec<f64> { dist(&cfg, &c).iter().map(|x| x.1).collect() };
        let close = |a: &[f64], b: &[f64]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
        };
        assert!(close(&probs(base), &[0.5, 0.3, 0.2]));
        for (p, n) in [
            (0.0, 1),
            (0.49, 1),
            (0.51, 2),
            (0.79, 2),
            (0.81, 3),
            (1.0, 3),
        ] {
            assert_eq!(kept(SampleCfg { top_p: p, ..base }), n, "top_p {p}");
        }
        assert!(close(
            &probs(SampleCfg { top_p: 0.6, ..base }),
            &[0.625, 0.375]
        ));
        // min_p: p_i >= min_p * 0.5.
        for (m, n) in [
            (0.0, 3),
            (0.39, 3),
            (0.41, 2),
            (0.59, 2),
            (0.61, 1),
            (1.0, 1),
        ] {
            assert_eq!(kept(SampleCfg { min_p: m, ..base }), n, "min_p {m}");
        }
        // top_p then min_p: 0.81 keeps 3, min_p 0.5 then 2.
        assert_eq!(
            kept(SampleCfg {
                top_p: 0.81,
                min_p: 0.5,
                ..base
            }),
            2
        );
        // top-k first: with k = 2, top-p sees 0.625 / 0.375.
        let c2 = top_k(&l, SampleCfg { top_k: 2, ..base }.spec().k);
        assert_eq!(dist(&SampleCfg { top_p: 0.6, ..base }, &c2).len(), 1);
        assert_eq!(
            dist(
                &SampleCfg {
                    top_p: 0.63,
                    ..base
                },
                &c2
            )
            .len(),
            2
        );
        // Temperature after the cut: probs ∝ p^(1/T).
        for t in [0.25f32, 0.5, 2.0, 7.0] {
            let w: Vec<f64> = [0.5f64, 0.3, 0.2]
                .iter()
                .map(|p| p.powf(1.0 / t as f64))
                .collect();
            let z: f64 = w.iter().sum();
            let want: Vec<f64> = w.iter().map(|x| x / z).collect();
            assert!(
                close(&probs(SampleCfg { temp: t, ..base }), &want),
                "temp {t}"
            );
        }
        // Inverse CDF over u.
        let cfg = base;
        for (u, id) in [
            (0.0, 0),
            (0.499, 0),
            (0.501, 1),
            (0.799, 1),
            (0.801, 2),
            (0.9999, 2),
        ] {
            assert_eq!(pick(&cfg, &c, u), id, "u {u}");
        }
        // Degenerate rows: one candidate, -inf / NaN tails, -inf top.
        assert_eq!(pick(&cfg, &cs(&[1.0]), 0.7), 0);
        let c3 = cs(&[f32::NEG_INFINITY, 2.0, f32::NAN]);
        assert_eq!(dist(&cfg, &c3), vec![(1, 1.0), (0, 0.0)]);
        assert_eq!(pick(&cfg, &c3, 0.9999), 1);
        assert_eq!(pick(&cfg, &cs(&[f32::NEG_INFINITY; 3]), 0.5), 2);
        assert_eq!(pick(&cfg, &cs(&[f32::NAN; 3]), 0.5), 2);
    }

    /// top_k 0 is the whole vocab, as llama.cpp: candidates are every
    /// token, top-p's softmax and the draw are over all of them.
    #[test]
    fn top_k_off_is_whole_vocab() {
        let mut s = 77u32;
        let n = 3000;
        let l: Vec<f32> = (0..n)
            .map(|_| (xorshift(&mut s) % 400) as f32 / 100.0)
            .collect();
        let cfg = SampleCfg {
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            temp: 1.0,
            repeat_penalty: 1.0,
            ..cfg()
        };
        let mut l2 = l.clone();
        let c = cands(&mut l2, &cfg.spec(), &[0]);
        assert_eq!(bits(&c), bits(&top_k_ref(&l, n)));
        let d = dist(&cfg, &c);
        assert_eq!(d.len(), n);
        let m = l.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let z: f64 = l.iter().map(|&x| (x as f64 - m).exp()).sum();
        for &(id, p) in &d {
            let want = (l[id as usize] as f64 - m).exp() / z;
            assert!((p - want).abs() < 1e-12, "id {id}");
        }
        // top-p over the whole vocab's mass: the kept prefix is the
        // smallest whose full-softmax mass reaches 0.5.
        let half = dist(&SampleCfg { top_p: 0.5, ..cfg }, &c).len();
        let mut cum = 0.0;
        let want = c
            .iter()
            .position(|x| {
                cum += (x.logit as f64 - m).exp() / z;
                cum >= 0.5
            })
            .unwrap()
            + 1;
        assert_eq!(half, want);
        assert!(
            half > CAND_MAX,
            "this row's 50% needs more than {CAND_MAX} tokens"
        );
        // top_k past CAND_MAX is exact too.
        let c1000 = top_k(&l, 1000);
        assert_eq!(bits(&c1000), bits(&top_k_ref(&l, 1000)));
    }

    /// `draw` never returns a zero-weight index, even at the edge of [0, 1)
    /// or past it.
    #[test]
    fn draw_skips_zero_weights() {
        let w = [1.0, 0.7, 0.3, 0.0, 0.0];
        assert_eq!(draw(&w, 0.0), 0);
        assert_eq!(draw(&w, 1.0 - f64::EPSILON / 2.0), 2);
        assert_eq!(draw(&w, 1.0), 2);
        assert_eq!(draw(&w, 7.0), 2);
        assert_eq!(draw(&[1.0, 0.0], 1.0), 0);
        let c = cs(&[3.0, 1.0, f32::NEG_INFINITY]);
        let cfg = SampleCfg {
            top_p: 1.0,
            min_p: 0.0,
            ..cfg()
        };
        assert_ne!(pick(&cfg, &c, 1.0 - f64::EPSILON / 2.0), 2);
    }

    /// Draws follow `dist`: chi-square over 200k draws.
    #[test]
    fn draws_follow_dist() {
        let l: Vec<f32> = (0..30).map(|i| ((i * 7919) % 31) as f32 / 9.0).collect();
        let cfg = SampleCfg { top_k: 20, ..cfg() };
        let c = top_k(&l, cfg.spec().k);
        let d = dist(&cfg, &c);
        let mut s = Sampler::new(cfg);
        let n = 200_000;
        let mut count = std::collections::HashMap::new();
        for _ in 0..n {
            *count.entry(s.pick(&c)).or_insert(0usize) += 1;
        }
        assert!(count.keys().all(|id| d.iter().any(|x| x.0 == *id)));
        let chi: f64 = d
            .iter()
            .map(|&(id, p)| {
                let e = p * n as f64;
                let o = *count.get(&id).unwrap_or(&0) as f64;
                (o - e).powi(2) / e
            })
            .sum();
        let df = (d.len() - 1) as f64;
        assert!(chi < df + 6.0 * (2.0 * df).sqrt(), "chi2 {chi} df {df}");
    }

    #[test]
    fn rng_is_seeded() {
        let a: Vec<u64> = (0..5)
            .scan(Rng::new(3), |r, _| Some(r.next_u64()))
            .collect();
        let b: Vec<u64> = (0..5)
            .scan(Rng::new(3), |r, _| Some(r.next_u64()))
            .collect();
        let c: Vec<u64> = (0..5)
            .scan(Rng::new(4), |r, _| Some(r.next_u64()))
            .collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut r = Rng::new(1);
        let m: f64 = (0..100_000).map(|_| r.f64()).sum::<f64>() / 1e5;
        assert!((m - 0.5).abs() < 0.01);
    }

    /// The inverse-CDF draw over a grid of u: each token takes a share
    /// p(token) of [0, 1), and so, given the draw isn't x, a share p(y) / (1 -
    /// p(x)) for each other y. (That's the draw alone; the coupling itself,
    /// draft vs no draft, is checked by drafting_matches_plain_sampling and
    /// drafting_distribution_chi_square.)
    #[test]
    fn inverse_cdf_shares() {
        let l = [1.0f32, 0.2, -0.5, 0.9, 0.0];
        let cfg = SampleCfg {
            top_p: 0.95,
            ..cfg()
        };
        let c = top_k(&l, 5);
        let d = dist(&cfg, &c);
        let n = 1_000_000;
        for x in 0..5u32 {
            let px = d.iter().find(|e| e.0 == x).map_or(0.0, |e| e.1);
            let mut acc = 0usize;
            let mut rej = [0usize; 5];
            for i in 0..n {
                let t = pick(&cfg, &c, (i as f64 + 0.5) / n as f64);
                if t == x {
                    acc += 1;
                } else {
                    rej[t as usize] += 1;
                }
            }
            assert!((acc as f64 / n as f64 - px).abs() < 1e-5, "x {x}");
            for &(y, py) in &d {
                if y != x {
                    let got = rej[y as usize] as f64 / (n - acc) as f64;
                    assert!((got - py / (1.0 - px)).abs() < 1e-5, "x {x} y {y}");
                }
            }
        }
    }

    /// A weightless `Lm` with soft logits: a hash of the token, its
    /// position and the token cached before it (so a misused cache shows).
    struct Soft;

    impl Soft {
        fn logits(t: u32, pos: usize, c: &mut [u32]) -> Vec<f32> {
            c[pos] = t;
            let prev = if pos > 0 { c[pos - 1] } else { 5 };
            (0..8u32)
                .map(|j| {
                    let mut h = t.wrapping_mul(2654435761)
                        ^ (pos as u32).wrapping_mul(40503)
                        ^ prev.wrapping_mul(97)
                        ^ j.wrapping_mul(7919)
                        | 1;
                    xorshift(&mut h);
                    (h % 1000) as f32 / 300.0
                })
                .collect()
        }
    }

    impl crate::lm::Sealed for Soft {}
    impl Lm for Soft {
        type Cache = Vec<u32>;
        fn vocab(&self) -> usize {
            8
        }
        fn new_cache(&self, max_ctx: usize) -> Vec<u32> {
            vec![u32::MAX; max_ctx]
        }
        fn forward(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> Vec<f32> {
            Soft::logits(t, pos, c)
        }
        fn feed_batch(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) {
            c[start..start + ts.len()].copy_from_slice(ts);
        }
        fn verify_argmax(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) -> Vec<u32> {
            (0..ts.len())
                .map(|b| argmax(&Soft::logits(ts[b], start + b, c)))
                .collect()
        }
    }

    /// `generate` sampled over `net` after `prompt`, drafting by `mode`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run<M: Lm>(
        net: &M,
        prompt: &[u32],
        n: usize,
        stop: &[u32],
        k: usize,
        cfg: SampleCfg,
        mode: &str,
        oracle: &[u32],
    ) -> GenStats {
        let mut cache = net.new_cache(prompt.len() + n + 1);
        let (&last, rest) = prompt.split_last().unwrap();
        if !rest.is_empty() {
            net.feed_batch(rest, 0, &mut cache);
        }
        let logits = net.forward(last, rest.len(), &mut cache);
        let mut s = Sampler::new(cfg);
        let mut rng = 0x2545_f491u32 ^ cfg.seed as u32;
        let mut lookup = crate::spec::PromptLookup::new();
        let mut draft = |h: &[u32], max: usize| -> Vec<u32> {
            let at = h.len() - prompt.len();
            let right = |i: usize| oracle.get(at + i).copied().unwrap_or(1);
            match mode {
                "oracle" => (0..max).map(right).collect(),
                "wrong" => (0..max).map(|i| right(i) ^ 1).collect(),
                "random" => (0..max)
                    .map(|_| xorshift(&mut rng) % net.vocab() as u32)
                    .collect(),
                _ => {
                    lookup.sync(h);
                    lookup.draft(max)
                }
            }
        };
        let mut emitted = Vec::new();
        let st = generate(
            net,
            &mut cache,
            prompt,
            logits,
            n,
            stop,
            k,
            &mut s,
            &mut draft,
            &mut |t| {
                emitted.push(t);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(emitted, st.ids);
        st
    }

    /// Speculative sampling is lossless, and more: for every seed, drafts
    /// that are always right, always wrong, random or real prompt lookup
    /// give the very tokens plain sampling gives.
    #[test]
    fn drafting_matches_plain_sampling() {
        let prompt = [1u32, 2, 3, 1, 2];
        for seed in 0..300u64 {
            let cfg = SampleCfg {
                seed,
                top_k: [40, 5, 0][seed as usize % 3],
                min_p: [0.0, 0.05][seed as usize % 2],
                temp: [0.8, 1.5, 0.3][seed as usize % 3],
                // Penalty window off, penalty off, a 1-token window, default.
                repeat_last_n: [0, 64, 1, 64, 0][seed as usize % 5],
                repeat_penalty: [1.1, 1.0, 1.3, 1.1, 1.0][seed as usize % 5],
                ..cfg()
            };
            let stop: &[u32] = if seed % 4 == 0 { &[6] } else { &[] };
            let plain = run(&Soft, &prompt, 24, stop, 0, cfg, "oracle", &[]);
            assert_eq!(plain.drafted, 0);
            for k in [1, 3, 7] {
                for mode in ["oracle", "wrong", "random", "lookup"] {
                    let st = run(&Soft, &prompt, 24, stop, k, cfg, mode, &plain.ids);
                    let ctx = format!("seed {seed} k {k} {mode}");
                    assert_eq!(st.ids, plain.ids, "{ctx}");
                    assert_eq!(st.decided, plain.decided, "{ctx}");
                    assert!(st.accepted <= st.drafted, "{ctx}");
                    if mode == "wrong" {
                        assert_eq!(st.accepted, 0, "{ctx}");
                    }
                    if mode == "oracle" && plain.ids.len() == 24 {
                        assert!(st.passes < plain.passes, "{ctx}");
                    }
                }
            }
        }
    }

    /// Chi-square of the first two tokens over 30k seeds, drafting (one
    /// random guess for the second) against their exact probability from
    /// `dist`.
    #[test]
    fn drafting_distribution_chi_square() {
        let prompt = [3u32, 4];
        let cfg0 = SampleCfg {
            top_k: 6,
            temp: 1.2,
            ..cfg()
        };
        let spec = cfg0.spec();
        // Exact P(t1, t2).
        let mut c = Soft.new_cache(8);
        c[0] = 3;
        let mut l = Soft::logits(4, 1, &mut c);
        let d1 = dist(&cfg0, &cands(&mut l, &spec, &prompt));
        let mut exact = std::collections::HashMap::new();
        for &(t1, p1) in &d1 {
            let mut c = Soft.new_cache(8);
            c[0] = 3;
            c[1] = 4;
            let mut l = Soft::logits(t1, 2, &mut c);
            let d2 = dist(&cfg0, &cands(&mut l, &spec, &[3, 4, t1]));
            for &(t2, p2) in &d2 {
                exact.insert((t1, t2), p1 * p2);
            }
        }
        let n = 30_000u64;
        let mut count = std::collections::HashMap::new();
        for seed in 0..n {
            let cfg = SampleCfg { seed, ..cfg0 };
            let st = run(&Soft, &prompt, 3, &[], 3, cfg, "random", &[]);
            assert_eq!(st.drafted, 1);
            *count.entry((st.ids[0], st.ids[1])).or_insert(0u64) += 1;
        }
        assert!(count.keys().all(|k| exact.contains_key(k)));
        let mut chi = 0.0;
        for (key, p) in &exact {
            let e = p * n as f64;
            let o = *count.get(key).unwrap_or(&0) as f64;
            chi += (o - e).powi(2) / e;
        }
        let df = (exact.len() - 1) as f64;
        assert!(chi < df + 6.0 * (2.0 * df).sqrt(), "chi2 {chi} df {df}");
    }

    /// The real model on the CPU: drafting gives plain sampling's tokens,
    /// the same seed the same tokens, and a different seed other tokens.
    #[test]
    fn real_model_sampling_cpu() {
        let Some(m) = crate::testutil::model() else {
            return;
        };
        let net = crate::model::Qwen2::new(&m);
        real_model_sampling_on(&net, &m);
    }

    /// `real_model_sampling_cpu` on qwen3.
    #[test]
    fn real_model_sampling_cpu_qwen3() {
        let Some(m) = crate::testutil::qwen3_model() else {
            return;
        };
        let net = crate::model::Qwen2::new(&m);
        real_model_sampling_on(&net, &m);
    }

    /// `real_model_sampling_cpu` on llama.
    #[test]
    fn real_model_sampling_cpu_llama() {
        let Some(m) = crate::testutil::llama_model() else {
            return;
        };
        let net = crate::model::Qwen2::new(&m);
        real_model_sampling_on(&net, &m);
    }

    pub(crate) fn real_model_sampling_on<M: Lm>(net: &M, m: &crate::gguf::Model) {
        let tok = crate::tokenizer::Tokenizer::from_gguf(m);
        let prompts = [
            "fn add(a: i32, b: i32) -> i32 { a + b }\nfn sub(a: i32, b: i32) -> i32 {",
            "The quick brown fox jumps over the lazy dog. The quick brown fox",
        ];
        for p in prompts {
            let ids = tok.encode(p);
            for seed in [1u64, 2, 3] {
                let cfg = SampleCfg { seed, ..cfg() };
                let plain = run(net, &ids, 24, &[tok.eos()], 0, cfg, "", &[]);
                let again = run(net, &ids, 24, &[tok.eos()], 0, cfg, "", &[]);
                assert_eq!(plain.ids, again.ids, "seed {seed}: deterministic");
                for k in [3, 7] {
                    for mode in ["lookup", "oracle", "wrong"] {
                        let st = run(net, &ids, 24, &[tok.eos()], k, cfg, mode, &plain.ids);
                        assert_eq!(st.ids, plain.ids, "{p:?} seed {seed} k {k} {mode}");
                    }
                }
            }
            let hot = SampleCfg {
                temp: 1.5,
                top_k: 0,
                top_p: 1.0,
                ..cfg()
            };
            let a = run(net, &ids, 16, &[], 0, SampleCfg { seed: 1, ..hot }, "", &[]);
            let b = run(net, &ids, 16, &[], 0, SampleCfg { seed: 2, ..hot }, "", &[]);
            assert_ne!(a.ids, b.ids, "{p:?}: seeds 1 and 2 at temp 1.5");
        }
    }

    /// Time of `top_k` (k 40, 256) next to `argmax` on a vocab of logits.
    /// `--ignored --nocapture` to see.
    #[test]
    #[ignore]
    fn top_k_speed() {
        let mut s = 5u32;
        let l: Vec<f32> = (0..151_936)
            .map(|_| xorshift(&mut s) as f32 / 4e8 - 5.0)
            .collect();
        let time = |name: &str, f: &dyn Fn() -> u32| {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let mut x = 0;
                for _ in 0..200 {
                    x ^= f();
                }
                std::hint::black_box(x);
                best = best.min(t.elapsed().as_secs_f64() * 1e6 / 200.0);
            }
            eprintln!("{name}: {best:.1} us");
        };
        time("argmax", &|| argmax(std::hint::black_box(&l)));
        time("top_k 40", &|| top_k(std::hint::black_box(&l), 40)[0].id);
        time("top_k 256", &|| top_k(std::hint::black_box(&l), 256)[0].id);
        time("full sort (top_k off)", &|| {
            top_k_ref(std::hint::black_box(&l), l.len())[0].id
        });
        time("full exp sum f64", &|| {
            let m = l.iter().cloned().fold(f32::MIN, f32::max) as f64;
            l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>() as u32
        });
        let spec = cfg().spec();
        let recent: Vec<u32> = (0..64).map(|i| i * 1000).collect();
        time("cands (penalty, k 40, incl. copy)", &|| {
            let mut c = l.clone();
            cands(&mut c, &spec, &recent)[0].id
        });
        time("copy only", &|| l.clone()[7] as u32);
    }
}
