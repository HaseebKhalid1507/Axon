//! Speculative decoding in the batch (specs/r14-batched-spec.md, stage 1,
//! greedy): a draft model's pool beside the main model's, slot s mirroring
//! slot s, and the goodput gate that picks each round's guesses per request.
//!
//! A round (`Batch::step` with a `DraftPool`): the gate gives each running
//! request K_i guesses (0..=K; 0 for sampled requests, whose rows are as
//! without drafting); the drafter runs max K_i steps of one row per request
//! still guessing (its first step also catches the request's history up:
//! the main model's correction token, or the last guess of a fully accepted
//! round); then one main-model step verifies, per request, its last token
//! and its guesses at their positions (`BatchLm::step` takes a slot's rows
//! in ascending positions: each row's pick is `verify_argmax`'s, bit for
//! bit), and the host walks the picks as `generate` does: each pick is the
//! token the plain step would give, so the bytes are the same whatever the
//! gate picks.
//!
//! The gate maximises expected tokens per ms. Its costs are measured, not
//! baked: at startup (`calibrate`) steps of both pools at a few row counts
//! and depths are timed and fitted, t(nb, Σd) = a + b nb + c Σd (Σd: the
//! rows' positions summed: the weights' share is per row, attention's per
//! position read), then corrected online per row count by the measured
//! steps (an EMA of measured / predicted). Each request's guess acceptance
//! α is an EMA over the guesses it reached, starting at the server's.

use crate::lm::{Backend, BatchLm, RowHead, SeqRow};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Most guesses per request per round (rows: slots x (K + 1)).
pub const K_MAX: usize = 8;

/// Weight of a new observation in a request's α EMA.
const ALPHA_BETA: f64 = 1.0 / 16.0;
/// Weight of a new step in the per-row-count cost corrections.
const CORR_BETA: f64 = 1.0 / 8.0;
/// Timed repetitions per calibration point (after one untimed: the
/// graph's capture).
const CALIB_REPS: usize = 3;

/// Calibration samples: (rows, Σ positions, ms).
pub type Samples = Vec<(usize, f64, f64)>;

/// A step's cost model, ms: a + b nb + c Σd.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Fit {
    pub fn at(&self, nb: usize, sd: f64) -> f64 {
        (self.a + self.b * nb as f64 + self.c * sd).max(0.0)
    }

    /// The least-squares fit of (nb, Σd, ms) samples, or None if they
    /// don't determine one (fewer than 3 distinct points, collinear).
    pub fn solve(s: &[(usize, f64, f64)]) -> Option<Fit> {
        // Normal equations, 3x3, solved by Cramer's rule (scaled: Σd is
        // thousands of times nb).
        let sc = s.iter().map(|p| p.1).fold(1.0f64, f64::max);
        let mut m = [[0.0f64; 3]; 3];
        let mut v = [0.0f64; 3];
        for &(nb, sd, t) in s {
            let x = [1.0, nb as f64, sd / sc];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] += x[i] * x[j];
                }
                v[i] += x[i] * t;
            }
        }
        let det = |m: &[[f64; 3]; 3]| {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        let d = det(&m);
        let scale = m[0][0] * m[1][1] * m[2][2];
        if d.abs().partial_cmp(&(1e-9 * scale.abs().max(1e-300)))
            != Some(std::cmp::Ordering::Greater)
        {
            return None;
        }
        let mut r = [0.0; 3];
        for (k, rk) in r.iter_mut().enumerate() {
            let mut mk = m;
            for i in 0..3 {
                mk[i][k] = v[i];
            }
            *rk = det(&mk) / d;
        }
        let f = Fit {
            a: r[0],
            b: r[1],
            c: r[2] / sc,
        };
        f.a.is_finite().then_some(f)
    }
}

/// The measured costs of both models' steps, corrected online.
#[derive(Clone, Debug)]
pub struct Costs {
    pub verify: Fit,
    pub draft: Fit,
    /// measured / predicted, per row count (EMA; 1 until a step of that
    /// many rows ran).
    vcorr: Vec<f64>,
    dcorr: Vec<f64>,
}

impl Costs {
    pub fn new(verify: Fit, draft: Fit, rows: usize) -> Costs {
        Costs {
            verify,
            draft,
            vcorr: vec![1.0; rows + 1],
            dcorr: vec![1.0; rows + 1],
        }
    }

    pub fn t_verify(&self, nb: usize, sd: f64) -> f64 {
        self.verify.at(nb, sd) * self.vcorr.get(nb).copied().unwrap_or(1.0)
    }

    pub fn t_draft(&self, nb: usize, sd: f64) -> f64 {
        self.draft.at(nb, sd) * self.dcorr.get(nb).copied().unwrap_or(1.0)
    }

    fn learn(corr: &mut [f64], fit: &Fit, nb: usize, sd: f64, ms: f64) {
        let p = fit.at(nb, sd);
        if let Some(c) = corr.get_mut(nb)
            && p > 0.0
            && ms.is_finite()
        {
            *c += CORR_BETA * ((ms / p).clamp(0.2, 5.0) - *c);
        }
    }

    pub fn saw_verify(&mut self, nb: usize, sd: f64, ms: f64) {
        Self::learn(&mut self.vcorr, &self.verify, nb, sd, ms)
    }

    pub fn saw_draft(&mut self, nb: usize, sd: f64, ms: f64) {
        Self::learn(&mut self.dcorr, &self.draft, nb, sd, ms)
    }
}

/// What the gate knows of a running request.
#[derive(Clone, Copy, Debug)]
pub struct GateIn {
    /// Its guess acceptance.
    pub alpha: f64,
    /// Most guesses it may take (0: none; sampled, near its end, past the
    /// drafter's context).
    pub cap: usize,
    /// Its row's position (= the positions its KV read covers).
    pub depth: usize,
}

/// Expected tokens of a round with k guesses at acceptance a: 1 + a + ...
/// + a^k (guess j is used only if every one before it was).
pub fn expected(a: f64, k: usize) -> f64 {
    let mut e = 1.0;
    let mut p = 1.0;
    for _ in 0..k {
        p *= a;
        e += p;
    }
    e
}

/// Each request's guesses this round (`K_i`), the most goodput (expected
/// tokens / predicted ms) over Kmax in 0..=k_max: K_i = min(Kmax, cap_i),
/// then, lowest α first, a request's last guess is dropped while its
/// expected token (α^K_i) is worth less than its row's cost at the
/// round's rate. Kmax = 0 is the plain step.
pub fn choose(c: &Costs, seqs: &[GateIn], k_max: usize) -> Vec<usize> {
    let n = seqs.len();
    let sd0: f64 = seqs.iter().map(|s| s.depth as f64).sum();
    let plain = n as f64 / c.t_verify(n, sd0).max(1e-6);
    let mut best = (plain, vec![0; n]);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| seqs[i].alpha.total_cmp(&seqs[j].alpha));
    for kmax in 1..=k_max {
        let mut ks: Vec<usize> = seqs.iter().map(|s| s.cap.min(kmax)).collect();
        if ks.iter().all(|&k| k == 0) {
            break;
        }
        let rate = best.0;
        let row = |i: usize, k: usize| c.verify.b + c.verify.c * (seqs[i].depth + k) as f64;
        for &i in &order {
            while ks[i] > 0 && seqs[i].alpha.powi(ks[i] as i32) < row(i, ks[i]) * rate {
                ks[i] -= 1;
            }
        }
        let g = goodput(c, seqs, &ks);
        if g > best.0 {
            best = (g, ks);
        }
    }
    best.1
}

/// Expected tokens per predicted ms of a round giving request i ks[i]
/// guesses (no guesses at all: the plain step).
pub fn goodput(c: &Costs, seqs: &[GateIn], ks: &[usize]) -> f64 {
    let km = ks.iter().copied().max().unwrap_or(0);
    let mut t = 0.0;
    for j in 1..=km {
        let (mut nb, mut sd) = (0, 0.0);
        for (s, &k) in seqs.iter().zip(ks) {
            if k >= j {
                nb += 1;
                sd += (s.depth + j - 1) as f64;
            }
        }
        t += c.t_draft(nb, sd);
    }
    let (mut nb, mut sd, mut e) = (0, 0.0, 0.0);
    for (s, &k) in seqs.iter().zip(ks) {
        nb += k + 1;
        sd += (0..=k).map(|r| (s.depth + r) as f64).sum::<f64>();
        e += expected(s.alpha, k);
    }
    t += c.t_verify(nb, sd);
    e / t.max(1e-6)
}

/// Counters for /metrics and the log.
#[derive(Default)]
pub struct DraftMetrics {
    /// Rounds by their most guesses (index K; 0: plain steps).
    pub rounds: [AtomicU64; K_MAX + 1],
    /// Guesses verified, and those kept.
    pub drafted: AtomicU64,
    pub accepted: AtomicU64,
    /// The server's α (EMA), x 1e6 (0: none yet).
    pub alpha_ppm: AtomicU64,
    /// Tokens from drafted rounds (rows used), and drafter time, µs.
    pub draft_us: AtomicU64,
    pub verify_us: AtomicU64,
}

/// The draft model's pool beside the batch's, and the gate.
pub struct DraftPool {
    pub pool: <Backend<'static> as BatchLm>::Pool,
    /// Positions per drafter slot (≤ the main model's: past it no guesses).
    pub ctx: usize,
    /// The tokens whose rows each drafter slot holds, from 0.
    pub cached: Vec<Vec<u32>>,
    /// Each slot's request's α.
    pub alpha: Vec<f64>,
    /// The server's α (an EMA over every reached guess), None before any.
    pub global: Option<f64>,
    /// Guesses per request at most.
    pub k: usize,
    /// None: no cost model (AXON_DRAFT_CALIB=0, or calibration failed):
    /// every greedy request takes min(k, cap) guesses.
    pub costs: Option<Costs>,
    pub metrics: std::sync::Arc<DraftMetrics>,
}

impl DraftPool {
    /// α for a request starting in a slot: the server's; before any guess
    /// was seen, 1 (the first round runs every guess: it measures).
    pub fn start_alpha(&self) -> f64 {
        self.global.unwrap_or(1.0)
    }

    /// Guess `x` (kept or not) of the request in slot `s` was reached.
    pub fn saw(&mut self, s: usize, kept: bool) {
        let x = if kept { 1.0 } else { 0.0 };
        self.alpha[s] += ALPHA_BETA * (x - self.alpha[s]);
        let g = self.global.map_or(x, |g| g + ALPHA_BETA * (x - g));
        self.global = Some(g);
        self.metrics
            .alpha_ppm
            .store((g * 1e6) as u64, Ordering::Relaxed);
    }

    /// The guesses per request this round.
    pub fn choose(&self, seqs: &[GateIn]) -> Vec<usize> {
        match &self.costs {
            Some(c) => choose(c, seqs, self.k),
            None => seqs.iter().map(|s| s.cap.min(self.k)).collect(),
        }
    }
}

/// AXON_DRAFT_CALIB=0: no startup calibration (the gate then always
/// drafts K).
pub fn calib_enabled() -> bool {
    std::env::var_os("AXON_DRAFT_CALIB").is_none_or(|v| v != "0")
}

/// Time steps of `nb` rows spread over `slots` slots (row r: slot r %
/// slots, at depth + r / slots) on `net`'s `pool`: the median of
/// CALIB_REPS after one untimed (it captures the graph). Tokens are
/// arbitrary: a step's time doesn't depend on them; the slots hold
/// nothing yet, so the rows written are never read.
fn time_step<P>(
    net: &impl BatchLm<Pool = P>,
    pool: &mut P,
    slots: usize,
    nb: usize,
    depth: usize,
) -> Result<(f64, f64), String> {
    let rows: Vec<SeqRow> = (0..nb)
        .map(|r| SeqRow {
            token: 100 + r as u32,
            pos: depth + r / slots,
            slot: r % slots,
            head: RowHead::Argmax,
        })
        .collect();
    let sd: f64 = rows.iter().map(|r| r.pos as f64).sum();
    let mut ts = Vec::new();
    for i in 0..=CALIB_REPS {
        let t = Instant::now();
        net.step(pool, &rows).map_err(|e| format!("{e:?}"))?;
        if i > 0 {
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    ts.sort_by(f64::total_cmp);
    Ok((sd, ts[ts.len() / 2]))
}

/// The depths calibration times at: 512, 4096, 16384, each clamped below
/// `ctx` (room for `per_slot` rows a slot), deduplicated.
fn calib_depths(ctx: usize, per_slot: usize) -> Vec<usize> {
    let top = ctx.saturating_sub(per_slot + 1);
    let mut d: Vec<usize> = [512, 4096, 16384].iter().map(|&x| x.min(top)).collect();
    d.dedup();
    d
}

/// Fit `net`'s step cost on its pool: steps of each of `nbs` rows over
/// `slots` slots at `calib_depths`.
pub fn fit_steps<P>(
    net: &impl BatchLm<Pool = P>,
    pool: &mut P,
    slots: usize,
    ctx: usize,
    nbs: &[usize],
) -> Result<(Fit, Samples), String> {
    let per = nbs.iter().map(|n| n.div_ceil(slots)).max().unwrap_or(1);
    let mut samples = Vec::new();
    for &d in &calib_depths(ctx, per) {
        for &nb in nbs {
            let (sd, ms) = time_step(net, pool, slots, nb, d)?;
            samples.push((nb, sd, ms));
        }
    }
    let f = Fit::solve(&samples).ok_or("calibration: the step times fit no line")?;
    Ok((f, samples))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn costs(a: f64, b: f64, c: f64, da: f64, db: f64, dc: f64) -> Costs {
        Costs::new(
            Fit { a, b, c },
            Fit {
                a: da,
                b: db,
                c: dc,
            },
            64,
        )
    }

    #[test]
    fn fit_recovers_a_plane() {
        let f = Fit {
            a: 5.0,
            b: 0.08,
            c: 2.5e-4,
        };
        let mut s = Vec::new();
        for nb in [4usize, 8, 16, 32] {
            for d in [512.0, 4096.0, 16384.0] {
                let sd = nb as f64 * d;
                s.push((nb, sd, f.at(nb, sd)));
            }
        }
        let g = Fit::solve(&s).unwrap();
        assert!(
            (g.a - f.a).abs() < 1e-6 && (g.b - f.b).abs() < 1e-8,
            "{g:?}"
        );
        assert!((g.c - f.c).abs() < 1e-10, "{g:?}");
        // One depth and one row count: underdetermined.
        assert!(Fit::solve(&[(4, 2048.0, 6.0), (4, 2048.0, 6.1)]).is_none());
    }

    #[test]
    fn expected_tokens() {
        assert_eq!(expected(0.5, 0), 1.0);
        assert!((expected(0.756, 3) - 2.760).abs() < 1e-3);
    }

    /// r14 spec §1/§3.3 on r13's measured costs (bella 7B + 0.5B): C=4 at
    /// depth 600 drafts K=3; C=8 at depth 4k prefers K=1 to K=3; deep (16k)
    /// with today's verify the gate picks the plain step; caps hold.
    #[test]
    fn gate_on_r13_costs() {
        // Verify: nb4 6.35, nb16 7.91, nb32@4k 18.03, nb64@4k 28.63 ms;
        // +0.257 ms per row from d600 to d4k (7.56e-5 ms per position).
        // Draft: nb4 1.5, nb16 1.66 ms, +0.051 ms per row from d600 to 4k.
        let c = costs(5.9, 0.04, 7.56e-5, 1.41, 0.0133, 1.5e-5);
        let a = 0.756;
        let seqs = |n: usize, d: usize| {
            vec![
                GateIn {
                    alpha: a,
                    cap: 3,
                    depth: d
                };
                n
            ]
        };
        assert_eq!(choose(&c, &seqs(4, 600), 3), vec![3; 4]);
        let k8 = choose(&c, &seqs(8, 4096), 3);
        let g = |ks: &[usize]| goodput(&c, &seqs(8, 4096), ks);
        assert!(g(&k8) >= g(&[1; 8]) && g(&k8) >= g(&[3; 8]), "{k8:?}");
        assert!(g(&k8) > g(&[0; 8]));
        assert_eq!(choose(&c, &seqs(8, 16384), 3), vec![0; 8]);
        // A request with no room drafts nothing; a hopeless one neither.
        let mut s = seqs(4, 600);
        s[1].cap = 0;
        s[2].alpha = 0.01;
        let ks = choose(&c, &s, 3);
        assert_eq!((ks[1], ks[2]), (0, 0), "{ks:?}");
        assert!(ks[0] > 0 && ks[0] == ks[3], "{ks:?}");
    }

    /// When attention dominates (c x depth >> the rest) no guesses.
    #[test]
    fn gate_deep_attention_is_plain() {
        let c = costs(2.0, 0.05, 2e-3, 1.0, 0.01, 1e-4);
        let s = vec![
            GateIn {
                alpha: 0.9,
                cap: 3,
                depth: 30000,
            };
            4
        ];
        assert_eq!(choose(&c, &s, 3), vec![0; 4]);
    }
}
