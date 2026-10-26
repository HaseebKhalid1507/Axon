//! Token generation: the decode loop behind `axon gen` (and, through
//! `sample::generate`, its sampled variant), plus the batch size the CPU
//! prefills with. Public: `axon::generate` is how an embedder decodes.

use crate::lm::Lm;
use crate::sample;

/// Prompt tokens per `feed_batch` call on the CPU. 128, not 64: each batch
/// streams every weight from RAM once and pays ~10 pool joins per layer, so
/// bigger batches amortize both (measured: pp128/pp512 +7-8% at 1.5B, +6% on
/// qwen3 0.6B Q8_0, over 64; activations of 128 tokens still sit in L2).
pub const PREFILL_BATCH: usize = 128;

/// What `generate` did. Non-exhaustive: it may report more. Read its
/// fields; outside the crate only `generate` (or `GenStats::default`)
/// makes one.
///
/// ```compile_fail,E0639
/// let st = axon::GenStats {
///     ids: Vec::new(),
///     passes: 0,
///     decided: 0,
///     drafted: 0,
///     accepted: 0,
///     forward: (0, 0.0),
///     verify: Vec::new(),
/// };
/// ```
#[derive(Default)]
#[non_exhaustive]
pub struct GenStats {
    /// The emitted tokens (EOS excluded).
    pub ids: Vec<u32>,
    /// Weight passes: `forward` and `verify_argmax` calls.
    pub passes: usize,
    /// Tokens picked from logits those passes produced, a final EOS included.
    pub decided: usize,
    /// Draft tokens verified, and how many of them the model agreed with.
    pub drafted: usize,
    pub accepted: usize,
    /// (count, seconds) of `forward` passes, and of verify passes by batch
    /// size: verify[nb] for nb = 1 + guesses tokens.
    pub forward: (usize, f64),
    pub verify: Vec<(usize, f64)>,
}

/// `generate`'s error for drafting on a backend whose decode steps stream
/// (`Lm::decode_streams`): the CLI and `serve` refuse the same pair up front.
pub(crate) const STREAM_DECODE_DRAFT: &str =
    "AXON_STREAM_DECODE=1 can't be combined with drafting (max_draft > 0): not lossless";

/// `generate`'s settings. Outside the crate, build them with
/// [`GenOpts::greedy`]: in 0.1 the library decodes greedily only (plus
/// lossless drafting); sampling is the CLI's (`--temp`).
pub struct GenOpts<'a> {
    /// Stop after this many tokens.
    pub n_gen: usize,
    /// Stop before any of these tokens.
    pub stop: &'a [u32],
    /// Draft up to this many tokens per pass; 0 = plain greedy.
    pub max_draft: usize,
    /// Sample (sample.rs) instead of picking greedily: crate-internal in
    /// 0.1 (the sampler's settings have no public, validated constructor yet).
    pub(crate) sample: Option<&'a mut sample::Sampler>,
}

impl<'a> GenOpts<'a> {
    /// Greedy decoding of up to `n_gen` tokens, stopping before any of `stop`
    /// (e.g. `Tokenizer::eog`), no drafting (`max_draft` 0; set it, and pass
    /// a `draft` function to `generate`, to draft: the output is the same).
    pub fn greedy(n_gen: usize, stop: &'a [u32]) -> GenOpts<'a> {
        GenOpts {
            n_gen,
            stop,
            max_draft: 0,
            sample: None,
        }
    }
}

/// Greedy decoding of up to `o.n_gen` tokens, stopping before any of `o.stop`
/// (sampled instead with `o.sample`, inside the crate). `logits`
/// is the prediction after `prompt`, whose tokens fill cache positions
/// 0..prompt.len(). Every token goes to `emit` as it's picked.
///
/// With `o.max_draft` > 0, `draft(history, max)` may guess up to `max` tokens
/// to follow history (prompt + emitted). [last token, guesses...] then go
/// through one `verify_argmax`, which gives `argmax` of each logits row that
/// `forward` would give (same logits bit for bit, same tie rule), and each
/// guess is kept only while it equals the model's own pick: pick i is the
/// right one exactly when guesses 0..i were the tokens emitted. So the output is the plain greedy output, whatever the drafts;
/// they only change how many tokens one weight pass yields. That needs
/// `verify_argmax` to pick as `forward_argmax` does: a backend whose decode
/// steps stream (`Lm::decode_streams`, AXON_STREAM_DECODE=1 on a split)
/// doesn't, so drafting there is an `InvalidInput` error. A rejected
/// guess's cache slot is rewritten by the next pass before anything reads it.
///
/// No low-acceptance gate: replaying the 5 bench prompts' exact greedy streams
/// through this loop and the real drafter (r5 policy sim) showed every gate
/// tried (backoff, n-gram EMA, expected-value) losing more on prose than it
/// saves on story; drafting there costs < 1% by the measured pass costs.
pub fn generate<M: Lm>(
    net: &M,
    cache: &mut M::Cache,
    prompt: &[u32],
    logits: Vec<f32>,
    o: GenOpts<'_>,
    draft: &mut dyn FnMut(&[u32], usize) -> Vec<u32>,
    emit: &mut dyn FnMut(u32) -> std::io::Result<()>,
) -> std::io::Result<GenStats> {
    if o.max_draft > 0 && net.decode_streams() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            STREAM_DECODE_DRAFT,
        ));
    }
    if let Some(s) = o.sample {
        return sample::generate(
            net,
            cache,
            prompt,
            logits,
            o.n_gen,
            o.stop,
            o.max_draft,
            s,
            draft,
            emit,
        );
    }
    let mut st = GenStats::default();
    let mut hist = prompt.to_vec();
    // (guess, the model's pick after it) from the last verify pass, in order.
    let mut queue = std::collections::VecDeque::new();
    let GenOpts {
        n_gen,
        stop,
        max_draft,
        ..
    } = o;
    let mut pick = argmax(&logits[..]);
    let mut picks = 0usize;
    while st.ids.len() < n_gen {
        let next = pick;
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
            Some((guess, p)) if guess == next => {
                pick = p;
                st.accepted += 1;
                continue;
            }
            _ => queue.clear(),
        }
        // Guesses past the token after the last one needing logits are useless:
        // at most left - 1, so no pass goes beyond plain greedy's positions.
        let max = max_draft.min(left - 1);
        let mut guesses = if max > 0 {
            draft(&hist, max)
        } else {
            Vec::new()
        };
        guesses.truncate(max);
        let pos = hist.len() - 1;
        let t = std::time::Instant::now();
        if guesses.is_empty() {
            pick = net.forward_argmax(next, pos, cache);
            st.forward.0 += 1;
            st.forward.1 += t.elapsed().as_secs_f64();
        } else {
            st.drafted += guesses.len();
            let mut toks = vec![next];
            toks.extend_from_slice(&guesses);
            let mut ps = net.verify_argmax(&toks, pos, cache).into_iter();
            pick = ps.next().expect("verify_argmax: one pick per token");
            queue.extend(guesses.into_iter().zip(ps));
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

multiversion! {
    /// Index of the largest logit, the last one on ties (as `Iterator::max_by`
    /// returns). Two passes that each vectorize: a lane-wise max, then a scan
    /// back from the end, 8 at a time, for the last logit equal to it. A vocab is
    /// ~152k logits, every token. Unlike a `total_cmp` max, NaN logits are never
    /// picked (unless all are NaN) and -0.0 ties with +0.0.
    pub fn argmax<V3>(l: &[f32]) -> u32 {
        assert!(!l.is_empty(), "argmax of no logits");
        let (chunks, tail) = l.as_chunks::<8>();
        let mut lanes = [f32::NEG_INFINITY; 8];
        for ch in chunks {
            for i in 0..8 {
                // Not f32::max: its NaN rule costs extra instructions per lane on
                // x86. A plain compare is one maxps, and skips NaN all the same.
                lanes[i] = if ch[i] > lanes[i] { ch[i] } else { lanes[i] };
            }
        }
        let mut m = f32::NEG_INFINITY;
        for &b in tail.iter().chain(&lanes) {
            m = m.max(b);
        }
        let base = chunks.len() * 8;
        for (j, &v) in tail.iter().enumerate().rev() {
            if v == m {
                return (base + j) as u32;
            }
        }
        for (c, ch) in chunks.iter().enumerate().rev() {
            let mut hit = false;
            for &v in ch {
                hit |= v == m;
            }
            if hit {
                for (j, &v) in ch.iter().enumerate().rev() {
                    if v == m {
                        return (c * 8 + j) as u32;
                    }
                }
            }
        }
        (l.len() - 1) as u32 // all NaN
    }
}

#[cfg(test)]
mod tests {
    use super::{GenOpts, argmax, generate};

    fn reference(l: &[f32]) -> u32 {
        (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32
    }

    #[test]
    fn argmax_matches_max_by_including_ties() {
        let mut rng = 0x1357_9bdfu32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        for len in [1usize, 2, 7, 8, 9, 15, 16, 17, 100, 1000, 151_936] {
            for _ in 0..20 {
                // Few distinct values, so ties are common, all within one lane or across.
                let l: Vec<f32> = (0..len).map(|_| (next() % 9) as f32 - 4.0).collect();
                assert_eq!(argmax(&l), reference(&l), "len {len}");
                let l: Vec<f32> = (0..len).map(|_| next() as f32 / 1e7 - 200.0).collect();
                assert_eq!(argmax(&l), reference(&l), "len {len}");
            }
        }
        assert_eq!(argmax(&[f32::NEG_INFINITY; 5]), 4);
    }

    /// argmax's portable and x86-64-v3 variants pick the same index.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn argmax_multiversion_identical() {
        use crate::model::tests::{can_v3, mixed};
        if !can_v3() {
            return;
        }
        for n in [1usize, 7, 8, 9, 64, 151_936] {
            for (k, edgy) in [false, true].into_iter().enumerate() {
                let mut l = mixed(n, 5 + k as u64, 9.0, edgy);
                let a = super::argmax::portable(&l);
                assert_eq!(a, unsafe { super::argmax::v3(&l) }, "n={n}");
                l.iter_mut().for_each(|v| *v = v.round()); // ties
                let a = super::argmax::portable(&l);
                assert_eq!(a, unsafe { super::argmax::v3(&l) }, "ties n={n}");
            }
        }
        let nan = [f32::NAN; 11];
        assert_eq!(super::argmax::portable(&nan), unsafe {
            super::argmax::v3(&nan)
        });
    }

    #[test]
    #[ignore = "timing, run with --ignored --nocapture"]
    fn argmax_speed() {
        let l: Vec<f32> = (0..151_936)
            .map(|i| ((i * 7919) % 10007) as f32 * 0.01)
            .collect();
        for (name, f) in [
            ("max_by", reference as fn(&[f32]) -> u32),
            ("argmax", argmax),
        ] {
            let t = std::time::Instant::now();
            let mut s = 0u64;
            for _ in 0..200 {
                s += f(std::hint::black_box(&l)) as u64;
            }
            eprintln!(
                "{name}: {:.1} us ({s})",
                t.elapsed().as_secs_f64() * 1e6 / 200.0
            );
        }
    }

    /// A weightless `Lm`: the pick after token t at position p is a hash of
    /// t, p and the token cached at p - 1 (so a misused cache shows), and
    /// its logits peak there. Counts the calls `generate` makes.
    #[derive(Default)]
    struct Toy {
        forwards: std::cell::Cell<usize>,
        argmaxes: std::cell::Cell<usize>,
        verifies: std::cell::Cell<usize>,
        /// Claims its decode steps stream (`Lm::decode_streams`).
        streams: bool,
    }

    impl Toy {
        fn pick(t: u32, pos: usize, c: &mut [u32]) -> u32 {
            c[pos] = t;
            let prev = if pos > 0 { c[pos - 1] } else { 5 };
            (t * 31 + pos as u32 * 7 + prev * 3) % 97
        }
    }

    impl crate::lm::Sealed for Toy {}
    impl crate::Lm for Toy {
        type Cache = Vec<u32>;
        fn decode_streams(&self) -> bool {
            self.streams
        }
        fn vocab(&self) -> usize {
            97
        }
        fn new_cache(&self, max_ctx: usize) -> Vec<u32> {
            vec![u32::MAX; max_ctx]
        }
        fn forward(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> Vec<f32> {
            self.forwards.set(self.forwards.get() + 1);
            let mut l = vec![0.0; 97];
            l[Toy::pick(t, pos, c) as usize] = 1.0;
            l
        }
        fn forward_argmax(&self, t: u32, pos: usize, c: &mut Vec<u32>) -> u32 {
            self.argmaxes.set(self.argmaxes.get() + 1);
            Toy::pick(t, pos, c)
        }
        fn feed_batch(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) {
            c[start..start + ts.len()].copy_from_slice(ts);
        }
        fn verify_argmax(&self, ts: &[u32], start: usize, c: &mut Vec<u32>) -> Vec<u32> {
            self.verifies.set(self.verifies.get() + 1);
            (0..ts.len())
                .map(|b| Toy::pick(ts[b], start + b, c))
                .collect()
        }
    }

    /// `generate` through the `Lm` seam on a backend that isn't Qwen2:
    /// plain decode takes `forward_argmax` (never `forward`'s logits),
    /// drafting `verify_argmax`, and both give the reference greedy output.
    #[test]
    fn generate_uses_the_lm_seam() {
        use crate::Lm;
        const N: usize = 20;
        let prompt = [1u32, 2, 3];
        let net = Toy::default();
        let mut c = net.new_cache(prompt.len() + N + 1);
        net.feed_batch(&prompt[..2], 0, &mut c);
        let first = net.forward(3, 2, &mut c);
        let mut want = vec![argmax(&first)];
        while want.len() < N {
            let pos = prompt.len() + want.len() - 1;
            want.push(Toy::pick(*want.last().unwrap(), pos, &mut c));
        }
        for k in [0, 3] {
            let net = Toy::default();
            let mut cache = net.new_cache(prompt.len() + N + 1);
            net.feed_batch(&prompt[..2], 0, &mut cache);
            let logits = net.forward(3, 2, &mut cache);
            let st = generate(
                &net,
                &mut cache,
                &prompt,
                logits,
                GenOpts {
                    n_gen: N,
                    stop: &[],
                    max_draft: k,
                    sample: None,
                },
                &mut |h, max| {
                    let at = h.len() - prompt.len();
                    want[at..].iter().take(max).copied().collect()
                },
                &mut |_| Ok(()),
            )
            .unwrap();
            assert_eq!(st.ids, want, "k {k}");
            assert_eq!(net.forwards.get(), 1, "k {k}: only the prompt's forward");
            if k == 0 {
                assert_eq!((net.argmaxes.get(), net.verifies.get()), (N - 1, 0));
            } else {
                assert_eq!(net.argmaxes.get(), 0);
                assert_eq!(net.verifies.get(), st.passes);
                assert!(st.passes < N - 1);
            }
        }
    }

    /// `Best`, pushed in order or merged over consecutive ranges of any
    /// size, must pick exactly `argmax`'s index: ties (last wins), -0.0 vs
    /// +0.0, NaNs, -inf, all-NaN.
    #[test]
    fn best_matches_argmax() {
        use crate::model::Best;
        let mut rng = 0x2468_ace1u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        let pick = |l: &[f32], chunk: usize| {
            let mut acc = Best::NONE;
            for (c, ch) in l.chunks(chunk).enumerate() {
                let mut b = Best::NONE;
                for (j, &v) in ch.iter().enumerate() {
                    b.push(v, (c * chunk + j) as u32);
                }
                acc.merge(b);
            }
            acc.index(l.len())
        };
        let vals = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.5,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        for len in [1usize, 2, 7, 8, 9, 31, 100, 1000] {
            for round in 0..200 {
                // Few distinct values so ties are common, randomly negated;
                // smaller pools leave out inf/NaN/2.5/±1 so the max varies.
                let pool = [8, 6, 5, 4, 2][round % 5];
                let l: Vec<f32> = (0..len)
                    .map(|_| {
                        let r = next() as usize;
                        let v = vals[r % pool];
                        if (r >> 8).is_multiple_of(7) { -v } else { v }
                    })
                    .collect();
                for chunk in [1, 2, 3, 8, 64, len] {
                    assert_eq!(pick(&l, chunk), argmax(&l), "{l:?} chunk {chunk}");
                }
            }
            let nan = vec![f32::NAN; len];
            assert_eq!(pick(&nan, 3), argmax(&nan));
            let ninf = vec![f32::NEG_INFINITY; len];
            assert_eq!(pick(&ninf, 3), argmax(&ninf));
        }
    }

    /// Drafting on a backend whose decode steps stream (AXON_STREAM_DECODE=1
    /// on a split) isn't lossless: `generate` refuses it before any pass,
    /// whatever the sampler; plain decode on it runs.
    #[test]
    fn generate_refuses_drafting_when_decode_streams() {
        use crate::Lm;
        let prompt = [1u32, 2, 3];
        for k in [0, 3] {
            let net = Toy {
                streams: true,
                ..Toy::default()
            };
            let mut cache = net.new_cache(16);
            net.feed_batch(&prompt[..2], 0, &mut cache);
            let logits = net.forward(3, 2, &mut cache);
            let mut o = GenOpts::greedy(8, &[]);
            o.max_draft = k;
            let r = generate(
                &net,
                &mut cache,
                &prompt,
                logits,
                o,
                &mut |_, _| vec![5],
                &mut |_| Ok(()),
            );
            match k {
                0 => assert_eq!(r.unwrap().ids.len(), 8),
                _ => {
                    let e = r.err().expect("drafting refused");
                    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
                    assert!(e.to_string().contains("AXON_STREAM_DECODE"), "{e}");
                    assert_eq!((net.argmaxes.get(), net.verifies.get()), (0, 0));
                }
            }
        }
    }
}
