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

/// What `generate` did.
#[derive(Default)]
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

/// `generate`'s settings.
pub struct GenOpts<'a> {
    /// Stop after this many tokens.
    pub n_gen: usize,
    /// Stop before any of these tokens.
    pub stop: &'a [u32],
    /// Draft up to this many tokens per pass; 0 = plain greedy.
    pub max_draft: usize,
    /// Sample (sample.rs) instead of picking greedily.
    pub sample: Option<&'a mut sample::Sampler>,
}

/// Greedy decoding of up to `o.n_gen` tokens, stopping before any of `o.stop`. `logits`
/// is the prediction after `prompt`, whose tokens fill cache positions
/// 0..prompt.len(). Every token goes to `emit` as it's picked.
///
/// With `o.max_draft` > 0, `draft(history, max)` may guess up to `max` tokens
/// to follow history (prompt + emitted). [last token, guesses...] then go
/// through one `verify_argmax`, which gives `argmax` of each logits row that
/// `forward` would give (same logits bit for bit, same tie rule), and each
/// guess is kept only while it equals the model's own pick: pick i is the
/// right one exactly when guesses 0..i were the tokens emitted. So the output is the plain greedy output, whatever the drafts;
/// they only change how many tokens one weight pass yields. A rejected
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
