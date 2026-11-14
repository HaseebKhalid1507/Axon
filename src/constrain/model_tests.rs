//! Model-gated tests of constrained decoding (docs/determinism.md's
//! contract with a mask): on the CPU and on CUDA, a constrained sequence's
//! candidates are the same bits token by token (`forward_cands_masked`),
//! in batched verify passes (`verify_cands_masked`) and, on CUDA, in pool
//! steps beside other requests (`step_rows` with `RowHead::Sample.mask`);
//! generation drafted or not gives the same tokens; the logits-row
//! fallback (too few allowed among the device's best) is `cands_masked` of
//! the row; and a constrained request's tokens are the unconstrained ones
//! until its trigger. The masks themselves on the real vocabulary, and the
//! per-token cost (`mask_cost_real`, ignored: `--ignored --nocapture`).
//!
//! The model: `testutil::MODEL` (Qwen2.5-0.5B), or
//! `AXON_CONSTRAIN_TEST_MODEL` (a Qwen2.5 GGUF: the 7B gate job).

use super::tests::mask_cost_on;
use super::{Constraint, Grammar, TokenMask, Vocab};
use crate::gguf::Model;
use crate::lm::Lm;
use crate::sample::{self, Cand, CandSpec, SampleCfg, Sampler, cands_masked};
use crate::tokenizer::Tokenizer;
use std::sync::Arc;

/// The model's logits per position (its embedding's rows).
fn logits_n(m: &Model) -> usize {
    let e = m
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .unwrap();
    e.dims[1] as usize
}

fn test_model() -> Option<Model> {
    match std::env::var_os("AXON_CONSTRAIN_TEST_MODEL") {
        Some(p) => Some(crate::gguf::load(p.to_str().expect("utf-8 path")).expect("load model")),
        None => crate::testutil::model(),
    }
}

/// Qwen2.5's Hermes tool prompt (its chat template's rendering of one
/// tool), asking for a call whose arguments hold quotes and backslashes:
/// the shape #259's malformed calls had.
const SYSTEM: &str = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a \
helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user \
query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\
{\"type\": \"function\", \"function\": {\"name\": \"exec_command\", \"description\": \"Run a \
shell command\", \"parameters\": {\"type\": \"object\", \"properties\": {\"cmd\": {\"type\": \
\"string\"}}, \"required\": [\"cmd\"]}}}\n</tools>\n\nFor each function call, return a json \
object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n\
<|im_start|>user\nIn ledger/cli.py replace raise KeyError(f'No such account: {name}') with \
raise ValueError(\"unknown \\\"account\\\"\"), then print \"done\".<|im_end|>\n<|im_start|>assistant\n";

/// (prompt, tokens of it after its trigger): the free prompt (the model
/// may call or not), and the same with `<tool_call>\n` after it (engaged
/// from the first generated token).
fn prompts(tok: &Tokenizer) -> [(Vec<u32>, usize); 2] {
    let free = tok.encode_special(SYSTEM);
    let on = tok.encode_special(&format!("{SYSTEM}<tool_call>\n"));
    let open = tok.special_id("<tool_call>").unwrap();
    let after = on.len() - 1 - on.iter().rposition(|&t| t == open).unwrap();
    [(free, usize::MAX), (on, after)]
}

/// A constraint at the end of `prompt`: engaged by its trigger if
/// `after` (tokens after the trigger) is set.
fn constraint_at(g: &Arc<Grammar>, prompt: &[u32], after: usize) -> Constraint {
    let mut c = Constraint::new(g.clone());
    if after != usize::MAX {
        for &t in &prompt[prompt.len() - after - 1..] {
            assert!(c.accept(t));
        }
        assert!(c.engaged());
    }
    c
}

fn bits(c: &[Cand]) -> Vec<(u32, u32)> {
    c.iter().map(|c| (c.logit.to_bits(), c.id)).collect()
}

/// The candidate spec the contract tests pick by (the first candidate:
/// deterministic, no RNG; the penalty on, so the logits row a fallback
/// reads must be the unpenalized one).
const SPEC: CandSpec = CandSpec {
    k: 40,
    penalty: 1.1,
    last_n: 64,
};

/// Decode `n` tokens after `prompt` one at a time: an engaged position by
/// `forward_cands_masked`, a free one by `forward_cands`, both `SPEC`,
/// taking the first candidate (stopping at `stop`). The tokens, each
/// position's candidates, and how many positions were engaged.
fn decode<M: Lm>(
    net: &M,
    g: &Arc<Grammar>,
    prompt: &[u32],
    after: usize,
    n: usize,
    stop: &[u32],
) -> (Vec<u32>, Vec<Vec<Cand>>, usize) {
    let mut cache = net.new_cache(prompt.len() + n + 1);
    let (&last, rest) = prompt.split_last().unwrap();
    net.feed_batch(rest, 0, &mut cache);
    let mut con = constraint_at(g, prompt, after);
    let mut hist = prompt.to_vec();
    let (mut ids, mut rows, mut engaged) = (Vec::new(), Vec::new(), 0);
    let mut t = last;
    for _ in 0..n {
        let pos = hist.len() - 1;
        let c = match con.mask().unwrap() {
            Some(m) => {
                engaged += 1;
                net.forward_cands_masked(t, pos, &mut cache, &SPEC, &hist, &m)
            }
            None => net.forward_cands(t, pos, &mut cache, &SPEC, &hist),
        };
        let next = c[0].id;
        rows.push(c);
        if stop.contains(&next) {
            break;
        }
        assert!(con.accept(next), "a candidate outside the mask");
        ids.push(next);
        hist.push(next);
        t = next;
    }
    (ids, rows, engaged)
}

/// The masks of the positions after `prompt` then `ids[..b]`, b = 0..n.
fn masks_along(
    g: &Arc<Grammar>,
    prompt: &[u32],
    after: usize,
    ids: &[u32],
    n: usize,
) -> Vec<Option<TokenMask>> {
    let mut con = constraint_at(g, prompt, after);
    let mut out = Vec::with_capacity(n);
    for b in 0..n {
        out.push(con.mask().unwrap());
        if b < ids.len() {
            assert!(con.accept(ids[b]));
        }
    }
    out
}

/// `decode`'s candidates again in verify passes (`verify_cands_masked`)
/// of `chunk` tokens: the same bits.
#[allow(clippy::too_many_arguments)]
fn verify_matches<M: Lm>(
    net: &M,
    g: &Arc<Grammar>,
    prompt: &[u32],
    after: usize,
    ids: &[u32],
    want: &[Vec<Cand>],
    chunk: usize,
    what: &str,
) {
    let n = want.len();
    let mut cache = net.new_cache(prompt.len() + n + 1);
    let (&last, rest) = prompt.split_last().unwrap();
    net.feed_batch(rest, 0, &mut cache);
    let masks = masks_along(g, prompt, after, ids, n);
    // Row b's token: the prompt's last, then ids.
    let mut toks = vec![last];
    toks.extend_from_slice(&ids[..n - 1]);
    let mut hist = prompt.to_vec();
    hist.extend_from_slice(&ids[..n - 1]);
    let mut b = 0;
    while b < n {
        let e = (b + chunk).min(n);
        let recent = &hist[..prompt.len() + e - 1];
        let m: Vec<Option<&TokenMask>> = masks[b..e].iter().map(Option::as_ref).collect();
        let rows =
            net.verify_cands_masked(&toks[b..e], rest.len() + b, &mut cache, &SPEC, recent, &m);
        for (j, r) in rows.iter().enumerate() {
            assert_eq!(
                bits(r),
                bits(&want[b + j]),
                "{what}: chunk {chunk}, position {}",
                b + j
            );
        }
        b = e;
    }
}

/// Generate through `crate::generate` (greedy or sampled), drafted by
/// `mode`: "" none, "oracle" (`oracle`'s tokens), "wrong", "lookup".
#[allow(clippy::too_many_arguments)]
fn generate_with<M: Lm>(
    net: &M,
    g: Option<&Arc<Grammar>>,
    prompt: &[u32],
    after: usize,
    n: usize,
    stop: &[u32],
    cfg: Option<SampleCfg>,
    mode: &str,
    oracle: &[u32],
) -> crate::GenStats {
    let mut cache = net.new_cache(prompt.len() + n + 1);
    let (&last, rest) = prompt.split_last().unwrap();
    net.feed_batch(rest, 0, &mut cache);
    let logits = net.forward(last, rest.len(), &mut cache);
    let mut sampler = cfg.map(Sampler::new);
    let mut con = g.map(|g| constraint_at(g, prompt, after));
    let mut lookup = crate::spec::PromptLookup::new();
    let mut draft = |h: &[u32], max: usize| -> Vec<u32> {
        let at = h.len() - prompt.len();
        let right = |i: usize| oracle.get(at + i).copied().unwrap_or(0);
        match mode {
            "oracle" => (0..max).map(right).collect(),
            "wrong" => (0..max).map(|i| right(i) ^ 1).collect(),
            _ => {
                lookup.sync(h);
                lookup.draft(max)
            }
        }
    };
    crate::generate(
        net,
        &mut cache,
        prompt,
        logits,
        crate::GenOpts {
            n_gen: n,
            stop,
            max_draft: if mode.is_empty() { 0 } else { 5 },
            sample: sampler.as_mut(),
            constraint: con.as_mut(),
        },
        &mut draft,
        &mut |_| Ok(()),
    )
    .unwrap()
}

/// The contract on one backend: decode == verify passes (chunks 1, 3, 8,
/// 16), generation drafted == undrafted (greedy and two seeds), and the
/// constrained request's tokens == the unconstrained ones until its
/// trigger. Returns `decode`'s results per prompt for the pool test.
#[allow(clippy::type_complexity)]
fn contract<M: Lm>(
    net: &M,
    tok: &Tokenizer,
    g: &Arc<Grammar>,
    n: usize,
    label: &str,
) -> Vec<(Vec<u32>, Vec<u32>, usize, Vec<Vec<Cand>>)> {
    let stop = tok.eog();
    let mut out = Vec::new();
    for (pi, (prompt, after)) in prompts(tok).into_iter().enumerate() {
        let what = format!("{label} prompt {pi}");
        let (ids, rows, engaged) = decode(net, g, &prompt, after, n, &stop);
        eprintln!(
            "{what}: {} tokens, {engaged} engaged: {:?}",
            ids.len(),
            tok.decode(&ids)
        );
        if after != usize::MAX {
            assert!(engaged > 0, "{what}: never engaged");
        }
        for chunk in [1, 3, 8, 16] {
            verify_matches(net, g, &prompt, after, &ids, &rows, chunk, &what);
        }
        let cfgs = [
            None,
            Some(SampleCfg {
                temp: 0.8,
                top_k: 40,
                top_p: 0.95,
                min_p: 0.0,
                repeat_penalty: 1.1,
                repeat_last_n: 64,
                seed: 7,
            }),
            Some(SampleCfg {
                temp: 1.2,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.02,
                repeat_penalty: 1.0,
                repeat_last_n: 64,
                seed: 1234,
            }),
        ];
        for cfg in cfgs {
            let plain = generate_with(net, Some(g), &prompt, after, n, &stop, cfg, "", &[]);
            for mode in ["oracle", "wrong", "lookup"] {
                let st = generate_with(
                    net,
                    Some(g),
                    &prompt,
                    after,
                    n,
                    &stop,
                    cfg,
                    mode,
                    &plain.ids,
                );
                let ctx = format!("{what} seed {:?} drafted by {mode}", cfg.map(|c| c.seed));
                assert_eq!(st.ids, plain.ids, "{ctx}");
                assert_eq!(st.decided, plain.decided, "{ctx}");
            }
            if after == usize::MAX {
                // Free until the trigger: the unconstrained tokens.
                let free = generate_with(net, None, &prompt, after, n, &stop, cfg, "", &[]);
                let open = tok.special_id("<tool_call>").unwrap();
                let upto = plain
                    .ids
                    .iter()
                    .position(|&t| t == open)
                    .map_or(plain.ids.len(), |t| t + 1);
                assert_eq!(
                    plain.ids[..upto],
                    free.ids[..upto.min(free.ids.len())],
                    "{what} seed {:?}: before the trigger",
                    cfg.map(|c| c.seed)
                );
            }
            if let Some(c) = cfg {
                eprintln!("{what} seed {}: {:?}", c.seed, tok.decode(&plain.ids));
            }
        }
        out.push((prompt, ids, after, rows));
    }
    out
}

/// `forward_cands_masked` with masks that allow few tokens, mostly unlikely
/// ones (the fallback's case on a device: too few allowed among its
/// best), == `cands_masked` of the `forward` logits at the same position;
/// `verify_cands_masked` with such masks == per-row. Returns how many
/// positions it checked.
fn restrictive<M: Lm>(net: &M, prompt: &[u32], n: usize, label: &str) -> usize {
    let v = net.vocab();
    let mut s = 0x0dd_ba11u64;
    let mut xs = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut c1 = net.new_cache(prompt.len() + n + 1);
    let mut c2 = net.new_cache(prompt.len() + n + 1);
    let mut c3 = net.new_cache(prompt.len() + n + 1);
    let (&last, rest) = prompt.split_last().unwrap();
    for c in [&mut c1, &mut c2, &mut c3] {
        net.feed_batch(rest, 0, c);
    }
    let mut hist = prompt.to_vec();
    let mut t = last;
    let mut masks = Vec::new();
    let mut want = Vec::new();
    for i in 0..n {
        let pos = hist.len() - 1;
        let ids: Vec<u32> = (0..1 + i % 4).map(|_| (xs() % v as u64) as u32).collect();
        let m = TokenMask::from_ids(v, &ids);
        let got = net.forward_cands_masked(t, pos, &mut c1, &SPEC, &hist, &m);
        let mut l = net.forward(t, pos, &mut c2);
        let w = cands_masked(&mut l, &SPEC, &hist, &m);
        assert_eq!(bits(&got), bits(&w), "{label}: restrictive mask at {pos}");
        assert!(got.iter().all(|c| m.allows(c.id)));
        want.push(w);
        masks.push(m);
        // Next: a likely token (the argmax), as decoding would.
        t = crate::argmax(&l);
        hist.push(t);
    }
    // The same rows in verify passes of up to 8 (each row restricted).
    let toks: Vec<u32> = std::iter::once(last)
        .chain(hist[prompt.len()..].iter().copied())
        .take(n)
        .collect();
    let mut b = 0;
    while b < n {
        let e = (b + 8).min(n);
        let m: Vec<Option<&TokenMask>> = masks[b..e].iter().map(Some).collect();
        let rows = net.verify_cands_masked(
            &toks[b..e],
            rest.len() + b,
            &mut c3,
            &SPEC,
            &hist[..prompt.len() + e - 1],
            &m,
        );
        for (j, r) in rows.iter().enumerate() {
            assert_eq!(
                bits(r),
                bits(&want[b + j]),
                "{label}: restrictive verify row {}",
                b + j
            );
        }
        b = e;
    }
    n
}

/// The grammar on the model's vocabulary.
fn grammar(m: &Model, net_vocab: usize) -> (&'static Tokenizer, Arc<Grammar>) {
    let tok: &'static Tokenizer = Box::leak(Box::new(Tokenizer::from_gguf(m)));
    let v = Arc::new(Vocab::new(tok, net_vocab));
    assert!(v.byte_complete(), "a Qwen2.5 vocabulary is byte-complete");
    (
        tok,
        Arc::new(Grammar::hermes(v).expect("Qwen2.5 has <tool_call>")),
    )
}

/// Fewer tokens on a big model (the CPU 7B runs at a few tokens a second).
fn n_for(m: &Model) -> usize {
    if m.total_params() > 3_000_000_000 {
        24
    } else {
        48
    }
}

#[test]
fn cpu_constrained_contract() {
    let Some(m) = test_model() else {
        return;
    };
    let net = crate::model::Qwen2::new(&m);
    let (tok, g) = grammar(&m, net.vocab());
    let n = n_for(&m);
    contract(&net, tok, &g, n, "cpu");
    let p = &prompts(tok)[1].0;
    restrictive(&net, p, 12, "cpu");
}

#[test]
fn cuda_constrained_contract() {
    let Some(gpu) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = test_model() else {
        return;
    };
    use crate::cuda::qwen2::CudaQwen2;
    use crate::lm::{RowHead, RowOut, SeqRow};
    let km = crate::cuda::Kernels::module(&gpu).unwrap();
    let ctx = 1024;
    let net = CudaQwen2::new(&gpu, &km, &m, ctx).unwrap();
    let (tok, g) = grammar(&m, net.vocab());
    let n = n_for(&m);
    let f0 = sample::MASK_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed);
    let runs = contract(&net, tok, &g, n, "cuda");
    let p1 = &prompts(tok)[1].0;
    restrictive(&net, p1, 12, "cuda");
    let f1 = sample::MASK_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(f1 > f0, "the logits-row fallback never ran");
    // Pool steps: each constrained run beside the others and an
    // unconstrained greedy request (its own argmax), in reversed slots,
    // every step's candidates == the solo decode's; then restrictive masks
    // in a pool step (the fallback reads its row: `step_row_logits`).
    let slots = 4;
    let mut p = net.pool(slots, ctx).unwrap();
    let free_prompt = &prompts(tok)[0].0;
    let (free_ids, _) = {
        let mut c = net.new_cache(ctx);
        let (&last, rest) = free_prompt.split_last().unwrap();
        net.feed_batch(rest, 0, &mut c);
        let mut t = last;
        let mut ids = Vec::new();
        for i in 0..n {
            t = net.forward_argmax(t, rest.len() + i, &mut c);
            ids.push(t);
        }
        (ids, ())
    };
    // (prompt, tokens after its trigger, the solo decode's candidates if
    // constrained, its tokens)
    type Req<'a> = (&'a [u32], usize, Option<&'a [Vec<Cand>]>, &'a [u32]);
    let reqs: Vec<Req> = vec![
        (&runs[0].0, runs[0].2, Some(&runs[0].3), &runs[0].1),
        (&runs[1].0, runs[1].2, Some(&runs[1].3), &runs[1].1),
        (free_prompt, usize::MAX, None, &free_ids),
        (&runs[1].0, runs[1].2, Some(&runs[1].3), &runs[1].1),
    ];
    let slot_of = |i: usize| slots - 1 - i;
    for (i, r) in reqs.iter().enumerate() {
        net.feed_slot(&mut p, slot_of(i), &r.0[..r.0.len() - 1], 0)
            .unwrap();
    }
    let mut cons: Vec<Constraint> = reqs.iter().map(|r| constraint_at(&g, r.0, r.1)).collect();
    let mut hist: Vec<Vec<u32>> = reqs.iter().map(|r| r.0.to_vec()).collect();
    let steps = reqs.iter().map(|r| r.3.len()).min().unwrap().min(n);
    for step in 0..steps {
        let masks: Vec<Option<TokenMask>> = (0..slots)
            .map(|i| match reqs[i].2 {
                Some(_) => cons[i].mask().unwrap(),
                None => None,
            })
            .collect();
        let rows: Vec<SeqRow> = (0..slots)
            .map(|i| {
                let h = &hist[i];
                SeqRow {
                    token: h[h.len() - 1],
                    pos: h.len() - 1,
                    slot: slot_of(i),
                    head: match reqs[i].2 {
                        None => RowHead::Argmax,
                        Some(_) => RowHead::Sample {
                            spec: SPEC,
                            recent: &h[h.len() - SPEC.last_n.clamp(1, h.len())..],
                            mask: masks[i].as_ref(),
                        },
                    },
                }
            })
            .collect();
        let outs = net.step_rows(&mut p, &rows).unwrap();
        for (i, o) in outs.into_iter().enumerate() {
            let next = match (o, reqs[i].2) {
                (RowOut::Pick(t), None) => t,
                (RowOut::Cands(c), Some(want)) => {
                    assert_eq!(bits(&c), bits(&want[step]), "pool request {i} step {step}");
                    c[0].id
                }
                _ => panic!("a result of another head"),
            };
            assert_eq!(next, reqs[i].3[step], "pool request {i} step {step}: token");
            if reqs[i].2.is_some() {
                assert!(cons[i].accept(next));
            }
            hist[i].push(next);
        }
    }
    // A pool step whose masked rows allow a few random ids each.
    let v = net.vocab();
    let rmask: Vec<TokenMask> = (0..slots)
        .map(|i| {
            TokenMask::from_ids(
                v,
                &[
                    (i * 7919 + 13) as u32 % v as u32,
                    (i * 104_729 + 5) as u32 % v as u32,
                ],
            )
        })
        .collect();
    let f2 = sample::MASK_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed);
    let rows: Vec<SeqRow> = (0..slots)
        .map(|i| {
            let h = &hist[i];
            SeqRow {
                token: h[h.len() - 1],
                pos: h.len() - 1,
                slot: slot_of(i),
                head: RowHead::Sample {
                    spec: SPEC,
                    recent: &h[h.len() - SPEC.last_n.clamp(1, h.len())..],
                    mask: Some(&rmask[i]),
                },
            }
        })
        .collect();
    let outs = net.step_rows(&mut p, &rows).unwrap();
    for (i, o) in outs.into_iter().enumerate() {
        // The reference: this request's history alone, forward's logits.
        let h = &hist[i];
        let mut c = net.new_cache(ctx);
        net.feed_batch(&h[..h.len() - 1], 0, &mut c);
        let mut l = net.forward(h[h.len() - 1], h.len() - 1, &mut c);
        let want = cands_masked(&mut l, &SPEC, h, &rmask[i]);
        let RowOut::Cands(got) = o else {
            panic!("a pick for a sampled row")
        };
        assert_eq!(bits(&got), bits(&want), "pool restrictive row {i}");
    }
    let f3 = sample::MASK_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(f3 > f2, "the pool's fallback never ran");
}

/// The masks on the real vocabulary (151k tokens, byte-level BPE) equal
/// walking every token, along a call tokenized as the model would write
/// it: special tokens (EOS, `<|im_end|>`, `<tool_call>`) never allowed in
/// a string, `</tool_call>` only at the close.
#[test]
fn real_vocab_masks() {
    let Some(m) = test_model() else {
        return;
    };
    let tok = Tokenizer::from_gguf(&m);
    let n = logits_n(&m);
    let v = Arc::new(Vocab::new(&tok, n));
    assert!(v.byte_complete());
    let g = Arc::new(Grammar::hermes(v.clone()).unwrap());
    let open = tok.special_id("<tool_call>").unwrap();
    let close = tok.special_id("</tool_call>").unwrap();
    let call = "\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"sed -i 's/raise \
                KeyError(msg)/raise ValueError(\\\"bad\\\")/' ledger/cli.py && echo \\\"done\\\"\", \
                \"n\": [1, -2.5e3, {\"k\": null}]}}\n";
    let mut ids = tok.encode(call);
    ids.push(close);
    let mut c = Constraint::new(g.clone());
    assert!(c.accept(open));
    let mut buf = Vec::new();
    let mut checked = 0;
    for (i, &t) in ids.iter().enumerate() {
        let mask = c.mask().unwrap().expect("engaged");
        // Every 3rd position walked in full (each walk is 151k tokens).
        if i % 3 == 0 || i + 1 == ids.len() {
            let super::Phase::On { node, stack } = &c.phase else {
                unreachable!()
            };
            let naive: Vec<u32> = (0..n as u32)
                .filter(|&id| g.masks.walk(*node, stack, id, &mut buf).is_some())
                .collect();
            let mut got: Vec<u32> = mask.ids().collect();
            got.sort_unstable();
            assert_eq!(got, naive, "position {i}");
            checked += 1;
        }
        let in_str = c.in_string();
        for &s in v.specials() {
            if in_str || s != close {
                assert!(
                    !mask.allows(s),
                    "special {s} ({:?}) allowed at {i}",
                    tok.decode(&[s])
                );
            }
        }
        assert!(mask.allows(t), "token {t} ({:?}) at {i}", tok.decode(&[t]));
        assert!(c.accept(t));
    }
    assert!(!c.engaged(), "the call closed");
    eprintln!("{checked} positions walked in full over {n} tokens");
}

/// The per-token mask cost on the real vocabulary: the call as the
/// tokenizer writes it, and byte by byte (a token per byte: the worst
/// case), against the spec's bar (median <= 20 us, p99 <= 100 us).
#[test]
#[ignore]
fn mask_cost_real() {
    let Some(m) = test_model() else {
        return;
    };
    let tok = Tokenizer::from_gguf(&m);
    let v = Arc::new(Vocab::new(&tok, logits_n(&m)));
    let t0 = std::time::Instant::now();
    let _ = Vocab::new(&tok, logits_n(&m));
    eprintln!("Vocab::new: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    let open = tok.special_id("<tool_call>").unwrap();
    let call = super::tests::COST_CALL;
    let byte_tok = |b: u8| {
        (0..v.len() as u32)
            .find(|&id| v.bytes(id) == [b] && !v.is_special(id))
            .unwrap()
    };
    eprintln!("tokenized call:");
    mask_cost_on(v.clone(), open, &tok.encode(call));
    eprintln!("byte by byte:");
    let bytes: Vec<u32> = call.bytes().map(byte_tok).collect();
    mask_cost_on(v, open, &bytes);
}

/// What a constrained token costs a decode step on CUDA, end to end: along
/// one constrained greedy reply (engaged from its first token, until the
/// call closes or 64 tokens), each position run three ways on three caches
/// fed the same tokens (the constrained step's picks): the greedy step
/// (`forward_argmax`), a sampled step (`forward_cands`, k 40) and the
/// constrained step (`Constraint::mask`, `forward_cands_masked` with
/// `GREEDY_SPEC`: the device's CAND_MAX best, the allowed kept, then
/// `accept`). Every round must pick the same tokens. Medians per token over
/// 10 rounds. `--ignored --nocapture` to see.
#[test]
#[ignore]
fn constrained_step_cost_cuda() {
    let Some(gpu) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = test_model() else {
        return;
    };
    use crate::cuda::qwen2::CudaQwen2;
    let km = crate::cuda::Kernels::module(&gpu).unwrap();
    let ctx = 1024;
    let net = CudaQwen2::new(&gpu, &km, &m, ctx).unwrap();
    let (tok, g) = grammar(&m, net.vocab());
    let (prompt, after) = prompts(tok)[1].clone();
    let k40 = CandSpec {
        k: 40,
        penalty: 1.0,
        last_n: 64,
    };
    let f0 = sample::MASK_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed);
    let (mut ta, mut ts, mut tm) = (Vec::new(), Vec::new(), Vec::new());
    let mut first: Option<Vec<u32>> = None;
    for _ in 0..10 {
        let mut cs: Vec<_> = (0..3).map(|_| net.new_cache(ctx)).collect();
        let (&last, rest) = prompt.split_last().unwrap();
        for c in cs.iter_mut() {
            net.feed_batch(rest, 0, c);
        }
        let mut con = constraint_at(&g, &prompt, after);
        let mut hist = prompt.clone();
        let mut t = last;
        let mut ids = Vec::new();
        while con.engaged() && ids.len() < 64 {
            let pos = hist.len() - 1;
            let t0 = std::time::Instant::now();
            std::hint::black_box(net.forward_argmax(t, pos, &mut cs[0]));
            let t1 = std::time::Instant::now();
            std::hint::black_box(net.forward_cands(t, pos, &mut cs[1], &k40, &hist));
            let t2 = std::time::Instant::now();
            let mask = con.mask().unwrap().unwrap();
            let c =
                net.forward_cands_masked(t, pos, &mut cs[2], &sample::GREEDY_SPEC, &hist, &mask);
            let next = c[0].id;
            assert!(con.accept(next));
            let t3 = std::time::Instant::now();
            ta.push((t1 - t0).as_secs_f64() * 1e3);
            ts.push((t2 - t1).as_secs_f64() * 1e3);
            tm.push((t3 - t2).as_secs_f64() * 1e3);
            ids.push(next);
            hist.push(next);
            t = next;
        }
        match &first {
            None => first = Some(ids),
            Some(f) => assert_eq!(&ids, f, "a round picked other tokens"),
        }
    }
    let ids = first.unwrap();
    let f1 = sample::MASK_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed);
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (a, s, c) = (med(&mut ta), med(&mut ts), med(&mut tm));
    eprintln!("constrained greedy reply: {:?}", tok.decode(&ids));
    eprintln!(
        "constrained step cost ({}, {} params, {} engaged tokens x 10): greedy {a:.3} ms, sampled \
         k40 {s:.3} ms ({:+.1}%), constrained {c:.3} ms ({:+.1}% vs greedy, {:+.1}% vs sampled); \
         fallbacks {} of {}",
        std::env::var("AXON_CONSTRAIN_TEST_MODEL")
            .unwrap_or_else(|_| crate::testutil::MODEL.into()),
        m.total_params(),
        ids.len(),
        100.0 * (s - a) / a,
        100.0 * (c - a) / a,
        100.0 * (c - s) / s,
        f1 - f0,
        10 * ids.len()
    );
}
