//! YaRN measurements (ignored: run by hand, `--ignored --exact`), on a GPU
//! with the model at $AXON_YARN_MODEL (default Qwen2.5-7B-Instruct
//! Q4_K_M) and text from $AXON_YARN_DATA (default yarn-data/ in the
//! worktree: Project Gutenberg's Sherlock Holmes and Pride and Prejudice,
//! not committed).
//!
//! `yarn_needle`: a pass key hidden at several depths of a book excerpt of
//! $AXON_YARN_LENS tokens (default 40000,64000,100000), with and without
//! `--rope-scaling yarn --rope-scale 4` (q8 cache), greedy answer checked.
//! `yarn_short_ctx`: next-token NLL, KL(plain || yarn) and top-1 agreement
//! of the two over windows of $AXON_YARN_PPL_LENS tokens (default
//! 2048,4096,8192), f16 cache, every position decoded.

use super::*;
use crate::cuda::Kernels;
use crate::gguf::Model;

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.into())
}

fn list(k: &str, d: &str) -> Vec<f64> {
    env_or(k, d)
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

fn model(yarn: bool) -> Option<Model> {
    let rel = env_or("AXON_YARN_MODEL", "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf");
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&rel);
    if !p.exists() {
        eprintln!("skipping: {} not present", p.display());
        return None;
    }
    let mut m = crate::gguf::load_with(p.to_str().unwrap(), Some(false)).unwrap();
    if yarn {
        let args = crate::rope::RopeArgs {
            scaling: Some("yarn".into()),
            factor: Some(env_or("AXON_YARN_FACTOR", "4").parse().unwrap()),
            ..Default::default()
        };
        eprintln!("{}", args.apply(&mut m).unwrap().join("\n"));
        crate::model::check_supported(&m).unwrap();
    }
    Some(m)
}

fn text(name: &str) -> String {
    let dir = env_or(
        "AXON_YARN_DATA",
        concat!(env!("CARGO_MANIFEST_DIR"), "/yarn-data"),
    );
    let t = std::fs::read_to_string(format!("{dir}/{name}")).unwrap();
    // Past the Gutenberg header.
    let start = t.find("*** START").map_or(0, |i| i + 200);
    t[start..].replace("\r\n", "\n")
}

#[test]
#[ignore]
fn yarn_needle() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let lens: Vec<usize> = list("AXON_YARN_LENS", "40000,64000,100000")
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let depths = list("AXON_YARN_DEPTHS", "0.1,0.5,0.9");
    let modes = env_or("AXON_YARN_MODES", "yarn,plain");
    let max = *lens.iter().max().unwrap() + 64;
    let book = text("sherlock.txt");
    let mut rows = Vec::new();
    for mode in modes.split(',') {
        let Some(m) = model(mode == "yarn") else {
            return;
        };
        let tok = crate::tokenizer::Tokenizer::from_gguf(&m);
        let mut hay = Vec::new();
        while hay.len() < max {
            hay.extend(tok.encode(&book));
        }
        let net = CudaQwen2::with_kv(&g, &km, &m, max, CacheKv::Q8 { group: 0 }, false).unwrap();
        let head = tok.encode_special(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\
             Below is a long excerpt from a book. Somewhere in it is a secret pass key. \
             Read it all, then answer the question after it.\n\n",
        );
        let tail = tok.encode_special(
            "\n\nWhat is the secret pass key hidden in the text above? Answer with the number only.\
             <|im_end|>\n<|im_start|>assistant\n",
        );
        for &len in &lens {
            for (di, &depth) in depths.iter().enumerate() {
                let key = 10000 + (len * 7 + di * 7919) % 89999;
                let needle = tok.encode(&format!(
                    "\n\nThe secret pass key is {key}. Remember it: the pass key is {key}.\n\n"
                ));
                let n_hay = len - head.len() - tail.len() - needle.len();
                let at = (n_hay as f64 * depth) as usize;
                let mut ids = head.clone();
                ids.extend_from_slice(&hay[..at]);
                ids.extend_from_slice(&needle);
                ids.extend_from_slice(&hay[at..n_hay]);
                ids.extend_from_slice(&tail);
                let mut c = net.new_cache(ids.len() + 17);
                let t0 = std::time::Instant::now();
                let (&last, rest) = ids.split_last().unwrap();
                net.feed_batch(rest, 0, &mut c);
                let secs = t0.elapsed().as_secs_f64();
                let mut t = last;
                let mut out = Vec::new();
                for i in 0..12 {
                    t = net.forward_argmax(t, rest.len() + i, &mut c);
                    if t == tok.eos() || tok.decode(&[t]).contains("<|im_end|>") {
                        break;
                    }
                    out.push(t);
                }
                let ans = tok.decode(&out);
                let ok = ans.contains(&key.to_string());
                let row = format!(
                    "needle {mode:5} len {:6} depth {depth:.1} key {key} {} answer {:?} (prefill {secs:.1} s)",
                    ids.len(),
                    if ok { "PASS" } else { "FAIL" },
                    ans.trim()
                );
                eprintln!("{row}");
                rows.push(row);
            }
        }
    }
    eprintln!("== summary\n{}", rows.join("\n"));
}

fn argmax64(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// log-softmax in f64.
fn lsm(l: &[f32]) -> Vec<f64> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = l.iter().map(|&x| (x as f64 - m).exp()).sum();
    l.iter().map(|&x| x as f64 - m - s.ln()).collect()
}

#[test]
#[ignore]
fn yarn_short_ctx() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let lens: Vec<usize> = list("AXON_YARN_PPL_LENS", "2048,4096,8192")
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let windows = env_or("AXON_YARN_WINDOWS", "2").parse::<usize>().unwrap();
    let (Some(mp), Some(my)) = (model(false), model(true)) else {
        return;
    };
    let tok = crate::tokenizer::Tokenizer::from_gguf(&mp);
    let max = *lens.iter().max().unwrap();
    let mut ids = Vec::new();
    for f in env_or("AXON_YARN_PPL_TEXTS", "pride.txt,sherlock.txt").split(',') {
        ids.push(tok.encode(&text(f)));
    }
    let plain = CudaQwen2::new(&g, &km, &mp, max + 1).unwrap();
    let yarn = CudaQwen2::new(&g, &km, &my, max + 1).unwrap();
    let mut rows = Vec::new();
    for &len in &lens {
        let (mut np, mut ny, mut kl, mut agree, mut n) = (0f64, 0f64, 0f64, 0usize, 0usize);
        for (ti, t) in ids.iter().enumerate() {
            for w in 0..windows {
                let s = w * max;
                let win = &t[s..s + len];
                let mut cp = plain.new_cache(len + 1);
                let mut cy = yarn.new_cache(len + 1);
                for p in 0..len - 1 {
                    let lp = lsm(&plain.forward(win[p], p, &mut cp));
                    let ly = lsm(&yarn.forward(win[p], p, &mut cy));
                    let nx = win[p + 1] as usize;
                    np -= lp[nx];
                    ny -= ly[nx];
                    kl += lp
                        .iter()
                        .zip(&ly)
                        .map(|(a, b)| a.exp() * (a - b))
                        .sum::<f64>();
                    agree += usize::from(argmax64(&lp) == argmax64(&ly));
                    n += 1;
                }
                eprintln!("len {len} text {ti} window {w}: done ({n} positions so far)");
            }
        }
        let nf = n as f64;
        let (pp, py) = ((np / nf).exp(), (ny / nf).exp());
        let row = format!(
            "short-ctx len {len:5}: {n} positions, ppl plain {pp:.4} yarn {py:.4} ({:+.2}%), \
             mean KL(plain||yarn) {:.5} nats, top-1 agreement {:.2}%",
            (py / pp - 1.0) * 100.0,
            kl / nf,
            agree as f64 * 100.0 / nf
        );
        eprintln!("{row}");
        rows.push(row);
    }
    eprintln!("== summary\n{}", rows.join("\n"));
}
