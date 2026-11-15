//! gpt-oss's MXFP4 expert twins (oss.cu; specs/gptoss-gpu-experts.md PR A)
//! == the CPU, bit for bit:
//! - `oss_twin_swiglu_matches_cpu`: `swiglu_oai_twin` vs `swiglu_oai1_f`
//!   over 1.25 M inputs (both clamps engaged, NaN, inf, -0), F true and
//!   false.
//! - `oss_twin_dot_matches_cpu`: `moe_gemm_mx` and `moe_gemv_mx` alone
//!   (random weight bytes, extreme and subnormal e8m0 scales, a partial K
//!   chunk and row group, tiles of 1..70 tokens) vs `quant::dot_row_q8`
//!   plus the bias.
//! - `cuda_experts_oss_fixture_match_cpu` / `cuda_experts_oss_match_cpu`:
//!   a whole layer through `CudaExperts` (biases, `swiglu_oai_quant_tw`)
//!   vs `Qwen2::moe_layer_batch` at nb 1, 7, 48, 129 and 1152, on the
//!   gpt-oss fixture and on the real model's layers (AXON_TEST_GPTOSS).
//!   The CPU's F is the host's (`vmath::fused()`): the swiglu test covers
//!   the other one.
//!
//! Each test also shows its inputs tell the wrong numerics apart (teeth):
//! plain SwiGLU and a NaN-blind clamp, a sequential lane sum instead of
//! hsum8, the layer without its biases.
//!
//! `cuda_experts_oss_bench` (ignored) is PR A's speed gate.

use super::tests::{OSS_NBS, Rng, check_layer_at, diffs, key};
use super::*;
use crate::cuda::Kernels;
use crate::model::Qwen2;
use crate::model::gptoss_tests::{OssDims, Ts, oss_fixture_dims};
use crate::quant::QuantizedAct;

/// (layers, dim, q heads, kv heads, head dim, expert ffn, experts, top-k,
/// vocab, window): dim 320 = 10 blocks (moe_gemm_mx's 8-block chunks and a
/// partial one), ffn 96 (a partial 64-row group; 3 blocks), 8 experts, top
/// 4.
const FX: OssDims = (3, 320, 4, 2, 16, 96, 8, 4, 64, 4);

fn fixture(edit: impl FnOnce(&mut Ts)) -> Model {
    oss_fixture_dims(FX, edit, |_| {})
}

/// The oss module (gpt-oss's kernels) of the fixture.
fn oss_module<'g>(g: &'g Gpu) -> Module<'g> {
    Kernels::module_for(g, &fixture(|_| {})).unwrap()
}

fn rng_bits(rng: &mut Rng) -> f32 {
    f32::from_bits(rng.next() as u32)
}

#[test]
fn oss_twin_swiglu_matches_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = oss_module(&g);
    let mut rng = Rng(0x051a_c0de);
    // The clamps' edges and the special values, against each other and
    // against everything else; then ranges that engage each branch, then
    // random bits.
    let edge = [
        0.0f32,
        -0.0,
        7.0,
        -7.0,
        f32::from_bits(7.0f32.to_bits() + 1),
        f32::from_bits(7.0f32.to_bits() - 1),
        f32::from_bits((-7.0f32).to_bits() + 1),
        f32::from_bits((-7.0f32).to_bits() - 1),
        1.0,
        -1.0,
        60.0,
        -60.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::MIN_POSITIVE,
        1e-40,
        f32::MAX,
        f32::MIN,
    ];
    let n = 1_250_000;
    let mut gs = Vec::with_capacity(n);
    let mut us = Vec::with_capacity(n);
    for &a in &edge {
        for &b in &edge {
            gs.push(a);
            us.push(b);
        }
    }
    while gs.len() < n {
        let (a, b) = match gs.len() % 4 {
            0 => (rng.f() * 12.0, rng.f() * 12.0),
            1 => (rng.f() * 100.0, rng.f() * 100.0),
            2 => (rng.f() * 2.0, rng.f() * 9.0),
            _ => (rng_bits(&mut rng), rng_bits(&mut rng)),
        };
        gs.push(a);
        us.push(b);
    }
    let (gb, ub) = (g.upload(&gs).unwrap(), g.upload(&us).unwrap());
    let ob = g.alloc(n * 4).unwrap();
    let f = km.function("swiglu_oai_twin_k").unwrap();
    for fused in [true, false] {
        // SAFETY: swiglu_oai_twin_k's signature; n floats each.
        unsafe {
            g.launch(
                &f,
                (n.div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &[
                    gb.arg(),
                    ub.arg(),
                    Arg::I32(n as i32),
                    Arg::I32(fused as i32),
                    ob.arg(),
                ],
            )
            .unwrap()
        };
        let got = ob.read::<f32>().unwrap();
        let cpu = |g: f32, u: f32| match fused {
            true => crate::vmath::swiglu_oai1_f::<true>(g, u),
            false => crate::vmath::swiglu_oai1_f::<false>(g, u),
        };
        let want: Vec<f32> = (0..n).map(|i| cpu(gs[i], us[i])).collect();
        let (bad, first) = diffs(&got, &want);
        assert_eq!(bad, 0, "F {fused}: {bad} of {n} differ, e.g. {first:?}");
        // Teeth: plain SwiGLU, and the clamp by fminf / fmaxf (NaN u -> -7).
        let plain = (0..n)
            .filter(|&i| key(got[i]) != key(crate::vmath::swiglu1_f::<true>(gs[i], us[i])))
            .count();
        let nan_blind = |g: f32, u: f32| {
            // fmaxf / fminf: NaN u becomes -7 (f32::max), unlike clamp.
            #[allow(clippy::manual_clamp)]
            let y = u.max(-7.0).min(7.0);
            cpu(g, if u.is_nan() { y } else { u })
        };
        let blind = (0..n)
            .filter(|&i| key(got[i]) != key(nan_blind(gs[i], us[i])))
            .count();
        assert!(
            plain > n / 2,
            "F {fused}: plain SwiGLU differs at {plain} only"
        );
        assert!(
            blind > 0,
            "F {fused}: the NaN-blind clamp is never told apart"
        );
        eprintln!(
            "swiglu_oai twin F {fused}: {n} equal the CPU; plain SwiGLU differs at {plain}, \
             a NaN-blind clamp at {blind}"
        );
    }
}

/// n_e random MXFP4 matrices of rows x cols: random qs bytes; e8m0 bytes
/// mostly 118..130, every 7th block from `extreme` (subnormal scales, 2^-1,
/// 2^32).
fn random_mx(rng: &mut Rng, n_e: usize, rows: usize, cols: usize) -> Vec<u8> {
    let extreme = [0u8, 1, 2, 3, 127, 128, 160];
    let nb = n_e * rows * cols / 32;
    let mut v = Vec::with_capacity(nb * 17);
    for b in 0..nb {
        let r = rng.next();
        v.push(match b % 7 {
            3 => extreme[(r >> 8) as usize % extreme.len()],
            _ => 118 + (r % 13) as u8,
        });
        for j in 0..16 {
            v.push((rng.next() >> (8 + j)) as u8);
        }
    }
    v
}

/// `quantize_b` of nb rows of n floats (h, token major) into act rows of
/// act_stride(n) bytes.
fn quantize_rows<'g>(g: &'g Gpu, km: &Module, h: &[f32], n: usize) -> DevBuf<'g> {
    let nb = h.len() / n;
    let ast = act_stride(n);
    let hb = g.upload(h).unwrap();
    let act = g.alloc(nb * ast).unwrap();
    act.zero().unwrap();
    // SAFETY: decode.cu quantize_b's signature; nb rows of each.
    unsafe {
        g.launch(
            &km.function("quantize_b").unwrap(),
            (n.div_ceil(256) as u32, nb as u32, 1),
            (32, 1, 1),
            0,
            &[
                hb.arg(),
                Arg::I32(n as i32),
                Arg::I32(n.is_multiple_of(256) as i32),
                act.arg(),
                Arg::U64(ast as u64),
            ],
        )
        .unwrap()
    };
    act
}

/// The CPU's MXFP4 chain spelled out on one row and the GPU's quantized
/// activation (act row bytes, n values), its lanes summed by `sum8`
/// (teeth: another order than hsum8 must differ somewhere).
fn chain(row: &[u8], act: &[u8], n: usize, sum8: impl Fn(&[f32; 8]) -> f32) -> f32 {
    let np = n.next_multiple_of(256);
    let q = &act[np..np + n];
    let d0 = np * 2 + np / 64 + np / 8;
    let kv = crate::quant::mxfp4_kvalues();
    let mut acc = [0f32; 8];
    for b in 0..n / 32 {
        let blk = &row[b * 17..b * 17 + 17];
        let e = blk[0] as u32;
        let s = f32::from_bits(if e < 2 {
            0x0020_0000 << e
        } else {
            (e - 1) << 23
        });
        let dx = f32::from_le_bytes(act[d0 + 4 * b..d0 + 4 * b + 4].try_into().unwrap());
        let d = s * dx;
        for (l, a) in acc.iter_mut().enumerate() {
            let mut sum = 0i32;
            for i in 4 * l..4 * l + 4 {
                let nib = if i < 16 {
                    blk[1 + i] & 0xf
                } else {
                    blk[1 + i - 16] >> 4
                };
                sum += kv[nib as usize] as i32 * q[b * 32 + i] as i8 as i32;
            }
            *a = d.mul_add(sum as f32, *a);
        }
    }
    sum8(&acc)
}

#[test]
fn oss_twin_dot_matches_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = oss_module(&g);
    let mut rng = Rng(0x6d78_646f_7400);
    // 3 experts of 96 x 2880 (gpt-oss's K: 90 blocks, a partial 8-block
    // chunk; a partial 64-row group); tiles of 1, 33 (32 + 1) and 70
    // tokens.
    let (n_e, rows, cols) = (3usize, 96usize, 2880usize);
    let raw = random_mx(&mut rng, n_e, rows, cols);
    let mats = Mats::new(&g, &km, &raw, MXFP4, rows, cols, n_e).unwrap();
    let rb = quant::row_bytes(MXFP4, cols);
    let counts = [1usize, 33, 70];
    let np: usize = counts.iter().sum();
    let mut off = vec![0i32];
    for c in counts {
        off.push(off.last().unwrap() + c as i32);
    }
    let h: Vec<f32> = (0..np * cols)
        .map(|i| rng.f() * if (i / cols) % 5 == 0 { 30.0 } else { 1.0 })
        .collect();
    let act = quantize_rows(&g, &km, &h, cols);
    let ast = act_stride(cols);
    let bias: Vec<f32> = (0..n_e * rows).map(|_| rng.f()).collect();
    let (bb, ob) = (g.upload(&bias).unwrap(), g.upload(&off).unwrap());
    let out = g.alloc(np * rows * 4).unwrap();
    // SAFETY: act np rows, off n_e + 1 offsets of np, out np rows, bias
    // n_e * rows floats; km the oss module.
    unsafe {
        launch_gemm_mx(
            &g,
            &km,
            &mats,
            act.arg(),
            ast,
            ob.arg(),
            n_e,
            np,
            out.arg(),
            rows,
            bb.arg(),
        )
        .unwrap()
    };
    let got = out.read::<f32>().unwrap();
    let qa: Vec<QuantizedAct> = h
        .chunks_exact(cols)
        .map(|x| QuantizedAct::for_types(x, &[MXFP4]))
        .collect();
    let row = |e: usize, r: usize| &raw[(e * rows + r) * rb..(e * rows + r + 1) * rb];
    let mut want = vec![0f32; np * rows];
    for e in 0..n_e {
        for s in off[e] as usize..off[e + 1] as usize {
            for r in 0..rows {
                want[s * rows + r] =
                    quant::dot_row_q8(MXFP4, row(e, r), &qa[s]) + bias[e * rows + r];
            }
        }
    }
    let (bad, first) = diffs(&got, &want);
    assert_eq!(
        bad,
        0,
        "moe_gemm_mx: {bad} of {} differ, e.g. {first:?}",
        np * rows
    );
    // Teeth: the chain spelled out == dot_row_q8 with hsum8, and a
    // sequential lane sum differs from the GPU somewhere.
    let acts = act.read::<u8>().unwrap();
    let hsum8 = |l: &[f32; 8]| ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]));
    let seq = |l: &[f32; 8]| l.iter().fold(0.0f32, |a, &v| a + v);
    let mut seq_bad = 0;
    for e in 0..n_e {
        for s in off[e] as usize..off[e + 1] as usize {
            let a = &acts[s * ast..(s + 1) * ast];
            for r in 0..rows {
                let c = chain(row(e, r), a, cols, hsum8);
                let d = quant::dot_row_q8(MXFP4, row(e, r), &qa[s]);
                assert_eq!(key(c), key(d), "the chain spelled out != dot_row_q8");
                let v = chain(row(e, r), a, cols, seq) + bias[e * rows + r];
                seq_bad += (key(v) != key(got[s * rows + r])) as usize;
            }
        }
    }
    assert!(seq_bad > 0, "a sequential lane sum is never told apart");
    // moe_gemv_mx: slots (ids 2, 0, 2) on act rows 0, 1, 2 (aslot = one
    // row), and one activation for every slot (aslot 0).
    let ids = [2i32, 0, 2];
    let ib = g.upload(&ids).unwrap();
    let k = ids.len();
    for aslot in [ast, 0] {
        let gout = g.alloc(k * rows * 4).unwrap();
        // SAFETY: ids k ids < n_e; act holds k rows; gout k rows.
        unsafe {
            launch_gemv_mx(
                &g,
                &km,
                &mats,
                ib.arg(),
                k,
                act.arg(),
                aslot,
                gout.arg(),
                rows,
                bb.arg(),
            )
            .unwrap()
        };
        let got = gout.read::<f32>().unwrap();
        let mut want = vec![0f32; k * rows];
        for (i, &e) in ids.iter().enumerate() {
            let s = if aslot == 0 { 0 } else { i };
            for r in 0..rows {
                want[i * rows + r] = quant::dot_row_q8(MXFP4, row(e as usize, r), &qa[s])
                    + bias[e as usize * rows + r];
            }
        }
        let (bad, first) = diffs(&got, &want);
        assert_eq!(
            bad, 0,
            "moe_gemv_mx aslot {aslot}: {bad} differ, e.g. {first:?}"
        );
    }
    eprintln!(
        "oss twin dot: moe_gemm_mx {} and moe_gemv_mx {} outputs equal the CPU; a sequential \
         lane sum differs at {seq_bad}",
        np * rows,
        2 * k * rows
    );
}

/// Layers `layers` of `m` through `upload` (biases too), k from the
/// metadata.
fn upload_oss<'g>(
    g: &'g Gpu,
    km: &'g Module<'g>,
    m: &Model,
    layers: &[usize],
) -> Vec<CudaExperts<'g>> {
    let key = format!("{}.expert_used_count", m.arch);
    let k = crate::model::meta_num(m, &key).expect("expert_used_count") as usize;
    layers
        .iter()
        .map(|&l| CudaExperts::upload(g, km, m, l, k).unwrap())
        .collect()
}

#[test]
fn cuda_experts_oss_fixture_match_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let m = fixture(|_| {});
    let km = Kernels::module_for(&g, &m).unwrap();
    // (`upload` takes the biases since stage B; it refused them in A.)
    let exs = upload_oss(&g, &km, &m, &[0, 1, 2]);
    let cpu = Qwen2::new(&m);
    for (ex, l) in exs.iter().zip([0, 1, 2]) {
        assert!(ex.bias.is_some() && ex.gate.qt == MXFP4 && ex.down.qt == MXFP4);
        check_layer_at(&g, ex, &cpu, l, "oss fixture", &OSS_NBS);
    }
    // Teeth: the CPU of the fixture without its expert biases differs.
    let zero = |ts: &mut Ts| {
        for t in ts.iter_mut().filter(|t| t.0.ends_with("_exps.bias")) {
            t.3.iter_mut().for_each(|b| *b = 0);
        }
    };
    let mz = fixture(zero);
    let cz = Qwen2::new(&mz);
    let (dim, l, nb) = (cpu.dim, 1, 7);
    let mut rng = Rng(0x7ee7);
    let h: Vec<f32> = (0..nb * dim).map(|_| rng.f()).collect();
    let hs: Vec<&[f32]> = h.chunks_exact(dim).collect();
    let routes: Vec<_> = hs.iter().map(|x| cpu.moe_route(l, x, false)).collect();
    let ids: Vec<i32> = routes
        .iter()
        .flat_map(|r| r.0.iter().map(|&e| e as i32))
        .collect();
    let ws: Vec<f32> = routes.iter().flat_map(|r| r.1.iter().copied()).collect();
    // (the decode path: any arch; nb invariance makes it the batch's bits)
    let got = exs[l].layer_gemv(&h, &ids, &ws).unwrap();
    let yz = cz.moe_layer_batch(l, &hs, &routes);
    let yz_t: Vec<f32> = (0..nb * dim)
        .map(|i| yz[(i % dim) * nb + i / dim])
        .collect();
    let (n, _) = diffs(&got, &yz_t);
    assert!(
        n > nb * dim / 2,
        "without biases the CPU differs at {n} only"
    );
}

/// The real model's layers 0, 1, 12 and 23 (AXON_TEST_GPTOSS).
#[test]
fn cuda_experts_oss_match_cpu() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::gptoss_model() else {
        return;
    };
    let layers = [0, 1, 12, 23];
    let km = Kernels::module_for(&g, &m).unwrap();
    let exs = upload_oss(&g, &km, &m, &layers);
    // The CPU's experts of those layers only (`new_experts_of` is deepseek2's).
    let skip: Vec<usize> = (0..24).filter(|l| !layers.contains(l)).collect();
    let cpu = Qwen2::new_experts_except(&m, &skip);
    for (ex, &l) in exs.iter().zip(&layers) {
        check_layer_at(&g, ex, &cpu, l, "gpt-oss-20b", &OSS_NBS);
    }
}

/// PR A's speed gate (spec §4: the grouped twin GEMM at nb 512 >= 25 T
/// MAC/s on a 4090; review: on the real shape). With the real model, layer
/// 12's matrices, else random ones of its shape (32 experts of 2880 x
/// 2880). moe_gemm_mx alone over three routings of nb 512 x top-4 = 2048
/// pairs: uniform (64 a expert), random top-4 (the router's spread), and
/// the spec's 4 experts (512 each); then the whole layer (`batched`: quantize,
/// sort, gather, three GEMMs, activation, sum) at nb 128, 512 and 1152, and
/// the decode step (nb 1). GPU time by events: 3 warm-ups, then the median
/// of 5 runs of 10. Run exclusive (`--timing`).
#[test]
#[ignore]
fn cuda_experts_oss_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let info = g.info().unwrap();
    if info.cc.0 < 8 {
        eprintln!("skipping: the batched path needs sm_80+");
        return;
    }
    let real = crate::testutil::gptoss_model();
    let fx = fixture(|_| {});
    let m = real.as_ref().unwrap_or(&fx);
    let km = Kernels::module_for(&g, m).unwrap();
    let mut rng = Rng(0xbe9c_0055);
    let (n_e, dim) = (32usize, 2880usize);
    let raw_owned;
    let raw: &[u8] = match &real {
        Some(m) => {
            let t = (m.tensors.iter())
                .find(|t| t.name == "blk.12.ffn_gate_exps.weight")
                .unwrap();
            m.tensor_data(t)
        }
        None => {
            raw_owned = random_mx(&mut rng, n_e, dim, dim);
            &raw_owned
        }
    };
    let what = if real.is_some() {
        "gpt-oss-20b layer 12 gate"
    } else {
        "random"
    };
    for f in ["moe_gemm_mx", "moe_gemv_mx", "moe_gu_mx"] {
        let (regs, local) = km.function(f).unwrap().regs_local().unwrap();
        eprintln!("BENCH {f}: {regs} registers, {local} bytes of local memory (spills)");
    }
    let mats = Mats::new(&g, &km, raw, MXFP4, dim, dim, n_e).unwrap();
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (nb, k) = (512usize, 4usize);
    let np = nb * k;
    let h: Vec<f32> = (0..np * dim).map(|_| rng.f()).collect();
    let act = quantize_rows(&g, &km, &h, dim);
    let ast = act_stride(dim);
    let out = g.alloc(np * dim * 4).unwrap();
    let uniform: Vec<usize> = vec![np / n_e; n_e];
    let mut random = vec![0usize; n_e];
    for _ in 0..nb {
        let mut pick = Vec::new();
        while pick.len() < k {
            let e = (rng.next() >> 20) as usize % n_e;
            if !pick.contains(&e) {
                pick.push(e);
            }
        }
        pick.iter().for_each(|&e| random[e] += 1);
    }
    let mut four = vec![0usize; n_e];
    four[..4].iter_mut().for_each(|c| *c = np / 4);
    for (name, counts) in [
        ("uniform", uniform),
        ("random top-4", random),
        ("4 experts", four),
    ] {
        let mut off = vec![0i32];
        for &c in &counts {
            off.push(off.last().unwrap() + c as i32);
        }
        let ob = g.upload(&off).unwrap();
        let run = |reps: usize| -> f64 {
            let t0 = g.record().unwrap();
            for _ in 0..reps {
                // SAFETY: act np rows, off of np pairs, out np rows.
                unsafe {
                    launch_gemm_mx(
                        &g,
                        &km,
                        &mats,
                        act.arg(),
                        ast,
                        ob.arg(),
                        n_e,
                        np,
                        out.arg(),
                        dim,
                        Arg::Ptr(0),
                    )
                    .unwrap()
                };
            }
            let t1 = g.record().unwrap();
            g.sync().unwrap();
            t0.ms_to(&t1).unwrap() as f64 / reps as f64
        };
        run(3);
        let ms = med((0..5).map(|_| run(10)).collect());
        let macs = (np * dim * dim) as f64;
        let tiles: usize = counts.iter().map(|c| c.div_ceil(MX_BN)).sum();
        eprintln!(
            "BENCH moe_gemm_mx {what} {name}: nb {nb} x top-{k} = {np} pairs, {tiles} tiles of \
             {MX_BN}: {ms:.3} ms, {:.2} T MAC/s ({})",
            macs / ms / 1e9,
            info.name
        );
    }
    // The whole layer and the decode step (real model only: biases).
    let Some(m) = real.as_ref() else {
        return;
    };
    let ex = CudaExperts::upload(&g, &km, m, 12, k).unwrap();
    for nb in [128usize, 512, 1152] {
        let h: Vec<f32> = (0..nb * dim).map(|_| rng.f()).collect();
        let mut ids = Vec::with_capacity(nb * k);
        for _ in 0..nb {
            let mut pick = Vec::new();
            while pick.len() < k {
                let e = ((rng.next() >> 20) % n_e as u64) as i32;
                if !pick.contains(&e) {
                    pick.push(e);
                }
            }
            ids.extend(pick);
        }
        let ws = vec![0.25f32; nb * k];
        ex.batched(&h, &ids, &ws, 3).unwrap();
        let ms = med((0..5)
            .map(|_| ex.batched(&h, &ids, &ws, 10).unwrap().1 as f64)
            .collect());
        let macs = (3 * nb * k * dim * dim) as f64;
        eprintln!(
            "BENCH oss layer 12 nb {nb}: {ms:.3} ms a layer ({:.2} T MAC/s of experts; x24 \
             layers = {:.1} ms)",
            macs / ms / 1e9,
            24.0 * ms
        );
    }
    let h: Vec<f32> = (0..dim).map(|_| rng.f()).collect();
    let (ids, ws) = ([3i32, 9, 17, 30], [0.25f32; 4]);
    ex.run_host(&h, &ids, &ws, 1, 3, false).unwrap();
    let ms = med((0..5)
        .map(|_| ex.run_host(&h, &ids, &ws, 1, 10, false).unwrap().1 as f64)
        .collect());
    let bytes = (3 * k * dim * (dim / 32) * 17) as f64;
    eprintln!(
        "BENCH oss layer 12 decode (nb 1): {:.1} us a layer ({:.0} GB/s of expert weights)",
        ms * 1e3,
        bytes / ms / 1e6
    );
}
