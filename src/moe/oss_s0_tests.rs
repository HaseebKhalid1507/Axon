//! gpt-oss experts on the GPU, stage 0 (specs/gptoss-gpu-experts.md §1 (b),
//! (c)): measurements, no assertions on speed. Ignored: run them by name on
//! a GPU with the real model (`AXON_TEST_GPTOSS`), e.g.
//! `axon-test oss_s0_ --ignored --nocapture --test-threads 1`.
//!
//! - `oss_s0_h2d`: one MoE layer's three expert tensors (423 MB) host to
//!   device by the three paths streamed prefill could take: from the file
//!   mapping as it is (pageable), through a page-locked staging buffer
//!   (`host_alloc` + a pool memcpy), and from the mapping page-locked in
//!   place (`Gpu::register`, read-only; it may be refused on a file mapping).
//! - `oss_s0_vram`: what the GPU half takes at ctx 8192 and 32768 (weights,
//!   the cache with its batch scratch, `MoeBufs`, and `ResBufs` at the batch
//!   max), and how many 403.3 MiB expert layers the rest holds on this card,
//!   by `place_resident`'s budget.

use crate::cuda::Kernels;
use crate::cuda::qwen2::experts::ResBufs;
use crate::cuda::qwen2::{CacheKv, CudaQwen2};
use crate::gguf::Tensor;
use crate::lm::Lm;
use std::time::Instant;

const MIB: f64 = (1u64 << 20) as f64;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn gbs(bytes: usize, s: f64) -> f64 {
    bytes as f64 / s / 1e9
}

/// Layer l's three expert tensors.
fn layer_exps(m: &crate::gguf::Model, l: usize) -> Vec<&Tensor> {
    ["gate", "up", "down"]
        .iter()
        .map(|p| {
            let name = format!("blk.{l}.ffn_{p}_exps.weight");
            m.tensors
                .iter()
                .find(|t| t.name == name)
                .expect("expert tensor")
        })
        .collect()
}

#[test]
#[ignore]
fn oss_s0_h2d() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::gptoss_model() else {
        return;
    };
    let l: usize = std::env::var("AXON_S0_LAYER").map_or(12, |v| v.parse().unwrap());
    let reps: usize = std::env::var("AXON_S0_REPS").map_or(7, |v| v.parse().unwrap());
    let data: Vec<&[u8]> = layer_exps(&m, l)
        .into_iter()
        .map(|t| m.tensor_data(t))
        .collect();
    let total: usize = data.iter().map(|d| d.len()).sum();
    // Fault the pages in first (the file is in the page cache after a
    // read; the copies then measure the bus, not the disk).
    let t = Instant::now();
    let mut sum = 0u64;
    for d in &data {
        sum += d.iter().step_by(4096).map(|&b| b as u64).sum::<u64>();
    }
    eprintln!(
        "s0 h2d: layer {l}, {} tensors, {total} bytes ({:.1} MiB), prefault {:.1} ms (sum {sum})",
        data.len(),
        total as f64 / MIB,
        t.elapsed().as_secs_f64() * 1e3
    );
    let dst = g.alloc(total).unwrap();
    let s = g.new_stream().unwrap();
    let copy = |srcs: &[*const u8]| -> f64 {
        let t = Instant::now();
        let mut off = 0;
        for (p, d) in srcs.iter().zip(&data) {
            // SAFETY: each source holds d.len() bytes, alive and unchanged
            // until the sync below.
            unsafe { s.h2d(&dst, off, *p, d.len()).unwrap() };
            off += d.len();
        }
        s.sync().unwrap();
        t.elapsed().as_secs_f64()
    };
    let report = |what: &str, v: &mut Vec<f64>| {
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(0.0, f64::max);
        let md = median(v);
        eprintln!(
            "S0-H2D\t{what}\tmedian {:.2} GB/s\t[{:.2}-{:.2}]\t{:.2} ms\t({} reps)",
            gbs(total, md),
            gbs(total, mx),
            gbs(total, mn),
            md * 1e3,
            v.len()
        );
    };
    // (1) Pageable: the mapping as it is (the driver stages it).
    let srcs: Vec<*const u8> = data.iter().map(|d| d.as_ptr()).collect();
    copy(&srcs);
    let mut v: Vec<f64> = (0..reps).map(|_| copy(&srcs)).collect();
    report("pageable (mapping as is)", &mut v);
    // (2) Pinned staging: a pool memcpy into host_alloc memory, then the copy.
    {
        let mut hb = g.host_alloc(total).unwrap();
        let fill = |hb: &mut crate::cuda::HostBuf| -> f64 {
            let b = hb.bytes_mut().unwrap();
            let t = Instant::now();
            let mut off = 0;
            for d in &data {
                const CH: usize = 4 << 20;
                crate::pool::for_each_chunk_mut(&mut b[off..off + d.len()], CH, |c, out| {
                    out.copy_from_slice(&d[c * CH..c * CH + out.len()])
                });
                off += d.len();
            }
            t.elapsed().as_secs_f64()
        };
        fill(&mut hb);
        let mut vm: Vec<f64> = (0..reps).map(|_| fill(&mut hb)).collect();
        report("pinned staging: host memcpy (pool) only", &mut vm);
        let base = hb.bytes_mut().unwrap().as_ptr();
        let mut off = 0;
        let hsrc: Vec<*const u8> = data
            .iter()
            .map(|d| {
                // SAFETY: within hb's total bytes.
                let p = unsafe { base.add(off) };
                off += d.len();
                p
            })
            .collect();
        copy(&hsrc);
        let mut vc: Vec<f64> = (0..reps).map(|_| copy(&hsrc)).collect();
        report("pinned staging: H2D copy only", &mut vc);
        let mut vs: Vec<f64> = (0..reps).map(|_| fill(&mut hb) + copy(&hsrc)).collect();
        report("pinned staging: memcpy then copy (serial)", &mut vs);
    }
    // (3) The mapping page-locked in place (read-only): the tensors' whole
    // pages, adjacent spans merged (a page registered twice is an error).
    let page = 4096usize;
    let mut spans: Vec<(usize, usize)> = data
        .iter()
        .map(|d| {
            let a = d.as_ptr() as usize;
            (a & !(page - 1), (a + d.len()).div_ceil(page) * page)
        })
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (lo, hi) in spans {
        match merged.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    let reg_bytes: usize = merged.iter().map(|(lo, hi)| hi - lo).sum();
    let t = Instant::now();
    let pins: Vec<_> = merged
        .iter()
        .map(|&(lo, hi)| {
            // SAFETY: lo..hi is whole pages of the tensors' spans in the
            // file mapping (page-granular), which outlives the pins
            // (dropped below, before `m`).
            unsafe { g.register(lo as *const u8, hi - lo, true) }
        })
        .collect();
    let reg_s = t.elapsed().as_secs_f64();
    match pins.iter().find_map(|p| p.as_ref().err()) {
        Some(e) => eprintln!("S0-H2D\tregistered mapping\tREFUSED: {e}"),
        None => {
            eprintln!(
                "S0-H2D\tregistered mapping: cuMemHostRegister of {:.1} MiB in {} span(s) took {:.1} ms",
                reg_bytes as f64 / MIB,
                merged.len(),
                reg_s * 1e3
            );
            copy(&srcs);
            let mut v: Vec<f64> = (0..reps).map(|_| copy(&srcs)).collect();
            report("registered mapping (read-only)", &mut v);
            let t = Instant::now();
            drop(pins);
            eprintln!(
                "S0-H2D\tregistered mapping: unregister took {:.1} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

#[test]
#[ignore]
fn oss_s0_vram() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let Some(m) = crate::testutil::gptoss_model() else {
        return;
    };
    let km = Kernels::module_for(&g, &m).unwrap();
    let free = || g.info().unwrap().free_bytes as f64 / MIB;
    let info = g.info().unwrap();
    // One layer's experts: three n_e x rows x cols MXFP4 tensors.
    let layer: usize = layer_exps(&m, 0)
        .iter()
        .map(|t| m.tensor_data(t).len())
        .sum();
    eprintln!(
        "S0-VRAM\tdevice {} ({:.0} MiB total, {:.0} free before), one expert layer {:.1} MiB",
        info.name,
        info.total_bytes as f64 / MIB,
        free(),
        layer as f64 / MIB
    );
    for ctx in [8192, 32768] {
        let f0 = free();
        let net = CudaQwen2::with_kv_moe(&g, &km, &m, ctx, CacheKv::F16, false, true).unwrap();
        let f1 = free();
        let c = Lm::new_cache(&net, ctx);
        let f2 = free();
        let max = CudaQwen2::batch_max(&c);
        let mb = net.moe_bufs(max).unwrap();
        let f3 = free();
        let d = net.moe.unwrap();
        let rb = ResBufs::bytes(max, net.dim, d.ffn, d.k) as f64 / MIB;
        let headroom = 512.0;
        let left = f3 - rb - headroom;
        eprintln!(
            "S0-VRAM\tctx {ctx}\tweights {:.0} MiB (weight_bytes {:.0}), cache + batch scratch \
             (max {max}) {:.0}, MoeBufs {:.0}, ResBufs at {max} {rb:.0}, headroom {headroom:.0}; \
             free after {f3:.0} -> {left:.0} for experts = {} layers of {:.1} MiB",
            f0 - f1,
            net.weight_bytes() as f64 / MIB,
            f1 - f2,
            f2 - f3,
            (left.max(0.0) * MIB / layer as f64).floor(),
            layer as f64 / MIB
        );
        drop(mb);
        drop(c);
        drop(net);
    }
}
