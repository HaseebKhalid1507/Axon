use super::*;
use crate::cuda::Kernels;
use crate::quant::{r8, r8q};

/// unpermute_r8 is the exact inverse of the CPU's repack, for every type
/// it handles, on arbitrary bytes (the repack is a permutation, with Q4_K's
/// packed 6-bit scales unpacked and Q8_0's quants offset: both bijective),
/// for several groups and chunks that start mid-matrix.
#[test]
fn unpermute_inverts_repack() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let f = km.function("unpermute_r8").unwrap();
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut byte = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as u8
    };
    for (qt, cols) in [(Q4_K, 512), (Q6_K, 768), (Q8_0, 96), (Q5_0, 64)] {
        let rows = 24;
        let rb = quant::row_bytes(qt, cols);
        let data: Vec<u8> = (0..rows * rb).map(|_| byte()).collect();
        let packed = if matches!(qt, Q4_K | Q6_K) {
            r8::repack(qt, &data, rows, cols)
        } else {
            r8q::repack(qt, &data, rows, cols)
        };
        let (b8, bb) = r8_block(qt);
        let nblk = cols / if matches!(qt, Q4_K | Q6_K) { 256 } else { 32 };
        let gb = nblk * b8;
        assert_eq!(packed.len(), rows / 8 * gb);
        let src = g.upload(&packed).unwrap();
        let dst = g.alloc(rows * rb).unwrap();
        dst.zero().unwrap();
        // Group 0 alone, then groups 1..3 in one launch.
        for (g0, ng) in [(0usize, 1usize), (1, 2)] {
            // SAFETY: unpermute_r8's signature; src holds the groups, dst
            // their rows.
            unsafe {
                g.launch(
                    &f,
                    ((ng * nblk) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    &[
                        Arg::I32(qt as i32),
                        src.arg_at(g0 * gb),
                        dst.arg_at(g0 * 8 * rb),
                        Arg::U64(rb as u64),
                        Arg::I32(nblk as i32),
                        Arg::I32(b8 as i32),
                        Arg::I32(bb as i32),
                        Arg::U64(ng as u64),
                    ],
                )
                .unwrap();
            }
        }
        let check = |what: &str| {
            let got: Vec<u8> = dst.read().unwrap();
            let bad = got.iter().zip(&data).position(|(a, b)| a != b);
            assert!(
                bad.is_none(),
                "{} {what}: byte {} differs",
                crate::gguf::qtype(qt).name,
                bad.unwrap()
            );
        };
        check("by chunks");
        // The grid-stride loop: all 3 groups by 1 and 2 blocks.
        for grid in [1u32, 2] {
            dst.zero().unwrap();
            // SAFETY: as above, every group at once.
            unsafe {
                g.launch(
                    &f,
                    (grid, 1, 1),
                    (128, 1, 1),
                    0,
                    &[
                        Arg::I32(qt as i32),
                        src.arg(),
                        dst.arg(),
                        Arg::U64(rb as u64),
                        Arg::I32(nblk as i32),
                        Arg::I32(b8 as i32),
                        Arg::I32(bb as i32),
                        Arg::U64((rows / 8) as u64),
                    ],
                )
                .unwrap();
            }
            check(&format!("grid {grid}"));
        }
    }
}

/// unpermute_r8 throughput on 32B-sized matrices (27648 x 5120 Q4_K and
/// Q6_K): GB/s of GGUF bytes written.
#[test]
#[ignore = "benchmark: unpermute_bench -- --ignored --nocapture"]
fn unpermute_bench() {
    let Some(g) = crate::testutil::gpu() else {
        return;
    };
    let km = Kernels::module(&g).unwrap();
    let f = km.function("unpermute_r8").unwrap();
    for qt in [Q4_K, Q6_K, Q8_0] {
        let (rows, cols) = (27648, 5120);
        let rb = quant::row_bytes(qt, cols);
        let (b8, bb) = r8_block(qt);
        let nblk = cols / if matches!(qt, Q4_K | Q6_K) { 256 } else { 32 };
        let src = g.alloc(rows / 8 * nblk * b8).unwrap();
        src.zero().unwrap();
        let dst = g.alloc(rows * rb).unwrap();
        for grid in [1024u32, 4096, 16384] {
            let run = || unsafe {
                g.launch(
                    &f,
                    (grid, 1, 1),
                    (128, 1, 1),
                    0,
                    &[
                        Arg::I32(qt as i32),
                        src.arg(),
                        dst.arg(),
                        Arg::U64(rb as u64),
                        Arg::I32(nblk as i32),
                        Arg::I32(b8 as i32),
                        Arg::I32(bb as i32),
                        Arg::U64((rows / 8) as u64),
                    ],
                )
                .unwrap()
            };
            run();
            let t0 = g.record().unwrap();
            for _ in 0..5 {
                run();
            }
            let t1 = g.record().unwrap();
            let ms = t0.ms_to(&t1).unwrap() / 5.0;
            eprintln!(
                "{} {} MB grid {grid}: {ms:.2} ms, {:.0} GB/s",
                crate::gguf::qtype(qt).name,
                (rows * rb) >> 20,
                (rows * rb) as f32 / ms / 1e6
            );
        }
    }
}
