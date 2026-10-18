//! Batched passes (prefill.cu): `feed_batch` and `verify_argmax` run nb
//! tokens through each layer together, so each weight row is read once per
//! batch instead of once per token.
//!
//! The contract is the CPU's: every token's KV cache rows and logits are
//! bit for bit what the one-token step (`CudaQwen2::step`) gives, whatever
//! the batch size or the token's place in it; prefill.cu's header says how.
//! So chat's cache reuse equals a fresh prefill, and drafting is lossless.
//!
//! Products of more than `SMALL` tokens run on int8 tensor cores (gemm2_*,
//! sm_80+); smaller ones (the verify pass of a draft) too, by gemvm_*, which
//! reads the weights once for the batch like a decode step. On GPUs without
//! int8 mma (sm < 80) gemvn, the decode GEMV with a token loop (dp4a), does
//! every batch, 16 tokens per pass over the weights: correct, not fast.
//!
//! The attention of a batch (sm_80+) is the decode kernels' canonical order
//! (decode.cu "Tensor-core attention"): one flash-attention launch per layer
//! (attn_fa_b) for the whole pass, or the decode kernel per query
//! (attn_blk_b) for a few queries (`att_per_query`).

use super::*;

/// Batches of up to this many tokens use gemvn (weight-read-once, CUDA
/// cores) even where the tensor-core GEMM exists.
pub(super) const SMALL: usize = 16;

/// The most tokens a gemvm kernel takes (gemvm32; `CudaQwen2::takes_gemvm`
/// picks it over gemm2 for the products too small to fill the GPU).
pub(super) const VM_MAX: usize = 32;

/// hopper.cu gemmh2's dynamic shared memory (GEMMH2_SMEM): 4 stages of
/// activation values (16 KiB) and two warpgroups' raw Q4_K rows, their dk /
/// b32 pieces, the barriers, 1 KiB of alignment slack.
pub(super) const GEMMH2_SMEM: usize = 4 * 16384 + 4 * 128 * 144 + 4 * 2 * 1280 + 2 * 4 * 8 + 1024;

/// Whether a Q4_K product of `rows` outputs over `nb` tokens goes to
/// gemmh2_q4_k by default: the row counts measured faster than gemm2s on an
/// H100 (3584 and 18944: +6-19 % in cuda_gemmh_bench at nb 512 and 1050,
/// +6-10 % in the 7B pass), with at least a full wave of its 128 x 64
/// blocks on `sms` SMs (the one loss measured, 512 rows at nb 512, had 32
/// blocks on 114 SMs).
pub(super) fn gemmh_plain(rows: usize, nb: usize, sms: usize) -> bool {
    rows >= 3584 && nb.div_ceil(64) * rows.div_ceil(128) >= sms
}

/// The device address in a pointer argument.
fn dev_ptr(a: Arg) -> u64 {
    match a {
        Arg::Ptr(p) => p,
        _ => unreachable!("not a pointer argument"),
    }
}
/// Query tokens per batched attention launch without tensor cores (sizes
/// `part_o`).
pub(super) const ATT_SUB: usize = 64;

/// Batches of up to this many queries take the decode attention per query
/// (attn_blk_b) rather than the flash kernel (attn_fa_b), which walks a
/// block's positions in order and so is latency-bound with few blocks of
/// rows: the crossover measured by cuda_attn_bench (RTX 5070 Ti Laptop: 64
/// wide heads ~48-64 queries, 128 wide ~16-24). Either gives the same bits.
pub(super) fn att_per_query(head_dim: usize) -> usize {
    if head_dim == 64 { 48 } else { SMALL }
}
/// Tokens per output-head launch in `verify_argmax` (sizes the logits).
pub(super) const HEAD_SUB: usize = 16;

/// attn_fa_b's launch shapes (prefill.cu): `Full` attn_fa_b, `Small`
/// attn_fa_bs (half the rows: short batches), `Kv` attn_fa_k (4 warps, 64
/// rows, K and V copied apart, several blocks per SM). Each row's ops are
/// the same in all of them (rows are independent, prefill.cu attn_fa_bt):
/// the shape is only speed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fa {
    Full = 0,
    Small = 1,
    Kv = 2,
}

/// attn_fa_b's variant v for head size hd and q8 scale group kg (0: the f16
/// cache; Kv is f16 only): (rows per block, threads, dynamic shared memory:
/// f16, its stages of K and V tiles of ATT_BLK rows (Kv: K, K and V); q8,
/// one f16 tile and per stage the raw rows and their scales).
pub(super) fn fa_shape(hd: usize, v: Fa, kg: usize) -> (usize, usize, usize) {
    let tile = ATT_BLK * hd * 2;
    let (rows, stages) = match (hd, v, kg) {
        (_, Fa::Kv, _) => return (64, 128, 3 * tile),
        (64, Fa::Full, _) => (64, 3),
        (64, Fa::Small, _) => (32, 2),
        (_, Fa::Full, _) => (128, 2),
        (_, Fa::Small, 0) => (64, 1),
        (_, Fa::Small, _) => (64, 2),
    };
    let smem = match kg {
        0 => stages * 2 * tile,
        _ => tile + stages * 2 * ATT_BLK * (hd + hd / kg * 4),
    };
    (rows, rows * 2, smem)
}

/// Blocks of variant v resident per SM (prefill.cu's launch bounds).
fn fa_per_sm(hd: usize, v: Fa) -> usize {
    match (hd, v) {
        (64, Fa::Small) | (64, Fa::Kv) => 3,
        (64, Fa::Full) | (_, Fa::Small) | (_, Fa::Kv) => 2,
        (_, Fa::Full) => 1,
    }
}

/// The work of one attn_fa_b launch of nb queries in variant v, in rows per
/// SM, padded to the waves the blocks run in: whole waves of per-SM blocks
/// on `sms` SMs, then the last partial wave as the most blocks it puts on
/// an SM (a block runs as fast with fewer neighbours). Blocks are equally
/// long at long context. Multiplied by the positions per row, this is
/// proportional to the kernel time per variant (MEASURED on 20, 128 and 170
/// SMs, lcpre NOTES).
pub(super) fn fa_work(hd: usize, v: Fa, n_kv: usize, group: usize, nb: usize, sms: usize) -> f64 {
    let (rows, _, _) = fa_shape(hd, v, 0);
    let per = fa_per_sm(hd, v);
    let blocks = n_kv * (nb * group).div_ceil(rows);
    let (full, rem) = (blocks / (per * sms), blocks % (per * sms));
    (full * per * rows + rem.div_ceil(sms) * rows) as f64
}

/// Variant v's speed at full occupancy relative to Full's (`fa_work` per
/// time) on compute capability cc (major * 10 + minor): the slowest seen on
/// a GPU of that architecture (cuda_fa_bench at 16k and 32k positions, 1016
/// to 1121 queries: sm_89 RTX 4050 Laptop 20 SMs and RTX 4090 128, sm_120
/// RTX 5070 Ti Laptop 46 and RTX 5090 170; lcpre NOTES). None: never picked -
/// Kv at head size 64 on sm_120 (5090 1.07-1.10, but the laptop's 0.88 at
/// 1121 queries), and every architecture not measured (Full and Small, as
/// before).
pub(super) fn fa_eff(cc: u32, hd: usize, v: Fa) -> Option<f64> {
    match (cc, hd, v) {
        (_, _, Fa::Full) => Some(1.0),
        (89, 128, Fa::Small) => Some(0.85),
        (89, 128, Fa::Kv) => Some(0.94),
        (89, 64, Fa::Kv) => Some(1.01),
        (120, 128, Fa::Small) => Some(0.93),
        (120, 128, Fa::Kv) => Some(1.0),
        _ => None,
    }
}

/// Architectures whose waves `fa_work` was checked on (`fa_eff`): passes
/// are sized for them only.
fn fa_measured(cc: u32) -> bool {
    matches!(cc, 89 | 120)
}

/// An alternative variant must look this much faster than Full.
const FA_MARGIN: f64 = 0.97;

/// What the flash attention launch depends on: the GPU's compute
/// capability (major * 10 + minor) and SMs, the head size, the q8 scale
/// group (0: f16), kv heads and query heads per kv head.
#[derive(Clone, Copy, Debug)]
pub(super) struct FaGeom {
    pub(super) cc: u32,
    pub(super) sms: usize,
    pub(super) hd: usize,
    pub(super) kg: usize,
    pub(super) n_kv: usize,
    pub(super) group: usize,
}

/// The flash attention variant for nb queries: Small when Full's blocks
/// don't fill the GPU (short batches, as before), else the variant with the
/// least `fa_work` / `fa_eff`: Full unless another looks FA_MARGIN better.
/// The q8 cache: Full or Small.
pub(super) fn fa_pick(g: FaGeom, nb: usize) -> Fa {
    let full = fa_shape(g.hd, Fa::Full, g.kg).0;
    if g.n_kv * (nb * g.group).div_ceil(full) < g.sms {
        return Fa::Small;
    }
    if g.kg > 0 {
        return Fa::Full;
    }
    let base = fa_work(g.hd, Fa::Full, g.n_kv, g.group, nb, g.sms);
    let mut best = (Fa::Full, base);
    for v in [Fa::Small, Fa::Kv] {
        if let Some(e) = fa_eff(g.cc, g.hd, v) {
            let t = fa_work(g.hd, v, g.n_kv, g.group, nb, g.sms) / e;
            if t <= FA_MARGIN * base && t < best.1 {
                best = (v, t);
            }
        }
    }
    best.0
}

/// `fa_pick`'s launch time for nb queries, in Full's rows per SM.
pub(super) fn fa_time(g: FaGeom, nb: usize) -> f64 {
    let v = fa_pick(g, nb);
    // Small for a short batch has no measured speed: Full's (it isn't compared).
    fa_work(g.hd, v, g.n_kv, g.group, nb, g.sms) / fa_eff(g.cc, g.hd, v).unwrap_or(1.0)
}

/// Prompts reaching this many positions get their passes sized for the
/// attention's waves (`fa_passes`); shorter ones keep `passes`.
pub(super) const FA_LONG: usize = 8192;

/// The passes of n tokens from position start with at most max per pass,
/// for the flash attention: `passes`, or for a long prompt (start + n >=
/// FA_LONG) with 128-wide heads a few more, equal, passes when that puts the attention's
/// blocks in fewer, fuller waves (`fa_work` of the variant `fa_pick` takes,
/// times the positions each pass reads) and doesn't add GEMM token tiles
/// (128 tokens). Results don't depend on it.
pub(super) fn fa_passes(
    n: usize,
    max: usize,
    start: usize,
    g: FaGeom,
) -> Vec<std::ops::Range<usize>> {
    let np0 = n.div_ceil(max).max(1);
    // 64-wide heads: their 2- and 3-per-SM blocks don't follow fa_work's
    // partial waves on the laptop (MEASURED, bella: 0.5B pp32512 in 32
    // passes of 1016 instead of 29 of 1121 was 6 % slower).
    if start + n < FA_LONG || !fa_measured(g.cc) || g.hd != 128 {
        return passes(n, max);
    }
    let tiles =
        |v: &[std::ops::Range<usize>]| -> usize { v.iter().map(|r| r.len().div_ceil(128)).sum() };
    let cost = |v: &[std::ops::Range<usize>]| -> f64 {
        v.iter()
            .map(|r| fa_time(g, r.len()) * (start + r.end) as f64)
            .sum()
    };
    let base = passes_n(n, np0);
    let (t0, mut best) = (tiles(&base), (cost(&base), base));
    for np in np0 + 1..=np0 + (np0 / 8).max(1) {
        let v = passes_n(n, np);
        let c = cost(&v);
        if tiles(&v) <= t0 && c < FA_MARGIN * best.0 {
            best = (c, v);
        }
    }
    best.1
}

/// Bytes per token of a batched activation buffer: a decode.cu Act, padded
/// so every token's starts 256-byte aligned (the GEMMs' 16-byte copies).
pub(super) fn act_stride(n: usize) -> usize {
    act_bytes(n).next_multiple_of(256)
}

/// What a batched pass starts from: tokens (first == 0), or nb hidden
/// states of dim floats each, token-major (first > 0; see `Input`).
#[derive(Clone, Copy)]
pub(crate) enum BatchIn<'x> {
    Tokens(&'x [u32]),
    Hidden(&'x [f32]),
    /// This many states already in the batch scratch's residual stream,
    /// and the RoPE table of their positions in its `rope` (left there by
    /// the streamed layers, stream.rs).
    Resident(usize),
    /// This many states already in the residual stream (a pipeline's
    /// previous stage copied them there, `pipeline`); the RoPE table is
    /// made here, as for `Hidden`.
    Placed(usize),
    /// This many tokens already in the batch scratch's `tokens` (a captured
    /// pass, slots.rs: the copy is made before the graph runs).
    Loaded(usize),
}

impl BatchIn<'_> {
    fn len(&self, dim: usize) -> usize {
        match self {
            BatchIn::Tokens(t) => t.len(),
            BatchIn::Hidden(x) => x.len() / dim,
            BatchIn::Resident(n) | BatchIn::Placed(n) | BatchIn::Loaded(n) => *n,
        }
    }

    /// Tokens r.
    fn range(&self, r: std::ops::Range<usize>, dim: usize) -> Self {
        match self {
            BatchIn::Tokens(t) => BatchIn::Tokens(&t[r]),
            BatchIn::Hidden(x) => BatchIn::Hidden(&x[r.start * dim..r.end * dim]),
            BatchIn::Resident(n) => {
                assert!(r == (0..*n), "resident states: one pass");
                BatchIn::Resident(*n)
            }
            BatchIn::Placed(n) => {
                assert!(r == (0..*n), "placed states: one pass");
                BatchIn::Placed(*n)
            }
            BatchIn::Loaded(n) => {
                assert!(r == (0..*n), "loaded tokens: one pass");
                BatchIn::Loaded(*n)
            }
        }
    }
}

/// `pass_hooked`'s mixture-of-experts FFN: (layer, the residual stream,
/// tokens).
pub(super) type MoeHook<'a> = dyn FnMut(usize, Arg, usize) -> Result<(), String> + 'a;

/// Where a pass's tokens go: one sequence, or (the server's decode step,
/// serve S2) a row per token of many (prefill.cu "Placement").
#[derive(Clone, Copy)]
pub(crate) enum Place {
    /// Positions start.., in the cache's rows base + start.. (a slot of a
    /// pool starts at row base; a plain cache at 0). Paged (`CudaCache::tab`):
    /// the sequence's table starts at entry base.
    Seq { start: usize, base: usize },
    /// Token z at position rows[z].pos of the sequence whose rows (paged:
    /// table entries) start at rows[z].base, those `RowAt`s in the scratch's `rp` (`upload_rows`
    /// first: the copy stays out of a captured pass). A sequence's rows of a
    /// pass in ascending positions; the attention of each row sees its
    /// sequence's rows up to its own (those of the pass included), so the
    /// per-query attention runs it.
    Rows,
}

/// A token's place in a multi-sequence pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowAt {
    pub pos: u32,
    pub base: u32,
}

/// Weight types of the tensor-core GEMMs (gemm2), in table order; the
/// first `NTYPES_VM` have gemvm kernels too (Q4_1 / Q5_1 / Q2_K / Q3_K / IQ4_NL
/// take gemvn).
const NTYPES: usize = 11;
const NTYPES_VM: usize = 6;

/// `qt`'s slot in the per-type kernel tables.
fn type_index(qt: u32) -> usize {
    match qt {
        Q4_K => 0,
        Q6_K => 1,
        Q8_0 => 2,
        Q5_0 => 3,
        Q5_K => 4,
        Q4_0 => 5,
        Q4_1 => 6,
        Q5_1 => 7,
        Q2_K => 8,
        Q3_K => 9,
        IQ4_NL => 10,
        _ => unreachable!("type_index: {qt}"),
    }
}

/// The batched kernels of a module.
pub(super) struct BatchFns<'m> {
    pub(super) embed: Func<'m>,
    rope_table: Func<'m>,
    rmsnorm_quant: Func<'m>,
    swiglu_quant: Func<'m>,
    rope_kv: Func<'m>,
    /// rope_kv with qwen3's per-head q/k norms.
    rope_kv_norm: Func<'m>,
    /// The split decode attention per query (attn_split_t: the q8 cache's
    /// kernels' f16 twin, for GPUs without tensor cores), and its combine.
    attn_split: Func<'m>,
    attn_combine: Func<'m>,
    /// The tensor-core attention per query (decode.cu attn_blk_t), and its
    /// fold: batches of at most att_per_query tokens.
    attn_blk: Func<'m>,
    attn_cmb: Func<'m>,
    /// attn_fa_b's variants (`Fa`, indexed by it) for this head size
    /// (larger batches), then quantize_b.
    attn_fa: [Func<'m>; 3],
    /// The device's compute capability, major * 10 + minor (`fa_pick`).
    cc: u32,
    quantize: Func<'m>,
    pub(super) argmax: Func<'m>,
    gemvn8: Func<'m>,
    gemvn16: Func<'m>,
    /// gemvm8_*, gemvm16_*, gemvm32_* per type (as `gemm`); None without
    /// tensor cores.
    gemvm: Option<[[Func<'m>; NTYPES_VM]; 3]>,
    /// gemvmx8_*, gemvmx16_*: gemvm over two or three matrices in one launch
    /// (`CudaQwen2::bgemm_x`); None without tensor cores.
    gemvmx: Option<[[Func<'m>; NTYPES_VM]; 2]>,
    /// AXON_GEMVMX=0: never `bgemm_x` (q, k, v and gate, up one launch each).
    pub(super) gemvmx_on: bool,
    /// AXON_GEMVM_MAX: 16, gemvm for at most 16 tokens (main's routing);
    /// 32, for up to 32 whatever the product; 0 or unset, by the product's
    /// shape (`CudaQwen2::takes_gemvm`).
    pub(super) vm_max: usize,
    /// Per type (Q4_K, Q6_K, Q8_0, Q5_0, Q5_K, Q4_0: `NTYPES`, indexed by
    /// `type_index`): gemm2_* (128 tokens per block)
    /// and gemm2s_* (64); None without int8 tensor cores.
    gemm: Option<[[Func<'m>; 2]; NTYPES]>,
    /// gemm2_q6k8 and gemm2s_q6k8 (Q6_K by its prefill copy, `Mat::c8`),
    /// and q6k_to_i8 which makes it; None without int8 tensor cores.
    gemm_q6k8: Option<[Func<'m>; 2]>,
    /// gemm2s_q4_k_gu: gate and up (both Q4_K) in one launch, silu(gate) *
    /// up out (`CudaQwen2::gate_up_fused`).
    pub(super) gemm_gu: Option<Func<'m>>,
    pub(super) q6k_to_i8: Option<Func<'m>>,
    /// hopper.cu gemmh2_q4_k and gemmh2_q4_k_gu: the Q4_K products and the
    /// fused gate + up on wgmma (same bits); sm_90a modules only, off with
    /// AXON_GEMMH=0. Measured on an H100 PCIe (r7 E0, 7B, nb 1050):
    /// gemmh2_q4_k_gu 80.3 vs gemm2s_q4_k_gu 68.1 TMAC/s, and in the pass
    /// q / wo / down 6-10 % faster than gemm2s; a 512-row matrix at nb 512
    /// (32 blocks) was slower (21.6 vs 35.6). See `gemmh_plain`.
    pub(super) gemmh: Option<[Func<'m>; 2]>,
    /// AXON_GEMMH=1: every Q4_K product of a batch through gemmh2_q4_k,
    /// whatever its shape (else only those `gemmh_plain` picks).
    gemmh_all: bool,
    /// SMs of the GPU (the GEMM tile choice).
    sms: usize,
    /// stream.cu: r8/r8q blocks back to GGUF rows.
    pub(super) unpermute: Func<'m>,
    /// gemma.cu (gemma3): the scaled embedding, the post-norm residual
    /// add, GeGLU, q/k norms at head size 256, the windowed attention.
    embed_s: Func<'m>,
    rmsnorm_add: Func<'m>,
    geglu_quant: Func<'m>,
    rope_kv_norm_256: Func<'m>,
    attn_split_w: Func<'m>,
    /// attn_split_w through a page table (a paged cache).
    attn_split_w_pg: Func<'m>,
    attn_combine_w: Func<'m>,
}

impl<'m> BatchFns<'m> {
    /// Whether the module has the tensor-core kernels (sm_80+).
    #[cfg(test)]
    pub(super) fn has_mma(&self) -> bool {
        self.gemm.is_some()
    }

    pub(super) fn new(
        module: &'m Module<'m>,
        head_dim: usize,
        sms: usize,
        cc: u32,
    ) -> Result<BatchFns<'m>, String> {
        let gemm = if module.mma() {
            let mut fs = Vec::new();
            for (qt, tn) in [
                (Q4_K, "q4_k"),
                (Q6_K, "q6_k"),
                (Q8_0, "q8_0"),
                (Q5_0, "q5_0"),
                (Q5_K, "q5_k"),
                (Q4_0, "q4_0"),
                (Q4_1, "q4_1"),
                (Q5_1, "q5_1"),
                (Q2_K, "q2_k"),
                (Q3_K, "q3_k"),
                (IQ4_NL, "iq4_nl"),
            ] {
                let big = module.function(&format!("gemm2_{tn}"))?;
                big.set_max_shared(gemm2_smem(qt, 128, 2))?;
                let small = module.function(&format!("gemm2s_{tn}"))?;
                small.set_max_shared(gemm2_smem(qt, 64, 3))?;
                fs.push([big, small]);
            }
            let f: [[Func<'m>; 2]; NTYPES] = fs.try_into().map_err(|_| "gemm2 kernels")?;
            Some(f)
        } else {
            None
        };
        let gemm_q6k8 = if module.mma() {
            let big = module.function("gemm2_q6k8")?;
            big.set_max_shared(gemm2_smem(Q6_K, 128, 2))?;
            let small = module.function("gemm2s_q6k8")?;
            small.set_max_shared(gemm2_smem(Q6_K, 64, 3))?;
            Some([big, small])
        } else {
            None
        };
        let gemvm = if module.mma() {
            let f = |nf: usize| -> Result<[Func<'m>; NTYPES_VM], String> {
                let n = |t: &str| module.function(&format!("gemvm{nf}_{t}"));
                Ok([
                    n("q4_k")?,
                    n("q6_k")?,
                    n("q8_0")?,
                    n("q5_0")?,
                    n("q5_k")?,
                    n("q4_0")?,
                ])
            };
            Some([f(8)?, f(16)?, f(32)?])
        } else {
            None
        };
        let gemvmx = if module.mma() {
            let f = |nf: usize| -> Result<[Func<'m>; NTYPES_VM], String> {
                let n = |t: &str| module.function(&format!("gemvmx{nf}_{t}"));
                Ok([
                    n("q4_k")?,
                    n("q6_k")?,
                    n("q8_0")?,
                    n("q5_0")?,
                    n("q5_k")?,
                    n("q4_0")?,
                ])
            };
            Some([f(8)?, f(16)?])
        } else {
            None
        };
        Ok(BatchFns {
            gemvmx,
            gemvmx_on: std::env::var("AXON_GEMVMX").map_or(true, |v| v != "0"),
            embed: module.function("embed_b")?,
            rope_table: module.function("rope_table_b")?,
            rmsnorm_quant: module.function("rmsnorm_quant_b")?,
            swiglu_quant: module.function("swiglu_quant_b")?,
            rope_kv: module.function("rope_kv_b")?,
            rope_kv_norm: module.function("rope_kv_norm_b")?,
            attn_split: module.function(if head_dim == 64 {
                "attn_split_b_64"
            } else {
                "attn_split_b_128"
            })?,
            attn_combine: module.function("attn_combine_b")?,
            attn_blk: module.function(if head_dim == 64 {
                "attn_blk_b_64"
            } else {
                "attn_blk_b_128"
            })?,
            attn_cmb: module.function("attn_cmb_b")?,
            attn_fa: {
                let f = |n: &str, v: Fa| -> Result<Func<'m>, String> {
                    let f = module.function(&format!("{n}_{head_dim}"))?;
                    if module.mma() {
                        f.set_max_shared(fa_shape(head_dim, v, 0).2)?;
                    }
                    Ok(f)
                };
                [
                    f("attn_fa_b", Fa::Full)?,
                    f("attn_fa_bs", Fa::Small)?,
                    f("attn_fa_k", Fa::Kv)?,
                ]
            },
            cc,
            quantize: module.function("quantize_b")?,
            argmax: module.function("argmax_b")?,
            gemvn8: module.function("gemvn_2x8")?,
            gemvn16: module.function("gemvn_2x16")?,
            gemvm,
            vm_max: std::env::var("AXON_GEMVM_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .map_or(
                    0,
                    |v: usize| if v == 0 { 0 } else { v.clamp(SMALL, VM_MAX) },
                ),
            gemm,
            gemm_q6k8,
            gemm_gu: if module.mma() {
                let f = module.function("gemm2s_q4_k_gu")?;
                f.set_max_shared(gemm2_smem(Q4_K, 64, 3))?;
                Some(f)
            } else {
                None
            },
            gemmh: if module.hopper() && std::env::var("AXON_GEMMH").as_deref() != Ok("0") {
                let (a, b) = (
                    module.function("gemmh2_q4_k")?,
                    module.function("gemmh2_q4_k_gu")?,
                );
                a.set_max_shared(GEMMH2_SMEM)?;
                b.set_max_shared(GEMMH2_SMEM)?;
                Some([a, b])
            } else {
                None
            },
            gemmh_all: std::env::var("AXON_GEMMH").as_deref() == Ok("1"),
            q6k_to_i8: if module.mma() {
                Some(module.function("q6k_to_i8")?)
            } else {
                None
            },
            sms,
            unpermute: module.function("unpermute_r8")?,
            embed_s: module.function("embed_bs")?,
            rmsnorm_add: module.function("rmsnorm_add_b")?,
            geglu_quant: module.function("geglu_quant_b")?,
            rope_kv_norm_256: module.function("rope_kv_norm_b_256")?,
            attn_split_w: module.function("attn_split_bw_256")?,
            attn_split_w_pg: module.function("attn_split_bw_256_pg")?,
            attn_combine_w: module.function("attn_combine_bw")?,
        })
    }
}

/// A cache's scratch for batched passes of up to `max` tokens.
pub(super) struct BatchBufs<'g> {
    pub(super) max: usize,
    /// Blocks of ATT_BLK positions of the cache it's for (the stride of
    /// attn_blk_b's partials).
    nblk_max: usize,
    pub(super) tokens: DevBuf<'g>,
    host_tokens: HostBuf<'g>,
    /// `Place::Rows` of the pass: per token (position, cache row), u32s;
    /// a pool's scratch only (`slots.rs`).
    pub(super) rp: Option<(DevBuf<'g>, HostBuf<'g>)>,
    /// Incoming hidden states, [max][dim] (first > 0 only).
    host_x: Option<HostBuf<'g>>,
    /// Residual stream, [max][dim].
    pub(super) x: DevBuf<'g>,
    /// Products of the q, k, v matrices (biases added), [max][q_dim + 2 kv_dim].
    qkv: DevBuf<'g>,
    /// Rotated queries, [max][q_dim].
    q: DevBuf<'g>,
    gate: DevBuf<'g>,
    up: DevBuf<'g>,
    /// Quantized activations, [max][act_stride(act_len())].
    pub(super) act: DevBuf<'g>,
    /// gemm2s_q4_k_gu's arrival counters (`CudaQwen2::gu_quant`): one u32
    /// per 64-token tile and 256 rows of ffn, zero between launches (empty
    /// unless ffn % 256 == 0).
    pub(super) gu_cnt: DevBuf<'g>,
    rope: DevBuf<'g>,
    /// gemma3's local layers' table (empty otherwise).
    rope_swa: DevBuf<'g>,
    part_o: DevBuf<'g>,
    part_ml: DevBuf<'g>,
    /// HEAD_SUB rows, reused per head launch; with `logits_rows` (tests:
    /// `keep_logits`) one row per token of the pass instead.
    pub(super) logits: DevBuf<'g>,
    pub(super) logits_rows: bool,
    pub(super) best: DevBuf<'g>,
    pub(super) host_best: HostBuf<'g>,
}

/// `pass_hooked`'s Err for a MoE model without its hook.
const MOE_DENSE_PASS: &str = "a mixture-of-experts model's batched pass needs its experts \
     (crate::moe: moe_tokens), not the dense pass";

impl<'g> CudaQwen2<'g> {
    /// Tokens per batched pass a cache of `max_ctx` positions gets scratch for.
    pub(super) fn batch_for(&self, max_ctx: usize) -> usize {
        self.batch.min(max_ctx).max(1)
    }

    /// Partials (per head, hd + 2 floats each) of the batched attention:
    /// att_per_query queries' blocks of ATT_BLK positions (attn_blk_b), or
    /// without tensor cores ATT_SUB queries' splits (attn_split_b).
    fn batch_parts(&self, max: usize, max_ctx: usize) -> usize {
        match self.tc_attn {
            true => max.min(att_per_query(self.head_dim)) * Self::nblk_max(max_ctx),
            false => max.min(ATT_SUB) * self.max_split,
        }
    }

    /// VRAM of the batch scratch for `max` tokens of a cache of max_ctx
    /// positions.
    pub(super) fn batch_bytes(&self, max: usize, max_ctx: usize) -> usize {
        let (f, kvd) = (4, self.kv_dim());
        let parts = self.n_heads * self.batch_parts(max, max_ctx) * (self.head_dim + 2) * f;
        let per_tok = (self.dim + 2 * self.q_dim + 2 * kvd + 2 * self.ffn + self.head_dim) * f
            + act_stride(self.act_len())
            + 8
            + if self.first > 0 { self.dim * f } else { 0 }
            + if self.freq_swa.len() > 0 {
                self.head_dim / 2 * 8
            } else {
                0
            };
        max * per_tok + parts + max.min(HEAD_SUB) * self.vocab * f + 256 + self.gu_cnt_bytes(max)
    }

    /// Bytes of `BatchBufs::gu_cnt` for passes of up to max tokens.
    fn gu_cnt_bytes(&self, max: usize) -> usize {
        match self.ffn.is_multiple_of(256) {
            true => max.div_ceil(64) * (self.ffn / 256) * 4,
            false => 0,
        }
    }

    pub(super) fn batch_bufs(&self, max: usize, max_ctx: usize) -> Result<BatchBufs<'g>, String> {
        let (g, f, kvd) = (self.gpu, 4, self.kv_dim());
        let parts = self.n_heads * self.batch_parts(max, max_ctx);
        Ok(BatchBufs {
            max,
            nblk_max: Self::nblk_max(max_ctx),
            tokens: g.alloc(max * 4)?,
            host_tokens: g.host_alloc(max * 4)?,
            rp: None,
            host_x: match self.first {
                0 => None,
                _ => Some(g.host_alloc(max * self.dim * f)?),
            },
            x: g.alloc(max * self.dim * f)?,
            qkv: g.alloc(max * (self.q_dim + 2 * kvd) * f)?,
            q: g.alloc(max * self.q_dim * f)?,
            gate: g.alloc(max * self.ffn * f)?,
            up: g.alloc(max * self.ffn * f)?,
            act: g.alloc(max * act_stride(self.act_len()))?,
            gu_cnt: {
                let c = g.alloc(self.gu_cnt_bytes(max))?;
                c.zero()?;
                c
            },
            rope: g.alloc(max * self.head_dim / 2 * 8)?,
            rope_swa: g.alloc(if self.freq_swa.len() > 0 {
                max * self.head_dim / 2 * 8
            } else {
                0
            })?,
            part_o: g.alloc(parts * self.head_dim * f)?,
            part_ml: g.alloc(parts * 2 * f)?,
            logits: g.alloc(max.min(HEAD_SUB) * self.vocab * f)?,
            logits_rows: false,
            best: g.alloc(max * 4)?,
            host_best: g.host_alloc(max * 4)?,
        })
    }

    /// out[z][r] (row stride ostride floats) = W act[z] (+ bias) for mode 0
    /// (W at mt.off in `wts`), += for mode 1, silu(bias[z][r]) * W act[z]
    /// for mode 2 (bias: the gate product, laid out as out; prefill.cu
    /// gm_store), for tokens z < nb: gemvn for small batches (or without
    /// tensor cores), else the type's GEMM (Q6_K: by its prefill copy if
    /// made, `Mat::c8`). Same bits either way.
    ///
    /// # Safety
    /// act holds nb tokens of mt.cols inputs at astride bytes apart; out has
    /// room for nb rows of ostride floats (mt.rows <= ostride); bias is null
    /// or has mt.rows floats (mode 2: nb rows of ostride floats); the matrix
    /// is whole in `wts`.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn bgemm(
        &self,
        wts: &DevBuf,
        mt: &Mat,
        act: Arg,
        astride: usize,
        nb: usize,
        bias: Arg,
        mode: i32,
        out: Arg,
        ostride: usize,
    ) -> Result<(), String> {
        let bf = &self.bf;
        let ti = type_index(mt.qt);
        // gemvm (weight-read-once) for up to vm_max tokens of its types.
        let vm = bf.gemvm.is_some() && ti < NTYPES_VM && self.takes_gemvm(mt.rows, nb);
        let small = vm || nb <= SMALL;
        // Q6_K by its prefill copy where made (the same products).
        let q8c = match (mt.c8, &bf.gemm_q6k8) {
            (Some(c8), Some(f)) if !small => Some((c8, f)),
            _ => None,
        };
        let (wa, rb) = match q8c {
            Some((c8, _)) => (self.w8.arg_at(c8), mt.cols / 256 * Q6K8_BYTES),
            None => (wts.arg_at(mt.off), mt.rb),
        };
        let common = [
            wa,
            Arg::U64(rb as u64),
            Arg::I32(mt.rows as i32),
            Arg::I32(mt.cols as i32),
            act,
            Arg::U64(astride as u64),
            Arg::I32(nb as i32),
            bias,
            Arg::I32(mode),
            out,
            Arg::U64(ostride as u64),
        ];
        let plain = bf.gemmh_all || gemmh_plain(mt.rows, nb, bf.sms);
        if let (Some(h), Q4_K, true, true) = (&bf.gemmh, mt.qt, !small, plain) {
            // TMA descriptors of the weights (rows of rb bytes, boxes of
            // 128 rows of one block) and the activations' Q8_K values.
            let rb = mt.rb as u64;
            let mw = self.gpu.tensor_map_2d(
                dev_ptr(wts.arg_at(mt.off)),
                (rb, mt.rows as u64),
                rb,
                (144, 128),
                false,
            )?;
            let mx = self.gpu.tensor_map_2d(
                dev_ptr(act),
                (mt.cols as u64, nb as u64),
                astride as u64,
                (128, 64),
                true,
            )?;
            // SAFETY: the caller's contract; gemmh2_q4_k's signature.
            return unsafe {
                self.gpu.launch(
                    &h[0],
                    (nb.div_ceil(64) as u32, mt.rows.div_ceil(128) as u32, 1),
                    (256, 1, 1),
                    GEMMH2_SMEM as u32,
                    &[
                        Arg::Blob128(mw),
                        Arg::Blob128(mx),
                        act,
                        Arg::U64(astride as u64),
                        Arg::I32(mt.rows as i32),
                        Arg::I32(mt.cols as i32),
                        Arg::I32(nb as i32),
                        bias,
                        Arg::I32(mode),
                        out,
                        Arg::U64(ostride as u64),
                    ],
                )
            };
        }
        // SAFETY: the caller's contract; the matrix is whole in `wts`.
        unsafe {
            match (&bf.gemm, &bf.gemvm) {
                (Some(fs), _) if !small => {
                    // Measured (cuda_gemm_bench): Q4_K is fastest in 64 x 64
                    // tiles (three stages, one barrier each) at every size;
                    // the register-staged types in 64 x 128, given at least
                    // ~1.5 blocks per SM.
                    let ctas = nb.div_ceil(128) * mt.rows.div_ceil(G2_BM);
                    // (Not for decode-sized batches: a 128-token tile of
                    // <= 64 tokens is half padding; the lm_head of a pool's
                    // 32 rows ran 2x slower in it, C-curve log.)
                    let big = !matches!(mt.qt, Q4_K | Q5_K) && 2 * ctas >= 3 * bf.sms && nb > 64;
                    let (bn, st) = if big { (128, 2) } else { (64, 3) };
                    let (f, smem) = match q8c {
                        Some((_, f8)) => (&f8[!big as usize], gemm2_smem(Q6_K, bn, st)),
                        None => (&fs[ti][!big as usize], gemm2_smem(mt.qt, bn, st)),
                    };
                    self.gpu.launch(
                        f,
                        (nb.div_ceil(bn) as u32, mt.rows.div_ceil(G2_BM) as u32, 1),
                        (256, 1, 1),
                        smem as u32,
                        &common,
                    )
                }
                (_, Some(fs)) if vm => self.gpu.launch(
                    &fs[nb.next_power_of_two().max(8).trailing_zeros() as usize - 3][ti],
                    (mt.rows.div_ceil(16) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    &common,
                ),
                _ => {
                    let (f, t) = if nb <= 8 {
                        (&bf.gemvn8, 8)
                    } else {
                        (&bf.gemvn16, 16)
                    };
                    let mut args = vec![Arg::I32(mt.qt as i32)];
                    args.extend_from_slice(&common);
                    self.gpu.launch(
                        f,
                        (mt.rows.div_ceil(2) as u32, nb.div_ceil(t) as u32, 1),
                        (GEMV_THREADS, 1, 1),
                        0,
                        &args,
                    )
                }
            }
        }
    }

    /// Tokens at positions start.. (nb <= the cache's batch max) through
    /// every layer, their K/V into the cache; with `head`, each one's greedy
    /// pick into b.best. Enqueued, not waited for.
    pub(super) fn pass(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        start: usize,
        head: bool,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
    ) -> Result<(), String> {
        self.pass_with(c, input, start, head.into(), 0, on)
    }

    /// `pass` with any `Head`: Sample puts token z's candidates in out row
    /// z, with penalty row `row0` + z of S (`samp_upload` first).
    pub(super) fn pass_with(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        start: usize,
        head: Head,
        row0: usize,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
    ) -> Result<(), String> {
        self.pass_at(
            c,
            input,
            Place::Seq { start, base: 0 },
            head,
            None,
            row0,
            on,
        )
    }

    /// `pass_with` at any `Place`. `split` Some((ng, ns)) with
    /// `Head::Sample`: tokens z < ng take the greedy pick (into b.best, as
    /// `Head::Argmax`), ng <= z < ns sample (candidates in out row z - ng,
    /// penalty row `row0` + z - ng), and the rest only get their logits
    /// (b.logits row z: a `logits_rows` scratch).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pass_at(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        place: Place,
        head: Head,
        split: Option<(usize, usize)>,
        row0: usize,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
    ) -> Result<(), String> {
        self.pass_hooked(c, input, place, head, split, row0, on, None)
    }

    /// `pass_at`, a mixture-of-experts layer's FFN by `moe`
    /// (`crate::moe`): after the layer's attention half, `moe(layer, x,
    /// nb)` with x the residual stream ([nb][dim] floats) leaves x + the
    /// layer's FFN output there. None (dense layers): `layer` as always;
    /// None on a MoE model: an Err, before anything is enqueued.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pass_hooked(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        place: Place,
        head: Head,
        split: Option<(usize, usize)>,
        row0: usize,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
        mut moe: Option<&mut MoeHook>,
    ) -> Result<(), String> {
        if moe.is_none() && self.moe.is_some() {
            return Err(MOE_DENSE_PASS.into());
        }
        let nb = input.len(self.dim);
        let dim = self.dim;
        // Paged (c.tab): the layers' arrays from row 0 and the rows through
        // the table, a sequence's from entry base on.
        if let Place::Seq { start, base } = place {
            c.debug_covers(base, start + nb);
        }
        let (base, tab) = match (place, c.tab.is_some()) {
            (Place::Seq { base, .. }, false) => (base, Arg::Ptr(0)),
            (Place::Seq { base, .. }, true) => (0, c.tab_at(base)),
            (Place::Rows, _) => (0, c.tab_at(0)),
        };
        let kvas: Vec<Vec<Arg>> = (0..self.layers.len())
            .map(|l| self.kv_args_at(c, l, base))
            .collect();
        let mode = c.mode;
        let b = c.b.as_mut().expect("a cache with batch scratch");
        assert!(nb > 0 && nb <= b.max);
        let (ng, ns) = match (head, split) {
            (Head::Sample, Some((ng, ns))) => {
                assert!(ng <= ns && ns <= nb && (ns == nb || b.logits_rows));
                (ng, ns)
            }
            (Head::Sample, None) => (0, nb),
            (_, None) => (nb, nb),
            _ => panic!("split rows: a sampling pass"),
        };
        self.load_input(b, input)?;
        let (start, rp) = match place {
            Place::Seq { start, .. } => (start, Arg::Ptr(0)),
            Place::Rows => (0, b.rp.as_ref().expect("a pool's scratch").0.arg()),
        };
        let b = &*b;
        let (g, bf) = (self.gpu, &self.bf);
        let w = |mt: &Mat| self.weights.arg_at(mt.off);
        let null = Arg::Ptr(0);
        let ast = act_stride(self.act_len());
        // SAFETY (every launch): arguments match prefill.cu's signatures;
        // the batch buffers hold b.max >= nb tokens of what each kernel
        // reads and writes (sizes in batch_bufs); positions start..start + nb
        // are in the cache (the callers check); matrices are whole in the arena.
        unsafe {
            if let (Some(embd), BatchIn::Tokens(_) | BatchIn::Loaded(_), Some(gm)) =
                (&self.embd, input, &self.gemma)
            {
                g.launch(
                    &bf.embed_s,
                    (dim.div_ceil(256) as u32, nb as u32, 1),
                    (256, 1, 1),
                    0,
                    &[
                        Arg::I32(embd.qt as i32),
                        w(embd),
                        Arg::U64(embd.rb as u64),
                        Arg::I32(dim as i32),
                        b.tokens.arg(),
                        Arg::F32(gm.embed_scale),
                        b.x.arg(),
                    ],
                )?;
                on("embed")?;
            } else if let (Some(embd), BatchIn::Tokens(_) | BatchIn::Loaded(_)) =
                (&self.embd, input)
            {
                g.launch(
                    &bf.embed,
                    (dim.div_ceil(256) as u32, nb as u32, 1),
                    (256, 1, 1),
                    0,
                    &[
                        Arg::I32(embd.qt as i32),
                        w(embd),
                        Arg::U64(embd.rb as u64),
                        Arg::I32(dim as i32),
                        b.tokens.arg(),
                        b.x.arg(),
                    ],
                )?;
                on("embed")?;
            }
            // Resident states come with their RoPE table (stream.rs
            // computed it for the same positions).
            if !matches!(input, BatchIn::Resident(_)) {
                self.rope_b(b, nb, start, rp)?;
                on("rope")?;
            }
            for (li, l) in self.layers.iter().enumerate() {
                match (&l.ffn, moe.as_deref_mut()) {
                    (Ffn::Moe { .. }, Some(f)) => {
                        self.attn_half(
                            b,
                            l,
                            &self.weights,
                            &self.vecs,
                            &kvas[li],
                            mode,
                            nb,
                            start,
                            rp,
                            tab,
                            on,
                        )?;
                        f(li, b.x.arg(), nb)?;
                    }
                    _ => self.layer(
                        b,
                        l,
                        &self.weights,
                        &self.vecs,
                        &kvas[li],
                        mode,
                        nb,
                        start,
                        rp,
                        tab,
                        on,
                    )?,
                }
            }
            if head == Head::Last {
                // The last row alone: its norm into act row 0, the head
                // (nb = 1: gemvm, as decode's gemv == gemv_ref) into logits
                // row 0.
                self.norm_rows(b, &self.vecs, self.output_norm, nb - 1, 1)?;
                on("rmsnorm")?;
                self.bgemm(
                    &self.weights,
                    &self.output,
                    b.act.arg(),
                    ast,
                    1,
                    null,
                    0,
                    b.logits.arg(),
                    self.vocab,
                )?;
                on("head")?;
            } else if head != Head::No {
                self.norm_b(b, &self.vecs, self.output_norm, nb)?;
                on("rmsnorm")?;
                let v = self.vocab;
                // Logits of every row (a pool's): the head in one product
                // over all of them (the weights read once), each row's
                // logits the same bits as in slices of HEAD_SUB.
                let sub = if b.logits_rows { nb } else { HEAD_SUB };
                for z0 in (0..nb).step_by(sub) {
                    let nz = (nb - z0).min(sub);
                    let lg = b.logits.arg_at(if b.logits_rows { z0 * v * 4 } else { 0 });
                    self.bgemm(
                        &self.weights,
                        &self.output,
                        b.act.arg_at(z0 * ast),
                        ast,
                        nz,
                        null,
                        0,
                        lg,
                        v,
                    )?;
                    // Rows z0..za greedy, za..zs sampled, zs.. logits only.
                    let za = ng.clamp(z0, z0 + nz);
                    let zs = ns.clamp(z0, z0 + nz);
                    let lo = if b.logits_rows { z0 * v * 4 } else { 0 };
                    if za > z0 {
                        g.launch(
                            &bf.argmax,
                            ((za - z0) as u32, 1, 1),
                            (1024, 1, 1),
                            0,
                            &[lg, Arg::I32(v as i32), b.best.arg_at(z0 * 4)],
                        )?;
                    }
                    // (samp_tail takes HEAD_SUB rows a launch.)
                    for z in (za..zs).step_by(HEAD_SUB) {
                        let nr = (zs - z).min(HEAD_SUB);
                        let s = c.samp.as_ref().expect("samp_upload before a sampling pass");
                        let ls = b.logits.arg_at(lo + (z - z0) * v * 4);
                        self.samp_tail(s, ls, v, nr, row0 + z - ng, z - ng, false)?;
                    }
                    on("head")?;
                }
            }
        }
        Ok(())
    }

    /// Enqueue the pass's input: tokens into b.tokens, or hidden states
    /// straight into the residual stream b.x ([max][dim] floats, the
    /// embedding's output otherwise); `Resident` states are there already.
    pub(super) fn load_input(&self, b: &mut BatchBufs<'g>, input: BatchIn) -> Result<(), String> {
        match input {
            BatchIn::Tokens(tokens) => {
                assert!(
                    self.first == 0,
                    "layers {}..: takes hidden states",
                    self.first
                );
                let hb = b.host_tokens.bytes_mut()?;
                for (i, t) in tokens.iter().enumerate() {
                    hb[4 * i..4 * i + 4].copy_from_slice(&t.to_le_bytes());
                }
                b.host_tokens.copy_to(&b.tokens, 0, 4 * tokens.len())?;
            }
            BatchIn::Hidden(xs) => {
                let hx = b.host_x.as_mut().expect("first > 0: host_x");
                for (d, v) in hx.bytes_mut()?.chunks_exact_mut(4).zip(xs) {
                    d.copy_from_slice(&v.to_le_bytes());
                }
                hx.copy_to(&b.x, 0, xs.len() * 4)?;
            }
            BatchIn::Resident(_) | BatchIn::Loaded(_) => {}
            BatchIn::Placed(_) => assert!(self.first > 0, "takes tokens"),
        }
        Ok(())
    }

    /// b.rope = the RoPE table of positions start..start + nb (rp: a
    /// `Place::Rows` table's).
    ///
    /// # Safety
    /// nb <= b.max; rp null or nb (position, row) pairs.
    pub(super) unsafe fn rope_b(
        &self,
        b: &BatchBufs,
        nb: usize,
        start: usize,
        rp: Arg,
    ) -> Result<(), String> {
        let hd = self.head_dim;
        // SAFETY: rope_table_b's signature; b.rope (and b.rope_swa, with
        // gemma3's local layers) holds b.max >= nb rows.
        unsafe {
            if b.rope_swa.len() > 0 {
                self.gpu.launch(
                    &self.bf.rope_table,
                    (nb as u32, 1, 1),
                    ((hd / 2) as u32, 1, 1),
                    0,
                    &[
                        self.freq_swa.arg(),
                        Arg::I32((hd / 2) as i32),
                        Arg::I32(start as i32),
                        b.rope_swa.arg(),
                        rp,
                        Arg::F32(1.0),
                    ],
                )?;
            }
            self.gpu.launch(
                &self.bf.rope_table,
                (nb as u32, 1, 1),
                ((hd / 2) as u32, 1, 1),
                0,
                &[
                    self.freq.arg(),
                    Arg::I32((hd / 2) as i32),
                    Arg::I32(start as i32),
                    b.rope.arg(),
                    rp,
                    Arg::F32(self.rope_m),
                ],
            )
        }
    }

    /// b.act = rmsnorm_quant of b.x's nb rows with the weights at `off`
    /// floats into `vecs`.
    ///
    /// # Safety
    /// nb <= b.max; vecs holds dim floats at off.
    pub(super) unsafe fn norm_b(
        &self,
        b: &BatchBufs,
        vecs: &DevBuf,
        off: usize,
        nb: usize,
    ) -> Result<(), String> {
        // SAFETY: the caller's contract.
        unsafe { self.norm_rows(b, vecs, off, 0, nb) }
    }

    /// `norm_b` of b.x's rows z0..z0 + n into b.act's rows 0..n.
    ///
    /// # Safety
    /// z0 + n <= b.max; vecs holds dim floats at off.
    pub(super) unsafe fn norm_rows(
        &self,
        b: &BatchBufs,
        vecs: &DevBuf,
        off: usize,
        z0: usize,
        n: usize,
    ) -> Result<(), String> {
        let dim = self.dim;
        let want_k = |n: usize| Arg::I32(n.is_multiple_of(256) as i32);
        // SAFETY: rmsnorm_quant_b's signature; sizes as `pass`.
        unsafe {
            self.gpu.launch(
                &self.bf.rmsnorm_quant,
                (n as u32, 1, 1),
                (256, 1, 1),
                0,
                &[
                    b.x.arg_at(z0 * dim * 4),
                    vecs.arg_at(off * 4),
                    Arg::F32(self.eps),
                    Arg::I32(dim as i32),
                    want_k(dim),
                    b.act.arg(),
                    Arg::U64(act_stride(self.act_len()) as u64),
                ],
            )
        }
    }

    /// One layer over the nb tokens in b.x at positions start..: matrices
    /// at their offsets into `wts`, norms and biases at theirs into `vecs`,
    /// the layer's cache at `kv` (`kv_args`: F16 K and V rows of kv_dim
    /// halves per position; Q8 int8 K and V rows and their scales) in
    /// format `mode`, which picks the attention kernels. b.rope holds the
    /// batch's table. rp non-null: the tokens' (position, cache row) pairs
    /// (`Place::Rows`, start unused), which takes the per-query attention.
    /// tab non-null: the page table (paged KV, `CudaCache::tab`; with rp,
    /// the pool's, and rp's rows are the tokens' entries into it).
    /// `on` runs after each step, by name.
    ///
    /// # Safety
    /// nb <= b.max; the matrices are whole in `wts`, the vectors in `vecs`;
    /// the K and V buffers hold positions 0..start + nb, rows before start
    /// written (rp: each token's row, and its sequence's rows before it);
    /// all stay live until the work completes.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn layer(
        &self,
        b: &BatchBufs,
        l: &LayerW,
        wts: &DevBuf,
        vecs: &DevBuf,
        kv: &[Arg],
        mode: CacheKv,
        nb: usize,
        start: usize,
        rp: Arg,
        tab: Arg,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
    ) -> Result<(), String> {
        // SAFETY: the caller's contract (`attn_half`'s and `ffn_half`'s).
        unsafe {
            self.attn_half(b, l, wts, vecs, kv, mode, nb, start, rp, tab, on)?;
            self.ffn_half(b, l, wts, vecs, nb, on)
        }
    }

    /// `layer`'s first half: the attention norm through x += W_o att
    /// (gemma3: and its post-norm). `crate::moe` runs it per layer.
    ///
    /// # Safety
    /// As `layer`.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn attn_half(
        &self,
        b: &BatchBufs,
        l: &LayerW,
        wts: &DevBuf,
        vecs: &DevBuf,
        kv: &[Arg],
        mode: CacheKv,
        nb: usize,
        start: usize,
        rp: Arg,
        tab: Arg,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
    ) -> Result<(), String> {
        let rows = !matches!(rp, Arg::Ptr(0));
        let paged = !matches!(tab, Arg::Ptr(0));
        let (dim, hd, kvd) = (self.dim, self.head_dim, self.kv_dim());
        let qd = self.q_dim;
        let (g, bf) = (self.gpu, &self.bf);
        let vec = |off: usize| vecs.arg_at(off * 4);
        let null = Arg::Ptr(0);
        let want_k = |n: usize| Arg::I32(n.is_multiple_of(256) as i32);
        let ast = act_stride(self.act_len());
        let nbu = nb as u32;
        let qkvw = qd + 2 * kvd;
        // SAFETY (every launch): as `pass`, with the caller's contract for
        // the matrices, vectors and cache rows.
        unsafe {
            self.norm_b(b, vecs, l.attn_norm, nb)?;
            on("rmsnorm")?;
            let bias = |o: Option<usize>| o.map_or(null, vec);
            let qkv_at = |off: usize| b.qkv.arg_at(off * 4);
            let (q, k, v) = (
                (&l.q, bias(l.bq), qkv_at(0)),
                (&l.k, bias(l.bk), qkv_at(qd)),
                (&l.v, bias(l.bv), qkv_at(qd + kvd)),
            );
            // q, k and v in one launch where they qualify (bgemm_x), else q
            // and k, else one by one.
            let one =
                |m: (&Mat, Arg, Arg)| self.bgemm(wts, m.0, b.act.arg(), ast, nb, m.1, 0, m.2, qkvw);
            if !self.bgemm_x(wts, &[q, k, v], b.act.arg(), ast, nb, qkvw)? {
                if !self.bgemm_x(wts, &[q, k], b.act.arg(), ast, nb, qkvw)? {
                    one(q)?;
                    on("q")?;
                    one(k)?;
                }
                on("k")?;
                one(v)?;
                // (Profile segments by type; "qkv", "gate_up", "down" are
                // the stream's weight-group events, stream.rs.)
                on(if l.v.qt == Q6_K { "v6" } else { "v4" })?;
            }
            on("qkv")?;
            // q8: rope_kv_b (rope_kv_norm_b) does the queries alone (null
            // cache), kv_q8_b the keys and values (normalized first on
            // qwen3, as decode).
            let q8 = matches!(mode, CacheKv::Q8 { .. });
            let (kc, vc) = if q8 { (null, null) } else { (kv[0], kv[1]) };
            // This layer's RoPE table (gemma3's local layers: their own).
            let rope = match l.win {
                0 => b.rope.arg(),
                _ => b.rope_swa.arg(),
            };
            if let (Some(qn), Some(kn), 256) = (l.qn, l.kn, hd) {
                // gemma3: rope_kv_norm_b at head size 256.
                let heads = (qd + 2 * kvd) / hd;
                g.launch(
                    &bf.rope_kv_norm_256,
                    (heads.div_ceil(4) as u32, nbu, 1),
                    (128, 1, 1),
                    0,
                    &[
                        b.qkv.arg(),
                        Arg::I32(qd as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(start as i32),
                        rope,
                        vec(qn),
                        vec(kn),
                        Arg::F32(self.eps),
                        b.q.arg(),
                        kc,
                        vc,
                        rp,
                        tab,
                    ],
                )?;
            } else if let (Some(qn), Some(kn)) = (l.qn, l.kn) {
                // qwen3: a warp per head and token (decode.cu warp_norm_rope).
                let heads = (qd + 2 * kvd) / hd;
                g.launch(
                    &bf.rope_kv_norm,
                    (heads.div_ceil(4) as u32, nbu, 1),
                    (128, 1, 1),
                    0,
                    &[
                        b.qkv.arg(),
                        Arg::I32(qd as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(start as i32),
                        b.rope.arg(),
                        vec(qn),
                        vec(kn),
                        Arg::F32(self.eps),
                        b.q.arg(),
                        kc,
                        vc,
                        rp,
                        tab,
                    ],
                )?;
            } else {
                let items = if q8 { qd / 2 } else { qd / 2 + kvd / 2 + kvd };
                g.launch(
                    &bf.rope_kv,
                    (items.div_ceil(128) as u32, nbu, 1),
                    (128, 1, 1),
                    0,
                    &[
                        b.qkv.arg(),
                        Arg::I32(qd as i32),
                        Arg::I32(kvd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(start as i32),
                        b.rope.arg(),
                        b.q.arg(),
                        kc,
                        vc,
                        rp,
                        tab,
                    ],
                )?;
            }
            if q8 {
                let mut args = vec![
                    b.qkv.arg(),
                    Arg::I32(qd as i32),
                    Arg::I32(kvd as i32),
                    Arg::I32(start as i32),
                    b.rope.arg(),
                ];
                args.extend_from_slice(kv);
                args.push(bias(l.ksh));
                args.push(bias(l.kn));
                args.push(Arg::F32(self.eps));
                args.push(rp);
                args.push(tab);
                g.launch(
                    &self.q8_fns(mode).store_b,
                    (self.n_kv_heads as u32, nbu, 1),
                    (128, 1, 1),
                    0,
                    &args,
                )?;
            }
            on("rope_kv")?;
            let pre = Arg::F32(att_pre(hd));
            let kg = match mode {
                CacheKv::F16 => 0,
                CacheKv::Q8 { group } => group,
            };
            assert!(
                rows || start + nb <= b.nblk_max * ATT_BLK,
                "batch past its cache"
            );
            if self.tc_attn && nb > att_per_query(hd) && !rows {
                // Flash attention, every query of the pass in one launch,
                // y in place of q; then act (the decode kernels' bits,
                // prefill.cu attn_fa_bt), in the variant fa_pick takes for
                // this GPU and batch.
                let group = self.n_heads / self.n_kv_heads;
                let v = fa_pick(self.fa_geom(kg), nb);
                let (rows, threads, smem) = fa_shape(hd, v, kg);
                let f = match q8 {
                    false => &bf.attn_fa[v as usize],
                    // The q8 cache has attn_fa_b and attn_fa_bs only (fa_pick).
                    true => match v {
                        Fa::Full => &self.q8_fns(mode).attn_fa[0],
                        Fa::Small => &self.q8_fns(mode).attn_fa[1],
                        Fa::Kv => unreachable!("fa_pick: attn_fa_k is f16 only"),
                    },
                };
                let mut args = vec![b.q.arg()];
                args.extend_from_slice(kv);
                args.extend_from_slice(&[
                    Arg::I32(start as i32),
                    Arg::I32(nb as i32),
                    Arg::I32(self.n_heads as i32),
                    Arg::I32(self.n_kv_heads as i32),
                    pre,
                    b.q.arg(),
                    tab,
                ]);
                g.launch(
                    f,
                    (
                        self.n_kv_heads as u32,
                        (nb * group).div_ceil(rows) as u32,
                        1,
                    ),
                    (threads as u32, 1, 1),
                    smem as u32,
                    &args,
                )?;
                on("attn")?;
                g.launch(
                    &bf.quantize,
                    (qd.div_ceil(256) as u32, nbu, 1),
                    (32, 1, 1),
                    0,
                    &[
                        b.q.arg(),
                        Arg::I32(qd as i32),
                        want_k(qd),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )?;
                on("combine")?;
            } else {
                // A few queries (the verify pass), or no tensor cores: the
                // decode kernels per query (the same code), attn_blk_t, or
                // attn_split_t ATT_SUB queries per launch.
                let tc = self.tc_attn;
                let per = if tc { att_per_query(hd) } else { ATT_SUB };
                let (gy_blk, ms) = (self.max_split.min(b.nblk_max) as u32, self.max_split as u32);
                for z0 in (0..nb).step_by(per) {
                    let nz = (nb - z0).min(per) as u32;
                    // (kernel, blocks per kv head and query, threads, the
                    // partials' stride: blocks or splits, the arguments
                    // after rp: the kernel's own, picked with it).
                    let gemma = self.gemma.is_some();
                    let (f, grid, block, stride, tail) = match (tc, q8, gemma) {
                        // gemma3: each query's window (0: a global layer);
                        // its plain-cache kernel takes no table (gemma.cu).
                        (false, false, true) => {
                            let win = Arg::I32(l.win as i32);
                            let (f, tail) = if paged {
                                (&bf.attn_split_w_pg, vec![win, tab])
                            } else {
                                (&bf.attn_split_w, vec![win])
                            };
                            (f, ms, ATT_TILE as u32, self.max_split, tail)
                        }
                        (_, _, true) => {
                            return Err("gemma3 attention: f16 split kernels only (load refuses \
                                 head sizes but 256, tensor cores and the q8 cache)"
                                .into());
                        }
                        (true, false, false) => (&bf.attn_blk, gy_blk, 32, b.nblk_max, vec![tab]),
                        (true, true, false) => (
                            &self.q8_fns(mode).attn_blk_b,
                            gy_blk,
                            32,
                            b.nblk_max,
                            vec![tab],
                        ),
                        (false, false, false) => (
                            &bf.attn_split,
                            ms,
                            ATT_TILE as u32,
                            self.max_split,
                            vec![tab],
                        ),
                        (false, true, false) => (
                            &self.q8_fns(mode).attn_split_b,
                            ms,
                            ATT_TILE as u32,
                            self.max_split,
                            vec![tab],
                        ),
                    };
                    let mut args = vec![b.q.arg()];
                    args.extend_from_slice(kv);
                    args.extend_from_slice(&[
                        Arg::I32(start as i32),
                        Arg::I32(z0 as i32),
                        Arg::I32(self.n_heads as i32),
                        Arg::I32(self.n_kv_heads as i32),
                        if tc {
                            pre
                        } else {
                            Arg::F32(1.0 / (hd as f32).sqrt())
                        },
                        Arg::I32(stride as i32),
                        b.part_o.arg(),
                        b.part_ml.arg(),
                        rp,
                    ]);
                    args.extend(tail);
                    g.launch(
                        f,
                        (self.n_kv_heads as u32, grid, nz),
                        (block, 1, 1),
                        0,
                        &args,
                    )?;
                    on("attn")?;
                    let mut cargs = vec![
                        b.part_o.arg(),
                        b.part_ml.arg(),
                        Arg::I32(start as i32),
                        Arg::I32(z0 as i32),
                        Arg::I32(qd as i32),
                        Arg::I32(hd as i32),
                        Arg::I32(stride as i32),
                        want_k(qd),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                        rp,
                    ];
                    let combine = if gemma {
                        cargs.push(Arg::I32(l.win as i32));
                        &bf.attn_combine_w
                    } else if tc {
                        &bf.attn_cmb
                    } else {
                        &bf.attn_combine
                    };
                    g.launch(
                        combine,
                        (qd.div_ceil(256) as u32, nz, 1),
                        (256, 1, 1),
                        0,
                        &cargs,
                    )?;
                    on("combine")?;
                }
            }
            // gemma3: x += rmsnorm(W act) per token (rmsnorm_add_b), the
            // product through b.up (free until the up product).
            let post_add = |off: usize| {
                g.launch(
                    &bf.rmsnorm_add,
                    (nbu, 1, 1),
                    (1024, 1, 1),
                    0,
                    &[
                        b.up.arg(),
                        Arg::U64(dim as u64),
                        vec(off),
                        Arg::F32(self.eps),
                        Arg::I32(dim as i32),
                        b.x.arg(),
                    ],
                )
            };
            match l.post_attn {
                Some(pa) => {
                    self.bgemm(wts, &l.o, b.act.arg(), ast, nb, null, 0, b.up.arg(), dim)?;
                    on("wo")?;
                    post_add(pa)?;
                    on("post_norm")?;
                }
                None => {
                    self.bgemm(wts, &l.o, b.act.arg(), ast, nb, null, 1, b.x.arg(), dim)?;
                    on("wo")?;
                }
            }
        }
        Ok(())
    }

    /// `layer`'s second half: the FFN norm, gate/up, SwiGLU (gemma3:
    /// GeGLU), down into x (gemma3: through its post-norm).
    ///
    /// # Safety
    /// As `layer` (a dense layer).
    pub(super) unsafe fn ffn_half(
        &self,
        b: &BatchBufs,
        l: &LayerW,
        wts: &DevBuf,
        vecs: &DevBuf,
        nb: usize,
        on: &mut dyn FnMut(&'static str) -> Result<(), String>,
    ) -> Result<(), String> {
        let (dim, ffn) = (self.dim, self.ffn);
        let (g, bf) = (self.gpu, &self.bf);
        let vec = |off: usize| vecs.arg_at(off * 4);
        let null = Arg::Ptr(0);
        let want_k = |n: usize| Arg::I32(n.is_multiple_of(256) as i32);
        let ast = act_stride(self.act_len());
        let nbu = nb as u32;
        let (gate, up, down) = l.dense();
        // SAFETY (every launch): as `pass`, with the caller's contract for
        // the matrices and vectors.
        unsafe {
            // gemma3: x += rmsnorm(W act) per token (rmsnorm_add_b), the
            // product through b.up (free until the up product).
            let post_add = |off: usize| {
                g.launch(
                    &bf.rmsnorm_add,
                    (nbu, 1, 1),
                    (1024, 1, 1),
                    0,
                    &[
                        b.up.arg(),
                        Arg::U64(dim as u64),
                        vec(off),
                        Arg::F32(self.eps),
                        Arg::I32(dim as i32),
                        b.x.arg(),
                    ],
                )
            };
            self.norm_b(b, vecs, l.ffn_norm, nb)?;
            on("rmsnorm")?;
            // Both Q4_K, a tensor-core batch: one launch for gate and up.
            let fuse = self.gate_up_fused
                && self.gemma.is_none()
                && bf.gemm_gu.is_some()
                && !self.takes_gemvm(gate.rows, nb)
                && gate.qt == Q4_K
                && up.qt == Q4_K
                && gate.cols == dim
                && up.cols == dim;
            // Gate and up in one gemvmx launch where they qualify (not
            // with the up product's swiglu epilogue, which reads the gate's).
            let epi = self.swiglu_epi && self.gemma.is_none();
            // Where the down product reads hid quantized: b.gate when the
            // fused launch quantizes it (`gu_quant`; its counters cover
            // nb, and nb tokens of act fit in b.gate's floats), else b.act.
            let hact_gate = fuse
                && bf.gemmh.is_none()
                && self.gu_quant
                && ffn.is_multiple_of(256)
                && b.gu_cnt.len() >= nb.div_ceil(64) * (ffn / 256) * 4
                && nb * ast <= b.gate.len();
            let hact = match hact_gate {
                true => b.gate.arg(),
                false => b.act.arg(),
            };
            let gu_x = !(fuse || epi)
                && self.bgemm_x(
                    wts,
                    &[(gate, null, b.gate.arg()), (up, null, b.up.arg())],
                    b.act.arg(),
                    ast,
                    nb,
                    ffn,
                )?;
            if !fuse && !gu_x {
                self.bgemm(wts, gate, b.act.arg(), ast, nb, null, 0, b.gate.arg(), ffn)?;
            }
            on("gate")?;
            if let (Some(h), true) = (&bf.gemmh, fuse) {
                // gemmh2_q4_k_gu: as gemm2s_q4_k_gu below, on wgmma.
                let rb = gate.rb as u64;
                let map = |off: usize| {
                    g.tensor_map_2d(
                        dev_ptr(wts.arg_at(off)),
                        (rb, ffn as u64),
                        rb,
                        (144, 64),
                        false,
                    )
                };
                let (mg, mu) = (map(gate.off)?, map(up.off)?);
                let mx = g.tensor_map_2d(
                    dev_ptr(b.act.arg()),
                    (dim as u64, nb as u64),
                    ast as u64,
                    (128, 64),
                    true,
                )?;
                g.launch(
                    &h[1],
                    (nb.div_ceil(64) as u32, ffn.div_ceil(64) as u32, 1),
                    (256, 1, 1),
                    GEMMH2_SMEM as u32,
                    &[
                        Arg::Blob128(mg),
                        Arg::Blob128(mu),
                        Arg::Blob128(mx),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                        Arg::I32(ffn as i32),
                        Arg::I32(dim as i32),
                        Arg::I32(nb as i32),
                        b.up.arg(),
                        Arg::U64(ffn as u64),
                    ],
                )?;
                on("gate_up")?;
                g.launch(
                    &bf.quantize,
                    (ffn.div_ceil(256) as u32, nbu, 1),
                    (32, 1, 1),
                    0,
                    &[
                        b.up.arg(),
                        Arg::I32(ffn as i32),
                        want_k(ffn),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )?;
            } else if let (Some(f), true) = (&bf.gemm_gu, fuse) {
                // gate and up in one launch, hid = silu(gate) * up out
                // (the same floats and ops as below), then quantize_b - or,
                // with `gu_quant`, quantize_b's work in the launch's tail
                // into b.gate (free: the gate product never goes to
                // memory), for the down product to read.
                let cnt = match hact_gate {
                    true => b.gu_cnt.arg(),
                    false => null,
                };
                g.launch(
                    f,
                    (nb.div_ceil(64) as u32, ffn.div_ceil(32) as u32, 1),
                    (256, 1, 1),
                    gemm2_smem(Q4_K, 64, 3) as u32,
                    &[
                        wts.arg_at(gate.off),
                        wts.arg_at(up.off),
                        Arg::U64(gate.rb as u64),
                        Arg::I32(ffn as i32),
                        Arg::I32(dim as i32),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                        Arg::I32(nb as i32),
                        b.up.arg(),
                        Arg::U64(ffn as u64),
                        hact,
                        Arg::U64(ast as u64),
                        want_k(ffn),
                        cnt,
                    ],
                )?;
                on("gate_up")?;
                if !hact_gate {
                    g.launch(
                        &bf.quantize,
                        (ffn.div_ceil(256) as u32, nbu, 1),
                        (32, 1, 1),
                        0,
                        &[
                            b.up.arg(),
                            Arg::I32(ffn as i32),
                            want_k(ffn),
                            b.act.arg(),
                            Arg::U64(ast as u64),
                        ],
                    )?;
                }
            } else if self.swiglu_epi && self.gemma.is_none() {
                // The up product's epilogue (mode 2) makes hid = silu(gate) *
                // up from the gate product and its own sum (swiglu_quant_b's
                // ops on the same floats) into b.up, then quantize_b: one
                // fp32 product less written and read back per layer.
                self.bgemm(
                    wts,
                    up,
                    b.act.arg(),
                    ast,
                    nb,
                    b.gate.arg(),
                    2,
                    b.up.arg(),
                    ffn,
                )?;
                on("gate_up")?;
                g.launch(
                    &bf.quantize,
                    (ffn.div_ceil(256) as u32, nbu, 1),
                    (32, 1, 1),
                    0,
                    &[
                        b.up.arg(),
                        Arg::I32(ffn as i32),
                        want_k(ffn),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )?;
            } else {
                if !gu_x {
                    self.bgemm(wts, up, b.act.arg(), ast, nb, null, 0, b.up.arg(), ffn)?;
                }
                on("gate_up")?;
                g.launch(
                    if self.gemma.is_some() {
                        &bf.geglu_quant
                    } else {
                        &bf.swiglu_quant
                    },
                    (ffn.div_ceil(256) as u32, nbu, 1),
                    (32, 1, 1),
                    0,
                    &[
                        b.gate.arg(),
                        b.up.arg(),
                        Arg::I32(ffn as i32),
                        want_k(ffn),
                        b.act.arg(),
                        Arg::U64(ast as u64),
                    ],
                )?;
            }
            on("swiglu")?;
            match l.post_ffn {
                // gemma3: as the post-attention norm (b.up free again).
                Some(pf) => {
                    self.bgemm(wts, down, hact, ast, nb, null, 0, b.up.arg(), dim)?;
                    on(if down.qt == Q6_K { "down6" } else { "down4" })?;
                    on("down")?;
                    post_add(pf)?;
                    on("post_norm")?;
                }
                None => {
                    self.bgemm(wts, down, hact, ast, nb, null, 1, b.x.arg(), dim)?;
                    on(if down.qt == Q6_K { "down6" } else { "down4" })?;
                    on("down")?;
                }
            }
        }
        Ok(())
    }

    /// Two or three products of the same activations (q, k, v; gate, up) in
    /// one gemvmx launch, each matrix with its bias and output (mode 0, the
    /// outputs `ostride` floats a token apart), if they qualify: at most
    /// SMALL tokens, one gemvm type and width. Same outputs as a bgemm each.
    /// Ok(false): nothing launched (the caller runs them one by one).
    ///
    /// # Safety
    /// As `bgemm`, for every matrix.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn bgemm_x(
        &self,
        wts: &DevBuf,
        ms: &[(&Mat, Arg, Arg)],
        act: Arg,
        astride: usize,
        nb: usize,
        ostride: usize,
    ) -> Result<bool, String> {
        let bf = &self.bf;
        let (Some(fs), true) = (&bf.gemvmx, bf.gemvmx_on) else {
            return Ok(false);
        };
        if nb > SMALL || !(2..=3).contains(&ms.len()) {
            return Ok(false);
        }
        let m0 = ms[0].0;
        let ti = type_index(m0.qt);
        if ti >= NTYPES_VM
            || ms
                .iter()
                .any(|(m, _, _)| m.qt != m0.qt || m.cols != m0.cols || m.rb != m0.rb)
        {
            return Ok(false);
        }
        let blocks: usize = ms.iter().map(|(m, _, _)| m.rows.div_ceil(16)).sum();
        let at = |i: usize| ms.get(i);
        let null = Arg::Ptr(0);
        let mut args = Vec::with_capacity(18);
        for i in 0..3 {
            args.push(at(i).map_or(null, |(m, _, _)| wts.arg_at(m.off)));
        }
        for i in 0..3 {
            args.push(Arg::I32(at(i).map_or(0, |(m, _, _)| m.rows as i32)));
        }
        for i in 0..3 {
            args.push(at(i).map_or(null, |&(_, b, _)| b));
        }
        for i in 0..3 {
            args.push(at(i).map_or(null, |&(_, _, o)| o));
        }
        args.extend_from_slice(&[
            Arg::U64(m0.rb as u64),
            Arg::I32(m0.cols as i32),
            act,
            Arg::U64(astride as u64),
            Arg::I32(nb as i32),
            Arg::U64(ostride as u64),
        ]);
        // SAFETY: the caller's contract; gemvmx's signature (prefill.cu).
        unsafe {
            self.gpu.launch(
                &fs[(nb > 8) as usize][ti],
                (blocks as u32, 1, 1),
                (128, 1, 1),
                0,
                &args,
            )?;
        }
        Ok(true)
    }

    /// Whether a product of `rows` outputs over nb tokens (of a gemvm type)
    /// takes gemvm rather than gemm2: up to SMALL tokens always; up to
    /// VM_MAX when gemm2's tiles would leave the GPU mostly idle (fewer
    /// 64-row tiles than a quarter of the SMs, e.g. the 7B's 512-row k / v,
    /// the 0.5B's 896-row q / o / down) and gemvm32's one pass over the
    /// weights wins (C-curve log, RTX 4090: 7B k at nb 32 12.3 vs 20.5 us);
    /// bigger products stay on gemm2, where gemvm32's 16-row blocks re-read
    /// the tokens' activations (7B gate at nb 32: 138 vs 96 us).
    /// `BatchFns::vm_max` (AXON_GEMVM_MAX) overrides.
    pub(super) fn takes_gemvm(&self, rows: usize, nb: usize) -> bool {
        match (nb, self.bf.vm_max) {
            (..=SMALL, _) => true,
            (_, _) if nb > VM_MAX => false,
            (_, 0) => 4 * rows.div_ceil(G2_BM) < self.bf.sms,
            (_, v) => nb <= v,
        }
    }

    /// The flash attention's shapes for cache scale group kg (0: f16).
    fn fa_geom(&self, kg: usize) -> FaGeom {
        FaGeom {
            cc: self.bf.cc,
            sms: self.bf.sms,
            hd: self.head_dim,
            kg,
            n_kv: self.n_kv_heads,
            group: self.n_heads / self.n_kv_heads,
        }
    }

    /// The passes of n tokens at start for cache c (`fa_passes` when the
    /// flash attention runs them, else `passes`).
    pub(super) fn prompt_passes(
        &self,
        c: &CudaCache<'g>,
        n: usize,
        start: usize,
    ) -> Vec<std::ops::Range<usize>> {
        let max = c.b.as_ref().expect("batch scratch").max;
        match self.tc_attn && max > att_per_query(self.head_dim) {
            true => fa_passes(
                n,
                max,
                start,
                self.fa_geom(match c.mode {
                    CacheKv::F16 => 0,
                    CacheKv::Q8 { group } => group,
                }),
            ),
            false => passes(n, max),
        }
    }

    /// `feed_batch`: passes of up to the cache's batch max (`prompt_passes`),
    /// then a wait (so a device fault surfaces here, as with `feed`).
    pub(super) fn feed_tokens(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        start: usize,
    ) -> Result<(), String> {
        for r in self.prompt_passes(c, input.len(self.dim), start) {
            let ch = input.range(r.clone(), self.dim);
            self.pass(c, ch, start + r.start, false, &mut |_| Ok(()))?;
        }
        self.gpu.sync()
    }

    /// `prefill_logits` (nb >= 2): the passes of `feed_tokens`, the last
    /// with `Head::Last`; its logits (waits for the passes).
    pub(super) fn prefill_tokens(
        &self,
        c: &mut CudaCache<'g>,
        tokens: &[u32],
        start: usize,
    ) -> Result<Vec<f32>, String> {
        let ps = self.prompt_passes(c, tokens.len(), start);
        for (i, r) in ps.iter().enumerate() {
            let head = if i + 1 == ps.len() {
                Head::Last
            } else {
                Head::No
            };
            let ch = BatchIn::Tokens(&tokens[r.clone()]);
            self.pass_with(c, ch, start + r.start, head, 0, &mut |_| Ok(()))?;
        }
        let n = self.vocab * 4;
        let b = c.b.as_ref().unwrap();
        c.host_out.copy_from(&b.logits, 0, n)?;
        let hb = &c.host_out.bytes_mut()?[..n]; // waits for the passes and copy
        Ok(hb
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect())
    }

    /// `verify_argmax`: each token's greedy pick.
    pub(super) fn verify_tokens(
        &self,
        c: &mut CudaCache<'g>,
        input: BatchIn,
        start: usize,
    ) -> Result<Vec<u32>, String> {
        let max = c.b.as_ref().expect("batch scratch").max;
        let nt = input.len(self.dim);
        let mut picks = Vec::with_capacity(nt);
        for r in passes(nt, max) {
            let ch = input.range(r.clone(), self.dim);
            self.pass(c, ch, start + r.start, true, &mut |_| Ok(()))?;
            let b = c.b.as_mut().unwrap();
            picks.extend(b.picks(ch.len(self.dim))?);
        }
        Ok(picks)
    }
}

/// The passes of n tokens with at most max per pass: as few as possible,
/// of equal size (the first ones one token longer), so a prompt just past a
/// multiple of max doesn't end in a short pass (latency-bound GEMMs: the
/// weights read once more for a few tokens). Results don't depend on it.
pub(super) fn passes(n: usize, max: usize) -> Vec<std::ops::Range<usize>> {
    passes_n(n, n.div_ceil(max))
}

/// n tokens in np equal passes (the first ones one token longer).
fn passes_n(n: usize, np: usize) -> Vec<std::ops::Range<usize>> {
    let (q, r) = (n / np.max(1), n % np.max(1));
    let mut v = Vec::with_capacity(np);
    let mut s = 0;
    for i in 0..np {
        let e = s + q + (i < r) as usize;
        v.push(s..e);
        s = e;
    }
    v
}

impl<'g> BatchBufs<'g> {
    /// The residual stream, [max][dim] floats: a pass's hidden states after
    /// its last layer (a pipeline stage's output), or its input
    /// (`BatchIn::Placed`).
    pub(super) fn x(&self) -> &DevBuf<'g> {
        &self.x
    }

    /// Enqueue the copy of `Place::Rows` into `rp` (a pool's scratch):
    /// per row its position and cache row, or (`tab` Some: `CudaCache::tab`
    /// of that many entries) its sequence's first entry in the table.
    pub(super) fn upload_rows(&mut self, rows: &[RowAt], tab: Option<usize>) -> Result<(), String> {
        let paged = tab.is_some();
        if let Some(n) = tab {
            for r in rows {
                debug_assert!(
                    r.base as usize + (r.pos as usize + 1).div_ceil(ATT_BLK) <= n,
                    "row at {} from table entry {}: the table has {n} entries",
                    r.pos,
                    r.base
                );
            }
        }
        let (d, h) = self.rp.as_mut().expect("a pool's scratch: Place::Rows");
        let hb = h.bytes_mut()?;
        for (i, r) in rows.iter().enumerate() {
            let row = if paged { r.base } else { r.base + r.pos };
            hb[8 * i..8 * i + 4].copy_from_slice(&r.pos.to_le_bytes());
            hb[8 * i + 4..8 * i + 8].copy_from_slice(&row.to_le_bytes());
        }
        h.copy_to(d, 0, 8 * rows.len())
    }

    /// The first n greedy picks of the pass enqueued (waits for it).
    pub(super) fn picks(&mut self, n: usize) -> Result<Vec<u32>, String> {
        let n = 4 * n;
        self.host_best.copy_from(&self.best, 0, n)?;
        let hb = &self.host_best.bytes_mut()?[..n]; // waits for the pass
        Ok(hb
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect())
    }
}

impl<'g> CudaQwen2<'g> {
    /// `verify_cands`: each token's sampling candidates, in passes of up to
    /// the cache's batch max.
    pub(super) fn verify_cands_tokens(
        &self,
        c: &mut CudaCache<'g>,
        tokens: &[u32],
        start: usize,
        spec: &CandSpec,
        recent: &[u32],
    ) -> Result<Vec<Vec<Cand>>, String> {
        let max = c.b.as_ref().expect("batch scratch").max;
        self.samp_upload(c, spec, recent, tokens.len(), tokens.len().min(max), None)?;
        let mut rows = Vec::with_capacity(tokens.len());
        for (i, ch) in tokens.chunks(max).enumerate() {
            let input = BatchIn::Tokens(ch);
            self.pass_with(
                c,
                input,
                start + i * max,
                Head::Sample,
                i * max,
                &mut |_| Ok(()),
            )?;
            rows.extend(self.samp_read(c, spec.k, self.vocab, ch.len())?);
        }
        Ok(rows)
    }
}
