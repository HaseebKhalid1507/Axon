//! Streaming the CPU's layers through the GPU (PR A): for a big prefill
//! batch of a model split across devices (`crate::hybrid`), layers 0..first
//! (the CPU's) run here too, their weights copied over once per batch
//! while the previous layer computes, then the GPU's own layers as usual.
//! A prompt chunk of a few hundred tokens needs ~30 ms of tensor-core work
//! per 32B layer and ~12 ms to copy that layer's ~0.3 GB at ~25 GB/s, so
//! the copies hide under the compute and the CPU layers run at GPU speed.
//!
//! Weights: the CPU keeps them repacked for its kernels (quant r8 / r8q, in
//! one anonymous mapping), with the file's pages released. That mapping is
//! page-locked in place (`Gpu::register`: no second copy in RAM), its bytes
//! are copied into a small staging buffer and `unpermute_r8` (stream.cu)
//! turns them back into GGUF rows, the exact inverse of the repack, in a
//! ring buffer in VRAM. Matrices the CPU didn't repack are copied as they
//! are (from pageable memory: slower). Three streams: copies (`copy`),
//! un-permutes (`unp`), and the model's own for the compute; events order
//! them: a staging buffer is reused once its un-permute ran, a ring region
//! once the compute that read it ran (per step group: qkv, wo, gate_up,
//! down), and layer i runs once its weights and cache rows are in.
//!
//! KV cache: the CPU's (f16) stays the only copy. Per streamed layer its
//! rows 0..start go into one of two VRAM slots (registered CPU buffers,
//! `StreamKv`), the batch's new rows are written there by the usual kernels
//! and copied back into the CPU cache afterwards: same f16 format, so the
//! CPU's later tokens read them as their own.
//!
//! Numerics: the CPU layers' prompt positions get the GPU's numerics (its
//! bits: the same kernels as a GPU-resident layer), not the CPU's. So this
//! is never used where bits must match a one-token step (speculative
//! verify), only for prefill; see `crate::hybrid`.

use super::batch::BatchIn;
use super::*;
use crate::cuda::{Pinned, Stream, SyncEvent};
use crate::kv::Half;

/// A matrix of a streamed layer as the CPU holds it: GGUF rows, or the r8
/// / r8q repack of them (`repacked`).
pub(crate) struct HostMat<'s> {
    pub qt: u32,
    pub rows: usize,
    pub cols: usize,
    pub bytes: &'s [u8],
    pub repacked: bool,
}

/// One CPU layer: q, k, v, o, gate, up, down, and its norms and biases.
pub(crate) struct HostLayer<'s> {
    pub mats: [HostMat<'s>; 7],
    pub attn_norm: &'s [f32],
    pub ffn_norm: &'s [f32],
    pub bq: Option<&'s [f32]>,
    pub bk: Option<&'s [f32]>,
    pub bv: Option<&'s [f32]>,
    /// qwen3's per-head q and k RMSNorm weights.
    pub qn: Option<&'s [f32]>,
    pub kn: Option<&'s [f32]>,
}

/// Step group (the ring's unit of reuse) of each matrix, in `mats` order.
const GROUP: [usize; 7] = [0, 0, 0, 1, 2, 2, 3];
const GROUPS: usize = 4;
/// Bytes per staging buffer (two of them).
const STAGE: usize = 16 << 20;

/// The VRAM ring the streamed matrices go through.
struct Ring {
    size: usize,
    head: usize,
    /// (start, end, done event) of the regions the compute may still read.
    live: Vec<(usize, usize, usize)>,
}

impl Ring {
    /// A region of n bytes, and the latest done event to wait for before
    /// writing it (the compute is in order: the latest implies the others).
    fn place(&mut self, n: usize) -> Result<(usize, Option<usize>), String> {
        let n = n.next_multiple_of(256);
        if n > self.size {
            return Err(format!(
                "stream ring of {} bytes < a {n}-byte matrix",
                self.size
            ));
        }
        if self.head + n > self.size {
            self.head = 0;
        }
        let (a, b) = (self.head, self.head + n);
        let mut wait = None;
        self.live.retain(|&(s, e, ev)| {
            let hit = s < b && a < e;
            if hit {
                wait = wait.max(Some(ev));
            }
            !hit
        });
        self.head = b;
        Ok((a, wait))
    }
}

/// Per split model: what streaming its CPU layers takes on the GPU.
pub struct Streamer<'g> {
    /// Per streamed layer: shapes and vector offsets (into `vecs`); the
    /// matrices' offsets are set per pass, into `ring`.
    layers: Vec<LayerW>,
    vecs: DevBuf<'g>,
    ring_buf: DevBuf<'g>,
    ring: Ring,
    stage: [DevBuf<'g>; 2],
    copy: Stream<'g>,
    unp: Stream<'g>,
    staged: [SyncEvent<'g>; 2],
    unpermed: [SyncEvent<'g>; 2],
    ready_c: [SyncEvent<'g>; 2],
    ready_u: [SyncEvent<'g>; 2],
    /// Per (layer, group): the compute that reads it is done.
    done: Vec<SyncEvent<'g>>,
    /// Staging chunks enqueued this pass.
    chunks: usize,
    /// The CPU's repacked weights, page-locked.
    _pins: Vec<Pinned<'g>>,
}

/// Per cache: two VRAM slots of K and V rows for the streamed layers, and
/// the CPU cache's buffers page-locked.
pub struct StreamKv<'g> {
    slots: DevBuf<'g>,
    max_ctx: usize,
    _pins: Vec<Pinned<'g>>,
}

/// Bytes of one 8-row r8/r8q block, and of the GGUF block it holds per row.
fn r8_block(qt: u32) -> (usize, usize) {
    match qt {
        Q4_K => (1184, 144),
        Q6_K => (1680, 210),
        Q8_0 => (272, 34),
        _ => (176, 22),
    }
}

/// Page-lock `a`, a whole read-only mapping (the CPU's repacked weights,
/// which must outlive the result: the hybrid model owns both and drops the
/// `Streamer` first). Where the driver can't register read-only memory
/// (CU_MEMHOSTREGISTER_READ_ONLY unsupported: this laptop's), the mapping
/// is made writable for the registration and read-only again after: the
/// pages stay pinned, the GPU only reads them, nothing writes them.
#[cfg(target_os = "linux")]
fn pin_read_only<'g>(g: &'g Gpu, a: &[u8]) -> Result<Pinned<'g>, String> {
    use std::ffi::{c_int, c_void};
    unsafe extern "C" {
        fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
    }
    const PROT_READ: c_int = 1;
    const PROT_WRITE: c_int = 2;
    // SAFETY: a is live for as long as the result (the caller's).
    if let Ok(p) = unsafe { g.register(a.as_ptr(), a.len(), true) } {
        return Ok(p);
    }
    let addr = a.as_ptr() as *mut c_void;
    assert!(
        (addr as usize).is_multiple_of(4096),
        "arena: a whole mapping"
    );
    let prot = |p: c_int, what: &str| {
        // SAFETY: the whole mapping `a`, page-aligned; no reference to it
        // is written through (only the protection changes).
        match unsafe { mprotect(addr, a.len(), p) } {
            0 => Ok(()),
            _ => Err(format!(
                "mprotect ({what}): {}",
                std::io::Error::last_os_error()
            )),
        }
    };
    prot(PROT_READ | PROT_WRITE, "rw")?;
    // SAFETY: as above.
    let r = unsafe { g.register(a.as_ptr(), a.len(), false) };
    prot(PROT_READ, "ro")?;
    r
}

#[cfg(not(target_os = "linux"))]
fn pin_read_only<'g>(_g: &'g Gpu, _a: &[u8]) -> Result<Pinned<'g>, String> {
    Err(crate::cuda::LINUX_ONLY.into())
}

impl<'g> CudaQwen2<'g> {
    /// What `streamer` allocates for these layers with a `ring`-byte ring.
    pub fn stream_bytes(&self, src: &[HostLayer], ring: usize) -> usize {
        let vecs: usize = src
            .iter()
            .map(|l| {
                let v = |s: &[f32]| s.len().next_multiple_of(64);
                v(l.attn_norm)
                    + v(l.ffn_norm)
                    + [l.bq, l.bk, l.bv, l.qn, l.kn]
                        .iter()
                        .flatten()
                        .map(|b| v(b))
                        .sum::<usize>()
            })
            .sum();
        ring + ARENA_PAD + 2 * Self::stage_for(src) + vecs * 4
    }

    /// Ring sizes for these layers: (smallest that works, best). The best
    /// holds two of the biggest layer: layer i + 1 copies in while layer i
    /// computes. The smallest holds one layer and its biggest matrix (the
    /// next layer's first matrices then wait for this one's compute).
    pub fn ring_bounds(src: &[HostLayer]) -> (usize, usize) {
        let gguf = |m: &HostMat| (m.rows * quant::row_bytes(m.qt, m.cols)).next_multiple_of(256);
        let layer = |l: &HostLayer| l.mats.iter().map(gguf).sum::<usize>();
        let min = src
            .iter()
            .map(|l| layer(l) + l.mats.iter().map(gguf).max().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let max = src.iter().map(layer).max().unwrap_or(0);
        (min, (2 * max).max(min))
    }

    /// Staging buffer size: STAGE, or one 8-row group if that's bigger.
    fn stage_for(src: &[HostLayer]) -> usize {
        let group = src
            .iter()
            .flat_map(|l| l.mats.iter())
            .filter(|m| m.repacked)
            .map(|m| m.bytes.len() / (m.rows / 8))
            .max()
            .unwrap_or(0);
        STAGE.max(group)
    }

    /// Set up streaming layers 0..first from `src` (one per layer), whose
    /// repacked bytes lie in `arena` (page-locked here), through a `ring`
    /// of VRAM (at least `ring_bounds().0` bytes). Err if a matrix
    /// type isn't supported here or VRAM is short.
    pub(crate) fn streamer(
        &self,
        src: &[HostLayer],
        arena: Option<&[u8]>,
        ring: usize,
    ) -> Result<Streamer<'g>, String> {
        let g = self.gpu;
        if src.len() != self.first {
            return Err(format!(
                "streaming {} layers before layer {}",
                src.len(),
                self.first
            ));
        }
        let mut vecs: Vec<f32> = Vec::new();
        let mut push = |v: &[f32]| {
            let off = vecs.len();
            vecs.extend_from_slice(v);
            vecs.resize(vecs.len().next_multiple_of(64), 0.0);
            off
        };
        let mut layers = Vec::with_capacity(src.len());
        for (i, l) in src.iter().enumerate() {
            for (m, s) in l
                .mats
                .iter()
                .zip(["q", "k", "v", "o", "gate", "up", "down"])
            {
                // Q5_K / Q4_0 / Q4_1 / Q5_1 have no CPU repack: they stream as GGUF rows.
                let raw_ok = matches!(m.qt, Q5_K | Q4_0 | Q4_1 | Q5_1) && !m.repacked;
                if !raw_ok && ![Q4_K, Q6_K, Q5_0, Q8_0].contains(&m.qt) {
                    return Err(format!(
                        "layer {i} {s}: type {} can't be streamed",
                        crate::gguf::qtype(m.qt).name
                    ));
                }
                let rb = quant::row_bytes(m.qt, m.cols);
                let want = if m.repacked {
                    let (b8, _) = r8_block(m.qt);
                    let per = if matches!(m.qt, Q4_K | Q6_K) { 256 } else { 32 };
                    m.rows / 8 * (m.cols / per) * b8
                } else {
                    m.rows * rb
                };
                if m.bytes.len() != want || (m.repacked && !m.rows.is_multiple_of(8)) {
                    return Err(format!(
                        "layer {i} {s}: {} bytes, want {want}",
                        m.bytes.len()
                    ));
                }
            }
            let mat = |m: &HostMat| Mat {
                qt: m.qt,
                off: 0,
                rb: quant::row_bytes(m.qt, m.cols),
                rows: m.rows,
                cols: m.cols,
                c8: None,
            };
            let [q, k, v, o, gate, up, down] = &l.mats;
            if gate.rows != self.ffn
                || q.cols != self.dim
                || q.rows != self.q_dim
                || k.rows != self.kv_dim()
            {
                return Err(format!("layer {i}: shapes differ from the GPU's layers"));
            }
            layers.push(LayerW {
                attn_norm: push(l.attn_norm),
                // Streamed layers' caches are the CPU's f16 rows: no q8
                // key shift.
                ksh: None,
                bq: l.bq.map(&mut push),
                bk: l.bk.map(&mut push),
                bv: l.bv.map(&mut push),
                qn: l.qn.map(&mut push),
                kn: l.kn.map(&mut push),
                ffn_norm: push(l.ffn_norm),
                q: mat(q),
                k: mat(k),
                v: mat(v),
                o: mat(o),
                gate: mat(gate),
                up: mat(up),
                down: mat(down),
                // gemma3 never streams (hybrid refuses it).
                post_attn: None,
                post_ffn: None,
                win: 0,
            });
        }
        let stage = Self::stage_for(src);
        let need = self.stream_bytes(src, ring) as u64 + CACHE_HEADROOM;
        let free = g.info()?.free_bytes;
        if need > free {
            return Err(format!(
                "streaming needs {:.2} GiB of GPU memory, {:.2} GiB is free",
                need as f64 / GIB,
                free as f64 / GIB
            ));
        }
        let mut pins = Vec::new();
        if let Some(a) = arena.filter(|a| !a.is_empty()) {
            pins.push(pin_read_only(g, a)?);
        }
        let ev = || g.new_event();
        Ok(Streamer {
            layers,
            vecs: g.upload(&vecs)?,
            ring_buf: g.alloc(ring + ARENA_PAD)?,
            ring: Ring {
                size: ring,
                head: 0,
                live: Vec::new(),
            },
            stage: [g.alloc(stage)?, g.alloc(stage)?],
            copy: g.new_stream()?,
            unp: g.new_stream()?,
            staged: [ev()?, ev()?],
            unpermed: [ev()?, ev()?],
            ready_c: [ev()?, ev()?],
            ready_u: [ev()?, ev()?],
            done: (0..src.len() * GROUPS)
                .map(|_| ev())
                .collect::<Result<_, _>>()?,
            chunks: 0,
            _pins: pins,
        })
    }

    /// VRAM `stream_kv(max_ctx, ..)` takes.
    pub fn stream_kv_bytes(&self, max_ctx: usize) -> usize {
        2 * 2 * max_ctx * self.kv_dim() * 2
    }

    /// KV slots for a cache of `max_ctx` positions whose CPU layers' K and
    /// V buffers are `host` (page-locked here: they must outlive the result).
    pub(crate) fn stream_kv(
        &self,
        max_ctx: usize,
        host: &[(&[Half], &[Half])],
    ) -> Result<StreamKv<'g>, String> {
        let g = self.gpu;
        let want = (self.stream_kv_bytes(max_ctx) as u64) + CACHE_HEADROOM;
        if want > g.info()?.free_bytes {
            return Err("no room for the streamed layers' KV slots".into());
        }
        let slots = g.alloc(self.stream_kv_bytes(max_ctx))?;
        let mut pins = Vec::new();
        for (k, v) in host {
            for b in [k, v] {
                // SAFETY: the caller keeps the buffers alive and in place
                // (KvBuf heap blocks) for as long as the StreamKv.
                pins.push(unsafe { g.register(b.as_ptr().cast(), size_of_val(*b), false)? });
            }
        }
        Ok(StreamKv {
            slots,
            max_ctx,
            _pins: pins,
        })
    }

    /// The nb hidden states `xs` (dim floats each, after the embedding) at
    /// positions start.. through layers 0..first streamed (`src`, `st`,
    /// their KV in `hkv`: rows 0..start read, start..start + nb written
    /// back, via `kv`'s slots), then through this model's own layers into
    /// `c` as `feed_hidden` would. With `head`, each token's greedy pick.
    /// nb is at most the cache's batch max.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stream_feed(
        &self,
        c: &mut CudaCache<'g>,
        st: &mut Streamer<'g>,
        kv: &StreamKv<'g>,
        src: &[HostLayer],
        hkv: &mut [(&mut [Half], &mut [Half])],
        xs: &[f32],
        start: usize,
        head: bool,
    ) -> Vec<u32> {
        self.stream_feed_r(c, st, kv, src, hkv, xs, start, head)
            .unwrap_or_else(|e| Self::die(e))
    }

    #[allow(clippy::too_many_arguments)]
    fn stream_feed_r(
        &self,
        c: &mut CudaCache<'g>,
        st: &mut Streamer<'g>,
        kv: &StreamKv<'g>,
        src: &[HostLayer],
        hkv: &mut [(&mut [Half], &mut [Half])],
        xs: &[f32],
        start: usize,
        head: bool,
    ) -> Result<Vec<u32>, String> {
        let (g, dim, kvd) = (self.gpu, self.dim, self.kv_dim());
        let nb = xs.len() / dim;
        Self::check_positions(nb, start, c);
        assert_eq!(c.net, self.id, "a CudaCache used with another CudaQwen2");
        assert!(
            c.tab.is_none(),
            "streamed layers copy plain rows: no paged cache"
        );
        assert_eq!(xs.len(), nb * dim, "hidden states: whole rows");
        let k = self.first;
        assert!(src.len() == k && st.layers.len() == k && hkv.len() == k);
        assert!(start + nb <= kv.max_ctx, "stream KV slots too small");
        let rows = kvd * (start + nb);
        assert!(hkv.iter().all(|(a, b)| a.len() >= rows && b.len() >= rows));
        let slot = |s: usize, v: bool| (2 * s + v as usize) * kv.max_ctx * kvd * 2;
        {
            let b = c.b.as_mut().expect("a cache with batch scratch");
            assert!(nb <= b.max, "stream_feed: {nb} > batch max {}", b.max);
            self.load_input(b, BatchIn::Hidden(xs))?;
        }
        let b = c.b.as_ref().unwrap();
        // SAFETY: nb <= b.max.
        unsafe { self.rope_b(b, nb, start, Arg::Ptr(0))? };
        let prof = std::env::var("AXON_STREAM_PROFILE").is_ok_and(|v| v == "1");
        let t0 = if prof { Some(g.record()?) } else { None };
        st.ring.head = 0;
        st.ring.live.clear();
        st.chunks = 0;
        let mut lw: Vec<LayerW> = st.layers.clone();
        self.copy_layer(st, kv, src, hkv, &mut lw, 0, start)?;
        for i in 0..k {
            let s = i % 2;
            g.wait_main(&st.ready_c[s])?;
            g.wait_main(&st.ready_u[s])?;
            // The CPU cache's format (f16), whatever the GPU half's
            // cache is (a q8 one too: `layer` picks the kernels per call).
            let kvp = [
                kv.slots.arg_at(slot(s, false)),
                kv.slots.arg_at(slot(s, true)),
            ];
            {
                let done = &st.done;
                let mut on = |name: &'static str| {
                    let gi = match name {
                        "qkv" => 0,
                        "wo" => 1,
                        "gate_up" => 2,
                        "down" => 3,
                        _ => return Ok(()),
                    };
                    g.record_main(&done[i * GROUPS + gi])
                };
                // SAFETY: nb <= b.max; layer i's matrices were placed whole
                // in the ring (copy_layer) and the main stream waited for
                // them (ready_*); its vectors are in st.vecs; the slot holds
                // rows 0..start (copied in before ready_c) and room up to
                // kv.max_ctx >= start + nb.
                unsafe {
                    self.layer(
                        b,
                        &lw[i],
                        &st.ring_buf,
                        &st.vecs,
                        &kvp,
                        CacheKv::F16,
                        nb,
                        start,
                        Arg::Ptr(0),
                        Arg::Ptr(0),
                        &mut on,
                    )?;
                }
            }
            if i + 1 < k {
                self.copy_layer(st, kv, src, hkv, &mut lw, i + 1, start)?;
            }
            // The new rows back to the CPU cache once the layer ran.
            st.copy.wait(&st.done[i * GROUPS + GROUPS - 1])?;
            let n = nb * kvd * 2;
            let (hk, hv) = &mut hkv[i];
            // SAFETY: rows start..start + nb of the registered CPU buffers
            // (checked above), not touched by the host until copy.sync below.
            unsafe {
                st.copy.d2h(
                    hk[start * kvd..].as_mut_ptr().cast(),
                    &kv.slots,
                    slot(s, false) + start * kvd * 2,
                    n,
                )?;
                st.copy.d2h(
                    hv[start * kvd..].as_mut_ptr().cast(),
                    &kv.slots,
                    slot(s, true) + start * kvd * 2,
                    n,
                )?;
            }
        }
        let t1 = if prof { Some(g.record()?) } else { None };
        // The GPU's own layers, from the states left in b.x.
        self.pass(c, BatchIn::Resident(nb), start, head, &mut |_| Ok(()))?;
        if let (Some(t0), Some(t1)) = (t0, t1) {
            let t2 = g.record()?;
            let (a, b) = (t0.ms_to(&t1)?, t1.ms_to(&t2)?);
            let own = self.layers.len();
            eprintln!(
                "stream: nb {nb} at {start}: {k} streamed layers {a:.1} ms ({:.2}/layer), \
                 {own} resident {b:.1} ms ({:.2}/layer)",
                a / k as f32,
                b / own as f32
            );
        }
        let picks = match head {
            true => c.b.as_mut().unwrap().picks(nb)?,
            false => Vec::new(),
        };
        g.sync()?;
        st.unp.sync()?;
        st.copy.sync()?;
        Ok(picks)
    }

    /// Enqueue layer i's cache rows 0..start into its slot and its matrices
    /// into the ring (offsets into lw[i]), then mark it ready.
    #[allow(clippy::too_many_arguments)]
    fn copy_layer(
        &self,
        st: &mut Streamer<'g>,
        kv: &StreamKv<'g>,
        src: &[HostLayer],
        hkv: &[(&mut [Half], &mut [Half])],
        lw: &mut [LayerW],
        i: usize,
        start: usize,
    ) -> Result<(), String> {
        let kvd = self.kv_dim();
        let s = i % 2;
        if start > 0 {
            let n = start * kvd * 2;
            for (v, h) in [(false, &hkv[i].0), (true, &hkv[i].1)] {
                let off = (2 * s + v as usize) * kv.max_ctx * kvd * 2;
                // SAFETY: rows 0..start of the registered CPU buffer, read
                // only until the stream is synced (end of the pass). The
                // slot's previous user (layer i - 2) was copied back before
                // this on the same stream, after its compute (done event).
                unsafe { st.copy.h2d(&kv.slots, off, h.as_ptr().cast(), n)? };
            }
        }
        let mats = &src[i].mats;
        let l = &mut lw[i];
        let dst: [&mut Mat; 7] = [
            &mut l.q,
            &mut l.k,
            &mut l.v,
            &mut l.o,
            &mut l.gate,
            &mut l.up,
            &mut l.down,
        ];
        for ((m, hm), d) in mats.iter().enumerate().zip(dst) {
            let bytes = d.rows * d.rb;
            let (off, wait) = st.ring.place(bytes)?;
            if wait.is_some_and(|w| w >= i * GROUPS) {
                return Err(format!(
                    "stream ring of {} MiB too small for layer {i}",
                    st.ring.size >> 20
                ));
            }
            st.ring.live.push((off, off + bytes, i * GROUPS + GROUP[m]));
            d.off = off;
            if !hm.repacked {
                if let Some(w) = wait {
                    st.copy.wait(&st.done[w])?;
                }
                // SAFETY: the model's bytes, live and unchanged for the
                // pass; ring region free (waited above).
                unsafe { st.copy.h2d(&st.ring_buf, off, hm.bytes.as_ptr(), bytes)? };
                continue;
            }
            if let Some(w) = wait {
                st.unp.wait(&st.done[w])?;
            }
            let (b8, bb) = r8_block(hm.qt);
            let nblk = hm.cols
                / if matches!(hm.qt, Q4_K | Q6_K) {
                    256
                } else {
                    32
                };
            let gb = nblk * b8;
            let groups = hm.rows / 8;
            let per = (st.stage[0].len() / gb).max(1);
            for g0 in (0..groups).step_by(per) {
                let ng = per.min(groups - g0);
                let j = st.chunks % 2;
                st.chunks += 1;
                st.copy.wait(&st.unpermed[j])?;
                // SAFETY: registered arena bytes (or pageable), live for the
                // pass; the staging buffer is free (its un-permute ran).
                unsafe {
                    st.copy
                        .h2d(&st.stage[j], 0, hm.bytes[g0 * gb..].as_ptr(), ng * gb)?
                };
                st.copy.record(&st.staged[j])?;
                st.unp.wait(&st.staged[j])?;
                let tiles = ng * nblk;
                // SAFETY: unpermute_r8's signature; the staging buffer holds
                // ng groups, the ring region rows 8 g0.. of rb bytes.
                unsafe {
                    self.gpu.launch_on(
                        &st.unp,
                        &self.bf.unpermute,
                        (tiles.div_ceil(4).min(4096) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        &[
                            Arg::I32(hm.qt as i32),
                            st.stage[j].arg(),
                            st.ring_buf.arg_at(off + g0 * 8 * d.rb),
                            Arg::U64(d.rb as u64),
                            Arg::I32(nblk as i32),
                            Arg::I32(b8 as i32),
                            Arg::I32(bb as i32),
                            Arg::U64(ng as u64),
                        ],
                    )?
                };
                st.unp.record(&st.unpermed[j])?;
            }
        }
        st.copy.record(&st.ready_c[s])?;
        st.unp.record(&st.ready_u[s])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
