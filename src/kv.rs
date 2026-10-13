//! KV cache storage: f16 by default (half the bytes decode attention streams
//! per token), f32 with AXON_KV=f32. Values are rounded to f16 once, when
//! stored (round to nearest even: `vcvtps2ph` or the scalar `f32_to_f16`,
//! same bits), and widened exactly on every read, so every attention path
//! computes on the same f32 values.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
use std::arch::x86_64::*;

/// IEEE half, as stored in the cache.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Half(pub u16);

/// A cache element type.
///
/// # Safety
/// Plain data: every bit pattern (in particular all zeros) is a valid value,
/// alignment <= 64.
pub unsafe trait KvElem: Copy + Send + Sync + 'static {
    fn to_f32(self) -> f32;
    fn from_f32(x: f32) -> Self;

    /// dst[i] = from_f32(src[i]).
    fn store(dst: &mut [Self], src: &[f32]) {
        assert_eq!(dst.len(), src.len(), "kv store: length");
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = Self::from_f32(s);
        }
    }

    /// The 8 values at p, widened to f32 (exact).
    ///
    /// # Safety
    /// p..p + 8 must be readable.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe fn load8(p: *const Self) -> __m256;
}

// SAFETY: f32 is plain data, align 4.
unsafe impl KvElem for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn from_f32(x: f32) -> f32 {
        x
    }
    fn store(dst: &mut [f32], src: &[f32]) {
        dst.copy_from_slice(src);
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[inline(always)]
    unsafe fn load8(p: *const f32) -> __m256 {
        // SAFETY: caller guarantees 8 readable floats.
        unsafe { _mm256_loadu_ps(p) }
    }
}

// SAFETY: Half is a u16, align 2.
unsafe impl KvElem for Half {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        crate::quant::f16_to_f32(self.0)
    }
    #[inline(always)]
    fn from_f32(x: f32) -> Half {
        Half(crate::quant::f32_to_f16(x))
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    fn store(dst: &mut [Half], src: &[f32]) {
        assert_eq!(dst.len(), src.len(), "kv store: length");
        let (sc, sr) = src.as_chunks::<8>();
        let (dc, dr) = dst.as_chunks_mut::<8>();
        for (d, s) in dc.iter_mut().zip(sc) {
            // SAFETY: F16C is enabled at compile time; 8 floats read, 8
            // halves (16 bytes) written, both whole chunks.
            unsafe {
                use std::arch::x86_64::*;
                let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(s.as_ptr()));
                _mm_storeu_si128(d.as_mut_ptr().cast(), h);
            }
        }
        for (d, &s) in dr.iter_mut().zip(sr) {
            *d = Half::from_f32(s);
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    #[inline(always)]
    unsafe fn load8(p: *const Half) -> __m256 {
        #[cfg(target_feature = "f16c")]
        // SAFETY: caller guarantees 8 readable halves (16 bytes).
        unsafe {
            _mm256_cvtph_ps(_mm_loadu_si128(p.cast()))
        }
        #[cfg(not(target_feature = "f16c"))]
        // SAFETY: as above; the widening is exact either way.
        unsafe {
            let w: [f32; 8] = std::array::from_fn(|i| (*p.add(i)).to_f32());
            _mm256_loadu_ps(w.as_ptr())
        }
    }
}

/// A zeroed buffer of `len` T starting on a 64-byte boundary (so cache rows
/// whose byte length is a multiple of 64 start on a cache line). Allocated
/// with 16-byte alignment and 64 spare bytes, so `alloc_zeroed` is calloc
/// (fresh zero pages; an over-aligned layout would memset the whole buffer):
/// untouched context costs no resident memory.
pub struct KvBuf<T: KvElem> {
    base: std::ptr::NonNull<u8>,
    ptr: std::ptr::NonNull<T>,
    len: usize,
}

// SAFETY: KvBuf owns its allocation like a Vec<T>.
unsafe impl<T: KvElem> Send for KvBuf<T> {}
// SAFETY: as above; shared access is read-only.
unsafe impl<T: KvElem> Sync for KvBuf<T> {}

impl<T: KvElem> KvBuf<T> {
    fn layout(len: usize) -> std::alloc::Layout {
        let bytes = len * std::mem::size_of::<T>() + 64;
        std::alloc::Layout::from_size_align(bytes, 16).expect("kv cache size")
    }

    pub fn zeroed(len: usize) -> KvBuf<T> {
        let layout = Self::layout(len);
        // SAFETY: layout has nonzero size.
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        let base =
            std::ptr::NonNull::new(p).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        // Start at the first 64-byte boundary, within the 64 spare bytes. All
        // zero bits are a valid T (KvElem); 64-byte alignment satisfies T's.
        let off = p.align_offset(64);
        assert!(off < 64, "kv cache alignment");
        // SAFETY: off < 64, inside the allocation.
        let ptr = unsafe { base.add(off) };
        KvBuf {
            base,
            ptr: ptr.cast(),
            len,
        }
    }
}

impl<T: KvElem> Drop for KvBuf<T> {
    fn drop(&mut self) {
        // SAFETY: allocated in `zeroed` with this layout.
        unsafe { std::alloc::dealloc(self.base.as_ptr(), Self::layout(self.len)) }
    }
}

impl<T: KvElem> std::ops::Deref for KvBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: len initialized (zeroed) T at ptr, owned by self.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: KvElem> std::ops::DerefMut for KvBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as above, borrowed mutably.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

/// Per-layer K and V of one element type.
pub struct Layers<T: KvElem> {
    pub k: Vec<KvBuf<T>>,
    pub v: Vec<KvBuf<T>>,
}

impl<T: KvElem> Layers<T> {
    pub fn new(n_layers: usize, len: usize) -> Layers<T> {
        Layers {
            k: (0..n_layers).map(|_| KvBuf::zeroed(len)).collect(),
            v: (0..n_layers).map(|_| KvBuf::zeroed(len)).collect(),
        }
    }
}

/// The cache's storage, picked at `new_cache` time.
pub enum KvData {
    F16(Layers<Half>),
    F32(Layers<f32>),
}

/// Whether new caches store f32 (AXON_KV=f32) rather than f16 (default,
/// or AXON_KV=f16). Read once.
pub fn use_f32() -> bool {
    static F32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F32.get_or_init(|| match std::env::var("AXON_KV").ok().as_deref() {
        None | Some("") | Some("f16") => false,
        Some("f32") => true,
        Some(other) => panic!("AXON_KV={other}: expected f16 or f32"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar rounding == F16C's `vcvtps2ph` (when the build has it) over a
    /// spread of every f32 class (stride sample of all 2^32 bit patterns plus
    /// every half-way case near each f16 exponent), and widening is exact.
    /// Buffers start on a cache line and read back zero.
    #[test]
    fn kvbuf_aligned_zeroed() {
        for len in [0, 1, 7, 1000, 1 << 20] {
            let mut b = KvBuf::<Half>::zeroed(len);
            assert_eq!(b.as_ptr() as usize % 64, 0);
            assert!(b.iter().all(|h| h.0 == 0));
            if len > 0 {
                b[len - 1] = Half(1);
            }
        }
    }

    #[test]
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // up may be NaN past +inf: !(a < b) is intended
    fn half_store_matches_scalar() {
        let mut xs: Vec<f32> = (0..(1u64 << 32))
            .step_by(4099)
            .map(|b| f32::from_bits(b as u32))
            .collect();
        for e in 100u32..145 {
            for m in [
                0x1000u32, 0x1001, 0x0fff, 0x3000, 0x2000, 0x7f_f000, 0x7f_ffff, 1,
            ] {
                for s in [0, 1u32 << 31] {
                    xs.push(f32::from_bits(s | (e << 23) | m));
                }
            }
        }
        xs.extend([
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            65504.0,
            65520.0,
            65519.99,
        ]);
        xs.truncate(xs.len() / 8 * 8);
        let mut v = vec![Half(0); xs.len()];
        Half::store(&mut v, &xs);
        for (x, h) in xs.iter().zip(&v) {
            let s = Half::from_f32(*x);
            assert_eq!(h.0, s.0, "{x:e} ({:08x})", x.to_bits());
            let back = h.to_f32();
            if x.is_nan() {
                assert!(back.is_nan());
            } else {
                assert_eq!(Half::from_f32(back).0, h.0, "{x:e}");
                // nearest: no other half is closer
                let up = Half(h.0.wrapping_add(1)).to_f32();
                assert!(!((up - x).abs() < (back - x).abs()), "{x:e}");
            }
        }
    }
}
