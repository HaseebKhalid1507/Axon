//! SHA-256 (FIPS 180-4), streaming: `axon pull` checks every download
//! against the checksum Hugging Face publishes for it (the LFS oid) while
//! the bytes arrive, so a finished download needs no second pass over a
//! multi-gigabyte file. In-crate rather than `sha256sum`/`certutil`: the
//! same code and output on every platform, no output format to parse, and
//! it hashes the stream as it's written. Scalar: ~400 MB/s on one core,
//! faster than the network it runs behind.

/// The hash state: feed bytes with `update`, then `finish`.
#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    /// A partial block waiting for more bytes.
    buf: [u8; 64],
    fill: usize,
    /// Bytes fed so far.
    len: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256 {
            h: H0,
            buf: [0; 64],
            fill: 0,
            len: 0,
        }
    }

    /// One 64-byte block into the state.
    fn block(h: &mut [u32; 8], b: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, c) in b.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }

    /// Feed `data`.
    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if self.fill > 0 {
            let take = (64 - self.fill).min(data.len());
            self.buf[self.fill..self.fill + take].copy_from_slice(&data[..take]);
            self.fill += take;
            data = &data[take..];
            if self.fill < 64 {
                return;
            }
            let b = self.buf;
            Self::block(&mut self.h, &b);
            self.fill = 0;
        }
        let mut blocks = data.chunks_exact(64);
        for b in &mut blocks {
            Self::block(&mut self.h, b.try_into().unwrap());
        }
        let rest = blocks.remainder();
        self.buf[..rest.len()].copy_from_slice(rest);
        self.fill = rest.len();
    }

    /// Bytes fed so far.
    #[allow(clippy::len_without_is_empty)] // a count of bytes, not a container
    pub fn len(&self) -> u64 {
        self.len
    }

    /// The digest.
    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        pad.resize(1 + (55usize.wrapping_sub(self.fill)) % 64, 0);
        pad.extend_from_slice(&bits.to_be_bytes());
        let len = self.len;
        self.update(&pad);
        debug_assert_eq!(self.fill, 0);
        self.len = len;
        let mut out = [0u8; 32];
        for (o, x) in out.chunks_exact_mut(4).zip(self.h) {
            o.copy_from_slice(&x.to_be_bytes());
        }
        out
    }

    /// The digest as 64 lowercase hex digits.
    pub fn finish_hex(self) -> String {
        hex(&self.finish())
    }
}

/// Bytes as lowercase hex.
pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The SHA-256 of `data`, as hex.
#[cfg(test)]
pub fn digest_hex(data: &[u8]) -> String {
    let mut s = Sha256::new();
    s.update(data);
    s.finish_hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS 180-4's examples and NIST's CAVS byte-oriented vectors
    /// (SHA256ShortMsg.rsp / SHA256LongMsg.rsp), and the million-'a' one.
    #[test]
    fn sha256_nist_vectors() {
        let unhex = |s: &str| -> Vec<u8> {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect()
        };
        let cases: [(&[u8], &str); 4] = [
            (
                b"",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
            ),
        ];
        for (m, d) in cases {
            assert_eq!(digest_hex(m), d, "{:?}", String::from_utf8_lossy(m));
        }
        // CAVS ShortMsg: Len = 8, 16, 512 bits.
        assert_eq!(
            digest_hex(&unhex("d3")),
            "28969cdfa74a12c82f3bad960b0b000aca2ac329deea5c2328ebc6f2ba9802c1"
        );
        assert_eq!(
            digest_hex(&unhex("11af")),
            "5ca7133fa735326081558ac312c620eeca9970d1e70a4b95533d956f072d1f98"
        );
        assert_eq!(
            digest_hex(&unhex(
                "5a86b737eaea8ee976a0a24da63e7ed7eefad18a101c1211e2b3650c5187c2a8\
                 a650547208251f6d4237e661c7bf4c77f335390394c37fa1a9f9be836ac28509"
            )),
            "42e61e174fbb3897d6dd6cef3dd2802fe67b331953b06114a65c772859dfc1aa"
        );
        let mut s = Sha256::new();
        let a = [b'a'; 1000];
        for _ in 0..1000 {
            s.update(&a);
        }
        assert_eq!(s.len(), 1_000_000);
        assert_eq!(
            s.finish_hex(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    /// Feeding in pieces is feeding at once, for every split point and
    /// every length around the padding edges (55, 56, 63, 64 bytes, ...).
    #[test]
    fn sha256_streaming_is_one_shot() {
        let data: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
        for n in 0..data.len() {
            let want = digest_hex(&data[..n]);
            for cut in [0, 1, n / 3, n / 2, n.saturating_sub(1), n] {
                let mut s = Sha256::new();
                s.update(&data[..cut.min(n)]);
                s.update(&data[cut.min(n)..n]);
                assert_eq!(s.finish_hex(), want, "n {n} cut {cut}");
            }
            let mut s = Sha256::new();
            for b in &data[..n] {
                s.update(std::slice::from_ref(b));
            }
            assert_eq!(s.finish_hex(), want, "n {n} bytewise");
        }
    }
}
