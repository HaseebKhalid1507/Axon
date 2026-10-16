# Independent model of identity::digest_file (axon-model-digest/2) in Python hashlib: prints the
# digests model_digest_golden pins. Run: python3 scripts/identity_digest_ref.py
import hashlib, struct
EDGE, WIN, WINS = 256, 4096, 1024
def gen(n): return bytes(((i * 2654435761) & 0xffffffff) >> 13 & 0xff for i in range(n))
def ranges(ln, off, ts):
    r = [(0, off)]
    for at, n in ts:
        e = min(n, EDGE); r += [(at, at + e), (at + n - e, at + n)]
    data = max(ln - off, 0)
    if data > 0:
        span = max(data - WIN, 0)
        for i in range(WINS):
            a = off + span * i // (WINS - 1); r.append((a, a + WIN))
    r = sorted((min(a, ln), min(b, ln)) for a, b in r if min(a, ln) < min(b, ln))
    out = []
    for a, b in r:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
def digest(b, off, ts):
    h = hashlib.sha256(b"axon-model-digest/2\0" + struct.pack("<Q", len(b)))
    for a, e in ranges(len(b), off, ts): h.update(b[a:e])
    return h.hexdigest(), sum(e - a for a, e in ranges(len(b), off, ts))
L = 6 * 1024 * 1024 + 123
big = gen(L)
tb = [(777, 0), (777, 300), (1077, 100), (1177, 5_000_000), (5_001_177, L - 5_001_177)]
print("big", *digest(big, 777, tb))
tiny = gen(3000)
print("tiny", *digest(tiny, 100, [(100, 0), (100, 2900)]))
print("nodata", *digest(gen(50), 50, []))
