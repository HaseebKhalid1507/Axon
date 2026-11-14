#!/usr/bin/env python3
"""packcmp.py CI.pack REF.pack [REF.pack...]: every entry of a kernel pack built elsewhere (the release workflow's
`kernels-pack` artifact) against the same (module, target) entry of packs built with `axon kernels build` by the
NVRTC that bella's md5 gate trusts (the REF packs), by the fnv128 of NVRTC's raw output each entry records
(src/cuda/pack.rs format). Prints one line per target and a verdict; exit 0 iff every CI entry has a REF twin with the
same raw hash and both packs name the same NVRTC version. docs/releasing.md: a hand check before publishing."""
import struct, sys
KIND = {0: "sm_", 1: "compute_", 2: "sm_90a"}
def parse(path):
    b = open(path, "rb").read()
    assert b[:8] == b"AXONPAK\x01", f"{path}: not a pack"
    maj, mnr, n = struct.unpack_from("<III", b, 8)
    ents = {}
    for i in range(n):
        o = 32 + 80 * i
        module = int.from_bytes(b[o:o + 16], "little")
        kind, form, arch = struct.unpack_from("<BBH", b, o + 16)
        raw = int.from_bytes(b[o + 40:o + 56], "little")
        t = "sm_90a" if kind == 2 else f"{KIND[kind]}{arch}"
        assert (module, t) not in ents, f"{path}: duplicate entry"
        ents[(module, t)] = raw
    return (maj, mnr), ents
ci_v, ci = parse(sys.argv[1])
ref, ref_v = {}, set()
for p in sys.argv[2:]:
    v, e = parse(p); ref_v.add(v); ref.update(e)
print(f"CI pack: NVRTC {ci_v[0]}.{ci_v[1]}, {len(ci)} entries; bella packs: NVRTC {sorted(ref_v)}, {len(ref)} entries")
bad = 0
for t in sorted({t for _, t in ci}, key=lambda t: (t.startswith("compute"), t)):
    keys = [k for k in ci if k[1] == t]
    same = sum(1 for k in keys if ref.get(k) == ci[k])
    miss = sum(1 for k in keys if k not in ref)
    diff = len(keys) - same - miss
    bad += miss + diff
    print(f"  {t:12} {len(keys):3} entries: {same} same raw bytes, {diff} different, {miss} not built on bella")
ok = bad == 0 and ref_v == {ci_v}
print("VERDICT:", "PASS: every CI entry is bella NVRTC's output, byte for byte" if ok else f"FAIL ({bad} entries)")
sys.exit(0 if ok else 1)
