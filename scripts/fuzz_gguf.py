#!/usr/bin/env python3
"""Mutation fuzzer for axon's GGUF reader and model loader (stdlib only).

    scripts/fuzz_gguf.py <axon binary> <model.gguf> <work dir> [--jobs N] [--only SUBSTR]
                         [--keep] [--mem-mb MB] [--timeout S]

Takes a real GGUF, writes mutants of its header (metadata + tensor table) and
of its byte layout, and runs `axon info` and `axon gen <m> Hello -n 1` (CPU)
on each one under a virtual-memory cap (RLIMIT_AS) and a timeout. A mutant
must either load or be refused with an error message: a panic, a signal, an
allocation failure, or a run past the timeout is a finding. Prints one line
per run and a summary; exits 1 if there is a finding.

Mutants share the weights with one base copy of the model through btrfs/XFS
reflinks (copy_file_range of whole blocks), so each costs a few KB of disk:
the base copy is the model with a padding key so that its tensor data starts
on a 4096-byte boundary, and every mutant pads its header the same way.
Where the filesystem can't reflink, the kernel copies instead (slow, and
the full model size per live mutant: run with --jobs 1 then).

Out of scope: a file truncated or rewritten while axon has it mapped (SIGBUS
is the documented behaviour of any mmap'd file, see gguf.rs `load_to_run`).
"""

import argparse
import os
import resource
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

MAGIC = 0x46554747
BLOCK = 4096
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
FIXED = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f",
         BOOL: "<B", U64: "<Q", I64: "<q", F64: "<d"}


def gstr(b):
    if isinstance(b, str):
        b = b.encode()
    return struct.pack("<Q", len(b)) + b


def enc(t, v):
    """Serialize value v of GGUF type t (arrays: v = (elem type, [values]))."""
    if t in FIXED:
        return struct.pack(FIXED[t], v)
    if t == STR:
        return gstr(v)
    if t == ARR:
        et, items = v
        return struct.pack("<IQ", et, len(items)) + b"".join(enc(et, x) for x in items)
    raise ValueError(t)


class Reader:
    def __init__(self, b):
        self.b, self.p = b, 0

    def get(self, fmt):
        v = struct.unpack_from(fmt, self.b, self.p)
        self.p += struct.calcsize(fmt)
        return v[0]

    def str(self):
        n = self.get("<Q")
        s = self.b[self.p:self.p + n]
        self.p += n
        return s

    def skip_val(self, t):
        if t in FIXED:
            self.p += struct.calcsize(FIXED[t])
        elif t == STR:
            self.str()
        elif t == ARR:
            et, n = self.get("<I"), self.get("<Q")
            if et in FIXED:
                self.p += n * struct.calcsize(FIXED[et])
            else:
                for _ in range(n):
                    self.skip_val(et)
        else:
            raise ValueError(f"type {t}")

    def val(self, t):
        if t in FIXED:
            return self.get(FIXED[t])
        if t == STR:
            return self.str()
        raise ValueError(t)


class Gguf:
    """The parsed header: kvs = [(key bytes, type, raw value bytes)],
    tensors = [[name bytes, dims, type, offset]], plus where things are."""

    def __init__(self, path):
        with open(path, "rb") as f:
            head = f.read(64 << 20)
        r = Reader(head)
        assert r.get("<I") == MAGIC, "not a GGUF"
        self.version = r.get("<I")
        nt, nk = r.get("<Q"), r.get("<Q")
        self.kvs = []
        for _ in range(nk):
            k = r.str()
            t = r.get("<I")
            s = r.p
            r.skip_val(t)
            self.kvs.append((k, t, head[s:r.p]))
        self.tensors = []
        for _ in range(nt):
            name = r.str()
            nd = r.get("<I")
            dims = [r.get("<Q") for _ in range(nd)]
            ty = r.get("<I")
            off = r.get("<Q")
            self.tensors.append([name, dims, ty, off])
        self.align = 32
        for k, t, v in self.kvs:
            if k == b"general.alignment":
                self.align = struct.unpack("<I", v)[0]
        self.header_len = r.p
        self.data_offset = -(-r.p // self.align) * self.align
        self.file_len = os.path.getsize(path)
        self.data_len = self.file_len - self.data_offset
        self.arch = self.get_str(b"general.architecture") or b""

    def get_raw(self, key):
        for k, t, v in self.kvs:
            if k == key:
                return t, v
        return None

    def get_str(self, key):
        x = self.get_raw(key)
        if x and x[0] == STR:
            return Reader(x[1]).str()
        return None

    def get_u32(self, key):
        x = self.get_raw(key)
        if x and x[0] == U32:
            return struct.unpack("<I", x[1])[0]
        return None


def serialize(version, kvs, tensors, n_kv=None, n_t=None, tail=b""):
    out = [struct.pack("<IIQQ", MAGIC, version,
                       len(tensors) if n_t is None else n_t,
                       len(kvs) if n_kv is None else n_kv)]
    for k, t, v in kvs:
        out.append(gstr(k) + struct.pack("<I", t) + v)
    for name, dims, ty, off in tensors:
        out.append(gstr(name) + struct.pack("<I", len(dims))
                   + b"".join(struct.pack("<Q", d) for d in dims)
                   + struct.pack("<IQ", ty, off))
    return b"".join(out) + tail


PAD_KEY = b"fuzz.pad"


def padded(version, kvs, tensors, align, **kw):
    """The header with a padding key appended last to the metadata so that
    its data starts (aligned to `align`) on a BLOCK boundary. Returns
    (header bytes, data offset)."""
    kw2 = dict(kw)
    if kw2.get("n_kv") is not None:
        kw2["n_kv"] += 1
    base = serialize(version, kvs + [(PAD_KEY, STR, gstr(b""))], tensors, **kw2)
    pad = (-len(base)) % BLOCK
    h = serialize(version, kvs + [(PAD_KEY, STR, gstr(b"p" * pad))], tensors, **kw2)
    assert len(h) % BLOCK == 0 and len(h) % align == 0
    return h, len(h)


def copy_range(src_fd, dst_fd, n, src_off, dst_off):
    while n > 0:
        k = os.copy_file_range(src_fd, dst_fd, n, src_off, dst_off)
        if k <= 0:
            raise OSError("copy_file_range returned 0")
        n, src_off, dst_off = n - k, src_off + k, dst_off + k


class Base:
    """The model re-laid with its data on a block boundary (`padded`)."""

    def __init__(self, src, work):
        self.g = Gguf(src)
        g = self.g
        self.path = os.path.join(work, "base.gguf")
        h, do = padded(g.version, g.kvs, g.tensors, g.align)
        self.do, self.data_len = do, g.data_len
        if not (os.path.exists(self.path) and os.path.getsize(self.path) == do + g.data_len):
            tmp = self.path + ".tmp"
            with open(src, "rb") as s, open(tmp, "wb") as d:
                d.write(h)
                d.flush()
                copy_range(s.fileno(), d.fileno(), g.data_len, g.data_offset, do)
            os.replace(tmp, self.path)

    def write(self, path, header, data_at=None, data_len=None, raw_prefix=None):
        """A mutant: `header`, then (from `data_at`, default the next block)
        `data_len` bytes of the base's tensor data (default all of it).
        raw_prefix=N: the first N bytes of the base file instead."""
        with open(self.path, "rb") as s, open(path, "wb") as d:
            if raw_prefix is not None:
                whole = raw_prefix // BLOCK * BLOCK
                if whole:
                    copy_range(s.fileno(), d.fileno(), whole, 0, 0)
                s.seek(whole)
                d.seek(whole)
                d.write(s.read(raw_prefix - whole))
                return
            d.write(header)
            at = -(-len(header) // BLOCK) * BLOCK if data_at is None else data_at
            n = self.data_len if data_len is None else data_len
            d.truncate(at)
            d.flush()
            if n:
                copy_range(s.fileno(), d.fileno(), n, self.do, at)


def mutants(base):
    """(name, writer) pairs; writer(path) writes the mutant."""
    g = base.g
    A = g.align
    V, K, T = g.version, g.kvs, g.tensors
    arch = g.arch
    out = []

    def hdr(name, kvs=None, tensors=None, align=A, **kw):
        def w(p, kvs=kvs, tensors=tensors):
            kvs = K if kvs is None else kvs
            tensors = T if tensors is None else tensors
            if kw.get("n_kv") is not None:
                # A wrong metadata count shifts everything after it: no
                # padding key (the count can't name it), no data needed.
                base.write(p, serialize(V, kvs, tensors, **kw), data_len=0)
                return
            h, _ = padded(V, kvs, tensors, align, **kw)
            base.write(p, h)
        out.append((name, w))

    def raw(name, f):
        out.append((name, f))

    def set_kv(key, t, v):
        """kvs with key set to (t, raw v); appended if missing."""
        key = key.encode() if isinstance(key, str) else key
        res, hit = [], False
        for k, tt, vv in K:
            if k == key:
                res.append((k, t, v))
                hit = True
            else:
                res.append((k, tt, vv))
        if not hit:
            res.append((key, t, v))
        return res

    def del_kv(key):
        key = key.encode() if isinstance(key, str) else key
        return [kv for kv in K if kv[0] != key]

    def set_t(i, **f):
        ts = [list(t) for t in T]
        for name, v in f.items():
            ts[i][{"name": 0, "dims": 1, "type": 2, "offset": 3}[name]] = v
        return ts

    # --- 1. Truncation at every section boundary (and one byte into each field).
    h, do = padded(V, K, T, A)
    bounds = [0, 1, 4, 5, 8, 12, 16, 20, 24]
    p = 24
    pad_kv_at = None
    for k, t, v in K + [(PAD_KEY, STR, None)]:
        if k == PAD_KEY:
            pad_kv_at = p
            break
        p_key = p + 8 + len(k)
        p_type = p_key + 4
        bounds += [p, p + 3, p + 8, p_key, p_type - 1, p_type, p_type + 1, p_type + len(v) - 1]
        p = p_type + len(v)
    # Skip the pad key itself, then each tensor info.
    p = pad_kv_at
    p += 8 + len(PAD_KEY) + 4
    p += struct.unpack_from("<Q", h, p)[0] + 8
    for name, dims, ty, off in T:
        e = p + 8 + len(name) + 4 + 8 * len(dims) + 4 + 8
        bounds += [p, p + 8 + len(name), e - 8, e - 1]
        p = e
    bounds += [p, do - 1, do, do + 1, do + base.data_len // 2]
    # The end of each tensor's data, and one byte short of it.
    ends = sorted({t[3] + tensor_bytes(t) for t in T})
    bounds += [do + e - 1 for e in ends[-8:]] + [do + base.data_len - 1]
    for b in sorted(set(bounds)):
        raw(f"trunc@{b}", lambda pth, b=b: base.write(pth, None, raw_prefix=b))

    # --- 2. Header counts.
    hdr("n_tensors+1", n_t=len(T) + 1)
    hdr("n_tensors-1", n_t=len(T) - 1)
    hdr("n_tensors=u64max", n_t=2**64 - 1)
    hdr("n_tensors=2^61", n_t=2**61)
    hdr("n_kv=u64max", n_kv=2**64 - 1)
    hdr("n_kv=2^60", n_kv=2**60)
    hdr("n_kv+1000", n_kv=len(K) + 1000)
    hdr("n_kv=0", n_kv=0)
    raw("version=1", lambda pth: base.write(pth, struct.pack("<IIQQ", MAGIC, 1, 0, 0)))
    raw("version=4", lambda pth: base.write(pth, padded(4, K, T, A)[0]))
    raw("big-endian", lambda pth: base.write(pth, struct.pack(">IIQQ", MAGIC, 3, 0, 0)))
    raw("bad-magic", lambda pth: base.write(pth, b"GGUG" + padded(V, K, T, A)[0][4:]))
    raw("empty-file", lambda pth: open(pth, "wb").close())
    raw("only-magic+ver", lambda pth: base.write(pth, struct.pack("<II", MAGIC, 3)))
    raw("no-tensors-no-kv", lambda pth: base.write(pth, struct.pack("<IIQQ", MAGIC, 3, 0, 0), data_len=0))

    # --- 3. Metadata encoding.
    first = K[0][0]
    hdr("key-len-past-eof", kvs=[(b"x", STR, struct.pack("<Q", 2**40))] + K)
    hdr("str-len-u64max", kvs=[(b"x", STR, struct.pack("<Q", 2**64 - 1))] + K)
    hdr("str-len-past-eof", kvs=K + [(b"x", STR, struct.pack("<Q", 10**12))])
    hdr("str-bad-utf8", kvs=[(b"x", STR, gstr(b"\xff\xfe\xc0"))] + K)
    hdr("key-bad-utf8", kvs=[(b"\xff\xfe", U32, enc(U32, 1))] + K)
    hdr("key-empty", kvs=[(b"", U32, enc(U32, 1))] + K)
    hdr("key-with-nul", kvs=[(b"general.archi\x00tecture", STR, gstr(b"qwen2"))] + K)
    hdr("val-type-13", kvs=[(b"x", 13, b"")] + K)
    hdr("val-type-u32max", kvs=[(b"x", 2**32 - 1, b"")] + K)
    hdr("arr-len-u64max", kvs=[(b"x", ARR, struct.pack("<IQ", U8, 2**64 - 1))] + K)
    hdr("arr-len-2^62-u32", kvs=[(b"x", ARR, struct.pack("<IQ", U32, 2**62))] + K)
    hdr("arr-len-2^63-u64", kvs=[(b"x", ARR, struct.pack("<IQ", U64, 2**63))] + K)
    hdr("arr-of-str-huge", kvs=[(b"x", ARR, struct.pack("<IQ", STR, 2**60))] + K)
    hdr("arr-elem-type-bad", kvs=[(b"x", ARR, struct.pack("<IQ", 77, 1))] + K)
    nested = struct.pack("<IQ", ARR, 1)
    for _ in range(64):
        nested += struct.pack("<IQ", ARR, 1)
    hdr("arr-nested-65", kvs=[(b"x", ARR, nested + struct.pack("<IQ", U8, 0))] + K)
    hdr("arr-nested-4", kvs=[(b"x", ARR, struct.pack("<IQ", ARR, 1) * 3 + struct.pack("<IQ", U8, 1) + b"\x01")] + K)
    # Memory amplification: each array element becomes an in-memory value
    # many times its encoded size (1 byte for a u8).
    hdr("arr-u8-150M", kvs=[(b"x", ARR, struct.pack("<IQ", U8, 150_000_000) + bytes(150_000_000))] + K)
    hdr("arr-arr-10M", kvs=[(b"x", ARR, struct.pack("<IQ", ARR, 10_000_000)
                                    + struct.pack("<IQ", U8, 0) * 10_000_000)] + K)
    hdr("dup-key-arch-first-bogus", kvs=[(b"general.architecture", STR, gstr(b"gemma"))] + K)
    hdr("dup-key-arch-last-bogus", kvs=K + [(b"general.architecture", STR, gstr(b"gemma"))])
    hdr("dup-key-block_count-last", kvs=K + [(f"{arch.decode()}.block_count".encode(), U32, enc(U32, 1000))])
    hdr("dup-key-block_count-first", kvs=[(f"{arch.decode()}.block_count".encode(), U32, enc(U32, 2))] + K)

    # --- 4. Known keys: missing, wrong type, bad values.
    a = arch.decode()
    num_keys = ["block_count", "embedding_length", "attention.head_count", "attention.head_count_kv",
                "context_length", "feed_forward_length", "attention.key_length", "attention.value_length",
                "rope.dimension_count"]
    for nk in num_keys:
        key = f"{a}.{nk}"
        hdr(f"del:{nk}", kvs=del_kv(key))
        hdr(f"{nk}=str", kvs=set_kv(key, STR, gstr(b"24")))
        hdr(f"{nk}=0", kvs=set_kv(key, U32, enc(U32, 0)))
        hdr(f"{nk}=1", kvs=set_kv(key, U32, enc(U32, 1)))
        hdr(f"{nk}=u32max", kvs=set_kv(key, U32, enc(U32, 2**32 - 1)))
        hdr(f"{nk}=u64huge", kvs=set_kv(key, U64, enc(U64, 2**63)))
        hdr(f"{nk}=i32neg", kvs=set_kv(key, I32, enc(I32, -1)))
        hdr(f"{nk}=f32-1.5", kvs=set_kv(key, F32, enc(F32, 1.5)))
        hdr(f"{nk}=f32nan", kvs=set_kv(key, F32, enc(F32, float("nan"))))
        hdr(f"{nk}=f64inf", kvs=set_kv(key, F64, enc(F64, float("inf"))))
        hdr(f"{nk}=arr", kvs=set_kv(key, ARR, enc(ARR, (U32, [1, 2]))))
        hdr(f"{nk}=bool", kvs=set_kv(key, BOOL, enc(BOOL, 1)))
        hdr(f"{nk}=u8", kvs=set_kv(key, U8, enc(U8, 7)))
    hdr("ctx=2^31", kvs=set_kv(f"{a}.context_length", U32, enc(U32, 2**31)))
    hdr("ctx=7", kvs=set_kv(f"{a}.context_length", U32, enc(U32, 7)))
    for fk in ["attention.layer_norm_rms_epsilon", "rope.freq_base"]:
        key = f"{a}.{fk}"
        for nm, t, v in [("0", F32, 0.0), ("neg", F32, -1.0), ("nan", F32, float("nan")),
                         ("inf", F32, float("inf")), ("huge", F32, 3e38), ("str", STR, b"1e-6"),
                         ("u64", U64, 2**63)]:
            hdr(f"{fk}={nm}", kvs=set_kv(key, t, enc(t, v)))
        hdr(f"del:{fk}", kvs=del_kv(key))
    for sk, nm, t, v in [
        ("rope.scaling.type", "yarn", STR, b"yarn"), ("rope.scaling.type", "bogus", STR, b"weird"),
        ("rope.scaling.type", "u32", U32, 1), ("rope.scaling.factor", "4", F32, 4.0),
        ("rope.scaling.factor", "nan", F32, float("nan")), ("rope.scaling.attn_factor", "nan", F32, float("nan")),
        ("expert_count", "8", U32, 8), ("expert_count", "str", STR, b"8"),
        ("attention.scale", "nan", F32, float("nan")), ("attention.sliding_window", "4", U32, 4),
    ]:
        hdr(f"{sk}={nm}", kvs=set_kv(f"{a}.{sk}", t, enc(t, v)))
    hdr("arch=u32", kvs=set_kv("general.architecture", U32, enc(U32, 2)))
    hdr("arch=empty", kvs=set_kv("general.architecture", STR, gstr(b"")))
    hdr("arch=gemma3", kvs=set_kv("general.architecture", STR, gstr(b"gemma3")))
    hdr("arch=llama-on-qwen2" if a != "llama" else "arch=qwen2-on-llama",
        kvs=[(k.replace(arch + b".", b"llama." if a != "llama" else b"qwen2."), t, v) if k != b"general.architecture"
             else (k, STR, gstr(b"llama" if a != "llama" else b"qwen2")) for k, t, v in K])
    hdr("del:arch", kvs=del_kv("general.architecture"))
    for al_nm, t, v in [("0", U32, 0), ("1", U32, 1), ("3", U32, 3), ("2^31", U32, 2**31),
                        ("u64-2^40", U64, 2**40), ("str", STR, b"32"), ("f32", F32, 32.0), ("i32", I32, 32),
                        ("u8-32", U8, 32), ("u16-64", U16, 64), ("64", U32, 64), ("4096", U32, 4096)]:
        hdr(f"alignment={al_nm}", kvs=set_kv("general.alignment", t, enc(t, v)))

    # --- 5. Tokenizer arrays.
    tok_t = g.get_raw(b"tokenizer.ggml.tokens")
    n_vocab = struct.unpack_from("<Q", tok_t[1], 4)[0] if tok_t else 0
    hdr("del:tokens", kvs=del_kv("tokenizer.ggml.tokens"))
    hdr("tokens=empty", kvs=set_kv("tokenizer.ggml.tokens", ARR, enc(ARR, (STR, []))))
    hdr("tokens=u32arr", kvs=set_kv("tokenizer.ggml.tokens", ARR, enc(ARR, (U32, list(range(1000))))))
    hdr("tokens=str", kvs=set_kv("tokenizer.ggml.tokens", STR, gstr(b"hello")))
    hdr("tokens=1", kvs=set_kv("tokenizer.ggml.tokens", ARR, enc(ARR, (STR, [b"a"]))))
    hdr("tokens=all-empty", kvs=set_kv("tokenizer.ggml.tokens", ARR, enc(ARR, (STR, [b""] * 1000))))
    hdr("tokens=all-same", kvs=set_kv("tokenizer.ggml.tokens", ARR, enc(ARR, (STR, [b"a"] * 1000))))
    hdr("tokens=vocab+1", kvs=set_kv("tokenizer.ggml.tokens", ARR,
                                     tok_t[1][:4] + struct.pack("<Q", n_vocab + 1) + tok_t[1][12:] + gstr(b"extra")))
    # Only control tokens: a pathological special-token list (every string a special).
    hdr("token_type=all-control", kvs=set_kv("tokenizer.ggml.token_type", ARR, enc(ARR, (I32, [3] * n_vocab))))
    hdr("token_type=short", kvs=set_kv("tokenizer.ggml.token_type", ARR, enc(ARR, (I32, [1, 3]))))
    hdr("token_type=f32", kvs=set_kv("tokenizer.ggml.token_type", ARR, enc(ARR, (F32, [1.0] * 10))))
    hdr("token_type=str", kvs=set_kv("tokenizer.ggml.token_type", STR, gstr(b"x")))
    hdr("token_type=long", kvs=set_kv("tokenizer.ggml.token_type", ARR, enc(ARR, (I32, [1] * (n_vocab + 500)))))
    hdr("del:merges", kvs=del_kv("tokenizer.ggml.merges"))
    hdr("merges=garbage", kvs=set_kv("tokenizer.ggml.merges", ARR,
                                     enc(ARR, (STR, [b"", b" ", b"a", b"  ", b"\xc4\xa0 \xc4\xa0", b"zz zz zz"] * 100))))
    hdr("merges=u32", kvs=set_kv("tokenizer.ggml.merges", ARR, enc(ARR, (U32, [1, 2, 3]))))
    hdr("merges=str", kvs=set_kv("tokenizer.ggml.merges", STR, gstr(b"a b")))
    for idk in ["eos_token_id", "bos_token_id", "padding_token_id"]:
        key = f"tokenizer.ggml.{idk}"
        for nm, t, v in [("u32max", U32, 2**32 - 1), ("vocab", U32, n_vocab), ("i32neg", I32, -1),
                         ("str", STR, b"1"), ("u64huge", U64, 2**63), ("f32", F32, 1.5)]:
            hdr(f"{idk}={nm}", kvs=set_kv(key, t, enc(t, v)))
    hdr("add_bos=1", kvs=set_kv("tokenizer.ggml.add_bos_token", BOOL, enc(BOOL, 1)))
    hdr("add_bos=1+bos=u32max", kvs=set_kv("tokenizer.ggml.add_bos_token", BOOL, enc(BOOL, 1))
        + [(b"tokenizer.ggml.bos_token_id", U32, enc(U32, 2**32 - 1))])
    for nm, t, v in [("unknown", STR, b"deepseek-llm"), ("u32", U32, 1), ("empty", STR, b"")]:
        hdr(f"pre={nm}", kvs=set_kv("tokenizer.ggml.pre", t, enc(t, v)))
    hdr("del:pre", kvs=del_kv("tokenizer.ggml.pre"))
    for nm, v in [("llama", b"llama"), ("bert", b"bert"), ("u32", None)]:
        if v is None:
            hdr(f"model={nm}", kvs=set_kv("tokenizer.ggml.model", U32, enc(U32, 1)))
        else:
            hdr(f"model={nm}", kvs=set_kv("tokenizer.ggml.model", STR, gstr(v)))
    hdr("chat_template=huge-garbage", kvs=set_kv("tokenizer.chat_template", STR, gstr(b"{%" * 100000)))
    hdr("chat_template=u32", kvs=set_kv("tokenizer.chat_template", U32, enc(U32, 1)))

    # --- 6. Tensor table.
    idx = {t[0]: i for i, t in enumerate(T)}
    emb = idx.get(b"token_embd.weight")
    q0 = idx.get(b"blk.0.attn_q.weight")
    n0 = idx.get(b"blk.0.attn_norm.weight")
    last = len(T) - 1
    for nm, i in [("embd", emb), ("q0", q0), ("norm0", n0), ("last", last)]:
        if i is None:
            continue
        name, dims, ty, off = T[i]
        hdr(f"t[{nm}].offset+1", tensors=set_t(i, offset=off + 1))
        hdr(f"t[{nm}].offset+align", tensors=set_t(i, offset=off + A))
        hdr(f"t[{nm}].offset=past-eof", tensors=set_t(i, offset=(base.data_len // A + 1) * A))
        hdr(f"t[{nm}].offset=u64max-al", tensors=set_t(i, offset=2**64 - A))
        hdr(f"t[{nm}].offset=0", tensors=set_t(i, offset=0))
        hdr(f"t[{nm}].ndims=5", tensors=set_t(i, dims=dims + [1, 1, 1, 1][:5 - len(dims)]))
        hdr(f"t[{nm}].ndims=0", tensors=set_t(i, dims=[]))
        hdr(f"t[{nm}].dim0=0", tensors=set_t(i, dims=[0] + dims[1:]))
        hdr(f"t[{nm}].dim=2^40", tensors=set_t(i, dims=[dims[0], 2**40] + dims[2:]))
        hdr(f"t[{nm}].dims-overflow", tensors=set_t(i, dims=[2**32, 2**32, 2**8, 1]))
        hdr(f"t[{nm}].dims-size-overflow", tensors=set_t(i, dims=[256, 2**56]))
        hdr(f"t[{nm}].dims-swapped", tensors=set_t(i, dims=dims[::-1]))
        hdr(f"t[{nm}].dims+[2]", tensors=set_t(i, dims=dims + [2]))
        hdr(f"t[{nm}].dims+[0]", tensors=set_t(i, dims=dims + [0]))
        hdr(f"t[{nm}].dim0-1", tensors=set_t(i, dims=[dims[0] - 1] + dims[1:]))
        hdr(f"t[{nm}].dim-last-1", tensors=set_t(i, dims=dims[:-1] + [dims[-1] - 1]))
        for tn, tid in [("F32", 0), ("F16", 1), ("Q8_1", 9), ("Q2_K", 10), ("BF16", 30), ("99", 99),
                        ("u32max", 2**32 - 1), ("Q4_0", 2), ("Q6_K", 14), ("Q8_K", 15)]:
            if tid != ty:
                hdr(f"t[{nm}].type={tn}", tensors=set_t(i, type=tid))
        hdr(f"t[{nm}].name=bad-utf8", tensors=set_t(i, name=b"\xff" + name))
        hdr(f"t[{nm}].name=empty", tensors=set_t(i, name=b""))
        hdr(f"t[{nm}].dup-name", tensors=set_t((i + 1) % len(T), name=name))
    hdr("t[q0].name=blk.huge", tensors=set_t(q0, name=b"blk.99999999999.attn_q.weight"))
    hdr("t[q0].name=blk.-1", tensors=set_t(q0, name=b"blk.-1.attn_q.weight"))
    hdr("t[q0].name=blk.", tensors=set_t(q0, name=b"blk."))
    hdr("t[q0].name=blk", tensors=set_t(q0, name=b"blk"))
    hdr("t.all-offset-0", tensors=[[n, d, ty, 0] for n, d, ty, _ in T])
    hdr("t.extra-unknown-tensor", tensors=T + [[b"fuzz.extra", [16], 99, 0]])
    hdr("t.extra-huge-unknown", tensors=T + [[b"fuzz.extra", [2**62], 99, 0]])
    hdr("t.del-embd", tensors=[t for t in T if t[0] != b"token_embd.weight"])
    hdr("t.del-output", tensors=[t for t in T if t[0] != b"output.weight"])
    hdr("t.del-output_norm", tensors=[t for t in T if t[0] != b"output_norm.weight"])
    hdr("t.del-last-layer", tensors=[t for t in T if not t[0].startswith(
        f"blk.{g.get_u32(f'{a}.block_count'.encode()) - 1}.".encode())])
    hdr("t.name-len-past-eof", tensors=[[b"x" * 10, [1], 0, 0]] + T, n_t=len(T) + 1)
    # A whole tensor table with the data region missing (header only).
    out.append(("header-only", lambda pth: base.write(pth, padded(V, K, T, A)[0], data_len=0)))
    # Data region shortened by one block (last tensor runs past EOF).
    out.append(("data-short-1block", lambda pth: base.write(pth, padded(V, K, T, A)[0],
                                                            data_len=(base.data_len - 1) // BLOCK * BLOCK)))
    if a == "llama":
        hdr("rope_freqs=del", tensors=[t for t in T if t[0] != b"rope_freqs.weight"])
        rf = idx.get(b"rope_freqs.weight")
        if rf is not None:
            hdr("rope_freqs=dims-1", tensors=set_t(rf, dims=[T[rf][1][0] - 1]))
            hdr("rope_freqs=type-F16", tensors=set_t(rf, type=1))
    return out


QSIZE = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2)}


def tensor_bytes(t):
    n = 1
    for d in t[1]:
        n *= d
    bs, bb = QSIZE.get(t[2], (1, 0))
    return n // bs * bb


def limit(mem_mb):
    def f():
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.nice(10)
    return f


def run1(cmd, mem_mb, timeout, env):
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, preexec_fn=limit(mem_mb), env=env)
    except subprocess.TimeoutExpired:
        return "HANG", f"> {timeout} s"
    err = p.stderr.decode(errors="replace")
    lines = [ln for ln in err.splitlines() if ln.strip()]
    msg = next((ln for ln in lines if "panicked" in ln or "error" in ln.lower() or "axon" in ln), lines[-1] if lines else "")
    if "panicked" in err:
        i = err.find("panicked")
        return "PANIC", err[i:i + 300].replace("\n", " | ")
    if "memory allocation" in err or "alloc" in err and p.returncode < 0:
        return "ALLOC", msg[:300]
    if p.returncode < 0:
        return "SIGNAL", f"signal {-p.returncode}: {msg[:200]}"
    if p.returncode == 0:
        return "ok", ""
    if not lines:
        return "SILENT", f"exit {p.returncode} with no message"
    return "refused", msg[:200]


BAD = {"PANIC", "ALLOC", "SIGNAL", "HANG", "SILENT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("axon")
    ap.add_argument("model")
    ap.add_argument("work")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--mem-mb", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=10)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    base = Base(a.model, a.work)
    ms = mutants(base)
    if a.only:
        ms = [m for m in ms if a.only in m[0]]
    if a.list:
        print("\n".join(m[0] for m in ms))
        return
    env = dict(os.environ, AXON_THREADS="2", AXON_SKIP_GPU="1", NO_COLOR="1")
    env.setdefault("AXON_DEVICE", "cpu")
    lock = threading.Lock()
    findings = []
    counts = {}

    def go(i_m):
        i, (name, w) = i_m
        safe = "".join(c if c.isalnum() or c in "._-=@+" else "_" for c in name)
        p = os.path.join(a.work, f"m{i:04d}-{safe}.gguf")
        try:
            w(p)
        except Exception as e:  # a harness bug, not an axon finding
            with lock:
                print(f"HARNESS {name}: {e!r}", flush=True)
                counts["harness-error"] = counts.get("harness-error", 0) + 1
            return
        res = []
        for label, cmd in [("info", [a.axon, "info", p]), ("gen", [a.axon, "gen", p, "Hello", "-n", "1"])]:
            st, msg = run1(cmd, a.mem_mb, a.timeout, env)
            res.append((label, st, msg))
        if not a.keep:
            os.unlink(p)
        with lock:
            for label, st, msg in res:
                counts[st] = counts.get(st, 0) + 1
                tag = "FINDING" if st in BAD else "       "
                print(f"{tag} {name:40s} {label:4s} {st:8s} {msg}", flush=True)
                if st in BAD:
                    findings.append((name, label, st, msg))

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, enumerate(ms)))
    print(f"\n{len(ms)} mutants, {sum(counts.values())} runs: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for f in findings:
        print("FINDING", *f)
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
