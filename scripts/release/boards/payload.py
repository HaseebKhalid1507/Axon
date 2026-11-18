#!/usr/bin/env python3
"""payload.py: a pod job's outputs, back to bella through the job's log (release 0.1's boards, amendment 3, 11-17).

A pod job (axon-podjob) keeps its files on the pod; only its stdout comes back, into axon-run's job log on bella,
copied in 64 KB chunks that axon-run's own lines ("[axon-run ...] FOREIGN_GPU sample ...") can split mid-line. So:

  payload.py emit FILE [LABEL]   (pod.sh, on the pod) prints FILE as numbered base64 lines, each with its CRC-32,
                                 the whole block twice, between a header (sha256, size, line count) and an end line
  payload.py extract LOG OUT     (by hand) writes the file a log carries to OUT; queue.py calls extract() and merge()

extract() keeps a line only if it matches exactly and its CRC holds, takes each line number from whichever copy is
intact, and checks the joined bytes against the header's sha256 and size: a log that lost a line in both copies
gives an error, never a wrong file. merge() unpacks a job's out/ tarball (members under out/ only, no links, no
absolute or .. paths) into a dir: a file that exists with other bytes is a conflict (an error), except the per-run
files a board round rewrites (meta.json, table-*.md, READING.txt: skipped, the job's own copy stays under jobs/) and
engines.jsonl (appended). --selftest: a round trip through a log with split and lost lines."""
import base64, hashlib, io, os, re, shutil, sys, tarfile, tempfile, zlib

CHUNK = 3000   # bytes per line (4000 base64 characters)
LINE = re.compile(r'^@@R (\d+) ([0-9a-f]{8}) ([A-Za-z0-9+/=]+)$')
HEAD = re.compile(r'^@@RESULT BEGIN (\S+) sha256 ([0-9a-f]{64}) bytes (\d+) lines (\d+)$')
SKIP = re.compile(r'^(meta\.json|table-[^/]*\.md|READING\.txt)$')   # rewritten by every board round: not merged
APPEND = {'engines.jsonl'}


def emit(data, label='-', out=sys.stdout):
    """Print data (bytes) as the payload block, twice."""
    parts = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)] or [b'']
    out.write(f'@@RESULT BEGIN {label} sha256 {hashlib.sha256(data).hexdigest()} bytes {len(data)} lines {len(parts)}\n')
    for _ in range(2):
        for i, p in enumerate(parts):
            out.write(f'@@R {i} {zlib.crc32(p):08x} {base64.b64encode(p).decode()}\n')
    out.write('@@RESULT END\n')
    out.flush()


def extract(text):
    """(bytes, label) from a log's text; raises ValueError if there is no intact block."""
    head = [HEAD.match(x) for x in text.splitlines()]
    head = [m for m in head if m]
    if not head:
        raise ValueError('no @@RESULT BEGIN line (the job printed no payload)')
    if len(head) > 1:
        raise ValueError(f'{len(head)} payload headers in one log')
    label, sha, size, n = head[0].group(1), head[0].group(2), int(head[0].group(3)), int(head[0].group(4))
    got = {}
    for x in text.splitlines():
        m = LINE.match(x)
        if not m or int(m.group(1)) in got:
            continue
        try:
            p = base64.b64decode(m.group(3), validate=True)
        except ValueError:
            continue
        if f'{zlib.crc32(p):08x}' == m.group(2):
            got[int(m.group(1))] = p
    miss = [i for i in range(n) if i not in got]
    if miss:
        raise ValueError(f'payload lines lost in both copies: {miss[:5]}{" ..." if len(miss) > 5 else ""} of {n}')
    data = b''.join(got[i] for i in range(n))
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('payload sha256 / size mismatch')
    return data, label


def unpack(data, dest):
    """A .tgz of out/... into dest (dest/out/...): regular files and dirs only, every path under out/."""
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as t:
        ms = t.getmembers()
        for m in ms:
            parts = m.name.split('/')
            if m.name.startswith('/') or '..' in parts or parts[0] != 'out' or not (m.isfile() or m.isdir()):
                raise ValueError(f'payload member refused: {m.name!r}')
        for m in ms:
            p = os.path.join(dest, m.name)
            if m.isdir():
                os.makedirs(p, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with t.extractfile(m) as src, open(p, 'wb') as dst:
                    shutil.copyfileobj(src, dst)


def merge(src, dst):
    """Copy the tree src into dst (see the module doc); returns the files written."""
    done = []
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for f in sorted(files):
            a = os.path.join(root, f)
            b = os.path.normpath(os.path.join(dst, rel, f))
            if os.path.exists(b):
                same = open(a, 'rb').read() == open(b, 'rb').read()
                if same:
                    continue
                if f in APPEND:
                    with open(b, 'ab') as o:
                        o.write(open(a, 'rb').read())
                    done.append(b)
                    continue
                if SKIP.match(f):
                    continue
                raise ValueError(f'merge conflict: {os.path.join(rel, f)} exists with other bytes')
            os.makedirs(os.path.dirname(b), exist_ok=True)
            shutil.copy2(a, b)
            done.append(b)
    return done


def selftest():
    t = tempfile.mkdtemp(prefix='payload-selftest-')
    try:
        os.makedirs(os.path.join(t, 'out', 'board'))
        big = os.urandom(20000)   # incompressible: several lines
        open(os.path.join(t, 'out', 'board', 'run-axon-f16-s1-r0.json'), 'wb').write(big)
        open(os.path.join(t, 'out', 'board', 'meta.json'), 'w').write('{"r": 0}')
        open(os.path.join(t, 'out', 'board', 'engines.jsonl'), 'w').write('{"round": 0}\n')
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tf:
            tf.add(os.path.join(t, 'out'), arcname='out')
        data = buf.getvalue()
        s = io.StringIO()
        emit(data, 'board-r0', s)
        L = s.getvalue().splitlines()
        n = int(HEAD.match(L[0]).group(4))
        assert n >= 3 and len(L) == 2 * n + 2
        # axon-run's own line lands inside payload line 1 of copy 1; line 2 of copy 2 is lost; junk around
        L[1 + 1] = L[1 + 1][:50] + '[axon-run 21:01:06] FOREIGN_GPU sample max=0% mean=0.0% n=1'
        del L[1 + n + 2]
        log = '\n'.join(['[axon-run 20:51:17] submitted ...', 'CELL END ok'] + L + ['[axon-run 21:02:43] exit 0'])
        got, label = extract(log)
        assert got == data and label == 'board-r0'
        # the same line lost in both copies: an error, not a wrong file
        L2 = s.getvalue().splitlines()
        del L2[1 + n + 1]
        del L2[1 + 1]
        try:
            extract('\n'.join(L2))
            raise AssertionError('a lost line passed')
        except ValueError as e:
            assert 'lost in both copies' in str(e)
        # a corrupted line (CRC fails) in one copy: the other copy's line is used
        L3 = s.getvalue().splitlines()
        m = LINE.match(L3[2])
        L3[2] = f'@@R {m.group(1)} {m.group(2)} ' + ('A' * len(m.group(3)))
        assert extract('\n'.join(L3))[0] == data
        try:
            extract('no payload here')
            raise AssertionError('no header passed')
        except ValueError as e:
            assert 'no @@RESULT BEGIN' in str(e)
        # unpack + merge: a second round appends engines.jsonl, skips meta.json, refuses a changed run file
        d1, dst = os.path.join(t, 'j1'), os.path.join(t, 'D')
        unpack(got, d1)
        merge(os.path.join(d1, 'out'), dst)
        open(os.path.join(d1, 'out', 'board', 'engines.jsonl'), 'w').write('{"round": 1}\n')
        open(os.path.join(d1, 'out', 'board', 'meta.json'), 'w').write('{"r": 1}')
        merge(os.path.join(d1, 'out'), dst)
        assert open(os.path.join(dst, 'board', 'engines.jsonl')).read() == '{"round": 0}\n{"round": 1}\n'
        assert open(os.path.join(dst, 'board', 'meta.json')).read() == '{"r": 0}'
        merge(os.path.join(d1, 'out'), dst)   # identical files: nothing to do (engines.jsonl now differs: appended)
        open(os.path.join(d1, 'out', 'board', 'run-axon-f16-s1-r0.json'), 'w').write('other')
        try:
            merge(os.path.join(d1, 'out'), dst)
            raise AssertionError('a conflict passed')
        except ValueError as e:
            assert 'merge conflict: board/run-axon-f16-s1-r0.json' in str(e)
        # unpack refuses anything outside out/, links, ..
        for name, kind in (('../x', tarfile.REGTYPE), ('out/../../x', tarfile.REGTYPE), ('/etc/x', tarfile.REGTYPE),
                           ('other/x', tarfile.REGTYPE), ('out/l', tarfile.SYMTYPE)):
            b = io.BytesIO()
            with tarfile.open(fileobj=b, mode='w:gz') as tf:
                ti = tarfile.TarInfo(name)
                ti.type = kind
                ti.linkname = '/etc/passwd' if kind == tarfile.SYMTYPE else ''
                tf.addfile(ti, io.BytesIO(b'') if kind == tarfile.REGTYPE else None)
            try:
                unpack(b.getvalue(), os.path.join(t, 'bad'))
                raise AssertionError(f'{name} passed')
            except ValueError as e:
                assert 'refused' in str(e)
        assert not os.path.exists(os.path.join(t, 'x'))
    finally:
        shutil.rmtree(t)
    print('SELFTEST payload OK: numbered CRC lines twice; a split or lost line is taken from the other copy; a line '
          'lost in both is an error; sha256 checked; unpack: out/ only, no links or ..; merge: conflicts refused, '
          'engines.jsonl appended, per-round rewrites skipped')


def main():
    a = sys.argv[1:]
    if a[:1] == ['--selftest']:
        selftest()
    elif a[:1] == ['emit'] and len(a) in (2, 3):
        emit(open(a[1], 'rb').read(), a[2] if len(a) == 3 else '-')
    elif a[:1] == ['extract'] and len(a) == 3:
        data, label = extract(open(a[1], errors='replace').read())
        open(a[2], 'wb').write(data)
        print(f'{label}: {len(data)} bytes -> {a[2]}')
    else:
        sys.exit('usage: payload.py emit FILE [LABEL] | extract LOG OUT | --selftest')


if __name__ == '__main__':
    main()
