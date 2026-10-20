# Gate set `adopt-380ba27-gates`: FAIL (2 of 6)

Published 2026-10-20 12:34:38 EDT by axon-gatelogs. Tested sha(s): `380ba27`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1020-112359-2d05 | adopt-380ba27-fast-rel | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 380ba27 | axon-test-380ba27 | 10-20 11:23–12:34 | [log](1020-112359-2d05.log) |
| 1020-112401-0b16 | adopt-380ba27-fast-dbg-cpu [shard 1/3] | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 380ba27 | axon-test-dbg-380ba27 | 10-20 11:24–11:29 | [log](1020-112401-0b16.log) |
| 1020-112401-966c | adopt-380ba27-fast-dbg-cpu [shard 2/3] | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 380ba27 | axon-test-dbg-380ba27 | 10-20 11:29–11:37 | [log](1020-112401-966c.log) |
| 1020-112401-38c9 | adopt-380ba27-fast-dbg-cpu [shard 3/3] | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 380ba27 | axon-test-dbg-380ba27 | 10-20 11:37–11:40 | [log](1020-112401-38c9.log) |
| 1020-112404-5e7d | adopt-380ba27-md5 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 380ba27 | md5gate.sh | ?–11:52 | [log](1020-112404-5e7d.log) |
| 1020-112405-7af4 | adopt-380ba27-serveab | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 380ba27 | serveab-run.sh | ?–11:52 | [log](1020-112405-7af4.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [avante](hw-avante.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
