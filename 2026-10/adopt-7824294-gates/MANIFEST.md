# Gate set `adopt-7824294-gates`: FAIL (2 of 6)

Published 2026-10-20 12:42:03 EDT by axon-gatelogs. Tested sha(s): `7824294`, `aecc49d`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1020-114634-ee10 | adopt-7824294-fast-rel | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 7824294 | axon-test-7824294 | 10-20 11:46–12:33 | [log](1020-114634-ee10.log) |
| 1020-114634-ab03 | adopt-7824294-fast-dbg-cpu [shard 1/3] | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 7824294 | axon-test-dbg-7824294 | 10-20 11:46–11:56 | [log](1020-114634-ab03.log) |
| 1020-114634-ff4f | adopt-7824294-fast-dbg-cpu [shard 2/3] | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 7824294 | axon-test-dbg-7824294 | 10-20 11:56–12:07 | [log](1020-114634-ff4f.log) |
| 1020-114634-6900 | adopt-7824294-fast-dbg-cpu [shard 3/3] | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 7824294 | axon-test-dbg-7824294 | 10-20 12:33–12:36 | [log](1020-114634-6900.log) |
| 1020-114635-4e40 | adopt-7824294-md5serve | FAIL (rc 1) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | aecc49d | md5serve.sh | 10-20 12:37–12:42 | [log](1020-114635-4e40.log) |
| 1020-114635-aa12 | adopt-7824294-abspeed | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | aecc49d | abspeed.sh | ?–11:52 | [log](1020-114635-aa12.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
