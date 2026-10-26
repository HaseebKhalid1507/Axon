# Gate set `ramdense-4a013d7`: FAIL (2 of 7)

Published 2026-10-26 00:43:02 EDT by axon-gatelogs. Tested sha(s): `4a013d7`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-234656-84f9 | ramdense-4a013d7-focused-rel- | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 4a013d7 | axon-test-4a013d7 | 10-25 23:46–23:47 | [log](1025-234656-84f9.log) |
| 1025-234659-c2e7 | ramdense-4a013d7-focused-dbg- | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 4a013d7 | axon-test-dbg-4a013d7 | 10-25 23:46–23:47 | [log](1025-234659-c2e7.log) |
| 1025-234700-f438 | ramdense-4a013d7-mincap-mmap0 | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | 4a013d7 | mincap.sh | 10-26 00:26–00:27 | [log](1025-234700-f438.log) |
| 1025-234700-99f4 | ramdense-4a013d7-mincap | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | 4a013d7 | mincap.sh | 10-26 00:26–00:26 | [log](1025-234700-99f4.log) |
| 1025-234701-58b3 | ramdense-4a013d7-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 4a013d7 | md5.sh | 10-26 00:27–00:28 | [log](1025-234701-58b3.log) |
| 1025-234701-2ce8 | ramdense-4a013d7-fast-rel | FAIL (rc 143) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 4a013d7 | axon-test-4a013d7 | 10-26 00:28–00:43 | [log](1025-234701-2ce8.log) |
| 1025-234701-a30d | ramdense-4a013d7-fast-dbg-cpu | FAIL (rc 130) | bella | CPU only (bella; GPU hidden) | 615.78.08 | 4a013d7 | axon-test-dbg-4a013d7 | ?–00:43 | [log](1025-234701-a30d.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
