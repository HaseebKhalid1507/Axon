# Gate set `ramdense-89b9ae4`: FAIL (6 of 7)

Published 2026-10-25 23:43:24 EDT by axon-gatelogs. Tested sha(s): `89b9ae4`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-234233-9ea6 | ramdense-89b9ae4-focused-rel- | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 89b9ae4 | axon-test-89b9ae4 | 10-25 23:42–23:42 | [log](1025-234233-9ea6.log) |
| 1025-234236-c996 | ramdense-89b9ae4-focused-dbg- | FAIL (rc 101) | avante | CPU only (avante; GPU hidden) | 610.43.02 | 89b9ae4 | axon-test-dbg-89b9ae4 | 10-25 23:42–23:42 | [log](1025-234236-c996.log) |
| 1025-234236-c2e3 | ramdense-89b9ae4-mincap-mmap0 | FAIL (rc 130) | bella | CPU only (bella; GPU hidden) | 615.78.08 | 89b9ae4 | mincap.sh | ?–23:43 | [log](1025-234236-c2e3.log) |
| 1025-234237-c5e5 | ramdense-89b9ae4-mincap | FAIL (rc 130) | bella | CPU only (bella; GPU hidden) | 615.78.08 | 89b9ae4 | mincap.sh | ?–23:43 | [log](1025-234237-c5e5.log) |
| 1025-234237-753c | ramdense-89b9ae4-md5 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 89b9ae4 | md5.sh | ?–23:43 | [log](1025-234237-753c.log) |
| 1025-234238-f15e | ramdense-89b9ae4-fast-rel | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 89b9ae4 | axon-test-89b9ae4 | ?–23:43 | [log](1025-234238-f15e.log) |
| 1025-234238-dfb2 | ramdense-89b9ae4-fast-dbg-cpu | FAIL (rc 130) | bella | CPU only (bella; GPU hidden) | 615.78.08 | 89b9ae4 | axon-test-dbg-89b9ae4 | ?–23:43 | [log](1025-234238-dfb2.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
