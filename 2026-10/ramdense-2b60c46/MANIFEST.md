# Gate set `ramdense-2b60c46`: FAIL (2 of 7)

Published 2026-10-25 23:38:43 EDT by axon-gatelogs. Tested sha(s): `2b60c46`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-231803-8875 | ramdense-2b60c46-fast-rel | FAIL (rc 143) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 2b60c46 | axon-test-2b60c46 | 10-25 23:35–23:38 | [log](1025-231803-8875.log) |
| 1025-231803-0d11 | ramdense-2b60c46-fast-dbg-cpu | FAIL (rc 130) | bella | CPU only (bella; GPU hidden) | 615.78.08 | 2b60c46 | axon-test-dbg-2b60c46 | ?–23:38 | [log](1025-231803-0d11.log) |
| 1025-231750-17e3 | ramdense-2b60c46-mincap | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | 2b60c46 | mincap.sh | 10-25 23:33–23:33 | [log](1025-231750-17e3.log) |
| 1025-231750-64ed | ramdense-2b60c46-mincap-mmap0 | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | 2b60c46 | mincap.sh | 10-25 23:33–23:34 | [log](1025-231750-64ed.log) |
| 1025-231751-a93d | ramdense-2b60c46-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 2b60c46 | md5.sh | 10-25 23:34–23:35 | [log](1025-231751-a93d.log) |
| 1025-231749-5ab2 | ramdense-2b60c46-focused-rel- | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 2b60c46 | axon-test-2b60c46 | 10-25 23:17–23:17 | [log](1025-231749-5ab2.log) |
| 1025-231749-e141 | ramdense-2b60c46-focused-dbg- | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 2b60c46 | axon-test-dbg-2b60c46 | 10-25 23:17–23:17 | [log](1025-231749-e141.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [avante](hw-avante.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
