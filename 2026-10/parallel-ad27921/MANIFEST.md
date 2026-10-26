# Gate set `parallel-ad27921`: FAIL (1 of 6)

Published 2026-10-25 23:33:06 EDT by axon-gatelogs. Tested sha(s): `ad27921`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-230928-c069 | parallel-ad27921-focused-rel- | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | ad27921 | axon-test-ad27921 | 10-25 23:09–23:09 | [log](1025-230928-c069.log) |
| 1025-230929-a81d | parallel-ad27921-focused-dbg- | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | ad27921 | axon-test-dbg-ad27921 | 10-25 23:09–23:10 | [log](1025-230929-a81d.log) |
| 1025-230929-0a5d | parallel-ad27921-demo | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | ad27921 | pardemo.sh | 10-25 23:19–23:19 | [log](1025-230929-0a5d.log) |
| 1025-230929-1a0e | parallel-ad27921-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | ad27921 | md5.sh | 10-25 23:19–23:20 | [log](1025-230929-1a0e.log) |
| 1025-230930-c777 | parallel-ad27921-fast-dbg-cpu | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | ad27921 | axon-test-dbg-ad27921 | 10-25 23:20–23:30 | [log](1025-230930-c777.log) |
| 1025-230930-bc90 | parallel-ad27921-fast-rel | FAIL (rc 143) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | ad27921 | axon-test-ad27921 | 10-25 23:30–23:33 | [log](1025-230930-bc90.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
