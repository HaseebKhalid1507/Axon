# Gate set `mla-2be8bfe`: FAIL (1 of 5)

Published 2026-10-26 05:24:05 EDT by axon-gatelogs. Tested sha(s): `2be8bfe`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1026-042321-dd18 | mla-2be8bfe-bits-avante | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 2be8bfe | axon-test-2be8bfe | 10-26 04:25–04:26 | [log](1026-042321-dd18.log) |
| 1026-042322-ef23 | mla-2be8bfe-md5 | FAIL (rc 1) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 2be8bfe | md5both.sh | 10-26 04:39–04:43 | [log](1026-042322-ef23.log) |
| 1026-042322-739c | mla-2be8bfe-ds2 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 2be8bfe | axon-test-2be8bfe | 10-26 04:43–04:45 | [log](1026-042322-739c.log) |
| 1026-042323-541b | mla-2be8bfe-fast-rel | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 2be8bfe | axon-test-2be8bfe | 10-26 04:42–05:24 | [log](1026-042323-541b.log) |
| 1026-042323-6fb7 | mla-2be8bfe-fast-dbg-cpu | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | 2be8bfe | axon-test-dbg-2be8bfe | 10-26 04:25–04:42 | [log](1026-042323-6fb7.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
