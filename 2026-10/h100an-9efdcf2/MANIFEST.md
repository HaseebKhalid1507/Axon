# Gate set `h100an-9efdcf2`: FAIL (1 of 5); TAINTED timing

Published 2026-10-25 00:37:56 EDT by axon-gatelogs. Tested sha(s): `9efdcf2`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1024-172141-75ad | h100an-9efdcf2-full-rel | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 9efdcf2 | axon-test-9efdcf2 | 10-24 19:08–20:11 | [log](1024-172141-75ad.log) |
| 1024-172141-f4f5 | h100an-9efdcf2-full-dbg | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 9efdcf2 | axon-test-dbg-9efdcf2 | 10-24 17:43–19:08 | [log](1024-172141-f4f5.log) |
| 1024-172219-04d4 | h100an-9efdcf2-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 9efdcf2 | md5all.sh | 10-25 00:03–00:05 | [log](1024-172219-04d4.log) |
| 1024-172306-3fcc | h100an-9efdcf2-avante-pieces | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 9efdcf2 | axon-test-9efdcf2 | 10-24 20:12–20:13 | [log](1024-172306-3fcc.log) |
| 1024-172540-a0d8 | an-rehearsal-9efdcf2-bella | FAIL (rc 1) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 9efdcf2 | rehearse.sh | 10-25 00:32–00:37 | [log](1024-172540-a0d8.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
