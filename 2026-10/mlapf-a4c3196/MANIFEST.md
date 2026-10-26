# Gate set `mlapf-a4c3196`: FAIL (3 of 6)

Published 2026-10-26 09:00:06 EDT by axon-gatelogs. Tested sha(s): `a4c3196`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1026-065341-4f31 | mlapf-a4c3196-bits-avante | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a4c3196 | axon-test-a4c3196 | 10-26 06:53–06:54 | [log](1026-065341-4f31.log) |
| 1026-065341-f1dc | mlapf-a4c3196-bits-bella | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a4c3196 | axon-test-a4c3196 | 10-26 08:59–09:00 | [log](1026-065341-f1dc.log) |
| 1026-065342-ed97 | mlapf-a4c3196-ds2 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a4c3196 | axon-test-a4c3196 | ?–07:09 | [log](1026-065342-ed97.log) |
| 1026-065344-7650 | mlapf-a4c3196-md5 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a4c3196 | md5all.sh | ?–07:09 | [log](1026-065344-7650.log) |
| 1026-065344-41d6 | mlapf-a4c3196-fast-rel | FAIL (rc 130) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a4c3196 | axon-test-a4c3196 | ?–07:09 | [log](1026-065344-41d6.log) |
| 1026-065344-b3ad | mlapf-a4c3196-fast-dbg-cpu | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | a4c3196 | axon-test-dbg-a4c3196 | 10-26 06:53–07:10 | [log](1026-065344-b3ad.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
