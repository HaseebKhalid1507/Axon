# Gate set `m6-3cb97e4`: FAIL (2 of 7)

Published 2026-10-26 20:15:19 EDT by axon-gatelogs. Tested sha(s): `3cb97e4`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1026-180135-ca25 | m6-3cb97e4-cuda-jade | FAIL (rc 130) | jade | NVIDIA GeForce GTX 1050 | 580.159.04 | 3cb97e4 | axon-test-3cb97e4 | 10-26 18:01–20:15 | [log](1026-180135-ca25.log) |
| 1026-180136-2739 | m6-3cb97e4-kchunk512 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 3cb97e4 | axon-test-3cb97e4 | 10-26 18:01–18:03 | [log](1026-180136-2739.log) |
| 1026-180136-babf | m6-3cb97e4-kchunk2048 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 3cb97e4 | axon-test-3cb97e4 | 10-26 18:03–18:04 | [log](1026-180136-babf.log) |
| 1026-180137-68f2 | m6-3cb97e4-fast-rel [shard 1/2] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 3cb97e4 | axon-test-3cb97e4 | 10-26 18:04–18:36 | [log](1026-180137-68f2.log) |
| 1026-180137-32e1 | m6-3cb97e4-fast-rel [shard 2/2] | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 3cb97e4 | axon-test-3cb97e4 | 10-26 18:15–18:31 | [log](1026-180137-32e1.log) |
| 1026-180137-2fd7 | m6-3cb97e4-fast-dbg-cpu [shard 1/2] | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | 3cb97e4 | axon-test-dbg-3cb97e4 | 10-26 18:05–18:15 | [log](1026-180137-2fd7.log) |
| 1026-180137-9df7 | m6-3cb97e4-fast-dbg-cpu [shard 2/2] | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | 3cb97e4 | axon-test-dbg-3cb97e4 | 10-26 18:31–18:33 | [log](1026-180137-9df7.log) |

Redactions: none.

Hardware records: [jade](hw-jade.txt), [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
