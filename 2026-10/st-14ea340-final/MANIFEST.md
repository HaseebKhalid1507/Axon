# Gate set `st-14ea340-final`: FAIL (1 of 10)

Published 2026-10-19 12:25:26 EDT by axon-gatelogs. Tested sha(s): `14ea340`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1019-040921-9ab5 | st-gates-14ea340 | PASS | bella | CPU only (bella; GPU hidden) | 615.71.09 | 14ea340 | gates.sh | 10-19 04:31–05:16 | [log](1019-040921-9ab5.log) |
| 1019-040921-690d | st-gatesB-llama-14ea340 | PASS | bella | CPU only (bella; GPU hidden) | 615.71.09 | 14ea340 | gatesB.sh | 10-19 04:51–05:58 | [log](1019-040921-690d.log) |
| 1019-040921-5f19 | st-fuzz-14ea340 | PASS | bella | CPU only (bella; GPU hidden) | 615.71.09 | 14ea340 | fuzz.sh | 10-19 04:39–04:42 | [log](1019-040921-5f19.log) |
| 1019-040922-90d2 | st-md5gpu-14ea340 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 14ea340 | md5-gen-gpu.sh | 10-19 10:20–10:21 | [log](1019-040922-90d2.log) |
| 1019-040926-9562 | st-14ea340-full-rel [shard 1/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 14ea340 | axon-test-14ea340 | 10-19 10:21–10:29 | [log](1019-040926-9562.log) |
| 1019-040926-bfcc | st-14ea340-full-rel [shard 2/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 14ea340 | axon-test-14ea340 | 10-19 10:38–10:53 | [log](1019-040926-bfcc.log) |
| 1019-040926-2620 | st-14ea340-full-rel [shard 3/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 14ea340 | axon-test-14ea340 | 10-19 11:21–11:52 | [log](1019-040926-2620.log) |
| 1019-040926-6f97 | st-14ea340-full-dbg [shard 1/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 14ea340 | axon-test-dbg-14ea340 | 10-19 10:29–10:38 | [log](1019-040926-6f97.log) |
| 1019-040926-5dcd | st-14ea340-full-dbg [shard 2/3] | FAIL (rc 125) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 14ea340 | axon-test-dbg-14ea340 | 10-19 10:43–11:18 | [log](1019-040926-5dcd.log) |
| 1019-040926-7a09 | st-14ea340-full-dbg [shard 3/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 14ea340 | axon-test-dbg-14ea340 | 10-19 11:53–12:25 | [log](1019-040926-7a09.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [avante](hw-avante.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
