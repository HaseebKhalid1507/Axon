# Gate set `oss136-main`: FAIL (1 of 5)

Published 2026-10-25 12:55:55 EDT by axon-gatelogs. Tested sha(s): `bdc80f8`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-085841-30f9 | oss136-bdc80f8-dbg [shard 1/2] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | bdc80f8 | axon-test-dbg-bdc80f8 | 10-25 11:31–11:59 | [log](1025-085841-30f9.log) |
| 1025-085841-8feb | oss136-bdc80f8-dbg [shard 2/2] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | bdc80f8 | axon-test-dbg-bdc80f8 | 10-25 11:59–12:27 | [log](1025-085841-8feb.log) |
| 1025-085841-e50c | oss136-bdc80f8-gptoss | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | bdc80f8 | oss136.sh | 10-25 12:27–12:55 | [log](1025-085841-e50c.log) |
| 1025-085841-ee61 | oss136-bdc80f8-rel [shard 2/2] | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | bdc80f8 | axon-test-bdc80f8 | 10-25 11:04–11:31 | [log](1025-085841-ee61.log) |
| 1025-085841-fa9c | oss136-bdc80f8-rel [shard 1/2] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | bdc80f8 | axon-test-bdc80f8 | 10-25 10:39–11:04 | [log](1025-085841-fa9c.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
