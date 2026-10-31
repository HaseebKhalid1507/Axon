# Gate set `a4-cp4`: FAIL (1 of 8)

Published 2026-10-31 11:10:25 EDT by axon-gatelogs. Tested sha(s): `fe20964`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1031-080832-6b69 | a4-gates-fe20964 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fe20964 | gates4.sh | 10-31 08:13–08:17 | [log](1031-080832-6b69.log) |
| 1031-080833-d8c5 | a4-md5-fe20964 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fe20964 | bash | 10-31 08:17–08:18 | [log](1031-080833-d8c5.log) |
| 1031-080835-2ccf | a4-fe20964-rel [shard 1/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fe20964 | axon-test-rel-fe20964 | 10-31 08:18–08:50 | [log](1031-080835-2ccf.log) |
| 1031-080835-746c | a4-fe20964-rel [shard 2/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fe20964 | axon-test-rel-fe20964 | 10-31 08:50–09:13 | [log](1031-080835-746c.log) |
| 1031-080835-f4d5 | a4-fe20964-rel [shard 3/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fe20964 | axon-test-rel-fe20964 | 10-31 09:13–09:32 | [log](1031-080835-f4d5.log) |
| 1031-080835-d64f | a4-fe20964-dbg [shard 1/3] | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fe20964 | axon-test-dbg-fe20964 | 10-31 09:32–10:24 | [log](1031-080835-d64f.log) |
| 1031-080835-02cf | a4-fe20964-dbg [shard 2/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fe20964 | axon-test-dbg-fe20964 | 10-31 10:24–10:48 | [log](1031-080835-02cf.log) |
| 1031-080835-0ed0 | a4-fe20964-dbg [shard 3/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fe20964 | axon-test-dbg-fe20964 | 10-31 10:48–11:10 | [log](1031-080835-0ed0.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
