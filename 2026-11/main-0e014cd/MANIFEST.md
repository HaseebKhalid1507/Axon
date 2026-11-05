# Gate set `main-0e014cd`: FAIL (2 of 7)

Published 2026-11-05 00:41:43 EST by axon-gatelogs. Tested sha(s): `0e014cd`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1104-212923-02da | main-0e014cd-rel [shard 1/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 0e014cd | axon-test-0e014cd | 11-04 21:31–21:43 | [log](1104-212923-02da.log) |
| 1104-212923-2d79 | main-0e014cd-rel [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 0e014cd | axon-test-0e014cd | 11-04 23:45–00:01 | [log](1104-212923-2d79.log) |
| 1104-212923-c2b0 | main-0e014cd-rel [shard 2/3] | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 0e014cd | axon-test-0e014cd | 11-04 23:26–23:45 | [log](1104-212923-c2b0.log) |
| 1104-212923-1ac0 | main-0e014cd-dbg [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 0e014cd | axon-test-dbg-0e014cd | 11-04 23:06–23:26 | [log](1104-212923-1ac0.log) |
| 1104-212923-7e0c | main-0e014cd-dbg [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 0e014cd | axon-test-dbg-0e014cd | 11-05 00:01–00:19 | [log](1104-212923-7e0c.log) |
| 1104-212923-db5f | main-0e014cd-dbg [shard 2/3] | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 0e014cd | axon-test-dbg-0e014cd | 11-05 00:21–00:41 | [log](1104-212923-db5f.log) |
| 1104-212923-d938 | main-0e014cd-md5 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 0e014cd | bash | 11-05 00:19–00:21 | [log](1104-212923-d938.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [pod1-11-04_23:45](hw-pod1-11-04_23:45.txt), [pod1-11-04_23:26](hw-pod1-11-04_23:26.txt), [pod1-11-04_23:06](hw-pod1-11-04_23:06.txt), [pod1-11-05_00:01](hw-pod1-11-05_00:01.txt), [pod1-11-05_00:21](hw-pod1-11-05_00:21.txt), [pod1-11-05_00:19](hw-pod1-11-05_00:19.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
