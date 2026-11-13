# Gate set `main-4f347a3-r2`: FAIL (2 of 7)

Published 2026-11-13 13:00:00 EST by axon-gatelogs. Tested sha(s): `4f347a3`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1113-033240-1daa | main-4f347a3-rel [shard 3/3] | FAIL (rc 130) | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 4f347a3 | axon-test-4f347a3 | 11-13 04:27–04:28 | [log](1113-033240-1daa.log) |
| 1113-033240-6625 | main-4f347a3-rel [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 4f347a3 | axon-test-4f347a3 | 11-13 03:55–04:06 | [log](1113-033240-6625.log) |
| 1113-033240-cce5 | main-4f347a3-rel [shard 2/3] | FAIL (rc 101) | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 4f347a3 | axon-test-4f347a3 | 11-13 04:07–05:24 | [log](1113-033240-cce5.log) |
| 1113-033240-4eff | main-4f347a3-dbg [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 4f347a3 | axon-test-dbg-4f347a3 | 11-13 03:55–04:11 | [log](1113-033240-4eff.log) |
| 1113-033240-5f0a | main-4f347a3-dbg [shard 2/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 4f347a3 | axon-test-dbg-4f347a3 | 11-13 04:28–04:55 | [log](1113-033240-5f0a.log) |
| 1113-033240-7ef4 | main-4f347a3-dbg [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 4f347a3 | axon-test-dbg-4f347a3 | 11-13 12:31–12:59 | [log](1113-033240-7ef4.log) |
| 1113-033240-b68c | main-4f347a3-md5 | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 4f347a3 | bash | 11-13 04:29–04:31 | [log](1113-033240-b68c.log) |

Redactions: none.

Hardware records: [pod2-11-13_04:27](hw-pod2-11-13_04:27.txt), [pod1-11-13_03:55](hw-pod1-11-13_03:55.txt), [pod2-11-13_04:07](hw-pod2-11-13_04:07.txt), [pod1-11-13_04:28](hw-pod1-11-13_04:28.txt), [pod1-11-13_12:31](hw-pod1-11-13_12:31.txt), [pod2-11-13_04:29](hw-pod2-11-13_04:29.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
