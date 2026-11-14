# Gate set `dec-e75ec01-gates`: FAIL (11 of 13)

Published 2026-11-14 17:54:44 EST by axon-gatelogs. Tested sha(s): `e75ec01`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1114-174039-e8dc | dec-e75ec01-fast-rel [shard 1/3] | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e75ec01 | axon-test-e75ec01 | 11-14 17:45–17:54 | [log](1114-174039-e8dc.log) |
| 1114-174039-be99 | dec-e75ec01-fast-rel [shard 2/3] | FAIL (rc 130) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e75ec01 | axon-test-e75ec01 | ?–17:46 | [log](1114-174039-be99.log) |
| 1114-174039-841a | dec-e75ec01-fast-rel [shard 3/3] | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | e75ec01 | axon-test-e75ec01 | ?–17:46 | [log](1114-174039-841a.log) |
| 1114-174039-9da2 | dec-e75ec01-fast-dbg-cpu [shard 1/3] | FAIL (rc 130) | bella | CPU only (bella; GPU hidden) | 620.70.06 | e75ec01 | axon-test-dbg-e75ec01 | ?–17:46 | [log](1114-174039-9da2.log) |
| 1114-174039-1451 | dec-e75ec01-fast-dbg-cpu [shard 2/3] | PASS | pod1 | CPU only (pod1; GPU hidden) | 595.91.07 | e75ec01 | axon-test-dbg-e75ec01 | 11-14 17:43–17:43 | [log](1114-174039-1451.log) |
| 1114-174039-7e1c | dec-e75ec01-fast-dbg-cpu [shard 3/3] | PASS | bella | CPU only (bella; GPU hidden) | 620.70.06 | e75ec01 | axon-test-dbg-e75ec01 | 11-14 17:43–17:50 | [log](1114-174039-7e1c.log) |
| 1114-174039-9861 | dec-e75ec01-share-dbg | FAIL (rc 2) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e75ec01 | axon-test-dbg-e75ec01 | 11-14 17:42–17:45 | [log](1114-174039-9861.log) |
| 1114-174040-8788 | dec-e75ec01-serve-sharekv0 | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e75ec01 | axon-test-e75ec01 | 11-14 17:42–17:46 | [log](1114-174040-8788.log) |
| 1114-174040-ab44 | dec-e75ec01-sjf20 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | e75ec01 | bash | ?–17:46 | [log](1114-174040-ab44.log) |
| 1114-174041-11d0 | dec-e75ec01-cancel20 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | e75ec01 | bash | ?–17:46 | [log](1114-174041-11d0.log) |
| 1114-174041-7a5c | dec-e75ec01-md5 | FAIL (rc 130) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e75ec01 | bash | ?–17:46 | [log](1114-174041-7a5c.log) |
| 1114-174041-c441 | dec-e75ec01-sharekv-cli | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | e75ec01 | bash | ?–17:46 | [log](1114-174041-c441.log) |
| 1114-174305-5d40 | dec-e75ec01-fixproof268 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | e75ec01 | bash | ?–17:46 | [log](1114-174305-5d40.log) |

Redactions: none.

Hardware records: [pod1-11-14_17:45](hw-pod1-11-14_17:45.txt), [pod1-?](hw-pod1-?.txt), [bella](hw-bella.txt), [pod1-11-14_17:43](hw-pod1-11-14_17:43.txt), [pod1-11-14_17:42](hw-pod1-11-14_17:42.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
