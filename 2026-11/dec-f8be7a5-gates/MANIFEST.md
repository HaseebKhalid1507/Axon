# Gate set `dec-f8be7a5-gates`: FAIL (2 of 13)

Published 2026-11-14 19:15:46 EST by axon-gatelogs. Tested sha(s): `f8be7a5`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1114-182950-b868 | dec-f8be7a5-fast-rel [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | axon-test-f8be7a5 | 11-14 18:37–18:43 | [log](1114-182950-b868.log) |
| 1114-182950-d18d | dec-f8be7a5-fast-rel [shard 2/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | axon-test-f8be7a5 | 11-14 18:57–19:10 | [log](1114-182950-d18d.log) |
| 1114-182950-d151 | dec-f8be7a5-fast-rel [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | axon-test-f8be7a5 | 11-14 18:43–18:51 | [log](1114-182950-d151.log) |
| 1114-182950-0a6e | dec-f8be7a5-fast-dbg-cpu [shard 1/3] | PASS | pod1 | CPU only (pod1; GPU hidden) | 595.91.07 | f8be7a5 | axon-test-dbg-f8be7a5 | 11-14 18:29–18:37 | [log](1114-182950-0a6e.log) |
| 1114-182950-f494 | dec-f8be7a5-fast-dbg-cpu [shard 2/3] | FAIL (rc 130) | bella | CPU only (bella; GPU hidden) | 620.70.06 | f8be7a5 | axon-test-dbg-f8be7a5 | ?–19:15 | [log](1114-182950-f494.log) |
| 1114-182950-7f5e | dec-f8be7a5-fast-dbg-cpu [shard 3/3] | PASS | pod1 | CPU only (pod1; GPU hidden) | 595.91.07 | f8be7a5 | axon-test-dbg-f8be7a5 | 11-14 18:51–18:54 | [log](1114-182950-7f5e.log) |
| 1114-182950-96c3 | dec-f8be7a5-share-dbg | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | axon-test-dbg-f8be7a5 | 11-14 18:43–18:44 | [log](1114-182950-96c3.log) |
| 1114-182950-ab71 | dec-f8be7a5-serve-sharekv0 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | axon-test-f8be7a5 | 11-14 18:54–18:57 | [log](1114-182950-ab71.log) |
| 1114-182951-ffaa | dec-f8be7a5-sjf20 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | bash | 11-14 18:32–18:33 | [log](1114-182951-ffaa.log) |
| 1114-182951-9ffd | dec-f8be7a5-cancel20 | FAIL (rc 1) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | bash | 11-14 18:32–18:33 | [log](1114-182951-9ffd.log) |
| 1114-182951-e03c | dec-f8be7a5-md5 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | bash | 11-14 18:33–18:36 | [log](1114-182951-e03c.log) |
| 1114-182952-f552 | dec-f8be7a5-sharekv-cli | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | bash | 11-14 18:36–18:36 | [log](1114-182952-f552.log) |
| 1114-183143-a35a | dec-f8be7a5-fixproof268 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | f8be7a5 | bash | 11-14 18:36–18:36 | [log](1114-183143-a35a.log) |

Redactions: none.

Hardware records: [pod1-11-14_18:37](hw-pod1-11-14_18:37.txt), [pod1-11-14_18:57](hw-pod1-11-14_18:57.txt), [pod1-11-14_18:43](hw-pod1-11-14_18:43.txt), [pod1-11-14_18:29](hw-pod1-11-14_18:29.txt), [bella](hw-bella.txt), [pod1-11-14_18:51](hw-pod1-11-14_18:51.txt), [pod1-11-14_18:54](hw-pod1-11-14_18:54.txt), [pod1-11-14_18:32](hw-pod1-11-14_18:32.txt), [pod1-11-14_18:33](hw-pod1-11-14_18:33.txt), [pod1-11-14_18:36](hw-pod1-11-14_18:36.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
