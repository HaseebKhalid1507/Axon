# Gate set `train-45b1f24`: FAIL (1 of 7)

Published 2026-11-14 13:49:18 EST by axon-gatelogs. Tested sha(s): `45b1f24`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1114-125202-826d | train-45b1f24-rel [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 45b1f24 | axon-test-45b1f24 | 11-14 12:54–13:24 | [log](1114-125202-826d.log) |
| 1114-125202-9ce3 | train-45b1f24-rel [shard 2/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | 45b1f24 | axon-test-45b1f24 | 11-14 12:52–13:07 | [log](1114-125202-9ce3.log) |
| 1114-125202-bc55 | train-45b1f24-rel [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 45b1f24 | axon-test-45b1f24 | 11-14 12:52–13:06 | [log](1114-125202-bc55.log) |
| 1114-125202-8a51 | train-45b1f24-dbg [shard 2/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | 45b1f24 | axon-test-dbg-45b1f24 | 11-14 13:07–13:26 | [log](1114-125202-8a51.log) |
| 1114-125202-b1eb | train-45b1f24-dbg [shard 1/3] | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 45b1f24 | axon-test-dbg-45b1f24 | 11-14 13:07–13:23 | [log](1114-125202-b1eb.log) |
| 1114-125202-cb61 | train-45b1f24-dbg [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 45b1f24 | axon-test-dbg-45b1f24 | 11-14 13:23–13:49 | [log](1114-125202-cb61.log) |
| 1114-125202-66a9 | train-45b1f24-md5 | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 45b1f24 | bash | 11-14 12:55–12:56 | [log](1114-125202-66a9.log) |

Redactions: none.

Hardware records: [pod1-11-14_12:54](hw-pod1-11-14_12:54.txt), [bella](hw-bella.txt), [pod1-11-14_12:52](hw-pod1-11-14_12:52.txt), [pod1-11-14_13:07](hw-pod1-11-14_13:07.txt), [pod1-11-14_13:23](hw-pod1-11-14_13:23.txt), [pod2-11-14_12:55](hw-pod2-11-14_12:55.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
