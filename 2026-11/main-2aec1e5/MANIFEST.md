# Gate set `main-2aec1e5`: FAIL (5 of 7)

Published 2026-11-06 17:11:45 EST by axon-gatelogs. Tested sha(s): `2aec1e5`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1106-134652-00d0 | main-2aec1e5-rel [shard 1/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 2aec1e5 | axon-test-2aec1e5 | 11-06 13:59–14:15 | [log](1106-134652-00d0.log) |
| 1106-134652-d485 | main-2aec1e5-rel [shard 3/3] | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 2aec1e5 | axon-test-2aec1e5 | ?–17:11 | [log](1106-134652-d485.log) |
| 1106-134652-e63d | main-2aec1e5-rel [shard 2/3] | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 2aec1e5 | axon-test-2aec1e5 | 11-06 17:04–17:10 | [log](1106-134652-e63d.log) |
| 1106-134652-1ea1 | main-2aec1e5-dbg [shard 2/3] | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 2aec1e5 | axon-test-dbg-2aec1e5 | 11-06 17:08–17:10 | [log](1106-134652-1ea1.log) |
| 1106-134652-2a60 | main-2aec1e5-dbg [shard 3/3] | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 2aec1e5 | axon-test-dbg-2aec1e5 | ?–17:11 | [log](1106-134652-2a60.log) |
| 1106-134652-d830 | main-2aec1e5-dbg [shard 1/3] | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 2aec1e5 | axon-test-dbg-2aec1e5 | ?–17:11 | [log](1106-134652-d830.log) |
| 1106-134652-7553 | main-2aec1e5-md5 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 2aec1e5 | bash | 11-06 13:49–13:51 | [log](1106-134652-7553.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [pod1-11-06_17:04](hw-pod1-11-06_17:04.txt), [pod1-11-06_17:08](hw-pod1-11-06_17:08.txt), [pod1-11-06_13:49](hw-pod1-11-06_13:49.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
