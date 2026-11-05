# Gate set `main-fc8f275`: FAIL (2 of 7)

Published 2026-11-05 04:05:24 EST by axon-gatelogs. Tested sha(s): `fc8f275`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1105-015958-2366 | main-fc8f275-rel [shard 3/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fc8f275 | axon-test-fc8f275 | 11-05 03:39–04:05 | [log](1105-015958-2366.log) |
| 1105-015958-66fa | main-fc8f275-rel [shard 1/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fc8f275 | axon-test-fc8f275 | 11-05 01:59–02:12 | [log](1105-015958-66fa.log) |
| 1105-015958-d086 | main-fc8f275-rel [shard 2/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fc8f275 | axon-test-fc8f275 | 11-05 02:12–03:07 | [log](1105-015958-d086.log) |
| 1105-015958-18cd | main-fc8f275-dbg [shard 2/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc8f275 | axon-test-dbg-fc8f275 | 11-05 02:25–03:07 | [log](1105-015958-18cd.log) |
| 1105-015958-5f08 | main-fc8f275-dbg [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc8f275 | axon-test-dbg-fc8f275 | 11-05 02:30–03:06 | [log](1105-015958-5f08.log) |
| 1105-015958-6c7e | main-fc8f275-dbg [shard 1/3] | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fc8f275 | axon-test-dbg-fc8f275 | ?–03:51 | [log](1105-015958-6c7e.log) |
| 1105-015959-e127 | main-fc8f275-md5 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fc8f275 | bash | ?–03:51 | [log](1105-015959-e127.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [pod1-11-05_02:25](hw-pod1-11-05_02:25.txt), [pod1-11-05_02:30](hw-pod1-11-05_02:30.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
