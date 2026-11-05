# Gate set `fp8sm89-evidence`: FAIL (3 of 12)

Published 2026-11-05 00:47:43 EST by axon-gatelogs. Tested sha(s): `af281d9`, `104294e`, `1589821`, `330146b`, `fc70a43`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1104-193717-bce3 | sm89-fp8anchor-axon-test-af281d9 | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | af281d9 | axon-test-af281d9 | 11-04 19:37–19:37 | [log](1104-193717-bce3.log) |
| 1104-193717-89a8 | sm89-fp8anchor-axon-test-dbg-af281d9 | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | af281d9 | axon-test-dbg-af281d9 | 11-04 19:37–19:38 | [log](1104-193717-89a8.log) |
| 1104-191235-2023 | main-af281d9-dbg-shard1-r2 | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | af281d9 | axon-test-dbg-af281d9 | 11-04 19:12–19:36 | [log](1104-191235-2023.log) |
| 1104-194742-ccb2 | fp8sm89-104294e-diag-sm120 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 104294e | axon-test-104294e | 11-04 19:47–19:48 | [log](1104-194742-ccb2.log) |
| 1104-195209-4f1e | fp8sm89-104294e-diag-sm89 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 104294e | axon-test-104294e | 11-04 19:52–19:53 | [log](1104-195209-4f1e.log) |
| 1104-195657-375e | fp8sm89-1589821-speed-sm89 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 1589821 | axon-test-1589821 | 11-04 19:58–20:00 | [log](1104-195657-375e.log) |
| 1104-195657-39d9 | fp8sm89-1589821-diag-sm89 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 1589821 | axon-test-1589821 | 11-04 19:57–19:58 | [log](1104-195657-39d9.log) |
| 1104-195656-82aa | fp8sm89-1589821-fp8rel-sm89 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 1589821 | axon-test-1589821 | 11-04 19:56–19:58 | [log](1104-195656-82aa.log) |
| 1104-195831-f361 | fp8sm89-1589821-fp8dbg-sm89 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 1589821 | axon-test-dbg-1589821 | 11-04 19:58–19:58 | [log](1104-195831-f361.log) |
| 1104-200031-3aba | fp8sm89-1589821-md5-sm120 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 330146b | bash | 11-04 20:00–20:02 | [log](1104-200031-3aba.log) |
| 1104-200803-4d05 | fp8sm89-fc70a43-md5-sm120 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fc70a43 | bash | 11-04 20:08–20:10 | [log](1104-200803-4d05.log) |
| 1104-212053-2aeb | fp8sm89-fc70a43-logx-sm89 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc70a43 | bash | 11-04 21:20–21:24 | [log](1104-212053-2aeb.log) |

Redactions: none.

Hardware records: [pod1-11-04_19:37](hw-pod1-11-04_19:37.txt), [pod1-11-04_19:12](hw-pod1-11-04_19:12.txt), [bella](hw-bella.txt), [pod1-11-04_19:52](hw-pod1-11-04_19:52.txt), [pod1-11-04_19:58](hw-pod1-11-04_19:58.txt), [pod1-11-04_19:57](hw-pod1-11-04_19:57.txt), [pod1-11-04_19:56](hw-pod1-11-04_19:56.txt), [pod1-11-04_21:20](hw-pod1-11-04_21:20.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
