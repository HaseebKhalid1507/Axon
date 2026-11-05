# Gate set `fp8sm89-fc70a43-sm89`: FAIL (2 of 5)

Published 2026-11-04 21:17:57 EST by axon-gatelogs. Tested sha(s): `fc70a43`, `1589821`, `af281d9`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1104-201238-8646 | fp8sm89-fc70a43-fp8rel-sm89 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc70a43 | axon-test-fc70a43 | 11-04 21:13–21:17 | [log](1104-201238-8646.log) |
| 1104-201238-9a9f | fp8sm89-fc70a43-fp8dbg-sm89 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc70a43 | axon-test-dbg-fc70a43 | 11-04 21:16–21:17 | [log](1104-201238-9a9f.log) |
| 1104-200729-4bab | fp8sm89-fc70a43-diag-sm89 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc70a43 | axon-test-fc70a43 | 11-04 20:13–21:17 | [log](1104-200729-4bab.log) |
| 1104-195956-a644 | fp8sm89-1589821-fp8rel-model-sm89 | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 1589821 | axon-test-1589821 | 11-04 20:00–21:16 | [log](1104-195956-a644.log) |
| 1104-195957-7b43 | fp8sm89-main-af281d9-graded-sm89 | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | af281d9 | axon-test-af281d9 | 11-04 20:00–21:13 | [log](1104-195957-7b43.log) |

Redactions: none.

Hardware records: [pod1-11-04_21:13](hw-pod1-11-04_21:13.txt), [pod1-11-04_21:16](hw-pod1-11-04_21:16.txt), [pod1-11-04_20:13](hw-pod1-11-04_20:13.txt), [pod1-11-04_20:00](hw-pod1-11-04_20:00.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
