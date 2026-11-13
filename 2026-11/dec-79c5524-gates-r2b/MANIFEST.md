# Gate set `dec-79c5524-gates-r2b`: FAIL (1 of 5)

Published 2026-11-13 01:32:58 EST by axon-gatelogs. Tested sha(s): `79c5524`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1113-002157-f044 | dec-79c5524-fast-rel-r2 [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 79c5524 | axon-test-79c5524 | 11-13 00:30–00:43 | [log](1113-002157-f044.log) |
| 1113-002157-3cd0 | dec-79c5524-fast-rel-r2 [shard 2/3] | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 79c5524 | axon-test-79c5524 | 11-13 00:30–01:32 | [log](1113-002157-3cd0.log) |
| 1113-002157-adb4 | dec-79c5524-fast-rel-r2 [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 79c5524 | axon-test-79c5524 | 11-13 00:32–00:43 | [log](1113-002157-adb4.log) |
| 1112-232744-e399 | dec-79c5524-fast-dbg-cpu-r2 | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | 79c5524 | axon-test-dbg-79c5524 | 11-13 00:16–00:37 | [log](1112-232744-e399.log) |
| 1112-230634-2628 | dec-79c5524-md5 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 79c5524 | bash | 11-12 23:24–23:28 | [log](1112-230634-2628.log) |

Redactions: none.

Hardware records: [pod1-11-13_00:30](hw-pod1-11-13_00:30.txt), [pod1-11-13_00:32](hw-pod1-11-13_00:32.txt), [bella](hw-bella.txt), [pod1-11-12_23:24](hw-pod1-11-12_23:24.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
