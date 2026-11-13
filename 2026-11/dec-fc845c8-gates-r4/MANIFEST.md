# Gate set `dec-fc845c8-gates-r4`: FAIL (1 of 10)

Published 2026-11-13 13:27:46 EST by axon-gatelogs. Tested sha(s): `fc845c8`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1113-004227-3676 | dec-fc845c8-full-rel [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc845c8 | axon-test-fc845c8 | 11-13 00:43–00:59 | [log](1113-004227-3676.log) |
| 1113-011050-5bb0 | dec-fc845c8-full-rel-shard2-r3 | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc845c8 | axon-test-fc845c8 | 11-13 01:30–01:42 | [log](1113-011050-5bb0.log) |
| 1113-010622-48ac | dec-fc845c8-fp8graded | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fc845c8 | axon-test-fc845c8 | 11-13 13:01–13:27 | [log](1113-010622-48ac.log) |
| 1113-004227-aa66 | dec-fc845c8-full-rel [shard 3/3] | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc845c8 | axon-test-fc845c8 | 11-13 01:02–01:30 | [log](1113-004227-aa66.log) |
| 1113-004227-87e4 | dec-fc845c8-full-dbg [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc845c8 | axon-test-dbg-fc845c8 | 11-13 01:06–01:25 | [log](1113-004227-87e4.log) |
| 1113-004227-db29 | dec-fc845c8-full-dbg [shard 2/3] | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc845c8 | axon-test-dbg-fc845c8 | 11-13 01:06–01:37 | [log](1113-004227-db29.log) |
| 1113-004227-da5e | dec-fc845c8-full-dbg [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc845c8 | axon-test-dbg-fc845c8 | 11-13 01:06–01:27 | [log](1113-004227-da5e.log) |
| 1113-010743-7286 | dec-fc845c8-serve-sharekv0-r2 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | fc845c8 | axon-test-fc845c8 | 11-13 01:27–01:30 | [log](1113-010743-7286.log) |
| 1113-004233-e238 | dec-fc845c8-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fc845c8 | bash | 11-13 00:58–01:47 | [log](1113-004233-e238.log) |
| 1113-004234-cb73 | dec-fc845c8-soaksmoke | FAIL (rc 1) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | fc845c8 | bash | 11-13 10:37–10:37 | [log](1113-004234-cb73.log) |

Redactions: none.

Hardware records: [pod1-11-13_00:43](hw-pod1-11-13_00:43.txt), [pod2-11-13_01:30](hw-pod2-11-13_01:30.txt), [bella](hw-bella.txt), [pod2-11-13_01:02](hw-pod2-11-13_01:02.txt), [pod1-11-13_01:06](hw-pod1-11-13_01:06.txt), [pod2-11-13_01:06](hw-pod2-11-13_01:06.txt), [pod1-11-13_01:27](hw-pod1-11-13_01:27.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
