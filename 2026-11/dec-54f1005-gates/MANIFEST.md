# Gate set `dec-54f1005-gates`: FAIL (3 of 6)

Published 2026-11-13 04:14:48 EST by axon-gatelogs. Tested sha(s): `54f1005`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1113-031552-6419 | dec-54f1005-fast-rel [shard 1/3] | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 54f1005 | axon-test-54f1005 | 11-13 03:35–03:53 | [log](1113-031552-6419.log) |
| 1113-031552-bdd5 | dec-54f1005-fast-rel [shard 2/3] | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 54f1005 | axon-test-54f1005 | 11-13 03:41–04:14 | [log](1113-031552-bdd5.log) |
| 1113-031552-79a4 | dec-54f1005-fast-rel [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 54f1005 | axon-test-54f1005 | 11-13 03:46–03:53 | [log](1113-031552-79a4.log) |
| 1113-032052-5331 | dec-54f1005-fast-dbg-cpu | PASS | pod1 | CPU only (pod1; GPU hidden) | 595.91.07 | 54f1005 | axon-test-dbg-54f1005 | 11-13 03:35–03:49 | [log](1113-032052-5331.log) |
| 1113-032053-efeb | dec-54f1005-dbg-fp8-gpu | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 54f1005 | axon-test-dbg-54f1005 | 11-13 03:53–03:55 | [log](1113-032053-efeb.log) |
| 1113-032053-d70b | dec-54f1005-equiv | FAIL (rc 130) | None | ? | ? | 54f1005 | axon-test-54f1005 | ?–03:56 | [log](1113-032053-d70b.log) |

Redactions: none.

Hardware records: [pod1-11-13_03:35](hw-pod1-11-13_03:35.txt), [pod2-11-13_03:41](hw-pod2-11-13_03:41.txt), [pod1-11-13_03:46](hw-pod1-11-13_03:46.txt), [pod1-11-13_03:53](hw-pod1-11-13_03:53.txt), [None](hw-None.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
