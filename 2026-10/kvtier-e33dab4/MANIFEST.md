# Gate set `kvtier-e33dab4`: FAIL (4 of 7)

Published 2026-10-27 14:34:31 EDT by axon-gatelogs. Tested sha(s): `e33dab4`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1027-134559-1457 | kvtier-e33dab4-rel [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e33dab4 | axon-test-e33dab4 | 10-27 14:03–14:25 | [log](1027-134559-1457.log) |
| 1027-134559-a101 | kvtier-e33dab4-rel [shard 2/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e33dab4 | axon-test-e33dab4 | 10-27 14:03–14:14 | [log](1027-134559-a101.log) |
| 1027-134559-e76a | kvtier-e33dab4-rel [shard 3/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e33dab4 | axon-test-e33dab4 | 10-27 14:14–14:34 | [log](1027-134559-e76a.log) |
| 1027-134559-55fd | kvtier-e33dab4-dbg [shard 1/3] | FAIL (rc 130) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e33dab4 | axon-test-dbg-e33dab4 | ?–14:31 | [log](1027-134559-55fd.log) |
| 1027-134559-ace0 | kvtier-e33dab4-dbg [shard 2/3] | FAIL (rc 130) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e33dab4 | axon-test-dbg-e33dab4 | ?–14:31 | [log](1027-134559-ace0.log) |
| 1027-134559-8b07 | kvtier-e33dab4-dbg [shard 3/3] | FAIL (rc 130) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | e33dab4 | axon-test-dbg-e33dab4 | ?–14:31 | [log](1027-134559-8b07.log) |
| 1027-134604-d173 | kvtier-e33dab4-md5 | FAIL (rc 130) | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | e33dab4 | bash | 10-27 14:26–14:32 | [log](1027-134604-d173.log) |

Redactions: none.

Hardware records: [pod1-10-27_14:03](hw-pod1-10-27_14:03.txt), [pod1-10-27_14:14](hw-pod1-10-27_14:14.txt), [pod1-?](hw-pod1-?.txt), [pod2-10-27_14:26](hw-pod2-10-27_14:26.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
