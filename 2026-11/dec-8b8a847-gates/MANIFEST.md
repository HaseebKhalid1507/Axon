# Gate set `dec-8b8a847-gates`: FAIL (1 of 13)

Published 2026-11-14 18:37:59 EST by axon-gatelogs. Tested sha(s): `8b8a847`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1114-180550-48dd | dec-8b8a847-fast-rel [shard 1/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | 8b8a847 | axon-test-8b8a847 | 11-14 18:08–18:16 | [log](1114-180550-48dd.log) |
| 1114-180550-553a | dec-8b8a847-fast-rel [shard 2/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 8b8a847 | axon-test-8b8a847 | 11-14 18:13–18:32 | [log](1114-180550-553a.log) |
| 1114-180550-811c | dec-8b8a847-fast-rel [shard 3/3] | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | 8b8a847 | axon-test-8b8a847 | 11-14 18:16–18:37 | [log](1114-180550-811c.log) |
| 1114-180550-7c79 | dec-8b8a847-fast-dbg-cpu [shard 1/3] | PASS | pod1 | CPU only (pod1; GPU hidden) | 595.91.07 | 8b8a847 | axon-test-dbg-8b8a847 | 11-14 18:13–18:20 | [log](1114-180550-7c79.log) |
| 1114-180550-eb05 | dec-8b8a847-fast-dbg-cpu [shard 2/3] | PASS | bella | CPU only (bella; GPU hidden) | 620.70.06 | 8b8a847 | axon-test-dbg-8b8a847 | 11-14 18:05–18:08 | [log](1114-180550-eb05.log) |
| 1114-180550-6d37 | dec-8b8a847-fast-dbg-cpu [shard 3/3] | PASS | pod1 | CPU only (pod1; GPU hidden) | 595.91.07 | 8b8a847 | axon-test-dbg-8b8a847 | 11-14 18:21–18:22 | [log](1114-180550-6d37.log) |
| 1114-180550-aaa2 | dec-8b8a847-share-dbg | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 8b8a847 | axon-test-dbg-8b8a847 | 11-14 18:20–18:23 | [log](1114-180550-aaa2.log) |
| 1114-180550-c0b6 | dec-8b8a847-serve-sharekv0 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 8b8a847 | axon-test-8b8a847 | 11-14 18:22–18:27 | [log](1114-180550-c0b6.log) |
| 1114-180551-fe25 | dec-8b8a847-sjf20 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 8b8a847 | bash | 11-14 18:20–18:21 | [log](1114-180551-fe25.log) |
| 1114-180551-db44 | dec-8b8a847-cancel20 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 8b8a847 | bash | 11-14 18:20–18:21 | [log](1114-180551-db44.log) |
| 1114-180552-88ca | dec-8b8a847-md5 | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 8b8a847 | bash | 11-14 18:20–18:22 | [log](1114-180552-88ca.log) |
| 1114-180552-7ae8 | dec-8b8a847-sharekv-cli | FAIL (rc 1) | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 8b8a847 | bash | 11-14 18:20–18:23 | [log](1114-180552-7ae8.log) |
| 1114-180808-f0b1 | dec-8b8a847-fixproof268 | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 8b8a847 | bash | 11-14 18:23–18:23 | [log](1114-180808-f0b1.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [pod1-11-14_18:13](hw-pod1-11-14_18:13.txt), [pod1-11-14_18:21](hw-pod1-11-14_18:21.txt), [pod2-11-14_18:20](hw-pod2-11-14_18:20.txt), [pod1-11-14_18:22](hw-pod1-11-14_18:22.txt), [pod1-11-14_18:20](hw-pod1-11-14_18:20.txt), [pod2-11-14_18:23](hw-pod2-11-14_18:23.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
