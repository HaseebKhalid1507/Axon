# Gate set `safeguards-bc11938-regate`: FAIL (1 of 8)

Published 2026-11-13 16:12:59 EST by axon-gatelogs. Tested sha(s): `bc11938`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1113-143301-ff85 | safeguards-bc11938-full-rel [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | bc11938 | axon-test-bc11938 | 11-13 14:50–15:12 | [log](1113-143301-ff85.log) |
| 1113-143301-1749 | safeguards-bc11938-full-rel [shard 2/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | bc11938 | axon-test-bc11938 | 11-13 14:51–15:04 | [log](1113-143301-1749.log) |
| 1113-143301-1a00 | safeguards-bc11938-full-rel [shard 3/3] | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | bc11938 | axon-test-bc11938 | 11-13 14:51–15:21 | [log](1113-143301-1a00.log) |
| 1113-143301-4f7e | safeguards-bc11938-full-dbg [shard 1/2] | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | bc11938 | axon-test-dbg-bc11938 | 11-13 14:57–15:32 | [log](1113-143301-4f7e.log) |
| 1113-143301-9854 | safeguards-bc11938-full-dbg [shard 2/2] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | bc11938 | axon-test-dbg-bc11938 | 11-13 15:04–15:34 | [log](1113-143301-9854.log) |
| 1113-143302-ad41 | safeguards-bc11938-fp8-graded | FAIL (rc 130) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | bc11938 | axon-test-bc11938 | 11-13 15:12–16:12 | [log](1113-143302-ad41.log) |
| 1113-143304-d5a7 | safeguards-bc11938-md5 | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | bc11938 | bash | 11-13 15:21–15:24 | [log](1113-143304-d5a7.log) |
| 1113-143305-7080 | safeguards-bc11938-golden | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | bc11938 | bash | 11-13 15:32–15:54 | [log](1113-143305-7080.log) |

Redactions: none.

Hardware records: [pod1-11-13_14:50](hw-pod1-11-13_14:50.txt), [pod1-11-13_14:51](hw-pod1-11-13_14:51.txt), [pod2-11-13_14:51](hw-pod2-11-13_14:51.txt), [pod2-11-13_14:57](hw-pod2-11-13_14:57.txt), [pod1-11-13_15:04](hw-pod1-11-13_15:04.txt), [pod1-11-13_15:12](hw-pod1-11-13_15:12.txt), [pod2-11-13_15:21](hw-pod2-11-13_15:21.txt), [pod2-11-13_15:32](hw-pod2-11-13_15:32.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
