# Gate set `dist-bb87723-fast-r3`: FAIL (1 of 4)

Published 2026-11-13 15:47:12 EST by axon-gatelogs. Tested sha(s): `bb87723`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1113-112439-b1c0 | dist-bb87723-fast-rel-r3 [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | bb87723 | axon-test-bb87723 | 11-13 12:12–12:24 | [log](1113-112439-b1c0.log) |
| 1113-112439-5b34 | dist-bb87723-fast-rel-r3 [shard 2/3] | PASS | pod2 | NVIDIA L40S | 550.144.03 | bb87723 | axon-test-bb87723 | 11-13 12:18–12:38 | [log](1113-112439-5b34.log) |
| 1113-112439-5a75 | dist-bb87723-fast-rel-r3 [shard 3/3] | PASS | pod2 | NVIDIA L40S | 550.144.03 | bb87723 | axon-test-bb87723 | 11-13 12:17–12:34 | [log](1113-112439-5a75.log) |
| 1113-112439-dda8 | dist-bb87723-fp8graded | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | bb87723 | axon-test-bb87723 | ?–15:47 | [log](1113-112439-dda8.log) |

Redactions: none.

Hardware records: [pod1-11-13_12:12](hw-pod1-11-13_12:12.txt), [pod2-11-13_12:18](hw-pod2-11-13_12:18.txt), [pod2-11-13_12:17](hw-pod2-11-13_12:17.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
