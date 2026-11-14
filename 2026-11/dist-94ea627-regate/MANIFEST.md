# Gate set `dist-94ea627-regate`: FAIL (1 of 7)

Published 2026-11-13 22:34:21 EST by axon-gatelogs. Tested sha(s): `94ea627`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1113-154520-e23b | dist-94ea627-full-rel [shard 1/3] | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 94ea627 | axon-test-pull-94ea627 | 11-13 15:54–16:08 | [log](1113-154520-e23b.log) |
| 1113-154520-26bb | dist-94ea627-full-rel [shard 2/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 94ea627 | axon-test-pull-94ea627 | 11-13 16:06–16:26 | [log](1113-154520-26bb.log) |
| 1113-154520-a04b | dist-94ea627-full-rel [shard 3/3] | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 94ea627 | axon-test-pull-94ea627 | 11-13 16:08–16:18 | [log](1113-154520-a04b.log) |
| 1113-154520-95a0 | dist-94ea627-fp8graded | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 94ea627 | axon-test-pull-94ea627 | 11-13 17:45–18:07 | [log](1113-154520-95a0.log) |
| 1113-154521-c866 | dist-94ea627-full-dbg-cpu | PASS | pod1 | CPU only (pod1; GPU hidden) | 595.91.07 | 94ea627 | axon-test-dbg-pull-94ea627 | 11-13 15:45–16:17 | [log](1113-154521-c866.log) |
| 1113-154521-df94 | dist-94ea627-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 94ea627 | bash | 11-13 22:29–22:34 | [log](1113-154521-df94.log) |
| 1113-154522-e27e | dist-94ea627-aliases | FAIL (rc 1) | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 94ea627 | bash | 11-13 16:16–17:27 | [log](1113-154522-e27e.log) |

Redactions: none.

Hardware records: [pod2-11-13_15:54](hw-pod2-11-13_15:54.txt), [pod1-11-13_16:06](hw-pod1-11-13_16:06.txt), [pod2-11-13_16:08](hw-pod2-11-13_16:08.txt), [bella](hw-bella.txt), [pod1-11-13_15:45](hw-pod1-11-13_15:45.txt), [pod2-11-13_16:16](hw-pod2-11-13_16:16.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
