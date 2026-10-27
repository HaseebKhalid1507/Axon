# Gate set `sharekv-2d78d7a-c2`: FAIL (3 of 4)

Published 2026-10-27 09:13:16 EDT by axon-gatelogs. Tested sha(s): `2d78d7a`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1027-034114-7ee2 | sharekv-2d78d7a-md5 | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 2d78d7a | bash | 10-27 05:23–05:26 | [log](1027-034114-7ee2.log) |
| 1027-034120-6b41 | sharekv-2d78d7a-full-rel | FAIL (rc 101) | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 2d78d7a | axon-test-2d78d7a | 10-27 03:59–04:58 | [log](1027-034120-6b41.log) |
| 1027-034120-5483 | sharekv-2d78d7a-full-dbg | FAIL (rc 101) | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | 2d78d7a | axon-test-dbg-2d78d7a | 10-27 03:59–05:01 | [log](1027-034120-5483.log) |
| 1027-034126-e21e | sharekv-2d78d7a-scalar | FAIL (rc 130) | avante | CPU only (avante; GPU hidden) | 610.43.02 | 2d78d7a | axon-test-2d78d7a | 10-27 03:41–09:13 | [log](1027-034126-e21e.log) |

Redactions: none.

Hardware records: [pod1-10-27_05:23](hw-pod1-10-27_05:23.txt), [pod2-10-27_03:59](hw-pod2-10-27_03:59.txt), [avante](hw-avante.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
