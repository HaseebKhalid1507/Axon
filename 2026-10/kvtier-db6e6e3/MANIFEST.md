# Gate set `kvtier-db6e6e3`: FAIL (3 of 8); TAINTED timing

Published 2026-10-27 10:07:28 EDT by axon-gatelogs. Tested sha(s): `db6e6e3`, `d5a2899`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1027-034832-0dea | kvtier-db6e6e3-rel [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | db6e6e3 | axon-test-db6e6e3 | 10-27 04:25–04:46 | [log](1027-034832-0dea.log) |
| 1027-034832-5d41 | kvtier-db6e6e3-rel [shard 2/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | db6e6e3 | axon-test-db6e6e3 | 10-27 04:25–04:40 | [log](1027-034832-5d41.log) |
| 1027-034832-ed58 | kvtier-db6e6e3-rel [shard 3/3] | FAIL (rc 101) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | db6e6e3 | axon-test-db6e6e3 | 10-27 04:40–05:01 | [log](1027-034832-ed58.log) |
| 1027-034832-fbd4 | kvtier-db6e6e3-dbg [shard 1/3] | PASS | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | db6e6e3 | axon-test-dbg-db6e6e3 | 10-27 04:46–05:13 | [log](1027-034832-fbd4.log) |
| 1027-034832-15b2 | kvtier-db6e6e3-dbg [shard 2/3] | PASS | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | db6e6e3 | axon-test-dbg-db6e6e3 | 10-27 04:58–05:10 | [log](1027-034832-15b2.log) |
| 1027-034832-7cea | kvtier-db6e6e3-dbg [shard 3/3] | FAIL (rc 101) | pod2 | NVIDIA GeForce RTX 4090 | 595.91.07 | db6e6e3 | axon-test-dbg-db6e6e3 | 10-27 05:02–05:23 | [log](1027-034832-7cea.log) |
| 1027-034836-d86a | kvtier-db6e6e3-md5 | FAIL (rc 1) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | db6e6e3 | bash | 10-27 03:56–04:02 | [log](1027-034836-d86a.log) |
| 1027-034307-2c0a | kvtier-d5a2899-timing | PASS TAINTED (FOREIGN_GPU max=96% mean=16.1%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | d5a2899 | timing.sh | 10-27 09:49–10:07 | [log](1027-034307-2c0a.log) |

Redactions: none.

Hardware records: [pod1-10-27_04:25](hw-pod1-10-27_04:25.txt), [pod1-10-27_04:40](hw-pod1-10-27_04:40.txt), [pod1-10-27_04:46](hw-pod1-10-27_04:46.txt), [pod2-10-27_04:58](hw-pod2-10-27_04:58.txt), [pod2-10-27_05:02](hw-pod2-10-27_05:02.txt), [pod1-10-27_03:56](hw-pod1-10-27_03:56.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
