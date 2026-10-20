# Gate set `gradedref-all`: FAIL (2 of 13)

Published 2026-10-20 02:07:04 EDT by axon-gatelogs. Tested sha(s): `97a99af`, `f8e6b10`, `af34f3e`, `764792f`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1020-010138-c875 | graded-default | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 97a99af | axon-test-97a99af | 10-20 01:01–01:22 | [log](1020-010138-c875.log) |
| 1020-010139-4ab6 | graded-repack0 | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 97a99af | axon-test-97a99af | 10-20 01:22–01:37 | [log](1020-010139-4ab6.log) |
| 1020-014803-a36f | graded-repack0 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | f8e6b10 | axon-test-f8e6b10 | 10-20 01:48–02:01 | [log](1020-014803-a36f.log) |
| 1020-014803-4317 | graded-default-gemma | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | f8e6b10 | axon-test-f8e6b10 | 10-20 02:01–02:06 | [log](1020-014803-4317.log) |
| 1020-010139-6bb9 | axon-wt-gradedref-fast-rel [shard 1/3] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 97a99af | axon-test-97a99af | 10-20 01:01–01:10 | [log](1020-010139-6bb9.log) |
| 1020-010139-c8d0 | axon-wt-gradedref-fast-rel [shard 2/3] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 97a99af | axon-test-97a99af | 10-20 01:10–01:26 | [log](1020-010139-c8d0.log) |
| 1020-010139-5ec4 | axon-wt-gradedref-fast-rel [shard 3/3] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 97a99af | axon-test-97a99af | 10-20 01:26–01:41 | [log](1020-010139-5ec4.log) |
| 1020-010139-67fa | axon-wt-gradedref-fast-dbg-cpu [shard 1/3] | PASS | bella | CPU only (bella; GPU hidden) | 615.71.09 | 97a99af | axon-test-dbg-97a99af | 10-20 01:03–01:09 | [log](1020-010139-67fa.log) |
| 1020-010139-ccf9 | axon-wt-gradedref-fast-dbg-cpu [shard 2/3] | PASS | bella | CPU only (bella; GPU hidden) | 615.71.09 | 97a99af | axon-test-dbg-97a99af | 10-20 01:09–01:12 | [log](1020-010139-ccf9.log) |
| 1020-010139-bc9b | axon-wt-gradedref-fast-dbg-cpu [shard 3/3] | PASS | bella | CPU only (bella; GPU hidden) | 615.71.09 | 97a99af | axon-test-dbg-97a99af | 10-20 01:12–01:19 | [log](1020-010139-bc9b.log) |
| 1020-010631-de37 | ref-spread | PASS | bella | CPU only (bella; GPU hidden) | 615.71.09 | af34f3e | axon-test-af34f3e | 10-20 01:06–01:06 | [log](1020-010631-de37.log) |
| 1020-011056-86b3 | ref-spread-dbg | PASS | bella | CPU only (bella; GPU hidden) | 615.71.09 | af34f3e | axon-test-dbg-af34f3e | 10-20 01:37–01:37 | [log](1020-011056-86b3.log) |
| 1020-013924-e8c8 | gemma3-ref-repack0-main | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 764792f | axon-test-base-764792f | 10-20 01:39–01:39 | [log](1020-013924-e8c8.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [avante](hw-avante.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
