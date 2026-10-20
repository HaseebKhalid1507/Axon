# Gate set `moeres-b-59f2e2e`: FAIL (5 of 8)

Published 2026-10-20 17:36:58 EDT by axon-gatelogs. Tested sha(s): `59f2e2e`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1020-173231-0afb | moeresb-59f2e2e-experts-bella | FAIL (rc 130) | None | ? | ? | 59f2e2e | axon-test-59f2e2e | ?–17:36 | [log](1020-173231-0afb.log) |
| 1020-173231-6893 | moeresb-59f2e2e-experts-avante | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 59f2e2e | axon-test-59f2e2e | 10-20 17:32–17:33 | [log](1020-173231-6893.log) |
| 1020-173232-6062 | moeresb-59f2e2e-experts-dbg-avante | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 59f2e2e | axon-test-dbg-59f2e2e | 10-20 17:33–17:33 | [log](1020-173232-6062.log) |
| 1020-173232-2a0d | moeresb-59f2e2e-experts-jade | PASS | jade | NVIDIA GeForce GTX 1050 | 580.159.04 | 59f2e2e | axon-test-59f2e2e | 10-20 17:32–17:32 | [log](1020-173232-2a0d.log) |
| 1020-173233-52b6 | moeresb-59f2e2e-fast-dbg-cpu [shard 1/2] | FAIL (rc 130) | avante | CPU only (avante; GPU hidden) | 610.43.02 | 59f2e2e | axon-test-dbg-59f2e2e | 10-20 17:33–17:36 | [log](1020-173233-52b6.log) |
| 1020-173233-c4ed | moeresb-59f2e2e-fast-dbg-cpu [shard 2/2] | FAIL (rc 130) | avante | CPU only (avante; GPU hidden) | 610.43.02 | 59f2e2e | axon-test-dbg-59f2e2e | ?–17:36 | [log](1020-173233-c4ed.log) |
| 1020-173233-0d09 | moeresb-59f2e2e-fast-rel [shard 1/2] | FAIL (rc 130) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 59f2e2e | axon-test-59f2e2e | ?–17:36 | [log](1020-173233-0d09.log) |
| 1020-173233-aace | moeresb-59f2e2e-fast-rel [shard 2/2] | FAIL (rc 130) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 59f2e2e | axon-test-59f2e2e | ?–17:36 | [log](1020-173233-aace.log) |

Redactions: none.

Hardware records: [None](hw-None.txt), [avante](hw-avante.txt), [jade](hw-jade.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
