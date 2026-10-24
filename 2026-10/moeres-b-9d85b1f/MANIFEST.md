# Gate set `moeres-b-9d85b1f`: FAIL (2 of 7)

Published 2026-10-24 17:04:11 EDT by axon-gatelogs. Tested sha(s): `9d85b1f`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1024-164652-4105 | moeresb-9d85b1f-ex-bella | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 9d85b1f | axon-test-9d85b1f | 10-24 17:00–17:01 | [log](1024-164652-4105.log) |
| 1024-164653-b997 | moeresb-9d85b1f-ex-dbg-bella | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 9d85b1f | axon-test-dbg-9d85b1f | 10-24 17:01–17:01 | [log](1024-164653-b997.log) |
| 1024-164653-5c02 | moeresb-9d85b1f-ex-jade | PASS | jade | NVIDIA GeForce GTX 1050 | 580.159.04 | 9d85b1f | axon-test-9d85b1f | 10-24 16:46–16:47 | [log](1024-164653-5c02.log) |
| 1024-164654-25e2 | moeresb-9d85b1f-fast-rel [shard 1/2] | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 9d85b1f | axon-test-9d85b1f | ?–16:55 | [log](1024-164654-25e2.log) |
| 1024-164654-b51a | moeresb-9d85b1f-fast-rel [shard 2/2] | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 9d85b1f | axon-test-9d85b1f | ?–16:55 | [log](1024-164654-b51a.log) |
| 1024-164654-6c2b | moeresb-9d85b1f-fast-dbg-cpu [shard 1/2] | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | 9d85b1f | axon-test-dbg-9d85b1f | 10-24 16:47–17:00 | [log](1024-164654-6c2b.log) |
| 1024-164654-8e2c | moeresb-9d85b1f-fast-dbg-cpu [shard 2/2] | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | 9d85b1f | axon-test-dbg-9d85b1f | 10-24 17:01–17:04 | [log](1024-164654-8e2c.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [jade](hw-jade.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
