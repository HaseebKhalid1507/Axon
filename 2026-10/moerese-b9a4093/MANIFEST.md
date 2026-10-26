# Gate set `moerese-b9a4093`: FAIL (1 of 8); TAINTED timing

Published 2026-10-26 06:28:28 EDT by axon-gatelogs. Tested sha(s): `b9a4093`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1026-052416-3bc7 | moerese-b9a4093-res-bella | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | b9a4093 | axon-test-b9a4093 | 10-26 05:24–05:26 | [log](1026-052416-3bc7.log) |
| 1026-052416-1719 | moerese-b9a4093-res-dbg-bella | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | b9a4093 | axon-test-dbg-b9a4093 | 10-26 05:26–05:28 | [log](1026-052416-1719.log) |
| 1026-052417-7f02 | moerese-b9a4093-res-avante | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | b9a4093 | axon-test-b9a4093 | 10-26 05:24–05:25 | [log](1026-052417-7f02.log) |
| 1026-052417-243f | moerese-b9a4093-res-jade | FAIL (rc 101) | jade | NVIDIA GeForce GTX 1050 | 580.159.04 | b9a4093 | axon-test-b9a4093 | 10-26 05:24–05:33 | [log](1026-052417-243f.log) |
| 1026-052418-0a96 | moerese-b9a4093-fast-rel | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | b9a4093 | axon-test-b9a4093 | 10-26 05:25–06:07 | [log](1026-052418-0a96.log) |
| 1026-052418-7fd6 | moerese-b9a4093-fast-dbg-cpu | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | b9a4093 | axon-test-dbg-b9a4093 | 10-26 05:28–05:40 | [log](1026-052418-7fd6.log) |
| 1026-052419-c3f9 | moerese-b9a4093-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | b9a4093 | md5all3-e.sh | 10-26 05:43–06:21 | [log](1026-052419-c3f9.log) |
| 1026-052419-63a6 | moerese-b9a4093-ab | PASS TAINTED (FOREIGN_GPU max=32% mean=6.2%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | b9a4093 | abq3.py | 10-26 06:24–06:28 | [log](1026-052419-63a6.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [avante](hw-avante.txt), [jade](hw-jade.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
