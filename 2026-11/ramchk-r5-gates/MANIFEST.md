# Gate set `ramchk-r5-gates`: FAIL (1 of 7)

Published 2026-11-06 12:21:35 EST by axon-gatelogs. Tested sha(s): `c5c015e`, `dfbf2a4`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1106-113105-5c70 | ramchk-c5c015e-rel-r5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | c5c015e | axon-test-c5c015e | 11-06 11:53–12:04 | [log](1106-113105-5c70.log) |
| 1106-113105-184a | ramchk-c5c015e-dbg-r5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | c5c015e | axon-test-dbg-c5c015e | 11-06 12:04–12:14 | [log](1106-113105-184a.log) |
| 1106-113106-4ebe | ramchk-c5c015e-cpu-rel-r5 | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | c5c015e | axon-test-c5c015e | 11-06 12:14–12:17 | [log](1106-113106-4ebe.log) |
| 1106-113106-47e4 | ramchk-c5c015e-cpu-dbg-r5 | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | c5c015e | axon-test-dbg-c5c015e | 11-06 12:17–12:21 | [log](1106-113106-47e4.log) |
| 1106-113106-b2a2 | ramchk-c5c015e-pure-r5 | PASS | jade | CPU only (jade; GPU hidden) | 580.159.04 | c5c015e | axon-test-c5c015e | 11-06 11:31–11:54 | [log](1106-113106-b2a2.log) |
| 1106-113107-1a3e | ramchk-expectfail-r5 | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | ? | axon-test-expectfail-r5 | 11-06 12:21–12:21 | [log](1106-113107-1a3e.log) |
| 1106-105219-0faf | ramchk-dfbf2a4-fp8cap-r4 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | dfbf2a4 | bash | 11-06 11:27–11:28 | [log](1106-105219-0faf.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [jade](hw-jade.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
