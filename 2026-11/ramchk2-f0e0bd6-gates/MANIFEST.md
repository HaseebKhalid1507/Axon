# Gate set `ramchk2-f0e0bd6-gates`: FAIL (5 of 8)

Published 2026-11-06 15:46:20 EST by axon-gatelogs. Tested sha(s): `f0e0bd6`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1106-141448-ed6a | ramchk2-f0e0bd6-rel | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | f0e0bd6 | axon-test-f0e0bd6 | 11-06 14:29–15:36 | [log](1106-141448-ed6a.log) |
| 1106-141449-23b6 | ramchk2-f0e0bd6-dbg | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | f0e0bd6 | axon-test-dbg-f0e0bd6 | 11-06 15:36–15:46 | [log](1106-141449-23b6.log) |
| 1106-141445-7415 | ramchk2-f0e0bd6-cpu-rel | FAIL (rc 143) | bella | CPU only (bella; GPU hidden) | 615.78.08 | f0e0bd6 | axon-test-f0e0bd6 | 11-06 14:15–14:23 | [log](1106-141445-7415.log) |
| 1106-141446-ae2a | ramchk2-f0e0bd6-cpu-dbg | FAIL (rc 130) | bella | CPU only (bella; GPU hidden) | 615.78.08 | f0e0bd6 | axon-test-dbg-f0e0bd6 | ?–14:23 | [log](1106-141446-ae2a.log) |
| 1106-141445-8df1 | ramchk2-f0e0bd6-pure | PASS | jade | CPU only (jade; GPU hidden) | 580.159.04 | f0e0bd6 | axon-test-f0e0bd6 | 11-06 14:14–14:49 | [log](1106-141445-8df1.log) |
| 1106-141446-e7d6 | ramchk2-f0e0bd6-fp8cap-floor | FAIL (rc 1) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | f0e0bd6 | bash | 11-06 14:23–14:24 | [log](1106-141446-e7d6.log) |
| 1106-141447-621a | ramchk2-f0e0bd6-fp8cap-steps-3b | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | f0e0bd6 | bash | 11-06 14:24–14:24 | [log](1106-141447-621a.log) |
| 1106-141448-f898 | ramchk2-f0e0bd6-fp8cap-steps-7b | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | f0e0bd6 | bash | 11-06 14:24–14:29 | [log](1106-141448-f898.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [jade](hw-jade.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
