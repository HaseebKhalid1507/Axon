# Gate set `budgetm-722139a-gates`: FAIL (2 of 4)

Published 2026-10-25 19:04:35 EDT by axon-gatelogs. Tested sha(s): `722139a`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-174835-dd10 | budgetm-722139a-fast-rel | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 722139a | axon-test-722139a | 10-25 17:48–18:30 | [log](1025-174835-dd10.log) |
| 1025-174836-d8e3 | budgetm-722139a-serve-slots-dbg | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 722139a | axon-test-dbg-722139a | 10-25 18:37–18:43 | [log](1025-174836-d8e3.log) |
| 1025-174836-e6fe | budgetm-722139a-md5 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 722139a | md5.sh | ?–19:04 | [log](1025-174836-e6fe.log) |
| 1025-174836-adcf | budgetm-722139a-servab | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 722139a | servab-main.sh | ?–19:04 | [log](1025-174836-adcf.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
