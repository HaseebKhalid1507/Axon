# Gate set `budgetm-a18ff8d-bella-final`: FAIL (2 of 3); TAINTED timing

Published 2026-10-25 08:00:18 EDT by axon-gatelogs. Tested sha(s): `a18ff8d`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-045944-27ab | budgetm-a18ff8d-mix7b-r4 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | axon-test-a18ff8d | ?–05:33 | [log](1025-045944-27ab.log) |
| 1025-045944-27a6 | budgetm-a18ff8d-mix15-bella-r3 | PASS TAINTED (FOREIGN_GPU max=26% mean=6.5%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | axon-test-a18ff8d | 10-25 07:48–07:49 | [log](1025-045944-27a6.log) |
| 1025-045945-8657 | budgetm-a18ff8d-sdab15-bella-r3 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | sdab.sh | ?–08:00 | [log](1025-045945-8657.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
