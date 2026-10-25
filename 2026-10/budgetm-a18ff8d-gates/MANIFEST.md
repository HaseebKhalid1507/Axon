# Gate set `budgetm-a18ff8d-gates`: FAIL (1 of 6)

Published 2026-10-25 02:34:46 EDT by axon-gatelogs. Tested sha(s): `a18ff8d`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-011101-7b35 | budgetm-a18ff8d-fast-rel | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a18ff8d | axon-test-a18ff8d | 10-25 01:25–02:14 | [log](1025-011101-7b35.log) |
| 1025-011102-d0a2 | budgetm-a18ff8d-dbg-cpu | PASS | avante | CPU only (avante; GPU hidden) | 610.43.02 | a18ff8d | axon-test-dbg-a18ff8d | 10-25 01:25–01:50 | [log](1025-011102-d0a2.log) |
| 1025-011102-6e5e | budgetm-a18ff8d-serve-slots-dbg | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a18ff8d | axon-test-dbg-a18ff8d | 10-25 02:20–02:26 | [log](1025-011102-6e5e.log) |
| 1025-011103-2a37 | budgetm-a18ff8d-serve-rel-roll | FAIL (rc 101) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a18ff8d | axon-test-a18ff8d | 10-25 02:30–02:34 | [log](1025-011103-2a37.log) |
| 1025-011103-bbeb | budgetm-a18ff8d-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | md5.sh | 10-25 02:10–02:12 | [log](1025-011103-bbeb.log) |
| 1025-011104-fadf | budgetm-a18ff8d-servab | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | servab-main.sh | 10-25 02:12–02:13 | [log](1025-011104-fadf.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
