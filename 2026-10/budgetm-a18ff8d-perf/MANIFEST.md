# Gate set `budgetm-a18ff8d-perf`: FAIL (1 of 6); TAINTED timing

Published 2026-10-25 02:23:52 EDT by axon-gatelogs. Tested sha(s): `a18ff8d`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-012508-948a | budgetm-a18ff8d-mix05 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a18ff8d | axon-test-a18ff8d | 10-25 02:15–02:16 | [log](1025-012508-948a.log) |
| 1025-012609-5381 | budgetm-a18ff8d-mix15 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a18ff8d | axon-test-a18ff8d | 10-25 02:16–02:19 | [log](1025-012609-5381.log) |
| 1025-021120-1964 | budgetm-a18ff8d-sd05 | FAIL (rc 101) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | a18ff8d | axon-test-a18ff8d | 10-25 02:20–02:20 | [log](1025-021120-1964.log) |
| 1025-021225-5f61 | budgetm-a18ff8d-mix7b | PASS TAINTED (FOREIGN_GPU max=56% mean=13.5%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | axon-test-a18ff8d | 10-25 02:16–02:18 | [log](1025-021225-5f61.log) |
| 1025-021528-f252 | budgetm-a18ff8d-mix15-bella | PASS TAINTED (FOREIGN_GPU max=19% mean=8.6%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | axon-test-a18ff8d | 10-25 02:18–02:19 | [log](1025-021528-f252.log) |
| 1025-021628-f3cf | budgetm-a18ff8d-sdab15-bella | PASS TAINTED (FOREIGN_GPU max=79% mean=15.2%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | sdab.sh | 10-25 02:20–02:23 | [log](1025-021628-f3cf.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
