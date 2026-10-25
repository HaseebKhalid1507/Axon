# Gate set `budget-eb14e53-perf`: FAIL (1 of 6); TAINTED timing

Published 2026-10-25 00:51:43 EDT by axon-gatelogs. Tested sha(s): `eb14e53`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1024-210909-32e9 | budget-eb14e53-mix05-h1 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | eb14e53 | axon-test-eb14e53 | 10-24 21:25–21:26 | [log](1024-210909-32e9.log) |
| 1024-212611-a640 | budget-eb14e53-sd05-h1 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | eb14e53 | axon-test-eb14e53 | 10-24 21:27–21:30 | [log](1024-212611-a640.log) |
| 1024-212712-7985 | budget-eb14e53-mix05-h0 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | eb14e53 | axon-test-eb14e53 | 10-24 21:30–21:32 | [log](1024-212712-7985.log) |
| 1024-213113-af45 | budget-eb14e53-mix15-h1 | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | eb14e53 | axon-test-eb14e53 | 10-24 21:32–21:35 | [log](1024-213113-af45.log) |
| 1024-213314-784b | budget-eb14e53-mix7b-h1 | PASS TAINTED (FOREIGN_GPU max=20% mean=7.0%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | eb14e53 | axon-test-eb14e53 | 10-25 00:47–00:49 | [log](1024-213314-784b.log) |
| 1024-213515-a5d0 | budget-eb14e53-sdab15-bella | FAIL (rc 143) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | eb14e53 | sdab.sh | 10-25 00:50–00:51 | [log](1024-213515-a5d0.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
