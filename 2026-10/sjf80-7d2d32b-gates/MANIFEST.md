# Gate set `sjf80-7d2d32b-gates`: FAIL (2 of 6)

Published 2026-10-27 13:18:11 EDT by axon-gatelogs. Tested sha(s): `7d2d32b`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1027-090606-5804 | sjf80-7d2d32b-fast-rel | FAIL (rc 101) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 7d2d32b | axon-test-7d2d32b | 10-27 09:13–09:56 | [log](1027-090606-5804.log) |
| 1027-090606-b980 | sjf80-7d2d32b-serve-slots-dbg | FAIL (rc 101) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 7d2d32b | axon-test-dbg-7d2d32b | 10-27 09:56–10:02 | [log](1027-090606-b980.log) |
| 1027-090607-bbfe | sjf80-7d2d32b-serve-rel-fifo | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 7d2d32b | axon-test-7d2d32b | 10-27 10:03–10:07 | [log](1027-090607-bbfe.log) |
| 1027-090607-ca07 | sjf80-7d2d32b-md5 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 7d2d32b | md5.sh | 10-27 11:11–11:13 | [log](1027-090607-ca07.log) |
| 1027-090608-2def | sjf80-7d2d32b-servab | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 7d2d32b | servab-sjf80.sh | 10-27 11:13–11:14 | [log](1027-090608-2def.log) |
| 1027-090608-b6c6 | sjf80-7d2d32b-norgr | PASS ON 4090 (bella fallback) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 7d2d32b | axon-test-7d2d32b | 10-27 13:12–13:18 | [log](1027-090608-b6c6.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt), [pod1-10-27_13:12](hw-pod1-10-27_13:12.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
