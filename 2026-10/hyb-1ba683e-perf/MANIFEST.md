# Gate set `hyb-1ba683e-perf`: ALL PASS; TAINTED timing

Published 2026-10-20 16:40:49 EDT by axon-gatelogs. Tested sha(s): `1ba683e`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1020-152516-f8a7 | hyb-1ba683e-ab-smoke | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 1ba683e | axon-test-1ba683e | 10-20 15:29–15:30 | [log](1020-152516-f8a7.log) |
| 1020-152517-9729 | hyb-1ba683e-ab-1.5b | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 1ba683e | axon-test-1ba683e | 10-20 15:25–15:28 | [log](1020-152517-9729.log) |
| 1020-152517-f575 | hyb-1ba683e-ab-7b | PASS TAINTED (FOREIGN_GPU max=48% mean=20.0%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | 1ba683e | axon-test-1ba683e | 10-20 16:35–16:40 | [log](1020-152517-f575.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
