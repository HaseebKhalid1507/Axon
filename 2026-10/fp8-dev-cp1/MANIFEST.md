# Gate set `fp8-dev-cp1`: FAIL (1 of 3)

Published 2026-10-29 18:06:13 EDT by axon-gatelogs. Tested sha(s): .

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1029-170302-cd38 | fp8-acc-cpu | PASS | bella | CPU only (bella; GPU hidden) | 615.78.08 | ? | acc_cpu.sh | 10-29 17:03–17:49 | [log](1029-170302-cd38.log) |
| 1029-171342-3b28 | fp8-wip1-tests | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | ? | axon-test-wip1 | 10-29 17:17–17:44 | [log](1029-171342-3b28.log) |
| 1029-171527-17ce | fp8-acc-gpu | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | ? | acc_gpu.sh | 10-29 17:52–18:06 | [log](1029-171527-17ce.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
