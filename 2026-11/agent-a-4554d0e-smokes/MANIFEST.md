# Gate set `agent-a-4554d0e-smokes`: FAIL (1 of 2)

Published 2026-11-04 09:07:22 EST by axon-gatelogs. Tested sha(s): `4554d0e`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1104-090227-c95c | agent-a-4554d0e-smoke | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 4554d0e | smoke-merge.sh | 11-04 09:02–09:03 | [log](1104-090227-c95c.log) |
| 1104-090227-04aa | agent-a-4554d0e-killsmoke | FAIL (rc 1) | bella | CPU only (bella; GPU hidden) | 615.78.08 | 4554d0e | agent_board_kill_smoke.sh | 11-04 09:03–09:07 | [log](1104-090227-04aa.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
