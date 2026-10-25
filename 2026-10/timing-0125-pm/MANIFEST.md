# Gate set `timing-0125-pm`: ALL PASS; TAINTED timing

Published 2026-10-25 17:36:31 EDT by axon-gatelogs. Tested sha(s): `a18ff8d`, `93a53b3`, `9efdcf2`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1025-153021-7de9 | budgetm-a18ff8d-mix15-bella-r4 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | axon-test-a18ff8d | 10-25 16:04–16:05 | [log](1025-153021-7de9.log) |
| 1025-153024-9998 | budgetm-a18ff8d-sdab15-bella-r4 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | sdab.sh | 10-25 16:31–16:35 | [log](1025-153024-9998.log) |
| 1025-153024-9a7f | budgetm-a18ff8d-mix7b-r6 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | a18ff8d | axon-test-a18ff8d | 10-25 16:35–16:37 | [log](1025-153024-9a7f.log) |
| 1025-153024-c92e | moeresc-93a53b3-killrule | PASS TAINTED (FOREIGN_GPU max=31% mean=7.8%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 93a53b3 | killrule2.sh | 10-25 17:02–17:03 | [log](1025-153024-c92e.log) |
| 1025-153025-5e47 | an-rehearsal-9efdcf2-bella-r3 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 9efdcf2 | rehearse.sh | 10-25 17:30–17:36 | [log](1025-153025-5e47.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
