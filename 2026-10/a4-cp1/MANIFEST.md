# Gate set `a4-cp1`: ALL PASS; TAINTED timing

Published 2026-10-30 14:27:30 EDT by axon-gatelogs. Tested sha(s): `7b32b3a`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1030-130921-4f26 | a4-gates-7b32b3a | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 7b32b3a | gates.sh | 10-30 13:09–13:17 | [log](1030-130921-4f26.log) |
| 1030-130922-c603 | a4-ab-7b32b3a | PASS TAINTED (FOREIGN_GPU max=39% mean=7.1%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 7b32b3a | ab.sh | 10-30 14:09–14:27 | [log](1030-130922-c603.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
