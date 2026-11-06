# Gate set `kvexcl-9829783-r2-bella`: FAIL (1 of 2); TAINTED timing

Published 2026-11-06 03:05:49 EST by axon-gatelogs. Tested sha(s): `7a2c9cf`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1105-210655-4117 | kvexcl-9829783-replay-s4 | FAIL (rc 1) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 7a2c9cf | bash | 11-06 00:26–01:54 | [log](1105-210655-4117.log) |
| 1105-211421-8e35 | kvexcl-9829783-resume-bella | PASS TAINTED (FOREIGN_GPU max=98% mean=7.7%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | ? | bash | 11-06 02:45–03:05 | [log](1105-211421-8e35.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
