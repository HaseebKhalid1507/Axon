# Gate set `kvexcl-28da662-cp1bw`: FAIL (1 of 2); TAINTED timing

Published 2026-11-05 11:23:53 EST by axon-gatelogs. Tested sha(s): `28da662`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1104-164604-7732 | kvexcl-28da662-bw25600 | PASS TAINTED (FOREIGN_GPU max=99% mean=53.5%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 28da662 | t-28da662-rel | 11-05 11:11–11:11 | [log](1104-164604-7732.log) |
| 1104-164605-53d3 | kvexcl-28da662-bw32768 | FAIL (rc 130) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 28da662 | t-28da662-rel | ?–11:23 | [log](1104-164605-53d3.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
