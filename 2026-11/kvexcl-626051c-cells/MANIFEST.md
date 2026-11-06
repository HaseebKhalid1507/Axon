# Gate set `kvexcl-626051c-cells`: FAIL (2 of 3); TAINTED timing

Published 2026-11-05 19:11:00 EST by axon-gatelogs. Tested sha(s): `626051c`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1105-124635-28ee | kvexcl-626051c-replay-s4 | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 626051c | bash | 11-05 12:50–13:25 | [log](1105-124635-28ee.log) |
| 1105-124636-2d32 | kvexcl-626051c-replay-s1 | FAIL (rc 1) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 626051c | bash | 11-05 13:25–15:12 | [log](1105-124636-2d32.log) |
| 1105-124636-68e9 | kvexcl-626051c-resume | FAIL (rc 3) ON 4090 (bella reroute) | pod1 | NVIDIA GeForce RTX 4090 | 595.91.07 | 626051c | bash | 11-05 18:49–19:11 | [log](1105-124636-68e9.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [pod1-11-05_18:49](hw-pod1-11-05_18:49.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
