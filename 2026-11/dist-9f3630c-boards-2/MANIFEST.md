# Gate set `dist-9f3630c-boards`: FAIL (2 of 3); TAINTED timing

Published 2026-11-15 08:29:56 EST by axon-gatelogs. Tested sha(s): `9f3630c`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1115-032610-223e | dist-9f3630c-rel01-agent-sweep | FAIL (rc 143) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | 9f3630c | agent_sweep.sh | 11-15 06:06–08:06 | [log](1115-032610-223e.log) |
| 1115-032610-bbd2 | dist-9f3630c-rel01-agent-board | FAIL (rc 1) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | 9f3630c | agent_board.sh | 11-15 08:20–08:20 | [log](1115-032610-bbd2.log) |
| 1115-032613-4ff7 | dist-9f3630c-rel01-sb | PASS TAINTED (FOREIGN_GPU max=55% mean=13.3%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 620.70.06 | 9f3630c | single_stream.sh | 11-15 08:20–08:29 | [log](1115-032613-4ff7.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
