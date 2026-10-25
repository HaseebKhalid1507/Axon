# Gate set `moeres-c-0685079`: FAIL (1 of 4); TAINTED timing

Published 2026-10-25 00:47:09 EDT by axon-gatelogs. Tested sha(s): `0685079`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1024-191038-552a | moeresc-0685079-res-jade | PASS | jade | NVIDIA GeForce GTX 1050 | 580.159.04 | 0685079 | axon-test-0685079 | 10-24 19:10–19:14 | [log](1024-191038-552a.log) |
| 1024-191038-d75b | moeresc-0685079-res-avante | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | 0685079 | axon-test-0685079 | 10-24 21:52–21:52 | [log](1024-191038-d75b.log) |
| 1024-191039-2c90 | moeresc-0685079-res-bella | FAIL (rc 101) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 0685079 | axon-test-0685079 | 10-25 00:19–00:19 | [log](1024-191039-2c90.log) |
| 1024-191040-267a | moeresc-0685079-killrule | PASS TAINTED (FOREIGN_GPU max=16% mean=9.2%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.78.08 | 0685079 | killrule.sh | 10-25 00:46–00:47 | [log](1024-191040-267a.log) |

Redactions: none.

Hardware records: [jade](hw-jade.txt), [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
