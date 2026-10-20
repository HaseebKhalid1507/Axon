# Gate set `sched-step0b`: ALL PASS; TAINTED timing

Published 2026-10-20 13:58:35 EDT by axon-gatelogs. Tested sha(s): .

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1020-130046-ecfa | sched-step0b-1.5b | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | ? | axon-test-step0b | 10-20 13:02–13:03 | [log](1020-130046-ecfa.log) |
| 1020-130046-087c | sched-step0b-0.5b | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | ? | axon-test-step0b | 10-20 13:03–13:04 | [log](1020-130046-087c.log) |
| 1020-130047-8238 | sched-step0b-3b | PASS TAINTED (FOREIGN_GPU max=33% mean=11.0%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | ? | axon-test-step0b | 10-20 13:01–13:02 | [log](1020-130047-8238.log) |
| 1020-130047-1741 | sched-step0b-7b | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | ? | axon-test-step0b | 10-20 13:56–13:58 | [log](1020-130047-1741.log) |

Redactions: none.

Hardware records: [avante](hw-avante.txt), [bella](hw-bella.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
