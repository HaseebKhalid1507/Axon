# Gate set `paged2-b35d3d5`: FAIL (1 of 7); TAINTED timing

Published 2026-10-19 14:58:53 EDT by axon-gatelogs. Tested sha(s): `b35d3d5`.

| job | name | verdict | host | GPU | driver | sha | binary | ran | log |
|---|---|---|---|---|---|---|---|---|---|
| 1019-105852-438e | paged2-b35d3d5-graphbytes7b-inject | PASS | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | b35d3d5 | axon-test-b35d3d5 | 10-19 14:58–14:58 | [log](1019-105852-438e.log) |
| 1019-105853-29a2 | paged2-b35d3d5-oom-avante | FAIL (rc 101) | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | b35d3d5 | axon-test-b35d3d5 | 10-19 12:08–12:08 | [log](1019-105853-29a2.log) |
| 1019-105853-3466 | paged2-b35d3d5-rel [shard 1/2] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | b35d3d5 | axon-test-b35d3d5 | 10-19 12:08–12:30 | [log](1019-105853-3466.log) |
| 1019-105853-a3f1 | paged2-b35d3d5-rel [shard 2/2] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | b35d3d5 | axon-test-b35d3d5 | 10-19 12:53–13:32 | [log](1019-105853-a3f1.log) |
| 1019-105853-5f93 | paged2-b35d3d5-dbg [shard 1/2] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | b35d3d5 | axon-test-dbg-b35d3d5 | 10-19 13:32–14:02 | [log](1019-105853-5f93.log) |
| 1019-105853-aa9f | paged2-b35d3d5-dbg [shard 2/2] | PASS | avante | NVIDIA GeForce RTX 4050 Laptop GPU | 610.43.02 | b35d3d5 | axon-test-dbg-b35d3d5 | 10-19 14:02–14:46 | [log](1019-105853-aa9f.log) |
| 1019-105853-7408 | paged2-b35d3d5-r12-excl | PASS TAINTED (FOREIGN_GPU max=49% mean=3.0%) | bella | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 615.71.09 | b35d3d5 | r12-excl-b35d3d5.sh | 10-19 14:18–14:26 | [log](1019-105853-7408.log) |

Redactions: none.

Hardware records: [bella](hw-bella.txt), [avante](hw-avante.txt). Bella/avante/jade records are captured at upload time; pod records at pod setup (or from the pod's bootstrap log).
